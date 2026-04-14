"""
Scenario, storage, and orchestration helpers for SCEPTer workflows.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>

This module remains part of the GPLv3-licensed SCEPTer project. The author
attribution above does not change the project-wide license.
"""

from __future__ import annotations

import atexit
import ctypes
import gc
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Mapping, Tuple

import h5py
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from pycraf import conversions as cnv

try:
    import psutil
except Exception:  # pragma: no cover - import availability is environment-dependent
    psutil = None

_HOST_MEMORY_HEADROOM_PROFILES = {
    "conservative": {
        "available_fraction": 0.55,
        "reserve_bytes": 4 * 1024 ** 3,
        "visible_satellite_factor": 1.50,
    },
    "balanced": {
        "available_fraction": 0.70,
        "reserve_bytes": 2 * 1024 ** 3,
        "visible_satellite_factor": 1.25,
    },
    "aggressive": {
        "available_fraction": 0.85,
        "reserve_bytes": 1 * 1024 ** 3,
        "visible_satellite_factor": 1.10,
    },
}

_SCHEDULER_TARGET_FRACTIONS = {
    "conservative": 0.75,
    "high_throughput": 0.90,
    "max_throughput": 0.97,
}
_SCHEDULER_BACKOFF_LADDER = (0.97, 0.90, 0.75, 0.60)
_DIRECT_EPFD_VISIBILITY_PROBE_SAMPLES = 5
_DIRECT_EPFD_CANDIDATE_BYTES_PER_PAIR_EST = 32
_DIRECT_EPFD_GPU_OOM_MARGIN_BYTES = 256 * 1024**2
_DIRECT_EPFD_GPU_BUDGET_RECOVERY_STEP_BYTES = 512 * 1024**2
_NVIDIA_SMI_PATH = shutil.which("nvidia-smi")
_DIRECT_EPFD_POWER_INPUT_QUANTITIES = frozenset(
    {"target_pfd", "satellite_ptx", "satellite_eirp"}
)
_DIRECT_EPFD_POWER_INPUT_BASES = frozenset({"per_mhz", "per_channel"})
_DIRECT_EPFD_CELL_ACTIVITY_MODES = frozenset({"whole_cell", "per_channel"})
_DIRECT_EPFD_SUPPORTED_REUSE_FACTORS: tuple[int, ...] = (1, 3, 4, 7, 9, 12, 13, 16, 19)
_DIRECT_EPFD_SPECTRUM_POWER_POLICIES = frozenset({"repeat_per_group", "split_total_cell_power"})
_DIRECT_EPFD_SPLIT_GROUP_DENOMINATOR_MODES = frozenset(
    {"configured_groups", "active_groups"}
)
_DIRECT_EPFD_REFERENCE_MODES = frozenset({"lower", "middle", "upper", "n_point"})
_DIRECT_EPFD_SPECTRAL_CUTOFF_BASES = frozenset({"channel_bandwidth", "service_bandwidth"})
_DIRECT_EPFD_MASK_PRESETS = frozenset(
    {
        "sm1541_fss",
        "sm1541_mss",
        "sm1541_sm329_fss",
        "sm1541_sm329_mss",
        "3gpp_ts_36_104",
        "wrc27_1_13_s1_dc_mss_imt",
        "adjacent_45_nonadjacent_50",
        "custom",
        "flat",
    }
)
_DIRECT_EPFD_RESULT_SCHEMA_VERSION = 4
_DIRECT_EPFD_CANONICAL_RAW_DATASET_NAMES: dict[str, str] = {
    "EPFD_W_m2_MHz": "EPFD_W_m2",
    "Prx_total_WperMHz": "Prx_total_W",
    "Prx_per_sat_RAS_STATION_WperMHz": "Prx_per_sat_RAS_STATION_W",
    "PFD_total_RAS_STATION_W_m2_MHz": "PFD_total_RAS_STATION_W_m2",
    "PFD_per_sat_RAS_STATION_W_m2_MHz": "PFD_per_sat_RAS_STATION_W_m2",
}
_DIRECT_EPFD_LEGACY_RAW_DATASET_NAMES: dict[str, str] = {
    canonical: legacy
    for legacy, canonical in _DIRECT_EPFD_CANONICAL_RAW_DATASET_NAMES.items()
}

# Global variable for the current thread count (used by simulation)
current_thread_count = 8

def set_num_threads(n):
    """
    Clamp n to [1, 32] and propagate that setting to:
      - OMP_NUM_THREADS
      - OPENBLAS_NUM_THREADS
      - MKL_NUM_THREADS
      - NUMEXPR_NUM_THREADS
      - threadpoolctl (if installed)
      - numba (if installed)
      - cysgp4 (if installed)

    Updates the module-level current_thread_count to the clamped value.
    """
    global current_thread_count

    # 1. Clamp to [1, 32]
    n_clamped = max(1, min(int(n), 32))

    # 2. Set environment variables for BLAS/OpenMP backends
    os.environ["OMP_NUM_THREADS"] = str(n_clamped)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_clamped)
    os.environ["MKL_NUM_THREADS"] = str(n_clamped)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_clamped)

    # 3. Set Numba's internal thread count, if Numba is installed
    try:
        import numba
        numba.set_num_threads(n_clamped)
    except ImportError:
        pass

    # 4. Set cysgp4 thread count, if cysgp4 is installed
    try:
        import cysgp4
        cysgp4.set_num_threads(n_clamped)
    except ImportError:
        pass

    # 5. Update the module‐level variable
    current_thread_count = n_clamped


def _timestep_to_seconds(timestep: float | u.Quantity) -> float:
    """
    Convert a timestep value to seconds as a positive float.

    Parameters
    ----------
    timestep : float or astropy.units.Quantity
        Simulation timestep. Plain numeric values are interpreted as seconds.

    Returns
    -------
    float
        Timestep in seconds.

    Raises
    ------
    ValueError
        If the timestep is not strictly positive.
    """
    if hasattr(timestep, "to_value"):
        timestep_sec = float(u.Quantity(timestep).to_value(u.s))
    else:
        timestep_sec = float(timestep)
    if timestep_sec <= 0.0:
        raise ValueError("timestep must be positive.")
    return timestep_sec


def _resolve_scheduler_target_fraction(profile: str | float | None) -> tuple[str, float]:
    if isinstance(profile, str):
        name = str(profile).strip().lower()
        if name not in _SCHEDULER_TARGET_FRACTIONS:
            raise ValueError(
                "scheduler_target_profile must be one of "
                f"{sorted(_SCHEDULER_TARGET_FRACTIONS)!r}; got {profile!r}."
            )
        return name, float(_SCHEDULER_TARGET_FRACTIONS[name])
    if profile is None:
        return "high_throughput", float(_SCHEDULER_TARGET_FRACTIONS["high_throughput"])
    fraction = float(profile)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("scheduler_target_fraction must lie in (0, 1].")
    return f"{fraction:.3f}", fraction


def _scheduler_backoff_fractions(active_fraction: float) -> list[float]:
    current = max(0.0, min(1.0, float(active_fraction)))
    values = [value for value in _SCHEDULER_BACKOFF_LADDER if value <= current + 1.0e-9]
    if not values or abs(values[0] - current) > 1.0e-9:
        values.insert(0, current)
    deduped: list[float] = []
    for value in values:
        if not deduped or abs(deduped[-1] - value) > 1.0e-9:
            deduped.append(float(value))
    return deduped


def _candidate_chunk_sizes(total: int) -> list[int]:
    total_i = int(max(1, total))
    values = {1, total_i}
    current = total_i
    while current > 1:
        current = max(1, (current + 1) // 2)
        values.add(int(current))
    quarter = max(1, total_i // 4)
    values.add(int(quarter))
    values.add(int(min(total_i, quarter * 3)))
    return sorted(values, reverse=True)


def iter_simulation_batches(
    start_time: Time,
    end_time: Time,
    timestep: float | u.Quantity,
    batch_size: int,
) -> Iterable[dict[str, Time | TimeDelta]]:
    """
    Yield simulation-time batches between two endpoints.

    Parameters
    ----------
    start_time : astropy.time.Time
        Global simulation start time.
    end_time : astropy.time.Time
        Global simulation end time. The generated sequence includes this end
        point when it lies exactly on the timestep grid, matching the existing
        eager batching behavior.
    timestep : float or astropy.units.Quantity
        Step size between samples. Plain numeric values are interpreted as
        seconds.
    batch_size : int
        Maximum number of timesteps to emit per yielded batch.

    Yields
    ------
    dict[str, astropy.time.Time | astropy.time.TimeDelta]
        One batch dictionary at a time with keys:
        ``"batch_start"``, ``"times"``, ``"td"``, and ``"batch_end"``.

    Raises
    ------
    ValueError
        If ``batch_size < 1``, if the timestep is not positive, or if
        ``end_time`` is earlier than ``start_time``.

    Notes
    -----
    This is the streaming sibling of :func:`generate_simulation_batches`.
    The yielded payload matches the per-batch elements that the eager helper
    stores in its output lists, so callers can switch without changing their
    downstream batch logic.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    timestep_sec = _timestep_to_seconds(timestep)
    total_duration_sec = float((end_time - start_time).to_value(u.s))
    if total_duration_sec < 0.0:
        raise ValueError("end_time must not be earlier than start_time.")

    total_steps = int(np.ceil(total_duration_sec / timestep_sec)) + 1

    for batch_start_idx in range(0, total_steps, int(batch_size)):
        batch_end_idx = min(batch_start_idx + int(batch_size), total_steps)
        n_steps_in_batch = batch_end_idx - batch_start_idx

        batch_td_array = np.arange(n_steps_in_batch, dtype=np.float64) * timestep_sec
        batch_td = TimeDelta(batch_td_array, format="sec")
        batch_start_time = start_time + TimeDelta(batch_start_idx * timestep_sec, format="sec")
        batch_times = batch_start_time + batch_td

        yield {
            "batch_start": batch_start_time,
            "times": batch_times,
            "td": batch_td,
            "batch_end": batch_times[-1],
        }


def recommend_observer_chunk_size(
    t_local: int,
    n_sats: int,
    *,
    ram_budget_gb: float,
    n_float_fields_per_pair: int = 7,
    dtype_itemsize: int = 8,
    safety_margin: float = 0.6,
) -> int:
    """
    Estimate how many observers can be propagated in one chunk.

    Parameters
    ----------
    t_local : int
        Number of timesteps in the current propagation batch.
    n_sats : int
        Number of satellites propagated against each observer.
    ram_budget_gb : float
        Approximate RAM budget reserved for dense propagation arrays.
    n_float_fields_per_pair : int, optional
        Estimated number of floating-point values kept live per
        ``(time, observer, satellite)`` tuple.
    dtype_itemsize : int, optional
        Byte width of the floating-point dtype used for the dense arrays.
    safety_margin : float, optional
        Fraction of the stated RAM budget that may be consumed by the exposed
        pairwise arrays. The remainder is reserved for cysgp4 and Python
        overheads.

    Returns
    -------
    int
        Recommended observer chunk size. The return value is always at least 1.

    Notes
    -----
    The estimate is intentionally conservative because :func:`cysgp4.propagate_many`
    also allocates internal temporary buffers that are not represented by the
    explicit output tensors.
    """
    if t_local < 1 or n_sats < 1:
        return 1
    if ram_budget_gb <= 0.0:
        return 1

    budget_bytes = int(ram_budget_gb * (1024 ** 3) * float(np.clip(safety_margin, 0.05, 1.0)))
    bytes_per_observer = int(t_local) * int(n_sats) * int(n_float_fields_per_pair) * int(dtype_itemsize)
    if bytes_per_observer <= 0:
        return 1

    return max(1, budget_bytes // bytes_per_observer)


def _normalise_memory_headroom_profile(profile: str) -> tuple[str, dict[str, float | int]]:
    name = str(profile).strip().lower()
    if name not in _HOST_MEMORY_HEADROOM_PROFILES:
        raise ValueError(
            "headroom_profile must be one of "
            f"{sorted(_HOST_MEMORY_HEADROOM_PROFILES)!r}; got {profile!r}."
        )
    return name, _HOST_MEMORY_HEADROOM_PROFILES[name]


_HOST_SNAPSHOT_CACHE: dict[str, Any] = {}
_HOST_SNAPSHOT_TIME: float = 0.0


def _runtime_host_memory_snapshot() -> dict[str, Any] | None:
    global _HOST_SNAPSHOT_CACHE, _HOST_SNAPSHOT_TIME
    now = perf_counter()
    if _HOST_SNAPSHOT_CACHE and (now - _HOST_SNAPSHOT_TIME) < 30.0:
        return _HOST_SNAPSHOT_CACHE
    if psutil is not None:
        vm = psutil.virtual_memory()
        snapshot = {
            "provider": "psutil",
            "available_bytes": int(vm.available),
            "total_bytes": int(vm.total),
        }
        _HOST_SNAPSHOT_CACHE = dict(snapshot)
        _HOST_SNAPSHOT_TIME = now
        return snapshot

    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {
                "provider": "win32",
                "available_bytes": int(status.ullAvailPhys),
                "total_bytes": int(status.ullTotalPhys),
            }
        return None

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None

    return {
        "provider": "sysconf",
        "available_bytes": int(page_size * avail_pages),
        "total_bytes": int(page_size * total_pages),
    }


_GPU_MEMORY_SNAPSHOT_CACHE: dict[str, Any] | None = None
_GPU_MEMORY_SNAPSHOT_TIME: float = 0.0


def _runtime_gpu_memory_snapshot(cp: Any, session: Any | None = None) -> dict[str, Any] | None:
    global _GPU_MEMORY_SNAPSHOT_CACHE, _GPU_MEMORY_SNAPSHOT_TIME
    now = perf_counter()
    if _GPU_MEMORY_SNAPSHOT_CACHE is not None and (now - _GPU_MEMORY_SNAPSHOT_TIME) < 5.0:
        return _GPU_MEMORY_SNAPSHOT_CACHE
    if session is not None and hasattr(session, "_get_memory_info"):
        try:
            info = session._get_memory_info()
            free_bytes = getattr(info, "free", None)
            total_bytes = getattr(info, "total", None)
            if free_bytes is not None and total_bytes is not None:
                snapshot = {
                    "provider": "session",
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                }
                _GPU_MEMORY_SNAPSHOT_CACHE = dict(snapshot)
                _GPU_MEMORY_SNAPSHOT_TIME = now
                return snapshot
        except Exception:
            pass
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception:
        return None
    snapshot = {
        "provider": "cuda",
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
    _GPU_MEMORY_SNAPSHOT_CACHE = dict(snapshot)
    _GPU_MEMORY_SNAPSHOT_TIME = now
    return snapshot


_GPU_ADAPTER_SNAPSHOT_CACHE: dict[str, Any] = {}
_GPU_ADAPTER_SNAPSHOT_TIME: float = 0.0
_GPU_ADAPTER_SNAPSHOT_INTERVAL: float = 60.0  # seconds


def _runtime_gpu_adapter_memory_snapshot(cp: Any) -> dict[str, Any] | None:
    global _GPU_ADAPTER_SNAPSHOT_CACHE, _GPU_ADAPTER_SNAPSHOT_TIME
    now = perf_counter()
    if _GPU_ADAPTER_SNAPSHOT_CACHE and (now - _GPU_ADAPTER_SNAPSHOT_TIME) < _GPU_ADAPTER_SNAPSHOT_INTERVAL:
        return _GPU_ADAPTER_SNAPSHOT_CACHE
    nvidia_smi_path = _NVIDIA_SMI_PATH
    if not nvidia_smi_path:
        return None
    device_index = 0
    if cp is not None:
        try:
            device_index = int(cp.cuda.Device().id)
        except Exception:
            device_index = 0
    try:
        result = subprocess.run(
            [
                nvidia_smi_path,
                f"--id={device_index}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    lines = str(result.stdout or "").splitlines()
    if not lines:
        return None
    try:
        used_text, total_text = [part.strip() for part in lines[0].split(",", 1)]
        used_mib = max(0.0, float(used_text))
        total_mib = max(0.0, float(total_text))
    except Exception:
        return None
    used_bytes = int(round(used_mib * (1024.0 ** 2)))
    total_bytes = int(round(total_mib * (1024.0 ** 2)))
    snapshot = {
        "provider": "nvidia-smi",
        "used_bytes": int(max(0, min(used_bytes, total_bytes))),
        "free_bytes": int(max(0, total_bytes - used_bytes)),
        "total_bytes": int(total_bytes),
    }
    _GPU_ADAPTER_SNAPSHOT_CACHE = dict(snapshot)
    _GPU_ADAPTER_SNAPSHOT_TIME = perf_counter()
    return snapshot


_PROCESS_RSS_CACHE: int | None = None
_PROCESS_RSS_TIME: float = 0.0


def _current_process_rss_bytes() -> int | None:
    global _PROCESS_RSS_CACHE, _PROCESS_RSS_TIME
    if psutil is None:
        return None
    now = perf_counter()
    if _PROCESS_RSS_CACHE is not None and (now - _PROCESS_RSS_TIME) < 30.0:
        return _PROCESS_RSS_CACHE
    try:
        rss = int(psutil.Process(os.getpid()).memory_info().rss)
        _PROCESS_RSS_CACHE = rss
        _PROCESS_RSS_TIME = now
        return rss
    except Exception:
        return None


_GPU_PROCESS_SNAPSHOT_CACHE: dict[str, Any] | None = None
_GPU_PROCESS_SNAPSHOT_TIME: float = 0.0


def _runtime_gpu_process_memory_snapshot(cp: Any) -> dict[str, Any] | None:
    global _GPU_PROCESS_SNAPSHOT_CACHE, _GPU_PROCESS_SNAPSHOT_TIME
    now = perf_counter()
    if _GPU_PROCESS_SNAPSHOT_CACHE is not None and (now - _GPU_PROCESS_SNAPSHOT_TIME) < 60.0:
        return _GPU_PROCESS_SNAPSHOT_CACHE
    if not _NVIDIA_SMI_PATH:
        return None
    device_index = 0
    if cp is not None:
        try:
            device_index = int(cp.cuda.Device().id)
        except Exception:
            device_index = 0
    try:
        result = subprocess.run(
            [
                _NVIDIA_SMI_PATH,
                f"--id={device_index}",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    target_pid = int(os.getpid())
    total_mib = 0.0
    found = False
    for line in str(result.stdout or "").splitlines():
        parts = [part.strip() for part in str(line).split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            pid_value = int(parts[0])
            used_mib = float(parts[1])
        except Exception:
            continue
        if pid_value != target_pid:
            continue
        total_mib += max(0.0, used_mib)
        found = True
    if not found:
        return None
    snapshot = {
        "provider": "nvidia-smi-compute-apps",
        "used_bytes": int(round(total_mib * (1024.0 ** 2))),
    }
    _GPU_PROCESS_SNAPSHOT_CACHE = dict(snapshot)
    _GPU_PROCESS_SNAPSHOT_TIME = now
    return snapshot


_LIVE_MEMORY_SNAPSHOT_CACHE: dict[str, Any] | None = None
_LIVE_MEMORY_SNAPSHOT_TIME: float = 0.0


def _capture_direct_epfd_live_memory_snapshot(
    cp: Any,
    session: Any | None = None,
) -> dict[str, Any]:
    global _LIVE_MEMORY_SNAPSHOT_CACHE, _LIVE_MEMORY_SNAPSHOT_TIME
    now = perf_counter()
    if _LIVE_MEMORY_SNAPSHOT_CACHE is not None and (now - _LIVE_MEMORY_SNAPSHOT_TIME) < 2.0:
        return _LIVE_MEMORY_SNAPSHOT_CACHE
    payload: dict[str, Any] = {}
    host_snapshot = _runtime_host_memory_snapshot()
    gpu_snapshot = _runtime_gpu_memory_snapshot(cp, session)
    gpu_adapter_snapshot = _runtime_gpu_adapter_memory_snapshot(cp)
    process_gpu_snapshot = _runtime_gpu_process_memory_snapshot(cp)
    process_rss_bytes = _current_process_rss_bytes()

    if host_snapshot is not None:
        payload["host_snapshot"] = dict(host_snapshot)
    if gpu_snapshot is not None:
        payload["gpu_snapshot"] = dict(gpu_snapshot)
    if gpu_adapter_snapshot is not None:
        payload["gpu_adapter_snapshot"] = dict(gpu_adapter_snapshot)
    if process_gpu_snapshot is not None:
        payload["process_gpu_snapshot"] = dict(process_gpu_snapshot)
    if process_rss_bytes is not None:
        payload["process_rss_bytes"] = int(process_rss_bytes)
    _LIVE_MEMORY_SNAPSHOT_CACHE = payload
    _LIVE_MEMORY_SNAPSHOT_TIME = now
    return payload


def _snapshot_gpu_used_bytes(snapshot: Mapping[str, Any] | None) -> int | None:
    if snapshot is None:
        return None
    process_gpu_snapshot = snapshot.get("process_gpu_snapshot")
    if isinstance(process_gpu_snapshot, Mapping):
        used_bytes = process_gpu_snapshot.get("used_bytes")
        if used_bytes is not None:
            try:
                return int(used_bytes)
            except Exception:
                pass
    gpu_adapter_snapshot = snapshot.get("gpu_adapter_snapshot")
    if isinstance(gpu_adapter_snapshot, Mapping):
        used_bytes = gpu_adapter_snapshot.get("used_bytes")
        if used_bytes is not None:
            try:
                return int(used_bytes)
            except Exception:
                pass
    return None


def _snapshot_gpu_free_bytes(snapshot: Mapping[str, Any] | None) -> int | None:
    if snapshot is None:
        return None
    candidates: list[int] = []
    gpu_snapshot = snapshot.get("gpu_snapshot")
    gpu_adapter_snapshot = snapshot.get("gpu_adapter_snapshot")
    for source in (gpu_snapshot, gpu_adapter_snapshot):
        if not isinstance(source, Mapping):
            continue
        free_bytes = source.get("free_bytes")
        if free_bytes is None:
            continue
        try:
            candidates.append(int(free_bytes))
        except Exception:
            continue
    if not candidates:
        return None
    return int(min(candidates))


def _refresh_direct_epfd_stage_observed_bytes(
    summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    refreshed = dict(summary or {})
    start_bytes = refreshed.get("observed_stage_gpu_start_bytes")
    end_bytes = refreshed.get("observed_stage_gpu_end_bytes")
    peak_bytes = refreshed.get("observed_stage_gpu_peak_bytes")
    try:
        if start_bytes is not None:
            start_i = int(start_bytes)
            refreshed["observed_stage_gpu_start_bytes"] = start_i
        else:
            start_i = None
        if end_bytes is not None:
            end_i = int(end_bytes)
            refreshed["observed_stage_gpu_end_bytes"] = end_i
        else:
            end_i = None
        if peak_bytes is not None:
            peak_i = int(peak_bytes)
            refreshed["observed_stage_gpu_peak_bytes"] = peak_i
        else:
            peak_i = None
    except Exception:
        return refreshed

    if start_i is not None and end_i is not None:
        refreshed["observed_stage_gpu_resident_bytes"] = int(max(0, end_i - start_i))
    elif "observed_stage_gpu_resident_bytes" not in refreshed:
        refreshed["observed_stage_gpu_resident_bytes"] = None
    if peak_i is not None:
        baseline_i = max(
            int(start_i) if start_i is not None else 0,
            int(end_i) if end_i is not None else 0,
        )
        refreshed["observed_stage_gpu_transient_peak_bytes"] = int(
            max(0, int(peak_i) - int(baseline_i))
        )
    elif "observed_stage_gpu_transient_peak_bytes" not in refreshed:
        refreshed["observed_stage_gpu_transient_peak_bytes"] = None
    return refreshed


def _update_direct_epfd_substage_memory_summary(
    container: Mapping[str, Any] | None,
    name: str,
    snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(container or {})
    substage_name = str(name)
    updated[substage_name] = _update_direct_epfd_stage_memory_summary(
        updated.get(substage_name),
        snapshot,
    )
    return updated


def _merge_direct_epfd_stage_memory_summaries(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(existing or {})
    incoming_map = _refresh_direct_epfd_stage_observed_bytes(incoming)
    if not incoming_map:
        return _refresh_direct_epfd_stage_observed_bytes(merged)
    if not merged:
        return dict(incoming_map)
    resident_peak_values = [
        int(source[key])
        for key in ("observed_stage_gpu_resident_bytes",)
        for source in (merged, incoming_map)
        if source.get(key) is not None
    ]
    transient_peak_values = [
        int(source[key])
        for key in ("observed_stage_gpu_transient_peak_bytes",)
        for source in (merged, incoming_map)
        if source.get(key) is not None
    ]
    merged.update(
        {
            key: value
            for key, value in dict(incoming_map).items()
            if key
            not in {
                "observed_stage_gpu_peak_bytes",
                "observed_stage_gpu_free_low_bytes",
                "observed_process_rss_bytes",
                "observed_stage_gpu_resident_bytes",
                "observed_stage_gpu_transient_peak_bytes",
                "planner_vs_observed_gpu_peak_error_bytes",
                "observed_stage_snapshot_count",
            }
            and value is not None
        }
    )
    for key in (
        "observed_stage_gpu_peak_bytes",
        "observed_process_rss_bytes",
        "observed_stage_gpu_resident_bytes",
        "observed_stage_gpu_transient_peak_bytes",
    ):
        values = [
            int(source[key])
            for source in (merged, incoming_map)
            if source.get(key) is not None
        ]
        if values:
            merged[key] = int(max(values))
    for key in ("observed_stage_gpu_free_low_bytes",):
        values = [
            int(source[key])
            for source in (merged, incoming_map)
            if source.get(key) is not None
        ]
        if values:
            merged[key] = int(min(values))
    if incoming_map.get("planner_vs_observed_gpu_peak_error_bytes") is not None:
        merged["planner_vs_observed_gpu_peak_error_bytes"] = int(
            incoming_map["planner_vs_observed_gpu_peak_error_bytes"]
        )
    if incoming_map.get("observed_stage_snapshot_count") is not None:
        merged["observed_stage_snapshot_count"] = int(
            max(
                int(merged.get("observed_stage_snapshot_count", 0) or 0),
                int(incoming_map["observed_stage_snapshot_count"]),
            )
        )
    if isinstance(merged.get("observed_stage_substages"), Mapping) and isinstance(
        incoming_map.get("observed_stage_substages"),
        Mapping,
    ):
        substage_map = dict(merged["observed_stage_substages"])
        for name, summary in dict(incoming_map["observed_stage_substages"]).items():
            substage_map[str(name)] = _merge_direct_epfd_stage_memory_summaries(
                substage_map.get(str(name)),
                summary,
            )
        merged["observed_stage_substages"] = substage_map
    elif isinstance(incoming_map.get("observed_stage_substages"), Mapping):
        merged["observed_stage_substages"] = dict(incoming_map["observed_stage_substages"])
    refreshed = _refresh_direct_epfd_stage_observed_bytes(merged)
    if resident_peak_values:
        refreshed["observed_stage_gpu_resident_bytes"] = int(max(resident_peak_values))
    if transient_peak_values:
        refreshed["observed_stage_gpu_transient_peak_bytes"] = int(max(transient_peak_values))
    return refreshed


def _start_direct_epfd_stage_memory_summary(
    stage: str,
    *,
    cp: Any,
    session: Any | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "observed_stage_name": str(stage),
        "observed_stage_gpu_start_bytes": None,
        "observed_stage_gpu_end_bytes": None,
        "observed_stage_gpu_peak_bytes": None,
        "observed_stage_gpu_resident_bytes": None,
        "observed_stage_gpu_transient_peak_bytes": None,
        "observed_stage_gpu_free_low_bytes": None,
        "observed_process_rss_bytes": None,
        "observed_stage_snapshot_count": 0,
        "observed_stage_substages": {},
    }
    return _update_direct_epfd_stage_memory_summary(
        summary,
        _capture_direct_epfd_live_memory_snapshot(cp, session),
    )


def _update_direct_epfd_stage_memory_summary(
    summary: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(summary or {})
    if snapshot is None:
        return merged
    used_bytes = _snapshot_gpu_used_bytes(snapshot)
    free_bytes = _snapshot_gpu_free_bytes(snapshot)
    process_rss_bytes = snapshot.get("process_rss_bytes")
    if used_bytes is not None:
        if merged.get("observed_stage_gpu_start_bytes") is None:
            merged["observed_stage_gpu_start_bytes"] = int(used_bytes)
        previous = merged.get("observed_stage_gpu_peak_bytes")
        merged["observed_stage_gpu_peak_bytes"] = int(
            used_bytes
            if previous is None
            else max(int(previous), int(used_bytes))
        )
        merged["observed_stage_gpu_end_bytes"] = int(used_bytes)
    if free_bytes is not None:
        previous = merged.get("observed_stage_gpu_free_low_bytes")
        merged["observed_stage_gpu_free_low_bytes"] = int(
            free_bytes
            if previous is None
            else min(int(previous), int(free_bytes))
        )
    if process_rss_bytes is not None:
        try:
            previous = merged.get("observed_process_rss_bytes")
            merged["observed_process_rss_bytes"] = int(
                process_rss_bytes
                if previous is None
                else max(int(previous), int(process_rss_bytes))
            )
        except Exception:
            pass
    if used_bytes is not None or free_bytes is not None or process_rss_bytes is not None:
        merged["observed_stage_snapshot_count"] = int(
            int(merged.get("observed_stage_snapshot_count", 0) or 0) + 1
        )
    for key in ("host_snapshot", "gpu_snapshot", "gpu_adapter_snapshot", "process_gpu_snapshot"):
        value = snapshot.get(key)
        if isinstance(value, Mapping):
            merged[key] = dict(value)
    return _refresh_direct_epfd_stage_observed_bytes(merged)


def _scheduler_runtime_state_extra(runtime_state: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    state = dict(runtime_state or {})
    if "gpu_effective_budget_lowered" in state:
        payload["gpu_effective_budget_lowered"] = bool(state["gpu_effective_budget_lowered"])
    if state.get("gpu_effective_budget_previous_bytes") is not None:
        payload["gpu_effective_budget_previous_bytes"] = int(state["gpu_effective_budget_previous_bytes"])
    if state.get("gpu_budget_lowered_stage") is not None:
        payload["gpu_budget_lowered_stage"] = str(state["gpu_budget_lowered_stage"])
    if "scheduler_retry_count" in state:
        payload["scheduler_retry_count"] = int(state["scheduler_retry_count"])
    if state.get("spectral_slab") is not None:
        payload["spectral_slab"] = int(state["spectral_slab"])
    if state.get("reuse_factor") is not None:
        payload["reuse_factor"] = int(state["reuse_factor"])
    if state.get("groups_per_cell") is not None:
        payload["groups_per_cell"] = int(state["groups_per_cell"])
    if state.get("spectral_backoff_active") is not None:
        payload["spectral_backoff_active"] = bool(state["spectral_backoff_active"])
    if state.get("limiting_dimension") is not None:
        payload["limiting_dimension"] = str(state["limiting_dimension"])
    if state.get("predicted_gpu_activity_resident_bytes") is not None:
        payload["predicted_gpu_activity_resident_bytes"] = int(
            state["predicted_gpu_activity_resident_bytes"]
        )
    if state.get("predicted_gpu_activity_scratch_bytes") is not None:
        payload["predicted_gpu_activity_scratch_bytes"] = int(
            state["predicted_gpu_activity_scratch_bytes"]
        )
    if state.get("predicted_gpu_activity_peak_bytes") is not None:
        payload["predicted_gpu_activity_peak_bytes"] = int(
            state["predicted_gpu_activity_peak_bytes"]
        )
    if state.get("predicted_gpu_link_library_resident_bytes") is not None:
        payload["predicted_gpu_link_library_resident_bytes"] = int(
            state["predicted_gpu_link_library_resident_bytes"]
        )
    if state.get("predicted_gpu_link_library_transient_peak_bytes") is not None:
        payload["predicted_gpu_link_library_transient_peak_bytes"] = int(
            state["predicted_gpu_link_library_transient_peak_bytes"]
        )
    if state.get("predicted_gpu_finalize_transient_peak_bytes") is not None:
        payload["predicted_gpu_finalize_transient_peak_bytes"] = int(
            state["predicted_gpu_finalize_transient_peak_bytes"]
        )
    if state.get("predicted_gpu_beam_finalize_working_bytes") is not None:
        payload["predicted_gpu_beam_finalize_working_bytes"] = int(
            state["predicted_gpu_beam_finalize_working_bytes"]
        )
    if state.get("compute_budget_utilization_fraction") is not None:
        payload["compute_budget_utilization_fraction"] = float(
            state["compute_budget_utilization_fraction"]
        )
    if state.get("export_budget_utilization_fraction") is not None:
        payload["export_budget_utilization_fraction"] = float(
            state["export_budget_utilization_fraction"]
        )
    if state.get("batch_count_estimate") is not None:
        payload["batch_count_estimate"] = int(state["batch_count_estimate"])
    if state.get("chunk_count_estimate") is not None:
        payload["chunk_count_estimate"] = int(state["chunk_count_estimate"])
    if state.get("underfill_reason") is not None:
        payload["underfill_reason"] = str(state["underfill_reason"])
    if state.get("predicted_gpu_spectrum_context_bytes") is not None:
        payload["predicted_gpu_spectrum_context_bytes"] = int(
            state["predicted_gpu_spectrum_context_bytes"]
        )
    stage_summary = state.get("last_observed_stage_summary")
    if isinstance(stage_summary, Mapping):
        if stage_summary.get("observed_stage_name") is not None:
            payload["observed_stage_name"] = str(stage_summary["observed_stage_name"])
        if stage_summary.get("observed_stage_gpu_peak_bytes") is not None:
            payload["observed_stage_gpu_peak_bytes"] = int(stage_summary["observed_stage_gpu_peak_bytes"])
        if stage_summary.get("observed_stage_gpu_start_bytes") is not None:
            payload["observed_stage_gpu_start_bytes"] = int(
                stage_summary["observed_stage_gpu_start_bytes"]
            )
        if stage_summary.get("observed_stage_gpu_end_bytes") is not None:
            payload["observed_stage_gpu_end_bytes"] = int(stage_summary["observed_stage_gpu_end_bytes"])
        if stage_summary.get("observed_stage_gpu_resident_bytes") is not None:
            payload["observed_stage_gpu_resident_bytes"] = int(
                stage_summary["observed_stage_gpu_resident_bytes"]
            )
        if stage_summary.get("observed_stage_gpu_transient_peak_bytes") is not None:
            payload["observed_stage_gpu_transient_peak_bytes"] = int(
                stage_summary["observed_stage_gpu_transient_peak_bytes"]
            )
        if stage_summary.get("observed_stage_gpu_free_low_bytes") is not None:
            payload["observed_stage_gpu_free_low_bytes"] = int(stage_summary["observed_stage_gpu_free_low_bytes"])
        if stage_summary.get("observed_process_rss_bytes") is not None:
            payload["observed_process_rss_bytes"] = int(stage_summary["observed_process_rss_bytes"])
        if stage_summary.get("planner_vs_observed_gpu_peak_error_bytes") is not None:
            payload["planner_vs_observed_gpu_peak_error_bytes"] = int(
                stage_summary["planner_vs_observed_gpu_peak_error_bytes"]
            )
    return payload


def _update_scheduler_runtime_state_from_plan(
    runtime_state: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    *,
    spectrum_context_bytes: int | None = None,
) -> dict[str, Any]:
    updated = dict(runtime_state or {})
    if plan is None:
        if spectrum_context_bytes is not None:
            updated["predicted_gpu_spectrum_context_bytes"] = int(max(0, spectrum_context_bytes))
        return updated

    if plan.get("spectral_slab") is not None:
        updated["spectral_slab"] = int(plan["spectral_slab"])
    if plan.get("limiting_dimension") is not None:
        updated["limiting_dimension"] = str(plan["limiting_dimension"])
    if plan.get("spectral_backoff_active") is not None:
        updated["spectral_backoff_active"] = bool(plan["spectral_backoff_active"])
    if plan.get("predicted_gpu_activity_resident_bytes") is not None:
        updated["predicted_gpu_activity_resident_bytes"] = int(
            plan["predicted_gpu_activity_resident_bytes"]
        )
    if plan.get("predicted_gpu_activity_scratch_bytes") is not None:
        updated["predicted_gpu_activity_scratch_bytes"] = int(
            plan["predicted_gpu_activity_scratch_bytes"]
        )
    if plan.get("predicted_gpu_activity_peak_bytes") is not None:
        updated["predicted_gpu_activity_peak_bytes"] = int(
            plan["predicted_gpu_activity_peak_bytes"]
        )
    if plan.get("predicted_gpu_link_library_resident_bytes") is not None:
        updated["predicted_gpu_link_library_resident_bytes"] = int(
            plan["predicted_gpu_link_library_resident_bytes"]
        )
    if plan.get("predicted_gpu_link_library_transient_peak_bytes") is not None:
        updated["predicted_gpu_link_library_transient_peak_bytes"] = int(
            plan["predicted_gpu_link_library_transient_peak_bytes"]
        )
    if plan.get("predicted_gpu_finalize_transient_peak_bytes") is not None:
        updated["predicted_gpu_finalize_transient_peak_bytes"] = int(
            plan["predicted_gpu_finalize_transient_peak_bytes"]
        )
    if plan.get("predicted_gpu_beam_finalize_working_bytes") is not None:
        updated["predicted_gpu_beam_finalize_working_bytes"] = int(
            plan["predicted_gpu_beam_finalize_working_bytes"]
        )
    if plan.get("compute_budget_utilization_fraction") is not None:
        updated["compute_budget_utilization_fraction"] = float(
            plan["compute_budget_utilization_fraction"]
        )
    if plan.get("export_budget_utilization_fraction") is not None:
        updated["export_budget_utilization_fraction"] = float(
            plan["export_budget_utilization_fraction"]
        )
    if plan.get("batch_count_estimate") is not None:
        updated["batch_count_estimate"] = int(plan["batch_count_estimate"])
    if plan.get("chunk_count_estimate") is not None:
        updated["chunk_count_estimate"] = int(plan["chunk_count_estimate"])
    if plan.get("underfill_reason") is not None:
        updated["underfill_reason"] = str(plan["underfill_reason"])
    if spectrum_context_bytes is not None:
        updated["predicted_gpu_spectrum_context_bytes"] = int(max(0, spectrum_context_bytes))
    return updated


def _apply_effective_budget_override(
    budget_info: Mapping[str, Any],
    *,
    effective_budget_bytes: int | None,
) -> dict[str, Any]:
    adjusted = dict(budget_info)
    if effective_budget_bytes is None:
        return adjusted
    hard_budget = int(
        adjusted.get("hard_budget_bytes", adjusted.get("effective_budget_bytes", effective_budget_bytes))
    )
    effective_i = max(1, min(int(effective_budget_bytes), hard_budget))
    adjusted["effective_budget_bytes"] = int(effective_i)
    if adjusted.get("planning_budget_bytes") is not None:
        adjusted["planning_budget_bytes"] = int(
            min(int(adjusted["planning_budget_bytes"]), effective_i)
        )
    else:
        adjusted["planning_budget_bytes"] = int(effective_i)
    if adjusted.get("runtime_advisory_budget_bytes") is not None:
        adjusted["runtime_advisory_budget_bytes"] = int(
            min(int(adjusted["runtime_advisory_budget_bytes"]), effective_i)
        )
    return adjusted


def _lower_runtime_effective_gpu_budget(
    runtime_state: Mapping[str, Any] | None,
    *,
    stage: str,
    post_cleanup_snapshot: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    state = dict(runtime_state or {})
    current_effective = state.get("gpu_effective_budget_bytes")
    if current_effective is None:
        return state, False
    observed_peak_bytes: int | None = None
    stage_summary = state.get("last_observed_stage_summary")
    if isinstance(stage_summary, Mapping) and stage_summary.get("observed_stage_gpu_peak_bytes") is not None:
        try:
            observed_peak_bytes = int(stage_summary["observed_stage_gpu_peak_bytes"])
        except Exception:
            observed_peak_bytes = None
    current_effective_i = int(max(1, current_effective))
    if observed_peak_bytes is None:
        return state, False
    # Lower only on a confirmed stage OOM, and only when the observed peak
    # materially exceeded the active planning cap. A noisy low-free snapshot on
    # WDDM should not collapse the entire run into near-sequential shapes.
    stage_oom_overrun_bytes = int(observed_peak_bytes) - int(current_effective_i)
    if stage_oom_overrun_bytes <= 0:
        return state, False
    lowered_effective = int(
        max(
            1,
            int(current_effective_i)
            - max(
                int(_DIRECT_EPFD_GPU_OOM_MARGIN_BYTES),
                int(stage_oom_overrun_bytes) + int(_DIRECT_EPFD_GPU_OOM_MARGIN_BYTES),
            ),
        )
    )
    if lowered_effective >= int(current_effective_i):
        return state, False
    state["gpu_effective_budget_previous_bytes"] = int(current_effective_i)
    state["gpu_effective_budget_bytes"] = int(lowered_effective)
    state["gpu_effective_budget_lowered"] = True
    state["gpu_budget_lowered_stage"] = str(stage)
    state["gpu_effective_budget_low_water_bytes"] = int(
        min(
            int(lowered_effective),
            int(state.get("gpu_effective_budget_low_water_bytes", lowered_effective) or lowered_effective),
        )
    )
    return state, True


def _recover_runtime_effective_gpu_budget(
    runtime_state: Mapping[str, Any] | None,
    *,
    hard_budget_bytes: int,
) -> tuple[dict[str, Any], bool]:
    state = dict(runtime_state or {})
    current_effective = state.get("gpu_effective_budget_bytes")
    if current_effective is None:
        return state, False
    hard_budget_i = int(max(1, hard_budget_bytes))
    current_effective_i = int(max(1, min(int(current_effective), hard_budget_i)))
    if current_effective_i >= hard_budget_i:
        state["gpu_effective_budget_lowered"] = False
        state["gpu_budget_lowered_stage"] = None
        return state, False
    recovered_effective = int(
        min(
            hard_budget_i,
            int(current_effective_i) + int(_DIRECT_EPFD_GPU_BUDGET_RECOVERY_STEP_BYTES),
        )
    )
    if recovered_effective <= current_effective_i:
        return state, False
    state["gpu_effective_budget_previous_bytes"] = int(current_effective_i)
    state["gpu_effective_budget_bytes"] = int(recovered_effective)
    state["gpu_effective_budget_lowered"] = bool(recovered_effective < hard_budget_i)
    if recovered_effective >= hard_budget_i:
        state["gpu_budget_lowered_stage"] = None
    return state, True


def _reset_direct_epfd_gpu_pools(cp: Any, session: Any | None) -> None:
    try:
        _sync_array_module(cp)
    except Exception:
        pass
    try:
        if session is not None and hasattr(session, "evict_idle_caches"):
            session.evict_idle_caches()
    except Exception:
        pass
    try:
        if hasattr(cp, "get_default_memory_pool"):
            cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        if hasattr(cp, "get_default_pinned_memory_pool"):
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def _is_direct_gpu_out_of_memory(exc: BaseException, *, cp: Any | None = None, gpu_module: Any | None = None) -> bool:
    if gpu_module is not None and hasattr(gpu_module, "_is_cupy_out_of_memory"):
        try:
            return bool(gpu_module._is_cupy_out_of_memory(exc))
        except Exception:
            pass
    if cp is not None:
        try:
            out_of_memory_error = getattr(getattr(cp, "cuda", None), "memory", None)
            out_of_memory_error = getattr(out_of_memory_error, "OutOfMemoryError", None)
            if out_of_memory_error is not None and isinstance(exc, out_of_memory_error):
                return True
        except Exception:
            pass
    text = str(exc).lower()
    return "out of memory" in text and (
        "cuda" in text
        or "cupy" in text
        or "device memory" in text
        or "vram" in text
    )


def _shape_nbytes(shape: tuple[int, ...], dtype: Any) -> int:
    dtype_obj = np.dtype(dtype)
    total = int(dtype_obj.itemsize)
    for dim in shape:
        total *= int(dim)
    return total


def _normalise_mjd_probe_value(mjds: Any) -> float:
    if isinstance(mjds, Time):
        values = np.asarray(mjds.mjd, dtype=np.float64).reshape(-1)
    else:
        values = np.asarray(mjds, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("mjds must contain at least one probe timestep.")
    return float(values[0])


def _deterministic_visibility_probe_mjds(
    start_time: Time,
    end_time: Time,
    *,
    sample_count: int = _DIRECT_EPFD_VISIBILITY_PROBE_SAMPLES,
) -> np.ndarray:
    total_duration_days = float((end_time - start_time).to_value(u.day))
    if total_duration_days <= 0.0 or int(sample_count) <= 1:
        return np.asarray([float(start_time.mjd)], dtype=np.float64)
    fractions = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float64)
    return np.asarray(start_time.mjd + total_duration_days * fractions, dtype=np.float64)


def _probe_visibility_profile_window(
    session: Any,
    mjds: np.ndarray,
    satellite_context: Any,
    *,
    observer_context: Any,
    observer_slice: slice,
    output_dtype: Any,
) -> dict[str, Any]:
    probe_summaries: list[dict[str, Any]] = []
    peak_profile: dict[str, Any] | None = None
    peak_visible_count = -1
    for probe_mjd in np.asarray(mjds, dtype=np.float64).reshape(-1):
        profile = session.probe_visibility_profile(
            np.asarray([probe_mjd], dtype=np.float64),
            satellite_context,
            observer_context=observer_context,
            observer_slice=observer_slice,
            output_dtype=output_dtype,
        )
        profile_dict = dict(profile or {})
        visible_count = int(profile_dict.get("visible_satellite_count", 0))
        probe_summaries.append(
            {
                "probe_mjd": float(profile_dict.get("probe_mjd", probe_mjd)),
                "visible_satellite_count": int(visible_count),
                "visible_fraction": float(profile_dict.get("visible_fraction", 0.0)),
            }
        )
        if visible_count > peak_visible_count:
            peak_visible_count = int(visible_count)
            peak_profile = profile_dict
    if peak_profile is None:
        raise RuntimeError("Visibility probing did not return any samples.")
    result = dict(peak_profile)
    result["probe_sample_count"] = int(len(probe_summaries))
    result["probe_samples"] = tuple(probe_summaries)
    result["visible_satellite_count"] = int(peak_visible_count)
    return result


def resolve_host_memory_budget_bytes(
    explicit_budget_gb: float | None,
    *,
    mode: str = "hybrid",
    headroom_profile: str = "balanced",
) -> dict[str, Any]:
    """
    Resolve the effective host-memory budget for notebook batch planning.

    Parameters
    ----------
    explicit_budget_gb : float or None
        User-visible RAM cap in gibibytes. The value is interpreted as a hard
        upper limit in ``"fixed"`` mode and as an additional cap in
        ``"hybrid"`` mode. Pass ``None`` only when `mode="runtime"` and no
        explicit cap should be applied.
    mode : {"fixed", "hybrid", "runtime"}, optional
        Budgeting policy.
    headroom_profile : {"conservative", "balanced", "aggressive"}, optional
        Runtime headroom policy used when runtime memory probing is available.

    Returns
    -------
    dict[str, Any]
        Effective host-memory budget with explicit and runtime diagnostics.

    Raises
    ------
    ValueError
        Raised when `mode` or `headroom_profile` is invalid, or when a fixed
        explicit budget is required but missing.

    Notes
    -----
    Runtime probing uses :mod:`psutil` when it is available and falls back to
    basic operating-system APIs otherwise.
    """
    mode_name = str(mode).strip().lower()
    if mode_name not in {"fixed", "hybrid", "runtime"}:
        raise ValueError("mode must be one of {'fixed', 'hybrid', 'runtime'}.")

    profile_name, profile = _normalise_memory_headroom_profile(headroom_profile)
    explicit_budget_bytes = None
    if explicit_budget_gb is not None:
        explicit_budget_bytes = int(float(explicit_budget_gb) * (1024 ** 3))
        if explicit_budget_bytes <= 0:
            raise ValueError("explicit_budget_gb must be positive when provided.")

    runtime_snapshot = _runtime_host_memory_snapshot()
    runtime_available_bytes = None
    runtime_total_bytes = None
    runtime_budget_bytes = None
    runtime_provider = None
    if runtime_snapshot is not None:
        runtime_available_bytes = int(runtime_snapshot["available_bytes"])
        runtime_total_bytes = int(runtime_snapshot["total_bytes"])
        runtime_provider = str(runtime_snapshot["provider"])
        reserve_bytes = min(int(profile["reserve_bytes"]), max(0, runtime_available_bytes - 1))
        usable_after_reserve = max(1, runtime_available_bytes - reserve_bytes)
        runtime_budget_bytes = max(
            1,
            min(
                int(runtime_available_bytes * float(profile["available_fraction"])),
                usable_after_reserve,
            ),
        )

    mode_used = mode_name
    if mode_name == "fixed":
        if explicit_budget_bytes is None:
            raise ValueError("explicit_budget_gb is required when mode='fixed'.")
        hard_budget_bytes = explicit_budget_bytes
        planning_budget_bytes = explicit_budget_bytes
    elif mode_name == "runtime":
        if runtime_budget_bytes is None:
            if explicit_budget_bytes is None:
                raise ValueError(
                    "Runtime host-memory probing is unavailable and no explicit "
                    "budget was supplied."
                )
            hard_budget_bytes = explicit_budget_bytes
            planning_budget_bytes = explicit_budget_bytes
            mode_used = "fixed_fallback"
        else:
            hard_budget_bytes = (
                explicit_budget_bytes
                if explicit_budget_bytes is not None
                else int(runtime_budget_bytes)
            )
            planning_budget_bytes = int(
                hard_budget_bytes if explicit_budget_bytes is not None else runtime_budget_bytes
            )
    else:
        if explicit_budget_bytes is None and runtime_budget_bytes is None:
            raise ValueError(
                "At least one of explicit_budget_gb or runtime memory probing "
                "must be available for mode='hybrid'."
            )
        if explicit_budget_bytes is None:
            hard_budget_bytes = int(runtime_budget_bytes)
            planning_budget_bytes = int(runtime_budget_bytes)
            mode_used = "runtime_only"
        elif runtime_budget_bytes is None:
            hard_budget_bytes = explicit_budget_bytes
            planning_budget_bytes = explicit_budget_bytes
            mode_used = "fixed_fallback"
        else:
            hard_budget_bytes = explicit_budget_bytes
            planning_budget_bytes = explicit_budget_bytes

    return {
        "mode_requested": mode_name,
        "mode_used": mode_used,
        "headroom_profile": profile_name,
        "explicit_budget_bytes": explicit_budget_bytes,
        "runtime_available_bytes": runtime_available_bytes,
        "runtime_total_bytes": runtime_total_bytes,
        "runtime_budget_bytes": runtime_budget_bytes,
        "hard_budget_bytes": int(hard_budget_bytes),
        "planning_budget_bytes": int(planning_budget_bytes),
        "runtime_advisory_budget_bytes": runtime_budget_bytes,
        "effective_budget_bytes": int(hard_budget_bytes),
        "headroom_fraction": float(profile["available_fraction"]),
        "headroom_reserve_bytes": int(profile["reserve_bytes"]),
        "runtime_provider": runtime_provider,
        "visible_satellite_factor": float(profile["visible_satellite_factor"]),
    }


def _normalise_runner_memory_budget_modes(mode: str) -> tuple[str, str]:
    """
    Map notebook/GUI runtime memory modes onto host/device budget policies.

    ``run_gpu_direct_epfd`` exposes ``"hybrid"``, ``"host_only"``, and
    ``"gpu_only"`` on the public control surface, while the underlying host and
    device budget resolvers accept ``{"fixed", "hybrid", "runtime"}``.
    """
    mode_name = str(mode).strip().lower()
    if mode_name in {"fixed", "hybrid", "runtime"}:
        return mode_name, mode_name
    if mode_name == "host_only":
        return "fixed", "runtime"
    if mode_name == "gpu_only":
        return "runtime", "fixed"
    raise ValueError(
        "memory_budget_mode must be one of "
        "{'fixed', 'hybrid', 'runtime', 'host_only', 'gpu_only'}."
    )


def recommend_time_batch_size_linear(
    *,
    total_timesteps: int,
    fixed_bytes: int,
    per_timestep_bytes: int,
    budget_bytes: int,
    min_batch_size: int = 1,
    max_batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Recommend a time-batch size for a linear working-set model.

    Parameters
    ----------
    total_timesteps : int
        Total timesteps available in the current iteration.
    fixed_bytes : int
        Time-independent memory term in bytes.
    per_timestep_bytes : int
        Linear memory growth per simulated timestep in bytes.
    budget_bytes : int
        Effective memory budget in bytes.
    min_batch_size : int, optional
        Smallest allowed recommendation.
    max_batch_size : int or None, optional
        Optional upper cap applied after solving the linear inequality.

    Returns
    -------
    dict[str, Any]
        Recommended batch size plus fit diagnostics and remaining budget.

    Raises
    ------
    ValueError
        Raised when the inputs are inconsistent or non-positive where a
        positive value is required.
    """
    total = int(total_timesteps)
    if total < 1:
        raise ValueError("total_timesteps must be >= 1.")
    if int(min_batch_size) < 1:
        raise ValueError("min_batch_size must be >= 1.")
    if int(budget_bytes) < 1:
        raise ValueError("budget_bytes must be positive.")

    min_batch = int(min_batch_size)
    max_batch = total if max_batch_size is None else max(min_batch, min(total, int(max_batch_size)))
    fixed_term = max(0, int(fixed_bytes))
    per_step_term = max(0, int(per_timestep_bytes))
    budget = int(budget_bytes)

    if per_step_term == 0:
        recommended = max_batch
    else:
        recommended = (budget - fixed_term) // per_step_term
        recommended = max(min_batch, min(max_batch, int(recommended)))

    slack = budget - (fixed_term + per_step_term * recommended)
    return {
        "recommended_batch_size": int(recommended),
        "fits_entire_span": bool(fixed_term + per_step_term * total <= budget),
        "fits_minimum_batch": bool(fixed_term + per_step_term * min_batch <= budget),
        "budget_slack_bytes": int(slack),
        "max_batch_size_considered": int(max_batch),
    }


def probe_visibility_profile_cpu(
    mjds: Any,
    tle_array: Any,
    observer_array: Any,
    *,
    observer_index: int = 0,
    sat_frame: str = "xyz",
    propagate_fn: Any | None = None,
) -> dict[str, Any]:
    """
    Run a one-step CPU propagation probe and measure horizon visibility.

    Parameters
    ----------
    mjds : Any
        One or more probe MJDs in days, or an :class:`astropy.time.Time`
        object. Only the first sample is used.
    tle_array : Any
        Satellite collection propagated during the probe.
    observer_array : Any
        Observer collection propagated during the probe.
    observer_index : int, optional
        Observer slot used for the returned visibility summary.
    sat_frame : str, optional
        Satellite reference-frame convention forwarded to the propagation
        helper.
    propagate_fn : Any or None, optional
        Propagation callable compatible with :func:`cysgp4.propagate_many`.

    Returns
    -------
    dict[str, Any]
        Probe MJD, visible-satellite count, visibility fraction, and
        per-observer counts for the selected probe step.
    """
    if propagate_fn is None:
        import cysgp4

        propagate_fn = cysgp4.propagate_many

    probe_mjd = _normalise_mjd_probe_value(mjds)
    tles = np.asarray(tle_array, dtype=object).reshape(1, 1, -1)
    observers = np.asarray(observer_array, dtype=object).reshape(1, -1, 1)
    probe_result = propagate_fn(
        np.asarray([probe_mjd], dtype=np.float64)[:, None, None],
        tles,
        observers,
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=False,
        do_sat_azel=False,
        do_sat_rotmat=False,
        sat_frame=sat_frame,
    )
    topo = probe_result.get("topo")
    if topo is None:
        raise RuntimeError("Visibility probe did not return a topocentric tensor.")

    topo_arr = np.asarray(topo)
    if topo_arr.ndim != 4 or topo_arr.shape[0] < 1 or topo_arr.shape[-1] < 2:
        raise RuntimeError(
            "Visibility probe expected topo with shape (T, O, S, K>=2); "
            f"got {tuple(topo_arr.shape)!r}."
        )
    if not 0 <= int(observer_index) < int(topo_arr.shape[1]):
        raise IndexError(
            f"observer_index={observer_index} is out of range for "
            f"{int(topo_arr.shape[1])} propagated observers."
        )

    elev_deg = topo_arr[0, :, :, 1].astype(np.float32, copy=False)
    vis = np.isfinite(elev_deg) & (elev_deg > np.float32(0.0))
    per_observer_count = np.count_nonzero(vis, axis=1).astype(np.int32, copy=False)
    visible_count = int(per_observer_count[int(observer_index)])
    sat_count = int(topo_arr.shape[2])
    return {
        "probe_mjd": float(probe_mjd),
        "observer_index": int(observer_index),
        "visible_satellite_count": visible_count,
        "visible_fraction": float(visible_count / sat_count) if sat_count > 0 else 0.0,
        "satellite_count": sat_count,
        "observer_count": int(topo_arr.shape[1]),
        "per_observer_visible_satellite_count": per_observer_count,
    }


def _estimate_step1_host_components(
    *,
    time_count: int,
    n_cells: int,
    n_sats: int,
    n_links: int,
    cell_chunk_size: int,
    store_eligible_mask: bool,
    propagation_dtype: Any,
    payload_dtype: Any,
    belt_id_dtype: Any,
    counts_dtype: Any,
    time_dtype: Any,
    gpu_resident: bool = False,
) -> dict[str, int]:
    t_count = int(max(0, time_count))
    cell_count = int(max(0, n_cells))
    sat_count = int(max(0, n_sats))
    link_count = int(max(0, n_links))
    chunk_count = int(max(0, min(cell_chunk_size, cell_count)))

    # In the GPU direct-EPFD path (gpu_resident=True), chunk arrays and payload
    # accumulation arrays live on the GPU device.  The host only sees the final
    # exported copies (accounted for by the export stage estimator).  Counting
    # them here would massively overestimate host memory and limit batch size.
    if gpu_resident:
        components = {
            "times": _shape_nbytes((t_count,), time_dtype),
            "ras_station_topo": _shape_nbytes((t_count, 1, sat_count, 4), propagation_dtype),
        }
        return components

    components = {
        "times": _shape_nbytes((t_count,), time_dtype),
        "payload_sat_azimuth": _shape_nbytes((t_count, cell_count, link_count), payload_dtype),
        "payload_sat_elevation": _shape_nbytes((t_count, cell_count, link_count), payload_dtype),
        "payload_sat_alpha": _shape_nbytes((t_count, cell_count, link_count), payload_dtype),
        "payload_sat_beta": _shape_nbytes((t_count, cell_count, link_count), payload_dtype),
        "payload_sat_belt_id": _shape_nbytes((t_count, cell_count, link_count), belt_id_dtype),
        "payload_sat_beam_counts_demand": _shape_nbytes((t_count, sat_count), counts_dtype),
        "payload_sat_beam_counts_eligible": _shape_nbytes((t_count, sat_count), counts_dtype),
        "ras_station_topo": _shape_nbytes((t_count, 1, sat_count, 4), propagation_dtype),
        "chunk_topo": _shape_nbytes((t_count, chunk_count, sat_count, 4), propagation_dtype),
        "chunk_sat_azel": _shape_nbytes((t_count, chunk_count, sat_count, 3), propagation_dtype),
        "chunk_selected_sat_azimuth": _shape_nbytes((t_count, chunk_count, link_count), payload_dtype),
        "chunk_selected_sat_elevation": _shape_nbytes((t_count, chunk_count, link_count), payload_dtype),
        "chunk_selected_sat_alpha": _shape_nbytes((t_count, chunk_count, link_count), payload_dtype),
        "chunk_selected_sat_beta": _shape_nbytes((t_count, chunk_count, link_count), payload_dtype),
        "chunk_selected_sat_belt_id": _shape_nbytes((t_count, chunk_count, link_count), belt_id_dtype),
        "chunk_sat_beam_counts_demand": _shape_nbytes((t_count, sat_count), counts_dtype),
        "chunk_sat_beam_counts_eligible": _shape_nbytes((t_count, sat_count), counts_dtype),
        "payload_sat_eligible_mask": (
            _shape_nbytes((t_count, cell_count, sat_count), np.bool_)
            if store_eligible_mask
            else 0
        ),
        "chunk_sat_eligible_mask": (
            _shape_nbytes((t_count, chunk_count, sat_count), np.bool_)
            if store_eligible_mask
            else 0
        ),
    }
    return components


def estimate_step1_host_batch_bytes(
    *,
    time_count: int,
    n_cells: int,
    n_sats: int,
    n_links: int,
    cell_chunk_size: int,
    store_eligible_mask: bool,
    propagation_dtype: Any = np.float64,
    payload_dtype: Any = np.float32,
    belt_id_dtype: Any = np.int16,
    counts_dtype: Any = np.int32,
    time_dtype: Any = np.float64,
    gpu_resident: bool = False,
) -> dict[str, Any]:
    """
    Estimate peak host memory for the NbeamEstimationFlow notebook batch loop.

    Parameters
    ----------
    time_count : int
        Number of timesteps in the candidate batch.
    n_cells : int
        Number of served cell observers processed in the NbeamEstimationFlow run.
    n_sats : int
        Number of propagated satellites.
    n_links : int
        Number of serving links retained per cell, typically ``Nco``.
    cell_chunk_size : int
        Number of cell observers propagated at once inside the chunk loop.
    store_eligible_mask : bool
        Whether the batch payload includes the full `sat_eligible_mask`
        tensor.
    propagation_dtype : Any, optional
        Floating-point dtype used for dense propagation tensors such as
        `topo` and `sat_azel`.
    payload_dtype : Any, optional
        Floating-point dtype used for the final link payloads written to HDF5.
    belt_id_dtype : Any, optional
        Integer dtype used for the stored belt identifiers.
    counts_dtype : Any, optional
        Integer dtype used for the per-satellite demand and eligibility
        counters.
    time_dtype : Any, optional
        Floating-point dtype used for the stored MJD timestamps.
    gpu_resident : bool, optional
        When True, assume chunk geometry and payload accumulation arrays live
        on the GPU device, excluding them from the host estimate.  This is
        the correct mode for the GPU direct-EPFD path where
        ``return_device=True``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `fixed_bytes`, `per_timestep_bytes`,
        `peak_bytes`, `dominant_component`, and a detailed `components_bytes`
        breakdown.
    """
    components = _estimate_step1_host_components(
        time_count=time_count,
        n_cells=n_cells,
        n_sats=n_sats,
        n_links=n_links,
        cell_chunk_size=cell_chunk_size,
        store_eligible_mask=store_eligible_mask,
        propagation_dtype=propagation_dtype,
        payload_dtype=payload_dtype,
        belt_id_dtype=belt_id_dtype,
        counts_dtype=counts_dtype,
        time_dtype=time_dtype,
        gpu_resident=gpu_resident,
    )
    per_step_components = _estimate_step1_host_components(
        time_count=1,
        n_cells=n_cells,
        n_sats=n_sats,
        n_links=n_links,
        cell_chunk_size=cell_chunk_size,
        store_eligible_mask=store_eligible_mask,
        propagation_dtype=propagation_dtype,
        payload_dtype=payload_dtype,
        belt_id_dtype=belt_id_dtype,
        counts_dtype=counts_dtype,
        time_dtype=time_dtype,
        gpu_resident=gpu_resident,
    )
    dominant_component = max(components, key=components.get) if components else None
    return {
        "fixed_bytes": 0,
        "per_timestep_bytes": int(sum(per_step_components.values())),
        "peak_bytes": int(sum(components.values())),
        "dominant_component": dominant_component,
        "components_bytes": components,
    }


def _estimate_step2_host_components(
    *,
    time_count: int,
    n_sats_total: int,
    n_visible_sats: int,
    n_links: int,
    n_beams: int,
    n_sky_cells: int,
    include_total_pfd: bool,
    include_per_satellite_pfd: bool,
    propagation_dtype: Any,
    working_dtype: Any,
    time_dtype: Any,
    sky_rx_chunk_size: int,
    stream_chunk_size: int,
    stream_rescue_chunk_size: int,
) -> tuple[dict[str, int], dict[str, int]]:
    t_count = int(max(0, time_count))
    sat_total = int(max(0, n_sats_total))
    sat_visible = int(max(0, min(n_visible_sats, sat_total)))
    link_count = int(max(0, n_links))
    beam_count = int(max(0, n_beams))
    sky_count = int(max(0, n_sky_cells))
    sky_chunk = int(max(1, min(sat_visible if sat_visible > 0 else 1, sky_rx_chunk_size)))
    stream_chunk = int(max(stream_chunk_size, stream_rescue_chunk_size, 1))

    fixed_components = {
        "filtered_satellite_indices": _shape_nbytes((sat_visible,), np.int32),
        "filtered_min_elevation_deg": _shape_nbytes((sat_visible,), np.float32),
        "filtered_beta_max_rad": _shape_nbytes((sat_visible,), np.float32),
        "filtered_belt_id": _shape_nbytes((sat_visible,), np.int16),
        "filtered_orbit_radius_m": _shape_nbytes((sat_visible,), np.float32),
    }
    components = {
        "times": _shape_nbytes((t_count,), time_dtype),
        "propagation_topo_full": _shape_nbytes((t_count, 1, sat_total, 4), propagation_dtype),
        "propagation_sat_azel_full": _shape_nbytes((t_count, 1, sat_total, 3), propagation_dtype),
        "full_visibility_mask": _shape_nbytes((t_count, sat_total), np.bool_),
        "filtered_topo": _shape_nbytes((t_count, 1, sat_visible, 4), propagation_dtype),
        "filtered_sat_azel": _shape_nbytes((t_count, 1, sat_visible, 3), propagation_dtype),
        "filtered_visibility_mask": _shape_nbytes((t_count, sat_visible), np.bool_),
        "selector_assignments": _shape_nbytes((t_count, link_count), np.int32),
        "selector_is_co_sat": _shape_nbytes((t_count, sat_visible), np.bool_),
        "source_sat_azimuth_deg": _shape_nbytes((t_count, sat_visible), np.float32),
        "source_sat_elevation_deg": _shape_nbytes((t_count, sat_visible), np.float32),
        "source_sat_belt_id": _shape_nbytes((t_count, sat_visible), np.int16),
        "source_skycell_id": _shape_nbytes((t_count, sat_visible), np.int32),
        "source_kind": _shape_nbytes((t_count, sat_visible), np.int8),
        "source_id": _shape_nbytes((t_count, sat_visible), np.int32),
        "source_valid_mask": _shape_nbytes((t_count, sat_visible), np.bool_),
        "beam_frame_alpha0_rad": _shape_nbytes((t_count, sat_visible), np.float32),
        "beam_frame_beta0_rad": _shape_nbytes((t_count, sat_visible), np.float32),
        "beam_frame_trig": 4 * _shape_nbytes((t_count, sat_visible), np.float32),
        "stream_candidate_chunk": 2 * _shape_nbytes((t_count * sat_visible, stream_chunk), np.float32),
        "beam_assignment_idx": _shape_nbytes((t_count, sat_visible, beam_count), np.int32),
        "beam_assignment_alpha_rad": _shape_nbytes((t_count, sat_visible, beam_count), np.float32),
        "beam_assignment_beta_rad": _shape_nbytes((t_count, sat_visible, beam_count), np.float32),
        "beam_assignment_valid": _shape_nbytes((t_count, sat_visible), np.int16),
        "pointing_angles_deg": 2 * _shape_nbytes((t_count, sky_count), np.float32),
        "pointing_trig": 6 * _shape_nbytes((t_count, sky_count), np.float32),
        "tx_beam_workspace_float": 15 * _shape_nbytes((t_count, sat_visible, beam_count), working_dtype),
        "tx_beam_workspace_bool": 3 * _shape_nbytes((t_count, sat_visible, beam_count), np.bool_),
        "power_per_satellite_workspace": 14 * _shape_nbytes((t_count, sat_visible), working_dtype),
        "rx_chunk_workspace": 5 * _shape_nbytes((t_count, sky_count, sky_chunk), working_dtype),
        "prx_total": _shape_nbytes((t_count, sky_count), np.float32),
        "output_total_pfd": _shape_nbytes((t_count,), np.float32) if include_total_pfd else 0,
        "output_per_satellite_pfd": (
            _shape_nbytes((t_count, sat_total), np.float32) if include_per_satellite_pfd else 0
        ),
    }
    return fixed_components, components


def estimate_step2_host_batch_bytes(
    *,
    time_count: int,
    n_sats_total: int,
    n_visible_sats: int,
    n_links: int,
    n_beams: int,
    n_sky_cells: int,
    include_total_pfd: bool,
    include_per_satellite_pfd: bool,
    propagation_dtype: Any = np.float64,
    working_dtype: Any = np.float32,
    time_dtype: Any = np.float64,
    sky_rx_chunk_size: int = 64,
    stream_chunk_size: int = 8,
    stream_rescue_chunk_size: int = 32,
) -> dict[str, Any]:
    """
    Estimate peak host memory for the conditioned EPFDflow CPU notebook.

    Parameters
    ----------
    time_count : int
        Number of timesteps in the candidate batch.
    n_sats_total : int
        Total number of propagated satellites before horizon filtering.
    n_visible_sats : int
        Estimated number of satellites that remain after horizon filtering for
        the batch.
    n_links : int
        Number of co-satellite links retained by the selector (`Nco`).
    n_beams : int
        Number of beams assigned per visible row (`Nbeam`).
    n_sky_cells : int
        Number of RAS-station telescope pointing cells.
    include_total_pfd : bool
        Whether the batch payload includes the total PFD time series.
    include_per_satellite_pfd : bool
        Whether the batch payload includes the dense per-satellite PFD tensor.
    propagation_dtype : Any, optional
        Floating-point dtype of the propagated `topo` and `sat_azel` tensors.
    working_dtype : Any, optional
        Floating-point dtype used by the post-propagation power-accumulation
        pipeline.
    time_dtype : Any, optional
        Floating-point dtype used for the stored MJD timestamps.
    sky_rx_chunk_size : int, optional
        Satellite chunk length used by the CPU receive-pattern accumulation
        loop.
    stream_chunk_size : int, optional
        Streaming conditioned beam-fill chunk size.
    stream_rescue_chunk_size : int, optional
        Rescue-round chunk size used by the streaming beam filler.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `fixed_bytes`, `per_timestep_bytes`,
        `peak_bytes`, `dominant_component`, and detailed byte breakdowns.
    """
    fixed_components, components = _estimate_step2_host_components(
        time_count=time_count,
        n_sats_total=n_sats_total,
        n_visible_sats=n_visible_sats,
        n_links=n_links,
        n_beams=n_beams,
        n_sky_cells=n_sky_cells,
        include_total_pfd=include_total_pfd,
        include_per_satellite_pfd=include_per_satellite_pfd,
        propagation_dtype=propagation_dtype,
        working_dtype=working_dtype,
        time_dtype=time_dtype,
        sky_rx_chunk_size=sky_rx_chunk_size,
        stream_chunk_size=stream_chunk_size,
        stream_rescue_chunk_size=stream_rescue_chunk_size,
    )
    _, per_step_components = _estimate_step2_host_components(
        time_count=1,
        n_sats_total=n_sats_total,
        n_visible_sats=n_visible_sats,
        n_links=n_links,
        n_beams=n_beams,
        n_sky_cells=n_sky_cells,
        include_total_pfd=include_total_pfd,
        include_per_satellite_pfd=include_per_satellite_pfd,
        propagation_dtype=propagation_dtype,
        working_dtype=working_dtype,
        time_dtype=time_dtype,
        sky_rx_chunk_size=sky_rx_chunk_size,
        stream_chunk_size=stream_chunk_size,
        stream_rescue_chunk_size=stream_rescue_chunk_size,
    )
    combined_components = {**fixed_components, **components}
    dominant_component = max(combined_components, key=combined_components.get) if combined_components else None
    return {
        "fixed_bytes": int(sum(fixed_components.values())),
        "per_timestep_bytes": int(sum(per_step_components.values())),
        "peak_bytes": int(sum(fixed_components.values()) + sum(components.values())),
        "dominant_component": dominant_component,
        "fixed_components_bytes": fixed_components,
        "components_bytes": components,
    }


def build_observer_layout(
    primary_observer: Any,
    cell_observers: Iterable[Any],
) -> dict[str, Any]:
    """
    Build the observer ordering used by step-level propagation front ends.

    Parameters
    ----------
    primary_observer : Any
        Observer object that must occupy index 0 in the returned layout. This
        is typically the RAS station observer propagated separately for
        reference-frame diagnostics.
    cell_observers : Iterable[Any]
        Observer objects representing Earth-grid cells or other served
        locations. These observers are appended after ``primary_observer`` in
        the returned order.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        ``"observer_arr"``
            ``np.ndarray[object]`` with shape ``(1 + C,)`` containing the
            primary observer followed by the cell observers.
        ``"primary_observer_idx"``
            Integer index of the primary observer. Always ``0``.
        ``"first_cell_observer_idx"``
            Integer index of the first cell observer. Always ``1``.
        ``"n_cell_observers"``
            Number of cell observers appended after the primary observer.

    Raises
    ------
    ValueError
        If ``primary_observer`` is ``None``.

    Notes
    -----
    The helper intentionally preserves the long-standing convention that the
    primary observer lives at index 0 and all cell observers follow it. This
    avoids re-implementing observer-index bookkeeping in front-end scripts and
    notebooks.

    Examples
    --------
    >>> layout = build_observer_layout(ras_station, active_cells)
    >>> layout["primary_observer_idx"]
    0
    >>> layout["first_cell_observer_idx"]
    1
    """
    if primary_observer is None:
        raise ValueError("primary_observer must not be None.")

    cell_observer_list = list(cell_observers)
    observer_arr = np.asarray([primary_observer, *cell_observer_list], dtype=object)
    return {
        "observer_arr": observer_arr,
        "primary_observer_idx": 0,
        "first_cell_observer_idx": 1,
        "n_cell_observers": len(cell_observer_list),
    }


def format_byte_count(n_bytes: int | float) -> str:
    """
    Format a byte count for notebook/runtime diagnostics.

    Parameters
    ----------
    n_bytes : int or float
        Byte count to format.

    Returns
    -------
    str
        Human-readable value using binary units (KiB, MiB, GiB, TiB).
    """
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(round(value))} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def record_profile_stage(
    stage_timings: dict[str, float] | None,
    stage_name: str,
    stage_start: float | None,
    *,
    enabled: bool,
    synchronize: Callable[[], None] | None = None,
) -> float | None:
    """
    Record the elapsed time of a profiling stage and return the next stage start.

    Parameters
    ----------
    stage_timings : dict[str, float] or None
        Mutable stage-timing dictionary updated in place when profiling is
        enabled.
    stage_name : str
        Profiling label to write into ``stage_timings``.
    stage_start : float or None
        Previous stage start timestamp from :func:`time.perf_counter`.
    enabled : bool
        Whether profiling is active.
    synchronize : callable or None, optional
        Optional device-synchronization callback executed immediately before
        the elapsed time is measured.

    Returns
    -------
    float or None
        New stage start timestamp when profiling is enabled, else the input
        ``stage_start`` unchanged.
    """
    if not enabled or stage_timings is None or stage_start is None:
        return stage_start
    if synchronize is not None:
        synchronize()
    stage_timings[stage_name] = perf_counter() - stage_start
    return perf_counter()


def _accumulate_profile_timing(
    stage_timings: dict[str, float] | None,
    stage_name: str,
    elapsed_seconds: float,
) -> None:
    """Accumulate a profiling duration into `stage_timings` in place."""
    if stage_timings is None:
        return
    stage_timings[stage_name] = float(stage_timings.get(stage_name, 0.0)) + float(
        elapsed_seconds
    )


class _DirectEpfdGpuStageProfiler:
    """Collect per-stage GPU timings without synchronizing every hot-loop slab."""

    def __init__(self, cp: Any, *, enabled: bool):
        self._cp = cp
        self._enabled = bool(enabled)
        cuda_module = getattr(cp, "cuda", None)
        event_factory = getattr(cuda_module, "Event", None)
        elapsed_fn = getattr(cuda_module, "get_elapsed_time", None)
        self._event_factory = event_factory if callable(event_factory) else None
        self._elapsed_fn = elapsed_fn if callable(elapsed_fn) else None
        self._use_events = bool(self._enabled and self._event_factory and self._elapsed_fn)
        self._event_pairs: dict[str, list[tuple[Any, Any]]] = {}
        self._last_event: Any | None = None

    def start(self, stage_name: str) -> tuple[str, Any] | None:
        """Start a stage token for later completion."""
        if not self._enabled:
            return None
        if self._use_events:
            start_event = self._event_factory()
            start_event.record()
            return str(stage_name), start_event
        return str(stage_name), perf_counter()

    def stop(
        self,
        token: tuple[str, Any] | None,
        *,
        stage_timings: dict[str, float] | None,
    ) -> None:
        """Stop a stage token and record elapsed time or defer it via CUDA events."""
        if not self._enabled or token is None:
            return
        stage_name, stage_start = token
        if self._use_events:
            end_event = self._event_factory()
            end_event.record()
            self._event_pairs.setdefault(str(stage_name), []).append((stage_start, end_event))
            self._last_event = end_event
            return

        sync_t0 = perf_counter()
        _sync_array_module(self._cp)
        _accumulate_profile_timing(
            stage_timings,
            "host_sync_telemetry",
            perf_counter() - sync_t0,
        )
        _accumulate_profile_timing(
            stage_timings,
            str(stage_name),
            perf_counter() - float(stage_start),
        )

    def finalize(self, stage_timings: dict[str, float] | None) -> None:
        """Flush deferred CUDA-event timings into `stage_timings`."""
        if not self._use_events or stage_timings is None or self._last_event is None:
            return
        sync_t0 = perf_counter()
        self._last_event.synchronize()
        _accumulate_profile_timing(
            stage_timings,
            "host_sync_telemetry",
            perf_counter() - sync_t0,
        )
        for stage_name, pairs in self._event_pairs.items():
            elapsed_ms = 0.0
            for start_event, end_event in pairs:
                elapsed_ms += float(self._elapsed_fn(start_event, end_event))
            _accumulate_profile_timing(
                stage_timings,
                str(stage_name),
                elapsed_ms / 1000.0,
            )
        self._event_pairs.clear()
        self._last_event = None


def _maybe_set_progress_description(
    pbar: Any,
    *,
    enabled: bool,
    text: str,
) -> None:
    """Best-effort tqdm description update used by long-running tail stages."""
    if not enabled or not hasattr(pbar, "set_description"):
        return
    text_value = str(text)
    try:
        if getattr(pbar, "_scepter_last_progress_desc", None) == text_value:
            return
    except Exception:
        pass
    try:
        pbar.set_description(text_value)
        try:
            setattr(pbar, "_scepter_last_progress_desc", text_value)
        except Exception:
            pass
    except Exception:
        return


_DIRECT_EPFD_PROGRESS_DESC_MODES = frozenset({"off", "coarse", "detailed"})


def _resolve_progress_desc_mode(
    *,
    enable_progress_desc_updates: bool,
    progress_desc_mode: str | None,
) -> str:
    """Resolve backward-compatible progress-description policy."""
    if not bool(enable_progress_desc_updates):
        return "off"
    if progress_desc_mode is None:
        return "coarse"
    mode_name = str(progress_desc_mode).strip().lower()
    if mode_name not in _DIRECT_EPFD_PROGRESS_DESC_MODES:
        raise ValueError(
            "`progress_desc_mode` must be one of 'off', 'coarse', or 'detailed'."
        )
    return mode_name


def _resolve_writer_checkpoint_interval_s(
    writer_checkpoint_interval_s: float | None,
) -> float | None:
    """Normalize the optional periodic durable-checkpoint interval."""
    if writer_checkpoint_interval_s is None:
        return None
    interval_value = float(writer_checkpoint_interval_s)
    if interval_value <= 0.0:
        return None
    return interval_value


def _direct_epfd_progress_text(
    progress_desc_mode: str,
    phase: str,
    *,
    chunk_i: int | None = None,
    n_cell_chunks: int | None = None,
    c0: int | None = None,
    c1: int | None = None,
) -> str | None:
    """Return the desired progress description for one runner phase."""
    mode_name = str(progress_desc_mode).strip().lower()
    if mode_name == "off":
        return None
    if mode_name == "coarse":
        coarse_map = {
            "chunks": "Chunks",
            "beam_finalize": "Write tail",
            "power_accumulation": "Write tail",
            "export_copy": "Write tail",
            "write_enqueue": "Write tail",
            "checkpoint": "Checkpoint",
            "final_flush": "Final flush",
        }
        return coarse_map.get(str(phase))
    if mode_name != "detailed":
        return None
    detailed_map = {
        "chunks": "Processing time slices",
        "beam_finalize": "Finalizing beams",
        "power_accumulation": "Accumulating power",
        "export_copy": "Exporting to host",
        "write_enqueue": "Queueing HDF5 write",
        "checkpoint": "Checkpoint",
        "final_flush": "Flushing writer",
    }
    if str(phase) == "chunk_detail":
        if None in (chunk_i, n_cell_chunks, c0, c1):
            return None
        return f"Chunk {int(chunk_i) + 1}/{int(n_cell_chunks)} cells {int(c0)}:{int(c1)}"
    return detailed_map.get(str(phase))


def _set_direct_epfd_progress_phase(
    pbar: Any,
    *,
    enable_progress_bars: bool,
    progress_desc_mode: str,
    phase: str,
    chunk_i: int | None = None,
    n_cell_chunks: int | None = None,
    c0: int | None = None,
    c1: int | None = None,
) -> None:
    """Apply the resolved progress-description policy to one phase update."""
    text = _direct_epfd_progress_text(
        progress_desc_mode,
        phase,
        chunk_i=chunk_i,
        n_cell_chunks=n_cell_chunks,
        c0=c0,
        c1=c1,
    )
    if text is None:
        return
    _maybe_set_progress_description(
        pbar,
        enabled=bool(enable_progress_bars),
        text=text,
    )


def _emit_direct_epfd_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    **payload: Any,
) -> None:
    """Best-effort structured progress notification for GUI consumers."""
    if progress_callback is None:
        return
    try:
        progress_callback(dict(payload))
    except Exception:
        return


class _RunCancellationRequested(RuntimeError):
    """Internal control-flow exception used for safe run cancellation."""

    def __init__(self, mode: str, boundary: str) -> None:
        super().__init__(f"Run cancellation requested: mode={mode}, boundary={boundary}")
        self.mode = str(mode)
        self.boundary = str(boundary)


class _DirectGpuOutOfMemory(RuntimeError):
    """Internal wrapper that tags a GPU OOM with the active direct-EPFD stage."""

    def __init__(
        self,
        stage: str,
        exc: BaseException,
        *,
        stage_memory_summary: Mapping[str, Any] | None = None,
    ) -> None:
        message = f"GPU out of memory during {stage}: {exc}"
        super().__init__(message)
        self.stage = str(stage)
        self.original_exception = exc
        self.stage_memory_summary = dict(stage_memory_summary or {})


def _normalise_direct_epfd_cancel_mode(mode: Any) -> str:
    text = str(mode or "none").strip().lower()
    if text in {"graceful", "stop", "soft"}:
        return "graceful"
    if text in {"force", "hard", "abort"}:
        return "force"
    return "none"


def _query_direct_epfd_cancel_mode(
    cancel_callback: Callable[[], str | None] | None,
) -> str:
    if cancel_callback is None:
        return "none"
    try:
        return _normalise_direct_epfd_cancel_mode(cancel_callback())
    except Exception:
        return "none"


def _empty_writer_stats_summary() -> dict[str, Any]:
    """Return a zeroed writer telemetry snapshot."""
    return {
        "queued_items": 0,
        "queued_bytes": 0,
        "queued_items_high_water": 0,
        "queued_bytes_high_water": 0,
        "submitted_seq": 0,
        "completed_seq": 0,
        "durable_seq": 0,
        "prepare_elapsed_total": 0.0,
        "apply_elapsed_total": 0.0,
        "submit_wait_elapsed_total": 0.0,
        "flush_count": 0,
        "writer_cycle_count": 0,
        "writer_cycle_items_high_water": 0,
        "writer_cycle_bytes_high_water": 0,
        "durable_flush_count": 0,
        "durable_elapsed_total": 0.0,
        "durability_mode": "flush_only",
    }


def _maybe_checkpoint_writer_durable(
    storage_filename: str,
    *,
    checkpoint_interval_s: float | None,
    last_checkpoint_monotonic: float | None,
    pbar: Any,
    enable_progress_bars: bool,
    progress_desc_mode: str,
) -> tuple[float | None, float, bool]:
    """Flush queued writes at safe post-batch boundaries when the interval expires."""
    if checkpoint_interval_s is None or last_checkpoint_monotonic is None:
        return last_checkpoint_monotonic, 0.0, False
    now = perf_counter()
    if (now - float(last_checkpoint_monotonic)) < float(checkpoint_interval_s):
        return last_checkpoint_monotonic, 0.0, False
    _set_direct_epfd_progress_phase(
        pbar,
        enable_progress_bars=enable_progress_bars,
        progress_desc_mode=progress_desc_mode,
        phase="checkpoint",
    )
    flush_t0 = perf_counter()
    flush_writes(storage_filename)
    flush_elapsed = perf_counter() - flush_t0
    return perf_counter(), float(flush_elapsed), True


def _sync_array_module(array_module: Any) -> None:
    try:
        array_module.cuda.Stream.null.synchronize()
    except Exception:
        return


def _scalar_from_device(value: Any) -> float:
    try:
        return float(value.get())
    except Exception:
        return float(np.asarray(value).reshape(()))


def _observe_visible_satellite_batch_state(
    cp: Any,
    *,
    sat_keep_batch: Any,
    visible_satellite_est: int,
    n_sats_total: int,
    need_exact_count: bool,
    stage_timings: dict[str, float] | None = None,
) -> tuple[bool, int | None]:
    """
    Return whether a batch has visible satellites and, when needed, the exact count.

    When the scheduler estimate already saturates the constellation size and the
    caller only needs to know whether the batch is empty, use a single-device
    boolean reduction instead of a full visible-count extraction.
    """
    sync_t0 = perf_counter() if stage_timings is not None else None
    if bool(need_exact_count) or int(visible_satellite_est) < int(n_sats_total):
        visible_count = int(_scalar_from_device(cp.count_nonzero(sat_keep_batch)))
        if sync_t0 is not None:
            _accumulate_profile_timing(
                stage_timings,
                "host_sync_telemetry",
                perf_counter() - sync_t0,
            )
        return bool(visible_count > 0), int(visible_count)

    any_visible = bool(_scalar_from_device(cp.any(sat_keep_batch)))
    if sync_t0 is not None:
        _accumulate_profile_timing(
            stage_timings,
            "host_sync_telemetry",
            perf_counter() - sync_t0,
        )
    if any_visible:
        return True, None
    return False, 0


def _aggregate_direct_epfd_profile_timing_rows(
    profile_rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, float]:
    """Aggregate per-batch stage timing rows into a run-level timing summary."""
    totals: dict[str, float] = {}
    if profile_rows is None:
        return totals
    for row in profile_rows:
        if not isinstance(row, Mapping):
            continue
        for key, value in row.items():
            key_str = str(key)
            if key_str in {"iteration", "batch_index"} or key_str.endswith("_count"):
                continue
            try:
                value_f = float(value)
            except Exception:
                continue
            totals[key_str] = float(totals.get(key_str, 0.0)) + value_f
    return totals


def _optional_angle_deg(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "to_value"):
        return float(u.Quantity(value).to_value(u.deg))
    return float(value)


def _normalize_direct_epfd_selection_strategy(selection_strategy: str) -> str:
    strategy_name = str(selection_strategy).strip().lower()
    if strategy_name in {"random", "random_pointing"}:
        return "random"
    if strategy_name in {"max_elevation", "maximum_elevation"}:
        return "max_elevation"
    raise ValueError("Use selection_strategy='max_elevation' or 'random'.")


def _normalize_direct_epfd_power_input_quantity(quantity: str | None) -> str:
    quantity_name = str(quantity or "target_pfd").strip().lower()
    if quantity_name not in _DIRECT_EPFD_POWER_INPUT_QUANTITIES:
        raise ValueError(
            "power_input_quantity must be one of "
            f"{sorted(_DIRECT_EPFD_POWER_INPUT_QUANTITIES)!r}; got {quantity!r}."
        )
    return quantity_name


def _normalize_direct_epfd_power_input_basis(basis: str | None) -> str:
    basis_name = str(basis or "per_mhz").strip().lower()
    if basis_name not in _DIRECT_EPFD_POWER_INPUT_BASES:
        raise ValueError(
            "power_input_basis must be one of "
            f"{sorted(_DIRECT_EPFD_POWER_INPUT_BASES)!r}; got {basis!r}."
        )
    return basis_name


def _normalize_direct_epfd_cell_activity_mode(mode: str | None) -> str:
    mode_name = str(mode or "whole_cell").strip().lower()
    if mode_name not in _DIRECT_EPFD_CELL_ACTIVITY_MODES:
        raise ValueError(
            "cell_activity_mode must be one of "
            f"{sorted(_DIRECT_EPFD_CELL_ACTIVITY_MODES)!r}; got {mode!r}."
        )
    return mode_name


def _canonical_direct_epfd_dataset_name(name: str) -> str:
    return str(_DIRECT_EPFD_CANONICAL_RAW_DATASET_NAMES.get(str(name), str(name)))


def _legacy_direct_epfd_dataset_name(name: str) -> str:
    return str(_DIRECT_EPFD_LEGACY_RAW_DATASET_NAMES.get(str(name), str(name)))


def convert_direct_epfd_power_basis_db(
    value_db: float,
    *,
    bandwidth_mhz: float,
    from_basis: str,
    to_basis: str,
) -> float:
    """Convert a power-like or PFD-like logarithmic value between MHz and channel bases."""
    from_basis_name = _normalize_direct_epfd_power_input_basis(from_basis)
    to_basis_name = _normalize_direct_epfd_power_input_basis(to_basis)
    if from_basis_name == to_basis_name:
        return float(value_db)
    bandwidth = float(bandwidth_mhz)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth_mhz must be finite and > 0.")
    offset_db = 10.0 * np.log10(bandwidth)
    if from_basis_name == "per_mhz":
        return float(value_db) + float(offset_db)
    return float(value_db) - float(offset_db)


def normalize_direct_epfd_power_input(
    *,
    bandwidth_mhz: float,
    power_input_quantity: str | None = "target_pfd",
    power_input_basis: str | None = "per_mhz",
    pfd0_dbw_m2_mhz: float | None = None,
    target_pfd_dbw_m2_mhz: float | None = None,
    target_pfd_dbw_m2_channel: float | None = None,
    satellite_ptx_dbw_mhz: float | None = None,
    satellite_ptx_dbw_channel: float | None = None,
    satellite_eirp_dbw_mhz: float | None = None,
    satellite_eirp_dbw_channel: float | None = None,
    power_variation_mode: str | None = None,
    power_range_min_db: float | None = None,
    power_range_max_db: float | None = None,
) -> dict[str, Any]:
    """Resolve the active quantity/basis pair and derive its complementary basis."""
    quantity = _normalize_direct_epfd_power_input_quantity(power_input_quantity)
    basis = _normalize_direct_epfd_power_input_basis(power_input_basis)
    bandwidth = float(bandwidth_mhz)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth_mhz must be finite and > 0.")

    target_mhz = target_pfd_dbw_m2_mhz
    if quantity == "target_pfd" and target_mhz is None and pfd0_dbw_m2_mhz is not None:
        target_mhz = float(pfd0_dbw_m2_mhz)
    if (
        quantity == "target_pfd"
        and target_pfd_dbw_m2_mhz is not None
        and pfd0_dbw_m2_mhz is not None
        and np.isfinite(float(target_pfd_dbw_m2_mhz))
        and np.isfinite(float(pfd0_dbw_m2_mhz))
        and not np.isclose(
            float(target_pfd_dbw_m2_mhz),
            float(pfd0_dbw_m2_mhz),
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise ValueError(
            "pfd0_dbw_m2_mhz and target_pfd_dbw_m2_mhz must agree when both are provided."
        )

    values_by_field: dict[tuple[str, str], float | None] = {
        ("target_pfd", "per_mhz"): (
            None if target_mhz is None else float(target_mhz)
        ),
        ("target_pfd", "per_channel"): (
            None if target_pfd_dbw_m2_channel is None else float(target_pfd_dbw_m2_channel)
        ),
        ("satellite_ptx", "per_mhz"): (
            None if satellite_ptx_dbw_mhz is None else float(satellite_ptx_dbw_mhz)
        ),
        ("satellite_ptx", "per_channel"): (
            None if satellite_ptx_dbw_channel is None else float(satellite_ptx_dbw_channel)
        ),
        ("satellite_eirp", "per_mhz"): (
            None if satellite_eirp_dbw_mhz is None else float(satellite_eirp_dbw_mhz)
        ),
        ("satellite_eirp", "per_channel"): (
            None if satellite_eirp_dbw_channel is None else float(satellite_eirp_dbw_channel)
        ),
    }
    per_mhz_value = values_by_field[(quantity, "per_mhz")]
    per_channel_value = values_by_field[(quantity, "per_channel")]
    if (
        per_mhz_value is not None
        and per_channel_value is not None
        and np.isfinite(float(per_mhz_value))
        and np.isfinite(float(per_channel_value))
    ):
        derived_channel = convert_direct_epfd_power_basis_db(
            float(per_mhz_value),
            bandwidth_mhz=bandwidth,
            from_basis="per_mhz",
            to_basis="per_channel",
        )
        if not np.isclose(
            float(per_channel_value),
            float(derived_channel),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                f"{quantity} per-MHz and per-channel values are inconsistent for "
                f"bandwidth_mhz={bandwidth!r}."
            )
    active_value = values_by_field[(quantity, basis)]
    # In variation mode the fixed-value field may be empty; accept
    # the range midpoint as the representative active value instead.
    var_mode_raw = str(power_variation_mode or "fixed").strip().lower()
    is_variation = var_mode_raw in {"uniform_random", "slant_range"} and quantity != "target_pfd"
    if is_variation and (active_value is None or not np.isfinite(float(active_value))):
        if (
            power_range_min_db is not None
            and power_range_max_db is not None
            and np.isfinite(float(power_range_min_db))
            and np.isfinite(float(power_range_max_db))
        ):
            active_value = 0.5 * (float(power_range_min_db) + float(power_range_max_db))
            values_by_field[(quantity, basis)] = active_value
        else:
            raise ValueError(
                f"Finite min and max range values are required for "
                f"power_input_quantity={quantity!r} in {var_mode_raw} mode."
            )
    elif active_value is None or not np.isfinite(float(active_value)):
        raise ValueError(
            f"A finite value is required for power_input_quantity={quantity!r} "
            f"and power_input_basis={basis!r}."
        )

    complementary_basis = "per_channel" if basis == "per_mhz" else "per_mhz"
    complementary_value = convert_direct_epfd_power_basis_db(
        float(active_value),
        bandwidth_mhz=bandwidth,
        from_basis=basis,
        to_basis=complementary_basis,
    )
    values_by_field[(quantity, complementary_basis)] = float(complementary_value)

    resolved = {
        "bandwidth_mhz": float(bandwidth),
        "power_input_quantity": str(quantity),
        "power_input_basis": str(basis),
        "active_value": float(active_value),
        "active_value_unit": {
            ("target_pfd", "per_mhz"): "dBW/m^2/MHz",
            ("target_pfd", "per_channel"): "dBW/m^2",
            ("satellite_ptx", "per_mhz"): "dBW/MHz",
            ("satellite_ptx", "per_channel"): "dBW",
            ("satellite_eirp", "per_mhz"): "dBW/MHz",
            ("satellite_eirp", "per_channel"): "dBW",
        }[(quantity, basis)],
        "target_pfd_dbw_m2_mhz": values_by_field[("target_pfd", "per_mhz")],
        "target_pfd_dbw_m2_channel": values_by_field[("target_pfd", "per_channel")],
        "satellite_ptx_dbw_mhz": values_by_field[("satellite_ptx", "per_mhz")],
        "satellite_ptx_dbw_channel": values_by_field[("satellite_ptx", "per_channel")],
        "satellite_eirp_dbw_mhz": values_by_field[("satellite_eirp", "per_mhz")],
        "satellite_eirp_dbw_channel": values_by_field[("satellite_eirp", "per_channel")],
    }
    # Power variation (optional range for satellite_eirp / satellite_ptx)
    var_mode = str(power_variation_mode or "fixed").strip().lower()
    if var_mode not in {"fixed", "uniform_random", "slant_range"}:
        var_mode = "fixed"
    if var_mode != "fixed" and quantity == "target_pfd":
        var_mode = "fixed"  # target_pfd already varies by geometry
    pwr_min_channel: float | None = None
    pwr_max_channel: float | None = None
    if var_mode != "fixed" and power_range_min_db is not None and power_range_max_db is not None:
        # Convert min/max to per-channel dB using the same basis as the active value
        if basis == "per_mhz":
            pwr_min_channel = convert_direct_epfd_power_basis_db(
                float(power_range_min_db), bandwidth_mhz=bandwidth, from_basis="per_mhz", to_basis="per_channel",
            )
            pwr_max_channel = convert_direct_epfd_power_basis_db(
                float(power_range_max_db), bandwidth_mhz=bandwidth, from_basis="per_mhz", to_basis="per_channel",
            )
        else:
            pwr_min_channel = float(power_range_min_db)
            pwr_max_channel = float(power_range_max_db)
        if pwr_min_channel > pwr_max_channel:
            pwr_min_channel, pwr_max_channel = pwr_max_channel, pwr_min_channel
    resolved["power_variation_mode"] = var_mode
    resolved["power_range_min_dbw_channel"] = pwr_min_channel
    resolved["power_range_max_dbw_channel"] = pwr_max_channel
    return resolved


def _normalize_direct_epfd_reuse_factor(reuse_factor: int | None) -> int:
    reuse_factor_i = int(1 if reuse_factor is None else reuse_factor)
    if reuse_factor_i not in _DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
        raise ValueError(
            "reuse_factor must be one of "
            f"{list(_DIRECT_EPFD_SUPPORTED_REUSE_FACTORS)!r}; got {reuse_factor!r}."
        )
    return int(reuse_factor_i)


def _normalize_direct_epfd_reference_mode(
    mode: str | None,
    *,
    default: str,
) -> str:
    mode_name = str(default if mode in {None, ""} else mode).strip().lower()
    if mode_name not in _DIRECT_EPFD_REFERENCE_MODES:
        raise ValueError(
            "reference mode must be one of "
            f"{sorted(_DIRECT_EPFD_REFERENCE_MODES)!r}; got {mode!r}."
        )
    return str(mode_name)


def _normalize_direct_epfd_power_policy(policy: str | None) -> str:
    policy_name = str("repeat_per_group" if policy in {None, ""} else policy).strip().lower()
    if policy_name not in _DIRECT_EPFD_SPECTRUM_POWER_POLICIES:
        raise ValueError(
            "multi_group_power_policy must be one of "
            f"{sorted(_DIRECT_EPFD_SPECTRUM_POWER_POLICIES)!r}; got {policy!r}."
        )
    return str(policy_name)


def _normalize_direct_epfd_split_group_denominator_mode(mode: str | None) -> str:
    mode_name = str(
        "configured_groups" if mode in {None, ""} else mode
    ).strip().lower()
    if mode_name not in _DIRECT_EPFD_SPLIT_GROUP_DENOMINATOR_MODES:
        raise ValueError(
            "split_total_group_denominator_mode must be one of "
            f"{sorted(_DIRECT_EPFD_SPLIT_GROUP_DENOMINATOR_MODES)!r}; got {mode!r}."
        )
    return str(mode_name)


def _normalize_direct_epfd_cutoff_basis(basis: str | None) -> str:
    basis_name = str("channel_bandwidth" if basis in {None, ""} else basis).strip().lower()
    if basis_name not in _DIRECT_EPFD_SPECTRAL_CUTOFF_BASES:
        raise ValueError(
            "spectral_integration_cutoff_basis must be one of "
            f"{sorted(_DIRECT_EPFD_SPECTRAL_CUTOFF_BASES)!r}; got {basis!r}."
        )
    return str(basis_name)


def _normalize_direct_epfd_mask_preset(preset: str | None) -> str:
    preset_name = str("sm1541_fss" if preset in {None, ""} else preset).strip().lower()
    if preset_name not in _DIRECT_EPFD_MASK_PRESETS:
        raise ValueError(
            "unwanted_emission_mask_preset must be one of "
            f"{sorted(_DIRECT_EPFD_MASK_PRESETS)!r}; got {preset!r}."
        )
    return str(preset_name)


def _resolve_direct_epfd_reference_frequencies_mhz(
    *,
    start_mhz: float,
    stop_mhz: float,
    mode: str,
    point_count: int,
) -> np.ndarray:
    lower = float(start_mhz)
    upper = float(stop_mhz)
    if mode == "lower":
        return np.asarray([lower], dtype=np.float64)
    if mode == "upper":
        return np.asarray([upper], dtype=np.float64)
    if mode == "middle":
        return np.asarray([0.5 * (lower + upper)], dtype=np.float64)
    count = max(2, int(point_count))
    return np.linspace(lower, upper, count, dtype=np.float64)


def _sm329_spurious_db(ras_frequency_ghz: float, category: str) -> float:
    """Return the SM.329 spurious domain attenuation level in dBc.

    Per ITU-R SM.329-13 Table 2 (Category A), the spurious-domain limit
    for space stations (and space-service earth stations) is:

        min(43 + 10·log10(P), 60 dBc)   in a 4 kHz reference bandwidth

    We apply the 60 dBc cap (the less-stringent branch, appropriate for
    typical high-power satellite transmitters where P > 500 W makes the
    relative formula looser than the cap). The cap is frequency-
    independent per SM.329-13 §4.1 — space services always use 4 kHz
    reference bandwidth regardless of the spurious frequency.

    Category B represents stricter regional (European) limits; we model
    them as +5 dB additional attenuation over the Category A cap as a
    representative design margin.
    """
    base = 60.0
    if category == "b":
        return base + 5.0
    return base


def _direct_epfd_relative_mask_points(
    *,
    preset: str,
    ras_frequency_ghz: float = 1.4,
) -> list[tuple[float, float]]:
    """Return OOB mask breakpoints as ``(multiplier, attenuation_dB)`` pairs.

    Multipliers are in units of channel bandwidth measured from the
    **channel centre** (per ITU-R SM.1541-7).  The channel edge is at
    multiplier = 0.5.

    For combined SM.1539 + SM.329 presets, the OOB mask (SM.1539) extends
    to ±2.5 × B_N from centre.  Beyond that the spurious domain (SM.329)
    applies at a frequency-dependent attenuation level.
    """
    # --- Flat preset: no suppression (UEMR baseline) ---
    if preset == "flat":
        return [(0.5, 0.0), (1000.0, 0.0)]
    # --- SM.1541-7 Annex 5 §2 (FSS) and §3 (MSS) OoB masks ---
    # ITU formula (dBsd, reference bandwidth 4 kHz below 15 GHz,
    # 1 MHz above 15 GHz):
    #   A(F) = 40 · log10(F/50 + 1)
    # where F is the frequency offset from the edge of the total
    # assigned band expressed as a percentage of the necessary
    # bandwidth B_N. The edge of the assigned band is at multiplier
    # k = 0.5 × B_N from the channel centre, so F = 100·(k − 0.5).
    # Substituting gives the equivalent centre-referenced form
    #   A(k) = 40 · log10(2·k)   for k ∈ [0.5, 2.5]
    # (A(0.5) = 0 dB at the channel edge; A(2.5) ≈ 27.96 dB at the
    # OoB/spurious boundary per SM.1539).
    #
    # We sample the continuous curve at 17 points because the
    # downstream integrator interpolates log-linearly in dB between
    # breakpoints — sparse sampling undershoots the integrated
    # leakage by up to ~0.66 dB near the band edge. 17 points gives
    # <0.1 dB integration error versus the closed form for typical
    # RAS-band widths.
    _sm1541_oob = [
        (float(k), 40.0 * np.log10(2.0 * float(k)) if k > 0.5 else 0.0)
        for k in np.linspace(0.5, 2.5, 17)
    ]
    if preset in {"sm1541_fss", "sm1541_mss"}:
        return _sm1541_oob
    # --- SM.1541-7 + SM.329-13 combined masks ---
    # OoB domain (SM.1541-7 Annex 5 §§2–3): 0.5 to 2.5 × B_N
    # Spurious domain (SM.329-13 Table 2, Category A space stations):
    #   43 + 10·log10(P) or 60 dBc, whichever is less stringent
    # beyond 2.5 × B_N from the centre (SM.1539-2 boundary).
    _sm1541_fss_oob = _sm1541_oob
    _sm1541_mss_oob = _sm1541_oob  # SM.1541-7 §3 MSS formula is identical to §2 FSS
    if preset == "sm1541_sm329_fss":
        spur = _sm329_spurious_db(ras_frequency_ghz, "a")
        return _sm1541_fss_oob + [(2.5 + 0.01, spur), (10.0, spur)]
    if preset == "sm1541_sm329_mss":
        spur = _sm329_spurious_db(ras_frequency_ghz, "a")
        return _sm1541_mss_oob + [(2.5 + 0.01, spur), (10.0, spur)]
    # --- Other presets ---
    if preset == "3gpp_ts_36_104":
        # 3GPP TS 36.104 Table 6.6.2.1-1 OoB emission limits (5 MHz ref BW).
        # Offsets are in multiples of channel bandwidth from channel centre.
        return [
            (0.5, 0.0),
            (1.5, 30.0),
            (2.5, 45.0),
            (3.5, 50.0),
        ]
    if preset == "wrc27_1_13_s1_dc_mss_imt":
        # WRC-27 AI 1.13 System 1 DC-MSS-IMT OOBE mask for 5 MHz reference
        # bandwidth (space-to-Earth downlink).  From WP 4C document 4C/319
        # (RUS), section A6.3.2.1, Figure 3 — referenced as "3GPP TS
        # 36.104 *1" in the system characteristics table (4C/356, row 31).
        # Breakpoints (from channel centre, in multiples of channel BW):
        #   0.5→0, 1.5→-38, 2.5→-45, 6.0→-52, 9.0→-60 dBc.
        return [
            (0.5, 0.0),
            (1.5, 38.0),
            (2.5, 45.0),
            (6.0, 52.0),
            (9.0, 60.0),
        ]
    if preset == "adjacent_45_nonadjacent_50":
        return [
            (0.5, 0.0),
            (1.5, 45.0),
            (2.5, 50.0),
        ]
    raise ValueError(f"Preset {preset!r} does not define relative mask points.")


def _normalize_direct_epfd_legacy_edge_mask_points(
    mask_points: Any,
) -> np.ndarray:
    points = np.asarray(mask_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError(
            "custom_mask_points must be a two-column array-like with at least two points."
        )
    if np.any(~np.isfinite(points)):
        raise ValueError("custom_mask_points must be finite.")
    order = np.argsort(points[:, 0], kind="mergesort")
    points = points[order]
    if np.any(points[:, 0] < 0.0):
        raise ValueError("custom_mask_points offsets must be non-negative.")
    if np.any(np.diff(points[:, 0]) <= 0.0):
        raise ValueError("custom_mask_points offsets must be strictly increasing.")
    if float(points[0, 0]) > 0.0:
        points = np.vstack([np.asarray([[0.0, 0.0]], dtype=np.float64), points])
    elif float(points[0, 1]) != 0.0:
        points[0, 1] = 0.0
    return np.asarray(points, dtype=np.float64)


def _legacy_direct_epfd_mask_points_to_signed_centered(
    mask_points: Any,
    *,
    channel_bandwidth_mhz: float,
) -> np.ndarray:
    legacy_points = _normalize_direct_epfd_legacy_edge_mask_points(mask_points)
    half_bandwidth_mhz = 0.5 * float(channel_bandwidth_mhz)
    left_points = np.column_stack(
        [
            -(half_bandwidth_mhz + legacy_points[::-1, 0]),
            legacy_points[::-1, 1],
        ]
    )
    right_points = np.column_stack(
        [
            half_bandwidth_mhz + legacy_points[:, 0],
            legacy_points[:, 1],
        ]
    )
    combined = np.vstack([left_points, right_points])
    return _normalize_direct_epfd_custom_mask_points(
        combined,
        band_halfwidth_mhz=half_bandwidth_mhz,
    )


def _normalize_direct_epfd_custom_mask_points(
    mask_points: Any,
    *,
    band_halfwidth_mhz: float,
) -> np.ndarray:
    halfwidth = float(band_halfwidth_mhz)
    if not np.isfinite(halfwidth) or halfwidth <= 0.0:
        raise ValueError("band_halfwidth_mhz must be finite and > 0.")
    points = np.asarray(mask_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("custom_mask_points must be a two-column array-like.")
    if points.shape[0] < 2:
        raise ValueError("custom_mask_points must contain at least two points.")
    if np.any(~np.isfinite(points)):
        raise ValueError("custom_mask_points must be finite.")
    order = np.argsort(points[:, 0], kind="mergesort")
    points = np.asarray(points[order], dtype=np.float64)
    points[:, 1] = np.clip(points[:, 1], 0.0, None)
    left_mask = points[:, 0] < (-halfwidth - 1.0e-9)
    right_mask = points[:, 0] > (halfwidth + 1.0e-9)
    left_points = np.asarray(points[left_mask], dtype=np.float64)
    right_points = np.asarray(points[right_mask], dtype=np.float64)
    default_dx = max(halfwidth, 0.1)
    if left_points.size == 0:
        left_points = np.asarray([[-(halfwidth + default_dx), 45.0]], dtype=np.float64)
    if right_points.size == 0:
        right_points = np.asarray([[halfwidth + default_dx, 45.0]], dtype=np.float64)
    if np.any(np.diff(left_points[:, 0]) <= 0.0) or np.any(np.diff(right_points[:, 0]) <= 0.0):
        raise ValueError("custom_mask_points offsets must be strictly increasing on each side.")
    anchor_points = np.asarray([[-halfwidth, 0.0], [halfwidth, 0.0]], dtype=np.float64)
    combined = np.vstack([left_points, anchor_points, right_points])
    if np.any(np.diff(combined[:, 0]) <= 0.0):
        raise ValueError(
            "custom_mask_points must keep strictly increasing signed offsets around the fixed in-band anchors."
        )
    return np.asarray(combined, dtype=np.float64)


def _resolve_direct_epfd_receiver_response_mode(mode: str | None) -> str:
    mode_name = str("rectangular" if mode in {None, ""} else mode).strip().lower()
    if mode_name not in {"rectangular", "custom"}:
        raise ValueError(
            "receiver_response_mode must be one of ['custom', 'rectangular']; "
            f"got {mode!r}."
        )
    return str(mode_name)


def _resolve_direct_epfd_mask_points_mhz(
    *,
    preset: str,
    channel_bandwidth_mhz: float,
    custom_mask_points: Any | None,
    ras_frequency_ghz: float = 1.4,
) -> np.ndarray:
    bandwidth = float(channel_bandwidth_mhz)
    half_bandwidth = 0.5 * float(bandwidth)
    if preset == "custom":
        if custom_mask_points is None:
            custom_mask_points = np.asarray(
                [
                    [-(half_bandwidth + bandwidth), 45.0],
                    [-(half_bandwidth + 2.0 * bandwidth), 50.0],
                    [half_bandwidth + bandwidth, 45.0],
                    [half_bandwidth + 2.0 * bandwidth, 50.0],
                ],
                dtype=np.float64,
            )
        return _normalize_direct_epfd_custom_mask_points(
            custom_mask_points,
            band_halfwidth_mhz=half_bandwidth,
        )
    relative = _direct_epfd_relative_mask_points(
        preset=preset,
        ras_frequency_ghz=float(ras_frequency_ghz),
    )
    points: list[tuple[float, float]] = []
    for multiplier, attenuation_db in relative:
        # Multipliers are centre-relative (in units of channel BW).
        signed_offset_mhz = float(multiplier) * bandwidth
        points.append((-signed_offset_mhz, float(attenuation_db)))
        points.append((signed_offset_mhz, float(attenuation_db)))
    return _normalize_direct_epfd_custom_mask_points(
        points,
        band_halfwidth_mhz=half_bandwidth,
    )


def _evaluate_direct_epfd_mask_attenuation_db(
    signed_offset_from_center_mhz: np.ndarray,
    *,
    mask_points_mhz: np.ndarray,
) -> np.ndarray:
    separation = np.asarray(signed_offset_from_center_mhz, dtype=np.float64)
    offsets = np.asarray(mask_points_mhz[:, 0], dtype=np.float64)
    attenuations = np.asarray(mask_points_mhz[:, 1], dtype=np.float64)
    return np.interp(
        separation,
        offsets,
        attenuations,
        left=float(attenuations[0]),
        right=float(attenuations[-1]),
    )


def _resolve_direct_epfd_receiver_response_points_mhz(
    *,
    response_mode: str,
    receiver_bandwidth_mhz: float,
    custom_mask_points: Any | None,
) -> np.ndarray:
    bandwidth = float(receiver_bandwidth_mhz)
    half_bandwidth = 0.5 * bandwidth
    if response_mode == "custom":
        if custom_mask_points is None:
            custom_mask_points = np.asarray(
                [
                    [-(half_bandwidth + bandwidth), 60.0],
                    [half_bandwidth + bandwidth, 60.0],
                ],
                dtype=np.float64,
            )
        return _normalize_direct_epfd_custom_mask_points(
            custom_mask_points,
            band_halfwidth_mhz=half_bandwidth,
        )
    return np.asarray(
        [
            [-(half_bandwidth + max(half_bandwidth, 0.1)), 120.0],
            [-half_bandwidth, 0.0],
            [half_bandwidth, 0.0],
            [half_bandwidth + max(half_bandwidth, 0.1), 120.0],
        ],
        dtype=np.float64,
    )


def _evaluate_direct_epfd_receiver_response_attenuation_db(
    signed_offset_from_center_mhz: np.ndarray,
    *,
    response_mode: str,
    receiver_bandwidth_mhz: float,
    response_points_mhz: np.ndarray | None,
) -> np.ndarray:
    offsets = np.asarray(signed_offset_from_center_mhz, dtype=np.float64)
    half_bandwidth = 0.5 * float(receiver_bandwidth_mhz)
    if response_mode == "rectangular":
        attenuation = np.full(offsets.shape, np.inf, dtype=np.float64)
        attenuation[np.abs(offsets) <= half_bandwidth + 1.0e-12] = 0.0
        return attenuation
    if response_points_mhz is None:
        raise ValueError("Custom receiver response requires receiver response points.")
    return _evaluate_direct_epfd_mask_attenuation_db(
        offsets,
        mask_points_mhz=np.asarray(response_points_mhz, dtype=np.float64),
    )


def _integrate_direct_epfd_log_linear_segment(
    *,
    segment_start_mhz: float,
    segment_stop_mhz: float,
    attenuation_start_db: float,
    attenuation_stop_db: float,
) -> float:
    width_mhz = float(segment_stop_mhz - segment_start_mhz)
    if width_mhz <= 0.0:
        return 0.0
    if not (np.isfinite(attenuation_start_db) and np.isfinite(attenuation_stop_db)):
        return 0.0
    start_linear = float(10.0 ** (-float(attenuation_start_db) / 10.0))
    exponent_delta = float(
        -np.log(10.0) * (float(attenuation_stop_db) - float(attenuation_start_db)) / 10.0
    )
    if abs(exponent_delta) <= 1.0e-12:
        return float(width_mhz * start_linear)
    return float(width_mhz * start_linear * (np.expm1(exponent_delta) / exponent_delta))


def _integrate_direct_epfd_channel_leakage_fraction(
    *,
    channel_start_mhz: float,
    channel_stop_mhz: float,
    ras_start_mhz: float,
    ras_stop_mhz: float,
    channel_bandwidth_mhz: float,
    mask_points_mhz: np.ndarray,
    integration_cutoff_mhz: float,
    receiver_response_mode: str = "rectangular",
    receiver_response_points_mhz: np.ndarray | None = None,
) -> float:
    channel_start = float(channel_start_mhz)
    channel_stop = float(channel_stop_mhz)
    ras_start = float(ras_start_mhz)
    ras_stop = float(ras_stop_mhz)
    bandwidth = float(channel_bandwidth_mhz)
    cutoff = max(0.0, float(integration_cutoff_mhz))
    if ras_stop <= ras_start or channel_stop <= channel_start or bandwidth <= 0.0:
        return 0.0
    channel_center = 0.5 * (channel_start + channel_stop)
    ras_center = 0.5 * (ras_start + ras_stop)
    ras_bandwidth = float(ras_stop - ras_start)
    breakpoints = {
        float(ras_start),
        float(ras_stop),
        float(channel_start),
        float(channel_stop),
        float(channel_center - cutoff),
        float(channel_center + cutoff),
    }
    for offset_mhz in np.asarray(mask_points_mhz[:, 0], dtype=np.float64):
        breakpoints.add(float(channel_center + offset_mhz))
    if receiver_response_mode == "custom" and receiver_response_points_mhz is not None:
        for offset_mhz in np.asarray(receiver_response_points_mhz[:, 0], dtype=np.float64):
            breakpoints.add(float(ras_center + offset_mhz))
    left_bound = min(float(value) for value in breakpoints)
    right_bound = max(float(value) for value in breakpoints)
    sorted_points = np.asarray(
        [
            point
            for point in sorted(breakpoints)
            if point >= left_bound - 1.0e-9 and point <= right_bound + 1.0e-9
        ],
        dtype=np.float64,
    )
    if sorted_points.size < 2:
        return 0.0
    leakage_integral = 0.0
    cutoff_start = float(channel_center - cutoff)
    cutoff_stop = float(channel_center + cutoff)
    for segment_start, segment_stop in zip(sorted_points[:-1], sorted_points[1:]):
        integration_start = max(float(segment_start), cutoff_start)
        integration_stop = min(float(segment_stop), cutoff_stop)
        if integration_stop <= integration_start:
            continue
        edge_positions_mhz = np.asarray(
            [integration_start, integration_stop],
            dtype=np.float64,
        )
        tx_attenuation_db = _evaluate_direct_epfd_mask_attenuation_db(
            edge_positions_mhz - channel_center,
            mask_points_mhz=mask_points_mhz,
        )
        if receiver_response_mode == "rectangular":
            ras_overlap_start = max(integration_start, float(ras_start))
            ras_overlap_stop = min(integration_stop, float(ras_stop))
            if ras_overlap_stop <= ras_overlap_start:
                continue
            edge_positions_mhz = np.asarray(
                [ras_overlap_start, ras_overlap_stop],
                dtype=np.float64,
            )
            tx_attenuation_db = _evaluate_direct_epfd_mask_attenuation_db(
                edge_positions_mhz - channel_center,
                mask_points_mhz=mask_points_mhz,
            )
            leakage_integral += _integrate_direct_epfd_log_linear_segment(
                segment_start_mhz=float(edge_positions_mhz[0]),
                segment_stop_mhz=float(edge_positions_mhz[1]),
                attenuation_start_db=float(tx_attenuation_db[0]),
                attenuation_stop_db=float(tx_attenuation_db[1]),
            )
            continue
        rx_attenuation_db = _evaluate_direct_epfd_receiver_response_attenuation_db(
            edge_positions_mhz - ras_center,
            response_mode=str(receiver_response_mode),
            receiver_bandwidth_mhz=ras_bandwidth,
            response_points_mhz=receiver_response_points_mhz,
        )
        if not np.all(np.isfinite(rx_attenuation_db)):
            continue
        total_attenuation_db = np.asarray(tx_attenuation_db, dtype=np.float64) + np.asarray(
            rx_attenuation_db,
            dtype=np.float64,
        )
        leakage_integral += _integrate_direct_epfd_log_linear_segment(
            segment_start_mhz=float(edge_positions_mhz[0]),
            segment_stop_mhz=float(edge_positions_mhz[1]),
            attenuation_start_db=float(total_attenuation_db[0]),
            attenuation_stop_db=float(total_attenuation_db[1]),
        )
    return float(max(0.0, leakage_integral / bandwidth))


def _estimate_direct_epfd_spectrum_context_bytes(
    *,
    cell_count: int,
    full_channel_count: int,
    reuse_factor: int,
    groups_per_cell: int,
    tx_reference_count: int,
    ras_reference_count: int,
    mask_point_count: int,
    receiver_mask_point_count: int = 0,
) -> int:
    return int(
        max(1, int(cell_count)) * (4 + 4)
        + max(1, int(cell_count)) * max(1, int(groups_per_cell)) * (4 + 1)
        + max(1, int(cell_count)) * 4
        + max(1, int(full_channel_count)) * (8 + 8)
        + max(1, int(reuse_factor)) * max(1, int(groups_per_cell)) * (4 + 4)
        + max(1, int(tx_reference_count) + int(ras_reference_count)) * 8
        + max(1, int(mask_point_count) + int(receiver_mask_point_count)) * (8 + 8)
    )


def _normalize_direct_epfd_channel_index_list(
    values: Any,
    *,
    full_channel_count: int,
) -> list[int]:
    """Return sorted unique service-channel indices clipped to the valid band."""
    if values is None:
        return []
    if isinstance(values, str):
        if values.strip() == "":
            return []
        raw_values: list[Any] = [values]
    elif isinstance(values, np.ndarray):
        raw_values = np.asarray(values).reshape(-1).tolist()
    elif np.isscalar(values):
        raw_values = [values]
    else:
        raw_values = list(values)
    normalized: set[int] = set()
    for raw_value in raw_values:
        channel_index = int(raw_value)
        if 0 <= channel_index < int(full_channel_count):
            normalized.add(channel_index)
    return sorted(normalized)


def _direct_epfd_optional_scalar_missing(value: Any) -> bool:
    """Return True when one optional spectrum payload value is semantically blank."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, np.ndarray):
        return int(np.asarray(value).size) == 0
    return False


def _resolve_direct_epfd_enabled_channel_indices(
    *,
    spectrum_plan: Mapping[str, Any],
    full_channel_count: int,
    reuse_factor: int,
    max_groups_per_cell: int,
) -> tuple[list[int], int]:
    """Resolve the explicit or legacy-enabled channel subset for one band plan."""
    explicit_enabled = spectrum_plan.get("enabled_channel_indices", None)
    explicit_enabled_missing = explicit_enabled is None or (
        isinstance(explicit_enabled, str) and explicit_enabled.strip() == ""
    )
    if "enabled_channel_indices" in spectrum_plan and not explicit_enabled_missing:
        return (
            _normalize_direct_epfd_channel_index_list(
                explicit_enabled,
                full_channel_count=full_channel_count,
            ),
            int(max_groups_per_cell),
        )
    if "enabled_channel_indices" in spectrum_plan and explicit_enabled_missing:
        return [int(channel_index) for channel_index in range(int(full_channel_count))], int(
            max_groups_per_cell
        )

    explicit_disabled = spectrum_plan.get("disabled_channel_indices", None)
    if "disabled_channel_indices" in spectrum_plan:
        disabled_indices = set(
            _normalize_direct_epfd_channel_index_list(
                explicit_disabled,
                full_channel_count=full_channel_count,
            )
        )
        enabled_indices = [
            int(channel_index)
            for channel_index in range(int(full_channel_count))
            if int(channel_index) not in disabled_indices
        ]
        return enabled_indices, int(max_groups_per_cell)

    groups_cap_raw = spectrum_plan.get("channel_groups_per_cell_cap", 1)
    groups_cap = (
        int(max_groups_per_cell)
        if _direct_epfd_optional_scalar_missing(groups_cap_raw)
        else int(groups_cap_raw)
    )
    if groups_cap < 1:
        raise ValueError("channel_groups_per_cell_cap must be >= 1.")
    groups_per_cell = max(1, min(int(groups_cap), int(max_groups_per_cell)))
    enabled_indices = [
        int(slot_id + reuse_factor * group_index)
        for slot_id in range(int(reuse_factor))
        for group_index in range(int(groups_per_cell))
        if int(slot_id + reuse_factor * group_index) < int(full_channel_count)
    ]
    return sorted(enabled_indices), int(groups_cap)


# Bounded memoization cache for normalize_direct_epfd_spectrum_plan.
# The function is pure w.r.t. its kwargs. GUI status refreshes call it
# thousands of times per run with stable per-system inputs; caching
# collapses that into O(n_distinct_plans) real evaluations.
_DIRECT_EPFD_SPECTRUM_PLAN_CACHE: "dict[tuple, dict[str, Any]]" = {}
_DIRECT_EPFD_SPECTRUM_PLAN_CACHE_MAX = 256


def _direct_epfd_spectrum_plan_hashable(value: Any) -> Any:
    """Best-effort convert an input into a stable hashable representation.

    Dict-like mappings become sorted tuples of (key, hashable(value)).
    Sequences become tuples. numpy arrays become tuples of their flat
    contents. Everything else is returned as-is (and TypeError at the
    call site signals uncacheable).
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return ("__ndarray__", value.shape, value.dtype.str,
                tuple(value.reshape(-1).tolist()))
    if isinstance(value, Mapping):
        return tuple(
            (str(k), _direct_epfd_spectrum_plan_hashable(v))
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_direct_epfd_spectrum_plan_hashable(v) for v in value)
    # Numpy scalars / other — fall back to str() which is deterministic.
    return ("__repr__", repr(value))


def normalize_direct_epfd_spectrum_plan(
    *,
    spectrum_plan: Mapping[str, Any] | None,
    channel_bandwidth_mhz: float,
    split_total_group_denominator_mode: str | None = None,
    active_cell_count: int | None = None,
    active_cell_reuse_slot_ids: Any | None = None,
) -> dict[str, Any] | None:
    """
    Normalize spectrum/reuse inputs into one GPU-ready spectrum plan.

    Parameters
    ----------
    spectrum_plan : Mapping[str, Any] or None
        Raw spectrum-plan payload. ``None`` preserves the historical single-
        channel full-overlap behavior. New payloads should provide either
        ``enabled_channel_indices`` or ``disabled_channel_indices`` to describe
        the explicit service-channel subset. Legacy payloads that only provide
        ``channel_groups_per_cell_cap`` still normalize to the historical
        bottom-up channel selection.
    channel_bandwidth_mhz : float
        Channel bandwidth in MHz. This remains the single source of truth for
        slot width and per-MHz/per-channel conversion semantics.
    active_cell_count : int or None, optional
        Expected ACTIVE-axis cell count used to validate or synthesize reuse
        slot ids when they are not supplied explicitly.
    split_total_group_denominator_mode : str or None, optional
        Default denominator semantics used when
        ``multi_group_power_policy="split_total_cell_power"`` and the runtime
        later applies per-channel activity.
    active_cell_reuse_slot_ids : Any or None, optional
        Optional ACTIVE-axis reuse-slot ids. When omitted, the helper falls
        back to slot ``0`` for every cell, matching the historical ``F1``
        assumption.

    Returns
    -------
    dict[str, Any] or None
        Normalized plan dictionary ready for HDF5 attrs, GUI summaries, and
        the GPU spectrum context. ``None`` means "use the legacy full-overlap
        single-channel semantics."
    """
    if spectrum_plan is None:
        return None

    # Memoization: this function is pure w.r.t. its inputs and is called many
    # times per batch by GUI status refreshes (10,000+ calls/run, ~12s cumtime
    # at realistic scale). Cache by a stable key derived from the hashable
    # representation of the inputs. On cache hit we return a shallow copy so
    # callers may not mutate the cached dict in-place (numpy arrays inside
    # are still shared — all downstream users re-wrap with np.asarray(...,
    # dtype=...) before GPU upload, so sharing is safe).
    try:
        _cache_key = (
            _direct_epfd_spectrum_plan_hashable(spectrum_plan),
            float(channel_bandwidth_mhz),
            (str(split_total_group_denominator_mode)
             if split_total_group_denominator_mode is not None else None),
            (int(active_cell_count) if active_cell_count is not None else None),
            _direct_epfd_spectrum_plan_hashable(active_cell_reuse_slot_ids),
        )
    except Exception:
        _cache_key = None  # Unhashable input — fall through to full compute.
    if _cache_key is not None:
        _cached = _DIRECT_EPFD_SPECTRUM_PLAN_CACHE.get(_cache_key)
        if _cached is not None:
            return dict(_cached)

    plan = dict(spectrum_plan)
    channel_bandwidth = float(channel_bandwidth_mhz)
    if not np.isfinite(channel_bandwidth) or channel_bandwidth <= 0.0:
        raise ValueError("channel_bandwidth_mhz must be finite and > 0.")

    service_start_mhz = float(plan.get("service_band_start_mhz", 2620.0))
    service_stop_mhz = float(plan.get("service_band_stop_mhz", 2690.0))
    ras_start_mhz = float(plan.get("ras_receiver_band_start_mhz", 2690.0))
    ras_stop_mhz = float(plan.get("ras_receiver_band_stop_mhz", 2700.0))
    if not (np.isfinite(service_start_mhz) and np.isfinite(service_stop_mhz) and service_stop_mhz > service_start_mhz):
        raise ValueError("service band must satisfy stop > start.")
    if not (np.isfinite(ras_start_mhz) and np.isfinite(ras_stop_mhz) and ras_stop_mhz > ras_start_mhz):
        raise ValueError("RAS receiver band must satisfy stop > start.")

    reuse_factor = _normalize_direct_epfd_reuse_factor(plan.get("reuse_factor", 1))
    service_bandwidth_total_mhz = float(service_stop_mhz - service_start_mhz)
    full_channel_count = int(np.floor(service_bandwidth_total_mhz / channel_bandwidth))
    if full_channel_count < 1:
        raise ValueError(
            "The service band does not contain one full channel at the configured channel bandwidth."
        )
    leftover_spectrum_mhz = float(service_bandwidth_total_mhz - full_channel_count * channel_bandwidth)
    max_groups_per_cell = max(1, int(full_channel_count // reuse_factor))
    enabled_channel_indices, groups_cap = _resolve_direct_epfd_enabled_channel_indices(
        spectrum_plan=plan,
        full_channel_count=int(full_channel_count),
        reuse_factor=int(reuse_factor),
        max_groups_per_cell=int(max_groups_per_cell),
    )
    _anchor_raw = plan.get("ras_anchor_reuse_slot", 0)
    # A None anchor-slot (e.g. from a UEMR config where the field is hidden)
    # is treated as slot 0 — the spectrum planner still needs a concrete
    # integer here, and UEMR never consults the reuse scheme downstream.
    anchor_slot = (int(_anchor_raw) if _anchor_raw is not None else 0) % int(reuse_factor)
    power_policy = _normalize_direct_epfd_power_policy(plan.get("multi_group_power_policy"))
    denominator_mode_raw = plan.get("split_total_group_denominator_mode")
    if (
        split_total_group_denominator_mode is not None
        and denominator_mode_raw not in {None, ""}
        and str(split_total_group_denominator_mode).strip().lower()
        != str(denominator_mode_raw).strip().lower()
    ):
        raise ValueError(
            "split_total_group_denominator_mode conflicts with the value supplied in spectrum_plan."
        )
    denominator_mode = _normalize_direct_epfd_split_group_denominator_mode(
        denominator_mode_raw
        if denominator_mode_raw not in {None, ""}
        else split_total_group_denominator_mode
    )
    tx_reference_mode = _normalize_direct_epfd_reference_mode(
        plan.get("tx_reference_mode"),
        default="middle",
    )
    ras_reference_mode = _normalize_direct_epfd_reference_mode(
        plan.get("ras_reference_mode"),
        default="lower",
    )
    tx_reference_point_count_raw = plan.get("tx_reference_point_count")
    ras_reference_point_count_raw = plan.get("ras_reference_point_count")
    tx_reference_point_count = max(
        2,
        int(3 if tx_reference_point_count_raw in {None, ""} else tx_reference_point_count_raw),
    )
    ras_reference_point_count = max(
        2,
        int(3 if ras_reference_point_count_raw in {None, ""} else ras_reference_point_count_raw),
    )
    tx_reference_frequencies_mhz = _resolve_direct_epfd_reference_frequencies_mhz(
        start_mhz=service_start_mhz,
        stop_mhz=service_stop_mhz,
        mode=tx_reference_mode,
        point_count=tx_reference_point_count,
    )
    ras_reference_frequencies_mhz = _resolve_direct_epfd_reference_frequencies_mhz(
        start_mhz=ras_start_mhz,
        stop_mhz=ras_stop_mhz,
        mode=ras_reference_mode,
        point_count=ras_reference_point_count,
    )
    cutoff_basis = _normalize_direct_epfd_cutoff_basis(
        plan.get("spectral_integration_cutoff_basis")
    )
    cutoff_percent = float(plan.get("spectral_integration_cutoff_percent", 250.0))
    if not np.isfinite(cutoff_percent) or cutoff_percent <= 0.0:
        raise ValueError("spectral_integration_cutoff_percent must be finite and > 0.")
    cutoff_span_mhz = (
        channel_bandwidth if cutoff_basis == "channel_bandwidth" else service_bandwidth_total_mhz
    )
    integration_cutoff_mhz = float(cutoff_span_mhz * cutoff_percent / 100.0)
    mask_preset = _normalize_direct_epfd_mask_preset(
        plan.get("unwanted_emission_mask_preset")
    )
    ras_centre_ghz = 0.5e-3 * (ras_start_mhz + ras_stop_mhz)
    mask_points_mhz = _resolve_direct_epfd_mask_points_mhz(
        preset=mask_preset,
        channel_bandwidth_mhz=channel_bandwidth,
        custom_mask_points=plan.get("custom_mask_points"),
        ras_frequency_ghz=float(ras_centre_ghz),
    )
    receiver_response_mode = _resolve_direct_epfd_receiver_response_mode(
        plan.get("receiver_response_mode")
    )
    receiver_response_points_mhz = _resolve_direct_epfd_receiver_response_points_mhz(
        response_mode=receiver_response_mode,
        receiver_bandwidth_mhz=float(ras_stop_mhz - ras_start_mhz),
        custom_mask_points=plan.get("receiver_custom_mask_points"),
    )
    slot_edges_mhz = service_start_mhz + np.arange(full_channel_count + 1, dtype=np.float64) * channel_bandwidth
    slot_centers_mhz = 0.5 * (slot_edges_mhz[:-1] + slot_edges_mhz[1:])

    if active_cell_reuse_slot_ids is None:
        if active_cell_count is None:
            slot_ids = np.empty((0,), dtype=np.int32)
        else:
            slot_ids = np.zeros((int(active_cell_count),), dtype=np.int32)
    else:
        slot_ids = np.asarray(active_cell_reuse_slot_ids, dtype=np.int32).reshape(-1)
        if active_cell_count is not None and int(slot_ids.size) != int(active_cell_count):
            raise ValueError(
                "active_cell_reuse_slot_ids must align with the ACTIVE cell axis."
            )
    if slot_ids.size:
        slot_ids = np.mod(slot_ids, np.int32(int(reuse_factor))).astype(np.int32, copy=False)

    enabled_channels_by_slot: list[list[int]] = [[] for _ in range(int(reuse_factor))]
    for channel_index in enabled_channel_indices:
        enabled_channels_by_slot[int(channel_index % reuse_factor)].append(int(channel_index))
    groups_per_cell = max(
        1,
        max((len(channel_indices) for channel_indices in enabled_channels_by_slot), default=0),
    )

    slot_group_channel_indices = np.full(
        (int(reuse_factor), int(groups_per_cell)),
        fill_value=-1,
        dtype=np.int32,
    )
    slot_group_valid_mask = np.zeros(
        (int(reuse_factor), int(groups_per_cell)),
        dtype=np.bool_,
    )
    slot_group_leakage_factors = np.zeros(
        (int(reuse_factor), int(groups_per_cell)),
        dtype=np.float32,
    )
    slot_leakage_factors: dict[int, float] = {}
    for slot_id in range(int(reuse_factor)):
        occupied_slots = enabled_channels_by_slot[int(slot_id)]
        for group_idx, channel_index in enumerate(occupied_slots):
            slot_group_channel_indices[int(slot_id), int(group_idx)] = np.int32(channel_index)
            slot_group_valid_mask[int(slot_id), int(group_idx)] = np.bool_(True)
        if not occupied_slots:
            slot_leakage_factors[int(slot_id)] = 0.0
            continue
        leakage_terms: list[float] = []
        for group_idx, channel_index in enumerate(occupied_slots):
            leakage_value = _integrate_direct_epfd_channel_leakage_fraction(
                channel_start_mhz=float(slot_edges_mhz[channel_index]),
                channel_stop_mhz=float(slot_edges_mhz[channel_index + 1]),
                ras_start_mhz=ras_start_mhz,
                ras_stop_mhz=ras_stop_mhz,
                channel_bandwidth_mhz=channel_bandwidth,
                mask_points_mhz=mask_points_mhz,
                integration_cutoff_mhz=integration_cutoff_mhz,
                receiver_response_mode=receiver_response_mode,
                receiver_response_points_mhz=receiver_response_points_mhz,
            )
            leakage_terms.append(float(leakage_value))
            slot_group_leakage_factors[int(slot_id), int(group_idx)] = np.float32(
                leakage_value
            )
        if power_policy == "split_total_cell_power":
            slot_leakage_factors[int(slot_id)] = float(np.mean(leakage_terms))
        else:
            slot_leakage_factors[int(slot_id)] = float(np.sum(leakage_terms))
    if slot_ids.size:
        cell_leakage_factors = np.asarray(
            [slot_leakage_factors[int(slot_id)] for slot_id in slot_ids.tolist()],
            dtype=np.float32,
        )
    else:
        cell_leakage_factors = np.empty((0,), dtype=np.float32)
    cell_group_leakage_factors = (
        np.asarray(slot_group_leakage_factors[slot_ids], dtype=np.float32)
        if slot_ids.size
        else np.empty((0, int(groups_per_cell)), dtype=np.float32)
    )
    cell_group_valid_mask = (
        np.asarray(slot_group_valid_mask[slot_ids], dtype=np.bool_)
        if slot_ids.size
        else np.empty((0, int(groups_per_cell)), dtype=np.bool_)
    )
    configured_group_counts_per_cell = (
        np.sum(cell_group_valid_mask, axis=1, dtype=np.int32)
        if slot_ids.size
        else np.empty((0,), dtype=np.int32)
    )

    zero_leftover_reuse_factors = tuple(
        int(candidate)
        for candidate in _DIRECT_EPFD_SUPPORTED_REUSE_FACTORS
        if leftover_spectrum_mhz <= 1.0e-9 and full_channel_count % int(candidate) == 0
    )
    context_bytes = _estimate_direct_epfd_spectrum_context_bytes(
        cell_count=int(slot_ids.size),
        full_channel_count=int(full_channel_count),
        reuse_factor=int(reuse_factor),
        groups_per_cell=int(groups_per_cell),
        tx_reference_count=int(tx_reference_frequencies_mhz.size),
        ras_reference_count=int(ras_reference_frequencies_mhz.size),
        mask_point_count=int(mask_points_mhz.shape[0]),
        receiver_mask_point_count=int(receiver_response_points_mhz.shape[0]),
    )
    _result = {
        "service_band_start_mhz": float(service_start_mhz),
        "service_band_stop_mhz": float(service_stop_mhz),
        "service_bandwidth_total_mhz": float(service_bandwidth_total_mhz),
        "channel_bandwidth_mhz": float(channel_bandwidth),
        "full_channel_count": int(full_channel_count),
        "leftover_spectrum_mhz": float(leftover_spectrum_mhz),
        "leftover_channel_fraction": float(leftover_spectrum_mhz / channel_bandwidth),
        "reuse_factor": int(reuse_factor),
        "zero_leftover_reuse_factors": tuple(zero_leftover_reuse_factors),
        "channel_groups_per_cell_cap": int(groups_cap),
        "channel_groups_per_cell": int(groups_per_cell),
        "max_groups_per_cell": int(max_groups_per_cell),
        "enabled_channel_indices": np.asarray(enabled_channel_indices, dtype=np.int32),
        "enabled_channel_count": int(len(enabled_channel_indices)),
        "ras_anchor_reuse_slot": int(anchor_slot),
        "multi_group_power_policy": str(power_policy),
        "split_total_group_denominator_mode": str(denominator_mode),
        "ras_receiver_band_start_mhz": float(ras_start_mhz),
        "ras_receiver_band_stop_mhz": float(ras_stop_mhz),
        "ras_receiver_bandwidth_mhz": float(ras_stop_mhz - ras_start_mhz),
        "receiver_response_mode": str(receiver_response_mode),
        "receiver_custom_mask_points": (
            None
            if receiver_response_mode != "custom"
            else np.asarray(receiver_response_points_mhz, dtype=np.float64).tolist()
        ),
        "receiver_response_points_mhz": np.asarray(
            receiver_response_points_mhz,
            dtype=np.float64,
        ),
        "unwanted_emission_mask_preset": str(mask_preset),
        "custom_mask_points": None if mask_preset != "custom" else mask_points_mhz.tolist(),
        "unwanted_emission_mask_points_mhz": np.asarray(mask_points_mhz, dtype=np.float64),
        "spectral_integration_cutoff_basis": str(cutoff_basis),
        "spectral_integration_cutoff_percent": float(cutoff_percent),
        "spectral_integration_cutoff_mhz": float(integration_cutoff_mhz),
        "tx_reference_mode": str(tx_reference_mode),
        "tx_reference_point_count": int(tx_reference_frequencies_mhz.size),
        "tx_reference_frequencies_mhz": np.asarray(tx_reference_frequencies_mhz, dtype=np.float64),
        "tx_reference_frequency_mhz_effective": float(np.mean(tx_reference_frequencies_mhz)),
        "ras_reference_mode": str(ras_reference_mode),
        "ras_reference_point_count": int(ras_reference_frequencies_mhz.size),
        "ras_reference_frequencies_mhz": np.asarray(ras_reference_frequencies_mhz, dtype=np.float64),
        "ras_reference_frequency_mhz_effective": float(np.mean(ras_reference_frequencies_mhz)),
        "slot_edges_mhz": np.asarray(slot_edges_mhz, dtype=np.float64),
        "slot_centers_mhz": np.asarray(slot_centers_mhz, dtype=np.float64),
        "slot_group_channel_indices": np.asarray(
            slot_group_channel_indices,
            dtype=np.int32,
        ),
        "slot_group_valid_mask": np.asarray(
            slot_group_valid_mask,
            dtype=np.bool_,
        ),
        "slot_group_leakage_factors": np.asarray(
            slot_group_leakage_factors,
            dtype=np.float32,
        ),
        "active_cell_reuse_slot_ids": np.asarray(slot_ids, dtype=np.int32),
        "cell_leakage_factors": np.asarray(cell_leakage_factors, dtype=np.float32),
        "cell_group_valid_mask": np.asarray(
            cell_group_valid_mask,
            dtype=np.bool_,
        ),
        "configured_group_counts_per_cell": np.asarray(
            configured_group_counts_per_cell,
            dtype=np.int32,
        ),
        "cell_group_leakage_factors": np.asarray(
            cell_group_leakage_factors,
            dtype=np.float32,
        ),
        "slot_leakage_factors": {
            int(slot_id): float(value)
            for slot_id, value in sorted(slot_leakage_factors.items())
        },
        "spectral_slab": 1,
        "spectrum_context_bytes": int(context_bytes),
    }
    if _cache_key is not None:
        # Bounded cache — evict oldest when exceeding cap to prevent unbounded
        # growth during long-lived GUI sessions.
        if len(_DIRECT_EPFD_SPECTRUM_PLAN_CACHE) >= _DIRECT_EPFD_SPECTRUM_PLAN_CACHE_MAX:
            try:
                _oldest_key = next(iter(_DIRECT_EPFD_SPECTRUM_PLAN_CACHE))
                _DIRECT_EPFD_SPECTRUM_PLAN_CACHE.pop(_oldest_key, None)
            except StopIteration:
                pass
        _DIRECT_EPFD_SPECTRUM_PLAN_CACHE[_cache_key] = _result
    return dict(_result)

# Absolute lower bound for dynamic histogram edges (dB).  Values below this
# are physically meaningless — even a single photon in the receiver bandwidth
# is above -500 dBW/m².  Clamping here prevents sliding-window integration
# (which can produce near-zero linear averages) from blowing the histogram
# out to -3000 dB and breaking CCDF plots.
_HISTOGRAM_DB_FLOOR: float = -500.0


def _align_down_to_step(value: float, step: float) -> float:
    return float(np.floor(float(value) / float(step)) * float(step))


def _align_up_to_step(value: float, step: float) -> float:
    return float(np.ceil(float(value) / float(step)) * float(step))


_OUTPUT_FAMILY_NAMES: tuple[str, ...] = (
    "prx_total_distribution",
    "epfd_distribution",
    "total_pfd_ras_distribution",
    "per_satellite_pfd_distribution",
    "prx_elevation_heatmap",
    "per_satellite_pfd_elevation_heatmap",
    "beam_statistics",
)
_OUTPUT_FAMILY_MODES = frozenset({"none", "raw", "preaccumulated", "both"})
_DEFAULT_OUTPUT_FAMILY_CONFIGS: dict[str, dict[str, Any]] = {
    "prx_total_distribution": {
        "mode": "both",
        "range_mode": "dynamic",
        "min_db": None,
        "max_db": None,
        "bin_step_db": 0.02,
        "margin_bins": 16,
        "bandwidth_mhz": 5.0,
    },
    "epfd_distribution": {
        "mode": "both",
        "range_mode": "dynamic",
        "min_db": None,
        "max_db": None,
        "bin_step_db": 0.02,
        "margin_bins": 16,
    },
    "total_pfd_ras_distribution": {
        "mode": "both",
        "range_mode": "dynamic",
        "min_db": None,
        "max_db": None,
        "bin_step_db": 0.02,
        "margin_bins": 16,
    },
    "per_satellite_pfd_distribution": {
        "mode": "both",
        "range_mode": "dynamic",
        "min_db": None,
        "max_db": None,
        "bin_step_db": 0.02,
        "margin_bins": 16,
    },
    "prx_elevation_heatmap": {
        "mode": "preaccumulated",
        "range_mode": "dynamic",
        "min_db": None,
        "max_db": None,
        "value_bin_step_db": 0.2,
        "value_margin_bins": 16,
        "elevation_bin_step_deg": 0.2,
        "bandwidth_mhz": 5.0,
        "sky_slab": 16,
    },
    "per_satellite_pfd_elevation_heatmap": {
        "mode": "both",
        "range_mode": "dynamic",
        "min_db": None,
        "max_db": None,
        "value_bin_step_db": 0.2,
        "value_margin_bins": 16,
        "elevation_bin_step_deg": 0.2,
        "sky_slab": 16,
    },
    "beam_statistics": {
        "mode": "both",
    },
}
_OUTPUT_FAMILY_RANGE_MODES = {"dynamic", "fixed"}


def default_output_families() -> dict[str, dict[str, Any]]:
    """Return the canonical public family-mode defaults."""
    return {
        str(name): {str(key): value for key, value in config.items()}
        for name, config in _DEFAULT_OUTPUT_FAMILY_CONFIGS.items()
    }


def normalize_output_family_configs(
    output_families: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return canonical output-family configs merged onto the public defaults.

    Parameters
    ----------
    output_families : mapping or None
        Caller-provided per-family overrides.
    ignore_unknown : bool, optional
        When true, silently skip unknown family names instead of raising.
        This is useful for GUI-facing config cleanup paths that prefer to keep
        loading partially stale configs and then normalize them back onto the
        currently supported family set.
    """
    if not ignore_unknown:
        return _normalize_output_family_configs(output_families)

    normalized = default_output_families()
    if output_families is None:
        return normalized

    for family_name, overrides in output_families.items():
        family_key = str(family_name)
        if family_key not in normalized:
            continue
        if overrides is None:
            continue
        if isinstance(overrides, str):
            normalized[family_key]["mode"] = _normalize_output_family_mode(
                overrides,
                family_name=family_key,
            )
            continue
        if not isinstance(overrides, Mapping):
            raise TypeError(
                f"output_families[{family_key!r}] must be a mapping, string mode, or None."
            )
        merged = dict(normalized[family_key])
        for key, value in overrides.items():
            merged[str(key)] = value
        merged["mode"] = _normalize_output_family_mode(
            merged.get("mode", normalized[family_key]["mode"]),
            family_name=family_key,
        )
        normalized[family_key] = merged
    return normalized


def validate_output_family_configs(
    output_families: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> tuple[bool, str]:
    """Validate output-family configs and return a GUI-friendly status tuple."""
    try:
        configs = normalize_output_family_configs(
            output_families,
            ignore_unknown=ignore_unknown,
        )
    except (TypeError, ValueError) as exc:
        message = str(exc)
        mode_match = re.search(r"output_families\[(?P<family>.+?)\]\['mode'\]", message)
        if mode_match is not None:
            family_name = mode_match.group("family").strip("'\"")
            return False, f"Output family mode is unsupported for {family_name!r}."
        return False, message

    for family_name, config in configs.items():
        range_mode = config.get("range_mode")
        if range_mode is not None and str(range_mode) not in _OUTPUT_FAMILY_RANGE_MODES:
            return False, f"Output family range mode is unsupported for {family_name!r}."
    return True, ""


def _normalize_output_family_mode(mode: Any, *, family_name: str) -> str:
    mode_name = str(mode).strip().lower()
    if mode_name not in _OUTPUT_FAMILY_MODES:
        raise ValueError(
            f"output_families[{family_name!r}]['mode'] must be one of "
            f"{sorted(_OUTPUT_FAMILY_MODES)}, got {mode!r}."
        )
    return mode_name


def _mode_includes_raw(mode: str) -> bool:
    return str(mode) in {"raw", "both"}


def _mode_includes_preaccumulated(mode: str) -> bool:
    return str(mode) in {"preaccumulated", "both"}


def _normalize_output_family_configs(
    output_families: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Return normalized per-family configs merged onto the public defaults."""
    normalized = default_output_families()
    if output_families is None:
        return normalized

    for family_name, overrides in output_families.items():
        family_key = str(family_name)
        if family_key not in normalized:
            raise ValueError(
                f"Unknown output family {family_key!r}. "
                f"Expected one of {list(_OUTPUT_FAMILY_NAMES)!r}."
            )
        if overrides is None:
            continue
        if isinstance(overrides, str):
            normalized[family_key]["mode"] = _normalize_output_family_mode(
                overrides,
                family_name=family_key,
            )
            continue
        if not isinstance(overrides, Mapping):
            raise TypeError(
                f"output_families[{family_key!r}] must be a mapping, string mode, or None."
            )
        merged = dict(normalized[family_key])
        for key, value in overrides.items():
            merged[str(key)] = value
        merged["mode"] = _normalize_output_family_mode(
            merged.get("mode", normalized[family_key]["mode"]),
            family_name=family_key,
        )
        normalized[family_key] = merged
    return normalized


def _resolve_output_family_configs(
    *,
    output_families: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Resolve canonical output-family configs for direct-EPFD runs."""
    if output_families is None:
        return _normalize_output_family_configs(default_output_families())
    return _normalize_output_family_configs(output_families)


def _resolve_direct_epfd_output_family_plan(
    family_configs: Mapping[str, Mapping[str, Any]],
    *,
    uemr_mode: bool = False,
) -> dict[str, Any]:
    """Translate canonical family configs to internal raw/preacc requirements.

    When ``uemr_mode=True`` the beam library is bypassed entirely, so all
    beam-related outputs (per-satellite beam counts, beam demand, beam
    statistics) are forced off regardless of their individual config —
    there is no beam data to write.
    """
    configs = _normalize_output_family_configs(family_configs)
    if uemr_mode:
        # Force beam_statistics off: no beam library, no counts. Keep the
        # other families alone — the user explicitly selected them and
        # they're still meaningful (EPFD distribution, per-sat PFD, etc.).
        configs = dict(configs)
        configs["beam_statistics"] = dict(configs["beam_statistics"])
        configs["beam_statistics"]["mode"] = "off"

    raw_epfd = _mode_includes_raw(configs["epfd_distribution"]["mode"])
    raw_prx_total = _mode_includes_raw(configs["prx_total_distribution"]["mode"])
    raw_prx_per_sat = _mode_includes_raw(configs["prx_elevation_heatmap"]["mode"])
    raw_total_pfd = _mode_includes_raw(configs["total_pfd_ras_distribution"]["mode"])
    raw_per_sat_pfd = (
        _mode_includes_raw(configs["per_satellite_pfd_distribution"]["mode"])
        or _mode_includes_raw(configs["per_satellite_pfd_elevation_heatmap"]["mode"])
    )
    raw_beam_counts = _mode_includes_raw(configs["beam_statistics"]["mode"])
    raw_sat_elevation = bool(
        _mode_includes_raw(configs["prx_elevation_heatmap"]["mode"])
        or _mode_includes_raw(configs["per_satellite_pfd_elevation_heatmap"]["mode"])
        or _mode_includes_raw(configs["beam_statistics"]["mode"])
    )
    raw_beam_demand = _mode_includes_raw(configs["beam_statistics"]["mode"])

    pre_prx_total = _mode_includes_preaccumulated(configs["prx_total_distribution"]["mode"])
    pre_epfd = _mode_includes_preaccumulated(configs["epfd_distribution"]["mode"])
    pre_total_pfd = _mode_includes_preaccumulated(
        configs["total_pfd_ras_distribution"]["mode"]
    )
    pre_per_sat_pfd = _mode_includes_preaccumulated(
        configs["per_satellite_pfd_distribution"]["mode"]
    )
    pre_prx_heatmap = _mode_includes_preaccumulated(
        configs["prx_elevation_heatmap"]["mode"]
    )
    pre_per_sat_pfd_heatmap = _mode_includes_preaccumulated(
        configs["per_satellite_pfd_elevation_heatmap"]["mode"]
    )
    pre_beam_stats = _mode_includes_preaccumulated(configs["beam_statistics"]["mode"])

    return {
        "family_configs": configs,
        "write_epfd": bool(raw_epfd),
        "write_prx_total": bool(raw_prx_total),
        "write_per_satellite_prx_ras_station": bool(raw_prx_per_sat),
        "write_total_pfd_ras_station": bool(raw_total_pfd),
        "write_per_satellite_pfd_ras_station": bool(raw_per_sat_pfd),
        "write_sat_beam_counts_used": bool(raw_beam_counts),
        "write_sat_elevation_ras_station": bool(raw_sat_elevation),
        "write_beam_demand_count": bool(raw_beam_demand),
        "preacc_prx_total_distribution": bool(pre_prx_total),
        "preacc_epfd_distribution": bool(pre_epfd),
        "preacc_total_pfd_ras_distribution": bool(pre_total_pfd),
        "preacc_per_satellite_pfd_distribution": bool(pre_per_sat_pfd),
        "preacc_prx_elevation_heatmap": bool(pre_prx_heatmap),
        "preacc_per_satellite_pfd_elevation_heatmap": bool(pre_per_sat_pfd_heatmap),
        "preacc_beam_statistics": bool(pre_beam_stats),
        "needs_per_satellite_prx": bool(raw_prx_per_sat or pre_prx_heatmap),
        "needs_per_satellite_pfd": bool(raw_per_sat_pfd or pre_per_sat_pfd or pre_per_sat_pfd_heatmap),
        "needs_total_prx": bool(raw_prx_total or pre_prx_total),
        "needs_epfd": bool(raw_epfd or pre_epfd),
        "needs_total_pfd": bool(raw_total_pfd or pre_total_pfd),
        "needs_sat_elevation": bool(
            raw_sat_elevation or pre_prx_heatmap or pre_per_sat_pfd_heatmap or pre_beam_stats
        ),
        "needs_beam_counts": bool(raw_beam_counts or pre_beam_stats),
        "needs_beam_demand": bool(raw_beam_demand or pre_beam_stats),
    }


def _histogram_range_is_stable(
    value_edges: np.ndarray | None,
    *,
    margin_bins: int,
    step_db: float,
    batch_index: int,
    recheck_interval: int = 5,
) -> bool:
    """Return True when the histogram edges are established and this batch
    can safely skip the ``value_range_db`` GPU round-trip.

    Once the edges are set (not None) and have sufficient margin, we only
    re-check the range every *recheck_interval* batches.  This reduces
    GPU synchronisation overhead by ~80 % during steady-state operation.
    """
    if value_edges is None:
        return False  # must establish edges
    if batch_index < 3:
        return False  # always check the first few batches
    if (batch_index % max(1, recheck_interval)) == 0:
        return False  # periodic re-check
    # Edges are established and this is a skip-eligible batch
    return True


def _accumulate_1d_distribution_batch(
    session: Any,
    collector: dict[str, Any],
    *,
    value_linear: Any,
    db_offset_db: float,
    batch_index: int,
) -> None:
    """Accumulate one batch into a 1-D distribution collector using a fused
    GPU call that computes the dB range and histogram in a single pass.

    This replaces the separate ``value_range_db`` + ``accumulate_value_histogram``
    pair, halving the number of GPU kernel launches and host synchronisation
    points per distribution family.
    """
    config = collector["config"]
    edges = collector.get("edges_dbw")
    # When edges are established and this isn't a periodic recheck batch,
    # skip the range computation entirely — just accumulate the histogram.
    skip_range = _histogram_range_is_stable(
        edges,
        margin_bins=int(config.get("margin_bins", 16)),
        step_db=float(config.get("bin_step_db", 0.02)),
        batch_index=batch_index,
    )
    if skip_range and edges is not None:
        # Fast path: histogram only, no range check
        batch_hist = session.accumulate_value_histogram(
            value_linear=value_linear,
            value_edges_dbw=edges,
            db_offset_db=float(db_offset_db),
            return_device=False,
        )
        if batch_hist is not None:
            collector["counts"] += np.asarray(batch_hist, dtype=np.int64)
            collector["sample_count"] += int(np.sum(batch_hist, dtype=np.int64))
        return
    # Full path: fused range + histogram in one GPU call
    fused = session.value_range_and_histogram(
        value_linear=value_linear,
        value_edges_dbw=edges,
        db_offset_db=float(db_offset_db),
    )
    batch_range = fused["range_db"]
    if batch_range is None:
        return
    # Ensure edges cover the batch range
    if edges is None:
        if str(config.get("range_mode", "dynamic")) == "fixed":
            collector["edges_dbw"] = _fixed_histogram_edges(
                min_db=float(config["min_db"]),
                max_db=float(config["max_db"]),
                step_db=float(config["bin_step_db"]),
            )
            collector["counts"] = np.zeros(
                int(collector["edges_dbw"].size) - 1, dtype=np.int64,
            )
        else:
            collector["counts"], collector["edges_dbw"] = (
                _ensure_dynamic_histogram_range_1d(
                    collector["counts"],
                    collector["edges_dbw"],
                    batch_min_dbw=float(batch_range[0]),
                    batch_max_dbw=float(batch_range[1]),
                    step_db=float(config["bin_step_db"]),
                    margin_bins=int(config["margin_bins"]),
                )
            )
    elif str(config.get("range_mode", "dynamic")) == "dynamic":
        collector["counts"], collector["edges_dbw"] = (
            _ensure_dynamic_histogram_range_1d(
                collector["counts"],
                collector["edges_dbw"],
                batch_min_dbw=float(batch_range[0]),
                batch_max_dbw=float(batch_range[1]),
                step_db=float(config["bin_step_db"]),
                margin_bins=int(config["margin_bins"]),
            )
        )
    # If the fused call produced a histogram AND the bin count matches
    # the current collector (i.e. edges weren't expanded), reuse it.
    batch_hist = fused.get("histogram")
    new_edges = collector.get("edges_dbw")
    edges_unchanged = (
        batch_hist is not None
        and edges is not None
        and new_edges is not None
        and int(new_edges.size) - 1 == int(batch_hist.size)
    )
    if edges_unchanged:
        collector["counts"] += np.asarray(batch_hist, dtype=np.int64)
        collector["sample_count"] += int(np.sum(batch_hist, dtype=np.int64))
    else:
        # Edges were created or expanded — recompute histogram with new edges
        batch_hist = session.accumulate_value_histogram(
            value_linear=value_linear,
            value_edges_dbw=collector["edges_dbw"],
            db_offset_db=float(db_offset_db),
            return_device=False,
        )
        if batch_hist is not None:
            collector["counts"] += np.asarray(batch_hist, dtype=np.int64)
            collector["sample_count"] += int(np.sum(batch_hist, dtype=np.int64))


def _ensure_dynamic_histogram_range_1d(
    counts: np.ndarray | None,
    value_edges: np.ndarray | None,
    *,
    batch_min_dbw: float,
    batch_max_dbw: float,
    step_db: float,
    margin_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    step = float(step_db)
    margin = int(max(0, margin_bins))

    need_lo = max(_HISTOGRAM_DB_FLOOR, _align_down_to_step(batch_min_dbw, step) - margin * step)
    need_hi = _align_up_to_step(batch_max_dbw, step) + margin * step

    if value_edges is None or counts is None:
        n_value_bins = max(1, int(round((need_hi - need_lo) / step)))
        value_edges = need_lo + step * np.arange(n_value_bins + 1, dtype=np.float64)
        counts = np.zeros(n_value_bins, dtype=np.int64)
        return counts, value_edges

    cur_lo = max(_HISTOGRAM_DB_FLOOR, float(value_edges[0]))
    cur_hi = float(value_edges[-1])
    add_left = 0
    add_right = 0
    if need_lo < cur_lo - 1e-12:
        add_left = int(round((cur_lo - need_lo) / step))
    if need_hi > cur_hi + 1e-12:
        add_right = int(round((need_hi - cur_hi) / step))
    if add_left == 0 and add_right == 0:
        return counts, value_edges

    counts = np.pad(counts, (add_left, add_right), mode="constant")
    new_lo = cur_lo - add_left * step
    value_edges = new_lo + step * np.arange(counts.shape[0] + 1, dtype=np.float64)
    return counts, value_edges


def _fixed_histogram_edges(
    *,
    min_db: float,
    max_db: float,
    step_db: float,
) -> np.ndarray:
    step = float(step_db)
    lo = float(min_db)
    hi = float(max_db)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Fixed histogram ranges require finite min_db < max_db.")
    n_bins = max(1, int(round((hi - lo) / step)))
    return lo + step * np.arange(n_bins + 1, dtype=np.float64)


def _default_preacc_value_edges_from_config(
    config: Mapping[str, Any],
    *,
    step_key: str,
) -> np.ndarray:
    """Return deterministic fallback edges for empty preaccumulated families."""
    step = float(config.get(step_key, 0.02) or 0.02)
    min_db = config.get("min_db")
    max_db = config.get("max_db")
    try:
        if min_db is not None and max_db is not None:
            return _fixed_histogram_edges(
                min_db=float(min_db),
                max_db=float(max_db),
                step_db=float(step),
            )
    except Exception:
        pass
    half_step = 0.5 * float(step)
    return np.asarray([-half_step, half_step], dtype=np.float64)


def _finalize_empty_distribution_collector(
    collector: MutableMapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a zero-sample distribution collector with stable edges."""
    edges_dbw = collector.get("edges_dbw")
    if edges_dbw is None:
        edges_dbw = _default_preacc_value_edges_from_config(
            collector["config"],
            step_key="bin_step_db",
        )
        collector["edges_dbw"] = edges_dbw
    counts = collector.get("counts")
    if counts is None:
        counts = np.zeros(max(1, int(np.asarray(edges_dbw).size) - 1), dtype=np.int64)
        collector["counts"] = counts
    return (
        np.asarray(counts, dtype=np.int64),
        np.asarray(edges_dbw, dtype=np.float64),
    )


def _finalize_empty_heatmap_collector(
    collector: MutableMapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize a zero-sample heatmap collector with stable axes."""
    value_edges_dbw = collector.get("value_edges_dbw")
    if value_edges_dbw is None:
        value_edges_dbw = _default_preacc_value_edges_from_config(
            collector["config"],
            step_key="value_bin_step_db",
        )
        collector["value_edges_dbw"] = value_edges_dbw
    elevation_edges_deg = np.asarray(collector["elevation_edges_deg"], dtype=np.float64)
    counts = collector.get("counts")
    if counts is None:
        counts = np.zeros(
            (
                max(1, int(elevation_edges_deg.size) - 1),
                max(1, int(np.asarray(value_edges_dbw).size) - 1),
            ),
            dtype=np.int64,
        )
        collector["counts"] = counts
    return (
        np.asarray(counts, dtype=np.int64),
        elevation_edges_deg,
        np.asarray(value_edges_dbw, dtype=np.float64),
    )


def _accumulate_heatmap_batch(
    session: Any,
    collector: dict[str, Any],
    *,
    value_linear: Any,
    sat_elevation_deg: Any,
    db_offset_db: float = 0.0,
) -> None:
    """Accumulate one batch into a 2-D elevation x value heatmap collector.

    This encapsulates the repeated pattern of:
    1. Compute the dB range of the batch values via ``session.value_range_db``.
    2. Initialise or expand the value-axis edges (fixed or dynamic mode).
    3. Call ``session.accumulate_value_elevation_heatmap`` and merge counts.

    Parameters
    ----------
    session : GpuScepterSession
        Active GPU session with ``value_range_db`` and
        ``accumulate_value_elevation_heatmap`` methods.
    collector : dict
        Mutable heatmap collector dict with keys ``config``,
        ``value_edges_dbw``, ``elevation_edges_deg``, ``counts``, and
        ``sample_count``.
    value_linear : device array
        Per-satellite linear-power values for this batch (device memory).
    sat_elevation_deg : device array
        Per-satellite elevation angles in degrees (device memory).
    db_offset_db : float, optional
        Additive dB offset passed through to the GPU helpers. Default 0.
    """
    config = collector["config"]
    batch_range = session.value_range_db(
        value_linear=value_linear,
        db_offset_db=db_offset_db,
    )
    if batch_range is None:
        return
    range_mode = str(config.get("range_mode", "dynamic"))
    if collector["value_edges_dbw"] is None:
        if range_mode == "fixed":
            collector["value_edges_dbw"] = _fixed_histogram_edges(
                min_db=float(config["min_db"]),
                max_db=float(config["max_db"]),
                step_db=float(config["value_bin_step_db"]),
            )
            collector["counts"] = np.zeros(
                (
                    int(collector["elevation_edges_deg"].size) - 1,
                    int(collector["value_edges_dbw"].size) - 1,
                ),
                dtype=np.int64,
            )
        else:
            collector["counts"], collector["value_edges_dbw"] = (
                _ensure_dynamic_histogram_range(
                    collector["counts"],
                    collector["value_edges_dbw"],
                    batch_min_dbw=float(batch_range[0]),
                    batch_max_dbw=float(batch_range[1]),
                    step_db=float(config["value_bin_step_db"]),
                    margin_bins=int(config["value_margin_bins"]),
                    n_elevation_bins=int(
                        collector["elevation_edges_deg"].size - 1
                    ),
                )
            )
    elif range_mode == "dynamic":
        collector["counts"], collector["value_edges_dbw"] = (
            _ensure_dynamic_histogram_range(
                collector["counts"],
                collector["value_edges_dbw"],
                batch_min_dbw=float(batch_range[0]),
                batch_max_dbw=float(batch_range[1]),
                step_db=float(config["value_bin_step_db"]),
                margin_bins=int(config["value_margin_bins"]),
                n_elevation_bins=int(
                    collector["elevation_edges_deg"].size - 1
                ),
            )
        )
    batch_hist = session.accumulate_value_elevation_heatmap(
        value_linear=value_linear,
        sat_elevation_deg=sat_elevation_deg,
        elevation_edges_deg=collector["elevation_edges_deg"],
        value_edges_dbw=collector["value_edges_dbw"],
        db_offset_db=db_offset_db,
        sky_slab=int(config.get("sky_slab", 16)),
        return_device=False,
    )
    collector["counts"] += np.asarray(batch_hist, dtype=np.int64)
    collector["sample_count"] += int(np.sum(batch_hist, dtype=np.int64))


def _merge_count_histograms(
    existing: np.ndarray | None,
    batch_hist: np.ndarray,
) -> np.ndarray:
    batch_arr = np.asarray(batch_hist, dtype=np.int64)
    if existing is None:
        return batch_arr.copy()
    if existing.shape[0] < batch_arr.shape[0]:
        existing = np.pad(existing, (0, batch_arr.shape[0] - existing.shape[0]), mode="constant")
    elif batch_arr.shape[0] < existing.shape[0]:
        batch_arr = np.pad(batch_arr, (0, existing.shape[0] - batch_arr.shape[0]), mode="constant")
    existing += batch_arr
    return existing


def _append_series_segment(parts: list[np.ndarray], value: Any) -> None:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    parts.append(np.asarray(arr))


def _bincount_device_to_host(
    cp: Any,
    gpu_module: Any,
    values_device: Any,
) -> np.ndarray:
    values_dev = cp.asarray(values_device, dtype=np.int64).reshape(-1)
    if int(values_dev.size) <= 0:
        return np.zeros(1, dtype=np.int64)
    if hasattr(cp, "bincount"):
        return np.asarray(
            gpu_module.copy_device_to_host(cp.bincount(values_dev, minlength=1)),
            dtype=np.int64,
        )
    values_host = np.asarray(gpu_module.copy_device_to_host(values_dev), dtype=np.int64)
    if values_host.size <= 0:
        return np.zeros(1, dtype=np.int64)
    return np.bincount(values_host, minlength=1).astype(np.int64, copy=False)


def _beam_count_samples_device(
    cp: Any,
    counts_device: Any,
) -> Any:
    """
    Collapse optional sky axes to a per-time/per-satellite sample view for
    histogram / CCDF style statistics.

    Note:
    This helper is intentionally conservative for per-satellite beam-load
    distributions. It must NOT be used to build network-total beam-over-time
    series, because max-per-satellite across sky and then summing satellites
    can overstate the real total.
    """
    counts_dev = cp.asarray(counts_device, dtype=np.int64)
    if counts_dev.ndim == 2:
        return counts_dev
    if counts_dev.ndim == 3:
        return counts_dev.max(axis=1)
    raise ValueError(
        f"Beam-count preaccumulation expects (T,S) or (T,sky,S) device arrays, got {counts_dev.shape!r}."
    )


def _beam_total_over_time_device(
    cp: Any,
    counts_device: Any,
) -> Any:
    """
    Return one physically consistent service-beam total per timestep.

    For sky-resolved boresight counts, sum satellites first for each sky row,
    then collapse sky with a max-over-sky reduction. That preserves the sanity
    bound against beam demand and avoids the invalid max-per-satellite stitch.
    """
    counts_dev = cp.asarray(counts_device, dtype=np.int64)
    if counts_dev.ndim == 2:
        return counts_dev.sum(axis=1, dtype=np.int64)
    if counts_dev.ndim == 3:
        sky_totals_dev = counts_dev.sum(axis=2, dtype=np.int64)
        return sky_totals_dev.max(axis=1)
    raise ValueError(
        f"Beam totals expect (T,S) or (T,sky,S) device arrays, got {counts_dev.shape!r}."
    )


def _visible_beam_statistics_device(
    cp: Any,
    gpu_module: Any,
    *,
    counts_samples_device: Any,
    visibility_mask_device: Any,
) -> tuple[np.ndarray, Any]:
    """Return visible-only beam histogram and per-timestep totals for already-collapsed (T,S) samples."""
    counts_samples_dev = cp.asarray(counts_samples_device, dtype=np.int64)
    visibility_mask_dev = cp.asarray(visibility_mask_device, dtype=bool)
    if counts_samples_dev.shape != visibility_mask_dev.shape:
        raise ValueError(
            "Visible beam statistics require matching (T,S) shapes for counts and visibility "
            f"masks, got {counts_samples_dev.shape!r} and {visibility_mask_dev.shape!r}."
        )
    visible_total_dev = cp.where(
        visibility_mask_dev,
        counts_samples_dev,
        np.int64(0),
    ).sum(axis=1, dtype=np.int64)
    visible_sample_count = int(cp.count_nonzero(visibility_mask_dev))
    if visible_sample_count <= 0:
        return np.zeros(1, dtype=np.int64), visible_total_dev
    visible_samples_dev = counts_samples_dev[visibility_mask_dev]
    visible_hist = _bincount_device_to_host(
        cp,
        gpu_module,
        visible_samples_dev.reshape(-1),
    )
    return visible_hist, visible_total_dev


def _visible_beam_total_over_time_device(
    cp: Any,
    *,
    counts_device: Any,
    visibility_mask_device: Any,
) -> Any:
    """
    Return one physically consistent visible-service-beam total per timestep.

    Visibility is satellite-only, so for sky-resolved boresight counts we apply
    the (T,S) visibility mask across the satellite axis, sum satellites per sky,
    then collapse sky with max-over-sky.
    """
    counts_dev = cp.asarray(counts_device, dtype=np.int64)
    visibility_mask_dev = cp.asarray(visibility_mask_device, dtype=bool)

    if counts_dev.ndim == 2:
        if counts_dev.shape != visibility_mask_dev.shape:
            raise ValueError(
                "Visible beam totals require matching (T,S) shapes for counts and visibility "
                f"masks, got {counts_dev.shape!r} and {visibility_mask_dev.shape!r}."
            )
        return cp.where(
            visibility_mask_dev,
            counts_dev,
            np.int64(0),
        ).sum(axis=1, dtype=np.int64)

    if counts_dev.ndim == 3:
        expected_shape = (int(counts_dev.shape[0]), int(counts_dev.shape[2]))
        if tuple(visibility_mask_dev.shape) != expected_shape:
            raise ValueError(
                "Visible beam totals require visibility mask shape "
                f"{expected_shape!r} for counts shape {tuple(counts_dev.shape)!r}; "
                f"got {tuple(visibility_mask_dev.shape)!r}."
            )
        masked_dev = cp.where(
            visibility_mask_dev[:, None, :],
            counts_dev,
            np.int64(0),
        )
        sky_totals_dev = masked_dev.sum(axis=2, dtype=np.int64)
        return sky_totals_dev.max(axis=1)

    raise ValueError(
        f"Visible beam totals expect (T,S) or (T,sky,S) device arrays, got {counts_dev.shape!r}."
    )


def _ensure_dynamic_histogram_range(
    counts: np.ndarray | None,
    value_edges: np.ndarray | None,
    *,
    batch_min_dbw: float,
    batch_max_dbw: float,
    step_db: float,
    margin_bins: int,
    n_elevation_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    step = float(step_db)
    margin = int(max(0, margin_bins))

    need_lo = max(_HISTOGRAM_DB_FLOOR, _align_down_to_step(batch_min_dbw, step) - margin * step)
    need_hi = _align_up_to_step(batch_max_dbw, step) + margin * step

    if value_edges is None or counts is None:
        n_value_bins = max(1, int(round((need_hi - need_lo) / step)))
        value_edges = need_lo + step * np.arange(n_value_bins + 1, dtype=np.float64)
        counts = np.zeros((int(n_elevation_bins), n_value_bins), dtype=np.int64)
        return counts, value_edges

    cur_lo = max(_HISTOGRAM_DB_FLOOR, float(value_edges[0]))
    cur_hi = float(value_edges[-1])

    add_left = 0
    add_right = 0

    if need_lo < cur_lo - 1e-12:
        add_left = int(round((cur_lo - need_lo) / step))
    if need_hi > cur_hi + 1e-12:
        add_right = int(round((need_hi - cur_hi) / step))

    if add_left == 0 and add_right == 0:
        return counts, value_edges

    counts = np.pad(counts, ((0, 0), (add_left, add_right)), mode="constant")
    new_lo = cur_lo - add_left * step
    value_edges = new_lo + step * np.arange(counts.shape[1] + 1, dtype=np.float64)
    return counts, value_edges


def _resolve_direct_epfd_output_names(
    *,
    write_epfd: bool,
    write_prx_total: bool,
    write_per_satellite_prx_ras_station: bool,
    write_total_pfd_ras_station: bool,
    write_per_satellite_pfd_ras_station: bool,
    write_sat_beam_counts_used: bool,
    write_sat_elevation_ras_station: bool,
    write_beam_demand_count: bool,
    write_sat_eligible_mask: bool = False,
) -> list[str]:
    output_names: list[str] = []
    if write_epfd:
        output_names.append("EPFD_W_m2")
    if write_prx_total:
        output_names.append("Prx_total_W")
    if write_per_satellite_prx_ras_station:
        output_names.append("Prx_per_sat_RAS_STATION_W")
    if write_total_pfd_ras_station:
        output_names.append("PFD_total_RAS_STATION_W_m2")
    if write_per_satellite_pfd_ras_station:
        output_names.append("PFD_per_sat_RAS_STATION_W_m2")
    if write_sat_beam_counts_used:
        output_names.append("sat_beam_counts_used")
    if write_sat_elevation_ras_station:
        output_names.append("sat_elevation_RAS_STATION_deg")
    if write_beam_demand_count:
        output_names.append("beam_demand_count")
    if write_sat_eligible_mask:
        output_names.append("sat_eligible_mask")
    return output_names


def _minimum_unsigned_count_dtype(max_count: int) -> np.dtype:
    max_count_int = max(0, int(max_count))
    if max_count_int <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_count_int <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_count_int <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _direct_epfd_count_dtype(nbeam: int) -> np.dtype:
    return _minimum_unsigned_count_dtype(nbeam)


def _beam_demand_count_dtype(n_cells: int, nco: int) -> np.dtype:
    return _minimum_unsigned_count_dtype(int(n_cells) * int(nco))


def _count_active_beam_demand_device(
    cp: Any,
    cell_active_mask: Any,
    *,
    nco: int,
    dtype: np.dtype,
) -> Any:
    active_mask_cp = cp.asarray(cell_active_mask, dtype=cp.int32)
    counts_cp = active_mask_cp.sum(axis=1, dtype=np.int64)
    if int(nco) != 1:
        counts_cp = counts_cp * np.int64(int(nco))
    return cp.asarray(counts_cp, dtype=np.dtype(dtype))


def _count_packed_beams_from_indices_device(cp: Any, beam_idx: Any) -> Any:
    """
    Count only demand-serving beams from packed beam tables.

    Reserved/protective boresight RAS slots are encoded as -2 and are not
    included in this count.
    """
    beam_idx_cp = cp.asarray(beam_idx)
    valid_beam_mask = beam_idx_cp >= 0
    if beam_idx_cp.ndim == 3:
        return valid_beam_mask.sum(axis=2, dtype=np.int32)
    if beam_idx_cp.ndim == 4:
        return valid_beam_mask.sum(axis=3, dtype=np.int32)
    raise ValueError(
        "beam_idx must have shape (T, S_eff, N_beam) or (T, N_sky, S_eff, N_beam); "
        f"got {tuple(beam_idx_cp.shape)!r}."
    )


def _sample_cell_activity_mask_device(
    cp: Any,
    *,
    time_count: int,
    cell_count: int,
    activity_factor: float,
    seed: int | None,
) -> Any:
    shape = (int(time_count), int(cell_count))
    factor = float(activity_factor)
    if factor >= 1.0:
        return cp.ones(shape, dtype=cp.bool_)
    if factor <= 0.0:
        return cp.zeros(shape, dtype=cp.bool_)
    return _direct_epfd_random_draws_device(
        cp,
        shape=shape,
        seed=seed,
    ) < cp.float32(factor)


def _direct_epfd_random_draws_device(
    cp: Any,
    *,
    shape: tuple[int, ...],
    seed: int | None = None,
    rng: Any | None = None,
) -> Any:
    if rng is not None:
        if hasattr(rng, "random_sample"):
            draws = rng.random_sample(shape)
        else:
            draws = rng.random(shape)
    elif seed is None:
        if hasattr(cp.random, "random_sample"):
            draws = cp.random.random_sample(shape)
        else:
            draws = cp.random.random(shape)
    else:
        if hasattr(cp.random, "RandomState"):
            rng_local = cp.random.RandomState(int(seed))
            if hasattr(rng_local, "random_sample"):
                draws = rng_local.random_sample(shape)
            else:
                draws = rng_local.random(shape)
        else:
            rng_local = cp.random.default_rng(int(seed))
            draws = rng_local.random(shape)
    return cp.asarray(draws, dtype=cp.float32)


def _sample_cell_group_activity_mask_device(
    cp: Any,
    *,
    time_count: int,
    cell_count: int,
    group_count: int,
    activity_factor: float,
    seed: int | None,
    group_valid_mask: Any | None = None,
) -> Any:
    shape = (int(time_count), int(cell_count), int(group_count))
    valid_mask_cp = None
    if group_valid_mask is not None:
        valid_mask_cp = cp.asarray(group_valid_mask, dtype=cp.bool_)
        if tuple(int(v) for v in valid_mask_cp.shape) != (int(cell_count), int(group_count)):
            raise ValueError(
                "group_valid_mask must have shape "
                f"({int(cell_count)}, {int(group_count)}); got {tuple(valid_mask_cp.shape)!r}."
            )
    factor = float(activity_factor)
    if factor >= 1.0:
        if valid_mask_cp is None:
            return cp.ones(shape, dtype=cp.bool_)
        return cp.broadcast_to(valid_mask_cp[None, :, :], shape)
    if factor <= 0.0:
        return cp.zeros(shape, dtype=cp.bool_)
    draws = _direct_epfd_random_draws_device(
        cp,
        shape=shape,
        seed=seed,
    ) < cp.float32(factor)
    if valid_mask_cp is not None:
        return cp.logical_and(draws, valid_mask_cp[None, :, :])
    return draws


def _collapse_cell_group_activity_mask_device(cp: Any, group_active_mask: Any) -> Any:
    return cp.asarray(cp.any(group_active_mask, axis=2), dtype=cp.bool_)


@dataclass
class _DirectEpfdDynamicSpectrumState:
    """Device-resident dynamic spectrum inputs reused across accumulation slabs."""

    group_active_mask_dev: Any
    cell_group_leakage_factors_dev: Any
    group_valid_mask_dev: Any | None
    power_policy: str
    split_total_group_denominator_mode: str
    configured_groups_per_cell: Any


def _accumulate_cell_group_weighted_sum_device(
    cp: Any,
    *,
    group_active_mask: Any,
    cell_group_leakage_factors: Any,
    out: Any,
    group_valid_mask: Any | None = None,
) -> Any:
    """Accumulate `(T, C)` group-weighted leakage without a full `(T, C, G)` temp."""
    group_mask_cp = cp.asarray(group_active_mask, dtype=cp.float32)
    leakage_cp = cp.asarray(cell_group_leakage_factors, dtype=cp.float32)
    weighted_sum_cp = cp.asarray(out, dtype=cp.float32)
    effective_leakage_cp = leakage_cp
    if group_valid_mask is not None:
        effective_leakage_cp = effective_leakage_cp * cp.asarray(
            group_valid_mask,
            dtype=cp.float32,
        )
    weighted_sum_cp[...] = cp.einsum(
        "tcg,cg->tc",
        group_mask_cp,
        effective_leakage_cp,
        optimize=True,
    )
    return weighted_sum_cp


def _accumulate_active_group_count_device(
    cp: Any,
    *,
    group_active_mask: Any,
    out: Any,
    group_valid_mask: Any | None = None,
) -> Any:
    """Accumulate `(T, C)` active-group counts without materializing cast copies."""
    group_mask_cp = cp.asarray(group_active_mask, dtype=cp.float32)
    active_groups_cp = cp.asarray(out, dtype=cp.float32)
    if group_valid_mask is None:
        active_groups_cp[...] = cp.sum(group_mask_cp, axis=2, dtype=cp.float32)
        return active_groups_cp
    active_groups_cp[...] = cp.einsum(
        "tcg,cg->tc",
        group_mask_cp,
        cp.asarray(group_valid_mask, dtype=cp.float32),
        optimize=True,
    )
    return active_groups_cp


def _configured_group_denominator_device(
    cp: Any,
    *,
    configured_groups_per_cell: Any,
    cell_count: int,
) -> tuple[Any, Any]:
    denominator_cp = cp.asarray(configured_groups_per_cell, dtype=cp.float32)
    if getattr(denominator_cp, "ndim", 0) == 0:
        scalar_value = cp.float32(max(1.0, float(denominator_cp)))
        return scalar_value, scalar_value
    if tuple(int(v) for v in denominator_cp.shape) != (int(cell_count),):
        raise ValueError(
            "configured_groups_per_cell must be a scalar or shape "
            f"({int(cell_count)},); got {tuple(denominator_cp.shape)!r}."
        )
    safe_denominator_cp = cp.where(
        denominator_cp > cp.float32(0.0),
        denominator_cp,
        cp.float32(1.0),
    )
    return denominator_cp, safe_denominator_cp


def _compute_cell_spectral_weight_device(
    cp: Any,
    *,
    group_active_mask: Any,
    cell_group_leakage_factors: Any,
    power_policy: str,
    split_total_group_denominator_mode: str,
    configured_groups_per_cell: Any,
    group_valid_mask: Any | None = None,
) -> Any:
    group_mask_cp = cp.asarray(group_active_mask)
    leakage_cp = cp.asarray(cell_group_leakage_factors, dtype=cp.float32)
    if getattr(leakage_cp, "ndim", 0) != 2:
        raise ValueError("cell_group_leakage_factors must have shape (C, G).")
    if getattr(group_mask_cp, "shape", ())[-1] != int(leakage_cp.shape[1]):
        raise ValueError(
            "group_active_mask and cell_group_leakage_factors must align over the group axis."
        )
    weighted_sum_cp = _accumulate_cell_group_weighted_sum_device(
        cp,
        group_active_mask=group_mask_cp,
        cell_group_leakage_factors=leakage_cp,
        out=cp.zeros(group_mask_cp.shape[:2], dtype=cp.float32),
        group_valid_mask=group_valid_mask,
    )
    if str(power_policy) == "repeat_per_group":
        return weighted_sum_cp

    if str(split_total_group_denominator_mode) == "active_groups":
        active_groups_cp = _accumulate_active_group_count_device(
            cp,
            group_active_mask=group_mask_cp,
            out=cp.zeros(group_mask_cp.shape[:2], dtype=cp.float32),
            group_valid_mask=group_valid_mask,
        )
        safe_active_groups_cp = cp.where(
            active_groups_cp > cp.float32(0.0),
            active_groups_cp,
            cp.float32(1.0),
        )
        return cp.where(
            active_groups_cp > cp.float32(0.0),
            weighted_sum_cp / safe_active_groups_cp,
            cp.float32(0.0),
        ).astype(cp.float32, copy=False)

    denominator_cp, safe_denominator_cp = _configured_group_denominator_device(
        cp,
        configured_groups_per_cell=configured_groups_per_cell,
        cell_count=int(group_mask_cp.shape[1]),
    )
    if getattr(cp.asarray(denominator_cp), "ndim", 0) == 0:
        return (weighted_sum_cp / safe_denominator_cp).astype(cp.float32, copy=False)
    return cp.where(
        denominator_cp[None, :] > cp.float32(0.0),
        weighted_sum_cp / safe_denominator_cp[None, :],
        cp.float32(0.0),
    ).astype(cp.float32, copy=False)


def _get_direct_epfd_activity_weight_scratch_device(
    cp: Any,
    scratch_cache: dict[tuple[int, int, bool], dict[str, Any]],
    *,
    slab_timesteps: int,
    cell_count: int,
    need_active_group_counts: bool,
) -> dict[str, Any]:
    """Reuse slab-local weight buffers across dynamic spectrum accumulation calls."""
    key = (int(slab_timesteps), int(cell_count), bool(need_active_group_counts))
    cached = scratch_cache.get(key)
    if cached is not None:
        return cached

    cached = {
        "weighted_sum": cp.zeros((int(slab_timesteps), int(cell_count)), dtype=cp.float32),
    }
    if bool(need_active_group_counts):
        cached["active_groups"] = cp.zeros(
            (int(slab_timesteps), int(cell_count)),
            dtype=cp.float32,
        )
    scratch_cache[key] = cached
    return cached


def _compute_cell_spectral_weight_slab_device(
    cp: Any,
    *,
    dynamic_spectrum_state: _DirectEpfdDynamicSpectrumState,
    time_start: int,
    time_stop: int,
    scratch_cache: dict[tuple[int, int, bool], dict[str, Any]],
) -> Any:
    """Compute slab-local cell spectral weights from resident group activity on device."""
    group_mask_cp = cp.asarray(
        dynamic_spectrum_state.group_active_mask_dev[int(time_start) : int(time_stop), :, :]
    )
    leakage_cp = cp.asarray(
        dynamic_spectrum_state.cell_group_leakage_factors_dev,
        dtype=cp.float32,
    )
    slab_timesteps = int(group_mask_cp.shape[0])
    cell_count = int(group_mask_cp.shape[1])
    need_active_group_counts = bool(
        str(dynamic_spectrum_state.power_policy) == "split_total_cell_power"
        and str(dynamic_spectrum_state.split_total_group_denominator_mode) == "active_groups"
    )
    scratch = _get_direct_epfd_activity_weight_scratch_device(
        cp,
        scratch_cache,
        slab_timesteps=slab_timesteps,
        cell_count=cell_count,
        need_active_group_counts=need_active_group_counts,
    )
    weighted_sum_cp = _accumulate_cell_group_weighted_sum_device(
        cp,
        group_active_mask=group_mask_cp,
        cell_group_leakage_factors=leakage_cp,
        out=scratch["weighted_sum"],
        group_valid_mask=dynamic_spectrum_state.group_valid_mask_dev,
    )
    if str(dynamic_spectrum_state.power_policy) == "repeat_per_group":
        return weighted_sum_cp

    if need_active_group_counts:
        active_groups_cp = _accumulate_active_group_count_device(
            cp,
            group_active_mask=group_mask_cp,
            out=scratch["active_groups"],
            group_valid_mask=dynamic_spectrum_state.group_valid_mask_dev,
        )
        safe_active_groups_cp = cp.where(
            active_groups_cp > cp.float32(0.0),
            active_groups_cp,
            cp.float32(1.0),
        )
        weighted_sum_cp[...] = cp.where(
            active_groups_cp > cp.float32(0.0),
            weighted_sum_cp / safe_active_groups_cp,
            cp.float32(0.0),
        ).astype(cp.float32, copy=False)
        return weighted_sum_cp

    denominator_cp, safe_denominator_cp = _configured_group_denominator_device(
        cp,
        configured_groups_per_cell=dynamic_spectrum_state.configured_groups_per_cell,
        cell_count=int(cell_count),
    )
    if getattr(cp.asarray(denominator_cp), "ndim", 0) == 0:
        weighted_sum_cp[...] = (weighted_sum_cp / safe_denominator_cp).astype(
            cp.float32,
            copy=False,
        )
        return weighted_sum_cp
    weighted_sum_cp[...] = cp.where(
        denominator_cp[None, :] > cp.float32(0.0),
        weighted_sum_cp / safe_denominator_cp[None, :],
        cp.float32(0.0),
    ).astype(cp.float32, copy=False)
    return weighted_sum_cp


def _compute_cell_activity_spectral_weight_time_slabbed_device(
    cp: Any,
    *,
    time_count: int,
    cell_count: int,
    group_count: int,
    activity_factor: float,
    seed: int | None,
    spectral_slab: int,
    need_power_outputs: bool,
    cell_group_leakage_factors: Any | None,
    power_policy: str,
    split_total_group_denominator_mode: str,
    configured_groups_per_cell: Any,
    group_valid_mask: Any | None = None,
) -> tuple[Any, Any | None]:
    time_count_i = int(max(0, time_count))
    cell_count_i = int(max(0, cell_count))
    group_count_i = int(max(1, group_count))
    spectral_slab_i = int(max(1, min(int(spectral_slab), max(1, time_count_i))))
    factor = float(activity_factor)

    cell_active_mask = cp.zeros((time_count_i, cell_count_i), dtype=cp.bool_)
    cell_spectral_weight = (
        None
        if not bool(need_power_outputs)
        else cp.zeros((time_count_i, cell_count_i), dtype=cp.float32)
    )
    valid_mask_cp = None
    if group_valid_mask is not None:
        valid_mask_cp = cp.asarray(group_valid_mask, dtype=cp.bool_)
        if tuple(int(v) for v in valid_mask_cp.shape) != (cell_count_i, group_count_i):
            raise ValueError(
                "group_valid_mask must have shape "
                f"({cell_count_i}, {group_count_i}); got {tuple(valid_mask_cp.shape)!r}."
            )

    if time_count_i == 0 or cell_count_i == 0:
        return cell_active_mask, cell_spectral_weight

    leakage_cp = None
    if cell_spectral_weight is not None:
        if cell_group_leakage_factors is None:
            raise ValueError("cell_group_leakage_factors are required when need_power_outputs=True.")
        leakage_cp = cp.asarray(cell_group_leakage_factors, dtype=cp.float32)
        if getattr(leakage_cp, "ndim", 0) != 2:
            raise ValueError("cell_group_leakage_factors must have shape (C, G).")
        if tuple(int(v) for v in leakage_cp.shape) != (cell_count_i, group_count_i):
            raise ValueError(
                "cell_group_leakage_factors must have shape "
                f"({cell_count_i}, {group_count_i}); got {tuple(leakage_cp.shape)!r}."
            )

    if factor <= 0.0:
        return cell_active_mask, cell_spectral_weight

    if factor >= 1.0:
        if valid_mask_cp is None:
            cell_active_mask = cp.ones((time_count_i, cell_count_i), dtype=cp.bool_)
        else:
            cell_active_rows = cp.any(valid_mask_cp, axis=1)
            cell_active_mask = cp.broadcast_to(
                cp.asarray(cell_active_rows, dtype=cp.bool_)[None, :],
                (time_count_i, cell_count_i),
            )
        if cell_spectral_weight is not None and leakage_cp is not None:
            all_groups_active = cp.ones((1, cell_count_i, group_count_i), dtype=cp.bool_)
            base_weight = _compute_cell_spectral_weight_device(
                cp,
                group_active_mask=all_groups_active,
                cell_group_leakage_factors=leakage_cp,
                power_policy=power_policy,
                split_total_group_denominator_mode=split_total_group_denominator_mode,
                configured_groups_per_cell=configured_groups_per_cell,
                group_valid_mask=valid_mask_cp,
            )
            cell_spectral_weight = (
                cp.ones((time_count_i, 1), dtype=cp.float32)
                * cp.asarray(base_weight[:1, :], dtype=cp.float32)
            ).astype(cp.float32, copy=False)
        return cell_active_mask, cell_spectral_weight

    rng = None
    if seed is not None:
        if hasattr(cp.random, "RandomState"):
            rng = cp.random.RandomState(int(seed))
        else:
            rng = cp.random.default_rng(int(seed))

    for t0 in range(0, time_count_i, spectral_slab_i):
        t1 = min(time_count_i, t0 + spectral_slab_i)
        group_active_mask = _direct_epfd_random_draws_device(
            cp,
            shape=(t1 - t0, cell_count_i, group_count_i),
            rng=rng,
        ) < cp.float32(factor)
        if valid_mask_cp is not None:
            group_active_mask = cp.logical_and(group_active_mask, valid_mask_cp[None, :, :])
        cell_active_mask[t0:t1, :] = _collapse_cell_group_activity_mask_device(
            cp,
            group_active_mask,
        )
        if cell_spectral_weight is not None and leakage_cp is not None:
            cell_spectral_weight[t0:t1, :] = _compute_cell_spectral_weight_device(
                cp,
                group_active_mask=group_active_mask,
                cell_group_leakage_factors=leakage_cp,
                power_policy=power_policy,
                split_total_group_denominator_mode=split_total_group_denominator_mode,
                configured_groups_per_cell=configured_groups_per_cell,
                group_valid_mask=valid_mask_cp,
            )

    return cell_active_mask, cell_spectral_weight


def _scatter_compact_per_satellite_device(
    cp: Any,
    compact_values: Any,
    sat_idx_device: Any,
    *,
    n_sats_total: int,
    dtype: np.dtype,
    boresight_active: bool,
    n_skycells: int,
) -> Any:
    sat_idx_cp = cp.asarray(sat_idx_device, dtype=cp.int32).reshape(-1)
    compact_cp = cp.asarray(compact_values)
    target_dtype = np.dtype(dtype)

    if not boresight_active:
        full_values = cp.zeros(
            (int(compact_cp.shape[0]), int(n_sats_total)),
            dtype=target_dtype,
        )
        if int(sat_idx_cp.size) > 0:
            full_values[:, sat_idx_cp] = cp.asarray(compact_cp, dtype=target_dtype)
        return full_values

    full_values = cp.zeros(
        (int(compact_cp.shape[0]), 1, int(n_sats_total), int(n_skycells)),
        dtype=target_dtype,
    )
    if int(sat_idx_cp.size) == 0:
        return full_values

    compact_cast = cp.asarray(compact_cp, dtype=target_dtype)
    if compact_cast.ndim == 4:
        full_values[:, 0, sat_idx_cp, :] = compact_cast[:, 0, :, :]
        return full_values
    if compact_cast.ndim == 3:
        full_values[:, 0, sat_idx_cp, :] = cp.transpose(compact_cast, (0, 2, 1))
        return full_values
    raise ValueError(
        "Boresight compact per-satellite tensors must have shape (T, N_sky, S_eff) "
        f"or (T, 1, S_eff, N_sky); got {tuple(compact_cast.shape)!r}."
    )


def _scatter_compact_satellite_time_series_device(
    cp: Any,
    compact_values: Any,
    sat_idx_device: Any,
    *,
    n_sats_total: int,
    dtype: np.dtype,
    fill_value: float | int = 0.0,
) -> Any:
    """Scatter compact `(T, S_eff)` values onto the full satellite axis on device."""
    sat_idx_cp = cp.asarray(sat_idx_device, dtype=cp.int32).reshape(-1)
    compact_cp = cp.asarray(compact_values, dtype=np.dtype(dtype))
    if compact_cp.ndim != 2:
        raise ValueError(
            "Compact satellite time-series tensors must have shape (T, S_eff); "
            f"got {tuple(compact_cp.shape)!r}."
        )

    full_values = cp.zeros(
        (int(compact_cp.shape[0]), int(n_sats_total)),
        dtype=np.dtype(dtype),
    )
    if fill_value != 0:
        full_values[...] = fill_value
    if int(sat_idx_cp.size) > 0:
        full_values[:, sat_idx_cp] = compact_cp
    return full_values


def _scatter_compact_per_satellite_host(
    compact_values: np.ndarray,
    sat_idx_host: np.ndarray,
    *,
    n_sats_total: int,
    dtype: np.dtype,
    boresight_active: bool,
    n_skycells: int,
) -> np.ndarray:
    sat_idx_np = np.asarray(sat_idx_host, dtype=np.int32).reshape(-1)
    compact_np = np.asarray(compact_values, dtype=np.dtype(dtype))
    # Auto-detect boresight from array shape: 2D = non-boresight, 3D/4D = boresight
    is_boresight = boresight_active and compact_np.ndim >= 3
    if not is_boresight:
        full_values = np.zeros(
            (int(compact_np.shape[0]), int(n_sats_total)),
            dtype=np.dtype(dtype),
        )
        if sat_idx_np.size > 0:
            full_values[:, sat_idx_np] = compact_np
        return full_values

    full_values = np.zeros(
        (int(compact_np.shape[0]), 1, int(n_sats_total), int(n_skycells)),
        dtype=np.dtype(dtype),
    )
    if sat_idx_np.size == 0:
        return full_values
    if compact_np.ndim == 4:
        full_values[:, 0, sat_idx_np, :] = compact_np[:, 0, :, :]
        return full_values
    if compact_np.ndim == 3:
        full_values[:, 0, sat_idx_np, :] = np.transpose(compact_np, (0, 2, 1))
        return full_values
    raise ValueError(
        "Boresight compact per-satellite tensors must have shape (T, N_sky, S_eff) "
        f"or (T, 1, S_eff, N_sky); got {tuple(compact_np.shape)!r}."
    )


def _scatter_compact_satellite_time_series_host(
    compact_values: np.ndarray,
    sat_idx_host: np.ndarray,
    *,
    n_sats_total: int,
    dtype: np.dtype,
    fill_value: float | int = 0.0,
) -> np.ndarray:
    sat_idx_np = np.asarray(sat_idx_host, dtype=np.int32).reshape(-1)
    compact_np = np.asarray(compact_values, dtype=np.dtype(dtype))
    if compact_np.ndim != 2:
        raise ValueError(
            "Compact satellite time-series tensors must have shape (T, S_eff); "
            f"got {tuple(compact_np.shape)!r}."
        )
    full_values = np.zeros(
        (int(compact_np.shape[0]), int(n_sats_total)),
        dtype=np.dtype(dtype),
    )
    if fill_value != 0:
        full_values[...] = fill_value
    if sat_idx_np.size > 0:
        full_values[:, sat_idx_np] = compact_np
    return full_values


def _copy_compact_satellite_indices_host(
    gpu_module: Any,
    sat_idx_device_or_host: Any,
) -> np.ndarray:
    """Return compact satellite indices as a host ``int32`` vector."""
    if isinstance(sat_idx_device_or_host, np.ndarray):
        return np.asarray(sat_idx_device_or_host, dtype=np.int32).reshape(-1)
    if hasattr(sat_idx_device_or_host, "__cuda_array_interface__"):
        return np.asarray(
            gpu_module.copy_device_to_host(sat_idx_device_or_host),
            dtype=np.int32,
        ).reshape(-1)
    try:
        host_view = np.asarray(sat_idx_device_or_host, dtype=np.int32)
    except TypeError as exc:
        raise TypeError(
            "Compact satellite indices must be a NumPy host vector or a CUDA-device "
            "array that can be copied explicitly at the export boundary."
        ) from exc
    return host_view.reshape(-1)


def _resolve_direct_epfd_writer_queue_limits(
    *,
    host_memory_budget_gb: float,
) -> tuple[int, int]:
    """Choose async writer backpressure limits using host memory, not GPU memory."""
    host_budget_bytes = max(0, int(float(host_memory_budget_gb) * (1024 ** 3)))
    queue_max_bytes = max(
        _DEFAULT_WRITER_QUEUE_MAX_BYTES,
        min(max(host_budget_bytes // 2, 0), 8 * 1024 ** 3),
    )
    queue_max_items = max(
        _DEFAULT_WRITER_QUEUE_MAX_ITEMS,
        min(32, int(np.ceil(float(queue_max_bytes) / float(128 * 1024 ** 2)))),
    )
    return int(queue_max_items), int(queue_max_bytes)


def _allocate_direct_epfd_power_result_device(
    cp: Any,
    sample: Mapping[str, Any],
    *,
    time_count: int,
    sky_count: int,
    boresight_active: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sample.items():
        sample_cp = cp.asarray(value)
        shape = list(sample_cp.shape)
        if not shape:
            result[key] = sample_cp
            continue
        shape[0] = int(time_count)
        if boresight_active:
            shape[-1] = int(sky_count)
        result[key] = cp.zeros(tuple(shape), dtype=sample_cp.dtype)
    return result


def _store_direct_epfd_power_slab_result(
    target: Mapping[str, Any],
    slab: Mapping[str, Any],
    *,
    boresight_active: bool,
    time_start: int,
    time_stop: int,
    sky_start: int,
    sky_stop: int,
) -> None:
    for key, value in slab.items():
        if key not in target:
            continue
        if boresight_active:
            if target[key].ndim == 4:
                target[key][time_start:time_stop, :, :, sky_start:sky_stop] = value
            elif target[key].ndim == 3:
                target[key][time_start:time_stop, :, sky_start:sky_stop] = value
            else:
                target[key][time_start:time_stop, sky_start:sky_stop] = value
        else:
            target[key][time_start:time_stop, ...] = value


def _estimate_direct_epfd_finalize_bytes_per_timestep(
    *,
    cell_count: int,
    sat_count_total: int,
    sat_output_count: int,
    n_links: int,
    n_beam: int,
    n_skycells: int,
    boresight_active: bool,
    ras_pointing_mode: str,
    include_diagnostics: bool,
    pattern_dtype: Any = np.float32,
) -> int:
    # Beam angle arrays (alpha, beta) use pattern_dtype; indices/counts use int32.
    pat_bytes = int(np.dtype(pattern_dtype).itemsize)
    sky_factor = int(n_skycells) if boresight_active else 1
    first_pass_factor = 1 if boresight_active else sky_factor
    row_factor = sky_factor

    # First-pass candidate arrays: index (4) + valid (1) + alpha (pat) + beta (pat)
    first_pass_bytes = first_pass_factor * int(cell_count) * int(n_links) * (4 + 1 + pat_bytes + pat_bytes)
    first_pass_bytes += first_pass_factor * (int(cell_count) + int(sat_count_total)) * 4

    # Beam output: index (4) + alpha (pat) + beta (pat)
    beam_output_bytes = row_factor * int(sat_output_count) * int(n_beam) * (4 + pat_bytes + pat_bytes)
    count_bytes = row_factor * int(sat_output_count) * 4
    if include_diagnostics:
        beam_output_bytes += row_factor * 5 * 4

    repair_bytes = 0
    if str(ras_pointing_mode).strip().lower() == "ras_station":
        repair_bytes += row_factor * int(cell_count) * int(n_links) * (4 + 1)
        repair_bytes += row_factor * int(sat_count_total) * (4 + 4 + 1 + 4)
    else:
        repair_bytes += row_factor * int(sat_count_total) * 4

    return int(first_pass_bytes + beam_output_bytes + count_bytes + repair_bytes)


def _estimate_direct_epfd_power_bytes_per_timestep(
    *,
    sat_visible_count: int,
    n_beam: int,
    n_skycells: int,
    boresight_active: bool,
    include_epfd: bool,
    include_prx_total: bool,
    include_per_satellite_prx: bool,
    include_total_pfd: bool,
    include_per_satellite_pfd: bool,
    power_dtype: Any = np.float32,
    pattern_dtype: Any = np.float32,
    surface_pfd_cap_enabled: bool = False,
    surface_pfd_cap_mode: str = "per_beam",
) -> int:
    include_receive_outputs = bool(
        include_epfd or include_prx_total or include_per_satellite_prx
    )
    pwr_b = int(np.dtype(power_dtype).itemsize)
    pat_b = int(np.dtype(pattern_dtype).itemsize)
    # Trig intermediates are at least fp32 even when pattern_dtype is fp16
    trig_b = max(4, pat_b)
    sky_factor = int(n_skycells) if boresight_active else 1
    sat_visible_i = int(max(0, sat_visible_count))
    beam_count_i = int(max(0, n_beam))

    # Surface-PFD cap: the hoisted cap factor tensor is allocated per slab
    # alongside the main power arrays.  Per-beam mode allocates a full
    # (T, [N_sky,] S, K) float32 tensor; per-satellite aggregate allocates
    # (T, [N_sky,] S) float32 plus ~3× that for the aggregate helper's
    # intermediate candidates+geometry (conservatively budgeted as 4×S×K
    # per timestep for memory safety even though the helper dynamically
    # sizes itself against free VRAM at runtime).
    cap_mode = str(surface_pfd_cap_mode or "per_beam").strip().lower()
    cap_bytes_per_timestep = 0
    if surface_pfd_cap_enabled:
        if cap_mode == "per_beam":
            # Hoisted (T, [N_sky,] S, K) cap factor + K lookup + peak_pfd
            cap_bytes_per_timestep = sky_factor * sat_visible_i * beam_count_i * 4 * 3
        else:
            # Hoisted (T, [N_sky,] S) cap factor + aggregate helper transient
            # (candidates × K_act × 4 × 3 floats — conservatively budgeted)
            cap_bytes_per_timestep = sky_factor * sat_visible_i * 4
            cap_bytes_per_timestep += sky_factor * sat_visible_i * beam_count_i * 4 * 3

    # The fused kernels (TX trig geometry, beam→ground geometry, atmosphere
    # LUT, RX angular distance + RAS pattern) eliminate most intermediate
    # arrays for the fp32 path.  The estimator below reflects the post-fusion
    # reality: only the fused kernel outputs (beam_sinb, beam_cosb,
    # cos_gamma_tx, d_target, e_target_deg, valid_geom, gtx_rel_lin,
    # gtx_abs_lin, atm) and the power-stage results (emitted_eirp, eirp_sum,
    # scale_w_channel, fspl_lin, vis_horizon) are simultaneously live.
    #
    # For fp64 profiles the sequential fallback path still allocates more
    # intermediates — the old (pre-fusion) estimate is used for those.
    is_fused = bool(pwr_b == 4 and trig_b == 4)

    if boresight_active:
        beam_rows = sky_factor * sat_visible_i * beam_count_i
        sat_rows = sky_factor * sat_visible_i
        # The 4-D boresight path uses sparse active-beam gather which
        # compacts to only the active beams.  The TX trig and beam
        # geometry fusions do NOT apply to this path (they are 3-D only).
        # The fused RX + RAS pattern kernel DOES apply when fp32.
        # The fused atmosphere LUT also applies.  Net effect: the 4-D
        # path has fewer fused intermediates than the 3-D path, so we
        # use a slightly higher estimate.
        total = beam_rows * (6 * trig_b + 8 * pwr_b + 4 * 1)
        total += sat_rows * (4 * pwr_b + 4 * trig_b + 1)
        total += sat_visible_i * (6 * pwr_b + 6 * trig_b + 1)
        total += sky_factor * (4 * pwr_b)
        if include_receive_outputs:
            if is_fused:
                # Fused RX kernel applies in the 4-D path too.
                total += sat_rows * (1 * pwr_b)
            else:
                total += sat_rows * (4 * trig_b + 3 * pwr_b)
        if include_total_pfd:
            total += sky_factor * pwr_b
        if include_per_satellite_pfd:
            total += sat_rows * pwr_b
        total += cap_bytes_per_timestep
        return int(total)

    beam_rows = sat_visible_i * beam_count_i
    if is_fused:
        # Fused 3-D path outputs: beam_sinb, beam_cosb, cos_gamma_tx (3×beam_rows×4),
        # gtx_offset_deg (beam_rows×4), gtx_rel_lin + gtx_abs_lin (2×beam_rows×4),
        # d_target + e_target_deg + valid_geom (2×beam_rows×4 + beam_rows×1),
        # emitted_eirp (beam_rows×4),
        # sat-level: range_m, sat_elev, atm_ras, fspl, vis, eirp_sum, scale
        total = beam_rows * (8 * pwr_b + 1)  # beam-level live arrays
        total += sat_visible_i * (6 * pwr_b + 1)  # sat-level arrays
    else:
        total = beam_rows * (6 * trig_b + 8 * pwr_b + 4 * 1)
        total += sat_visible_i * (8 * pwr_b + 6 * trig_b + 2 * 1)
    if include_receive_outputs:
        if is_fused:
            # Fused RX: grx_lin (T, N_tel, S) + prx_scale (T, N_tel, S)
            total += sat_visible_i * (2 * pwr_b)
        else:
            total += sat_visible_i * (4 * trig_b + 3 * pwr_b)
    if include_total_pfd:
        total += pwr_b
    if include_per_satellite_pfd:
        total += sat_visible_i * pwr_b
    total += cap_bytes_per_timestep
    return int(total)


def _estimate_direct_epfd_export_bytes_per_timestep(
    *,
    sat_count_total: int,
    sat_visible_count: int,
    n_skycells: int,
    boresight_active: bool,
    write_epfd: bool,
    write_prx_total: bool,
    write_per_satellite_prx_ras_station: bool,
    write_prx_elevation_heatmap: bool,
    write_total_pfd_ras_station: bool,
    write_per_satellite_pfd_ras_station: bool,
    write_sat_beam_counts_used: bool,
    write_sat_elevation_ras_station: bool,
    write_beam_demand_count: bool,
    write_sat_eligible_mask: bool,
    count_dtype: np.dtype,
    demand_count_dtype: np.dtype,
    power_dtype: Any = np.float32,
) -> int:
    pwr_b = int(np.dtype(power_dtype).itemsize)
    total = 0
    sky_factor = int(n_skycells) if boresight_active else 1
    sat_total_i = int(sat_count_total)
    sat_visible_i = int(max(0, sat_visible_count))
    if write_epfd:
        total += sky_factor * pwr_b
    if write_prx_total:
        total += sky_factor * pwr_b
    if write_per_satellite_prx_ras_station:
        total += sky_factor * sat_total_i * pwr_b
    if write_prx_elevation_heatmap:
        total += sky_factor * sat_visible_i * pwr_b
    if write_total_pfd_ras_station:
        total += (sky_factor if boresight_active else 1) * pwr_b
    if write_per_satellite_pfd_ras_station:
        total += (sky_factor if boresight_active else 1) * sat_total_i * pwr_b
    if write_sat_beam_counts_used:
        total += (sky_factor if boresight_active else 1) * sat_total_i * np.dtype(count_dtype).itemsize
    if write_sat_elevation_ras_station:
        total += sat_total_i * 4
    if write_beam_demand_count:
        total += np.dtype(demand_count_dtype).itemsize
    if write_sat_eligible_mask:
        total += sat_total_i
    return int(total)


def _estimate_direct_epfd_orbit_state_gpu_bytes(
    *,
    time_count: int,
    sat_count_total: int,
    compute_dtype: Any,
) -> int:
    dtype_obj = np.dtype(compute_dtype)
    time_count_i = int(max(0, time_count))
    sat_total_i = int(max(0, sat_count_total))
    total = 0
    total += 2 * _shape_nbytes((time_count_i,), compute_dtype)
    total += _shape_nbytes((sat_total_i,), compute_dtype)
    total += _shape_nbytes((time_count_i, sat_total_i), np.uint8)
    total += 2 * _shape_nbytes((time_count_i, sat_total_i, 3), dtype_obj)
    return int(total)


def _estimate_direct_epfd_setup_gpu_bytes(
    *,
    time_count: int,
    cell_count: int,
    sat_count_total: int,
    n_skycells: int,
    output_dtype: Any,
    need_pointings: bool,
    need_beam_demand: bool,
    demand_count_dtype: Any,
) -> int:
    total = 0
    time_count_i = int(max(0, time_count))
    sat_total_i = int(max(0, sat_count_total))
    cell_count_i = int(max(0, cell_count))
    sky_total_i = int(max(1, n_skycells))
    total += _shape_nbytes((time_count_i, sat_total_i, 4), output_dtype)
    total += _shape_nbytes((time_count_i, sat_total_i, 3), output_dtype)
    total += _shape_nbytes((sat_total_i,), np.bool_)
    total += _shape_nbytes((time_count_i, cell_count_i), np.bool_)
    if need_beam_demand:
        total += _shape_nbytes((time_count_i,), demand_count_dtype)
    if need_pointings:
        total += 2 * _shape_nbytes((time_count_i, sky_total_i), np.float32)
    return int(total)


def _estimate_direct_epfd_activity_gpu_memory(
    *,
    time_count: int,
    cell_count: int,
    groups_per_cell: int,
    cell_activity_mode: str,
    need_power_outputs: bool,
    spectral_slab: int | None = None,
    power_policy: str = "repeat_per_group",
    split_total_group_denominator_mode: str = "configured_groups",
) -> dict[str, int]:
    mode_name = _normalize_direct_epfd_cell_activity_mode(cell_activity_mode)
    group_count = int(max(1, groups_per_cell))
    if mode_name != "per_channel" or group_count <= 1:
        return {
            "spectral_slab": 1,
            "resident_bytes": 0,
            "scratch_bytes": 0,
            "peak_bytes": 0,
        }

    time_count_i = int(max(0, time_count))
    cell_count_i = int(max(0, cell_count))
    slab_i = int(
        max(
            1,
            min(
                int(time_count_i if spectral_slab is None else spectral_slab),
                max(1, time_count_i),
            ),
        )
    )
    resident_bytes = (
        _shape_nbytes((time_count_i, cell_count_i, group_count), np.bool_)
        if need_power_outputs
        else 0
    )
    scratch_bytes = 0
    if need_power_outputs:
        scratch_bytes += _shape_nbytes((slab_i, cell_count_i), np.float32)
        if (
            str(power_policy) == "split_total_cell_power"
            and str(split_total_group_denominator_mode) == "active_groups"
        ):
            scratch_bytes += _shape_nbytes((slab_i, cell_count_i), np.float32)
    return {
        "spectral_slab": int(slab_i),
        "resident_bytes": int(resident_bytes),
        "scratch_bytes": int(scratch_bytes),
        "peak_bytes": int(resident_bytes + scratch_bytes),
    }


def _estimate_direct_epfd_activity_gpu_bytes(
    *,
    time_count: int,
    cell_count: int,
    groups_per_cell: int,
    cell_activity_mode: str,
    need_power_outputs: bool,
    spectral_slab: int | None = None,
    power_policy: str = "repeat_per_group",
    split_total_group_denominator_mode: str = "configured_groups",
) -> int:
    return int(
        _estimate_direct_epfd_activity_gpu_memory(
            time_count=time_count,
            cell_count=cell_count,
            groups_per_cell=groups_per_cell,
            cell_activity_mode=cell_activity_mode,
            need_power_outputs=need_power_outputs,
            spectral_slab=spectral_slab,
            power_policy=power_policy,
            split_total_group_denominator_mode=split_total_group_denominator_mode,
        )["peak_bytes"]
    )


def _estimate_direct_epfd_visible_resident_gpu_bytes(
    *,
    time_count: int,
    sat_visible_count: int,
    output_dtype: Any,
) -> int:
    time_count_i = int(max(0, time_count))
    sat_visible_i = int(max(0, sat_visible_count))
    total = 0
    total += _shape_nbytes((sat_visible_i,), np.int32)
    total += _shape_nbytes((time_count_i, sat_visible_i, 4), output_dtype)
    total += _shape_nbytes((time_count_i, sat_visible_i, 3), output_dtype)
    total += _shape_nbytes((sat_visible_i,), np.float32)
    return int(total)


def _estimate_direct_epfd_link_library_gpu_bytes(
    *,
    time_count: int,
    cell_count: int,
    sat_count_total: int,
    sat_visible_count: int,
    n_skycells: int,
    store_eligible_mask: bool,
    boresight_active: bool,
) -> dict[str, int]:
    time_count_i = int(max(0, time_count))
    cell_count_i = int(max(0, cell_count))
    sat_total_i = int(max(0, sat_count_total))
    sat_visible_i = int(max(0, min(sat_visible_count, sat_total_i)))
    sky_total_i = int(max(1, n_skycells))
    candidate_pairs = int(time_count_i * cell_count_i * sat_visible_i)
    candidate_part_bytes_per_pair = (
        np.dtype(np.int32).itemsize * 3
        + np.dtype(np.float64).itemsize
        + np.dtype(np.float32).itemsize * 2
    )
    candidate_part_bytes = int(candidate_pairs * candidate_part_bytes_per_pair)
    packed_base_bytes_per_pair = candidate_part_bytes_per_pair + np.dtype(np.float32).itemsize * 2
    packed_base_bytes = int(candidate_pairs * packed_base_bytes_per_pair)
    selector_view_bytes = int(
        candidate_pairs * np.dtype(np.int32).itemsize
        + time_count_i * np.dtype(np.int64).itemsize * 2
    )
    direct_view_bytes = int(
        candidate_pairs * np.dtype(np.int32).itemsize
        + (time_count_i * cell_count_i + 1) * np.dtype(np.int64).itemsize
    )
    eligible_mask_bytes = (
        _shape_nbytes((time_count_i, cell_count_i, sat_total_i), np.bool_)
        if store_eligible_mask
        else 0
    )
    boresight_mask_bytes = (
        2 * _shape_nbytes((time_count_i, sky_total_i, sat_total_i), np.bool_)
        if boresight_active
        else 0
    )
    chunk_visible_mask_bytes = _shape_nbytes((time_count_i, cell_count_i, sat_total_i), np.bool_)
    chunk_theta_abs_bytes = _shape_nbytes((time_count_i, cell_count_i, sat_total_i), np.float32)
    chunk_eligible_temp_bytes = int(chunk_visible_mask_bytes if store_eligible_mask else 0)
    chunk_index_scratch_bytes = int(
        candidate_pairs
        * (
            np.dtype(np.int64).itemsize
            + np.dtype(np.int32).itemsize * 4
        )
    )
    chunk_transient_peak_bytes = int(
        chunk_visible_mask_bytes
        + chunk_theta_abs_bytes
        + chunk_eligible_temp_bytes
        + chunk_index_scratch_bytes
    )
    resident_bytes = int(candidate_part_bytes + eligible_mask_bytes)
    finalize_pack_peak_bytes = int(
        resident_bytes
        + packed_base_bytes
        + selector_view_bytes
        + direct_view_bytes
        + boresight_mask_bytes
    )
    finalize_resident_bytes = int(
        packed_base_bytes
        + selector_view_bytes
        + direct_view_bytes
        + eligible_mask_bytes
        + boresight_mask_bytes
    )
    return {
        "candidate_pairs": int(candidate_pairs),
        "candidate_part_bytes_per_pair": int(candidate_part_bytes_per_pair),
        "candidate_part_bytes": int(candidate_part_bytes),
        "packed_base_bytes_per_pair": int(packed_base_bytes_per_pair),
        "packed_base_bytes": int(packed_base_bytes),
        "selector_view_bytes": int(selector_view_bytes),
        "direct_view_bytes": int(direct_view_bytes),
        "eligible_mask_bytes": int(eligible_mask_bytes),
        "boresight_mask_bytes": int(boresight_mask_bytes),
        "chunk_visible_mask_bytes": int(chunk_visible_mask_bytes),
        "chunk_theta_abs_bytes": int(chunk_theta_abs_bytes),
        "chunk_eligible_temp_bytes": int(chunk_eligible_temp_bytes),
        "chunk_index_scratch_bytes": int(chunk_index_scratch_bytes),
        "chunk_transient_peak_bytes": int(chunk_transient_peak_bytes),
        "resident_bytes": int(resident_bytes),
        "finalize_pack_peak_bytes": int(finalize_pack_peak_bytes),
        "finalize_resident_bytes": int(finalize_resident_bytes),
    }


def _estimate_direct_epfd_finalize_accumulator_gpu_bytes(
    *,
    time_count: int,
    sat_count_total: int,
    n_skycells: int,
    boresight_active: bool,
    write_sat_beam_counts_used: bool,
    profile_stages: bool,
    count_dtype: Any,
) -> int:
    time_count_i = int(max(0, time_count))
    sat_total_i = int(max(0, sat_count_total))
    sky_total_i = int(max(1, n_skycells if boresight_active else 1))
    total = 0
    if write_sat_beam_counts_used:
        if boresight_active:
            total += _shape_nbytes((time_count_i, sky_total_i, sat_total_i), count_dtype)
        else:
            total += _shape_nbytes((time_count_i, sat_total_i), count_dtype)
    if profile_stages:
        if boresight_active:
            total += 5 * _shape_nbytes((time_count_i, sky_total_i), np.int32)
        else:
            total += 5 * _shape_nbytes((time_count_i,), np.int32)
    return int(total)


def _estimate_direct_epfd_power_result_gpu_bytes(
    *,
    time_count: int,
    sat_visible_count: int,
    n_skycells: int,
    boresight_active: bool,
    write_epfd: bool,
    write_prx_total: bool,
    write_per_satellite_prx_ras_station: bool,
    write_prx_elevation_heatmap: bool,
    write_total_pfd_ras_station: bool,
    write_per_satellite_pfd_ras_station: bool,
) -> int:
    time_count_i = int(max(0, time_count))
    sat_visible_i = int(max(0, sat_visible_count))
    sky_total_i = int(max(1, n_skycells if boresight_active else 1))
    total = 0
    if write_epfd:
        total += _shape_nbytes((time_count_i, sky_total_i), np.float32)
    if write_prx_total:
        total += _shape_nbytes((time_count_i, sky_total_i), np.float32)
    if write_per_satellite_prx_ras_station or write_prx_elevation_heatmap:
        if boresight_active:
            total += _shape_nbytes((time_count_i, sat_visible_i, sky_total_i), np.float32)
        else:
            total += _shape_nbytes((time_count_i, sat_visible_i), np.float32)
    if write_total_pfd_ras_station:
        total += (
            _shape_nbytes((time_count_i, sky_total_i), np.float32)
            if boresight_active
            else _shape_nbytes((time_count_i,), np.float32)
        )
    if write_per_satellite_pfd_ras_station:
        if boresight_active:
            total += _shape_nbytes((time_count_i, sat_visible_i, sky_total_i), np.float32)
        else:
            total += _shape_nbytes((time_count_i, sat_visible_i), np.float32)
    return int(total)


def _estimate_direct_epfd_slabbed_stage_peak_bytes(
    *,
    batch_timesteps: int,
    bytes_per_timestep: int,
    n_skycells: int,
    boresight_active: bool,
    sky_slab: int,
    safety_factor: float = 1.10,
    fixed_overhead_bytes: int = 16 * 1024 * 1024,
) -> int:
    batch_timesteps_i = int(max(1, batch_timesteps))
    bytes_per_timestep_i = int(max(0, bytes_per_timestep))
    if boresight_active:
        sky_total_i = int(max(1, n_skycells))
        slab_i = int(max(1, min(int(sky_slab), sky_total_i)))
        stage_bytes = int(
            np.ceil(
                float(batch_timesteps_i)
                * float(bytes_per_timestep_i)
                * (float(slab_i) / float(sky_total_i))
            )
        )
    else:
        stage_bytes = int(batch_timesteps_i * bytes_per_timestep_i)
    if stage_bytes <= 0:
        return int(max(1, fixed_overhead_bytes))
    return int(np.ceil(float(stage_bytes) * float(safety_factor))) + int(max(0, fixed_overhead_bytes))


def _estimate_direct_epfd_combined_gpu_peaks(
    *,
    batch_timesteps: int,
    cell_count: int,
    sat_count_total: int,
    sat_visible_count: int,
    n_skycells: int,
    boresight_active: bool,
    need_pointings: bool,
    need_beam_demand: bool,
    store_eligible_mask: bool,
    profile_stages: bool,
    output_dtype: Any,
    compute_dtype: Any,
    count_dtype: Any,
    demand_count_dtype: Any,
    predicted_gpu_cell_chunk_peak_bytes: int,
    predicted_gpu_finalize_slab_bytes: int,
    predicted_gpu_power_slab_bytes: int,
    predicted_gpu_export_peak_bytes: int,
    write_epfd: bool,
    write_prx_total: bool,
    write_per_satellite_prx_ras_station: bool,
    write_prx_elevation_heatmap: bool,
    write_total_pfd_ras_station: bool,
    write_per_satellite_pfd_ras_station: bool,
    write_sat_beam_counts_used: bool,
    spectrum_context_bytes: int = 0,
    activity_gpu_bytes: int = 0,
    activity_gpu_resident_bytes: int | None = None,
    activity_gpu_peak_bytes: int | None = None,
    uemr_mode: bool = False,
) -> dict[str, int]:
    orbit_state_bytes = _estimate_direct_epfd_orbit_state_gpu_bytes(
        time_count=int(batch_timesteps),
        sat_count_total=int(sat_count_total),
        compute_dtype=compute_dtype,
    )
    setup_bytes = _estimate_direct_epfd_setup_gpu_bytes(
        time_count=int(batch_timesteps),
        cell_count=int(cell_count),
        sat_count_total=int(sat_count_total),
        n_skycells=int(n_skycells),
        output_dtype=output_dtype,
        need_pointings=bool(need_pointings),
        need_beam_demand=bool(need_beam_demand),
        demand_count_dtype=demand_count_dtype,
    )
    visible_bytes = _estimate_direct_epfd_visible_resident_gpu_bytes(
        time_count=int(batch_timesteps),
        sat_visible_count=int(sat_visible_count),
        output_dtype=output_dtype,
    )
    if uemr_mode:
        # UEMR bypass: no beam library, no finalize. The link-library and
        # finalize-accumulator bytes are pure waste in the budget — they
        # never get allocated. Zero them so the planner can fit far more
        # timesteps per batch (the user observed many-small-batches
        # behaviour caused by these overestimates). Keep all the dict
        # keys the downstream code reads — just zero them.
        link_library_bytes = {
            "candidate_pairs": 0,
            "candidate_part_bytes_per_pair": 0,
            "candidate_part_bytes": 0,
            "packed_base_bytes_per_pair": 0,
            "packed_base_bytes": 0,
            "selector_view_bytes": 0,
            "direct_view_bytes": 0,
            "eligible_mask_bytes": 0,
            "boresight_mask_bytes": 0,
            "chunk_visible_mask_bytes": 0,
            "chunk_theta_abs_bytes": 0,
            "chunk_eligible_temp_bytes": 0,
            "chunk_index_scratch_bytes": 0,
            "chunk_transient_peak_bytes": 0,
            "resident_bytes": 0,
            "finalize_pack_peak_bytes": 0,
            "finalize_resident_bytes": 0,
        }
        finalize_accumulator_bytes = 0
    else:
        link_library_bytes = _estimate_direct_epfd_link_library_gpu_bytes(
            time_count=int(batch_timesteps),
            cell_count=int(cell_count),
            sat_count_total=int(sat_count_total),
            sat_visible_count=int(sat_visible_count),
            n_skycells=int(n_skycells),
            store_eligible_mask=bool(store_eligible_mask),
            boresight_active=bool(boresight_active),
        )
        finalize_accumulator_bytes = _estimate_direct_epfd_finalize_accumulator_gpu_bytes(
            time_count=int(batch_timesteps),
            sat_count_total=int(sat_count_total),
            n_skycells=int(n_skycells),
            boresight_active=bool(boresight_active),
            write_sat_beam_counts_used=bool(write_sat_beam_counts_used),
            profile_stages=bool(profile_stages),
            count_dtype=count_dtype,
        )
    power_result_bytes = _estimate_direct_epfd_power_result_gpu_bytes(
        time_count=int(batch_timesteps),
        sat_visible_count=int(sat_visible_count),
        n_skycells=int(n_skycells),
        boresight_active=bool(boresight_active),
        write_epfd=bool(write_epfd),
        write_prx_total=bool(write_prx_total),
        write_per_satellite_prx_ras_station=bool(write_per_satellite_prx_ras_station),
        write_prx_elevation_heatmap=bool(write_prx_elevation_heatmap),
        write_total_pfd_ras_station=bool(write_total_pfd_ras_station),
        write_per_satellite_pfd_ras_station=bool(write_per_satellite_pfd_ras_station),
    )

    activity_resident_bytes = int(
        max(
            0,
            activity_gpu_bytes if activity_gpu_resident_bytes is None else activity_gpu_resident_bytes,
        )
    )
    activity_stage_peak_bytes = int(
        max(
            0,
            activity_gpu_bytes if activity_gpu_peak_bytes is None else activity_gpu_peak_bytes,
        )
    )
    base_without_orbit = (
        int(setup_bytes)
        + int(max(0, spectrum_context_bytes))
        + int(activity_resident_bytes)
    )
    activity_peak = int(
        int(setup_bytes)
        + int(max(0, spectrum_context_bytes))
        + int(activity_stage_peak_bytes)
    )
    link_library_chunk_transient_peak_bytes = int(
        link_library_bytes["chunk_transient_peak_bytes"]
    )
    finalize_pack_overhang_bytes = int(
        max(
            0,
            int(link_library_bytes["finalize_pack_peak_bytes"])
            - int(link_library_bytes["finalize_resident_bytes"]),
        )
    )
    cell_link_peak = int(
        predicted_gpu_cell_chunk_peak_bytes
        + base_without_orbit
        + link_library_bytes["resident_bytes"]
        + link_library_chunk_transient_peak_bytes
    )
    finalize_peak = int(
        orbit_state_bytes
        + base_without_orbit
        + visible_bytes
        + link_library_bytes["finalize_pack_peak_bytes"]
        + finalize_accumulator_bytes
        + int(predicted_gpu_finalize_slab_bytes)
    )
    power_peak = int(
        orbit_state_bytes
        + base_without_orbit
        + visible_bytes
        + link_library_bytes["finalize_resident_bytes"]
        + finalize_accumulator_bytes
        + power_result_bytes
        + int(predicted_gpu_finalize_slab_bytes)
        + int(predicted_gpu_power_slab_bytes)
    )
    export_peak = int(
        orbit_state_bytes
        + base_without_orbit
        + visible_bytes
        + finalize_accumulator_bytes
        + power_result_bytes
        + int(predicted_gpu_export_peak_bytes)
    )
    return {
        "orbit_state_bytes": int(orbit_state_bytes),
        "setup_bytes": int(setup_bytes),
        "spectrum_context_bytes": int(max(0, spectrum_context_bytes)),
        "activity_gpu_bytes": int(activity_stage_peak_bytes),
        "activity_gpu_resident_bytes": int(activity_resident_bytes),
        "predicted_gpu_activity_stage_peak_bytes": int(activity_peak),
        "visible_bytes": int(visible_bytes),
        "link_library_resident_bytes": int(link_library_bytes["resident_bytes"]),
        "link_library_chunk_transient_peak_bytes": int(
            link_library_chunk_transient_peak_bytes
        ),
        "link_library_finalize_pack_peak_bytes": int(link_library_bytes["finalize_pack_peak_bytes"]),
        "link_library_finalize_pack_overhang_bytes": int(finalize_pack_overhang_bytes),
        "finalize_accumulator_bytes": int(finalize_accumulator_bytes),
        "power_result_bytes": int(power_result_bytes),
        "predicted_gpu_cell_link_peak_bytes": int(cell_link_peak),
        "predicted_gpu_finalize_peak_bytes": int(finalize_peak),
        "predicted_gpu_finalize_transient_peak_bytes": int(
            int(predicted_gpu_finalize_slab_bytes) + int(finalize_pack_overhang_bytes)
        ),
        "predicted_gpu_power_peak_bytes": int(power_peak),
        "predicted_gpu_export_peak_bytes": int(export_peak),
        "predicted_gpu_peak_bytes": int(
            max(activity_peak, cell_link_peak, finalize_peak, power_peak, export_peak)
        ),
    }


def _resolve_direct_epfd_live_stage_budget_bytes(
    *,
    effective_budget_bytes: int,
    scheduler_active_target_fraction: float,
    live_available_bytes: int | None,
) -> dict[str, int]:
    hard_budget = int(max(1, effective_budget_bytes))
    advisory_budget = max(
        1,
        min(
            hard_budget,
            int(np.floor(float(hard_budget) * float(scheduler_active_target_fraction))),
        ),
    )
    live_fit_budget = int(hard_budget)
    if live_available_bytes is not None:
        live_fit_raw = max(1, min(hard_budget, int(max(1, live_available_bytes))))
        # On Windows WDDM the reported free memory fluctuates by 200-500 MB
        # due to the desktop compositor and driver overhead.  Clamp the
        # live-fit floor to at least 85% of the hard budget so a momentary
        # free-bytes dip does not trigger a false OOM that collapses the
        # scheduler into degenerate shapes.  On Linux the CUDA driver
        # reports stable values so the raw live-fit is used as-is.
        if sys.platform == "win32":
            _LIVE_FIT_FLOOR_FRACTION = 0.85
            live_fit_floor = int(
                max(1, int(np.floor(float(hard_budget) * _LIVE_FIT_FLOOR_FRACTION)))
            )
            live_fit_budget = max(live_fit_floor, live_fit_raw)
        else:
            live_fit_budget = live_fit_raw
    return {
        "hard_budget_bytes": int(hard_budget),
        "advisory_budget_bytes": int(advisory_budget),
        "live_fit_budget_bytes": int(live_fit_budget),
    }


def _raise_if_direct_epfd_stage_live_fit_is_unsafe(
    *,
    stage: str,
    host_peak_bytes: int | None,
    gpu_peak_bytes: int | None,
    host_effective_budget_bytes: int,
    gpu_effective_budget_bytes: int,
    scheduler_active_target_fraction: float,
    live_host_snapshot: Mapping[str, Any] | None,
    live_gpu_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    host_live_budget = _resolve_direct_epfd_live_stage_budget_bytes(
        effective_budget_bytes=int(host_effective_budget_bytes),
        scheduler_active_target_fraction=float(scheduler_active_target_fraction),
        live_available_bytes=(
            None
            if live_host_snapshot is None
            else live_host_snapshot.get("available_bytes")
        ),
    )
    gpu_live_budget = _resolve_direct_epfd_live_stage_budget_bytes(
        effective_budget_bytes=int(gpu_effective_budget_bytes),
        scheduler_active_target_fraction=float(scheduler_active_target_fraction),
        live_available_bytes=(
            None
            if live_gpu_snapshot is None
            else live_gpu_snapshot.get("free_bytes")
        ),
    )
    hard_issues: list[str] = []
    live_fit_issues: list[str] = []
    advisory_issues: list[str] = []
    if host_peak_bytes is not None:
        host_peak = int(host_peak_bytes)
        if host_peak > int(host_live_budget["hard_budget_bytes"]):
            hard_issues.append(
                "host "
                f"{format_byte_count(host_peak)} > {format_byte_count(int(host_live_budget['hard_budget_bytes']))}"
            )
        elif host_peak > int(host_live_budget["live_fit_budget_bytes"]):
            live_fit_issues.append(
                "host "
                f"{format_byte_count(host_peak)} > {format_byte_count(int(host_live_budget['live_fit_budget_bytes']))}"
            )
        elif host_peak > int(host_live_budget["advisory_budget_bytes"]):
            advisory_issues.append(
                "host "
                f"{format_byte_count(host_peak)} > {format_byte_count(int(host_live_budget['advisory_budget_bytes']))}"
            )
    if gpu_peak_bytes is not None:
        gpu_peak = int(gpu_peak_bytes)
        if gpu_peak > int(gpu_live_budget["hard_budget_bytes"]):
            hard_issues.append(
                "gpu "
                f"{format_byte_count(gpu_peak)} > {format_byte_count(int(gpu_live_budget['hard_budget_bytes']))}"
            )
        elif gpu_peak > int(gpu_live_budget["live_fit_budget_bytes"]):
            live_fit_issues.append(
                "gpu "
                f"{format_byte_count(gpu_peak)} > {format_byte_count(int(gpu_live_budget['live_fit_budget_bytes']))}"
            )
        elif gpu_peak > int(gpu_live_budget["advisory_budget_bytes"]):
            advisory_issues.append(
                "gpu "
                f"{format_byte_count(gpu_peak)} > {format_byte_count(int(gpu_live_budget['advisory_budget_bytes']))}"
            )
    if hard_issues or live_fit_issues:
        detail_parts: list[str] = []
        if hard_issues:
            detail_parts.append(
                "Hard budget exceeded before "
                f"{stage}: " + "; ".join(hard_issues)
            )
        if live_fit_issues:
            detail_parts.append(
                "Live allocatable memory below required fit before "
                f"{stage}: " + "; ".join(live_fit_issues)
            )
        raise _DirectGpuOutOfMemory(
            str(stage),
            RuntimeError(". ".join(detail_parts) + ". Replanning with a lower scheduler target."),
        )
    return {
        "stage": str(stage),
        "host_hard_budget_bytes": int(host_live_budget["hard_budget_bytes"]),
        "host_advisory_budget_bytes": int(host_live_budget["advisory_budget_bytes"]),
        "host_live_fit_budget_bytes": int(host_live_budget["live_fit_budget_bytes"]),
        "gpu_hard_budget_bytes": int(gpu_live_budget["hard_budget_bytes"]),
        "gpu_advisory_budget_bytes": int(gpu_live_budget["advisory_budget_bytes"]),
        "gpu_live_fit_budget_bytes": int(gpu_live_budget["live_fit_budget_bytes"]),
        "advisory_issues": tuple(advisory_issues),
    }


def _resolve_direct_epfd_stage_working_budgets(
    *,
    effective_budget_bytes: int,
    batch_timesteps: int,
    export_bytes_per_timestep: int,
    finalize_memory_budget_bytes: int | None,
    power_memory_budget_bytes: int | None,
    export_memory_budget_bytes: int | None,
    host_effective_budget_bytes: int | None = None,
) -> dict[str, int]:
    effective_budget_i = int(max(1, effective_budget_bytes))
    # Export uses HOST memory (D2H copy + HDF5 write), not GPU memory.
    # Give export a proper share of the host budget, not the GPU budget.
    host_budget_i = int(max(1, host_effective_budget_bytes)) if host_effective_budget_bytes is not None else effective_budget_i
    if export_memory_budget_bytes is not None:
        export_budget = int(export_memory_budget_bytes)
    else:
        # Reserve up to 50% of host budget for export buffering, capped at actual need
        max_export_need = int(export_bytes_per_timestep) * int(batch_timesteps)
        export_budget = max(
            32 * 1024 * 1024,
            min(max_export_need, host_budget_i // 2),
        )
    export_budget = max(0, min(export_budget, host_budget_i))
    # GPU stages (finalize, power) use the full GPU budget minus a small reserve
    reserve_bytes = min(128 * 1024 * 1024, max(0, effective_budget_i // 8))
    working_default = max(64 * 1024 * 1024, effective_budget_i - reserve_bytes)
    return {
        "export_memory_budget_bytes": int(export_budget),
        "finalize_memory_budget_bytes": int(
            working_default if finalize_memory_budget_bytes is None else max(1, int(finalize_memory_budget_bytes))
        ),
        "power_memory_budget_bytes": int(
            working_default if power_memory_budget_bytes is None else max(1, int(power_memory_budget_bytes))
        ),
    }


def _estimate_direct_epfd_stage_sky_slab(
    *,
    working_budget_bytes: int,
    bytes_per_timestep: int,
    n_skycells: int,
    boresight_active: bool,
    explicit_sky_slab: int | None = None,
) -> int:
    if not boresight_active:
        return 1
    sky_total = int(max(1, n_skycells))
    if explicit_sky_slab is not None:
        return max(1, min(int(explicit_sky_slab), sky_total))
    bytes_per_time_sky = max(1, int(np.ceil(float(max(0, bytes_per_timestep)) / float(sky_total))))
    return max(1, min(sky_total, int(max(1, working_budget_bytes) // bytes_per_time_sky)))


def _estimate_direct_epfd_batch_seconds(
    *,
    bulk_timesteps: int,
    cell_chunk: int,
    visible_satellite_est: int,
    nco: int,
    nbeam: int,
    n_skycells: int,
    boresight_active: bool,
    sky_slab: int,
    output_family_plan: Mapping[str, bool],
    multi_system_count: int = 1,
    multi_system_uemr_count: int = 0,
    uemr_mode: bool = False,
) -> float:
    work_units = float(max(1, bulk_timesteps)) * float(max(1, cell_chunk))
    work_units *= max(1.0, float(max(1, visible_satellite_est)) * float(max(1, nco)) / 8.0)
    work_units *= max(1.0, float(max(1, nbeam)) / 16.0)
    if boresight_active:
        work_units *= max(1.0, float(max(1, min(int(sky_slab), int(max(1, n_skycells))))) / 8.0)
    # UEMR bypass: no cell-link library, no beam-finalize, no surface-PFD
    # cap — just the per-satellite kernel on visible sats. Empirically
    # 4-6× cheaper than the directive path, so scale work_units down to
    # ~20% of the directive estimate. The scheduler's ETA uses this
    # value directly; without the factor the ETA overshoots by 5×.
    if bool(uemr_mode):
        work_units *= 0.20
    # Multi-system scaling: shared propagation, but the power stage and
    # bookkeeping run once per additional system. Empirically each extra
    # directive system adds ~7%, UEMR systems ~2.5% (they bypass the beam
    # library). See [scenario.py:10548] for the prior post-hoc scale factor.
    n_total = max(1, int(multi_system_count))
    n_uemr = max(0, int(multi_system_uemr_count))
    n_directive = max(0, n_total - n_uemr)
    if n_total > 1:
        n_dir_extra = max(0, n_directive - 1) if n_directive >= 1 else 0
        n_uemr_extra = n_uemr if n_directive >= 1 else max(0, n_uemr - 1)
        ms_scale = 1.0 + 0.07 * float(n_dir_extra) + 0.025 * float(n_uemr_extra)
        work_units *= ms_scale
    output_multiplier = 1.0
    if bool(output_family_plan.get("needs_epfd")):
        output_multiplier += 0.15
    if bool(output_family_plan.get("needs_total_prx")):
        output_multiplier += 0.10
    if bool(output_family_plan.get("needs_per_satellite_prx")):
        output_multiplier += 0.20
    if bool(output_family_plan.get("needs_total_pfd")):
        output_multiplier += 0.10
    if bool(output_family_plan.get("needs_per_satellite_pfd")):
        output_multiplier += 0.15
    if bool(output_family_plan.get("needs_beam_counts")):
        output_multiplier += 0.08
    if bool(output_family_plan.get("needs_beam_demand")):
        output_multiplier += 0.05
    return max(1.0, (work_units * output_multiplier) / 2.0e6)


def _build_direct_epfd_iteration_candidate_record(
    *,
    candidate_bulk: int,
    cell_chunk: int,
    sky_slab: int,
    spectral_slab: int,
    cells_total_i: int,
    steps_total_i: int,
    sky_total_i: int,
    sats_total_i: int,
    visible_satellite_est: int,
    boresight_active: bool,
    nco: int,
    nbeam: int,
    need_pointings: bool,
    need_power_outputs: bool,
    write_prx_elevation_heatmap: bool,
    output_family_plan: Mapping[str, bool],
    store_eligible_mask: bool,
    profile_stages: bool,
    gpu_output_dtype: Any,
    compute_dtype: Any,
    count_dtype: Any,
    demand_count_dtype: Any,
    predicted_host_link_peak_bytes: int,
    predicted_host_export_peak_bytes: int,
    predicted_gpu_cell_link_peak_bytes: int,
    predicted_gpu_finalize_slab_bytes: int,
    predicted_gpu_power_slab_bytes: int,
    host_effective_budget: int,
    host_working_budget: int,
    gpu_effective_budget: int,
    gpu_working_budget: int,
    host_recommendation: Mapping[str, Any],
    gpu_recommendation: Mapping[str, Any],
    finalize_recommendation: Mapping[str, Any],
    power_recommendation: Mapping[str, Any],
    export_recommendation: Mapping[str, Any],
    spectrum_context_bytes: int,
    cell_activity_mode: str,
    activity_groups_per_cell: int,
    activity_power_policy: str,
    activity_split_total_group_denominator_mode: str,
    planner_source: str,
    anchor_obs_fraction: float,
    anchor_time_fraction: float,
    force_bulk_timesteps: int | None,
    multi_system_count: int = 1,
    multi_system_uemr_count: int = 0,
) -> dict[str, Any] | None:
    # When the entire workload is UEMR (single-system UEMR or every system
    # in the multi-system list is UEMR) the beam library + finalize stages
    # are bypassed at run time. Tell the memory model so it doesn't
    # over-budget those workspaces — that overestimate caused the planner
    # to pick small bulk_timesteps and produce many small batches even
    # though UEMR has plenty of GPU headroom.
    _uemr_primary = bool(multi_system_uemr_count >= max(1, int(multi_system_count)))
    predicted_host_peak_bytes = int(
        int(predicted_host_link_peak_bytes) + int(predicted_host_export_peak_bytes)
    )
    activity_memory = _estimate_direct_epfd_activity_gpu_memory(
        time_count=int(candidate_bulk),
        cell_count=int(cells_total_i),
        groups_per_cell=int(activity_groups_per_cell),
        cell_activity_mode=str(cell_activity_mode),
        need_power_outputs=bool(need_power_outputs),
        spectral_slab=int(spectral_slab),
        power_policy=str(activity_power_policy),
        split_total_group_denominator_mode=str(
            activity_split_total_group_denominator_mode
        ),
    )
    combined_gpu_peaks = _estimate_direct_epfd_combined_gpu_peaks(
        batch_timesteps=int(candidate_bulk),
        cell_count=int(cells_total_i),
        sat_count_total=int(sats_total_i),
        sat_visible_count=int(visible_satellite_est),
        n_skycells=int(sky_total_i),
        boresight_active=bool(boresight_active),
        need_pointings=bool(need_pointings),
        need_beam_demand=bool(output_family_plan["needs_beam_demand"]),
        store_eligible_mask=bool(store_eligible_mask),
        profile_stages=bool(profile_stages),
        output_dtype=gpu_output_dtype,
        compute_dtype=compute_dtype,
        count_dtype=count_dtype,
        demand_count_dtype=demand_count_dtype,
        predicted_gpu_cell_chunk_peak_bytes=int(predicted_gpu_cell_link_peak_bytes),
        predicted_gpu_finalize_slab_bytes=int(predicted_gpu_finalize_slab_bytes),
        predicted_gpu_power_slab_bytes=int(predicted_gpu_power_slab_bytes),
        predicted_gpu_export_peak_bytes=0,
        write_epfd=bool(output_family_plan["needs_epfd"]),
        write_prx_total=bool(output_family_plan["needs_total_prx"]),
        write_per_satellite_prx_ras_station=bool(output_family_plan["needs_per_satellite_prx"]),
        write_prx_elevation_heatmap=bool(write_prx_elevation_heatmap),
        write_total_pfd_ras_station=bool(output_family_plan["needs_total_pfd"]),
        write_per_satellite_pfd_ras_station=bool(output_family_plan["needs_per_satellite_pfd"]),
        write_sat_beam_counts_used=bool(output_family_plan["needs_beam_counts"]),
        spectrum_context_bytes=int(spectrum_context_bytes),
        activity_gpu_resident_bytes=int(activity_memory["resident_bytes"]),
        activity_gpu_peak_bytes=int(activity_memory["peak_bytes"]),
        uemr_mode=bool(_uemr_primary),
    )
    predicted_gpu_propagation_peak_bytes = int(combined_gpu_peaks["predicted_gpu_cell_link_peak_bytes"])
    predicted_gpu_finalize_peak_bytes = int(combined_gpu_peaks["predicted_gpu_finalize_peak_bytes"])
    predicted_gpu_power_peak_bytes = int(combined_gpu_peaks["predicted_gpu_power_peak_bytes"])
    predicted_gpu_export_peak_bytes = int(combined_gpu_peaks["predicted_gpu_export_peak_bytes"])
    predicted_gpu_activity_peak_bytes = int(
        combined_gpu_peaks["predicted_gpu_activity_stage_peak_bytes"]
    )
    predicted_gpu_link_library_resident_bytes = int(
        combined_gpu_peaks["link_library_resident_bytes"]
    )
    predicted_gpu_link_library_transient_peak_bytes = int(
        combined_gpu_peaks["link_library_chunk_transient_peak_bytes"]
    )
    predicted_gpu_finalize_transient_peak_bytes = int(
        combined_gpu_peaks["predicted_gpu_finalize_transient_peak_bytes"]
    )
    predicted_gpu_peak_bytes = int(combined_gpu_peaks["predicted_gpu_peak_bytes"])

    if predicted_host_peak_bytes > int(host_effective_budget):
        return None
    if predicted_gpu_peak_bytes > int(gpu_effective_budget):
        return None

    n_batch_estimate = int(np.ceil(float(steps_total_i) / float(max(1, int(candidate_bulk)))))
    n_chunk_estimate = int(np.ceil(float(cells_total_i) / float(max(1, int(cell_chunk)))))
    host_fill = min(1.0, float(predicted_host_peak_bytes) / float(max(1, host_working_budget)))
    gpu_fill = min(1.0, float(predicted_gpu_peak_bytes) / float(max(1, gpu_working_budget)))
    compute_budget_utilization_fraction = min(
        1.0,
        float(
            max(
                int(predicted_gpu_propagation_peak_bytes),
                int(predicted_gpu_finalize_peak_bytes),
                int(predicted_gpu_power_peak_bytes),
                int(predicted_gpu_activity_peak_bytes),
            )
        )
        / float(max(1, gpu_working_budget)),
    )
    export_budget_utilization_fraction = min(
        1.0,
        float(max(0, int(predicted_host_export_peak_bytes)))
        / float(max(1, host_working_budget)),
    )
    cell_fraction = float(int(cell_chunk)) / float(cells_total_i)
    time_fraction = float(int(candidate_bulk)) / float(steps_total_i)
    sky_fraction = (
        float(int(sky_slab)) / float(sky_total_i)
        if boresight_active
        else 1.0
    )
    throughput_units = float(int(candidate_bulk)) * float(int(cell_chunk)) * float(max(1, int(sky_slab)))
    shape_factor = (
        0.45
        + 0.55
        * np.sqrt(max(cell_fraction, 1.0 / float(cells_total_i)))
        * np.sqrt(max(time_fraction, 1.0 / float(steps_total_i)))
        * np.sqrt(max(sky_fraction, 1.0 / float(sky_total_i)))
    )
    anchor_distance = abs(cell_fraction - anchor_obs_fraction) + abs(time_fraction - anchor_time_fraction)
    anchor_factor = max(0.80, 1.12 - 0.22 * anchor_distance)
    score = throughput_units * (0.68 + 0.32 * shape_factor) * (0.84 + 0.16 * gpu_fill) * anchor_factor
    if boresight_active:
        score *= 0.85 + 0.15 * np.sqrt(max(sky_fraction, 1.0 / float(sky_total_i)))
    launch_fragmentation_penalty = max(
        0.18,
        1.0
        - 0.08 * np.log2(float(max(1, n_batch_estimate)))
        - 0.18 * np.log2(float(max(1, n_chunk_estimate))),
    )
    score *= float(launch_fragmentation_penalty)
    transient_spike_bytes = float(
        max(
            int(predicted_gpu_link_library_transient_peak_bytes),
            int(predicted_gpu_finalize_transient_peak_bytes),
        )
    )
    if transient_spike_bytes > 0.0:
        transient_fraction = min(
            1.0,
            transient_spike_bytes / float(max(1, int(predicted_gpu_peak_bytes))),
        )
        score *= max(0.72, 1.0 - 0.22 * transient_fraction)
    spectral_backoff_active = bool(
        _normalize_direct_epfd_cell_activity_mode(cell_activity_mode) == "per_channel"
        and int(max(1, activity_groups_per_cell)) > 1
        and int(activity_memory["spectral_slab"]) < int(candidate_bulk)
    )
    if spectral_backoff_active:
        spectral_fraction = float(int(activity_memory["spectral_slab"])) / float(max(1, int(candidate_bulk)))
        score *= 0.92 + 0.08 * np.sqrt(
            max(spectral_fraction, 1.0 / float(max(1, int(candidate_bulk))))
        )
    if int(cell_chunk) <= max(1, cells_total_i // 32):
        score *= 0.35
    elif int(cell_chunk) <= max(1, cells_total_i // 16):
        score *= 0.55
    if int(candidate_bulk) <= max(1, steps_total_i // 32):
        score *= 0.70
    if int(n_chunk_estimate) >= 32 and int(n_batch_estimate) <= 2:
        score *= 0.30

    limiting_resource = "host"
    limiting_dimension = "bulk_timesteps"
    limiting_value = int(host_recommendation["recommended_batch_size"])
    for label, value in (
        ("gpu-propagation", int(gpu_recommendation["recommended_batch_size"])),
        ("finalize-stage", int(finalize_recommendation["recommended_batch_size"])),
        ("power-stage", int(power_recommendation["recommended_batch_size"])),
        ("export-stage", int(export_recommendation["recommended_batch_size"])),
    ):
        if int(value) < limiting_value:
            limiting_resource = label
            limiting_value = int(value)
    if limiting_resource == "gpu-propagation":
        limiting_dimension = "cell_chunk"
    elif limiting_resource == "finalize-stage":
        limiting_dimension = "sky_slab" if bool(boresight_active) else "bulk_timesteps"
    if spectral_backoff_active and predicted_gpu_activity_peak_bytes >= max(
        predicted_gpu_propagation_peak_bytes,
        predicted_gpu_finalize_peak_bytes,
        predicted_gpu_power_peak_bytes,
        predicted_gpu_export_peak_bytes,
    ):
        limiting_resource = "spectral-activity"
        limiting_dimension = "spectral_slab"
    if force_bulk_timesteps is not None:
        limiting_resource = "forced"
        limiting_dimension = "bulk_timesteps"

    underfill_reason = "none"
    if int(n_chunk_estimate) >= 32 and int(n_batch_estimate) <= 2:
        underfill_reason = "tiny_cell_chunk_high_chunk_fanout"
    elif int(cell_chunk) <= max(1, cells_total_i // 16):
        underfill_reason = "cell_chunk_underfill"
    elif int(candidate_bulk) == 1 and int(n_batch_estimate) > 1:
        underfill_reason = "single_timestep_batches"
    elif limiting_resource == "export-stage":
        underfill_reason = "export_limited"
    elif limiting_resource == "host":
        underfill_reason = "host_limited"
    elif limiting_resource == "gpu-propagation":
        underfill_reason = "gpu_propagation_limited"
    elif limiting_resource == "finalize-stage":
        underfill_reason = "beam_finalize_limited"
    elif limiting_resource == "power-stage":
        underfill_reason = "power_limited"
    elif limiting_resource == "spectral-activity":
        underfill_reason = "spectral_activity_limited"

    # ``_uemr_primary`` was computed at function entry and reused below.
    planned_batch_seconds = _estimate_direct_epfd_batch_seconds(
        bulk_timesteps=int(candidate_bulk),
        cell_chunk=int(cell_chunk),
        visible_satellite_est=int(visible_satellite_est),
        nco=int(nco),
        nbeam=int(nbeam),
        n_skycells=int(sky_total_i),
        boresight_active=bool(boresight_active),
        sky_slab=int(sky_slab),
        output_family_plan=output_family_plan,
        multi_system_count=int(multi_system_count),
        multi_system_uemr_count=int(multi_system_uemr_count),
        uemr_mode=_uemr_primary,
    )
    if spectral_backoff_active:
        planned_batch_seconds *= 1.0 + 0.05 * float(
            max(
                0,
                int(np.ceil(float(candidate_bulk) / float(activity_memory["spectral_slab"]))) - 1,
            )
        )

    return {
        "bulk_timesteps": int(candidate_bulk),
        "cell_chunk": int(cell_chunk),
        "sky_slab": int(sky_slab),
        "spectral_slab": int(activity_memory["spectral_slab"]),
        "spectral_backoff_active": bool(spectral_backoff_active),
        "limiting_dimension": str(limiting_dimension),
        "score": float(score),
        "planner_source": str(planner_source),
        "predicted_host_peak_bytes": int(predicted_host_peak_bytes),
        "predicted_host_link_peak_bytes": int(predicted_host_link_peak_bytes),
        "predicted_host_export_peak_bytes": int(predicted_host_export_peak_bytes),
        "predicted_gpu_peak_bytes": int(predicted_gpu_peak_bytes),
        "predicted_gpu_propagation_peak_bytes": int(predicted_gpu_propagation_peak_bytes),
        "predicted_gpu_finalize_peak_bytes": int(predicted_gpu_finalize_peak_bytes),
        "predicted_gpu_power_peak_bytes": int(predicted_gpu_power_peak_bytes),
        "predicted_gpu_export_peak_bytes": int(predicted_gpu_export_peak_bytes),
        "predicted_gpu_activity_resident_bytes": int(activity_memory["resident_bytes"]),
        "predicted_gpu_activity_scratch_bytes": int(activity_memory["scratch_bytes"]),
        "predicted_gpu_activity_peak_bytes": int(predicted_gpu_activity_peak_bytes),
        "predicted_gpu_spectrum_context_bytes": int(max(0, spectrum_context_bytes)),
        "predicted_gpu_setup_bytes": int(combined_gpu_peaks["setup_bytes"]),
        "predicted_gpu_orbit_state_bytes": int(combined_gpu_peaks["orbit_state_bytes"]),
        "predicted_gpu_visible_bytes": int(combined_gpu_peaks["visible_bytes"]),
        "predicted_gpu_link_library_resident_bytes": int(
            predicted_gpu_link_library_resident_bytes
        ),
        "predicted_gpu_link_library_transient_peak_bytes": int(
            predicted_gpu_link_library_transient_peak_bytes
        ),
        "predicted_gpu_finalize_accumulator_bytes": int(
            combined_gpu_peaks["finalize_accumulator_bytes"]
        ),
        "predicted_gpu_beam_finalize_working_bytes": int(predicted_gpu_finalize_slab_bytes),
        "predicted_gpu_finalize_transient_peak_bytes": int(
            predicted_gpu_finalize_transient_peak_bytes
        ),
        "predicted_gpu_power_result_bytes": int(combined_gpu_peaks["power_result_bytes"]),
        "compute_budget_utilization_fraction": float(compute_budget_utilization_fraction),
        "export_budget_utilization_fraction": float(export_budget_utilization_fraction),
        "batch_count_estimate": int(n_batch_estimate),
        "chunk_count_estimate": int(n_chunk_estimate),
        "underfill_reason": str(underfill_reason),
        "limiting_resource": str(limiting_resource),
        "planned_batch_seconds": float(planned_batch_seconds),
    }


def _build_direct_epfd_scheduler_payload(
    *,
    host_budget_info: Mapping[str, Any],
    gpu_budget_info: Mapping[str, Any],
    scheduler_target_fraction: float,
    scheduler_active_target_fraction: float,
    boresight_active: bool,
    n_earthgrid_cells: int,
    n_skycells_s1586: int,
    visible_satellite_est: int,
    bulk_timesteps: int,
    cell_chunk: int,
    sky_slab: int,
    predicted_host_peak_bytes: int,
    predicted_gpu_peak_bytes: int,
    planner_source: str,
    limiting_resource: str,
    planned_total_seconds: float | None = None,
    planned_remaining_seconds: float | None = None,
    live_host_snapshot: Mapping[str, Any] | None = None,
    live_gpu_snapshot: Mapping[str, Any] | None = None,
    live_gpu_adapter_snapshot: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "boresight_active": bool(boresight_active),
        "n_earthgrid_cells": int(n_earthgrid_cells),
        "n_skycells_s1586": int(n_skycells_s1586),
        "visible_satellite_est": int(visible_satellite_est),
        "bulk_timesteps": int(bulk_timesteps),
        "cell_chunk": int(cell_chunk),
        "sky_slab": int(sky_slab),
        "host_hard_budget_bytes": int(
            host_budget_info.get("hard_budget_bytes", host_budget_info["effective_budget_bytes"])
        ),
        "host_planning_budget_bytes": int(
            host_budget_info.get("planning_budget_bytes", host_budget_info["effective_budget_bytes"])
        ),
        "host_runtime_advisory_budget_bytes": (
            None
            if host_budget_info.get("runtime_advisory_budget_bytes") is None
            else int(host_budget_info["runtime_advisory_budget_bytes"])
        ),
        "gpu_hard_budget_bytes": int(
            gpu_budget_info.get("hard_budget_bytes", gpu_budget_info["effective_budget_bytes"])
        ),
        "gpu_planning_budget_bytes": int(
            gpu_budget_info.get("planning_budget_bytes", gpu_budget_info["effective_budget_bytes"])
        ),
        "gpu_runtime_advisory_budget_bytes": (
            None
            if gpu_budget_info.get("runtime_advisory_budget_bytes") is None
            else int(gpu_budget_info["runtime_advisory_budget_bytes"])
        ),
        "gpu_budget_reason": str(gpu_budget_info.get("effective_budget_reason") or ""),
        "host_effective_budget_bytes": int(host_budget_info["effective_budget_bytes"]),
        "gpu_effective_budget_bytes": int(gpu_budget_info["effective_budget_bytes"]),
        "scheduler_target_fraction": float(scheduler_target_fraction),
        "scheduler_active_target_fraction": float(scheduler_active_target_fraction),
        "predicted_host_peak_bytes": int(predicted_host_peak_bytes),
        "predicted_gpu_peak_bytes": int(predicted_gpu_peak_bytes),
        "planner_source": str(planner_source),
        "limiting_resource": str(limiting_resource),
    }
    if planned_total_seconds is not None:
        payload["planned_total_seconds"] = float(planned_total_seconds)
    if planned_remaining_seconds is not None:
        payload["planned_remaining_seconds"] = float(planned_remaining_seconds)
    if live_host_snapshot is not None:
        total_bytes = live_host_snapshot.get("total_bytes")
        available_bytes = live_host_snapshot.get("available_bytes")
        if total_bytes is not None:
            payload["live_host_total_bytes"] = int(total_bytes)
        if available_bytes is not None:
            payload["live_host_available_bytes"] = int(available_bytes)
    if live_gpu_snapshot is not None:
        total_bytes = live_gpu_snapshot.get("total_bytes")
        free_bytes = live_gpu_snapshot.get("free_bytes")
        if total_bytes is not None:
            payload["live_gpu_total_bytes"] = int(total_bytes)
        if free_bytes is not None:
            payload["live_gpu_free_bytes"] = int(free_bytes)
    if live_gpu_adapter_snapshot is not None:
        total_bytes = live_gpu_adapter_snapshot.get("total_bytes")
        used_bytes = live_gpu_adapter_snapshot.get("used_bytes")
        free_bytes = live_gpu_adapter_snapshot.get("free_bytes")
        if total_bytes is not None:
            payload["live_gpu_adapter_total_bytes"] = int(total_bytes)
        if used_bytes is not None:
            payload["live_gpu_adapter_used_bytes"] = int(used_bytes)
        if free_bytes is not None:
            payload["live_gpu_adapter_free_bytes"] = int(free_bytes)
    if extra:
        payload.update(dict(extra))
    return payload


def _plan_direct_epfd_iteration_schedule(
    *,
    session: Any,
    observer_context: Any,
    satellite_context: Any,
    gpu_output_dtype: Any,
    n_steps_total: int,
    n_cells_total: int,
    n_sats_total: int,
    n_skycells_s1586: int,
    visible_satellite_est: int,
    nco: int,
    nbeam: int,
    boresight_active: bool,
    effective_ras_pointing_mode: str,
    output_family_plan: Mapping[str, bool],
    store_eligible_mask: bool,
    profile_stages: bool,
    host_budget_info: Mapping[str, Any],
    gpu_budget_info: Mapping[str, Any],
    scheduler_target_fraction: float,
    scheduler_active_target_fraction: float,
    finalize_memory_budget_bytes: int | None,
    power_memory_budget_bytes: int | None,
    export_memory_budget_bytes: int | None,
    power_sky_slab: int | None,
    force_bulk_timesteps: int | None,
    force_cell_observer_chunk: int | None,
    allow_warmup_calibration: bool,
    spectrum_context_bytes: int = 0,
    cell_activity_mode: str = "whole_cell",
    activity_groups_per_cell: int = 1,
    activity_power_policy: str = "repeat_per_group",
    activity_split_total_group_denominator_mode: str = "configured_groups",
    surface_pfd_cap_enabled: bool = False,
    surface_pfd_cap_mode: str = "per_beam",
    multi_system_count: int = 1,
    multi_system_uemr_count: int = 0,
) -> dict[str, Any]:
    steps_total_i = int(max(1, n_steps_total))
    cells_total_i = int(max(1, n_cells_total))
    sats_total_i = int(max(1, n_sats_total))
    sky_total_i = int(max(1, n_skycells_s1586 if boresight_active else 1))
    host_effective_budget = int(max(1, host_budget_info["effective_budget_bytes"]))
    gpu_effective_budget = int(max(1, gpu_budget_info["effective_budget_bytes"]))
    host_planning_budget = int(max(1, host_budget_info.get("planning_budget_bytes", host_effective_budget)))
    gpu_planning_budget = int(max(1, gpu_budget_info.get("planning_budget_bytes", gpu_effective_budget)))
    host_working_budget = max(
        1,
        min(
            host_planning_budget,
            int(np.floor(float(host_planning_budget) * float(scheduler_active_target_fraction))),
        ),
    )
    gpu_working_budget = max(
        1,
        min(
            gpu_planning_budget,
            int(np.floor(float(gpu_planning_budget) * float(scheduler_active_target_fraction))),
        ),
    )

    propagation_anchor_plan = session.plan_propagation_execution(
        np.zeros(steps_total_i, dtype=np.float64),
        satellite_context,
        observer_context=observer_context,
        observer_slice=slice(1, 1 + cells_total_i),
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=False,
        do_sat_azel=True,
        do_sat_rotmat=False,
        output_dtype=gpu_output_dtype,
        return_device=True,
        budget_bytes=int(gpu_effective_budget),
        scheduler_target_profile=float(scheduler_active_target_fraction),
        allow_warmup_calibration=bool(allow_warmup_calibration),
    )
    anchor_obs_fraction = float(
        max(1, int(propagation_anchor_plan.get("observer_chunk_size", cells_total_i)))
    ) / float(cells_total_i)
    anchor_time_fraction = float(
        max(1, int(propagation_anchor_plan.get("time_chunk_size", steps_total_i)))
    ) / float(steps_total_i)
    planner_source = str(propagation_anchor_plan.get("planner_source", "analytic_fallback"))

    if force_cell_observer_chunk is not None:
        cell_candidates = [
            max(1, min(int(force_cell_observer_chunk), cells_total_i))
        ]
    else:
        cell_candidates = _candidate_chunk_sizes(cells_total_i)

    compute_dtype = np.dtype(getattr(session, "compute_dtype", np.float32))
    power_dtype = np.dtype(getattr(session, "power_dtype", np.float32))
    pattern_dtype = np.dtype(getattr(session, "pattern_dtype", np.float32))
    need_pointings = bool(
        bool(boresight_active)
        or bool(output_family_plan.get("needs_epfd"))
        or bool(output_family_plan.get("needs_total_prx"))
        or bool(output_family_plan.get("needs_per_satellite_prx"))
    )
    write_prx_elevation_heatmap = bool(output_family_plan.get("preacc_prx_elevation_heatmap"))
    need_power_outputs = bool(
        output_family_plan.get("needs_epfd")
        or output_family_plan.get("needs_total_prx")
        or output_family_plan.get("needs_per_satellite_prx")
        or output_family_plan.get("needs_total_pfd")
        or output_family_plan.get("needs_per_satellite_pfd")
    )
    activity_mode_name = _normalize_direct_epfd_cell_activity_mode(cell_activity_mode)
    activity_power_policy = str(activity_power_policy)
    activity_split_total_group_denominator_mode = str(
        activity_split_total_group_denominator_mode
    )
    activity_slab_enabled = bool(
        activity_mode_name == "per_channel" and int(max(1, activity_groups_per_cell)) > 1
    )
    count_dtype = _direct_epfd_count_dtype(int(nbeam))
    demand_count_dtype = _beam_demand_count_dtype(cells_total_i, int(nco))

    power_per_timestep_bytes = _estimate_direct_epfd_power_bytes_per_timestep(
        sat_visible_count=visible_satellite_est,
        n_beam=int(nbeam),
        n_skycells=sky_total_i,
        boresight_active=boresight_active,
        include_epfd=bool(output_family_plan["needs_epfd"]),
        include_prx_total=bool(output_family_plan["needs_total_prx"]),
        include_per_satellite_prx=bool(output_family_plan["needs_per_satellite_prx"]),
        include_total_pfd=bool(output_family_plan["needs_total_pfd"]),
        include_per_satellite_pfd=bool(output_family_plan["needs_per_satellite_pfd"]),
        power_dtype=power_dtype,
        pattern_dtype=pattern_dtype,
        surface_pfd_cap_enabled=bool(surface_pfd_cap_enabled),
        surface_pfd_cap_mode=str(surface_pfd_cap_mode),
    )
    export_per_timestep_bytes = _estimate_direct_epfd_export_bytes_per_timestep(
        sat_count_total=sats_total_i,
        sat_visible_count=visible_satellite_est,
        n_skycells=sky_total_i,
        boresight_active=boresight_active,
        write_epfd=bool(output_family_plan["needs_epfd"]),
        write_prx_total=bool(output_family_plan["needs_total_prx"]),
        write_per_satellite_prx_ras_station=bool(output_family_plan["needs_per_satellite_prx"]),
        write_prx_elevation_heatmap=bool(write_prx_elevation_heatmap),
        write_total_pfd_ras_station=bool(output_family_plan["needs_total_pfd"]),
        write_per_satellite_pfd_ras_station=bool(output_family_plan["needs_per_satellite_pfd"]),
        write_sat_beam_counts_used=bool(output_family_plan["needs_beam_counts"]),
        write_sat_elevation_ras_station=False,
        write_beam_demand_count=bool(output_family_plan["needs_beam_demand"]),
        write_sat_eligible_mask=bool(store_eligible_mask),
        count_dtype=count_dtype,
        demand_count_dtype=demand_count_dtype,
        power_dtype=power_dtype,
    )
    finalize_per_timestep_bytes = _estimate_direct_epfd_finalize_bytes_per_timestep(
        cell_count=cells_total_i,
        sat_count_total=sats_total_i,
        sat_output_count=visible_satellite_est,
        n_links=int(nco),
        n_beam=int(nbeam),
        n_skycells=sky_total_i,
        boresight_active=boresight_active,
        ras_pointing_mode=effective_ras_pointing_mode,
        include_diagnostics=bool(profile_stages),
        pattern_dtype=pattern_dtype,
    )

    candidate_records: list[dict[str, Any]] = []
    for cell_chunk in cell_candidates:
        host_one_step_estimate = estimate_step1_host_batch_bytes(
            time_count=1,
            n_cells=cells_total_i,
            n_sats=sats_total_i,
            n_links=int(nco),
            cell_chunk_size=int(cell_chunk),
            store_eligible_mask=bool(store_eligible_mask),
            gpu_resident=True,
        )
        if int(host_one_step_estimate["peak_bytes"]) > host_effective_budget and int(cell_chunk) > 1:
            continue
        host_recommendation = recommend_time_batch_size_linear(
            total_timesteps=steps_total_i,
            fixed_bytes=int(host_one_step_estimate["fixed_bytes"]),
            per_timestep_bytes=int(host_one_step_estimate["per_timestep_bytes"]),
            budget_bytes=int(host_working_budget),
        )
        gpu_one_step_estimate = session.estimate_propagation_memory(
            np.zeros(1, dtype=np.float64),
            satellite_context,
            observer_context=observer_context,
            observer_slice=slice(1, 1 + int(cell_chunk)),
            do_eci_pos=False,
            do_eci_vel=False,
            do_geo=False,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=True,
            do_sat_rotmat=False,
            output_dtype=gpu_output_dtype,
            return_device=True,
        )
        gpu_fixed_bytes = int(
            gpu_one_step_estimate["cache_bytes"] + gpu_one_step_estimate["reserve_bytes"]
        )
        gpu_per_timestep_bytes_full = max(
            1,
            int(gpu_one_step_estimate["total_bytes"]) - gpu_fixed_bytes,
        )
        # Satellite pre-filtering: the cell-geometry kernel runs on the
        # visible-sat subset (~5-10 % of the full constellation), so the
        # dominant topo/sat_azel memory scales with visible_satellite_est
        # rather than sats_total_i.  Apply a 2x safety margin to account
        # for the -2 deg elevation pre-filter buffer and the padding
        # quantum (64) applied to the filtered orbit state.
        if int(visible_satellite_est) > 0 and int(visible_satellite_est) < sats_total_i:
            vis_fraction = min(
                1.0,
                float(int(visible_satellite_est) * 2) / float(sats_total_i),
            )
        else:
            vis_fraction = 1.0
        gpu_per_timestep_bytes = max(1, int(gpu_per_timestep_bytes_full * vis_fraction))
        gpu_recommendation = recommend_time_batch_size_linear(
            total_timesteps=steps_total_i,
            fixed_bytes=int(gpu_fixed_bytes),
            per_timestep_bytes=int(gpu_per_timestep_bytes),
            budget_bytes=int(gpu_working_budget),
        )
        base_candidate_bulk = min(
            int(host_recommendation["recommended_batch_size"]),
            int(gpu_recommendation["recommended_batch_size"]),
        )
        if force_bulk_timesteps is not None:
            base_candidate_bulk = max(1, min(int(force_bulk_timesteps), steps_total_i))

        bulk_candidates = (
            [int(base_candidate_bulk)]
            if force_bulk_timesteps is not None
            else _candidate_chunk_sizes(int(base_candidate_bulk))
        )
        seen_bulk_candidates: set[int] = set()
        for bulk_candidate in bulk_candidates:
            candidate_bulk = max(1, min(int(bulk_candidate), steps_total_i))
            if candidate_bulk in seen_bulk_candidates:
                continue
            seen_bulk_candidates.add(int(candidate_bulk))

            stage_budget_info: dict[str, int] = {}
            power_slab_candidate = 1
            finalize_slab_candidate = 1
            power_recommendation: dict[str, Any] = {"recommended_batch_size": int(candidate_bulk)}
            finalize_recommendation: dict[str, Any] = {"recommended_batch_size": int(candidate_bulk)}
            export_recommendation: dict[str, Any] = {"recommended_batch_size": int(candidate_bulk)}
            for _ in range(2):
                stage_budget_info = _resolve_direct_epfd_stage_working_budgets(
                    effective_budget_bytes=int(gpu_working_budget),
                    batch_timesteps=int(candidate_bulk),
                    export_bytes_per_timestep=int(export_per_timestep_bytes),
                    finalize_memory_budget_bytes=finalize_memory_budget_bytes,
                    power_memory_budget_bytes=power_memory_budget_bytes,
                    export_memory_budget_bytes=export_memory_budget_bytes,
                    host_effective_budget_bytes=int(host_working_budget),
                )
                power_slab_candidate = _estimate_direct_epfd_stage_sky_slab(
                    working_budget_bytes=int(stage_budget_info["power_memory_budget_bytes"]),
                    bytes_per_timestep=int(power_per_timestep_bytes),
                    n_skycells=sky_total_i,
                    boresight_active=bool(boresight_active),
                    explicit_sky_slab=power_sky_slab,
                )
                finalize_slab_candidate = _estimate_direct_epfd_stage_sky_slab(
                    working_budget_bytes=int(stage_budget_info["finalize_memory_budget_bytes"]),
                    bytes_per_timestep=int(finalize_per_timestep_bytes),
                    n_skycells=sky_total_i,
                    boresight_active=bool(boresight_active),
                )
                power_recommendation = recommend_time_batch_size_linear(
                    total_timesteps=steps_total_i,
                    fixed_bytes=0,
                    per_timestep_bytes=int(
                        _estimate_direct_epfd_slabbed_stage_peak_bytes(
                            batch_timesteps=1,
                            bytes_per_timestep=int(power_per_timestep_bytes),
                            n_skycells=sky_total_i,
                            boresight_active=bool(boresight_active),
                            sky_slab=int(power_slab_candidate),
                            fixed_overhead_bytes=0,
                        )
                    ),
                    budget_bytes=int(stage_budget_info["power_memory_budget_bytes"]),
                )
                finalize_recommendation = recommend_time_batch_size_linear(
                    total_timesteps=steps_total_i,
                    fixed_bytes=0,
                    per_timestep_bytes=int(
                        _estimate_direct_epfd_slabbed_stage_peak_bytes(
                            batch_timesteps=1,
                            bytes_per_timestep=int(finalize_per_timestep_bytes),
                            n_skycells=sky_total_i,
                            boresight_active=bool(boresight_active),
                            sky_slab=int(finalize_slab_candidate),
                            fixed_overhead_bytes=0,
                        )
                    ),
                    budget_bytes=int(stage_budget_info["finalize_memory_budget_bytes"]),
                )
                export_recommendation = recommend_time_batch_size_linear(
                    total_timesteps=steps_total_i,
                    fixed_bytes=0,
                    per_timestep_bytes=int(export_per_timestep_bytes),
                    budget_bytes=int(stage_budget_info["export_memory_budget_bytes"]),
                )
                if force_bulk_timesteps is not None:
                    candidate_bulk = max(1, min(int(force_bulk_timesteps), steps_total_i))
                    break
                updated_bulk = min(
                    int(candidate_bulk),
                    int(power_recommendation["recommended_batch_size"]),
                    int(finalize_recommendation["recommended_batch_size"]),
                    int(export_recommendation["recommended_batch_size"]),
                )
                updated_bulk = max(1, min(int(updated_bulk), steps_total_i))
                if int(updated_bulk) == int(candidate_bulk):
                    candidate_bulk = int(updated_bulk)
                    break
                candidate_bulk = int(updated_bulk)

            sky_slab = min(int(power_slab_candidate), int(finalize_slab_candidate))
            if not boresight_active:
                sky_slab = 1

            predicted_host_link_peak_bytes = int(
                estimate_step1_host_batch_bytes(
                    time_count=int(candidate_bulk),
                    n_cells=cells_total_i,
                    n_sats=sats_total_i,
                    n_links=int(nco),
                    cell_chunk_size=int(cell_chunk),
                    store_eligible_mask=bool(store_eligible_mask),
                    gpu_resident=True,
                )["peak_bytes"]
            )
            predicted_host_export_peak_bytes = int(
                max(0, int(export_per_timestep_bytes)) * int(candidate_bulk)
            )
            isolated_gpu_cell_link_peak_bytes = int(
                gpu_fixed_bytes + gpu_per_timestep_bytes * int(candidate_bulk)
            )
            isolated_gpu_finalize_peak_bytes = int(
                _estimate_direct_epfd_slabbed_stage_peak_bytes(
                    batch_timesteps=int(candidate_bulk),
                    bytes_per_timestep=int(finalize_per_timestep_bytes),
                    n_skycells=sky_total_i,
                    boresight_active=bool(boresight_active),
                    sky_slab=int(finalize_slab_candidate),
                )
            )
            isolated_gpu_power_peak_bytes = int(
                _estimate_direct_epfd_slabbed_stage_peak_bytes(
                    batch_timesteps=int(candidate_bulk),
                    bytes_per_timestep=int(power_per_timestep_bytes),
                    n_skycells=sky_total_i,
                    boresight_active=bool(boresight_active),
                    sky_slab=int(power_slab_candidate),
                )
            )
            spectral_candidates = [1]
            if activity_slab_enabled:
                spectral_candidates = _candidate_chunk_sizes(int(candidate_bulk))

            for spectral_slab in spectral_candidates:
                candidate_record = _build_direct_epfd_iteration_candidate_record(
                    candidate_bulk=int(candidate_bulk),
                    cell_chunk=int(cell_chunk),
                    sky_slab=int(sky_slab),
                    spectral_slab=int(spectral_slab),
                    cells_total_i=int(cells_total_i),
                    steps_total_i=int(steps_total_i),
                    sky_total_i=int(sky_total_i),
                    sats_total_i=int(sats_total_i),
                    visible_satellite_est=int(visible_satellite_est),
                    boresight_active=bool(boresight_active),
                    nco=int(nco),
                    nbeam=int(nbeam),
                    need_pointings=bool(need_pointings),
                    need_power_outputs=bool(need_power_outputs),
                    write_prx_elevation_heatmap=bool(write_prx_elevation_heatmap),
                    output_family_plan=output_family_plan,
                    store_eligible_mask=bool(store_eligible_mask),
                    profile_stages=bool(profile_stages),
                    gpu_output_dtype=gpu_output_dtype,
                    compute_dtype=compute_dtype,
                    count_dtype=count_dtype,
                    demand_count_dtype=demand_count_dtype,
                    predicted_host_link_peak_bytes=int(predicted_host_link_peak_bytes),
                    predicted_host_export_peak_bytes=int(predicted_host_export_peak_bytes),
                    predicted_gpu_cell_link_peak_bytes=int(isolated_gpu_cell_link_peak_bytes),
                    predicted_gpu_finalize_slab_bytes=int(isolated_gpu_finalize_peak_bytes),
                    predicted_gpu_power_slab_bytes=int(isolated_gpu_power_peak_bytes),
                    host_effective_budget=int(host_effective_budget),
                    host_working_budget=int(host_working_budget),
                    gpu_effective_budget=int(gpu_effective_budget),
                    gpu_working_budget=int(gpu_working_budget),
                    host_recommendation=host_recommendation,
                    gpu_recommendation=gpu_recommendation,
                    finalize_recommendation=finalize_recommendation,
                    power_recommendation=power_recommendation,
                    export_recommendation=export_recommendation,
                    spectrum_context_bytes=int(spectrum_context_bytes),
                    cell_activity_mode=str(cell_activity_mode),
                    activity_groups_per_cell=int(activity_groups_per_cell),
                    activity_power_policy=str(activity_power_policy),
                    activity_split_total_group_denominator_mode=str(
                        activity_split_total_group_denominator_mode
                    ),
                    planner_source=str(planner_source),
                    anchor_obs_fraction=float(anchor_obs_fraction),
                    anchor_time_fraction=float(anchor_time_fraction),
                    force_bulk_timesteps=force_bulk_timesteps,
                    multi_system_count=int(multi_system_count),
                    multi_system_uemr_count=int(multi_system_uemr_count),
                )
                if candidate_record is None:
                    continue
                candidate_record["stage_budget_info"] = dict(stage_budget_info)
                candidate_records.append(candidate_record)

    if not candidate_records:
        stage_budget_info = _resolve_direct_epfd_stage_working_budgets(
            effective_budget_bytes=int(gpu_working_budget),
            batch_timesteps=1,
            export_bytes_per_timestep=int(export_per_timestep_bytes),
            finalize_memory_budget_bytes=finalize_memory_budget_bytes,
            power_memory_budget_bytes=power_memory_budget_bytes,
            export_memory_budget_bytes=export_memory_budget_bytes,
            host_effective_budget_bytes=int(host_working_budget),
        )
        fallback_finalize_slab = int(
            _estimate_direct_epfd_slabbed_stage_peak_bytes(
                batch_timesteps=1,
                bytes_per_timestep=int(finalize_per_timestep_bytes),
                n_skycells=sky_total_i,
                boresight_active=bool(boresight_active),
                sky_slab=int(
                    _estimate_direct_epfd_stage_sky_slab(
                        working_budget_bytes=int(stage_budget_info["finalize_memory_budget_bytes"]),
                        bytes_per_timestep=int(finalize_per_timestep_bytes),
                        n_skycells=sky_total_i,
                        boresight_active=bool(boresight_active),
                    )
                ),
            )
        )
        fallback_power_slab = int(
            _estimate_direct_epfd_slabbed_stage_peak_bytes(
                batch_timesteps=1,
                bytes_per_timestep=int(power_per_timestep_bytes),
                n_skycells=sky_total_i,
                boresight_active=bool(boresight_active),
                sky_slab=int(
                    _estimate_direct_epfd_stage_sky_slab(
                        working_budget_bytes=int(stage_budget_info["power_memory_budget_bytes"]),
                        bytes_per_timestep=int(power_per_timestep_bytes),
                        n_skycells=sky_total_i,
                        boresight_active=bool(boresight_active),
                        explicit_sky_slab=power_sky_slab,
                    )
                ),
            )
        )
        fallback_combined_gpu = _estimate_direct_epfd_combined_gpu_peaks(
            batch_timesteps=1,
            cell_count=int(cells_total_i),
            sat_count_total=int(sats_total_i),
            sat_visible_count=int(visible_satellite_est),
            n_skycells=int(sky_total_i),
            boresight_active=bool(boresight_active),
            need_pointings=bool(need_pointings),
            need_beam_demand=bool(output_family_plan["needs_beam_demand"]),
            store_eligible_mask=bool(store_eligible_mask),
            profile_stages=bool(profile_stages),
            output_dtype=gpu_output_dtype,
            compute_dtype=compute_dtype,
            count_dtype=count_dtype,
            demand_count_dtype=demand_count_dtype,
            predicted_gpu_cell_chunk_peak_bytes=0,
            predicted_gpu_finalize_slab_bytes=int(fallback_finalize_slab),
            predicted_gpu_power_slab_bytes=int(fallback_power_slab),
            predicted_gpu_export_peak_bytes=0,
            write_epfd=bool(output_family_plan["needs_epfd"]),
            write_prx_total=bool(output_family_plan["needs_total_prx"]),
            write_per_satellite_prx_ras_station=bool(output_family_plan["needs_per_satellite_prx"]),
            write_prx_elevation_heatmap=bool(write_prx_elevation_heatmap),
            write_total_pfd_ras_station=bool(output_family_plan["needs_total_pfd"]),
            write_per_satellite_pfd_ras_station=bool(output_family_plan["needs_per_satellite_pfd"]),
            write_sat_beam_counts_used=bool(output_family_plan["needs_beam_counts"]),
            spectrum_context_bytes=int(spectrum_context_bytes),
            activity_gpu_resident_bytes=int(
                _estimate_direct_epfd_activity_gpu_memory(
                    time_count=1,
                    cell_count=int(cells_total_i),
                    groups_per_cell=int(activity_groups_per_cell),
                    cell_activity_mode=str(cell_activity_mode),
                    need_power_outputs=bool(need_power_outputs),
                    spectral_slab=1,
                    power_policy=str(activity_power_policy),
                    split_total_group_denominator_mode=str(
                        activity_split_total_group_denominator_mode
                    ),
                )["resident_bytes"]
            ),
            activity_gpu_peak_bytes=int(
                _estimate_direct_epfd_activity_gpu_memory(
                    time_count=1,
                    cell_count=int(cells_total_i),
                    groups_per_cell=int(activity_groups_per_cell),
                    cell_activity_mode=str(cell_activity_mode),
                    need_power_outputs=bool(need_power_outputs),
                    spectral_slab=1,
                    power_policy=str(activity_power_policy),
                    split_total_group_denominator_mode=str(
                        activity_split_total_group_denominator_mode
                    ),
                )["peak_bytes"]
            ),
        )
        fallback_activity_memory = _estimate_direct_epfd_activity_gpu_memory(
            time_count=1,
            cell_count=int(cells_total_i),
            groups_per_cell=int(activity_groups_per_cell),
            cell_activity_mode=str(cell_activity_mode),
            need_power_outputs=bool(need_power_outputs),
            spectral_slab=1,
            power_policy=str(activity_power_policy),
            split_total_group_denominator_mode=str(
                activity_split_total_group_denominator_mode
            ),
        )
        fallback = {
            "bulk_timesteps": 1,
            "cell_chunk": 1,
            "sky_slab": _estimate_direct_epfd_stage_sky_slab(
                working_budget_bytes=int(stage_budget_info["power_memory_budget_bytes"]),
                bytes_per_timestep=int(power_per_timestep_bytes),
                n_skycells=sky_total_i,
                boresight_active=bool(boresight_active),
                explicit_sky_slab=power_sky_slab,
            ),
            "spectral_slab": 1,
            "spectral_backoff_active": False,
            "limiting_dimension": "bulk_timesteps",
            "score": 1.0,
            "stage_budget_info": stage_budget_info,
            "planner_source": planner_source,
            "predicted_host_link_peak_bytes": int(
                estimate_step1_host_batch_bytes(
                    time_count=1,
                    n_cells=cells_total_i,
                    n_sats=sats_total_i,
                    n_links=int(nco),
                    cell_chunk_size=1,
                    store_eligible_mask=bool(store_eligible_mask),
                )["peak_bytes"]
            ),
            "predicted_host_export_peak_bytes": int(export_per_timestep_bytes),
            "predicted_host_peak_bytes": int(
                estimate_step1_host_batch_bytes(
                    time_count=1,
                    n_cells=cells_total_i,
                    n_sats=sats_total_i,
                    n_links=int(nco),
                    cell_chunk_size=1,
                    store_eligible_mask=bool(store_eligible_mask),
                )["peak_bytes"]
                + int(export_per_timestep_bytes)
            ),
            "predicted_gpu_peak_bytes": int(fallback_combined_gpu["predicted_gpu_peak_bytes"]),
            "predicted_gpu_propagation_peak_bytes": int(
                fallback_combined_gpu["predicted_gpu_cell_link_peak_bytes"]
            ),
            "predicted_gpu_finalize_peak_bytes": int(
                fallback_combined_gpu["predicted_gpu_finalize_peak_bytes"]
            ),
            "predicted_gpu_power_peak_bytes": int(
                fallback_combined_gpu["predicted_gpu_power_peak_bytes"]
            ),
            "predicted_gpu_export_peak_bytes": int(
                fallback_combined_gpu["predicted_gpu_export_peak_bytes"]
            ),
            "predicted_gpu_activity_resident_bytes": int(
                fallback_activity_memory["resident_bytes"]
            ),
            "predicted_gpu_activity_scratch_bytes": int(
                fallback_activity_memory["scratch_bytes"]
            ),
            "predicted_gpu_activity_peak_bytes": int(
                fallback_combined_gpu["predicted_gpu_activity_stage_peak_bytes"]
            ),
            "predicted_gpu_spectrum_context_bytes": int(spectrum_context_bytes),
            "predicted_gpu_setup_bytes": int(fallback_combined_gpu["setup_bytes"]),
            "predicted_gpu_orbit_state_bytes": int(fallback_combined_gpu["orbit_state_bytes"]),
            "predicted_gpu_visible_bytes": int(fallback_combined_gpu["visible_bytes"]),
            "predicted_gpu_link_library_resident_bytes": int(
                fallback_combined_gpu["link_library_resident_bytes"]
            ),
            "predicted_gpu_link_library_transient_peak_bytes": int(
                fallback_combined_gpu["link_library_chunk_transient_peak_bytes"]
            ),
            "predicted_gpu_finalize_accumulator_bytes": int(
                fallback_combined_gpu["finalize_accumulator_bytes"]
            ),
            "predicted_gpu_beam_finalize_working_bytes": int(fallback_finalize_slab),
            "predicted_gpu_finalize_transient_peak_bytes": int(
                fallback_combined_gpu["predicted_gpu_finalize_transient_peak_bytes"]
            ),
            "predicted_gpu_power_result_bytes": int(
                fallback_combined_gpu["power_result_bytes"]
            ),
            "compute_budget_utilization_fraction": float(
                min(
                    1.0,
                    float(
                        max(
                            int(fallback_combined_gpu["predicted_gpu_cell_link_peak_bytes"]),
                            int(fallback_combined_gpu["predicted_gpu_finalize_peak_bytes"]),
                            int(fallback_combined_gpu["predicted_gpu_power_peak_bytes"]),
                            int(
                                fallback_combined_gpu[
                                    "predicted_gpu_activity_stage_peak_bytes"
                                ]
                            ),
                        )
                    )
                    / float(max(1, gpu_working_budget)),
                )
            ),
            "export_budget_utilization_fraction": float(
                min(
                    1.0,
                    float(max(0, int(export_per_timestep_bytes)))
                    / float(max(1, host_working_budget)),
                )
            ),
            "batch_count_estimate": int(steps_total_i),
            "chunk_count_estimate": int(cells_total_i),
            "underfill_reason": "fallback_minimum_shape",
            "limiting_resource": "fallback",
            "planned_batch_seconds": 1.0,
        }
        candidate_records.append(fallback)

    candidate_records.sort(
        key=lambda item: (
            float(item["score"]),
            int(item["bulk_timesteps"]) * int(item["cell_chunk"]) * int(item["sky_slab"]),
            int(item["cell_chunk"]),
            -int(item.get("chunk_count_estimate", cells_total_i)),
            int(item.get("spectral_slab", 1)),
        ),
        reverse=True,
    )
    best = candidate_records[0]
    return {
        **best,
        "host_working_budget_bytes": int(host_working_budget),
        "gpu_working_budget_bytes": int(gpu_working_budget),
        "scheduler_target_fraction": float(scheduler_target_fraction),
        "scheduler_active_target_fraction": float(scheduler_active_target_fraction),
        "candidate_count": int(len(candidate_records)),
        "planner_source": str(best["planner_source"]),
    }


def _compute_gpu_direct_epfd_batch_device(
    *,
    session: Any,
    cp: Any,
    observer_context: Any,
    orbit_state: Any,
    sat_topo_ras_station: Any,
    sat_azel_ras_station: Any,
    sat_keep_batch: Any,
    sat_min_elev_deg_per_sat_f64: np.ndarray,
    sat_beta_max_deg_per_sat_f32: np.ndarray,
    sat_belt_id_per_sat_i16: np.ndarray,
    selection_mode: str,
    nco: int,
    nbeam: int,
    n_cells_total: int,
    cell_active_mask_dev: Any,
    cell_spectral_weight_dev: Any | None,
    dynamic_spectrum_state: _DirectEpfdDynamicSpectrumState | None,
    ras_service_cell_index: int,
    effective_ras_pointing_mode: str,
    ras_guard_angle_rad: float,
    boresight_active: bool,
    boresight_theta1_deg: float | None,
    boresight_theta2_deg: float | None,
    boresight_theta2_cell_ids: np.ndarray | None,
    pointings: Any | None,
    time_count_local: int,
    cell_chunk: int,
    n_cell_chunks: int,
    gpu_output_dtype: Any,
    profile_stages: bool,
    stage_timings: dict[str, float],
    stage_start: float,
    enable_progress_bars: bool,
    progress_desc_mode: str,
    pbar: Any,
    ii: int,
    bi: int,
    orbit_radius_full: Any,
    observer_alt_km_ras_station: float,
    power_input: Mapping[str, Any],
    spectrum_plan_context: Any | None,
    target_alt_km: float,
    use_ras_station_alt_for_co: bool,
    s1528_pattern_context: Any | None,
    ras_pattern_context: Any | None,
    atmosphere_context: Any | None,
    peak_pfd_lut_context: Any | None = None,
    max_surface_pfd_dbw_m2_channel: float | None = None,
    max_surface_pfd_dbw_m2_mhz: float | None = None,
    surface_pfd_cap_mode: str = "per_beam",
    surface_pfd_stats_enabled: bool = False,
    host_effective_budget_bytes: int,
    gpu_effective_budget_bytes: int,
    scheduler_active_target_fraction: float,
    predicted_host_peak_bytes: int,
    predicted_gpu_propagation_peak_bytes: int,
    predicted_gpu_finalize_peak_bytes: int,
    predicted_gpu_power_peak_bytes: int,
    finalize_memory_budget_bytes: int | None,
    power_memory_budget_bytes: int | None,
    power_sky_slab: int | None,
    spectral_slab: int,
    visibility_elev_threshold_deg: float,
    debug_direct_epfd: bool,
    write_epfd: bool,
    write_prx_total: bool,
    write_per_satellite_prx_ras_station: bool,
    write_prx_elevation_heatmap: bool,
    write_total_pfd_ras_station: bool,
    write_per_satellite_pfd_ras_station: bool,
    write_sat_beam_counts_used: bool,
    write_sat_eligible_mask: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_callback: Callable[[], str | None] | None = None,
    uemr_mode: bool = False,
) -> dict[str, Any]:
    def _capture_stage_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
        return _update_direct_epfd_stage_memory_summary(
            summary,
            _capture_direct_epfd_live_memory_snapshot(cp, session),
        )

    def _capture_stage_substage_summary(
        summary: Mapping[str, Any] | None,
        *,
        substage: str,
    ) -> dict[str, Any]:
        merged = dict(summary or {})
        merged["observed_stage_substages"] = _update_direct_epfd_substage_memory_summary(
            merged.get("observed_stage_substages"),
            str(substage),
            _capture_direct_epfd_live_memory_snapshot(cp, session),
        )
        return merged

    def _finalize_stage_summary(
        summary: Mapping[str, Any] | None,
        *,
        predicted_peak_bytes: int | None = None,
    ) -> dict[str, Any]:
        finalized = _refresh_direct_epfd_stage_observed_bytes(summary)
        if predicted_peak_bytes is not None:
            observed_peak = finalized.get("observed_stage_gpu_peak_bytes")
            if observed_peak is not None:
                try:
                    finalized["planner_vs_observed_gpu_peak_error_bytes"] = int(
                        int(observed_peak) - int(predicted_peak_bytes)
                    )
                except Exception:
                    pass
        return finalized

    def _raise_stage_oom(
        stage: str,
        exc: BaseException,
        summary: Mapping[str, Any] | None,
    ) -> None:
        original_exception = getattr(exc, "original_exception", exc)
        raise _DirectGpuOutOfMemory(
            stage,
            original_exception,
            stage_memory_summary=_capture_stage_summary(summary),
        ) from exc

    def _merge_max_numeric_mapping(
        existing: Mapping[str, Any] | None,
        incoming: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(existing or {})
        if not isinstance(incoming, Mapping):
            return merged
        for key, value in dict(incoming).items():
            key_str = str(key)
            if value is None:
                continue
            try:
                numeric_value = int(value)
            except Exception:
                merged[key_str] = value
                continue
            merged[key_str] = int(max(int(merged.get(key_str, 0) or 0), numeric_value))
        return merged

    any_power_outputs = bool(
        write_epfd
        or write_prx_total
        or write_per_satellite_prx_ras_station
        or write_prx_elevation_heatmap
        or write_total_pfd_ras_station
        or write_per_satellite_pfd_ras_station
    )
    # --- UEMR bypass branch -------------------------------------------------
    # UEMR systems radiate isotropically, so there is no beam library, no
    # beam-finalize, and no cell-link selection. We invoke the dedicated
    # UEMR kernel directly on the visible-filtered topocentric slice and
    # return a payload with all the dict keys the caller's unpacking
    # expects. Per-beam-library fields (sat_beam_counts, eligible_mask,
    # diag_result) are None / empty.
    if uemr_mode and any_power_outputs:
        from scepter.gpu_accel import _accumulate_uemr_power_cp as _uemr_accum

        _sat_idx_g = cp.nonzero(sat_keep_batch)[0].astype(cp.int32, copy=False)
        _sat_topo_visible = sat_topo_ras_station[:, _sat_idx_g, :]
        _sat_azel_visible = sat_azel_ras_station[:, _sat_idx_g, :]
        _orbit_radius_eff = orbit_radius_full[_sat_idx_g]
        # UEMR kernel documents sat_topo as (T, S, 3); strip the 4th
        # propagator column (xyz sentinel / range padding) if present.
        _uemr_topo = _sat_topo_visible[:, :, :3]
        _tel_az = None if pointings is None else pointings["azimuth_deg"]
        _tel_el = None if pointings is None else pointings["elevation_deg"]
        _uemr_result = _uemr_accum(
            uemr_pattern_context=s1528_pattern_context,
            ras_pattern_context=ras_pattern_context,
            atmosphere_lut_context=atmosphere_context,
            sat_topo=_uemr_topo,
            telescope_azimuth_deg=_tel_az,
            telescope_elevation_deg=_tel_el,
            observer_alt_km=float(observer_alt_km_ras_station),
            bandwidth_mhz=float(power_input["bandwidth_mhz"]),
            power_input_quantity=str(power_input["power_input_quantity"]),
            target_pfd_dbw_m2_channel=power_input.get("target_pfd_dbw_m2_channel"),
            satellite_ptx_dbw_channel=power_input.get("satellite_ptx_dbw_channel"),
            satellite_eirp_dbw_channel=power_input.get("satellite_eirp_dbw_channel"),
            eirp_slab_fraction_lin=1.0,
            visibility_elev_threshold_deg=float(visibility_elev_threshold_deg),
            include_epfd=bool(write_epfd),
            include_prx_total=bool(write_prx_total),
            include_per_satellite_prx=bool(write_per_satellite_prx_ras_station),
            include_total_pfd=bool(write_total_pfd_ras_station),
            include_per_satellite_pfd=bool(write_per_satellite_pfd_ras_station),
        )
        # The UEMR kernel produces tensors with the SAME layout as the
        # directive kernel: (T, 1, N_sky) for Prx_total / EPFD,
        # (T, 1, 1) for PFD_total, (T, S) for per-satellite. Across
        # systems sharing the same telescope context, N_sky is identical,
        # so the combiner can sum them directly without any uniform-
        # broadcast logic. We do NOT flag _spatially_uniform — that flag
        # was for the old (T, 1, 1) UEMR layout that needed scalar
        # broadcasting; the new layout doesn't need it.
        _uemr_bypass_payload: dict[str, Any] = dict(_uemr_result)
        return {
            "power_result": _uemr_bypass_payload,
            "sat_idx_g": _sat_idx_g,
            "sat_topo_visible": _sat_topo_visible,
            "sat_azel_visible": _sat_azel_visible,
            "orbit_radius_eff": _orbit_radius_eff,
            "sat_beam_counts_used_full": None,
            "sat_eligible_mask": None,
            "diag_result": None,
            "debug_direct_epfd_stats": None,
            "beam_finalize_substage_timings": {},
            "cell_link_library_chunk_telemetry": {},
            "cell_link_library_stage_memory_summary": {},
            "beam_finalize_stage_memory_summary": {},
            "power_stage_memory_summary": {},
            "stage_start": stage_start,
            "stage_memory_summary": {},
            "beam_finalize_chunk_shape": {},
            "boresight_compaction_stats": {},
        }
    cell_link_stage_summary = _start_direct_epfd_stage_memory_summary(
        "cell_link_library",
        cp=cp,
        session=session,
    )
    cell_link_substage_timings: dict[str, float] = {}
    last_stage_memory_summary: dict[str, Any] = dict(cell_link_stage_summary)
    try:
        _raise_if_direct_epfd_stage_live_fit_is_unsafe(
            stage="cell_link_library",
            host_peak_bytes=int(predicted_host_peak_bytes),
            gpu_peak_bytes=int(predicted_gpu_propagation_peak_bytes),
            host_effective_budget_bytes=int(host_effective_budget_bytes),
            gpu_effective_budget_bytes=int(gpu_effective_budget_bytes),
            scheduler_active_target_fraction=float(scheduler_active_target_fraction),
            live_host_snapshot=_runtime_host_memory_snapshot(),
            live_gpu_snapshot=_runtime_gpu_memory_snapshot(cp, session),
        )
    except _DirectGpuOutOfMemory as exc:
        _raise_stage_oom("cell_link_library", exc, cell_link_stage_summary)
    try:
        link_library = session.prepare_satellite_link_selection_library(
            time_count=time_count_local,
            cell_count=n_cells_total,
            sat_count=int(sat_topo_ras_station.shape[1]),
            min_elevation_deg=sat_min_elev_deg_per_sat_f64,
            n_links=int(nco),
            n_beam=int(nbeam),
            strategy=selection_mode,
            sat_belt_id_per_sat=sat_belt_id_per_sat_i16,
            beta_max_deg_per_sat=sat_beta_max_deg_per_sat_f32,
            ras_topo=sat_topo_ras_station,
            cell_active_mask=cell_active_mask_dev,
            rng=int((16001 + ii * 1_000_000 + bi * 10_000) % (2**32 - 1)),
            include_counts=False,
            include_payload=True,
            include_eligible_mask=bool(write_sat_eligible_mask),
            boresight_pointing_azimuth_deg=(
                pointings["azimuth_deg"] if boresight_active and pointings is not None else None
            ),
            boresight_pointing_elevation_deg=(
                pointings["elevation_deg"] if boresight_active and pointings is not None else None
            ),
            boresight_theta1_deg=boresight_theta1_deg if boresight_active else None,
            boresight_theta2_deg=boresight_theta2_deg if boresight_active else None,
            boresight_theta2_cell_ids=boresight_theta2_cell_ids if boresight_active else None,
            beta_tol_deg=float(np.float32(1e-3)),
        )
    except Exception as exc:
        if _is_direct_gpu_out_of_memory(exc, cp=cp):
            _raise_stage_oom("cell_link_library", exc, cell_link_stage_summary)
        raise
    cell_link_stage_summary = _capture_stage_summary(cell_link_stage_summary)

    # -- Satellite pre-filtering optimisation ---------------------------------
    # Only ~5-10 % of satellites are above the horizon from the RAS station
    # at any given timestep.  Computing cell-level geometry for all 3360 sats
    # wastes GPU memory and kernel time.  We use a -2 deg elevation buffer
    # to ensure satellites visible from nearby cells (within the satellite
    # footprint) but marginally below the RAS station horizon are included.
    _prefilter_mask = cp.any(
        sat_topo_ras_station[..., 1] > cp.float32(visibility_elev_threshold_deg - 2.0),
        axis=0,
    )
    sat_visible_indices_cp = cp.nonzero(_prefilter_mask)[0].astype(cp.int32, copy=False)
    n_vis = int(sat_visible_indices_cp.size)
    if n_vis > 0 and n_vis < int(_prefilter_mask.size):
        from dataclasses import replace as _dc_replace

        # Pad the filtered satellite count to a fixed quantum so that the
        # downstream workspace cache key stays stable across batches.  This
        # prevents expensive re-allocation of ~1 GB geometry buffers every
        # batch due to minor visible-count fluctuations (e.g. 210→212).
        # Padding positions are set to zero (Earth centre) so they have
        # extreme negative elevation and are filtered out by add_chunk.
        _SAT_PAD_QUANTUM = 64
        n_vis_padded = ((n_vis + _SAT_PAD_QUANTUM - 1) // _SAT_PAD_QUANTUM) * _SAT_PAD_QUANTUM
        n_vis_padded = min(n_vis_padded, int(_prefilter_mask.size))

        d_pos_full = cp.asarray(orbit_state.d_eci_pos)
        d_vel_full = cp.asarray(orbit_state.d_eci_vel)
        d_pos_filtered = d_pos_full[:, sat_visible_indices_cp, :]
        d_vel_filtered = d_vel_full[:, sat_visible_indices_cp, :]
        if n_vis_padded > n_vis:
            pad_count = n_vis_padded - n_vis
            T = d_pos_filtered.shape[0]
            d_pos_filtered = cp.concatenate([
                d_pos_filtered,
                cp.zeros((T, pad_count, 3), dtype=d_pos_filtered.dtype),
            ], axis=1)
            d_vel_filtered = cp.concatenate([
                d_vel_filtered,
                cp.zeros((T, pad_count, 3), dtype=d_vel_filtered.dtype),
            ], axis=1)
        filtered_orbit_state = _dc_replace(
            orbit_state,
            d_eci_pos=d_pos_filtered,
            d_eci_vel=d_vel_filtered,
        )
        # Extend the remap array to cover padding slots so it matches
        # the padded satellite dimension.  Padding satellites are at Earth
        # centre and will be filtered out by the visibility check; the
        # remap value for these slots is irrelevant (use 0).
        if n_vis_padded > n_vis:
            sat_index_remap_cp = cp.concatenate([
                sat_visible_indices_cp,
                cp.zeros(n_vis_padded - n_vis, dtype=cp.int32),
            ])
        else:
            sat_index_remap_cp = sat_visible_indices_cp
    else:
        filtered_orbit_state = orbit_state
        sat_index_remap_cp = None

    for chunk_i, c0 in enumerate(range(0, n_cells_total, cell_chunk)):
        if _query_direct_epfd_cancel_mode(cancel_callback) == "force":
            raise _RunCancellationRequested("force", "chunk_boundary")
        c1 = min(n_cells_total, c0 + cell_chunk)
        c_obs0 = 1 + c0
        c_obs1 = 1 + c1
        _set_direct_epfd_progress_phase(
            pbar,
            enable_progress_bars=enable_progress_bars,
            progress_desc_mode=progress_desc_mode,
            phase="chunk_detail",
            chunk_i=chunk_i,
            n_cell_chunks=n_cell_chunks,
            c0=c0,
            c1=c1,
        )
        _emit_direct_epfd_progress(
            progress_callback,
            kind="chunk",
            phase="chunk_detail",
            iteration_index=int(ii),
            batch_index=int(bi),
            chunk_index=int(chunk_i),
            chunk_total=int(n_cell_chunks),
            cell_start=int(c0),
            cell_stop=int(c1),
            description=_direct_epfd_progress_text(
                progress_desc_mode,
                "chunk_detail",
                chunk_i=chunk_i,
                n_cell_chunks=n_cell_chunks,
                c0=c0,
                c1=c1,
            ),
        )
        try:
            derive_t0 = perf_counter()
            chunk_geom = session.derive_from_eci(
                filtered_orbit_state,
                observer_context=observer_context,
                observer_slice=slice(c_obs0, c_obs1),
                do_eci_pos=False,
                do_eci_vel=False,
                do_geo=False,
                do_topo=True,
                do_obs_pos=False,
                do_sat_azel=True,
                do_sat_rotmat=False,
                output_dtype=gpu_output_dtype,
                return_device=True,
            )
            _accumulate_profile_timing(
                cell_link_substage_timings,
                "derive_from_eci",
                perf_counter() - derive_t0,
            )
            cell_link_stage_summary = _capture_stage_substage_summary(
                cell_link_stage_summary,
                substage="derive_from_eci",
            )
            add_chunk_t0 = perf_counter()
            # Pass full (possibly padded) geometry to add_chunk.
            # Padding satellites are at Earth centre with extreme negative
            # elevation and will be filtered out by the visibility check.
            # We do NOT slice off padding here because CuPy non-contiguous
            # slicing on inner axes is catastrophically slow (~59 s for a
            # 582 MB array due to element-wise copy).
            chunk_topo = chunk_geom["topo"]
            chunk_sat_azel = chunk_geom["sat_azel"]
            link_library.add_chunk(
                c0,
                chunk_topo,
                sat_azel=chunk_sat_azel,
                sat_index_remap=sat_index_remap_cp,
            )
            _accumulate_profile_timing(
                cell_link_substage_timings,
                "add_chunk",
                perf_counter() - add_chunk_t0,
            )
            cell_link_stage_summary = _capture_stage_substage_summary(
                cell_link_stage_summary,
                substage="add_chunk",
            )
            chunk_telemetry = getattr(link_library, "last_add_chunk_telemetry", None)
            if isinstance(chunk_telemetry, Mapping):
                aggregated_chunk_telemetry = dict(
                    cell_link_stage_summary.get("chunk_telemetry") or {}
                )
                for key, value in dict(chunk_telemetry).items():
                    key_str = str(key)
                    try:
                        numeric_value = int(value)
                    except Exception:
                        aggregated_chunk_telemetry[key_str] = value
                        continue
                    aggregated_chunk_telemetry[key_str] = int(
                        max(int(aggregated_chunk_telemetry.get(key_str, 0) or 0), numeric_value)
                    )
                cell_link_stage_summary["chunk_telemetry"] = aggregated_chunk_telemetry
        except Exception as exc:
            if _is_direct_gpu_out_of_memory(exc, cp=cp):
                _raise_stage_oom("cell_link_library", exc, cell_link_stage_summary)
            raise
        del chunk_geom
        cell_link_stage_summary = _capture_stage_summary(cell_link_stage_summary)
    stage_start = record_profile_stage(
        stage_timings,
        "cell_link_library",
        stage_start,
        enabled=profile_stages,
        synchronize=lambda: _sync_array_module(cp),
    )
    # Expose cell_link substage breakdown in stage_timings for diagnostics
    if profile_stages and stage_timings is not None and cell_link_substage_timings:
        for _sub_k, _sub_v in cell_link_substage_timings.items():
            stage_timings[f"cll_{_sub_k}"] = float(_sub_v)
    cell_link_stage_summary = _capture_stage_summary(cell_link_stage_summary)
    cell_link_stage_summary["substage_timings"] = dict(cell_link_substage_timings)
    last_stage_memory_summary = _finalize_stage_summary(
        cell_link_stage_summary,
        predicted_peak_bytes=int(predicted_gpu_propagation_peak_bytes),
    )
    _set_direct_epfd_progress_phase(
        pbar,
        enable_progress_bars=enable_progress_bars,
        progress_desc_mode=progress_desc_mode,
        phase="beam_finalize",
    )
    _emit_direct_epfd_progress(
        progress_callback,
        kind="phase",
        phase="beam_finalize",
        iteration_index=int(ii),
        batch_index=int(bi),
        description=_direct_epfd_progress_text(progress_desc_mode, "beam_finalize"),
    )
    beam_finalize_stage_summary = _start_direct_epfd_stage_memory_summary(
        "beam_finalize",
        cp=cp,
        session=session,
    )
    beam_finalize_substage_timings: dict[str, float] = {}
    try:
        _raise_if_direct_epfd_stage_live_fit_is_unsafe(
            stage="beam_finalize",
            host_peak_bytes=int(predicted_host_peak_bytes),
            gpu_peak_bytes=int(predicted_gpu_finalize_peak_bytes),
            host_effective_budget_bytes=int(host_effective_budget_bytes),
            gpu_effective_budget_bytes=int(gpu_effective_budget_bytes),
            scheduler_active_target_fraction=float(scheduler_active_target_fraction),
            live_host_snapshot=_runtime_host_memory_snapshot(),
            live_gpu_snapshot=_runtime_gpu_memory_snapshot(cp, session),
        )
    except _DirectGpuOutOfMemory as exc:
        _raise_stage_oom("beam_finalize", exc, beam_finalize_stage_summary)

    sat_idx_g = cp.nonzero(sat_keep_batch)[0].astype(cp.int32, copy=False)
    sat_topo_visible = sat_topo_ras_station[:, sat_idx_g, :]
    sat_azel_visible = sat_azel_ras_station[:, sat_idx_g, :]
    orbit_radius_eff = orbit_radius_full[sat_idx_g]
    power_result = None
    sat_beam_counts_used_full = None
    diag_result: dict[str, Any] | None = None
    debug_direct_epfd_stats: list[dict[str, Any]] = []
    need_beam_finalize = bool(
        any_power_outputs or write_sat_beam_counts_used or write_sat_eligible_mask
    )
    if not need_beam_finalize:
        last_stage_memory_summary = _finalize_stage_summary(
            last_stage_memory_summary,
            predicted_peak_bytes=int(predicted_gpu_propagation_peak_bytes),
        )
        return {
            "power_result": None,
            "sat_idx_g": sat_idx_g,
            "sat_topo_visible": sat_topo_visible,
            "sat_azel_visible": sat_azel_visible,
            "orbit_radius_eff": orbit_radius_eff,
            "sat_beam_counts_used_full": None,
            "diag_result": None,
            "debug_direct_epfd_stats": debug_direct_epfd_stats,
            "beam_finalize_substage_timings": {},
            "cell_link_library_chunk_telemetry": dict(
                cell_link_stage_summary.get("chunk_telemetry") or {}
            ),
            "cell_link_library_stage_memory_summary": dict(last_stage_memory_summary),
            "beam_finalize_stage_memory_summary": {},
            "power_stage_memory_summary": {},
            "stage_start": stage_start,
            "stage_memory_summary": dict(last_stage_memory_summary),
        }

    beam_finalize_kwargs: dict[str, Any] = {
        "ras_pointing_mode": effective_ras_pointing_mode,
        "include_diagnostics": bool(profile_stages or debug_direct_epfd),
        "include_full_sat_beam_counts_used": bool(write_sat_beam_counts_used),
        "output_sat_indices": sat_idx_g,
        "debug_direct_epfd": bool(debug_direct_epfd),
        "working_memory_budget_bytes": (
            None if finalize_memory_budget_bytes is None else int(finalize_memory_budget_bytes)
        ),
    }
    if effective_ras_pointing_mode == "ras_station":
        beam_finalize_kwargs.update(
            ras_cell_index=int(ras_service_cell_index),
            ras_sat_azel=sat_azel_visible,
            ras_guard_angle_rad=ras_guard_angle_rad,
        )

    accumulate_direct_epfd = getattr(session, "accumulate_direct_epfd_from_link_library", None)
    if callable(accumulate_direct_epfd):
        power_stage_summary: dict[str, Any] | None = None
        beam_finalize_chunk_shape: dict[str, Any] = {}
        boresight_compaction_stats: dict[str, Any] = {}
        if any_power_outputs:
            power_stage_summary = _start_direct_epfd_stage_memory_summary(
                "power_accumulation",
                cp=cp,
                session=session,
            )
            # Emit the "Accumulating power" progress phase here as well —
            # the fused path bundles beam_finalize + power into one call,
            # so without this explicit emission the progress bar / progress
            # callback would skip the power-phase label entirely.  Users
            # expect the same phase cadence in both fused and separated
            # paths, and GUI tests assert on it.
            _set_direct_epfd_progress_phase(
                pbar,
                enable_progress_bars=enable_progress_bars,
                progress_desc_mode=progress_desc_mode,
                phase="power_accumulation",
            )
            _emit_direct_epfd_progress(
                progress_callback,
                kind="phase",
                phase="power_accumulation",
                iteration_index=int(ii),
                batch_index=int(bi),
                description=_direct_epfd_progress_text(
                    progress_desc_mode, "power_accumulation"
                ),
            )
        try:
            fused_payload = accumulate_direct_epfd(
                link_library=link_library,
                s1528_pattern_context=s1528_pattern_context,
                ras_pattern_context=ras_pattern_context,
                sat_topo=sat_topo_visible,
                sat_azel=sat_azel_visible,
                orbit_radius_m_per_sat=orbit_radius_eff,
                observer_alt_km=float(observer_alt_km_ras_station),
                telescope_azimuth_deg=None if pointings is None else pointings["azimuth_deg"],
                telescope_elevation_deg=None if pointings is None else pointings["elevation_deg"],
                atmosphere_lut_context=atmosphere_context,
                spectrum_plan_context=spectrum_plan_context,
                cell_spectral_weight=cell_spectral_weight_dev,
                dynamic_group_active_mask=(
                    None
                    if dynamic_spectrum_state is None
                    else dynamic_spectrum_state.group_active_mask_dev
                ),
                dynamic_cell_group_leakage_factors=(
                    None
                    if dynamic_spectrum_state is None
                    else dynamic_spectrum_state.cell_group_leakage_factors_dev
                ),
                dynamic_group_valid_mask=(
                    None
                    if dynamic_spectrum_state is None
                    else dynamic_spectrum_state.group_valid_mask_dev
                ),
                dynamic_power_policy=(
                    None if dynamic_spectrum_state is None else dynamic_spectrum_state.power_policy
                ),
                dynamic_split_total_group_denominator_mode=(
                    None
                    if dynamic_spectrum_state is None
                    else dynamic_spectrum_state.split_total_group_denominator_mode
                ),
                dynamic_configured_groups_per_cell=(
                    None
                    if dynamic_spectrum_state is None
                    else dynamic_spectrum_state.configured_groups_per_cell
                ),
                bandwidth_mhz=float(power_input["bandwidth_mhz"]),
                power_input_quantity=str(power_input["power_input_quantity"]),
                target_pfd_dbw_m2_mhz=power_input["target_pfd_dbw_m2_mhz"],
                target_pfd_dbw_m2_channel=power_input["target_pfd_dbw_m2_channel"],
                satellite_ptx_dbw_mhz=power_input["satellite_ptx_dbw_mhz"],
                satellite_ptx_dbw_channel=power_input["satellite_ptx_dbw_channel"],
                satellite_eirp_dbw_mhz=power_input["satellite_eirp_dbw_mhz"],
                satellite_eirp_dbw_channel=power_input["satellite_eirp_dbw_channel"],
                power_variation_mode=str(power_input.get("power_variation_mode", "fixed")),
                power_range_min_dbw_channel=power_input.get("power_range_min_dbw_channel"),
                power_range_max_dbw_channel=power_input.get("power_range_max_dbw_channel"),
                n_links=int(nco),
                ras_service_cell_index=int(ras_service_cell_index),
                ras_pointing_mode=effective_ras_pointing_mode,
                ras_guard_angle_rad=float(ras_guard_angle_rad),
                target_alt_km=float(target_alt_km),
                use_ras_station_alt_for_co=bool(use_ras_station_alt_for_co),
                include_epfd=bool(write_epfd),
                include_prx_total=bool(write_prx_total),
                include_per_satellite_prx=bool(
                    write_per_satellite_prx_ras_station or write_prx_elevation_heatmap
                ),
                include_total_pfd=bool(write_total_pfd_ras_station),
                include_per_satellite_pfd=bool(write_per_satellite_pfd_ras_station),
                include_diagnostics=bool(profile_stages or debug_direct_epfd),
                include_full_sat_beam_counts_used=bool(write_sat_beam_counts_used),
                include_sat_eligible_mask=bool(write_sat_eligible_mask),
                output_sat_indices=sat_idx_g,
                finalize_working_memory_budget_bytes=finalize_memory_budget_bytes,
                power_working_memory_budget_bytes=power_memory_budget_bytes,
                power_sky_slab=power_sky_slab,
                spectral_slab=int(spectral_slab),
                peak_pfd_lut_context=peak_pfd_lut_context,
                max_surface_pfd_dbw_m2_channel=max_surface_pfd_dbw_m2_channel,
                max_surface_pfd_dbw_m2_mhz=max_surface_pfd_dbw_m2_mhz,
                surface_pfd_cap_mode=str(surface_pfd_cap_mode),
                surface_pfd_stats_enabled=bool(surface_pfd_stats_enabled),
                scheduler_target_profile="high_throughput",
                debug_direct_epfd=bool(debug_direct_epfd),
                return_device=True,
            )
        except Exception as exc:
            stage_name = str(getattr(exc, "stage", "beam_finalize") or "beam_finalize")
            original_exc = getattr(exc, "original_exception", exc)
            target_summary = (
                power_stage_summary
                if stage_name == "power_accumulation" and power_stage_summary is not None
                else beam_finalize_stage_summary
            )
            if _is_direct_gpu_out_of_memory(original_exc, cp=cp):
                _raise_stage_oom(stage_name, exc, target_summary)
            raise

        for name, value in dict(fused_payload.get("stage_timings") or {}).items():
            _accumulate_profile_timing(stage_timings, str(name), float(value))
        for name, value in dict(fused_payload.get("beam_finalize_substage_timings") or {}).items():
            _accumulate_profile_timing(beam_finalize_substage_timings, str(name), float(value))
        beam_finalize_stage_summary = _capture_stage_summary(beam_finalize_stage_summary)
        beam_finalize_stage_summary["substage_timings"] = dict(beam_finalize_substage_timings)
        if isinstance(fused_payload.get("beam_finalize_observed_memory"), Mapping):
            beam_finalize_stage_summary["observed_finalize_slab_memory"] = _merge_max_numeric_mapping(
                beam_finalize_stage_summary.get("observed_finalize_slab_memory"),
                fused_payload.get("beam_finalize_observed_memory"),
            )
        hot_path_host_copy_count = int(fused_payload.get("hot_path_device_to_host_copy_count", 0) or 0)
        hot_path_host_copy_bytes = int(fused_payload.get("hot_path_device_to_host_copy_bytes", 0) or 0)
        device_scalar_sync_count = int(fused_payload.get("device_scalar_sync_count", 0) or 0)
        beam_finalize_stage_summary["hot_path_device_to_host_copy_count"] = hot_path_host_copy_count
        beam_finalize_stage_summary["hot_path_device_to_host_copy_bytes"] = hot_path_host_copy_bytes
        beam_finalize_stage_summary["device_scalar_sync_count"] = device_scalar_sync_count
        if isinstance(fused_payload.get("beam_finalize_chunk_shape"), Mapping):
            beam_finalize_chunk_shape = dict(fused_payload["beam_finalize_chunk_shape"])
            beam_finalize_stage_summary["chunk_shape"] = dict(beam_finalize_chunk_shape)
        if isinstance(fused_payload.get("boresight_compaction_stats"), Mapping):
            boresight_compaction_stats = _merge_max_numeric_mapping(
                boresight_compaction_stats,
                fused_payload["boresight_compaction_stats"],
            )
            beam_finalize_stage_summary["boresight_compaction_stats"] = dict(
                boresight_compaction_stats
            )

        power_result = fused_payload.get("power_result")
        sat_beam_counts_used_full = fused_payload.get("sat_beam_counts_used_full")
        diag_result = fused_payload.get("diag_result")
        debug_direct_epfd_stats = list(fused_payload.get("debug_direct_epfd_stats") or [])
        sat_eligible_mask = fused_payload.get("sat_eligible_mask")

        if power_stage_summary is not None:
            power_stage_summary = _capture_stage_summary(power_stage_summary)
            if isinstance(fused_payload.get("power_observed_memory"), Mapping):
                power_stage_summary["observed_power_slab_memory"] = _merge_max_numeric_mapping(
                    power_stage_summary.get("observed_power_slab_memory"),
                    fused_payload.get("power_observed_memory"),
                )
            power_stage_summary["hot_path_device_to_host_copy_count"] = hot_path_host_copy_count
            power_stage_summary["hot_path_device_to_host_copy_bytes"] = hot_path_host_copy_bytes
            power_stage_summary["device_scalar_sync_count"] = device_scalar_sync_count
            last_stage_memory_summary = _finalize_stage_summary(
                power_stage_summary,
                predicted_peak_bytes=int(predicted_gpu_power_peak_bytes),
            )
        else:
            last_stage_memory_summary = _finalize_stage_summary(
                beam_finalize_stage_summary,
                predicted_peak_bytes=int(predicted_gpu_finalize_peak_bytes),
            )

        if profile_stages:
            stage_start = perf_counter()

        return {
            "power_result": power_result,
            "sat_idx_g": sat_idx_g,
            "sat_topo_visible": sat_topo_visible,
            "sat_azel_visible": sat_azel_visible,
            "orbit_radius_eff": orbit_radius_eff,
            "sat_beam_counts_used_full": sat_beam_counts_used_full,
            "sat_eligible_mask": sat_eligible_mask if write_sat_eligible_mask else None,
            "diag_result": diag_result,
            "debug_direct_epfd_stats": debug_direct_epfd_stats,
            "beam_finalize_substage_timings": dict(beam_finalize_substage_timings),
            "cell_link_library_chunk_telemetry": dict(
                cell_link_stage_summary.get("chunk_telemetry") or {}
            ),
            "cell_link_library_stage_memory_summary": _finalize_stage_summary(
                cell_link_stage_summary,
                predicted_peak_bytes=int(predicted_gpu_propagation_peak_bytes),
            ),
            "beam_finalize_stage_memory_summary": _finalize_stage_summary(
                beam_finalize_stage_summary,
                predicted_peak_bytes=int(predicted_gpu_finalize_peak_bytes),
            ),
            "power_stage_memory_summary": (
                {}
                if power_stage_summary is None
                else _finalize_stage_summary(
                    power_stage_summary,
                    predicted_peak_bytes=int(predicted_gpu_power_peak_bytes),
                )
            ),
            "stage_start": stage_start,
            "stage_memory_summary": dict(last_stage_memory_summary),
            "beam_finalize_chunk_shape": dict(beam_finalize_chunk_shape),
            "boresight_compaction_stats": dict(boresight_compaction_stats),
            "hot_path_device_to_host_copy_count": int(hot_path_host_copy_count),
            "hot_path_device_to_host_copy_bytes": int(hot_path_host_copy_bytes),
            "device_scalar_sync_count": int(device_scalar_sync_count),
        }

    stage_profiler = _DirectEpfdGpuStageProfiler(cp, enabled=profile_stages)
    power_stage_summary: dict[str, Any] | None = None
    beam_finalize_chunk_shape: dict[str, Any] = {}
    boresight_compaction_stats: dict[str, Any] = {}
    spectral_slab_i = int(max(1, spectral_slab))
    spectral_weight_scratch_cache: dict[tuple[int, int, bool], dict[str, Any]] = {}
    slab_iter = iter(link_library.iter_direct_epfd_beam_slabs(**beam_finalize_kwargs))
    while True:
        if _query_direct_epfd_cancel_mode(cancel_callback) == "force":
            raise _RunCancellationRequested("force", "beam_slab_boundary")
        slab_stage_token = stage_profiler.start("beam_finalize")
        try:
            slab_info = next(slab_iter)
        except StopIteration:
            break
        except Exception as exc:
            if _is_direct_gpu_out_of_memory(exc, cp=cp):
                _raise_stage_oom("beam_finalize", exc, beam_finalize_stage_summary)
            raise
        stage_profiler.stop(slab_stage_token, stage_timings=stage_timings)
        beam_finalize_stage_summary = _capture_stage_summary(beam_finalize_stage_summary)
        if isinstance(slab_info.get("substage_timings"), Mapping):
            for name, value in dict(slab_info["substage_timings"]).items():
                try:
                    _accumulate_profile_timing(
                        beam_finalize_substage_timings,
                        str(name),
                        float(value),
                    )
                except Exception:
                    continue
            beam_finalize_stage_summary["substage_timings"] = dict(beam_finalize_substage_timings)
        if isinstance(slab_info.get("observed_memory"), Mapping):
            aggregated_finalize_memory = dict(
                beam_finalize_stage_summary.get("observed_finalize_slab_memory") or {}
            )
            for name, value in dict(slab_info["observed_memory"]).items():
                if value is None:
                    continue
                try:
                    numeric_value = int(value)
                except Exception:
                    aggregated_finalize_memory[str(name)] = value
                    continue
                aggregated_finalize_memory[str(name)] = int(
                    max(int(aggregated_finalize_memory.get(str(name), 0) or 0), numeric_value)
                )
            beam_finalize_stage_summary["observed_finalize_slab_memory"] = (
                aggregated_finalize_memory
            )
        if isinstance(slab_info.get("estimated_component_bytes"), Mapping):
            beam_finalize_stage_summary["estimated_component_bytes"] = dict(
                slab_info["estimated_component_bytes"]
            )
        if isinstance(slab_info.get("chunk_shape"), Mapping):
            beam_finalize_chunk_shape = dict(slab_info["chunk_shape"])
            beam_finalize_stage_summary["chunk_shape"] = dict(beam_finalize_chunk_shape)
        if isinstance(slab_info.get("compaction_stats"), Mapping):
            boresight_compaction_stats = _merge_max_numeric_mapping(
                boresight_compaction_stats,
                slab_info["compaction_stats"],
            )
            beam_finalize_stage_summary["boresight_compaction_stats"] = dict(
                boresight_compaction_stats
            )
        last_stage_memory_summary = dict(beam_finalize_stage_summary)

        slab_result = slab_info["result"]
        slab_time_start = int(slab_info["time_start"])
        slab_time_stop = int(slab_info["time_stop"])
        slab_sky_start = int(slab_info["sky_start"])
        slab_sky_stop = int(slab_info["sky_stop"])
        if debug_direct_epfd and "debug_direct_epfd_stats" in slab_result:
            slab_debug = dict(slab_result["debug_direct_epfd_stats"])
            slab_debug.update(
                time_start=slab_time_start,
                time_stop=slab_time_stop,
                sky_start=slab_sky_start,
                sky_stop=slab_sky_stop,
            )
            debug_direct_epfd_stats.append(slab_debug)

        if write_sat_beam_counts_used:
            counts_key = "sat_beam_counts_used_full"
            counts_sample = cp.asarray(slab_result[counts_key])

            expected_sat_count = int(sat_topo_ras_station.shape[1])
            if boresight_active:
                if counts_sample.ndim != 3 or int(counts_sample.shape[2]) != expected_sat_count:
                    raise ValueError(
                        "sat_beam_counts_used_full must have full-network boresight shape "
                        f"(T_slab, sky_slab, {expected_sat_count}); got {tuple(counts_sample.shape)!r}."
                    )
            else:
                if counts_sample.ndim != 2 or int(counts_sample.shape[1]) != expected_sat_count:
                    raise ValueError(
                        "sat_beam_counts_used_full must have full-network shape "
                        f"(T_slab, {expected_sat_count}); got {tuple(counts_sample.shape)!r}."
                    )

            if sat_beam_counts_used_full is None:
                full_shape = (
                    (int(time_count_local), int(pointings["azimuth_deg"].shape[-1]), expected_sat_count)
                    if boresight_active
                    else (int(time_count_local), expected_sat_count)
                )
                sat_beam_counts_used_full = cp.zeros(full_shape, dtype=counts_sample.dtype)
            if boresight_active:
                sat_beam_counts_used_full[
                    slab_time_start:slab_time_stop,
                    slab_sky_start:slab_sky_stop,
                    :,
                ] = counts_sample
            else:
                sat_beam_counts_used_full[slab_time_start:slab_time_stop, :] = counts_sample

        if profile_stages:
            diag_keys = (
                "ras_retargeted_count",
                "ras_reserved_count",
                "direct_kept_count",
                "repaired_link_count",
                "dropped_link_count",
            )
            if diag_result is None:
                diag_shape = (
                    (int(time_count_local), int(pointings["azimuth_deg"].shape[-1]))
                    if boresight_active
                    else (int(time_count_local),)
                )
                diag_result = {
                    key: cp.zeros(diag_shape, dtype=cp.int32) for key in diag_keys
                }
            for key in diag_keys:
                if boresight_active:
                    diag_result[key][
                        slab_time_start:slab_time_stop,
                        slab_sky_start:slab_sky_stop,
                    ] = cp.asarray(slab_result[key])
                else:
                    diag_result[key][slab_time_start:slab_time_stop] = cp.asarray(
                        slab_result[key]
                    )

        if any_power_outputs:
            beam_idx_slab = cp.asarray(slab_result["beam_idx"])
            beam_alpha_slab = cp.asarray(slab_result["beam_alpha_rad"])
            beam_beta_slab = cp.asarray(slab_result["beam_beta_rad"])
            _set_direct_epfd_progress_phase(
                pbar,
                enable_progress_bars=enable_progress_bars,
                progress_desc_mode=progress_desc_mode,
                phase="power_accumulation",
            )
            _emit_direct_epfd_progress(
                progress_callback,
                kind="phase",
                phase="power_accumulation",
                iteration_index=int(ii),
                batch_index=int(bi),
                time_start=int(slab_time_start),
                time_stop=int(slab_time_stop),
                sky_start=int(slab_sky_start),
                sky_stop=int(slab_sky_stop),
                description=_direct_epfd_progress_text(progress_desc_mode, "power_accumulation"),
            )
            telescope_az = None
            telescope_el = None
            if pointings is not None:
                if boresight_active:
                    telescope_az = pointings["azimuth_deg"][
                        slab_time_start:slab_time_stop,
                        slab_sky_start:slab_sky_stop,
                    ]
                    telescope_el = pointings["elevation_deg"][
                        slab_time_start:slab_time_stop,
                        slab_sky_start:slab_sky_stop,
                    ]
                else:
                    telescope_az = pointings["azimuth_deg"][
                        slab_time_start:slab_time_stop
                    ]
                    telescope_el = pointings["elevation_deg"][
                        slab_time_start:slab_time_stop
                    ]
            if power_stage_summary is None:
                power_stage_summary = _start_direct_epfd_stage_memory_summary(
                    "power_accumulation",
                    cp=cp,
                    session=session,
                )
            try:
                _raise_if_direct_epfd_stage_live_fit_is_unsafe(
                    stage="power_accumulation",
                    host_peak_bytes=int(predicted_host_peak_bytes),
                    gpu_peak_bytes=int(predicted_gpu_power_peak_bytes),
                    host_effective_budget_bytes=int(host_effective_budget_bytes),
                    gpu_effective_budget_bytes=int(gpu_effective_budget_bytes),
                    scheduler_active_target_fraction=float(scheduler_active_target_fraction),
                    live_host_snapshot=_runtime_host_memory_snapshot(),
                    live_gpu_snapshot=_runtime_gpu_memory_snapshot(cp, session),
                )
            except _DirectGpuOutOfMemory as exc:
                _raise_stage_oom("power_accumulation", exc, power_stage_summary)
            inner_spectral_slab = int(
                max(1, min(spectral_slab_i, max(1, slab_time_stop - slab_time_start)))
            )
            for spectral_time_start in range(slab_time_start, slab_time_stop, inner_spectral_slab):
                spectral_time_stop = min(slab_time_stop, spectral_time_start + inner_spectral_slab)
                slab_local_start = int(spectral_time_start - slab_time_start)
                slab_local_stop = int(spectral_time_stop - slab_time_start)

                spectral_weight_slab = None
                if dynamic_spectrum_state is not None:
                    spectrum_stage_token = stage_profiler.start("spectrum_activity_weighting")
                    spectral_weight_slab = _compute_cell_spectral_weight_slab_device(
                        cp,
                        dynamic_spectrum_state=dynamic_spectrum_state,
                        time_start=spectral_time_start,
                        time_stop=spectral_time_stop,
                        scratch_cache=spectral_weight_scratch_cache,
                    )
                    stage_profiler.stop(spectrum_stage_token, stage_timings=stage_timings)
                elif cell_spectral_weight_dev is not None:
                    spectral_weight_slab = cell_spectral_weight_dev[
                        spectral_time_start:spectral_time_stop,
                        :,
                    ]

                spectral_telescope_az = None
                spectral_telescope_el = None
                if telescope_az is not None:
                    spectral_telescope_az = telescope_az[slab_local_start:slab_local_stop, ...]
                if telescope_el is not None:
                    spectral_telescope_el = telescope_el[slab_local_start:slab_local_stop, ...]

                power_stage_token = stage_profiler.start("power_accumulation")
                try:
                    power_slab = session.accumulate_ras_power(
                        s1528_pattern_context=s1528_pattern_context,
                        ras_pattern_context=ras_pattern_context,
                        sat_topo=sat_topo_visible[spectral_time_start:spectral_time_stop, :, :],
                        sat_azel=sat_azel_visible[spectral_time_start:spectral_time_stop, :, :],
                        beam_idx=beam_idx_slab[slab_local_start:slab_local_stop, ...],
                        beam_alpha_rad=beam_alpha_slab[slab_local_start:slab_local_stop, ...],
                        beam_beta_rad=beam_beta_slab[slab_local_start:slab_local_stop, ...],
                        telescope_azimuth_deg=spectral_telescope_az,
                        telescope_elevation_deg=spectral_telescope_el,
                        orbit_radius_m_per_sat=orbit_radius_eff,
                        observer_alt_km=observer_alt_km_ras_station,
                        atmosphere_lut_context=atmosphere_context,
                        spectrum_plan_context=spectrum_plan_context,
                        cell_spectral_weight=spectral_weight_slab,
                        bandwidth_mhz=float(power_input["bandwidth_mhz"]),
                        power_input_quantity=str(power_input["power_input_quantity"]),
                        pfd0_dbw_m2_mhz=None,
                        target_pfd_dbw_m2_mhz=power_input["target_pfd_dbw_m2_mhz"],
                        target_pfd_dbw_m2_channel=power_input["target_pfd_dbw_m2_channel"],
                        satellite_ptx_dbw_mhz=power_input["satellite_ptx_dbw_mhz"],
                        satellite_ptx_dbw_channel=power_input["satellite_ptx_dbw_channel"],
                        satellite_eirp_dbw_mhz=power_input["satellite_eirp_dbw_mhz"],
                        satellite_eirp_dbw_channel=power_input["satellite_eirp_dbw_channel"],
                        power_variation_mode=str(power_input.get("power_variation_mode", "fixed")),
                        power_range_min_dbw_channel=power_input.get("power_range_min_dbw_channel"),
                        power_range_max_dbw_channel=power_input.get("power_range_max_dbw_channel"),
                        n_links=int(nco),
                        ras_service_cell_index=int(ras_service_cell_index),
                        target_alt_km=float(target_alt_km),
                        use_ras_station_alt_for_co=bool(use_ras_station_alt_for_co),
                        include_epfd=bool(write_epfd),
                        include_prx_total=bool(write_prx_total),
                        include_per_satellite_prx=bool(
                            write_per_satellite_prx_ras_station or write_prx_elevation_heatmap
                        ),
                        include_total_pfd=bool(write_total_pfd_ras_station),
                        include_per_satellite_pfd=bool(write_per_satellite_pfd_ras_station),
                        peak_pfd_lut_context=peak_pfd_lut_context,
                        max_surface_pfd_dbw_m2_channel=max_surface_pfd_dbw_m2_channel,
                        max_surface_pfd_dbw_m2_mhz=max_surface_pfd_dbw_m2_mhz,
                        surface_pfd_cap_mode=str(surface_pfd_cap_mode),
                        surface_pfd_stats_enabled=bool(surface_pfd_stats_enabled),
                        working_memory_budget_bytes=power_memory_budget_bytes,
                        sky_slab=power_sky_slab,
                        return_device=True,
                    )
                except Exception as exc:
                    if _is_direct_gpu_out_of_memory(exc, cp=cp):
                        _raise_stage_oom("power_accumulation", exc, power_stage_summary)
                    raise
                stage_profiler.stop(power_stage_token, stage_timings=stage_timings)
                power_stage_summary = _capture_stage_summary(power_stage_summary)
                if power_result is None:
                    try:
                        power_result = _allocate_direct_epfd_power_result_device(
                            cp,
                            power_slab,
                            time_count=time_count_local,
                            sky_count=(
                                int(pointings["azimuth_deg"].shape[-1]) if boresight_active else 1
                            ),
                            boresight_active=boresight_active,
                        )
                    except Exception as exc:
                        if _is_direct_gpu_out_of_memory(exc, cp=cp):
                            _raise_stage_oom("power_accumulation", exc, power_stage_summary)
                        raise
                    power_stage_summary = _capture_stage_summary(power_stage_summary)
                try:
                    _store_direct_epfd_power_slab_result(
                        power_result,
                        power_slab,
                        boresight_active=boresight_active,
                        time_start=spectral_time_start,
                        time_stop=spectral_time_stop,
                        sky_start=slab_sky_start,
                        sky_stop=slab_sky_stop,
                    )
                except Exception as exc:
                    if _is_direct_gpu_out_of_memory(exc, cp=cp):
                        _raise_stage_oom("power_accumulation", exc, power_stage_summary)
                    raise
                power_stage_summary = _capture_stage_summary(power_stage_summary)
                last_stage_memory_summary = dict(power_stage_summary)

    if profile_stages:
        stage_profiler.finalize(stage_timings)
        stage_start = perf_counter()
    if power_stage_summary is None:
        beam_finalize_stage_summary = _capture_stage_summary(beam_finalize_stage_summary)
        beam_finalize_stage_summary["substage_timings"] = dict(beam_finalize_substage_timings)
        last_stage_memory_summary = _finalize_stage_summary(
            beam_finalize_stage_summary,
            predicted_peak_bytes=int(predicted_gpu_finalize_peak_bytes),
        )
    elif isinstance(last_stage_memory_summary, Mapping):
        last_stage_memory_summary = _finalize_stage_summary(
            last_stage_memory_summary,
            predicted_peak_bytes=int(predicted_gpu_power_peak_bytes),
        )

    return {
        "power_result": power_result,
        "sat_idx_g": sat_idx_g,
        "sat_topo_visible": sat_topo_visible,
        "sat_azel_visible": sat_azel_visible,
        "orbit_radius_eff": orbit_radius_eff,
        "sat_beam_counts_used_full": sat_beam_counts_used_full,
        "sat_eligible_mask": (
            getattr(link_library, "eligible_mask_cp", None)
            if write_sat_eligible_mask
            else None
        ),
        "diag_result": diag_result,
        "debug_direct_epfd_stats": debug_direct_epfd_stats,
        "beam_finalize_substage_timings": dict(beam_finalize_substage_timings),
        "cell_link_library_chunk_telemetry": dict(
            cell_link_stage_summary.get("chunk_telemetry") or {}
        ),
        "cell_link_library_stage_memory_summary": _finalize_stage_summary(
            cell_link_stage_summary,
            predicted_peak_bytes=int(predicted_gpu_propagation_peak_bytes),
        ),
        "beam_finalize_stage_memory_summary": _finalize_stage_summary(
            beam_finalize_stage_summary,
            predicted_peak_bytes=int(predicted_gpu_finalize_peak_bytes),
        ),
        "power_stage_memory_summary": (
            {}
            if power_stage_summary is None
            else _finalize_stage_summary(
                power_stage_summary,
                predicted_peak_bytes=int(predicted_gpu_power_peak_bytes),
            )
        ),
        "stage_start": stage_start,
        "stage_memory_summary": dict(last_stage_memory_summary),
        "beam_finalize_chunk_shape": dict(beam_finalize_chunk_shape),
        "boresight_compaction_stats": dict(boresight_compaction_stats),
    }


def _prepare_multi_system_extra_context(
    *,
    session: Any,
    sys_kw: dict[str, Any],
    method: Any,
    any_power_outputs: bool,
    any_receive_outputs: bool,
    gpu_compute_dtype: Any,
    profile_stages: bool,
    cp_module: Any,
    observer_alt_km_ras_station: float,
    target_alt_km: float,
    include_atmosphere: bool,
    atmosphere_elev_bin_deg: float,
    atmosphere_elev_min_deg: float,
    atmosphere_elev_max_deg: float,
    atmosphere_max_path_length_km: float,
    ras_station_elev_range: tuple[Any, Any],
    normalized_cell_activity_mode: str,
    normalized_split_denominator_mode: str,
    n_cells_total: int,
    sys_idx: int,
) -> dict[str, Any]:
    """Prepare per-system GPU contexts for a secondary system in multi-system mode.

    Returns a dict containing all per-system contexts and parameters needed by
    the batch loop to run propagation + power computation for this system.
    """
    sys_tle_list = np.asarray(sys_kw["tle_list"], dtype=object)
    sys_n_sats = int(sys_tle_list.size)

    sys_satellite_context = session.prepare_satellite_context(
        sys_tle_list,
        method=method,
    )

    # Pattern context
    sys_pattern_kwargs = sys_kw.get("pattern_kwargs", {})
    sys_wavelength = sys_kw.get("wavelength", 1.0)
    sys_wavelength_m = float(
        u.Quantity(sys_wavelength, copy=False).to_value(u.m)
        if hasattr(sys_wavelength, "to_value")
        else u.Quantity(sys_wavelength, u.m).to_value(u.m)
    )
    sys_frequency = sys_kw.get("frequency", 1.0)
    sys_frequency_ghz = float(
        u.Quantity(sys_frequency, copy=False).to_value(u.GHz)
        if hasattr(sys_frequency, "to_value")
        else u.Quantity(sys_frequency, u.GHz).to_value(u.GHz)
    )

    sys_s1528_pattern_context = None
    # UEMR mode can be carried either at the top-level kwargs (set by the
    # GUI's run-request builder) or inside pattern_kwargs (set by the
    # ``isotropic`` antenna model spec). Accept either source.
    sys_uemr_mode = (
        bool(sys_kw.get("uemr_mode", False))
        or bool(sys_pattern_kwargs.get("uemr_mode", False))
        or bool(sys_pattern_kwargs.get("isotropic", False))
    )
    if any_power_outputs:
        if sys_uemr_mode:
            # UEMR: isotropic per-satellite source — uses a minimal
            # pattern context that returns 0 dBi everywhere. Skips the
            # Lt / Gm / M.2101 parameter checks since none apply.
            sys_s1528_pattern_context = session.prepare_isotropic_pattern_context(
                wavelength_m=sys_wavelength_m,
            )
        elif "Lt" in sys_pattern_kwargs:
            sys_s1528_pattern_context = session.prepare_s1528_pattern_context(
                wavelength_m=sys_wavelength_m,
                lt_m=sys_pattern_kwargs["Lt"],
                lr_m=sys_pattern_kwargs["Lr"],
                slr_db=sys_pattern_kwargs["SLR"],
                l=int(sys_pattern_kwargs["l"]),
                far_sidelobe_start_deg=sys_pattern_kwargs["far_sidelobe_start"],
                far_sidelobe_level_db=sys_pattern_kwargs["far_sidelobe_level"],
                gm_db=sys_pattern_kwargs.get("Gm"),
            )
        elif "N_H" in sys_pattern_kwargs:
            def _qval(v, default=0.0):
                """Extract scalar from Quantity or plain number."""
                if hasattr(v, "value"):
                    return float(v.value)
                return float(v) if v is not None else float(default)

            sys_s1528_pattern_context = session.prepare_m2101_pattern_context(
                g_emax_db=_qval(sys_pattern_kwargs.get("G_Emax", 2.0)),
                a_m_db=_qval(sys_pattern_kwargs.get("A_m", 30.0)),
                sla_nu_db=_qval(sys_pattern_kwargs.get("SLA_nu", 30.0)),
                phi_3db_deg=_qval(sys_pattern_kwargs.get("phi_3db", 120.0)),
                theta_3db_deg=_qval(sys_pattern_kwargs.get("theta_3db", 120.0)),
                d_h=_qval(sys_pattern_kwargs.get("d_H", 0.5)),
                d_v=_qval(sys_pattern_kwargs.get("d_V", 0.5)),
                n_h=int(_qval(sys_pattern_kwargs.get("N_H", 28))),
                n_v=int(_qval(sys_pattern_kwargs.get("N_V", 28))),
                wavelength_m=float(sys_wavelength_m),
            )
        else:
            sys_s1528_pattern_context = session.prepare_s1528_rec12_pattern_context(
                wavelength_m=sys_wavelength_m,
                gm_dbi=sys_pattern_kwargs["Gm"],
                ln_db=sys_pattern_kwargs.get("LN", -15.0),
                z=float(sys_pattern_kwargs.get("z", 1.0)),
                diameter_m=sys_pattern_kwargs.get("D"),
            )

    # RAS pattern (shared antenna, but uses per-system wavelength)
    sys_ras_station_ant_diam = sys_kw.get("ras_station_ant_diam", None)
    if sys_ras_station_ant_diam is not None:
        sys_ras_station_ant_diam_m = float(
            u.Quantity(sys_ras_station_ant_diam, copy=False).to_value(u.m)
            if hasattr(sys_ras_station_ant_diam, "to_value")
            else u.Quantity(sys_ras_station_ant_diam, u.m).to_value(u.m)
        )
    else:
        sys_ras_station_ant_diam_m = None
    sys_ras_pattern_context = None
    if any_receive_outputs and sys_ras_station_ant_diam_m is not None:
        sys_ras_pattern_context = session.prepare_ras_pattern_context(
            diameter_m=sys_ras_station_ant_diam_m,
            wavelength_m=sys_wavelength_m,
        )

    # Atmosphere context (per-system if frequency differs)
    sys_atmosphere_context = None
    if include_atmosphere and any_power_outputs:
        sys_atmosphere_context = session.prepare_atmosphere_lut_context(
            frequency_ghz=sys_frequency_ghz,
            altitude_km_values=np.asarray(
                [observer_alt_km_ras_station, float(target_alt_km)],
                dtype=np.float32,
            ),
            bin_deg=atmosphere_elev_bin_deg,
            elev_min_deg=atmosphere_elev_min_deg,
            elev_max_deg=atmosphere_elev_max_deg,
            max_path_length_km=atmosphere_max_path_length_km,
        )

    # Spectrum plan context
    sys_bandwidth_mhz = float(sys_kw.get("bandwidth_mhz", 5.0))
    sys_spectrum_plan = sys_kw.get("spectrum_plan", None)
    # pfd0_dbw_m2_mhz is the legacy target-PFD baseline; None when the
    # caller picked Ptx/EIRP. Coerce only if present, otherwise pass
    # None through (the normaliser treats None as "no baseline").
    _sys_pfd0 = sys_kw.get("pfd0_dbw_m2_mhz", -83.5)
    sys_power_input = normalize_direct_epfd_power_input(
        bandwidth_mhz=sys_bandwidth_mhz,
        power_input_quantity=sys_kw.get("power_input_quantity", "target_pfd"),
        power_input_basis=sys_kw.get("power_input_basis", "per_mhz"),
        pfd0_dbw_m2_mhz=None if _sys_pfd0 is None else float(_sys_pfd0),
        target_pfd_dbw_m2_mhz=sys_kw.get("target_pfd_dbw_m2_mhz"),
        target_pfd_dbw_m2_channel=sys_kw.get("target_pfd_dbw_m2_channel"),
        satellite_ptx_dbw_mhz=sys_kw.get("satellite_ptx_dbw_mhz"),
        satellite_ptx_dbw_channel=sys_kw.get("satellite_ptx_dbw_channel"),
        satellite_eirp_dbw_mhz=sys_kw.get("satellite_eirp_dbw_mhz"),
        satellite_eirp_dbw_channel=sys_kw.get("satellite_eirp_dbw_channel"),
        power_variation_mode=sys_kw.get("power_variation_mode", "fixed"),
        power_range_min_db=sys_kw.get("power_range_min_db"),
        power_range_max_db=sys_kw.get("power_range_max_db"),
    )
    # Per-system cell count from the system's own active grid
    if "active_cell_longitudes" in sys_kw:
        sys_n_cells = int(np.asarray(
            u.Quantity(sys_kw["active_cell_longitudes"], copy=False).to_value(u.deg),
            dtype=np.float64,
        ).reshape(-1).size)
    else:
        sys_n_cells = n_cells_total
    sys_spectrum_plan_effective = normalize_direct_epfd_spectrum_plan(
        spectrum_plan=sys_spectrum_plan,
        channel_bandwidth_mhz=float(sys_power_input["bandwidth_mhz"]),
        split_total_group_denominator_mode=normalized_split_denominator_mode,
        active_cell_count=sys_n_cells,
        active_cell_reuse_slot_ids=(
            None if sys_spectrum_plan is None else sys_spectrum_plan.get("active_cell_reuse_slot_ids")
        ),
    )
    sys_spectrum_plan_context = None
    if sys_spectrum_plan_effective is not None and any_power_outputs:
        sys_spectrum_plan_context = session.prepare_spectrum_plan_context(
            reuse_factor=int(sys_spectrum_plan_effective["reuse_factor"]),
            groups_per_cell=int(sys_spectrum_plan_effective["channel_groups_per_cell"]),
            channel_bandwidth_mhz=float(sys_spectrum_plan_effective["channel_bandwidth_mhz"]),
            service_band_start_mhz=float(sys_spectrum_plan_effective["service_band_start_mhz"]),
            service_band_stop_mhz=float(sys_spectrum_plan_effective["service_band_stop_mhz"]),
            ras_band_start_mhz=float(sys_spectrum_plan_effective["ras_receiver_band_start_mhz"]),
            ras_band_stop_mhz=float(sys_spectrum_plan_effective["ras_receiver_band_stop_mhz"]),
            power_policy=str(sys_spectrum_plan_effective["multi_group_power_policy"]),
            cell_reuse_slot_ids=np.asarray(
                sys_spectrum_plan_effective["active_cell_reuse_slot_ids"], dtype=np.int32,
            ),
            cell_leakage_factors=np.asarray(
                sys_spectrum_plan_effective["cell_leakage_factors"], dtype=np.float32,
            ),
            cell_group_leakage_factors=np.asarray(
                sys_spectrum_plan_effective["cell_group_leakage_factors"], dtype=np.float32,
            ),
            cell_group_valid_mask=np.asarray(
                sys_spectrum_plan_effective["cell_group_valid_mask"], dtype=bool,
            ),
            configured_group_counts_per_cell=np.asarray(
                sys_spectrum_plan_effective["configured_group_counts_per_cell"], dtype=np.int32,
            ),
            slot_edges_mhz=np.asarray(
                sys_spectrum_plan_effective["slot_edges_mhz"], dtype=np.float64,
            ),
            mask_points_mhz=np.asarray(
                sys_spectrum_plan_effective["unwanted_emission_mask_points_mhz"], dtype=np.float64,
            ),
            tx_reference_frequencies_mhz=np.asarray(
                sys_spectrum_plan_effective["tx_reference_frequencies_mhz"], dtype=np.float64,
            ),
            ras_reference_frequencies_mhz=np.asarray(
                sys_spectrum_plan_effective["ras_reference_frequencies_mhz"], dtype=np.float64,
            ),
        )

    # Per-satellite arrays
    sys_nco = int(sys_kw.get("nco", 1))
    sys_nbeam = int(sys_kw.get("nbeam", 1))
    sys_selection_strategy = sys_kw.get("selection_strategy", "random")
    sys_selection_mode = _normalize_direct_epfd_selection_strategy(sys_selection_strategy)

    # Prepare per-system observer context if the system has its own grid
    sys_observer_context = None
    if "observer_arr" in sys_kw:
        sys_observer_arr = np.asarray(sys_kw["observer_arr"], dtype=object)
        sys_observer_context = session.prepare_observer_context(sys_observer_arr)

    # Prepare the per-system surface-PFD cap LUT when the cap is enabled.
    # The LUT atomatically mirrors the host run's atmosphere flag (via the
    # ``atmosphere_lut_context`` passed in) and the pattern-eval mode set
    # by ``set_pattern_eval_mode`` / the Review & Run control.
    sys_peak_pfd_lut_context = None
    sys_surface_cap_enabled = bool(sys_kw.get("max_surface_pfd_enabled", False))
    sys_surface_cap_mode = str(sys_kw.get("surface_pfd_cap_mode", "per_beam"))
    sys_surface_stats_enabled = bool(sys_kw.get("surface_pfd_stats_enabled", False))
    sys_surface_cap_dbw_m2_mhz = sys_kw.get("max_surface_pfd_dbw_m2_mhz")
    sys_surface_cap_dbw_m2_channel = sys_kw.get("max_surface_pfd_dbw_m2_channel")
    if sys_surface_cap_enabled and any_power_outputs and sys_s1528_pattern_context is not None:
        try:
            sys_peak_pfd_lut_context = session.prepare_peak_pfd_lut_context(
                pattern_context=sys_s1528_pattern_context,
                sat_orbit_radius_m_per_sat=np.asarray(
                    sys_kw.get("sat_orbit_radius_m_per_sat", np.full(sys_n_sats, 7000e3)),
                    dtype=np.float64,
                ),
                atmosphere_lut_context=sys_atmosphere_context,
                target_alt_km=float(target_alt_km),
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Surface-PFD cap is enabled for system {sys_idx + 1} but "
                f"the pattern does not yet support it: {exc}"
            ) from exc

    # Cell chunking will be set after the scheduler runs (see _finalize_multi_system_cell_chunks)
    sys_cell_chunk = sys_n_cells
    sys_n_cell_chunks = 1

    ctx: dict[str, Any] = {
        "observer_context": sys_observer_context,
        "satellite_context": sys_satellite_context,
        "s1528_pattern_context": sys_s1528_pattern_context,
        "ras_pattern_context": sys_ras_pattern_context,
        "atmosphere_context": sys_atmosphere_context,
        "spectrum_plan_context": sys_spectrum_plan_context,
        "spectrum_plan_effective": sys_spectrum_plan_effective,
        "power_input": sys_power_input,
        "peak_pfd_lut_context": sys_peak_pfd_lut_context,
        "surface_pfd_cap_enabled": sys_surface_cap_enabled,
        "surface_pfd_cap_mode": sys_surface_cap_mode,
        "surface_pfd_stats_enabled": sys_surface_stats_enabled,
        "max_surface_pfd_dbw_m2_mhz": sys_surface_cap_dbw_m2_mhz,
        "max_surface_pfd_dbw_m2_channel": sys_surface_cap_dbw_m2_channel,
        "sat_min_elev_deg_per_sat_f64": np.asarray(
            sys_kw.get("sat_min_elevation_deg_per_sat", np.zeros(sys_n_sats)),
            dtype=np.float64,
        ),
        "sat_beta_max_deg_per_sat_f32": np.asarray(
            sys_kw.get("sat_beta_max_deg_per_sat", np.full(sys_n_sats, 90.0)),
            dtype=np.float32,
        ),
        "sat_belt_id_per_sat_i16": np.asarray(
            sys_kw.get("sat_belt_id_per_sat", np.zeros(sys_n_sats, dtype=np.int16)),
            dtype=np.int16,
        ),
        "orbit_radius_host": np.asarray(
            sys_kw.get("sat_orbit_radius_m_per_sat", np.full(sys_n_sats, 7000e3)),
            dtype=np.float32,
        ),
        "n_sats_total": sys_n_sats,
        "nco": sys_nco,
        "nbeam": sys_nbeam,
        "selection_mode": sys_selection_mode,
        "wavelength_m": sys_wavelength_m,
        "frequency_ghz": sys_frequency_ghz,
        "cell_activity_factor": float(sys_kw.get("cell_activity_factor", 1.0)),
        "cell_activity_mode": str(
            sys_kw.get("cell_activity_mode", normalized_cell_activity_mode)
        ),
        "cell_activity_seed_base": sys_kw.get("cell_activity_seed_base", 42001),
        "n_cells_total": sys_n_cells,
        "cell_chunk": sys_cell_chunk,
        "n_cell_chunks": sys_n_cell_chunks,
        "observer_arr": np.asarray(sys_kw.get("observer_arr", np.array([])), dtype=object)
        if "observer_arr" in sys_kw
        else None,
        "ras_service_cell_index": int(sys_kw.get("ras_service_cell_index", 0)),
        "ras_service_cell_active": bool(sys_kw.get("ras_service_cell_active", False)),
        "system_name": str(sys_kw.get("_system_name", f"System {sys_idx + 1}")),
        "storage_attrs": sys_kw.get("storage_attrs", {}),
    }
    # --- Per-system boresight parameters ---
    sys_boresight_theta1 = sys_kw.get("boresight_theta1", None)
    sys_boresight_theta2 = sys_kw.get("boresight_theta2", None)
    sys_boresight_theta1_deg = _optional_angle_deg(sys_boresight_theta1)
    sys_boresight_theta2_deg = _optional_angle_deg(sys_boresight_theta2)
    sys_boresight_active = bool(
        sys_boresight_theta1_deg is not None or sys_boresight_theta2_deg is not None
    )
    sys_boresight_theta2_cell_ids = sys_kw.get("boresight_theta2_cell_ids", None)
    if sys_boresight_theta2_cell_ids is not None:
        sys_boresight_theta2_cell_ids = np.asarray(sys_boresight_theta2_cell_ids, dtype=np.int32)
    ctx["boresight_active"] = sys_boresight_active
    ctx["boresight_theta1_deg"] = sys_boresight_theta1_deg
    ctx["boresight_theta2_deg"] = sys_boresight_theta2_deg
    ctx["boresight_theta2_cell_ids"] = sys_boresight_theta2_cell_ids
    # UEMR can arrive as a top-level kwarg or via pattern_kwargs
    # (isotropic / uemr_mode); accept either.
    _sys_pk = sys_kw.get("pattern_kwargs", {}) or {}
    ctx["uemr_mode"] = (
        bool(sys_kw.get("uemr_mode", False))
        or bool(_sys_pk.get("uemr_mode", False))
        or bool(_sys_pk.get("isotropic", False))
    )
    # Antenna-model identifier for HDF5 per-system attrs. Distinguish
    # "isotropic" UEMR systems from directive ones in the output file
    # so post-processing can branch.
    if ctx["uemr_mode"]:
        ctx["antenna_model"] = "isotropic"
    elif "Lt" in _sys_pk:
        ctx["antenna_model"] = "s1528_rec1_4"
    elif "N_H" in _sys_pk:
        ctx["antenna_model"] = "m2101"
    else:
        ctx["antenna_model"] = "s1528_rec1_2"
    return ctx


def _default_output_groups(n_systems: int) -> list[dict[str, Any]]:
    """Build default output groups: one per system + one combined."""
    groups: list[dict[str, Any]] = []
    for i in range(n_systems):
        groups.append({
            "name": f"System {i + 1}",
            "system_indices": [i],
            "enabled": True,
        })
    if n_systems > 1:
        groups.append({
            "name": "Combined",
            "system_indices": list(range(n_systems)),
            "enabled": True,
        })
    return groups


def _group_hdf5_prefix(group: dict[str, Any], n_systems: int) -> str:
    """Determine the HDF5 group prefix for an output group.

    - "Combined" group (all systems) writes to root ``""`` for backward compat.
    - Single-system groups write to ``"system_N/"`` for backward compat.
    - Custom multi-system groups write to ``"group/<sanitized_name>/"``.
    """
    indices = sorted(group.get("system_indices", []))
    name = str(group.get("name", ""))
    # Combined = all systems
    if indices == list(range(n_systems)) and n_systems > 1:
        return ""
    # Single-system group
    if len(indices) == 1:
        return f"system_{indices[0]}/"
    # Custom multi-system combination
    safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not safe_name:
        safe_name = "_".join(str(i) for i in indices)
    return f"group/{safe_name}/"


def _pad_visible_to_full(
    compact: np.ndarray,
    sat_idx: np.ndarray,
    n_sats_total: int,
    *,
    fill: float | int = 0,
) -> np.ndarray:
    """Pad a visible-satellite array to the full system satellite count.

    The satellite axis is identified as the axis whose length equals
    ``len(sat_idx)`` (the visible count).  All other axes keep their sizes.

    Parameters
    ----------
    compact : np.ndarray
        Array with one axis equal to ``len(sat_idx)``.  Common shapes:
        ``(T, S_vis)``, ``(T, sky, S_vis)``.
    sat_idx : np.ndarray
        1-D int32 index mapping visible → global satellite indices.
    n_sats_total : int
        Full satellite count for this system.
    fill : float or int
        Value used for non-visible satellite slots (0 or NaN).
    """
    n_vis = int(sat_idx.size)
    if compact.ndim == 0 or n_vis == 0:
        return compact
    # Check if the array already has the full satellite count on any axis.
    # If so, it's already full-network and doesn't need padding.
    for ax in range(compact.ndim):
        if int(compact.shape[ax]) == n_sats_total and n_sats_total != n_vis:
            return compact
    # Find the satellite axis: the one whose length matches n_vis.
    # Search from the LAST axis backwards because per-satellite arrays
    # always have satellites on a trailing dimension (axis -1 or -2).
    # Searching forward would incorrectly match the time axis when
    # T == S_vis (e.g., T=11 timesteps and 11 visible satellites).
    sat_axis = None
    for ax in range(compact.ndim - 1, -1, -1):
        if int(compact.shape[ax]) == n_vis:
            sat_axis = ax
            break
    if sat_axis is None:
        # No axis matches visible count — data may already be full-network
        return compact
    if int(compact.shape[sat_axis]) == n_sats_total:
        return compact  # already padded
    full_shape = list(compact.shape)
    full_shape[sat_axis] = n_sats_total
    if np.isnan(fill) if isinstance(fill, float) else False:
        full = np.full(full_shape, fill, dtype=compact.dtype)
    else:
        full = np.zeros(full_shape, dtype=compact.dtype)
        if fill != 0:
            full[:] = fill
    idx = [slice(None)] * compact.ndim
    idx[sat_axis] = sat_idx
    full[tuple(idx)] = compact
    return full


def _collapse_per_sat_to_2d(arr: np.ndarray) -> np.ndarray:
    """Collapse per-satellite array from (T, obs, S, sky) → (T, S) for non-boresight.

    Takes the max over the sky dimension and squeezes the observer dimension.
    No-op for arrays that are already 2D.
    """
    if arr.ndim <= 2:
        return arr
    # (T, obs, S, sky) → max over sky → (T, obs, S) → squeeze obs → (T, S)
    result = np.max(arr, axis=-1)
    while result.ndim > 2:
        result = result.squeeze(axis=1)
    return result


def _combine_multi_system_power_results_device(
    cp: Any,
    power_results: list[dict[str, Any] | None],
    n_skycells_s1586: int,
    boresight_active: bool,
) -> dict[str, Any] | None:
    """Combine power results from multiple systems by summing linear power.

    All power quantities (EPFD, Prx, PFD) are already in linear (Watts or
    W/m^2) space, so direct addition is correct.

    Parameters
    ----------
    cp : module
        CuPy module.
    power_results : list
        Per-system power_result dicts from ``_compute_gpu_direct_epfd_batch_device``.
    n_skycells_s1586 : int
        Number of sky cells.
    boresight_active : bool
        Whether boresight avoidance is active.

    Returns
    -------
    dict or None
        Combined power result with all keys summed across systems.
    """
    valid_results = [pr for pr in power_results if pr is not None]
    if not valid_results:
        return None
    if len(valid_results) == 1:
        single = dict(valid_results[0])
        single.pop("_spatially_uniform", None)
        return single

    # Start from a copy of the first system's result
    combined = dict(valid_results[0])

    # Keys that should be summed in linear power across systems.
    # Per-satellite keys are NOT summed (they have different satellite axes
    # per system); only aggregate keys are combined.
    _summable_keys = {
        "EPFD_W_m2",
        "Prx_total_W",
        "PFD_total_RAS_STATION_W_m2",
    }

    def _normalize_pair(lhs: Any, rhs: Any) -> tuple[Any, Any]:
        """Broadcast lhs/rhs to a common shape for linear power summation.

        Handles the three emit shapes produced by the kernels:
          * (T,)            — 3-D directive non-boresight & UEMR PFD_total
          * (T, 1, 1)       — legacy UEMR-bypass spatially-uniform
          * (T, 1, N_sky)   — 4-D directive boresight & sky-aware EPFD/Prx
        PFD/EPFD/Prx at the antenna site are independent of pointing
        direction for spatially-uniform emitters, so broadcasting a scalar-
        per-t quantity across the sky axis is physically correct.
        """
        if lhs.shape == rhs.shape:
            return lhs, rhs
        # Promote 1-D (T,) to (T, 1, 1) first so both sides are ≥3-D.
        if lhs.ndim == 1:
            lhs = lhs[:, None, None]
        if rhs.ndim == 1:
            rhs = rhs[:, None, None]
        if lhs.shape == rhs.shape:
            return lhs, rhs
        # Broadcast to the elementwise-max shape (works for (T,1,1) vs
        # (T,1,N_sky) and similar per-axis subset relations).
        target = tuple(max(a, b) for a, b in zip(lhs.shape, rhs.shape))
        return cp.broadcast_to(lhs, target), cp.broadcast_to(rhs, target)

    for key in _summable_keys:
        if key not in combined or combined[key] is None:
            continue
        for pr in valid_results[1:]:
            if key not in pr or pr[key] is None:
                continue
            lhs = combined[key]
            rhs = pr[key]
            try:
                lhs_n, rhs_n = _normalize_pair(lhs, rhs)
            except Exception as exc:
                raise ValueError(
                    f"_combine_multi_system_power_results_device: cannot combine "
                    f"key {key!r} with shapes {lhs.shape} vs {rhs.shape}: {exc}"
                ) from exc
            combined[key] = lhs_n + rhs_n

    combined.pop("_spatially_uniform", None)
    return combined


def run_gpu_direct_epfd(
    *,
    tle_list: np.ndarray,
    observer_arr: np.ndarray,
    active_cell_longitudes: u.Quantity | np.ndarray,
    sat_min_elevation_deg_per_sat: np.ndarray,
    sat_beta_max_deg_per_sat: np.ndarray,
    sat_belt_id_per_sat: np.ndarray,
    sat_orbit_radius_m_per_sat: np.ndarray,
    pattern_kwargs: Mapping[str, Any],
    wavelength: u.Quantity | float,
    ras_station_ant_diam: u.Quantity | float,
    frequency: u.Quantity | float,
    ras_station_elev_range: tuple[Any, Any],
    observer_alt_km_ras_station: float,
    storage_filename: str,
    base_start_time: Time,
    base_end_time: Time,
    timestep: float | u.Quantity,
    iteration_count: int,
    iteration_rng_seed: int,
    nco: int,
    nbeam: int,
    selection_strategy: str,
    ras_pointing_mode: str,
    ras_service_cell_index: int,
    ras_service_cell_active: bool,
    ras_guard_angle: u.Quantity | float,
    boresight_theta1: Any | None = None,
    boresight_theta2: Any | None = None,
    boresight_theta2_cell_ids: np.ndarray | None = None,
    include_atmosphere: bool = True,
    use_radio_horizon: bool = False,
    radio_horizon_tec: float = 30.0,
    atmosphere_elev_bin_deg: float = 0.1,
    atmosphere_elev_min_deg: float = 0.1,
    atmosphere_elev_max_deg: float = 90.0,
    atmosphere_max_path_length: u.Quantity | float = 10_000.0 * u.km,
    pfd0_dbw_m2_mhz: float = -83.5,
    bandwidth_mhz: float = 5.0,
    spectrum_plan: Mapping[str, Any] | None = None,
    power_input_quantity: str = "target_pfd",
    power_input_basis: str = "per_mhz",
    target_pfd_dbw_m2_mhz: float | None = None,
    target_pfd_dbw_m2_channel: float | None = None,
    satellite_ptx_dbw_mhz: float | None = None,
    satellite_ptx_dbw_channel: float | None = None,
    satellite_eirp_dbw_mhz: float | None = None,
    satellite_eirp_dbw_channel: float | None = None,
    power_variation_mode: str = "fixed",
    power_range_min_db: float | None = None,
    power_range_max_db: float | None = None,
    max_surface_pfd_enabled: bool = False,
    max_surface_pfd_dbw_m2_mhz: float | None = None,
    max_surface_pfd_dbw_m2_channel: float | None = None,
    surface_pfd_cap_mode: str = "per_beam",
    surface_pfd_stats_enabled: bool = False,
    target_alt_km: float = 0.0,
    use_ras_station_alt_for_co: bool = True,
    cell_activity_factor: float = 1.0,
    cell_activity_mode: str = "whole_cell",
    cell_activity_seed_base: int | None = 42001,
    split_total_group_denominator_mode: str = "configured_groups",
    beamforming_collapsed: bool = False,
    collapsed_baseline_eirp_dbw_hz: float = -55.6,
    collapsed_eval_freq_mhz: float = 2695.0,
    collapsed_ref_freq_mhz: float = 2000.0,
    host_memory_budget_gb: float = 4.0,
    gpu_memory_budget_gb: float = 4.0,
    memory_budget_mode: str = "hybrid",
    memory_headroom_profile: str = "balanced",
    scheduler_target_profile: str = "high_throughput",
    force_bulk_timesteps: int | None = None,
    force_cell_observer_chunk: int | None = None,
    finalize_memory_budget_bytes: int | None = None,
    power_memory_budget_bytes: int | None = None,
    export_memory_budget_bytes: int | None = None,
    power_sky_slab: int | None = None,
    debug_direct_epfd: bool = False,
    gpu_method: Any | None = None,
    gpu_compute_dtype: Any = np.float32,
    gpu_output_dtype: Any = np.float32,
    gpu_on_error: str = "raise",
    sat_frame: str = "xyz",
    gpu_pattern_eval_mode: str = "lut",
    gpu_precision_profile: str | None = None,
    enable_progress_bars: bool = True,
    enable_progress_desc_updates: bool = True,
    progress_desc_mode: str | None = None,
    writer_checkpoint_interval_s: float | None = 60.0,
    output_families: Mapping[str, Any] | None = None,
    store_eligible_mask: bool = False,
    profile_stages: bool = False,
    terminal_gpu_cleanup: bool = False,
    hdf5_compression: str | None = "lzf",
    hdf5_compression_opts: Any = None,
    storage_constants: Mapping[str, Any] | None = None,
    storage_attrs: Mapping[str, Any] | None = None,
    cupy_module: Any | None = None,
    gpu_module: Any | None = None,
    session: Any | None = None,
    gpu_device_id: int | None = None,
    progress_factory: Callable[..., Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_callback: Callable[[], str | None] | None = None,
    systems: list[dict[str, Any]] | None = None,
    output_system_groups: list[dict[str, Any]] | None = None,
    uemr_mode: bool = False,
) -> dict[str, Any]:
    """
    Run the notebook-facing GPU direct-EPFD workflow.

    Parameters
    ----------
    tle_list, observer_arr : np.ndarray
        Satellite and observer axes used by the GPU workflow.
    active_cell_longitudes : astropy.units.Quantity or np.ndarray
        ACTIVE-axis Earth-grid longitudes. Only the length is consumed by the
        runner, but the axis is explicit in the signature for notebook clarity.
    sat_min_elevation_deg_per_sat, sat_beta_max_deg_per_sat, sat_belt_id_per_sat, sat_orbit_radius_m_per_sat : np.ndarray
        Per-satellite metadata arrays aligned to ``tle_list``.
    pattern_kwargs : Mapping[str, Any]
        Antenna-pattern arguments compatible with
        ``prepare_s1528_pattern_context``.
    wavelength, ras_station_ant_diam, frequency : astropy.units.Quantity or float
        RF geometry inputs. Plain floats are interpreted as metres, metres, and
        GHz respectively.
    storage_filename : str
        Output HDF5 file path. The file is initialized by this helper.
    base_start_time, base_end_time : astropy.time.Time
        Base time span jittered independently for each iteration.
    timestep : float or astropy.units.Quantity
        Simulation timestep. Plain numeric values are interpreted as seconds.
    iteration_count, iteration_rng_seed, nco, nbeam : int
        Iteration count, jitter seed, links-per-cell, and beam-cap controls.
    selection_strategy : {"random", "random_pointing", "max_elevation", "maximum_elevation"}
        Direct-link selection strategy.
    ras_pointing_mode : {"ras_station", "cell_center"}
        Requested direct-beam pointing mode.
    ras_service_cell_index, ras_service_cell_active, ras_guard_angle : optional
        RAS-cell metadata used by the exact-RAS direct-beam path.
    boresight_theta1, boresight_theta2, boresight_theta2_cell_ids : optional
        Optional boresight-avoidance controls on the ACTIVE cell axis.
    cell_activity_factor : float, optional
        Independent Bernoulli activation probability applied per
        active unit on the GPU.
    cell_activity_mode : {"whole_cell", "per_channel"}, optional
        Whether the Bernoulli activity draw is sampled once per
        ``(time, cell)`` and applied to all configured groups, or
        independently per ``(time, cell, group)``.
    cell_activity_seed_base : int or None, optional
        Base seed for reproducible GPU-side cell-activity draws. Set to
        ``None`` to request nondeterministic GPU sampling.
    split_total_group_denominator_mode : {"configured_groups", "active_groups"}, optional
        When ``multi_group_power_policy="split_total_cell_power"`` and
        ``cell_activity_mode="per_channel"``, choose whether the cell-total
        power is divided by all configured groups or only by the groups that
        are active at that timestep.
    finalize_memory_budget_bytes, power_memory_budget_bytes, export_memory_budget_bytes : int or None, optional
        Developer-facing byte caps used to override the internal direct-EPFD
        stage budgets derived from ``gpu_memory_budget_gb``.
    power_sky_slab : int or None, optional
        Developer-facing override for boresight receive-power sky chunking.
    debug_direct_epfd : bool, optional
        Developer-facing GPU finalize debug mode. When enabled, the direct-EPFD
        finalize path records slab diagnostics and may run additional validation
        checks on small slabs.
    output_families : Mapping[str, Any] or None, optional
        Canonical output-family configuration controlling which dynamic and
        preaccumulated datasets are produced. When omitted, the default
        family configuration returned by :func:`default_output_families`
        is used.
    store_eligible_mask : bool, optional
        Persist the dense ``sat_eligible_mask`` payload for later exact
        pure-reroute beam-cap analysis. This increases host memory, writer
        cost, and file size, so it stays off by default.
    pfd0_dbw_m2_mhz : float, optional
        Backward-compatible alias for ``target_pfd_dbw_m2_mhz`` when
        ``power_input_quantity="target_pfd"`` and
        ``power_input_basis="per_mhz"``.
    bandwidth_mhz : float, optional
        Backward-compatible channel bandwidth alias. When ``spectrum_plan`` is
        provided, ``spectrum_plan["channel_bandwidth_mhz"]`` is canonical and
        this value is used only as a fallback.
    spectrum_plan : Mapping[str, Any] or None, optional
        Optional normalized spectrum/reuse description. When omitted, the
        runner preserves the historical single-channel full-overlap behavior.
        When provided, the receive-side raw outputs represent power integrated
        over the configured RAS receiver band after reuse-slot occupancy and
        unwanted-emission leakage weighting have been applied exactly once.
    power_input_quantity : {"target_pfd", "satellite_ptx", "satellite_eirp"}, optional
        Active service/transmit power quantity definition.
    power_input_basis : {"per_mhz", "per_channel"}, optional
        Whether the active quantity is entered as a spectral density or as a
        channel-total quantity.
    target_pfd_dbw_m2_mhz, target_pfd_dbw_m2_channel, satellite_ptx_dbw_mhz, satellite_ptx_dbw_channel, satellite_eirp_dbw_mhz, satellite_eirp_dbw_channel : float or None, optional
        Quantity-specific inputs. Only the field selected by
        ``power_input_quantity`` and ``power_input_basis`` is required; the
        complementary basis is derived from ``bandwidth_mhz``.
    storage_constants, storage_attrs : Mapping[str, Any] or None, optional
        Constant datasets and root attrs written immediately after output-file
        initialization.
    cupy_module, gpu_module, session, progress_factory : optional
        Dependency-injection hooks used by tests.
    systems : list[dict[str, Any]] or None, optional
        When provided with N > 1 entries, enables per-batch multi-system
        interleaving.  Each dict contains per-system parameters (TLEs,
        pattern, spectrum, power, etc.).  System 0 is the primary system
        whose parameters also populate the top-level keyword arguments.
        Within each time batch, all systems are propagated and computed
        independently, then their aggregate power results (EPFD, Prx_total,
        PFD_total) are summed in linear power before histogram accumulation
        and HDF5 export.  When ``None`` or length 1, behavior is unchanged.
    max_surface_pfd_enabled : bool, optional
        Whether to enforce the Service & Demand "max PFD on Earth surface"
        cap. When ``False`` (default), satellites radiate at the EIRP implied
        by the service power input (``target_pfd`` / ``satellite_ptx`` /
        ``satellite_eirp``) without any surface-PFD bound. When ``True``, the
        runner builds a peak-PFD LUT at run start and caps each beam's
        (or each satellite's, in aggregate mode) emitted EIRP so no point on
        Earth sees more than the configured limit.
    max_surface_pfd_dbw_m2_mhz, max_surface_pfd_dbw_m2_channel : float or None, optional
        The cap limit, expressed either per-MHz (regulatory PFD mask units)
        or per-channel (channel-total PFD). Exactly one of these must be
        provided when ``max_surface_pfd_enabled=True``. If both are supplied
        they must agree at ``bandwidth_mhz``.
    surface_pfd_cap_mode : {"per_beam", "per_satellite"}, optional
        ``"per_beam"`` (default) caps each beam's emitted EIRP independently
        based on its own peak surface PFD. ``"per_satellite"`` caps the
        aggregate surface PFD from all beams of one satellite — the only
        mode that bounds coincident or overlapping beams correctly. Both
        modes support the 3-D and 4-D (boresight-avoidance) paths as of
        the follow-up landing in 2026-Q2; the per-satellite mode in the
        4-D path is computed once per batch via the hoisted precomputed
        cap factor so the spectral-slab loop stays on the fast path.
        Supported transmit patterns: S.1528, S.1528 Rec 1.2, and
        ITU-R M.2101 phased arrays (the latter via a 2-D ``K(α, β)`` LUT).
    surface_pfd_stats_enabled : bool, optional
        When the cap is enabled, emit three per-batch scalar time series to
        the HDF5 output: ``surface_pfd_cap_n_beams_capped`` (int64),
        ``surface_pfd_cap_mean_cap_db`` (float32), and
        ``surface_pfd_cap_max_cap_db`` (float32). Useful for quantifying
        how often the cap bound and by how much. Default ``False``.

    Returns
    -------
    dict[str, Any]
        High-level run metadata including the effective RAS pointing mode,
        boresight activation flags, requested output names, and whether the
        dynamic execution loop was skipped.

    Notes
    -----
    ``gpu_memory_budget_gb`` is the configured hard cap for the run. The
    scheduler may temporarily lower the current-run effective GPU budget after
    live-fit pressure or OOM retries, but that reduction is not written back to
    project state or future runs. Progress payloads also expose process RSS
    (resident set size), which is the SCEPTer process RAM currently resident in
    physical memory rather than VRAM or total virtual memory.
    """
    if sat_frame != "xyz":
        raise RuntimeError(f"Unsupported SAT_FRAME={sat_frame!r} in this workflow.")

    _prepare_total = 10
    _prepare_step = [0]

    def _emit_prepare(description: str) -> None:
        _prepare_step[0] += 1
        _emit_direct_epfd_progress(
            progress_callback,
            kind="prepare",
            phase="prepare",
            prepare_index=_prepare_step[0],
            prepare_total=_prepare_total,
            description=description,
        )

    _emit_prepare("Validating inputs and resolving parameters...")

    observer_arr = np.asarray(observer_arr, dtype=object)
    n_cells_total = int(u.Quantity(active_cell_longitudes, copy=False).size)
    n_sats_total = int(np.asarray(tle_list, dtype=object).size)
    if n_cells_total < 1:
        raise RuntimeError("No active grid cells to process.")
    if nbeam is None or int(nbeam) <= 0:
        raise ValueError("run_gpu_direct_epfd requires a positive integer nbeam.")
    if not (0.0 <= float(cell_activity_factor) <= 1.0):
        raise ValueError("cell_activity_factor must lie in [0, 1].")
    normalized_cell_activity_mode = _normalize_direct_epfd_cell_activity_mode(
        cell_activity_mode
    )
    normalized_split_denominator_mode = (
        _normalize_direct_epfd_split_group_denominator_mode(
            split_total_group_denominator_mode
        )
    )
    # pfd0_dbw_m2_mhz is the legacy target-PFD pass-through — None when
    # the caller selected Ptx or EIRP. Coerce only if present; otherwise
    # leave it None for the normaliser (which treats None as "no target
    # PFD baseline was provided").
    _pfd0_coerced = (
        None if pfd0_dbw_m2_mhz is None else float(pfd0_dbw_m2_mhz)
    )
    power_input = normalize_direct_epfd_power_input(
        bandwidth_mhz=float(bandwidth_mhz),
        power_input_quantity=power_input_quantity,
        power_input_basis=power_input_basis,
        pfd0_dbw_m2_mhz=_pfd0_coerced,
        target_pfd_dbw_m2_mhz=target_pfd_dbw_m2_mhz,
        target_pfd_dbw_m2_channel=target_pfd_dbw_m2_channel,
        satellite_ptx_dbw_mhz=satellite_ptx_dbw_mhz,
        satellite_ptx_dbw_channel=satellite_ptx_dbw_channel,
        satellite_eirp_dbw_mhz=satellite_eirp_dbw_mhz,
        satellite_eirp_dbw_channel=satellite_eirp_dbw_channel,
        power_variation_mode=power_variation_mode,
        power_range_min_db=power_range_min_db,
        power_range_max_db=power_range_max_db,
    )
    spectrum_plan_effective = normalize_direct_epfd_spectrum_plan(
        spectrum_plan=spectrum_plan,
        channel_bandwidth_mhz=float(power_input["bandwidth_mhz"]),
        split_total_group_denominator_mode=normalized_split_denominator_mode,
        active_cell_count=int(n_cells_total),
        active_cell_reuse_slot_ids=(
            None
            if spectrum_plan is None
            else spectrum_plan.get("active_cell_reuse_slot_ids")
        ),
    )

    selection_mode = _normalize_direct_epfd_selection_strategy(selection_strategy)
    ras_pointing_mode_name = str(ras_pointing_mode).strip().lower()
    if ras_pointing_mode_name not in {"ras_station", "cell_center"}:
        raise ValueError("ras_pointing_mode must be 'ras_station' or 'cell_center'.")

    boresight_theta1_deg = _optional_angle_deg(boresight_theta1)
    boresight_theta2_deg = _optional_angle_deg(boresight_theta2)
    boresight_active = bool(
        boresight_theta1_deg is not None or boresight_theta2_deg is not None
    )
    if boresight_theta2_deg is not None and boresight_theta2_cell_ids is None:
        raise ValueError(
            "boresight_theta2_cell_ids must be provided when boresight_theta2 is enabled."
        )

    effective_ras_pointing_mode = ras_pointing_mode_name
    if effective_ras_pointing_mode == "ras_station" and not bool(ras_service_cell_active):
        effective_ras_pointing_mode = "cell_center"
    beam_generation_method = (
        "direct_link_ras_cell_retarget_local_repair"
        if effective_ras_pointing_mode == "ras_station"
        else "direct_link_cell_center"
    )

    _emit_prepare("Planning output families and memory schedules...")

    # UEMR mode may arrive via either the top-level ``uemr_mode`` kwarg
    # (set by the GUI when it knows explicitly) or via pattern_kwargs
    # (set by the ``isotropic`` antenna model spec in antenna.py). Derive
    # a single canonical flag and propagate it — without this, the batch
    # loop would receive uemr_mode=False and run the full beam library,
    # defeating the bypass even when the kernel would have skipped beams.
    uemr_mode = (
        bool(uemr_mode)
        or bool((pattern_kwargs or {}).get("uemr_mode", False))
        or bool((pattern_kwargs or {}).get("isotropic", False))
    )
    # When UEMR is active, beam-related output families have no data to
    # write (the beam library is bypassed). Pass uemr_mode through so
    # ``_resolve_direct_epfd_output_family_plan`` can force them off and
    # the HDF5 file doesn't carry stale beam artefacts.
    output_family_plan = _resolve_direct_epfd_output_family_plan(
        _resolve_output_family_configs(output_families=output_families),
        uemr_mode=bool(uemr_mode),
    )
    family_configs = output_family_plan["family_configs"]
    for family_name in ("prx_total_distribution", "prx_elevation_heatmap"):
        if family_name in family_configs:
            family_configs[family_name]["bandwidth_mhz"] = float(power_input["bandwidth_mhz"])
    write_epfd = bool(output_family_plan["write_epfd"])
    write_prx_total = bool(output_family_plan["write_prx_total"])
    write_per_satellite_prx_ras_station = bool(
        output_family_plan["write_per_satellite_prx_ras_station"]
    )
    write_total_pfd_ras_station = bool(output_family_plan["write_total_pfd_ras_station"])
    write_per_satellite_pfd_ras_station = bool(
        output_family_plan["write_per_satellite_pfd_ras_station"]
    )
    write_sat_beam_counts_used = bool(output_family_plan["write_sat_beam_counts_used"])
    write_sat_elevation_ras_station = bool(output_family_plan["write_sat_elevation_ras_station"])
    write_beam_demand_count = bool(output_family_plan["write_beam_demand_count"])
    write_prx_elevation_heatmap = bool(output_family_plan["preacc_prx_elevation_heatmap"])

    any_receive_outputs = bool(
        output_family_plan["needs_epfd"]
        or output_family_plan["needs_total_prx"]
        or output_family_plan["needs_per_satellite_prx"]
    )
    any_pfd_outputs = bool(
        output_family_plan["needs_total_pfd"] or output_family_plan["needs_per_satellite_pfd"]
    )
    any_power_outputs = bool(any_receive_outputs or any_pfd_outputs)
    activity_groups_per_cell = (
        1
        if spectrum_plan_effective is None
        else int(spectrum_plan_effective["channel_groups_per_cell"])
    )
    activity_power_policy = (
        "repeat_per_group"
        if spectrum_plan_effective is None
        else str(spectrum_plan_effective["multi_group_power_policy"])
    )
    activity_split_total_group_denominator_mode = (
        "configured_groups"
        if spectrum_plan_effective is None
        else str(spectrum_plan_effective["split_total_group_denominator_mode"])
    )
    any_dynamic_outputs = bool(
        any_power_outputs
        or output_family_plan["needs_beam_counts"]
        or output_family_plan["needs_sat_elevation"]
        or output_family_plan["needs_beam_demand"]
        or bool(store_eligible_mask)
    )
    written_output_names = _resolve_direct_epfd_output_names(
        write_epfd=bool(write_epfd),
        write_prx_total=bool(write_prx_total),
        write_per_satellite_prx_ras_station=bool(write_per_satellite_prx_ras_station),
        write_total_pfd_ras_station=bool(write_total_pfd_ras_station),
        write_per_satellite_pfd_ras_station=bool(write_per_satellite_pfd_ras_station),
        write_sat_beam_counts_used=bool(write_sat_beam_counts_used),
        write_sat_elevation_ras_station=bool(write_sat_elevation_ras_station),
        write_beam_demand_count=bool(write_beam_demand_count),
        write_sat_eligible_mask=bool(store_eligible_mask),
    )
    written_output_names.extend(
        [
            f"preaccumulated/{family_name}"
            for family_name in _OUTPUT_FAMILY_NAMES
            if _mode_includes_preaccumulated(family_configs[family_name]["mode"])
        ]
    )
    writer_stats_summary: dict[str, Any] = _empty_writer_stats_summary()
    writer_checkpoint_count = 0
    writer_checkpoint_wait_s = 0.0
    writer_final_flush_s = 0.0
    run_state = "completed"
    stop_mode = "none"
    stop_boundary: str | None = None
    stop_notice_emitted = False

    wavelength_m = float(
        u.Quantity(wavelength, copy=False).to_value(u.m)
        if hasattr(wavelength, "to_value")
        else u.Quantity(wavelength, u.m).to_value(u.m)
    )
    ras_station_ant_diam_m = float(
        u.Quantity(ras_station_ant_diam, copy=False).to_value(u.m)
        if hasattr(ras_station_ant_diam, "to_value")
        else u.Quantity(ras_station_ant_diam, u.m).to_value(u.m)
    )
    frequency_ghz = float(
        u.Quantity(frequency, copy=False).to_value(u.GHz)
        if hasattr(frequency, "to_value")
        else u.Quantity(frequency, u.GHz).to_value(u.GHz)
    )
    ras_guard_angle_rad = float(
        u.Quantity(ras_guard_angle, copy=False).to_value(u.rad)
        if hasattr(ras_guard_angle, "to_value")
        else u.Quantity(ras_guard_angle, u.deg).to_value(u.rad)
    )

    # Radio horizon: atmospheric refraction extends visibility beyond the
    # geometric horizon.  The correction combines:
    #
    # Tropospheric (ITU-R P.834): ~0.57 deg at the horizon.  Frequency-
    # independent for all radio frequencies.
    #
    # Ionospheric: ~0.0256 * (TEC/30) / f_GHz^2 deg for near-horizon paths.
    # The reference value 0.0256 deg was derived for TEC = 30 TECU (moderate
    # daytime).  The user can override TEC via radio_horizon_tec.
    # Significant below ~300 MHz, negligible above ~1 GHz.
    if bool(use_radio_horizon):
        _tropo_deg = 0.57
        _freq_ghz_sq = float(frequency_ghz) ** 2
        _tec_factor = max(0.0, float(radio_horizon_tec)) / 30.0
        _iono_deg = 0.0256 * _tec_factor / _freq_ghz_sq if _freq_ghz_sq > 0 else 0.0
        visibility_elev_threshold_deg = -(_tropo_deg + _iono_deg)
    else:
        visibility_elev_threshold_deg = 0.0
    atmosphere_max_path_length_km = float(
        u.Quantity(atmosphere_max_path_length, copy=False).to_value(u.km)
        if hasattr(atmosphere_max_path_length, "to_value")
        else u.Quantity(atmosphere_max_path_length, u.km).to_value(u.km)
    )
    timestep_s = _timestep_to_seconds(timestep)
    total_span_s = float((base_end_time - base_start_time).to_value(u.s))
    n_steps_total = int(np.ceil(total_span_s / timestep_s)) + 1
    storage_constants_effective = dict(storage_constants or {})
    storage_attrs_effective = dict(storage_attrs or {})
    for family_name, config in family_configs.items():
        storage_attrs_effective[f"output_family_{family_name}_mode"] = str(config["mode"])
    storage_attrs_effective["result_schema_version"] = int(_DIRECT_EPFD_RESULT_SCHEMA_VERSION)
    storage_attrs_effective["stored_power_basis"] = (
        "ras_receiver_band" if spectrum_plan_effective is not None else "channel_total"
    )
    storage_attrs_effective["bandwidth_mhz"] = float(power_input["bandwidth_mhz"])
    storage_attrs_effective["power_input_quantity"] = str(power_input["power_input_quantity"])
    storage_attrs_effective["power_input_basis"] = str(power_input["power_input_basis"])
    storage_attrs_effective["power_input_value"] = float(power_input["active_value"])
    storage_attrs_effective["power_input_value_unit"] = str(power_input["active_value_unit"])
    storage_attrs_effective["cell_activity_mode"] = str(normalized_cell_activity_mode)
    storage_attrs_effective["split_total_group_denominator_mode"] = str(
        normalized_split_denominator_mode
    )
    if power_input["target_pfd_dbw_m2_mhz"] is not None:
        storage_attrs_effective["pfd0_dbw_m2_mhz"] = float(power_input["target_pfd_dbw_m2_mhz"])
        storage_attrs_effective["target_pfd_dbw_m2_mhz"] = float(
            power_input["target_pfd_dbw_m2_mhz"]
        )
    if power_input["target_pfd_dbw_m2_channel"] is not None:
        storage_attrs_effective["target_pfd_dbw_m2_channel"] = float(
            power_input["target_pfd_dbw_m2_channel"]
        )
    if power_input["satellite_ptx_dbw_mhz"] is not None:
        storage_attrs_effective["satellite_ptx_dbw_mhz"] = float(
            power_input["satellite_ptx_dbw_mhz"]
        )
    if power_input["satellite_ptx_dbw_channel"] is not None:
        storage_attrs_effective["satellite_ptx_dbw_channel"] = float(
            power_input["satellite_ptx_dbw_channel"]
        )
    if power_input["satellite_eirp_dbw_mhz"] is not None:
        storage_attrs_effective["satellite_eirp_dbw_mhz"] = float(
            power_input["satellite_eirp_dbw_mhz"]
        )
    if power_input["satellite_eirp_dbw_channel"] is not None:
        storage_attrs_effective["satellite_eirp_dbw_channel"] = float(
            power_input["satellite_eirp_dbw_channel"]
        )
    if write_sat_beam_counts_used:
        storage_attrs_effective["sat_beam_counts_used_scope"] = "full_network_per_satellite"
    storage_attrs_effective["store_eligible_mask"] = int(bool(store_eligible_mask))
    # Ensure shared RAS/runtime fields are always in root attrs for recoverability
    storage_attrs_effective["observer_alt_km_ras_station"] = float(observer_alt_km_ras_station)
    storage_attrs_effective["target_alt_km"] = float(target_alt_km)
    storage_attrs_effective["use_ras_station_alt_for_co"] = int(bool(use_ras_station_alt_for_co))
    if systems is not None and len(systems) > 1:
        storage_attrs_effective["multi_system_count"] = len(systems)
        storage_attrs_effective["multi_system_interleaved"] = 1
        # Remove per-system fields from root attrs — they're in /system_N/
        for _rmk in (
            "nco", "nbeam", "active_cell_count", "selection_strategy",
            "power_input_quantity", "power_input_basis",
            "power_input_value", "power_input_value_unit",
            "pfd0_dbw_m2_mhz",
            "target_pfd_dbw_m2_mhz", "target_pfd_dbw_m2_channel",
            "satellite_eirp_dbw_mhz", "satellite_eirp_dbw_channel",
            "satellite_ptx_dbw_mhz", "satellite_ptx_dbw_channel",
            "bandwidth_mhz", "wavelength_m",
            "reuse_factor", "ras_anchor_reuse_slot",
            "unwanted_emission_mask_preset",
            "service_band_start_mhz", "service_band_stop_mhz",
            "cell_activity_mode", "cell_spacing_km",
            "channel_groups_per_cell", "multi_group_power_policy",
            "split_total_group_denominator_mode",
            "pre_ras_cell_count", "ras_guard_angle_deg",
            "enabled_channel_count", "enabled_channel_indices",
            "disabled_channel_indices", "max_groups_per_cell",
            "leftover_spectrum_mhz", "spectral_slab",
            "spectral_integration_cutoff_basis",
            "spectral_integration_cutoff_percent",
            "tx_reference_mode", "tx_reference_point_count",
            "tx_reference_frequency_mhz_effective",
            "ras_reference_frequency_mhz_effective",
            "stored_power_basis",
        ):
            storage_attrs_effective.pop(_rmk, None)
    storage_attrs_effective["wavelength_m"] = float(wavelength_m)
    storage_attrs_effective["ras_station_ant_diam_m"] = float(ras_station_ant_diam_m)
    # Store max RAS gain for optional S.1586 normalization in postprocess.
    _d_wlen = float(ras_station_ant_diam_m) / float(wavelength_m)
    storage_attrs_effective["ras_max_gain_dbi"] = float(
        10.0 * np.log10(float(np.pi * _d_wlen) ** 2)
    )
    if spectrum_plan_effective is not None:
        storage_attrs_effective.update(
            {
                "service_band_start_mhz": float(spectrum_plan_effective["service_band_start_mhz"]),
                "service_band_stop_mhz": float(spectrum_plan_effective["service_band_stop_mhz"]),
                "ras_receiver_band_start_mhz": float(
                    spectrum_plan_effective["ras_receiver_band_start_mhz"]
                ),
                "ras_receiver_band_stop_mhz": float(
                    spectrum_plan_effective["ras_receiver_band_stop_mhz"]
                ),
                "reuse_factor": int(spectrum_plan_effective["reuse_factor"]),
                "channel_groups_per_cell": int(
                    spectrum_plan_effective["channel_groups_per_cell"]
                ),
                "max_groups_per_cell": int(spectrum_plan_effective["max_groups_per_cell"]),
                "enabled_channel_count": int(spectrum_plan_effective["enabled_channel_count"]),
                "leftover_spectrum_mhz": float(spectrum_plan_effective["leftover_spectrum_mhz"]),
                "multi_group_power_policy": str(
                    spectrum_plan_effective["multi_group_power_policy"]
                ),
                "split_total_group_denominator_mode": str(
                    spectrum_plan_effective["split_total_group_denominator_mode"]
                ),
                "ras_anchor_reuse_slot": int(
                    spectrum_plan_effective["ras_anchor_reuse_slot"]
                ),
                "unwanted_emission_mask_preset": str(
                    spectrum_plan_effective["unwanted_emission_mask_preset"]
                ),
                "spectral_integration_cutoff_basis": str(
                    spectrum_plan_effective["spectral_integration_cutoff_basis"]
                ),
                "spectral_integration_cutoff_percent": float(
                    spectrum_plan_effective["spectral_integration_cutoff_percent"]
                ),
                "tx_reference_mode": str(spectrum_plan_effective["tx_reference_mode"]),
                "tx_reference_point_count": int(
                    spectrum_plan_effective["tx_reference_point_count"]
                ),
                "tx_reference_frequency_mhz_effective": float(
                    spectrum_plan_effective["tx_reference_frequency_mhz_effective"]
                ),
                "ras_reference_mode": str(spectrum_plan_effective["ras_reference_mode"]),
                "ras_reference_point_count": int(
                    spectrum_plan_effective["ras_reference_point_count"]
                ),
                "ras_reference_frequency_mhz_effective": float(
                    spectrum_plan_effective["ras_reference_frequency_mhz_effective"]
                ),
                "spectral_slab": int(spectrum_plan_effective["spectral_slab"]),
            }
        )
        storage_constants_effective.update(
            {
                "cell_reuse_slot_id_active": np.asarray(
                    spectrum_plan_effective["active_cell_reuse_slot_ids"],
                    dtype=np.int32,
                ),
                "cell_spectral_leakage_factor_active": np.asarray(
                    spectrum_plan_effective["cell_leakage_factors"],
                    dtype=np.float32,
                ),
                "cell_group_spectral_leakage_factor_active": np.asarray(
                    spectrum_plan_effective["cell_group_leakage_factors"],
                    dtype=np.float32,
                ),
                "spectrum_slot_edges_mhz": np.asarray(
                    spectrum_plan_effective["slot_edges_mhz"],
                    dtype=np.float64,
                ),
                "spectrum_slot_centers_mhz": np.asarray(
                    spectrum_plan_effective["slot_centers_mhz"],
                    dtype=np.float64,
                ),
                "spectrum_slot_group_channel_index": np.asarray(
                    spectrum_plan_effective["slot_group_channel_indices"],
                    dtype=np.int32,
                ),
                "spectrum_enabled_channel_index": np.asarray(
                    spectrum_plan_effective["enabled_channel_indices"],
                    dtype=np.int32,
                ),
                "spectrum_slot_group_leakage_factor": np.asarray(
                    spectrum_plan_effective["slot_group_leakage_factors"],
                    dtype=np.float32,
                ),
            }
        )

    _emit_prepare("Initializing HDF5 output file...")

    # For multi-system, strip per-satellite arrays from root /const/
    # BEFORE writing — they belong in /system_N/const/ only.
    if systems is not None and len(systems) > 1:
        for _psk_rm in (
            "sat_orbit_radius_m_per_sat", "sat_min_elev_deg_per_sat",
            "sat_beta_max_deg_per_sat", "sat_belt_id_per_sat",
        ):
            storage_constants_effective.pop(_psk_rm, None)

    writer_queue_max_items, writer_queue_max_bytes = _resolve_direct_epfd_writer_queue_limits(
        host_memory_budget_gb=float(host_memory_budget_gb)
    )
    init_simulation_results(
        storage_filename,
        write_mode="async",
        writer_queue_max_items=writer_queue_max_items,
        writer_queue_max_bytes=writer_queue_max_bytes,
    )
    if storage_constants_effective or storage_attrs_effective:
        write_data(
            storage_filename,
            constants=storage_constants_effective,
            attrs=storage_attrs_effective,
            compression=None,
            compression_opts=None,
        )
    print(f"HDF5 results file: {storage_filename!r} (initialized)")
    _emit_direct_epfd_progress(
        progress_callback,
        kind="run_start",
        phase="initialized",
        iteration_total=int(iteration_count),
        storage_filename=str(storage_filename),
        n_steps_total=int(n_steps_total),
        boresight_active=bool(boresight_active),
    )

    if not any_dynamic_outputs:
        print("No dynamic outputs requested; skipping GPU direct-EPFD execution loop.")
        _emit_direct_epfd_progress(
            progress_callback,
            kind="phase",
            phase="dynamic_execution_skipped",
            iteration_total=int(iteration_count),
            description="No dynamic outputs requested",
        )
        writer = _get_registered_writer(storage_filename)
        writer_flush_t0 = perf_counter() if writer is not None else None
        close_writer(storage_filename)
        if writer_flush_t0 is not None:
            writer_final_flush_s = perf_counter() - writer_flush_t0
        if writer is not None:
            writer_stats_summary = writer.stats_snapshot()
        _emit_direct_epfd_progress(
            progress_callback,
            kind="phase",
            phase="final_flush",
            iteration_total=int(iteration_count),
            description=_direct_epfd_progress_text(progress_desc_mode_name, "final_flush")
            if "progress_desc_mode_name" in locals()
            else "Final flush",
        )
        _emit_direct_epfd_progress(
            progress_callback,
            kind="run_complete",
            phase="completed",
            iteration_total=int(iteration_count),
            storage_filename=str(storage_filename),
            writer_checkpoint_count=int(writer_checkpoint_count),
            writer_checkpoint_wait_s=float(writer_checkpoint_wait_s),
            writer_final_flush_s=float(writer_final_flush_s),
            writer_stats_summary=dict(writer_stats_summary),
        )
        return {
            "storage_filename": storage_filename,
            "effective_ras_pointing_mode": effective_ras_pointing_mode,
            "beam_generation_method": beam_generation_method,
            "boresight_active": boresight_active,
            "boresight_theta1_deg": boresight_theta1_deg,
            "boresight_theta2_deg": boresight_theta2_deg,
            "debug_direct_epfd": bool(debug_direct_epfd),
            "debug_direct_epfd_stats": [],
            "profile_stage_timings": [],
            "profile_stage_timings_summary": {},
            "writer_stats_summary": writer_stats_summary,
            "writer_checkpoint_count": int(writer_checkpoint_count),
            "writer_checkpoint_wait_s": float(writer_checkpoint_wait_s),
            "writer_final_flush_s": float(writer_final_flush_s),
            "n_steps_total": n_steps_total,
            "output_families": family_configs,
            "written_output_names": written_output_names,
            "bandwidth_mhz": float(power_input["bandwidth_mhz"]),
            "power_input_quantity": str(power_input["power_input_quantity"]),
            "power_input_basis": str(power_input["power_input_basis"]),
            "dynamic_execution_skipped": True,
            "run_state": "completed",
            "stop_mode": "none",
            "stop_boundary": None,
        }

    _emit_prepare("Importing GPU modules...")

    if cupy_module is None:
        import cupy as cupy_module  # type: ignore[import-not-found]
    if gpu_module is None:
        from . import gpu_accel as gpu_module

    cp = cupy_module
    if progress_factory is None and enable_progress_bars:
        from tqdm.auto import tqdm as progress_factory

    method = gpu_method if gpu_method is not None else getattr(gpu_module, "METHOD_DWARNER", None)
    count_dtype = _direct_epfd_count_dtype(int(nbeam))
    demand_count_dtype = _beam_demand_count_dtype(n_cells_total, int(nco))
    debug_direct_epfd_stats_all: list[dict[str, Any]] = []
    profile_stage_timings_all: list[dict[str, Any]] = []
    profile_stage_timings_summary: dict[str, float] = {}
    beam_finalize_substage_timings_summary: dict[str, float] = {}
    cell_link_library_chunk_telemetry_summary: dict[str, int] = {}
    observed_stage_memory_summary_by_name: dict[str, dict[str, Any]] = {}
    beam_finalize_chunk_shape_summary: dict[str, Any] = {}
    boresight_compaction_stats_summary: dict[str, Any] = {}
    hot_path_device_to_host_copy_count_summary = 0
    hot_path_device_to_host_copy_bytes_summary = 0
    device_scalar_sync_count_summary = 0

    def _merge_run_max_numeric_mapping(
        existing: Mapping[str, Any] | None,
        incoming: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(existing or {})
        if not isinstance(incoming, Mapping):
            return merged
        for key, value in dict(incoming).items():
            key_str = str(key)
            if value is None:
                continue
            try:
                numeric_value = int(value)
            except Exception:
                merged[key_str] = value
                continue
            merged[key_str] = int(max(int(merged.get(key_str, 0) or 0), numeric_value))
        return merged

    need_pointings = bool(boresight_active or any_receive_outputs)
    progress_desc_mode_name = _resolve_progress_desc_mode(
        enable_progress_desc_updates=bool(enable_progress_desc_updates),
        progress_desc_mode=progress_desc_mode,
    )
    writer_checkpoint_interval_s_name = _resolve_writer_checkpoint_interval_s(
        writer_checkpoint_interval_s
    )
    last_writer_checkpoint_monotonic = perf_counter()

    distribution_collectors: dict[str, dict[str, Any]] = {}
    _distribution_family_names = (
        "prx_total_distribution",
        "epfd_distribution",
        "total_pfd_ras_distribution",
        "per_satellite_pfd_distribution",
    )
    for family_name in _distribution_family_names:
        family_config = family_configs[family_name]
        if not _mode_includes_preaccumulated(family_config["mode"]):
            continue
        distribution_collectors[family_name] = {
            "counts": None,
            "edges_dbw": None,
            "sample_count": 0,
            "config": family_config,
        }
    heatmap_collectors: dict[str, dict[str, Any]] = {}
    for family_name in (
        "prx_elevation_heatmap",
        "per_satellite_pfd_elevation_heatmap",
    ):
        family_config = family_configs[family_name]
        if not _mode_includes_preaccumulated(family_config["mode"]):
            continue
        elev_step = float(family_config.get("elevation_bin_step_deg", 0.2))
        elev_bin_count = max(1, int(round(90.0 / elev_step)))
        heatmap_collectors[family_name] = {
            "counts": None,
            "value_edges_dbw": None,
            "elevation_edges_deg": np.linspace(
                0.0,
                90.0,
                elev_bin_count + 1,
                dtype=np.float64,
            ),
            "sample_count": 0,
            "config": family_config,
        }

    beam_stats_collector: dict[str, Any] | None = None
    if output_family_plan["preacc_beam_statistics"]:
        beam_stats_collector = {
            "full_network_count_hist": None,
            "visible_count_hist": None,
            "network_total_beams_over_time": [],
            "visible_total_beams_over_time": [],
            "beam_demand_over_time": [],
            "count_dtype": np.int64,
            "config": family_configs["beam_statistics"],
        }

    # Apply pattern evaluation mode before session creation so the first
    # batch already uses the correct path (LUT precomputation happens lazily).
    pattern_mode = str(gpu_pattern_eval_mode or "lut").strip().lower()
    if hasattr(gpu_module, "set_pattern_eval_mode"):
        gpu_module.set_pattern_eval_mode(pattern_mode)

    # Resolve mixed-precision profile.  When a profile is given, it overrides
    # ``gpu_compute_dtype`` for the propagation stage and sets per-stage dtypes
    # on the session for pattern evaluation and power accumulation.
    precision_dtypes: dict[str, Any] | None = None
    if gpu_precision_profile is not None and hasattr(gpu_module, "resolve_precision_profile"):
        precision_dtypes = gpu_module.resolve_precision_profile(gpu_precision_profile)
        gpu_compute_dtype = precision_dtypes["propagation"]

    _emit_prepare("Creating GPU session...")

    owns_session = session is None
    if session is None:
        _session_kwargs: dict[str, Any] = {
            "compute_dtype": gpu_compute_dtype,
            "sat_frame": sat_frame,
            "watchdog_enabled": False,
        }
        if gpu_device_id is not None:
            _session_kwargs["device_id"] = int(gpu_device_id)
        session = gpu_module.GpuScepterSession(**_session_kwargs)
    if precision_dtypes is not None:
        session.pattern_dtype = np.dtype(precision_dtypes["pattern"])
        session.power_dtype = np.dtype(precision_dtypes["power"])

    _emit_prepare("Uploading satellite TLE data...")

    satellite_context = session.prepare_satellite_context(
        np.asarray(tle_list, dtype=object),
        method=method,
    )

    _emit_prepare("Uploading observer data...")

    observer_context = session.prepare_observer_context(observer_arr)
    pointing_context = (
        session.prepare_s1586_pointing_context(
            elev_range_deg=ras_station_elev_range
        )
        if need_pointings
        else None
    )
    _emit_prepare("Preparing antenna patterns and atmosphere...")

    if any_power_outputs:
        # UEMR isotropic: pattern_kwargs carries ``isotropic=True`` (and
        # ``uemr_mode=True``) instead of Rec 1.2 / 1.4 / M.2101 keys.
        # Route to the minimal isotropic pattern context — the UEMR
        # bypass in _compute_gpu_direct_epfd_batch_device never reads
        # Gm / Lt / N_H.
        _single_uemr = (
            bool(uemr_mode)
            or bool(pattern_kwargs.get("uemr_mode", False))
            or bool(pattern_kwargs.get("isotropic", False))
        )
        if _single_uemr:
            s1528_pattern_context = session.prepare_isotropic_pattern_context(
                wavelength_m=wavelength_m,
            )
        elif "Lt" in pattern_kwargs:
            # Rec 1.4 analytical (Taylor/Bessel) pattern
            s1528_pattern_context = session.prepare_s1528_pattern_context(
                wavelength_m=wavelength_m,
                lt_m=pattern_kwargs["Lt"],
                lr_m=pattern_kwargs["Lr"],
                slr_db=pattern_kwargs["SLR"],
                l=int(pattern_kwargs["l"]),
                far_sidelobe_start_deg=pattern_kwargs["far_sidelobe_start"],
                far_sidelobe_level_db=pattern_kwargs["far_sidelobe_level"],
                gm_db=pattern_kwargs.get("Gm"),
            )
        elif "N_H" in pattern_kwargs:
            # M.2101 phased array pattern.  GUI state values may be
            # Astropy Quantities (e.g. ``65 deg``); strip units via
            # ``.value`` when present so ``float()`` sees a plain scalar.
            def _scalar(v: Any, default: float = 0.0) -> float:
                return float(getattr(v, "value", v) if v is not None else default)
            s1528_pattern_context = session.prepare_m2101_pattern_context(
                g_emax_db=_scalar(pattern_kwargs.get("G_Emax", 2.0)),
                a_m_db=_scalar(pattern_kwargs.get("A_m", 30.0)),
                sla_nu_db=_scalar(pattern_kwargs.get("SLA_nu", 30.0)),
                phi_3db_deg=_scalar(pattern_kwargs.get("phi_3db", 120.0)),
                theta_3db_deg=_scalar(pattern_kwargs.get("theta_3db", 120.0)),
                d_h=_scalar(pattern_kwargs.get("d_H", 0.5)),
                d_v=_scalar(pattern_kwargs.get("d_V", 0.5)),
                n_h=int(_scalar(pattern_kwargs.get("N_H", 28))),
                n_v=int(_scalar(pattern_kwargs.get("N_V", 28))),
                wavelength_m=_scalar(wavelength_m),
            )
        else:
            # Rec 1.2 piecewise envelope pattern
            s1528_pattern_context = session.prepare_s1528_rec12_pattern_context(
                wavelength_m=wavelength_m,
                gm_dbi=pattern_kwargs["Gm"],
                ln_db=pattern_kwargs.get("LN", -15.0),
                z=float(pattern_kwargs.get("z", 1.0)),
                diameter_m=pattern_kwargs.get("D"),
            )
    else:
        s1528_pattern_context = None
    ras_pattern_context = (
        session.prepare_ras_pattern_context(
            diameter_m=ras_station_ant_diam_m,
            wavelength_m=wavelength_m,
        )
        if any_receive_outputs
        else None
    )
    atmosphere_context = (
        session.prepare_atmosphere_lut_context(
            frequency_ghz=frequency_ghz,
            altitude_km_values=np.asarray(
                [observer_alt_km_ras_station, float(target_alt_km)],
                dtype=np.float32,
            ),
            bin_deg=atmosphere_elev_bin_deg,
            elev_min_deg=atmosphere_elev_min_deg,
            elev_max_deg=atmosphere_elev_max_deg,
            max_path_length_km=atmosphere_max_path_length_km,
        )
        if include_atmosphere and any_power_outputs
        else None
    )

    # Surface-PFD cap LUT for system 0 (primary contexts).  Honours the
    # host run's atmosphere toggle automatically because we pass the
    # already-prepared atmosphere_context (or None).
    peak_pfd_lut_context = None
    if bool(max_surface_pfd_enabled) and any_power_outputs and s1528_pattern_context is not None:
        try:
            peak_pfd_lut_context = session.prepare_peak_pfd_lut_context(
                pattern_context=s1528_pattern_context,
                sat_orbit_radius_m_per_sat=np.asarray(
                    sat_orbit_radius_m_per_sat, dtype=np.float64,
                ),
                atmosphere_lut_context=atmosphere_context,
                target_alt_km=float(target_alt_km),
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Surface-PFD cap is enabled but the pattern does not yet "
                f"support it: {exc}"
            ) from exc

    _emit_prepare("Preparing spectrum plan...")

    spectrum_plan_setup_t0 = perf_counter() if profile_stages else None
    spectrum_plan_context = (
        session.prepare_spectrum_plan_context(
            reuse_factor=int(spectrum_plan_effective["reuse_factor"]),
            groups_per_cell=int(spectrum_plan_effective["channel_groups_per_cell"]),
            channel_bandwidth_mhz=float(spectrum_plan_effective["channel_bandwidth_mhz"]),
            service_band_start_mhz=float(spectrum_plan_effective["service_band_start_mhz"]),
            service_band_stop_mhz=float(spectrum_plan_effective["service_band_stop_mhz"]),
            ras_band_start_mhz=float(spectrum_plan_effective["ras_receiver_band_start_mhz"]),
            ras_band_stop_mhz=float(spectrum_plan_effective["ras_receiver_band_stop_mhz"]),
            power_policy=str(spectrum_plan_effective["multi_group_power_policy"]),
            cell_reuse_slot_ids=np.asarray(
                spectrum_plan_effective["active_cell_reuse_slot_ids"],
                dtype=np.int32,
            ),
            cell_leakage_factors=np.asarray(
                spectrum_plan_effective["cell_leakage_factors"],
                dtype=np.float32,
            ),
            cell_group_leakage_factors=np.asarray(
                spectrum_plan_effective["cell_group_leakage_factors"],
                dtype=np.float32,
            ),
            cell_group_valid_mask=np.asarray(
                spectrum_plan_effective["cell_group_valid_mask"],
                dtype=bool,
            ),
            configured_group_counts_per_cell=np.asarray(
                spectrum_plan_effective["configured_group_counts_per_cell"],
                dtype=np.int32,
            ),
            slot_edges_mhz=np.asarray(spectrum_plan_effective["slot_edges_mhz"], dtype=np.float64),
            mask_points_mhz=np.asarray(
                spectrum_plan_effective["unwanted_emission_mask_points_mhz"],
                dtype=np.float64,
            ),
            tx_reference_frequencies_mhz=np.asarray(
                spectrum_plan_effective["tx_reference_frequencies_mhz"],
                dtype=np.float64,
            ),
            ras_reference_frequencies_mhz=np.asarray(
                spectrum_plan_effective["ras_reference_frequencies_mhz"],
                dtype=np.float64,
            ),
        )
        if spectrum_plan_effective is not None and any_power_outputs
        else None
    )
    if profile_stages and spectrum_plan_setup_t0 is not None:
        _sync_array_module(cp)
        profile_stage_timings_summary["spectrum_context_setup"] = float(
            perf_counter() - spectrum_plan_setup_t0
        )

    # ---- Multi-system context preparation ----
    # When ``systems`` has N > 1 entries, prepare additional per-system GPU
    # contexts.  System 0 always uses the primary contexts created above.
    _multi_system_active = bool(systems is not None and len(systems) > 1)
    _multi_system_contexts: list[dict[str, Any]] = []
    if _multi_system_active:
        assert systems is not None  # type narrowing
        # System 0 uses the primary contexts already prepared above.
        _multi_system_contexts.append({
            "satellite_context": satellite_context,
            "s1528_pattern_context": s1528_pattern_context,
            "ras_pattern_context": ras_pattern_context,
            "spectrum_plan_context": spectrum_plan_context,
            "spectrum_plan_effective": spectrum_plan_effective,
            "power_input": power_input,
            "peak_pfd_lut_context": peak_pfd_lut_context,
            "surface_pfd_cap_enabled": bool(max_surface_pfd_enabled),
            "surface_pfd_cap_mode": str(surface_pfd_cap_mode),
            "surface_pfd_stats_enabled": bool(surface_pfd_stats_enabled),
            "max_surface_pfd_dbw_m2_mhz": max_surface_pfd_dbw_m2_mhz,
            "max_surface_pfd_dbw_m2_channel": max_surface_pfd_dbw_m2_channel,
            "sat_min_elev_deg_per_sat_f64": np.asarray(
                sat_min_elevation_deg_per_sat, dtype=np.float64,
            ),
            "sat_beta_max_deg_per_sat_f32": np.asarray(
                sat_beta_max_deg_per_sat, dtype=np.float32,
            ),
            "sat_belt_id_per_sat_i16": np.asarray(sat_belt_id_per_sat, dtype=np.int16),
            "orbit_radius_host": np.asarray(sat_orbit_radius_m_per_sat, dtype=np.float32),
            "n_sats_total": n_sats_total,
            "nco": int(nco),
            "nbeam": int(nbeam),
            "selection_mode": selection_mode,
            "wavelength_m": wavelength_m,
            "frequency_ghz": frequency_ghz,
            "cell_activity_factor": float(cell_activity_factor),
            "cell_activity_mode": str(normalized_cell_activity_mode),
            "cell_activity_seed_base": cell_activity_seed_base,
            "n_cells_total": n_cells_total,
            "observer_arr": observer_arr,
            "ras_service_cell_index": int(ras_service_cell_index),
            "ras_service_cell_active": bool(ras_service_cell_active),
            "system_name": str(systems[0].get("_system_name", "System 1") if systems else "System 1"),
            "storage_attrs": systems[0].get("storage_attrs", {}) if systems else {},
            # Per-system boresight: system 0 uses the primary boresight params
            "boresight_active": bool(boresight_active),
            "boresight_theta1_deg": boresight_theta1_deg,
            "boresight_theta2_deg": boresight_theta2_deg,
            "boresight_theta2_cell_ids": boresight_theta2_cell_ids,
            "uemr_mode": bool(uemr_mode),
            # antenna_model identifier for HDF5 per-system attrs. Mirrors
            # the dispatch in _prepare_multi_system_extra_context so the
            # PRIMARY system (index 0) carries the same metadata as the
            # secondaries — otherwise system_0 reads with empty model in
            # the merged HDF5 output.
            "antenna_model": (
                "isotropic" if bool(uemr_mode) or bool((pattern_kwargs or {}).get("isotropic", False))
                else (
                    "s1528_rec1_4" if "Lt" in (pattern_kwargs or {})
                    else ("m2101" if "N_H" in (pattern_kwargs or {})
                          else "s1528_rec1_2")
                )
            ),
        })
        for sys_idx in range(1, len(systems)):
            sys_kw = systems[sys_idx]
            sys_ctx = _prepare_multi_system_extra_context(
                session=session,
                sys_kw=sys_kw,
                method=method,
                any_power_outputs=any_power_outputs,
                any_receive_outputs=any_receive_outputs,
                gpu_compute_dtype=gpu_compute_dtype,
                profile_stages=profile_stages,
                cp_module=cp,
                observer_alt_km_ras_station=observer_alt_km_ras_station,
                target_alt_km=target_alt_km,
                include_atmosphere=include_atmosphere,
                atmosphere_elev_bin_deg=atmosphere_elev_bin_deg,
                atmosphere_elev_min_deg=atmosphere_elev_min_deg,
                atmosphere_elev_max_deg=atmosphere_elev_max_deg,
                atmosphere_max_path_length_km=atmosphere_max_path_length_km,
                ras_station_elev_range=ras_station_elev_range,
                normalized_cell_activity_mode=str(normalized_cell_activity_mode),
                normalized_split_denominator_mode=str(normalized_split_denominator_mode),
                n_cells_total=n_cells_total,
                sys_idx=sys_idx,
            )
            _multi_system_contexts.append(sys_ctx)
        # If any secondary system has boresight, we need pointings even if
        # system 0 does not.  ``need_pointings`` was set earlier from system 0;
        # widen it to the union of all systems.
        _any_system_boresight = any(
            bool(ctx.get("boresight_active", False))
            for ctx in _multi_system_contexts
        )
        if _any_system_boresight and not need_pointings:
            need_pointings = True
            # pointing_context was not prepared earlier because system 0
            # had no boresight — prepare it now for secondary systems.
            if pointing_context is None:
                pointing_context = session.prepare_s1586_pointing_context(
                    elev_range_deg=ras_station_elev_range
                )
        print(
            f"Multi-system interleaving active: {len(_multi_system_contexts)} systems "
            f"will be processed per batch."
        )
    # --- Per-group output collectors (replaces per-system collectors) ---
    # Build group definitions: either from the caller or default per-system + combined.
    _n_multi_systems = len(_multi_system_contexts) if _multi_system_active else 0
    if output_system_groups and _multi_system_active:
        _output_groups = [dict(g) for g in output_system_groups]
    elif _multi_system_active:
        _output_groups = _default_output_groups(_n_multi_systems)
    else:
        _output_groups = []

    # Determine which system indices are actually needed.
    _active_system_indices: set[int] = set()
    for _og in _output_groups:
        if _og.get("enabled", True):
            _active_system_indices.update(_og.get("system_indices", []))

    # Build collectors for each enabled group.
    _group_collectors: list[dict[str, Any]] = []
    for _og in _output_groups:
        if not _og.get("enabled", True):
            continue
        _gc_entry: dict[str, Any] = {
            "name": str(_og.get("name", "")),
            "system_indices": set(int(i) for i in _og.get("system_indices", [])),
            "group_def": _og,
        }
        # Distribution collectors
        _gc_dist: dict[str, dict[str, Any]] = {}
        for family_name in _distribution_family_names:
            family_config = family_configs[family_name]
            if not _mode_includes_preaccumulated(family_config["mode"]):
                continue
            _gc_dist[family_name] = {
                "counts": None,
                "edges_dbw": None,
                "sample_count": 0,
                "config": family_config,
            }
        _gc_entry["distribution_collectors"] = _gc_dist
        # Heatmap collectors
        _gc_hm: dict[str, dict[str, Any]] = {}
        for family_name in (
            "prx_elevation_heatmap",
            "per_satellite_pfd_elevation_heatmap",
        ):
            family_config = family_configs[family_name]
            if not _mode_includes_preaccumulated(family_config["mode"]):
                continue
            elev_step = float(family_config.get("elevation_bin_step_deg", 0.2))
            elev_bin_count = max(1, int(round(90.0 / elev_step)))
            _gc_hm[family_name] = {
                "counts": None,
                "value_edges_dbw": None,
                "elevation_edges_deg": np.linspace(
                    0.0, 90.0, elev_bin_count + 1, dtype=np.float64,
                ),
                "sample_count": 0,
                "config": family_config,
            }
        _gc_entry["heatmap_collectors"] = _gc_hm
        # Beam stats collector
        if output_family_plan["preacc_beam_statistics"]:
            _gc_entry["beam_stats_collector"] = {
                "full_network_count_hist": None,
                "visible_count_hist": None,
                "network_total_beams_over_time": [],
                "visible_total_beams_over_time": [],
                "beam_demand_over_time": [],
                "count_dtype": np.int64,
                "config": family_configs["beam_statistics"],
            }
        else:
            _gc_entry["beam_stats_collector"] = None
        _group_collectors.append(_gc_entry)

    # Backward-compat aliases — legacy code references these names
    _per_system_distribution_collectors: list[dict[str, dict[str, Any]]] = []
    _per_system_heatmap_collectors: list[dict[str, dict[str, Any]]] = []
    _per_system_beam_stats_collectors: list[dict[str, Any] | None] = []
    if _multi_system_active and _group_collectors:
        # Build per-system views from single-system groups for the in-loop
        # per-system accumulation (heatmaps + beam stats are per-system only).
        for _sys_i in range(_n_multi_systems):
            # Find the single-system group that matches this index
            _found_gc = None
            for _gc in _group_collectors:
                if _gc["system_indices"] == {_sys_i}:
                    _found_gc = _gc
                    break
            if _found_gc is not None:
                _per_system_distribution_collectors.append(_found_gc["distribution_collectors"])
                _per_system_heatmap_collectors.append(_found_gc["heatmap_collectors"])
                _per_system_beam_stats_collectors.append(_found_gc["beam_stats_collector"])
            else:
                _per_system_distribution_collectors.append({})
                _per_system_heatmap_collectors.append({})
                _per_system_beam_stats_collectors.append(None)

    # Per-satellite arrays were already stripped from storage_constants_effective
    # before the initial write_data call above.

    _ms_batch_stash: list[dict[str, Any]] = []  # per-system CPU copies for raw HDF5

    n_skycells_s1586 = int(pointing_context.n_cells) if pointing_context is not None else 0
    n_active_beams_nominal = int(n_cells_total * int(nco))
    print(
        "GPU session prepared: "
        f"n_observers={observer_context.n_observers}, "
        f"n_satellites={satellite_context.n_sats}, "
        f"n_earthgrid_cells={n_cells_total}, "
        f"n_skycells_s1586={n_skycells_s1586}, "
        f"nominal_active_beams={n_active_beams_nominal}"
    )
    print(
        "EarthGrid cells define transmit beams. S.1586 SkyCells define "
        "receive-side RAS telescope pointings."
    )
    _emit_direct_epfd_progress(
        progress_callback,
        kind="phase",
        phase="session_prepared",
        iteration_total=int(iteration_count),
        description="GPU session prepared",
        n_observers=int(observer_context.n_observers),
        n_satellites=int(satellite_context.n_sats),
        n_earthgrid_cells=int(n_cells_total),
        n_skycells_s1586=int(n_skycells_s1586),
    )

    sat_min_elev_deg_per_sat_f64 = np.asarray(
        sat_min_elevation_deg_per_sat,
        dtype=np.float64,
    )
    sat_beta_max_deg_per_sat_f32 = np.asarray(
        sat_beta_max_deg_per_sat,
        dtype=np.float32,
    )
    sat_belt_id_per_sat_i16 = np.asarray(sat_belt_id_per_sat, dtype=np.int16)
    orbit_radius_host = np.asarray(sat_orbit_radius_m_per_sat, dtype=np.float32)
    rng = np.random.default_rng(int(iteration_rng_seed))
    host_budget_mode_name, gpu_budget_mode_name = _normalise_runner_memory_budget_modes(
        memory_budget_mode
    )
    scheduler_target_name, scheduler_target_fraction = _resolve_scheduler_target_fraction(
        scheduler_target_profile
    )
    run_active_target_fraction = float(scheduler_target_fraction)
    spectrum_context_bytes = (
        0 if spectrum_plan_effective is None else int(spectrum_plan_effective["spectrum_context_bytes"])
    )
    scheduler_runtime_state: dict[str, Any] = {
        "gpu_effective_budget_bytes": None,
        "gpu_effective_budget_lowered": False,
        "gpu_effective_budget_previous_bytes": None,
        "gpu_budget_lowered_stage": None,
        "scheduler_retry_count": 0,
        "last_observed_stage_summary": {},
        "spectral_slab": (
            1 if spectrum_plan_effective is None else int(spectrum_plan_effective["spectral_slab"])
        ),
        "reuse_factor": (
            1 if spectrum_plan_effective is None else int(spectrum_plan_effective["reuse_factor"])
        ),
        "groups_per_cell": (
            1
            if spectrum_plan_effective is None
            else int(spectrum_plan_effective["channel_groups_per_cell"])
        ),
    }
    pbar_ext = (
        progress_factory(
            range(int(iteration_count)),
            desc=(
                None
                if progress_desc_mode_name == "off"
                else "Iterations"
            ),
        )
        if enable_progress_bars and progress_factory is not None
        else range(int(iteration_count))
    )

    def _per_iteration_gpu_cleanup() -> None:
        """
        Best-effort end-of-iteration GPU cleanup.

        Important:
        - This is intentionally called only after the full iteration has
          finished, including all per-batch preaccumulated histogram / heatmap
          updates and write enqueue work.
        - It must not reset or recreate host-side preaccumulated collectors.
        - It must not close the session.
        """
        try:
            _sync_array_module(cp)
            session.evict_idle_caches()

            if hasattr(cp, "get_default_memory_pool"):
                cp.get_default_memory_pool().free_all_blocks()

            if hasattr(cp, "get_default_pinned_memory_pool"):
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            # Cleanup is best-effort only. Never let allocator/cache cleanup
            # corrupt or abort otherwise successful accumulated results.
            pass

        gc.collect()

    _emit_prepare("Starting simulation loop...")

    try:
        with session.activate():
            orbit_radius_full = cp.asarray(orbit_radius_host)

            for ii in pbar_ext:
                cancel_mode = _query_direct_epfd_cancel_mode(cancel_callback)
                if cancel_mode in {"graceful", "force"}:
                    run_state = "stopped"
                    stop_mode = cancel_mode
                    stop_boundary = "iteration_boundary"
                    if not stop_notice_emitted:
                        _emit_direct_epfd_progress(
                            progress_callback,
                            kind="phase",
                            phase="stopping",
                            iteration_index=int(ii),
                            iteration_total=int(iteration_count),
                            stop_mode=str(stop_mode),
                            stop_boundary=str(stop_boundary),
                            description=(
                                "Graceful stop requested. Finishing at the next safe boundary."
                                if stop_mode == "graceful"
                                else "Force stop requested. Aborting at the next safe boundary."
                            ),
                        )
                        stop_notice_emitted = True
                    break
                _emit_direct_epfd_progress(
                    progress_callback,
                    kind="iteration_start",
                    phase="iteration_start",
                    iteration_index=int(ii),
                    iteration_total=int(iteration_count),
                    description=f"Iteration {int(ii) + 1}/{int(iteration_count)}",
                )
                jitter_sec = float(rng.uniform(-36.0 * 3600.0, 36.0 * 3600.0))
                time_shift = TimeDelta(jitter_sec, format="sec")
                start_time = base_start_time + time_shift
                end_time = base_end_time + time_shift

                host_budget_info = resolve_host_memory_budget_bytes(
                    host_memory_budget_gb,
                    mode=host_budget_mode_name,
                    headroom_profile=memory_headroom_profile,
                )
                probe_mjds = _deterministic_visibility_probe_mjds(
                    start_time,
                    end_time,
                    sample_count=_DIRECT_EPFD_VISIBILITY_PROBE_SAMPLES,
                )
                probe_profile = _probe_visibility_profile_window(
                    session,
                    probe_mjds,
                    satellite_context,
                    observer_context=observer_context,
                    observer_slice=slice(0, 1),
                    output_dtype=gpu_output_dtype,
                )
                visible_satellite_probe = int(probe_profile["visible_satellite_count"])
                # Apply the headroom profile factor plus a fixed 5% safety
                # margin for edge-case batches where the actual visible count
                # slightly exceeds the probe-time estimate (orbital geometry
                # shifts between the probe window and the run window can add
                # a handful of extra visible satellites).
                _vis_factor = float(host_budget_info["visible_satellite_factor"])
                _vis_with_margin = np.ceil(
                    visible_satellite_probe * _vis_factor * 1.05
                )
                visible_satellite_est = int(
                    min(n_sats_total, max(1, int(_vis_with_margin)))
                )
                gpu_budget_info_raw = session.resolve_device_memory_budget_bytes(
                    gpu_memory_budget_gb,
                    mode=gpu_budget_mode_name,
                    headroom_profile=memory_headroom_profile,
                )
                if scheduler_runtime_state.get("gpu_effective_budget_bytes") is None:
                    scheduler_runtime_state["gpu_effective_budget_bytes"] = int(
                        gpu_budget_info_raw["effective_budget_bytes"]
                    )
                gpu_budget_info = _apply_effective_budget_override(
                    gpu_budget_info_raw,
                    effective_budget_bytes=(
                        None
                        if scheduler_runtime_state.get("gpu_effective_budget_bytes") is None
                        else int(scheduler_runtime_state["gpu_effective_budget_bytes"])
                    ),
                )
                # For multi-system, plan for worst-case satellite/cell counts
                _sched_n_sats = n_sats_total
                _sched_n_cells = n_cells_total
                _sched_ms_total = 1
                # Single-system UEMR: the primary system itself bypasses
                # the beam library, so the scheduler's per-system cost
                # model must know — otherwise it estimates batch time as
                # if a full directive run is happening, and the ETA is
                # 5× too long.
                _sched_ms_uemr = 1 if bool(uemr_mode) else 0
                if _multi_system_active:
                    _sched_n_sats = max(
                        n_sats_total,
                        *(ctx["n_sats_total"] for ctx in _multi_system_contexts),
                    )
                    _sched_n_cells = max(
                        n_cells_total,
                        *(ctx["n_cells_total"] for ctx in _multi_system_contexts),
                    )
                    # Count primary system + per-batch interleaved systems for
                    # the planner's multi-system cost-model adjustment. UEMR
                    # systems bypass the beam library and are ~1/3 as expensive
                    # as directive systems per the empirical scale factor.
                    _sched_ms_total = 1 + len(_multi_system_contexts)
                    _sched_ms_uemr = (
                        (1 if bool(uemr_mode) else 0)  # primary is UEMR too
                        + sum(
                            1 for _c in _multi_system_contexts
                            if bool(_c.get("uemr_mode", False))
                        )
                    )
                iteration_plan = _plan_direct_epfd_iteration_schedule(
                    session=session,
                    observer_context=observer_context,
                    satellite_context=satellite_context,
                    gpu_output_dtype=gpu_output_dtype,
                    n_steps_total=n_steps_total,
                    n_cells_total=_sched_n_cells,
                    n_sats_total=_sched_n_sats,
                    n_skycells_s1586=n_skycells_s1586,
                    visible_satellite_est=visible_satellite_est,
                    nco=int(nco),
                    nbeam=int(nbeam),
                    boresight_active=boresight_active,
                    effective_ras_pointing_mode=effective_ras_pointing_mode,
                    output_family_plan=output_family_plan,
                    store_eligible_mask=bool(store_eligible_mask),
                    profile_stages=bool(profile_stages),
                    host_budget_info=host_budget_info,
                    gpu_budget_info=gpu_budget_info,
                    scheduler_target_fraction=float(scheduler_target_fraction),
                    scheduler_active_target_fraction=float(run_active_target_fraction),
                    finalize_memory_budget_bytes=finalize_memory_budget_bytes,
                    power_memory_budget_bytes=power_memory_budget_bytes,
                    export_memory_budget_bytes=export_memory_budget_bytes,
                    power_sky_slab=power_sky_slab,
                    force_bulk_timesteps=force_bulk_timesteps,
                    force_cell_observer_chunk=force_cell_observer_chunk,
                    spectrum_context_bytes=int(spectrum_context_bytes),
                    cell_activity_mode=str(normalized_cell_activity_mode),
                    activity_groups_per_cell=int(activity_groups_per_cell),
                    activity_power_policy=str(activity_power_policy),
                    activity_split_total_group_denominator_mode=str(
                        activity_split_total_group_denominator_mode
                    ),
                    allow_warmup_calibration=True,
                    surface_pfd_cap_enabled=bool(max_surface_pfd_enabled),
                    surface_pfd_cap_mode=str(surface_pfd_cap_mode),
                    multi_system_count=int(_sched_ms_total),
                    multi_system_uemr_count=int(_sched_ms_uemr),
                )
                scheduler_runtime_state = _update_scheduler_runtime_state_from_plan(
                    scheduler_runtime_state,
                    iteration_plan,
                    spectrum_context_bytes=int(spectrum_context_bytes),
                )
                iter_cell_chunk_plan = int(iteration_plan["cell_chunk"])
                bulk_timesteps = int(iteration_plan["bulk_timesteps"])
                # Finalize per-system GPU constants now that cell_chunk is known
                if _multi_system_active:
                    for _ms_ctx_fin in _multi_system_contexts:
                        _ms_nc = _ms_ctx_fin["n_cells_total"]
                        _ms_cc = min(iter_cell_chunk_plan, _ms_nc)
                        _ms_ctx_fin["cell_chunk"] = _ms_cc
                        _ms_ctx_fin["n_cell_chunks"] = max(1, -(-_ms_nc // max(1, _ms_cc)))
                        # Upload orbit radius to GPU once (not per-batch)
                        if "orbit_radius_dev" not in _ms_ctx_fin:
                            _ms_ctx_fin["orbit_radius_dev"] = cp.asarray(
                                _ms_ctx_fin["orbit_radius_host"]
                            )
                stage_budget_info = dict(iteration_plan["stage_budget_info"])
                planned_power_sky_slab = int(iteration_plan["sky_slab"])
                limiting_resource = str(iteration_plan["limiting_resource"])
                planned_batch_seconds = float(iteration_plan["planned_batch_seconds"])
                # NOTE: multi-system scale factor is now applied inside the
                # planner via _estimate_direct_epfd_batch_seconds (see the
                # multi_system_count / multi_system_uemr_count kwargs threaded
                # in above). The post-hoc scale here would double-count.
                n_batches = int(np.ceil(n_steps_total / bulk_timesteps))
                planned_iteration_seconds = float(planned_batch_seconds) * float(n_batches)
                remaining_iterations = max(0, int(iteration_count) - int(ii))
                planned_total_seconds = float(planned_iteration_seconds) * float(int(iteration_count))
                planned_remaining_seconds = float(planned_iteration_seconds) * float(remaining_iterations)
                live_host_snapshot = _runtime_host_memory_snapshot()
                live_gpu_snapshot = _runtime_gpu_memory_snapshot(cp, session)
                live_gpu_adapter_snapshot = _runtime_gpu_adapter_memory_snapshot(cp)
                scheduler_payload = _build_direct_epfd_scheduler_payload(
                    host_budget_info=host_budget_info,
                    gpu_budget_info=gpu_budget_info,
                    scheduler_target_fraction=float(scheduler_target_fraction),
                    scheduler_active_target_fraction=float(run_active_target_fraction),
                    boresight_active=bool(boresight_active),
                    n_earthgrid_cells=int(n_cells_total),
                    n_skycells_s1586=int(n_skycells_s1586),
                    visible_satellite_est=int(visible_satellite_est),
                    bulk_timesteps=int(bulk_timesteps),
                    cell_chunk=int(iter_cell_chunk_plan),
                    sky_slab=int(planned_power_sky_slab),
                    predicted_host_peak_bytes=int(iteration_plan["predicted_host_peak_bytes"]),
                    predicted_gpu_peak_bytes=int(iteration_plan["predicted_gpu_peak_bytes"]),
                    planner_source=str(iteration_plan["planner_source"]),
                    limiting_resource=str(limiting_resource),
                    planned_total_seconds=float(planned_total_seconds),
                    planned_remaining_seconds=float(planned_remaining_seconds),
                    live_host_snapshot=live_host_snapshot,
                    live_gpu_snapshot=live_gpu_snapshot,
                    live_gpu_adapter_snapshot=live_gpu_adapter_snapshot,
                    extra={
                        "visible_satellite_probe": int(visible_satellite_probe),
                        "visible_satellite_probe_samples": int(
                            probe_profile.get("probe_sample_count", 1)
                        ),
                        "candidate_count": int(iteration_plan["candidate_count"]),
                        "scheduler_target_profile": str(scheduler_target_name),
                        "spectrum_plan_active": bool(spectrum_plan_effective is not None),
                        "reuse_factor": (
                            1 if spectrum_plan_effective is None else int(spectrum_plan_effective["reuse_factor"])
                        ),
                        "groups_per_cell": (
                            1
                            if spectrum_plan_effective is None
                            else int(spectrum_plan_effective["channel_groups_per_cell"])
                        ),
                        "cell_activity_mode": str(normalized_cell_activity_mode),
                        "split_total_group_denominator_mode": str(
                            normalized_split_denominator_mode
                        ),
                        **_scheduler_runtime_state_extra(scheduler_runtime_state),
                    },
                )

                print(
                    f"[Direct EPFD plan] iter={ii} "
                    f"host_budget={format_byte_count(host_budget_info['effective_budget_bytes'])} "
                    f"gpu_budget={format_byte_count(gpu_budget_info['effective_budget_bytes'])} "
                    f"target={float(scheduler_target_fraction) * 100.0:.0f}% "
                    f"active={float(run_active_target_fraction) * 100.0:.0f}% "
                    f"visible_probe={visible_satellite_probe}/{n_sats_total} "
                    f"({probe_profile['visible_fraction']:.1%}) "
                    f"visible_est={visible_satellite_est} "
                    f"limit={limiting_resource} "
                    f"planner={iteration_plan['planner_source']} "
                    f"finalize_budget={format_byte_count(stage_budget_info['finalize_memory_budget_bytes'])} "
                    f"power_budget={format_byte_count(stage_budget_info['power_memory_budget_bytes'])} "
                    f"export_budget={format_byte_count(stage_budget_info['export_memory_budget_bytes'])} "
                    f"cell_chunk={iter_cell_chunk_plan} "
                    f"bulk_timesteps={bulk_timesteps} "
                    f"sky_slab={planned_power_sky_slab} "
                    f"spectral_slab={int(iteration_plan.get('spectral_slab', 1))} "
                    f"batches={n_batches}"
                )
                _emit_direct_epfd_progress(
                    progress_callback,
                    kind="iteration_plan",
                    phase="chunks",
                    iteration_index=int(ii),
                    iteration_total=int(iteration_count),
                    batch_total=int(n_batches),
                    description=_direct_epfd_progress_text(progress_desc_mode_name, "chunks")
                    or "Processing time slices",
                    **scheduler_payload,
                )

                time_batches_iter = iter_simulation_batches(
                    start_time,
                    end_time,
                    timestep_s,
                    bulk_timesteps,
                )
                pbar = (
                    progress_factory(
                        time_batches_iter,
                        total=n_batches,
                        desc=_direct_epfd_progress_text(
                            progress_desc_mode_name,
                            "chunks",
                        ),
                        leave=False,
                    )
                    if enable_progress_bars and progress_factory is not None
                    else time_batches_iter
                )

                for bi, time_batch in enumerate(pbar):
                    cancel_mode = _query_direct_epfd_cancel_mode(cancel_callback)
                    if cancel_mode in {"graceful", "force"}:
                        run_state = "stopped"
                        stop_mode = cancel_mode
                        stop_boundary = "batch_boundary"
                        if not stop_notice_emitted:
                            _emit_direct_epfd_progress(
                                progress_callback,
                                kind="phase",
                                phase="stopping",
                                iteration_index=int(ii),
                                iteration_total=int(iteration_count),
                                batch_index=int(bi),
                                batch_total=int(n_batches),
                                stop_mode=str(stop_mode),
                                stop_boundary=str(stop_boundary),
                                description=(
                                    "Graceful stop requested. Waiting for the current safe boundary."
                                    if stop_mode == "graceful"
                                    else "Force stop requested. Stopping as soon as a cancellation poll is reached."
                                ),
                            )
                            stop_notice_emitted = True
                        break
                    stage_timings = {} if profile_stages else None
                    stage_start = perf_counter() if profile_stages else None
                    local_times = time_batch["times"]
                    mjd = local_times.mjd.astype(np.float64, copy=False)
                    time_count_local = int(mjd.size)
                    current_batch_plan = dict(iteration_plan)
                    cell_chunk = max(1, min(int(current_batch_plan["cell_chunk"]), n_cells_total))
                    n_cell_chunks = int(np.ceil(n_cells_total / cell_chunk))
                    remaining_iterations_after = max(0, int(iteration_count) - int(ii) - 1)
                    batch_planned_remaining_seconds = (
                        float(planned_batch_seconds) * float(max(1, n_batches - int(bi)))
                        + float(planned_iteration_seconds) * float(remaining_iterations_after)
                    )
                    batch_scheduler_payload = _build_direct_epfd_scheduler_payload(
                        host_budget_info=host_budget_info,
                        gpu_budget_info=gpu_budget_info,
                        scheduler_target_fraction=float(scheduler_target_fraction),
                        scheduler_active_target_fraction=float(run_active_target_fraction),
                        boresight_active=bool(boresight_active),
                        n_earthgrid_cells=int(n_cells_total),
                        n_skycells_s1586=int(n_skycells_s1586),
                        visible_satellite_est=int(visible_satellite_est),
                        bulk_timesteps=int(bulk_timesteps),
                        cell_chunk=int(cell_chunk),
                        sky_slab=int(current_batch_plan["sky_slab"]),
                        predicted_host_peak_bytes=int(
                            current_batch_plan["predicted_host_peak_bytes"]
                        ),
                        predicted_gpu_peak_bytes=int(
                            current_batch_plan["predicted_gpu_peak_bytes"]
                        ),
                        planner_source=str(current_batch_plan["planner_source"]),
                        limiting_resource=str(current_batch_plan["limiting_resource"]),
                        planned_total_seconds=float(planned_total_seconds),
                        planned_remaining_seconds=float(batch_planned_remaining_seconds),
                        live_host_snapshot=_runtime_host_memory_snapshot(),
                        live_gpu_snapshot=_runtime_gpu_memory_snapshot(cp, session),
                        live_gpu_adapter_snapshot=_runtime_gpu_adapter_memory_snapshot(cp),
                        extra={
                            "visible_satellite_probe": int(visible_satellite_probe),
                            "visible_satellite_probe_samples": int(
                                probe_profile.get("probe_sample_count", 1)
                            ),
                            "candidate_count": int(current_batch_plan["candidate_count"]),
                            "scheduler_target_profile": str(scheduler_target_name),
                            "spectrum_plan_active": bool(spectrum_plan_effective is not None),
                            "reuse_factor": (
                                1 if spectrum_plan_effective is None else int(spectrum_plan_effective["reuse_factor"])
                            ),
                            "groups_per_cell": (
                                1
                                if spectrum_plan_effective is None
                                else int(spectrum_plan_effective["channel_groups_per_cell"])
                            ),
                            "cell_activity_mode": str(normalized_cell_activity_mode),
                            "split_total_group_denominator_mode": str(
                                normalized_split_denominator_mode
                            ),
                            **_scheduler_runtime_state_extra(scheduler_runtime_state),
                        },
                    )
                    _batch_loop_t0 = perf_counter()
                    _emit_direct_epfd_progress(
                        progress_callback,
                        kind="batch_start",
                        phase="chunks",
                        iteration_index=int(ii),
                        iteration_total=int(iteration_count),
                        batch_index=int(bi),
                        batch_total=int(n_batches),
                        description=_direct_epfd_progress_text(progress_desc_mode_name, "chunks")
                        or "Processing time slices",
                        **batch_scheduler_payload,
                    )

                    orbit_state = session.propagate_orbit(
                        mjd,
                        satellite_context,
                        on_error=gpu_on_error,
                    )
                    stage_start = record_profile_stage(
                        stage_timings,
                        "orbit_propagation",
                        stage_start,
                        enabled=profile_stages,
                        synchronize=lambda: _sync_array_module(cp),
                    )

                    ras_result = session.derive_from_eci(
                        orbit_state,
                        observer_context=observer_context,
                        observer_slice=slice(0, 1),
                        do_eci_pos=False,
                        do_eci_vel=False,
                        do_geo=False,
                        do_topo=True,
                        do_obs_pos=False,
                        do_sat_azel=True,
                        do_sat_rotmat=False,
                        output_dtype=gpu_output_dtype,
                        return_device=True,
                    )
                    sat_topo_ras_station = cp.asarray(ras_result["topo"])[:, 0, :, :]
                    sat_azel_ras_station = cp.asarray(ras_result["sat_azel"])[:, 0, :, :]
                    sat_keep_batch = cp.any(
                        sat_topo_ras_station[..., 1] > cp.float32(visibility_elev_threshold_deg),
                        axis=0,
                    )
                    stage_start = record_profile_stage(
                        stage_timings,
                        "ras_geometry",
                        stage_start,
                        enabled=profile_stages,
                        synchronize=lambda: _sync_array_module(cp),
                    )

                    pointings = None
                    if need_pointings:
                        if pointing_context is None:
                            raise RuntimeError("Pointing context is required but was not prepared.")
                        pointings = session.sample_s1586_pointings(
                            pointing_context,
                            n_samples=time_count_local,
                            seed=int((26001 + ii * 1_000_000 + bi * 10_000) % (2**32 - 1)),
                            return_device=True,
                        )
                        stage_start = record_profile_stage(
                            stage_timings,
                            "pointings",
                            stage_start,
                            enabled=profile_stages,
                            synchronize=lambda: _sync_array_module(cp),
                        )

                    batch_has_visible_sats, visible_count_observed = (
                        _observe_visible_satellite_batch_state(
                            cp,
                            sat_keep_batch=sat_keep_batch,
                            visible_satellite_est=int(visible_satellite_est),
                            n_sats_total=int(n_sats_total),
                            need_exact_count=bool(profile_stages or debug_direct_epfd),
                            stage_timings=stage_timings if profile_stages else None,
                        )
                    )
                    visible_count = int(
                        visible_satellite_est
                        if visible_count_observed is None
                        else visible_count_observed
                    )
                    cell_activity_seed = (
                        None
                        if cell_activity_seed_base is None
                        else int(
                            (int(cell_activity_seed_base) + ii * 1_000_000 + bi * 10_000)
                            % (2**32 - 1)
                        )
                    )
                    cell_spectral_weight_dev = None
                    dynamic_spectrum_state = None
                    if (
                        normalized_cell_activity_mode == "per_channel"
                        and int(activity_groups_per_cell) > 1
                    ):
                        if spectrum_plan_context is None:
                            raise RuntimeError(
                                "Per-channel activity requires a prepared spectrum plan context."
                            )
                        cell_group_leakage_factors_dev = getattr(
                            spectrum_plan_context,
                            "d_cell_group_leakage_factors",
                            getattr(
                                spectrum_plan_context,
                                "cell_group_leakage_factors",
                                None,
                            ),
                        )
                        if any_power_outputs and cell_group_leakage_factors_dev is None:
                            raise RuntimeError(
                                "Spectrum plan context is missing per-group leakage factors."
                            )
                        if any_power_outputs:
                            cell_group_valid_mask_dev = getattr(
                                spectrum_plan_context,
                                "d_cell_group_valid_mask",
                                getattr(
                                    spectrum_plan_context,
                                    "cell_group_valid_mask",
                                    None,
                                ),
                            )
                            configured_group_counts_per_cell_dev = getattr(
                                spectrum_plan_context,
                                "d_configured_group_counts_per_cell",
                                getattr(
                                    spectrum_plan_context,
                                    "configured_group_counts_per_cell",
                                    None,
                                ),
                            )
                            if float(cell_activity_factor) <= 0.0:
                                cell_active_mask_dev = cp.zeros(
                                    (time_count_local, n_cells_total),
                                    dtype=cp.bool_,
                                )
                                cell_spectral_weight_dev = cp.zeros(
                                    (time_count_local, n_cells_total),
                                    dtype=cp.float32,
                                )
                            elif float(cell_activity_factor) >= 1.0:
                                if cell_group_valid_mask_dev is None:
                                    cell_active_mask_dev = cp.ones(
                                        (time_count_local, n_cells_total),
                                        dtype=cp.bool_,
                                    )
                                else:
                                    cell_active_rows = cp.asarray(
                                        cp.any(cell_group_valid_mask_dev, axis=1),
                                        dtype=cp.bool_,
                                    )
                                    cell_active_mask_dev = cp.broadcast_to(
                                        cell_active_rows[None, :],
                                        (time_count_local, n_cells_total),
                                    )
                                base_weight_dev = _compute_cell_spectral_weight_device(
                                    cp,
                                    group_active_mask=cp.ones(
                                        (1, n_cells_total, int(activity_groups_per_cell)),
                                        dtype=cp.bool_,
                                    ),
                                    cell_group_leakage_factors=cell_group_leakage_factors_dev,
                                    power_policy=str(
                                        spectrum_plan_effective["multi_group_power_policy"]
                                    ),
                                    split_total_group_denominator_mode=str(
                                        spectrum_plan_effective[
                                            "split_total_group_denominator_mode"
                                        ]
                                    ),
                                    configured_groups_per_cell=configured_group_counts_per_cell_dev,
                                    group_valid_mask=cell_group_valid_mask_dev,
                                )
                                cell_spectral_weight_dev = cp.broadcast_to(
                                    cp.asarray(base_weight_dev[:1, :], dtype=cp.float32),
                                    (time_count_local, n_cells_total),
                                ).astype(cp.float32, copy=False)
                            else:
                                group_active_mask_dev = _sample_cell_group_activity_mask_device(
                                    cp,
                                    time_count=time_count_local,
                                    cell_count=n_cells_total,
                                    group_count=int(activity_groups_per_cell),
                                    activity_factor=float(cell_activity_factor),
                                    seed=cell_activity_seed,
                                    group_valid_mask=cell_group_valid_mask_dev,
                                )
                                cell_active_mask_dev = _collapse_cell_group_activity_mask_device(
                                    cp,
                                    group_active_mask_dev,
                                )
                                dynamic_spectrum_state = _DirectEpfdDynamicSpectrumState(
                                    group_active_mask_dev=group_active_mask_dev,
                                    cell_group_leakage_factors_dev=cell_group_leakage_factors_dev,
                                    group_valid_mask_dev=cell_group_valid_mask_dev,
                                    power_policy=str(
                                        spectrum_plan_effective["multi_group_power_policy"]
                                    ),
                                    split_total_group_denominator_mode=str(
                                        spectrum_plan_effective[
                                            "split_total_group_denominator_mode"
                                        ]
                                    ),
                                    configured_groups_per_cell=configured_group_counts_per_cell_dev,
                                )
                        else:
                            cell_active_mask_dev, cell_spectral_weight_dev = (
                                _compute_cell_activity_spectral_weight_time_slabbed_device(
                                    cp,
                                    time_count=time_count_local,
                                    cell_count=n_cells_total,
                                    group_count=int(activity_groups_per_cell),
                                    activity_factor=float(cell_activity_factor),
                                    seed=cell_activity_seed,
                                    spectral_slab=int(
                                        current_batch_plan.get("spectral_slab", time_count_local)
                                    ),
                                    need_power_outputs=False,
                                    cell_group_leakage_factors=None,
                                    power_policy=str(
                                        spectrum_plan_effective["multi_group_power_policy"]
                                    ),
                                    split_total_group_denominator_mode=str(
                                        spectrum_plan_effective[
                                            "split_total_group_denominator_mode"
                                        ]
                                    ),
                                    configured_groups_per_cell=np.asarray(
                                        spectrum_plan_effective[
                                            "configured_group_counts_per_cell"
                                        ],
                                        dtype=np.int32,
                                    ),
                                    group_valid_mask=np.asarray(
                                        spectrum_plan_effective["cell_group_valid_mask"],
                                        dtype=bool,
                                    ),
                                )
                            )
                    else:
                        cell_active_mask_dev = _sample_cell_activity_mask_device(
                            cp,
                            time_count=time_count_local,
                            cell_count=n_cells_total,
                            activity_factor=float(cell_activity_factor),
                            seed=cell_activity_seed,
                        )
                    stage_start = record_profile_stage(
                        stage_timings,
                        "cell_activity_setup",
                        stage_start,
                        enabled=profile_stages,
                        synchronize=lambda: _sync_array_module(cp),
                    )
                    beam_demand_count_dev = None
                    if output_family_plan["needs_beam_demand"]:
                        beam_demand_count_dev = _count_active_beam_demand_device(
                            cp,
                            cell_active_mask_dev,
                            nco=int(nco),
                            dtype=demand_count_dtype,
                        )
                    if (
                        visible_count_observed is not None
                        and int(visible_count_observed) > int(visible_satellite_est)
                    ):
                        previous_visible_est = int(visible_satellite_est)
                        visible_satellite_est = int(
                            min(
                                n_sats_total,
                                max(previous_visible_est, int(visible_count_observed)),
                            )
                        )
                        current_batch_plan = _plan_direct_epfd_iteration_schedule(
                            session=session,
                            observer_context=observer_context,
                            satellite_context=satellite_context,
                            gpu_output_dtype=gpu_output_dtype,
                            n_steps_total=int(time_count_local),
                            n_cells_total=n_cells_total,
                            n_sats_total=n_sats_total,
                            n_skycells_s1586=n_skycells_s1586,
                            visible_satellite_est=int(visible_satellite_est),
                            nco=int(nco),
                            nbeam=int(nbeam),
                            boresight_active=boresight_active,
                            effective_ras_pointing_mode=effective_ras_pointing_mode,
                            output_family_plan=output_family_plan,
                            store_eligible_mask=bool(store_eligible_mask),
                            profile_stages=bool(profile_stages),
                            host_budget_info=host_budget_info,
                            gpu_budget_info=gpu_budget_info,
                            scheduler_target_fraction=float(scheduler_target_fraction),
                            scheduler_active_target_fraction=float(run_active_target_fraction),
                            finalize_memory_budget_bytes=finalize_memory_budget_bytes,
                            power_memory_budget_bytes=power_memory_budget_bytes,
                            export_memory_budget_bytes=export_memory_budget_bytes,
                            power_sky_slab=power_sky_slab,
                            force_bulk_timesteps=int(time_count_local),
                            force_cell_observer_chunk=force_cell_observer_chunk,
                            spectrum_context_bytes=int(spectrum_context_bytes),
                            cell_activity_mode=str(normalized_cell_activity_mode),
                            activity_groups_per_cell=int(activity_groups_per_cell),
                            activity_power_policy=str(activity_power_policy),
                            activity_split_total_group_denominator_mode=str(
                                activity_split_total_group_denominator_mode
                            ),
                            allow_warmup_calibration=False,
                        )
                        scheduler_runtime_state = _update_scheduler_runtime_state_from_plan(
                            scheduler_runtime_state,
                            current_batch_plan,
                            spectrum_context_bytes=int(spectrum_context_bytes),
                        )
                        cell_chunk = max(1, min(int(current_batch_plan["cell_chunk"]), n_cells_total))
                        n_cell_chunks = int(np.ceil(n_cells_total / cell_chunk))
                        iteration_plan = _plan_direct_epfd_iteration_schedule(
                            session=session,
                            observer_context=observer_context,
                            satellite_context=satellite_context,
                            gpu_output_dtype=gpu_output_dtype,
                            n_steps_total=int(n_steps_total),
                            n_cells_total=n_cells_total,
                            n_sats_total=n_sats_total,
                            n_skycells_s1586=n_skycells_s1586,
                            visible_satellite_est=int(visible_satellite_est),
                            nco=int(nco),
                            nbeam=int(nbeam),
                            boresight_active=boresight_active,
                            effective_ras_pointing_mode=effective_ras_pointing_mode,
                            output_family_plan=output_family_plan,
                            store_eligible_mask=bool(store_eligible_mask),
                            profile_stages=bool(profile_stages),
                            host_budget_info=host_budget_info,
                            gpu_budget_info=gpu_budget_info,
                            scheduler_target_fraction=float(scheduler_target_fraction),
                            scheduler_active_target_fraction=float(run_active_target_fraction),
                            finalize_memory_budget_bytes=finalize_memory_budget_bytes,
                            power_memory_budget_bytes=power_memory_budget_bytes,
                            export_memory_budget_bytes=export_memory_budget_bytes,
                            power_sky_slab=power_sky_slab,
                            force_bulk_timesteps=int(bulk_timesteps),
                            force_cell_observer_chunk=force_cell_observer_chunk,
                            spectrum_context_bytes=int(spectrum_context_bytes),
                            cell_activity_mode=str(normalized_cell_activity_mode),
                            activity_groups_per_cell=int(activity_groups_per_cell),
                            activity_power_policy=str(activity_power_policy),
                            activity_split_total_group_denominator_mode=str(
                                activity_split_total_group_denominator_mode
                            ),
                            allow_warmup_calibration=False,
                            surface_pfd_cap_enabled=bool(max_surface_pfd_enabled),
                            surface_pfd_cap_mode=str(surface_pfd_cap_mode),
                        )
                        scheduler_runtime_state = _update_scheduler_runtime_state_from_plan(
                            scheduler_runtime_state,
                            iteration_plan,
                            spectrum_context_bytes=int(spectrum_context_bytes),
                        )
                        iter_cell_chunk_plan = int(iteration_plan["cell_chunk"])
                        stage_budget_info = dict(iteration_plan["stage_budget_info"])
                        planned_power_sky_slab = int(iteration_plan["sky_slab"])
                        planned_batch_seconds = float(
                            iteration_plan.get("planned_batch_seconds", planned_batch_seconds)
                        )
                        planned_iteration_seconds = float(planned_batch_seconds) * float(n_batches)
                        planned_total_seconds = float(planned_iteration_seconds) * float(
                            int(iteration_count)
                        )
                        planned_remaining_seconds = (
                            float(planned_batch_seconds)
                            * float(max(1, n_batches - int(bi)))
                            + float(planned_iteration_seconds) * float(remaining_iterations_after)
                        )
                        visibility_payload = _build_direct_epfd_scheduler_payload(
                            host_budget_info=host_budget_info,
                            gpu_budget_info=gpu_budget_info,
                            scheduler_target_fraction=float(scheduler_target_fraction),
                            scheduler_active_target_fraction=float(run_active_target_fraction),
                            boresight_active=bool(boresight_active),
                            n_earthgrid_cells=int(n_cells_total),
                            n_skycells_s1586=int(n_skycells_s1586),
                            visible_satellite_est=int(visible_satellite_est),
                            bulk_timesteps=int(time_count_local),
                            cell_chunk=int(cell_chunk),
                            sky_slab=int(current_batch_plan["sky_slab"]),
                            predicted_host_peak_bytes=int(current_batch_plan["predicted_host_peak_bytes"]),
                            predicted_gpu_peak_bytes=int(current_batch_plan["predicted_gpu_peak_bytes"]),
                            planner_source=str(current_batch_plan["planner_source"]),
                            limiting_resource=str(current_batch_plan["limiting_resource"]),
                            planned_total_seconds=float(planned_total_seconds),
                            planned_remaining_seconds=float(batch_planned_remaining_seconds),
                            live_host_snapshot=_runtime_host_memory_snapshot(),
                            live_gpu_snapshot=_runtime_gpu_memory_snapshot(cp, session),
                            live_gpu_adapter_snapshot=_runtime_gpu_adapter_memory_snapshot(cp),
                            extra={
                                "visible_satellite_probe": int(visible_satellite_probe),
                                "visible_satellite_observed": int(visible_count_observed),
                                "scheduler_target_profile": str(scheduler_target_name),
                                "cell_activity_mode": str(normalized_cell_activity_mode),
                                "split_total_group_denominator_mode": str(
                                    normalized_split_denominator_mode
                                ),
                                **_scheduler_runtime_state_extra(scheduler_runtime_state),
                            },
                        )
                        _emit_direct_epfd_progress(
                            progress_callback,
                            kind="warning",
                            phase="compute",
                            iteration_index=int(ii),
                            iteration_total=int(iteration_count),
                            batch_index=int(bi),
                            batch_total=int(n_batches),
                            description=(
                                f"Observed {int(visible_count_observed)} visible satellites in batch "
                                f"{int(bi) + 1}/{int(n_batches)}; replanning from the earlier "
                                f"estimate of {int(previous_visible_est)}."
                            ),
                            **visibility_payload,
                        )
                    if not batch_has_visible_sats:
                        batch_payload = {"times": local_times.mjd}
                        if write_epfd:
                            batch_payload["EPFD_W_m2"] = np.zeros(
                                (time_count_local, 1, n_skycells_s1586),
                                dtype=np.float32,
                            )
                        if write_prx_total:
                            batch_payload["Prx_total_W"] = np.zeros(
                                (time_count_local, 1, n_skycells_s1586),
                                dtype=np.float32,
                            )
                        if write_per_satellite_prx_ras_station:
                            batch_payload["Prx_per_sat_RAS_STATION_W"] = (
                                np.zeros(
                                    (time_count_local, 1, n_sats_total, n_skycells_s1586),
                                    dtype=np.float32,
                                )
                                if boresight_active
                                else np.zeros((time_count_local, n_sats_total), dtype=np.float32)
                            )
                        if write_total_pfd_ras_station:
                            batch_payload["PFD_total_RAS_STATION_W_m2"] = (
                                np.zeros(
                                    (time_count_local, 1, n_skycells_s1586),
                                    dtype=np.float32,
                                )
                                if boresight_active
                                else np.zeros(time_count_local, dtype=np.float32)
                            )
                        if write_per_satellite_pfd_ras_station:
                            batch_payload["PFD_per_sat_RAS_STATION_W_m2"] = (
                                np.zeros(
                                    (time_count_local, 1, n_sats_total, n_skycells_s1586),
                                    dtype=np.float32,
                                )
                                if boresight_active
                                else np.zeros((time_count_local, n_sats_total), dtype=np.float32)
                            )
                        if write_sat_beam_counts_used:
                            batch_payload["sat_beam_counts_used"] = (
                                np.zeros(
                                    (time_count_local, 1, n_sats_total, n_skycells_s1586),
                                    dtype=count_dtype,
                                )
                                if boresight_active
                                else np.zeros((time_count_local, n_sats_total), dtype=count_dtype)
                            )
                        if write_sat_elevation_ras_station:
                            batch_payload["sat_elevation_RAS_STATION_deg"] = np.full(
                                (time_count_local, n_sats_total),
                                np.nan,
                                dtype=np.float32,
                            )
                        if write_beam_demand_count:
                            batch_payload["beam_demand_count"] = gpu_module.copy_device_to_host(
                                beam_demand_count_dev
                            )
                        if beam_stats_collector is not None:
                            full_zero_hist = np.zeros(1, dtype=np.int64)
                            full_zero_hist[0] = int(time_count_local) * int(n_sats_total)
                            beam_stats_collector["full_network_count_hist"] = _merge_count_histograms(
                                beam_stats_collector["full_network_count_hist"],
                                full_zero_hist,
                            )
                            beam_stats_collector["visible_count_hist"] = _merge_count_histograms(
                                beam_stats_collector["visible_count_hist"],
                                np.zeros(1, dtype=np.int64),
                            )
                            _append_series_segment(
                                beam_stats_collector["network_total_beams_over_time"],
                                np.zeros((time_count_local,), dtype=np.int64),
                            )
                            _append_series_segment(
                                beam_stats_collector["visible_total_beams_over_time"],
                                np.zeros((time_count_local,), dtype=np.int64),
                            )
                            if beam_demand_count_dev is not None:
                                _append_series_segment(
                                    beam_stats_collector["beam_demand_over_time"],
                                    gpu_module.copy_device_to_host(beam_demand_count_dev),
                                )
                        write_enqueue_stage_summary = _start_direct_epfd_stage_memory_summary(
                            "write_enqueue",
                            cp=cp,
                            session=session,
                        )
                        _set_direct_epfd_progress_phase(
                            pbar,
                            enable_progress_bars=enable_progress_bars,
                            progress_desc_mode=progress_desc_mode_name,
                            phase="write_enqueue",
                        )
                        _emit_direct_epfd_progress(
                            progress_callback,
                            kind="phase",
                            phase="write_enqueue",
                            iteration_index=int(ii),
                            iteration_total=int(iteration_count),
                            batch_index=int(bi),
                            batch_total=int(n_batches),
                            description=_direct_epfd_progress_text(
                                progress_desc_mode_name,
                                "write_enqueue",
                            ),
                        )
                        write_enqueue_t0 = perf_counter() if profile_stages else None
                        _write_iteration_batch_owned(
                            storage_filename,
                            iteration=ii,
                            batch_items=tuple(batch_payload.items()),
                            compression=hdf5_compression,
                            compression_opts=hdf5_compression_opts,
                            writer_queue_max_items=writer_queue_max_items,
                            writer_queue_max_bytes=writer_queue_max_bytes,
                        )
                        (
                            last_writer_checkpoint_monotonic,
                            checkpoint_wait_elapsed,
                            checkpoint_triggered,
                        ) = _maybe_checkpoint_writer_durable(
                            storage_filename,
                            checkpoint_interval_s=writer_checkpoint_interval_s_name,
                            last_checkpoint_monotonic=last_writer_checkpoint_monotonic,
                            pbar=pbar,
                            enable_progress_bars=enable_progress_bars,
                            progress_desc_mode=progress_desc_mode_name,
                        )
                        if checkpoint_triggered:
                            writer_checkpoint_count += 1
                            writer_checkpoint_wait_s += float(checkpoint_wait_elapsed)
                            _emit_direct_epfd_progress(
                                progress_callback,
                                kind="phase",
                                phase="checkpoint",
                                iteration_index=int(ii),
                                iteration_total=int(iteration_count),
                                batch_index=int(bi),
                                batch_total=int(n_batches),
                                elapsed_s=float(checkpoint_wait_elapsed),
                                checkpoint_count=int(writer_checkpoint_count),
                                description=_direct_epfd_progress_text(
                                    progress_desc_mode_name,
                                    "checkpoint",
                                ),
                            )
                        write_enqueue_stage_summary = _update_direct_epfd_stage_memory_summary(
                            write_enqueue_stage_summary,
                            _capture_direct_epfd_live_memory_snapshot(cp, session),
                        )
                        scheduler_runtime_state["last_observed_stage_summary"] = dict(
                            write_enqueue_stage_summary
                        )
                        if profile_stages and stage_timings is not None and write_enqueue_t0 is not None:
                            stage_timings["export_copy"] = 0.0
                            stage_timings["write_enqueue"] = perf_counter() - write_enqueue_t0
                            stage_timings["writer_checkpoint_wait"] = float(
                                checkpoint_wait_elapsed
                            )
                            profile_stage_timings_all.append(
                                {
                                    "iteration": int(ii),
                                    "batch_index": int(bi),
                                    **{k: float(v) for k, v in stage_timings.items()},
                                }
                            )
                            for name, value in stage_timings.items():
                                profile_stage_timings_summary[name] = (
                                    float(profile_stage_timings_summary.get(name, 0.0)) + float(value)
                                )
                            _emit_direct_epfd_progress(
                                progress_callback,
                                kind="batch_done",
                                phase="compute",
                                iteration_index=int(ii),
                                iteration_total=int(iteration_count),
                                batch_index=int(bi),
                                batch_total=int(n_batches),
                                stage_timings={k: float(v) for k, v in stage_timings.items()},
                            )
                            _batch_wall = perf_counter() - _batch_loop_t0
                            _stage_sum = sum(float(v) for v in stage_timings.values())
                            print(f"[batch {bi}] wall={_batch_wall:.3f}s stages={_stage_sum:.3f}s overhead={_batch_wall - _stage_sum:.3f}s")
                        cancel_mode = _query_direct_epfd_cancel_mode(cancel_callback)
                        if cancel_mode in {"graceful", "force"}:
                            run_state = "stopped"
                            stop_mode = cancel_mode
                            stop_boundary = "post_batch_boundary"
                            if not stop_notice_emitted:
                                _emit_direct_epfd_progress(
                                    progress_callback,
                                    kind="phase",
                                    phase="stopping",
                                    iteration_index=int(ii),
                                    iteration_total=int(iteration_count),
                                    batch_index=int(bi),
                                    batch_total=int(n_batches),
                                    stop_mode=str(stop_mode),
                                    stop_boundary=str(stop_boundary),
                                    description="Stop requested. Flushing completed work before exit.",
                                )
                                stop_notice_emitted = True
                            break
                        continue

                    batch_payload = {"times": local_times.mjd}
                    if write_beam_demand_count:
                        batch_payload["beam_demand_count"] = gpu_module.copy_device_to_host(
                            beam_demand_count_dev
                        )
                    need_beam_finalize = bool(
                        any_power_outputs
                        or output_family_plan["needs_beam_counts"]
                        or bool(store_eligible_mask)
                    )
                    if need_beam_finalize:
                        batch_compute_active_target_fraction = float(run_active_target_fraction)
                        batch_stage_budget_info = dict(current_batch_plan["stage_budget_info"])
                        batch_power_sky_slab = int(current_batch_plan["sky_slab"])
                        batch_plan_for_retry = dict(current_batch_plan)
                        retry_targets = _scheduler_backoff_fractions(
                            batch_compute_active_target_fraction
                        )[1:]
                        retry_count = 0
                        try:
                            while True:
                                retry_signature_before = (
                                    int(gpu_budget_info["effective_budget_bytes"]),
                                    float(batch_compute_active_target_fraction),
                                    int(cell_chunk),
                                    int(batch_power_sky_slab),
                                    int(batch_plan_for_retry.get("spectral_slab", 1)),
                                )
                                try:
                                    compute_payload = _compute_gpu_direct_epfd_batch_device(
                                        session=session,
                                        cp=cp,
                                        observer_context=observer_context,
                                        orbit_state=orbit_state,
                                        sat_topo_ras_station=sat_topo_ras_station,
                                        sat_azel_ras_station=sat_azel_ras_station,
                                        sat_keep_batch=sat_keep_batch,
                                        sat_min_elev_deg_per_sat_f64=sat_min_elev_deg_per_sat_f64,
                                        sat_beta_max_deg_per_sat_f32=sat_beta_max_deg_per_sat_f32,
                                        sat_belt_id_per_sat_i16=sat_belt_id_per_sat_i16,
                                        selection_mode=selection_mode,
                                        nco=int(nco),
                                        nbeam=int(nbeam),
                                        n_cells_total=n_cells_total,
                                        cell_active_mask_dev=cell_active_mask_dev,
                                        cell_spectral_weight_dev=cell_spectral_weight_dev,
                                        dynamic_spectrum_state=dynamic_spectrum_state,
                                        ras_service_cell_index=int(ras_service_cell_index),
                                        effective_ras_pointing_mode=effective_ras_pointing_mode,
                                        ras_guard_angle_rad=ras_guard_angle_rad,
                                        boresight_active=boresight_active,
                                        boresight_theta1_deg=boresight_theta1_deg,
                                        boresight_theta2_deg=boresight_theta2_deg,
                                        boresight_theta2_cell_ids=boresight_theta2_cell_ids,
                                        pointings=pointings,
                                        time_count_local=time_count_local,
                                        cell_chunk=cell_chunk,
                                        n_cell_chunks=n_cell_chunks,
                                        gpu_output_dtype=gpu_output_dtype,
                                        profile_stages=profile_stages,
                                        stage_timings=stage_timings,
                                        stage_start=stage_start,
                                        enable_progress_bars=enable_progress_bars,
                                        progress_desc_mode=progress_desc_mode_name,
                                        pbar=pbar,
                                        ii=ii,
                                        bi=bi,
                                        orbit_radius_full=orbit_radius_full,
                                        observer_alt_km_ras_station=observer_alt_km_ras_station,
                                        power_input=power_input,
                                        spectrum_plan_context=spectrum_plan_context,
                                        target_alt_km=target_alt_km,
                                        use_ras_station_alt_for_co=bool(use_ras_station_alt_for_co),
                                        s1528_pattern_context=s1528_pattern_context,
                                        ras_pattern_context=ras_pattern_context,
                                        atmosphere_context=atmosphere_context,
                                        peak_pfd_lut_context=peak_pfd_lut_context,
                                        max_surface_pfd_dbw_m2_channel=max_surface_pfd_dbw_m2_channel,
                                        max_surface_pfd_dbw_m2_mhz=max_surface_pfd_dbw_m2_mhz,
                                        surface_pfd_cap_mode=str(surface_pfd_cap_mode),
                                        surface_pfd_stats_enabled=bool(surface_pfd_stats_enabled),
                                        host_effective_budget_bytes=int(
                                            host_budget_info["effective_budget_bytes"]
                                        ),
                                        gpu_effective_budget_bytes=int(
                                            gpu_budget_info["effective_budget_bytes"]
                                        ),
                                        scheduler_active_target_fraction=float(
                                            batch_compute_active_target_fraction
                                        ),
                                        predicted_host_peak_bytes=int(
                                            batch_plan_for_retry["predicted_host_peak_bytes"]
                                        ),
                                        predicted_gpu_propagation_peak_bytes=int(
                                            batch_plan_for_retry["predicted_gpu_propagation_peak_bytes"]
                                        ),
                                        predicted_gpu_finalize_peak_bytes=int(
                                            batch_plan_for_retry["predicted_gpu_finalize_peak_bytes"]
                                        ),
                                        predicted_gpu_power_peak_bytes=int(
                                            batch_plan_for_retry["predicted_gpu_power_peak_bytes"]
                                        ),
                                        finalize_memory_budget_bytes=int(
                                            batch_stage_budget_info["finalize_memory_budget_bytes"]
                                        ),
                                        power_memory_budget_bytes=int(
                                            batch_stage_budget_info["power_memory_budget_bytes"]
                                        ),
                                        power_sky_slab=batch_power_sky_slab,
                                        spectral_slab=int(batch_plan_for_retry.get("spectral_slab", 1)),
                                        visibility_elev_threshold_deg=float(visibility_elev_threshold_deg),
                                        debug_direct_epfd=bool(debug_direct_epfd),
                                        write_epfd=bool(output_family_plan["needs_epfd"]),
                                        write_prx_total=bool(output_family_plan["needs_total_prx"]),
                                        write_per_satellite_prx_ras_station=bool(
                                            output_family_plan["needs_per_satellite_prx"]
                                        ),
                                        write_prx_elevation_heatmap=bool(write_prx_elevation_heatmap),
                                        write_total_pfd_ras_station=bool(output_family_plan["needs_total_pfd"]),
                                        write_per_satellite_pfd_ras_station=bool(
                                            output_family_plan["needs_per_satellite_pfd"]
                                        ),
                                        write_sat_beam_counts_used=bool(output_family_plan["needs_beam_counts"]),
                                        write_sat_eligible_mask=bool(store_eligible_mask),
                                        progress_callback=progress_callback,
                                        cancel_callback=cancel_callback,
                                        uemr_mode=bool(uemr_mode),
                                    )
                                    power_result = compute_payload["power_result"]
                                    sat_idx_g = compute_payload["sat_idx_g"]
                                    sat_topo = compute_payload["sat_topo_visible"]
                                    sat_azel = compute_payload["sat_azel_visible"]
                                    orbit_radius_eff = compute_payload["orbit_radius_eff"]
                                    sat_beam_counts_used_full = compute_payload["sat_beam_counts_used_full"]
                                    sat_eligible_mask = compute_payload["sat_eligible_mask"]
                                    diag_result = compute_payload["diag_result"]

                                    # --- Multi-system per-batch interleaving ---
                                    # Process additional systems and combine power.
                                    if _multi_system_active and power_result is not None:
                                        _ms_power_results = [power_result]
                                        for _ms_idx in range(1, len(_multi_system_contexts)):
                                            _ms_ctx = _multi_system_contexts[_ms_idx]
                                            _ms_sat_ctx = _ms_ctx["satellite_context"]
                                            _ms_orbit_radius_full = _ms_ctx["orbit_radius_dev"]
                                            _ms_n_sats = _ms_ctx["n_sats_total"]

                                            # Propagate this system's satellites
                                            _ms_orbit_state = session.propagate_orbit(
                                                mjd,
                                                _ms_sat_ctx,
                                                on_error=gpu_on_error,
                                            )
                                            _ms_observer_ctx_geo = _ms_ctx.get("observer_context") or observer_context
                                            _ms_ras_result = session.derive_from_eci(
                                                _ms_orbit_state,
                                                observer_context=_ms_observer_ctx_geo,
                                                observer_slice=slice(0, 1),
                                                do_eci_pos=False,
                                                do_eci_vel=False,
                                                do_geo=False,
                                                do_topo=True,
                                                do_obs_pos=False,
                                                do_sat_azel=True,
                                                do_sat_rotmat=False,
                                                output_dtype=gpu_output_dtype,
                                                return_device=True,
                                            )
                                            _ms_sat_topo = cp.asarray(_ms_ras_result["topo"])[:, 0, :, :]
                                            _ms_sat_azel = cp.asarray(_ms_ras_result["sat_azel"])[:, 0, :, :]
                                            _ms_sat_keep = cp.any(
                                                _ms_sat_topo[..., 1] > cp.float32(visibility_elev_threshold_deg), axis=0,
                                            )
                                            _ms_has_visible = bool(int(cp.any(_ms_sat_keep).get()) != 0)

                                            if not _ms_has_visible:
                                                del _ms_ras_result, _ms_orbit_state
                                                _ms_power_results.append(None)
                                                continue

                                            # Cell activity for this system
                                            _ms_caf = float(_ms_ctx["cell_activity_factor"])
                                            _ms_seed = _ms_ctx["cell_activity_seed_base"]
                                            _ms_cell_seed = (
                                                None if _ms_seed is None
                                                else int(
                                                    (int(_ms_seed) + ii * 1_000_000 + bi * 10_000 + _ms_idx * 100)
                                                    % (2**32 - 1)
                                                )
                                            )
                                            _ms_n_cells = _ms_ctx["n_cells_total"]
                                            _ms_cell_chunk = _ms_ctx["cell_chunk"]
                                            _ms_n_cell_chunks = _ms_ctx["n_cell_chunks"]
                                            _ms_cell_active = _sample_cell_activity_mask_device(
                                                cp,
                                                time_count=time_count_local,
                                                cell_count=_ms_n_cells,
                                                activity_factor=_ms_caf,
                                                seed=_ms_cell_seed,
                                            )

                                            # Per-system beam demand
                                            _ms_beam_demand_dev = None
                                            if output_family_plan["needs_beam_demand"]:
                                                _ms_beam_demand_dev = _count_active_beam_demand_device(
                                                    cp,
                                                    _ms_cell_active,
                                                    nco=int(_ms_ctx["nco"]),
                                                    dtype=demand_count_dtype,
                                                )

                                            # Compute power for this system
                                            try:
                                                _ms_observer_ctx = _ms_ctx.get("observer_context") or observer_context
                                                _ms_compute = _compute_gpu_direct_epfd_batch_device(
                                                    session=session,
                                                    cp=cp,
                                                    observer_context=_ms_observer_ctx,
                                                    orbit_state=_ms_orbit_state,
                                                    sat_topo_ras_station=_ms_sat_topo,
                                                    sat_azel_ras_station=_ms_sat_azel,
                                                    sat_keep_batch=_ms_sat_keep,
                                                    sat_min_elev_deg_per_sat_f64=_ms_ctx["sat_min_elev_deg_per_sat_f64"],
                                                    sat_beta_max_deg_per_sat_f32=_ms_ctx["sat_beta_max_deg_per_sat_f32"],
                                                    sat_belt_id_per_sat_i16=_ms_ctx["sat_belt_id_per_sat_i16"],
                                                    selection_mode=_ms_ctx["selection_mode"],
                                                    nco=_ms_ctx["nco"],
                                                    nbeam=_ms_ctx["nbeam"],
                                                    n_cells_total=_ms_n_cells,
                                                    cell_active_mask_dev=_ms_cell_active,
                                                    cell_spectral_weight_dev=None,
                                                    dynamic_spectrum_state=None,
                                                    ras_service_cell_index=_ms_ctx["ras_service_cell_index"],
                                                    effective_ras_pointing_mode=effective_ras_pointing_mode,
                                                    ras_guard_angle_rad=ras_guard_angle_rad,
                                                    boresight_active=bool(_ms_ctx.get("boresight_active", False)),
                                                    boresight_theta1_deg=_ms_ctx.get("boresight_theta1_deg"),
                                                    boresight_theta2_deg=_ms_ctx.get("boresight_theta2_deg"),
                                                    boresight_theta2_cell_ids=_ms_ctx.get("boresight_theta2_cell_ids"),
                                                    pointings=pointings,
                                                    time_count_local=time_count_local,
                                                    cell_chunk=_ms_cell_chunk,
                                                    n_cell_chunks=_ms_n_cell_chunks,
                                                    gpu_output_dtype=gpu_output_dtype,
                                                    profile_stages=False,
                                                    stage_timings={},
                                                    stage_start=perf_counter(),
                                                    enable_progress_bars=False,
                                                    progress_desc_mode=progress_desc_mode_name,
                                                    pbar=None,
                                                    ii=ii,
                                                    bi=bi,
                                                    orbit_radius_full=_ms_orbit_radius_full,
                                                    observer_alt_km_ras_station=observer_alt_km_ras_station,
                                                    power_input=_ms_ctx["power_input"],
                                                    spectrum_plan_context=_ms_ctx["spectrum_plan_context"],
                                                    target_alt_km=target_alt_km,
                                                    use_ras_station_alt_for_co=bool(use_ras_station_alt_for_co),
                                                    s1528_pattern_context=_ms_ctx["s1528_pattern_context"],
                                                    ras_pattern_context=(
                                                        _ms_ctx["ras_pattern_context"]
                                                        if _ms_ctx["ras_pattern_context"] is not None
                                                        else ras_pattern_context
                                                    ),
                                                    atmosphere_context=(
                                                        _ms_ctx["atmosphere_context"]
                                                        if _ms_ctx["atmosphere_context"] is not None
                                                        else atmosphere_context
                                                    ),
                                                    peak_pfd_lut_context=_ms_ctx.get(
                                                        "peak_pfd_lut_context"
                                                    ),
                                                    max_surface_pfd_dbw_m2_channel=_ms_ctx.get(
                                                        "max_surface_pfd_dbw_m2_channel"
                                                    ),
                                                    max_surface_pfd_dbw_m2_mhz=_ms_ctx.get(
                                                        "max_surface_pfd_dbw_m2_mhz"
                                                    ),
                                                    surface_pfd_cap_mode=str(
                                                        _ms_ctx.get("surface_pfd_cap_mode", "per_beam")
                                                    ),
                                                    surface_pfd_stats_enabled=bool(
                                                        _ms_ctx.get("surface_pfd_stats_enabled", False)
                                                    ),
                                                    host_effective_budget_bytes=int(
                                                        host_budget_info["effective_budget_bytes"]
                                                    ),
                                                    gpu_effective_budget_bytes=int(
                                                        gpu_budget_info["effective_budget_bytes"]
                                                    ),
                                                    scheduler_active_target_fraction=float(
                                                        batch_compute_active_target_fraction
                                                    ),
                                                    predicted_host_peak_bytes=int(
                                                        batch_plan_for_retry["predicted_host_peak_bytes"]
                                                    ),
                                                    predicted_gpu_propagation_peak_bytes=int(
                                                        batch_plan_for_retry["predicted_gpu_propagation_peak_bytes"]
                                                    ),
                                                    predicted_gpu_finalize_peak_bytes=int(
                                                        batch_plan_for_retry["predicted_gpu_finalize_peak_bytes"]
                                                    ),
                                                    predicted_gpu_power_peak_bytes=int(
                                                        batch_plan_for_retry["predicted_gpu_power_peak_bytes"]
                                                    ),
                                                    finalize_memory_budget_bytes=int(
                                                        batch_stage_budget_info["finalize_memory_budget_bytes"]
                                                    ),
                                                    power_memory_budget_bytes=int(
                                                        batch_stage_budget_info["power_memory_budget_bytes"]
                                                    ),
                                                    power_sky_slab=batch_power_sky_slab,
                                                    spectral_slab=int(batch_plan_for_retry.get("spectral_slab", 1)),
                                                    visibility_elev_threshold_deg=float(visibility_elev_threshold_deg),
                                                    debug_direct_epfd=False,
                                                    write_epfd=bool(output_family_plan["needs_epfd"]),
                                                    write_prx_total=bool(output_family_plan["needs_total_prx"]),
                                                    write_per_satellite_prx_ras_station=bool(
                                                        output_family_plan["needs_per_satellite_prx"]
                                                    ),
                                                    write_prx_elevation_heatmap=False,
                                                    write_total_pfd_ras_station=bool(
                                                        output_family_plan["needs_total_pfd"]
                                                    ),
                                                    write_per_satellite_pfd_ras_station=bool(
                                                        output_family_plan["needs_per_satellite_pfd"]
                                                    ),
                                                    write_sat_beam_counts_used=bool(
                                                        output_family_plan["needs_beam_counts"]
                                                    ),
                                                    write_sat_eligible_mask=False,
                                                    progress_callback=None,
                                                    cancel_callback=cancel_callback,
                                                    uemr_mode=bool(_ms_ctx.get("uemr_mode", False)),
                                                )
                                                _ms_pr = _ms_compute.get("power_result")
                                                # Inject beam counts, elevation, and diagnostic data
                                                # into power result for per-system raw HDF5 stash
                                                if _ms_pr is not None:
                                                    _ms_beam_full = _ms_compute.get("sat_beam_counts_used_full")
                                                    if _ms_beam_full is not None:
                                                        _ms_pr["_sat_beam_counts_used"] = _ms_beam_full
                                                    _ms_sat_idx_g = _ms_compute.get("sat_idx_g")
                                                    if _ms_sat_idx_g is not None:
                                                        _ms_pr["_sat_idx_g"] = _ms_sat_idx_g
                                                    if _ms_sat_topo is not None:
                                                        _ms_pr["_sat_elevation_deg"] = _ms_sat_topo[:, :, 1]
                                                    # Inject per-system beam demand (computed before power)
                                                    if _ms_beam_demand_dev is not None:
                                                        _ms_pr["_beam_demand_count"] = _ms_beam_demand_dev
                                                _ms_power_results.append(_ms_pr)

                                                # Per-system heatmap accumulation
                                                if (
                                                    _ms_pr is not None
                                                    and _ms_idx < len(_per_system_heatmap_collectors)
                                                ):
                                                    _ps_hm = _per_system_heatmap_collectors[_ms_idx]
                                                    _ms_sat_topo_vis = _ms_compute.get("sat_topo_visible")
                                                    if _ms_sat_topo_vis is not None:
                                                        if (
                                                            "prx_elevation_heatmap" in _ps_hm
                                                            and "Prx_per_sat_RAS_STATION_W" in _ms_pr
                                                        ):
                                                            _accumulate_heatmap_batch(
                                                                session,
                                                                _ps_hm["prx_elevation_heatmap"],
                                                                value_linear=_ms_pr["Prx_per_sat_RAS_STATION_W"],
                                                                sat_elevation_deg=_ms_sat_topo_vis[:, :, 1],
                                                                db_offset_db=0.0,
                                                            )
                                                        if (
                                                            "per_satellite_pfd_elevation_heatmap" in _ps_hm
                                                            and "PFD_per_sat_RAS_STATION_W_m2" in _ms_pr
                                                        ):
                                                            _accumulate_heatmap_batch(
                                                                session,
                                                                _ps_hm["per_satellite_pfd_elevation_heatmap"],
                                                                value_linear=_ms_pr["PFD_per_sat_RAS_STATION_W_m2"],
                                                                sat_elevation_deg=_ms_sat_topo_vis[:, :, 1],
                                                                db_offset_db=0.0,
                                                            )

                                                # Per-system beam stats accumulation
                                                if (
                                                    _ms_idx < len(_per_system_beam_stats_collectors)
                                                    and _per_system_beam_stats_collectors[_ms_idx] is not None
                                                ):
                                                    _ps_bs = _per_system_beam_stats_collectors[_ms_idx]
                                                    _ms_beam_full = _ms_compute.get("sat_beam_counts_used_full")
                                                    if _ms_beam_full is not None:
                                                        _ms_beam_dev = cp.asarray(_ms_beam_full, dtype=np.int64)
                                                        _ms_beam_samples = _beam_count_samples_device(cp, _ms_beam_dev)
                                                        _ms_full_hist = _bincount_device_to_host(
                                                            cp, gpu_module,
                                                            _ms_beam_samples.reshape(-1),
                                                        )
                                                        _ps_bs["full_network_count_hist"] = _merge_count_histograms(
                                                            _ps_bs["full_network_count_hist"],
                                                            _ms_full_hist,
                                                        )
                                                        _ms_net_total = _beam_total_over_time_device(cp, _ms_beam_dev)
                                                        _ms_vis_mask = cp.asarray(
                                                            _ms_sat_topo[..., 1] > cp.float32(visibility_elev_threshold_deg),
                                                            dtype=bool,
                                                        )
                                                        _ms_vis_hist, _ = _visible_beam_statistics_device(
                                                            cp, gpu_module,
                                                            counts_samples_device=_ms_beam_samples,
                                                            visibility_mask_device=_ms_vis_mask,
                                                        )
                                                        _ms_vis_total = _visible_beam_total_over_time_device(
                                                            cp,
                                                            counts_device=_ms_beam_dev,
                                                            visibility_mask_device=_ms_vis_mask,
                                                        )
                                                        _ps_bs["visible_count_hist"] = _merge_count_histograms(
                                                            _ps_bs["visible_count_hist"],
                                                            _ms_vis_hist,
                                                        )
                                                        _append_series_segment(
                                                            _ps_bs["network_total_beams_over_time"],
                                                            gpu_module.copy_device_to_host(_ms_net_total),
                                                        )
                                                        _append_series_segment(
                                                            _ps_bs["visible_total_beams_over_time"],
                                                            gpu_module.copy_device_to_host(_ms_vis_total),
                                                        )
                                                    # Per-system beam demand (computed before the compute call)
                                                    if _ms_beam_demand_dev is not None:
                                                        _append_series_segment(
                                                            _ps_bs["beam_demand_over_time"],
                                                            gpu_module.copy_device_to_host(_ms_beam_demand_dev),
                                                        )
                                                    if _ms_beam_full is not None:
                                                        del _ms_beam_dev, _ms_beam_samples, _ms_vis_mask
                                            except Exception as _ms_exc:
                                                # If a secondary system fails, log and skip for this batch
                                                import traceback as _ms_tb
                                                print(
                                                    f"[WARNING] Secondary system {_ms_idx} "
                                                    f"({_ms_ctx.get('system_name', '?')}) "
                                                    f"failed: {_ms_exc}",
                                                    flush=True,
                                                )
                                                _ms_tb.print_exc()
                                                _ms_power_results.append(None)
                                            finally:
                                                del _ms_ras_result, _ms_orbit_state

                                        # Accumulate per-system histograms before combining
                                        for _ps_idx, _ps_pr in enumerate(_ms_power_results):
                                            if _ps_pr is None or _ps_idx >= len(_per_system_distribution_collectors):
                                                continue
                                            _ps_colls = _per_system_distribution_collectors[_ps_idx]
                                            if "epfd_distribution" in _ps_colls and "EPFD_W_m2" in _ps_pr:
                                                _accumulate_1d_distribution_batch(
                                                    session, _ps_colls["epfd_distribution"],
                                                    value_linear=_ps_pr["EPFD_W_m2"],
                                                    db_offset_db=0.0, batch_index=bi,
                                                )
                                            if "total_pfd_ras_distribution" in _ps_colls and "PFD_total_RAS_STATION_W_m2" in _ps_pr:
                                                _accumulate_1d_distribution_batch(
                                                    session, _ps_colls["total_pfd_ras_distribution"],
                                                    value_linear=_ps_pr["PFD_total_RAS_STATION_W_m2"],
                                                    db_offset_db=0.0, batch_index=bi,
                                                )
                                            if "per_satellite_pfd_distribution" in _ps_colls and "PFD_per_sat_RAS_STATION_W_m2" in _ps_pr:
                                                _accumulate_1d_distribution_batch(
                                                    session, _ps_colls["per_satellite_pfd_distribution"],
                                                    value_linear=_ps_pr["PFD_per_sat_RAS_STATION_W_m2"],
                                                    db_offset_db=0.0, batch_index=bi,
                                                )
                                            if "prx_total_distribution" in _ps_colls and "Prx_total_W" in _ps_pr:
                                                _accumulate_1d_distribution_batch(
                                                    session, _ps_colls["prx_total_distribution"],
                                                    value_linear=_ps_pr["Prx_total_W"],
                                                    db_offset_db=0.0, batch_index=bi,
                                                )

                                        # Accumulate system 0 heatmap and beam stats
                                        # (system 0 uses the primary compute payload)
                                        if (
                                            _per_system_heatmap_collectors
                                            and power_result is not None
                                            and sat_topo is not None
                                        ):
                                            _ps0_hm = _per_system_heatmap_collectors[0]
                                            if (
                                                "prx_elevation_heatmap" in _ps0_hm
                                                and "Prx_per_sat_RAS_STATION_W" in power_result
                                            ):
                                                _accumulate_heatmap_batch(
                                                    session,
                                                    _ps0_hm["prx_elevation_heatmap"],
                                                    value_linear=power_result["Prx_per_sat_RAS_STATION_W"],
                                                    sat_elevation_deg=sat_topo[:, :, 1],
                                                    db_offset_db=0.0,
                                                )
                                            if (
                                                "per_satellite_pfd_elevation_heatmap" in _ps0_hm
                                                and "PFD_per_sat_RAS_STATION_W_m2" in power_result
                                            ):
                                                _accumulate_heatmap_batch(
                                                    session,
                                                    _ps0_hm["per_satellite_pfd_elevation_heatmap"],
                                                    value_linear=power_result["PFD_per_sat_RAS_STATION_W_m2"],
                                                    sat_elevation_deg=sat_topo[:, :, 1],
                                                    db_offset_db=0.0,
                                                )
                                        if (
                                            _per_system_beam_stats_collectors
                                            and _per_system_beam_stats_collectors[0] is not None
                                            and sat_beam_counts_used_full is not None
                                        ):
                                            _ps0_bs = _per_system_beam_stats_collectors[0]
                                            _ps0_beam_dev = cp.asarray(sat_beam_counts_used_full, dtype=np.int64)
                                            _ps0_beam_samples = _beam_count_samples_device(cp, _ps0_beam_dev)
                                            _ps0_full_hist = _bincount_device_to_host(
                                                cp, gpu_module,
                                                _ps0_beam_samples.reshape(-1),
                                            )
                                            _ps0_bs["full_network_count_hist"] = _merge_count_histograms(
                                                _ps0_bs["full_network_count_hist"],
                                                _ps0_full_hist,
                                            )
                                            _ps0_net_total = _beam_total_over_time_device(cp, _ps0_beam_dev)
                                            _ps0_vis_mask = cp.asarray(
                                                sat_topo_ras_station[..., 1] > cp.float32(visibility_elev_threshold_deg),
                                                dtype=bool,
                                            )
                                            _ps0_vis_hist, _ = _visible_beam_statistics_device(
                                                cp, gpu_module,
                                                counts_samples_device=_ps0_beam_samples,
                                                visibility_mask_device=_ps0_vis_mask,
                                            )
                                            _ps0_vis_total = _visible_beam_total_over_time_device(
                                                cp,
                                                counts_device=_ps0_beam_dev,
                                                visibility_mask_device=_ps0_vis_mask,
                                            )
                                            _ps0_bs["visible_count_hist"] = _merge_count_histograms(
                                                _ps0_bs["visible_count_hist"],
                                                _ps0_vis_hist,
                                            )
                                            _append_series_segment(
                                                _ps0_bs["network_total_beams_over_time"],
                                                gpu_module.copy_device_to_host(_ps0_net_total),
                                            )
                                            _append_series_segment(
                                                _ps0_bs["visible_total_beams_over_time"],
                                                gpu_module.copy_device_to_host(_ps0_vis_total),
                                            )
                                            del _ps0_beam_dev, _ps0_beam_samples, _ps0_vis_mask
                                            # System 0 beam demand (from the primary activity computation)
                                            if beam_demand_count_dev is not None:
                                                _append_series_segment(
                                                    _ps0_bs["beam_demand_over_time"],
                                                    gpu_module.copy_device_to_host(beam_demand_count_dev),
                                                )

                                        # Accumulate multi-system group distributions
                                        # (groups spanning >1 system that are NOT the
                                        # full combined set handled by root collectors)
                                        for _gc in _group_collectors:
                                            if len(_gc["system_indices"]) <= 1:
                                                continue  # already handled by per-system
                                            _gc_indices = _gc["system_indices"]
                                            # Sum power across the group's systems
                                            _gc_prs = [
                                                _ms_power_results[si]
                                                for si in sorted(_gc_indices)
                                                if si < len(_ms_power_results) and _ms_power_results[si] is not None
                                            ]
                                            if not _gc_prs:
                                                continue
                                            _gc_combined = _combine_multi_system_power_results_device(
                                                cp, _gc_prs,
                                                n_skycells_s1586=n_skycells_s1586,
                                                boresight_active=boresight_active,
                                            )
                                            if _gc_combined is None:
                                                continue
                                            _gc_dist = _gc["distribution_collectors"]
                                            if "epfd_distribution" in _gc_dist and "EPFD_W_m2" in _gc_combined:
                                                _accumulate_1d_distribution_batch(
                                                    session, _gc_dist["epfd_distribution"],
                                                    value_linear=_gc_combined["EPFD_W_m2"],
                                                    db_offset_db=0.0, batch_index=bi,
                                                )
                                            if "total_pfd_ras_distribution" in _gc_dist and "PFD_total_RAS_STATION_W_m2" in _gc_combined:
                                                _accumulate_1d_distribution_batch(
                                                    session, _gc_dist["total_pfd_ras_distribution"],
                                                    value_linear=_gc_combined["PFD_total_RAS_STATION_W_m2"],
                                                    db_offset_db=0.0, batch_index=bi,
                                                )
                                            if "prx_total_distribution" in _gc_dist and "Prx_total_W" in _gc_combined:
                                                _accumulate_1d_distribution_batch(
                                                    session, _gc_dist["prx_total_distribution"],
                                                    value_linear=_gc_combined["Prx_total_W"],
                                                    db_offset_db=0.0, batch_index=bi,
                                                )
                                            del _gc_combined

                                        # Combine power from all systems
                                        # Stash per-system CPU copies for raw HDF5 write
                                        _ms_batch_stash.clear()
                                        for _msi, _ms_pr in enumerate(_ms_power_results):
                                            if _ms_pr is None:
                                                _ms_batch_stash.append({})
                                                continue
                                            _stash: dict[str, Any] = {}
                                            _ms_sys_sat_idx = _ms_pr.get("_sat_idx_g")
                                            if _ms_sys_sat_idx is None and _msi == 0:
                                                _ms_sys_sat_idx = sat_idx_g  # primary system uses main sat_idx_g
                                            _ms_sys_n_sats = int(
                                                _multi_system_contexts[_msi].get("n_sats_total", n_sats_total)
                                            ) if _msi < len(_multi_system_contexts) else n_sats_total
                                            _ms_sidx_host = (
                                                cp.asnumpy(_ms_sys_sat_idx).astype(np.int32) if _ms_sys_sat_idx is not None
                                                else None
                                            )
                                            for _sk in (
                                                "EPFD_W_m2", "PFD_total_RAS_STATION_W_m2", "Prx_total_W",
                                            ):
                                                if _sk in _ms_pr and _ms_pr[_sk] is not None:
                                                    _stash[_sk] = cp.asnumpy(_ms_pr[_sk])
                                            # Per-satellite power arrays: only write if output plan requests raw.
                                            # Collapse to 2D for non-boresight, pad to full system sat count.
                                            _ps_raw_keys = []
                                            if write_per_satellite_pfd_ras_station:
                                                _ps_raw_keys.append("PFD_per_sat_RAS_STATION_W_m2")
                                            if write_per_satellite_prx_ras_station:
                                                _ps_raw_keys.append("Prx_per_sat_RAS_STATION_W")
                                            # Use per-system boresight flag for collapse decision
                                            _ms_sys_boresight = bool(
                                                _multi_system_contexts[_msi].get("boresight_active", False)
                                            ) if _msi < len(_multi_system_contexts) else boresight_active
                                            if _ms_sidx_host is not None:
                                                for _sk in _ps_raw_keys:
                                                    if _sk in _ms_pr and _ms_pr[_sk] is not None:
                                                        _psat = cp.asnumpy(_ms_pr[_sk]).astype(np.float32)
                                                        if not _ms_sys_boresight:
                                                            _psat = _collapse_per_sat_to_2d(_psat)
                                                        _stash[_sk] = _pad_visible_to_full(
                                                            _psat,
                                                            _ms_sidx_host, _ms_sys_n_sats, fill=0.0,
                                                        )
                                                # Beam counts and elevation also need sat_idx for padding
                                                if _ms_sidx_host is not None:
                                                    _inj_beam = _ms_pr.get("_sat_beam_counts_used")
                                                    if _inj_beam is not None:
                                                        _stash["sat_beam_counts_used"] = _pad_visible_to_full(
                                                            np.asarray(
                                                                cp.asnumpy(_inj_beam) if hasattr(_inj_beam, 'get') else _inj_beam,
                                                                dtype=np.int32,
                                                            ),
                                                            _ms_sidx_host, _ms_sys_n_sats, fill=0,
                                                        )
                                                    elif _msi == 0 and sat_beam_counts_used_full is not None:
                                                        _stash["sat_beam_counts_used"] = _pad_visible_to_full(
                                                            np.asarray(
                                                                cp.asnumpy(sat_beam_counts_used_full) if hasattr(sat_beam_counts_used_full, 'get') else sat_beam_counts_used_full,
                                                                dtype=np.int32,
                                                            ),
                                                            _ms_sidx_host, _ms_sys_n_sats, fill=0,
                                                        )
                                                    _inj_elev = _ms_pr.get("_sat_elevation_deg")
                                                    if _inj_elev is not None:
                                                        _stash["sat_elevation_RAS_STATION_deg"] = _pad_visible_to_full(
                                                            np.asarray(
                                                                cp.asnumpy(_inj_elev) if hasattr(_inj_elev, 'get') else _inj_elev,
                                                                dtype=np.float32,
                                                            ),
                                                            _ms_sidx_host, _ms_sys_n_sats, fill=np.nan,
                                                        )
                                                    elif _msi == 0 and sat_topo is not None:
                                                        _stash["sat_elevation_RAS_STATION_deg"] = _pad_visible_to_full(
                                                            np.asarray(
                                                                cp.asnumpy(sat_topo[:, :, 1]) if hasattr(sat_topo, 'get') else sat_topo[:, :, 1],
                                                                dtype=np.float32,
                                                        ),
                                                        _ms_sidx_host, _ms_sys_n_sats, fill=np.nan,
                                                    )
                                            # Beam demand count (injected as _beam_demand_count)
                                            _inj_demand = _ms_pr.get("_beam_demand_count") if _ms_pr is not None else None
                                            if _inj_demand is not None:
                                                _stash["beam_demand_count"] = np.asarray(
                                                    gpu_module.copy_device_to_host(_inj_demand)
                                                    if hasattr(_inj_demand, 'get')
                                                    else np.asarray(_inj_demand),
                                                    dtype=np.int64,
                                                )
                                            elif _msi == 0 and beam_demand_count_dev is not None:
                                                # System 0 fallback: use the primary demand
                                                _stash["beam_demand_count"] = np.asarray(
                                                    gpu_module.copy_device_to_host(beam_demand_count_dev),
                                                    dtype=np.int64,
                                                )
                                            _ms_batch_stash.append(_stash)

                                        power_result = _combine_multi_system_power_results_device(
                                            cp,
                                            _ms_power_results,
                                            n_skycells_s1586=n_skycells_s1586,
                                            boresight_active=boresight_active,
                                        )
                                        del _ms_power_results
                                    # --- End multi-system interleaving ---
                                    if isinstance(
                                        compute_payload.get("beam_finalize_substage_timings"),
                                        Mapping,
                                    ):
                                        for name, value in dict(
                                            compute_payload["beam_finalize_substage_timings"]
                                        ).items():
                                            try:
                                                beam_finalize_substage_timings_summary[str(name)] = float(
                                                    beam_finalize_substage_timings_summary.get(str(name), 0.0)
                                                ) + float(value)
                                            except Exception:
                                                continue
                                    if isinstance(
                                        compute_payload.get("cell_link_library_chunk_telemetry"),
                                        Mapping,
                                    ):
                                        for name, value in dict(
                                            compute_payload["cell_link_library_chunk_telemetry"]
                                        ).items():
                                            try:
                                                numeric_value = int(value)
                                            except Exception:
                                                continue
                                            cell_link_library_chunk_telemetry_summary[str(name)] = int(
                                                max(
                                                    int(
                                                        cell_link_library_chunk_telemetry_summary.get(
                                                            str(name),
                                                            0,
                                                        )
                                                    ),
                                                    numeric_value,
                                                )
                                            )
                                    for summary_key in (
                                        "cell_link_library_stage_memory_summary",
                                        "beam_finalize_stage_memory_summary",
                                        "power_stage_memory_summary",
                                    ):
                                        summary_value = compute_payload.get(summary_key)
                                        if not isinstance(summary_value, Mapping):
                                            continue
                                        stage_name = str(
                                            summary_value.get("observed_stage_name")
                                            or summary_key.replace("_stage_memory_summary", "")
                                        )
                                        observed_stage_memory_summary_by_name[stage_name] = (
                                            _merge_direct_epfd_stage_memory_summaries(
                                                observed_stage_memory_summary_by_name.get(stage_name),
                                                summary_value,
                                            )
                                        )
                                    if isinstance(
                                        compute_payload.get("beam_finalize_chunk_shape"),
                                        Mapping,
                                    ):
                                        beam_finalize_chunk_shape_summary = dict(
                                            compute_payload["beam_finalize_chunk_shape"]
                                        )
                                        if isinstance(
                                            compute_payload.get("boresight_compaction_stats"),
                                            Mapping,
                                        ):
                                            boresight_compaction_stats_summary = _merge_run_max_numeric_mapping(
                                                boresight_compaction_stats_summary,
                                                compute_payload["boresight_compaction_stats"],
                                            )
                                    if debug_direct_epfd:
                                        for slab_stats in compute_payload.get("debug_direct_epfd_stats", []):
                                            slab_stats_out = dict(slab_stats)
                                            slab_stats_out["iteration"] = int(ii)
                                            slab_stats_out["batch_index"] = int(bi)
                                            debug_direct_epfd_stats_all.append(slab_stats_out)
                                    if isinstance(compute_payload.get("stage_memory_summary"), Mapping):
                                        scheduler_runtime_state["last_observed_stage_summary"] = dict(
                                            compute_payload["stage_memory_summary"]
                                        )
                                    hot_path_device_to_host_copy_count_summary += int(
                                        compute_payload.get("hot_path_device_to_host_copy_count", 0) or 0
                                    )
                                    hot_path_device_to_host_copy_bytes_summary += int(
                                        compute_payload.get("hot_path_device_to_host_copy_bytes", 0) or 0
                                    )
                                    device_scalar_sync_count_summary += int(
                                        compute_payload.get("device_scalar_sync_count", 0) or 0
                                    )
                                    stage_start = compute_payload["stage_start"]
                                    run_active_target_fraction = min(
                                        float(run_active_target_fraction),
                                        float(batch_compute_active_target_fraction),
                                    )
                                    if retry_count > 0:
                                        current_batch_plan = dict(batch_plan_for_retry)
                                        iteration_plan = _plan_direct_epfd_iteration_schedule(
                                            session=session,
                                            observer_context=observer_context,
                                            satellite_context=satellite_context,
                                            gpu_output_dtype=gpu_output_dtype,
                                            n_steps_total=int(n_steps_total),
                                            n_cells_total=n_cells_total,
                                            n_sats_total=n_sats_total,
                                            n_skycells_s1586=n_skycells_s1586,
                                            visible_satellite_est=int(visible_satellite_est),
                                            nco=int(nco),
                                            nbeam=int(nbeam),
                                            boresight_active=boresight_active,
                                            effective_ras_pointing_mode=effective_ras_pointing_mode,
                                            output_family_plan=output_family_plan,
                                            store_eligible_mask=bool(store_eligible_mask),
                                            profile_stages=bool(profile_stages),
                                            host_budget_info=host_budget_info,
                                            gpu_budget_info=gpu_budget_info,
                                            scheduler_target_fraction=float(scheduler_target_fraction),
                                            scheduler_active_target_fraction=float(run_active_target_fraction),
                                            finalize_memory_budget_bytes=finalize_memory_budget_bytes,
                                            power_memory_budget_bytes=power_memory_budget_bytes,
                                            export_memory_budget_bytes=export_memory_budget_bytes,
                                            power_sky_slab=power_sky_slab,
                                            force_bulk_timesteps=int(bulk_timesteps),
                                            force_cell_observer_chunk=force_cell_observer_chunk,
                                            spectrum_context_bytes=int(spectrum_context_bytes),
                                            cell_activity_mode=str(normalized_cell_activity_mode),
                                            activity_groups_per_cell=int(activity_groups_per_cell),
                                            activity_power_policy=str(activity_power_policy),
                                            activity_split_total_group_denominator_mode=str(
                                                activity_split_total_group_denominator_mode
                                            ),
                                            allow_warmup_calibration=False,
                                            surface_pfd_cap_enabled=bool(max_surface_pfd_enabled),
                                            surface_pfd_cap_mode=str(surface_pfd_cap_mode),
                                        )
                                        scheduler_runtime_state = _update_scheduler_runtime_state_from_plan(
                                            scheduler_runtime_state,
                                            current_batch_plan,
                                            spectrum_context_bytes=int(spectrum_context_bytes),
                                        )
                                        scheduler_runtime_state = _update_scheduler_runtime_state_from_plan(
                                            scheduler_runtime_state,
                                            iteration_plan,
                                            spectrum_context_bytes=int(spectrum_context_bytes),
                                        )
                                        iter_cell_chunk_plan = int(iteration_plan["cell_chunk"])
                                        planned_batch_seconds = float(
                                            iteration_plan.get(
                                                "planned_batch_seconds", planned_batch_seconds
                                            )
                                        )
                                        planned_iteration_seconds = float(planned_batch_seconds) * float(
                                            n_batches
                                        )
                                        remaining_iterations = max(
                                            0, int(iteration_count) - int(ii)
                                        )
                                        planned_total_seconds = (
                                            float(planned_iteration_seconds)
                                            * float(int(iteration_count))
                                        )
                                        planned_remaining_seconds = (
                                            float(planned_batch_seconds)
                                            * float(max(1, n_batches - int(bi)))
                                            + float(planned_iteration_seconds)
                                            * float(max(0, remaining_iterations - 1))
                                        )
                                    stage_budget_info = dict(iteration_plan["stage_budget_info"])
                                    planned_power_sky_slab = int(iteration_plan["sky_slab"])
                                    break
                                except _DirectGpuOutOfMemory as oom_exc:
                                    retry_count += 1
                                    if isinstance(oom_exc.stage_memory_summary, Mapping) and oom_exc.stage_memory_summary:
                                        scheduler_runtime_state["last_observed_stage_summary"] = dict(
                                            oom_exc.stage_memory_summary
                                        )
                                    _reset_direct_epfd_gpu_pools(cp, session)
                                    post_cleanup_snapshot = _capture_direct_epfd_live_memory_snapshot(
                                        cp,
                                        session,
                                    )
                                    scheduler_runtime_state, cap_lowered = _lower_runtime_effective_gpu_budget(
                                        scheduler_runtime_state,
                                        stage=str(oom_exc.stage),
                                        post_cleanup_snapshot=post_cleanup_snapshot,
                                    )
                                    scheduler_runtime_state["scheduler_retry_count"] = int(
                                        scheduler_runtime_state.get("scheduler_retry_count", 0)
                                    ) + 1
                                    gpu_budget_info = _apply_effective_budget_override(
                                        gpu_budget_info_raw,
                                        effective_budget_bytes=(
                                            None
                                            if scheduler_runtime_state.get("gpu_effective_budget_bytes") is None
                                            else int(
                                                scheduler_runtime_state["gpu_effective_budget_bytes"]
                                            )
                                        ),
                                    )
                                    next_target_fraction = float(batch_compute_active_target_fraction)
                                    if retry_targets:
                                        next_target_fraction = float(retry_targets.pop(0))
                                    if profile_stages:
                                        stage_timings = {}
                                        stage_start = perf_counter()
                                    batch_plan_for_retry = _plan_direct_epfd_iteration_schedule(
                                        session=session,
                                        observer_context=observer_context,
                                        satellite_context=satellite_context,
                                        gpu_output_dtype=gpu_output_dtype,
                                        n_steps_total=int(time_count_local),
                                        n_cells_total=n_cells_total,
                                        n_sats_total=n_sats_total,
                                        n_skycells_s1586=n_skycells_s1586,
                                        visible_satellite_est=visible_satellite_est,
                                        nco=int(nco),
                                        nbeam=int(nbeam),
                                        boresight_active=boresight_active,
                                        effective_ras_pointing_mode=effective_ras_pointing_mode,
                                        output_family_plan=output_family_plan,
                                        store_eligible_mask=bool(store_eligible_mask),
                                        profile_stages=bool(profile_stages),
                                        host_budget_info=host_budget_info,
                                        gpu_budget_info=gpu_budget_info,
                                        scheduler_target_fraction=float(scheduler_target_fraction),
                                        scheduler_active_target_fraction=float(next_target_fraction),
                                        finalize_memory_budget_bytes=finalize_memory_budget_bytes,
                                        power_memory_budget_bytes=power_memory_budget_bytes,
                                        export_memory_budget_bytes=export_memory_budget_bytes,
                                        power_sky_slab=power_sky_slab,
                                        force_bulk_timesteps=int(time_count_local),
                                        force_cell_observer_chunk=force_cell_observer_chunk,
                                        spectrum_context_bytes=int(spectrum_context_bytes),
                                        cell_activity_mode=str(normalized_cell_activity_mode),
                                        activity_groups_per_cell=int(activity_groups_per_cell),
                                        activity_power_policy=str(activity_power_policy),
                                        activity_split_total_group_denominator_mode=str(
                                            activity_split_total_group_denominator_mode
                                        ),
                                        allow_warmup_calibration=False,
                                        surface_pfd_cap_enabled=bool(max_surface_pfd_enabled),
                                        surface_pfd_cap_mode=str(surface_pfd_cap_mode),
                                    )
                                    scheduler_runtime_state = _update_scheduler_runtime_state_from_plan(
                                        scheduler_runtime_state,
                                        batch_plan_for_retry,
                                        spectrum_context_bytes=int(spectrum_context_bytes),
                                    )
                                    batch_stage_budget_info = dict(
                                        batch_plan_for_retry["stage_budget_info"]
                                    )
                                    batch_power_sky_slab = int(batch_plan_for_retry["sky_slab"])
                                    cell_chunk = max(
                                        1, min(int(batch_plan_for_retry["cell_chunk"]), n_cells_total)
                                    )
                                    n_cell_chunks = int(np.ceil(n_cells_total / cell_chunk))
                                    retry_signature_after = (
                                        int(gpu_budget_info["effective_budget_bytes"]),
                                        float(next_target_fraction),
                                        int(cell_chunk),
                                        int(batch_power_sky_slab),
                                        int(batch_plan_for_retry.get("spectral_slab", 1)),
                                    )
                                    no_progress = retry_signature_after == retry_signature_before
                                    min_safe_shape = (
                                        int(cell_chunk) <= 1
                                        and int(batch_power_sky_slab) <= 1
                                        and int(batch_plan_for_retry.get("spectral_slab", 1)) <= 1
                                        and abs(
                                            float(next_target_fraction)
                                            - float(batch_compute_active_target_fraction)
                                        )
                                        <= 1.0e-9
                                        and not bool(cap_lowered)
                                    )
                                    if no_progress or min_safe_shape:
                                        stage_summary = dict(oom_exc.stage_memory_summary or {})
                                        configured_gpu_cap_bytes = int(
                                            gpu_budget_info_raw.get(
                                                "hard_budget_bytes",
                                                gpu_budget_info_raw["effective_budget_bytes"],
                                            )
                                        )
                                        detail_bits = [
                                            f"configured cap {format_byte_count(configured_gpu_cap_bytes)}",
                                            f"effective cap {format_byte_count(gpu_budget_info['effective_budget_bytes'])}",
                                            f"target={float(next_target_fraction) * 100.0:.0f}%",
                                            f"cell_chunk={int(cell_chunk)}",
                                            f"sky_slab={int(batch_power_sky_slab)}",
                                            f"spectral_slab={int(batch_plan_for_retry.get('spectral_slab', 1))}",
                                        ]
                                        if stage_summary.get("observed_stage_gpu_peak_bytes") is not None:
                                            detail_bits.append(
                                                "observed_peak="
                                                f"{format_byte_count(stage_summary['observed_stage_gpu_peak_bytes'])}"
                                            )
                                        if stage_summary.get("observed_stage_gpu_free_low_bytes") is not None:
                                            detail_bits.append(
                                                "free_low="
                                                f"{format_byte_count(stage_summary['observed_stage_gpu_free_low_bytes'])}"
                                            )
                                        raise _DirectGpuOutOfMemory(
                                            str(oom_exc.stage),
                                            RuntimeError(
                                                "GPU memory remained unsafe after scheduler replanning; "
                                                + ", ".join(detail_bits)
                                            ),
                                            stage_memory_summary=stage_summary,
                                        ) from oom_exc
                                    batch_compute_active_target_fraction = float(next_target_fraction)
                                    retry_payload = _build_direct_epfd_scheduler_payload(
                                        host_budget_info=host_budget_info,
                                        gpu_budget_info=gpu_budget_info,
                                        scheduler_target_fraction=float(scheduler_target_fraction),
                                        scheduler_active_target_fraction=float(
                                            batch_compute_active_target_fraction
                                        ),
                                        boresight_active=bool(boresight_active),
                                        n_earthgrid_cells=int(n_cells_total),
                                        n_skycells_s1586=int(n_skycells_s1586),
                                        visible_satellite_est=int(visible_satellite_est),
                                        bulk_timesteps=int(time_count_local),
                                        cell_chunk=int(cell_chunk),
                                        sky_slab=int(batch_power_sky_slab),
                                        predicted_host_peak_bytes=int(
                                            batch_plan_for_retry["predicted_host_peak_bytes"]
                                        ),
                                        predicted_gpu_peak_bytes=int(
                                            batch_plan_for_retry["predicted_gpu_peak_bytes"]
                                        ),
                                        planner_source=str(batch_plan_for_retry["planner_source"]),
                                        limiting_resource=str(
                                            batch_plan_for_retry["limiting_resource"]
                                        ),
                                        planned_total_seconds=float(planned_total_seconds),
                                        planned_remaining_seconds=float(
                                            batch_planned_remaining_seconds
                                        ),
                                        live_host_snapshot=_runtime_host_memory_snapshot(),
                                        live_gpu_snapshot=_runtime_gpu_memory_snapshot(cp, session),
                                        live_gpu_adapter_snapshot=_runtime_gpu_adapter_memory_snapshot(cp),
                                        extra={
                                            "warning_stage": str(oom_exc.stage),
                                            "warning_retry_count": int(retry_count),
                                            "scheduler_target_profile": str(
                                                scheduler_target_name
                                            ),
                                            **_scheduler_runtime_state_extra(scheduler_runtime_state),
                                        },
                                    )
                                    warning_bits = []
                                    previous_effective = scheduler_runtime_state.get(
                                        "gpu_effective_budget_previous_bytes"
                                    )
                                    current_effective = scheduler_runtime_state.get(
                                        "gpu_effective_budget_bytes"
                                    )
                                    if (
                                        previous_effective is not None
                                        and current_effective is not None
                                        and int(current_effective) < int(previous_effective)
                                    ):
                                        configured_gpu_cap_bytes = int(
                                            gpu_budget_info_raw.get(
                                                "hard_budget_bytes",
                                                gpu_budget_info_raw["effective_budget_bytes"],
                                            )
                                        )
                                        warning_bits.append(
                                            "configured "
                                            f"{format_byte_count(configured_gpu_cap_bytes)} -> "
                                            f"effective {format_byte_count(current_effective)} for this run"
                                        )
                                    _emit_direct_epfd_progress(
                                        progress_callback,
                                        kind="warning",
                                        phase="compute",
                                        iteration_index=int(ii),
                                        iteration_total=int(iteration_count),
                                        batch_index=int(bi),
                                        batch_total=int(n_batches),
                                        description=(
                                            f"{str(oom_exc.original_exception)}; "
                                            + (
                                                f"{warning_bits[0]}; " if warning_bits else ""
                                            )
                                            + f"retrying batch {int(bi) + 1}/{int(n_batches)} at "
                                            f"{float(batch_compute_active_target_fraction) * 100.0:.0f}% target."
                                        ),
                                        **retry_payload,
                                    )
                        except _RunCancellationRequested as cancel_exc:
                            run_state = "stopped"
                            stop_mode = str(cancel_exc.mode)
                            stop_boundary = str(cancel_exc.boundary)
                            if not stop_notice_emitted:
                                _emit_direct_epfd_progress(
                                    progress_callback,
                                    kind="phase",
                                    phase="stopping",
                                    iteration_index=int(ii),
                                    iteration_total=int(iteration_count),
                                    batch_index=int(bi),
                                    batch_total=int(n_batches),
                                    stop_mode=str(stop_mode),
                                    stop_boundary=str(stop_boundary),
                                    description=(
                                        "Force stop requested. Halting before the next batch is written."
                                        if stop_mode == "force"
                                        else "Stop requested."
                                    ),
                                )
                                stop_notice_emitted = True
                            break
                    if not need_beam_finalize:
                        power_result = None
                        sat_idx_g = cp.nonzero(sat_keep_batch)[0].astype(cp.int32, copy=False)
                        sat_topo = sat_topo_ras_station[:, sat_idx_g, :]
                        sat_azel = sat_azel_ras_station[:, sat_idx_g, :]
                        orbit_radius_eff = orbit_radius_full[sat_idx_g]
                        sat_beam_counts_used_full = None
                        sat_eligible_mask = None
                        diag_result = None
                    del ras_result
                    sat_idx_host: np.ndarray | None = None
                    export_copy_t0 = perf_counter() if profile_stages else None
                    export_stage_summary = _start_direct_epfd_stage_memory_summary(
                        "export_copy",
                        cp=cp,
                        session=session,
                    )
                    _set_direct_epfd_progress_phase(
                        pbar,
                        enable_progress_bars=enable_progress_bars,
                        progress_desc_mode=progress_desc_mode_name,
                        phase="export_copy",
                    )
                    _emit_direct_epfd_progress(
                        progress_callback,
                        kind="phase",
                        phase="export_copy",
                        iteration_index=int(ii),
                        iteration_total=int(iteration_count),
                        batch_index=int(bi),
                        batch_total=int(n_batches),
                        description=_direct_epfd_progress_text(progress_desc_mode_name, "export_copy"),
                    )

                    if any_power_outputs and power_result is not None:
                        if output_family_plan["preacc_prx_total_distribution"]:
                            collector = distribution_collectors["prx_total_distribution"]
                            _accumulate_1d_distribution_batch(
                                session, collector,
                                value_linear=power_result["Prx_total_W"],
                                db_offset_db=0.0,
                                batch_index=bi,
                            )

                        if output_family_plan["preacc_epfd_distribution"]:
                            _accumulate_1d_distribution_batch(
                                session, distribution_collectors["epfd_distribution"],
                                value_linear=power_result["EPFD_W_m2"],
                                db_offset_db=0.0,
                                batch_index=bi,
                                )

                        if output_family_plan["preacc_total_pfd_ras_distribution"]:
                            _accumulate_1d_distribution_batch(
                                session, distribution_collectors["total_pfd_ras_distribution"],
                                value_linear=power_result["PFD_total_RAS_STATION_W_m2"],
                                db_offset_db=0.0,
                                batch_index=bi,
                            )

                        if output_family_plan["preacc_per_satellite_pfd_distribution"]:
                            _accumulate_1d_distribution_batch(
                                session, distribution_collectors["per_satellite_pfd_distribution"],
                                value_linear=power_result["PFD_per_sat_RAS_STATION_W_m2"],
                                db_offset_db=0.0,
                                batch_index=bi,
                                )

                        if output_family_plan["preacc_prx_elevation_heatmap"]:
                            _accumulate_heatmap_batch(
                                session,
                                heatmap_collectors["prx_elevation_heatmap"],
                                value_linear=power_result["Prx_per_sat_RAS_STATION_W"],
                                sat_elevation_deg=sat_topo[:, :, 1],
                                db_offset_db=0.0,
                            )

                        if output_family_plan["preacc_per_satellite_pfd_elevation_heatmap"]:
                            _accumulate_heatmap_batch(
                                session,
                                heatmap_collectors["per_satellite_pfd_elevation_heatmap"],
                                value_linear=power_result["PFD_per_sat_RAS_STATION_W_m2"],
                                sat_elevation_deg=sat_topo[:, :, 1],
                                db_offset_db=0.0,
                            )

                        if write_epfd:
                            batch_payload["EPFD_W_m2"] = gpu_module.copy_device_to_host(
                                power_result["EPFD_W_m2"]
                            )
                        if write_prx_total:
                            batch_payload["Prx_total_W"] = gpu_module.copy_device_to_host(
                                power_result["Prx_total_W"]
                            )
                        if write_per_satellite_prx_ras_station:
                            if sat_idx_host is None:
                                sat_idx_host = _copy_compact_satellite_indices_host(
                                    gpu_module,
                                    sat_idx_g,
                                )
                            _prx_ps = gpu_module.copy_device_to_host(
                                power_result["Prx_per_sat_RAS_STATION_W"]
                            )
                            if not boresight_active:
                                _prx_ps = _collapse_per_sat_to_2d(_prx_ps)
                            batch_payload["Prx_per_sat_RAS_STATION_W"] = _pad_visible_to_full(
                                _prx_ps,
                                np.asarray(sat_idx_host, dtype=np.int32),
                                int(n_sats_total), fill=0.0,
                            )
                        if write_total_pfd_ras_station:
                            batch_payload["PFD_total_RAS_STATION_W_m2"] = gpu_module.copy_device_to_host(
                                power_result["PFD_total_RAS_STATION_W_m2"]
                            )
                        if write_per_satellite_pfd_ras_station:
                            if sat_idx_host is None:
                                sat_idx_host = _copy_compact_satellite_indices_host(
                                    gpu_module,
                                    sat_idx_g,
                                )
                            _pfd_ps = gpu_module.copy_device_to_host(
                                power_result["PFD_per_sat_RAS_STATION_W_m2"]
                            )
                            if not boresight_active:
                                _pfd_ps = _collapse_per_sat_to_2d(_pfd_ps)
                            batch_payload["PFD_per_sat_RAS_STATION_W_m2"] = _pad_visible_to_full(
                                _pfd_ps,
                                np.asarray(sat_idx_host, dtype=np.int32),
                                int(n_sats_total), fill=0.0,
                            )

                        # Surface-PFD cap statistics: store as 1-D length-1
                        # arrays so the writer concatenates them along the
                        # batch axis, giving a ``(n_batches,)`` time series
                        # per stat.  Only populated by the power kernel
                        # when ``surface_pfd_stats_enabled`` was passed on.
                        if "surface_pfd_cap_n_beams_capped" in power_result:
                            batch_payload["surface_pfd_cap_n_beams_capped"] = np.asarray(
                                gpu_module.copy_device_to_host(
                                    power_result["surface_pfd_cap_n_beams_capped"]
                                ),
                                dtype=np.int64,
                            ).reshape(1)
                        if "surface_pfd_cap_mean_cap_db" in power_result:
                            batch_payload["surface_pfd_cap_mean_cap_db"] = np.asarray(
                                gpu_module.copy_device_to_host(
                                    power_result["surface_pfd_cap_mean_cap_db"]
                                ),
                                dtype=np.float32,
                            ).reshape(1)
                        if "surface_pfd_cap_max_cap_db" in power_result:
                            batch_payload["surface_pfd_cap_max_cap_db"] = np.asarray(
                                gpu_module.copy_device_to_host(
                                    power_result["surface_pfd_cap_max_cap_db"]
                                ),
                                dtype=np.float32,
                            ).reshape(1)

                    if beam_stats_collector is not None and sat_beam_counts_used_full is not None:
                        counts_full_dev = cp.asarray(sat_beam_counts_used_full, dtype=np.int64)

                        # Keep the conservative per-satellite collapse only for histogram / CCDF
                        # style statistics.
                        counts_hist_samples_dev = _beam_count_samples_device(cp, counts_full_dev)
                        full_hist_batch = _bincount_device_to_host(
                            cp,
                            gpu_module,
                            counts_hist_samples_dev.reshape(-1),
                        )
                        beam_stats_collector["full_network_count_hist"] = _merge_count_histograms(
                            beam_stats_collector["full_network_count_hist"],
                            full_hist_batch,
                        )

                        # For beam-over-time totals, never do max-per-satellite and then sum.
                        network_total_dev = _beam_total_over_time_device(cp, counts_full_dev)

                        visibility_mask_full_dev = cp.asarray(
                            sat_topo_ras_station[..., 1] > cp.float32(visibility_elev_threshold_deg),
                            dtype=bool,
                        )

                        visible_hist_batch, _visible_hist_total_dev = _visible_beam_statistics_device(
                            cp,
                            gpu_module,
                            counts_samples_device=counts_hist_samples_dev,
                            visibility_mask_device=visibility_mask_full_dev,
                        )
                        visible_total_dev = _visible_beam_total_over_time_device(
                            cp,
                            counts_device=counts_full_dev,
                            visibility_mask_device=visibility_mask_full_dev,
                        )

                        beam_stats_collector["visible_count_hist"] = _merge_count_histograms(
                            beam_stats_collector["visible_count_hist"],
                            visible_hist_batch,
                        )

                        # For multi-system: sum per-system beam totals for the combined view
                        if _multi_system_active and _per_system_beam_stats_collectors:
                            _combined_net = gpu_module.copy_device_to_host(network_total_dev)
                            _combined_vis = gpu_module.copy_device_to_host(visible_total_dev)
                            for _ps_bsc in _per_system_beam_stats_collectors[1:]:
                                if _ps_bsc is None:
                                    continue
                                _ps_net = _ps_bsc["network_total_beams_over_time"]
                                _ps_vis = _ps_bsc["visible_total_beams_over_time"]
                                if _ps_net and len(_ps_net) > 0:
                                    _last_net = np.asarray(_ps_net[-1], dtype=np.int64)
                                    if _last_net.shape == _combined_net.shape:
                                        _combined_net = _combined_net + _last_net
                                if _ps_vis and len(_ps_vis) > 0:
                                    _last_vis = np.asarray(_ps_vis[-1], dtype=np.int64)
                                    if _last_vis.shape == _combined_vis.shape:
                                        _combined_vis = _combined_vis + _last_vis
                            _append_series_segment(
                                beam_stats_collector["network_total_beams_over_time"],
                                _combined_net,
                            )
                            _append_series_segment(
                                beam_stats_collector["visible_total_beams_over_time"],
                                _combined_vis,
                            )
                        else:
                            _append_series_segment(
                                beam_stats_collector["network_total_beams_over_time"],
                                gpu_module.copy_device_to_host(network_total_dev),
                            )
                            _append_series_segment(
                                beam_stats_collector["visible_total_beams_over_time"],
                                gpu_module.copy_device_to_host(visible_total_dev),
                            )

                        # Combined beam demand: sum across all systems
                        if _multi_system_active and _per_system_beam_stats_collectors:
                            _combined_demand = (
                                gpu_module.copy_device_to_host(beam_demand_count_dev)
                                if beam_demand_count_dev is not None
                                else np.zeros((time_count_local,), dtype=np.int64)
                            )
                            for _ps_bsc in _per_system_beam_stats_collectors[1:]:
                                if _ps_bsc is None:
                                    continue
                                _ps_dem = _ps_bsc["beam_demand_over_time"]
                                if _ps_dem and len(_ps_dem) > 0:
                                    _last_dem = np.asarray(_ps_dem[-1], dtype=np.int64)
                                    if _last_dem.shape == _combined_demand.shape:
                                        _combined_demand = _combined_demand + _last_dem
                            _append_series_segment(
                                beam_stats_collector["beam_demand_over_time"],
                                _combined_demand,
                            )
                        elif beam_demand_count_dev is not None:
                            _append_series_segment(
                                beam_stats_collector["beam_demand_over_time"],
                                gpu_module.copy_device_to_host(beam_demand_count_dev),
                            )

                    if write_sat_beam_counts_used:
                        counts_compact = cp.asarray(
                            sat_beam_counts_used_full,
                            dtype=count_dtype,
                        )
                        if boresight_active:
                            counts_compact = cp.transpose(counts_compact, (0, 2, 1))[:, None, :, :]
                        if sat_idx_host is None:
                            sat_idx_host = _copy_compact_satellite_indices_host(
                                gpu_module,
                                sat_idx_g,
                            )
                        batch_payload["sat_beam_counts_used"] = _pad_visible_to_full(
                            gpu_module.copy_device_to_host(counts_compact),
                            np.asarray(sat_idx_host, dtype=np.int32),
                            int(n_sats_total), fill=0,
                        )
                    if write_sat_elevation_ras_station:
                        sat_elevation_visible = cp.asarray(
                            sat_topo[:, :, 1],
                            dtype=cp.float32,
                        )
                        sat_elevation_visible = cp.where(
                            sat_elevation_visible > cp.float32(visibility_elev_threshold_deg),
                            sat_elevation_visible,
                            cp.float32(np.nan),
                        )
                        if sat_idx_host is None:
                            sat_idx_host = _copy_compact_satellite_indices_host(
                                gpu_module,
                                sat_idx_g,
                            )
                        batch_payload["sat_elevation_RAS_STATION_deg"] = _pad_visible_to_full(
                            gpu_module.copy_device_to_host(sat_elevation_visible),
                            np.asarray(sat_idx_host, dtype=np.int32),
                            int(n_sats_total), fill=np.nan,
                        )
                    if bool(store_eligible_mask) and sat_eligible_mask is not None:
                        batch_payload["sat_eligible_mask"] = np.asarray(
                            sat_eligible_mask,
                            dtype=np.bool_,
                        )

                    export_stage_summary = _update_direct_epfd_stage_memory_summary(
                        export_stage_summary,
                        _capture_direct_epfd_live_memory_snapshot(cp, session),
                    )
                    scheduler_runtime_state["last_observed_stage_summary"] = dict(
                        export_stage_summary
                    )
                    if profile_stages and stage_timings is not None and export_copy_t0 is not None:
                        stage_timings["export_copy"] = perf_counter() - export_copy_t0

                    write_enqueue_stage_summary = _start_direct_epfd_stage_memory_summary(
                        "write_enqueue",
                        cp=cp,
                        session=session,
                    )
                    _set_direct_epfd_progress_phase(
                        pbar,
                        enable_progress_bars=enable_progress_bars,
                        progress_desc_mode=progress_desc_mode_name,
                        phase="write_enqueue",
                    )
                    _emit_direct_epfd_progress(
                        progress_callback,
                        kind="phase",
                        phase="write_enqueue",
                        iteration_index=int(ii),
                        iteration_total=int(iteration_count),
                        batch_index=int(bi),
                        batch_total=int(n_batches),
                        description=_direct_epfd_progress_text(
                            progress_desc_mode_name,
                            "write_enqueue",
                        ),
                    )
                    write_enqueue_t0 = perf_counter() if profile_stages else None
                    _write_iteration_batch_owned(
                        storage_filename,
                        iteration=ii,
                        batch_items=tuple(batch_payload.items()),
                        compression=hdf5_compression,
                        compression_opts=hdf5_compression_opts,
                        writer_queue_max_items=writer_queue_max_items,
                        writer_queue_max_bytes=writer_queue_max_bytes,
                    )
                    # Write per-system raw iteration data
                    if _multi_system_active and _ms_batch_stash:
                        for _ps_bi, _ps_stash in enumerate(_ms_batch_stash):
                            if _ps_stash:
                                _ps_batch = {"times": local_times.mjd}
                                _ps_batch.update(_ps_stash)
                                _write_iteration_batch_owned(
                                    storage_filename,
                                    iteration=ii,
                                    batch_items=tuple(_ps_batch.items()),
                                    compression=hdf5_compression,
                                    compression_opts=hdf5_compression_opts,
                                    writer_queue_max_items=writer_queue_max_items,
                                    writer_queue_max_bytes=writer_queue_max_bytes,
                                    group_prefix=f"system_{_ps_bi}/",
                                )
                        _ms_batch_stash.clear()
                    (
                        last_writer_checkpoint_monotonic,
                        checkpoint_wait_elapsed,
                        checkpoint_triggered,
                    ) = _maybe_checkpoint_writer_durable(
                        storage_filename,
                        checkpoint_interval_s=writer_checkpoint_interval_s_name,
                        last_checkpoint_monotonic=last_writer_checkpoint_monotonic,
                        pbar=pbar,
                        enable_progress_bars=enable_progress_bars,
                        progress_desc_mode=progress_desc_mode_name,
                    )
                    if checkpoint_triggered:
                        writer_checkpoint_count += 1
                        writer_checkpoint_wait_s += float(checkpoint_wait_elapsed)
                        _emit_direct_epfd_progress(
                            progress_callback,
                            kind="phase",
                            phase="checkpoint",
                            iteration_index=int(ii),
                            iteration_total=int(iteration_count),
                            batch_index=int(bi),
                            batch_total=int(n_batches),
                            elapsed_s=float(checkpoint_wait_elapsed),
                            checkpoint_count=int(writer_checkpoint_count),
                            description=_direct_epfd_progress_text(
                                progress_desc_mode_name,
                                "checkpoint",
                            ),
                        )
                    write_enqueue_stage_summary = _update_direct_epfd_stage_memory_summary(
                        write_enqueue_stage_summary,
                        _capture_direct_epfd_live_memory_snapshot(cp, session),
                    )
                    scheduler_runtime_state["last_observed_stage_summary"] = dict(
                        write_enqueue_stage_summary
                    )
                    if profile_stages and stage_timings is not None and write_enqueue_t0 is not None:
                        stage_timings["write_enqueue"] = perf_counter() - write_enqueue_t0
                        stage_timings["writer_checkpoint_wait"] = float(
                            checkpoint_wait_elapsed
                        )
                        profile_stage_timings_all.append(
                            {
                                "iteration": int(ii),
                                "batch_index": int(bi),
                                **{k: float(v) for k, v in stage_timings.items()},
                            }
                        )
                        for name, value in stage_timings.items():
                            profile_stage_timings_summary[name] = (
                                float(profile_stage_timings_summary.get(name, 0.0)) + float(value)
                            )
                        ras_retargeted_total = 0
                        repaired_total = 0
                        dropped_total = 0
                        if diag_result is not None:
                            ras_retargeted_total = int(
                                _scalar_from_device(cp.sum(cp.asarray(diag_result["ras_retargeted_count"])))
                            )
                            repaired_total = int(
                                _scalar_from_device(cp.sum(cp.asarray(diag_result["repaired_link_count"])))
                            )
                            dropped_total = int(
                                _scalar_from_device(cp.sum(cp.asarray(diag_result["dropped_link_count"])))
                            )
                        print(
                            f"[Direct EPFD gpu] iter={ii} batch={bi} visible_rows={visible_count} "
                            f"ras_retargeted={ras_retargeted_total} "
                            f"repaired={repaired_total} "
                            f"dropped={dropped_total} "
                            f"propagate={stage_timings.get('orbit_propagation', 0.0):.3f}s "
                            f"ras_geom={stage_timings.get('ras_geometry', 0.0):.3f}s "
                            f"library={stage_timings.get('cell_link_library', 0.0):.3f}s "
                            f"(derive={stage_timings.get('cll_derive_from_eci', 0.0):.3f}s "
                            f"add_chunk={stage_timings.get('cll_add_chunk', 0.0):.3f}s) "
                            f"finalize={stage_timings.get('beam_finalize', 0.0):.3f}s "
                            f"pointings={stage_timings.get('pointings', 0.0):.3f}s "
                            f"activity={stage_timings.get('cell_activity_setup', 0.0):.3f}s "
                            f"spec_weight={stage_timings.get('spectrum_activity_weighting', 0.0):.3f}s "
                            f"power={stage_timings.get('power_accumulation', 0.0):.3f}s "
                            f"export_copy={stage_timings.get('export_copy', 0.0):.3f}s "
                            f"write_enqueue={stage_timings.get('write_enqueue', 0.0):.3f}s"
                        )
                    if profile_stages and stage_timings:
                        _batch_wall = perf_counter() - _batch_loop_t0
                        _stage_sum = sum(float(v) for v in stage_timings.values())
                        print(f"[batch {bi}] wall={_batch_wall:.3f}s stages={_stage_sum:.3f}s overhead={_batch_wall - _stage_sum:.3f}s")
                    if scheduler_runtime_state.get("gpu_effective_budget_lowered"):
                        configured_gpu_cap_bytes = int(
                            gpu_budget_info_raw.get(
                                "hard_budget_bytes",
                                gpu_budget_info_raw["effective_budget_bytes"],
                            )
                        )
                        scheduler_runtime_state, _ = _recover_runtime_effective_gpu_budget(
                            scheduler_runtime_state,
                            hard_budget_bytes=configured_gpu_cap_bytes,
                        )
                    cancel_mode = _query_direct_epfd_cancel_mode(cancel_callback)
                    if cancel_mode in {"graceful", "force"}:
                        run_state = "stopped"
                        stop_mode = cancel_mode
                        stop_boundary = "post_batch_boundary"
                        if not stop_notice_emitted:
                            _emit_direct_epfd_progress(
                                progress_callback,
                                kind="phase",
                                phase="stopping",
                                iteration_index=int(ii),
                                iteration_total=int(iteration_count),
                                batch_index=int(bi),
                                batch_total=int(n_batches),
                                stop_mode=str(stop_mode),
                                stop_boundary=str(stop_boundary),
                                description="Stop requested. Flushing completed work before exit.",
                            )
                            stop_notice_emitted = True
                        break

                # End-of-iteration cleanup happens only here, after the full
                # batch loop has completed and all preaccumulated host-side
                # collectors have already been updated for this iteration.
                _per_iteration_gpu_cleanup()

                if run_state == "stopped":
                    break
    finally:
        pending_exc = sys.exc_info()[1]
        _sync_array_module(cp)
        try:
            if terminal_gpu_cleanup and hasattr(cp, "get_default_memory_pool"):
                cp.get_default_memory_pool().free_all_blocks()
            if terminal_gpu_cleanup and hasattr(cp, "get_default_pinned_memory_pool"):
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        gc.collect()
        if owns_session:
            try:
                if terminal_gpu_cleanup:
                    session.close()
                else:
                    session.close(reset_device=False)
            except Exception:
                if pending_exc is None:
                    raise
        preaccumulated_families_payload: dict[str, dict[str, Any]] = {}
        for family_name, collector in distribution_collectors.items():
            counts_np, edges_np = _finalize_empty_distribution_collector(collector)
            family_attrs = {
                **{
                    str(key): value
                    for key, value in dict(collector["config"]).items()
                    if key != "mode"
                },
                "mode": str(collector["config"]["mode"]),
                "sample_count": int(collector["sample_count"]),
            }
            if family_name in {
                "prx_total_distribution",
                "epfd_distribution",
                "total_pfd_ras_distribution",
                "per_satellite_pfd_distribution",
            }:
                if spectrum_plan_effective is not None:
                    family_attrs["stored_value_basis"] = "ras_receiver_band"
                else:
                    family_attrs["stored_value_basis"] = "channel_total"
                family_attrs["bandwidth_mhz"] = float(power_input["bandwidth_mhz"])
            preaccumulated_families_payload[family_name] = {
                "attrs": family_attrs,
                "datasets": {
                    "counts": counts_np,
                    "edges_dbw": edges_np,
                },
            }
        for family_name, collector in heatmap_collectors.items():
            counts_np, elevation_edges_np, value_edges_np = _finalize_empty_heatmap_collector(
                collector
            )
            family_attrs = {
                **{
                    str(key): value
                    for key, value in dict(collector["config"]).items()
                    if key != "mode"
                },
                "mode": str(collector["config"]["mode"]),
                "sample_count": int(collector["sample_count"]),
            }
            if family_name in {
                "prx_elevation_heatmap",
                "per_satellite_pfd_elevation_heatmap",
            }:
                family_attrs["stored_value_basis"] = "channel_total"
                family_attrs["bandwidth_mhz"] = float(power_input["bandwidth_mhz"])
            preaccumulated_families_payload[family_name] = {
                "attrs": family_attrs,
                "datasets": {
                    "counts": counts_np,
                    "elevation_edges_deg": elevation_edges_np,
                    "value_edges_dbw": value_edges_np,
                },
            }
        if beam_stats_collector is not None:
            full_hist = beam_stats_collector["full_network_count_hist"]
            visible_hist = beam_stats_collector["visible_count_hist"]
            if full_hist is not None or visible_hist is not None:
                full_hist_np = np.asarray(
                    full_hist if full_hist is not None else np.zeros(1, dtype=np.int64),
                    dtype=np.int64,
                )
                visible_hist_np = np.asarray(
                    visible_hist if visible_hist is not None else np.zeros(1, dtype=np.int64),
                    dtype=np.int64,
                )
                hist_len = max(int(full_hist_np.size), int(visible_hist_np.size), 1)
                if int(full_hist_np.size) < hist_len:
                    full_hist_np = np.pad(
                        full_hist_np,
                        (0, hist_len - int(full_hist_np.size)),
                        mode="constant",
                    )
                if int(visible_hist_np.size) < hist_len:
                    visible_hist_np = np.pad(
                        visible_hist_np,
                        (0, hist_len - int(visible_hist_np.size)),
                        mode="constant",
                    )
                network_total_series = (
                    np.concatenate(
                        [np.asarray(part) for part in beam_stats_collector["network_total_beams_over_time"]],
                        axis=0,
                    )
                    if beam_stats_collector["network_total_beams_over_time"]
                    else np.zeros((0,), dtype=np.int64)
                )
                visible_total_series = (
                    np.concatenate(
                        [np.asarray(part) for part in beam_stats_collector["visible_total_beams_over_time"]],
                        axis=0,
                    )
                    if beam_stats_collector["visible_total_beams_over_time"]
                    else np.zeros((0,), dtype=np.int64)
                )
                beam_demand_series = (
                    np.concatenate(
                        [np.asarray(part) for part in beam_stats_collector["beam_demand_over_time"]],
                        axis=0,
                    )
                    if beam_stats_collector["beam_demand_over_time"]
                    else np.zeros((0,), dtype=np.int64)
                )
                expected_series_rows = int(iteration_count) * int(n_steps_total)
                if expected_series_rows > 0:
                    if (
                        network_total_series.ndim >= 1
                        and int(network_total_series.shape[0]) == expected_series_rows
                    ):
                        network_total_series = network_total_series.reshape(
                            (int(iteration_count), int(n_steps_total))
                            + tuple(network_total_series.shape[1:])
                        )
                    if (
                        visible_total_series.ndim >= 1
                        and int(visible_total_series.shape[0]) == expected_series_rows
                    ):
                        visible_total_series = visible_total_series.reshape(
                            (int(iteration_count), int(n_steps_total))
                            + tuple(visible_total_series.shape[1:])
                        )
                    if (
                        beam_demand_series.ndim >= 1
                        and int(beam_demand_series.shape[0]) == expected_series_rows
                    ):
                        beam_demand_series = beam_demand_series.reshape(
                            (int(iteration_count), int(n_steps_total))
                            + tuple(beam_demand_series.shape[1:])
                        )
                preaccumulated_families_payload["beam_statistics"] = {
                    "attrs": {
                        "mode": str(beam_stats_collector["config"]["mode"]),
                    },
                    "datasets": {
                        "full_network_count_hist": full_hist_np,
                        "visible_count_hist": visible_hist_np,
                        "count_edges": np.arange(hist_len + 1, dtype=np.int64),
                        "network_total_beams_over_time": np.asarray(
                            network_total_series,
                            dtype=np.int64,
                        ),
                        "visible_total_beams_over_time": np.asarray(
                            visible_total_series,
                            dtype=np.int64,
                        ),
                        "beam_demand_over_time": np.asarray(
                            beam_demand_series,
                            dtype=np.int64,
                        ),
                    },
                }
        if preaccumulated_families_payload:
            _write_preaccumulated_families(
                storage_filename,
                families=preaccumulated_families_payload,
                compression=None,
                compression_opts=None,
            )
        # Write per-group preaccumulated histograms
        if _multi_system_active and _group_collectors:
            for _gc_write in _group_collectors:
                _gc_prefix = _group_hdf5_prefix(_gc_write["group_def"], _n_multi_systems)
                # Skip the "combined" group (prefix="") — already written above
                # as the root preaccumulated data.
                if _gc_prefix == "":
                    continue
                _gc_write_colls = _gc_write["distribution_collectors"]
                _gc_preacc_payload: dict[str, dict[str, Any]] = {}
                for _gc_fn, _gc_coll in _gc_write_colls.items():
                    _gc_counts_np, _gc_edges_np = _finalize_empty_distribution_collector(_gc_coll)
                    _gc_family_attrs = {
                        **{str(k): v for k, v in dict(_gc_coll["config"]).items() if k != "mode"},
                        "mode": str(_gc_coll["config"]["mode"]),
                        "sample_count": int(_gc_coll["sample_count"]),
                    }
                    if _gc_fn in _distribution_family_names:
                        _gc_family_attrs["stored_value_basis"] = (
                            "ras_receiver_band" if spectrum_plan_effective is not None else "channel_total"
                        )
                        _gc_family_attrs["bandwidth_mhz"] = float(power_input["bandwidth_mhz"])
                    _gc_preacc_payload[_gc_fn] = {
                        "attrs": _gc_family_attrs,
                        "datasets": {"counts": _gc_counts_np, "edges_dbw": _gc_edges_np},
                    }
                if _gc_preacc_payload:
                    _write_preaccumulated_families(
                        storage_filename,
                        families=_gc_preacc_payload,
                        compression=None,
                        compression_opts=None,
                        group_prefix=_gc_prefix,
                    )
                # Write group metadata attrs
                try:
                    _maybe_flush_pending_writes(storage_filename)
                    _gc_indices = sorted(_gc_write["system_indices"])
                    # For single-system groups, write system-level metadata
                    if len(_gc_indices) == 1:
                        _ps_write_idx = _gc_indices[0]
                        _ps_ctx = _multi_system_contexts[_ps_write_idx] if _ps_write_idx < len(_multi_system_contexts) else {}
                        _ps_pi = _ps_ctx.get("power_input", {})
                        with h5py.File(storage_filename, "a") as _sf:
                            _sg = _sf.require_group(_gc_prefix.rstrip("/"))
                            _sg.attrs["system_name"] = str(_ps_ctx.get("system_name", f"System {_ps_write_idx + 1}"))
                            _sg.attrs["system_index"] = int(_ps_write_idx)
                            _sg.attrs["n_sats_total"] = int(_ps_ctx.get("n_sats_total", 0))
                            _sg.attrs["n_cells_total"] = int(_ps_ctx.get("n_cells_total", 0))
                            # UEMR systems have no Nco/Nbeam/selection — record
                            # "n/a" so a downstream HDF5 reader doesn't mistake
                            # the kernel sentinel for a user-chosen value. Match
                            # the shared root-attr convention from the GUI.
                            _ps_uemr = bool(_ps_ctx.get("uemr_mode", False))
                            _sg.attrs["uemr_mode"] = _ps_uemr
                            _sg.attrs["antenna_model"] = str(_ps_ctx.get("antenna_model", ""))
                            if _ps_uemr:
                                _sg.attrs["nco"] = "n/a"
                                _sg.attrs["nbeam"] = "n/a"
                                _sg.attrs["selection_mode"] = "n/a"
                            else:
                                _sg.attrs["nco"] = int(_ps_ctx.get("nco", 1))
                                _sg.attrs["nbeam"] = int(_ps_ctx.get("nbeam", 1))
                                _sg.attrs["selection_mode"] = str(_ps_ctx.get("selection_mode", ""))
                            _sg.attrs["cell_activity_factor"] = float(_ps_ctx.get("cell_activity_factor", 1.0))
                            _sg.attrs["cell_activity_mode"] = str(_ps_ctx.get("cell_activity_mode", ""))
                            _sg.attrs["wavelength_m"] = float(_ps_ctx.get("wavelength_m", 0.0))
                            _sg.attrs["frequency_ghz"] = float(_ps_ctx.get("frequency_ghz", 0.0))
                            _sg.attrs["power_input_quantity"] = str(_ps_pi.get("power_input_quantity", ""))
                            _sg.attrs["power_input_basis"] = str(_ps_pi.get("power_input_basis", ""))
                            _sg.attrs["bandwidth_mhz"] = float(_ps_pi.get("bandwidth_mhz", 0.0))
                            for _pk in ("target_pfd_dbw_m2_mhz", "target_pfd_dbw_m2_channel",
                                        "satellite_eirp_dbw_mhz", "satellite_eirp_dbw_channel",
                                        "satellite_ptx_dbw_mhz", "satellite_ptx_dbw_channel"):
                                _pv = _ps_pi.get(_pk)
                                _sg.attrs[_pk] = float(_pv) if _pv is not None else float("nan")
                            _sg.attrs["power_variation_mode"] = str(
                                _ps_pi.get("power_variation_mode", "fixed")
                            )
                            _ps_pmin = _ps_pi.get("power_range_min_dbw_channel")
                            _ps_pmax = _ps_pi.get("power_range_max_dbw_channel")
                            _sg.attrs["power_range_min_dbw_channel"] = (
                                float(_ps_pmin) if _ps_pmin is not None else float("nan")
                            )
                            _sg.attrs["power_range_max_dbw_channel"] = (
                                float(_ps_pmax) if _ps_pmax is not None else float("nan")
                            )
                    else:
                        # Custom multi-system group — write group name and indices
                        with h5py.File(storage_filename, "a") as _sf:
                            _sg = _sf.require_group(_gc_prefix.rstrip("/"))
                            _sg.attrs["group_name"] = str(_gc_write["name"])
                            _sg.attrs["system_indices"] = np.array(_gc_indices, dtype=np.int32)
                except Exception:
                    pass

                # Per-group heatmap writes
                _gc_hm_colls = _gc_write["heatmap_collectors"]
                _gc_hm_payload: dict[str, dict[str, Any]] = {}
                for _hm_fn, _hm_coll in _gc_hm_colls.items():
                    _hm_counts, _hm_elev, _hm_val = _finalize_empty_heatmap_collector(_hm_coll)
                    _hm_attrs = {
                        **{
                            str(k): v
                            for k, v in dict(_hm_coll["config"]).items()
                            if k != "mode"
                        },
                        "mode": str(_hm_coll["config"]["mode"]),
                        "sample_count": int(_hm_coll["sample_count"]),
                    }
                    if _hm_fn in {
                        "prx_elevation_heatmap",
                        "per_satellite_pfd_elevation_heatmap",
                    }:
                        _hm_attrs["stored_value_basis"] = "channel_total"
                        # Use first system's bandwidth as representative
                        _gc_hm_pi = {}
                        if _gc_indices:
                            _gc_hm_sys0 = _gc_indices[0]
                            if _gc_hm_sys0 < len(_multi_system_contexts):
                                _gc_hm_pi = _multi_system_contexts[_gc_hm_sys0].get("power_input", {})
                        _hm_attrs["bandwidth_mhz"] = float(
                            _gc_hm_pi.get("bandwidth_mhz", power_input["bandwidth_mhz"])
                        )
                    _gc_hm_payload[_hm_fn] = {
                        "attrs": _hm_attrs,
                        "datasets": {
                            "counts": _hm_counts,
                            "elevation_edges_deg": _hm_elev,
                            "value_edges_dbw": _hm_val,
                        },
                    }
                if _gc_hm_payload:
                    _write_preaccumulated_families(
                        storage_filename,
                        families=_gc_hm_payload,
                        compression=None,
                        compression_opts=None,
                        group_prefix=_gc_prefix,
                    )

                # Per-group beam stats writes
                _gc_bs_coll = _gc_write.get("beam_stats_collector")
                if _gc_bs_coll is not None:
                    _gc_full_h = _gc_bs_coll["full_network_count_hist"]
                    _gc_vis_h = _gc_bs_coll["visible_count_hist"]
                    if _gc_full_h is not None or _gc_vis_h is not None:
                        _gc_full_np = np.asarray(
                            _gc_full_h if _gc_full_h is not None else np.zeros(1, dtype=np.int64),
                            dtype=np.int64,
                        )
                        _gc_vis_np = np.asarray(
                            _gc_vis_h if _gc_vis_h is not None else np.zeros(1, dtype=np.int64),
                            dtype=np.int64,
                        )
                        _gc_hist_len = max(int(_gc_full_np.size), int(_gc_vis_np.size), 1)
                        if int(_gc_full_np.size) < _gc_hist_len:
                            _gc_full_np = np.pad(
                                _gc_full_np,
                                (0, _gc_hist_len - int(_gc_full_np.size)),
                                mode="constant",
                            )
                        if int(_gc_vis_np.size) < _gc_hist_len:
                            _gc_vis_np = np.pad(
                                _gc_vis_np,
                                (0, _gc_hist_len - int(_gc_vis_np.size)),
                                mode="constant",
                            )
                        _gc_net_series = (
                            np.concatenate(
                                [np.asarray(p) for p in _gc_bs_coll["network_total_beams_over_time"]],
                                axis=0,
                            )
                            if _gc_bs_coll["network_total_beams_over_time"]
                            else np.zeros((0,), dtype=np.int64)
                        )
                        _gc_vis_series = (
                            np.concatenate(
                                [np.asarray(p) for p in _gc_bs_coll["visible_total_beams_over_time"]],
                                axis=0,
                            )
                            if _gc_bs_coll["visible_total_beams_over_time"]
                            else np.zeros((0,), dtype=np.int64)
                        )
                        _gc_demand_series = (
                            np.concatenate(
                                [np.asarray(p) for p in _gc_bs_coll["beam_demand_over_time"]],
                                axis=0,
                            )
                            if _gc_bs_coll["beam_demand_over_time"]
                            else np.zeros((0,), dtype=np.int64)
                        )
                        _gc_beam_datasets: dict[str, Any] = {
                            "full_network_count_hist": _gc_full_np,
                            "visible_count_hist": _gc_vis_np,
                            "count_edges": np.arange(_gc_hist_len + 1, dtype=np.int64),
                            "network_total_beams_over_time": np.asarray(
                                _gc_net_series, dtype=np.int64,
                            ),
                            "visible_total_beams_over_time": np.asarray(
                                _gc_vis_series, dtype=np.int64,
                            ),
                        }
                        if _gc_demand_series.size > 0:
                            _gc_beam_datasets["beam_demand_over_time"] = np.asarray(
                                _gc_demand_series, dtype=np.int64,
                            )
                        _gc_beam_payload: dict[str, dict[str, Any]] = {
                            "beam_statistics": {
                                "attrs": {
                                    "mode": str(_gc_bs_coll["config"]["mode"]),
                                },
                                "datasets": _gc_beam_datasets,
                            },
                        }
                        _write_preaccumulated_families(
                            storage_filename,
                            families=_gc_beam_payload,
                            compression=None,
                            compression_opts=None,
                            group_prefix=_gc_prefix,
                        )

                # Per-system constants (only for single-system groups)
                if len(_gc_indices) == 1:
                    _ps_write_idx = _gc_indices[0]
                    if _ps_write_idx < len(_multi_system_contexts):
                        try:
                            _ps_const_ctx = _multi_system_contexts[_ps_write_idx]
                            _ps_const_pi = _ps_const_ctx.get("power_input", {})
                            _maybe_flush_pending_writes(storage_filename)
                            with h5py.File(storage_filename, "a") as _cf:
                                _cg = _cf.require_group(f"{_gc_prefix}const")
                                _cg.attrs["system_name"] = str(
                                    _ps_const_ctx.get("system_name", f"System {_ps_write_idx + 1}")
                                )
                                _cg.attrs["system_index"] = int(_ps_write_idx)
                                _cg.attrs["n_sats_total"] = int(_ps_const_ctx.get("n_sats_total", 0))
                                _cg.attrs["n_cells_total"] = int(_ps_const_ctx.get("n_cells_total", 0))
                                _cg.attrs["nco"] = int(_ps_const_ctx.get("nco", 1))
                                _cg.attrs["nbeam"] = int(_ps_const_ctx.get("nbeam", 1))
                                _cg.attrs["selection_mode"] = str(
                                    _ps_const_ctx.get("selection_mode", "")
                                )
                                _cg.attrs["cell_activity_factor"] = float(
                                    _ps_const_ctx.get("cell_activity_factor", 1.0)
                                )
                                _cg.attrs["cell_activity_mode"] = str(
                                    _ps_const_ctx.get("cell_activity_mode", "")
                                )
                                _cg.attrs["wavelength_m"] = float(
                                    _ps_const_ctx.get("wavelength_m", 0.0)
                                )
                                _cg.attrs["frequency_ghz"] = float(
                                    _ps_const_ctx.get("frequency_ghz", 0.0)
                                )
                                _cg.attrs["power_input_quantity"] = str(
                                    _ps_const_pi.get("power_input_quantity", "")
                                )
                                _cg.attrs["power_input_basis"] = str(
                                    _ps_const_pi.get("power_input_basis", "")
                                )
                                _cg.attrs["bandwidth_mhz"] = float(
                                    _ps_const_pi.get("bandwidth_mhz", 0.0)
                                )
                                for _cpk in (
                                    "target_pfd_dbw_m2_mhz",
                                    "target_pfd_dbw_m2_channel",
                                    "satellite_eirp_dbw_mhz",
                                    "satellite_eirp_dbw_channel",
                                    "satellite_ptx_dbw_mhz",
                                    "satellite_ptx_dbw_channel",
                                ):
                                    _cpv = _ps_const_pi.get(_cpk)
                                    _cg.attrs[_cpk] = (
                                        float(_cpv) if _cpv is not None else float("nan")
                                    )
                                # Write per-system storage_attrs (antenna model, spectrum, etc.)
                                _ps_sa = _ps_const_ctx.get("storage_attrs", {})
                                for _sak, _sav in _ps_sa.items():
                                    if _sak.startswith("_") or _sak in {"system_count", "system_names"}:
                                        continue
                                    try:
                                        if isinstance(_sav, np.ndarray):
                                            if _sak not in _cg:
                                                _cg.create_dataset(_sak, data=_sav)
                                        else:
                                            _cg.attrs[_sak] = _sav
                                    except Exception:
                                        pass
                                # Write per-system constant datasets
                                for _cds_name, _cds_key, _cds_dtype in [
                                    ("sat_orbit_radius_m_per_sat", "orbit_radius_host", np.float32),
                                    ("sat_min_elev_deg_per_sat", "sat_min_elev_deg_per_sat_f64", np.float64),
                                    ("sat_beta_max_deg_per_sat", "sat_beta_max_deg_per_sat_f32", np.float32),
                                    ("sat_belt_id_per_sat", "sat_belt_id_per_sat_i16", np.int16),
                                ]:
                                    _cds_val = _ps_const_ctx.get(_cds_key)
                                    if _cds_val is not None:
                                        _cds_arr = np.asarray(_cds_val, dtype=_cds_dtype)
                                        if _cds_name not in _cg:
                                            _cg.create_dataset(_cds_name, data=_cds_arr)
                        except Exception:
                            pass

            # Write output_system_groups metadata to HDF5 for postprocess
            try:
                _maybe_flush_pending_writes(storage_filename)
                with h5py.File(storage_filename, "a") as _gf:
                    _gf.attrs["output_group_count"] = len(_group_collectors)
                    _group_names_list = []
                    _group_prefixes_list = []
                    for _gc_meta in _group_collectors:
                        _group_names_list.append(str(_gc_meta["name"]))
                        _group_prefixes_list.append(
                            _group_hdf5_prefix(_gc_meta["group_def"], _n_multi_systems)
                        )
                    _gf.attrs["output_group_names"] = ";".join(_group_names_list)
                    _gf.attrs["output_group_prefixes"] = ";".join(_group_prefixes_list)
            except Exception:
                pass

        writer = _get_registered_writer(storage_filename)
        _set_direct_epfd_progress_phase(
            locals().get("pbar", None),
            enable_progress_bars=enable_progress_bars,
            progress_desc_mode=progress_desc_mode_name,
            phase="final_flush",
        )
        _emit_direct_epfd_progress(
            progress_callback,
            kind="phase",
            phase="final_flush",
            iteration_total=int(iteration_count),
            checkpoint_count=int(writer_checkpoint_count),
            description=_direct_epfd_progress_text(progress_desc_mode_name, "final_flush")
            or "Final flush",
        )
        writer_flush_t0 = perf_counter() if writer is not None else None
        try:
            close_writer(storage_filename)
        except Exception:
            if pending_exc is None:
                raise
        if writer_flush_t0 is not None:
            writer_final_flush_s = perf_counter() - writer_flush_t0
        if writer is not None:
            writer_stats_summary = writer.stats_snapshot()
        else:
            writer_stats_summary = _get_writer_stats_snapshot(storage_filename)
        if profile_stages:
            profile_stage_timings_summary["writer_flush"] = float(writer_final_flush_s)
            profile_stage_timings_summary["writer_checkpoint_wait"] = float(
                writer_checkpoint_wait_s
            )
            profile_stage_timings_summary["writer_checkpoint_count"] = float(
                writer_checkpoint_count
            )
            profile_stage_timings_summary["export_scatter"] = float(
                writer_stats_summary.get("prepare_elapsed_total", 0.0)
            )
            profile_stage_timings_summary["writer_apply"] = float(
                writer_stats_summary.get("apply_elapsed_total", 0.0)
            )
            profile_stage_timings_summary["writer_flush_count"] = float(
                writer_stats_summary.get("flush_count", 0)
            )
        writer_total_elapsed = (
            float(writer_stats_summary.get("prepare_elapsed_total", 0.0))
            + float(writer_stats_summary.get("apply_elapsed_total", 0.0))
            + float(writer_checkpoint_wait_s)
            + float(writer_final_flush_s)
        )
        if profile_stages or writer_total_elapsed >= 0.25:
            print(
                "[Direct EPFD gpu] writer "
                f"prepare={float(writer_stats_summary.get('prepare_elapsed_total', 0.0)):.3f}s "
                f"apply={float(writer_stats_summary.get('apply_elapsed_total', 0.0)):.3f}s "
                f"checkpoint_wait={float(writer_checkpoint_wait_s):.3f}s "
                f"final_flush={float(writer_final_flush_s):.3f}s "
                f"peak_queue_items={int(writer_stats_summary.get('queued_items_high_water', 0))} "
                f"peak_queue_mb={float(writer_stats_summary.get('queued_bytes_high_water', 0)) / float(1024 ** 2):.1f}"
            )
        if run_state == "stopped":
            _emit_direct_epfd_progress(
                progress_callback,
                kind="run_stopped",
                phase="stopped",
                iteration_total=int(iteration_count),
                storage_filename=str(storage_filename),
                stop_mode=str(stop_mode),
                stop_boundary=None if stop_boundary is None else str(stop_boundary),
                writer_checkpoint_count=int(writer_checkpoint_count),
                writer_checkpoint_wait_s=float(writer_checkpoint_wait_s),
                writer_final_flush_s=float(writer_final_flush_s),
                writer_stats_summary=dict(writer_stats_summary),
                **_scheduler_runtime_state_extra(scheduler_runtime_state),
            )
        else:
            _emit_direct_epfd_progress(
                progress_callback,
                kind="run_complete",
                phase="completed",
                iteration_total=int(iteration_count),
                storage_filename=str(storage_filename),
                writer_checkpoint_count=int(writer_checkpoint_count),
                writer_checkpoint_wait_s=float(writer_checkpoint_wait_s),
                writer_final_flush_s=float(writer_final_flush_s),
                writer_stats_summary=dict(writer_stats_summary),
                profile_stage_timings_summary=(
                    dict(profile_stage_timings_summary) if profile_stages else {}
                ),
                beam_finalize_substage_timings=dict(beam_finalize_substage_timings_summary),
                cell_link_library_chunk_telemetry=dict(cell_link_library_chunk_telemetry_summary),
                beam_finalize_chunk_shape=dict(beam_finalize_chunk_shape_summary),
                boresight_compaction_stats=dict(boresight_compaction_stats_summary),
                hot_path_device_to_host_copy_count=int(
                    hot_path_device_to_host_copy_count_summary
                ),
                hot_path_device_to_host_copy_bytes=int(
                    hot_path_device_to_host_copy_bytes_summary
                ),
                device_scalar_sync_count=int(device_scalar_sync_count_summary),
                observed_stage_memory_summary_by_name={
                    str(name): dict(summary)
                    for name, summary in observed_stage_memory_summary_by_name.items()
                },
                **_scheduler_runtime_state_extra(scheduler_runtime_state),
            )

    return {
        "storage_filename": storage_filename,
        "effective_ras_pointing_mode": effective_ras_pointing_mode,
        "beam_generation_method": beam_generation_method,
        "boresight_active": boresight_active,
        "boresight_theta1_deg": boresight_theta1_deg,
        "boresight_theta2_deg": boresight_theta2_deg,
        "debug_direct_epfd": bool(debug_direct_epfd),
        "debug_direct_epfd_stats": debug_direct_epfd_stats_all if debug_direct_epfd else [],
        "profile_stage_timings": profile_stage_timings_all if profile_stages else [],
        "profile_stage_timings_summary": profile_stage_timings_summary if profile_stages else {},
        "beam_finalize_substage_timings": dict(beam_finalize_substage_timings_summary),
        "cell_link_library_chunk_telemetry": dict(cell_link_library_chunk_telemetry_summary),
        "beam_finalize_chunk_shape": dict(beam_finalize_chunk_shape_summary),
        "boresight_compaction_stats": dict(boresight_compaction_stats_summary),
        "hot_path_device_to_host_copy_count": int(hot_path_device_to_host_copy_count_summary),
        "hot_path_device_to_host_copy_bytes": int(hot_path_device_to_host_copy_bytes_summary),
        "device_scalar_sync_count": int(device_scalar_sync_count_summary),
        "observed_stage_memory_summary_by_name": {
            str(name): dict(summary)
            for name, summary in observed_stage_memory_summary_by_name.items()
        },
        "writer_stats_summary": writer_stats_summary,
        "writer_checkpoint_count": int(writer_checkpoint_count),
        "writer_checkpoint_wait_s": float(writer_checkpoint_wait_s),
        "writer_final_flush_s": float(writer_final_flush_s),
        "n_steps_total": n_steps_total,
        "output_families": family_configs,
        "written_output_names": written_output_names,
        "bandwidth_mhz": float(power_input["bandwidth_mhz"]),
        "power_input_quantity": str(power_input["power_input_quantity"]),
        "power_input_basis": str(power_input["power_input_basis"]),
        "dynamic_execution_skipped": False,
        "run_state": str(run_state),
        "stop_mode": str(stop_mode),
        "stop_boundary": None if stop_boundary is None else str(stop_boundary),
    }

def generate_simulation_batches(start_time, end_time, timestep, batch_size):
    """
    Generate batches of simulation times between start_time and end_time.

    Each batch contains up to 'batch_size' timesteps. The function returns a dictionary 
    with keys corresponding to each batch parameter. Specifically, it returns a dictionary
    with:
      - 'batch_start': list of batch start times (astropy Time objects),
      - 'times': list of astropy Time arrays for each batch,
      - 'td': list of TimeDelta arrays for each batch,
      - 'batch_end': list of batch end times (astropy Time objects).
    
    Parameters
    ----------
    start_time : astropy.time.Time
        Global simulation start time.
    end_time : astropy.time.Time
        Global simulation end time.
    timestep : float
        Time step in seconds.
    batch_size : int
        Maximum number of timesteps per batch.

    Returns
    -------
    batches : dict
        A dictionary containing lists for each batch parameter.
    """
    batches = {
        'batch_start': [],
        'times': [],
        'td': [],
        'batch_end': [],
    }

    for batch in iter_simulation_batches(start_time, end_time, timestep, batch_size):
        batches['batch_start'].append(batch['batch_start'])
        batches['times'].append(batch['times'])
        batches['td'].append(batch['td'])
        batches['batch_end'].append(batch['batch_end'])

    return batches

@contextmanager
def block_interrupts():
    """
    A context manager to temporarily ignore SIGINT (KeyboardInterrupt).
    This ensures that the enclosed critical section (e.g., file I/O) is not interrupted
    by Ctrl+C. Once the block finishes (or if it errors), the original handler is restored.
    """
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore future SIGINT
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)  # Restore original handler


def flush_writes(filename: str | None = None) -> None:
    """
    Flush pending background HDF5 writes.

    Parameters
    ----------
    filename : str, optional
        Specific HDF5 file to flush. When omitted, all active background
        writers are flushed.
    """
    if filename is None:
        with _WRITER_REGISTRY_LOCK:
            writers = list(_WRITER_REGISTRY.values())
            failed_snapshots = list(_FAILED_WRITER_REGISTRY.values())
        first_error: BaseException | None = None
        for writer in writers:
            try:
                writer.flush()
            except BaseException as exc:
                _remember_failed_writer_snapshot(writer.failure_snapshot())
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        if failed_snapshots:
            raise RuntimeError(failed_snapshots[0].render_message())
        return

    filename = _normalize_storage_path(filename)
    with _WRITER_REGISTRY_LOCK:
        writer = _WRITER_REGISTRY.get(filename)
        failed_snapshot = _FAILED_WRITER_REGISTRY.get(filename)
    if writer is not None:
        try:
            writer.flush()
        except BaseException:
            _remember_failed_writer_snapshot(writer.failure_snapshot())
            raise
        return
    if failed_snapshot is not None:
        raise RuntimeError(failed_snapshot.render_message())


def close_writer(filename: str | None = None) -> None:
    """
    Flush and stop background HDF5 writers.

    Parameters
    ----------
    filename : str, optional
        Specific HDF5 file to close. When omitted, all active writers are
        closed.
    """
    if filename is None:
        with _WRITER_REGISTRY_LOCK:
            items = list(_WRITER_REGISTRY.items())
            _WRITER_REGISTRY.clear()
            _FAILED_WRITER_REGISTRY.clear()
        for _, writer in items:
            try:
                writer.close()
            except Exception:
                pass
        return

    filename = _normalize_storage_path(filename)
    with _WRITER_REGISTRY_LOCK:
        writer = _WRITER_REGISTRY.get(filename)
        failed_snapshot = _FAILED_WRITER_REGISTRY.get(filename)
    if writer is None:
        if failed_snapshot is not None:
            raise RuntimeError(failed_snapshot.render_message())
        return
    try:
        writer.close()
    except BaseException:
        _remember_failed_writer_snapshot(writer.failure_snapshot())
        raise
    with _WRITER_REGISTRY_LOCK:
        if _WRITER_REGISTRY.get(filename) is writer:
            _WRITER_REGISTRY.pop(filename, None)
        _FAILED_WRITER_REGISTRY.pop(filename, None)


def init_simulation_results(
    filename: str,
    *,
    write_mode: str = "async",
    writer_queue_max_items: int = 8,
    writer_queue_max_bytes: int = 1 * 1024 ** 3,
) -> None:
    """
    Initialize or reset the HDF5 file that stores simulation results.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    write_mode : {"async", "sync"}, optional
        Default write mode used for subsequent :func:`write_data` calls that
        target this file.
    writer_queue_max_items : int, optional
        Maximum number of queued async write operations allowed before producer
        backpressure blocks.
    writer_queue_max_bytes : int, optional
        Maximum estimated bytes allowed in the async writer queue before
        producer backpressure blocks.

    Returns
    -------
    None

    Notes
    -----
    The file is truncated if it already exists. When ``write_mode="async"``,
    the default background writer for ``filename`` is also started eagerly.
    """
    filename = _normalize_storage_path(filename)
    if write_mode not in {"async", "sync"}:
        raise ValueError("`write_mode` must be 'async' or 'sync'.")

    try:
        close_writer(filename)
    except Exception:
        pass
    if os.path.exists(filename):
        os.remove(filename)
    with _interrupt_ctx(), h5py.File(filename, "w"):
        pass
    with _WRITER_REGISTRY_LOCK:
        _FAILED_WRITER_REGISTRY.pop(filename, None)
    _WRITER_DEFAULTS[filename] = {
        "write_mode": write_mode,
        "writer_queue_max_items": max(1, int(writer_queue_max_items)),
        "writer_queue_max_bytes": max(1, int(writer_queue_max_bytes)),
    }
    if write_mode == "async":
        _ensure_async_writer(
            filename,
            writer_queue_max_items=writer_queue_max_items,
            writer_queue_max_bytes=writer_queue_max_bytes,
        )




# -----------------------------------------------------------------------------
# Small internal helpers
# -----------------------------------------------------------------------------

def _interrupt_ctx():
    """
    Use your project's block_interrupts() if present; otherwise no-op.
    """
    if threading.current_thread() is not threading.main_thread():
        return nullcontext()
    ctx = globals().get("block_interrupts", None)
    return ctx() if callable(ctx) else nullcontext()


def _is_time_obj(x: Any) -> bool:
    return (Time is not None) and isinstance(x, Time)


def _to_array_unit_kind(value: Any) -> Tuple[np.ndarray, Any | None, str | None]:
    """
    Normalise a supported object into (array, unit, kind).

    kind:
        - "time/mjd" for astropy.Time
        - "quantity" for astropy.Quantity
        - None for plain arrays
    """
    if _is_time_obj(value):
        arr = np.asarray(value.mjd)               # float days
        unit = u.day if u is not None else "d"    # keep unit as days
        kind = "time/mjd"
        return arr, unit, kind

    if hasattr(value, "unit"):                    # Quantity-like
        arr = np.asarray(value.value)
        unit = getattr(value, "unit")
        return arr, unit, "quantity"

    # Plain array-like
    return np.asarray(value), None, None


def _ensure_row_first(arr: np.ndarray) -> np.ndarray:
    """
    For streaming appends we require shape (N, ...). If user passes a scalar
    or a 1D vector, reshape accordingly so that we can extend on axis 0.
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr


def _to_unit_str(unit: Any | None) -> str | None:
    if unit is None:
        return None
    if u is not None and isinstance(unit, u.UnitBase):
        return str(unit)
    return str(unit)


def _units_convertible(u_from: Any, u_to: Any) -> bool:
    if u is None:
        return False
    try:
        _ = (1 * u_from).to(u_to)
        return True
    except Exception:
        return False


def _convert_array_unit(arr: np.ndarray, u_from: Any, u_to: Any) -> np.ndarray:
    if u is None or (u_from is None) or (u_to is None) or str(u_from) == str(u_to):
        return arr
    return (arr * u_from).to(u_to).value  # type: ignore


_AUTO_STORAGE_ARG = object()
_STREAM_SENTINEL = object()
_DEFAULT_SLOT_CHUNK_TARGET_BYTES = 8 * 1024 ** 2
_DEFAULT_WRITER_QUEUE_MAX_ITEMS = 8
_DEFAULT_WRITER_QUEUE_MAX_BYTES = 1 * 1024 ** 3
_DEFAULT_WRITER_CYCLE_MAX_ITEMS = 2
_DEFAULT_WRITER_CYCLE_MAX_BYTES = 256 * 1024 ** 2
_COMPRESSION_PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "balanced": {"compression": "gzip", "compression_opts": 4, "shuffle": True},
    "smallest": {"compression": "gzip", "compression_opts": 9, "shuffle": True},
    "fastest": {"compression": "lzf", "compression_opts": None, "shuffle": True},
    "off": {"compression": None, "compression_opts": None, "shuffle": False},
}
_WRITER_REGISTRY_LOCK = threading.RLock()
_WRITER_REGISTRY: dict[str, "_AsyncH5Writer"] = {}
_FAILED_WRITER_REGISTRY: dict[str, "_FailedWriterSnapshot"] = {}
_WRITER_DEFAULTS: dict[str, dict[str, Any]] = {}
_WRITER_ATEXIT_REGISTERED = False


def _exception_chain_messages(exc: BaseException) -> tuple[str, ...]:
    """Return a compact exception/cause chain for user-facing writer failures."""
    messages: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        label = type(current).__name__ if not message else f"{type(current).__name__}: {message}"
        messages.append(label)
        cause = current.__cause__
        if cause is not None:
            current = cause
            continue
        if getattr(current, "__suppress_context__", False):
            break
        current = current.__context__
    return tuple(messages)


@dataclass(frozen=True)
class _FailedWriterSnapshot:
    """Persistent writer failure record kept after async writer teardown fails."""

    filename: str
    error_type: str
    error_message: str
    error_chain: tuple[str, ...]
    submitted_seq: int
    completed_seq: int
    durable_seq: int
    writer_stats_summary: Mapping[str, Any]

    def render_message(self) -> str:
        """Return the rich user-facing writer failure message."""
        chain_text = " <- ".join(self.error_chain) if self.error_chain else self.error_type
        queued_peak_mb = (
            float(self.writer_stats_summary.get("queued_bytes_high_water", 0)) / float(1024 ** 2)
        )
        return (
            f"Background HDF5 writer for {self.filename!r} failed: {chain_text} "
            f"[submitted={int(self.submitted_seq)} completed={int(self.completed_seq)} "
            f"durable={int(self.durable_seq)} peak_queue_items="
            f"{int(self.writer_stats_summary.get('queued_items_high_water', 0))} "
            f"peak_queue_mb={queued_peak_mb:.1f} "
            f"durability={str(self.writer_stats_summary.get('durability_mode', 'flush_only'))}]"
        )


@dataclass(frozen=True)
class _StoredArray:
    """Normalized array payload stored in queued HDF5 operations."""

    name: str
    array: np.ndarray
    unit_str: str | None
    kind: str | None


@dataclass(frozen=True)
class _QueuedWriteOperation:
    """Queue-safe write payload owned by the background HDF5 writer."""

    attrs: Mapping[str, Any]
    constants: tuple[_StoredArray, ...]
    iteration: int | None
    batch: tuple[_StoredArray, ...]
    overwrite_constants: bool
    allow_unit_auto_convert: bool
    compression: str | None
    compression_opts: int | None
    shuffle: bool
    chunk_target_bytes: int
    queue_bytes: int
    sequence: int = 0
    group_prefix: str = ""


@dataclass(frozen=True)
class _DeferredPerSatelliteScatter:
    """Compact per-satellite host payload that will be expanded by the writer."""

    compact_values: np.ndarray
    sat_idx_host: np.ndarray
    n_sats_total: int
    dtype: np.dtype
    boresight_active: bool
    n_skycells: int


@dataclass(frozen=True)
class _DeferredSatelliteTimeSeriesScatter:
    """Compact satellite time-series host payload expanded by the writer."""

    compact_values: np.ndarray
    sat_idx_host: np.ndarray
    n_sats_total: int
    dtype: np.dtype
    fill_value: float | int = 0.0


@dataclass(frozen=True)
class _OwnedIterationWriteOperation:
    """Owned iteration batch queued without producer-thread normalization."""

    iteration: int
    batch_items: tuple[tuple[str, Any], ...]
    allow_unit_auto_convert: bool
    compression: str | None
    compression_opts: int | None
    shuffle: bool
    chunk_target_bytes: int
    queue_bytes: int
    group_prefix: str = ""
    sequence: int = 0


_WriterOperation = _QueuedWriteOperation | _OwnedIterationWriteOperation


def _normalize_storage_path(filename: str) -> str:
    """Return a normalized absolute path used as the writer-registry key."""
    return os.path.abspath(os.fspath(filename))


def _coerce_attr_value(value: Any) -> Any:
    """Convert HDF5 attrs to plain Python scalars where practical."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _collect_h5_attrs(obj: h5py.Group | h5py.Dataset | h5py.File) -> dict[str, Any]:
    """Return HDF5 attrs as a plain Python dictionary."""
    return {str(k): _coerce_attr_value(v) for k, v in obj.attrs.items()}


def _dataset_metadata(ds: h5py.Dataset) -> dict[str, Any]:
    """Return lightweight metadata for one dataset without reading its payload."""
    meta = {
        "shape": tuple(int(v) for v in ds.shape),
        "dtype": str(ds.dtype),
        "chunks": None if ds.chunks is None else tuple(int(v) for v in ds.chunks),
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
        "shuffle": bool(getattr(ds, "shuffle", False)),
        "maxshape": None if ds.maxshape is None else tuple(None if v is None else int(v) for v in ds.maxshape),
    }
    meta.update(_collect_h5_attrs(ds))
    return meta


def _collect_group_dataset_metadata(
    group: h5py.Group,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Collect recursive dataset metadata using slash-delimited relative paths."""
    out: dict[str, Any] = {}
    for name, item in group.items():
        item_name = f"{prefix}{name}"
        if isinstance(item, h5py.Dataset):
            out[item_name] = _dataset_metadata(item)
        elif isinstance(item, h5py.Group):
            out.update(_collect_group_dataset_metadata(item, prefix=f"{item_name}/"))
    return out


def _iter_ids_from_file(h5: h5py.File, *, iter_root_key: str = "iter") -> list[int]:
    """Return sorted iteration ids present under the iter root."""
    if iter_root_key not in h5:
        return []
    iter_ids: list[int] = []
    for gname in h5[iter_root_key].keys():
        if gname.startswith("iter_") and len(gname) == len("iter_00000"):
            try:
                iter_ids.append(int(gname.split("_")[1]))
            except Exception:
                continue
    iter_ids.sort()
    return iter_ids


def _normalize_slot_selection(slot_selection: Any) -> slice:
    """Normalize user-facing slot selections to a plain ``slice``."""
    if slot_selection is None:
        return slice(None)
    if isinstance(slot_selection, slice):
        if slot_selection.step not in (None, 1):
            raise ValueError("`slot_selection` does not support step values other than 1.")
        return slot_selection
    if isinstance(slot_selection, tuple) and len(slot_selection) == 2:
        return slice(slot_selection[0], slot_selection[1])
    raise TypeError("`slot_selection` must be None, a slice, or a (start, stop) tuple.")


def _slice_bounds(sel: slice, length: int) -> tuple[int, int]:
    """Return normalized inclusive-exclusive bounds for ``sel`` against ``length``."""
    start, stop, step = sel.indices(length)
    if step != 1:
        raise ValueError("Normalized slot selections must have step=1.")
    stop = max(start, stop)
    return int(start), int(stop)


def _restore_dataset_view(raw: np.ndarray, *, unit_attr: Any, kind: Any, times_as: str) -> Any:
    """Restore a dataset slice into ndarray/Quantity/Time according to attrs."""
    if kind == "time/mjd" and times_as == "time" and Time is not None and u is not None:
        return Time(raw, format="mjd")
    if unit_attr is not None and u is not None:
        try:
            unit = u.Unit(unit_attr)
            return raw * unit
        except Exception:
            pass
    return raw


def _read_dataset_selection(ds: h5py.Dataset, selection: Any = slice(None), *, times_as: str = "time") -> Any:
    """Read one dataset selection and restore units/time metadata."""
    raw = ds[selection]
    return _restore_dataset_view(
        raw,
        unit_attr=ds.attrs.get("unit", None),
        kind=ds.attrs.get("kind", None),
        times_as=times_as,
    )


def _read_dataset(ds: h5py.Dataset, *, times_as: str = "time"):
    """
    Load a dataset back into numpy/Quantity/Time depending on attrs.
    """
    return _read_dataset_selection(ds, slice(None), times_as=times_as)


def _iter_group_name(k: int) -> str:
    return f"iter_{k:05d}"


def _register_writer_atexit() -> None:
    """Register best-effort writer cleanup once per interpreter session."""
    global _WRITER_ATEXIT_REGISTERED
    if _WRITER_ATEXIT_REGISTERED:
        return
    atexit.register(close_writer)
    _WRITER_ATEXIT_REGISTERED = True


def _remember_failed_writer_snapshot(snapshot: _FailedWriterSnapshot) -> None:
    """Persist one writer failure snapshot for later read/flush diagnostics."""
    filename = _normalize_storage_path(snapshot.filename)
    with _WRITER_REGISTRY_LOCK:
        _WRITER_REGISTRY.pop(filename, None)
        _FAILED_WRITER_REGISTRY[filename] = snapshot


def _get_failed_writer_snapshot(filename: str) -> _FailedWriterSnapshot | None:
    """Return the persisted failure snapshot for ``filename`` when available."""
    filename = _normalize_storage_path(filename)
    with _WRITER_REGISTRY_LOCK:
        return _FAILED_WRITER_REGISTRY.get(filename)


def _raise_failed_writer_snapshot(filename: str) -> None:
    """Raise the persisted failure snapshot for ``filename`` if one exists."""
    snapshot = _get_failed_writer_snapshot(filename)
    if snapshot is not None:
        raise RuntimeError(snapshot.render_message())


def _resolve_compression_settings(
    *,
    compression_profile: str,
    compression: Any,
    compression_opts: Any,
) -> tuple[str | None, int | None, bool]:
    """Resolve effective compression and shuffle settings for new datasets."""
    if compression_profile not in _COMPRESSION_PROFILE_SETTINGS:
        raise ValueError(
            f"`compression_profile` must be one of {sorted(_COMPRESSION_PROFILE_SETTINGS)}, "
            f"got {compression_profile!r}."
        )
    profile = _COMPRESSION_PROFILE_SETTINGS[compression_profile]
    if compression is _AUTO_STORAGE_ARG:
        return (
            profile["compression"],
            profile["compression_opts"],
            bool(profile["shuffle"]),
        )

    effective_compression = compression
    if effective_compression is None:
        return None, None, False

    if compression_opts is _AUTO_STORAGE_ARG:
        effective_opts = profile["compression_opts"] if effective_compression == profile["compression"] else None
    else:
        effective_opts = compression_opts
    return effective_compression, effective_opts, bool(profile["shuffle"])


def _slot_first_chunk_shape(
    arr: np.ndarray,
    *,
    target_bytes: int = _DEFAULT_SLOT_CHUNK_TARGET_BYTES,
) -> tuple[int, ...]:
    """Return a slot-first chunk shape tuned for iteration streaming."""
    if arr.ndim == 0:
        return (1,)
    trailing_elems = int(np.prod(arr.shape[1:], dtype=np.int64)) if arr.ndim > 1 else 1
    slot_bytes = max(1, trailing_elems * np.dtype(arr.dtype).itemsize)
    slots_per_chunk = max(1, int(target_bytes) // slot_bytes)
    slots_per_chunk = max(1, min(int(arr.shape[0]), slots_per_chunk))
    return (int(slots_per_chunk),) + tuple(int(v) for v in arr.shape[1:])


def _dataset_create_kwargs(
    arr: np.ndarray,
    *,
    extendable: bool,
    compression: str | None,
    compression_opts: int | None,
    shuffle: bool,
    chunk_target_bytes: int,
) -> dict[str, Any]:
    """Return HDF5 dataset-creation kwargs for one normalized array."""
    kwargs: dict[str, Any] = {
        "dtype": arr.dtype,
    }
    if extendable:
        kwargs["maxshape"] = (None,) + arr.shape[1:]
        kwargs["chunks"] = _slot_first_chunk_shape(arr, target_bytes=chunk_target_bytes)
    elif compression is not None:
        kwargs["chunks"] = _slot_first_chunk_shape(arr, target_bytes=chunk_target_bytes)

    if compression is not None:
        kwargs["compression"] = compression
        if compression_opts is not None:
            kwargs["compression_opts"] = compression_opts
        kwargs["shuffle"] = bool(shuffle)
    return kwargs


def _apply_root_attrs(f: h5py.File, attrs: Mapping[str, Any] | None) -> None:
    """Apply root attrs with basic HDF5-compatible fallback coercion."""
    if not attrs:
        return
    for key, value in attrs.items():
        try:
            f.attrs[key] = value
        except TypeError:
            f.attrs[key] = repr(value)


def _write_constants_group(
    g_const: h5py.Group,
    constants: Iterable[_StoredArray],
    *,
    overwrite_constants: bool,
    compression: str | None,
    compression_opts: int | None,
    shuffle: bool,
    chunk_target_bytes: int,
) -> None:
    """Write or validate constants under ``/const``."""
    for stored in constants:
        arr = stored.array
        if stored.name in g_const:
            ds = g_const[stored.name]
            if overwrite_constants:
                del g_const[stored.name]
            else:
                same_shape = tuple(ds.shape) == tuple(arr.shape)
                same_unit = ds.attrs.get("unit", None) == stored.unit_str
                same_kind = ds.attrs.get("kind", None) == stored.kind
                if same_shape and same_unit and same_kind:
                    if ds.size <= 1_000 and np.array_equal(ds[()], arr):
                        continue
                    continue
                raise ValueError(
                    f"Constant '{stored.name}' already exists with different metadata. "
                    "Set overwrite_constants=True to replace it."
                )

        kwargs = _dataset_create_kwargs(
            arr,
            extendable=False,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            chunk_target_bytes=chunk_target_bytes,
        )
        ds = g_const.create_dataset(stored.name, data=arr, **kwargs)
        ds.attrs["compression_profile"] = "explicit" if compression is not None else "off"
        if stored.unit_str is not None:
            ds.attrs["unit"] = stored.unit_str
        if stored.kind is not None:
            ds.attrs["kind"] = stored.kind


def _write_group_dataset_map(
    group: h5py.Group,
    dataset_map: Mapping[str, Any],
    *,
    compression: str | None,
    compression_opts: int | None,
    shuffle: bool,
    chunk_target_bytes: int,
) -> None:
    """Overwrite datasets in one existing group from a plain mapping."""
    for name, value in dataset_map.items():
        stored = _prepare_stored_array(str(name), value)
        if stored.name in group:
            del group[stored.name]
        kwargs = _dataset_create_kwargs(
            stored.array,
            extendable=False,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            chunk_target_bytes=chunk_target_bytes,
        )
        ds = group.create_dataset(stored.name, data=stored.array, **kwargs)
        ds.attrs["compression_profile"] = "explicit" if compression is not None else "off"
        if stored.unit_str is not None:
            ds.attrs["unit"] = stored.unit_str
        if stored.kind is not None:
            ds.attrs["kind"] = stored.kind


def _write_preaccumulated_families(
    filename: str,
    *,
    families: Mapping[str, Mapping[str, Any]],
    compression: Any = _AUTO_STORAGE_ARG,
    compression_opts: Any = _AUTO_STORAGE_ARG,
    chunk_target_bytes: int = _DEFAULT_SLOT_CHUNK_TARGET_BYTES,
    compression_profile: str = "balanced",
    group_prefix: str = "",
) -> None:
    """Synchronously overwrite preaccumulated family outputs.

    When *group_prefix* is non-empty (e.g. ``"system_0/"``), writes to
    ``/{group_prefix}preaccumulated/`` instead of ``/preaccumulated/``.
    """
    filename = _normalize_storage_path(filename)
    effective_compression, effective_compression_opts, effective_shuffle = _resolve_compression_settings(
        compression_profile=compression_profile,
        compression=compression,
        compression_opts=compression_opts,
    )
    _maybe_flush_pending_writes(filename)
    with _interrupt_ctx(), h5py.File(filename, "a") as f:
        root = f.require_group(f"{group_prefix}preaccumulated")
        for family_name, payload in families.items():
            family_group_name = str(family_name)
            if family_group_name in root:
                del root[family_group_name]
            family_group = root.create_group(family_group_name)
            attrs = payload.get("attrs", {})
            if isinstance(attrs, Mapping):
                _apply_root_attrs(family_group, attrs)
            datasets = payload.get("datasets", {})
            if isinstance(datasets, Mapping):
                _write_group_dataset_map(
                    family_group,
                    datasets,
                    compression=effective_compression,
                    compression_opts=effective_compression_opts,
                    shuffle=effective_shuffle,
                    chunk_target_bytes=chunk_target_bytes,
                )
        f.flush()


def _append_fragments_to_dataset(
    g_iter: h5py.Group,
    *,
    name: str,
    fragments: list[_StoredArray],
    allow_unit_auto_convert: bool,
    compression: str | None,
    compression_opts: int | None,
    shuffle: bool,
    chunk_target_bytes: int,
) -> None:
    """Append one or more fragments to a dataset with a single resize."""
    if not fragments:
        return

    total_rows = int(sum(fragment.array.shape[0] for fragment in fragments))
    if name in g_iter:
        ds = g_iter[name]
        ds_unit = ds.attrs.get("unit", None)
        old_n = int(ds.shape[0])
        new_n = old_n + total_rows
        ds.resize((new_n,) + ds.shape[1:])
        cursor = old_n
        for fragment in fragments:
            arr = fragment.array
            if ds_unit is None and fragment.unit_str is not None:
                raise ValueError(f"Cannot append unit '{fragment.unit_str}' to existing unitless dataset '{name}'.")
            if ds_unit is not None and fragment.unit_str is None:
                raise ValueError(f"Cannot append unitless data to existing unit '{ds_unit}' in dataset '{name}'.")
            if (ds_unit is not None) and (fragment.unit_str is not None) and (ds_unit != fragment.unit_str):
                if allow_unit_auto_convert and (u is not None) and _units_convertible(u.Unit(fragment.unit_str), u.Unit(ds_unit)):
                    arr = _convert_array_unit(arr, u.Unit(fragment.unit_str), u.Unit(ds_unit))
                else:
                    raise ValueError(
                        f"Units differ for '{name}': incoming {fragment.unit_str} vs existing {ds_unit}."
                    )
            next_cursor = cursor + int(arr.shape[0])
            ds[cursor:next_cursor, ...] = arr
            cursor = next_cursor
        if (ds.attrs.get("kind", None) is None) and (fragments[0].kind is not None):
            ds.attrs["kind"] = fragments[0].kind
        return

    template = fragments[0]
    total_shape = (total_rows,) + template.array.shape[1:]
    empty = np.empty(total_shape, dtype=template.array.dtype)
    kwargs = _dataset_create_kwargs(
        empty,
        extendable=True,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=shuffle,
        chunk_target_bytes=chunk_target_bytes,
    )
    ds = g_iter.create_dataset(name, shape=total_shape, **kwargs)
    cursor = 0
    for fragment in fragments:
        next_cursor = cursor + int(fragment.array.shape[0])
        ds[cursor:next_cursor, ...] = fragment.array
        cursor = next_cursor
    ds.attrs["compression_profile"] = "explicit" if compression is not None else "off"
    if template.unit_str is not None:
        ds.attrs["unit"] = template.unit_str
    if template.kind is not None:
        ds.attrs["kind"] = template.kind


def _append_iteration_batch(
    g_iter: h5py.Group,
    batch: Iterable[_StoredArray],
    *,
    allow_unit_auto_convert: bool,
    compression: str | None,
    compression_opts: int | None,
    shuffle: bool,
    chunk_target_bytes: int,
) -> None:
    """Append one normalized iteration batch without queue-level coalescing."""
    for stored in batch:
        _append_fragments_to_dataset(
            g_iter,
            name=stored.name,
            fragments=[stored],
            allow_unit_auto_convert=allow_unit_auto_convert,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            chunk_target_bytes=chunk_target_bytes,
        )


def _apply_write_operation(f: h5py.File, op: _QueuedWriteOperation) -> None:
    """Apply one prepared write operation to an open HDF5 file."""
    _apply_root_attrs(f, op.attrs)
    if op.constants:
        g_const = f.require_group("const")
        _write_constants_group(
            g_const,
            op.constants,
            overwrite_constants=op.overwrite_constants,
            compression=op.compression,
            compression_opts=op.compression_opts,
            shuffle=op.shuffle,
            chunk_target_bytes=op.chunk_target_bytes,
        )
    if op.iteration is not None and op.batch:
        g_iter_root = f.require_group(f"{op.group_prefix}iter")
        g_iter = g_iter_root.require_group(_iter_group_name(int(op.iteration)))
        _append_iteration_batch(
            g_iter,
            op.batch,
            allow_unit_auto_convert=op.allow_unit_auto_convert,
            compression=op.compression,
            compression_opts=op.compression_opts,
            shuffle=op.shuffle,
            chunk_target_bytes=op.chunk_target_bytes,
        )


def _coalescing_signature(op: _QueuedWriteOperation) -> tuple[Any, ...] | None:
    """Return the grouping signature for append-only queued operations."""
    if op.attrs or op.constants or op.iteration is None or not op.batch:
        return None
    return (
        str(op.group_prefix),
        int(op.iteration),
        tuple((stored.name, stored.unit_str, stored.kind, str(stored.array.dtype), stored.array.shape[1:]) for stored in op.batch),
        op.allow_unit_auto_convert,
        op.compression,
        op.compression_opts,
        bool(op.shuffle),
        int(op.chunk_target_bytes),
    )


def _apply_coalesced_write_batch(f: h5py.File, ops: list[_QueuedWriteOperation]) -> None:
    """Apply queued operations while coalescing compatible append-only runs."""
    pending_group: list[_QueuedWriteOperation] = []
    pending_signature: tuple[Any, ...] | None = None

    def _flush_pending_group() -> None:
        nonlocal pending_group, pending_signature
        if not pending_group:
            return
        first = pending_group[0]
        g_iter_root = f.require_group(f"{first.group_prefix}iter")
        g_iter = g_iter_root.require_group(_iter_group_name(int(first.iteration)))
        batch_names = [stored.name for stored in first.batch]
        for name in batch_names:
            fragments: list[_StoredArray] = []
            for op in pending_group:
                fragments.extend([stored for stored in op.batch if stored.name == name])
            _append_fragments_to_dataset(
                g_iter,
                name=name,
                fragments=fragments,
                allow_unit_auto_convert=first.allow_unit_auto_convert,
                compression=first.compression,
                compression_opts=first.compression_opts,
                shuffle=first.shuffle,
                chunk_target_bytes=first.chunk_target_bytes,
            )
        pending_group = []
        pending_signature = None

    for op in ops:
        signature = _coalescing_signature(op)
        if signature is None:
            _flush_pending_group()
            _apply_write_operation(f, op)
            continue
        if pending_signature is None:
            pending_group = [op]
            pending_signature = signature
            continue
        if signature == pending_signature:
            pending_group.append(op)
            continue
        _flush_pending_group()
        pending_group = [op]
        pending_signature = signature

    _flush_pending_group()


def _estimate_owned_write_value_bytes(value: Any) -> int:
    """Estimate host-queue pressure for one owned async write value."""
    if isinstance(value, _DeferredPerSatelliteScatter):
        dtype = np.dtype(value.dtype)
        compact_bytes = int(np.asarray(value.compact_values).nbytes)
        full_shape = (
            (int(np.asarray(value.compact_values).shape[0]), 1, int(value.n_sats_total), int(value.n_skycells))
            if value.boresight_active
            else (int(np.asarray(value.compact_values).shape[0]), int(value.n_sats_total))
        )
        full_bytes = int(np.prod(full_shape, dtype=np.int64)) * int(dtype.itemsize)
        return compact_bytes + full_bytes + int(np.asarray(value.sat_idx_host).nbytes)
    if isinstance(value, _DeferredSatelliteTimeSeriesScatter):
        dtype = np.dtype(value.dtype)
        compact_bytes = int(np.asarray(value.compact_values).nbytes)
        full_shape = (
            int(np.asarray(value.compact_values).shape[0]),
            int(value.n_sats_total),
        )
        full_bytes = int(np.prod(full_shape, dtype=np.int64)) * int(dtype.itemsize)
        return compact_bytes + full_bytes + int(np.asarray(value.sat_idx_host).nbytes)
    arr, _, _ = _to_array_unit_kind(value)
    arr = _ensure_row_first(np.asarray(arr))
    return int(arr.nbytes)


def _resolve_owned_write_value(value: Any) -> Any:
    """Resolve one owned async value to the final host array written to HDF5."""
    if isinstance(value, _DeferredPerSatelliteScatter):
        return _scatter_compact_per_satellite_host(
            np.asarray(value.compact_values),
            np.asarray(value.sat_idx_host, dtype=np.int32),
            n_sats_total=int(value.n_sats_total),
            dtype=np.dtype(value.dtype),
            boresight_active=bool(value.boresight_active),
            n_skycells=int(value.n_skycells),
        )
    if isinstance(value, _DeferredSatelliteTimeSeriesScatter):
        return _scatter_compact_satellite_time_series_host(
            np.asarray(value.compact_values),
            np.asarray(value.sat_idx_host, dtype=np.int32),
            n_sats_total=int(value.n_sats_total),
            dtype=np.dtype(value.dtype),
            fill_value=value.fill_value,
        )
    return value


def _prepare_stored_array(name: str, value: Any) -> _StoredArray:
    """Normalize one public write value to a queue-safe array payload."""
    arr, unit, kind = _to_array_unit_kind(value)
    arr = _ensure_row_first(arr)
    arr = np.asarray(arr)

    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    elif (not arr.flags.owndata) or (not arr.flags.writeable):
        arr = arr.copy()

    return _StoredArray(
        name=str(name),
        array=arr,
        unit_str=_to_unit_str(unit),
        kind=kind,
    )


def _prepare_owned_stored_array(name: str, value: Any) -> _StoredArray:
    """Normalize one owned async write value on the writer thread."""
    return _prepare_stored_array(name, _resolve_owned_write_value(value))


def _prepare_write_operation(
    *,
    attrs: Mapping[str, Any] | None,
    constants: Mapping[str, Any] | None,
    iteration: int | None,
    batch: Mapping[str, Any],
    overwrite_constants: bool,
    allow_unit_auto_convert: bool,
    compression_profile: str,
    compression: Any,
    compression_opts: Any,
    chunk_target_bytes: int,
) -> _QueuedWriteOperation:
    """Normalize public write arguments to a queue-safe write operation."""
    effective_compression, effective_compression_opts, effective_shuffle = _resolve_compression_settings(
        compression_profile=compression_profile,
        compression=compression,
        compression_opts=compression_opts,
    )
    constants_payload = tuple(_prepare_stored_array(name, value) for name, value in (constants or {}).items())
    batch_payload = tuple(_prepare_stored_array(name, value) for name, value in batch.items())
    queue_bytes = int(
        sum(stored.array.nbytes for stored in constants_payload)
        + sum(stored.array.nbytes for stored in batch_payload)
    )
    return _QueuedWriteOperation(
        attrs=dict(attrs or {}),
        constants=constants_payload,
        iteration=None if iteration is None else int(iteration),
        batch=batch_payload,
        overwrite_constants=bool(overwrite_constants),
        allow_unit_auto_convert=bool(allow_unit_auto_convert),
        compression=effective_compression,
        compression_opts=effective_compression_opts,
        shuffle=bool(effective_shuffle),
        chunk_target_bytes=max(1, int(chunk_target_bytes)),
        queue_bytes=queue_bytes,
    )


def _prepare_owned_iteration_write_operation(
    op: _OwnedIterationWriteOperation,
) -> _QueuedWriteOperation:
    """Convert an owned async iteration op to a normal prepared write op."""
    batch_payload = tuple(
        _prepare_owned_stored_array(name, value)
        for name, value in op.batch_items
    )
    return _QueuedWriteOperation(
        attrs={},
        constants=(),
        iteration=int(op.iteration),
        batch=batch_payload,
        overwrite_constants=False,
        allow_unit_auto_convert=bool(op.allow_unit_auto_convert),
        compression=op.compression,
        compression_opts=op.compression_opts,
        shuffle=bool(op.shuffle),
        chunk_target_bytes=max(1, int(op.chunk_target_bytes)),
        queue_bytes=int(op.queue_bytes),
        sequence=int(op.sequence),
        group_prefix=str(op.group_prefix),
    )


def _normalize_h5_vfd_handle(value: Any) -> int | None:
    """Return the first usable integer VFD handle exposed by HDF5, if any."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (tuple, list)):
        for item in value:
            handle = _normalize_h5_vfd_handle(item)
            if handle is not None:
                return handle
    return None


def _durable_flush_open_h5_file(h5: h5py.File) -> tuple[str, float]:
    """Flush one open HDF5 file and try to make it OS-durable when possible."""
    flush_t0 = perf_counter()
    h5.flush()
    mode = "flush_only"
    try:
        get_vfd_handle = getattr(h5.id, "get_vfd_handle", None)
        raw_handle = get_vfd_handle() if callable(get_vfd_handle) else None
        handle = _normalize_h5_vfd_handle(raw_handle)
        if handle is not None and hasattr(os, "fsync"):
            os.fsync(handle)
            mode = "fsync"
    except Exception:
        mode = "flush_only"
    return mode, float(perf_counter() - flush_t0)


class _AsyncH5Writer:
    """Single-writer background thread responsible for one HDF5 file."""

    def __init__(
        self,
        filename: str,
        *,
        queue_max_items: int = _DEFAULT_WRITER_QUEUE_MAX_ITEMS,
        queue_max_bytes: int = _DEFAULT_WRITER_QUEUE_MAX_BYTES,
        max_ops_per_cycle: int = _DEFAULT_WRITER_CYCLE_MAX_ITEMS,
        max_bytes_per_cycle: int = _DEFAULT_WRITER_CYCLE_MAX_BYTES,
    ) -> None:
        self.filename = _normalize_storage_path(filename)
        self.queue_max_items = max(1, int(queue_max_items))
        self.queue_max_bytes = max(1, int(queue_max_bytes))
        self.max_ops_per_cycle = max(1, int(max_ops_per_cycle))
        self.max_bytes_per_cycle = max(1, int(max_bytes_per_cycle))
        self._condition = threading.Condition()
        self._queue: Deque[_WriterOperation] = deque()
        self._queued_bytes = 0
        self._queued_items_high_water = 0
        self._queued_bytes_high_water = 0
        self._submitted_seq = 0
        self._completed_seq = 0
        self._durable_seq = 0
        self._durable_requested_seq = 0
        self._error: BaseException | None = None
        self._failed_snapshot: _FailedWriterSnapshot | None = None
        self._closed = False
        self._prepare_elapsed_total = 0.0
        self._apply_elapsed_total = 0.0
        self._submit_wait_elapsed_total = 0.0
        self._flush_count = 0
        self._writer_cycle_count = 0
        self._writer_cycle_items_high_water = 0
        self._writer_cycle_bytes_high_water = 0
        self._durable_flush_count = 0
        self._durable_elapsed_total = 0.0
        self._durability_mode = "flush_only"
        self._thread = threading.Thread(
            target=self._run,
            name=f"scepter-h5-writer-{os.path.basename(self.filename)}",
            daemon=True,
        )
        self._thread.start()

    def _stats_snapshot_locked(self) -> dict[str, Any]:
        """Return cumulative writer stats while the condition lock is held."""
        return {
            "queued_items": int(len(self._queue)),
            "queued_bytes": int(self._queued_bytes),
            "queued_items_high_water": int(self._queued_items_high_water),
            "queued_bytes_high_water": int(self._queued_bytes_high_water),
            "submitted_seq": int(self._submitted_seq),
            "completed_seq": int(self._completed_seq),
            "durable_seq": int(self._durable_seq),
            "prepare_elapsed_total": float(self._prepare_elapsed_total),
            "apply_elapsed_total": float(self._apply_elapsed_total),
            "submit_wait_elapsed_total": float(self._submit_wait_elapsed_total),
            "flush_count": int(self._flush_count),
            "writer_cycle_count": int(self._writer_cycle_count),
            "writer_cycle_items_high_water": int(self._writer_cycle_items_high_water),
            "writer_cycle_bytes_high_water": int(self._writer_cycle_bytes_high_water),
            "durable_flush_count": int(self._durable_flush_count),
            "durable_elapsed_total": float(self._durable_elapsed_total),
            "durability_mode": str(self._durability_mode),
        }

    def _build_failed_snapshot_locked(
        self,
        exc: BaseException,
    ) -> _FailedWriterSnapshot:
        """Build one immutable failure snapshot while holding the condition lock."""
        error_chain = _exception_chain_messages(exc)
        return _FailedWriterSnapshot(
            filename=str(self.filename),
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_chain=error_chain,
            submitted_seq=int(self._submitted_seq),
            completed_seq=int(self._completed_seq),
            durable_seq=int(self._durable_seq),
            writer_stats_summary=self._stats_snapshot_locked(),
        )

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            snapshot = (
                self._failed_snapshot
                if self._failed_snapshot is not None
                else self._build_failed_snapshot_locked(self._error)
            )
            raise RuntimeError(snapshot.render_message()) from self._error

    def reconfigure_limits(
        self,
        *,
        queue_max_items: int | None = None,
        queue_max_bytes: int | None = None,
    ) -> None:
        """Update async backpressure limits for future producer submits."""
        with self._condition:
            if queue_max_items is not None:
                self.queue_max_items = max(1, int(queue_max_items))
            if queue_max_bytes is not None:
                self.queue_max_bytes = max(1, int(queue_max_bytes))
            self._condition.notify_all()

    def stats_snapshot(self) -> dict[str, Any]:
        """Return cumulative writer stats without blocking the producer."""
        with self._condition:
            return self._stats_snapshot_locked()

    def failure_snapshot(self) -> _FailedWriterSnapshot:
        """Return the stored immutable failure snapshot for this writer."""
        with self._condition:
            if self._error is None:
                raise RuntimeError(f"Background HDF5 writer for {self.filename!r} has not failed.")
            if self._failed_snapshot is None:
                self._failed_snapshot = self._build_failed_snapshot_locked(self._error)
            return self._failed_snapshot

    def submit(self, op: _WriterOperation) -> int:
        """Enqueue one prepared write operation with bounded backpressure."""
        with self._condition:
            self._raise_if_error_locked()
            if self._closed:
                raise RuntimeError(f"Background HDF5 writer for {self.filename!r} is closed.")

            wait_t0 = perf_counter()
            while True:
                self._raise_if_error_locked()
                over_items = len(self._queue) >= self.queue_max_items
                over_bytes = (
                    self._queued_bytes > 0
                    and (self._queued_bytes + op.queue_bytes) > self.queue_max_bytes
                )
                if not (over_items or over_bytes):
                    break
                self._condition.wait()
            wait_elapsed = perf_counter() - wait_t0
            if wait_elapsed > 0.0:
                self._submit_wait_elapsed_total += float(wait_elapsed)

            self._submitted_seq += 1
            op = replace(op, sequence=self._submitted_seq)
            self._queue.append(op)
            self._queued_bytes += op.queue_bytes
            self._queued_items_high_water = max(
                int(self._queued_items_high_water),
                int(len(self._queue)),
            )
            self._queued_bytes_high_water = max(
                int(self._queued_bytes_high_water),
                int(self._queued_bytes),
            )
            self._condition.notify_all()
            return int(op.sequence)

    def flush(self) -> None:
        """Block until all queued writes submitted so far are durable."""
        with self._condition:
            target = self._submitted_seq
            self._durable_requested_seq = max(
                int(self._durable_requested_seq),
                int(target),
            )
            self._condition.notify_all()
            while self._durable_seq < target:
                self._raise_if_error_locked()
                self._condition.wait()
            self._raise_if_error_locked()

    def close(self) -> None:
        """Flush pending writes, then stop the writer thread."""
        self.flush()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()
        with self._condition:
            self._raise_if_error_locked()

    def _dequeue_cycle_locked(self) -> tuple[list[_WriterOperation], int]:
        """Pop a bounded writer cycle from the queue."""
        ops: list[_WriterOperation] = []
        cycle_bytes = 0
        while self._queue:
            next_op = self._queue[0]
            next_bytes = max(1, int(next_op.queue_bytes))
            if ops and (
                len(ops) >= self.max_ops_per_cycle
                or (cycle_bytes + next_bytes) > self.max_bytes_per_cycle
            ):
                break
            ops.append(self._queue.popleft())
            cycle_bytes += next_bytes
            self._queued_bytes = max(0, int(self._queued_bytes) - next_bytes)
        return ops, int(cycle_bytes)

    def _run(self) -> None:
        """Writer-thread main loop."""
        try:
            with _interrupt_ctx(), h5py.File(self.filename, "a") as h5:
                while True:
                    ops: list[_WriterOperation] = []
                    cycle_bytes = 0
                    with self._condition:
                        while (
                            not self._queue
                            and not self._closed
                            and self._durable_requested_seq <= self._durable_seq
                        ):
                            self._condition.wait()
                        if (
                            not self._queue
                            and self._closed
                            and self._durable_requested_seq <= self._durable_seq
                        ):
                            self._completed_seq = self._submitted_seq
                            self._condition.notify_all()
                            return
                        if self._queue:
                            ops, cycle_bytes = self._dequeue_cycle_locked()
                            self._condition.notify_all()

                    if ops:
                        prepare_t0 = perf_counter()
                        prepared_ops = [
                            _prepare_owned_iteration_write_operation(op)
                            if isinstance(op, _OwnedIterationWriteOperation)
                            else op
                            for op in ops
                        ]
                        prepare_elapsed = perf_counter() - prepare_t0
                        apply_t0 = perf_counter()
                        _apply_coalesced_write_batch(h5, prepared_ops)
                        h5.flush()
                        apply_elapsed = perf_counter() - apply_t0
                        max_sequence = max((op.sequence for op in ops), default=0)
                        with self._condition:
                            self._prepare_elapsed_total += float(prepare_elapsed)
                            self._apply_elapsed_total += float(apply_elapsed)
                            self._flush_count += 1
                            self._writer_cycle_count += 1
                            self._writer_cycle_items_high_water = max(
                                int(self._writer_cycle_items_high_water),
                                int(len(ops)),
                            )
                            self._writer_cycle_bytes_high_water = max(
                                int(self._writer_cycle_bytes_high_water),
                                int(cycle_bytes),
                            )
                            self._completed_seq = max(self._completed_seq, int(max_sequence))
                            self._condition.notify_all()

                    durable_target = 0
                    with self._condition:
                        requested_seq = int(self._durable_requested_seq)
                        if requested_seq > self._durable_seq and self._completed_seq >= requested_seq:
                            durable_target = requested_seq

                    if durable_target > 0:
                        durability_mode, durable_elapsed = _durable_flush_open_h5_file(h5)
                        with self._condition:
                            self._durable_flush_count += 1
                            self._durable_elapsed_total += float(durable_elapsed)
                            if durability_mode == "fsync":
                                self._durability_mode = "fsync"
                            elif self._durability_mode != "fsync":
                                self._durability_mode = "flush_only"
                            self._durable_seq = max(self._durable_seq, int(durable_target))
                            self._condition.notify_all()
        except BaseException as exc:  # pragma: no cover - exercised indirectly
            with self._condition:
                self._error = exc
                self._failed_snapshot = self._build_failed_snapshot_locked(exc)
                self._condition.notify_all()


def _get_writer_defaults(filename: str) -> dict[str, Any]:
    """Return per-file writer defaults or the module defaults if none were configured."""
    return {
        "write_mode": _WRITER_DEFAULTS.get(filename, {}).get("write_mode", "async"),
        "writer_queue_max_items": int(_WRITER_DEFAULTS.get(filename, {}).get("writer_queue_max_items", _DEFAULT_WRITER_QUEUE_MAX_ITEMS)),
        "writer_queue_max_bytes": int(_WRITER_DEFAULTS.get(filename, {}).get("writer_queue_max_bytes", _DEFAULT_WRITER_QUEUE_MAX_BYTES)),
    }


def _ensure_async_writer(
    filename: str,
    *,
    writer_queue_max_items: int | None = None,
    writer_queue_max_bytes: int | None = None,
) -> _AsyncH5Writer:
    """Return the single background writer registered for ``filename``."""
    filename = _normalize_storage_path(filename)
    _register_writer_atexit()
    with _WRITER_REGISTRY_LOCK:
        writer = _WRITER_REGISTRY.get(filename)
        if writer is None:
            _FAILED_WRITER_REGISTRY.pop(filename, None)
            defaults = _get_writer_defaults(filename)
            writer = _AsyncH5Writer(
                filename,
                queue_max_items=defaults["writer_queue_max_items"] if writer_queue_max_items is None else writer_queue_max_items,
                queue_max_bytes=defaults["writer_queue_max_bytes"] if writer_queue_max_bytes is None else writer_queue_max_bytes,
            )
            _WRITER_REGISTRY[filename] = writer
        else:
            writer.reconfigure_limits(
                queue_max_items=writer_queue_max_items,
                queue_max_bytes=writer_queue_max_bytes,
            )
        return writer


def _get_registered_writer(filename: str) -> _AsyncH5Writer | None:
    """Return the active writer instance for ``filename`` when one exists."""
    filename = _normalize_storage_path(filename)
    with _WRITER_REGISTRY_LOCK:
        return _WRITER_REGISTRY.get(filename)


def _get_writer_stats_snapshot(filename: str) -> dict[str, Any]:
    """Return lightweight writer stats for ``filename`` or zeros when inactive."""
    writer = _get_registered_writer(filename)
    if writer is not None:
        return writer.stats_snapshot()
    failed_snapshot = _get_failed_writer_snapshot(filename)
    if failed_snapshot is not None:
        return dict(failed_snapshot.writer_stats_summary)
    return _empty_writer_stats_summary()


def _maybe_flush_pending_writes(filename: str) -> None:
    """Flush queued writes for ``filename`` when a local writer is active."""
    filename = _normalize_storage_path(filename)
    with _WRITER_REGISTRY_LOCK:
        writer = _WRITER_REGISTRY.get(filename)
        failed_snapshot = _FAILED_WRITER_REGISTRY.get(filename)
    if writer is not None:
        try:
            writer.flush()
        except BaseException:
            _remember_failed_writer_snapshot(writer.failure_snapshot())
            raise
        return
    if failed_snapshot is not None:
        raise RuntimeError(failed_snapshot.render_message())


# -----------------------------------------------------------------------------
# Multi-system scheduling and orchestration helpers
# -----------------------------------------------------------------------------


def plan_multi_system_schedule(
    *,
    system_configs: list[dict[str, Any]],
    shared_schedule_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Plan the iteration schedule for a multi-system simulation.

    Takes the worst-case parameters across all systems (max satellite count,
    max Nco, max Nbeam) and passes them to the single-system scheduler.  The
    result is a schedule that fits the largest system; smaller systems will
    execute faster within the same batch dimensions.

    Parameters
    ----------
    system_configs : list[dict]
        Per-system config dicts.  Each must contain at least:
        ``n_sats_total``, ``nco``, ``nbeam``, ``visible_satellite_est``.
    shared_schedule_kwargs : dict
        Keyword arguments for ``_plan_direct_epfd_iteration_schedule`` that
        are shared across all systems (session, observer, grid, budgets, etc.).

    Returns
    -------
    dict[str, Any]
        Schedule dictionary with ``bulk_timesteps``, ``cell_observer_chunk``,
        etc., valid for all systems.
    """
    if not system_configs:
        raise ValueError("At least one system config is required.")
    if len(system_configs) == 1:
        # Single-system fast path: no worst-case overhead
        kw = dict(shared_schedule_kwargs)
        kw.update(system_configs[0])
        return _plan_direct_epfd_iteration_schedule(**kw)

    # Worst-case across all systems
    max_sats = max(int(s.get("n_sats_total", 1)) for s in system_configs)
    max_nco = max(int(s.get("nco", 1)) for s in system_configs)
    max_nbeam = max(int(s.get("nbeam", 1)) for s in system_configs)
    max_visible = max(int(s.get("visible_satellite_est", max_sats)) for s in system_configs)
    n_systems = len(system_configs)

    kw = dict(shared_schedule_kwargs)
    kw.update({
        "n_sats_total": max_sats,
        "nco": max_nco,
        "nbeam": max_nbeam,
        "visible_satellite_est": max_visible,
    })
    schedule = _plan_direct_epfd_iteration_schedule(**kw)

    # Reduce bulk_timesteps to account for N systems per batch.
    # Each system processes sequentially within a batch, so effective
    # per-batch GPU time scales with N.  Halve bulk_timesteps for every
    # doubling of system count (empirical heuristic, refined by warm-up).
    if n_systems > 1:
        scale = max(1, int(schedule.get("bulk_timesteps", 1)) // n_systems)
        schedule["bulk_timesteps"] = max(1, scale)
        schedule["multi_system_count"] = n_systems
        schedule["multi_system_max_sats"] = max_sats

    return schedule


def run_gpu_multi_system_epfd(
    *,
    system_run_kwargs: list[dict[str, Any]],
    shared_run_kwargs: dict[str, Any],
    system_names: list[str] | None = None,
    output_system_groups: list[dict[str, Any]] | None = None,
    progress_callback: Any | None = None,
    cancel_controller: Any | None = None,
) -> dict[str, Any]:
    """Run a multi-system EPFD/PFD simulation.

    For single-system projects (``len(system_run_kwargs) == 1``), this
    delegates directly to :func:`run_gpu_direct_epfd` with zero overhead.

    For N > 1 systems, each system is run sequentially via
    :func:`run_gpu_direct_epfd` with per-system storage filenames, then
    results are merged into a combined HDF5 with ``/system_N/`` groups
    and a ``/combined/`` section that sums EPFD/PFD in linear power.

    Parameters
    ----------
    system_run_kwargs : list[dict]
        Per-system keyword arguments for ``run_gpu_direct_epfd``.
    shared_run_kwargs : dict
        Shared keyword arguments (observer, time, GPU, atmosphere, etc.).
    system_names : list[str], optional
        Human-readable system names for HDF5 metadata.
    progress_callback : callable, optional
        Progress reporting callback.
    cancel_controller : object, optional
        Cancellation controller.

    Returns
    -------
    dict[str, Any]
        Result dictionary with ``storage_filename`` pointing to the
        combined HDF5 file.
    """
    if not system_run_kwargs:
        raise ValueError("At least one system is required.")

    # Build cancel_callback from controller.
    cancel_cb: Callable[[], str | None] | None = None
    if cancel_controller is not None:
        cancel_cb = (
            cancel_controller.current_mode
            if hasattr(cancel_controller, "current_mode")
            else lambda: None
        )

    if len(system_run_kwargs) == 1:
        merged = dict(shared_run_kwargs)
        merged.update(system_run_kwargs[0])
        if progress_callback is not None:
            merged["progress_callback"] = progress_callback
        if cancel_cb is not None and "cancel_callback" not in merged:
            merged["cancel_callback"] = cancel_cb
        return run_gpu_direct_epfd(**merged)

    # --- Per-batch interleaving path (N > 1 systems) ---
    # Use system 0 as the primary system (its parameters go into
    # run_gpu_direct_epfd's positional kwargs) and pass systems 1..N-1
    # as the ``systems`` list for per-batch interleaving.
    names = list(system_names or [f"System {i + 1}" for i in range(len(system_run_kwargs))])
    merged = dict(shared_run_kwargs)
    merged.update(system_run_kwargs[0])
    if progress_callback is not None:
        merged["progress_callback"] = progress_callback
    if cancel_cb is not None and "cancel_callback" not in merged:
        merged["cancel_callback"] = cancel_cb

    # Pass ALL system configs (including system 0) so the batch loop
    # knows about all systems.  Inject system names into the per-system dicts.
    systems_with_names = []
    for si, skw in enumerate(system_run_kwargs):
        skw_copy = dict(skw)
        skw_copy["_system_name"] = names[si] if si < len(names) else f"System {si + 1}"
        systems_with_names.append(skw_copy)
    merged["systems"] = systems_with_names
    if output_system_groups:
        merged["output_system_groups"] = output_system_groups

    # Fix storage_attrs to reflect ALL systems, not just system 0.
    # Remove per-system fields from root — they belong in /system_N/ attrs.
    if "storage_attrs" in merged and isinstance(merged["storage_attrs"], dict):
        merged["storage_attrs"] = dict(merged["storage_attrs"])
        merged["storage_attrs"]["system_count"] = len(system_run_kwargs)
        merged["storage_attrs"]["system_names"] = ";".join(names)
        _per_system_attr_keys = {
            "nco", "nbeam", "active_cell_count", "selection_strategy",
            "power_input_quantity", "power_input_basis",
            "power_input_value", "power_input_value_unit",
            "pfd0_dbw_m2_mhz",
            "target_pfd_dbw_m2_mhz", "target_pfd_dbw_m2_channel",
            "satellite_eirp_dbw_mhz", "satellite_eirp_dbw_channel",
            "satellite_ptx_dbw_mhz", "satellite_ptx_dbw_channel",
            "reuse_factor", "ras_anchor_reuse_slot",
            "unwanted_emission_mask_preset",
            "service_band_start_mhz", "service_band_stop_mhz",
            "wavelength_m", "cell_activity_mode",
            "channel_groups_per_cell", "multi_group_power_policy",
            "split_total_group_denominator_mode",
            "pre_ras_cell_count", "ras_guard_angle_deg",
            "enabled_channel_count", "enabled_channel_indices",
            "disabled_channel_indices", "max_groups_per_cell",
            "leftover_spectrum_mhz", "spectral_slab",
            "ras_anchor_reuse_slot",
            "spectral_integration_cutoff_basis",
            "spectral_integration_cutoff_percent",
            "tx_reference_mode", "tx_reference_point_count",
            "tx_reference_frequency_mhz_effective",
            "ras_reference_frequency_mhz_effective",
        }
        for _psk in _per_system_attr_keys:
            merged["storage_attrs"].pop(_psk, None)

    if progress_callback is not None:
        progress_callback({
            "event": "multi_system_start",
            "system_index": 0,
            "system_name": names[0],
            "system_count": len(system_run_kwargs),
            "interleaving": True,
        })

    result = run_gpu_direct_epfd(**merged)
    result["system_count"] = len(system_run_kwargs)
    result["multi_system_interleaved"] = True
    return result


def _merge_multi_system_hdf5(
    *,
    combined_filename: str,
    per_system_files: list[str],
    system_names: list[str],
    cancelled: bool = False,
) -> None:
    """Merge per-system HDF5 files into a combined output.

    Creates ``/system_N/`` groups with per-system raw datasets and
    a root-level section copied from system 0 (first system) as the
    default/combined baseline.

    The root ``/preaccumulated/`` and ``/iter/`` data come from system 0
    when no system filter is applied.  Selecting a specific system in the
    GUI reads from ``/system_N/preaccumulated/`` instead.
    """
    if not per_system_files:
        return
    with h5py.File(combined_filename, "w") as out_f:
        # Copy root attrs from first system file
        with h5py.File(per_system_files[0], "r") as src_f:
            for attr_name, attr_value in src_f.attrs.items():
                out_f.attrs[attr_name] = attr_value
            # Copy constants
            if "const" in src_f:
                src_f.copy("const", out_f)
            # Copy preaccumulated from first system as combined baseline
            if "preaccumulated" in src_f:
                src_f.copy("preaccumulated", out_f)
            # Copy iterations from first system as combined baseline
            if "iter" in src_f:
                src_f.copy("iter", out_f)

        # Add multi-system metadata
        out_f.attrs["system_count"] = len(per_system_files)
        out_f.attrs["system_names"] = ";".join(system_names)
        out_f.attrs["multi_system_cancelled"] = bool(cancelled)
        out_f.attrs["multi_system_combined_note"] = (
            "Root-level preaccumulated and iter data are from system 0. "
            "Per-system data are under /system_N/ groups. "
            "True combined power (linear sum) requires per-batch interleaving "
            "which is not yet implemented."
        )

        # Copy each system's data into /system_N/ groups
        for sys_idx, (sys_file, sys_name) in enumerate(zip(per_system_files, system_names)):
            sys_group_name = f"system_{sys_idx}"
            try:
                with h5py.File(sys_file, "r") as src_f:
                    sys_group = out_f.create_group(sys_group_name)
                    sys_group.attrs["system_name"] = str(sys_name)
                    sys_group.attrs["system_index"] = int(sys_idx)
                    # Copy root attrs
                    for attr_name, attr_value in src_f.attrs.items():
                        sys_group.attrs[attr_name] = attr_value
                    # Copy constants
                    if "const" in src_f:
                        src_f.copy("const", sys_group, name="const")
                    # Copy iterations
                    if "iter" in src_f:
                        src_f.copy("iter", sys_group, name="iter")
                    # Copy preaccumulated
                    if "preaccumulated" in src_f:
                        src_f.copy("preaccumulated", sys_group, name="preaccumulated")
            except Exception:
                pass  # Skip systems whose files are missing/corrupt


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def describe_data(
    filename: str,
    *,
    iter_selection: Iterable[int] | None = None,
    var_selection: Iterable[str] | None = None,
    slot_selection: Any = None,
    sync_pending_writes: bool = True,
) -> dict[str, Any]:
    """
    Inspect HDF5 simulation results without materializing dataset payloads.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file produced by :func:`write_data`.
    iter_selection : iterable of int, optional
        Iteration ids to include.
    var_selection : iterable of str, optional
        Dataset names to include under each selected iteration.
    slot_selection : None, slice, or tuple, optional
        Per-iteration slot selection applied only for the reported selected slot
        counts. Dataset metadata still describe the full stored dataset.
    sync_pending_writes : bool, optional
        Flush any active local background writer for ``filename`` before
        opening the file.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary with ``attrs``, ``const``, ``preaccumulated``, and
        ``iter`` sections.
    """
    filename = _normalize_storage_path(filename)
    if sync_pending_writes:
        _maybe_flush_pending_writes(filename)
    slot_sel = _normalize_slot_selection(slot_selection)
    var_filter = set(var_selection) if var_selection is not None else None
    out: dict[str, Any] = {"attrs": {}, "const": {}, "preaccumulated": {}, "iter": {}, "systems": []}
    with h5py.File(filename, "r") as f:
        out["attrs"] = _collect_h5_attrs(f)
        # Multi-system metadata
        sys_count = int(out["attrs"].get("system_count", 0))
        sys_names_raw = out["attrs"].get("system_names")
        if sys_count > 0 and sys_names_raw:
            sys_names = [s.strip() for s in str(sys_names_raw).split(";")]
        else:
            sys_names = []
        for si in range(sys_count):
            sys_key = f"system_{si}"
            sys_info: dict[str, Any] = {
                "system_id": si,
                "system_name": sys_names[si] if si < len(sys_names) else f"System {si + 1}",
            }
            if sys_key in f:
                sys_info["attrs"] = _collect_h5_attrs(f[sys_key])
                sys_info["has_preaccumulated"] = "preaccumulated" in f[sys_key]
                sys_info["has_iter"] = "iter" in f[sys_key]
                sys_info["has_const"] = "const" in f[sys_key]
            out["systems"].append(sys_info)
        # Output groups metadata
        output_group_count = int(out["attrs"].get("output_group_count", 0))
        output_group_names_raw = out["attrs"].get("output_group_names", "")
        output_group_prefixes_raw = out["attrs"].get("output_group_prefixes", "")
        if output_group_count > 0 and output_group_names_raw:
            _og_names = [s.strip() for s in str(output_group_names_raw).split(";")]
            _og_prefixes = [s.strip() for s in str(output_group_prefixes_raw).split(";")]
            _output_groups_meta: list[dict[str, Any]] = []
            for _ogi in range(output_group_count):
                _og_info: dict[str, Any] = {
                    "name": _og_names[_ogi] if _ogi < len(_og_names) else f"Group {_ogi + 1}",
                    "prefix": _og_prefixes[_ogi] if _ogi < len(_og_prefixes) else "",
                }
                _og_prefix = _og_info["prefix"]
                if _og_prefix == "":
                    # Combined group — uses root preaccumulated
                    _og_info["has_preaccumulated"] = "preaccumulated" in f
                else:
                    _og_key = _og_prefix.rstrip("/")
                    _og_info["has_preaccumulated"] = (
                        _og_key in f and "preaccumulated" in f[_og_key]
                    )
                _output_groups_meta.append(_og_info)
            out["output_groups"] = _output_groups_meta
        else:
            out["output_groups"] = []
        if "const" in f:
            out["const"] = {name: _dataset_metadata(ds) for name, ds in f["const"].items()}
        if "preaccumulated" in f:
            out["preaccumulated"] = _collect_group_dataset_metadata(f["preaccumulated"])
        iter_ids = _iter_ids_from_file(f)
        if iter_selection is not None:
            want = sorted(set(int(v) for v in iter_selection))
            iter_ids = [ii for ii in iter_ids if ii in want]
        for ii in iter_ids:
            g = f["iter"][_iter_group_name(ii)]
            row_meta: dict[str, Any] = {}
            selected_slot_count = None
            for name, ds in g.items():
                if var_filter is not None and name not in var_filter:
                    continue
                meta = _dataset_metadata(ds)
                if ds.ndim >= 1 and selected_slot_count is None:
                    s0, s1 = _slice_bounds(slot_sel, int(ds.shape[0]))
                    selected_slot_count = s1 - s0
                row_meta[name] = meta
            out["iter"][ii] = {
                "datasets": row_meta,
                "selected_slot_count": selected_slot_count,
            }
    return out


def read_dataset_slice(
    filename: str,
    *,
    name: str,
    iteration: int | None = None,
    selection: Any = slice(None),
    times_as: str = "time",
    sync_pending_writes: bool = True,
) -> Any:
    """
    Read one arbitrary dataset slice without loading the whole file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file produced by :func:`write_data`.
    name : str
        Dataset name under ``/const`` or ``/iter/iter_xxxxx``.
    iteration : int, optional
        Iteration id when reading from ``/iter``. Omit to read from ``/const``.
    selection : Any, optional
        Slice/index object forwarded to the HDF5 dataset.
    times_as : {"time", "quantity"}, optional
        Reconstruction mode for time datasets.
    sync_pending_writes : bool, optional
        Flush any active local background writer for ``filename`` before
        opening the file.
    """
    filename = _normalize_storage_path(filename)
    if sync_pending_writes:
        _maybe_flush_pending_writes(filename)
    with h5py.File(filename, "r") as f:
        if iteration is None:
            _name_s = str(name)
            if (
                _name_s.startswith("preaccumulated/")
                or _name_s.startswith("iter/")
                or "/preaccumulated/" in _name_s
                or "/iter/" in _name_s
            ):
                ds = f[_name_s]
            else:
                ds = f["const"][name]
        else:
            ds = f["iter"][_iter_group_name(int(iteration))][name]
        return _read_dataset_selection(ds, selection, times_as=times_as)


def _stream_chunks_worker(
    filename: str,
    *,
    iter_ids: list[int],
    var_filter: set[str] | None,
    slot_sel: slice,
    slot_chunk_size: int,
    times_as: str,
    out_queue: "queue.Queue[Any]",
    group_prefix: str = "",
) -> None:
    """Background producer for streamed HDF5 slot chunks."""
    try:
        _iter_root = f"{group_prefix}iter"
        with h5py.File(filename, "r") as f:
            for ii in iter_ids:
                g = f[_iter_root][_iter_group_name(ii)]
                selected_names = [name for name in g.keys() if var_filter is None or name in var_filter]
                if not selected_names:
                    continue
                first_len = None
                for name in selected_names:
                    ds = g[name]
                    if ds.ndim < 1:
                        continue
                    first_len = int(ds.shape[0])
                    break
                if first_len is None:
                    continue
                base_start, base_stop = _slice_bounds(slot_sel, first_len)
                for s0 in range(base_start, base_stop, slot_chunk_size):
                    s1 = min(base_stop, s0 + slot_chunk_size)
                    chunk_data: dict[str, Any] = {}
                    for name in selected_names:
                        ds = g[name]
                        if ds.ndim < 1:
                            chunk_data[name] = _read_dataset_selection(ds, slice(None), times_as=times_as)
                        else:
                            chunk_data[name] = _read_dataset_selection(ds, slice(s0, s1), times_as=times_as)
                    out_queue.put(
                        {
                            "iteration": ii,
                            "iter_name": _iter_group_name(ii),
                            "slot_start": int(s0),
                            "slot_stop": int(s1),
                            "data": chunk_data,
                        }
                    )
        out_queue.put(_STREAM_SENTINEL)
    except BaseException as exc:  # pragma: no cover - exercised via consumer tests
        out_queue.put(exc)


def iter_data_chunks(
    filename: str,
    *,
    iter_selection: Iterable[int] | None = None,
    var_selection: Iterable[str] | None = None,
    slot_selection: Any = None,
    slot_chunk_size: int = 256,
    times_as: str = "time",
    prefetch_chunks: int = 1,
    sync_pending_writes: bool = True,
    group_prefix: str = "",
) -> Iterator[dict[str, Any]]:
    """
    Yield selected per-iteration timestep chunks from a results file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file produced by :func:`write_data`.
    iter_selection : iterable of int, optional
        Iteration ids to include.
    var_selection : iterable of str, optional
        Dataset names to include in every chunk.
    slot_selection : None, slice, or tuple, optional
        Per-iteration slot selection applied before chunking.
    slot_chunk_size : int, optional
        Maximum number of slots per yielded chunk.
    times_as : {"time", "quantity"}, optional
        Reconstruction mode for time datasets.
    prefetch_chunks : int, optional
        Number of prefetched chunks to keep buffered ahead of the consumer. Set
        to 0 to disable background prefetching.
    sync_pending_writes : bool, optional
        Flush any active local background writer for ``filename`` before
        opening the file.
    """
    filename = _normalize_storage_path(filename)
    if sync_pending_writes:
        _maybe_flush_pending_writes(filename)
    slot_sel = _normalize_slot_selection(slot_selection)
    var_filter = set(var_selection) if var_selection is not None else None
    slot_chunk_size = max(1, int(slot_chunk_size))
    prefetch_chunks = max(0, int(prefetch_chunks))

    with h5py.File(filename, "r") as f:
        iter_ids = _iter_ids_from_file(f)
    if iter_selection is not None:
        want = sorted(set(int(v) for v in iter_selection))
        iter_ids = [ii for ii in iter_ids if ii in want]

    _iter_root = f"{group_prefix}iter"
    if prefetch_chunks <= 0:
        with h5py.File(filename, "r") as f:
            for ii in iter_ids:
                g = f[_iter_root][_iter_group_name(ii)]
                selected_names = [name for name in g.keys() if var_filter is None or name in var_filter]
                if not selected_names:
                    continue
                first_len = None
                for name in selected_names:
                    ds = g[name]
                    if ds.ndim >= 1:
                        first_len = int(ds.shape[0])
                        break
                if first_len is None:
                    continue
                base_start, base_stop = _slice_bounds(slot_sel, first_len)
                for s0 in range(base_start, base_stop, slot_chunk_size):
                    s1 = min(base_stop, s0 + slot_chunk_size)
                    data = {}
                    for name in selected_names:
                        ds = g[name]
                        if ds.ndim < 1:
                            data[name] = _read_dataset_selection(ds, slice(None), times_as=times_as)
                        else:
                            data[name] = _read_dataset_selection(ds, slice(s0, s1), times_as=times_as)
                    yield {
                        "iteration": ii,
                        "iter_name": _iter_group_name(ii),
                        "slot_start": int(s0),
                        "slot_stop": int(s1),
                        "data": data,
                    }
        return

    out_queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, prefetch_chunks))
    reader = threading.Thread(
        target=_stream_chunks_worker,
        kwargs={
            "filename": filename,
            "iter_ids": iter_ids,
            "var_filter": var_filter,
            "slot_sel": slot_sel,
            "slot_chunk_size": slot_chunk_size,
            "times_as": times_as,
            "out_queue": out_queue,
            "group_prefix": group_prefix,
        },
        name=f"scepter-h5-reader-{os.path.basename(filename)}",
        daemon=True,
    )
    reader.start()
    while True:
        item = out_queue.get()
        if item is _STREAM_SENTINEL:
            break
        if isinstance(item, BaseException):
            raise RuntimeError(f"Streaming HDF5 reader for {filename!r} failed.") from item
        yield item


def _write_iteration_batch_owned(
    filename: str,
    *,
    iteration: int,
    batch_items: Iterable[tuple[str, Any]],
    compression_profile: str = "balanced",
    compression: Any = _AUTO_STORAGE_ARG,
    compression_opts: Any = _AUTO_STORAGE_ARG,
    chunk_target_bytes: int = _DEFAULT_SLOT_CHUNK_TARGET_BYTES,
    write_mode: str | None = None,
    writer_queue_max_items: int | None = None,
    writer_queue_max_bytes: int | None = None,
    allow_unit_auto_convert: bool = True,
    group_prefix: str = "",
) -> int | None:
    """Queue or apply one iteration batch while deferring normalization/scatter.

    When *group_prefix* is non-empty, writes to ``/{group_prefix}iter/``
    instead of ``/iter/``.
    """
    filename = _normalize_storage_path(filename)
    effective_mode = write_mode or _get_writer_defaults(filename)["write_mode"]
    if effective_mode not in {"async", "sync"}:
        raise ValueError("`write_mode` must be 'async' or 'sync'.")

    batch_items_tuple = tuple((str(name), value) for name, value in batch_items)
    if not batch_items_tuple:
        return None

    effective_compression, effective_compression_opts, effective_shuffle = _resolve_compression_settings(
        compression_profile=compression_profile,
        compression=compression,
        compression_opts=compression_opts,
    )
    op = _OwnedIterationWriteOperation(
        iteration=int(iteration),
        batch_items=batch_items_tuple,
        allow_unit_auto_convert=bool(allow_unit_auto_convert),
        compression=effective_compression,
        compression_opts=effective_compression_opts,
        shuffle=bool(effective_shuffle),
        chunk_target_bytes=max(1, int(chunk_target_bytes)),
        queue_bytes=int(sum(_estimate_owned_write_value_bytes(value) for _, value in batch_items_tuple)),
        group_prefix=str(group_prefix),
    )
    if effective_mode == "sync":
        _maybe_flush_pending_writes(filename)
        prepared = _prepare_owned_iteration_write_operation(op)
        with _interrupt_ctx(), h5py.File(filename, "a") as f:
            _apply_write_operation(f, prepared)
            f.flush()
        return None

    writer = _ensure_async_writer(
        filename,
        writer_queue_max_items=writer_queue_max_items,
        writer_queue_max_bytes=writer_queue_max_bytes,
    )
    return writer.submit(op)


def write_data(
    filename: str,
    *,
    # Store constants once (e.g., grid_info, config blobs, seeds)
    constants: Dict[str, Any] | None = None,
    # Stream a batch for a given iteration (append along axis 0)
    iteration: int | None = None,
    # Optional HDF5/file attrs to write/update at root
    attrs: Dict[str, Any] | None = None,
    # Compression/storage policy for NEW datasets only
    compression_profile: str = "balanced",
    compression: Any = _AUTO_STORAGE_ARG,
    compression_opts: Any = _AUTO_STORAGE_ARG,
    chunk_target_bytes: int = _DEFAULT_SLOT_CHUNK_TARGET_BYTES,
    # Write dispatch
    write_mode: str | None = None,
    writer_queue_max_items: int | None = None,
    writer_queue_max_bytes: int | None = None,
    # Behaviour flags
    overwrite_constants: bool = False,          # if const exists: False→verify equal, True→replace
    allow_unit_auto_convert: bool = True,       # when appending, auto-convert if units are compatible
    # Streaming payload when iteration is provided
    **batch: Any,
):
    """
    Universal writer.

    Use cases:
      1) Store constants once:
            write_data(fn, constants={'grid_info': grid, 'cfg': cfg_dict})
         Constants are stored under /const/<name> and never extended.

      2) Stream a batch for an iteration:
            write_data(fn, iteration=ii, times=..., epfd_0=..., power=...)
         Data live under /iter/iter_00000/<var> and are extendable on axis 0.

      3) You can do both in one call: pass constants=... and iteration=... together.

    Supported value types:
      - astropy.Time          → stored as float MJD with attrs: kind="time/mjd", unit="d"
      - astropy.Quantity      → raw array stored, attrs: unit="..."; unit is preserved
      - plain array-likes     → raw numpy arrays, unitless

    Compression/storage policy:
      - ``compression_profile="balanced"`` is the default for new datasets
      - explicit ``compression`` / ``compression_opts`` override the profile
      - iteration datasets use explicit slot-first chunk shapes instead of
        generic ``chunks=True``

    Write dispatch:
      - async mode is the default and enqueues work into the per-file
        background writer
      - sync mode is opt-in and performs an immediate durable write
      - call `flush_writes(...)` or `close_writer(...)` before downstream reads
        when you need durability guarantees at a known synchronization point

    Unit safety on appends:
      - existing has unit, new is unitless      → error
      - existing unitless, new has unit         → error
      - both have units but different:
            * if compatible and allow_unit_auto_convert=True → new data are converted to existing unit
            * otherwise → error
    """
    filename = _normalize_storage_path(filename)
    effective_mode = write_mode or _get_writer_defaults(filename)["write_mode"]
    if effective_mode not in {"async", "sync"}:
        raise ValueError("`write_mode` must be 'async' or 'sync'.")

    op = _prepare_write_operation(
        attrs=attrs,
        constants=constants,
        iteration=iteration,
        batch=batch,
        overwrite_constants=overwrite_constants,
        allow_unit_auto_convert=allow_unit_auto_convert,
        compression_profile=compression_profile,
        compression=compression,
        compression_opts=compression_opts,
        chunk_target_bytes=chunk_target_bytes,
    )
    if not op.attrs and not op.constants and op.iteration is None:
        return

    if effective_mode == "sync":
        _maybe_flush_pending_writes(filename)
        with _interrupt_ctx(), h5py.File(filename, "a") as f:
            _apply_write_operation(f, op)
            f.flush()
        return

    writer = _ensure_async_writer(
        filename,
        writer_queue_max_items=writer_queue_max_items,
        writer_queue_max_bytes=writer_queue_max_bytes,
    )
    writer.submit(op)


def read_data(
    filename: str,
    *,
    mode: str = "eager",
    # selections
    iter_selection: Iterable[int] | None = None,
    var_selection: Iterable[str] | None = None,
    slot_selection: Any = None,
    slot_chunk_size: int = 256,
    prefetch_chunks: int = 1,
    # how to return time datasets
    times_as: str = "time",        # "time" (default) or "quantity"
    # stacking options
    stack: bool = True,            # <— now defaults to True
    pad_value: float = np.nan,
    return_masks: bool = False,
    # also return raw per-iteration values alongside stacked
    include_by_iter: bool = False,
    sync_pending_writes: bool = True,
    # multi-system: read from /system_N/iter/ instead of /iter/
    group_prefix: str = "",
) -> Dict[str, Any]:
    """
    Read simulation results from an HDF5 file produced by :func:`write_data`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file produced by :func:`write_data`.
    mode : {"eager", "slice", "stream"}, optional
        Read mode. ``"eager"`` loads the selected iteration datasets fully.
        ``"slice"`` applies ``slot_selection`` before materializing datasets.
        ``"stream"`` returns metadata plus a chunk iterator instead of loading
        the selected iteration datasets into memory.
    iter_selection : iterable of int, optional
        Iteration ids to include. When omitted, all stored ``/iter/iter_*``
        groups are considered.
    var_selection : iterable of str, optional
        Dataset names to include under each selected iteration.
    slot_selection : None, slice, or tuple, optional
        Per-iteration slot selection applied on the leading stored slot axis.
        ``None`` means the full stored extent. ``(start, stop)`` and
        ``slice(start, stop)`` are both supported.
    slot_chunk_size : int, optional
        Maximum number of slots yielded per streamed chunk when
        ``mode="stream"``.
    prefetch_chunks : int, optional
        Number of streamed chunks to prefetch ahead of the consumer. Set to
        ``0`` to disable background prefetching.
    times_as : {"time", "quantity"}, optional
        Reconstruction mode for stored time datasets.
    stack : bool, optional
        When ``True``, eager and sliced reads return padded stacked arrays
        under ``out["iter"]``. When ``False``, the raw per-iteration mapping is
        returned instead. Streaming mode requires ``stack=False``.
    pad_value : float, optional
        Fill value used when stacking variables whose selected slot lengths
        differ across iterations.
    return_masks : bool, optional
        When stacking, also return boolean padding masks under
        ``out["masks"]``.
    include_by_iter : bool, optional
        Also include the raw per-iteration mapping under ``out["by_iter"]`` for
        eager and sliced reads.
    sync_pending_writes : bool, optional
        Flush any active local background writer for ``filename`` before
        opening the file.

    Returns
    -------
    out : dict
        In ``mode="eager"`` and ``mode="slice"``, the return structure matches
        the historical eager loader.

        In ``mode="stream"``, the dictionary contains eagerly loaded
        ``"const"`` and ``"attrs"`` metadata plus:

        ``"iter_meta"``
            Per-iteration dataset metadata and selected slot counts.
        ``"stream"``
            Iterator yielding ``{iteration, slot_start, slot_stop, data}``.

    Raises
    ------
    ValueError
        Raised when ``mode`` is invalid, when ``stack=True`` is requested for
        streaming reads, or when slot selections are malformed.

    Notes
    -----
    Root attrs and constants are still read eagerly in all modes because they
    are expected to be small. Streaming mode is intended for large timestep
    arrays under ``/iter``.
    """
    filename = _normalize_storage_path(filename)
    if sync_pending_writes:
        _maybe_flush_pending_writes(filename)
    mode = str(mode).lower().strip()
    if mode not in {"eager", "slice", "stream"}:
        raise ValueError("`mode` must be 'eager', 'slice', or 'stream'.")
    if mode == "stream" and stack:
        raise ValueError("`stack=True` is not supported when `mode='stream'`.")
    slot_sel = _normalize_slot_selection(slot_selection)
    out: Dict[str, Any] = {"const": {}, "preaccumulated": {}, "iter": {}}

    # ----- load constants and per-iteration raw data -----
    by_iter: Dict[int, Dict[str, Any]] = {}

    with h5py.File(filename, "r") as f:
        out["attrs"] = _collect_h5_attrs(f)
        # constants
        if "const" in f:
            for name, ds in f["const"].items():
                out["const"][name] = _read_dataset(ds, times_as=times_as)
        if "preaccumulated" in f:
            for path, _meta in _collect_group_dataset_metadata(f["preaccumulated"]).items():
                ds = f["preaccumulated"][path]
                out["preaccumulated"][path] = _read_dataset(ds, times_as=times_as)

        # iterations
        _iter_root_key = f"{group_prefix}iter"
        if _iter_root_key in f:
            all_iters = _iter_ids_from_file(f, iter_root_key=_iter_root_key)

            if iter_selection is not None:
                want = sorted(set(int(i) for i in iter_selection))
                iters = [i for i in all_iters if i in want]
            else:
                iters = all_iters

            var_filter = set(var_selection) if var_selection is not None else None

            if mode == "stream":
                iter_meta = {}
                for ii in iters:
                    g = f[_iter_root_key][_iter_group_name(ii)]
                    row_meta: dict[str, Any] = {}
                    selected_slot_count = None
                    for dname, ds in g.items():
                        if var_filter is not None and dname not in var_filter:
                            continue
                        row_meta[dname] = _dataset_metadata(ds)
                        if ds.ndim >= 1 and selected_slot_count is None:
                            s0, s1 = _slice_bounds(slot_sel, int(ds.shape[0]))
                            selected_slot_count = s1 - s0
                    iter_meta[ii] = {
                        "datasets": row_meta,
                        "selected_slot_count": selected_slot_count,
                    }
                out["iter_meta"] = iter_meta
                out["stream"] = iter_data_chunks(
                    filename,
                    iter_selection=iters,
                    var_selection=var_selection,
                    slot_selection=slot_sel,
                    slot_chunk_size=slot_chunk_size,
                    times_as=times_as,
                    prefetch_chunks=prefetch_chunks,
                    sync_pending_writes=False,
                    group_prefix=group_prefix,
                )
                return out

            for ii in iters:
                g = f[_iter_root_key][f"iter_{ii:05d}"]
                row: Dict[str, Any] = {}
                for dname, ds in g.items():
                    if var_filter is not None and dname not in var_filter:
                        continue
                    if ds.ndim >= 1 and mode == "slice":
                        s0, s1 = _slice_bounds(slot_sel, int(ds.shape[0]))
                        row[dname] = _read_dataset_selection(ds, slice(s0, s1), times_as=times_as)
                    else:
                        row[dname] = _read_dataset(ds, times_as=times_as)
                by_iter[ii] = row

    # No iteration data found
    if not by_iter:
        if include_by_iter:
            out["by_iter"] = {}
        return out

    # Short-circuit if caller asked explicitly not to stack
    if not stack:
        if include_by_iter:
            out["by_iter"] = by_iter
        else:
            # mirror your previous shape: results["iter"][iter_id] -> {var: ...}
            out["iter"] = by_iter
        return out

    # ----- build stacked tensors per variable -----

    # union of variable names across all loaded iterations
    iter_ids = sorted(by_iter)
    var_names = set()
    for ii in iter_ids:
        var_names.update(by_iter[ii].keys())
    var_names = sorted(var_names)

    stacked: Dict[str, Any] = {}
    masks: Dict[str, np.ndarray] = {}

    for var in var_names:
        # Collect rows for all selected iterations; allow missing (treated as length 0)
        rows_numeric: List[np.ndarray] = []
        kinds: List[str | None] = []
        units_seen: List[Any | None] = []
        trailing_shape = None
        stackable = True

        for ii in iter_ids:
            val = by_iter[ii].get(var, None)

            # Missing in this iteration → 0-length row
            if val is None:
                arr = np.empty((0,), dtype=float)
                row_kind = None
                row_unit = None
            else:
                # Normalise to numeric ndarray for stacking and capture kind/unit
                if _is_time_obj(val):                      # astropy.Time
                    arr = np.asarray(val.mjd)
                    row_kind = "time/mjd"
                    row_unit = u.day if u is not None else None
                elif hasattr(val, "unit") and u is not None:  # Quantity
                    arr = np.asarray(val.value)
                    row_kind = "quantity"
                    row_unit = val.unit
                else:
                    arr = np.asarray(val)
                    row_kind = None
                    row_unit = None

                if arr.ndim == 0:
                    arr = arr.reshape(1)

            # Track trailing shape (beyond time axis)
            ts = arr.shape[1:]
            if trailing_shape is None:
                trailing_shape = ts
            elif trailing_shape != ts:
                # Incompatible shapes across iterations → cannot stack this variable
                stackable = False
                break

            rows_numeric.append(arr)
            kinds.append(row_kind)
            units_seen.append(row_unit)

        if not stackable:
            # Skip stacking; leave variable accessible via raw layout if requested
            continue

        # Determine max time length across iterations
        Tmax = max(r.shape[0] for r in rows_numeric)
        n_iter = len(rows_numeric)
        pad_shape = (n_iter, Tmax) + (trailing_shape or ())
        pad = np.full(pad_shape, pad_value, dtype=(rows_numeric[0].dtype if rows_numeric else float))
        mask = np.ones((n_iter, Tmax), dtype=bool)  # True where padded

        # Choose a “reference” unit/kind from the first non-empty row
        ref_unit = None
        ref_kind = None
        for k, (rk, ru) in enumerate(zip(kinds, units_seen)):
            if rows_numeric[k].size > 0:
                ref_kind = rk
                ref_unit = ru
                break

        # Fill rows
        for k, arr in enumerate(rows_numeric):
            t = arr.shape[0]
            if t > 0:
                pad[k, :t, ...] = arr
                mask[k, :t] = False

        # Attach unit or convert to Time as needed
        if ref_kind == "time/mjd" and times_as == "time" and Time is not None:
            # Convert numeric MJD to astropy.Time; NaNs remain as NaN times
            stacked[var] = Time(pad, format="mjd")
        elif ref_kind == "quantity" and (u is not None) and (ref_unit is not None):
            stacked[var] = pad * ref_unit
        else:
            stacked[var] = pad

        if return_masks:
            masks[var] = mask

    # Put results in the requested top-level shape
    out["iter"] = stacked
    if return_masks:
        out["masks"] = masks
    if include_by_iter:
        out["by_iter"] = by_iter

    return out


def _summarize_direct_epfd_progress_events(
    progress_events: Iterable[Mapping[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Summarize direct-EPFD progress events into compact performance metrics."""
    plan_shapes: set[tuple[int, int, int, int]] = set()
    active_shapes: set[tuple[int, int, int, int]] = set()
    limiting_resources: set[str] = set()
    limiting_dimensions: set[str] = set()
    planner_sources: set[str] = set()
    underfill_reasons: set[str] = set()
    warning_count = 0
    retry_count = 0
    spectral_backoff_active = False
    attempted_work_units = 0.0
    planned_total_seconds: float | None = None
    planned_remaining_seconds: float | None = None
    predicted_gpu_peak_bytes_max = 0
    predicted_gpu_activity_peak_bytes_max = 0
    predicted_gpu_spectrum_context_bytes_max = 0
    compute_budget_utilization_fraction_max = 0.0
    export_budget_utilization_fraction_max = 0.0

    for event in progress_events:
        payload = dict(event)
        kind = str(payload.get("kind", "")).strip().lower()
        if kind == "warning":
            warning_count += 1
        retry_count = max(retry_count, int(payload.get("scheduler_retry_count", 0) or 0))
        spectral_backoff_active = spectral_backoff_active or bool(
            payload.get("spectral_backoff_active")
        )
        predicted_gpu_peak_bytes_max = max(
            predicted_gpu_peak_bytes_max,
            int(payload.get("predicted_gpu_peak_bytes", 0) or 0),
        )
        predicted_gpu_activity_peak_bytes_max = max(
            predicted_gpu_activity_peak_bytes_max,
            int(payload.get("predicted_gpu_activity_peak_bytes", 0) or 0),
        )
        predicted_gpu_spectrum_context_bytes_max = max(
            predicted_gpu_spectrum_context_bytes_max,
            int(payload.get("predicted_gpu_spectrum_context_bytes", 0) or 0),
        )
        if payload.get("planned_total_seconds") is not None:
            planned_total_seconds = float(payload["planned_total_seconds"])
        if payload.get("planned_remaining_seconds") is not None:
            planned_remaining_seconds = float(payload["planned_remaining_seconds"])
        if payload.get("limiting_resource"):
            limiting_resources.add(str(payload["limiting_resource"]))
        if payload.get("limiting_dimension"):
            limiting_dimensions.add(str(payload["limiting_dimension"]))
        if payload.get("planner_source"):
            planner_sources.add(str(payload["planner_source"]))
        if payload.get("underfill_reason"):
            underfill_reasons.add(str(payload["underfill_reason"]))
        if payload.get("compute_budget_utilization_fraction") is not None:
            compute_budget_utilization_fraction_max = max(
                compute_budget_utilization_fraction_max,
                float(payload["compute_budget_utilization_fraction"]),
            )
        if payload.get("export_budget_utilization_fraction") is not None:
            export_budget_utilization_fraction_max = max(
                export_budget_utilization_fraction_max,
                float(payload["export_budget_utilization_fraction"]),
            )

        shape = (
            int(payload.get("bulk_timesteps", 0) or 0),
            int(payload.get("cell_chunk", 0) or 0),
            int(payload.get("sky_slab", 1) or 1),
            int(payload.get("spectral_slab", 1) or 1),
        )
        if kind == "iteration_plan":
            plan_shapes.add(shape)
        elif kind == "batch_start":
            active_shapes.add(shape)
            attempted_work_units += (
                float(max(0, shape[0]))
                * float(max(0, shape[1]))
                * float(max(1, shape[2]))
            )

    elapsed_s = max(float(elapsed_seconds), 1e-9)
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "attempted_work_units": float(attempted_work_units),
        "attempted_work_units_per_second": float(attempted_work_units / elapsed_s),
        "plan_shapes": tuple(sorted(plan_shapes)),
        "active_shapes": tuple(sorted(active_shapes)),
        "planner_sources": tuple(sorted(planner_sources)),
        "limiting_resources": tuple(sorted(limiting_resources)),
        "limiting_dimensions": tuple(sorted(limiting_dimensions)),
        "warning_count": int(warning_count),
        "retry_count": int(retry_count),
        "spectral_backoff_active": bool(spectral_backoff_active),
        "planned_total_seconds": planned_total_seconds,
        "planned_remaining_seconds": planned_remaining_seconds,
        "predicted_gpu_peak_bytes_max": int(predicted_gpu_peak_bytes_max),
        "predicted_gpu_activity_peak_bytes_max": int(predicted_gpu_activity_peak_bytes_max),
        "predicted_gpu_spectrum_context_bytes_max": int(
            predicted_gpu_spectrum_context_bytes_max
        ),
        "compute_budget_utilization_fraction_max": float(
            compute_budget_utilization_fraction_max
        ),
        "export_budget_utilization_fraction_max": float(
            export_budget_utilization_fraction_max
        ),
        "underfill_reasons": tuple(sorted(underfill_reasons)),
    }


def _direct_epfd_stage_timings_time_total(stage_timings_summary: Mapping[str, Any]) -> float:
    """Sum time-like stage entries while excluding synthetic counter fields."""
    total = 0.0
    for name, value in dict(stage_timings_summary).items():
        name_str = str(name)
        if name_str.endswith("_count"):
            continue
        total += float(value)
    return float(total)


_DIRECT_EPFD_WRITER_TIMING_NAMES = frozenset(
    {
        "export_copy",
        "write_enqueue",
        "writer_checkpoint_wait",
        "export_scatter",
        "writer_apply",
        "writer_flush",
    }
)


def _summarize_direct_epfd_stage_timings(
    stage_timings_summary: Mapping[str, Any] | None,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Normalize run-level stage timings into benchmark-friendly scalar metrics."""
    timings = (
        {}
        if stage_timings_summary is None
        else {str(key): float(value) for key, value in dict(stage_timings_summary).items()}
    )
    writer_overhead_seconds = float(
        sum(float(timings.get(name, 0.0)) for name in _DIRECT_EPFD_WRITER_TIMING_NAMES)
    )
    host_sync_seconds = float(timings.get("host_sync_telemetry", 0.0))
    one_time_setup_seconds = float(timings.get("spectrum_context_setup", 0.0))
    known_time_total = _direct_epfd_stage_timings_time_total(timings)
    compute_stage_seconds = max(
        0.0,
        float(known_time_total) - float(writer_overhead_seconds) - float(host_sync_seconds),
    )
    return {
        "stage_timings_summary": timings,
        "one_time_spectrum_setup_seconds": float(one_time_setup_seconds),
        "orbit_propagation_seconds": float(timings.get("orbit_propagation", 0.0)),
        "ras_geometry_seconds": float(timings.get("ras_geometry", 0.0)),
        "cell_link_library_seconds": float(timings.get("cell_link_library", 0.0)),
        "cell_activity_setup_seconds": float(timings.get("cell_activity_setup", 0.0)),
        "spectrum_activity_weighting_seconds": float(
            timings.get("spectrum_activity_weighting", 0.0)
        ),
        "beam_finalize_seconds": float(timings.get("beam_finalize", 0.0)),
        "boresight_screening_seconds": float(timings.get("boresight_screening", 0.0)),
        "pointings_seconds": float(timings.get("pointings", 0.0)),
        "power_accumulation_seconds": float(timings.get("power_accumulation", 0.0)),
        "export_copy_seconds": float(timings.get("export_copy", 0.0)),
        "write_enqueue_seconds": float(timings.get("write_enqueue", 0.0)),
        "compute_stage_seconds": float(compute_stage_seconds),
        "writer_overhead_seconds": float(writer_overhead_seconds),
        "export_writer_overhead_seconds": float(writer_overhead_seconds),
        "host_sync_telemetry_overhead_seconds": float(host_sync_seconds),
        "unattributed_elapsed_seconds": float(
            max(0.0, float(elapsed_seconds) - float(known_time_total))
        ),
    }


def _benchmark_gpu_metric_snapshot(*, device_index: int = 0) -> dict[str, Any] | None:
    """Sample adapter-wide GPU utilization and VRAM use for benchmark summaries."""
    if not _NVIDIA_SMI_PATH:
        return None
    try:
        result = subprocess.run(
            [
                _NVIDIA_SMI_PATH,
                f"--id={int(device_index)}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    lines = str(result.stdout or "").splitlines()
    if not lines:
        return None
    try:
        util_text, used_text, total_text = [part.strip() for part in lines[0].split(",", 2)]
        util_percent = max(0.0, float(util_text))
        used_mib = max(0.0, float(used_text))
        total_mib = max(0.0, float(total_text))
    except Exception:
        return None
    return {
        "timestamp_s": float(perf_counter()),
        "gpu_utilization_percent": float(util_percent),
        "gpu_memory_used_bytes": int(round(used_mib * (1024.0 ** 2))),
        "gpu_memory_total_bytes": int(round(total_mib * (1024.0 ** 2))),
    }


class _BenchmarkGpuMetricSampler:
    """Best-effort benchmark-only GPU sampler backed by ``nvidia-smi``."""

    def __init__(
        self,
        *,
        enabled: bool,
        sample_interval_s: float,
        device_index: int = 0,
        snapshot_func: Callable[..., Mapping[str, Any] | None] = _benchmark_gpu_metric_snapshot,
    ) -> None:
        self.enabled = bool(enabled)
        self.sample_interval_s = float(max(0.05, sample_interval_s))
        self.device_index = int(device_index)
        self.snapshot_func = snapshot_func
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []

    def _capture_once(self) -> None:
        if not self.enabled:
            return
        try:
            payload = self.snapshot_func(device_index=int(self.device_index))
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            self.samples.append({str(key): value for key, value in dict(payload).items()})

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self._capture_once()

    def start(self) -> None:
        if not self.enabled:
            return
        self._capture_once()
        self._thread = threading.Thread(
            target=self._run,
            name="direct-epfd-benchmark-gpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> tuple[dict[str, Any], ...]:
        if not self.enabled:
            return tuple()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_s * 4.0))
        self._capture_once()
        return tuple(self.samples)


def _make_benchmark_gpu_metric_sampler(
    *,
    enabled: bool,
    sample_interval_s: float = 0.25,
    device_index: int = 0,
) -> _BenchmarkGpuMetricSampler:
    """Create the benchmark GPU metric sampler used by ``benchmark_direct_epfd_runs``."""
    return _BenchmarkGpuMetricSampler(
        enabled=bool(enabled),
        sample_interval_s=float(sample_interval_s),
        device_index=int(device_index),
    )


def _summarize_benchmark_gpu_metric_samples(
    samples: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Reduce benchmark GPU metric samples to stable utilization and VRAM stats."""
    rows = [dict(sample) for sample in (samples or []) if isinstance(sample, Mapping)]
    if not rows:
        return {}
    util_values = np.asarray(
        [float(row["gpu_utilization_percent"]) for row in rows if row.get("gpu_utilization_percent") is not None],
        dtype=np.float64,
    )
    mem_values = np.asarray(
        [int(row["gpu_memory_used_bytes"]) for row in rows if row.get("gpu_memory_used_bytes") is not None],
        dtype=np.int64,
    )
    total_values = np.asarray(
        [int(row["gpu_memory_total_bytes"]) for row in rows if row.get("gpu_memory_total_bytes") is not None],
        dtype=np.int64,
    )
    summary: dict[str, Any] = {
        "gpu_metric_sample_count": int(len(rows)),
    }
    if util_values.size > 0:
        summary.update(
            gpu_utilization_percent_min=float(np.min(util_values)),
            gpu_utilization_percent_median=float(np.median(util_values)),
            gpu_utilization_percent_max=float(np.max(util_values)),
        )
    if mem_values.size > 0:
        summary.update(
            gpu_memory_used_bytes_min=int(np.min(mem_values)),
            gpu_memory_used_bytes_max=int(np.max(mem_values)),
            gpu_memory_used_bytes_span=int(np.max(mem_values) - np.min(mem_values)),
        )
    if total_values.size > 0:
        summary["gpu_memory_total_bytes_max"] = int(np.max(total_values))
    return summary


def _build_direct_epfd_run_request_from_gui_state(
    *,
    window: Any,
    sgui_module: Any,
    state: Any,
) -> dict[str, Any]:
    """Build a benchmark run request from a GUI state using the real GUI path."""
    window._load_state_into_widgets(state)
    current_state = window.current_state()
    constellation = sgui_module.build_constellation_from_state(current_state)
    antenna_func, wavelength, pattern_kwargs = sgui_module._satellite_antenna_pattern_spec(
        current_state.active_system().satellite_antennas
    )
    contour_summary = window._build_run_contour_summary(
        current_state,
        constellation,
        antenna_func,
        wavelength,
        pattern_kwargs,
    )
    effective_cell_km = window._resolve_effective_run_cell_size_km(
        current_state,
        contour_summary,
    )
    window._last_analyser_selected_cell_km = float(effective_cell_km)
    window._last_analyser_signature = window._analyser_signature(current_state)
    preview_signature = window._hexgrid_preview_signature(current_state)
    commit_signature = window._hexgrid_commit_signature(current_state)
    if preview_signature is None or commit_signature is None:
        raise ValueError(
            "Benchmark GUI config does not produce current coverage/boresight settings."
        )
    window._hexgrid_completed_signature = preview_signature
    window._hexgrid_completed_commit_signature = commit_signature
    window._hexgrid_completed_overlay_signature = window._hexgrid_overlay_signature(
        current_state
    )
    window._hexgrid_outdated = False
    # For multi-system configs, set up derived state for each system
    n_systems = len(current_state.systems) if current_state.systems else 1
    if n_systems > 1:
        window._save_per_system_derived_state(0)
        for sys_idx in range(1, n_systems):
            window._active_system_index = sys_idx
            window._restore_per_system_derived_state(sys_idx)
            window._load_system_into_widgets(current_state.systems[sys_idx])
            sys_state = window.current_state()
            sys_constellation = sgui_module.build_constellation_from_state(sys_state)
            sys_ant = sys_state.active_system().satellite_antennas
            _, sys_wl, sys_pk = sgui_module._satellite_antenna_pattern_spec(sys_ant)
            sys_contour = window._build_run_contour_summary(
                sys_state, sys_constellation, _, sys_wl, sys_pk,
            )
            sys_cell_km = window._resolve_effective_run_cell_size_km(sys_state, sys_contour)
            window._last_analyser_selected_cell_km = float(sys_cell_km)
            window._last_analyser_signature = window._analyser_signature(sys_state)
            sys_preview_sig = window._hexgrid_preview_signature(sys_state)
            sys_commit_sig = window._hexgrid_commit_signature(sys_state)
            window._hexgrid_completed_signature = sys_preview_sig
            window._hexgrid_completed_commit_signature = sys_commit_sig
            window._hexgrid_completed_overlay_signature = window._hexgrid_overlay_signature(sys_state)
            window._save_per_system_derived_state(sys_idx)
        # Restore system 0
        window._active_system_index = 0
        window._restore_per_system_derived_state(0)
        window._load_system_into_widgets(current_state.systems[0])
    window._update_simulation_page_indicators(current_state)
    window._update_guidance_panel(current_state)
    window._update_run_controls(current_state)
    if n_systems > 1:
        return window._build_multi_system_run_request(window.current_state())
    return window._build_run_request(window.current_state())


def build_direct_epfd_run_request_from_gui_config(
    config_path: str | Path,
    *,
    timestep_s: float | None = None,
    gpu_memory_budget_gb: float | None = None,
    host_memory_budget_gb: float | None = None,
    profile_stages: bool | None = None,
) -> dict[str, Any]:
    """
    Build a direct-EPFD runner payload from a saved GUI project config.

    Parameters
    ----------
    config_path : str or Path
        Path to a GUI JSON project file such as ``benchmark_config.json``.
    timestep_s : float or None, optional
        Optional timestep override in seconds. ``None`` preserves the saved GUI
        timestep.
    gpu_memory_budget_gb, host_memory_budget_gb : float or None, optional
        Optional VRAM and host-RAM budget overrides in gibibytes.
    profile_stages : bool or None, optional
        Optional profiling flag override for benchmark-oriented runs.

    Returns
    -------
    dict[str, Any]
        Exact keyword payload suitable for :func:`run_gpu_direct_epfd`.

    Raises
    ------
    ImportError
        If the GUI stack required to normalize the config is unavailable.
    ValueError
        If the loaded GUI state is not run-ready after applying the overrides.

    Notes
    -----
    This helper intentionally routes through the real GUI request builder so
    benchmark cases stay aligned with the production GUI semantics for
    spectrum, power input, coverage, and runtime defaults.
    """
    try:
        from PySide6 import QtWidgets
        from scepter import scepter_GUI as sgui
    except Exception as exc:  # pragma: no cover - depends on optional GUI stack
        raise ImportError(
            "GUI benchmark request building requires PySide6 and scepter.scepter_GUI."
        ) from exc

    config_path_use = Path(config_path)
    state = sgui.load_project_state(config_path_use)
    runtime_updates: dict[str, Any] = {}
    if timestep_s is not None:
        runtime_updates["timestep_s"] = float(timestep_s)
    if gpu_memory_budget_gb is not None:
        runtime_updates["gpu_memory_budget_gb"] = float(gpu_memory_budget_gb)
    if host_memory_budget_gb is not None:
        runtime_updates["host_memory_budget_gb"] = float(host_memory_budget_gb)
    if profile_stages is not None:
        runtime_updates["profile_stages"] = bool(profile_stages)
    if runtime_updates:
        state = replace(state, runtime=replace(state.runtime, **runtime_updates))

    created_app = False
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
        created_app = True
    window = sgui.ScepterMainWindow()
    try:
        return _build_direct_epfd_run_request_from_gui_state(
            window=window,
            sgui_module=sgui,
            state=state,
        )
    finally:
        try:
            window._dirty = False
            window.close()
        except Exception:
            pass
        if created_app and app is not None:
            try:
                app.quit()
            except Exception:
                pass


def build_direct_epfd_benchmark_cases_from_gui_config(
    config_path: str | Path,
    *,
    timestep_values_s: Iterable[float] | None = None,
    memory_budget_pairs_gb: Iterable[tuple[float, float]] | None = None,
    profile_stages: bool = True,
    graceful_stop_after_first_batch: bool = True,
    graceful_stop_after_batch_count: int | None = None,
    sample_live_gpu_metrics: bool = True,
    gpu_metric_sample_interval_s: float = 0.25,
    name_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build benchmark case dictionaries from a saved GUI benchmark config.

    Parameters
    ----------
    config_path : str or Path
        Path to the canonical GUI JSON project configuration.
    timestep_values_s : iterable of float or None, optional
        Timestep ladder in seconds. ``None`` uses the timestep stored in the
        config.
    memory_budget_pairs_gb : iterable of ``(host_gb, gpu_gb)`` tuples or None, optional
        Host/GPU memory-budget pairs in gibibytes. ``None`` uses the config's
        saved budgets.
    profile_stages : bool, optional
        Benchmark-oriented override for ``profile_stages``.
    graceful_stop_after_first_batch : bool, optional
        When `True`, benchmark runs request a graceful stop after the first
        batch starts so short representative measurements can be gathered
        quickly.
    graceful_stop_after_batch_count : int or None, optional
        Optional graceful-stop batch count for short benchmark probes.
        ``None`` falls back to ``1`` when
        ``graceful_stop_after_first_batch=True`` and disables the automatic
        graceful stop otherwise.
    sample_live_gpu_metrics : bool, optional
        When `True`, benchmark cases request live GPU-utilization / VRAM
        sampling during the run.
    gpu_metric_sample_interval_s : float, optional
        Sampling cadence in seconds for the benchmark-only GPU metric sampler.
    name_prefix : str or None, optional
        Optional prefix for generated case names.

    Returns
    -------
    list of dict
        Case dictionaries ready for :func:`benchmark_direct_epfd_runs`.
    """
    base_request = build_direct_epfd_run_request_from_gui_config(
        config_path,
        profile_stages=bool(profile_stages),
    )
    timestep_values = (
        [float(base_request["timestep"])]
        if timestep_values_s is None
        else [float(value) for value in timestep_values_s]
    )
    budget_pairs = (
        [
            (
                float(base_request["host_memory_budget_gb"]),
                float(base_request["gpu_memory_budget_gb"]),
            )
        ]
        if memory_budget_pairs_gb is None
        else [(float(host_gb), float(gpu_gb)) for host_gb, gpu_gb in memory_budget_pairs_gb]
    )
    prefix = (
        str(name_prefix)
        if name_prefix is not None
        else Path(config_path).stem
    )
    graceful_batch_count = (
        1
        if graceful_stop_after_batch_count is None and graceful_stop_after_first_batch
        else (
            None
            if graceful_stop_after_batch_count is None
            else max(1, int(graceful_stop_after_batch_count))
        )
    )

    cases: list[dict[str, Any]] = []
    for timestep_value in timestep_values:
        for host_gb, gpu_gb in budget_pairs:
            run_kwargs = build_direct_epfd_run_request_from_gui_config(
                config_path,
                timestep_s=float(timestep_value),
                gpu_memory_budget_gb=float(gpu_gb),
                host_memory_budget_gb=float(host_gb),
                profile_stages=bool(profile_stages),
            )
            cases.append(
                {
                    "name": (
                        f"{prefix}_dt{float(timestep_value):g}s_"
                        f"host{float(host_gb):g}g_gpu{float(gpu_gb):g}g"
                    ),
                    "kwargs": run_kwargs,
                    "graceful_stop_after_first_batch": bool(
                        graceful_stop_after_first_batch
                    ),
                    "graceful_stop_after_batch_count": graceful_batch_count,
                    "sample_live_gpu_metrics": bool(sample_live_gpu_metrics),
                    "gpu_metric_sample_interval_s": float(gpu_metric_sample_interval_s),
                }
            )
    return cases


def benchmark_direct_epfd_runs_from_gui_config(
    config_path: str | Path,
    *,
    timestep_values_s: Iterable[float] | None = None,
    memory_budget_pairs_gb: Iterable[tuple[float, float]] | None = None,
    profile_stages: bool = True,
    graceful_stop_after_first_batch: bool = True,
    graceful_stop_after_batch_count: int | None = None,
    sample_live_gpu_metrics: bool = True,
    gpu_metric_sample_interval_s: float = 0.25,
    runner: Callable[..., Any] | None = None,
    time_func: Callable[[], float] = perf_counter,
) -> list[dict[str, Any]]:
    """
    Benchmark direct-EPFD runs derived from a canonical GUI JSON config.

    Parameters
    ----------
    config_path : str or Path
        Canonical GUI project config such as ``benchmark_config.json``.
    timestep_values_s, memory_budget_pairs_gb, profile_stages, graceful_stop_after_first_batch, graceful_stop_after_batch_count, sample_live_gpu_metrics, gpu_metric_sample_interval_s
        Forwarded to :func:`build_direct_epfd_benchmark_cases_from_gui_config`.
    runner, time_func : callable, optional
        Forwarded to :func:`benchmark_direct_epfd_runs`.

    Returns
    -------
    list of dict
        Benchmark summaries returned by :func:`benchmark_direct_epfd_runs`.
    """
    cases = build_direct_epfd_benchmark_cases_from_gui_config(
        config_path,
        timestep_values_s=timestep_values_s,
        memory_budget_pairs_gb=memory_budget_pairs_gb,
        profile_stages=bool(profile_stages),
        graceful_stop_after_first_batch=bool(graceful_stop_after_first_batch),
        graceful_stop_after_batch_count=graceful_stop_after_batch_count,
        sample_live_gpu_metrics=bool(sample_live_gpu_metrics),
        gpu_metric_sample_interval_s=float(gpu_metric_sample_interval_s),
    )
    return benchmark_direct_epfd_runs(
        cases,
        runner=runner,
        time_func=time_func,
    )


def benchmark_direct_epfd_runs(
    cases: Iterable[Mapping[str, Any]],
    *,
    runner: Callable[..., Any] | None = None,
    time_func: Callable[[], float] = perf_counter,
) -> list[dict[str, Any]]:
    """
    Run direct-EPFD benchmark/debug cases and collect scheduler telemetry.

    Parameters
    ----------
    cases : iterable of mapping
        Benchmark case definitions. Each mapping may either contain a nested
        ``"kwargs"`` mapping with the keyword arguments for
        :func:`run_gpu_direct_epfd`, or provide those keyword arguments at the
        top level. The optional ``"name"`` field labels the case in the
        returned summaries.
    runner : callable, optional
        Callable used to execute one case. Defaults to
        :func:`run_gpu_direct_epfd`. The callable must accept a
        ``progress_callback`` keyword argument compatible with the standard
        direct-EPFD progress payloads.
    time_func : callable, optional
        Monotonic timer used to measure wall-clock elapsed time.

        Returns
        -------
        list of dict
            One summary per case. Each summary includes wall-clock timing, planned
            versus active chunk shapes, retry/backoff indicators, predicted memory
            peaks, attempted work throughput derived from emitted
            ``"iteration_plan"`` and ``"batch_start"`` events, and any available
            run-level stage timings such as spectrum setup/weighting and power
            accumulation.

    Notes
    -----
    This helper is intentionally non-gating. It is meant for workstation-side
    performance investigations of direct-EPFD studies, including reuse-enabled
    cases such as F1, F4, and F7, without changing the main run API or the
    result-file schema.
    """
    runner_fn = run_gpu_direct_epfd if runner is None else runner
    summaries: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        case_map = dict(case)
        case_name = str(case_map.pop("name", f"case_{case_index}"))
        graceful_stop_after_first_batch = bool(
            case_map.pop("graceful_stop_after_first_batch", False)
        )
        graceful_stop_after_batch_count_raw = case_map.pop(
            "graceful_stop_after_batch_count",
            None,
        )
        graceful_stop_after_batch_count = (
            None
            if graceful_stop_after_batch_count_raw is None
            else max(1, int(graceful_stop_after_batch_count_raw))
        )
        if graceful_stop_after_batch_count is None and graceful_stop_after_first_batch:
            graceful_stop_after_batch_count = 1
        sample_live_gpu_metrics = bool(case_map.pop("sample_live_gpu_metrics", False))
        gpu_metric_sample_interval_s = float(
            case_map.pop("gpu_metric_sample_interval_s", 0.25)
        )
        gpu_metric_device_index = int(case_map.pop("gpu_metric_device_index", 0))
        kwargs_raw = case_map.pop("kwargs", None)
        run_kwargs = dict(case_map if kwargs_raw is None else kwargs_raw)
        user_progress_callback = run_kwargs.pop("progress_callback", None)
        user_cancel_callback = run_kwargs.pop("cancel_callback", None)
        progress_events: list[dict[str, Any]] = []
        benchmark_control = {"graceful_ready": False, "batch_count": 0}

        def _capture_progress(payload: Mapping[str, Any]) -> None:
            payload_dict = dict(payload)
            progress_events.append(payload_dict)
            if (
                graceful_stop_after_batch_count is not None
                and str(payload_dict.get("kind", "")).strip().lower() == "batch_start"
            ):
                benchmark_control["batch_count"] = int(benchmark_control["batch_count"]) + 1
                if int(benchmark_control["batch_count"]) >= int(graceful_stop_after_batch_count):
                    benchmark_control["graceful_ready"] = True
            if callable(user_progress_callback):
                user_progress_callback(payload_dict)

        def _capture_cancel() -> Any:
            if callable(user_cancel_callback):
                cancel_value = user_cancel_callback()
                if cancel_value is not None:
                    return cancel_value
            if (
                graceful_stop_after_batch_count is not None
                and benchmark_control["graceful_ready"]
            ):
                return "graceful"
            return None

        gpu_sampler = _make_benchmark_gpu_metric_sampler(
            enabled=bool(sample_live_gpu_metrics),
            sample_interval_s=float(gpu_metric_sample_interval_s),
            device_index=int(gpu_metric_device_index),
        )
        started = float(time_func())
        gpu_metric_samples: tuple[dict[str, Any], ...] = tuple()
        gpu_sampler.start()
        try:
            if graceful_stop_after_batch_count is not None or callable(user_cancel_callback):
                run_kwargs["cancel_callback"] = _capture_cancel
            result = runner_fn(progress_callback=_capture_progress, **run_kwargs)
        except Exception as exc:
            elapsed_seconds = max(0.0, float(time_func()) - started)
            gpu_metric_samples = gpu_sampler.stop()
            summary = _summarize_direct_epfd_progress_events(
                progress_events,
                elapsed_seconds=elapsed_seconds,
            )
            summary.update(
                _summarize_benchmark_gpu_metric_samples(gpu_metric_samples)
            )
            summary.update(
                name=case_name,
                ok=False,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            summaries.append(summary)
            continue
        else:
            gpu_metric_samples = gpu_sampler.stop()

        elapsed_seconds = max(0.0, float(time_func()) - started)
        summary = _summarize_direct_epfd_progress_events(
            progress_events,
            elapsed_seconds=elapsed_seconds,
        )
        summary.update(_summarize_benchmark_gpu_metric_samples(gpu_metric_samples))
        summary.update(name=case_name, ok=True)
        if isinstance(result, Mapping):
            run_stage_timings = result.get("profile_stage_timings_summary", None)
            profile_stage_rows = result.get("profile_stage_timings", None)
            if not run_stage_timings:
                for event in reversed(progress_events):
                    if str(event.get("kind", "")).strip().lower() == "run_complete":
                        run_stage_timings = event.get("profile_stage_timings_summary", None)
                        if run_stage_timings:
                            break
            aggregated_row_timings = _aggregate_direct_epfd_profile_timing_rows(
                profile_stage_rows if isinstance(profile_stage_rows, Iterable) else None
            )
            if aggregated_row_timings:
                merged_stage_timings = (
                    {}
                    if run_stage_timings is None
                    else {
                        str(key): float(value)
                        for key, value in dict(run_stage_timings).items()
                    }
                )
                for key, value in aggregated_row_timings.items():
                    merged_stage_timings.setdefault(str(key), float(value))
                run_stage_timings = merged_stage_timings
            summary.update(
                _summarize_direct_epfd_stage_timings(
                    None if run_stage_timings is None else dict(run_stage_timings),
                    elapsed_seconds=elapsed_seconds,
                )
            )
            filename_value = result.get("filename", result.get("storage_filename"))
            if filename_value is not None:
                summary["filename"] = str(filename_value)
            observed_stage_summaries = result.get("observed_stage_memory_summary_by_name", None)
            if not isinstance(observed_stage_summaries, Mapping):
                for event in reversed(progress_events):
                    if str(event.get("kind", "")).strip().lower() == "run_complete":
                        observed_stage_summaries = event.get(
                            "observed_stage_memory_summary_by_name",
                            None,
                        )
                        if isinstance(observed_stage_summaries, Mapping):
                            break
            if isinstance(observed_stage_summaries, Mapping):
                cell_link_summary = observed_stage_summaries.get("cell_link_library")
                beam_finalize_summary = observed_stage_summaries.get("beam_finalize")
                if isinstance(cell_link_summary, Mapping):
                    if cell_link_summary.get("observed_stage_gpu_resident_bytes") is not None:
                        summary["cell_link_library_resident_bytes_observed"] = int(
                            cell_link_summary["observed_stage_gpu_resident_bytes"]
                        )
                    if cell_link_summary.get("observed_stage_gpu_transient_peak_bytes") is not None:
                        summary["cell_link_library_transient_peak_bytes_observed"] = int(
                            cell_link_summary["observed_stage_gpu_transient_peak_bytes"]
                        )
                if isinstance(beam_finalize_summary, Mapping):
                    if beam_finalize_summary.get("observed_stage_gpu_transient_peak_bytes") is not None:
                        summary["beam_finalize_transient_peak_bytes_observed"] = int(
                            beam_finalize_summary["observed_stage_gpu_transient_peak_bytes"]
                        )
                    if beam_finalize_summary.get("planner_vs_observed_gpu_peak_error_bytes") is not None:
                        summary["planner_vs_observed_gpu_peak_error_bytes"] = int(
                            beam_finalize_summary["planner_vs_observed_gpu_peak_error_bytes"]
                        )
            finalize_substage_timings = result.get("beam_finalize_substage_timings", None)
            if not isinstance(finalize_substage_timings, Mapping):
                for event in reversed(progress_events):
                    if str(event.get("kind", "")).strip().lower() == "run_complete":
                        finalize_substage_timings = event.get("beam_finalize_substage_timings", None)
                        if isinstance(finalize_substage_timings, Mapping):
                            break
            if isinstance(finalize_substage_timings, Mapping):
                summary["beam_finalize_substage_timings"] = {
                    str(key): float(value)
                    for key, value in dict(finalize_substage_timings).items()
                }
            finalize_chunk_shape = result.get("beam_finalize_chunk_shape", None)
            if not isinstance(finalize_chunk_shape, Mapping):
                for event in reversed(progress_events):
                    if str(event.get("kind", "")).strip().lower() == "run_complete":
                        finalize_chunk_shape = event.get("beam_finalize_chunk_shape", None)
                        if isinstance(finalize_chunk_shape, Mapping):
                            break
            if isinstance(finalize_chunk_shape, Mapping):
                summary["beam_finalize_chunk_shape"] = {
                    str(key): int(value) for key, value in dict(finalize_chunk_shape).items()
                }
            boresight_compaction_stats = result.get("boresight_compaction_stats", None)
            if not isinstance(boresight_compaction_stats, Mapping):
                for event in reversed(progress_events):
                    if str(event.get("kind", "")).strip().lower() == "run_complete":
                        boresight_compaction_stats = event.get("boresight_compaction_stats", None)
                        if isinstance(boresight_compaction_stats, Mapping):
                            break
            if isinstance(boresight_compaction_stats, Mapping):
                summary["boresight_compaction_stats"] = {
                    str(key): int(value) for key, value in dict(boresight_compaction_stats).items()
                }
            if result.get("hot_path_device_to_host_copy_count") is not None:
                summary["hot_path_device_to_host_copy_count"] = int(
                    result["hot_path_device_to_host_copy_count"]
                )
            if result.get("hot_path_device_to_host_copy_bytes") is not None:
                summary["hot_path_device_to_host_copy_bytes"] = int(
                    result["hot_path_device_to_host_copy_bytes"]
                )
            if result.get("device_scalar_sync_count") is not None:
                summary["device_scalar_sync_count"] = int(
                    result["device_scalar_sync_count"]
                )
        summaries.append(summary)

    return summaries


def analyse_time(times: Any) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """
    Compute timing stats for a single time series or a stacked set.

    Input
    -----
    times : astropy.time.Time | Quantity (MJD) | array-like (MJD)
        Shapes supported:
          (T,)           → one iteration with T samples
          (N, T)         → N iterations, T samples each
          (N, T, 1)      → will be squeezed to (N, T)
        NaNs are allowed (padding) and ignored.

    Output
    ------
    (timestep, iter_durations, total_duration)
        timestep        : mean consecutive spacing across all iterations [s]
        iter_durations  : per-iteration (last - first) duration [s], shape (N,)
        total_duration  : sum(iter_durations) [s]
    """
    # --- normalise to float MJD array -----------------------------------------
    if isinstance(times, Time):
        mjd = np.asarray(times.mjd, dtype=float)
    elif hasattr(times, "unit"):
        try:
            mjd = np.asarray(u.Quantity(times).to(u.day).value, dtype=float)
        except Exception:
            mjd = np.asarray(times, dtype=float)
    else:
        mjd = np.asarray(times, dtype=float)

    # Drop singleton axes so (N, T, 1) → (N, T)
    mjd = np.squeeze(mjd)

    # Ensure (N, T) shape
    if mjd.ndim == 0:
        mjd = mjd.reshape(1, 1)
    elif mjd.ndim == 1:
        mjd = mjd.reshape(1, -1)
    elif mjd.ndim > 2:
        # Flatten everything after the first axis into the time axis.
        mjd = mjd.reshape(mjd.shape[0], -1)

    n_iter, T = mjd.shape
    day_to_sec = u.day.to(u.s)

    # --- collect diffs and durations ------------------------------------------
    all_deltas_sec: list[float] = []
    iter_durs_sec = np.zeros(n_iter, dtype=float)

    for i in range(n_iter):
        row = mjd[i]
        mask = np.isfinite(row)
        if not np.any(mask):
            iter_durs_sec[i] = 0.0
            continue

        t = row[mask]
        # Ensure increasing order before diffs
        t = t[np.argsort(t)]

        if t.size >= 2:
            diffs = np.diff(t) * day_to_sec
            diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
            if diffs.size:
                all_deltas_sec.extend(diffs.tolist())
                iter_durs_sec[i] = (t[-1] - t[0]) * day_to_sec
            else:
                iter_durs_sec[i] = 0.0
        else:
            iter_durs_sec[i] = 0.0

    if len(all_deltas_sec) < 1:
        raise ValueError("Need at least two valid time points globally to compute timestep.")

    timestep        = (np.mean(all_deltas_sec)) * u.s
    iter_durations  = iter_durs_sec * u.s
    total_duration  = np.sum(iter_durs_sec) * u.s
    return timestep, iter_durations, total_duration

_DAY_TO_SEC = u.day.to(u.s)
_TIME_EQUAL_ATOL_SEC = 1.0e-6
_TIME_EQUAL_RTOL = 1.0e-9


def _normalise_process_integration_times(times: Any) -> np.ndarray:
    """Return `times` as a squeezed float array in MJD days."""
    if isinstance(times, Time):
        mjd = np.asarray(times.mjd, dtype=np.float64)
    elif hasattr(times, "unit"):
        try:
            mjd = np.asarray(u.Quantity(times).to_value(u.day), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise ValueError("`times` must be convertible to days or provided as MJD-like values.") from exc
    else:
        mjd = np.asarray(times, dtype=np.float64)

    mjd = np.squeeze(mjd)
    if mjd.ndim == 0:
        mjd = mjd.reshape(1)
    if mjd.ndim > 2:
        raise ValueError("`times` must be 1D or 2D after squeezing singleton axes.")
    return mjd


def _integration_output_dtype(values: np.ndarray) -> np.dtype:
    """Preserve float32/float64 outputs while promoting integer-like inputs."""
    dtype = np.dtype(values.dtype)
    if np.issubdtype(dtype, np.floating) and dtype.itemsize >= np.dtype(np.float32).itemsize:
        return dtype
    return np.dtype(np.float64)


def _split_contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive slices for contiguous True runs."""
    if mask.ndim != 1:
        raise ValueError("Expected a 1D boolean mask for contiguous-run detection.")

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    split_points = np.flatnonzero(np.diff(idx) != 1) + 1
    chunks = np.split(idx, split_points)
    return [(int(chunk[0]), int(chunk[-1] + 1)) for chunk in chunks]


def _is_near_uniform_timestep(diffs_sec: np.ndarray) -> bool:
    """Return True when `diffs_sec` represent a numerically uniform cadence."""
    if diffs_sec.size == 0:
        return False

    reference = float(np.mean(diffs_sec))
    atol = max(_TIME_EQUAL_ATOL_SEC, _TIME_EQUAL_RTOL * max(abs(reference), 1.0))
    return bool(np.all(np.abs(diffs_sec - reference) <= atol))


def _window_time_tolerance(period_sec: float, span_sec: float) -> float:
    """Tolerance used when comparing accumulated durations to a target window."""
    return max(_TIME_EQUAL_ATOL_SEC, _TIME_EQUAL_RTOL * max(period_sec, span_sec, 1.0))


def _uniform_window_starts(
    sample_count: int,
    window_size: int,
    windowing: str,
    random_window_count: int | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Return local sample-start indices for uniform-cadence windows."""
    max_start = sample_count - window_size
    if max_start < 0:
        return np.empty(0, dtype=np.int64)

    if windowing == "sliding":
        return np.arange(max_start + 1, dtype=np.int64)
    if windowing == "subsequent":
        return np.arange(0, max_start + 1, window_size, dtype=np.int64)
    if windowing != "random":  # pragma: no cover - validated by caller
        raise ValueError(f"Unsupported windowing mode {windowing!r}.")

    if random_window_count is None:
        raise ValueError("`random_window_count` must be provided when windowing='random'.")
    if rng is None:  # pragma: no cover - guarded by caller
        raise ValueError("A random-number generator is required for random window selection.")

    candidates = np.arange(max_start + 1, dtype=np.int64)
    if candidates.size < random_window_count:
        raise ValueError(
            "windowing='random' requires at least `random_window_count` full windows "
            "in every processed slice."
        )

    chosen = rng.choice(candidates, size=random_window_count, replace=False)
    chosen.sort()
    return chosen.astype(np.int64, copy=False)


def _uniform_window_means(values: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    """Compute uniform-window means for selected start indices using prefix sums."""
    prefix = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, axis=0, dtype=np.float64, out=prefix[1:])
    return (prefix[starts + window_size] - prefix[starts]) / float(window_size)


def _irregular_sliding_window_starts(boundaries: np.ndarray, period_sec: float) -> np.ndarray:
    """Return start indices whose forward-hold windows have the full duration."""
    tolerance = _window_time_tolerance(period_sec, float(boundaries[-1]))
    remaining = boundaries[-1] - boundaries[:-1]
    return np.flatnonzero(remaining + tolerance >= period_sec).astype(np.int64, copy=False)


def _irregular_subsequent_window_starts(boundaries: np.ndarray, period_sec: float) -> np.ndarray:
    """Return non-overlapping full-window starts for irregular cadences."""
    eligible = _irregular_sliding_window_starts(boundaries, period_sec)
    if eligible.size == 0:
        return eligible

    tolerance = _window_time_tolerance(period_sec, float(boundaries[-1]))
    selected: list[int] = []
    next_allowed = -np.inf

    for start in eligible:
        start_time = float(boundaries[start])
        if start_time + tolerance < next_allowed:
            continue
        selected.append(int(start))
        next_allowed = start_time + period_sec

    return np.asarray(selected, dtype=np.int64)


def _irregular_window_means(
    values: np.ndarray,
    boundaries: np.ndarray,
    starts: np.ndarray,
    period_sec: float,
) -> np.ndarray:
    """Compute forward-hold time-weighted means for selected irregular windows."""
    durations = np.diff(boundaries)
    energy_prefix = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    energy_prefix[0] = 0.0
    np.multiply(values, durations[:, np.newaxis], out=energy_prefix[1:], casting="unsafe")
    np.cumsum(energy_prefix[1:], axis=0, dtype=np.float64, out=energy_prefix[1:])

    targets = boundaries[starts] + period_sec
    end_idx = np.searchsorted(boundaries, targets, side="right") - 1
    end_idx = np.clip(end_idx, 0, values.shape[0] - 1)

    target_energy = energy_prefix[end_idx].copy()
    target_energy += (targets - boundaries[end_idx])[:, np.newaxis] * values[end_idx]
    return (target_energy - energy_prefix[starts]) / period_sec


def _prepare_segment_windows(
    values: np.ndarray,
    times_mjd: np.ndarray | None,
    integration_samples: int | None,
    integration_seconds: float | None,
    windowing: str,
) -> tuple[str, np.ndarray, dict[str, Any]] | None:
    """Describe the integration work for one contiguous valid segment."""
    if values.shape[0] == 0:
        return None

    if integration_samples is not None:
        starts = _uniform_window_starts(
            values.shape[0],
            integration_samples,
            "sliding" if windowing == "random" else windowing,
            None,
            None,
        )
        if starts.size == 0:
            return None
        return (
            "uniform",
            starts,
            {
                "values": values,
                "window_size": integration_samples,
            },
        )

    if integration_seconds is None or times_mjd is None:  # pragma: no cover - validated by caller
        raise ValueError("Time-based integration requires both `times_mjd` and `integration_seconds`.")

    if values.shape[0] < 2:
        return None

    times_sec = (times_mjd - times_mjd[0]) * _DAY_TO_SEC
    diffs = np.diff(times_sec)
    if np.any(~np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError("`times` must be strictly increasing within each contiguous valid segment.")

    if _is_near_uniform_timestep(diffs):
        dt_sec = float(np.mean(diffs))
        window_size = max(1, int(round(integration_seconds / dt_sec)))
        starts = _uniform_window_starts(
            values.shape[0],
            window_size,
            "sliding" if windowing == "random" else windowing,
            None,
            None,
        )
        if starts.size == 0:
            return None
        return (
            "uniform",
            starts,
            {
                "values": values,
                "window_size": window_size,
            },
        )

    durations = np.empty(values.shape[0], dtype=np.float64)
    durations[:-1] = diffs
    durations[-1] = diffs[-1]

    boundaries = np.empty(values.shape[0] + 1, dtype=np.float64)
    boundaries[0] = 0.0
    np.cumsum(durations, axis=0, dtype=np.float64, out=boundaries[1:])

    if windowing == "subsequent":
        starts = _irregular_subsequent_window_starts(boundaries, integration_seconds)
    else:
        starts = _irregular_sliding_window_starts(boundaries, integration_seconds)

    if starts.size == 0:
        return None

    return (
        "irregular",
        starts,
        {
            "values": values,
            "boundaries": boundaries,
            "period_sec": integration_seconds,
        },
    )


def _compute_segment_window_means(kind: str, starts: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """Evaluate the windows described by `_prepare_segment_windows`."""
    if kind == "uniform":
        return _uniform_window_means(payload["values"], starts, payload["window_size"])
    if kind == "irregular":
        return _irregular_window_means(payload["values"], payload["boundaries"], starts, payload["period_sec"])
    raise ValueError(f"Unsupported segment kind {kind!r}.")


def _process_integration_batch(
    values: np.ndarray,
    *,
    times_row: np.ndarray | None,
    integration_samples: int | None,
    integration_seconds: float | None,
    windowing: str,
    random_window_count: int | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Integrate one flattened batch with shape `(T, lanes)`."""
    row_valid = np.all(np.isfinite(values), axis=1)
    if times_row is not None:
        row_valid &= np.isfinite(times_row)

    segments = _split_contiguous_true_runs(row_valid)
    if not segments:
        if windowing == "random" and random_window_count:
            raise ValueError(
                "windowing='random' requires at least `random_window_count` full windows "
                "in every processed slice."
            )
        return np.empty((0, values.shape[1]), dtype=np.float64)

    if windowing != "random":
        outputs: list[np.ndarray] = []

        for start, stop in segments:
            segment = _prepare_segment_windows(
                values[start:stop],
                None if times_row is None else times_row[start:stop],
                integration_samples,
                integration_seconds,
                windowing,
            )
            if segment is None:
                continue
            kind, starts, payload = segment
            outputs.append(_compute_segment_window_means(kind, starts, payload))

        if not outputs:
            return np.empty((0, values.shape[1]), dtype=np.float64)
        return np.concatenate(outputs, axis=0)

    if random_window_count is None:
        raise ValueError("`random_window_count` must be provided when windowing='random'.")
    if rng is None:  # pragma: no cover - guarded by caller
        raise ValueError("A random-number generator is required for random window selection.")

    candidate_segments: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    candidate_sizes: list[int] = []

    for start, stop in segments:
        segment = _prepare_segment_windows(
            values[start:stop],
            None if times_row is None else times_row[start:stop],
            integration_samples,
            integration_seconds,
            "random",
        )
        if segment is None:
            continue
        candidate_segments.append(segment)
        candidate_sizes.append(int(segment[1].size))

    total_candidates = int(np.sum(candidate_sizes, dtype=np.int64))
    if total_candidates < random_window_count:
        raise ValueError(
            "windowing='random' requires at least `random_window_count` full windows "
            "in every processed slice."
        )

    chosen = np.sort(rng.choice(total_candidates, size=random_window_count, replace=False))
    outputs: list[np.ndarray] = []
    offset = 0

    for (kind, starts, payload), size in zip(candidate_segments, candidate_sizes):
        mask = (chosen >= offset) & (chosen < offset + size)
        if np.any(mask):
            local_positions = chosen[mask] - offset
            local_starts = starts[local_positions]
            outputs.append(_compute_segment_window_means(kind, local_starts, payload))
        offset += size

    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, values.shape[1]), dtype=np.float64)


def process_integration(
    epfd: Any,
    *,
    integration_period: Any = 2000 * u.s,
    times: Any | None = None,
    timestep: Any | None = None,
    windowing: str = "sliding",
    time_axis: int = 1,
    random_window_count: int | None = None,
    random_seed: int | None = None,
) -> Any:
    """
    Average linear power-like samples over a configurable integration window.

    This helper is intended for RA.769-style post-processing where the
    physically meaningful operation is a linear-domain average over time.
    Logarithmic quantities, such as dB-valued Astropy quantities, are rejected;
    convert those to linear units before calling this function and convert back
    to dB only after the integration step.

    Parameters
    ----------
    epfd : array-like or astropy.Quantity
        Linear, non-negative samples with time along `time_axis`. Expected
        shapes are `(T,)`, `(I, T)`, `(I, T, C)`, or any array where the
        dimensions before `time_axis` represent independent batches and the
        dimensions after `time_axis` are lanes processed together. Non-finite
        samples mark the whole sample time as unavailable, so windows do not
        span those samples.
    integration_period : int, float, or Quantity
        Window length. Plain numeric values are interpreted as a sample count.
        Time-valued quantities are interpreted in seconds and require either
        `times` or `timestep`.
    times : astropy.time.Time or array-like, optional
        Time stamps aligned to `epfd` along the time axis. Supported shapes are
        `(T,)` for one shared time axis across all batches, `(B, T)` for one
        time row per flattened batch, and `(B, T, 1)` which is squeezed to
        `(B, T)`. When provided, near-uniform cadences use a fast prefix-sum
        path and irregular cadences use exact forward-hold time weighting.
    timestep : float or Quantity, optional
        Fixed cadence override in seconds. This is used only when `times` is
        omitted and `integration_period` is a time-valued quantity.
    windowing : {"sliding", "subsequent", "random"}
        Window-selection mode. `"sliding"` evaluates overlapping full windows
        aligned to each eligible start sample. `"subsequent"` evaluates
        back-to-back non-overlapping full windows and drops any partial tail.
        `"random"` samples `random_window_count` full windows without
        replacement from the same eligible start positions as `"sliding"`.
    time_axis : int
        Axis index corresponding to time.
    random_window_count : int, optional
        Number of windows to sample per batch when `windowing="random"`.
    random_seed : int, optional
        Seed used for reproducible random-window selection.

    Returns
    -------
    averaged : numpy.ndarray or astropy.Quantity
        Time-averaged samples in the same linear unit as `epfd` when units are
        present. The returned time axis contains only full windows:

        - `"sliding"` returns one sample per eligible start position.
        - `"subsequent"` returns one sample per non-overlapping full window.
        - `"random"` returns exactly `random_window_count` samples per batch.

        When different batches yield different numbers of windows, the result is
        padded with `NaN` values along the time axis to keep the output
        stackable.

    Raises
    ------
    ValueError
        Raised when the inputs are not linear, contain negative finite samples,
        request unsupported windowing semantics, or do not contain enough full
        windows for the requested operation.

    Notes
    -----
    - The integration is always carried out in the linear domain. This is the
      correct physical behavior for RA.769-style averaging.
    - For irregular `times`, each sample is interpreted with forward-hold
      semantics: sample `i` represents the interval from `t[i]` to `t[i+1]`.
      The final sample in a contiguous segment is held for one additional copy
      of the previous positive timestep so that full windows ending inside that
      last hold interval remain representable.
    - Plain numeric `integration_period` values are sample counts even if
      `times` is provided.
    - The implementation processes one flattened batch at a time and uses
      prefix sums or prefix energy integrals rather than materializing rolling
      windows, which keeps memory usage bounded by one batch plus its prefix
      accumulator.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from scepter import scenario
    >>> power = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * u.W
    >>> averaged = scenario.process_integration(
    ...     power,
    ...     integration_period=2,
    ...     windowing="sliding",
    ...     time_axis=0,
    ... )
    >>> averaged.value
    array([1.5, 2.5, 3.5], dtype=float32)
    """
    has_unit = hasattr(epfd, "unit")
    unit = epfd.unit if has_unit else None
    if has_unit and isinstance(unit, u.LogUnit):
        raise ValueError("`process_integration` requires linear-domain input, not logarithmic quantities.")

    vals = np.asarray(epfd.value if has_unit else epfd)
    result_dtype = _integration_output_dtype(vals)

    if vals.ndim < 1:
        raise ValueError("`epfd` must have at least one dimension (time axis).")
    if time_axis < 0:
        time_axis = vals.ndim + time_axis
    if not (0 <= time_axis < vals.ndim):
        raise ValueError("`time_axis` is out of range.")

    time_length = vals.shape[time_axis]
    if time_length < 1:
        raise ValueError("Time axis has length 0.")

    finite_negative = np.isfinite(vals) & (vals < 0.0)
    if np.any(finite_negative):
        raise ValueError("`epfd` must not contain negative finite values.")

    windowing = windowing.lower().strip()
    if windowing not in {"sliding", "subsequent", "random"}:
        raise ValueError("`windowing` must be 'sliding', 'subsequent', or 'random'.")

    if windowing == "random":
        if random_window_count is None or int(random_window_count) < 1:
            raise ValueError("`random_window_count` must be a positive integer when windowing='random'.")
        random_window_count = int(random_window_count)
    elif random_window_count is not None:
        raise ValueError("`random_window_count` is only supported when windowing='random'.")

    integration_samples: int | None = None
    integration_seconds: float | None = None

    if hasattr(integration_period, "to"):
        integration_seconds = float(u.Quantity(integration_period).to_value(u.s))
        if integration_seconds <= 0.0:
            raise ValueError("`integration_period` must be > 0.")
        if times is None and timestep is None:
            raise ValueError(
                "Provide `times` or `timestep` when `integration_period` carries time units."
            )
    else:
        integration_samples = int(round(float(integration_period)))
        if integration_samples < 1:
            raise ValueError("Numeric `integration_period` values must round to at least one sample.")

    if times is None and timestep is not None and integration_seconds is not None:
        dt_sec = float(u.Quantity(timestep).to_value(u.s)) if hasattr(timestep, "to") else float(timestep)
        if dt_sec <= 0.0:
            raise ValueError("`timestep` must be > 0.")
        integration_samples = max(1, int(round(integration_seconds / dt_sec)))
        integration_seconds = None

    prefix_shape = vals.shape[:time_axis]
    suffix_shape = vals.shape[time_axis + 1 :]
    batch_count = int(np.prod(prefix_shape, dtype=np.int64)) if prefix_shape else 1
    lane_count = int(np.prod(suffix_shape, dtype=np.int64)) if suffix_shape else 1
    values_view = np.reshape(vals, (batch_count, time_length, lane_count))

    times_array: np.ndarray | None = None
    if times is not None:
        times_array = _normalise_process_integration_times(times)
        if times_array.ndim == 1:
            if times_array.shape[0] != time_length:
                raise ValueError("1D `times` must have the same length as the time axis of `epfd`.")
        else:
            if times_array.shape[1] != time_length:
                raise ValueError("The trailing dimension of `times` must match the time axis of `epfd`.")
            if times_array.shape[0] != batch_count:
                raise ValueError(
                    "2D `times` must provide one row per flattened batch formed by the axes before `time_axis`."
                )

    rng = np.random.default_rng(random_seed) if windowing == "random" else None
    batch_outputs: list[np.ndarray] = []
    max_windows = 0

    for batch_index in range(batch_count):
        time_row = None
        if times_array is not None:
            time_row = times_array if times_array.ndim == 1 else times_array[batch_index]

        integrated = _process_integration_batch(
            values_view[batch_index],
            times_row=time_row,
            integration_samples=integration_samples,
            integration_seconds=integration_seconds,
            windowing=windowing,
            random_window_count=random_window_count,
            rng=rng,
        )
        batch_outputs.append(integrated)
        max_windows = max(max_windows, integrated.shape[0])

    padded = np.full((batch_count, max_windows, lane_count), np.nan, dtype=result_dtype)
    for batch_index, integrated in enumerate(batch_outputs):
        if integrated.size == 0:
            continue
        padded[batch_index, : integrated.shape[0]] = integrated.astype(result_dtype, copy=False)

    out_shape = prefix_shape + (max_windows,) + suffix_shape
    out_vals = padded.reshape(out_shape)
    return (out_vals * unit) if has_unit else out_vals


def process_integration_stream(
    chunks: Iterable[Mapping[str, Any]],
    *,
    value_key: str,
    integration_period: Any = 2000 * u.s,
    times_key: str | None = "times",
    timestep: Any | None = None,
    windowing: str = "sliding",
) -> Any:
    """
    Streamed variant of :func:`process_integration` for timestep-chunk iterators.

    Parameters
    ----------
    chunks : iterable of mapping
        Chunk iterator produced by :func:`iter_data_chunks` or
        ``read_data(..., mode="stream")["stream"]``. Each chunk must provide
        ``"iteration"`` and ``"data"`` keys.
    value_key : str
        Dataset name inside each chunk's ``"data"`` dictionary.
    integration_period : int, float, or Quantity, optional
        Same semantics as :func:`process_integration`.
    times_key : str, optional
        Dataset name for chunk-local times. Required for time-valued integration
        periods unless ``timestep`` is supplied.
    timestep : float or Quantity, optional
        Fixed cadence override in seconds when ``times_key`` is omitted.
    windowing : {"sliding", "subsequent"}, optional
        Streaming currently supports exact sliding and subsequent windows.

    Returns
    -------
    numpy.ndarray or astropy.Quantity
        Integrated values stacked across iterations with the same padding
        convention as :func:`process_integration`.

    Raises
    ------
    ValueError
        Raised when the streamed input is non-finite, malformed, requests
        ``windowing="random"``, or cannot satisfy the requested integration
        semantics.
    """
    windowing = windowing.lower().strip()
    if windowing not in {"sliding", "subsequent"}:
        raise ValueError("`process_integration_stream` supports only 'sliding' and 'subsequent'.")

    integration_samples: int | None = None
    integration_seconds: float | None = None
    if hasattr(integration_period, "to"):
        integration_seconds = float(u.Quantity(integration_period).to_value(u.s))
        if integration_seconds <= 0.0:
            raise ValueError("`integration_period` must be > 0.")
        if times_key is None and timestep is None:
            raise ValueError(
                "Provide `times_key` or `timestep` when `integration_period` carries time units."
            )
    else:
        integration_samples = int(round(float(integration_period)))
        if integration_samples < 1:
            raise ValueError("Numeric `integration_period` values must round to at least one sample.")

    if integration_seconds is not None and times_key is None and timestep is not None:
        dt_sec = float(u.Quantity(timestep).to_value(u.s)) if hasattr(timestep, "to") else float(timestep)
        if dt_sec <= 0.0:
            raise ValueError("`timestep` must be > 0.")
        integration_samples = max(1, int(round(integration_seconds / dt_sec)))
        integration_seconds = None

    iter_order: list[int] = []
    outputs_by_iter: dict[int, list[np.ndarray]] = {}
    state_by_iter: dict[int, dict[str, Any]] = {}
    global_unit: Any | None = None
    suffix_shape: tuple[int, ...] | None = None
    lane_count: int | None = None
    result_dtype: np.dtype | None = None

    def _ensure_iter_state(iteration: int) -> dict[str, Any]:
        state = state_by_iter.get(iteration)
        if state is None:
            state = {
                "values": np.empty((0, int(lane_count or 0)), dtype=result_dtype or np.float64),
                "times": np.empty((0,), dtype=np.float64),
                "next_allowed_mjd": None,
            }
            state_by_iter[iteration] = state
            outputs_by_iter[iteration] = []
            iter_order.append(iteration)
        return state

    def _emit_uniform(state: dict[str, Any], final: bool = False) -> None:
        values = state["values"]
        if integration_samples is None:
            raise RuntimeError("Uniform streaming emission requires `integration_samples`.")
        if values.shape[0] < integration_samples:
            return
        if windowing == "sliding":
            starts = np.arange(values.shape[0] - integration_samples + 1, dtype=np.int64)
            outputs_by_iter[state["iteration"]].append(_uniform_window_means(values, starts, integration_samples))
            keep = integration_samples - 1
            state["values"] = values[-keep:].copy() if keep > 0 else np.empty((0, values.shape[1]), dtype=values.dtype)
            return

        n_full = int(values.shape[0] // integration_samples)
        if n_full > 0:
            starts = np.arange(0, n_full * integration_samples, integration_samples, dtype=np.int64)
            outputs_by_iter[state["iteration"]].append(_uniform_window_means(values, starts, integration_samples))
        state["values"] = values[n_full * integration_samples :].copy()

    def _emit_irregular(state: dict[str, Any], *, final: bool) -> None:
        values = state["values"]
        times_mjd = state["times"]
        if integration_seconds is None:
            raise RuntimeError("Irregular streaming emission requires `integration_seconds`.")
        if times_mjd.size < 2 or values.shape[0] < 2:
            return
        if np.any(np.diff(times_mjd) <= 0.0):
            raise ValueError("`times` must be strictly increasing within each streamed iteration.")

        if final:
            times_sec = (times_mjd - times_mjd[0]) * _DAY_TO_SEC
            diffs = np.diff(times_sec)
            durations = np.empty(values.shape[0], dtype=np.float64)
            durations[:-1] = diffs
            durations[-1] = diffs[-1]
            boundaries = np.empty(values.shape[0] + 1, dtype=np.float64)
            boundaries[0] = 0.0
            np.cumsum(durations, axis=0, dtype=np.float64, out=boundaries[1:])
            if windowing == "subsequent":
                starts = _irregular_subsequent_window_starts(boundaries, integration_seconds)
                next_allowed_mjd = state["next_allowed_mjd"]
                if next_allowed_mjd is not None and starts.size > 0:
                    starts = starts[times_mjd[starts] >= next_allowed_mjd]
                if starts.size > 0:
                    state["next_allowed_mjd"] = float(times_mjd[starts[-1]]) + (integration_seconds / _DAY_TO_SEC)
            else:
                starts = _irregular_sliding_window_starts(boundaries, integration_seconds)
            if starts.size > 0:
                outputs_by_iter[state["iteration"]].append(
                    _irregular_window_means(values, boundaries, starts, integration_seconds)
                )
            state["values"] = np.empty((0, values.shape[1]), dtype=values.dtype)
            state["times"] = np.empty((0,), dtype=np.float64)
            return

        usable_values = values[:-1]
        times_sec = (times_mjd - times_mjd[0]) * _DAY_TO_SEC
        boundaries = times_sec
        if windowing == "subsequent":
            starts = _irregular_subsequent_window_starts(boundaries, integration_seconds)
            next_allowed_mjd = state["next_allowed_mjd"]
            if next_allowed_mjd is not None and starts.size > 0:
                starts = starts[times_mjd[starts] >= next_allowed_mjd]
            if starts.size > 0:
                outputs_by_iter[state["iteration"]].append(
                    _irregular_window_means(usable_values, boundaries, starts, integration_seconds)
                )
                state["next_allowed_mjd"] = float(times_mjd[starts[-1]]) + (integration_seconds / _DAY_TO_SEC)
        else:
            starts = _irregular_sliding_window_starts(boundaries, integration_seconds)
            if starts.size > 0:
                outputs_by_iter[state["iteration"]].append(
                    _irregular_window_means(usable_values, boundaries, starts, integration_seconds)
                )

        retain_threshold = float(times_mjd[-1]) - (integration_seconds / _DAY_TO_SEC)
        retain_start = int(np.searchsorted(times_mjd, retain_threshold, side="right"))
        state["values"] = values[retain_start:].copy()
        state["times"] = times_mjd[retain_start:].copy()

    for chunk in chunks:
        iteration = int(chunk["iteration"])
        data = chunk["data"]
        if value_key not in data:
            raise KeyError(f"Chunk for iteration {iteration} does not contain {value_key!r}.")
        raw_values = data[value_key]
        has_unit = hasattr(raw_values, "unit")
        unit = raw_values.unit if has_unit else None
        if has_unit and isinstance(unit, u.LogUnit):
            raise ValueError("`process_integration_stream` requires linear-domain input, not logarithmic quantities.")
        if global_unit is None:
            global_unit = unit
        elif has_unit and str(unit) != str(global_unit):
            raise ValueError("All streamed chunks must share the same unit.")

        vals = np.asarray(raw_values.value if has_unit else raw_values)
        if vals.ndim < 1:
            raise ValueError("Streamed integration requires chunk arrays with a time axis.")
        if not np.all(np.isfinite(vals)):
            raise ValueError("`process_integration_stream` currently requires finite chunk values.")

        local_suffix_shape = tuple(vals.shape[1:])
        local_lane_count = int(np.prod(local_suffix_shape, dtype=np.int64)) if local_suffix_shape else 1
        if suffix_shape is None:
            suffix_shape = local_suffix_shape
            lane_count = local_lane_count
            result_dtype = _integration_output_dtype(vals)
        else:
            if local_suffix_shape != suffix_shape:
                raise ValueError("All streamed chunks must share the same trailing shape.")
            if local_lane_count != lane_count:
                raise ValueError("All streamed chunks must share the same lane count.")

        state = _ensure_iter_state(iteration)
        state["iteration"] = iteration
        vals_2d = np.asarray(vals, dtype=result_dtype).reshape(vals.shape[0], int(lane_count))
        state["values"] = np.concatenate([state["values"], vals_2d], axis=0)

        if integration_seconds is not None:
            if times_key is None:
                raise ValueError("`times_key` is required for time-valued streaming integration unless `timestep` is given.")
            if times_key not in data:
                raise KeyError(f"Chunk for iteration {iteration} does not contain {times_key!r}.")
            times_chunk = np.asarray(_normalise_process_integration_times(data[times_key]), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(times_chunk)):
                raise ValueError("`process_integration_stream` currently requires finite chunk times.")
            if times_chunk.shape[0] != vals.shape[0]:
                raise ValueError("Chunk-local times must have the same slot length as the values chunk.")
            if state["times"].size > 0 and times_chunk.size > 0 and float(times_chunk[0]) <= float(state["times"][-1]):
                raise ValueError("Streamed chunk times must be strictly increasing across chunk boundaries.")
            state["times"] = np.concatenate([state["times"], times_chunk], axis=0)
            _emit_irregular(state, final=False)
        else:
            _emit_uniform(state, final=False)

    if suffix_shape is None or lane_count is None or result_dtype is None:
        raise ValueError("No streamed chunks were provided to `process_integration_stream`.")

    for iteration in iter_order:
        state = state_by_iter[iteration]
        if integration_seconds is not None:
            _emit_irregular(state, final=True)
        else:
            _emit_uniform(state, final=True)

    integrated_rows: list[np.ndarray] = []
    max_windows = 0
    for iteration in iter_order:
        outputs = outputs_by_iter[iteration]
        merged = np.concatenate(outputs, axis=0) if outputs else np.empty((0, int(lane_count)), dtype=result_dtype)
        integrated_rows.append(merged)
        max_windows = max(max_windows, int(merged.shape[0]))

    padded = np.full((len(iter_order), max_windows, int(lane_count)), np.nan, dtype=result_dtype)
    for idx, merged in enumerate(integrated_rows):
        if merged.size == 0:
            continue
        padded[idx, : merged.shape[0], :] = merged

    out_shape = (len(iter_order), max_windows) + tuple(suffix_shape)
    out_vals = padded.reshape(out_shape)
    return (out_vals * global_unit) if global_unit is not None else out_vals
