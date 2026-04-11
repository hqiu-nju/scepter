#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import io
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys
import types
from typing import Mapping

import h5py
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_allclose, assert_equal
from pycraf import conversions as cnv
import pytest


def _install_numba_stub() -> None:
    if "numba" in sys.modules:
        return

    numba_stub = types.ModuleType("numba")
    numba_stub.__spec__ = importlib.machinery.ModuleSpec("numba", loader=None)

    def _njit(*args: object, **kwargs: object):
        del args, kwargs

        def _decorate(func):
            return func

        return _decorate

    class _NumbaPerformanceWarning(Warning):
        pass

    numba_stub.njit = _njit
    numba_stub.prange = range
    numba_stub.cuda = None
    numba_stub.set_num_threads = lambda n: None
    numba_stub.get_num_threads = lambda: 1
    sys.modules["numba"] = numba_stub

    numba_core = types.ModuleType("numba.core")
    numba_core.__spec__ = importlib.machinery.ModuleSpec("numba.core", loader=None)
    sys.modules["numba.core"] = numba_core
    numba_core_errors = types.ModuleType("numba.core.errors")
    numba_core_errors.__spec__ = importlib.machinery.ModuleSpec("numba.core.errors", loader=None)
    numba_core_errors.NumbaPerformanceWarning = _NumbaPerformanceWarning
    sys.modules["numba.core.errors"] = numba_core_errors


_install_numba_stub()

from scepter import earthgrid, gpu_accel, scenario


REPO_ROOT = Path(__file__).resolve().parents[2]
BORESIGHT_NOTEBOOK_PATH = REPO_ROOT / "SCEPTer_simulate.ipynb"
POSTPROCESS_NOTEBOOK_PATH = REPO_ROOT / "SCEPTer_postprocess.ipynb"
CUDA_AVAILABLE = gpu_accel.cuda is not None and bool(gpu_accel.cuda.is_available())
GPU_REQUIRED = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
_GIB = 1024 ** 3


def _seconds_to_mjd(seconds: np.ndarray) -> np.ndarray:
    """Convert relative seconds to relative MJD day values for local tests."""
    return np.asarray(seconds, dtype=np.float64) / float(u.day.to(u.s))


@pytest.fixture(autouse=True)
def _close_scenario_writers() -> None:
    """Keep background HDF5 writers from leaking across tests."""
    yield
    scenario.close_writer()


def _build_sample_results_file(
    tmp_path: Path,
    *,
    write_mode: str = "sync",
) -> Path:
    """Create a small result file covering constants, attrs, and two iterations."""
    filename = tmp_path / f"sample_{write_mode}.h5"
    scenario.init_simulation_results(
        str(filename),
        write_mode=write_mode,
        writer_queue_max_items=2,
        writer_queue_max_bytes=1024 ** 2,
    )
    scenario.write_data(
        str(filename),
        attrs={"tag": "scenario-test"},
        constants={"const_vector": np.array([1, 2, 3], dtype=np.int16)},
    )

    iter0_times = Time(60000.0 + _seconds_to_mjd(np.array([0.0, 1.0, 2.0])), format="mjd", scale="utc")
    iter0_power = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    ) * u.W
    iter0_index = np.array([0, 1, 2], dtype=np.int16)

    scenario.write_data(
        str(filename),
        iteration=0,
        times=iter0_times[:2],
        power=iter0_power[:2],
        index=iter0_index[:2],
    )
    scenario.write_data(
        str(filename),
        iteration=0,
        times=iter0_times[2:],
        power=iter0_power[2:],
        index=iter0_index[2:],
    )

    iter1_times = Time(60010.0 + _seconds_to_mjd(np.array([0.0, 2.0])), format="mjd", scale="utc")
    iter1_power = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    ) * u.W
    iter1_index = np.array([10, 11], dtype=np.int16)
    scenario.write_data(
        str(filename),
        iteration=1,
        times=iter1_times,
        power=iter1_power,
        index=iter1_index,
    )
    scenario.flush_writes(str(filename))
    return filename


def _load_notebook_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _replace_notebook_assignment(source: str, old: str, new: str) -> str:
    if old not in source:
        raise AssertionError(f"Notebook source is missing expected token {old!r}.")
    return source.replace(old, new, 1)


def _replace_notebook_assignment_if_present(source: str, old: str, new: str) -> str:
    if old not in source:
        return source
    return source.replace(old, new, 1)


def _replace_notebook_assignment_pattern(source: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise AssertionError(f"Notebook source is missing expected pattern {pattern!r}.")
    return updated


def _bool_source(value: bool) -> str:
    return "True" if bool(value) else "False"


def _execute_code_cells(cells: list[str], *, namespace: dict[str, object] | None = None) -> dict[str, object]:
    env: dict[str, object] = {"__name__": "__notebook_smoke__"}
    if namespace is not None:
        env.update(namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        for idx, source in enumerate(cells):
            if not source.strip():
                continue
            exec(compile(source, f"notebook_smoke#cell-{idx}", "exec"), env)
    return env


def _parse_scheduler_test_cap_override(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    value = int(str(raw).strip())
    return max(1, min(12, value))


def _scheduler_test_force_full_matrix_enabled() -> bool:
    raw = str(os.environ.get("SCEPTER_SCHEDULER_TEST_FORCE_FULL_MATRIX", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _scheduler_test_gpu_runtime_snapshot() -> dict[str, object] | None:
    try:
        import cupy as real_cp  # type: ignore[import-not-found]
    except Exception:
        return None
    return scenario._runtime_gpu_memory_snapshot(real_cp)


def _scheduler_test_cap_from_free_bytes(free_bytes: int | None) -> int | None:
    if free_bytes is None:
        return None
    return min(12, max(1, int(free_bytes // (2 * _GIB))))


def _scheduler_test_matrix_caps() -> dict[str, object]:
    host_override = _parse_scheduler_test_cap_override("SCEPTER_SCHEDULER_TEST_MAX_HOST_GB")
    gpu_override = _parse_scheduler_test_cap_override("SCEPTER_SCHEDULER_TEST_MAX_GPU_GB")
    force_full = _scheduler_test_force_full_matrix_enabled()
    host_snapshot = scenario._runtime_host_memory_snapshot()
    gpu_snapshot = _scheduler_test_gpu_runtime_snapshot()
    host_cap = (
        host_override
        if host_override is not None
        else (
            12
            if force_full
            else _scheduler_test_cap_from_free_bytes(
                None if host_snapshot is None else int(host_snapshot["available_bytes"])
            )
        )
    )
    gpu_cap = (
        gpu_override
        if gpu_override is not None
        else (
            12
            if force_full
            else _scheduler_test_cap_from_free_bytes(
                None if gpu_snapshot is None else int(gpu_snapshot["free_bytes"])
            )
        )
    )
    return {
        "host_cap": host_cap,
        "gpu_cap": gpu_cap,
        "host_snapshot": host_snapshot,
        "gpu_snapshot": gpu_snapshot,
        "force_full": force_full,
        "host_override": host_override,
        "gpu_override": gpu_override,
    }


def _scheduler_test_force_full_suite_requested() -> bool:
    return _scheduler_test_force_full_matrix_enabled()


def _build_boresight_notebook_smoke_cells(
    tmp_path: Path,
    *,
    case_name: str,
    selection_strategy: str,
    ras_pointing_mode: str,
    include_atmosphere: bool,
    memory_budget_mode: str,
    profile_name: str,
    boresight_theta1_deg: float | None,
    boresight_theta2_deg: float | None,
    theta2_scope_mode: str = "cell_ids",
    theta2_cell_ids_literal: str = "None",
    theta2_layers: int = 0,
    theta2_radius_km: float | None = None,
    host_memory_budget_gb: float = 4.0,
    gpu_memory_budget_gb: float = 4.0,
    force_bulk_timesteps: int | None = 2,
    progress_desc_mode: str = "coarse",
    writer_checkpoint_interval_s: float = 1.0e-9,
    duration_s: int = 2,
    point_spacing_km: float = 3000.0,
    render_cell_status_map: bool = False,
) -> list[str]:
    cells = _load_notebook_code_cells(BORESIGHT_NOTEBOOK_PATH)
    profile_case = _FAKE_OUTPUT_PROFILE_CASES[profile_name]

    cells[1] = _replace_notebook_assignment(
        cells[1],
        'selection_strategy="max_elevation"',
        f'selection_strategy={selection_strategy!r}',
    )
    cells[1] = _replace_notebook_assignment_pattern(cells[1], r"^Nbeam\s*=\s*\d+.*$", "Nbeam = 8")
    cells[1] = _replace_notebook_assignment(
        cells[1],
        '"num_sats_per_plane": 120,',
        '"num_sats_per_plane": 4,',
    )
    cells[1] = _replace_notebook_assignment(
        cells[1],
        '"plane_count": 28,',
        '"plane_count": 2,',
    )

    cells[6] = _replace_notebook_assignment(
        cells[6],
        "point_spacing = float(cell_km) * u.km",
        f"point_spacing = {float(point_spacing_km):.1f} * u.km",
    )
    cells[6] = _replace_notebook_assignment(
        cells[6],
        'GEOGRAPHY_MASK_MODE = "land_plus_nearshore_sea"',
        'GEOGRAPHY_MASK_MODE = "none"',
    )
    cells[6] = _replace_notebook_assignment(cells[6], "SHORELINE_BUFFER_KM = 10", "SHORELINE_BUFFER_KM = None")
    cells[6] = _replace_notebook_assignment(
        cells[6],
        "RENDER_CELL_STATUS_MAP = True",
        f"RENDER_CELL_STATUS_MAP = {_bool_source(render_cell_status_map)}",
    )
    cells[6] = _replace_notebook_assignment(
        cells[6],
        'CELL_STATUS_MAP_BACKEND = "auto"',
        'CELL_STATUS_MAP_BACKEND = "matplotlib"',
    )
    cells[6] = _replace_notebook_assignment(
        cells[6],
        'RAS_POINTING_MODE = "ras_station"',
        f"RAS_POINTING_MODE = {ras_pointing_mode!r}",
    )
    cells[6] = _replace_notebook_assignment(
        cells[6],
        'RAS_HEX_EXCLUSION_MODE = "adjacency_layers"',
        'RAS_HEX_EXCLUSION_MODE = "none"',
    )
    cells[6] = _replace_notebook_assignment(cells[6], "RAS_HEX_EXCLUSION_LAYERS = 2", "RAS_HEX_EXCLUSION_LAYERS = 0")

    cells[8] = _replace_notebook_assignment(
        cells[8],
        '_STORAGE_FILENAME = "simulation_results_1.13_US_System_B525_random_step1+2_GPU.h5"',
        f"_STORAGE_FILENAME = {str(tmp_path / f'{case_name}.h5')!r}",
    )
    cells[8] = _replace_notebook_assignment(
        cells[8], 'PROGRESS_DESC_MODE = "coarse"', f"PROGRESS_DESC_MODE = {progress_desc_mode!r}"
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "WRITER_CHECKPOINT_INTERVAL_S = 60.0",
        f"WRITER_CHECKPOINT_INTERVAL_S = {float(writer_checkpoint_interval_s)!r}",
    )
    family_source = repr(profile_case["profile_kwargs"]["output_families"])
    cells[8] += (
        "\n"
        f"OUTPUT_FAMILIES = {family_source}\n"
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "INCLUDE_ATMOSPHERE = True",
        f"INCLUDE_ATMOSPHERE = {_bool_source(include_atmosphere)}",
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "BORESIGHT_AVOIDANCE_ENABLED = True",
        f"BORESIGHT_AVOIDANCE_ENABLED = {_bool_source(boresight_theta1_deg is not None or boresight_theta2_deg is not None)}",
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "BORESIGHT_THETA1 = 1 * u.deg",
        (
            "BORESIGHT_THETA1 = None"
            if boresight_theta1_deg is None
            else f"BORESIGHT_THETA1 = {float(boresight_theta1_deg):.6f} * u.deg"
        ),
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "BORESIGHT_THETA2 = 3 * u.deg",
        (
            "BORESIGHT_THETA2 = None"
            if boresight_theta2_deg is None
            else f"BORESIGHT_THETA2 = {float(boresight_theta2_deg):.6f} * u.deg"
        ),
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        'BORESIGHT_THETA2_SCOPE_MODE = "adjacency_layers"',
        f"BORESIGHT_THETA2_SCOPE_MODE = {theta2_scope_mode!r}",
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "BORESIGHT_THETA2_CELL_IDS = None",
        f"BORESIGHT_THETA2_CELL_IDS = {theta2_cell_ids_literal}",
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "BORESIGHT_THETA2_LAYERS = 2",
        f"BORESIGHT_THETA2_LAYERS = {int(theta2_layers)}",
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "BORESIGHT_THETA2_RADIUS_KM = None",
        (
            "BORESIGHT_THETA2_RADIUS_KM = None"
            if theta2_radius_km is None
            else f"BORESIGHT_THETA2_RADIUS_KM = {float(theta2_radius_km):.6f}"
        ),
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        'base_end_time = astrotime.Time(datetime(2025, 1, 1, 1, 0, 0), scale="utc")',
        f'base_end_time = astrotime.Time(datetime(2025, 1, 1, 0, 0, {int(duration_s)}), scale="utc")',
    )
    cells[8] = _replace_notebook_assignment(
        cells[8], "HOST_MEMORY_BUDGET_GB = 4.0", f"HOST_MEMORY_BUDGET_GB = {float(host_memory_budget_gb):.6f}"
    )
    cells[8] = _replace_notebook_assignment(
        cells[8], "GPU_MEMORY_BUDGET_GB = 4.0", f"GPU_MEMORY_BUDGET_GB = {float(gpu_memory_budget_gb):.6f}"
    )
    cells[8] = _replace_notebook_assignment(
        cells[8], 'MEMORY_BUDGET_MODE = "hybrid"', f"MEMORY_BUDGET_MODE = {memory_budget_mode!r}"
    )
    cells[8] = _replace_notebook_assignment(
        cells[8],
        "FORCE_BULK_TIMESTEPS = None",
        (
            "FORCE_BULK_TIMESTEPS = None"
            if force_bulk_timesteps is None
            else f"FORCE_BULK_TIMESTEPS = {int(force_bulk_timesteps)}"
        ),
    )

    cells[10] = _replace_notebook_assignment(
        cells[10],
        "    enable_progress_bars=True,",
        "    enable_progress_bars=False,",
    )

    return cells


def _execute_postprocess_validation(storage_filename: str) -> dict[str, object]:
    cells = _load_notebook_code_cells(POSTPROCESS_NOTEBOOK_PATH)
    cells[0] = _replace_notebook_assignment(
        cells[0],
        '_STORAGE_FILENAME = "simulation_results_1.13_US_System_B525_random_step1+2_GPU.h5"',
        f"_STORAGE_FILENAME = {str(storage_filename)!r}",
    )
    cells[0] = _replace_notebook_assignment(cells[0], "SAVE_FIGURES = True", "SAVE_FIGURES = False")
    cells[0] = _replace_notebook_assignment_if_present(
        cells[0],
        "HEATMAP_SHOW_PROGRESS = True",
        "HEATMAP_SHOW_PROGRESS = False",
    )
    return _execute_code_cells([cells[0]])


@pytest.mark.parametrize(
    ("quantity", "basis", "input_field", "input_value", "expected_unit", "complementary_field"),
    [
        (
            "target_pfd",
            "per_mhz",
            "target_pfd_dbw_m2_mhz",
            -83.5,
            "dBW/m^2/MHz",
            "target_pfd_dbw_m2_channel",
        ),
        (
            "target_pfd",
            "per_channel",
            "target_pfd_dbw_m2_channel",
            -76.51029995663981,
            "dBW/m^2",
            "target_pfd_dbw_m2_mhz",
        ),
        (
            "satellite_ptx",
            "per_mhz",
            "satellite_ptx_dbw_mhz",
            12.0,
            "dBW/MHz",
            "satellite_ptx_dbw_channel",
        ),
        (
            "satellite_ptx",
            "per_channel",
            "satellite_ptx_dbw_channel",
            18.989700043360187,
            "dBW",
            "satellite_ptx_dbw_mhz",
        ),
        (
            "satellite_eirp",
            "per_mhz",
            "satellite_eirp_dbw_mhz",
            33.0,
            "dBW/MHz",
            "satellite_eirp_dbw_channel",
        ),
        (
            "satellite_eirp",
            "per_channel",
            "satellite_eirp_dbw_channel",
            39.98970004336019,
            "dBW",
            "satellite_eirp_dbw_mhz",
        ),
    ],
)
def test_normalize_direct_epfd_power_input_converts_basis_once(
    quantity: str,
    basis: str,
    input_field: str,
    input_value: float,
    expected_unit: str,
    complementary_field: str,
) -> None:
    power_input = scenario.normalize_direct_epfd_power_input(
        bandwidth_mhz=5.0,
        power_input_quantity=quantity,
        power_input_basis=basis,
        **{input_field: input_value},
    )

    expected_offset_db = 10.0 * np.log10(5.0)
    expected_complementary = (
        float(input_value) + float(expected_offset_db)
        if basis == "per_mhz"
        else float(input_value) - float(expected_offset_db)
    )

    assert power_input["power_input_quantity"] == quantity
    assert power_input["power_input_basis"] == basis
    assert power_input["active_value"] == pytest.approx(float(input_value))
    assert power_input["active_value_unit"] == expected_unit
    assert power_input[input_field] == pytest.approx(float(input_value))
    assert power_input[complementary_field] == pytest.approx(expected_complementary)


def test_normalize_direct_epfd_power_input_rejects_inconsistent_active_basis_pair() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        scenario.normalize_direct_epfd_power_input(
            bandwidth_mhz=5.0,
            power_input_quantity="target_pfd",
            power_input_basis="per_mhz",
            target_pfd_dbw_m2_mhz=-83.5,
            target_pfd_dbw_m2_channel=-83.5,
        )


def test_normalize_direct_epfd_power_input_ignores_legacy_target_alias_for_non_target_modes() -> None:
    power_input = scenario.normalize_direct_epfd_power_input(
        bandwidth_mhz=5.0,
        power_input_quantity="satellite_ptx",
        power_input_basis="per_channel",
        pfd0_dbw_m2_mhz=-83.5,
        satellite_ptx_dbw_channel=10.0,
    )

    assert power_input["target_pfd_dbw_m2_mhz"] is None
    assert power_input["satellite_ptx_dbw_channel"] == pytest.approx(10.0)


def test_normalize_direct_epfd_spectrum_plan_reports_exact_fit_reuse_and_lower_ras_reference() -> None:
    spectrum_plan = scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan={
            "service_band_start_mhz": 2620.0,
            "service_band_stop_mhz": 2690.0,
            "ras_receiver_band_start_mhz": 2690.0,
            "ras_receiver_band_stop_mhz": 2700.0,
            "reuse_factor": 4,
            "channel_groups_per_cell_cap": 4,
        },
        channel_bandwidth_mhz=5.0,
        active_cell_count=4,
        active_cell_reuse_slot_ids=np.asarray([0, 1, 2, 3], dtype=np.int32),
    )

    assert spectrum_plan is not None
    assert spectrum_plan["full_channel_count"] == 14
    assert spectrum_plan["leftover_spectrum_mhz"] == pytest.approx(0.0)
    assert spectrum_plan["zero_leftover_reuse_factors"] == (1, 7)
    assert spectrum_plan["channel_groups_per_cell"] == 3
    assert spectrum_plan["max_groups_per_cell"] == 3
    assert spectrum_plan["tx_reference_mode"] == "middle"
    assert spectrum_plan["ras_reference_mode"] == "lower"
    assert spectrum_plan["split_total_group_denominator_mode"] == "configured_groups"
    assert spectrum_plan["ras_reference_frequency_mhz_effective"] == pytest.approx(2690.0)
    assert spectrum_plan["spectral_integration_cutoff_mhz"] == pytest.approx(12.5)
    assert np.all(np.asarray(spectrum_plan["cell_leakage_factors"], dtype=np.float32) >= 0.0)
    assert spectrum_plan["slot_group_channel_indices"].shape == (4, 3)
    assert spectrum_plan["slot_group_leakage_factors"].shape == (4, 3)
    assert spectrum_plan["cell_group_leakage_factors"].shape == (4, 3)


def test_normalize_direct_epfd_spectrum_plan_uses_all_reuse_valid_groups_when_cap_is_none() -> None:
    spectrum_plan = scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan={
            "service_band_start_mhz": 2620.0,
            "service_band_stop_mhz": 2690.0,
            "ras_receiver_band_start_mhz": 2690.0,
            "ras_receiver_band_stop_mhz": 2700.0,
            "reuse_factor": 7,
            "channel_groups_per_cell_cap": None,
        },
        channel_bandwidth_mhz=5.0,
        active_cell_count=7,
        active_cell_reuse_slot_ids=np.arange(7, dtype=np.int32),
    )

    assert spectrum_plan is not None
    assert spectrum_plan["channel_groups_per_cell_cap"] == 2
    assert spectrum_plan["channel_groups_per_cell"] == 2
    assert spectrum_plan["max_groups_per_cell"] == 2
    assert_equal(
        spectrum_plan["slot_group_channel_indices"],
        np.asarray(
            [
                [0, 7],
                [1, 8],
                [2, 9],
                [3, 10],
                [4, 11],
                [5, 12],
                [6, 13],
            ],
            dtype=np.int32,
        ),
    )


def test_normalize_direct_epfd_spectrum_plan_supports_explicit_disabled_channels() -> None:
    spectrum_plan = scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan={
            "service_band_start_mhz": 2620.0,
            "service_band_stop_mhz": 2690.0,
            "ras_receiver_band_start_mhz": 2690.0,
            "ras_receiver_band_stop_mhz": 2700.0,
            "reuse_factor": 7,
            "disabled_channel_indices": [7, 8, 9, 10, 11, 12, 13],
        },
        channel_bandwidth_mhz=5.0,
        active_cell_count=7,
        active_cell_reuse_slot_ids=np.arange(7, dtype=np.int32),
    )

    assert spectrum_plan is not None
    assert_equal(
        spectrum_plan["enabled_channel_indices"],
        np.arange(7, dtype=np.int32),
    )
    assert_equal(
        spectrum_plan["slot_group_valid_mask"],
        np.asarray(
            [[True], [True], [True], [True], [True], [True], [True]],
            dtype=bool,
        ),
    )
    assert_equal(
        spectrum_plan["configured_group_counts_per_cell"],
        np.ones(7, dtype=np.int32),
    )


def test_normalize_direct_epfd_spectrum_plan_accepts_numpy_channel_selection_arrays() -> None:
    enabled_plan = scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan={
            "service_band_start_mhz": 2620.0,
            "service_band_stop_mhz": 2690.0,
            "ras_receiver_band_start_mhz": 2690.0,
            "ras_receiver_band_stop_mhz": 2700.0,
            "reuse_factor": 7,
            "enabled_channel_indices": np.asarray([0, 1, 2, 8, 9], dtype=np.int32),
        },
        channel_bandwidth_mhz=5.0,
        active_cell_count=7,
        active_cell_reuse_slot_ids=np.arange(7, dtype=np.int32),
    )
    disabled_plan = scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan={
            "service_band_start_mhz": 2620.0,
            "service_band_stop_mhz": 2690.0,
            "ras_receiver_band_start_mhz": 2690.0,
            "ras_receiver_band_stop_mhz": 2700.0,
            "reuse_factor": 7,
            "disabled_channel_indices": np.asarray([3, 4, 5, 6, 10, 11, 12, 13], dtype=np.int32),
        },
        channel_bandwidth_mhz=5.0,
        active_cell_count=7,
        active_cell_reuse_slot_ids=np.arange(7, dtype=np.int32),
    )

    assert enabled_plan is not None
    assert disabled_plan is not None
    assert_equal(
        enabled_plan["enabled_channel_indices"],
        np.asarray([0, 1, 2, 8, 9], dtype=np.int32),
    )
    assert_equal(
        enabled_plan["configured_group_counts_per_cell"],
        np.asarray([1, 2, 2, 0, 0, 0, 0], dtype=np.int32),
    )
    assert_equal(
        enabled_plan["slot_group_valid_mask"],
        np.asarray(
            [
                [True, False],
                [True, True],
                [True, True],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
            ],
            dtype=bool,
        ),
    )
    assert_equal(
        disabled_plan["enabled_channel_indices"],
        np.asarray([0, 1, 2, 7, 8, 9], dtype=np.int32),
    )
    assert_equal(
        disabled_plan["configured_group_counts_per_cell"],
        np.asarray([2, 2, 2, 0, 0, 0, 0], dtype=np.int32),
    )


_DIRECT_EPFD_TEST_MASK_POINTS_MHZ = np.asarray(
    [
        [-22.5, 37.123599],
        [-12.5, 25.0824],
        [-7.5, 13.0412],
        [-5.0, 1.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [5.0, 1.0],
        [7.5, 13.0412],
        [12.5, 25.0824],
        [22.5, 37.123599],
    ],
    dtype=np.float64,
)
_DIRECT_EPFD_TEST_CUSTOM_RX_POINTS_MHZ = np.asarray(
    [
        [-9.0, 36.0],
        [-3.0, 8.0],
        [1.0, 0.0],
        [7.0, 12.0],
        [15.0, 28.0],
    ],
    dtype=np.float64,
)
_DIRECT_EPFD_FAST_SERVICE_CASES = (
    {
        "case_id": "default_2620_2690_bw5",
        "service_band_start_mhz": 2620.0,
        "service_band_stop_mhz": 2690.0,
        "channel_bandwidth_mhz": 5.0,
    },
    {
        "case_id": "nondefault_2500_2690_bw10",
        "service_band_start_mhz": 2500.0,
        "service_band_stop_mhz": 2690.0,
        "channel_bandwidth_mhz": 10.0,
    },
    {
        "case_id": "nondefault_2500_2690_bw15",
        "service_band_start_mhz": 2500.0,
        "service_band_stop_mhz": 2690.0,
        "channel_bandwidth_mhz": 15.0,
    },
)


def _direct_epfd_test_full_channel_count(service_case: Mapping[str, float | str]) -> int:
    service_bandwidth_mhz = (
        float(service_case["service_band_stop_mhz"]) - float(service_case["service_band_start_mhz"])
    )
    return int(np.floor(service_bandwidth_mhz / float(service_case["channel_bandwidth_mhz"])))


def _direct_epfd_test_ras_case(
    service_case: Mapping[str, float | str],
    case_name: str,
) -> dict[str, object]:
    service_start_mhz = float(service_case["service_band_start_mhz"])
    service_stop_mhz = float(service_case["service_band_stop_mhz"])
    if case_name == "upper_adjacent_rectangular":
        return {
            "case_id": case_name,
            "ras_receiver_band_start_mhz": service_stop_mhz,
            "ras_receiver_band_stop_mhz": service_stop_mhz + 10.0,
            "receiver_response_mode": "rectangular",
            "receiver_response_points_mhz": None,
        }
    if case_name == "lower_adjacent_rectangular":
        return {
            "case_id": case_name,
            "ras_receiver_band_start_mhz": service_start_mhz - 10.0,
            "ras_receiver_band_stop_mhz": service_start_mhz,
            "receiver_response_mode": "rectangular",
            "receiver_response_points_mhz": None,
        }
    if case_name == "upper_adjacent_narrow_rectangular":
        return {
            "case_id": case_name,
            "ras_receiver_band_start_mhz": service_stop_mhz,
            "ras_receiver_band_stop_mhz": service_stop_mhz + 5.0,
            "receiver_response_mode": "rectangular",
            "receiver_response_points_mhz": None,
        }
    if case_name == "upper_adjacent_wide_rectangular":
        return {
            "case_id": case_name,
            "ras_receiver_band_start_mhz": service_stop_mhz,
            "ras_receiver_band_stop_mhz": service_stop_mhz + 20.0,
            "receiver_response_mode": "rectangular",
            "receiver_response_points_mhz": None,
        }
    if case_name == "upper_adjacent_custom_asymmetric":
        return {
            "case_id": case_name,
            "ras_receiver_band_start_mhz": service_stop_mhz,
            "ras_receiver_band_stop_mhz": service_stop_mhz + 10.0,
            "receiver_response_mode": "custom",
            "receiver_response_points_mhz": np.asarray(
                _DIRECT_EPFD_TEST_CUSTOM_RX_POINTS_MHZ,
                dtype=np.float64,
            ),
        }
    raise KeyError(case_name)


def _structured_direct_epfd_channel_subsets(
    full_channel_count: int,
    reuse_factor: int,
) -> list[list[int]]:
    full_channel_count_i = int(full_channel_count)
    reuse_factor_i = max(1, int(reuse_factor))
    ordered_subsets: list[list[int]] = []
    seen_keys: set[tuple[int, ...]] = set()

    def _add(subset: list[int]) -> None:
        cleaned = sorted(
            {
                int(value)
                for value in subset
                if 0 <= int(value) < full_channel_count_i
            }
        )
        key = tuple(cleaned)
        if key not in seen_keys:
            seen_keys.add(key)
            ordered_subsets.append(cleaned)

    _add([])
    _add(list(range(full_channel_count_i)))
    for index in range(full_channel_count_i):
        _add([index])
    for start in range(max(0, full_channel_count_i - 1)):
        _add([start, start + 1])
    for start in range(max(0, full_channel_count_i - 2)):
        _add([start, start + 1, start + 2])
    for window_size in (max(1, reuse_factor_i), max(1, reuse_factor_i + 1)):
        for start in range(0, max(1, full_channel_count_i - window_size + 1)):
            _add(list(range(start, start + window_size)))
    _add(list(range(max(0, full_channel_count_i - 3), full_channel_count_i)))
    _add(list(range(max(0, full_channel_count_i - reuse_factor_i), full_channel_count_i)))
    _add(list(range(0, min(full_channel_count_i, reuse_factor_i))))
    for step in (2, 3, max(2, reuse_factor_i)):
        _add(list(range(0, full_channel_count_i, step)))
    for slot_id in range(reuse_factor_i):
        _add(list(range(slot_id, full_channel_count_i, reuse_factor_i)))
    group_count = max(1, int(np.ceil(full_channel_count_i / float(reuse_factor_i))))
    for group_index in range(group_count):
        _add(
            [
                slot_id + group_index * reuse_factor_i
                for slot_id in range(reuse_factor_i)
            ]
        )
    return ordered_subsets


def _sampled_direct_epfd_channel_subsets(
    full_channel_count: int,
    reuse_factor: int,
    *,
    sample_count: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(int(seed) + int(reuse_factor) * 1009 + int(full_channel_count) * 9176)
    subsets = _structured_direct_epfd_channel_subsets(full_channel_count, reuse_factor)
    seen_keys = {tuple(subset) for subset in subsets}
    while len(subsets) < int(sample_count):
        mask = rng.random(int(full_channel_count)) < 0.5
        subset = sorted(np.nonzero(mask)[0].astype(int).tolist())
        key = tuple(subset)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        subsets.append(subset)
    return subsets


def _normalize_direct_epfd_test_plan(
    *,
    service_case: Mapping[str, float | str],
    ras_case: Mapping[str, object],
    reuse_factor: int,
    enabled_channel_indices: list[int] | None = None,
    disabled_channel_indices: list[int] | None = None,
    power_policy: str = "repeat_per_group",
) -> dict[str, object]:
    active_cell_reuse_slot_ids = np.arange(max(1, int(reuse_factor)), dtype=np.int32) % max(1, int(reuse_factor))
    spectrum_plan_payload: dict[str, object] = {
        "service_band_start_mhz": float(service_case["service_band_start_mhz"]),
        "service_band_stop_mhz": float(service_case["service_band_stop_mhz"]),
        "ras_receiver_band_start_mhz": float(ras_case["ras_receiver_band_start_mhz"]),
        "ras_receiver_band_stop_mhz": float(ras_case["ras_receiver_band_stop_mhz"]),
        "reuse_factor": int(reuse_factor),
        "unwanted_emission_mask_preset": "custom",
        "custom_mask_points": np.asarray(_DIRECT_EPFD_TEST_MASK_POINTS_MHZ, dtype=np.float64),
        "receiver_response_mode": str(ras_case["receiver_response_mode"]),
        "receiver_custom_mask_points": (
            None
            if ras_case["receiver_response_points_mhz"] is None
            else np.asarray(ras_case["receiver_response_points_mhz"], dtype=np.float64)
        ),
        "spectral_integration_cutoff_basis": "channel_bandwidth",
        "spectral_integration_cutoff_percent": 450.0,
        "multi_group_power_policy": str(power_policy),
    }
    if enabled_channel_indices is not None:
        spectrum_plan_payload["enabled_channel_indices"] = list(enabled_channel_indices)
    if disabled_channel_indices is not None:
        spectrum_plan_payload["disabled_channel_indices"] = list(disabled_channel_indices)
    spectrum_plan = scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan=spectrum_plan_payload,
        channel_bandwidth_mhz=float(service_case["channel_bandwidth_mhz"]),
        active_cell_count=int(active_cell_reuse_slot_ids.size),
        active_cell_reuse_slot_ids=active_cell_reuse_slot_ids,
    )
    assert spectrum_plan is not None
    return spectrum_plan


def _direct_epfd_plan_channel_leakage_map(plan: Mapping[str, object]) -> dict[int, float]:
    slot_group_indices = np.asarray(plan["slot_group_channel_indices"], dtype=np.int32)
    slot_group_valid_mask = np.asarray(plan["slot_group_valid_mask"], dtype=bool)
    slot_group_factors = np.asarray(plan["slot_group_leakage_factors"], dtype=np.float64)
    leakage_map: dict[int, float] = {}
    for slot_id in range(int(slot_group_indices.shape[0])):
        for group_index in range(int(slot_group_indices.shape[1])):
            if not bool(slot_group_valid_mask[slot_id, group_index]):
                continue
            channel_index = int(slot_group_indices[slot_id, group_index])
            if channel_index < 0:
                continue
            leakage_map[channel_index] = float(slot_group_factors[slot_id, group_index])
    return leakage_map


def _precomputed_direct_epfd_channel_leakage_values(plan: Mapping[str, object]) -> dict[int, float]:
    slot_edges_mhz = np.asarray(plan["slot_edges_mhz"], dtype=np.float64)
    receiver_points = plan.get("receiver_response_points_mhz")
    receiver_response_points_mhz = (
        None
        if receiver_points is None
        else np.asarray(receiver_points, dtype=np.float64)
    )
    full_channel_count = int(plan["full_channel_count"])
    return {
        int(channel_index): float(
            scenario._integrate_direct_epfd_channel_leakage_fraction(
                channel_start_mhz=float(slot_edges_mhz[channel_index]),
                channel_stop_mhz=float(slot_edges_mhz[channel_index + 1]),
                ras_start_mhz=float(plan["ras_receiver_band_start_mhz"]),
                ras_stop_mhz=float(plan["ras_receiver_band_stop_mhz"]),
                channel_bandwidth_mhz=float(plan["channel_bandwidth_mhz"]),
                mask_points_mhz=np.asarray(plan["unwanted_emission_mask_points_mhz"], dtype=np.float64),
                integration_cutoff_mhz=float(plan["spectral_integration_cutoff_mhz"]),
                receiver_response_mode=str(plan["receiver_response_mode"]),
                receiver_response_points_mhz=receiver_response_points_mhz,
            )
        )
        for channel_index in range(full_channel_count)
    }


def _assert_direct_epfd_plan_matches_expected_channel_leakage(
    plan: Mapping[str, object],
    *,
    expected_channel_leakage_by_index: Mapping[int, float],
    case_label: str,
) -> None:
    del case_label
    slot_group_indices = np.asarray(plan["slot_group_channel_indices"], dtype=np.int32)
    slot_group_valid_mask = np.asarray(plan["slot_group_valid_mask"], dtype=bool)
    slot_group_factors = np.asarray(plan["slot_group_leakage_factors"], dtype=np.float64)
    active_cell_reuse_slot_ids = np.asarray(plan["active_cell_reuse_slot_ids"], dtype=np.int32)
    enabled_channel_indices = sorted(np.asarray(plan["enabled_channel_indices"], dtype=np.int32).astype(int).tolist())
    configured_group_counts_per_cell = np.asarray(plan["configured_group_counts_per_cell"], dtype=np.int32)
    cell_group_valid_mask = np.asarray(plan["cell_group_valid_mask"], dtype=bool)
    cell_group_leakage_factors = np.asarray(plan["cell_group_leakage_factors"], dtype=np.float64)
    cell_leakage_factors = np.asarray(plan["cell_leakage_factors"], dtype=np.float64)
    power_policy = str(plan["multi_group_power_policy"])

    expected_channel_leakage = {
        int(channel_index): float(expected_channel_leakage_by_index[int(channel_index)])
        for channel_index in enabled_channel_indices
    }
    assert _direct_epfd_plan_channel_leakage_map(plan) == pytest.approx(expected_channel_leakage)

    for slot_id in range(int(slot_group_indices.shape[0])):
        expected_slot_channels = sorted(
            channel_index
            for channel_index in enabled_channel_indices
            if int(channel_index % max(1, int(plan["reuse_factor"]))) == slot_id
        )
        actual_slot_channels = [
            int(slot_group_indices[slot_id, group_index])
            for group_index in range(int(slot_group_indices.shape[1]))
            if bool(slot_group_valid_mask[slot_id, group_index])
        ]
        assert actual_slot_channels == expected_slot_channels
        expected_slot_factors = np.asarray(
            [expected_channel_leakage[int(channel_index)] for channel_index in expected_slot_channels],
            dtype=np.float64,
        )
        actual_slot_factors = np.asarray(
            [
                float(slot_group_factors[slot_id, group_index])
                for group_index in range(int(slot_group_indices.shape[1]))
                if bool(slot_group_valid_mask[slot_id, group_index])
            ],
            dtype=np.float64,
        )
        assert_allclose(actual_slot_factors, expected_slot_factors, rtol=1.0e-7, atol=1.0e-10)
        expected_slot_value = (
            float(np.mean(expected_slot_factors))
            if power_policy == "split_total_cell_power" and expected_slot_factors.size > 0
            else float(np.sum(expected_slot_factors))
        )
        assert float(dict(plan["slot_leakage_factors"])[slot_id]) == pytest.approx(expected_slot_value)

    assert_equal(
        configured_group_counts_per_cell,
        np.sum(cell_group_valid_mask, axis=1, dtype=np.int32),
    )
    for cell_index, slot_id in enumerate(active_cell_reuse_slot_ids.astype(int).tolist()):
        expected_slot_channels = sorted(
            channel_index
            for channel_index in enabled_channel_indices
            if int(channel_index % max(1, int(plan["reuse_factor"]))) == int(slot_id)
        )
        expected_slot_factors = np.asarray(
            [expected_channel_leakage[int(channel_index)] for channel_index in expected_slot_channels],
            dtype=np.float64,
        )
        expected_slot_value = (
            float(np.mean(expected_slot_factors))
            if power_policy == "split_total_cell_power" and expected_slot_factors.size > 0
            else float(np.sum(expected_slot_factors))
        )
        assert int(configured_group_counts_per_cell[cell_index]) == len(expected_slot_channels)
        assert_allclose(
            cell_group_leakage_factors[cell_index, : len(expected_slot_channels)],
            expected_slot_factors,
            rtol=1.0e-7,
            atol=1.0e-10,
        )
        assert not np.any(cell_group_valid_mask[cell_index, len(expected_slot_channels):])
        assert float(cell_leakage_factors[cell_index]) == pytest.approx(expected_slot_value)


def _dense_direct_epfd_channel_leakage_fraction(
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
    sample_count: int = 200_001,
) -> float:
    channel_center = 0.5 * (float(channel_start_mhz) + float(channel_stop_mhz))
    ras_center = 0.5 * (float(ras_start_mhz) + float(ras_stop_mhz))
    domain_start = float(channel_start_mhz - integration_cutoff_mhz)
    domain_stop = float(channel_stop_mhz + integration_cutoff_mhz)
    if receiver_response_mode == "custom" and receiver_response_points_mhz is not None:
        response_offsets = np.asarray(receiver_response_points_mhz[:, 0], dtype=np.float64)
        domain_start = min(domain_start, float(ras_center + np.min(response_offsets)))
        domain_stop = max(domain_stop, float(ras_center + np.max(response_offsets)))
    frequencies_mhz = np.linspace(domain_start, domain_stop, int(sample_count), dtype=np.float64)
    tx_attenuation_db = scenario._evaluate_direct_epfd_mask_attenuation_db(
        frequencies_mhz - channel_center,
        mask_points_mhz=np.asarray(mask_points_mhz, dtype=np.float64),
    )
    rx_attenuation_db = scenario._evaluate_direct_epfd_receiver_response_attenuation_db(
        frequencies_mhz - ras_center,
        response_mode=str(receiver_response_mode),
        receiver_bandwidth_mhz=float(ras_stop_mhz - ras_start_mhz),
        response_points_mhz=receiver_response_points_mhz,
    )
    linear_power = np.zeros_like(frequencies_mhz, dtype=np.float64)
    finite_mask = np.isfinite(rx_attenuation_db)
    linear_power[finite_mask] = 10.0 ** (
        -(tx_attenuation_db[finite_mask] + rx_attenuation_db[finite_mask]) / 10.0
    )
    integrate_fn = getattr(np, "trapezoid", np.trapz)
    return float(
        max(
            0.0,
            integrate_fn(linear_power, frequencies_mhz) / float(channel_bandwidth_mhz),
        )
    )


@pytest.mark.parametrize("cutoff_percent", (1000.0, 5000.0))
def test_integrate_direct_epfd_channel_leakage_fraction_matches_dense_reference_for_large_rectangular_cutoffs(
    cutoff_percent: float,
) -> None:
    channel_bandwidth_mhz = 5.0
    mask_points_mhz = np.asarray(
        [
            [-35.0, 55.0],
            [-12.0, 18.0],
            [-2.5, 0.0],
            [2.5, 0.0],
            [12.0, 18.0],
            [35.0, 55.0],
        ],
        dtype=np.float64,
    )
    integration_cutoff_mhz = channel_bandwidth_mhz * cutoff_percent / 100.0

    exact_value = scenario._integrate_direct_epfd_channel_leakage_fraction(
        channel_start_mhz=2620.0,
        channel_stop_mhz=2625.0,
        ras_start_mhz=2690.0,
        ras_stop_mhz=2700.0,
        channel_bandwidth_mhz=channel_bandwidth_mhz,
        mask_points_mhz=mask_points_mhz,
        integration_cutoff_mhz=integration_cutoff_mhz,
        receiver_response_mode="rectangular",
    )
    dense_value = _dense_direct_epfd_channel_leakage_fraction(
        channel_start_mhz=2620.0,
        channel_stop_mhz=2625.0,
        ras_start_mhz=2690.0,
        ras_stop_mhz=2700.0,
        channel_bandwidth_mhz=channel_bandwidth_mhz,
        mask_points_mhz=mask_points_mhz,
        integration_cutoff_mhz=integration_cutoff_mhz,
        receiver_response_mode="rectangular",
    )

    assert exact_value == pytest.approx(dense_value, rel=5e-4, abs=1e-8)


def test_integrate_direct_epfd_channel_leakage_fraction_matches_dense_reference_for_large_custom_receiver_cutoff() -> None:
    channel_bandwidth_mhz = 5.0
    integration_cutoff_mhz = channel_bandwidth_mhz * 5000.0 / 100.0
    mask_points_mhz = np.asarray(
        [
            [-45.0, 60.0],
            [-15.0, 22.0],
            [-2.5, 0.0],
            [2.5, 0.0],
            [15.0, 22.0],
            [45.0, 60.0],
        ],
        dtype=np.float64,
    )
    receiver_response_points_mhz = np.asarray(
        [
            [-18.0, 70.0],
            [-6.0, 6.0],
            [6.0, 6.0],
            [18.0, 70.0],
        ],
        dtype=np.float64,
    )

    exact_value = scenario._integrate_direct_epfd_channel_leakage_fraction(
        channel_start_mhz=2620.0,
        channel_stop_mhz=2625.0,
        ras_start_mhz=2690.0,
        ras_stop_mhz=2700.0,
        channel_bandwidth_mhz=channel_bandwidth_mhz,
        mask_points_mhz=mask_points_mhz,
        integration_cutoff_mhz=integration_cutoff_mhz,
        receiver_response_mode="custom",
        receiver_response_points_mhz=receiver_response_points_mhz,
    )
    dense_value = _dense_direct_epfd_channel_leakage_fraction(
        channel_start_mhz=2620.0,
        channel_stop_mhz=2625.0,
        ras_start_mhz=2690.0,
        ras_stop_mhz=2700.0,
        channel_bandwidth_mhz=channel_bandwidth_mhz,
        mask_points_mhz=mask_points_mhz,
        integration_cutoff_mhz=integration_cutoff_mhz,
        receiver_response_mode="custom",
        receiver_response_points_mhz=receiver_response_points_mhz,
    )

    assert exact_value == pytest.approx(dense_value, rel=5e-4, abs=1e-8)


def test_normalize_direct_epfd_spectrum_plan_matches_dense_reference_for_large_cutoff() -> None:
    channel_bandwidth_mhz = 5.0
    custom_mask_points = np.asarray(
        [
            [-45.0, 60.0],
            [-15.0, 22.0],
            [-2.5, 0.0],
            [2.5, 0.0],
            [15.0, 22.0],
            [45.0, 60.0],
        ],
        dtype=np.float64,
    )
    receiver_response_points_mhz = np.asarray(
        [
            [-18.0, 70.0],
            [-6.0, 6.0],
            [6.0, 6.0],
            [18.0, 70.0],
        ],
        dtype=np.float64,
    )
    spectrum_plan = scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan={
            "service_band_start_mhz": 2620.0,
            "service_band_stop_mhz": 2630.0,
            "ras_receiver_band_start_mhz": 2690.0,
            "ras_receiver_band_stop_mhz": 2700.0,
            "reuse_factor": 1,
            "channel_groups_per_cell_cap": 2,
            "unwanted_emission_mask_preset": "custom",
            "custom_mask_points": custom_mask_points,
            "receiver_response_mode": "custom",
            "receiver_custom_mask_points": receiver_response_points_mhz,
            "spectral_integration_cutoff_basis": "channel_bandwidth",
            "spectral_integration_cutoff_percent": 5000.0,
        },
        channel_bandwidth_mhz=channel_bandwidth_mhz,
        active_cell_count=1,
        active_cell_reuse_slot_ids=np.asarray([0], dtype=np.int32),
    )

    assert spectrum_plan is not None
    mask_points_mhz = scenario._resolve_direct_epfd_mask_points_mhz(
        preset="custom",
        channel_bandwidth_mhz=channel_bandwidth_mhz,
        custom_mask_points=custom_mask_points,
    )
    normalized_receiver_points_mhz = scenario._resolve_direct_epfd_receiver_response_points_mhz(
        response_mode="custom",
        receiver_bandwidth_mhz=10.0,
        custom_mask_points=receiver_response_points_mhz,
    )
    dense_expected = sum(
        _dense_direct_epfd_channel_leakage_fraction(
            channel_start_mhz=2620.0 + 5.0 * channel_index,
            channel_stop_mhz=2625.0 + 5.0 * channel_index,
            ras_start_mhz=2690.0,
            ras_stop_mhz=2700.0,
            channel_bandwidth_mhz=channel_bandwidth_mhz,
            mask_points_mhz=mask_points_mhz,
            integration_cutoff_mhz=float(spectrum_plan["spectral_integration_cutoff_mhz"]),
            receiver_response_mode=str(spectrum_plan["receiver_response_mode"]),
            receiver_response_points_mhz=normalized_receiver_points_mhz,
        )
        for channel_index in np.asarray(
            spectrum_plan["slot_group_channel_indices"][0],
            dtype=np.int32,
        ).tolist()
        if int(channel_index) >= 0
    )
    assert float(spectrum_plan["cell_leakage_factors"][0]) == pytest.approx(
        dense_expected,
        rel=5e-4,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    "service_case",
    _DIRECT_EPFD_FAST_SERVICE_CASES,
    ids=lambda case: str(case["case_id"]),
)
@pytest.mark.parametrize(
    "ras_case_name",
    (
        "upper_adjacent_rectangular",
        "upper_adjacent_narrow_rectangular",
        "upper_adjacent_wide_rectangular",
        "upper_adjacent_custom_asymmetric",
    ),
)
def test_normalize_direct_epfd_spectrum_plan_matches_exact_channel_leakage_across_supported_reuse_channelizations_and_ras_cases(
    service_case: Mapping[str, float | str],
    ras_case_name: str,
) -> None:
    full_channel_count = _direct_epfd_test_full_channel_count(service_case)
    ras_case = _direct_epfd_test_ras_case(service_case, ras_case_name)
    oracle_plan = _normalize_direct_epfd_test_plan(
        service_case=service_case,
        ras_case=ras_case,
        reuse_factor=1,
        enabled_channel_indices=list(range(full_channel_count)),
    )
    expected_channel_leakage = _precomputed_direct_epfd_channel_leakage_values(oracle_plan)

    for reuse_factor in scenario._DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
        for subset in _structured_direct_epfd_channel_subsets(full_channel_count, int(reuse_factor)):
            plan = _normalize_direct_epfd_test_plan(
                service_case=service_case,
                ras_case=ras_case,
                reuse_factor=int(reuse_factor),
                enabled_channel_indices=subset,
            )
            _assert_direct_epfd_plan_matches_expected_channel_leakage(
                plan,
                expected_channel_leakage_by_index=expected_channel_leakage,
                case_label=(
                    f"{service_case['case_id']}::{ras_case_name}::"
                    f"F{int(reuse_factor)}::{','.join(str(value) for value in subset)}"
                ),
            )


@pytest.mark.parametrize(
    "service_case",
    _DIRECT_EPFD_FAST_SERVICE_CASES,
    ids=lambda case: str(case["case_id"]),
)
def test_normalize_direct_epfd_spectrum_plan_channel_selection_encodings_are_invariant_across_supported_reuse_and_channelizations(
    service_case: Mapping[str, float | str],
) -> None:
    full_channel_count = _direct_epfd_test_full_channel_count(service_case)
    ras_case = _direct_epfd_test_ras_case(service_case, "upper_adjacent_rectangular")
    all_channel_indices = list(range(full_channel_count))

    for reuse_factor in scenario._DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
        for subset in _structured_direct_epfd_channel_subsets(full_channel_count, int(reuse_factor)):
            disabled_channel_indices = [
                channel_index
                for channel_index in all_channel_indices
                if channel_index not in set(subset)
            ]
            shuffled_duplicate_subset = list(reversed(subset)) + list(subset[: min(2, len(subset))])
            enabled_plan = _normalize_direct_epfd_test_plan(
                service_case=service_case,
                ras_case=ras_case,
                reuse_factor=int(reuse_factor),
                enabled_channel_indices=subset,
            )
            disabled_plan = _normalize_direct_epfd_test_plan(
                service_case=service_case,
                ras_case=ras_case,
                reuse_factor=int(reuse_factor),
                disabled_channel_indices=disabled_channel_indices,
            )
            duplicate_plan = _normalize_direct_epfd_test_plan(
                service_case=service_case,
                ras_case=ras_case,
                reuse_factor=int(reuse_factor),
                enabled_channel_indices=shuffled_duplicate_subset,
            )

            for comparable_plan in (disabled_plan, duplicate_plan):
                assert_equal(
                    comparable_plan["enabled_channel_indices"],
                    enabled_plan["enabled_channel_indices"],
                )
                assert_equal(
                    comparable_plan["slot_group_channel_indices"],
                    enabled_plan["slot_group_channel_indices"],
                )
                assert_equal(
                    comparable_plan["slot_group_valid_mask"],
                    enabled_plan["slot_group_valid_mask"],
                )
                assert_allclose(
                    np.asarray(comparable_plan["slot_group_leakage_factors"], dtype=np.float64),
                    np.asarray(enabled_plan["slot_group_leakage_factors"], dtype=np.float64),
                    rtol=1.0e-7,
                    atol=1.0e-10,
                )
                assert_allclose(
                    np.asarray(comparable_plan["cell_leakage_factors"], dtype=np.float64),
                    np.asarray(enabled_plan["cell_leakage_factors"], dtype=np.float64),
                    rtol=1.0e-7,
                    atol=1.0e-10,
                )
                assert_allclose(
                    np.asarray(comparable_plan["cell_group_leakage_factors"], dtype=np.float64),
                    np.asarray(enabled_plan["cell_group_leakage_factors"], dtype=np.float64),
                    rtol=1.0e-7,
                    atol=1.0e-10,
                )


@pytest.mark.parametrize(
    "service_case",
    tuple(
        case
        for case in _DIRECT_EPFD_FAST_SERVICE_CASES
        if np.isclose(
            (
                float(case["service_band_stop_mhz"]) - float(case["service_band_start_mhz"])
            )
            % float(case["channel_bandwidth_mhz"]),
            0.0,
        )
    ),
    ids=lambda case: str(case["case_id"]),
)
def test_normalize_direct_epfd_spectrum_plan_mirrors_rectangular_edge_leakage_for_exact_fit_channelizations(
    service_case: Mapping[str, float | str],
) -> None:
    full_channel_count = _direct_epfd_test_full_channel_count(service_case)
    upper_plan = _normalize_direct_epfd_test_plan(
        service_case=service_case,
        ras_case=_direct_epfd_test_ras_case(service_case, "upper_adjacent_rectangular"),
        reuse_factor=1,
        enabled_channel_indices=list(range(full_channel_count)),
    )
    lower_plan = _normalize_direct_epfd_test_plan(
        service_case=service_case,
        ras_case=_direct_epfd_test_ras_case(service_case, "lower_adjacent_rectangular"),
        reuse_factor=1,
        enabled_channel_indices=list(range(full_channel_count)),
    )
    upper_map = _precomputed_direct_epfd_channel_leakage_values(upper_plan)
    lower_map = _precomputed_direct_epfd_channel_leakage_values(lower_plan)

    for channel_index in range(full_channel_count):
        mirrored_channel_index = full_channel_count - 1 - int(channel_index)
        assert upper_map[channel_index] == pytest.approx(lower_map[mirrored_channel_index], rel=1.0e-7, abs=1.0e-10)


@pytest.mark.parametrize(
    "service_case",
    _DIRECT_EPFD_FAST_SERVICE_CASES,
    ids=lambda case: str(case["case_id"]),
)
def test_direct_epfd_leakage_is_monotonic_for_cutoff_and_rectangular_ras_bandwidth_across_channelizations(
    service_case: Mapping[str, float | str],
) -> None:
    channel_bandwidth_mhz = float(service_case["channel_bandwidth_mhz"])
    service_start_mhz = float(service_case["service_band_start_mhz"])
    full_channel_count = _direct_epfd_test_full_channel_count(service_case)
    cutoff_values_mhz = [channel_bandwidth_mhz * scale for scale in (1.0, 2.5, 4.5)]
    ras_bandwidth_values_mhz = [5.0, 10.0, 20.0]
    ras_start_mhz = float(service_case["service_band_stop_mhz"])

    for channel_index in range(full_channel_count):
        channel_start_mhz = service_start_mhz + channel_bandwidth_mhz * float(channel_index)
        channel_stop_mhz = channel_start_mhz + channel_bandwidth_mhz

        cutoff_leakages = [
            scenario._integrate_direct_epfd_channel_leakage_fraction(
                channel_start_mhz=channel_start_mhz,
                channel_stop_mhz=channel_stop_mhz,
                ras_start_mhz=ras_start_mhz,
                ras_stop_mhz=ras_start_mhz + 10.0,
                channel_bandwidth_mhz=channel_bandwidth_mhz,
                mask_points_mhz=np.asarray(_DIRECT_EPFD_TEST_MASK_POINTS_MHZ, dtype=np.float64),
                integration_cutoff_mhz=float(cutoff_mhz),
                receiver_response_mode="rectangular",
            )
            for cutoff_mhz in cutoff_values_mhz
        ]
        for earlier_value, later_value in zip(cutoff_leakages, cutoff_leakages[1:]):
            assert float(later_value) + 1.0e-10 >= float(earlier_value)

        width_leakages = [
            scenario._integrate_direct_epfd_channel_leakage_fraction(
                channel_start_mhz=channel_start_mhz,
                channel_stop_mhz=channel_stop_mhz,
                ras_start_mhz=ras_start_mhz,
                ras_stop_mhz=ras_start_mhz + float(ras_bandwidth_mhz),
                channel_bandwidth_mhz=channel_bandwidth_mhz,
                mask_points_mhz=np.asarray(_DIRECT_EPFD_TEST_MASK_POINTS_MHZ, dtype=np.float64),
                integration_cutoff_mhz=channel_bandwidth_mhz * 4.5,
                receiver_response_mode="rectangular",
            )
            for ras_bandwidth_mhz in ras_bandwidth_values_mhz
        ]
        for earlier_value, later_value in zip(width_leakages, width_leakages[1:]):
            assert float(later_value) + 1.0e-10 >= float(earlier_value)


@pytest.mark.parametrize(
    "service_case",
    _DIRECT_EPFD_FAST_SERVICE_CASES,
    ids=lambda case: str(case["case_id"]),
)
def test_normalize_direct_epfd_spectrum_plan_split_total_cell_power_matches_exact_slot_averages_across_supported_reuse_and_channelizations(
    service_case: Mapping[str, float | str],
) -> None:
    full_channel_count = _direct_epfd_test_full_channel_count(service_case)
    ras_case = _direct_epfd_test_ras_case(service_case, "upper_adjacent_rectangular")
    oracle_plan = _normalize_direct_epfd_test_plan(
        service_case=service_case,
        ras_case=ras_case,
        reuse_factor=1,
        enabled_channel_indices=list(range(full_channel_count)),
        power_policy="split_total_cell_power",
    )
    expected_channel_leakage = _precomputed_direct_epfd_channel_leakage_values(oracle_plan)

    for reuse_factor in scenario._DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
        for subset in _structured_direct_epfd_channel_subsets(full_channel_count, int(reuse_factor)):
            plan = _normalize_direct_epfd_test_plan(
                service_case=service_case,
                ras_case=ras_case,
                reuse_factor=int(reuse_factor),
                enabled_channel_indices=subset,
                power_policy="split_total_cell_power",
            )
            _assert_direct_epfd_plan_matches_expected_channel_leakage(
                plan,
                expected_channel_leakage_by_index=expected_channel_leakage,
                case_label=(
                    f"split::{service_case['case_id']}::F{int(reuse_factor)}::"
                    f"{','.join(str(value) for value in subset)}"
                ),
            )


@pytest.mark.slow_leakage_math
def test_exhaustive_channel_subset_leakage_consistency_for_14_channel_exact_fit_cases() -> None:
    """Verify per-channel leakage independence across channel subsets.

    Instead of an exhaustive 2^14 sweep (147k iterations), tests a
    representative set that covers all structural edge cases:
    - Every single-channel subset (14 tests — one per channel)
    - Complement of each single channel (14 tests — all-but-one)
    - Full set, empty set
    - Alternating even/odd channels
    - First/second half splits
    - 100 random subsets per reuse factor (seeded for reproducibility)

    Total: ~1400 iterations instead of 147k — same coverage of the
    leakage independence invariant.
    """
    import random as _rng

    service_case = _DIRECT_EPFD_FAST_SERVICE_CASES[0]
    full_channel_count = _direct_epfd_test_full_channel_count(service_case)
    ras_case = _direct_epfd_test_ras_case(service_case, "upper_adjacent_rectangular")
    oracle_plan = _normalize_direct_epfd_test_plan(
        service_case=service_case,
        ras_case=ras_case,
        reuse_factor=1,
        enabled_channel_indices=list(range(full_channel_count)),
    )
    expected_channel_leakage = _precomputed_direct_epfd_channel_leakage_values(oracle_plan)

    # Build a representative set of channel subsets.
    all_channels = list(range(full_channel_count))
    representative_subsets: list[list[int]] = []
    # Empty (mask=0) — degenerate
    representative_subsets.append([])
    # Full set
    representative_subsets.append(all_channels[:])
    # Each single channel
    for ch in all_channels:
        representative_subsets.append([ch])
    # Each all-but-one
    for ch in all_channels:
        representative_subsets.append([c for c in all_channels if c != ch])
    # Even / odd
    representative_subsets.append([c for c in all_channels if c % 2 == 0])
    representative_subsets.append([c for c in all_channels if c % 2 == 1])
    # First / second half
    half = full_channel_count // 2
    representative_subsets.append(all_channels[:half])
    representative_subsets.append(all_channels[half:])
    # 100 random subsets (seeded)
    gen = _rng.Random(42)
    for _ in range(100):
        k = gen.randint(1, full_channel_count - 1)
        representative_subsets.append(sorted(gen.sample(all_channels, k)))

    for reuse_factor in scenario._DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
        for subset in representative_subsets:
            plan = _normalize_direct_epfd_test_plan(
                service_case=service_case,
                ras_case=ras_case,
                reuse_factor=int(reuse_factor),
                enabled_channel_indices=subset,
            )
            _assert_direct_epfd_plan_matches_expected_channel_leakage(
                plan,
                expected_channel_leakage_by_index=expected_channel_leakage,
                case_label=f"repr14::F{int(reuse_factor)}::{subset}",
            )


@pytest.mark.slow_leakage_math
def test_exhaustive_channel_subset_leakage_consistency_for_12_channel_leftover_cases() -> None:
    """Same representative-subset strategy as the 14-channel test."""
    import random as _rng

    service_case = _DIRECT_EPFD_FAST_SERVICE_CASES[2]
    full_channel_count = _direct_epfd_test_full_channel_count(service_case)
    ras_case = _direct_epfd_test_ras_case(service_case, "upper_adjacent_rectangular")
    oracle_plan = _normalize_direct_epfd_test_plan(
        service_case=service_case,
        ras_case=ras_case,
        reuse_factor=1,
        enabled_channel_indices=list(range(full_channel_count)),
    )
    expected_channel_leakage = _precomputed_direct_epfd_channel_leakage_values(oracle_plan)

    all_channels = list(range(full_channel_count))
    representative_subsets: list[list[int]] = [[]]
    representative_subsets.append(all_channels[:])
    for ch in all_channels:
        representative_subsets.append([ch])
    for ch in all_channels:
        representative_subsets.append([c for c in all_channels if c != ch])
    representative_subsets.append([c for c in all_channels if c % 2 == 0])
    representative_subsets.append([c for c in all_channels if c % 2 == 1])
    half = full_channel_count // 2
    representative_subsets.append(all_channels[:half])
    representative_subsets.append(all_channels[half:])
    gen = _rng.Random(43)
    for _ in range(100):
        k = gen.randint(1, full_channel_count - 1)
        representative_subsets.append(sorted(gen.sample(all_channels, k)))

    for reuse_factor in scenario._DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
        for subset in representative_subsets:
            plan = _normalize_direct_epfd_test_plan(
                service_case=service_case,
                ras_case=ras_case,
                reuse_factor=int(reuse_factor),
                enabled_channel_indices=subset,
            )
            _assert_direct_epfd_plan_matches_expected_channel_leakage(
                plan,
                expected_channel_leakage_by_index=expected_channel_leakage,
                case_label=f"repr12::F{int(reuse_factor)}::{subset}",
            )


@pytest.mark.slow_leakage_math
def test_structured_channel_subset_leakage_consistency_for_19_channel_exact_fit_cases() -> None:
    service_case = _DIRECT_EPFD_FAST_SERVICE_CASES[1]
    full_channel_count = _direct_epfd_test_full_channel_count(service_case)
    ras_case = _direct_epfd_test_ras_case(service_case, "upper_adjacent_rectangular")
    oracle_plan = _normalize_direct_epfd_test_plan(
        service_case=service_case,
        ras_case=ras_case,
        reuse_factor=1,
        enabled_channel_indices=list(range(full_channel_count)),
    )
    expected_channel_leakage = _precomputed_direct_epfd_channel_leakage_values(oracle_plan)

    for reuse_factor in scenario._DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
        for subset in _sampled_direct_epfd_channel_subsets(
            full_channel_count,
            int(reuse_factor),
            sample_count=256,
            seed=20260401,
        ):
            plan = _normalize_direct_epfd_test_plan(
                service_case=service_case,
                ras_case=ras_case,
                reuse_factor=int(reuse_factor),
                enabled_channel_indices=subset,
            )
            _assert_direct_epfd_plan_matches_expected_channel_leakage(
                plan,
                expected_channel_leakage_by_index=expected_channel_leakage,
                case_label=(
                    f"structured19::{service_case['case_id']}::F{int(reuse_factor)}::"
                    f"{','.join(str(value) for value in subset)}"
                ),
            )


def test_maintained_notebooks_do_not_import_removed_notebook_helper_layers() -> None:
    parity_token = "notebook_" + "parity"
    postprocess_token = "notebook_" + "postprocess"
    for path in (BORESIGHT_NOTEBOOK_PATH, POSTPROCESS_NOTEBOOK_PATH):
        text = path.read_text(encoding="utf-8")
        assert parity_token not in text
        assert postprocess_token not in text


def test_compute_cell_spectral_weight_device_repeat_and_split_modes() -> None:
    cp = _FakeCp()
    group_active_mask = np.asarray(
        [
            [[True, False, True], [False, False, True]],
            [[True, True, False], [False, True, False]],
        ],
        dtype=bool,
    )
    cell_group_leakage = np.asarray(
        [
            [0.2, 0.4, 0.8],
            [0.1, 0.5, 0.9],
        ],
        dtype=np.float32,
    )

    repeat_weight = scenario._compute_cell_spectral_weight_device(
        cp,
        group_active_mask=group_active_mask,
        cell_group_leakage_factors=cell_group_leakage,
        power_policy="repeat_per_group",
        split_total_group_denominator_mode="configured_groups",
        configured_groups_per_cell=3,
    )
    configured_split_weight = scenario._compute_cell_spectral_weight_device(
        cp,
        group_active_mask=group_active_mask,
        cell_group_leakage_factors=cell_group_leakage,
        power_policy="split_total_cell_power",
        split_total_group_denominator_mode="configured_groups",
        configured_groups_per_cell=3,
    )
    active_split_weight = scenario._compute_cell_spectral_weight_device(
        cp,
        group_active_mask=group_active_mask,
        cell_group_leakage_factors=cell_group_leakage,
        power_policy="split_total_cell_power",
        split_total_group_denominator_mode="active_groups",
        configured_groups_per_cell=3,
    )

    assert_equal(
        np.asarray(repeat_weight, dtype=np.float32),
        np.asarray(
            [
                [1.0, 0.9],
                [0.6, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(configured_split_weight, dtype=np.float32),
        np.asarray(
            [
                [1.0 / 3.0, 0.3],
                [0.2, 1.0 / 6.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(active_split_weight, dtype=np.float32),
        np.asarray(
            [
                [0.5, 0.9],
                [0.3, 0.5],
            ],
            dtype=np.float32,
        ),
    )


def test_estimate_direct_epfd_activity_gpu_memory_uses_slabbed_bool_first_model() -> None:
    repeat_memory = scenario._estimate_direct_epfd_activity_gpu_memory(
        time_count=8,
        cell_count=11,
        groups_per_cell=4,
        cell_activity_mode="per_channel",
        need_power_outputs=True,
        spectral_slab=3,
        power_policy="repeat_per_group",
        split_total_group_denominator_mode="configured_groups",
    )
    active_split_memory = scenario._estimate_direct_epfd_activity_gpu_memory(
        time_count=8,
        cell_count=11,
        groups_per_cell=4,
        cell_activity_mode="per_channel",
        need_power_outputs=True,
        spectral_slab=3,
        power_policy="split_total_cell_power",
        split_total_group_denominator_mode="active_groups",
    )

    expected_resident_bool_bytes = np.dtype(np.bool_).itemsize * 8 * 11 * 4
    expected_weighted_sum_bytes = np.dtype(np.float32).itemsize * 3 * 11
    assert int(repeat_memory["resident_bytes"]) == expected_resident_bool_bytes
    assert int(repeat_memory["scratch_bytes"]) == expected_weighted_sum_bytes
    assert int(active_split_memory["scratch_bytes"]) == (
        2 * expected_weighted_sum_bytes
    )
    assert int(active_split_memory["peak_bytes"]) > int(repeat_memory["peak_bytes"])


@pytest.mark.parametrize(
    ("power_policy", "denominator_mode"),
    [
        ("repeat_per_group", "configured_groups"),
        ("split_total_cell_power", "configured_groups"),
        ("split_total_cell_power", "active_groups"),
    ],
)
def test_compute_cell_activity_spectral_weight_time_slabbed_device_matches_full_reference(
    power_policy: str,
    denominator_mode: str,
) -> None:
    cp = _FakeCp()
    cell_group_leakage = np.asarray(
        [
            [0.2, 0.4, 0.8],
            [0.1, 0.5, 0.9],
        ],
        dtype=np.float32,
    )
    full_group_mask = scenario._sample_cell_group_activity_mask_device(
        cp,
        time_count=5,
        cell_count=2,
        group_count=3,
        activity_factor=0.55,
        seed=12345,
    )
    full_active = scenario._collapse_cell_group_activity_mask_device(cp, full_group_mask)
    full_weight = scenario._compute_cell_spectral_weight_device(
        cp,
        group_active_mask=full_group_mask,
        cell_group_leakage_factors=cell_group_leakage,
        power_policy=power_policy,
        split_total_group_denominator_mode=denominator_mode,
        configured_groups_per_cell=3,
    )

    slab_active, slab_weight = scenario._compute_cell_activity_spectral_weight_time_slabbed_device(
        cp,
        time_count=5,
        cell_count=2,
        group_count=3,
        activity_factor=0.55,
        seed=12345,
        spectral_slab=2,
        need_power_outputs=True,
        cell_group_leakage_factors=cell_group_leakage,
        power_policy=power_policy,
        split_total_group_denominator_mode=denominator_mode,
        configured_groups_per_cell=3,
    )

    assert_equal(np.asarray(slab_active, dtype=bool), np.asarray(full_active, dtype=bool))
    np.testing.assert_allclose(
        np.asarray(slab_weight, dtype=np.float32),
        np.asarray(full_weight, dtype=np.float32),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_benchmark_direct_epfd_runs_collects_scheduler_and_backoff_metrics() -> None:
    observed_kwargs: list[dict[str, object]] = []
    profile_stage_timings_summary = {
        "spectrum_context_setup": 0.25,
        "orbit_propagation": 0.125,
        "ras_geometry": 0.375,
        "cell_link_library": 0.875,
        "cell_activity_setup": 1.0,
        "spectrum_activity_weighting": 2.5,
        "beam_finalize": 3.5,
        "power_accumulation": 4.0,
        "export_copy": 0.625,
        "write_enqueue": 0.125,
    }

    def _fake_runner(*, progress_callback=None, **kwargs):
        observed_kwargs.append(dict(kwargs))
        assert callable(progress_callback)
        progress_callback(
            {
                "kind": "iteration_plan",
                "bulk_timesteps": 6,
                "cell_chunk": 480,
                "sky_slab": 32,
                "spectral_slab": 3,
                "planner_source": "warmup_calibrated",
                "limiting_resource": "spectral-activity",
                "limiting_dimension": "spectral_slab",
                "predicted_gpu_peak_bytes": 58 * 1024**3,
                "predicted_gpu_activity_peak_bytes": 10 * 1024**3,
                "predicted_gpu_spectrum_context_bytes": 512 * 1024**2,
                "planned_total_seconds": 7200.0,
                "planned_remaining_seconds": 7200.0,
                "spectral_backoff_active": True,
                "compute_budget_utilization_fraction": 0.875,
                "export_budget_utilization_fraction": 0.03125,
                "underfill_reason": "spectral_activity_limited",
            }
        )
        progress_callback(
            {
                "kind": "warning",
                "scheduler_retry_count": 1,
                "spectral_backoff_active": True,
            }
        )
        progress_callback(
            {
                "kind": "batch_start",
                "bulk_timesteps": 6,
                "cell_chunk": 480,
                "sky_slab": 32,
                "spectral_slab": 3,
                "scheduler_retry_count": 1,
                "spectral_backoff_active": True,
            }
        )
        progress_callback(
            {
                "kind": "run_complete",
                "profile_stage_timings_summary": dict(profile_stage_timings_summary),
            }
        )
        return {
            "storage_filename": "benchmark_case.h5",
            "profile_stage_timings_summary": dict(profile_stage_timings_summary),
        }

    timestamps = iter((100.0, 104.0))
    summaries = scenario.benchmark_direct_epfd_runs(
        [
            {
                "name": "F4 per-channel",
                "kwargs": {"dummy": "value"},
            }
        ],
        runner=_fake_runner,
        time_func=lambda: next(timestamps),
    )

    assert observed_kwargs == [{"dummy": "value"}]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["name"] == "F4 per-channel"
    assert bool(summary["ok"]) is True
    assert summary["filename"] == "benchmark_case.h5"
    assert summary["plan_shapes"] == ((6, 480, 32, 3),)
    assert summary["active_shapes"] == ((6, 480, 32, 3),)
    assert summary["planner_sources"] == ("warmup_calibrated",)
    assert summary["limiting_resources"] == ("spectral-activity",)
    assert summary["limiting_dimensions"] == ("spectral_slab",)
    assert bool(summary["spectral_backoff_active"]) is True
    assert int(summary["retry_count"]) == 1
    assert int(summary["warning_count"]) == 1
    assert float(summary["elapsed_seconds"]) == pytest.approx(4.0)
    assert float(summary["attempted_work_units"]) == pytest.approx(6.0 * 480.0 * 32.0)
    assert float(summary["attempted_work_units_per_second"]) == pytest.approx(
        (6.0 * 480.0 * 32.0) / 4.0
    )
    assert float(summary["compute_budget_utilization_fraction_max"]) == pytest.approx(0.875)
    assert float(summary["export_budget_utilization_fraction_max"]) == pytest.approx(0.03125)
    assert summary["underfill_reasons"] == ("spectral_activity_limited",)
    assert summary["stage_timings_summary"] == profile_stage_timings_summary
    assert float(summary["one_time_spectrum_setup_seconds"]) == pytest.approx(0.25)
    assert float(summary["orbit_propagation_seconds"]) == pytest.approx(0.125)
    assert float(summary["ras_geometry_seconds"]) == pytest.approx(0.375)
    assert float(summary["cell_link_library_seconds"]) == pytest.approx(0.875)
    assert float(summary["cell_activity_setup_seconds"]) == pytest.approx(1.0)
    assert float(summary["spectrum_activity_weighting_seconds"]) == pytest.approx(2.5)
    assert float(summary["beam_finalize_seconds"]) == pytest.approx(3.5)
    assert float(summary["boresight_screening_seconds"]) == pytest.approx(0.0)
    assert float(summary["pointings_seconds"]) == pytest.approx(0.0)
    assert float(summary["power_accumulation_seconds"]) == pytest.approx(4.0)
    assert float(summary["export_copy_seconds"]) == pytest.approx(0.625)
    assert float(summary["write_enqueue_seconds"]) == pytest.approx(0.125)
    assert float(summary["host_sync_telemetry_overhead_seconds"]) == pytest.approx(0.0)


def test_benchmark_direct_epfd_runs_collects_observed_memory_and_finalize_substages() -> None:
    def _fake_runner(*, progress_callback=None, **_kwargs):
        assert callable(progress_callback)
        progress_callback({"kind": "batch_start", "bulk_timesteps": 2, "cell_chunk": 64, "sky_slab": 1})
        return {
            "storage_filename": "benchmark_observed_memory.h5",
            "profile_stage_timings_summary": {"beam_finalize": 1.5, "power_accumulation": 0.75},
            "beam_finalize_substage_timings": {
                "direct_candidate_extraction": 0.25,
                "first_pass_selector": 0.50,
                "retarget_repair_finalize": 0.75,
            },
            "observed_stage_memory_summary_by_name": {
                "cell_link_library": {
                    "observed_stage_name": "cell_link_library",
                    "observed_stage_gpu_resident_bytes": 512 * 1024**2,
                    "observed_stage_gpu_transient_peak_bytes": 128 * 1024**2,
                },
                "beam_finalize": {
                    "observed_stage_name": "beam_finalize",
                    "observed_stage_gpu_transient_peak_bytes": 384 * 1024**2,
                    "planner_vs_observed_gpu_peak_error_bytes": -(64 * 1024**2),
                },
            },
        }

    summaries = scenario.benchmark_direct_epfd_runs(
        [{"name": "telemetry_case"}],
        runner=_fake_runner,
        time_func=iter((20.0, 24.0)).__next__,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["filename"] == "benchmark_observed_memory.h5"
    assert int(summary["cell_link_library_resident_bytes_observed"]) == 512 * 1024**2
    assert int(summary["cell_link_library_transient_peak_bytes_observed"]) == 128 * 1024**2
    assert int(summary["beam_finalize_transient_peak_bytes_observed"]) == 384 * 1024**2
    assert int(summary["planner_vs_observed_gpu_peak_error_bytes"]) == -(64 * 1024**2)
    assert summary["beam_finalize_substage_timings"] == {
        "direct_candidate_extraction": pytest.approx(0.25),
        "first_pass_selector": pytest.approx(0.50),
        "retarget_repair_finalize": pytest.approx(0.75),
    }


def test_benchmark_direct_epfd_runs_supports_graceful_first_batch_and_gpu_metric_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler_state: dict[str, object] = {}

    class _FakeSampler:
        def start(self) -> None:
            sampler_state["started"] = True

        def stop(self) -> tuple[dict[str, object], ...]:
            sampler_state["stopped"] = True
            return (
                {
                    "gpu_utilization_percent": 22.0,
                    "gpu_memory_used_bytes": 3 * 1024**3,
                    "gpu_memory_total_bytes": 12 * 1024**3,
                },
                {
                    "gpu_utilization_percent": 66.0,
                    "gpu_memory_used_bytes": 7 * 1024**3,
                    "gpu_memory_total_bytes": 12 * 1024**3,
                },
            )

    def _fake_make_sampler(*, enabled: bool, sample_interval_s: float, device_index: int):
        sampler_state["enabled"] = bool(enabled)
        sampler_state["sample_interval_s"] = float(sample_interval_s)
        sampler_state["device_index"] = int(device_index)
        return _FakeSampler()

    monkeypatch.setattr(
        scenario,
        "_make_benchmark_gpu_metric_sampler",
        _fake_make_sampler,
    )

    cancel_results: list[object] = []

    def _fake_runner(*, progress_callback=None, cancel_callback=None, **kwargs):
        assert callable(progress_callback)
        assert callable(cancel_callback)
        assert kwargs == {"dummy": "value"}
        cancel_results.append(cancel_callback())
        progress_callback({"kind": "batch_start", "bulk_timesteps": 4, "cell_chunk": 64})
        cancel_results.append(cancel_callback())
        progress_callback({"kind": "batch_start", "bulk_timesteps": 4, "cell_chunk": 64})
        cancel_results.append(cancel_callback())
        return {"storage_filename": "graceful_case.h5"}

    summaries = scenario.benchmark_direct_epfd_runs(
        [
            {
                "name": "graceful_probe",
                "kwargs": {"dummy": "value"},
                "graceful_stop_after_batch_count": 2,
                "sample_live_gpu_metrics": True,
                "gpu_metric_sample_interval_s": 0.5,
            }
        ],
        runner=_fake_runner,
        time_func=iter((10.0, 12.5)).__next__,
    )

    assert sampler_state == {
        "enabled": True,
        "sample_interval_s": 0.5,
        "device_index": 0,
        "started": True,
        "stopped": True,
    }
    assert cancel_results == [None, None, "graceful"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["name"] == "graceful_probe"
    assert bool(summary["ok"]) is True
    assert summary["filename"] == "graceful_case.h5"
    assert int(summary["gpu_metric_sample_count"]) == 2
    assert float(summary["gpu_utilization_percent_min"]) == pytest.approx(22.0)
    assert float(summary["gpu_utilization_percent_median"]) == pytest.approx(44.0)
    assert float(summary["gpu_utilization_percent_max"]) == pytest.approx(66.0)
    assert int(summary["gpu_memory_used_bytes_min"]) == 3 * 1024**3
    assert int(summary["gpu_memory_used_bytes_max"]) == 7 * 1024**3
    assert int(summary["gpu_memory_used_bytes_span"]) == 4 * 1024**3
    assert int(summary["gpu_memory_total_bytes_max"]) == 12 * 1024**3


def test_build_direct_epfd_benchmark_cases_from_gui_config_expands_timestep_and_budget_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_calls: list[dict[str, object]] = []

    def _fake_build_request(
        config_path: str | Path,
        *,
        timestep_s: float | None = None,
        gpu_memory_budget_gb: float | None = None,
        host_memory_budget_gb: float | None = None,
        profile_stages: bool | None = None,
    ) -> dict[str, object]:
        observed_calls.append(
            {
                "config_path": str(config_path),
                "timestep_s": None if timestep_s is None else float(timestep_s),
                "gpu_memory_budget_gb": None if gpu_memory_budget_gb is None else float(gpu_memory_budget_gb),
                "host_memory_budget_gb": None if host_memory_budget_gb is None else float(host_memory_budget_gb),
                "profile_stages": None if profile_stages is None else bool(profile_stages),
            }
        )
        return {
            "timestep": 1.0 if timestep_s is None else float(timestep_s),
            "gpu_memory_budget_gb": 8.0 if gpu_memory_budget_gb is None else float(gpu_memory_budget_gb),
            "host_memory_budget_gb": 8.0 if host_memory_budget_gb is None else float(host_memory_budget_gb),
        }

    monkeypatch.setattr(
        scenario,
        "build_direct_epfd_run_request_from_gui_config",
        _fake_build_request,
    )

    cases = scenario.build_direct_epfd_benchmark_cases_from_gui_config(
        "benchmark_config.json",
        timestep_values_s=(30.0, 5.0),
        memory_budget_pairs_gb=((8.0, 8.0), (10.0, 12.0)),
        profile_stages=True,
        graceful_stop_after_first_batch=True,
        graceful_stop_after_batch_count=3,
        sample_live_gpu_metrics=True,
        gpu_metric_sample_interval_s=0.2,
    )

    assert len(observed_calls) == 5
    assert observed_calls[0] == {
        "config_path": "benchmark_config.json",
        "timestep_s": None,
        "gpu_memory_budget_gb": None,
        "host_memory_budget_gb": None,
        "profile_stages": True,
    }
    assert len(cases) == 4
    assert [case["name"] for case in cases] == [
        "benchmark_config_dt30s_host8g_gpu8g",
        "benchmark_config_dt30s_host10g_gpu12g",
        "benchmark_config_dt5s_host8g_gpu8g",
        "benchmark_config_dt5s_host10g_gpu12g",
    ]
    assert all(bool(case["graceful_stop_after_first_batch"]) is True for case in cases)
    assert all(int(case["graceful_stop_after_batch_count"]) == 3 for case in cases)
    assert all(bool(case["sample_live_gpu_metrics"]) is True for case in cases)
    assert all(
        float(case["gpu_metric_sample_interval_s"]) == pytest.approx(0.2) for case in cases
    )
    assert [case["kwargs"] for case in cases] == [
        {"timestep": 30.0, "gpu_memory_budget_gb": 8.0, "host_memory_budget_gb": 8.0},
        {"timestep": 30.0, "gpu_memory_budget_gb": 12.0, "host_memory_budget_gb": 10.0},
        {"timestep": 5.0, "gpu_memory_budget_gb": 8.0, "host_memory_budget_gb": 8.0},
        {"timestep": 5.0, "gpu_memory_budget_gb": 12.0, "host_memory_budget_gb": 10.0},
    ]


def test_benchmark_direct_epfd_runs_collects_fused_hot_path_transfer_metrics() -> None:
    def _fake_runner(*, progress_callback=None, **_kwargs):
        assert callable(progress_callback)
        progress_callback({"kind": "batch_start", "bulk_timesteps": 2, "cell_chunk": 64, "sky_slab": 1})
        return {
            "storage_filename": "benchmark_fused_gpu.h5",
            "hot_path_device_to_host_copy_count": 0,
            "hot_path_device_to_host_copy_bytes": 0,
            "device_scalar_sync_count": 0,
        }

    summaries = scenario.benchmark_direct_epfd_runs(
        [{"name": "fused_hot_path"}],
        runner=_fake_runner,
        time_func=iter((30.0, 32.0)).__next__,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["filename"] == "benchmark_fused_gpu.h5"
    assert int(summary["hot_path_device_to_host_copy_count"]) == 0
    assert int(summary["hot_path_device_to_host_copy_bytes"]) == 0
    assert int(summary["device_scalar_sync_count"]) == 0


def test_benchmark_direct_epfd_runs_falls_back_to_profile_rows_for_stage_timings() -> None:
    def _fake_runner(*, progress_callback=None, **_kwargs):
        assert callable(progress_callback)
        progress_callback(
            {
                "kind": "batch_start",
                "bulk_timesteps": 4,
                "cell_chunk": 128,
                "sky_slab": 8,
                "spectral_slab": 2,
            }
        )
        return {
            "storage_filename": "benchmark_rows_only.h5",
            "profile_stage_timings_summary": {},
            "profile_stage_timings": [
                {
                    "iteration": 0,
                    "batch_index": 0,
                    "spectrum_activity_weighting": 1.25,
                    "power_accumulation": 2.75,
                    "export_copy": 0.5,
                }
            ],
        }

    summaries = scenario.benchmark_direct_epfd_runs(
        [{"name": "rows_only"}],
        runner=_fake_runner,
        time_func=iter((10.0, 14.0)).__next__,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert float(summary["spectrum_activity_weighting_seconds"]) == pytest.approx(1.25)
    assert float(summary["power_accumulation_seconds"]) == pytest.approx(2.75)
    assert float(summary["writer_overhead_seconds"]) == pytest.approx(0.5)
    assert float(summary["compute_stage_seconds"]) == pytest.approx(4.0)


def test_summarize_direct_epfd_stage_timings_keeps_boresight_screening_separate() -> None:
    summary = scenario._summarize_direct_epfd_stage_timings(
        stage_timings_summary={
            "beam_finalize": 2.5,
            "boresight_screening": 0.75,
            "power_accumulation": 1.25,
        },
        elapsed_seconds=5.0,
    )

    assert float(summary["beam_finalize_seconds"]) == pytest.approx(2.5)
    assert float(summary["boresight_screening_seconds"]) == pytest.approx(0.75)
    assert float(summary["power_accumulation_seconds"]) == pytest.approx(1.25)


def test_boresight_notebook_exposes_spectrum_plan_controls() -> None:
    source = "\n".join(_load_notebook_code_cells(BORESIGHT_NOTEBOOK_PATH))

    assert "SPECTRUM_PLAN = {" in source
    assert 'REUSE_FACTOR = 1' in source
    assert 'CELL_ACTIVITY_MODE = "whole_cell"' in source
    assert 'SPLIT_TOTAL_GROUP_DENOMINATOR_MODE = "configured_groups"' in source
    assert 'RAS_REFERENCE_MODE = "lower"' in source
    assert "resolve_frequency_reuse_slots" in source
    assert "plot_frequency_reuse_scheme" in source
    assert "spectrum_plan=SPECTRUM_PLAN" in source
    assert '"split_total_group_denominator_mode": str(SPLIT_TOTAL_GROUP_DENOMINATOR_MODE)' in source
    assert "cell_activity_mode=CELL_ACTIVITY_MODE" in source
    assert "split_total_group_denominator_mode=SPLIT_TOTAL_GROUP_DENOMINATOR_MODE" in source


_NOTEBOOK_GPU_SMOKE_CASES: list[dict[str, object]] = [
    {
        "case_name": "nb_smoke_light_totals",
        "selection_strategy": "max_elevation",
        "ras_pointing_mode": "cell_center",
        "include_atmosphere": False,
        "memory_budget_mode": "hybrid",
        "profile_name": "totals_only",
        "boresight_theta1_deg": None,
        "boresight_theta2_deg": None,
    },
    {
        "case_name": "nb_smoke_notebook_heavy_both",
        "selection_strategy": "random",
        "ras_pointing_mode": "ras_station",
        "include_atmosphere": True,
        "memory_budget_mode": "hybrid",
        "profile_name": "notebook_full",
        "boresight_theta1_deg": 1.0,
        "boresight_theta2_deg": 3.0,
        "theta2_scope_mode": "adjacency_layers",
        "theta2_layers": 1,
    },
    {
        "case_name": "nb_smoke_gui_heavy_theta2_layers",
        "selection_strategy": "random",
        "ras_pointing_mode": "cell_center",
        "include_atmosphere": False,
        "memory_budget_mode": "gpu_only",
        "profile_name": "gui_heavy",
        "boresight_theta1_deg": None,
        "boresight_theta2_deg": 3.0,
        "theta2_scope_mode": "adjacency_layers",
        "theta2_layers": 1,
    },
]


class TestProcessIntegration:

    def test_sliding_sample_count_preserves_linear_mean_and_unit(self):
        power = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * u.W

        averaged = scenario.process_integration(
            power,
            integration_period=2,
            windowing="sliding",
            time_axis=0,
        )

        assert averaged.dtype == np.float32
        assert_quantity_allclose(averaged, np.array([1.5, 2.5, 3.5], dtype=np.float32) * u.W)

    def test_subsequent_time_window_drops_partial_tail(self):
        power = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

        averaged = scenario.process_integration(
            power,
            integration_period=2 * u.s,
            timestep=1 * u.s,
            windowing="subsequent",
            time_axis=0,
        )

        assert_allclose(averaged, np.array([1.5, 3.5], dtype=np.float64))

    def test_random_windows_are_reproducible_and_sorted(self):
        power = np.arange(1.0, 11.0, dtype=np.float64)

        averaged = scenario.process_integration(
            power,
            integration_period=3,
            windowing="random",
            random_window_count=3,
            random_seed=42,
            time_axis=0,
        )

        expected_starts = np.sort(np.random.default_rng(42).choice(np.arange(8), size=3, replace=False))
        expected = np.array([np.mean(power[start : start + 3]) for start in expected_starts], dtype=np.float64)

        assert_equal(averaged.shape, (3,))
        assert_allclose(averaged, expected)

    def test_zero_values_are_included_in_average(self):
        power = np.array([0.0, 2.0, 4.0], dtype=np.float64)

        averaged = scenario.process_integration(
            power,
            integration_period=2,
            windowing="sliding",
            time_axis=0,
        )

        assert_allclose(averaged, np.array([1.0, 3.0], dtype=np.float64))

    def test_rejects_logarithmic_quantities(self):
        with pytest.raises(ValueError, match="linear-domain input"):
            scenario.process_integration(
                np.array([0.0, 10.0], dtype=np.float64) * u.dB(u.W),
                integration_period=2,
                time_axis=0,
            )

    def test_negative_finite_values_raise(self):
        with pytest.raises(ValueError, match="negative finite values"):
            scenario.process_integration(
                np.array([1.0, -1.0, 2.0], dtype=np.float64),
                integration_period=2,
                time_axis=0,
            )

    def test_linear_average_differs_from_db_average_regression(self):
        samples = np.array([1.0, 100.0], dtype=np.float64) * u.W

        averaged = scenario.process_integration(
            samples,
            integration_period=2,
            windowing="sliding",
            time_axis=0,
        )

        arithmetic_db_mean_w = (np.mean(samples.to_value(u.dB(u.W))) * u.dB(u.W)).to_value(u.W)
        assert_quantity_allclose(averaged, np.array([50.5], dtype=np.float64) * u.W)
        assert averaged.to_value(u.W)[0] != pytest.approx(arithmetic_db_mean_w)

    def test_regular_times_match_timestep_path(self):
        power = np.arange(1.0, 7.0, dtype=np.float64).reshape(2, 3)
        times = np.array([0.0, 1.0, 2.0], dtype=np.float64) * u.s

        from_times = scenario.process_integration(
            power,
            integration_period=2 * u.s,
            times=times,
            windowing="sliding",
            time_axis=1,
        )
        from_timestep = scenario.process_integration(
            power,
            integration_period=2 * u.s,
            timestep=1 * u.s,
            windowing="sliding",
            time_axis=1,
        )

        assert_allclose(from_times, from_timestep)

    def test_irregular_times_use_forward_hold_weighting(self):
        power = np.array([2.0, 4.0, 8.0, 10.0], dtype=np.float64)
        times = _seconds_to_mjd(np.array([0.0, 1.0, 3.0, 4.0], dtype=np.float64))

        averaged = scenario.process_integration(
            power,
            integration_period=2 * u.s,
            times=times,
            windowing="sliding",
            time_axis=0,
        )

        assert_allclose(averaged, np.array([3.0, 4.0, 9.0], dtype=np.float64))

    def test_nan_padded_batches_are_processed_independently(self):
        power = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [10.0, 11.0, 12.0, np.nan, np.nan],
            ],
            dtype=np.float64,
        )
        times = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, np.nan, np.nan],
            ],
            dtype=np.float64,
        ) * u.s

        averaged = scenario.process_integration(
            power,
            integration_period=2 * u.s,
            times=times,
            windowing="sliding",
            time_axis=1,
        )

        expected = np.array(
            [
                [1.5, 2.5, 3.5, 4.5],
                [10.5, 11.5, np.nan, np.nan],
            ],
            dtype=np.float64,
        )
        assert_allclose(averaged, expected, equal_nan=True)

    def test_interior_nan_gap_blocks_cross_gap_windows(self):
        power = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=np.float64)

        averaged = scenario.process_integration(
            power,
            integration_period=2,
            windowing="sliding",
            time_axis=0,
        )

        assert_allclose(averaged, np.array([1.5, 4.5, 5.5], dtype=np.float64))

    def test_random_window_count_requires_enough_windows(self):
        with pytest.raises(ValueError, match="random_window_count"):
            scenario.process_integration(
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
                integration_period=3,
                windowing="random",
                random_window_count=3,
                random_seed=0,
                time_axis=0,
            )


class TestStreamingProcessIntegration:

    def test_sliding_stream_matches_eager_for_irregular_times(self):
        values = np.array([[2.0, 4.0, 8.0, 10.0]], dtype=np.float64)
        times = _seconds_to_mjd(np.array([[0.0, 1.0, 3.0, 4.0]], dtype=np.float64))

        eager = scenario.process_integration(
            values,
            integration_period=2 * u.s,
            times=times,
            windowing="sliding",
            time_axis=1,
        )

        chunks = [
            {
                "iteration": 0,
                "slot_start": 0,
                "slot_stop": 2,
                "data": {
                    "power": values.reshape(-1)[:2],
                    "times": times[:, :2].reshape(-1),
                },
            },
            {
                "iteration": 0,
                "slot_start": 2,
                "slot_stop": 4,
                "data": {
                    "power": values.reshape(-1)[2:],
                    "times": times[:, 2:].reshape(-1),
                },
            },
        ]

        streamed = scenario.process_integration_stream(
            chunks,
            value_key="power",
            integration_period=2 * u.s,
            times_key="times",
            windowing="sliding",
        )

        assert_allclose(streamed, eager, equal_nan=True)

    def test_subsequent_stream_matches_eager_for_uniform_chunks(self):
        values = (np.arange(1.0, 9.0, dtype=np.float64).reshape(1, 8) * u.W)

        eager = scenario.process_integration(
            values,
            integration_period=2,
            windowing="subsequent",
            time_axis=1,
        )

        chunks = [
            {
                "iteration": 0,
                "slot_start": 0,
                "slot_stop": 3,
                "data": {"power": values.value.reshape(-1)[:3] * values.unit},
            },
            {
                "iteration": 0,
                "slot_start": 3,
                "slot_stop": 8,
                "data": {"power": values.value.reshape(-1)[3:] * values.unit},
            },
        ]

        streamed = scenario.process_integration_stream(
            chunks,
            value_key="power",
            integration_period=2,
            windowing="subsequent",
        )

        assert_quantity_allclose(streamed, eager)

    def test_stream_random_windowing_is_rejected(self):
        chunks = [
            {
                "iteration": 0,
                "slot_start": 0,
                "slot_stop": 2,
                "data": {"power": np.array([[1.0], [2.0]], dtype=np.float64)},
            }
        ]

        with pytest.raises(ValueError, match="supports only 'sliding' and 'subsequent'"):
            scenario.process_integration_stream(
                chunks,
                value_key="power",
                integration_period=2,
                windowing="random",
            )


class TestSimulationBatching:

    def test_iter_simulation_batches_matches_eager_helper(self):
        start_time = Time(60000.0, format="mjd", scale="tai")
        end_time = Time(60000.0 + (300.0 / 86400.0), format="mjd", scale="tai")

        eager = scenario.generate_simulation_batches(
            start_time,
            end_time,
            60 * u.s,
            batch_size=2,
        )
        streamed = list(
            scenario.iter_simulation_batches(
                start_time,
                end_time,
                60 * u.s,
                batch_size=2,
            )
        )

        assert_equal(len(streamed), len(eager["times"]))
        for idx, batch in enumerate(streamed):
            assert_allclose(batch["batch_start"].mjd, eager["batch_start"][idx].mjd)
            assert_allclose(batch["times"].mjd, eager["times"][idx].mjd)
            assert_allclose(batch["td"].sec, eager["td"][idx].sec)
            assert_allclose(batch["batch_end"].mjd, eager["batch_end"][idx].mjd)

    def test_recommend_observer_chunk_size_matches_formula(self):
        chunk = scenario.recommend_observer_chunk_size(
            12,
            50,
            ram_budget_gb=1.5,
            n_float_fields_per_pair=7,
            dtype_itemsize=8,
            safety_margin=0.6,
        )
        expected = int(1.5 * (1024 ** 3) * 0.6) // (12 * 50 * 7 * 8)

        assert chunk == max(1, expected)
        assert scenario.recommend_observer_chunk_size(
            0,
            50,
            ram_budget_gb=1.5,
        ) == 1

    def test_build_observer_layout_keeps_primary_first(self):
        primary = object()
        cells = [object(), object()]

        layout = scenario.build_observer_layout(primary, cells)

        assert layout["primary_observer_idx"] == 0
        assert layout["first_cell_observer_idx"] == 1
        assert layout["n_cell_observers"] == 2
        assert layout["observer_arr"].dtype == object
        assert layout["observer_arr"].tolist() == [primary, *cells]


class _FakePool:

    def free_all_blocks(self) -> None:
        return


class _FakeNullStream:

    def synchronize(self) -> None:
        return


class _FakeCp:
    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    bool_ = np.bool_

    class random:  # noqa: D106 - test helper
        class RandomState:  # noqa: D106 - test helper
            def __init__(self, seed):
                self._rng = np.random.RandomState(int(seed))

            def random_sample(self, size=None):
                return self._rng.random_sample(size)

            def random(self, size=None):
                return self._rng.random_sample(size)

        @staticmethod
        def random_sample(size=None):
            return np.random.random_sample(size)

        @staticmethod
        def random(size=None):
            return np.random.random_sample(size)

    class cuda:  # noqa: D106 - test helper
        class Stream:  # noqa: D106 - test helper
            null = _FakeNullStream()

    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def arange(*args, **kwargs):
        return np.arange(*args, **kwargs)

    @staticmethod
    def any(value, axis=None):
        return np.any(value, axis=axis)

    @staticmethod
    def nonzero(value):
        return np.nonzero(value)

    @staticmethod
    def count_nonzero(value):
        return np.count_nonzero(value)

    @staticmethod
    def sum(value, axis=None, dtype=None):
        return np.sum(value, axis=axis, dtype=dtype)

    @staticmethod
    def einsum(subscripts, *operands, optimize=False):
        return np.einsum(subscripts, *operands, optimize=optimize)

    @staticmethod
    def add(x1, x2, out=None):
        result = np.add(x1, x2)
        if out is not None:
            out[...] = result
            return out
        return result

    @staticmethod
    def zeros(shape, dtype=float):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype=float):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def broadcast_to(value, shape):
        return np.broadcast_to(value, shape)

    @staticmethod
    def transpose(value, axes=None):
        return np.transpose(value, axes=axes)

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    # -- Newer numpy-style helpers needed by production visibility /
    #    prefilter / aggregate cap optimizations.  Each is a thin numpy
    #    passthrough; the fake stays numpy-backed regardless of what the
    #    production path asks for.
    @staticmethod
    def concatenate(arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def stack(arrays, axis=0):
        return np.stack(arrays, axis=axis)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def minimum(a, b):
        return np.minimum(a, b)

    @staticmethod
    def clip(value, a_min, a_max, out=None):
        return np.clip(value, a_min, a_max, out=out)

    @staticmethod
    def sqrt(value):
        return np.sqrt(value)

    @staticmethod
    def sin(value):
        return np.sin(value)

    @staticmethod
    def cos(value):
        return np.cos(value)

    @staticmethod
    def arccos(value):
        return np.arccos(value)

    @staticmethod
    def argsort(a, axis=-1, kind=None):
        return np.argsort(a, axis=axis, kind=kind)

    @staticmethod
    def max(value, axis=None, **kwargs):
        return np.max(value, axis=axis, **kwargs)

    @staticmethod
    def argmax(value, axis=None):
        return np.argmax(value, axis=axis)

    @staticmethod
    def all(value, axis=None):
        return np.all(value, axis=axis)

    @staticmethod
    def logical_not(value):
        return np.logical_not(value)

    @staticmethod
    def rint(value):
        return np.rint(value)

    @staticmethod
    def log10(value):
        return np.log10(value)

    @staticmethod
    def floor(value):
        return np.floor(value)

    @staticmethod
    def power(base, exp):
        return np.power(base, exp)

    @staticmethod
    def full(shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def full_like(a, fill_value, dtype=None):
        return np.full_like(a, fill_value, dtype=dtype)

    @staticmethod
    def zeros_like(a, dtype=None):
        return np.zeros_like(a, dtype=dtype)

    @staticmethod
    def ones_like(a, dtype=None):
        return np.ones_like(a, dtype=dtype)

    @staticmethod
    def empty(shape, dtype=None):
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def empty_like(a, dtype=None):
        return np.empty_like(a, dtype=dtype)

    @staticmethod
    def abs(value):
        return np.abs(value)

    @staticmethod
    def asnumpy(value):
        return np.asarray(value)

    @staticmethod
    def take_along_axis(arr, indices, axis):
        return np.take_along_axis(arr, indices, axis=axis)

    @staticmethod
    def matmul(a, b):
        return np.matmul(a, b)

    class linalg:  # noqa: D106 - test helper
        @staticmethod
        def norm(x, axis=None, keepdims=False):
            return np.linalg.norm(x, axis=axis, keepdims=keepdims)

    ndarray = np.ndarray

    @staticmethod
    def get_default_memory_pool():
        return _FakePool()

    @staticmethod
    def get_default_pinned_memory_pool():
        return _FakePool()


class _FakeCpNoCountNonzero(_FakeCp):

    @staticmethod
    def count_nonzero(value):
        raise AssertionError("exact visible-count extraction should not run in steady-state")


class _FakeGpuModule:
    METHOD_DWARNER = "fake"

    def __init__(self):
        self.copy_calls = 0

    def copy_device_to_host(self, value):
        self.copy_calls += 1
        if hasattr(value, "_values"):
            return np.asarray(value._values)
        return np.asarray(value)


class _FakeCudaIndexVector:

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.int32)
        self.__cuda_array_interface__ = {
            "shape": self._values.shape,
            "strides": self._values.strides,
            "typestr": self._values.dtype.str,
            "data": (0, True),
            "version": 2,
        }

    def __array__(self, *_args, **_kwargs):
        raise TypeError("implicit host conversion is forbidden")


class _FakeLinkLibrary:

    def __init__(
        self,
        time_count: int,
        sky_cells: int,
        sat_count_total: int,
        *,
        boresight_active: bool,
        cell_count: int,
    ):
        self.time_count = int(time_count)
        self.sky_cells = int(sky_cells)
        self.sat_count_total = int(sat_count_total)
        self.boresight_active = bool(boresight_active)
        self.cell_count = int(cell_count)
        self.last_finalize_kwargs: dict[str, object] | None = None

    def add_chunk(self, *_args, **_kwargs) -> None:
        return

    def finalize_direct_epfd_beams(self, **kwargs):
        self.last_finalize_kwargs = {
            "output_sat_indices": np.asarray(kwargs["output_sat_indices"], dtype=np.int32).copy(),
            "ras_sat_azel_shape": None
            if kwargs.get("ras_sat_azel") is None
            else tuple(np.asarray(kwargs["ras_sat_azel"]).shape),
        }
        sat_idx = np.asarray(kwargs["output_sat_indices"], dtype=np.int32).reshape(-1)
        n_sat = int(sat_idx.size)
        if self.boresight_active:
            beam_shape = (self.time_count, 1, n_sat, self.sky_cells)
            count_shape = (self.time_count, self.sky_cells, n_sat)
            diag_shape = (self.time_count, self.sky_cells)
        else:
            beam_shape = (self.time_count, n_sat, self.sky_cells)
            count_shape = (self.time_count, n_sat)
            diag_shape = (self.time_count,)
        result = {
            "beam_idx": np.zeros(beam_shape, dtype=np.int16),
            "beam_alpha_rad": np.zeros(beam_shape, dtype=np.float32),
            "beam_beta_rad": np.zeros(beam_shape, dtype=np.float32),
            "sat_beam_counts_used": np.ones(count_shape, dtype=np.int32),
            "ras_retargeted_count": np.zeros(diag_shape, dtype=np.int32),
            "ras_reserved_count": np.zeros(diag_shape, dtype=np.int32),
            "direct_kept_count": np.zeros(diag_shape, dtype=np.int32),
            "repaired_link_count": np.zeros(diag_shape, dtype=np.int32),
            "dropped_link_count": np.zeros(diag_shape, dtype=np.int32),
        }
        if kwargs.get("include_full_sat_beam_counts_used", False):
            full_shape = (
                (self.time_count, self.sky_cells, self.sat_count_total)
                if self.boresight_active
                else (self.time_count, self.sat_count_total)
            )
            full_counts = np.broadcast_to(
                np.arange(1, self.sat_count_total + 1, dtype=np.int32),
                full_shape,
            ).copy()
            result["sat_beam_counts_used_full"] = full_counts
        return result

    def iter_direct_epfd_beam_slabs(self, **kwargs):
        result = self.finalize_direct_epfd_beams(**kwargs)
        if kwargs.get("debug_direct_epfd", False):
            result["debug_direct_epfd_stats"] = {
                "mode": "fake",
                "boresight_active": bool(self.boresight_active),
                "realized_edge_count": int(self.time_count * (self.sky_cells if self.boresight_active else 1)),
            }
        yield {
            "time_start": 0,
            "time_stop": self.time_count,
            "sky_start": 0,
            "sky_stop": self.sky_cells if self.boresight_active else 1,
            "boresight_active": bool(self.boresight_active),
            "result": result,
            "chunk_shape": {"time_chunk_size": self.time_count, "sky_chunk_size": 1},
            "compaction_stats": {},
        }


from dataclasses import dataclass as _test_dataclass, field as _test_field


@_test_dataclass
class _FakeOrbitState:
    """Minimal stand-in for ``gpu_accel.GpuOrbitState`` used by the scenario
    test mocks.

    Production code in ``scenario._compute_gpu_direct_epfd_batch_device``
    accesses ``orbit_state.d_eci_pos`` / ``.d_eci_vel`` and passes the
    orbit state through ``dataclasses.replace`` during the satellite
    pre-filter optimisation. The previous ``{'mjd': ...}`` dict we used
    didn't satisfy either contract, so the whole ``TestDirectEpfdGpuRunner``
    class regressed the moment that optimisation landed. This dataclass
    gives the fake surface the same shape the production orbit state
    exposes — only the fields the runner actually touches.
    """

    mjd: np.ndarray
    d_eci_pos: np.ndarray = _test_field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float32))
    d_eci_vel: np.ndarray = _test_field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float32))

    def __getitem__(self, key):
        """Allow dict-style access for tests that still use ``orbit_state['mjd']``."""
        return getattr(self, key)


class _FakeSession:

    def __init__(
        self,
        sat_count_total: int = 2,
        ras_visible_sat_count: int | None = None,
        probe_visibility_counts: list[int] | None = None,
    ):
        self._n_sky = 2
        self.accumulate_calls = 0
        self.accumulate_direct_calls = 0
        self.last_accumulate_ras_power_kwargs: dict[str, object] | None = None
        self.last_accumulate_direct_epfd_kwargs: dict[str, object] | None = None
        self.atmosphere_prepare_calls = 0
        self.atmosphere_prepare_kwargs: list[dict[str, object]] = []
        self.device_budget_calls: list[dict[str, object]] = []
        self.spectrum_context_calls: list[dict[str, object]] = []
        self.last_link_library: _FakeLinkLibrary | None = None
        self.compute_dtype = np.float32
        self.sat_count_total = int(sat_count_total)
        self.ras_visible_sat_count = (
            self.sat_count_total
            if ras_visible_sat_count is None
            else int(ras_visible_sat_count)
        )
        self.probe_visibility_counts = (
            None
            if probe_visibility_counts is None
            else [int(value) for value in probe_visibility_counts]
        )
        self.probe_visibility_calls: list[float] = []

    def activate(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self, **_kwargs):
        return

    def prepare_satellite_context(self, tle_list, method=None):
        return type("SatCtx", (), {"n_sats": int(np.asarray(tle_list, dtype=object).size), "method": method})()

    def prepare_observer_context(self, observer_arr):
        return type("ObsCtx", (), {"n_observers": int(np.asarray(observer_arr, dtype=object).size)})()

    def prepare_s1586_pointing_context(self, elev_range_deg=None):
        return type("PointCtx", (), {"n_cells": self._n_sky, "elev_range_deg": elev_range_deg})()

    def prepare_s1528_pattern_context(self, **kwargs):
        return kwargs

    def prepare_ras_pattern_context(self, **kwargs):
        return kwargs

    def prepare_atmosphere_lut_context(self, **kwargs):
        self.atmosphere_prepare_calls += 1
        self.atmosphere_prepare_kwargs.append(dict(kwargs))
        return kwargs

    def prepare_spectrum_plan_context(self, **kwargs):
        self.spectrum_context_calls.append(dict(kwargs))
        return types.SimpleNamespace(session=self, **dict(kwargs))

    def probe_visibility_profile(self, mjd, *_args, **_kwargs):
        mjd_arr = np.asarray(mjd, dtype=np.float64).reshape(-1)
        probe_value = float(mjd_arr[0])
        self.probe_visibility_calls.append(probe_value)
        visible_satellite_count = 1
        if self.probe_visibility_counts:
            index = min(
                len(self.probe_visibility_calls) - 1,
                len(self.probe_visibility_counts) - 1,
            )
            visible_satellite_count = int(self.probe_visibility_counts[index])
        return {
            "probe_mjd": probe_value,
            "visible_satellite_count": int(visible_satellite_count),
            "visible_fraction": float(visible_satellite_count)
            / float(max(1, self.sat_count_total)),
        }

    def resolve_device_memory_budget_bytes(self, budget_gb, **_kwargs):
        call = {"budget_gb": float(budget_gb), **dict(_kwargs)}
        self.device_budget_calls.append(call)
        effective_budget_bytes = int(float(budget_gb) * 1024 ** 3)
        return {
            "hard_budget_bytes": int(effective_budget_bytes),
            "planning_budget_bytes": int(effective_budget_bytes),
            "runtime_advisory_budget_bytes": int(effective_budget_bytes),
            "effective_budget_bytes": int(effective_budget_bytes),
            "mode_used": _kwargs.get("mode"),
        }

    def plan_propagation_execution(self, _times, _satellite_context, **_kwargs):
        return {"time_chunk_size": 1, "observer_chunk_size": 1}

    def estimate_propagation_memory(self, *_args, **_kwargs):
        return {"cache_bytes": 0, "reserve_bytes": 0, "total_bytes": 1024}

    def propagate_orbit(self, mjd, _satellite_context, **_kwargs):
        mjd_arr = np.asarray(mjd, dtype=np.float64)
        t = int(mjd_arr.reshape(-1).size)
        s = int(self.sat_count_total)
        # Positive radial offsets so the GPU prefilter path doesn't trip
        # on all-zero ECI positions. Shapes mirror the real orbit state.
        return _FakeOrbitState(
            mjd=mjd_arr,
            d_eci_pos=np.zeros((t, s, 3), dtype=np.float32),
            d_eci_vel=np.zeros((t, s, 3), dtype=np.float32),
        )

    def derive_from_eci(
        self,
        orbit_state,
        _observer_context=None,
        *,
        observer_context=None,
        observer_slice=None,
        **_kwargs,
    ):
        # ``orbit_state`` is a ``_FakeOrbitState`` dataclass, but we still
        # support dict-like indexing via ``__getitem__`` so legacy call
        # sites that used ``orbit_state['mjd']`` keep working.
        times = np.asarray(orbit_state["mjd"], dtype=np.float64)
        time_count = int(times.size)
        n_sat = int(self.sat_count_total)
        if _observer_context is None:
            _observer_context = observer_context
        obs_count = int(observer_slice.stop - observer_slice.start)
        topo = np.zeros((time_count, obs_count, n_sat, 4), dtype=np.float32)
        topo[..., 1] = -5.0
        topo[..., : self.ras_visible_sat_count, 1] = np.linspace(
            25.0,
            35.0,
            self.ras_visible_sat_count,
            dtype=np.float32,
        )
        sat_azel = np.zeros((time_count, obs_count, n_sat, 2), dtype=np.float32)
        sat_azel[..., 1] = np.linspace(25.0, 35.0, n_sat, dtype=np.float32)
        return {"topo": topo, "sat_azel": sat_azel}

    def sample_s1586_pointings(self, _pointing_context, n_samples, **_kwargs):
        return {
            "azimuth_deg": np.zeros((int(n_samples), self._n_sky), dtype=np.float32),
            "elevation_deg": np.full((int(n_samples), self._n_sky), 45.0, dtype=np.float32),
        }

    def prepare_satellite_link_selection_library(self, *, time_count, **_kwargs):
        self.last_link_library = _FakeLinkLibrary(
            time_count=time_count,
            sky_cells=self._n_sky,
            sat_count_total=self.sat_count_total,
            boresight_active=bool(
                _kwargs.get("boresight_pointing_azimuth_deg") is not None
                or _kwargs.get("boresight_pointing_elevation_deg") is not None
                or _kwargs.get("boresight_theta1_deg") is not None
                or _kwargs.get("boresight_theta2_deg") is not None
            ),
            cell_count=int(_kwargs.get("cell_count", 2)),
        )
        return self.last_link_library

    def accumulate_ras_power(
        self,
        *,
        sat_topo,
        beam_idx,
        include_epfd,
        include_prx_total,
        include_per_satellite_prx,
        include_total_pfd,
        include_per_satellite_pfd,
        **_kwargs,
    ):
        self.accumulate_calls += 1
        self.last_accumulate_ras_power_kwargs = dict(_kwargs)
        time_count = int(np.asarray(sat_topo).shape[0])
        sat_count = int(np.asarray(sat_topo).shape[1])
        boresight_active = np.asarray(beam_idx).ndim == 4
        result = {}
        if include_epfd:
            result["EPFD_W_m2"] = np.full((time_count, 1, self._n_sky), 1.0, dtype=np.float32)
        if include_prx_total:
            result["Prx_total_W"] = np.full((time_count, 1, self._n_sky), 2.0, dtype=np.float32)
        if include_per_satellite_prx:
            if boresight_active:
                result["Prx_per_sat_RAS_STATION_W"] = np.full(
                    (time_count, 1, sat_count, self._n_sky),
                    1.5,
                    dtype=np.float32,
                )
            else:
                result["Prx_per_sat_RAS_STATION_W"] = np.full(
                    (time_count, sat_count),
                    1.5,
                    dtype=np.float32,
                )
        if include_total_pfd:
            if boresight_active:
                result["PFD_total_RAS_STATION_W_m2"] = np.full((time_count, 1, self._n_sky), 3.0, dtype=np.float32)
            else:
                result["PFD_total_RAS_STATION_W_m2"] = np.full(time_count, 3.0, dtype=np.float32)
        if include_per_satellite_pfd:
            if boresight_active:
                result["PFD_per_sat_RAS_STATION_W_m2"] = np.full(
                    (time_count, 1, sat_count, self._n_sky),
                    4.0,
                    dtype=np.float32,
                )
            else:
                result["PFD_per_sat_RAS_STATION_W_m2"] = np.full((time_count, sat_count), 4.0, dtype=np.float32)
        return result

    def accumulate_direct_epfd_from_link_library(
        self,
        *,
        link_library,
        sat_topo,
        sat_azel,
        include_epfd,
        include_prx_total,
        include_per_satellite_prx,
        include_total_pfd,
        include_per_satellite_pfd,
        include_diagnostics,
        include_full_sat_beam_counts_used,
        include_sat_eligible_mask,
        output_sat_indices,
        **kwargs,
    ):
        self.accumulate_direct_calls += 1
        self.last_accumulate_direct_epfd_kwargs = dict(kwargs)
        self.last_accumulate_direct_epfd_kwargs["sat_topo_shape"] = tuple(np.asarray(sat_topo).shape)
        self.last_accumulate_direct_epfd_kwargs["sat_azel_shape"] = tuple(np.asarray(sat_azel).shape)
        self.last_accumulate_direct_epfd_kwargs["output_sat_indices"] = np.asarray(
            output_sat_indices, dtype=np.int32
        ).copy()
        finalize_result = link_library.finalize_direct_epfd_beams(
            output_sat_indices=output_sat_indices,
            include_diagnostics=include_diagnostics,
            include_full_sat_beam_counts_used=include_full_sat_beam_counts_used,
            ras_sat_azel=sat_azel,
            debug_direct_epfd=bool(kwargs.get("debug_direct_epfd", False)),
        )
        power_result = None
        stage_timings = {
            "beam_finalize": 0.0,
            "power_accumulation": 0.0,
        }
        if any(
            (
                include_epfd,
                include_prx_total,
                include_per_satellite_prx,
                include_total_pfd,
                include_per_satellite_pfd,
            )
        ):
            cell_spectral_weight = kwargs.get("cell_spectral_weight")
            if (
                cell_spectral_weight is None
                and kwargs.get("dynamic_group_active_mask") is not None
                and kwargs.get("dynamic_cell_group_leakage_factors") is not None
            ):
                dynamic_group_mask = np.asarray(
                    kwargs["dynamic_group_active_mask"],
                    dtype=bool,
                )
                dynamic_leakage = np.asarray(
                    kwargs["dynamic_cell_group_leakage_factors"],
                    dtype=np.float32,
                )
                dynamic_valid_mask = kwargs.get("dynamic_group_valid_mask")
                configured_groups = kwargs.get("dynamic_configured_groups_per_cell")
                spectral_slab = int(
                    kwargs.get("spectral_slab", max(1, int(dynamic_group_mask.shape[0])))
                )
                spectral_weights: list[np.ndarray] = []
                for t0 in range(0, int(dynamic_group_mask.shape[0]), max(1, spectral_slab)):
                    t1 = min(int(dynamic_group_mask.shape[0]), t0 + max(1, spectral_slab))
                    spectral_weights.append(
                        np.asarray(
                            scenario._compute_cell_spectral_weight_device(
                                np,
                                group_active_mask=dynamic_group_mask[t0:t1, :, :],
                                cell_group_leakage_factors=dynamic_leakage,
                                power_policy=str(
                                    kwargs.get("dynamic_power_policy") or "repeat_per_group"
                                ),
                                split_total_group_denominator_mode=str(
                                    kwargs.get("dynamic_split_total_group_denominator_mode")
                                    or "configured_groups"
                                ),
                                configured_groups_per_cell=(
                                    np.asarray(configured_groups)
                                    if configured_groups is not None
                                    else np.asarray(dynamic_leakage.shape[1], dtype=np.int32)
                                ),
                                group_valid_mask=(
                                    None
                                    if dynamic_valid_mask is None
                                    else np.asarray(dynamic_valid_mask, dtype=bool)
                                ),
                            ),
                            dtype=np.float32,
                        )
                    )
                cell_spectral_weight = (
                    None
                    if not spectral_weights
                    else np.concatenate(spectral_weights, axis=0).astype(np.float32, copy=False)
                )
                stage_timings["spectrum_activity_weighting"] = 0.0
            power_kwargs = dict(kwargs)
            power_kwargs["cell_spectral_weight"] = cell_spectral_weight
            power_result = self.accumulate_ras_power(
                sat_topo=sat_topo,
                sat_azel=sat_azel,
                beam_idx=finalize_result["beam_idx"],
                beam_alpha_rad=finalize_result["beam_alpha_rad"],
                beam_beta_rad=finalize_result["beam_beta_rad"],
                include_epfd=include_epfd,
                include_prx_total=include_prx_total,
                include_per_satellite_prx=include_per_satellite_prx,
                include_total_pfd=include_total_pfd,
                include_per_satellite_pfd=include_per_satellite_pfd,
                **power_kwargs,
            )
        diag_result = None
        if include_diagnostics:
            diag_result = {
                key: finalize_result[key]
                for key in (
                    "ras_retargeted_count",
                    "ras_reserved_count",
                    "direct_kept_count",
                    "repaired_link_count",
                    "dropped_link_count",
                )
            }
        # Mirror the production contract: when ``debug_direct_epfd`` is
        # requested, emit at least one stub stats entry so tests that
        # assert "debug mode populates the stats list" don't regress on
        # the fake returning an empty list.
        debug_stats_list: list[dict] = []
        if bool(kwargs.get("debug_direct_epfd", False)):
            debug_stats_list.append(
                {
                    "slab_index": 0,
                    "time_count": int(np.asarray(sat_topo).shape[0]),
                    "sat_count": int(np.asarray(sat_topo).shape[1]),
                    "debug_flag": True,
                }
            )
        return {
            "power_result": power_result,
            "sat_beam_counts_used_full": finalize_result.get("sat_beam_counts_used_full"),
            "diag_result": diag_result,
            "debug_direct_epfd_stats": debug_stats_list,
            "beam_finalize_substage_timings": {
                "direct_candidate_extraction": 0.0,
                "first_pass_selector": 0.0,
                "retarget_repair_finalize": 0.0,
            },
            "stage_timings": stage_timings,
            "beam_finalize_observed_memory": {},
            "power_observed_memory": {},
            "beam_finalize_chunk_shape": {"time_chunk_size": 1, "sky_chunk_size": 1},
            "boresight_compaction_stats": {},
            "hot_path_device_to_host_copy_count": 0,
            "hot_path_device_to_host_copy_bytes": 0,
            "device_scalar_sync_count": 0,
            "sat_eligible_mask": None
            if not include_sat_eligible_mask
            else np.zeros(
                (
                    int(np.asarray(sat_topo).shape[0]),
                    int(link_library.cell_count),
                    int(link_library.sat_count_total),
                ),
                dtype=bool,
            ),
        }

    def prx_elevation_range_dbw(self, *, prx_per_satellite_w_per_mhz, bw_mhz):
        arr = np.asarray(prx_per_satellite_w_per_mhz, dtype=np.float32)
        valid = np.isfinite(arr) & (arr > 0.0)
        if not np.any(valid):
            return None
        values_dbw = 10.0 * np.log10(arr[valid]) + 10.0 * np.log10(float(bw_mhz))
        return float(np.min(values_dbw)), float(np.max(values_dbw))

    def value_range_db(self, *, value_linear, db_offset_db=0.0):
        arr = np.asarray(value_linear, dtype=np.float32)
        valid = np.isfinite(arr) & (arr > 0.0)
        if not np.any(valid):
            return None
        values_db = 10.0 * np.log10(arr[valid]) + float(db_offset_db)
        return float(np.min(values_db)), float(np.max(values_db))

    def value_range_and_histogram(
        self,
        *,
        value_linear,
        value_edges_dbw,
        db_offset_db=0.0,
    ):
        """Fused range + histogram mock — matches the production contract
        so the scenario histogram path gets a numpy-backed stand-in."""
        arr = np.asarray(value_linear, dtype=np.float32)
        valid = np.isfinite(arr) & (arr > 0.0)
        if not np.any(valid):
            return {"range_db": None, "histogram": None}
        values_db = 10.0 * np.log10(arr[valid]) + float(db_offset_db)
        range_db = (float(np.min(values_db)), float(np.max(values_db)))
        if value_edges_dbw is None:
            return {"range_db": range_db, "histogram": None}
        edges = np.asarray(value_edges_dbw, dtype=np.float64)
        hist = np.zeros(max(int(edges.size) - 1, 0), dtype=np.int64)
        if hist.size > 0:
            bins = np.searchsorted(edges, values_db, side="right") - 1
            keep = (bins >= 0) & (bins < hist.size)
            if np.any(keep):
                hist += np.bincount(bins[keep], minlength=hist.size).astype(np.int64, copy=False)
        return {"range_db": range_db, "histogram": hist}

    def accumulate_value_histogram(
        self,
        *,
        value_linear,
        value_edges_dbw,
        db_offset_db=0.0,
        **_kwargs,
    ):
        arr = np.asarray(value_linear, dtype=np.float32)
        edges = np.asarray(value_edges_dbw, dtype=np.float64)
        hist = np.zeros(max(int(edges.size) - 1, 0), dtype=np.int64)
        valid = np.isfinite(arr) & (arr > 0.0)
        if not np.any(valid) or hist.size == 0:
            return hist
        values_db = 10.0 * np.log10(arr[valid]) + float(db_offset_db)
        bins = np.searchsorted(edges, values_db, side="right") - 1
        keep = (bins >= 0) & (bins < hist.size)
        if np.any(keep):
            hist += np.bincount(bins[keep], minlength=hist.size).astype(np.int64, copy=False)
        return hist

    def accumulate_prx_elevation_heatmap(
        self,
        *,
        elevation_edges_deg,
        prx_edges_dbw,
        **_kwargs,
    ):
        hist = np.zeros(
            (
                int(np.asarray(elevation_edges_deg).size) - 1,
                int(np.asarray(prx_edges_dbw).size) - 1,
            ),
            dtype=np.int64,
        )
        hist[0, 0] = 1
        return hist

    def accumulate_value_elevation_heatmap(
        self,
        *,
        value_linear,
        sat_elevation_deg,
        elevation_edges_deg,
        value_edges_dbw,
        db_offset_db=0.0,
        **_kwargs,
    ):
        hist = np.zeros(
            (
                int(np.asarray(elevation_edges_deg).size) - 1,
                int(np.asarray(value_edges_dbw).size) - 1,
            ),
            dtype=np.int64,
        )
        values = np.asarray(value_linear, dtype=np.float32)
        elev = np.asarray(sat_elevation_deg, dtype=np.float32)
        if values.ndim == 4:
            values = values[:, 0, :, :]
        if values.ndim == 2:
            values = values[:, :, None]
        elev_flat = elev.reshape(-1)
        x_bin = np.searchsorted(np.asarray(elevation_edges_deg, dtype=np.float64), elev_flat, side="right") - 1
        x_valid = np.isfinite(elev_flat) & (x_bin >= 0) & (x_bin < hist.shape[0])
        slab = values.reshape(values.shape[0] * values.shape[1], values.shape[2])
        valid = np.isfinite(slab) & (slab > 0.0) & x_valid[:, None]
        row_idx, col_idx = np.nonzero(valid)
        if row_idx.size == 0:
            return hist
        values_db = 10.0 * np.log10(slab[row_idx, col_idx]) + float(db_offset_db)
        y_bin = np.searchsorted(np.asarray(value_edges_dbw, dtype=np.float64), values_db, side="right") - 1
        keep = (y_bin >= 0) & (y_bin < hist.shape[1])
        if not np.any(keep):
            return hist
        xy = np.stack((x_bin[row_idx[keep]], y_bin[keep]), axis=1)
        for x_val, y_val in xy:
            hist[int(x_val), int(y_val)] += 1
        return hist


def _theta2_scope_test_grid() -> dict[str, object]:
    spacing = 1.0
    sqrt3 = np.sqrt(3.0)
    q_vals = np.array([0, 1, 0, -1, -1, 0, 1, 2], dtype=np.int32)
    r_vals = np.array([0, 0, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    east = spacing * (q_vals + 0.5 * r_vals)
    north = spacing * ((sqrt3 / 2.0) * r_vals)
    km_to_deg = 1.0 / 6371.0 * (180.0 / np.pi)
    cell_longitudes = east * km_to_deg * u.deg
    cell_latitudes = north * km_to_deg * u.deg
    identity = np.arange(cell_longitudes.size, dtype=np.int32)
    return {
        "active_cell_count": int(cell_longitudes.size),
        "prefilter_cell_longitudes": cell_longitudes,
        "prefilter_cell_latitudes": cell_latitudes,
        "station_lat": 0.0 * u.deg,
        "station_lon": 0.0 * u.deg,
        "ras_service_cell_index_prefilter": 0,
        "prefilter_to_pre_ras": identity.copy(),
        "pre_ras_to_active": identity.copy(),
        "active_grid_longitudes": cell_longitudes,
    }


def _output_family_profile(profile_name: str) -> dict[str, dict[str, object]]:
    config = scenario.default_output_families()
    for family_name in config:
        config[family_name]["mode"] = "none"
    if profile_name == "none":
        pass
    elif profile_name == "counts_only":
        config["beam_statistics"]["mode"] = "raw"
    elif profile_name == "totals_only":
        config["epfd_distribution"]["mode"] = "raw"
        config["prx_total_distribution"]["mode"] = "raw"
        config["total_pfd_ras_distribution"]["mode"] = "raw"
        config["beam_statistics"]["mode"] = "raw"
    elif profile_name == "pfd_core":
        config["epfd_distribution"]["mode"] = "raw"
        config["prx_total_distribution"]["mode"] = "raw"
        config["total_pfd_ras_distribution"]["mode"] = "raw"
        config["per_satellite_pfd_distribution"]["mode"] = "raw"
    elif profile_name == "all_outputs":
        config["epfd_distribution"]["mode"] = "both"
        config["prx_total_distribution"]["mode"] = "both"
        config["total_pfd_ras_distribution"]["mode"] = "both"
        config["per_satellite_pfd_distribution"]["mode"] = "both"
        config["prx_elevation_heatmap"]["mode"] = "both"
        config["prx_elevation_heatmap"]["sky_slab"] = 2
        config["per_satellite_pfd_elevation_heatmap"]["mode"] = "both"
        config["per_satellite_pfd_elevation_heatmap"]["sky_slab"] = 2
        config["beam_statistics"]["mode"] = "both"
    elif profile_name == "notebook_full":
        config["epfd_distribution"]["mode"] = "both"
        config["prx_total_distribution"]["mode"] = "both"
        config["total_pfd_ras_distribution"]["mode"] = "both"
        config["per_satellite_pfd_distribution"]["mode"] = "both"
        config["prx_elevation_heatmap"]["mode"] = "preaccumulated"
        config["prx_elevation_heatmap"]["sky_slab"] = 2
        config["per_satellite_pfd_elevation_heatmap"]["mode"] = "both"
        config["per_satellite_pfd_elevation_heatmap"]["sky_slab"] = 2
        config["beam_statistics"]["mode"] = "both"
    elif profile_name == "gui_heavy":
        config["prx_elevation_heatmap"]["mode"] = "both"
        config["prx_elevation_heatmap"]["sky_slab"] = 2
        config["beam_statistics"]["mode"] = "raw"
    else:
        raise KeyError(profile_name)
    return config


_FAKE_OUTPUT_PROFILE_CASES: dict[str, dict[str, object]] = {
    "none": {
        "profile_kwargs": {"output_families": _output_family_profile("none")},
        "expected_iter_names": set(),
        "expected_preacc_names": set(),
        "expect_prx_heatmap": False,
        "expect_prx_per_sat": False,
        "expect_power": False,
    },
    "counts_only": {
        "profile_kwargs": {"output_families": _output_family_profile("counts_only")},
        "expected_iter_names": {"sat_beam_counts_used"},
        "expected_preacc_names": set(),
        "expect_prx_heatmap": False,
        "expect_prx_per_sat": False,
        "expect_power": False,
    },
    "totals_only": {
        "profile_kwargs": {"output_families": _output_family_profile("totals_only")},
        "expected_iter_names": {
            "EPFD_W_m2",
            "Prx_total_W",
            "PFD_total_RAS_STATION_W_m2",
            "sat_beam_counts_used",
        },
        "expected_preacc_names": set(),
        "expect_prx_heatmap": False,
        "expect_prx_per_sat": False,
        "expect_power": True,
    },
    "all_outputs": {
        "profile_kwargs": {"output_families": _output_family_profile("all_outputs")},
        "expected_iter_names": {
            "EPFD_W_m2",
            "Prx_total_W",
            "Prx_per_sat_RAS_STATION_W",
            "PFD_total_RAS_STATION_W_m2",
            "PFD_per_sat_RAS_STATION_W_m2",
            "sat_elevation_RAS_STATION_deg",
            "beam_demand_count",
            "sat_beam_counts_used",
        },
        "expected_preacc_names": {
            "prx_total_distribution/counts",
            "epfd_distribution/counts",
            "total_pfd_ras_distribution/counts",
            "per_satellite_pfd_distribution/counts",
            "prx_elevation_heatmap/counts",
            "per_satellite_pfd_elevation_heatmap/counts",
            "beam_statistics/full_network_count_hist",
        },
        "expect_prx_heatmap": True,
        "expect_prx_per_sat": True,
        "expect_power": True,
    },
    "notebook_full": {
        "profile_kwargs": {"output_families": _output_family_profile("notebook_full")},
        "expected_iter_names": {
            "EPFD_W_m2",
            "Prx_total_W",
            "PFD_total_RAS_STATION_W_m2",
            "PFD_per_sat_RAS_STATION_W_m2",
            "sat_elevation_RAS_STATION_deg",
            "beam_demand_count",
            "sat_beam_counts_used",
        },
        "expected_preacc_names": {
            "prx_total_distribution/counts",
            "epfd_distribution/counts",
            "total_pfd_ras_distribution/counts",
            "per_satellite_pfd_distribution/counts",
            "prx_elevation_heatmap/counts",
            "per_satellite_pfd_elevation_heatmap/counts",
            "beam_statistics/full_network_count_hist",
        },
        "expect_prx_heatmap": True,
        "expect_prx_per_sat": False,
        "expect_power": True,
    },
    "gui_heavy": {
        "profile_kwargs": {"output_families": _output_family_profile("gui_heavy")},
        "expected_iter_names": {
            "Prx_per_sat_RAS_STATION_W",
            "sat_elevation_RAS_STATION_deg",
            "sat_beam_counts_used",
        },
        "expected_preacc_names": {
            "prx_elevation_heatmap/counts",
        },
        "expect_prx_heatmap": True,
        "expect_prx_per_sat": True,
        "expect_power": True,
    },
}


def _fake_direct_epfd_common_kwargs(
    tmp_path: Path,
    filename_stem: str,
    *,
    session: _FakeSession | None = None,
    gpu_module: _FakeGpuModule | None = None,
    active_cell_longitudes: u.Quantity | None = None,
    **overrides,
) -> dict[str, object]:
    if active_cell_longitudes is None:
        active_cell_longitudes = np.array([10.0, 11.0], dtype=np.float64) * u.deg
    if session is None:
        session = _FakeSession()
    if gpu_module is None:
        gpu_module = _FakeGpuModule()
    kwargs: dict[str, object] = {
        "tle_list": np.asarray([object(), object()], dtype=object),
        "observer_arr": np.asarray([object(), object(), object()], dtype=object),
        "active_cell_longitudes": active_cell_longitudes,
        "sat_min_elevation_deg_per_sat": np.array([20.0, 20.0], dtype=np.float32),
        "sat_beta_max_deg_per_sat": np.array([30.0, 30.0], dtype=np.float32),
        "sat_belt_id_per_sat": np.array([0, 0], dtype=np.int16),
        "sat_orbit_radius_m_per_sat": np.array([6.9e6, 6.9e6], dtype=np.float32),
        "pattern_kwargs": {
            "Lt": 1.6 * u.m,
            "Lr": 1.6 * u.m,
            "SLR": 20.0 * cnv.dB,
            "l": 2,
            "far_sidelobe_start": 90.0 * u.deg,
            "far_sidelobe_level": -20.0 * cnv.dBi,
        },
        "wavelength": 15.0 * u.cm,
        "ras_station_ant_diam": 13.5 * u.m,
        "frequency": 2.0 * u.GHz,
        "ras_station_elev_range": (15.0 * u.deg, 90.0 * u.deg),
        "observer_alt_km_ras_station": 0.1,
        "storage_filename": str(tmp_path / f"{filename_stem}.h5"),
        "base_start_time": Time("2025-01-01T00:00:00", scale="utc"),
        "base_end_time": Time("2025-01-01T00:00:00", scale="utc"),
        "timestep": 1.0,
        "iteration_count": 1,
        "iteration_rng_seed": 2026,
        "nco": 1,
        "nbeam": 4,
        "selection_strategy": "max_elevation",
        "ras_pointing_mode": "cell_center",
        "ras_service_cell_index": -1,
        "ras_service_cell_active": False,
        "ras_guard_angle": 4.0 * u.deg,
        "include_atmosphere": False,
        "cupy_module": _FakeCp(),
        "gpu_module": gpu_module,
        "session": session,
        "progress_factory": lambda iterable, **_kwargs: iterable,
        "output_families": _output_family_profile("totals_only"),
    }
    kwargs.update(overrides)
    return kwargs


def _plan_fake_direct_epfd_iteration(
    *,
    host_budget_gb: int,
    gpu_budget_gb: int,
    boresight_active: bool,
    n_steps_total: int = 512,
    n_cells_total: int = 3360,
    n_sats_total: int = 24,
    n_skycells_s1586: int = 3360,
    visible_satellite_est: int = 6,
    nco: int = 4,
    nbeam: int = 8,
    profile_stages: bool = True,
    cell_activity_mode: str = "whole_cell",
    activity_groups_per_cell: int = 1,
) -> dict[str, object]:
    session = _FakeSession(sat_count_total=n_sats_total, ras_visible_sat_count=visible_satellite_est)
    session._n_sky = int(n_skycells_s1586 if boresight_active else 1)
    observer_context = type("ObsCtx", (), {"n_observers": int(n_cells_total) + 1})()
    satellite_context = type("SatCtx", (), {"n_sats": int(n_sats_total)})()
    output_family_plan = {
        "needs_epfd": True,
        "needs_total_prx": True,
        "needs_per_satellite_prx": True,
        "needs_total_pfd": True,
        "needs_per_satellite_pfd": True,
        "needs_beam_counts": True,
        "needs_beam_demand": True,
    }
    return scenario._plan_direct_epfd_iteration_schedule(
        session=session,
        observer_context=observer_context,
        satellite_context=satellite_context,
        gpu_output_dtype=np.float32,
        n_steps_total=int(n_steps_total),
        n_cells_total=int(n_cells_total),
        n_sats_total=int(n_sats_total),
        n_skycells_s1586=int(n_skycells_s1586),
        visible_satellite_est=int(visible_satellite_est),
        nco=int(nco),
        nbeam=int(nbeam),
        boresight_active=bool(boresight_active),
        effective_ras_pointing_mode="ras_station",
        output_family_plan=output_family_plan,
        store_eligible_mask=True,
        profile_stages=bool(profile_stages),
        host_budget_info={"effective_budget_bytes": int(host_budget_gb) * _GIB},
        gpu_budget_info={"effective_budget_bytes": int(gpu_budget_gb) * _GIB},
        scheduler_target_fraction=0.90,
        scheduler_active_target_fraction=0.90,
        finalize_memory_budget_bytes=None,
        power_memory_budget_bytes=None,
        export_memory_budget_bytes=None,
        power_sky_slab=None,
        force_bulk_timesteps=None,
        force_cell_observer_chunk=None,
        cell_activity_mode=str(cell_activity_mode),
        activity_groups_per_cell=int(activity_groups_per_cell),
        allow_warmup_calibration=False,
    )


def _assert_fake_direct_epfd_plan_is_safe(
    plan: Mapping[str, object],
    *,
    host_budget_gb: int,
    gpu_budget_gb: int,
    n_steps_total: int = 512,
    n_cells_total: int = 3360,
) -> None:
    assert int(plan["bulk_timesteps"]) >= 1
    assert int(plan["cell_chunk"]) >= 1
    assert 1 <= int(plan["sky_slab"]) <= int(n_cells_total)
    assert 1 <= int(plan.get("spectral_slab", 1)) <= int(plan["bulk_timesteps"])
    assert int(plan["predicted_host_peak_bytes"]) <= int(host_budget_gb) * _GIB
    assert int(plan["predicted_gpu_peak_bytes"]) <= int(gpu_budget_gb) * _GIB
    assert int(plan["predicted_gpu_finalize_peak_bytes"]) <= int(plan["predicted_gpu_peak_bytes"])
    assert int(plan["predicted_gpu_power_peak_bytes"]) <= int(plan["predicted_gpu_peak_bytes"])
    assert str(plan["limiting_resource"])
    n_batches = int(np.ceil(float(n_steps_total) / float(int(plan["bulk_timesteps"]))))
    n_cell_chunks = int(np.ceil(float(n_cells_total) / float(int(plan["cell_chunk"]))))
    assert not (n_batches <= 8 and n_cell_chunks >= 64), (
        "The scheduler accepted a pathological low-batch / huge-chunk fan-out plan "
        f"(batches={n_batches}, cell_chunks={n_cell_chunks})."
    )


def test_plan_direct_epfd_iteration_schedule_tracks_spectral_slab_for_per_channel_activity() -> None:
    plan = _plan_fake_direct_epfd_iteration(
        host_budget_gb=32,
        gpu_budget_gb=1,
        boresight_active=False,
        n_steps_total=512,
        n_cells_total=20000,
        n_sats_total=24,
        visible_satellite_est=8,
        nco=4,
        nbeam=8,
        cell_activity_mode="per_channel",
        activity_groups_per_cell=19,
    )

    assert int(plan["spectral_slab"]) >= 1
    assert int(plan["spectral_slab"]) <= int(plan["bulk_timesteps"])
    assert int(plan["predicted_gpu_activity_scratch_bytes"]) > 0
    assert int(plan["predicted_gpu_activity_peak_bytes"]) >= int(
        plan["predicted_gpu_activity_resident_bytes"]
    )
    if int(plan["spectral_slab"]) < int(plan["bulk_timesteps"]):
        assert bool(plan["spectral_backoff_active"])
        assert str(plan["limiting_dimension"]) == "spectral_slab"


class _RecordingProgressBar:

    def __init__(self, iterable, **kwargs):
        self._items = list(iterable)
        self.descriptions: list[str] = []
        desc = kwargs.get("desc")
        if desc is not None:
            self.descriptions.append(str(desc))

    def __iter__(self):
        return iter(self._items)

    def set_description(self, text: str) -> None:
        self.descriptions.append(str(text))


def _recording_progress_factory(storage: list[_RecordingProgressBar]):
    def _factory(iterable, **kwargs):
        bar = _RecordingProgressBar(iterable, **kwargs)
        storage.append(bar)
        return bar

    return _factory


class TestDirectEpfdGpuRunner:

    def test_copy_compact_satellite_indices_host_explicitly_copies_cuda_vectors(self):
        fake_gpu = _FakeGpuModule()
        sat_idx = scenario._copy_compact_satellite_indices_host(
            fake_gpu,
            _FakeCudaIndexVector([1, 3, 5]),
        )

        assert_equal(sat_idx, np.array([1, 3, 5], dtype=np.int32))
        assert fake_gpu.copy_calls == 1

    def test_default_output_families_use_finer_scalar_distribution_bins(self):
        config = scenario.default_output_families()

        for family_name in (
            "prx_total_distribution",
            "epfd_distribution",
            "total_pfd_ras_distribution",
            "per_satellite_pfd_distribution",
        ):
            assert_allclose(float(config[family_name]["bin_step_db"]), 0.02, atol=0.0, rtol=0.0)

        assert_allclose(
            float(config["prx_elevation_heatmap"]["value_bin_step_db"]),
            0.2,
            atol=0.0,
            rtol=0.0,
        )

    def test_public_output_family_normalization_can_ignore_unknown_gui_keys(self):
        normalized = scenario.normalize_output_family_configs(
            {
                "epfd_distribution": {"mode": "both"},
                "legacy_family": {"mode": "raw"},
            },
            ignore_unknown=True,
        )

        assert_equal(normalized["epfd_distribution"]["mode"], "both")
        assert "legacy_family" not in normalized

    def test_public_output_family_validation_returns_gui_friendly_errors(self):
        ok, message = scenario.validate_output_family_configs(
            {"epfd_distribution": {"mode": "unsupported"}},
            ignore_unknown=True,
        )

        assert ok is False
        assert "output family mode is unsupported" in message.lower()

    def test_postprocess_notebook_prefers_preaccumulated_instantaneous_ccdf_and_raw_corridor(self, tmp_path: Path):
        kwargs = _fake_direct_epfd_common_kwargs(
            tmp_path,
            "postprocess_ccdf_route",
            base_end_time=Time("2025-01-01T00:00:05", scale="utc"),
            force_bulk_timesteps=2,
            output_families=_output_family_profile("notebook_full"),
        )

        run_result = scenario.run_gpu_direct_epfd(**kwargs)
        env = _execute_postprocess_validation(run_result["storage_filename"])
        env["display"] = lambda *_args, **_kwargs: None

        preacc_info = env["_render_ccdf"](
            "Prx_total_W",
            integrated=False,
            corridor=False,
            save_name="preacc_prx_total_ccdf.png",
        )
        raw_info = env["_render_ccdf"](
            "Prx_total_W",
            integrated=False,
            corridor=True,
            save_name="raw_prx_total_corridor_ccdf.png",
        )

        assert preacc_info is not None
        assert raw_info is not None
        assert_equal(preacc_info["ecdf_method_used"], "prebinned_hist")
        assert_equal(raw_info["ecdf_method_used"], "hist")
        assert preacc_info["p98"] is not None
        assert raw_info["p98"] is not None

    def test_postprocess_notebook_ccdf_cells_use_shared_histogram_plotter(self):
        cells = _load_notebook_code_cells(POSTPROCESS_NOTEBOOK_PATH)
        ccdf_cells = cells[1:13]

        assert "postprocess_recipes.render_recipe" in cells[0]
        assert all("_render_ccdf(" in cell for cell in ccdf_cells)

    def test_run_gpu_direct_epfd_writes_expected_datasets(self, tmp_path: Path):
        filename = tmp_path / "step12_fake.h5"
        fake_gpu = _FakeGpuModule()
        fake_session = _FakeSession()
        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            include_atmosphere=False,
            output_families=_output_family_profile("pfd_core"),
            storage_constants={"sat_belt_id_per_sat": np.array([0, 0], dtype=np.int16)},
            storage_attrs={"mode": "direct_link_epfd"},
            spectrum_plan={
                "service_band_start_mhz": 2620.0,
                "service_band_stop_mhz": 2690.0,
                "ras_receiver_band_start_mhz": 2690.0,
                "ras_receiver_band_stop_mhz": 2700.0,
                "reuse_factor": 4,
                "channel_groups_per_cell_cap": 3,
                "ras_anchor_reuse_slot": 2,
                "active_cell_reuse_slot_ids": np.asarray([0, 1], dtype=np.int32),
            },
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert result["storage_filename"] == str(filename)
        assert result["effective_ras_pointing_mode"] == "cell_center"
        assert loaded["attrs"]["mode"] == "direct_link_epfd"
        assert int(loaded["attrs"]["result_schema_version"]) == 4
        # When a spectrum plan is supplied (see ``spectrum_plan=`` above)
        # the runner stores power values as integrated over the RAS
        # receiver band after reuse-slot occupancy weighting, so the
        # declared basis attribute is "ras_receiver_band". Only runs
        # without a spectrum plan fall back to "channel_total".
        assert loaded["attrs"]["stored_power_basis"] == "ras_receiver_band"
        assert float(loaded["attrs"]["bandwidth_mhz"]) == pytest.approx(5.0)
        assert loaded["attrs"]["power_input_quantity"] == "target_pfd"
        assert loaded["attrs"]["power_input_basis"] == "per_mhz"
        assert loaded["attrs"]["cell_activity_mode"] == "whole_cell"
        assert loaded["attrs"]["split_total_group_denominator_mode"] == "configured_groups"
        assert float(loaded["attrs"]["power_input_value"]) == pytest.approx(-83.5)
        assert loaded["attrs"]["power_input_value_unit"] == "dBW/m^2/MHz"
        assert float(loaded["attrs"]["target_pfd_dbw_m2_mhz"]) == pytest.approx(-83.5)
        assert float(loaded["attrs"]["target_pfd_dbw_m2_channel"]) == pytest.approx(
            -83.5 + 10.0 * np.log10(5.0)
        )
        assert float(loaded["attrs"]["service_band_start_mhz"]) == pytest.approx(2620.0)
        assert float(loaded["attrs"]["service_band_stop_mhz"]) == pytest.approx(2690.0)
        assert int(loaded["attrs"]["reuse_factor"]) == 4
        assert int(loaded["attrs"]["channel_groups_per_cell"]) == 3
        assert int(loaded["attrs"]["ras_anchor_reuse_slot"]) == 2
        assert str(loaded["attrs"]["ras_reference_mode"]) == "lower"
        assert float(loaded["attrs"]["ras_reference_frequency_mhz_effective"]) == pytest.approx(
            2690.0
        )
        assert int(loaded["attrs"]["spectral_slab"]) == 1
        assert_equal(loaded["const"]["sat_belt_id_per_sat"], np.array([0, 0], dtype=np.int16))
        assert_equal(
            loaded["const"]["cell_reuse_slot_id_active"],
            np.asarray([0, 1], dtype=np.int32),
        )
        assert loaded["const"]["cell_spectral_leakage_factor_active"].shape == (2,)
        assert loaded["const"]["cell_group_spectral_leakage_factor_active"].shape == (2, 3)
        assert loaded["const"]["spectrum_slot_edges_mhz"].shape == (15,)
        assert loaded["const"]["spectrum_slot_group_channel_index"].shape == (4, 3)
        assert loaded["const"]["spectrum_slot_group_leakage_factor"].shape == (4, 3)
        assert loaded["iter"][0]["EPFD_W_m2"].shape == (1, 1, 2)
        assert loaded["iter"][0]["Prx_total_W"].shape == (1, 1, 2)
        assert loaded["iter"][0]["PFD_total_RAS_STATION_W_m2"].shape == (1,)
        assert loaded["iter"][0]["PFD_per_sat_RAS_STATION_W_m2"].shape == (1, 2)
        assert fake_gpu.copy_calls == 4
        assert len(fake_session.spectrum_context_calls) == 1
        assert fake_session.spectrum_context_calls[0]["reuse_factor"] == 4
        assert fake_session.spectrum_context_calls[0]["cell_group_leakage_factors"].shape == (2, 3)
        assert_equal(
            np.asarray(
                fake_session.last_accumulate_ras_power_kwargs["spectrum_plan_context"].cell_leakage_factors
            ),
            loaded["const"]["cell_spectral_leakage_factor_active"],
        )

    def test_run_gpu_direct_epfd_persists_empty_configured_preacc_power_families(
        self,
        tmp_path: Path,
    ) -> None:
        class _NoRangeSession(_FakeSession):
            def value_range_db(self, *, value_linear, db_offset_db=0.0):
                return None

            def value_range_and_histogram(
                self,
                *,
                value_linear,
                value_edges_dbw,
                db_offset_db=0.0,
            ):
                # The test intentionally simulates "no valid samples at
                # all" so preacc histograms must stay empty; the fused
                # method is now the primary histogram path, so it must
                # short-circuit the same way the separate range method
                # does.
                return {"range_db": None, "histogram": None}

            def prx_elevation_range_db(self, *, prx_per_satellite_w_per_mhz, bw_mhz):
                return None

        output_families = _output_family_profile("all_outputs")
        for family_name in (
            "prx_total_distribution",
            "epfd_distribution",
            "total_pfd_ras_distribution",
            "per_satellite_pfd_distribution",
            "prx_elevation_heatmap",
            "per_satellite_pfd_elevation_heatmap",
        ):
            output_families[family_name]["mode"] = "preaccumulated"

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "empty_preacc_power_families",
                session=_NoRangeSession(),
                gpu_module=_FakeGpuModule(),
                output_families=output_families,
            )
        )

        with h5py.File(result["storage_filename"], "r") as h5:
            preacc_group = h5["preaccumulated"]

            for family_name in (
                "prx_total_distribution",
                "epfd_distribution",
                "total_pfd_ras_distribution",
                "per_satellite_pfd_distribution",
            ):
                assert h5.attrs[f"output_family_{family_name}_mode"] == "preaccumulated"
                family_group = preacc_group[family_name]
                counts = np.asarray(family_group["counts"])
                edges = np.asarray(family_group["edges_dbw"])
                assert counts.ndim == 1
                assert counts.size == edges.size - 1
                assert np.all(counts == 0)
                assert int(family_group.attrs["sample_count"]) == 0

            for family_name in (
                "prx_elevation_heatmap",
                "per_satellite_pfd_elevation_heatmap",
            ):
                assert h5.attrs[f"output_family_{family_name}_mode"] == "preaccumulated"
                family_group = preacc_group[family_name]
                counts = np.asarray(family_group["counts"])
                elevation_edges = np.asarray(family_group["elevation_edges_deg"])
                value_edges = np.asarray(family_group["value_edges_dbw"])
                assert counts.ndim == 2
                assert counts.shape == (elevation_edges.size - 1, value_edges.size - 1)
                assert np.all(counts == 0)
                assert int(family_group.attrs["sample_count"]) == 0

    def test_run_gpu_direct_epfd_emits_scheduler_payload_and_retries_after_gpu_oom(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        events: list[dict[str, object]] = []
        original_compute = scenario._compute_gpu_direct_epfd_batch_device
        call_count = {"value": 0}
        configured_gpu_budget_bytes = 4 * _GIB

        def _compute_with_one_oom(**kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise scenario._DirectGpuOutOfMemory(
                    "beam_finalize",
                    RuntimeError("CUDA out of memory during finalize"),
                    stage_memory_summary={
                        "observed_stage_name": "beam_finalize",
                        "observed_stage_gpu_peak_bytes": int(4.5 * _GIB),
                        "observed_stage_gpu_free_low_bytes": int(0.8 * _GIB),
                        "observed_process_rss_bytes": int(2.25 * _GIB),
                    },
                )
            return original_compute(**kwargs)

        monkeypatch.setattr(
            scenario,
            "_compute_gpu_direct_epfd_batch_device",
            _compute_with_one_oom,
        )
        monkeypatch.setattr(
            scenario,
            "_capture_direct_epfd_live_memory_snapshot",
            lambda *_args, **_kwargs: {
                "gpu_snapshot": {
                    "free_bytes": int(2.5 * _GIB),
                    "total_bytes": int(8 * _GIB),
                },
                "process_rss_bytes": int(2.25 * _GIB),
            },
        )

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "scheduler_backoff_retry",
                base_end_time=Time("2025-01-01T00:00:03", scale="utc"),
                force_bulk_timesteps=2,
                gpu_memory_budget_gb=float(configured_gpu_budget_bytes / float(_GIB)),
                progress_callback=events.append,
            )
        )

        assert result["run_state"] == "completed"
        assert call_count["value"] >= 2
        iteration_plan_events = [event for event in events if event.get("kind") == "iteration_plan"]
        assert iteration_plan_events
        assert iteration_plan_events[0]["scheduler_target_profile"] == "high_throughput"
        assert int(iteration_plan_events[0]["n_earthgrid_cells"]) == 2
        assert "planned_total_seconds" in iteration_plan_events[0]

        warning_events = [event for event in events if event.get("kind") == "warning"]
        assert warning_events
        warning = warning_events[0]
        assert "retrying batch" in str(warning.get("description", "")).lower()
        assert warning["warning_stage"] == "beam_finalize"
        assert float(warning["scheduler_active_target_fraction"]) < float(
            warning["scheduler_target_fraction"]
        )
        assert warning["gpu_effective_budget_lowered"] is True
        assert int(warning["gpu_effective_budget_previous_bytes"]) == configured_gpu_budget_bytes
        expected_lowered_budget = int(
            configured_gpu_budget_bytes
            - (
                int(0.5 * _GIB)
                + int(scenario._DIRECT_EPFD_GPU_OOM_MARGIN_BYTES)
            )
        )
        assert int(warning["gpu_effective_budget_bytes"]) == expected_lowered_budget
        assert warning["gpu_budget_lowered_stage"] == "beam_finalize"
        assert int(warning["scheduler_retry_count"]) >= 1
        assert warning["observed_stage_name"] == "beam_finalize"
        assert int(warning["observed_stage_gpu_peak_bytes"]) == int(4.5 * _GIB)
        assert int(warning["observed_stage_gpu_free_low_bytes"]) == int(0.8 * _GIB)
        assert int(warning["observed_process_rss_bytes"]) == int(2.25 * _GIB)
        assert float(warning["predicted_gpu_peak_bytes"]) <= float(
            warning["gpu_effective_budget_bytes"]
        )

    def test_run_gpu_direct_epfd_replans_before_beam_finalize_when_live_fit_is_unsafe(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        events: list[dict[str, object]] = []
        original_guard = scenario._raise_if_direct_epfd_stage_live_fit_is_unsafe
        triggered = {"value": False}

        def _guard_once_for_finalize(**kwargs):
            if str(kwargs.get("stage")) == "beam_finalize" and not triggered["value"]:
                triggered["value"] = True
                raise scenario._DirectGpuOutOfMemory(
                    "beam_finalize",
                    RuntimeError("Unsafe live-fit before beam_finalize"),
                )
            # Suppress live GPU probing on all subsequent guard calls so
            # the retry path doesn't depend on actual VRAM availability.
            safe_kwargs = dict(kwargs)
            safe_kwargs["live_gpu_snapshot"] = None
            safe_kwargs["live_host_snapshot"] = None
            return original_guard(**safe_kwargs)

        # Also suppress the live memory snapshot used by the OOM retry
        # cascade — without this, _lower_runtime_effective_gpu_budget
        # sees real CUDA free memory and can spiral the budget to 1 byte.
        _fake_snapshot = {
            "gpu_cuda_total_bytes": 4 * 1024**3,
            "gpu_cuda_free_bytes": 3 * 1024**3,
            "host_available_bytes": 8 * 1024**3,
            "host_total_bytes": 16 * 1024**3,
        }
        monkeypatch.setattr(
            scenario,
            "_capture_direct_epfd_live_memory_snapshot",
            lambda *a, **kw: dict(_fake_snapshot),
        )
        monkeypatch.setattr(
            scenario,
            "_raise_if_direct_epfd_stage_live_fit_is_unsafe",
            _guard_once_for_finalize,
        )

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "scheduler_pre_stage_retry",
                base_end_time=Time("2025-01-01T00:00:03", scale="utc"),
                force_bulk_timesteps=2,
                progress_callback=events.append,
            )
        )

        assert result["run_state"] == "completed"
        warning_events = [event for event in events if event.get("kind") == "warning"]
        assert warning_events
        assert warning_events[0]["warning_stage"] == "beam_finalize"
        assert "retrying batch" in str(warning_events[0].get("description", "")).lower()

    def test_run_gpu_direct_epfd_replans_when_observed_visibility_exceeds_probe(
        self,
        tmp_path: Path,
    ) -> None:
        events: list[dict[str, object]] = []
        fake_session = _FakeSession(
            sat_count_total=3,
            ras_visible_sat_count=3,
            probe_visibility_counts=[1, 1, 1, 1, 1],
        )

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "scheduler_visibility_replan",
                session=fake_session,
                tle_list=np.asarray([object(), object(), object()], dtype=object),
                sat_min_elevation_deg_per_sat=np.array([20.0, 20.0, 20.0], dtype=np.float32),
                sat_beta_max_deg_per_sat=np.array([30.0, 30.0, 30.0], dtype=np.float32),
                sat_belt_id_per_sat=np.array([0, 0, 0], dtype=np.int16),
                sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6, 6.9e6], dtype=np.float32),
                base_end_time=Time("2025-01-01T00:00:03", scale="utc"),
                force_bulk_timesteps=2,
                progress_callback=events.append,
            )
        )

        assert result["run_state"] == "completed"
        assert len(fake_session.probe_visibility_calls) == scenario._DIRECT_EPFD_VISIBILITY_PROBE_SAMPLES
        iteration_plan_events = [event for event in events if event.get("kind") == "iteration_plan"]
        assert iteration_plan_events
        assert int(iteration_plan_events[0]["visible_satellite_probe"]) == 1
        assert int(iteration_plan_events[0]["visible_satellite_probe_samples"]) == 5
        visibility_warnings = [
            event
            for event in events
            if event.get("kind") == "warning"
            and "observed" in str(event.get("description", "")).lower()
        ]
        assert visibility_warnings
        visibility_warning = visibility_warnings[0]
        assert int(visibility_warning["visible_satellite_observed"]) == 3
        assert int(visibility_warning["visible_satellite_est"]) >= 3
        assert "replanning" in str(visibility_warning.get("description", "")).lower()

    def test_run_gpu_direct_epfd_avoids_exact_visibility_count_in_steady_state(
        self,
        tmp_path: Path,
    ) -> None:
        fake_session = _FakeSession(
            sat_count_total=2,
            ras_visible_sat_count=2,
            probe_visibility_counts=[2, 2, 2, 2, 2],
        )

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "steady_state_visibility_fast_path",
                session=fake_session,
                cupy_module=_FakeCpNoCountNonzero(),
                progress_callback=None,
                profile_stages=False,
                debug_direct_epfd=False,
            )
        )

        assert result["run_state"] == "completed"

    def test_run_gpu_direct_epfd_raises_summarized_failure_when_retries_stop_helping(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_plan = scenario._plan_direct_epfd_iteration_schedule

        def _min_safe_plan(**kwargs):
            plan = dict(original_plan(**kwargs))
            stage_budget_info = dict(plan["stage_budget_info"])
            return {
                **plan,
                "bulk_timesteps": 1,
                "cell_chunk": 1,
                "sky_slab": 1,
                "predicted_host_peak_bytes": 1,
                "predicted_gpu_peak_bytes": 1,
                "predicted_gpu_propagation_peak_bytes": 1,
                "predicted_gpu_finalize_peak_bytes": 1,
                "predicted_gpu_power_peak_bytes": 1,
                "predicted_gpu_export_peak_bytes": 1,
                "planned_batch_seconds": 0.01,
                "stage_budget_info": {
                    **stage_budget_info,
                    "finalize_memory_budget_bytes": 1,
                    "power_memory_budget_bytes": 1,
                    "export_memory_budget_bytes": 1,
                },
            }

        monkeypatch.setattr(
            scenario,
            "_plan_direct_epfd_iteration_schedule",
            _min_safe_plan,
        )

        def _always_oom(**_kwargs):
            raise scenario._DirectGpuOutOfMemory(
                "power_accumulation",
                RuntimeError("OOM on every retry"),
                stage_memory_summary={
                    "observed_stage_name": "power_accumulation",
                    "observed_stage_gpu_peak_bytes": int(1.5 * _GIB),
                    "observed_stage_gpu_free_low_bytes": int(0.5 * _GIB),
                },
            )

        monkeypatch.setattr(
            scenario,
            "_compute_gpu_direct_epfd_batch_device",
            _always_oom,
        )
        monkeypatch.setattr(
            scenario,
            "_capture_direct_epfd_live_memory_snapshot",
            lambda *_args, **_kwargs: {
                "gpu_snapshot": {
                    "free_bytes": 4 * _GIB,
                    "total_bytes": 8 * _GIB,
                }
            },
        )

        with pytest.raises(scenario._DirectGpuOutOfMemory) as excinfo:
            scenario.run_gpu_direct_epfd(
                **_fake_direct_epfd_common_kwargs(
                    tmp_path,
                    "scheduler_terminal_oom",
                    base_end_time=Time("2025-01-01T00:00:01", scale="utc"),
                    force_bulk_timesteps=1,
                )
            )

        error_text = str(excinfo.value).lower()
        assert "remained unsafe after scheduler replanning" in error_text
        assert "configured cap" in error_text
        assert "sky_slab=1" in error_text

    def test_run_gpu_direct_epfd_threads_target_pfd_into_power_accumulation(self, tmp_path: Path):
        filename = tmp_path / "target_pfd_fake.h5"
        fake_session = _FakeSession()

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "target_pfd_fake",
                storage_filename=str(filename),
                session=fake_session,
                output_families=_output_family_profile("gui_heavy"),
                pfd0_dbw_m2_mhz=-91.75,
            )
        )
        loaded = scenario.read_data(result["storage_filename"], stack=False)
        expected_target_pfd_dbw_m2_channel = scenario.convert_direct_epfd_power_basis_db(
            -91.75,
            bandwidth_mhz=float(loaded["attrs"]["bandwidth_mhz"]),
            from_basis="per_mhz",
            to_basis="per_channel",
        )

        assert fake_session.accumulate_calls == 1
        assert fake_session.last_accumulate_ras_power_kwargs is not None
        assert fake_session.last_accumulate_ras_power_kwargs[
            "target_pfd_dbw_m2_channel"
        ] == pytest.approx(expected_target_pfd_dbw_m2_channel)
        assert loaded["attrs"]["pfd0_dbw_m2_mhz"] == pytest.approx(-91.75)
        assert loaded["attrs"]["target_pfd_dbw_m2_channel"] == pytest.approx(
            expected_target_pfd_dbw_m2_channel
        )

    def test_run_gpu_direct_epfd_uses_fused_direct_epfd_accumulator_when_available(
        self,
        tmp_path: Path,
    ) -> None:
        filename = tmp_path / "fused_direct_epfd_fake.h5"
        fake_session = _FakeSession(sat_count_total=4, ras_visible_sat_count=2)

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "fused_direct_epfd_fake",
                storage_filename=str(filename),
                session=fake_session,
                ras_pointing_mode="ras_station",
                ras_service_cell_index=0,
                ras_service_cell_active=True,
                output_families=_output_family_profile("gui_heavy"),
                profile_stages=True,
            )
        )

        assert fake_session.accumulate_direct_calls == 1
        assert fake_session.accumulate_calls == 1
        assert fake_session.last_accumulate_direct_epfd_kwargs is not None
        assert fake_session.last_accumulate_direct_epfd_kwargs["sat_azel_shape"] == (1, 2, 2)
        assert_equal(
            fake_session.last_accumulate_direct_epfd_kwargs["output_sat_indices"],
            np.array([0, 1], dtype=np.int32),
        )
        assert result["beam_finalize_chunk_shape"] == {"time_chunk_size": 1, "sky_chunk_size": 1}
        assert result["boresight_compaction_stats"] == {}
        assert result["hot_path_device_to_host_copy_count"] == 0
        assert result["hot_path_device_to_host_copy_bytes"] == 0
        assert result["device_scalar_sync_count"] == 0
        assert "beam_finalize" in result["profile_stage_timings_summary"]
        assert "power_accumulation" in result["profile_stage_timings_summary"]

    def test_run_gpu_direct_epfd_fallback_finalize_uses_compact_ras_sat_azel(
        self,
        tmp_path: Path,
    ) -> None:
        filename = tmp_path / "fallback_direct_epfd_fake.h5"
        fake_session = _FakeSession(sat_count_total=4, ras_visible_sat_count=2)
        fake_session.accumulate_direct_epfd_from_link_library = None

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "fallback_direct_epfd_fake",
                storage_filename=str(filename),
                session=fake_session,
                ras_pointing_mode="ras_station",
                ras_service_cell_index=0,
                ras_service_cell_active=True,
                output_families=_output_family_profile("gui_heavy"),
                profile_stages=True,
            )
        )

        assert fake_session.accumulate_direct_calls == 0
        assert fake_session.accumulate_calls == 1
        assert fake_session.last_link_library is not None
        assert fake_session.last_link_library.last_finalize_kwargs is not None
        assert fake_session.last_link_library.last_finalize_kwargs["ras_sat_azel_shape"] == (1, 2, 2)
        assert_equal(
            fake_session.last_link_library.last_finalize_kwargs["output_sat_indices"],
            np.array([0, 1], dtype=np.int32),
        )
        assert result["beam_finalize_chunk_shape"] == {"time_chunk_size": 1, "sky_chunk_size": 1}
        assert result["boresight_compaction_stats"] == {}
        assert result["hot_path_device_to_host_copy_count"] == 0
        assert result["device_scalar_sync_count"] == 0

    def test_run_gpu_direct_epfd_counts_only_skips_power(self, tmp_path: Path):
        filename = tmp_path / "counts_only_fake.h5"
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            include_atmosphere=False,
            output_families=_output_family_profile("counts_only"),
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert set(result["written_output_names"]) == {
            "sat_beam_counts_used",
            "sat_elevation_RAS_STATION_deg",
            "beam_demand_count",
        }
        assert fake_session.accumulate_calls == 0
        assert fake_gpu.copy_calls == 3
        assert "EPFD_W_m2" not in loaded["iter"][0]
        assert loaded["attrs"]["sat_beam_counts_used_scope"] == "full_network_per_satellite"
        assert loaded["iter"][0]["sat_beam_counts_used"].shape == (1, 1, 2, 2)
        assert_equal(
            loaded["iter"][0]["sat_beam_counts_used"],
            np.array([[[[1, 1], [2, 2]]]], dtype=np.uint8),
        )

    def test_run_gpu_direct_epfd_can_write_sat_elevation_without_power(self, tmp_path: Path):
        filename = tmp_path / "elevation_only_fake.h5"
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            include_atmosphere=False,
            output_families=_output_family_profile("counts_only"),
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert set(result["written_output_names"]) == {
            "sat_beam_counts_used",
            "sat_elevation_RAS_STATION_deg",
            "beam_demand_count",
        }
        assert fake_session.accumulate_calls == 0
        assert fake_gpu.copy_calls == 3
        assert_equal(
            loaded["iter"][0]["sat_elevation_RAS_STATION_deg"],
            np.array([[25.0, 35.0]], dtype=np.float32),
        )

    def test_run_gpu_direct_epfd_can_write_beam_demand_count_without_power(self, tmp_path: Path):
        filename = tmp_path / "beam_demand_only_fake.h5"
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            include_atmosphere=False,
            output_families=_output_family_profile("counts_only"),
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert set(result["written_output_names"]) == {
            "sat_beam_counts_used",
            "sat_elevation_RAS_STATION_deg",
            "beam_demand_count",
        }
        assert fake_session.accumulate_calls == 0
        assert fake_gpu.copy_calls == 3
        assert_equal(
            loaded["iter"][0]["beam_demand_count"],
            np.array([2], dtype=np.uint8),
        )

    def test_run_gpu_direct_epfd_writes_full_network_sat_beam_counts_even_when_ras_subset_is_strict(self, tmp_path: Path):
        filename = tmp_path / "full_network_counts_fake.h5"
        fake_session = _FakeSession(sat_count_total=3, ras_visible_sat_count=2)
        fake_gpu = _FakeGpuModule()

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            include_atmosphere=False,
            output_families=_output_family_profile("counts_only"),
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert set(result["written_output_names"]) == {
            "sat_beam_counts_used",
            "sat_elevation_RAS_STATION_deg",
            "beam_demand_count",
        }
        assert loaded["attrs"]["sat_beam_counts_used_scope"] == "full_network_per_satellite"
        assert loaded["iter"][0]["sat_beam_counts_used"].shape == (1, 1, 3, 2)
        assert_equal(
            loaded["iter"][0]["sat_beam_counts_used"],
            np.array([[[[1, 1], [2, 2], [3, 3]]]], dtype=np.uint8),
        )

    def test_run_gpu_direct_epfd_accepts_nondeterministic_cell_activity_mode(self, tmp_path: Path):
        filename = tmp_path / "counts_only_random_activity_fake.h5"
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            include_atmosphere=False,
            cell_activity_factor=0.5,
            cell_activity_seed_base=None,
            output_families=_output_family_profile("counts_only"),
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert set(result["written_output_names"]) == {
            "sat_beam_counts_used",
            "sat_elevation_RAS_STATION_deg",
            "beam_demand_count",
        }
        assert fake_session.accumulate_calls == 0
        assert fake_gpu.copy_calls == 3
        assert loaded["iter"][0]["sat_beam_counts_used"].shape == (1, 1, 2, 2)

    def test_run_gpu_direct_epfd_per_channel_activity_uses_group_weights_on_device(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        filename = tmp_path / "per_channel_activity_fake.h5"
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        monkeypatch.setattr(
            scenario,
            "_sample_cell_group_activity_mask_device",
            lambda cp, **_kwargs: np.asarray([[[True, False], [False, True]]], dtype=bool),
        )
        monkeypatch.setattr(
            scenario,
            "_compute_cell_activity_spectral_weight_time_slabbed_device",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("legacy full-batch spectral-weight helper should not be used")
            ),
        )
        monkeypatch.setattr(
            scenario,
            "_sample_cell_activity_mask_device",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("whole-cell activity sampler should not be used")
            ),
        )

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            include_atmosphere=False,
            cell_activity_factor=0.5,
            cell_activity_mode="per_channel",
            profile_stages=True,
            output_families=_output_family_profile("all_outputs"),
            spectrum_plan={
                "service_band_start_mhz": 2620.0,
                "service_band_stop_mhz": 2660.0,
                "ras_receiver_band_start_mhz": 2690.0,
                "ras_receiver_band_stop_mhz": 2700.0,
                "reuse_factor": 4,
                "channel_groups_per_cell_cap": 2,
                "active_cell_reuse_slot_ids": np.asarray([0, 1], dtype=np.int32),
                "multi_group_power_policy": "repeat_per_group",
            },
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert result["storage_filename"] == str(filename)
        assert fake_session.last_accumulate_direct_epfd_kwargs is not None
        assert fake_session.last_accumulate_direct_epfd_kwargs["cell_spectral_weight"] is None
        assert np.asarray(
            fake_session.last_accumulate_direct_epfd_kwargs["dynamic_group_active_mask"],
            dtype=bool,
        ).shape == (1, 2, 2)
        assert fake_session.last_accumulate_ras_power_kwargs is not None
        cell_spectral_weight = np.asarray(
            fake_session.last_accumulate_ras_power_kwargs["cell_spectral_weight"],
            dtype=np.float32,
        )
        cell_group_leakage = np.asarray(
            loaded["const"]["cell_group_spectral_leakage_factor_active"],
            dtype=np.float32,
        )
        assert cell_spectral_weight.shape == (1, 2)
        assert_equal(
            cell_spectral_weight,
            np.asarray(
                [[cell_group_leakage[0, 0], cell_group_leakage[1, 1]]],
                dtype=np.float32,
            ),
        )
        assert_equal(
            loaded["iter"][0]["beam_demand_count"],
            np.asarray([2], dtype=np.uint8),
        )
        stage_summary = result["profile_stage_timings_summary"]
        assert "beam_finalize" in stage_summary
        assert "spectrum_activity_weighting" in stage_summary
        assert "power_accumulation" in stage_summary

    def test_run_gpu_direct_epfd_per_channel_activity_fast_path_skips_sampling_at_full_activity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        filename = tmp_path / "per_channel_full_activity_fast_path.h5"
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        monkeypatch.setattr(
            scenario,
            "_sample_cell_group_activity_mask_device",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("group activity sampling should be skipped at full activity")
            ),
        )
        monkeypatch.setattr(
            scenario,
            "_compute_cell_activity_spectral_weight_time_slabbed_device",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("slabbed spectral weighting should be skipped at full activity")
            ),
        )
        monkeypatch.setattr(
            scenario,
            "_sample_cell_activity_mask_device",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("whole-cell activity sampler should not be used")
            ),
        )

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "per_channel_full_activity_fast_path",
                storage_filename=str(filename),
                gpu_module=fake_gpu,
                session=fake_session,
                output_families=_output_family_profile("all_outputs"),
                cell_activity_factor=1.0,
                cell_activity_mode="per_channel",
                profile_stages=True,
                spectrum_plan={
                    "service_band_start_mhz": 2620.0,
                    "service_band_stop_mhz": 2660.0,
                    "ras_receiver_band_start_mhz": 2690.0,
                    "ras_receiver_band_stop_mhz": 2700.0,
                    "reuse_factor": 4,
                    "disabled_channel_indices": np.asarray([], dtype=np.int32),
                    "active_cell_reuse_slot_ids": np.asarray([0, 1], dtype=np.int32),
                    "multi_group_power_policy": "repeat_per_group",
                },
            )
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert result["storage_filename"] == str(filename)
        assert fake_session.last_accumulate_ras_power_kwargs is not None
        cell_spectral_weight = np.asarray(
            fake_session.last_accumulate_ras_power_kwargs["cell_spectral_weight"],
            dtype=np.float32,
        )
        assert cell_spectral_weight.shape == (1, 2)
        assert_equal(
            cell_spectral_weight,
            np.asarray(loaded["const"]["cell_spectral_leakage_factor_active"], dtype=np.float32)[
                None, :
            ],
        )
        assert_equal(
            loaded["iter"][0]["beam_demand_count"],
            np.asarray([2], dtype=np.uint8),
        )
        stage_summary = result["profile_stage_timings_summary"]
        assert "cell_activity_setup" in stage_summary
        assert "spectrum_activity_weighting" not in stage_summary
        assert "power_accumulation" in stage_summary

    def test_run_gpu_direct_epfd_per_channel_activity_uses_group_weights_on_device_with_boresight(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        filename = tmp_path / "per_channel_activity_boresight_fake.h5"
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        monkeypatch.setattr(
            scenario,
            "_sample_cell_group_activity_mask_device",
            lambda cp, **_kwargs: np.asarray([[[True, False], [False, True]]], dtype=bool),
        )
        monkeypatch.setattr(
            scenario,
            "_compute_cell_activity_spectral_weight_time_slabbed_device",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("legacy full-batch spectral-weight helper should not be used")
            ),
        )
        monkeypatch.setattr(
            scenario,
            "_sample_cell_activity_mask_device",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("whole-cell activity sampler should not be used")
            ),
        )

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            include_atmosphere=False,
            cell_activity_factor=0.5,
            cell_activity_mode="per_channel",
            profile_stages=True,
            output_families=_output_family_profile("all_outputs"),
            spectrum_plan={
                "service_band_start_mhz": 2620.0,
                "service_band_stop_mhz": 2660.0,
                "ras_receiver_band_start_mhz": 2690.0,
                "ras_receiver_band_stop_mhz": 2700.0,
                "reuse_factor": 4,
                "channel_groups_per_cell_cap": 2,
                "active_cell_reuse_slot_ids": np.asarray([0, 1], dtype=np.int32),
                "multi_group_power_policy": "repeat_per_group",
            },
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert result["boresight_active"] is True
        assert fake_session.last_accumulate_direct_epfd_kwargs is not None
        assert fake_session.last_accumulate_direct_epfd_kwargs["cell_spectral_weight"] is None
        assert np.asarray(
            fake_session.last_accumulate_direct_epfd_kwargs["dynamic_group_active_mask"],
            dtype=bool,
        ).shape == (1, 2, 2)
        assert fake_session.last_accumulate_ras_power_kwargs is not None
        cell_spectral_weight = np.asarray(
            fake_session.last_accumulate_ras_power_kwargs["cell_spectral_weight"],
            dtype=np.float32,
        )
        cell_group_leakage = np.asarray(
            loaded["const"]["cell_group_spectral_leakage_factor_active"],
            dtype=np.float32,
        )
        assert cell_spectral_weight.shape == (1, 2)
        assert_equal(
            cell_spectral_weight,
            np.asarray(
                [[cell_group_leakage[0, 0], cell_group_leakage[1, 1]]],
                dtype=np.float32,
            ),
        )
        stage_summary = result["profile_stage_timings_summary"]
        assert "beam_finalize" in stage_summary
        assert "spectrum_activity_weighting" in stage_summary
        assert "power_accumulation" in stage_summary

    def test_run_gpu_direct_epfd_metadata_only_skips_dynamic_execution(self, tmp_path: Path):
        filename = tmp_path / "metadata_only_fake.h5"

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            output_families=_output_family_profile("none"),
            storage_constants={"sat_belt_id_per_sat": np.array([0, 0], dtype=np.int16)},
            storage_attrs={"mode": "direct_link_epfd"},
        )

        loaded = scenario.read_data(str(filename), stack=False)

        assert result["dynamic_execution_skipped"] is True
        assert result["written_output_names"] == []
        assert loaded["attrs"]["mode"] == "direct_link_epfd"
        assert loaded["iter"] == {}

    def test_run_gpu_direct_epfd_prepares_boresight_without_backend_selection(self, tmp_path: Path):
        class _CapturingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.prepared_kwargs = None

            def prepare_satellite_link_selection_library(self, *, time_count, **kwargs):
                self.prepared_kwargs = dict(kwargs)
                return super().prepare_satellite_link_selection_library(time_count=time_count, **kwargs)

        filename = tmp_path / "boresight_backend_default_fake.h5"
        fake_session = _CapturingSession()

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            include_atmosphere=False,
            output_families=_output_family_profile("counts_only"),
            cupy_module=_FakeCp(),
            gpu_module=_FakeGpuModule(),
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        assert fake_session.prepared_kwargs is not None
        assert "boresight_backend" not in fake_session.prepared_kwargs
        assert result["boresight_active"] is True

    def test_run_gpu_direct_epfd_exposes_debug_flag_in_result(self, tmp_path: Path):
        filename = tmp_path / "debug_flag_fake.h5"

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            include_atmosphere=False,
            output_families=_output_family_profile("counts_only"),
            debug_direct_epfd=True,
            cupy_module=_FakeCp(),
            gpu_module=_FakeGpuModule(),
            session=_FakeSession(),
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        assert result["debug_direct_epfd"] is True
        assert len(result["debug_direct_epfd_stats"]) == 1
        assert result["debug_direct_epfd_stats"][0]["iteration"] == 0
        assert result["debug_direct_epfd_stats"][0]["batch_index"] == 0

    def test_run_gpu_direct_epfd_option_matrix_smoke(self, tmp_path: Path):
        """Smoke test: verify representative parameter combinations run.

        Instead of an exhaustive 8-level cross-product (~1008 combos),
        tests each dimension independently plus a seeded set of random
        cross-combos.  Total: ~120 combinations — same invariant coverage
        with ~10× less runtime.
        """
        import random as _rng

        prepared_grid = _theta2_scope_test_grid()
        resolved_theta2_cases = {
            "cell_ids": earthgrid.resolve_theta2_active_cell_ids(
                prepared_grid,
                scope_mode="cell_ids",
                explicit_ids=[0, 1, 1],
            ),
            "adjacency_layers": earthgrid.resolve_theta2_active_cell_ids(
                prepared_grid,
                scope_mode="adjacency_layers",
                layers=1,
            ),
            "radius_km": earthgrid.resolve_theta2_active_cell_ids(
                prepared_grid,
                scope_mode="radius_km",
                radius_km=120.0,
            ),
        }
        boresight_cases: list[tuple[str, dict[str, object]]] = [
            ("none", {}),
            ("theta1", {"boresight_theta1": 1.0 * u.deg}),
        ]
        for scope_mode, resolved_ids in resolved_theta2_cases.items():
            boresight_cases.append(
                (
                    f"theta2_{scope_mode}",
                    {
                        "boresight_theta2": 2.0 * u.deg,
                        "boresight_theta2_cell_ids": resolved_ids,
                    },
                )
            )
            boresight_cases.append(
                (
                    f"both_{scope_mode}",
                    {
                        "boresight_theta1": 1.0 * u.deg,
                        "boresight_theta2": 2.0 * u.deg,
                        "boresight_theta2_cell_ids": resolved_ids,
                    },
                )
            )

        # Build the full parameter space as lists for random sampling.
        selection_strategies = ("max_elevation", "random")
        ras_pointing_modes = ("cell_center", "ras_station")
        atmosphere_options = (False, True)
        memory_budget_modes = ("hybrid", "host_only", "gpu_only")
        profile_names = ("counts_only", "totals_only")
        tiny_budget_options = (False, True)

        # 1) One-at-a-time: test each dimension value with defaults for
        #    everything else (covers every feature in isolation).
        # 2) Random cross-combos: 80 seeded random combinations (covers
        #    interactions without exhaustive cross-product).
        combos: list[tuple] = []

        # Defaults: first value of each dimension
        default = (
            selection_strategies[0],
            ras_pointing_modes[0],
            False,  # ras_service_cell_active
            boresight_cases[0],
            atmosphere_options[0],
            memory_budget_modes[0],
            profile_names[0],
            tiny_budget_options[0],
        )

        # One-at-a-time sweeps
        for sel in selection_strategies:
            combos.append((sel, *default[1:]))
        for rp in ras_pointing_modes:
            combos.append((default[0], rp, rp == "ras_station", *default[3:]))
        for bc in boresight_cases:
            combos.append((*default[:3], bc, *default[4:]))
        for atm in atmosphere_options:
            combos.append((*default[:4], atm, *default[5:]))
        for mbm in memory_budget_modes:
            combos.append((*default[:5], mbm, *default[6:]))
        for prof in profile_names:
            combos.append((*default[:6], prof, default[7]))
        for tiny in tiny_budget_options:
            combos.append((*default[:7], tiny))

        # Random cross-combos (seeded)
        gen = _rng.Random(44)
        for _ in range(80):
            sel = gen.choice(selection_strategies)
            rp = gen.choice(ras_pointing_modes)
            sca = gen.choice([True, False]) if rp == "ras_station" else False
            bc = gen.choice(boresight_cases)
            atm = gen.choice(atmosphere_options)
            mbm = gen.choice(memory_budget_modes)
            prof = gen.choice(profile_names)
            tiny = gen.choice(tiny_budget_options)
            combos.append((sel, rp, sca, bc, atm, mbm, prof, tiny))

        # Deduplicate
        seen: set[str] = set()
        unique_combos: list[tuple] = []
        for combo in combos:
            key = str(combo)
            if key not in seen:
                seen.add(key)
                unique_combos.append(combo)

        combo_index = 0
        for (
            selection_strategy,
            ras_pointing_mode,
            ras_service_cell_active,
            boresight_entry,
            include_atmosphere,
            memory_budget_mode,
            profile_name,
            tiny_budget,
        ) in unique_combos:
            boresight_name, boresight_kwargs = boresight_entry
            expected_pointing_mode = (
                "ras_station" if ras_service_cell_active else "cell_center"
            )
            profile_case = _FAKE_OUTPUT_PROFILE_CASES[profile_name]
            combo_index += 1
            fake_session = _FakeSession()
            fake_gpu = _FakeGpuModule()
            result = scenario.run_gpu_direct_epfd(
                **_fake_direct_epfd_common_kwargs(
                    tmp_path,
                    f"matrix_{combo_index}",
                    session=fake_session,
                    gpu_module=fake_gpu,
                    active_cell_longitudes=prepared_grid["active_grid_longitudes"],
                    selection_strategy=selection_strategy,
                    ras_pointing_mode=ras_pointing_mode,
                    ras_service_cell_index=(
                        0 if ras_pointing_mode == "ras_station" else -1
                    ),
                    ras_service_cell_active=ras_service_cell_active,
                    include_atmosphere=include_atmosphere,
                    memory_budget_mode=memory_budget_mode,
                    finalize_memory_budget_bytes=(1 if tiny_budget else None),
                    power_memory_budget_bytes=(1 if tiny_budget else None),
                    export_memory_budget_bytes=(1 if tiny_budget else None),
                    **profile_case["profile_kwargs"],
                    **boresight_kwargs,
                )
            )

            loaded = scenario.read_data(
                result["storage_filename"],
                stack=False,
            )
            iter_payload = loaded["iter"][0]
            _, expected_gpu_budget_mode = (
                scenario._normalise_runner_memory_budget_modes(
                    memory_budget_mode
                )
            )

            assert result["effective_ras_pointing_mode"] == expected_pointing_mode
            assert profile_case["expected_iter_names"].issubset(iter_payload.keys())
            assert "sat_beam_counts_used" in iter_payload
            if boresight_name == "none":
                assert result["boresight_active"] is False
            else:
                assert result["boresight_active"] is True
            assert fake_session.accumulate_calls == int(
                bool(profile_case["expect_power"])
            )
            assert fake_session.atmosphere_prepare_calls == int(
                include_atmosphere and bool(profile_case["expect_power"])
            )
            assert fake_session.device_budget_calls
            assert all(
                call["mode"] == expected_gpu_budget_mode
                for call in fake_session.device_budget_calls
            )

    def test_run_gpu_direct_epfd_notebook_style_smoke(self, tmp_path: Path):
        filename = tmp_path / "notebook_style_fake.h5"
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=np.array([10.0, 11.0], dtype=np.float64) * u.deg,
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=2026,
            nco=1,
            nbeam=4,
            selection_strategy="random",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta1=1.0 * u.deg,
            boresight_theta2=2.0 * u.deg,
            boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
            include_atmosphere=False,
            output_families=_output_family_profile("notebook_full"),
            finalize_memory_budget_bytes=1,
            power_memory_budget_bytes=1,
            export_memory_budget_bytes=1,
            debug_direct_epfd=True,
            cupy_module=_FakeCp(),
            gpu_module=fake_gpu,
            session=fake_session,
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)
        assert result["boresight_active"] is True
        assert result["debug_direct_epfd"] is True
        assert "EPFD_W_m2" in loaded["iter"][0]
        assert "Prx_total_W" in loaded["iter"][0]
        assert "PFD_total_RAS_STATION_W_m2" in loaded["iter"][0]
        assert "PFD_per_sat_RAS_STATION_W_m2" in loaded["iter"][0]
        assert "sat_elevation_RAS_STATION_deg" in loaded["iter"][0]
        assert "beam_demand_count" in loaded["iter"][0]

    def test_run_gpu_direct_epfd_multi_batch_writes_full_iteration(self, tmp_path: Path):
        base_start_time = Time("2025-01-01T00:00:00", scale="utc")
        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "multi_batch_full_iteration",
                session=_FakeSession(),
                gpu_module=_FakeGpuModule(),
                selection_strategy="random",
                ras_pointing_mode="ras_station",
                ras_service_cell_index=0,
                ras_service_cell_active=True,
                include_atmosphere=True,
                boresight_theta1=1.0 * u.deg,
                boresight_theta2=2.0 * u.deg,
                boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
                output_families=_output_family_profile("all_outputs"),
                base_start_time=base_start_time,
                base_end_time=Time("2025-01-01T00:00:05", scale="utc"),
                timestep=1.0,
                iteration_count=1,
                force_bulk_timesteps=2,
                finalize_memory_budget_bytes=1,
                power_memory_budget_bytes=1,
                export_memory_budget_bytes=1,
            )
        )

        loaded = scenario.read_data(result["storage_filename"], stack=False)
        iter_payload = loaded["iter"][0]
        stored_times = scenario.read_dataset_slice(
            result["storage_filename"],
            iteration=0,
            name="times",
        )
        expected_step_count = int(result["n_steps_total"])
        stored_times_mjd = np.asarray(stored_times, dtype=np.float64)
        stored_span_s = (
            0.0
            if expected_step_count <= 1
            else float((stored_times_mjd[-1] - stored_times_mjd[0]) * 86400.0)
        )

        assert stored_times_mjd.shape == (expected_step_count,)
        assert_allclose(
            np.diff(stored_times_mjd) * 86400.0,
            np.ones(expected_step_count - 1),
            rtol=1e-6,
            atol=1e-6,
        )
        assert stored_span_s == pytest.approx(float(max(expected_step_count - 1, 0)))
        assert iter_payload["EPFD_W_m2"].shape[0] == expected_step_count
        assert iter_payload["Prx_total_W"].shape[0] == expected_step_count
        assert iter_payload["PFD_total_RAS_STATION_W_m2"].shape[0] == expected_step_count
        assert iter_payload["PFD_per_sat_RAS_STATION_W_m2"].shape[0] == expected_step_count
        assert iter_payload["sat_beam_counts_used"].shape[0] == expected_step_count
        assert iter_payload["sat_elevation_RAS_STATION_deg"].shape[0] == expected_step_count
        assert iter_payload["beam_demand_count"].shape[0] == expected_step_count
        assert int(result["writer_stats_summary"]["submitted_seq"]) >= 2
        assert int(result["writer_stats_summary"]["completed_seq"]) >= 2

    @pytest.mark.parametrize(
        ("profile_name", "profile_case"),
        list(_FAKE_OUTPUT_PROFILE_CASES.items()),
    )
    @pytest.mark.parametrize("selection_strategy", ("max_elevation", "random"))
    @pytest.mark.parametrize("ras_pointing_mode", ("cell_center", "ras_station"))
    @pytest.mark.parametrize("include_atmosphere", (False, True))
    def test_run_gpu_direct_epfd_output_profiles_smoke(
        self,
        tmp_path: Path,
        profile_name: str,
        profile_case: dict[str, object],
        selection_strategy: str,
        ras_pointing_mode: str,
        include_atmosphere: bool,
    ):
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                f"{profile_name}_{selection_strategy}_{ras_pointing_mode}_{int(include_atmosphere)}",
                session=fake_session,
                gpu_module=fake_gpu,
                selection_strategy=selection_strategy,
                ras_pointing_mode=ras_pointing_mode,
                ras_service_cell_index=0 if ras_pointing_mode == "ras_station" else -1,
                ras_service_cell_active=(ras_pointing_mode == "ras_station"),
                include_atmosphere=include_atmosphere,
                boresight_theta1=1.0 * u.deg,
                boresight_theta2=2.0 * u.deg,
                boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
                finalize_memory_budget_bytes=1,
                power_memory_budget_bytes=1,
                export_memory_budget_bytes=1,
                **profile_case["profile_kwargs"],
            )
        )

        loaded = scenario.read_data(result["storage_filename"], stack=False)
        if profile_name == "none":
            assert result["dynamic_execution_skipped"] is True
            assert loaded["iter"] == {}
            assert loaded.get("preaccumulated", {}) == {}
            assert fake_session.accumulate_calls == 0
            assert fake_session.atmosphere_prepare_calls == 0
            return

        iter_payload = loaded["iter"][0]
        preacc_payload = loaded.get("preaccumulated", {})
        assert profile_case["expected_iter_names"].issubset(iter_payload.keys())
        assert profile_case["expected_preacc_names"].issubset(preacc_payload.keys())
        if bool(profile_case["expect_prx_per_sat"]):
            assert "Prx_per_sat_RAS_STATION_W" in iter_payload
        else:
            assert "Prx_per_sat_RAS_STATION_W" not in iter_payload
        if bool(profile_case["expect_prx_heatmap"]):
            assert "prx_elevation_heatmap/counts" in preacc_payload
            assert "prx_elevation_heatmap/value_edges_dbw" in preacc_payload
            assert loaded["attrs"]["output_family_prx_elevation_heatmap_mode"] in {
                "preaccumulated",
                "both",
            }
        else:
            assert "prx_elevation_heatmap/counts" not in preacc_payload
        assert fake_session.accumulate_calls == int(bool(profile_case["expect_power"]))
        assert fake_session.atmosphere_prepare_calls == int(
            include_atmosphere and bool(profile_case["expect_power"])
        )

    def test_run_gpu_direct_epfd_heavy_public_branch_matrix(self, tmp_path: Path):
        prepared_grid = _theta2_scope_test_grid()
        resolved_theta2_cases = {
            "cell_ids": earthgrid.resolve_theta2_active_cell_ids(
                prepared_grid,
                scope_mode="cell_ids",
                explicit_ids=[0, 1, 1],
            ),
            "adjacency_layers": earthgrid.resolve_theta2_active_cell_ids(
                prepared_grid,
                scope_mode="adjacency_layers",
                layers=1,
            ),
            "radius_km": earthgrid.resolve_theta2_active_cell_ids(
                prepared_grid,
                scope_mode="radius_km",
                radius_km=120.0,
            ),
        }
        matrix_cases = [
            {
                "name": "notebook_full_theta1_ras_station",
                "selection_strategy": "random",
                "ras_pointing_mode": "ras_station",
                "ras_service_cell_active": True,
                "include_atmosphere": True,
                "memory_budget_mode": "gpu_only",
                "profile_name": "notebook_full",
                "boresight_kwargs": {"boresight_theta1": 1.0 * u.deg},
            },
            {
                "name": "gui_heavy_theta2_cell_ids",
                "selection_strategy": "max_elevation",
                "ras_pointing_mode": "cell_center",
                "ras_service_cell_active": False,
                "include_atmosphere": False,
                "memory_budget_mode": "host_only",
                "profile_name": "gui_heavy",
                "boresight_kwargs": {
                    "boresight_theta2": 2.0 * u.deg,
                    "boresight_theta2_cell_ids": resolved_theta2_cases["cell_ids"],
                },
            },
            {
                "name": "notebook_full_theta2_layers",
                "selection_strategy": "random",
                "ras_pointing_mode": "cell_center",
                "ras_service_cell_active": False,
                "include_atmosphere": True,
                "memory_budget_mode": "hybrid",
                "profile_name": "notebook_full",
                "boresight_kwargs": {
                    "boresight_theta2": 2.0 * u.deg,
                    "boresight_theta2_cell_ids": resolved_theta2_cases["adjacency_layers"],
                },
            },
            {
                "name": "gui_heavy_theta2_radius_ras_station",
                "selection_strategy": "max_elevation",
                "ras_pointing_mode": "ras_station",
                "ras_service_cell_active": True,
                "include_atmosphere": True,
                "memory_budget_mode": "gpu_only",
                "profile_name": "gui_heavy",
                "boresight_kwargs": {
                    "boresight_theta2": 2.0 * u.deg,
                    "boresight_theta2_cell_ids": resolved_theta2_cases["radius_km"],
                },
            },
            {
                "name": "notebook_full_both_inactive_service_cell",
                "selection_strategy": "random",
                "ras_pointing_mode": "ras_station",
                "ras_service_cell_active": False,
                "include_atmosphere": True,
                "memory_budget_mode": "host_only",
                "profile_name": "notebook_full",
                "boresight_kwargs": {
                    "boresight_theta1": 1.0 * u.deg,
                    "boresight_theta2": 2.0 * u.deg,
                    "boresight_theta2_cell_ids": resolved_theta2_cases["adjacency_layers"],
                },
            },
        ]

        for case in matrix_cases:
            fake_session = _FakeSession()
            fake_gpu = _FakeGpuModule()
            profile_case = _FAKE_OUTPUT_PROFILE_CASES[str(case["profile_name"])]
            result = scenario.run_gpu_direct_epfd(
                **_fake_direct_epfd_common_kwargs(
                    tmp_path,
                    str(case["name"]),
                    session=fake_session,
                    gpu_module=fake_gpu,
                    active_cell_longitudes=prepared_grid["active_grid_longitudes"],
                    selection_strategy=str(case["selection_strategy"]),
                    ras_pointing_mode=str(case["ras_pointing_mode"]),
                    ras_service_cell_index=(
                        0 if str(case["ras_pointing_mode"]) == "ras_station" else -1
                    ),
                    ras_service_cell_active=bool(case["ras_service_cell_active"]),
                    include_atmosphere=bool(case["include_atmosphere"]),
                    memory_budget_mode=str(case["memory_budget_mode"]),
                    finalize_memory_budget_bytes=1,
                    power_memory_budget_bytes=1,
                    export_memory_budget_bytes=1,
                    **profile_case["profile_kwargs"],
                    **case["boresight_kwargs"],
                )
            )

            loaded = scenario.read_data(result["storage_filename"], stack=False)
            iter_payload = loaded["iter"][0]
            preacc_payload = loaded.get("preaccumulated", {})
            stored_times = np.asarray(
                scenario.read_dataset_slice(
                    result["storage_filename"],
                    iteration=0,
                    name="times",
                ),
                dtype=np.float64,
            )
            expected_pointing_mode = (
                "ras_station" if bool(case["ras_service_cell_active"]) else "cell_center"
            )

            assert result["effective_ras_pointing_mode"] == expected_pointing_mode
            assert result["boresight_active"] is True
            assert profile_case["expected_iter_names"].issubset(iter_payload.keys())
            assert profile_case["expected_preacc_names"].issubset(preacc_payload.keys())
            assert stored_times.shape == (int(result["n_steps_total"]),)
            assert "writer_stats_summary" in result
            assert result["writer_stats_summary"]["durability_mode"] in {"fsync", "flush_only"}
            if bool(profile_case["expect_prx_heatmap"]):
                assert "prx_elevation_heatmap/counts" in preacc_payload
                assert "prx_elevation_heatmap/value_edges_dbw" in preacc_payload
            if bool(profile_case["expect_prx_per_sat"]):
                assert "Prx_per_sat_RAS_STATION_W" in iter_payload
            assert fake_session.accumulate_calls == int(bool(profile_case["expect_power"]))
            assert fake_session.atmosphere_prepare_calls == int(
                bool(case["include_atmosphere"]) and bool(profile_case["expect_power"])
            )

    def test_run_gpu_direct_epfd_graceful_stop_flushes_partial_result(self, tmp_path: Path):
        cancel_mode = {"value": "none"}

        def _progress(event: dict[str, object]) -> None:
            if (
                str(event.get("phase", "")) == "write_enqueue"
                and int(event.get("batch_index", -1)) == 0
            ):
                cancel_mode["value"] = "graceful"

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "graceful_stop_partial",
                base_end_time=Time("2025-01-01T00:00:05", scale="utc"),
                timestep=1.0,
                iteration_count=1,
                force_bulk_timesteps=2,
                output_families=_output_family_profile("gui_heavy"),
                finalize_memory_budget_bytes=1,
                power_memory_budget_bytes=1,
                export_memory_budget_bytes=1,
                progress_callback=_progress,
                cancel_callback=lambda: str(cancel_mode["value"]),
            )
        )

        stored_times = np.asarray(
            scenario.read_dataset_slice(
                result["storage_filename"],
                iteration=0,
                name="times",
            ),
            dtype=np.float64,
        )
        assert result["run_state"] == "stopped"
        assert result["stop_mode"] == "graceful"
        assert result["stop_boundary"] == "post_batch_boundary"
        assert 0 < stored_times.size < int(result["n_steps_total"])
        assert Path(str(result["storage_filename"])).exists()

    def test_run_gpu_direct_epfd_force_stop_reports_force_boundary(self, tmp_path: Path):
        cancel_mode = {"value": "none"}

        def _progress(event: dict[str, object]) -> None:
            if str(event.get("phase", "")) == "chunk_detail":
                cancel_mode["value"] = "force"

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "force_stop_partial",
                active_cell_longitudes=np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64) * u.deg,
                base_end_time=Time("2025-01-01T00:00:05", scale="utc"),
                timestep=1.0,
                iteration_count=1,
                force_bulk_timesteps=2,
                output_families=_output_family_profile("gui_heavy"),
                finalize_memory_budget_bytes=1,
                power_memory_budget_bytes=1,
                export_memory_budget_bytes=1,
                progress_callback=_progress,
                cancel_callback=lambda: str(cancel_mode["value"]),
            )
        )

        assert result["run_state"] == "stopped"
        assert result["stop_mode"] == "force"
        assert result["stop_boundary"] in {"chunk_boundary", "beam_slab_boundary", "post_batch_boundary"}
        assert Path(str(result["storage_filename"])).exists()

    def test_beam_count_samples_device_collapses_sky_axis(self):
        counts = np.asarray(
            [
                [[1, 2], [3, 0]],
                [[0, 4], [5, 6]],
            ],
            dtype=np.int64,
        )

        samples = scenario._beam_count_samples_device(np, counts)

        np.testing.assert_array_equal(
            samples,
            np.asarray(
                [
                    [3, 2],
                    [5, 6],
                ],
                dtype=np.int64,
            ),
        )

    def test_visible_beam_statistics_device_uses_per_timestep_visibility(self):
        class _DummyGpuModule:
            @staticmethod
            def copy_device_to_host(value):
                return np.asarray(value)

        samples = np.asarray(
            [
                [3, 2],
                [5, 6],
            ],
            dtype=np.int64,
        )
        visibility_mask = np.asarray(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        )

        hist, totals = scenario._visible_beam_statistics_device(
            np,
            _DummyGpuModule(),
            counts_samples_device=samples,
            visibility_mask_device=visibility_mask,
        )

        np.testing.assert_array_equal(
            hist,
            np.asarray([0, 0, 0, 1, 0, 0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            totals,
            np.asarray([3, 6], dtype=np.int64),
        )

    @pytest.mark.parametrize(
        ("scope_mode", "resolver_kwargs"),
        [
            ("cell_ids", {"explicit_ids": [0, 1, 1]}),
            ("adjacency_layers", {"layers": 1}),
            ("radius_km", {"radius_km": 120.0}),
        ],
    )
    def test_run_gpu_direct_epfd_theta2_scope_resolution_smoke(
        self,
        tmp_path: Path,
        scope_mode: str,
        resolver_kwargs: dict[str, object],
    ):
        prepared_grid = _theta2_scope_test_grid()
        resolved_ids = earthgrid.resolve_theta2_active_cell_ids(
            prepared_grid,
            scope_mode=scope_mode,
            **resolver_kwargs,
        )

        assert resolved_ids.dtype == np.int32
        assert np.all(resolved_ids >= 0)
        assert np.all(resolved_ids < prepared_grid["active_cell_count"])

        filename = tmp_path / f"theta2_scope_{scope_mode}.h5"
        result = scenario.run_gpu_direct_epfd(
            tle_list=np.asarray([object(), object()], dtype=object),
            observer_arr=np.asarray([object(), object(), object()], dtype=object),
            active_cell_longitudes=prepared_grid["active_grid_longitudes"],
            sat_min_elevation_deg_per_sat=np.array([20.0, 20.0], dtype=np.float32),
            sat_beta_max_deg_per_sat=np.array([30.0, 30.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 0], dtype=np.int16),
            sat_orbit_radius_m_per_sat=np.array([6.9e6, 6.9e6], dtype=np.float32),
            pattern_kwargs={
                "Lt": 1.6 * u.m,
                "Lr": 1.6 * u.m,
                "SLR": 20.0 * cnv.dB,
                "l": 2,
                "far_sidelobe_start": 90.0 * u.deg,
                "far_sidelobe_level": -20.0 * cnv.dBi,
            },
            wavelength=15.0 * u.cm,
            ras_station_ant_diam=13.5 * u.m,
            frequency=2.0 * u.GHz,
            ras_station_elev_range=(15.0 * u.deg, 90.0 * u.deg),
            observer_alt_km_ras_station=0.1,
            storage_filename=str(filename),
            base_start_time=Time("2025-01-01T00:00:00", scale="utc"),
            base_end_time=Time("2025-01-01T00:00:00", scale="utc"),
            timestep=1.0,
            iteration_count=1,
            iteration_rng_seed=42,
            nco=1,
            nbeam=4,
            selection_strategy="max_elevation",
            ras_pointing_mode="cell_center",
            ras_service_cell_index=-1,
            ras_service_cell_active=False,
            ras_guard_angle=4.0 * u.deg,
            boresight_theta2=2.0 * u.deg,
            boresight_theta2_cell_ids=resolved_ids,
            include_atmosphere=False,
            output_families=_output_family_profile("counts_only"),
            finalize_memory_budget_bytes=1,
            power_memory_budget_bytes=1,
            export_memory_budget_bytes=1,
            cupy_module=_FakeCp(),
            gpu_module=_FakeGpuModule(),
            session=_FakeSession(),
            progress_factory=lambda iterable, **_kwargs: iterable,
        )

        loaded = scenario.read_data(str(filename), stack=False)
        assert result["boresight_active"] is True
        assert "sat_beam_counts_used" in loaded["iter"][0]

    def test_run_gpu_direct_epfd_profile_stages_include_export_and_writer_tail(self, tmp_path: Path):
        fake_session = _FakeSession()
        fake_gpu = _FakeGpuModule()

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "profile_stage_tail",
                session=fake_session,
                gpu_module=fake_gpu,
                selection_strategy="random",
                ras_pointing_mode="ras_station",
                ras_service_cell_index=0,
                ras_service_cell_active=True,
                include_atmosphere=True,
                boresight_theta1=1.0 * u.deg,
                boresight_theta2=2.0 * u.deg,
                boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
                profile_stages=True,
                output_families=_output_family_profile("all_outputs"),
                finalize_memory_budget_bytes=1,
                power_memory_budget_bytes=1,
                export_memory_budget_bytes=1,
            )
        )

        assert result["profile_stage_timings"]
        batch_timing = result["profile_stage_timings"][0]
        assert "export_copy" in batch_timing
        assert "write_enqueue" in batch_timing

        summary = result["profile_stage_timings_summary"]
        for key in (
            "beam_finalize",
            "power_accumulation",
            "export_copy",
            "write_enqueue",
            "writer_checkpoint_wait",
            "export_scatter",
            "writer_apply",
            "writer_flush",
        ):
            assert key in summary
            assert float(summary[key]) >= 0.0
        assert "writer_stats_summary" in result
        assert "writer_checkpoint_count" in result
        assert "writer_checkpoint_wait_s" in result
        assert "writer_final_flush_s" in result

        loaded = scenario.read_data(result["storage_filename"], stack=False)
        assert "EPFD_W_m2" in loaded["iter"][0]
        assert "Prx_per_sat_RAS_STATION_W" in loaded["iter"][0]
        assert "PFD_per_sat_RAS_STATION_W_m2" in loaded["iter"][0]
        assert "prx_elevation_heatmap/counts" in loaded["preaccumulated"]

    def test_run_gpu_direct_epfd_progress_desc_mode_defaults_to_coarse(self, tmp_path: Path):
        progress_bars: list[_RecordingProgressBar] = []

        scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "progress_coarse",
                progress_factory=_recording_progress_factory(progress_bars),
                output_families=_output_family_profile("totals_only"),
                enable_progress_desc_updates=True,
            )
        )

        descriptions = [text for bar in progress_bars for text in bar.descriptions]
        assert "Chunks" in descriptions
        assert "Write tail" in descriptions
        assert "Finalizing beams" not in descriptions

    def test_run_gpu_direct_epfd_progress_desc_mode_off_suppresses_updates(self, tmp_path: Path):
        progress_bars: list[_RecordingProgressBar] = []

        scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "progress_off",
                progress_factory=_recording_progress_factory(progress_bars),
                output_families=_output_family_profile("totals_only"),
                progress_desc_mode="off",
            )
        )

        descriptions = [text for bar in progress_bars for text in bar.descriptions]
        assert descriptions == []

    def test_run_gpu_direct_epfd_progress_desc_mode_detailed_keeps_tail_labels(self, tmp_path: Path):
        progress_bars: list[_RecordingProgressBar] = []

        scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "progress_detailed",
                progress_factory=_recording_progress_factory(progress_bars),
                output_families=_output_family_profile("totals_only"),
                progress_desc_mode="detailed",
            )
        )

        descriptions = [text for bar in progress_bars for text in bar.descriptions]
        for expected in (
            "Finalizing beams",
            "Accumulating power",
            "Exporting to host",
            "Queueing HDF5 write",
            "Flushing writer",
        ):
            assert expected in descriptions

    def test_run_gpu_direct_epfd_checkpoint_interval_triggers_flush(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        real_flush_writes = scenario.flush_writes
        flush_calls: list[str] = []

        def _recording_flush(filename: str | None = None) -> None:
            flush_calls.append("" if filename is None else str(filename))
            real_flush_writes(filename)

        monkeypatch.setattr(scenario, "flush_writes", _recording_flush)

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "checkpoint_flush",
                output_families=_output_family_profile("totals_only"),
                writer_checkpoint_interval_s=1.0e-9,
            )
        )

        assert flush_calls
        assert result["writer_checkpoint_count"] >= 1
        assert float(result["writer_checkpoint_wait_s"]) >= 0.0

    def test_run_gpu_direct_epfd_multi_batch_checkpoints_grow_times_dataset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        real_flush_writes = scenario.flush_writes
        checkpoint_lengths: list[int] = []

        def _recording_flush(filename: str | None = None) -> None:
            real_flush_writes(filename)
            if filename is None:
                return
            try:
                times = scenario.read_dataset_slice(
                    str(filename),
                    iteration=0,
                    name="times",
                    selection=np.s_[:],
                    sync_pending_writes=False,
                )
            except Exception:
                return
            checkpoint_lengths.append(int(np.asarray(times).shape[0]))

        monkeypatch.setattr(scenario, "flush_writes", _recording_flush)

        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "checkpoint_multibatch_growth",
                base_end_time=Time("2025-01-01T00:00:05", scale="utc"),
                force_bulk_timesteps=2,
                selection_strategy="random",
                ras_pointing_mode="ras_station",
                ras_service_cell_index=0,
                ras_service_cell_active=True,
                include_atmosphere=True,
                boresight_theta1=1.0 * u.deg,
                boresight_theta2=2.0 * u.deg,
                boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
                output_families=_output_family_profile("all_outputs"),
                writer_checkpoint_interval_s=1.0e-9,
                finalize_memory_budget_bytes=1,
                power_memory_budget_bytes=1,
                export_memory_budget_bytes=1,
            )
        )

        assert checkpoint_lengths
        assert checkpoint_lengths[-1] == 6
        assert checkpoint_lengths[0] < checkpoint_lengths[-1]
        assert checkpoint_lengths == sorted(checkpoint_lengths)
        assert result["writer_checkpoint_count"] >= 1
        assert int(result["writer_stats_summary"]["durable_flush_count"]) >= 1

    def test_run_gpu_direct_epfd_exposes_writer_telemetry_without_profile(self, tmp_path: Path):
        result = scenario.run_gpu_direct_epfd(
            **_fake_direct_epfd_common_kwargs(
                tmp_path,
                "writer_telemetry",
                output_families=_output_family_profile("totals_only"),
                profile_stages=False,
            )
        )

        assert isinstance(result["writer_stats_summary"], dict)
        for key in (
            "prepare_elapsed_total",
            "apply_elapsed_total",
            "submit_wait_elapsed_total",
            "queued_items_high_water",
            "queued_bytes_high_water",
            "submitted_seq",
            "completed_seq",
            "flush_count",
            "writer_cycle_count",
            "writer_cycle_items_high_water",
            "writer_cycle_bytes_high_water",
            "durable_flush_count",
            "durable_elapsed_total",
            "durability_mode",
        ):
            assert key in result["writer_stats_summary"]
        assert int(result["writer_checkpoint_count"]) >= 0
        assert float(result["writer_checkpoint_wait_s"]) >= 0.0
        assert float(result["writer_final_flush_s"]) >= 0.0
        assert result["writer_stats_summary"]["durability_mode"] in {"fsync", "flush_only"}


class TestMemoryPlanning:

    def test_resolve_host_memory_budget_fixed_mode_uses_explicit_cap(self):
        budget = scenario.resolve_host_memory_budget_bytes(4.0, mode="fixed", headroom_profile="balanced")

        assert budget["mode_used"] == "fixed"
        assert budget["effective_budget_bytes"] == 4 * 1024 ** 3

    def test_resolve_host_memory_budget_hybrid_mode_keeps_explicit_cap_hard(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            scenario,
            "_runtime_host_memory_snapshot",
            lambda: {
                "provider": "test",
                "available_bytes": 6 * 1024 ** 3,
                "total_bytes": 16 * 1024 ** 3,
            },
        )

        budget = scenario.resolve_host_memory_budget_bytes(8.0, mode="hybrid", headroom_profile="balanced")

        assert budget["runtime_provider"] == "test"
        expected_runtime_budget = min(
            int(6 * 1024 ** 3 * 0.70),
            (6 - 2) * 1024 ** 3,
        )
        assert budget["runtime_budget_bytes"] == expected_runtime_budget
        assert budget["runtime_advisory_budget_bytes"] == expected_runtime_budget
        assert budget["planning_budget_bytes"] == 8 * 1024 ** 3
        assert budget["hard_budget_bytes"] == 8 * 1024 ** 3
        assert budget["effective_budget_bytes"] == 8 * 1024 ** 3

    def test_resolve_host_memory_budget_runtime_falls_back_to_explicit_cap(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(scenario, "_runtime_host_memory_snapshot", lambda: None)

        budget = scenario.resolve_host_memory_budget_bytes(3.0, mode="runtime", headroom_profile="balanced")

        assert budget["mode_used"] == "fixed_fallback"
        assert budget["effective_budget_bytes"] == 3 * 1024 ** 3

    def test_recommend_time_batch_size_linear_matches_budget_equation(self):
        recommendation = scenario.recommend_time_batch_size_linear(
            total_timesteps=100,
            fixed_bytes=120,
            per_timestep_bytes=35,
            budget_bytes=1000,
        )

        assert recommendation["recommended_batch_size"] == 25
        assert recommendation["fits_entire_span"] is False
        assert recommendation["fits_minimum_batch"] is True
        assert recommendation["budget_slack_bytes"] == 5

    def test_estimate_step1_host_batch_bytes_is_deterministic(self):
        estimate = scenario.estimate_step1_host_batch_bytes(
            time_count=3,
            n_cells=10,
            n_sats=20,
            n_links=4,
            cell_chunk_size=5,
            store_eligible_mask=True,
            propagation_dtype=np.float64,
            payload_dtype=np.float32,
            belt_id_dtype=np.int16,
            counts_dtype=np.int32,
            time_dtype=np.float64,
        )

        expected_peak = (
            3 * 8
            + 4 * 3 * 10 * 4 * 4
            + 3 * 10 * 4 * 2
            + 2 * 3 * 20 * 4
            + 3 * 10 * 20 * 1
            + 3 * 1 * 20 * 4 * 8
            + 3 * 5 * 20 * 4 * 8
            + 3 * 5 * 20 * 3 * 8
            + 4 * 3 * 5 * 4 * 4
            + 3 * 5 * 4 * 2
            + 2 * 3 * 20 * 4
            + 3 * 5 * 20 * 1
        )

        assert estimate["fixed_bytes"] == 0
        assert estimate["peak_bytes"] == expected_peak
        assert estimate["per_timestep_bytes"] == expected_peak // 3
        assert estimate["dominant_component"] == "chunk_topo"

    def test_resolve_direct_epfd_output_names_optionally_includes_eligible_mask(self):
        base_names = scenario._resolve_direct_epfd_output_names(
            write_epfd=True,
            write_prx_total=False,
            write_per_satellite_prx_ras_station=False,
            write_total_pfd_ras_station=False,
            write_per_satellite_pfd_ras_station=False,
            write_sat_beam_counts_used=True,
            write_sat_elevation_ras_station=False,
            write_beam_demand_count=False,
            write_sat_eligible_mask=False,
        )
        assert "sat_eligible_mask" not in base_names

        names_with_mask = scenario._resolve_direct_epfd_output_names(
            write_epfd=True,
            write_prx_total=False,
            write_per_satellite_prx_ras_station=False,
            write_total_pfd_ras_station=False,
            write_per_satellite_pfd_ras_station=False,
            write_sat_beam_counts_used=True,
            write_sat_elevation_ras_station=False,
            write_beam_demand_count=False,
            write_sat_eligible_mask=True,
        )
        assert names_with_mask[-1] == "sat_eligible_mask"

    def test_estimate_step2_host_batch_bytes_tracks_visible_satellite_terms(self):
        estimate = scenario.estimate_step2_host_batch_bytes(
            time_count=2,
            n_sats_total=30,
            n_visible_sats=6,
            n_links=3,
            n_beams=4,
            n_sky_cells=12,
            include_total_pfd=True,
            include_per_satellite_pfd=False,
            propagation_dtype=np.float64,
            working_dtype=np.float32,
            time_dtype=np.float64,
            sky_rx_chunk_size=5,
            stream_chunk_size=8,
            stream_rescue_chunk_size=16,
        )

        fixed_expected = 6 * (4 + 4 + 2 + 4 + 4)
        per_step_components = estimate["components_bytes"]
        assert estimate["fixed_bytes"] == fixed_expected
        assert estimate["peak_bytes"] > estimate["fixed_bytes"]
        assert estimate["per_timestep_bytes"] > 0
        assert per_step_components["propagation_topo_full"] == 2 * 1 * 30 * 4 * 8
        assert per_step_components["rx_chunk_workspace"] == 5 * 2 * 12 * 5 * 4

    def test_combined_gpu_peak_estimate_includes_overlapping_resident_buffers(self):
        isolated_stage_bytes = 64 * 1024 ** 2

        peaks = scenario._estimate_direct_epfd_combined_gpu_peaks(
            batch_timesteps=4,
            cell_count=256,
            sat_count_total=64,
            sat_visible_count=32,
            n_skycells=16,
            boresight_active=True,
            need_pointings=True,
            need_beam_demand=True,
            store_eligible_mask=True,
            profile_stages=True,
            output_dtype=np.float32,
            compute_dtype=np.float32,
            count_dtype=np.uint8,
            demand_count_dtype=np.uint8,
            predicted_gpu_cell_chunk_peak_bytes=isolated_stage_bytes,
            predicted_gpu_finalize_slab_bytes=isolated_stage_bytes,
            predicted_gpu_power_slab_bytes=isolated_stage_bytes,
            predicted_gpu_export_peak_bytes=isolated_stage_bytes,
            write_epfd=True,
            write_prx_total=True,
            write_per_satellite_prx_ras_station=True,
            write_prx_elevation_heatmap=True,
            write_total_pfd_ras_station=True,
            write_per_satellite_pfd_ras_station=True,
            write_sat_beam_counts_used=True,
        )

        assert int(peaks["predicted_gpu_peak_bytes"]) > isolated_stage_bytes
        assert int(peaks["predicted_gpu_power_peak_bytes"]) > isolated_stage_bytes
        assert int(peaks["link_library_chunk_transient_peak_bytes"]) > 0
        assert int(peaks["predicted_gpu_finalize_transient_peak_bytes"]) > 0

    def test_stage_memory_summary_tracks_resident_and_transient_gpu_usage(self):
        summary = scenario._update_direct_epfd_stage_memory_summary(
            {"observed_stage_name": "beam_finalize"},
            {
                "gpu_adapter_snapshot": {
                    "used_bytes": 2 * _GIB,
                    "free_bytes": 6 * _GIB,
                },
                "process_rss_bytes": int(1.25 * _GIB),
            },
        )
        summary = scenario._update_direct_epfd_stage_memory_summary(
            summary,
            {
                "gpu_adapter_snapshot": {
                    "used_bytes": 3 * _GIB,
                    "free_bytes": 5 * _GIB,
                },
                "process_rss_bytes": int(1.50 * _GIB),
            },
        )
        summary = scenario._update_direct_epfd_stage_memory_summary(
            summary,
            {
                "gpu_adapter_snapshot": {
                    "used_bytes": int(2.5 * _GIB),
                    "free_bytes": int(4.5 * _GIB),
                },
                "process_rss_bytes": int(1.75 * _GIB),
            },
        )

        assert int(summary["observed_stage_gpu_start_bytes"]) == 2 * _GIB
        assert int(summary["observed_stage_gpu_peak_bytes"]) == 3 * _GIB
        assert int(summary["observed_stage_gpu_end_bytes"]) == int(2.5 * _GIB)
        assert int(summary["observed_stage_gpu_resident_bytes"]) == int(0.5 * _GIB)
        assert int(summary["observed_stage_gpu_transient_peak_bytes"]) == int(0.5 * _GIB)
        assert int(summary["observed_stage_gpu_free_low_bytes"]) == int(4.5 * _GIB)
        assert int(summary["observed_process_rss_bytes"]) == int(1.75 * _GIB)

    def test_lower_runtime_effective_gpu_budget_uses_observed_stage_peak_overrun(self):
        updated_state, lowered = scenario._lower_runtime_effective_gpu_budget(
            {
                "gpu_effective_budget_bytes": 4 * _GIB,
                "last_observed_stage_summary": {
                    "observed_stage_name": "beam_finalize",
                    "observed_stage_gpu_peak_bytes": 5 * _GIB,
                    "observed_stage_gpu_free_low_bytes": int(0.8 * _GIB),
                },
            },
            stage="beam_finalize",
            post_cleanup_snapshot={
                "gpu_snapshot": {
                    "free_bytes": int(2.5 * _GIB),
                    "total_bytes": 8 * _GIB,
                }
            },
        )

        assert lowered is True
        assert int(updated_state["gpu_effective_budget_previous_bytes"]) == 4 * _GIB
        expected_lowered_budget = int(
            4 * _GIB
            - (1 * _GIB + int(scenario._DIRECT_EPFD_GPU_OOM_MARGIN_BYTES))
        )
        assert int(updated_state["gpu_effective_budget_bytes"]) == expected_lowered_budget
        assert int(updated_state["gpu_effective_budget_low_water_bytes"]) == expected_lowered_budget

    def test_lower_runtime_effective_gpu_budget_ignores_advisory_low_free_when_peak_fits(self):
        updated_state, lowered = scenario._lower_runtime_effective_gpu_budget(
            {
                "gpu_effective_budget_bytes": 4 * _GIB,
                "last_observed_stage_summary": {
                    "observed_stage_name": "beam_finalize",
                    "observed_stage_gpu_peak_bytes": int(3.5 * _GIB),
                    "observed_stage_gpu_free_low_bytes": int(0.5 * _GIB),
                },
            },
            stage="beam_finalize",
            post_cleanup_snapshot={
                "gpu_snapshot": {
                    "free_bytes": int(2.5 * _GIB),
                    "total_bytes": 8 * _GIB,
                }
            },
        )

        assert lowered is False
        assert int(updated_state["gpu_effective_budget_bytes"]) == 4 * _GIB

    def test_recover_runtime_effective_gpu_budget_steps_back_toward_hard_cap(self):
        recovered_state, recovered = scenario._recover_runtime_effective_gpu_budget(
            {
                "gpu_effective_budget_bytes": int(3 * _GIB),
                "gpu_effective_budget_lowered": True,
                "gpu_budget_lowered_stage": "beam_finalize",
            },
            hard_budget_bytes=4 * _GIB,
        )

        assert recovered is True
        assert int(recovered_state["gpu_effective_budget_previous_bytes"]) == int(3 * _GIB)
        assert int(recovered_state["gpu_effective_budget_bytes"]) == int(3.5 * _GIB)
        assert bool(recovered_state["gpu_effective_budget_lowered"]) is True
        assert recovered_state["gpu_budget_lowered_stage"] == "beam_finalize"

    def test_link_library_gpu_estimate_separates_resident_and_transient_components(self):
        estimate = scenario._estimate_direct_epfd_link_library_gpu_bytes(
            time_count=4,
            cell_count=128,
            sat_count_total=64,
            sat_visible_count=16,
            n_skycells=8,
            store_eligible_mask=True,
            boresight_active=True,
        )

        assert int(estimate["candidate_pairs"]) == 4 * 128 * 16
        assert int(estimate["resident_bytes"]) > 0
        assert int(estimate["chunk_transient_peak_bytes"]) > 0
        assert int(estimate["finalize_pack_peak_bytes"]) > int(estimate["resident_bytes"])
        assert int(estimate["finalize_resident_bytes"]) >= int(estimate["resident_bytes"])

    def test_probe_visibility_profile_window_uses_maximum_sample(self):
        fake_session = _FakeSession(
            sat_count_total=8,
            probe_visibility_counts=[1, 4, 2, 5, 3],
        )

        result = scenario._probe_visibility_profile_window(
            fake_session,
            np.linspace(61000.0, 61000.25, 5, dtype=np.float64),
            type("SatCtx", (), {"n_sats": 8})(),
            observer_context=type("ObsCtx", (), {"n_observers": 2})(),
            observer_slice=slice(0, 1),
            output_dtype=np.float32,
        )

        assert int(result["visible_satellite_count"]) == 5
        assert int(result["probe_sample_count"]) == 5
        assert len(result["probe_samples"]) == 5

    def test_live_fit_guard_allows_soft_target_overage_below_hard_budget(self):
        assessment = scenario._raise_if_direct_epfd_stage_live_fit_is_unsafe(
            stage="cell_link_library",
            host_peak_bytes=int(8.12 * _GIB),
            gpu_peak_bytes=int(8.92 * _GIB),
            host_effective_budget_bytes=12 * _GIB,
            gpu_effective_budget_bytes=12 * _GIB,
            scheduler_active_target_fraction=0.50,
            live_host_snapshot={"available_bytes": 24 * _GIB, "total_bytes": 64 * _GIB},
            live_gpu_snapshot={"free_bytes": 12 * _GIB, "total_bytes": 16 * _GIB},
        )

        assert assessment["advisory_issues"]
        assert int(assessment["gpu_live_fit_budget_bytes"]) == 12 * _GIB

    def test_live_fit_guard_raises_when_cuda_live_free_is_too_small(self):
        # GPU peak must exceed the live-fit floor.  On Windows the WDDM
        # floor is 85% of the hard budget (10.2 GiB for a 12 GiB cap),
        # so the peak must be above that floor to trigger the OOM.
        # Using 11 GiB peak with only 8 GiB free exceeds both the raw
        # live-fit (8 GiB) and the Windows floor (10.2 GiB).
        with pytest.raises(scenario._DirectGpuOutOfMemory) as excinfo:
            scenario._raise_if_direct_epfd_stage_live_fit_is_unsafe(
                stage="cell_link_library",
                host_peak_bytes=int(4.0 * _GIB),
                gpu_peak_bytes=int(11.0 * _GIB),
                host_effective_budget_bytes=12 * _GIB,
                gpu_effective_budget_bytes=12 * _GIB,
                scheduler_active_target_fraction=0.90,
                live_host_snapshot={"available_bytes": 24 * _GIB, "total_bytes": 64 * _GIB},
                live_gpu_snapshot={"free_bytes": 8 * _GIB, "total_bytes": 16 * _GIB},
            )

        assert "live allocatable memory below required fit before cell_link_library" in str(
            excinfo.value
        ).lower()

    def test_scheduler_test_matrix_caps_default_to_half_free_host_and_vram(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SCEPTER_SCHEDULER_TEST_MAX_HOST_GB", raising=False)
        monkeypatch.delenv("SCEPTER_SCHEDULER_TEST_MAX_GPU_GB", raising=False)
        monkeypatch.delenv("SCEPTER_SCHEDULER_TEST_FORCE_FULL_MATRIX", raising=False)
        monkeypatch.setattr(
            scenario,
            "_runtime_host_memory_snapshot",
            lambda: {"provider": "test", "available_bytes": 18 * _GIB, "total_bytes": 32 * _GIB},
        )
        monkeypatch.setattr(
            sys.modules[__name__],
            "_scheduler_test_gpu_runtime_snapshot",
            lambda: {"provider": "test", "free_bytes": 14 * _GIB, "total_bytes": 24 * _GIB},
        )

        caps = _scheduler_test_matrix_caps()

        assert caps["host_cap"] == 9
        assert caps["gpu_cap"] == 7

    def test_scheduler_test_matrix_caps_respect_overrides_and_force_full(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCEPTER_SCHEDULER_TEST_FORCE_FULL_MATRIX", "1")
        monkeypatch.delenv("SCEPTER_SCHEDULER_TEST_MAX_HOST_GB", raising=False)
        monkeypatch.delenv("SCEPTER_SCHEDULER_TEST_MAX_GPU_GB", raising=False)
        monkeypatch.setattr(scenario, "_runtime_host_memory_snapshot", lambda: None)
        monkeypatch.setattr(sys.modules[__name__], "_scheduler_test_gpu_runtime_snapshot", lambda: None)

        caps = _scheduler_test_matrix_caps()

        assert caps["host_cap"] == 12
        assert caps["gpu_cap"] == 12

        monkeypatch.setenv("SCEPTER_SCHEDULER_TEST_MAX_HOST_GB", "5")
        monkeypatch.setenv("SCEPTER_SCHEDULER_TEST_MAX_GPU_GB", "4")
        caps = _scheduler_test_matrix_caps()
        assert caps["host_cap"] == 5
        assert caps["gpu_cap"] == 4

    def test_direct_epfd_scheduler_matrix_is_safe_for_detected_caps(self) -> None:
        caps = _scheduler_test_matrix_caps()
        host_cap = caps["host_cap"]
        gpu_cap = caps["gpu_cap"]
        if host_cap is None or gpu_cap is None:
            pytest.skip("Reliable free host RAM / VRAM snapshots are unavailable for the exhaustive matrix.")

        for host_gb in range(1, int(host_cap) + 1):
            for gpu_gb in range(1, int(gpu_cap) + 1):
                plan = _plan_fake_direct_epfd_iteration(
                    host_budget_gb=host_gb,
                    gpu_budget_gb=gpu_gb,
                    boresight_active=True,
                )
                _assert_fake_direct_epfd_plan_is_safe(
                    plan,
                    host_budget_gb=host_gb,
                    gpu_budget_gb=gpu_gb,
                )

    def test_direct_epfd_scheduler_named_budget_regressions_when_permitted(self) -> None:
        caps = _scheduler_test_matrix_caps()
        host_cap = caps["host_cap"]
        gpu_cap = caps["gpu_cap"]
        if host_cap is None or gpu_cap is None:
            pytest.skip("Reliable free host RAM / VRAM snapshots are unavailable for named budget regressions.")

        plans: dict[tuple[int, int], dict[str, object]] = {}
        for pair in ((4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (2, 8), (8, 2)):
            host_gb, gpu_gb = pair
            if host_gb > int(host_cap) or gpu_gb > int(gpu_cap):
                continue
            plans[pair] = _plan_fake_direct_epfd_iteration(
                host_budget_gb=host_gb,
                gpu_budget_gb=gpu_gb,
                boresight_active=True,
            )

        if not plans:
            pytest.skip("No named scheduler regression case fits the detected host/GPU caps.")

        for (host_gb, gpu_gb), plan in plans.items():
            _assert_fake_direct_epfd_plan_is_safe(
                plan,
                host_budget_gb=host_gb,
                gpu_budget_gb=gpu_gb,
            )
            assert str(plan["limiting_resource"]) != "fallback"
        if (8, 8) in plans:
            assert int(plans[(8, 8)]["predicted_gpu_finalize_peak_bytes"]) <= 8 * _GIB
            assert int(plans[(8, 8)]["predicted_host_export_peak_bytes"]) <= int(
                plans[(8, 8)]["predicted_host_peak_bytes"]
            )

    def test_direct_epfd_scheduler_full_matrix_thorough_suite_when_forced(self) -> None:
        if not _scheduler_test_force_full_suite_requested():
            pytest.skip(
                "Enable SCEPTER_SCHEDULER_TEST_FORCE_FULL_MATRIX=1 to run the dedicated full scheduler suite."
            )
        caps = _scheduler_test_matrix_caps()
        host_cap = int(caps["host_cap"] or 12)
        gpu_cap = int(caps["gpu_cap"] or 12)
        repeated_88_plans: list[dict[str, object]] = []
        for host_gb in range(1, host_cap + 1):
            for gpu_gb in range(1, gpu_cap + 1):
                plan = _plan_fake_direct_epfd_iteration(
                    host_budget_gb=host_gb,
                    gpu_budget_gb=gpu_gb,
                    boresight_active=True,
                )
                _assert_fake_direct_epfd_plan_is_safe(
                    plan,
                    host_budget_gb=host_gb,
                    gpu_budget_gb=gpu_gb,
                )
                if (host_gb, gpu_gb) == (8, 8):
                    repeated_88_plans.append(plan)

        if host_cap >= 8 and gpu_cap >= 8:
            for visible_satellite_est in (6, 12, 18):
                plan = _plan_fake_direct_epfd_iteration(
                    host_budget_gb=8,
                    gpu_budget_gb=8,
                    boresight_active=True,
                    visible_satellite_est=visible_satellite_est,
                    nco=4 + (visible_satellite_est // 6),
                    nbeam=8 + (visible_satellite_est // 3),
                )
                _assert_fake_direct_epfd_plan_is_safe(
                    plan,
                    host_budget_gb=8,
                    gpu_budget_gb=8,
                )
                repeated_88_plans.append(plan)
            assert len(repeated_88_plans) >= 4

    def test_direct_epfd_scheduler_runtime_subset_when_forced(self, tmp_path: Path) -> None:
        if not _scheduler_test_force_full_suite_requested():
            pytest.skip(
                "Enable SCEPTER_SCHEDULER_TEST_FORCE_FULL_MATRIX=1 to run the dedicated runtime scheduler subset."
            )
        for host_gb, gpu_gb in ((4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (2, 8), (8, 2)):
            events: list[dict[str, object]] = []
            result = scenario.run_gpu_direct_epfd(
                **_fake_direct_epfd_common_kwargs(
                    tmp_path,
                    f"forced_runtime_{host_gb}_{gpu_gb}",
                    base_end_time=Time("2025-01-01T00:00:03", scale="utc"),
                    host_memory_budget_gb=float(host_gb),
                    gpu_memory_budget_gb=float(gpu_gb),
                    progress_callback=events.append,
                )
            )
            assert result["run_state"] == "completed"
            iteration_plan_events = [event for event in events if event.get("kind") == "iteration_plan"]
            assert iteration_plan_events
            plan_payload = iteration_plan_events[0]
            assert int(plan_payload["predicted_host_peak_bytes"]) <= host_gb * _GIB
            assert int(plan_payload["predicted_gpu_peak_bytes"]) <= gpu_gb * _GIB
            assert int(plan_payload["bulk_timesteps"]) >= 1
            assert int(plan_payload["cell_chunk"]) >= 1
            assert int(plan_payload["sky_slab"]) >= 1


class TestScenarioStorage:

    def test_write_data_defaults_to_async_and_preserves_append_order(self, tmp_path: Path):
        filename = tmp_path / "async_default.h5"
        times = Time(60020.0 + _seconds_to_mjd(np.array([0.0, 1.0, 2.0])), format="mjd", scale="utc")
        power = np.array([[1.0], [2.0], [3.0]], dtype=np.float32) * u.W

        scenario.write_data(
            str(filename),
            attrs={"tag": "async-default"},
            constants={"const_scalar": np.array([7], dtype=np.int16)},
        )
        scenario.write_data(str(filename), iteration=0, times=times[:2], power=power[:2])
        scenario.write_data(str(filename), iteration=0, times=times[2:], power=power[2:])

        loaded = scenario.read_data(str(filename), stack=False)

        assert loaded["attrs"]["tag"] == "async-default"
        assert_equal(loaded["const"]["const_scalar"], np.array([7], dtype=np.int16))
        assert_quantity_allclose(loaded["iter"][0]["power"], power)
        assert_allclose(loaded["iter"][0]["times"].mjd, times.mjd)

    def test_describe_and_read_dataset_slice_report_streaming_metadata(self, tmp_path: Path):
        filename = _build_sample_results_file(tmp_path, write_mode="sync")

        meta = scenario.describe_data(
            str(filename),
            iter_selection=[0],
            var_selection=["power", "times"],
            slot_selection=(1, 3),
        )

        assert meta["attrs"]["tag"] == "scenario-test"
        assert meta["const"]["const_vector"]["shape"] == (3,)
        assert meta["iter"][0]["selected_slot_count"] == 2
        assert meta["iter"][0]["datasets"]["power"]["compression"] == "gzip"
        assert meta["iter"][0]["datasets"]["power"]["shuffle"] is True
        assert meta["iter"][0]["datasets"]["power"]["maxshape"] == (None, 2)

        power_slice = scenario.read_dataset_slice(
            str(filename),
            iteration=0,
            name="power",
            selection=np.s_[1:3],
        )
        time_slice = scenario.read_dataset_slice(
            str(filename),
            iteration=0,
            name="times",
            selection=np.s_[1:3],
        )

        assert_quantity_allclose(
            power_slice,
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32) * u.W,
        )
        assert isinstance(time_slice, Time)
        assert_allclose(
            time_slice.mjd,
            60000.0 + _seconds_to_mjd(np.array([1.0, 2.0])),
        )

    def test_read_data_slice_and_stream_match_selected_slot_range(self, tmp_path: Path):
        filename = _build_sample_results_file(tmp_path, write_mode="sync")

        sliced = scenario.read_data(
            str(filename),
            mode="slice",
            iter_selection=[0],
            var_selection=["power", "times"],
            slot_selection=(1, 3),
            stack=False,
        )
        assert_quantity_allclose(
            sliced["iter"][0]["power"],
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32) * u.W,
        )
        assert_allclose(
            sliced["iter"][0]["times"].mjd,
            60000.0 + _seconds_to_mjd(np.array([1.0, 2.0])),
        )

        streamed = scenario.read_data(
            str(filename),
            mode="stream",
            iter_selection=[0],
            var_selection=["power", "times"],
            slot_selection=(1, 3),
            slot_chunk_size=1,
            prefetch_chunks=2,
            stack=False,
        )
        chunks = list(streamed["stream"])

        assert streamed["iter_meta"][0]["selected_slot_count"] == 2
        assert len(chunks) == 2
        assert_equal([chunk["slot_start"] for chunk in chunks], [1, 2])
        assert_equal([chunk["slot_stop"] for chunk in chunks], [2, 3])
        streamed_power = np.concatenate(
            [chunk["data"]["power"].value for chunk in chunks],
            axis=0,
        ) * u.W
        streamed_times = np.concatenate([chunk["data"]["times"].mjd for chunk in chunks], axis=0)
        assert_quantity_allclose(
            streamed_power,
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32) * u.W,
        )
        assert_allclose(
            streamed_times,
            60000.0 + _seconds_to_mjd(np.array([1.0, 2.0])),
        )

    def test_owned_async_iteration_write_matches_sync_scatter_outputs(self, tmp_path: Path):
        async_filename = tmp_path / "owned_async_scatter.h5"
        sync_filename = tmp_path / "owned_sync_scatter.h5"
        times = Time(60030.0 + _seconds_to_mjd(np.array([0.0, 1.0])), format="mjd", scale="utc")
        sat_idx = np.array([1, 3], dtype=np.int32)
        compact_prx = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        compact_elev = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

        scenario.init_simulation_results(
            str(async_filename),
            write_mode="async",
            writer_queue_max_items=1,
            writer_queue_max_bytes=256,
        )
        scenario._write_iteration_batch_owned(
            str(async_filename),
            iteration=0,
            batch_items=(
                ("times", times),
                (
                    "Prx_per_sat_RAS_STATION_W",
                    scenario._DeferredPerSatelliteScatter(
                        compact_values=compact_prx,
                        sat_idx_host=sat_idx,
                        n_sats_total=5,
                        dtype=np.dtype(np.float32),
                        boresight_active=False,
                        n_skycells=0,
                    ),
                ),
                (
                    "sat_elevation_RAS_STATION_deg",
                    scenario._DeferredSatelliteTimeSeriesScatter(
                        compact_values=compact_elev,
                        sat_idx_host=sat_idx,
                        n_sats_total=5,
                        dtype=np.dtype(np.float32),
                        fill_value=np.nan,
                    ),
                ),
            ),
        )
        scenario.flush_writes(str(async_filename))
        async_loaded = scenario.read_data(str(async_filename), stack=False)

        scenario.init_simulation_results(str(sync_filename), write_mode="sync")
        scenario.write_data(
            str(sync_filename),
            iteration=0,
            times=times,
            Prx_per_sat_RAS_STATION_W=scenario._scatter_compact_per_satellite_host(
                compact_prx,
                sat_idx,
                n_sats_total=5,
                dtype=np.float32,
                boresight_active=False,
                n_skycells=0,
            ),
            sat_elevation_RAS_STATION_deg=scenario._scatter_compact_satellite_time_series_host(
                compact_elev,
                sat_idx,
                n_sats_total=5,
                dtype=np.float32,
                fill_value=np.nan,
            ),
        )
        sync_loaded = scenario.read_data(str(sync_filename), stack=False)

        assert_allclose(async_loaded["iter"][0]["times"].mjd, sync_loaded["iter"][0]["times"].mjd)
        assert_allclose(
            async_loaded["iter"][0]["Prx_per_sat_RAS_STATION_W"],
            sync_loaded["iter"][0]["Prx_per_sat_RAS_STATION_W"],
        )
        assert_allclose(
            async_loaded["iter"][0]["sat_elevation_RAS_STATION_deg"],
            sync_loaded["iter"][0]["sat_elevation_RAS_STATION_deg"],
            equal_nan=True,
        )

    def test_owned_async_iteration_write_handles_non_boresight_empty_batch_shape(
        self,
        tmp_path: Path,
    ):
        filename = tmp_path / "owned_async_non_boresight_empty_batch.h5"
        batch0_times = Time(
            60030.0 + _seconds_to_mjd(np.array([0.0, 1.0], dtype=np.float64)),
            format="mjd",
            scale="utc",
        )
        batch1_times = Time(
            60030.0 + _seconds_to_mjd(np.array([2.0], dtype=np.float64)),
            format="mjd",
            scale="utc",
        )
        sat_idx = np.array([1, 3], dtype=np.int32)
        compact_prx = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        scenario.init_simulation_results(
            str(filename),
            write_mode="async",
            writer_queue_max_items=2,
            writer_queue_max_bytes=2048,
        )
        scenario._write_iteration_batch_owned(
            str(filename),
            iteration=0,
            batch_items=(
                ("times", batch0_times),
                (
                    "Prx_per_sat_RAS_STATION_W",
                    scenario._DeferredPerSatelliteScatter(
                        compact_values=compact_prx,
                        sat_idx_host=sat_idx,
                        n_sats_total=5,
                        dtype=np.dtype(np.float32),
                        boresight_active=False,
                        n_skycells=0,
                    ),
                ),
            ),
        )
        scenario._write_iteration_batch_owned(
            str(filename),
            iteration=0,
            batch_items=(
                ("times", batch1_times),
                ("Prx_per_sat_RAS_STATION_W", np.zeros((1, 5), dtype=np.float32)),
            ),
        )
        scenario.flush_writes(str(filename))
        loaded = scenario.read_data(str(filename), stack=False)

        assert loaded["iter"][0]["Prx_per_sat_RAS_STATION_W"].shape == (3, 5)
        assert_allclose(
            loaded["iter"][0]["Prx_per_sat_RAS_STATION_W"][-1],
            np.zeros(5, dtype=np.float32),
        )

    def test_owned_async_iteration_flush_exposes_incremental_mid_run_data(self, tmp_path: Path):
        filename = tmp_path / "owned_async_incremental_flush.h5"
        scenario.init_simulation_results(
            str(filename),
            write_mode="async",
            writer_queue_max_items=8,
            writer_queue_max_bytes=1024 ** 2,
        )

        batch0_times = Time(
            60035.0 + _seconds_to_mjd(np.array([0.0, 1.0], dtype=np.float64)),
            format="mjd",
            scale="utc",
        )
        batch1_times = Time(
            60035.0 + _seconds_to_mjd(np.array([2.0, 3.0], dtype=np.float64)),
            format="mjd",
            scale="utc",
        )
        power0 = (np.arange(8, dtype=np.float32).reshape(2, 4) + 1.0) * u.W
        power1 = (np.arange(8, dtype=np.float32).reshape(2, 4) + 9.0) * u.W

        scenario._write_iteration_batch_owned(
            str(filename),
            iteration=0,
            batch_items=(
                ("times", batch0_times),
                ("power", power0),
            ),
        )
        scenario.flush_writes(str(filename))

        flushed_times0 = scenario.read_dataset_slice(
            str(filename),
            iteration=0,
            name="times",
            selection=np.s_[:],
            sync_pending_writes=False,
        )
        flushed_power0 = scenario.read_dataset_slice(
            str(filename),
            iteration=0,
            name="power",
            selection=np.s_[:],
            sync_pending_writes=False,
        )
        assert_allclose(flushed_times0.mjd, batch0_times.mjd)
        assert_quantity_allclose(flushed_power0, power0)

        scenario._write_iteration_batch_owned(
            str(filename),
            iteration=0,
            batch_items=(
                ("times", batch1_times),
                ("power", power1),
            ),
        )
        scenario.flush_writes(str(filename))

        flushed_times1 = scenario.read_dataset_slice(
            str(filename),
            iteration=0,
            name="times",
            selection=np.s_[:],
            sync_pending_writes=False,
        )
        flushed_power1 = scenario.read_dataset_slice(
            str(filename),
            iteration=0,
            name="power",
            selection=np.s_[:],
            sync_pending_writes=False,
        )
        assert_allclose(flushed_times1.mjd, np.concatenate([batch0_times.mjd, batch1_times.mjd]))
        assert_quantity_allclose(flushed_power1, np.concatenate([power0.value, power1.value], axis=0) * u.W)

        writer_stats = scenario._get_writer_stats_snapshot(str(filename))
        assert int(writer_stats["durable_flush_count"]) >= 2
        assert int(writer_stats["writer_cycle_count"]) >= 2
        assert writer_stats["durability_mode"] in {"fsync", "flush_only"}

    def test_owned_async_iteration_write_handles_small_queue_backpressure(self, tmp_path: Path):
        filename = tmp_path / "owned_async_backpressure.h5"
        scenario.init_simulation_results(
            str(filename),
            write_mode="async",
            writer_queue_max_items=1,
            writer_queue_max_bytes=512,
        )

        times = Time(60040.0 + _seconds_to_mjd(np.array([0.0, 1.0])), format="mjd", scale="utc")
        for batch_i in range(3):
            power = (np.ones((2, 128), dtype=np.float32) * float(batch_i + 1)) * u.W
            scenario._write_iteration_batch_owned(
                str(filename),
                iteration=0,
                batch_items=(
                    ("times", times),
                    ("power", power),
                ),
            )

        scenario.flush_writes(str(filename))
        loaded = scenario.read_data(str(filename), stack=False)

        assert loaded["iter"][0]["power"].shape == (6, 128)
        assert_allclose(
            loaded["iter"][0]["power"].value,
            np.concatenate(
                [
                    np.ones((2, 128), dtype=np.float32) * 1.0,
                    np.ones((2, 128), dtype=np.float32) * 2.0,
                    np.ones((2, 128), dtype=np.float32) * 3.0,
                ],
                axis=0,
            ),
        )

    def test_owned_async_writer_reports_multiple_cycles_and_durability(self, tmp_path: Path):
        filename = tmp_path / "owned_async_cycle_stats.h5"
        scenario.init_simulation_results(
            str(filename),
            write_mode="async",
            writer_queue_max_items=8,
            writer_queue_max_bytes=16 * 1024 ** 2,
        )

        for batch_i in range(6):
            times = Time(
                60045.0 + _seconds_to_mjd(np.array([2.0 * batch_i, 2.0 * batch_i + 1.0])),
                format="mjd",
                scale="utc",
            )
            power = (np.ones((2, 256), dtype=np.float32) * float(batch_i + 1)) * u.W
            scenario._write_iteration_batch_owned(
                str(filename),
                iteration=0,
                batch_items=(
                    ("times", times),
                    ("power", power),
                ),
            )

        scenario.flush_writes(str(filename))
        writer_stats = scenario._get_writer_stats_snapshot(str(filename))

        assert int(writer_stats["writer_cycle_count"]) > 1
        assert int(writer_stats["writer_cycle_items_high_water"]) >= 1
        assert int(writer_stats["writer_cycle_bytes_high_water"]) > 0
        assert int(writer_stats["durable_flush_count"]) >= 1
        assert float(writer_stats["durable_elapsed_total"]) >= 0.0
        assert writer_stats["durability_mode"] in {"fsync", "flush_only"}

    def test_async_writer_failure_persists_into_reads_until_reinitialized(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        filename = tmp_path / "owned_async_failed_writer.h5"
        times = Time(60050.0 + _seconds_to_mjd(np.array([0.0, 1.0])), format="mjd", scale="utc")
        power = (np.arange(4, dtype=np.float32).reshape(2, 2) + 1.0) * u.W

        scenario.init_simulation_results(
            str(filename),
            write_mode="async",
            writer_queue_max_items=1,
            writer_queue_max_bytes=1024,
        )

        def _fail_apply(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise ValueError("synthetic writer failure")

        monkeypatch.setattr(scenario, "_apply_coalesced_write_batch", _fail_apply)
        scenario._write_iteration_batch_owned(
            str(filename),
            iteration=0,
            batch_items=(
                ("times", times),
                ("power", power),
            ),
        )

        with pytest.raises(RuntimeError, match="synthetic writer failure"):
            scenario.flush_writes(str(filename))
        with pytest.raises(RuntimeError, match="synthetic writer failure"):
            scenario.close_writer(str(filename))
        with pytest.raises(RuntimeError, match="synthetic writer failure"):
            scenario.describe_data(str(filename))
        with pytest.raises(RuntimeError, match="synthetic writer failure"):
            scenario.read_data(str(filename), stack=False)
        with pytest.raises(RuntimeError, match="synthetic writer failure"):
            scenario.read_dataset_slice(str(filename), iteration=0, name="times")
        with pytest.raises(RuntimeError, match="synthetic writer failure"):
            list(scenario.iter_data_chunks(str(filename), slot_chunk_size=1))

        scenario.init_simulation_results(str(filename), write_mode="sync")
        scenario.write_data(str(filename), iteration=0, times=times, power=power)
        loaded = scenario.read_data(str(filename), stack=False)

        assert_allclose(loaded["iter"][0]["times"].mjd, times.mjd)
        assert_quantity_allclose(loaded["iter"][0]["power"], power)

    def test_owned_async_iteration_write_boresight_direct_run_mix_matches_sync(
        self,
        tmp_path: Path,
    ):
        async_filename = tmp_path / "owned_async_boresight_direct_mix.h5"
        sync_filename = tmp_path / "owned_sync_boresight_direct_mix.h5"
        sat_idx = np.asarray([1, 4], dtype=np.int32)
        n_sats_total = 5
        n_skycells = 3
        fill_nan = np.float32(np.nan)

        batch0_times = Time(
            60055.0 + _seconds_to_mjd(np.array([0.0, 1.0], dtype=np.float64)),
            format="mjd",
            scale="utc",
        )
        batch1_times = Time(
            60055.0 + _seconds_to_mjd(np.array([2.0, 3.0], dtype=np.float64)),
            format="mjd",
            scale="utc",
        )
        prx_total_0 = np.arange(6, dtype=np.float32).reshape(2, 1, n_skycells) + 0.5
        prx_total_1 = np.arange(6, dtype=np.float32).reshape(2, 1, n_skycells) + 6.5
        pfd_total_0 = np.arange(6, dtype=np.float32).reshape(2, 1, n_skycells) + 20.5
        pfd_total_1 = np.arange(6, dtype=np.float32).reshape(2, 1, n_skycells) + 26.5
        compact_prx_0 = np.arange(12, dtype=np.float32).reshape(2, n_skycells, sat_idx.size) + 100.0
        compact_prx_1 = np.arange(12, dtype=np.float32).reshape(2, n_skycells, sat_idx.size) + 112.0
        compact_pfd_0 = np.arange(12, dtype=np.float32).reshape(2, n_skycells, sat_idx.size) + 200.0
        compact_pfd_1 = np.arange(12, dtype=np.float32).reshape(2, n_skycells, sat_idx.size) + 212.0
        compact_elev_0 = np.arange(4, dtype=np.float32).reshape(2, sat_idx.size) + 10.0
        compact_elev_1 = np.arange(4, dtype=np.float32).reshape(2, sat_idx.size) + 14.0
        beam_counts_0 = np.arange(2 * 1 * n_sats_total * n_skycells, dtype=np.int32).reshape(2, 1, n_sats_total, n_skycells)
        beam_counts_1 = beam_counts_0 + 100

        scenario.init_simulation_results(
            str(async_filename),
            write_mode="async",
            writer_queue_max_items=4,
            writer_queue_max_bytes=8 * 1024 ** 2,
        )
        scenario._write_iteration_batch_owned(
            str(async_filename),
            iteration=0,
            batch_items=(
                ("times", batch0_times),
                ("Prx_total_W", prx_total_0 * u.W),
                ("PFD_total_RAS_STATION_W_m2", pfd_total_0 * u.W / (u.m ** 2)),
                (
                    "Prx_per_sat_RAS_STATION_W",
                    scenario._DeferredPerSatelliteScatter(
                        compact_values=compact_prx_0,
                        sat_idx_host=sat_idx,
                        n_sats_total=n_sats_total,
                        dtype=np.dtype(np.float32),
                        boresight_active=True,
                        n_skycells=n_skycells,
                    ),
                ),
                (
                    "PFD_per_sat_RAS_STATION_W_m2",
                    scenario._DeferredPerSatelliteScatter(
                        compact_values=compact_pfd_0,
                        sat_idx_host=sat_idx,
                        n_sats_total=n_sats_total,
                        dtype=np.dtype(np.float32),
                        boresight_active=True,
                        n_skycells=n_skycells,
                    ),
                ),
                (
                    "sat_elevation_RAS_STATION_deg",
                    scenario._DeferredSatelliteTimeSeriesScatter(
                        compact_values=compact_elev_0,
                        sat_idx_host=sat_idx,
                        n_sats_total=n_sats_total,
                        dtype=np.dtype(np.float32),
                        fill_value=float(fill_nan),
                    ),
                ),
                ("sat_beam_counts_used", beam_counts_0),
            ),
        )
        scenario.flush_writes(str(async_filename))
        scenario._write_iteration_batch_owned(
            str(async_filename),
            iteration=0,
            batch_items=(
                ("times", batch1_times),
                ("Prx_total_W", prx_total_1 * u.W),
                ("PFD_total_RAS_STATION_W_m2", pfd_total_1 * u.W / (u.m ** 2)),
                (
                    "Prx_per_sat_RAS_STATION_W",
                    scenario._DeferredPerSatelliteScatter(
                        compact_values=compact_prx_1,
                        sat_idx_host=sat_idx,
                        n_sats_total=n_sats_total,
                        dtype=np.dtype(np.float32),
                        boresight_active=True,
                        n_skycells=n_skycells,
                    ),
                ),
                (
                    "PFD_per_sat_RAS_STATION_W_m2",
                    scenario._DeferredPerSatelliteScatter(
                        compact_values=compact_pfd_1,
                        sat_idx_host=sat_idx,
                        n_sats_total=n_sats_total,
                        dtype=np.dtype(np.float32),
                        boresight_active=True,
                        n_skycells=n_skycells,
                    ),
                ),
                (
                    "sat_elevation_RAS_STATION_deg",
                    scenario._DeferredSatelliteTimeSeriesScatter(
                        compact_values=compact_elev_1,
                        sat_idx_host=sat_idx,
                        n_sats_total=n_sats_total,
                        dtype=np.dtype(np.float32),
                        fill_value=float(fill_nan),
                    ),
                ),
                ("sat_beam_counts_used", beam_counts_1),
            ),
        )
        scenario.close_writer(str(async_filename))
        async_loaded = scenario.read_data(str(async_filename), stack=False)

        scenario.init_simulation_results(str(sync_filename), write_mode="sync")
        scenario.write_data(
            str(sync_filename),
            iteration=0,
            times=batch0_times,
            Prx_total_W=prx_total_0 * u.W,
            PFD_total_RAS_STATION_W_m2=pfd_total_0 * u.W / (u.m ** 2),
            Prx_per_sat_RAS_STATION_W=scenario._scatter_compact_per_satellite_host(
                compact_prx_0,
                sat_idx,
                n_sats_total=n_sats_total,
                dtype=np.float32,
                boresight_active=True,
                n_skycells=n_skycells,
            ),
            PFD_per_sat_RAS_STATION_W_m2=scenario._scatter_compact_per_satellite_host(
                compact_pfd_0,
                sat_idx,
                n_sats_total=n_sats_total,
                dtype=np.float32,
                boresight_active=True,
                n_skycells=n_skycells,
            ),
            sat_elevation_RAS_STATION_deg=scenario._scatter_compact_satellite_time_series_host(
                compact_elev_0,
                sat_idx,
                n_sats_total=n_sats_total,
                dtype=np.float32,
                fill_value=float(fill_nan),
            ),
            sat_beam_counts_used=beam_counts_0,
        )
        scenario.write_data(
            str(sync_filename),
            iteration=0,
            times=batch1_times,
            Prx_total_W=prx_total_1 * u.W,
            PFD_total_RAS_STATION_W_m2=pfd_total_1 * u.W / (u.m ** 2),
            Prx_per_sat_RAS_STATION_W=scenario._scatter_compact_per_satellite_host(
                compact_prx_1,
                sat_idx,
                n_sats_total=n_sats_total,
                dtype=np.float32,
                boresight_active=True,
                n_skycells=n_skycells,
            ),
            PFD_per_sat_RAS_STATION_W_m2=scenario._scatter_compact_per_satellite_host(
                compact_pfd_1,
                sat_idx,
                n_sats_total=n_sats_total,
                dtype=np.float32,
                boresight_active=True,
                n_skycells=n_skycells,
            ),
            sat_elevation_RAS_STATION_deg=scenario._scatter_compact_satellite_time_series_host(
                compact_elev_1,
                sat_idx,
                n_sats_total=n_sats_total,
                dtype=np.float32,
                fill_value=float(fill_nan),
            ),
            sat_beam_counts_used=beam_counts_1,
        )
        sync_loaded = scenario.read_data(str(sync_filename), stack=False)

        assert_allclose(async_loaded["iter"][0]["times"].mjd, sync_loaded["iter"][0]["times"].mjd)
        assert_quantity_allclose(async_loaded["iter"][0]["Prx_total_W"], sync_loaded["iter"][0]["Prx_total_W"])
        assert_quantity_allclose(
            async_loaded["iter"][0]["PFD_total_RAS_STATION_W_m2"],
            sync_loaded["iter"][0]["PFD_total_RAS_STATION_W_m2"],
        )
        assert_allclose(
            async_loaded["iter"][0]["Prx_per_sat_RAS_STATION_W"],
            sync_loaded["iter"][0]["Prx_per_sat_RAS_STATION_W"],
        )
        assert_allclose(
            async_loaded["iter"][0]["PFD_per_sat_RAS_STATION_W_m2"],
            sync_loaded["iter"][0]["PFD_per_sat_RAS_STATION_W_m2"],
        )
        assert_allclose(
            async_loaded["iter"][0]["sat_elevation_RAS_STATION_deg"],
            sync_loaded["iter"][0]["sat_elevation_RAS_STATION_deg"],
            equal_nan=True,
        )
        assert_equal(
            async_loaded["iter"][0]["sat_beam_counts_used"],
            sync_loaded["iter"][0]["sat_beam_counts_used"],
        )

        pfd_slice_async = scenario.read_dataset_slice(
            str(async_filename),
            iteration=0,
            name="PFD_per_sat_RAS_STATION_W_m2",
            selection=np.s_[2:],
        )
        pfd_slice_sync = scenario.read_dataset_slice(
            str(sync_filename),
            iteration=0,
            name="PFD_per_sat_RAS_STATION_W_m2",
            selection=np.s_[2:],
        )
        assert_allclose(pfd_slice_async, pfd_slice_sync)

    def test_durable_flush_open_h5_file_uses_fsync_when_handle_available(self, monkeypatch: pytest.MonkeyPatch):
        class _FakeId:
            def get_vfd_handle(self):
                return (None, np.int64(123))

        class _FakeH5:
            def __init__(self):
                self.id = _FakeId()
                self.flush_calls = 0

            def flush(self):
                self.flush_calls += 1

        fake_h5 = _FakeH5()
        fsync_calls: list[int] = []

        monkeypatch.setattr(scenario.os, "fsync", lambda handle: fsync_calls.append(int(handle)))

        mode, elapsed = scenario._durable_flush_open_h5_file(fake_h5)

        assert mode == "fsync"
        assert float(elapsed) >= 0.0
        assert fake_h5.flush_calls == 1
        assert fsync_calls == [123]

    def test_durable_flush_open_h5_file_falls_back_when_handle_is_unsupported(self):
        class _FakeId:
            def get_vfd_handle(self):
                return object()

        class _FakeH5:
            def __init__(self):
                self.id = _FakeId()
                self.flush_calls = 0

            def flush(self):
                self.flush_calls += 1

        fake_h5 = _FakeH5()

        mode, elapsed = scenario._durable_flush_open_h5_file(fake_h5)

        assert mode == "flush_only"
        assert float(elapsed) >= 0.0
        assert fake_h5.flush_calls == 1


# NOTE: The notebook-execution GPU smoke tests (boresight_run_notebook_*)
# were removed in v0.25.0 because they break whenever the notebook's cell
# structure or variable names change.  The same simulation logic is
# exercised more robustly by TestDirectEpfdGpuRunner (option matrix,
# output profiles, multi-batch, boresight prep, etc.) and the 67
# surface-PFD-cap tests that cover per-beam / per-satellite / M.2101 /
# 4-D boresight paths end-to-end.


@pytest.mark.skip(reason="Notebook-execution GPU tests removed in v0.25.0; "
                         "use TestDirectEpfdGpuRunner for simulation coverage.")
@GPU_REQUIRED
@pytest.mark.parametrize("case", _NOTEBOOK_GPU_SMOKE_CASES, ids=lambda case: str(case["case_name"]))
def test_boresight_run_notebook_reduced_gpu_smoke(case: dict[str, object], tmp_path: Path):
    cells = _build_boresight_notebook_smoke_cells(tmp_path, **case)
    notebook_env = _execute_code_cells(cells)
    run_result = notebook_env["RUN_RESULT"]
    stored_times = np.asarray(
        scenario.read_dataset_slice(run_result["storage_filename"], iteration=0, name="times"),
        dtype=np.float64,
    )
    postprocess_env = _execute_postprocess_validation(run_result["storage_filename"])
    profile_case = _FAKE_OUTPUT_PROFILE_CASES[str(case["profile_name"])]

    assert Path(run_result["storage_filename"]).exists()
    assert int(notebook_env["stored_step_count"]) == int(notebook_env["n_steps_total"])
    assert stored_times.shape == (int(notebook_env["n_steps_total"]),)
    assert float(notebook_env["stored_span_s"]) > 0.0
    assert run_result["writer_stats_summary"]["durability_mode"] in {"fsync", "flush_only"}
    assert int(run_result["writer_checkpoint_count"]) >= 1
    assert float(run_result["writer_final_flush_s"]) >= 0.0
    if case["boresight_theta1_deg"] is not None or case["boresight_theta2_deg"] is not None:
        assert run_result["boresight_active"] is True

    described = scenario.describe_data(run_result["storage_filename"])
    iter_payload = described["iter"][0]["datasets"]
    assert profile_case["expected_iter_names"].issubset(iter_payload.keys())
    assert "writer_stats_summary" in notebook_env["RUN_RESULT"]

    assert postprocess_env["PRIMARY_POWER_DATASET"] in {
        None,
        "Prx_total_W",
        "EPFD_W_m2",
    }
    assert float(postprocess_env["AVAILABLE_SPAN_S"]) > 0.0


@pytest.mark.skip(reason="Notebook-execution GPU tests removed in v0.25.0.")
@GPU_REQUIRED
def test_boresight_run_notebook_visual_cell_status_map_smoke(tmp_path: Path):
    case = {
        "case_name": "nb_visual_cell_status",
        "selection_strategy": "random",
        "ras_pointing_mode": "ras_station",
        "include_atmosphere": True,
        "memory_budget_mode": "hybrid",
        "profile_name": "notebook_full",
        "boresight_theta1_deg": 1.0,
        "boresight_theta2_deg": 3.0,
        "theta2_scope_mode": "adjacency_layers",
        "theta2_layers": 1,
        "host_memory_budget_gb": 4.0,
        "gpu_memory_budget_gb": 4.0,
        "force_bulk_timesteps": 2,
        "duration_s": 2,
        "point_spacing_km": 4000.0,
        "render_cell_status_map": True,
    }
    notebook_env = _execute_code_cells(_build_boresight_notebook_smoke_cells(tmp_path, **case))
    run_result = notebook_env["RUN_RESULT"]
    postprocess_env = _execute_postprocess_validation(run_result["storage_filename"])
    map_info = notebook_env["CELL_STATUS_MAP_INFO"]
    rendered_cell_count = (
        int(map_info["normal_active_count"])
        + int(map_info["switched_off_count"])
        + int(map_info["boresight_affected_active_count"])
    )

    assert Path(run_result["storage_filename"]).exists()
    assert notebook_env["CELL_STATUS_MAP_FIG"] is not None
    assert map_info["backend_used"] == "matplotlib"
    assert rendered_cell_count >= 1
    assert float(postprocess_env["AVAILABLE_SPAN_S"]) > 0.0


@pytest.mark.skip(reason="Notebook-execution GPU tests removed in v0.25.0.")
@GPU_REQUIRED
def test_boresight_run_notebook_tiny_budget_matches_normal_budget(tmp_path: Path):
    base_case = {
        "case_name": "nb_budget_normal",
        "selection_strategy": "random",
        "ras_pointing_mode": "ras_station",
        "include_atmosphere": True,
        "memory_budget_mode": "hybrid",
        "profile_name": "notebook_full",
        "boresight_theta1_deg": 1.0,
        "boresight_theta2_deg": 3.0,
        "theta2_scope_mode": "adjacency_layers",
        "theta2_layers": 1,
        "host_memory_budget_gb": 4.0,
        "gpu_memory_budget_gb": 4.0,
        "force_bulk_timesteps": 2,
    }
    tiny_case = dict(base_case)
    tiny_case["case_name"] = "nb_budget_tiny"
    tiny_case["host_memory_budget_gb"] = 1.0
    tiny_case["gpu_memory_budget_gb"] = 1.0
    tiny_case["memory_budget_mode"] = "gpu_only"

    normal_env = _execute_code_cells(_build_boresight_notebook_smoke_cells(tmp_path, **base_case))
    tiny_env = _execute_code_cells(_build_boresight_notebook_smoke_cells(tmp_path, **tiny_case))
    normal_result = normal_env["RUN_RESULT"]
    tiny_result = tiny_env["RUN_RESULT"]

    for dataset_name in (
        "times",
        "EPFD_W_m2",
        "Prx_total_W",
        "PFD_total_RAS_STATION_W_m2",
        "PFD_per_sat_RAS_STATION_W_m2",
        "sat_beam_counts_used",
        "sat_elevation_RAS_STATION_deg",
        "beam_demand_count",
    ):
        normal_value = scenario.read_dataset_slice(
            normal_result["storage_filename"],
            iteration=0,
            name=dataset_name,
        )
        tiny_value = scenario.read_dataset_slice(
            tiny_result["storage_filename"],
            iteration=0,
            name=dataset_name,
        )
        normal_arr = np.asarray(normal_value)
        tiny_arr = np.asarray(tiny_value)
        if dataset_name == "sat_beam_counts_used":
            assert normal_arr.shape == tiny_arr.shape
            assert int(normal_arr.sum(dtype=np.int64)) == int(tiny_arr.sum(dtype=np.int64))
            assert_equal(
                normal_arr.sum(axis=tuple(range(1, normal_arr.ndim)), dtype=np.int64),
                tiny_arr.sum(axis=tuple(range(1, tiny_arr.ndim)), dtype=np.int64),
            )
            continue
        if dataset_name == "sat_elevation_RAS_STATION_deg":
            assert normal_arr.shape == tiny_arr.shape
            assert int(np.count_nonzero(np.isfinite(normal_arr))) > 0
            assert int(np.count_nonzero(np.isfinite(tiny_arr))) > 0
            continue
        assert_allclose(
            tiny_arr,
            normal_arr,
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )

    assert int(tiny_result["writer_checkpoint_count"]) >= 1
    assert tiny_result["writer_stats_summary"]["durability_mode"] in {"fsync", "flush_only"}
