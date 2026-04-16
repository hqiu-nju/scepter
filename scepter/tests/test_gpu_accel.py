#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import inspect
import json
import os
import threading
import time
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any
import importlib.machinery
import types

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth
from astropy.time import Time
from numpy.testing import assert_allclose, assert_equal
from pycraf import conversions as cnv
from pycraf.antenna import ras_pattern
from pycraf.geometry import true_angular_distance

import cysgp4
from cysgp4 import PyObserver, PyTle


def _install_numba_stub() -> None:
    if "numba" in sys.modules:
        return
    try:
        __import__("numba")
        return
    except Exception:
        pass

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

from scepter.angle_sampler import JointAngleSampler
from scepter.antenna import calculate_beamwidth_1d, s_1528_rec1_4_pattern_amend
from scepter import gpu_accel, satsim, scenario, tleforger
from scepter.skynet import pointgen_S_1586_1


CUDA_AVAILABLE = gpu_accel.cuda is not None and bool(gpu_accel.cuda.is_available())
GPU_REQUIRED = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")

NEAR_TLES = [
    PyTle(
        "ISS",
        "1 25544U 98067A   25001.00000000  .00016717  00000+0  10270-3 0  9991",
        "2 25544  51.6421 164.6866 0003884 276.1957 170.2534 15.50057708487109",
    ),
    PyTle(
        "HST",
        "1 20580U 90037B   25001.00000000  .00000850  00000+0  36266-4 0  9992",
        "2 20580  28.4692 116.4136 0002805  23.6494 336.4611 15.09201640374246",
    ),
]
DEEP_TLE = PyTle(
    "EUTELSAT 1-F1",
    "1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627",
    "2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093",
)


def _example_observers() -> list[PyObserver]:
    return [
        PyObserver(21.443611, -30.712777, 1.052),
        PyObserver(10.0, 45.0, 0.2),
    ]


def _example_mjds() -> np.ndarray:
    return np.array([60676.0, 60676.01, 60676.02], dtype=np.float64)


def _step2_sampler_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    for candidate in ("1.13_US_System_B525_random.npz", "1_13_US_System_B525_random.npz"):
        path = root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError("EPFDflow sampler artifact was not found in the repository root.")


def _load_step2_sampler() -> JointAngleSampler:
    try:
        return JointAngleSampler.load(str(_step2_sampler_path()))
    except FileNotFoundError:
        pytest.skip("EPFDflow sampler artifact not present in this checkout.")


def _s1586_cell_center_deg(cell_id: int) -> tuple[float, float]:
    ring_offsets = gpu_accel._S1586_RING_OFFSETS
    ring_counts = gpu_accel._S1586_RING_COUNTS
    az_steps = gpu_accel._S1586_AZ_STEPS_DEG
    ring = int(np.searchsorted(ring_offsets[1:], int(cell_id), side="right"))
    az_bin = int(cell_id) - int(ring_offsets[ring])
    az_step = float(az_steps[ring])
    el_low = 3.0 * ring
    return (az_bin + 0.5) * az_step, el_low + 1.5


def _cpu_expected_sampler_sources(
    sampler: JointAngleSampler,
    sat_azimuth_deg: np.ndarray,
    sat_elevation_deg: np.ndarray,
    sat_belt_id: np.ndarray,
) -> dict[str, np.ndarray]:
    az = np.asarray(sat_azimuth_deg, dtype=np.float64)
    el = np.asarray(sat_elevation_deg, dtype=np.float64)
    belt = np.asarray(sat_belt_id, dtype=np.float64)
    sky = gpu_accel._s1586_skycell_id_host(az.astype(np.float32), el.astype(np.float32))
    source_kind = np.full(az.shape, gpu_accel.SAMPLER_SOURCE_INVALID, dtype=np.int8)
    source_id = np.full(az.shape, -1, dtype=np.int32)

    valid = np.isfinite(az) & np.isfinite(el) & np.isfinite(belt)
    belt_i = np.rint(belt).astype(np.int64, copy=False)
    belt_is_int = np.abs(belt - belt_i.astype(np.float64, copy=False)) <= 1.0e-6
    in_belt = valid & belt_is_int & (belt_i >= 0) & (belt_i < int(sampler.n_belts))
    in_sky = in_belt & (sky >= 0) & (sky < int(sampler.n_skycells))
    if np.any(in_sky):
        group_id = belt_i[in_sky] * np.int64(sampler.n_skycells) + sky[in_sky].astype(np.int64, copy=False)
        group_raw = np.asarray(sampler.group_raw_counts, dtype=np.int64)
        group_ptr = np.asarray(sampler.group_ptr, dtype=np.int64)
        belt_raw = np.asarray(sampler.belt_raw_counts, dtype=np.int64)
        belt_ptr = np.asarray(sampler.belt_ptr, dtype=np.int64)

        use_group = (
            (group_raw[group_id] >= int(sampler.cond_min_group_samples))
            & (group_ptr[group_id + 1] > group_ptr[group_id])
        )
        belt_use = belt_i[in_sky]
        use_belt = (
            ~use_group
            & (belt_raw[belt_use] >= int(sampler.cond_min_belt_samples))
            & (belt_ptr[belt_use + 1] > belt_ptr[belt_use])
        )
        use_global = ~(use_group | use_belt)

        rows = np.argwhere(in_sky)
        rows_group = rows[use_group]
        rows_belt = rows[use_belt]
        rows_global = rows[use_global]
        source_kind[tuple(rows_group.T)] = gpu_accel.SAMPLER_SOURCE_GROUP
        source_id[tuple(rows_group.T)] = group_id[use_group].astype(np.int32, copy=False)
        source_kind[tuple(rows_belt.T)] = gpu_accel.SAMPLER_SOURCE_BELT
        source_id[tuple(rows_belt.T)] = belt_use[use_belt].astype(np.int32, copy=False)
        source_kind[tuple(rows_global.T)] = gpu_accel.SAMPLER_SOURCE_GLOBAL
        source_id[tuple(rows_global.T)] = 0

    return {
        "skycell_id": sky.astype(np.int32, copy=False),
        "source_kind": source_kind,
        "source_id": source_id,
        "valid_mask": in_sky.astype(bool, copy=False),
    }


def _atm_lin_from_elev_deg_binned_reference(
    elev_deg_arr: np.ndarray,
    *,
    altitude_km: float,
    frequency_ghz: float,
    bin_deg: float,
    elev_min_deg: float,
    elev_max_deg: float,
    max_path_length_km: float,
) -> np.ndarray:
    from pycraf import atm

    elev = np.asarray(elev_deg_arr, dtype=np.float32)
    out = np.ones_like(elev, dtype=np.float32)
    vis = elev > 0.0
    if not np.any(vis):
        return out

    layers = atm.atm_layers(np.array([frequency_ghz], dtype=np.float64) * u.GHz, atm.profile_standard)
    elev_clip = np.clip(elev[vis], np.float32(elev_min_deg), np.float32(elev_max_deg))
    bins = np.rint(elev_clip / np.float32(bin_deg)).astype(np.int32)
    uniq = np.unique(bins)
    vals = np.empty(uniq.size, dtype=np.float32)
    for idx, bin_id in enumerate(uniq.tolist()):
        atten_db, _, _ = atm.atten_slant_annex1(
            float(bin_id) * float(bin_deg) * u.deg,
            float(altitude_km) * u.km,
            layers,
            do_tebb=False,
            max_path_length=float(max_path_length_km) * u.km,
        )
        vals[idx] = np.float32(10.0 ** (-float(np.ravel(atten_db.to_value(cnv.dB))[0]) / 10.0))
    out[vis] = vals[np.searchsorted(uniq, bins)]
    return out


def _build_b525_step2_case() -> dict[str, Any]:
    tleforger.reset_tle_counter()
    belt_package = tleforger.forge_tle_constellation_from_belt_definitions(
        [
            {
                "belt_name": "System3_Belt_1",
                "num_sats_per_plane": 120,
                "plane_count": 28,
                "altitude": 525 * u.km,
                "eccentricity": 0.0,
                "inclination_deg": 53.0 * u.deg,
                "argp_deg": 0.0 * u.deg,
                "RAAN_min": 0 * u.deg,
                "RAAN_max": 360 * u.deg,
                "min_elevation": 20 * u.deg,
                "adjacent_plane_offset": True,
            },
        ]
    )
    tle_list = belt_package["tle_list"]
    satellite_metadata = tleforger.expand_belt_metadata_to_satellites(belt_package)
    sat_min_elev_deg = satellite_metadata["sat_min_elevation_deg"]
    sat_beta_max_rad = np.deg2rad(satellite_metadata["sat_beta_max_deg"]).astype(np.float32, copy=False)
    sat_belt_id = satellite_metadata["sat_belt_id"]

    ras_station = cysgp4.PyObserver(21.443611, -30.712777, 1.052)
    layout = scenario.build_observer_layout(ras_station, [])
    observer_list = layout["observer_arr"]
    observers_new = observer_list[np.newaxis, :, np.newaxis]
    tles_new = tle_list[np.newaxis, np.newaxis, :]
    mjd = Time(datetime(2025, 1, 1, 0, 0, 0), scale="utc").mjd
    result = cysgp4.propagate_many(
        np.asarray([mjd], dtype=np.float64)[:, None, None],
        tles_new,
        observers_new,
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=False,
        do_sat_azel=True,
        do_sat_rotmat=False,
        sat_frame="xyz",
    )
    sat_topo_full = result["topo"]
    sat_azel_full = result["sat_azel"]
    vis_mask_horizon_full = sat_topo_full[:, 0, :, 1].astype(np.float32, copy=False) > np.float32(0.0)
    sat_idx_g = np.flatnonzero(np.any(vis_mask_horizon_full, axis=0)).astype(np.int32, copy=False)
    sat_topo = sat_topo_full[:, :, sat_idx_g, :]
    sat_azel = sat_azel_full[:, :, sat_idx_g, :]
    vis_mask_horizon = vis_mask_horizon_full[:, sat_idx_g]
    min_elev_deg_eff = sat_min_elev_deg[sat_idx_g].astype(np.float32, copy=False)
    beta_max_rad_eff = sat_beta_max_rad[sat_idx_g].astype(np.float32, copy=False)
    sat_belt_id_eff = sat_belt_id[sat_idx_g].astype(np.int16, copy=False)

    selection = satsim.select_satellite_links(
        sat_topo,
        min_elevation_deg=min_elev_deg_eff,
        n_links=1,
        strategy="random",
        rng=np.random.default_rng(16001),
        include_counts=False,
        include_payload=False,
        prefer_numba=True,
    )
    assignments = np.asarray(selection["assignments"], dtype=np.int32)[:, 0, :]
    is_co_sat = np.zeros(vis_mask_horizon.shape, dtype=bool)
    valid_assign = assignments >= 0
    if np.any(valid_assign):
        row_idx = np.broadcast_to(np.arange(vis_mask_horizon.shape[0], dtype=np.int32)[:, None], assignments.shape)
        is_co_sat[row_idx[valid_assign], assignments[valid_assign]] = True

    theta_sep = 0.5 * calculate_beamwidth_1d(
        s_1528_rec1_4_pattern_amend,
        level_drop=7.0 * cnv.dB,
        wavelength=15 * u.cm,
        Lt=1.6 * u.m,
        Lr=1.6 * u.m,
        l=2,
        SLR=20 * cnv.dB,
        far_sidelobe_start=90 * u.deg,
        far_sidelobe_level=-20 * cnv.dBi,
    )
    alpha0 = sat_azel[:, 0, :, 0].astype(np.float32, copy=False)
    beta0 = sat_azel[:, 0, :, 1].astype(np.float32, copy=False)
    alpha0_rad = np.remainder(alpha0 * np.float32(np.pi / 180.0), np.float32(2.0 * np.pi))
    beta0_rad = beta0 * np.float32(np.pi / 180.0)

    return {
        "sampler": _load_step2_sampler(),
        "observer_list": observer_list,
        "tle_list": tle_list,
        "sat_topo": sat_topo,
        "sat_azel": sat_azel,
        "vis_mask_horizon": vis_mask_horizon,
        "is_co_sat": is_co_sat,
        "alpha0_rad": alpha0_rad,
        "beta0_rad": beta0_rad,
        "beta_max_rad_per_sat": beta_max_rad_eff,
        "sat_belt_id_rows": np.broadcast_to(sat_belt_id_eff[None, :], vis_mask_horizon.shape).astype(np.int16, copy=False),
        "cos_min_sep": float(np.cos(np.float32(theta_sep.to_value(u.rad)))),
    }


def _run_python_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )


def _cpu_build_conditioned_templates(
    candidate_pools: dict[str, np.ndarray],
    beta_max_rad_per_unit: np.ndarray,
    *,
    template_size: int,
    cos_min_sep: float,
    start_offsets: np.ndarray,
) -> dict[str, np.ndarray]:
    pool_alpha = np.asarray(candidate_pools["alpha_rad"], dtype=np.float32)
    pool_beta = np.asarray(candidate_pools["beta_rad"], dtype=np.float32)
    pool_sina = np.asarray(candidate_pools["sin_alpha"], dtype=np.float32)
    pool_cosa = np.asarray(candidate_pools["cos_alpha"], dtype=np.float32)
    pool_sinb = np.asarray(candidate_pools["sin_beta"], dtype=np.float32)
    pool_cosb = np.asarray(candidate_pools["cos_beta"], dtype=np.float32)

    unit_count, pool_size = pool_alpha.shape
    template_idx = np.full((unit_count, template_size), -1, dtype=np.int32)
    template_alpha = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_beta = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_sina = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_cosa = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_sinb = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_cosb = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_valid = np.zeros(unit_count, dtype=np.int32)

    for unit in range(unit_count):
        for scan_offset in range(pool_size):
            if template_valid[unit] >= template_size:
                break
            pool_pos = int((int(start_offsets[unit]) + scan_offset) % pool_size)
            if not np.isfinite(pool_beta[unit, pool_pos]) or pool_beta[unit, pool_pos] > beta_max_rad_per_unit[unit]:
                continue
            ok = True
            for template_slot in range(int(template_valid[unit])):
                cos_da = template_cosa[unit, template_slot] * pool_cosa[unit, pool_pos]
                cos_da += template_sina[unit, template_slot] * pool_sina[unit, pool_pos]
                cos_gamma = template_cosb[unit, template_slot] * pool_cosb[unit, pool_pos]
                cos_gamma += template_sinb[unit, template_slot] * pool_sinb[unit, pool_pos] * cos_da
                if cos_gamma > cos_min_sep:
                    ok = False
                    break
            if not ok:
                continue
            slot = int(template_valid[unit])
            template_idx[unit, slot] = pool_pos
            template_alpha[unit, slot] = pool_alpha[unit, pool_pos]
            template_beta[unit, slot] = pool_beta[unit, pool_pos]
            template_sina[unit, slot] = pool_sina[unit, pool_pos]
            template_cosa[unit, slot] = pool_cosa[unit, pool_pos]
            template_sinb[unit, slot] = pool_sinb[unit, pool_pos]
            template_cosb[unit, slot] = pool_cosb[unit, pool_pos]
            template_valid[unit] += 1

    return {
        "template_idx": template_idx,
        "template_alpha_rad": template_alpha,
        "template_beta_rad": template_beta,
        "template_sin_alpha": template_sina,
        "template_cos_alpha": template_cosa,
        "template_sin_beta": template_sinb,
        "template_cos_beta": template_cosb,
        "template_valid_count": template_valid,
        "template_start_offsets": np.asarray(start_offsets, dtype=np.int32),
    }


def _cpu_assign_conditioned_beams(
    plan: gpu_accel.GpuConditionedTemplatePlan,
    templates: dict[str, np.ndarray],
    *,
    vis_mask_horizon: np.ndarray,
    is_co_sat: np.ndarray,
    alpha0_rad: np.ndarray,
    beta0_rad: np.ndarray,
    sina0: np.ndarray,
    cosa0: np.ndarray,
    sinb0: np.ndarray,
    cosb0: np.ndarray,
    beta_max_rad_per_sat: np.ndarray,
    n_beams: int,
    cos_min_sep: float,
    start_offsets: np.ndarray,
) -> dict[str, np.ndarray]:
    flat_shape = tuple(plan.flat_shape)
    flat_rows = int(np.prod(flat_shape, dtype=np.int64))
    sat_count = flat_shape[-1]

    active_rows = gpu_accel.copy_device_to_host(plan.d_active_rows).astype(np.int32, copy=False)
    row_to_unit = gpu_accel.copy_device_to_host(plan.d_row_to_unit).astype(np.int32, copy=False)
    template_counts = np.asarray(templates["template_valid_count"], dtype=np.int32)
    template_idx = np.asarray(templates["template_idx"], dtype=np.int32)
    template_alpha = np.asarray(templates["template_alpha_rad"], dtype=np.float32)
    template_beta = np.asarray(templates["template_beta_rad"], dtype=np.float32)
    template_sina = np.asarray(templates["template_sin_alpha"], dtype=np.float32)
    template_cosa = np.asarray(templates["template_cos_alpha"], dtype=np.float32)
    template_sinb = np.asarray(templates["template_sin_beta"], dtype=np.float32)
    template_cosb = np.asarray(templates["template_cos_beta"], dtype=np.float32)

    beam_idx_flat = np.full((flat_rows, n_beams), -1, dtype=np.int32)
    beam_alpha_flat = np.full((flat_rows, n_beams), np.nan, dtype=np.float32)
    beam_beta_flat = np.full((flat_rows, n_beams), np.nan, dtype=np.float32)
    beam_valid_flat = np.zeros(flat_rows, dtype=np.int16)

    beta_max_row = np.broadcast_to(np.asarray(beta_max_rad_per_sat, dtype=np.float32)[None, :], flat_shape).reshape(-1)
    vis_flat = np.asarray(vis_mask_horizon, dtype=bool).reshape(-1)
    co_flat = np.asarray(is_co_sat, dtype=bool).reshape(-1)
    alpha0_flat = np.asarray(alpha0_rad, dtype=np.float32).reshape(-1)
    beta0_flat = np.asarray(beta0_rad, dtype=np.float32).reshape(-1)
    sina0_flat = np.asarray(sina0, dtype=np.float32).reshape(-1)
    cosa0_flat = np.asarray(cosa0, dtype=np.float32).reshape(-1)
    sinb0_flat = np.asarray(sinb0, dtype=np.float32).reshape(-1)
    cosb0_flat = np.asarray(cosb0, dtype=np.float32).reshape(-1)

    for active_row_idx, flat_row in enumerate(active_rows.tolist()):
        if not vis_flat[flat_row]:
            continue
        beta_max = float(beta_max_row[flat_row])
        unit = int(row_to_unit[active_row_idx])
        n_acc = 0
        if co_flat[flat_row] and beta0_flat[flat_row] <= beta_max:
            beam_idx_flat[flat_row, 0] = -2
            beam_alpha_flat[flat_row, 0] = alpha0_flat[flat_row]
            beam_beta_flat[flat_row, 0] = beta0_flat[flat_row]
            n_acc = 1

        template_count = int(template_counts[unit])
        for scan_step in range(template_count):
            if n_acc >= n_beams:
                break
            pos = int((int(start_offsets[active_row_idx]) + scan_step) % max(1, template_count))
            cand_idx = int(template_idx[unit, pos])
            if cand_idx < 0 or template_beta[unit, pos] > beta_max:
                continue
            if co_flat[flat_row] and n_acc > 0:
                cos_da = cosa0_flat[flat_row] * template_cosa[unit, pos] + sina0_flat[flat_row] * template_sina[unit, pos]
                cos_gamma0 = cosb0_flat[flat_row] * template_cosb[unit, pos]
                cos_gamma0 += sinb0_flat[flat_row] * template_sinb[unit, pos] * cos_da
                if cos_gamma0 > cos_min_sep:
                    continue
            beam_idx_flat[flat_row, n_acc] = cand_idx
            beam_alpha_flat[flat_row, n_acc] = template_alpha[unit, pos]
            beam_beta_flat[flat_row, n_acc] = template_beta[unit, pos]
            n_acc += 1
        beam_valid_flat[flat_row] = np.int16(n_acc)

    return {
        "beam_idx": beam_idx_flat.reshape(flat_shape + (n_beams,)),
        "beam_alpha_rad": beam_alpha_flat.reshape(flat_shape + (n_beams,)),
        "beam_beta_rad": beam_beta_flat.reshape(flat_shape + (n_beams,)),
        "beam_valid": beam_valid_flat.reshape(flat_shape),
    }


def _cpu_reference(
    mjds: np.ndarray,
    tles: list[PyTle],
    observers: list[PyObserver],
    *,
    method: str,
    sat_frame: str,
    do_geo: bool,
    do_sat_azel: bool,
    do_obs_pos: bool,
    do_sat_rotmat: bool,
) -> dict[str, np.ndarray]:
    return cysgp4.propagate_many(
        mjds[:, np.newaxis, np.newaxis],
        np.asarray(tles, dtype=object)[np.newaxis, np.newaxis, :],
        np.asarray(observers, dtype=object)[np.newaxis, :, np.newaxis],
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=do_geo,
        do_topo=True,
        do_obs_pos=do_obs_pos,
        do_sat_azel=do_sat_azel,
        do_sat_rotmat=do_sat_rotmat,
        sat_frame=sat_frame,
        method=method,
    )


def _cpu_vallado_sgp4_orbit_reference(
    mjds: np.ndarray,
    tles: list[PyTle],
    *,
    gravity_model: str,
) -> dict[str, np.ndarray]:
    from sgp4.earth_gravity import wgs72, wgs84
    from sgp4.io import twoline2rv

    gravity_spec = wgs72 if gravity_model == gpu_accel.EARTH_MODEL_WGS72 else wgs84
    mjd_vec = np.asarray(mjds, dtype=np.float64)
    datetimes = Time(mjd_vec, format="mjd", scale="utc").to_datetime()
    eci_pos = np.empty((mjd_vec.size, len(tles), 3), dtype=np.float64)
    eci_vel = np.empty((mjd_vec.size, len(tles), 3), dtype=np.float64)
    for sat_idx, tle in enumerate(tles):
        _, line1, line2 = tle.tle_strings()
        sat = twoline2rv(str(line1), str(line2), gravity_spec)
        for time_idx, timestamp in enumerate(datetimes):
            second = float(timestamp.second) + float(timestamp.microsecond) / 1.0e6
            pos, vel = sat.propagate(
                int(timestamp.year),
                int(timestamp.month),
                int(timestamp.day),
                int(timestamp.hour),
                int(timestamp.minute),
                second,
            )
            eci_pos[time_idx, sat_idx, :] = np.asarray(pos, dtype=np.float64)
            eci_vel[time_idx, sat_idx, :] = np.asarray(vel, dtype=np.float64)
    return {"eci_pos": eci_pos, "eci_vel": eci_vel}


def _wrapped_angle_difference_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def _assert_against_cpu(
    gpu_result: dict[str, np.ndarray],
    cpu_result: dict[str, np.ndarray],
    *,
    sat_frame: str,
    atol_pos: float,
    atol_vel: float,
    atol_angle: float,
) -> None:
    assert_allclose(gpu_result["eci_pos"], cpu_result["eci_pos"][:, 0, :, :], atol=atol_pos, rtol=0.0)
    assert_allclose(gpu_result["eci_vel"], cpu_result["eci_vel"][:, 0, :, :], atol=atol_vel, rtol=0.0)
    if "geo" in gpu_result:
        assert_allclose(gpu_result["geo"], cpu_result["geo"][:, 0, :, :], atol=5e-3, rtol=0.0)
    assert_allclose(
        _wrapped_angle_difference_deg(gpu_result["topo"][..., 0], cpu_result["topo"][..., 0]),
        0.0,
        atol=atol_angle,
        rtol=0.0,
    )
    assert_allclose(gpu_result["topo"][..., 1:], cpu_result["topo"][..., 1:], atol=max(atol_pos, atol_vel), rtol=0.0)
    if "obs_pos" in gpu_result:
        assert_allclose(gpu_result["obs_pos"], cpu_result["obs_pos"][:, :, 0, :], atol=atol_pos, rtol=0.0)
    if "sat_azel" in gpu_result:
        assert_allclose(
            _wrapped_angle_difference_deg(gpu_result["sat_azel"][..., 0], cpu_result["sat_azel"][..., 0]),
            0.0,
            atol=atol_angle,
            rtol=0.0,
        )
        assert_allclose(gpu_result["sat_azel"][..., 1:], cpu_result["sat_azel"][..., 1:], atol=atol_angle, rtol=0.0)
    if "sat_rotmat" in gpu_result:
        assert_allclose(gpu_result["sat_rotmat"], cpu_result["sat_rotmat"][:, 0, :, :, :], atol=5e-4, rtol=0.0)
    if sat_frame == "zxy" and "sat_azel" in gpu_result:
        assert np.nanmax(np.abs(gpu_result["sat_azel"][..., 1])) <= 90.0001


def _expected_full_eligible_mask(
    sat_topo: np.ndarray,
    sat_azel: np.ndarray,
    *,
    min_elevation_deg: np.ndarray,
    beta_max_deg_per_sat: np.ndarray | None,
    beta_tol_deg: float,
) -> np.ndarray:
    elev_ok = sat_topo[..., 1] >= min_elevation_deg[None, None, :]
    if beta_max_deg_per_sat is None:
        return elev_ok.astype(bool, copy=False)
    beta = np.abs(sat_azel[..., 1])
    cone_ok = beta <= (beta_max_deg_per_sat[None, None, :] + float(beta_tol_deg))
    return np.asarray(elev_ok & cone_ok, dtype=bool)


def test_true_angular_distance_auto_matches_pycraf_for_arrays_and_quantities():
    l1 = np.array([0.0, 15.0], dtype=np.float64)
    b1 = np.array([0.0, -10.0], dtype=np.float64)
    l2 = np.array([90.0, 20.0], dtype=np.float64)
    b2 = np.array([0.0, 5.0], dtype=np.float64)

    expected = true_angular_distance(l1 * u.deg, b1 * u.deg, l2 * u.deg, b2 * u.deg).value
    assert_allclose(gpu_accel.true_angular_distance_auto(l1, b1, l2, b2), expected)
    assert_allclose(
        gpu_accel.true_angular_distance_auto(l1 * u.deg, b1 * u.deg, l2 * u.deg, b2 * u.deg),
        expected,
    )
    quantity_result = gpu_accel.true_angular_distance_auto(
        l1 * u.deg,
        b1 * u.deg,
        l2 * u.deg,
        b2 * u.deg,
        as_quantity=True,
    )
    assert isinstance(quantity_result, u.Quantity)
    assert quantity_result.unit == u.deg
    assert_allclose(quantity_result.value, expected)


@GPU_REQUIRED
def test_true_angular_distance_gpu_matches_pycraf_and_session_method():
    l1 = np.array([0.0, 15.0, 45.0], dtype=np.float32)
    b1 = np.array([0.0, -10.0, 22.0], dtype=np.float32)
    l2 = np.array([90.0, 20.0, 135.0], dtype=np.float32)
    b2 = np.array([0.0, 5.0, -30.0], dtype=np.float32)
    expected = true_angular_distance(l1 * u.deg, b1 * u.deg, l2 * u.deg, b2 * u.deg).value

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        host_result = gpu_accel.true_angular_distance_gpu(
            l1,
            b1,
            l2,
            b2,
            session=session,
            output_dtype=np.float32,
        )
        device_result = gpu_accel.true_angular_distance_gpu(
            l1,
            b1,
            l2,
            b2,
            session=session,
            output_dtype=np.float32,
            return_device=True,
        )
        session_result = session.true_angular_distance(
            l1,
            b1,
            l2,
            b2,
            output_dtype=np.float32,
            return_device=True,
        )
    assert_allclose(host_result, expected, atol=5e-4, rtol=0.0)
    assert_allclose(device_result.copy_to_host(), expected, atol=5e-4, rtol=0.0)
    assert_allclose(session_result.copy_to_host(), expected, atol=5e-4, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_true_angular_distance_auto_routes_device_inputs_to_gpu():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    l1 = np.array([0.0, 15.0], dtype=np.float32)
    b1 = np.array([0.0, -10.0], dtype=np.float32)
    l2 = np.array([90.0, 20.0], dtype=np.float32)
    b2 = np.array([0.0, 5.0], dtype=np.float32)
    expected = true_angular_distance(l1 * u.deg, b1 * u.deg, l2 * u.deg, b2 * u.deg).value
    with session.activate():
        d_l1 = gpu_accel.cuda.to_device(l1)
        d_b1 = gpu_accel.cuda.to_device(b1)
        d_l2 = gpu_accel.cuda.to_device(l2)
        d_b2 = gpu_accel.cuda.to_device(b2)
        result = gpu_accel.true_angular_distance_auto(
            d_l1,
            d_b1,
            d_l2,
            d_b2,
            session=session,
            output_dtype=np.float32,
        )
    assert_allclose(result, expected, atol=5e-4, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_true_angular_distance_rejects_quantity_device_return():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with pytest.raises(ValueError, match="as_quantity=True is only supported for host-return mode"):
        with session.activate():
            session.true_angular_distance(
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([90.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                return_device=True,
                as_quantity=True,
            )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_select_satellite_links_matches_cpu_max_elevation_and_eligible_mask():
    sat_topo = np.zeros((1, 2, 3, 2), dtype=np.float32)
    sat_topo[0, 0, :, 1] = [20.0, 40.0, 35.0]
    sat_topo[0, 1, :, 1] = [50.0, 10.0, 45.0]

    sat_azel = np.zeros((1, 2, 3, 2), dtype=np.float32)
    sat_azel[0, 0, :, 0] = [10.0, 20.0, 30.0]
    sat_azel[0, 1, :, 0] = [40.0, 50.0, 60.0]
    sat_azel[0, 0, :, 1] = [5.0, 15.0, 30.0]
    sat_azel[0, 1, :, 1] = [5.0, 10.0, 25.0]

    ras_topo = np.array(
        [[[100.0, 45.0], [110.0, 50.0], [120.0, 55.0]]],
        dtype=np.float32,
    )
    min_elev = np.array([10.0, 30.0, 40.0], dtype=np.float64)
    beta_max = np.array([10.0, 20.0, 20.0], dtype=np.float32)
    belt_id = np.array([0, 1, 1], dtype=np.int16)

    expected = satsim.select_satellite_links(
        sat_topo,
        sat_azel=sat_azel,
        ras_topo=ras_topo,
        min_elevation_deg=min_elev,
        beta_max_deg_per_sat=beta_max,
        sat_belt_id_per_sat=belt_id,
        n_links=2,
        strategy="max_elevation",
        include_counts=True,
        include_payload=True,
        include_eligible_mask=True,
        prefer_numba=False,
    )
    expected_eligible = _expected_full_eligible_mask(
        sat_topo,
        sat_azel,
        min_elevation_deg=min_elev,
        beta_max_deg_per_sat=beta_max,
        beta_tol_deg=1e-3,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        result = session.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=min_elev,
            beta_max_deg_per_sat=beta_max,
            sat_belt_id_per_sat=belt_id,
            n_links=2,
            strategy="max_elevation",
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            return_device=False,
        )

    assert_allclose(result["assignments"], expected["assignments"], atol=0.0, rtol=0.0)
    assert_allclose(result["sat_beam_counts_demand"], expected["sat_beam_counts_demand"], atol=0.0, rtol=0.0)
    assert_allclose(result["sat_beam_counts_eligible"], expected["sat_beam_counts_eligible"], atol=0.0, rtol=0.0)
    assert_allclose(result["sat_belt_id"], expected["sat_belt_id"], atol=0.0, rtol=0.0)
    assert_allclose(result["cone_ok"], expected["cone_ok"], atol=0.0, rtol=0.0)
    assert_allclose(result["sat_azimuth"], expected["sat_azimuth"], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(result["sat_beta"], expected["sat_beta"], atol=0.0, rtol=0.0, equal_nan=True)
    assert np.array_equal(expected["sat_eligible_mask"], expected_eligible)
    assert np.array_equal(result["sat_eligible_mask"], expected_eligible)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_select_satellite_links_random_is_seed_reproducible_and_supports_device_return():
    sat_topo = np.zeros((2, 3, 4, 2), dtype=np.float32)
    sat_topo[..., 1] = np.array(
        [
            [[10.0, 25.0, 35.0, 5.0], [0.0, 0.0, 0.0, 0.0], [30.0, 45.0, 50.0, 10.0]],
            [[22.0, 18.0, 40.0, 41.0], [19.0, 21.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]],
        ],
        dtype=np.float32,
    )
    sat_azel = np.zeros((2, 3, 4, 2), dtype=np.float32)
    sat_azel[..., 0] = 17.0
    sat_azel[..., 1] = 3.0
    ras_topo = np.stack(
        (
            np.array([[100.0, 110.0, 120.0, 130.0], [140.0, 150.0, 160.0, 170.0]], dtype=np.float32),
            np.array([[45.0, 46.0, 47.0, 48.0], [49.0, 50.0, 51.0, 52.0]], dtype=np.float32),
        ),
        axis=-1,
    )
    belt_id = np.array([0, 1, 2, 3], dtype=np.int16)
    min_elev = 20.0
    beta_max = 10.0

    expected = satsim.select_satellite_links(
        sat_topo,
        sat_azel=sat_azel,
        ras_topo=ras_topo,
        min_elevation_deg=min_elev,
        beta_max_deg_per_sat=beta_max,
        sat_belt_id_per_sat=belt_id,
        n_links=1,
        strategy="random",
        rng=np.random.default_rng(123),
        include_counts=True,
        include_payload=True,
        prefer_numba=satsim.HAS_NUMBA,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        device_result = session.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=min_elev,
            beta_max_deg_per_sat=beta_max,
            sat_belt_id_per_sat=belt_id,
            n_links=1,
            strategy="random",
            rng=np.random.default_rng(123),
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            return_device=True,
        )
        repeat_device_result = session.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=min_elev,
            beta_max_deg_per_sat=beta_max,
            sat_belt_id_per_sat=belt_id,
            n_links=1,
            strategy="random",
            rng=np.random.default_rng(123),
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            return_device=True,
        )
    result = {key: value.copy_to_host() if hasattr(value, "copy_to_host") else value for key, value in device_result.items()}
    repeat_result = {
        key: value.copy_to_host() if hasattr(value, "copy_to_host") else value
        for key, value in repeat_device_result.items()
    }

    assert_allclose(result["assignments"], expected["assignments"], atol=0.0, rtol=0.0)
    assert_allclose(result["assignments"], repeat_result["assignments"], atol=0.0, rtol=0.0)
    assert_allclose(result["sat_beam_counts_demand"], expected["sat_beam_counts_demand"], atol=0.0, rtol=0.0)
    assert_allclose(result["sat_beam_counts_eligible"], expected["sat_beam_counts_eligible"], atol=0.0, rtol=0.0)
    assert_allclose(result["sat_belt_id"], expected["sat_belt_id"], atol=0.0, rtol=0.0)
    assert_allclose(result["cone_ok"], expected["cone_ok"], atol=0.0, rtol=0.0)
    assert result["sat_eligible_mask"].dtype == np.bool_
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_select_satellite_links_cell_active_mask_matches_cpu():
    sat_topo = np.zeros((1, 2, 2, 2), dtype=np.float32)
    sat_topo[0, :, :, 1] = np.array([[40.0, 35.0], [50.0, 45.0]], dtype=np.float32)
    sat_azel = np.zeros((1, 2, 2, 2), dtype=np.float32)
    sat_azel[0, :, :, 0] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    sat_azel[0, :, :, 1] = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    ras_topo = np.array([[[100.0, 45.0], [110.0, 46.0]]], dtype=np.float32)
    cell_active_mask = np.array([[True, False]], dtype=bool)

    expected = satsim.select_satellite_links(
        sat_topo,
        sat_azel=sat_azel,
        ras_topo=ras_topo,
        min_elevation_deg=20.0,
        beta_max_deg_per_sat=np.array([10.0, 10.0], dtype=np.float32),
        sat_belt_id_per_sat=np.array([0, 1], dtype=np.int16),
        n_links=1,
        strategy="max_elevation",
        include_counts=True,
        include_payload=True,
        include_eligible_mask=True,
        cell_active_mask=cell_active_mask,
        prefer_numba=False,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        result = session.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=20.0,
            beta_max_deg_per_sat=np.array([10.0, 10.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 1], dtype=np.int16),
            n_links=1,
            strategy="max_elevation",
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            cell_active_mask=cell_active_mask,
            return_device=False,
        )

    for key in (
        "assignments",
        "sat_beam_counts_demand",
        "sat_beam_counts_eligible",
        "sat_eligible_mask",
        "cone_ok",
        "sat_belt_id",
    ):
        assert_equal(result[key], expected[key])
    assert_allclose(result["sat_azimuth"], expected["sat_azimuth"], atol=0.0, rtol=0.0, equal_nan=True)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_select_satellite_links_accepts_device_per_satellite_controls():
    sat_topo = np.zeros((2, 2, 4, 2), dtype=np.float32)
    sat_topo[..., 1] = np.array(
        [
            [[25.0, 35.0, 5.0, 45.0], [15.0, 55.0, 22.0, 0.0]],
            [[30.0, 10.0, 40.0, 41.0], [18.0, 12.0, 0.0, 50.0]],
        ],
        dtype=np.float32,
    )
    sat_azel = np.zeros((2, 2, 4, 2), dtype=np.float32)
    sat_azel[..., 0] = 12.0
    sat_azel[..., 1] = np.array(
        [
            [[3.0, 4.0, 20.0, 5.0], [6.0, 2.0, 9.0, 30.0]],
            [[4.0, 12.0, 3.0, 2.0], [7.0, 6.0, 15.0, 3.0]],
        ],
        dtype=np.float32,
    )
    ras_topo = np.stack(
        (
            np.array([[100.0, 110.0, 120.0, 130.0], [140.0, 150.0, 160.0, 170.0]], dtype=np.float32),
            np.array([[45.0, 46.0, 47.0, 48.0], [49.0, 50.0, 51.0, 52.0]], dtype=np.float32),
        ),
        axis=-1,
    )
    min_elev = np.array([20.0, 15.0, 10.0, 35.0], dtype=np.float32)
    beta_max = np.array([5.0, 8.0, 10.0, 6.0], dtype=np.float32)
    belt_id = np.array([0, 1, 1, 2], dtype=np.int16)

    expected = satsim.select_satellite_links(
        sat_topo,
        sat_azel=sat_azel,
        ras_topo=ras_topo,
        min_elevation_deg=min_elev,
        beta_max_deg_per_sat=beta_max,
        sat_belt_id_per_sat=belt_id,
        n_links=2,
        strategy="max_elevation",
        include_counts=True,
        include_payload=True,
        include_eligible_mask=True,
        prefer_numba=satsim.HAS_NUMBA,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        d_min_elev = gpu_accel.cuda.to_device(min_elev)
        d_beta_max = gpu_accel.cuda.to_device(beta_max)
        d_belt_id = gpu_accel.cuda.to_device(belt_id)
        host_result = session.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=d_min_elev,
            beta_max_deg_per_sat=d_beta_max,
            sat_belt_id_per_sat=d_belt_id,
            n_links=2,
            strategy="max_elevation",
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            return_device=False,
        )
        device_result = session.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=d_min_elev,
            beta_max_deg_per_sat=d_beta_max,
            sat_belt_id_per_sat=d_belt_id,
            n_links=2,
            strategy="max_elevation",
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            return_device=True,
        )

    device_host = {
        key: value.copy_to_host() if hasattr(value, "copy_to_host") else value
        for key, value in device_result.items()
    }

    for result in (host_result, device_host):
        assert_allclose(result["assignments"], expected["assignments"], atol=0.0, rtol=0.0)
        assert_allclose(result["sat_beam_counts_demand"], expected["sat_beam_counts_demand"], atol=0.0, rtol=0.0)
        assert_allclose(result["sat_beam_counts_eligible"], expected["sat_beam_counts_eligible"], atol=0.0, rtol=0.0)
        assert_allclose(result["sat_belt_id"], expected["sat_belt_id"], atol=0.0, rtol=0.0)
        assert_allclose(result["cone_ok"], expected["cone_ok"], atol=0.0, rtol=0.0)
        assert_allclose(result["sat_azimuth"], expected["sat_azimuth"], atol=0.0, rtol=0.0, equal_nan=True)
        assert_allclose(result["sat_beta"], expected["sat_beta"], atol=0.0, rtol=0.0, equal_nan=True)
        assert np.array_equal(result["sat_eligible_mask"], expected["sat_eligible_mask"])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_select_satellite_links_single_observer_matches_cpu_selector():
    sat_topo = np.zeros((4, 1, 6, 2), dtype=np.float32)
    sat_topo[:, 0, :, 1] = np.array(
        [
            [10.0, 40.0, 20.0, 50.0, -5.0, 35.0],
            [0.0, 15.0, 60.0, 61.0, 5.0, 30.0],
            [45.0, 44.0, 43.0, 42.0, 41.0, 40.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    min_elev = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        for strategy in ("max_elevation", "random"):
            expected_kwargs = {}
            result_kwargs = {}
            if strategy == "random":
                expected_kwargs["rng"] = np.random.default_rng(123)
                result_kwargs["rng"] = np.random.default_rng(123)
            expected = satsim.select_satellite_links(
                sat_topo,
                min_elevation_deg=min_elev,
                n_links=2,
                strategy=strategy,
                include_counts=False,
                include_payload=False,
                prefer_numba=satsim.HAS_NUMBA,
                **expected_kwargs,
            )
            result = session.select_satellite_links(
                sat_topo,
                min_elevation_deg=min_elev,
                n_links=2,
                strategy=strategy,
                include_counts=False,
                include_payload=False,
                return_device=False,
                **result_kwargs,
            )
            assert_allclose(result["assignments"], expected["assignments"], atol=0.0, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_select_satellite_links_finite_n_beam_matches_cpu_selector():
    sat_topo = np.zeros((1, 3, 3, 2), dtype=np.float32)
    sat_topo[0, :, :, 1] = np.array(
        [
            [80.0, 50.0, 20.0],
            [70.0, 60.0, 30.0],
            [65.0, 55.0, 40.0],
        ],
        dtype=np.float32,
    )
    sat_azel = np.zeros((1, 3, 3, 2), dtype=np.float32)
    sat_azel[0, :, :, 0] = np.array(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
        ],
        dtype=np.float32,
    )
    sat_azel[0, :, :, 1] = np.array(
        [
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0],
            [11.0, 12.0, 35.0],
        ],
        dtype=np.float32,
    )
    ras_topo = np.array(
        [[[100.0, 45.0], [110.0, 46.0], [120.0, 47.0]]],
        dtype=np.float32,
    )
    belt_id = np.array([0, 1, 2], dtype=np.int16)
    beta_max = np.array([20.0, 20.0, 20.0], dtype=np.float32)

    expected = satsim.select_satellite_links(
        sat_topo,
        min_elevation_deg=0.0,
        n_links=1,
        n_beam=1,
        strategy="max_elevation",
        sat_azel=sat_azel,
        ras_topo=ras_topo,
        beta_max_deg_per_sat=beta_max,
        sat_belt_id_per_sat=belt_id,
        include_counts=True,
        include_payload=True,
        prefer_numba=False,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        host_result = session.select_satellite_links(
            sat_topo,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=1,
            strategy="max_elevation",
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            beta_max_deg_per_sat=beta_max,
            sat_belt_id_per_sat=belt_id,
            include_counts=True,
            include_payload=True,
            return_device=False,
        )
        device_result = session.select_satellite_links(
            sat_topo,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=1,
            strategy="max_elevation",
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            beta_max_deg_per_sat=beta_max,
            sat_belt_id_per_sat=belt_id,
            include_counts=True,
            include_payload=True,
            return_device=True,
        )
    device_host = {
        key: value.copy_to_host() if hasattr(value, "copy_to_host") else value
        for key, value in device_result.items()
    }

    for result in (host_result, device_host):
        assert_allclose(result["assignments"], expected["assignments"], atol=0.0, rtol=0.0)
        assert_allclose(result["sat_beam_counts_demand"], expected["sat_beam_counts_demand"], atol=0.0, rtol=0.0)
        assert_allclose(result["sat_beam_counts_eligible"], expected["sat_beam_counts_eligible"], atol=0.0, rtol=0.0)
        assert_allclose(result["sat_belt_id"], expected["sat_belt_id"], atol=0.0, rtol=0.0)
        assert_allclose(result["cone_ok"], expected["cone_ok"], atol=0.0, rtol=0.0)
        assert_allclose(result["sat_azimuth"], expected["sat_azimuth"], atol=0.0, rtol=0.0, equal_nan=True)
        assert_allclose(result["sat_beta"], expected["sat_beta"], atol=0.0, rtol=0.0, equal_nan=True)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_matches_cpu_one_shot():
    sat_topo = np.zeros((2, 4, 3, 2), dtype=np.float32)
    sat_topo[..., 1] = np.array(
        [
            [[70.0, 60.0, 10.0], [69.0, 59.0, 20.0], [68.0, 58.0, 30.0], [67.0, 57.0, 40.0]],
            [[50.0, 49.0, 20.0], [48.0, 47.0, 30.0], [46.0, 45.0, 40.0], [44.0, 43.0, 41.0]],
        ],
        dtype=np.float32,
    )
    sat_azel = np.zeros((2, 4, 3, 2), dtype=np.float32)
    sat_azel[..., 0] = np.array(
        [
            [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0], [13.0, 23.0, 33.0], [14.0, 24.0, 34.0]],
            [[15.0, 25.0, 35.0], [16.0, 26.0, 36.0], [17.0, 27.0, 37.0], [18.0, 28.0, 38.0]],
        ],
        dtype=np.float32,
    )
    sat_azel[..., 1] = np.array(
        [
            [[5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [7.0, 8.0, 9.0], [8.0, 9.0, 10.0]],
            [[9.0, 10.0, 11.0], [10.0, 11.0, 12.0], [11.0, 12.0, 13.0], [12.0, 13.0, 14.0]],
        ],
        dtype=np.float32,
    )
    ras_topo = np.array(
        [
            [[100.0, 45.0], [110.0, 46.0], [120.0, 47.0]],
            [[130.0, 48.0], [140.0, 49.0], [150.0, 50.0]],
        ],
        dtype=np.float32,
    )
    belt_id = np.array([0, 1, 1], dtype=np.int16)
    beta_max = np.array([20.0, 20.0, 20.0], dtype=np.float32)

    expected = satsim.select_satellite_links(
        sat_topo,
        min_elevation_deg=0.0,
        n_links=1,
        n_beam=2,
        strategy="max_elevation",
        sat_azel=sat_azel,
        ras_topo=ras_topo,
        beta_max_deg_per_sat=beta_max,
        sat_belt_id_per_sat=belt_id,
        include_counts=True,
        include_payload=True,
        include_eligible_mask=True,
        prefer_numba=False,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=2,
            cell_count=4,
            sat_count=3,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=belt_id,
            beta_max_deg_per_sat=beta_max,
            ras_topo=ras_topo,
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
        )
        library.add_chunk(2, sat_topo[:, 2:, :, :], sat_azel=sat_azel[:, 2:, :, :])
        library.add_chunk(0, sat_topo[:, :2, :, :], sat_azel=sat_azel[:, :2, :, :])
        device_result = library.finalize(return_device=True)
    result = {
        key: value.copy_to_host() if hasattr(value, "copy_to_host") else value
        for key, value in device_result.items()
    }

    for key, expected_value in expected.items():
        if expected_value.dtype.kind == "f":
            assert_allclose(result[key], expected_value, atol=0.0, rtol=0.0, equal_nan=True)
        else:
            assert_allclose(result[key], expected_value, atol=0.0, rtol=0.0)
    session.close(reset_device=False)


def _direct_epfd_small_case() -> dict[str, np.ndarray]:
    sat_topo = np.zeros((1, 3, 3, 2), dtype=np.float32)
    sat_topo[0, 0, :, 1] = np.array([80.0, 70.0, 60.0], dtype=np.float32)
    sat_topo[0, 1, :, 1] = np.array([79.0, 78.0, 10.0], dtype=np.float32)
    sat_topo[0, 2, :, 1] = np.array([40.0, 50.0, 77.0], dtype=np.float32)

    sat_azel = np.zeros((1, 3, 3, 2), dtype=np.float32)
    sat_azel[0, 0, :, 0] = np.array([12.0, 60.0, 120.0], dtype=np.float32)
    sat_azel[0, 1, :, 0] = np.array([10.5, 40.0, 160.0], dtype=np.float32)
    sat_azel[0, 2, :, 0] = np.array([35.0, 41.0, 200.0], dtype=np.float32)
    sat_azel[0, 0, :, 1] = np.array([5.0, 8.0, 9.0], dtype=np.float32)
    sat_azel[0, 1, :, 1] = np.array([5.1, 10.0, 12.0], dtype=np.float32)
    sat_azel[0, 2, :, 1] = np.array([6.0, 7.0, 8.0], dtype=np.float32)

    ras_topo = np.array([[[100.0, 50.0], [110.0, 55.0], [120.0, 60.0]]], dtype=np.float32)
    ras_sat_azel = np.array([[[10.0, 5.0], [40.0, 9.0], [200.0, 8.0]]], dtype=np.float32)
    sat_beta_max = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    sat_belt_id = np.array([0, 1, 1], dtype=np.int16)
    return {
        "sat_topo": sat_topo,
        "sat_azel": sat_azel,
        "ras_topo": ras_topo,
        "ras_sat_azel": ras_sat_azel,
        "sat_beta_max": sat_beta_max,
        "sat_belt_id": sat_belt_id,
    }


def _boresight_direct_epfd_case() -> dict[str, np.ndarray]:
    sat_topo = np.zeros((1, 2, 2, 2), dtype=np.float32)
    sat_topo[0, 0, :, 1] = np.array([80.0, 70.0], dtype=np.float32)
    sat_topo[0, 1, :, 1] = np.array([76.0, 75.0], dtype=np.float32)

    sat_azel = np.zeros((1, 2, 2, 2), dtype=np.float32)
    sat_azel[0, 0, :, 0] = np.array([10.0, 90.0], dtype=np.float32)
    sat_azel[0, 1, :, 0] = np.array([20.0, 100.0], dtype=np.float32)
    sat_azel[0, 0, :, 1] = np.array([5.0, 6.0], dtype=np.float32)
    sat_azel[0, 1, :, 1] = np.array([7.0, 8.0], dtype=np.float32)

    ras_topo = np.array([[[10.0, 50.0], [90.0, 50.0]]], dtype=np.float32)
    ras_sat_azel = np.array([[[10.0, 4.0], [90.0, 5.0]]], dtype=np.float32)
    ras_topo_full = np.array(
        [[[10.0, 50.0, 720.0, 0.0], [90.0, 50.0, 730.0, 0.0]]],
        dtype=np.float32,
    )
    ras_sat_azel_full = np.array(
        [[[10.0, 4.0, 0.0], [90.0, 5.0, 0.0]]],
        dtype=np.float32,
    )
    return {
        "sat_topo": sat_topo,
        "sat_azel": sat_azel,
        "ras_topo": ras_topo,
        "ras_sat_azel": ras_sat_azel,
        "ras_topo_full": ras_topo_full,
        "ras_sat_azel_full": ras_sat_azel_full,
        "pointing_az_deg": np.array([[10.0, 90.0]], dtype=np.float32),
        "pointing_el_deg": np.array([[50.0, 50.0]], dtype=np.float32),
        "sat_beta_max": np.array([20.0, 20.0], dtype=np.float32),
        "sat_belt_id": np.array([0, 1], dtype=np.int16),
        "orbit_radius_m": np.array(
            [
                R_earth.to_value(u.m) + 525_000.0,
                R_earth.to_value(u.m) + 530_000.0,
            ],
            dtype=np.float32,
        ),
    }


def _append_inactive_ras_satellite(case: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    result = {
        key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
        for key, value in case.items()
    }
    visible_subset = np.arange(int(case["sat_topo"].shape[2]), dtype=np.int32)

    inactive_sat_topo = np.zeros(
        (
            int(case["sat_topo"].shape[0]),
            int(case["sat_topo"].shape[1]),
            1,
            int(case["sat_topo"].shape[3]),
        ),
        dtype=np.float32,
    )
    inactive_sat_topo[..., 1] = -5.0
    result["sat_topo"] = np.concatenate([result["sat_topo"], inactive_sat_topo], axis=2)

    inactive_sat_azel = np.zeros(
        (
            int(case["sat_azel"].shape[0]),
            int(case["sat_azel"].shape[1]),
            1,
            int(case["sat_azel"].shape[3]),
        ),
        dtype=np.float32,
    )
    inactive_sat_azel[..., 1] = 45.0
    result["sat_azel"] = np.concatenate([result["sat_azel"], inactive_sat_azel], axis=2)

    inactive_ras_topo = np.zeros(
        (int(case["ras_topo"].shape[0]), 1, int(case["ras_topo"].shape[2])),
        dtype=np.float32,
    )
    inactive_ras_topo[..., 1] = -5.0
    result["ras_topo"] = np.concatenate([result["ras_topo"], inactive_ras_topo], axis=1)

    inactive_ras_sat_azel = np.zeros(
        (int(case["ras_sat_azel"].shape[0]), 1, int(case["ras_sat_azel"].shape[2])),
        dtype=np.float32,
    )
    inactive_ras_sat_azel[..., 1] = 90.0
    result["ras_sat_azel"] = np.concatenate([result["ras_sat_azel"], inactive_ras_sat_azel], axis=1)

    result["sat_beta_max"] = np.concatenate(
        [result["sat_beta_max"], np.array([20.0], dtype=np.float32)],
        axis=0,
    )
    result["sat_belt_id"] = np.concatenate(
        [result["sat_belt_id"], np.array([0], dtype=np.int16)],
        axis=0,
    )

    if "ras_topo_full" in result:
        inactive_ras_topo_full = np.zeros(
            (int(result["ras_topo_full"].shape[0]), 1, int(result["ras_topo_full"].shape[2])),
            dtype=np.float32,
        )
        inactive_ras_topo_full[..., 1] = -5.0
        result["ras_topo_full"] = np.concatenate([result["ras_topo_full"], inactive_ras_topo_full], axis=1)
    if "ras_sat_azel_full" in result:
        inactive_ras_sat_azel_full = np.zeros(
            (int(result["ras_sat_azel_full"].shape[0]), 1, int(result["ras_sat_azel_full"].shape[2])),
            dtype=np.float32,
        )
        inactive_ras_sat_azel_full[..., 1] = 90.0
        result["ras_sat_azel_full"] = np.concatenate(
            [result["ras_sat_azel_full"], inactive_ras_sat_azel_full],
            axis=1,
        )
    if "orbit_radius_m" in result:
        result["orbit_radius_m"] = np.concatenate(
            [result["orbit_radius_m"], np.array([result["orbit_radius_m"][-1]], dtype=np.float32)],
            axis=0,
        )

    return result, visible_subset


def _prepare_direct_epfd_library(
    session: gpu_accel.GpuScepterSession,
    case: dict[str, np.ndarray],
    *,
    strategy: str,
    rng: Any,
    cell_active_mask: np.ndarray | None = None,
) -> gpu_accel.GpuSatelliteLinkSelectionLibrary:
    library = session.prepare_satellite_link_selection_library(
        time_count=int(case["sat_topo"].shape[0]),
        cell_count=int(case["sat_topo"].shape[1]),
        sat_count=int(case["sat_topo"].shape[2]),
        min_elevation_deg=0.0,
        n_links=1,
        n_beam=2,
        strategy=strategy,
        sat_belt_id_per_sat=case["sat_belt_id"],
        beta_max_deg_per_sat=case["sat_beta_max"],
        ras_topo=case["ras_topo"],
        cell_active_mask=cell_active_mask,
        rng=rng,
        include_counts=True,
        include_payload=True,
        include_eligible_mask=False,
    )
    library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
    return library


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_add_chunk_tracks_chunk_telemetry():
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=int(case["sat_topo"].shape[0]),
            cell_count=int(case["sat_topo"].shape[1]),
            sat_count=int(case["sat_topo"].shape[2]),
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
        )
        assert library.last_add_chunk_telemetry == {}
        assert library.max_add_chunk_telemetry == {}

        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])

        assert isinstance(library.last_add_chunk_telemetry, dict)
        assert isinstance(library.max_add_chunk_telemetry, dict)
        assert int(library.last_add_chunk_telemetry["chunk_cell_count"]) == int(case["sat_topo"].shape[1])
        assert int(library.last_add_chunk_telemetry["visible_candidate_count"]) > 0
        assert (
            int(library.max_add_chunk_telemetry["visible_candidate_count"])
            == int(library.last_add_chunk_telemetry["visible_candidate_count"])
        )

        empty_library = session.prepare_satellite_link_selection_library(
            time_count=int(case["sat_topo"].shape[0]),
            cell_count=int(case["sat_topo"].shape[1]),
            sat_count=int(case["sat_topo"].shape[2]),
            min_elevation_deg=89.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
        )
        empty_library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])

        assert isinstance(empty_library.last_add_chunk_telemetry, dict)
        assert isinstance(empty_library.max_add_chunk_telemetry, dict)
        assert int(empty_library.last_add_chunk_telemetry["visible_candidate_count"]) == 0
        assert int(empty_library.max_add_chunk_telemetry["visible_candidate_count"]) == 0

    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_repairs_ras_conflicts():
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )

    assert_allclose(result["ras_retargeted_count"], np.array([1], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["ras_reserved_count"], np.array([1], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["direct_kept_count"], np.array([1], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["repaired_link_count"], np.array([1], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["dropped_link_count"], np.array([0], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["sat_beam_counts_used"], np.array([[1, 1, 1]], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_equal(
        result["sat_beam_counts_used"],
        np.count_nonzero((result["beam_idx"] >= 0) | (result["beam_idx"] == -2), axis=-1),
    )

    beam_idx = result["beam_idx"]
    beam_alpha = result["beam_alpha_rad"]
    beam_beta = result["beam_beta_rad"]
    assert int(beam_idx[0, 0, 0]) == -2
    assert int(beam_idx[0, 0, 1]) == -1
    assert int(beam_idx[0, 1, 0]) == 1
    assert int(beam_idx[0, 2, 0]) == 2
    assert_allclose(beam_alpha[0, 0, 0], np.deg2rad(10.0), atol=1e-7, rtol=0.0)
    assert_allclose(beam_beta[0, 0, 0], np.deg2rad(5.0), atol=1e-7, rtol=0.0)
    assert_allclose(beam_alpha[0, 1, 0], np.deg2rad(40.0), atol=1e-7, rtol=0.0)
    assert_allclose(beam_beta[0, 1, 0], np.deg2rad(10.0), atol=1e-7, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_supports_cell_center_mode():
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        result = library.finalize_direct_epfd_beams(
            ras_pointing_mode="cell_center",
            return_device=False,
        )

    assert_allclose(result["ras_retargeted_count"], np.array([0], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["ras_reserved_count"], np.array([0], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["repaired_link_count"], np.array([0], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["direct_kept_count"], np.array([3], dtype=np.int32), atol=0.0, rtol=0.0)
    assert int(np.count_nonzero(result["beam_idx"] == -2)) == 0
    assert_allclose(result["beam_idx"][0, 0, 0], np.int32(0), atol=0.0, rtol=0.0)
    assert_allclose(result["beam_alpha_rad"][0, 0, 0], np.deg2rad(12.0), atol=1e-7, rtol=0.0)
    assert_allclose(result["beam_beta_rad"][0, 0, 0], np.deg2rad(5.0), atol=1e-7, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_output_sat_indices_matches_full_cell_center():
    case = _direct_epfd_small_case()
    subset = np.array([0, 2], dtype=np.int32)
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        full = library.finalize_direct_epfd_beams(
            ras_pointing_mode="cell_center",
            include_full_sat_beam_counts_used=True,
            return_device=False,
        )
        reduced = library.finalize_direct_epfd_beams(
            ras_pointing_mode="cell_center",
            include_full_sat_beam_counts_used=True,
            output_sat_indices=subset,
            return_device=False,
        )

    assert_equal(reduced["beam_idx"], full["beam_idx"][:, subset, :])
    assert_allclose(reduced["beam_alpha_rad"], full["beam_alpha_rad"][:, subset, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(reduced["beam_beta_rad"], full["beam_beta_rad"][:, subset, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_equal(reduced["sat_beam_counts_used"], full["sat_beam_counts_used"][:, subset])
    assert_equal(reduced["sat_beam_counts_used_full"], full["sat_beam_counts_used_full"])
    assert_equal(reduced["direct_kept_count"], full["direct_kept_count"])
    assert_equal(reduced["dropped_link_count"], full["dropped_link_count"])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_output_sat_indices_matches_full_ras_station():
    case = _direct_epfd_small_case()
    subset = np.array([1, 2], dtype=np.int32)
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            return_device=False,
        )
        reduced = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            output_sat_indices=subset,
            return_device=False,
        )

    assert_equal(reduced["beam_idx"], full["beam_idx"][:, subset, :])
    assert_allclose(reduced["beam_alpha_rad"], full["beam_alpha_rad"][:, subset, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(reduced["beam_beta_rad"], full["beam_beta_rad"][:, subset, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_equal(reduced["sat_beam_counts_used"], full["sat_beam_counts_used"][:, subset])
    assert_equal(reduced["sat_beam_counts_used_full"], full["sat_beam_counts_used_full"])
    assert_equal(reduced["ras_retargeted_count"], full["ras_retargeted_count"])
    assert_equal(reduced["repaired_link_count"], full["repaired_link_count"])
    assert_equal(reduced["dropped_link_count"], full["dropped_link_count"])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_output_sat_indices_accepts_cupy_ras_station():
    case = _direct_epfd_small_case()
    subset_host = np.array([1, 2], dtype=np.int32)
    subset_device = gpu_accel.cp.asarray(subset_host, dtype=gpu_accel.cp.int32)
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            return_device=False,
        )
        reduced = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            output_sat_indices=subset_device,
            return_device=False,
        )

    assert_equal(reduced["beam_idx"], full["beam_idx"][:, subset_host, :])
    assert_allclose(reduced["beam_alpha_rad"], full["beam_alpha_rad"][:, subset_host, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(reduced["beam_beta_rad"], full["beam_beta_rad"][:, subset_host, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_equal(reduced["sat_beam_counts_used"], full["sat_beam_counts_used"][:, subset_host])
    assert_equal(reduced["sat_beam_counts_used_full"], full["sat_beam_counts_used_full"])
    assert_equal(reduced["ras_retargeted_count"], full["ras_retargeted_count"])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_compact_ras_station_matches_full_visible_subset():
    case, visible_subset = _append_inactive_ras_satellite(_direct_epfd_small_case())
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            output_sat_indices=visible_subset,
            return_device=False,
        )
        compact = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"][:, visible_subset, :],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            output_sat_indices=visible_subset,
            return_device=False,
        )

    assert_equal(compact["beam_idx"], full["beam_idx"])
    assert_allclose(compact["beam_alpha_rad"], full["beam_alpha_rad"], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(compact["beam_beta_rad"], full["beam_beta_rad"], atol=0.0, rtol=0.0, equal_nan=True)
    assert_equal(compact["sat_beam_counts_used"], full["sat_beam_counts_used"])
    assert_equal(compact["sat_beam_counts_used_full"], full["sat_beam_counts_used_full"])
    assert_equal(compact["ras_retargeted_count"], full["ras_retargeted_count"])
    assert_equal(compact["repaired_link_count"], full["repaired_link_count"])
    assert_equal(compact["dropped_link_count"], full["dropped_link_count"])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_skips_inactive_ras_cell():
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(
            session,
            case,
            strategy="max_elevation",
            rng=0,
            cell_active_mask=np.array([[False, True, True]], dtype=bool),
        )
        result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )

    assert_allclose(result["ras_retargeted_count"], np.array([0], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["ras_reserved_count"], np.array([0], dtype=np.int32), atol=0.0, rtol=0.0)
    assert int(np.count_nonzero(result["beam_idx"] == -2)) == 0
    assert int(np.count_nonzero(result["beam_idx"] == 0)) == 0
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_reuses_candidate_cache():
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        assert library._packed_candidate_base_cp is None
        first = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        assert library._packed_candidate_base_cp is not None
        assert library._direct_candidate_view_cp is not None
        assert "row_ptr_host" in library._direct_candidate_view_cp
        assert isinstance(library._direct_candidate_view_cp["row_ptr_host"], np.ndarray)
        assert library._selector_candidate_view_cp is None
        assert library._candidate_time_parts == []
        assert library._candidate_cell_parts == []
        assert library._candidate_sat_parts == []
        second = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )

    for key in first:
        if np.asarray(first[key]).dtype.kind == "f":
            assert_allclose(first[key], second[key], atol=0.0, rtol=0.0, equal_nan=True)
        else:
            assert_equal(first[key], second[key])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_build_link_library_candidate_base_cp_reuses_single_chunk_parts():
    cp = gpu_accel.cp
    candidate_time_cp = cp.asarray([0, 1, 2], dtype=cp.int32)
    candidate_cell_cp = cp.asarray([3, 4, 5], dtype=cp.int32)
    candidate_sat_cp = cp.asarray([6, 7, 8], dtype=cp.int32)
    candidate_key_cp = cp.asarray([0.1, 0.2, 0.3], dtype=cp.float64)
    candidate_alpha_cp = cp.asarray([10.0, 20.0, 30.0], dtype=cp.float32)
    candidate_beta_cp = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float32)

    library = types.SimpleNamespace(
        _packed_candidate_base_cp=None,
        _candidate_time_parts=[candidate_time_cp],
        _candidate_cell_parts=[candidate_cell_cp],
        _candidate_sat_parts=[candidate_sat_cp],
        _candidate_key_parts=[candidate_key_cp],
        _candidate_alpha_parts=[candidate_alpha_cp],
        _candidate_beta_parts=[candidate_beta_cp],
    )

    base = gpu_accel._build_link_library_candidate_base_cp(library)

    assert base["time"] is candidate_time_cp
    assert base["cell"] is candidate_cell_cp
    assert base["sat"] is candidate_sat_cp
    assert base["key"] is candidate_key_cp
    assert base["alpha_deg"] is candidate_alpha_cp
    assert base["beta_deg"] is candidate_beta_cp
    assert library._candidate_time_parts == []
    assert library._candidate_cell_parts == []
    assert library._candidate_sat_parts == []
    assert library._candidate_key_parts == []
    assert library._candidate_alpha_parts == []
    assert library._candidate_beta_parts == []
    assert_allclose(
        cp.asnumpy(base["alpha_rad"]),
        np.remainder(np.asarray([10.0, 20.0, 30.0], dtype=np.float32) * (np.pi / 180.0), 2.0 * np.pi),
        atol=0.0,
        rtol=1.0e-6,
    )
    assert_allclose(
        cp.asnumpy(base["beta_rad"]),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32) * (np.pi / 180.0),
        atol=0.0,
        rtol=1.0e-6,
    )


def test_direct_epfd_repair_launch_grid_flattens_large_row_counts():
    max_blocks = int(gpu_accel.GPU_DIRECT_REPAIR_MAX_BLOCKS_PER_LAUNCH)

    grid_small, launch_count_small = gpu_accel._direct_epfd_repair_launch_grid(row_count=17)
    grid_large, launch_count_large = gpu_accel._direct_epfd_repair_launch_grid(
        row_count=max_blocks + 17
    )
    grid_huge, launch_count_huge = gpu_accel._direct_epfd_repair_launch_grid(
        row_count=max_blocks * max_blocks + 17
    )

    assert grid_small == (17, 1, 1)
    assert launch_count_small == 1
    assert grid_large == (max_blocks, 2, 1)
    assert launch_count_large == 1
    assert grid_huge == (max_blocks, max_blocks, 2)
    assert launch_count_huge == 1


@GPU_REQUIRED
def test_dedupe_impacted_edge_and_row_parts_match_unique_baseline(monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", os.environ.get("CONDA_PREFIX") or sys.prefix)
    row_a = gpu_accel.cp.asarray([0, 0, 1, 2], dtype=gpu_accel.cp.int32)
    edge_a = gpu_accel.cp.asarray([1, 1, 3, 0], dtype=gpu_accel.cp.int64)
    row_b = gpu_accel.cp.asarray([2, 3, 3], dtype=gpu_accel.cp.int32)
    edge_b = gpu_accel.cp.asarray([0, 4, 4], dtype=gpu_accel.cp.int64)

    row_out, edge_out, edge_stats = gpu_accel._dedupe_impacted_edge_pair_parts_cp(
        pair_parts=((row_a, edge_a), (row_b, edge_b)),
        base_edge_count=8,
    )
    expected_keys = np.unique(
        np.concatenate(
            (
                np.asarray(row_a.get(), dtype=np.int64) * 8 + np.asarray(edge_a.get(), dtype=np.int64),
                np.asarray(row_b.get(), dtype=np.int64) * 8 + np.asarray(edge_b.get(), dtype=np.int64),
            )
        )
    )
    assert_equal(np.asarray(row_out.get()), (expected_keys // 8).astype(np.int32))
    assert_equal(np.asarray(edge_out.get()), (expected_keys % 8).astype(np.int64))
    assert edge_stats == {"raw_impacted_edge_count": 7, "deduped_impacted_edge_count": int(expected_keys.size)}

    row_out_rows, row_stats = gpu_accel._dedupe_impacted_row_parts_cp(
        row_parts=(
            gpu_accel.cp.asarray([0, 3, 4, 7], dtype=gpu_accel.cp.int32),
            gpu_accel.cp.asarray([4, 7, 8], dtype=gpu_accel.cp.int32),
        ),
        cell_count=4,
        exclude_cell=3,
    )
    assert_equal(np.asarray(row_out_rows.get()), np.array([0, 4, 8], dtype=np.int32))
    assert row_stats == {"raw_impacted_row_count": 7, "deduped_impacted_row_count": 3}


@GPU_REQUIRED
def test_accumulate_direct_epfd_from_link_library_avoids_public_iterator_seam(monkeypatch):
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    wavelength = (2690 * u.MHz).to(u.m, equivalencies=u.spectral())

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=wavelength,
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )

        def _fail_iter(*_args, **_kwargs):
            raise AssertionError("production fused direct-EPFD path should not use iter_direct_epfd_beam_slabs")

        monkeypatch.setattr(
            gpu_accel.GpuSatelliteLinkSelectionLibrary,
            "iter_direct_epfd_beam_slabs",
            _fail_iter,
        )
        result = session.accumulate_direct_epfd_from_link_library(
            link_library=library,
            s1528_pattern_context=tx_context,
            ras_pattern_context=None,
            sat_topo=case["ras_topo"],
            sat_azel=case["ras_sat_azel"],
            orbit_radius_m_per_sat=np.full(int(case["ras_topo"].shape[1]), R_earth.to_value(u.m) + 525_000.0, dtype=np.float32),
            observer_alt_km=0.1,
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            atmosphere_lut_context=None,
            spectrum_plan_context=None,
            cell_spectral_weight=None,
            bandwidth_mhz=5.0,
            power_input_quantity="target_pfd",
            target_pfd_dbw_m2_channel=-120.0,
            satellite_ptx_dbw_channel=None,
            satellite_eirp_dbw_channel=None,
            n_links=1,
            ras_service_cell_index=0,
            ras_pointing_mode="ras_station",
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            target_alt_km=525.0,
            use_ras_station_alt_for_co=False,
            include_epfd=False,
            include_prx_total=False,
            include_per_satellite_prx=False,
            include_total_pfd=False,
            include_per_satellite_pfd=False,
            include_diagnostics=True,
            include_full_sat_beam_counts_used=True,
            include_sat_eligible_mask=False,
            output_sat_indices=np.arange(int(case["ras_topo"].shape[1]), dtype=np.int32),
            finalize_working_memory_budget_bytes=None,
            power_working_memory_budget_bytes=None,
            power_sky_slab=None,
            return_device=False,
        )

    assert result["beam_finalize_slab_count"] >= 1
    assert "time_chunk_size" in result["beam_finalize_chunk_shape"]
    assert "sky_chunk_size" in result["beam_finalize_chunk_shape"]
    assert isinstance(result["boresight_compaction_stats"], dict)
    assert result["sat_beam_counts_used_full"] is not None
    session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_direct_epfd_from_link_library_dynamic_group_weights_match_precomputed_weights():
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    wavelength = (2690 * u.MHz).to(u.m, equivalencies=u.spectral())
    ras_topo_full = np.array(
        [[[100.0, 50.0, 720.0], [110.0, 55.0, 730.0], [120.0, 60.0, 740.0]]],
        dtype=np.float32,
    )
    group_active_mask = np.asarray(
        [[[True, False], [False, True], [True, True]]],
        dtype=bool,
    )
    cell_group_leakage = np.asarray(
        [[0.25, 0.75], [0.50, 0.20], [0.10, 0.40]],
        dtype=np.float32,
    )
    precomputed_weight = np.einsum(
        "tcg,cg->tc",
        group_active_mask.astype(np.float32),
        cell_group_leakage,
        optimize=True,
    ).astype(np.float32, copy=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=wavelength,
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        common_kwargs = dict(
            link_library=library,
            s1528_pattern_context=tx_context,
            ras_pattern_context=None,
            sat_topo=ras_topo_full,
            sat_azel=case["ras_sat_azel"],
            orbit_radius_m_per_sat=np.full(
                int(ras_topo_full.shape[1]),
                R_earth.to_value(u.m) + 525_000.0,
                dtype=np.float32,
            ),
            observer_alt_km=0.1,
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            atmosphere_lut_context=None,
            spectrum_plan_context=None,
            bandwidth_mhz=5.0,
            power_input_quantity="target_pfd",
            target_pfd_dbw_m2_channel=-120.0,
            satellite_ptx_dbw_channel=None,
            satellite_eirp_dbw_channel=None,
            n_links=1,
            ras_service_cell_index=0,
            ras_pointing_mode="ras_station",
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            target_alt_km=525.0,
            use_ras_station_alt_for_co=False,
            include_epfd=False,
            include_prx_total=False,
            include_per_satellite_prx=False,
            include_total_pfd=True,
            include_per_satellite_pfd=False,
            include_diagnostics=False,
            include_full_sat_beam_counts_used=False,
            include_sat_eligible_mask=False,
            output_sat_indices=np.arange(int(ras_topo_full.shape[1]), dtype=np.int32),
            finalize_working_memory_budget_bytes=None,
            power_working_memory_budget_bytes=None,
            power_sky_slab=1,
            spectral_slab=1,
            return_device=False,
        )
        static_result = session.accumulate_direct_epfd_from_link_library(
            **common_kwargs,
            cell_spectral_weight=precomputed_weight,
        )
        dynamic_result = session.accumulate_direct_epfd_from_link_library(
            **common_kwargs,
            cell_spectral_weight=None,
            dynamic_group_active_mask=group_active_mask,
            dynamic_cell_group_leakage_factors=cell_group_leakage,
            dynamic_group_valid_mask=None,
            dynamic_power_policy="repeat_per_group",
            dynamic_split_total_group_denominator_mode="configured_groups",
            dynamic_configured_groups_per_cell=np.asarray([2, 2, 2], dtype=np.int32),
        )

    assert_allclose(
        np.asarray(dynamic_result["power_result"]["PFD_total_RAS_STATION_W_m2"], dtype=np.float32),
        np.asarray(static_result["power_result"]["PFD_total_RAS_STATION_W_m2"], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert "spectrum_activity_weighting" in dynamic_result["stage_timings"]
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_reports_current_path_stats_non_boresight():
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            debug_direct_epfd=True,
            return_device=False,
        )

    assert result["debug_direct_epfd"] is True
    assert len(result["debug_direct_epfd_stats"]) == 1
    stats = result["debug_direct_epfd_stats"][0]
    assert stats["repair_launch_count"] >= 1
    assert stats["fast_backend_used"] is True
    assert stats["selector_reference_compared"] is True
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_keeps_close_direct_beams_without_generic_sep():
    sat_topo = np.zeros((1, 3, 2, 2), dtype=np.float32)
    sat_topo[0, 0, :, 1] = np.array([80.0, 70.0], dtype=np.float32)
    sat_topo[0, 1, :, 1] = np.array([20.0, 79.0], dtype=np.float32)
    sat_topo[0, 2, :, 1] = np.array([10.0, 78.0], dtype=np.float32)

    sat_azel = np.zeros((1, 3, 2, 2), dtype=np.float32)
    sat_azel[0, 0, :, 0] = np.array([5.0, 90.0], dtype=np.float32)
    sat_azel[0, 1, :, 0] = np.array([25.0, 40.0], dtype=np.float32)
    sat_azel[0, 2, :, 0] = np.array([35.0, 40.2], dtype=np.float32)
    sat_azel[0, :, :, 1] = np.array(
        [[[4.0, 9.0], [4.5, 10.0], [5.0, 10.1]]],
        dtype=np.float32,
    )[0]

    ras_topo = np.array([[[95.0, 48.0], [120.0, 52.0]]], dtype=np.float32)
    ras_sat_azel = np.array([[[5.0, 4.0], [90.0, 9.0]]], dtype=np.float32)
    sat_beta_max = np.array([20.0, 20.0], dtype=np.float32)
    sat_belt_id = np.array([0, 1], dtype=np.int16)

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=3,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=sat_belt_id,
            beta_max_deg_per_sat=sat_beta_max,
            ras_topo=ras_topo,
            include_counts=True,
            include_payload=True,
        )
        library.add_chunk(0, sat_topo, sat_azel=sat_azel)
        result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=ras_sat_azel,
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )

    assert_allclose(result["ras_retargeted_count"], np.array([1], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["ras_reserved_count"], np.array([1], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["direct_kept_count"], np.array([2], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["repaired_link_count"], np.array([0], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["dropped_link_count"], np.array([0], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["sat_beam_counts_used"], np.array([[1, 2]], dtype=np.int32), atol=0.0, rtol=0.0)
    assert int(result["beam_idx"][0, 1, 0]) == 1
    assert int(result["beam_idx"][0, 1, 1]) == 2
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_feeds_accumulate_ras_power():
    case = _direct_epfd_small_case()
    wavelength = (2690 * u.MHz).to(u.m, equivalencies=u.spectral())
    ras_topo_full = np.array(
        [[[100.0, 50.0, 720.0, 0.0], [110.0, 55.0, 730.0, 0.0], [120.0, 60.0, 740.0, 0.0]]],
        dtype=np.float32,
    )
    ras_sat_azel_full = np.array(
        [[[10.0, 5.0, 0.0], [40.0, 9.0, 0.0], [200.0, 8.0, 0.0]]],
        dtype=np.float32,
    )
    orbit_radius = np.array(
        [
            R_earth.to_value(u.m) + 525_000.0,
            R_earth.to_value(u.m) + 530_000.0,
            R_earth.to_value(u.m) + 540_000.0,
        ],
        dtype=np.float32,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        pointing_context = session.prepare_s1586_pointing_context(elev_range_deg=(15.0, 90.0))
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=wavelength,
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        rx_context = session.prepare_ras_pattern_context(diameter_m=15 * u.m, wavelength_m=wavelength)
        pointings = session.sample_s1586_pointings(pointing_context, n_samples=1, seed=7, return_device=False)
        power_result = session.accumulate_ras_power(
            s1528_pattern_context=tx_context,
            ras_pattern_context=rx_context,
            sat_topo=ras_topo_full,
            sat_azel=ras_sat_azel_full,
            beam_idx=beam_result["beam_idx"],
            beam_alpha_rad=beam_result["beam_alpha_rad"],
            beam_beta_rad=beam_result["beam_beta_rad"],
            telescope_azimuth_deg=pointings["azimuth_deg"],
            telescope_elevation_deg=pointings["elevation_deg"],
            orbit_radius_m_per_sat=orbit_radius,
            observer_alt_km=1.052,
            atmosphere_lut_context=None,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )

    assert power_result["EPFD_W_m2"].shape == (1, 1, pointing_context.n_cells)
    assert power_result["Prx_total_W"].shape == (1, 1, pointing_context.n_cells)
    assert power_result["PFD_total_RAS_STATION_W_m2"].shape == (1,)
    assert power_result["PFD_per_sat_RAS_STATION_W_m2"].shape == (1, 3)
    assert np.all(np.isfinite(power_result["EPFD_W_m2"]))
    assert np.all(np.isfinite(power_result["Prx_total_W"]))
    assert np.all(np.isfinite(power_result["PFD_total_RAS_STATION_W_m2"]))
    assert np.all(np.isfinite(power_result["PFD_per_sat_RAS_STATION_W_m2"]))
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_retargets_all_ras_cell_links():
    sat_topo = np.zeros((1, 2, 3, 2), dtype=np.float32)
    sat_topo[0, 0, :, 1] = np.array([80.0, 79.0, 10.0], dtype=np.float32)
    sat_topo[0, 1, :, 1] = np.array([20.0, 78.0, 77.0], dtype=np.float32)

    sat_azel = np.zeros((1, 2, 3, 2), dtype=np.float32)
    sat_azel[0, 0, :, 0] = np.array([15.0, 35.0, 140.0], dtype=np.float32)
    sat_azel[0, 1, :, 0] = np.array([25.0, 60.0, 200.0], dtype=np.float32)
    sat_azel[0, 0, :, 1] = np.array([5.0, 6.0, 15.0], dtype=np.float32)
    sat_azel[0, 1, :, 1] = np.array([10.0, 7.0, 8.0], dtype=np.float32)

    ras_topo = np.array([[[100.0, 50.0], [120.0, 52.0], [150.0, 30.0]]], dtype=np.float32)
    ras_sat_azel = np.array([[[10.0, 4.0], [40.0, 5.0], [200.0, 8.0]]], dtype=np.float32)
    sat_beta_max = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    sat_belt_id = np.array([0, 0, 1], dtype=np.int16)

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=3,
            min_elevation_deg=0.0,
            n_links=2,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=sat_belt_id,
            beta_max_deg_per_sat=sat_beta_max,
            ras_topo=ras_topo,
            include_counts=True,
            include_payload=True,
        )
        library.add_chunk(0, sat_topo, sat_azel=sat_azel)
        result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=ras_sat_azel,
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )

    assert_allclose(result["ras_retargeted_count"], np.array([2], dtype=np.int32), atol=0.0, rtol=0.0)
    assert_allclose(result["sat_beam_counts_used"], np.array([[1, 2, 1]], dtype=np.int32), atol=0.0, rtol=0.0)
    assert int(np.sum(result["sat_beam_counts_used"], dtype=np.int64)) == 4
    assert int(np.count_nonzero(result["beam_idx"][0] == -2)) == 2
    assert_allclose(result["beam_alpha_rad"][0, 0, 0], np.deg2rad(10.0), atol=1e-7, rtol=0.0)
    assert_allclose(result["beam_alpha_rad"][0, 1, 0], np.deg2rad(40.0), atol=1e-7, rtol=0.0)
    assert_allclose(result["beam_beta_rad"][0, 0, 0], np.deg2rad(4.0), atol=1e-7, rtol=0.0)
    assert_allclose(result["beam_beta_rad"][0, 1, 0], np.deg2rad(5.0), atol=1e-7, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_select_satellite_links_boresight_theta2_matches_cpu_selector():
    case = _boresight_direct_epfd_case()
    expected = satsim.select_satellite_links(
        case["sat_topo"],
        min_elevation_deg=0.0,
        n_links=1,
        n_beam=2,
        strategy="max_elevation",
        sat_azel=case["sat_azel"],
        ras_topo=case["ras_topo"],
        beta_max_deg_per_sat=case["sat_beta_max"],
        sat_belt_id_per_sat=case["sat_belt_id"],
        include_counts=True,
        include_payload=True,
        include_eligible_mask=True,
        prefer_numba=False,
        boresight_pointing_azimuth_deg=case["pointing_az_deg"],
        boresight_pointing_elevation_deg=case["pointing_el_deg"],
        boresight_theta2_deg=5.0,
        boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        result = session.select_satellite_links(
            case["sat_topo"],
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_azel=case["sat_azel"],
            ras_topo=case["ras_topo"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            sat_belt_id_per_sat=case["sat_belt_id"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            return_device=False,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=None,
            boresight_theta2_deg=5.0,
            boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
        )

    for key in expected:
        if expected[key].dtype.kind == "f":
            assert_allclose(result[key], expected[key], atol=0.0, rtol=0.0, equal_nan=True)
        else:
            assert_equal(result[key], expected[key])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_boresight_shapes_and_power_outputs():
    case = _boresight_direct_epfd_case()
    wavelength = (2690 * u.MHz).to(u.m, equivalencies=u.spectral())

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        assert_equal(
            beam_result["sat_beam_counts_used"],
            np.count_nonzero(
                beam_result["beam_idx"] >= 0,
                axis=-1,
            ),
        )

        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=wavelength,
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        rx_context = session.prepare_ras_pattern_context(diameter_m=15 * u.m, wavelength_m=wavelength)
        power_result = session.accumulate_ras_power(
            s1528_pattern_context=tx_context,
            ras_pattern_context=rx_context,
            sat_topo=case["ras_topo_full"],
            sat_azel=case["ras_sat_azel_full"],
            beam_idx=beam_result["beam_idx"],
            beam_alpha_rad=beam_result["beam_alpha_rad"],
            beam_beta_rad=beam_result["beam_beta_rad"],
            telescope_azimuth_deg=case["pointing_az_deg"],
            telescope_elevation_deg=case["pointing_el_deg"],
            orbit_radius_m_per_sat=case["orbit_radius_m"],
            observer_alt_km=1.052,
            atmosphere_lut_context=None,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )

    assert beam_result["beam_idx"].shape == (1, 2, 2, 2)
    assert beam_result["beam_alpha_rad"].shape == (1, 2, 2, 2)
    assert beam_result["beam_beta_rad"].shape == (1, 2, 2, 2)
    assert beam_result["sat_beam_counts_used"].shape == (1, 2, 2)
    assert_equal(beam_result["ras_retargeted_count"], np.array([[1, 1]], dtype=np.int32))
    assert_equal(beam_result["sat_beam_counts_used"], np.array([[[0, 1], [1, 0]]], dtype=np.int32))
    assert int(np.count_nonzero(beam_result["beam_idx"] == -2)) >= 1

    assert power_result["EPFD_W_m2"].shape == (1, 1, 2)
    assert power_result["Prx_total_W"].shape == (1, 1, 2)
    assert power_result["PFD_total_RAS_STATION_W_m2"].shape == (1, 1, 2)
    assert power_result["PFD_per_sat_RAS_STATION_W_m2"].shape == (1, 1, 2, 2)
    assert np.all(np.isfinite(power_result["EPFD_W_m2"]))
    assert np.all(np.isfinite(power_result["Prx_total_W"]))
    assert np.all(np.isfinite(power_result["PFD_total_RAS_STATION_W_m2"]))
    assert np.all(np.isfinite(power_result["PFD_per_sat_RAS_STATION_W_m2"]))
    session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_ras_power_supports_pfd_only_outputs():
    case = _boresight_direct_epfd_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )

        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=(15 * u.cm).to_value(u.m),
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        power_result = session.accumulate_ras_power(
            s1528_pattern_context=tx_context,
            ras_pattern_context=None,
            sat_topo=case["ras_topo_full"],
            sat_azel=case["ras_sat_azel_full"],
            beam_idx=beam_result["beam_idx"],
            beam_alpha_rad=beam_result["beam_alpha_rad"],
            beam_beta_rad=beam_result["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m"],
            observer_alt_km=1.052,
            atmosphere_lut_context=None,
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )

    assert "EPFD_W_m2" not in power_result
    assert "Prx_total_W" not in power_result
    assert power_result["PFD_total_RAS_STATION_W_m2"].shape == (1, 1, 2)
    assert power_result["PFD_per_sat_RAS_STATION_W_m2"].shape == (1, 1, 2, 2)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_ras_power_rejects_all_false_output_flags():
    case = _boresight_direct_epfd_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=(15 * u.cm).to_value(u.m),
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        with pytest.raises(ValueError, match="At least one accumulate_ras_power output flag must be enabled"):
            session.accumulate_ras_power(
                s1528_pattern_context=tx_context,
                ras_pattern_context=None,
                sat_topo=case["ras_topo_full"],
                sat_azel=case["ras_sat_azel_full"],
                beam_idx=np.full((1, 2, 2, 2), -1, dtype=np.int16),
                beam_alpha_rad=np.zeros((1, 2, 2, 2), dtype=np.float32),
                beam_beta_rad=np.zeros((1, 2, 2, 2), dtype=np.float32),
                telescope_azimuth_deg=None,
                telescope_elevation_deg=None,
                orbit_radius_m_per_sat=case["orbit_radius_m"],
                observer_alt_km=1.052,
                atmosphere_lut_context=None,
                include_epfd=False,
                include_prx_total=False,
                include_total_pfd=False,
                include_per_satellite_pfd=False,
                return_device=False,
            )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_output_sat_indices_matches_full_boresight():
    case = _boresight_direct_epfd_case()
    subset = np.array([1], dtype=np.int32)
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            return_device=False,
        )
        reduced = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            output_sat_indices=subset,
            return_device=False,
        )

    assert_equal(reduced["beam_idx"], full["beam_idx"][:, :, subset, :])
    assert_allclose(reduced["beam_alpha_rad"], full["beam_alpha_rad"][:, :, subset, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(reduced["beam_beta_rad"], full["beam_beta_rad"][:, :, subset, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_equal(reduced["sat_beam_counts_used"], full["sat_beam_counts_used"][:, :, subset])
    assert_equal(reduced["sat_beam_counts_used_full"], full["sat_beam_counts_used_full"])
    assert_equal(reduced["ras_retargeted_count"], full["ras_retargeted_count"])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_output_sat_indices_accepts_cupy_boresight():
    case = _boresight_direct_epfd_case()
    subset_host = np.array([1], dtype=np.int32)
    subset_device = gpu_accel.cp.asarray(subset_host, dtype=gpu_accel.cp.int32)
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            return_device=False,
        )
        reduced = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            output_sat_indices=subset_device,
            return_device=False,
        )

    assert_equal(reduced["beam_idx"], full["beam_idx"][:, :, subset_host, :])
    assert_allclose(reduced["beam_alpha_rad"], full["beam_alpha_rad"][:, :, subset_host, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(reduced["beam_beta_rad"], full["beam_beta_rad"][:, :, subset_host, :], atol=0.0, rtol=0.0, equal_nan=True)
    assert_equal(reduced["sat_beam_counts_used"], full["sat_beam_counts_used"][:, :, subset_host])
    assert_equal(reduced["sat_beam_counts_used_full"], full["sat_beam_counts_used_full"])
    assert_equal(reduced["ras_retargeted_count"], full["ras_retargeted_count"])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_compact_boresight_matches_full_visible_subset():
    case, visible_subset = _append_inactive_ras_satellite(_boresight_direct_epfd_case())
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            output_sat_indices=visible_subset,
            return_device=False,
        )
        compact = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"][:, visible_subset, :],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            include_full_sat_beam_counts_used=True,
            output_sat_indices=visible_subset,
            return_device=False,
        )

    assert_equal(compact["beam_idx"], full["beam_idx"])
    assert_allclose(compact["beam_alpha_rad"], full["beam_alpha_rad"], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(compact["beam_beta_rad"], full["beam_beta_rad"], atol=0.0, rtol=0.0, equal_nan=True)
    assert_equal(compact["sat_beam_counts_used"], full["sat_beam_counts_used"])
    assert_equal(compact["sat_beam_counts_used_full"], full["sat_beam_counts_used_full"])
    assert_equal(compact["ras_retargeted_count"], full["ras_retargeted_count"])
    assert_equal(compact["repaired_link_count"], full["repaired_link_count"])
    assert_equal(compact["dropped_link_count"], full["dropped_link_count"])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_matches_unbatched_with_small_budget():
    case = _direct_epfd_small_case()
    case["sat_topo"] = np.repeat(case["sat_topo"], 3, axis=0)
    case["sat_azel"] = np.repeat(case["sat_azel"], 3, axis=0)
    case["ras_topo"] = np.repeat(case["ras_topo"], 3, axis=0)
    case["ras_sat_azel"] = np.repeat(case["ras_sat_azel"], 3, axis=0)
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy="max_elevation", rng=0)
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        batched = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            working_memory_budget_bytes=1,
            return_device=False,
        )

    for key in full:
        if np.asarray(full[key]).dtype.kind == "f":
            assert_allclose(batched[key], full[key], atol=0.0, rtol=0.0, equal_nan=True)
        else:
            assert_equal(batched[key], full[key])
    session.close(reset_device=False)


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("strategy", "rng"),
    [
        ("max_elevation", 0),
        ("random", 12345),
    ],
)
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_fast_selector_is_budget_invariant(
    strategy: str,
    rng: int,
):
    case = _direct_epfd_small_case()
    case["sat_topo"] = np.repeat(case["sat_topo"], 3, axis=0)
    case["sat_azel"] = np.repeat(case["sat_azel"], 3, axis=0)
    case["ras_topo"] = np.repeat(case["ras_topo"], 3, axis=0)
    case["ras_sat_azel"] = np.repeat(case["ras_sat_azel"], 3, axis=0)
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = _prepare_direct_epfd_library(session, case, strategy=strategy, rng=rng)
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        tiny_budget = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            working_memory_budget_bytes=1,
            return_device=False,
        )

    for key in full:
        if np.asarray(full[key]).dtype.kind == "f":
            assert_allclose(tiny_budget[key], full[key], atol=0.0, rtol=0.0, equal_nan=True)
        else:
            assert_equal(tiny_budget[key], full[key])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_random_strategy_is_repeatable():
    case = _direct_epfd_small_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library_a = _prepare_direct_epfd_library(session, case, strategy="random", rng=2026)
        result_a = library_a.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        library_b = _prepare_direct_epfd_library(session, case, strategy="random", rng=2026)
        result_b = library_b.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )

    for key in result_a:
        if np.asarray(result_a[key]).dtype.kind == "f":
            assert_allclose(result_b[key], result_a[key], atol=0.0, rtol=0.0, equal_nan=True)
        else:
            assert_equal(result_b[key], result_a[key])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_boresight_matches_unbatched_with_small_budget():
    case = _boresight_direct_epfd_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        full = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        batched = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            working_memory_budget_bytes=1,
            return_device=False,
        )

    for key in full:
        if np.asarray(full[key]).dtype.kind == "f":
            assert_allclose(batched[key], full[key], atol=0.0, rtol=0.0, equal_nan=True)
        else:
            assert_equal(batched[key], full[key])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_staged_backend_defers_dense_boresight_masks():
    case = _boresight_direct_epfd_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=False,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        assert library.boresight_full_mask_cp is None
        assert library.boresight_partial_mask_cp is None

        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )

    assert library.boresight_full_mask_cp is None
    assert library.boresight_partial_mask_cp is None
    assert result["beam_idx"].shape == (1, 2, 2, 2)
    assert_equal(
        result["sat_beam_counts_used"],
        np.count_nonzero(result["beam_idx"] >= 0, axis=-1),
    )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_staged_finalize_defers_dense_boresight_masks():
    case = _boresight_direct_epfd_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)

    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta2_deg=5.0,
            boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
        )
        assert library.boresight_full_mask_cp is None
        assert library.boresight_partial_mask_cp is None

        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        result = library.finalize(return_device=False)

    assert library.boresight_full_mask_cp is None
    assert library.boresight_partial_mask_cp is None
    assert result["assignments"].shape == (1, 2, 2, 1)
    assert "sat_eligible_mask" in result
    session.close(reset_device=False)


@GPU_REQUIRED
def test_staged_public_boresight_paths_avoid_legacy_mask_and_selector_helpers(
    monkeypatch: pytest.MonkeyPatch,
):
    case = _boresight_direct_epfd_case()

    def _fail_legacy(*_args, **_kwargs):
        raise AssertionError("legacy boresight helper should not run on staged production path")

    monkeypatch.setattr(gpu_accel, "_ensure_boresight_masks_cp", _fail_legacy)
    monkeypatch.setattr(gpu_accel, "_get_link_library_selector_candidate_view_cp", _fail_legacy)
    monkeypatch.setattr(gpu_accel, "_finalize_boresight_link_selection_library_cupy", _fail_legacy)

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    common_kwargs = dict(
        time_count=1,
        cell_count=2,
        sat_count=2,
        min_elevation_deg=0.0,
        n_links=1,
        n_beam=2,
        strategy="max_elevation",
        sat_belt_id_per_sat=case["sat_belt_id"],
        beta_max_deg_per_sat=case["sat_beta_max"],
        ras_topo=case["ras_topo"],
        include_counts=True,
        include_payload=True,
        include_eligible_mask=True,
        boresight_pointing_azimuth_deg=case["pointing_az_deg"],
        boresight_pointing_elevation_deg=case["pointing_el_deg"],
        boresight_theta2_deg=5.0,
        boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
    )

    with session.activate():
        library = session.prepare_satellite_link_selection_library(**common_kwargs)
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        finalize_result = library.finalize(return_device=False)
        direct_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        select_result = session.select_satellite_links(
            case["sat_topo"],
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_azel=case["sat_azel"],
            ras_topo=case["ras_topo"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            sat_belt_id_per_sat=case["sat_belt_id"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            return_device=False,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta2_deg=5.0,
            boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
        )

    assert finalize_result["assignments"].shape == (1, 2, 2, 1)
    assert direct_result["beam_idx"].shape == (1, 2, 2, 2)
    assert select_result["assignments"].shape == (1, 2, 2, 1)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_satellite_link_selection_library_finalize_direct_epfd_beams_reports_current_path_stats_staged():
    case = _boresight_direct_epfd_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    common_kwargs = dict(
        time_count=1,
        cell_count=2,
        sat_count=2,
        min_elevation_deg=0.0,
        n_links=1,
        n_beam=2,
        strategy="max_elevation",
        sat_belt_id_per_sat=case["sat_belt_id"],
        beta_max_deg_per_sat=case["sat_beta_max"],
        ras_topo=case["ras_topo"],
        include_counts=True,
        include_payload=True,
        include_eligible_mask=False,
        boresight_pointing_azimuth_deg=case["pointing_az_deg"],
        boresight_pointing_elevation_deg=case["pointing_el_deg"],
        boresight_theta1_deg=5.0,
    )

    with session.activate():
        library = session.prepare_satellite_link_selection_library(**common_kwargs)
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            debug_direct_epfd=True,
            return_device=False,
        )

    assert result["debug_direct_epfd"] is True
    assert len(result["debug_direct_epfd_stats"]) >= 1
    for stats in result["debug_direct_epfd_stats"]:
        assert stats["impacted_edge_extract_count"] >= 0
        assert stats["removed_base_edge_count"] >= 0
        assert stats["repair_row_count"] >= 0
        assert stats["local_sat_count_peak"] >= 0
    session.close(reset_device=False)


@GPU_REQUIRED
def test_allocate_direct_guard_state_uses_dummy_shape_when_disabled():
    alpha_cp, beta_cp, present_cp = gpu_accel._allocate_direct_guard_state_cp(
        shape=(4, 3, 2),
        enabled=False,
    )

    assert tuple(alpha_cp.shape) == (1, 1, 1)
    assert tuple(beta_cp.shape) == (1, 1, 1)
    assert tuple(present_cp.shape) == (1, 1, 1)


@GPU_REQUIRED
def test_positive_value_range_dbw_cp_matches_expected_bounds():
    values = np.array(
        [
            [
                [1.0e-6, 1.0e-5],
                [np.nan, 1.0e-4],
            ]
        ],
        dtype=np.float32,
    )

    actual = gpu_accel._positive_value_range_dbw_cp(
        values,
        bw_mhz=5.0,
    )

    expected_vals = np.array([1.0e-6, 1.0e-5, 1.0e-4], dtype=np.float32)
    expected_dbw = 10.0 * np.log10(expected_vals) + 10.0 * np.log10(5.0)
    assert actual is not None
    assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.array([np.min(expected_dbw), np.max(expected_dbw)], dtype=np.float32),
        atol=0.0,
        rtol=0.0,
    )


@GPU_REQUIRED
def test_positive_value_range_dbw_cp_accepts_time_satellite_shape():
    values = np.array(
        [
            [1.0e-6, 1.0e-5],
            [np.nan, 1.0e-4],
        ],
        dtype=np.float32,
    )

    actual = gpu_accel._positive_value_range_dbw_cp(
        values,
        bw_mhz=5.0,
    )

    expected_vals = np.array([1.0e-6, 1.0e-5, 1.0e-4], dtype=np.float32)
    expected_dbw = 10.0 * np.log10(expected_vals) + 10.0 * np.log10(5.0)
    assert actual is not None
    assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.array([np.min(expected_dbw), np.max(expected_dbw)], dtype=np.float32),
        atol=0.0,
        rtol=0.0,
    )


@GPU_REQUIRED
def test_histogram_per_satellite_value_vs_elevation_cp_matches_numpy_reference():
    values = np.array(
        [
            [
                [1.0e-6, 1.0e-5],
                [1.0e-4, 1.0e-3],
            ]
        ],
        dtype=np.float32,
    )
    elevations = np.array([[10.0, 40.0]], dtype=np.float32)
    elev_edges = np.array([0.0, 20.0, 60.0], dtype=np.float32)
    value_edges = np.array([-70.0, -50.0, -30.0, -10.0], dtype=np.float32)

    hist_cp = gpu_accel._histogram_per_satellite_value_vs_elevation_cp(
        value_w_per_mhz=values,
        sat_elevation_deg=elevations,
        elevation_edges_deg=elev_edges,
        value_edges_dbw=value_edges,
        bw_mhz=5.0,
        sky_slab=1,
    )
    hist = gpu_accel.copy_device_to_host(hist_cp)

    expected = np.zeros((2, 3), dtype=np.int64)
    values_dbw = 10.0 * np.log10(values) + 10.0 * np.log10(5.0)
    for sat_idx, elev in enumerate(elevations[0]):
        x_bin = int(np.searchsorted(elev_edges, elev, side="right") - 1)
        if x_bin < 0 or x_bin >= expected.shape[0]:
            continue
        for sky_idx in range(values.shape[2]):
            value_dbw = float(values_dbw[0, sat_idx, sky_idx])
            y_bin = int(np.searchsorted(value_edges, value_dbw, side="right") - 1)
            if 0 <= y_bin < expected.shape[1]:
                expected[x_bin, y_bin] += 1

    assert_equal(hist, expected)


@GPU_REQUIRED
def test_histogram_per_satellite_value_vs_elevation_cp_accepts_time_satellite_shape():
    values = np.array(
        [
            [1.0e-6, 1.0e-5],
            [1.0e-4, 1.0e-3],
        ],
        dtype=np.float32,
    )
    elevations = np.array(
        [
            [10.0, 40.0],
            [15.0, 35.0],
        ],
        dtype=np.float32,
    )
    elev_edges = np.array([0.0, 20.0, 60.0], dtype=np.float32)
    value_edges = np.array([-70.0, -50.0, -30.0, -10.0], dtype=np.float32)

    hist_cp = gpu_accel._histogram_per_satellite_value_vs_elevation_cp(
        value_w_per_mhz=values,
        sat_elevation_deg=elevations,
        elevation_edges_deg=elev_edges,
        value_edges_dbw=value_edges,
        bw_mhz=5.0,
        sky_slab=1,
    )
    hist = gpu_accel.copy_device_to_host(hist_cp)

    expected = np.zeros((2, 3), dtype=np.int64)
    values_dbw = 10.0 * np.log10(values) + 10.0 * np.log10(5.0)
    for time_idx in range(values.shape[0]):
        for sat_idx in range(values.shape[1]):
            elev = float(elevations[time_idx, sat_idx])
            x_bin = int(np.searchsorted(elev_edges, elev, side="right") - 1)
            if x_bin < 0 or x_bin >= expected.shape[0]:
                continue
            value_dbw = float(values_dbw[time_idx, sat_idx])
            y_bin = int(np.searchsorted(value_edges, value_dbw, side="right") - 1)
            if 0 <= y_bin < expected.shape[1]:
                expected[x_bin, y_bin] += 1

    assert_equal(hist, expected)


@GPU_REQUIRED
def test_cone_valid_selected_links_are_subset_of_full_eligible_mask():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_DWARNER)
    obs_context = session.prepare_observer_context(_example_observers())
    mjds = _example_mjds()[:2]
    min_elev = np.array([5.0, 5.0], dtype=np.float64)
    beta_max = np.array([180.0, 45.0], dtype=np.float32)
    belt_id = np.array([0, 1], dtype=np.int16)

    with session.activate():
        orbit = session.propagate_orbit(mjds, sat_context, on_error="coerce_to_nan")
        ras_topo = session.derive_from_eci(
            orbit,
            observer_context=obs_context,
            observer_slice=slice(0, 1),
            do_topo=True,
            do_sat_azel=False,
            return_device=True,
        )["topo"]
        cell_geom = session.derive_from_eci(
            orbit,
            observer_context=obs_context,
            observer_slice=slice(1, 2),
            do_topo=True,
            do_sat_azel=True,
            return_device=True,
        )
        result = session.select_satellite_links(
            cell_geom["topo"],
            sat_azel=cell_geom["sat_azel"],
            ras_topo=ras_topo,
            min_elevation_deg=min_elev,
            beta_max_deg_per_sat=beta_max,
            sat_belt_id_per_sat=belt_id,
            n_links=1,
            strategy="max_elevation",
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            return_device=False,
        )

    assignments = result["assignments"]
    cone_ok = result["cone_ok"]
    eligible = result["sat_eligible_mask"]
    for t in range(assignments.shape[0]):
        for c in range(assignments.shape[1]):
            sat = int(assignments[t, c, 0])
            if sat >= 0 and bool(cone_ok[t, c, 0]):
                assert bool(eligible[t, c, sat])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_activate_supports_decorator():
    session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
    seen_active: list[bool] = []

    @session.activate()
    def _run_once() -> bool:
        seen_active.append(session.is_active)
        return session.is_active

    assert _run_once() is True
    assert seen_active == [True]
    session.close(reset_device=False)


@GPU_REQUIRED
def test_auto_threshold_cache_can_be_calibrated_and_reused(monkeypatch: pytest.MonkeyPatch):
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    cache_path = Path(".pytest_gpu_auto_thresholds.json")
    if cache_path.exists():
        cache_path.unlink()
    monkeypatch.setattr(gpu_accel, "_AUTO_THRESHOLD_CACHE_PATH", cache_path)
    calls: list[np.dtype] = []

    def _fake_calibration(*, session, compute_dtype):
        del session
        calls.append(np.dtype(compute_dtype))
        return {
            "operation": "true_angular_distance",
            "threshold_elements": 1024,
            "sizes": [256, 1024],
            "cpu_timings_s": [0.2, 0.8],
            "gpu_timings_s": [0.5, 0.4],
            "compute_dtype": np.dtype(compute_dtype).name,
            "calibrated_at_unix_s": 1.0,
        }

    monkeypatch.setattr(gpu_accel, "_calibrate_angular_threshold_entry", _fake_calibration)
    first = gpu_accel.calibrate_auto_thresholds(session=session, force=True)
    second = gpu_accel.calibrate_auto_thresholds(session=session, force=False)
    cached = gpu_accel.get_auto_thresholds()

    assert "float32" in first
    assert "float32" in second
    assert len(calls) == 1
    assert len(cached) == 1
    if cache_path.exists():
        cache_path.unlink()
    session.close(reset_device=False)


def test_true_angular_distance_auto_force_backend_threshold_routing(monkeypatch: pytest.MonkeyPatch):
    # Ensure CUDA appears available so the auto function reaches the
    # threshold routing path instead of short-circuiting to CPU.
    # This can be needed when numba's CUDA context was invalidated by
    # earlier test modules (e.g. test_earthgrid JIT compilation).
    monkeypatch.setattr(gpu_accel, "_has_cuda", lambda: True)
    l1 = np.array([0.0, 15.0], dtype=np.float64)
    b1 = np.array([0.0, -10.0], dtype=np.float64)
    l2 = np.array([90.0, 20.0], dtype=np.float64)
    b2 = np.array([0.0, 5.0], dtype=np.float64)
    expected = true_angular_distance(l1 * u.deg, b1 * u.deg, l2 * u.deg, b2 * u.deg).value

    monkeypatch.setattr(
        gpu_accel,
        "_ensure_angular_threshold_entry",
        lambda **kwargs: {"threshold_elements": 1},
    )

    called_gpu: list[bool] = []

    def _fake_gpu(*args, **kwargs):
        called_gpu.append(True)
        return expected

    monkeypatch.setattr(gpu_accel, "true_angular_distance_gpu", _fake_gpu)
    assert_allclose(gpu_accel.true_angular_distance_auto(l1, b1, l2, b2), expected)
    assert called_gpu == [True]

    monkeypatch.setattr(
        gpu_accel,
        "_ensure_angular_threshold_entry",
        lambda **kwargs: {"threshold_elements": 1 << 20},
    )
    monkeypatch.setattr(
        gpu_accel,
        "true_angular_distance_gpu",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GPU path should not run")),
    )
    assert_allclose(gpu_accel.true_angular_distance_auto(l1, b1, l2, b2), expected)


def test_top_level_wrapper_matches_cysgp4_core_signature():
    gpu_params = list(inspect.signature(gpu_accel.propagate_many).parameters.values())
    cpu_params = list(inspect.signature(cysgp4.propagate_many).parameters.values())
    assert len(gpu_params) >= len(cpu_params)
    for gpu_param, cpu_param in zip(gpu_params[: len(cpu_params)], cpu_params):
        assert gpu_param.name == cpu_param.name
        assert gpu_param.kind == cpu_param.kind
        assert gpu_param.default == cpu_param.default


@GPU_REQUIRED
def test_earth_model_prepared_contexts_are_distinct():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float64, sat_frame="xyz", watchdog_enabled=False)
    sat_wgs72 = session.prepare_satellite_context(NEAR_TLES, gravity_model=gpu_accel.EARTH_MODEL_WGS72)
    sat_wgs84 = session.prepare_satellite_context(NEAR_TLES, gravity_model=gpu_accel.EARTH_MODEL_WGS84)
    obs_wgs72 = session.prepare_observer_context(_example_observers(), ellipsoid_model=gpu_accel.EARTH_MODEL_WGS72)
    obs_wgs84 = session.prepare_observer_context(_example_observers(), ellipsoid_model=gpu_accel.EARTH_MODEL_WGS84)
    assert sat_wgs72 is not sat_wgs84
    assert obs_wgs72 is not obs_wgs84
    assert sat_wgs72.gravity_model == gpu_accel.EARTH_MODEL_WGS72
    assert sat_wgs84.gravity_model == gpu_accel.EARTH_MODEL_WGS84
    assert obs_wgs72.ellipsoid_model == gpu_accel.EARTH_MODEL_WGS72
    assert obs_wgs84.ellipsoid_model == gpu_accel.EARTH_MODEL_WGS84
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_activation_is_reentrant():
    session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(_example_observers())

    assert not session.is_active
    with session.activate():
        ctx_a = gpu_accel.cuda.current_context()
        with session.activate():
            ctx_b = gpu_accel.cuda.current_context()
            result = session.propagate_many(
                _example_mjds()[:1],
                sat_context,
                observer_context=obs_context,
                do_eci_pos=False,
                do_eci_vel=False,
                do_topo=True,
                do_sat_azel=False,
                return_device=True,
                on_error="coerce_to_nan",
            )
            assert "topo" in result
        assert ctx_a == ctx_b
    session.close(reset_device=False)


@GPU_REQUIRED
@pytest.mark.parametrize("sat_frame", ["xyz", "zxy"])
def test_vallado_matches_cpu_reference_for_all_outputs(sat_frame: str):
    mjds = _example_mjds()
    observers = _example_observers()
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float64,
        sat_frame=sat_frame,
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_VALLADO)
    obs_context = session.prepare_observer_context(observers)

    with session.activate():
        gpu_result = session.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=True,
            do_sat_azel=True,
            do_sat_rotmat=True,
            output_dtype=np.float64,
            return_device=False,
        )
    cpu_result = _cpu_reference(
        mjds,
        NEAR_TLES,
        observers,
        method="vallado",
        sat_frame=sat_frame,
        do_geo=True,
        do_sat_azel=True,
        do_obs_pos=True,
        do_sat_rotmat=True,
    )
    _assert_against_cpu(
        gpu_result,
        cpu_result,
        sat_frame=sat_frame,
        atol_pos=1e-3,
        atol_vel=1e-6,
        atol_angle=1e-4,
    )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_observer_slice_and_memory_plan_work_together():
    mjds = _example_mjds()
    observers = _example_observers()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(observers)

    first_chunk = session.propagate_many(
        mjds,
        sat_context,
        observer_context=obs_context,
        observer_slice=slice(0, 1),
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=False,
        do_sat_rotmat=False,
        output_dtype=np.float32,
        return_device=False,
        max_vram_fraction=1.0e-6,
    )
    assert first_chunk["topo"].shape == (mjds.size, 1, len(NEAR_TLES), 4)
    assert first_chunk["obs_pos"].shape == (mjds.size, 1, 3)
    assert first_chunk["memory_plan"]["needs_chunking"]
    session.close(reset_device=False)


@GPU_REQUIRED
def test_single_observer_fast_path_matches_generic_geometry_path():
    mjds = _example_mjds()[:2]
    observers = _example_observers()
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float64,
        sat_frame="xyz",
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    single_obs_context = session.prepare_observer_context(observers[:1])
    generic_obs_context = session.prepare_observer_context(observers)

    fast_result = session.propagate_many(
        mjds,
        sat_context,
        observer_context=single_obs_context,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=True,
        do_sat_rotmat=True,
        return_device=False,
    )
    generic_result = session.propagate_many(
        mjds,
        sat_context,
        observer_context=generic_obs_context,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=True,
        do_sat_rotmat=True,
        return_device=False,
    )

    assert_allclose(fast_result["eci_pos"], generic_result["eci_pos"], atol=1e-6, rtol=0.0)
    assert_allclose(fast_result["eci_vel"], generic_result["eci_vel"], atol=1e-6, rtol=0.0)
    assert_allclose(fast_result["geo"], generic_result["geo"], atol=1e-6, rtol=0.0)
    assert_allclose(fast_result["sat_rotmat"], generic_result["sat_rotmat"], atol=1e-6, rtol=0.0)
    assert_allclose(fast_result["obs_pos"], generic_result["obs_pos"][:, :1, :], atol=1e-6, rtol=0.0)
    assert_allclose(fast_result["topo"], generic_result["topo"][:, :1, :, :], atol=1e-6, rtol=0.0)
    assert_allclose(fast_result["sat_azel"], generic_result["sat_azel"][:, :1, :, :], atol=1e-6, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_mixed_vallado_case_matches_cpu_reference():
    mjds = _example_mjds()[:2]
    observers = _example_observers()[:1]
    mixed_tles = [NEAR_TLES[0], DEEP_TLE]
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float64, sat_frame="xyz", watchdog_enabled=False)
    obs_context = session.prepare_observer_context(observers)
    gpu_result = session.propagate_many(
        mjds,
        session.prepare_satellite_context(mixed_tles),
        observer_context=obs_context,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=False,
        do_sat_rotmat=True,
        return_device=False,
    )
    cpu_result = _cpu_reference(
        mjds,
        mixed_tles,
        observers,
        method="vallado",
        sat_frame="xyz",
        do_geo=True,
        do_sat_azel=False,
        do_obs_pos=True,
        do_sat_rotmat=True,
    )
    _assert_against_cpu(
        gpu_result,
        cpu_result,
        sat_frame="xyz",
        atol_pos=1e-3,
        atol_vel=1e-6,
        atol_angle=1e-4,
    )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_mixed_dwarner_case_matches_cpu_reference():
    mjds = _example_mjds()[:2]
    observers = _example_observers()[:1]
    mixed_tles = [NEAR_TLES[0], DEEP_TLE]
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float64, sat_frame="xyz", watchdog_enabled=False)
    obs_context = session.prepare_observer_context(observers)
    gpu_result = session.propagate_many(
        mjds,
        session.prepare_satellite_context(mixed_tles, method=gpu_accel.METHOD_DWARNER),
        observer_context=obs_context,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=True,
        do_sat_rotmat=True,
        return_device=False,
        on_error="coerce_to_nan",
    )
    cpu_result = _cpu_reference(
        mjds,
        mixed_tles,
        observers,
        method="dwarner",
        sat_frame="xyz",
        do_geo=True,
        do_sat_azel=True,
        do_obs_pos=True,
        do_sat_rotmat=True,
    )
    _assert_against_cpu(
        gpu_result,
        cpu_result,
        sat_frame="xyz",
        atol_pos=1e-3,
        atol_vel=1e-6,
        atol_angle=1e-4,
    )
    session.close(reset_device=False)


@GPU_REQUIRED
@pytest.mark.parametrize("sat_frame", ["xyz", "zxy"])
def test_dwarner_path_matches_cpu_reference(sat_frame: str):
    mjds = _example_mjds()
    observers = _example_observers()[:1]
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32,
        sat_frame=sat_frame,
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_DWARNER)
    obs_context = session.prepare_observer_context(observers)
    gpu_result = session.propagate_many(
        mjds,
        sat_context,
        observer_context=obs_context,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=True,
        do_sat_rotmat=True,
        output_dtype=np.float32,
        return_device=False,
        on_error="coerce_to_nan",
    )
    cpu_result = _cpu_reference(
        mjds,
        NEAR_TLES,
        observers,
        method="dwarner",
        sat_frame=sat_frame,
        do_geo=True,
        do_sat_azel=True,
        do_obs_pos=True,
        do_sat_rotmat=True,
    )
    _assert_against_cpu(
        gpu_result,
        cpu_result,
        sat_frame=sat_frame,
        atol_pos=5e-2,
        atol_vel=5e-5,
        atol_angle=5e-3,
    )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_wgs84_vallado_orbit_matches_cpu_sgp4_reference():
    mjds = _example_mjds()[:2]
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float64,
        sat_frame="xyz",
        gravity_model=gpu_accel.EARTH_MODEL_WGS84,
        ellipsoid_model=gpu_accel.EARTH_MODEL_WGS84,
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(
        NEAR_TLES,
        method=gpu_accel.METHOD_VALLADO,
        gravity_model=gpu_accel.EARTH_MODEL_WGS84,
    )
    gpu_result = session.propagate_many(
        mjds,
        sat_context,
        observer_context=session.prepare_observer_context(
            _example_observers()[:1],
            ellipsoid_model=gpu_accel.EARTH_MODEL_WGS84,
        ),
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=False,
        do_topo=False,
        do_obs_pos=False,
        do_sat_azel=False,
        do_sat_rotmat=False,
        return_device=False,
        on_error="coerce_to_nan",
    )
    cpu_orbit = _cpu_vallado_sgp4_orbit_reference(
        mjds,
        NEAR_TLES,
        gravity_model=gpu_accel.EARTH_MODEL_WGS84,
    )
    assert_allclose(gpu_result["eci_pos"], cpu_orbit["eci_pos"], atol=1e-3, rtol=0.0)
    assert_allclose(gpu_result["eci_vel"], cpu_orbit["eci_vel"], atol=1e-6, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_gravity_model_change_only_changes_orbit_outputs():
    mjds = _example_mjds()[:2]
    observers = _example_observers()[:1]
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float64, sat_frame="xyz", watchdog_enabled=False)
    sat_wgs72 = session.prepare_satellite_context(NEAR_TLES, gravity_model=gpu_accel.EARTH_MODEL_WGS72)
    sat_wgs84 = session.prepare_satellite_context(NEAR_TLES, gravity_model=gpu_accel.EARTH_MODEL_WGS84)
    obs_wgs72 = session.prepare_observer_context(observers, ellipsoid_model=gpu_accel.EARTH_MODEL_WGS72)
    base = session.propagate_many(
        mjds,
        sat_wgs72,
        observer_context=obs_wgs72,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=False,
        do_sat_rotmat=False,
        return_device=False,
        on_error="coerce_to_nan",
    )
    with pytest.warns(gpu_accel.GpuMixedEarthModelWarning):
        changed = session.propagate_many(
            mjds,
            sat_wgs84,
            observer_context=obs_wgs72,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=True,
            do_sat_azel=False,
            do_sat_rotmat=False,
            return_device=False,
            on_error="coerce_to_nan",
        )
    assert not np.allclose(base["eci_pos"], changed["eci_pos"], atol=1e-9, rtol=0.0)
    assert not np.allclose(base["eci_vel"], changed["eci_vel"], atol=1e-12, rtol=0.0)
    assert_allclose(base["obs_pos"], changed["obs_pos"], atol=0.0, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_ellipsoid_model_change_only_changes_geometry_outputs():
    mjds = _example_mjds()[:2]
    observers = _example_observers()[:1]
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float64, sat_frame="xyz", watchdog_enabled=False)
    sat_wgs72 = session.prepare_satellite_context(NEAR_TLES, gravity_model=gpu_accel.EARTH_MODEL_WGS72)
    obs_wgs72 = session.prepare_observer_context(observers, ellipsoid_model=gpu_accel.EARTH_MODEL_WGS72)
    obs_wgs84 = session.prepare_observer_context(observers, ellipsoid_model=gpu_accel.EARTH_MODEL_WGS84)
    base = session.propagate_many(
        mjds,
        sat_wgs72,
        observer_context=obs_wgs72,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=False,
        do_sat_rotmat=False,
        return_device=False,
        on_error="coerce_to_nan",
    )
    with pytest.warns(gpu_accel.GpuMixedEarthModelWarning):
        changed = session.propagate_many(
            mjds,
            sat_wgs72,
            observer_context=obs_wgs84,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=True,
            do_sat_azel=False,
            do_sat_rotmat=False,
            return_device=False,
            on_error="coerce_to_nan",
        )
    assert_allclose(base["eci_pos"], changed["eci_pos"], atol=0.0, rtol=0.0)
    assert_allclose(base["eci_vel"], changed["eci_vel"], atol=0.0, rtol=0.0)
    assert not np.allclose(base["obs_pos"], changed["obs_pos"], atol=1e-12, rtol=0.0)
    assert not np.allclose(base["geo"], changed["geo"], atol=1e-12, rtol=0.0)
    assert not np.allclose(base["topo"], changed["topo"], atol=1e-12, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_model_override_and_mixed_model_warnings_are_emitted():
    mjds = _example_mjds()[:1]
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES, gravity_model=gpu_accel.EARTH_MODEL_WGS84)
    obs_context = session.prepare_observer_context(
        _example_observers()[:1],
        ellipsoid_model=gpu_accel.EARTH_MODEL_WGS72,
    )
    with pytest.warns((gpu_accel.GpuEarthModelOverrideWarning, gpu_accel.GpuMixedEarthModelWarning)) as caught:
        result = gpu_accel.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            session=session,
            do_eci_pos=False,
            do_eci_vel=False,
            do_geo=False,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=False,
            do_sat_rotmat=False,
            gravity_model=gpu_accel.EARTH_MODEL_WGS72,
            ellipsoid_model=gpu_accel.EARTH_MODEL_WGS84,
            return_device=False,
            on_error="coerce_to_nan",
        )
    categories = {warning.category for warning in caught}
    assert gpu_accel.GpuEarthModelOverrideWarning in categories
    assert gpu_accel.GpuMixedEarthModelWarning in categories
    assert "topo" in result
    session.close(reset_device=False)


@GPU_REQUIRED
def test_chunked_device_return_matches_unchunked_results(monkeypatch: pytest.MonkeyPatch):
    mjds = _example_mjds()[:2]
    observers = _example_observers()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float64, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(observers)

    baseline = session.propagate_many(
        mjds,
        sat_context,
        observer_context=obs_context,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=True,
        do_sat_rotmat=True,
        return_device=False,
        on_error="coerce_to_nan",
    )

    original_plan = session.plan_propagation_execution

    def _forced_chunk_plan(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        plan["needs_chunking"] = True
        plan["observer_chunk_size"] = 1
        plan["time_chunk_size"] = 1
        plan["can_return_device"] = True
        return plan

    monkeypatch.setattr(session, "plan_propagation_execution", _forced_chunk_plan)
    with session.activate():
        chunked = session.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=True,
            do_sat_azel=True,
            do_sat_rotmat=True,
            return_device=True,
            on_error="coerce_to_nan",
        )
        chunked_host = {
            key: value.copy_to_host()
            for key, value in chunked.items()
            if key != "memory_plan"
        }

    for key in ("eci_pos", "eci_vel", "geo", "obs_pos", "topo", "sat_azel", "sat_rotmat"):
        assert_allclose(chunked_host[key], baseline[key], atol=1e-6, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_host_spill_plan_is_reported_for_tight_vram_budget():
    mjds = _example_mjds()[:2]
    observers = _example_observers()
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32,
        sat_frame="xyz",
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(observers)

    result = session.propagate_many(
        mjds,
        sat_context,
        observer_context=obs_context,
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=True,
        do_sat_rotmat=True,
        output_dtype=np.float32,
        return_device=False,
        on_error="coerce_to_nan",
        max_vram_fraction=1.0e-6,
    )
    assert result["memory_plan"]["needs_chunking"]
    assert result["memory_plan"]["uses_host_spill"]
    session.close(reset_device=False)


@GPU_REQUIRED
def test_probe_visibility_profile_reports_visible_satellite_count():
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32,
        sat_frame="xyz",
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(_example_observers()[:1])

    profile = session.probe_visibility_profile(
        _example_mjds()[:1],
        sat_context,
        observer_context=obs_context,
        observer_slice=slice(0, 1),
        output_dtype=np.float32,
    )

    assert profile["observer_count"] == 1
    assert profile["satellite_count"] == len(NEAR_TLES)
    assert 0 <= profile["visible_satellite_count"] <= len(NEAR_TLES)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_estimate_step2_batch_memory_counts_rx_workspace():
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32,
        sat_frame="xyz",
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(_example_observers()[:1])
    sampler_context = session.prepare_angle_sampler_context(_load_step2_sampler())
    pointing_context = session.prepare_s1586_pointing_context(elev_range_deg=(15.0, 90.0))
    s1528_context = session.prepare_s1528_pattern_context(
        wavelength_m=0.21,
        lt_m=2.4,
        lr_m=1.1,
        slr_db=20.0 * cnv.dB,
        l=4,
        far_sidelobe_start_deg=20.0,
        far_sidelobe_level_db=-10.0 * cnv.dBi,
    )
    ras_context = session.prepare_ras_pattern_context(diameter_m=15.0, wavelength_m=0.21)

    estimate = session.estimate_step2_batch_memory(
        time_count=3,
        satellite_context=sat_context,
        observer_context=obs_context,
        angle_sampler_context=sampler_context,
        pointing_context=pointing_context,
        s1528_pattern_context=s1528_context,
        ras_pattern_context=ras_context,
        atmosphere_lut_context=None,
        visible_satellite_count=2,
        n_links=1,
        n_beams=4,
        include_total_pfd=True,
        include_per_satellite_pfd=False,
        observer_slice=slice(0, 1),
        output_dtype=np.float32,
    )

    expected_rx_workspace = 5 * 3 * int(pointing_context.n_cells) * 2 * np.dtype(np.float32).itemsize
    assert estimate["components_bytes"]["power_rx_workspace"] == expected_rx_workspace
    assert estimate["components_bytes"]["filtered_topo"] == 3 * 1 * 2 * 4 * np.dtype(np.float32).itemsize
    assert estimate["peak_bytes"] > estimate["fixed_bytes"] > 0
    session.close(reset_device=False)


@GPU_REQUIRED
def test_plan_step2_time_batch_uses_mocked_runtime_budget(monkeypatch: pytest.MonkeyPatch):
    class _MemInfo:
        def __init__(self, free: int, total: int) -> None:
            self.free = int(free)
            self.total = int(total)

    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32,
        sat_frame="xyz",
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(_example_observers()[:1])
    sampler_context = session.prepare_angle_sampler_context(_load_step2_sampler())
    pointing_context = session.prepare_s1586_pointing_context(elev_range_deg=(15.0, 90.0))
    s1528_context = session.prepare_s1528_pattern_context(
        wavelength_m=0.21,
        lt_m=2.4,
        lr_m=1.1,
        slr_db=20.0 * cnv.dB,
        l=4,
        far_sidelobe_start_deg=20.0,
        far_sidelobe_level_db=-10.0 * cnv.dBi,
    )
    ras_context = session.prepare_ras_pattern_context(diameter_m=15.0, wavelength_m=0.21)

    visible_probe = 2
    free_bytes = 3 * 1024 ** 3
    monkeypatch.setattr(session, "_get_memory_info", lambda: _MemInfo(free=free_bytes, total=free_bytes * 2))

    budget = session.resolve_device_memory_budget_bytes(mode="runtime", headroom_profile="balanced")
    visible_est = min(
        int(sat_context.n_sats),
        int(np.ceil(visible_probe * budget["visible_satellite_factor"])),
    )
    one_step = session.estimate_step2_batch_memory(
        time_count=1,
        satellite_context=sat_context,
        observer_context=obs_context,
        angle_sampler_context=sampler_context,
        pointing_context=pointing_context,
        s1528_pattern_context=s1528_context,
        ras_pattern_context=ras_context,
        atmosphere_lut_context=None,
        visible_satellite_count=visible_est,
        n_links=1,
        n_beams=4,
        include_total_pfd=True,
        include_per_satellite_pfd=False,
        observer_slice=slice(0, 1),
        output_dtype=np.float32,
    )
    expected = scenario.recommend_time_batch_size_linear(
        total_timesteps=20,
        fixed_bytes=int(one_step["fixed_bytes"]),
        per_timestep_bytes=int(one_step["per_timestep_bytes"]),
        budget_bytes=int(budget["effective_budget_bytes"]),
    )

    plan = session.plan_step2_time_batch(
        total_timesteps=20,
        satellite_context=sat_context,
        observer_context=obs_context,
        angle_sampler_context=sampler_context,
        pointing_context=pointing_context,
        s1528_pattern_context=s1528_context,
        ras_pattern_context=ras_context,
        atmosphere_lut_context=None,
        visible_satellite_count=visible_probe,
        n_links=1,
        n_beams=4,
        include_total_pfd=True,
        include_per_satellite_pfd=False,
        mode="runtime",
        headroom_profile="balanced",
        observer_slice=slice(0, 1),
        output_dtype=np.float32,
    )

    assert plan["budget"]["effective_budget_bytes"] == budget["effective_budget_bytes"]
    assert plan["recommended_batch_size"] == expected["recommended_batch_size"]
    assert plan["visible_satellite_count_estimated"] == visible_est
    session.close(reset_device=False)


@GPU_REQUIRED
def test_resolve_device_memory_budget_hybrid_clamps_planning_to_live_headroom(
    monkeypatch: pytest.MonkeyPatch,
):
    class _MemInfo:
        def __init__(self, free: int, total: int) -> None:
            self.free = int(free)
            self.total = int(total)

    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32,
        sat_frame="xyz",
        watchdog_enabled=False,
    )
    monkeypatch.setattr(
        session,
        "_get_memory_info",
        lambda: _MemInfo(free=6 * 1024**3, total=16 * 1024**3),
    )

    budget = session.resolve_device_memory_budget_bytes(
        12.0,
        mode="hybrid",
        headroom_profile="balanced",
    )

    expected_runtime_budget = min(
        int(6 * 1024**3 * 0.70),
        (6 * 1024**3) - (1024 * 1024**2),
    )
    assert budget["runtime_budget_bytes"] == expected_runtime_budget
    assert budget["runtime_advisory_budget_bytes"] == expected_runtime_budget
    assert budget["planning_budget_bytes"] == expected_runtime_budget
    assert budget["hard_budget_bytes"] == 12 * 1024**3
    assert budget["effective_budget_bytes"] == expected_runtime_budget
    assert budget["effective_budget_reason"] == "hybrid cap clamped to live VRAM headroom"
    session.close(reset_device=False)


@GPU_REQUIRED
def test_device_return_refuses_when_final_outputs_do_not_fit_budget():
    mjds = _example_mjds()[:2]
    observers = _example_observers()
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32,
        sat_frame="xyz",
        watchdog_enabled=False,
    )
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(observers)

    with pytest.raises(MemoryError):
        session.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=True,
            do_sat_azel=True,
            do_sat_rotmat=True,
            output_dtype=np.float32,
            return_device=True,
            on_error="coerce_to_nan",
            max_vram_fraction=1.0e-9,
        )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_top_level_wrapper_requires_explicit_session_for_device_return():
    mjds = _example_mjds()[:1]
    observers = _example_observers()[:1]
    with pytest.raises(ValueError, match="return_device=True requires an explicit live GpuScepterSession"):
        gpu_accel.propagate_many(
            mjds[:, np.newaxis, np.newaxis],
            np.asarray(NEAR_TLES, dtype=object)[np.newaxis, np.newaxis, :],
            np.asarray(observers, dtype=object)[np.newaxis, :, np.newaxis],
            do_eci_pos=False,
            do_eci_vel=False,
            do_geo=False,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=False,
            do_sat_rotmat=False,
            return_device=True,
        )


@GPU_REQUIRED
def test_top_level_wrapper_broadcasts_host_outputs():
    mjds = _example_mjds()
    observers = _example_observers()[:1]
    result = gpu_accel.propagate_many(
        mjds[:, np.newaxis, np.newaxis],
        np.asarray(NEAR_TLES, dtype=object)[np.newaxis, np.newaxis, :],
        np.asarray(observers, dtype=object)[np.newaxis, :, np.newaxis],
        do_eci_pos=True,
        do_eci_vel=True,
        do_geo=True,
        do_topo=True,
        do_obs_pos=True,
        do_sat_azel=True,
        do_sat_rotmat=True,
        sat_frame="xyz",
        method=gpu_accel.METHOD_VALLADO,
        compute_dtype=np.float64,
        output_dtype=np.float16,
        return_device=False,
    )
    assert result["eci_pos"].shape == (mjds.size, len(observers), len(NEAR_TLES), 3)
    assert result["geo"].shape == (mjds.size, len(observers), len(NEAR_TLES), 3)
    assert result["obs_pos"].shape == (mjds.size, len(observers), len(NEAR_TLES), 3)
    assert result["sat_rotmat"].shape == (mjds.size, len(observers), len(NEAR_TLES), 3, 3)
    assert result["eci_pos"].dtype == np.float16
    assert result["topo"].dtype == np.float16


@GPU_REQUIRED
def test_top_level_wrapper_can_run_repeatedly_without_explicit_session():
    mjds = _example_mjds()[:2]
    observers = _example_observers()[:1]

    first = gpu_accel.propagate_many(
        mjds[:, np.newaxis, np.newaxis],
        np.asarray(NEAR_TLES, dtype=object)[np.newaxis, np.newaxis, :],
        np.asarray(observers, dtype=object)[np.newaxis, :, np.newaxis],
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=False,
        do_sat_azel=True,
        do_sat_rotmat=False,
        sat_frame="xyz",
        method=gpu_accel.METHOD_DWARNER,
        compute_dtype=np.float32,
        output_dtype=np.float32,
        return_device=False,
    )
    second = gpu_accel.propagate_many(
        mjds[:, np.newaxis, np.newaxis],
        np.asarray(NEAR_TLES, dtype=object)[np.newaxis, np.newaxis, :],
        np.asarray(observers, dtype=object)[np.newaxis, :, np.newaxis],
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=False,
        do_sat_azel=True,
        do_sat_rotmat=False,
        sat_frame="xyz",
        method=gpu_accel.METHOD_DWARNER,
        compute_dtype=np.float32,
        output_dtype=np.float32,
        return_device=False,
    )

    assert_allclose(first["topo"], second["topo"], atol=0.0, rtol=0.0)
    assert_allclose(first["sat_azel"], second["sat_azel"], atol=0.0, rtol=0.0)


@GPU_REQUIRED
def test_session_close_default_hard_close_exits_cleanly_in_subprocess():
    result = _run_python_subprocess(
        """
        import numpy as np
        from scepter import gpu_accel
        from scepter.tests.test_gpu_accel import NEAR_TLES, _example_mjds, _example_observers

        session = gpu_accel.GpuScepterSession(
            compute_dtype=np.float32,
            sat_frame="xyz",
            watchdog_enabled=False,
        )
        sat_context = session.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_DWARNER)
        obs_context = session.prepare_observer_context(_example_observers()[:1])
        _ = session.propagate_many(
            _example_mjds()[:2],
            sat_context,
            observer_context=obs_context,
            do_eci_pos=False,
            do_eci_vel=False,
            do_geo=False,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=True,
            do_sat_rotmat=False,
            output_dtype=np.float32,
            return_device=False,
            on_error="coerce_to_nan",
        )
        session.close()
        print("hard_close_ok")
        """
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "hard_close_ok" in result.stdout


@GPU_REQUIRED
def test_session_close_default_invalidates_cached_device_state_in_subprocess():
    result = _run_python_subprocess(
        """
        import numpy as np
        from scepter import gpu_accel
        from scepter.tests.test_gpu_accel import NEAR_TLES, _example_mjds, _example_observers

        session = gpu_accel.GpuScepterSession(
            compute_dtype=np.float32,
            sat_frame="xyz",
            watchdog_enabled=False,
        )
        sat_context = session.prepare_satellite_context(NEAR_TLES)
        obs_context = session.prepare_observer_context(_example_observers()[:1])
        _ = session.propagate_many(
            _example_mjds()[:2],
            sat_context,
            observer_context=obs_context,
            do_eci_pos=False,
            do_eci_vel=False,
            do_geo=False,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=True,
            do_sat_rotmat=False,
            output_dtype=np.float32,
            return_device=False,
            on_error="coerce_to_nan",
        )
        workspace = next(iter(session._workspace_cache.values()))
        session.close()
        assert sat_context.d_params == {}
        assert sat_context.d_near_indices is None
        assert sat_context.d_deep_indices is None
        assert obs_context.d_observer_ecef_x_km is None
        assert obs_context.d_observer_ecef_y_km is None
        assert obs_context.d_observer_ecef_z_km is None
        assert workspace.d_topo is None
        assert workspace.d_sat_azel is None
        print("invalidated_ok")
        """
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "invalidated_ok" in result.stdout


@GPU_REQUIRED
def test_watchdog_idle_eviction_runs_only_on_owner_thread_reentry(monkeypatch):
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32,
        sat_frame="xyz",
        idle_timeout_s=0.01,
        watchdog_enabled=True,
    )
    clear_calls: list[int] = []
    original_clear = session._clear_session_caches

    def _recording_clear() -> None:
        clear_calls.append(threading.get_ident())
        original_clear()

    monkeypatch.setattr(session, "_clear_session_caches", _recording_clear)
    session._last_used_monotonic = time.monotonic() - 1.0

    time.sleep(0.05)
    assert clear_calls == []

    with session.activate():
        pass

    assert clear_calls == [session.owner_thread_id]
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_close_terminal_close_warns_and_preserves_sibling_session():
    session_a = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    session_b = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session_b.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_DWARNER)
    obs_context = session_b.prepare_observer_context(_example_observers()[:1])

    with pytest.warns(gpu_accel.GpuTerminalCloseWarning):
        session_a.close()

    result = session_b.propagate_many(
        _example_mjds()[:2],
        sat_context,
        observer_context=obs_context,
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=False,
        do_sat_azel=True,
        do_sat_rotmat=False,
        output_dtype=np.float32,
        return_device=False,
        on_error="coerce_to_nan",
    )

    assert result["topo"].shape == (2, 1, len(NEAR_TLES), 4)
    assert result["sat_azel"].shape == (2, 1, len(NEAR_TLES), 3)
    session_b.close(reset_device=False)


@GPU_REQUIRED
def test_top_level_wrapper_accepts_prepared_contexts():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float64, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES)
    obs_context = session.prepare_observer_context(_example_observers()[:1])
    result = gpu_accel.propagate_many(
        _example_mjds()[:2],
        sat_context,
        observer_context=obs_context,
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=False,
        do_sat_azel=True,
        do_sat_rotmat=False,
        return_device=False,
    )
    assert result["topo"].shape == (2, 1, len(NEAR_TLES), 4)
    assert result["sat_azel"].shape == (2, 1, len(NEAR_TLES), 3)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_orbit_only_plus_geometry_matches_full_propagate_many():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float64, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_VALLADO)
    obs_context = session.prepare_observer_context(_example_observers())
    mjds = _example_mjds()[:2]

    with session.activate():
        full = session.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=True,
            do_sat_azel=True,
            do_sat_rotmat=True,
            output_dtype=np.float64,
            return_device=False,
            on_error="coerce_to_nan",
        )
        orbit = session.propagate_orbit(mjds, sat_context, on_error="coerce_to_nan")
        derived = session.derive_from_eci(
            orbit,
            observer_context=obs_context,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=True,
            do_sat_azel=True,
            do_sat_rotmat=True,
            output_dtype=np.float64,
            return_device=False,
        )

    for key in ("eci_pos", "eci_vel", "geo", "topo", "obs_pos", "sat_azel", "sat_rotmat"):
        assert_allclose(full[key], derived[key], atol=0.0, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_hybrid_cpu_eci_to_gpu_geometry_accepts_host_and_device_inputs():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_DWARNER)
    obs_context = session.prepare_observer_context(_example_observers()[:1])
    mjds = _example_mjds()[:2]

    with session.activate():
        orbit = session.propagate_orbit(mjds, sat_context, on_error="coerce_to_nan")
        baseline = session.derive_from_eci(
            orbit,
            observer_context=obs_context,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=True,
            do_sat_rotmat=False,
            output_dtype=np.float32,
            return_device=False,
        )

        host_result = gpu_accel.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            session=session,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=True,
            do_sat_rotmat=False,
            output_dtype=np.float32,
            return_device=False,
            eci_pos=orbit.d_eci_pos.copy_to_host(),
            eci_vel=orbit.d_eci_vel.copy_to_host(),
        )
        device_result = gpu_accel.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            session=session,
            do_eci_pos=True,
            do_eci_vel=True,
            do_geo=True,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=True,
            do_sat_rotmat=False,
            output_dtype=np.float32,
            return_device=True,
            eci_pos=orbit.d_eci_pos,
            eci_vel=orbit.d_eci_vel,
        )

    for key in ("eci_pos", "eci_vel", "geo"):
        assert_allclose(host_result[key][:, 0, :, :], baseline[key], atol=0.0, rtol=0.0)
    for key in ("topo", "sat_azel"):
        assert_allclose(host_result[key], baseline[key], atol=0.0, rtol=0.0)
    assert_allclose(device_result["topo"].copy_to_host(), baseline["topo"], atol=0.0, rtol=0.0)
    assert_allclose(device_result["sat_azel"].copy_to_host(), baseline["sat_azel"], atol=0.0, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_observer_chunking_reuses_orbit_propagation(monkeypatch: pytest.MonkeyPatch):
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_DWARNER)
    obs_context = session.prepare_observer_context(_example_observers())
    mjds = _example_mjds()

    original_propagate_orbit = session.propagate_orbit
    orbit_calls: list[int] = []

    def _counting_propagate_orbit(mjds_arg, satellite_context_arg, *, on_error="raise"):
        orbit_calls.append(int(np.asarray(mjds_arg).size))
        return original_propagate_orbit(mjds_arg, satellite_context_arg, on_error=on_error)

    original_plan = session.plan_propagation_execution

    def _forced_plan(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        plan["needs_chunking"] = True
        plan["observer_chunk_size"] = 1
        plan["time_chunk_size"] = int(np.asarray(args[0]).size)
        plan["can_return_device"] = True
        return plan

    monkeypatch.setattr(session, "propagate_orbit", _counting_propagate_orbit)
    monkeypatch.setattr(session, "plan_propagation_execution", _forced_plan)

    _ = session.propagate_many(
        mjds,
        sat_context,
        observer_context=obs_context,
        do_eci_pos=False,
        do_eci_vel=False,
        do_geo=False,
        do_topo=True,
        do_obs_pos=False,
        do_sat_azel=True,
        do_sat_rotmat=False,
        output_dtype=np.float32,
        return_device=False,
        on_error="coerce_to_nan",
    )
    assert orbit_calls == [mjds.size]
    session.close(reset_device=False)


@GPU_REQUIRED
def test_top_level_wrapper_with_session_matches_session_device_return():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    sat_context = session.prepare_satellite_context(NEAR_TLES, method=gpu_accel.METHOD_DWARNER)
    obs_context = session.prepare_observer_context(_example_observers()[:1])
    mjds = _example_mjds()[:2]

    with session.activate():
        direct = session.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            do_eci_pos=False,
            do_eci_vel=False,
            do_geo=False,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=True,
            do_sat_rotmat=False,
            output_dtype=np.float32,
            return_device=True,
            on_error="coerce_to_nan",
        )
        wrapped = gpu_accel.propagate_many(
            mjds,
            sat_context,
            observer_context=obs_context,
            session=session,
            do_eci_pos=False,
            do_eci_vel=False,
            do_geo=False,
            do_topo=True,
            do_obs_pos=False,
            do_sat_azel=True,
            do_sat_rotmat=False,
            output_dtype=np.float32,
            return_device=True,
            on_error="coerce_to_nan",
        )
        assert_allclose(direct["topo"].copy_to_host(), wrapped["topo"].copy_to_host(), atol=0.0, rtol=0.0)
        assert_allclose(direct["sat_azel"].copy_to_host(), wrapped["sat_azel"].copy_to_host(), atol=0.0, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_resolve_angle_sampler_sources_matches_cpu_fallback_logic():
    sampler = _load_step2_sampler()
    group_id = int(
        np.flatnonzero(
            (np.asarray(sampler.group_raw_counts) >= int(sampler.cond_min_group_samples))
            & (np.asarray(sampler.group_ptr[1:]) > np.asarray(sampler.group_ptr[:-1]))
        )[0]
    )
    belt_fallback_id = int(
        np.flatnonzero(
            (np.asarray(sampler.group_raw_counts) < int(sampler.cond_min_group_samples))
            & (np.asarray(sampler.belt_raw_counts)[0] >= int(sampler.cond_min_belt_samples))
        )[0]
    )
    az_group, el_group = _s1586_cell_center_deg(group_id)
    az_belt, el_belt = _s1586_cell_center_deg(belt_fallback_id)
    sat_azimuth_deg = np.array([[az_group, az_belt, np.nan, az_group]], dtype=np.float32)
    sat_elevation_deg = np.array([[el_group, el_belt, 20.0, el_group]], dtype=np.float32)
    sat_belt_id = np.array([[0, 0, 0, -1]], dtype=np.float32)

    expected = _cpu_expected_sampler_sources(
        sampler,
        sat_azimuth_deg,
        sat_elevation_deg,
        sat_belt_id,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        context = session.prepare_angle_sampler_context(sampler)
        result = session.resolve_angle_sampler_sources(
            context,
            sat_azimuth_deg,
            sat_elevation_deg,
            sat_belt_id,
            return_device=False,
        )
    for key in ("skycell_id", "source_kind", "source_id", "valid_mask"):
        assert_allclose(result[key], expected[key], atol=0.0, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_sample_conditioned_candidate_pools_is_seeded_and_source_specific():
    sampler = _load_step2_sampler()
    group_id = int(
        np.flatnonzero(
            (np.asarray(sampler.group_raw_counts) >= int(sampler.cond_min_group_samples))
            & (np.asarray(sampler.group_ptr[1:]) > np.asarray(sampler.group_ptr[:-1]))
        )[0]
    )
    belt_id = 0
    source_kind = np.array(
        [
            gpu_accel.SAMPLER_SOURCE_GROUP,
            gpu_accel.SAMPLER_SOURCE_BELT,
            gpu_accel.SAMPLER_SOURCE_GLOBAL,
        ],
        dtype=np.int8,
    )
    source_id = np.array([group_id, belt_id, 0], dtype=np.int32)

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        context = session.prepare_angle_sampler_context(sampler)
        first = session.sample_conditioned_candidate_pools(
            context,
            source_kind,
            source_id,
            pool_size=64,
            seed=123,
            return_device=False,
        )
        second = session.sample_conditioned_candidate_pools(
            context,
            source_kind,
            source_id,
            pool_size=64,
            seed=123,
            return_device=False,
        )
    for key in first:
        assert_allclose(first[key], second[key], atol=0.0, rtol=0.0, equal_nan=True)

    group_p0 = int(sampler.group_ptr[group_id])
    group_p1 = int(sampler.group_ptr[group_id + 1])
    belt_p0 = int(sampler.belt_ptr[belt_id])
    belt_p1 = int(sampler.belt_ptr[belt_id + 1])
    assert np.all(np.isin(first["beta_deg"][0], sampler.group_beta_pool[group_p0:group_p1]))
    assert np.all(np.isin(first["alpha_deg"][0], sampler.group_alpha_pool[group_p0:group_p1]))
    assert np.all(np.isin(first["beta_deg"][1], sampler.belt_beta_pool[belt_p0:belt_p1]))
    assert np.all(np.isin(first["alpha_deg"][1], sampler.belt_alpha_pool[belt_p0:belt_p1]))
    assert np.all(np.isfinite(first["beta_deg"][2]))
    assert np.all(first["beta_deg"][2] >= float(sampler.beta_edges[0]))
    assert np.all(first["beta_deg"][2] <= float(sampler.beta_edges[-1]))
    session.close(reset_device=False)


@GPU_REQUIRED
def test_fill_conditioned_beams_streaming_matches_cpu_reference(monkeypatch):
    source_kind = np.array([[gpu_accel.SAMPLER_SOURCE_GROUP, gpu_accel.SAMPLER_SOURCE_GROUP]], dtype=np.int8)
    source_id = np.array([[1, 2]], dtype=np.int32)
    vis_mask = np.array([[True, True]], dtype=bool)
    is_co_sat = np.array([[True, False]], dtype=bool)
    alpha0_rad = np.array([[0.10, 0.20]], dtype=np.float32)
    beta0_rad = np.array([[0.10, 0.10]], dtype=np.float32)
    beta_max = np.array([0.50, 0.50], dtype=np.float32)
    cos_min_sep = float(np.cos(np.deg2rad(10.0)))
    batch_alpha = np.array([[0.12, 1.20], [0.40, 1.60]], dtype=np.float32)
    batch_beta = np.array([[0.11, 0.20], [0.12, 0.30]], dtype=np.float32)

    def _fake_cpu_batch(*args, **kwargs):
        return {
            "alpha_rad": batch_alpha.copy(),
            "beta_rad": batch_beta.copy(),
        }

    def _fake_gpu_batch(*args, **kwargs):
        return {
            "alpha_rad": gpu_accel.cp.asarray(batch_alpha),
            "beta_rad": gpu_accel.cp.asarray(batch_beta),
        }

    monkeypatch.setattr(satsim, "sample_conditioned_candidate_batch_cpu", _fake_cpu_batch)
    monkeypatch.setattr(gpu_accel, "_sample_conditioned_candidate_batch_cp", _fake_gpu_batch)

    cpu_result = satsim.fill_conditioned_beams_streaming_cpu(
        sampler=_load_step2_sampler(),
        source_kind=source_kind,
        source_id=source_id,
        vis_mask_horizon=vis_mask,
        is_co_sat=is_co_sat,
        alpha0_rad=alpha0_rad,
        beta0_rad=beta0_rad,
        beta_max_rad_per_sat=beta_max,
        n_beams=2,
        cos_min_sep=cos_min_sep,
        seed=123,
        chunk_size=2,
        max_rounds=1,
        rescue_chunk_size=2,
        max_rescue_rounds=0,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        context = session.prepare_angle_sampler_context(_load_step2_sampler())
        gpu_result = session.fill_conditioned_beams_streaming(
            context,
            source_kind,
            source_id,
            vis_mask_horizon=vis_mask,
            is_co_sat=is_co_sat,
            alpha0_rad=alpha0_rad,
            beta0_rad=beta0_rad,
            beta_max_rad_per_sat=beta_max,
            n_beams=2,
            cos_min_sep=cos_min_sep,
            seed=123,
            chunk_size=2,
            max_rounds=1,
            rescue_chunk_size=2,
            max_rescue_rounds=0,
            return_device=False,
        )

    for key in ("beam_idx", "beam_alpha_rad", "beam_beta_rad", "beam_valid"):
        assert_allclose(gpu_result[key], cpu_result[key], atol=0.0, rtol=0.0, equal_nan=True)
    assert int(gpu_result["draws_attempted"]) == 4
    assert int(gpu_result["rounds_used"]) == 1
    session.close(reset_device=False)


@GPU_REQUIRED
def test_fill_conditioned_beams_streaming_exclude_ras_radius_matches_cpu_reference(monkeypatch):
    source_kind = np.array([[gpu_accel.SAMPLER_SOURCE_GROUP]], dtype=np.int8)
    source_id = np.array([[1]], dtype=np.int32)
    vis_mask = np.array([[True]], dtype=bool)
    is_co_sat = np.array([[True]], dtype=bool)
    alpha0_rad = np.array([[0.0]], dtype=np.float32)
    beta0_rad = np.array([[0.10]], dtype=np.float32)
    beta_max = np.array([0.50], dtype=np.float32)
    orbit_radius = np.array([R_earth.to_value(u.m) + 525_000.0], dtype=np.float32)
    batch_alpha = np.array([[0.0, 1.2]], dtype=np.float32)
    batch_beta = np.array([[0.10, 0.10]], dtype=np.float32)
    ras_sat_azel = np.array([[[0.0, np.rad2deg(0.10), 0.0]]], dtype=np.float32)

    def _fake_cpu_batch(*args, **kwargs):
        return {
            "alpha_rad": batch_alpha.copy(),
            "beta_rad": batch_beta.copy(),
        }

    def _fake_gpu_batch(*args, **kwargs):
        return {
            "alpha_rad": gpu_accel.cp.asarray(batch_alpha),
            "beta_rad": gpu_accel.cp.asarray(batch_beta),
        }

    monkeypatch.setattr(satsim, "sample_conditioned_candidate_batch_cpu", _fake_cpu_batch)
    monkeypatch.setattr(gpu_accel, "_sample_conditioned_candidate_batch_cp", _fake_gpu_batch)

    cpu_result = satsim.fill_conditioned_beams_streaming_cpu(
        sampler=_load_step2_sampler(),
        source_kind=source_kind,
        source_id=source_id,
        vis_mask_horizon=vis_mask,
        is_co_sat=is_co_sat,
        alpha0_rad=alpha0_rad,
        beta0_rad=beta0_rad,
        beta_max_rad_per_sat=beta_max,
        n_beams=1,
        cos_min_sep=float(np.cos(np.deg2rad(5.0))),
        seed=123,
        chunk_size=2,
        max_rounds=1,
        rescue_chunk_size=2,
        max_rescue_rounds=0,
        beam_placement_policy=satsim.STEP2_BEAM_PLACEMENT_EXCLUDE_RAS_RADIUS,
        ras_sat_azel=ras_sat_azel,
        orbit_radius_m_per_sat=orbit_radius,
        ras_exclusion_radius_km=50.0,
    )

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        context = session.prepare_angle_sampler_context(_load_step2_sampler())
        gpu_result = session.fill_conditioned_beams_streaming(
            context,
            source_kind,
            source_id,
            vis_mask_horizon=vis_mask,
            is_co_sat=is_co_sat,
            alpha0_rad=alpha0_rad,
            beta0_rad=beta0_rad,
            beta_max_rad_per_sat=beta_max,
            n_beams=1,
            cos_min_sep=float(np.cos(np.deg2rad(5.0))),
            seed=123,
            chunk_size=2,
            max_rounds=1,
            rescue_chunk_size=2,
            max_rescue_rounds=0,
            beam_placement_policy=gpu_accel.STEP2_BEAM_PLACEMENT_EXCLUDE_RAS_RADIUS,
            ras_sat_azel=ras_sat_azel,
            orbit_radius_m_per_sat=orbit_radius,
            ras_exclusion_radius_km=50.0,
            return_device=False,
        )

    for key in ("beam_idx", "beam_alpha_rad", "beam_beta_rad", "beam_valid"):
        assert_allclose(gpu_result[key], cpu_result[key], atol=0.0, rtol=0.0, equal_nan=True)
    assert int(gpu_result["beam_idx"][0, 0, 0]) != -2
    session.close(reset_device=False)


@GPU_REQUIRED
def test_make_conditioned_template_plan_selects_expected_modes():
    source_kind = np.array(
        [
            [gpu_accel.SAMPLER_SOURCE_GROUP, gpu_accel.SAMPLER_SOURCE_GROUP, gpu_accel.SAMPLER_SOURCE_BELT],
            [gpu_accel.SAMPLER_SOURCE_GLOBAL, gpu_accel.SAMPLER_SOURCE_INVALID, gpu_accel.SAMPLER_SOURCE_GROUP],
        ],
        dtype=np.int8,
    )
    source_id = np.array([[5, 5, 0], [0, -1, 7]], dtype=np.int32)
    vis_mask = np.array([[True, True, True], [True, False, True]], dtype=bool)
    beta_max = np.array([0.3, 0.4, 0.5], dtype=np.float32)

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        plan_per_source = session.make_conditioned_template_plan(
            source_kind,
            source_id,
            vis_mask,
            beta_max,
            mode=gpu_accel.CONDITIONED_TEMPLATE_MODE_PER_SOURCE,
            pool_size=16,
            template_size=8,
        )
        plan_per_row = session.make_conditioned_template_plan(
            source_kind,
            source_id,
            vis_mask,
            beta_max,
            mode=gpu_accel.CONDITIONED_TEMPLATE_MODE_PER_ROW,
            pool_size=16,
            template_size=8,
        )
        plan_hybrid_small = session.make_conditioned_template_plan(
            source_kind,
            source_id,
            vis_mask,
            beta_max,
            mode=gpu_accel.CONDITIONED_TEMPLATE_MODE_HYBRID,
            pool_size=16,
            template_size=8,
        )
        plan_hybrid_large = session.make_conditioned_template_plan(
            source_kind,
            source_id,
            vis_mask,
            beta_max,
            mode=gpu_accel.CONDITIONED_TEMPLATE_MODE_HYBRID,
            pool_size=1_000_000,
            template_size=8,
        )

    assert plan_per_source.mode_used == gpu_accel.CONDITIONED_TEMPLATE_MODE_PER_SOURCE
    assert plan_per_source.active_row_count == 5
    assert plan_per_source.unit_count == 4
    assert plan_per_row.mode_used == gpu_accel.CONDITIONED_TEMPLATE_MODE_PER_ROW
    assert plan_per_row.unit_count == plan_per_row.active_row_count
    assert plan_hybrid_small.mode_used == gpu_accel.CONDITIONED_TEMPLATE_MODE_PER_ROW
    assert plan_hybrid_large.mode_used == gpu_accel.CONDITIONED_TEMPLATE_MODE_PER_SOURCE
    session.close(reset_device=False)


@GPU_REQUIRED
def test_exact_pattern_contexts_match_cpu_reference_functions():
    angles_deg = np.array([0.0, 0.1, 1.0, 10.0, 45.0, 89.9], dtype=np.float32)
    wavelength = (2690 * u.MHz).to(u.m, equivalencies=u.spectral())

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        rx_context = session.prepare_ras_pattern_context(
            diameter_m=15 * u.m,
            wavelength_m=wavelength,
        )
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=wavelength,
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0 * cnv.dB,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0 * cnv.dBi,
        )
        ras_gpu = gpu_accel.copy_device_to_host(gpu_accel._evaluate_ras_pattern_cp(rx_context, angles_deg))
        tx_gpu = gpu_accel.copy_device_to_host(gpu_accel._evaluate_s1528_pattern_cp(tx_context, angles_deg))
    ras_cpu = ras_pattern(angles_deg * u.deg, 15 * u.m, wavelength).to_value(cnv.dB)
    tx_cpu = s_1528_rec1_4_pattern_amend(
        angles_deg * u.deg,
        wavelength=wavelength,
        Lt=1.6 * u.m,
        Lr=1.6 * u.m,
        l=2,
        SLR=20.0 * cnv.dB,
        far_sidelobe_start=90 * u.deg,
        far_sidelobe_level=-20.0 * cnv.dBi,
        use_numba=False,
    ).to_value(cnv.dB)
    assert_allclose(ras_gpu, ras_cpu, atol=2e-4, rtol=0.0)
    assert_allclose(tx_gpu, tx_cpu, atol=2e-4, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_s1528_pattern_context_accepts_logarithmic_gain_units():
    wavelength = (2690 * u.MHz).to(u.m, equivalencies=u.spectral())

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        context = session.prepare_s1528_pattern_context(
            wavelength_m=wavelength,
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0 * cnv.dB,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0 * cnv.dBi,
            gm_db=48.0 * u.dB,
        )

    assert context.slr_db == pytest.approx(20.0)
    assert context.far_sidelobe_level_db == pytest.approx(-20.0)
    assert context.gm_db == pytest.approx(48.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_s1528_rec14_lut_and_analytical_match():
    """Rec 1.4: LUT and analytical paths must produce near-identical results."""
    import cupy as cp
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_s1528_pattern_context(
            wavelength_m=0.15, lt_m=1.6, lr_m=1.6, slr_db=20.0, l=2,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0, gm_db=34.1,
        )
        theta = cp.array([0.0, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 90.0, 180.0], dtype=cp.float32)

        gpu_accel.set_pattern_eval_mode("analytical")
        gain_analytical = gpu_accel._evaluate_s1528_pattern_cp(ctx, theta).get()

        gpu_accel.set_pattern_eval_mode("lut")
        gain_lut = gpu_accel._evaluate_s1528_pattern_cp(ctx, theta).get()

        assert_allclose(gain_lut, gain_analytical, atol=0.05, rtol=0)
    gpu_accel.set_pattern_eval_mode("lut")
    session.close(reset_device=False)


@GPU_REQUIRED
def test_s1528_rec12_direct_kernel_matches_cpu():
    """Rec 1.2: GPU piecewise kernel must match the CPU antenna.py reference."""
    import cupy as cp
    from scepter.antenna import s_1528_rec1_2_pattern

    wavelength_m = 0.11145
    diameter_m = 4.0
    gm_dbi = 38.0
    ln_db = -20.0
    z_val = 1.0

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_s1528_rec12_pattern_context(
            wavelength_m=wavelength_m, gm_dbi=gm_dbi, ln_db=ln_db, z=z_val,
            diameter_m=diameter_m,
        )
        theta_np = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0, 90.0, 120.0, 179.0], dtype=np.float32)
        theta_cp = cp.asarray(theta_np)

        # Direct GPU kernel
        gain_gpu = gpu_accel._evaluate_s1528_rec12_pattern_cp(ctx, theta_cp).get()

    # CPU reference (pass D explicitly for consistent psi_b computation)
    cpu_result = s_1528_rec1_2_pattern(
        theta_np * u.deg, axis="major",
        Gm=gm_dbi * cnv.dBi, LN=ln_db * cnv.dB, z=z_val,
        wavelength=wavelength_m * u.m, D=diameter_m * u.m,
    )
    gain_cpu = np.asarray(cpu_result[0] if isinstance(cpu_result, (tuple, list)) else cpu_result, dtype=np.float32)

    assert_allclose(gain_gpu, gain_cpu, atol=0.15, rtol=0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_s1528_rec12_lut_matches_direct_kernel():
    """Rec 1.2: LUT path must match the direct piecewise kernel."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_s1528_rec12_pattern_context(
            wavelength_m=0.11145, gm_dbi=38.0, ln_db=-20.0, z=1.0,
        )
        theta = cp.array([0.0, 0.5, 1.0, 3.0, 10.0, 45.0, 90.0, 135.0, 179.0], dtype=cp.float32)

        # Direct kernel
        gain_direct = gpu_accel._evaluate_s1528_rec12_pattern_cp(ctx, theta).get()

        # Build LUT and evaluate through it
        gpu_accel._ensure_s1528_gain_lut(ctx)
        assert ctx.d_gain_lut is not None
        gain_lut = gpu_accel._evaluate_s1528_pattern_cp_lut(ctx, theta).get()

        assert_allclose(gain_lut, gain_direct, atol=0.05, rtol=0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_s1528_rec12_dispatch_uses_correct_path():
    """The unified _evaluate_s1528_pattern_cp dispatches to Rec 1.2 for Rec12 contexts."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx12 = session.prepare_s1528_rec12_pattern_context(
            wavelength_m=0.11145, gm_dbi=38.0, ln_db=-20.0, z=1.0,
        )
        ctx14 = session.prepare_s1528_pattern_context(
            wavelength_m=0.15, lt_m=1.6, lr_m=1.6, slr_db=20.0, l=2,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0, gm_db=34.1,
        )
        theta = cp.array([0.0, 5.0, 90.0], dtype=cp.float32)

        # Both contexts should work through the unified dispatcher
        gain12 = gpu_accel._evaluate_s1528_pattern_cp(ctx12, theta).get()
        gain14 = gpu_accel._evaluate_s1528_pattern_cp(ctx14, theta).get()

        # Rec 1.2 peak = 38 dBi, Rec 1.4 peak ~ 34.1 dBi
        assert gain12[0] == pytest.approx(38.0, abs=0.1)
        assert gain14[0] == pytest.approx(34.1, abs=0.5)

        # Both should produce finite non-NaN values
        assert np.all(np.isfinite(gain12))
        assert np.all(np.isfinite(gain14))
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
# M.2101 phased array antenna pattern tests
# ---------------------------------------------------------------------------

@GPU_REQUIRED
def test_m2101_peak_gain_matches_pycraf():
    """M.2101: GPU kernel peak gain matches pycraf at boresight."""
    import cupy as cp
    from pycraf.antenna import imt2020_composite_pattern
    from pycraf import conversions as cnv

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_m2101_pattern_context(
            g_emax_db=2.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=120.0, theta_3db_deg=120.0,
            d_h=0.5, d_v=0.5, n_h=28, n_v=28,
        )
        angles = [0, 5, 10, 60, 180]
        gpu_gain = gpu_accel._evaluate_m2101_pattern_cp(
            ctx,
            az_deg=cp.array(angles, dtype=cp.float32),
            el_deg=cp.zeros(len(angles), dtype=cp.float32),
            az_i_deg=cp.zeros(len(angles), dtype=cp.float32),
            el_i_deg=cp.zeros(len(angles), dtype=cp.float32),
        ).get()

    pycraf_gain = imt2020_composite_pattern(
        np.array(angles) * u.deg, np.zeros(len(angles)) * u.deg,
        azim_i=0 * u.deg, elev_i=0 * u.deg,
        G_Emax=2 * cnv.dBi, A_m=30 * cnv.dB, SLA_nu=30 * cnv.dB,
        phi_3db=120 * u.deg, theta_3db=120 * u.deg,
        d_H=0.5 * cnv.dimless, d_V=0.5 * cnv.dimless,
        N_H=28, N_V=28,
    ).value

    # Compare at non-null angles (nulls give -inf in pycraf, finite in GPU)
    for i, a in enumerate(angles):
        if np.isfinite(pycraf_gain[i]):
            assert gpu_gain[i] == pytest.approx(pycraf_gain[i], abs=0.1), \
                f"Mismatch at az={a}°: GPU={gpu_gain[i]:.2f} vs pycraf={pycraf_gain[i]:.2f}"
    # Peak gain should be ~30.9 dBi for 28x28 array with 2 dBi elements
    assert gpu_gain[0] == pytest.approx(30.94, abs=0.5)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_m2101_beam_steering():
    """M.2101: steered beam peaks at steering direction, not boresight."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_m2101_pattern_context(
            g_emax_db=2.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=120.0, theta_3db_deg=120.0,
            d_h=0.5, d_v=0.5, n_h=16, n_v=16,
        )
        # Beam steered to az=20°, el=0°
        steer_az = 20.0
        test_az = cp.array([0, 10, 20, 30, 40], dtype=cp.float32)
        gain = gpu_accel._evaluate_m2101_pattern_cp(
            ctx,
            az_deg=test_az,
            el_deg=cp.zeros(5, dtype=cp.float32),
            az_i_deg=cp.full(5, steer_az, dtype=cp.float32),
            el_i_deg=cp.zeros(5, dtype=cp.float32),
        ).get()
        # Gain should be highest at az=20° (the steering direction)
        peak_idx = int(np.argmax(gain))
        assert peak_idx == 2, f"Peak should be at az=20° (index 2), got index {peak_idx}"
    session.close(reset_device=False)


@GPU_REQUIRED
def test_m2101_2d_pattern_asymmetry():
    """M.2101: pattern is asymmetric when phi_3db != theta_3db."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_m2101_pattern_context(
            g_emax_db=5.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=65.0,   # narrow horizontal
            theta_3db_deg=120.0,  # wide vertical
            d_h=0.5, d_v=0.5, n_h=8, n_v=8,
        )
        # Compare gain at (az=30, el=0) vs (az=0, el=30)
        gain_h = gpu_accel._evaluate_m2101_pattern_cp(
            ctx,
            az_deg=cp.array([30], dtype=cp.float32),
            el_deg=cp.array([0], dtype=cp.float32),
            az_i_deg=cp.array([0], dtype=cp.float32),
            el_i_deg=cp.array([0], dtype=cp.float32),
        ).get()[0]
        gain_v = gpu_accel._evaluate_m2101_pattern_cp(
            ctx,
            az_deg=cp.array([0], dtype=cp.float32),
            el_deg=cp.array([30], dtype=cp.float32),
            az_i_deg=cp.array([0], dtype=cp.float32),
            el_i_deg=cp.array([0], dtype=cp.float32),
        ).get()[0]
        # With narrower horizontal beamwidth, gain should drop faster in az
        assert gain_h < gain_v, \
            f"Pattern should be asymmetric: az=30° gain ({gain_h:.1f}) should be less than el=30° gain ({gain_v:.1f})"
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
# M.2101 2-D LUT tests
# ---------------------------------------------------------------------------

@GPU_REQUIRED
def test_m2101_lut_matches_analytical():
    """M.2101 2-D element LUT matches analytical evaluation within 0.1 dB."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_m2101_pattern_context(
            g_emax_db=5.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=120.0, theta_3db_deg=120.0,
            d_h=0.5, d_v=0.5, n_h=8, n_v=8,
        )
        rng = np.random.default_rng(42)
        n = 500
        az = rng.uniform(-170, 170, n).astype(np.float32)
        el = rng.uniform(-85, 85, n).astype(np.float32)
        steer_az = rng.uniform(-30, 30, n).astype(np.float32)
        steer_el = rng.uniform(-15, 15, n).astype(np.float32)

        analytical = gpu_accel._evaluate_m2101_pattern_cp(
            ctx,
            cp.asarray(az), cp.asarray(el),
            cp.asarray(steer_az), cp.asarray(steer_el),
        ).get()

        lut_result = gpu_accel._evaluate_m2101_pattern_cp_lut(
            ctx,
            cp.asarray(az), cp.asarray(el),
            cp.asarray(steer_az), cp.asarray(steer_el),
        ).get()

    # The array factor is identical (analytical in both paths).
    # The element pattern differs by LUT interpolation error.
    max_err = float(np.max(np.abs(analytical - lut_result)))
    assert max_err < 0.5, f"M.2101 LUT vs analytical max error {max_err:.3f} dB > 0.5 dB"
    session.close(reset_device=False)


@GPU_REQUIRED
def test_m2101_lut_to_linear_matches():
    """M.2101 fused LUT-to-linear matches dB→linear of analytical."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_m2101_pattern_context(
            g_emax_db=5.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=120.0, theta_3db_deg=120.0,
            d_h=0.5, d_v=0.5, n_h=8, n_v=8,
        )
        az = cp.array([0, 10, 30, 60, 120], dtype=cp.float32)
        el = cp.zeros(5, dtype=cp.float32)
        steer_az = cp.zeros(5, dtype=cp.float32)
        steer_el = cp.zeros(5, dtype=cp.float32)

        analytical_db = gpu_accel._evaluate_m2101_pattern_cp(
            ctx, az, el, steer_az, steer_el,
        ).get()
        gmax_lin = 10.0 ** (analytical_db[0] / 10.0)
        expected_rel_lin = 10.0 ** (analytical_db / 10.0) / gmax_lin

        fused_lin = gpu_accel._evaluate_m2101_pattern_cp_lut_to_linear(
            ctx, az, el, steer_az, steer_el,
            inv_gmax_lin=1.0 / gmax_lin,
        ).get()

    np.testing.assert_allclose(fused_lin, expected_rel_lin, rtol=0.02,
                               err_msg="Fused LUT-to-linear does not match analytical dB→linear")
    session.close(reset_device=False)


@GPU_REQUIRED
def test_m2101_lut_preserves_asymmetry():
    """M.2101 LUT preserves the asymmetry between phi_3db and theta_3db."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_m2101_pattern_context(
            g_emax_db=5.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=65.0,    # narrow horizontal
            theta_3db_deg=120.0,  # wide vertical
            d_h=0.5, d_v=0.5, n_h=8, n_v=8,
        )
        # LUT path: gain at (az=30, el=0) vs (az=0, el=30)
        gain_h = gpu_accel._evaluate_m2101_pattern_cp_lut(
            ctx,
            cp.array([30], dtype=cp.float32), cp.array([0], dtype=cp.float32),
            cp.array([0], dtype=cp.float32), cp.array([0], dtype=cp.float32),
        ).get()[0]
        gain_v = gpu_accel._evaluate_m2101_pattern_cp_lut(
            ctx,
            cp.array([0], dtype=cp.float32), cp.array([30], dtype=cp.float32),
            cp.array([0], dtype=cp.float32), cp.array([0], dtype=cp.float32),
        ).get()[0]
        assert gain_h < gain_v, \
            f"LUT should preserve asymmetry: az=30° ({gain_h:.1f}) < el=30° ({gain_v:.1f})"
    session.close(reset_device=False)


@GPU_REQUIRED
def test_m2101_lut_memory_size():
    """M.2101 2-D LUT is within expected memory budget."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_m2101_pattern_context(
            g_emax_db=5.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=120.0, theta_3db_deg=120.0,
            d_h=0.5, d_v=0.5, n_h=28, n_v=28,
        )
        gpu_accel._ensure_m2101_element_lut_2d(ctx)
        lut = ctx.d_element_lut
        lut_bytes = lut.nbytes
        # 0.5° step: (720+1) × (360+1) × 4 bytes ≈ 1.04 MB
        assert lut_bytes < 2 * 1024 * 1024, f"LUT too large: {lut_bytes / 1024:.0f} KB"
        assert ctx.element_lut_n_az == 721
        assert ctx.element_lut_n_el == 361
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
# Beamforming collapsed mode tests
# ---------------------------------------------------------------------------

@GPU_REQUIRED
def test_beamforming_collapsed_produces_finite_epfd():
    """Collapsed mode: produces finite EPFD for visible satellites, zero for below-horizon."""
    import cupy as cp

    sat_topo = cp.array([
        [[0.0, 60.0, 700.0], [90.0, 30.0, 1200.0], [180.0, -5.0, 3000.0]],
    ], dtype=cp.float32)
    orbit_r = cp.full(3, 6903000.0, dtype=cp.float32)

    result = gpu_accel._accumulate_beamforming_collapsed_power_cp(
        sat_topo=sat_topo,
        orbit_radius_m_per_sat=orbit_r,
        observer_alt_km=1.0,
        wavelength_m=0.11145,
        atmosphere_lut_context=None,
        ras_pattern_context=None,
        telescope_azimuth_deg=None,
        telescope_elevation_deg=None,
        collapsed_baseline_eirp_dbw_hz=-55.6,
        collapsed_eval_freq_mhz=2695.0,
        collapsed_ref_freq_mhz=2000.0,
        ras_bandwidth_hz=10e6,
    )
    epfd = result["EPFD_W_m2"].get().flatten()
    assert np.all(np.isfinite(epfd))
    assert epfd[0] > 0, "EPFD should be positive (visible satellites contribute)"


@GPU_REQUIRED
def test_beamforming_collapsed_power_decreases_with_range():
    """Collapsed mode: higher elevation (shorter range) gives more power than lower elevation."""
    import cupy as cp

    # Two timesteps: one with high-elevation sats, one with low-elevation sats
    sat_topo_high = cp.array([
        [[0.0, 80.0, 600.0], [90.0, 70.0, 650.0]],
    ], dtype=cp.float32)
    sat_topo_low = cp.array([
        [[0.0, 20.0, 1800.0], [90.0, 15.0, 2200.0]],
    ], dtype=cp.float32)
    orbit_r = cp.full(2, 6903000.0, dtype=cp.float32)
    common = dict(
        orbit_radius_m_per_sat=orbit_r, observer_alt_km=1.0,
        wavelength_m=0.11145, atmosphere_lut_context=None,
        ras_pattern_context=None, telescope_azimuth_deg=None,
        telescope_elevation_deg=None,
        collapsed_baseline_eirp_dbw_hz=-55.6,
        collapsed_eval_freq_mhz=2695.0, collapsed_ref_freq_mhz=2000.0,
        ras_bandwidth_hz=10e6,
    )
    epfd_high = gpu_accel._accumulate_beamforming_collapsed_power_cp(
        sat_topo=sat_topo_high, **common,
    )["EPFD_W_m2"].get().flatten()[0]
    epfd_low = gpu_accel._accumulate_beamforming_collapsed_power_cp(
        sat_topo=sat_topo_low, **common,
    )["EPFD_W_m2"].get().flatten()[0]
    assert epfd_high > epfd_low, \
        f"High-elevation EPFD ({epfd_high:.2e}) should exceed low-elevation ({epfd_low:.2e})"


@GPU_REQUIRED
def test_beamforming_collapsed_frequency_term():
    """Collapsed mode: different evaluation frequencies produce different EPFD levels."""
    import cupy as cp

    sat_topo = cp.array([[[0.0, 45.0, 900.0]]], dtype=cp.float32)
    orbit_r = cp.full(1, 6903000.0, dtype=cp.float32)
    common = dict(
        sat_topo=sat_topo, orbit_radius_m_per_sat=orbit_r,
        observer_alt_km=1.0, wavelength_m=0.11145,
        atmosphere_lut_context=None, ras_pattern_context=None,
        telescope_azimuth_deg=None, telescope_elevation_deg=None,
        collapsed_baseline_eirp_dbw_hz=-55.6,
        collapsed_ref_freq_mhz=2000.0, ras_bandwidth_hz=10e6,
    )
    epfd_2695 = gpu_accel._accumulate_beamforming_collapsed_power_cp(
        collapsed_eval_freq_mhz=2695.0, **common,
    )["EPFD_W_m2"].get().flatten()[0]
    epfd_2500 = gpu_accel._accumulate_beamforming_collapsed_power_cp(
        collapsed_eval_freq_mhz=2500.0, **common,
    )["EPFD_W_m2"].get().flatten()[0]
    # Higher frequency → more OOBE power (20*log10(f/f_ref) is larger)
    assert epfd_2695 > epfd_2500, \
        f"EPFD at 2695 MHz ({epfd_2695:.2e}) should exceed EPFD at 2500 MHz ({epfd_2500:.2e})"


# ---------------------------------------------------------------------------
# Isotropic UEMR mode tests
# ---------------------------------------------------------------------------


@GPU_REQUIRED
def test_accumulate_uemr_power_matches_analytic_free_space():
    """UEMR: Prx per sat = Ptx · (λ/4π·d)² and EPFD = Prx·(4π/λ²). No beams."""
    import cupy as cp
    import math
    session = gpu_accel.GpuScepterSession()
    try:
        wl = 0.1113
        ctx = session.prepare_isotropic_pattern_context(wavelength_m=wl)
        # 2 timesteps × 3 sats, known geometry.
        topo = np.array([
            [[0.0, 90.0, 550.0], [0.0, 30.0, 800.0], [0.0, -5.0, 2000.0]],
            [[0.0, 60.0, 700.0], [0.0, 45.0, 900.0], [0.0, 0.0, 1500.0]],
        ], dtype=np.float32)
        out = gpu_accel._accumulate_uemr_power_cp(
            uemr_pattern_context=ctx,
            ras_pattern_context=None,
            atmosphere_lut_context=None,
            sat_topo=cp.asarray(topo),
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            observer_alt_km=0.0,
            bandwidth_mhz=10.0,
            power_input_quantity="satellite_ptx",
            target_pfd_dbw_m2_channel=None,
            satellite_ptx_dbw_channel=0.0,  # 1 W channel
            satellite_eirp_dbw_channel=None,
            include_epfd=True,
            include_prx_total=True,
            include_total_pfd=True,
        )
        # Analytic Prx at t=0: two visible sats (90°, 30°), below-horizon skipped.
        prx_t0 = (wl / (4 * math.pi * 550e3)) ** 2 + (wl / (4 * math.pi * 800e3)) ** 2
        prx_out = out["Prx_total_W"].get().squeeze()
        assert np.isclose(prx_out[0], prx_t0, rtol=1e-4), (prx_out[0], prx_t0)
        # EPFD = Prx * 4π/λ²
        epfd_expected = prx_out * (4.0 * math.pi / wl ** 2)
        epfd_out = out["EPFD_W_m2"].get().squeeze()
        assert np.allclose(epfd_out, epfd_expected, rtol=1e-5)
        # t=1 sat 2 is at el=0 (strictly > 0 required) → skip; only two visible.
        prx_t1 = (wl / (4 * math.pi * 700e3)) ** 2 + (wl / (4 * math.pi * 900e3)) ** 2
        assert np.isclose(prx_out[1], prx_t1, rtol=1e-4)
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_uemr_power_slab_fraction_scales_linearly():
    """UEMR: ``eirp_slab_fraction_lin`` multiplies all outputs linearly."""
    import cupy as cp
    session = gpu_accel.GpuScepterSession()
    try:
        ctx = session.prepare_isotropic_pattern_context(wavelength_m=0.1113)
        topo = cp.asarray(np.array([
            [[0.0, 80.0, 600.0], [0.0, 45.0, 900.0]],
        ], dtype=np.float32))
        kw = dict(
            uemr_pattern_context=ctx, ras_pattern_context=None,
            atmosphere_lut_context=None, sat_topo=topo,
            telescope_azimuth_deg=None, telescope_elevation_deg=None,
            observer_alt_km=0.0, bandwidth_mhz=10.0,
            power_input_quantity="satellite_ptx",
            target_pfd_dbw_m2_channel=None,
            satellite_ptx_dbw_channel=0.0,
            satellite_eirp_dbw_channel=None,
            include_epfd=True, include_prx_total=True,
        )
        full = gpu_accel._accumulate_uemr_power_cp(eirp_slab_fraction_lin=1.0, **kw)
        half = gpu_accel._accumulate_uemr_power_cp(eirp_slab_fraction_lin=0.5, **kw)
        assert np.allclose(
            half["Prx_total_W"].get(), 0.5 * full["Prx_total_W"].get(), rtol=1e-5,
        )
        assert np.allclose(
            half["EPFD_W_m2"].get(), 0.5 * full["EPFD_W_m2"].get(), rtol=1e-5,
        )
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_uemr_power_ptx_eirp_match_when_gtx_is_zero():
    """UEMR: Gtx=0 dBi ⇒ EIRP(dBW) == Ptx(dBW). Both inputs give the same result."""
    import cupy as cp
    session = gpu_accel.GpuScepterSession()
    try:
        ctx = session.prepare_isotropic_pattern_context(wavelength_m=0.1113)
        topo = cp.asarray(np.array([
            [[0.0, 80.0, 600.0], [0.0, 45.0, 900.0]],
        ], dtype=np.float32))
        base = dict(
            uemr_pattern_context=ctx, ras_pattern_context=None,
            atmosphere_lut_context=None, sat_topo=topo,
            telescope_azimuth_deg=None, telescope_elevation_deg=None,
            observer_alt_km=0.0, bandwidth_mhz=10.0,
            include_epfd=True, include_prx_total=True,
        )
        ptx = gpu_accel._accumulate_uemr_power_cp(
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=3.0,
            target_pfd_dbw_m2_channel=None,
            satellite_eirp_dbw_channel=None, **base,
        )
        eirp = gpu_accel._accumulate_uemr_power_cp(
            power_input_quantity="satellite_eirp",
            satellite_eirp_dbw_channel=3.0,
            target_pfd_dbw_m2_channel=None,
            satellite_ptx_dbw_channel=None, **base,
        )
        assert np.allclose(ptx["Prx_total_W"].get(), eirp["Prx_total_W"].get(), rtol=1e-6)
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_uemr_power_rejects_non_isotropic_context():
    """UEMR kernel must reject S.1528 / M.2101 contexts."""
    import cupy as cp
    session = gpu_accel.GpuScepterSession()
    try:
        bad = session.prepare_s1528_rec12_pattern_context(
            wavelength_m=0.1113, gm_dbi=30.0,
        )
        import pytest as _pt
        with _pt.raises(TypeError, match="GpuIsotropicPatternContext"):
            gpu_accel._accumulate_uemr_power_cp(
                uemr_pattern_context=bad, ras_pattern_context=None,
                atmosphere_lut_context=None,
                sat_topo=cp.zeros((1, 1, 3), dtype=cp.float32),
                telescope_azimuth_deg=None, telescope_elevation_deg=None,
                observer_alt_km=0.0, bandwidth_mhz=10.0,
                power_input_quantity="satellite_ptx",
                target_pfd_dbw_m2_channel=None,
                satellite_ptx_dbw_channel=0.0,
                satellite_eirp_dbw_channel=None,
            )
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_isotropic_pattern_peak_gain_linear_is_one():
    """Isotropic ``_pattern_peak_gain_linear`` must return exactly 1.0."""
    session = gpu_accel.GpuScepterSession()
    try:
        ctx = session.prepare_isotropic_pattern_context(wavelength_m=0.1113)
        assert gpu_accel._pattern_peak_gain_linear(ctx) == 1.0
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_isotropic_evaluator_dispatches_through_s1528_wrapper():
    """``_evaluate_s1528_pattern_cp`` must dispatch isotropic context to zero-dB output."""
    import cupy as cp
    session = gpu_accel.GpuScepterSession()
    try:
        ctx = session.prepare_isotropic_pattern_context(wavelength_m=0.1113)
        theta = cp.asarray([0.0, 5.0, 30.0, 60.0, 90.0, 180.0], dtype=cp.float32)
        out = gpu_accel._evaluate_s1528_pattern_cp(ctx, theta)
        assert out.shape == theta.shape
        assert cp.allclose(out, cp.zeros_like(theta)).get()
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_prepare_isotropic_pattern_context_returns_expected_type():
    """``session.prepare_isotropic_pattern_context`` returns GpuIsotropicPatternContext."""
    session = gpu_accel.GpuScepterSession()
    try:
        ctx = session.prepare_isotropic_pattern_context(wavelength_m=0.123)
        assert isinstance(ctx, gpu_accel.GpuIsotropicPatternContext)
        assert float(ctx.wavelength_m) == pytest.approx(0.123)
    finally:
        session.close(reset_device=False)


def test_flat_mask_preset_returns_zero_db_across_service_band():
    """``flat`` mask preset must produce 0 dB attenuation for every offset."""
    from scepter import scenario as sc
    import numpy as _np
    points = sc._resolve_direct_epfd_mask_points_mhz(
        preset="flat", channel_bandwidth_mhz=5.0, custom_mask_points=None,
    )
    # Shape (n, 2): [offset_mhz, attenuation_db]
    assert points.shape[1] == 2
    assert _np.allclose(points[:, 1], 0.0), f"flat preset not zero-dB: {points}"
    # Breakpoints span a wide range of offsets (tight to far), so any
    # piecewise-linear interpolator / lookup will return 0 dB across the
    # service band.
    offsets = points[:, 0]
    assert float(offsets.min()) <= -10.0 and float(offsets.max()) >= 10.0, (
        f"flat preset breakpoints don't span ±10 MHz: {offsets}"
    )


@GPU_REQUIRED
def test_accumulate_uemr_power_respects_radio_horizon_threshold():
    """Radio horizon extends visibility below 0°; UEMR kernel must honour it."""
    import cupy as cp
    import math
    session = gpu_accel.GpuScepterSession()
    try:
        ctx = session.prepare_isotropic_pattern_context(wavelength_m=0.1113)
        topo = cp.asarray(np.array([[[0.0, -0.3, 2500.0]]], dtype=np.float32))
        kw = dict(
            uemr_pattern_context=ctx,
            ras_pattern_context=None,
            atmosphere_lut_context=None,
            sat_topo=topo,
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            observer_alt_km=0.0,
            bandwidth_mhz=5.0,
            power_input_quantity="satellite_ptx",
            target_pfd_dbw_m2_channel=None,
            satellite_ptx_dbw_channel=0.0,
            satellite_eirp_dbw_channel=None,
            include_epfd=False,
            include_prx_total=True,
        )
        out_strict = gpu_accel._accumulate_uemr_power_cp(
            visibility_elev_threshold_deg=0.0, **kw,
        )
        assert float(out_strict["Prx_total_W"].get().squeeze()) == 0.0
        out_radio = gpu_accel._accumulate_uemr_power_cp(
            visibility_elev_threshold_deg=-0.6, **kw,
        )
        prx_radio = float(out_radio["Prx_total_W"].get().squeeze())
        expected = (0.1113 / (4.0 * math.pi * 2.5e6)) ** 2
        assert prx_radio == pytest.approx(expected, rel=1e-4)
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_peak_pfd_cap_lut_accepts_directive_isotropic_and_is_flat():
    """Directive isotropic: cap LUT builds successfully and K(β) is β-independent."""
    session = gpu_accel.GpuScepterSession()
    try:
        ctx = session.prepare_isotropic_pattern_context(wavelength_m=0.1113)
        lut = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.asarray([7000e3]),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut.is_2d is False
        assert lut.n_beta > 0
        k = lut.d_k_lut.get().reshape(-1)
        # All K values should be identical (flat pattern → K independent of β).
        assert np.allclose(k, k[0], rtol=1e-6), f"K varies: min={k.min()} max={k.max()}"
    finally:
        session.close(reset_device=False)


# ---------------------------------------------------------------------------
# Variable power (slant-range and uniform-random) tests
# ---------------------------------------------------------------------------

@GPU_REQUIRED
def test_slant_range_power_kernel_produces_correct_range():
    """Slant-range kernel: max power at overhead, min at horizon, monotonic."""
    import cupy as cp

    kernel = gpu_accel._get_slant_range_power_kernel()
    orbit_r = np.float32(6_903_000.0)  # 525 km altitude
    earth_r = np.float32(6_378_000.0)
    pwr_min_db = np.float32(26.0)
    pwr_max_db = np.float32(52.0)

    # Range values from directly overhead (altitude) to near-horizon
    altitude_m = orbit_r - earth_r  # 525 km
    import math
    r_max = math.sqrt(float(orbit_r)**2 - float(earth_r)**2) - 0  # el_min=0
    ranges = cp.array([
        float(altitude_m),       # overhead
        float(altitude_m) * 1.5,
        float(altitude_m) * 2.0,
        float(r_max) * 0.5,
        float(r_max) * 0.9,
        float(r_max),           # horizon
    ], dtype=cp.float32)

    result = kernel(
        ranges,
        cp.float32(orbit_r), cp.float32(earth_r),
        cp.float32(0.0), cp.float32(1.0),  # sin(0), cos(0)
        cp.float32(pwr_min_db), cp.float32(pwr_max_db),
    )
    pwr = result.get()

    # All values should be finite and positive
    assert np.all(np.isfinite(pwr))
    assert np.all(pwr > 0.0)

    # Convert to dB for checking
    pwr_db = 10.0 * np.log10(pwr.astype(np.float64))

    # First value (overhead) should be close to max power
    assert pwr_db[0] == pytest.approx(52.0, abs=0.5)
    # Last value (horizon) should be close to min power
    assert pwr_db[-1] == pytest.approx(26.0, abs=1.0)
    # Should be monotonically decreasing (power drops with range)
    for i in range(len(pwr_db) - 1):
        assert pwr_db[i] >= pwr_db[i + 1] - 0.1, \
            f"Power not monotonic at index {i}: {pwr_db[i]:.2f} < {pwr_db[i+1]:.2f}"


@GPU_REQUIRED
def test_accumulate_ras_power_fixed_mode_unchanged():
    """Fixed power variation mode produces identical results to scalar path."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx14 = session.prepare_s1528_pattern_context(
            wavelength_m=0.15, lt_m=1.6, lr_m=1.6, slr_db=20.0, l=2,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0, gm_db=34.1,
        )
        # Minimal inputs for a small test
        T, S = 3, 5
        sat_topo = cp.random.uniform(0.5, 2.0, (T, S, 3), dtype=cp.float32)
        sat_topo[:, :, 2] = cp.abs(sat_topo[:, :, 2]) * 500 + 500  # range in km
        sat_azel = cp.random.uniform(-180, 180, (T, S, 2), dtype=cp.float32)
        beam_idx = cp.full((T, S, 1), -1, dtype=cp.int32)
        beam_idx[:, 0, 0] = 0  # one active beam
        beam_alpha = cp.zeros((T, S, 1), dtype=cp.float32)
        beam_beta = cp.zeros((T, S, 1), dtype=cp.float32)
        orbit_r = cp.full((S,), 6_903_000.0, dtype=cp.float32)

        common_kwargs = dict(
            s1528_pattern_context=ctx14,
            ras_pattern_context=None,
            atmosphere_lut_context=None,
            spectrum_plan_context=None,
            cell_spectral_weight=None,
            sat_topo=sat_topo, sat_azel=sat_azel,
            beam_idx=beam_idx, beam_alpha_rad=beam_alpha, beam_beta_rad=beam_beta,
            telescope_azimuth_deg=None, telescope_elevation_deg=None,
            orbit_radius_m_per_sat=orbit_r,
            observer_alt_km=1.0,
            bandwidth_mhz=5.0,
            power_input_quantity="satellite_eirp",
            target_pfd_dbw_m2_channel=None,
            satellite_ptx_dbw_channel=None,
            satellite_eirp_dbw_channel=52.0,
            n_links=1, ras_service_cell_index=None,
            target_alt_km=0.0, use_ras_station_alt_for_co=True,
            include_epfd=False, include_prx_total=False,
            include_per_satellite_prx=False,
            include_total_pfd=True, include_per_satellite_pfd=False,
        )

        # Fixed mode (default)
        r_fixed = gpu_accel._accumulate_ras_power_cp(
            **common_kwargs,
            power_variation_mode="fixed",
            power_range_min_dbw_channel=None,
            power_range_max_dbw_channel=None,
        )

        # Fixed mode explicit (should be identical)
        r_fixed2 = gpu_accel._accumulate_ras_power_cp(
            **common_kwargs,
            power_variation_mode="fixed",
            power_range_min_dbw_channel=26.0,
            power_range_max_dbw_channel=52.0,
        )

        pfd_fixed = r_fixed["PFD_total_RAS_STATION_W_m2"].get()
        pfd_fixed2 = r_fixed2["PFD_total_RAS_STATION_W_m2"].get()
        assert_allclose(pfd_fixed, pfd_fixed2, atol=0, rtol=0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_ras_power_slant_range_differs_from_fixed():
    """Slant-range mode produces different (generally lower) results than fixed max."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx14 = session.prepare_s1528_pattern_context(
            wavelength_m=0.15, lt_m=1.6, lr_m=1.6, slr_db=20.0, l=2,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0, gm_db=34.1,
        )
        T, S = 3, 5
        sat_topo = cp.random.uniform(0.5, 2.0, (T, S, 3), dtype=cp.float32)
        sat_topo[:, :, 1] = cp.abs(sat_topo[:, :, 1]) * 30 + 10  # elevation > 0
        sat_topo[:, :, 2] = cp.abs(sat_topo[:, :, 2]) * 500 + 600  # range in km
        sat_azel = cp.random.uniform(-180, 180, (T, S, 2), dtype=cp.float32)
        beam_idx = cp.full((T, S, 1), -1, dtype=cp.int32)
        beam_idx[:, :, 0] = 0  # all active
        beam_alpha = cp.zeros((T, S, 1), dtype=cp.float32)
        beam_beta = cp.zeros((T, S, 1), dtype=cp.float32)
        orbit_r = cp.full((S,), 6_903_000.0, dtype=cp.float32)

        common_kwargs = dict(
            s1528_pattern_context=ctx14,
            ras_pattern_context=None,
            atmosphere_lut_context=None,
            spectrum_plan_context=None,
            cell_spectral_weight=None,
            sat_topo=sat_topo, sat_azel=sat_azel,
            beam_idx=beam_idx, beam_alpha_rad=beam_alpha, beam_beta_rad=beam_beta,
            telescope_azimuth_deg=None, telescope_elevation_deg=None,
            orbit_radius_m_per_sat=orbit_r,
            observer_alt_km=1.0,
            bandwidth_mhz=5.0,
            power_input_quantity="satellite_eirp",
            target_pfd_dbw_m2_channel=None,
            satellite_ptx_dbw_channel=None,
            satellite_eirp_dbw_channel=52.0,
            n_links=1, ras_service_cell_index=None,
            target_alt_km=0.0, use_ras_station_alt_for_co=True,
            include_epfd=False, include_prx_total=False,
            include_per_satellite_prx=False,
            include_total_pfd=True, include_per_satellite_pfd=False,
        )

        r_fixed = gpu_accel._accumulate_ras_power_cp(
            **common_kwargs,
            power_variation_mode="fixed",
        )
        r_slant = gpu_accel._accumulate_ras_power_cp(
            **common_kwargs,
            power_variation_mode="slant_range",
            power_range_min_dbw_channel=26.0,
            power_range_max_dbw_channel=52.0,
        )

        pfd_fixed = r_fixed["PFD_total_RAS_STATION_W_m2"].get()
        pfd_slant = r_slant["PFD_total_RAS_STATION_W_m2"].get()

        # Slant-range should generally produce lower total PFD since
        # some satellites are at lower power (further away)
        fixed_total = float(np.sum(pfd_fixed[np.isfinite(pfd_fixed)]))
        slant_total = float(np.sum(pfd_slant[np.isfinite(pfd_slant)]))
        if fixed_total > 0:
            assert slant_total <= fixed_total * 1.01, \
                f"Slant-range PFD ({slant_total:.6e}) should not exceed fixed ({fixed_total:.6e})"
    session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_ras_power_uniform_random_in_range():
    """Uniform random mode produces values that differ across satellites."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx14 = session.prepare_s1528_pattern_context(
            wavelength_m=0.15, lt_m=1.6, lr_m=1.6, slr_db=20.0, l=2,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0, gm_db=34.1,
        )
        T, S = 3, 10
        sat_topo = cp.random.uniform(0.5, 2.0, (T, S, 3), dtype=cp.float32)
        sat_topo[:, :, 1] = cp.abs(sat_topo[:, :, 1]) * 30 + 10
        sat_topo[:, :, 2] = cp.abs(sat_topo[:, :, 2]) * 500 + 600
        sat_azel = cp.random.uniform(-180, 180, (T, S, 2), dtype=cp.float32)
        beam_idx = cp.full((T, S, 1), -1, dtype=cp.int32)
        beam_idx[:, :, 0] = 0
        beam_alpha = cp.zeros((T, S, 1), dtype=cp.float32)
        beam_beta = cp.zeros((T, S, 1), dtype=cp.float32)
        orbit_r = cp.full((S,), 6_903_000.0, dtype=cp.float32)

        r_rand = gpu_accel._accumulate_ras_power_cp(
            s1528_pattern_context=ctx14,
            ras_pattern_context=None,
            atmosphere_lut_context=None,
            spectrum_plan_context=None,
            cell_spectral_weight=None,
            sat_topo=sat_topo, sat_azel=sat_azel,
            beam_idx=beam_idx, beam_alpha_rad=beam_alpha, beam_beta_rad=beam_beta,
            telescope_azimuth_deg=None, telescope_elevation_deg=None,
            orbit_radius_m_per_sat=orbit_r,
            observer_alt_km=1.0,
            bandwidth_mhz=5.0,
            power_input_quantity="satellite_eirp",
            target_pfd_dbw_m2_channel=None,
            satellite_ptx_dbw_channel=None,
            satellite_eirp_dbw_channel=52.0,
            power_variation_mode="uniform_random",
            power_range_min_dbw_channel=26.0,
            power_range_max_dbw_channel=52.0,
            n_links=1, ras_service_cell_index=None,
            target_alt_km=0.0, use_ras_station_alt_for_co=True,
            include_epfd=False, include_prx_total=False,
            include_per_satellite_prx=False,
            include_total_pfd=True, include_per_satellite_pfd=True,
        )

        pfd = r_rand["PFD_total_RAS_STATION_W_m2"].get()
        assert np.all(np.isfinite(pfd) | (pfd == 0))

        # Per-satellite PFD should show variation (not all identical)
        per_sat = r_rand["PFD_per_sat_RAS_STATION_W_m2"].get()
        positive = per_sat[per_sat > 0]
        if len(positive) > 2:
            # Should have some variation, not all identical
            assert np.std(positive) > 0, "Random mode should produce varied per-sat power"
    session.close(reset_device=False)


@GPU_REQUIRED
def test_normalize_power_input_variation_fields():
    """Scenario normalization passes through variation fields correctly."""
    power_input = scenario.normalize_direct_epfd_power_input(
        bandwidth_mhz=5.0,
        power_input_quantity="satellite_eirp",
        power_input_basis="per_channel",
        satellite_eirp_dbw_channel=52.0,
        power_variation_mode="slant_range",
        power_range_min_db=26.0,
        power_range_max_db=52.0,
    )
    assert power_input["power_variation_mode"] == "slant_range"
    assert power_input["power_range_min_dbw_channel"] == pytest.approx(26.0)
    assert power_input["power_range_max_dbw_channel"] == pytest.approx(52.0)

    # target_pfd should force fixed mode
    power_pfd = scenario.normalize_direct_epfd_power_input(
        bandwidth_mhz=5.0,
        power_input_quantity="target_pfd",
        power_input_basis="per_mhz",
        target_pfd_dbw_m2_mhz=-83.5,
        power_variation_mode="slant_range",
        power_range_min_db=-90.0,
        power_range_max_db=-80.0,
    )
    assert power_pfd["power_variation_mode"] == "fixed"

    # per_mhz basis should convert min/max to per_channel
    import math
    power_mhz = scenario.normalize_direct_epfd_power_input(
        bandwidth_mhz=5.0,
        power_input_quantity="satellite_eirp",
        power_input_basis="per_mhz",
        satellite_eirp_dbw_mhz=45.0,
        power_variation_mode="uniform_random",
        power_range_min_db=20.0,
        power_range_max_db=45.0,
    )
    offset = 10.0 * math.log10(5.0)
    assert power_mhz["power_range_min_dbw_channel"] == pytest.approx(20.0 + offset, abs=0.01)
    assert power_mhz["power_range_max_dbw_channel"] == pytest.approx(45.0 + offset, abs=0.01)


# ---------------------------------------------------------------------------
# Comprehensive feature tests: configs, masks, helpers, GUI integration
# ---------------------------------------------------------------------------

def test_keplerian_positions_altitude_and_count():
    """Keplerian position helper produces correct satellite count and altitude."""
    from scepter.scepter_GUI import _keplerian_positions_eci_km, BeltConfig
    belts = [
        BeltConfig(belt_name="B1", num_sats_per_plane=10, plane_count=5,
                   altitude_km=600.0, eccentricity=0.0, inclination_deg=53.0,
                   argp_deg=0.0, raan_min_deg=0.0, raan_max_deg=360.0,
                   min_elevation_deg=20.0, adjacent_plane_offset=False),
        BeltConfig(belt_name="B2", num_sats_per_plane=3, plane_count=2,
                   altitude_km=800.0, eccentricity=0.0, inclination_deg=97.0,
                   argp_deg=0.0, raan_min_deg=0.0, raan_max_deg=360.0,
                   min_elevation_deg=20.0, adjacent_plane_offset=True),
    ]
    positions = _keplerian_positions_eci_km(belts)
    assert len(positions) == 2
    assert positions[0].shape == (50, 3), f"Belt 1: expected (50,3), got {positions[0].shape}"
    assert positions[1].shape == (6, 3), f"Belt 2: expected (6,3), got {positions[1].shape}"
    # Altitude check (circular orbit → all at same radius)
    r1 = np.sqrt(np.sum(positions[0] ** 2, axis=1))
    assert_allclose(r1, 6378.137 + 600.0, atol=0.1)
    r2 = np.sqrt(np.sum(positions[1] ** 2, axis=1))
    assert_allclose(r2, 6378.137 + 800.0, atol=0.1)


def test_keplerian_positions_adjacent_offset():
    """Adjacent plane offset shifts every other plane by half a slot."""
    from scepter.scepter_GUI import _keplerian_positions_eci_km, BeltConfig
    belt_no_offset = BeltConfig(
        belt_name="A", num_sats_per_plane=4, plane_count=2,
        altitude_km=500.0, eccentricity=0.0, inclination_deg=0.0,
        argp_deg=0.0, raan_min_deg=0.0, raan_max_deg=360.0,
        min_elevation_deg=0.0, adjacent_plane_offset=False,
    )
    belt_with_offset = BeltConfig(
        belt_name="A", num_sats_per_plane=4, plane_count=2,
        altitude_km=500.0, eccentricity=0.0, inclination_deg=0.0,
        argp_deg=0.0, raan_min_deg=0.0, raan_max_deg=360.0,
        min_elevation_deg=0.0, adjacent_plane_offset=True,
    )
    pos_no = _keplerian_positions_eci_km([belt_no_offset])[0]
    pos_yes = _keplerian_positions_eci_km([belt_with_offset])[0]
    # Both have 8 sats, but positions should differ for plane 2
    assert pos_no.shape == pos_yes.shape == (8, 3)
    # Plane 1 (sats 0-3) should be identical
    assert_allclose(pos_no[:4], pos_yes[:4], atol=1e-3)
    # Plane 2 (sats 4-7) should differ due to offset
    assert not np.allclose(pos_no[4:], pos_yes[4:], atol=1e-3)


def test_wrc27_oobe_mask_preset_breakpoints():
    """WRC-27 System 1 DC-MSS-IMT OOBE mask has correct breakpoints for 5 MHz BW."""
    resolved = scenario._resolve_direct_epfd_mask_points_mhz(
        preset="wrc27_1_13_s1_dc_mss_imt",
        channel_bandwidth_mhz=5.0,
        custom_mask_points=None,
    )
    pts = {round(float(row[0]), 1): round(float(row[1]), 1) for row in resolved}
    # Verify symmetric and correct breakpoints
    assert pts[2.5] == 0.0, "Channel edge should be 0 dBc"
    assert pts[-2.5] == 0.0
    assert pts[7.5] == 38.0, "First step should be 38 dBc"
    assert pts[12.5] == 45.0, "Second step should be 45 dBc"
    assert pts[30.0] == 52.0
    assert pts[45.0] == 60.0, "Floor should be 60 dBc"


def test_config_roundtrip_m2101():
    """M.2101 antenna config survives JSON round-trip."""
    from scepter.scepter_GUI import AntennaM2101Config
    cfg = AntennaM2101Config(
        g_emax_dbi=5.0, a_m_db=25.0, sla_nu_db=25.0,
        phi_3db_deg=65.0, theta_3db_deg=65.0,
        d_h=0.5, d_v=0.5, n_h=16, n_v=16,
    )
    d = cfg.to_json_dict()
    cfg2 = AntennaM2101Config.from_json_dict(d)
    assert cfg2.g_emax_dbi == 5.0
    assert cfg2.n_h == 16
    assert cfg2.phi_3db_deg == 65.0


def test_config_roundtrip_power_variation():
    """Power variation fields survive JSON round-trip."""
    from scepter.scepter_GUI import ServiceConfig
    cfg = ServiceConfig(
        nco=1, nbeam=30, selection_strategy="max_elevation",
        cell_activity_factor=1.0, cell_activity_mode="whole_cell",
        cell_activity_seed_base=42, bandwidth_mhz=5.0,
        power_input_quantity="satellite_eirp", power_input_basis="per_channel",
        target_pfd_dbw_m2_mhz=None, target_pfd_dbw_m2_channel=None,
        satellite_ptx_dbw_mhz=None, satellite_ptx_dbw_channel=None,
        satellite_eirp_dbw_mhz=None, satellite_eirp_dbw_channel=52.0,
        power_variation_mode="slant_range",
        power_range_min_db=26.0, power_range_max_db=52.0,
    )
    d = cfg.to_json_dict()
    cfg2 = ServiceConfig.from_json_dict(d)
    assert cfg2.power_variation_mode == "slant_range"
    assert cfg2.power_range_min_db == 26.0
    assert cfg2.power_range_max_db == 52.0


def test_config_roundtrip_beamforming_collapsed():
    """Beamforming collapsed fields survive JSON round-trip."""
    from scepter.scepter_GUI import RuntimeConfig, _default_runtime_config
    defaults = _default_runtime_config()
    d = defaults.to_json_dict()
    d["beamforming_collapsed"] = True
    d["collapsed_baseline_eirp_dbw_hz"] = -60.0
    d["collapsed_eval_freq_mhz"] = 2700.0
    d["collapsed_ref_freq_mhz"] = 2000.0
    loaded = RuntimeConfig.from_json_dict(d)
    assert loaded.beamforming_collapsed is True
    assert loaded.collapsed_baseline_eirp_dbw_hz == -60.0
    assert loaded.collapsed_eval_freq_mhz == 2700.0


def test_config_roundtrip_rec12_diameter_efficiency():
    """Rec 1.2 diameter and efficiency survive JSON round-trip."""
    from scepter.scepter_GUI import AntennaRec12Config
    cfg = AntennaRec12Config(gm_dbi=38.0, diameter_m=4.0, efficiency_pct=90.0, ln_db=-20.0, z=1.0)
    d = cfg.to_json_dict()
    cfg2 = AntennaRec12Config.from_json_dict(d)
    assert cfg2.diameter_m == 4.0
    assert cfg2.efficiency_pct == 90.0
    assert cfg2.gm_dbi == 38.0


def test_config_roundtrip_ras_grx_max():
    """RAS G_rx,max survives JSON round-trip."""
    from scepter.scepter_GUI import RasAntennaConfig
    cfg = RasAntennaConfig(antenna_diameter_m=15.0, grx_max_dbi=52.5,
                           operational_elevation_min_deg=15.0,
                           operational_elevation_max_deg=90.0)
    d = cfg.to_json_dict()
    cfg2 = RasAntennaConfig.from_json_dict(d)
    assert cfg2.grx_max_dbi == 52.5


def test_all_mask_presets_produce_valid_points():
    """Every mask preset generates valid symmetric mask points."""
    for preset in ("sm1541_fss", "sm1541_mss", "3gpp_ts_36_104",
                    "wrc27_1_13_s1_dc_mss_imt", "adjacent_45_nonadjacent_50"):
        resolved = scenario._resolve_direct_epfd_mask_points_mhz(
            preset=preset, channel_bandwidth_mhz=5.0, custom_mask_points=None,
        )
        assert resolved.shape[1] == 2, f"{preset}: expected (N,2)"
        assert resolved.shape[0] >= 4, f"{preset}: too few points"
        # Should be symmetric: for each positive offset, a matching negative
        offsets = set(round(float(row[0]), 4) for row in resolved)
        for off in offsets:
            assert -off in offsets or off == 0.0, f"{preset}: not symmetric at offset {off}"


@GPU_REQUIRED
def test_fused_histogram_kernel_matches_reference():
    """Fused log10 masked minmax kernel matches separate CuPy operations."""
    import cupy as cp
    values = cp.array([0.0, 1e-16, 1e-12, 1e-8, 1e-4, 1.0, float('nan'), float('inf'), -1.0],
                      dtype=cp.float32)
    result = gpu_accel._positive_value_range_db_cp(values, db_offset_db=0.0)
    assert result is not None
    min_val, max_val = result
    # 1e-16 → -160 dBW, 1.0 → 0 dBW
    assert min_val == pytest.approx(-160.0, abs=0.1)
    assert max_val == pytest.approx(0.0, abs=0.1)


@GPU_REQUIRED
def test_scenario_beamforming_collapsed_params_flow():
    """Beamforming collapsed parameters are accepted by run_gpu_direct_epfd signature."""
    import inspect
    sig = inspect.signature(scenario.run_gpu_direct_epfd)
    assert "beamforming_collapsed" in sig.parameters
    assert "collapsed_baseline_eirp_dbw_hz" in sig.parameters
    assert "collapsed_eval_freq_mhz" in sig.parameters
    assert "collapsed_ref_freq_mhz" in sig.parameters
    # Defaults should be sensible
    assert sig.parameters["beamforming_collapsed"].default is False
    assert sig.parameters["collapsed_baseline_eirp_dbw_hz"].default == -55.6


@GPU_REQUIRED
def test_atmosphere_lut_lookup_matches_binned_reference():
    elevations = np.array([[-5.0, 0.05, 0.1, 15.3, 44.9, 89.95]], dtype=np.float32)
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        context = session.prepare_atmosphere_lut_context(
            frequency_ghz=2690 * u.MHz,
            altitude_km_values=[0.0, 1.052],
            bin_deg=0.1,
            elev_min_deg=0.1,
            elev_max_deg=90.0,
        )
        gpu_result = gpu_accel.copy_device_to_host(
            gpu_accel._lookup_atmosphere_lut_cp(context, elevations, altitude_km=1.052)
        )
    expected = _atm_lin_from_elev_deg_binned_reference(
        elevations,
        altitude_km=1.052,
        frequency_ghz=2.69,
        bin_deg=0.1,
        elev_min_deg=0.1,
        elev_max_deg=90.0,
        max_path_length_km=10_000.0,
    )
    assert_allclose(gpu_result, expected, atol=1e-6, rtol=0.0)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_sample_s1586_pointings_is_seeded_and_respects_cell_bounds():
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        context = session.prepare_s1586_pointing_context(elev_range_deg=(15 * u.deg, 90 * u.deg))
        first = session.sample_s1586_pointings(context, n_samples=4, seed=123, return_device=False)
        second = session.sample_s1586_pointings(context, n_samples=4, seed=123, return_device=False)

    assert_allclose(first["azimuth_deg"], second["azimuth_deg"], atol=0.0, rtol=0.0)
    assert_allclose(first["elevation_deg"], second["elevation_deg"], atol=0.0, rtol=0.0)
    assert first["azimuth_deg"].shape == (4, context.n_cells)
    assert np.all(first["azimuth_deg"] >= context.az_low_deg[None, :])
    assert np.all(first["azimuth_deg"] <= context.az_high_deg[None, :])
    assert np.all(first["elevation_deg"] >= context.el_low_deg[None, :])
    assert np.all(first["elevation_deg"] <= context.el_high_deg[None, :])
    _, _, grid_info = pointgen_S_1586_1(1, rnd_seed=0, elev_range=(15 * u.deg, 90 * u.deg))
    assert context.n_cells == int(grid_info.shape[0])
    session.close(reset_device=False)


@GPU_REQUIRED
def test_conditioned_template_build_and_assignment_match_cpu_reference():
    source_kind = np.array(
        [
            [gpu_accel.SAMPLER_SOURCE_GROUP, gpu_accel.SAMPLER_SOURCE_GROUP],
            [gpu_accel.SAMPLER_SOURCE_GLOBAL, gpu_accel.SAMPLER_SOURCE_INVALID],
        ],
        dtype=np.int8,
    )
    source_id = np.array([[1, 1], [0, -1]], dtype=np.int32)
    vis_mask = np.array([[True, True], [True, False]], dtype=bool)
    beta_max_sat = np.array([0.6, 0.6], dtype=np.float32)
    cos_min_sep = float(np.cos(np.deg2rad(10.0)))

    candidate_pools = {
        "alpha_rad": np.array(
            [
                [0.00, 0.35, 0.75, 1.10],
                [0.10, 0.55, 1.00, 1.45],
            ],
            dtype=np.float32,
        ),
        "beta_rad": np.array(
            [
                [0.08, 0.18, 0.28, 0.38],
                [0.12, 0.22, 0.32, 0.42],
            ],
            dtype=np.float32,
        ),
    }
    candidate_pools["sin_alpha"] = np.sin(candidate_pools["alpha_rad"]).astype(np.float32, copy=False)
    candidate_pools["cos_alpha"] = np.cos(candidate_pools["alpha_rad"]).astype(np.float32, copy=False)
    candidate_pools["sin_beta"] = np.sin(candidate_pools["beta_rad"]).astype(np.float32, copy=False)
    candidate_pools["cos_beta"] = np.cos(candidate_pools["beta_rad"]).astype(np.float32, copy=False)

    alpha0_rad = np.array([[0.05, 0.15], [0.25, 0.35]], dtype=np.float32)
    beta0_rad = np.array([[0.07, 0.09], [0.11, 0.13]], dtype=np.float32)
    start_offsets_template = np.array([0, 1], dtype=np.int32)
    start_offsets_rows = np.array([0, 1, 0], dtype=np.int32)

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        plan = session.make_conditioned_template_plan(
            source_kind,
            source_id,
            vis_mask,
            beta_max_sat,
            mode=gpu_accel.CONDITIONED_TEMPLATE_MODE_PER_SOURCE,
            pool_size=4,
            template_size=3,
        )
        templates_gpu = session.build_conditioned_beam_templates(
            plan,
            candidate_pools,
            cos_min_sep=cos_min_sep,
            start_offsets=start_offsets_template,
            return_device=False,
        )
        assigned_gpu = session.assign_conditioned_beams(
            plan,
            templates_gpu,
            vis_mask_horizon=vis_mask,
            is_co_sat=np.array([[False, True], [False, False]], dtype=bool),
            alpha0_rad=alpha0_rad,
            beta0_rad=beta0_rad,
            sina0=np.sin(alpha0_rad).astype(np.float32, copy=False),
            cosa0=np.cos(alpha0_rad).astype(np.float32, copy=False),
            sinb0=np.sin(beta0_rad).astype(np.float32, copy=False),
            cosb0=np.cos(beta0_rad).astype(np.float32, copy=False),
            beta_max_rad_per_sat=beta_max_sat,
            n_beams=2,
            cos_min_sep=cos_min_sep,
            start_offsets=start_offsets_rows,
            return_device=False,
        )

    templates_cpu = _cpu_build_conditioned_templates(
        candidate_pools,
        gpu_accel.copy_device_to_host(plan.d_unit_beta_max_rad),
        template_size=3,
        cos_min_sep=cos_min_sep,
        start_offsets=start_offsets_template,
    )
    assigned_cpu = _cpu_assign_conditioned_beams(
        plan,
        templates_cpu,
        vis_mask_horizon=vis_mask,
        is_co_sat=np.array([[False, True], [False, False]], dtype=bool),
        alpha0_rad=alpha0_rad,
        beta0_rad=beta0_rad,
        sina0=np.sin(alpha0_rad).astype(np.float32, copy=False),
        cosa0=np.cos(alpha0_rad).astype(np.float32, copy=False),
        sinb0=np.sin(beta0_rad).astype(np.float32, copy=False),
        cosb0=np.cos(beta0_rad).astype(np.float32, copy=False),
        beta_max_rad_per_sat=beta_max_sat,
        n_beams=2,
        cos_min_sep=cos_min_sep,
        start_offsets=start_offsets_rows,
    )

    for key in ("template_idx", "template_valid_count"):
        assert_allclose(templates_gpu[key], templates_cpu[key], atol=0.0, rtol=0.0)
    for key in ("beam_idx", "beam_valid"):
        assert_allclose(assigned_gpu[key], assigned_cpu[key], atol=0.0, rtol=0.0)
    assert_allclose(assigned_gpu["beam_alpha_rad"], assigned_cpu["beam_alpha_rad"], atol=0.0, rtol=0.0, equal_nan=True)
    assert_allclose(assigned_gpu["beam_beta_rad"], assigned_cpu["beam_beta_rad"], atol=0.0, rtol=0.0, equal_nan=True)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_fill_conditioned_beams_streaming_reaches_nbeam_on_reduced_b525_step2_case():
    case = _build_b525_step2_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        context = session.prepare_angle_sampler_context(case["sampler"])
        source_result = session.resolve_angle_sampler_sources(
            context,
            case["sat_topo"][:, 0, :, 0].astype(np.float32, copy=False),
            case["sat_topo"][:, 0, :, 1].astype(np.float32, copy=False),
            case["sat_belt_id_rows"],
            return_device=False,
        )
        invalid_visible = case["vis_mask_horizon"] & (~np.asarray(source_result["valid_mask"], dtype=bool))
        assert not np.any(invalid_visible)

        result = session.fill_conditioned_beams_streaming(
            context,
            source_result["source_kind"],
            source_result["source_id"],
            vis_mask_horizon=case["vis_mask_horizon"],
            is_co_sat=case["is_co_sat"],
            alpha0_rad=case["alpha0_rad"],
            beta0_rad=case["beta0_rad"],
            beta_max_rad_per_sat=case["beta_max_rad_per_sat"],
            n_beams=7,
            cos_min_sep=case["cos_min_sep"],
            seed=12345,
            return_device=False,
        )

    assert np.all(result["beam_valid"][case["vis_mask_horizon"]] == 7)
    assert int(result["unfinished_row_count"]) == 0
    session.close(reset_device=False)


@GPU_REQUIRED
def test_accumulate_ras_power_matches_cpu_reference_for_small_batch():
    wavelength = (2690 * u.MHz).to(u.m, equivalencies=u.spectral())
    sat_topo = np.array([[[[12.0, 35.0, 700.0, 0.0], [44.0, 48.0, 810.0, 0.0]]]], dtype=np.float32)
    sat_azel = np.array([[[[8.0, 4.5, 0.0], [15.0, 6.0, 0.0]]]], dtype=np.float32)
    beam_idx = np.array([[[-2, 0], [0, -1]]], dtype=np.int32)
    beam_alpha = np.array([[[np.deg2rad(8.0), np.deg2rad(18.0)], [np.deg2rad(12.0), np.nan]]], dtype=np.float32)
    beam_beta = np.array([[[np.deg2rad(4.5), np.deg2rad(16.0)], [np.deg2rad(14.0), np.nan]]], dtype=np.float32)
    telescope_az = np.array([[20.0, 100.0, 220.0]], dtype=np.float32)
    telescope_el = np.array([[25.0, 55.0, 75.0]], dtype=np.float32)
    orbit_radius = np.array([R_earth.to_value(u.m) + 525_000.0, R_earth.to_value(u.m) + 540_000.0], dtype=np.float32)

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=wavelength,
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        rx_context = session.prepare_ras_pattern_context(
            diameter_m=15 * u.m,
            wavelength_m=wavelength,
        )
        atm_context = session.prepare_atmosphere_lut_context(
            frequency_ghz=2690 * u.MHz,
            altitude_km_values=[0.0, 1.052],
        )
        gpu_result = session.accumulate_ras_power(
            s1528_pattern_context=tx_context,
            ras_pattern_context=rx_context,
            atmosphere_lut_context=atm_context,
            sat_topo=sat_topo,
            sat_azel=sat_azel,
            beam_idx=beam_idx,
            beam_alpha_rad=beam_alpha,
            beam_beta_rad=beam_beta,
            telescope_azimuth_deg=telescope_az,
            telescope_elevation_deg=telescope_el,
            orbit_radius_m_per_sat=orbit_radius,
            observer_alt_km=1.052,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )

    deg2rad = np.float32(np.pi / 180.0)
    rad2deg = np.float32(180.0 / np.pi)
    four_pi = np.float32(4.0 * np.pi)
    wavelength_m = np.float32(wavelength.to_value(u.m))
    earth_radius_m = np.float32(R_earth.to_value(u.m))
    pfd0_lin = np.float32(10.0 ** (-83.5 / 10.0))
    pfd_from_prx_iso_scale = np.float32((4.0 * np.pi) / (float(wavelength_m) ** 2))

    alpha0 = sat_azel[:, 0, :, 0].astype(np.float32, copy=False)
    beta0 = sat_azel[:, 0, :, 1].astype(np.float32, copy=False)
    alpha0_rad = np.remainder(alpha0 * deg2rad, np.float32(2.0 * np.pi)).astype(np.float32, copy=False)
    beta0_rad = (beta0 * deg2rad).astype(np.float32, copy=False)
    sina0 = np.sin(alpha0_rad).astype(np.float32, copy=False)
    cosa0 = np.cos(alpha0_rad).astype(np.float32, copy=False)
    sinb0 = np.sin(beta0_rad).astype(np.float32, copy=False)
    cosb0 = np.cos(beta0_rad).astype(np.float32, copy=False)

    valid_beam = (beam_idx >= 0) | (beam_idx == -2)
    mask_bore = beam_idx == -2
    beam_alpha_safe = np.where(valid_beam, beam_alpha, np.float32(0.0)).astype(np.float32, copy=False)
    beam_beta_safe = np.where(valid_beam, beam_beta, np.float32(0.0)).astype(np.float32, copy=False)
    beam_sinb = np.sin(beam_beta_safe).astype(np.float32, copy=False)
    beam_cosb = np.cos(beam_beta_safe).astype(np.float32, copy=False)
    cos_da = np.cos(alpha0_rad[:, :, None] - beam_alpha_safe).astype(np.float32, copy=False)
    cos_gamma_tx = (
        cosb0[:, :, None] * beam_cosb
        + sinb0[:, :, None] * beam_sinb * cos_da
    ).astype(np.float32, copy=False)
    cos_gamma_tx = np.where(mask_bore, np.float32(1.0), cos_gamma_tx).astype(np.float32, copy=False)
    cos_gamma_tx = np.where(valid_beam, cos_gamma_tx, np.float32(-1.0)).astype(np.float32, copy=False)
    np.clip(cos_gamma_tx, np.float32(-1.0), np.float32(1.0), out=cos_gamma_tx)
    gtx_offset_deg = (np.arccos(cos_gamma_tx) * rad2deg).astype(np.float32, copy=False)
    gtx_db = s_1528_rec1_4_pattern_amend(
        gtx_offset_deg * u.deg,
        wavelength=wavelength,
        Lt=1.6 * u.m,
        Lr=1.6 * u.m,
        l=2,
        SLR=20.0 * cnv.dB,
        far_sidelobe_start=90 * u.deg,
        far_sidelobe_level=-20.0 * cnv.dBi,
        use_numba=False,
    ).to_value(cnv.dB).astype(np.float32, copy=False)
    gtx_lin = (np.float32(10.0) ** (gtx_db / np.float32(10.0))).astype(np.float32, copy=False)
    gtx_lin = np.where(valid_beam, gtx_lin, np.float32(0.0)).astype(np.float32, copy=False)

    range_m = (sat_topo[:, 0, :, 2] * np.float32(1000.0)).astype(np.float32, copy=False)
    sat_el_deg = sat_topo[:, 0, :, 1].astype(np.float32, copy=False)
    vis_horizon = sat_el_deg > np.float32(0.0)
    r_m = orbit_radius[None, :, None].astype(np.float32, copy=False)
    term = r_m * beam_cosb
    disc = term * term - (r_m * r_m - earth_radius_m * earth_radius_m)
    valid_geom = valid_beam & (~mask_bore) & (disc >= 0.0) & (beam_cosb > 0.0)
    d_target_geom_m = (term - np.sqrt(np.where(valid_geom, disc, 0.0))).astype(np.float32, copy=False)
    atm_ras = _atm_lin_from_elev_deg_binned_reference(
        sat_el_deg,
        altitude_km=1.052,
        frequency_ghz=2.69,
        bin_deg=0.1,
        elev_min_deg=0.1,
        elev_max_deg=90.0,
        max_path_length_km=10_000.0,
    )
    cos_e_target = (r_m / earth_radius_m) * beam_sinb
    np.clip(cos_e_target, np.float32(0.0), np.float32(1.0), out=cos_e_target)
    e_target_deg = (np.arccos(cos_e_target) * rad2deg).astype(np.float32, copy=False)
    e_target_deg = np.where(valid_geom, np.maximum(e_target_deg, np.float32(0.1)), np.float32(0.0)).astype(np.float32, copy=False)
    atm_target = _atm_lin_from_elev_deg_binned_reference(
        e_target_deg,
        altitude_km=0.0,
        frequency_ghz=2.69,
        bin_deg=0.1,
        elev_min_deg=0.1,
        elev_max_deg=90.0,
        max_path_length_km=10_000.0,
    )
    atm_target_safe = np.where(valid_geom, atm_target, np.float32(1.0)).astype(np.float32, copy=False)
    ptx = np.where(
        valid_geom,
        pfd0_lin * four_pi * d_target_geom_m * d_target_geom_m / atm_target_safe,
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    ptx_co = (pfd0_lin * four_pi * range_m * range_m / atm_ras).astype(np.float32, copy=False)
    for beam_slot in range(beam_idx.shape[2]):
        ptx[:, :, beam_slot] = np.where(mask_bore[:, :, beam_slot], ptx_co, ptx[:, :, beam_slot]).astype(np.float32, copy=False)

    fspl = (wavelength_m / (four_pi * range_m)) ** np.float32(2.0)
    fspl = (fspl * atm_ras).astype(np.float32, copy=False)
    scale = (np.sum(ptx * gtx_lin, axis=2).astype(np.float32, copy=False) * fspl).astype(np.float32, copy=False)
    scale *= vis_horizon.astype(np.float32, copy=False)
    pfd_per_sat = (scale * pfd_from_prx_iso_scale).astype(np.float32, copy=False)

    tel_az_rad = telescope_az * deg2rad
    tel_el_rad = telescope_el * deg2rad
    sat_az_rad = sat_topo[:, 0, :, 0].astype(np.float32, copy=False) * deg2rad
    sat_el_rad = sat_el_deg * deg2rad
    cos_daz = (
        np.cos(tel_az_rad)[:, :, None] * np.cos(sat_az_rad)[:, None, :]
        + np.sin(tel_az_rad)[:, :, None] * np.sin(sat_az_rad)[:, None, :]
    ).astype(np.float32, copy=False)
    cos_gamma_rx = (
        np.sin(tel_el_rad)[:, :, None] * np.sin(sat_el_rad)[:, None, :]
        + np.cos(tel_el_rad)[:, :, None] * np.cos(sat_el_rad)[:, None, :] * cos_daz
    ).astype(np.float32, copy=False)
    np.clip(cos_gamma_rx, np.float32(-1.0), np.float32(1.0), out=cos_gamma_rx)
    grx_offset_deg = (np.arccos(cos_gamma_rx) * rad2deg).astype(np.float32, copy=False)
    grx_db = ras_pattern(grx_offset_deg * u.deg, 15 * u.m, wavelength).to_value(cnv.dB).astype(np.float32, copy=False)
    grx_lin = (np.float32(10.0) ** (grx_db / np.float32(10.0))).astype(np.float32, copy=False)
    grx_lin *= vis_horizon[:, None, :].astype(np.float32, copy=False)
    epfd_total = np.sum(grx_lin * pfd_per_sat[:, None, :], axis=2).astype(np.float32, copy=False)[:, None, :]
    prx_total = np.sum(grx_lin * scale[:, None, :], axis=2).astype(np.float32, copy=False)[:, None, :]

    assert_allclose(gpu_result["EPFD_W_m2"], epfd_total, atol=5e-5, rtol=5e-5)
    assert_allclose(gpu_result["Prx_total_W"], prx_total, atol=5e-5, rtol=5e-5)
    assert_allclose(
        gpu_result["PFD_total_RAS_STATION_W_m2"],
        np.sum(pfd_per_sat, axis=1, dtype=np.float32),
        atol=5e-5,
        rtol=5e-5,
    )
    assert_allclose(
        gpu_result["PFD_per_sat_RAS_STATION_W_m2"],
        pfd_per_sat,
        atol=5e-5,
        rtol=5e-5,
    )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_session_pure_reroute_service_curve_matches_cpu_exact_solver():
    eligible_mask = np.array(
        [
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            [
                [True, True, True],
                [False, True, True],
                [True, False, False],
            ],
        ],
        dtype=bool,
    )
    beam_caps = np.arange(3, dtype=np.int32)
    expected = satsim.pure_reroute_service_curve(
        eligible_mask,
        nco=1,
        beam_caps=beam_caps,
    )

    session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
    try:
        gpu_host = session.pure_reroute_service_curve(
            eligible_mask,
            nco=1,
            beam_caps=beam_caps,
            return_device=False,
        )
        gpu_device = session.pure_reroute_service_curve(
            eligible_mask,
            nco=1,
            beam_caps=beam_caps,
            return_device=True,
        )

        for key, expected_value in expected.items():
            host_value = gpu_host[key]
            device_value = gpu_accel.copy_device_to_host(gpu_device[key])
            assert_allclose(host_value, expected_value, atol=0.0, rtol=0.0)
            assert_allclose(device_value, expected_value, atol=0.0, rtol=0.0)
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_session_pure_reroute_service_curve_fills_tail_after_first_full_service_cap():
    eligible_mask = np.array(
        [
            [
                [True, True],
                [True, False],
                [False, True],
            ]
        ],
        dtype=bool,
    )
    beam_caps = np.arange(8, dtype=np.int32)
    expected = satsim.pure_reroute_service_curve(
        eligible_mask,
        nco=1,
        beam_caps=beam_caps,
    )

    session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
    try:
        gpu_host = session.pure_reroute_service_curve(
            eligible_mask,
            nco=1,
            beam_caps=beam_caps,
            return_device=False,
        )

        for key, expected_value in expected.items():
            assert_allclose(gpu_host[key], expected_value, atol=0.0, rtol=0.0)
        assert_equal(gpu_host["matched_links"], np.array([[0, 2, 3, 3, 3, 3, 3, 3]], dtype=np.int32))
        assert_equal(gpu_host["required_beam_cap"], np.array([2], dtype=np.int32))
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_session_pure_reroute_service_curve_supports_device_input_and_nco_gt_one():
    eligible_mask = np.array(
        [
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, True, True, True],
            ],
            [
                [True, True, True, False],
                [True, False, False, True],
                [False, True, True, True],
            ],
        ],
        dtype=bool,
    )
    beam_caps = np.arange(4, dtype=np.int32)
    expected = satsim.pure_reroute_service_curve(
        eligible_mask,
        nco=2,
        beam_caps=beam_caps,
    )

    session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
    try:
        eligible_device = gpu_accel.cp.asarray(eligible_mask)
        gpu_host = session.pure_reroute_service_curve(
            eligible_device,
            nco=2,
            beam_caps=beam_caps,
            return_device=False,
        )
        gpu_device = session.pure_reroute_service_curve(
            eligible_device,
            nco=2,
            beam_caps=beam_caps,
            return_device=True,
        )

        for key, expected_value in expected.items():
            assert_allclose(gpu_host[key], expected_value, atol=0.0, rtol=0.0)
            assert_allclose(
                gpu_accel.copy_device_to_host(gpu_device[key]),
                expected_value,
                atol=0.0,
                rtol=0.0,
            )
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_session_pure_reroute_service_curve_supports_csr_input():
    eligible_mask = np.array(
        [
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            [
                [True, True, True],
                [False, True, True],
                [True, False, False],
            ],
        ],
        dtype=bool,
    )
    csr_payload = satsim._pure_reroute_dense_mask_to_csr_payload(eligible_mask)
    beam_caps = np.arange(3, dtype=np.int32)
    expected = satsim.pure_reroute_service_curve(
        csr_payload,
        nco=1,
        beam_caps=beam_caps,
    )

    session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
    try:
        gpu_host = session.pure_reroute_service_curve(
            csr_payload,
            nco=1,
            beam_caps=beam_caps,
            return_device=False,
        )
        gpu_device = session.pure_reroute_service_curve(
            csr_payload,
            nco=1,
            beam_caps=beam_caps,
            return_device=True,
        )

        for key, expected_value in expected.items():
            assert_allclose(gpu_host[key], expected_value, atol=0.0, rtol=0.0)
            assert_allclose(
                gpu_accel.copy_device_to_host(gpu_device[key]),
                expected_value,
                atol=0.0,
                rtol=0.0,
            )
    finally:
        session.close(reset_device=False)


@GPU_REQUIRED
def test_session_pure_reroute_service_curve_internal_batching_matches_unsplit(monkeypatch):
    eligible_mask = np.array(
        [
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, True, True, False],
            ],
            [
                [True, True, True, False],
                [False, True, True, True],
                [True, False, False, True],
            ],
            [
                [True, False, True, True],
                [True, True, False, False],
                [False, True, True, True],
            ],
        ],
        dtype=bool,
    )
    beam_caps = np.arange(4, dtype=np.int32)
    expected = satsim.pure_reroute_service_curve(
        eligible_mask,
        nco=1,
        beam_caps=beam_caps,
    )

    session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
    try:
        monkeypatch.setattr(gpu_accel, "_estimate_pure_reroute_batch_slots", lambda *_args: 1)
        batched = session.pure_reroute_service_curve(
            eligible_mask,
            nco=1,
            beam_caps=beam_caps,
            return_device=False,
        )
        for key, expected_value in expected.items():
            assert_allclose(batched[key], expected_value, atol=0.0, rtol=0.0)
    finally:
        session.close(reset_device=False)


def _available_numba_performance_warning_categories() -> list[type[Warning]]:
    return list(gpu_accel._iter_numba_low_occupancy_warning_categories())


def _emit_dispatcher_warning(
    message: str,
    *,
    category: type[Warning],
    module_name: str,
) -> None:
    warnings.warn_explicit(
        message,
        category,
        filename=f"{module_name.replace('.', '/')}.py",
        lineno=716,
        module=module_name,
    )


def _strip_warning_ansi(text: str) -> str:
    return text.replace("\x1b[1m", "").replace("\x1b[0m", "")


@pytest.mark.parametrize(
    "module_name",
    ("numba_cuda.numba.cuda.dispatcher", "numba.cuda.dispatcher"),
)
def test_configure_numba_low_occupancy_warnings_suppresses_only_target_message(module_name: str):
    categories = _available_numba_performance_warning_categories()
    if not categories:
        pytest.skip("Numba performance warning classes are unavailable in this environment.")

    gpu_accel.configure_numba_low_occupancy_warnings(show=False)
    category = categories[0]
    with warnings.catch_warnings(record=True) as caught:
        _emit_dispatcher_warning(
            "Grid size 1 will likely result in GPU under-utilization due to low occupancy.",
            category=category,
            module_name=module_name,
        )
        _emit_dispatcher_warning(
            "This is a different NumbaPerformanceWarning.",
            category=category,
            module_name=module_name,
        )

    messages = [_strip_warning_ansi(str(item.message)) for item in caught]
    assert "Grid size 1 will likely result in GPU under-utilization due to low occupancy." not in messages
    assert messages == ["This is a different NumbaPerformanceWarning."]


def test_configure_numba_low_occupancy_warnings_show_true_reenables_warning():
    categories = _available_numba_performance_warning_categories()
    if not categories:
        pytest.skip("Numba performance warning classes are unavailable in this environment.")

    gpu_accel.configure_numba_low_occupancy_warnings(show=True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            _emit_dispatcher_warning(
                "Grid size 1 will likely result in GPU under-utilization due to low occupancy.",
                category=categories[0],
                module_name="numba_cuda.numba.cuda.dispatcher",
            )
        assert [_strip_warning_ansi(str(item.message)) for item in caught] == [
            "Grid size 1 will likely result in GPU under-utilization due to low occupancy."
        ]
    finally:
        gpu_accel.configure_numba_low_occupancy_warnings(show=False)


def test_configure_numba_low_occupancy_warnings_is_idempotent():
    categories = _available_numba_performance_warning_categories()
    if not categories:
        pytest.skip("Numba performance warning classes are unavailable in this environment.")

    gpu_accel.configure_numba_low_occupancy_warnings(show=False)
    count_once = gpu_accel._count_managed_numba_low_occupancy_filters()
    gpu_accel.configure_numba_low_occupancy_warnings(show=False)
    count_twice = gpu_accel._count_managed_numba_low_occupancy_filters()
    gpu_accel.configure_numba_low_occupancy_warnings(show=True)
    count_cleared = gpu_accel._count_managed_numba_low_occupancy_filters()
    gpu_accel.configure_numba_low_occupancy_warnings(show=False)
    count_restored = gpu_accel._count_managed_numba_low_occupancy_filters()

    assert count_once > 0
    assert count_once == count_twice
    assert count_cleared == 0
    assert count_restored == count_once


def test_gpu_accel_import_env_override_disables_low_occupancy_suppression():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["SCEPTER_SHOW_NUMBA_LOW_OCCUPANCY_WARNINGS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from scepter import gpu_accel; "
                "print(json.dumps({'count': gpu_accel._count_managed_numba_low_occupancy_filters()}))"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["count"] == 0


def test_gpu_accel_public_docs_cover_lifecycle_and_exports():
    module_doc = inspect.getdoc(gpu_accel)
    assert module_doc is not None
    for heading in (
        "Session lifecycle",
        "Prepared contexts",
        "Propagation and geometry APIs",
        "Sampler, pointing, and pattern helpers",
        "Link selection and EPFD helpers",
        "Cleanup semantics",
        "Warning policy",
    ):
        assert heading in module_doc

    for name in gpu_accel.__all__:
        value = getattr(gpu_accel, name)
        if inspect.isclass(value) or inspect.isfunction(value):
            assert inspect.getdoc(value), f"{name} is missing a public docstring"

    for obj in (
        gpu_accel.GpuScepterSession.pure_reroute_service_curve,
        gpu_accel.GpuScepterSession.propagate_many,
    ):
        doc = inspect.getdoc(obj)
        assert doc is not None
        assert "Parameters" in doc
        assert "Returns" in doc
        assert "Notes" in doc


# ---------------------------------------------------------------------------
# Multi-system registration and combination tests
# ---------------------------------------------------------------------------

@GPU_REQUIRED
def test_multi_system_register_two_systems():
    """Register two systems with different patterns, verify bundle storage."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        sat_ctx_a = session.prepare_satellite_context(NEAR_TLES[:2])
        pat_ctx_a = session.prepare_s1528_rec12_pattern_context(
            wavelength_m=0.15, gm_dbi=38.0, ln_db=-20.0, z=1.0,
        )
        sat_ctx_b = session.prepare_satellite_context(NEAR_TLES)
        pat_ctx_b = session.prepare_m2101_pattern_context(
            g_emax_db=5.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=120.0, theta_3db_deg=120.0,
            d_h=0.5, d_v=0.5, n_h=8, n_v=8,
        )
        bundle_a = session.register_system(
            0, system_name="A", satellite_context=sat_ctx_a,
            pattern_context=pat_ctx_a, nco=1, nbeam=1,
        )
        bundle_b = session.register_system(
            1, system_name="B", satellite_context=sat_ctx_b,
            pattern_context=pat_ctx_b, nco=2, nbeam=2,
        )
        systems = session.registered_systems()
        assert len(systems) == 2
        assert systems[0].system_name == "A"
        assert systems[0].satellite_count == 2
        assert systems[1].satellite_count == len(NEAR_TLES)
        assert session.max_satellite_count() == max(2, len(NEAR_TLES))
    session.close(reset_device=False)


@GPU_REQUIRED
def test_multi_system_combine_powers_sum():
    """combine_system_powers with sum policy equals element-wise addition."""
    import cupy as cp
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        a = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        b = cp.array([0.5, 1.0, 1.5], dtype=cp.float32)
        combined = session.combine_system_powers([a, b], policy="sum")
        expected = cp.array([1.5, 3.0, 4.5], dtype=cp.float32)
        np.testing.assert_allclose(combined.get(), expected.get(), rtol=1e-6)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_multi_system_combine_single_system_passthrough():
    """combine_system_powers with one system returns it directly."""
    import cupy as cp
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        a = cp.array([1.0, 2.0], dtype=cp.float32)
        combined = session.combine_system_powers([a], policy="sum")
        assert combined is a  # should be the same object, no copy
    session.close(reset_device=False)


@GPU_REQUIRED
def test_multi_system_unregister():
    """Unregistering a system removes it from the registry."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        sat_ctx = session.prepare_satellite_context(NEAR_TLES[:2])
        pat_ctx = session.prepare_s1528_rec12_pattern_context(
            wavelength_m=0.15, gm_dbi=38.0, ln_db=-20.0, z=1.0,
        )
        session.register_system(0, satellite_context=sat_ctx, pattern_context=pat_ctx)
        session.register_system(1, satellite_context=sat_ctx, pattern_context=pat_ctx)
        assert len(session.registered_systems()) == 2
        session.unregister_system(0)
        assert len(session.registered_systems()) == 1
        assert 1 in session.registered_systems()
    session.close(reset_device=False)
