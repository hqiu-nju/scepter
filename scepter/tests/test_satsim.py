#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy import time as astrotime
from numpy.testing import assert_allclose, assert_equal
import pytest

import cysgp4
from pycraf import conversions as cnv

from scepter import satsim, scenario, tleforger
from scepter.angle_sampler import JointAngleSampler
from scepter.antenna import calculate_beamwidth_1d, s_1528_rec1_4_pattern_amend


def _step2_sampler_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    for candidate in ("1.13_US_System_B525_random.npz", "1_13_US_System_B525_random.npz"):
        path = root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError("Step-2 sampler artifact was not found in the repository root.")


def _load_step2_sampler() -> JointAngleSampler:
    try:
        return JointAngleSampler.load(str(_step2_sampler_path()))
    except FileNotFoundError:
        pytest.skip("EPFDflow sampler artifact not present in this checkout.")


def _build_b525_step2_cpu_case() -> dict[str, np.ndarray]:
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
    satellite_metadata = tleforger.expand_belt_metadata_to_satellites(belt_package)
    sat_min_elev_deg = satellite_metadata["sat_min_elevation_deg"]
    sat_beta_max_rad = np.deg2rad(satellite_metadata["sat_beta_max_deg"]).astype(np.float32, copy=False)
    sat_belt_id = satellite_metadata["sat_belt_id"]

    ras_station = cysgp4.PyObserver(21.443611, -30.712777, 1.052)
    layout = scenario.build_observer_layout(ras_station, [])
    observer_list = layout["observer_arr"]
    observers_new = observer_list[np.newaxis, :, np.newaxis]
    tles_new = belt_package["tle_list"][np.newaxis, np.newaxis, :]
    mjd = astrotime.Time(datetime(2025, 1, 1, 0, 0, 0), scale="utc").mjd
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
    sat_belt_id_rows = np.broadcast_to(sat_belt_id_eff[None, :], vis_mask_horizon.shape).astype(np.int16, copy=False)
    return {
        "sampler": _load_step2_sampler(),
        "sat_topo": sat_topo,
        "vis_mask_horizon": vis_mask_horizon,
        "is_co_sat": is_co_sat,
        "alpha0_rad": alpha0_rad,
        "beta0_rad": beta0_rad,
        "beta_max_rad_per_sat": beta_max_rad_eff,
        "sat_belt_id_rows": sat_belt_id_rows,
        "cos_min_sep": float(np.cos(np.float32(theta_sep.to_value(u.rad)))),
    }


def _random_prefer_numba_values() -> list[bool]:
    values = [False]
    if satsim.HAS_NUMBA:
        values.append(True)
    return values


def _boresight_selection_case() -> dict[str, np.ndarray]:
    sat_topo = np.zeros((1, 2, 2, 2), dtype=np.float32)
    sat_topo[0, 0, :, 1] = np.array([80.0, 70.0], dtype=np.float32)
    sat_topo[0, 1, :, 1] = np.array([76.0, 75.0], dtype=np.float32)

    sat_azel = np.zeros((1, 2, 2, 2), dtype=np.float32)
    sat_azel[0, 0, :, 0] = np.array([10.0, 90.0], dtype=np.float32)
    sat_azel[0, 1, :, 0] = np.array([20.0, 100.0], dtype=np.float32)
    sat_azel[0, 0, :, 1] = np.array([5.0, 6.0], dtype=np.float32)
    sat_azel[0, 1, :, 1] = np.array([7.0, 8.0], dtype=np.float32)

    ras_topo = np.array(
        [[[10.0, 50.0], [90.0, 50.0]]],
        dtype=np.float32,
    )

    return {
        "sat_topo": sat_topo,
        "sat_azel": sat_azel,
        "ras_topo": ras_topo,
        "pointing_az_deg": np.array([[10.0, 90.0]], dtype=np.float32),
        "pointing_el_deg": np.array([[50.0, 50.0]], dtype=np.float32),
        "sat_beta_max": np.array([20.0, 20.0], dtype=np.float32),
        "sat_belt_id": np.array([0, 1], dtype=np.int16),
    }


class TestSelectSatelliteLinks:

    def test_max_elevation_supports_per_satellite_thresholds_and_payload(self):
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

        result = satsim.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=np.array([10.0, 30.0, 40.0], dtype=np.float64),
            beta_max_deg_per_sat=np.array([10.0, 20.0, 20.0], dtype=np.float32),
            sat_belt_id_per_sat=np.array([0, 1, 1], dtype=np.int16),
            n_links=2,
            strategy="max_elevation",
            include_counts=True,
            include_payload=True,
            prefer_numba=False,
        )

        assert_equal(result["assignments"], np.array([[[1, 0], [0, 2]]], dtype=np.int32))
        assert_equal(result["sat_beam_counts_demand"], np.array([[2, 1, 1]], dtype=np.int32))
        assert_equal(result["sat_beam_counts_eligible"], np.array([[2, 1, 0]], dtype=np.int32))
        assert_equal(result["sat_belt_id"], np.array([[[1, 0], [0, -1]]], dtype=np.int16))
        assert_equal(result["cone_ok"], np.array([[[True, True], [True, False]]], dtype=bool))
        assert_allclose(
            result["sat_azimuth"],
            np.array([[[110.0, 100.0], [100.0, np.nan]]], dtype=np.float32),
            equal_nan=True,
        )
        assert_allclose(
            result["sat_beta"],
            np.array([[[15.0, 5.0], [5.0, np.nan]]], dtype=np.float32),
            equal_nan=True,
        )

    @pytest.mark.parametrize("prefer_numba", _random_prefer_numba_values())
    def test_random_selection_is_reproducible_and_uses_sentinels(self, prefer_numba):
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
        sat_belt_id = np.array([0, 1, 2, 3], dtype=np.int16)

        result_a = satsim.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=20.0,
            beta_max_deg_per_sat=10.0,
            sat_belt_id_per_sat=sat_belt_id,
            n_links=1,
            strategy="random",
            rng=np.random.default_rng(123),
            prefer_numba=prefer_numba,
        )
        result_b = satsim.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=20.0,
            beta_max_deg_per_sat=10.0,
            sat_belt_id_per_sat=sat_belt_id,
            n_links=1,
            strategy="random",
            rng=np.random.default_rng(123),
            prefer_numba=prefer_numba,
        )

        assert_equal(result_a["assignments"], result_b["assignments"])
        assert_equal(result_a["sat_beam_counts_demand"], result_b["sat_beam_counts_demand"])
        assert_equal(result_a["sat_beam_counts_eligible"], result_b["sat_beam_counts_eligible"])

        no_link_mask = result_a["assignments"] < 0
        assert np.any(no_link_mask)
        assert np.all(result_a["sat_belt_id"][no_link_mask] == -1)
        assert np.all(~result_a["cone_ok"][no_link_mask])
        assert np.all(np.isnan(result_a["sat_azimuth"][no_link_mask]))
        assert np.all(np.isnan(result_a["sat_beta"][no_link_mask]))

    def test_cell_active_mask_suppresses_inactive_rows_in_direct_selector(self):
        sat_topo = np.zeros((1, 2, 2, 2), dtype=np.float32)
        sat_topo[0, :, :, 1] = np.array([[40.0, 35.0], [50.0, 45.0]], dtype=np.float32)
        sat_azel = np.zeros((1, 2, 2, 2), dtype=np.float32)
        sat_azel[0, :, :, 0] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        sat_azel[0, :, :, 1] = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
        ras_topo = np.array([[[100.0, 45.0], [110.0, 46.0]]], dtype=np.float32)

        result = satsim.select_satellite_links(
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
            cell_active_mask=np.array([[True, False]], dtype=bool),
            prefer_numba=False,
        )

        assert_equal(result["assignments"], np.array([[[0], [-1]]], dtype=np.int32))
        assert_equal(result["sat_beam_counts_demand"], np.array([[1, 0]], dtype=np.int32))
        assert_equal(result["sat_beam_counts_eligible"], np.array([[1, 0]], dtype=np.int32))
        assert_equal(result["sat_eligible_mask"][0, 1], np.array([False, False], dtype=bool))
        assert np.isnan(result["sat_azimuth"][0, 1, 0])
        assert not bool(result["cone_ok"][0, 1, 0])

    def test_cell_active_mask_is_honoured_by_finite_chunk_library(self):
        sat_topo = np.zeros((1, 2, 2, 2), dtype=np.float32)
        sat_topo[0, :, :, 1] = np.array([[60.0, 55.0], [58.0, 57.0]], dtype=np.float32)
        sat_azel = np.zeros((1, 2, 2, 2), dtype=np.float32)
        sat_azel[0, :, :, 0] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        sat_azel[0, :, :, 1] = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
        ras_topo = np.array([[[100.0, 45.0], [110.0, 46.0]]], dtype=np.float32)

        library = satsim.SatelliteLinkSelectionLibrary(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=20.0,
            n_links=1,
            n_beam=1,
            strategy="max_elevation",
            sat_belt_id_per_sat=np.array([0, 1], dtype=np.int16),
            beta_max_deg_per_sat=np.array([10.0, 10.0], dtype=np.float32),
            ras_topo=ras_topo,
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            cell_active_mask=np.array([True, False], dtype=bool),
        )
        library.add_chunk(0, sat_topo, sat_azel=sat_azel)
        result = library.finalize()

        assert_equal(result["assignments"], np.array([[[0], [-1]]], dtype=np.int32))
        assert_equal(result["sat_beam_counts_demand"], np.array([[1, 0]], dtype=np.int32))
        assert_equal(result["sat_beam_counts_eligible"], np.array([[1, 0]], dtype=np.int32))
        assert_equal(result["sat_eligible_mask"][0, 1], np.array([False, False], dtype=bool))

    def test_link_summary_uses_counts_and_belt_ids(self):
        sat_topo = np.zeros((2, 2, 3, 2), dtype=np.float32)
        sat_topo[..., 1] = np.array(
            [
                [[10.0, 35.0, 20.0], [45.0, 0.0, 30.0]],
                [[40.0, 10.0, 25.0], [15.0, 28.0, 32.0]],
            ],
            dtype=np.float32,
        )
        sat_azel = np.zeros((2, 2, 3, 2), dtype=np.float32)
        sat_azel[..., 0] = 10.0
        sat_azel[..., 1] = 5.0
        ras_topo = np.array(
            [
                [[100.0, 45.0], [110.0, 46.0], [120.0, 47.0]],
                [[130.0, 48.0], [140.0, 49.0], [150.0, 50.0]],
            ],
            dtype=np.float32,
        )

        result = satsim.select_satellite_links(
            sat_topo,
            sat_azel=sat_azel,
            ras_topo=ras_topo,
            min_elevation_deg=20.0 * u.deg,
            n_links=1,
            strategy="max_elevation",
            cell_observer_offset=0,
            sat_belt_id_per_sat=np.array([0, 1, 1], dtype=np.int16),
            beta_max_deg_per_sat=15.0,
            include_counts=True,
            include_payload=True,
            prefer_numba=False,
        )
        summary = satsim.summarize_link_selection(result, n_belts=2)

        assert summary["selected_links"] == 4
        assert summary["cone_ok_links"] == 4
        assert summary["frac_ok"] == pytest.approx(1.0)
        assert_equal(summary["belt_hist"], np.array([2, 2], dtype=np.int64))

    def test_finite_n_beam_max_elevation_matches_capacity_and_counts(self):
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

        result = satsim.select_satellite_links(
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

        assert_equal(result["assignments"], np.array([[[0], [1], [2]]], dtype=np.int32))
        assert_equal(result["sat_beam_counts_demand"], np.array([[1, 1, 1]], dtype=np.int32))
        assert_equal(result["sat_beam_counts_eligible"], np.array([[1, 1, 0]], dtype=np.int32))
        assert_equal(result["cone_ok"], np.array([[[True], [True], [False]]], dtype=bool))
        assert_allclose(
            result["sat_beta"],
            np.array([[[5.0], [9.0], [np.nan]]], dtype=np.float32),
            equal_nan=True,
        )

    def test_finite_n_beam_chunk_library_matches_one_shot_and_chunk_order(self):
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

        direct = satsim.select_satellite_links(
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

        library = satsim.SatelliteLinkSelectionLibrary(
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
        chunked = library.finalize()

        for key in (
            "assignments",
            "sat_azimuth",
            "sat_elevation",
            "sat_alpha",
            "sat_beta",
            "sat_belt_id",
            "cone_ok",
            "sat_beam_counts_demand",
            "sat_beam_counts_eligible",
            "sat_eligible_mask",
        ):
            if chunked[key].dtype.kind == "f":
                assert_allclose(chunked[key], direct[key], atol=0.0, rtol=0.0, equal_nan=True)
            else:
                assert_equal(chunked[key], direct[key])

    def test_boresight_theta1_global_shutdown_matches_chunk_library(self):
        case = _boresight_selection_case()
        direct = satsim.select_satellite_links(
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
            boresight_theta1_deg=5.0,
        )

        library = satsim.SatelliteLinkSelectionLibrary(
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
        chunked = library.finalize()

        expected_assignments = np.array([[[[1], [1]], [[0], [0]]]], dtype=np.int32)
        assert_equal(direct["assignments"], expected_assignments)
        assert_equal(chunked["assignments"], expected_assignments)
        assert_equal(direct["sat_eligible_mask"][:, :, :, 0], np.array([[[False, False], [True, True]]], dtype=bool))
        assert_equal(direct["sat_eligible_mask"][:, :, :, 1], np.array([[[True, True], [False, False]]], dtype=bool))

        for key in (
            "assignments",
            "sat_azimuth",
            "sat_elevation",
            "sat_alpha",
            "sat_beta",
            "sat_belt_id",
            "cone_ok",
            "sat_beam_counts_demand",
            "sat_beam_counts_eligible",
            "sat_eligible_mask",
        ):
            if direct[key].dtype.kind == "f":
                assert_allclose(chunked[key], direct[key], atol=0.0, rtol=0.0, equal_nan=True)
            else:
                assert_equal(chunked[key], direct[key])

    def test_boresight_theta2_restricts_only_explicit_cell_ids(self):
        case = _boresight_selection_case()

        with pytest.raises(ValueError, match="boresight_theta2_cell_ids"):
            satsim.select_satellite_links(
                case["sat_topo"],
                min_elevation_deg=0.0,
                n_links=1,
                n_beam=2,
                strategy="max_elevation",
                sat_azel=case["sat_azel"],
                ras_topo=case["ras_topo"],
                beta_max_deg_per_sat=case["sat_beta_max"],
                sat_belt_id_per_sat=case["sat_belt_id"],
                boresight_pointing_azimuth_deg=case["pointing_az_deg"],
                boresight_pointing_elevation_deg=case["pointing_el_deg"],
                boresight_theta2_deg=5.0,
            )

        result = satsim.select_satellite_links(
            case["sat_topo"],
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_azel=case["sat_azel"],
            ras_topo=case["ras_topo"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            sat_belt_id_per_sat=case["sat_belt_id"],
            include_eligible_mask=True,
            prefer_numba=False,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=None,
            boresight_theta2_deg=5.0,
            boresight_theta2_cell_ids=np.array([0], dtype=np.int32),
        )

        expected_assignments = np.array([[[[1], [0]], [[0], [0]]]], dtype=np.int32)
        assert_equal(result["assignments"], expected_assignments)
        assert_equal(result["sat_eligible_mask"][0, 0, 0], np.array([False, True], dtype=bool))
        assert_equal(result["sat_eligible_mask"][0, 0, 1], np.array([True, True], dtype=bool))
        assert_equal(result["sat_eligible_mask"][0, 1, 0], np.array([True, False], dtype=bool))
        assert_equal(result["sat_eligible_mask"][0, 1, 1], np.array([True, True], dtype=bool))

    def test_boresight_none_none_matches_legacy_behavior(self):
        case = _boresight_selection_case()
        legacy = satsim.select_satellite_links(
            case["sat_topo"],
            min_elevation_deg=0.0,
            n_links=1,
            strategy="max_elevation",
            sat_azel=case["sat_azel"],
            ras_topo=case["ras_topo"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            sat_belt_id_per_sat=case["sat_belt_id"],
            include_counts=True,
            include_payload=True,
            prefer_numba=False,
        )
        boresight_off = satsim.select_satellite_links(
            case["sat_topo"],
            min_elevation_deg=0.0,
            n_links=1,
            strategy="max_elevation",
            sat_azel=case["sat_azel"],
            ras_topo=case["ras_topo"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            sat_belt_id_per_sat=case["sat_belt_id"],
            include_counts=True,
            include_payload=True,
            prefer_numba=False,
            boresight_theta1_deg=None,
            boresight_theta2_deg=None,
        )

        for key in legacy:
            if legacy[key].dtype.kind == "f":
                assert_allclose(boresight_off[key], legacy[key], atol=0.0, rtol=0.0, equal_nan=True)
            else:
                assert_equal(boresight_off[key], legacy[key])


def test_sample_conditioned_candidate_batch_cpu_is_seeded_and_source_specific():
    sampler = _load_step2_sampler()
    group_id = int(
        np.flatnonzero(
            (np.asarray(sampler.group_raw_counts) >= int(sampler.cond_min_group_samples))
            & (np.asarray(sampler.group_ptr[1:]) > np.asarray(sampler.group_ptr[:-1]))
        )[0]
    )
    source_kind = np.array(
        [
            satsim.SAMPLER_SOURCE_GROUP,
            satsim.SAMPLER_SOURCE_BELT,
            satsim.SAMPLER_SOURCE_GLOBAL,
        ],
        dtype=np.int8,
    )
    source_id = np.array([group_id, 0, 0], dtype=np.int32)

    first = satsim.sample_conditioned_candidate_batch_cpu(
        sampler,
        source_kind,
        source_id,
        chunk_size=16,
        rng=np.random.default_rng(123),
    )
    second = satsim.sample_conditioned_candidate_batch_cpu(
        sampler,
        source_kind,
        source_id,
        chunk_size=16,
        rng=np.random.default_rng(123),
    )

    for key in ("alpha_rad", "beta_rad"):
        assert_allclose(first[key], second[key], atol=0.0, rtol=0.0, equal_nan=True)

    group_p0 = int(sampler.group_ptr[group_id])
    group_p1 = int(sampler.group_ptr[group_id + 1])
    belt_p0 = int(sampler.belt_ptr[0])
    belt_p1 = int(sampler.belt_ptr[1])
    group_beta_pool = np.asarray(sampler.group_beta_pool[group_p0:group_p1], dtype=np.float32)
    group_alpha_pool = np.asarray(sampler.group_alpha_pool[group_p0:group_p1], dtype=np.float32)
    belt_beta_pool = np.asarray(sampler.belt_beta_pool[belt_p0:belt_p1], dtype=np.float32)
    belt_alpha_pool = np.asarray(sampler.belt_alpha_pool[belt_p0:belt_p1], dtype=np.float32)
    assert all(np.any(np.isclose(val, group_beta_pool, atol=1e-4)) for val in np.rad2deg(first["beta_rad"][0]))
    assert all(np.any(np.isclose(val, group_alpha_pool, atol=1e-4)) for val in np.rad2deg(first["alpha_rad"][0]))
    assert all(np.any(np.isclose(val, belt_beta_pool, atol=1e-4)) for val in np.rad2deg(first["beta_rad"][1]))
    assert all(np.any(np.isclose(val, belt_alpha_pool, atol=1e-4)) for val in np.rad2deg(first["alpha_rad"][1]))
    assert np.all(np.isfinite(first["alpha_rad"][2]))
    assert np.all(np.isfinite(first["beta_rad"][2]))


def test_fill_conditioned_beams_streaming_cpu_accepts_co_and_rejects_too_close(monkeypatch):
    source_kind = np.array([[satsim.SAMPLER_SOURCE_GROUP, satsim.SAMPLER_SOURCE_GROUP]], dtype=np.int8)
    source_id = np.array([[1, 2]], dtype=np.int32)
    vis_mask = np.array([[True, True]], dtype=bool)
    is_co_sat = np.array([[True, False]], dtype=bool)
    alpha0_rad = np.array([[0.10, 0.20]], dtype=np.float32)
    beta0_rad = np.array([[0.10, 0.10]], dtype=np.float32)
    beta_max = np.array([0.50, 0.50], dtype=np.float32)
    cos_min_sep = float(np.cos(np.deg2rad(10.0)))

    batches = [
        {
            "alpha_rad": np.array([[0.12, 1.20], [0.40, 1.60]], dtype=np.float32),
            "beta_rad": np.array([[0.11, 0.20], [0.12, 0.30]], dtype=np.float32),
        }
    ]

    def _fake_batch(*args, **kwargs):
        return batches.pop(0)

    monkeypatch.setattr(satsim, "sample_conditioned_candidate_batch_cpu", _fake_batch)

    result = satsim.fill_conditioned_beams_streaming_cpu(
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

    assert_equal(result["beam_valid"], np.array([[2, 2]], dtype=np.int16))
    assert result["beam_idx"][0, 0, 0] == -2
    assert_allclose(result["beam_alpha_rad"][0, 0, 0], alpha0_rad[0, 0], atol=0.0, rtol=0.0)
    assert_allclose(result["beam_alpha_rad"][0, 0, 1], np.float32(1.20), atol=0.0, rtol=0.0)
    assert_allclose(result["beam_alpha_rad"][0, 1, :], np.array([0.40, 1.60], dtype=np.float32), atol=0.0, rtol=0.0)


def test_fill_conditioned_beams_streaming_cpu_exclude_ras_radius_rejects_inside_zone(monkeypatch):
    source_kind = np.array([[satsim.SAMPLER_SOURCE_GROUP]], dtype=np.int8)
    source_id = np.array([[1]], dtype=np.int32)
    vis_mask = np.array([[True]], dtype=bool)
    is_co_sat = np.array([[True]], dtype=bool)
    alpha0_rad = np.array([[0.0]], dtype=np.float32)
    beta0_rad = np.array([[0.10]], dtype=np.float32)
    beta_max = np.array([0.50], dtype=np.float32)
    orbit_radius = np.array([6_378_136.6 + 525_000.0], dtype=np.float32)

    def _fake_batch(*args, **kwargs):
        return {
            "alpha_rad": np.array([[0.0, 1.2]], dtype=np.float32),
            "beta_rad": np.array([[0.10, 0.10]], dtype=np.float32),
        }

    monkeypatch.setattr(satsim, "sample_conditioned_candidate_batch_cpu", _fake_batch)

    result = satsim.fill_conditioned_beams_streaming_cpu(
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
        ras_sat_azel=np.array([[[0.0, np.rad2deg(0.10), 0.0]]], dtype=np.float32),
        orbit_radius_m_per_sat=orbit_radius,
        ras_exclusion_radius_km=50.0,
    )

    assert_equal(result["beam_valid"], np.array([[1]], dtype=np.int16))
    assert int(result["beam_idx"][0, 0, 0]) != -2
    assert_allclose(result["beam_alpha_rad"][0, 0, 0], np.float32(1.2), atol=0.0, rtol=0.0)


def test_pure_reroute_service_curve_exact_small_mask():
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
    result = satsim.pure_reroute_service_curve(
        eligible_mask,
        nco=1,
        beam_caps=np.arange(3, dtype=np.int32),
    )

    assert_equal(result["eligible_demand"], np.array([3], dtype=np.int32))
    assert_equal(result["matched_links"], np.array([[0, 2, 3]], dtype=np.int32))
    assert_equal(result["required_beam_cap"], np.array([2], dtype=np.int32))
    assert_allclose(result["delta"], np.array([1.0, 1.0 / 3.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)
    assert_allclose(result["eps"], np.array([1.0, 1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)
    assert_allclose(result["tail_risk"], result["eps"], atol=0.0, rtol=0.0)


def test_pure_reroute_service_curve_fills_tail_after_first_full_service_cap():
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

    result = satsim.pure_reroute_service_curve(
        eligible_mask,
        nco=1,
        beam_caps=beam_caps,
    )

    assert_equal(result["eligible_demand"], np.array([3], dtype=np.int32))
    assert_equal(result["matched_links"], np.array([[0, 2, 3, 3, 3, 3, 3, 3]], dtype=np.int32))
    assert_equal(result["required_beam_cap"], np.array([2], dtype=np.int32))
    assert_allclose(
        result["delta"],
        np.array([1.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert_allclose(
        result["eps"],
        np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )


def test_pure_reroute_service_curve_supports_nco_gt_one_and_contiguous_grid_limit():
    eligible_mask = np.array(
        [
            [
                [True, True, True],
                [False, True, True],
            ]
        ],
        dtype=bool,
    )
    result = satsim.pure_reroute_service_curve(
        eligible_mask,
        nco=2,
        beam_caps=np.arange(2, dtype=np.int32),
    )

    assert_equal(result["eligible_demand"], np.array([4], dtype=np.int32))
    assert_equal(result["matched_links"], np.array([[0, 3]], dtype=np.int32))
    assert_equal(result["required_beam_cap"], np.array([2], dtype=np.int32))
    assert_allclose(result["delta"], np.array([1.0, 0.25], dtype=np.float64), atol=0.0, rtol=0.0)
    assert_allclose(result["eps"], np.array([1.0, 1.0], dtype=np.float64), atol=0.0, rtol=0.0)
    assert int(result["draws_attempted"]) == 4
    assert int(result["rounds_used"]) == 1


def test_pure_reroute_service_curve_csr_input_matches_dense_exact_solver():
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
    dense = satsim.pure_reroute_service_curve(
        eligible_mask,
        nco=2,
        beam_caps=beam_caps,
    )
    csr_payload = satsim._pure_reroute_dense_mask_to_csr_payload(eligible_mask)
    sparse = satsim.pure_reroute_service_curve(
        csr_payload,
        nco=2,
        beam_caps=beam_caps,
    )

    for key, dense_value in dense.items():
        assert_allclose(sparse[key], dense_value, atol=0.0, rtol=0.0)


def test_select_satellite_links_can_return_eligible_mask_in_csr_and_both_modes():
    case = _boresight_selection_case()
    kwargs = dict(
        min_elevation_deg=0.0,
        n_links=1,
        n_beam=2,
        strategy="max_elevation",
        sat_azel=case["sat_azel"],
        ras_topo=case["ras_topo"],
        beta_max_deg_per_sat=case["sat_beta_max"],
        sat_belt_id_per_sat=case["sat_belt_id"],
        include_counts=False,
        include_payload=False,
        include_eligible_mask=True,
        prefer_numba=False,
    )
    dense = satsim.select_satellite_links(case["sat_topo"], **kwargs)
    csr_only = satsim.select_satellite_links(
        case["sat_topo"],
        eligible_mask_encoding="csr",
        **kwargs,
    )
    both = satsim.select_satellite_links(
        case["sat_topo"],
        eligible_mask_encoding="both",
        **kwargs,
    )
    expected_csr = satsim._pure_reroute_dense_mask_to_csr_payload(
        np.asarray(dense["sat_eligible_mask"], dtype=np.bool_, copy=False)
    )

    assert "sat_eligible_mask" not in csr_only
    for key, expected_value in expected_csr.items():
        if isinstance(expected_value, np.ndarray):
            assert_equal(csr_only[key], expected_value)
            assert_equal(both[key], expected_value)
        else:
            assert both[key] == expected_value
            assert csr_only[key] == expected_value
    assert_equal(both["sat_eligible_mask"], dense["sat_eligible_mask"])


def test_satellite_link_selection_library_can_emit_sparse_eligible_mask_payload():
    case = _boresight_selection_case()
    library = satsim.SatelliteLinkSelectionLibrary(
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
        include_counts=False,
        include_payload=False,
        include_eligible_mask=True,
        eligible_mask_encoding="csr",
    )
    library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
    result = library.finalize()
    dense = satsim.select_satellite_links(
        case["sat_topo"],
        min_elevation_deg=0.0,
        n_links=1,
        n_beam=2,
        strategy="max_elevation",
        sat_azel=case["sat_azel"],
        ras_topo=case["ras_topo"],
        beta_max_deg_per_sat=case["sat_beta_max"],
        sat_belt_id_per_sat=case["sat_belt_id"],
        include_counts=False,
        include_payload=False,
        include_eligible_mask=True,
        prefer_numba=False,
    )
    expected_csr = satsim._pure_reroute_dense_mask_to_csr_payload(
        np.asarray(dense["sat_eligible_mask"], dtype=np.bool_, copy=False)
    )

    assert "sat_eligible_mask" not in result
    for key, expected_value in expected_csr.items():
        if isinstance(expected_value, np.ndarray):
            assert_equal(result[key], expected_value)
        else:
            assert result[key] == expected_value


def test_fill_conditioned_beams_streaming_cpu_reports_rescue_failure(monkeypatch):
    def _nan_batch(*args, chunk_size, **kwargs):
        rows = int(np.asarray(args[1]).reshape(-1).size)
        return {
            "alpha_rad": np.full((rows, int(chunk_size)), np.nan, dtype=np.float32),
            "beta_rad": np.full((rows, int(chunk_size)), np.nan, dtype=np.float32),
        }

    monkeypatch.setattr(satsim, "sample_conditioned_candidate_batch_cpu", _nan_batch)

    with pytest.raises(RuntimeError, match=r"1 visible rows.*after 5 candidate draws"):
        satsim.fill_conditioned_beams_streaming_cpu(
            sampler=_load_step2_sampler(),
            source_kind=np.array([[satsim.SAMPLER_SOURCE_GROUP]], dtype=np.int8),
            source_id=np.array([[1]], dtype=np.int32),
            vis_mask_horizon=np.array([[True]], dtype=bool),
            is_co_sat=np.array([[False]], dtype=bool),
            alpha0_rad=np.array([[0.0]], dtype=np.float32),
            beta0_rad=np.array([[0.0]], dtype=np.float32),
            beta_max_rad_per_sat=np.array([0.5], dtype=np.float32),
            n_beams=1,
            cos_min_sep=float(np.cos(np.deg2rad(5.0))),
            seed=123,
            chunk_size=2,
            max_rounds=1,
            rescue_chunk_size=3,
            max_rescue_rounds=1,
        )


def test_fill_conditioned_beams_streaming_cpu_reaches_nbeam_on_reduced_b525_step2_case():
    case = _build_b525_step2_cpu_case()
    source_result = satsim.resolve_conditioned_sampler_sources_cpu(
        case["sampler"],
        case["sat_topo"][:, 0, :, 0].astype(np.float32, copy=False),
        case["sat_topo"][:, 0, :, 1].astype(np.float32, copy=False),
        case["sat_belt_id_rows"],
    )
    invalid_visible = case["vis_mask_horizon"] & (~np.asarray(source_result["valid_mask"], dtype=bool))
    assert not np.any(invalid_visible)

    result = satsim.fill_conditioned_beams_streaming_cpu(
        case["sampler"],
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
    )

    assert np.all(result["beam_valid"][case["vis_mask_horizon"]] == 7)
    assert int(result["unfinished_row_count"]) == 0
