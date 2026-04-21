#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib.machinery
import importlib.util
import sys
import types

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_equal
from pycraf import conversions as cnv


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

    numba_stub.njit = _njit
    numba_stub.prange = range
    numba_stub.cuda = None
    numba_stub.set_num_threads = lambda n: None
    numba_stub.get_num_threads = lambda: 1
    sys.modules["numba"] = numba_stub


_install_numba_stub()

from scepter import earthgrid
from scepter.antenna import (
    build_satellite_pattern_spec,
    calculate_beamwidth_1d,
    calculate_beamwidth_2d,
    pattern_wavelength_cm_from_frequency_mhz,
    resolve_pattern_wavelength_cm,
    s_1528_rec1_4_pattern_amend,
)


def _prepared_reuse_grid(*, radius: int, spacing_km: float = 80.0) -> dict[str, object]:
    q_coords: list[int] = []
    r_coords: list[int] = []
    for q_coord in range(-radius, radius + 1):
        r_min = max(-radius, -q_coord - radius)
        r_max = min(radius, -q_coord + radius)
        for r_coord in range(r_min, r_max + 1):
            q_coords.append(int(q_coord))
            r_coords.append(int(r_coord))
    q_arr = np.asarray(q_coords, dtype=np.int32)
    r_arr = np.asarray(r_coords, dtype=np.int32)
    sqrt3 = float(np.sqrt(3.0))
    east_km = float(spacing_km) * (q_arr.astype(np.float64) + 0.5 * r_arr.astype(np.float64))
    north_km = float(spacing_km) * (sqrt3 / 2.0) * r_arr.astype(np.float64)
    km_to_deg = (180.0 / np.pi) / float(R_earth.to_value(u.km))
    lon_deg = east_km * km_to_deg
    lat_deg = north_km * km_to_deg
    center_index = int(np.nonzero((q_arr == 0) & (r_arr == 0))[0][0])
    return {
        "pre_ras_cell_longitudes": lon_deg * u.deg,
        "pre_ras_cell_latitudes": lat_deg * u.deg,
        "pre_ras_to_active": np.arange(q_arr.size, dtype=np.int32),
        "ras_service_cell_index_pre_ras": center_index,
        "ras_service_cell_index": center_index,
        "point_spacing_km": float(spacing_km),
        "station_lon": 0.0 * u.deg,
        "station_lat": 0.0 * u.deg,
    }


class TestMaskHexgridForConstellation:

    def test_single_belt_matches_direct_mask_and_latitude_limit(self):
        grid_longitudes = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float64) * u.deg
        grid_latitudes = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64) * u.deg
        altitude = 600.0 * u.km
        min_elevation = 25.0 * u.deg
        inclination = 35.0 * u.deg
        station_lat = 0.0 * u.deg
        station_lon = 0.0 * u.deg

        mask_info = earthgrid.mask_hexgrid_for_constellation(
            grid_longitudes,
            grid_latitudes,
            altitudes=u.Quantity([altitude]),
            min_elevations=u.Quantity([min_elevation]),
            inclinations=u.Quantity([inclination]),
            station_lat=station_lat,
            station_lon=station_lon,
        )

        expected_base_mask = earthgrid.trunc_hexgrid_to_impactful(
            grid_longitudes,
            grid_latitudes,
            altitude,
            min_elevation,
            station_lat,
            station_lon,
        )

        r_km = float(R_earth.to_value(u.km))
        rs_km = r_km + float(altitude.to_value(u.km))
        min_elev_rad = float(min_elevation.to_value(u.rad))
        inclination_rad = float(inclination.to_value(u.rad))
        beta_b = np.arcsin(np.clip((r_km / rs_km) * np.cos(min_elev_rad), -1.0, 1.0))
        gamma_max = 0.5 * np.pi + min_elev_rad - beta_b
        gamma_horizon = np.arccos(np.clip(r_km / rs_km, -1.0, 1.0))
        gamma_max = float(np.clip(gamma_max, 0.0, gamma_horizon))
        expected_phi_limit = min(inclination_rad + gamma_max, 0.5 * np.pi) * u.rad
        expected_lat_mask = np.abs(grid_latitudes) <= expected_phi_limit.to(u.deg)

        assert_equal(mask_info["base_mask"], expected_base_mask)
        assert_equal(mask_info["lat_mask"], expected_lat_mask)
        assert_equal(mask_info["combined_mask"], expected_base_mask & expected_lat_mask)
        assert_quantity_allclose(mask_info["phi_max_per_belt"], np.array([expected_phi_limit.to_value(u.deg)]) * u.deg)
        assert_quantity_allclose(mask_info["phi_limit"], expected_phi_limit.to(u.deg))

    def test_multi_belt_any_policy_is_union_and_tighter_than_conservative_envelope(self):
        grid_longitudes = np.arange(0.0, 91.0, 5.0, dtype=np.float64) * u.deg
        grid_latitudes = np.zeros(grid_longitudes.size, dtype=np.float64) * u.deg
        altitudes = np.array([1200.0, 500.0], dtype=np.float64) * u.km
        min_elevations = np.array([40.0, 10.0], dtype=np.float64) * u.deg
        inclinations = np.array([20.0, 60.0], dtype=np.float64) * u.deg
        station_lat = 0.0 * u.deg
        station_lon = 0.0 * u.deg

        mask_info = earthgrid.mask_hexgrid_for_constellation(
            grid_longitudes,
            grid_latitudes,
            altitudes=altitudes,
            min_elevations=min_elevations,
            inclinations=inclinations,
            station_lat=station_lat,
            station_lon=station_lon,
            latitude_policy="any",
        )

        direct_union = (
            earthgrid.trunc_hexgrid_to_impactful(
                grid_longitudes,
                grid_latitudes,
                altitudes[0],
                min_elevations[0],
                station_lat,
                station_lon,
            )
            | earthgrid.trunc_hexgrid_to_impactful(
                grid_longitudes,
                grid_latitudes,
                altitudes[1],
                min_elevations[1],
                station_lat,
                station_lon,
            )
        )
        conservative_mask = earthgrid.trunc_hexgrid_to_impactful(
            grid_longitudes,
            grid_latitudes,
            altitudes.max(),
            min_elevations.min(),
            station_lat,
            station_lon,
        )

        assert_equal(mask_info["base_mask"], direct_union)
        assert_equal(mask_info["combined_mask"], direct_union)
        assert np.all((~mask_info["base_mask"]) | conservative_mask)
        assert np.any(conservative_mask & ~mask_info["base_mask"])

    def test_geography_modes_classify_land_and_nearshore_sea(self):
        grid_longitudes = np.array([21.4436, 18.0, 17.8, 15.0, 21.5], dtype=np.float64) * u.deg
        grid_latitudes = np.array([-30.7128, -34.0, -34.2, -34.5, -40.0], dtype=np.float64) * u.deg
        station_lat = -30.7128 * u.deg
        station_lon = 21.4436 * u.deg
        common_kwargs = dict(
            altitudes=np.array([525.0], dtype=np.float64) * u.km,
            min_elevations=np.array([5.0], dtype=np.float64) * u.deg,
            inclinations=np.array([90.0], dtype=np.float64) * u.deg,
            station_lat=station_lat,
            station_lon=station_lon,
        )

        land_only = earthgrid.mask_hexgrid_for_constellation(
            grid_longitudes,
            grid_latitudes,
            geography_mask_mode="land_only",
            **common_kwargs,
        )
        nearshore = earthgrid.mask_hexgrid_for_constellation(
            grid_longitudes,
            grid_latitudes,
            geography_mask_mode="land_plus_nearshore_sea",
            shoreline_buffer_km=80.0,
            **common_kwargs,
        )

        assert_equal(land_only["combined_mask"], np.array([True, False, False, False, False], dtype=bool))
        assert_equal(nearshore["combined_mask"], np.array([True, True, True, False, False], dtype=bool))
        assert_equal(nearshore["land_mask"], np.array([True, False, False, False, False], dtype=bool))
        assert_equal(nearshore["nearshore_sea_mask"], np.array([False, True, True, False, False], dtype=bool))
        np.testing.assert_allclose(
            np.asarray(nearshore["shore_distance_km"], dtype=np.float64),
            np.array([np.nan, 30.30, 49.56, 319.61, 593.99], dtype=np.float64),
            atol=1.0,
            rtol=0.0,
            equal_nan=True,
        )

    def test_geography_mode_alias_normalises_shoreline_buffer_label(self):
        assert earthgrid._normalise_geography_mask_mode("land_plus_shoreline_buffer") == "land_plus_nearshore_sea"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                dict(
                    geography_mask_mode="land_plus_nearshore_sea",
                    shoreline_buffer_km=None,
                ),
                "shoreline_buffer_km is required",
            ),
        ],
    )
    def test_geography_invalid_inputs_raise_clear_errors(self, kwargs, message):
        with pytest.raises((ValueError, RuntimeError), match=message):
            earthgrid.mask_hexgrid_for_constellation(
                np.array([21.4436], dtype=np.float64) * u.deg,
                np.array([-30.7128], dtype=np.float64) * u.deg,
                altitudes=np.array([525.0], dtype=np.float64) * u.km,
                min_elevations=np.array([5.0], dtype=np.float64) * u.deg,
                inclinations=np.array([90.0], dtype=np.float64) * u.deg,
                station_lat=-30.7128 * u.deg,
                station_lon=21.4436 * u.deg,
                **kwargs,
            )

    def test_cartopy_backend_selection_path_is_clear(self):
        kwargs = dict(
            grid_longitudes=np.array([21.4436], dtype=np.float64) * u.deg,
            grid_latitudes=np.array([-30.7128], dtype=np.float64) * u.deg,
            altitudes=np.array([525.0], dtype=np.float64) * u.km,
            min_elevations=np.array([5.0], dtype=np.float64) * u.deg,
            inclinations=np.array([90.0], dtype=np.float64) * u.deg,
            station_lat=-30.7128 * u.deg,
            station_lon=21.4436 * u.deg,
            geography_mask_mode="land_only",
            coastline_backend="cartopy",
        )

        if importlib.util.find_spec("cartopy") is None:
            with pytest.raises(RuntimeError, match="cartopy is required"):
                earthgrid.mask_hexgrid_for_constellation(**kwargs)
            return

        result = earthgrid.mask_hexgrid_for_constellation(**kwargs)
        assert_equal(result["combined_mask"], np.array([True], dtype=bool))

    def test_land_only_skips_shoreline_distance(self, monkeypatch: pytest.MonkeyPatch):
        earthgrid._GEOGRAPHY_CLASSIFICATION_CACHE.clear()
        calls: dict[str, int] = {"distance": 0}

        class _FakeShapely:
            @staticmethod
            def points(x, y):
                return np.column_stack((np.asarray(x), np.asarray(y)))

            @staticmethod
            def covered_by(points, land_union):
                del land_union
                return np.asarray([True, False], dtype=bool)

            @staticmethod
            def distance(points, coastline_union):
                del points, coastline_union
                calls["distance"] += 1
                return np.asarray([3.0, 4.0], dtype=np.float64)

        monkeypatch.setattr(
            earthgrid,
            "_require_shapely",
            lambda purpose: (_FakeShapely, None, None),
        )
        monkeypatch.setattr(
            earthgrid,
            "_build_projected_geography_reference",
            lambda *args, **kwargs: (object(), object()),
        )

        result = earthgrid._classify_hexgrid_geography(
            np.asarray([21.0, 21.5], dtype=np.float64),
            np.asarray([-30.0, -30.5], dtype=np.float64),
            candidate_mask=np.asarray([True, True], dtype=bool),
            station_lon_deg=21.0,
            station_lat_deg=-30.0,
            geography_mask_mode="land_only",
            shoreline_buffer_km=None,
            coastline_backend="vendored",
        )

        assert_equal(result["land_mask"], np.asarray([True, False], dtype=bool))
        assert_equal(result["geography_mask"], np.asarray([True, False], dtype=bool))
        assert np.isnan(np.asarray(result["shore_distance_km"], dtype=np.float64)).all()
        assert calls["distance"] == 0

    def test_nearshore_distance_only_runs_for_non_land_candidates_and_uses_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        earthgrid._GEOGRAPHY_CLASSIFICATION_CACHE.clear()
        calls: list[int] = []

        class _FakeShapely:
            @staticmethod
            def points(x, y):
                return np.column_stack((np.asarray(x), np.asarray(y)))

            @staticmethod
            def covered_by(points, land_union):
                del points, land_union
                return np.asarray([True, False, False], dtype=bool)

            @staticmethod
            def distance(points, coastline_union):
                del coastline_union
                calls.append(int(np.asarray(points).shape[0]))
                return np.asarray([1.0, 5.0, 20.0], dtype=np.float64)

        monkeypatch.setattr(
            earthgrid,
            "_require_shapely",
            lambda purpose: (_FakeShapely, None, None),
        )
        monkeypatch.setattr(
            earthgrid,
            "_build_projected_geography_reference",
            lambda *args, **kwargs: (object(), object()),
        )

        kwargs = dict(
            cell_longitudes_deg=np.asarray([21.0, 21.5, 22.0], dtype=np.float64),
            cell_latitudes_deg=np.asarray([-30.0, -30.2, -30.4], dtype=np.float64),
            candidate_mask=np.asarray([True, True, True], dtype=bool),
            station_lon_deg=21.0,
            station_lat_deg=-30.0,
            geography_mask_mode="land_plus_nearshore_sea",
            shoreline_buffer_km=10.0,
            coastline_backend="vendored",
        )
        result_first = earthgrid._classify_hexgrid_geography(**kwargs)
        result_second = earthgrid._classify_hexgrid_geography(**kwargs)

        assert calls == [3]
        assert_equal(result_first["land_mask"], np.asarray([True, False, False], dtype=bool))
        assert_equal(result_first["nearshore_sea_mask"], np.asarray([False, True, False], dtype=bool))
        assert_equal(result_first["geography_mask"], np.asarray([True, True, False], dtype=bool))
        np.testing.assert_allclose(
            np.asarray(result_first["shore_distance_km"], dtype=np.float64),
            np.asarray([np.nan, 5.0, 20.0], dtype=np.float64),
            equal_nan=True,
        )
        assert_equal(result_first["geography_mask"], result_second["geography_mask"])

    def test_signed_nearshore_buffer_zero_and_negative_modes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        earthgrid._GEOGRAPHY_CLASSIFICATION_CACHE.clear()

        class _FakeShapely:
            @staticmethod
            def points(x, y):
                return np.column_stack((np.asarray(x), np.asarray(y)))

            @staticmethod
            def covered_by(points, land_union):
                del points, land_union
                return np.asarray([True, True, False], dtype=bool)

            @staticmethod
            def distance(points, coastline_union):
                del points, coastline_union
                return np.asarray([2.0, 9.0, 4.0], dtype=np.float64)

        monkeypatch.setattr(
            earthgrid,
            "_require_shapely",
            lambda purpose: (_FakeShapely, None, None),
        )
        monkeypatch.setattr(
            earthgrid,
            "_build_projected_geography_reference",
            lambda *args, **kwargs: (object(), object()),
        )

        common_kwargs = dict(
            cell_longitudes_deg=np.asarray([21.0, 21.5, 22.0], dtype=np.float64),
            cell_latitudes_deg=np.asarray([-30.0, -30.2, -30.4], dtype=np.float64),
            candidate_mask=np.asarray([True, True, True], dtype=bool),
            station_lon_deg=21.0,
            station_lat_deg=-30.0,
            geography_mask_mode="land_plus_nearshore_sea",
            coastline_backend="vendored",
        )
        zero_buffer = earthgrid._classify_hexgrid_geography(
            shoreline_buffer_km=0.0,
            **common_kwargs,
        )
        negative_buffer = earthgrid._classify_hexgrid_geography(
            shoreline_buffer_km=-5.0,
            **common_kwargs,
        )

        assert_equal(zero_buffer["geography_mask"], np.asarray([True, True, False], dtype=bool))
        assert_equal(zero_buffer["nearshore_sea_mask"], np.asarray([False, False, False], dtype=bool))
        assert_equal(negative_buffer["geography_mask"], np.asarray([False, True, False], dtype=bool))
        assert_equal(negative_buffer["nearshore_sea_mask"], np.asarray([False, False, False], dtype=bool))


def test_resolve_frequency_reuse_slots_f7_avoids_same_slot_immediate_neighbors() -> None:
    prepared_grid = _prepared_reuse_grid(radius=4, spacing_km=90.0)

    reuse_plan = earthgrid.resolve_frequency_reuse_slots(
        prepared_grid,
        reuse_factor=7,
        anchor_slot=0,
    )

    assert reuse_plan["adjacent_same_slot_pair_count"] == 0
    axial_q = np.asarray(reuse_plan["axial_q_pre_ras"], dtype=np.int32)
    axial_r = np.asarray(reuse_plan["axial_r_pre_ras"], dtype=np.int32)
    slot_ids = np.asarray(reuse_plan["pre_ras_slot_ids"], dtype=np.int32)
    coord_to_index = {
        (int(q_coord), int(r_coord)): int(index)
        for index, (q_coord, r_coord) in enumerate(zip(axial_q.tolist(), axial_r.tolist()))
    }
    for index, (q_coord, r_coord) in enumerate(zip(axial_q.tolist(), axial_r.tolist())):
        for dq, dr in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            neighbor_index = coord_to_index.get((int(q_coord + dq), int(r_coord + dr)))
            if neighbor_index is None:
                continue
            assert int(slot_ids[index]) != int(slot_ids[neighbor_index])


def test_resolve_frequency_reuse_slots_maps_active_slots_by_active_index_order() -> None:
    prepared_grid = _prepared_reuse_grid(radius=4, spacing_km=90.0)
    pre_ras_to_active = np.arange(
        int(u.Quantity(prepared_grid["pre_ras_cell_longitudes"], copy=False).size) - 1,
        -1,
        -1,
        dtype=np.int32,
    )
    active_order = np.argsort(pre_ras_to_active)
    prepared_grid["pre_ras_to_active"] = pre_ras_to_active
    prepared_grid["active_grid_longitudes"] = u.Quantity(prepared_grid["pre_ras_cell_longitudes"], copy=False)[active_order]
    prepared_grid["active_grid_latitudes"] = u.Quantity(prepared_grid["pre_ras_cell_latitudes"], copy=False)[active_order]
    prepared_grid["ras_service_cell_index"] = int(pre_ras_to_active[int(prepared_grid["ras_service_cell_index_pre_ras"])])

    reuse_plan = earthgrid.resolve_frequency_reuse_slots(
        prepared_grid,
        reuse_factor=7,
        anchor_slot=0,
    )

    pre_ras_slot_ids = np.asarray(reuse_plan["pre_ras_slot_ids"], dtype=np.int32)
    active_slot_ids = np.asarray(reuse_plan["active_slot_ids"], dtype=np.int32)
    for pre_ras_index, active_index in enumerate(pre_ras_to_active.tolist()):
        assert int(active_slot_ids[active_index]) == int(pre_ras_slot_ids[pre_ras_index])
    assert reuse_plan["active_adjacent_same_slot_pair_count"] == 0


def test_resolve_frequency_reuse_slots_minimises_conflicts_after_active_masking() -> None:
    prepared_grid = _prepared_reuse_grid(radius=4, spacing_km=90.0)
    pre_ras_count = int(u.Quantity(prepared_grid["pre_ras_cell_longitudes"], copy=False).size)
    active_mask = np.ones(pre_ras_count, dtype=bool)
    active_mask[::6] = False
    active_mask[5::11] = False
    active_mask[int(prepared_grid["ras_service_cell_index_pre_ras"])] = True
    pre_ras_to_active = np.full(pre_ras_count, -1, dtype=np.int32)
    active_indices = np.flatnonzero(active_mask)[::-1]
    pre_ras_to_active[active_indices] = np.arange(int(active_indices.size), dtype=np.int32)
    prepared_grid["pre_ras_to_active"] = pre_ras_to_active
    prepared_grid["active_grid_longitudes"] = u.Quantity(prepared_grid["pre_ras_cell_longitudes"], copy=False)[active_indices]
    prepared_grid["active_grid_latitudes"] = u.Quantity(prepared_grid["pre_ras_cell_latitudes"], copy=False)[active_indices]
    prepared_grid["ras_service_cell_index"] = int(pre_ras_to_active[int(prepared_grid["ras_service_cell_index_pre_ras"])])

    reuse_plan = earthgrid.resolve_frequency_reuse_slots(
        prepared_grid,
        reuse_factor=7,
        anchor_slot=0,
    )

    assert reuse_plan["active_adjacent_same_slot_pair_count"] == 0
    assert reuse_plan["adjacent_same_slot_pair_count"] == reuse_plan["active_adjacent_same_slot_pair_count"]
    active_axial_q = np.asarray(reuse_plan["axial_q_active"], dtype=np.int32)
    active_axial_r = np.asarray(reuse_plan["axial_r_active"], dtype=np.int32)
    active_slot_ids = np.asarray(reuse_plan["active_slot_ids"], dtype=np.int32)
    valid_mask = active_slot_ids >= 0
    coord_to_index = {
        (int(q_coord), int(r_coord)): int(index)
        for index, (q_coord, r_coord) in enumerate(
            zip(active_axial_q[valid_mask].tolist(), active_axial_r[valid_mask].tolist())
        )
    }
    compact_slots = active_slot_ids[valid_mask]
    compact_q = active_axial_q[valid_mask]
    compact_r = active_axial_r[valid_mask]
    for index, (q_coord, r_coord) in enumerate(zip(compact_q.tolist(), compact_r.tolist())):
        for dq, dr in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            neighbor_index = coord_to_index.get((int(q_coord + dq), int(r_coord + dr)))
            if neighbor_index is None:
                continue
            assert int(compact_slots[index]) != int(compact_slots[neighbor_index])


def test_resolve_frequency_reuse_slots_f7_avoids_same_slot_neighbors_on_active_projected_grid() -> None:
    prepared_grid = _prepared_reuse_grid(radius=6, spacing_km=90.0)
    pre_ras_count = int(u.Quantity(prepared_grid["pre_ras_cell_longitudes"], copy=False).size)
    lon_deg = np.asarray(
        u.Quantity(prepared_grid["pre_ras_cell_longitudes"], copy=False).to_value(u.deg),
        dtype=np.float64,
    )
    lat_deg = np.asarray(
        u.Quantity(prepared_grid["pre_ras_cell_latitudes"], copy=False).to_value(u.deg),
        dtype=np.float64,
    )
    active_mask = np.ones(pre_ras_count, dtype=bool)
    active_mask[::5] = False
    active_mask[3::8] = False
    active_mask &= ~((lon_deg > 0.02) & (lat_deg > 0.02))
    active_mask &= ~((lon_deg < -0.03) & (lat_deg > 0.03))
    active_mask[int(prepared_grid["ras_service_cell_index_pre_ras"])] = True
    pre_ras_to_active = np.full(pre_ras_count, -1, dtype=np.int32)
    active_indices = np.flatnonzero(active_mask)[::-1]
    pre_ras_to_active[active_indices] = np.arange(int(active_indices.size), dtype=np.int32)
    prepared_grid["pre_ras_to_active"] = pre_ras_to_active
    prepared_grid["active_grid_longitudes"] = u.Quantity(
        prepared_grid["pre_ras_cell_longitudes"],
        copy=False,
    )[active_indices]
    prepared_grid["active_grid_latitudes"] = u.Quantity(
        prepared_grid["pre_ras_cell_latitudes"],
        copy=False,
    )[active_indices]
    prepared_grid["ras_service_cell_index"] = int(
        pre_ras_to_active[int(prepared_grid["ras_service_cell_index_pre_ras"])]
    )

    reuse_plan = earthgrid.resolve_frequency_reuse_slots(
        prepared_grid,
        reuse_factor=7,
        anchor_slot=0,
    )

    assert reuse_plan["active_adjacent_same_slot_pair_count"] == 0
    active_lon_deg = np.asarray(
        u.Quantity(prepared_grid["active_grid_longitudes"], copy=False).to_value(u.deg),
        dtype=np.float64,
    )
    active_lat_deg = np.asarray(
        u.Quantity(prepared_grid["active_grid_latitudes"], copy=False).to_value(u.deg),
        dtype=np.float64,
    )
    active_x_km, active_y_km = earthgrid._local_tangent_plane_xy_km(
        active_lon_deg,
        active_lat_deg,
        ref_lon_deg=0.0,
        ref_lat_deg=0.0,
    )
    active_neighbors = earthgrid._build_reuse_neighbors_from_xy(
        active_x_km,
        active_y_km,
        point_spacing_km=float(prepared_grid["point_spacing_km"]),
    )
    active_slot_ids = np.asarray(reuse_plan["active_slot_ids"], dtype=np.int32)
    assert earthgrid._count_reuse_adjacency_conflicts_from_neighbors(active_neighbors, active_slot_ids) == 0


@pytest.mark.parametrize("reuse_factor", [3, 4, 7, 9, 12, 13, 16, 19])
def test_resolve_frequency_reuse_slots_supported_factors_avoid_immediate_neighbor_conflicts(
    reuse_factor: int,
) -> None:
    prepared_grid = _prepared_reuse_grid(radius=5, spacing_km=90.0)

    reuse_plan = earthgrid.resolve_frequency_reuse_slots(
        prepared_grid,
        reuse_factor=reuse_factor,
        anchor_slot=0,
    )

    assert reuse_plan["active_adjacent_same_slot_pair_count"] == 0
    assert reuse_plan["adjacent_same_slot_pair_count"] == 0


class TestResolveRasHexgridCellIds:

    def test_radius_mode_selects_cells_by_station_distance(self):
        cell_longitudes = np.array([0.0, 0.5, 1.0, 2.5], dtype=np.float64) * u.deg
        cell_latitudes = np.zeros(4, dtype=np.float64) * u.deg

        selected = earthgrid.resolve_ras_hexgrid_cell_ids(
            cell_longitudes,
            cell_latitudes,
            station_lat=0.0 * u.deg,
            station_lon=0.0 * u.deg,
            mode="radius_km",
            radius_km=120.0,
        )

        assert_equal(selected, np.array([0, 1, 2], dtype=np.int32))

    def test_adjacency_layers_mode_returns_expected_hex_rings(self):
        spacing = 1.0
        sqrt3 = np.sqrt(3.0)
        q_vals = np.array([0, 1, 0, -1, -1, 0, 1, 2], dtype=np.int32)
        r_vals = np.array([0, 0, 1, 1, 0, -1, -1, -1], dtype=np.int32)
        east = spacing * (q_vals + 0.5 * r_vals)
        north = spacing * ((sqrt3 / 2.0) * r_vals)
        km_to_deg = 1.0 / float(R_earth.to_value(u.km)) * (180.0 / np.pi)
        cell_longitudes = east * km_to_deg * u.deg
        cell_latitudes = north * km_to_deg * u.deg

        layer0 = earthgrid.resolve_ras_hexgrid_cell_ids(
            cell_longitudes,
            cell_latitudes,
            station_lat=0.0 * u.deg,
            station_lon=0.0 * u.deg,
            mode="adjacency_layers",
            ras_cell_index=0,
            layers=0,
        )
        layer1 = earthgrid.resolve_ras_hexgrid_cell_ids(
            cell_longitudes,
            cell_latitudes,
            station_lat=0.0 * u.deg,
            station_lon=0.0 * u.deg,
            mode="adjacency_layers",
            ras_cell_index=0,
            layers=1,
        )
        layer2 = earthgrid.resolve_ras_hexgrid_cell_ids(
            cell_longitudes,
            cell_latitudes,
            station_lat=0.0 * u.deg,
            station_lon=0.0 * u.deg,
            mode="adjacency_layers",
            ras_cell_index=0,
            layers=2,
        )

        assert_equal(layer0, np.array([0], dtype=np.int32))
        assert_equal(layer1, np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32))
        assert_equal(layer2, np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32))

    def test_adjacency_layers_matches_real_prefilter_hex_ring_counts(self):
        station_lat = -30.7128 * u.deg
        station_lon = 21.4436 * u.deg
        grid_lons, grid_lats, _ = earthgrid.generate_hexgrid_full(79.141 * u.km)
        mask_info = earthgrid.mask_hexgrid_for_constellation(
            grid_lons,
            grid_lats,
            altitudes=np.array([525.0], dtype=np.float64) * u.km,
            min_elevations=np.array([5.0], dtype=np.float64) * u.deg,
            inclinations=np.array([90.0], dtype=np.float64) * u.deg,
            station_lat=station_lat,
            station_lon=station_lon,
            latitude_policy="any",
            geography_mask_mode="none",
        )

        prefilter_mask = np.asarray(mask_info["base_mask"], dtype=bool) & np.asarray(
            mask_info["lat_mask"],
            dtype=bool,
        )
        prefilter_lons = grid_lons[prefilter_mask]
        prefilter_lats = grid_lats[prefilter_mask]
        ras_lon_deg = float(station_lon.to_value(u.deg))
        ras_lat_deg = float(station_lat.to_value(u.deg))
        dlon_deg = (prefilter_lons.to_value(u.deg) - ras_lon_deg) * np.cos(np.deg2rad(ras_lat_deg))
        dlat_deg = prefilter_lats.to_value(u.deg) - ras_lat_deg
        ras_cell_index = int(np.argmin(dlon_deg * dlon_deg + dlat_deg * dlat_deg))

        expected_counts = {0: 1, 1: 7, 2: 19, 3: 37, 4: 61, 5: 91}
        for layers, expected_count in expected_counts.items():
            selected = earthgrid.resolve_ras_hexgrid_cell_ids(
                prefilter_lons,
                prefilter_lats,
                station_lat=station_lat,
                station_lon=station_lon,
                mode="adjacency_layers",
                ras_cell_index=ras_cell_index,
                layers=layers,
            )
            assert_equal(selected.size, expected_count)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                dict(
                    mode="adjacency_layers",
                    ras_cell_index=None,
                    layers=1,
                ),
                "ras_cell_index is required",
            ),
            (
                dict(
                    mode="adjacency_layers",
                    ras_cell_index=0,
                    layers=-1,
                ),
                "layers must be non-negative",
            ),
            (
                dict(
                    mode="radius_km",
                    radius_km=None,
                ),
                "radius_km is required",
            ),
            (
                dict(
                    mode="radius_km",
                    radius_km=-1.0,
                ),
                "radius_km must be non-negative",
            ),
        ],
    )
    def test_invalid_ras_hexgrid_resolution_inputs_raise(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            earthgrid.resolve_ras_hexgrid_cell_ids(
                np.array([0.0, 1.0], dtype=np.float64) * u.deg,
                np.array([0.0, 0.0], dtype=np.float64) * u.deg,
                station_lat=0.0 * u.deg,
                station_lon=0.0 * u.deg,
                **kwargs,
            )


class TestBeamSpacingRule:

    _PATTERN_KW = dict(
        wavelength=15 * u.cm,
        Lt=1.6 * u.m,
        Lr=1.6 * u.m,
        l=2,
        SLR=20 * cnv.dB,
        far_sidelobe_start=90 * u.deg,
        far_sidelobe_level=-20 * cnv.dBi,
    )

    def test_full_footprint_spacing_matches_nadir_footprint_diameter(self):
        altitude = 525.0 * u.km
        theta_sep_edge = 0.5 * calculate_beamwidth_1d(
            s_1528_rec1_4_pattern_amend,
            level_drop=7.0 * cnv.dB,
            **self._PATTERN_KW,
        )

        spacing = earthgrid.ground_separation_from_beta(
            0.0 * u.deg,
            theta_sep_edge,
            altitude,
            beam_spacing_rule="full_footprint_diameter",
        )
        footprint = earthgrid.calculate_footprint_size(
            s_1528_rec1_4_pattern_amend,
            altitude=altitude,
            off_nadir_angle=0.0 * u.deg,
            theta=theta_sep_edge,
            **self._PATTERN_KW,
        )

        assert_quantity_allclose(spacing, footprint.to(u.km))

    def test_full_footprint_spacing_is_strictly_larger_than_center_to_contour(self):
        altitude = 525.0 * u.km
        theta_sep_edge = 0.5 * calculate_beamwidth_1d(
            s_1528_rec1_4_pattern_amend,
            level_drop=7.0 * cnv.dB,
            **self._PATTERN_KW,
        )
        beta = 15.0 * u.deg

        full_spacing = earthgrid.ground_separation_from_beta(
            beta,
            theta_sep_edge,
            altitude,
            beam_spacing_rule="full_footprint_diameter",
        )
        center_spacing = earthgrid.ground_separation_from_beta(
            beta,
            theta_sep_edge,
            altitude,
            beam_spacing_rule="center_to_contour",
        )

        assert full_spacing > center_spacing

    def test_recommend_cell_diameter_full_spacing_rule_matches_geometry_samples(self):
        altitude = 525.0 * u.km
        theta_sep_edge = 0.5 * calculate_beamwidth_1d(
            s_1528_rec1_4_pattern_amend,
            level_drop=7.0 * cnv.dB,
            **self._PATTERN_KW,
        )
        stats = earthgrid.recommend_cell_diameter(
            s_1528_rec1_4_pattern_amend,
            altitude=altitude,
            min_elevation=20.0 * u.deg,
            strategy="maximum_elevation",
            n_vis_override=4,
            vis_count_model="mean",
            spacing_drop=7.0 * cnv.dB,
            beam_spacing_rule="full_footprint_diameter",
            leading_metric="spacing_contour",
            n_samples=256,
            seed=123,
            return_samples=True,
            **self._PATTERN_KW,
        )

        beta_samples = np.asarray(stats["beta_samples_deg"], dtype=np.float64) * u.deg
        geom = earthgrid.ground_separation_from_beta(
            beta_samples,
            theta_sep_edge,
            altitude,
            beam_spacing_rule="full_footprint_diameter",
        ).to_value(u.km)

        assert stats["beam_spacing_rule"] == "full_footprint_diameter"
        assert_equal(
            np.isfinite(np.asarray(stats["spacing_samples_km"], dtype=np.float64)),
            np.isfinite(geom),
        )
        np.testing.assert_allclose(
            np.asarray(stats["spacing_samples_km"], dtype=np.float64),
            geom,
            atol=1e-6,
            rtol=1e-6,
        )


class TestStep1ContourHelpers:

    _PATTERN_KW = dict(
        wavelength=15 * u.cm,
        Lt=1.6 * u.m,
        Lr=1.6 * u.m,
        l=2,
        SLR=20 * cnv.dB,
        far_sidelobe_start=90 * u.deg,
        far_sidelobe_level=-20 * cnv.dBi,
    )

    @pytest.mark.parametrize(
        ("value", "expected_db"),
        [
            ("db3", 3.0),
            ("db7", 7.0),
            ("db15", 15.0),
            (5.0, 5.0),
            (20.0 * cnv.dB, 20.0),
        ],
    )
    def test_normalize_contour_drop_accepts_presets_floats_and_quantities(self, value, expected_db):
        result = earthgrid.normalize_contour_drop(value)

        assert result["drop_db"] == pytest.approx(expected_db)
        assert result["label"] == f"-{int(expected_db) if float(expected_db).is_integer() else expected_db} dB"

    @pytest.mark.parametrize("value", ["db11", 0.0, -5.0, -7.0 * cnv.dB])
    def test_normalize_contour_drop_rejects_invalid_values(self, value):
        with pytest.raises((TypeError, ValueError)):
            earthgrid.normalize_contour_drop(value)

    def test_recommend_cell_diameter_uses_generic_metric_names_only(self):
        with pytest.raises(ValueError, match="leading_metric must be one of"):
            earthgrid.recommend_cell_diameter(
                s_1528_rec1_4_pattern_amend,
                altitude=525.0 * u.km,
                min_elevation=20.0 * u.deg,
                wavelength=15 * u.cm,
                leading_metric="sep_15db",
                **{k: v for k, v in self._PATTERN_KW.items() if k != "wavelength"},
            )

    def test_summarize_contour_spacing_returns_selected_spacing_and_summary_lines(self):
        summary = earthgrid.summarize_contour_spacing(
            s_1528_rec1_4_pattern_amend,
            belt_names=["System3_Belt_1"],
            altitudes=np.array([525.0], dtype=np.float64) * u.km,
            min_elevations=np.array([20.0], dtype=np.float64) * u.deg,
            max_betas=np.array([35.0], dtype=np.float64) * u.deg,
            wavelength=15 * u.cm,
            strategy="max_elevation",
            indicative_footprint_drop="db3",
            spacing_drop="db7",
            leading_metric="spacing_contour",
            cell_spacing_rule="full_footprint_diameter",
            belt_satellite_counts=np.array([3360], dtype=np.int64),
            n_samples=256,
            seed=123,
            **{k: v for k, v in self._PATTERN_KW.items() if k != "wavelength"},
        )

        assert summary["selected_cell_spacing_km"] > 0.0
        assert summary["cell_spacing_km_per_belt"].shape == (1,)
        assert any("Selected cell_km" in line for line in summary["summary_lines"])

    def test_prepare_active_grid_matches_real_prefilter_and_ras_counts(self):
        prepared = earthgrid.prepare_active_grid(
            point_spacing=79.141 * u.km,
            altitudes=np.array([525.0], dtype=np.float64) * u.km,
            min_elevations=np.array([5.0], dtype=np.float64) * u.deg,
            inclinations=np.array([90.0], dtype=np.float64) * u.deg,
            station_lat=-30.7128 * u.deg,
            station_lon=21.4436 * u.deg,
            geography_mask_mode="none",
            ras_exclusion_mode="adjacency_layers",
            ras_exclusion_layers=4,
        )

        assert prepared["prefilter_cell_count"] > 0
        assert prepared["pre_ras_cell_count"] == prepared["prefilter_cell_count"]
        assert prepared["ras_hex_exclusion_requested_cell_count"] == 61
        assert prepared["ras_excluded_cell_count"] == 61
        assert prepared["active_cell_count"] == prepared["pre_ras_cell_count"] - 61

    def test_resolve_theta2_active_cell_ids_projects_prefilter_scope_to_active_axis(self):
        prepared_no_exclusion = earthgrid.prepare_active_grid(
            point_spacing=79.141 * u.km,
            altitudes=np.array([525.0], dtype=np.float64) * u.km,
            min_elevations=np.array([5.0], dtype=np.float64) * u.deg,
            inclinations=np.array([90.0], dtype=np.float64) * u.deg,
            station_lat=-30.7128 * u.deg,
            station_lon=21.4436 * u.deg,
            geography_mask_mode="none",
            ras_exclusion_mode="none",
        )
        prepared = earthgrid.prepare_active_grid(
            point_spacing=79.141 * u.km,
            altitudes=np.array([525.0], dtype=np.float64) * u.km,
            min_elevations=np.array([5.0], dtype=np.float64) * u.deg,
            inclinations=np.array([90.0], dtype=np.float64) * u.deg,
            station_lat=-30.7128 * u.deg,
            station_lon=21.4436 * u.deg,
            geography_mask_mode="none",
            ras_exclusion_mode="adjacency_layers",
            ras_exclusion_layers=1,
        )

        explicit = earthgrid.resolve_theta2_active_cell_ids(
            prepared_no_exclusion,
            scope_mode="cell_ids",
            explicit_ids=[0, 1, 1],
        )
        layered = earthgrid.resolve_theta2_active_cell_ids(
            prepared_no_exclusion,
            scope_mode="adjacency_layers",
            layers=1,
        )
        fully_excluded = earthgrid.resolve_theta2_active_cell_ids(
            prepared,
            scope_mode="adjacency_layers",
            layers=1,
        )

        assert_equal(explicit, np.array([0, 1], dtype=np.int32))
        assert layered.size > 0
        assert np.all(layered >= 0)
        assert np.all(layered < prepared_no_exclusion["active_cell_count"])
        assert fully_excluded.size == 0

    def test_pattern_wavelength_helpers_share_gui_frequency_logic(self):
        derived = pattern_wavelength_cm_from_frequency_mhz(2690.0)
        resolved = resolve_pattern_wavelength_cm(
            frequency_mhz=2690.0,
            pattern_wavelength_cm=None,
            derive_from_frequency=True,
        )

        assert derived == pytest.approx(resolved)

        with pytest.raises(ValueError, match="Frequency is required"):
            resolve_pattern_wavelength_cm(
                frequency_mhz=None,
                pattern_wavelength_cm=None,
                derive_from_frequency=True,
            )

    def test_resolve_contour_half_angle_deg_matches_direct_beamwidth(self):
        antenna_func, wavelength, pattern_kwargs = build_satellite_pattern_spec(
            antenna_model="s1528_rec1_4",
            frequency_mhz=2690.0,
            pattern_wavelength_cm=15.0,
            derive_pattern_wavelength_from_frequency=False,
            rec14_gm_dbi=34.1,
            rec14_lt_m=1.6,
            rec14_lr_m=1.6,
            rec14_l=2,
            rec14_slr_db=20.0,
            rec14_far_sidelobe_start_deg=90.0,
            rec14_far_sidelobe_level_dbi=-20.0,
            use_numba=False,
        )

        assert float(pattern_kwargs["Gm"].to_value(cnv.dBi)) == pytest.approx(34.1)

        half_angle_deg = earthgrid.resolve_contour_half_angle_deg(
            antenna_func,
            wavelength=wavelength,
            contour_drop="db7",
            **pattern_kwargs,
        )
        # resolve_contour_half_angle_deg dispatches through
        # calculate_beamwidth_2d which scans both principal planes with
        # 0.1° step and returns the geometric-mean beamwidth. Compare
        # against the same 2-D path to match scan resolution.
        beamwidth = calculate_beamwidth_2d(
            antenna_func,
            level_drop=7.0 * cnv.dB,
            wavelength=wavelength,
            **pattern_kwargs,
        )

        assert half_angle_deg == pytest.approx(0.5 * float(beamwidth.to_value(u.deg)))
