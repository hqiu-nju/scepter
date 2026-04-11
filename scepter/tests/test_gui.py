import importlib.machinery
import json
import os
import sys
import threading
import types
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import h5py
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.testing import assert_allclose, assert_equal

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")

from PySide6 import QtCore, QtGui, QtWidgets


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

import scepter.gui_bootstrap as gui_bootstrap
import scepter.scepter_GUI as sgui


class _DummyProperty:
    def __init__(self) -> None:
        self.color = None
        self.ambient = None
        self.diffuse = None
        self.specular = None

    def SetColor(self, r: float, g: float, b: float) -> None:
        self.color = (r, g, b)

    def SetAmbient(self, value: float) -> None:
        self.ambient = float(value)

    def SetDiffuse(self, value: float) -> None:
        self.diffuse = float(value)

    def SetSpecular(self, value: float) -> None:
        self.specular = float(value)


class _DummyActor:
    def __init__(self) -> None:
        self.mapper = None
        self.orientation = (0.0, 0.0, 0.0)
        self.property = _DummyProperty()
        self.position = (0.0, 0.0, 0.0)

    def SetMapper(self, mapper: object) -> None:
        self.mapper = mapper

    def GetProperty(self) -> _DummyProperty:
        return self.property

    def SetOrientation(self, x: float, y: float, z: float) -> None:
        self.orientation = (float(x), float(y), float(z))

    def SetPosition(self, x: float, y: float, z: float) -> None:
        self.position = (float(x), float(y), float(z))


class _DummyCamera:
    def __init__(self) -> None:
        self.position = (0.0, 0.0, 1.0)
        self.focal_point = (0.0, 0.0, 0.0)
        self.up = (0.0, 0.0, 1.0)


class _DummyRenderer:
    def __init__(self) -> None:
        self.environment_texture: object | None = None
        self.use_image_based_lighting = False

    def SetEnvironmentTexture(self, texture: object | None) -> None:
        self.environment_texture = texture

    def SetUseImageBasedLighting(self, enabled: bool) -> None:
        self.use_image_based_lighting = bool(enabled)


class _DummyRenderWindow:
    def __init__(self) -> None:
        self.desired_update_rate: float | None = None
        self.finalize_calls = 0
        self.generic_window_id = 1

    def SetDesiredUpdateRate(self, value: float) -> None:
        self.desired_update_rate = float(value)

    def Finalize(self) -> None:
        self.finalize_calls += 1

    def GetGenericWindowId(self) -> int:
        return int(self.generic_window_id)


class _DummySkyboxTexture:
    def __init__(self, label: str) -> None:
        self.label = label

    def to_skybox(self) -> object:
        return {"skybox": self.label}


class _DummyPlotter(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.actors: dict[str, object] = {}
        self.background: tuple[object, ...] | None = None
        self.environment_texture: object | None = None
        self.mesh_calls: list[dict[str, object]] = []
        self.actor_calls: list[dict[str, object]] = []
        self.render_calls = 0
        self.close_calls = 0
        self.renderer = _DummyRenderer()
        self.camera = _DummyCamera()
        self.render_window = _DummyRenderWindow()
        self.ren_win = self.render_window

    def set_background(self, *args: object, **kwargs: object) -> None:
        del kwargs
        self.background = args

    def add_mesh(self, mesh: object, **kwargs: object) -> _DummyActor:
        name = str(kwargs.get("name", f"mesh_{len(self.actors)}"))
        actor = _DummyActor()
        self.actors[name] = actor
        self.mesh_calls.append({"name": name, "kwargs": dict(kwargs), "mesh": mesh, "actor": actor})
        return actor

    def show_axes(self) -> None:
        return

    def enable_anti_aliasing(self) -> None:
        return

    def reset_camera(self) -> None:
        return

    def render(self) -> None:
        self.render_calls += 1

    def add_actor(self, actor: object, **kwargs: object) -> _DummyActor:
        name = str(kwargs.get("name", f"actor_{len(self.actors)}"))
        self.actors[name] = actor
        self.actor_calls.append({"name": name, "kwargs": dict(kwargs), "actor": actor})
        return actor if isinstance(actor, _DummyActor) else _DummyActor()

    def remove_actor(self, name: str, render: bool = False) -> None:
        del render
        self.actors.pop(name, None)

    def set_environment_texture(self, texture: object, **kwargs: object) -> None:
        del kwargs
        self.environment_texture = texture
        self.renderer.SetEnvironmentTexture(texture)

    def GetRenderWindow(self) -> _DummyRenderWindow:
        return self.render_window

    def close(self) -> bool:
        self.close_calls += 1
        return bool(super().close())


class _FailingRenderPlotter(_DummyPlotter):
    def render(self) -> None:
        self.render_calls += 1
        raise RuntimeError("dummy render failure")


def _default_antennas() -> sgui.AntennasConfig:
    return sgui.AntennasConfig(
        frequency_mhz=2690.0,
        pattern_wavelength_cm=15.0,
        derive_pattern_wavelength_from_frequency=False,
        antenna_model="s1528_rec1_4",
        rec12=sgui.AntennaRec12Config(gm_dbi=25.9, ln_db=-15.0, z=1.0),
        rec14=sgui.AntennaRec14Config(
            gm_dbi=34.1,
            lt_m=1.6,
            lr_m=1.6,
            l=2,
            slr_db=20.0,
            far_sidelobe_start_deg=90.0,
            far_sidelobe_level_dbi=-20.0,
        ),
        ras=sgui.RasAntennaConfig(
            antenna_diameter_m=15.0,
            operational_elevation_min_deg=15.0,
            operational_elevation_max_deg=90.0,
        ),
    )


def _tiny_state(*, include_ras: bool = True, include_antennas: bool = True) -> sgui.ScepterProjectState:
    antennas_legacy = _default_antennas() if include_antennas else None
    sat_antennas = (
        sgui.SatelliteAntennasConfig.from_antennas_config(antennas_legacy)
        if antennas_legacy is not None
        else sgui.SatelliteAntennasConfig()
    )
    ras_antenna = (
        sgui.RasAntennaConfig.from_json_dict(antennas_legacy.ras.to_json_dict())
        if antennas_legacy is not None
        else sgui.RasAntennaConfig()
    )
    system = sgui.SatelliteSystemConfig(
        system_name="System 1",
        belts=[
            sgui.BeltConfig(
                belt_name="Tiny_1",
                num_sats_per_plane=2,
                plane_count=2,
                altitude_km=525.0,
                eccentricity=0.0,
                inclination_deg=53.0,
                argp_deg=0.0,
                raan_min_deg=0.0,
                raan_max_deg=360.0,
                min_elevation_deg=20.0,
                adjacent_plane_offset=True,
            )
        ],
        satellite_antennas=sat_antennas,
        service=sgui._default_service_config(),
        spectrum=sgui._default_spectrum_config(),
        grid_analysis=sgui.GridAnalysisConfig(
            indicative_footprint_drop="db3",
            spacing_drop="db7",
            leading_metric="spacing_contour",
            cell_spacing_rule="full_footprint_diameter",
            cell_size_override_enabled=False,
            cell_size_override_km=None,
        ),
        hexgrid=sgui.HexgridConfig(
            geography_mask_mode="none",
            shoreline_buffer_km=None,
            coastline_backend="cartopy",
            ras_pointing_mode="ras_station",
            ras_exclusion_mode="none",
            ras_exclusion_layers=0,
            ras_exclusion_radius_km=None,
            boresight_avoidance_enabled=False,
            boresight_theta1_deg=None,
            boresight_theta2_deg=None,
            boresight_theta2_scope_mode="cell_ids",
            boresight_theta2_cell_ids=None,
            boresight_theta2_layers=0,
            boresight_theta2_radius_km=None,
        ),
        boresight=sgui.BoresightConfig(
            boresight_avoidance_enabled=False,
            boresight_theta1_deg=None,
            boresight_theta2_deg=None,
            boresight_theta2_scope_mode="cell_ids",
            boresight_theta2_cell_ids=None,
            boresight_theta2_layers=0,
            boresight_theta2_radius_km=None,
        ),
    )
    return sgui.ScepterProjectState(
        systems=[system],
        ras_station=(
            sgui.RasStationConfig(
                longitude_deg=21.443611,
                latitude_deg=-30.712777,
                elevation_m=1052.0,
                ras_reference_mode="lower",
                ras_reference_point_count=1,
                receiver_band_start_mhz=2690.0,
                receiver_band_stop_mhz=2700.0,
                receiver_response_mode="rectangular",
                receiver_custom_mask_points=None,
            )
            if include_ras
            else None
        ),
        ras_antenna=ras_antenna,
    )


def _ras_only_state(*, incomplete: bool = False) -> sgui.ScepterProjectState:
    state = _tiny_state(include_ras=True, include_antennas=True)
    state.active_system().belts = []
    if incomplete:
        assert state.ras_station is not None
        state.ras_station = sgui.RasStationConfig(
            longitude_deg=state.ras_station.longitude_deg,
            latitude_deg=state.ras_station.latitude_deg,
            elevation_m=None,
        )
    return state


def _invalid_constellation_state(*, include_ras: bool = False) -> sgui.ScepterProjectState:
    state = _tiny_state(include_ras=include_ras, include_antennas=True)
    state.active_system().belts = [
        sgui.BeltConfig(
            belt_name="Broken",
            num_sats_per_plane=2,
            plane_count=0,
            altitude_km=525.0,
            eccentricity=0.0,
            inclination_deg=53.0,
            argp_deg=0.0,
            raan_min_deg=0.0,
            raan_max_deg=360.0,
            min_elevation_deg=20.0,
            adjacent_plane_offset=True,
        )
    ]
    return state


def _sats_with_incomplete_ras_state() -> sgui.ScepterProjectState:
    state = _tiny_state(include_ras=True, include_antennas=True)
    assert state.ras_station is not None
    state.ras_station = sgui.RasStationConfig(
        longitude_deg=state.ras_station.longitude_deg,
        latitude_deg=state.ras_station.latitude_deg,
        elevation_m=None,
    )
    return state


def _output_family_profile(profile_name: str) -> dict[str, dict[str, object]]:
    config = sgui.scenario.default_output_families()
    for family_name in config:
        config[family_name]["mode"] = "none"
    if profile_name == "counts_only":
        config["beam_statistics"]["mode"] = "raw"
    elif profile_name == "totals_only":
        config["epfd_distribution"]["mode"] = "raw"
        config["prx_total_distribution"]["mode"] = "raw"
        config["total_pfd_ras_distribution"]["mode"] = "raw"
        config["beam_statistics"]["mode"] = "raw"
    elif profile_name == "notebook_full":
        config["epfd_distribution"]["mode"] = "both"
        config["prx_total_distribution"]["mode"] = "both"
        config["total_pfd_ras_distribution"]["mode"] = "both"
        config["per_satellite_pfd_distribution"]["mode"] = "both"
        config["prx_elevation_heatmap"]["mode"] = "preaccumulated"
        config["per_satellite_pfd_elevation_heatmap"]["mode"] = "both"
        config["beam_statistics"]["mode"] = "both"
    elif profile_name == "gui_heavy":
        config["prx_elevation_heatmap"]["mode"] = "both"
        config["beam_statistics"]["mode"] = "raw"
    else:
        raise KeyError(profile_name)
    return config


def _stub_scene_assets(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    skyboxes = {
        "4K": _DummySkyboxTexture("4K"),
        "16K": _DummySkyboxTexture("16K"),
    }
    monkeypatch.setattr(
        sgui,
        "_load_earth_assets",
        lambda: (pv.Sphere(radius=sgui._EARTH_RADIUS_KM), object()),
    )
    monkeypatch.setattr(
        sgui,
        "_load_skybox_texture",
        lambda mode: None if mode == "Off" else skyboxes[mode],
    )
    return skyboxes


# Module-scoped asset stubs applied once — avoids per-test monkeypatch
# overhead.  The originals are saved and restored when the module finishes.
_original_load_earth_assets = None
_original_load_skybox_texture = None


@pytest.fixture(scope="module", autouse=True)
def _module_scene_stubs():
    """Stub PyVista asset loading once for the entire module."""
    global _original_load_earth_assets, _original_load_skybox_texture
    _original_load_earth_assets = sgui._load_earth_assets
    _original_load_skybox_texture = sgui._load_skybox_texture
    skyboxes = {
        "4K": _DummySkyboxTexture("4K"),
        "16K": _DummySkyboxTexture("16K"),
    }
    sgui._load_earth_assets = lambda: (pv.Sphere(radius=sgui._EARTH_RADIUS_KM), object())
    sgui._load_skybox_texture = lambda mode: None if mode == "Off" else skyboxes[mode]
    yield
    sgui._load_earth_assets = _original_load_earth_assets
    sgui._load_skybox_texture = _original_load_skybox_texture


def _manual_frame_set() -> sgui.PreviewFrameSet:
    ras_position = sgui._geodetic_to_cartesian_km(
        np.asarray([21.443611], dtype=np.float64),
        np.asarray([-30.712777], dtype=np.float64),
        np.asarray([1.052], dtype=np.float64),
    )[0]
    frame0 = np.asarray(
        (
            (6900.0, 0.0, 0.0),
            (0.0, 6900.0, 0.0),
            (-6900.0, 0.0, 0.0),
            (0.0, -6900.0, 0.0),
        ),
        dtype=np.float32,
    )
    frame1 = frame0 + np.asarray((0.0, 45.0, 25.0), dtype=np.float32)
    vel0 = np.asarray(
        (
            (0.0, 7.5, 0.1),
            (-7.5, 0.0, 0.1),
            (0.0, -7.5, 0.1),
            (7.5, 0.0, 0.1),
        ),
        dtype=np.float32,
    )
    vel1 = vel0.copy()
    return sgui.PreviewFrameSet(
        times_utc=[
            datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
        ],
        sample_times_s=np.asarray([0.0, 5.0], dtype=np.float32),
        positions_ecef_km=np.stack((frame0, frame1), axis=0),
        positions_eci_km=np.stack((frame0 + 10.0, frame1 + 10.0), axis=0),
        velocities_eci_km_s=np.stack((vel0, vel1), axis=0),
        earth_rotation_deg=np.asarray([5.0, 25.0], dtype=np.float32),
        belt_names=["Tiny_1"],
        belt_counts=np.asarray([4], dtype=np.int64),
        belt_offsets=np.asarray([0, 4], dtype=np.int64),
        belt_colors=["#22c55e"],
        ras_position_ecef_km=ras_position,
    )


def _small_preview_params() -> sgui.PreviewParameters:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return sgui.PreviewParameters(
        start_utc=start,
        end_utc=start + timedelta(seconds=10.0),
        frame_step_s=5.0,
        playback_fps=30.0,
    )


def _wait_for_viewer_build(
    viewer: sgui.ConstellationViewerWindow,
    *,
    timeout_ms: int = 2000,
) -> None:
    _wait_until(lambda: viewer._build_thread is None, timeout_ms=timeout_ms)


def _manual_frame_set_with_sat_count(sat_count: int) -> sgui.PreviewFrameSet:
    frame_set = _manual_frame_set()
    sat_count = int(sat_count)
    return sgui.PreviewFrameSet(
        times_utc=frame_set.times_utc,
        sample_times_s=frame_set.sample_times_s,
        positions_ecef_km=frame_set.positions_ecef_km[:, :sat_count, :].copy(),
        positions_eci_km=frame_set.positions_eci_km[:, :sat_count, :].copy(),
        velocities_eci_km_s=frame_set.velocities_eci_km_s[:, :sat_count, :].copy(),
        earth_rotation_deg=frame_set.earth_rotation_deg,
        belt_names=frame_set.belt_names,
        belt_counts=np.asarray([sat_count], dtype=np.int64),
        belt_offsets=np.asarray([0, sat_count], dtype=np.int64),
        belt_colors=frame_set.belt_colors,
        ras_position_ecef_km=frame_set.ras_position_ecef_km,
        status_note=frame_set.status_note,
    )


def _stub_preview_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _propagate_many(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        del kwargs
        mjds = np.asarray(args[0], dtype=np.float64)
        tle_list = np.asarray(args[1], dtype=object)
        frame_count = int(mjds.shape[0])
        sat_count = int(tle_list.shape[-1])
        frame_axis = np.arange(frame_count, dtype=np.float64)[:, np.newaxis]
        sat_axis = np.arange(sat_count, dtype=np.float64)[np.newaxis, :]
        geo = np.zeros((frame_count, sat_count, 3), dtype=np.float64)
        geo[..., 0] = 20.0 + frame_axis + sat_axis
        geo[..., 1] = -30.0 + 0.25 * frame_axis - 0.1 * sat_axis
        geo[..., 2] = 525.0 + 2.0 * frame_axis + sat_axis
        eci_pos = np.stack((geo[..., 0] + 10.0, geo[..., 1] + 10.0, geo[..., 2] + 10.0), axis=-1)
        eci_vel = np.stack(
            (
                np.full((frame_count, sat_count), 0.1, dtype=np.float64),
                np.full((frame_count, sat_count), 0.2, dtype=np.float64),
                np.full((frame_count, sat_count), 0.3, dtype=np.float64),
            ),
            axis=-1,
        )
        return {"geo": geo, "eci_pos": eci_pos, "eci_vel": eci_vel}

    monkeypatch.setattr(sgui.cysgp4, "propagate_many", _propagate_many)


def _manual_hexgrid_result() -> dict[str, object]:
    return {
        "pre_ras_cell_longitudes": np.asarray([20.0, 20.5, 21.0], dtype=np.float64) * sgui.u.deg,
        "pre_ras_cell_latitudes": np.asarray([-30.8, -30.7, -30.6], dtype=np.float64) * sgui.u.deg,
        "active_grid_longitudes": np.asarray([20.0, 21.0], dtype=np.float64) * sgui.u.deg,
        "active_grid_latitudes": np.asarray([-30.8, -30.6], dtype=np.float64) * sgui.u.deg,
        "ras_exclusion_mask_pre_ras": np.asarray([False, True, False], dtype=bool),
        "active_cell_count": 2,
        "ras_excluded_cell_count": 1,
        "point_spacing_km": 79.141,
        "station_lon": 21.443611 * sgui.u.deg,
        "station_lat": -30.712777 * sgui.u.deg,
    }


def _clear_gui_settings() -> None:
    # When a cached window exists, clear its own _settings instance to avoid
    # the Windows bug where an external QSettings.clear() silently breaks
    # the window's instance (subsequent setValue calls become no-ops).
    if _CACHED_WINDOW is not None and hasattr(_CACHED_WINDOW, '_settings'):
        for _key in list(_CACHED_WINDOW._settings.allKeys()):
            _CACHED_WINDOW._settings.remove(_key)
        _CACHED_WINDOW._settings.sync()
        return
    settings = QtCore.QSettings("SKAO", "SCEPTer GUI")
    settings.clear()
    settings.sync()


def _make_run_window(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: sgui.ScepterProjectState | None = None,
    current_hexgrid: bool = True,
    disable_live_sampling: bool = True,
) -> sgui.ScepterMainWindow:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state() if state is None else state)
    if current_hexgrid:
        _mark_hexgrid_preview_current(window)
    window._refresh_summary()
    if disable_live_sampling:
        monkeypatch.setattr(window.run_monitor, "_sample_live_resource_usage", lambda: {})
        monkeypatch.setattr(window.run_monitor, "_sample_gpu_device_metrics", lambda: {})
        window.run_monitor._timing_timer.stop()
    return window


def _simulation_page_ready_by_label(window: sgui.ScepterMainWindow) -> dict[str, bool]:
    return {
        window.simulation_page_list.item(idx).text(): bool(
            window.simulation_page_list.item(idx).data(QtCore.Qt.UserRole + 1)
        )
        for idx in range(window.simulation_page_list.count())
    }


def _synthetic_spectrum_preview_plan(
    *,
    service_start_mhz: float = 2620.0,
    service_stop_mhz: float = 2625.0,
    ras_start_mhz: float = 2690.0,
    ras_stop_mhz: float = 2700.0,
    cutoff_mhz: float = 40.0,
    receiver_mode: str = "rectangular",
    receiver_points_mhz: np.ndarray | None = None,
    unwanted_emission_mask_points_mhz: np.ndarray | None = None,
) -> dict[str, object]:
    mask_points = (
        np.asarray(
            [
                [-2.5, 0.0],
                [2.5, 0.0],
                [7.5, 30.0],
                [22.5, 60.0],
            ],
            dtype=np.float64,
        )
        if unwanted_emission_mask_points_mhz is None
        else np.asarray(unwanted_emission_mask_points_mhz, dtype=np.float64)
    )
    return {
        "slot_edges_mhz": np.asarray([service_start_mhz, service_stop_mhz], dtype=np.float64),
        "slot_centers_mhz": np.asarray(
            [0.5 * (float(service_start_mhz) + float(service_stop_mhz))],
            dtype=np.float64,
        ),
        "unwanted_emission_mask_points_mhz": mask_points,
        "channel_bandwidth_mhz": float(service_stop_mhz - service_start_mhz),
        "ras_receiver_band_start_mhz": float(ras_start_mhz),
        "ras_receiver_band_stop_mhz": float(ras_stop_mhz),
        "receiver_response_mode": str(receiver_mode),
        "receiver_response_points_mhz": (
            None
            if receiver_points_mhz is None
            else np.asarray(receiver_points_mhz, dtype=np.float64)
        ),
        "spectral_integration_cutoff_mhz": float(cutoff_mhz),
        "service_band_start_mhz": float(service_start_mhz),
        "service_band_stop_mhz": float(service_stop_mhz),
        "reuse_factor": 1,
        "enabled_channel_indices": np.asarray([0], dtype=np.int32),
    }


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


def _normalized_compacted_spectrum_preview_plan(
    enabled_channel_indices: list[int],
    *,
    service_band_start_mhz: float = 2620.0,
    service_band_stop_mhz: float = 2690.0,
    ras_receiver_band_start_mhz: float = 2690.0,
    ras_receiver_band_stop_mhz: float = 2700.0,
    channel_bandwidth_mhz: float = 5.0,
    reuse_factor: int = 1,
    receiver_response_mode: str = "rectangular",
    receiver_response_points_mhz: np.ndarray | None = None,
    cutoff_percent: float = 450.0,
) -> dict[str, object]:
    active_cell_reuse_slot_ids = np.arange(max(1, int(reuse_factor)), dtype=np.int32) % max(1, int(reuse_factor))
    spectrum_plan = sgui.scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan={
            "service_band_start_mhz": float(service_band_start_mhz),
            "service_band_stop_mhz": float(service_band_stop_mhz),
            "ras_receiver_band_start_mhz": float(ras_receiver_band_start_mhz),
            "ras_receiver_band_stop_mhz": float(ras_receiver_band_stop_mhz),
            "reuse_factor": int(reuse_factor),
            "enabled_channel_indices": list(enabled_channel_indices),
            "unwanted_emission_mask_preset": "custom",
            "custom_mask_points": np.asarray(_DIRECT_EPFD_TEST_MASK_POINTS_MHZ, dtype=np.float64),
            "receiver_response_mode": str(receiver_response_mode),
            "receiver_custom_mask_points": (
                None
                if receiver_response_points_mhz is None
                else np.asarray(receiver_response_points_mhz, dtype=np.float64)
            ),
            "spectral_integration_cutoff_basis": "channel_bandwidth",
            "spectral_integration_cutoff_percent": float(cutoff_percent),
        },
        channel_bandwidth_mhz=float(channel_bandwidth_mhz),
        active_cell_count=int(active_cell_reuse_slot_ids.size),
        active_cell_reuse_slot_ids=active_cell_reuse_slot_ids,
    )
    assert spectrum_plan is not None
    return dict(spectrum_plan)


_DIRECT_EPFD_GUI_SERVICE_CASES = (
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


def _gui_test_full_channel_count(service_case: Mapping[str, float | str]) -> int:
    return int(
        np.floor(
            (
                float(service_case["service_band_stop_mhz"])
                - float(service_case["service_band_start_mhz"])
            )
            / float(service_case["channel_bandwidth_mhz"])
        )
    )


def _gui_test_ras_case(
    service_case: Mapping[str, float | str],
    case_name: str,
) -> dict[str, object]:
    service_start_mhz = float(service_case["service_band_start_mhz"])
    service_stop_mhz = float(service_case["service_band_stop_mhz"])
    if case_name == "upper_adjacent_rectangular":
        return {
            "ras_receiver_band_start_mhz": service_stop_mhz,
            "ras_receiver_band_stop_mhz": service_stop_mhz + 10.0,
            "receiver_response_mode": "rectangular",
            "receiver_response_points_mhz": None,
        }
    if case_name == "lower_adjacent_rectangular":
        return {
            "ras_receiver_band_start_mhz": service_start_mhz - 10.0,
            "ras_receiver_band_stop_mhz": service_start_mhz,
            "receiver_response_mode": "rectangular",
            "receiver_response_points_mhz": None,
        }
    if case_name == "upper_adjacent_custom_asymmetric":
        return {
            "ras_receiver_band_start_mhz": service_stop_mhz,
            "ras_receiver_band_stop_mhz": service_stop_mhz + 10.0,
            "receiver_response_mode": "custom",
            "receiver_response_points_mhz": np.asarray(
                _DIRECT_EPFD_TEST_CUSTOM_RX_POINTS_MHZ,
                dtype=np.float64,
            ),
        }
    raise KeyError(case_name)


def _gui_test_channel_leakage_map_from_plan(plan: Mapping[str, object]) -> dict[int, float]:
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


def _gui_structured_channel_subsets(
    full_channel_count: int,
    reuse_factor: int,
) -> list[list[int]]:
    subsets: list[list[int]] = []
    seen_keys: set[tuple[int, ...]] = set()

    def _add(subset: list[int]) -> None:
        cleaned = sorted(
            {
                int(value)
                for value in subset
                if 0 <= int(value) < int(full_channel_count)
            }
        )
        key = tuple(cleaned)
        if key not in seen_keys:
            seen_keys.add(key)
            subsets.append(cleaned)

    reuse_factor_i = max(1, int(reuse_factor))
    _add([])
    _add(list(range(int(full_channel_count))))
    for index in range(int(full_channel_count)):
        _add([index])
    for start in range(max(0, int(full_channel_count) - 1)):
        _add([start, start + 1])
    for start in range(max(0, int(full_channel_count) - 2)):
        _add([start, start + 1, start + 2])
    _add(list(range(0, int(full_channel_count), 2)))
    _add(list(range(1, int(full_channel_count), 2)))
    _add(list(range(max(0, int(full_channel_count) - 3), int(full_channel_count))))
    _add(list(range(max(0, int(full_channel_count) - reuse_factor_i), int(full_channel_count))))
    for slot_id in range(reuse_factor_i):
        _add(list(range(slot_id, int(full_channel_count), reuse_factor_i)))
    group_count = max(1, int(np.ceil(int(full_channel_count) / float(reuse_factor_i))))
    for group_index in range(group_count):
        _add([slot_id + group_index * reuse_factor_i for slot_id in range(reuse_factor_i)])
    return subsets


def _gui_sampled_channel_subsets(
    full_channel_count: int,
    reuse_factor: int,
    *,
    sample_count: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(int(seed) + int(reuse_factor) * 1009 + int(full_channel_count) * 9176)
    subsets = _gui_structured_channel_subsets(full_channel_count, reuse_factor)
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


def _find_axis_line_by_label_prefix(axis: Any, prefix: str) -> Any:
    for line in axis.lines:
        label = str(line.get_label())
        if label.startswith(prefix):
            return line
    raise AssertionError(f"Line starting with {prefix!r} was not found.")


def _find_axis_lines_by_label_prefix(axis: Any, prefix: str) -> list[Any]:
    return [
        line
        for line in axis.lines
        if str(line.get_label()).startswith(prefix)
    ]


def _axis_nonempty_title(axis: Any) -> Any:
    for attr_name in ("_left_title", "title", "_right_title"):
        text = getattr(axis, attr_name, None)
        if text is not None and str(text.get_text()).strip():
            return text
    raise AssertionError("Axis title was not found.")


def _axis_vertical_boundary_lines(axis: Any) -> list[Any]:
    vertical_lines: list[Any] = []
    for line in axis.lines:
        xdata = np.asarray(line.get_xdata(), dtype=np.float64).reshape(-1)
        if xdata.size == 2 and np.allclose(xdata, xdata[0]):
            vertical_lines.append(line)
    return vertical_lines


def _line_uses_solid_style(line: Any) -> bool:
    return str(line.get_linestyle()) in {"-", "solid"}


def _write_minimal_result_file(
    path: Path,
    *,
    include_times: bool = True,
    times_days: np.ndarray | None = None,
    legacy_raw_names: bool = False,
    include_bandwidth_attrs: bool = True,
) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["boresight"] = 1
        if not legacy_raw_names:
            h5.attrs["result_schema_version"] = 2
            h5.attrs["stored_power_basis"] = "channel_total"
        if include_bandwidth_attrs:
            h5.attrs["bandwidth_mhz"] = 5.0
            h5.attrs["power_input_quantity"] = "target_pfd"
            h5.attrs["power_input_basis"] = "per_mhz"
        const = h5.create_group("const")
        const.create_dataset("sat_ids", data=np.asarray([0, 1], dtype=np.int32))
        pre = h5.create_group("preaccumulated")
        prx = pre.create_group("prx_total_distribution")
        if include_bandwidth_attrs:
            prx.attrs["bandwidth_mhz"] = 5.0
        prx.attrs["stored_value_basis"] = "channel_total" if not legacy_raw_names else "per_mhz"
        prx.create_dataset("counts", data=np.asarray([1, 2, 1], dtype=np.int64))
        prx.create_dataset("edges_dbw", data=np.asarray([-130.0, -129.0, -128.0, -127.0], dtype=np.float64))
        epfd = pre.create_group("epfd_distribution")
        if include_bandwidth_attrs:
            epfd.attrs["bandwidth_mhz"] = 5.0
        epfd.attrs["stored_value_basis"] = "channel_total" if not legacy_raw_names else "per_mhz"
        epfd.create_dataset("counts", data=np.asarray([1, 1, 2], dtype=np.int64))
        epfd.create_dataset("edges_dbw", data=np.asarray([-180.0, -179.0, -178.0, -177.0], dtype=np.float64))
        pfd = pre.create_group("total_pfd_ras_distribution")
        if include_bandwidth_attrs:
            pfd.attrs["bandwidth_mhz"] = 5.0
        pfd.attrs["stored_value_basis"] = "channel_total" if not legacy_raw_names else "per_mhz"
        pfd.create_dataset("counts", data=np.asarray([2, 1, 1], dtype=np.int64))
        pfd.create_dataset("edges_dbw", data=np.asarray([-170.0, -169.0, -168.0, -167.0], dtype=np.float64))
        heat = pre.create_group("prx_elevation_heatmap")
        heat.attrs["elevation_bin_step_deg"] = 5.0
        heat.attrs["value_bin_step_db"] = 0.5
        if include_bandwidth_attrs:
            heat.attrs["bandwidth_mhz"] = 5.0
        heat.attrs["stored_value_basis"] = "channel_total" if not legacy_raw_names else "per_mhz"
        heat.create_dataset(
            "counts",
            data=np.asarray(
                [
                    [0, 2, 1],
                    [1, 3, 0],
                    [0, 1, 4],
                ],
                dtype=np.int64,
            ),
        )
        heat.create_dataset(
            "elevation_edges_deg",
            data=np.asarray([0.0, 5.0, 10.0, 15.0], dtype=np.float64),
        )
        heat.create_dataset(
            "value_edges_dbw",
            data=np.asarray([-135.0, -134.5, -134.0, -133.5], dtype=np.float64),
        )
        beams = pre.create_group("beam_statistics")
        beams.create_dataset("count_edges", data=np.asarray([0, 1, 2, 3, 4], dtype=np.int64))
        beams.create_dataset("full_network_count_hist", data=np.asarray([1, 2, 2, 1], dtype=np.int64))
        beams.create_dataset("visible_count_hist", data=np.asarray([1, 2, 2, 1], dtype=np.int64))
        beams.create_dataset("network_total_beams_over_time", data=np.asarray([3, 3, 3], dtype=np.int64))
        beams.create_dataset("visible_total_beams_over_time", data=np.asarray([3, 3, 3], dtype=np.int64))
        beams.create_dataset("beam_demand_over_time", data=np.asarray([2, 2, 3], dtype=np.int64))
        it = h5.create_group("iter")
        row = it.create_group("iter_00000")
        if include_times:
            row.create_dataset(
                "times",
                data=np.asarray(
                    [0.0, 1.0, 2.0] if times_days is None else times_days,
                    dtype=np.float64,
                ),
            )
        prx_samples = np.full((3, 1, 1734), 1.0e-13, dtype=np.float64)
        prx_samples[1, 0, 0] = 2.0e-13
        prx_samples[2, 0, 1] = 3.0e-13
        raw_names = {
            "Prx_total_W": "Prx_total_WperMHz",
            "EPFD_W_m2": "EPFD_W_m2_MHz",
            "PFD_total_RAS_STATION_W_m2": "PFD_total_RAS_STATION_W_m2_MHz",
            "PFD_per_sat_RAS_STATION_W_m2": "PFD_per_sat_RAS_STATION_W_m2_MHz",
            "Prx_per_sat_RAS_STATION_W": "Prx_per_sat_RAS_STATION_WperMHz",
        }
        def _name(canonical: str) -> str:
            return raw_names[canonical] if legacy_raw_names else canonical
        row.create_dataset(_name("Prx_total_W"), data=prx_samples)
        row.create_dataset(_name("EPFD_W_m2"), data=prx_samples * 1.0e-3)
        row.create_dataset(_name("PFD_total_RAS_STATION_W_m2"), data=np.asarray([1e-17, 2e-17, 3e-17], dtype=np.float64))
        row.create_dataset(
            _name("PFD_per_sat_RAS_STATION_W_m2"),
            data=np.asarray([[1e-18, 2e-18], [2e-18, 3e-18], [1e-18, 4e-18]], dtype=np.float64),
        )
        row.create_dataset(
            _name("Prx_per_sat_RAS_STATION_W"),
            data=np.asarray([[1e-13, 2e-13], [2e-13, 3e-13], [1e-13, 4e-13]], dtype=np.float64),
        )
        row.create_dataset(
            "sat_elevation_RAS_STATION_deg",
            data=np.asarray([[10.0, 20.0], [15.0, 25.0], [30.0, 40.0]], dtype=np.float64),
        )
        row.create_dataset(
            "sat_beam_counts_used",
            data=np.asarray([[1, 2], [2, 1], [0, 3]], dtype=np.int32),
        )
        row.create_dataset("beam_demand_count", data=np.asarray([2, 2, 3], dtype=np.int32))


@pytest.fixture(scope="module")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# ---------------------------------------------------------------------------
# Cached window pool — construct ScepterMainWindow once, reuse across tests.
# Each call to sgui.ScepterMainWindow() within a test returns the cached
# instance after a lightweight state reset.  This eliminates ~140 × 0.4 s =
# 56 s of construction overhead.
# ---------------------------------------------------------------------------
_CACHED_WINDOW: sgui.ScepterMainWindow | None = None
_ORIGINAL_MAIN_WINDOW_CLS: type | None = None
_ORIG_PLOT_CELL_STATUS_MAP = None  # set by _stub_cartopy_heavy_functions


def _reset_cached_window(window: sgui.ScepterMainWindow) -> None:
    """Cheaply reset the shared window to a blank default state."""
    # Neutralize matplotlib canvases FIRST to prevent cartopy draws during reset
    _neutralize_all_mpl_qt_canvases()
    # Clear settings via the window's OWN _settings instance.  Using a
    # separate QSettings instance (as _clear_gui_settings does) can leave
    # the window's object in a broken state on Windows where subsequent
    # setValue calls silently fail.
    for _key in list(window._settings.allKeys()):
        window._settings.remove(_key)
    window._settings.sync()
    window._current_path = None
    window._hexgrid_completed_signature = None
    window._hexgrid_completed_commit_signature = None
    window._hexgrid_completed_overlay_signature = None
    window._hexgrid_outdated = True
    window._last_analyser_signature = None
    window._last_analyser_selected_cell_km = None
    window._run_worker = None
    window._run_in_progress = False
    window._run_status_override = None
    window._review_run_state = None
    # Stop any lingering timers from previous test
    if hasattr(window, 'run_monitor'):
        window.run_monitor._timing_timer.stop()
    if hasattr(window, '_side_pane_auto_hide_timer'):
        window._side_pane_auto_hide_timer.stop()
    if hasattr(window, '_summary_refresh_timer'):
        window._summary_refresh_timer.stop()
    # Stop hexgrid preview timer to prevent deferred refreshes
    if hasattr(window, '_hexgrid_preview_timer'):
        window._hexgrid_preview_timer.stop()
    # Clear hexgrid preview cached result to prevent cartopy redraws during
    # appearance mode changes.
    window._hexgrid_preview_cached_result = None
    if hasattr(window, '_hexgrid_preview_window') and window._hexgrid_preview_window is not None:
        try:
            window._hexgrid_preview_window.hide()
            window._hexgrid_preview_window.setParent(None)
        except Exception:
            pass
        window._hexgrid_preview_window = None
    # Neutralize any matplotlib Qt canvases to prevent deferred draws
    _neutralize_all_mpl_qt_canvases()
    # Reset UI chrome to defaults (block combo signals to prevent
    # _on_appearance_mode_changed → _restyle_detached_preview_windows chain
    # and _on_complexity_mode_changed → layout refresh chain)
    blockers = [
        QtCore.QSignalBlocker(window.appearance_mode_combo),
        QtCore.QSignalBlocker(window.complexity_mode_combo),
    ]
    window._restore_default_session_preferences()
    del blockers
    # Apply complexity mode layout since the combo signal was blocked above
    window._apply_complexity_mode()
    # Reset open_result_button state (can be left enabled from a successful run)
    window.open_result_button.setEnabled(False)
    window.run_monitor.open_result_button.setEnabled(False)
    # Clear status bar text from previous test
    window._persistent_status_bar_message = "Ready"
    window._status_bar_context_source = None
    window.statusBar().showMessage("Ready")
    # Reset run-progress fingerprint so dedup does not suppress first event
    window._last_run_progress_fingerprint = None
    window._pending_run_progress_event = None
    if hasattr(window, '_run_monitor_update_timer'):
        window._run_monitor_update_timer.stop()
    # Reset run-related labels and state
    window.run_status_label.setText("")
    window._run_thread = None
    window._run_cancel_controller = None
    # Reset postprocess widget state
    if hasattr(window, 'postprocess_widget'):
        window.postprocess_widget._current_filename = None
        window.postprocess_widget._recipe_param_widgets = {}
        if hasattr(window.postprocess_widget, '_render_thread') and window.postprocess_widget._render_thread is not None:
            try:
                window.postprocess_widget._render_thread.quit()
                window.postprocess_widget._render_thread.wait(500)
            except Exception:
                pass
            window.postprocess_widget._render_thread = None
    # Refresh home recent lists (settings were cleared at top of reset)
    window._refresh_home_recent_lists()
    # Reset side pane state
    window.side_pane_pin_button.setChecked(False)
    window.side_pane_toggle_button.setChecked(True)
    window._side_pane_pinned = False
    if hasattr(window, '_right_container'):
        window._right_container.setHidden(False)
    # Close any open hexgrid/viewer windows from previous test
    if hasattr(window, '_hexgrid_preview_window') and window._hexgrid_preview_window is not None:
        try:
            window._hexgrid_preview_window.close()
        except Exception:
            pass
        window._hexgrid_preview_window = None
    if hasattr(window, '_hexgrid_preview_thread') and window._hexgrid_preview_thread is not None:
        try:
            window._hexgrid_preview_thread.quit()
            window._hexgrid_preview_thread.wait(500)
        except Exception:
            pass
        window._hexgrid_preview_thread = None
    if hasattr(window, '_viewer_window') and window._viewer_window is not None:
        try:
            window._viewer_window.close()
        except Exception:
            pass
        window._viewer_window = None
    # Reset window size and layout so geometry-dependent assertions see
    # consistent values after tests that change scale or workspace nav.
    window.resize(1280, 800)
    window._workspace_nav_expanded = True
    window._workspace_nav_current_width = sgui._WORKSPACE_NAV_EXPANDED_WIDTH
    window._refresh_responsive_layouts()
    window._stabilize_visible_geometry()
    # Reset workspace to simulation
    window._set_workspace(sgui._WORKSPACE_SIMULATION)
    # Load blank state into widgets (blocks signals, sets _state_load_in_progress)
    window._load_state_into_widgets(sgui.ScepterProjectState())
    # Override _prompt_save_changes to never show a QMessageBox dialog —
    # always returns True (proceed) so tests don't hang on modal dialogs.
    window._prompt_save_changes = lambda: True  # type: ignore[assignment]
    # Ensure dirty is False (must be last)
    window._dirty = False


def _get_or_create_cached_window() -> sgui.ScepterMainWindow:
    """Return the cached window, constructing on first call, resetting after."""
    global _CACHED_WINDOW
    if _CACHED_WINDOW is None:
        assert _ORIGINAL_MAIN_WINDOW_CLS is not None
        _CACHED_WINDOW = _ORIGINAL_MAIN_WINDOW_CLS()
        # Override close() to prevent destruction during tests.
        # Tests that do ``window._dirty = False; window.close()`` won't
        # actually destroy the Qt widget tree.
        _CACHED_WINDOW._real_close = _CACHED_WINDOW.close
        _CACHED_WINDOW.close = lambda: True  # type: ignore[assignment]
    else:
        _reset_cached_window(_CACHED_WINDOW)
    return _CACHED_WINDOW


@pytest.fixture(scope="module", autouse=True)
def _stub_cartopy_heavy_functions():
    """Replace cartopy-heavy visualization functions with fast stubs for the
    entire test module.  Individual tests that need specific behavior still
    override via monkeypatch."""
    global _ORIG_PLOT_CELL_STATUS_MAP
    _orig_plot = sgui.visualise.plot_cell_status_map
    _ORIG_PLOT_CELL_STATUS_MAP = _orig_plot

    def _stub_plot_cell_status_map(*args, **kwargs):
        fig = Figure()
        fig.add_subplot(111)
        return (
            fig,
            {
                "switched_off_count": 0,
                "normal_active_count": 0,
                "boresight_affected_active_count": 0,
                "map_style_used": "clean",
                "backend_used": "matplotlib",
            },
        )

    sgui.visualise.plot_cell_status_map = _stub_plot_cell_status_map
    yield
    sgui.visualise.plot_cell_status_map = _orig_plot


@pytest.fixture(scope="module", autouse=True)
def _install_window_cache():
    """Replace sgui.ScepterMainWindow with a factory that returns a cached
    instance.  First call constructs the real window; subsequent calls
    reset and return the same instance.  Saves ~140 × 0.4 s = 56 s."""
    global _CACHED_WINDOW, _ORIGINAL_MAIN_WINDOW_CLS

    _ORIGINAL_MAIN_WINDOW_CLS = sgui.ScepterMainWindow

    # Replace the class with a callable factory.  Tests that do
    # ``window = sgui.ScepterMainWindow()`` will get the cached window.
    sgui.ScepterMainWindow = _get_or_create_cached_window  # type: ignore[assignment]

    # Prevent WA_DeleteOnClose on FigureWindow — deferred deletion of
    # matplotlib canvases causes access violations on Windows.
    _orig_fw_init = sgui.FigureWindow.__init__

    def _fw_init_no_delete_on_close(self, *args, **kwargs):
        _orig_fw_init(self, *args, **kwargs)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

    sgui.FigureWindow.__init__ = _fw_init_no_delete_on_close

    yield

    # Teardown — close all open FigureWindow / dialog instances first,
    # then the cached main window, then flush deferred deletions so the
    # next test module starts with a clean Qt widget tree.
    sgui.FigureWindow.__init__ = _orig_fw_init
    sgui.ScepterMainWindow = _ORIGINAL_MAIN_WINDOW_CLS
    _neutralize_all_mpl_qt_canvases()
    plt.close("all")
    app = QtWidgets.QApplication.instance()
    if app is not None:
        # Close all top-level windows except the cached main window
        for w in list(app.topLevelWidgets()):
            if w is not _CACHED_WINDOW:
                try:
                    w.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
                    w.hide()
                    w.close()
                except Exception:
                    pass
    if _CACHED_WINDOW is not None:
        _CACHED_WINDOW._dirty = False
        if hasattr(_CACHED_WINDOW, '_real_close'):
            _CACHED_WINDOW._real_close()
        else:
            _ORIGINAL_MAIN_WINDOW_CLS.close(_CACHED_WINDOW)
    # Process ALL pending deletions before the next module starts
    if app is not None:
        import gc
        gc.collect()
        sgui._flush_deferred_deletions()
        app.processEvents()
        sgui._flush_deferred_deletions()
        app.processEvents()
        gc.collect()
    _CACHED_WINDOW = None
    _ORIGINAL_MAIN_WINDOW_CLS = None


def _neutralize_all_mpl_qt_canvases() -> None:
    """Kill all matplotlib Qt canvas timers to prevent deferred cartopy draws
    and close any detached FigureWindows to prevent access violations.

    The matplotlib Qt backend uses a QTimer for idle draws.  If a cartopy
    GeoAxes is queued for redraw when the timer fires, the cartopy
    projection code can block indefinitely.  This neutralizes ALL
    matplotlib Qt canvases that are alive in the current process.
    """
    try:
        from matplotlib.backends.backend_qt import FigureCanvasQT
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        for obj in list(app.allWidgets()):
            if isinstance(obj, FigureCanvasQT):
                # Stop the idle draw timer
                timer = getattr(obj, "_idle_draw_timer", None)
                if timer is not None:
                    timer.stop()
                # Kill draw/draw_idle/paintEvent to prevent ANY rendering
                obj.draw_idle = lambda: None  # type: ignore[assignment]
                obj.draw = lambda: None  # type: ignore[assignment]
                obj.paintEvent = lambda e: None  # type: ignore[assignment]
                # Clear figure
                fig = getattr(obj, "figure", None)
                if fig is not None:
                    fig.clf()
                # Hide to prevent any paint events
                obj.hide()
        # Disable WA_DeleteOnClose on FigureWindow instances to prevent
        # access violations during deferred Qt object destruction.
        for obj in list(app.topLevelWidgets()):
            if isinstance(obj, sgui.FigureWindow):
                obj.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
                obj.hide()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _cleanup_qt_objects(qapp: QtWidgets.QApplication):
    """Force deferred Qt object deletion after each test.

    With the cached window approach, deferred deletions are minimal.
    We still flush them periodically but neutralize any cartopy canvases
    first to prevent paint-event hangs.
    """
    yield
    _neutralize_all_mpl_qt_canvases()
    plt.close("all")
    # Close and destroy all FigureWindow / dialog instances to prevent
    # accumulated VTK/canvas resources from triggering segfaults.
    import gc
    app = QtWidgets.QApplication.instance()
    if app is not None:
        for w in list(app.topLevelWidgets()):
            if w is _CACHED_WINDOW:
                continue
            if isinstance(w, (sgui.FigureWindow, QtWidgets.QDialog)):
                try:
                    w.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
                    w.hide()
                    w.setParent(None)
                    w.deleteLater()
                except Exception:
                    pass
        # Process deferred deletions to actually free the resources
        try:
            sgui._flush_deferred_deletions()
            app.processEvents()
        except Exception:
            pass
    gc.collect()


@pytest.fixture(scope="module")
def shared_window(qapp: QtWidgets.QApplication) -> sgui.ScepterMainWindow:
    """Module-scoped main window shared across tests."""
    window = sgui.ScepterMainWindow()
    yield window
    window._dirty = False


def _make_viewer(
    *,
    state_provider,
    auto_build: bool = False,
    plotter_cls: type[QtWidgets.QWidget] = _DummyPlotter,
) -> sgui.ConstellationViewerWindow:
    return sgui.ConstellationViewerWindow(
        state_provider=state_provider,
        off_screen=True,
        auto_build=auto_build,
        plotter_factory=lambda parent, offscreen: plotter_cls(parent),
    )


def _wait_for(ms: int) -> None:
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _wait_until(predicate, *, timeout_ms: int = 1000, step_ms: int = 20) -> None:
    remaining = int(timeout_ms)
    while remaining > 0:
        QtWidgets.QApplication.processEvents()
        if predicate():
            return
        _wait_for(min(step_ms, remaining))
        remaining -= step_ms
    QtWidgets.QApplication.processEvents()
    assert predicate()


def _thread_is_stopped(thread: QtCore.QThread) -> bool:
    try:
        return not bool(thread.isRunning())
    except RuntimeError:
        return True


def _wait_for_postprocess_render(
    widget: sgui.PostprocessStudioWidget,
    *,
    timeout_ms: int = 3000,
) -> None:
    _wait_until(lambda: widget._render_thread is None, timeout_ms=timeout_ms)


def _grab_widget_image(widget: QtWidgets.QWidget) -> QtGui.QImage:
    pixmap = widget.grab()
    assert pixmap.isNull() is False
    return pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)


def _mean_rgb_for_points(image: QtGui.QImage, points: list[tuple[int, int]]) -> np.ndarray:
    colors = []
    for x, y in points:
        x_pos = max(1, min(image.width() - 2, int(x)))
        y_pos = max(1, min(image.height() - 2, int(y)))
        color = image.pixelColor(x_pos, y_pos)
        colors.append((color.red(), color.green(), color.blue()))
    return np.asarray(colors, dtype=np.float64).mean(axis=0)


def _button_background_points(button: QtWidgets.QPushButton) -> list[tuple[int, int]]:
    width = max(button.width(), 24)
    height = max(button.height(), 24)
    return [
        (8, height // 3),
        (8, (2 * height) // 3),
        (width - 8, height // 3),
        (width - 8, (2 * height) // 3),
    ]


def _panel_margin_points(widget: QtWidgets.QWidget) -> list[tuple[int, int]]:
    width = max(widget.width(), 24)
    height = max(widget.height(), 24)
    return [
        (8, height // 3),
        (8, height // 2),
        (width - 8, height // 3),
        (width - 8, height // 2),
    ]


def _rgb_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.linalg.norm(lhs - rhs))


def _mark_hexgrid_preview_current(
    window: sgui.ScepterMainWindow,
    *,
    effective_cell_km: float = 79.141,
) -> None:
    if window._effective_hexgrid_cell_size_km(window.current_state()) is None:
        window._last_analyser_selected_cell_km = float(effective_cell_km)
    window._last_analyser_signature = window._analyser_signature(window.current_state())
    state = window.current_state()
    preview_signature = window._hexgrid_preview_signature(state)
    commit_signature = window._hexgrid_commit_signature(state)
    assert preview_signature is not None
    assert commit_signature is not None
    window._hexgrid_completed_signature = preview_signature
    window._hexgrid_completed_commit_signature = commit_signature
    window._hexgrid_completed_overlay_signature = window._hexgrid_overlay_signature(state)
    window._hexgrid_outdated = False
    window._update_simulation_page_indicators(window.current_state())
    window._update_guidance_panel(window.current_state())
    window._update_run_controls(window.current_state())


def _write_minimal_result_file_with_2d_preacc_beams(path: Path) -> None:
    _write_minimal_result_file(path)
    with h5py.File(path, "a") as h5:
        beams = h5["preaccumulated"]["beam_statistics"]
        for name, data in {
            "network_total_beams_over_time": np.asarray([[3, 3, 3]], dtype=np.int64),
            "visible_total_beams_over_time": np.asarray([[3, 3, 3]], dtype=np.int64),
            "beam_demand_over_time": np.asarray([[2, 2, 3]], dtype=np.int64),
        }.items():
            del beams[name]
            beams.create_dataset(name, data=data)


def _write_result_file_missing_configured_preacc_power(path: Path) -> None:
    _write_minimal_result_file(path)
    with h5py.File(path, "a") as h5:
        for family_name in (
            "prx_total_distribution",
            "epfd_distribution",
            "total_pfd_ras_distribution",
            "per_satellite_pfd_distribution",
            "prx_elevation_heatmap",
            "per_satellite_pfd_elevation_heatmap",
        ):
            h5.attrs[f"output_family_{family_name}_mode"] = "preaccumulated"
        preacc = h5["preaccumulated"]
        for group_name in (
            "prx_total_distribution",
            "epfd_distribution",
            "total_pfd_ras_distribution",
            "per_satellite_pfd_distribution",
            "prx_elevation_heatmap",
            "per_satellite_pfd_elevation_heatmap",
        ):
            if group_name in preacc:
                del preacc[group_name]
        row = h5["iter"]["iter_00000"]
        for dataset_name in (
            "Prx_total_W",
            "EPFD_W_m2",
            "PFD_total_RAS_STATION_W_m2",
            "PFD_per_sat_RAS_STATION_W_m2",
            "Prx_per_sat_RAS_STATION_W",
        ):
            if dataset_name in row:
                del row[dataset_name]


def _write_result_file_preacc_power_no_raw(path: Path) -> None:
    _write_minimal_result_file(path)
    with h5py.File(path, "a") as h5:
        for family_name in (
            "prx_total_distribution",
            "epfd_distribution",
            "total_pfd_ras_distribution",
            "prx_elevation_heatmap",
        ):
            h5.attrs[f"output_family_{family_name}_mode"] = "preaccumulated"
        row = h5["iter"]["iter_00000"]
        for dataset_name in (
            "Prx_total_W",
            "EPFD_W_m2",
            "PFD_total_RAS_STATION_W_m2",
            "PFD_per_sat_RAS_STATION_W_m2",
            "Prx_per_sat_RAS_STATION_W",
        ):
            if dataset_name in row:
                del row[dataset_name]


def _write_result_file_zero_preacc_leakage(path: Path) -> None:
    _write_minimal_result_file(path)
    with h5py.File(path, "a") as h5:
        h5.attrs["service_band_start_mhz"] = 2620.0
        h5.attrs["service_band_stop_mhz"] = 2690.0
        h5.attrs["ras_receiver_band_start_mhz"] = 2690.0
        h5.attrs["ras_receiver_band_stop_mhz"] = 2700.0
        h5.attrs["reuse_factor"] = 7
        h5.attrs["channel_groups_per_cell_cap"] = 1
        h5.attrs["channel_groups_per_cell"] = 1
        h5.attrs["max_groups_per_cell"] = 2
        prx = h5["preaccumulated"]["prx_total_distribution"]
        del prx["counts"]
        prx.create_dataset("counts", data=np.asarray([0, 0, 0], dtype=np.int64))
        prx.attrs["sample_count"] = 0
        const = h5["const"]
        const.create_dataset(
            "cell_spectral_leakage_factor_active",
            data=np.asarray([0.0, 0.0], dtype=np.float32),
        )
        const.create_dataset(
            "cell_group_spectral_leakage_factor_active",
            data=np.zeros((2, 1), dtype=np.float32),
        )
        const.create_dataset(
            "spectrum_slot_edges_mhz",
            data=np.asarray(np.linspace(2620.0, 2690.0, 15), dtype=np.float64),
        )
        const.create_dataset(
            "spectrum_slot_group_channel_index",
            data=np.asarray([[0], [1], [2], [3], [4], [5], [6]], dtype=np.int32),
        )
        const.create_dataset(
            "spectrum_slot_group_leakage_factor",
            data=np.zeros((7, 1), dtype=np.float32),
        )


def _patch_run_request_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ras_service_cell_active: bool,
    theta2_ids: np.ndarray | None = None,
    resolver_calls: list[dict[str, object]] | None = None,
) -> None:
    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_definitions": [
                {
                    "belt_name": "Tiny_1",
                    "num_sats_per_plane": 2,
                    "plane_count": 2,
                    "altitude": 525.0 * sgui.u.km,
                    "eccentricity": 0.0,
                    "inclination_deg": 53.0 * sgui.u.deg,
                    "argp_deg": 0.0 * sgui.u.deg,
                    "RAAN_min": 0.0 * sgui.u.deg,
                    "RAAN_max": 360.0 * sgui.u.deg,
                    "min_elevation": 20.0 * sgui.u.deg,
                    "adjacent_plane_offset": True,
                }
            ],
            "belt_names": ["Tiny_1"],
            "tle_list": ["TLE1", "TLE2"],
            "altitudes_q": np.asarray([525.0]) * sgui.u.km,
            "inclinations_q": np.asarray([53.0]) * sgui.u.deg,
            "min_elevations_q": np.asarray([20.0]) * sgui.u.deg,
            "max_betas_q": np.asarray([35.0]) * sgui.u.deg,
            "belt_sats": np.asarray([2], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        sgui.tleforger,
        "summarize_constellation_geometry",
        lambda constellation: {
            "summary_lines": ["Tiny summary"],
            "slant_range_abs_per_belt": np.asarray([1.0], dtype=np.float64) * sgui.u.km,
            "slant_range_oper_per_belt": np.asarray([1.0], dtype=np.float64) * sgui.u.km,
            "slant_distance_abs_max": 1.0 * sgui.u.km,
            "slant_distance_max": 1.0 * sgui.u.km,
        },
    )
    monkeypatch.setattr(
        sgui,
        "_satellite_antenna_pattern_spec",
        lambda antennas: (object(), 0.028 * sgui.u.m, {"use_numba": False}),
    )
    monkeypatch.setattr(
        sgui.earthgrid,
        "summarize_contour_spacing",
        lambda *args, **kwargs: {
            "selected_cell_spacing_km": 79.141,
            "spacing_theta_edge": 3.0 * sgui.u.deg,
            "summary_lines": ["Analyser"],
        },
    )

    def _prepare_active_grid(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "pre_ras_cell_longitudes": np.asarray([21.0, 21.5], dtype=np.float64) * sgui.u.deg,
            "pre_ras_cell_latitudes": np.asarray([-30.0, -30.5], dtype=np.float64) * sgui.u.deg,
            "active_grid_longitudes": np.asarray([21.0, 21.5], dtype=np.float64) * sgui.u.deg,
            "active_grid_latitudes": np.asarray([-30.0, -30.5], dtype=np.float64) * sgui.u.deg,
            "pre_ras_to_active": np.asarray([0, 1], dtype=np.int32),
            "ras_service_cell_active": bool(ras_service_cell_active),
            "ras_service_cell_index": 0,
            "ras_service_cell_index_pre_ras": 0,
            "station_lon": 21.443611 * sgui.u.deg,
            "station_lat": -30.712777 * sgui.u.deg,
            "point_spacing_km": 79.141,
            "active_cell_count": 2,
            "pre_ras_cell_count": 2,
            "geography_kept_cell_count": 2,
            "geography_excluded_cell_count": 0,
            "ras_excluded_cell_count": 0,
        }

    monkeypatch.setattr(sgui.earthgrid, "prepare_active_grid", _prepare_active_grid)

    def _resolve_theta2_active_cell_ids(*args: object, **kwargs: object) -> np.ndarray:
        del args
        if resolver_calls is not None:
            resolver_calls.append(dict(kwargs))
        return (
            np.asarray([1], dtype=np.int32)
            if theta2_ids is None
            else np.asarray(theta2_ids, dtype=np.int32)
        )

    monkeypatch.setattr(
        sgui.earthgrid,
        "resolve_theta2_active_cell_ids",
        _resolve_theta2_active_cell_ids,
    )
    monkeypatch.setattr(
        sgui.earthgrid,
        "resolve_frequency_reuse_slots",
        lambda prepared_grid, *, reuse_factor, anchor_slot=0: {
            "reuse_factor": int(reuse_factor),
            "anchor_slot": int(anchor_slot),
            "anchor_pre_ras_index": 0,
            "anchor_active_index": 0,
            "point_spacing_km_used": 79.141,
            "orientation_used": "pointy",
            "fit_residual_km2": 0.0,
            "axial_q_pre_ras": np.asarray([0, 1], dtype=np.int32),
            "axial_r_pre_ras": np.asarray([0, 0], dtype=np.int32),
            "pre_ras_slot_ids": np.asarray([0, 0], dtype=np.int32),
            "active_slot_ids": np.asarray([0, 0], dtype=np.int32),
            "cluster_representatives": [
                {"slot_id": 0, "base_slot_id": 0, "axial_q": 0, "axial_r": 0}
            ],
        },
    )
    monkeypatch.setattr(
        sgui.scenario,
        "build_observer_layout",
        lambda ras_observer, active_cell_observers: {
            "observer_arr": np.asarray([ras_observer] + list(active_cell_observers), dtype=object)
        },
    )
    monkeypatch.setattr(
        sgui.tleforger,
        "expand_belt_metadata_to_satellites",
        lambda constellation: {"satellite_count": 2},
    )
    monkeypatch.setattr(
        sgui.tleforger,
        "build_satellite_storage_constants",
        lambda satellite_metadata, orbit_radius_m_per_sat: {
            "sat_min_elev_deg_per_sat": np.asarray([20.0, 20.0], dtype=np.float32),
            "sat_beta_max_deg_per_sat": np.asarray([35.0, 35.0], dtype=np.float32),
            "sat_belt_id_per_sat": np.asarray([0, 0], dtype=np.int32),
        },
    )


def test_json_roundtrip_current_schema_empty_workspace(tmp_path: Path) -> None:
    state = sgui.ScepterProjectState()
    path = tmp_path / "gui_empty.json"
    sgui.save_project_state(path, state)
    payload = path.read_text(encoding="utf-8")
    assert f'"schema_version": {sgui.GUI_CONFIG_SCHEMA_VERSION}' in payload
    assert '"base_start_utc_iso"' in payload
    assert '"gpu_method"' in payload
    assert '"progress_desc_mode"' in payload
    assert '"writer_checkpoint_interval_s"' in payload
    assert '"service"' in payload
    assert '"runtime"' in payload
    loaded = sgui.load_project_state(path)
    assert loaded.to_json_dict() == state.to_json_dict()


def test_json_roundtrip_with_antennas_and_grid(tmp_path: Path) -> None:
    state = _tiny_state()
    state.active_system().service.bandwidth_mhz = 7.5
    state.active_system().service.cell_activity_mode = "per_channel"
    state.active_system().service.power_input_quantity = "satellite_eirp"
    state.active_system().service.power_input_basis = "per_channel"
    state.active_system().service.target_pfd_dbw_m2_mhz = -87.25
    state.active_system().service.satellite_eirp_dbw_channel = 44.0
    state.active_system().spectrum.service_band_start_mhz = 2620.0
    state.active_system().spectrum.service_band_stop_mhz = 2690.0
    state.active_system().spectrum.reuse_factor = 7
    state.active_system().spectrum.disabled_channel_indices = [0, 13]
    state.active_system().spectrum.ras_anchor_reuse_slot = 3
    state.active_system().spectrum.multi_group_power_policy = "split_total_cell_power"
    state.active_system().spectrum.split_total_group_denominator_mode = "active_groups"
    state.active_system().spectrum.unwanted_emission_mask_preset = "sm1541_mss"
    state.ras_station.receiver_band_start_mhz = 2690.0
    state.ras_station.receiver_band_stop_mhz = 2700.0
    state.ras_station.ras_reference_mode = "lower"
    state.active_system().spectrum.custom_mask_points = [[-12.5, 55.0], [-3.75, 0.0], [3.75, 0.0], [12.5, 40.0]]
    state.active_system().satellite_antennas.rec14.gm_dbi = 36.4
    path = tmp_path / "gui_full.json"
    sgui.save_project_state(path, state)
    loaded = sgui.load_project_state(path)
    assert loaded.to_json_dict() == state.to_json_dict()


def test_json_invalid_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema_version": 5, "constellation": {"belts": []}, "ras_station": null, "antennas": null, "grid_analysis": {"indicative_footprint_drop": "db3", "spacing_drop": "db7", "leading_metric": "spacing_contour", "cell_spacing_rule": "full_footprint_diameter", "cell_size_override_enabled": false, "cell_size_override_km": null}, "hexgrid": {"geography_mask_mode": "none", "shoreline_buffer_km": null, "coastline_backend": "cartopy", "ras_pointing_mode": "ras_station", "ras_exclusion_mode": "none", "ras_exclusion_layers": 0, "ras_exclusion_radius_km": null, "boresight_avoidance_enabled": false, "boresight_theta1_deg": null, "boresight_theta2_deg": null, "boresight_theta2_scope_mode": "cell_ids", "boresight_theta2_cell_ids": null, "boresight_theta2_layers": 0, "boresight_theta2_radius_km": null}, "service": {"nco": 1, "nbeam": 97, "selection_strategy": "max_elevation", "cell_activity_factor": 1.0, "cell_activity_seed_base": 42001}, "runtime": {"storage_filename": "x.h5", "iteration_rng_seed": 42, "write_epfd": true, "write_prx_total": true, "write_total_pfd_ras_station": true, "write_per_satellite_pfd_ras_station": true, "write_sat_beam_counts_used": true, "write_sat_elevation_ras_station": true, "write_beam_demand_count": true, "atmosphere_elev_bin_deg": 0.1, "atmosphere_elev_min_deg": 0.1, "atmosphere_elev_max_deg": 90.0, "atmosphere_max_path_length_km": 10000.0, "target_alt_km": 0.0, "use_ras_station_alt_for_co": true, "host_memory_budget_gb": 4.0, "gpu_memory_budget_gb": 4.0, "memory_budget_mode": "hybrid", "memory_headroom_profile": "balanced", "terminal_gpu_cleanup": false}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        sgui.load_project_state(path)


def test_belt_table_model_unique_names_and_remove_last_row() -> None:
    model = sgui.BeltTableModel()
    model.add_default_belt()
    model.add_default_belt()
    assert model.belts()[0].belt_name == "System3_Belt_1"
    assert model.belts()[1].belt_name == "System3_Belt_2"
    model.duplicate_row(0)
    assert model.belts()[1].belt_name == "System3_Belt_3"
    name_index = model.index(1, 0)
    assert model.setData(name_index, "Custom")
    assert model.belts()[1].belt_name == "Custom"
    assert model.setData(name_index, "System3_Belt_1")
    assert model.belts()[1].belt_name == "System3_Belt_3"
    model.remove_row(2)
    model.remove_row(1)
    model.remove_row(0)
    assert model.rowCount() == 0


def test_validate_project_state_variants() -> None:
    empty_state = sgui.ScepterProjectState()
    is_valid, summary, constellation = sgui.validate_project_state(empty_state)
    assert is_valid is False
    assert constellation is None
    assert "Nco must be positive" in summary

    no_ras_state = _tiny_state(include_ras=False)
    is_valid, summary, constellation = sgui.validate_project_state(no_ras_state)
    assert is_valid is True
    assert constellation is not None
    assert "None configured" in summary

    bad_antennas_state = _tiny_state()
    bad_antennas_state.ras_antenna.operational_elevation_min_deg = 90.0
    bad_antennas_state.ras_antenna.operational_elevation_max_deg = 15.0
    is_valid, summary, constellation = sgui.validate_project_state(bad_antennas_state)
    assert is_valid is False
    assert constellation is None
    assert "operational elevation range" in summary


def test_compute_earth_texture_coordinates_matches_geodetic_convention() -> None:
    radius = float(sgui._EARTH_RADIUS_KM)
    points = np.asarray(
        (
            (radius, 0.0, 0.0),
            (0.0, radius, 0.0),
            (0.0, 0.0, radius),
            (-radius, 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    uv_coords = sgui._compute_earth_texture_coordinates(points)
    np.testing.assert_allclose(uv_coords[0], np.asarray((0.5, 0.5), dtype=np.float32))
    np.testing.assert_allclose(uv_coords[1], np.asarray((0.75, 0.5), dtype=np.float32))
    np.testing.assert_allclose(uv_coords[2], np.asarray((0.5, 1.0), dtype=np.float32))
    assert float(uv_coords[3, 0]) > 0.99
    assert float(uv_coords[3, 1]) == pytest.approx(0.5)
    anti_meridian_points = sgui._geodetic_to_cartesian_km(
        np.asarray([179.5, -179.5], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
    )
    anti_meridian_uv = sgui._compute_earth_texture_coordinates(anti_meridian_points)
    seam_gap = float(abs(anti_meridian_uv[0, 0] - anti_meridian_uv[1, 0]))
    assert min(seam_gap, 1.0 - seam_gap) < 0.01


def test_build_preview_frames_shape() -> None:
    state = _tiny_state()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    params = sgui.PreviewParameters(
        start_utc=start,
        end_utc=start + timedelta(seconds=10.0),
        frame_step_s=5.0,
        playback_fps=30.0,
    )
    frame_set = sgui.build_preview_frames(state, params)
    assert frame_set.positions_ecef_km.shape == (3, 4, 3)
    assert frame_set.positions_eci_km.shape == (3, 4, 3)
    assert frame_set.velocities_eci_km_s.shape == (3, 4, 3)
    assert frame_set.sample_times_s.tolist() == pytest.approx([0.0, 5.0, 10.0])


@pytest.mark.parametrize(
    ("case_name", "belt_count", "ras_mode", "expected_ras", "expected_note"),
    [
        ("blank", 0, "none", False, "Preview shows Earth only."),
        (
            "blank_with_ras",
            0,
            "complete",
            True,
            "Preview shows Earth and RAS station only because no belts are defined.",
        ),
        ("sats_only", 1, "none", False, ""),
        ("sats_plus_ras", 1, "complete", True, ""),
        (
            "incomplete_ras",
            1,
            "incomplete",
            False,
            "RAS station is hidden because longitude, latitude, and elevation are incomplete.",
        ),
    ],
)
def test_build_preview_frames_state_matrix(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    belt_count: int,
    ras_mode: str,
    expected_ras: bool,
    expected_note: str,
) -> None:
    del case_name
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    params = sgui.PreviewParameters(
        start_utc=start,
        end_utc=start + timedelta(seconds=10.0),
        frame_step_s=5.0,
        playback_fps=30.0,
    )
    state = _tiny_state(include_ras=ras_mode != "none")
    if belt_count == 0:
        state.active_system().belts = []
    if ras_mode == "incomplete":
        state.ras_station = sgui.RasStationConfig(
            longitude_deg=21.443611,
            latitude_deg=None,
            elevation_m=1052.0,
        )

    _stub_preview_propagation(monkeypatch)
    if belt_count == 0:
        monkeypatch.setattr(
            sgui,
            "build_constellation_from_state",
            lambda state, **kw: pytest.fail("build_constellation_from_state should not run for blank previews"),
        )
    else:
        monkeypatch.setattr(
            sgui,
            "build_constellation_from_state",
            lambda state, **kw: {
                "belt_names": ["Tiny_1"],
                "belt_sats": np.asarray([4], dtype=np.int64),
                "tle_list": ["TLE1", "TLE2", "TLE3", "TLE4"],
            },
        )

    frame_set = sgui.build_preview_frames(state, params)

    assert frame_set.positions_ecef_km.shape == (3, belt_count * 4, 3)
    assert frame_set.positions_eci_km.shape == (3, belt_count * 4, 3)
    assert frame_set.velocities_eci_km_s.shape == (3, belt_count * 4, 3)
    assert frame_set.sample_times_s.tolist() == pytest.approx([0.0, 5.0, 10.0])
    if expected_ras:
        assert frame_set.ras_position_ecef_km is not None
    else:
        assert frame_set.ras_position_ecef_km is None
    assert frame_set.status_note == expected_note


def test_build_preview_frames_invalid_constellation_without_ras(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _tiny_state(include_ras=False)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    params = sgui.PreviewParameters(
        start_utc=start,
        end_utc=start + timedelta(seconds=10.0),
        frame_step_s=5.0,
        playback_fps=30.0,
    )
    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: (_ for _ in ()).throw(ValueError("broken constellation")),
    )

    frame_set = sgui.build_preview_frames(state, params)

    assert frame_set.positions_ecef_km.shape == (3, 0, 3)
    assert frame_set.ras_position_ecef_km is None
    assert frame_set.status_note == "Preview shows Earth only; constellation is invalid: broken constellation"


def test_build_preview_frames_invalid_constellation_with_ras(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _tiny_state(include_ras=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    params = sgui.PreviewParameters(
        start_utc=start,
        end_utc=start + timedelta(seconds=10.0),
        frame_step_s=5.0,
        playback_fps=30.0,
    )
    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: (_ for _ in ()).throw(ValueError("broken constellation")),
    )

    frame_set = sgui.build_preview_frames(state, params)

    assert frame_set.positions_ecef_km.shape == (3, 0, 3)
    assert frame_set.ras_position_ecef_km is not None
    assert frame_set.status_note == (
        "Preview shows Earth and RAS station only; constellation is invalid: broken constellation"
    )


def test_interpolate_position_frames_midpoint() -> None:
    frame_set = _manual_frame_set()
    out = sgui._interpolate_position_frames(
        frame_set.sample_times_s,
        frame_set.positions_ecef_km,
        2.5,
    )
    np.testing.assert_allclose(
        out,
        0.5 * (frame_set.positions_ecef_km[0] + frame_set.positions_ecef_km[1]),
    )


def test_fast_vs_detailed_mesh_assets_differ() -> None:
    fast_mesh = sgui._build_satellite_mesh("Fast")
    detailed_mesh = sgui._build_satellite_mesh("Detailed")
    assert fast_mesh.n_cells != detailed_mesh.n_cells
    fast_ras = sgui._build_ras_station_mesh("Fast")
    detailed_ras = sgui._build_ras_station_mesh("Detailed")
    assert fast_ras.n_cells != detailed_ras.n_cells


def test_viewer_scene_controls_and_selection_overlay(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    skyboxes = _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_resolve_contour_half_angle_deg", lambda *args, **kwargs: 5.0)
    viewer = _make_viewer(state_provider=_tiny_state)
    plotter = viewer.plotter
    assert isinstance(plotter, _DummyPlotter)
    viewer.show()
    qapp.processEvents()

    viewer._current_frame_set = _manual_frame_set()
    viewer._applied_frame_view_mode = "ECEF"
    viewer._applied_render_mode = "Fast"
    viewer._applied_skybox_mode = "Off"
    viewer._rebuild_scene(reset_camera=False)

    assert len(viewer._belt_render_bundles) == 1
    viewer.frame_view_combo.setCurrentText("ECI")
    viewer.render_mode_combo.setCurrentText("Detailed")
    viewer.skybox_combo.setCurrentText("4K")
    viewer.apply_preview_settings(force_frame_rebuild=False)
    assert viewer._applied_frame_view_mode == "ECI"
    assert viewer._applied_render_mode == "Detailed"
    assert viewer._applied_skybox_mode == "4K"
    assert plotter.environment_texture is skyboxes["4K"]
    assert sgui._SKYBOX_ACTOR_NAME in plotter.actors

    viewer._set_selected_satellite(1)
    assert viewer.situational_group.isHidden() is False
    assert sgui._SELECTED_SAT_ACTOR_NAME in plotter.actors

    viewer.selected_beta_spin.setValue(0.0)
    assert viewer.selected_alpha_spin.value() == pytest.approx(0.0)
    viewer._set_selected_satellite(None)
    assert viewer.situational_group.isVisible() is False
    viewer.close()


@pytest.mark.parametrize(
    ("state", "expected_belts", "expect_ras", "status_fragment"),
    [
        (sgui.ScepterProjectState(), 0, False, "Earth only"),
        (_ras_only_state(), 0, True, "RAS station only"),
        (_tiny_state(include_ras=False), 1, False, None),
        (_tiny_state(), 1, True, None),
        (_tiny_state(include_ras=False, include_antennas=False), 1, False, None),
        (_sats_with_incomplete_ras_state(), 1, False, "RAS station is hidden"),
        (_invalid_constellation_state(include_ras=True), 0, True, "constellation is invalid"),
    ],
)
def test_viewer_scene_matrix_rebuilds_without_errors(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    state: sgui.ScepterProjectState,
    expected_belts: int,
    expect_ras: bool,
    status_fragment: str | None,
) -> None:
    _stub_scene_assets(monkeypatch)
    viewer = _make_viewer(state_provider=lambda: state)
    viewer.show()
    qapp.processEvents()

    viewer._current_frame_set = sgui.build_preview_frames(state, _small_preview_params())
    viewer._rebuild_scene(reset_camera=False)
    viewer._set_status("Preview ready")

    assert len(viewer._belt_render_bundles) == expected_belts
    assert (sgui._RAS_ACTOR_NAME in viewer.plotter.actors) is expect_ras
    if status_fragment is not None:
        assert status_fragment in viewer.status_label.text()
    viewer.close()


def test_continuous_update_and_playback_rate_do_not_force_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer._rebuild_debounce_timer.stop()
    viewer.continuous_update_checkbox.setChecked(True)
    viewer.playback_fps_spin.setValue(24.0)
    assert viewer._rebuild_debounce_timer.isActive() is False
    viewer.frame_view_combo.setCurrentText("ECI")
    assert viewer._rebuild_debounce_timer.isActive() is True

    apply_calls: list[bool] = []
    monkeypatch.setattr(
        viewer,
        "apply_preview_settings",
        lambda force_frame_rebuild=False: apply_calls.append(bool(force_frame_rebuild)),
    )
    viewer._apply_debounced_preview_settings()
    assert apply_calls == [False]
    viewer.close()


def test_viewer_scene_rebuild_is_safe_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_resolve_contour_half_angle_deg", lambda *args, **kwargs: 5.0)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer.show()
    qapp.processEvents()
    viewer._current_frame_set = _manual_frame_set()
    viewer._applied_frame_view_mode = "ECEF"
    viewer._applied_render_mode = "Fast"
    viewer._applied_skybox_mode = "Off"
    viewer._rebuild_scene(reset_camera=False)

    first_actor_names = sorted(viewer.plotter.actors)
    first_bundle_count = len(viewer._belt_render_bundles)
    first_render_calls = viewer.plotter.render_calls

    viewer._rebuild_scene(reset_camera=False)

    assert len(viewer._belt_render_bundles) == first_bundle_count
    assert sorted(viewer.plotter.actors) == first_actor_names
    assert viewer.plotter.render_calls >= first_render_calls
    viewer.close()


def test_viewer_preview_settings_only_rebuilds_what_changed(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_resolve_contour_half_angle_deg", lambda *args, **kwargs: 5.0)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer._current_frame_set = _manual_frame_set()
    viewer._applied_frame_view_mode = "ECEF"
    viewer._applied_render_mode = "Fast"
    viewer._applied_skybox_mode = "Off"
    viewer._preview_params = viewer._current_preview_parameters()

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        viewer,
        "_rebuild_scene",
        lambda *, reset_camera: calls.append(("rebuild", bool(reset_camera))),
    )
    monkeypatch.setattr(
        viewer,
        "_render_at_offset_s",
        lambda offset_s: calls.append(("render", float(offset_s))),
    )

    viewer.render_mode_combo.setCurrentText("Detailed")
    viewer.apply_preview_settings(force_frame_rebuild=False)
    assert calls == [("rebuild", False)]
    assert viewer._applied_render_mode == "Detailed"
    calls.clear()

    viewer.frame_view_combo.setCurrentText("ECI")
    viewer.apply_preview_settings(force_frame_rebuild=False)
    assert calls == [("render", 0.0)]
    assert viewer._applied_frame_view_mode == "ECI"
    viewer.close()


def test_viewer_runtime_hints_keep_update_rate_at_least_sixty(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer.show()
    qapp.processEvents()

    assert viewer._timer.timerType() == QtCore.Qt.TimerType.PreciseTimer
    assert viewer.plotter.render_window.desired_update_rate == pytest.approx(60.0)

    viewer.playback_fps_spin.setValue(24.0)
    assert viewer.plotter.render_window.desired_update_rate == pytest.approx(60.0)

    viewer.playback_fps_spin.setValue(90.0)
    assert viewer.plotter.render_window.desired_update_rate == pytest.approx(90.0)
    viewer.close()


def test_viewer_update_rate_hint_tracks_live_playback(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_resolve_contour_half_angle_deg", lambda *args, **kwargs: 5.0)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer._current_frame_set = _manual_frame_set()
    viewer._rebuild_scene(reset_camera=False)
    viewer._timer.start(1000)
    viewer._last_render_wall_s = 99.0
    viewer._last_playback_stats_refresh_wall_s = 0.0
    monkeypatch.setattr(sgui, "perf_counter", lambda: 99.25)

    viewer._render_at_offset_s(2.5)

    assert viewer._rolling_playback_fps == pytest.approx(4.0)
    assert viewer.stats_fps_label.text() == "Rolling FPS: 4.0"
    viewer._timer.stop()
    viewer.close()


def test_viewer_selected_satellite_out_of_range_clears_safely(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer.show()
    qapp.processEvents()
    viewer._current_frame_set = _manual_frame_set()
    viewer._rebuild_scene(reset_camera=False)

    viewer._set_selected_satellite(999)

    assert viewer._selected_satellite_index is None
    assert viewer.situational_group.isVisible() is False
    assert viewer.selected_satellite_label.text() == "None"
    assert sgui._SELECTED_SAT_ACTOR_NAME not in viewer.plotter.actors
    viewer.close()


def test_viewer_selected_satellite_clears_when_new_frame_set_is_too_small(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_resolve_contour_half_angle_deg", lambda *args, **kwargs: 5.0)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer._current_frame_set = _manual_frame_set()
    viewer._selected_satellite_index = 3
    viewer._rebuild_scene(reset_camera=False)
    viewer._current_frame_set = _manual_frame_set_with_sat_count(2)

    monkeypatch.setattr(viewer, "_rebuild_scene", lambda *, reset_camera: None)
    monkeypatch.setattr(viewer, "_safe_render", lambda: None)

    viewer._on_preview_built(viewer._current_frame_set, token=viewer._active_build_token, build_started=0.0)

    assert viewer._selected_satellite_index is None
    assert viewer.situational_group.isVisible() is False
    assert viewer.selected_satellite_label.text() == "None"
    viewer.close()


def test_viewer_stale_preview_token_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_resolve_contour_half_angle_deg", lambda *args, **kwargs: 5.0)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer._current_frame_set = _manual_frame_set()
    viewer._selected_satellite_index = 1
    viewer._active_build_token = 7
    viewer._rebuild_scene(reset_camera=False)
    current_actor_names = sorted(viewer.plotter.actors)
    current_frame_set = viewer._current_frame_set

    rebuild_calls: list[bool] = []
    monkeypatch.setattr(
        viewer,
        "_rebuild_scene",
        lambda *, reset_camera: rebuild_calls.append(bool(reset_camera)),
    )
    viewer._on_preview_built(_manual_frame_set_with_sat_count(2), token=6, build_started=0.0)

    assert rebuild_calls == []
    assert viewer._current_frame_set is current_frame_set
    assert sorted(viewer.plotter.actors) == current_actor_names
    assert viewer._selected_satellite_index == 1
    viewer.close()


def test_viewer_close_while_preview_build_active_detaches_thread_and_suppresses_warning(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    started = threading.Event()
    cancelled = threading.Event()
    sleeper = threading.Event()
    warning_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _slow_build(
        state: sgui.ScepterProjectState,
        parameters: sgui.PreviewParameters,
        *,
        cancel_callback=None,
    ) -> sgui.PreviewFrameSet:
        del state, parameters
        started.set()
        while not cancelled.is_set():
            if callable(cancel_callback) and cancel_callback():
                cancelled.set()
                raise sgui._PreviewBuildCancelledError("Preview build cancelled during test.")
            sleeper.wait(0.01)
        raise AssertionError("Preview build test loop should exit via cancellation.")

    monkeypatch.setattr(sgui, "build_preview_frames", _slow_build)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warning_calls.append((args, kwargs)) or QtWidgets.QMessageBox.Ok,
    )

    viewer = _make_viewer(state_provider=_tiny_state)
    viewer.show()
    qapp.processEvents()
    viewer.apply_preview_settings(force_frame_rebuild=True)
    _wait_until(lambda: started.is_set() and viewer._build_thread is not None, timeout_ms=1000)

    thread = viewer._build_thread
    assert thread is not None
    assert thread.parent() is None

    viewer.close()
    qapp.processEvents()

    assert viewer._build_thread is None
    _wait_until(lambda: cancelled.is_set() or not thread.isRunning(), timeout_ms=2000)
    assert warning_calls == []


def test_viewer_reopen_rebuild_cycle_handles_mutating_state_provider(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    state_box = {"state": sgui.ScepterProjectState()}
    viewer = _make_viewer(state_provider=lambda: state_box["state"])
    viewer.show()
    qapp.processEvents()

    viewer._current_frame_set = sgui.build_preview_frames(state_box["state"], _small_preview_params())
    viewer._rebuild_scene(reset_camera=False)
    assert viewer._current_frame_set is not None
    assert viewer._current_frame_set.positions_ecef_km.shape[1] == 0
    assert viewer._current_frame_set.ras_position_ecef_km is None

    state_box["state"] = _tiny_state()
    viewer._current_frame_set = sgui.build_preview_frames(state_box["state"], _small_preview_params())
    viewer._rebuild_scene(reset_camera=False)
    assert viewer._current_frame_set is not None
    assert viewer._current_frame_set.positions_ecef_km.shape[1] == 4
    assert viewer._current_frame_set.ras_position_ecef_km is not None

    state_box["state"] = _tiny_state(include_ras=False, include_antennas=False)
    viewer._current_frame_set = sgui.build_preview_frames(state_box["state"], _small_preview_params())
    viewer._rebuild_scene(reset_camera=False)
    assert viewer._current_frame_set is not None
    assert viewer._current_frame_set.positions_ecef_km.shape[1] == 4
    assert viewer._current_frame_set.ras_position_ecef_km is None
    viewer.close()


def test_viewer_playback_advances_without_full_scene_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_resolve_contour_half_angle_deg", lambda *args, **kwargs: 5.0)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer._current_frame_set = _manual_frame_set()
    viewer._rebuild_scene(reset_camera=False)
    mesh_call_count = len(viewer.plotter.mesh_calls)
    render_call_count = viewer.plotter.render_calls
    rebuild_calls: list[bool] = []
    monkeypatch.setattr(
        viewer,
        "_rebuild_scene",
        lambda *, reset_camera: rebuild_calls.append(bool(reset_camera)),
    )
    viewer._timer.start(1000)
    viewer._playback_anchor_offset_s = 0.0
    viewer._playback_anchor_wall_s = 100.0
    viewer._playback_multiplier = 1.0
    monkeypatch.setattr(sgui, "perf_counter", lambda: 100.25)

    viewer._advance_frame()

    assert rebuild_calls == []
    assert len(viewer.plotter.mesh_calls) == mesh_call_count
    assert viewer.plotter.render_calls >= render_call_count
    viewer._timer.stop()
    viewer.close()


def test_safe_render_disables_repeated_render_attempts(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    viewer = _make_viewer(state_provider=_tiny_state, plotter_cls=_FailingRenderPlotter)
    assert isinstance(viewer.plotter, _FailingRenderPlotter)
    viewer.show()
    qapp.processEvents()
    viewer._safe_render()
    first_calls = viewer.plotter.render_calls
    assert first_calls == 1
    assert viewer._render_failed is True
    viewer._safe_render()
    assert viewer.plotter.render_calls == first_calls
    viewer.close()


def test_main_window_empty_defaults_antennas_and_grid(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    state = window.current_state()
    assert state.active_system().belts == []
    assert state.ras_station is None
    assert state.active_system().satellite_antennas.antenna_model is None
    assert state.active_system().service.nco is None
    assert state.active_system().service.target_pfd_dbw_m2_mhz is None
    assert state.active_system().grid_analysis.indicative_footprint_drop == "db3"
    assert window.run_grid_analyzer_button.isEnabled() is False
    assert window.longitude_spin.text() == ""
    assert window.ras_frequency_spin.text() == ""
    assert window.frequency_spin.text() == ""
    ras_idx = window.tab_widget.indexOf(window.ras_tab)
    antennas_idx = window.tab_widget.indexOf(window.antennas_tab)
    assert window.tab_widget.widget(window.tab_widget.count() - 1) is window.runtime_tab
    assert window.run_simulation_button.isEnabled() is False
    window.tab_widget.setCurrentIndex(ras_idx)
    qapp.processEvents()
    window._show_context_toast_for_current_tab()
    assert "operational elevation range" in window.ras_notice_banner._label.text().lower()
    window.tab_widget.setCurrentIndex(antennas_idx)
    qapp.processEvents()
    window._show_context_toast_for_current_tab()
    _ant_notice = window.antennas_notice_banner._label.text().lower()
    assert "antenna" in _ant_notice, f"Expected 'antenna' in notice banner text: {_ant_notice!r}"
    hex_idx = window.tab_widget.indexOf(window.hexgrid_tab)
    window.tab_widget.setCurrentIndex(hex_idx)
    qapp.processEvents()
    window._show_context_toast_for_current_tab()
    assert "at least one valid belt" in window.hexgrid_notice_banner._label.text().lower()
    assert not hasattr(window, "summary_text")
    overview = window.snapshot_overview_label.text().lower()
    assert "workflow" in overview
    assert "guidance next" in overview
    step_text = "\n".join(
        window.snapshot_step_list.item(idx).text().lower()
        for idx in range(window.snapshot_step_list.count())
    )
    assert "service & demand: needs attention" in step_text
    assert "getting started" in window.snapshot_chip.text().lower()
    assert window.service_nco_edit.text() == ""
    assert window.runtime_storage_edit.text().endswith(".h5")

    window._add_belt()
    window._add_ras_station()
    window._set_all_antenna_defaults()
    window._set_service_defaults()
    updated = window.current_state()
    assert len(updated.active_system().belts) == 1
    assert updated.ras_station is not None
    assert updated.active_system().satellite_antennas.antenna_model == "s1528_rec1_4"
    assert updated.ras_antenna.antenna_diameter_m == pytest.approx(15.0)
    window._refresh_summary()
    assert window.run_simulation_button.isEnabled() is False
    _mark_hexgrid_preview_current(window)
    window._refresh_summary()
    assert window.run_simulation_button.isEnabled() is True
    window._dirty = False; window.close()


def test_runtime_controls_toggle_advanced_widgets(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    assert window.runtime_include_atmosphere_checkbox.isChecked() is True
    assert window.runtime_atm_bin_edit.isEnabled() is True
    window.runtime_include_atmosphere_checkbox.setChecked(False)
    qapp.processEvents()
    assert window.runtime_atm_bin_edit.isEnabled() is False
    assert window.runtime_atm_min_edit.isEnabled() is False
    assert window.runtime_atm_max_edit.isEnabled() is False
    assert window.runtime_atm_path_edit.isEnabled() is False

    off_idx = window.runtime_hdf5_compression_combo.findData(None)
    gzip_idx = window.runtime_hdf5_compression_combo.findData("gzip")
    window.runtime_hdf5_compression_combo.setCurrentIndex(off_idx)
    qapp.processEvents()
    assert window.runtime_hdf5_compression_opts_spin.isEnabled() is False
    window.runtime_hdf5_compression_combo.setCurrentIndex(gzip_idx)
    qapp.processEvents()
    assert window.runtime_hdf5_compression_opts_spin.isEnabled() is True

    assert window.runtime_sat_frame_combo.currentData() == "xyz"
    assert window.runtime_sat_frame_combo.isEnabled() is False
    window._dirty = False; window.close()


def test_grid_analyzer_uses_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._add_belt()
    window._set_all_antenna_defaults()

    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_names": ["Tiny_1"],
            "altitudes_q": np.asarray([525.0]) * sgui.u.km,
            "min_elevations_q": np.asarray([20.0]) * sgui.u.deg,
            "max_betas_q": np.asarray([35.0]) * sgui.u.deg,
            "belt_sats": np.asarray([56], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        sgui.earthgrid,
        "summarize_contour_spacing",
        lambda *args, **kwargs: {
            "selected_cell_spacing_km": 79.141,
            "summary_lines": ["Analyser", "Recommended cell = 79.141 km"],
        },
    )
    monkeypatch.setattr(sgui, "validate_project_state", lambda state: (True, "ok", []))
    window._run_grid_analyzer()
    # The analyser runs on a background thread with QTimer polling;
    # process events until the label updates (max 2 seconds).
    import time as _time
    _deadline = _time.monotonic() + 2.0
    while "computing" in window.grid_recommended_label.text().lower():
        qapp.processEvents()
        if _time.monotonic() > _deadline:
            break
        _time.sleep(0.05)
    assert "79.141" in window.grid_recommended_label.text()
    assert "Analyser" in window.grid_summary_text.toPlainText()
    window._dirty = False; window.close()


def test_hexgrid_preview_uses_shared_helpers(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._add_belt()
    window._add_ras_station()
    window._set_all_antenna_defaults()
    clean_idx = window.hexgrid_map_style_combo.findData("clean")
    window.hexgrid_map_style_combo.setCurrentIndex(max(clean_idx, 0))
    window._set_combo_to_data(window.spectrum_reuse_factor_combo, 7)
    window._set_combo_to_data(window.spectrum_anchor_slot_spin, 2)
    window._last_analyser_selected_cell_km = 79.141
    window._last_analyser_signature = window._analyser_signature(window.current_state())
    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_names": ["Tiny_1"],
            "altitudes_q": np.asarray([525.0]) * sgui.u.km,
            "inclinations_q": np.asarray([53.0]) * sgui.u.deg,
            "min_elevations_q": np.asarray([20.0]) * sgui.u.deg,
            "belt_sats": np.asarray([56], dtype=np.int64),
        },
    )
    prepare_calls: list[float] = []
    plot_kwargs_seen: list[dict[str, object]] = []

    def _prepare_active_grid(**kwargs: object) -> dict[str, object]:
        prepare_calls.append(float(sgui.u.Quantity(kwargs["point_spacing"]).to_value(sgui.u.km)))
        return _manual_hexgrid_result()

    monkeypatch.setattr(
        sgui.earthgrid,
        "resolve_frequency_reuse_slots",
        lambda prepared_grid, *, reuse_factor, anchor_slot=0: {
            "reuse_factor": int(reuse_factor),
            "anchor_slot": int(anchor_slot),
            "anchor_pre_ras_index": 0,
            "anchor_active_index": 0,
            "point_spacing_km_used": 79.141,
            "orientation_used": "pointy",
            "fit_residual_km2": 0.0,
            "adjacent_same_slot_pair_count": 2,
            "active_adjacent_same_slot_pair_count": 2,
            "pre_ras_adjacent_same_slot_pair_count": 0,
            "axial_q_pre_ras": np.asarray([0, 1], dtype=np.int32),
            "axial_r_pre_ras": np.asarray([0, 0], dtype=np.int32),
            "pre_ras_slot_ids": np.asarray([2, 0], dtype=np.int32),
            "active_slot_ids": np.asarray([2, 0], dtype=np.int32),
            "cluster_representatives": [
                {"slot_id": 0, "base_slot_id": 0, "axial_q": 0, "axial_r": 0}
            ],
        },
    )
    monkeypatch.setattr(sgui.earthgrid, "prepare_active_grid", _prepare_active_grid)
    monkeypatch.setattr(
        sgui.earthgrid,
        "resolve_theta2_active_cell_ids",
        lambda *args, **kwargs: np.asarray([1], dtype=np.int32),
    )

    def _plot_cell_status_map(*args: object, **kwargs: object) -> tuple[Figure, dict[str, object]]:
        del args
        plot_kwargs_seen.append(dict(kwargs))
        fig = Figure()
        fig.add_subplot(111)
        return (
            fig,
            {
                "switched_off_count": 1,
                "normal_active_count": 1,
                "boresight_affected_active_count": 1,
                "map_style_used": "clean",
                "backend_used": "matplotlib",
            },
        )

    monkeypatch.setattr(sgui.visualise, "plot_cell_status_map", _plot_cell_status_map)
    hex_idx = window.tab_widget.indexOf(window.hexgrid_tab)
    window.tab_widget.setCurrentIndex(hex_idx)
    qapp.processEvents()
    assert prepare_calls == []
    window.hexgrid_boresight_enabled_checkbox.setChecked(True)
    qapp.processEvents()
    assert prepare_calls == []
    window._refresh_hexgrid_preview()
    _wait_until(lambda: prepare_calls == pytest.approx([79.141]))
    _wait_until(
        lambda: window._hexgrid_preview_window is not None
        and "ready" in window.hexgrid_preview_status_label.text().lower()
    )
    assert len(plot_kwargs_seen) == 1
    assert_equal(
        np.asarray(plot_kwargs_seen[0]["active_reuse_slot_ids"], dtype=np.int32),
        np.asarray([2, 0], dtype=np.int32),
    )
    assert isinstance(plot_kwargs_seen[0]["hex_lattice"], dict)
    assert int(plot_kwargs_seen[0]["hex_lattice"]["reuse_factor"]) == 7
    assert int(plot_kwargs_seen[0]["reuse_factor"]) == 7
    assert int(plot_kwargs_seen[0]["anchor_active_index"]) == 0
    assert float(plot_kwargs_seen[0]["point_spacing_km"]) == pytest.approx(79.141)
    assert "Adjacent same-slot pairs (active grid) = 2" in window._hexgrid_preview_window._summary_label.text()
    window._dirty = False; window.close()


def test_hexgrid_preview_cached_style_refresh_reuses_window_and_skips_grid_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._add_belt()
    window._add_ras_station()
    window._set_all_antenna_defaults()
    window._last_analyser_selected_cell_km = 79.141
    window._last_analyser_signature = window._analyser_signature(window.current_state())
    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_names": ["Tiny_1"],
            "altitudes_q": np.asarray([525.0]) * sgui.u.km,
            "inclinations_q": np.asarray([53.0]) * sgui.u.deg,
            "min_elevations_q": np.asarray([20.0]) * sgui.u.deg,
            "belt_sats": np.asarray([56], dtype=np.int64),
        },
    )
    prepare_calls: list[float] = []
    render_styles: list[str] = []

    def _prepare_active_grid(**kwargs: object) -> dict[str, object]:
        prepare_calls.append(float(sgui.u.Quantity(kwargs["point_spacing"]).to_value(sgui.u.km)))
        return _manual_hexgrid_result()

    def _plot_cell_status_map(*args: object, **kwargs: object) -> tuple[Figure, dict[str, object]]:
        del args
        render_styles.append(str(kwargs.get("map_style", "clean")))
        fig = Figure()
        fig.add_subplot(111)
        return (
            fig,
            {
                "switched_off_count": 1,
                "normal_active_count": 1,
                "boresight_affected_active_count": 1,
                "map_style_used": render_styles[-1],
                "backend_used": "matplotlib",
            },
        )

    monkeypatch.setattr(sgui.earthgrid, "prepare_active_grid", _prepare_active_grid)
    monkeypatch.setattr(
        sgui.earthgrid,
        "resolve_theta2_active_cell_ids",
        lambda *args, **kwargs: np.asarray([1], dtype=np.int32),
    )
    monkeypatch.setattr(sgui.visualise, "plot_cell_status_map", _plot_cell_status_map)

    window._refresh_hexgrid_preview()
    _wait_until(
        lambda: prepare_calls == pytest.approx([79.141])
        and render_styles == ["clean"]
        and window._hexgrid_preview_window is not None
    )
    assert render_styles == ["clean"]
    assert window._hexgrid_preview_window is not None
    first_window = window._hexgrid_preview_window

    terrain_idx = window.hexgrid_map_style_combo.findData("terrain")
    window.hexgrid_map_style_combo.setCurrentIndex(terrain_idx)
    _wait_until(lambda: render_styles == ["clean", "terrain"])
    assert prepare_calls == pytest.approx([79.141])
    assert window._hexgrid_preview_window is first_window
    assert "cached render" in window.hexgrid_preview_status_label.text().lower()

    window.hexgrid_ras_exclusion_layers_spin.setValue(2)
    window._hexgrid_preview_timer.stop()
    window._refresh_hexgrid_preview()
    _wait_until(lambda: prepare_calls == pytest.approx([79.141, 79.141]))
    assert window._hexgrid_preview_window is first_window
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_hexgrid_preview_cached_appearance_refresh_reuses_window_and_skips_grid_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    # Flush pending events from prior tests (especially lingering QThread teardown).
    QtWidgets.QApplication.processEvents()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._add_belt()
    window._add_ras_station()
    window._set_all_antenna_defaults()
    window._last_analyser_selected_cell_km = 79.141
    window._last_analyser_signature = window._analyser_signature(window.current_state())
    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_names": ["Tiny_1"],
            "altitudes_q": np.asarray([525.0]) * sgui.u.km,
            "inclinations_q": np.asarray([53.0]) * sgui.u.deg,
            "min_elevations_q": np.asarray([20.0]) * sgui.u.deg,
            "belt_sats": np.asarray([56], dtype=np.int64),
        },
    )
    prepare_calls: list[float] = []
    render_modes: list[str] = []

    def _prepare_active_grid(**kwargs: object) -> dict[str, object]:
        prepare_calls.append(float(sgui.u.Quantity(kwargs["point_spacing"]).to_value(sgui.u.km)))
        return _manual_hexgrid_result()

    def _plot_cell_status_map(*args: object, **kwargs: object) -> tuple[Figure, dict[str, object]]:
        del args, kwargs
        render_modes.append(window._selected_appearance_mode())
        fig = Figure()
        fig.add_subplot(111)
        return (
            fig,
            {
                "switched_off_count": 1,
                "normal_active_count": 1,
                "boresight_affected_active_count": 1,
                "map_style_used": "clean",
                "backend_used": "matplotlib",
            },
        )

    monkeypatch.setattr(sgui.earthgrid, "prepare_active_grid", _prepare_active_grid)
    monkeypatch.setattr(
        sgui.earthgrid,
        "resolve_theta2_active_cell_ids",
        lambda *args, **kwargs: np.asarray([1], dtype=np.int32),
    )
    monkeypatch.setattr(sgui.visualise, "plot_cell_status_map", _plot_cell_status_map)

    window._refresh_hexgrid_preview()
    _wait_until(
        lambda: prepare_calls == pytest.approx([79.141])
        and render_modes == ["system"]
        and window._hexgrid_preview_window is not None,
        timeout_ms=5000,
    )
    first_window = window._hexgrid_preview_window

    light_idx = window.appearance_mode_combo.findData("light")
    window.appearance_mode_combo.setCurrentIndex(light_idx)
    _wait_until(lambda: len(render_modes) >= 2, timeout_ms=3000)
    # Appearance change re-renders from cache (second plot_cell_status_map
    # call) so map-specific colours follow the new theme — no grid rebuild.
    assert prepare_calls == pytest.approx([79.141])
    assert len(render_modes) == 2
    assert window._hexgrid_preview_window is first_window
    assert window._selected_appearance_mode() == "light"
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_plot_cell_status_map_uses_cached_mpl_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Restore the real plot_cell_status_map (module-scoped stub replaced it)
    monkeypatch.setattr(sgui.visualise, "plot_cell_status_map", _ORIG_PLOT_CELL_STATUS_MAP)
    class _FakeExterior:
        def __init__(self, xs: list[float], ys: list[float]) -> None:
            self.xy = (xs, ys)

    class _FakePolygon:
        geom_type = "Polygon"

        def __init__(self, xs: list[float], ys: list[float]) -> None:
            self.exterior = _FakeExterior(xs, ys)

    class _FakeLine:
        geom_type = "LineString"

        def __init__(self, xs: list[float], ys: list[float]) -> None:
            self.xy = (xs, ys)

    load_calls: list[tuple[str, str]] = []

    def _load_natural_earth_geometries(kind: str, backend: str = "vendored") -> tuple[object, ...]:
        load_calls.append((kind, backend))
        if kind == "land":
            return (_FakePolygon([18.0, 24.0, 24.0, 18.0, 18.0], [-34.0, -34.0, -28.0, -28.0, -34.0]),)
        if kind == "coastline":
            return (_FakeLine([18.0, 24.0], [-31.0, -31.0]),)
        raise AssertionError(kind)

    monkeypatch.setattr("scepter.earthgrid._load_natural_earth_geometries", _load_natural_earth_geometries)
    sgui.visualise._cached_mpl_natural_earth_vertices.cache_clear()
    try:
        land_first = sgui.visualise._cached_mpl_natural_earth_vertices("land", backend="vendored")
        land_second = sgui.visualise._cached_mpl_natural_earth_vertices("land", backend="vendored")
        coast_first = sgui.visualise._cached_mpl_natural_earth_vertices("coastline", backend="vendored")
        coast_second = sgui.visualise._cached_mpl_natural_earth_vertices("coastline", backend="vendored")

        assert land_first is land_second
        assert coast_first is coast_second
        assert load_calls == [("land", "vendored"), ("coastline", "vendored")]

        fig, info = sgui.visualise.plot_cell_status_map(
            np.asarray([20.0, 20.5, 21.0], dtype=np.float64) * sgui.u.deg,
            np.asarray([-30.8, -30.7, -30.6], dtype=np.float64) * sgui.u.deg,
            active_cell_longitudes=np.asarray([20.0, 21.0], dtype=np.float64) * sgui.u.deg,
            active_cell_latitudes=np.asarray([-30.8, -30.6], dtype=np.float64) * sgui.u.deg,
            switched_off_mask=np.asarray([False, True, False], dtype=bool),
            boresight_affected_cell_ids=np.asarray([1], dtype=np.int32),
            ras_longitude=21.443611 * sgui.u.deg,
            ras_latitude=-30.712777 * sgui.u.deg,
            backend="matplotlib",
            map_style="terrain",
            return_info=True,
        )

        ax = fig.axes[0]
        assert any(isinstance(collection, mcollections.PolyCollection) for collection in ax.collections)
        assert any(isinstance(collection, mcollections.LineCollection) for collection in ax.collections)
        assert any(str(collection.get_gid() or "").startswith("hexgrid_") for collection in ax.collections)
        assert info["backend_used"] == "matplotlib"
        plt.close(fig)
    finally:
        sgui.visualise._cached_mpl_natural_earth_vertices.cache_clear()


def test_main_window_debounces_summary_refresh(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    calls: list[str] = []
    window._summary_refresh_timer.stop()
    window._summary_refresh_timer.timeout.disconnect()
    window._summary_refresh_timer.setInterval(20)
    window._summary_refresh_timer.timeout.connect(lambda: calls.append("refresh"))
    window._on_state_changed()
    window._on_state_changed()
    assert calls == []
    _wait_for(40)
    qapp.processEvents()
    assert calls == ["refresh"]
    window._dirty = False; window.close()


def test_async_workflow_summary_drops_stale_generations(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    del qapp
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    applied: list[tuple[str, ...]] = []

    def _record_apply(
        state: sgui.ScepterProjectState,
        status_payloads: Mapping[str, Mapping[str, object]],
    ) -> None:
        del state
        applied.append(tuple(status_payloads.keys()))

    monkeypatch.setattr(window, "_apply_workflow_status_payloads", _record_apply)
    window._workflow_summary_generation = 3
    state = window.current_state()
    window._on_workflow_summary_completed(
        2,
        {
            "state": state,
            "payloads": {"RAS Station": {"ready": False}},
            "error": None,
        },
    )
    assert applied == []

    window._on_workflow_summary_completed(
        3,
        {
            "state": state,
            "payloads": {"RAS Station": {"ready": False}},
            "error": None,
        },
    )
    assert applied == [("RAS Station",)]
    window._dirty = False; window.close()


def test_build_run_request_uses_runtime_knobs_and_cell_sizes(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state())
    window.cell_size_override_checkbox.setChecked(True)
    window.cell_size_override_spin.setValue(55.0)
    window.runtime_include_atmosphere_checkbox.setChecked(False)
    window.runtime_profile_stages_checkbox.setChecked(True)
    window.runtime_progress_desc_mode_combo.setCurrentIndex(
        window.runtime_progress_desc_mode_combo.findData("detailed")
    )
    window.runtime_writer_checkpoint_interval_edit.set_value(12.5)
    window.runtime_force_bulk_timesteps_edit.set_value(7)
    window.runtime_hdf5_compression_combo.setCurrentIndex(
        window.runtime_hdf5_compression_combo.findData("gzip")
    )
    window.runtime_hdf5_compression_opts_spin.setValue(4)
    window.runtime_gpu_method_combo.setCurrentIndex(window.runtime_gpu_method_combo.findData("vallado"))
    window.runtime_gpu_compute_dtype_combo.setCurrentIndex(
        window.runtime_gpu_compute_dtype_combo.findData("float64")
    )
    window.runtime_gpu_output_dtype_combo.setCurrentIndex(
        window.runtime_gpu_output_dtype_combo.findData("float16")
    )
    window.runtime_gpu_on_error_combo.setCurrentIndex(
        window.runtime_gpu_on_error_combo.findData("coerce_to_nan")
    )
    window.runtime_scheduler_target_combo.setCurrentIndex(
        window.runtime_scheduler_target_combo.findData("max_throughput")
    )
    window.target_pfd_edit.set_value(-88.25)
    qapp.processEvents()

    point_spacings_km: list[float] = []

    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_names": ["Tiny_1"],
            "tle_list": ["TLE1", "TLE2"],
            "altitudes_q": np.asarray([525.0]) * sgui.u.km,
            "inclinations_q": np.asarray([53.0]) * sgui.u.deg,
            "min_elevations_q": np.asarray([20.0]) * sgui.u.deg,
            "max_betas_q": np.asarray([35.0]) * sgui.u.deg,
            "belt_sats": np.asarray([2], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        sgui,
        "_satellite_antenna_pattern_spec",
        lambda antennas: (object(), 0.028 * sgui.u.m, {"use_numba": False}),
    )
    monkeypatch.setattr(
        sgui.earthgrid,
        "summarize_contour_spacing",
        lambda *args, **kwargs: {
            "selected_cell_spacing_km": 79.141,
            "spacing_theta_edge": 3.0 * sgui.u.deg,
            "summary_lines": ["Analyser"],
        },
    )

    def _prepare_active_grid(**kwargs: object) -> dict[str, object]:
        point_spacings_km.append(float(sgui.u.Quantity(kwargs["point_spacing"]).to_value(sgui.u.km)))
        return {
            "active_grid_longitudes": np.asarray([21.0], dtype=np.float64) * sgui.u.deg,
            "active_grid_latitudes": np.asarray([-30.0], dtype=np.float64) * sgui.u.deg,
            "ras_service_cell_active": True,
            "ras_service_cell_index": 0,
            "active_cell_count": 1,
            "pre_ras_cell_count": 2,
            "geography_kept_cell_count": 2,
            "geography_excluded_cell_count": 0,
            "ras_excluded_cell_count": 1,
        }

    monkeypatch.setattr(sgui.earthgrid, "prepare_active_grid", _prepare_active_grid)
    monkeypatch.setattr(
        sgui.scenario,
        "build_observer_layout",
        lambda ras_observer, active_cell_observers: {
            "observer_arr": np.asarray([ras_observer] + list(active_cell_observers), dtype=object)
        },
    )
    monkeypatch.setattr(
        sgui.tleforger,
        "expand_belt_metadata_to_satellites",
        lambda constellation: {"satellite_count": 2},
    )
    monkeypatch.setattr(
        sgui.tleforger,
        "build_satellite_storage_constants",
        lambda satellite_metadata, orbit_radius_m_per_sat: {
            "sat_min_elev_deg_per_sat": np.asarray([20.0, 20.0], dtype=np.float32),
            "sat_beta_max_deg_per_sat": np.asarray([35.0, 35.0], dtype=np.float32),
            "sat_belt_id_per_sat": np.asarray([0, 0], dtype=np.int32),
        },
    )

    _mark_hexgrid_preview_current(window)
    request_override = window._build_run_request(window.current_state())
    assert point_spacings_km == pytest.approx([55.0])
    assert request_override["include_atmosphere"] is False
    assert request_override["profile_stages"] is True
    assert request_override["progress_desc_mode"] == "detailed"
    assert request_override["enable_progress_desc_updates"] is True
    assert request_override["writer_checkpoint_interval_s"] == pytest.approx(12.5)
    assert request_override["force_bulk_timesteps"] == 7
    assert request_override["hdf5_compression"] == "gzip"
    assert request_override["hdf5_compression_opts"] == 4
    assert request_override["gpu_method"] == sgui.gpu_accel.METHOD_VALLADO
    assert request_override["gpu_compute_dtype"] is np.float64
    assert request_override["gpu_output_dtype"] is np.float16
    assert request_override["gpu_on_error"] == "coerce_to_nan"
    assert request_override["scheduler_target_profile"] == "max_throughput"
    assert request_override["sat_frame"] == "xyz"
    assert request_override["pfd0_dbw_m2_mhz"] == pytest.approx(-88.25)
    assert request_override["storage_attrs"]["target_pfd_dbw_m2_mhz"] == pytest.approx(-88.25)
    assert float(request_override["ras_guard_angle"].to_value(sgui.u.deg)) == pytest.approx(3.0)

    window.cell_size_override_checkbox.setChecked(False)
    window.runtime_progress_desc_mode_combo.setCurrentIndex(
        window.runtime_progress_desc_mode_combo.findData("off")
    )
    qapp.processEvents()
    _mark_hexgrid_preview_current(window)
    request_analyser = window._build_run_request(window.current_state())
    assert point_spacings_km == pytest.approx([55.0, 79.141])
    assert request_analyser["hdf5_compression"] == "gzip"
    assert request_analyser["enable_progress_desc_updates"] is False
    assert request_analyser["progress_desc_mode"] == "off"
    window._dirty = False; window.close()


def test_build_direct_epfd_run_request_from_gui_config_matches_direct_gui_request(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _stub_scene_assets(monkeypatch)
    base_state = _tiny_state()
    override_state = replace(
        base_state,
        runtime=replace(
            base_state.runtime,
            timestep_s=5.0,
            gpu_memory_budget_gb=10.0,
            host_memory_budget_gb=12.0,
            profile_stages=True,
        ),
    )

    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_names": ["Tiny_1"],
            "tle_list": ["TLE1", "TLE2"],
            "altitudes_q": np.asarray([525.0]) * sgui.u.km,
            "inclinations_q": np.asarray([53.0]) * sgui.u.deg,
            "min_elevations_q": np.asarray([20.0]) * sgui.u.deg,
            "max_betas_q": np.asarray([35.0]) * sgui.u.deg,
            "belt_sats": np.asarray([2], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        sgui,
        "_satellite_antenna_pattern_spec",
        lambda antennas: (object(), 0.028 * sgui.u.m, {"use_numba": False}),
    )
    monkeypatch.setattr(
        sgui.earthgrid,
        "summarize_contour_spacing",
        lambda *args, **kwargs: {
            "selected_cell_spacing_km": 79.141,
            "spacing_theta_edge": 3.0 * sgui.u.deg,
            "summary_lines": ["Analyser"],
        },
    )
    monkeypatch.setattr(
        sgui.earthgrid,
        "prepare_active_grid",
        lambda **kwargs: {
            "active_grid_longitudes": np.asarray([21.0], dtype=np.float64) * sgui.u.deg,
            "active_grid_latitudes": np.asarray([-30.0], dtype=np.float64) * sgui.u.deg,
            "ras_service_cell_active": True,
            "ras_service_cell_index": 0,
            "active_cell_count": 1,
            "pre_ras_cell_count": 2,
            "geography_kept_cell_count": 2,
            "geography_excluded_cell_count": 0,
            "ras_excluded_cell_count": 1,
        },
    )
    monkeypatch.setattr(
        sgui.scenario,
        "build_observer_layout",
        lambda ras_observer, active_cell_observers: {
            "observer_arr": np.asarray([ras_observer] + list(active_cell_observers), dtype=object)
        },
    )
    monkeypatch.setattr(
        sgui.tleforger,
        "expand_belt_metadata_to_satellites",
        lambda constellation: {"satellite_count": 2},
    )
    monkeypatch.setattr(
        sgui.tleforger,
        "build_satellite_storage_constants",
        lambda satellite_metadata, orbit_radius_m_per_sat: {
            "sat_min_elev_deg_per_sat": np.asarray([20.0, 20.0], dtype=np.float32),
            "sat_beta_max_deg_per_sat": np.asarray([35.0, 35.0], dtype=np.float32),
            "sat_belt_id_per_sat": np.asarray([0, 0], dtype=np.int32),
        },
    )

    config_path = tmp_path / "benchmark_like.json"
    sgui.save_project_state(config_path, base_state)

    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(override_state)
    _mark_hexgrid_preview_current(window)
    direct_request = window._build_run_request(window.current_state())
    config_request = sgui.scenario.build_direct_epfd_run_request_from_gui_config(
        config_path,
        timestep_s=5.0,
        gpu_memory_budget_gb=10.0,
        host_memory_budget_gb=12.0,
        profile_stages=True,
    )

    assert float(config_request["timestep"]) == pytest.approx(float(direct_request["timestep"]))
    assert float(config_request["gpu_memory_budget_gb"]) == pytest.approx(
        float(direct_request["gpu_memory_budget_gb"])
    )
    assert float(config_request["host_memory_budget_gb"]) == pytest.approx(
        float(direct_request["host_memory_budget_gb"])
    )
    assert bool(config_request["profile_stages"]) is bool(direct_request["profile_stages"])
    assert config_request["output_families"] == direct_request["output_families"]
    assert config_request["selection_strategy"] == direct_request["selection_strategy"]
    assert bool(config_request["include_atmosphere"]) is bool(direct_request["include_atmosphere"])
    assert float(config_request["bandwidth_mhz"]) == pytest.approx(float(direct_request["bandwidth_mhz"]))
    assert config_request["power_input_quantity"] == direct_request["power_input_quantity"]
    assert config_request["power_input_basis"] == direct_request["power_input_basis"]
    assert config_request["target_pfd_dbw_m2_mhz"] == pytest.approx(
        float(direct_request["target_pfd_dbw_m2_mhz"])
    )
    assert config_request["target_pfd_dbw_m2_channel"] == direct_request["target_pfd_dbw_m2_channel"]
    assert config_request["satellite_ptx_dbw_mhz"] == direct_request["satellite_ptx_dbw_mhz"]
    assert config_request["satellite_ptx_dbw_channel"] == direct_request["satellite_ptx_dbw_channel"]
    assert config_request["satellite_eirp_dbw_mhz"] == direct_request["satellite_eirp_dbw_mhz"]
    assert config_request["satellite_eirp_dbw_channel"] == direct_request["satellite_eirp_dbw_channel"]
    assert int(config_request["ras_service_cell_index"]) == int(direct_request["ras_service_cell_index"])
    assert config_request["observer_arr"].shape == direct_request["observer_arr"].shape
    assert set(config_request["spectrum_plan"]) == set(direct_request["spectrum_plan"])
    for key in sorted(direct_request["spectrum_plan"]):
        lhs = config_request["spectrum_plan"][key]
        rhs = direct_request["spectrum_plan"][key]
        if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
            assert_equal(lhs, rhs)
        else:
            assert lhs == rhs
    window._dirty = False; window.close()


@pytest.mark.parametrize(
    "case",
    [
        {
            "selection_strategy": "max_elevation",
            "ras_pointing_mode": "cell_center",
            "ras_service_cell_active": False,
            "include_atmosphere": False,
            "memory_budget_mode": "hybrid",
            "progress_desc_mode": "coarse",
            "profile_name": "counts_only",
            "boresight_enabled": False,
        },
        {
            "selection_strategy": "random",
            "ras_pointing_mode": "ras_station",
            "ras_service_cell_active": True,
            "include_atmosphere": True,
            "memory_budget_mode": "host_only",
            "progress_desc_mode": "detailed",
            "profile_name": "totals_only",
            "boresight_enabled": False,
        },
        {
            "selection_strategy": "random",
            "ras_pointing_mode": "ras_station",
            "ras_service_cell_active": True,
            "include_atmosphere": True,
            "memory_budget_mode": "gpu_only",
            "progress_desc_mode": "coarse",
            "profile_name": "notebook_full",
            "boresight_enabled": True,
            "theta1_deg": 1.0,
            "theta2_deg": None,
        },
        {
            "selection_strategy": "max_elevation",
            "ras_pointing_mode": "cell_center",
            "ras_service_cell_active": False,
            "include_atmosphere": False,
            "memory_budget_mode": "gpu_only",
            "progress_desc_mode": "off",
            "profile_name": "gui_heavy",
            "boresight_enabled": True,
            "theta1_deg": None,
            "theta2_deg": 3.0,
            "theta2_scope_mode": "cell_ids",
            "theta2_cell_ids": "0, 2",
            "theta2_return_ids": np.asarray([0, 2], dtype=np.int32),
        },
        {
            "selection_strategy": "random",
            "ras_pointing_mode": "cell_center",
            "ras_service_cell_active": False,
            "include_atmosphere": True,
            "memory_budget_mode": "hybrid",
            "progress_desc_mode": "coarse",
            "profile_name": "notebook_full",
            "boresight_enabled": True,
            "theta1_deg": None,
            "theta2_deg": 3.0,
            "theta2_scope_mode": "adjacency_layers",
            "theta2_layers": 2,
            "theta2_return_ids": np.asarray([0, 1], dtype=np.int32),
        },
        {
            "selection_strategy": "max_elevation",
            "ras_pointing_mode": "ras_station",
            "ras_service_cell_active": True,
            "include_atmosphere": True,
            "memory_budget_mode": "host_only",
            "progress_desc_mode": "detailed",
            "profile_name": "gui_heavy",
            "boresight_enabled": True,
            "theta1_deg": None,
            "theta2_deg": 3.0,
            "theta2_scope_mode": "radius_km",
            "theta2_radius_km": 180.0,
            "theta2_return_ids": np.asarray([1], dtype=np.int32),
        },
        {
            "selection_strategy": "random",
            "ras_pointing_mode": "ras_station",
            "ras_service_cell_active": False,
            "include_atmosphere": True,
            "memory_budget_mode": "hybrid",
            "progress_desc_mode": "coarse",
            "profile_name": "notebook_full",
            "boresight_enabled": True,
            "theta1_deg": 1.0,
            "theta2_deg": 3.0,
            "theta2_scope_mode": "adjacency_layers",
            "theta2_layers": 1,
            "theta2_return_ids": np.asarray([0, 1], dtype=np.int32),
        },
    ],
    ids=[
        "cell_center_counts",
        "ras_station_totals",
        "theta1_only",
        "theta2_cell_ids",
        "theta2_layers",
        "theta2_radius",
        "inactive_service_fallback",
    ],
)
def test_build_run_request_branch_matrix(
    case: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    resolver_calls: list[dict[str, object]] = []
    _patch_run_request_dependencies(
        monkeypatch,
        ras_service_cell_active=bool(case["ras_service_cell_active"]),
        theta2_ids=case.get("theta2_return_ids"),
        resolver_calls=resolver_calls,
    )
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().service.selection_strategy = str(case["selection_strategy"])
    state.active_system().hexgrid.ras_pointing_mode = str(case["ras_pointing_mode"])
    state.active_system().hexgrid.boresight_avoidance_enabled = bool(case["boresight_enabled"])
    state.active_system().hexgrid.boresight_theta1_deg = case.get("theta1_deg")
    state.active_system().hexgrid.boresight_theta2_deg = case.get("theta2_deg")
    state.active_system().hexgrid.boresight_theta2_scope_mode = str(case.get("theta2_scope_mode", "cell_ids"))
    state.active_system().hexgrid.boresight_theta2_cell_ids = case.get("theta2_cell_ids")
    state.active_system().hexgrid.boresight_theta2_layers = int(case.get("theta2_layers", 0))
    state.active_system().hexgrid.boresight_theta2_radius_km = case.get("theta2_radius_km")
    state.runtime.include_atmosphere = bool(case["include_atmosphere"])
    state.runtime.memory_budget_mode = str(case["memory_budget_mode"])
    state.runtime.progress_desc_mode = str(case["progress_desc_mode"])
    state.runtime.writer_checkpoint_interval_s = 30.0
    profile_name = str(case["profile_name"])
    state.runtime.output_families = _output_family_profile(profile_name)

    window._load_state_into_widgets(state)
    qapp.processEvents()
    _mark_hexgrid_preview_current(window)
    request = window._build_run_request(window.current_state())

    assert request["selection_strategy"] == case["selection_strategy"]
    assert request["ras_pointing_mode"] == case["ras_pointing_mode"]
    assert request["include_atmosphere"] is bool(case["include_atmosphere"])
    assert request["memory_budget_mode"] == case["memory_budget_mode"]
    assert request["progress_desc_mode"] == case["progress_desc_mode"]
    assert request["writer_checkpoint_interval_s"] == pytest.approx(30.0)
    assert request["enable_progress_desc_updates"] is (case["progress_desc_mode"] != "off")
    assert request["output_families"] == state.runtime.output_families
    assert request["output_families"]["beam_statistics"]["mode"] == state.runtime.output_families["beam_statistics"]["mode"]
    assert request["output_families"]["prx_elevation_heatmap"]["mode"] == state.runtime.output_families["prx_elevation_heatmap"]["mode"]
    assert request["pfd0_dbw_m2_mhz"] == pytest.approx(-83.5)
    assert request["storage_attrs"]["target_pfd_dbw_m2_mhz"] == pytest.approx(-83.5)

    if case.get("theta2_deg") is None:
        assert request["boresight_theta2"] is None
        assert request["boresight_theta2_cell_ids"] is None
        assert resolver_calls == []
    else:
        assert request["boresight_theta2"] == pytest.approx(float(case["theta2_deg"]))
        assert resolver_calls
        resolver_call = resolver_calls[-1]
        assert resolver_call["scope_mode"] == case["theta2_scope_mode"]
        if case["theta2_scope_mode"] == "cell_ids":
            assert_equal(resolver_call["explicit_ids"], np.asarray([0, 2], dtype=np.int32))
        if case["theta2_scope_mode"] == "adjacency_layers":
            assert int(resolver_call["layers"]) == int(case["theta2_layers"])
        if case["theta2_scope_mode"] == "radius_km":
            assert float(resolver_call["radius_km"]) == pytest.approx(
                float(case["theta2_radius_km"])
            )
        assert_equal(
            request["boresight_theta2_cell_ids"],
            np.asarray(case["theta2_return_ids"], dtype=np.int32),
        )

    expected_effective = (
        "ras_station"
        if case["ras_pointing_mode"] == "ras_station" and bool(case["ras_service_cell_active"])
        else "cell_center"
    )
    assert request["storage_attrs"]["effective_ras_pointing_mode"] == expected_effective
    assert request["ras_service_cell_active"] is bool(case["ras_service_cell_active"])
    window._dirty = False; window.close()


@pytest.mark.parametrize(
    ("quantity", "basis", "value_field", "value", "request_field", "complementary_field"),
    [
        (
            "target_pfd",
            "per_mhz",
            "target_pfd_dbw_m2_mhz",
            -83.5,
            "target_pfd_dbw_m2_mhz",
            "target_pfd_dbw_m2_channel",
        ),
        (
            "target_pfd",
            "per_channel",
            "target_pfd_dbw_m2_channel",
            -76.51029995663981,
            "target_pfd_dbw_m2_channel",
            "target_pfd_dbw_m2_mhz",
        ),
        (
            "satellite_ptx",
            "per_mhz",
            "satellite_ptx_dbw_mhz",
            12.0,
            "satellite_ptx_dbw_mhz",
            "satellite_ptx_dbw_channel",
        ),
        (
            "satellite_ptx",
            "per_channel",
            "satellite_ptx_dbw_channel",
            18.989700043360187,
            "satellite_ptx_dbw_channel",
            "satellite_ptx_dbw_mhz",
        ),
        (
            "satellite_eirp",
            "per_mhz",
            "satellite_eirp_dbw_mhz",
            33.0,
            "satellite_eirp_dbw_mhz",
            "satellite_eirp_dbw_channel",
        ),
        (
            "satellite_eirp",
            "per_channel",
            "satellite_eirp_dbw_channel",
            39.98970004336019,
            "satellite_eirp_dbw_channel",
            "satellite_eirp_dbw_mhz",
        ),
    ],
)
def test_build_run_request_uses_selected_service_power_quantity_and_basis(
    quantity: str,
    basis: str,
    value_field: str,
    value: float,
    request_field: str,
    complementary_field: str,
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    _patch_run_request_dependencies(monkeypatch, ras_service_cell_active=True)
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().service.bandwidth_mhz = 5.0
    state.active_system().service.power_input_quantity = quantity
    state.active_system().service.power_input_basis = basis
    state.active_system().service.target_pfd_dbw_m2_mhz = None
    state.active_system().service.target_pfd_dbw_m2_channel = None
    state.active_system().service.satellite_ptx_dbw_mhz = None
    state.active_system().service.satellite_ptx_dbw_channel = None
    state.active_system().service.satellite_eirp_dbw_mhz = None
    state.active_system().service.satellite_eirp_dbw_channel = None
    setattr(state.active_system().service, value_field, float(value))

    window._load_state_into_widgets(state)
    qapp.processEvents()
    _mark_hexgrid_preview_current(window)
    request = window._build_run_request(window.current_state())

    expected_offset_db = 10.0 * np.log10(5.0)
    expected_complementary = (
        float(value) + float(expected_offset_db)
        if basis == "per_mhz"
        else float(value) - float(expected_offset_db)
    )

    assert request["bandwidth_mhz"] == pytest.approx(5.0)
    assert request["power_input_quantity"] == quantity
    assert request["power_input_basis"] == basis
    assert request[request_field] == pytest.approx(float(value))
    assert request[complementary_field] == pytest.approx(expected_complementary)
    assert request["storage_attrs"]["power_input_quantity"] == quantity
    assert request["storage_attrs"]["power_input_basis"] == basis
    assert request["storage_attrs"]["power_input_value"] == pytest.approx(float(value))
    if quantity == "target_pfd":
        assert request["pfd0_dbw_m2_mhz"] == pytest.approx(
            request["target_pfd_dbw_m2_mhz"]
        )
    else:
        assert request["pfd0_dbw_m2_mhz"] is None
    window._dirty = False; window.close()


def test_build_run_request_includes_spectrum_plan_and_reuse_slots(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _patch_run_request_dependencies(monkeypatch, ras_service_cell_active=True)
    monkeypatch.setattr(
        sgui.earthgrid,
        "resolve_frequency_reuse_slots",
        lambda prepared_grid, *, reuse_factor, anchor_slot=0: {
            "reuse_factor": int(reuse_factor),
            "anchor_slot": int(anchor_slot),
            "anchor_pre_ras_index": 0,
            "anchor_active_index": 0,
            "point_spacing_km_used": 79.141,
            "orientation_used": "pointy",
            "fit_residual_km2": 0.0,
            "axial_q_pre_ras": np.asarray([0, 1], dtype=np.int32),
            "axial_r_pre_ras": np.asarray([0, 0], dtype=np.int32),
            "pre_ras_slot_ids": np.asarray([2, 0], dtype=np.int32),
            "active_slot_ids": np.asarray([2, 0], dtype=np.int32),
            "cluster_representatives": [
                {"slot_id": 0, "base_slot_id": 0, "axial_q": 0, "axial_r": 0}
            ],
        },
    )
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().spectrum.reuse_factor = 4
    state.active_system().spectrum.disabled_channel_indices = None
    state.active_system().spectrum.ras_anchor_reuse_slot = 2
    state.active_system().service.cell_activity_mode = "per_channel"
    state.active_system().spectrum.multi_group_power_policy = "split_total_cell_power"
    state.active_system().spectrum.split_total_group_denominator_mode = "active_groups"
    state.active_system().spectrum.service_band_start_mhz = 2620.0
    state.active_system().spectrum.service_band_stop_mhz = 2690.0
    state.ras_station.receiver_band_start_mhz = 2690.0
    state.ras_station.receiver_band_stop_mhz = 2700.0
    window._load_state_into_widgets(state)
    qapp.processEvents()
    _mark_hexgrid_preview_current(window)

    request = window._build_run_request(window.current_state())

    assert request["spectrum_plan"]["reuse_factor"] == 4
    assert request["spectrum_plan"]["channel_groups_per_cell"] == 4
    assert len(request["spectrum_plan"]["enabled_channel_indices"]) == 14
    assert request["spectrum_plan"]["ras_anchor_reuse_slot"] == 2
    assert request["spectrum_plan"]["ras_reference_mode"] == "lower"
    assert request["spectrum_plan"]["split_total_group_denominator_mode"] == "active_groups"
    assert request["cell_activity_mode"] == "per_channel"
    assert request["split_total_group_denominator_mode"] == "active_groups"
    assert request["storage_attrs"]["reuse_factor"] == 4
    assert request["storage_attrs"]["ras_reference_mode"] == "lower"
    assert_equal(
        request["spectrum_plan"]["active_cell_reuse_slot_ids"],
        np.asarray([2, 0], dtype=np.int32),
    )
    assert_equal(
        request["storage_constants"]["cell_reuse_slot_id_active"],
        np.asarray([2, 0], dtype=np.int32),
    )
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_split_denominator_control_enables_only_for_per_channel_split_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    expert_idx = window.complexity_mode_combo.findData("Expert")
    window.complexity_mode_combo.setCurrentIndex(expert_idx)

    assert window.spectrum_split_denominator_combo.isEnabled() is False

    window._set_combo_to_data(window.spectrum_reuse_factor_combo, 4)
    window._set_combo_to_data(window.service_cell_activity_mode_combo, "per_channel")
    window._set_combo_to_data(
        window.spectrum_power_policy_combo,
        "split_total_cell_power",
    )
    window._sync_spectrum_controls()

    assert window.spectrum_split_denominator_combo.isEnabled() is True

    window._set_combo_to_data(window.service_cell_activity_mode_combo, "whole_cell")
    window._sync_spectrum_controls()
    assert window.spectrum_split_denominator_combo.isEnabled() is False

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_edit_custom_mask_points_uses_graphical_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    starting_points = [[0.0, 0.0], [5.0, 45.0], [10.0, 50.0]]
    window._spectrum_custom_mask_points = starting_points

    class _FakeDialog:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.Accepted)

        def selected_points(self) -> list[list[float]]:
            return [[0.0, 0.0], [3.0, 30.0], [8.0, 48.0]]

    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getMultiLineText",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy text mask editor should not be used")
        ),
    )
    monkeypatch.setattr(sgui, "SpectrumMaskEditorDialog", _FakeDialog)

    window._edit_custom_mask_points()

    assert window._spectrum_custom_mask_points == [[0.0, 0.0], [3.0, 30.0], [8.0, 48.0]]
    assert window.spectrum_mask_preset_combo.currentData() == "custom"

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_graphical_mask_editor_supports_add_remove_and_anchor_constraints(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    del monkeypatch, qapp
    preview_plan = {
        "slot_edges_mhz": np.asarray([2620.0, 2625.0, 2630.0], dtype=np.float64),
        "slot_centers_mhz": np.asarray([2622.5, 2627.5], dtype=np.float64),
        "slot_leakage_factors": {0: 0.15, 1: 0.05},
        "ras_receiver_band_start_mhz": 2690.0,
        "ras_receiver_band_stop_mhz": 2700.0,
        "receiver_response_mode": "rectangular",
        "receiver_response_points_mhz": None,
        "unwanted_emission_mask_points_mhz": np.asarray(
            [[-6.0, 45.0], [-2.5, 0.0], [2.5, 0.0], [6.0, 50.0]],
            dtype=np.float64,
        ),
        "spectral_integration_cutoff_mhz": 40.0,
        "channel_bandwidth_mhz": 5.0,
        "reuse_factor": 1,
    }
    dialog = sgui.SpectrumMaskEditorDialog(
        initial_points=[[-6.0, 45.0], [-2.5, 0.0], [2.5, 0.0], [6.0, 50.0]],
        channel_bandwidth_mhz=5.0,
        preview_plan_factory=lambda points: dict(
            preview_plan,
            unwanted_emission_mask_points_mhz=np.asarray(points, dtype=np.float64),
        ),
    )

    assert dialog._constrain_point(1, -1.0, 9.0) == (-2.5, 0.0)
    assert np.max(dialog._mask_axis.lines[0].get_ydata()) <= 0.0
    assert dialog._mask_axis.get_xlim()[0] < 0.0
    assert dialog._mask_axis.get_xlim()[1] > 0.0

    dialog._insert_point_from_event(
        types.SimpleNamespace(
            inaxes=dialog._mask_axis,
            xdata=7.5,
            ydata=-47.0,
        )
    )
    assert len(dialog.selected_points()) == 5

    points_before_remove = dialog.selected_points()
    dialog._select_point(0)
    assert dialog._remove_selected_point() is True
    assert dialog.selected_points() != points_before_remove
    dialog._select_point(1)
    assert dialog._remove_selected_point() is False

    dialog._reset_points()
    assert dialog.selected_points() == [[-6.0, 45.0], [-2.5, 0.0], [2.5, 0.0], [6.0, 50.0]]
    dialog.close()


def test_spectrum_controls_highlight_exact_fit_reuse_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    idx_f1 = window.spectrum_reuse_factor_combo.findData(1)
    idx_f4 = window.spectrum_reuse_factor_combo.findData(4)
    idx_f7 = window.spectrum_reuse_factor_combo.findData(7)

    assert "exact fit" in window.spectrum_reuse_factor_combo.itemText(idx_f1).lower()
    assert "exact fit" not in window.spectrum_reuse_factor_combo.itemText(idx_f4).lower()
    assert "exact fit" in window.spectrum_reuse_factor_combo.itemText(idx_f7).lower()
    assert "F1" in window.spectrum_zero_leftover_label.text()
    assert "F7" in window.spectrum_zero_leftover_label.text()
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_edit_spectrum_selection_updates_disabled_channel_indices(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    assert window.current_state().ras_station is not None
    window._set_ras_station_widgets(sgui._default_ras_station_config())
    window._sync_spectrum_controls()
    qapp.processEvents()

    assert window.current_state().active_system().spectrum.disabled_channel_indices is None

    window._set_combo_to_data(window.spectrum_reuse_factor_combo, 7)
    qapp.processEvents()

    window._apply_spectrum_enabled_channel_selection(set(range(7)))
    qapp.processEvents()

    assert window.current_state().active_system().spectrum.disabled_channel_indices == list(range(7, 14))
    assert "channels enabled" in window.spectrum_summary_label.text().lower()

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_ras_receiver_action_button_opens_preview_for_builtin_response(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    state = _tiny_state()
    assert state.ras_station is not None
    state.ras_station.receiver_band_start_mhz = 2690.0
    state.ras_station.receiver_band_stop_mhz = 2700.0
    state.ras_station.receiver_response_mode = "rectangular"
    window = _make_run_window(monkeypatch, state=state)
    preview_calls: list[str] = []
    edit_calls: list[str] = []

    monkeypatch.setattr(window, "_show_ras_receiver_response_preview", lambda: preview_calls.append("preview"))
    monkeypatch.setattr(window, "_edit_ras_receiver_mask_points", lambda: edit_calls.append("edit"))
    monkeypatch.setattr(
        window,
        "_show_spectrum_preview",
        lambda: (_ for _ in ()).throw(
            AssertionError("RAS receiver action should not open the spectrum editor")
        ),
    )

    window._sync_spectrum_controls()
    qapp.processEvents()
    assert window.ras_receiver_response_tool_button.text() == "Preview Receiver Response"

    window._open_ras_receiver_response_tool()

    assert preview_calls == ["preview"]
    assert edit_calls == []
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_ras_receiver_action_button_opens_editor_for_custom_response(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    state = _tiny_state()
    assert state.ras_station is not None
    state.ras_station.receiver_band_start_mhz = 2690.0
    state.ras_station.receiver_band_stop_mhz = 2700.0
    state.ras_station.receiver_response_mode = "custom"
    state.ras_station.receiver_custom_mask_points = [[-5.0, 30.0], [5.0, 30.0]]
    window = _make_run_window(monkeypatch, state=state)
    preview_calls: list[str] = []
    edit_calls: list[str] = []

    monkeypatch.setattr(window, "_show_ras_receiver_response_preview", lambda: preview_calls.append("preview"))
    monkeypatch.setattr(window, "_edit_ras_receiver_mask_points", lambda: edit_calls.append("edit"))

    window._sync_spectrum_controls()
    qapp.processEvents()
    assert window.ras_receiver_response_tool_button.text() == "Edit Receiver Response"

    window._open_ras_receiver_response_tool()

    assert edit_calls == ["edit"]
    assert preview_calls == []
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_blank_state_and_defaults_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(
        monkeypatch,
        state=sgui.ScepterProjectState(),
        current_hexgrid=False,
    )

    assert window.spectrum_service_band_start_edit.value_or_none() is None
    assert window.spectrum_reuse_factor_combo.currentData() is None
    assert window.show_reuse_scheme_button.isEnabled() is False
    assert window.preview_spectrum_plot_button.isEnabled() is False
    assert "blank" in window.spectrum_summary_label.text().lower()

    window._set_spectrum_defaults()
    qapp.processEvents()
    assert window.spectrum_service_band_start_edit.value_or_none() == pytest.approx(2620.0)
    assert window.spectrum_reuse_factor_combo.currentData() == 1
    assert window.show_reuse_scheme_button.isEnabled() is True
    assert window.preview_spectrum_plot_button.isEnabled() is False
    assert "channel bandwidth" in window.spectrum_summary_label.text().lower()

    window._set_service_defaults()
    window._sync_spectrum_controls()
    qapp.processEvents()
    assert window.show_reuse_scheme_button.isEnabled() is True
    assert window.preview_spectrum_plot_button.isEnabled() is True

    window._clear_spectrum_inputs()
    qapp.processEvents()
    assert window.spectrum_service_band_start_edit.value_or_none() is None
    assert window.spectrum_reuse_factor_combo.currentData() is None
    assert window.show_reuse_scheme_button.isEnabled() is False
    assert window.preview_spectrum_plot_button.isEnabled() is False
    assert window.spectrum_tx_reference_points_spin.value() == 1
    assert window.spectrum_ras_reference_points_spin.value() == 1
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_reference_points_hide_when_not_n_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    expert_idx = window.complexity_mode_combo.findData("Expert")
    window.complexity_mode_combo.setCurrentIndex(expert_idx)
    window._set_combo_to_data(window.spectrum_tx_reference_combo, "middle")
    window._set_combo_to_data(window.spectrum_ras_reference_combo, "lower")
    window.spectrum_tx_reference_points_spin.setValue(9)
    window.spectrum_ras_reference_points_spin.setValue(11)
    window._sync_spectrum_controls()

    assert window.spectrum_tx_reference_points_spin.value() == 1
    assert window.spectrum_ras_reference_points_spin.value() == 1
    assert window.spectrum_tx_reference_points_field.isHidden() is True
    assert window.spectrum_ras_reference_points_field.isHidden() is True

    window._set_combo_to_data(window.spectrum_tx_reference_combo, "n_point")
    window._set_combo_to_data(window.spectrum_ras_reference_combo, "n_point")
    window._sync_spectrum_controls()

    assert window.spectrum_tx_reference_points_field.isHidden() is False
    assert window.spectrum_ras_reference_points_field.isHidden() is False
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_service_and_ras_defaults_do_not_activate_spectrum_buttons_without_explicit_spectrum_inputs(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(
        monkeypatch,
        state=sgui.ScepterProjectState(),
        current_hexgrid=False,
    )

    window._set_service_defaults()
    window._set_ras_station_widgets(sgui._default_ras_station_config())
    window._sync_spectrum_controls()
    qapp.processEvents()

    assert window._workflow_has_explicit_spectrum_inputs() is False
    assert window.show_reuse_scheme_button.isEnabled() is False
    assert window.preview_spectrum_plot_button.isEnabled() is False
    assert "blank" in window.spectrum_summary_label.text().lower()
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_help_registry_updates_status_bar_for_spectrum_controls(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)

    assert window.service_cell_activity_mode_combo in window._help_registry
    assert window.spectrum_reuse_factor_combo in window._help_registry
    assert window.show_reuse_scheme_button in window._help_registry

    help_button = window._help_buttons[window.spectrum_reuse_factor_combo]
    label = window._form_label_for_widget(window.spectrum_reuse_factor_combo)
    assert label is not None

    QtWidgets.QApplication.sendEvent(
        window.spectrum_reuse_factor_combo,
        QtGui.QFocusEvent(QtCore.QEvent.FocusIn),
    )
    qapp.processEvents()
    assert "reuse" in window.statusBar().currentMessage().lower()

    QtWidgets.QApplication.sendEvent(label, QtCore.QEvent(QtCore.QEvent.Enter))
    qapp.processEvents()
    assert "reuse" in window.statusBar().currentMessage().lower()

    QtWidgets.QApplication.sendEvent(help_button, QtCore.QEvent(QtCore.QEvent.Enter))
    qapp.processEvents()
    assert "reuse" in window.statusBar().currentMessage().lower()

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_run_readiness_rejects_invalid_runtime_modes(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().grid_analysis.cell_size_override_enabled = True
    state.active_system().grid_analysis.cell_size_override_km = 55.0
    window._hexgrid_applied_signature = window._hexgrid_commit_signature(state)
    state.runtime.progress_desc_mode = "invalid"
    ready, message = window._run_readiness_payload(state)
    assert ready is False
    assert "progress detail mode is unsupported" in message.lower()

    state.runtime.progress_desc_mode = "coarse"
    state.runtime.memory_budget_mode = "invalid"
    ready, message = window._run_readiness_payload(state)
    assert ready is False
    assert "memory budget mode is unsupported" in message.lower()

    state.runtime.memory_budget_mode = "hybrid"
    state.runtime.output_families["beam_statistics"]["mode"] = "invalid"
    ready, message = window._run_readiness_payload(state)
    assert ready is False
    assert "output family mode is unsupported" in message.lower()
    window._dirty = False; window.close()


def test_run_simulation_readiness_rejection_does_not_start_worker(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch, current_hexgrid=False)
    assert window._run_thread is None
    window._run_simulation()
    qapp.processEvents()
    assert window._run_thread is None
    assert window._run_in_progress is False
    status_text = window.run_status_label.text().lower()
    assert "coverage" in status_text  # may say "coverage and boresight" or "coverage contour analysis"
    assert window.run_monitor.log_view.toPlainText() == ""
    window._dirty = False; window.close()


def test_run_simulation_button_reenables_after_success(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": "run_out.h5"})
    monkeypatch.setattr(
        sgui.scenario,
        "run_gpu_direct_epfd",
        lambda **kwargs: (
            QtCore.QThread.msleep(50),
            {"storage_filename": kwargs["storage_filename"]},
        )[1],
    )
    assert window.run_simulation_button.isEnabled() is True
    window._run_simulation()
    assert window._run_in_progress is True
    assert window.run_simulation_button.isEnabled() is False
    _wait_until(lambda: window._run_thread is None, timeout_ms=1000)
    assert window._run_in_progress is False
    assert window.run_simulation_button.isEnabled() is True
    assert "run_out.h5" in window.run_status_label.text()
    window._dirty = False; window.close()


def test_run_simulation_failure_updates_status(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": "run_fail.h5"})

    def _fail_run(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _fail_run)
    window._run_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=1000)
    assert window._run_in_progress is False
    assert window.run_simulation_button.isEnabled() is True
    assert "boom" in window.run_status_label.text().lower()
    window._dirty = False; window.close()


def test_run_simulation_preflight_failure_updates_status_and_disables_postprocess(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    run_called = {"value": False}

    def _fail_preflight(state: object) -> dict[str, object]:
        del state
        raise RuntimeError("synthetic preflight issue")

    def _unexpected_run(**kwargs: object) -> object:
        del kwargs
        run_called["value"] = True
        return {"storage_filename": "unexpected.h5"}

    monkeypatch.setattr(window, "_build_run_request", _fail_preflight)
    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _unexpected_run)
    window._run_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=1000)

    assert run_called["value"] is False
    assert window.run_simulation_button.isEnabled() is True
    assert window.stop_simulation_button.isEnabled() is False
    assert window.force_stop_simulation_button.isEnabled() is False
    assert "simulation preflight failed: synthetic preflight issue" in window.run_status_label.text().lower()
    assert "simulation preflight failed: synthetic preflight issue" in window.run_monitor.status_text().lower()
    assert window.run_monitor.open_result_button.isEnabled() is False
    assert window.open_result_button.isEnabled() is False
    window._dirty = False; window.close()


def test_run_simulation_writer_failure_surfaces_inner_cause(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "run_writer_fail.h5"
    window = _make_run_window(monkeypatch)
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": str(result_file)})

    def _fail_apply(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("synthetic writer failure")

    monkeypatch.setattr(sgui.scenario, "_apply_coalesced_write_batch", _fail_apply)

    def _run_with_writer_failure(**kwargs: object) -> object:
        del kwargs
        times = sgui.Time([60060.0, 60060.0 + (1.0 / 86400.0)], format="mjd", scale="utc")
        power = (np.arange(4, dtype=np.float32).reshape(2, 2) + 1.0) * sgui.u.W
        sgui.scenario.init_simulation_results(
            str(result_file),
            write_mode="async",
            writer_queue_max_items=1,
            writer_queue_max_bytes=1024,
        )
        sgui.scenario._write_iteration_batch_owned(
            str(result_file),
            iteration=0,
            batch_items=(
                ("times", times),
                ("power", power),
            ),
        )
        sgui.scenario.close_writer(str(result_file))
        return {"storage_filename": str(result_file)}

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _run_with_writer_failure)
    window._run_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=1500)
    status_text = window.run_status_label.text().lower()
    log_text = window.run_monitor.log_view.toPlainText().lower()

    assert "background hdf5 writer" in status_text
    assert "synthetic writer failure" in status_text
    assert "synthetic writer failure" in log_text
    assert window.run_monitor.open_result_button.isEnabled() is False

    sgui.scenario.close_writer()
    window._dirty = False; window.close()


def test_run_simulation_worker_graceful_stop_during_prepare_skips_gpu_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = sgui.RunCancelController()
    run_called = {"value": False}
    progress_events: list[dict[str, object]] = []
    finished_results: list[object] = []
    failures: list[str] = []

    def _preflight_then_request_stop(state: object) -> dict[str, object]:
        del state
        controller.request("graceful")
        return {"storage_filename": "should_not_run.h5"}

    def _unexpected_run(**kwargs: object) -> object:
        del kwargs
        run_called["value"] = True
        return {"storage_filename": "unexpected.h5"}

    worker = sgui.RunSimulationWorker(
        project_state=_tiny_state(),
        prepare_callback=_preflight_then_request_stop,
        cancel_controller=controller,
    )
    worker.progress_event.connect(lambda payload: progress_events.append(dict(payload)))
    worker.finished.connect(lambda result: finished_results.append(result))
    worker.failed.connect(lambda message: failures.append(str(message)))
    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _unexpected_run)

    worker.run()

    assert run_called["value"] is False
    assert failures == []
    assert finished_results == [{"run_state": "stopped"}]
    assert any(
        payload.get("kind") == "run_stopped" and payload.get("stop_mode") == "graceful"
        for payload in progress_events
    )


def test_run_simulation_status_tracks_checkpoint_and_final_flush(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    monkeypatch.setattr(
        window,
        "_build_run_request",
        lambda state: {"storage_filename": "run_flush.h5"},
    )

    def _run_with_progress(**kwargs: object) -> object:
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback(
                {
                    "kind": "run_start",
                    "phase": "prepare",
                    "iteration_total": 1,
                }
            )
            progress_callback(
                {
                    "kind": "phase",
                    "phase": "checkpoint",
                    "iteration_index": 0,
                    "iteration_total": 1,
                    "batch_index": 0,
                    "batch_total": 1,
                    "checkpoint_count": 1,
                }
            )
        QtCore.QThread.msleep(25)
        if callable(progress_callback):
            progress_callback(
                {
                    "kind": "phase",
                    "phase": "final_flush",
                    "iteration_total": 1,
                }
            )
        QtCore.QThread.msleep(25)
        if callable(progress_callback):
            progress_callback(
                {
                    "kind": "run_complete",
                    "phase": "completed",
                    "storage_filename": "run_flush.h5",
                    "iteration_total": 1,
                    "writer_stats_summary": {
                        "apply_elapsed_total": 0.1,
                        "durable_elapsed_total": 0.25,
                        "durability_mode": "flush_only",
                    },
                }
            )
        return {
            "storage_filename": "run_flush.h5",
            "run_state": "completed",
            "writer_checkpoint_count": 1,
            "writer_final_flush_s": 0.25,
        }

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _run_with_progress)
    window._run_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=1000)
    assert window.run_monitor.phase_chip.text() == "Complete"
    assert window.run_monitor.open_result_button.isEnabled() is True
    assert "durable flush" in window.run_monitor.telemetry_label.text().lower()
    assert "run_flush.h5" in window.run_status_label.text()
    window._dirty = False; window.close()


def test_run_simulation_successful_async_writer_result_opens_in_postprocess(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "run_async_success.h5"
    window = _make_run_window(monkeypatch)
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": str(result_file)})

    def _run_with_real_async_writer(**kwargs: object) -> object:
        progress_callback = kwargs.get("progress_callback")
        times = sgui.Time(
            [60061.0, 60061.0 + (1.0 / 86400.0), 60061.0 + (2.0 / 86400.0)],
            format="mjd",
            scale="utc",
        )
        power = (np.arange(6, dtype=np.float32).reshape(3, 2) + 1.0) * sgui.u.W
        sgui.scenario.init_simulation_results(
            str(result_file),
            write_mode="async",
            writer_queue_max_items=4,
            writer_queue_max_bytes=4096,
        )
        sgui.scenario.write_data(
            str(result_file),
            attrs={"tag": "gui-async-success"},
            constants={"sat_ids": np.asarray([0, 1], dtype=np.int32)},
            iteration=0,
            times=times,
            power=power,
        )
        sgui.scenario.flush_writes(str(result_file))
        writer_stats = sgui.scenario._get_writer_stats_snapshot(str(result_file))
        sgui.scenario.close_writer(str(result_file))
        if callable(progress_callback):
            progress_callback(
                {
                    "kind": "run_complete",
                    "phase": "completed",
                    "storage_filename": str(result_file),
                    "iteration_total": 1,
                    "writer_stats_summary": writer_stats,
                }
            )
        return {
            "storage_filename": str(result_file),
            "run_state": "completed",
            "writer_stats_summary": writer_stats,
        }

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _run_with_real_async_writer)
    window._run_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=1500)

    described = sgui.scenario.describe_data(str(result_file))

    assert window.run_monitor.open_result_button.isEnabled() is True
    assert "run complete" in window.run_status_label.text().lower()
    assert described["attrs"]["tag"] == "gui-async-success"
    assert "power" in described["iter"][0]["datasets"]

    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()
    assert window.postprocess_widget.current_filename() == str(result_file)
    window._dirty = False; window.close()


def test_run_simulation_force_stop_escalates_and_preserves_force_terminal_mode(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    applied_payloads: list[dict[str, object]] = []
    graceful_seen = threading.Event()
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": "partial_force.h5"})

    original_apply = window._apply_run_progress_event

    def _record_apply(payload: object) -> None:
        if isinstance(payload, dict):
            applied_payloads.append(dict(payload))
        original_apply(payload)

    monkeypatch.setattr(window, "_apply_run_progress_event", _record_apply)

    def _run_until_force_stop(**kwargs: object) -> object:
        cancel_callback = kwargs.get("cancel_callback")
        progress_callback = kwargs.get("progress_callback")
        while True:
            if callable(progress_callback):
                progress_callback(
                    {
                        "kind": "chunk",
                        "phase": "chunk_detail",
                        "iteration_index": 0,
                        "iteration_total": 1,
                        "batch_index": 0,
                        "batch_total": 1,
                        "chunk_index": 0,
                        "chunk_total": 1,
                        "description": "Working",
                    }
                )
            mode = str(cancel_callback()) if callable(cancel_callback) else "none"
            if mode == "graceful":
                graceful_seen.set()
            if mode == "force":
                if callable(progress_callback):
                    progress_callback(
                        {
                            "kind": "run_stopped",
                            "phase": "stopped",
                            "storage_filename": "partial_force.h5",
                            "stop_mode": "force",
                            "writer_stats_summary": {},
                        }
                    )
                return {"storage_filename": "partial_force.h5", "run_state": "stopped"}
            QtCore.QThread.msleep(10)

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _run_until_force_stop)
    window._run_simulation()
    _wait_until(lambda: window._run_worker is not None, timeout_ms=3000)
    window._request_stop_simulation()
    _wait_until(lambda: graceful_seen.is_set(), timeout_ms=3000)
    window._request_force_stop_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=5000)

    stop_request_modes = [
        str(payload.get("stop_mode"))
        for payload in applied_payloads
        if payload.get("phase") == "stop_requested"
    ]
    stopped_payloads = [
        payload
        for payload in applied_payloads
        if payload.get("kind") == "run_stopped"
    ]

    assert stop_request_modes == ["graceful", "force"]
    assert stopped_payloads[-1]["stop_mode"] == "force"
    assert window.run_monitor.phase_chip.text() == "Stopped"
    assert "partial_force.h5" in window.run_status_label.text()
    assert window.run_monitor.open_result_button.isEnabled() is True
    assert window.open_result_button.isEnabled() is True
    window._dirty = False; window.close()


def test_run_simulation_stopped_without_storage_keeps_postprocess_disabled(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": "ignored.h5"})

    def _run_stopped_without_storage(**kwargs: object) -> object:
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback(
                {
                    "kind": "run_stopped",
                    "phase": "stopped",
                    "stop_mode": "graceful",
                    "writer_stats_summary": {},
                }
            )
        return {"run_state": "stopped"}

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _run_stopped_without_storage)
    window._run_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=1000)

    assert window.run_status_label.text() == "Run stopped."
    assert window.run_monitor.open_result_button.isEnabled() is False
    assert window.open_result_button.isEnabled() is False
    window._dirty = False; window.close()


def test_stop_request_uses_shared_cancel_controller(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": "partial.h5"})
    run_started = threading.Event()
    release_run = threading.Event()
    captured_callbacks: dict[str, object] = {}

    def _run_with_cancel(**kwargs: object) -> object:
        cancel_callback = kwargs.get("cancel_callback")
        progress_callback = kwargs.get("progress_callback")
        captured_callbacks["cancel_callback"] = cancel_callback
        captured_callbacks["progress_callback"] = progress_callback
        run_started.set()
        if callable(progress_callback):
            progress_callback(
                {
                    "kind": "chunk",
                    "phase": "chunk_detail",
                    "iteration_index": 0,
                    "iteration_total": 1,
                    "batch_index": 0,
                    "batch_total": 1,
                    "chunk_index": 0,
                    "chunk_total": 1,
                    "description": "Working",
                }
            )
        release_run.wait(2.0)
        mode = str(cancel_callback()) if callable(cancel_callback) else "none"
        if mode in {"graceful", "force"}:
            if callable(progress_callback):
                progress_callback(
                    {
                        "kind": "run_stopped",
                        "phase": "stopped",
                        "storage_filename": "partial.h5",
                        "stop_mode": mode,
                        "writer_stats_summary": {},
                    }
                )
            return {"storage_filename": "partial.h5", "run_state": "stopped"}
        return {"storage_filename": "partial.h5", "run_state": "completed"}

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _run_with_cancel)
    window._run_simulation()
    _wait_until(lambda: window._run_worker is not None, timeout_ms=3000)
    _wait_until(lambda: run_started.is_set(), timeout_ms=3000)
    cancel_callback = captured_callbacks.get("cancel_callback")
    assert callable(cancel_callback)
    assert str(cancel_callback()) == "none"
    window._request_stop_simulation()
    assert str(cancel_callback()) == "graceful"
    _wait_until(
        lambda: "graceful stop requested" in window.run_status_label.text().lower(),
        timeout_ms=3000,
    )
    release_run.set()
    _wait_until(lambda: window._run_thread is None, timeout_ms=5000)
    window._dirty = False; window.close()


def test_run_simulation_finish_flushes_pending_throttled_progress_event(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    applied_payloads: list[dict[str, object]] = []
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": "flush_finish.h5"})

    original_apply = window._apply_run_progress_event

    def _record_apply(payload: object) -> None:
        if isinstance(payload, dict):
            applied_payloads.append(dict(payload))
        original_apply(payload)

    monkeypatch.setattr(window, "_apply_run_progress_event", _record_apply)

    def _run_with_pending_chunk(**kwargs: object) -> object:
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback(
                {
                    "kind": "chunk",
                    "phase": "chunk_detail",
                    "iteration_index": 0,
                    "iteration_total": 1,
                    "batch_index": 0,
                    "batch_total": 1,
                    "chunk_index": 0,
                    "chunk_total": 2,
                    "description": "Chunk 1",
                }
            )
            progress_callback(
                {
                    "kind": "chunk",
                    "phase": "chunk_detail",
                    "iteration_index": 0,
                    "iteration_total": 1,
                    "batch_index": 0,
                    "batch_total": 1,
                    "chunk_index": 1,
                    "chunk_total": 2,
                    "description": "Chunk 2",
                }
            )
            progress_callback(
                {
                    "kind": "run_complete",
                    "phase": "completed",
                    "storage_filename": "flush_finish.h5",
                    "iteration_total": 1,
                    "writer_stats_summary": {},
                }
            )
        return {"storage_filename": "flush_finish.h5", "run_state": "completed"}

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _run_with_pending_chunk)
    window._run_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=1000)

    chunk_indices = [int(payload["chunk_index"]) for payload in applied_payloads if payload.get("kind") == "chunk"]
    assert chunk_indices == [0, 1]
    assert applied_payloads[-1]["kind"] == "run_complete"
    assert "flush_finish.h5" in window.run_status_label.text()
    window._dirty = False; window.close()


def test_run_simulation_failure_flushes_pending_throttled_progress_event(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    window = _make_run_window(monkeypatch)
    applied_payloads: list[dict[str, object]] = []
    monkeypatch.setattr(window, "_build_run_request", lambda state: {"storage_filename": "flush_fail.h5"})

    original_apply = window._apply_run_progress_event

    def _record_apply(payload: object) -> None:
        if isinstance(payload, dict):
            applied_payloads.append(dict(payload))
        original_apply(payload)

    monkeypatch.setattr(window, "_apply_run_progress_event", _record_apply)

    def _run_then_fail(**kwargs: object) -> object:
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback(
                {
                    "kind": "chunk",
                    "phase": "chunk_detail",
                    "iteration_index": 0,
                    "iteration_total": 1,
                    "batch_index": 0,
                    "batch_total": 1,
                    "chunk_index": 0,
                    "chunk_total": 2,
                    "description": "Chunk 1",
                }
            )
            progress_callback(
                {
                    "kind": "chunk",
                    "phase": "chunk_detail",
                    "iteration_index": 0,
                    "iteration_total": 1,
                    "batch_index": 0,
                    "batch_total": 1,
                    "chunk_index": 1,
                    "chunk_total": 2,
                    "description": "Chunk 2",
                }
            )
        raise RuntimeError("boom after pending chunk")

    monkeypatch.setattr(sgui.scenario, "run_gpu_direct_epfd", _run_then_fail)
    window._run_simulation()
    _wait_until(lambda: window._run_thread is None, timeout_ms=1000)

    chunk_indices = [int(payload["chunk_index"]) for payload in applied_payloads if payload.get("kind") == "chunk"]
    assert chunk_indices == [0, 1]
    assert "boom after pending chunk" in window.run_status_label.text().lower()
    assert window.run_monitor.open_result_button.isEnabled() is False
    window._dirty = False; window.close()


def test_side_pane_toggle_controls_visibility(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    assert window._right_container.isHidden() is False
    assert window._right_container.isAncestorOf(window.side_pane_pin_button) is True
    assert window.side_pane_toggle_button.text() == "Assistant"
    assert window.side_pane_toggle_button.toolButtonStyle() == QtCore.Qt.ToolButtonTextBesideIcon
    assert window.side_pane_toggle_button.width() >= window.side_pane_toggle_button.height()
    window.side_pane_toggle_button.setChecked(False)
    qapp.processEvents()
    assert window._right_container.isHidden() is True
    window.side_pane_toggle_button.setChecked(True)
    qapp.processEvents()
    assert window._right_container.isHidden() is False
    window.side_pane_pin_button.setChecked(True)
    qapp.processEvents()
    assert window._side_pane_pinned is True
    window._dirty = False; window.close()


def test_side_pane_auto_hides_when_unpinned_and_stays_when_pinned(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_ASSISTANT_PANEL_AUTO_HIDE_MS", 80)
    window = sgui.ScepterMainWindow()
    window._set_workspace(sgui._WORKSPACE_SIMULATION)
    qapp.processEvents()

    monkeypatch.setattr(window, "_assistant_panel_interaction_active", lambda: False)
    window._restart_side_pane_auto_hide_timer()
    _wait_until(lambda: window._right_container.isHidden(), timeout_ms=1000, step_ms=20)
    assert window.side_pane_toggle_button.isChecked() is False

    window.side_pane_pin_button.setChecked(True)
    _wait_until(
        lambda: window._side_pane_pinned and not window._right_container.isHidden(),
        timeout_ms=1000,
        step_ms=20,
    )
    assert window.side_pane_toggle_button.isChecked() is True
    window._restart_side_pane_auto_hide_timer()
    QtCore.QThread.msleep(140)
    qapp.processEvents()
    assert window._right_container.isHidden() is False
    window._dirty = False; window.close()


def test_antenna_pattern_plot_buttons_create_embedded_canvas(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._set_all_antenna_defaults()
    window._plot_full_antenna_pattern()
    assert len(window._antenna_pattern_windows) == 1
    assert "pattern" in window.antenna_pattern_status.text().lower()
    window._plot_main_lobe_antenna_pattern()
    assert len(window._antenna_pattern_windows) == 2
    window._dirty = False; window.close()


def test_ras_antenna_pattern_plot_button_opens_window(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._set_all_antenna_defaults()
    window._plot_ras_antenna_pattern()
    assert len(window._antenna_pattern_windows) == 1
    assert "opened" in window.ras_pattern_status.text().lower()
    window._dirty = False; window.close()


def test_grouped_pick_resolves_from_hidden_pick_actor(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    viewer = _make_viewer(state_provider=_tiny_state)
    viewer._current_frame_set = _manual_frame_set()
    viewer._rebuild_scene(reset_camera=False)
    bundle = viewer._belt_render_bundles[0]

    class _Picker:
        def Pick(self, x: float, y: float, z: float, renderer: object) -> bool:
            del x, y, z, renderer
            return True

        def GetActor(self) -> object:
            return bundle.actor

        def GetPickPosition(self) -> tuple[float, float, float]:
            point = np.asarray(bundle.points_polydata.points[2], dtype=np.float32)
            return (float(point[0]), float(point[1]), float(point[2]))

    monkeypatch.setattr(sgui, "vtkPropPicker", _Picker)
    picked = viewer._pick_satellite_from_click(QtCore.QPoint(20, 20))
    assert picked == 2
    viewer.close()


def test_main_window_open_viewer_uses_embedded_plotter_only(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "QtInteractor", lambda parent, off_screen=None, auto_update=False: _DummyPlotter(parent))
    monkeypatch.setattr(
        sgui,
        "build_preview_frames",
        lambda state, params, *, cancel_callback=None: _manual_frame_set(),
    )
    window = sgui.ScepterMainWindow()
    top_before = {
        widget
        for widget in qapp.topLevelWidgets()
        if isinstance(widget, sgui.ConstellationViewerWindow)
    }
    window.open_viewer()
    qapp.processEvents()
    assert window._viewer_window is not None
    assert isinstance(window._viewer_window.plotter, _DummyPlotter)
    assert window._viewer_window.plotter.isWindow() is False
    top_after = {
        widget
        for widget in qapp.topLevelWidgets()
        if isinstance(widget, sgui.ConstellationViewerWindow)
    }
    assert len(top_after - top_before) == 1
    window._viewer_window.close()
    window._dirty = False; window.close()


def test_main_window_close_delegates_viewer_vtk_teardown_once(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(
        sgui,
        "QtInteractor",
        lambda parent, off_screen=None, auto_update=False: _DummyPlotter(parent),
    )
    monkeypatch.setattr(
        sgui,
        "build_preview_frames",
        lambda state, params, *, cancel_callback=None: _manual_frame_set(),
    )
    window = sgui.ScepterMainWindow()
    window.open_viewer()
    qapp.processEvents()

    viewer = window._viewer_window
    assert viewer is not None
    plotter = viewer.plotter
    assert isinstance(plotter, _DummyPlotter)

    # Test the cleanup path directly (close() is a no-op on the cached window)
    window._close_all_vtk_viewers()
    qapp.processEvents()

    assert plotter.close_calls == 1
    assert plotter.render_window.finalize_calls == 0


def test_viewer_buttons_are_enabled_on_blank_project(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    qapp.processEvents()
    assert window.open_viewer_button.isEnabled() is True
    assert window.home_open_viewer_button.isEnabled() is True
    window._dirty = False; window.close()


def test_constellation_wizard_close_delegates_viewer_vtk_teardown_once(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(
        sgui,
        "QtInteractor",
        lambda parent, off_screen=None, auto_update=False: _DummyPlotter(parent),
    )
    monkeypatch.setattr(
        sgui,
        "build_preview_frames",
        lambda state, params, *, cancel_callback=None: _manual_frame_set(),
    )
    wizard = sgui.ConstellationWizardDialog(_tiny_state().active_system().belts)
    viewer = wizard._viewer
    assert viewer is not None
    plotter = viewer.plotter
    assert isinstance(plotter, _DummyPlotter)

    wizard.close()
    qapp.processEvents()

    assert plotter.close_calls == 1
    assert plotter.render_window.finalize_calls == 0


def test_main_window_close_detaches_active_hexgrid_preview_thread(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    started = threading.Event()
    released = threading.Event()

    def _slow_hexgrid_run(self: sgui.HexgridPreviewWorker) -> None:
        started.set()
        while not released.wait(0.01):
            continue
        self.finished.emit(
            sgui.HexgridPreviewComputation(
                prepared_grid={},
                boresight_ids=np.empty(0, dtype=np.int32),
                reuse_plan=None,
                preview_notes=[],
                effective_cell_km=79.141,
            ),
            self._build_token,
        )
        try:
            current_thread = QtCore.QThread.currentThread()
            if current_thread is not None:
                current_thread.quit()
        except Exception:
            pass

    monkeypatch.setattr(sgui.HexgridPreviewWorker, "run", _slow_hexgrid_run)

    window = sgui.ScepterMainWindow()
    monkeypatch.setattr(window, "current_state", lambda: _tiny_state())
    monkeypatch.setattr(window, "_commit_visible_editors", lambda: None)
    monkeypatch.setattr(window, "_effective_hexgrid_cell_size_km", lambda state: 79.141)
    monkeypatch.setattr(window, "_hexgrid_preview_signature", lambda state: ("hexgrid",))
    monkeypatch.setattr(window, "_contour_is_current", lambda state: True)
    monkeypatch.setattr(window, "_render_hexgrid_preview_from_cache", lambda state: False)

    window._refresh_hexgrid_preview()
    _wait_until(lambda: started.is_set() and window._hexgrid_thread is not None, timeout_ms=1000)

    thread = window._hexgrid_thread
    assert thread is not None
    assert thread.parent() is None

    # Test the cleanup path directly (close() is a no-op on the cached window)
    window._cleanup_hexgrid_preview_worker()
    qapp.processEvents()

    assert window._hexgrid_thread is None
    released.set()
    _wait_until(lambda: _thread_is_stopped(thread), timeout_ms=2000)


def test_postprocess_widget_close_detaches_active_render_thread(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _stub_scene_assets(monkeypatch)
    started = threading.Event()
    released = threading.Event()
    browser_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _slow_postprocess_run(self: sgui.PostprocessRenderWorker) -> None:
        started.set()
        while not released.wait(0.01):
            continue
        self.finished.emit(
            {
                "engine": sgui.postprocess_recipes.ENGINE_PLOTLY,
                "html": "<html><body>ok</body></html>",
                "info": {},
            }
        )
        try:
            current_thread = QtCore.QThread.currentThread()
            if current_thread is not None:
                current_thread.quit()
        except Exception:
            pass

    monkeypatch.setattr(sgui.PostprocessRenderWorker, "run", _slow_postprocess_run)
    monkeypatch.setattr(
        sgui.webbrowser,
        "open",
        lambda *args, **kwargs: browser_calls.append((args, kwargs)) or True,
    )

    widget = sgui.PostprocessStudioWidget()
    widget.show()
    qapp.processEvents()
    widget._current_filename = "dummy.h5"
    recipe_id = next(iter(sgui.postprocess_recipes.RECIPE_BY_ID))
    monkeypatch.setattr(widget, "_current_recipe_id", lambda: recipe_id)
    monkeypatch.setattr(widget, "_current_recipe_params", lambda: {})
    monkeypatch.setattr(widget, "_selected_primary_power_dataset", lambda: None)
    monkeypatch.setattr(widget, "_current_system_filter", lambda: None)

    widget._start_recipe_render(
        action="open_browser",
        engine=sgui.postprocess_recipes.ENGINE_PLOTLY,
    )
    _wait_until(lambda: started.is_set() and widget._render_thread is not None, timeout_ms=1000)

    thread = widget._render_thread
    assert thread is not None
    assert thread.parent() is None

    widget.close()
    qapp.processEvents()

    assert widget._render_thread is None
    released.set()
    _wait_until(lambda: _thread_is_stopped(thread), timeout_ms=2000)
    assert browser_calls == []


def test_orbital_header_help_and_adjacent_offset_toggle(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._add_belt()
    qapp.processEvents()

    tooltip = window._belt_model.headerData(
        len(sgui.BeltTableModel._columns) - 1,
        QtCore.Qt.Horizontal,
        QtCore.Qt.ToolTipRole,
    )
    assert isinstance(tooltip, str)
    assert "phase-offset" in tooltip.lower() or "adjacent" in tooltip.lower()

    index = window._belt_model.index(0, len(sgui.BeltTableModel._columns) - 1)
    delegate = window.belt_table.itemDelegateForColumn(index.column())
    before = window._belt_model.data(index, QtCore.Qt.CheckStateRole)
    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_Space,
        QtCore.Qt.NoModifier,
    )
    assert delegate.editorEvent(key_event, window._belt_model, QtWidgets.QStyleOptionViewItem(), index) is True
    after = window._belt_model.data(index, QtCore.Qt.CheckStateRole)
    assert after != before
    assert delegate.editorEvent(key_event, window._belt_model, QtWidgets.QStyleOptionViewItem(), index) is True
    assert window._belt_model.data(index, QtCore.Qt.CheckStateRole) == before
    window._dirty = False; window.close()


def test_workspace_navigation_and_recent_files(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    cfg = tmp_path / "scenario.json"
    sgui.save_project_state(cfg, _tiny_state())
    result_file = tmp_path / "result.h5"
    _write_minimal_result_file(result_file)

    window = sgui.ScepterMainWindow()
    window._remember_recent_config(cfg)
    window._remember_recent_result(result_file)
    window._set_workspace(sgui._WORKSPACE_HOME)
    qapp.processEvents()

    assert window.current_workspace() == sgui._WORKSPACE_HOME
    assert window.home_recent_configs.count() == 1
    assert window.home_recent_results.count() == 1

    item = window.home_recent_results.item(0)
    assert item is not None
    window._open_recent_result_item(item)
    qapp.processEvents()

    assert window.current_workspace() == sgui._WORKSPACE_POSTPROCESS
    assert window.postprocess_widget.current_filename() == str(result_file)
    window._dirty = False; window.close()


def test_gui_workflow_roundtrip_interactions_remain_stable(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "workflow_result.h5"
    _write_minimal_result_file(result_file)

    monkeypatch.setattr(sgui, "QtInteractor", lambda parent, off_screen=None, auto_update=False: _DummyPlotter(parent))
    monkeypatch.setattr(
        sgui,
        "build_preview_frames",
        lambda state, params, *, cancel_callback=None: _manual_frame_set(),
    )
    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_names": ["Tiny_1"],
            "altitudes_q": np.asarray([525.0]) * sgui.u.km,
            "inclinations_q": np.asarray([53.0]) * sgui.u.deg,
            "min_elevations_q": np.asarray([20.0]) * sgui.u.deg,
            "belt_sats": np.asarray([56], dtype=np.int64),
        },
    )
    monkeypatch.setattr(sgui.earthgrid, "prepare_active_grid", lambda **kwargs: _manual_hexgrid_result())
    monkeypatch.setattr(
        sgui.earthgrid,
        "resolve_theta2_active_cell_ids",
        lambda *args, **kwargs: np.asarray([1], dtype=np.int32),
    )
    monkeypatch.setattr(
        sgui.visualise,
        "plot_cell_status_map",
        lambda *args, **kwargs: (
            Figure(),
            {
                "switched_off_count": 1,
                "normal_active_count": 1,
                "boresight_affected_active_count": 1,
                "map_style_used": str(kwargs.get("map_style", "clean")),
                "backend_used": "matplotlib",
            },
        ),
    )

    window = sgui.ScepterMainWindow()
    window.show()
    window._load_state_into_widgets(_tiny_state())
    window._last_analyser_selected_cell_km = 79.141
    window._last_analyser_signature = window._analyser_signature(window.current_state())

    for _ in range(2):
        window._set_workspace(sgui._WORKSPACE_HOME)
        qapp.processEvents()
        window.open_viewer()
        qapp.processEvents()
        assert window._viewer_window is not None
        _wait_for_viewer_build(window._viewer_window)
        window._viewer_window.close()
        qapp.processEvents()

        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        hex_idx = window.tab_widget.indexOf(window.hexgrid_tab)
        window.tab_widget.setCurrentIndex(hex_idx)
        qapp.processEvents()
        window._refresh_hexgrid_preview()
        _wait_until(lambda: window._hexgrid_thread is None and window._hexgrid_preview_window is not None)

        window._set_workspace(sgui._WORKSPACE_POSTPROCESS)
        qapp.processEvents()
        window._open_result_in_postprocess(str(result_file))
        window.postprocess_widget._render_current_recipe()
        _wait_for_postprocess_render(window.postprocess_widget)

    assert window._hexgrid_thread is None
    assert window.postprocess_widget._render_thread is None
    window._dirty = False; window.close()


def test_complexity_mode_hides_advanced_simulation_tabs(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    def _tab_visible(label: str) -> bool:
        for idx in range(window.tab_widget.count()):
            if window.tab_widget.tabText(idx) == label:
                return window.tab_widget.tabBar().isTabVisible(idx)
        raise AssertionError(label)

    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Basic"))
    qapp.processEvents()
    assert _tab_visible("Coverage & Contours") is True
    assert _tab_visible("Coverage & Boresight") is True
    assert window.grid_advanced_group.isHidden() is True
    assert window.hexgrid_preview_tuning_group.isHidden() is True
    assert window.hexgrid_boresight_scope_group.isHidden() is True
    assert window.runtime_advanced_group.isHidden() is True
    assert window.runtime_expert_notice.isHidden() is False

    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Advanced"))
    qapp.processEvents()
    assert _tab_visible("Coverage & Contours") is True
    assert _tab_visible("Coverage & Boresight") is True
    assert window.grid_advanced_group.isHidden() is False
    assert window.hexgrid_preview_tuning_group.isHidden() is False
    assert window.hexgrid_boresight_scope_group.isHidden() is False
    assert window.runtime_advanced_group.isHidden() is True
    assert window.runtime_expert_notice.isHidden() is False

    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Expert"))
    qapp.processEvents()
    assert window.runtime_advanced_group.isHidden() is False
    assert window.runtime_expert_notice.isHidden() is True
    window._dirty = False; window.close()


def test_runtime_gpu_controls_have_help_metadata_and_help_buttons(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    runtime_controls = [
        window.runtime_gpu_method_combo,
        window.runtime_gpu_compute_dtype_combo,
        window.runtime_gpu_output_dtype_combo,
        window.runtime_gpu_on_error_combo,
        window.runtime_host_memory_edit,
        window.runtime_gpu_memory_edit,
        window.runtime_memory_budget_combo,
        window.runtime_headroom_combo,
        window.runtime_profile_stages_checkbox,
        window.runtime_progress_desc_mode_combo,
        window.runtime_writer_checkpoint_interval_edit,
        window.runtime_force_bulk_timesteps_edit,
        window.runtime_hdf5_compression_combo,
        window.runtime_hdf5_compression_opts_spin,
        window.runtime_terminal_cleanup_checkbox,
        window.runtime_sat_frame_combo,
    ]
    for widget in runtime_controls:
        assert widget in window._help_registry
        assert window._help_registry[widget].summary
        assert window._help_registry[widget].details
        assert widget in window._help_buttons
    window._dirty = False; window.close()


def test_non_runtime_tabs_expose_help_buttons(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    # Check that key controls have help entries registered.  Not all
    # widgets get a physical ? button (widgets inside two-column grid
    # forms or checkable group boxes skip the wrapper), but they all
    # should have help registry entries so the help popover can show
    # content when the user clicks the field.
    for widget in (
        window.longitude_spin,
        window.latitude_spin,
        window.elevation_spin,
        window.service_nco_edit,
        window.service_nbeam_edit,
        window.service_selection_combo,
        window.service_cell_activity_edit,
        window.service_cell_seed_edit,
        window.target_pfd_edit,
        window.frequency_spin,
        window.antenna_model_combo,
        window.hexgrid_geography_mask_combo,
        window.hexgrid_ras_pointing_combo,
        window.hexgrid_ras_exclusion_mode_combo,
        window.hexgrid_map_style_combo,
        window.hexgrid_boresight_enabled_checkbox,
        window.hexgrid_boresight_theta1_spin,
        window.hexgrid_boresight_theta2_spin,
        window.hexgrid_boresight_scope_combo,
    ):
        assert widget in window._help_registry, f"{widget} missing from _help_registry"
    # At least some of these should also have physical ? buttons
    assert len(window._help_buttons) > 50, f"Expected 50+ help buttons, got {len(window._help_buttons)}"
    assert window.belt_columns_help_button in window._help_registry
    window._dirty = False; window.close()


def test_runtime_help_button_opens_popover_without_help_mode(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Expert"))
    window.tab_widget.setCurrentWidget(window.runtime_tab)
    qapp.processEvents()
    button = window._help_buttons[window.runtime_gpu_compute_dtype_combo]
    button.click()
    qapp.processEvents()
    assert window._help_popover.isVisible() is True
    assert "GPU compute dtype" in window._help_popover._title_label.text()
    assert "Safe default" in window._help_popover._body_label.text()
    window._dirty = False; window.close()


def test_runtime_store_eligible_mask_checkbox_is_expert_only(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    window.tab_widget.setCurrentWidget(window.runtime_tab)
    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Advanced"))
    qapp.processEvents()
    assert window.runtime_advanced_group.isHidden() is True

    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Expert"))
    qapp.processEvents()
    assert window.runtime_advanced_group.isHidden() is False
    assert window.runtime_store_eligible_mask_checkbox.isEnabled() is True
    assert window.runtime_store_eligible_mask_checkbox.isChecked() is False
    window._dirty = False; window.close()


def test_boresight_basic_mode_hides_advanced_theta2_and_hexgrid_tuning(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.tab_widget.setCurrentWidget(window.hexgrid_tab)
    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Basic"))
    qapp.processEvents()
    assert window.hexgrid_boresight_scope_group.isHidden() is True
    assert window.hexgrid_preview_tuning_group.isHidden() is True
    assert window.hexgrid_boresight_theta1_spin.isHidden() is False
    assert window.hexgrid_ras_pointing_combo.isHidden() is False
    assert window.hexgrid_ras_exclusion_mode_combo.isHidden() is False

    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Advanced"))
    qapp.processEvents()
    assert window.hexgrid_boresight_scope_group.isHidden() is False
    assert window.hexgrid_preview_tuning_group.isHidden() is False
    window._dirty = False; window.close()


def test_help_popover_auto_hides_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_HELP_POPOVER_TIMEOUT_MS", 120)
    window = sgui.ScepterMainWindow()
    window.show()
    window._set_workspace(sgui._WORKSPACE_SIMULATION)
    hidden = {"value": False}
    original_hide = window._help_popover.hide
    window._help_popover._timer.timeout.disconnect()

    def _hide_and_flag() -> None:
        hidden["value"] = True
        original_hide()

    window._help_popover._timer.timeout.connect(_hide_and_flag)
    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Expert"))
    window.tab_widget.setCurrentWidget(window.runtime_tab)
    qapp.processEvents()
    assert window._show_help_for(window.runtime_gpu_compute_dtype_combo) is True
    assert window._help_popover._anchor_widget is window.runtime_gpu_compute_dtype_combo
    assert window._help_popover._timer.isActive() is True
    _wait_until(lambda: hidden["value"], timeout_ms=1000, step_ms=20)
    assert window._help_popover._timer.isActive() is False
    window._dirty = False; window.close()


def test_help_popover_hides_on_outside_focus_transfer(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    window._set_workspace(sgui._WORKSPACE_SIMULATION)
    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Expert"))
    window.tab_widget.setCurrentWidget(window.runtime_tab)
    qapp.processEvents()
    assert window._show_help_for(window.runtime_gpu_compute_dtype_combo) is True
    _wait_until(lambda: window._help_popover.isVisible(), timeout_ms=1000, step_ms=20)
    window.eventFilter(window.runtime_storage_edit, QtCore.QEvent(QtCore.QEvent.FocusIn))
    _wait_until(lambda: not window._help_popover.isVisible(), timeout_ms=1000, step_ms=20)
    window._dirty = False; window.close()


def test_runtime_disabled_reason_tooltips_are_explained(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.runtime_include_atmosphere_checkbox.setChecked(False)
    window.runtime_hdf5_compression_combo.setCurrentIndex(
        window.runtime_hdf5_compression_combo.findData(None)
    )
    window._sync_runtime_controls()
    qapp.processEvents()
    assert "enabled when 'include atmosphere' is turned on" in window.runtime_atm_bin_edit.toolTip().lower()
    assert "only used when hdf5 compression is set to gzip" in window.runtime_hdf5_compression_opts_spin.toolTip().lower()
    window.runtime_hdf5_compression_combo.setCurrentIndex(
        window.runtime_hdf5_compression_combo.findData("gzip")
    )
    window._sync_runtime_controls()
    qapp.processEvents()
    assert "gzip compression level" in window.runtime_hdf5_compression_opts_spin.toolTip().lower()
    window._dirty = False; window.close()


def test_basic_visible_runtime_controls_use_basic_help_layer(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    for widget in (
        window.runtime_memory_budget_combo,
        window.runtime_headroom_combo,
        window.runtime_include_atmosphere_checkbox,
        window.target_pfd_edit,
    ):
        assert window._help_registry[widget].layer == "Basic"
    window._dirty = False; window.close()


def test_guidance_action_highlights_runtime_target_when_ready(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    highlighted: list[QtWidgets.QWidget | None] = []
    state = _tiny_state()
    state.active_system().grid_analysis.cell_size_override_enabled = True
    state.active_system().grid_analysis.cell_size_override_km = 55.0
    window._load_state_into_widgets(state)
    _mark_hexgrid_preview_current(window, effective_cell_km=55.0)
    window._refresh_summary()
    monkeypatch.setattr(window, "_highlight_widget", lambda widget: highlighted.append(widget))
    window._jump_to_guidance_target()
    _wait_until(lambda: window.tab_widget.currentWidget() is window.runtime_tab, timeout_ms=1000, step_ms=20)
    assert window.tab_widget.currentWidget() is window.runtime_tab
    assert highlighted[-1] is window.runtime_run_group
    window._dirty = False; window.close()


def test_snapshot_tracks_guidance_and_per_block_status(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    assert not hasattr(window, "snapshot_primary_action_button")
    assert not hasattr(window, "snapshot_secondary_action_button")
    assert "guidance next" in window.snapshot_detail_label.text().lower()
    assert not hasattr(window, "summary_text")
    blank_text = "\n".join(
        window.snapshot_step_list.item(idx).text().lower()
        for idx in range(window.snapshot_step_list.count())
    )
    assert "service & demand: needs attention" in blank_text
    assert "nco unset" in blank_text
    assert "ras station: needs attention" in blank_text

    state = _tiny_state()
    state.active_system().grid_analysis.cell_size_override_enabled = True
    state.active_system().grid_analysis.cell_size_override_km = 55.0
    window._load_state_into_widgets(state)
    _mark_hexgrid_preview_current(window, effective_cell_km=55.0)
    window._refresh_summary()
    qapp.processEvents()
    text = "\n".join(
        window.snapshot_step_list.item(idx).text().lower()
        for idx in range(window.snapshot_step_list.count())
    )
    assert "ras station: ready" in text
    assert "satellite orbitals: ready" in text
    assert "satellite antennas: ready" in text
    assert "coverage & boresight: ready" in text
    assert "review & run: ready for review" in text
    assert "review parameters and outputs" in window.snapshot_overview_label.text().lower()
    window._dirty = False; window.close()


def test_simulation_page_order_and_field_placement(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    labels = [window.tab_widget.tabText(idx) for idx in range(window.tab_widget.count())]
    assert labels == [
        "RAS Station",
        "Satellite Orbitals",
        "Satellite Antennas",
        "Service & Demand",
        "Spectrum & Reuse",
        "Coverage & Contours",
        "Coverage & Boresight",
        "Review & Run",
    ]
    assert window.ras_tab.isAncestorOf(window.antenna_spin) is True
    assert window.antennas_tab.isAncestorOf(window.antenna_spin) is False
    assert window.service_tab.isAncestorOf(window.target_pfd_edit) is True
    assert window.ras_tab.isAncestorOf(window.target_pfd_edit) is False
    assert window.target_pfd_edit.value_or_none() is None
    assert window.tab_widget.currentWidget() is window.ras_tab
    window._dirty = False; window.close()


def test_left_nav_no_longer_has_duplicate_viewer_button(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    assert hasattr(window, "open_viewer_button") is True
    assert hasattr(window, "home_open_viewer_button") is True
    assert hasattr(window, "open_viewer_quick_button") is False
    window._dirty = False; window.close()


def test_ras_and_satellite_antenna_defaults_are_separate(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.antenna_spin.set_value(99.0)
    window.ras_oper_min_spin.set_value(7.0)
    window.ras_oper_max_spin.set_value(77.0)
    window.frequency_spin.set_value(1234.0)

    window._set_satellite_antenna_defaults()
    assert window.antenna_spin.value_or_none() == pytest.approx(99.0)
    assert window.ras_oper_min_spin.value_or_none() == pytest.approx(7.0)
    assert window.ras_oper_max_spin.value_or_none() == pytest.approx(77.0)
    assert window.pattern_wavelength_spin.value_or_none() == pytest.approx(15.0)
    assert window.frequency_spin.value_or_none() == pytest.approx(_default_antennas().frequency_mhz)
    # RAS frequency may or may not be set by satellite antenna defaults
    # — it is now independent of the satellite frequency.

    window._remove_ras_antenna()
    assert window.antenna_spin.value_or_none() is None
    assert window.ras_oper_min_spin.value_or_none() is None
    assert window.ras_oper_max_spin.value_or_none() is None
    assert window.target_pfd_edit.value_or_none() is None

    window.pattern_wavelength_spin.set_value(31.0)
    window._set_ras_antenna_defaults()
    assert window.antenna_spin.value_or_none() is not None
    assert window.ras_oper_min_spin.value_or_none() is not None
    assert window.ras_oper_max_spin.value_or_none() is not None
    assert window.pattern_wavelength_spin.value_or_none() == pytest.approx(31.0)
    assert window.frequency_spin.value_or_none() == pytest.approx(_default_antennas().frequency_mhz)
    window._dirty = False; window.close()


def test_ras_and_satellite_frequency_editors_are_mirrored(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """RAS and satellite frequencies were decoupled in v0.25.0."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    window.frequency_spin.set_value(1234.5)
    qapp.processEvents()
    assert window.frequency_spin.value_or_none() == pytest.approx(1234.5)

    window.ras_frequency_spin.set_value(2690.0)
    qapp.processEvents()
    assert window.ras_frequency_spin.value_or_none() == pytest.approx(2690.0)
    # RAS and satellite frequencies are decoupled since v0.25.0 —
    # setting RAS frequency does not clear it.
    assert window.frequency_spin.value_or_none() == pytest.approx(1234.5)
    window._dirty = False; window.close()


def test_ras_defaults_enable_pattern_from_shared_frequency(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._add_ras_station()
    qapp.processEvents()
    window._set_ras_antenna_defaults()
    qapp.processEvents()
    assert window.plot_ras_pattern_button.isEnabled() is True
    assert "ready" in window.ras_pattern_status.text().lower()
    window._dirty = False; window.close()


def test_ras_pattern_requires_ras_frequency(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    # Start with a state that has RAS station but clear the receiver band
    state = _tiny_state()
    assert state.ras_station is not None
    state.ras_station = replace(state.ras_station, receiver_band_start_mhz=None, receiver_band_stop_mhz=None)
    window._load_state_into_widgets(state)
    window._refresh_summary()

    # RAS pattern depends only on the RAS receiver band frequency
    window._update_antenna_plot_controls()
    assert window.plot_ras_pattern_button.isEnabled() is False

    # Set RAS receiver band → button should enable
    window.ras_frequency_spin.set_value(2690.0)
    qapp.processEvents()
    window._update_antenna_plot_controls()
    assert window.plot_ras_pattern_button.isEnabled() is True

    # Clear RAS frequency → button should disable again
    window.ras_frequency_spin.set_value(None)
    qapp.processEvents()
    window._update_antenna_plot_controls()
    assert window.plot_ras_pattern_button.isEnabled() is False
    window._dirty = False; window.close()


def test_service_target_pfd_help_uses_service_side_wording(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    help_def = window._help_registry[window.target_pfd_edit]
    assert "earth surface" in help_def.summary.lower() or "ground" in help_def.summary.lower()
    assert "satellite system tries to deliver" in help_def.summary.lower()
    assert "cell centre" in help_def.details.lower() or "cell target" in help_def.details.lower()
    window._dirty = False; window.close()


def test_service_power_help_tracks_active_quantity_and_basis(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._set_combo_to_data(window.service_power_quantity_combo, "target_pfd")
    window._set_combo_to_data(window.service_power_basis_combo, "per_mhz")
    window._sync_service_power_controls()
    help_def = window._help_registry[window.service_power_value_stack]
    assert "target pfd" in help_def.title.lower()
    assert "dbw/m^2/mhz" in help_def.summary.lower()

    window._set_combo_to_data(window.service_power_quantity_combo, "satellite_eirp")
    window._set_combo_to_data(window.service_power_basis_combo, "per_channel")
    window._sync_service_power_controls()
    qapp.processEvents()

    help_def = window._help_registry[window.service_power_value_stack]
    assert "satellite eirp" in help_def.title.lower()
    assert "channel-total" in help_def.summary.lower() or "per channel" in help_def.summary.lower()
    assert "dbw" in help_def.details.lower()
    assert window._show_help_for(window.service_power_value_stack) is True
    assert "satellite eirp" in window._help_popover._title_label.text().lower()
    window._dirty = False; window.close()


def test_hexgrid_geography_mode_uses_shoreline_buffer_label_but_keeps_legacy_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    shoreline_idx = window.hexgrid_geography_mask_combo.findData("land_plus_nearshore_sea")

    assert shoreline_idx >= 0
    assert window.hexgrid_geography_mask_combo.itemText(shoreline_idx) == "land_plus_shoreline_buffer"
    assert window.hexgrid_geography_mask_combo.itemData(shoreline_idx) == "land_plus_nearshore_sea"
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_current_hexgrid_config_preserves_zero_and_negative_shoreline_buffer_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    window._set_combo_to_data(window.hexgrid_geography_mask_combo, "land_plus_nearshore_sea")

    window.hexgrid_shoreline_buffer_spin.setValue(0.0)
    config_zero = window._current_hexgrid_config()
    assert config_zero.shoreline_buffer_km == pytest.approx(0.0)

    window.hexgrid_shoreline_buffer_spin.setValue(-12.5)
    config_negative = window._current_hexgrid_config()
    assert config_negative.shoreline_buffer_km == pytest.approx(-12.5)
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_anchor_slot_ui_is_one_based_while_internal_state_stays_zero_based(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().spectrum.reuse_factor = 7
    state.active_system().spectrum.ras_anchor_reuse_slot = 2

    window._load_state_into_widgets(state)
    qapp.processEvents()

    assert window.spectrum_anchor_slot_spin.currentData() == 2
    window._set_combo_to_data(window.spectrum_anchor_slot_spin, 3)
    qapp.processEvents()

    assert window.current_state().active_system().spectrum.ras_anchor_reuse_slot == 3
    window._dirty = False; window.close()


def test_ras_frequency_help_mentions_shared_frequency_and_pattern_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    freq_help = window._help_registry[window.ras_frequency_spin]
    assert "shared" in freq_help.summary.lower()
    assert "satellite antennas" in freq_help.details.lower()

    ras_help = window._help_registry[window.antenna_spin]
    assert "shared frequency" in ras_help.details.lower() or "manual pattern wavelength" in ras_help.details.lower()
    window._dirty = False; window.close()


def test_hexgrid_readiness_tracks_commit_inputs_not_cosmetic_preview_style(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state())
    _mark_hexgrid_preview_current(window)
    assert window._hexgrid_outdated is False

    window.runtime_writer_checkpoint_interval_edit.set_value(45.0)
    qapp.processEvents()
    assert window._hexgrid_outdated is False

    next_idx = (window.hexgrid_map_style_combo.currentIndex() + 1) % max(
        window.hexgrid_map_style_combo.count(), 1
    )
    window.hexgrid_map_style_combo.setCurrentIndex(next_idx)
    qapp.processEvents()
    assert window._hexgrid_outdated is False

    # Shoreline buffer only affects the commit signature when geography
    # mask mode is "land_plus_nearshore_sea".
    nearshore_idx = window.hexgrid_geography_mask_combo.findData("land_plus_nearshore_sea")
    if nearshore_idx >= 0:
        window.hexgrid_geography_mask_combo.setCurrentIndex(nearshore_idx)
        qapp.processEvents()
        # Re-mark current after geography mode change
        _mark_hexgrid_preview_current(window)
    window.hexgrid_shoreline_buffer_spin.setValue(10.0)
    qapp.processEvents()
    assert window._hexgrid_outdated is True
    assert "out of date" in window.hexgrid_notice_banner._label.text().lower()
    window._dirty = False; window.close()


def test_hexgrid_apply_marks_step_current_without_preview(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state())
    window._last_analyser_selected_cell_km = 55.0
    window._last_analyser_signature = window._analyser_signature(window.current_state())
    window._apply_hexgrid_settings()
    qapp.processEvents()

    assert window._hexgrid_applied_signature == window._hexgrid_commit_signature(window.current_state())
    assert window._hexgrid_outdated is False
    ready_by_label = {
        window.simulation_page_list.item(idx).text(): bool(
            window.simulation_page_list.item(idx).data(QtCore.Qt.UserRole + 1)
        )
        for idx in range(window.simulation_page_list.count())
    }
    assert ready_by_label["Coverage & Boresight"] is True
    assert window.guidance_title_label.text() == "Review parameters and outputs"
    window._dirty = False; window.close()


def test_review_and_run_is_ready_for_review_not_completed(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state())
    _mark_hexgrid_preview_current(window)
    window._refresh_summary()

    status = window._simulation_page_statuses(window.current_state())["Review & Run"]
    assert status["ready"] is False
    assert status["review_pending"] is True
    assert "review" in str(status["title"]).lower()
    assert window.run_simulation_button.isEnabled() is True

    window._run_in_progress = True
    window._review_run_state = "running"
    status_running = window._simulation_page_statuses(window.current_state())["Review & Run"]
    assert status_running["ready"] is True
    assert "running" in str(status_running["title"]).lower()

    window._run_in_progress = False
    window._review_run_state = "completed"
    status_completed = window._simulation_page_statuses(window.current_state())["Review & Run"]
    assert status_completed["ready"] is True
    assert "finished" in str(status_completed["title"]).lower()
    window._dirty = False; window.close()


def test_contour_and_hexgrid_become_stale_when_satellite_antenna_inputs_change(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state())
    window._last_analyser_selected_cell_km = 61.25
    window._last_analyser_signature = window._analyser_signature(window.current_state())
    window._apply_hexgrid_settings()
    window._refresh_summary()

    before = window._simulation_page_statuses(window.current_state())
    assert before["Coverage & Contours"]["ready"] is True
    assert before["Coverage & Boresight"]["ready"] is True

    window.frequency_spin.set_value(2450.0)
    qapp.processEvents()
    window._refresh_summary()

    after = window._simulation_page_statuses(window.current_state())
    assert after["Coverage & Contours"]["ready"] is False
    assert after["Coverage & Boresight"]["ready"] is False
    assert "stale" in str(after["Coverage & Contours"]["message"]).lower()
    assert "stale" in window.grid_recommended_label.text().lower()
    assert "stale" in window.grid_effective_label.text().lower()
    window._dirty = False; window.close()


def test_contour_override_keeps_contour_and_hexgrid_ready_after_analyser_inputs_change(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state())
    window._last_analyser_selected_cell_km = 61.25
    window._last_analyser_signature = window._analyser_signature(window.current_state())
    window.cell_size_override_checkbox.setChecked(True)
    window.cell_size_override_spin.setValue(61.25)
    window._apply_hexgrid_settings()
    window._refresh_summary()

    window.frequency_spin.set_value(2450.0)
    qapp.processEvents()
    window._refresh_summary()

    status = window._simulation_page_statuses(window.current_state())
    assert status["Coverage & Contours"]["ready"] is True
    assert status["Coverage & Boresight"]["ready"] is True
    assert "stale" not in window.grid_effective_label.text().lower()
    window._dirty = False; window.close()


def test_contour_analyser_updates_guidance_immediately(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state())

    monkeypatch.setattr(
        sgui,
        "build_constellation_from_state",
        lambda state, **kw: {
            "belt_names": ["Tiny_1"],
            "altitudes_q": np.asarray([550.0]) * sgui.u.km,
            "min_elevations_q": np.asarray([35.0]) * sgui.u.deg,
            "max_betas_q": np.asarray([48.0]) * sgui.u.deg,
            "belt_sats": np.asarray([4], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        sgui,
        "_satellite_antenna_pattern_spec",
        lambda antennas: (lambda *args, **kwargs: np.zeros(1), 0.15 * sgui.u.m, {}),
    )
    monkeypatch.setattr(
        sgui.earthgrid,
        "summarize_contour_spacing",
        lambda *args, **kwargs: {
            "selected_cell_spacing_km": 61.25,
            "summary_lines": ["ok"],
        },
    )
    monkeypatch.setattr(sgui, "validate_project_state", lambda state: (True, "ok", []))
    window._run_grid_analyzer()
    _wait_until(
        lambda: "61.250 km" in window.grid_recommended_label.text(),
        timeout_ms=2000,
    )

    assert "61.250 km" in window.grid_recommended_label.text()
    assert window._simulation_page_for_target(window._guidance_target) is window.hexgrid_tab
    window._dirty = False; window.close()


def test_page_indicators_track_step_readiness(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_state())
    _mark_hexgrid_preview_current(window)
    window._refresh_summary()

    ready_by_label = {
        window.simulation_page_list.item(idx).text(): bool(
            window.simulation_page_list.item(idx).data(QtCore.Qt.UserRole + 1)
        )
        for idx in range(window.simulation_page_list.count())
    }
    assert ready_by_label["RAS Station"] is True
    assert ready_by_label["Satellite Orbitals"] is True
    assert ready_by_label["Satellite Antennas"] is True
    assert ready_by_label["Service & Demand"] is True
    assert ready_by_label["Coverage & Boresight"] is True
    assert ready_by_label["Review & Run"] is False
    window._dirty = False; window.close()


def test_service_bandwidth_drives_service_readiness_without_ticking_blank_spectrum_tab(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    state = _tiny_state()
    state.active_system().service = sgui._blank_service_config()
    state.active_system().spectrum = sgui._blank_spectrum_config()
    window = _make_run_window(monkeypatch, state=state, current_hexgrid=False)

    window.service_nco_edit.set_value(4)
    window.service_nbeam_edit.set_value(2)
    window.service_selection_combo.setCurrentIndex(
        window.service_selection_combo.findData("max_elevation")
    )
    window.service_cell_activity_edit.set_value(0.8)
    window.target_pfd_edit.set_value(-128.0)
    qapp.processEvents()
    window._refresh_summary()

    status_without_bandwidth = window._simulation_page_statuses(window.current_state())
    ready_without_bandwidth = _simulation_page_ready_by_label(window)
    assert status_without_bandwidth["Service & Demand"]["ready"] is False
    assert status_without_bandwidth["Spectrum & Reuse"]["ready"] is False
    assert ready_without_bandwidth["Service & Demand"] is False
    assert ready_without_bandwidth["Spectrum & Reuse"] is False

    window.service_bandwidth_edit.set_value(5.0)
    window.service_bandwidth_edit.editingFinished.emit()
    qapp.processEvents()
    window._refresh_summary()

    current_spectrum = window.current_state().active_system().spectrum
    status_with_bandwidth = window._simulation_page_statuses(window.current_state())
    ready_with_bandwidth = _simulation_page_ready_by_label(window)
    snapshot_text = "\n".join(
        window.snapshot_step_list.item(idx).text().lower()
        for idx in range(window.snapshot_step_list.count())
    )

    assert current_spectrum.service_band_start_mhz is not None
    assert current_spectrum.service_band_stop_mhz is not None
    assert current_spectrum.reuse_factor is not None
    assert window._workflow_has_explicit_spectrum_inputs() is False
    assert status_with_bandwidth["Service & Demand"]["ready"] is True
    assert status_with_bandwidth["Spectrum & Reuse"]["ready"] is False
    assert ready_with_bandwidth["Service & Demand"] is True
    assert ready_with_bandwidth["Spectrum & Reuse"] is False
    assert "service & demand: ready" in snapshot_text
    assert "spectrum & reuse: needs attention" in snapshot_text
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_defaults_count_as_explicit_configuration_for_workflow_status(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    state = _tiny_state()
    state.active_system().service = sgui._blank_service_config()
    state.active_system().spectrum = sgui._blank_spectrum_config()
    window = _make_run_window(monkeypatch, state=state, current_hexgrid=False)

    window.service_nco_edit.set_value(4)
    window.service_nbeam_edit.set_value(2)
    window.service_selection_combo.setCurrentIndex(
        window.service_selection_combo.findData("max_elevation")
    )
    window.service_cell_activity_edit.set_value(0.8)
    window.target_pfd_edit.set_value(-128.0)
    window.service_bandwidth_edit.set_value(5.0)
    window.service_bandwidth_edit.editingFinished.emit()
    qapp.processEvents()
    window._refresh_summary()

    before_defaults = window._simulation_page_statuses(window.current_state())
    assert before_defaults["Service & Demand"]["ready"] is True
    assert before_defaults["Spectrum & Reuse"]["ready"] is False
    assert window._workflow_has_explicit_spectrum_inputs() is False

    window._set_spectrum_defaults()
    qapp.processEvents()
    window._refresh_summary()

    after_defaults = window._simulation_page_statuses(window.current_state())
    ready_by_label = _simulation_page_ready_by_label(window)

    assert window._workflow_has_explicit_spectrum_inputs() is True
    assert after_defaults["Spectrum & Reuse"]["ready"] is True
    assert ready_by_label["Spectrum & Reuse"] is True
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_run_monitor_stop_states_are_rendered(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()
    window.run_monitor.update_from_event(
        {"kind": "control", "phase": "stop_requested", "stop_mode": "graceful"}
    )
    assert window.run_monitor.phase_chip.text() == "Stopping"
    assert "graceful stop requested" in window.run_monitor.status_text().lower()

    window.run_monitor.update_from_event(
        {
            "kind": "phase",
            "phase": "stopping",
            "stop_mode": "force",
            "description": "Stopping at chunk boundary.",
            "iteration_index": 0,
            "iteration_total": 1,
            "batch_index": 0,
            "batch_total": 2,
        }
    )
    assert "stopping at chunk boundary" in window.run_monitor.status_text().lower()

    window.run_monitor.update_from_event(
        {
            "kind": "run_stopped",
            "phase": "stopped",
            "storage_filename": "partial.h5",
            "stop_mode": "graceful",
            "writer_stats_summary": {
                "apply_elapsed_total": 0.1,
                "durable_elapsed_total": 0.2,
                "durability_mode": "flush_only",
            },
        }
    )
    assert window.run_monitor.phase_chip.text() == "Stopped"
    assert window.run_monitor.open_result_button.isEnabled() is True
    assert "partial results" in window.run_monitor.status_text().lower()
    window._dirty = False; window.close()


def test_postprocess_workspace_loads_result_and_renders_recipe(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "result.h5"
    _write_minimal_result_file(result_file)

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    assert window.postprocess_widget.result_path_edit.text().endswith("result.h5")
    assert "result.h5" not in window.postprocess_widget.result_summary_label.text()
    assert (
        window.postprocess_widget.result_summary_scroll.horizontalScrollBarPolicy()
        == QtCore.Qt.ScrollBarAlwaysOff
    )
    assert (
        window.postprocess_widget.result_summary_scroll.verticalScrollBarPolicy()
        == QtCore.Qt.ScrollBarAsNeeded
    )
    assert window.postprocess_widget.primary_power_combo.isEnabled() is True
    assert window.postprocess_widget.primary_power_combo.count() == 3
    assert window.postprocess_widget.primary_power_combo.currentData() == "Prx_total_W"
    assert window.postprocess_widget.recipe_list.count() >= 1
    assert not hasattr(window.postprocess_widget, "recipe_engine_combo")

    for row in range(window.postprocess_widget.recipe_list.count()):
        item = window.postprocess_widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "prx_total_distribution":
            window.postprocess_widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()
    window.postprocess_widget._render_current_recipe()
    _wait_for_postprocess_render(window.postprocess_widget)
    assert (
        window.postprocess_widget.plot_canvas_container.currentWidget()
        is window.postprocess_widget.plot_canvas
    )
    assert "source_used" in window.postprocess_widget.recipe_info_text.toPlainText()
    assert window.postprocess_widget.detach_plot_button.text() == "Open figure"
    assert window.postprocess_widget.browser_plot_button.text() == "Open in browser"
    window._dirty = False; window.close()


def test_postprocess_workspace_surfaces_missing_bandwidth_warning_and_scrollable_controls(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "legacy_missing_bandwidth.h5"
    _write_minimal_result_file(
        result_file,
        include_bandwidth_attrs=False,
    )

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    widget = window.postprocess_widget
    assert isinstance(widget.recipe_controls_scroll, QtWidgets.QScrollArea)
    assert widget.recipe_controls_scroll.widgetResizable() is True
    assert "bandwidth" in widget.result_summary_label.text().lower() or "raw" in widget.result_summary_label.text().lower()

    for row in range(widget.recipe_list.count()):
        item = widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "prx_total_distribution":
            widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()

    assert {
        "bandwidth_view_mode",
        "reference_bandwidth_mhz",
    } <= set(widget._recipe_param_widgets)
    assert "bandwidth" in widget.recipe_info_text.toPlainText().lower() or "raw" in widget.recipe_info_text.toPlainText().lower()
    window._dirty = False; window.close()


def test_postprocess_workspace_exposes_parameter_aware_recipes(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "result.h5"
    _write_minimal_result_file(result_file)
    monkeypatch.setattr(
        sgui.postprocess_recipes.visualise,
        "plot_hemisphere_2D",
        lambda *args, **kwargs: (Figure(), np.zeros(1734, dtype=np.float64)),
    )

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    def _select_recipe(recipe_id: str) -> None:
        for row in range(window.postprocess_widget.recipe_list.count()):
            item = window.postprocess_widget.recipe_list.item(row)
            if item is not None and item.data(QtCore.Qt.UserRole) == recipe_id:
                window.postprocess_widget.recipe_list.setCurrentRow(row)
                qapp.processEvents()
                return
        raise AssertionError(recipe_id)

    _select_recipe("prx_total_distribution")
    assert {"integrated", "corridor", "reference_levels_db", "show_margin", "margin_at"} <= set(
        window.postprocess_widget._recipe_param_widgets
    )
    integrated_combo = window.postprocess_widget._recipe_param_widgets["integrated"]
    corridor_combo = window.postprocess_widget._recipe_param_widgets["corridor"]
    reference_edit = window.postprocess_widget._recipe_param_widgets["reference_levels_db"]
    assert isinstance(integrated_combo, QtWidgets.QComboBox)
    assert isinstance(corridor_combo, QtWidgets.QComboBox)
    assert isinstance(reference_edit, QtWidgets.QPlainTextEdit)
    integrated_combo.setCurrentIndex(integrated_combo.findData(True))
    corridor_combo.setCurrentIndex(corridor_combo.findData(True))
    qapp.processEvents()
    assert window.postprocess_widget.recipe_source_combo.isEnabled() is False
    assert "raw data only" in window.postprocess_widget.recipe_availability_label.text().lower()

    _select_recipe("hemisphere_percentile_map")
    assert {
        "integrated",
        "integration_window_s",
        "worst_percent",
    } <= set(window.postprocess_widget._recipe_param_widgets)
    assert "override_colormap_min" not in window.postprocess_widget._recipe_param_widgets
    assert "colormap_min_pct" not in window.postprocess_widget._recipe_param_widgets
    hemisphere_integrated_combo = window.postprocess_widget._recipe_param_widgets["integrated"]
    assert isinstance(hemisphere_integrated_combo, QtWidgets.QComboBox)
    assert hemisphere_integrated_combo.currentData() is False
    window.postprocess_widget._render_current_recipe()
    _wait_for_postprocess_render(window.postprocess_widget)
    assert (
        window.postprocess_widget.plot_canvas_container.currentWidget()
        is window.postprocess_widget.plot_canvas
    )
    assert "primary_power_dataset" in window.postprocess_widget.recipe_info_text.toPlainText()

    _select_recipe("hemisphere_data_loss_map")
    assert {
        "integrated",
        "integration_window_s",
        "protection_criterion_db",
        "override_colormap_min",
        "colormap_min_pct",
        "override_colormap_max",
        "colormap_max_pct",
    } <= set(window.postprocess_widget._recipe_param_widgets)
    override_min_widget = window.postprocess_widget._recipe_param_widgets["override_colormap_min"]
    min_widget = window.postprocess_widget._recipe_param_widgets["colormap_min_pct"]
    override_max_widget = window.postprocess_widget._recipe_param_widgets["override_colormap_max"]
    max_widget = window.postprocess_widget._recipe_param_widgets["colormap_max_pct"]
    assert isinstance(override_min_widget, QtWidgets.QCheckBox)
    assert isinstance(min_widget, QtWidgets.QDoubleSpinBox)
    assert isinstance(override_max_widget, QtWidgets.QCheckBox)
    assert isinstance(max_widget, QtWidgets.QDoubleSpinBox)
    assert min_widget.isEnabled() is False
    assert max_widget.isEnabled() is False
    override_min_widget.setChecked(True)
    override_max_widget.setChecked(True)
    qapp.processEvents()
    assert min_widget.isEnabled() is True
    assert max_widget.isEnabled() is True
    window._dirty = False; window.close()


def test_postprocess_primary_power_selector_is_shared_per_file_and_updates_hemisphere_state(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "result_primary.h5"
    second_result_file = tmp_path / "result_primary_second.h5"
    _write_minimal_result_file(result_file)
    _write_minimal_result_file(second_result_file)

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    combo = window.postprocess_widget.primary_power_combo
    assert [combo.itemData(idx) for idx in range(combo.count())] == [
        "Prx_total_W",
        "EPFD_W_m2",
        "PFD_total_RAS_STATION_W_m2",
    ]

    for row in range(window.postprocess_widget.recipe_list.count()):
        item = window.postprocess_widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "hemisphere_percentile_map":
            window.postprocess_widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()

    combo.setCurrentIndex(combo.findData("EPFD_W_m2"))
    qapp.processEvents()
    assert combo.currentData() == "EPFD_W_m2"
    assert "epfd_w_m2" in window.postprocess_widget.recipe_info_text.toPlainText().lower()
    assert window.postprocess_widget.render_recipe_button.isEnabled() is True

    window.postprocess_widget.refresh_current_file()
    qapp.processEvents()
    assert window.postprocess_widget.primary_power_combo.currentData() == "EPFD_W_m2"

    combo = window.postprocess_widget.primary_power_combo
    combo.setCurrentIndex(combo.findData("PFD_total_RAS_STATION_W_m2"))
    qapp.processEvents()
    assert "skycell axis" in window.postprocess_widget.recipe_availability_label.text().lower()
    assert window.postprocess_widget.render_recipe_button.isEnabled() is False

    window._open_result_in_postprocess(str(second_result_file))
    qapp.processEvents()
    assert window.postprocess_widget.primary_power_combo.currentData() == "Prx_total_W"
    window._dirty = False; window.close()


def test_inspect_result_file_reports_missing_configured_preacc_power_families(tmp_path: Path) -> None:
    result_file = tmp_path / "missing_configured_preacc_power.h5"
    _write_result_file_missing_configured_preacc_power(result_file)

    inspection = sgui.postprocess_recipes.inspect_result_file(result_file)

    assert inspection["available_primary_power_datasets"] == ()
    assert inspection["primary_power_dataset"] is None
    assert inspection["available_preacc_power_families"] == ()
    assert {
        "prx_total_distribution",
        "epfd_distribution",
        "total_pfd_ras_distribution",
        "prx_elevation_heatmap",
    } <= set(inspection["missing_configured_preacc_power_families"])
    assert any(
        "configured preaccumulated power families are missing" in str(item).lower()
        for item in inspection["postprocess_warnings"]
    )


def test_postprocess_workspace_handles_missing_configured_preacc_power_families(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "missing_configured_preacc_power.h5"
    _write_result_file_missing_configured_preacc_power(result_file)

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    widget = window.postprocess_widget
    assert widget.primary_power_combo.isEnabled() is False
    assert widget.primary_power_combo.count() == 1
    assert widget.primary_power_combo.currentData() is None
    assert "no" in widget.primary_power_combo.currentText().lower() or "raw" in widget.primary_power_combo.currentText().lower()
    assert "configured preaccumulated power families are missing" in widget.result_summary_label.text().lower()
    assert widget.result_summary_scroll.viewport().objectName() == "postprocessResultSummaryViewport"
    assert widget.recipe_controls_scroll.viewport().objectName() == "postprocessRecipeControlsViewport"
    assert widget.detach_plot_button.property("plotStudioAction") is True
    assert widget.browser_plot_button.property("plotStudioAction") is True

    for row in range(widget.recipe_list.count()):
        item = widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "beam_overview_over_time":
            widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()
    assert "ready to render" in widget.recipe_availability_label.text().lower()
    assert widget.recipe_availability_label.property("postprocessTone") == "muted"
    assert widget.recipe_availability_label.styleSheet() == ""
    assert widget.render_recipe_button.isEnabled() is True

    for row in range(widget.recipe_list.count()):
        item = widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "prx_total_distribution":
            widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()
    assert "configured preaccumulated family" in widget.recipe_availability_label.text().lower()
    assert "missing from this file" in widget.recipe_availability_label.text().lower()
    assert widget.recipe_availability_label.property("postprocessTone") == "error"
    assert widget.recipe_availability_label.styleSheet() == ""
    assert "postprocess warnings" in widget.recipe_info_text.toPlainText().lower()
    assert widget.render_recipe_button.isEnabled() is False
    window._dirty = False; window.close()


def test_postprocess_preacc_only_power_recipe_remains_available_without_raw(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "preacc_only_power.h5"
    _write_result_file_preacc_power_no_raw(result_file)

    inspection = sgui.postprocess_recipes.inspect_result_file(result_file)
    capability = sgui.postprocess_recipes.recipe_capability(
        inspection["meta"],
        "prx_total_distribution",
        filename=result_file,
        primary_power_dataset=None,
    )
    hemisphere_capability = sgui.postprocess_recipes.recipe_capability(
        inspection["meta"],
        "hemisphere_percentile_map",
        filename=result_file,
        primary_power_dataset=None,
    )

    assert inspection["available_primary_power_datasets"] == ()
    assert inspection["missing_configured_preacc_power_families"] == ()
    assert capability["available_sources"] == (sgui.postprocess_recipes.SOURCE_PREACC,)
    assert capability["blocked_reason"] == ""
    assert hemisphere_capability["available_sources"] == ()
    assert "no primary power raw dataset" in hemisphere_capability["blocked_reason"].lower()

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    widget = window.postprocess_widget
    assert widget.primary_power_combo.isEnabled() is False
    assert widget.primary_power_combo.currentData() is None
    for row in range(widget.recipe_list.count()):
        item = widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "prx_total_distribution":
            widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()
    assert "ready to render from preaccumulated data" in widget.recipe_availability_label.text().lower()
    assert widget.render_recipe_button.isEnabled() is True
    window._dirty = False; window.close()


def test_postprocess_workspace_renders_integrated_distribution_using_time_units(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "result.h5"
    _write_minimal_result_file(result_file)

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    for row in range(window.postprocess_widget.recipe_list.count()):
        item = window.postprocess_widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "prx_total_distribution":
            window.postprocess_widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()

    integrated_combo = window.postprocess_widget._recipe_param_widgets["integrated"]
    assert isinstance(integrated_combo, QtWidgets.QComboBox)
    integrated_combo.setCurrentIndex(integrated_combo.findData(True))
    qapp.processEvents()

    assert window.postprocess_widget.render_recipe_button.isEnabled() is True
    window.postprocess_widget._render_current_recipe()
    _wait_for_postprocess_render(window.postprocess_widget)
    assert (
        window.postprocess_widget.plot_canvas_container.currentWidget()
        is window.postprocess_widget.plot_canvas
    )
    assert "source_used" in window.postprocess_widget.recipe_info_text.toPlainText()
    window._dirty = False; window.close()


def test_postprocess_workspace_locks_heatmap_bins_when_source_resolves_to_preaccumulated(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "result.h5"
    _write_minimal_result_file(result_file)

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    for row in range(window.postprocess_widget.recipe_list.count()):
        item = window.postprocess_widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "prx_elevation_heatmap":
            window.postprocess_widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()

    elev_widget = window.postprocess_widget._recipe_param_widgets["elevation_bin_step_deg"]
    value_widget = window.postprocess_widget._recipe_param_widgets["value_bin_step_db"]
    assert isinstance(elev_widget, QtWidgets.QDoubleSpinBox)
    assert isinstance(value_widget, QtWidgets.QDoubleSpinBox)
    assert elev_widget.isEnabled() is False
    assert value_widget.isEnabled() is False
    assert elev_widget.value() == pytest.approx(5.0)
    assert value_widget.value() == pytest.approx(0.5)
    assert "locked to the stored preaccumulated structure" in elev_widget.toolTip().lower()

    source_combo = window.postprocess_widget.recipe_source_combo
    assert isinstance(source_combo, QtWidgets.QComboBox)
    source_combo.setCurrentIndex(source_combo.findData(sgui.postprocess_recipes.SOURCE_RAW))
    qapp.processEvents()

    assert elev_widget.isEnabled() is True
    assert value_widget.isEnabled() is True
    window._dirty = False; window.close()


def test_postprocess_workspace_opens_browser_and_saves_html(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "result.h5"
    _write_minimal_result_file(result_file)
    opened_urls: list[str] = []
    monkeypatch.setattr(sgui.webbrowser, "open", lambda url: opened_urls.append(str(url)) or True)
    export_path = tmp_path / "interactive_plot.html"
    monkeypatch.setattr(
        sgui.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "HTML file (*.html)"),
    )

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    for row in range(window.postprocess_widget.recipe_list.count()):
        item = window.postprocess_widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "prx_total_distribution":
            window.postprocess_widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()

    window.postprocess_widget._open_plotly_browser()
    _wait_for_postprocess_render(window.postprocess_widget)
    assert opened_urls
    assert window.postprocess_widget._current_plotly_html is not None

    window.postprocess_widget._export_plotly_html()
    _wait_for_postprocess_render(window.postprocess_widget)
    assert export_path.exists()
    assert "plotly" in export_path.read_text(encoding="utf-8").lower()
    window._dirty = False; window.close()


def test_postprocess_workspace_shows_beam_overview_and_bool_controls(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "result.h5"
    _write_minimal_result_file(result_file)

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    recipe_ids = [
        str(window.postprocess_widget.recipe_list.item(row).data(QtCore.Qt.UserRole))
        for row in range(window.postprocess_widget.recipe_list.count())
        if window.postprocess_widget.recipe_list.item(row) is not None
    ]
    assert "beam_overview_over_time" in recipe_ids
    assert "beam_count_total_over_time" not in recipe_ids
    assert "beam_count_visible_over_time" not in recipe_ids
    assert "beam_demand_over_time" not in recipe_ids

    for row in range(window.postprocess_widget.recipe_list.count()):
        item = window.postprocess_widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "beam_overview_over_time":
            window.postprocess_widget.recipe_list.setCurrentRow(row)
            break
    qapp.processEvents()

    total_widget = window.postprocess_widget._recipe_param_widgets["show_total_beams"]
    visible_widget = window.postprocess_widget._recipe_param_widgets["show_visible_beams"]
    demand_widget = window.postprocess_widget._recipe_param_widgets["show_beam_demand"]
    difference_widget = window.postprocess_widget._recipe_param_widgets["show_demand_minus_service"]
    assert isinstance(total_widget, QtWidgets.QCheckBox)
    assert isinstance(visible_widget, QtWidgets.QCheckBox)
    assert isinstance(demand_widget, QtWidgets.QCheckBox)
    assert isinstance(difference_widget, QtWidgets.QCheckBox)
    assert total_widget.isChecked() is True
    assert visible_widget.isChecked() is True
    assert demand_widget.isChecked() is True
    assert difference_widget.isChecked() is True

    window.postprocess_widget._render_current_recipe()
    _wait_for_postprocess_render(window.postprocess_widget)
    assert len(window.postprocess_widget.plot_canvas.figure.axes[0].lines) == 5

    demand_widget.setChecked(False)
    qapp.processEvents()
    window.postprocess_widget._render_current_recipe()
    _wait_for_postprocess_render(window.postprocess_widget)
    assert len(window.postprocess_widget.plot_canvas.figure.axes[0].lines) == 4
    window._dirty = False; window.close()


def test_postprocess_distribution_reference_levels_render_for_matplotlib(tmp_path: Path) -> None:
    result_file = tmp_path / "reference_levels_mpl.h5"
    _write_minimal_result_file(result_file)

    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "prx_total_distribution",
        source_preference=sgui.postprocess_recipes.SOURCE_PREACC,
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
        params={
            "reference_levels_db": "Damage=-128.6; Saturation=-128.2",
            "show_margin": True,
            "margin_at": "p98",
        },
    )

    labels = fig.axes[0].get_legend_handles_labels()[1]
    text_blob = "\n".join(text.get_text() for text in fig.axes[0].texts)
    assert "Damage" in labels
    assert "Saturation" in labels
    assert "Margin vs p98" in text_blob
    assert [ref["label"] for ref in info["reference_lines"]] == ["Damage", "Saturation"]
    plt.close(fig)


def test_postprocess_distribution_explains_zero_stored_leakage_for_preacc_files(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "zero_preacc_leakage.h5"
    _write_result_file_zero_preacc_leakage(result_file)

    with pytest.raises(RuntimeError) as excinfo:
        sgui.postprocess_recipes.render_recipe(
            result_file,
            "prx_total_distribution",
            source_preference=sgui.postprocess_recipes.SOURCE_PREACC,
            engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
        )

    message = str(excinfo.value).lower()
    assert "zero spectral leakage into the ras band" in message
    assert "reuse factor f7" in message
    assert "enabled channel subset uses channels 1-7 of 14" in message
    assert "no enabled channel reaches the ras-adjacent service edge" in message


def test_postprocess_distribution_reference_levels_render_for_plotly(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    result_file = tmp_path / "reference_levels_plotly.h5"
    _write_minimal_result_file(result_file)

    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "prx_total_distribution",
        source_preference=sgui.postprocess_recipes.SOURCE_PREACC,
        engine=sgui.postprocess_recipes.ENGINE_PLOTLY,
        params={
            "reference_levels_db": "Damage=-128.6\nSaturation=-128.2",
            "show_margin": True,
            "margin_at": "p98",
        },
    )

    annotations = [str(getattr(annotation, "text", "")) for annotation in (fig.layout.annotations or ())]
    assert any("Damage" in text for text in annotations)
    assert any("Margin vs p98" in text for text in annotations)
    assert info["engine_used"] == sgui.postprocess_recipes.ENGINE_PLOTLY


def test_postprocess_distribution_p95_marker_lies_on_ccdf_curve(tmp_path: Path) -> None:
    result_file = tmp_path / "reference_levels_p95.h5"
    _write_minimal_result_file(result_file)

    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "prx_total_distribution",
        source_preference=sgui.postprocess_recipes.SOURCE_PREACC,
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
        params={
            "reference_levels_db": "Damage=-128.6",
            "show_margin": True,
            "margin_at": "p95",
        },
    )

    ax = fig.axes[0]
    marker_offsets = np.asarray(ax.collections[0].get_offsets(), dtype=np.float64)
    marker_x = float(marker_offsets[0, 0])
    marker_y = float(marker_offsets[0, 1])
    line = ax.lines[0]
    point = sgui.postprocess_recipes._ccdf_display_percentile_point(
        line.get_xdata(),
        line.get_ydata(),
        target_probability=0.05,
    )
    text_blob = "\n".join(text.get_text() for text in ax.texts)
    assert info["p95"] is not None
    assert point is not None
    assert marker_x == pytest.approx(float(point["x"]))
    assert marker_y == pytest.approx(float(point["y"]))
    assert "Margin vs p95" in text_blob
    plt.close(fig)


def test_postprocess_beam_count_ccdf_markers_follow_displayed_tail_geometry(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "beam_count_ccdf_tail_geometry.h5"
    _write_minimal_result_file(result_file)

    cases = (
        ("beam_count_full_network_ccdf", "p95", 0.05),
        ("beam_count_visible_ccdf", "p98", 0.02),
    )
    for recipe_id, margin_at, target_probability in cases:
        fig, info = sgui.postprocess_recipes.render_recipe(
            result_file,
            recipe_id,
            source_preference=sgui.postprocess_recipes.SOURCE_RAW,
            engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
            params={"margin_at": margin_at},
        )

        ax = fig.axes[0]
        marker_offsets = np.asarray(ax.collections[0].get_offsets(), dtype=np.float64)
        marker_x = float(marker_offsets[0, 0])
        marker_y = float(marker_offsets[0, 1])
        line = ax.lines[0]
        point = sgui.postprocess_recipes._ccdf_display_percentile_point(
            line.get_xdata(),
            line.get_ydata(),
            target_probability=target_probability,
        )
        text_blob = "\n".join(text.get_text() for text in ax.texts)
        assert point is not None
        assert info[margin_at] is not None
        assert marker_x == pytest.approx(float(point["x"]))
        assert marker_y == pytest.approx(float(point["y"]))
        assert margin_at in text_blob.lower()
        plt.close(fig)


def test_postprocess_distribution_raw_accepts_one_dimensional_sample_stream() -> None:
    fig, info = sgui.postprocess_recipes._plot_distribution_raw(
        np.asarray([-132.1, -131.7, -130.4], dtype=np.float64),
        title="Per-satellite PFD contribution CCDF",
        xlabel="PFD contribution [dBW/m^2/MHz]",
    )
    assert isinstance(fig, Figure)
    assert fig.axes
    assert info["p95"] is not None
    plt.close(fig)


def test_postprocess_per_satellite_pfd_distribution_handles_one_dimensional_raw_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_file = tmp_path / "per_satellite_pfd_raw_1d.h5"
    _write_minimal_result_file(result_file)
    stack_real = sgui.postprocess_recipes._stack_across_iterations

    def _stack_1d(filename: str | Path, dataset_name: str, **kwargs: object) -> np.ndarray:
        if dataset_name == "PFD_per_sat_RAS_STATION_W_m2":
            return np.asarray([1.0e-18, 2.0e-18, 4.0e-18], dtype=np.float64)
        return stack_real(filename, dataset_name, **kwargs)

    monkeypatch.setattr(sgui.postprocess_recipes, "_stack_across_iterations", _stack_1d)
    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "per_satellite_pfd_distribution",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
    )
    assert fig.axes
    assert info["p95"] is not None
    plt.close(fig)


def test_postprocess_beam_cap_recipe_renders_with_mocked_nbeam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "beam_cap.h5"
    _write_minimal_result_file(result_file)

    def _fake_run_beam_cap_sizing(filename: str | Path, *, config: object, **kwargs: object) -> dict[str, object]:
        assert Path(filename) == result_file
        assert getattr(config, "emit_progress_output", True) is False
        return {
            "beam_caps": np.asarray([40, 60, 80], dtype=np.int64),
            "selected_caps": {"simpson": 60, "full_reroute": 80},
            "policy_curves": {
                "simpson": {
                    "beam_caps": np.asarray([40, 60, 80], dtype=np.int64),
                    "tail_risk_percent": np.asarray([5.0, 2.0, 1.0], dtype=np.float64),
                    "delta_percent": np.asarray([8.0, 3.0, 1.5], dtype=np.float64),
                    "eps_percent": np.asarray([7.0, 2.5, 1.2], dtype=np.float64),
                },
                "full_reroute": {
                    "beam_caps": np.asarray([40, 60, 80], dtype=np.int64),
                    "tail_risk_percent": np.asarray([4.0, 1.5, 0.8], dtype=np.float64),
                    "delta_percent": np.asarray([7.0, 2.7, 1.1], dtype=np.float64),
                    "eps_percent": np.asarray([6.0, 2.2, 0.9], dtype=np.float64),
                },
            },
            "run_diagnostics": {"pure_reroute_backend_selected": None},
        }

    monkeypatch.setattr(sgui.postprocess_recipes.nbeam, "run_beam_cap_sizing", _fake_run_beam_cap_sizing)
    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "beam_cap_sizing_analysis",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={"beam_cap_min": 40, "beam_cap_max": 80, "nco_override": 0},
    )

    assert len(fig.axes) == 3
    assert info["selected_caps"]["simpson"] == 60
    assert "Simpson: 60" in info["selected_caps_summary"]
    assert "Full reroute: 80" in info["selected_caps_summary"]
    assert info["nco_used"] == 1
    top_axis_text = "\n".join(text.get_text() for text in fig.axes[0].texts)
    assert "Simpson: 60" in top_axis_text
    assert "Full reroute: 80" in top_axis_text
    plt.close(fig)


def test_postprocess_workspace_remembers_recipe_specific_sources_and_parameters(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "recipe_state.h5"
    _write_minimal_result_file(result_file)

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    def _select_recipe(recipe_id: str) -> None:
        for row in range(window.postprocess_widget.recipe_list.count()):
            item = window.postprocess_widget.recipe_list.item(row)
            if item is not None and item.data(QtCore.Qt.UserRole) == recipe_id:
                window.postprocess_widget.recipe_list.setCurrentRow(row)
                qapp.processEvents()
                return
        raise AssertionError(recipe_id)

    _select_recipe("prx_total_distribution")
    source_combo = window.postprocess_widget.recipe_source_combo
    source_combo.setCurrentIndex(source_combo.findData(sgui.postprocess_recipes.SOURCE_PREACC))
    reference_widget = window.postprocess_widget._recipe_param_widgets["reference_levels_db"]
    assert isinstance(reference_widget, QtWidgets.QPlainTextEdit)
    reference_widget.setPlainText("Damage=-100.0")
    qapp.processEvents()

    window.postprocess_widget.refresh_current_file()
    qapp.processEvents()
    assert window.postprocess_widget._current_recipe_id() == "prx_total_distribution"
    assert window.postprocess_widget.recipe_source_combo.currentData() == sgui.postprocess_recipes.SOURCE_PREACC
    reference_widget = window.postprocess_widget._recipe_param_widgets["reference_levels_db"]
    assert isinstance(reference_widget, QtWidgets.QPlainTextEdit)
    assert reference_widget.toPlainText() == "Damage=-100.0"

    _select_recipe("hemisphere_percentile_map")
    assert window.postprocess_widget.recipe_source_combo.currentData() == sgui.postprocess_recipes.SOURCE_RAW
    hemisphere_integrated = window.postprocess_widget._recipe_param_widgets["integrated"]
    worst_percent = window.postprocess_widget._recipe_param_widgets["worst_percent"]
    assert isinstance(hemisphere_integrated, QtWidgets.QComboBox)
    assert isinstance(worst_percent, QtWidgets.QDoubleSpinBox)
    hemisphere_integrated.setCurrentIndex(hemisphere_integrated.findData(True))
    worst_percent.setValue(5.0)
    qapp.processEvents()

    _select_recipe("prx_total_distribution")
    assert window.postprocess_widget.recipe_source_combo.currentData() == sgui.postprocess_recipes.SOURCE_PREACC
    reference_widget = window.postprocess_widget._recipe_param_widgets["reference_levels_db"]
    assert isinstance(reference_widget, QtWidgets.QPlainTextEdit)
    assert reference_widget.toPlainText() == "Damage=-100.0"

    _select_recipe("hemisphere_percentile_map")
    hemisphere_integrated = window.postprocess_widget._recipe_param_widgets["integrated"]
    worst_percent = window.postprocess_widget._recipe_param_widgets["worst_percent"]
    assert isinstance(hemisphere_integrated, QtWidgets.QComboBox)
    assert isinstance(worst_percent, QtWidgets.QDoubleSpinBox)
    assert hemisphere_integrated.currentData() is True
    assert worst_percent.value() == pytest.approx(5.0)

    window._set_workspace(sgui._WORKSPACE_HOME)
    qapp.processEvents()
    window._set_workspace(sgui._WORKSPACE_POSTPROCESS)
    qapp.processEvents()
    assert window.postprocess_widget._current_recipe_id() == "hemisphere_percentile_map"
    hemisphere_integrated = window.postprocess_widget._recipe_param_widgets["integrated"]
    worst_percent = window.postprocess_widget._recipe_param_widgets["worst_percent"]
    assert isinstance(hemisphere_integrated, QtWidgets.QComboBox)
    assert isinstance(worst_percent, QtWidgets.QDoubleSpinBox)
    assert hemisphere_integrated.currentData() is False
    assert worst_percent.value() == pytest.approx(2.0)

    other_file = tmp_path / "recipe_state_other.h5"
    _write_minimal_result_file(other_file)
    window.postprocess_widget.open_result_file(str(other_file))
    qapp.processEvents()
    _select_recipe("prx_total_distribution")
    assert window.postprocess_widget.recipe_source_combo.currentData() == sgui.postprocess_recipes.SOURCE_AUTO
    reference_widget = window.postprocess_widget._recipe_param_widgets["reference_levels_db"]
    assert isinstance(reference_widget, QtWidgets.QPlainTextEdit)
    assert reference_widget.toPlainText() == ""
    window._dirty = False; window.close()


def test_postprocess_workspace_groups_beam_cap_policies_in_two_by_two_grid(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    result_file = tmp_path / "beam_cap_layout.h5"
    _write_minimal_result_file(result_file)

    window = sgui.ScepterMainWindow()
    window._open_result_in_postprocess(str(result_file))
    qapp.processEvents()

    for row in range(window.postprocess_widget.recipe_list.count()):
        item = window.postprocess_widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "beam_cap_sizing_analysis":
            window.postprocess_widget.recipe_list.setCurrentRow(row)
            qapp.processEvents()
            break

    form = window.postprocess_widget.recipe_parameter_form
    policies_container = None
    for row in range(form.rowCount()):
        label_item = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
        field_item = form.itemAt(row, QtWidgets.QFormLayout.FieldRole)
        label_widget = label_item.widget() if label_item is not None else None
        field_widget = field_item.widget() if field_item is not None else None
        if isinstance(label_widget, QtWidgets.QLabel) and label_widget.text() == "Policies":
            policies_container = field_widget
            break
    assert isinstance(policies_container, QtWidgets.QWidget)
    layout = policies_container.layout()
    assert isinstance(layout, QtWidgets.QGridLayout)
    assert layout.count() == 4
    assert {
        "policy_simpson",
        "policy_full_reroute",
        "policy_no_reroute",
        "policy_pure_reroute",
    } <= set(window.postprocess_widget._recipe_param_widgets)
    window._dirty = False; window.close()


def test_postprocess_recipe_registry_covers_notebook_surface() -> None:
    recipe_ids = {recipe.recipe_id for recipe in sgui.postprocess_recipes.RECIPES}
    assert {
        "prx_total_distribution",
        "epfd_distribution",
        "total_pfd_ras_distribution",
        "per_satellite_pfd_distribution",
        "prx_elevation_heatmap",
        "per_satellite_pfd_elevation_heatmap",
        "hemisphere_percentile_map",
        "hemisphere_data_loss_map",
        "hemisphere_percentile_map_3d",
        "hemisphere_data_loss_map_3d",
        "beam_count_full_network_ccdf",
        "beam_count_visible_ccdf",
        "beam_overview_over_time",
        "beam_count_total_over_time",
        "beam_count_visible_over_time",
        "beam_demand_over_time",
        "beam_cap_sizing_analysis",
        "total_pfd_over_time",
    } <= recipe_ids


def test_hemisphere_recipe_capability_allows_instantaneous_without_times(tmp_path: Path) -> None:
    result_file = tmp_path / "hemisphere_no_times.h5"
    _write_minimal_result_file(result_file, include_times=False)
    meta = sgui.scenario.describe_data(str(result_file))

    instantaneous = sgui.postprocess_recipes.recipe_capability(
        meta,
        "hemisphere_percentile_map",
        params={"integrated": False},
        filename=result_file,
    )
    integrated = sgui.postprocess_recipes.recipe_capability(
        meta,
        "hemisphere_percentile_map",
        params={"integrated": True},
        filename=result_file,
    )

    assert instantaneous["available_sources"] == (sgui.postprocess_recipes.SOURCE_RAW,)
    assert instantaneous["blocked_reason"] == ""
    assert integrated["available_sources"] == ()
    assert "stored times dataset" in integrated["blocked_reason"].lower()


def test_hemisphere_instantaneous_render_works_without_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "hemisphere_render_no_times.h5"
    _write_minimal_result_file(result_file, include_times=False)
    calls: list[dict[str, object]] = []

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[Figure, np.ndarray]:
        calls.append(dict(kwargs))
        return Figure(), np.full(int(values_db.shape[-1]), 1.0, dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, "plot_hemisphere_2D", _fake_plot)

    _figure, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "hemisphere_percentile_map",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={"integrated": False},
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
    )

    assert len(calls) == 1
    assert str(calls[0]["title"]).startswith("Instantaneous")
    assert info["integrated"] is False


def test_hemisphere_primary_power_selection_changes_capability_and_render_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "hemisphere_primary_power.h5"
    _write_minimal_result_file(result_file)
    meta = sgui.scenario.describe_data(str(result_file))
    calls: list[dict[str, object]] = []

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[Figure, np.ndarray]:
        calls.append(dict(kwargs))
        return Figure(), np.full(int(values_db.shape[-1]), 1.0, dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, "plot_hemisphere_2D", _fake_plot)

    available = sgui.postprocess_recipes.inspect_result_file(result_file)
    assert tuple(available["available_primary_power_datasets"]) == (
        "Prx_total_W",
        "EPFD_W_m2",
        "PFD_total_RAS_STATION_W_m2",
    )

    epfd_capability = sgui.postprocess_recipes.recipe_capability(
        meta,
        "hemisphere_percentile_map",
        filename=result_file,
        primary_power_dataset="EPFD_W_m2",
    )
    pfd_capability = sgui.postprocess_recipes.recipe_capability(
        meta,
        "hemisphere_percentile_map",
        filename=result_file,
        primary_power_dataset="PFD_total_RAS_STATION_W_m2",
    )
    assert epfd_capability["available_sources"] == (sgui.postprocess_recipes.SOURCE_RAW,)
    assert pfd_capability["available_sources"] == ()
    assert "skycell axis" in pfd_capability["blocked_reason"].lower()

    _figure, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "hemisphere_percentile_map",
        primary_power_dataset="EPFD_W_m2",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={"integrated": False},
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
    )

    assert len(calls) == 1
    assert info["primary_power_dataset"] == "EPFD_W_m2"


@pytest.mark.parametrize(
    "recipe_id",
    (
        "hemisphere_percentile_map",
        "hemisphere_data_loss_map",
        "hemisphere_percentile_map_3d",
        "hemisphere_data_loss_map_3d",
    ),
)
def test_hemisphere_recipe_integration_window_gating_only_applies_when_integrated(
    recipe_id: str,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / f"{recipe_id}_short_span.h5"
    _write_minimal_result_file(
        result_file,
        times_days=np.asarray([0.0, 500.0 / 86400.0, 1000.0 / 86400.0], dtype=np.float64),
    )
    meta = sgui.scenario.describe_data(str(result_file))
    params = {"integration_window_s": 2000.0}

    instantaneous = sgui.postprocess_recipes.recipe_capability(
        meta,
        recipe_id,
        params={**params, "integrated": False},
        filename=result_file,
    )
    integrated = sgui.postprocess_recipes.recipe_capability(
        meta,
        recipe_id,
        params={**params, "integrated": True},
        filename=result_file,
    )

    assert instantaneous["available_sources"] == (sgui.postprocess_recipes.SOURCE_RAW,)
    assert instantaneous["blocked_reason"] == ""
    assert integrated["available_sources"] == ()
    assert "requested integration window" in integrated["blocked_reason"].lower()


@pytest.mark.parametrize(
    "recipe_id, plot_attr",
    (
        ("hemisphere_percentile_map", "plot_hemisphere_2D"),
        ("hemisphere_data_loss_map", "plot_hemisphere_2D"),
        ("hemisphere_percentile_map_3d", "plot_hemisphere_3D"),
        ("hemisphere_data_loss_map_3d", "plot_hemisphere_3D"),
    ),
)
@pytest.mark.parametrize("integrated", (False, True))
def test_hemisphere_recipes_report_selected_mode_in_render_info(
    recipe_id: str,
    plot_attr: str,
    integrated: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / f"{recipe_id}_{'integrated' if integrated else 'instantaneous'}.h5"
    _write_minimal_result_file(result_file)
    calls: list[dict[str, object]] = []

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[Figure, np.ndarray]:
        del values_db
        calls.append(dict(kwargs))
        return Figure(), np.full(1734, 1.0, dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, plot_attr, _fake_plot)

    _figure, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        recipe_id,
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={"integrated": integrated},
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
    )

    expected_prefix = "Integrated" if integrated else "Instantaneous"
    assert len(calls) == 1
    assert str(calls[0]["title"]).startswith(expected_prefix)
    assert info["integrated"] is integrated
    assert str(info["title"]).startswith(expected_prefix)
    assert str(info["recipe_label"]).startswith(expected_prefix)


@pytest.mark.parametrize(
    "recipe_id, plot_attr",
    (
        ("hemisphere_data_loss_map", "plot_hemisphere_2D"),
        ("hemisphere_data_loss_map_3d", "plot_hemisphere_3D"),
    ),
)
@pytest.mark.parametrize(
    "engine, expected_engine",
    (
        (sgui.postprocess_recipes.ENGINE_MATPLOTLIB, "mpl"),
        (sgui.postprocess_recipes.ENGINE_PLOTLY, "plotly"),
    ),
)
@pytest.mark.parametrize("integrated", (False, True))
def test_hemisphere_data_loss_recipes_forward_explicit_colormap_limits(
    recipe_id: str,
    plot_attr: str,
    engine: str,
    expected_engine: str,
    integrated: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / f"{recipe_id}_{engine}.h5"
    _write_minimal_result_file(result_file)
    calls: list[dict[str, object]] = []

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[object, np.ndarray]:
        del values_db
        calls.append(dict(kwargs))
        payload = Figure() if kwargs.get("engine") == "mpl" else object()
        return payload, np.full(1734, 100.0, dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, plot_attr, _fake_plot)

    _figure, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        recipe_id,
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={
            "integrated": integrated,
            "override_colormap_min": True,
            "colormap_min_pct": 0.0,
            "override_colormap_max": True,
            "colormap_max_pct": 100.0,
        },
        engine=engine,
    )

    assert len(calls) == 1
    assert calls[0]["engine"] == expected_engine
    assert calls[0]["vmin"] == pytest.approx(0.0)
    assert calls[0]["vmax"] == pytest.approx(100.0)
    assert info["integrated"] is integrated


def test_hemisphere_data_loss_recipe_keeps_auto_colormap_limits_when_overrides_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "hemisphere_data_loss_auto.h5"
    _write_minimal_result_file(result_file)
    calls: list[dict[str, object]] = []

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[Figure, np.ndarray]:
        del values_db
        calls.append(dict(kwargs))
        return Figure(), np.full(1734, 100.0, dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, "plot_hemisphere_2D", _fake_plot)

    _figure, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "hemisphere_data_loss_map",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={},
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
    )

    assert len(calls) == 1
    assert "vmin" not in calls[0]
    assert "vmax" not in calls[0]
    assert info["loss_summary_avg_pct"] is None
    assert info["loss_summary_max_pct"] is None
    assert info["loss_summary_annotated"] is False


def test_hemisphere_data_loss_recipe_rejects_invalid_explicit_colormap_limits(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "hemisphere_data_loss_invalid.h5"
    _write_minimal_result_file(result_file)

    with pytest.raises(ValueError, match="Colormap max"):
        sgui.postprocess_recipes.render_recipe(
            result_file,
            "hemisphere_data_loss_map",
            source_preference=sgui.postprocess_recipes.SOURCE_RAW,
            params={
                "override_colormap_min": True,
                "colormap_min_pct": 80.0,
                "override_colormap_max": True,
                "colormap_max_pct": 20.0,
            },
            engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
        )


def _expected_hemisphere_summary_text(
    metric_label: str,
    avg_value: float,
    max_value: float,
    unit: str,
) -> str:
    if unit == "%":
        avg_text = f"{avg_value:.2f} %"
        max_text = f"{max_value:.2f} %"
    else:
        avg_text = f"{avg_value:.2f} {unit}"
        max_text = f"{max_value:.2f} {unit}"
    return f"{metric_label} avg: {avg_text} | max: {max_text}"


@pytest.mark.parametrize(
    "recipe_id",
    (
        "hemisphere_percentile_map",
        "hemisphere_data_loss_map",
        "hemisphere_percentile_map_3d",
        "hemisphere_data_loss_map_3d",
    ),
)
def test_hemisphere_recipes_expose_shared_summary_toggle(recipe_id: str) -> None:
    recipe = sgui.postprocess_recipes.RECIPE_BY_ID[recipe_id]
    summary_param = next(
        spec for spec in recipe.parameter_specs if spec.name == "show_summary_stats"
    )
    assert summary_param.label == "Show avg/max summary"
    assert summary_param.default is False
    assert all(spec.name != "show_loss_summary_stats" for spec in recipe.parameter_specs)


@pytest.mark.parametrize(
    ("recipe_id", "plot_attr", "is_3d", "expected_label", "expected_unit"),
    (
        ("hemisphere_percentile_map", "plot_hemisphere_2D", False, "Input power", "dBW"),
        ("hemisphere_data_loss_map", "plot_hemisphere_2D", False, "Data loss", "%"),
        ("hemisphere_percentile_map_3d", "plot_hemisphere_3D", True, "Input power", "dBW"),
        ("hemisphere_data_loss_map_3d", "plot_hemisphere_3D", True, "Data loss", "%"),
    ),
)
def test_hemisphere_recipe_can_annotate_summary_stats_matplotlib(
    recipe_id: str,
    plot_attr: str,
    is_3d: bool,
    expected_label: str,
    expected_unit: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / f"{recipe_id}_summary_stats_mpl.h5"
    _write_minimal_result_file(result_file)

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[Figure, np.ndarray]:
        del values_db, kwargs
        fig = Figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig.add_subplot(111, projection="3d")
        else:
            fig.add_subplot(111)
        return fig, np.asarray([10.0, 20.0, 30.0, np.nan], dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, plot_attr, _fake_plot)

    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        recipe_id,
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={"show_summary_stats": True},
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
    )

    assert info["summary_avg_value"] == pytest.approx(20.0)
    assert info["summary_max_value"] == pytest.approx(30.0)
    assert info["summary_metric_label"] == expected_label
    assert info["summary_unit"] == expected_unit
    assert info["summary_annotated"] is True
    footer_text = "\n".join(text.get_text() for text in fig.texts)
    assert _expected_hemisphere_summary_text(expected_label, 20.0, 30.0, expected_unit) in footer_text
    if "data_loss" in recipe_id:
        assert info["loss_summary_avg_pct"] == pytest.approx(20.0)
        assert info["loss_summary_max_pct"] == pytest.approx(30.0)
        assert info["loss_summary_annotated"] is True
    else:
        assert info["loss_summary_avg_pct"] is None
        assert info["loss_summary_max_pct"] is None
        assert info["loss_summary_annotated"] is False
    plt.close(fig)


@pytest.mark.parametrize(
    ("recipe_id", "plot_attr", "expected_label", "expected_unit"),
    (
        ("hemisphere_percentile_map", "plot_hemisphere_2D", "Input power", "dBW"),
        ("hemisphere_data_loss_map", "plot_hemisphere_2D", "Data loss", "%"),
        ("hemisphere_percentile_map_3d", "plot_hemisphere_3D", "Input power", "dBW"),
        ("hemisphere_data_loss_map_3d", "plot_hemisphere_3D", "Data loss", "%"),
    ),
)
def test_hemisphere_recipe_can_annotate_summary_stats_plotly(
    recipe_id: str,
    plot_attr: str,
    expected_label: str,
    expected_unit: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / f"{recipe_id}_summary_stats_plotly.h5"
    _write_minimal_result_file(result_file)

    class _DummyPlotlyFigure:
        def __init__(self) -> None:
            self.annotations: list[dict[str, object]] = []
            self.layout_updates: list[dict[str, object]] = []
            self.layout = types.SimpleNamespace(
                margin=types.SimpleNamespace(b=16),
            )

        def add_annotation(self, **kwargs: object) -> None:
            self.annotations.append(dict(kwargs))

        def update_layout(self, **kwargs: object) -> None:
            self.layout_updates.append(dict(kwargs))
            margin = kwargs.get("margin")
            if isinstance(margin, dict) and "b" in margin:
                self.layout.margin.b = int(margin["b"])

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[_DummyPlotlyFigure, np.ndarray]:
        del values_db, kwargs
        return _DummyPlotlyFigure(), np.asarray([5.0, 15.0, 25.0], dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, plot_attr, _fake_plot)

    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        recipe_id,
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={"show_summary_stats": True},
        engine=sgui.postprocess_recipes.ENGINE_PLOTLY,
    )

    assert info["summary_avg_value"] == pytest.approx(15.0)
    assert info["summary_max_value"] == pytest.approx(25.0)
    assert info["summary_metric_label"] == expected_label
    assert info["summary_unit"] == expected_unit
    assert info["summary_annotated"] is True
    assert fig.layout.margin.b >= 80
    assert len(fig.annotations) == 1
    assert fig.annotations[0]["x"] == pytest.approx(0.5)
    assert fig.annotations[0]["y"] == pytest.approx(0.01)
    assert _expected_hemisphere_summary_text(expected_label, 15.0, 25.0, expected_unit) in str(
        fig.annotations[0]["text"]
    )
    if "data_loss" in recipe_id:
        assert info["loss_summary_avg_pct"] == pytest.approx(15.0)
        assert info["loss_summary_max_pct"] == pytest.approx(25.0)
        assert info["loss_summary_annotated"] is True
    else:
        assert info["loss_summary_avg_pct"] is None
        assert info["loss_summary_max_pct"] is None
        assert info["loss_summary_annotated"] is False


def test_hemisphere_data_loss_recipe_accepts_legacy_summary_toggle_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "hemisphere_data_loss_summary_alias.h5"
    _write_minimal_result_file(result_file)

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[Figure, np.ndarray]:
        del values_db, kwargs
        fig = Figure()
        fig.add_subplot(111)
        return fig, np.asarray([5.0, 15.0, 25.0], dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, "plot_hemisphere_2D", _fake_plot)

    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "hemisphere_data_loss_map",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={"show_loss_summary_stats": True},
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
    )

    assert info["summary_avg_value"] == pytest.approx(15.0)
    assert info["summary_max_value"] == pytest.approx(25.0)
    assert info["summary_metric_label"] == "Data loss"
    assert info["summary_unit"] == "%"
    assert info["summary_annotated"] is True
    assert info["loss_summary_avg_pct"] == pytest.approx(15.0)
    assert info["loss_summary_max_pct"] == pytest.approx(25.0)
    assert info["loss_summary_annotated"] is True
    footer_text = "\n".join(text.get_text() for text in fig.texts)
    assert _expected_hemisphere_summary_text("Data loss", 15.0, 25.0, "%") in footer_text
    plt.close(fig)


def test_hemisphere_recipe_skips_summary_stats_when_no_finite_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "hemisphere_summary_empty.h5"
    _write_minimal_result_file(result_file)

    def _fake_plot(values_db: np.ndarray, **kwargs: object) -> tuple[Figure, np.ndarray]:
        del values_db, kwargs
        fig = Figure()
        fig.add_subplot(111)
        return fig, np.asarray([np.nan, np.nan], dtype=np.float64)

    monkeypatch.setattr(sgui.postprocess_recipes.visualise, "plot_hemisphere_2D", _fake_plot)

    fig, info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "hemisphere_data_loss_map",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={"show_summary_stats": True},
        engine=sgui.postprocess_recipes.ENGINE_MATPLOTLIB,
    )

    assert info["summary_avg_value"] is None
    assert info["summary_max_value"] is None
    assert info["summary_metric_label"] == "Data loss"
    assert info["summary_unit"] == "%"
    assert info["summary_annotated"] is False
    assert info["loss_summary_avg_pct"] is None
    assert info["loss_summary_max_pct"] is None
    assert info["loss_summary_annotated"] is False
    assert len(fig.texts) == 0
    plt.close(fig)


def test_hemisphere_polar_top_azimuth_label_sits_above_arrow_tip_and_clears_title() -> None:
    fig = sgui.visualise.plot_hemisphere_2D(
        np.linspace(0.0, 1.0, 2334, dtype=np.float64).reshape(1, 2334),
        mode="data_loss",
        protection_criterion=0.5,
        projection="polar",
        engine="mpl",
        title="Instantaneous hemisphere data-loss map",
        show=False,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    title_bbox = ax.title.get_window_extent(renderer)
    az0_text = next(text for text in ax.texts if text.get_text() == "Az 0°")
    label_bbox = az0_text.get_window_extent(renderer)
    arrow_tip = ax.transData.transform((0.0, 1.1))
    assert label_bbox.y0 > arrow_tip[1]
    assert not title_bbox.overlaps(label_bbox)
    plt.close(fig)


def test_hemisphere_polar_side_and_bottom_labels_sit_outside_arrows_and_clear_colorbar() -> None:
    fig = sgui.visualise.plot_hemisphere_2D(
        np.linspace(0.0, 1.0, 2334, dtype=np.float64).reshape(1, 2334),
        mode="power",
        projection="polar",
        engine="mpl",
        title="Instantaneous input power percentile map",
        show=False,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    colorbar_bbox = fig.axes[1].get_window_extent(renderer)
    labels = {
        text.get_text(): text.get_window_extent(renderer)
        for text in ax.texts
        if text.get_text().startswith("Az ")
    }
    arrow_90 = ax.transData.transform((1.1, 0.0))
    arrow_180 = ax.transData.transform((0.0, -1.1))
    arrow_270 = ax.transData.transform((-1.1, 0.0))

    assert labels["Az 90°"].x0 > arrow_90[0]
    assert labels["Az 90°"].x1 < colorbar_bbox.x0
    assert labels["Az 180°"].y1 < arrow_180[1]
    assert labels["Az 270°"].x1 < arrow_270[0]
    plt.close(fig)


@pytest.mark.parametrize(
    "recipe_id",
    (
        "beam_count_total_over_time",
        "beam_count_visible_over_time",
        "beam_demand_over_time",
    ),
)
def test_postprocess_beam_time_series_match_between_raw_and_2d_preacc(
    recipe_id: str,
    tmp_path: Path,
) -> None:
    result_file = tmp_path / f"{recipe_id}.h5"
    _write_minimal_result_file_with_2d_preacc_beams(result_file)

    fig_preacc, _info_preacc = sgui.postprocess_recipes.render_recipe(
        result_file,
        recipe_id,
        source_preference=sgui.postprocess_recipes.SOURCE_PREACC,
    )
    fig_raw, _info_raw = sgui.postprocess_recipes.render_recipe(
        result_file,
        recipe_id,
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
    )

    preacc_line = fig_preacc.axes[0].lines[0]
    raw_line = fig_raw.axes[0].lines[0]
    np.testing.assert_allclose(preacc_line.get_xdata(), raw_line.get_xdata())
    np.testing.assert_allclose(preacc_line.get_ydata(), raw_line.get_ydata())
    plt.close(fig_preacc)
    plt.close(fig_raw)


def test_postprocess_beam_overview_difference_matches_between_raw_and_preacc(tmp_path: Path) -> None:
    result_file = tmp_path / "beam_overview_difference.h5"
    _write_minimal_result_file_with_2d_preacc_beams(result_file)

    params = {
        "show_total_beams": False,
        "show_visible_beams": False,
        "show_beam_demand": False,
        "show_demand_minus_service": True,
    }
    fig_preacc, info_preacc = sgui.postprocess_recipes.render_recipe(
        result_file,
        "beam_overview_over_time",
        source_preference=sgui.postprocess_recipes.SOURCE_PREACC,
        params=params,
    )
    fig_raw, info_raw = sgui.postprocess_recipes.render_recipe(
        result_file,
        "beam_overview_over_time",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params=params,
    )

    preacc_line = next(line for line in fig_preacc.axes[0].lines if line.get_label() == "Demand minus service")
    raw_line = next(line for line in fig_raw.axes[0].lines if line.get_label() == "Demand minus service")
    np.testing.assert_allclose(preacc_line.get_xdata(), raw_line.get_xdata())
    np.testing.assert_allclose(preacc_line.get_ydata(), raw_line.get_ydata())
    np.testing.assert_allclose(raw_line.get_ydata(), np.asarray([-1.0, -1.0, 0.0], dtype=np.float64))
    assert info_preacc["zero_reference_line"] is True
    assert info_raw["zero_reference_line"] is True
    assert len(fig_preacc.axes[0].lines) == 2
    assert len(fig_raw.axes[0].lines) == 2
    plt.close(fig_preacc)
    plt.close(fig_raw)


def test_postprocess_visible_beam_ccdf_matches_between_raw_and_preacc(tmp_path: Path) -> None:
    result_file = tmp_path / "beam_visible.h5"
    _write_minimal_result_file(result_file)

    fig_preacc, _info_preacc = sgui.postprocess_recipes.render_recipe(
        result_file,
        "beam_count_visible_ccdf",
        source_preference=sgui.postprocess_recipes.SOURCE_PREACC,
    )
    fig_raw, _info_raw = sgui.postprocess_recipes.render_recipe(
        result_file,
        "beam_count_visible_ccdf",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
    )

    preacc_line = fig_preacc.axes[0].lines[0]
    raw_line = fig_raw.axes[0].lines[0]
    np.testing.assert_allclose(preacc_line.get_xdata(), raw_line.get_xdata())
    np.testing.assert_allclose(preacc_line.get_ydata(), raw_line.get_ydata())
    plt.close(fig_preacc)
    plt.close(fig_raw)


def test_run_monitor_prefers_structured_progress_events(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()
    window._on_run_progress_event(
        {
            "kind": "iteration_plan",
            "phase": "chunks",
            "iteration_index": 0,
            "iteration_total": 2,
            "batch_total": 4,
        }
    )
    window._on_run_progress_event(
        {
            "kind": "chunk",
            "phase": "chunk_detail",
            "iteration_index": 0,
            "iteration_total": 2,
            "batch_index": 1,
            "batch_total": 4,
            "chunk_index": 2,
            "chunk_total": 5,
            "description": "Chunk 3/5 cells 200:300",
        }
    )
    qapp.processEvents()
    assert "computing direct-epfd chunks" in window.run_monitor.status_text().lower()
    assert "iteration 1/2" in window.run_monitor.overall_label.text().lower()
    assert "batch 2/4" in window.run_monitor.detail_label.text().lower()
    assert "chunk 3/5" in window.run_monitor.chunk_label.text().lower()
    assert window.run_monitor.chunk_label.isHidden() is False
    assert window.run_monitor.phase_chip.text() == "Compute"
    assert window.run_monitor._phase_step_labels["compute"].property("phaseState") == "active"
    assert window.statusBar().currentMessage() == "Running"
    window._dirty = False; window.close()


def test_run_monitor_surfaces_scheduler_load_stats_and_backoff_warning(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()
    monkeypatch.setattr(window.run_monitor, "_sample_live_resource_usage", lambda: {})
    window.run_monitor._timing_timer.stop()
    window._on_run_progress_event(
        {
            "kind": "iteration_plan",
            "phase": "chunks",
            "iteration_index": 0,
            "iteration_total": 3,
            "batch_total": 4,
            "boresight_active": True,
            "n_earthgrid_cells": 3360,
            "n_skycells_s1586": 200,
            "visible_satellite_est": 48,
            "bulk_timesteps": 6,
            "cell_chunk": 480,
            "sky_slab": 32,
            "spectral_slab": 3,
            "spectral_backoff_active": True,
            "host_hard_budget_bytes": 16 * 1024**3,
            "host_effective_budget_bytes": 16 * 1024**3,
            "gpu_hard_budget_bytes": 80 * 1024**3,
            "gpu_effective_budget_bytes": 80 * 1024**3,
            "predicted_host_peak_bytes": 9 * 1024**3,
            "predicted_gpu_peak_bytes": 58 * 1024**3,
            "predicted_gpu_activity_resident_bytes": 3 * 1024**3,
            "predicted_gpu_activity_scratch_bytes": 6 * 1024**3,
            "predicted_gpu_activity_peak_bytes": 10 * 1024**3,
            "predicted_gpu_spectrum_context_bytes": 512 * 1024**2,
            "scheduler_target_fraction": 0.90,
            "scheduler_active_target_fraction": 0.90,
            "planner_source": "warmup_calibrated",
            "limiting_resource": "gpu-propagation",
            "limiting_dimension": "spectral_slab",
            "planned_total_seconds": 7200.0,
            "planned_remaining_seconds": 7200.0,
            "live_host_total_bytes": 32 * 1024**3,
            "live_host_available_bytes": 20 * 1024**3,
            "live_gpu_total_bytes": 80 * 1024**3,
            "live_gpu_free_bytes": 30 * 1024**3,
            "live_gpu_adapter_total_bytes": 80 * 1024**3,
            "live_gpu_adapter_used_bytes": 50 * 1024**3,
            "live_gpu_adapter_free_bytes": 30 * 1024**3,
            "gpu_budget_reason": "hybrid explicit ceiling within live VRAM headroom",
        }
    )
    qapp.processEvents()
    load_text = window.run_monitor.load_stats_label.text().lower()
    assert "boresight=on" in load_text
    assert "cells=3360" in load_text
    assert "sky=200" in load_text
    assert "chunk=480" in load_text
    assert "spectral_slab=3" in load_text
    assert "target=90%" in load_text
    assert "planner=warmup_calibrated" in load_text
    assert "dim=spectral_slab" in load_text
    assert "gpu_budget=hybrid explicit ceiling within live vram headroom" in load_text
    assert "spectral-backoff" in load_text
    assert ("spectrum=512 mb" in load_text) or ("spectrum=512.0 mb" in load_text)
    assert (
        "activity peak=10.0 gb res=3.0 gb scratch=6.0 gb" in load_text
        or "activity peak=10.0 gb res=3.00 gb scratch=6.00 gb" in load_text
    )
    assert "device gpu 50.0 gb/80.0 gb" in load_text
    assert "cuda free 30.0 gb/80.0 gb" in load_text
    assert window.run_monitor.eta_meta_label.text() != "ETA --"
    assert "scepter rss:" in window.run_monitor._live_resource_cards["ram"]["value_label"].text().lower()
    assert "system:" in window.run_monitor._live_resource_cards["ram"]["value_label"].text().lower()
    assert "12.0 gb used" in window.run_monitor._live_resource_cards["ram"]["value_label"].text().lower()
    assert "50.0 gb used" in window.run_monitor._live_resource_cards["vram"]["value_label"].text().lower()
    assert window.run_monitor._live_resource_cards["ram"]["frame"].property("usageState") == "ok"
    assert window.run_monitor._live_resource_cards["vram"]["frame"].property("usageState") == "danger"

    window._on_run_progress_event(
        {
            "kind": "warning",
            "phase": "compute",
            "iteration_index": 0,
            "iteration_total": 3,
            "batch_index": 1,
            "batch_total": 4,
            "description": "GPU memory pressure during beam_finalize; configured 80.0 GB -> effective 42.0 GB for this run; retrying batch 2/4 at 75% target.",
            "boresight_active": True,
            "n_earthgrid_cells": 3360,
            "n_skycells_s1586": 200,
            "visible_satellite_est": 48,
            "bulk_timesteps": 6,
            "cell_chunk": 320,
            "sky_slab": 16,
            "host_hard_budget_bytes": 16 * 1024**3,
            "host_effective_budget_bytes": 16 * 1024**3,
            "gpu_hard_budget_bytes": 80 * 1024**3,
            "gpu_effective_budget_bytes": 42 * 1024**3,
            "gpu_effective_budget_lowered": True,
            "gpu_effective_budget_previous_bytes": 80 * 1024**3,
            "gpu_budget_lowered_stage": "beam_finalize",
            "predicted_host_peak_bytes": 8 * 1024**3,
            "predicted_gpu_peak_bytes": 44 * 1024**3,
            "scheduler_target_fraction": 0.90,
            "scheduler_active_target_fraction": 0.75,
            "scheduler_retry_count": 1,
            "planner_source": "analytic_fallback",
            "limiting_resource": "beam_finalize",
            "limiting_dimension": "spectral_slab",
            "observed_stage_name": "beam_finalize",
            "observed_stage_gpu_peak_bytes": 70 * 1024**3,
            "observed_stage_gpu_free_low_bytes": 10 * 1024**3,
            "observed_process_rss_bytes": 14 * 1024**3,
        }
    )
    qapp.processEvents()
    assert window.run_monitor.phase_chip.text() == "Warning"
    assert "retrying batch 2/4 at 75% target" in window.run_monitor.summary_label.text().lower()
    warning_load_text = window.run_monitor.load_stats_label.text().lower()
    assert "active=75%" in warning_load_text
    assert "chunk=320" in warning_load_text
    assert "gpu 44.0 gb/42.0 gb eff (cfg 80.0 gb)" in warning_load_text
    assert "dim=spectral_slab" in warning_load_text
    assert "retries=1" in warning_load_text
    assert "lowered@beam_finalize" in warning_load_text
    assert "observed beam_finalize peak=70.0 gb free_low=10.0 gb rss=14.0 gb" in warning_load_text
    assert "gpu memory pressure during beam_finalize" in warning_load_text
    assert window.run_monitor._live_resource_cards["vram"]["frame"].property("usageState") == "danger"
    window._dirty = False; window.close()


def test_run_monitor_live_resource_sampler_updates_cards_and_handles_missing_utilization(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()
    window.run_monitor._latest_scheduler_payload = {
        "host_hard_budget_bytes": 8 * 1024**3,
        "host_effective_budget_bytes": 8 * 1024**3,
        "gpu_hard_budget_bytes": 8 * 1024**3,
        "gpu_effective_budget_bytes": 8 * 1024**3,
        "scheduler_active_target_fraction": 0.75,
    }
    monkeypatch.setattr(
        window.run_monitor,
        "_sample_live_resource_usage",
        lambda: {
            "host_total_bytes": 32 * 1024**3,
            "host_available_bytes": 24 * 1024**3,
            "process_rss_bytes": 3 * 1024**3,
            "gpu_adapter_total_bytes": 16 * 1024**3,
            "gpu_adapter_used_bytes": 4 * 1024**3,
            "gpu_adapter_free_bytes": 12 * 1024**3,
            "gpu_cuda_total_bytes": 16 * 1024**3,
            "gpu_cuda_free_bytes": 12 * 1024**3,
            "process_gpu_used_bytes": 2 * 1024**3,
            "cpu_percent": 41.0,
        },
    )

    window.run_monitor._refresh_live_resource_cards()
    qapp.processEvents()

    assert "scepter rss: 3.00 gb" in window.run_monitor._live_resource_cards["ram"]["value_label"].text().lower()
    assert "system: 8.00 gb used" in window.run_monitor._live_resource_cards["ram"]["value_label"].text().lower()
    assert "scepter: 2.00 gb" in window.run_monitor._live_resource_cards["vram"]["value_label"].text().lower()
    assert "system: 4.00 gb used" in window.run_monitor._live_resource_cards["vram"]["value_label"].text().lower()
    assert "41%" in window.run_monitor._live_resource_cards["cpu"]["value_label"].text()
    assert window.run_monitor._live_resource_cards["gpu"]["value_label"].text() == "--"
    assert window.run_monitor._live_resource_cards["ram"]["frame"].property("usageState") == "ok"
    assert window.run_monitor._live_resource_cards["cpu"]["frame"].property("usageState") == "ok"
    assert window.run_monitor._live_resource_cards["gpu"]["frame"].property("usageState") == "na"
    window._dirty = False; window.close()


def test_run_monitor_vram_card_uses_windows_delta_fallback_when_process_query_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()
    monkeypatch.setattr(sgui.os, "name", "nt")
    window.run_monitor._gpu_adapter_used_baseline_bytes = 3 * 1024**3
    monkeypatch.setattr(window.run_monitor, "_sample_process_gpu_memory_bytes", lambda: None)
    monkeypatch.setattr(
        window.run_monitor,
        "_sample_gpu_device_metrics",
        lambda: {
            "gpu_adapter_total_bytes": 16 * 1024**3,
            "gpu_adapter_used_bytes": 7 * 1024**3,
            "gpu_adapter_free_bytes": 9 * 1024**3,
            "gpu_percent": 52.0,
        },
    )
    monkeypatch.setattr(
        sgui.scenario,
        "_runtime_host_memory_snapshot",
        lambda: {"provider": "test", "total_bytes": 32 * 1024**3, "available_bytes": 20 * 1024**3},
    )

    window.run_monitor._refresh_live_resource_cards()
    qapp.processEvents()

    vram_text = window.run_monitor._live_resource_cards["vram"]["value_label"].text().lower()
    assert "scepter: ~4.00 gb" in vram_text
    assert "system: 7.00 gb used" in vram_text
    window._dirty = False; window.close()


def test_run_monitor_terminal_states_clear_live_resource_cards(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    sample = {
        "host_total_bytes": 32 * 1024**3,
        "host_available_bytes": 20 * 1024**3,
        "process_rss_bytes": 4 * 1024**3,
        "gpu_adapter_total_bytes": 16 * 1024**3,
        "gpu_adapter_used_bytes": 6 * 1024**3,
        "gpu_adapter_free_bytes": 10 * 1024**3,
        "process_gpu_used_bytes": 3 * 1024**3,
        "cpu_percent": 41.0,
        "gpu_percent": 55.0,
    }

    def _assert_cards_reset() -> None:
        for key in ("ram", "vram", "cpu", "gpu"):
            assert window.run_monitor._live_resource_cards[key]["value_label"].text() == "--"
            assert window.run_monitor._live_resource_cards[key]["frame"].property("usageState") == "idle"

    window.run_monitor.reset_for_run()
    window.run_monitor._apply_live_resource_sample(sample)
    window.run_monitor.update_from_event(
        {
            "kind": "run_complete",
            "phase": "completed",
            "storage_filename": "done.h5",
            "iteration_total": 1,
            "writer_stats_summary": {},
        }
    )
    qapp.processEvents()
    _assert_cards_reset()

    window.run_monitor.reset_for_run()
    window.run_monitor._apply_live_resource_sample(sample)
    window.run_monitor.update_from_event(
        {
            "kind": "run_stopped",
            "phase": "stopped",
            "storage_filename": "partial.h5",
            "stop_mode": "graceful",
            "writer_stats_summary": {},
        }
    )
    qapp.processEvents()
    _assert_cards_reset()

    window.run_monitor.reset_for_run()
    window.run_monitor._apply_live_resource_sample(sample)
    window.run_monitor.set_failure("synthetic failure")
    qapp.processEvents()
    _assert_cards_reset()
    window._dirty = False; window.close()


def test_run_monitor_prepare_phase_and_raw_counts_keep_overall_bar_stable(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()
    window._on_run_progress_event(
        {
            "kind": "prepare",
            "phase": "prepare",
            "prepare_index": 1,
            "prepare_total": 2,
            "description": "Building active grid",
        }
    )
    assert window.run_monitor.overall_label.text() == "Iteration progress"
    assert window.run_monitor.detail_bar.maximum() == 2
    assert window.run_monitor.detail_bar.value() == 1
    assert window.run_monitor.eta_meta_label.text() == "ETA --"

    window._run_worker = object()
    window._on_run_progress_event(
        {
            "kind": "iteration_plan",
            "phase": "chunks",
            "iteration_index": 0,
            "iteration_total": 2,
            "batch_total": 4,
        }
    )
    overall_max_before = window.run_monitor.overall_bar.maximum()
    overall_value_before = window.run_monitor.overall_bar.value()
    detail_max_before = window.run_monitor.detail_bar.maximum()
    detail_value_before = window.run_monitor.detail_bar.value()
    detail_label_before = window.run_monitor.detail_label.text()
    window._on_run_progress_text(
        "Write tail:  99%|###################################################################4| 124/125 [00:30<00:00,  4.19it/s]"
    )
    qapp.processEvents()
    assert window.run_monitor.overall_bar.maximum() == overall_max_before
    assert window.run_monitor.overall_bar.value() == overall_value_before
    assert window.run_monitor.detail_bar.maximum() == detail_max_before
    assert window.run_monitor.detail_bar.value() == detail_value_before
    assert window.run_monitor.detail_label.text() == detail_label_before
    window._run_worker = None
    window._dirty = False; window.close()


def test_run_monitor_can_use_raw_progress_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()
    window._run_worker = object()  # sentinel for fallback path
    window._on_run_progress_value(4, 10)
    qapp.processEvents()
    assert window.run_monitor.detail_bar.maximum() == 10
    assert window.run_monitor.detail_bar.value() == 4
    assert "4/10" in window.run_monitor.detail_label.text()
    window._run_worker = None
    window._dirty = False; window.close()


def test_blank_project_keeps_satellite_pattern_wavelength_empty_until_defaults(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    assert window.pattern_wavelength_spin.value_or_none() is None
    assert window.pattern_wavelength_spin.text().strip() == ""
    assert window.derive_pattern_wavelength_checkbox.isChecked() is False

    window._set_satellite_antenna_defaults()
    assert window.pattern_wavelength_spin.value_or_none() == pytest.approx(15.0)
    assert window.derive_pattern_wavelength_checkbox.isChecked() is False

    window.new_configuration()
    qapp.processEvents()
    window._set_ras_antenna_defaults()
    assert window.pattern_wavelength_spin.value_or_none() is None
    assert window.pattern_wavelength_spin.text().strip() == ""
    assert window.derive_pattern_wavelength_checkbox.isChecked() is False

    window.new_configuration()
    qapp.processEvents()
    assert window.pattern_wavelength_spin.value_or_none() is None
    assert window.pattern_wavelength_spin.text().strip() == ""
    assert window.derive_pattern_wavelength_checkbox.isChecked() is False
    window._dirty = False; window.close()


def test_run_monitor_keeps_batch_and_eta_stable_through_sparse_phase_updates(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()

    window._on_run_progress_event(
        {
            "kind": "iteration_plan",
            "phase": "chunks",
            "iteration_index": 0,
            "iteration_total": 1,
            "batch_total": 10,
        }
    )
    window._on_run_progress_event(
        {
            "kind": "batch_start",
            "phase": "compute",
            "iteration_index": 0,
            "iteration_total": 1,
            "batch_index": 2,
            "batch_total": 10,
            "chunk_index": 1,
            "chunk_total": 4,
        }
    )
    qapp.processEvents()

    overall_label_before = window.run_monitor.overall_label.text()
    overall_max_before = window.run_monitor.overall_bar.maximum()
    overall_value_before = window.run_monitor.overall_bar.value()
    batch_label_before = window.run_monitor.batch_meta_label.text()
    chunk_label_before = window.run_monitor.chunk_label.text()
    detail_label_before = window.run_monitor.detail_label.text()
    eta_label_before = window.run_monitor.eta_meta_label.text()

    assert batch_label_before == "Batch 3/10"
    assert chunk_label_before == "Chunk 2/4"
    assert detail_label_before == "Batch 3/10"
    assert eta_label_before != "ETA --"

    for payload in (
        {
            "kind": "phase",
            "phase": "power_accumulation",
            "iteration_index": 0,
            "description": "Accumulating power outputs...",
        },
        {
            "kind": "phase",
            "phase": "beam_finalize",
            "iteration_index": 0,
            "description": "Finalising beam selections...",
        },
        {
            "kind": "phase",
            "phase": "write_enqueue",
            "iteration_index": 0,
            "description": "Queueing batch data for the writer...",
        },
        {
            "kind": "phase",
            "phase": "final_flush",
            "iteration_index": 0,
            "description": "Final flush",
        },
    ):
        window._on_run_progress_event(payload)
        qapp.processEvents()
        assert window.run_monitor.overall_label.text() == overall_label_before
        assert window.run_monitor.overall_bar.maximum() == overall_max_before
        assert window.run_monitor.overall_bar.value() == overall_value_before
        assert window.run_monitor.batch_meta_label.text() == batch_label_before
        assert window.run_monitor.chunk_label.text() == chunk_label_before
        assert window.run_monitor.detail_label.text() == detail_label_before
        assert window.run_monitor.elapsed_meta_label.text() != "Elapsed --"
        assert window.run_monitor.eta_meta_label.text() != "ETA --"

    window._dirty = False; window.close()


def test_run_monitor_raw_write_tail_fallback_keeps_known_chunk_state(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()
    window._on_run_progress_event(
        {
            "kind": "batch_start",
            "phase": "compute",
            "iteration_index": 0,
            "iteration_total": 1,
            "batch_index": 132,
            "batch_total": 246,
            "chunk_index": 3,
            "chunk_total": 4,
        }
    )
    qapp.processEvents()
    assert window.run_monitor.batch_meta_label.text() == "Batch 133/246"
    assert window.run_monitor.chunk_label.text() == "Chunk 4/4"

    window._run_worker = object()
    window._on_run_progress_text(
        "Write tail: 54%|#########################               | 133/246 [01:09<00:58,  1.94it/s]"
    )
    qapp.processEvents()

    assert window.run_monitor.batch_meta_label.text() == "Batch 133/246"
    assert window.run_monitor.chunk_label.text() == "Chunk 4/4"
    assert window.run_monitor.detail_label.text() == "Batch 133/246"
    assert window.run_monitor.detail_bar.maximum() == 246
    assert window.run_monitor.detail_bar.value() == 133
    window._run_worker = None
    window._dirty = False; window.close()


def test_run_monitor_keeps_last_known_batch_and_chunk_until_next_batch_update(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.run_monitor.reset_for_run()

    window._on_run_progress_event(
        {
            "kind": "iteration_plan",
            "phase": "chunks",
            "iteration_index": 0,
            "iteration_total": 1,
            "batch_total": 123,
        }
    )
    window._on_run_progress_event(
        {
            "kind": "batch_start",
            "phase": "compute",
            "iteration_index": 0,
            "iteration_total": 1,
            "batch_index": 76,
            "batch_total": 123,
            "chunk_index": 1,
            "chunk_total": 4,
        }
    )
    qapp.processEvents()

    assert window.run_monitor.batch_meta_label.text() == "Batch 77/123"
    assert window.run_monitor.chunk_label.text() == "Chunk 2/4"
    assert window.run_monitor.detail_label.text() == "Batch 77/123"

    for payload in (
        {
            "kind": "chunk",
            "phase": "chunk_detail",
            "iteration_index": 0,
            "iteration_total": 1,
            "chunk_index": 2,
            "chunk_total": 4,
        },
        {
            "kind": "phase",
            "phase": "power_accumulation",
            "iteration_index": 0,
            "description": "Accumulating power outputs...",
        },
        {
            "kind": "phase",
            "phase": "beam_finalize",
            "iteration_index": 0,
            "description": "Finalising beam selections...",
        },
        {
            "kind": "phase",
            "phase": "write_enqueue",
            "iteration_index": 0,
            "description": "Queueing batch data for the writer...",
        },
        ):
        window._on_run_progress_event(payload)
        qapp.processEvents()
        window._flush_pending_run_progress_event()
        qapp.processEvents()
        assert window.run_monitor.batch_meta_label.text() == "Batch 77/123"
        assert window.run_monitor.batch_meta_label.text() != "Batch --"
        assert window.run_monitor.chunk_label.text() == "Chunk 3/4"
        assert window.run_monitor.chunk_label.text() != "Chunk --"
        assert window.run_monitor.detail_label.text() == "Batch 77/123"

    window._run_worker = object()
    window._on_run_progress_text(
        "Write tail: 63%|###############################         | 77/123 [00:58<00:34,  1.35it/s]"
    )
    qapp.processEvents()
    assert window.run_monitor.batch_meta_label.text() == "Batch 77/123"
    assert window.run_monitor.chunk_label.text() == "Chunk 3/4"
    assert window.run_monitor.detail_label.text() == "Batch 77/123"

    window._on_run_progress_event(
        {
            "kind": "batch_start",
            "phase": "compute",
            "iteration_index": 0,
            "iteration_total": 1,
            "batch_index": 77,
            "batch_total": 123,
            "chunk_index": 0,
            "chunk_total": 4,
        }
    )
    qapp.processEvents()
    window._flush_pending_run_progress_event()
    qapp.processEvents()
    assert window.run_monitor.batch_meta_label.text() == "Batch 78/123"
    assert window.run_monitor.chunk_label.text() == "Chunk 1/4"
    assert window.run_monitor.detail_label.text() == "Batch 78/123"
    window._run_worker = None
    window._dirty = False; window.close()


def test_postprocess_beam_overview_legend_sits_above_axes(tmp_path: Path) -> None:
    result_file = tmp_path / "beam_overview_legend.h5"
    _write_minimal_result_file(result_file)

    fig, _info = sgui.postprocess_recipes.render_recipe(
        result_file,
        "beam_overview_over_time",
        source_preference=sgui.postprocess_recipes.SOURCE_RAW,
        params={
            "show_total_beams": True,
            "show_visible_beams": True,
            "show_beam_demand": True,
            "max_points": 4000,
            "smoothing_window_s": 0.0,
        },
    )

    ax = fig.axes[0]
    legend = fig.legends[0] if fig.legends else None
    assert legend is not None
    assert fig._suptitle is not None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    legend_bbox = legend.get_window_extent(renderer)
    title_bbox = fig._suptitle.get_window_extent(renderer)
    assert legend_bbox.y1 <= title_bbox.y0 + 1.0
    assert legend_bbox.y0 >= axes_bbox.y1 - 1.0
    plt.close(fig)


def test_postprocess_action_groups_stack_vertically_and_labels_fit(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    widget = window.postprocess_widget
    window.resize(1400, 900)
    window._set_workspace(sgui._WORKSPACE_POSTPROCESS)
    window.show()
    qapp.processEvents()

    groups = {
        box.title(): box
        for box in widget.findChildren(QtWidgets.QGroupBox)
        if box.title() in {"Static", "Interactive"}
    }
    static_pos = groups["Static"].mapTo(widget, QtCore.QPoint(0, 0))
    interactive_pos = groups["Interactive"].mapTo(widget, QtCore.QPoint(0, 0))
    assert interactive_pos.y() > static_pos.y()
    assert widget.detach_plot_button.minimumWidth() >= 160
    assert widget.browser_plot_button.minimumWidth() >= 160
    assert widget.detach_plot_button.text() == "Open figure"
    assert widget.browser_plot_button.text() == "Open in browser"
    assert groups["Static"].objectName() == "postprocessActionGroup"
    assert groups["Interactive"].objectName() == "postprocessActionGroup"
    assert widget.open_result_button.property("shellPrimaryAction") is True
    assert widget.refresh_result_button.property("shellPrimaryAction") is True
    assert widget.render_recipe_button.property("shellPrimaryAction") is True
    assert widget.detach_plot_button.property("plotStudioAction") is True
    assert widget.browser_plot_button.property("plotStudioAction") is True
    assert widget.detach_plot_button.property("shellPrimaryAction") is True
    assert widget.browser_plot_button.property("shellPrimaryAction") is True
    assert widget.open_result_button.property("plotStudioAction") is None
    assert widget.refresh_result_button.property("plotStudioAction") is None
    style = window.styleSheet()
    assert 'QPushButton[shellPrimaryAction="true"]' in style
    shell_action_rule = style.split('QPushButton[shellPrimaryAction="true"],', 1)[1].split("}", 1)[0]
    assert "stop:0 #0ea5e9" in shell_action_rule
    assert "stop:1 #0284c7" in shell_action_rule
    assert "color: white;" in shell_action_rule
    assert 'QPushButton[shellPrimaryAction="true"]:hover' in style
    assert 'QPushButton[shellPrimaryAction="true"]:disabled' in style
    plot_action_rule = style.split('QPushButton[plotStudioAction="true"] {', 1)[1].split("}", 1)[0]
    assert "min-height: 18px;" in plot_action_rule
    assert "border-radius: 14px;" in plot_action_rule
    assert "background:" not in plot_action_rule
    assert "color:" not in plot_action_rule
    assert 'QPushButton[plotStudioAction="true"]:hover' not in style
    assert 'QPushButton[plotStudioAction="true"]:disabled' not in style
    assert widget.result_summary_scroll.viewport().objectName() == "postprocessResultSummaryViewport"
    assert widget.recipe_controls_scroll.viewport().objectName() == "postprocessRecipeControlsViewport"
    summary_content = widget.findChild(QtWidgets.QWidget, "postprocessResultSummaryContent")
    controls_content = widget.findChild(QtWidgets.QWidget, "postprocessRecipeControlsContent")
    assert summary_content is not None
    assert controls_content is not None
    assert summary_content.styleSheet() == ""
    assert controls_content.styleSheet() == ""
    widget._set_recipe_availability_text("Blocked: test", tone="error")
    window.appearance_mode_combo.setCurrentIndex(window.appearance_mode_combo.findData("light"))
    qapp.processEvents()
    assert widget.recipe_availability_label.property("postprocessTone") == "error"
    assert widget.recipe_availability_label.styleSheet() == ""
    assert any(
        child.objectName() == "pageSafeArea" and child.parent() is widget
        for child in widget.findChildren(QtWidgets.QWidget)
    )
    window._dirty = False; window.close()


@pytest.mark.parametrize("appearance_mode", ("light", "dark"))
def test_postprocess_action_buttons_match_top_toolbar_visual_family_when_disabled(
    appearance_mode: str,
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    widget = window.postprocess_widget
    window.resize(1400, 900)
    window._set_workspace(sgui._WORKSPACE_POSTPROCESS)
    window.show()
    qapp.processEvents()

    appearance_index = window.appearance_mode_combo.findData(appearance_mode)
    window.appearance_mode_combo.setCurrentIndex(appearance_index)
    qapp.processEvents()

    widget.refresh_result_button.setEnabled(True)
    widget.render_recipe_button.setEnabled(False)
    qapp.processEvents()

    groups = {
        box.title(): box
        for box in widget.findChildren(QtWidgets.QGroupBox)
        if box.title() in {"Static", "Interactive"}
    }
    static_group = groups["Static"]

    refresh_rgb = _mean_rgb_for_points(
        _grab_widget_image(widget.refresh_result_button),
        _button_background_points(widget.refresh_result_button),
    )
    render_rgb = _mean_rgb_for_points(
        _grab_widget_image(widget.render_recipe_button),
        _button_background_points(widget.render_recipe_button),
    )
    panel_rgb = _mean_rgb_for_points(
        _grab_widget_image(static_group),
        _panel_margin_points(static_group),
    )

    assert _rgb_distance(refresh_rgb, render_rgb) < 45.0
    assert _rgb_distance(render_rgb, panel_rgb) > 90.0

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_theme_stylesheet_exposes_high_contrast_controls_and_scrollbars(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    qapp.processEvents()

    style = window.styleSheet()
    assert "QCheckBox::indicator" in style
    assert "QRadioButton::indicator" in style
    assert "QScrollBar:vertical" in style
    assert "QScrollBar::handle:vertical" in style
    assert "QStackedWidget#postprocessPlotCanvasContainer" in style
    assert "QWidget#postprocessPlotCanvas" in style
    assert "QLabel#matplotlibResizeProxy" in style
    modern_toggle_rule = style.split('QToolButton[modernToggle="true"] {', 1)[1].split("}", 1)[0]
    assert "background:" in modern_toggle_rule
    assert "color:" in modern_toggle_rule
    help_button_rule = style.split('QToolButton[helpButton="true"] {', 1)[1].split("}", 1)[0]
    assert "background:" in help_button_rule
    assert "color:" in help_button_rule
    assert window.postprocess_widget.plot_canvas_container.objectName() == "postprocessPlotCanvasContainer"
    assert window.postprocess_widget.plot_canvas.objectName() == "postprocessPlotCanvas"

    window.appearance_mode_combo.setCurrentIndex(window.appearance_mode_combo.findData("light"))
    qapp.processEvents()
    light_style = window.styleSheet()
    assert "QCheckBox::indicator" in light_style
    assert "QScrollBar::handle:horizontal" in light_style
    assert 'QPushButton[shellPrimaryAction="true"]:disabled' in light_style

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_review_run_scroll_area_uses_styled_background_hooks(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    qapp.processEvents()

    assert window.runtime_tab.objectName() == "runtimeScrollArea"
    assert window.runtime_tab.viewport().objectName() == "runtimeScrollViewport"
    assert window.runtime_tab.widget().objectName() == "runtimeScrollPage"
    window._dirty = False; window.close()


def test_simulation_pages_use_scroll_wrappers_and_runtime_maps_to_inner_page(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    qapp.processEvents()

    for page in (
        window.constellation_tab,
        window.ras_tab,
        window.antennas_tab,
        window.service_tab,
        window.grid_tab,
        window.hexgrid_tab,
        window.runtime_page,
    ):
        container = window._simulation_tab_container_for_page(page)
        assert isinstance(container, QtWidgets.QScrollArea)

    window._select_simulation_page(window.runtime_page)
    qapp.processEvents()
    assert window._current_simulation_page() is window.runtime_page
    assert window._simulation_page_for_target(window.runtime_tab) is window.runtime_page
    window._dirty = False; window.close()


def test_workspace_nav_can_collapse_to_icon_strip(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._animations_enabled = False
    window.show()
    qapp.processEvents()

    window._set_workspace_nav_expanded(False)
    qapp.processEvents()

    assert window.workspace_nav.width() == sgui._WORKSPACE_NAV_COLLAPSED_WIDTH
    assert window.nav_home_button.text() == ""
    assert bool(window.nav_home_button.property("navCollapsed")) is True
    assert window._workspace_nav_brand_mark.isHidden() is False
    assert window._workspace_nav_brand_text_container.isHidden() is True
    assert window._workspace_nav_brand_label.isHidden() is True
    assert window._workspace_nav_tag_label.isHidden() is True
    assert window.workspace_nav_pin_button.geometry().intersects(window.workspace_nav_toggle_button.geometry()) is False
    assert window.workspace_nav_toggle_button.geometry().top() > window.workspace_nav_pin_button.geometry().top()
    assert window.workspace_nav_pin_button.text() == ""
    assert window.side_pane_pin_button.text() == ""
    window._dirty = False; window.close()


def test_workspace_nav_uses_branded_lockup_and_larger_pane_controls(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    qapp.processEvents()
    window.workspace_nav_pin_button.setChecked(True)
    _wait_until(
        lambda: window.workspace_nav.width() == sgui._WORKSPACE_NAV_EXPANDED_WIDTH,
        timeout_ms=1000,
        step_ms=20,
    )

    brand_mark = window._workspace_nav_brand_mark.pixmap()
    assert brand_mark is not None
    assert brand_mark.isNull() is False
    assert window.workspace_nav.width() == sgui._WORKSPACE_NAV_EXPANDED_WIDTH
    assert window._workspace_nav_brand_text_container.isHidden() is False
    assert window._workspace_nav_brand_label.text() == "SCEPTer"
    assert window._workspace_nav_brand_label.isHidden() is False
    assert window._workspace_nav_tag_label.isHidden() is False
    assert window.nav_postprocess_button.property("navIconKind") == "postprocess"
    assert window.nav_home_button.icon().isNull() is False
    assert window.nav_simulation_button.icon().isNull() is False
    assert window.nav_postprocess_button.icon().isNull() is False
    assert window.workspace_nav_pin_button.text() == ""
    assert window.side_pane_toggle_button.text() == "Assistant"
    assert window.side_pane_toggle_button.toolButtonStyle() == QtCore.Qt.ToolButtonTextBesideIcon
    assert window.workspace_nav_pin_button.width() >= sgui._CONTROL_BUTTON_MIN_SIZE
    assert window.workspace_nav_toggle_button.width() >= sgui._CONTROL_BUTTON_MIN_SIZE
    assert window.side_pane_pin_button.width() >= sgui._CONTROL_BUTTON_MIN_SIZE
    assert window.side_pane_toggle_button.width() >= sgui._CONTROL_BUTTON_MIN_SIZE
    assert window.workspace_nav_pin_button.icon().isNull() is False
    assert window.workspace_nav_toggle_button.icon().isNull() is False
    assert window.side_pane_pin_button.icon().isNull() is False
    assert window.side_pane_toggle_button.icon().isNull() is False
    assert window.workspace_nav_pin_button.iconSize().width() >= sgui._CONTROL_ICON_MIN_SIZE
    assert window.workspace_nav_toggle_button.iconSize().width() >= sgui._CONTROL_ICON_MIN_SIZE
    assert window.side_pane_pin_button.iconSize().width() >= sgui._CONTROL_ICON_MIN_SIZE
    assert window.side_pane_toggle_button.iconSize().width() >= sgui._CONTROL_ICON_MIN_SIZE
    assert window.side_pane_toggle_button.width() > window.side_pane_toggle_button.height()
    window._dirty = False; window.close()


@pytest.mark.parametrize("scale", [1.25, 1.5, 2.0])
def test_scale_proxy_keeps_nav_controls_clear_and_assistant_labeled(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    scale: float,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    qapp.processEvents()

    monkeypatch.setattr(window, "_effective_ui_scale", lambda: scale)
    window._refresh_workspace_nav_layout()
    window._update_side_pane_toggle_button()
    qapp.processEvents()

    assert window.side_pane_toggle_button.text() == "Assistant"
    assert window.side_pane_toggle_button.width() > window.side_pane_toggle_button.height()
    assert window.workspace_nav_pin_button.iconSize().width() >= sgui._CONTROL_ICON_MIN_SIZE

    window._set_workspace_nav_expanded(False)
    qapp.processEvents()

    assert window.workspace_nav.width() == sgui._WORKSPACE_NAV_COLLAPSED_WIDTH
    assert window.workspace_nav_pin_button.geometry().intersects(window.workspace_nav_toggle_button.geometry()) is False
    window._dirty = False; window.close()


def test_appearance_mode_persists_across_window_restarts(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    light_idx = window.appearance_mode_combo.findData("light")
    window.appearance_mode_combo.setCurrentIndex(light_idx)
    qapp.processEvents()
    assert window._selected_appearance_mode() == "light"
    assert "#f5efe3" in window.styleSheet()
    assert "QMenuBar" in window.styleSheet()

    # Simulate "restart": reload session preferences from QSettings
    # without constructing a new window.
    window._restore_session_state()
    qapp.processEvents()
    # The QSettings value should have been read back → still light
    assert window._selected_appearance_mode() == "light"
    assert "#f5efe3" in window.styleSheet()
    assert "QMenuBar" in window.styleSheet()
    _clear_gui_settings()


def test_startup_defaults_and_reset_return_to_advanced_system(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    qapp.processEvents()

    assert window._selected_appearance_mode() == "system"
    assert window.complexity_mode_combo.currentData() == "Advanced"

    window.appearance_mode_combo.setCurrentIndex(window.appearance_mode_combo.findData("light"))
    window.complexity_mode_combo.setCurrentIndex(window.complexity_mode_combo.findData("Basic"))
    qapp.processEvents()
    assert window._selected_appearance_mode() == "light"
    assert window.complexity_mode_combo.currentData() == "Basic"

    window.reset_to_defaults()
    qapp.processEvents()

    assert window._selected_appearance_mode() == "system"
    assert window.complexity_mode_combo.currentData() == "Advanced"
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_system_theme_refresh_only_runs_when_system_mode_is_selected(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    qapp.processEvents()
    calls: list[str] = []

    def _fake_refresh() -> None:
        calls.append(window._selected_appearance_mode())
        window._system_theme_refresh_pending = False

    monkeypatch.setattr(window, "_refresh_system_theme", _fake_refresh)
    window._system_theme_refresh_pending = False

    window._schedule_system_theme_refresh()
    window._schedule_system_theme_refresh()
    qapp.processEvents()
    assert calls == ["system"]

    qapp.processEvents()
    calls.clear()
    window.appearance_mode_combo.setCurrentIndex(window.appearance_mode_combo.findData("light"))
    qapp.processEvents()
    window._system_theme_refresh_pending = False
    window._schedule_system_theme_refresh()
    window._schedule_system_theme_refresh()
    qapp.processEvents()
    assert calls == []
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_system_theme_refresh_is_noop_when_signature_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    qapp.processEvents()
    window._last_applied_system_theme_signature = window._current_system_theme_signature()

    apply_calls: list[str] = []
    real_apply = window._apply_visual_style

    def _tracked_apply() -> None:
        apply_calls.append("apply")
        real_apply()

    monkeypatch.setattr(window, "_apply_visual_style", _tracked_apply)

    window._refresh_system_theme()
    qapp.processEvents()

    assert apply_calls == []
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_child_palette_change_does_not_trigger_global_theme_or_layout_refresh(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    qapp.processEvents()

    theme_calls: list[str] = []
    layout_calls: list[str] = []
    monkeypatch.setattr(window, "_schedule_system_theme_refresh", lambda: theme_calls.append("theme"))
    monkeypatch.setattr(
        window,
        "_schedule_responsive_layout_refresh",
        lambda: layout_calls.append("layout"),
    )

    child_event = QtCore.QEvent(QtCore.QEvent.PaletteChange)
    assert window.eventFilter(window.nav_home_button, child_event) is False
    assert theme_calls == []
    assert layout_calls == []

    app_event = QtCore.QEvent(QtCore.QEvent.ApplicationPaletteChange)
    qapp_ref = QtWidgets.QApplication.instance()
    assert qapp_ref is not None
    assert window.eventFilter(qapp_ref, app_event) is False
    assert theme_calls == ["theme"]
    assert layout_calls == []
    qapp.processEvents()
    assert layout_calls == ["layout"]

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_global_appearance_control_remains_visible_in_postprocess(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._set_workspace(sgui._WORKSPACE_POSTPROCESS)
    window.show()
    qapp.processEvents()

    assert window.appearance_mode_combo.isVisible() is True
    assert window.appearance_mode_combo.parentWidget() is window._appearance_shell_container
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_workspace_nav_appearance_control_fits_expanded_and_collapsed(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    qapp.processEvents()

    window._refresh_workspace_nav_layout()
    qapp.processEvents()
    assert window._appearance_shell_label.isVisible() is True
    assert window._appearance_shell_label.text() == "Theme"
    assert window.appearance_mode_combo.width() > 100
    assert window.appearance_mode_combo.geometry().right() <= (
        window._appearance_shell_container.geometry().right() + 2
    )
    assert abs(
        window.workspace_nav_pin_button.geometry().center().y()
        - window.workspace_nav_toggle_button.geometry().center().y()
    ) <= 4
    assert window.workspace_nav_pin_button.geometry().right() <= (
        window.workspace_nav_toggle_button.geometry().left() + 12
    )

    window._set_workspace_nav_expanded(False)
    qapp.processEvents()
    assert window._appearance_shell_label.isVisible() is True
    assert window.appearance_mode_combo.isVisible() is True
    assert window.appearance_mode_combo.width() >= max(window._appearance_shell_container.width() - 12, 48)
    assert window.appearance_mode_combo.geometry().right() <= window._appearance_shell_container.rect().right()
    assert window.workspace_nav_pin_button.geometry().bottom() <= window.workspace_nav_toggle_button.geometry().top()

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_theme_change_does_not_reopen_hexgrid_preview_from_cache(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    monkeypatch.setattr(
        window,
        "_render_hexgrid_preview_from_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("theme changes should restyle windows in place, not reopen cached hexgrid previews")
        ),
    )

    light_idx = window.appearance_mode_combo.findData("light")
    window.appearance_mode_combo.setCurrentIndex(light_idx)
    qapp.processEvents()

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_repolish_shell_theme_surfaces_handles_appearance_changes_with_item_views(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    qapp.processEvents()

    list_view = QtWidgets.QListView(window._right_container)
    list_view.setObjectName("themeRepolishListView")
    list_view.setGeometry(0, 0, 120, 60)
    list_view.show()
    scroll_area = QtWidgets.QScrollArea(window._right_container)
    scroll_area.setObjectName("themeRepolishScrollArea")
    scroll_area.setGeometry(0, 70, 120, 60)
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(QtWidgets.QWidget())
    scroll_area.show()
    qapp.processEvents()

    assert list_view.parentWidget() is window._right_container
    assert isinstance(list_view.viewport(), QtWidgets.QWidget)
    assert isinstance(scroll_area.viewport(), QtWidgets.QWidget)

    for appearance_mode in ("light", "dark", "system"):
        appearance_index = window.appearance_mode_combo.findData(appearance_mode)
        with QtCore.QSignalBlocker(window.appearance_mode_combo):
            window.appearance_mode_combo.setCurrentIndex(appearance_index)
        window._on_appearance_mode_changed()
        qapp.processEvents()

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_assistant_panel_theme_switch_uses_theme_driven_chip_styles(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    window.show()
    qapp.processEvents()

    light_idx = window.appearance_mode_combo.findData("light")
    dark_idx = window.appearance_mode_combo.findData("dark")
    window.appearance_mode_combo.setCurrentIndex(light_idx)
    qapp.processEvents()
    window.appearance_mode_combo.setCurrentIndex(dark_idx)
    qapp.processEvents()

    assert window.guidance_chip.styleSheet() == ""
    assert window.snapshot_chip.styleSheet() == ""
    assert window._assistant_scroll.objectName() == "assistantScrollArea"
    assert window._assistant_scroll.viewport().objectName() == "assistantScrollViewport"
    assert window._assistant_scroll_content.objectName() == "assistantScrollContent"

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_show_reuse_scheme_opens_detached_window(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    state = window.current_state()
    state.active_system().spectrum.reuse_factor = 4
    state.active_system().spectrum.ras_anchor_reuse_slot = 2
    window._load_state_into_widgets(state)
    monkeypatch.setattr(
        window,
        "_preview_ready_hexgrid_result",
        lambda: (_ for _ in ()).throw(AssertionError("geographic preview should not be required")),
    )

    window._show_reuse_scheme_viewer()
    qapp.processEvents()

    assert window._reuse_scheme_window is not None
    assert window._reuse_scheme_window.windowTitle() == "Frequency Reuse Scheme"
    assert len(window._reuse_scheme_window._figure.axes) == 1
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_preview_handlers_require_explicit_spectrum_inputs(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(
        monkeypatch,
        state=sgui.ScepterProjectState(),
        current_hexgrid=False,
    )
    window._set_service_defaults()
    window._set_ras_station_widgets(sgui._default_ras_station_config())
    window._sync_spectrum_controls()
    qapp.processEvents()

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sgui.QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((str(title), str(message))),
    )

    window._show_reuse_scheme_viewer()
    window._show_spectrum_preview()
    qapp.processEvents()

    assert len(warnings) == 2
    assert warnings[0] == ("Reuse preview unavailable", "Choose a reuse scheme first.")
    assert warnings[1] == ("Edit Spectrum unavailable", "Enter Spectrum & Reuse settings first.")
    assert window._reuse_scheme_window is None
    assert window._spectrum_preview_window is None
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_ras_receiver_preview_moves_summary_text_outside_the_figure(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    assert state.ras_station is not None
    state.ras_station.receiver_band_start_mhz = 2690.0
    state.ras_station.receiver_band_stop_mhz = 2700.0
    state.ras_station.receiver_response_mode = "rectangular"
    window._load_state_into_widgets(state)
    window.service_bandwidth_edit.set_value(None)
    qapp.processEvents()

    window._show_ras_receiver_response_preview()
    qapp.processEvents()

    assert window._ras_receiver_preview_window is not None
    assert window._ras_receiver_preview_window._summary_label.isVisible() is True
    assert "ras receiver band" in window._ras_receiver_preview_window._summary_label.text().lower()
    figure_texts = [text.get_text() for text in window._ras_receiver_preview_window._figure.texts]
    assert not any("ras receiver band" in text.lower() for text in figure_texts)
    window._dirty = False; window.close()


@pytest.mark.parametrize(
    ("reuse_factor", "expected_groups_per_cell"),
    ((1, 14), (7, 2)),
)
def test_spectrum_preview_defaults_to_all_checked_channels_with_auto_cap(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    reuse_factor: int,
    expected_groups_per_cell: int,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().service.bandwidth_mhz = 5.0
    state.active_system().spectrum.service_band_start_mhz = 2620.0
    state.active_system().spectrum.service_band_stop_mhz = 2690.0
    state.active_system().spectrum.reuse_factor = reuse_factor
    state.active_system().spectrum.disabled_channel_indices = None
    assert state.ras_station is not None
    state.ras_station.receiver_band_start_mhz = 2690.0
    state.ras_station.receiver_band_stop_mhz = 2700.0
    spectrum_plan = sgui._normalize_spectrum_config(
        state.active_system().service,
        state.active_system().spectrum,
        state.ras_station,
        active_cell_count=max(1, reuse_factor),
        active_cell_reuse_slot_ids=np.arange(max(1, reuse_factor), dtype=np.int32) % int(reuse_factor),
    )
    dialog = sgui.SpectrumPreviewDialog(
        spectrum_plan=spectrum_plan,
        parent=window,
    )
    dialog.show()
    qapp.processEvents()

    checked_indices = dialog._selected_channel_indices()
    selection = sgui._resolve_preview_channel_selection(
        spectrum_plan,
        active_channel_indices=checked_indices,
    )

    assert spectrum_plan["channel_groups_per_cell"] == expected_groups_per_cell
    assert len(checked_indices) == 14
    assert len(selection["selected_channel_indices"]) == 14
    assert len(selection["selected_occupied_channel_indices"]) == 14
    assert len(selection["selected_excluded_channel_indices"]) == 0
    assert "enabled channels: 14/14" in dialog.summary_label.text().lower()
    assert "enabled-channel overlap" in dialog.metrics_label.text().lower()

    dialog.close()
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_preview_marks_excluded_channels_when_some_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().service.bandwidth_mhz = 5.0
    state.active_system().spectrum.service_band_start_mhz = 2620.0
    state.active_system().spectrum.service_band_stop_mhz = 2690.0
    state.active_system().spectrum.reuse_factor = 7
    state.active_system().spectrum.disabled_channel_indices = list(range(7, 14))
    assert state.ras_station is not None
    state.ras_station.receiver_band_start_mhz = 2690.0
    state.ras_station.receiver_band_stop_mhz = 2700.0
    spectrum_plan = sgui._normalize_spectrum_config(
        state.active_system().service,
        state.active_system().spectrum,
        state.ras_station,
        active_cell_count=7,
        active_cell_reuse_slot_ids=np.arange(7, dtype=np.int32),
    )
    dialog = sgui.SpectrumPreviewDialog(
        spectrum_plan=spectrum_plan,
        parent=window,
    )
    dialog.show()
    qapp.processEvents()

    checked_indices = dialog._selected_channel_indices()
    selection = sgui._resolve_preview_channel_selection(
        spectrum_plan,
        active_channel_indices=checked_indices,
    )
    enabled_metrics = sgui._spectrum_preview_metrics(
        spectrum_plan,
        sgui._build_spectrum_preview_curves(
            spectrum_plan,
            active_channel_indices=checked_indices,
        ),
        active_channel_indices=checked_indices,
    )
    assert len(checked_indices) == 7
    assert selection["selected_occupied_channel_indices"] == set(range(7))
    assert selection["selected_excluded_channel_indices"] == set(range(7, 14))
    assert "excluded" in dialog.channel_list.item(7).text().lower()
    assert float(enabled_metrics["overlap_mhz_equivalent"] or 0.0) == pytest.approx(0.0)
    assert "enabled channels: 7/14" in dialog.summary_label.text().lower()
    assert "mean=0" in dialog.summary_label.text().lower()

    excluded_meta = dialog._channel_metadata[7]
    excluded_midpoint = 0.5 * (
        float(excluded_meta["start_mhz"]) + float(excluded_meta["stop_mhz"])
    )
    dialog._on_canvas_motion(
        types.SimpleNamespace(
            inaxes=dialog._band_axis,
            xdata=excluded_midpoint,
        )
    )
    hover_text = dialog.hover_label.text().lower()
    assert "excluded from the run" in hover_text

    dialog.close()
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_preview_supports_channel_filtering_and_hover_slot_metadata(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().service.bandwidth_mhz = 5.0
    state.active_system().spectrum.service_band_start_mhz = 2620.0
    state.active_system().spectrum.service_band_stop_mhz = 2690.0
    state.active_system().spectrum.reuse_factor = 4
    state.active_system().spectrum.disabled_channel_indices = [12, 13]
    assert state.ras_station is not None
    state.ras_station.receiver_band_start_mhz = 2690.0
    state.ras_station.receiver_band_stop_mhz = 2700.0
    spectrum_plan = sgui._normalize_spectrum_config(
        state.active_system().service,
        state.active_system().spectrum,
        state.ras_station,
        active_cell_count=2,
        active_cell_reuse_slot_ids=np.asarray([0, 1], dtype=np.int32),
    )
    power_input = sgui._normalize_service_config_power_input(state.active_system().service)
    dialog = sgui.SpectrumPreviewDialog(
        spectrum_plan=spectrum_plan,
        power_input=power_input,
        parent=window,
    )
    dialog.show()
    qapp.processEvents()

    assert isinstance(dialog._toolbar, sgui.ScepterNavigationToolbar)
    assert "Axes & Grid" in [action.text() for action in dialog._toolbar.actions() if action.text()]
    assert dialog.channel_list.count() > 0
    initial_summary = dialog.summary_label.text()
    first_item = dialog.channel_list.item(0)
    first_item.setCheckState(QtCore.Qt.Unchecked)
    qapp.processEvents()
    assert dialog.summary_label.text() != initial_summary
    first_meta = dialog._channel_metadata[0]
    midpoint = 0.5 * (float(first_meta["start_mhz"]) + float(first_meta["stop_mhz"]))
    dialog._on_canvas_motion(
        types.SimpleNamespace(
            inaxes=dialog._band_axis,
            xdata=midpoint,
        )
    )
    hover_text = dialog.hover_label.text().lower()
    assert "channel 1" in hover_text
    assert "slot " in hover_text
    assert "excluded from the run" in hover_text
    metrics_text = dialog.metrics_label.text().lower()
    assert "enabled-channel overlap" in metrics_text
    x_limits_band = tuple(float(value) for value in dialog._band_axis.get_xlim())
    x_limits_mask = tuple(float(value) for value in dialog._mask_axis.get_xlim())
    assert x_limits_band == pytest.approx(x_limits_mask)
    first_item = dialog.channel_list.item(0)
    first_item.setCheckState(QtCore.Qt.Checked)
    qapp.processEvents()
    dialog._on_canvas_click(
        types.SimpleNamespace(
            inaxes=dialog._band_axis,
            xdata=midpoint,
            dblclick=True,
            button=1,
        )
    )
    qapp.processEvents()
    first_item = dialog.channel_list.item(0)
    assert first_item.checkState() == QtCore.Qt.Unchecked
    dialog.close()
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_response_axis_hides_cutoff_and_rectangular_out_of_band_tails() -> None:
    plan = _synthetic_spectrum_preview_plan(
        unwanted_emission_mask_points_mhz=np.asarray(
            [
                [-2.5, 0.0],
                [2.5, 0.0],
                [7.5, 25.0],
                [22.5, 25.0],
            ],
            dtype=np.float64,
        ),
        cutoff_mhz=40.0,
        receiver_mode="rectangular",
        receiver_points_mhz=None,
        ras_start_mhz=2690.0,
        ras_stop_mhz=2700.0,
    )
    curves = sgui._build_spectrum_preview_curves(plan)
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    sgui._populate_spectrum_response_axis(
        ax,
        plan=plan,
        curves=curves,
        title="Tx envelope, receiver response, and overlap",
        default_floor_db=-30.0,
        show_cutoff_boundaries=True,
        show_display_floor=True,
        style_mode="old_style_endpoint_drops",
    )

    floor_line = _find_axis_line_by_label_prefix(ax, "_display_floor")
    tx_drops = _find_axis_lines_by_label_prefix(ax, "_endpoint_drop_tx")
    receiver_drops = _find_axis_lines_by_label_prefix(ax, "_endpoint_drop_receiver")
    combined_drops = _find_axis_lines_by_label_prefix(ax, "_endpoint_drop_combined")
    clipped_floor_lines = _find_axis_lines_by_label_prefix(ax, "_clipped_overlap_floor")
    receiver_line = _find_axis_line_by_label_prefix(ax, "RAS receiver response")
    receiver_underlay = _find_axis_line_by_label_prefix(ax, "_receiver_response_underlay")
    legend = ax.get_legend()
    legend_labels = [] if legend is None else [text.get_text() for text in legend.get_texts()]
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer)
    title_bbox = _axis_nonempty_title(ax).get_window_extent(renderer)
    axes_bbox = ax.get_window_extent(renderer)

    assert tuple(float(value) for value in ax.get_ylim()) == pytest.approx((-30.0, 0.0))
    assert np.isfinite(np.asarray(curves["tx_total_display_db"], dtype=np.float64)).any()
    assert np.isnan(np.asarray(curves["tx_total_display_db"], dtype=np.float64)).any()
    assert np.isfinite(np.asarray(curves["receiver_display_db"], dtype=np.float64)).any()
    assert np.isnan(np.asarray(curves["receiver_display_db"], dtype=np.float64)).any()
    assert np.all(np.isnan(np.asarray(curves["combined_display_db"], dtype=np.float64)))
    assert np.allclose(np.asarray(floor_line.get_ydata(), dtype=np.float64), -30.0)
    assert clipped_floor_lines == []
    assert len(tx_drops) == 2
    assert len(receiver_drops) == 2
    assert combined_drops == []
    assert all(line.get_color() == "#0f766e" for line in tx_drops)
    assert all(line.get_color() == "#2563eb" for line in receiver_drops)
    assert legend_labels == [
        "Enabled-channel Tx envelope",
        "RAS receiver response",
        "Enabled-channel overlap weighting",
    ]
    assert not legend_bbox.overlaps(title_bbox)
    assert legend_bbox.y0 >= axes_bbox.y1 - 1.0
    assert legend_bbox.x0 >= axes_bbox.x0 - 1.0
    assert legend_bbox.x1 <= axes_bbox.x1 + 1.0
    assert receiver_underlay.get_clip_on() is False
    assert receiver_underlay.get_zorder() < receiver_line.get_zorder()
    assert receiver_line.get_clip_on() is False
    assert receiver_line.get_zorder() > ax.spines["top"].get_zorder()
    for drop_line in tx_drops + receiver_drops:
        ydata = np.asarray(drop_line.get_ydata(), dtype=np.float64)
        assert ydata.shape == (2,)
        assert ydata[0] > ydata[1]
        assert ydata[1] == pytest.approx(-30.0)
        assert _line_uses_solid_style(drop_line)
    plt.close(fig)


def test_spectrum_response_axis_draws_solid_cutoff_boundaries_in_default_style() -> None:
    plan = _synthetic_spectrum_preview_plan(
        ras_start_mhz=2630.0,
        ras_stop_mhz=2635.0,
        cutoff_mhz=12.0,
    )
    curves = sgui._build_spectrum_preview_curves(plan)
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    sgui._populate_spectrum_response_axis(
        ax,
        plan=plan,
        curves=curves,
        title="Tx envelope, receiver response, and overlap",
        default_floor_db=-30.0,
        show_cutoff_boundaries=True,
        show_display_floor=False,
        style_mode="default",
    )

    boundary_lines = [
        line
        for line in _axis_vertical_boundary_lines(ax)
        if str(line.get_label()) == "Cutoff boundary"
    ]

    assert len(boundary_lines) == 1
    assert _line_uses_solid_style(boundary_lines[0])
    plt.close(fig)


def test_spectrum_response_axis_extends_only_for_real_tx_or_receiver_levels() -> None:
    low_tx_plan = _synthetic_spectrum_preview_plan(
        unwanted_emission_mask_points_mhz=np.asarray(
            [
                [-2.5, 0.0],
                [2.5, 0.0],
                [7.5, 80.0],
                [22.5, 90.0],
            ],
            dtype=np.float64,
        ),
        cutoff_mhz=40.0,
        receiver_mode="rectangular",
        receiver_points_mhz=None,
        ras_start_mhz=2690.0,
        ras_stop_mhz=2700.0,
    )
    low_tx_curves = sgui._build_spectrum_preview_curves(low_tx_plan)
    low_tx_fig = Figure(figsize=(6.0, 4.0))
    low_tx_ax = low_tx_fig.add_subplot(111)
    sgui._populate_spectrum_response_axis(
        low_tx_ax,
        plan=low_tx_plan,
        curves=low_tx_curves,
        title="Tx envelope, receiver response, and overlap",
        default_floor_db=-30.0,
        show_cutoff_boundaries=True,
        show_display_floor=True,
        style_mode="old_style_endpoint_drops",
    )

    low_tx_floor_line = _find_axis_line_by_label_prefix(low_tx_ax, "_display_floor")
    low_tx_receiver_underlay = _find_axis_line_by_label_prefix(
        low_tx_ax,
        "_receiver_response_underlay",
    )
    assert float(low_tx_ax.get_ylim()[0]) < -30.0
    assert float(np.nanmin(np.asarray(low_tx_curves["tx_total_display_db"], dtype=np.float64))) < -30.0
    assert np.allclose(
        np.asarray(low_tx_floor_line.get_ydata(), dtype=np.float64),
        float(low_tx_ax.get_ylim()[0]),
    )
    assert low_tx_receiver_underlay.get_clip_on() is False
    plt.close(low_tx_fig)

    combined_only_plan = _synthetic_spectrum_preview_plan(
        service_start_mhz=2620.0,
        service_stop_mhz=2625.0,
        ras_start_mhz=2620.0,
        ras_stop_mhz=2625.0,
        cutoff_mhz=12.0,
        receiver_mode="custom",
        receiver_points_mhz=np.asarray(
            [
                [-10.0, 15.0],
                [-2.5, 0.0],
                [2.5, 0.0],
                [10.0, 15.0],
            ],
            dtype=np.float64,
        ),
        unwanted_emission_mask_points_mhz=np.asarray(
            [
                [-2.5, 0.0],
                [2.5, 0.0],
                [7.5, 20.0],
                [12.0, 20.0],
            ],
            dtype=np.float64,
        ),
    )
    combined_only_curves = sgui._build_spectrum_preview_curves(combined_only_plan)
    combined_only_fig = Figure(figsize=(6.0, 4.0))
    combined_only_ax = combined_only_fig.add_subplot(111)
    sgui._populate_spectrum_response_axis(
        combined_only_ax,
        plan=combined_only_plan,
        curves=combined_only_curves,
        title="Tx envelope, receiver response, and overlap",
        default_floor_db=-30.0,
        show_cutoff_boundaries=True,
        show_display_floor=True,
        style_mode="old_style_endpoint_drops",
    )

    combined_floor_line = _find_axis_line_by_label_prefix(combined_only_ax, "_display_floor")
    clipped_floor_line = _find_axis_line_by_label_prefix(
        combined_only_ax,
        "_clipped_overlap_floor",
    )
    clipped_y = np.asarray(clipped_floor_line.get_ydata(), dtype=np.float64)
    clipped_mask = np.isfinite(clipped_y)
    expected_clipped_mask = np.asarray(
        combined_only_curves["combined_display_valid"],
        dtype=bool,
    ) & (
        np.asarray(combined_only_curves["combined_display_db"], dtype=np.float64) <= -30.0
    )

    assert float(np.nanmin(np.asarray(combined_only_curves["tx_total_display_db"], dtype=np.float64))) > -30.0
    assert float(np.nanmin(np.asarray(combined_only_curves["receiver_display_db"], dtype=np.float64))) > -30.0
    assert float(np.nanmin(np.asarray(combined_only_curves["combined_display_db"], dtype=np.float64))) < -30.0
    assert tuple(float(value) for value in combined_only_ax.get_ylim()) == pytest.approx((-30.0, 0.0))
    assert np.allclose(np.asarray(combined_floor_line.get_ydata(), dtype=np.float64), -30.0)
    assert np.array_equal(clipped_mask, expected_clipped_mask)
    assert np.allclose(clipped_y[clipped_mask], -30.0)
    plt.close(combined_only_fig)


def test_spectrum_response_axis_auto_expands_top_for_positive_overlap_levels() -> None:
    spectrum_plan = sgui.scenario.normalize_direct_epfd_spectrum_plan(
        spectrum_plan={
            "service_band_start_mhz": 2620.0,
            "service_band_stop_mhz": 2690.0,
            "ras_receiver_band_start_mhz": 2690.0,
            "ras_receiver_band_stop_mhz": 2700.0,
            "reuse_factor": 1,
            "enabled_channel_indices": list(range(14)),
            "unwanted_emission_mask_preset": "custom",
            "custom_mask_points": np.asarray(
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
            ),
            "receiver_response_mode": "rectangular",
            "spectral_integration_cutoff_basis": "channel_bandwidth",
            "spectral_integration_cutoff_percent": 450.0,
        },
        channel_bandwidth_mhz=5.0,
        active_cell_count=1,
        active_cell_reuse_slot_ids=np.asarray([0], dtype=np.int32),
    )
    assert spectrum_plan is not None
    selected_channels = {11, 12, 13}
    curves = sgui._build_spectrum_preview_curves(
        spectrum_plan,
        active_channel_indices=selected_channels,
    )
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    sgui._populate_spectrum_response_axis(
        ax,
        plan=spectrum_plan,
        curves=curves,
        title="Tx envelope, receiver response, and overlap",
        default_floor_db=-30.0,
        show_cutoff_boundaries=True,
        show_display_floor=True,
        style_mode="old_style_endpoint_drops",
    )

    finite_visible_values = np.concatenate(
        [
            np.asarray(values, dtype=np.float64)[np.isfinite(np.asarray(values, dtype=np.float64))]
            for values in (
                curves["tx_total_display_db"],
                curves["receiver_display_db"],
                curves["combined_display_db"],
            )
        ]
    )
    max_visible_db = float(np.max(finite_visible_values))

    assert max_visible_db > 0.0
    assert float(ax.get_ylim()[1]) == pytest.approx(max_visible_db + 0.5)
    plt.close(fig)


def test_spectrum_response_axis_metrics_match_exact_selected_channel_overlap() -> None:
    selected_channels = [11, 12, 13]
    spectrum_plan = _normalized_compacted_spectrum_preview_plan(selected_channels)
    curves = sgui._build_spectrum_preview_curves(
        spectrum_plan,
    )
    metrics = sgui._spectrum_preview_metrics(
        spectrum_plan,
        curves,
    )
    leakage_factor_map = sgui._preview_channel_leakage_factor_map(spectrum_plan)
    expected_overlap_mhz = float(
        np.sum(
            np.asarray(
                [leakage_factor_map[int(channel_index)] for channel_index in selected_channels],
                dtype=np.float64,
            )
        )
        * float(spectrum_plan["channel_bandwidth_mhz"])
    )
    summary_text = sgui._spectrum_leakage_summary_text(spectrum_plan).lower()

    assert set(leakage_factor_map) == set(selected_channels)
    assert expected_overlap_mhz > 0.0
    assert float(metrics["overlap_mhz_equivalent"] or 0.0) == pytest.approx(expected_overlap_mhz)
    assert float(metrics["overlap_db_relative_1mhz"] or 0.0) == pytest.approx(
        10.0 * np.log10(expected_overlap_mhz)
    )
    assert "all selected min=" in summary_text
    assert "leaking channels only min=" in summary_text
    assert "no channels are enabled" not in summary_text


def test_spectrum_preview_metrics_keep_compacted_mixed_leaking_channels() -> None:
    enabled_channel_indices = list(range(10)) + [11]
    spectrum_plan = _normalized_compacted_spectrum_preview_plan(enabled_channel_indices)
    curves = sgui._build_spectrum_preview_curves(spectrum_plan)
    metrics = sgui._spectrum_preview_metrics(spectrum_plan, curves)
    leakage_factor_map = sgui._preview_channel_leakage_factor_map(spectrum_plan)
    expected_overlap_mhz = float(
        np.sum(np.asarray(list(leakage_factor_map.values()), dtype=np.float64))
        * float(spectrum_plan["channel_bandwidth_mhz"])
    )
    summary_text = sgui._spectrum_leakage_summary_text(spectrum_plan).lower()

    assert 11 in leakage_factor_map
    assert float(metrics["overlap_mhz_equivalent"] or 0.0) == pytest.approx(expected_overlap_mhz)
    assert expected_overlap_mhz > 0.0
    assert "all selected min=" in summary_text
    assert "leaking channels only min=" in summary_text
    assert "no channels are enabled" not in summary_text


@pytest.mark.parametrize(
    "service_case",
    _DIRECT_EPFD_GUI_SERVICE_CASES,
    ids=lambda case: str(case["case_id"]),
)
@pytest.mark.parametrize(
    "ras_case_name",
    (
        "upper_adjacent_rectangular",
        "lower_adjacent_rectangular",
        "upper_adjacent_custom_asymmetric",
    ),
)
def test_gui_preview_helpers_match_exact_leakage_across_supported_reuse_channelizations_and_ras_cases(
    service_case: Mapping[str, float | str],
    ras_case_name: str,
) -> None:
    full_channel_count = _gui_test_full_channel_count(service_case)
    ras_case = _gui_test_ras_case(service_case, ras_case_name)

    for reuse_factor in sgui.scenario._DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
        full_plan = _normalized_compacted_spectrum_preview_plan(
            list(range(full_channel_count)),
            service_band_start_mhz=float(service_case["service_band_start_mhz"]),
            service_band_stop_mhz=float(service_case["service_band_stop_mhz"]),
            ras_receiver_band_start_mhz=float(ras_case["ras_receiver_band_start_mhz"]),
            ras_receiver_band_stop_mhz=float(ras_case["ras_receiver_band_stop_mhz"]),
            channel_bandwidth_mhz=float(service_case["channel_bandwidth_mhz"]),
            reuse_factor=int(reuse_factor),
            receiver_response_mode=str(ras_case["receiver_response_mode"]),
            receiver_response_points_mhz=(
                None
                if ras_case["receiver_response_points_mhz"] is None
                else np.asarray(ras_case["receiver_response_points_mhz"], dtype=np.float64)
            ),
        )
        expected_leakage_map = _gui_test_channel_leakage_map_from_plan(full_plan)
        assert sgui._preview_channel_leakage_factor_map(full_plan) == pytest.approx(expected_leakage_map)

        for subset in _gui_structured_channel_subsets(full_channel_count, int(reuse_factor)):
            plan = _normalized_compacted_spectrum_preview_plan(
                subset,
                service_band_start_mhz=float(service_case["service_band_start_mhz"]),
                service_band_stop_mhz=float(service_case["service_band_stop_mhz"]),
                ras_receiver_band_start_mhz=float(ras_case["ras_receiver_band_start_mhz"]),
                ras_receiver_band_stop_mhz=float(ras_case["ras_receiver_band_stop_mhz"]),
                channel_bandwidth_mhz=float(service_case["channel_bandwidth_mhz"]),
                reuse_factor=int(reuse_factor),
                receiver_response_mode=str(ras_case["receiver_response_mode"]),
                receiver_response_points_mhz=(
                    None
                    if ras_case["receiver_response_points_mhz"] is None
                    else np.asarray(ras_case["receiver_response_points_mhz"], dtype=np.float64)
                ),
            )
            curves = sgui._build_spectrum_preview_curves(plan)
            metrics = sgui._spectrum_preview_metrics(plan, curves)
            summary_text = sgui._spectrum_leakage_summary_text(plan).lower()
            expected_selected_values = np.asarray(
                [expected_leakage_map[int(channel_index)] for channel_index in subset if int(channel_index) in expected_leakage_map],
                dtype=np.float64,
            )
            expected_overlap_mhz = float(
                np.sum(expected_selected_values, dtype=np.float64) * float(plan["channel_bandwidth_mhz"])
            )

            assert sgui._preview_channel_leakage_factor_map(plan) == pytest.approx(
                _gui_test_channel_leakage_map_from_plan(plan)
            )
            assert_allclose(
                sgui._selected_preview_channel_leakage_factors(plan),
                expected_selected_values,
                rtol=1.0e-7,
                atol=1.0e-10,
            )
            assert float(metrics["overlap_mhz_equivalent"] or 0.0) == pytest.approx(expected_overlap_mhz)

            if not subset:
                assert "no channels are enabled" in summary_text
            elif np.any(expected_selected_values > 0.0):
                assert "all selected min=" in summary_text
                assert "leaking channels only min=" in summary_text
                assert "no channels are enabled" not in summary_text
            else:
                assert "leaking channels only: none" in summary_text
                assert "no channels are enabled" not in summary_text


@pytest.mark.parametrize(
    ("service_case", "reuse_factor", "enabled_channel_indices"),
    (
        (_DIRECT_EPFD_GUI_SERVICE_CASES[1], 19, [16, 17, 18]),
        (_DIRECT_EPFD_GUI_SERVICE_CASES[2], 13, [9, 10, 11]),
    ),
    ids=("bw10_tail_channels", "bw15_tail_channels"),
)
def test_nondefault_channelization_dialogs_match_exact_overlap_and_summary_text(
    qapp: QtWidgets.QApplication,
    service_case: Mapping[str, float | str],
    reuse_factor: int,
    enabled_channel_indices: list[int],
) -> None:
    del qapp
    ras_case = _gui_test_ras_case(service_case, "upper_adjacent_rectangular")
    plan = _normalized_compacted_spectrum_preview_plan(
        enabled_channel_indices,
        service_band_start_mhz=float(service_case["service_band_start_mhz"]),
        service_band_stop_mhz=float(service_case["service_band_stop_mhz"]),
        ras_receiver_band_start_mhz=float(ras_case["ras_receiver_band_start_mhz"]),
        ras_receiver_band_stop_mhz=float(ras_case["ras_receiver_band_stop_mhz"]),
        channel_bandwidth_mhz=float(service_case["channel_bandwidth_mhz"]),
        reuse_factor=int(reuse_factor),
    )
    curves = sgui._build_spectrum_preview_curves(plan)
    metrics = sgui._spectrum_preview_metrics(plan, curves)
    expected_summary_text = sgui._spectrum_leakage_summary_text(plan)
    expected_overlap_text = f"{float(metrics['overlap_mhz_equivalent'] or 0.0):.6g} MHz-equiv"

    preview_dialog = sgui.SpectrumPreviewDialog(
        spectrum_plan=plan,
    )
    tx_mask_dialog = sgui.SpectrumMaskEditorDialog(
        initial_points=np.asarray(plan["unwanted_emission_mask_points_mhz"], dtype=np.float64).tolist(),
        channel_bandwidth_mhz=float(plan["channel_bandwidth_mhz"]),
        preview_plan_factory=lambda points: dict(
            plan,
            unwanted_emission_mask_points_mhz=np.asarray(points, dtype=np.float64),
        ),
    )

    assert expected_summary_text in preview_dialog.summary_label.text()
    assert expected_overlap_text in preview_dialog.metrics_label.text()
    assert expected_summary_text in tx_mask_dialog.preview_summary_label.text()

    preview_dialog.close()
    tx_mask_dialog.close()


@pytest.mark.slow_leakage_math
def test_gui_preview_helpers_match_exact_overlap_for_structured_sweeps_across_supported_reuse_channelizations() -> None:
    for service_case in _DIRECT_EPFD_GUI_SERVICE_CASES:
        full_channel_count = _gui_test_full_channel_count(service_case)
        for ras_case_name in (
            "upper_adjacent_rectangular",
            "lower_adjacent_rectangular",
            "upper_adjacent_custom_asymmetric",
        ):
            ras_case = _gui_test_ras_case(service_case, ras_case_name)
            for reuse_factor in sgui.scenario._DIRECT_EPFD_SUPPORTED_REUSE_FACTORS:
                full_plan = _normalized_compacted_spectrum_preview_plan(
                    list(range(full_channel_count)),
                    service_band_start_mhz=float(service_case["service_band_start_mhz"]),
                    service_band_stop_mhz=float(service_case["service_band_stop_mhz"]),
                    ras_receiver_band_start_mhz=float(ras_case["ras_receiver_band_start_mhz"]),
                    ras_receiver_band_stop_mhz=float(ras_case["ras_receiver_band_stop_mhz"]),
                    channel_bandwidth_mhz=float(service_case["channel_bandwidth_mhz"]),
                    reuse_factor=int(reuse_factor),
                    receiver_response_mode=str(ras_case["receiver_response_mode"]),
                    receiver_response_points_mhz=(
                        None
                        if ras_case["receiver_response_points_mhz"] is None
                        else np.asarray(ras_case["receiver_response_points_mhz"], dtype=np.float64)
                    ),
                )
                expected_leakage_map = _gui_test_channel_leakage_map_from_plan(full_plan)
                for subset in _gui_sampled_channel_subsets(
                    full_channel_count,
                    int(reuse_factor),
                    sample_count=64,
                    seed=20260401,
                ):
                    plan = _normalized_compacted_spectrum_preview_plan(
                        subset,
                        service_band_start_mhz=float(service_case["service_band_start_mhz"]),
                        service_band_stop_mhz=float(service_case["service_band_stop_mhz"]),
                        ras_receiver_band_start_mhz=float(ras_case["ras_receiver_band_start_mhz"]),
                        ras_receiver_band_stop_mhz=float(ras_case["ras_receiver_band_stop_mhz"]),
                        channel_bandwidth_mhz=float(service_case["channel_bandwidth_mhz"]),
                        reuse_factor=int(reuse_factor),
                        receiver_response_mode=str(ras_case["receiver_response_mode"]),
                        receiver_response_points_mhz=(
                            None
                            if ras_case["receiver_response_points_mhz"] is None
                            else np.asarray(ras_case["receiver_response_points_mhz"], dtype=np.float64)
                        ),
                    )
                    curves = sgui._build_spectrum_preview_curves(plan)
                    metrics = sgui._spectrum_preview_metrics(plan, curves)
                    summary_text = sgui._spectrum_leakage_summary_text(plan).lower()
                    expected_selected_values = np.asarray(
                        [expected_leakage_map[int(channel_index)] for channel_index in subset if int(channel_index) in expected_leakage_map],
                        dtype=np.float64,
                    )
                    expected_overlap_mhz = float(
                        np.sum(expected_selected_values, dtype=np.float64) * float(plan["channel_bandwidth_mhz"])
                    )

                    assert float(metrics["overlap_mhz_equivalent"] or 0.0) == pytest.approx(expected_overlap_mhz)
                    if not subset:
                        assert "no channels are enabled" in summary_text
                    elif np.any(expected_selected_values > 0.0):
                        assert "leaking channels only min=" in summary_text
                    else:
                        assert "leaking channels only: none" in summary_text


def test_edit_spectrum_dialog_uses_roomier_layout_and_floor_guides(
    qapp: QtWidgets.QApplication,
) -> None:
    spectrum_plan = _synthetic_spectrum_preview_plan(
        service_start_mhz=2620.0,
        service_stop_mhz=2625.0,
        ras_start_mhz=2620.0,
        ras_stop_mhz=2625.0,
        cutoff_mhz=12.0,
        receiver_mode="custom",
        receiver_points_mhz=np.asarray(
            [
                [-10.0, 10.0],
                [-2.5, 0.0],
                [2.5, 0.0],
                [10.0, 10.0],
            ],
            dtype=np.float64,
        ),
        unwanted_emission_mask_points_mhz=np.asarray(
            [
                [-12.0, 15.0],
                [-7.5, 15.0],
                [-2.5, 0.0],
                [2.5, 0.0],
                [7.5, 15.0],
                [12.0, 15.0],
            ],
            dtype=np.float64,
        ),
    )
    dialog = sgui.SpectrumPreviewDialog(
        spectrum_plan=spectrum_plan,
    )
    dialog.show()
    qapp.processEvents()

    gap = float(dialog._band_axis.get_position().y0 - dialog._mask_axis.get_position().y1)
    floor_line = _find_axis_line_by_label_prefix(dialog._mask_axis, "_display_floor")
    tx_drops = _find_axis_lines_by_label_prefix(dialog._mask_axis, "_endpoint_drop_tx")
    receiver_drops = _find_axis_lines_by_label_prefix(dialog._mask_axis, "_endpoint_drop_receiver")
    combined_drops = _find_axis_lines_by_label_prefix(dialog._mask_axis, "_endpoint_drop_combined")
    tx_line = _find_axis_line_by_label_prefix(dialog._mask_axis, "Enabled-channel Tx envelope")
    receiver_curve_line = _find_axis_line_by_label_prefix(dialog._mask_axis, "RAS receiver response")
    combined_line = _find_axis_line_by_label_prefix(
        dialog._mask_axis,
        "Enabled-channel overlap weighting",
    )
    receiver_line = _find_axis_line_by_label_prefix(dialog._mask_axis, "RAS receiver response")
    receiver_underlay = _find_axis_line_by_label_prefix(
        dialog._mask_axis,
        "_receiver_response_underlay",
    )
    legend = dialog._mask_axis.get_legend()
    legend_labels = [] if legend is None else [text.get_text() for text in legend.get_texts()]
    dialog._figure.canvas.draw()
    renderer = dialog._figure.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer)
    title_bbox = _axis_nonempty_title(dialog._mask_axis).get_window_extent(renderer)
    axes_bbox = dialog._mask_axis.get_window_extent(renderer)
    y_label_bbox = dialog._mask_axis.yaxis.label.get_window_extent(renderer)
    figure_bbox = dialog._figure.bbox

    assert gap > 0.09
    assert len(tx_drops) == 2
    assert len(receiver_drops) == 2
    assert len(combined_drops) == 2
    assert np.allclose(np.asarray(floor_line.get_ydata(), dtype=np.float64), -30.0)
    assert legend_labels == [
        "Enabled-channel Tx envelope",
        "RAS receiver response",
        "Enabled-channel overlap weighting",
    ]
    assert not legend_bbox.overlaps(title_bbox)
    assert legend_bbox.y0 >= axes_bbox.y1 - 1.0
    assert legend_bbox.x0 >= axes_bbox.x0 - 1.0
    assert legend_bbox.x1 <= figure_bbox.x1 + 1.0
    assert y_label_bbox.x0 >= figure_bbox.x0 - 1.0
    assert float(dialog._band_axis.get_position().x0) < 0.18
    assert float(dialog._mask_axis.get_position().x0) < 0.18
    assert receiver_underlay.get_clip_on() is False
    assert receiver_line.get_clip_on() is False
    assert receiver_line.get_zorder() > dialog._mask_axis.spines["top"].get_zorder()
    for drop_line in tx_drops:
        assert float(drop_line.get_linewidth()) == pytest.approx(float(tx_line.get_linewidth()))
        assert float(drop_line.get_alpha()) == pytest.approx(float(tx_line.get_alpha()))
        assert str(drop_line.get_solid_capstyle()) == str(tx_line.get_solid_capstyle())
    for drop_line in receiver_drops:
        assert float(drop_line.get_linewidth()) == pytest.approx(float(receiver_curve_line.get_linewidth()))
        assert float(drop_line.get_alpha()) == pytest.approx(float(receiver_curve_line.get_alpha()))
        assert str(drop_line.get_solid_capstyle()) == str(receiver_curve_line.get_solid_capstyle())
    for drop_line in combined_drops:
        assert float(drop_line.get_linewidth()) == pytest.approx(float(combined_line.get_linewidth()))
        assert float(drop_line.get_alpha()) == pytest.approx(float(combined_line.get_alpha()))
        assert str(drop_line.get_solid_capstyle()) == str(combined_line.get_solid_capstyle())
    for drop_line in tx_drops + receiver_drops + combined_drops:
        ydata = np.asarray(drop_line.get_ydata(), dtype=np.float64)
        assert ydata.shape == (2,)
        assert ydata[0] > ydata[1]
        assert ydata[1] == pytest.approx(-30.0)
        assert _line_uses_solid_style(drop_line)

    dialog.close()


def test_edit_spectrum_dialog_keeps_channel_list_position_stable_across_zero_and_leaking_metrics(
    qapp: QtWidgets.QApplication,
) -> None:
    def _build_preview_plan(selected_channel_indices: set[int]) -> dict[str, object]:
        return _normalized_compacted_spectrum_preview_plan(sorted(int(value) for value in selected_channel_indices))

    initial_selection = set(range(9))
    dialog = sgui.SpectrumPreviewDialog(
        spectrum_plan=_build_preview_plan(initial_selection),
        apply_selection_callback=lambda selected: (_build_preview_plan(selected), None),
    )
    dialog.show()
    qapp.processEvents()

    initial_list_top = int(dialog.channel_list.geometry().top())
    initial_summary_text = dialog.summary_label.text().lower()

    assert "leaking channels only: none" in initial_summary_text
    assert "no channels are enabled" not in initial_summary_text

    dialog.channel_list.item(11).setCheckState(QtCore.Qt.Checked)
    qapp.processEvents()
    leaking_list_top = int(dialog.channel_list.geometry().top())
    leaking_summary_text = dialog.summary_label.text().lower()

    dialog.channel_list.item(11).setCheckState(QtCore.Qt.Unchecked)
    qapp.processEvents()
    restored_list_top = int(dialog.channel_list.geometry().top())
    restored_summary_text = dialog.summary_label.text().lower()

    assert leaking_list_top == initial_list_top
    assert restored_list_top == initial_list_top
    assert "all selected min=" in leaking_summary_text
    assert "leaking channels only min=" in leaking_summary_text
    assert "mhz-equiv" in dialog.metrics_label.text().lower()
    assert "leaking channels only: none" in restored_summary_text

    dialog.close()


def test_tx_mask_editor_preview_uses_old_style_endpoint_drops(
    qapp: QtWidgets.QApplication,
) -> None:
    preview_plan = _synthetic_spectrum_preview_plan(
        service_start_mhz=2620.0,
        service_stop_mhz=2625.0,
        ras_start_mhz=2620.0,
        ras_stop_mhz=2625.0,
        cutoff_mhz=12.0,
        receiver_mode="custom",
        receiver_points_mhz=np.asarray(
            [
                [-10.0, 10.0],
                [-2.5, 0.0],
                [2.5, 0.0],
                [10.0, 10.0],
            ],
            dtype=np.float64,
        ),
        unwanted_emission_mask_points_mhz=np.asarray(
            [
                [-12.0, 15.0],
                [-7.5, 15.0],
                [-2.5, 0.0],
                [2.5, 0.0],
                [7.5, 15.0],
                [12.0, 15.0],
            ],
            dtype=np.float64,
        ),
    )
    dialog = sgui.SpectrumMaskEditorDialog(
        initial_points=np.asarray(
            preview_plan["unwanted_emission_mask_points_mhz"],
            dtype=np.float64,
        ).tolist(),
        channel_bandwidth_mhz=float(preview_plan["channel_bandwidth_mhz"]),
        preview_plan_factory=lambda points: dict(
            preview_plan,
            unwanted_emission_mask_points_mhz=np.asarray(points, dtype=np.float64),
        ),
    )
    dialog.show()
    qapp.processEvents()

    preview_labels = [str(line.get_label()) for line in dialog._preview_axis.lines]
    legend = dialog._preview_axis.get_legend()
    legend_labels = [] if legend is None else [text.get_text() for text in legend.get_texts()]
    receiver_underlay = _find_axis_line_by_label_prefix(
        dialog._preview_axis,
        "_receiver_response_underlay",
    )
    tx_line = _find_axis_line_by_label_prefix(dialog._preview_axis, "Enabled-channel Tx envelope")
    receiver_line = _find_axis_line_by_label_prefix(
        dialog._preview_axis,
        "RAS receiver response",
    )
    combined_line = _find_axis_line_by_label_prefix(
        dialog._preview_axis,
        "Enabled-channel overlap weighting",
    )
    tx_drops = _find_axis_lines_by_label_prefix(dialog._preview_axis, "_endpoint_drop_tx")
    receiver_drops = _find_axis_lines_by_label_prefix(dialog._preview_axis, "_endpoint_drop_receiver")
    combined_drops = _find_axis_lines_by_label_prefix(dialog._preview_axis, "_endpoint_drop_combined")
    dialog._figure.canvas.draw()
    renderer = dialog._figure.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer)
    title_bbox = _axis_nonempty_title(dialog._preview_axis).get_window_extent(renderer)
    axes_bbox = dialog._preview_axis.get_window_extent(renderer)
    y_label_bbox = dialog._preview_axis.yaxis.label.get_window_extent(renderer)
    figure_bbox = dialog._figure.bbox

    assert tuple(float(value) for value in dialog._preview_axis.get_ylim()) == pytest.approx((-30.0, 0.0))
    assert "_display_floor" in preview_labels
    assert len(tx_drops) == 2
    assert len(receiver_drops) == 2
    assert len(combined_drops) == 2
    assert legend_labels == [
        "Enabled-channel Tx envelope",
        "RAS receiver response",
        "Enabled-channel overlap weighting",
    ]
    assert not legend_bbox.overlaps(title_bbox)
    assert legend_bbox.y0 >= axes_bbox.y1 - 1.0
    assert legend_bbox.x0 >= axes_bbox.x0 - 1.0
    assert legend_bbox.x1 <= axes_bbox.x1 + 1.0
    assert y_label_bbox.x0 >= figure_bbox.x0 - 1.0
    assert receiver_underlay.get_clip_on() is False
    assert receiver_line.get_zorder() > dialog._preview_axis.spines["top"].get_zorder()
    for drop_line in tx_drops:
        assert float(drop_line.get_linewidth()) == pytest.approx(float(tx_line.get_linewidth()))
        assert float(drop_line.get_alpha()) == pytest.approx(float(tx_line.get_alpha()))
        assert str(drop_line.get_solid_capstyle()) == str(tx_line.get_solid_capstyle())
    for drop_line in receiver_drops:
        assert float(drop_line.get_linewidth()) == pytest.approx(float(receiver_line.get_linewidth()))
        assert float(drop_line.get_alpha()) == pytest.approx(float(receiver_line.get_alpha()))
        assert str(drop_line.get_solid_capstyle()) == str(receiver_line.get_solid_capstyle())
    for drop_line in combined_drops:
        assert float(drop_line.get_linewidth()) == pytest.approx(float(combined_line.get_linewidth()))
        assert float(drop_line.get_alpha()) == pytest.approx(float(combined_line.get_alpha()))
        assert str(drop_line.get_solid_capstyle()) == str(combined_line.get_solid_capstyle())
    for drop_line in (
        tx_drops + receiver_drops + combined_drops
    ):
        assert _line_uses_solid_style(drop_line)

    dialog.close()


def test_tx_mask_editor_preview_summary_tracks_zero_and_positive_leakage(
    qapp: QtWidgets.QApplication,
) -> None:
    del qapp
    zero_leak_plan = _normalized_compacted_spectrum_preview_plan(list(range(9)))
    leaking_plan = _normalized_compacted_spectrum_preview_plan([11, 12, 13])

    zero_dialog = sgui.SpectrumMaskEditorDialog(
        initial_points=np.asarray(
            zero_leak_plan["unwanted_emission_mask_points_mhz"],
            dtype=np.float64,
        ).tolist(),
        channel_bandwidth_mhz=float(zero_leak_plan["channel_bandwidth_mhz"]),
        preview_plan_factory=lambda points: dict(
            zero_leak_plan,
            unwanted_emission_mask_points_mhz=np.asarray(points, dtype=np.float64),
        ),
    )
    zero_summary_text = zero_dialog.preview_summary_label.text().lower()
    zero_overlap_line = _find_axis_line_by_label_prefix(
        zero_dialog._preview_axis,
        "Enabled-channel overlap weighting",
    )

    assert "leaking channels only: none" in zero_summary_text
    assert "no channels are enabled" not in zero_summary_text
    assert not np.isfinite(np.asarray(zero_overlap_line.get_ydata(), dtype=np.float64)).any()

    zero_dialog.close()

    leaking_dialog = sgui.SpectrumMaskEditorDialog(
        initial_points=np.asarray(
            leaking_plan["unwanted_emission_mask_points_mhz"],
            dtype=np.float64,
        ).tolist(),
        channel_bandwidth_mhz=float(leaking_plan["channel_bandwidth_mhz"]),
        preview_plan_factory=lambda points: dict(
            leaking_plan,
            unwanted_emission_mask_points_mhz=np.asarray(points, dtype=np.float64),
        ),
    )
    leaking_summary_text = leaking_dialog.preview_summary_label.text().lower()
    leaking_overlap_line = _find_axis_line_by_label_prefix(
        leaking_dialog._preview_axis,
        "Enabled-channel overlap weighting",
    )

    assert "all selected min=" in leaking_summary_text
    assert "leaking channels only min=" in leaking_summary_text
    assert "no channels are enabled" not in leaking_summary_text
    assert np.isfinite(np.asarray(leaking_overlap_line.get_ydata(), dtype=np.float64)).any()

    leaking_dialog.close()


def test_receiver_mask_editor_preview_keeps_default_response_styling(
    qapp: QtWidgets.QApplication,
) -> None:
    preview_plan = _synthetic_spectrum_preview_plan(
        receiver_mode="custom",
        receiver_points_mhz=np.asarray(
            [
                [-10.0, 40.0],
                [-2.5, 0.0],
                [2.5, 0.0],
                [10.0, 40.0],
            ],
            dtype=np.float64,
        ),
    )
    dialog = sgui.SpectrumMaskEditorDialog(
        initial_points=np.asarray(
            preview_plan["receiver_response_points_mhz"],
            dtype=np.float64,
        ).tolist(),
        channel_bandwidth_mhz=float(preview_plan["channel_bandwidth_mhz"]),
        preview_plan_factory=lambda points: dict(
            preview_plan,
            receiver_response_points_mhz=np.asarray(points, dtype=np.float64),
        ),
        mask_kind="rx",
    )
    dialog.show()
    qapp.processEvents()

    preview_labels = [str(line.get_label()).lower() for line in dialog._preview_axis.lines]

    assert tuple(float(value) for value in dialog._preview_axis.get_ylim()) == pytest.approx((-75.0, 0.0))
    assert not any("display floor" in label for label in preview_labels)
    assert not any("clipped to display floor" in label for label in preview_labels)
    assert len(_axis_vertical_boundary_lines(dialog._preview_axis)) == 0

    dialog.close()


def test_service_changes_recompute_spectrum_exact_fit_without_revisiting_spectrum_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(
        monkeypatch,
        state=sgui.ScepterProjectState(),
        current_hexgrid=False,
    )

    window._set_spectrum_defaults()
    assert "channel bandwidth" in window.spectrum_zero_leftover_label.text().lower()

    window.service_bandwidth_edit.set_value(5.0)
    window.service_bandwidth_edit.editingFinished.emit()
    QtWidgets.QApplication.processEvents()

    assert "F1" in window.spectrum_zero_leftover_label.text()
    assert "F7" in window.spectrum_zero_leftover_label.text()
    window._dirty = False; window.close()
    _clear_gui_settings()


def test_spectrum_change_marks_hexgrid_visual_stale_without_reopening_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gui_settings()
    window = _make_run_window(monkeypatch)
    render_calls = {"count": 0}

    def _unexpected_render(*_args: object, **_kwargs: object) -> bool:
        render_calls["count"] += 1
        return False

    monkeypatch.setattr(window, "_render_hexgrid_preview_from_cache", _unexpected_render)
    assert window._hexgrid_visual_is_current(window.current_state()) is True

    idx_f4 = window.spectrum_reuse_factor_combo.findData(4)
    window.spectrum_reuse_factor_combo.setCurrentIndex(idx_f4)
    QtWidgets.QApplication.processEvents()

    assert render_calls["count"] == 0
    assert window._hexgrid_visual_is_current(window.current_state()) is False
    assert window._hexgrid_outdated is True
    assert "stale" in window.hexgrid_preview_status_label.text().lower()

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_workspace_switch_tolerates_deleted_animation_object(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    qapp.processEvents()

    animation = QtCore.QPropertyAnimation(window, b"windowOpacity", window)
    window._workspace_animation = animation
    animation.deleteLater()
    qapp.processEvents()

    window.new_configuration()
    qapp.processEvents()

    assert window.current_workspace() == sgui._WORKSPACE_SIMULATION
    window._dirty = False; window.close()


def test_workspace_switch_runs_geometry_stabilization(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    calls: list[str] = []
    monkeypatch.setattr(
        window,
        "_schedule_workspace_geometry_stabilization",
        lambda: calls.append("stabilized"),
    )

    # Switch away first (reset already set SIMULATION), then switch back
    window._set_workspace(sgui._WORKSPACE_HOME)
    qapp.processEvents()
    calls.clear()
    window._set_workspace(sgui._WORKSPACE_SIMULATION)
    qapp.processEvents()

    assert calls == ["stabilized"]
    assert window.current_workspace() == sgui._WORKSPACE_SIMULATION
    window._dirty = False; window.close()


def test_resize_refresh_reflows_runtime_buttons_without_clipping(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.resize(1280, 720)
    window.show()
    qapp.processEvents()

    window._select_simulation_page(window.ras_tab)
    qapp.processEvents()
    assert window._ras_body_layout.direction() == QtWidgets.QBoxLayout.TopToBottom

    window._select_simulation_page(window.runtime_page)
    qapp.processEvents()

    assert window._current_simulation_page() is window.runtime_page
    assert window.run_simulation_button.isHidden() is False
    assert window.force_stop_simulation_button.isHidden() is False
    assert window.runtime_tab.viewport().width() > 0
    assert window.runtime_tab.verticalScrollBar().maximum() >= 0
    window._dirty = False; window.close()


def test_home_workspace_scrolls_and_reflows_at_minimum_width(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.resize(1280, 720)
    window.show()
    window._set_workspace(sgui._WORKSPACE_HOME)
    qapp.processEvents()

    assert isinstance(window.home_workspace, QtWidgets.QScrollArea)
    assert window.home_workspace.objectName() == "homeScrollArea"
    assert window.home_workspace.viewport().objectName() == "homeScrollViewport"
    assert window.home_workspace.widget().objectName() == "homeScrollPage"
    assert window.home_workspace.viewport().width() > 0
    assert window._home_hero_button_layout.itemAtPosition(1, 0).widget() is window.home_open_result_button
    assert window.home_workspace.verticalScrollBar().maximum() >= 0

    window.resize(980, 720)
    qapp.processEvents()

    assert window._home_body_layout.direction() == QtWidgets.QBoxLayout.TopToBottom
    window._dirty = False; window.close()


def test_simulation_assistant_panel_uses_scroll_area_without_unbounded_snapshot_growth(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._set_workspace(sgui._WORKSPACE_SIMULATION)
    qapp.processEvents()

    assert isinstance(window._assistant_scroll, QtWidgets.QScrollArea)
    assert window._assistant_scroll.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
    assert window.snapshot_step_list.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
    assert window.snapshot_step_list.maximumHeight() <= 300
    window._dirty = False; window.close()


def test_detached_figure_window_uses_branded_toolbar_and_modern_marker_toggle(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    fig = Figure(figsize=(4.0, 3.0))
    window = sgui.FigureWindow(figure=fig, title="Test figure")
    qapp.processEvents()

    assert isinstance(window._toolbar, sgui.ScepterNavigationToolbar)
    assert window._toolbar is not None
    toolbar_actions = [action.text() for action in window._toolbar.actions() if action.text()]
    assert "Axes & Grid" in toolbar_actions
    assert "Marker mode" in toolbar_actions
    assert "Clear markers" in toolbar_actions
    snap_combo = window._toolbar.findChild(QtWidgets.QComboBox)
    assert snap_combo is not None
    assert snap_combo.currentData() == "off"
    assert window.windowIcon().isNull() is False
    window._dirty = False; window.close()


def test_axes_grid_dialog_updates_detached_figure_axes(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    fig = Figure(figsize=(4.0, 3.0))
    axis = fig.add_subplot(111)
    axis.plot([0.0, 1.0], [0.0, 1.0])
    axis.grid(True, alpha=0.18, linestyle=":")
    window = sgui.FigureWindow(figure=fig, title="Axes control test")
    window.show()
    qapp.processEvents()

    assert window._toolbar is not None
    window._toolbar.open_axes_grid_dialog()
    qapp.processEvents()
    dialog = window._toolbar._axes_grid_dialog
    assert isinstance(dialog, sgui.AxesGridDialog)

    dialog.autoscale_x_check.setChecked(False)
    dialog.x_min_spin.setValue(-2.0)
    dialog.x_max_spin.setValue(4.0)
    dialog.grid_enabled_check.setChecked(False)
    dialog.x_ticks_check.setChecked(False)
    dialog.y_ticklabels_check.setChecked(False)
    qapp.processEvents()

    assert tuple(float(value) for value in axis.get_xlim()) == pytest.approx((-2.0, 4.0))
    assert not any(line.get_visible() for line in axis.get_xgridlines())
    assert not any(line.get_visible() for line in axis.xaxis.get_ticklines())
    assert not any(label.get_visible() for label in axis.get_yticklabels())
    dialog.close()
    qapp.processEvents()
    window._dirty = False; window.close()
    plt.close(fig)


def test_detached_figure_window_reuses_dialog_when_swapping_figures(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    fig1 = Figure(figsize=(4.0, 3.0))
    fig1.add_subplot(111)
    fig2 = Figure(figsize=(5.0, 4.0))
    fig2.add_subplot(111)
    window = sgui.FigureWindow(figure=fig1, title="First figure")
    window.show()
    qapp.processEvents()

    first_canvas = window._canvas
    first_toolbar = window._toolbar

    window.update_figure(figure=fig2, title="Updated figure")
    qapp.processEvents()

    assert window.windowTitle() == "Updated figure"
    assert window._canvas is not first_canvas
    assert window._toolbar is not first_toolbar
    assert window.isVisible() is True
    window._dirty = False; window.close()


def test_postprocess_canvas_uses_resize_proxy_during_resize(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_MATPLOTLIB_RESIZE_SETTLE_MS", 1000)
    window = sgui.ScepterMainWindow()
    widget = window.postprocess_widget
    fig = Figure(figsize=(4.0, 3.0))
    fig.add_subplot(111)
    widget._replace_plot_canvas(fig)
    window._set_workspace(sgui._WORKSPACE_POSTPROCESS)
    window.show()
    widget.plot_canvas_container.setCurrentWidget(widget.plot_canvas)
    qapp.processEvents()

    canvas = widget.plot_canvas
    assert isinstance(canvas, sgui.DeferredResizeFigureCanvas)
    assert canvas.isVisible() is True
    proxy_pixmap = QtGui.QPixmap(120, 90)
    proxy_pixmap.fill(QtGui.QColor("#0f172a"))
    monkeypatch.setattr(canvas, "grab", lambda: proxy_pixmap)

    current_size = canvas.size()
    target_size = QtCore.QSize(current_size.width() + 123, current_size.height() + 57)
    canvas.resizeEvent(QtGui.QResizeEvent(target_size, current_size))
    qapp.processEvents()

    assert canvas._resize_proxy_active is True
    assert canvas._resize_proxy_label is not None
    assert canvas._resize_proxy_label.isVisible() is True

    _wait_until(lambda: not canvas._resize_proxy_active, timeout_ms=1000)
    qapp.processEvents()
    assert canvas._resize_proxy_label is not None
    assert canvas._resize_proxy_label.isVisible() is False
    assert isinstance(widget.plot_toolbar, sgui.ScepterNavigationToolbar)
    assert "Axes & Grid" in [action.text() for action in widget.plot_toolbar.actions() if action.text()]
    window._dirty = False; window.close()


def test_embedded_dark_mode_postprocess_render_releases_resize_proxy_after_programmatic_swap(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_MATPLOTLIB_RESIZE_SETTLE_MS", 1000)
    legacy_file = (
        Path(__file__).resolve().parents[2]
        / "simulation_results_1_13_System3_B525_1_4_GHz_baseline.h5"
    )
    if not legacy_file.exists():
        pytest.skip("Legacy baseline compatibility file is not present in this checkout.")

    window = sgui.ScepterMainWindow()
    window.show()
    qapp.processEvents()
    window.appearance_mode_combo.setCurrentIndex(window.appearance_mode_combo.findData("dark"))
    qapp.processEvents()
    window._open_result_in_postprocess(str(legacy_file))
    qapp.processEvents()

    widget = window.postprocess_widget
    for row in range(widget.recipe_list.count()):
        item = widget.recipe_list.item(row)
        if item is not None and item.data(QtCore.Qt.UserRole) == "prx_total_distribution":
            widget.recipe_list.setCurrentRow(row)
            qapp.processEvents()
            break
    else:
        raise AssertionError("prx_total_distribution")

    widget._render_current_recipe()
    qapp.processEvents()
    _wait_until(
        lambda: widget.plot_canvas_container.currentWidget() is widget.plot_canvas,
        timeout_ms=1000,
    )
    _wait_until(lambda: not widget.plot_canvas._resize_proxy_active, timeout_ms=250)
    qapp.processEvents()

    assert widget.plot_canvas._resize_proxy_active is False
    assert widget.plot_canvas._resize_proxy_label is None or not widget.plot_canvas._resize_proxy_label.isVisible()
    window._dirty = False; window.close()


def test_detached_figure_window_canvas_uses_resize_proxy_during_resize(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    monkeypatch.setattr(sgui, "_MATPLOTLIB_RESIZE_SETTLE_MS", 1000)
    fig = Figure(figsize=(4.0, 3.0))
    fig.add_subplot(111)
    window = sgui.FigureWindow(figure=fig, title="Resizable figure")
    window.show()
    qapp.processEvents()

    assert isinstance(window._canvas, sgui.DeferredResizeFigureCanvas)
    proxy_pixmap = QtGui.QPixmap(120, 90)
    proxy_pixmap.fill(QtGui.QColor("#0f172a"))
    monkeypatch.setattr(window._canvas, "grab", lambda: proxy_pixmap)

    window.resize(980, 680)
    qapp.processEvents()

    assert window._canvas._resize_proxy_active is True
    assert window._canvas._resize_proxy_label is not None
    assert window._canvas._resize_proxy_label.isVisible() is True

    _wait_until(lambda: not window._canvas._resize_proxy_active, timeout_ms=1000)
    qapp.processEvents()
    assert window._canvas._resize_proxy_label is not None
    assert window._canvas._resize_proxy_label.isVisible() is False
    window._dirty = False; window.close()


def test_app_icon_helpers_return_non_null_icon_and_splash(
    qapp: QtWidgets.QApplication,
) -> None:
    icon = sgui._load_app_icon()
    splash = gui_bootstrap.create_startup_splash(icon, minimum_visible_ms=0)

    assert icon.isNull() is False
    assert splash.pixmap().isNull() is False
    splash.set_stage(progress=0.5, status_text="Halfway there")
    assert splash._progress_value == pytest.approx(0.5)
    assert splash._status_text == "Halfway there"
    splash.close()


def test_resolve_app_icon_path_prefers_windows_ico(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gui_bootstrap.sys, "platform", "win32")
    icon_path = gui_bootstrap.resolve_app_icon_path()

    assert icon_path is not None
    assert icon_path.suffix.lower() == ".ico"


def test_apply_windows_window_icon_noops_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gui_bootstrap.sys, "platform", "linux")

    class _Widget:
        def winId(self) -> int:
            return 123

    assert gui_bootstrap.apply_windows_window_icon(_Widget()) is False


def test_apply_windows_window_icon_uses_native_shell_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int, int]] = []

    class _User32:
        @staticmethod
        def LoadImageW(_instance: object, _path: str, _image_type: int, width: int, height: int, _flags: int) -> int:
            return 1000 + int(width) + int(height)

        @staticmethod
        def SendMessageW(hwnd: int, _message: int, which: int, handle: int) -> int:
            calls.append(("send", int(hwnd), int(which), int(handle)))
            return 0

        @staticmethod
        def SetClassLongPtrW(hwnd: int, which: int, handle: int) -> int:
            calls.append(("class", int(hwnd), int(which), int(handle)))
            return 0

    class _Widget:
        def winId(self) -> int:
            return 321

    monkeypatch.setattr(gui_bootstrap.sys, "platform", "win32")
    monkeypatch.setattr(
        gui_bootstrap.ctypes,
        "windll",
        types.SimpleNamespace(user32=_User32()),
        raising=False,
    )

    assert gui_bootstrap.apply_windows_window_icon(_Widget(), gui_bootstrap._WINDOWS_APP_ICON_PATH) is True
    assert any(call[:3] == ("send", 321, 1) for call in calls)
    assert any(call[:3] == ("send", 321, 0) for call in calls)


def test_windows_shell_identity_helper_sets_app_user_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Shell32:
        @staticmethod
        def SetCurrentProcessExplicitAppUserModelID(app_id: str) -> int:
            calls.append(str(app_id))
            return 0

    monkeypatch.setattr(gui_bootstrap.ctypes, "windll", types.SimpleNamespace(shell32=_Shell32()), raising=False)
    monkeypatch.setattr(gui_bootstrap.sys, "platform", "win32")

    assert gui_bootstrap.configure_windows_shell_identity("org.skao.scepter.test") is True
    assert calls == ["org.skao.scepter.test"]


def test_run_monitor_raw_log_uses_themed_object_names(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window.show()
    qapp.processEvents()
    monitor = window.run_monitor

    assert monitor.log_toggle.objectName() == "runLogToggle"
    assert monitor.log_view.objectName() == "runLogView"
    assert "QToolButton#runLogToggle" in window.styleSheet()
    assert "QPlainTextEdit#runLogView" in window.styleSheet()

    monitor.log_toggle.click()
    qapp.processEvents()
    assert monitor.log_view.isHidden() is False

    window._dirty = False; window.close()
    _clear_gui_settings()


def test_help_menu_and_about_text_expose_v0111(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    menu_titles = [action.text() for action in window.menuBar().actions()]
    assert "&Help" in menu_titles
    assert window.action_about.text() == "About SCEPTer"
    assert "v0.25.0" in window._about_dialog_text()
    assert "v0.25.0" in window.windowTitle().lower()
    window._dirty = False; window.close()


def test_v01_layout_keeps_wider_antenna_combo_and_taller_contour_summary(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    assert window.antenna_model_combo.maximumWidth() >= sgui._FIELD_WIDTH_WIDE
    assert window.antenna_model_combo.maximumWidth() > sgui._FIELD_WIDTH_MEDIUM
    assert window.grid_summary_text.minimumHeight() >= 176
    assert window.grid_summary_text.maximumHeight() >= 176
    window._dirty = False; window.close()


def test_build_run_request_rejects_missing_service_target_pfd_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    state = _tiny_state()
    state.active_system().service.target_pfd_dbw_m2_mhz = None
    window._load_state_into_widgets(state)
    _mark_hexgrid_preview_current(window)

    with pytest.raises(ValueError, match="service.*incomplete|finite service power"):
        window._build_run_request(window.current_state())

    window._dirty = False; window.close()
