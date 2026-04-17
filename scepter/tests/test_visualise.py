import importlib.machinery
import sys
import types
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

matplotlib.use("Agg")


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

from scepter import earthgrid, visualise


def _sample_prepared_grid() -> dict[str, object]:
    return {
        "pre_ras_cell_longitudes": np.asarray([20.0, 20.5, 21.0], dtype=np.float64) * u.deg,
        "pre_ras_cell_latitudes": np.asarray([-30.8, -30.7, -30.6], dtype=np.float64) * u.deg,
        "active_grid_longitudes": np.asarray([20.0, 21.0], dtype=np.float64) * u.deg,
        "active_grid_latitudes": np.asarray([-30.8, -30.6], dtype=np.float64) * u.deg,
        "ras_exclusion_mask_pre_ras": np.asarray([False, True, False], dtype=bool),
        "pre_ras_to_active": np.asarray([0, -1, 1], dtype=np.int32),
        "ras_service_cell_index_pre_ras": 0,
        "ras_service_cell_index": 0,
        "point_spacing_km": 79.141,
        "station_lon": 21.443611 * u.deg,
        "station_lat": -30.712777 * u.deg,
    }


def _sample_shared_lattice_preview(
    *,
    reuse_factor: int,
) -> tuple[dict[str, object], dict[str, object]]:
    point_spacing_km = 79.141
    station_lon_deg = 21.443611
    station_lat_deg = -30.712777
    axial_q = np.asarray([0, 1], dtype=np.int32)
    axial_r = np.asarray([0, 0], dtype=np.int32)
    center_x_km, center_y_km = visualise._hexgrid_center_xy_km_from_axial(
        axial_q,
        axial_r,
        point_spacing_km=point_spacing_km,
        orientation_name="pointy",
    )
    lon_deg, lat_deg = earthgrid._local_tangent_plane_lonlat_from_xy_km(
        center_x_km,
        center_y_km,
        ref_lon_deg=station_lon_deg,
        ref_lat_deg=station_lat_deg,
    )
    prepared_grid = {
        "pre_ras_cell_longitudes": np.asarray(lon_deg, dtype=np.float64) * u.deg,
        "pre_ras_cell_latitudes": np.asarray(lat_deg, dtype=np.float64) * u.deg,
        "active_grid_longitudes": np.asarray(lon_deg, dtype=np.float64) * u.deg,
        "active_grid_latitudes": np.asarray(lat_deg, dtype=np.float64) * u.deg,
        "ras_exclusion_mask_pre_ras": np.asarray([False, False], dtype=bool),
        "pre_ras_to_active": np.asarray([0, 1], dtype=np.int32),
        "ras_service_cell_index_pre_ras": 0,
        "ras_service_cell_index": 0,
        "point_spacing_km": point_spacing_km,
        "station_lon": station_lon_deg * u.deg,
        "station_lat": station_lat_deg * u.deg,
    }
    reuse_plan = {
        "reuse_factor": int(reuse_factor),
        "anchor_slot": 0,
        "anchor_pre_ras_index": 0,
        "anchor_active_index": 0,
        "point_spacing_km_used": point_spacing_km,
        "orientation_used": "pointy",
        "fit_residual_km2": 0.0,
        "axial_q_pre_ras": axial_q,
        "axial_r_pre_ras": axial_r,
        "axial_q_active": axial_q.copy(),
        "axial_r_active": axial_r.copy(),
        "pre_ras_slot_ids": np.mod(np.asarray([0, 1], dtype=np.int32), max(1, int(reuse_factor))),
        "active_slot_ids": np.mod(np.asarray([0, 1], dtype=np.int32), max(1, int(reuse_factor))),
        "cluster_representatives": [
            {"slot_id": 0, "base_slot_id": 0, "axial_q": 0, "axial_r": 0}
        ],
    }
    return prepared_grid, reuse_plan


def test_plot_cell_status_map_hides_empty_legend_entries_for_reuse_coloring() -> None:
    prepared_grid = _sample_prepared_grid()
    fig, info = visualise.plot_cell_status_map(
        prepared_grid["pre_ras_cell_longitudes"],
        prepared_grid["pre_ras_cell_latitudes"],
        active_cell_longitudes=prepared_grid["active_grid_longitudes"],
        active_cell_latitudes=prepared_grid["active_grid_latitudes"],
        switched_off_mask=prepared_grid["ras_exclusion_mask_pre_ras"],
        boresight_affected_cell_ids=np.asarray([1], dtype=np.int32),
        active_reuse_slot_ids=np.asarray([0, 1], dtype=np.int32),
        reuse_factor=4,
        anchor_active_index=0,
        ras_longitude=prepared_grid["station_lon"],
        ras_latitude=prepared_grid["station_lat"],
        backend="matplotlib",
        map_style="clean",
        point_spacing_km=float(prepared_grid["point_spacing_km"]),
        return_info=True,
    )

    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert getattr(legend, "_ncols", None) == 1
    bbox = legend.get_bbox_to_anchor()
    assert bbox._bbox.x0 == pytest.approx(1.02)
    assert bbox._bbox.y0 == pytest.approx(0.5)
    assert "Normal active (1)" not in labels
    assert "Boresight-affected active (1)" not in labels
    assert "Reuse slot 1" in labels
    assert "Reuse slot 2" in labels
    assert "Reuse slot 3" not in labels
    assert "Reuse slot 4" not in labels
    assert "Reuse anchor cell" not in labels
    assert "RAS station" in labels
    cell_collections = [
        collection
        for collection in fig.axes[0].collections
        if str(collection.get_gid() or "").startswith("hexgrid_")
    ]
    assert cell_collections
    first_paths = cell_collections[0].get_paths()
    assert first_paths
    assert first_paths[0].vertices.shape[0] >= 6
    assert info["point_spacing_km_used"] == pytest.approx(float(prepared_grid["point_spacing_km"]), rel=0.05)
    assert info["reuse_coloring_active"] is True
    assert info["hex_geometry_mode"] in {"center_offsets", "inferred_lattice", "voronoi"}


def test_plot_cell_status_map_hex_polygons_scale_with_zoom() -> None:
    prepared_grid = _sample_prepared_grid()
    fig, info = visualise.plot_cell_status_map(
        prepared_grid["pre_ras_cell_longitudes"],
        prepared_grid["pre_ras_cell_latitudes"],
        active_cell_longitudes=prepared_grid["active_grid_longitudes"],
        active_cell_latitudes=prepared_grid["active_grid_latitudes"],
        switched_off_mask=prepared_grid["ras_exclusion_mask_pre_ras"],
        active_reuse_slot_ids=np.asarray([0, 1], dtype=np.int32),
        reuse_factor=4,
        anchor_active_index=0,
        ras_longitude=prepared_grid["station_lon"],
        ras_latitude=prepared_grid["station_lat"],
        backend="matplotlib",
        map_style="clean",
        point_spacing_km=float(prepared_grid["point_spacing_km"]),
        return_info=True,
    )

    assert info["hex_orientation_used"] in {"pointy", "flat", "voronoi"}
    assert info["hex_geometry_mode"] in {"center_offsets", "inferred_lattice", "voronoi"}
    ax = fig.axes[0]
    slot_collection = next(
        collection
        for collection in ax.collections
        if str(collection.get_gid() or "").startswith("hexgrid_reuse_slot_")
    )
    path = slot_collection.get_paths()[0]
    polygon_vertices = np.asarray(path.vertices[:-1], dtype=np.float64)
    expected_center = np.asarray(
        [
            float(u.Quantity(prepared_grid["active_grid_longitudes"][0]).to_value(u.deg)),
            float(u.Quantity(prepared_grid["active_grid_latitudes"][0]).to_value(u.deg)),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.mean(polygon_vertices, axis=0), expected_center, atol=5.0e-4)

    fig.canvas.draw()
    width_before = float(np.ptp(ax.transData.transform(polygon_vertices)[:, 0]))
    ax.set_xlim(expected_center[0] - 0.2, expected_center[0] + 0.2)
    ax.set_ylim(expected_center[1] - 0.2, expected_center[1] + 0.2)
    fig.canvas.draw()
    width_after = float(np.ptp(ax.transData.transform(polygon_vertices)[:, 0]))
    assert width_after > width_before
    plt.close(fig)


def test_plot_cell_status_map_uses_shared_lattice_geometry_when_available() -> None:
    prepared_grid, reuse_plan = _sample_shared_lattice_preview(reuse_factor=4)
    fig, info = visualise.plot_cell_status_map(
        prepared_grid["pre_ras_cell_longitudes"],
        prepared_grid["pre_ras_cell_latitudes"],
        active_cell_longitudes=prepared_grid["active_grid_longitudes"],
        active_cell_latitudes=prepared_grid["active_grid_latitudes"],
        switched_off_mask=prepared_grid["ras_exclusion_mask_pre_ras"],
        active_reuse_slot_ids=reuse_plan["active_slot_ids"],
        reuse_factor=int(reuse_plan["reuse_factor"]),
        anchor_active_index=int(reuse_plan["anchor_active_index"]),
        ras_longitude=prepared_grid["station_lon"],
        ras_latitude=prepared_grid["station_lat"],
        backend="matplotlib",
        map_style="clean",
        point_spacing_km=float(prepared_grid["point_spacing_km"]),
        hex_lattice=reuse_plan,
        return_info=True,
    )

    assert info["hex_geometry_mode"] == "shared_lattice"
    assert info["hex_anchor_pre_ras_index"] == 0
    assert info["hex_center_render_residual_km_max"] == pytest.approx(0.0, abs=5.0e-12)
    assert info["hex_center_render_residual_km_median"] == pytest.approx(0.0, abs=5.0e-12)

    ax = fig.axes[0]
    slot_one = next(
        collection
        for collection in ax.collections
        if str(collection.get_gid() or "") == "hexgrid_reuse_slot_1"
    )
    slot_two = next(
        collection
        for collection in ax.collections
        if str(collection.get_gid() or "") == "hexgrid_reuse_slot_2"
    )
    anchor_vertices = np.asarray(slot_one.get_paths()[0].vertices[:-1], dtype=np.float64)
    east_vertices = np.asarray(slot_two.get_paths()[0].vertices[:-1], dtype=np.float64)
    shared_vertex_count = sum(
        any(np.allclose(anchor_vertex, east_vertex, atol=1.0e-12, rtol=0.0) for east_vertex in east_vertices)
        for anchor_vertex in anchor_vertices
    )
    assert shared_vertex_count == 2

    expected_center = np.asarray(
        [
            float(u.Quantity(prepared_grid["active_grid_longitudes"][0]).to_value(u.deg)),
            float(u.Quantity(prepared_grid["active_grid_latitudes"][0]).to_value(u.deg)),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.mean(anchor_vertices, axis=0), expected_center, atol=5.0e-4)
    plt.close(fig)


def test_plot_cell_status_map_uses_shared_lattice_geometry_for_f1_preview() -> None:
    prepared_grid, reuse_plan = _sample_shared_lattice_preview(reuse_factor=1)
    fig, info = visualise.plot_cell_status_map(
        prepared_grid["pre_ras_cell_longitudes"],
        prepared_grid["pre_ras_cell_latitudes"],
        active_cell_longitudes=prepared_grid["active_grid_longitudes"],
        active_cell_latitudes=prepared_grid["active_grid_latitudes"],
        switched_off_mask=prepared_grid["ras_exclusion_mask_pre_ras"],
        reuse_factor=int(reuse_plan["reuse_factor"]),
        anchor_active_index=int(reuse_plan["anchor_active_index"]),
        ras_longitude=prepared_grid["station_lon"],
        ras_latitude=prepared_grid["station_lat"],
        backend="matplotlib",
        map_style="clean",
        point_spacing_km=float(prepared_grid["point_spacing_km"]),
        hex_lattice=reuse_plan,
        return_info=True,
    )

    assert info["hex_geometry_mode"] == "shared_lattice"
    assert info["reuse_factor"] == 1
    active_collection = next(
        collection
        for collection in fig.axes[0].collections
        if str(collection.get_gid() or "") == "hexgrid_active_cells"
    )
    assert len(active_collection.get_paths()) == 2
    plt.close(fig)


def test_plot_frequency_reuse_scheme_smoke() -> None:
    prepared_grid = _sample_prepared_grid()
    reuse_plan = earthgrid.resolve_frequency_reuse_slots(
        prepared_grid,
        reuse_factor=4,
        anchor_slot=2,
    )

    fig, info = visualise.plot_frequency_reuse_scheme(
        prepared_grid=prepared_grid,
        reuse_plan=reuse_plan,
        boresight_affected_cell_ids=np.asarray([1], dtype=np.int32),
        return_info=True,
    )

    assert len(fig.axes) == 1
    assert info["reuse_factor"] == 4
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Frequency Reuse Scheme"
    assert info["slot_ids_present"] == tuple(
        int(value)
        for value in sorted(
            {int(representative["slot_id"]) for representative in reuse_plan["cluster_representatives"]}
        )
    )
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert getattr(legend, "_ncols", None) == 1
    bbox = legend.get_bbox_to_anchor()
    assert bbox._bbox.x0 == pytest.approx(1.02)
    assert bbox._bbox.y0 == pytest.approx(0.5)


def test_plot_frequency_reuse_scheme_f1_uses_gapless_hex_center_spacing() -> None:
    fig, info = visualise.plot_frequency_reuse_scheme(
        reuse_factor=1,
        return_info=True,
    )

    assert info["reuse_factor"] == 1
    expected_orientation = pytest.approx(float(np.pi / 6.0))
    assert all(patch.orientation == expected_orientation for patch in fig.axes[0].patches)
    centers = np.asarray(
        [np.asarray(patch.xy, dtype=np.float64) for patch in fig.axes[0].patches],
        dtype=np.float64,
    )
    assert centers.shape[0] == 7
    deltas = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=2))
    distances = distances[np.triu_indices_from(distances, k=1)]
    nearest = distances[distances > 1.0e-6]
    assert nearest.size > 0
    assert float(np.min(nearest)) == pytest.approx(float(np.sqrt(3.0)), rel=0.02)
    center_index = int(np.argmin(np.sum(centers**2, axis=1)))
    first_ring = np.delete(centers, center_index, axis=0)
    non_digit_texts = [text.get_text() for text in fig.axes[0].texts if not text.get_text().strip().isdigit()]
    assert "F1 reuse cluster" in non_digit_texts
    assert "A cell cluster is replicated over the coverage area." in non_digit_texts
    first_ring_angles = np.mod(np.degrees(np.arctan2(first_ring[:, 1], first_ring[:, 0])), 360.0)
    expected_angles = np.asarray([30.0, 90.0, 150.0, 210.0, 270.0, 330.0], dtype=np.float64)
    assert np.allclose(np.sort(first_ring_angles), expected_angles, atol=8.0)


def test_plot_frequency_reuse_scheme_f12_hides_in_cell_slot_numbers() -> None:
    fig, info = visualise.plot_frequency_reuse_scheme(
        reuse_factor=12,
        return_info=True,
    )

    assert info["reuse_factor"] == 12
    schematic_text = [text.get_text() for text in fig.axes[0].texts]
    assert not any(text.strip().isdigit() for text in schematic_text)


def test_plot_frequency_reuse_scheme_f7_uses_seamless_hex_spacing() -> None:
    fig, info = visualise.plot_frequency_reuse_scheme(
        reuse_factor=7,
        return_info=True,
    )

    assert info["reuse_factor"] == 7
    expected_orientation = pytest.approx(float(np.pi / 6.0))
    assert all(patch.orientation == expected_orientation for patch in fig.axes[0].patches)
    assert len(fig.axes[0].patches) == 49
    slot_labels = [
        text.get_text().strip()
        for text in fig.axes[0].texts
        if text.get_text().strip().isdigit()
    ]
    assert Counter(slot_labels) == {str(index): 7 for index in range(1, 8)}
    centers = np.asarray(
        [np.asarray(patch.xy, dtype=np.float64) for patch in fig.axes[0].patches],
        dtype=np.float64,
    )
    deltas = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=2))
    distances = distances[np.triu_indices_from(distances, k=1)]
    nearest = distances[distances > 1.0e-6]
    assert nearest.size > 0
    assert float(np.min(nearest)) == pytest.approx(float(np.sqrt(3.0)), rel=0.02)
    center_index = int(np.argmin(np.sum(centers**2, axis=1)))
    center = centers[center_index]
    radial_distances = np.sqrt(np.sum((centers - center) ** 2, axis=1))
    first_ring_mask = np.isclose(radial_distances, float(np.sqrt(3.0)), atol=0.08)
    first_ring = centers[first_ring_mask]
    assert first_ring.shape[0] >= 6
    non_digit_texts = [text.get_text() for text in fig.axes[0].texts if not text.get_text().strip().isdigit()]
    assert "F7 reuse cluster" in non_digit_texts
    assert "A cell cluster is replicated over the coverage area." in non_digit_texts
    first_ring_angles = np.mod(
        np.degrees(np.arctan2(first_ring[:, 1] - center[1], first_ring[:, 0] - center[0])),
        360.0,
    )
    expected_angles = np.asarray([30.0, 90.0, 150.0, 210.0, 270.0, 330.0], dtype=np.float64)
    assert np.allclose(np.sort(first_ring_angles)[:6], expected_angles, atol=8.0)
