#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib.util
import tempfile
import typing
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
import matplotlib.axes
from numpy.testing import assert_allclose, assert_equal
import pytest

import scepter.visualise as vis


DB_UNIT = u.dB(u.W)
GRID_INFO_DTYPE = np.dtype(
    [
        ("cell_lon_low", np.float64),
        ("cell_lon_high", np.float64),
        ("cell_lat_low", np.float64),
        ("cell_lat_high", np.float64),
    ]
)


def _small_grid_info() -> np.ndarray:
    """Return a two-cell custom grid for focused plotting tests."""
    return np.array(
        [
            (0.0, 10.0, 5.0, 15.0),
            (10.0, 20.0, 5.0, 15.0),
        ],
        dtype=GRID_INFO_DTYPE,
    )


def _nearest_rank_linear_reference_axis0(
    values_db: u.Quantity,
    percentile: float,
) -> u.Quantity:
    """Return per-column nearest-rank percentile via linear-power reference math."""
    lin = np.asarray(values_db.to_value(u.W), dtype=np.float64)
    out = np.full(lin.shape[1], np.nan, dtype=np.float64)

    for idx in range(lin.shape[1]):
        col = lin[:, idx]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        col.sort()
        rank = int(np.ceil((percentile / 100.0) * col.size)) - 1
        rank = min(max(rank, 0), col.size - 1)
        out[idx] = col[rank]

    return (out * u.W).to(DB_UNIT)


def _nearest_rank_linear_reference_flat(
    values_db: u.Quantity,
    percentile: float,
) -> u.Quantity:
    """Return a nearest-rank percentile after flattening all finite samples."""
    lin = np.asarray(values_db.to_value(u.W), dtype=np.float64).reshape(-1)
    lin = lin[np.isfinite(lin)]
    lin.sort()
    rank = int(np.ceil((percentile / 100.0) * lin.size)) - 1
    rank = min(max(rank, 0), lin.size - 1)
    return (lin[rank] * u.W).to(DB_UNIT)


def _finite_exceedance_reference_axis0(
    values_db: u.Quantity,
    threshold_db: u.Quantity,
) -> np.ndarray:
    """Return per-column finite-sample exceedance percentage via linear power."""
    lin = np.asarray(values_db.to_value(u.W), dtype=np.float64)
    threshold = float(threshold_db.to_value(u.W))
    out = np.full(lin.shape[1], np.nan, dtype=np.float64)

    for idx in range(lin.shape[1]):
        col = lin[:, idx]
        finite = np.isfinite(col)
        if not np.any(finite):
            continue
        out[idx] = 100.0 * np.count_nonzero(col[finite] > threshold) / np.count_nonzero(finite)

    return out


class TestDistributionPlots:

    def setup_method(self):
        self.grid_info = _small_grid_info()
        self.samples_db = np.array(
            [
                [0.0, np.nan],
                [20.0, 30.0],
            ],
            dtype=np.float64,
        ) * DB_UNIT

    def test_plot_cdf_ccdf_matches_linear_domain_reference(self):
        data_db = np.arange(20.0, dtype=np.float64).reshape(10, 2) * DB_UNIT

        _, info = vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            show=False,
            return_values=True,
            show_five_percent=True,
            show_two_percent=True,
            ecdf_method="sort",
        )

        assert_equal(info["ecdf_method_used"], "sort")
        assert_quantity_allclose(info["p95"], _nearest_rank_linear_reference_flat(data_db, 95.0))
        assert_quantity_allclose(info["p98"], _nearest_rank_linear_reference_flat(data_db, 98.0))

    def test_plot_cdf_ccdf_corridor_tail_extends_to_axis_floor(self, monkeypatch):
        data_db = np.array(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [20.0, 20.0],
            ],
            dtype=np.float64,
        ) * DB_UNIT

        fill_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, object]] = []
        original_fill_between = matplotlib.axes.Axes.fill_between

        def _capture_fill_between(self, x, y1, y2=0, *args, **kwargs):
            fill_calls.append(
                (
                    np.asarray(x, dtype=np.float64).copy(),
                    np.asarray(y1, dtype=np.float64).copy(),
                    np.asarray(y2, dtype=np.float64).copy(),
                    kwargs.get("label"),
                )
            )
            return original_fill_between(self, x, y1, y2, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _capture_fill_between)

        vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            show_skycell_corridor=True,
            skycell_corridor_bins=64,
            y_log=True,
            ccdf_ymin_pct=10.0,
            show=False,
        )

        assert_equal(len(fill_calls), 2)

        tail_x, tail_y1, tail_y2, tail_label = fill_calls[-1]
        assert_equal(tail_label, "_nolegend_")
        assert_equal(tail_x.shape, (2,))
        assert_allclose(tail_y1, np.array([0.10, 0.10], dtype=np.float64))
        assert_allclose(tail_y2, np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64))
        assert tail_x[1] > tail_x[0]

    def test_plot_cdf_ccdf_tail_extension_starts_at_last_corridor_support(self, monkeypatch):
        data_db = np.array(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [20.0, 20.0],
            ],
            dtype=np.float64,
        ) * DB_UNIT

        fill_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, object]] = []
        original_fill_between = matplotlib.axes.Axes.fill_between

        def _capture_fill_between(self, x, y1, y2=0, *args, **kwargs):
            fill_calls.append(
                (
                    np.asarray(x, dtype=np.float64).copy(),
                    np.asarray(y1, dtype=np.float64).copy(),
                    np.asarray(y2, dtype=np.float64).copy(),
                    kwargs.get("label"),
                )
            )
            return original_fill_between(self, x, y1, y2, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _capture_fill_between)

        vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            show_skycell_corridor=True,
            skycell_corridor_bins=64,
            y_log=True,
            ccdf_ymin_pct=10.0,
            show=False,
        )

        corridor_x, _, _, corridor_label = fill_calls[0]
        tail_x, _, _, tail_label = fill_calls[-1]

        assert_equal(corridor_label, "Skycell corridor (min-max)")
        assert_equal(tail_label, "_nolegend_")
        assert_allclose(tail_x[0], corridor_x[-1], rtol=0.0, atol=0.0)

    def test_plot_cdf_ccdf_honors_explicit_log_floor_below_empirical_step(self):
        data_db = np.array([[0.0], [10.0], [20.0]], dtype=np.float64) * DB_UNIT

        fig = vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            y_log=True,
            ccdf_ymin_pct=1.0e-9,
            show=False,
        )

        ax = fig.axes[0]
        assert_allclose(ax.get_ylim()[0], 1.0e-11, rtol=1.0e-12, atol=0.0)

    def test_plot_cdf_ccdf_tiny_explicit_log_floor_is_labeled(self):
        data_db = np.array([[0.0], [10.0], [20.0]], dtype=np.float64) * DB_UNIT

        fig = vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            y_log=True,
            ccdf_ymin_pct=1.0e-9,
            show=False,
        )

        ax = fig.axes[0]
        fig.canvas.draw()
        ticks = np.asarray(ax.get_yticks(), dtype=np.float64)

        assert np.any(np.isclose(ticks, 1.0e-11, rtol=1.0e-12, atol=np.finfo(np.float64).tiny))

    def test_plot_cdf_ccdf_tiny_explicit_log_floor_label_text_is_not_zero(self):
        data_db = np.array([[0.0], [10.0], [20.0]], dtype=np.float64) * DB_UNIT

        fig = vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            y_log=True,
            ccdf_ymin_pct=1.0e-7,
            show=False,
        )

        ax = fig.axes[0]
        fig.canvas.draw()
        tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]

        assert "1e-07%" in tick_labels
        assert tick_labels.count("0%") <= 1

    def test_plot_cdf_ccdf_hist_corridor_uses_main_ccdf_support(self, monkeypatch):
        data_db = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float64,
        ) * DB_UNIT

        captured: dict[str, np.ndarray] = {}
        original_step = matplotlib.axes.Axes.step
        original_fill_between = matplotlib.axes.Axes.fill_between

        def _capture_step(self, x, y, *args, **kwargs):
            if kwargs.get("label") == "CCDF":
                captured["step_x"] = np.asarray(x, dtype=np.float64).copy()
            return original_step(self, x, y, *args, **kwargs)

        def _capture_fill_between(self, x, y1, y2=0, *args, **kwargs):
            if kwargs.get("label") == "Skycell corridor (min-max)":
                captured["fill_x"] = np.asarray(x, dtype=np.float64).copy()
            return original_fill_between(self, x, y1, y2, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "step", _capture_step)
        monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _capture_fill_between)

        vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            show_skycell_corridor=True,
            ecdf_method="hist",
            hist_bins=96,
            skycell_corridor_bins=257,
            hist_range=(0.0, 4.0),
            show=False,
        )

        assert_equal(captured["step_x"].shape, captured["fill_x"].shape)
        assert_equal(captured["step_x"].size, 96)
        assert_equal(captured["fill_x"].size, 96)

    def test_plot_cdf_ccdf_legend_outside_corridor_layout_runs(self):
        data_db = np.array(
            [
                [0.0, 1.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ],
            dtype=np.float64,
        ) * DB_UNIT

        fig = vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            show_skycell_corridor=True,
            show_skycell_p98_note=True,
            legend_outside=True,
            y_log=True,
            ccdf_ymin_pct=0.001,
            show=False,
        )

        assert "Empirical distribution" in fig._suptitle.get_text()
        assert_equal(len(fig.axes), 1)
        assert_equal(len(fig.legends), 1)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_right = fig.axes[0].get_position().x1
        legend_bbox = fig.transFigure.inverted().transform_bbox(
            fig.legends[0].get_window_extent(renderer=renderer)
        )
        gap = float(legend_bbox.x0 - ax_right)
        assert gap > 0.0
        assert gap < 0.04

    def test_plot_cdf_ccdf_legend_outside_single_axis_gap_is_tight(self):
        data_db = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float64) * DB_UNIT

        fig = vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            legend_outside=True,
            y_log=True,
            ccdf_ymin_pct=0.001,
            prot_value=[15.0],
            prot_legend=["Reference"],
            show=False,
        )

        assert_equal(len(fig.axes), 1)
        assert_equal(len(fig.legends), 1)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_right = fig.axes[0].get_position().x1
        legend_bbox = fig.transFigure.inverted().transform_bbox(
            fig.legends[0].get_window_extent(renderer=renderer)
        )
        gap = float(legend_bbox.x0 - ax_right)
        assert gap > 0.0
        assert gap < 0.04

    def test_plot_cdf_ccdf_from_histogram_matches_hist_mode_percentiles(self):
        data_db = np.linspace(-25.0, 35.0, 48, dtype=np.float64).reshape(16, 3) * DB_UNIT
        hist_range = (-30.0, 40.0)
        hist_bins = 2800
        counts, edges = np.histogram(
            np.asarray(data_db.to_value(DB_UNIT), dtype=np.float64).ravel(),
            bins=hist_bins,
            range=hist_range,
        )

        _, raw_info = vis.plot_cdf_ccdf(
            data_db,
            cell_axis=-1,
            plot_type="ccdf",
            show=False,
            return_values=True,
            show_five_percent=True,
            show_two_percent=True,
            ecdf_method="hist",
            hist_bins=hist_bins,
            hist_range=hist_range,
            assume_finite=True,
        )
        _, hist_info = vis.plot_cdf_ccdf_from_histogram(
            counts,
            edges=edges * DB_UNIT,
            plot_type="ccdf",
            show=False,
            return_values=True,
            show_five_percent=True,
            show_two_percent=True,
        )

        half_bin = 0.5 * float(edges[1] - edges[0])
        assert_equal(hist_info["ecdf_method_used"], "prebinned_hist")
        assert abs(hist_info["p95"].to_value(DB_UNIT) - raw_info["p95"].to_value(DB_UNIT)) <= half_bin
        assert abs(hist_info["p98"].to_value(DB_UNIT) - raw_info["p98"].to_value(DB_UNIT)) <= half_bin

    def test_plot_cdf_ccdf_from_histogram_draws_two_percent_and_protection_lines(self):
        counts = np.array([5, 10, 20, 8, 2], dtype=np.int64)
        edges = np.array([-150.0, -140.0, -130.0, -120.0, -110.0, -100.0], dtype=np.float64)

        fig, info = vis.plot_cdf_ccdf_from_histogram(
            counts,
            edges=edges,
            plot_type="ccdf",
            show_two_percent=True,
            prot_value=[-118.0],
            prot_legend=["Reference"],
            show=False,
            return_values=True,
        )

        ax = fig.axes[0]
        labels = ax.get_legend_handles_labels()[1]
        annotation_text = f"{float(info['p98']):.3f}"

        assert "CCDF" in labels
        assert "2%" in labels
        assert "Reference" in labels
        assert any(text.get_text() == annotation_text for text in ax.texts)

    def test_plot_cdf_ccdf_from_histogram_honors_explicit_log_floor_label(self):
        counts = np.array([1, 1, 1], dtype=np.int64)
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

        fig = vis.plot_cdf_ccdf_from_histogram(
            counts,
            edges=edges,
            plot_type="ccdf",
            y_log=True,
            ccdf_ymin_pct=1.0e-7,
            show=False,
        )

        ax = fig.axes[0]
        fig.canvas.draw()
        tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]

        assert "1e-07%" in tick_labels

    def test_plot_cdf_ccdf_from_histogram_warns_when_corridor_is_requested(self):
        counts = np.array([2, 3, 4], dtype=np.int64)
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

        with pytest.warns(UserWarning, match="show_skycell_corridor"):
            vis.plot_cdf_ccdf_from_histogram(
                counts,
                edges=edges,
                plot_type="ccdf",
                show_skycell_corridor=True,
                show=False,
            )

    def test_plot_hemisphere_2d_power_matches_linear_reference(self):
        _, values = vis.plot_hemisphere_2D(
            self.samples_db,
            grid_info=self.grid_info,
            mode="power",
            worst_percent=25.0,
            cell_axis=-1,
            projection="rect",
            engine="mpl",
            colorbar=False,
            draw_cell_borders=False,
            show=False,
            return_values=True,
        )

        expected = _nearest_rank_linear_reference_axis0(self.samples_db, 75.0)
        assert_quantity_allclose(values, expected)

    def test_plot_hemisphere_2d_data_loss_ignores_non_finite_samples(self):
        threshold = 10.0 * DB_UNIT

        _, values = vis.plot_hemisphere_2D(
            self.samples_db,
            grid_info=self.grid_info,
            mode="data_loss",
            protection_criterion=threshold,
            cell_axis=-1,
            projection="rect",
            engine="mpl",
            colorbar=False,
            draw_cell_borders=False,
            show=False,
            return_values=True,
        )

        assert_allclose(values, _finite_exceedance_reference_axis0(self.samples_db, threshold))

    def test_plot_hemisphere_3d_power_matches_linear_reference(self):
        _, values = vis.plot_hemisphere_3D(
            self.samples_db,
            grid_info=self.grid_info,
            mode="power",
            worst_percent=25.0,
            cell_axis=-1,
            engine="mpl",
            colorbar=False,
            draw_guides=False,
            draw_cell_borders=False,
            show=False,
            return_values=True,
        )

        expected = _nearest_rank_linear_reference_axis0(self.samples_db, 75.0)
        assert_quantity_allclose(values, expected)

    def test_plot_hemisphere_3d_data_loss_ignores_non_finite_samples(self):
        threshold = 10.0 * DB_UNIT

        _, values = vis.plot_hemisphere_3D(
            self.samples_db,
            grid_info=self.grid_info,
            mode="data_loss",
            protection_criterion=threshold,
            cell_axis=-1,
            engine="mpl",
            colorbar=False,
            draw_guides=False,
            draw_cell_borders=False,
            show=False,
            return_values=True,
        )

        assert_allclose(values, _finite_exceedance_reference_axis0(self.samples_db, threshold))

    def test_db_arithmetic_mean_differs_from_linear_power_mean(self):
        samples = np.array([0.0, 20.0], dtype=np.float64) * DB_UNIT
        arithmetic_db_mean = np.mean(samples.value)
        linear_power_mean_db = (np.mean(samples.to_value(u.W)) * u.W).to_value(DB_UNIT)

        assert_allclose(arithmetic_db_mean, 10.0)
        assert_allclose(linear_power_mean_db, 17.03291378, rtol=1e-9)
        assert arithmetic_db_mean != pytest.approx(linear_power_mean_db)


class TestPlotly3D:

    def setup_method(self):
        self.grid_info = _small_grid_info()
        self.samples_db = np.array(
            [
                [0.0, 5.0],
                [20.0, 30.0],
            ],
            dtype=np.float64,
        ) * DB_UNIT

    def test_plot_hemisphere_3d_export_modes(self, monkeypatch):
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        calls: list[tuple[object, object, object]] = []

        def _fake_write_html(self, file, include_plotlyjs=True, full_html=True, **kwargs):
            calls.append((file, include_plotlyjs, full_html))

        monkeypatch.setattr(go.Figure, "write_html", _fake_write_html)

        vis.plot_hemisphere_3D(
            self.samples_db,
            grid_info=self.grid_info,
            mode="power",
            worst_percent=25.0,
            cell_axis=-1,
            engine="plotly",
            colorbar=False,
            draw_guides=False,
            draw_cell_borders=False,
            show=False,
            export_html_path="bundled.html",
            html_include_plotlyjs=True,
        )
        assert_equal(calls[-1][1], True)
        assert_equal(calls[-1][2], True)

        vis.plot_hemisphere_3D(
            self.samples_db,
            grid_info=self.grid_info,
            mode="power",
            worst_percent=25.0,
            cell_axis=-1,
            engine="plotly",
            colorbar=False,
            draw_guides=False,
            draw_cell_borders=False,
            show=False,
            export_html_path="cdn.html",
            html_include_plotlyjs="cdn",
        )
        assert_equal(calls[-1][1], "cdn")
        assert_equal(calls[-1][2], True)

        with pytest.raises(ValueError, match="html_include_plotlyjs=False"):
            vis.plot_hemisphere_3D(
                self.samples_db,
                grid_info=self.grid_info,
                mode="power",
                worst_percent=25.0,
                cell_axis=-1,
                engine="plotly",
                colorbar=False,
                draw_guides=False,
                draw_cell_borders=False,
                show=False,
                export_html_path="invalid.html",
                html_include_plotlyjs=False,
            )

    @pytest.mark.parametrize(
        ("hover_mode", "expected_trace_count"),
        [
            ("rich", 3),
            ("single_trace", 2),
            ("none", 1),
        ],
    )
    def test_plot_hemisphere_3d_hover_modes(self, hover_mode, expected_trace_count):
        pytest.importorskip("plotly")

        fig = vis.plot_hemisphere_3D(
            self.samples_db,
            grid_info=self.grid_info,
            mode="power",
            worst_percent=25.0,
            cell_axis=-1,
            engine="plotly",
            colorbar=False,
            draw_guides=False,
            draw_cell_borders=False,
            show=False,
            plotly_hover_mode=hover_mode,
        )

        assert_equal(len(fig.data), expected_trace_count)
        assert_equal(fig.data[0].type, "mesh3d")


class TestSatelliteElevationPfdHeatmap:

    def test_heatmap_returns_info_and_axis_labels(self):
        histogram = np.array(
            [
                [0, 3, 0],
                [2, 4, 1],
            ],
            dtype=np.int64,
        )
        elev_edges = np.array([0.0, 30.0, 60.0], dtype=np.float64)
        pfd_edges = np.array([-180.0, -170.0, -160.0, -150.0], dtype=np.float64)

        fig, info = vis.plot_satellite_elevation_pfd_heatmap(
            histogram,
            elevation_edges_deg=elev_edges,
            pfd_edges_db=pfd_edges,
            show=False,
            return_values=True,
        )

        ax = fig.axes[0]
        assert_equal(ax.get_xlabel(), "Satellite elevation at RAS station [deg]")
        assert_equal(ax.get_ylabel(), "Instantaneous PFD contribution [dBW/m^2/MHz]")
        assert_equal(info["sample_count"], int(np.sum(histogram)))
        assert_equal(info["positive_bin_count"], 4)

    def test_heatmap_rejects_edge_length_mismatch(self):
        histogram = np.ones((2, 3), dtype=np.int64)

        with pytest.raises(ValueError, match="N_elevation_bins"):
            vis.plot_satellite_elevation_pfd_heatmap(
                histogram,
                elevation_edges_deg=np.array([0.0, 30.0], dtype=np.float64),
                pfd_edges_db=np.array([-180.0, -170.0, -160.0, -150.0], dtype=np.float64),
                show=False,
            )

    def test_heatmap_honors_grid_and_colormap_controls(self):
        histogram = np.array([[1, 2], [3, 4]], dtype=np.int64)

        fig, _ = vis.plot_satellite_elevation_pfd_heatmap(
            histogram,
            elevation_edges_deg=np.array([0.0, 20.0, 40.0], dtype=np.float64),
            pfd_edges_db=np.array([-150.0, -140.0, -130.0], dtype=np.float64),
            cmap="plasma",
            grid=True,
            grid_color="#112233",
            grid_alpha=0.55,
            grid_linewidth=1.25,
            show=False,
            return_values=True,
        )

        ax = fig.axes[0]
        quad_mesh = ax.collections[0]
        gridlines = ax.xaxis.get_gridlines()

        assert_equal(quad_mesh.get_cmap().name, "plasma")
        assert gridlines
        assert_equal(gridlines[0].get_color(), "#112233")
        assert_allclose(gridlines[0].get_alpha(), 0.55, atol=0.0, rtol=0.0)
        assert_allclose(float(gridlines[0].get_linewidth()), 1.25, atol=0.0, rtol=0.0)


class TestPipelineHelpers:

    def test_callable_annotations_resolve(self):
        hints = typing.get_type_hints(vis._build_sky_mapper)
        backend_hints = typing.get_type_hints(vis._choose_sky_mapper_backend)

        assert "return" in hints
        assert "return" in backend_hints

    def test_average_counts_are_arithmetic_means(self, monkeypatch):
        azimuth = np.array(
            [
                [0.0, 10.0, 20.0],
                [0.0, 10.0, 200.0],
            ],
            dtype=np.float32,
        )
        elevation = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 40.0, 10.0],
            ],
            dtype=np.float32,
        )

        class _FakeDataset:

            def __init__(self, data):
                self._data = np.asarray(data)
                self.shape = self._data.shape
                self.ndim = self._data.ndim

            def __getitem__(self, item):
                return self._data[item]

        class _FakeGroup(dict):
            """Minimal dict-backed h5py group stand-in for pipeline tests."""

        class _FakeH5File(dict):

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_h5 = _FakeH5File(
            {
                "iter": _FakeGroup(
                    {
                        "iter_0000": _FakeGroup(
                            {
                                "sat_azimuth": _FakeDataset(azimuth),
                                "sat_elevation": _FakeDataset(elevation),
                            }
                        )
                    }
                )
            }
        )

        n_skycells, mapper = vis._build_sky_mapper("s1586", n_cells=2334)
        slot_counts = []
        for slot_az, slot_el in zip(azimuth, elevation):
            sky_ids = mapper(slot_az, slot_el)
            sky_ids = sky_ids[(sky_ids >= 0) & (sky_ids < n_skycells)]
            slot_counts.append(
                np.bincount(sky_ids.astype(np.int64, copy=False), minlength=n_skycells).astype(np.float64)
            )
        expected_average = np.mean(np.stack(slot_counts, axis=0), axis=0)

        saved_npz: dict[str, object] = {}
        fake_input = "satellite_counts.h5"

        def _fake_exists(self):
            return str(self) == fake_input

        def _fake_h5_file(filename, mode):
            assert_equal(str(filename), fake_input)
            assert_equal(mode, "r")
            return fake_h5

        def _fake_savez_compressed(path, **kwargs):
            saved_npz["path"] = path
            saved_npz["payload"] = kwargs

        monkeypatch.setattr(vis, "plot_hemisphere_2D", None)
        monkeypatch.setattr(vis.Path, "exists", _fake_exists)
        monkeypatch.setattr(vis.h5py, "File", _fake_h5_file)
        monkeypatch.setattr(vis, "_make_output_run_dir", lambda base_dir, prefix: vis.Path("dummy_run_dir"))
        monkeypatch.setattr(vis.np, "savez_compressed", _fake_savez_compressed)

        result = vis.satellite_distribution_over_sky(
            fake_input,
            config=vis.SatelliteDistributionConfig(
                enable_random_plot_samples=False,
                enable_average_plot=True,
                enable_animation_random_set=False,
                enable_animation_subsequent_set=False,
                random_timestep_plots=0,
                save_outputs=True,
                output_dir="unused_output_dir",
                output_prefix="visualise_test",
                save_plots_html=False,
                save_plots_png=False,
                save_plots_jpg=False,
                save_results_json=False,
                save_results_npz=True,
                show_plots=False,
                save_animation=False,
                average_slot_stride=1,
                average_max_slots=0,
                max_raw_slots_to_read=0,
                deduplicate_by_az_el=False,
                progress_every_slots=1000,
            ),
        )

        assert "payload" in saved_npz
        payload = typing.cast(dict[str, np.ndarray], saved_npz["payload"])
        assert_allclose(payload["average_counts"], expected_average)
        assert_equal(payload["average_slots_used"], np.array([2], dtype=np.int64))
        assert result["results_npz_path"].endswith("results_curves.npz")


def test_resolve_animation_backend_auto_prefers_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(vis, "_probe_visualise_cuda_support", lambda: (False, "no cuda"))

    backend, reason, direct = vis._resolve_animation_render_backend(
        requested_backend="auto",
        skycell_mode="s1586",
        n_skycells=2334,
        vis_projection="polar",
    )

    assert_equal(backend, "cpu_raster")
    assert_equal(reason, "auto_cpu_raster")
    assert_equal(direct, True)


@pytest.mark.parametrize(
    "removed_key",
    [
        "animation_frame_progress_use_tqdm",
        "animation_frame_progress_every",
    ],
)
def test_resolve_config_rejects_removed_progress_overrides(removed_key: str):
    with pytest.raises(TypeError, match=removed_key):
        vis._resolve_config(None, {removed_key: True})


def test_resolve_animation_backend_gpu_falls_back_with_warning(monkeypatch):
    monkeypatch.setattr(vis, "_probe_visualise_cuda_support", lambda: (False, "driver unavailable"))

    with pytest.warns(RuntimeWarning, match="gpu_raster"):
        backend, reason, direct = vis._resolve_animation_render_backend(
            requested_backend="gpu_raster",
            skycell_mode="s1586",
            n_skycells=2334,
            vis_projection="polar",
        )

    assert_equal(backend, "cpu_raster")
    assert_equal(reason, "gpu_raster_unavailable_fallback_cpu")
    assert_equal(direct, True)


def test_resolve_animation_backend_unsupported_geometry_falls_back_to_plotly(monkeypatch):
    monkeypatch.setattr(vis, "_probe_visualise_cuda_support", lambda: (True, "cuda ok"))

    with pytest.warns(RuntimeWarning, match="full-grid S.1586 polar"):
        backend, reason, direct = vis._resolve_animation_render_backend(
            requested_backend="auto",
            skycell_mode="s1586",
            n_skycells=100,
            vis_projection="rect",
        )

    assert_equal(backend, "plotly")
    assert_equal(reason, "fast_path_unsupported_geometry")
    assert_equal(direct, False)


def test_render_animation_frames_fast_skips_frame_dirs_when_not_requested(monkeypatch):
    samples = [
        vis.RandomSlotSample(
            iter_name="iter_0000",
            slot_local_idx=0,
            slot_global_idx=0,
            time_mjd=60000.0,
            counts=np.zeros(2334, dtype=np.int32),
            n_satellites=0,
            n_active_skycells=0,
        ),
        vis.RandomSlotSample(
            iter_name="iter_0000",
            slot_local_idx=1,
            slot_global_idx=1,
            time_mjd=60000.1,
            counts=np.ones(2334, dtype=np.int32),
            n_satellites=1,
            n_active_skycells=2334,
        ),
    ]

    monkeypatch.setattr(vis.shutil, "which", lambda name: None)
    monkeypatch.setattr(vis, "_save_animation_gif_from_arrays", lambda **kwargs: True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = vis.Path(tmp_dir)
        frame_count, mp4_outputs, gif_outputs = vis._render_animation_frames_fast(
            set_key="random",
            samples=samples,
            animations_dir=tmp_path,
            title_prefix="Random",
            fps_values=[5],
            vmax=4.0,
            vis_cmap="turbo",
            frame_width=320,
            frame_height=240,
            frame_progress=False,
            render_backend="cpu_raster",
            keep_frame_pngs=False,
            save_animation_mp4=False,
            save_animation_gif=True,
            ffmpeg_preset="veryfast",
            ffmpeg_crf=24,
            ffmpeg_threads=0,
            ffmpeg_loglevel="error",
        )

        assert_equal(frame_count, 2)
        assert_equal(mp4_outputs, [])
        assert_equal(len(gif_outputs), 1)
        assert not (tmp_path / "random_frames_002").exists()


def test_render_animation_frames_fast_keeps_pngs_without_plotly(monkeypatch):
    samples = [
        vis.RandomSlotSample(
            iter_name="iter_0000",
            slot_local_idx=0,
            slot_global_idx=0,
            time_mjd=60000.0,
            counts=np.full(2334, 2, dtype=np.int32),
            n_satellites=2,
            n_active_skycells=2334,
        )
    ]
    saved_paths: list[vis.Path] = []

    monkeypatch.setattr(vis.shutil, "which", lambda name: None)
    monkeypatch.setattr(vis, "_save_animation_gif_from_arrays", lambda **kwargs: True)
    monkeypatch.setattr(vis, "_render_animation_png_frames", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy Plotly frame renderer should not be used for cpu_raster")
    ))

    def _fake_save_animation_frame_png(frame_rgba, output_path):
        saved_paths.append(output_path)
        return True

    monkeypatch.setattr(vis, "_save_animation_frame_png", _fake_save_animation_frame_png)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = vis.Path(tmp_dir)
        frame_count, _, gif_outputs = vis._render_animation_frames_fast(
            set_key="subsequent",
            samples=samples,
            animations_dir=tmp_path,
            title_prefix="Subsequent",
            fps_values=[2],
            vmax=4.0,
            vis_cmap="turbo",
            frame_width=320,
            frame_height=240,
            frame_progress=False,
            render_backend="cpu_raster",
            keep_frame_pngs=True,
            save_animation_mp4=False,
            save_animation_gif=True,
            ffmpeg_preset="veryfast",
            ffmpeg_crf=24,
            ffmpeg_threads=0,
            ffmpeg_loglevel="error",
        )

        assert_equal(frame_count, 1)
        assert_equal(len(saved_paths), 1)
        assert_equal(saved_paths[0].name, "frame_001.png")
        assert (tmp_path / "subsequent_frames_001").exists()
        assert_equal(len(gif_outputs), 1)


def test_satellite_distribution_reports_fast_animation_backend_diagnostics(monkeypatch):
    azimuth = np.array([[0.0, 10.0]], dtype=np.float32)
    elevation = np.array([[10.0, 20.0]], dtype=np.float32)

    class _FakeDataset:

        def __init__(self, data):
            self._data = np.asarray(data)
            self.shape = self._data.shape
            self.ndim = self._data.ndim

        def __getitem__(self, item):
            return self._data[item]

    class _FakeGroup(dict):
        """Minimal dict-backed h5py group stand-in for pipeline tests."""

    class _FakeH5File(dict):

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_h5 = _FakeH5File(
        {
            "iter": _FakeGroup(
                {
                    "iter_0000": _FakeGroup(
                        {
                            "sat_azimuth": _FakeDataset(azimuth),
                            "sat_elevation": _FakeDataset(elevation),
                        }
                    )
                }
            )
        }
    )

    fake_input = "satellite_counts_anim.h5"
    export_calls: list[dict[str, object]] = []

    def _fake_exists(self):
        return str(self) == fake_input

    def _fake_h5_file(filename, mode):
        assert_equal(str(filename), fake_input)
        assert_equal(mode, "r")
        return fake_h5

    def _fake_export_animation_set(**kwargs):
        export_calls.append(dict(kwargs))
        return 1, ["dummy.mp4"], []

    monkeypatch.setattr(vis.Path, "exists", _fake_exists)
    monkeypatch.setattr(vis.h5py, "File", _fake_h5_file)
    monkeypatch.setattr(vis, "_make_output_run_dir", lambda base_dir, prefix: vis.Path("dummy_run_dir"))
    monkeypatch.setattr(vis, "_export_animation_set", _fake_export_animation_set)
    monkeypatch.setattr(vis, "plot_hemisphere_2D", None)
    monkeypatch.setattr(vis, "_probe_visualise_cuda_support", lambda: (False, "no cuda"))

    result = vis.satellite_distribution_over_sky(
        fake_input,
        config=vis.SatelliteDistributionConfig(
            enable_random_plot_samples=False,
            enable_average_plot=False,
            enable_animation_random_set=True,
            enable_animation_subsequent_set=False,
            random_timestep_plots=0,
            save_outputs=True,
            output_dir="unused_output_dir",
            output_prefix="visualise_test",
            save_plots_html=False,
            save_plots_png=False,
            save_plots_jpg=False,
            save_results_json=False,
            save_results_npz=False,
            show_plots=False,
            save_animation=True,
            save_animation_mp4=True,
            save_animation_gif=False,
            animation_random_sample_count=1,
            animation_subsequent_sample_count=0,
            animation_render_backend="auto",
            keep_animation_frame_pngs=False,
            max_raw_slots_to_read=0,
            deduplicate_by_az_el=False,
            progress_every_slots=1000,
        ),
    )

    assert_equal(len(export_calls), 1)
    assert_equal(export_calls[0]["effective_backend"], "cpu_raster")
    assert_equal(result["animation_render_backend_requested"], "auto")
    assert_equal(result["animation_render_backend_effective"], "cpu_raster")
    assert_equal(result["animation_render_backend_reason"], "auto_cpu_raster")
    assert_equal(result["animation_used_direct_streaming"], True)
    assert_equal(result["animation_kept_frame_pngs"], False)
    assert_equal(result["animation_random_frame_count"], 1)
    assert_equal(result["animation_mp4_outputs"], ["dummy.mp4"])


def test_cpu_and_gpu_s1586_animation_rasters_match():
    cuda_ok, _ = vis._probe_visualise_cuda_support()
    if not cuda_ok:
        pytest.skip("CuPy/CUDA unavailable for GPU raster equivalence test.")

    _, _, disc_mask, cell_idx = vis._s1586_polar_heatmap_lookup(320, "equal_area", False)
    counts = np.arange(2334, dtype=np.float32) % 17

    cpu_raster = vis._rasterize_s1586_counts_cpu(
        counts,
        disc_mask=disc_mask,
        cell_idx=cell_idx,
    )
    gpu_state = vis._S1586GpuRasterState.from_lookup(
        disc_mask=disc_mask,
        cell_idx=cell_idx,
    )
    gpu_raster = vis._rasterize_s1586_counts_gpu(
        counts,
        gpu_state=gpu_state,
    )

    assert_allclose(cpu_raster, gpu_raster, rtol=0.0, atol=0.0)


class TestPlotCellStatusMap:

    def test_groups_are_mutually_exclusive_and_backend_auto_falls_back(self):
        expected_backend = (
            "cartopy" if importlib.util.find_spec("cartopy") is not None else "matplotlib"
        )
        pre_lon = np.array([21.4, 21.5, 21.6], dtype=np.float64) * u.deg
        pre_lat = np.array([-30.7, -30.8, -30.9], dtype=np.float64) * u.deg
        active_lon = np.array([21.5, 21.6], dtype=np.float64) * u.deg
        active_lat = np.array([-30.8, -30.9], dtype=np.float64) * u.deg

        fig, info = vis.plot_cell_status_map(
            pre_lon,
            pre_lat,
            active_cell_longitudes=active_lon,
            active_cell_latitudes=active_lat,
            switched_off_mask=np.array([True, False, False], dtype=bool),
            boresight_affected_cell_ids=np.array([1], dtype=np.int32),
            ras_longitude=21.4436 * u.deg,
            ras_latitude=-30.7128 * u.deg,
            backend="auto",
            return_info=True,
        )

        assert_equal(info["backend_used"], expected_backend)
        assert_equal(info["extent_mode_used"], "auto")
        assert_equal(info["switched_off_count"], 1)
        assert_equal(info["normal_active_count"], 1)
        assert_equal(info["boresight_affected_active_count"], 1)
        assert_equal(fig.axes[0].get_title(), "Cell Status Map")
        assert fig.axes

    def test_radius_override_filters_points_and_geography_cut_cells_stay_unrendered(self):
        pre_lon = np.array([21.4436, 21.50], dtype=np.float64) * u.deg
        pre_lat = np.array([-30.7128, -30.75], dtype=np.float64) * u.deg
        active_lon = np.array([21.50, 22.20], dtype=np.float64) * u.deg
        active_lat = np.array([-30.75, -31.80], dtype=np.float64) * u.deg

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = f"{tmpdir}/cell_status.png"
            fig, info = vis.plot_cell_status_map(
                pre_lon,
                pre_lat,
                active_cell_longitudes=active_lon,
                active_cell_latitudes=active_lat,
                switched_off_mask=np.array([True, False], dtype=bool),
                boresight_affected_cell_ids=np.array([1], dtype=np.int32),
                ras_longitude=21.4436 * u.deg,
                ras_latitude=-30.7128 * u.deg,
                backend="matplotlib",
                radius_km=30.0,
                save_path=save_path,
                return_info=True,
            )

            assert_equal(info["extent_mode_used"], "radius_km")
            assert_equal(info["switched_off_count"], 1)
            assert_equal(info["normal_active_count"], 1)
            assert_equal(info["boresight_affected_active_count"], 0)
            assert_equal(fig.axes[0].collections[0].get_offsets().shape[0], 1)
            assert np.isfinite(np.asarray(info["extent"], dtype=np.float64)).all()
            assert fig.axes
            assert Path(save_path).exists()

    def test_explicit_cartopy_backend_path_is_clear(self):
        kwargs = dict(
            pre_ras_cell_longitudes=np.array([21.4], dtype=np.float64) * u.deg,
            pre_ras_cell_latitudes=np.array([-30.7], dtype=np.float64) * u.deg,
            active_cell_longitudes=np.array([21.4], dtype=np.float64) * u.deg,
            active_cell_latitudes=np.array([-30.7], dtype=np.float64) * u.deg,
            switched_off_mask=np.array([False], dtype=bool),
            boresight_affected_cell_ids=None,
            ras_longitude=21.4436 * u.deg,
            ras_latitude=-30.7128 * u.deg,
            backend="cartopy",
        )

        if importlib.util.find_spec("cartopy") is None:
            with pytest.raises(RuntimeError, match="cartopy is required"):
                vis.plot_cell_status_map(**kwargs)
            return

        fig, info = vis.plot_cell_status_map(return_info=True, **kwargs)
        assert_equal(info["backend_used"], "cartopy")
        assert fig.axes

    @pytest.mark.parametrize("map_style", ["clean", "terrain", "relief"])
    def test_map_style_options_are_reported(self, map_style):
        fig, info = vis.plot_cell_status_map(
            np.array([21.4], dtype=np.float64) * u.deg,
            np.array([-30.7], dtype=np.float64) * u.deg,
            active_cell_longitudes=np.array([21.4], dtype=np.float64) * u.deg,
            active_cell_latitudes=np.array([-30.7], dtype=np.float64) * u.deg,
            switched_off_mask=np.array([False], dtype=bool),
            boresight_affected_cell_ids=None,
            ras_longitude=21.4436 * u.deg,
            ras_latitude=-30.7128 * u.deg,
            backend="matplotlib",
            map_style=map_style,
            return_info=True,
        )
        assert_equal(info["map_style_used"], map_style)
        assert fig.axes
