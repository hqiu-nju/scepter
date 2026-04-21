"""Tests for Stage 21 custom-antenna-pattern preview figures.

The figure factory is pure matplotlib — runs headless without Qt.
These tests confirm the factory dispatches correctly, produces
well-formed figures for each (kind, grid_mode) combination, and the
embedded gains match the runtime evaluators (so the preview shows
exactly what the GPU path will consume).
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.figure  # noqa: E402

from scepter.custom_antenna import (
    CustomAntennaPattern,
    evaluate_pattern_1d,
    evaluate_pattern_2d,
)
from scepter.custom_antenna_preview import build_custom_pattern_preview_figure


def _pattern_1d() -> CustomAntennaPattern:
    return CustomAntennaPattern.from_json_dict({
        "scepter_antenna_pattern_format": "v1",
        "kind": "1d_axisymmetric",
        "normalisation": "absolute",
        "peak_gain_source": "explicit",
        "peak_gain_dbi": 38.0,
        "grid_deg": [0.0, 1.0, 3.0, 10.0, 30.0, 90.0, 180.0],
        "gain_db": [38.0, 32.0, 18.0, -5.0, -20.0, -35.0, -40.0],
    })


def _pattern_2d_az_el() -> CustomAntennaPattern:
    return CustomAntennaPattern.from_json_dict({
        "scepter_antenna_pattern_format": "v1",
        "kind": "2d",
        "grid_mode": "az_el",
        "normalisation": "absolute",
        "peak_gain_source": "explicit",
        "peak_gain_dbi": 34.5,
        "az_wraps": True,
        "az_grid_deg": [-180.0, -90.0, -30.0, 0.0, 30.0, 90.0, 180.0],
        "el_grid_deg": [-90.0, -30.0, 0.0, 30.0, 90.0],
        "gain_db": [
            [-40.0, -28.0, -14.0, -28.0, -40.0],
            [-28.0, -15.0,   2.0, -15.0, -28.0],
            [-18.0,  -3.0,  24.0,  -3.0, -18.0],
            [-14.0,   2.0,  34.5,   2.0, -14.0],
            [-18.0,  -3.0,  24.0,  -3.0, -18.0],
            [-28.0, -15.0,   2.0, -15.0, -28.0],
            [-40.0, -28.0, -14.0, -28.0, -40.0],
        ],
    })


def _pattern_2d_theta_phi() -> CustomAntennaPattern:
    return CustomAntennaPattern.from_json_dict({
        "scepter_antenna_pattern_format": "v1",
        "kind": "2d",
        "grid_mode": "theta_phi",
        "normalisation": "relative",
        "peak_gain_source": "explicit",
        "peak_gain_dbi": 34.5,
        "phi_wraps": True,
        "theta_grid_deg": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 90.0, 180.0],
        "phi_grid_deg":   [-180.0, -90.0, 0.0, 90.0, 180.0],
        "gain_db": [
            [   0.0,    0.0,    0.0,    0.0,    0.0],
            [  -1.0,   -2.5,   -1.0,   -2.5,   -1.0],
            [  -4.0,  -10.0,   -4.0,  -10.0,   -4.0],
            [ -15.0,  -25.0,  -15.0,  -25.0,  -15.0],
            [ -25.0,  -35.0,  -25.0,  -35.0,  -25.0],
            [ -40.0,  -50.0,  -40.0,  -50.0,  -40.0],
            [ -55.0,  -60.0,  -55.0,  -60.0,  -55.0],
            [ -70.0,  -75.0,  -70.0,  -75.0,  -70.0],
        ],
    })


def test_build_preview_dispatches_on_kind_and_mode():
    fig_1d = build_custom_pattern_preview_figure(_pattern_1d())
    fig_2d_azel = build_custom_pattern_preview_figure(_pattern_2d_az_el())
    fig_2d_tp = build_custom_pattern_preview_figure(_pattern_2d_theta_phi())

    assert isinstance(fig_1d, matplotlib.figure.Figure)
    assert isinstance(fig_2d_azel, matplotlib.figure.Figure)
    assert isinstance(fig_2d_tp, matplotlib.figure.Figure)

    # 1-D figure → one cartesian axes by default. Cartesian matches
    # the 1-D editor's default view and keeps far-sidelobe / plateau
    # structure visible instead of clipping below the polar radial
    # floor.
    axes_1d = fig_1d.get_axes()
    assert len(axes_1d) == 1
    assert axes_1d[0].name == "rectilinear"

    # Polar projection is opt-in.
    fig_1d_polar = build_custom_pattern_preview_figure(
        _pattern_1d(), projection="polar",
    )
    polar_axes = fig_1d_polar.get_axes()
    assert len(polar_axes) == 1
    assert polar_axes[0].name == "polar"

    # 2-D figures → two cartesian axes (heatmap + cut panel) plus the
    # colorbar axes auto-added by ``fig.colorbar(...)``.
    for fig2d in (fig_2d_azel, fig_2d_tp):
        axes = fig2d.get_axes()
        assert len(axes) >= 2
        # None of the 2-D axes are polar.
        assert all(ax.name != "polar" for ax in axes)


def test_preview_embeds_absolute_dbi_matching_runtime_evaluator():
    """The 1-D plot's y-values must equal ``evaluate_pattern_1d``
    output — the preview shows exactly what the GPU path will see
    (no hidden renormalisation / clamping). Applies to both
    cartesian (default) and polar projections.
    """
    pat = _pattern_1d()
    # Cartesian default — single line, y-values are absolute dBi.
    fig = build_custom_pattern_preview_figure(pat)
    (ax,) = fig.get_axes()
    lines = ax.get_lines()
    assert len(lines) == 1
    gain_from_plot = np.asarray(lines[0].get_ydata())
    gain_reference = evaluate_pattern_1d(pat, np.asarray(pat.grid_deg))
    np.testing.assert_allclose(gain_from_plot, gain_reference, rtol=0, atol=1e-9)

    # Polar — mirror pair across boresight, each carrying the same
    # radial (absolute dBi) values.
    fig_polar = build_custom_pattern_preview_figure(pat, projection="polar")
    (ax_polar,) = fig_polar.get_axes()
    polar_lines = ax_polar.get_lines()
    assert len(polar_lines) == 2
    polar_gain = np.asarray(polar_lines[0].get_ydata())
    np.testing.assert_allclose(polar_gain, gain_reference, rtol=0, atol=1e-9)


def _find_cut_axes(fig):
    """Return the principal-plane cut axes (the one titled 'Principal-plane cuts')."""
    return [ax for ax in fig.get_axes()
            if "Principal" in ax.get_title()]


def test_preview_2d_relative_pattern_renders_in_absolute_dbi():
    """A relative-mode 2-D pattern must be shifted to absolute dBi
    before plotting so the colorbar range and cut gains line up with
    ``peak_gain_dbi``.
    """
    pat = _pattern_2d_theta_phi()
    fig = build_custom_pattern_preview_figure(pat)
    cut_axes = _find_cut_axes(fig)
    assert len(cut_axes) == 1
    lines = cut_axes[0].get_lines()
    # Two principal-plane cuts.
    assert len(lines) == 2

    # At θ=0, both cuts pass through boresight gain → must equal
    # peak_gain_dbi (34.5 in the fixture).
    for line in lines:
        gain_vs_theta = np.asarray(line.get_ydata())
        # The first sample is at θ=0 for theta_phi mode.
        assert abs(float(gain_vs_theta[0]) - 34.5) < 1.0e-6


def test_preview_rejects_non_pattern_input():
    with pytest.raises(TypeError, match="CustomAntennaPattern"):
        build_custom_pattern_preview_figure({"not": "a pattern"})


def test_preview_2d_cuts_match_evaluate_pattern_2d_at_principal_planes():
    """The two cut lines must equal ``evaluate_pattern_2d`` on the
    dense display sweep along the declared principal planes — a
    regression guard against subtle axis-ordering bugs in the cut
    extraction."""
    pat = _pattern_2d_az_el()
    fig = build_custom_pattern_preview_figure(pat)
    cut_axes = _find_cut_axes(fig)
    (cut_ax,) = cut_axes
    lines = cut_ax.get_lines()
    assert len(lines) == 2

    # The cuts are evaluated at the pattern-matched display resolution.
    # Reconstruct the same sweep and compare.
    from scepter.custom_antenna_preview import _display_npts

    n0, n1 = _display_npts(pat)
    az_grid = np.asarray(pat.az_grid_deg, dtype=np.float64)
    el_grid = np.asarray(pat.el_grid_deg, dtype=np.float64)
    az = np.linspace(float(az_grid[0]), float(az_grid[-1]), n0)
    el = np.linspace(float(el_grid[0]), float(el_grid[-1]), n1)
    h_ref = evaluate_pattern_2d(pat, az, np.zeros_like(az))
    e_ref = evaluate_pattern_2d(pat, np.zeros_like(el), el)

    np.testing.assert_allclose(np.asarray(lines[0].get_ydata()), h_ref, atol=1e-9)
    np.testing.assert_allclose(np.asarray(lines[1].get_ydata()), e_ref, atol=1e-9)
