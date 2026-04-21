"""Matplotlib-based preview figures for custom antenna patterns.

Stage 21 of the 30-stage custom-antenna plan. Provides a single
factory function :func:`build_custom_pattern_preview_figure` that
returns a :class:`matplotlib.figure.Figure` rendering a loaded
:class:`~scepter.custom_antenna.CustomAntennaPattern`:

- **1-D** (``kind="1d_axisymmetric"``) → polar plot of G(θ) in dBi.
- **2-D** (``kind="2d"``) → heatmap of G over the product grid plus
  two principal-plane 1-D cuts so asymmetry is immediately visible.
  For ``grid_mode="theta_phi"`` the cuts are at φ=0° and φ=90°
  (matching the schema doc's "1-D cuts along φ=0° / φ=90°" guidance);
  for ``grid_mode="az_el"`` the cuts are at el=0° (H-plane) and
  az=0° (E-plane) — the two principal planes users recognise from a
  typical phased-array datasheet.

The factory is pure CPU / matplotlib-only — no Qt, no GPU, no session.
The actual Qt dialog that embeds the figure lives in the GUI layer
(Stage 21's remaining widget work); this module is deliberately
independent so the figure is testable headless, reusable from
Jupyter, and exportable to PNG/PDF without pulling a Qt display.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import numpy as np

from scepter.custom_antenna import (
    CustomAntennaPattern,
    GRID_MODE_AZEL,
    GRID_MODE_THETAPHI,
    KIND_1D,
    KIND_2D,
    evaluate_pattern_1d,
    evaluate_pattern_2d,
)


# Convert the stored (possibly relative-normalised) LUT to absolute
# dBi via the existing CPU evaluators — the preview must show exactly
# what the runtime evaluator sees, so a user sanity-checking a newly
# loaded file gets the same shape that downstream EPFD / PFD
# accumulation will consume.


def _plot_1d(
    pattern: CustomAntennaPattern,
    *,
    projection: str = "cartesian",
) -> matplotlib.figure.Figure:
    """Plot G(θ) in either cartesian (default) or polar projection.

    Cartesian is the default because antenna engineers read far-
    sidelobe / plateau / null structure from a linear θ vs dB plot;
    polar clips deep nulls at an arbitrary radial floor so shape
    detail below ~peak−60 dB is invisible. Polar is kept as an
    optional view for users who want the classic "petal" look.
    """
    theta_deg = np.asarray(pattern.grid_deg, dtype=np.float64)
    # ``evaluate_pattern_1d`` honours step discontinuities and
    # shifts relative-mode patterns into absolute dBi.
    gain_dbi = evaluate_pattern_1d(pattern, theta_deg)

    if projection == "polar":
        fig = matplotlib.figure.Figure(figsize=(6.4, 6.4), layout="constrained")
        ax = fig.add_subplot(1, 1, 1, projection="polar")
        theta_rad = np.deg2rad(theta_deg)
        ax.plot(theta_rad, gain_dbi, lw=1.4, color="tab:blue")
        # Mirror across boresight so the viewer sees a classic
        # pattern "petal" rather than a half-circle.
        ax.plot(-theta_rad, gain_dbi, lw=1.4, color="tab:blue")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise — antenna convention
        ax.set_title(
            f"Custom 1-D axisymmetric — peak {pattern.peak_gain_dbi:.1f} dBi",
            pad=12,
        )
        # Radial range anchored at the actual LUT floor (clipped at
        # peak − 80 dB so extreme analytical nulls don't crush the
        # scale). This keeps far-out plateaus visible where polar
        # previously hid them.
        r_floor = float(np.nanmin(gain_dbi))
        r_min = max(r_floor - 2.0, float(pattern.peak_gain_dbi) - 80.0)
        r_max = float(pattern.peak_gain_dbi) + 2.0
        ax.set_ylim(r_min, r_max)
        ax.set_ylabel("Gain (dBi)", labelpad=28)
        ax.grid(True, ls=":", alpha=0.6)
        return fig

    if projection != "cartesian":
        raise ValueError(
            f"Unsupported 1-D projection {projection!r}; "
            "expected 'cartesian' or 'polar'."
        )

    fig = matplotlib.figure.Figure(figsize=(7.5, 5.2), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(theta_deg, gain_dbi, lw=1.4, color="tab:blue")
    ax.set_xlabel("Off-axis angle θ (deg)")
    ax.set_ylabel("Gain (dBi)")
    ax.set_title(
        f"Custom 1-D axisymmetric — peak {pattern.peak_gain_dbi:.1f} dBi"
    )
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_xlim(float(theta_deg[0]), float(theta_deg[-1]))
    return fig


def _grid_labels(pattern: CustomAntennaPattern) -> tuple[str, str]:
    """Return axis labels matching the pattern's ``grid_mode``."""
    if pattern.grid_mode == GRID_MODE_AZEL:
        return ("Azimuth (deg)", "Elevation (deg)")
    if pattern.grid_mode == GRID_MODE_THETAPHI:
        return ("θ (deg)", "φ (deg)")
    raise ValueError(f"Unsupported grid_mode {pattern.grid_mode!r}.")  # pragma: no cover


_DISPLAY_NPTS_MIN = 256   # minimum samples per axis for smooth rendering
_DISPLAY_NPTS_MAX = 2048  # cap to avoid excessive memory in the preview


def _display_npts(pattern: CustomAntennaPattern) -> tuple[int, int]:
    """Choose display grid resolution that matches the pattern's own grid.

    The preview should show exactly what the GPU bilinear kernel
    produces — so the display grid matches the pattern grid resolution,
    clamped to a reasonable rendering range.
    """
    if pattern.grid_mode == GRID_MODE_THETAPHI:
        n0 = len(pattern.theta_grid_deg)
        n1 = len(pattern.phi_grid_deg)
    elif pattern.grid_mode == GRID_MODE_AZEL:
        n0 = len(pattern.az_grid_deg)
        n1 = len(pattern.el_grid_deg)
    else:
        return (_DISPLAY_NPTS_MIN, _DISPLAY_NPTS_MIN)
    # Use at least the pattern grid resolution (so no downsampling),
    # but add a minimum for visual smoothness on very coarse grids.
    n0 = max(n0, _DISPLAY_NPTS_MIN)
    n1 = max(n1, _DISPLAY_NPTS_MIN)
    n0 = min(n0, _DISPLAY_NPTS_MAX)
    n1 = min(n1, _DISPLAY_NPTS_MAX)
    return (n0, n1)


def _principal_cuts(
    pattern: CustomAntennaPattern,
) -> tuple[tuple[str, np.ndarray, np.ndarray], tuple[str, np.ndarray, np.ndarray]]:
    """Return two ``(label, x_deg, gain_dbi)`` principal-plane cuts.

    ``theta_phi`` mode: φ=0° and φ=90°. ``az_el`` mode: el=0°
    (H-plane) and az=0° (E-plane).

    Resolution matches the pattern's grid along the swept axis so
    the cuts show the actual bilinear interpolation shape.
    """
    n0, n1 = _display_npts(pattern)
    if pattern.grid_mode == GRID_MODE_THETAPHI:
        theta_grid = np.asarray(pattern.theta_grid_deg, dtype=np.float64)
        theta = np.linspace(float(theta_grid[0]), float(theta_grid[-1]), n0)
        cut_a = evaluate_pattern_2d(pattern, theta, np.zeros_like(theta))
        cut_b = evaluate_pattern_2d(pattern, theta, np.full_like(theta, 90.0))
        return (("φ = 0°", theta, cut_a), ("φ = 90°", theta, cut_b))
    if pattern.grid_mode == GRID_MODE_AZEL:
        az_grid = np.asarray(pattern.az_grid_deg, dtype=np.float64)
        el_grid = np.asarray(pattern.el_grid_deg, dtype=np.float64)
        az = np.linspace(float(az_grid[0]), float(az_grid[-1]), n0)
        el = np.linspace(float(el_grid[0]), float(el_grid[-1]), n1)
        h_cut = evaluate_pattern_2d(pattern, az, np.zeros_like(az))
        e_cut = evaluate_pattern_2d(pattern, np.zeros_like(el), el)
        return (("el = 0° (H-plane)", az, h_cut), ("az = 0° (E-plane)", el, e_cut))
    raise ValueError(f"Unsupported grid_mode {pattern.grid_mode!r}.")  # pragma: no cover


def _plot_2d(pattern: CustomAntennaPattern) -> matplotlib.figure.Figure:
    """Heatmap of the interpolated surface plus two principal-plane cuts.

    Display resolution matches the pattern's own grid so the preview
    shows exactly what the GPU bilinear kernel produces — no
    artificial upsampling or smoothing.  Coarse LUTs look visibly
    faceted; fine LUTs look smooth.  Grid node positions are overlaid
    as small markers.
    """
    if pattern.grid_mode == GRID_MODE_THETAPHI:
        grid0 = np.asarray(pattern.theta_grid_deg, dtype=np.float64)
        grid1 = np.asarray(pattern.phi_grid_deg, dtype=np.float64)
    elif pattern.grid_mode == GRID_MODE_AZEL:
        grid0 = np.asarray(pattern.az_grid_deg, dtype=np.float64)
        grid1 = np.asarray(pattern.el_grid_deg, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported grid_mode {pattern.grid_mode!r}.")

    # Display grid matched to the pattern's own resolution so the
    # preview shows exactly what the GPU bilinear kernel produces —
    # coarse LUTs look visibly faceted, fine LUTs look smooth.
    n0, n1 = _display_npts(pattern)
    disp0 = np.linspace(float(grid0[0]), float(grid0[-1]), n0)
    disp1 = np.linspace(float(grid1[0]), float(grid1[-1]), n1)
    mesh0, mesh1 = np.meshgrid(disp0, disp1, indexing="ij")
    gain = evaluate_pattern_2d(pattern, mesh0, mesh1)

    fig = matplotlib.figure.Figure(figsize=(10.5, 5.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cuts = fig.add_subplot(gs[0, 1])

    x_label, y_label = _grid_labels(pattern)

    vmin = float(pattern.peak_gain_dbi) - 50.0
    vmax = float(pattern.peak_gain_dbi) + 2.0

    # Heatmap of the interpolated surface.
    im = ax_heat.pcolormesh(
        mesh0, mesh1, gain,
        shading="gouraud",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    # Overlay coarse grid lines + intersections so the user can see
    # where reference points are defined.
    for v in grid0:
        ax_heat.axvline(v, color="white", lw=0.3, alpha=0.45)
    for v in grid1:
        ax_heat.axhline(v, color="white", lw=0.3, alpha=0.45)
    ref0, ref1 = np.meshgrid(grid0, grid1, indexing="ij")
    ax_heat.scatter(
        ref0.ravel(), ref1.ravel(),
        s=6, c="white", edgecolors="black", linewidths=0.3,
        zorder=3, alpha=0.7,
    )

    ax_heat.set_xlabel(x_label)
    ax_heat.set_ylabel(y_label)
    ax_heat.set_title(
        f"Custom 2-D ({pattern.grid_mode}) — peak "
        f"{pattern.peak_gain_dbi:.1f} dBi"
    )
    fig.colorbar(im, ax=ax_heat, label="Gain (dBi)")

    # Principal-plane cuts — dense evaluation shows interpolation shape.
    cut_a, cut_b = _principal_cuts(pattern)
    label_a, x_a, g_a = cut_a
    label_b, x_b, g_b = cut_b
    ax_cuts.plot(x_a, g_a, lw=1.3, label=label_a, color="tab:blue")
    ax_cuts.plot(x_b, g_b, lw=1.3, label=label_b, color="tab:orange")
    ax_cuts.set_xlabel("Angle along cut (deg)")
    ax_cuts.set_ylabel("Gain (dBi)")
    ax_cuts.set_title("Principal-plane cuts")
    ax_cuts.grid(True, ls=":", alpha=0.6)
    ax_cuts.legend(loc="upper right", frameon=False)

    return fig


def build_custom_pattern_preview_figure(
    pattern: CustomAntennaPattern,
    *,
    projection: str = "cartesian",
) -> matplotlib.figure.Figure:
    """Build a preview figure for a loaded custom antenna pattern.

    Dispatches on ``pattern.kind``. The returned figure is a
    :class:`matplotlib.figure.Figure` with no attached backend — ready
    to embed in a Qt canvas, save to PNG via ``fig.savefig(...)``,
    display in Jupyter, or discard.

    ``projection`` controls the 1-D rendering: ``"cartesian"`` (default)
    shows θ vs dBi on linear axes — the engineer-friendly default that
    matches the 1-D editor view; ``"polar"`` shows the classic compass
    petal. Ignored for 2-D patterns (heatmap + principal-plane cuts).

    The figure shows exactly what the GPU bilinear kernel produces.
    Display resolution matches the pattern's own grid (clamped to
    [256, 2048] per axis for rendering) so coarse LUTs look visibly
    faceted and fine LUTs look smooth — no artificial upsampling.
    Grid node positions are overlaid as markers.
    """
    if not isinstance(pattern, CustomAntennaPattern):
        raise TypeError(
            "build_custom_pattern_preview_figure requires a "
            f"CustomAntennaPattern instance; got {type(pattern).__name__}."
        )
    if pattern.kind == KIND_1D:
        return _plot_1d(pattern, projection=projection)
    if pattern.kind == KIND_2D:
        return _plot_2d(pattern)
    raise ValueError(f"Unsupported pattern kind {pattern.kind!r}.")  # pragma: no cover


__all__ = ["build_custom_pattern_preview_figure"]
