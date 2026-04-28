#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
w_corrected_image.py - W-term corrected quick-look imaging from UVW data.

Loads UVW coordinates and complex visibilities from an NPZ archive and
produces a w-term corrected dirty image using the w-stacking algorithm.
The pointing direction (the phase centre encoded in the UVW frame) is used
as the centre of the output image.

Background
----------
The standard 2-D quick-look imager (``quick_dirty_image_from_uvw.py``) ignores
the w-term, treating all baselines as coplanar.  For wide-field imaging or long
baselines the w-term introduces a position- and baseline-dependent phase error:

    V(u, v, w) = ∫∫ I(l, m) / n · exp(−2πi [ul + vm + w(n − 1)]) dl dm

where (l, m) are direction cosines on the celestial sphere and
n = √(1 − l² − m²).  The term w(n − 1) is zero only at the phase centre
(l = m = 0 → n = 1) and grows with distance from the pointing.

This script applies the **w-stacking** correction:

1. Extend visibilities with Hermitian conjugates at (−u, −v, −w).
2. Sort all samples into narrow w-planes.
3. For each w-plane with centre w_k:

   a. Grid samples onto the UV plane.
   b. Apply the inverse 2-D FFT → complex image D_k(l, m).
   c. Multiply by the w-correction kernel: exp(−2πi · w_k · (n − 1)).

4. Sum corrected plane images → final w-corrected image.
5. Optionally divide by n to correct the direction-cosine volume element
   (``--apply-n-correction``).

The output is a three-panel comparison figure:

- Panel 1: standard dirty image (no w-correction, using all samples in one plane).
- Panel 2: w-corrected dirty image (w-stacking result).
- Panel 3: difference (w-corrected − standard).

Usage
-----
    conda activate scepter-dev
    python scripts/w_corrected_image.py tracking_uvw.npz \\
        --output w_corrected.png \\
        --n-wplanes 16 \\
        --fov-deg 2.0

Input NPZ format
----------------
The input ``.npz`` file must contain:

- ``pointing_uvw_m`` (preferred) or ``uvw`` (fallback): UVW coordinates in
  metres with shape ``(N_ant, T, 3)`` or ``(N_vis, 3)``.
- ``vis``: complex visibilities aligned with the UVW array.
- ``freq_hz`` or ``freq_mhz``: observing frequency (or pass ``--freq-mhz``).

Notes
-----
- This is a quick-look imager: nearest-neighbour UV gridding, no
  deconvolution, no primary beam correction.
- Increasing ``--n-wplanes`` improves accuracy at the cost of runtime.
  16 planes is a good default for FOVs up to a few degrees.
- The ``w`` components in the UVW frame are assumed to be in metres, aligned
  with the pointing direction as the phase centre.
- Both images (standard and w-corrected) are normalised by the peak of the
  w-corrected dirty beam.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "scepter-mpl-cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quick_dirty_image_from_uvw import (
    SPEED_OF_LIGHT_M_PER_S,
    flatten_uvw_visibility_samples,
    resolve_frequency_hz,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_l_m_n_grid(npix: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (l, m, n) direction-cosine grids for the image plane.

    Parameters
    ----------
    npix : int
        Number of pixels along each image axis.
    fov_deg : float
        Total field of view in degrees.  The image covers
        ``[−fov_deg/2, +fov_deg/2]`` in both l and m.

    Returns
    -------
    ll : numpy.ndarray, shape (npix, npix)
        l direction cosines (east–west), varying along axis 1.
    mm : numpy.ndarray, shape (npix, npix)
        m direction cosines (north–south), varying along axis 0.
    nn : numpy.ndarray, shape (npix, npix)
        n = √(1 − l² − m²), clipped to zero outside the celestial hemisphere.

    Notes
    -----
    The pixel size is ``fov_rad / npix`` radians, consistent with the UV cell
    spacing ``1 / fov_rad`` wavelengths used by the gridder.  Axis 0 of the
    image array corresponds to v (m direction) and axis 1 to u (l direction).
    """
    fov_rad = np.deg2rad(float(fov_deg))
    pixel_size_rad = fov_rad / float(npix)
    centre = npix // 2
    axis = (np.arange(npix) - centre) * pixel_size_rad  # direction cosines (small-angle approx)
    mm, ll = np.meshgrid(axis, axis, indexing="ij")  # axis 0 = m, axis 1 = l
    nn = np.sqrt(np.maximum(1.0 - ll ** 2 - mm ** 2, 0.0))
    return ll, mm, nn


def _grid_uv_plane(
    u_lambda: np.ndarray,
    v_lambda: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    npix: int,
    uv_cell_lambda: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Nearest-neighbour grid UV samples onto a regular (npix × npix) grid.

    Parameters
    ----------
    u_lambda, v_lambda : numpy.ndarray, shape (N,)
        UV coordinates in wavelengths.
    vis : numpy.ndarray, shape (N,)
        Complex visibilities.
    weights : numpy.ndarray, shape (N,)
        Per-sample weights.
    npix : int
        Grid size.
    uv_cell_lambda : float
        UV grid spacing in wavelengths.

    Returns
    -------
    uv_grid : numpy.ndarray, shape (npix, npix), complex128
        Gridded weighted visibilities.
    samp_grid : numpy.ndarray, shape (npix, npix), float64
        Gridded sampling function (sum of weights per cell).
    n_kept : int
        Number of samples within the grid bounds.
    n_dropped : int
        Number of samples discarded outside the grid bounds.
    """
    centre = npix // 2
    u_idx = np.rint(u_lambda / uv_cell_lambda).astype(np.int64) + centre
    v_idx = np.rint(v_lambda / uv_cell_lambda).astype(np.int64) + centre
    in_bounds = (u_idx >= 0) & (u_idx < npix) & (v_idx >= 0) & (v_idx < npix)
    n_kept = int(np.count_nonzero(in_bounds))
    n_dropped = int(np.count_nonzero(~in_bounds))

    uv_grid = np.zeros((npix, npix), dtype=np.complex128)
    samp_grid = np.zeros((npix, npix), dtype=np.float64)
    np.add.at(uv_grid, (v_idx[in_bounds], u_idx[in_bounds]), vis[in_bounds] * weights[in_bounds])
    np.add.at(samp_grid, (v_idx[in_bounds], u_idx[in_bounds]), weights[in_bounds])
    return uv_grid, samp_grid, n_kept, n_dropped


def _compute_uniform_weights(
    u_lambda: np.ndarray,
    v_lambda: np.ndarray,
    npix: int,
    uv_cell_lambda: float,
) -> np.ndarray:
    """Compute per-sample uniform (Briggs-like) weights via inverse cell occupancy."""
    centre = npix // 2
    u_idx = np.rint(u_lambda / uv_cell_lambda).astype(np.int64) + centre
    v_idx = np.rint(v_lambda / uv_cell_lambda).astype(np.int64) + centre
    in_bounds = (u_idx >= 0) & (u_idx < npix) & (v_idx >= 0) & (v_idx < npix)
    density = np.zeros((npix, npix), dtype=np.float64)
    np.add.at(density, (v_idx[in_bounds], u_idx[in_bounds]), 1.0)
    weights = np.ones(len(u_lambda), dtype=np.float64)
    occupied = density[v_idx[in_bounds], u_idx[in_bounds]]
    weights[in_bounds] = np.where(occupied > 0.0, 1.0 / occupied, 1.0)
    return weights


# ---------------------------------------------------------------------------
# Satellite helpers (mirrors diff_sat_uvw.py)
# ---------------------------------------------------------------------------

def _validate_sat_uvw_inputs(
    pointing_uvw_m: np.ndarray,
    satellite_uvw_m: np.ndarray,
) -> tuple[int, int, int]:
    """Validate pointing and satellite UVW arrays and return (N_ant, T, N_sat)."""
    pointing = np.asarray(pointing_uvw_m, dtype=np.float64)
    satellite = np.asarray(satellite_uvw_m, dtype=np.float64)
    if pointing.ndim != 3 or pointing.shape[-1] != 3:
        raise ValueError(
            f"pointing_uvw_m must have shape (N_ant, T, 3). Got {pointing.shape!r}."
        )
    if satellite.ndim != 4 or satellite.shape[-1] != 3:
        raise ValueError(
            f"satellite_uvw_m must have shape (N_ant, T, N_sat, 3). Got {satellite.shape!r}."
        )
    if satellite.shape[2] == 0:
        raise ValueError("satellite_uvw_m must include at least one satellite track (N_sat > 0).")
    if satellite.shape[0] != pointing.shape[0] or satellite.shape[1] != pointing.shape[1]:
        raise ValueError(
            "satellite_uvw_m leading axes must match pointing_uvw_m (N_ant, T). "
            f"Got pointing {pointing.shape!r} and satellite {satellite.shape!r}."
        )
    if not np.isfinite(pointing).all():
        raise ValueError("pointing_uvw_m contains non-finite values.")
    if not np.isfinite(satellite).all():
        raise ValueError("satellite_uvw_m contains non-finite values.")
    return int(pointing.shape[0]), int(pointing.shape[1]), int(satellite.shape[2])


def _parse_satellite_indices(indices_spec: str | None, total_satellites: int) -> np.ndarray:
    """Parse a comma-separated list of zero-based satellite indices."""
    if total_satellites <= 0:
        raise ValueError("At least one satellite must be available for selection.")
    if indices_spec is None or indices_spec.strip() == "":
        return np.arange(total_satellites, dtype=np.int64)
    parsed: list[int] = []
    for token in indices_spec.split(","):
        value_text = token.strip()
        if value_text == "":
            continue
        try:
            value = int(value_text)
        except ValueError as exc:
            raise ValueError(
                "--satellite-indices must be a comma-separated list of integers, "
                f"got token {value_text!r}."
            ) from exc
        parsed.append(value)
    if not parsed:
        raise ValueError("--satellite-indices resolved to an empty selection.")
    seen: set[int] = set()
    unique_ordered: list[int] = []
    for value in parsed:
        if value < 0 or value >= total_satellites:
            raise ValueError(
                f"Satellite index out of range. Got {value}, valid range is [0, {total_satellites - 1}]."
            )
        if value not in seen:
            seen.add(value)
            unique_ordered.append(value)
    if not unique_ordered:
        raise ValueError("No valid satellite indices remain after de-duplication.")
    return np.asarray(unique_ordered, dtype=np.int64)


def _select_satellite_tracks(
    satellite_uvw_m: np.ndarray,
    indices_spec: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a satellite subset from the UVW tensor using zero-based indices."""
    total_satellites = int(satellite_uvw_m.shape[2])
    selected_indices = _parse_satellite_indices(indices_spec, total_satellites)
    selected_tracks = np.asarray(satellite_uvw_m[:, :, selected_indices, :], dtype=np.float64)
    if selected_tracks.shape[2] == 0:
        raise ValueError("Satellite selection produced zero tracks.")
    return selected_tracks, selected_indices


def _build_with_satellite_samples(
    pointing_uvw_m: np.ndarray,
    satellite_uvw_m: np.ndarray,
    vis_samples_no_sat: np.ndarray,
    vis_template: np.ndarray,
    *,
    drop_zero_baseline: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate pointing + per-satellite UVW/visibility samples."""
    pointing_samples_m, _ = flatten_uvw_visibility_samples(
        pointing_uvw_m,
        vis_template,
        drop_zero_baseline=drop_zero_baseline,
    )
    n_sat = satellite_uvw_m.shape[2]
    satellite_sample_blocks: list[np.ndarray] = []
    satellite_vis_blocks: list[np.ndarray] = []
    for sat_idx in range(n_sat):
        sat_samples_m, _ = flatten_uvw_visibility_samples(
            satellite_uvw_m[:, :, sat_idx, :],
            vis_template,
            drop_zero_baseline=drop_zero_baseline,
        )
        satellite_sample_blocks.append(sat_samples_m)
        sat_vis = np.resize(vis_samples_no_sat, sat_samples_m.shape[0])
        satellite_vis_blocks.append(sat_vis)
    with_sat_samples = np.concatenate([pointing_samples_m, *satellite_sample_blocks], axis=0)
    with_sat_vis = np.concatenate([vis_samples_no_sat, *satellite_vis_blocks], axis=0)
    return with_sat_samples, with_sat_vis


# ---------------------------------------------------------------------------
# Core imaging function
# ---------------------------------------------------------------------------

def make_w_corrected_dirty_image(
    uvw_samples_m: np.ndarray,
    vis_samples: np.ndarray,
    frequency_hz: float,
    *,
    npix: int = 512,
    fov_deg: float = 2.0,
    weighting: str = "natural",
    hermitian_mirror: bool = True,
    n_wplanes: int = 16,
    apply_n_correction: bool = False,
    n_correction_floor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """
    Form a w-term corrected quick dirty image using the w-stacking algorithm.

    Parameters
    ----------
    uvw_samples_m : numpy.ndarray, shape (N_sample, 3)
        Flattened UVW coordinates in metres.  All three components are used:
        u and v for gridding, w for the w-stacking correction.
    vis_samples : numpy.ndarray, shape (N_sample,)
        Complex visibilities aligned with ``uvw_samples_m``.
    frequency_hz : float
        Observing frequency in Hz.  UVW coordinates are converted from metres
        to wavelengths via ``λ = c / frequency_hz``.
    npix : int, optional
        Output image size in pixels.  Both image axes have length ``npix``.
        Default: 512.
    fov_deg : float, optional
        Total image field of view in degrees.  The tangent-plane image covers
        ``[−fov_deg/2, +fov_deg/2]`` in l and m.  Default: 2.0.
    weighting : {"natural", "uniform"}, optional
        UV weighting scheme.  ``"natural"`` weights all samples equally.
        ``"uniform"`` applies inverse cell-occupancy weighting to suppress
        densely sampled regions.  Default: ``"natural"``.
    hermitian_mirror : bool, optional
        If ``True``, add conjugate baselines at ``(−u, −v, −w)`` so that
        one-sided UV coverage still produces a real-valued image.  The
        w-sign is also negated so conjugate samples land in the correct
        (negative) w-plane.  Default: ``True``.
    n_wplanes : int, optional
        Number of w-planes into which the baseline range is divided.  More
        planes give a more accurate correction at higher computational cost.
        Default: 16.
    apply_n_correction : bool, optional
        If ``True``, divide the w-corrected image by
        n = √(1 − l² − m²) after beam normalisation.  This corrects the
        direction-cosine volume element and projects from the tangent plane
        onto the celestial sphere.  Pixels where n < ``n_correction_floor``
        are set to NaN to avoid division instability near the horizon.
        Default: ``False``.
    n_correction_floor : float, optional
        Minimum n value for the n-correction.  Pixels below this threshold
        are masked to NaN when ``apply_n_correction=True``.  Default: 0.1.

    Returns
    -------
    w_corr_image : numpy.ndarray, shape (npix, npix)
        W-term corrected dirty image, normalised by the peak of the
        w-corrected dirty beam.
    std_image : numpy.ndarray, shape (npix, npix)
        Standard dirty image (single w-plane, no w-correction), normalised
        by the same beam peak as ``w_corr_image`` for consistent comparison.
    dirty_beam : numpy.ndarray, shape (npix, npix)
        W-corrected dirty beam, normalised to peak unity.
    metadata : dict[str, float | int]
        Scalar diagnostics:

        - ``wavelength_m``: observing wavelength in metres.
        - ``uv_cell_lambda``: UV grid cell spacing in wavelengths.
        - ``n_wplanes_used``: effective number of w-planes (1 when all w = 0).
        - ``w_min_lambda``: minimum w coordinate in wavelengths.
        - ``w_max_lambda``: maximum w coordinate in wavelengths.
        - ``gridded_sample_count``: total samples kept across all w-planes.
        - ``dropped_sample_count``: total samples discarded outside the grid.

    Raises
    ------
    ValueError
        If inputs are empty, misaligned, or physically invalid; if no
        gridded samples remain; or if the dirty beam peak is non-positive.

    Notes
    -----
    The imaging model is far-field quasi-monochromatic interferometry.  The
    pointing direction (phase centre of the UVW frame) is the image centre
    (l = m = 0).  No near-field, bandwidth-smearing, or time-smearing
    corrections are applied.

    The w-stacking sum converges to the exact visibility equation in the limit
    n_wplanes → ∞ with infinitely fine UV gridding.  For a typical 2-degree
    FOV, 16 planes is sufficient to reduce the residual w-phase error to below
    1 percent of the beam width.
    """
    uvw_array = np.asarray(uvw_samples_m, dtype=np.float64)
    vis_array = np.asarray(vis_samples, dtype=np.complex128)

    if uvw_array.ndim != 2 or uvw_array.shape[1] != 3:
        raise ValueError(
            "uvw_samples_m must have shape (N_sample, 3). "
            f"Got {uvw_array.shape!r}."
        )
    if vis_array.ndim != 1 or vis_array.shape[0] != uvw_array.shape[0]:
        raise ValueError(
            "vis_samples must be 1-D and aligned with uvw_samples_m. "
            f"Got {vis_array.shape!r} for {uvw_array.shape[0]} UVW samples."
        )
    if npix <= 0:
        raise ValueError(f"npix must be positive, got {npix}.")
    if not np.isfinite(fov_deg) or fov_deg <= 0.0:
        raise ValueError(f"fov_deg must be positive, got {fov_deg!r}.")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz!r}.")
    if weighting not in {"natural", "uniform"}:
        raise ValueError(f"weighting must be 'natural' or 'uniform', got {weighting!r}.")
    if n_wplanes < 1:
        raise ValueError(f"n_wplanes must be >= 1, got {n_wplanes}.")

    wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    u_lambda = uvw_array[:, 0] / wavelength_m
    v_lambda = uvw_array[:, 1] / wavelength_m
    w_lambda = uvw_array[:, 2] / wavelength_m
    vis = vis_array.copy()

    # Hermitian extension: add conjugate baseline at (−u, −v, −w).
    # Negating w ensures the conjugate sample lands in the correct (negative)
    # w-plane so that the phase corrections cancel and the stacked image is real.
    if hermitian_mirror:
        u_lambda = np.concatenate([u_lambda, -u_lambda])
        v_lambda = np.concatenate([v_lambda, -v_lambda])
        w_lambda = np.concatenate([w_lambda, -w_lambda])
        vis = np.concatenate([vis, np.conj(vis)])

    fov_rad = np.deg2rad(float(fov_deg))
    uv_cell_lambda = 1.0 / fov_rad

    # Image-plane coordinate grids.
    ll, mm, nn = _make_l_m_n_grid(npix, fov_deg)

    # Per-sample weights (computed once over the full extended UV set).
    if weighting == "uniform":
        weights = _compute_uniform_weights(u_lambda, v_lambda, npix, uv_cell_lambda)
    else:
        weights = np.ones(len(vis), dtype=np.float64)

    # W-plane edges and centres.
    w_min = float(w_lambda.min())
    w_max = float(w_lambda.max())
    if w_min == w_max:
        # Degenerate case: all visibilities at the same w (including w = 0).
        w_edges = np.array([w_min - 0.5, w_max + 0.5])
        w_centres = np.array([0.5 * (w_min + w_max)])
        n_wplanes_eff = 1
    else:
        w_edges = np.linspace(w_min, w_max, n_wplanes + 1)
        w_centres = 0.5 * (w_edges[:-1] + w_edges[1:])
        n_wplanes_eff = int(n_wplanes)

    accumulated = np.zeros((npix, npix), dtype=np.complex128)
    accumulated_beam = np.zeros((npix, npix), dtype=np.complex128)
    total_kept = 0
    total_dropped = 0

    for k in range(n_wplanes_eff):
        in_plane = w_lambda >= w_edges[k]
        if k < n_wplanes_eff - 1:
            in_plane &= w_lambda < w_edges[k + 1]
        else:
            in_plane &= w_lambda <= w_edges[k + 1]
        if not np.any(in_plane):
            continue

        uv_grid, samp_grid, n_kept, n_dropped = _grid_uv_plane(
            u_lambda[in_plane],
            v_lambda[in_plane],
            vis[in_plane],
            weights[in_plane],
            npix,
            uv_cell_lambda,
        )
        total_kept += n_kept
        total_dropped += n_dropped

        D_k = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid)))
        B_k = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(samp_grid)))

        # W-correction kernel: exp(−2πi · w_k · (n − 1)).
        # At the phase centre (l=m=0, n=1) the phase is zero; it grows away
        # from centre where n < 1.
        w_k = float(w_centres[k])
        w_phase = np.exp(-2j * np.pi * w_k * (nn - 1.0))
        accumulated += D_k * w_phase
        accumulated_beam += B_k * w_phase

    if total_kept == 0:
        raise ValueError(
            "All UV samples fall outside the FFT grid. Increase --npix or decrease --fov-deg."
        )

    w_corr_image = accumulated.real
    dirty_beam = accumulated_beam.real

    beam_peak = float(np.nanmax(dirty_beam))
    if beam_peak <= 0.0:
        raise ValueError("W-corrected dirty beam peak is non-positive; cannot normalise.")

    w_corr_image = w_corr_image / beam_peak
    dirty_beam = dirty_beam / beam_peak

    # Optional n-correction: divide by n = sqrt(1 − l² − m²).
    if apply_n_correction:
        safe_nn = np.where(nn >= float(n_correction_floor), nn, np.nan)
        w_corr_image = w_corr_image / safe_nn

    # Standard dirty image (no w-correction) using the same extended samples and
    # weights but treating all w as zero (single-plane, exp(0) = 1 everywhere).
    uv_grid_std, samp_grid_std, _, _ = _grid_uv_plane(
        u_lambda, v_lambda, vis, weights, npix, uv_cell_lambda
    )
    std_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid_std))).real / beam_peak

    metadata: dict[str, float | int] = {
        "wavelength_m": float(wavelength_m),
        "uv_cell_lambda": float(uv_cell_lambda),
        "n_wplanes_used": int(n_wplanes_eff),
        "w_min_lambda": float(w_min),
        "w_max_lambda": float(w_max),
        "gridded_sample_count": int(total_kept),
        "dropped_sample_count": int(total_dropped),
    }
    return w_corr_image, std_image, dirty_beam, metadata


# ---------------------------------------------------------------------------
# Figure output
# ---------------------------------------------------------------------------

def save_comparison_figure(
    std_image: np.ndarray,
    w_corr_image: np.ndarray,
    dirty_beam: np.ndarray,
    output_path: Path,
    *,
    fov_deg: float,
    title: str,
    include_beam: bool = False,
) -> None:
    """
    Save a comparison figure with standard, w-corrected, and difference panels.

    Parameters
    ----------
    std_image : numpy.ndarray, shape (npix, npix)
        Standard dirty image (no w-correction).
    w_corr_image : numpy.ndarray, shape (npix, npix)
        W-term corrected dirty image.
    dirty_beam : numpy.ndarray, shape (npix, npix)
        W-corrected dirty beam.
    output_path : Path
        Output PNG file path.  Parent directories are created if needed.
    fov_deg : float
        Image field of view in degrees (used to set image extents).
    title : str
        Figure suptitle.
    include_beam : bool, optional
        If ``True``, add a fourth panel showing the dirty beam.
        Default: ``False``.
    """
    n_panels = 4 if include_beam else 3
    figwidth = 5.5 * n_panels
    extent_deg = (-0.5 * fov_deg, 0.5 * fov_deg, -0.5 * fov_deg, 0.5 * fov_deg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diff_image = w_corr_image - std_image
    diff_abs_max = float(np.nanmax(np.abs(diff_image)))
    if not np.isfinite(diff_abs_max) or diff_abs_max == 0.0:
        diff_abs_max = 1.0

    fig, axes = plt.subplots(1, n_panels, figsize=(figwidth, 4.8), constrained_layout=True)

    def _add_panel(ax: plt.Axes, image: np.ndarray, panel_title: str, cmap: str, clim: tuple | None) -> None:
        kwargs: dict = {
            "origin": "lower",
            "extent": extent_deg,
            "cmap": cmap,
            "interpolation": "nearest",
        }
        if clim is not None:
            kwargs["vmin"], kwargs["vmax"] = clim
        artist = ax.imshow(image, **kwargs)
        ax.set_title(panel_title)
        ax.set_xlabel("l offset (deg)")
        ax.set_ylabel("m offset (deg)")
        label = "Delta" if "Difference" in panel_title else "Dirty image"
        fig.colorbar(artist, ax=ax, shrink=0.86, label=label)

    _add_panel(axes[0], std_image, "Standard (no w-correction)", "inferno", None)
    _add_panel(axes[1], w_corr_image, "W-corrected", "inferno", None)
    _add_panel(axes[2], diff_image, "Difference (W-corr − Standard)",
               "coolwarm", (-diff_abs_max, diff_abs_max))
    if include_beam:
        _add_panel(axes[3], dirty_beam, "Dirty beam (w-corrected)", "viridis", None)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_w_corrected_diff_figure(
    no_sat_image: np.ndarray,
    with_sat_image: np.ndarray,
    diff_image: np.ndarray,
    dirty_beam: np.ndarray,
    output_path: Path,
    *,
    fov_deg: float,
    title: str,
    include_beam: bool = False,
) -> None:
    """
    Save a three/four-panel w-corrected differential figure.

    Panels: pointing-only (w-corrected) | pointing+satellites (w-corrected) |
    differential (with − without).  Optional fourth panel: dirty beam.
    """
    n_panels = 4 if include_beam else 3
    figwidth = 5.5 * n_panels
    extent_deg = (-0.5 * fov_deg, 0.5 * fov_deg, -0.5 * fov_deg, 0.5 * fov_deg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diff_abs_max = float(np.nanmax(np.abs(diff_image)))
    if not np.isfinite(diff_abs_max) or diff_abs_max == 0.0:
        diff_abs_max = 1.0

    fig, axes = plt.subplots(1, n_panels, figsize=(figwidth, 4.8), constrained_layout=True)

    def _add_panel(
        ax: "plt.Axes",
        image: np.ndarray,
        panel_title: str,
        cmap: str,
        clim: tuple | None,
    ) -> None:
        kwargs: dict = {
            "origin": "lower",
            "extent": extent_deg,
            "cmap": cmap,
            "interpolation": "nearest",
        }
        if clim is not None:
            kwargs["vmin"], kwargs["vmax"] = clim
        artist = ax.imshow(image, **kwargs)
        ax.set_title(panel_title)
        ax.set_xlabel("l offset (deg)")
        ax.set_ylabel("m offset (deg)")
        label = "Delta" if "Differential" in panel_title else "Dirty image"
        fig.colorbar(artist, ax=ax, shrink=0.86, label=label)

    _add_panel(axes[0], no_sat_image, "Pointing Only (W-corrected)", "inferno", None)
    _add_panel(axes[1], with_sat_image, "Pointing + Satellites (W-corrected)", "inferno", None)
    _add_panel(
        axes[2],
        diff_image,
        "Differential (With − Without)",
        "coolwarm",
        (-diff_abs_max, diff_abs_max),
    )
    if include_beam:
        _add_panel(axes[3], dirty_beam, "Dirty beam (w-corrected)", "viridis", None)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_output_npz(
    output_path: Path,
    *,
    no_sat_image: np.ndarray,
    with_sat_image: np.ndarray,
    diff_image: np.ndarray,
    dirty_beam: np.ndarray,
    frequency_hz: float,
    fov_deg: float,
    npix: int,
    n_wplanes: int,
) -> None:
    """
    Save w-corrected image arrays to an NPZ archive.

    Arrays
    ------
    no_sat_image : float32 (npix, npix) — pointing-only w-corrected dirty image.
    with_sat_image : float32 (npix, npix) — pointing+satellites w-corrected dirty image.
    diff_image : float32 (npix, npix) — differential (with_sat − no_sat).
    dirty_beam : float32 (npix, npix) — w-corrected dirty beam (peak = 1).
    frequency_hz : float64 scalar.
    fov_deg : float64 scalar — field of view in degrees.
    npix : int64 scalar — image size.
    n_wplanes : int64 scalar — w-planes used.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        no_sat_image=no_sat_image.astype(np.float32),
        with_sat_image=with_sat_image.astype(np.float32),
        diff_image=diff_image.astype(np.float32),
        dirty_beam=dirty_beam.astype(np.float32),
        frequency_hz=np.float64(frequency_hz),
        fov_deg=np.float64(fov_deg),
        npix=np.int64(npix),
        n_wplanes=np.int64(n_wplanes),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for w-term corrected imaging."""
    parser = argparse.ArgumentParser(
        description=(
            "Produce a w-term corrected quick dirty image from UVW + visibility "
            "data stored in an .npz archive.  The pointing direction is the "
            "image phase centre.  Saves a three-panel comparison PNG."
        )
    )
    parser.add_argument("input_npz", type=Path, help="Input .npz file.")
    parser.add_argument(
        "--uvw-key",
        default="pointing_uvw_m",
        help=(
            "NPZ key for the UVW array in metres.  Falls back to --fallback-uvw-key "
            "if not found.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--fallback-uvw-key",
        default="uvw",
        help="Fallback NPZ key for UVW when --uvw-key is missing.  Default: %(default)s.",
    )
    parser.add_argument(
        "--vis-key",
        default="vis",
        help="NPZ key for complex visibilities.  Default: %(default)s.",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=None,
        help="Observing frequency in MHz.  Overrides freq_hz / freq_mhz in the file.",
    )
    parser.add_argument(
        "--npix",
        type=int,
        default=512,
        help="Square image size in pixels.  Default: %(default)s.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=2.0,
        help="Total image field of view in degrees.  Default: %(default)s.",
    )
    parser.add_argument(
        "--weighting",
        choices=("natural", "uniform"),
        default="natural",
        help="UV weighting scheme.  Default: %(default)s.",
    )
    parser.add_argument(
        "--n-wplanes",
        type=int,
        default=16,
        help=(
            "Number of w-planes for the w-stacking decomposition.  More planes "
            "give a more accurate w-correction.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--no-hermitian-mirror",
        action="store_true",
        help="Do not extend visibilities with Hermitian conjugates at (−u, −v, −w).",
    )
    parser.add_argument(
        "--apply-n-correction",
        action="store_true",
        help=(
            "Divide the w-corrected image by n = √(1 − l² − m²) to correct the "
            "direction-cosine volume element."
        ),
    )
    parser.add_argument(
        "--keep-zero-baseline",
        action="store_true",
        help="Keep samples with u = v = 0 instead of dropping them.",
    )
    parser.add_argument(
        "--show-beam",
        action="store_true",
        help="Add a fourth panel showing the w-corrected dirty beam.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("w_corrected_image.png"),
        help="Output PNG path.  Default: %(default)s.",
    )
    parser.add_argument(
        "--satellite-key",
        default="satellite_uvw_m",
        help=(
            "NPZ key for satellite UVW tensor (N_ant, T, N_sat, 3).  When this key "
            "is present in the file, the script runs in differential mode: it produces "
            "pointing-only, pointing+satellites, and differential w-corrected images.  "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--satellite-indices",
        default=None,
        help=(
            "Comma-separated zero-based satellite indices to include in the "
            "with-satellite branch (e.g. 0,2,5).  Default: all satellites."
        ),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help=(
            "Optional path to save the w-corrected image arrays as an NPZ archive "
            "(differential mode only).  Default: not saved."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure suptitle.  Defaults to the input file stem.",
    )
    return parser


def _resolve_uvw(dataset: "np.lib.npyio.NpzFile", uvw_key: str, fallback_key: str) -> np.ndarray:
    """Resolve UVW array from preferred key with controlled fallback."""
    if uvw_key in dataset.files:
        return np.asarray(dataset[uvw_key])
    if fallback_key in dataset.files:
        print(
            f"UVW key {uvw_key!r} not found; using fallback {fallback_key!r}."
        )
        return np.asarray(dataset[fallback_key])
    available = ", ".join(dataset.files) if dataset.files else "<empty>"
    raise KeyError(
        f"Missing {uvw_key!r} and fallback {fallback_key!r}. "
        f"Available keys: {available}."
    )


def main() -> None:
    """Run the w-term corrected imaging workflow from the command line.

    Operates in one of two modes, selected automatically based on the NPZ
    contents:

    **Differential mode** (``satellite_uvw_m`` or ``--satellite-key`` found in
    the NPZ): loads pointing UVW and satellite UVW separately, builds
    pointing-only and pointing+satellite sample sets, applies w-stacking to
    both, and outputs a three-panel differential figure plus an optional NPZ
    archive of the corrected images.

    **Single-image comparison mode** (no satellite key found): existing
    behaviour — compares the standard (no w-correction) dirty image against
    the w-stacking result.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    with np.load(args.input_npz, allow_pickle=False) as dataset:
        uvw_m = _resolve_uvw(dataset, args.uvw_key, args.fallback_uvw_key)
        frequency_hz = resolve_frequency_hz(dataset, args.freq_mhz)
        differential_mode = args.satellite_key in dataset.files

        if differential_mode:
            satellite_uvw_m = np.asarray(dataset[args.satellite_key])
            if args.vis_key in dataset.files:
                vis: np.ndarray | None = np.asarray(dataset[args.vis_key], dtype=np.complex128)
            elif uvw_m.ndim == 3 and uvw_m.shape[0] >= 2:
                vis = np.ones(uvw_m.shape[:2], dtype=np.complex128)[1:, :]
                print(
                    f"Visibility key {args.vis_key!r} not found; "
                    "using synthetic unit point-source vis model."
                )
            else:
                raise KeyError(
                    f"Visibility key {args.vis_key!r} not found and pointing UVW "
                    "shape is unsuitable for synthesis."
                )
        else:
            satellite_uvw_m = None
            vis = np.asarray(dataset[args.vis_key], dtype=np.complex128) \
                if args.vis_key in dataset.files else None

    drop_zero = not args.keep_zero_baseline

    if differential_mode:
        assert satellite_uvw_m is not None
        assert vis is not None
        n_ant, n_time, n_sat = _validate_sat_uvw_inputs(uvw_m, satellite_uvw_m)
        selected_sat_uvw_m, selected_indices = _select_satellite_tracks(
            satellite_uvw_m, args.satellite_indices
        )
        n_sat_selected = int(selected_sat_uvw_m.shape[2])

        no_sat_samples_m, no_sat_vis = flatten_uvw_visibility_samples(
            uvw_m, vis, drop_zero_baseline=drop_zero
        )
        with_sat_samples_m, with_sat_vis = _build_with_satellite_samples(
            uvw_m,
            selected_sat_uvw_m,
            no_sat_vis,
            vis,
            drop_zero_baseline=drop_zero,
        )

        no_sat_w_corr, _, dirty_beam_no, meta_no = make_w_corrected_dirty_image(
            no_sat_samples_m,
            no_sat_vis,
            frequency_hz,
            npix=args.npix,
            fov_deg=args.fov_deg,
            weighting=args.weighting,
            hermitian_mirror=not args.no_hermitian_mirror,
            n_wplanes=args.n_wplanes,
            apply_n_correction=args.apply_n_correction,
        )
        with_sat_w_corr, _, _, meta_with = make_w_corrected_dirty_image(
            with_sat_samples_m,
            with_sat_vis,
            frequency_hz,
            npix=args.npix,
            fov_deg=args.fov_deg,
            weighting=args.weighting,
            hermitian_mirror=not args.no_hermitian_mirror,
            n_wplanes=args.n_wplanes,
            apply_n_correction=args.apply_n_correction,
        )

        diff_image = with_sat_w_corr - no_sat_w_corr

        title = args.title or f"W-corrected differential: {args.input_npz.stem}"
        save_w_corrected_diff_figure(
            no_sat_w_corr,
            with_sat_w_corr,
            diff_image,
            dirty_beam_no,
            args.output,
            fov_deg=args.fov_deg,
            title=title,
            include_beam=args.show_beam,
        )

        if args.output_npz is not None:
            save_output_npz(
                args.output_npz,
                no_sat_image=no_sat_w_corr,
                with_sat_image=with_sat_w_corr,
                diff_image=diff_image,
                dirty_beam=dirty_beam_no,
                frequency_hz=frequency_hz,
                fov_deg=args.fov_deg,
                npix=args.npix,
                n_wplanes=int(meta_no["n_wplanes_used"]),
            )

        wavelength_m = SPEED_OF_LIGHT_M_PER_S / frequency_hz
        print(f"Mode                      : differential w-stacking")
        print(f"Pointing UVW shape        : {uvw_m.shape}")
        print(f"Satellite UVW shape       : {satellite_uvw_m.shape}")
        print(f"Selected satellite indices: {selected_indices.tolist()}")
        print(f"Selected satellite count  : {n_sat_selected}")
        print(f"No-sat sample count       : {no_sat_samples_m.shape[0]}")
        print(f"With-sat sample count     : {with_sat_samples_m.shape[0]}")
        print(f"Frequency                 : {frequency_hz / 1.0e6:.6f} MHz")
        print(f"Wavelength                : {wavelength_m:.6f} m")
        print(
            f"W range (no-sat)          : "
            f"[{meta_no['w_min_lambda']:.2f}, {meta_no['w_max_lambda']:.2f}] λ"
        )
        print(
            f"W range (with-sat)        : "
            f"[{meta_with['w_min_lambda']:.2f}, {meta_with['w_max_lambda']:.2f}] λ"
        )
        print(f"W-planes used             : {int(meta_no['n_wplanes_used'])}")
        print(f"Peak |diff|               : {float(np.nanmax(np.abs(diff_image))):.6f}")
        print(f"N-correction applied      : {args.apply_n_correction}")
        print(f"Imaging model             : far-field w-stacking (no near-field correction).")
        if args.output_npz is not None:
            print(f"Saved w-corrected dataset  to {args.output_npz}")
        print(f"Saved differential figure  to {args.output}")

    else:
        # ----------------------------------------------------------------
        # Single-image comparison mode (existing behaviour)
        # ----------------------------------------------------------------
        if vis is None:
            raise KeyError(
                f"Visibility key {args.vis_key!r} not found in {args.input_npz}. "
                "Pass --vis-key or ensure the NPZ contains a 'vis' array."
            )

        uvw_samples_m, vis_samples = flatten_uvw_visibility_samples(
            uvw_m, vis, drop_zero_baseline=drop_zero
        )

        w_corr_image, std_image, dirty_beam, metadata = make_w_corrected_dirty_image(
            uvw_samples_m,
            vis_samples,
            frequency_hz,
            npix=args.npix,
            fov_deg=args.fov_deg,
            weighting=args.weighting,
            hermitian_mirror=not args.no_hermitian_mirror,
            n_wplanes=args.n_wplanes,
            apply_n_correction=args.apply_n_correction,
        )

        title = args.title or f"W-term correction: {args.input_npz.stem}"
        save_comparison_figure(
            std_image,
            w_corr_image,
            dirty_beam,
            args.output,
            fov_deg=args.fov_deg,
            title=title,
            include_beam=args.show_beam,
        )

        diff_peak = float(np.nanmax(np.abs(w_corr_image - std_image)))
        print(f"Mode                      : single-image comparison")
        print(f"Input UVW shape           : {uvw_m.shape}")
        print(f"Input vis shape           : {vis.shape}")
        print(f"Flattened sample count    : {uvw_samples_m.shape[0]}")
        print(f"Frequency                 : {frequency_hz / 1.0e6:.6f} MHz")
        print(f"Wavelength                : {metadata['wavelength_m']:.6f} m")
        print(f"UV cell size              : {metadata['uv_cell_lambda']:.4f} wavelengths")
        print(
            f"W range                   : "
            f"[{metadata['w_min_lambda']:.2f}, {metadata['w_max_lambda']:.2f}] wavelengths"
        )
        print(f"W-planes used             : {int(metadata['n_wplanes_used'])}")
        print(f"Gridded samples kept      : {int(metadata['gridded_sample_count'])}")
        print(f"Samples dropped (OOB)     : {int(metadata['dropped_sample_count'])}")
        print(f"Peak |W-corr − Standard|  : {diff_peak:.6f}")
        print(f"N-correction applied      : {args.apply_n_correction}")
        print(f"Imaging model             : far-field w-stacking (no near-field correction).")
        print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
