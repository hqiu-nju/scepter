"""
Public visualisation helpers for SCEPTer workflows.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>

This module remains part of the GPLv3-licensed SCEPTer project. The author
attribution above does not change the project-wide license.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence, Dict, Tuple, List
import warnings
import h5py
from astropy import units as u
from astropy.constants import R_earth
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, RegularPolygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scepter import scenario

try:
    import scepter.angle_sampler as _angle_sampler_mod
    from scepter.angle_sampler import _S1586_N_CELLS, _skycell_id_s1586
except Exception:  # pragma: no cover
    _angle_sampler_mod = None
    _S1586_N_CELLS = 2334
    _skycell_id_s1586 = None


# -----------------------------------------------------------------------------
# Shared helpers (units, dB math, S.1586 grid, geometry, polar mapping)
# -----------------------------------------------------------------------------

def _new_mpl_figure(*, figsize: tuple[float, float]) -> Figure:
    """Create a canvas-backed Matplotlib figure without going through pyplot."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _new_mpl_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float],
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Return ``Figure.subplots`` output backed by a non-GUI Agg canvas."""
    fig = _new_mpl_figure(figsize=figsize)
    axes = fig.subplots(nrows, ncols, **kwargs)
    return fig, axes


@lru_cache(maxsize=8)
def _cached_mpl_natural_earth_vertices(
    kind: str,
    backend: str = "vendored",
) -> tuple[np.ndarray, ...]:
    """Return cached Matplotlib-ready Natural Earth vertices."""
    from scepter import earthgrid as earthgrid_mod

    geometries = earthgrid_mod._load_natural_earth_geometries(kind, backend=backend)
    vertices: list[np.ndarray] = []
    if kind == "land":
        for geom in geometries:
            polygons = (geom,) if geom.geom_type == "Polygon" else tuple(getattr(geom, "geoms", ()))
            for polygon in polygons:
                if getattr(polygon, "geom_type", "") != "Polygon":
                    continue
                xs, ys = polygon.exterior.xy
                vertices.append(
                    np.column_stack(
                        [
                            np.asarray(xs, dtype=np.float64),
                            np.asarray(ys, dtype=np.float64),
                        ]
                    )
                )
        return tuple(vertices)

    for geom in geometries:
        line_geoms = (geom,) if geom.geom_type == "LineString" else tuple(getattr(geom, "geoms", ()))
        for line in line_geoms:
            if getattr(line, "geom_type", "") != "LineString":
                continue
            xs, ys = line.xy
            vertices.append(
                np.column_stack(
                    [
                        np.asarray(xs, dtype=np.float64),
                        np.asarray(ys, dtype=np.float64),
                    ]
                )
            )
    return tuple(vertices)

def _to_plain_array(data: Any) -> Tuple[np.ndarray, Any | None]:
    """
    Return (values, unit-like) where `values` is a plain ndarray.
    If `data` has a .unit attribute (Quantity), the numeric values are returned
    and the unit is passed through. Otherwise, the unit is None.
    """
    if hasattr(data, "unit"):
        return np.asarray(data.value), getattr(data, "unit")
    return np.asarray(data), None


def _resolve_threshold_numeric(protection_criterion: Any, unit: Any | None) -> float:
    """
    Return a scalar threshold in the same unit/domain as the input data.

    Parameters
    ----------
    protection_criterion : Any
        Threshold expressed as a scalar or quantity-like object.
    unit : Any or None
        Unit-like object associated with the plotted data.

    Returns
    -------
    float
        Numeric threshold in the native data unit/domain.
    """
    if hasattr(protection_criterion, "to") and (unit is not None):
        return float(protection_criterion.to(unit).value)
    if hasattr(protection_criterion, "value"):
        return float(protection_criterion.value)
    return float(protection_criterion)


def _normalize_plotly_html_include_mode(
    include_plotlyjs: bool | str,
    *,
    context: str,
) -> bool | str:
    """
    Validate standalone Plotly HTML export mode for wrapper-level file exports.

    Parameters
    ----------
    include_plotlyjs : bool or str
        Requested Plotly JavaScript inclusion mode.
    context : str
        Human-readable export context used in error messages.

    Returns
    -------
    bool or str
        Normalized mode accepted by ``plotly.io.write_html``.

    Raises
    ------
    ValueError
        Raised when the requested mode would not produce a usable standalone
        HTML export in this wrapper.
    """
    if include_plotlyjs is True:
        return True
    if include_plotlyjs is False:
        raise ValueError(
            f"{context} does not support html_include_plotlyjs=False because that "
            "mode omits plotly.js entirely and only works when embedded into a page "
            "that already loads Plotly. Use True for bundled standalone HTML or "
            "'cdn' for CDN-backed standalone HTML."
        )
    if isinstance(include_plotlyjs, str):
        mode = include_plotlyjs.strip().lower()
        if mode == "cdn":
            return "cdn"
        raise ValueError(
            f"{context} only supports html_include_plotlyjs=True or 'cdn' for "
            f"standalone HTML export; got {include_plotlyjs!r}."
        )
    raise ValueError(
        f"{context} only supports html_include_plotlyjs=True or 'cdn' for "
        f"standalone HTML export; got {include_plotlyjs!r}."
    )


def _nearest_rank_percentile_axis0(values: np.ndarray, p: float) -> np.ndarray:
    """
    Return axis-0 percentiles using nearest-rank selection on finite samples.

    Parameters
    ----------
    values : ndarray, shape (n_samples, n_series)
        Sample matrix to process column-wise.
    p : float
        Percentile in the inclusive range [0, 100].

    Returns
    -------
    ndarray
        Percentile value per column. Columns with no finite samples are filled
        with ``np.nan``.

    Raises
    ------
    ValueError
        Raised when ``values`` is not two-dimensional or ``p`` is outside the
        valid percentile range.

    Notes
    -----
    This returns an observed sample rather than an interpolated value. For dB-
    valued power data that keeps percentile selection exactly invariant under a
    monotonic dB-to-linear transform.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of samples; got shape {arr.shape}.")
    if not (0.0 <= p <= 100.0):
        raise ValueError("Percentile p must be in [0, 100].")

    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=0, dtype=np.int64)
    masked = np.where(finite, arr, np.inf)
    sorted_vals = np.sort(masked, axis=0)

    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    valid_cols = counts > 0
    if not np.any(valid_cols):
        return out

    cols = np.nonzero(valid_cols)[0]
    ranks = np.ceil((p / 100.0) * counts[valid_cols]).astype(np.int64) - 1
    ranks = np.clip(ranks, 0, counts[valid_cols] - 1)
    out[valid_cols] = sorted_vals[ranks, cols]
    return out


def _finite_exceedance_percent_axis0(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Return finite-sample exceedance percentages column-wise.

    Parameters
    ----------
    values : ndarray, shape (n_samples, n_series)
        Sample matrix to process column-wise.
    threshold : float
        Threshold in the same native domain/unit as ``values``.

    Returns
    -------
    ndarray
        Percentage of finite samples exceeding ``threshold`` in each column.
        Columns with no finite samples are filled with ``np.nan``.

    Raises
    ------
    ValueError
        Raised when ``values`` is not two-dimensional.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of samples; got shape {arr.shape}.")

    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=0, dtype=np.int64)
    hits = np.sum(finite & (arr > float(threshold)), axis=0, dtype=np.int64)

    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    valid_cols = counts > 0
    out[valid_cols] = hits[valid_cols].astype(np.float64) * 100.0 / counts[valid_cols]
    return out


# S.1586 azimuth step per ring (lower elevation edge in degrees → az step)
_S1586_AZ_STEPS: Dict[int, int] = {
    0: 3, 3: 3, 6: 3, 9: 3, 12: 3, 15: 3, 18: 3, 21: 3, 24: 3, 27: 3,
    30: 4, 33: 4, 36: 4, 39: 4, 42: 4, 45: 4,
    48: 5, 51: 5, 54: 5,
    57: 6, 60: 6, 63: 6,
    66: 8, 69: 9, 72: 10, 75: 12, 78: 18, 81: 24, 84: 40, 87: 120,
}

# Pre-built azimuth step array for fast / JITted paths (ring order: 0..27° → 87–90°)
_S1586_AZ_STEPS_ARR = np.array([_S1586_AZ_STEPS[e] for e in range(0, 90, 3)], dtype=np.int64)

# Optional numba acceleration (safe no-op decorator when missing)
_HAS_NUMBA = importlib.util.find_spec("numba") is not None
if _HAS_NUMBA:
    from numba import njit  # type: ignore
else:
    def njit(*args, **kwargs):  # type: ignore
        """Fallback no-op decorator used when Numba is unavailable."""
        def _decorator(func):
            """Return the original function unchanged."""
            return func
        return _decorator
# -----------------------------------------------------------------------------
# HPC helpers for ECDF/CCDF (avoid full sort + avoid full-size masks)
# -----------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _hist_counts_numba(x: np.ndarray, xmin: float, inv_dx: float, bins: int):
    """
    Build a fixed-bin histogram in ONE pass without allocating a global finite-mask.

    Parameters
    ----------
    x : ndarray
        1D view of samples (may be strided).
    xmin : float
        Lower bound of histogram range.
    inv_dx : float
        1 / bin_width.
    bins : int
        Number of bins.

    Returns
    -------
    counts : int64[bins]
        Histogram counts.
    n_finite : int
        Number of finite samples processed.

    Notes
    -----
    - This is deliberately simple: a single loop with isfinite() and integer binning.
    - It is *much* lighter on memory than `mask = np.isfinite(x)` for huge x.
    """
    counts = np.zeros(bins, dtype=np.int64)
    n_finite = 0

    for i in range(x.size):
        v = x[i]
        if np.isfinite(v):
            n_finite += 1
            k = int((v - xmin) * inv_dx)

            # Clamp to [0, bins-1] to avoid branchy range-check logic
            if k < 0:
                k = 0
            elif k >= bins:
                k = bins - 1

            counts[k] += 1

    return counts, n_finite


@njit(cache=True, fastmath=True)
def _percentile_from_hist_numba(edges: np.ndarray, cdf_counts: np.ndarray, p: float) -> float:
    """
    Approximate a percentile from histogram CDF with linear interpolation inside a bin.

    edges      : float64[bins+1]
    cdf_counts : int64[bins]  (monotonic cumulative sum)
    p          : percentile in [0, 100]
    """
    if p < 0.0:
        p = 0.0
    if p > 100.0:
        p = 100.0

    n = int(cdf_counts[-1])
    if n <= 0:
        return np.nan

    target = (p / 100.0) * n

    # searchsorted in numba: manual loop (bins is small ~4096-16384)
    j = 0
    while j < cdf_counts.size and cdf_counts[j] < target:
        j += 1
    if j >= cdf_counts.size:
        j = cdf_counts.size - 1

    left_cum = 0 if j == 0 else int(cdf_counts[j - 1])
    in_bin = int(cdf_counts[j] - left_cum)
    if in_bin <= 0:
        return float(edges[j])

    frac = (target - left_cum) / in_bin
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0

    return float(edges[j] + frac * (edges[j + 1] - edges[j]))


@lru_cache(maxsize=1)
def _s1586_cells() -> tuple[np.ndarray, ...]:
    """
    Build the *reference* S.1586-1 upper-hemisphere grid (2334 rectangular cells).

    Rings are 3° thick: edges at 0, 3, ..., 90 (30 rings). Each ring has its own
    azimuth step size per the Recommendation’s table.

    Returns
    -------
    az_lo, az_hi, el_lo, el_hi : ndarray of float64 (len=2334)
        Per-cell lower/upper azimuth and elevation bounds in degrees.
    el_edges : ndarray of float64 (len=31)
        Ring edges for elevation (0..90 in steps of 3).
    cells_per_ring : ndarray of int (len=30)
        Number of cells in each ring, from 0–3° up to 87–90°.
    """
    el_edges = np.arange(0, 90 + 3, 3, dtype=np.float64)
    n_rings = el_edges.size - 1
    
    # Pre-compute expected cell counts and validate total
    expected = [120]*10 + [90]*6 + [72]*3 + [60]*3 + [45, 40, 36, 30, 20, 15, 9, 3]
    total_cells = sum(expected)  # 2334
    
    # Pre-allocate arrays for better performance
    az_lo = np.empty(total_cells, dtype=np.float64)
    az_hi = np.empty(total_cells, dtype=np.float64)
    el_lo = np.empty(total_cells, dtype=np.float64)
    el_hi = np.empty(total_cells, dtype=np.float64)
    cells_per_ring = np.empty(n_rings, dtype=np.int32)

    idx = 0
    for i in range(n_rings):
        el0 = float(el_edges[i])
        el1 = float(el_edges[i + 1])
        step = _S1586_AZ_STEPS.get(int(el0))
        if step is None or (360 % step) != 0:
            raise RuntimeError(f"Bad azimuth step at ring starting {el0}°.")
        n = 360 // step
        cells_per_ring[i] = n
        az_edges = np.arange(n + 1, dtype=np.float64) * step
        
        # Vectorized assignment instead of loop appends
        az_lo[idx:idx+n] = az_edges[:-1]
        az_hi[idx:idx+n] = az_edges[1:]
        el_lo[idx:idx+n] = el0
        el_hi[idx:idx+n] = el1
        idx += n

    # Guardrail against accidental changes: row counts must match the table.
    if not np.array_equal(cells_per_ring, expected) or idx != total_cells:
        raise RuntimeError("S.1586 ring/cell construction mismatch.")

    return (
        az_lo, az_hi,
        el_lo, el_hi,
        el_edges, cells_per_ring,
    )


def _cart_from_azel(az_deg: np.ndarray | float,
                    el_deg: np.ndarray | float,
                    r: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (azimuth, elevation) in degrees to Cartesian on a sphere of radius `r`.
    Azimuth: 0°→+x, 90°→+y. Elevation: 0° at horizon, 90° at zenith.
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def _eye_from_elev_azim(elev_deg: float, azim_deg: float, distance: float) -> np.ndarray:
    """
    Camera eye position (Plotly style) from Matplotlib's (elev, azim).
    """
    th = np.radians(azim_deg)
    ph = np.radians(elev_deg)
    x = distance * np.cos(ph) * np.cos(th)
    y = distance * np.cos(ph) * np.sin(th)
    z = distance * np.sin(ph)
    return np.array([x, y, z], dtype=float)


def _rgba_from_value(v: float,
                     cmin: float,
                     cmax: float,
                     cmap_name: str,
                     alpha_val: float = 1.0) -> tuple[str, tuple[float, float, float]]:
    """
    Map a numeric value onto a Matplotlib colormap and return:

    - CSS-like rgba string "rgba(r,g,b,a)" with 0..255 channels,
    - The same colour as an (r,g,b) triple in 0..1 for contrast decisions.
    """
    if not np.isfinite(v) or cmax == cmin:
        t = 0.0
    else:
        t = (v - cmin) / max(cmax - cmin, 1e-12)
        t = float(np.clip(t, 0.0, 1.0))
    cmap_obj = plt.get_cmap(cmap_name)
    r, g, b, a = cmap_obj(t)
    a = float(alpha_val) * float(a)
    return f"rgba({int(round(r*255))},{int(round(g*255))},{int(round(b*255))},{a:.3f})", (r, g, b)


def _hover_font_color_from_rgb(rgb: tuple[float, float, float]) -> str:
    """
    Choose black/white text for a coloured hover box using simple luminance.
    """
    r, g, b = rgb
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if y > 0.60 else "white"


# --- Polar mapping used by the 2D "polar" projection ---

def _r_base_from_el(el_deg: np.ndarray | float, mapping: str = "equal_area"):
    """
    Convert elevation to a base polar radius in [0..1]:
    - "equal_area": Lambert azimuthal equal-area (disc area ~ solid angle)
    - "linear":     linear in zenith angle (simple radius)
    """
    theta = np.radians(90.0 - np.asarray(el_deg, float))  # zenith angle
    return (np.sqrt(2.0) * np.sin(theta / 2.0)) if mapping == "equal_area" else (theta / (0.5 * np.pi))


def _el_from_r_base(r: np.ndarray | float, mapping: str = "equal_area"):
    """
    Inverse of _r_base_from_el.
    """
    r = np.asarray(r, float)
    if mapping == "equal_area":
        theta = 2.0 * np.arcsin(np.clip(r / np.sqrt(2.0), 0.0, 1.0))
    else:
        theta = r * 0.5 * np.pi
    return 90.0 - np.degrees(theta)


def _r_from_el(el_deg, mapping="equal_area", invert=False):
    """
    Display radius from elevation. If `invert=True`, zenith is at the rim and
    horizon at the centre.
    """
    r = _r_base_from_el(el_deg, mapping)
    return (1.0 - r) if invert else r


def _el_from_r_display(r_disp, mapping="equal_area", invert=False):
    """
    Inverse of _r_from_el for display radii.
    """
    r_base = (1.0 - r_disp) if invert else r_disp
    return _el_from_r_base(r_base, mapping)


@lru_cache(maxsize=8)
def _s1586_polar_heatmap_lookup(
    raster_res: int,
    radial_mapping: str,
    invert_polar: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return cached raster lookup arrays for full-grid S.1586 polar heatmaps.

    Parameters
    ----------
    raster_res : int
        Base square raster size used for the Plotly heatmap.
    radial_mapping : {"equal_area", "linear"}
        Polar-radius mapping used by the display.
    invert_polar : bool
        If True, place zenith at the rim and horizon at the centre.

    Returns
    -------
    x : ndarray of float64, shape (N,)
        X-axis coordinates for the heatmap trace.
    y : ndarray of float64, shape (N,)
        Y-axis coordinates for the heatmap trace.
    disc : ndarray of bool, shape (N, N)
        Mask selecting pixels inside the unit-disc projection.
    cell_idx : ndarray of int64, shape (N, N)
        Canonical S.1586 cell index for each raster pixel.
    """
    n = max(256, int(raster_res))
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    x_grid, y_grid = np.meshgrid(x, y)
    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    disc = r_grid <= 1.0

    el = _el_from_r_display(r_grid, radial_mapping, invert=invert_polar)
    azimuth_deg = (90.0 - np.degrees(np.arctan2(y_grid, x_grid))) % 360.0

    _, _, _, _, el_edges, cells_per_ring = _s1586_cells()
    ring_idx = np.digitize(el, el_edges, right=True) - 1
    ring_idx = np.clip(ring_idx, 0, len(el_edges) - 2)
    steps = 360.0 / np.take(cells_per_ring, ring_idx)
    az_bin = (azimuth_deg // steps).astype(np.int64)
    az_bin = np.clip(az_bin, 0, np.take(cells_per_ring, ring_idx) - 1)
    offsets = np.concatenate([[0], np.cumsum(cells_per_ring[:-1])]).astype(np.int64, copy=False)
    cell_idx = np.take(offsets, ring_idx) + az_bin

    return x, y, disc, cell_idx


def _mpl_backend(interactive_flag: bool | None) -> None:
    """
    In notebooks, switch Matplotlib backend if requested:
    - True  → widget (or inline fallback)
    - False → inline
    - None  → leave as-is
    """
    try:
        from IPython import get_ipython  # lazy import
        ip = get_ipython()
        if ip is None:
            return
        if interactive_flag is True:
            try:
                ip.run_line_magic("matplotlib", "widget")
            except Exception:
                ip.run_line_magic("matplotlib", "inline")
        elif interactive_flag is False:
            ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass


def _subset_indices_from_grid_info(gi: np.ndarray) -> np.ndarray:
    """
    Convert a clipped subset described by `grid_info` back to canonical 0..2333 indices.

    The subset is assumed to be aligned to S.1586 rings (3° elevation) and azimuth
    bins per ring. We use the subset's *lower* elevation to pick a ring and the
    subset's lower azimuth to pick an azimuth bin in that ring.

    Returns
    -------
    idx : ndarray of int
        Index into the 2334 reference grid for each subset row.
    """
    el_lo_clip = gi["cell_lat_low"].astype(np.float64)
    az_lo = gi["cell_lon_low"].astype(np.float64)
    _, _, _, _, el_edges, cells_per_ring = _s1586_cells()

    # Fast numba path when available (particularly useful for large subsets)
    if _HAS_NUMBA:
        return _subset_indices_from_grid_info_numba(
            el_lo_clip, az_lo, _S1586_AZ_STEPS_ARR, cells_per_ring.astype(np.int64)
        )

    # Pure-numpy fallback
    ring_idx = np.clip((np.floor(el_lo_clip / 3.0)).astype(int), 0, len(el_edges) - 2)
    step = _S1586_AZ_STEPS_ARR[ring_idx]
    n_in_ring = cells_per_ring[ring_idx]
    offsets = np.concatenate([[0], np.cumsum(cells_per_ring[:-1])])
    az_bin = np.floor(az_lo / step).astype(int) % n_in_ring
    return offsets[ring_idx] + az_bin


@njit(cache=True)
def _subset_indices_from_grid_info_numba(
    el_lo_clip: np.ndarray,
    az_lo: np.ndarray,
    step_per_ring: np.ndarray,
    cells_per_ring: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated mapping of subset grid_info rows to canonical indices."""
    n = el_lo_clip.shape[0]
    out = np.empty(n, dtype=np.int64)

    offsets = np.empty(cells_per_ring.size + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(cells_per_ring.size - 1):
        offsets[i + 1] = offsets[i] + cells_per_ring[i]
    offsets[-1] = offsets[-2] + cells_per_ring[-1]

    for i in range(n):
        ring_idx = int(min(np.floor(el_lo_clip[i] / 3.0), 29.0))
        step = step_per_ring[ring_idx]
        n_in_ring = cells_per_ring[ring_idx]
        az_bin = int(np.floor(az_lo[i] / step)) % n_in_ring
        out[i] = offsets[ring_idx] + az_bin

    return out


# -----------------------------------------------------------------------------
# CDF / CCDF
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CDF / CCDF (HPC helpers)
# -----------------------------------------------------------------------------

def _iter_1d_chunks(x: np.ndarray, chunk_size: int):
    """
    Yield 1D views of `x` in chunks to avoid allocating huge masks/copies.

    Notes
    -----
    - `x` is expected to be 1D (typically a view of a flattened higher-D array).
    - Chunking keeps peak memory low while still using fast NumPy kernels per chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    n = x.size
    for i0 in range(0, n, chunk_size):
        yield x[i0:i0 + chunk_size]


def _hist_ecdf_streaming(
    x_flat: np.ndarray,
    *,
    bins: int,
    x_range: tuple[float, float] | None,
    assume_finite: bool,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build an approximate ECDF using a fixed-bin histogram accumulated in chunks.

    Returns
    -------
    edges : ndarray, shape (bins+1,)
        Histogram bin edges in the native domain (typically dB).
    cdf_counts : ndarray, shape (bins,)
        Cumulative counts per bin (monotonic, last value == n_finite).
    n_finite : int
        Number of finite samples processed.

    Why this exists
    ---------------
    Sorting N~1e8 points is prohibitive. Histogram ECDF is O(N) and memory O(bins).
    """
    if bins < 16:
        raise ValueError("bins is too small; use at least ~256 for smooth curves.")

    # Determine histogram range robustly, without allocating a giant mask.
    if x_range is not None:
        xmin, xmax = float(x_range[0]), float(x_range[1])
    else:
        if assume_finite:
            xmin = float(np.min(x_flat))
            xmax = float(np.max(x_flat))
        else:
            # nanmin/nanmax skip NaNs and do not require explicit masking.
            xmin = float(np.nanmin(x_flat))
            xmax = float(np.nanmax(x_flat))

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError("Non-finite x_range/min/max; check your input (NaNs/Infs).")
    if xmax == xmin:
        # Degenerate distribution → widen just enough to build bins.
        xmax = xmin + 1e-9

    edges = np.linspace(xmin, xmax, bins + 1, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)

    # Accumulate histogram in chunks to avoid huge intermediate arrays.
    n_finite = 0
    for chunk in _iter_1d_chunks(x_flat, chunk_size):
        if assume_finite:
            v = chunk
        else:
            # Chunk-level filtering only (bounded memory).
            m = np.isfinite(chunk)
            if not np.any(m):
                continue
            v = chunk[m]

        # NOTE: numpy histogram is highly optimized in C; this is usually faster
        # than a naive numba loop for large arrays.
        h, _ = np.histogram(v, bins=edges)
        counts += h
        n_finite += int(v.size)

    if n_finite == 0:
        raise ValueError("No finite samples to plot.")

    cdf_counts = np.cumsum(counts, dtype=np.int64)
    return edges, cdf_counts, n_finite


def _percentile_from_hist(edges: np.ndarray, cdf_counts: np.ndarray, p: float) -> float:
    """
    Approximate percentile value from histogram CDF via linear interpolation in-bin.

    Parameters
    ----------
    edges : (bins+1,) float
    cdf_counts : (bins,) int
    p : float in [0, 100]

    Returns
    -------
    x_p : float
        Approximate p-th percentile.
    """
    if not (0.0 <= p <= 100.0):
        raise ValueError("Percentile p must be in [0, 100].")

    n = int(cdf_counts[-1])
    if n <= 0:
        return float("nan")

    # Target rank in [1..n]. We use "ceil" behaviour consistent with many ECDF defs.
    target = (p / 100.0) * n
    # First bin where CDF >= target
    j = int(np.searchsorted(cdf_counts, target, side="left"))
    j = max(0, min(j, cdf_counts.size - 1))

    left_cum = 0 if j == 0 else int(cdf_counts[j - 1])
    in_bin = int(cdf_counts[j] - left_cum)
    if in_bin <= 0:
        # Empty bin edge case: return left edge.
        return float(edges[j])

    frac = (target - left_cum) / in_bin
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(edges[j] + frac * (edges[j + 1] - edges[j]))


def _skycell_ccdf_corridor_exact(
    x2d: np.ndarray,
    *,
    bins: int | None = None,
    x_range: tuple[float, float] | None = None,
    x_support: np.ndarray | None = None,
    assume_finite: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None, int]:
    """
    Build an exact CCDF corridor across skycells on fixed x-bin lower edges.

    Parameters
    ----------
    x2d : ndarray, shape (n_samples, n_skycells)
        Sample matrix where each column corresponds to one skycell.
    bins : int, optional
        Number of x bins used to evaluate CCDF(x) on lower edges when
        ``x_support`` is not provided.
    x_range : (float, float), optional
        Inclusive plotting range (xmin, xmax) for corridor support when
        ``x_support`` is not provided.
    x_support : ndarray, optional
        Explicit lower-edge x support for CCDF evaluation. When provided, the
        corridor is evaluated exactly on these x positions instead of building
        its own fixed grid.
    assume_finite : bool
        If True, assumes each skycell vector is finite already.

    Returns
    -------
    x_edges_low : (bins,) float64
        Lower edges used for step="pre" corridor plotting.
    ccdf_min : (bins,) float64
        Minimum CCDF across skycells at each x edge.
    ccdf_max : (bins,) float64
        Maximum CCDF across skycells at each x edge.
    p98_minmax : tuple[float, float] or None
        Exact min/max of per-skycell 98th percentile values, if available.
    max_n_finite : int
        Largest finite sample count among all skycells.

    Notes
    -----
    Per-skycell CCDF values are computed exactly at each x edge via
    ``searchsorted`` on sorted cell values. This preserves very small tail
    probabilities down to each cell's native step size (1 / n_cell).
    """
    if x2d.ndim != 2:
        raise ValueError(f"x2d must be 2D with shape (samples, skycells); got {x2d.shape}.")

    if x_support is not None:
        x_edges_low = np.asarray(x_support, dtype=np.float64).reshape(-1)
        if x_edges_low.size < 1:
            raise ValueError("skycell corridor x_support must contain at least one x position.")
        if not np.all(np.isfinite(x_edges_low)):
            raise ValueError("skycell corridor x_support must be finite.")
        if x_edges_low.size > 1 and np.any(np.diff(x_edges_low) <= 0.0):
            raise ValueError("skycell corridor x_support must be strictly increasing.")
        bins = int(x_edges_low.size)
    else:
        if bins is None or x_range is None:
            raise ValueError("Provide either x_support or both bins and x_range for the skycell corridor.")
        bins = int(bins)
        if bins < 16:
            raise ValueError("skycell corridor bins is too small; use at least 64.")

        xmin, xmax = float(x_range[0]), float(x_range[1])
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            raise ValueError("skycell corridor range must be finite.")
        if xmax <= xmin:
            raise ValueError(f"skycell corridor x_range must satisfy xmax > xmin (got {x_range}).")

        edges = np.linspace(xmin, xmax, bins + 1, dtype=np.float64)
        x_edges_low = edges[:-1]

    n_sky = int(x2d.shape[1])
    ccdf_min = np.full(bins, np.inf, dtype=np.float64)
    ccdf_max = np.full(bins, -np.inf, dtype=np.float64)
    p98_vals = np.full(n_sky, np.nan, dtype=np.float64)

    valid_cells = 0
    max_n_finite = 0

    for c in range(n_sky):
        v = x2d[:, c]
        if assume_finite:
            v_cell = np.asarray(v, dtype=np.float64)
            if not np.all(np.isfinite(v_cell)):
                v_cell = v_cell[np.isfinite(v_cell)]
        else:
            m = np.isfinite(v)
            if not np.any(m):
                continue
            v_cell = np.asarray(v[m], dtype=np.float64)

        n_c = int(v_cell.size)
        if n_c <= 0:
            continue

        valid_cells += 1
        if n_c > max_n_finite:
            max_n_finite = n_c

        v_sorted = np.sort(v_cell)
        idx = np.searchsorted(v_sorted, x_edges_low, side="left")
        ccdf_c = 1.0 - (idx.astype(np.float64) / float(n_c))
        ccdf_min = np.minimum(ccdf_min, ccdf_c)
        ccdf_max = np.maximum(ccdf_max, ccdf_c)

        k = int(np.ceil(0.98 * n_c)) - 1
        k = max(0, min(k, n_c - 1))
        p98_vals[c] = float(v_sorted[k])

    if valid_cells == 0:
        raise ValueError("No finite skycell samples available for CCDF corridor.")

    p98_min = float(np.nanmin(p98_vals))
    p98_max = float(np.nanmax(p98_vals))
    return x_edges_low, ccdf_min, ccdf_max, (p98_min, p98_max), max_n_finite



def _dedup_legend_labels(labels: List[str]) -> List[str]:
    """Make legend labels unique by appending '(N)' suffixes for duplicates."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        out.append(lab if seen[lab] == 1 else f"{lab} ({seen[lab]})")
    return out


def _normalize_distribution_protection_lines(
    *,
    prot_value: Any | List[Any] | None,
    prot_legend: List[str] | None,
    prot_colors: List[str] | None,
    unit: Any,
) -> tuple[List[float], List[str], List[Any]]:
    """Normalize protection thresholds, labels, and colors for distribution plots."""
    if prot_value is None:
        prot_list: List[Any] = []
    elif isinstance(prot_value, (list, tuple, np.ndarray)):
        prot_list = list(prot_value)
    else:
        prot_list = [prot_value]

    if prot_legend is None:
        prot_legend = ["Protection limit"] * len(prot_list)
    else:
        if len(prot_legend) < len(prot_list):
            prot_legend = prot_legend + ["Protection limit"] * (len(prot_list) - len(prot_legend))
        elif len(prot_legend) > len(prot_list):
            prot_legend = prot_legend[:len(prot_list)]
    prot_legend = _dedup_legend_labels(prot_legend)

    prot_x_vals: List[float] = []
    for pv in prot_list:
        if hasattr(pv, "to") and (unit is not None):
            prot_x_vals.append(float(pv.to(unit).value))
        elif hasattr(pv, "value"):
            prot_x_vals.append(float(pv.value))
        else:
            prot_x_vals.append(float(pv))

    if prot_colors is None or len(prot_colors) == 0:
        cmap = plt.get_cmap("tab10")
        prot_line_colors = [cmap(i % cmap.N) for i in range(len(prot_x_vals))]
    else:
        prot_line_colors = [prot_colors[i % len(prot_colors)] for i in range(len(prot_x_vals))]

    return prot_x_vals, prot_legend, prot_line_colors


def _fraction_floor(percent_value: float | None, *, empirical_floor: float, log_scale: bool) -> float:
    """Return a plotting floor in fractional units, honoring explicit inputs."""
    if percent_value is None:
        return empirical_floor
    floor = min(max(float(percent_value) / 100.0, 0.0), 1.0)
    if log_scale:
        return max(np.finfo(np.float64).tiny, floor)
    return floor


def _ensure_log_floor_tick(ax: Any, floor: float) -> None:
    """Ensure an explicit log-axis floor appears in the major tick set."""
    ticks = np.asarray(ax.get_yticks(), dtype=np.float64)
    if ticks.size == 0:
        ticks = np.array([floor, 1.0], dtype=np.float64)
    else:
        ticks = ticks[np.isfinite(ticks)]
        ticks = ticks[(ticks >= floor) & (ticks <= 1.0)]
        if ticks.size == 0:
            ticks = np.array([floor, 1.0], dtype=np.float64)

    if not np.any(np.isclose(ticks, floor, rtol=1.0e-10, atol=np.finfo(np.float64).tiny)):
        ticks = np.sort(np.append(ticks, floor))

    ax.yaxis.set_major_locator(mtick.FixedLocator(ticks.tolist()))


def _style_distribution_axis(ax: Any) -> None:
    """Apply a light-weight house style for readable distribution plots."""
    ax.set_facecolor("#fbfdff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155")
    ax.xaxis.label.set_color("#0f172a")
    ax.yaxis.label.set_color("#0f172a")
    ax.margins(x=0.02)


def _plot_cdf_ccdf_precomputed(
    *,
    x_plot: np.ndarray,
    y_cdf: np.ndarray,
    x_ccdf: np.ndarray,
    y_ccdf: np.ndarray,
    n: int,
    unit: Any,
    p95: float | None,
    p98: float | None,
    plot_type: str,
    show_two_percent: bool,
    show_five_percent: bool,
    prot_value: Any | List[Any] | None,
    prot_legend: List[str] | None,
    prot_colors: List[str] | None,
    title: str | None,
    figsize: tuple[float, float],
    line_width: float,
    line_alpha: float,
    cdf_color: str,
    ccdf_color: str,
    marker_size: float,
    grid: bool,
    legend_outside: bool,
    show_skycell_corridor: bool,
    skycell_corridor_alpha: float,
    skycell_corridor_color: str,
    skycell_corridor_label: str,
    show_skycell_p98_note: bool,
    y_percent_decimals: int,
    ccdf_ymin_pct: float | None,
    cdf_ymin_pct: float | None,
    y_log: bool,
    show: bool,
    return_values: bool,
    x_label: str | None,
    show_margin: bool,
    margin_at: str,
    save_path: str | None,
    save_dpi: int,
    corridor_x: np.ndarray | None = None,
    corridor_ccdf_min: np.ndarray | None = None,
    corridor_ccdf_max: np.ndarray | None = None,
    corridor_p98_range: tuple[float, float] | None = None,
    corridor_max_n: int = 0,
    corridor_x_right: float | None = None,
    method_name: str = "unknown",
) -> Figure | tuple[Figure, dict[str, Any]]:
    """Render CDF/CCDF plots from precomputed support curves."""
    if int(n) <= 0:
        raise ValueError("Sample count must be positive.")

    p95_q = (p95 * unit) if (p95 is not None and unit is not None) else p95
    p98_q = (p98 * unit) if (p98 is not None and unit is not None) else p98
    prot_x_vals, prot_legend, prot_line_colors = _normalize_distribution_protection_lines(
        prot_value=prot_value,
        prot_legend=prot_legend,
        prot_colors=prot_colors,
        unit=unit,
    )

    pt = plot_type.lower()
    if pt not in {"cdf", "ccdf", "both"}:
        raise ValueError("plot_type must be 'cdf', 'ccdf', or 'both'.")

    if pt == "both":
        fig, (ax_cdf, ax_ccdf) = _new_mpl_subplots(1, 2, figsize=figsize, sharex=True)
    elif pt == "cdf":
        fig, ax_cdf = _new_mpl_subplots(1, 1, figsize=figsize)
        ax_ccdf = None
    else:
        fig, ax_ccdf = _new_mpl_subplots(1, 1, figsize=figsize)
        ax_cdf = None

    xlab = str(x_label) if x_label is not None else ("Value" + (f" [{unit}]" if unit is not None else ""))
    percent_fmt = mtick.PercentFormatter(xmax=1.0, decimals=y_percent_decimals)
    y_decimals = max(0, int(y_percent_decimals))
    sci_switch_pct = 0.5 * (10.0 ** (-y_decimals))

    def _percent_or_sci_formatter(y: float, _pos: int) -> str:
        pct = 100.0 * float(y)
        if not np.isfinite(pct):
            return ""
        if abs(pct) <= np.finfo(np.float64).tiny:
            return "0%"
        if abs(pct) < sci_switch_pct:
            return f"{pct:.0e}%"
        return f"{pct:.{y_decimals}f}%"

    if show_skycell_corridor and ax_ccdf is None:
        warnings.warn("show_skycell_corridor=True is ignored when plot_type='cdf'.")

    if ax_cdf is not None:
        ax = ax_cdf
        _style_distribution_axis(ax)
        ax.step(np.asarray(x_plot, dtype=np.float64), np.asarray(y_cdf, dtype=np.float64), where="post", linewidth=line_width, alpha=line_alpha, color=cdf_color, label="CDF")
        if grid:
            ax.grid(True, color="#9ca3af", alpha=0.4, linewidth=0.7)
        if show_five_percent:
            ax.axhline(0.95, color="#9ca3af", linestyle="--", linewidth=1.6, label="95% (5% worst)")
            if p95 is not None and np.isfinite(p95):
                ax.scatter([p95], [0.95], s=marker_size, color=cdf_color, zorder=5)
                ax.annotate(f"{p95_q:.3f}" if unit is not None else f"{p95:.3f}", xy=(p95, 0.95), xytext=(5, 8), textcoords="offset points", fontsize=9, color=cdf_color)
        if show_two_percent:
            ax.axhline(0.98, color="#6b7280", linestyle="--", linewidth=1.6, label="98% (2% worst)")
            if p98 is not None and np.isfinite(p98):
                ax.scatter([p98], [0.98], s=marker_size, color=cdf_color, zorder=5)
                ax.annotate(f"{p98_q:.3f}" if unit is not None else f"{p98:.3f}", xy=(p98, 0.98), xytext=(5, 8), textcoords="offset points", fontsize=9, color=cdf_color)
        for xv, lab, col in zip(prot_x_vals, prot_legend, prot_line_colors):
            ax.axvline(xv, color=col, linestyle="-.", linewidth=1.6, label=lab, alpha=0.95)
        ax.set_xlabel(xlab)
        ax.set_ylabel("CDF")
        if y_log:
            ymin = 1.0 / int(n)
            if cdf_ymin_pct is not None:
                ymin = max(ymin, min(max(cdf_ymin_pct / 100.0, 0.0), 1.0))
            ax.set_yscale("log")
            ax.set_ylim(ymin, 1.0)
        else:
            ymin = 0.0 if cdf_ymin_pct is None else min(max(cdf_ymin_pct / 100.0, 0.0), 1.0)
            ax.set_ylim(ymin, 1.0)
        ax.yaxis.set_major_formatter(percent_fmt)
        if not legend_outside:
            ax.legend(loc="lower right", frameon=True)

    if ax_ccdf is not None:
        ax = ax_ccdf
        _style_distribution_axis(ax)
        ax.step(np.asarray(x_ccdf, dtype=np.float64), np.asarray(y_ccdf, dtype=np.float64), where="pre", linewidth=line_width, alpha=line_alpha, color=ccdf_color, label="CCDF")
        if grid:
            ax.grid(True, color="#9ca3af", alpha=0.4, linewidth=0.7)
        if corridor_x is not None and corridor_ccdf_min is not None and corridor_ccdf_max is not None:
            if y_log:
                axis_floor = _fraction_floor(ccdf_ymin_pct, empirical_floor=1.0 / int(n), log_scale=True)
                corridor_floor = np.finfo(np.float64).tiny
                if corridor_max_n > 0:
                    corridor_floor = max(corridor_floor, 1.0 / float(corridor_max_n))
                y_floor = max(axis_floor, corridor_floor)
            else:
                axis_floor = _fraction_floor(ccdf_ymin_pct, empirical_floor=0.0, log_scale=False)
                y_floor = axis_floor
            corridor_ccdf_min_plot = np.clip(corridor_ccdf_min, y_floor, 1.0)
            corridor_ccdf_max_plot = np.clip(corridor_ccdf_max, y_floor, 1.0)
            ax.fill_between(corridor_x, corridor_ccdf_min_plot, corridor_ccdf_max_plot, step="pre", alpha=skycell_corridor_alpha, color=skycell_corridor_color, label=skycell_corridor_label)
            if y_log and corridor_x_right is not None and axis_floor < y_floor and corridor_ccdf_max_plot.size > 0:
                tail_hi = float(corridor_ccdf_max_plot[-1])
                tail_lo = float(corridor_ccdf_min_plot[-1])
                tail_tol = max(1.0e-12, 1.0e-9 * max(abs(tail_hi), abs(tail_lo), 1.0))
                if np.isclose(tail_hi, tail_lo, rtol=0.0, atol=tail_tol):
                    tail_x0 = float(corridor_x[-1])
                    tail_x1 = float(corridor_x_right)
                    if tail_x1 > tail_x0:
                        ax.fill_between(np.array([tail_x0, tail_x1], dtype=np.float64), np.array([axis_floor, axis_floor], dtype=np.float64), np.array([tail_hi, tail_hi], dtype=np.float64), step="pre", alpha=skycell_corridor_alpha, color=skycell_corridor_color, label="_nolegend_")
            if show_skycell_p98_note and corridor_p98_range is not None:
                unit_s = f" {unit}" if unit is not None else ""
                p98_min, p98_max = corridor_p98_range
                ax.text(0.02, 0.98, f"Skycell p98 range: [{p98_min:.3f}, {p98_max:.3f}]{unit_s}", transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#d1d5db", alpha=0.95))
        if show_five_percent:
            ax.axhline(0.05, color="#9ca3af", linestyle="--", linewidth=1.6, label="5%")
            if p95 is not None and np.isfinite(p95):
                ax.scatter([p95], [0.05], s=marker_size, color=ccdf_color, zorder=5)
                ax.annotate(f"{p95_q:.3f}" if unit is not None else f"{p95:.3f}", xy=(p95, 0.05), xytext=(5, 8), textcoords="offset points", fontsize=9, color=ccdf_color)
        if show_two_percent:
            ax.axhline(0.02, color="#6b7280", linestyle="--", linewidth=1.6, label="2%")
            if p98 is not None and np.isfinite(p98):
                ax.scatter([p98], [0.02], s=marker_size, color=ccdf_color, zorder=5)
                ax.annotate(f"{p98_q:.3f}" if unit is not None else f"{p98:.3f}", xy=(p98, 0.02), xytext=(5, 8), textcoords="offset points", fontsize=9, color=ccdf_color)
        for xv, lab, col in zip(prot_x_vals, prot_legend, prot_line_colors):
            ax.axvline(xv, color=col, linestyle="-.", linewidth=1.6, label=lab, alpha=0.95)
        if show_margin and prot_x_vals:
            ref = margin_at.lower()
            ref_val = p98 if ref == "p98" else p95
            ref_lbl = "p98 (2% point)" if ref == "p98" else "p95 (5% point)"
            if ref_val is not None and np.isfinite(ref_val):
                unit_s = f" {unit}" if unit is not None else ""
                lines = []
                for lab, xv in zip(prot_legend, prot_x_vals):
                    margin = float(xv - ref_val)
                    need = float(ref_val - xv)
                    lines.append(f"{lab}: margin={margin:+.3f}{unit_s}, need={max(0.0, need):.3f}{unit_s}")
                ax.text(0.02, 0.02, f"Margin vs {ref_lbl}:\n" + "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d1d5db", alpha=0.95))
        ax.set_xlabel(xlab)
        ax.set_ylabel("CCDF")
        if y_log:
            ymin = _fraction_floor(ccdf_ymin_pct, empirical_floor=1.0 / int(n), log_scale=True)
            ax.set_yscale("log")
            ax.set_ylim(ymin, 1.0)
            if ccdf_ymin_pct is not None:
                _ensure_log_floor_tick(ax, ymin)
        else:
            ymin = _fraction_floor(ccdf_ymin_pct, empirical_floor=0.0, log_scale=False)
            ax.set_ylim(ymin, 1.0)
        ccdf_percent_fmt = percent_fmt
        if (100.0 * ymin) < sci_switch_pct:
            ccdf_percent_fmt = mtick.FuncFormatter(_percent_or_sci_formatter)
        ax.yaxis.set_major_formatter(ccdf_percent_fmt)
        if not legend_outside:
            ax.legend(loc="upper right", frameon=True)

    outside_legend = None
    if legend_outside:
        seen: set[str] = set()
        handles_out: List[Any] = []
        labels_out: List[str] = []
        for axis in (ax_cdf, ax_ccdf):
            if axis is None:
                continue
            handles, labels = axis.get_legend_handles_labels()
            for h, lab in zip(handles, labels):
                if lab and lab not in seen:
                    seen.add(lab)
                    handles_out.append(h)
                    labels_out.append(lab)
        if handles_out:
            outside_legend = fig.legend(handles_out, labels_out, loc="center left", bbox_to_anchor=(0.99, 0.5), bbox_transform=fig.transFigure, borderaxespad=0.0, frameon=True)

    base = {"cdf": "CDF", "ccdf": "CCDF", "both": "CDF & CCDF"}[pt]
    ttl = f"{base} — Empirical distribution" if title is None else title
    fig.set_facecolor("white")
    fig.suptitle(ttl, y=0.992, fontsize=14, fontweight="semibold")
    layout_rect = [0.01, 0.01, 0.985, 0.955]
    fig.tight_layout(rect=layout_rect)
    if outside_legend is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_bbox = fig.get_window_extent(renderer=renderer)
        legend_bbox = outside_legend.get_window_extent(renderer=renderer)
        legend_width_frac = float(legend_bbox.width / fig_bbox.width)
        legend_gutter_frac = 0.015
        layout_rect[2] = float(np.clip(0.99 - legend_width_frac - legend_gutter_frac, 0.55, 0.985))
        fig.tight_layout(rect=layout_rect)
        fig.canvas.draw()
        axes_right = max(axis.get_position().x1 for axis in (ax_cdf, ax_ccdf) if axis is not None)
        legend_left = min(0.99 - legend_width_frac, axes_right + legend_gutter_frac)
        outside_legend.set_bbox_to_anchor((legend_left, 0.5), transform=fig.transFigure)
        fig.canvas.draw()

    if save_path is not None:
        fig.savefig(save_path, dpi=int(save_dpi), bbox_inches="tight")

    if show:
        plt.show()
    elif not return_values:
        plt.close(fig)

    info = {
        "unit": unit,
        "p95": p95_q if p95 is not None else None,
        "p98": p98_q if p98 is not None else None,
        "n": int(n),
        "ecdf_method_used": str(method_name),
        "skycell_corridor_p98_min": (corridor_p98_range[0] if corridor_p98_range is not None else None),
        "skycell_corridor_p98_max": (corridor_p98_range[1] if corridor_p98_range is not None else None),
    }
    return (fig, info) if return_values else fig


def plot_cdf_ccdf(
    data: Any,
    *,
    # shape / units / math
    cell_axis: int = -1,          # axis with skycells; others flattened
    # what to plot
    plot_type: str = "both",      # "cdf" | "ccdf" | "both"
    show_two_percent: bool = False,
    show_five_percent: bool = False,
    prot_value: Any | List[Any] | None = None,   # float or Quantity, or list thereof
    prot_legend: List[str] | None = None,        # labels for prot_value(s)
    prot_colors: List[str] | None = None,        # per-line colors
    # visuals
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
    line_width: float = 2.0,
    line_alpha: float = 1.0,
    cdf_color: str = "#2563eb",
    ccdf_color: str = "#16a34a",
    marker_size: float = 60.0,
    grid: bool = True,
    legend_outside: bool = False,              # True -> single figure legend box on the right
    show_skycell_corridor: bool = False,       # CCDF-only min/max corridor across skycells
    skycell_corridor_bins: int = 2048,         # x-bins used for corridor evaluation
    skycell_corridor_x_range: tuple[float, float] | None = None,  # None -> data range
    skycell_corridor_alpha: float = 0.20,
    skycell_corridor_color: str = "#f59e0b",
    skycell_corridor_label: str = "Skycell corridor (min-max)",
    show_skycell_p98_note: bool = False,       # annotate min/max p98 across skycells
    y_percent_decimals: int = 2,                 # % axis decimals for CDF/CCDF
    ccdf_ymin_pct: float | None = None,          # CCDF lower cutoff in percent (0..100)
    cdf_ymin_pct: float | None = None,           # CDF  lower cutoff in percent (0..100)
    y_log: bool = False,                         # log-scale Y axis for tails
    show: bool = True,
    return_values: bool = False,                 # if True → (fig, info_dict)

    # ----------------------------
    # labels / margin / saving
    # ----------------------------
    x_label: str | None = None,                  # override x-axis label ("Value" by default)
    show_margin: bool = False,                   # annotate margin vs protection at p98/p95
    margin_at: str = "p98",                      # "p98" or "p95"
    save_path: str | None = None,                # if set -> fig.savefig(save_path)
    save_dpi: int = 300,

    # ----------------------------
    # performance controls
    # ----------------------------
    ecdf_method: str = "auto",                   # "auto" | "sort" | "hist"
    sort_threshold: int = 2_000_000,             # above this, auto -> histogram
    max_plot_points: int = 250_000,              # decimate sorted plot to this many points
    hist_bins: int = 8192,                       # histogram bins (smoothness)
    hist_range: tuple[float, float] | None = None,  # optional explicit (xmin,xmax)
    assume_finite: bool = False,                 # True if you guarantee no NaN/Inf
):
    """
    Plot empirical CDF and/or CCDF curves for a sample distribution.

    Parameters
    ----------
    data : array-like or Quantity
        Input sample array. One axis must correspond to sky cells when using
        sky-cell corridor overlays; all remaining axes are flattened into the
        sample dimension.
    cell_axis : int, optional
        Axis index corresponding to the sky-cell dimension.
    plot_type : {"cdf", "ccdf", "both"}, optional
        Which curve(s) to render.
    prot_value : scalar, Quantity, or sequence, optional
        One or more protection thresholds to draw as vertical reference lines.
    title : str, optional
        Figure title. When omitted, a default title is generated from the
        chosen plot type.
    return_values : bool, optional
        If True, return ``(fig, info_dict)`` instead of only the figure.
    ecdf_method : {"auto", "sort", "hist"}, optional
        ECDF construction method. ``"sort"`` is exact and best for moderate
        sample counts; ``"hist"`` is an O(N) approximation for very large
        sample counts; ``"auto"`` selects based on ``sort_threshold``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Rendered Matplotlib figure.
    info : dict, optional
        Returned only when ``return_values=True``. Includes selected percentile
        values, unit metadata, sample count, and the effective ECDF method.

    Raises
    ------
    ValueError
        Raised when the input shape, ECDF method, or plotting options are
        inconsistent with the requested operation.

    Notes
    -----
    - Exact sorted-mode percentiles use nearest-rank order statistics. For dB-
      valued power data this is exactly invariant under monotonic dB-to-linear
      conversion.
    - ``ecdf_method="hist"`` is approximate. Percentile accuracy improves with
      ``hist_bins`` and an appropriate ``hist_range``.
    - ``show_skycell_corridor=True`` adds an exact min/max CCDF corridor across
      sky cells. In histogram mode the corridor is evaluated on the same
      lower-edge x support as the main CCDF to avoid tail misalignment.
    - On log-scale CCDF plots, an explicit ``ccdf_ymin_pct`` is honored even
      below the empirical minimum CCDF step. When the corridor reaches its
      native floor earlier, the final flat tail is extended visually down to
      the plotted axis floor.
    - ``legend_outside=True`` collects all legend entries into a single figure-
      level legend box.
    """
    # Prepare data
    arr, unit = _to_plain_array(data)
    if arr.ndim < 2:
        raise ValueError(f"'data' must have at least 2 dims; got {arr.shape}.")
    if cell_axis < 0:
        cell_axis = arr.ndim + cell_axis
    if not (0 <= cell_axis < arr.ndim):
        raise ValueError(f"cell_axis {cell_axis} out of range for shape {arr.shape}.")

    # Move skycells to last axis and flatten everything else.
    # IMPORTANT: order="K" maximises the chance this is a view (no giant copy).
    arr = np.moveaxis(arr, cell_axis, -1)
    samples_all = arr.ravel(order="K")

    n_total = int(samples_all.size)
    method = ecdf_method.lower()
    if method not in {"auto", "sort", "hist"}:
        raise ValueError("ecdf_method must be 'auto', 'sort', or 'hist'.")

    if method == "auto":
        method = "sort" if n_total <= int(sort_threshold) else "hist"
    if method == "sort" and n_total > int(sort_threshold):
        warnings.warn(
            f"ecdf_method='sort' with N={n_total:,} is likely very slow; falling back to 'hist'."
        )
        method = "hist"

    # ------------------------------------------------------------------
    # Build ECDF curve and percentiles
    # ------------------------------------------------------------------
    p95 = None
    p98 = None
    x_data_min = float("nan")
    x_data_max = float("nan")

    if method == "sort":
        # Small-N exact mode: boolean mask is acceptable here.
        if assume_finite:
            x_native = np.asarray(samples_all, dtype=np.float64)
        else:
            m = np.isfinite(samples_all)
            x_native = np.asarray(samples_all[m], dtype=np.float64)

        n = int(x_native.size)
        if n == 0:
            raise ValueError("No finite samples to plot.")

        # Sort in-place to reduce allocations
        x_sorted = x_native
        x_sorted.sort()
        x_data_min = float(x_sorted[0])
        x_data_max = float(x_sorted[-1])

        # Percentiles in native domain using nearest-rank selection.
        def _q_from_sorted(p: float) -> float:
            """Return percentile `p` from sorted samples using rank-based indexing."""
            k = int(np.ceil((p / 100.0) * n)) - 1
            k = max(0, min(k, n - 1))
            return float(x_sorted[k])

        if show_five_percent:
            p95 = _q_from_sorted(95.0)
        if show_two_percent:
            p98 = _q_from_sorted(98.0)

        # Plot decimation for huge-but-allowed sorted arrays
        if max_plot_points is not None and n > int(max_plot_points):
            idx = np.linspace(0, n - 1, int(max_plot_points), dtype=np.int64)
            x_plot = x_sorted[idx]
            y_cdf = (idx.astype(np.float64) + 1.0) / n
            y_ccdf = 1.0 - (idx.astype(np.float64)) / n
        else:
            x_plot = x_sorted
            y_cdf = np.arange(1, n + 1, dtype=np.float64) / n
            y_ccdf = 1.0 - np.arange(0, n, dtype=np.float64) / n

        x_ccdf = x_plot  # for consistent naming

    else:
        # Large-N histogram mode: O(N) time, O(bins) memory
        bins = int(hist_bins)
        if bins < 64:
            raise ValueError("hist_bins too small; use at least ~256 for smooth curves.")

        # Range determination: if you can supply hist_range for EPFD plots, do it!
        # That avoids a full nanmin/nanmax scan.
        if hist_range is None:
            if assume_finite:
                xmin = float(np.min(samples_all))
                xmax = float(np.max(samples_all))
            else:
                xmin = float(np.nanmin(samples_all))
                xmax = float(np.nanmax(samples_all))
        else:
            xmin, xmax = float(hist_range[0]), float(hist_range[1])

        if not np.isfinite(xmin) or not np.isfinite(xmax):
            raise ValueError("Non-finite min/max; check input for NaN/Inf or provide hist_range.")
        if xmax == xmin:
            xmax = xmin + 1e-9
        x_data_min = float(xmin)
        x_data_max = float(xmax)

        dx = (xmax - xmin) / bins
        inv_dx = 1.0 / dx

        # Bin edges as float64 for stable interpolation
        edges = xmin + dx * np.arange(bins + 1, dtype=np.float64)

        # Numba one-pass histogram (skips global isfinite mask)
        counts, n = _hist_counts_numba(samples_all, xmin, inv_dx, bins)
        n = int(n)
        if n == 0:
            raise ValueError("No finite samples to plot.")

        cdf_counts = np.cumsum(counts, dtype=np.int64)

        # Percentiles from histogram (approx; accuracy improves with more bins)
        if show_five_percent:
            p95 = float(_percentile_from_hist_numba(edges, cdf_counts, 95.0))
        if show_two_percent:
            p98 = float(_percentile_from_hist_numba(edges, cdf_counts, 98.0))

        # CDF curve (post-step): use upper edges with CDF up to that edge
        x_plot = edges[1:]
        y_cdf = cdf_counts.astype(np.float64) / n

        # CCDF curve (pre-step): use lower edges with survival before bin mass
        cum_before = np.concatenate(([0], cdf_counts[:-1])).astype(np.float64)
        x_ccdf = edges[:-1]
        y_ccdf = 1.0 - (cum_before / n)

    corridor_x: np.ndarray | None = None
    corridor_ccdf_min: np.ndarray | None = None
    corridor_ccdf_max: np.ndarray | None = None
    corridor_p98_range: tuple[float, float] | None = None
    corridor_max_n = 0
    corridor_x_right: float | None = None
    if show_skycell_corridor:
        if str(plot_type).lower() == "cdf":
            warnings.warn("show_skycell_corridor=True is ignored when plot_type='cdf'.")
        else:
            x2d = arr.reshape(-1, arr.shape[-1])
            if method == "hist":
                corridor_x_right = float(x_data_max)
                (
                    corridor_x,
                    corridor_ccdf_min,
                    corridor_ccdf_max,
                    corridor_p98_range,
                    corridor_max_n,
                ) = _skycell_ccdf_corridor_exact(
                    x2d,
                    x_support=x_ccdf,
                    assume_finite=assume_finite,
                )
            else:
                corridor_range = skycell_corridor_x_range
                if corridor_range is None:
                    if np.isfinite(x_data_min) and np.isfinite(x_data_max):
                        corridor_range = (x_data_min, x_data_max)
                    elif assume_finite:
                        corridor_range = (float(np.min(samples_all)), float(np.max(samples_all)))
                    else:
                        corridor_range = (float(np.nanmin(samples_all)), float(np.nanmax(samples_all)))
                corridor_x_right = float(corridor_range[1])
                (
                    corridor_x,
                    corridor_ccdf_min,
                    corridor_ccdf_max,
                    corridor_p98_range,
                    corridor_max_n,
                ) = _skycell_ccdf_corridor_exact(
                    x2d,
                    bins=int(skycell_corridor_bins),
                    x_range=(float(corridor_range[0]), float(corridor_range[1])),
                    assume_finite=assume_finite,
                )
    return _plot_cdf_ccdf_precomputed(
        x_plot=np.asarray(x_plot, dtype=np.float64),
        y_cdf=np.asarray(y_cdf, dtype=np.float64),
        x_ccdf=np.asarray(x_ccdf, dtype=np.float64),
        y_ccdf=np.asarray(y_ccdf, dtype=np.float64),
        n=int(n),
        unit=unit,
        p95=p95,
        p98=p98,
        plot_type=plot_type,
        show_two_percent=show_two_percent,
        show_five_percent=show_five_percent,
        prot_value=prot_value,
        prot_legend=prot_legend,
        prot_colors=prot_colors,
        title=title,
        figsize=figsize,
        line_width=line_width,
        line_alpha=line_alpha,
        cdf_color=cdf_color,
        ccdf_color=ccdf_color,
        marker_size=marker_size,
        grid=grid,
        legend_outside=legend_outside,
        show_skycell_corridor=show_skycell_corridor,
        skycell_corridor_alpha=skycell_corridor_alpha,
        skycell_corridor_color=skycell_corridor_color,
        skycell_corridor_label=skycell_corridor_label,
        show_skycell_p98_note=show_skycell_p98_note,
        y_percent_decimals=y_percent_decimals,
        ccdf_ymin_pct=ccdf_ymin_pct,
        cdf_ymin_pct=cdf_ymin_pct,
        y_log=y_log,
        show=show,
        return_values=return_values,
        x_label=x_label,
        show_margin=show_margin,
        margin_at=margin_at,
        save_path=save_path,
        save_dpi=save_dpi,
        corridor_x=corridor_x,
        corridor_ccdf_min=corridor_ccdf_min,
        corridor_ccdf_max=corridor_ccdf_max,
        corridor_p98_range=corridor_p98_range,
        corridor_max_n=int(corridor_max_n),
        corridor_x_right=corridor_x_right,
        method_name=method,
    )




def plot_cdf_ccdf_from_histogram(
    counts: Any,
    *,
    edges: Any,
    plot_type: str = "both",
    show_two_percent: bool = False,
    show_five_percent: bool = False,
    prot_value: Any | List[Any] | None = None,
    prot_legend: List[str] | None = None,
    prot_colors: List[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
    line_width: float = 2.0,
    line_alpha: float = 1.0,
    cdf_color: str = "#2563eb",
    ccdf_color: str = "#16a34a",
    marker_size: float = 60.0,
    grid: bool = True,
    legend_outside: bool = False,
    show_skycell_corridor: bool = False,
    skycell_corridor_alpha: float = 0.20,
    skycell_corridor_color: str = "#f59e0b",
    skycell_corridor_label: str = "Skycell corridor (min-max)",
    show_skycell_p98_note: bool = False,
    y_percent_decimals: int = 2,
    ccdf_ymin_pct: float | None = None,
    cdf_ymin_pct: float | None = None,
    y_log: bool = False,
    show: bool = True,
    return_values: bool = False,
    x_label: str | None = None,
    show_margin: bool = False,
    margin_at: str = "p98",
    save_path: str | None = None,
    save_dpi: int = 300,
):
    """
    Plot CDF/CCDF curves directly from a pre-binned 1D histogram.

    Percentiles use the same histogram interpolation semantics as
    :func:`plot_cdf_ccdf` with ``ecdf_method="hist"``.
    """
    counts_arr = np.asarray(counts, dtype=np.int64).reshape(-1)
    edges_arr, unit = _to_plain_array(edges)
    edges_arr = np.asarray(edges_arr, dtype=np.float64).reshape(-1)

    if counts_arr.ndim != 1:
        raise ValueError("counts must be one-dimensional.")
    if edges_arr.ndim != 1:
        raise ValueError("edges must be one-dimensional.")
    if edges_arr.size != counts_arr.size + 1:
        raise ValueError("edges must have length len(counts) + 1.")
    if np.any(counts_arr < 0):
        raise ValueError("counts must be non-negative.")
    if not np.all(np.isfinite(edges_arr)):
        raise ValueError("edges must be finite.")
    if not np.all(np.diff(edges_arr) > 0.0):
        raise ValueError("edges must be strictly increasing.")

    n = int(np.sum(counts_arr, dtype=np.int64))
    if n <= 0:
        raise ValueError("Histogram contains no positive samples.")

    if show_skycell_corridor:
        warnings.warn(
            "show_skycell_corridor is unsupported for pre-binned histograms and will be ignored."
        )

    cdf_counts = np.cumsum(counts_arr, dtype=np.int64)
    x_plot = edges_arr[1:]
    y_cdf = cdf_counts.astype(np.float64) / float(n)
    cum_before = np.concatenate(([0], cdf_counts[:-1])).astype(np.float64)
    x_ccdf = edges_arr[:-1]
    y_ccdf = 1.0 - (cum_before / float(n))

    p95 = None
    p98 = None
    if show_five_percent:
        p95 = float(_percentile_from_hist_numba(edges_arr, cdf_counts, 95.0))
    if show_two_percent:
        p98 = float(_percentile_from_hist_numba(edges_arr, cdf_counts, 98.0))

    return _plot_cdf_ccdf_precomputed(
        x_plot=x_plot,
        y_cdf=y_cdf,
        x_ccdf=x_ccdf,
        y_ccdf=y_ccdf,
        n=n,
        unit=unit,
        p95=p95,
        p98=p98,
        plot_type=plot_type,
        show_two_percent=show_two_percent,
        show_five_percent=show_five_percent,
        prot_value=prot_value,
        prot_legend=prot_legend,
        prot_colors=prot_colors,
        title=title,
        figsize=figsize,
        line_width=line_width,
        line_alpha=line_alpha,
        cdf_color=cdf_color,
        ccdf_color=ccdf_color,
        marker_size=marker_size,
        grid=grid,
        legend_outside=legend_outside,
        show_skycell_corridor=False,
        skycell_corridor_alpha=skycell_corridor_alpha,
        skycell_corridor_color=skycell_corridor_color,
        skycell_corridor_label=skycell_corridor_label,
        show_skycell_p98_note=show_skycell_p98_note,
        y_percent_decimals=y_percent_decimals,
        ccdf_ymin_pct=ccdf_ymin_pct,
        cdf_ymin_pct=cdf_ymin_pct,
        y_log=y_log,
        show=show,
        return_values=return_values,
        x_label=x_label,
        show_margin=show_margin,
        margin_at=margin_at,
        save_path=save_path,
        save_dpi=save_dpi,
        method_name="prebinned_hist",
    )


def plot_satellite_elevation_pfd_heatmap(
    histogram: Any,
    *,
    elevation_edges_deg: Any,
    pfd_edges_db: Any,
    title: str = "Instantaneous per-satellite PFD vs elevation",
    xlabel: str = "Satellite elevation at RAS station [deg]",
    ylabel: str = "Instantaneous PFD contribution [dBW/m^2/MHz]",
    colorbar_label: str = "Sample count",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10.0, 6.8),
    grid: bool = True,
    grid_color: str = "#94a3b8",
    grid_alpha: float = 0.4,
    grid_linewidth: float = 0.9,
    show: bool = True,
    return_values: bool = False,
    save_path: str | None = None,
    save_dpi: int = 300,
) -> Figure | tuple[Figure, dict[str, Any]]:
    """
    Plot a 2D frequency heatmap of per-satellite PFD against satellite elevation.

    Parameters
    ----------
    histogram : array-like
        Two-dimensional non-negative sample-count histogram with shape
        ``(N_elevation_bins, N_pfd_bins)``.
    elevation_edges_deg, pfd_edges_db : array-like
        Monotonically increasing bin-edge arrays with lengths
        ``N_elevation_bins + 1`` and ``N_pfd_bins + 1``.
    title, xlabel, ylabel, colorbar_label : str, optional
        Plot labels.
    cmap : str, optional
        Matplotlib colormap name.
    figsize : tuple of float, optional
        Figure size in inches.
    grid : bool, optional
        If `True`, draw a major grid over the heatmap.
    grid_color : str, optional
        Matplotlib-compatible grid color.
    grid_alpha : float, optional
        Grid transparency in the closed interval ``[0, 1]``.
    grid_linewidth : float, optional
        Grid line width in points.
    show : bool, optional
        If `True`, display the figure with ``plt.show()``.
    return_values : bool, optional
        If `True`, return ``(fig, info_dict)``.
    save_path : str or None, optional
        If provided, save the figure to this path.
    save_dpi : int, optional
        Output DPI when ``save_path`` is set.

    Returns
    -------
    matplotlib.figure.Figure or tuple
        Figure alone, or ``(figure, info)`` when ``return_values=True``.

    Raises
    ------
    ValueError
        Raised when the histogram dimensionality or the edge-array lengths are
        inconsistent.

    Notes
    -----
    This helper accepts a pre-accumulated histogram so callers can keep their
    own streaming HDF5 read logic and hand only the compact result to the
    plotting layer.
    """
    hist = np.asarray(histogram, dtype=np.float64)
    elev_edges = np.asarray(elevation_edges_deg, dtype=np.float64).reshape(-1)
    pfd_edges = np.asarray(pfd_edges_db, dtype=np.float64).reshape(-1)
    if hist.ndim != 2:
        raise ValueError(f"histogram must be 2-D; got shape {hist.shape!r}.")
    if elev_edges.size != hist.shape[0] + 1:
        raise ValueError(
            "elevation_edges_deg must have length N_elevation_bins + 1; "
            f"got {elev_edges.size} for histogram shape {hist.shape!r}."
        )
    if pfd_edges.size != hist.shape[1] + 1:
        raise ValueError(
            "pfd_edges_db must have length N_pfd_bins + 1; "
            f"got {pfd_edges.size} for histogram shape {hist.shape!r}."
        )
    if np.any(hist < 0):
        raise ValueError("histogram counts must be non-negative.")
    if not (0.0 <= float(grid_alpha) <= 1.0):
        raise ValueError("grid_alpha must lie in [0, 1].")
    if float(grid_linewidth) < 0.0:
        raise ValueError("grid_linewidth must be non-negative.")

    positive = hist > 0
    vmin = float(np.min(hist[positive])) if np.any(positive) else 1.0
    vmax = float(np.max(hist)) if np.any(positive) else 1.0

    fig, ax = _new_mpl_subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("#fbfdff")
    mesh = ax.pcolormesh(
        elev_edges,
        pfd_edges,
        hist.T,
        shading="auto",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=max(vmin, vmax)),
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.035, fraction=0.05)
    cbar.set_label(colorbar_label)
    ax.set_title(title, pad=16, fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(
        bool(grid),
        color=grid_color,
        alpha=float(grid_alpha),
        linewidth=float(grid_linewidth),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155")
    fig.tight_layout(pad=1.2)

    if save_path is not None:
        fig.savefig(save_path, dpi=int(save_dpi), bbox_inches="tight")
    if show:
        plt.show()
    elif not return_values:
        plt.close(fig)

    info = {
        "sample_count": int(np.sum(hist, dtype=np.int64)),
        "positive_bin_count": int(np.count_nonzero(positive)),
        "max_bin_count": int(vmax),
    }
    return (fig, info) if return_values else fig


def plot_hemisphere_2D(
    data: Any,
    *,
    # ---------- DATA LAYOUT ----------
    grid_info: np.ndarray | None = None,
    # Display crop only (does not rebin); accepts float degrees or astropy quantities.
    elev_range: Tuple[float | u.Quantity, float | u.Quantity] | None = None,

    # ---------- STATISTICS ----------
    mode: str = "power",                        # "power" or "data_loss"
    worst_percent: float = 2.0,                 # for "power": 100 - worst_percent percentile
    protection_criterion: Any | None = None,    # for "data_loss": threshold
    cell_axis: int = -1,                        # axis with skycells; 2334 or len(grid_info)

    # ---------- COLOR ----------
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,

    # ---------- CELL FACE VISUALS (MPL only) ----------
    edgecolor: str = "none",
    linewidth: float = 0.0,
    alpha: float = 1.0,                         # face opacity

    # ---------- LABELS / COLORBAR ----------
    title: str | None = None,
    colorbar: bool = True,

    # ---------- GUIDES ----------
    draw_guides: bool | None = None,            # None → True for polar, False for rect
    guide_color: str = "#111111",
    guide_alpha: float = 1.0,
    guide_linewidth: float = 2.2,
    guide_length: float = 1.1,                  # arrows length in “plot radius” units
    show_axis_arrows: bool = True,
    arrow_direction: str = "outward",           # "outward" | "inward"
    arrow_scale: float = 1.5,
    label_offset_extra: float = 0.20,

    # ---------- SKY-CELL BORDERS ----------
    draw_cell_borders: bool = True,
    border_color: str = "#1f2937",
    border_width: float = 1.0,
    border_alpha: float = 0.85,
    border_ring_samples: int = 180,

    # ---------- VIEW / LAYOUT ----------
    tight: bool = True,
    tight_pad: float = 0.00,

    # ---------- ENGINE / INTERACTIVITY ----------
    engine: str = "auto",                       # "auto" | "mpl" | "plotly"
    interactive: bool | None = None,            # None→auto; True interactive; False static
    figsize: tuple[float, float] = (8.5, 7.5),  # MPL only
    show: bool = True,
    return_values: bool = False,

    # ---------- 2D-SPECIFIC ----------
    projection: str = "polar",                  # "polar" or "rect"
    radial_mapping: str = "equal_area",         # polar radius: "equal_area" or "linear"
    invert_polar: bool = False,                 # if True: zenith at rim, horizon at center
    raster_res: int = 800,                      # Plotly raster base resolution
    save_html: str | None = None,               # Plotly: export standalone HTML
):
    """
    Plot an upper-hemisphere sky map in polar or rectangular projection.

    The function accepts either the canonical 2334-cell S.1586 grid or a custom
    subset/grid described by ``grid_info``. For custom grids, each sky cell is
    drawn from its exact azimuth/elevation bounds rather than being re-binned
    onto the reference S.1586 layout.

    Parameters
    ----------
    data : array-like or Quantity
        Sample array with one sky-cell axis and one or more sample axes. The
        sky-cell axis must have length 2334 when ``grid_info`` is omitted, or
        length ``len(grid_info)`` when a custom grid is provided.
    grid_info : structured ndarray, optional
        Per-cell geometry with fields ``cell_lon_low``, ``cell_lon_high``,
        ``cell_lat_low``, and ``cell_lat_high`` in degrees.
    elev_range : tuple, optional
        Display-only elevation crop ``(low, high)`` in degrees or angle
        quantities. Data outside the visible range are not re-binned.
    mode : {"power", "data_loss"}, optional
        Statistic to display per cell.
    worst_percent : float, optional
        Worst-case percentage used in ``mode="power"``. The plotted value is
        the ``100 - worst_percent`` percentile of the per-cell samples.
    protection_criterion : float or Quantity, optional
        Threshold used in ``mode="data_loss"``.
    cell_axis : int, optional
        Axis index of the sky-cell dimension.
    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Rendered figure.
    values : ndarray or Quantity, optional
        Returned only when ``return_values=True``. Contains the per-cell values
        used for colouring, in the same domain/unit as the input data for power
        mode and in percent for data-loss mode.

    Raises
    ------
    ValueError
        Raised when the cell axis is invalid, the grid description does not
        match the data, or the requested statistic cannot be computed.

    Notes
    -----
    - Percentiles use nearest-rank order statistics on the native data values.
      For dB-valued power samples this is exactly invariant under monotonic
      dB-to-linear conversion.
    - Threshold exceedance probabilities are computed on finite native-domain
      samples only.
    - Additive physical-power statistics are not computed by this function.
    - Polar azimuth uses compass convention: 0° up (North), 90° right (East),
      180° down (South), 270° left (West).
    - When ``grid_info`` is omitted and ``projection="polar"``, the full-grid
      Plotly path uses a rasterized S.1586 lookup for speed.
    """
    # Light theme if seaborn is around (optional, non-fatal if missing)
    try:
        import seaborn as sns  # type: ignore
        sns.set_theme(style="white", context="notebook")
    except Exception:
        pass

    # --- normalise inputs ---
    arr, unit = _to_plain_array(data)
    if arr.ndim < 2:
        raise ValueError(f"'data' must have at least 2 dims; got {arr.shape}.")
    if cell_axis < 0:
        cell_axis = arr.ndim + cell_axis
    if not (0 <= cell_axis < arr.ndim):
        raise ValueError(f"cell_axis {cell_axis} out of range for shape {arr.shape}.")

    arr = np.moveaxis(arr, cell_axis, -1)
    C = arr.shape[-1]
    samples = arr.reshape(-1, C)

    using_subset = grid_info is not None
    if using_subset:
        if C != len(grid_info):
            raise ValueError(
                f"With grid_info provided, cell axis length ({C}) must equal len(grid_info) ({len(grid_info)})."
            )
    else:
        if C != 2334:
            raise ValueError(f"Axis {cell_axis} must be 2334 skycells when grid_info is not provided (got {C}).")

    # --- per-cell statistic ---
    mode = mode.lower()
    if mode not in ("power", "data_loss"):
        raise ValueError("mode must be 'power' or 'data_loss'.")

    if mode == "power":
        if not (0.0 < worst_percent < 100.0):
            raise ValueError("worst_percent must be in (0, 100).")
        pct = 100.0 - float(worst_percent)
        vals = _nearest_rank_percentile_axis0(samples, pct)
        cbar_title = f"Power{'' if unit is None else f' {unit}'}"
    else:
        if protection_criterion is None:
            raise ValueError("protection_criterion is required for mode='data_loss'.")
        thr_num = _resolve_threshold_numeric(protection_criterion, unit)
        loss = _finite_exceedance_percent_axis0(samples, thr_num)
        vals = loss
        cbar_title = "Data loss [%]"

    # Keep numeric values and a compact unit string for labels/tooltips
    if (mode == "power") and (unit is not None):
        vals_q = vals * unit
        vals_num = vals_q.value
        unit_str = f" {unit}"
    else:
        vals_q = vals
        vals_num = vals
        unit_str = "" if mode == "power" else " %"

    # Colour limits (explicit or from data)
    cmin = np.nanmin(vals_num) if vmin is None else float(vmin)
    cmax = np.nanmax(vals_num) if vmax is None else float(vmax)
    if (not np.isfinite(cmin)) or (not np.isfinite(cmax)) or (cmax == cmin):
        cmin, cmax = float(np.nanmin(vals_num)), float(np.nanmax(vals_num) + 1.0)

    # --- geometry and optional subset mapping ---
    az_lo_full, az_hi_full, el_lo_full, el_hi_full, el_edges, cells_per_ring = _s1586_cells()

    if using_subset:
        # exact bounds from `grid_info` (works for both S.1586-clip and generic grids)
        az_lo_cells = grid_info["cell_lon_low"].astype(float)
        az_hi_cells = grid_info["cell_lon_high"].astype(float)
        el_lo_cells = grid_info["cell_lat_low"].astype(float)
        el_hi_cells = grid_info["cell_lat_high"].astype(float)

        # For the *old* polar-heatmap path we filled a sparse 2334 array.
        # We keep that only for S.1586 (no grid_info) — below we draw polygons instead.
        cover_lo = float(np.nanmin(el_lo_cells)) if len(el_lo_cells) else 0.0
        cover_hi = float(np.nanmax(el_hi_cells)) if len(el_hi_cells) else 0.0
    else:
        # Full S.1586 grid (no `grid_info`): use canonical geometry.
        az_lo_cells, az_hi_cells = az_lo_full, az_hi_full
        el_lo_cells, el_hi_cells = el_lo_full, el_hi_full
        cover_lo, cover_hi = 0.0, 90.0

    # --- elevation crop (display only) ---
    if elev_range is not None:
        if (not isinstance(elev_range, Sequence)) or (len(elev_range) != 2):
            raise ValueError("`elev_range` must be a pair like (low, high).")
        lo, hi = elev_range
        lo_deg = float(lo.to_value(u.deg)) if hasattr(lo, "to") else float(lo)
        hi_deg = float(hi.to_value(u.deg)) if hasattr(hi, "to") else float(hi)
        if lo_deg > hi_deg:
            lo_deg, hi_deg = hi_deg, lo_deg
        req_lo, req_hi = max(0.0, lo_deg), min(90.0, hi_deg)
    else:
        req_lo, req_hi = cover_lo, cover_hi

    show_lo = max(req_lo, cover_lo)
    show_hi = min(req_hi, cover_hi)

    if (elev_range is not None) and (req_lo < cover_lo or req_hi > cover_hi):
        missing = []
        if req_lo < cover_lo:
            missing.append(f"[{req_lo:.1f}°, {cover_lo:.1f}°]")
        if req_hi > cover_hi:
            missing.append(f"[{cover_hi:.1f}°, {req_hi:.1f}°]")
        if missing:
            warnings.warn(
                "Requested elev_range extends beyond data coverage; "
                f"no data for: {', '.join(missing)}. Showing {show_lo:.1f}°–{show_hi:.1f}°."
            )

    # Compass-style azimuth orientation for polar display:
    # 0°=North (up), 90°=East (right), 180°=South (down), 270°=West (left).
    def _az_to_plot_angle_rad(az_deg: float | np.ndarray) -> float | np.ndarray:
        """Map compass azimuth degrees to plotting angle in radians."""
        az = np.asarray(az_deg, dtype=float)
        ang = np.radians((90.0 - az) % 360.0)
        return float(ang) if np.ndim(ang) == 0 else ang

    def _xy_to_azimuth_deg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Invert XY polar coordinates back to compass azimuth degrees."""
        return (90.0 - np.degrees(np.arctan2(y, x))) % 360.0

    # --- engine selection ---
    use_plotly = False
    if engine == "plotly":
        use_plotly = True
    elif engine == "auto":
        if interactive is not False:
            try:
                import plotly  # noqa: F401
                use_plotly = True
            except Exception:
                use_plotly = False

    # ============================ PLOTLY ============================
    if use_plotly:
        import plotly.graph_objects as go
        import plotly.io as pio

        # Build a Plotly-compatible colorscale from the Matplotlib cmap.
        mpl_cmap = plt.get_cmap(cmap)
        colorscale = [
            [t, f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"]
            for t, (r, g, b, a) in zip(np.linspace(0, 1, 256), mpl_cmap(np.linspace(0, 1, 256)))
        ]

        # Helper to map a numeric value to an RGBA string using the chosen cmap + [cmin,cmax].
        def _rgba_for_value(v: float, alpha_override: float | None = None) -> str:
            """Map a scalar value to Plotly RGBA text using active color limits."""
            if not np.isfinite(v) or cmax == cmin:
                t = 0.0
            else:
                t = float(np.clip((v - cmin) / max(cmax - cmin, 1e-12), 0.0, 1.0))
            r, g, b, a = mpl_cmap(t)
            a = (alpha_override if alpha_override is not None else alpha) * a
            return f"rgba({int(round(r*255))},{int(round(g*255))},{int(round(b*255))},{a:.3f})"

        if projection.lower() == "polar":

            if not using_subset:
                # ------------------------------
                # FAST PATH (no grid_info): keep the original S.1586 raster heatmap
                # ------------------------------
                N = max(256, int(raster_res))
                x = np.linspace(-1, 1, N)
                y = np.linspace(-1, 1, N)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                disc = R <= 1.0

                el = _el_from_r_display(R, radial_mapping, invert=invert_polar)
                azimuth_deg = _xy_to_azimuth_deg(X, Y)

                # Map each pixel to an S.1586 ring and azimuth bin
                ring_idx = np.digitize(el, el_edges, right=True) - 1
                ring_idx = np.clip(ring_idx, 0, len(el_edges) - 2)
                steps = 360.0 / np.take(cells_per_ring, ring_idx)
                az_bin = (azimuth_deg // steps).astype(int)
                az_bin = np.clip(az_bin, 0, np.take(cells_per_ring, ring_idx) - 1)
                offsets = np.concatenate([[0], np.cumsum(cells_per_ring[:-1])])
                cell_idx = np.take(offsets, ring_idx) + az_bin

                Z = np.full_like(R, np.nan, float)
                Z[disc] = vals_num[cell_idx[disc]]
                keep = disc & (el >= show_lo) & (el <= show_hi)
                Z[~keep] = np.nan

                hovertemplate = (
                    (f"Power: %{{z:.3f}}{unit_str}<extra></extra>")
                    if mode == "power"
                    else "Data loss: %{z:.2f}%<extra></extra>"
                )

                traces = [go.Heatmap(
                    x=x, y=y, z=Z,
                    zmin=cmin, zmax=cmax,
                    colorscale=colorscale,
                    opacity=float(alpha),
                    colorbar=dict(title=cbar_title) if colorbar else None,
                    hovertemplate=hovertemplate,
                    showscale=colorbar,
                    name=""
                )]

                # S.1586 borders (unchanged) if requested
                if draw_cell_borders:
                    tt = np.linspace(0, 2*np.pi, 361)
                    # rings
                    for elb in el_edges:
                        rr = _r_from_el(elb, radial_mapping, invert=invert_polar)
                        traces.append(go.Scatter(
                            x=rr*np.cos(tt), y=rr*np.sin(tt),
                            mode="lines",
                            line=dict(color=border_color, width=max(1, int(round(border_width*2)))),
                            opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                        ))
                    # meridians
                    for i_ring, n_in_ring in enumerate(cells_per_ring):
                        el0 = el_edges[i_ring]
                        el1 = el_edges[i_ring+1]
                        r0 = _r_from_el(el0, radial_mapping, invert=invert_polar)
                        r1 = _r_from_el(el1, radial_mapping, invert=invert_polar)
                        step = 360 // int(n_in_ring)
                        for az in np.arange(0, 360, step):
                            t = _az_to_plot_angle_rad(float(az))
                            traces.append(go.Scatter(
                                x=[r0*np.cos(t), r1*np.cos(t)],
                                y=[r0*np.sin(t), r1*np.sin(t)],
                                mode="lines",
                                line=dict(color=border_color, width=max(1, int(round(border_width*2)))),
                                opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                            ))

            else:
                # ------------------------------
                # GENERIC GRID PATH (grid_info present): per-cell polygons + grid_info-driven borders
                # ------------------------------
                traces: List[go.Scatter] = []

                # 1) Colorbar host: add an invisible heatmap that only provides a colorbar.
                if colorbar:
                    traces.append(go.Heatmap(
                        z=[[cmin, cmax]],
                        x=[-1.2, -1.1], y=[-1.2, -1.1],  # tuck it outside the unit disc
                        zmin=cmin, zmax=cmax,
                        colorscale=colorscale,
                        showscale=True,
                        opacity=0.0,
                        colorbar=dict(title=cbar_title),
                        hoverinfo="skip",
                        visible=True,
                        name=""
                    ))

                # 2) Draw one filled polygon per cell (quad in polar display coords).
                for i in range(len(vals_num)):
                    lo_el = float(el_lo_cells[i]); hi_el = float(el_hi_cells[i])
                    if not (max(lo_el, show_lo) < min(hi_el, show_hi)):
                        continue  # cell completely outside requested band → skip

                    lo_az = float(az_lo_cells[i]); hi_az = float(az_hi_cells[i])
                    v = float(vals_num[i])
                    if not np.isfinite(v):
                        continue

                    # Corners of the (ring segment) cell in polar display coords:
                    #  (r(el_lo), az_lo) → (r(el_lo), az_hi) → (r(el_hi), az_hi) → (r(el_hi), az_lo).
                    t0 = _az_to_plot_angle_rad(lo_az)
                    t1 = _az_to_plot_angle_rad(hi_az)
                    r0 = _r_from_el(lo_el, radial_mapping, invert=invert_polar)
                    r1 = _r_from_el(hi_el, radial_mapping, invert=invert_polar)

                    xs = [r0*np.cos(t0), r0*np.cos(t1), r1*np.cos(t1), r1*np.cos(t0), r0*np.cos(t0)]
                    ys = [r0*np.sin(t0), r0*np.sin(t1), r1*np.sin(t1), r1*np.sin(t0), r0*np.sin(t0)]

                    # Hover text mirrors rectangular path
                    if mode == "power":
                        value_str = f"{v:.3f}{unit_str}"
                        label_title = "Power"
                    else:
                        value_str = f"{v:.2f}%"
                        label_title = "Data loss"
                    hover_text = (
                        f"Az: {lo_az:.1f}–{hi_az:.1f}°<br>"
                        f"El: {lo_el:.1f}–{hi_el:.1f}°<br>"
                        f"{label_title}: {value_str}"
                    )

                    traces.append(go.Scatter(
                        x=xs, y=ys,
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0)", width=0),
                        fill="toself",
                        fillcolor=_rgba_for_value(v),
                        hoveron="fills",
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=False,
                        name=""
                    ))

                # 3) Optional borders derived from grid_info
                if draw_cell_borders:
                    line_w = max(1, int(round(border_width * 2)))
                    # a) RINGS: unique elevation edges (bottoms + tops), clipped to display band
                    unique_els = np.unique(
                        np.round(np.concatenate([el_lo_cells, el_hi_cells]).astype(float), 6)
                    )
                    tt = np.linspace(0, 2*np.pi, max(180, border_ring_samples))
                    for elb in unique_els:
                        if not (show_lo <= elb <= show_hi):
                            continue
                        rr = _r_from_el(float(elb), radial_mapping, invert=invert_polar)
                        traces.append(go.Scatter(
                            x=rr*np.cos(tt), y=rr*np.sin(tt),
                            mode="lines",
                            line=dict(color=border_color, width=line_w),
                            opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                        ))

                    # b) MERIDIANS: for each elevation band, draw radial segments at every unique az edge
                    #    across that band's y-extent (clipped to display band).
                    rnd = 6  # rounding for stable uniqueness
                    band_map: Dict[Tuple[float, float], Dict[str, Any]] = {}
                    for i in range(len(grid_info)):
                        y0 = float(el_lo_cells[i]); y1 = float(el_hi_cells[i])
                        # only consider bands that intersect the display window
                        if not (max(y0, show_lo) < min(y1, show_hi)):
                            continue
                        key = (round(y0, rnd), round(y1, rnd))
                        got = band_map.get(key)
                        if got is None:
                            band_map[key] = {"y0": y0, "y1": y1, "lon_edges": []}
                            got = band_map[key]
                        got["lon_edges"].extend([float(az_lo_cells[i]), float(az_hi_cells[i])])

                    for (_, _), info in band_map.items():
                        y0 = max(info["y0"], show_lo)
                        y1 = min(info["y1"], show_hi)
                        if y0 >= y1:
                            continue
                        r0 = _r_from_el(y0, radial_mapping, invert=invert_polar)
                        r1 = _r_from_el(y1, radial_mapping, invert=invert_polar)

                        uniq_lons = sorted(set(round(l, rnd) for l in info["lon_edges"]))
                        for L in uniq_lons:
                            t = _az_to_plot_angle_rad(float(L))
                            traces.append(go.Scatter(
                                x=[r0*np.cos(t), r1*np.cos(t)],
                                y=[r0*np.sin(t), r1*np.sin(t)],
                                mode="lines",
                                line=dict(color=border_color, width=line_w),
                                opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                            ))

            # 4) Guides (outer unit circle + rays + labels) in compass azimuth orientation
            annotations: list[dict] = []
            label_points: list[tuple[float, float]] = []
            if (draw_guides is None and projection.lower() == "polar") or (draw_guides is True):
                tt = np.linspace(0, 2*np.pi, 361)
                traces.append(go.Scatter(
                    x=np.cos(tt), y=np.sin(tt),
                    mode="lines", line=dict(color=guide_color, width=3),
                    hoverinfo="skip", showlegend=False, name=""
                ))

                def _ray(az_deg: float, label: str, extra_dx: float = 0.0, extra_dy: float = 0.0):
                    """Draw one compass ray plus optional arrow and label."""
                    t = _az_to_plot_angle_rad(az_deg)
                    bx, by = guide_length*np.cos(t), guide_length*np.sin(t)
                    traces.append(go.Scatter(
                        x=[0.0, bx], y=[0.0, by],
                        mode="lines",
                        line=dict(color=guide_color, width=3),
                        hoverinfo="skip", showlegend=False, name=""
                    ))
                    if show_axis_arrows:
                        arrow = dict(arrowhead=3, arrowsize=arrow_scale*0.7, arrowwidth=2, arrowcolor=guide_color)
                        if arrow_direction.lower() == "outward":
                            annotations.append(dict(x=bx, y=by, ax=0.0, ay=0.0, showarrow=True, **arrow))
                        else:
                            annotations.append(dict(x=0.0, y=0.0, ax=bx, ay=by, showarrow=True, **arrow))
                    lx = bx + np.cos(t) * label_offset_extra + extra_dx
                    ly = by + np.sin(t) * label_offset_extra + extra_dy
                    label_points.append((lx, ly))
                    annotations.append(dict(
                        x=lx, y=ly, text=label, showarrow=False,
                        xanchor="center", yanchor="bottom",
                        font=dict(color=guide_color, size=14)
                    ))

                _ray(0,   "Az 0°")
                _ray(90,  "Az 90°")
                _ray(180, "Az 180°")
                _ray(270, "Az 270°")

            # 5) Axis bounds big enough for arrows + labels (same logic)
            max_r = max(1.0, guide_length)
            if label_points:
                max_r = max(max_r, max(np.hypot(px, py) for (px, py) in label_points))
            max_r *= (1.0 + float(tight_pad))
            xr = [-max_r, +max_r]
            yr = [-max_r, +max_r]

            layout = dict(
                title=title or f"S.1586-1 Hemisphere — {'Power' if mode=='power' else 'Data loss'} (polar)",
                xaxis=dict(visible=False, scaleanchor="y", scaleratio=1, range=xr),
                yaxis=dict(visible=False, range=yr),
                margin=dict(l=0, r=0, t=60, b=0),
                paper_bgcolor="white",
                plot_bgcolor="white",
                annotations=annotations if annotations else None,
            )
            fig = go.Figure(data=traces, layout=layout)

        else:
            # --- rectangular (Plotly) ---
            # (This branch was already modernized earlier to follow grid_info)
            import plotly.graph_objects as go

            def _rgba_for_value(v: float, alpha_override: float | None = None) -> str:
                """Map a scalar value to Plotly RGBA text using active color limits."""
                t = 0.0 if not np.isfinite(v) or cmax == cmin else (v - cmin) / max(cmax - cmin, 1e-12)
                t = float(np.clip(t, 0.0, 1.0))
                r, g, b, a = mpl_cmap(t)
                a = (alpha_override if alpha_override is not None else alpha) * a
                return f"rgba({int(round(r*255))},{int(round(g*255))},{int(round(b*255))},{a:.3f})"

            _az_lo, _az_hi = az_lo_cells, az_hi_cells
            _el_lo, _el_hi = el_lo_cells, el_hi_cells
            _vals = vals_num

            traces: List[go.Scatter] = []

            # colorbar host
            if colorbar:
                traces.append(go.Heatmap(
                    z=[[cmin, cmax]],
                    x=[-1, -0.5], y=[-1, -0.5],
                    zmin=cmin, zmax=cmax,
                    colorscale=colorscale,
                    showscale=True,
                    opacity=0.0,
                    colorbar=dict(title=cbar_title),
                    hoverinfo="skip",
                    visible=True,
                    name=""
                ))

            # faces
            for i in range(len(_vals)):
                if not (max(_el_lo[i], show_lo) < min(_el_hi[i], show_hi)):
                    continue
                v = _vals[i]
                if not np.isfinite(v):
                    continue
                x0, x1 = float(_az_lo[i]), float(_az_hi[i])
                y0, y1 = float(_el_lo[i]), float(_el_hi[i])
                if mode == "power":
                    value_str = f"{v:.3f}{unit_str}"; label_title = "Power"
                else:
                    value_str = f"{v:.2f}%"; label_title = "Data loss"
                hover_text = (
                    f"Az: {x0:.1f}–{x1:.1f}°<br>"
                    f"El: {y0:.1f}–{y1:.1f}°<br>"
                    f"{label_title}: {value_str}"
                )
                traces.append(go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[y0, y0, y1, y1, y0],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    fill="toself",
                    fillcolor=_rgba_for_value(v),
                    hoveron="fills",
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                    name=""
                ))

            # borders (modernized previously)
            shapes = []
            if draw_cell_borders:
                line_w = max(1, int(round(border_width*2)))
                if not using_subset:
                    for elb in el_edges:
                        seg_y = float(elb)
                        shapes.append(dict(
                            type="line", x0=0, x1=360, y0=seg_y, y1=seg_y,
                            line=dict(color=border_color, width=line_w),
                            opacity=border_alpha, layer="above"
                        ))
                    for i_ring, n_in_ring in enumerate(cells_per_ring):
                        el0 = float(el_edges[i_ring])
                        el1 = float(el_edges[i_ring + 1])
                        seg_y0 = max(el0, show_lo)
                        seg_y1 = min(el1, show_hi)
                        if seg_y0 >= seg_y1:
                            continue
                        step = 360 // int(n_in_ring)
                        for az in np.arange(0, 360, step):
                            shapes.append(dict(
                                type="line", x0=float(az), x1=float(az), y0=seg_y0, y1=seg_y1,
                                line=dict(color=border_color, width=line_w),
                                opacity=border_alpha, layer="above"
                            ))
                else:
                    rnd = 6; eps = 10.0 ** (-rnd)
                    band_map: Dict[Tuple[float, float], Dict[str, Any]] = {}
                    for i in range(len(grid_info)):
                        lo_el = float(el_lo_cells[i]); hi_el = float(el_hi_cells[i])
                        if not (max(lo_el, show_lo) < min(hi_el, show_hi)):
                            continue
                        key = (round(lo_el, rnd), round(hi_el, rnd))
                        item = band_map.get(key)
                        if item is None:
                            band_map[key] = {"y0": lo_el, "y1": hi_el, "intervals": [], "lon_edges": []}
                            item = band_map[key]
                        az0 = float(az_lo_cells[i]); az1 = float(az_hi_cells[i])
                        item["intervals"].append((az0, az1))
                        item["lon_edges"].extend([az0, az1])

                    def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
                        """Merge overlapping azimuth segments within one elevation band."""
                        if not intervals: return []
                        norm = [(min(a, b), max(a, b)) for a, b in intervals]
                        norm.sort(key=lambda t: (t[0], t[1]))
                        merged: List[Tuple[float, float]] = [norm[0]]
                        for s, e in norm[1:]:
                            ls, le = merged[-1]
                            if s <= le + eps: merged[-1] = (ls, max(le, e))
                            else: merged.append((s, e))
                        return merged

                    for (y0_raw, y1_raw), info in band_map.items():
                        y0 = max(info["y0"], show_lo); y1 = min(info["y1"], show_hi)
                        if y0 >= y1: continue
                        uniq_lons = sorted(set(round(float(x), rnd) for x in info["lon_edges"]))
                        for L in uniq_lons:
                            shapes.append(dict(
                                type="line", x0=float(L), x1=float(L), y0=float(y0), y1=float(y1),
                                line=dict(color=border_color, width=line_w),
                                opacity=border_alpha, layer="above"
                            ))
                        merged = _merge_intervals(info["intervals"])
                        for a, b in merged:
                            shapes.append(dict(
                                type="line", x0=float(a), x1=float(b), y0=float(y0), y1=float(y0),
                                line=dict(color=border_color, width=line_w),
                                opacity=border_alpha, layer="above"
                            ))
                            shapes.append(dict(
                                type="line", x0=float(a), x1=float(b), y0=float(y1), y1=float(y1),
                                line=dict(color=border_color, width=line_w),
                                opacity=border_alpha, layer="above"
                            ))

            if elev_range is not None:
                xr = [0.0, 360.0]; yr = [show_lo, show_hi]
            else:
                pad_x = 360.0 * (tight_pad if tight else 0.02)
                pad_y = 90.0  * (tight_pad if tight else 0.02)
                xr = [-pad_x, 360.0 + pad_x]; yr = [-pad_y,  90.0 + pad_y]

            layout = dict(
                title=title or f"S.1586-1 Hemisphere — {'Power' if mode=='power' else 'Data loss'} (rect)",
                xaxis=dict(title="Azimuth [deg]", range=xr, constrain="domain"),
                yaxis=dict(title="Elevation [deg]", range=yr),
                margin=dict(l=60, r=20, t=60, b=60),
                paper_bgcolor="white",
                plot_bgcolor="white",
                shapes=shapes if draw_cell_borders else None,
            )
            fig = go.Figure(data=traces, layout=layout)

        fig.update_traces(name="")  # never show "trace N" in hover headers

        if save_html:
            pio.write_html(fig, save_html, include_plotlyjs=True, full_html=True)

        if show:
            cfg = {} if (interactive is None or interactive is True) else {"staticPlot": True}
            fig.show(config=cfg)
        return (fig, vals_q) if return_values else fig

    # ============================ MATPLOTLIB ============================
    _mpl_backend(interactive)
    fig, ax = _new_mpl_subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=cmin, vmax=cmax)

    if projection.lower() == "polar":
        # --- polar (MPL) faces already use cell polygons; we keep that.
        from matplotlib.patches import Polygon

        def _cell_poly(az0, az1, el0, el1) -> np.ndarray:
            """Four-vertex polygon for a cell in polar display coordinates."""
            t0 = _az_to_plot_angle_rad(az0)
            t1 = _az_to_plot_angle_rad(az1)
            r0 = _r_from_el(el0, radial_mapping, invert=invert_polar)
            r1 = _r_from_el(el1, radial_mapping, invert=invert_polar)
            return np.array([
                [r0*np.cos(t0), r0*np.sin(t0)],
                [r0*np.cos(t1), r0*np.sin(t1)],
                [r1*np.cos(t1), r1*np.sin(t1)],
                [r1*np.cos(t0), r1*np.sin(t0)],
            ])

        _az_lo, _az_hi = az_lo_cells, az_hi_cells
        _el_lo, _el_hi = el_lo_cells, el_hi_cells
        _vals = vals_num

        for i in range(len(_vals)):
            if not (max(_el_lo[i], show_lo) < min(_el_hi[i], show_hi)):
                continue
            poly = Polygon(
                _cell_poly(_az_lo[i], _az_hi[i], _el_lo[i], _el_hi[i]),
                closed=True,
                facecolor=cmap_obj(norm(_vals[i])),
                edgecolor=edgecolor, linewidth=linewidth, alpha=alpha
            )
            ax.add_patch(poly)

        # --- borders: S.1586 when no grid_info; grid_info-driven otherwise
        if draw_cell_borders:
            if not using_subset:
                tt = np.linspace(0, 2*np.pi, max(180, border_ring_samples))
                for elb in el_edges:
                    rr = _r_from_el(elb, radial_mapping, invert=invert_polar)
                    ax.plot(rr*np.cos(tt), rr*np.sin(tt),
                            color=border_color, linewidth=border_width, alpha=border_alpha)
                for i_ring, n_in_ring in enumerate(cells_per_ring):
                    el0 = el_edges[i_ring]
                    el1 = el_edges[i_ring + 1]
                    r0 = _r_from_el(el0, radial_mapping, invert=invert_polar)
                    r1 = _r_from_el(el1, radial_mapping, invert=invert_polar)
                    step = 360 // int(n_in_ring)
                    for az in np.arange(0, 360, step):
                        t = _az_to_plot_angle_rad(float(az))
                        ax.plot([r0*np.cos(t), r1*np.cos(t)],
                                [r0*np.sin(t), r1*np.sin(t)],
                                color=border_color, linewidth=border_width, alpha=border_alpha)
            else:
                # Rings from unique elevation edges in grid_info
                tt = np.linspace(0, 2*np.pi, max(180, border_ring_samples))
                unique_els = np.unique(
                    np.round(np.concatenate([el_lo_cells, el_hi_cells]).astype(float), 6)
                )
                for elb in unique_els:
                    if not (show_lo <= elb <= show_hi):
                        continue
                    rr = _r_from_el(float(elb), radial_mapping, invert=invert_polar)
                    ax.plot(rr*np.cos(tt), rr*np.sin(tt),
                            color=border_color, linewidth=border_width, alpha=border_alpha)

                # Meridians per elevation band (radial segments at unique az edges)
                rnd = 6
                band_map: Dict[Tuple[float, float], Dict[str, Any]] = {}
                for i in range(len(grid_info)):
                    y0 = float(el_lo_cells[i]); y1 = float(el_hi_cells[i])
                    if not (max(y0, show_lo) < min(y1, show_hi)):
                        continue
                    key = (round(y0, rnd), round(y1, rnd))
                    got = band_map.get(key)
                    if got is None:
                        band_map[key] = {"y0": y0, "y1": y1, "lon_edges": []}
                        got = band_map[key]
                    got["lon_edges"].extend([float(az_lo_cells[i]), float(az_hi_cells[i])])

                for (_, _), info in band_map.items():
                    y0 = max(info["y0"], show_lo)
                    y1 = min(info["y1"], show_hi)
                    if y0 >= y1:
                        continue
                    r0 = _r_from_el(y0, radial_mapping, invert=invert_polar)
                    r1 = _r_from_el(y1, radial_mapping, invert=invert_polar)
                    uniq_lons = sorted(set(round(l, rnd) for l in info["lon_edges"]))
                    for L in uniq_lons:
                        t = _az_to_plot_angle_rad(float(L))
                        ax.plot([r0*np.cos(t), r1*np.cos(t)],
                                [r0*np.sin(t), r1*np.sin(t)],
                                color=border_color, linewidth=border_width, alpha=border_alpha)

        # Guides (outer circle + rays + labels) in compass azimuth orientation
        max_r = 1.0

        if (draw_guides is None and projection.lower() == "polar") or (draw_guides is True):
            tt = np.linspace(0, 2*np.pi, 361)
            ax.plot(
                np.cos(tt),
                np.sin(tt),
                color=guide_color,
                linewidth=guide_linewidth,
                alpha=guide_alpha,
            )

            def _label_spec(az_deg: float) -> dict[str, Any]:
                """Return outward label placement and text style for compass labels."""
                az_key = int(round(float(az_deg))) % 360

                if az_key == 0:
                    return dict(
                        extra_dx=0.04,
                        extra_dy=float(label_offset_extra),
                        rotation=0,
                        ha="center",
                        va="bottom",
                    )
                if az_key == 180:
                    return dict(
                        extra_dx=-0.04,
                        extra_dy=-float(label_offset_extra),
                        rotation=0,
                        ha="center",
                        va="top",
                    )
                if az_key == 90:
                    return dict(
                        extra_dx=float(label_offset_extra),
                        extra_dy=0.0,
                        rotation=270,
                        ha="left",
                        va="center",
                    )
                if az_key == 270:
                    return dict(
                        extra_dx=-float(label_offset_extra),
                        extra_dy=0.0,
                        rotation=90,
                        ha="right",
                        va="center",
                    )

                return dict(
                    extra_dx=0.0,
                    extra_dy=0.0,
                    rotation=0,
                    ha="center",
                    va="center",
                )

            def _ray(az_deg: float, label: str) -> float:
                """Draw one compass ray in Matplotlib and return required plot radius."""
                t = _az_to_plot_angle_rad(az_deg)
                bx, by = guide_length * np.cos(t), guide_length * np.sin(t)

                ax.plot(
                    [0.0, bx],
                    [0.0, by],
                    color=guide_color,
                    linewidth=guide_linewidth,
                    alpha=guide_alpha,
                )

                if show_axis_arrows:
                    if arrow_direction.lower() == "outward":
                        ax.annotate(
                            "",
                            xy=(bx, by),
                            xytext=(0.0, 0.0),
                            arrowprops=dict(
                                arrowstyle="->",
                                lw=guide_linewidth,
                                color=guide_color,
                                shrinkA=0,
                                shrinkB=0,
                                mutation_scale=14 * arrow_scale,
                            ),
                        )
                    else:
                        ax.annotate(
                            "",
                            xy=(0.0, 0.0),
                            xytext=(bx, by),
                            arrowprops=dict(
                                arrowstyle="->",
                                lw=guide_linewidth,
                                color=guide_color,
                                shrinkA=0,
                                shrinkB=0,
                                mutation_scale=14 * arrow_scale,
                            ),
                        )

                spec = _label_spec(az_deg)
                lx = bx + float(spec["extra_dx"])
                ly = by + float(spec["extra_dy"])
                ax.text(
                    lx,
                    ly,
                    label,
                    ha=spec["ha"],
                    va=spec["va"],
                    rotation=spec["rotation"],
                    rotation_mode="anchor",
                    color=guide_color,
                    fontsize=12,
                    fontweight="bold",
                    clip_on=False,
                    zorder=6,
                )

                return max(np.hypot(bx, by), np.hypot(lx, ly))

            max_r = max(max_r, _ray(0,   "Az 0°"))
            max_r = max(max_r, _ray(90,  "Az 90°"))
            max_r = max(max_r, _ray(180, "Az 180°"))
            max_r = max(max_r, _ray(270, "Az 270°"))

        # Restore outward compass labels and reserve room for them in the polar extent.
        max_r = max(max_r, float(guide_length))
        max_r *= (1.0 + float(tight_pad))

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([-max_r, max_r])
        ax.set_ylim([-max_r, max_r])
        ax.set_xticks([])
        ax.set_yticks([])

    else:
        # --- rectangular (MPL) --- (left as in your updated version)
        from matplotlib.patches import Polygon

        def _rect(az0, az1, el0, el1) -> np.ndarray:
            """Return rectangle vertices for one skycell in rect-projection axes."""
            return np.array([[az0, el0], [az1, el0], [az1, el1], [az0, el1]])

        _az_lo, _az_hi = az_lo_cells, az_hi_cells
        _el_lo, _el_hi = el_lo_cells, el_hi_cells
        _vals = vals_num

        for i in range(len(_vals)):
            if not (max(_el_lo[i], show_lo) < min(_el_hi[i], show_hi)):
                continue
            poly = Polygon(
                _rect(_az_lo[i], _az_hi[i], _el_lo[i], _el_hi[i]),
                closed=True,
                facecolor=cmap_obj(norm(_vals[i])),
                edgecolor=edgecolor, linewidth=linewidth, alpha=alpha
            )
            ax.add_patch(poly)

        if draw_cell_borders:
            if not using_subset:
                for elb in el_edges:
                    ax.axhline(elb, color=border_color, linewidth=border_width, alpha=border_alpha)
                for i_ring, n_in_ring in enumerate(cells_per_ring):
                    el0 = el_edges[i_ring]
                    el1 = el_edges[i_ring + 1]
                    step = 360 // int(n_in_ring)
                    for az in np.arange(0, 360, step):
                        ax.plot([az, az], [max(el0, show_lo), min(el1, show_hi)],
                                color=border_color, linewidth=border_width, alpha=border_alpha)
            else:
                rnd = 6; eps = 10.0 ** (-rnd)
                band_map: Dict[Tuple[float, float], Dict[str, Any]] = {}
                for i in range(len(grid_info)):
                    lo_el = float(el_lo_cells[i])
                    hi_el = float(el_hi_cells[i])
                    if not (max(lo_el, show_lo) < min(hi_el, show_hi)):
                        continue
                    key = (round(lo_el, rnd), round(hi_el, rnd))
                    item = band_map.get(key)
                    if item is None:
                        band_map[key] = {"y0": lo_el, "y1": hi_el, "intervals": [], "lon_edges": []}
                        item = band_map[key]
                    az0 = float(az_lo_cells[i]); az1 = float(az_hi_cells[i])
                    item["intervals"].append((az0, az1))
                    item["lon_edges"].extend([az0, az1])

                def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
                    """Merge overlapping azimuth segments within one elevation band."""
                    if not intervals: return []
                    norm = [(min(a, b), max(a, b)) for a, b in intervals]
                    norm.sort(key=lambda t: (t[0], t[1]))
                    merged: List[Tuple[float, float]] = [norm[0]]
                    for s, e in norm[1:]:
                        ls, le = merged[-1]
                        if s <= le + eps: merged[-1] = (ls, max(le, e))
                        else: merged.append((s, e))
                    return merged

                for (y0_raw, y1_raw), info in band_map.items():
                    y0 = max(info["y0"], show_lo); y1 = min(info["y1"], show_hi)
                    if y0 >= y1: continue
                    uniq_lons = sorted(set(round(float(x), rnd) for x in info["lon_edges"]))
                    for L in uniq_lons:
                        ax.plot([float(L), float(L)], [float(y0), float(y1)],
                                color=border_color, linewidth=border_width, alpha=border_alpha)
                    merged = _merge_intervals(info["intervals"])
                    for a, b in merged:
                        ax.plot([float(a), float(b)], [float(y0), float(y0)],
                                color=border_color, linewidth=border_width, alpha=border_alpha)
                        ax.plot([float(a), float(b)], [float(y1), float(y1)],
                                color=border_color, linewidth=border_width, alpha=border_alpha)

        if elev_range is not None:
            ax.set_xlim(0.0, 360.0)
            ax.set_ylim(show_lo, show_hi)
        else:
            pad_x = 360.0 * (tight_pad if tight else 0.02)
            pad_y = 90.0  * (tight_pad if tight else 0.02)
            ax.set_xlim(-pad_x, 360.0 + pad_x)
            ax.set_ylim(-pad_y,  90.0 + pad_y)

        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel("Elevation [deg]")
        ax.set_aspect(360 / 90)

        # Ensure the lower elevation bound is visibly ticked — the default
        # MaxNLocator picks round values (20, 30, ...) and silently hides
        # the operational-range minimum (e.g. 15°), which is the most
        # load-bearing number on this axis.
        _y_lo, _y_hi = ax.get_ylim()
        _auto_ticks = [float(t) for t in ax.get_yticks() if _y_lo <= float(t) <= _y_hi]
        if not _auto_ticks or abs(_auto_ticks[0] - _y_lo) > 1e-6:
            _auto_ticks = [float(_y_lo)] + [t for t in _auto_ticks if t > _y_lo + 1e-6]
            ax.set_yticks(_auto_ticks)

        # Matplotlib's rectangular projection already renders clean native
        # axes (spines + ticks + labels) matching the Plotly rectangular
        # layout. The arrow-guides only make sense for the polar projection;
        # drawing them on top of the native rectangular axes produces a
        # duplicated "arrow + spine" look. Require draw_guides=True to
        # explicitly opt in on rect.
        if draw_guides is True:
            def _axis_arrow(x0, y0, x1, y1, label):
                """Draw one rectangular guide axis with optional directional arrow."""
                ax.plot([x0, x1], [y0, y1], color=guide_color,
                        linewidth=guide_linewidth, alpha=guide_alpha)
                if show_axis_arrows:
                    if arrow_direction.lower() == "outward":
                        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                    arrowprops=dict(arrowstyle="->", lw=guide_linewidth,
                                                    color=guide_color, shrinkA=0, shrinkB=0,
                                                    mutation_scale=14*arrow_scale))
                    else:
                        ax.annotate("", xy=(x0, y0), xytext=(x1, y1),
                                    arrowprops=dict(arrowstyle="->", lw=guide_linewidth,
                                                    color=guide_color, shrinkA=0, shrinkB=0,
                                                    mutation_scale=14*arrow_scale))
                ax.text(x1 + 4, y1 + 2, label,
                        color=guide_color, fontsize=12, weight="bold",
                        ha="left", va="bottom")
            _axis_arrow(0, 0, 360, 0, "Az")
            _axis_arrow(0, 0, 0, 90,  "El")

    # Title & colorbar
    ax.set_title(
        title or f"S.1586-1 Hemisphere — {'Power' if mode=='power' else 'Data loss'} ({projection})",
        pad=18,
    )
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.05)
        cbar.set_label(cbar_title)
    if projection.lower() == "rect":
        fig.subplots_adjust(
            left=0.10,
            right=0.88 if colorbar else 0.98,
            bottom=0.11,
            top=0.90,
        )
    else:
        fig.subplots_adjust(
            left=0.06,
            right=0.82 if colorbar else 0.95,
            bottom=0.08,
            top=0.87,
        )
    if show:
        plt.show()
    elif not return_values:
        plt.close(fig)
    return (fig, vals_q) if return_values else fig




# -----------------------------------------------------------------------------
# 3D hemisphere (Plotly or Matplotlib)
# -----------------------------------------------------------------------------

def plot_hemisphere_3D(
    data: Any,
    *,
    # ---------- SUBSET / DISPLAY CROP ----------
    grid_info: np.ndarray | None = None,        # optional clipped subset; order must match data cell axis
    elev_range: Tuple[float | Any, float | Any] | None = None,  # display crop only (deg or Quantity)

    # ---------- STATISTICS ----------
    worst_percent: float = 2.0,                 # mode="power": 100 - worst_percent percentile (e.g. 98th)
    mode: str | bool = "power",                 # "power" | "data_loss"  OR  True->data_loss / False->power
    protection_criterion: Any | None = None,    # required for data_loss (float or Quantity)
    cell_axis: int = -1,                        # axis with skycells (2334 or len(grid_info))

    # ---------- COLOR ----------
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,

    # ---------- CELL FACE VISUALS ----------
    edgecolor: str = "none",
    linewidth: float = 0.0,
    alpha: float = 1.0,

    # ---------- LABELS / COLORBAR ----------
    title: str | None = None,
    colorbar: bool = True,

    # ---------- GUIDES ----------
    draw_guides: bool = True,
    guide_color: str = "#111111",
    guide_alpha: float = 1.0,
    guide_linewidth: float = 2.2,
    guide_length: float = 1.5,                  # sphere radius is 1; guides extend beyond it
    show_axis_arrows: bool = True,
    axis_arrow_size: float = 0.08,              # base size for arrowheads
    arrow_direction: str = "outward",           # "outward" | "inward"
    arrow_scale: float = 1.5,                   # multiplies arrow size
    label_offset_extra: float = 0.03,           # extra label offset beyond arrow tip

    # ---------- SKY-CELL BORDERS ----------
    draw_cell_borders: bool = True,
    border_color: str = "#1f2937",
    border_width: float = 1.0,
    border_alpha: float = 1.0,
    border_ring_samples: int = 180,

    # ---------- CULL BACK FACES ----------
    front_only: bool = False,

    # ---------- CAMERA / VIEW ----------
    elev: float = 45.0,                         # degrees above horizon
    azim: float = 165.0,                        # 0→+x, 90→+y (matches S.1586 azimuth)
    z_aspect: float = 0.65,                     # compress vertical a bit
    tight: bool = True,                         # trim margins to content
    tight_pad: float = 0.02,                    # small fractional padding
    camera_distance_factor: float = 0.9,        # Plotly camera distance multiplier
    plotly_projection: str = "perspective",     # or "orthographic"

    # ---------- HOVER ----------
    show_hover: bool = True,
    plotly_hover_mode: str = "rich",           # "rich" | "single_trace" | "none"
    hover_precision: int = 2,
    hover_offset_3d: float = 0.06,              # shift per-cell hover target toward the camera
    hover_marker_size: int = 18,                # invisible point size to make hovering easy
    hover_xytext: Tuple[int, int] = (18, 18),   # MPL tooltip offset (pixels) away from cursor

    # ---------- ENGINE / INTERACTIVITY ----------
    engine: str = "auto",                       # "auto" | "mpl" | "plotly"
    interactive: bool | None = None,            # kept for API parity
    figsize: tuple[float, float] = (8.5, 7.0),  # Matplotlib only
    show: bool = True,
    return_values: bool = False,

    # ---------- EXPORT ----------
    export_html_path: str | None = None,        # if set, write HTML
    html_include_plotlyjs: bool | str = True,   # standalone export modes: True or "cdn"
    html_auto_open: bool = False,
    export_png_path: str | None = None,         # needs kaleido
    png_width: int = 1600,
    png_height: int = 1200,
    png_scale: float = 1.5,
):
    """
    Plot a 3D upper-hemisphere map coloured by a per-cell sky statistic.

    Parameters
    ----------
    data : array-like or Quantity
        Sample array with one sky-cell axis and one or more sample axes. The
        sky-cell axis must have length 2334 when ``grid_info`` is omitted, or
        length ``len(grid_info)`` when a custom grid is provided.
    grid_info : structured ndarray, optional
        Per-cell geometry with fields ``cell_lon_low``, ``cell_lon_high``,
        ``cell_lat_low``, and ``cell_lat_high`` in degrees.
    elev_range : tuple, optional
        Display-only elevation crop ``(low, high)`` in degrees or angle
        quantities. Data outside the visible range are not re-binned.
    worst_percent : float, optional
        Worst-case percentage used in ``mode="power"``. The plotted value is
        the ``100 - worst_percent`` percentile of the per-cell samples.
    mode : {"power", "data_loss"} or bool, optional
        Statistic to render. Boolean values are accepted for backward
        compatibility, where ``True`` maps to ``"data_loss"``.
    protection_criterion : float or Quantity, optional
        Threshold used in ``mode="data_loss"``.
    cell_axis : int, optional
        Axis index of the sky-cell dimension.
    plotly_hover_mode : {"rich", "single_trace", "none"}, optional
        Hover behavior for Plotly figures. ``"rich"`` preserves per-cell hover
        label styling, ``"single_trace"`` reduces trace count by using one
        invisible marker trace for all cells, and ``"none"`` disables Plotly
        hover markers entirely.
    export_html_path : str, optional
        If provided, write a standalone HTML file.
    html_include_plotlyjs : {True, "cdn"}, optional
        Plotly JavaScript inclusion mode for standalone HTML export. ``True``
        bundles plotly.js; ``"cdn"`` references the Plotly CDN. Embed-only
        modes such as ``False`` are rejected for file export by this wrapper.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Rendered figure.
    values : ndarray or Quantity, optional
        Returned only when ``return_values=True``. Contains the per-cell values
        used for colouring, in the same domain/unit as the input data for power
        mode and in percent for data-loss mode.

    Raises
    ------
    ValueError
        Raised when the cell axis is invalid, the grid description does not
        match the data, or the requested statistic/export mode is unsupported.

    Notes
    -----
    - Percentiles use nearest-rank order statistics on the native data values.
      For dB-valued power samples this is exactly invariant under monotonic
      dB-to-linear conversion.
    - Threshold exceedance probabilities are computed on finite native-domain
      samples only.
    - Additive physical-power statistics are not computed by this function.
    - ``html_include_plotlyjs=False`` is intentionally rejected for file export
      because the resulting HTML would only work when embedded into a page that
      already loads Plotly.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # local import for safety

    # --- input normalisation ---
    arr, unit = _to_plain_array(data)
    if arr.ndim < 2:
        raise ValueError(f"'data' must have at least 2 dims (samples × cells). Got {arr.shape}.")
    if cell_axis < 0:
        cell_axis = arr.ndim + cell_axis
    if not (0 <= cell_axis < arr.ndim):
        raise ValueError(f"cell_axis out of range for array with shape {arr.shape}.")

    # Move cells to last axis; flatten the rest → (samples_flat, C0)
    arr = np.moveaxis(arr, cell_axis, -1)
    C0 = arr.shape[-1]
    samples_flat = arr.reshape(-1, C0)

    # Reference grid (for borders/guides and for "full grid" data layout)
    az_lo_ref, az_hi_ref, el_lo_ref, el_hi_ref, el_edges, cells_per_ring = _s1586_cells()

    # Geometry from full grid or provided subset
    if grid_info is None:
        if C0 != 2334:
            raise ValueError(f"When grid_info is not provided, cell axis must be 2334 (got {C0}).")
        az_lo_all, az_hi_all, el_lo_all, el_hi_all = az_lo_ref, az_hi_ref, el_lo_ref, el_hi_ref
        cover_lo, cover_hi = 0.0, 90.0
        using_subset = False
    else:
        if C0 != len(grid_info):
            raise ValueError(f"cell axis length ({C0}) must equal len(grid_info) ({len(grid_info)}).")
        az_lo_all = grid_info["cell_lon_low"].astype(float)
        az_hi_all = grid_info["cell_lon_high"].astype(float)
        el_lo_all = grid_info["cell_lat_low"].astype(float)
        el_hi_all = grid_info["cell_lat_high"].astype(float)
        cover_lo = float(np.nanmin(el_lo_all)) if len(el_lo_all) else 0.0
        cover_hi = float(np.nanmax(el_hi_all)) if len(el_hi_all) else 0.0
        using_subset = True

    # --- display crop in elevation ---
    if elev_range is not None:
        lo, hi = elev_range
        try:
            lo_deg = float(lo.to_value(u.deg)) if hasattr(lo, "to") else float(getattr(lo, "value", lo))
            hi_deg = float(hi.to_value(u.deg)) if hasattr(hi, "to") else float(getattr(hi, "value", hi))
        except Exception:
            lo_deg = float(getattr(lo, "value", lo))
            hi_deg = float(getattr(hi, "value", hi))
        if lo_deg > hi_deg:
            lo_deg, hi_deg = hi_deg, lo_deg
        req_lo, req_hi = max(0.0, lo_deg), min(90.0, hi_deg)
    else:
        req_lo, req_hi = cover_lo, cover_hi

    show_lo = max(req_lo, cover_lo)
    show_hi = min(req_hi, cover_hi)
    if elev_range is not None and (req_lo < cover_lo or req_hi > cover_hi):
        missing = []
        if req_lo < cover_lo:
            missing.append(f"[{req_lo:.1f}°, {cover_lo:.1f}°]")
        if req_hi > cover_hi:
            missing.append(f"[{cover_hi:.1f}°, {req_hi:.1f}°]")
        if missing:
            warnings.warn(
                "Requested elev_range extends beyond data coverage; "
                f"no data for: {', '.join(missing)}. Showing {show_lo:.1f}°–{show_hi:.1f}°."
            )

    # Keep only cells overlapping the visible elevation band
    keep_mask = (el_hi_all > show_lo) & (el_lo_all < show_hi)
    if not np.any(keep_mask):
        raise ValueError("No cells remain after elevation cropping.")

    az_lo = az_lo_all[keep_mask]
    az_hi = az_hi_all[keep_mask]
    el_lo = el_lo_all[keep_mask]
    el_hi = el_hi_all[keep_mask]
    arr_kept = samples_flat[:, keep_mask]
    C = arr_kept.shape[1]

    # --- per-cell statistic ---
    if isinstance(mode, bool):
        use_data_loss = bool(mode)
    else:
        use_data_loss = str(mode).lower() in {"data_loss", "dataloss", "loss", "data", "dl"}

    if use_data_loss:
        if protection_criterion is None:
            raise ValueError("mode='data_loss' requires protection_criterion.")
        thr_num = _resolve_threshold_numeric(protection_criterion, unit)
        vals = _finite_exceedance_percent_axis0(arr_kept, thr_num)

        vals = np.clip(vals, 0.0, 100.0)
        cell_vals_q = vals
        vals_for_cmap = vals
        default_vmin = np.nanmin(vals_for_cmap)
        default_vmax = np.nanmax(vals_for_cmap)
        colorbar_label = "Data loss [%]"
        default_title = f"S.1586-1 Hemisphere — Data Loss (thr={thr_num:.{hover_precision}f}{'' if unit is None else ' ' + str(unit)})"
    else:
        if not (0.0 < worst_percent < 100.0):
            raise ValueError("worst_percent must be in (0, 100).")
        percentile = 100.0 - float(worst_percent)
        cell_vals = _nearest_rank_percentile_axis0(arr_kept, percentile)

        if unit is not None:
            cell_vals_q = cell_vals * unit
            vals_for_cmap = cell_vals_q.value
            colorbar_label = f"Power [{unit}]"
        else:
            cell_vals_q = cell_vals
            vals_for_cmap = cell_vals
            colorbar_label = "Power"

        default_vmin = np.nanmin(vals_for_cmap)
        default_vmax = np.nanmax(vals_for_cmap)
        default_title = f"S.1586-1 Hemisphere — Power ({percentile:.0f}th percentile)"

    # Colour limits (explicit or from data)
    cmin = default_vmin if vmin is None else float(vmin)
    cmax = default_vmax if vmax is None else float(vmax)
    if (not np.isfinite(cmin)) or (not np.isfinite(cmax)):
        raise ValueError("Color limits are not finite (check your input).")
    if cmax == cmin:
        cmin, cmax = float(cmin), float(cmax + 1e-9)

    # --- extents, camera, back-face culling ---
    arrow_len = float(arrow_scale) * float(axis_arrow_size) if show_axis_arrows else 0.0
    label_shift = float(label_offset_extra) + (arrow_len if show_axis_arrows else 0.0)
    base_extent = float(guide_length)
    extent_xy = (base_extent + label_shift) * (1.0 + float(tight_pad) if tight else 1.0)
    extent_z  = (base_extent + label_shift * 1.25) * (1.0 + float(tight_pad) if tight else 1.0)

    eye_vec = _eye_from_elev_azim(elev, azim, distance=1.0)
    eye_norm = eye_vec / (np.linalg.norm(eye_vec) + 1e-15)

    def _is_front(p):
        """Return True if point/vector `p` is on the camera-facing hemisphere."""
        return (np.dot(p, eye_norm) >= 0.0)

    # --- engine selection ---
    use_plotly = False
    if engine == "plotly":
        use_plotly = True
    elif engine == "auto":
        try:
            import plotly  # noqa: F401
            if interactive is not False:
                use_plotly = True
        except Exception:
            use_plotly = False

    def _fmt_val(v: float) -> str:
        """Format hover value string in either power or data-loss mode."""
        if use_data_loss:
            return f"{v:.{hover_precision}f} %"
        return f"{v:.{hover_precision}f}{'' if unit is None else ' ' + str(unit)}"

    hover_mode = str(plotly_hover_mode).lower().strip()
    if hover_mode not in {"rich", "single_trace", "none"}:
        raise ValueError("plotly_hover_mode must be 'rich', 'single_trace', or 'none'.")

    # ============================ PLOTLY ============================
    if use_plotly:
        import plotly.graph_objects as go

        # colormap for Plotly Mesh3d
        mpl_cmap = plt.get_cmap(cmap)
        colorscale = [
            [t, f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"]
            for t, (r, g, b, a) in zip(np.linspace(0, 1, 256), mpl_cmap(np.linspace(0, 1, 256)))
        ]

        # Build a single Mesh3d for faces (fast). Hover is provided by separate points.
        X: List[float] = []
        Y: List[float] = []
        Z: List[float] = []
        I: List[int] = []
        J: List[int] = []
        K: List[int] = []
        INT: List[float] = []
        tri_idx = 0

        # Per-cell hover points (centroid nudged toward the camera).
        collect_hover = bool(show_hover) and (hover_mode != "none")
        centroids_xyz: List[tuple[float, float, float]] = []
        cell_text: List[str] = []
        cell_bg: List[str] = []
        cell_font_color: List[str] = []

        for pos in range(C):
            az0, az1 = float(az_lo[pos]), float(az_hi[pos])
            el0, el1 = float(el_lo[pos]), float(el_hi[pos])

            if not (max(el0, show_lo) < min(el1, show_hi)):
                continue

            ax4 = np.array([az0, az1, az1, az0], dtype=float)
            el4 = np.array([el0, el0, el1, el1], dtype=float)
            x4, y4, z4 = _cart_from_azel(ax4, el4, r=1.0)

            if front_only:
                cx_tmp, cy_tmp, cz_tmp = float(np.mean(x4)), float(np.mean(y4)), float(np.mean(z4))
                if not _is_front((cx_tmp, cy_tmp, cz_tmp)):
                    continue

            v = float(vals_for_cmap[pos])

            # Tri A: (0,1,2), Tri B: (0,2,3)
            X.extend([x4[0], x4[1], x4[2]]); Y.extend([y4[0], y4[1], y4[2]]); Z.extend([z4[0], z4[1], z4[2]])
            I.append(tri_idx+0); J.append(tri_idx+1); K.append(tri_idx+2); INT.extend([v, v, v]); tri_idx += 3
            X.extend([x4[0], x4[2], x4[3]]); Y.extend([y4[0], y4[2], y4[3]]); Z.extend([z4[0], z4[2], z4[3]])
            I.append(tri_idx+0); J.append(tri_idx+1); K.append(tri_idx+2); INT.extend([v, v, v]); tri_idx += 3

            if collect_hover:
                # Centroid → normalise to sphere → nudge along camera eye direction
                cx, cy, cz = float(np.mean(x4)), float(np.mean(y4)), float(np.mean(z4))
                norm = np.sqrt(cx * cx + cy * cy + cz * cz) + 1e-15
                cx, cy, cz = cx / norm, cy / norm, cz / norm
                cx += hover_offset_3d * eye_norm[0]
                cy += hover_offset_3d * eye_norm[1]
                cz += hover_offset_3d * eye_norm[2]
                centroids_xyz.append((cx, cy, cz))

                label = (
                    f"Az: {az0:.1f}–{az1:.1f}°<br>"
                    f"El: {el0:.1f}–{el1:.1f}°<br>"
                    f"{'Data loss' if use_data_loss else 'Power'}: {_fmt_val(v)}"
                )
                cell_text.append(label)

                if hover_mode == "rich":
                    rgba, rgb01 = _rgba_from_value(v, cmin, cmax, cmap, alpha_val=1.0)
                    cell_bg.append(rgba)
                    cell_font_color.append(_hover_font_color_from_rgb(rgb01))

        traces: List[go.BaseTraceType] = []

        # Surface (colourbar lives here). Hover is disabled on the mesh itself.
        traces.append(go.Mesh3d(
            x=X, y=Y, z=Z,
            i=I, j=J, k=K,
            intensity=INT, intensitymode="vertex",
            colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=bool(colorbar),
            colorbar=dict(title=colorbar_label),
            flatshading=True,
            opacity=float(alpha),
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
            hoverinfo="skip",
            name=""
        ))

        # Invisible hover points for Plotly interactivity.
        if collect_hover and len(centroids_xyz) > 0:
            if hover_mode == "rich":
                for (cx, cy, cz), text, bg, fcol in zip(
                    centroids_xyz,
                    cell_text,
                    cell_bg,
                    cell_font_color,
                ):
                    traces.append(go.Scatter3d(
                        x=[cx], y=[cy], z=[cz],
                        mode="markers",
                        marker=dict(size=int(hover_marker_size), opacity=0.0),
                        text=[text],
                        hovertemplate="%{text}<extra></extra>",
                        hoverlabel=dict(
                            bgcolor=bg,
                            bordercolor=bg,
                            font=dict(color=fcol, size=14),
                            align="left",
                        ),
                        showlegend=False,
                        name="",
                    ))
            else:
                hover_xyz = np.asarray(centroids_xyz, dtype=float)
                traces.append(go.Scatter3d(
                    x=hover_xyz[:, 0],
                    y=hover_xyz[:, 1],
                    z=hover_xyz[:, 2],
                    mode="markers",
                    marker=dict(size=int(hover_marker_size), opacity=0.0),
                    text=cell_text,
                    hovertemplate="%{text}<extra></extra>",
                    hoverlabel=dict(
                        bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#d1d5db",
                        font=dict(color="black", size=13),
                        align="left",
                    ),
                    showlegend=False,
                    name="",
                ))

        # Optional borders (clipped to [show_lo, show_hi])
        if draw_cell_borders:
            line_w = max(1, int(round(border_width * 2)))

            if not using_subset:
                # ---- S.1586 canonical borders ----
                tt = np.linspace(0, 2*np.pi, max(12, int(border_ring_samples)))

                # rings
                ring_elows_all = el_edges[:-1]
                ring_keep = (ring_elows_all >= show_lo) & (ring_elows_all <= show_hi)
                rings_x: List[float] = []; rings_y: List[float] = []; rings_z: List[float] = []
                for elb in ring_elows_all[ring_keep]:
                    x = np.cos(np.radians(elb)) * np.cos(tt)
                    y = np.cos(np.radians(elb)) * np.sin(tt)
                    z = np.full_like(tt, np.sin(np.radians(elb)))
                    for idx in range(len(tt) - 1):
                        xm = 0.5*(x[idx] + x[idx+1]); ym = 0.5*(y[idx] + y[idx+1]); zm = 0.5*(z[idx] + z[idx+1])
                        if (not front_only) or (np.dot([xm, ym, zm], eye_norm) >= 0.0):
                            rings_x += [x[idx], x[idx+1], None]
                            rings_y += [y[idx], y[idx+1], None]
                            rings_z += [z[idx], z[idx+1], None]
                if rings_x:
                    traces.append(go.Scatter3d(
                        x=rings_x, y=rings_y, z=rings_z, mode="lines",
                        line=dict(color=border_color, width=line_w),
                        opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                    ))

                # meridians
                mer_x: List[float] = []; mer_y: List[float] = []; mer_z: List[float] = []
                for idx_ring, n_in_ring in enumerate(cells_per_ring):
                    el0_ring = float(el_edges[idx_ring]); el1_ring = float(el_edges[idx_ring+1])
                    seg_el0 = max(el0_ring, show_lo)
                    seg_el1 = min(el1_ring, show_hi)
                    if seg_el0 >= seg_el1:
                        continue
                    step = 360 // int(n_in_ring)
                    for az in np.arange(0, 360, step):
                        x0,y0,z0 = _cart_from_azel(az, seg_el0, r=1.0)
                        x1,y1,z1 = _cart_from_azel(az, seg_el1, r=1.0)
                        xm, ym, zm = 0.5*(x0+x1), 0.5*(y0+y1), 0.5*(z0+z1)
                        if (not front_only) or (np.dot([xm, ym, zm], eye_norm) >= 0.0):
                            mer_x += [x0, x1, None]; mer_y += [y0, y1, None]; mer_z += [z0, z1, None]
                if mer_x:
                    traces.append(go.Scatter3d(
                        x=mer_x, y=mer_y, z=mer_z, mode="lines",
                        line=dict(color=border_color, width=line_w),
                        opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                    ))
            else:
                # ---- Borders derived from grid_info ----
                rnd = 6
                # a) RINGS: circles at every unique elevation edge
                unique_els = np.unique(np.round(np.concatenate([el_lo, el_hi]).astype(float), rnd))
                tt = np.linspace(0, 2*np.pi, max(12, int(border_ring_samples)))
                rings_x: List[float] = []; rings_y: List[float] = []; rings_z: List[float] = []
                for elb in unique_els:
                    if not (show_lo <= elb <= show_hi):
                        continue
                    x = np.cos(np.radians(elb)) * np.cos(tt)
                    y = np.cos(np.radians(elb)) * np.sin(tt)
                    z = np.full_like(tt, np.sin(np.radians(elb)))
                    for idx in range(len(tt) - 1):
                        xm = 0.5*(x[idx] + x[idx+1]); ym = 0.5*(y[idx] + y[idx+1]); zm = 0.5*(z[idx] + z[idx+1])
                        if (not front_only) or (np.dot([xm, ym, zm], eye_norm) >= 0.0):
                            rings_x += [x[idx], x[idx+1], None]
                            rings_y += [y[idx], y[idx+1], None]
                            rings_z += [z[idx], z[idx+1], None]
                if rings_x:
                    traces.append(go.Scatter3d(
                        x=rings_x, y=rings_y, z=rings_z, mode="lines",
                        line=dict(color=border_color, width=line_w),
                        opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                    ))

                # b) MERIDIANS: for each elevation band, draw radial segments at each unique lon edge
                band_map: Dict[Tuple[float,float], Dict[str, Any]] = {}
                for i in range(len(el_lo)):
                    y0 = float(el_lo[i]); y1 = float(el_hi[i])
                    if not (max(y0, show_lo) < min(y1, show_hi)):
                        continue
                    key = (round(y0, rnd), round(y1, rnd))
                    got = band_map.get(key)
                    if got is None:
                        band_map[key] = {"y0": y0, "y1": y1, "lon_edges": []}
                        got = band_map[key]
                    got["lon_edges"].extend([float(az_lo[i]), float(az_hi[i])])

                mer_x: List[float] = []; mer_y: List[float] = []; mer_z: List[float] = []
                for (_, _), info in band_map.items():
                    y0 = max(info["y0"], show_lo)
                    y1 = min(info["y1"], show_hi)
                    if y0 >= y1:
                        continue
                    uniq_lons = sorted(set(round(l, rnd) for l in info["lon_edges"]))
                    for L in uniq_lons:
                        x0,y0_,z0 = _cart_from_azel(float(L), y0, r=1.0)
                        x1,y1_,z1 = _cart_from_azel(float(L), y1, r=1.0)
                        xm, ym, zm = 0.5*(x0+x1), 0.5*(y0_+y1_), 0.5*(z0+z1)
                        if (not front_only) or (np.dot([xm, ym, zm], eye_norm) >= 0.0):
                            mer_x += [x0, x1, None]; mer_y += [y0_, y1_, None]; mer_z += [z0, z1, None]
                if mer_x:
                    traces.append(go.Scatter3d(
                        x=mer_x, y=mer_y, z=mer_z, mode="lines",
                        line=dict(color=border_color, width=line_w),
                        opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                    ))

        # Guides and labels
        def _add_text(pt, s):
            """Add one text label trace at 3D point `pt`."""
            traces.append(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]],
                mode="text", text=[s],
                textfont=dict(color=guide_color, size=16),
                hoverinfo="skip", showlegend=False, name=""
            ))

        if draw_guides:
            tt = np.linspace(0, 2*np.pi, 361)
            hx, hy, hz = np.cos(tt), np.sin(tt), np.zeros_like(tt)
            traces.append(go.Scatter3d(
                x=hx, y=hy, z=hz, mode="lines",
                line=dict(color=guide_color, width=7),
                hoverinfo="skip", showlegend=False, name=""
            ))

            def _ray(az_deg, el_deg, label=None, is_zenith: bool = False):
                """Draw one 3D guide ray and optional label/arrow cone."""
                x1, y1, z1 = _cart_from_azel(az_deg, el_deg, r=guide_length)
                traces.append(go.Scatter3d(
                    x=[0, x1], y=[0, y1], z=[0, z1],
                    mode="lines",
                    line=dict(color=guide_color, width=8),
                    hoverinfo="skip", showlegend=False, name=""
                ))
                nx, ny, nz = np.array([x1, y1, z1], float)
                L = float(arrow_scale) * float(axis_arrow_size)
                ray_len = max(np.linalg.norm([nx, ny, nz]), 1e-12)
                nx, ny, nz = nx/ray_len, ny/ray_len, nz/ray_len

                if show_axis_arrows:
                    traces.append(go.Cone(
                        x=[x1], y=[y1], z=[z1],
                        u=[(nx if arrow_direction.lower()=="outward" else -nx)*L],
                        v=[(ny if arrow_direction.lower()=="outward" else -ny)*L],
                        w=[(nz if arrow_direction.lower()=="outward" else -nz)*L],
                        anchor=("tail" if arrow_direction.lower()=="outward" else "tip"),
                        sizemode="absolute", sizeref=L,
                        showscale=False,
                        colorscale=[[0, guide_color], [1, guide_color]],
                        lighting=dict(ambient=1.0),
                        opacity=guide_alpha,
                        hoverinfo="skip",
                        name=""
                    ))
                    label_shift_local = L + float(label_offset_extra)
                else:
                    label_shift_local = float(label_offset_extra)

                if label:
                    extra_z = 0.25 * label_shift_local if is_zenith else 0.0
                    pt = (
                        x1 + nx*label_shift_local,
                        y1 + ny*label_shift_local,
                        z1 + nz*label_shift_local + extra_z
                    )
                    _add_text(pt, label)

            _ray(0,   0,   "Az 0°")
            _ray(90,  0,   "Az 90°")
            _ray(180, 0,   "Az 180°")
            _ray(270, 0,   "Az 270°")
            _ray(0,  90,   "Zenith", is_zenith=True)

        # Scene and camera
        xr = [-extent_xy, extent_xy]
        yr = [-extent_xy, extent_xy]
        zr = [0.0, extent_z]
        eye = _eye_from_elev_azim(elev, azim, distance=camera_distance_factor * max(xr[1], yr[1], zr[1]))

        layout = dict(
            title=title or default_title,
            scene=dict(
                xaxis=dict(visible=False, range=xr),
                yaxis=dict(visible=False, range=yr),
                zaxis=dict(visible=False, range=zr),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=z_aspect),
                camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                            projection=dict(type=plotly_projection)),
                bgcolor="white",
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        fig_p = go.Figure(data=traces, layout=layout)

        # Optional export
        if export_html_path:
            include_js = _normalize_plotly_html_include_mode(
                html_include_plotlyjs,
                context="plot_hemisphere_3D HTML export",
            )
            fig_p.write_html(export_html_path, include_plotlyjs=include_js, full_html=True)
            if html_auto_open:
                import os, webbrowser
                webbrowser.open(f"file://{os.path.abspath(export_html_path)}")
        if export_png_path:
            fig_p.write_image(export_png_path, width=png_width, height=png_height, scale=png_scale)

        if show:
            cfg = {} if (interactive is None or interactive is True) else {"staticPlot": True}
            fig_p.show(config=cfg)
        return (fig_p, cell_vals_q) if return_values else fig_p

    # ============================ MATPLOTLIB ============================
    norm = plt.Normalize(vmin=cmin, vmax=cmax)
    cmap_obj = plt.get_cmap(cmap)

    polys: List[np.ndarray] = []
    facecolors: List[tuple] = []
    poly_labels: List[str] = []      # text for hover
    poly_bg_colors: List[tuple] = [] # rgba colour used as tooltip background

    for pos in range(C):
        az0, az1 = float(az_lo[pos]), float(az_hi[pos])
        el0, el1 = float(el_lo[pos]), float(el_hi[pos])

        if not (max(el0, show_lo) < min(el1, show_hi)):
            continue

        ax4 = np.array([az0, az1, az1, az0], dtype=float)
        el4 = np.array([el0, el0, el1, el1], dtype=float)
        x4, y4, z4 = _cart_from_azel(ax4, el4, r=1.0)

        if front_only:
            cx, cy, cz = float(np.mean(x4)), float(np.mean(y4)), float(np.mean(z4))
            if not _is_front((cx, cy, cz)):
                continue

        v = float(vals_for_cmap[pos])
        polys.append(np.column_stack([x4, y4, z4]))
        fc = cmap_obj(norm(v))
        facecolors.append(fc)
        poly_labels.append(
            f"Az: {az0:.1f}–{az1:.1f}°\nEl: {el0:.1f}–{el1:.1f}°\n"
            f"{'Data loss' if use_data_loss else 'Power'}: {_fmt_val(v)}"
        )
        poly_bg_colors.append(fc)

    fig = _new_mpl_figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    coll = Poly3DCollection(
        polys, facecolors=facecolors,
        edgecolors=edgecolor, linewidths=linewidth, alpha=alpha
    )
    ax.add_collection3d(coll)

    # Axis ranges; keep z from 0 so horizon/guides remain visible
    ax.set_xlim([-extent_xy, extent_xy])
    ax.set_ylim([-extent_xy, extent_xy])
    ax.set_zlim([0.0, extent_z])
    try:
        ax.set_box_aspect((1, 1, z_aspect))
    except Exception:
        pass

    # Clean look
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_axis_off()
    except Exception:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis.line.set_color((1, 1, 1, 0))

    # Optional borders, clipped to [show_lo, show_hi]
    if draw_cell_borders:
        if not using_subset:
            # ---- S.1586 canonical borders ----
            t = np.linspace(0, 2*np.pi, max(12, int(border_ring_samples)))
            ring_elows_all = el_edges[:-1]
            ring_keep = (ring_elows_all >= show_lo) & (ring_elows_all <= show_hi)

            # Elevation rings
            ring_segs: List[np.ndarray] = []
            for elb in ring_elows_all[ring_keep]:
                x = np.cos(np.radians(elb)) * np.cos(t)
                y = np.cos(np.radians(elb)) * np.sin(t)
                z = np.full_like(t, np.sin(np.radians(elb)))
                for idx in range(len(t) - 1):
                    xm = 0.5*(x[idx] + x[idx+1]); ym = 0.5*(y[idx] + y[idx+1]); zm = 0.5*(z[idx] + z[idx+1])
                    if (not front_only) or (np.dot([xm, ym, zm], eye_norm) >= 0.0):
                        ring_segs.append(np.array([[x[idx], y[idx], z[idx]],
                                                   [x[idx+1], y[idx+1], z[idx+1]]]))
            if ring_segs:
                ring_coll = Line3DCollection(ring_segs, colors=border_color,
                                             linewidths=border_width, alpha=border_alpha)
                ax.add_collection3d(ring_coll)

            # Azimuth meridians (clipped vertically)
            mer_segs: List[np.ndarray] = []
            for idx_ring, n_in_ring in enumerate(cells_per_ring):
                el0_ring = float(el_edges[idx_ring]); el1_ring = float(el_edges[idx_ring+1])
                seg_el0 = max(el0_ring, show_lo)
                seg_el1 = min(el1_ring, show_hi)
                if seg_el0 >= seg_el1:
                    continue
                step = 360 // int(n_in_ring)
                for az in np.arange(0, 360, step):
                    x0,y0,z0 = _cart_from_azel(az, seg_el0, r=1.0)
                    x1,y1,z1 = _cart_from_azel(az, seg_el1, r=1.0)
                    xm, ym, zm = 0.5*(x0+x1), 0.5*(y0+y1), 0.5*(z0+z1)
                    if (not front_only) or (np.dot([xm, ym, zm], eye_norm) >= 0.0):
                        mer_segs.append(np.array([[x0,y0,z0],[x1,y1,z1]]))
            if mer_segs:
                mer_coll = Line3DCollection(mer_segs, colors=border_color,
                                            linewidths=border_width, alpha=border_alpha)
                ax.add_collection3d(mer_coll)
        else:
            # ---- Borders derived from grid_info ----
            rnd = 6
            t = np.linspace(0, 2*np.pi, max(12, int(border_ring_samples)))

            # a) RINGS: circles at unique elevation edges inside the visible band
            unique_els = np.unique(np.round(np.concatenate([el_lo, el_hi]).astype(float), rnd))
            ring_segs: List[np.ndarray] = []
            for elb in unique_els:
                if not (show_lo <= elb <= show_hi):
                    continue
                x = np.cos(np.radians(elb)) * np.cos(t)
                y = np.cos(np.radians(elb)) * np.sin(t)
                z = np.full_like(t, np.sin(np.radians(elb)))
                for idx in range(len(t) - 1):
                    xm = 0.5*(x[idx] + x[idx+1]); ym = 0.5*(y[idx] + y[idx+1]); zm = 0.5*(z[idx] + z[idx+1])
                    if (not front_only) or (np.dot([xm, ym, zm], eye_norm) >= 0.0):
                        ring_segs.append(np.array([[x[idx], y[idx], z[idx]],
                                                   [x[idx+1], y[idx+1], z[idx+1]]]))
            if ring_segs:
                ring_coll = Line3DCollection(ring_segs, colors=border_color,
                                             linewidths=border_width, alpha=border_alpha)
                ax.add_collection3d(ring_coll)

            # b) MERIDIANS: per elevation band, radial segments at unique lon edges
            band_map: Dict[Tuple[float,float], Dict[str, Any]] = {}
            for i in range(len(el_lo)):
                y0 = float(el_lo[i]); y1 = float(el_hi[i])
                if not (max(y0, show_lo) < min(y1, show_hi)):
                    continue
                key = (round(y0, rnd), round(y1, rnd))
                got = band_map.get(key)
                if got is None:
                    band_map[key] = {"y0": y0, "y1": y1, "lon_edges": []}
                    got = band_map[key]
                got["lon_edges"].extend([float(az_lo[i]), float(az_hi[i])])

            mer_segs: List[np.ndarray] = []
            for (_, _), info in band_map.items():
                y0 = max(info["y0"], show_lo)
                y1 = min(info["y1"], show_hi)
                if y0 >= y1:
                    continue
                uniq_lons = sorted(set(round(l, rnd) for l in info["lon_edges"]))
                for L in uniq_lons:
                    x0,y0_,z0 = _cart_from_azel(float(L), y0, r=1.0)
                    x1,y1_,z1 = _cart_from_azel(float(L), y1, r=1.0)
                    xm, ym, zm = 0.5*(x0+x1), 0.5*(y0_+y1_), 0.5*(z0+z1)
                    if (not front_only) or (np.dot([xm, ym, zm], eye_norm) >= 0.0):
                        mer_segs.append(np.array([[x0,y0_,z0],[x1,y1_,z1]]))
            if mer_segs:
                mer_coll = Line3DCollection(mer_segs, colors=border_color,
                                            linewidths=border_width, alpha=border_alpha)
                ax.add_collection3d(mer_coll)

    # Guides / labels (always draw the horizon circle so axes remain visible)
    if draw_guides:
        tt = np.linspace(0, 2*np.pi, 361)
        hx, hy, hz = np.cos(tt), np.sin(tt), np.zeros_like(tt)
        ax.plot(hx, hy, hz, color=guide_color, alpha=guide_alpha, linewidth=guide_linewidth)

        def _ray_mpl(az_deg, el_deg, label=None, is_zenith=False):
            """Draw one Matplotlib 3D guide ray plus optional label/arrow."""
            x1, y1, z1 = _cart_from_azel(az_deg, el_deg, r=guide_length)
            ax.plot([0, x1], [0, y1], [0, z1],
                    color=guide_color, alpha=guide_alpha, linewidth=guide_linewidth)

            nx, ny, nz = np.array([x1, y1, z1], float)
            L = float(arrow_scale) * float(axis_arrow_size)
            ray_len = max(np.linalg.norm([nx, ny, nz]), 1e-12)
            nx, ny, nz = nx/ray_len, ny/ray_len, nz/ray_len

            if show_axis_arrows:
                if arrow_direction.lower() == "outward":
                    ax.quiver(0, 0, 0, x1, y1, z1,
                              length=1.0, normalize=True,
                              arrow_length_ratio=L, color=guide_color,
                              linewidth=guide_linewidth)
                else:
                    ax.quiver(x1, y1, z1, -x1, -y1, -z1,
                              length=L*1.2, normalize=True,
                              arrow_length_ratio=0.5*L,
                              color=guide_color, linewidth=guide_linewidth)
                label_shift_local = L + float(label_offset_extra)
            else:
                label_shift_local = float(label_offset_extra)

            if label:
                extra_z = 0.25 * label_shift_local if is_zenith else 0.0
                lx = x1 + nx * label_shift_local
                ly = y1 + ny * label_shift_local
                lz = z1 + nz * label_shift_local + extra_z
                ax.text(lx, ly, lz, label, color=guide_color,
                        ha="center", va="bottom", fontsize=11, weight="bold")

        _ray_mpl(0,   0,   "Az 0°")
        _ray_mpl(90,  0,   "Az 90°")
        _ray_mpl(180, 0,   "Az 180°")
        _ray_mpl(270, 0,   "Az 270°")
        _ray_mpl(0,  90,   "Zenith", is_zenith=True)

    # Camera and title
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title or default_title, pad=18)

    # Colorbar
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.03)
        cbar.set_label(colorbar_label)
    fig.subplots_adjust(
        left=0.02,
        right=0.86 if colorbar else 0.98,
        bottom=0.03,
        top=0.90,
    )

    # Matplotlib hover: coloured background, offset away from cursor
    if show_hover and len(polys) > 0:
        annot = ax.annotate(
            "", xy=(0, 0), xytext=hover_xytext,
            textcoords="offset points", ha="left", va="bottom",
            fontsize=9, color="white",  # will adjust for contrast below
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333", alpha=0.95)
        )
        annot.set_visible(False)

        def _mpl_font_color_from_rgba(rgba):
            """Pick black/white text for contrast against RGBA background."""
            r, g, b, a = rgba
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return "black" if y > 0.60 else "white"

        def _on_move(event):
            """Update hover annotation based on current mouse hit-test."""
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return
            hit, info = coll.contains(event)
            if not hit or "ind" not in info or len(info["ind"]) == 0:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return
            idx = info["ind"][0]
            if 0 <= idx < len(poly_labels):
                annot.set_text(poly_labels[idx])
                bg_rgba = poly_bg_colors[idx]
                annot.get_bbox_patch().set_facecolor(bg_rgba)
                annot.get_bbox_patch().set_edgecolor(bg_rgba)
                annot.set_color(_mpl_font_color_from_rgba(bg_rgba))
                if event.xdata is not None and event.ydata is not None:
                    annot.xy = (event.xdata, event.ydata)
                annot.set_visible(True)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _on_move)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return (fig, cell_vals_q) if return_values else fig



# -----------------------------------------------------------------------------
# Satellite Distribution API
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
_AZ_EL_KEY_DTYPE = np.dtype([("az_bits", np.uint32), ("el_bits", np.uint32)])


@dataclass(slots=True)
class SatelliteDistributionConfig:
    """
    Runtime configuration for the satellite-per-skycell processing pipeline.

    Notes
    -----
    The configuration groups controls for four areas:

    - sky-cell mapping and deduplication
    - random/average sample selection
    - plot and animation export
    - runtime progress and ffmpeg behavior

    Average sky-cell maps are arithmetic means of per-slot count arrays. They
    are not logarithmic power averages.

    Animation export has two families of backends:

    - ``"plotly"`` keeps the legacy Plotly/Kaleido per-frame PNG workflow.
    - ``"cpu_raster"`` and ``"gpu_raster"`` use a reusable off-screen
      Matplotlib Agg canvas and stream raw frames directly to encoders.

    The fast raster path is currently only supported for the default full-grid
    S.1586 polar animation workflow used by
    :func:`satellite_distribution_over_sky`.
    """

    skycell_mode: str = "s1586"
    s1586_n_cells: int = int(_S1586_N_CELLS) if _S1586_N_CELLS is not None else 2334
    deduplicate_by_az_el: bool = True
    deduplicate_round_decimals: int | None = 5
    coord_edge_eps_deg: float = 1e-3

    random_timestep_plots: int = 100
    random_seed: int = 13
    random_sample_nonempty_only: bool = True

    enable_random_plot_samples: bool = True
    enable_average_plot: bool = True
    enable_animation_random_set: bool = True
    enable_animation_subsequent_set: bool = True

    average_slot_stride: int = 1
    average_max_slots: int = 0
    average_sample_nonempty_only: bool = False

    random_plot_require_full_pass: bool = True
    random_animation_require_full_pass: bool = True
    max_raw_slots_to_read: int = 0

    read_slot_chunk: int = 256
    progress_every_slots: int = 250
    sky_mapper_backend: str = "auto"
    numba_mapper_min_work: int = 250_000

    save_outputs: bool = True
    output_dir: str = "SatSamples_1.13_sys3_B525"
    output_prefix: str = "SatSamples_parser"
    save_plots_html: bool = True
    save_plots_png: bool = True
    save_plots_jpg: bool = True
    save_results_json: bool = True
    save_results_npz: bool = True
    show_plots: bool = False

    export_image_width: int = 1600
    export_image_height: int = 1200
    export_image_scale: float = 1.0

    save_animation: bool = True
    animation_fps: int = 30
    animation_extra_fps: int = 1
    save_animation_mp4: bool = True
    save_animation_gif: bool = True
    animation_output_subdir: str = "animations"
    animation_random_sample_count: int = 300
    animation_subsequent_sample_count: int = 300
    animation_random_nonempty_only: bool = True
    animation_subsequent_nonempty_only: bool = False

    animation_frame_width: int = 1280
    animation_frame_height: int = 960
    animation_frame_scale: float = 1.0
    # Animation frame renderer:
    # - "auto": prefer GPU raster when CuPy+CUDA is usable, otherwise CPU raster
    # - "plotly": legacy Plotly/Kaleido per-frame PNG workflow
    # - "cpu_raster": reusable Agg canvas + direct encoder streaming on CPU
    # - "gpu_raster": CuPy-assisted rasterization with CPU fallback if unavailable
    animation_render_backend: str = "auto"
    # Keep per-frame PNG files for fast raster backends. Legacy Plotly export
    # still needs frame PNGs regardless of this flag.
    keep_animation_frame_pngs: bool = False

    ffmpeg_preset: str = "veryfast"
    ffmpeg_crf: int = 24
    ffmpeg_threads: int = 0
    ffmpeg_loglevel: str = "error"
    ffmpeg_show_progress: bool = True
    ffmpeg_progress_step_percent: float = 2.0

    animation_frame_progress: bool = True

    skycell_vis_engine: str = "plotly"
    skycell_vis_2d_projection: str = "polar"
    skycell_vis_cmap: str = "turbo"


_SATELLITE_DISTRIBUTION_CONFIG_FIELD_NAMES = frozenset(
    field.name for field in fields(SatelliteDistributionConfig)
)


def _resolve_config(
    base_config: SatelliteDistributionConfig | None,
    overrides: dict[str, Any],
) -> SatelliteDistributionConfig:
    """
    Build an effective plotting config from a base config plus keyword overrides.

    Parameters
    ----------
    base_config : SatelliteDistributionConfig or None
        Optional pre-existing configuration. When provided, it is copied before
        overrides are applied.
    overrides : dict[str, Any]
        Keyword-style field overrides for :class:`SatelliteDistributionConfig`.

    Returns
    -------
    SatelliteDistributionConfig
        Effective configuration after validation and field replacement.

    Raises
    ------
    TypeError
        Raised when an override key is not a valid
        :class:`SatelliteDistributionConfig` field.

    Notes
    -----
    Removed compatibility knobs are intentionally no longer accepted here.
    Callers must use the current canonical config fields so front ends and
    notebooks fail fast when they drift from the supported API.
    """
    cfg = replace(base_config) if base_config is not None else SatelliteDistributionConfig()
    if not overrides:
        return cfg

    valid_fields = _SATELLITE_DISTRIBUTION_CONFIG_FIELD_NAMES
    unknown = [key for key in overrides if key not in valid_fields]
    if unknown:
        valid = ", ".join(sorted(valid_fields))
        unknown_txt = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown config override(s): {unknown_txt}. Valid keys: {valid}")

    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@dataclass(slots=True)
class RandomSlotSample:
    """Container for one selected slot snapshot used for plots/animations."""

    iter_name: str
    slot_local_idx: int
    slot_global_idx: int
    time_mjd: float | None
    counts: np.ndarray
    n_satellites: int
    n_active_skycells: int


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _iter_names(h5: h5py.File) -> list[str]:
    """Return sorted iteration group names under `/iter`."""
    names = [name for name in h5.get("iter", {}).keys() if name.startswith("iter_")]
    names.sort()
    return names


def _build_sky_mapper(
    mode: str,
    *,
    n_cells: int,
) -> tuple[int, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Return `(n_skycells, mapper)` for configured skycell layout."""
    if str(mode).lower().strip() != "s1586":
        raise ValueError(f"Unsupported skycell_mode={mode!r}.")
    if _skycell_id_s1586 is None:
        raise RuntimeError("s1586 mapping requires scepter.angle_sampler._skycell_id_s1586.")

    def mapper(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        """Map azimuth/elevation arrays to S.1586 cell IDs."""
        return np.asarray(_skycell_id_s1586(az_deg, el_deg), dtype=np.int32)

    return int(n_cells), mapper


def _get_s1586_numba_mapper() -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
    """Return direct Numba S.1586 mapper when available in `angle_sampler`."""
    if _angle_sampler_mod is None:
        return None
    mapper = getattr(_angle_sampler_mod, "_skycell_id_s1586_numba", None)
    return mapper if callable(mapper) else None


def _choose_sky_mapper_backend(
    *,
    requested_backend: str,
    skycell_mode: str,
    expected_slots: int,
    points_per_slot_hint: int,
    numba_mapper_min_work: int,
) -> tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray] | None]:
    """Choose mapper backend for repeated slot mapping.

    Returns `(backend_name, numba_mapper_or_none)`, where `backend_name` is one of:
    - `"dispatcher"`: use `_skycell_id_s1586` runtime dispatcher.
    - `"numba"`: use direct `_skycell_id_s1586_numba` mapper.
    """
    backend = str(requested_backend).lower().strip()
    if backend not in {"auto", "dispatcher", "numba"}:
        raise ValueError("sky_mapper_backend must be one of: 'auto', 'dispatcher', 'numba'.")

    if str(skycell_mode).lower().strip() != "s1586":
        return "dispatcher", None

    numba_mapper = _get_s1586_numba_mapper()
    if numba_mapper is None:
        if backend == "numba":
            warnings.warn(
                "sky_mapper_backend='numba' requested but Numba S.1586 mapper is unavailable; "
                "falling back to dispatcher."
            )
        return "dispatcher", None

    if backend == "numba":
        return "numba", numba_mapper
    if backend == "dispatcher":
        return "dispatcher", None

    # Auto mode: pick direct numba mapper when there is enough expected mapping work
    # to amortize compilation/dispatch overhead.
    work = max(0, int(expected_slots)) * max(1, int(points_per_slot_hint))
    if work >= max(1, int(numba_mapper_min_work)):
        return "numba", numba_mapper
    return "dispatcher", None


def _make_output_run_dir(base_dir: str, prefix: str) -> Path:
    """Create a timestamped unique output directory and return its path."""
    out_root = Path(base_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{prefix}_{stamp}"
    idx = 1
    while run_dir.exists():
        run_dir = out_root / f"{prefix}_{stamp}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _format_mjd(mjd: float | None) -> str:
    """Format MJD for titles/logs; return 'n/a' for missing/non-finite values."""
    if mjd is None or not np.isfinite(mjd):
        return "n/a"
    return f"{float(mjd):.6f}"


def _normalize_az_el_for_skycell_mapping(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    *,
    edge_eps_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize az/el for robust skycell mapping near boundaries.

    - Azimuth: wrap to [0, 360), snap values very close to 360 back to 0.
    - Elevation: clip to [0, 90], with a small tolerance around edges.
    """
    az = np.remainder(np.asarray(az_deg, dtype=np.float64), 360.0)
    el = np.asarray(el_deg, dtype=np.float64)
    eps = max(0.0, float(edge_eps_deg))

    if eps > 0.0:
        az = np.where(az >= 360.0 - eps, 0.0, az)
        el = np.where(el < eps, 0.0, el)
        el = np.where(el > 90.0 - eps, 90.0, el)

    el = np.clip(el, 0.0, 90.0)
    return az.astype(np.float32, copy=False), el.astype(np.float32, copy=False)


def _count_satellites_per_skycell_slot(
    az_slot: np.ndarray,
    el_slot: np.ndarray,
    *,
    sky_mapper: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_skycells: int,
    deduplicate: bool,
    edge_eps_deg: float,
    deduplicate_round_decimals: int | None,
    dedup_quant_scale: float | None = None,
    dedup_az_wrap: int | None = None,
    counts_buffer: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int, int]:
    """Count mappable satellites per skycell for one slot.

    Returns:
        counts: Per-cell counts (`int32`, possibly reusing `counts_buffer`).
        n_valid_points: Finite az/el points inside broad elevation tolerance.
        n_unique_satellites: Unique points after optional az/el deduplication.
        n_mappable_satellites: Points successfully mapped to valid skycell IDs.
    """
    az = np.asarray(az_slot, dtype=np.float32).reshape(-1)
    el = np.asarray(el_slot, dtype=np.float32).reshape(-1)
    eps = max(0.0, float(edge_eps_deg))
    valid = np.isfinite(az) & np.isfinite(el) & (el >= -eps) & (el <= 90.0 + eps)
    n_valid_points = int(np.count_nonzero(valid))
    if counts_buffer is None:
        counts = np.zeros(n_skycells, dtype=np.int32)
    else:
        counts = counts_buffer
        counts.fill(0)
    if n_valid_points == 0:
        return counts, 0, 0, 0

    az_valid, el_valid = _normalize_az_el_for_skycell_mapping(
        az[valid],
        el[valid],
        edge_eps_deg=eps,
    )

    if deduplicate:
        if deduplicate_round_decimals is None:
            keys = np.empty(az_valid.size, dtype=_AZ_EL_KEY_DTYPE)
            keys["az_bits"] = az_valid.view(np.uint32)
            keys["el_bits"] = el_valid.view(np.uint32)
        else:
            if dedup_quant_scale is not None:
                scale = float(dedup_quant_scale)
            else:
                decimals = max(0, int(deduplicate_round_decimals))
                scale = float(10 ** decimals)
            az_quant = np.rint(np.asarray(az_valid, dtype=np.float64) * scale).astype(np.int64)
            az_wrap = int(dedup_az_wrap) if dedup_az_wrap is not None else int(round(360.0 * scale))
            if az_wrap > 0:
                az_quant %= az_wrap
            el_quant = np.rint(np.asarray(el_valid, dtype=np.float64) * scale).astype(np.int64)
            keys = np.empty(az_valid.size, dtype=[("az_q", np.int64), ("el_q", np.int64)])
            keys["az_q"] = az_quant
            keys["el_q"] = el_quant
        _, first_idx = np.unique(keys, return_index=True)
        first_idx.sort()
        az_used = az_valid[first_idx]
        el_used = el_valid[first_idx]
    else:
        az_used = az_valid
        el_used = el_valid

    n_unique_satellites = int(az_used.size)
    if n_unique_satellites == 0:
        return counts, n_valid_points, 0, 0

    sky_ids = np.asarray(sky_mapper(az_used, el_used), dtype=np.int32).reshape(-1)
    bad = (sky_ids < 0) | (sky_ids >= n_skycells)
    if np.any(bad):
        # Retry with fully clipped coordinates for borderline numeric cases.
        az_retry, el_retry = _normalize_az_el_for_skycell_mapping(
            az_used[bad],
            el_used[bad],
            edge_eps_deg=eps,
        )
        sky_ids_retry = np.asarray(sky_mapper(az_retry, el_retry), dtype=np.int32).reshape(-1)
        sky_ids[bad] = sky_ids_retry

    sky_valid = sky_ids[(sky_ids >= 0) & (sky_ids < n_skycells)]
    n_mappable_satellites = int(sky_valid.size)
    if n_mappable_satellites > 0:
        counts += np.bincount(
            sky_valid.astype(np.int64, copy=False),
            minlength=n_skycells,
        ).astype(np.int32, copy=False)

    return counts, n_valid_points, n_unique_satellites, n_mappable_satellites


@dataclass(slots=True)
class _ReusableSlotCountPlotter:
    """Reusable full-grid Plotly renderer for per-skycell count maps."""

    vmax: float
    vis_cmap: str
    vis_projection: str
    vis_engine: str
    show_plots: bool
    raster_res: int = 800
    _fig: Any | None = None
    _disc_mask: np.ndarray | None = None
    _cell_idx: np.ndarray | None = None
    _z_buffer: np.ndarray | None = None

    def can_reuse(self, counts: np.ndarray) -> bool:
        """Return True when the fast reusable Plotly path can be used."""
        return (
            plot_hemisphere_2D is not None
            and str(self.vis_engine).lower() == "plotly"
            and str(self.vis_projection).lower() == "polar"
            and int(np.asarray(counts).size) == 2334
        )

    def _ensure_initialized(self) -> None:
        """Build the reusable base figure and cached raster lookup on first use."""
        if self._fig is not None:
            return
        if plot_hemisphere_2D is None:
            raise RuntimeError("plot_hemisphere_2D is unavailable.")

        self._fig = plot_hemisphere_2D(
            data=np.zeros((1, 2334), dtype=np.float64),
            mode="power",
            worst_percent=50.0,
            cell_axis=-1,
            cmap=self.vis_cmap,
            vmin=0.0,
            vmax=float(self.vmax),
            projection=self.vis_projection,
            engine=self.vis_engine,
            show=False,
            save_html=None,
        )
        _, _, disc_mask, cell_idx = _s1586_polar_heatmap_lookup(
            self.raster_res,
            "equal_area",
            False,
        )
        self._disc_mask = disc_mask
        self._cell_idx = cell_idx
        self._z_buffer = np.full(cell_idx.shape, np.nan, dtype=np.float64)

    def render(
        self,
        counts: np.ndarray,
        *,
        title: str,
        save_html_path: Path | None,
    ) -> Any:
        """Update and return the reusable Plotly figure for one count array."""
        counts_arr = np.asarray(counts, dtype=np.float64).reshape(-1)
        if not self.can_reuse(counts_arr):
            return _plot_slot_counts_via_api(
                counts_arr,
                title=title,
                vmax=self.vmax,
                save_html_path=save_html_path,
                vis_cmap=self.vis_cmap,
                vis_projection=self.vis_projection,
                vis_engine=self.vis_engine,
                show_plots=self.show_plots,
            )

        self._ensure_initialized()
        assert self._fig is not None
        assert self._disc_mask is not None
        assert self._cell_idx is not None
        assert self._z_buffer is not None

        self._z_buffer.fill(np.nan)
        self._z_buffer[self._disc_mask] = counts_arr[self._cell_idx[self._disc_mask]]
        self._fig.data[0].z = self._z_buffer
        self._fig.update_layout(title=title)

        if save_html_path is not None:
            self._fig.write_html(str(save_html_path), include_plotlyjs=True, full_html=True)
        if self.show_plots:
            self._fig.show()
        return self._fig


def _plot_slot_counts_via_api(
    counts: np.ndarray,
    *,
    title: str,
    vmax: float,
    save_html_path: Path | None,
    vis_cmap: str,
    vis_projection: str,
    vis_engine: str,
    show_plots: bool,
) -> Any:
    """Render one hemisphere map of per-skycell counts using `plot_hemisphere_2D`."""
    if plot_hemisphere_2D is None:
        raise RuntimeError("plot_hemisphere_2D is unavailable (scepter.visualise import failed).")

    return plot_hemisphere_2D(
        data=np.asarray(counts, dtype=np.float64).reshape(1, -1),
        mode="power",
        worst_percent=50.0,
        cell_axis=-1,
        cmap=vis_cmap,
        vmin=0.0,
        vmax=float(vmax),
        title=title,
        projection=vis_projection,
        engine=vis_engine,
        show=show_plots,
        save_html=None if save_html_path is None else str(save_html_path),
    )


def _plot_slot_counts(
    counts: np.ndarray,
    *,
    title: str,
    vmax: float,
    save_html_path: Path | None,
    vis_cmap: str,
    vis_projection: str,
    vis_engine: str,
    show_plots: bool,
    plotter: _ReusableSlotCountPlotter | None = None,
) -> Any:
    """Render one hemisphere map of per-skycell counts, reusing Plotly state when possible."""
    if plotter is not None:
        return plotter.render(
            np.asarray(counts, dtype=np.float64),
            title=title,
            save_html_path=save_html_path,
        )
    return _plot_slot_counts_via_api(
        np.asarray(counts, dtype=np.float64),
        title=title,
        vmax=vmax,
        save_html_path=save_html_path,
        vis_cmap=vis_cmap,
        vis_projection=vis_projection,
        vis_engine=vis_engine,
        show_plots=show_plots,
    )


def _save_plotly_static_images(
    fig: Any,
    *,
    png_path: Path | None,
    jpg_path: Path | None,
    export_width: int,
    export_height: int,
    export_scale: float,
) -> tuple[bool, bool]:
    """Export Plotly figure to PNG/JPG when backend supports `write_image`."""
    has_writer = hasattr(fig, "write_image")
    if not has_writer:
        if png_path is not None or jpg_path is not None:
            print("[plot] Static PNG/JPG export skipped: figure backend has no write_image().")
        return False, False

    png_saved = False
    jpg_saved = False

    if png_path is not None:
        try:
            fig.write_image(
                str(png_path),
                format="png",
                width=int(export_width),
                height=int(export_height),
                scale=float(export_scale),
            )
            png_saved = True
        except Exception as exc:
            print(f"[plot] PNG export failed for '{png_path.name}': {exc}")

    if jpg_path is not None:
        try:
            fig.write_image(
                str(jpg_path),
                format="jpg",
                width=int(export_width),
                height=int(export_height),
                scale=float(export_scale),
            )
            jpg_saved = True
        except Exception as exc:
            print(f"[plot] JPG export failed for '{jpg_path.name}': {exc}")

    return png_saved, jpg_saved


def _save_animation_mp4_ffmpeg(
    *,
    frame_pattern: str,
    output_path: Path,
    fps: int,
    total_frames: int,
    ffmpeg_preset: str,
    ffmpeg_crf: int,
    ffmpeg_threads: int,
    ffmpeg_loglevel: str,
    ffmpeg_show_progress: bool,
    ffmpeg_progress_step_percent: float,
) -> bool:
    """Encode PNG frame sequence into MP4 via ffmpeg with optional tqdm progress."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("[anim] MP4 export skipped: ffmpeg is not available on PATH.")
        return False

    fps_i = max(1, int(fps))
    total_frames_i = max(1, int(total_frames))
    total_duration_us = (float(total_frames_i) / float(fps_i)) * 1_000_000.0

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        str(ffmpeg_loglevel),
        "-nostats",
        "-progress",
        "pipe:1",
        "-framerate",
        str(fps_i),
        "-start_number",
        "1",
        "-i",
        frame_pattern,
        "-threads",
        str(int(ffmpeg_threads)),
        "-c:v",
        "libx264",
        "-preset",
        str(ffmpeg_preset),
        "-crf",
        str(int(ffmpeg_crf)),
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    def _progress_percent(fields: dict[str, str]) -> tuple[float | None, str | None]:
        """Extract progress percent from ffmpeg `-progress` key/value fields."""
        frame_raw = fields.get("frame")
        if frame_raw is not None:
            try:
                frame_now = max(0, int(frame_raw))
                pct = min(100.0, (100.0 * float(frame_now)) / float(total_frames_i))
                return pct, frame_raw
            except Exception:
                pass

        out_us_raw = fields.get("out_time_us")
        if out_us_raw is None:
            out_us_raw = fields.get("out_time_ms")
        if out_us_raw is not None:
            try:
                out_us = max(0.0, float(out_us_raw))
                pct = min(100.0, (100.0 * out_us) / max(total_duration_us, 1.0))
                return pct, None
            except Exception:
                pass
        return None, frame_raw

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    progress_fields: dict[str, str] = {}
    last_pct_reported = -1.0
    last_frame_reported = 0
    encode_bar: Any = None
    if ffmpeg_show_progress:
        tqdm = _load_tqdm()
        if tqdm is not None:
            encode_bar = tqdm(
                total=total_frames_i,
                desc=f"[anim][encode] {output_path.name}",
                unit="frame",
                leave=False,
                dynamic_ncols=True,
            )

    if proc.stdout is not None:
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if (not line) or ("=" not in line):
                continue
            key, value = line.split("=", 1)
            progress_fields[key] = value

            if key != "progress":
                continue

            pct, frame_txt = _progress_percent(progress_fields)
            if pct is None:
                continue

            if value == "end":
                if encode_bar is not None:
                    if last_frame_reported < total_frames_i:
                        encode_bar.update(total_frames_i - last_frame_reported)
                        last_frame_reported = total_frames_i
                    speed_txt = progress_fields.get("speed")
                    if speed_txt is not None:
                        encode_bar.set_postfix_str(f"speed={speed_txt}")
                continue

            if encode_bar is None:
                continue

            frame_now: int | None = None
            if frame_txt is not None:
                try:
                    frame_now = max(0, min(total_frames_i, int(frame_txt)))
                except Exception:
                    frame_now = None

            if frame_now is None:
                if (
                    (last_pct_reported < 0.0)
                    or (pct >= last_pct_reported + float(ffmpeg_progress_step_percent))
                ):
                    frame_now = max(0, min(total_frames_i, int(round((pct / 100.0) * total_frames_i))))
                    last_pct_reported = pct

            if frame_now is not None and frame_now > last_frame_reported:
                encode_bar.update(frame_now - last_frame_reported)
                last_frame_reported = frame_now

            speed_txt = progress_fields.get("speed")
            if speed_txt is not None:
                encode_bar.set_postfix_str(f"speed={speed_txt}")

    stderr_text = ""
    if proc.stderr is not None:
        stderr_text = proc.stderr.read().strip()
    return_code = proc.wait()

    if encode_bar is not None:
        if return_code == 0 and last_frame_reported < total_frames_i:
            encode_bar.update(total_frames_i - last_frame_reported)
        encode_bar.close()

    if return_code != 0:
        err = stderr_text if stderr_text else "unknown ffmpeg error"
        print(f"[anim] MP4 export failed: {err}")
        return False
    return True


def _save_animation_gif(
    *,
    frame_paths: list[Path],
    output_path: Path,
    fps: int,
    frame_pattern: str | None = None,
    ffmpeg_loglevel: str = "error",
) -> bool:
    """Encode GIF from PNG frames (ffmpeg fast-path, Pillow fallback)."""
    if not frame_paths:
        print("[anim] GIF export skipped: no PNG frames available.")
        return False

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None and frame_pattern is not None:
        fps_i = max(1, int(fps))
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            str(ffmpeg_loglevel),
            "-framerate",
            str(fps_i),
            "-start_number",
            "1",
            "-i",
            frame_pattern,
            "-vf",
            "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse",
            "-loop",
            "0",
            str(output_path),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            return True
        err = (proc.stderr or "").strip()
        print(f"[anim] GIF export via ffmpeg failed, falling back to Pillow: {err or 'unknown error'}")

    try:
        from PIL import Image
    except Exception as exc:
        print(f"[anim] GIF export skipped: Pillow unavailable ({exc}).")
        return False

    duration_ms = max(1, int(round(1000.0 / float(max(1, int(fps))))))
    frames: list[Any] = []
    try:
        for path in frame_paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB").copy())
        if not frames:
            print("[anim] GIF export skipped: no readable PNG frames.")
            return False
        first, rest = frames[0], frames[1:]
        first.save(
            output_path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    except Exception as exc:
        print(f"[anim] GIF export failed: {exc}")
        return False
    return True


@lru_cache(maxsize=1)
def _get_cupy_module() -> Any | None:
    """Return the imported CuPy module, or ``None`` when CuPy is unavailable."""
    try:
        import cupy as cp
    except Exception:
        return None
    return cp


@lru_cache(maxsize=1)
def _probe_visualise_cuda_support() -> tuple[bool, str]:
    """Return whether CuPy-backed CUDA rasterization is usable."""
    cp = _get_cupy_module()
    if cp is None:
        return False, "CuPy is unavailable."
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return False, f"CuPy/CUDA probe failed: {exc}"
    if device_count <= 0:
        return False, "CuPy is available, but no CUDA devices were reported."
    return True, f"CuPy detected {device_count} CUDA device(s)."


def _animation_fast_path_supported(
    *,
    skycell_mode: str,
    n_skycells: int,
    vis_projection: str,
) -> bool:
    """Return True when the fast animation raster path is supported."""
    return (
        str(skycell_mode).lower().strip() == "s1586"
        and int(n_skycells) == 2334
        and str(vis_projection).lower().strip() == "polar"
    )


def _resolve_animation_render_backend(
    *,
    requested_backend: str,
    skycell_mode: str,
    n_skycells: int,
    vis_projection: str,
) -> tuple[str, str, bool]:
    """
    Resolve the effective animation render backend.

    Returns
    -------
    effective_backend : {"plotly", "cpu_raster", "gpu_raster"}
        Chosen backend after capability and geometry checks.
    reason : str
        Short machine-readable reason used in run diagnostics.
    direct_streaming : bool
        True when animation frames can be streamed directly without relying on
        per-frame Plotly/Kaleido PNG export.
    """
    backend = str(requested_backend).lower().strip()
    if backend not in {"auto", "plotly", "cpu_raster", "gpu_raster"}:
        raise ValueError(
            "animation_render_backend must be one of: "
            "'auto', 'plotly', 'cpu_raster', 'gpu_raster'."
        )

    if backend == "plotly":
        return "plotly", "requested_plotly", False

    if not _animation_fast_path_supported(
        skycell_mode=skycell_mode,
        n_skycells=n_skycells,
        vis_projection=vis_projection,
    ):
        warnings.warn(
            "Fast animation rendering is currently only supported for the "
            "full-grid S.1586 polar workflow. Falling back to legacy "
            "Plotly/Kaleido animation export.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "plotly", "fast_path_unsupported_geometry", False

    if backend == "cpu_raster":
        return "cpu_raster", "requested_cpu_raster", True

    cuda_ok, cuda_reason = _probe_visualise_cuda_support()
    if backend == "gpu_raster":
        if cuda_ok:
            return "gpu_raster", "requested_gpu_raster", True
        warnings.warn(
            "animation_render_backend='gpu_raster' requested but CuPy/CUDA is "
            f"unavailable ({cuda_reason}). Falling back to 'cpu_raster'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu_raster", "gpu_raster_unavailable_fallback_cpu", True

    if cuda_ok:
        return "gpu_raster", "auto_gpu_raster", True
    return "cpu_raster", "auto_cpu_raster", True


def _build_animation_frame_title(
    *,
    sample: RandomSlotSample,
    frame_idx_zero: int,
    total_frames: int,
    title_prefix: str,
) -> str:
    """Return the standard per-frame animation title string."""
    idx = int(frame_idx_zero) + 1
    return (
        f"{title_prefix} {idx}/{total_frames}"
        f"<br><sup>iter={sample.iter_name}, local_slot={sample.slot_local_idx}, "
        f"global_slot={sample.slot_global_idx}, MJD={_format_mjd(sample.time_mjd)}, "
        f"satellites={sample.n_satellites}</sup>"
    )


def _rasterize_s1586_counts_cpu(
    counts: np.ndarray,
    *,
    disc_mask: np.ndarray,
    cell_idx: np.ndarray,
) -> np.ndarray:
    """Map full-grid S.1586 counts to a dense polar raster on CPU."""
    counts_arr = np.asarray(counts, dtype=np.float32).reshape(-1)
    if counts_arr.size != 2334:
        raise ValueError(
            f"Expected 2334 S.1586 skycell counts for fast animation, got {counts_arr.size}."
        )
    raster = np.take(counts_arr, cell_idx, mode="clip").astype(np.float32, copy=False)
    raster = np.array(raster, dtype=np.float32, copy=True)
    raster[~disc_mask] = 0.0
    return raster


@dataclass(slots=True)
class _S1586GpuRasterState:
    """Reusable CuPy buffers for per-frame S.1586 animation rasterization."""

    cp: Any
    disc_mask_dev: Any
    cell_idx_dev: Any
    counts_dev: Any
    raster_dev: Any

    @classmethod
    def from_lookup(
        cls,
        *,
        disc_mask: np.ndarray,
        cell_idx: np.ndarray,
    ) -> "_S1586GpuRasterState":
        """Build persistent device buffers from cached host lookup arrays."""
        cp = _get_cupy_module()
        if cp is None:
            raise RuntimeError("CuPy is unavailable.")
        return cls(
            cp=cp,
            disc_mask_dev=cp.asarray(disc_mask, dtype=cp.bool_),
            cell_idx_dev=cp.asarray(cell_idx, dtype=cp.int32),
            counts_dev=cp.empty(2334, dtype=cp.float32),
            raster_dev=cp.empty(cell_idx.shape, dtype=cp.float32),
        )

    def rasterize_to_host(self, counts: np.ndarray) -> np.ndarray:
        """Rasterize one count array on GPU and return the host copy."""
        counts_arr = np.asarray(counts, dtype=np.float32).reshape(-1)
        if counts_arr.size != 2334:
            raise ValueError(
                f"Expected 2334 S.1586 skycell counts for fast animation, got {counts_arr.size}."
            )
        self.counts_dev[...] = self.cp.asarray(counts_arr, dtype=self.cp.float32)
        self.raster_dev[...] = self.counts_dev[self.cell_idx_dev]
        self.raster_dev[~self.disc_mask_dev] = self.cp.float32(0.0)
        return self.cp.asnumpy(self.raster_dev)


def _rasterize_s1586_counts_gpu(
    counts: np.ndarray,
    *,
    gpu_state: _S1586GpuRasterState,
) -> np.ndarray:
    """Map full-grid S.1586 counts to a dense polar raster on GPU."""
    return gpu_state.rasterize_to_host(counts)


def _draw_s1586_polar_static_overlays(ax: Any) -> None:
    """Draw static S.1586 polar guides and borders on a Matplotlib axis."""
    _, _, _, _, el_edges, cells_per_ring = _s1586_cells()
    tt = np.linspace(0.0, 2.0 * np.pi, 361)

    for elb in el_edges:
        rr = _r_from_el(elb, "equal_area", invert=False)
        ax.plot(
            rr * np.cos(tt),
            rr * np.sin(tt),
            color="#1f2937",
            linewidth=0.6,
            alpha=0.35,
            zorder=2,
        )

    for i_ring, n_in_ring in enumerate(cells_per_ring):
        el0 = el_edges[i_ring]
        el1 = el_edges[i_ring + 1]
        r0 = _r_from_el(el0, "equal_area", invert=False)
        r1 = _r_from_el(el1, "equal_area", invert=False)
        step = 360 // int(n_in_ring)
        for az in np.arange(0, 360, step):
            t = np.radians((90.0 - float(az)) % 360.0)
            ax.plot(
                [r0 * np.cos(t), r1 * np.cos(t)],
                [r0 * np.sin(t), r1 * np.sin(t)],
                color="#1f2937",
                linewidth=0.45,
                alpha=0.25,
                zorder=2,
            )

    cardinal_labels = {
        "N": (0.0, 1.08),
        "E": (1.08, 0.0),
        "S": (0.0, -1.08),
        "W": (-1.08, 0.0),
    }
    for label, (x_pos, y_pos) in cardinal_labels.items():
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="#111827",
            zorder=3,
        )


@dataclass(slots=True)
class _FastS1586AnimationRenderer:
    """Reusable off-screen Agg renderer for full-grid S.1586 polar animations."""

    frame_width: int
    frame_height: int
    vmax: float
    vis_cmap: str
    backend: str = "cpu_raster"
    raster_res: int = 800
    _canvas: FigureCanvasAgg | None = None
    _figure: Figure | None = None
    _image_artist: Any | None = None
    _title_artist: Any | None = None
    _disc_mask: np.ndarray | None = None
    _cell_idx: np.ndarray | None = None
    _gpu_state: _S1586GpuRasterState | None = None

    def _ensure_initialized(self) -> None:
        """Create the static Agg figure, overlays, and reusable image artist."""
        if self._canvas is not None:
            return

        _, _, disc_mask, cell_idx = _s1586_polar_heatmap_lookup(
            max(256, int(self.raster_res)),
            "equal_area",
            False,
        )
        self._disc_mask = disc_mask
        self._cell_idx = cell_idx.astype(np.int32, copy=False)
        if self.backend == "gpu_raster":
            self._gpu_state = _S1586GpuRasterState.from_lookup(
                disc_mask=self._disc_mask,
                cell_idx=self._cell_idx,
            )

        dpi = 100.0
        fig = Figure(
            figsize=(
                max(1, int(self.frame_width)) / dpi,
                max(1, int(self.frame_height)) / dpi,
            ),
            dpi=dpi,
            facecolor="white",
        )
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.035, 0.05, 0.78, 0.9])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_axis_off()

        image_artist = ax.imshow(
            np.zeros(self._cell_idx.shape, dtype=np.float32),
            extent=(-1.0, 1.0, -1.0, 1.0),
            origin="lower",
            cmap=self.vis_cmap,
            vmin=0.0,
            vmax=float(self.vmax),
            interpolation="nearest",
            alpha=self._disc_mask.astype(np.float32, copy=False),
            zorder=1,
        )
        _draw_s1586_polar_static_overlays(ax)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Satellites per skycell")
        title_artist = fig.suptitle("", y=0.985, fontsize=11)
        canvas.draw()

        self._canvas = canvas
        self._figure = fig
        self._image_artist = image_artist
        self._title_artist = title_artist

    def render_frame(
        self,
        counts: np.ndarray,
        *,
        title: str,
    ) -> np.ndarray:
        """Render one animation frame and return an RGBA uint8 image buffer."""
        self._ensure_initialized()
        assert self._canvas is not None
        assert self._image_artist is not None
        assert self._title_artist is not None
        assert self._disc_mask is not None
        assert self._cell_idx is not None

        if self.backend == "gpu_raster":
            assert self._gpu_state is not None
            raster = _rasterize_s1586_counts_gpu(
                counts,
                gpu_state=self._gpu_state,
            )
        else:
            raster = _rasterize_s1586_counts_cpu(
                counts,
                disc_mask=self._disc_mask,
                cell_idx=self._cell_idx,
            )

        self._image_artist.set_data(raster)
        self._title_artist.set_text(title)
        self._canvas.draw()
        return np.asarray(self._canvas.buffer_rgba(), dtype=np.uint8).copy()


def _save_animation_frame_png(
    frame_rgba: np.ndarray,
    output_path: Path,
) -> bool:
    """Save one RGBA animation frame as PNG."""
    try:
        plt.imsave(str(output_path), frame_rgba)
    except Exception as exc:
        print(f"[anim] PNG frame save failed for '{output_path.name}': {exc}")
        return False
    return True


@dataclass(slots=True)
class _RawVideoFfmpegSink:
    """One ffmpeg process consuming raw RGBA frames from stdin."""

    process: Any
    output_path: Path
    label: str
    active: bool = True

    def write_frame(self, frame_rgba: np.ndarray) -> bool:
        """Write one frame to the ffmpeg stdin pipe."""
        if (not self.active) or (self.process.stdin is None):
            return False
        try:
            self.process.stdin.write(
                np.ascontiguousarray(frame_rgba, dtype=np.uint8).tobytes()
            )
            return True
        except Exception as exc:
            print(f"[anim] {self.label} stream failed for '{self.output_path.name}': {exc}")
            self.active = False
            try:
                self.process.stdin.close()
            except Exception:
                pass
            return False

    def finalize(self) -> bool:
        """Close the ffmpeg pipe and return True when encoding succeeded."""
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except Exception:
                pass

        stderr_text = ""
        if self.process.stderr is not None:
            try:
                stderr_text = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            except Exception:
                stderr_text = ""

        return_code = self.process.wait()
        if return_code != 0:
            err = stderr_text if stderr_text else "unknown ffmpeg error"
            print(f"[anim] {self.label} export failed for '{self.output_path.name}': {err}")
            return False
        return bool(self.active)


def _start_rawvideo_mp4_ffmpeg(
    *,
    output_path: Path,
    fps: int,
    frame_width: int,
    frame_height: int,
    ffmpeg_preset: str,
    ffmpeg_crf: int,
    ffmpeg_threads: int,
    ffmpeg_loglevel: str,
) -> _RawVideoFfmpegSink | None:
    """Start an ffmpeg MP4 encoder that consumes raw RGBA frames."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("[anim] MP4 export skipped: ffmpeg is not available on PATH.")
        return None

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        str(ffmpeg_loglevel),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-video_size",
        f"{int(frame_width)}x{int(frame_height)}",
        "-framerate",
        str(max(1, int(fps))),
        "-i",
        "pipe:0",
        "-threads",
        str(int(ffmpeg_threads)),
        "-c:v",
        "libx264",
        "-preset",
        str(ffmpeg_preset),
        "-crf",
        str(int(ffmpeg_crf)),
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return _RawVideoFfmpegSink(
        process=process,
        output_path=output_path,
        label="MP4",
    )


def _start_rawvideo_gif_ffmpeg(
    *,
    output_path: Path,
    fps: int,
    frame_width: int,
    frame_height: int,
    ffmpeg_loglevel: str,
) -> _RawVideoFfmpegSink | None:
    """Start an ffmpeg GIF encoder that consumes raw RGBA frames."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return None

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        str(ffmpeg_loglevel),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-video_size",
        f"{int(frame_width)}x{int(frame_height)}",
        "-framerate",
        str(max(1, int(fps))),
        "-i",
        "pipe:0",
        "-vf",
        "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse",
        "-loop",
        "0",
        str(output_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return _RawVideoFfmpegSink(
        process=process,
        output_path=output_path,
        label="GIF",
    )


def _save_animation_gif_from_arrays(
    *,
    frames_rgba: list[np.ndarray],
    output_path: Path,
    fps: int,
) -> bool:
    """Save GIF from in-memory RGBA frames using Pillow."""
    if not frames_rgba:
        print("[anim] GIF export skipped: no in-memory frames available.")
        return False
    try:
        from PIL import Image
    except Exception as exc:
        print(f"[anim] GIF export skipped: Pillow unavailable ({exc}).")
        return False

    duration_ms = max(1, int(round(1000.0 / float(max(1, int(fps))))))
    try:
        pil_frames = [
            Image.fromarray(np.asarray(frame[..., :3], dtype=np.uint8), mode="RGB")
            for frame in frames_rgba
        ]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    except Exception as exc:
        print(f"[anim] GIF export failed: {exc}")
        return False
    return True


def _render_animation_frames_fast(
    *,
    set_key: str,
    samples: list[RandomSlotSample],
    animations_dir: Path,
    title_prefix: str,
    fps_values: list[int],
    vmax: float,
    vis_cmap: str,
    frame_width: int,
    frame_height: int,
    frame_progress: bool,
    render_backend: str,
    keep_frame_pngs: bool,
    save_animation_mp4: bool,
    save_animation_gif: bool,
    ffmpeg_preset: str,
    ffmpeg_crf: int,
    ffmpeg_threads: int,
    ffmpeg_loglevel: str,
) -> tuple[int, list[str], list[str]]:
    """Render and encode animation frames using the fast raster path."""
    if not samples:
        return 0, [], []

    total_frames = len(samples)
    renderer = _FastS1586AnimationRenderer(
        frame_width=max(1, int(frame_width)),
        frame_height=max(1, int(frame_height)),
        vmax=float(vmax),
        vis_cmap=vis_cmap,
        backend=render_backend,
        raster_res=max(256, min(max(int(frame_width), int(frame_height)), 1024)),
    )

    frame_dir: Path | None = None
    if keep_frame_pngs:
        frame_dir = animations_dir / f"{set_key}_frames_{total_frames:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

    mp4_sinks: list[tuple[_RawVideoFfmpegSink, str]] = []
    gif_sinks: list[tuple[_RawVideoFfmpegSink, str]] = []
    gif_fallback_targets: list[tuple[Path, int]] = []
    mp4_outputs: list[str] = []
    gif_outputs: list[str] = []

    for fps in fps_values:
        if save_animation_mp4:
            mp4_path = animations_dir / f"skycell_satellite_count_{set_key}_{total_frames:03d}_{fps}fps.mp4"
            sink = _start_rawvideo_mp4_ffmpeg(
                output_path=mp4_path,
                fps=int(fps),
                frame_width=max(1, int(frame_width)),
                frame_height=max(1, int(frame_height)),
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_crf=ffmpeg_crf,
                ffmpeg_threads=ffmpeg_threads,
                ffmpeg_loglevel=ffmpeg_loglevel,
            )
            if sink is not None:
                mp4_sinks.append((sink, str(mp4_path)))
        if save_animation_gif:
            gif_path = animations_dir / f"skycell_satellite_count_{set_key}_{total_frames:03d}_{fps}fps.gif"
            sink = _start_rawvideo_gif_ffmpeg(
                output_path=gif_path,
                fps=int(fps),
                frame_width=max(1, int(frame_width)),
                frame_height=max(1, int(frame_height)),
                ffmpeg_loglevel=ffmpeg_loglevel,
            )
            if sink is not None:
                gif_sinks.append((sink, str(gif_path)))
            else:
                gif_fallback_targets.append((gif_path, int(fps)))

    keep_frames_in_memory = bool(gif_fallback_targets)
    frames_rgba_cache: list[np.ndarray] = [] if keep_frames_in_memory else []

    progress_bar: Any = None
    if frame_progress:
        tqdm = _load_tqdm()
        if tqdm is not None:
            progress_bar = tqdm(
                total=total_frames,
                desc=f"[anim][frames] {set_key}",
                unit="frame",
                leave=False,
                dynamic_ncols=True,
            )

    rendered_frames = 0
    try:
        for frame_idx_zero, sample in enumerate(samples):
            title = _build_animation_frame_title(
                sample=sample,
                frame_idx_zero=frame_idx_zero,
                total_frames=total_frames,
                title_prefix=title_prefix,
            )
            try:
                frame_rgba = renderer.render_frame(
                    sample.counts,
                    title=title,
                )
            except Exception as exc:
                print(f"[anim] Fast frame rendering failed at {frame_idx_zero + 1}/{total_frames}: {exc}")
                break

            rendered_frames += 1

            if frame_dir is not None:
                frame_path = frame_dir / f"frame_{rendered_frames:03d}.png"
                _save_animation_frame_png(frame_rgba, frame_path)

            if keep_frames_in_memory:
                frames_rgba_cache.append(frame_rgba.copy())

            for sink, _ in mp4_sinks:
                sink.write_frame(frame_rgba)
            for sink, _ in gif_sinks:
                sink.write_frame(frame_rgba)

            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    for sink, output_path in mp4_sinks:
        if sink.finalize():
            print(f"[anim] Saved MP4 animation: {output_path}")
            mp4_outputs.append(output_path)
    for sink, output_path in gif_sinks:
        if sink.finalize():
            print(f"[anim] Saved GIF animation: {output_path}")
            gif_outputs.append(output_path)

    if keep_frames_in_memory and frames_rgba_cache:
        for gif_path, fps in gif_fallback_targets:
            if _save_animation_gif_from_arrays(
                frames_rgba=frames_rgba_cache,
                output_path=gif_path,
                fps=int(fps),
            ):
                print(f"[anim] Saved GIF animation: {gif_path}")
                gif_outputs.append(str(gif_path))

    return rendered_frames, mp4_outputs, gif_outputs


def _export_animation_set(
    *,
    set_key: str,
    set_samples: list[RandomSlotSample],
    title_prefix: str,
    animations_dir: Path,
    fps_values: list[int],
    vmax: float,
    plotter: _ReusableSlotCountPlotter | None,
    cfg: SatelliteDistributionConfig,
    effective_backend: str,
) -> tuple[int, list[str], list[str]]:
    """Export one animation set using either the legacy or fast backend."""
    if effective_backend == "plotly":
        frame_dir = animations_dir / f"{set_key}_frames_{len(set_samples):03d}"
        frame_paths = _render_animation_png_frames(
            samples=set_samples,
            frame_dir=frame_dir,
            title_prefix=title_prefix,
            vmax=vmax,
            vis_cmap=cfg.skycell_vis_cmap,
            vis_projection=cfg.skycell_vis_2d_projection,
            vis_engine=cfg.skycell_vis_engine,
            show_plots=cfg.show_plots,
            frame_width=cfg.animation_frame_width,
            frame_height=cfg.animation_frame_height,
            frame_scale=cfg.animation_frame_scale,
            frame_progress=cfg.animation_frame_progress,
            plotter=plotter,
        )
        if not frame_paths:
            return 0, [], []

        frame_pattern = str(frame_dir / "frame_%03d.png")
        mp4_outputs: list[str] = []
        gif_outputs: list[str] = []
        for fps in fps_values:
            if cfg.save_animation_mp4:
                mp4_path = animations_dir / (
                    f"skycell_satellite_count_{set_key}_{len(frame_paths):03d}_{fps}fps.mp4"
                )
                if _save_animation_mp4_ffmpeg(
                    frame_pattern=frame_pattern,
                    output_path=mp4_path,
                    fps=int(fps),
                    total_frames=len(frame_paths),
                    ffmpeg_preset=cfg.ffmpeg_preset,
                    ffmpeg_crf=cfg.ffmpeg_crf,
                    ffmpeg_threads=cfg.ffmpeg_threads,
                    ffmpeg_loglevel=cfg.ffmpeg_loglevel,
                    ffmpeg_show_progress=cfg.ffmpeg_show_progress,
                    ffmpeg_progress_step_percent=cfg.ffmpeg_progress_step_percent,
                ):
                    print(f"[anim] Saved MP4 animation: {mp4_path}")
                    mp4_outputs.append(str(mp4_path))
            if cfg.save_animation_gif:
                gif_path = animations_dir / (
                    f"skycell_satellite_count_{set_key}_{len(frame_paths):03d}_{fps}fps.gif"
                )
                if _save_animation_gif(
                    frame_paths=frame_paths,
                    output_path=gif_path,
                    fps=int(fps),
                    frame_pattern=frame_pattern,
                    ffmpeg_loglevel=cfg.ffmpeg_loglevel,
                ):
                    print(f"[anim] Saved GIF animation: {gif_path}")
                    gif_outputs.append(str(gif_path))
        return len(frame_paths), mp4_outputs, gif_outputs

    effective_width = max(1, int(round(cfg.animation_frame_width * cfg.animation_frame_scale)))
    effective_height = max(1, int(round(cfg.animation_frame_height * cfg.animation_frame_scale)))
    return _render_animation_frames_fast(
        set_key=set_key,
        samples=set_samples,
        animations_dir=animations_dir,
        title_prefix=title_prefix,
        fps_values=fps_values,
        vmax=vmax,
        vis_cmap=cfg.skycell_vis_cmap,
        frame_width=effective_width,
        frame_height=effective_height,
        frame_progress=cfg.animation_frame_progress,
        render_backend=effective_backend,
        keep_frame_pngs=bool(cfg.keep_animation_frame_pngs),
        save_animation_mp4=bool(cfg.save_animation_mp4),
        save_animation_gif=bool(cfg.save_animation_gif),
        ffmpeg_preset=cfg.ffmpeg_preset,
        ffmpeg_crf=cfg.ffmpeg_crf,
        ffmpeg_threads=cfg.ffmpeg_threads,
        ffmpeg_loglevel=cfg.ffmpeg_loglevel,
    )


def _ordered_unique_fps(values: list[int]) -> list[int]:
    """Return positive unique FPS values while preserving input order."""
    fps_values: list[int] = []
    for value in values:
        v = max(1, int(value))
        if v not in fps_values:
            fps_values.append(v)
    return fps_values


@lru_cache(maxsize=1)
def _load_tqdm():
    """Return `tqdm.auto.tqdm` if available, otherwise `None`."""
    try:
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        return None


def _render_animation_png_frames(
    *,
    samples: list[RandomSlotSample],
    frame_dir: Path,
    title_prefix: str,
    vmax: float,
    vis_cmap: str,
    vis_projection: str,
    vis_engine: str,
    show_plots: bool,
    frame_width: int,
    frame_height: int,
    frame_scale: float,
    frame_progress: bool,
    plotter: _ReusableSlotCountPlotter | None = None,
) -> list[Path]:
    """Render per-slot maps to numbered PNG frames for animation encoders."""
    if not samples:
        return []

    frame_dir.mkdir(parents=True, exist_ok=True)
    total_frames = len(samples)
    frame_paths: list[Path] = []
    can_write_images: bool | None = None

    frame_indices: Any = range(total_frames)
    progress_bar: Any = None
    if frame_progress:
        tqdm = _load_tqdm()
        if tqdm is not None:
            progress_bar = tqdm(
                total=total_frames,
                desc=f"[anim][frames] {frame_dir.name}",
                unit="frame",
                leave=False,
                dynamic_ncols=True,
            )

    try:
        for frame_idx_zero in frame_indices:
            sample = samples[int(frame_idx_zero)]

            title = _build_animation_frame_title(
                sample=sample,
                frame_idx_zero=int(frame_idx_zero),
                total_frames=total_frames,
                title_prefix=title_prefix,
            )
            fig = _plot_slot_counts(
                sample.counts,
                title=title,
                vmax=vmax,
                save_html_path=None,
                vis_cmap=vis_cmap,
                vis_projection=vis_projection,
                vis_engine=vis_engine,
                show_plots=show_plots,
                plotter=plotter,
            )

            if can_write_images is None:
                can_write_images = hasattr(fig, "write_image")
                if not can_write_images:
                    print("[anim] Frame export skipped: figure backend has no write_image().")
                    return []

            idx = int(frame_idx_zero) + 1
            frame_path = frame_dir / f"frame_{idx:03d}.png"
            try:
                fig.write_image(
                    str(frame_path),
                    format="png",
                    width=int(frame_width),
                    height=int(frame_height),
                    scale=float(frame_scale),
                )
                frame_paths.append(frame_path)
            except Exception as exc:
                print(f"[anim] Frame export failed at {idx}/{total_frames}: {exc}")
                break

            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return frame_paths


def satellite_distribution_over_sky(
    storage_filename: str | Path,
    *,
    config: SatelliteDistributionConfig | None = None,
    **config_overrides: Any,
) -> dict[str, Any]:
    """
    Run the satellite-per-skycell processing pipeline.

    Parameters
    ----------
    storage_filename : str or Path
        Input HDF5 file containing ``/iter/iter_*/sat_azimuth`` and
        ``/iter/iter_*/sat_elevation`` datasets. Each dataset is expected to
        have slot index on axis 0.
    config : SatelliteDistributionConfig, optional
        Base configuration object. When omitted, the default configuration is
        used.
    **config_overrides : Any
        Keyword overrides matching ``SatelliteDistributionConfig`` field names.

    Returns
    -------
    dict
        Run diagnostics, selected sample counts, output paths, and export
        metadata. When results are saved, the returned paths point to files in
        the created run directory.

    Raises
    ------
    FileNotFoundError
        Raised when the input HDF5 file does not exist.
    ValueError
        Raised when required datasets are missing, malformed, or no slots are
        processed.
    KeyError
        Raised when an iteration group is missing required datasets.

    Notes
    -----
    - Per-slot values are integer satellite counts per sky cell.
    - ``average_counts`` is the arithmetic mean of these count arrays across
      selected slots.
    - Random plot and animation samples are selected with reservoir sampling
      when a full pass is required.
    - Still-image plots continue to use the regular hemisphere plotting path.
    - Animation export supports both the legacy Plotly/Kaleido backend and a
      fast raster backend. The fast backend currently applies only to the
      default full-grid S.1586 polar workflow.
    """
    cfg = _resolve_config(config, config_overrides)

    file_path = Path(storage_filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    n_skycells, sky_mapper = _build_sky_mapper(
        cfg.skycell_mode,
        n_cells=max(1, int(cfg.s1586_n_cells)),
    )

    enable_random_plots = bool(cfg.enable_random_plot_samples) and int(cfg.random_timestep_plots) > 0
    enable_average_plot = bool(cfg.enable_average_plot)
    enable_animation_random = (
        bool(cfg.save_animation)
        and bool(cfg.save_outputs)
        and bool(cfg.enable_animation_random_set)
        and int(cfg.animation_random_sample_count) > 0
    )
    enable_animation_subsequent = (
        bool(cfg.save_animation)
        and bool(cfg.save_outputs)
        and bool(cfg.enable_animation_subsequent_set)
        and int(cfg.animation_subsequent_sample_count) > 0
    )

    if not any(
        [
            enable_random_plots,
            enable_average_plot,
            enable_animation_random,
            enable_animation_subsequent,
        ]
    ):
        print("No processing modules enabled. Nothing to do.")
        return {
            "storage_filename": str(file_path),
            "config": asdict(cfg),
            "run_dir": None,
            "processed_slots": 0,
            "average_slots_used": 0,
            "stop_reason": "no_modules_enabled",
            "stopped_early": False,
        }

    random_plot_target_size = max(0, int(cfg.random_timestep_plots))
    random_animation_target_size = max(0, int(cfg.animation_random_sample_count))
    subsequent_animation_target_size = max(0, int(cfg.animation_subsequent_sample_count))

    average_slot_stride = max(1, int(cfg.average_slot_stride))
    average_max_slots = max(0, int(cfg.average_max_slots))
    max_raw_slots_to_read = max(0, int(cfg.max_raw_slots_to_read))
    read_slot_chunk = max(1, int(cfg.read_slot_chunk))
    progress_every_slots = max(1, int(cfg.progress_every_slots))

    full_pass_required = (
        (enable_random_plots and bool(cfg.random_plot_require_full_pass))
        or (enable_animation_random and bool(cfg.random_animation_require_full_pass))
        or (enable_average_plot and average_max_slots <= 0)
    )

    rng = np.random.default_rng(int(cfg.random_seed))
    dedup_quant_scale: float | None = None
    dedup_az_wrap: int | None = None
    if cfg.deduplicate_by_az_el and cfg.deduplicate_round_decimals is not None:
        decimals = max(0, int(cfg.deduplicate_round_decimals))
        dedup_quant_scale = float(10 ** decimals)
        dedup_az_wrap = int(round(360.0 * dedup_quant_scale))

    random_plot_samples: list[RandomSlotSample] = []
    random_animation_samples: list[RandomSlotSample] = []
    subsequent_animation_samples: list[RandomSlotSample] = []
    # Reused per-slot buffer to avoid allocating a new skycell array for each slot.
    counts_scratch = np.zeros(n_skycells, dtype=np.int32)

    sum_counts = np.zeros(n_skycells, dtype=np.float64)
    processed_slots = 0
    average_slots_used = 0
    slots_with_valid_points = 0
    slots_with_mappable_satellites = 0
    total_valid_points = 0
    total_unique_satellites = 0
    total_mappable_satellites = 0
    sampling_seen_plots = 0
    sampling_seen_random_animation = 0
    stop_requested = False
    stop_reason = "completed_full_scan"

    def _targets_reached() -> bool:
        """Return True when all enabled sampling/average quotas are satisfied."""
        if enable_random_plots and len(random_plot_samples) < random_plot_target_size:
            return False
        if enable_animation_random and len(random_animation_samples) < random_animation_target_size:
            return False
        if enable_animation_subsequent and len(subsequent_animation_samples) < subsequent_animation_target_size:
            return False
        if enable_average_plot and average_max_slots > 0 and average_slots_used < average_max_slots:
            return False
        return True

    t0 = time.perf_counter()
    print(f"Input file: {file_path}")
    print(
        "[config] modules: "
        f"random_plots={enable_random_plots}, "
        f"average={enable_average_plot}, "
        f"anim_random={enable_animation_random}, "
        f"anim_subsequent={enable_animation_subsequent}"
    )
    print(
        "[config] average controls: "
        f"stride={average_slot_stride}, "
        f"max_slots={average_max_slots if average_max_slots > 0 else 'all'}, "
        f"nonempty_only={cfg.average_sample_nonempty_only}"
    )
    if not full_pass_required:
        print("[config] full-pass disabled: run may stop early after all enabled targets are filled.")

    scenario.flush_writes(str(file_path))
    with h5py.File(file_path, "r") as h5:
        iter_names = _iter_names(h5)
        if not iter_names:
            raise ValueError("No '/iter/iter_*' groups found in input file.")

        print(f"Iterations found: {len(iter_names)}")
        iter_datasets: list[tuple[str, Any, Any, Any, int]] = []
        total_slots_available = 0
        for iter_name in iter_names:
            group = h5["iter"][iter_name]
            if "sat_azimuth" not in group or "sat_elevation" not in group:
                raise KeyError(f"{iter_name}: missing required datasets sat_azimuth/sat_elevation.")

            ds_az = group["sat_azimuth"]
            ds_el = group["sat_elevation"]
            ds_times = group["times"] if "times" in group else None
            if ds_az.shape != ds_el.shape:
                raise ValueError(
                    f"{iter_name}: sat_azimuth shape {tuple(ds_az.shape)} "
                    f"does not match sat_elevation shape {tuple(ds_el.shape)}."
                )
            if ds_az.ndim < 1:
                raise ValueError(f"{iter_name}: expected sat_azimuth to have slot axis, got ndim={ds_az.ndim}.")

            n_slots = int(ds_az.shape[0])
            iter_datasets.append((iter_name, ds_az, ds_el, ds_times, n_slots))
            total_slots_available += n_slots

        points_per_slot_hint = 1
        if iter_datasets:
            first_shape = tuple(iter_datasets[0][1].shape)
            if len(first_shape) > 1:
                points_per_slot_hint = int(np.prod(first_shape[1:], dtype=np.int64))

        expected_slots = (
            min(total_slots_available, max_raw_slots_to_read)
            if max_raw_slots_to_read > 0
            else total_slots_available
        )
        mapper_backend, numba_mapper = _choose_sky_mapper_backend(
            requested_backend=cfg.sky_mapper_backend,
            skycell_mode=cfg.skycell_mode,
            expected_slots=expected_slots,
            points_per_slot_hint=points_per_slot_hint,
            numba_mapper_min_work=cfg.numba_mapper_min_work,
        )
        slot_sky_mapper = sky_mapper
        if mapper_backend == "numba" and numba_mapper is not None:
            try:
                # Warm-up ensures first expensive JIT compile is paid once upfront.
                numba_mapper(np.asarray([0.0], dtype=np.float64), np.asarray([0.0], dtype=np.float64))

                def _numba_slot_mapper(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
                    return np.asarray(
                        numba_mapper(
                            np.asarray(az_deg, dtype=np.float64).reshape(-1),
                            np.asarray(el_deg, dtype=np.float64).reshape(-1),
                        ),
                        dtype=np.int32,
                    )

                slot_sky_mapper = _numba_slot_mapper
            except Exception as exc:
                warnings.warn(
                    f"Failed to initialize direct Numba sky mapper ({exc}); using dispatcher backend."
                )
                mapper_backend = "dispatcher"

        print(
            "[config] sky mapper backend: "
            f"requested={cfg.sky_mapper_backend}, effective={mapper_backend}, "
            f"expected_slots={expected_slots:,}, points_per_slot_hint={points_per_slot_hint:,}"
        )

        effective_total_slots = (
            min(total_slots_available, max_raw_slots_to_read)
            if max_raw_slots_to_read > 0
            else total_slots_available
        )

        slot_progress_bar: Any = None
        tqdm = _load_tqdm()
        if tqdm is not None:
            slot_progress_bar = tqdm(
                total=max(0, int(effective_total_slots)),
                desc="[progress] slots",
                unit="slot",
                leave=False,
                dynamic_ncols=True,
            )
        slots_since_progress_update = 0
        slot_global_idx = 0

        try:
            for ii, (iter_name, ds_az, ds_el, ds_times, n_slots) in enumerate(iter_datasets, start=1):
                print(f"[iter {ii}/{len(iter_names)}] {iter_name}: slots={n_slots:,}, shape={tuple(ds_az.shape)}")

                for s0 in range(0, n_slots, read_slot_chunk):
                    s1 = min(n_slots, s0 + read_slot_chunk)
                    az_chunk = np.asarray(ds_az[s0:s1], dtype=np.float32)
                    el_chunk = np.asarray(ds_el[s0:s1], dtype=np.float32)
                    if az_chunk.shape != el_chunk.shape:
                        raise ValueError(
                            f"{iter_name}: chunk shape mismatch for slots [{s0}, {s1}) "
                            f"({az_chunk.shape} vs {el_chunk.shape})."
                        )

                    times_chunk: np.ndarray | None = None
                    if ds_times is not None:
                        times_chunk = np.asarray(ds_times[s0:s1], dtype=np.float64).reshape(-1)

                    for local_offset in range(int(az_chunk.shape[0])):
                        slot_local_idx = s0 + local_offset
                        counts, n_valid_points, n_unique_satellites, n_mappable_satellites = (
                            _count_satellites_per_skycell_slot(
                                az_chunk[local_offset],
                                el_chunk[local_offset],
                                sky_mapper=slot_sky_mapper,
                                n_skycells=n_skycells,
                                deduplicate=cfg.deduplicate_by_az_el,
                                edge_eps_deg=cfg.coord_edge_eps_deg,
                                deduplicate_round_decimals=cfg.deduplicate_round_decimals,
                                dedup_quant_scale=dedup_quant_scale,
                                dedup_az_wrap=dedup_az_wrap,
                                counts_buffer=counts_scratch,
                            )
                        )

                        processed_slots += 1
                        slots_since_progress_update += 1
                        total_valid_points += n_valid_points
                        total_unique_satellites += n_unique_satellites
                        total_mappable_satellites += n_mappable_satellites

                        if n_valid_points > 0:
                            slots_with_valid_points += 1
                        if n_mappable_satellites > 0:
                            slots_with_mappable_satellites += 1

                        slot_time_mjd: float | None = None
                        if times_chunk is not None and local_offset < times_chunk.size:
                            t_mjd = float(times_chunk[local_offset])
                            if np.isfinite(t_mjd):
                                slot_time_mjd = t_mjd

                        if enable_average_plot:
                            stride_hit = (slot_global_idx % average_slot_stride) == 0
                            avg_nonempty_ok = (
                                (n_mappable_satellites > 0)
                                if cfg.average_sample_nonempty_only
                                else True
                            )
                            avg_quota_ok = (average_max_slots <= 0) or (average_slots_used < average_max_slots)
                            if stride_hit and avg_nonempty_ok and avg_quota_ok:
                                sum_counts += counts
                                average_slots_used += 1

                        eligible_plot = (
                            enable_random_plots
                            and ((n_mappable_satellites > 0) if cfg.random_sample_nonempty_only else True)
                        )
                        eligible_random_animation = (
                            enable_animation_random
                            and ((n_mappable_satellites > 0) if cfg.animation_random_nonempty_only else True)
                        )
                        eligible_subsequent_animation = (
                            enable_animation_subsequent
                            and (
                                (n_mappable_satellites > 0)
                                if cfg.animation_subsequent_nonempty_only
                                else True
                            )
                        )
                        need_subsequent = (
                            len(subsequent_animation_samples) < subsequent_animation_target_size
                            and eligible_subsequent_animation
                        )

                        sample_obj_cache: RandomSlotSample | None = None

                        def _build_sample_obj() -> RandomSlotSample:
                            """Lazily materialize slot payload only when selection requires it."""
                            nonlocal sample_obj_cache
                            if sample_obj_cache is None:
                                # Materialize sample payload only if this slot is actually selected
                                # by at least one sink (reservoir/subsequent list).
                                sample_obj_cache = RandomSlotSample(
                                    iter_name=iter_name,
                                    slot_local_idx=int(slot_local_idx),
                                    slot_global_idx=int(slot_global_idx),
                                    time_mjd=slot_time_mjd,
                                    counts=counts.copy(),
                                    n_satellites=int(n_mappable_satellites),
                                    n_active_skycells=int(np.count_nonzero(counts)),
                                )
                            return sample_obj_cache

                        if eligible_plot:
                            sampling_seen_plots += 1
                            if len(random_plot_samples) < random_plot_target_size:
                                random_plot_samples.append(_build_sample_obj())
                            else:
                                j = int(rng.integers(0, sampling_seen_plots))
                                if j < random_plot_target_size:
                                    random_plot_samples[j] = _build_sample_obj()

                        if eligible_random_animation:
                            sampling_seen_random_animation += 1
                            if len(random_animation_samples) < random_animation_target_size:
                                random_animation_samples.append(_build_sample_obj())
                            else:
                                j = int(rng.integers(0, sampling_seen_random_animation))
                                if j < random_animation_target_size:
                                    random_animation_samples[j] = _build_sample_obj()

                        if need_subsequent:
                            subsequent_animation_samples.append(_build_sample_obj())

                        slot_global_idx += 1

                        if (
                            slot_progress_bar is not None
                            and processed_slots % progress_every_slots == 0
                        ):
                            slot_progress_bar.update(slots_since_progress_update)
                            slots_since_progress_update = 0
                            avg_target_text = (
                                "all"
                                if (not enable_average_plot or average_max_slots <= 0)
                                else f"{average_max_slots}"
                            )
                            slot_progress_bar.set_postfix_str(
                                f"avg={average_slots_used}/{avg_target_text}, "
                                f"plots={len(random_plot_samples)}/{random_plot_target_size}, "
                                f"arand={len(random_animation_samples)}/{random_animation_target_size}, "
                                f"aseq={len(subsequent_animation_samples)}/{subsequent_animation_target_size}"
                            )

                        if max_raw_slots_to_read > 0 and processed_slots >= max_raw_slots_to_read:
                            stop_requested = True
                            stop_reason = f"max_raw_slots_to_read_reached({max_raw_slots_to_read})"
                        elif (not full_pass_required) and _targets_reached():
                            stop_requested = True
                            stop_reason = "sampling_targets_reached"

                        if stop_requested:
                            break
                    if stop_requested:
                        break
                if stop_requested:
                    break
        finally:
            if slot_progress_bar is not None:
                if slots_since_progress_update > 0:
                    slot_progress_bar.update(slots_since_progress_update)
                slot_progress_bar.close()

    if processed_slots == 0:
        raise ValueError("No slots processed from sat_azimuth/sat_elevation datasets.")

    elapsed = time.perf_counter() - t0
    if enable_average_plot and average_slots_used > 0:
        average_counts = sum_counts / float(average_slots_used)
    else:
        average_counts = np.zeros(n_skycells, dtype=np.float64)

    random_plot_samples.sort(key=lambda x: x.slot_global_idx)
    random_animation_samples.sort(key=lambda x: x.slot_global_idx)

    max_random_cell_count = (
        max(float(np.max(sample.counts)) for sample in random_plot_samples)
        if (enable_random_plots and random_plot_samples)
        else 0.0
    )
    max_random_animation_cell_count = (
        max(float(np.max(sample.counts)) for sample in random_animation_samples)
        if (enable_animation_random and random_animation_samples)
        else 0.0
    )
    max_subsequent_animation_cell_count = (
        max(float(np.max(sample.counts)) for sample in subsequent_animation_samples)
        if (enable_animation_subsequent and subsequent_animation_samples)
        else 0.0
    )
    max_average_cell_count = (
        float(np.max(average_counts))
        if (enable_average_plot and average_slots_used > 0 and average_counts.size)
        else 0.0
    )
    vmax_common = max(
        1.0,
        max_random_cell_count,
        max_random_animation_cell_count,
        max_subsequent_animation_cell_count,
        max_average_cell_count,
    )
    slot_count_plotter = _ReusableSlotCountPlotter(
        vmax=vmax_common,
        vis_cmap=cfg.skycell_vis_cmap,
        vis_projection=cfg.skycell_vis_2d_projection,
        vis_engine=cfg.skycell_vis_engine,
        show_plots=cfg.show_plots,
    )
    animation_render_backend_requested = str(cfg.animation_render_backend)
    animation_render_backend_effective: str | None = None
    animation_render_backend_reason = "animations_disabled"
    animation_used_direct_streaming = False
    animation_kept_frame_pngs = False
    if enable_animation_random or enable_animation_subsequent:
        (
            animation_render_backend_effective,
            animation_render_backend_reason,
            animation_used_direct_streaming,
        ) = _resolve_animation_render_backend(
            requested_backend=cfg.animation_render_backend,
            skycell_mode=cfg.skycell_mode,
            n_skycells=n_skycells,
            vis_projection=cfg.skycell_vis_2d_projection,
        )
        animation_kept_frame_pngs = (
            bool(cfg.keep_animation_frame_pngs)
            if animation_used_direct_streaming
            else True
        )

    run_dir: Path | None = None
    if cfg.save_outputs:
        run_dir = _make_output_run_dir(cfg.output_dir, cfg.output_prefix)

    random_plot_png_count = 0
    random_plot_jpg_count = 0
    average_plot_png_count = 0
    average_plot_jpg_count = 0
    animations_dir: Path | None = None
    animation_mp4_outputs: list[str] = []
    animation_gif_outputs: list[str] = []
    animation_random_frame_count = 0
    animation_subsequent_frame_count = 0

    if plot_hemisphere_2D is None:
        if enable_random_plots or enable_average_plot:
            print("[plot] Skipped hemisphere plots: plot_hemisphere_2D is unavailable.")
    else:
        if enable_random_plots:
            if random_plot_samples:
                for idx, sample in enumerate(random_plot_samples, start=1):
                    base_name = f"skycell_satellite_count_random_{idx:03d}"
                    html_path = (
                        (run_dir / f"{base_name}.html")
                        if (run_dir is not None and cfg.save_plots_html)
                        else None
                    )
                    png_path = (
                        (run_dir / f"{base_name}.png")
                        if (run_dir is not None and cfg.save_plots_png)
                        else None
                    )
                    jpg_path = (
                        (run_dir / f"{base_name}.jpg")
                        if (run_dir is not None and cfg.save_plots_jpg)
                        else None
                    )
                    title = (
                        f"Satellites per skycell - random timestep {idx}/{len(random_plot_samples)}"
                        f"<br><sup>iter={sample.iter_name}, local_slot={sample.slot_local_idx}, "
                        f"global_slot={sample.slot_global_idx}, MJD={_format_mjd(sample.time_mjd)}, "
                        f"satellites={sample.n_satellites}</sup>"
                    )
                    fig = _plot_slot_counts(
                        sample.counts,
                        title=title,
                        vmax=vmax_common,
                        save_html_path=html_path,
                        vis_cmap=cfg.skycell_vis_cmap,
                        vis_projection=cfg.skycell_vis_2d_projection,
                        vis_engine=cfg.skycell_vis_engine,
                        show_plots=cfg.show_plots,
                        plotter=slot_count_plotter,
                    )
                    if html_path is not None:
                        print(f"[plot] Saved random timestep map: {html_path}")
                    png_saved, jpg_saved = _save_plotly_static_images(
                        fig,
                        png_path=png_path,
                        jpg_path=jpg_path,
                        export_width=cfg.export_image_width,
                        export_height=cfg.export_image_height,
                        export_scale=cfg.export_image_scale,
                    )
                    if png_saved and png_path is not None:
                        print(f"[plot] Saved random timestep PNG: {png_path}")
                        random_plot_png_count += 1
                    if jpg_saved and jpg_path is not None:
                        print(f"[plot] Saved random timestep JPG: {jpg_path}")
                        random_plot_jpg_count += 1
            else:
                print("[plot] No eligible slots found for random timestep maps.")
        else:
            print("[plot] Random timestep maps disabled by config.")

        if enable_average_plot:
            if average_slots_used > 0:
                avg_html_path = (
                    (run_dir / "skycell_satellite_count_average.html")
                    if (run_dir is not None and cfg.save_plots_html)
                    else None
                )
                avg_png_path = (
                    (run_dir / "skycell_satellite_count_average.png")
                    if (run_dir is not None and cfg.save_plots_png)
                    else None
                )
                avg_jpg_path = (
                    (run_dir / "skycell_satellite_count_average.jpg")
                    if (run_dir is not None and cfg.save_plots_jpg)
                    else None
                )
                avg_title = (
                    "Average satellites per skycell"
                    f"<br><sup>average_slots={average_slots_used:,}, "
                    f"stride={average_slot_stride}, "
                    f"deduplicate_az_el={cfg.deduplicate_by_az_el}</sup>"
                )
                avg_fig = _plot_slot_counts(
                    np.asarray(average_counts, dtype=np.float64),
                    title=avg_title,
                    vmax=vmax_common,
                    save_html_path=avg_html_path,
                    vis_cmap=cfg.skycell_vis_cmap,
                    vis_projection=cfg.skycell_vis_2d_projection,
                    vis_engine=cfg.skycell_vis_engine,
                    show_plots=cfg.show_plots,
                    plotter=slot_count_plotter,
                )
                if avg_html_path is not None:
                    print(f"[plot] Saved average map: {avg_html_path}")
                avg_png_saved, avg_jpg_saved = _save_plotly_static_images(
                    avg_fig,
                    png_path=avg_png_path,
                    jpg_path=avg_jpg_path,
                    export_width=cfg.export_image_width,
                    export_height=cfg.export_image_height,
                    export_scale=cfg.export_image_scale,
                )
                if avg_png_saved and avg_png_path is not None:
                    print(f"[plot] Saved average PNG: {avg_png_path}")
                    average_plot_png_count += 1
                if avg_jpg_saved and avg_jpg_path is not None:
                    print(f"[plot] Saved average JPG: {avg_jpg_path}")
                    average_plot_jpg_count += 1
            else:
                print("[plot] Average map skipped: no slots matched average sampling rules.")
        else:
            print("[plot] Average map disabled by config.")

    if (enable_animation_random or enable_animation_subsequent) and run_dir is not None:
        if animation_render_backend_effective == "plotly" and plot_hemisphere_2D is None:
            print("[anim] Skipped animation export: plot_hemisphere_2D is unavailable.")
        else:
            animations_dir = run_dir / str(cfg.animation_output_subdir)
            animations_dir.mkdir(parents=True, exist_ok=True)
            fps_values = _ordered_unique_fps([int(cfg.animation_fps), int(cfg.animation_extra_fps)])

            animation_sets: list[tuple[str, list[RandomSlotSample], str]] = []
            if enable_animation_random:
                animation_sets.append(
                    (
                        "random",
                        random_animation_samples,
                        "Random-sample animation frame",
                    )
                )
            if enable_animation_subsequent:
                animation_sets.append(
                    (
                        "subsequent",
                        subsequent_animation_samples,
                        "Subsequent-sample animation frame",
                    )
                )

            for set_key, set_samples, title_prefix in animation_sets:
                if not set_samples:
                    print(f"[anim] Skipped '{set_key}' set: no samples available.")
                    continue
                frame_count, mp4_out, gif_out = _export_animation_set(
                    set_key=set_key,
                    set_samples=set_samples,
                    title_prefix=title_prefix,
                    animations_dir=animations_dir,
                    fps_values=fps_values,
                    vmax=vmax_common,
                    plotter=slot_count_plotter,
                    cfg=cfg,
                    effective_backend=str(animation_render_backend_effective),
                )
                if set_key == "random":
                    animation_random_frame_count = int(frame_count)
                else:
                    animation_subsequent_frame_count = int(frame_count)
                animation_mp4_outputs.extend(mp4_out)
                animation_gif_outputs.extend(gif_out)
                if frame_count == 0:
                    print(f"[anim] Skipped '{set_key}' exports: no frames were rendered.")
    if run_dir is not None and cfg.save_results_json:
        config_payload = asdict(cfg)
        config_payload.update(
            {
                "n_skycells": int(n_skycells),
                "effective_enable_random_plot_samples": bool(enable_random_plots),
                "effective_enable_average_plot": bool(enable_average_plot),
                "effective_enable_animation_random_set": bool(enable_animation_random),
                "effective_enable_animation_subsequent_set": bool(enable_animation_subsequent),
                "effective_sky_mapper_backend": str(mapper_backend),
                "effective_animation_render_backend": animation_render_backend_effective,
                # Backward-compatible aliases used by earlier summaries.
                "random_timestep_plots_requested": int(cfg.random_timestep_plots),
                "plot_engine": str(cfg.skycell_vis_engine),
                "plot_projection": str(cfg.skycell_vis_2d_projection),
            }
        )
        payload = {
            "storage_filename": str(file_path),
            "config": config_payload,
            "run_diagnostics": {
                "elapsed_s": float(elapsed),
                "processed_slots": int(processed_slots),
                "stop_reason": str(stop_reason),
                "stopped_early": bool(stop_reason != "completed_full_scan"),
                "full_pass_required": bool(full_pass_required),
                "sky_mapper_backend": str(mapper_backend),
                "slots_with_valid_points": int(slots_with_valid_points),
                "slots_with_mappable_satellites": int(slots_with_mappable_satellites),
                "average_slots_used": int(average_slots_used),
                "average_slots_target": (
                    None if average_max_slots <= 0 else int(average_max_slots)
                ),
                "sampling_seen_plots": int(sampling_seen_plots),
                "sampling_seen_random_animation": int(sampling_seen_random_animation),
                "valid_points_total": int(total_valid_points),
                "unique_satellites_total": int(total_unique_satellites),
                "mappable_satellites_total": int(total_mappable_satellites),
                "max_random_cell_count": float(max_random_cell_count),
                "max_random_animation_cell_count": float(max_random_animation_cell_count),
                "max_subsequent_animation_cell_count": float(max_subsequent_animation_cell_count),
                "max_average_cell_count": float(max_average_cell_count),
                "vmax_common": float(vmax_common),
                "plot_random_png_count": int(random_plot_png_count),
                "plot_random_jpg_count": int(random_plot_jpg_count),
                "plot_average_png_count": int(average_plot_png_count),
                "plot_average_jpg_count": int(average_plot_jpg_count),
                "animation_random_frame_count": int(animation_random_frame_count),
                "animation_subsequent_frame_count": int(animation_subsequent_frame_count),
                "animations_dir": str(animations_dir) if animations_dir is not None else None,
                "animation_mp4_outputs": list(animation_mp4_outputs),
                "animation_gif_outputs": list(animation_gif_outputs),
                "animation_render_backend_requested": animation_render_backend_requested,
                "animation_render_backend_effective": animation_render_backend_effective,
                "animation_render_backend_reason": animation_render_backend_reason,
                "animation_used_direct_streaming": bool(animation_used_direct_streaming),
                "animation_kept_frame_pngs": bool(animation_kept_frame_pngs),
            },
            "samples": [
                {
                    "plot_index": int(i + 1),
                    "iter_name": s.iter_name,
                    "slot_local_idx": int(s.slot_local_idx),
                    "slot_global_idx": int(s.slot_global_idx),
                    "time_mjd": None if s.time_mjd is None else float(s.time_mjd),
                    "n_satellites": int(s.n_satellites),
                    "n_active_skycells": int(s.n_active_skycells),
                }
                for i, s in enumerate(random_plot_samples)
            ],
        }
        with (run_dir / "results_summary.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if run_dir is not None and cfg.save_results_npz:
        selected_counts = (
            np.stack([s.counts for s in random_plot_samples], axis=0).astype(np.int32, copy=False)
            if random_plot_samples
            else np.empty((0, n_skycells), dtype=np.int32)
        )
        selected_global_slots = np.asarray([s.slot_global_idx for s in random_plot_samples], dtype=np.int64)
        selected_local_slots = np.asarray([s.slot_local_idx for s in random_plot_samples], dtype=np.int64)
        selected_times_mjd = np.asarray(
            [np.nan if s.time_mjd is None else float(s.time_mjd) for s in random_plot_samples],
            dtype=np.float64,
        )
        np.savez_compressed(
            run_dir / "results_curves.npz",
            average_counts=np.asarray(average_counts, dtype=np.float64),
            average_slots_used=np.asarray([int(average_slots_used)], dtype=np.int64),
            processed_slots=np.asarray([int(processed_slots)], dtype=np.int64),
            selected_counts=selected_counts,
            selected_global_slots=selected_global_slots,
            selected_local_slots=selected_local_slots,
            selected_times_mjd=selected_times_mjd,
        )

    print("\nRun completed.")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Processed slots: {processed_slots:,}")
    print(f"Stop reason: {stop_reason}")
    print(f"Slots with valid points: {slots_with_valid_points:,}")
    print(f"Slots with mappable satellites: {slots_with_mappable_satellites:,}")
    if total_unique_satellites > 0:
        mapped_pct = 100.0 * float(total_mappable_satellites) / float(total_unique_satellites)
        print(
            f"Mapping coverage after normalization: "
            f"{total_mappable_satellites:,}/{total_unique_satellites:,} ({mapped_pct:.3f}%)"
        )
    if enable_average_plot:
        avg_target_summary = "all-eligible" if average_max_slots <= 0 else str(average_max_slots)
        print(
            f"Average slots used: {average_slots_used:,}/{avg_target_summary} "
            f"(stride={average_slot_stride}, nonempty_only={cfg.average_sample_nonempty_only})"
        )
    else:
        print("Average map: disabled")

    if enable_random_plots:
        print(
            f"Random plot samples selected: "
            f"{len(random_plot_samples):,}/{random_plot_target_size}"
        )
    else:
        print("Random plot samples: disabled")

    if enable_animation_random:
        print(
            f"Random animation samples selected: "
            f"{len(random_animation_samples):,}/{random_animation_target_size}"
        )
    else:
        print("Random animation samples: disabled")

    if enable_animation_subsequent:
        print(
            f"Subsequent animation samples selected: "
            f"{len(subsequent_animation_samples):,}/{subsequent_animation_target_size}"
        )
    else:
        print("Subsequent animation samples: disabled")

    print(f"Max cell count (random): {max_random_cell_count:.3f}")
    print(f"Max cell count (average): {max_average_cell_count:.3f}")
    if cfg.save_plots_png:
        print(f"Random plot PNGs saved: {random_plot_png_count:,}")
        print(f"Average plot PNGs saved: {average_plot_png_count:,}")
    if cfg.save_plots_jpg:
        print(f"Random plot JPGs saved: {random_plot_jpg_count:,}")
        print(f"Average plot JPGs saved: {average_plot_jpg_count:,}")
    if animations_dir is not None:
        print(f"Animation random frames: {animation_random_frame_count:,}")
        print(f"Animation subsequent frames: {animation_subsequent_frame_count:,}")
        if animation_render_backend_effective is not None:
            print(
                "Animation render backend: "
                f"requested={animation_render_backend_requested}, "
                f"effective={animation_render_backend_effective}, "
                f"reason={animation_render_backend_reason}"
            )
        print(
            f"ffmpeg x264 settings: preset={cfg.ffmpeg_preset}, crf={cfg.ffmpeg_crf}, "
            f"threads={cfg.ffmpeg_threads}"
        )
        print(f"MP4 animations saved: {len(animation_mp4_outputs):,}")
        print(f"GIF animations saved: {len(animation_gif_outputs):,}")
        print(f"Animations folder: {animations_dir}")
    if run_dir is not None:
        print(f"Saved outputs to: {run_dir}")

    return {
        "storage_filename": str(file_path),
        "config": asdict(cfg),
        "run_dir": None if run_dir is None else str(run_dir),
        "sky_mapper_backend": str(mapper_backend),
        "processed_slots": int(processed_slots),
        "average_slots_used": int(average_slots_used),
        "stop_reason": str(stop_reason),
        "stopped_early": bool(stop_reason != "completed_full_scan"),
        "random_plot_samples_selected": int(len(random_plot_samples)),
        "random_animation_samples_selected": int(len(random_animation_samples)),
        "subsequent_animation_samples_selected": int(len(subsequent_animation_samples)),
        "animation_random_frame_count": int(animation_random_frame_count),
        "animation_subsequent_frame_count": int(animation_subsequent_frame_count),
        "animation_mp4_outputs": list(animation_mp4_outputs),
        "animation_gif_outputs": list(animation_gif_outputs),
        "animation_render_backend_requested": animation_render_backend_requested,
        "animation_render_backend_effective": animation_render_backend_effective,
        "animation_render_backend_reason": animation_render_backend_reason,
        "animation_used_direct_streaming": bool(animation_used_direct_streaming),
        "animation_kept_frame_pngs": bool(animation_kept_frame_pngs),
        "summary_json_path": (
            None
            if (run_dir is None or (not cfg.save_results_json))
            else str(run_dir / "results_summary.json")
        ),
        "results_npz_path": (
            None
            if (run_dir is None or (not cfg.save_results_npz))
            else str(run_dir / "results_curves.npz")
        ),
    }


def _coerce_angle_array_deg_visualise(value: Any, *, name: str) -> np.ndarray:
    quantity = u.Quantity(value, copy=False)
    if quantity.unit == u.dimensionless_unscaled:
        quantity = quantity * u.deg
    else:
        quantity = quantity.to(u.deg)
    arr = np.asarray(quantity.value, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after coercion.")
    return arr


def _coerce_scalar_angle_deg_visualise(value: Any, *, name: str) -> float:
    quantity = u.Quantity(value, copy=False)
    if quantity.unit == u.dimensionless_unscaled:
        quantity = quantity * u.deg
    else:
        quantity = quantity.to(u.deg)
    arr = np.asarray(quantity.value, dtype=np.float64).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"{name} must be a scalar angle.")
    return float(arr[0])


def _great_circle_distance_km_visualise(
    longitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
    *,
    ref_lon_deg: float,
    ref_lat_deg: float,
) -> np.ndarray:
    lon_rad = np.deg2rad(np.asarray(longitudes_deg, dtype=np.float64))
    lat_rad = np.deg2rad(np.asarray(latitudes_deg, dtype=np.float64))
    ref_lon_rad = np.deg2rad(float(ref_lon_deg))
    ref_lat_rad = np.deg2rad(float(ref_lat_deg))
    delta_lon = (lon_rad - ref_lon_rad + np.pi) % (2.0 * np.pi) - np.pi
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_ref_lat = np.sin(ref_lat_rad)
    cos_ref_lat = np.cos(ref_lat_rad)
    cos_delta = sin_ref_lat * sin_lat + cos_ref_lat * cos_lat * np.cos(delta_lon)
    return np.arccos(np.clip(cos_delta, -1.0, 1.0)) * float(R_earth.to_value(u.km))


def _frequency_reuse_slot_colors(reuse_factor: int) -> list[str]:
    reuse_factor_i = max(1, int(reuse_factor))
    cmap = plt.get_cmap("tab20")
    return [
        cmap(float(idx) / float(max(1, reuse_factor_i - 1)))
        for idx in range(reuse_factor_i)
    ]


def _normalise_hexgrid_orientation_name(orientation_name: Any) -> str:
    """Return a validated regular-hex orientation name."""
    orientation_use = str(orientation_name or "pointy").strip().lower()
    if orientation_use not in {"pointy", "flat"}:
        raise ValueError("orientation_name must be 'pointy' or 'flat'.")
    return orientation_use


def _hexgrid_voronoi_polygons_lonlat(
    pre_lon_deg: np.ndarray,
    pre_lat_deg: np.ndarray,
    *,
    active_lon_deg: np.ndarray,
    active_lat_deg: np.ndarray,
    ref_lon_deg: float,
    ref_lat_deg: float,
    point_spacing_km: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]] | None:
    """Build gap-free, overlap-free hex-like polygons via Spherical Voronoi.

    Uses ``scipy.spatial.SphericalVoronoi`` to tessellate cell positions
    directly on the unit sphere.  This avoids any tangent-plane projection
    distortion — cells look correctly hexagonal at all latitudes (equator,
    mid-latitudes, and near the poles).

    On a sphere every Voronoi region is bounded, so no ghost points are
    needed.  Adjacent cells share exactly the same great-circle edge
    vertices → the tiling is provably seamless (zero gaps, zero overlaps).

    Returns ``None`` when the computation fails so callers can fall back
    to the regular-hex centre-offset path.
    """
    from scipy.spatial import SphericalVoronoi, cKDTree
    from scepter import earthgrid as earthgrid_mod

    if int(pre_lon_deg.size) < 4:
        return None

    spacing_value = point_spacing_km
    if spacing_value is None:
        spacing_value = earthgrid_mod._estimate_local_hexgrid_spacing_km(
            pre_lon_deg, pre_lat_deg,
            ref_lon_deg=float(ref_lon_deg),
            ref_lat_deg=float(ref_lat_deg),
        )
    spacing_value = float(spacing_value) if spacing_value else 90.0
    if not np.isfinite(spacing_value) or spacing_value <= 0.0:
        spacing_value = 90.0

    hex_radius = float(spacing_value) / float(np.sqrt(3.0))

    try:
        # --- Convert lon/lat → 3-D Cartesian on unit sphere ----------------
        lon_rad = np.deg2rad(np.asarray(pre_lon_deg, dtype=np.float64).ravel())
        lat_rad = np.deg2rad(np.asarray(pre_lat_deg, dtype=np.float64).ravel())
        n_real = int(lon_rad.size)

        xyz = np.column_stack([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ])

        # --- Add 2 phantom layers of ghost cells around the boundary so
        # that edge Voronoi regions have proper neighbours and look
        # hexagonal.  For each boundary point we push it outward from
        # the centroid by 1× and 2× the angular cell spacing.
        R_earth_km = 6371.0
        spacing_rad = spacing_value / R_earth_km  # angular spacing

        centroid_3d = xyz.mean(axis=0)
        c_norm = np.linalg.norm(centroid_3d)
        centroid_3d = centroid_3d / max(c_norm, 1e-12)

        # --- Ghost cells via neighbour-mirroring. -------------------------
        # For each boundary cell (fewer than 6 close neighbours), mirror
        # every existing neighbour through the cell to place a ghost
        # exactly where a real hex lattice neighbour would sit.  This
        # respects the actual lattice geometry instead of guessing
        # outward directions.
        #
        # Running 3 iterations builds 3 ghost layers so that the real
        # edge cells are fully surrounded and Voronoi distortion only
        # affects the outermost ghost layer (which is never rendered).
        chord_1x = 2.0 * np.sin(spacing_rad * 0.5)
        nn_threshold = chord_1x * 1.5
        dedup_threshold = chord_1x * 0.4

        all_pts = xyz.copy()  # grows each iteration

        for _ghost_iter in range(3):
            n_cur = int(all_pts.shape[0])
            tree_cur = cKDTree(all_pts)
            k_nn = min(7, n_cur)
            nn_dists, nn_idx = tree_cur.query(all_pts, k=k_nn)

            new_ghosts: list[np.ndarray] = []
            for ci in range(n_cur):
                close_count = int(np.sum(nn_dists[ci, 1:] < nn_threshold))
                if close_count >= 6:
                    continue

                cell_pt = all_pts[ci]
                close_mask = nn_dists[ci, 1:] < nn_threshold
                if not np.any(close_mask):
                    continue
                close_indices = nn_idx[ci, 1:][close_mask]

                # Mirror each existing neighbour through the cell centre
                # on the unit sphere.  If N is a neighbour, the ghost G
                # is placed at the antipodal point of N relative to the
                # cell: G = 2*(cell projected) - N, re-normalised.
                for ni in close_indices:
                    nb = all_pts[ni]
                    # Spherical reflection: rotate N by 2× the angle
                    # from N to cell, along the great circle N→cell.
                    # Equivalent to: G = 2*dot(cell,N)*cell - N on the
                    # unit sphere (Householder-style reflection), then
                    # re-normalise.
                    g = 2.0 * float(np.dot(cell_pt, nb)) * cell_pt - nb
                    g_norm = np.linalg.norm(g)
                    if g_norm < 1e-12:
                        continue
                    g /= g_norm
                    new_ghosts.append(g)

            if not new_ghosts:
                break

            candidates = np.array(new_ghosts, dtype=np.float64)
            # Remove candidates too close to existing points
            c_dists, _ = tree_cur.query(candidates, k=1)
            keep_mask = c_dists > dedup_threshold
            candidates = candidates[keep_mask]
            if len(candidates) == 0:
                break

            # De-duplicate among candidates
            if len(candidates) > 1:
                c_tree = cKDTree(candidates)
                pairs = c_tree.query_pairs(r=dedup_threshold)
                remove = set()
                for a, b in pairs:
                    remove.add(max(a, b))
                if remove:
                    keep_idx = np.array(sorted(set(range(len(candidates))) - remove))
                    candidates = candidates[keep_idx]

            all_pts = np.vstack([all_pts, candidates])

        ghosts = all_pts[n_real:]  # everything beyond the original points

        augmented = np.vstack([xyz, ghosts])
        # Re-normalise to unit sphere (numerical safety)
        norms_aug = np.linalg.norm(augmented, axis=1, keepdims=True)
        augmented = augmented / np.maximum(norms_aug, 1e-12)

        sv = SphericalVoronoi(
            augmented,
            radius=1.0,
            center=np.array([0.0, 0.0, 0.0]),
        )
        sv.sort_vertices_of_regions()

        # Pre-compute vertex lon/lat once
        sv_verts = sv.vertices
        sv_vlat_deg = np.rad2deg(np.arcsin(np.clip(sv_verts[:, 2], -1.0, 1.0)))
        sv_vlon_deg = np.rad2deg(np.arctan2(sv_verts[:, 1], sv_verts[:, 0]))

        def _extract_polygons_for(lon_arr, lat_arr):
            """Extract polygon lon/lat vertices for each cell centre."""
            if int(lon_arr.size) == 0:
                return []
            lon_r = np.deg2rad(np.asarray(lon_arr, dtype=np.float64).ravel())
            lat_r = np.deg2rad(np.asarray(lat_arr, dtype=np.float64).ravel())
            xyz_q = np.column_stack([
                np.cos(lat_r) * np.cos(lon_r),
                np.cos(lat_r) * np.sin(lon_r),
                np.sin(lat_r),
            ])
            # Map each query point to the nearest generator in the augmented set
            tree_aug = cKDTree(augmented)
            _, indices = tree_aug.query(xyz_q, k=1)

            polygons: list[np.ndarray] = []
            for i in range(int(lon_arr.size)):
                pt_idx = int(indices[i])
                region = sv.regions[pt_idx]

                if not region or pt_idx >= n_real:
                    polygons.append(np.empty((0, 2), dtype=np.float64))
                    continue

                vlon = sv_vlon_deg[region]
                vlat = sv_vlat_deg[region]
                polygons.append(
                    np.column_stack([vlon, vlat]).astype(np.float64, copy=False)
                )
            return polygons

        pre_polygons = _extract_polygons_for(pre_lon_deg, pre_lat_deg)
        active_polygons = _extract_polygons_for(active_lon_deg, active_lat_deg)

    except Exception:
        return None

    return pre_polygons, active_polygons, {
        "point_spacing_km_used": float(spacing_value),
        "hex_radius_km_used": float(hex_radius),
        "hex_orientation_used": "voronoi",
        "hex_fit_residual_km2": 0.0,
        "hex_geometry_mode": "voronoi",
        "hex_anchor_pre_ras_index": None,
        "hex_center_render_residual_km_max": 0.0,
        "hex_center_render_residual_km_median": 0.0,
    }


def _hexgrid_regular_polygon_offsets_km(
    *,
    point_spacing_km: float,
    orientation_name: Any,
) -> tuple[np.ndarray, float]:
    """Return regular-hex vertex offsets for one local tangent-plane cell."""
    spacing_value = float(point_spacing_km)
    if not np.isfinite(spacing_value) or spacing_value <= 0.0:
        raise ValueError("point_spacing_km must be finite and > 0.")
    orientation_use = _normalise_hexgrid_orientation_name(orientation_name)
    hex_radius_km = float(spacing_value) / float(np.sqrt(3.0))
    base_angle_deg = 30.0 if orientation_use == "pointy" else 0.0
    angle_rad = np.deg2rad(base_angle_deg + 60.0 * np.arange(6, dtype=np.float64))
    offsets_km = np.column_stack(
        [
            hex_radius_km * np.cos(angle_rad),
            hex_radius_km * np.sin(angle_rad),
        ]
    ).astype(np.float64, copy=False)
    return offsets_km, float(hex_radius_km)


def _hexgrid_center_xy_km_from_axial(
    axial_q: Any,
    axial_r: Any,
    *,
    point_spacing_km: float,
    orientation_name: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert axial hex-lattice coordinates into local tangent-plane centres."""
    q_arr = np.asarray(axial_q, dtype=np.float64).reshape(-1)
    r_arr = np.asarray(axial_r, dtype=np.float64).reshape(-1)
    if int(q_arr.size) != int(r_arr.size):
        raise ValueError("axial_q and axial_r must have the same size.")

    spacing_value = float(point_spacing_km)
    if not np.isfinite(spacing_value) or spacing_value <= 0.0:
        raise ValueError("point_spacing_km must be finite and > 0.")
    orientation_use = _normalise_hexgrid_orientation_name(orientation_name)
    sqrt3 = float(np.sqrt(3.0))
    if orientation_use == "pointy":
        x_km = spacing_value * (q_arr + 0.5 * r_arr)
        y_km = spacing_value * (sqrt3 / 2.0) * r_arr
    else:
        x_km = spacing_value * (sqrt3 / 2.0) * q_arr
        y_km = spacing_value * (r_arr + 0.5 * q_arr)
    return x_km.astype(np.float64, copy=False), y_km.astype(np.float64, copy=False)


def _resolve_hexgrid_polygon_offsets_km(
    reference_longitudes_deg: np.ndarray,
    reference_latitudes_deg: np.ndarray,
    *,
    ref_lon_deg: float,
    ref_lat_deg: float,
    point_spacing_km: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return local tangent-plane vertex offsets for regular hex preview cells."""
    from scepter import earthgrid as earthgrid_mod

    lon_arr = np.asarray(reference_longitudes_deg, dtype=np.float64).reshape(-1)
    lat_arr = np.asarray(reference_latitudes_deg, dtype=np.float64).reshape(-1)
    if int(lon_arr.size) != int(lat_arr.size):
        raise ValueError("Hexgrid polygon reference longitude/latitude arrays must match.")

    spacing_value = None if point_spacing_km is None else float(point_spacing_km)
    if spacing_value is None:
        spacing_value = earthgrid_mod._estimate_local_hexgrid_spacing_km(
            lon_arr,
            lat_arr,
            ref_lon_deg=float(ref_lon_deg),
            ref_lat_deg=float(ref_lat_deg),
        )
    if not np.isfinite(spacing_value) or spacing_value <= 0.0:
        raise ValueError("point_spacing_km must be finite and > 0 when drawing hexgrid polygons.")

    orientation_name = "pointy"
    fit_residual_km2 = 0.0
    if int(lon_arr.size) >= 2:
        x_km, y_km = earthgrid_mod._local_tangent_plane_xy_km(
            lon_arr,
            lat_arr,
            ref_lon_deg=float(ref_lon_deg),
            ref_lat_deg=float(ref_lat_deg),
        )
        _q, _r, orientation_name, fit_residual_km2 = earthgrid_mod._infer_hexgrid_axial_coordinates(
            x_km,
            y_km,
            point_spacing_km=float(spacing_value),
        )
    offsets_km, hex_radius_km = _hexgrid_regular_polygon_offsets_km(
        point_spacing_km=float(spacing_value),
        orientation_name=orientation_name,
    )
    return offsets_km, {
        "point_spacing_km_used": float(spacing_value),
        "hex_radius_km_used": float(hex_radius_km),
        "hex_orientation_used": str(orientation_name),
        "hex_fit_residual_km2": float(fit_residual_km2),
        "hex_geometry_mode": "center_offsets",
        "hex_anchor_pre_ras_index": None,
        "hex_center_render_residual_km_max": 0.0,
        "hex_center_render_residual_km_median": 0.0,
    }


def _hexgrid_polygons_lonlat_from_offsets(
    longitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
    *,
    offsets_km: np.ndarray,
    ref_lon_deg: float,
    ref_lat_deg: float,
) -> list[np.ndarray]:
    """Convert lon/lat centres into six-vertex lon/lat regular-hex polygons."""
    from scepter import earthgrid as earthgrid_mod

    lon_arr = np.asarray(longitudes_deg, dtype=np.float64).reshape(-1)
    lat_arr = np.asarray(latitudes_deg, dtype=np.float64).reshape(-1)
    if int(lon_arr.size) != int(lat_arr.size):
        raise ValueError("Hexgrid polygon longitude/latitude arrays must match.")
    if int(lon_arr.size) == 0:
        return []

    x_km, y_km = earthgrid_mod._local_tangent_plane_xy_km(
        lon_arr,
        lat_arr,
        ref_lon_deg=float(ref_lon_deg),
        ref_lat_deg=float(ref_lat_deg),
    )
    offsets_arr = np.asarray(offsets_km, dtype=np.float64)
    vertex_x_km = x_km[:, None] + offsets_arr[None, :, 0]
    vertex_y_km = y_km[:, None] + offsets_arr[None, :, 1]
    vertex_lon_deg, vertex_lat_deg = earthgrid_mod._local_tangent_plane_lonlat_from_xy_km(
        vertex_x_km,
        vertex_y_km,
        ref_lon_deg=float(ref_lon_deg),
        ref_lat_deg=float(ref_lat_deg),
    )
    return [
        np.column_stack([vertex_lon_deg[idx], vertex_lat_deg[idx]]).astype(np.float64, copy=False)
        for idx in range(int(lon_arr.size))
    ]


def _resolve_shared_hexgrid_polygons_lonlat(
    pre_ras_longitudes_deg: np.ndarray,
    pre_ras_latitudes_deg: np.ndarray,
    *,
    active_longitudes_deg: np.ndarray,
    active_latitudes_deg: np.ndarray,
    hex_lattice: Mapping[str, Any] | None,
    point_spacing_km: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]] | None:
    """Build seamless hex polygons from one shared axial lattice when possible."""
    from scepter import earthgrid as earthgrid_mod

    if not isinstance(hex_lattice, Mapping):
        return None

    pre_lon_arr = np.asarray(pre_ras_longitudes_deg, dtype=np.float64).reshape(-1)
    pre_lat_arr = np.asarray(pre_ras_latitudes_deg, dtype=np.float64).reshape(-1)
    active_lon_arr = np.asarray(active_longitudes_deg, dtype=np.float64).reshape(-1)
    active_lat_arr = np.asarray(active_latitudes_deg, dtype=np.float64).reshape(-1)
    if int(pre_lon_arr.size) != int(pre_lat_arr.size):
        raise ValueError("pre-RAS longitude/latitude arrays must have the same size.")
    if int(active_lon_arr.size) != int(active_lat_arr.size):
        raise ValueError("active longitude/latitude arrays must have the same size.")
    if int(pre_lon_arr.size) < 1:
        return None

    try:
        anchor_pre_ras_index = int(hex_lattice.get("anchor_pre_ras_index", -1))
    except (TypeError, ValueError):
        return None
    if anchor_pre_ras_index < 0 or anchor_pre_ras_index >= int(pre_lon_arr.size):
        return None

    try:
        orientation_name = _normalise_hexgrid_orientation_name(
            hex_lattice.get("orientation_used")
        )
    except ValueError:
        return None

    spacing_candidate = (
        point_spacing_km
        if point_spacing_km is not None
        else hex_lattice.get("point_spacing_km_used")
    )
    if spacing_candidate is None:
        return None
    try:
        spacing_value = float(spacing_candidate)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(spacing_value) or spacing_value <= 0.0:
        return None

    def _coerce_axial_array(key: str, *, expected_size: int) -> np.ndarray | None:
        values = hex_lattice.get(key)
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=np.int32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if int(arr.size) != int(expected_size):
            return None
        return arr.astype(np.int32, copy=False)

    axial_q_pre_ras = _coerce_axial_array(
        "axial_q_pre_ras",
        expected_size=int(pre_lon_arr.size),
    )
    axial_r_pre_ras = _coerce_axial_array(
        "axial_r_pre_ras",
        expected_size=int(pre_lon_arr.size),
    )
    axial_q_active = _coerce_axial_array(
        "axial_q_active",
        expected_size=int(active_lon_arr.size),
    )
    axial_r_active = _coerce_axial_array(
        "axial_r_active",
        expected_size=int(active_lon_arr.size),
    )
    if (
        axial_q_pre_ras is None
        or axial_r_pre_ras is None
        or axial_q_active is None
        or axial_r_active is None
    ):
        return None

    anchor_lon_deg = float(pre_lon_arr[anchor_pre_ras_index])
    anchor_lat_deg = float(pre_lat_arr[anchor_pre_ras_index])

    offsets_km, hex_radius_km = _hexgrid_regular_polygon_offsets_km(
        point_spacing_km=spacing_value,
        orientation_name=orientation_name,
    )

    # Stamp regular hexagons at the ACTUAL cell centres (not reconstructed
    # lattice centres).  The axial coordinates are still used for reuse-slot
    # assignment, but using real positions avoids the gap/overlap artifacts
    # caused by snapping to an imperfect lattice fit.
    pre_polygons = _hexgrid_polygons_lonlat_from_offsets(
        pre_lon_arr,
        pre_lat_arr,
        offsets_km=offsets_km,
        ref_lon_deg=anchor_lon_deg,
        ref_lat_deg=anchor_lat_deg,
    )
    active_polygons = _hexgrid_polygons_lonlat_from_offsets(
        active_lon_arr,
        active_lat_arr,
        offsets_km=offsets_km,
        ref_lon_deg=anchor_lon_deg,
        ref_lat_deg=anchor_lat_deg,
    )

    residual_arr = np.zeros(1, dtype=np.float64)

    return pre_polygons, active_polygons, {
        "point_spacing_km_used": float(spacing_value),
        "hex_radius_km_used": float(hex_radius_km),
        "hex_orientation_used": str(orientation_name),
        "hex_fit_residual_km2": float(hex_lattice.get("fit_residual_km2", 0.0) or 0.0),
        "hex_geometry_mode": "shared_lattice",
        "hex_anchor_pre_ras_index": int(anchor_pre_ras_index),
        "hex_center_render_residual_km_max": float(np.max(residual_arr)),
        "hex_center_render_residual_km_median": float(np.median(residual_arr)),
    }


def _infer_and_build_lattice_polygons(
    pre_lon_deg: np.ndarray,
    pre_lat_deg: np.ndarray,
    *,
    active_lon_deg: np.ndarray,
    active_lat_deg: np.ndarray,
    ref_lon_deg: float,
    ref_lat_deg: float,
    point_spacing_km: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Build regular hex polygons at actual cell centres.

    Stamps a correctly-sized hexagon at each real cell position using the
    measured neighbour spacing.  This preserves the hex look while placing
    cells exactly where they belong — any tiny residual gaps from the
    icosahedral grid's non-uniform spacing are virtually invisible at
    normal zoom levels.
    """
    from scepter import earthgrid as earthgrid_mod

    geometry_lon = pre_lon_deg if int(pre_lon_deg.size) >= 2 else active_lon_deg
    geometry_lat = pre_lat_deg if int(pre_lat_deg.size) >= 2 else active_lat_deg

    spacing_value = point_spacing_km
    if spacing_value is None:
        spacing_value = earthgrid_mod._estimate_local_hexgrid_spacing_km(
            geometry_lon,
            geometry_lat,
            ref_lon_deg=float(ref_lon_deg),
            ref_lat_deg=float(ref_lat_deg),
        )
    spacing_value = float(spacing_value)
    if not np.isfinite(spacing_value) or spacing_value <= 0.0:
        spacing_value = 1.0

    # Infer orientation from the cell positions
    x_km_ref, y_km_ref = earthgrid_mod._local_tangent_plane_xy_km(
        geometry_lon, geometry_lat,
        ref_lon_deg=float(ref_lon_deg),
        ref_lat_deg=float(ref_lat_deg),
    )
    _, _, orientation_used, fit_residual = (
        earthgrid_mod._infer_hexgrid_axial_coordinates(
            x_km_ref, y_km_ref, point_spacing_km=spacing_value,
        )
    )

    offsets_km, hex_radius_km = _hexgrid_regular_polygon_offsets_km(
        point_spacing_km=spacing_value,
        orientation_name=orientation_used,
    )

    pre_polygons = _hexgrid_polygons_lonlat_from_offsets(
        pre_lon_deg, pre_lat_deg,
        offsets_km=offsets_km,
        ref_lon_deg=float(ref_lon_deg),
        ref_lat_deg=float(ref_lat_deg),
    )
    active_polygons = _hexgrid_polygons_lonlat_from_offsets(
        active_lon_deg, active_lat_deg,
        offsets_km=offsets_km,
        ref_lon_deg=float(ref_lon_deg),
        ref_lat_deg=float(ref_lat_deg),
    )

    return pre_polygons, active_polygons, {
        "point_spacing_km_used": spacing_value,
        "hex_radius_km_used": float(hex_radius_km),
        "hex_orientation_used": str(orientation_used),
        "hex_fit_residual_km2": float(fit_residual),
        "hex_geometry_mode": "center_offsets",
        "hex_anchor_pre_ras_index": None,
        "hex_center_render_residual_km_max": 0.0,
        "hex_center_render_residual_km_median": 0.0,
    }


def plot_cell_status_map(
    pre_ras_cell_longitudes: Any,
    pre_ras_cell_latitudes: Any,
    *,
    active_cell_longitudes: Any,
    active_cell_latitudes: Any,
    switched_off_mask: Any,
    boresight_affected_cell_ids: Any | None = None,
    active_reuse_slot_ids: Any | None = None,
    reuse_factor: int | None = None,
    anchor_active_index: int | None = None,
    ras_longitude: Any,
    ras_latitude: Any,
    backend: str = "auto",
    map_style: str = "clean",
    appearance_variant: str | None = None,
    point_spacing_km: float | None = None,
    hex_lattice: Mapping[str, Any] | None = None,
    radius_km: float | None = None,
    save_path: str | Path | None = None,
    return_info: bool = False,
):
    """
    Plot a static Earth-grid cell-status map around the RAS station.

    Parameters
    ----------
    pre_ras_cell_longitudes, pre_ras_cell_latitudes : array-like
        One-dimensional longitude/latitude arrays, in degrees or angle
        quantities, for the geography-kept cell centers before the optional
        RAS exclusion stage. Cells removed by geography masking should be
        omitted from this axis entirely.
    active_cell_longitudes, active_cell_latitudes : array-like
        One-dimensional longitude/latitude arrays for the final active-cell
        centers after the RAS exclusion stage.
    switched_off_mask : array-like of bool
        Boolean mask on the pre-RAS cell axis. ``True`` marks cells that were
        removed by the RAS exclusion stage. Geography-masked cells are not
        represented here and are not plotted.
    boresight_affected_cell_ids : array-like of int or None, optional
        Active-cell indices affected by static Theta_2 boresight redirection.
        The ids refer to the final active-cell axis.
    active_reuse_slot_ids : array-like of int or None, optional
        Optional ACTIVE-axis reuse-slot ids. When provided together with
        ``reuse_factor > 1``, active cells are colored by reuse slot instead of
        the legacy status-only palette.
    reuse_factor : int or None, optional
        Frequency-reuse factor used only for legend/title context and reuse-slot
        validation. ``None`` or ``1`` preserves the status-only color scheme.
    anchor_active_index : int or None, optional
        ACTIVE-axis index of the reuse anchor cell. Retained for compatibility
        and included in the returned ``info`` payload, but not drawn on the
        geographic preview.
    ras_longitude, ras_latitude : scalar angle-like
        RAS-station geodetic longitude and latitude.
    backend : {"auto", "cartopy", "matplotlib"}, optional
        Rendering backend. ``"auto"`` prefers Cartopy when importable and falls
        back to plain Matplotlib otherwise.
    point_spacing_km : float or None, optional
        Prepared-grid nearest-neighbour spacing, in kilometres, used to draw
        true data-space hex polygons. When omitted, the spacing is estimated
        from the plotted cell centres.
    hex_lattice : Mapping[str, Any] or None, optional
        Optional shared axial-lattice metadata, typically from
        :func:`scepter.earthgrid.resolve_frequency_reuse_slots`. When valid,
        the preview reconstructs cell polygons from the shared PRE_RAS/ACTIVE
        lattice around the anchor cell to keep adjacent hexes seamless under
        zoom. Missing or inconsistent metadata falls back to the legacy
        centre-offset polygon path without changing the plotted cell classes.
    radius_km : float or None, optional
        Optional clip radius around the RAS station. When omitted the plot
        auto-fits all plotted cells with padding.
    save_path : str, pathlib.Path, or None, optional
        Optional output path for ``fig.savefig``.
    return_info : bool, optional
        When ``True``, return ``(fig, info)`` where ``info`` contains backend
        and plotted-count diagnostics.

    Raises
    ------
    ValueError
        Raised when longitude/latitude inputs do not share a common one-
        dimensional axis, when ``switched_off_mask`` does not match the
        pre-RAS cell axis, when ``boresight_affected_cell_ids`` do not
        reference the active-cell axis, or when ``backend`` / ``radius_km``
        inputs are invalid.
    RuntimeError
        Raised when ``backend="cartopy"`` is requested but Cartopy is not
        importable in the active environment.

    Returns
    -------
    matplotlib.figure.Figure or tuple[matplotlib.figure.Figure, dict[str, Any]]
        Figure only by default, or ``(figure, info)`` when
        ``return_info=True``.

    Notes
    -----
    The plotted classes are mutually exclusive by construction:

    - ``switched_off`` is evaluated on the pre-RAS axis and means
      RAS-excluded only.
    - ``boresight_affected_active`` and ``normal_active`` are evaluated on the
      final active-cell axis after the RAS exclusion stage.
    - ``hex_lattice`` affects only preview geometry; simulation-facing cell
      centres, ACTIVE/PRE_RAS ids, and reuse-slot inputs remain unchanged.
    - When ``backend="auto"``, Cartopy is preferred when available and the
      helper falls back to plain Matplotlib longitude/latitude axes otherwise.
    """
    pre_lon_deg = _coerce_angle_array_deg_visualise(
        pre_ras_cell_longitudes,
        name="pre_ras_cell_longitudes",
    )
    pre_lat_deg = _coerce_angle_array_deg_visualise(
        pre_ras_cell_latitudes,
        name="pre_ras_cell_latitudes",
    )
    active_lon_deg = _coerce_angle_array_deg_visualise(
        active_cell_longitudes,
        name="active_cell_longitudes",
    )
    active_lat_deg = _coerce_angle_array_deg_visualise(
        active_cell_latitudes,
        name="active_cell_latitudes",
    )
    if pre_lon_deg.size != pre_lat_deg.size:
        raise ValueError("pre-RAS longitude/latitude arrays must have the same size.")
    if active_lon_deg.size != active_lat_deg.size:
        raise ValueError("active longitude/latitude arrays must have the same size.")

    switched_off_mask_use = np.asarray(switched_off_mask, dtype=bool).reshape(-1)
    if switched_off_mask_use.shape != pre_lon_deg.shape:
        raise ValueError("switched_off_mask must have the same shape as the pre-RAS cell axis.")

    boresight_ids = np.empty(0, dtype=np.int32)
    if boresight_affected_cell_ids is not None:
        boresight_ids = np.asarray(boresight_affected_cell_ids, dtype=np.int32).reshape(-1)
        if boresight_ids.size:
            if np.any(boresight_ids < 0) or np.any(boresight_ids >= int(active_lon_deg.size)):
                raise ValueError("boresight_affected_cell_ids must reference the active-cell axis.")
            boresight_ids = np.unique(boresight_ids.astype(np.int32, copy=False))

    ras_lon_deg = _coerce_scalar_angle_deg_visualise(ras_longitude, name="ras_longitude")
    ras_lat_deg = _coerce_scalar_angle_deg_visualise(ras_latitude, name="ras_latitude")

    radius_value = None
    if radius_km is not None:
        radius_value = float(radius_km)
        if radius_value < 0.0:
            raise ValueError("radius_km must be non-negative.")
    point_spacing_value = None
    if point_spacing_km is not None:
        point_spacing_value = float(point_spacing_km)
        if not np.isfinite(point_spacing_value) or point_spacing_value <= 0.0:
            raise ValueError("point_spacing_km must be finite and > 0.")

    backend_name = str(backend or "auto").strip().lower()
    if backend_name not in {"auto", "cartopy", "matplotlib"}:
        raise ValueError("backend must be 'auto', 'cartopy', or 'matplotlib'.")
    map_style_name = str(map_style or "clean").strip().lower()
    if map_style_name not in {"clean", "terrain", "relief"}:
        raise ValueError("map_style must be 'clean', 'terrain', or 'relief'.")
    appearance_use = str(appearance_variant or "light").strip().lower()
    if appearance_use not in {"light", "dark"}:
        appearance_use = "light"
    _STYLES: dict[str, dict[str, dict[str, Any]]] = {
        "light": {
            "clean": {
                "figure": "#f8fbff",
                "axes": "#eff6ff",
                "land_face": "#eaf4e8",
                "land_edge": "#cbd5c0",
                "ocean_face": "#eff6ff",
                "lake_face": "#e0f2fe",
                "river_edge": "#7dd3fc",
                "coast": "#64748b",
                "grid_alpha": 0.20,
                "text": "#0f172a",
                "grid_color": "#94a3b8",
                "cell_edge": "none",
            },
            "terrain": {
                "figure": "#eef7fb",
                "axes": "#dceff7",
                "land_face": "#cad8a3",
                "land_edge": "#8fa36e",
                "ocean_face": "#c7dfea",
                "lake_face": "#b7d6ea",
                "river_edge": "#6aa9c7",
                "coast": "#586b57",
                "grid_alpha": 0.14,
                "text": "#0f172a",
                "grid_color": "#64748b",
                "cell_edge": "none",
            },
            "relief": {
                "figure": "#f3f1eb",
                "axes": "#dde8ef",
                "land_face": "#d8cfb8",
                "land_edge": "#9c8c67",
                "ocean_face": "#ccdbe6",
                "lake_face": "#b8d3e5",
                "river_edge": "#7ca2ba",
                "coast": "#6f6758",
                "grid_alpha": 0.12,
                "text": "#1e293b",
                "grid_color": "#64748b",
                "cell_edge": "none",
            },
        },
        "dark": {
            "clean": {
                "figure": "#07111f",
                "axes": "#0c182b",
                "land_face": "#1a2e1a",
                "land_edge": "#2d4a2d",
                "ocean_face": "#0c182b",
                "lake_face": "#0e2240",
                "river_edge": "#1e5a8a",
                "coast": "#64748b",
                "grid_alpha": 0.15,
                "text": "#e2e8f0",
                "grid_color": "#334155",
                "cell_edge": "none",
            },
            "terrain": {
                "figure": "#07111f",
                "axes": "#0c1a2e",
                "land_face": "#263818",
                "land_edge": "#3d5c26",
                "ocean_face": "#0a1628",
                "lake_face": "#0c1f3a",
                "river_edge": "#1a4d73",
                "coast": "#4a6a4a",
                "grid_alpha": 0.12,
                "text": "#e2e8f0",
                "grid_color": "#334155",
                "cell_edge": "none",
            },
            "relief": {
                "figure": "#0a0f18",
                "axes": "#0d1520",
                "land_face": "#2a2418",
                "land_edge": "#4a3e28",
                "ocean_face": "#0c1520",
                "lake_face": "#0e1a2c",
                "river_edge": "#2a5570",
                "coast": "#5a5040",
                "grid_alpha": 0.10,
                "text": "#d4cfc0",
                "grid_color": "#3a3528",
                "cell_edge": "none",
            },
        },
    }
    style = _STYLES[appearance_use][map_style_name]

    cartopy_available = False
    ccrs = None
    if backend_name in {"auto", "cartopy"}:
        try:
            import cartopy.crs as ccrs_mod

            cartopy_available = True
            ccrs = ccrs_mod
        except ImportError:
            if backend_name == "cartopy":
                raise RuntimeError(
                    "cartopy is required for backend='cartopy'. Install it with "
                    "'conda install -n scepter-dev-full -c conda-forge shapely cartopy'."
                )
    backend_used = (
        "cartopy"
        if (backend_name == "cartopy" or (backend_name == "auto" and cartopy_available))
        else "matplotlib"
    )

    pre_radius_mask = np.ones(pre_lon_deg.shape, dtype=bool)
    active_radius_mask = np.ones(active_lon_deg.shape, dtype=bool)
    if radius_value is not None:
        pre_radius_mask = _great_circle_distance_km_visualise(
            pre_lon_deg,
            pre_lat_deg,
            ref_lon_deg=ras_lon_deg,
            ref_lat_deg=ras_lat_deg,
        ) <= radius_value
        active_radius_mask = _great_circle_distance_km_visualise(
            active_lon_deg,
            active_lat_deg,
            ref_lon_deg=ras_lon_deg,
            ref_lat_deg=ras_lat_deg,
        ) <= radius_value

    boresight_active_mask = np.zeros(active_lon_deg.shape, dtype=bool)
    boresight_active_mask[boresight_ids] = True
    boresight_plot_mask = boresight_active_mask & active_radius_mask
    normal_active_plot_mask = (~boresight_active_mask) & active_radius_mask
    switched_off_plot_mask = switched_off_mask_use & pre_radius_mask
    reuse_factor_i = 1 if reuse_factor is None else max(1, int(reuse_factor))
    active_slot_ids = None
    use_reuse_coloring = False
    if active_reuse_slot_ids is not None:
        active_slot_ids_use = np.asarray(active_reuse_slot_ids, dtype=np.int32).reshape(-1)
        if active_slot_ids_use.shape != active_lon_deg.shape:
            raise ValueError("active_reuse_slot_ids must match the active-cell axis.")
        active_slot_ids = np.mod(active_slot_ids_use, np.int32(reuse_factor_i)).astype(
            np.int32,
            copy=False,
        )
        use_reuse_coloring = reuse_factor_i > 1
    anchor_index = None if anchor_active_index is None else int(anchor_active_index)
    anchor_valid = (
        anchor_index is not None
        and anchor_index >= 0
        and anchor_index < int(active_lon_deg.size)
        and bool(active_radius_mask[anchor_index])
    )

    plotted_lon_all = np.concatenate(
        [
            pre_lon_deg[switched_off_plot_mask],
            active_lon_deg[normal_active_plot_mask],
            active_lon_deg[boresight_plot_mask],
            np.asarray([ras_lon_deg], dtype=np.float64),
        ]
    )
    plotted_lat_all = np.concatenate(
        [
            pre_lat_deg[switched_off_plot_mask],
            active_lat_deg[normal_active_plot_mask],
            active_lat_deg[boresight_plot_mask],
            np.asarray([ras_lat_deg], dtype=np.float64),
        ]
    )

    if radius_value is not None:
        lat_half_span = np.degrees(radius_value / float(R_earth.to_value(u.km)))
        cos_lat = max(np.cos(np.deg2rad(ras_lat_deg)), 1.0e-6)
        lon_half_span = lat_half_span / cos_lat
        extent = (
            ras_lon_deg - lon_half_span,
            ras_lon_deg + lon_half_span,
            ras_lat_deg - lat_half_span,
            ras_lat_deg + lat_half_span,
        )
        extent_mode_used = "radius_km"
    else:
        lon_span = float(np.max(plotted_lon_all) - np.min(plotted_lon_all)) if plotted_lon_all.size else 1.0
        lat_span = float(np.max(plotted_lat_all) - np.min(plotted_lat_all)) if plotted_lat_all.size else 1.0
        lon_pad = max(0.25, 0.05 * lon_span)
        lat_pad = max(0.25, 0.05 * lat_span)
        extent = (
            float(np.min(plotted_lon_all)) - lon_pad,
            float(np.max(plotted_lon_all)) + lon_pad,
            float(np.min(plotted_lat_all)) - lat_pad,
            float(np.max(plotted_lat_all)) + lat_pad,
        )
        extent_mode_used = "auto"

    point_count = (
        int(np.count_nonzero(switched_off_plot_mask))
        + int(np.count_nonzero(normal_active_plot_mask))
        + int(np.count_nonzero(boresight_plot_mask))
    )
    marker_size = 110.0 if point_count <= 500 else 50.0 if point_count <= 3_000 else 18.0

    # --- Build hex polygons at actual cell centres. ---
    # Primary path: Voronoi tessellation gives gap-free, overlap-free cells
    # whose centres are the exact simulation cell positions.  Shared-lattice
    # and inferred-lattice paths are kept as fallbacks.
    voronoi_result = _hexgrid_voronoi_polygons_lonlat(
        pre_lon_deg,
        pre_lat_deg,
        active_lon_deg=active_lon_deg,
        active_lat_deg=active_lat_deg,
        ref_lon_deg=ras_lon_deg,
        ref_lat_deg=ras_lat_deg,
        point_spacing_km=point_spacing_value,
    )
    if voronoi_result is not None:
        pre_cell_polygons, active_cell_polygons, hex_geometry_info = voronoi_result
    else:
        shared_hex_geometry = _resolve_shared_hexgrid_polygons_lonlat(
            pre_lon_deg,
            pre_lat_deg,
            active_longitudes_deg=active_lon_deg,
            active_latitudes_deg=active_lat_deg,
            hex_lattice=hex_lattice,
            point_spacing_km=point_spacing_value,
        )
        if shared_hex_geometry is not None:
            pre_cell_polygons, active_cell_polygons, hex_geometry_info = shared_hex_geometry
        else:
            pre_cell_polygons, active_cell_polygons, hex_geometry_info = (
                _infer_and_build_lattice_polygons(
                    pre_lon_deg,
                    pre_lat_deg,
                    active_lon_deg=active_lon_deg,
                    active_lat_deg=active_lat_deg,
                    ref_lon_deg=ras_lon_deg,
                    ref_lat_deg=ras_lat_deg,
                    point_spacing_km=point_spacing_value,
                )
            )
    use_shared_hex_lattice = str(hex_geometry_info["hex_geometry_mode"]) in {
        "shared_lattice",
        "inferred_lattice",
        "voronoi",
    }

    fig = _new_mpl_figure(figsize=(9.0, 7.5))
    fig.set_facecolor(style["figure"])
    if backend_used == "cartopy":
        # Use a local azimuthal projection centred on the RAS station so
        # hex cells appear with correct shape at all latitudes instead of
        # being horizontally stretched near the poles in PlateCarree.
        map_proj = ccrs.AzimuthalEquidistant(
            central_longitude=float(ras_lon_deg),
            central_latitude=float(ras_lat_deg),
        )
        ax = fig.add_subplot(1, 1, 1, projection=map_proj)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        # Correct aspect ratio so cells are not stretched at high latitudes.
        center_lat = 0.5 * (extent[2] + extent[3])
        cos_lat = max(np.cos(np.deg2rad(center_lat)), 0.1)
        ax.set_aspect(1.0 / cos_lat)
    ax.set_facecolor(style["axes"])

    if backend_used == "cartopy":
        try:
            import cartopy.feature as cfeature
        except ImportError:
            cfeature = None
        if cfeature is not None:
            if map_style_name == "relief":
                try:
                    ax.stock_img()
                except Exception:
                    pass
            if map_style_name in {"terrain", "relief"}:
                try:
                    ax.add_feature(
                        cfeature.OCEAN.with_scale("50m"),
                        facecolor=style["ocean_face"],
                        edgecolor="none",
                        alpha=0.95,
                        zorder=-2,
                    )
                    ax.add_feature(
                        cfeature.LAKES.with_scale("50m"),
                        facecolor=style["lake_face"],
                        edgecolor="none",
                        alpha=0.95,
                        zorder=-1,
                    )
                    ax.add_feature(
                        cfeature.RIVERS.with_scale("50m"),
                        facecolor="none",
                        edgecolor=style["river_edge"],
                        linewidth=0.45,
                        alpha=0.55,
                        zorder=1,
                    )
                    if map_style_name == "terrain":
                        ax.add_feature(
                            cfeature.LAND.with_scale("50m"),
                            facecolor=style["land_face"],
                            edgecolor="none",
                            alpha=0.45,
                            zorder=-1,
                        )
                except Exception:
                    pass

    try:
        if backend_used == "cartopy":
            from scepter import earthgrid as earthgrid_mod

            land_geometries = earthgrid_mod._load_natural_earth_geometries(
                "land",
                backend="vendored",
            )
            coastline_geometries = earthgrid_mod._load_natural_earth_geometries(
                "coastline",
                backend="vendored",
            )
            if land_geometries:
                ax.add_geometries(
                    land_geometries,
                    crs=ccrs.PlateCarree(),
                    facecolor=style["land_face"],
                    edgecolor=style["land_edge"],
                    linewidth=0.35,
                    alpha=0.78,
                    zorder=0,
                )
            if coastline_geometries:
                ax.add_geometries(
                    coastline_geometries,
                    crs=ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor=style["coast"],
                    linewidth=0.6 if map_style_name != "clean" else 0.5,
                    alpha=0.7,
                )
        else:
            land_vertices = _cached_mpl_natural_earth_vertices("land", backend="vendored")
            coastline_vertices = _cached_mpl_natural_earth_vertices("coastline", backend="vendored")
            if land_vertices:
                ax.add_collection(
                    PolyCollection(
                        land_vertices,
                        facecolors=style["land_face"],
                        edgecolors=style["land_edge"],
                        linewidths=0.35,
                        alpha=0.78,
                        zorder=0,
                    )
                )
            if coastline_vertices:
                ax.add_collection(
                    LineCollection(
                        coastline_vertices,
                        colors=style["coast"],
                        linewidths=0.6 if map_style_name != "clean" else 0.5,
                        alpha=0.7,
                        zorder=1,
                    )
                )
    except Exception:
        pass

    def _masked_polygons(polygons: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
        return [
            polygons[int(idx)]
            for idx in np.flatnonzero(mask)
            if polygons[int(idx)].size > 0
        ]

    def _add_hex_collection(
        polygons: list[np.ndarray],
        *,
        facecolor: Any,
        edgecolor: Any,
        linewidth: float,
        alpha: float,
        zorder: float,
        gid: str,
        antialiased: bool = True,
    ) -> None:
        if not polygons:
            return
        collection = PolyCollection(
            polygons,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
            zorder=zorder,
            closed=True,
        )
        collection.set_gid(gid)
        collection.set_antialiaseds([bool(antialiased)])
        if backend_used == "cartopy":
            collection.set_transform(ccrs.PlateCarree())
        ax.add_collection(collection)

    text_color = str(style.get("text", "#0f172a"))
    grid_color = str(style.get("grid_color", "#94a3b8"))
    boresight_outline = text_color if appearance_use == "dark" else "#111827"
    boresight_legend_face = style["axes"] if appearance_use == "dark" else "#ffffff"

    legend_handles: list[Any] = []
    if int(np.count_nonzero(switched_off_plot_mask)) > 0:
        _add_hex_collection(
            _masked_polygons(pre_cell_polygons, switched_off_plot_mask),
            facecolor="#6B7280",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.28,
            zorder=2,
            gid="hexgrid_switched_off_cells",
            antialiased=not use_shared_hex_lattice,
        )
        legend_handles.append(
            Patch(
                facecolor="#6B7280",
                edgecolor="none",
                alpha=0.28,
                label=f"Switched off ({int(np.count_nonzero(switched_off_plot_mask))})",
            )
        )

    if use_reuse_coloring and active_slot_ids is not None:
        slot_colors = _frequency_reuse_slot_colors(reuse_factor_i)
        for slot_id in range(reuse_factor_i):
            slot_mask = active_slot_ids == int(slot_id)
            normal_slot_mask = normal_active_plot_mask & slot_mask
            boresight_slot_mask = boresight_plot_mask & slot_mask
            if not np.any(normal_slot_mask) and not np.any(boresight_slot_mask):
                continue
            slot_color = slot_colors[int(slot_id)]
            if np.any(normal_slot_mask):
                _add_hex_collection(
                    _masked_polygons(active_cell_polygons, normal_slot_mask),
                    facecolor=slot_color,
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.50,
                    zorder=3,
                    gid=f"hexgrid_reuse_slot_{int(slot_id) + 1}",
                    antialiased=not use_shared_hex_lattice,
                )
            if np.any(boresight_slot_mask):
                _add_hex_collection(
                    _masked_polygons(active_cell_polygons, boresight_slot_mask),
                    facecolor=slot_color,
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.65,
                    zorder=4,
                    gid=f"hexgrid_reuse_slot_{int(slot_id) + 1}_boresight_fill",
                    antialiased=not use_shared_hex_lattice,
                )
            legend_handles.append(
                Patch(
                    facecolor=slot_color,
                    edgecolor=boresight_outline if np.any(boresight_slot_mask) else "none",
                    linewidth=0.9 if np.any(boresight_slot_mask) else 0.0,
                    label=f"Reuse slot {int(slot_id) + 1}",
                )
            )
        if int(np.count_nonzero(boresight_plot_mask)) > 0:
            _add_hex_collection(
                _masked_polygons(active_cell_polygons, boresight_plot_mask),
                facecolor="none",
                edgecolor=boresight_outline,
                linewidth=1.0,
                alpha=0.95,
                zorder=5,
                gid="hexgrid_boresight_outline",
            )
            legend_handles.append(
                Patch(
                    facecolor=boresight_legend_face,
                    edgecolor=boresight_outline,
                    linewidth=1.0,
                    label=f"Boresight outline ({int(np.count_nonzero(boresight_plot_mask))})",
                )
            )
    else:
        if int(np.count_nonzero(normal_active_plot_mask)) > 0:
            _add_hex_collection(
                _masked_polygons(active_cell_polygons, normal_active_plot_mask),
                facecolor="#0F766E",
                edgecolor="none",
                linewidth=0.0,
                alpha=0.28,
                zorder=3,
                gid="hexgrid_active_cells",
                antialiased=not use_shared_hex_lattice,
            )
            legend_handles.append(
                Patch(
                    facecolor="#0F766E",
                    edgecolor="none",
                    alpha=0.28,
                    label=f"Normal active ({int(np.count_nonzero(normal_active_plot_mask))})",
                )
            )
        if int(np.count_nonzero(boresight_plot_mask)) > 0:
            _add_hex_collection(
                _masked_polygons(active_cell_polygons, boresight_plot_mask),
                facecolor="#C2410C",
                edgecolor="none",
                linewidth=0.0,
                alpha=0.38,
                zorder=4,
                gid="hexgrid_boresight_cells",
                antialiased=not use_shared_hex_lattice,
            )
            legend_handles.append(
                Patch(
                    facecolor="#C2410C",
                    edgecolor="none",
                    alpha=0.38,
                    label=f"Boresight-affected active ({int(np.count_nonzero(boresight_plot_mask))})",
                )
            )
    ras_marker_face = text_color
    ras_marker_edge = style["axes"] if appearance_use == "dark" else "white"
    ras_scatter_kwargs: dict[str, Any] = {}
    if backend_used == "cartopy":
        ras_scatter_kwargs["transform"] = ccrs.PlateCarree()
    ax.scatter(
        [ras_lon_deg],
        [ras_lat_deg],
        marker="*",
        s=max(180.0, marker_size * 3.5),
        color=ras_marker_face,
        edgecolors=ras_marker_edge,
        linewidths=0.8,
        zorder=5,
        **ras_scatter_kwargs,
    )
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markersize=13,
            markerfacecolor=ras_marker_face,
            markeredgecolor=ras_marker_edge,
            markeredgewidth=0.8,
            label="RAS station",
        )
    )

    title = "Cell Status Map"
    if use_reuse_coloring:
        title = f"Cell Status Map (F{reuse_factor_i} reuse)"
    ax.set_title(title, pad=12, color=text_color)
    ax.set_xlabel("Longitude [deg]", color=text_color)
    ax.set_ylabel("Latitude [deg]", color=text_color)
    ax.tick_params(colors=text_color, which="both")
    ax.grid(True, alpha=float(style["grid_alpha"]), linestyle=":", color=grid_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    if legend_handles:
        fig.subplots_adjust(right=0.78)
        legend = ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=True,
            ncol=1,
        )
        if legend is not None:
            legend.get_frame().set_facecolor(style["figure"])
            legend.get_frame().set_edgecolor(grid_color)
            for label in legend.get_texts():
                label.set_color(text_color)

    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_obj, dpi=160, bbox_inches="tight")

    info = {
        "backend_used": backend_used,
        "extent_mode_used": extent_mode_used,
        "extent": tuple(float(v) for v in extent),
        "switched_off_count": int(np.count_nonzero(switched_off_plot_mask)),
        "normal_active_count": int(np.count_nonzero(normal_active_plot_mask)),
        "boresight_affected_active_count": int(np.count_nonzero(boresight_plot_mask)),
        "radius_km": None if radius_value is None else float(radius_value),
        "map_style_used": map_style_name,
        "reuse_factor": int(reuse_factor_i),
        "reuse_coloring_active": bool(use_reuse_coloring),
        "anchor_active_index": None if not anchor_valid else int(anchor_index),
        "point_spacing_km_used": float(hex_geometry_info["point_spacing_km_used"]),
        "hex_radius_km_used": float(hex_geometry_info["hex_radius_km_used"]),
        "hex_orientation_used": str(hex_geometry_info["hex_orientation_used"]),
        "hex_geometry_mode": str(hex_geometry_info["hex_geometry_mode"]),
        "hex_anchor_pre_ras_index": hex_geometry_info["hex_anchor_pre_ras_index"],
        "hex_center_render_residual_km_max": float(
            hex_geometry_info["hex_center_render_residual_km_max"]
        ),
        "hex_center_render_residual_km_median": float(
            hex_geometry_info["hex_center_render_residual_km_median"]
        ),
    }
    return (fig, info) if return_info else fig


def plot_frequency_reuse_scheme(
    *,
    prepared_grid: Mapping[str, Any] | None = None,
    reuse_plan: Mapping[str, Any] | None = None,
    reuse_factor: int | None = None,
    anchor_slot: int = 0,
    boresight_affected_cell_ids: Any | None = None,
    save_path: str | Path | None = None,
    return_info: bool = False,
):
    """
    Plot a didactic schematic of the selected frequency-reuse cluster.

    Parameters
    ----------
    prepared_grid : Mapping[str, Any] or None, optional
        Legacy compatibility parameter. The schematic-only viewer no longer
        requires prepared geographic grid data.
    reuse_plan : Mapping[str, Any] or None, optional
        Optional reuse-slot mapping returned by
        :func:`scepter.earthgrid.resolve_frequency_reuse_slots`.
    reuse_factor : int or None, optional
        Explicit reuse factor used when ``reuse_plan`` is not supplied.
    anchor_slot : int, optional
        Slot assigned to the schematic anchor cell when ``reuse_plan`` is not
        supplied.
    boresight_affected_cell_ids : array-like of int or None, optional
        Ignored in the schematic-only viewer and retained only for backwards
        compatibility.
    save_path : str, pathlib.Path, or None, optional
        Optional output path for ``fig.savefig``.
    return_info : bool, optional
        When ``True``, return ``(fig, info)`` with reuse metadata.
    """
    from scepter import earthgrid as earthgrid_mod

    del prepared_grid, boresight_affected_cell_ids

    if reuse_plan is not None:
        reuse_factor_i = int(reuse_plan["reuse_factor"])
        anchor_slot_i = int(reuse_plan.get("anchor_slot", anchor_slot)) % reuse_factor_i
    else:
        if reuse_factor is None:
            raise ValueError("A reuse_factor or reuse_plan is required for the reuse schematic.")
        reuse_factor_i = int(reuse_factor)
        anchor_slot_i = int(anchor_slot) % reuse_factor_i
    if reuse_factor_i < 1:
        raise ValueError("reuse_factor must be positive.")

    colors = _frequency_reuse_slot_colors(reuse_factor_i)
    shift_pair = earthgrid_mod._resolve_hexgrid_reuse_shift_pair(reuse_factor_i)
    _coset_to_slot, cluster_representatives = earthgrid_mod._enumerate_reuse_cluster_slots(
        reuse_factor=reuse_factor_i,
        shift_pair=shift_pair,
        anchor_slot=anchor_slot_i,
    )

    def _axial_to_xy(q_coord: float, r_coord: float) -> tuple[float, float]:
        # Keep the current didactic axial-to-display transform unchanged: the
        # reuse-scheme viewer is schematic-only and intentionally independent
        # from the geographic hexgrid renderer.
        return (
            1.5 * float(q_coord),
            float(np.sqrt(3.0)) * (float(r_coord) + 0.5 * float(q_coord)),
        )

    cluster_translation_radius_by_reuse = {
        1: 1,
        3: 1,
        4: 1,
        7: 1,
        9: 1,
        12: 1,
        13: 1,
        16: 1,
        19: 1,
    }
    cluster_translation_radius = int(cluster_translation_radius_by_reuse.get(reuse_factor_i, 1))
    shift_q, shift_r = (int(shift_pair[0]), int(shift_pair[1]))
    lattice_q_step = (int(shift_q), int(shift_r))
    lattice_r_step = (int(-shift_r), int(shift_q + shift_r))
    displayed_cells: dict[tuple[int, int], int] = {}
    for cluster_q in range(-cluster_translation_radius, cluster_translation_radius + 1):
        cluster_r_min = max(-cluster_translation_radius, -cluster_q - cluster_translation_radius)
        cluster_r_max = min(cluster_translation_radius, -cluster_q + cluster_translation_radius)
        for cluster_r in range(cluster_r_min, cluster_r_max + 1):
            offset_q = (
                int(cluster_q) * int(lattice_q_step[0])
                + int(cluster_r) * int(lattice_r_step[0])
            )
            offset_r = (
                int(cluster_q) * int(lattice_q_step[1])
                + int(cluster_r) * int(lattice_r_step[1])
            )
            for representative in cluster_representatives:
                cell_key = (
                    int(representative["axial_q"]) + int(offset_q),
                    int(representative["axial_r"]) + int(offset_r),
                )
                displayed_cells.setdefault(cell_key, int(representative["slot_id"]))

    fig, schematic_ax = _new_mpl_subplots(1, 1, figsize=(9.8, 7.0))
    fig.patch.set_facecolor("#fbfaf7")
    schematic_ax.set_facecolor("#fffdf8")

    plotted_slots: list[int] = []
    show_slot_numbers = reuse_factor_i < 12
    # Pointy-top hexes rotated by 30 degrees are the reference reuse-scheme
    # look; regressions here visibly break the seamless schematic tiling.
    hex_orientation = float(np.pi / 6.0)
    for (q_coord, r_coord), slot_id in sorted(
        displayed_cells.items(),
        key=lambda item: (item[0][0] + item[0][1], item[0][1], item[0][0]),
    ):
        x_coord, y_coord = _axial_to_xy(q_coord, r_coord)
        plotted_slots.append(int(slot_id))
        hex_patch = RegularPolygon(
            (x_coord, y_coord),
            numVertices=6,
            radius=1.0,
            orientation=hex_orientation,
            facecolor=colors[int(slot_id)],
            edgecolor="#1f2937",
            linewidth=1.2,
            alpha=0.96,
        )
        schematic_ax.add_patch(hex_patch)
        if show_slot_numbers:
            schematic_ax.text(
                x_coord,
                y_coord,
                str(int(slot_id) + 1),
                ha="center",
                va="center",
                fontsize=15 if reuse_factor_i <= 7 else 13,
                fontweight="bold",
                color="#111827",
            )
    schematic_ax.set_aspect("equal", adjustable="box")
    schematic_ax.autoscale_view()
    schematic_ax.margins(x=0.08, y=0.20)
    schematic_ax.axis("off")
    schematic_ax.text(
        0.01,
        0.965,
        f"F{reuse_factor_i} reuse cluster",
        ha="left",
        va="top",
        transform=schematic_ax.transAxes,
        fontsize=13,
        fontweight="bold",
    )
    schematic_ax.text(
        0.01,
        0.90,
        "A cell cluster is replicated over the coverage area.",
        ha="left",
        va="top",
        transform=schematic_ax.transAxes,
        fontsize=10,
        color="#475569",
    )
    legend_handles = [
        Patch(facecolor=colors[idx], edgecolor="#1f2937", label=f"Slot {idx + 1}")
        for idx in range(reuse_factor_i)
    ]
    # Keep the legend on the right so the cluster itself keeps the maximum
    # amount of horizontal drawing space.
    schematic_ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
        ncol=1,
    )
    fig.suptitle("Frequency Reuse Scheme", fontsize=16, y=0.975)
    fig.tight_layout(rect=(0.0, 0.02, 0.84, 0.94))

    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_obj, dpi=160, bbox_inches="tight")

    info = {
        "reuse_factor": int(reuse_factor_i),
        "anchor_slot": int(anchor_slot_i),
        "cluster_slot_count": int(len(cluster_representatives)),
        "slot_ids_present": tuple(
            int(value)
            for value in sorted(set(plotted_slots))
        ),
    }
    return (fig, info) if return_info else fig
