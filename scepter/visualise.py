"""
visualise.py
=============
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence, Dict, Tuple, List
import warnings
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


# -----------------------------------------------------------------------------
# Shared helpers (units, dB math, S.1586 grid, geometry, polar mapping)
# -----------------------------------------------------------------------------

def _to_plain_array(data: Any) -> Tuple[np.ndarray, Any | None]:
    """
    Return (values, unit-like) where `values` is a plain ndarray.
    If `data` has a .unit attribute (Quantity), the numeric values are returned
    and the unit is passed through. Otherwise, the unit is None.
    """
    if hasattr(data, "unit"):
        return np.asarray(data.value), getattr(data, "unit")
    return np.asarray(data), None


def _to_lin(x: np.ndarray | float) -> np.ndarray:
    """Convert dB to linear power: P = 10^(x/10)."""
    return np.power(10.0, np.asarray(x, dtype=float) / 10.0)


def _to_db(x: np.ndarray | float) -> np.ndarray:
    """
    Convert linear power to dB. Values are clamped to a tiny positive number
    to avoid log(0).
    """
    x = np.maximum(np.asarray(x, dtype=float), np.finfo(float).tiny)
    return 10.0 * np.log10(x)


# S.1586 azimuth step per ring (lower elevation edge in degrees → az step)
_S1586_AZ_STEPS: Dict[int, int] = {
    0: 3, 3: 3, 6: 3, 9: 3, 12: 3, 15: 3, 18: 3, 21: 3, 24: 3, 27: 3,
    30: 4, 33: 4, 36: 4, 39: 4, 42: 4, 45: 4,
    48: 5, 51: 5, 54: 5,
    57: 6, 60: 6, 63: 6,
    66: 8, 69: 9, 72: 10, 75: 12, 78: 18, 81: 24, 84: 40, 87: 120,
}


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
    _, _, _, _, el_edges, cells_per_ring = _s1586_cells()
    ring_lows = el_edges[:-1].astype(int)
    ring_to_idx = {rl: i for i, rl in enumerate(ring_lows)}
    offsets = np.concatenate([[0], np.cumsum(cells_per_ring[:-1])])

    el_lo_clip = gi["cell_lat_low"].astype(float)
    ring_low = np.minimum(np.floor(el_lo_clip / 3.0) * 3.0, 87.0).astype(int)
    # Vectorized lookup using numpy's vectorize for dictionary lookup
    # All keys should exist due to clamping, but provide defaults for safety
    ring_idx = np.array([ring_to_idx.get(int(rl), 0) for rl in ring_low], dtype=int)
    step = np.array([_S1586_AZ_STEPS.get(int(rl), 3) for rl in ring_low], dtype=int)
    n_in_ring = cells_per_ring[ring_idx]

    az_lo = gi["cell_lon_low"].astype(float)
    az_bin = np.floor(az_lo / step).astype(int) % n_in_ring
    return offsets[ring_idx] + az_bin


# -----------------------------------------------------------------------------
# CDF / CCDF
# -----------------------------------------------------------------------------

def plot_cdf_ccdf(
    data: Any,
    *,
    # shape / units / math
    cell_axis: int = -1,          # axis with skycells; others flattened
    log_mode: bool = True,        # percentiles computed in linear if values are dB-like
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
    y_percent_decimals: int = 2,                 # % axis decimals for CDF/CCDF
    ccdf_ymin_pct: float | None = None,          # CCDF lower cutoff in percent (0..100)
    cdf_ymin_pct: float | None = None,           # CDF  lower cutoff in percent (0..100)
    y_log: bool = False,                         # log-scale Y axis for tails
    show: bool = True,
    return_values: bool = False,                 # if True → (fig, info_dict)
):
    """
    Plot the empirical CDF/CCDF of the input sample distribution.

    - One axis in `data` holds the sky-cells (given by `cell_axis`); all others
      are flattened into one long sample vector.
    - If `log_mode=True`, percentiles are computed in linear power, then converted
      back to dB at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
    info : dict (optional if `return_values=True`)
        {
          "unit": astropy Unit or None,
          "p95": value or Quantity (95th percentile),
          "p98": value or Quantity (98th percentile),
          "n":   number of samples used (after NaN filtering)
        }
    """
    # Small helpers
    def _dedup_labels(labels: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        out: List[str] = []
        for lab in labels:
            seen[lab] = seen.get(lab, 0) + 1
            out.append(lab if seen[lab] == 1 else f"{lab} ({seen[lab]})")
        return out

    # Accept scalar/list for protection values and turn into a clean list
    if prot_value is None:
        prot_list: List[Any] = []
    elif isinstance(prot_value, (list, tuple, np.ndarray)):
        prot_list = list(prot_value)
    else:
        prot_list = [prot_value]

    # Default legends sized to match
    if prot_legend is None:
        prot_legend = ["Protection limit"] * len(prot_list)
    else:
        if len(prot_legend) < len(prot_list):
            prot_legend = prot_legend + ["Protection limit"] * (len(prot_list) - len(prot_legend))
        elif len(prot_legend) > len(prot_list):
            prot_legend = prot_legend[:len(prot_list)]
    prot_legend = _dedup_labels(prot_legend)

    # Prepare data
    arr, unit = _to_plain_array(data)
    if arr.ndim < 2:
        raise ValueError(f"'data' must have at least 2 dims; got {arr.shape}.")
    if cell_axis < 0:
        cell_axis = arr.ndim + cell_axis
    if not (0 <= cell_axis < arr.ndim):
        raise ValueError(f"cell_axis {cell_axis} out of range for shape {arr.shape}.")

    arr = np.moveaxis(arr, cell_axis, -1)
    samples_all = arr.reshape(-1)

    # Drop NaNs
    mask = np.isfinite(samples_all)
    x_native = samples_all[mask]
    n = x_native.size
    if n == 0:
        raise ValueError("No finite samples to plot.")

    # ECDF arrays: both start above zero, so log-scale is safe
    x_sorted = np.sort(x_native)
    y_cdf = np.arange(1, n + 1, dtype=float) / n
    y_ccdf = 1.0 - np.arange(0, n, dtype=float) / n

    # Percentiles (computed in linear if requested)
    def _pct_value(p: float) -> float:
        return (_to_db(np.nanpercentile(_to_lin(x_native), p))
                if log_mode else np.nanpercentile(x_native, p))

    p95 = _pct_value(95.0) if show_five_percent else None
    p98 = _pct_value(98.0) if show_two_percent  else None

    def _maybe_Q(val):
        return (val * unit) if (val is not None and unit is not None) else val

    p95_q = _maybe_Q(p95)
    p98_q = _maybe_Q(p98)

    # Protection thresholds normalised to the same unit as the data
    prot_x_vals: List[float] = []
    for pv in prot_list:
        if hasattr(pv, "to") and (unit is not None):
            prot_x_vals.append(float(pv.to(unit).value))
        elif hasattr(pv, "value"):
            prot_x_vals.append(float(pv.value))
        else:
            prot_x_vals.append(float(pv))

    # Distinct colours for multiple protection lines if user didn’t specify
    if prot_colors is None or len(prot_colors) == 0:
        cmap = plt.get_cmap("tab10")
        prot_line_colors = [cmap(i % cmap.N) for i in range(len(prot_x_vals))]
    else:
        prot_line_colors = [prot_colors[i % len(prot_colors)] for i in range(len(prot_x_vals))]

    # Which view(s) to draw
    pt = plot_type.lower()
    if pt not in {"cdf", "ccdf", "both"}:
        raise ValueError("plot_type must be 'cdf', 'ccdf', or 'both'.")

    if pt == "both":
        fig, (ax_cdf, ax_ccdf) = plt.subplots(1, 2, figsize=figsize, sharex=True)
    elif pt == "cdf":
        fig, ax_cdf = plt.subplots(1, 1, figsize=figsize)
        ax_ccdf = None
    else:
        fig, ax_ccdf = plt.subplots(1, 1, figsize=figsize)
        ax_cdf = None

    xlab = "Value" + (f" [{unit}]" if unit is not None else "")
    percent_fmt = mtick.PercentFormatter(xmax=1.0, decimals=y_percent_decimals)

    # CDF
    if ax_cdf is not None:
        ax = ax_cdf
        ax.step(x_sorted, y_cdf, where="post", linewidth=line_width, alpha=line_alpha,
                color=cdf_color, label="CDF")
        if grid:
            ax.grid(True, alpha=0.35)

        if show_five_percent:
            ax.axhline(0.95, color="#9ca3af", linestyle="--", linewidth=1.6, label="95% (5% worst)")
            if p95 is not None and np.isfinite(p95):
                ax.scatter([p95], [0.95], s=marker_size, color=cdf_color, zorder=5)
                ax.annotate(f"{p95_q:.3f}" if unit is not None else f"{p95:.3f}",
                            xy=(p95, 0.95), xytext=(5, 8), textcoords="offset points",
                            fontsize=9, color=cdf_color)

        if show_two_percent:
            ax.axhline(0.98, color="#6b7280", linestyle="--", linewidth=1.6, label="98% (2% worst)")
            if p98 is not None and np.isfinite(p98):
                ax.scatter([p98], [0.98], s=marker_size, color=cdf_color, zorder=5)
                ax.annotate(f"{p98_q:.3f}" if unit is not None else f"{p98:.3f}",
                            xy=(p98, 0.98), xytext=(5, 8), textcoords="offset points",
                            fontsize=9, color=cdf_color)

        for xv, lab, col in zip(prot_x_vals, prot_legend, prot_line_colors):
            ax.axvline(xv, color=col, linestyle="-.", linewidth=1.6, label=lab, alpha=0.95)

        ax.set_xlabel(xlab)
        ax.set_ylabel("CDF")

        if y_log:
            ymin = 1.0 / n
            if cdf_ymin_pct is not None:
                ymin = max(ymin, min(max(cdf_ymin_pct / 100.0, 0.0), 1.0))
            ax.set_yscale("log")
            ax.set_ylim(ymin, 1.0)
        else:
            ymin = 0.0 if cdf_ymin_pct is None else min(max(cdf_ymin_pct / 100.0, 0.0), 1.0)
            ax.set_ylim(ymin, 1.0)

        ax.yaxis.set_major_formatter(percent_fmt)
        ax.legend(loc="lower right", frameon=True)

    # CCDF
    if ax_ccdf is not None:
        ax = ax_ccdf
        ax.step(x_sorted, y_ccdf, where="pre", linewidth=line_width, alpha=line_alpha,
                color=ccdf_color, label="CCDF")
        if grid:
            ax.grid(True, alpha=0.35)

        if show_five_percent:
            ax.axhline(0.05, color="#9ca3af", linestyle="--", linewidth=1.6, label="5%")
            if p95 is not None and np.isfinite(p95):
                ax.scatter([p95], [0.05], s=marker_size, color=ccdf_color, zorder=5)
                ax.annotate(f"{p95_q:.3f}" if unit is not None else f"{p95:.3f}",
                            xy=(p95, 0.05), xytext=(5, 8), textcoords="offset points",
                            fontsize=9, color=ccdf_color)

        if show_two_percent:
            ax.axhline(0.02, color="#6b7280", linestyle="--", linewidth=1.6, label="2%")
            if p98 is not None and np.isfinite(p98):
                ax.scatter([p98], [0.02], s=marker_size, color=ccdf_color, zorder=5)
                ax.annotate(f"{p98_q:.3f}" if unit is not None else f"{p98:.3f}",
                            xy=(p98, 0.02), xytext=(5, 8), textcoords="offset points",
                            fontsize=9, color=ccdf_color)

        for xv, lab, col in zip(prot_x_vals, prot_legend, prot_line_colors):
            ax.axvline(xv, color=col, linestyle="-.", linewidth=1.6, label=lab, alpha=0.95)

        ax.set_xlabel(xlab)
        ax.set_ylabel("CCDF")

        if y_log:
            ymin = 1.0 / n
            if ccdf_ymin_pct is not None:
                ymin = max(ymin, min(max(ccdf_ymin_pct / 100.0, 0.0), 1.0))
            ax.set_yscale("log")
            ax.set_ylim(ymin, 1.0)
        else:
            ymin = 0.0 if ccdf_ymin_pct is None else min(max(ccdf_ymin_pct / 100.0, 0.0), 1.0)
            ax.set_ylim(ymin, 1.0)

        ax.yaxis.set_major_formatter(percent_fmt)
        ax.legend(loc="upper right", frameon=True)

    # Title and layout
    base = {"cdf": "CDF", "ccdf": "CCDF", "both": "CDF & CCDF"}[pt]
    ttl = f"{base} — Empirical distribution" if title is None else title
    fig.suptitle(ttl)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        plt.close(fig)

    info = {
        "unit": unit,
        "p95": p95_q if p95 is not None else None,
        "p98": p98_q if p98 is not None else None,
        "n": int(n),
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
    log_mode: bool = True,                      # do math in linear domain if values are in dB

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
    Plot ITU-R S.1586-1 upper hemisphere as a 2D map (polar or rectangular).

    Key behavior in this version (polar, when `grid_info` is present)
    ---------------------------------------------------------------
    - We no longer force S.1586 rings/bins. Instead, we draw **one polygon per cell**
      using that cell’s edges from `grid_info`. This makes polar behave like rect:
        * works with S.1586 subsets, *and* with generic grids (e.g., from `pointgen`).
        * no gaps: every provided cell is filled, colored, and hoverable.
    - Borders for polar are derived from `grid_info`:
        * **Rings** = unique elevation edges (circles in the polar disc).
        * **Meridians** = for each elevation band, radial segments at each unique
          azimuth edge, clipped to that band and to the requested display elevation band.
    - If `grid_info` is None, we keep the fast S.1586 heatmap path (unchanged).
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

    samp_lin = _to_lin(samples) if log_mode else samples

    if mode == "power":
        if not (0.0 < worst_percent < 100.0):
            raise ValueError("worst_percent must be in (0, 100).")
        pct = 100.0 - float(worst_percent)
        vals_lin = np.nanpercentile(samp_lin, pct, axis=0)
        vals = _to_db(vals_lin) if log_mode else vals_lin
        cbar_title = f"Power{'' if unit is None else f' {unit}'}"
    else:
        if protection_criterion is None:
            raise ValueError("protection_criterion is required for mode='data_loss'.")
        if hasattr(protection_criterion, "to") and (unit is not None):
            thr_num = float(protection_criterion.to(unit).value)
        elif hasattr(protection_criterion, "value"):
            thr_num = float(protection_criterion.value)
        else:
            thr_num = float(protection_criterion)
        loss = (np.mean(samp_lin > _to_lin(thr_num), axis=0) * 100.0) if log_mode \
               else (np.mean(samples > thr_num, axis=0) * 100.0)
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
                theta = (np.degrees(np.arctan2(Y, X)) + 360.0) % 360.0

                # Map each pixel to an S.1586 ring and azimuth bin
                ring_idx = np.digitize(el, el_edges, right=True) - 1
                ring_idx = np.clip(ring_idx, 0, len(el_edges) - 2)
                steps = 360.0 / np.take(cells_per_ring, ring_idx)
                az_bin = (theta // steps).astype(int)
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
                            t = np.radians(az)
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
                    t0 = np.radians(lo_az); t1 = np.radians(hi_az)
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
                            t = np.radians(float(L))
                            traces.append(go.Scatter(
                                x=[r0*np.cos(t), r1*np.cos(t)],
                                y=[r0*np.sin(t), r1*np.sin(t)],
                                mode="lines",
                                line=dict(color=border_color, width=line_w),
                                opacity=border_alpha, hoverinfo="skip", showlegend=False, name=""
                            ))

            # 4) Guides (outer unit circle + rays + labels) — unchanged
            annotations: list[dict] = []
            label_points: list[tuple[float, float]] = []
            if (draw_guides is None and projection.lower() == "polar") or (draw_guides is True):
                tt = np.linspace(0, 2*np.pi, 361)
                traces.append(go.Scatter(
                    x=np.cos(tt), y=np.sin(tt),
                    mode="lines", line=dict(color=guide_color, width=3),
                    hoverinfo="skip", showlegend=False, name=""
                ))

                def _ray(angle_deg: float, label: str, extra_dx: float = 0.0):
                    t = np.radians(angle_deg)
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
                    ly = by + np.sin(t) * label_offset_extra
                    label_points.append((lx, ly))
                    annotations.append(dict(
                        x=lx, y=ly, text=label, showarrow=False,
                        xanchor="center", yanchor="bottom",
                        font=dict(color=guide_color, size=14)
                    ))

                nudge = 0.04
                _ray(0,   "Az 0°",   extra_dx=+nudge)
                _ray(90,  "Az 90°")
                _ray(180, "Az 180°", extra_dx=-nudge)
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
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=cmin, vmax=cmax)

    if projection.lower() == "polar":
        # --- polar (MPL) faces already use cell polygons; we keep that.
        from matplotlib.patches import Polygon

        def _cell_poly(az0, az1, el0, el1) -> np.ndarray:
            """Four-vertex polygon for a cell in polar display coordinates."""
            t0, t1 = np.radians([az0, az1])
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
                        t = np.radians(az)
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
                        t = np.radians(float(L))
                        ax.plot([r0*np.cos(t), r1*np.cos(t)],
                                [r0*np.sin(t), r1*np.sin(t)],
                                color=border_color, linewidth=border_width, alpha=border_alpha)

        # Guides (outer circle + rays + labels) — unchanged
        max_r = 1.0
        if (draw_guides is None and projection.lower() == "polar") or (draw_guides is True):
            tt = np.linspace(0, 2*np.pi, 361)
            ax.plot(np.cos(tt), np.sin(tt),
                    color=guide_color, linewidth=guide_linewidth, alpha=guide_alpha)

            def _ray(angle_deg: float, label: str, extra_dx: float = 0.0) -> float:
                t = np.radians(angle_deg)
                bx, by = guide_length*np.cos(t), guide_length*np.sin(t)
                ax.plot([0.0, bx], [0.0, by], color=guide_color,
                        linewidth=guide_linewidth, alpha=guide_alpha)
                if show_axis_arrows:
                    if arrow_direction.lower() == "outward":
                        ax.annotate("", xy=(bx, by), xytext=(0.0, 0.0),
                                    arrowprops=dict(arrowstyle="->", lw=guide_linewidth,
                                                    color=guide_color, shrinkA=0, shrinkB=0,
                                                    mutation_scale=14*arrow_scale))
                    else:
                        ax.annotate("", xy=(0.0, 0.0), xytext=(bx, by),
                                    arrowprops=dict(arrowstyle="->", lw=guide_linewidth,
                                                    color=guide_color, shrinkA=0, shrinkB=0,
                                                    mutation_scale=14*arrow_scale))
                lx = bx + np.cos(t) * label_offset_extra + extra_dx
                ly = by + np.sin(t) * label_offset_extra
                ax.text(lx, ly, label, ha="center", va="bottom",
                        color=guide_color, fontsize=12, weight="bold")
                return max(np.hypot(bx, by), np.hypot(lx, ly))

            nudge = 0.04
            max_r = max(max_r, _ray(0,   "Az 0°",   extra_dx=+nudge))
            max_r = max(max_r, _ray(90,  "Az 90°"))
            max_r = max(max_r, _ray(180, "Az 180°", extra_dx=-nudge))
            max_r = max(max_r, _ray(270, "Az 270°"))

        max_r = max(max_r, guide_length) * (1.0 + float(tight_pad))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([-max_r, max_r])
        ax.set_ylim([-max_r, max_r])
        ax.set_xticks([])
        ax.set_yticks([])

    else:
        # --- rectangular (MPL) --- (left as in your updated version)
        from matplotlib.patches import Polygon

        def _rect(az0, az1, el0, el1) -> np.ndarray:
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

        if (draw_guides is True) or (draw_guides is None and projection.lower() == "rect"):
            def _axis_arrow(x0, y0, x1, y1, label):
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
    ax.set_title(title or f"S.1586-1 Hemisphere — {'Power' if mode=='power' else 'Data loss'} ({projection})")
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Power" + (f" {unit}" if (mode == "power" and unit is not None) else "Data loss [%]"))

    fig.tight_layout()
    if show:
        plt.show()
    else:
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
    log_mode: bool = True,                      # percentile/threshold in linear if data are in dB

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
    html_include_plotlyjs: bool | str = True,   # True bundles JS; "cdn" uses CDN
    html_auto_open: bool = False,
    export_png_path: str | None = None,         # needs kaleido
    png_width: int = 1600,
    png_height: int = 1200,
    png_scale: float = 1.5,
):
    """
    Render the ITU-R S.1586-1 hemisphere in 3D and colour each sky cell by either:
      • mode="power"     → per-cell (100 - worst_percent)th percentile, or
      • mode="data_loss" → per-cell % of samples above a threshold.

    Data layouts:
      1) Full grid:  data[..., 2334] with no grid_info.
      2) Subset:     data[..., K]    with grid_info length K (each row has exact az/el edges).

    `elev_range=(lo, hi)` crops only what’s displayed (no rebin). If the request extends
    beyond your subset’s coverage you’ll get a warning and the view clamps to the covered band.

    Hover:
      • Plotly: one invisible point per cell sits slightly in front of the sphere
        (see `hover_offset_3d`). Each point has a hoverlabel whose background is the
        cell’s colour, just like the 2D version.
      • Matplotlib: a small annotation with the cell’s colour as the background,
        offset by `hover_xytext` pixels so it doesn’t hide the target cell.
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
        if unit is not None and hasattr(protection_criterion, "to"):
            thr_num = float(protection_criterion.to(unit).value)
        elif hasattr(protection_criterion, "value"):
            thr_num = float(protection_criterion.value)
        else:
            thr_num = float(protection_criterion)

        if log_mode:
            thr_lin = _to_lin(thr_num)
            vals = (_to_lin(arr_kept) > thr_lin).mean(axis=0) * 100.0
        else:
            vals = (arr_kept > thr_num).mean(axis=0) * 100.0

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
        if log_mode:
            cell_vals = _to_db(np.nanpercentile(_to_lin(arr_kept), percentile, axis=0))
        else:
            cell_vals = np.nanpercentile(arr_kept, percentile, axis=0)

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
        if use_data_loss:
            return f"{v:.{hover_precision}f} %"
        return f"{v:.{hover_precision}f}{'' if unit is None else ' ' + str(unit)}"

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

            # Centroid → normalise to sphere → nudge along camera eye direction
            cx, cy, cz = float(np.mean(x4)), float(np.mean(y4)), float(np.mean(z4))
            norm = np.sqrt(cx*cx + cy*cy + cz*cz) + 1e-15
            cx, cy, cz = cx / norm, cy / norm, cz / norm
            cx += hover_offset_3d * eye_norm[0]
            cy += hover_offset_3d * eye_norm[1]
            cz += hover_offset_3d * eye_norm[2]
            centroids_xyz.append((cx, cy, cz))

            # Per-cell hover label and colours
            rgba, rgb01 = _rgba_from_value(v, cmin, cmax, cmap, alpha_val=1.0)
            font_col = _hover_font_color_from_rgb(rgb01)
            label = (
                f"Az: {az0:.1f}–{az1:.1f}°<br>"
                f"El: {el0:.1f}–{el1:.1f}°<br>"
                f"{'Data loss' if use_data_loss else 'Power'}: {_fmt_val(v)}"
            )
            cell_text.append(label)
            cell_bg.append(rgba)
            cell_font_color.append(font_col)

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

        # Invisible hover points with coloured labels (one per cell)
        if show_hover and len(centroids_xyz) > 0:
            for (cx, cy, cz), text, bg, fcol in zip(centroids_xyz, cell_text, cell_bg, cell_font_color):
                traces.append(go.Scatter3d(
                    x=[cx], y=[cy], z=[cz],
                    mode="markers",
                    marker=dict(size=int(hover_marker_size), opacity=0.0),
                    text=[text],
                    hovertemplate="%{text}<extra></extra>",
                    hoverlabel=dict(bgcolor=bg, bordercolor=bg, font=dict(color=fcol, size=14), align="left"),
                    showlegend=False,
                    name=""
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
            include_js = html_include_plotlyjs
            if isinstance(include_js, str):
                include_js = (include_js.lower() != "cdn")
            fig_p.write_html(export_html_path, include_plotlyjs=bool(include_js), full_html=True)
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

    fig = plt.figure(figsize=figsize)
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
    ax.set_title(title or default_title)

    # Colorbar
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.02)
        cbar.set_label(colorbar_label)

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
            r, g, b, a = rgba
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return "black" if y > 0.60 else "white"

        def _on_move(event):
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

