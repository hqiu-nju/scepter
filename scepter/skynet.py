#!/usr/bin/env python3

"""
skynet.py

This is the module for generating the sky grid

Author: Harry Qiu <hqiu678@outlook.com>
Collaborator: Boris Sorokin <mralin@protonmail.com>
Date: 12-03-2024
"""

import cysgp4
from cysgp4 import PyTle, PyObserver
from cysgp4 import get_example_tles, propagate_many
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pycraf
from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry
from astropy import units as u, constants as const
from astropy.time import Time
from astropy.utils.misc import NumpyRNGContext
from functools import lru_cache
from typing import Tuple, Sequence

def pointgen_S_1586_1(
    niters: int = 1,
    rnd_seed: int | None = None,
    elev_range: Tuple[float | u.Quantity, float | u.Quantity] = (0 * u.deg, 90 * u.deg),
):
    """
    Generate random pointings (azimuth, elevation) inside the ITU-R S.1586-1 sky-cell grid
    (Table 1) and return both the sampled directions and a per-cell descriptor array.

    Overview
    --------
    - ITU-R S.1586-1 partitions the **upper hemisphere** (0°..90° elevation) into 30 rings
      of 3° each (edges at 0, 3, …, 87, 90). Each ring is subdivided in azimuth with a
      ring-specific step (3°, 4°, 5°, …), resulting in **exactly 2334** cells overall.
    - This function:
        1) Builds (and caches) the full S.1586-1 grid exactly as per Table 1;
        2) **Restricts** it to a user-provided elevation interval `elev_range` (dropping
           non-overlapping cells and **clipping** partially overlapping cells to the
           intersection);
        3) Draws `niters` random directions **uniform in solid angle** within each
           retained (possibly clipped) cell.

    Uniform-in-solid-angle sampling
    --------------------------------
    For a cell bounded by (az_low..az_high, el_low..el_high), we sample
      • φ (azimuth) ~ Uniform[az_low, az_high]
      • z = cos(zenith) ~ Uniform[sin(el_low), sin(el_high)]
    and convert back to elevation: `el = 90° - arccos(z)`.
    This yields uniform density over solid angle within the (clipped) cell.

    Parameters
    ----------
    niters : int, default 1
        Number of random pointings **per retained cell**.
        Output arrays have shape (niters, n_cells_kept).
    rnd_seed : int or None, default None
        Random seed for reproducibility (wrapped in `astropy.utils.NumpyRNGContext`).
        If `None`, the global RNG state is used.
    elev_range : (float/Quantity, float/Quantity), default (0 deg, 90 deg)
        Elevation interval (0° = horizon, 90° = zenith) to include.
        Accepts bare floats (assumed degrees) or `astropy` quantities.
        The pair is sorted if reversed and **clamped** to [0°, 90°].
        Cells outside this interval are discarded; cells crossing it are **clipped**
        to the intersection before sampling and area computation.

    Returns
    -------
    tel_az_deg : np.ndarray, shape (niters, n_cells_kept)
        Random azimuth samples in **degrees**.
    tel_el_deg : np.ndarray, shape (niters, n_cells_kept)
        Random elevation samples in **degrees**.
    grid_info : np.ndarray (structured), length = n_cells_kept
        One row per retained (clipped) cell with fields (all **degrees** except `solid_angle`):
          - 'cell_lon'      : cell center longitude
          - 'cell_lat'      : cell center **elevation** (name retained for compatibility)
          - 'cell_lon_low'  : cell longitude lower edge
          - 'cell_lon_high' : cell longitude upper edge
          - 'cell_lat_low'  : cell elevation lower edge (after clipping)
          - 'cell_lat_high' : cell elevation upper edge (after clipping)
          - 'solid_angle'   : cell area in **square degrees**
            (Recall: 1 sr = (180/π)^2 ≈ 3282.80635 deg²)

    Raises
    ------
    ValueError
        If `niters` is not a positive integer or `elev_range` is not a 2-tuple.

    Notes
    -----
    - The S.1586-1 grid is constructed **exactly** as in Table 1; we verify the per-ring
      cell counts and total (2334) as guardrails.
    - Grid construction is cached for the Python session (`lru_cache(maxsize=1)`).
    - If the effective `elev_range` collapses after clamping (e.g., entirely out of
      [0°, 90°]), the function returns empty arrays of shape (niters, 0) and an empty
      `grid_info` with the expected dtype.

    Example
    -------
    >>> # Keep only directions between 30° and 80° elevation:
    >>> az, el, info = pointgen_S_1586_1(
    ...     niters=100, rnd_seed=42, elev_range=(30*u.deg, 80*u.deg)
    ... )
    >>> az.shape, el.shape, info.shape
    ((100, info.shape[0]), (100, info.shape[0]), (info.shape[0],))
    """
    # -------------------------------------------------------------------------
    # 0) Validate & normalize inputs
    # -------------------------------------------------------------------------
    if not isinstance(niters, int) or niters <= 0:
        raise ValueError("`niters` must be a positive integer.")

    # Parse and normalize elev_range to sorted, clamped float degrees in [0, 90]
    if not isinstance(elev_range, Sequence) or len(elev_range) != 2:
        raise ValueError("`elev_range` must be a 2-tuple like (low, high).")

    el_lo, el_hi = elev_range
    el_lo_deg = float(el_lo.to_value(u.deg)) if hasattr(el_lo, "to") else float(el_lo)
    el_hi_deg = float(el_hi.to_value(u.deg)) if hasattr(el_hi, "to") else float(el_hi)

    # Sort if reversed
    if el_lo_deg > el_hi_deg:
        el_lo_deg, el_hi_deg = el_hi_deg, el_lo_deg

    # Clamp to physically valid domain [0°, 90°]
    el_lo_deg = max(0.0, el_lo_deg)
    el_hi_deg = min(90.0, el_hi_deg)

    # Degenerate range? -> empty outputs with correct shapes and dtype
    empty_dtype = np.dtype([
        ('cell_lon', np.float64), ('cell_lat', np.float64),
        ('cell_lon_low', np.float64), ('cell_lon_high', np.float64),
        ('cell_lat_low', np.float64), ('cell_lat_high', np.float64),
        ('solid_angle', np.float64),
    ])
    if not (el_lo_deg < el_hi_deg):
        with NumpyRNGContext(rnd_seed):
            tel_az = np.empty((niters, 0), dtype=np.float64)
            tel_el = np.empty((niters, 0), dtype=np.float64)
        grid_info = np.empty(0, dtype=empty_dtype)
        return tel_az, tel_el, grid_info

    # -------------------------------------------------------------------------
    # 1) Build & cache the full S.1586-1 grid (upper hemisphere)
    # -------------------------------------------------------------------------
    @lru_cache(maxsize=1)
    def _get_cached_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Construct S.1586-1 cell boundaries once and cache them.

        Returns
        -------
        cell_az_low_bc, cell_az_high_bc : (1, N) float64 arrays (degrees)
            Per-cell azimuth lower/upper edges; leading singleton dim for broadcasting.
        cell_el_low_bc, cell_el_high_bc : (1, N) float64 arrays (degrees)
            Per-cell elevation lower/upper edges; leading singleton dim for broadcasting.
        n_total_cells : int
            Total number of cells (must be 2334).

        Implementation details
        ----------------------
        - Rings are 3° thick with edges at 0, 3, …, 90 → 30 rings.
        - Ring-specific azimuth steps (degrees) per Table 1 are encoded below.
        - We assert the per-ring counts and the total number of cells.
        """
        # Table-1 azimuth steps keyed by the ring's lower elevation edge (degrees)
        s1586_az_steps = {
              0: 3,   3: 3,   6: 3,   9: 3,  12: 3,  15: 3,  18: 3,  21: 3,  24: 3,
             27: 3,
             30: 4,  33: 4,  36: 4,  39: 4,  42: 4,
             45: 4,
             48: 5,  51: 5,
             54: 5,
             57: 6,  60: 6,
             63: 6,
             66: 8,
             69: 9,
             72: 10,
             75: 12,
             78: 18,
             81: 24,
             84: 40,
             87: 120,
        }

        # Elevation edges (degrees): 0, 3, ..., 87, 90  →  30 rings
        s1586_el_edges_deg = np.arange(0, 90 + 3, 3, dtype=np.float64)
        n_rings = s1586_el_edges_deg.size - 1

        # Pre-compute expected counts and total cells
        expected_counts = [120]*10 + [90]*6 + [72]*3 + [60]*3 + [45, 40, 36, 30, 20, 15, 9, 3]
        n_total_cells = sum(expected_counts)  # 2334
        
        # Pre-allocate arrays for better performance
        cell_az_low = np.empty(n_total_cells, dtype=np.float64)
        cell_az_high = np.empty(n_total_cells, dtype=np.float64)
        cell_el_low = np.empty(n_total_cells, dtype=np.float64)
        cell_el_high = np.empty(n_total_cells, dtype=np.float64)
        calculated_cells_per_ring = np.empty(n_rings, dtype=np.int32)

        # Build each ring
        idx = 0
        for i in range(n_rings):
            el_low = float(s1586_el_edges_deg[i])
            el_high = float(s1586_el_edges_deg[i + 1])

            # Lookup & validate azimuth step (must divide 360 exactly)
            az_step_deg = s1586_az_steps.get(int(el_low))
            if az_step_deg is None:
                raise ValueError(
                    f"Implementation error: azimuth step undefined for ring starting at {el_low}°."
                )
            if 360 % az_step_deg != 0:
                raise ValueError(
                    f"Invalid azimuth step {az_step_deg}° for ring {el_low}–{el_high}° (360 not divisible)."
                )

            # Cells in this ring and its azimuth edges
            n_cells_in_ring = 360 // az_step_deg
            calculated_cells_per_ring[i] = n_cells_in_ring
            az_edges_deg = np.arange(n_cells_in_ring + 1, dtype=np.float64) * az_step_deg

            # Vectorized assignment instead of loop appends
            cell_az_low[idx:idx+n_cells_in_ring] = az_edges_deg[:-1]
            cell_az_high[idx:idx+n_cells_in_ring] = az_edges_deg[1:]
            cell_el_low[idx:idx+n_cells_in_ring] = el_low
            cell_el_high[idx:idx+n_cells_in_ring] = el_high
            idx += n_cells_in_ring

        # Verification against Table 1
        if not np.array_equal(calculated_cells_per_ring, expected_counts):
            raise RuntimeError("Implementation error: per-ring counts do not match S.1586-1 Table 1.")
        if idx != n_total_cells:
            raise RuntimeError(f"Implementation error: total cells ({idx}) != expected 2334.")

        # Convert to broadcastable arrays (1, N)
        cell_az_low_bc  = cell_az_low[np.newaxis, :]
        cell_az_high_bc = cell_az_high[np.newaxis, :]
        cell_el_low_bc  = cell_el_low[np.newaxis, :]
        cell_el_high_bc = cell_el_high[np.newaxis, :]

        return cell_az_low_bc, cell_az_high_bc, cell_el_low_bc, cell_el_high_bc, n_total_cells

    # Retrieve cached full-sky arrays
    (cell_az_low_bc,
     cell_az_high_bc,
     cell_el_low_bc,
     cell_el_high_bc,
     _n_total_cells) = _get_cached_data()

    # -------------------------------------------------------------------------
    # 2) Restrict grid to the requested elevation range: filter + clip
    # -------------------------------------------------------------------------
    # Keep only cells that strictly overlap with [el_lo_deg, el_hi_deg]
    # Overlap condition: (cell_high > range_low) & (cell_low < range_high)
    keep_mask = (cell_el_high_bc > el_lo_deg) & (cell_el_low_bc < el_hi_deg)
    if keep_mask.ndim == 2:
        keep_mask = keep_mask[0]  # (1, N) → (N,)

    n_kept = int(np.count_nonzero(keep_mask))
    if n_kept == 0:
        with NumpyRNGContext(rnd_seed):
            tel_az = np.empty((niters, 0), dtype=np.float64)
            tel_el = np.empty((niters, 0), dtype=np.float64)
        grid_info = np.empty(0, dtype=empty_dtype)
        return tel_az, tel_el, grid_info

    # Slice retained cells
    az_low_kept   = cell_az_low_bc[:, keep_mask]     # (1, K)
    az_high_kept  = cell_az_high_bc[:, keep_mask]    # (1, K)
    el_low_kept   = cell_el_low_bc[:, keep_mask]     # (1, K)
    el_high_kept  = cell_el_high_bc[:, keep_mask]    # (1, K)

    # Clip elevation bounds to intersection with [el_lo_deg, el_hi_deg]
    eff_el_low  = np.maximum(el_low_kept,  el_lo_deg)  # (1, K)
    eff_el_high = np.minimum(el_high_kept, el_hi_deg)  # (1, K)

    # Sanity: ensure strictly positive elevation span
    positive_span = (eff_el_high > eff_el_low)
    if not np.all(positive_span):
        m2 = positive_span[0]
        az_low_kept   = az_low_kept[:, m2]
        az_high_kept  = az_high_kept[:, m2]
        eff_el_low    = eff_el_low[:, m2]
        eff_el_high   = eff_el_high[:, m2]
        n_kept = az_low_kept.shape[1]

    # -------------------------------------------------------------------------
    # 3) Compute per-cell solid angle (deg²) and centers for grid_info
    # -------------------------------------------------------------------------
    # Δλ (deg) for each cell is simply (lon_high − lon_low) after slicing
    delta_lambda_deg = (az_high_kept - az_low_kept)[0]                 # (K,)
    # Δsin(el) for the (clipped) elevation span
    delta_sin = np.sin(np.radians(eff_el_high[0])) - np.sin(np.radians(eff_el_low[0]))  # (K,)
    # Solid angle in deg²: Ω_deg2 = Δλ_deg * degrees(Δsin)
    area_deg2 = delta_lambda_deg * np.degrees(delta_sin)               # (K,)

    # Cell centers (degrees); note: centers ignore 0/360 wrap, which is fine
    cell_lon_mid = 0.5 * (az_low_kept[0] + az_high_kept[0])            # (K,)
    cell_el_mid  = 0.5 * (eff_el_low[0] + eff_el_high[0])              # (K,)

    # Flatten the (1, K) arrays to (K,) for table construction
    cell_lon_low  = az_low_kept[0].astype(np.float64)
    cell_lon_high = az_high_kept[0].astype(np.float64)
    cell_el_low   = eff_el_low[0].astype(np.float64)
    cell_el_high  = eff_el_high[0].astype(np.float64)

    # Assemble structured grid_info; keep field names aligned with `pointgen`
    grid_matrix = np.column_stack([
        cell_lon_mid,            # 'cell_lon'
        cell_el_mid,             # 'cell_lat' (elevation)
        cell_lon_low,            # 'cell_lon_low'
        cell_lon_high,           # 'cell_lon_high'
        cell_el_low,             # 'cell_lat_low'
        cell_el_high,            # 'cell_lat_high'
        area_deg2,               # 'solid_angle' (deg^2)
    ])
    grid_dtype = empty_dtype  # same layout as `pointgen`
    grid_info = np.empty(grid_matrix.shape[0], dtype=grid_dtype)
    for i, name in enumerate(grid_dtype.names):
        grid_info[name] = grid_matrix[:, i]

    # -------------------------------------------------------------------------
    # 4) Sample uniformly within each (clipped) cell
    # -------------------------------------------------------------------------
    # Convert clipped elevation bounds to z = cos(zenith) bounds
    # (z = cos(90° − el) = sin(el))
    z_low  = np.cos(np.radians(90.0 - eff_el_low))    # (1, K)
    z_high = np.cos(np.radians(90.0 - eff_el_high))   # (1, K)
    z_bound_lower = np.minimum(z_low,  z_high)        # (1, K)
    z_bound_upper = np.maximum(z_low,  z_high)        # (1, K)

    with NumpyRNGContext(rnd_seed):
        u_az = np.random.uniform(0.0, 1.0, size=(niters, n_kept))
        u_z  = np.random.uniform(0.0, 1.0, size=(niters, n_kept))

    tel_az_deg = az_low_kept + u_az * (az_high_kept - az_low_kept)     # (niters, K)
    sampled_z  = z_bound_lower + u_z * (z_bound_upper - z_bound_lower) # (niters, K)
    sampled_z  = np.clip(sampled_z, -1.0, 1.0)                          # numerical safety
    tel_el_deg = 90.0 - np.degrees(np.arccos(sampled_z))                # (niters, K)

    # -------------------------------------------------------------------------
    # 5) Return the sampled directions and the per-cell descriptor table
    # -------------------------------------------------------------------------
    return tel_az_deg, tel_el_deg, grid_info
    
def pointgen(
    niters: int,
    step_size: u.Quantity | float = 3 * u.deg,
    elev_range: Tuple[u.Quantity | float, u.Quantity | float] = (0 * u.deg, 90 * u.deg),
    rnd_seed: int | None = None,
):
    """
    Generate random pointings (azimuth, elevation) over a quasi-equal-area grid
    covering a user-specified elevation range, with sampling uniform in solid angle.

    Grid construction
    -----------------
    1) The elevation (0°=horizon, 90°=zenith) interval `elev_range` is split into
       bands of ~`step_size` thickness (the last band can be narrower to land exactly
       on the upper bound).
    2) Each band is divided in longitude into
           n_lon ≈ round( 360° * cos(mid_elev) / step_size )
       so longitudinal width shrinks toward the zenith, roughly equalizing cell areas.
    3) Inside each cell, we draw:
         - φ (azimuth) ~ Uniform[lon_low, lon_high]
         - z = cos(zenith) ~ Uniform[sin(el_low), sin(el_high)]
       and convert z back to elevation. This produces **uniform-in-solid-angle** samples.

    Parameters
    ----------
    niters : int
        Number of random pointings to generate **per cell**.
        Output az/el arrays have shape (niters, n_cells).
    step_size : float or astropy.units.Quantity, default 3 deg
        Target angular step (degrees). Controls elevation band thickness and
        the *approximate* azimuthal span of cells. Smaller -> more (smaller) cells.
        May be a bare float (interpreted as degrees) or a Quantity.
    elev_range : (float/Quantity, float/Quantity), default (0 deg, 90 deg)
        Elevation interval to cover, inclusive of endpoints. Accepts numbers
        (assumed degrees) or Quantities. Values outside [0°, 90°] are clamped.
        If the effective range is empty, empty outputs are returned.
    rnd_seed : int or None, default None
        Seed for reproducible sampling (via `astropy.utils.NumpyRNGContext`).
        If None, NumPy's global RNG state is used as-is.

    Returns
    -------
    tel_az_deg : np.ndarray, shape (niters, n_cells)
        Azimuth samples in **degrees** for all cells.
    tel_el_deg : np.ndarray, shape (niters, n_cells)
        Elevation samples in **degrees** for all cells.
    grid_info : np.ndarray (structured), length = n_cells
        One row per cell with the following fields (all **degrees** except `solid_angle`):
          - 'cell_lon'      : cell center longitude
          - 'cell_lat'      : cell center **elevation** (name kept for compatibility)
          - 'cell_lon_low'  : cell longitude lower edge
          - 'cell_lon_high' : cell longitude upper edge
          - 'cell_lat_low'  : cell elevation lower edge
          - 'cell_lat_high' : cell elevation upper edge
          - 'solid_angle'   : cell area in **square degrees**
            (Recall: 1 sr = (180/π)^2 ≈ 3282.80635 deg²)

    Notes
    -----
    - Elevation vs. “lat”: Field names retain “lat” for backward-compatibility,
      but they represent **elevation** (0°..90°).
    - Uniform solid-angle sampling avoids clustering at high elevations.
    - Near zenith, cos(mid_elev) → 0, so we enforce at least one longitude cell.

    Example
    -------
    >>> az, el, info = pointgen(
    ...     niters=100,
    ...     step_size=3*u.deg,
    ...     elev_range=(20*u.deg, 80*u.deg),
    ...     rnd_seed=123,
    ... )
    >>> az.shape, el.shape, info.shape
    ((100, info.shape[0]), (100, info.shape[0]), (info.shape[0],))
    """
    # ----------------------------
    # 0) Validate & normalize inputs
    # ----------------------------
    if not isinstance(niters, int) or niters <= 0:
        raise ValueError("`niters` must be a positive integer.")

    # Normalize step_size to a plain float (degrees)
    step_size_deg = (
        float(step_size.to_value(u.deg)) if hasattr(step_size, "to") else float(step_size)
    )
    if not np.isfinite(step_size_deg) or step_size_deg <= 0:
        raise ValueError("`step_size` must be a positive, finite angle in degrees.")

    # Normalize elev_range (accepts numbers or Quantities)
    if not isinstance(elev_range, Sequence) or len(elev_range) != 2:
        raise ValueError("`elev_range` must be a 2-tuple like (low, high).")

    el_low = elev_range[0]
    el_high = elev_range[1]
    el_low_deg = float(el_low.to_value(u.deg)) if hasattr(el_low, "to") else float(el_low)
    el_high_deg = float(el_high.to_value(u.deg)) if hasattr(el_high, "to") else float(el_high)

    # Sort if reversed
    if el_low_deg > el_high_deg:
        el_low_deg, el_high_deg = el_high_deg, el_low_deg

    # Clamp to [0, 90]
    el_low_deg = max(0.0, el_low_deg)
    el_high_deg = min(90.0, el_high_deg)

    # Empty/degenerate range → return consistent empties
    if not (el_low_deg < el_high_deg):
        empty_az = np.empty((niters, 0), dtype=np.float64)
        empty_el = np.empty((niters, 0), dtype=np.float64)
        empty_info = np.empty(0, dtype=[
            ('cell_lon', np.float64), ('cell_lat', np.float64),
            ('cell_lon_low', np.float64), ('cell_lon_high', np.float64),
            ('cell_lat_low', np.float64), ('cell_lat_high', np.float64),
            ('solid_angle', np.float64),
        ])
        return empty_az, empty_el, empty_info

    # ---------------------------------------------
    # 1) Build elevation band edges
    #    Use arange-style progress to avoid cumulative rounding drift.
    #    Ensure the exact top bound is included; last band may be narrower.
    # ---------------------------------------------
    edge_elevs = np.arange(el_low_deg, el_high_deg, step_size_deg, dtype=np.float64)
    if edge_elevs.size == 0 or not np.isclose(edge_elevs[-1], el_high_deg):
        edge_elevs = np.append(edge_elevs, el_high_deg)

    # Midpoint elevations for bands
    if edge_elevs.size < 2:
        edge_elevs = np.array([el_low_deg, el_high_deg], dtype=np.float64)
    mid_elevs = 0.5 * (edge_elevs[:-1] + edge_elevs[1:])

    # ---------------------------------------------
    # 2) Helper: sample uniformly in solid angle within a cell
    # ---------------------------------------------
    def _sample_cell(n: int, lon_lo: float, lon_hi: float, el_lo: float, el_hi: float):
        """
        Draw `n` samples with φ ~ U[lon_lo, lon_hi] and
        z = cos(zenith) ~ U[sin(el_lo), sin(el_hi)], then convert to elevation.
        Returns (az_deg, el_deg).
        """
        # Bounds for z = cos(zenith) = sin(elevation)
        z_low, z_high = np.sin(np.radians(el_lo)), np.sin(np.radians(el_hi))

        # Uniform azimuth in the cell
        az = np.random.uniform(lon_lo, lon_hi, size=n)

        # Uniform z → elevation
        z_samp = np.random.uniform(z_low, z_high, size=n)
        z_samp = np.clip(z_samp, -1.0, 1.0)  # numerical safety for arccos
        el = 90.0 - np.degrees(np.arccos(z_samp))
        return az.astype(np.float64), el.astype(np.float64)

    # ---------------------------------------------
    # 3) Loop over elevation bands; tile in longitude per band
    # ---------------------------------------------
    cell_edges: list[tuple[float, float, float, float]] = []   # (lon_lo, lon_hi, el_lo, el_hi)
    cell_mids: list[tuple[float, float]] = []                  # (lon_mid, el_mid)
    solid_angles_deg2: list[float] = []                        # deg^2 per cell
    tel_az_cells: list[np.ndarray] = []
    tel_el_cells: list[np.ndarray] = []

    with NumpyRNGContext(rnd_seed):
        for el_lo, el_mid, el_hi in zip(edge_elevs[:-1], mid_elevs, edge_elevs[1:]):
            # Number of longitude cells for quasi-equal-area behavior.
            # `round` + ≥1 ensures we do not collapse near zenith.
            n_lon = int(np.round(360.0 * np.cos(np.radians(el_mid)) / step_size_deg))
            n_lon = max(1, n_lon)

            # Uniform longitudinal edges and centers across the band
            edge_lons = np.linspace(0.0, 360.0, n_lon + 1, endpoint=True, dtype=np.float64)
            mid_lons = 0.5 * (edge_lons[:-1] + edge_lons[1:])

            # Cell solid angle (deg^2):
            # Ω_sr = Δλ_rad * (sin el_hi - sin el_lo)
            # Convert sr → deg^2 by (180/π)^2; algebra yields:
            # Ω_deg2 = Δλ_deg * degrees(sin el_hi - sin el_lo)
            delta_lambda_deg = edge_lons[1] - edge_lons[0]          # uniform width
            delta_sin = np.sin(np.radians(el_hi)) - np.sin(np.radians(el_lo))
            area_deg2 = delta_lambda_deg * np.degrees(delta_sin)

            for lon_lo, lon_mid, lon_hi in zip(edge_lons[:-1], mid_lons, edge_lons[1:]):
                cell_edges.append((lon_lo, lon_hi, el_lo, el_hi))
                cell_mids.append((lon_mid, el_mid))
                solid_angles_deg2.append(float(area_deg2))

                # Draw samples in this cell (uniform-in-solid-angle)
                cell_az, cell_el = _sample_cell(niters, lon_lo, lon_hi, el_lo, el_hi)
                tel_az_cells.append(cell_az)
                tel_el_cells.append(cell_el)

    # ---------------------------------------------
    # 4) Collate outputs with consistent shapes
    # ---------------------------------------------
    if len(tel_az_cells) == 0:
        tel_az = np.empty((niters, 0), dtype=np.float64)
        tel_el = np.empty((niters, 0), dtype=np.float64)
        grid_info = np.empty(0, dtype=[
            ('cell_lon', np.float64), ('cell_lat', np.float64),
            ('cell_lon_low', np.float64), ('cell_lon_high', np.float64),
            ('cell_lat_low', np.float64), ('cell_lat_high', np.float64),
            ('solid_angle', np.float64),
        ])
        return tel_az, tel_el, grid_info

    # (n_cells, niters) -> (niters, n_cells)
    tel_az = np.asarray(tel_az_cells, dtype=np.float64).T
    tel_el = np.asarray(tel_el_cells, dtype=np.float64).T

    # Build structured grid_info (names retained for compatibility: "lat" == elevation)
    cell_mids_arr = np.asarray(cell_mids, dtype=np.float64)         # (n_cells, 2)
    cell_edges_arr = np.asarray(cell_edges, dtype=np.float64)       # (n_cells, 4)
    solid_angles_arr = np.asarray(solid_angles_deg2, dtype=np.float64)[:, None]  # (n_cells, 1)

    grid_matrix = np.column_stack([cell_mids_arr, cell_edges_arr, solid_angles_arr])
    grid_dtype = np.dtype([
        ('cell_lon', np.float64), ('cell_lat', np.float64),          # 'cell_lat' is elevation
        ('cell_lon_low', np.float64), ('cell_lon_high', np.float64),
        ('cell_lat_low', np.float64), ('cell_lat_high', np.float64),
        ('solid_angle', np.float64),  # deg^2
    ])
    grid_info = np.empty(grid_matrix.shape[0], dtype=grid_dtype)
    for i, name in enumerate(grid_dtype.names):
        grid_info[name] = grid_matrix[:, i]

    return tel_az, tel_el, grid_info

def gridmatch(az,el,grid_info):
    ### get the grid for a fixed list of az and el pointings
    # Vectorized approach: broadcast comparisons for all grids at once
    cell_lon_low = grid_info['cell_lon_low'][:, np.newaxis]  # (n_grids, 1)
    cell_lon_high = grid_info['cell_lon_high'][:, np.newaxis]
    cell_lat_low = grid_info['cell_lat_low'][:, np.newaxis]
    cell_lat_high = grid_info['cell_lat_high'][:, np.newaxis]
    
    # Broadcast comparisons: (n_grids, n_points)
    azmask = (az >= cell_lon_low) & (az <= cell_lon_high)
    elmask = (el >= cell_lat_low) & (el <= cell_lat_high)
    grid_indx = azmask & elmask
    
    # Find grids that have at least one match
    used_grids = np.where(grid_indx.sum(axis=1) > 0)[0]
    return used_grids, grid_indx[used_grids]

def plantime(epochs,cadence,trange,tint,startdate=cysgp4.PyDateTime()):
    '''
    Description: This function generates the time grid for the simulation

    Parameters:

    epochs: astropy quantity
        number of time steps
    cadence: astropy quantity
        cadence between epochs
    trange: astropy quantity
        time range of the simulation
    tint: astropy quantity  
        sample integration time of the simulation 
    startdate: cysgp4 PyDateTime object
        start date of the simulation, default cysgp4.PyDateTime() for current date and time

    Returns:
    mjds: numpy array
        a 2d array of time intervals for the simulation, first dimension is the number of epochs, 
        second dimension is the separate time stamps for each integration time sample, in MJD
    '''
    pydt = startdate ## take current date and time
    start_mjd=pydt.mjd  ## get mjd step
    niters = epochs

    start_times_window = cadence

    time_range, time_resol = trange.to_value(u.s), tint.to_value(u.s)  # seconds


    start_times = start_mjd + np.arange(epochs) * start_times_window.to_value(u.day)
    td = np.arange(0, time_range, time_resol) *u.s
    td = td.to_value(u.day)
    mjds = np.array(start_times[np.newaxis,np.newaxis,np.newaxis, :, np.newaxis,np.newaxis] + 
                td[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis])
    return mjds

def plotgrid(val, grid_info,  point_az=[], point_el=[],elmin=30, elmax=85,zlabel='PFD average / cell [dB(W/m2)]',xlabel='Azimuth [deg]',ylabel='Elevation [deg]',azmin=0,azmax=360):
    fig = plt.figure(figsize=(12, 4))
    # val = pfd_avg.to_value(cnv.dB_W_m2)
    vmin, vmax = val.min(), val.max()
    val_norm = (val - vmin) / (vmax - vmin)
    plt.bar(
        grid_info['cell_lon_low'],
        height=grid_info['cell_lat_high'] - grid_info['cell_lat_low'],
        width=grid_info['cell_lon_high'] - grid_info['cell_lon_low'],
        bottom=grid_info['cell_lat_low'],
        color=plt.cm.viridis(val_norm),
        align='edge'
        )
    plt.scatter(point_az,point_el,c='r',s=1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(zlabel)
    plt.ylim(elmin, elmax)
    plt.xlim(azmin,azmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show() ### don't show here just load figure into matplotlib
