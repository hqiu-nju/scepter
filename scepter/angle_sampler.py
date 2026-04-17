"""
Angle-sampling helpers for SCEPTer workflows.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>

This module remains part of the GPLv3-licensed SCEPTer project. The author
attribution above does not change the project-wide license.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
except Exception:  # pragma: no cover
    plt = None
    Line2D = None
    LinearSegmentedColormap = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None
    make_subplots = None

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = None

try:
    from tqdm.auto import tqdm as _tqdm_auto
except Exception:  # pragma: no cover
    _tqdm_auto = None

try:
    from astropy import units as u
except Exception:  # pragma: no cover
    u = None

# Optional acceleration: used only for smoothing (never required for loading/sampling)
try:
    import numba as nb
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    nb = None
    _HAVE_NUMBA = False

# -----------------------------------------------------------------------------
# Optional acceleration thresholds
# -----------------------------------------------------------------------------
# These thresholds are intentionally conservative:
# - Below threshold: pure NumPy usually wins because JIT dispatch overhead and
#   temporary allocation costs dominate.
# - Above threshold: JIT kernels tend to be faster and/or lower-memory.
#
# The exact crossover depends on CPU, memory bandwidth and array contiguity.
# These defaults are selected to be robust across developer laptops and CI VMs.
_NUMBA_MIN_SKYCELL_ROWS = 50_000
_NUMBA_MIN_STREAM_HIST_ROWS = 2_000_000
_NUMBA_MIN_SW_WORK = 250_000  # n_points * n_slices
_STREAM_HIST_AUTO_ROWS = 50_000_000
_FILTER_CHUNK_AUTO_ROWS = 5_000_000


SizeLike = Union[None, int, Tuple[int, ...]]

# Fixed S.1586-1 ring definition (0..90 deg elevation, 3-deg rings).
# We keep it local to this module so the sampler can work without importing
# heavy simulation modules.
_S1586_AZ_STEPS_DEG = np.asarray(
    [3] * 10 + [4] * 6 + [5] * 3 + [6] * 3 + [8, 9, 10, 12, 18, 24, 40, 120],
    dtype=np.int32,
)
_S1586_RING_COUNTS = (360 // _S1586_AZ_STEPS_DEG).astype(np.int32, copy=False)
_S1586_RING_OFFSETS = np.zeros(_S1586_RING_COUNTS.size + 1, dtype=np.int32)
_S1586_RING_OFFSETS[1:] = np.cumsum(_S1586_RING_COUNTS, dtype=np.int32)
_S1586_N_CELLS = int(_S1586_RING_COUNTS.sum())


# ============================================================================
# Utilities
# ============================================================================

def _progress_iter(
    iterable: Any,
    *,
    enabled: bool,
    total: Optional[int] = None,
    desc: Optional[str] = None,
) -> Any:
    """
    Wrap an iterable in tqdm when available and explicitly enabled.
    """
    if bool(enabled) and _tqdm_auto is not None:
        return _tqdm_auto(iterable, total=total, desc=desc, leave=False)
    return iterable

def _to_degrees(arr: Any) -> np.ndarray:
    """
    Convert input to a NumPy array of degrees.

    Supports:
      - astropy Quantity (if astropy is available)
      - array-like assumed already in degrees
    """
    if u is not None and hasattr(arr, "unit") and hasattr(arr, "to_value"):
        return np.asarray(arr.to_value(u.deg), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _normalize_size(size: SizeLike) -> Tuple[Tuple[int, ...], int]:
    """
    Normalize numpy-like `size` into:
      - shape tuple (possibly empty for scalar),
      - total number of samples (product(shape); 1 for scalar).
    """
    if size is None:
        return (), 1
    if isinstance(size, (int, np.integer)):
        n = int(size)
        if n < 0:
            raise ValueError("`size` must be non-negative.")
        return (n,), n
    shape = tuple(int(x) for x in size)
    if any(s < 0 for s in shape):
        raise ValueError("All dimensions in `size` must be non-negative.")
    n = int(np.prod(shape, dtype=np.int64))
    return shape, n


def _finite_mask(beta_deg: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
    return np.isfinite(beta_deg) & np.isfinite(alpha_deg)


def _wrap_alpha_deg(alpha_deg: np.ndarray, alpha_range: Tuple[float, float]) -> np.ndarray:
    """
    Wrap alpha into [alpha_min, alpha_max) using modulo arithmetic.
    Typical use-case: [0, 360).
    """
    a0, a1 = float(alpha_range[0]), float(alpha_range[1])
    width = a1 - a0
    if width <= 0:
        raise ValueError("alpha_range must have positive width.")
    return (alpha_deg - a0) % width + a0


def _filter_range(
    beta_deg: np.ndarray,
    alpha_deg: np.ndarray,
    beta_range: Tuple[float, float],
    alpha_range: Tuple[float, float],
    *,
    out_of_range: str = "drop",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Handle out-of-range samples before histogramming.

    out_of_range:
      - "drop": remove samples outside [range)
      - "clip": clip into [range) (may distort tails)
      - "keep": keep as-is (out-of-range samples will be ignored by histogramming anyway)
    """
    b0, b1 = float(beta_range[0]), float(beta_range[1])
    a0, a1 = float(alpha_range[0]), float(alpha_range[1])

    info = {"dropped": 0}

    if out_of_range == "keep":
        return beta_deg, alpha_deg, info

    if out_of_range == "clip":
        beta_c = np.clip(beta_deg, b0, np.nextafter(b1, b0))
        alpha_c = np.clip(alpha_deg, a0, np.nextafter(a1, a0))
        return beta_c, alpha_c, info

    if out_of_range == "drop":
        m = (beta_deg >= b0) & (beta_deg < b1) & (alpha_deg >= a0) & (alpha_deg < a1)
        info["dropped"] = int(beta_deg.size - int(m.sum()))
        return beta_deg[m], alpha_deg[m], info

    raise ValueError("out_of_range must be one of: 'drop', 'clip', 'keep'.")


def _subsample_two_sets(
    beta: np.ndarray,
    alpha: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build two approximately independent real subsets of size n.

    Prefer: sample 2n without replacement and split.
    Fallback: bootstrap with replacement if not enough data.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")

    if beta.size >= 2 * n:
        idx = rng.choice(beta.size, size=2 * n, replace=False)
        b = beta[idx]
        a = alpha[idx]
        return b[:n], a[:n], b[n:], a[n:]

    idx1 = rng.choice(beta.size, size=n, replace=True)
    idx2 = rng.choice(beta.size, size=n, replace=True)
    return beta[idx1], alpha[idx1], beta[idx2], alpha[idx2]


def _bin_centers_from_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Quantize x into bin centers defined by edges.

    This is used to estimate the "resolution floor" of a histogram model:
      real data vs its quantized version at the same binning resolution.
    """
    edges = np.asarray(edges, dtype=np.float64)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, edges.size - 2)
    left = edges[idx]
    right = edges[idx + 1]
    return 0.5 * (left + right)


def _skycell_id_s1586_numpy(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """
    Map sky directions (az, el) in degrees to ITU-R S.1586-1 sky-cell indices.

    Returns
    -------
    np.ndarray[int32]
        Flat sky-cell index in [0, 2334) for valid directions.
        Invalid rows (non-finite or outside 0..90 deg elevation) are set to -1.
    """
    az = np.asarray(az_deg, dtype=np.float64).reshape(-1)
    el = np.asarray(el_deg, dtype=np.float64).reshape(-1)
    if az.size != el.size:
        raise ValueError("az_deg and el_deg must have identical sizes.")

    out = np.full(az.size, -1, dtype=np.int32)
    m = np.isfinite(az) & np.isfinite(el) & (el >= 0.0) & (el <= 90.0)
    if not np.any(m):
        return out

    az_m = np.remainder(az[m], 360.0)
    el_m = el[m]

    # Ring index (30 rings, each 3 deg thick). Elevation exactly at 90 deg
    # belongs to the top ring [87, 90].
    ring = np.floor(el_m / 3.0).astype(np.int32, copy=False)
    ring = np.minimum(ring, np.int32(_S1586_AZ_STEPS_DEG.size - 1))

    step = _S1586_AZ_STEPS_DEG[ring].astype(np.float64, copy=False)
    ring_counts = _S1586_RING_COUNTS[ring]
    az_bin = np.floor(az_m / step).astype(np.int32, copy=False)
    az_bin = np.minimum(az_bin, ring_counts - 1)

    out[m] = _S1586_RING_OFFSETS[ring] + az_bin
    return out


if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _skycell_id_s1586_numba(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        n = az_deg.size
        out = np.empty(n, dtype=np.int32)
        for i in range(n):
            az = float(az_deg[i])
            el = float(el_deg[i])

            if (not np.isfinite(az)) or (not np.isfinite(el)) or (el < 0.0) or (el > 90.0):
                out[i] = -1
                continue

            az = az % 360.0
            ring = int(el / 3.0)
            if ring > 29:
                ring = 29

            step = int(_S1586_AZ_STEPS_DEG[ring])
            n_az = int(_S1586_RING_COUNTS[ring])
            az_bin = int(az / float(step))
            if az_bin >= n_az:
                az_bin = n_az - 1

            out[i] = int(_S1586_RING_OFFSETS[ring]) + az_bin
        return out


def _skycell_id_s1586(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """
    Runtime dispatcher for S.1586-1 sky-cell mapping.

    Numba path is used automatically when available and the input is large
    enough to amortize JIT dispatch overhead.
    """
    az = np.asarray(az_deg, dtype=np.float64).reshape(-1)
    el = np.asarray(el_deg, dtype=np.float64).reshape(-1)
    if az.size != el.size:
        raise ValueError("az_deg and el_deg must have identical sizes.")

    if _HAVE_NUMBA and az.size >= _NUMBA_MIN_SKYCELL_ROWS:
        try:
            return _skycell_id_s1586_numba(az, el)
        except Exception:
            # Robust fallback in case numba cache/runtime is unavailable.
            pass
    return _skycell_id_s1586_numpy(az, el)


if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _skycell_id_regular_numba(
        az_deg: np.ndarray,
        el_deg: np.ndarray,
        az_step: float,
        el_step: float,
        el_min: float,
        el_max: float,
        n_az: int,
        n_el: int,
    ) -> np.ndarray:
        """
        Numba kernel for regular-grid sky-cell indexing.

        The kernel is scalar-loop based to avoid large temporary vectors and to
        keep peak memory stable when called on very large arrays.
        """
        n = az_deg.size
        out = np.empty(n, dtype=np.int32)
        for i in range(n):
            az = float(az_deg[i])
            el = float(el_deg[i])
            if (not np.isfinite(az)) or (not np.isfinite(el)) or (el < el_min) or (el > el_max):
                out[i] = -1
                continue

            # Keep azimuth in [0, 360).
            az = az % 360.0

            az_bin = int(az / az_step)
            el_bin = int((el - el_min) / el_step)
            if az_bin >= n_az:
                az_bin = n_az - 1
            if el_bin >= n_el:
                el_bin = n_el - 1
            if az_bin < 0 or el_bin < 0:
                out[i] = -1
                continue

            out[i] = el_bin * n_az + az_bin
        return out


def _skycell_id_regular(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    *,
    az_step_deg: float,
    el_step_deg: float,
    el_min_deg: float,
    el_max_deg: float,
) -> Tuple[np.ndarray, int, int]:
    """
    Map sky directions (az, el) to a regular az/el grid.

    Grid convention
    ---------------
    - Azimuth bins partition [0, 360) with width ``az_step_deg``.
    - Elevation bins partition [el_min_deg, el_max_deg] with width
      ``el_step_deg`` (upper edge included then clamped to last bin).
    - Output cell id is row-major in (el_bin, az_bin):
        ``cell_id = el_bin * n_az_bins + az_bin``

    Invalid rows
    ------------
    Non-finite az/el or elevation outside [el_min_deg, el_max_deg] return -1.

    Performance notes
    -----------------
    - For large arrays, an optional Numba kernel is used.
    - For small arrays, NumPy vectorization is typically faster due to lower
      dispatch overhead.

    Returns
    -------
    (cell_id, n_az_bins, n_el_bins)
    """
    az = np.asarray(az_deg, dtype=np.float64).reshape(-1)
    el = np.asarray(el_deg, dtype=np.float64).reshape(-1)
    if az.size != el.size:
        raise ValueError("az_deg and el_deg must have identical sizes.")

    az_step = float(az_step_deg)
    el_step = float(el_step_deg)
    el_min = float(el_min_deg)
    el_max = float(el_max_deg)
    if az_step <= 0.0 or el_step <= 0.0:
        raise ValueError("Regular sky-cell steps must be positive.")
    if not (el_max > el_min):
        raise ValueError("Regular sky-cell elevation range must satisfy el_max > el_min.")

    n_az = int(np.ceil(360.0 / az_step))
    n_el = int(np.ceil((el_max - el_min) / el_step))
    n_az = max(1, n_az)
    n_el = max(1, n_el)

    # Large-array fast path with Numba (if available).
    # We use this path only when row count is high enough to amortize overhead.
    if _HAVE_NUMBA and az.size >= _NUMBA_MIN_SKYCELL_ROWS:
        try:
            out_nb = _skycell_id_regular_numba(
                az,
                el,
                float(az_step),
                float(el_step),
                float(el_min),
                float(el_max),
                int(n_az),
                int(n_el),
            )
            return out_nb, n_az, n_el
        except Exception:
            # Robust fallback if numba cache/runtime is unavailable.
            pass

    out = np.full(az.size, -1, dtype=np.int32)

    m = (
        np.isfinite(az) & np.isfinite(el) &
        (el >= el_min) & (el <= el_max)
    )
    if not np.any(m):
        return out, n_az, n_el

    az_m = np.remainder(az[m], 360.0)
    el_m = el[m]

    az_bin = np.floor(az_m / az_step).astype(np.int32, copy=False)
    el_bin = np.floor((el_m - el_min) / el_step).astype(np.int32, copy=False)

    az_bin = np.minimum(az_bin, np.int32(n_az - 1))
    el_bin = np.minimum(el_bin, np.int32(n_el - 1))

    out[m] = el_bin * np.int32(n_az) + az_bin
    return out, n_az, n_el


def _describe_skycells_s1586(sky_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Describe S.1586-1 skycell IDs with az/el bounds and centers (degrees).
    """
    sid = np.asarray(sky_ids, dtype=np.int64).reshape(-1)
    if sid.size == 0:
        zf = np.empty(0, dtype=np.float64)
        zi = np.empty(0, dtype=np.int64)
        return {
            "skycell_id": zi,
            "az_low_deg": zf,
            "az_high_deg": zf,
            "az_center_deg": zf,
            "el_low_deg": zf,
            "el_high_deg": zf,
            "el_center_deg": zf,
        }

    if np.any((sid < 0) | (sid >= int(_S1586_N_CELLS))):
        raise ValueError("sky_ids contain out-of-range S.1586-1 ids.")

    ring = np.searchsorted(_S1586_RING_OFFSETS, sid, side="right") - 1
    ring = ring.astype(np.int64, copy=False)
    az_bin = sid - _S1586_RING_OFFSETS[ring]

    step = _S1586_AZ_STEPS_DEG[ring].astype(np.float64, copy=False)
    az_low = az_bin.astype(np.float64, copy=False) * step
    az_high = az_low + step
    el_low = 3.0 * ring.astype(np.float64, copy=False)
    el_high = el_low + 3.0

    return {
        "skycell_id": sid.astype(np.int64, copy=False),
        "az_low_deg": az_low,
        "az_high_deg": az_high,
        "az_center_deg": 0.5 * (az_low + az_high),
        "el_low_deg": el_low,
        "el_high_deg": el_high,
        "el_center_deg": 0.5 * (el_low + el_high),
    }


def _describe_skycells_regular(
    sky_ids: np.ndarray,
    *,
    n_az: int,
    n_el: int,
    az_step_deg: float,
    el_step_deg: float,
    el_min_deg: float,
) -> Dict[str, np.ndarray]:
    """
    Describe regular-grid skycell IDs with az/el bounds and centers (degrees).
    """
    sid = np.asarray(sky_ids, dtype=np.int64).reshape(-1)
    if sid.size == 0:
        zf = np.empty(0, dtype=np.float64)
        zi = np.empty(0, dtype=np.int64)
        return {
            "skycell_id": zi,
            "az_low_deg": zf,
            "az_high_deg": zf,
            "az_center_deg": zf,
            "el_low_deg": zf,
            "el_high_deg": zf,
            "el_center_deg": zf,
        }

    n_az_i = int(n_az)
    n_el_i = int(n_el)
    n_tot = int(n_az_i * n_el_i)
    if n_az_i <= 0 or n_el_i <= 0:
        raise ValueError("Regular skycell descriptor requires positive n_az and n_el.")
    if np.any((sid < 0) | (sid >= n_tot)):
        raise ValueError("sky_ids contain out-of-range regular-grid ids.")

    az_bin = sid % n_az_i
    el_bin = sid // n_az_i

    az_step = float(az_step_deg)
    el_step = float(el_step_deg)
    el_min = float(el_min_deg)

    az_low = az_bin.astype(np.float64, copy=False) * az_step
    az_high = az_low + az_step
    el_low = el_min + el_bin.astype(np.float64, copy=False) * el_step
    el_high = el_low + el_step

    return {
        "skycell_id": sid.astype(np.int64, copy=False),
        "az_low_deg": az_low,
        "az_high_deg": az_high,
        "az_center_deg": 0.5 * (az_low + az_high),
        "el_low_deg": el_low,
        "el_high_deg": el_high,
        "el_center_deg": 0.5 * (el_low + el_high),
    }


def _build_indexed_reservoir_pools(
    ids: np.ndarray,
    beta_deg: np.ndarray,
    alpha_deg: np.ndarray,
    *,
    n_ids: int,
    max_samples_per_id: int,
    rng: np.random.Generator,
    progress: bool = False,
    progress_desc: str = "Building pools",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build compact per-ID sample pools with deterministic memory bounds.

    For each ID:
      - keep all samples if count <= max_samples_per_id
      - otherwise keep a random subset without replacement

    Why this structure exists
    -------------------------
    Conditional sampling needs fast random access to "all samples belonging to
    a given key" where key is either:
      - group_id = belt_id * n_skycells + skycell_id
      - belt_id

    A Python ``dict[id] -> np.ndarray`` would work functionally, but has higher
    overhead and fragmented memory. Here we use a CSR-like layout:
      - ``ptr[id]:ptr[id+1]`` gives the slice for that id in flat pools
      - contiguous ``beta_pool``/``alpha_pool`` store all kept samples

    This design is:
      - memory-bounded (hard cap via ``max_samples_per_id``)
      - cache-friendly (contiguous slices)
      - serialization-friendly (simple NumPy arrays)

    Complexity
    ----------
    - Sorting dominates build time: O(N log N)
    - Sampling from pools at runtime is O(1) per draw (after slice lookup)

    Returns
    -------
    (ptr, beta_pool, alpha_pool, raw_counts)
        ptr        : int64, shape (n_ids + 1,), CSR-like offsets
        beta_pool  : float32, concatenated values
        alpha_pool : float32, concatenated values
        raw_counts : int32, original counts per ID before truncation
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    beta = np.asarray(beta_deg, dtype=np.float64).reshape(-1)
    alpha = np.asarray(alpha_deg, dtype=np.float64).reshape(-1)
    n = ids.size

    if beta.size != n or alpha.size != n:
        raise ValueError("ids, beta_deg and alpha_deg must have identical sizes.")
    if n_ids <= 0:
        raise ValueError("n_ids must be positive.")
    if max_samples_per_id <= 0:
        raise ValueError("max_samples_per_id must be positive.")

    ptr = np.zeros(int(n_ids) + 1, dtype=np.int64)
    raw_counts = np.zeros(int(n_ids), dtype=np.int32)
    if n == 0:
        return ptr, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), raw_counts

    if np.any(ids < 0) or np.any(ids >= int(n_ids)):
        raise ValueError("Found ids outside [0, n_ids).")

    # Group equal IDs into contiguous segments once; this enables compact pool
    # construction without per-ID boolean masks.
    order = np.argsort(ids, kind="mergesort")
    ids_sorted = ids[order]

    cuts = np.flatnonzero(np.diff(ids_sorted)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [n]))

    unique_ids = ids_sorted[starts].astype(np.int64, copy=False)
    seg_counts = (ends - starts).astype(np.int32, copy=False)
    raw_counts[unique_ids] = seg_counts

    keep_counts = np.zeros(int(n_ids), dtype=np.int32)
    keep_counts[unique_ids] = np.minimum(seg_counts, np.int32(max_samples_per_id))

    ptr[1:] = np.cumsum(keep_counts, dtype=np.int64)
    total_keep = int(ptr[-1])

    beta_pool = np.empty(total_keep, dtype=np.float32)
    alpha_pool = np.empty(total_keep, dtype=np.float32)

    idx_iter = _progress_iter(
        range(unique_ids.size),
        enabled=bool(progress),
        total=int(unique_ids.size),
        desc=str(progress_desc),
    )
    for k in idx_iter:
        uid = int(unique_ids[k])
        s = int(starts[k])
        e = int(ends[k])
        cnt = e - s
        keep = int(keep_counts[uid])
        if keep <= 0:
            continue

        # Candidate rows for this ID in original arrays.
        ord_seg = order[s:e]
        if keep < cnt:
            # Reservoir-style truncation: unbiased subset without replacement.
            pick_local = rng.choice(cnt, size=keep, replace=False)
            ord_seg = ord_seg[pick_local]

        d0 = int(ptr[uid])
        d1 = int(ptr[uid + 1])
        beta_pool[d0:d1] = beta[ord_seg].astype(np.float32, copy=False)
        alpha_pool[d0:d1] = alpha[ord_seg].astype(np.float32, copy=False)

    return ptr, beta_pool, alpha_pool, raw_counts

# ============================================================================
# Streaming histogram builder for uniform edges
# ============================================================================

if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _hist2d_uniform_numba(
        beta: np.ndarray,
        alpha: np.ndarray,
        b0: float,
        a0: float,
        db: float,
        da: float,
        n_beta: int,
        n_alpha: int,
    ) -> np.ndarray:
        """
        Low-memory single-pass histogram kernel for uniform bins.

        Notes
        -----
        - Uses scalar loops to avoid chunk-level temporary arrays.
        - Semantics match NumPy path: values outside [min, max) are ignored.
        """
        counts = np.zeros((n_beta, n_alpha), dtype=np.float64)
        n = beta.size
        for i in range(n):
            ib = int((beta[i] - b0) / db)
            ia = int((alpha[i] - a0) / da)
            if 0 <= ib < n_beta and 0 <= ia < n_alpha:
                counts[ib, ia] += 1.0
        return counts


def _hist2d_stream_uniform_edges(
    beta: np.ndarray,
    alpha: np.ndarray,
    beta_edges: np.ndarray,
    alpha_edges: np.ndarray,
    *,
    chunk_size: int = 5_000_000,
    progress: bool = False,
    progress_desc: str = "Histogram chunks",
) -> np.ndarray:
    """
    Build 2D histogram counts using streaming accumulation.

    This avoids potential heavy temporary allocations of ``np.histogram2d`` on
    very large datasets.

    Assumptions:
      - edges are uniform (linspace), which is exactly how we build them in this library.

    Acceleration policy:
      - If Numba is available and row count is very large, use a single-pass JIT
        kernel with O(1) extra memory.
      - Otherwise use chunked ``np.bincount`` accumulation to keep Python
        overhead low while bounding temporary memory.
      - Optional tqdm progress bars are available via ``progress=True``.

    Returns:
      counts shape (N_beta, N_alpha) as float64.
    """
    beta_edges = np.asarray(beta_edges, dtype=np.float64)
    alpha_edges = np.asarray(alpha_edges, dtype=np.float64)

    n_beta = beta_edges.size - 1
    n_alpha = alpha_edges.size - 1
    n_bins_total = n_beta * n_alpha

    b0, b1 = float(beta_edges[0]), float(beta_edges[-1])
    a0, a1 = float(alpha_edges[0]), float(alpha_edges[-1])

    db = (b1 - b0) / n_beta
    da = (a1 - a0) / n_alpha
    if db <= 0 or da <= 0:
        raise ValueError("Invalid edges: non-positive bin width.")

    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    n = beta.size
    if n != alpha.size:
        raise ValueError("beta and alpha must have the same length.")

    # For very large arrays, JIT path avoids chunk-level temporary vectors
    # (`ib`, `ia`, boolean mask, `flat`) and can reduce both memory traffic and
    # wall-clock time.
    if _HAVE_NUMBA and n >= _NUMBA_MIN_STREAM_HIST_ROWS:
        try:
            return _hist2d_uniform_numba(beta, alpha, b0, a0, db, da, n_beta, n_alpha)
        except Exception:
            # Robust fallback if numba cache/runtime is unavailable.
            pass

    counts_flat = np.zeros(n_bins_total, dtype=np.float64)

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    n_chunks = int((n + chunk_size - 1) // chunk_size)
    starts = _progress_iter(
        range(0, n, chunk_size),
        enabled=bool(progress),
        total=n_chunks,
        desc=str(progress_desc),
    )
    for start in starts:
        end = min(start + chunk_size, n)
        b = beta[start:end]
        a = alpha[start:end]

        # Fast uniform-edge binning via scaling
        ib = ((b - b0) / db).astype(np.int64)
        ia = ((a - a0) / da).astype(np.int64)

        m = (ib >= 0) & (ib < n_beta) & (ia >= 0) & (ia < n_alpha)
        if not np.any(m):
            continue

        flat = ib[m] * n_alpha + ia[m]
        counts_flat += np.bincount(flat, minlength=n_bins_total).astype(np.float64, copy=False)

    return counts_flat.reshape(n_beta, n_alpha)


# ============================================================================
# Gaussian smoothing (separable). Alpha axis is circular (wrap), beta is reflect.
# ============================================================================

def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """
    Build a 1D Gaussian kernel normalized to sum to 1.

    sigma is measured in bins.
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)

    radius = int(np.ceil(truncate * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def _smooth_2d_gaussian_numpy(
    arr: np.ndarray,
    sigma_beta: float,
    sigma_alpha: float,
    truncate: float,
) -> np.ndarray:
    """
    Separable Gaussian smoothing with boundary conditions:
      - alpha axis (axis=1): circular wrap
      - beta axis  (axis=0): reflect

    This is the NumPy fallback if Numba is unavailable.
    """
    out = arr.astype(np.float64, copy=True)

    # Smooth along alpha axis (wrap)
    if sigma_alpha > 0:
        k = _gaussian_kernel1d(sigma_alpha, truncate=truncate)
        r = k.size // 2
        padded = np.pad(out, ((0, 0), (r, r)), mode="wrap")
        tmp = np.empty_like(out)
        for i in range(out.shape[0]):
            tmp[i, :] = np.convolve(padded[i, :], k, mode="valid")
        out = tmp

    # Smooth along beta axis (reflect)
    if sigma_beta > 0:
        k = _gaussian_kernel1d(sigma_beta, truncate=truncate)
        r = k.size // 2
        padded = np.pad(out, ((r, r), (0, 0)), mode="reflect")
        tmp = np.empty_like(out)
        for j in range(out.shape[1]):
            tmp[:, j] = np.convolve(padded[:, j], k, mode="valid")
        out = tmp

    return out


if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _conv_axis1_wrap_numba(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Convolve each row with a 1D kernel, wrap (circular) padding on axis=1.
        """
        h, w = src.shape
        r = kernel.size // 2
        dst = np.empty_like(src)
        for i in range(h):
            for j in range(w):
                acc = 0.0
                for kk in range(-r, r + 1):
                    jj = j + kk
                    # wrap
                    jj %= w
                    acc += src[i, jj] * kernel[kk + r]
                dst[i, j] = acc
        return dst

    @nb.njit(cache=True)
    def _conv_axis0_reflect_numba(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Convolve each column with a 1D kernel, reflect padding on axis=0.
        """
        h, w = src.shape
        r = kernel.size // 2
        dst = np.empty_like(src)
        for i in range(h):
            for j in range(w):
                acc = 0.0
                for kk in range(-r, r + 1):
                    ii = i + kk
                    if ii < 0:
                        ii = -ii - 1
                    elif ii >= h:
                        ii = 2 * h - ii - 1
                    acc += src[ii, j] * kernel[kk + r]
                dst[i, j] = acc
        return dst

    def _smooth_2d_gaussian(
        arr: np.ndarray,
        sigma_beta: float,
        sigma_alpha: float,
        truncate: float,
    ) -> np.ndarray:
        out = arr.astype(np.float64, copy=True)
        if sigma_alpha > 0:
            k = _gaussian_kernel1d(sigma_alpha, truncate=truncate)
            out = _conv_axis1_wrap_numba(out, k)
        if sigma_beta > 0:
            k = _gaussian_kernel1d(sigma_beta, truncate=truncate)
            out = _conv_axis0_reflect_numba(out, k)
        return out
else:
    def _smooth_2d_gaussian(
        arr: np.ndarray,
        sigma_beta: float,
        sigma_alpha: float,
        truncate: float,
    ) -> np.ndarray:
        return _smooth_2d_gaussian_numpy(arr, sigma_beta, sigma_alpha, truncate)


def _prepare_auto_smooth_candidates(
    candidates: Any,
    *,
    fixed_sigma_beta: float,
    fixed_sigma_alpha: float,
) -> list[Tuple[float, float]]:
    """
    Normalize candidate smoothing sigma pairs for auto-tuning.
    """
    if candidates is None:
        out: list[Tuple[float, float]] = [
            (0.0, 0.0),
            (0.2, 0.3),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0),
            (1.0, 1.2),
            (1.2, 1.4),
        ]
    else:
        out = []
        for item in candidates:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("auto_smooth_candidates must contain (sigma_beta, sigma_alpha) pairs.")
            sb = float(item[0])
            sa = float(item[1])
            if sb < 0.0 or sa < 0.0:
                raise ValueError("auto_smooth_candidates values must be non-negative.")
            out.append((sb, sa))
        if not out:
            raise ValueError("auto_smooth_candidates must not be empty.")

    fixed_pair = (float(fixed_sigma_beta), float(fixed_sigma_alpha))
    if fixed_pair[0] < 0.0 or fixed_pair[1] < 0.0:
        raise ValueError("smooth_sigma_beta and smooth_sigma_alpha must be non-negative.")
    if fixed_pair not in out:
        out.append(fixed_pair)

    seen: set[Tuple[float, float]] = set()
    uniq: list[Tuple[float, float]] = []
    for pair in out:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)
    return uniq


def _auto_tune_smoothing_sigmas(
    counts_raw: np.ndarray,
    beta_edges: np.ndarray,
    alpha_edges: np.ndarray,
    beta_val: np.ndarray,
    alpha_val: np.ndarray,
    *,
    candidates: list[Tuple[float, float]],
    smooth_truncate: float,
    progress: bool,
    progress_desc: str,
) -> Tuple[float, float, np.ndarray, Dict[str, np.ndarray]]:
    """
    Select smoothing sigmas by minimizing validation negative log-likelihood.

    Returns
    -------
    (best_sigma_beta, best_sigma_alpha, best_counts, diagnostics)
    """
    beta_edges = np.asarray(beta_edges, dtype=np.float64)
    alpha_edges = np.asarray(alpha_edges, dtype=np.float64)
    beta_val = np.asarray(beta_val, dtype=np.float64).reshape(-1)
    alpha_val = np.asarray(alpha_val, dtype=np.float64).reshape(-1)
    if beta_val.size != alpha_val.size:
        raise ValueError("beta_val and alpha_val must have identical sizes.")
    if beta_val.size == 0:
        raise ValueError("Validation set for auto smoothing is empty.")

    n_beta = int(beta_edges.size - 1)
    n_alpha = int(alpha_edges.size - 1)
    b0 = float(beta_edges[0])
    b1 = float(beta_edges[-1])
    a0 = float(alpha_edges[0])
    a1 = float(alpha_edges[-1])
    db = float((b1 - b0) / max(n_beta, 1))
    da = float((a1 - a0) / max(n_alpha, 1))
    if db <= 0.0 or da <= 0.0:
        raise ValueError("Invalid edges for auto smoothing.")

    ib = ((beta_val - b0) / db).astype(np.int64, copy=False)
    ia = ((alpha_val - a0) / da).astype(np.int64, copy=False)
    m = (ib >= 0) & (ib < n_beta) & (ia >= 0) & (ia < n_alpha)
    if not np.any(m):
        raise ValueError("Validation rows fall outside configured beta/alpha ranges.")
    ib = ib[m]
    ia = ia[m]
    n_used = int(ib.size)

    eps = 1e-15
    best_idx = -1
    best_nll = np.inf
    best_counts = counts_raw
    nll_list: list[float] = []

    cand_iter = _progress_iter(
        range(len(candidates)),
        enabled=bool(progress),
        total=len(candidates),
        desc=f"{progress_desc}: tune smooth",
    )
    for i in cand_iter:
        sb, sa = candidates[i]
        if sb > 0.0 or sa > 0.0:
            counts_i = _smooth_2d_gaussian(
                counts_raw,
                sigma_beta=float(sb),
                sigma_alpha=float(sa),
                truncate=float(smooth_truncate),
            )
        else:
            counts_i = counts_raw

        total_i = float(counts_i.sum())
        if not np.isfinite(total_i) or total_i <= 0.0:
            nll = np.inf
        else:
            pmf_i = counts_i / total_i
            p = pmf_i[ib, ia]
            nll = float(-np.mean(np.log(np.maximum(p, eps))))

        nll_list.append(nll)
        if nll < best_nll:
            best_nll = nll
            best_idx = int(i)
            best_counts = counts_i

    if best_idx < 0:
        raise RuntimeError("Auto smoothing failed to find a valid candidate.")

    best_sb, best_sa = candidates[best_idx]
    diag = {
        "auto_smooth_sigma_beta_candidates": np.asarray([x[0] for x in candidates], dtype=np.float64),
        "auto_smooth_sigma_alpha_candidates": np.asarray([x[1] for x in candidates], dtype=np.float64),
        "auto_smooth_nll": np.asarray(nll_list, dtype=np.float64),
        "auto_smooth_validation_rows": np.asarray(n_used, dtype=np.int64),
        "auto_smooth_best_index": np.asarray(best_idx, dtype=np.int64),
        "auto_smooth_best_sigma_beta": np.asarray(best_sb, dtype=np.float64),
        "auto_smooth_best_sigma_alpha": np.asarray(best_sa, dtype=np.float64),
    }
    return float(best_sb), float(best_sa), best_counts, diag


# ============================================================================
# Metrics
# ============================================================================

def joint_hist_metrics(
    beta_real: np.ndarray,
    alpha_real: np.ndarray,
    beta_emp: np.ndarray,
    alpha_emp: np.ndarray,
    beta_edges: np.ndarray,
    alpha_edges: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Histogram-based diagnostics between joint distributions P (real) and Q (empirical)
    on a shared (beta, alpha) grid.

    These metrics are useful for debugging, but they can be overly strict on fine grids.
    """
    h_real, _, _ = np.histogram2d(beta_real, alpha_real, bins=[beta_edges, alpha_edges])
    h_emp,  _, _ = np.histogram2d(beta_emp,  alpha_emp,  bins=[beta_edges, alpha_edges])

    p = h_real.astype(np.float64, copy=False)
    q = h_emp.astype(np.float64, copy=False)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("Histogram sums are zero; cannot compute diagnostics.")

    p /= p_sum
    q /= q_sum

    tv = 0.5 * np.abs(p - q).sum()
    rms = float(np.sqrt(np.mean((p - q) ** 2)))

    p_safe = p + eps
    q_safe = q + eps
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()

    kl_pq = float(np.sum(p_safe * np.log(p_safe / q_safe)))
    kl_qp = float(np.sum(q_safe * np.log(q_safe / p_safe)))

    return {"tv": float(tv), "rms": float(rms), "kl_pq": kl_pq, "kl_qp": kl_qp}


def _make_unit_directions(
    d: int,
    n_slices: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a fixed set of random unit directions to reduce randomness in comparisons.
    """
    v = rng.normal(size=(int(n_slices), int(d))).astype(np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-15
    v /= norms
    return v


if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _sliced_wasserstein_from_embeddings_numba(
        x_real: np.ndarray,
        x_emp: np.ndarray,
        directions: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Numba implementation of sliced Wasserstein once embeddings are prepared.

        Inputs
        ------
        x_real, x_emp : float64 arrays, shape (N, D)
            Embedded samples for real and empirical distributions.
        directions : float64 array, shape (K, D)
            Unit projection directions.

        Returns
        -------
        (sw_mean, sw_std)
            Mean and std over per-direction 1D Wasserstein distances.

        Notes
        -----
        - This kernel keeps memory bounded: it allocates two temporary vectors
          of length N per direction and reuses them through the loop.
        - It intentionally mirrors the Python/Numpy path for numerical parity.
        """
        n = x_real.shape[0]
        d = x_real.shape[1]
        k = directions.shape[0]

        sw_vals = np.empty(k, dtype=np.float64)
        for i in range(k):
            v = directions[i]
            pr = np.empty(n, dtype=np.float64)
            pe = np.empty(n, dtype=np.float64)

            for t in range(n):
                acc_r = 0.0
                acc_e = 0.0
                for j in range(d):
                    acc_r += x_real[t, j] * v[j]
                    acc_e += x_emp[t, j] * v[j]
                pr[t] = acc_r
                pe[t] = acc_e

            pr.sort()
            pe.sort()

            acc = 0.0
            for t in range(n):
                dv = pr[t] - pe[t]
                if dv < 0.0:
                    dv = -dv
                acc += dv
            sw_vals[i] = acc / n

        return float(sw_vals.mean()), float(sw_vals.std())


def sliced_wasserstein(
    beta_real: np.ndarray,
    alpha_real: np.ndarray,
    beta_emp: np.ndarray,
    alpha_emp: np.ndarray,
    *,
    directions: Optional[np.ndarray] = None,
    circular_alpha: bool = True,
    beta_scale: Optional[float] = None,
    alpha_weight: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Transport-aware metric: Sliced Wasserstein-1 distance.

    We compare two point clouds by projecting them onto many 1D lines (directions),
    computing 1D Wasserstein-1 (sort + mean absolute diff) per direction, then averaging.

    Representation:
      - beta is normalized by beta_scale (default: beta range width).
      - alpha is embedded as (cos(alpha), sin(alpha)) if circular_alpha=True to avoid 0/360 discontinuity.
      - alpha_weight controls relative contribution of alpha vs beta in the embedding.

    Returns:
      - sw_mean: mean over directions
      - sw_std: standard deviation over directions (directional variability)

    Performance notes
    -----------------
    The function has two execution modes:
      1) Numba path (if available and workload is large enough):
         loops over directions in compiled code with bounded memory.
      2) NumPy path:
         pure NumPy loop over directions (robust fallback).

    The threshold is driven by ``n_points * n_slices`` to avoid JIT overhead on
    small evaluations.
    """
    if rng is None:
        rng = np.random.default_rng()

    beta_real = np.asarray(beta_real, dtype=np.float64)
    alpha_real = np.asarray(alpha_real, dtype=np.float64)
    beta_emp = np.asarray(beta_emp, dtype=np.float64)
    alpha_emp = np.asarray(alpha_emp, dtype=np.float64)

    n = min(beta_real.size, beta_emp.size)
    if n <= 0:
        raise ValueError("sliced_wasserstein requires non-empty input arrays.")
    if beta_real.size != n:
        idx = rng.choice(beta_real.size, size=n, replace=False)
        beta_real = beta_real[idx]
        alpha_real = alpha_real[idx]
    if beta_emp.size != n:
        idx = rng.choice(beta_emp.size, size=n, replace=False)
        beta_emp = beta_emp[idx]
        alpha_emp = alpha_emp[idx]

    if beta_scale is None:
        bmin = min(beta_real.min(initial=0.0), beta_emp.min(initial=0.0))
        bmax = max(beta_real.max(initial=1.0), beta_emp.max(initial=1.0))
        beta_scale = float(max(bmax - bmin, 1e-12))

    beta_r = beta_real / beta_scale
    beta_e = beta_emp / beta_scale

    if circular_alpha:
        ar = np.deg2rad(alpha_real)
        ae = np.deg2rad(alpha_emp)
        x_real = np.stack([beta_r, alpha_weight * np.cos(ar), alpha_weight * np.sin(ar)], axis=1)
        x_emp  = np.stack([beta_e, alpha_weight * np.cos(ae), alpha_weight * np.sin(ae)], axis=1)
        d = 3
    else:
        amin = min(alpha_real.min(initial=0.0), alpha_emp.min(initial=0.0))
        amax = max(alpha_real.max(initial=1.0), alpha_emp.max(initial=1.0))
        alpha_scale = float(max(amax - amin, 1e-12))
        x_real = np.stack([beta_r, (alpha_real - amin) / alpha_scale], axis=1)
        x_emp  = np.stack([beta_e, (alpha_emp  - amin) / alpha_scale], axis=1)
        d = 2

    if directions is None:
        directions = _make_unit_directions(d, 32, rng)
    else:
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1] != d:
            raise ValueError(f"directions must have shape (K, {d}).")

    k = directions.shape[0]
    n_work = int(n * k)

    # Use Numba for large workloads where Python-loop overhead becomes dominant.
    if _HAVE_NUMBA and n_work >= _NUMBA_MIN_SW_WORK:
        try:
            sw_mean, sw_std = _sliced_wasserstein_from_embeddings_numba(
                x_real.astype(np.float64, copy=False),
                x_emp.astype(np.float64, copy=False),
                directions.astype(np.float64, copy=False),
            )
            return {"sw_mean": float(sw_mean), "sw_std": float(sw_std)}
        except Exception:
            # Fallback is intentionally robust; accuracy is unchanged.
            pass

    sw_vals = np.empty(k, dtype=np.float64)
    for i in range(k):
        v = directions[i]
        pr = x_real @ v
        pe = x_emp @ v
        pr.sort()
        pe.sort()
        sw_vals[i] = np.mean(np.abs(pr - pe))

    return {"sw_mean": float(sw_vals.mean()), "sw_std": float(sw_vals.std(ddof=0))}


def _auto_ratio_scale_from_baseline(
    denom_mean: float,
    denom_std: float,
    *,
    z: float = 2.0,
    target_score: float = 85.0,
    min_scale: float = 0.10,
) -> float:
    """
    Auto-select ratio_scale from variability of the chosen denominator.

    Let CV = std/mean, define a typical "high" ratio:
      ratio_hi = 1 + z * CV

    Choose ratio_scale so that score(ratio_hi) == target_score, where:
      score = 100 * exp(-(ratio - 1)/ratio_scale) for ratio > 1
    """
    mean = float(denom_mean)
    std = float(denom_std)
    if mean <= 0:
        return 0.5

    cv = std / mean
    ratio_hi = 1.0 + float(z) * float(cv)
    denom = np.log(100.0 / float(target_score))
    if denom <= 0:
        return 0.5

    scale = (ratio_hi - 1.0) / denom
    return float(max(scale, min_scale))


def _score_from_ratio(ratio: float, ratio_scale: float) -> Dict[str, Any]:
    """
    Convert a ratio into a 0..100 score using an exponential penalty:
      score = 100 * exp(-(max(0, ratio-1))/ratio_scale)

    Grade is derived from score (practical interpretation).
    """
    ratio = float(ratio)
    ratio_scale = float(ratio_scale)

    excess = max(0.0, ratio - 1.0)
    score = 100.0 * float(np.exp(-excess / max(ratio_scale, 1e-12)))
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 90.0:
        grade = "A"
    elif score >= 70.0:
        grade = "B"
    elif score >= 40.0:
        grade = "C"
    else:
        grade = "D"

    return {"score": score, "grade": grade}


# ============================================================================
# Sampler object
# ============================================================================

@dataclass(frozen=True)
class JointAngleSampler:
    """
    Joint sampler for (beta_deg, alpha_deg) based on a 2D histogram CDF.

    Minimal state required for sampling:
      - beta_edges, alpha_edges
      - cdf (flattened), n_alpha
      - beta_range, alpha_range, alpha_wrapped

    Optional metadata (tiny):
      - smoothing parameters

    Optional conditional model:
      - conditioned sampling by (sat_azimuth, sat_elevation, sat_belt_id)
      - sky-cell binning (S.1586-1 or regular az/el grid)
      - bounded per-group and per-belt sample reservoirs
    """

    beta_edges: np.ndarray
    alpha_edges: np.ndarray
    cdf: np.ndarray
    n_alpha: int
    beta_range: Tuple[float, float]
    alpha_range: Tuple[float, float]
    alpha_wrapped: bool

    smooth_sigma_beta: float = 0.0
    smooth_sigma_alpha: float = 0.0
    smooth_truncate: float = 3.0

    # Optional conditioned-sampling payload.
    conditional_enabled: bool = False
    skycell_mode: str = "none"   # "none" | "s1586" | "regular"
    n_skycells: int = 0
    n_belts: int = 0

    # Regular-grid sky-cell parameters (used only when skycell_mode="regular").
    skycell_az_step_deg: float = 0.0
    skycell_el_step_deg: float = 0.0
    skycell_el_min_deg: float = 0.0
    skycell_el_max_deg: float = 90.0
    skycell_n_az: int = 0
    skycell_n_el: int = 0

    # Fallback policy: group -> belt -> global.
    cond_min_group_samples: int = 0
    cond_min_belt_samples: int = 0

    # Per-(belt, skycell) pools; group_id = belt_id * n_skycells + skycell_id.
    group_ptr: Optional[np.ndarray] = None
    group_beta_pool: Optional[np.ndarray] = None
    group_alpha_pool: Optional[np.ndarray] = None
    group_raw_counts: Optional[np.ndarray] = None

    # Per-belt fallback pools.
    belt_ptr: Optional[np.ndarray] = None
    belt_beta_pool: Optional[np.ndarray] = None
    belt_alpha_pool: Optional[np.ndarray] = None
    belt_raw_counts: Optional[np.ndarray] = None

    # --------------------------
    # Internal helpers (no state mutation)
    # --------------------------

    def _pmf_2d(self) -> np.ndarray:
        """
        Recover the joint probability mass function from the stored CDF.

        Notes
        -----
        * The CDF is kept flattened (beta-major). A quick diff is enough
          to reconstruct the PMF without allocating intermediate histograms.
        * Lazily cached on the frozen instance to avoid repeated diff/reshape
          work when multiple stats/plots are requested in quick succession.
        """

        cached = getattr(self, "_pmf_cache", None)
        if cached is not None:
            return cached

        # Use prepend to keep dtype stable and avoid an explicit concat.
        pmf_flat = np.diff(self.cdf.astype(np.float64, copy=False), prepend=0.0)
        pmf_2d = pmf_flat.reshape(-1, self.n_alpha)
        total = pmf_2d.sum(dtype=np.float64)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Sampler CDF produces a non-positive PMF.")

        pmf_norm = pmf_2d / total
        object.__setattr__(self, "_pmf_cache", pmf_norm)
        return pmf_norm

    def sampler_statistics(self) -> Dict[str, np.ndarray]:
        """
        Fast access to 1D marginal densities derived from the sampler only.

        Returns
        -------
        dict
            "beta_centers" : bin centers for β in degrees
            "beta_density" : probability density per β bin (normalized by width)
            "alpha_centers": bin centers for α in degrees
            "alpha_density": probability density per α bin (normalized by width)

        This avoids drawing random samples purely to inspect the current
        distribution, which keeps the latency low when statistics are queried
        repeatedly (e.g., extracting α/β curves in a UI loop).
        """

        pmf = self._pmf_2d()

        beta_edges_f = self.beta_edges.astype(np.float64, copy=False)
        alpha_edges_f = self.alpha_edges.astype(np.float64, copy=False)

        beta_widths = np.diff(beta_edges_f)
        alpha_widths = np.diff(alpha_edges_f)

        beta_centers = 0.5 * (beta_edges_f[:-1] + beta_edges_f[1:])
        alpha_centers = 0.5 * (alpha_edges_f[:-1] + alpha_edges_f[1:])

        # Marginals: sum over the opposite axis, then normalize by bin width
        beta_marginal = pmf.sum(axis=1) / beta_widths
        alpha_marginal = pmf.sum(axis=0) / alpha_widths

        return {
            "beta_centers": beta_centers,
            "beta_density": beta_marginal,
            "alpha_centers": alpha_centers,
            "alpha_density": alpha_marginal,
        }

    def _has_conditional_model(self) -> bool:
        """
        Internal consistency check for the optional conditional model.
        """
        if not self.conditional_enabled:
            return False

        required = (
            self.group_ptr,
            self.group_beta_pool,
            self.group_alpha_pool,
            self.group_raw_counts,
            self.belt_ptr,
            self.belt_beta_pool,
            self.belt_alpha_pool,
            self.belt_raw_counts,
        )
        return all(x is not None for x in required)

    def has_conditional_model(self) -> bool:
        """
        Public flag indicating whether this sampler contains a conditional model.
        """
        return self._has_conditional_model()

    def _sample_unconditional_flat(
        self,
        rng: np.random.Generator,
        n: int,
        *,
        dtype: Any = np.float32,
        chunk: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast internal sampler that always returns flat arrays of length `n`.
        """
        n = int(n)
        if n < 0:
            raise ValueError("`n` must be non-negative.")
        if n == 0:
            return np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)

        beta_out = np.empty(n, dtype=dtype)
        alpha_out = np.empty(n, dtype=dtype)

        if chunk is None:
            chunk = n
        chunk = int(chunk)
        if chunk <= 0:
            raise ValueError("`chunk` must be a positive integer.")

        start = 0
        while start < n:
            end = min(start + chunk, n)
            m = end - start

            u0 = rng.random(m)
            idx_flat = np.searchsorted(self.cdf, u0, side="right")

            i_beta = idx_flat // self.n_alpha
            i_alpha = idx_flat - i_beta * self.n_alpha

            b0 = self.beta_edges[i_beta].astype(np.float64, copy=False)
            b1 = self.beta_edges[i_beta + 1].astype(np.float64, copy=False)
            a0 = self.alpha_edges[i_alpha].astype(np.float64, copy=False)
            a1 = self.alpha_edges[i_alpha + 1].astype(np.float64, copy=False)

            wb = rng.random(m)
            wa = rng.random(m)

            beta_out[start:end] = (b0 + wb * (b1 - b0)).astype(dtype, copy=False)
            alpha_out[start:end] = (a0 + wa * (a1 - a0)).astype(dtype, copy=False)

            start = end

        return beta_out, alpha_out

    def _skycell_id_from_observer_angles(
        self,
        sat_azimuth_deg: np.ndarray,
        sat_elevation_deg: np.ndarray,
    ) -> np.ndarray:
        """
        Compute sky-cell IDs for observer-frame satellite directions.
        """
        mode = str(self.skycell_mode).lower()
        if mode == "s1586":
            return _skycell_id_s1586(sat_azimuth_deg, sat_elevation_deg)
        if mode == "regular":
            ids, _, _ = _skycell_id_regular(
                sat_azimuth_deg,
                sat_elevation_deg,
                az_step_deg=float(self.skycell_az_step_deg),
                el_step_deg=float(self.skycell_el_step_deg),
                el_min_deg=float(self.skycell_el_min_deg),
                el_max_deg=float(self.skycell_el_max_deg),
            )
            return ids
        raise RuntimeError(
            "Conditional sampling requested but skycell_mode is not configured "
            f"(got {self.skycell_mode!r})."
        )

    def conditional_statistics(self) -> Dict[str, Any]:
        """
        Return compact diagnostics for the optional conditional model.
        """
        if not self._has_conditional_model():
            return {
                "conditional_enabled": False,
                "n_skycells": int(self.n_skycells),
                "n_belts": int(self.n_belts),
            }

        assert self.group_raw_counts is not None
        assert self.belt_raw_counts is not None
        group_raw = self.group_raw_counts.astype(np.int64, copy=False)
        belt_raw = self.belt_raw_counts.astype(np.int64, copy=False)

        return {
            "conditional_enabled": True,
            "skycell_mode": str(self.skycell_mode),
            "n_skycells": int(self.n_skycells),
            "n_belts": int(self.n_belts),
            "n_groups_total": int(group_raw.size),
            "n_groups_nonempty": int(np.count_nonzero(group_raw)),
            "group_samples_raw_total": int(group_raw.sum()),
            "belt_samples_raw_total": int(belt_raw.sum()),
            "cond_min_group_samples": int(self.cond_min_group_samples),
            "cond_min_belt_samples": int(self.cond_min_belt_samples),
        }

    def _describe_group_ids(self, group_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decode group ids into belt id + skycell geometry descriptors.

        Group id convention:
          group_id = belt_id * n_skycells + skycell_id
        """
        gid = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if gid.size == 0:
            zf = np.empty(0, dtype=np.float64)
            zi = np.empty(0, dtype=np.int64)
            return {
                "group_id": zi,
                "belt_id": zi,
                "skycell_id": zi,
                "az_low_deg": zf,
                "az_high_deg": zf,
                "az_center_deg": zf,
                "el_low_deg": zf,
                "el_high_deg": zf,
                "el_center_deg": zf,
            }

        if int(self.n_skycells) <= 0:
            raise RuntimeError("Cannot describe groups: n_skycells is not configured.")

        n_groups = int(self.n_skycells) * int(self.n_belts)
        if np.any((gid < 0) | (gid >= n_groups)):
            raise ValueError("group_ids contain out-of-range values.")

        belt = (gid // int(self.n_skycells)).astype(np.int64, copy=False)
        sky = (gid % int(self.n_skycells)).astype(np.int64, copy=False)

        mode = str(self.skycell_mode).lower()
        if mode == "s1586":
            sky_desc = _describe_skycells_s1586(sky)
        elif mode == "regular":
            sky_desc = _describe_skycells_regular(
                sky,
                n_az=int(self.skycell_n_az),
                n_el=int(self.skycell_n_el),
                az_step_deg=float(self.skycell_az_step_deg),
                el_step_deg=float(self.skycell_el_step_deg),
                el_min_deg=float(self.skycell_el_min_deg),
            )
        else:
            raise RuntimeError(f"Unsupported skycell_mode={self.skycell_mode!r}.")

        out = {
            "group_id": gid.astype(np.int64, copy=False),
            "belt_id": belt,
            "skycell_id": sky,
            "az_low_deg": sky_desc["az_low_deg"].astype(np.float64, copy=False),
            "az_high_deg": sky_desc["az_high_deg"].astype(np.float64, copy=False),
            "az_center_deg": sky_desc["az_center_deg"].astype(np.float64, copy=False),
            "el_low_deg": sky_desc["el_low_deg"].astype(np.float64, copy=False),
            "el_high_deg": sky_desc["el_high_deg"].astype(np.float64, copy=False),
            "el_center_deg": sky_desc["el_center_deg"].astype(np.float64, copy=False),
        }
        return out

    def conditional_group_counts(self) -> np.ndarray:
        """
        Return raw conditional counts as a [n_belts, n_skycells] matrix.
        """
        if not self._has_conditional_model():
            raise RuntimeError("Conditional model is not available.")
        assert self.group_raw_counts is not None
        return self.group_raw_counts.reshape(int(self.n_belts), int(self.n_skycells))

    # --------------------------
    # Construction
    # --------------------------

    @classmethod
    def from_recovered(
        cls,
        sat_beta_recovered: Any,
        sat_alpha_recovered: Any,
        *,
        beta_bins: int = 900,
        alpha_bins: int = 3600,
        beta_range: Tuple[float, float] = (0.0, 90.0),
        alpha_range: Tuple[float, float] = (0.0, 360.0),
        wrap_alpha: bool = True,
        out_of_range: str = "drop",
        cdf_dtype: Any = np.float32,
        edges_dtype: Any = np.float32,
        smooth_sigma_beta: float = 0.0,
        smooth_sigma_alpha: float = 0.0,
        smooth_truncate: float = 3.0,
        auto_smooth: bool = False,
        auto_smooth_candidates: Any = None,
        auto_smooth_val_size: int = 250_000,
        auto_smooth_seed: int = 2025,
        show_comparison: bool = False,
        comparison_n_vis: int = 200_000,
        comparison_seed: int = 9876,
        comparison_slices: int = 32,
        save_prefix: Optional[str] = None,
        comparison_metrics_path: Optional[str] = None,
        plot_beta_bins: int = 250,
        plot_alpha_bins: int = 360,
        histogram: str = "auto",
        histogram_chunk_size: int = 5_000_000,
        filter_chunk_size: Optional[int] = None,
        progress: bool = False,
        progress_desc: str = "Joint sampler build",
        sat_azimuth_recovered: Any = None,
        sat_elevation_recovered: Any = None,
        sat_belt_id_recovered: Any = None,
        build_conditional: bool = False,
        skycell_mode: str = "s1586",
        skycell_az_step_deg: float = 3.0,
        skycell_el_step_deg: float = 3.0,
        skycell_el_min_deg: float = 0.0,
        skycell_el_max_deg: float = 90.0,
        cond_min_group_samples: int = 128,
        cond_min_belt_samples: int = 512,
        cond_max_group_samples: int = 4096,
        cond_max_belt_samples: int = 200_000,
        conditional_seed: int = 12345,
    ) -> "JointAngleSampler":
        """
        Build a sampler from recovered arrays.

        Default resolution:
          - alpha_bins=3600 (0.1 deg over 0..360)
          - beta_bins =900  (0.1 deg over 0..90)

        Smoothing (optional):
          - alpha axis uses circular boundary (wrap)
          - beta axis uses reflect boundary
          - set ``auto_smooth=True`` to choose sigma values from candidate pairs
            using validation negative log-likelihood on recovered samples.
          - ``auto_smooth_candidates`` accepts ``[(sigma_beta, sigma_alpha), ...]``;
            ``None`` uses a robust default grid.
          - ``auto_smooth_val_size`` controls tuning-set size (quality vs speed).

        Optional quality artifact persistence:
          - if ``show_comparison=True`` and ``comparison_metrics_path`` is set,
            comparison metrics are saved to a standalone ``.npz`` file.

        histogram:
          - "auto": use streaming accumulation for very large arrays, else numpy histogram2d
          - "stream": always streaming
          - "numpy": always numpy histogram2d
          - when streaming is active and Numba is available, a JIT histogram
            kernel can be selected for very large inputs

        Optional chunking/progress:
          - filter_chunk_size: chunk size for finite/range filtering; useful on very
            large arrays to reduce peak temporary memory.
          - progress=True enables tqdm-based progress bars for chunked filtering,
            streaming histogram accumulation and conditional pool construction.
          - auto smoothing candidate search also reports progress when enabled.

        Optional conditioned model:
          - Provide sat_azimuth_recovered, sat_elevation_recovered, sat_belt_id_recovered
          - enable via build_conditional=True (auto-enabled when all three arrays are passed)
          - supports skycell_mode="s1586" or "regular"
          - conditioned sampling then uses fallback chain:
                (belt, skycell) -> belt -> global

        Notes on expected invalid rows:
          - Simulation pipelines often mark non-links with NaN az/el and belt_id=-1.
          - These rows are intentionally excluded while building conditional pools.

        Performance/memory behavior
        ---------------------------
        - Histogram edges are uniform by construction, enabling fast uniform-bin
          index math and low-overhead streaming accumulation.
        - Conditional pools are built with bounded reservoir truncation
          (``cond_max_group_samples`` and ``cond_max_belt_samples``), keeping
          model size predictable even for very large training datasets.
        """
        beta_deg_all = _to_degrees(sat_beta_recovered).reshape(-1)
        alpha_deg_all = _to_degrees(sat_alpha_recovered).reshape(-1)
        if beta_deg_all.size != alpha_deg_all.size:
            raise ValueError("sat_beta_recovered and sat_alpha_recovered must have identical sizes.")

        have_any_cond = (
            sat_azimuth_recovered is not None or
            sat_elevation_recovered is not None or
            sat_belt_id_recovered is not None
        )
        have_all_cond = (
            sat_azimuth_recovered is not None and
            sat_elevation_recovered is not None and
            sat_belt_id_recovered is not None
        )
        if have_any_cond and not have_all_cond:
            raise ValueError(
                "Conditional model requires all three arrays: "
                "sat_azimuth_recovered, sat_elevation_recovered, sat_belt_id_recovered."
            )
        if have_all_cond:
            build_conditional = True

        az_deg_all = None
        el_deg_all = None
        belt_id_all = None
        if have_all_cond:
            az_deg_all = _to_degrees(sat_azimuth_recovered).reshape(-1)
            el_deg_all = _to_degrees(sat_elevation_recovered).reshape(-1)
            try:
                belt_id_all = np.asarray(sat_belt_id_recovered, dtype=np.float64).reshape(-1)
            except Exception as exc:
                raise ValueError("sat_belt_id_recovered must be numeric.") from exc
            n_ref = beta_deg_all.size
            if az_deg_all.size != n_ref or el_deg_all.size != n_ref or belt_id_all.size != n_ref:
                raise ValueError(
                    "Conditional arrays must match sat_beta/sat_alpha size after flattening."
                )
        elif build_conditional:
            raise ValueError(
                "build_conditional=True requires sat_azimuth_recovered, "
                "sat_elevation_recovered and sat_belt_id_recovered."
            )

        # --- unified filtering ---
        alpha_wrapped = bool(wrap_alpha)
        info = {"dropped": 0}
        b0, b1 = float(beta_range[0]), float(beta_range[1])
        a0, a1 = float(alpha_range[0]), float(alpha_range[1])
        if out_of_range not in ("drop", "clip", "keep"):
            raise ValueError("out_of_range must be one of: 'drop', 'clip', 'keep'.")
        if float(smooth_sigma_beta) < 0.0 or float(smooth_sigma_alpha) < 0.0:
            raise ValueError("smooth_sigma_beta and smooth_sigma_alpha must be non-negative.")

        beta_deg: np.ndarray
        alpha_deg: np.ndarray
        az_deg: Optional[np.ndarray] = None
        el_deg: Optional[np.ndarray] = None
        belt_id: Optional[np.ndarray] = None

        chunk_i: Optional[int]
        if filter_chunk_size is None:
            chunk_i = int(histogram_chunk_size) if (bool(progress) and beta_deg_all.size >= _FILTER_CHUNK_AUTO_ROWS) else None
        else:
            chunk_i = int(filter_chunk_size)
            if chunk_i <= 0:
                raise ValueError("filter_chunk_size must be a positive integer when provided.")

        if chunk_i is None:
            # Vectorized fast path for moderate-size arrays.
            m = _finite_mask(beta_deg_all, alpha_deg_all)
            if have_all_cond:
                m &= np.isfinite(az_deg_all) & np.isfinite(el_deg_all) & np.isfinite(belt_id_all)
                m &= (belt_id_all >= 0)

            beta_deg = beta_deg_all[m]
            alpha_deg = alpha_deg_all[m]

            if have_all_cond:
                az_deg = az_deg_all[m]
                el_deg = el_deg_all[m]
                belt_id = belt_id_all[m]

            if alpha_wrapped:
                alpha_deg = _wrap_alpha_deg(alpha_deg, alpha_range)

            if out_of_range == "clip":
                beta_deg = np.clip(beta_deg, b0, np.nextafter(b1, b0))
                alpha_deg = np.clip(alpha_deg, a0, np.nextafter(a1, a0))
            elif out_of_range == "drop":
                m_rng = (beta_deg >= b0) & (beta_deg < b1) & (alpha_deg >= a0) & (alpha_deg < a1)
                info["dropped"] = int(beta_deg.size - int(m_rng.sum()))
                beta_deg = beta_deg[m_rng]
                alpha_deg = alpha_deg[m_rng]
                if have_all_cond:
                    az_deg = az_deg[m_rng]
                    el_deg = el_deg[m_rng]
                    belt_id = belt_id[m_rng]
            elif out_of_range == "keep":
                pass
            else:
                raise ValueError("out_of_range must be one of: 'drop', 'clip', 'keep'.")
        else:
            # Chunked path for very large arrays: lower peak temporary memory
            # and optional tqdm progress feedback.
            beta_chunks: list[np.ndarray] = []
            alpha_chunks: list[np.ndarray] = []
            az_chunks: list[np.ndarray] = []
            el_chunks: list[np.ndarray] = []
            belt_chunks: list[np.ndarray] = []
            n_rows = int(beta_deg_all.size)
            n_chunks = int((n_rows + chunk_i - 1) // chunk_i)

            starts = _progress_iter(
                range(0, n_rows, chunk_i),
                enabled=bool(progress),
                total=n_chunks,
                desc=f"{progress_desc}: filter rows",
            )
            for start in starts:
                end = min(start + chunk_i, n_rows)
                b = beta_deg_all[start:end]
                a = alpha_deg_all[start:end]
                m = _finite_mask(b, a)

                az_c = None
                el_c = None
                belt_c = None
                if have_all_cond:
                    az_c = az_deg_all[start:end]
                    el_c = el_deg_all[start:end]
                    belt_c = belt_id_all[start:end]
                    m &= np.isfinite(az_c) & np.isfinite(el_c) & np.isfinite(belt_c)
                    m &= (belt_c >= 0)

                if not np.any(m):
                    continue

                b = b[m]
                a = a[m]
                if have_all_cond:
                    az_c = az_c[m]
                    el_c = el_c[m]
                    belt_c = belt_c[m]

                if alpha_wrapped:
                    a = _wrap_alpha_deg(a, alpha_range)

                if out_of_range == "clip":
                    b = np.clip(b, b0, np.nextafter(b1, b0))
                    a = np.clip(a, a0, np.nextafter(a1, a0))
                elif out_of_range == "drop":
                    m_rng = (b >= b0) & (b < b1) & (a >= a0) & (a < a1)
                    info["dropped"] += int(b.size - int(m_rng.sum()))
                    if not np.any(m_rng):
                        continue
                    b = b[m_rng]
                    a = a[m_rng]
                    if have_all_cond:
                        az_c = az_c[m_rng]
                        el_c = el_c[m_rng]
                        belt_c = belt_c[m_rng]
                elif out_of_range == "keep":
                    pass
                else:
                    raise ValueError("out_of_range must be one of: 'drop', 'clip', 'keep'.")

                beta_chunks.append(np.asarray(b, dtype=np.float64))
                alpha_chunks.append(np.asarray(a, dtype=np.float64))
                if have_all_cond:
                    az_chunks.append(np.asarray(az_c, dtype=np.float64))
                    el_chunks.append(np.asarray(el_c, dtype=np.float64))
                    belt_chunks.append(np.asarray(belt_c, dtype=np.float64))

            if not beta_chunks:
                raise ValueError("No valid samples left after filtering.")

            beta_deg = np.concatenate(beta_chunks)
            alpha_deg = np.concatenate(alpha_chunks)
            if have_all_cond:
                az_deg = np.concatenate(az_chunks)
                el_deg = np.concatenate(el_chunks)
                belt_id = np.concatenate(belt_chunks)

        if beta_deg.size == 0:
            raise ValueError("No valid samples left after filtering.")

        beta_edges = np.linspace(beta_range[0], beta_range[1], int(beta_bins) + 1, dtype=np.float64).astype(edges_dtype)
        alpha_edges = np.linspace(alpha_range[0], alpha_range[1], int(alpha_bins) + 1, dtype=np.float64).astype(edges_dtype)

        # Choose histogram mode
        if histogram not in ("auto", "stream", "numpy"):
            raise ValueError("histogram must be 'auto', 'stream', or 'numpy'.")

        use_stream = False
        if histogram == "stream":
            use_stream = True
        elif histogram == "numpy":
            use_stream = False
        else:
            # Auto: streaming is typically more robust for very large arrays
            use_stream = (beta_deg.size >= _STREAM_HIST_AUTO_ROWS)

        if use_stream:
            counts = _hist2d_stream_uniform_edges(
                beta_deg,
                alpha_deg,
                beta_edges.astype(np.float64),
                alpha_edges.astype(np.float64),
                chunk_size=int(histogram_chunk_size),
                progress=bool(progress),
                progress_desc=f"{progress_desc}: histogram",
            )
        else:
            counts, _, _ = np.histogram2d(
                beta_deg,
                alpha_deg,
                bins=[beta_edges.astype(np.float64), alpha_edges.astype(np.float64)],
            )
            counts = counts.astype(np.float64, copy=False)

        if bool(auto_smooth):
            if int(auto_smooth_val_size) <= 0:
                raise ValueError("auto_smooth_val_size must be positive.")
            rng_auto = np.random.default_rng(int(auto_smooth_seed))
            n_val = int(min(int(auto_smooth_val_size), beta_deg.size))
            if beta_deg.size > n_val:
                idx_val = rng_auto.choice(beta_deg.size, size=n_val, replace=False)
                beta_val = beta_deg[idx_val]
                alpha_val = alpha_deg[idx_val]
            else:
                beta_val = beta_deg
                alpha_val = alpha_deg

            cand_pairs = _prepare_auto_smooth_candidates(
                auto_smooth_candidates,
                fixed_sigma_beta=float(smooth_sigma_beta),
                fixed_sigma_alpha=float(smooth_sigma_alpha),
            )
            best_sb, best_sa, counts_best, _ = _auto_tune_smoothing_sigmas(
                counts,
                beta_edges.astype(np.float64, copy=False),
                alpha_edges.astype(np.float64, copy=False),
                beta_val,
                alpha_val,
                candidates=cand_pairs,
                smooth_truncate=float(smooth_truncate),
                progress=bool(progress),
                progress_desc=f"{progress_desc}: auto smooth",
            )
            smooth_sigma_beta = float(best_sb)
            smooth_sigma_alpha = float(best_sa)
            counts = counts_best
        elif smooth_sigma_beta > 0.0 or smooth_sigma_alpha > 0.0:
            counts = _smooth_2d_gaussian(
                counts,
                sigma_beta=float(smooth_sigma_beta),
                sigma_alpha=float(smooth_sigma_alpha),
                truncate=float(smooth_truncate),
            )

        total = float(counts.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Joint histogram counts sum to zero (or non-finite).")

        n_alpha = int(counts.shape[1])
        pmf = (counts.ravel() / total).astype(np.float64, copy=False)

        cdf = np.cumsum(pmf, dtype=np.float64)
        cdf[-1] = 1.0
        cdf = cdf.astype(cdf_dtype, copy=False)

        sampler_kwargs: Dict[str, Any] = dict(
            beta_edges=beta_edges,
            alpha_edges=alpha_edges,
            cdf=cdf,
            n_alpha=n_alpha,
            beta_range=(float(beta_range[0]), float(beta_range[1])),
            alpha_range=(float(alpha_range[0]), float(alpha_range[1])),
            alpha_wrapped=alpha_wrapped,
            smooth_sigma_beta=float(smooth_sigma_beta),
            smooth_sigma_alpha=float(smooth_sigma_alpha),
            smooth_truncate=float(smooth_truncate),
            conditional_enabled=False,
            skycell_mode="none",
            n_skycells=0,
            n_belts=0,
            skycell_az_step_deg=0.0,
            skycell_el_step_deg=0.0,
            skycell_el_min_deg=0.0,
            skycell_el_max_deg=90.0,
            skycell_n_az=0,
            skycell_n_el=0,
            cond_min_group_samples=0,
            cond_min_belt_samples=0,
            group_ptr=None,
            group_beta_pool=None,
            group_alpha_pool=None,
            group_raw_counts=None,
            belt_ptr=None,
            belt_beta_pool=None,
            belt_alpha_pool=None,
            belt_raw_counts=None,
        )

        if build_conditional:
            if int(cond_max_group_samples) <= 0 or int(cond_max_belt_samples) <= 0:
                raise ValueError("cond_max_group_samples and cond_max_belt_samples must be positive.")
            if int(cond_min_group_samples) < 0 or int(cond_min_belt_samples) < 0:
                raise ValueError("cond_min_group_samples and cond_min_belt_samples must be non-negative.")

            belt_id_f = np.asarray(belt_id, dtype=np.float64).reshape(-1)
            belt_id_i = np.rint(belt_id_f).astype(np.int64, copy=False)
            if np.any(np.abs(belt_id_f - belt_id_i.astype(np.float64, copy=False)) > 1e-6):
                raise ValueError("sat_belt_id_recovered must contain integer-like belt IDs.")
            if belt_id_i.size == 0:
                raise ValueError("No conditional rows available after filtering.")

            sky_mode = str(skycell_mode).lower().strip()
            if sky_mode == "s1586":
                sky_id = _skycell_id_s1586(az_deg, el_deg)
                n_skycells = int(_S1586_N_CELLS)
                sky_n_az = 0
                sky_n_el = 0
                sky_az_step = 0.0
                sky_el_step = 0.0
            elif sky_mode == "regular":
                sky_id, sky_n_az, sky_n_el = _skycell_id_regular(
                    az_deg,
                    el_deg,
                    az_step_deg=float(skycell_az_step_deg),
                    el_step_deg=float(skycell_el_step_deg),
                    el_min_deg=float(skycell_el_min_deg),
                    el_max_deg=float(skycell_el_max_deg),
                )
                n_skycells = int(sky_n_az * sky_n_el)
                sky_az_step = float(skycell_az_step_deg)
                sky_el_step = float(skycell_el_step_deg)
            else:
                raise ValueError("skycell_mode must be 's1586' or 'regular'.")

            cond_valid = (sky_id >= 0) & (belt_id_i >= 0)
            if not np.any(cond_valid):
                raise ValueError("Conditional rows exist but none map to valid sky cells.")

            beta_c = beta_deg[cond_valid]
            alpha_c = alpha_deg[cond_valid]
            belt_c = belt_id_i[cond_valid].astype(np.int64, copy=False)
            sky_c = sky_id[cond_valid].astype(np.int64, copy=False)

            n_belts = int(belt_c.max()) + 1
            if n_belts <= 0:
                raise ValueError("Could not infer a positive number of belts from sat_belt_id_recovered.")

            group_id = belt_c * np.int64(n_skycells) + sky_c
            n_groups = int(n_belts * n_skycells)
            cond_rng = np.random.default_rng(int(conditional_seed))

            group_ptr, group_beta_pool, group_alpha_pool, group_raw_counts = _build_indexed_reservoir_pools(
                group_id,
                beta_c,
                alpha_c,
                n_ids=int(n_groups),
                max_samples_per_id=int(cond_max_group_samples),
                rng=cond_rng,
                progress=bool(progress),
                progress_desc=f"{progress_desc}: group pools",
            )
            belt_ptr, belt_beta_pool, belt_alpha_pool, belt_raw_counts = _build_indexed_reservoir_pools(
                belt_c,
                beta_c,
                alpha_c,
                n_ids=int(n_belts),
                max_samples_per_id=int(cond_max_belt_samples),
                rng=cond_rng,
                progress=bool(progress),
                progress_desc=f"{progress_desc}: belt pools",
            )

            sampler_kwargs.update(
                conditional_enabled=True,
                skycell_mode=sky_mode,
                n_skycells=int(n_skycells),
                n_belts=int(n_belts),
                skycell_az_step_deg=float(sky_az_step),
                skycell_el_step_deg=float(sky_el_step),
                skycell_el_min_deg=float(skycell_el_min_deg),
                skycell_el_max_deg=float(skycell_el_max_deg),
                skycell_n_az=int(sky_n_az),
                skycell_n_el=int(sky_n_el),
                cond_min_group_samples=int(cond_min_group_samples),
                cond_min_belt_samples=int(cond_min_belt_samples),
                group_ptr=group_ptr.astype(np.int64, copy=False),
                group_beta_pool=group_beta_pool.astype(np.float32, copy=False),
                group_alpha_pool=group_alpha_pool.astype(np.float32, copy=False),
                group_raw_counts=group_raw_counts.astype(np.int32, copy=False),
                belt_ptr=belt_ptr.astype(np.int64, copy=False),
                belt_beta_pool=belt_beta_pool.astype(np.float32, copy=False),
                belt_alpha_pool=belt_alpha_pool.astype(np.float32, copy=False),
                belt_raw_counts=belt_raw_counts.astype(np.int32, copy=False),
            )

        sampler = cls(**sampler_kwargs)

        if show_comparison:
            sampler.show_comparison(
                beta_deg, alpha_deg,
                n_vis=int(comparison_n_vis),
                seed=int(comparison_seed),
                n_slices=int(comparison_slices),
                save_prefix=save_prefix,
                metrics_path=comparison_metrics_path,
                dropped=int(info.get("dropped", 0)),
                plot_beta_bins=int(plot_beta_bins),
                plot_alpha_bins=int(plot_alpha_bins),
            )

        return sampler

    # --------------------------
    # Sampling
    # --------------------------

    def sample(
        self,
        rng: np.random.Generator,
        size: SizeLike = None,
        *,
        chunk: Optional[int] = None,
        dtype: Any = np.float32,
        sat_azimuth_deg: Any = None,
        sat_elevation_deg: Any = None,
        sat_belt_id: Any = None,
        return_context: bool = False,
    ) -> Any:
        """
        Sample (beta_deg, alpha_deg) with output shape matching `size`.

        Examples:
          beta, alpha = sampler.sample(rng, size=200000)
          beta, alpha = sampler.sample(rng, size=some_array.shape)
          beta, alpha = sampler.sample(rng, size=(T, O, S, B))

        Conditioned usage:
          beta, alpha = sampler.sample(
              rng,
              sat_azimuth_deg=sat_az,
              sat_elevation_deg=sat_el,
              sat_belt_id=sat_belt,
              size=pool_size,
          )

        In conditioned mode, `size` is interpreted as "samples per condition row".
        Output shape becomes:
          sat_azimuth_deg.shape + normalize(size)

        Set return_context=True to additionally receive:
          (sat_azimuth_deg, sat_elevation_deg, sat_belt_id, skycell_id)
        aligned with sampled beta/alpha outputs.
        """
        have_any_cond = (
            sat_azimuth_deg is not None or
            sat_elevation_deg is not None or
            sat_belt_id is not None
        )
        have_all_cond = (
            sat_azimuth_deg is not None and
            sat_elevation_deg is not None and
            sat_belt_id is not None
        )
        if have_any_cond and not have_all_cond:
            raise ValueError(
                "Conditioned sampling requires all three inputs: "
                "sat_azimuth_deg, sat_elevation_deg, sat_belt_id."
            )

        if have_all_cond:
            return self.sample_conditioned(
                rng,
                sat_azimuth_deg=sat_azimuth_deg,
                sat_elevation_deg=sat_elevation_deg,
                sat_belt_id=sat_belt_id,
                size=size,
                chunk=chunk,
                dtype=dtype,
                return_context=return_context,
            )

        if return_context:
            raise ValueError("return_context=True requires conditioned inputs.")

        shape, n = _normalize_size(size)
        beta_flat, alpha_flat = self._sample_unconditional_flat(
            rng, n, dtype=dtype, chunk=chunk
        )

        if shape == ():
            return dtype(beta_flat[0]), dtype(alpha_flat[0])
        return beta_flat.reshape(shape), alpha_flat.reshape(shape)

    def sample_conditioned(
        self,
        rng: np.random.Generator,
        *,
        sat_azimuth_deg: Any,
        sat_elevation_deg: Any,
        sat_belt_id: Any,
        size: SizeLike = None,
        chunk: Optional[int] = None,
        dtype: Any = np.float32,
        return_context: bool = False,
    ) -> Any:
        """
        Conditioned sampling of (beta_deg, alpha_deg) by observer-frame sky position
        and belt ID.

        Fallback chain per condition:
          1) if enough samples in (belt, skycell): sample from that pool
          2) else if enough samples in belt: sample from belt pool
          3) else: sample from global unconditional CDF

        Invalid condition rows (NaN angles, invalid/non-integer belt, out-of-grid skycell)
        return NaN for beta/alpha.

        Memory behavior
        ---------------
        Work arrays are allocated as ``(n_conditions, n_per_condition)`` once.
        Sampling then fills rows in grouped batches (group -> belt -> global),
        which keeps temporary allocations bounded and avoids Python loops over
        individual links.

        Shape semantics
        ---------------
        Let:
          cond_shape = sat_azimuth_deg.shape  (same as sat_elevation_deg/sat_belt_id)
          sample_shape = normalized `size`

        Then output beta/alpha shape is:
          cond_shape + sample_shape

        Examples:
          - size=None      -> one sample per condition, output shape cond_shape
          - size=128       -> 128 samples per condition, output shape cond_shape + (128,)
          - size=(4, 32)   -> output shape cond_shape + (4, 32)

        Returns
        -------
        If return_context=False:
          (beta_deg, alpha_deg)
        If return_context=True:
          (beta_deg, alpha_deg, sat_azimuth_deg_ctx, sat_elevation_deg_ctx, sat_belt_id_ctx, skycell_id_ctx)
        Context arrays are returned at condition resolution (shape = cond_shape), not
        replicated across sample_shape, to avoid unnecessary memory blow-up.
        """
        if not self._has_conditional_model():
            raise RuntimeError(
                "This sampler does not contain a conditional model. "
                "Rebuild using from_recovered(..., sat_azimuth_recovered=..., "
                "sat_elevation_recovered=..., sat_belt_id_recovered=..., build_conditional=True)."
            )
        assert self.group_raw_counts is not None
        assert self.group_ptr is not None
        assert self.group_beta_pool is not None
        assert self.group_alpha_pool is not None
        assert self.belt_raw_counts is not None
        assert self.belt_ptr is not None
        assert self.belt_beta_pool is not None
        assert self.belt_alpha_pool is not None

        az_arr = _to_degrees(sat_azimuth_deg)
        el_arr = _to_degrees(sat_elevation_deg)
        belt_arr = np.asarray(sat_belt_id)
        try:
            belt_num = np.asarray(belt_arr, dtype=np.float64)
        except Exception as exc:
            raise ValueError("sat_belt_id must be numeric.") from exc
        if az_arr.shape != el_arr.shape or az_arr.shape != belt_arr.shape:
            raise ValueError("sat_azimuth_deg, sat_elevation_deg and sat_belt_id must have identical shapes.")

        shape_cond = tuple(int(x) for x in az_arr.shape)
        n_cond = int(np.prod(shape_cond, dtype=np.int64)) if shape_cond else 1
        sample_shape, n_per = _normalize_size(size)
        if size is None:
            sample_shape = ()
            n_per = 1

        az = az_arr.reshape(-1).astype(np.float64, copy=False)
        el = el_arr.reshape(-1).astype(np.float64, copy=False)
        belt_f = belt_num.reshape(-1)

        # Work in 2D [condition_row, sample_id] for efficient grouped writes.
        beta_mat = np.full((n_cond, n_per), np.nan, dtype=dtype)
        alpha_mat = np.full((n_cond, n_per), np.nan, dtype=dtype)

        m = np.isfinite(az) & np.isfinite(el) & np.isfinite(belt_f)
        if np.any(m):
            valid_idx = np.nonzero(m)[0].astype(np.int64, copy=False)
            belt_valid_f = belt_f[m]
            belt_i = np.rint(belt_valid_f).astype(np.int64, copy=False)
            belt_is_int = np.abs(belt_valid_f - belt_i.astype(np.float64, copy=False)) <= 1e-6

            in_belt = belt_is_int & (belt_i >= 0) & (belt_i < int(self.n_belts))
            if np.any(in_belt):
                idx_belt = valid_idx[in_belt]
                belt_v = belt_i[in_belt]

                sky_v = self._skycell_id_from_observer_angles(az[idx_belt], el[idx_belt]).astype(np.int64, copy=False)
                in_sky = (sky_v >= 0) & (sky_v < int(self.n_skycells))

                if np.any(in_sky):
                    idx_use = idx_belt[in_sky]
                    belt_use = belt_v[in_sky]
                    sky_use = sky_v[in_sky]

                    group_id = belt_use * np.int64(self.n_skycells) + sky_use

                    group_raw = self.group_raw_counts.astype(np.int64, copy=False)
                    group_ptr = self.group_ptr.astype(np.int64, copy=False)
                    belt_raw = self.belt_raw_counts.astype(np.int64, copy=False)
                    belt_ptr = self.belt_ptr.astype(np.int64, copy=False)

                    use_group = (
                        (group_raw[group_id] >= int(self.cond_min_group_samples)) &
                        (group_ptr[group_id + 1] > group_ptr[group_id])
                    )
                    use_belt = (
                        ~use_group &
                        (belt_raw[belt_use] >= int(self.cond_min_belt_samples)) &
                        (belt_ptr[belt_use + 1] > belt_ptr[belt_use])
                    )
                    use_global = ~(use_group | use_belt)

                    # ----------------------------------------------------------
                    # 1) Group-conditioned rows
                    # ----------------------------------------------------------
                    if np.any(use_group):
                        rows = idx_use[use_group].astype(np.int64, copy=False)
                        gids = group_id[use_group].astype(np.int64, copy=False)

                        ord_g = np.argsort(gids, kind="mergesort")
                        gids_s = gids[ord_g]
                        rows_s = rows[ord_g]

                        cuts = np.flatnonzero(np.diff(gids_s)) + 1
                        starts = np.concatenate(([0], cuts))
                        ends = np.concatenate((cuts, [gids_s.size]))

                        for s, e in zip(starts, ends):
                            gid = int(gids_s[s])
                            p0 = int(group_ptr[gid])
                            p1 = int(group_ptr[gid + 1])
                            if p1 <= p0:
                                continue
                            rr = rows_s[s:e]
                            draws = rng.integers(p0, p1, size=(rr.size, n_per))
                            beta_mat[rr, :] = self.group_beta_pool[draws]
                            alpha_mat[rr, :] = self.group_alpha_pool[draws]

                    # ----------------------------------------------------------
                    # 2) Belt-conditioned fallback rows
                    # ----------------------------------------------------------
                    if np.any(use_belt):
                        rows = idx_use[use_belt].astype(np.int64, copy=False)
                        bids = belt_use[use_belt].astype(np.int64, copy=False)

                        ord_b = np.argsort(bids, kind="mergesort")
                        bids_s = bids[ord_b]
                        rows_s = rows[ord_b]

                        cuts = np.flatnonzero(np.diff(bids_s)) + 1
                        starts = np.concatenate(([0], cuts))
                        ends = np.concatenate((cuts, [bids_s.size]))

                        for s, e in zip(starts, ends):
                            bid = int(bids_s[s])
                            p0 = int(belt_ptr[bid])
                            p1 = int(belt_ptr[bid + 1])
                            if p1 <= p0:
                                continue
                            rr = rows_s[s:e]
                            draws = rng.integers(p0, p1, size=(rr.size, n_per))
                            beta_mat[rr, :] = self.belt_beta_pool[draws]
                            alpha_mat[rr, :] = self.belt_alpha_pool[draws]

                    # ----------------------------------------------------------
                    # 3) Global fallback rows
                    # ----------------------------------------------------------
                    if np.any(use_global):
                        rows = idx_use[use_global].astype(np.int64, copy=False)
                        n_global = int(rows.size)
                        bg, ag = self._sample_unconditional_flat(
                            rng, n_global * n_per, dtype=dtype, chunk=chunk
                        )
                        beta_mat[rows, :] = bg.reshape(n_global, n_per)
                        alpha_mat[rows, :] = ag.reshape(n_global, n_per)

        # Materialize final output shape:
        #   cond_shape (+ sample_shape, if requested)
        if sample_shape == ():
            beta_out = beta_mat[:, 0].reshape(shape_cond)
            alpha_out = alpha_mat[:, 0].reshape(shape_cond)
        else:
            beta_out = beta_mat.reshape(shape_cond + sample_shape)
            alpha_out = alpha_mat.reshape(shape_cond + sample_shape)

        if not return_context:
            return beta_out, alpha_out

        # Optional context payload for downstream samplers/pipelines.
        # Context is returned at condition resolution only (no replication over sample axis).
        az_ctx = np.remainder(az, 360.0).astype(dtype, copy=False).reshape(shape_cond)
        el_ctx = el.astype(dtype, copy=False).reshape(shape_cond)

        belt_ctx = np.full(n_cond, -1, dtype=np.int32)
        mb = np.isfinite(belt_f)
        if np.any(mb):
            belt_tmp_f = belt_f[mb]
            belt_tmp_i = np.rint(belt_tmp_f).astype(np.int32, copy=False)
            belt_ok = np.abs(belt_tmp_f - belt_tmp_i.astype(np.float64, copy=False)) <= 1e-6
            mb_idx = np.nonzero(mb)[0]
            belt_ctx[mb_idx[belt_ok]] = belt_tmp_i[belt_ok]
        belt_ctx[(belt_ctx < 0) | (belt_ctx >= int(self.n_belts))] = -1
        belt_ctx = belt_ctx.reshape(shape_cond)

        sky_ctx = np.full(n_cond, -1, dtype=np.int32)
        msky = np.isfinite(az) & np.isfinite(el) & (belt_ctx.reshape(-1) >= 0)
        if np.any(msky):
            sky_ctx[np.nonzero(msky)[0]] = self._skycell_id_from_observer_angles(az[msky], el[msky])
        sky_ctx = sky_ctx.reshape(shape_cond)

        return beta_out, alpha_out, az_ctx, el_ctx, belt_ctx, sky_ctx

    # --------------------------
    # Persistence (minimal/full)
    # --------------------------

    @staticmethod
    def _metric_value_to_ndarray(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if np.isscalar(value):
            return np.asarray(value)
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype == object:
                raise TypeError("metrics list/tuple values must not produce object-dtype arrays.")
            return arr
        raise TypeError(f"Unsupported metric payload type: {type(value)!r}")

    @classmethod
    def save_metrics(
        cls,
        metrics: Dict[str, Any],
        path: str,
        *,
        compressed: bool = True,
    ) -> None:
        """
        Save a metrics dictionary to ``.npz`` for headless or post-run analysis.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, np.ndarray] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            payload[str(key)] = cls._metric_value_to_ndarray(value)

        if not payload:
            raise ValueError("metrics dictionary is empty; nothing to save.")

        save_fn = np.savez_compressed if compressed else np.savez
        save_fn(str(path_obj), **payload)

    @staticmethod
    def load_metrics(path: str) -> Dict[str, Any]:
        """
        Load metrics saved via :meth:`save_metrics`.
        """
        out: Dict[str, Any] = {}
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                arr = np.asarray(data[key])
                out[key] = arr.item() if arr.ndim == 0 else arr
        return out

    def save(self, path: str, *, mode: str = "minimal", compressed: bool = True) -> None:
        """
        Save sampler to disk as .npz.

        mode:
          - "minimal": store only what is required for sampling (smallest).
          - "full": also store smoothing parameters (still tiny).

        The file is always loadable without Numba.
        """
        if mode not in ("minimal", "full"):
            raise ValueError("mode must be 'minimal' or 'full'.")

        save_fn = np.savez_compressed if compressed else np.savez

        payload = dict(
            beta_edges=self.beta_edges,
            alpha_edges=self.alpha_edges,
            cdf=self.cdf,
            n_alpha=np.int64(self.n_alpha),
            beta_range=np.asarray(self.beta_range, dtype=np.float64),
            alpha_range=np.asarray(self.alpha_range, dtype=np.float64),
            alpha_wrapped=np.int8(1 if self.alpha_wrapped else 0),
            conditional_enabled=np.int8(1 if self._has_conditional_model() else 0),
            skycell_mode=np.asarray(str(self.skycell_mode)),
            n_skycells=np.int64(self.n_skycells),
            n_belts=np.int64(self.n_belts),
            skycell_az_step_deg=np.float64(self.skycell_az_step_deg),
            skycell_el_step_deg=np.float64(self.skycell_el_step_deg),
            skycell_el_min_deg=np.float64(self.skycell_el_min_deg),
            skycell_el_max_deg=np.float64(self.skycell_el_max_deg),
            skycell_n_az=np.int64(self.skycell_n_az),
            skycell_n_el=np.int64(self.skycell_n_el),
            cond_min_group_samples=np.int64(self.cond_min_group_samples),
            cond_min_belt_samples=np.int64(self.cond_min_belt_samples),
        )

        if mode == "full":
            payload.update(
                smooth_sigma_beta=np.float64(self.smooth_sigma_beta),
                smooth_sigma_alpha=np.float64(self.smooth_sigma_alpha),
                smooth_truncate=np.float64(self.smooth_truncate),
            )

        if self._has_conditional_model():
            payload.update(
                group_ptr=self.group_ptr,
                group_beta_pool=self.group_beta_pool,
                group_alpha_pool=self.group_alpha_pool,
                group_raw_counts=self.group_raw_counts,
                belt_ptr=self.belt_ptr,
                belt_beta_pool=self.belt_beta_pool,
                belt_alpha_pool=self.belt_alpha_pool,
                belt_raw_counts=self.belt_raw_counts,
            )

        save_fn(path, **payload)

    @classmethod
    def load(cls, path: str, *, mode: str = "minimal") -> "JointAngleSampler":
        """
        Load sampler from .npz created by `save()`.

        mode:
          - "minimal": ignore any optional metadata even if present in file.
          - "full": if metadata exists, load it.
        """
        if mode not in ("minimal", "full"):
            raise ValueError("mode must be 'minimal' or 'full'.")

        data = np.load(path, allow_pickle=False)

        beta_edges = data["beta_edges"]
        alpha_edges = data["alpha_edges"]
        cdf = data["cdf"]
        n_alpha = int(data["n_alpha"])

        beta_range = tuple(np.asarray(data["beta_range"], dtype=np.float64).tolist())
        alpha_range = tuple(np.asarray(data["alpha_range"], dtype=np.float64).tolist())
        alpha_wrapped = bool(int(data["alpha_wrapped"]))

        smooth_sigma_beta = 0.0
        smooth_sigma_alpha = 0.0
        smooth_truncate = 3.0

        if mode == "full":
            if "smooth_sigma_beta" in data:
                smooth_sigma_beta = float(np.asarray(data["smooth_sigma_beta"]).item())
            if "smooth_sigma_alpha" in data:
                smooth_sigma_alpha = float(np.asarray(data["smooth_sigma_alpha"]).item())
            if "smooth_truncate" in data:
                smooth_truncate = float(np.asarray(data["smooth_truncate"]).item())

        conditional_enabled = bool(int(np.asarray(data["conditional_enabled"]).item())) if "conditional_enabled" in data else False
        skycell_mode = str(np.asarray(data["skycell_mode"]).item()) if "skycell_mode" in data else "none"
        n_skycells = int(np.asarray(data["n_skycells"]).item()) if "n_skycells" in data else 0
        n_belts = int(np.asarray(data["n_belts"]).item()) if "n_belts" in data else 0

        skycell_az_step_deg = float(np.asarray(data["skycell_az_step_deg"]).item()) if "skycell_az_step_deg" in data else 0.0
        skycell_el_step_deg = float(np.asarray(data["skycell_el_step_deg"]).item()) if "skycell_el_step_deg" in data else 0.0
        skycell_el_min_deg = float(np.asarray(data["skycell_el_min_deg"]).item()) if "skycell_el_min_deg" in data else 0.0
        skycell_el_max_deg = float(np.asarray(data["skycell_el_max_deg"]).item()) if "skycell_el_max_deg" in data else 90.0
        skycell_n_az = int(np.asarray(data["skycell_n_az"]).item()) if "skycell_n_az" in data else 0
        skycell_n_el = int(np.asarray(data["skycell_n_el"]).item()) if "skycell_n_el" in data else 0

        cond_min_group_samples = int(np.asarray(data["cond_min_group_samples"]).item()) if "cond_min_group_samples" in data else 0
        cond_min_belt_samples = int(np.asarray(data["cond_min_belt_samples"]).item()) if "cond_min_belt_samples" in data else 0

        group_ptr = data["group_ptr"] if "group_ptr" in data else None
        group_beta_pool = data["group_beta_pool"] if "group_beta_pool" in data else None
        group_alpha_pool = data["group_alpha_pool"] if "group_alpha_pool" in data else None
        group_raw_counts = data["group_raw_counts"] if "group_raw_counts" in data else None
        belt_ptr = data["belt_ptr"] if "belt_ptr" in data else None
        belt_beta_pool = data["belt_beta_pool"] if "belt_beta_pool" in data else None
        belt_alpha_pool = data["belt_alpha_pool"] if "belt_alpha_pool" in data else None
        belt_raw_counts = data["belt_raw_counts"] if "belt_raw_counts" in data else None

        if conditional_enabled:
            must_have = (
                group_ptr, group_beta_pool, group_alpha_pool, group_raw_counts,
                belt_ptr, belt_beta_pool, belt_alpha_pool, belt_raw_counts,
            )
            if not all(x is not None for x in must_have):
                # File claims conditional mode, but payload is incomplete.
                # Fall back to unconditional behavior for robustness.
                conditional_enabled = False

        return cls(
            beta_edges=beta_edges,
            alpha_edges=alpha_edges,
            cdf=cdf,
            n_alpha=n_alpha,
            beta_range=(float(beta_range[0]), float(beta_range[1])),
            alpha_range=(float(alpha_range[0]), float(alpha_range[1])),
            alpha_wrapped=alpha_wrapped,
            smooth_sigma_beta=float(smooth_sigma_beta),
            smooth_sigma_alpha=float(smooth_sigma_alpha),
            smooth_truncate=float(smooth_truncate),
            conditional_enabled=bool(conditional_enabled),
            skycell_mode=str(skycell_mode),
            n_skycells=int(n_skycells),
            n_belts=int(n_belts),
            skycell_az_step_deg=float(skycell_az_step_deg),
            skycell_el_step_deg=float(skycell_el_step_deg),
            skycell_el_min_deg=float(skycell_el_min_deg),
            skycell_el_max_deg=float(skycell_el_max_deg),
            skycell_n_az=int(skycell_n_az),
            skycell_n_el=int(skycell_n_el),
            cond_min_group_samples=int(cond_min_group_samples),
            cond_min_belt_samples=int(cond_min_belt_samples),
            group_ptr=(group_ptr.astype(np.int64, copy=False) if group_ptr is not None else None),
            group_beta_pool=(group_beta_pool.astype(np.float32, copy=False) if group_beta_pool is not None else None),
            group_alpha_pool=(group_alpha_pool.astype(np.float32, copy=False) if group_alpha_pool is not None else None),
            group_raw_counts=(group_raw_counts.astype(np.int32, copy=False) if group_raw_counts is not None else None),
            belt_ptr=(belt_ptr.astype(np.int64, copy=False) if belt_ptr is not None else None),
            belt_beta_pool=(belt_beta_pool.astype(np.float32, copy=False) if belt_beta_pool is not None else None),
            belt_alpha_pool=(belt_alpha_pool.astype(np.float32, copy=False) if belt_alpha_pool is not None else None),
            belt_raw_counts=(belt_raw_counts.astype(np.int32, copy=False) if belt_raw_counts is not None else None),
        )

    # --------------------------
    # Quality evaluation (SW leading + resolution floor)
    # --------------------------

    def evaluate_quality(
        self,
        beta_real_deg: Any,
        alpha_real_deg: Any,
        *,
        n_vis: int = 200_000,
        seed: int = 9876,
        n_slices: int = 64,
        alpha_weight: float = 1.0,
        baseline_trials: int = 10,
        auto_ratio_scale: bool = True,
        ratio_scale: float = 0.5,
        auto_z: float = 2.0,
        auto_target_score: float = 85.0,
        compute_hist_metrics: bool = True,
        hist_beta_bins: int = 180,
        hist_alpha_bins: int = 360,
    ) -> Dict[str, Any]:
        """
        Evaluate sampler quality with a SW-leading metric and a resolution-aware floor.

        Definitions:
          - SW_rs = SW(real, sampler)
          - SW_rr = SW(real1, real2)  (finite-sample noise baseline)
          - SW_floor = SW(real, quantized(real))  (discretization floor for this binning)

        Effective denominator:
          - SW_den = max(SW_rr_mean, SW_floor_mean)

        Effective ratio:
          - ratio_eff = SW_rs / SW_den

        Score is derived from ratio_eff.
        """
        rng = np.random.default_rng(int(seed))

        beta = _to_degrees(beta_real_deg).reshape(-1)
        alpha = _to_degrees(alpha_real_deg).reshape(-1)

        m = _finite_mask(beta, alpha)
        beta = beta[m]
        alpha = alpha[m]

        if self.alpha_wrapped:
            alpha = _wrap_alpha_deg(alpha, self.alpha_range)

        if beta.size == 0:
            raise ValueError("No finite real samples provided.")

        n = int(min(n_vis, beta.size))
        beta_scale = float(self.beta_range[1] - self.beta_range[0])

        # Fixed set of directions for all SW computations (stability)
        directions = _make_unit_directions(3, int(n_slices), rng)

        bt = int(max(1, baseline_trials))
        sw_rr_vals = np.empty(bt, dtype=np.float64)
        sw_floor_vals = np.empty(bt, dtype=np.float64)

        # Fixed reference set for sampler comparison
        b_ref, a_ref, _, _ = _subsample_two_sets(beta, alpha, n=n, rng=rng)

        # Compute quantized version of the same reference (bin centers)
        b_ref_q = _bin_centers_from_edges(b_ref, self.beta_edges.astype(np.float64))
        a_ref_q = _bin_centers_from_edges(a_ref, self.alpha_edges.astype(np.float64))

        # Baseline trials (real-real) and floor trials (real-quantized)
        for t in range(bt):
            b1, a1, b2, a2 = _subsample_two_sets(beta, alpha, n=n, rng=rng)

            rr = sliced_wasserstein(
                b1, a1, b2, a2,
                directions=directions,
                circular_alpha=True,
                beta_scale=beta_scale,
                alpha_weight=float(alpha_weight),
                rng=rng,
            )
            sw_rr_vals[t] = rr["sw_mean"]

            b1q = _bin_centers_from_edges(b1, self.beta_edges.astype(np.float64))
            a1q = _bin_centers_from_edges(a1, self.alpha_edges.astype(np.float64))
            fl = sliced_wasserstein(
                b1, a1, b1q, a1q,
                directions=directions,
                circular_alpha=True,
                beta_scale=beta_scale,
                alpha_weight=float(alpha_weight),
                rng=rng,
            )
            sw_floor_vals[t] = fl["sw_mean"]

        sw_rr_mean = float(sw_rr_vals.mean())
        sw_rr_std = float(sw_rr_vals.std(ddof=0))
        sw_floor_mean = float(sw_floor_vals.mean())
        sw_floor_std = float(sw_floor_vals.std(ddof=0))

        # Sampler vs real (reference set)
        b_s, a_s = self.sample(rng, size=b_ref.shape[0])
        rs = sliced_wasserstein(
            b_ref, a_ref, b_s, a_s,
            directions=directions,
            circular_alpha=True,
            beta_scale=beta_scale,
            alpha_weight=float(alpha_weight),
            rng=rng,
        )
        sw_rs_mean = float(rs["sw_mean"])
        sw_rs_std = float(rs["sw_std"])

        # Choose denominator: do not demand matching below the discretization floor
        if sw_rr_mean >= sw_floor_mean:
            denom_mean = sw_rr_mean
            denom_std = sw_rr_std
            denom_kind = "real-real"
        else:
            denom_mean = sw_floor_mean
            denom_std = sw_floor_std
            denom_kind = "quantization-floor"

        denom_mean = float(max(denom_mean, 1e-15))
        ratio_eff = sw_rs_mean / denom_mean

        if auto_ratio_scale:
            ratio_scale_used = _auto_ratio_scale_from_baseline(
                denom_mean, denom_std,
                z=float(auto_z),
                target_score=float(auto_target_score),
            )
        else:
            ratio_scale_used = float(ratio_scale)

        score_pack = _score_from_ratio(ratio_eff, ratio_scale_used)

        out: Dict[str, Any] = {
            "n_vis": int(n),
            "n_slices": int(n_slices),
            "alpha_weight": float(alpha_weight),
            "baseline_trials": int(bt),

            "sw_real_sampler": sw_rs_mean,
            "sw_real_sampler_std": sw_rs_std,

            "sw_real_real_mean": sw_rr_mean,
            "sw_real_real_std": sw_rr_std,

            "sw_floor_mean": sw_floor_mean,
            "sw_floor_std": sw_floor_std,

            "denom_kind": denom_kind,
            "denom_mean": float(denom_mean),
            "denom_std": float(denom_std),

            "ratio_eff": float(ratio_eff),
            "ratio_scale_used": float(ratio_scale_used),

            **score_pack,
        }

        if compute_hist_metrics:
            beta_edges = np.linspace(self.beta_range[0], self.beta_range[1], int(hist_beta_bins) + 1)
            alpha_edges = np.linspace(self.alpha_range[0], self.alpha_range[1], int(hist_alpha_bins) + 1)
            hist = joint_hist_metrics(b_ref, a_ref, b_s, a_s, beta_edges, alpha_edges)
            out.update({
                "hist_beta_bins": int(hist_beta_bins),
                "hist_alpha_bins": int(hist_alpha_bins),
                **hist,
            })

        return out

    def evaluate_quality_conditioned(
        self,
        beta_real_deg: Any = None,
        alpha_real_deg: Any = None,
        *,
        sat_azimuth_deg: Any = None,
        sat_elevation_deg: Any = None,
        sat_belt_id: Any = None,
        n_vis_per_group: int = 20_000,
        max_groups: int = 24,
        min_group_samples: int = 500,
        group_selection: str = "largest",
        seed: int = 9876,
        n_slices: int = 64,
        alpha_weight: float = 1.0,
        baseline_trials: int = 6,
        auto_ratio_scale: bool = True,
        ratio_scale: float = 0.5,
        auto_z: float = 2.0,
        auto_target_score: float = 85.0,
        compute_hist_metrics: bool = False,
        hist_beta_bins: int = 180,
        hist_alpha_bins: int = 360,
        example_n_vis_plot: int = 50_000,
        representative_group_id: Optional[int] = None,
        include_representative_payload: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate conditional sampler quality by scoring multiple sub-samplers,
        where each sub-sampler corresponds to one group:
            group = (belt_id, skycell_id)

        Two evaluation modes:
          1) External-conditioned mode:
             provide sat_azimuth_deg/sat_elevation_deg/sat_belt_id along with
             beta_real_deg/alpha_real_deg from simulation output.
          2) Internal-pool mode:
             if no conditioning arrays are provided, evaluate directly from
             stored conditional pools built at training time.

        Representative controls:
          - representative_group_id: force a specific group id for detailed plots
          - include_representative_payload: include per-group payload for plotting
        """
        if not self._has_conditional_model():
            raise RuntimeError("Conditional quality requires a sampler built with build_conditional=True.")
        if int(n_vis_per_group) <= 0:
            raise ValueError("n_vis_per_group must be positive.")
        if int(max_groups) == 0:
            raise ValueError("max_groups must be non-zero (use a positive number or -1 for all).")
        if int(min_group_samples) <= 0:
            raise ValueError("min_group_samples must be positive.")
        if int(example_n_vis_plot) <= 0:
            raise ValueError("example_n_vis_plot must be positive.")

        rng = np.random.default_rng(int(seed))

        have_any_cond = (
            sat_azimuth_deg is not None or
            sat_elevation_deg is not None or
            sat_belt_id is not None
        )
        have_all_cond = (
            sat_azimuth_deg is not None and
            sat_elevation_deg is not None and
            sat_belt_id is not None
        )
        if have_any_cond and not have_all_cond:
            raise ValueError(
                "Conditional quality requires all three arrays: "
                "sat_azimuth_deg, sat_elevation_deg, sat_belt_id."
            )

        eval_mode = "external-conditioned" if have_all_cond else "internal-pools"

        # ------------------------------------------------------------------
        # Build per-group row index map:
        #   gids_unique[k] has rows from starts[k]:ends[k] in `order`.
        # ------------------------------------------------------------------
        if have_all_cond:
            if beta_real_deg is None or alpha_real_deg is None:
                raise ValueError("beta_real_deg and alpha_real_deg are required when conditioning arrays are provided.")

            beta = _to_degrees(beta_real_deg).reshape(-1)
            alpha = _to_degrees(alpha_real_deg).reshape(-1)
            az = _to_degrees(sat_azimuth_deg).reshape(-1)
            el = _to_degrees(sat_elevation_deg).reshape(-1)
            belt_raw = np.asarray(sat_belt_id, dtype=np.float64).reshape(-1)

            n0 = beta.size
            if alpha.size != n0 or az.size != n0 or el.size != n0 or belt_raw.size != n0:
                raise ValueError("All real and conditioning arrays must share the same total size.")

            if self.alpha_wrapped:
                alpha = _wrap_alpha_deg(alpha, self.alpha_range)

            m = np.isfinite(beta) & np.isfinite(alpha) & np.isfinite(az) & np.isfinite(el) & np.isfinite(belt_raw)
            m &= (beta >= float(self.beta_range[0])) & (beta < float(self.beta_range[1]))
            m &= (alpha >= float(self.alpha_range[0])) & (alpha < float(self.alpha_range[1]))

            belt_i = np.rint(belt_raw).astype(np.int64, copy=False)
            m &= (np.abs(belt_raw - belt_i.astype(np.float64, copy=False)) <= 1e-6)
            m &= (belt_i >= 0) & (belt_i < int(self.n_belts))

            if not np.any(m):
                raise ValueError("No valid rows remain for conditional quality evaluation.")

            beta = beta[m]
            alpha = alpha[m]
            az = az[m]
            el = el[m]
            belt_i = belt_i[m]

            sky = self._skycell_id_from_observer_angles(az, el).astype(np.int64, copy=False)
            msky = (sky >= 0) & (sky < int(self.n_skycells))
            if not np.any(msky):
                raise ValueError("No valid skycell rows remain for conditional quality evaluation.")

            beta = beta[msky]
            alpha = alpha[msky]
            az = az[msky]
            el = el[msky]
            belt_i = belt_i[msky]
            sky = sky[msky]

            group_id = belt_i * np.int64(self.n_skycells) + sky

            order = np.argsort(group_id, kind="mergesort")
            gid_s = group_id[order]
            cuts = np.flatnonzero(np.diff(gid_s)) + 1
            starts = np.concatenate(([0], cuts))
            ends = np.concatenate((cuts, [gid_s.size]))
            gids_unique = gid_s[starts].astype(np.int64, copy=False)
            counts_unique = (ends - starts).astype(np.int64, copy=False)

            n_rows_valid = int(beta.size)
            n_groups_total_nonempty = int(gids_unique.size)
        else:
            assert self.group_raw_counts is not None
            assert self.group_ptr is not None
            assert self.group_beta_pool is not None
            assert self.group_alpha_pool is not None

            gids_unique = np.nonzero(self.group_raw_counts > 0)[0].astype(np.int64, copy=False)
            if gids_unique.size == 0:
                raise ValueError("Conditional model has no non-empty groups.")
            counts_unique = self.group_raw_counts[gids_unique].astype(np.int64, copy=False)

            # Internal mode does not use `order/starts/ends` over link rows.
            order = np.empty(0, dtype=np.int64)
            starts = np.empty(0, dtype=np.int64)
            ends = np.empty(0, dtype=np.int64)

            n_rows_valid = int(counts_unique.sum())
            n_groups_total_nonempty = int(gids_unique.size)

        # Select groups for evaluation.
        cand = np.nonzero(counts_unique >= int(min_group_samples))[0]
        if cand.size == 0:
            raise ValueError(
                f"No group reaches min_group_samples={int(min_group_samples)}. "
                f"Try lowering the threshold."
            )

        if group_selection not in ("largest", "random"):
            raise ValueError("group_selection must be 'largest' or 'random'.")

        if int(max_groups) > 0 and cand.size > int(max_groups):
            if group_selection == "largest":
                ord_c = np.argsort(counts_unique[cand])[::-1]
                sel = cand[ord_c[: int(max_groups)]]
            else:
                sel = rng.choice(cand, size=int(max_groups), replace=False)
                # keep deterministic display order by descending support
                ord_c = np.argsort(counts_unique[sel])[::-1]
                sel = sel[ord_c]
        else:
            sel = cand[np.argsort(counts_unique[cand])[::-1]]

        # Representative selection policy:
        #   - explicit group id if provided
        #   - otherwise first selected (largest support under current selection policy)
        rep_gid_target: Optional[int]
        if representative_group_id is not None:
            rep_gid_target = int(representative_group_id)
            rep_pos = np.nonzero(gids_unique == rep_gid_target)[0]
            if rep_pos.size == 0:
                raise ValueError(f"representative_group_id={rep_gid_target} not found in available groups.")
            rep_idx = int(rep_pos[0])
            if rep_idx not in set(int(x) for x in sel.tolist()):
                sel = np.concatenate([sel, np.asarray([rep_idx], dtype=sel.dtype)])
        else:
            rep_gid_target = int(gids_unique[int(sel[0])]) if sel.size > 0 else None

        beta_scale = float(self.beta_range[1] - self.beta_range[0])
        directions = _make_unit_directions(3, int(n_slices), rng)
        bt = int(max(1, baseline_trials))

        beta_edges_f64 = self.beta_edges.astype(np.float64, copy=False)
        alpha_edges_f64 = self.alpha_edges.astype(np.float64, copy=False)

        group_ids_out: list[int] = []
        group_counts_out: list[int] = []
        group_sw_rs: list[float] = []
        group_sw_rs_std: list[float] = []
        group_sw_rr_mean: list[float] = []
        group_sw_rr_std: list[float] = []
        group_sw_floor_mean: list[float] = []
        group_sw_floor_std: list[float] = []
        group_denom_kind: list[str] = []
        group_denom: list[float] = []
        group_denom_std: list[float] = []
        group_ratio_eff: list[float] = []
        group_ratio_scale: list[float] = []
        group_score: list[float] = []
        group_grade: list[str] = []

        # Keep one representative group for legacy-style plots.
        example_payload: Dict[str, Any] = {}

        for pos in sel:
            gid = int(gids_unique[pos])
            cnt = int(counts_unique[pos])
            if cnt < 2:
                continue

            n_local = int(min(int(n_vis_per_group), cnt))

            if have_all_cond:
                s = int(starts[pos])
                e = int(ends[pos])
                rows = order[s:e]
                if rows.size > n_local:
                    pick = rng.choice(rows.size, size=n_local, replace=False)
                    rows = rows[pick]

                b_r = beta[rows].astype(np.float64, copy=False)
                a_r = alpha[rows].astype(np.float64, copy=False)
                az_r = az[rows].astype(np.float64, copy=False)
                el_r = el[rows].astype(np.float64, copy=False)
                belt_r = belt_i[rows].astype(np.int32, copy=False)
            else:
                # Internal mode: real reference comes from stored conditional pool.
                p0 = int(self.group_ptr[gid])  # type: ignore[index]
                p1 = int(self.group_ptr[gid + 1])  # type: ignore[index]
                if p1 <= p0:
                    continue
                pool_len = p1 - p0
                if pool_len >= n_local:
                    pick = rng.choice(pool_len, size=n_local, replace=False)
                else:
                    pick = rng.choice(pool_len, size=n_local, replace=True)

                src = (p0 + pick).astype(np.int64, copy=False)
                b_r = self.group_beta_pool[src].astype(np.float64, copy=False)   # type: ignore[index]
                a_r = self.group_alpha_pool[src].astype(np.float64, copy=False)  # type: ignore[index]

                gdesc_single = self._describe_group_ids(np.asarray([gid], dtype=np.int64))
                az_c = float(gdesc_single["az_center_deg"][0])
                el_c = float(gdesc_single["el_center_deg"][0])
                belt_c = int(gdesc_single["belt_id"][0])
                az_r = np.full(n_local, az_c, dtype=np.float64)
                el_r = np.full(n_local, el_c, dtype=np.float64)
                belt_r = np.full(n_local, belt_c, dtype=np.int32)

            b_s, a_s = self.sample_conditioned(
                rng,
                sat_azimuth_deg=az_r,
                sat_elevation_deg=el_r,
                sat_belt_id=belt_r,
                size=None,
                dtype=np.float32,
                return_context=False,
            )
            b_s = np.asarray(b_s, dtype=np.float64).reshape(-1)
            a_s = np.asarray(a_s, dtype=np.float64).reshape(-1)

            sw_rr_vals = np.empty(bt, dtype=np.float64)
            sw_floor_vals = np.empty(bt, dtype=np.float64)

            for t in range(bt):
                b1, a1, b2, a2 = _subsample_two_sets(b_r, a_r, n=b_r.size, rng=rng)
                rr = sliced_wasserstein(
                    b1, a1, b2, a2,
                    directions=directions,
                    circular_alpha=True,
                    beta_scale=beta_scale,
                    alpha_weight=float(alpha_weight),
                    rng=rng,
                )
                sw_rr_vals[t] = rr["sw_mean"]

                b1q = _bin_centers_from_edges(b1, beta_edges_f64)
                a1q = _bin_centers_from_edges(a1, alpha_edges_f64)
                fl = sliced_wasserstein(
                    b1, a1, b1q, a1q,
                    directions=directions,
                    circular_alpha=True,
                    beta_scale=beta_scale,
                    alpha_weight=float(alpha_weight),
                    rng=rng,
                )
                sw_floor_vals[t] = fl["sw_mean"]

            sw_rr_mean = float(sw_rr_vals.mean())
            sw_rr_std = float(sw_rr_vals.std(ddof=0))
            sw_floor_mean = float(sw_floor_vals.mean())
            sw_floor_std = float(sw_floor_vals.std(ddof=0))

            rs = sliced_wasserstein(
                b_r, a_r, b_s, a_s,
                directions=directions,
                circular_alpha=True,
                beta_scale=beta_scale,
                alpha_weight=float(alpha_weight),
                rng=rng,
            )
            sw_rs_mean = float(rs["sw_mean"])
            sw_rs_std = float(rs["sw_std"])

            if sw_rr_mean >= sw_floor_mean:
                denom_mean = sw_rr_mean
                denom_std = sw_rr_std
                denom_kind = "real-real"
            else:
                denom_mean = sw_floor_mean
                denom_std = sw_floor_std
                denom_kind = "quantization-floor"

            denom_mean = float(max(denom_mean, 1e-15))
            ratio_eff = float(sw_rs_mean / denom_mean)

            if auto_ratio_scale:
                ratio_scale_i = _auto_ratio_scale_from_baseline(
                    denom_mean, denom_std,
                    z=float(auto_z),
                    target_score=float(auto_target_score),
                )
            else:
                ratio_scale_i = float(ratio_scale)

            score_i = _score_from_ratio(ratio_eff, ratio_scale_i)

            group_ids_out.append(gid)
            group_counts_out.append(cnt)
            group_sw_rs.append(sw_rs_mean)
            group_sw_rs_std.append(sw_rs_std)
            group_sw_rr_mean.append(sw_rr_mean)
            group_sw_rr_std.append(sw_rr_std)
            group_sw_floor_mean.append(sw_floor_mean)
            group_sw_floor_std.append(sw_floor_std)
            group_denom_kind.append(str(denom_kind))
            group_denom.append(denom_mean)
            group_denom_std.append(float(denom_std))
            group_ratio_eff.append(ratio_eff)
            group_ratio_scale.append(float(ratio_scale_i))
            group_score.append(float(score_i["score"]))
            group_grade.append(str(score_i["grade"]))

            if include_representative_payload:
                take_rep = False
                if rep_gid_target is not None and gid == int(rep_gid_target):
                    take_rep = True
                elif not example_payload and rep_gid_target is None:
                    take_rep = True

                if take_rep:
                    n_plot = int(min(int(example_n_vis_plot), b_r.size))
                    if b_r.size > n_plot:
                        pidx = rng.choice(b_r.size, size=n_plot, replace=False)
                        b_plot = b_r[pidx]
                        a_plot = a_r[pidx]
                        bs_plot = b_s[pidx]
                        as_plot = a_s[pidx]
                    else:
                        b_plot = b_r
                        a_plot = a_r
                        bs_plot = b_s
                        as_plot = a_s

                    example_payload = {
                        "example_group_id": int(gid),
                        "example_beta_real": b_plot.astype(np.float32, copy=False),
                        "example_alpha_real": a_plot.astype(np.float32, copy=False),
                        "example_beta_sampler": bs_plot.astype(np.float32, copy=False),
                        "example_alpha_sampler": as_plot.astype(np.float32, copy=False),
                        "example_sw_real_sampler": float(sw_rs_mean),
                        "example_sw_real_sampler_std": float(sw_rs_std),
                        "example_sw_real_real_mean": float(sw_rr_mean),
                        "example_sw_real_real_std": float(sw_rr_std),
                        "example_sw_floor_mean": float(sw_floor_mean),
                        "example_sw_floor_std": float(sw_floor_std),
                        "example_denom_kind": str(denom_kind),
                        "example_denom_mean": float(denom_mean),
                        "example_denom_std": float(denom_std),
                        "example_ratio_eff": float(ratio_eff),
                        "example_ratio_scale_used": float(ratio_scale_i),
                        "example_score": float(score_i["score"]),
                        "example_grade": str(score_i["grade"]),
                    }

        if len(group_ids_out) == 0:
            raise ValueError("No eligible groups were evaluated. Try lowering thresholds.")

        gids_arr = np.asarray(group_ids_out, dtype=np.int64)
        gcounts_arr = np.asarray(group_counts_out, dtype=np.float64)
        sw_rs_arr = np.asarray(group_sw_rs, dtype=np.float64)
        denom_arr = np.asarray(group_denom, dtype=np.float64)
        ratio_arr = np.asarray(group_ratio_eff, dtype=np.float64)
        score_arr = np.asarray(group_score, dtype=np.float64)

        weights = gcounts_arr / max(gcounts_arr.sum(), 1.0)
        overall_sw_rs = float(np.sum(weights * sw_rs_arr))
        overall_denom = float(np.sum(weights * denom_arr))
        overall_denom = float(max(overall_denom, 1e-15))
        overall_ratio_eff = float(overall_sw_rs / overall_denom)

        denom_var_w = float(np.sum(weights * (denom_arr - overall_denom) ** 2))
        overall_denom_std = float(np.sqrt(max(denom_var_w, 0.0)))

        if auto_ratio_scale:
            overall_ratio_scale = _auto_ratio_scale_from_baseline(
                overall_denom, overall_denom_std,
                z=float(auto_z),
                target_score=float(auto_target_score),
            )
        else:
            overall_ratio_scale = float(ratio_scale)

        overall_score_pack = _score_from_ratio(overall_ratio_eff, overall_ratio_scale)

        coverage_rows = int(gcounts_arr.sum())
        coverage_fraction = float(coverage_rows / max(n_rows_valid, 1))

        group_desc = self._describe_group_ids(gids_arr)
        group_belt_ids = group_desc["belt_id"].astype(np.int64, copy=False)

        # Belt-level aggregates make multi-shell diagnostics easier to read and
        # can be reported directly in paper figures.
        belt_ids_eval = np.unique(group_belt_ids)
        belt_group_counts = np.empty(belt_ids_eval.size, dtype=np.int64)
        belt_support_rows = np.empty(belt_ids_eval.size, dtype=np.int64)
        belt_ratio_eff_weighted = np.empty(belt_ids_eval.size, dtype=np.float64)
        belt_score_weighted = np.empty(belt_ids_eval.size, dtype=np.float64)
        belt_sw_real_sampler_weighted = np.empty(belt_ids_eval.size, dtype=np.float64)
        belt_denom_weighted = np.empty(belt_ids_eval.size, dtype=np.float64)

        for ib, b in enumerate(belt_ids_eval):
            mb = (group_belt_ids == b)
            belt_group_counts[ib] = int(np.count_nonzero(mb))
            supp = gcounts_arr[mb]
            supp_sum = float(supp.sum())
            belt_support_rows[ib] = int(supp_sum)
            if supp_sum > 0.0:
                wb = supp / supp_sum
                belt_ratio_eff_weighted[ib] = float(np.sum(wb * ratio_arr[mb]))
                belt_score_weighted[ib] = float(np.sum(wb * score_arr[mb]))
                belt_sw_real_sampler_weighted[ib] = float(np.sum(wb * sw_rs_arr[mb]))
                belt_denom_weighted[ib] = float(np.sum(wb * denom_arr[mb]))
            else:
                belt_ratio_eff_weighted[ib] = np.nan
                belt_score_weighted[ib] = np.nan
                belt_sw_real_sampler_weighted[ib] = np.nan
                belt_denom_weighted[ib] = np.nan

        out: Dict[str, Any] = {
            "mode": "conditional",
            "conditional_eval_mode": eval_mode,
            "group_definition": "group = (belt_id, skycell_id)",
            "n_rows_valid": int(n_rows_valid),
            "n_groups_total_nonempty": int(n_groups_total_nonempty),
            "n_groups_candidate": int(cand.size),
            "n_groups_evaluated": int(gids_arr.size),
            "coverage_rows": int(coverage_rows),
            "coverage_fraction": float(coverage_fraction),
            "n_slices": int(n_slices),
            "baseline_trials": int(bt),
            "alpha_weight": float(alpha_weight),
            "n_vis_per_group": int(n_vis_per_group),
            "group_ids": gids_arr,
            "group_belt_id": group_belt_ids,
            "group_skycell_id": group_desc["skycell_id"].astype(np.int64, copy=False),
            "group_az_center_deg": group_desc["az_center_deg"].astype(np.float64, copy=False),
            "group_el_center_deg": group_desc["el_center_deg"].astype(np.float64, copy=False),
            "group_az_low_deg": group_desc["az_low_deg"].astype(np.float64, copy=False),
            "group_az_high_deg": group_desc["az_high_deg"].astype(np.float64, copy=False),
            "group_el_low_deg": group_desc["el_low_deg"].astype(np.float64, copy=False),
            "group_el_high_deg": group_desc["el_high_deg"].astype(np.float64, copy=False),
            "group_counts": gcounts_arr.astype(np.int64, copy=False),
            "group_sw_real_sampler": np.asarray(group_sw_rs, dtype=np.float64),
            "group_sw_real_sampler_std": np.asarray(group_sw_rs_std, dtype=np.float64),
            "group_sw_real_real_mean": np.asarray(group_sw_rr_mean, dtype=np.float64),
            "group_sw_real_real_std": np.asarray(group_sw_rr_std, dtype=np.float64),
            "group_sw_floor_mean": np.asarray(group_sw_floor_mean, dtype=np.float64),
            "group_sw_floor_std": np.asarray(group_sw_floor_std, dtype=np.float64),
            "group_denom_kind": np.asarray(group_denom_kind, dtype="<U32"),
            "group_denom_mean": np.asarray(group_denom, dtype=np.float64),
            "group_denom_std": np.asarray(group_denom_std, dtype=np.float64),
            "group_ratio_eff": ratio_arr,
            "group_ratio_scale_used": np.asarray(group_ratio_scale, dtype=np.float64),
            "group_score": score_arr,
            "group_grade": np.asarray(group_grade, dtype="<U1"),
            "belt_ids_evaluated": belt_ids_eval.astype(np.int64, copy=False),
            "belt_group_counts": belt_group_counts,
            "belt_support_rows": belt_support_rows,
            "belt_ratio_eff_weighted": belt_ratio_eff_weighted,
            "belt_score_weighted": belt_score_weighted,
            "belt_sw_real_sampler_weighted": belt_sw_real_sampler_weighted,
            "belt_denom_weighted": belt_denom_weighted,
            "overall_sw_real_sampler": float(overall_sw_rs),
            "overall_denom_mean": float(overall_denom),
            "overall_denom_std": float(overall_denom_std),
            "overall_ratio_eff": float(overall_ratio_eff),
            "overall_ratio_scale_used": float(overall_ratio_scale),
            "overall_group_score_weighted": float(np.sum(weights * score_arr)),
            **{f"overall_{k}": v for k, v in overall_score_pack.items()},
        }

        if example_payload:
            ex_gid = int(example_payload["example_group_id"])
            ex_desc = self._describe_group_ids(np.asarray([ex_gid], dtype=np.int64))
            example_payload.update(
                example_belt_id=int(ex_desc["belt_id"][0]),
                example_skycell_id=int(ex_desc["skycell_id"][0]),
                example_az_center_deg=float(ex_desc["az_center_deg"][0]),
                example_el_center_deg=float(ex_desc["el_center_deg"][0]),
                example_az_low_deg=float(ex_desc["az_low_deg"][0]),
                example_az_high_deg=float(ex_desc["az_high_deg"][0]),
                example_el_low_deg=float(ex_desc["el_low_deg"][0]),
                example_el_high_deg=float(ex_desc["el_high_deg"][0]),
            )
            out.update(example_payload)

        if compute_hist_metrics and example_payload:
            b_ref = np.asarray(example_payload["example_beta_real"], dtype=np.float64)
            a_ref = np.asarray(example_payload["example_alpha_real"], dtype=np.float64)
            b_emp = np.asarray(example_payload["example_beta_sampler"], dtype=np.float64)
            a_emp = np.asarray(example_payload["example_alpha_sampler"], dtype=np.float64)
            beta_edges = np.linspace(self.beta_range[0], self.beta_range[1], int(hist_beta_bins) + 1)
            alpha_edges = np.linspace(self.alpha_range[0], self.alpha_range[1], int(hist_alpha_bins) + 1)
            hist = joint_hist_metrics(b_ref, a_ref, b_emp, a_emp, beta_edges, alpha_edges)
            out.update({
                "hist_beta_bins": int(hist_beta_bins),
                "hist_alpha_bins": int(hist_alpha_bins),
                **hist,
            })

        return out

    # --------------------------
    # Visualization
    # --------------------------

    def show_comparison(
        self,
        beta_real_deg: Any,
        alpha_real_deg: Any,
        *,
        n_vis: int = 200_000,
        seed: int = 9876,
        n_slices: int = 64,
        alpha_weight: float = 1.0,
        baseline_trials: int = 10,
        auto_ratio_scale: bool = True,
        ratio_scale: float = 0.5,
        save_prefix: Optional[str] = None,
        metrics_path: Optional[str] = None,
        dropped: int = 0,
        plot_beta_bins: int = 250,
        plot_alpha_bins: int = 360,
        sat_azimuth_deg: Any = None,
        sat_elevation_deg: Any = None,
        sat_belt_id: Any = None,
        conditional_min_group_samples: int = 500,
        conditional_max_groups: int = 24,
        conditional_group_selection: str = "largest",
        conditional_n_vis_per_group: int = 20_000,
        conditional_mode: str = "auto",
        representative_group_id: Optional[int] = None,
        show_representative: bool = True,
        show_group_map: bool = True,
        backend: str = "matplotlib",
        interactive_plotly: bool = False,
        plotly_click_select: bool = True,
    ) -> Dict[str, Any]:
        """
        Draw:
          - 1D beta comparison
          - 1D alpha comparison
          - 3-panel: real polar, sampler polar, quality panel

        Conditional behavior is controlled by `conditional_mode`:
          - "auto"        : use conditional diagnostics when possible
          - "conditional" : force conditional diagnostics
          - "legacy"      : force classic single-sampler diagnostics

        Rendering behavior is controlled by `backend`:
          - "matplotlib" : static publication-style figures
          - "plotly"     : interactive notebook figures
          - "both"       : render both families

        In conditional diagnostics, this method:
          - evaluate multiple (belt, skycell) sub-samplers independently
          - show aggregate quality + per-group quality spread
          - show sky linkage map (group centers in observer az/el sky)
          - optionally render interactive Plotly versions (for notebooks)
          - optional click-to-select helper for representative_group_id
          - optionally render legacy-style plots for a representative group

        Returns metrics dict from evaluate_quality() or evaluate_quality_conditioned().
        If ``metrics_path`` is provided, metrics are also written to ``.npz``.
        """
        backend_mode = str(backend).strip().lower()
        if backend_mode not in ("matplotlib", "plotly", "both"):
            raise ValueError("backend must be one of: 'matplotlib', 'plotly', 'both'.")

        render_matplotlib = backend_mode in ("matplotlib", "both")
        render_plotly = backend_mode in ("plotly", "both")
        plotly_show = bool(interactive_plotly or backend_mode == "plotly")

        if render_matplotlib and plt is None:
            raise RuntimeError("matplotlib is required for show_comparison(backend='matplotlib').")
        if render_plotly and (go is None or make_subplots is None):
            raise RuntimeError("plotly is required for show_comparison(backend='plotly').")

        have_any_cond = (
            sat_azimuth_deg is not None or
            sat_elevation_deg is not None or
            sat_belt_id is not None
        )
        have_all_cond = (
            sat_azimuth_deg is not None and
            sat_elevation_deg is not None and
            sat_belt_id is not None
        )
        if have_any_cond and not have_all_cond:
            raise ValueError(
                "Conditional comparison requires all three arrays: "
                "sat_azimuth_deg, sat_elevation_deg, sat_belt_id."
            )

        mode = str(conditional_mode).lower().strip()
        if mode not in ("auto", "conditional", "legacy"):
            raise ValueError("conditional_mode must be one of: 'auto', 'conditional', 'legacy'.")

        if mode == "legacy":
            use_conditional = False
        elif mode == "conditional":
            use_conditional = True
        else:
            use_conditional = self._has_conditional_model() and (have_all_cond or (not have_any_cond))

        # ------------------------------------------------------------------
        # Conditional comparison path (new multi-sub-sampler diagnostics)
        # ------------------------------------------------------------------
        if use_conditional:
            if not self._has_conditional_model():
                raise RuntimeError(
                    "conditional_mode requests conditional diagnostics, but this sampler has no conditional model."
                )

            metrics = self.evaluate_quality_conditioned(
                beta_real_deg,
                alpha_real_deg,
                sat_azimuth_deg=(sat_azimuth_deg if have_all_cond else None),
                sat_elevation_deg=(sat_elevation_deg if have_all_cond else None),
                sat_belt_id=(sat_belt_id if have_all_cond else None),
                n_vis_per_group=int(conditional_n_vis_per_group),
                max_groups=int(conditional_max_groups),
                min_group_samples=int(conditional_min_group_samples),
                group_selection=str(conditional_group_selection),
                seed=int(seed),
                n_slices=int(n_slices),
                alpha_weight=float(alpha_weight),
                baseline_trials=int(max(1, baseline_trials)),
                auto_ratio_scale=bool(auto_ratio_scale),
                ratio_scale=float(ratio_scale),
                compute_hist_metrics=True,
                representative_group_id=representative_group_id,
                include_representative_payload=bool(show_representative),
            )

            # ------------------------------------------------------------------
            # Summary figure: publication-style aggregate + group spread.
            # ------------------------------------------------------------------
            gids = np.asarray(metrics["group_ids"], dtype=np.int64)
            gcounts = np.asarray(metrics["group_counts"], dtype=np.float64)
            gr = np.asarray(metrics["group_ratio_eff"], dtype=np.float64)
            gs = np.asarray(metrics["group_score"], dtype=np.float64)
            gbelt = np.asarray(metrics["group_belt_id"], dtype=np.int64)
            support_k = gcounts / 1e3  # common axis multiplier for readability
            unique_belts = np.unique(gbelt)

            # Muted belt palette to keep figures publication-friendly.
            belt_palette = np.asarray(
                [
                    "#4E5968", "#627285", "#7A8896", "#506B78",
                    "#6A7F8F", "#88939D", "#56606E", "#738292",
                ],
                dtype=object,
            )
            belt_markers_mpl = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "p")
            belt_symbols_plotly = (
                "circle", "square", "triangle-up", "diamond", "cross", "x",
                "triangle-down", "triangle-left", "triangle-right", "hexagon",
                "pentagon", "star",
            )
            rank = np.arange(gids.size) + 1
            score_low = "#6B0F1A"   # dark red: poor quality
            score_mid = "#2B6CB0"   # blue: mid quality
            score_high = "#0F9D58"  # emerald: strong quality
            # High-contrast threshold styling for paper readability.
            thr_a_color = "#111111"
            thr_b_color = "#3A3A3A"
            thr_c_color = "#707070"
            thr_width = 2.4
            score_colorscale_plotly = [
                [0.0, score_low],
                [0.5, score_mid],
                [1.0, score_high],
            ]
            if LinearSegmentedColormap is not None:
                score_cmap_mpl = LinearSegmentedColormap.from_list(
                    "score_rbe",
                    [score_low, score_mid, score_high],
                )
            else:
                score_cmap_mpl = plt.cm.RdYlGn

            if render_matplotlib:
                fig = plt.figure(figsize=(20, 7.5))
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)

                fs_title = 14
                fs_axis = 12
                fs_tick = 11
                fs_text = 10.8

                score_colors = score_cmap_mpl(np.clip(gs / 100.0, 0.0, 1.0))
                ax1.bar(rank, gs, color=score_colors, alpha=0.96, width=0.92)
                ax1.axhline(90.0, color=thr_a_color, linestyle="--", linewidth=thr_width, alpha=0.95, label="A-threshold")
                ax1.axhline(70.0, color=thr_b_color, linestyle="--", linewidth=thr_width, alpha=0.95, label="B-threshold")
                ax1.axhline(40.0, color=thr_c_color, linestyle="--", linewidth=thr_width, alpha=0.95, label="C-threshold")
                ax1.set_xlabel("Evaluated Group Rank (sorted by support)", fontsize=fs_axis)
                ax1.set_ylabel("Group Quality Score [0..100]", fontsize=fs_axis)
                ax1.set_title("Per-Group Conditional Quality", fontsize=fs_title)
                ax1.set_ylim(0.0, 100.0)
                ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax1.tick_params(axis="both", labelsize=fs_tick)
                ax1.legend(loc="lower right", fontsize=9.5, frameon=False)

                for bi, b in enumerate(unique_belts):
                    mb = (gbelt == b)
                    ax2.scatter(
                        support_k[mb],
                        gr[mb],
                        s=36,
                        alpha=0.90,
                        color=belt_palette[bi % belt_palette.size],
                        label=f"Belt {int(b)}",
                    )
                ax2.set_xlabel("Group Support [x10^3 Valid Links]", fontsize=fs_axis)
                ax2.set_ylabel("Effective Ratio (Lower is Better)", fontsize=fs_axis)
                ax2.set_title("Support vs Effective Ratio by Belt", fontsize=fs_title)
                ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax2.tick_params(axis="both", labelsize=fs_tick)
                if unique_belts.size <= 10:
                    ax2.legend(loc="best", fontsize=10, frameon=False)

            belt_ids_eval = np.asarray(metrics.get("belt_ids_evaluated", []), dtype=np.int64)
            belt_group_counts = np.asarray(metrics.get("belt_group_counts", []), dtype=np.int64)
            belt_support_rows = np.asarray(metrics.get("belt_support_rows", []), dtype=np.int64)
            belt_ratio_w = np.asarray(metrics.get("belt_ratio_eff_weighted", []), dtype=np.float64)
            belt_score_w = np.asarray(metrics.get("belt_score_weighted", []), dtype=np.float64)
            belt_lines = ["Per-belt weighted summary:"]
            if belt_ids_eval.size:
                for b, ng, ns, rr, ss in zip(
                    belt_ids_eval,
                    belt_group_counts,
                    belt_support_rows,
                    belt_ratio_w,
                    belt_score_w,
                ):
                    belt_lines.append(
                        f"  Belt {int(b):>2}: groups={int(ng):>3}, links={int(ns):>8}, "
                        f"ratio_eff={float(rr):>5.2f}, score={float(ss):>6.2f}"
                    )
            else:
                belt_lines.append("  (not available)")
            belt_text_block = "\n".join(belt_lines[:12])

            if render_matplotlib:
                ax3.axis("off")
                txt = (
                    "Conditional Quality Summary\n"
                    "===========================\n"
                    f"Evaluation mode            : {metrics.get('conditional_eval_mode', '-')}\n"
                    f"Group definition           : one (belt_id, skycell_id) pair\n"
                    f"Groups evaluated           : {metrics['n_groups_evaluated']:,}\n"
                    f"Candidate groups           : {metrics['n_groups_candidate']:,}\n"
                    f"Coverage                   : {metrics['coverage_rows']:,} / {metrics['n_rows_valid']:,}\n"
                    f"Coverage fraction          : {metrics['coverage_fraction']:.3f}\n"
                    "\n"
                    "Aggregate metrics (count-weighted)\n"
                    f"SW(real,sampler)          : {metrics['overall_sw_real_sampler']:.4e}\n"
                    f"Denominator               : {metrics['overall_denom_mean']:.4e} ± {metrics['overall_denom_std']:.2e}\n"
                    f"ratio_eff                 : {metrics['overall_ratio_eff']:.3f}\n"
                    f"ratio_scale               : {metrics['overall_ratio_scale_used']:.3f}\n"
                    f"Overall score             : {metrics['overall_score']:.2f} / 100\n"
                    f"Overall grade             : {metrics['overall_grade']}\n"
                    f"Weighted group score      : {metrics['overall_group_score_weighted']:.2f}\n"
                    "\n"
                    "Interpretation\n"
                    "ratio_eff = SW(real,sampler) / max(SW(real,real), SW(floor))\n"
                    "SW(real,real): finite-sample noise baseline\n"
                    "SW(floor)    : discretization floor of histogram model\n"
                    "\n"
                    f"Samples per evaluated group: {metrics['n_vis_per_group']:,}\n"
                    f"Sliced-Wasserstein directions: {metrics['n_slices']}\n"
                    f"Baseline trial count         : {metrics['baseline_trials']}\n"
                    "\n"
                    f"{belt_text_block}\n"
                )
                ax3.text(
                    0.0,
                    0.5,
                    txt,
                    transform=ax3.transAxes,
                    va="center",
                    family="monospace",
                    fontsize=fs_text,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D0D0D0", alpha=0.92),
                )

                fig.suptitle("Conditional Sampler Quality by Belt and Skycell", y=1.01, fontsize=15.5)
                fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
                if save_prefix:
                    fig.savefig(f"{save_prefix}_conditional_quality.png", dpi=200)

            # ------------------------------------------------------------------
            # Optional sky-linkage map:
            #   - each evaluated group is plotted at its skycell center (az/el)
            #   - marker shape identifies belt
            #   - marker size encodes support
            #   - marker color encodes group quality score
            # This gives a direct visual link between group ID and satellite
            # position relative to the RAS site.
            # ------------------------------------------------------------------
            if render_matplotlib and show_group_map:
                gaz = np.asarray(metrics["group_az_center_deg"], dtype=np.float64)
                gel = np.asarray(metrics["group_el_center_deg"], dtype=np.float64)
                sizes = np.clip(
                    16.0 + 14.0 * np.log10(np.maximum(gcounts, 1.0)),
                    16.0,
                    120.0,
                )

                # Keep the same fixed score scale used across all quality plots.
                vmin = 0.0
                vmax = 100.0

                fig_map = plt.figure(figsize=(12, 6))
                axm = fig_map.add_subplot(1, 1, 1)

                cb_handle = None
                for bi, b in enumerate(unique_belts):
                    mb = (gbelt == b)
                    if not np.any(mb):
                        continue
                    sc = axm.scatter(
                        gaz[mb],
                        gel[mb],
                        c=gs[mb],
                        cmap=score_cmap_mpl,
                        vmin=vmin,
                        vmax=vmax,
                        s=sizes[mb],
                        marker=belt_markers_mpl[bi % len(belt_markers_mpl)],
                        alpha=0.90,
                        edgecolors="black",
                        linewidths=0.25,
                    )
                    if cb_handle is None:
                        cb_handle = sc

                axm.set_xlim(0.0, 360.0)
                axm.set_ylim(0.0, 90.0)
                axm.set_xlabel("Satellite Azimuth at RAS Site [deg]")
                axm.set_ylabel("Satellite Elevation at RAS Site [deg]")
                axm.set_title("Evaluated Group Linkage in Observer Sky")
                axm.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

                if cb_handle is not None:
                    cbar = fig_map.colorbar(cb_handle, ax=axm, pad=0.015)
                    cbar.set_label("Group Quality Score [0..100]")

                if unique_belts.size <= 12 and Line2D is not None:
                    belt_handles = []
                    for bi, b in enumerate(unique_belts):
                        belt_handles.append(
                            Line2D(
                                [0], [0],
                                marker=belt_markers_mpl[bi % len(belt_markers_mpl)],
                                linestyle="None",
                                markerfacecolor="#F5F5F5",
                                markeredgecolor="#2F3B46",
                                markersize=8,
                                label=f"Belt {int(b)}",
                            )
                        )
                    axm.legend(
                        handles=belt_handles,
                        loc="upper left",
                        bbox_to_anchor=(0.01, 0.99),
                        ncol=min(2, len(belt_handles)),
                        fontsize=9.5,
                        frameon=True,
                        fancybox=True,
                        title="Belt ID (marker shape)",
                    )

                axm.text(
                    0.01,
                    0.02,
                    (
                        "Each point = one evaluated group: (belt_id, skycell_id)\n"
                        "Color = quality score, marker = belt_id, size = valid-link support"
                    ),
                    transform=axm.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9.0,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", alpha=0.9),
                )

                fig_map.tight_layout(rect=[0.0, 0.06, 0.89, 1.0])
                if save_prefix:
                    fig_map.savefig(f"{save_prefix}_conditional_group_sky_map.png", dpi=200)

            # ------------------------------------------------------------------
            # Optional Plotly figures (interactive exploration in notebooks):
            #   1) quality dashboard (rank + support-ratio + summary)
            #   2) sky linkage scatter with hover-rich group metadata
            #
            # If FigureWidget callbacks are available, clicking a point prints a
            # concrete representative_group_id to reuse in a second run:
            #   sampler.show_comparison(..., representative_group_id=<id>)
            # ------------------------------------------------------------------
            if render_plotly:
                gaz = np.asarray(metrics["group_az_center_deg"], dtype=np.float64)
                gel = np.asarray(metrics["group_el_center_deg"], dtype=np.float64)
                gsky = np.asarray(metrics["group_skycell_id"], dtype=np.int64)
                gsize = np.clip(8.0 + 7.0 * np.log10(np.maximum(gcounts, 1.0)), 8.0, 28.0)

                belt_html_lines = []
                for b, ng, ns, rr, ss in zip(
                    belt_ids_eval,
                    belt_group_counts,
                    belt_support_rows,
                    belt_ratio_w,
                    belt_score_w,
                ):
                    belt_html_lines.append(
                        f"Belt {int(b)}: groups={int(ng)}, links={int(ns):,}, "
                        f"ratio_eff={float(rr):.2f}, score={float(ss):.2f}"
                    )
                belt_html = "<br>".join(belt_html_lines[:10]) if belt_html_lines else "(not available)"

                # Dashboard figure.
                fig_q = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=(
                        "Per-Group Score (Sorted by Support)",
                        "Support vs Effective Ratio by Belt",
                    ),
                    horizontal_spacing=0.10,
                )
                fig_q.add_trace(
                    go.Bar(
                        x=rank,
                        y=gs,
                        marker=dict(
                            color=gs,
                            colorscale=score_colorscale_plotly,
                            cmin=0.0,
                            cmax=100.0,
                        ),
                        hovertemplate=(
                            "Rank=%{x}<br>"
                            "Score=%{y:.2f}<br>"
                            "Group=%{customdata[0]}<br>"
                            "Belt=%{customdata[1]}<br>"
                            "Skycell=%{customdata[2]}<br>"
                            "Support=%{customdata[3]:,}<extra></extra>"
                        ),
                        customdata=np.stack([gids, gbelt, gsky, gcounts.astype(np.int64)], axis=1),
                        name="Group score",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig_q.add_hline(y=90.0, line_dash="longdash", line_color=thr_a_color, line_width=thr_width, row=1, col=1)
                fig_q.add_hline(y=70.0, line_dash="longdash", line_color=thr_b_color, line_width=thr_width, row=1, col=1)
                fig_q.add_hline(y=40.0, line_dash="longdash", line_color=thr_c_color, line_width=thr_width, row=1, col=1)

                for bi, b in enumerate(unique_belts):
                    mb = (gbelt == b)
                    fig_q.add_trace(
                        go.Scatter(
                            x=support_k[mb],
                            y=gr[mb],
                            mode="markers",
                            name=f"Belt {int(b)}",
                            marker=dict(
                                size=9,
                                opacity=0.88,
                                color=str(belt_palette[bi % belt_palette.size]),
                                symbol=belt_symbols_plotly[bi % len(belt_symbols_plotly)],
                                line=dict(width=0.5, color="#2F3B46"),
                            ),
                            customdata=np.stack(
                                [
                                    gids[mb],
                                    np.full(np.count_nonzero(mb), int(b), dtype=np.int64),
                                    gcounts[mb].astype(np.int64),
                                    gs[mb],
                                ],
                                axis=1,
                            ),
                            hovertemplate=(
                                "Group=%{customdata[0]}<br>"
                                "Belt=%{customdata[1]}<br>"
                                "Support=%{customdata[2]:,}<br>"
                                "Score=%{customdata[3]:.2f}<br>"
                                "ratio_eff=%{y:.3f}<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=2,
                    )

                fig_q.update_xaxes(title_text="Evaluated Group Rank", row=1, col=1)
                fig_q.update_yaxes(title_text="Group Quality Score [0..100]", range=[0.0, 100.0], row=1, col=1)
                fig_q.update_xaxes(title_text="Group Support [x10^3 Valid Links]", row=1, col=2)
                fig_q.update_yaxes(title_text="Effective Ratio (Lower is Better)", row=1, col=2)
                fig_q.update_layout(
                    width=1580,
                    height=560,
                    template="plotly_white",
                    legend=dict(
                        x=0.965,
                        y=0.965,
                        xanchor="right",
                        yanchor="top",
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="rgba(0,0,0,0.20)",
                        borderwidth=1,
                        font=dict(size=12),
                    ),
                    margin=dict(l=70, r=560, t=85, b=75),
                    title=dict(
                        text="Conditional Sampler Quality by Belt and Skycell (Interactive)",
                        x=0.01,
                        xanchor="left",
                        font=dict(size=21),
                    ),
                )
                fig_q.add_annotation(
                    x=0.02,
                    y=0.03,
                    xref="x domain",
                    yref="y domain",
                    row=1,
                    col=1,
                    xanchor="left",
                    yanchor="bottom",
                    showarrow=False,
                    align="left",
                    font=dict(size=11.5, family="Courier New"),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="rgba(0,0,0,0.20)",
                    borderwidth=1,
                    text=(
                        f"<span style='color:{thr_a_color}'><b>A</b></span> threshold: 90<br>"
                        f"<span style='color:{thr_b_color}'><b>B</b></span> threshold: 70<br>"
                        f"<span style='color:{thr_c_color}'><b>C</b></span> threshold: 40"
                    ),
                )
                fig_q.add_annotation(
                    x=1.02,
                    y=0.35,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    showarrow=False,
                    align="left",
                    font=dict(size=12.5, family="Courier New"),
                    bgcolor="rgba(255,255,255,0.90)",
                    bordercolor="rgba(0,0,0,0.20)",
                    borderwidth=1,
                    text=(
                        "<b>Conditional Quality Summary</b><br>"
                        f"Evaluation mode: {metrics.get('conditional_eval_mode', '-')}<br>"
                        f"Group definition: (belt_id, skycell_id)<br>"
                        f"Groups evaluated: {metrics['n_groups_evaluated']:,}<br>"
                        f"Candidate groups: {metrics['n_groups_candidate']:,}<br>"
                        f"Coverage: {metrics['coverage_rows']:,} / {metrics['n_rows_valid']:,} ({metrics['coverage_fraction']:.3f})<br>"
                        f"SW(real,sampler): {metrics['overall_sw_real_sampler']:.4e}<br>"
                        f"Denominator: {metrics['overall_denom_mean']:.4e} ± {metrics['overall_denom_std']:.2e}<br>"
                        f"ratio_eff: {metrics['overall_ratio_eff']:.3f}<br>"
                        f"ratio_scale: {metrics['overall_ratio_scale_used']:.3f}<br>"
                        f"Overall score: {metrics['overall_score']:.2f} / 100<br>"
                        f"Overall grade: {metrics['overall_grade']}<br>"
                        f"Weighted group score: {metrics['overall_group_score_weighted']:.2f}<br>"
                        "<br><b>Interpretation</b><br>"
                        "ratio_eff = SW(real,sampler) / max(SW(real,real), SW(floor))<br>"
                        "SW(real,real): finite-sample baseline<br>"
                        "SW(floor): histogram discretization floor<br>"
                        f"Samples per evaluated group: {metrics['n_vis_per_group']:,}<br>"
                        f"Sliced-Wasserstein directions: {metrics['n_slices']}<br>"
                        f"Baseline trial count: {metrics['baseline_trials']}<br>"
                        "<br><b>Per-belt weighted summary</b><br>"
                        f"{belt_html}"
                    ),
                )

                # Sky linkage figure (group metadata on hover).
                fig_sky = go.Figure()
                for bi, b in enumerate(unique_belts):
                    mb = (gbelt == b)
                    fig_sky.add_trace(
                        go.Scatter(
                            x=gaz[mb],
                            y=gel[mb],
                            mode="markers",
                            name=f"Belt {int(b)}",
                            marker=dict(
                                size=gsize[mb],
                                symbol=belt_symbols_plotly[bi % len(belt_symbols_plotly)],
                                color=gs[mb],
                                coloraxis="coloraxis",
                                opacity=0.92,
                                line=dict(width=0.5, color="rgba(0,0,0,0.55)"),
                            ),
                            customdata=np.stack(
                                [
                                    gids[mb],
                                    gsky[mb],
                                    gcounts[mb].astype(np.int64),
                                    gr[mb],
                                    gs[mb],
                                ],
                                axis=1,
                            ),
                            hovertemplate=(
                                "Group=%{customdata[0]}<br>"
                                "Belt=" + str(int(b)) + "<br>"
                                "Skycell=%{customdata[1]}<br>"
                                "Az=%{x:.2f} deg<br>"
                                "El=%{y:.2f} deg<br>"
                                "Support=%{customdata[2]:,}<br>"
                                "ratio_eff=%{customdata[3]:.3f}<br>"
                                "Score=%{customdata[4]:.2f}<extra></extra>"
                            ),
                        )
                    )

                fig_sky.update_layout(
                    width=1420,
                    height=640,
                    template="plotly_white",
                    coloraxis=dict(
                        colorscale=score_colorscale_plotly,
                        cmin=0.0,
                        cmax=100.0,
                        colorbar=dict(title="Group Score [0..100]", x=1.02, len=0.82),
                    ),
                    xaxis=dict(title="Satellite Azimuth at RAS Site [deg]", range=[0.0, 360.0]),
                    yaxis=dict(title="Satellite Elevation at RAS Site [deg]", range=[0.0, 90.0]),
                    legend=dict(
                        x=1.19,
                        y=1.0,
                        xanchor="left",
                        yanchor="top",
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="rgba(0,0,0,0.20)",
                        borderwidth=1,
                        font=dict(size=12),
                    ),
                    margin=dict(l=80, r=560, t=85, b=75),
                    title=dict(
                        text="Evaluated Group Linkage in Observer Sky (Interactive)",
                        x=0.01,
                        xanchor="left",
                        font=dict(size=21),
                    ),
                )
                fig_sky.add_annotation(
                    x=1.19,
                    y=0.22,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    showarrow=False,
                    align="left",
                    font=dict(size=12, family="Courier New"),
                    bgcolor="rgba(255,255,255,0.90)",
                    bordercolor="rgba(0,0,0,0.20)",
                    borderwidth=1,
                    text=(
                        "Point semantics:<br>"
                        "  color  = group score<br>"
                        "  size   = valid-link support<br>"
                        "  symbol = belt id"
                    ),
                )

                if plotly_click_select and plotly_show:
                    try:
                        fw = go.FigureWidget(fig_sky)

                        def _make_click_handler(trace_belt: int):
                            def _handler(trace, points, state):  # pragma: no cover - interactive path
                                if not points.point_inds:
                                    return
                                pi = int(points.point_inds[0])
                                gid_sel = int(trace.customdata[pi][0])
                                print(
                                    "[PICK] representative_group_id="
                                    f"{gid_sel} (belt={trace_belt}). "
                                    "Re-run show_comparison(..., representative_group_id="
                                    f"{gid_sel})"
                                )

                            return _handler

                        for tr in fw.data:
                            belt_lbl = str(getattr(tr, "name", "Belt ?"))
                            try:
                                belt_int = int(belt_lbl.split()[-1])
                            except Exception:
                                belt_int = -1
                            tr.on_click(_make_click_handler(belt_int))

                        if display is not None:
                            display(fig_q)
                            display(fw)
                        else:
                            fig_q.show()
                            fw.show()
                    except Exception:
                        print(
                            "[INFO] Plotly FigureWidget callbacks are unavailable in this environment; "
                            "hover to inspect group_id and pass it as representative_group_id manually."
                        )
                        fig_q.show()
                        fig_sky.show()
                elif plotly_show:
                    fig_q.show()
                    fig_sky.show()

                if save_prefix:
                    fig_q.write_html(f"{save_prefix}_conditional_quality_interactive.html")
                    fig_sky.write_html(f"{save_prefix}_conditional_group_sky_map_interactive.html")

            # ------------------------------------------------------------------
            # Representative group: legacy-style, fully annotated comparison.
            # ------------------------------------------------------------------
            if render_matplotlib and show_representative and (
                "example_beta_real" in metrics and "example_alpha_real" in metrics and
                "example_beta_sampler" in metrics and "example_alpha_sampler" in metrics
            ):
                beta_r = np.asarray(metrics["example_beta_real"], dtype=np.float64)
                alpha_r = np.asarray(metrics["example_alpha_real"], dtype=np.float64)
                beta_s = np.asarray(metrics["example_beta_sampler"], dtype=np.float64)
                alpha_s = np.asarray(metrics["example_alpha_sampler"], dtype=np.float64)
                rep_axis_fs = 13
                rep_title_fs = 15
                rep_text_fs = 11.5

                beta_edges_plot = np.linspace(self.beta_range[0], self.beta_range[1], int(plot_beta_bins) + 1)
                c_ref, _ = np.histogram(beta_r, bins=beta_edges_plot)
                c_sam, _ = np.histogram(beta_s, bins=beta_edges_plot)
                w = np.diff(beta_edges_plot)
                ref_d = c_ref / (c_ref.sum() * w)
                sam_d = c_sam / (c_sam.sum() * w)

                fig1 = plt.figure(figsize=(12, 5))
                ax = fig1.add_subplot(1, 1, 1)
                ax.bar(beta_edges_plot[:-1], ref_d, width=w, align="edge", alpha=0.35, color="#4C78A8", label="Real β")
                ax.step(beta_edges_plot[:-1], sam_d, where="post", linewidth=2.4, color="#F58518", label="Sampler β")
                ax.set_xlabel(r"$\beta$ [deg]", fontsize=rep_axis_fs)
                ax.set_ylabel("Probability Density", fontsize=rep_axis_fs)
                ax.set_title("β Distribution: Real vs Sampler", fontsize=rep_title_fs)
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax.tick_params(axis="both", labelsize=11)
                ax.legend(
                    loc="upper right",
                    fontsize=10.5,
                    frameon=True,
                    fancybox=True,
                    title="Legend",
                    title_fontsize=10.5,
                )
                fig1.tight_layout()
                if save_prefix:
                    fig1.savefig(f"{save_prefix}_conditional_example_beta_1d.png", dpi=200)

                alpha_edges_plot = np.linspace(self.alpha_range[0], self.alpha_range[1], int(plot_alpha_bins) + 1)
                c_ref, _ = np.histogram(alpha_r, bins=alpha_edges_plot)
                c_sam, _ = np.histogram(alpha_s, bins=alpha_edges_plot)
                w = np.diff(alpha_edges_plot)
                ref_d = c_ref / (c_ref.sum() * w)
                sam_d = c_sam / (c_sam.sum() * w)

                fig2 = plt.figure(figsize=(12, 5))
                ax = fig2.add_subplot(1, 1, 1)
                ax.bar(alpha_edges_plot[:-1], ref_d, width=w, align="edge", alpha=0.35, color="#4C78A8", label="Real α")
                ax.step(alpha_edges_plot[:-1], sam_d, where="post", linewidth=2.4, color="#F58518", label="Sampler α")
                ax.set_xlabel(r"$\alpha$ [deg]", fontsize=rep_axis_fs)
                ax.set_ylabel("Probability Density", fontsize=rep_axis_fs)
                ax.set_title("α Distribution: Real vs Sampler", fontsize=rep_title_fs)
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax.tick_params(axis="both", labelsize=11)
                ax.legend(
                    loc="upper right",
                    fontsize=10.5,
                    frameon=True,
                    fancybox=True,
                    title="Legend",
                    title_fontsize=10.5,
                )
                fig2.tight_layout()
                if save_prefix:
                    fig2.savefig(f"{save_prefix}_conditional_example_alpha_1d.png", dpi=200)

                phi_r = np.deg2rad(alpha_r)
                r_r = beta_r
                phi_s = np.deg2rad(alpha_s)
                r_s = beta_s

                fig3 = plt.figure(figsize=(20, 7))
                ax1 = fig3.add_subplot(1, 3, 1, projection="polar")
                ax2 = fig3.add_subplot(1, 3, 2, projection="polar")
                ax3 = fig3.add_subplot(1, 3, 3)

                ax1.scatter(phi_r, r_r, s=1, alpha=0.12, color="#4C78A8")
                ax1.set_title("Real (α, β)", pad=18, fontsize=rep_title_fs)
                ax2.scatter(phi_s, r_s, s=1, alpha=0.12, color="#F58518")
                ax2.set_title("Sampler (α, β)", pad=18, fontsize=rep_title_fs)

                r_max = float(max(np.max(r_r), np.max(r_s))) if r_r.size and r_s.size else 1.0
                ax1.set_rlim(0, r_max)
                ax2.set_rlim(0, r_max)

                ax3.axis("off")
                ex_hist_beta_bins = metrics.get("hist_beta_bins", "-")
                ex_hist_alpha_bins = metrics.get("hist_alpha_bins", "-")
                ax3.text(
                    0.0,
                    0.5,
                    (
                        "Group Details\n"
                        "=============\n"
                        f"group_id                 : {metrics.get('example_group_id', -1)}\n"
                        f"belt_id                  : {metrics.get('example_belt_id', -1)}\n"
                        f"skycell_id               : {metrics.get('example_skycell_id', -1)}\n"
                        f"skycell az bounds [deg]  : [{metrics.get('example_az_low_deg', float('nan')):.1f}, {metrics.get('example_az_high_deg', float('nan')):.1f}]\n"
                        f"skycell el bounds [deg]  : [{metrics.get('example_el_low_deg', float('nan')):.1f}, {metrics.get('example_el_high_deg', float('nan')):.1f}]\n"
                        f"skycell center [deg]     : az={metrics.get('example_az_center_deg', float('nan')):.1f}, el={metrics.get('example_el_center_deg', float('nan')):.1f}\n"
                        "\n"
                        "Quality\n"
                        f"N_vis                    : {beta_r.size:,}\n"
                        f"SW(real vs sampler)      : {metrics.get('example_sw_real_sampler', float('nan')):.4e} ± {metrics.get('example_sw_real_sampler_std', float('nan')):.2e}\n"
                        f"SW(real vs real)         : {metrics.get('example_sw_real_real_mean', float('nan')):.4e} ± {metrics.get('example_sw_real_real_std', float('nan')):.2e}\n"
                        f"SW(floor)                : {metrics.get('example_sw_floor_mean', float('nan')):.4e} ± {metrics.get('example_sw_floor_std', float('nan')):.2e}\n"
                        f"denominator kind         : {metrics.get('example_denom_kind', '-')}\n"
                        f"denominator              : {metrics.get('example_denom_mean', float('nan')):.4e} ± {metrics.get('example_denom_std', float('nan')):.2e}\n"
                        f"ratio_eff                : {metrics.get('example_ratio_eff', float('nan')):.3f}\n"
                        f"ratio_scale              : {metrics.get('example_ratio_scale_used', float('nan')):.3f}\n"
                        f"score                    : {metrics.get('example_score', float('nan')):.2f} / 100\n"
                        f"grade                    : {metrics.get('example_grade', '-')}\n"
                        "\n"
                        "Histogram diagnostics\n"
                        f"bins (β, α)              : {ex_hist_beta_bins}, {ex_hist_alpha_bins}\n"
                        f"TV                       : {metrics.get('tv', float('nan')):.4e}\n"
                        f"RMS                      : {metrics.get('rms', float('nan')):.4e}\n"
                        f"KL(P||Q)                 : {metrics.get('kl_pq', float('nan')):.4e}\n"
                        f"KL(Q||P)                 : {metrics.get('kl_qp', float('nan')):.4e}\n"
                        "\n"
                        "Metric notes:\n"
                        "TV       = total variation distance of 2D histograms.\n"
                        "RMS      = root-mean-square per-bin difference.\n"
                        "KL(P||Q) = forward KL divergence (real -> sampler).\n"
                        "KL(Q||P) = reverse KL divergence (sampler -> real).\n"
                    ),
                    transform=ax3.transAxes,
                    va="center",
                    family="monospace",
                    fontsize=rep_text_fs,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D0D0D0", alpha=0.92),
                )
                fig3.tight_layout()
                if save_prefix:
                    fig3.savefig(f"{save_prefix}_conditional_example_joint.png", dpi=200)

            if render_matplotlib:
                plt.show()
            if metrics_path:
                self.save_metrics(metrics, metrics_path)
            return metrics

        # ------------------------------------------------------------------
        # Legacy global-comparison path (single sampler)
        # ------------------------------------------------------------------
        rng = np.random.default_rng(int(seed))

        beta = _to_degrees(beta_real_deg).reshape(-1)
        alpha = _to_degrees(alpha_real_deg).reshape(-1)

        m = _finite_mask(beta, alpha)
        beta = beta[m]
        alpha = alpha[m]

        if self.alpha_wrapped:
            alpha = _wrap_alpha_deg(alpha, self.alpha_range)

        n = int(min(n_vis, beta.size))
        if beta.size > n:
            idx = rng.choice(beta.size, size=n, replace=False)
            beta_r = beta[idx]
            alpha_r = alpha[idx]
        else:
            beta_r = beta
            alpha_r = alpha

        beta_s, alpha_s = self.sample(rng, size=beta_r.shape[0])

        metrics = self.evaluate_quality(
            beta_r, alpha_r,
            n_vis=int(beta_r.size),
            seed=int(seed),
            n_slices=int(n_slices),
            alpha_weight=float(alpha_weight),
            baseline_trials=int(baseline_trials),
            auto_ratio_scale=bool(auto_ratio_scale),
            ratio_scale=float(ratio_scale),
            compute_hist_metrics=True,
        )

        # Shared histogram stats used by both backends.
        beta_edges_plot = np.linspace(self.beta_range[0], self.beta_range[1], int(plot_beta_bins) + 1)
        c_ref_b, _ = np.histogram(beta_r, bins=beta_edges_plot)
        c_sam_b, _ = np.histogram(beta_s, bins=beta_edges_plot)
        w_beta = np.diff(beta_edges_plot)
        ref_d_b = c_ref_b / (c_ref_b.sum() * w_beta)
        sam_d_b = c_sam_b / (c_sam_b.sum() * w_beta)

        alpha_edges_plot = np.linspace(self.alpha_range[0], self.alpha_range[1], int(plot_alpha_bins) + 1)
        c_ref_a, _ = np.histogram(alpha_r, bins=alpha_edges_plot)
        c_sam_a, _ = np.histogram(alpha_s, bins=alpha_edges_plot)
        w_alpha = np.diff(alpha_edges_plot)
        ref_d_a = c_ref_a / (c_ref_a.sum() * w_alpha)
        sam_d_a = c_sam_a / (c_sam_a.sum() * w_alpha)

        if render_matplotlib:
            # 1D beta plot
            fig1 = plt.figure(figsize=(12, 5))
            ax = fig1.add_subplot(1, 1, 1)
            ax.bar(beta_edges_plot[:-1], ref_d_b, width=w_beta, align="edge", alpha=0.35, label="real β")
            ax.step(beta_edges_plot[:-1], sam_d_b, where="post", linewidth=2.0, label="sampler β")
            ax.set_xlabel(r"$\beta$ [deg]")
            ax.set_ylabel("density")
            ax.set_title("β distribution: real vs sampler")
            ax.legend()
            fig1.tight_layout()
            if save_prefix:
                fig1.savefig(f"{save_prefix}_beta_1d.png", dpi=200)

            # 1D alpha plot
            fig2 = plt.figure(figsize=(12, 5))
            ax = fig2.add_subplot(1, 1, 1)
            ax.bar(alpha_edges_plot[:-1], ref_d_a, width=w_alpha, align="edge", alpha=0.35, label="real α")
            ax.step(alpha_edges_plot[:-1], sam_d_a, where="post", linewidth=2.0, label="sampler α")
            ax.set_xlabel(r"$\alpha$ [deg]")
            ax.set_ylabel("density")
            ax.set_title("α distribution: real vs sampler")
            ax.legend()
            fig2.tight_layout()
            if save_prefix:
                fig2.savefig(f"{save_prefix}_alpha_1d.png", dpi=200)

            # Polar + quality panel
            phi_r = np.deg2rad(alpha_r)
            r_r = beta_r
            phi_s = np.deg2rad(alpha_s)
            r_s = beta_s

            fig3 = plt.figure(figsize=(18, 6))
            ax1 = fig3.add_subplot(1, 3, 1, projection="polar")
            ax2 = fig3.add_subplot(1, 3, 2, projection="polar")
            ax3 = fig3.add_subplot(1, 3, 3)

            ax1.scatter(phi_r, r_r, s=1, alpha=0.1)
            ax1.set_title("Real (α, β)", pad=18)

            ax2.scatter(phi_s, r_s, s=1, alpha=0.1)
            ax2.set_title("Sampler (α, β)", pad=18)

            r_max = float(max(np.max(r_r), np.max(r_s))) if r_r.size and r_s.size else 1.0
            ax1.set_rlim(0, r_max)
            ax2.set_rlim(0, r_max)

            ax3.axis("off")
            txt = (
                "Quality\n"
                "=======\n"
                f"N_vis               : {metrics['n_vis']:,}\n"
                f"sampler bins (β, α) : {len(self.beta_edges)-1}, {len(self.alpha_edges)-1}\n"
                f"dropped (range)     : {dropped:,}\n"
                "\n"
                "Sliced Wasserstein\n"
                f"SW(real vs sampler) : {metrics['sw_real_sampler']:.4e} ± {metrics['sw_real_sampler_std']:.2e}\n"
                f"SW(real vs real)    : {metrics['sw_real_real_mean']:.4e} ± {metrics['sw_real_real_std']:.2e}\n"
                f"SW(floor)           : {metrics['sw_floor_mean']:.4e} ± {metrics['sw_floor_std']:.2e}\n"
                f"denominator         : {metrics['denom_kind']}\n"
                f"ratio_eff           : {metrics['ratio_eff']:.3f}\n"
                "\n"
                f"Score               : {metrics['score']:.2f} / 100\n"
                f"Grade               : {metrics['grade']}\n"
                "\n"
                "Hist diagnostics (coarse)\n"
                f"bins (β, α)          : {metrics.get('hist_beta_bins','-')}, {metrics.get('hist_alpha_bins','-')}\n"
                f"TV                   : {metrics.get('tv', float('nan')):.4e}\n"
                f"RMS                  : {metrics.get('rms', float('nan')):.4e}\n"
                f"KL(P||Q)             : {metrics.get('kl_pq', float('nan')):.4e}\n"
                f"KL(Q||P)             : {metrics.get('kl_qp', float('nan')):.4e}\n"
            )
            ax3.text(
                0.0,
                0.5,
                txt,
                transform=ax3.transAxes,
                va="center",
                family="monospace",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D0D0D0", alpha=0.92),
            )

            fig3.suptitle("Real vs sampler joint (α, β)", y=1.04)
            fig3.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

            if save_prefix:
                fig3.savefig(f"{save_prefix}_joint_polar_metrics.png", dpi=200)

            plt.show()

        if render_plotly:
            fig_beta = go.Figure()
            fig_beta.add_trace(
                go.Bar(
                    x=beta_edges_plot[:-1],
                    y=ref_d_b,
                    name="Real β",
                    marker=dict(color="#8E9AA6"),
                    opacity=0.40,
                )
            )
            fig_beta.add_trace(
                go.Scatter(
                    x=beta_edges_plot[:-1],
                    y=sam_d_b,
                    name="Sampler β",
                    mode="lines",
                    line=dict(color="#2F3B46", width=2.3),
                )
            )
            fig_beta.update_layout(
                template="plotly_white",
                title="β distribution: real vs sampler",
                xaxis_title="β [deg]",
                yaxis_title="Probability Density",
                width=1100,
                height=430,
            )

            fig_alpha = go.Figure()
            fig_alpha.add_trace(
                go.Bar(
                    x=alpha_edges_plot[:-1],
                    y=ref_d_a,
                    name="Real α",
                    marker=dict(color="#8E9AA6"),
                    opacity=0.40,
                )
            )
            fig_alpha.add_trace(
                go.Scatter(
                    x=alpha_edges_plot[:-1],
                    y=sam_d_a,
                    name="Sampler α",
                    mode="lines",
                    line=dict(color="#2F3B46", width=2.3),
                )
            )
            fig_alpha.update_layout(
                template="plotly_white",
                title="α distribution: real vs sampler",
                xaxis_title="α [deg]",
                yaxis_title="Probability Density",
                width=1100,
                height=430,
            )

            fig_joint = make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "polar"}, {"type": "polar"}]],
                subplot_titles=("Real (α, β)", "Sampler (α, β)"),
                horizontal_spacing=0.10,
            )
            fig_joint.add_trace(
                go.Scatterpolar(
                    theta=alpha_r,
                    r=beta_r,
                    mode="markers",
                    marker=dict(size=3, opacity=0.16, color="#5B6B7A"),
                    name="Real",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig_joint.add_trace(
                go.Scatterpolar(
                    theta=alpha_s,
                    r=beta_s,
                    mode="markers",
                    marker=dict(size=3, opacity=0.16, color="#2F3B46"),
                    name="Sampler",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig_joint.update_layout(
                template="plotly_white",
                width=1250,
                height=620,
                title=dict(text="Real vs sampler joint (α, β)", x=0.01, xanchor="left", font=dict(size=20)),
                margin=dict(l=70, r=360, t=80, b=60),
            )
            fig_joint.add_annotation(
                x=1.02,
                y=0.50,
                xref="paper",
                yref="paper",
                xanchor="left",
                showarrow=False,
                align="left",
                font=dict(size=12, family="Courier New"),
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="rgba(0,0,0,0.20)",
                borderwidth=1,
                text=(
                    "<b>Quality summary</b><br>"
                    f"N_vis: {metrics['n_vis']:,}<br>"
                    f"SW(real,sampler): {metrics['sw_real_sampler']:.4e} ± {metrics['sw_real_sampler_std']:.2e}<br>"
                    f"SW(real,real): {metrics['sw_real_real_mean']:.4e} ± {metrics['sw_real_real_std']:.2e}<br>"
                    f"SW(floor): {metrics['sw_floor_mean']:.4e} ± {metrics['sw_floor_std']:.2e}<br>"
                    f"ratio_eff: {metrics['ratio_eff']:.3f}<br>"
                    f"Score: {metrics['score']:.2f} / 100<br>"
                    f"Grade: {metrics['grade']}<br>"
                    "<br>TV: total variation<br>"
                    "RMS: root-mean-square bin error<br>"
                    "KL(P||Q), KL(Q||P): KL divergences"
                ),
            )

            if plotly_show:
                fig_beta.show()
                fig_alpha.show()
                fig_joint.show()
            if save_prefix:
                fig_beta.write_html(f"{save_prefix}_beta_1d_interactive.html")
                fig_alpha.write_html(f"{save_prefix}_alpha_1d_interactive.html")
                fig_joint.write_html(f"{save_prefix}_joint_polar_metrics_interactive.html")

        if metrics_path:
            self.save_metrics(metrics, metrics_path)
        return metrics

    def plot_sampler_only(
        self,
        *,
        save_prefix: Optional[str] = None,
        n_vis: int = 200_000,
        seed: int = 42,
        plot_beta_bins: Optional[int] = None,
        plot_alpha_bins: Optional[int] = None,
        backend: str = "matplotlib",
        interactive_plotly: bool = True,
        show_joint_pmf: bool = False,
        conditioned: Optional[bool] = None,
        representative_group_id: Optional[int] = None,
        group_selection: str = "auto",
        sat_azimuth_deg: Any = None,
        sat_elevation_deg: Any = None,
        sat_belt_id: Any = None,
        return_stats: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Visualize the sampler's own α/β distributions without real-data input.

        This is a lightweight companion to :meth:`show_comparison` and relies
        solely on the stored histogram, so it can be used even when the
        original dataset is unavailable (e.g., downstream deployment).

        Parameters
        ----------
        save_prefix : str, optional
            If provided, figures are also saved as ``<prefix>_beta_1d.png``,
            ``<prefix>_alpha_1d.png`` and ``<prefix>_joint_polar.png`` for
            Matplotlib, or ``*.html`` for Plotly.
        n_vis : int
            Number of random samples used for the polar scatter view.
        seed : int
            RNG seed for reproducible sampler-only draws.
        plot_beta_bins : int, optional
            Optional override for how many β bins to display. Defaults to the
            sampler resolution.
        plot_alpha_bins : int, optional
            Optional override for how many α bins to display. Defaults to the
            sampler resolution.
        backend : {"matplotlib", "plotly"}
            Plotting backend. Plotly enables rich hover and zoom interactions.
        interactive_plotly : bool
            If True and backend="plotly", figures are displayed immediately.
        show_joint_pmf : bool
            If True, add a heatmap of the stored joint PMF. Disabled by default
            because the polar view is usually easier to interpret visually.
        conditioned : bool, optional
            Control whether a conditional sub-sampler is visualized instead of
            the global sampler.
            - ``None`` (default): auto mode. Uses a conditional sub-sampler only
              when explicit conditional context is provided.
            - ``True``: force conditional mode (requires a built conditional model).
            - ``False``: force global sampler view.
        representative_group_id : int, optional
            Explicit conditional group id to visualize when conditional mode is
            enabled. Group id convention:
            ``group_id = belt_id * n_skycells + skycell_id``.
        group_selection : {"auto", "largest", "random", "from_conditions"}
            Conditional group picker used when ``representative_group_id`` is not
            provided.
            - ``auto``: ``from_conditions`` if sat_az/el/belt inputs are given,
              otherwise ``largest``.
            - ``largest``: highest-support non-empty conditional group.
            - ``random``: random non-empty conditional group.
            - ``from_conditions``: most frequent valid group inferred from
              ``sat_azimuth_deg/sat_elevation_deg/sat_belt_id``.
        sat_azimuth_deg, sat_elevation_deg, sat_belt_id : array-like, optional
            Optional condition arrays used by ``group_selection='from_conditions'``.
            All three must be passed together.
        return_stats : bool
            If True, return a dictionary with plotted marginals and selection
            metadata; otherwise return ``None``. This keeps notebook output
            clean by default.

        Returns
        -------
        dict or None
            Plot payload summary when ``return_stats=True``, else ``None``.
        """
        backend_norm = str(backend).strip().lower()
        if backend_norm not in ("matplotlib", "plotly"):
            raise ValueError("backend must be 'matplotlib' or 'plotly'.")
        if backend_norm == "matplotlib" and plt is None:
            raise RuntimeError("matplotlib is required for plot_sampler_only(backend='matplotlib').")
        if backend_norm == "plotly" and go is None:
            raise RuntimeError("plotly is required for plot_sampler_only(backend='plotly').")
        plotly_show = bool(interactive_plotly or backend_norm == "plotly")

        group_selection_norm = str(group_selection).strip().lower()
        if group_selection_norm not in ("auto", "largest", "random", "from_conditions"):
            raise ValueError("group_selection must be one of: 'auto', 'largest', 'random', 'from_conditions'.")

        have_any_cond = (
            sat_azimuth_deg is not None or
            sat_elevation_deg is not None or
            sat_belt_id is not None
        )
        have_all_cond = (
            sat_azimuth_deg is not None and
            sat_elevation_deg is not None and
            sat_belt_id is not None
        )
        if have_any_cond and not have_all_cond:
            raise ValueError(
                "sat_azimuth_deg, sat_elevation_deg and sat_belt_id must be passed together."
            )

        cond_ready = self._has_conditional_model()
        if conditioned is None:
            use_conditional = bool(
                cond_ready and (
                    representative_group_id is not None or
                    have_all_cond or
                    group_selection_norm != "auto"
                )
            )
        else:
            use_conditional = bool(conditioned)

        if use_conditional and not cond_ready:
            raise RuntimeError(
                "conditioned=True requested but this sampler has no conditional model. "
                "Rebuild with from_recovered(..., build_conditional=True)."
            )

        n_vis_i = int(max(1, n_vis))
        rng = np.random.default_rng(int(seed))

        def _density_from_samples(x: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            edges_f = np.asarray(edges, dtype=np.float64)
            counts, _ = np.histogram(x, bins=edges_f)
            widths = np.diff(edges_f)
            tot = float(counts.sum())
            if tot > 0.0:
                dens = counts.astype(np.float64, copy=False) / (tot * widths)
            else:
                dens = np.zeros(widths.size, dtype=np.float64)
            centers = 0.5 * (edges_f[:-1] + edges_f[1:])
            return centers, dens

        # Plot context defaults to global sampler and is overridden in conditional mode.
        plot_mode = "global"
        selection_mode = "global"
        selected_group_id: Optional[int] = None
        selected_belt_id: Optional[int] = None
        selected_skycell_id: Optional[int] = None
        context_lines = ["Global sampler (all belts and sky directions combined)"]
        beta_draw = np.empty(0, dtype=np.float64)
        alpha_draw = np.empty(0, dtype=np.float64)

        if use_conditional:
            assert self.group_raw_counts is not None
            assert self.group_ptr is not None
            assert self.group_beta_pool is not None
            assert self.group_alpha_pool is not None

            gids_nonempty = np.nonzero(
                (self.group_raw_counts > 0) & (self.group_ptr[1:] > self.group_ptr[:-1])
            )[0].astype(np.int64, copy=False)
            if gids_nonempty.size == 0:
                raise RuntimeError("Conditional model is enabled but no non-empty group pools are available.")

            gid_sel: Optional[int] = None

            if representative_group_id is not None:
                gid_sel = int(representative_group_id)
                if gid_sel < 0 or gid_sel >= int(self.group_raw_counts.size):
                    raise ValueError(f"representative_group_id={gid_sel} is out of range.")
                if int(self.group_ptr[gid_sel + 1]) <= int(self.group_ptr[gid_sel]):
                    raise ValueError(f"representative_group_id={gid_sel} has no stored pool.")
                selection_mode = "explicit-group"
            else:
                mode_use = group_selection_norm
                if mode_use == "auto":
                    mode_use = "from_conditions" if have_all_cond else "largest"

                if mode_use == "largest":
                    counts_ne = self.group_raw_counts[gids_nonempty].astype(np.int64, copy=False)
                    gid_sel = int(gids_nonempty[int(np.argmax(counts_ne))])
                elif mode_use == "random":
                    gid_sel = int(rng.choice(gids_nonempty))
                elif mode_use == "from_conditions":
                    if not have_all_cond:
                        raise ValueError(
                            "group_selection='from_conditions' requires sat_azimuth_deg, "
                            "sat_elevation_deg and sat_belt_id."
                        )

                    az_c = _to_degrees(sat_azimuth_deg).reshape(-1)
                    el_c = _to_degrees(sat_elevation_deg).reshape(-1)
                    belt_c_raw = np.asarray(sat_belt_id, dtype=np.float64).reshape(-1)
                    if az_c.size != el_c.size or az_c.size != belt_c_raw.size:
                        raise ValueError("Condition arrays must have the same flattened size.")

                    m = np.isfinite(az_c) & np.isfinite(el_c) & np.isfinite(belt_c_raw)
                    if not np.any(m):
                        raise ValueError("No finite condition rows are available to infer a conditional group.")

                    belt_c_i = np.rint(belt_c_raw[m]).astype(np.int64, copy=False)
                    m_int = np.abs(belt_c_raw[m] - belt_c_i.astype(np.float64, copy=False)) <= 1e-6
                    belt_c_i = belt_c_i[m_int]
                    az_use = az_c[m][m_int]
                    el_use = el_c[m][m_int]
                    m_belt = (belt_c_i >= 0) & (belt_c_i < int(self.n_belts))
                    belt_c_i = belt_c_i[m_belt]
                    az_use = az_use[m_belt]
                    el_use = el_use[m_belt]
                    if belt_c_i.size == 0:
                        raise ValueError("No valid belt IDs remain while inferring conditional group.")

                    sky_use = self._skycell_id_from_observer_angles(az_use, el_use).astype(np.int64, copy=False)
                    m_sky = (sky_use >= 0) & (sky_use < int(self.n_skycells))
                    sky_use = sky_use[m_sky]
                    belt_use = belt_c_i[m_sky]
                    if sky_use.size == 0:
                        raise ValueError("No valid sky-cell rows remain while inferring conditional group.")

                    gid_c = belt_use * np.int64(self.n_skycells) + sky_use
                    m_pool = self.group_raw_counts[gid_c] > 0
                    gid_c = gid_c[m_pool]
                    if gid_c.size == 0:
                        raise ValueError("Condition rows map to groups with no stored pool samples.")

                    gid_u, gid_n = np.unique(gid_c, return_counts=True)
                    gid_sel = int(gid_u[int(np.argmax(gid_n))])
                else:
                    raise RuntimeError(f"Unhandled group selection mode: {mode_use!r}")

                selection_mode = mode_use

            if gid_sel is None:
                raise RuntimeError("Failed to choose a conditional group for plotting.")
            selected_group_id = int(gid_sel)

            p0 = int(self.group_ptr[gid_sel])
            p1 = int(self.group_ptr[gid_sel + 1])
            if p1 <= p0:
                raise RuntimeError(f"Selected conditional group {gid_sel} has an empty pool.")

            pool_len = p1 - p0
            if n_vis_i <= pool_len:
                draw_idx = (p0 + rng.choice(pool_len, size=n_vis_i, replace=False)).astype(np.int64, copy=False)
            else:
                draw_idx = rng.integers(p0, p1, size=n_vis_i).astype(np.int64, copy=False)

            beta_draw = self.group_beta_pool[draw_idx].astype(np.float64, copy=False)
            alpha_draw = self.group_alpha_pool[draw_idx].astype(np.float64, copy=False)

            gdesc = self._describe_group_ids(np.asarray([gid_sel], dtype=np.int64))
            belt_sel = int(gdesc["belt_id"][0])
            sky_sel = int(gdesc["skycell_id"][0])
            selected_belt_id = belt_sel
            selected_skycell_id = sky_sel
            az_c = float(gdesc["az_center_deg"][0])
            el_c = float(gdesc["el_center_deg"][0])
            az_lo = float(gdesc["az_low_deg"][0])
            az_hi = float(gdesc["az_high_deg"][0])
            el_lo = float(gdesc["el_low_deg"][0])
            el_hi = float(gdesc["el_high_deg"][0])

            plot_mode = "conditional-group"
            context_lines = [
                f"Conditional sub-sampler: group_id={gid_sel}, belt={belt_sel}, skycell={sky_sel}",
                f"Sky-cell center: az={az_c:.1f} deg, el={el_c:.1f} deg",
                f"Sky-cell bounds: az=[{az_lo:.1f}, {az_hi:.1f}] deg, el=[{el_lo:.1f}, {el_hi:.1f}] deg",
                f"Selection mode: {selection_mode}",
                f"Stored pool size: {pool_len:,}",
            ]

        # Optionally re-bin for cleaner plots without touching the underlying PMF.
        if plot_beta_bins:
            beta_edges_plot = np.linspace(self.beta_range[0], self.beta_range[1], int(plot_beta_bins) + 1)
        else:
            beta_edges_plot = self.beta_edges.astype(np.float64, copy=False)

        if plot_alpha_bins:
            alpha_edges_plot = np.linspace(self.alpha_range[0], self.alpha_range[1], int(plot_alpha_bins) + 1)
        else:
            alpha_edges_plot = self.alpha_edges.astype(np.float64, copy=False)

        if plot_mode == "global":
            stats = self.sampler_statistics()
            beta_draw, alpha_draw = self.sample(rng, size=n_vis_i)
            beta_draw = np.asarray(beta_draw, dtype=np.float64).reshape(-1)
            alpha_draw = np.asarray(alpha_draw, dtype=np.float64).reshape(-1)

            beta_centers = stats["beta_centers"]
            alpha_centers = stats["alpha_centers"]
            beta_density = stats["beta_density"]
            alpha_density = stats["alpha_density"]
            beta_widths = np.diff(self.beta_edges.astype(np.float64, copy=False))
            alpha_widths = np.diff(self.alpha_edges.astype(np.float64, copy=False))
            beta_probs = beta_density * beta_widths
            alpha_probs = alpha_density * alpha_widths

            if int(beta_edges_plot.size) != int(beta_centers.size + 1):
                beta_weights, _ = np.histogram(beta_centers, bins=beta_edges_plot, weights=beta_probs)
                beta_widths_plot = np.diff(beta_edges_plot)
                beta_centers_plot = 0.5 * (beta_edges_plot[:-1] + beta_edges_plot[1:])
                beta_total = float(beta_weights.sum())
                beta_density_plot = beta_weights / (beta_widths_plot * beta_total if beta_total > 0 else 1.0)
            else:
                beta_centers_plot = beta_centers
                beta_density_plot = beta_density

            if int(alpha_edges_plot.size) != int(alpha_centers.size + 1):
                alpha_weights, _ = np.histogram(alpha_centers, bins=alpha_edges_plot, weights=alpha_probs)
                alpha_widths_plot = np.diff(alpha_edges_plot)
                alpha_centers_plot = 0.5 * (alpha_edges_plot[:-1] + alpha_edges_plot[1:])
                alpha_total = float(alpha_weights.sum())
                alpha_density_plot = alpha_weights / (alpha_widths_plot * alpha_total if alpha_total > 0 else 1.0)
            else:
                alpha_centers_plot = alpha_centers
                alpha_density_plot = alpha_density
        else:
            beta_centers_plot, beta_density_plot = _density_from_samples(beta_draw, beta_edges_plot)
            alpha_centers_plot, alpha_density_plot = _density_from_samples(alpha_draw, alpha_edges_plot)

        context_header = context_lines[0]
        context_note = "<br>".join(context_lines[1:])

        if backend_norm == "matplotlib":
            # 1D beta
            fig1 = plt.figure(figsize=(12, 4))
            ax1 = fig1.add_subplot(1, 1, 1)
            ax1.bar(
                beta_centers_plot,
                beta_density_plot,
                width=(self.beta_range[1] - self.beta_range[0]) / beta_centers_plot.size,
                align="center",
                alpha=0.7,
                color="#4C78A8",
                label="sampler β",
            )
            ax1.set_xlabel(r"$\beta$ [deg]")
            ax1.set_ylabel("density")
            ax1.set_title(f"Sampler β Distribution ({plot_mode})")
            ax1.legend()
            fig1.tight_layout()
            if save_prefix:
                fig1.savefig(f"{save_prefix}_beta_1d.png", dpi=200)

            # 1D alpha
            fig2 = plt.figure(figsize=(12, 4))
            ax2 = fig2.add_subplot(1, 1, 1)
            ax2.bar(
                alpha_centers_plot,
                alpha_density_plot,
                width=(self.alpha_range[1] - self.alpha_range[0]) / alpha_centers_plot.size,
                align="center",
                alpha=0.7,
                color="#4C78A8",
                label="sampler α",
            )
            ax2.set_xlabel(r"$\alpha$ [deg]")
            ax2.set_ylabel("density")
            ax2.set_title(f"Sampler α Distribution ({plot_mode})")
            ax2.legend()
            fig2.tight_layout()
            if save_prefix:
                fig2.savefig(f"{save_prefix}_alpha_1d.png", dpi=200)

            # Polar view from random draws (usually the most interpretable view).
            fig3 = plt.figure(figsize=(12, 6))
            ax3 = fig3.add_subplot(1, 1, 1, projection="polar")
            ax3.scatter(np.deg2rad(alpha_draw), beta_draw, s=1, alpha=0.10, color="#1F77B4")
            ax3.set_title(f"Sampler joint (α, β) - random draws ({plot_mode})", pad=18)
            rmax = float(np.nanmax(beta_draw)) if beta_draw.size else 1.0
            ax3.set_rlim(0.0, max(rmax, 1.0))
            ax3.text(
                0.01,
                0.02,
                "\n".join(context_lines),
                transform=ax3.transAxes,
                ha="left",
                va="bottom",
                fontsize=9.5,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", alpha=0.90),
            )
            fig3.tight_layout()
            if save_prefix:
                fig3.savefig(f"{save_prefix}_joint_polar.png", dpi=200)

            if show_joint_pmf:
                if plot_mode == "global":
                    pmf = self._pmf_2d()
                else:
                    pmf, _, _ = np.histogram2d(
                        beta_draw, alpha_draw,
                        bins=[beta_edges_plot, alpha_edges_plot],
                    )
                    p_tot = float(pmf.sum())
                    if p_tot > 0.0:
                        pmf = pmf / p_tot
                fig4 = plt.figure(figsize=(10, 6))
                ax4 = fig4.add_subplot(1, 1, 1)
                im = ax4.imshow(
                    pmf.T,
                    origin="lower",
                    aspect="auto",
                    extent=[self.beta_range[0], self.beta_range[1], self.alpha_range[0], self.alpha_range[1]],
                    cmap="Blues",
                )
                ax4.set_xlabel(r"$\beta$ [deg]")
                ax4.set_ylabel(r"$\alpha$ [deg]")
                ax4.set_title(f"Sampler joint PMF (β, α) ({plot_mode})")
                fig4.colorbar(im, ax=ax4, label="Probability")
                fig4.tight_layout()
                if save_prefix:
                    fig4.savefig(f"{save_prefix}_joint_heatmap.png", dpi=200)

            plt.show()
        else:
            fig_beta = go.Figure()
            fig_beta.add_trace(
                go.Bar(
                    x=beta_centers_plot,
                    y=beta_density_plot,
                    name="sampler β",
                    marker=dict(color="#4C78A8"),
                    hovertemplate="β=%{x:.3f} deg<br>density=%{y:.4e}<extra></extra>",
                )
            )
            fig_beta.update_layout(
                template="plotly_white",
                title=f"Sampler β Distribution ({plot_mode})",
                xaxis_title="β [deg]",
                yaxis_title="Probability Density",
                width=1100,
                height=420,
            )

            fig_alpha = go.Figure()
            fig_alpha.add_trace(
                go.Bar(
                    x=alpha_centers_plot,
                    y=alpha_density_plot,
                    name="sampler α",
                    marker=dict(color="#4C78A8"),
                    hovertemplate="α=%{x:.3f} deg<br>density=%{y:.4e}<extra></extra>",
                )
            )
            fig_alpha.update_layout(
                template="plotly_white",
                title=f"Sampler α Distribution ({plot_mode})",
                xaxis_title="α [deg]",
                yaxis_title="Probability Density",
                width=1100,
                height=420,
            )

            fig_polar = go.Figure(
                data=go.Scatterpolar(
                    theta=alpha_draw,
                    r=beta_draw,
                    mode="markers",
                    marker=dict(size=3, opacity=0.16, color="#1F77B4"),
                    hovertemplate="α=%{theta:.2f} deg<br>β=%{r:.2f} deg<extra></extra>",
                    showlegend=False,
                )
            )
            fig_polar.update_layout(
                template="plotly_white",
                title=f"Sampler joint (α, β) - random draws ({plot_mode})",
                polar=dict(
                    radialaxis=dict(title="β [deg]"),
                    angularaxis=dict(direction="counterclockwise"),
                ),
                width=1100,
                height=520,
            )
            fig_polar.add_annotation(
                x=0.01,
                y=0.01,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                align="left",
                font=dict(size=11, family="Courier New"),
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="rgba(0,0,0,0.20)",
                borderwidth=1,
                text=(
                    f"<b>{context_header}</b><br>"
                    f"{context_note}"
                ),
            )

            fig_heat = None
            if show_joint_pmf:
                if plot_mode == "global":
                    pmf = self._pmf_2d()
                else:
                    pmf, _, _ = np.histogram2d(
                        beta_draw, alpha_draw,
                        bins=[beta_edges_plot, alpha_edges_plot],
                    )
                    p_tot = float(pmf.sum())
                    if p_tot > 0.0:
                        pmf = pmf / p_tot
                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=pmf.T,
                        x=0.5 * (beta_edges_plot[:-1] + beta_edges_plot[1:]),
                        y=0.5 * (alpha_edges_plot[:-1] + alpha_edges_plot[1:]),
                        colorscale="Blues",
                        colorbar=dict(title="Probability"),
                        hovertemplate="β=%{x:.3f} deg<br>α=%{y:.3f} deg<br>P=%{z:.4e}<extra></extra>",
                    )
                )
                fig_heat.update_layout(
                    template="plotly_white",
                    title=f"Sampler joint PMF (β, α) ({plot_mode})",
                    xaxis_title="β [deg]",
                    yaxis_title="α [deg]",
                    width=1100,
                    height=520,
                )

            if plotly_show:
                fig_beta.show()
                fig_alpha.show()
                fig_polar.show()
                if fig_heat is not None:
                    fig_heat.show()
            if save_prefix:
                fig_beta.write_html(f"{save_prefix}_beta_1d_interactive.html")
                fig_alpha.write_html(f"{save_prefix}_alpha_1d_interactive.html")
                fig_polar.write_html(f"{save_prefix}_joint_polar_interactive.html")
                if fig_heat is not None:
                    fig_heat.write_html(f"{save_prefix}_joint_heatmap_interactive.html")

        stats_out: Dict[str, Any] = {
            "plot_mode": plot_mode,
            "selection_mode": selection_mode,
            "beta_centers": np.asarray(beta_centers_plot, dtype=np.float64),
            "beta_density": np.asarray(beta_density_plot, dtype=np.float64),
            "alpha_centers": np.asarray(alpha_centers_plot, dtype=np.float64),
            "alpha_density": np.asarray(alpha_density_plot, dtype=np.float64),
            "context_lines": tuple(context_lines),
            "n_vis_used": int(beta_draw.size),
        }

        if use_conditional:
            stats_out["group_id"] = int(selected_group_id) if selected_group_id is not None else -1
            stats_out["belt_id"] = int(selected_belt_id) if selected_belt_id is not None else -1
            stats_out["skycell_id"] = int(selected_skycell_id) if selected_skycell_id is not None else -1

        return stats_out if return_stats else None

    def show_sampler_only(
        self,
        *,
        save_prefix: Optional[str] = None,
        plot_beta_bins: Optional[int] = None,
        plot_alpha_bins: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Backward-compatible alias for :meth:`plot_sampler_only` using Matplotlib.
        """
        return self.plot_sampler_only(
            save_prefix=save_prefix,
            plot_beta_bins=plot_beta_bins,
            plot_alpha_bins=plot_alpha_bins,
            backend="matplotlib",
            interactive_plotly=False,
            return_stats=True,
        )
