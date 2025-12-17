# scepter/angle_sampler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

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


SizeLike = Union[None, int, Tuple[int, ...]]


# ============================================================================
# Utilities
# ============================================================================

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


# ============================================================================
# Streaming histogram builder for uniform edges
# ============================================================================

def _hist2d_stream_uniform_edges(
    beta: np.ndarray,
    alpha: np.ndarray,
    beta_edges: np.ndarray,
    alpha_edges: np.ndarray,
    *,
    chunk_size: int = 5_000_000,
) -> np.ndarray:
    """
    Build 2D histogram counts using streaming accumulation and bincount.

    This avoids potential heavy temporary allocations of np.histogram2d on very large datasets.

    Assumptions:
      - edges are uniform (linspace), which is exactly how we build them in this library.

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

    counts_flat = np.zeros(n_bins_total, dtype=np.float64)

    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    n = beta.size
    if n != alpha.size:
        raise ValueError("beta and alpha must have the same length.")

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    for start in range(0, n, chunk_size):
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
    """
    if rng is None:
        rng = np.random.default_rng()

    beta_real = np.asarray(beta_real, dtype=np.float64)
    alpha_real = np.asarray(alpha_real, dtype=np.float64)
    beta_emp = np.asarray(beta_emp, dtype=np.float64)
    alpha_emp = np.asarray(alpha_emp, dtype=np.float64)

    n = min(beta_real.size, beta_emp.size)
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
        show_comparison: bool = False,
        comparison_n_vis: int = 200_000,
        comparison_seed: int = 9876,
        comparison_slices: int = 32,
        save_prefix: Optional[str] = None,
        plot_beta_bins: int = 250,
        plot_alpha_bins: int = 360,
        histogram: str = "auto",
        histogram_chunk_size: int = 5_000_000,
    ) -> "JointAngleSampler":
        """
        Build a sampler from recovered arrays.

        Default resolution:
          - alpha_bins=3600 (0.1 deg over 0..360)
          - beta_bins =900  (0.1 deg over 0..90)

        Smoothing (optional):
          - alpha axis uses circular boundary (wrap)
          - beta axis uses reflect boundary

        histogram:
          - "auto": use streaming accumulation for very large arrays, else numpy histogram2d
          - "stream": always streaming
          - "numpy": always numpy histogram2d
        """
        beta_deg = _to_degrees(sat_beta_recovered).reshape(-1)
        alpha_deg = _to_degrees(sat_alpha_recovered).reshape(-1)

        m = _finite_mask(beta_deg, alpha_deg)
        beta_deg = beta_deg[m]
        alpha_deg = alpha_deg[m]

        alpha_wrapped = False
        if wrap_alpha:
            alpha_deg = _wrap_alpha_deg(alpha_deg, alpha_range)
            alpha_wrapped = True

        beta_deg, alpha_deg, info = _filter_range(
            beta_deg, alpha_deg, beta_range, alpha_range, out_of_range=out_of_range
        )
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
            use_stream = (beta_deg.size >= 50_000_000)

        if use_stream:
            counts = _hist2d_stream_uniform_edges(
                beta_deg,
                alpha_deg,
                beta_edges.astype(np.float64),
                alpha_edges.astype(np.float64),
                chunk_size=int(histogram_chunk_size),
            )
        else:
            counts, _, _ = np.histogram2d(
                beta_deg,
                alpha_deg,
                bins=[beta_edges.astype(np.float64), alpha_edges.astype(np.float64)],
            )
            counts = counts.astype(np.float64, copy=False)

        if smooth_sigma_beta > 0.0 or smooth_sigma_alpha > 0.0:
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

        sampler = cls(
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
        )

        if show_comparison:
            sampler.show_comparison(
                beta_deg, alpha_deg,
                n_vis=int(comparison_n_vis),
                seed=int(comparison_seed),
                n_slices=int(comparison_slices),
                save_prefix=save_prefix,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample (beta_deg, alpha_deg) with output shape matching `size`.

        Examples:
          beta, alpha = sampler.sample(rng, size=200000)
          beta, alpha = sampler.sample(rng, size=some_array.shape)
          beta, alpha = sampler.sample(rng, size=(T, O, S, B))
        """
        shape, n = _normalize_size(size)

        # Scalar fast-path
        if n == 1 and shape == ():
            u0 = rng.random()
            idx = int(np.searchsorted(self.cdf, u0, side="right"))
            i_beta, i_alpha = divmod(idx, self.n_alpha)

            b0, b1 = float(self.beta_edges[i_beta]), float(self.beta_edges[i_beta + 1])
            a0, a1 = float(self.alpha_edges[i_alpha]), float(self.alpha_edges[i_alpha + 1])

            beta = b0 + rng.random() * (b1 - b0)
            alpha = a0 + rng.random() * (a1 - a0)
            return dtype(beta), dtype(alpha)

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

        return beta_out.reshape(shape), alpha_out.reshape(shape)

    # --------------------------
    # Persistence (minimal/full)
    # --------------------------

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
        )

        if mode == "full":
            payload.update(
                smooth_sigma_beta=np.float64(self.smooth_sigma_beta),
                smooth_sigma_alpha=np.float64(self.smooth_sigma_alpha),
                smooth_truncate=np.float64(self.smooth_truncate),
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
        dropped: int = 0,
        plot_beta_bins: int = 250,
        plot_alpha_bins: int = 360,
    ) -> Dict[str, Any]:
        """
        Draw:
          - 1D beta comparison
          - 1D alpha comparison
          - 3-panel: real polar, sampler polar, quality panel

        Returns metrics dict from evaluate_quality().
        """
        if plt is None:
            raise RuntimeError("matplotlib is required for show_comparison().")

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

        # 1D beta plot
        beta_edges_plot = np.linspace(self.beta_range[0], self.beta_range[1], int(plot_beta_bins) + 1)
        c_ref, _ = np.histogram(beta_r, bins=beta_edges_plot)
        c_sam, _ = np.histogram(beta_s, bins=beta_edges_plot)
        w = np.diff(beta_edges_plot)
        ref_d = c_ref / (c_ref.sum() * w)
        sam_d = c_sam / (c_sam.sum() * w)

        fig1 = plt.figure(figsize=(12, 5))
        ax = fig1.add_subplot(1, 1, 1)
        ax.bar(beta_edges_plot[:-1], ref_d, width=w, align="edge", alpha=0.35, label="real β")
        ax.step(beta_edges_plot[:-1], sam_d, where="post", linewidth=2.0, label="sampler β")
        ax.set_xlabel(r"$\beta$ [deg]")
        ax.set_ylabel("density")
        ax.set_title("β distribution: real vs sampler")
        ax.legend()
        fig1.tight_layout()
        if save_prefix:
            fig1.savefig(f"{save_prefix}_beta_1d.png", dpi=200)

        # 1D alpha plot
        alpha_edges_plot = np.linspace(self.alpha_range[0], self.alpha_range[1], int(plot_alpha_bins) + 1)
        c_ref, _ = np.histogram(alpha_r, bins=alpha_edges_plot)
        c_sam, _ = np.histogram(alpha_s, bins=alpha_edges_plot)
        w = np.diff(alpha_edges_plot)
        ref_d = c_ref / (c_ref.sum() * w)
        sam_d = c_sam / (c_sam.sum() * w)

        fig2 = plt.figure(figsize=(12, 5))
        ax = fig2.add_subplot(1, 1, 1)
        ax.bar(alpha_edges_plot[:-1], ref_d, width=w, align="edge", alpha=0.35, label="real α")
        ax.step(alpha_edges_plot[:-1], sam_d, where="post", linewidth=2.0, label="sampler α")
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
        ax3.text(0.0, 0.5, txt, transform=ax3.transAxes, va="center", family="monospace", fontsize=10)

        fig3.suptitle("Real vs sampler joint (α, β)", y=1.04)
        fig3.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

        if save_prefix:
            fig3.savefig(f"{save_prefix}_joint_polar_metrics.png", dpi=200)

        plt.show()

        return metrics