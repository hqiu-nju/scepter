# gpu_accel.py
# SPDX-License-Identifier: BSD-3-Clause
# Author: <you>
#
# True angular distance on the sphere (Vincenty formula) accelerated on GPU.
# Works with CuPy (preferred for ease + broadcasting) or Numba CUDA (portable).
# Falls back to NumPy if no GPU backend is available.
#
# Function to call:
#     true_angular_distance_CUDA(l1, b1, l2, b2, ...)
#
# Notes:
# - l* are longitudes/azimuths [deg], b* are latitudes/elevations [deg].
# - Accepts scalars, NumPy arrays, or Astropy Quantities; returns an Astropy
#   Quantity [deg] unless return_quantity=False.
# - Broadcasting:
#     * CuPy path: uses CuPy's native broadcasting (very convenient).
#     * Numba path: expects inputs broadcast to a common shape on host.
#       (We do that safely; see _broadcast_to_match.) For huge cross-products
#       prefer manual chunking (chunk_elems) to avoid allocating giant arrays.

from __future__ import annotations

import math
import numpy as np
from numpy import pi
from typing import Tuple, Optional

try:
    import cupy as cp  # Optional; preferred if available
except Exception:
    cp = None

try:
    from numba import cuda  # Optional; used if CuPy isn't available
except Exception:
    cuda = None

try:
    from astropy import units as apu
    from astropy.units import Quantity
except Exception:
    apu = None
    Quantity = None  # type: ignore


# ----------------------------- helpers ------------------------------------- #

def _to_deg_array(x, dtype=np.float64):
    """Convert scalar/array/Quantity to a NumPy array in degrees, dtype as given."""
    if apu is not None and hasattr(x, "to"):
        x = x.to_value(apu.deg)
    return np.asanyarray(x, dtype=dtype)

def _broadcast_to_match(*arrs, dtype=np.float64) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, ...]]:
    """Broadcast all inputs to a single shape, return contiguous arrays + shape."""
    arrs = [_to_deg_array(a, dtype=dtype) for a in arrs]
    bcast = np.broadcast_arrays(*arrs)
    # Ensure contiguous on host before moving to device (safer for Numba).
    return tuple(np.ascontiguousarray(a) for a in bcast), bcast[0].shape

def _as_quantity_deg(arr: np.ndarray, return_quantity: bool):
    """Wrap NumPy array as Quantity[deg] if requested."""
    if return_quantity and apu is not None:
        return arr * apu.deg
    return arr


# --------------------------- Numba CUDA kernel ------------------------------ #

if cuda is not None:
    @cuda.jit(fastmath=True)
    def _vincenty_kernel_deg(l1_deg, b1_deg, l2_deg, b2_deg, out_deg):
        """
        Elementwise great-circle distance in degrees using the Vincenty formula
        for a sphere (numerically stable atan2 form).
        """
        i = cuda.grid(1)
        stride = cuda.gridsize(1)
        d2r = 0.017453292519943295  # pi/180
        r2d = 57.29577951308232     # 180/pi

        n = out_deg.size
        for idx in range(i, n, stride):
            l1 = l1_deg[idx] * d2r
            b1 = b1_deg[idx] * d2r
            l2 = l2_deg[idx] * d2r
            b2 = b2_deg[idx] * d2r

            dl = l2 - l1
            sdl = math.sin(dl)
            cdl = math.cos(dl)
            sb1 = math.sin(b1); cb1 = math.cos(b1)
            sb2 = math.sin(b2); cb2 = math.cos(b2)

            # Vincenty (sphere) via atan2/hypot; stable near small angles
            num1 = cb2 * sdl
            num2 = cb1 * sb2 - sb1 * cb2 * cdl
            denom = sb1 * sb2 + cb1 * cb2 * cdl
            ang = math.atan2(math.hypot(num1, num2), denom)

            out_deg[idx] = ang * r2d


# ------------------------------ main API ----------------------------------- #

def true_angular_distance_CUDA(
    l1,
    b1,
    l2,
    b2,
    *,
    backend: str = "auto",          # "auto" | "cupy" | "numba" | "cpu"
    dtype=np.float64,               # float64 for astro-grade accuracy; float32 for more speed
    return_quantity: bool = True,   # return astropy Quantity [deg] if astropy is available
    cupy_out: bool = False,         # if using cupy backend, return a CuPy array (not Quantity)
    chunk_elems: Optional[int] = None,  # process in chunks of this many elements (Numba path)
    stream=None                     # optional CUDA stream (CuPy or Numba path)
):
    """
    Compute true angular distance between (l1,b1) and (l2,b2) in degrees on GPU.

    Parameters
    ----------
    l1, b1, l2, b2 : array-like or astropy.units.Quantity
        Longitudes/latitudes (or azimuths/elevations) in degrees.
        Scalars or arrays; shapes may be broadcastable.

    backend : {"auto","cupy","numba","cpu"}, optional
        "auto" chooses CuPy if available, else Numba CUDA, else CPU.

    dtype : numpy dtype, optional
        Use float64 for accuracy, float32 for speed. (CuPy/Numba will use this.)

    return_quantity : bool, optional
        If True (and Astropy is installed), return Quantity[deg]. Otherwise NumPy/CuPy array.

    cupy_out : bool, optional
        If True with CuPy backend, return a CuPy array on device (no host copy, no Quantity).

    chunk_elems : int or None, optional
        For very large problems on Numba path, split the flattened workload into
        chunks of this many elements to limit GPU memory usage.

    stream : optional
        CuPy cuda.Stream or Numba cuda.stream() to overlap copies/compute.

    Returns
    -------
    adist : array-like
        Angular distance in degrees (Quantity if return_quantity and Astropy present).
        Type is NumPy array by default; CuPy array if backend="cupy" and cupy_out=True.
    """
    # ---------------- choose backend ----------------
    if backend == "auto":
        if cp is not None:
            backend = "cupy"
        elif cuda is not None and (hasattr(cuda, "is_available") and cuda.is_available()):
            backend = "numba"
        else:
            backend = "cpu"

    # ---------------- CPU fallback (NumPy) ----------------
    if backend == "cpu":
        (l1a, b1a, l2a, b2a), out_shape = _broadcast_to_match(l1, b1, l2, b2, dtype=dtype)
        dl = np.radians(l2a - l1a)
        sb1, cb1 = np.sin(np.radians(b1a)), np.cos(np.radians(b1a))
        sb2, cb2 = np.sin(np.radians(b2a)), np.cos(np.radians(b2a))
        num1 = cb2 * np.sin(dl)
        num2 = cb1 * sb2 - sb1 * cb2 * np.cos(dl)
        denom = sb1 * sb2 + cb1 * cb2 * np.cos(dl)
        out = np.degrees(np.arctan2(np.hypot(num1, num2), denom)).reshape(out_shape)
        return _as_quantity_deg(out, return_quantity)

    # ---------------- CuPy backend ----------------
    if backend == "cupy":
        if cp is None:
            raise RuntimeError("CuPy not available but backend='cupy' requested.")
        l1c = cp.asarray(_to_deg_array(l1, dtype=dtype))
        b1c = cp.asarray(_to_deg_array(b1, dtype=dtype))
        l2c = cp.asarray(_to_deg_array(l2, dtype=dtype))
        b2c = cp.asarray(_to_deg_array(b2, dtype=dtype))

        with (stream or cp.cuda.Stream.null):
            dl = cp.radians(l2c - l1c)
            sb1, cb1 = cp.sin(cp.radians(b1c)), cp.cos(cp.radians(b1c))
            sb2, cb2 = cp.sin(cp.radians(b2c)), cp.cos(cp.radians(b2c))
            num1 = cb2 * cp.sin(dl)
            num2 = cb1 * sb2 - sb1 * cb2 * cp.cos(dl)
            denom = sb1 * sb2 + cb1 * cb2 * cp.cos(dl)
            outc = cp.degrees(cp.arctan2(cp.hypot(num1, num2), denom))

        if cupy_out:
            # Caller wants a device array; cannot wrap as Quantity on device.
            return outc
        # Move to host and optionally wrap as Quantity
        outh = cp.asnumpy(outc)
        return _as_quantity_deg(outh, return_quantity)

    # ---------------- Numba-CUDA backend ----------------
    if cuda is None or not (hasattr(cuda, "is_available") and cuda.is_available()):
        raise RuntimeError("Numba CUDA not available but backend='numba' requested.")

    (l1a, b1a, l2a, b2a), out_shape = _broadcast_to_match(l1, b1, l2, b2, dtype=dtype)
    n = int(np.prod(out_shape))
    out = np.empty(n, dtype=dtype)

    # Choose chunk size if not given; try to be memory-friendly.
    if chunk_elems is None:
        chunk_elems = n  # no chunking by default
        try:
            # Rough heuristic: keep total device footprint ~< 60% of free mem.
            free_mem, total_mem = cuda.current_context().get_memory_info()  # type: ignore
            bytes_per_elem = 5 * np.dtype(dtype).itemsize  # 4 inputs + 1 output
            max_elems = int(0.6 * free_mem // bytes_per_elem)
            if max_elems > 0:
                chunk_elems = min(n, max_elems)
        except Exception:
            pass

    threads = 256
    stream_nb = stream or cuda.stream()

    # Flatten views once
    l1f = l1a.ravel()
    b1f = b1a.ravel()
    l2f = l2a.ravel()
    b2f = b2a.ravel()

    start = 0
    while start < n:
        end = min(start + chunk_elems, n)
        m = end - start

        # Slice the current chunk
        l1_chunk = l1f[start:end]
        b1_chunk = b1f[start:end]
        l2_chunk = l2f[start:end]
        b2_chunk = b2f[start:end]

        # Allocate device buffers for the chunk
        d_l1 = cuda.to_device(l1_chunk, stream=stream_nb)
        d_b1 = cuda.to_device(b1_chunk, stream=stream_nb)
        d_l2 = cuda.to_device(l2_chunk, stream=stream_nb)
        d_b2 = cuda.to_device(b2_chunk, stream=stream_nb)
        d_out = cuda.device_array(m, dtype=dtype, stream=stream_nb)

        blocks = (m + threads - 1) // threads
        _vincenty_kernel_deg[blocks, threads, stream_nb](d_l1, d_b1, d_l2, d_b2, d_out)

        d_out.copy_to_host(out[start:end], stream=stream_nb)
        stream_nb.synchronize()
        start = end

    out = out.reshape(out_shape)
    return _as_quantity_deg(out, return_quantity)


# ------------------------------ quick tests -------------------------------- #

def _self_test_small():
    # A couple of sanity checks (CPU vs chosen backend).
    a1 = np.array([0., 0.,  10.], dtype=np.float64)
    b1 = np.array([0., 45., 20.], dtype=np.float64)
    a2 = np.array([0., 90.,  30.], dtype=np.float64)
    b2 = np.array([0., 45., -10.], dtype=np.float64)

    cpu = true_angular_distance_CUDA(a1, b1, a2, b2, backend="cpu", return_quantity=False)
    auto = true_angular_distance_CUDA(a1, b1, a2, b2, backend="auto", return_quantity=False)
    assert np.allclose(cpu, auto, rtol=1e-12, atol=1e-10), (cpu, auto)
    print("OK: CPU ≈ AUTO", cpu, auto)


if __name__ == "__main__":
    _self_test_small()
