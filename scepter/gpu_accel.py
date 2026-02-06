"""
gpu_accel.py

This module contains GPU-accelerated routines for computing the true angular distance
between points. It uses Numba's CUDA support for maximum efficiency on large datasets.
For smaller inputs (where data transfer overhead dominates), the code automatically
falls back to a CPU vectorized implementation.

Functions provided:
  - get_true_angular_distance_kernel: Returns a specialized CUDA kernel for inputs with a given number of dimensions.
  - true_angular_distance_gpu_nd: Computes the angular distances on the GPU using the specialized kernel.
  - true_angular_distance_auto: Automatically selects GPU or CPU computation based on the input size,
      issuing a runtime warning when arrays become too large for efficient GPU processing.
  
Note:
    Please ensure that input variables can fit in GPU memory.
"""

import math
import numpy as np
from numba import cuda, int32
from astropy import units as u, time
from pycraf.utils import ranged_quantity_input
import cysgp4
import warnings

# Global cache to store compiled CUDA kernels keyed by number of dimensions (ndim).
# This avoids recompiling kernels for inputs with the same number of dimensions.
kernel_cache = {}

def compute_strides(shape):
    """
    Compute the row-major strides for a given shape.
    
    In a row-major order, the stride for a dimension is the number of elements
    that must be skipped to advance one index in that dimension in the flattened array.
    
    Parameters:
      shape (tuple): Tuple of integers representing the shape of the array.
    
    Returns:
      tuple: A tuple containing the computed strides for each dimension.
    """
    # Initialize the strides list with 1 for each dimension.
    strides = [1] * len(shape)
    # Iterate backwards over dimensions (excluding the last dimension).
    for i in range(len(shape) - 2, -1, -1):
        # The stride for the current dimension is the product of the sizes of all subsequent dimensions.
        strides[i] = strides[i + 1] * shape[i + 1]
    # Convert the list to a tuple before returning.
    return tuple(strides)

def get_true_angular_distance_kernel(ndim):
    """
    Return a CUDA kernel specialized for a given number of dimensions (ndim).
    
    The kernel computes the true angular distance on the GPU by mapping a flat index
    (of the broadcasted output array) into multi-dimensional indices and then resolving
    those indices into corresponding positions in the broadcasted input arrays.
    
    The compiled kernel is cached in a global dictionary for reuse.
    
    Parameters:
      ndim (int): The number of dimensions in the broadcasted arrays.
    
    Returns:
      function: The compiled CUDA kernel function ready to launch on the GPU.
    """
    # Check if a kernel specialized for this number of dimensions already exists.
    if ndim in kernel_cache:
        return kernel_cache[ndim]
    
    # Define the specialized CUDA kernel with the provided ndim.
    @cuda.jit
    def kernel(l1, b1, l2, b2,
               s_l1, strides_l1,
               s_b1, strides_b1,
               s_l2, strides_l2,
               s_b2, strides_b2,
               b_shape, total, out):
        """
        CUDA kernel to compute angular distances.
        
        Parameters:
          l1, b1, l2, b2 (device arrays): Flattened input arrays representing angles in degrees.
          s_l1, s_b1, s_l2, s_b2 (device arrays): Original shape arrays for each input.
          strides_l1, strides_b1, strides_l2, strides_b2 (device arrays): Stride arrays computed from the input shapes.
          b_shape (device array): The shape of the broadcasted output array.
          total (int): Total number of elements in the broadcasted array.
          out (device array): Output array where the computed angular distances will be stored.
        """
        # Compute the global thread index.
        i = cuda.grid(1)
        # Ensure that the thread index does not exceed the total number of elements.
        if i < total:
            # Start with the flat index to be decomposed into multi-indices.
            temp = i
            # Allocate local memory for storing the multi-index.
            idxs = cuda.local.array(shape=ndim, dtype=int32)
            # Convert the flat index into multi-dimensional indices.
            for d in range(ndim - 1, -1, -1):
                idxs[d] = temp % b_shape[d]  # Compute the index for dimension d.
                temp //= b_shape[d]          # Update temp for the next dimension.
            
            # Initialize variables to accumulate the flat indices for each input array.
            flat_idx_l1 = 0
            flat_idx_b1 = 0
            flat_idx_l2 = 0
            flat_idx_b2 = 0
            # For each dimension, compute the corresponding flat index in the original array.
            for d in range(ndim):
                # If the current dimension of the input matches the broadcasted dimension,
                # use the multi-index; otherwise, use index 0.
                flat_idx_l1 += (idxs[d] if s_l1[d] == b_shape[d] else 0) * strides_l1[d]
                flat_idx_b1 += (idxs[d] if s_b1[d] == b_shape[d] else 0) * strides_b1[d]
                flat_idx_l2 += (idxs[d] if s_l2[d] == b_shape[d] else 0) * strides_l2[d]
                flat_idx_b2 += (idxs[d] if s_b2[d] == b_shape[d] else 0) * strides_b2[d]
            
            # Convert degrees to radians for trigonometric operations.
            r = math.pi / 180.0
            # Compute the difference in longitude (l2 - l1) in radians.
            dlon = (l2[flat_idx_l2] - l1[flat_idx_l1]) * r
            # Calculate cosine and sine of the longitude difference.
            cos_diff_lon = math.cos(dlon)
            sin_diff_lon = math.sin(dlon)
            # Convert latitudes from degrees to radians.
            lat1 = b1[flat_idx_b1] * r
            lat2 = b2[flat_idx_b2] * r
            # Compute sine and cosine values for both latitudes.
            sin_lat1 = math.sin(lat1)
            sin_lat2 = math.sin(lat2)
            cos_lat1 = math.cos(lat1)
            cos_lat2 = math.cos(lat2)
            # Compute intermediate components for the angular distance formula.
            num1 = cos_lat2 * sin_diff_lon
            num2 = cos_lat1 * math.sin(lat2) - sin_lat1 * cos_lat2 * cos_diff_lon
            # Compute the denominator of the angular distance formula.
            denominator = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_diff_lon
            # Calculate the angular distance (in radians) using the arctangent function.
            angle = math.atan2(math.sqrt(num1 * num1 + num2 * num2), denominator)
            # Convert the angular distance from radians to degrees and store it in the output.
            out[i] = angle * (180.0 / math.pi)
    
    # Cache the compiled kernel for future calls with the same ndim.
    kernel_cache[ndim] = kernel
    return kernel

def true_angular_distance_gpu_nd(l1, b1, l2, b2,
                                 shape_l1, shape_b1, shape_l2, shape_b2,
                                 broadcast_shape):
    """
    Compute angular distances for broadcastable N-dimensional inputs on the GPU.
    
    This function launches a single CUDA kernel to process the entire broadcasted array,
    assuming the data fits in GPU memory. It handles flattening of input arrays, transferring
    necessary metadata (shapes and strides) to the device, and reshapes the output back to
    the broadcasted shape.
    
    Parameters:
      l1, b1, l2, b2 (numpy.ndarray): Flattened input arrays containing angle values in degrees.
      shape_l1, shape_b1, shape_l2, shape_b2 (tuple): Original shapes of the corresponding input arrays.
      broadcast_shape (tuple): The shape of the broadcasted output array.
    
    Returns:
      numpy.ndarray: Output array of computed angular distances (in degrees) reshaped to broadcast_shape.
    """
    # Determine the number of dimensions in the broadcasted output.
    ndim = len(broadcast_shape)
    # Compute the total number of elements in the output array.
    total = np.prod(broadcast_shape)
    # Allocate a host array to receive the output (initially one-dimensional).
    out = np.empty(total, dtype=np.float64)
    
    # Transfer the flattened input arrays from host (CPU) memory to device (GPU) memory.
    d_l1 = cuda.to_device(l1)
    d_b1 = cuda.to_device(b1)
    d_l2 = cuda.to_device(l2)
    d_b2 = cuda.to_device(b2)
    
    # Prepare the shape arrays for each input, converting them to 32-bit integers.
    np_shape_l1 = np.array(shape_l1, dtype=np.int32)
    np_shape_b1 = np.array(shape_b1, dtype=np.int32)
    np_shape_l2 = np.array(shape_l2, dtype=np.int32)
    np_shape_b2 = np.array(shape_b2, dtype=np.int32)
    # Prepare the broadcast shape array.
    np_b_shape  = np.array(broadcast_shape, dtype=np.int32)
    
    # Compute the strides for each input array to map multi-index positions.
    np_strides_l1 = np.array(compute_strides(shape_l1), dtype=np.int32)
    np_strides_b1 = np.array(compute_strides(shape_b1), dtype=np.int32)
    np_strides_l2 = np.array(compute_strides(shape_l2), dtype=np.int32)
    np_strides_b2 = np.array(compute_strides(shape_b2), dtype=np.int32)
    
    # Transfer the shape and stride arrays to the device memory.
    d_shape_l1   = cuda.to_device(np_shape_l1)
    d_shape_b1   = cuda.to_device(np_shape_b1)
    d_shape_l2   = cuda.to_device(np_shape_l2)
    d_shape_b2   = cuda.to_device(np_shape_b2)
    d_strides_l1 = cuda.to_device(np_strides_l1)
    d_strides_b1 = cuda.to_device(np_strides_b1)
    d_strides_l2 = cuda.to_device(np_strides_l2)
    d_strides_b2 = cuda.to_device(np_strides_b2)
    d_b_shape    = cuda.to_device(np_b_shape)
    
    # Retrieve the CUDA kernel specialized for the determined number of dimensions.
    kernel = get_true_angular_distance_kernel(ndim)
    # Determine the number of threads per block; here we choose 256.
    threads_per_block = 256
    # Compute the number of blocks needed to cover all elements.
    blocks = (total + threads_per_block - 1) // threads_per_block
    
    # Allocate a device output array of the appropriate size.
    # (total,) denotes a one-dimensional array with 'total' elements.
    d_out = cuda.device_array((total,), dtype=np.float64)
    
    # Launch the CUDA kernel with the computed grid and block configuration.
    kernel[blocks, threads_per_block](
        d_l1, d_b1, d_l2, d_b2,
        d_shape_l1, d_strides_l1,
        d_shape_b1, d_strides_b1,
        d_shape_l2, d_strides_l2,
        d_shape_b2, d_strides_b2,
        d_b_shape,
        total, d_out
    )
    
    # Copy the result from device memory back to host memory.
    d_out.copy_to_host(out)
    # Reshape the output array to match the broadcasted shape before returning.
    return out.reshape(broadcast_shape)

@ranged_quantity_input(l1=(None, None, u.deg),
                         b1=(None, None, u.deg),
                         l2=(None, None, u.deg),
                         b2=(None, None, u.deg),
                         strip_input_units=True,
                         allow_none=True,
                         output_unit=u.deg)
def true_angular_distance_auto(l1, b1, l2, b2, threshold=13*2334*1549, threshold_warning=500*2334*1549):
    """
    Compute the true angular distance between points from broadcastable arrays.
    
    This function automatically selects the computation method based on the size of the input:
      - If the total number of elements (n_elements) is smaller than `threshold`, it uses the CPU
        vectorized method provided by pycraf.geometry.true_angular_distance.
      - Otherwise, it uses the GPU-accelerated version (via true_angular_distance_gpu_nd).
    
    Additionally, if n_elements exceeds the `threshold_warning`, a runtime warning is issued to alert
    the user that the array size might be too large for efficient GPU execution.
    
    Parameters:
      l1, b1, l2, b2 (numpy.ndarray): Arrays containing angle values (in degrees) which must be broadcast-compatible.
      threshold (int): If the total number of elements is below this value, the CPU version is used.
      threshold_warning (int): If exceeded, a warning is raised about potential inefficiency on the GPU.
    
    Returns:
      numpy.ndarray: Angular distances (in degrees) in the shape determined by broadcasting l1, b1, l2, and b2.
    """
    # Determine the broadcasted shape from the inputs.
    broadcast_shape = np.broadcast(l1, b1, l2, b2).shape
    # Calculate the total number of elements in the broadcasted output.
    n_elements = np.prod(broadcast_shape)

    # Issue a runtime warning if the number of elements is above the warning threshold.
    if n_elements > threshold_warning:
        warnings.warn(
            "Kernel won't be able to process such big arrays on GPU "
            "Consider reducing the input data size.",
            RuntimeWarning
        )
    
    # If the total number of elements is below the lower threshold,
    # use the CPU vectorized implementation.
    if n_elements < threshold:
        from pycraf.geometry import true_angular_distance
        # Multiply each input by the unit to ensure proper unit handling.
        return true_angular_distance(l1 * u.deg, b1 * u.deg, l2 * u.deg, b2 * u.deg).value
    else:
        # Otherwise, use the GPU-accelerated version.
        # np.ravel is used to flatten the arrays.
        return true_angular_distance_gpu_nd(np.ravel(l1), np.ravel(b1),
                                             np.ravel(l2), np.ravel(b2),
                                             l1.shape, b1.shape, l2.shape, b2.shape,
                                             broadcast_shape)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

def _extract_llh(obs_array):
    """
    Parameters
    ----------
    obs_array : 1D numpy array of ``cysgp4.PyObserver`` objects

    Returns
    -------
    lat_rad, lon_rad, h_m : float32
        * lat_rad … geodetic **latitude**  (radians, north positive)
        * lon_rad … geodetic **longitude** (radians, east positive)
        * h_m     … **height** above WGS-84 ellipsoid (metres)
    """
    n = obs_array.size
    lat = np.empty(n, dtype=np.float32)
    lon = np.empty_like(lat)
    h   = np.empty_like(lat)

    for k, obs in enumerate(obs_array):
        loc = obs.loc                    # PyCoordGeodetic
        lat[k] = np.deg2rad(loc.lat).astype(np.float32)
        lon[k] = np.deg2rad(loc.lon).astype(np.float32)
        h[k]   = np.array(loc.alt * 1_000).astype(np.float32)  # km → m

    return lat, lon, h

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CUDA kernel:  observer frame                                             │
# ╰──────────────────────────────────────────────────────────────────────────╯
@cuda.jit
def _topo_kernel(r_sat_flat,              # (TS,3)  float32   ECI XYZ
                 lat, lon, h,             # (Obs,)  float32   observer LLH
                 gmst,                    # (T,)    float32   GMST [rad]
                 az_out, el_out, rng_out, # (T,Obs,S) float32
                 T, S):
    """
    Each CUDA thread handles ONE triple (time t, satellite s, observer o).

    We launch a 2-D grid:

        x-axis :  T*S   (flattened as ``ts_idx``)
        y-axis :  Obs   (index ``o``)
    """
    ts_idx, o = cuda.grid(2)
    if ts_idx >= r_sat_flat.shape[0] or o >= lat.size:
        return

    # ------------------ unravel flattened (time, sat) index ------------
    t = ts_idx // S
    s = ts_idx - t * S                   # same as ts_idx % S but cheaper

    # ------------------ observer ECEF (WGS-84) -------------------------
    lat_o = lat[o]
    lon_o = lon[o]
    h_o   = h[o]

    sin_lat = math.sin(lat_o)
    cos_lat = math.cos(lat_o)
    sin_lon = math.sin(lon_o)
    cos_lon = math.cos(lon_o)

    # Ellipsoid constants
    a  = 6_378.137                     # semi-major axis [m]
    f  = 1.0 / 298.257223563
    N  = a / math.sqrt(1.0 - (2*f - f*f) * sin_lat * sin_lat)

    x_obs = (N + h_o) * cos_lat * cos_lon
    y_obs = (N + h_o) * cos_lat * sin_lon
    z_obs = (N * (1 - (2*f - f*f)) + h_o) * sin_lat

    # ------------------ rotate satellite: ECI → ECEF -------------------
    cg = math.cos(gmst[t])
    sg = math.sin(gmst[t])

    x_eci = r_sat_flat[ts_idx, 0]
    y_eci = r_sat_flat[ts_idx, 1]
    z_eci = r_sat_flat[ts_idx, 2]

    x_ecef =  cg * x_eci + sg * y_eci
    y_ecef = -sg * x_eci + cg * y_eci
    z_ecef =  z_eci

    # ------------------ line-of-sight vector in ENU --------------------
    dx = x_ecef - x_obs
    dy = y_ecef - y_obs
    dz = z_ecef - z_obs

    east  = -sin_lon * dx +  cos_lon * dy
    north = (-sin_lat * cos_lon * dx
             - sin_lat * sin_lon * dy
             +  cos_lat * dz)
    up    = ( cos_lat * cos_lon * dx
             + cos_lat * sin_lon * dy
             + sin_lat * dz)

    rng = math.sqrt(east*east + north*north + up*up)
    az  = math.atan2(east, north)
    if az < 0.0:                         # wrap → [0, 2π)
        az += 2.0 * math.pi
    el  = math.asin(up / rng)

    az_out[t, o, s]  = az
    el_out[t, o, s]  = el
    rng_out[t, o, s] = rng

def gpu_topo(eci_sat, lat_rad, lon_rad, h_m, gmst_rad, *, blk=(128, 4)):
    """
    Compute **az [deg]**, **el [deg]**, **range [m]** for every
    (time-step T, observer Obs, satellite S) tuple.

    All arrays must be ``float32``   (``eci_sat`` is cast if needed).

    Returns
    -------
    (az, el, rng) with shape **(T, Obs, S)**  - already on the host.
    """
    # ---------------- dimensions ---------------------------------------
    T, S, _ = eci_sat.shape
    Obs     = lat_rad.size
    TS      = T * S                       # flattened first axis

    # ---------------- copy constants → GPU -----------------------------
    d_rsat = cuda.to_device(eci_sat.astype(np.float32).reshape(TS, 3))
    d_lat  = cuda.to_device(lat_rad)
    d_lon  = cuda.to_device(lon_rad)
    d_h    = cuda.to_device(h_m)
    d_gmst = cuda.to_device(gmst_rad)

    # ---------------- output buffers -----------------------------------
    az_dev  = cuda.device_array((T, Obs, S), dtype=np.float32)
    el_dev  = cuda.device_array_like(az_dev)
    rng_dev = cuda.device_array_like(az_dev)

    # ---------------- kernel launch config -----------------------------
    grd = ((TS  + blk[0] - 1)//blk[0],
           (Obs + blk[1] - 1)//blk[1])

    _topo_kernel[grd, blk](d_rsat, d_lat, d_lon, d_h, d_gmst,
                           az_dev, el_dev, rng_dev,
                           np.int32(T), np.int32(S))
    cuda.synchronize()

    return (
        np.rad2deg(az_dev.copy_to_host()),
        np.rad2deg(el_dev.copy_to_host()),
        rng_dev.copy_to_host(),
    )

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CUDA kernel:   satellite frame                                           │
# ╰──────────────────────────────────────────────────────────────────────────╯
@cuda.jit
def _sat_azel_kernel(r_sat_flat, v_sat_flat,   # (TS,3)  ECI position / velocity
                     lat, lon, h,              # (Obs,)   observer LLH (rad,m)
                     gmst,                     # (T,)     GMST per epoch (rad)
                     az_out, el_out, rng_out,  # (T, Obs, S)  results
                     T, S, mode):
    """
    Parameters (device arrays, **float32** unless noted)
    ---------------------------------------------------
    r_sat_flat  : ECI Cartesian **position** (x,y,z)  [km]   (TS,3)
    v_sat_flat  : ECI Cartesian **velocity** (vx,vy,vz)[m/s]     (TS,3)
    lat,lon,h   : observer geodetic latitude [radians],
                  longitude [radians, east positive],
                  height [km]                            (Obs,)
    gmst        : Greenwich Mean Sidereal Time per epoch [radians] (T,)
    az_out      : resulting azimuth  [deg]                (T,Obs,S)
    el_out      : resulting elevation[deg]                (T,Obs,S)
    rng_out     : resulting slant range [km]          (T,Obs,S)
    T, S        : int32 scalars (#epochs, #satellites)
    mode        : 0 ⇒ sat-frame “zxy”  ( z ‖ velocity )
                  1 ⇒ sat-frame “xyz”  ( x ‖ velocity )

    Thread layout
    -------------
      grid.x * blockDim.x   spans   TS  (= T * S)        → each thread picks
      grid.y * blockDim.y   spans   Obs                  → one  (t,s,o) triple
    """
    # ------------------------------------------------------------------ #
    # ➊  translate thread‑ID ➜ (t,s,o) triple
    # ------------------------------------------------------------------ #
    ts_idx, o = cuda.grid(2)
    if ts_idx >= r_sat_flat.shape[0] or o >= lat.size:
        return

    t  = ts_idx // S                 # epoch index   (0 … T‑1)
    s  = ts_idx - t*S                # satellite idx (0 … S‑1)

    # ------------------------------------------------------------------ #
    # ➋  observer ECEF coordinates  (WGS‑84 ellipsoid)
    # ------------------------------------------------------------------ #
    lat_o = lat[o]
    lon_o = lon[o]
    h_o   = h[o]

    sin_lat = math.sin(lat_o)
    cos_lat = math.cos(lat_o)
    sin_lon = math.sin(lon_o)
    cos_lon = math.cos(lon_o)

    a  = 6_378.137                     # semi‑major axis
    f  = 1.0/298.257223563               # flattening
    N  = a / math.sqrt(1.0 - (2*f - f*f) * sin_lat*sin_lat)

    x_obs = (N + h_o) * cos_lat * cos_lon
    y_obs = (N + h_o) * cos_lat * sin_lon
    z_obs = (N * (1 - (2*f - f*f)) + h_o) * sin_lat

    # ------------------------------------------------------------------ #
    # ➌  rotate observer ECEF ➜ ECI by  –GMST(t)
    # ------------------------------------------------------------------ #
    cg = math.cos(gmst[t])
    sg = math.sin(gmst[t])

    x_obs_eci =  cg*x_obs - sg*y_obs
    y_obs_eci =  sg*x_obs + cg*y_obs
    z_obs_eci =  z_obs                      # z unchanged

    # ------------------------------------------------------------------ #
    # ➍  satellite ECI position & velocity
    # ------------------------------------------------------------------ #
    x_sat = r_sat_flat[ts_idx, 0]
    y_sat = r_sat_flat[ts_idx, 1]
    z_sat = r_sat_flat[ts_idx, 2]

    vx_sat = v_sat_flat[ts_idx, 0]
    vy_sat = v_sat_flat[ts_idx, 1]
    vz_sat = v_sat_flat[ts_idx, 2]

    # normalise velocity → ê_v  (unit vector of motion)
    v_norm = math.sqrt(vx_sat*vx_sat + vy_sat*vy_sat + vz_sat*vz_sat)
    if v_norm == 0.0:                      # should never happen
        return
    evx, evy, evz = vx_sat/v_norm, vy_sat/v_norm, vz_sat/v_norm

    # unit vector pointing roughly “nadir” (towards Earth centre)
    r_norm = math.sqrt(x_sat*x_sat + y_sat*y_sat + z_sat*z_sat)
    erx, ery, erz = -x_sat/r_norm, -y_sat/r_norm, -z_sat/r_norm

    # build right‑handed orthonormal basis  {ê_x, ê_y, ê_z}
    if mode == 0:                          # ---- sat_frame 'zxy' ----
        ezx, ezy, ezz = evx, evy, evz      # ê_z = velocity
        exx, exy, exz = erx, ery, erz      # provisional ê_x ≈ nadir
        # ê_y = norm( ê_z × ê_x )
        eyx =  ezy*exz - ezz*exy
        eyy =  ezz*exx - ezx*exz
        eyz =  ezx*exy - ezy*exx
        ey_norm = math.sqrt(eyx*eyx + eyy*eyy + eyz*eyz)
        eyx /= ey_norm; eyy /= ey_norm; eyz /= ey_norm
        # re‑orthogonalise  ê_x = ê_y × ê_z
        exx =  eyy*ezz - eyz*ezy
        exy =  eyz*ezx - eyx*ezz
        exz =  eyx*ezy - eyy*ezx
    else:                                  # ---- sat_frame 'xyz' ----
        exx, exy, exz = evx, evy, evz      # ê_x = velocity
        ezx, ezy, ezz = erx, ery, erz      # provisional ê_z ≈ nadir
        # ê_y = norm( ê_z × ê_x )
        eyx =  ezy*exz - ezz*exy
        eyy =  ezz*exx - ezx*exz
        eyz =  ezx*exy - ezy*exx
        ey_norm = math.sqrt(eyx*eyx + eyy*eyy + eyz*eyz)
        eyx /= ey_norm; eyy /= ey_norm; eyz /= ey_norm
        # re‑orthogonalise  ê_z = ê_x × ê_y
        ezx =  exy*eyz - exz*eyy
        ezy =  exz*eyx - exx*eyz
        ezz =  exx*eyy - exy*eyx

    # ------------------------------------------------------------------ #
    # ➎  line‑of‑sight vector in satellite frame
    # ------------------------------------------------------------------ #
    dx = x_obs_eci - x_sat
    dy = y_obs_eci - y_sat
    dz = z_obs_eci - z_sat

    # components of LOS in satellite basis
    bx = dx*exx + dy*exy + dz*exz
    by = dx*eyx + dy*eyy + dz*eyz
    bz = dx*ezx + dy*ezy + dz*ezz

    rng = math.sqrt(bx*bx + by*by + bz*bz)
    az  = math.atan2(by, bx)                     # radians (0…2π)
    if az < 0.0:
        az += 2.0 * math.pi
    # elevation definition depends on sat_frame
    el  = math.asin(bz / rng) if mode == 0 else math.acos(bz / rng)

    # ------------------------------------------------------------------ #
    # ➏  write results (convert angles → degrees)
    # ------------------------------------------------------------------ #
    az_out[t, o, s]  = az
    el_out[t, o, s]  = el
    rng_out[t, o, s] = rng

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ GPU driver – small‑footprint wrapper around the kernel                   │
# ╰──────────────────────────────────────────────────────────────────────────╯
def gpu_sat_azel(eci_pos, eci_vel,
                      lat_rad, lon_rad, h_m,
                      gmst_rad, *,
                      sat_frame="zxy",
                      blk=(128, 4)):
    """
    GPU version of “satellite-centric az/el”.

    The driver launches `_sat_azel_kernel` twice-nested:
    * x-dimension covers **time * satellite** (`TS`)
    * y-dimension covers **observer / cell** (`Obs`)

    Only *three* device arrays are returned (az, el, rng) shaped exactly
    like the CPU result from ``cysgp4.propagate_many(..., do_sat_azel=True)``.

    Notes
    -----
    * All inputs must be **float32** except `blk`.
    * Output angles are converted back to **degrees**.
    """
    if sat_frame not in ("zxy", "xyz"):
        raise ValueError("sat_frame must be 'zxy' or 'xyz'")

    mode = 0 if sat_frame == "zxy" else 1

    # ---------- dimensions ------------------------------------------------
    T, S, _ = eci_pos.shape
    Obs     = lat_rad.size
    TS      = T * S

    # ---------- push constant data to device ------------------------------
    d_rpos = cuda.to_device(eci_pos.reshape(TS, 3).astype(np.float32))
    d_rvel = cuda.to_device(eci_vel.reshape(TS, 3).astype(np.float32))
    d_lat  = cuda.to_device(lat_rad.astype(np.float32))
    d_lon  = cuda.to_device(lon_rad.astype(np.float32))
    d_h    = cuda.to_device(h_m.astype(np.float32))
    d_gmst = cuda.to_device(gmst_rad.astype(np.float32))

    # ---------- allocate outputs ------------------------------------------
    az_dev  = cuda.device_array((T, Obs, S), dtype=np.float32)
    el_dev  = cuda.device_array_like(az_dev)
    rng_dev = cuda.device_array_like(az_dev)

    # ---------- kernel launch geometry ------------------------------------
    grd = ((TS  + blk[0]-1)//blk[0],
           (Obs + blk[1]-1)//blk[1])

    _sat_azel_kernel[grd, blk](d_rpos, d_rvel,
                               d_lat, d_lon, d_h,
                               d_gmst,
                               az_dev, el_dev, rng_dev,
                               np.int32(T), np.int32(S), np.int32(mode))
    cuda.synchronize()

    # host copies  (angles to deg)
    return (np.rad2deg(az_dev.copy_to_host()),
            np.rad2deg(el_dev.copy_to_host()),
            np.rad2deg(rng_dev.copy_to_host()))
    
# ╭──────────────────────────────────────────────────────────────────────────╮
# │ High‑level helper – one‑shot “propagate & transform” on GPU             │
# ╰──────────────────────────────────────────────────────────────────────────╯
def propagate_many_gpu(mjds,
                       tles,
                       observers,
                       *,
                       sat_frame="xyz",
                       block_size=(128, 4),
                       **ignored_flags):
    """
    GPU-accelerated alternative to :pyfunc:`cysgp4.propagate_many`
    that returns **both**

      * ``topo``     - (T, Obs, S, 4)  az / el / range / NaN
      * ``sat_azel`` - (T, Obs, S, 3)  range / az / el (sat-centric)

    **Strategy**

    1.  Call the CPU version *once* (for the first observer only) to get
        reliable satellite **ECI position & velocity** - that step is
        already highly vectorised in Cython and very fast.
    2.  Feed those arrays into two tiny GPU helpers
        (`gpu_topo_safe`, `gpu_sat_azel_safe`) that fan-out results to all
        remaining observers / Earth-grid cells.
    3.  Assemble the two result cubes and return them in a dictionary.

    Parameters
    ----------
    mjds        : array-like, float64
        Modified Julian Dates (can be broadcast).
    tles        : array-like, PyTle
        Satellite TLE objects (broadcast).
    observers   : array-like, PyObserver
        Observer / cell objects (broadcast).
    sat_frame   : 'xyz' or 'zxy', optional
        Defines the satellite body frame (see cysgp4 docs).
    block_size  : 2-tuple, optional
        CUDA block geometry - tune if you like.  Default ``(128,4)`` is
        safe on sm_70+ GPUs.

    Returns
    -------
    dict  with keys  ``'topo'``  and  ``'sat_azel'``

    Examples
    --------
    ```python
    g = propagate_many_gpu(mjds_new, tles_new, observers_new,
                           sat_frame="xyz")
    topo_cube     = g['topo']      # (time, obs, sat, 4)
    cell_in_sats  = g['sat_azel']  # ditto, but sat-centric
    ```
    """
    # --- 1 ↠ broadcast inputs ( NumPy handles the heavy lifting ) -------
    mjd_b, tle_b, obs_b = np.broadcast_arrays(
        np.asarray(mjds, dtype=np.float64),
        np.asarray(tles, dtype=object),
        np.asarray(observers, dtype=object),
    )
    T, Obs, S = mjd_b.shape

    # --- 2 ↠ call CPU version just for observer[0] ----------------------
    cpu = cysgp4.propagate_many(
        mjd_b[:,0,:],
        tle_b[:,0,:],
        obs_b[:,0,:],                 # only RAS station
        do_eci_pos=True, do_eci_vel=True,
        do_geo=False, do_topo=False,
        do_obs_pos=False, do_sat_azel=False,
        sat_frame=sat_frame,
    )
    eci_pos = cpu['eci_pos'].astype(np.float32)  # (T,S,3)
    eci_vel = cpu['eci_vel'].astype(np.float32)  # (T,S,3)

    # --- 3 ↠ observer geodetic arrays ----------------------------------
    lat_rad, lon_rad, h_m = _extract_llh(obs_b.ravel())
    lat_rad = lat_rad.astype(np.float32)
    lon_rad = lon_rad.astype(np.float32)
    h_m     = h_m.astype(np.float32)

    # --- 4 ↠ GMST [rad] per epoch --------------------------------------
    gmst = time.Time(mjd_b[:, 0, 0], format='mjd').sidereal_time(
        'mean', 'greenwich'
    ).to(u.rad).value.astype(np.float32)

    # --- 5 ↠ GPU transforms --------------------------------------------
    az, el, rng = gpu_topo(eci_pos, lat_rad, lon_rad, h_m,
                                gmst, blk=block_size)
    # saz_az, saz_el, saz_rng = gpu_sat_azel(
    #     eci_pos, eci_vel,
    #     lat_rad, lon_rad, h_m,
    #     gmst,
    #     sat_frame=sat_frame,
    #     blk=block_size,
    # )

    # --- 6 ↠ pack results like cysgp4.propagate_many -------------------
    topo = np.empty((T, Obs, S, 4), dtype=np.float32)
    topo[..., 0] = az        # az (deg)
    topo[..., 1] = el        # el (deg)
    topo[..., 2] = rng       # range (m)
    topo[..., 3] = np.nan    # distance_rate not available – fill with NaN

    sat_azel = np.empty((T, Obs, S, 3), dtype=np.float32)
    # sat_azel[..., 0] = saz_az
    # sat_azel[..., 1] = saz_el
    # sat_azel[..., 2] = saz_rng

    return {"topo": topo, "sat_azel": sat_azel}

