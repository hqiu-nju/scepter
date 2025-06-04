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
    
Author: boris.sorokin <mralin@protonmail.com>
Date: 16-04-2025
"""

import math
import numpy as np
from numba import cuda, int32
from astropy import units as u
from pycraf.utils import ranged_quantity_input
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

###############################
# Benchmark Section
# This block is only executed when running the module as a standalone program.
# It benchmarks the GPU/auto implementation against the pure CPU version.
###############################
if __name__ == '__main__':
    import time
    # Define dimensions and number of iterations for the benchmark.
    Times, SkyCell, EarthCell, Satellite = 15, 2334, 1549, 1  
    # Generate random input data for l1 and b1 with appropriate shapes and units.
    l1 = np.random.uniform(0, 360, size=(Times, SkyCell, 1, 1)) * u.deg
    b1 = np.random.uniform(-90, 90, size=(Times, SkyCell, 1, 1)) * u.deg
    # Generate random input data for l2 and b2 with a different shape to ensure broadcasting.
    l2 = np.random.uniform(0, 360, size=(Times, 1, EarthCell, Satellite)) * u.deg
    b2 = np.random.uniform(-90, 90, size=(Times, 1, EarthCell, Satellite)) * u.deg
    
    print("Benchmarking true_angular_distance_auto:")
    # Time the GPU/auto version.
    start = time.time()
    out_auto = true_angular_distance_auto(l1, b1, l2, b2)
    gpu_time = time.time() - start
    print(f"  GPU/auto version time: {gpu_time:.4f} seconds")
    
    # Time the pure CPU version.
    start_cpu = time.time()
    from pycraf.geometry import true_angular_distance
    out_cpu = true_angular_distance(l1, b1, l2, b2)
    cpu_time = time.time() - start_cpu
    print(f"  Pure CPU version time: {cpu_time:.4f} seconds")
    # Compare the results from both implementations.
    diff = np.abs(out_auto - out_cpu).mean()
    print(f"  Mean absolute difference: {diff:.6f}")
    if gpu_time > 0:
        print(f"  Speedup factor: {cpu_time / gpu_time:.2f}")
