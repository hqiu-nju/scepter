"""
scenario.py

This module is providing supplemental function to enhance the simulation.

Author: boris.sorokin <mralin@protonmail.com>
Date of creation: 16-04-2025
Latest amend date: 11-06-2025; Added logic to read stored data by keyword
"""
import os
import h5py
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from pycraf import conversions as cnv
import signal
from contextlib import contextmanager

# Global variable for the current thread count (used by simulation)
current_thread_count = 8

def set_num_threads(n):
    """
    Clamp n to [1, 32] and propagate that setting to:
      - OMP_NUM_THREADS
      - OPENBLAS_NUM_THREADS
      - MKL_NUM_THREADS
      - NUMEXPR_NUM_THREADS
      - threadpoolctl (if installed)
      - numba (if installed)
      - cysgp4 (if installed)

    Updates the module‐level current_thread_count to the clamped value.
    """
    global current_thread_count

    # 1. Clamp to [1, 32]
    n_clamped = max(1, min(int(n), 32))

    # 2. Set environment variables for BLAS/OpenMP backends
    os.environ["OMP_NUM_THREADS"] = str(n_clamped)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_clamped)
    os.environ["MKL_NUM_THREADS"] = str(n_clamped)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_clamped)

    # 3. Set Numba's internal thread count, if Numba is installed
    try:
        import numba
        numba.set_num_threads(n_clamped)
    except ImportError:
        pass

    # 4. Set cysgp4 thread count, if cysgp4 is installed
    try:
        import cysgp4
        cysgp4.set_num_threads(n_clamped)
    except ImportError:
        pass

    # 5. Update the module‐level variable
    current_thread_count = n_clamped

def generate_simulation_batches(start_time, end_time, timestep, batch_size):
    """
    Generate batches of simulation times between start_time and end_time.

    Each batch contains up to 'batch_size' timesteps. The function returns a dictionary 
    with keys corresponding to each batch parameter. Specifically, it returns a dictionary
    with:
      - 'batch_start': list of batch start times (astropy Time objects),
      - 'times': list of astropy Time arrays for each batch,
      - 'td': list of TimeDelta arrays for each batch,
      - 'batch_end': list of batch end times (astropy Time objects).
    
    Parameters
    ----------
    start_time : astropy.time.Time
        Global simulation start time.
    end_time : astropy.time.Time
        Global simulation end time.
    timestep : float
        Time step in seconds.
    batch_size : int
        Maximum number of timesteps per batch.

    Returns
    -------
    batches : dict
        A dictionary containing lists for each batch parameter.
    """
    total_duration_sec = (end_time - start_time).sec
    total_steps = int(np.ceil(total_duration_sec / timestep))+1
    
    # Prepare lists to store batch parameters.
    batch_start_list = []
    batch_times_list = []
    batch_td_list = []
    batch_end_list = []
    
    for batch_start_idx in range(0, total_steps, batch_size):
        # Determine batch indices.
        batch_end_idx = min(batch_start_idx + batch_size, total_steps)
        n_steps_in_batch = batch_end_idx - batch_start_idx
        
        # Create the time delta array for the batch.
        batch_td_array = np.arange(0, n_steps_in_batch * timestep, timestep)
        
        # Compute the batch's start time.
        batch_start_time = start_time + TimeDelta(batch_start_idx * timestep, format='sec')
        # Compute the simulation times for the batch.
        batch_times = batch_start_time + TimeDelta(batch_td_array, format='sec')
        # The batch end time is the last time in the batch.
        batch_end_time = batch_times[-1]
        
        # Append results to lists.
        batch_start_list.append(batch_start_time)
        batch_times_list.append(batch_times)
        batch_td_list.append(TimeDelta(batch_td_array, format='sec'))
        batch_end_list.append(batch_end_time)
    
    # Return a dictionary containing all batch information.
    batches = {
        'batch_start': batch_start_list,
        'times': batch_times_list,
        'td': batch_td_list,
        'batch_end': batch_end_list
    }
    
    return batches

@contextmanager
def block_interrupts():
    """
    A context manager to temporarily ignore SIGINT (KeyboardInterrupt).
    This ensures that the enclosed critical section (e.g., file I/O) is not interrupted
    by Ctrl+C. Once the block finishes (or if it errors), the original handler is restored.
    """
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore future SIGINT
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)  # Restore original handler

def init_simulation_results(filename):
    """
    Delete the HDF5 file that stores simulation results if it exists.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function deletes the file if it exists. It can be called at the start
    of the simulation to refresh stored results. It's a fancy wrapper for os.remove.
    """
    if os.path.exists(filename):
        os.remove(filename)

def store_simulation_results(
    filename="simulation_results.h5",
    compression="gzip",
    compression_opts=9,
    **datasets
):
    """
    Store or append named arrays, quantities, or Time objects into an HDF5 file.

    Each keyword argument corresponds to a dataset name and its associated data:
      - astropy.Time → will be converted to MJD (in days) and stored with unit 'd'.
      - astropy.Quantity → unit is stripped, data saved as raw array; unit is recorded in attrs['unit'].
      - array-like without .unit → saved as a NumPy array, unitless.

    If a dataset already exists, it will be extended along the first axis:
      • If the existing dataset has a 'unit' attribute and the new data also have one,
        an attempt to convert the new data to the existing unit is made.
      • If a unitless dataset exists but new data have a unit (or vice versa), a ValueError is raised.
      • Otherwise, rows are simply appended.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file in which data will be stored or appended.
    compression : str or None, optional
        Compression filter to be applied when creating a new dataset. Default is "gzip".
    compression_opts : int or None, optional
        Compression level for gzip. Default is 9.
    **datasets : {str: array-like or astropy.Quantity or astropy.Time}
        Arbitrary keyword arguments; each key is the dataset name, each value is the data to store.
        Each value must be indexable along a “row” or “time” axis (i.e., shape (N, ...) for some N).

    Raises
    ------
    ValueError
        - If attempting to mix unitless data with unit-bearing data in the same dataset.
        - If unit conversion between existing and new data is not possible.
    """
    with block_interrupts():
        with h5py.File(filename, "a") as f:
            for name, value in datasets.items():

                # ———— Special handling for astropy.time.Time ————
                if isinstance(value, Time):
                    # Convert to MJD (float) and record unit as days
                    incoming_unit = u.day
                    incoming_array = np.asarray(value.mjd)
                else:
                    # If it has a .unit attribute (i.e. it's a Quantity), strip unit and get raw array
                    if hasattr(value, "unit"):
                        incoming_unit = value.unit
                        incoming_array = np.asarray(value.value)
                    else:
                        incoming_unit = None
                        incoming_array = np.asarray(value)

                # Ensure we have a 2D “rows first” shape for appending: (N_new, ...)
                incoming_array = np.reshape(incoming_array, (-1,) + incoming_array.shape[1:])
                n_new = incoming_array.shape[0]

                if name in f:
                    # ———— Appending to an existing dataset ————
                    dset = f[name]
                    old_len = dset.shape[0]

                    # Check if the existing dataset has a unit
                    existing_unit = None
                    if "unit" in dset.attrs:
                        existing_unit = u.Unit(dset.attrs["unit"])

                    # Case A: existing dataset has a unit
                    if existing_unit is not None:
                        if incoming_unit is None:
                            raise ValueError(
                                f"Cannot append unitless data to dataset '{name}' which has unit '{existing_unit}'."
                            )
                        try:
                            # Convert new data to the existing unit
                            converted = (incoming_array * incoming_unit).to(existing_unit).value
                        except u.UnitConversionError:
                            raise ValueError(
                                f"New data unit '{incoming_unit}' is not convertible to existing unit '{existing_unit}' "
                                f"for dataset '{name}'."
                            )
                        array_to_store = converted
                        unit_to_store = str(existing_unit)

                    # Case B: existing dataset is unitless
                    else:
                        if incoming_unit is not None:
                            raise ValueError(
                                f"Cannot append data with unit '{incoming_unit}' to unitless dataset '{name}'."
                            )
                        array_to_store = incoming_array
                        unit_to_store = None

                    # Resize and append
                    new_len = old_len + n_new
                    dset.resize((new_len,) + dset.shape[1:])
                    dset[old_len:new_len, ...] = array_to_store

                    # Preserve or set unit attribute
                    if unit_to_store is not None:
                        dset.attrs["unit"] = unit_to_store

                else:
                    # ———— Creating a new dataset ————
                    if incoming_unit is not None:
                        unit_to_store = str(incoming_unit)
                        array_to_store = incoming_array
                    else:
                        unit_to_store = None
                        array_to_store = incoming_array

                    maxshape = (None,) + array_to_store.shape[1:]
                    dset = f.create_dataset(
                        name,
                        data=array_to_store,
                        maxshape=maxshape,
                        chunks=True,
                        compression=compression,
                        compression_opts=compression_opts,
                        dtype=array_to_store.dtype,
                    )
                    if unit_to_store is not None:
                        dset.attrs["unit"] = unit_to_store

def read_simulation_results_keywords(
    filename: str = "simulation_results.h5",
    *,
    with_info: bool = False
):
    """
    Inspect an HDF5 results file **without loading the data arrays**.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file produced by ``store_simulation_results``.
    with_info : bool, optional
        *False* (default)  → return just a list of dataset names.  
        *True*             → return a dict {name: {'shape', 'dtype', 'unit'}}.

    Returns
    -------
    list[str]  *or*  dict
        • If *with_info=False*: list of dataset names.  
        • If *with_info=True* : mapping with cheap metadata
          (array *shape* & *dtype* are header attributes, so no data load;
          *unit* comes from the dataset's ``attrs`` if present).
    """
    result = [] if not with_info else {}
    with h5py.File(filename, "r") as f:
        for name, dset in f.items():
            if with_info:
                result[name] = dict(
                    shape=dset.shape,
                    dtype=str(dset.dtype),
                    unit=dset.attrs.get("unit", None),
                )
            else:
                result.append(name)
    return result

def read_simulation_results(
    filename: str = "simulation_results.h5",
    *,
    keywords: list[str] | set[str] | None = None
):
    """
    Read selected datasets from an HDF5 results file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    keywords : list / set of str, optional
        Names of the datasets to load.  
        • *None* (default) → load **all** datasets.  
        • Otherwise       → load only those present in *keywords*;
          silently ignore names that are not found.

    Returns
    -------
    results : dict
        Mapping name → data.  Each value is an ``astropy.Quantity`` when a
        ``unit`` attribute is present, otherwise a NumPy array.
    """
    want_all = keywords is None
    if not want_all:
        keywords = set(keywords)

    results = {}
    with h5py.File(filename, "r") as f:
        for name, dset in f.items():
            if not want_all and name not in keywords:
                continue  # skip unrequested dataset, no I/O incurred

            raw = dset[()]                   # load the (selected) data
            if "unit" in dset.attrs:
                unit_str = dset.attrs["unit"]
                results[name] = raw * u.Unit(unit_str)
            else:
                results[name] = raw
    return results