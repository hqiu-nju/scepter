"""
scenario.py

This module is providing supplemental function to enhance the simulation.

Author: boris.sorokin <mralin@protonmail.com>
Date of creation: 16-04-2025
Latest amend date: 11-06-2025; Added logic to read stored data by keyword
"""

from __future__ import annotations
import os
import h5py
import numpy as np
from typing import Any, Dict, List, Tuple, Iterable
from contextlib import nullcontext, contextmanager
from astropy import units as u
from astropy.time import Time, TimeDelta
from pycraf import conversions as cnv
import signal

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

    Updates the module-level current_thread_count to the clamped value.
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




# -----------------------------------------------------------------------------
# Small internal helpers
# -----------------------------------------------------------------------------

def _interrupt_ctx():
    """
    Use your project's block_interrupts() if present; otherwise no-op.
    """
    ctx = globals().get("block_interrupts", None)
    return ctx() if callable(ctx) else nullcontext()


def _is_time_obj(x: Any) -> bool:
    return (Time is not None) and isinstance(x, Time)


def _to_array_unit_kind(value: Any) -> Tuple[np.ndarray, Any | None, str | None]:
    """
    Normalise a supported object into (array, unit, kind).

    kind:
        - "time/mjd" for astropy.Time
        - "quantity" for astropy.Quantity
        - None for plain arrays
    """
    if _is_time_obj(value):
        arr = np.asarray(value.mjd)               # float days
        unit = u.day if u is not None else "d"    # keep unit as days
        kind = "time/mjd"
        return arr, unit, kind

    if hasattr(value, "unit"):                    # Quantity-like
        arr = np.asarray(value.value)
        unit = getattr(value, "unit")
        return arr, unit, "quantity"

    # Plain array-like
    return np.asarray(value), None, None


def _ensure_row_first(arr: np.ndarray) -> np.ndarray:
    """
    For streaming appends we require shape (N, ...). If user passes a scalar
    or a 1D vector, reshape accordingly so that we can extend on axis 0.
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)             # (1,)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)         # (N,1) so appending works consistently
    return arr.reshape((-1,) + arr.shape[1:])


def _to_unit_str(unit: Any | None) -> str | None:
    if unit is None:
        return None
    if u is not None and isinstance(unit, u.UnitBase):
        return str(unit)
    return str(unit)


def _units_convertible(u_from: Any, u_to: Any) -> bool:
    if u is None:
        return False
    try:
        _ = (1 * u_from).to(u_to)
        return True
    except Exception:
        return False


def _convert_array_unit(arr: np.ndarray, u_from: Any, u_to: Any) -> np.ndarray:
    if u is None or (u_from is None) or (u_to is None) or str(u_from) == str(u_to):
        return arr
    return (arr * u_from).to(u_to).value  # type: ignore


def _read_dataset(ds: h5py.Dataset, *, times_as: str = "time"):
    """
    Load a dataset back into numpy/Quantity/Time depending on attrs.
    """
    raw = ds[()]
    unit_attr = ds.attrs.get("unit", None)
    kind = ds.attrs.get("kind", None)

    # Restore Time if requested and it's a time dataset
    if kind == "time/mjd" and times_as == "time" and Time is not None and u is not None:
        return Time(raw, format="mjd")

    # Restore Quantity if unit is present
    if unit_attr is not None and u is not None:
        try:
            unit = u.Unit(unit_attr)
            return raw * unit
        except Exception:
            pass  # fall through to plain ndarray

    return raw


def _iter_group_name(k: int) -> str:
    return f"iter_{k:05d}"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def write_data(
    filename: str,
    *,
    # Store constants once (e.g., grid_info, config blobs, seeds)
    constants: Dict[str, Any] | None = None,
    # Stream a batch for a given iteration (append along axis 0)
    iteration: int | None = None,
    # Optional HDF5/file attrs to write/update at root
    attrs: Dict[str, Any] | None = None,
    # Compression for NEW datasets only
    compression: str | None = "gzip",
    compression_opts: int | None = 9,
    # Behaviour flags
    overwrite_constants: bool = False,          # if const exists: False→verify equal, True→replace
    allow_unit_auto_convert: bool = True,       # when appending, auto-convert if units are compatible
    # Streaming payload when iteration is provided
    **batch: Any,
):
    """
    Universal writer.

    Use cases:
      1) Store constants once:
            write_data(fn, constants={'grid_info': grid, 'cfg': cfg_dict})
         Constants are stored under /const/<name> and never extended.

      2) Stream a batch for an iteration:
            write_data(fn, iteration=ii, times=..., epfd_0=..., power=...)
         Data live under /iter/iter_00000/<var> and are extendable on axis 0.

      3) You can do both in one call: pass constants=... and iteration=... together.

    Supported value types:
      - astropy.Time          → stored as float MJD with attrs: kind="time/mjd", unit="d"
      - astropy.Quantity      → raw array stored, attrs: unit="..."; unit is preserved
      - plain array-likes     → raw numpy arrays, unitless

    Unit safety on appends:
      - existing has unit, new is unitless      → error
      - existing unitless, new has unit         → error
      - both have units but different:
            * if compatible and allow_unit_auto_convert=True → new data are converted to existing unit
            * otherwise → error
    """
    with _interrupt_ctx(), h5py.File(filename, "a") as f:
        # --- file-level attrs (cheap config/logging) ---
        if attrs:
            for k, v in attrs.items():
                try:
                    f.attrs[k] = v
                except TypeError:
                    # HDF5 attrs must be basic types; store repr as a fallback
                    f.attrs[k] = repr(v)

        # --- constants ---
        if constants:
            g_const = f.require_group("const")
            for name, value in constants.items():
                arr, unit, kind = _to_array_unit_kind(value)
                unit_str = _to_unit_str(unit)

                if name in g_const:
                    ds = g_const[name]
                    if overwrite_constants:
                        del g_const[name]
                    else:
                        # Verify identical content and metadata
                        same_shape = tuple(ds.shape) == tuple(arr.shape)
                        same_unit  = (ds.attrs.get("unit", None) == unit_str)
                        same_kind  = (ds.attrs.get("kind", None) == kind)
                        if same_shape and same_unit and same_kind:
                            # shallow equality check if small; otherwise trust metadata
                            if ds.size <= 1_000 and np.array_equal(ds[()], arr):
                                continue
                            # If sizes match but content unknown (large), we assume user wants to keep existing
                            continue
                        else:
                            raise ValueError(
                                f"Constant '{name}' already exists with different metadata. "
                                f"Set overwrite_constants=True to replace it."
                            )

                # Create fresh dataset for the constant
                ds = g_const.create_dataset(
                    name,
                    data=arr,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                if unit_str is not None:
                    ds.attrs["unit"] = unit_str
                if kind is not None:
                    ds.attrs["kind"] = kind

        # --- streaming batch for an iteration ---
        if iteration is not None:
            if not batch:
                return  # nothing to write for this iteration

            g_iter_root = f.require_group("iter")
            g_iter = g_iter_root.require_group(_iter_group_name(int(iteration)))

            for name, value in batch.items():
                arr, unit, kind = _to_array_unit_kind(value)
                unit_str = _to_unit_str(unit)
                arr = _ensure_row_first(arr)  # (N, ...)

                if name in g_iter:
                    ds = g_iter[name]

                    # Unit compatibility checks
                    ds_unit = ds.attrs.get("unit", None)
                    ds_kind = ds.attrs.get("kind", None)

                    if ds_unit is None and unit_str is not None:
                        raise ValueError(f"Cannot append unit '{unit_str}' to existing unitless dataset '{name}'.")
                    if ds_unit is not None and unit_str is None:
                        raise ValueError(f"Cannot append unitless data to existing unit '{ds_unit}' in dataset '{name}'.")

                    # Convert if both have units and differ
                    if (ds_unit is not None) and (unit_str is not None) and (ds_unit != unit_str):
                        if allow_unit_auto_convert and (u is not None) and _units_convertible(u.Unit(unit_str), u.Unit(ds_unit)):
                            arr = _convert_array_unit(arr, u.Unit(unit_str), u.Unit(ds_unit))
                            unit_str = ds_unit  # now aligned
                        else:
                            raise ValueError(
                                f"Units differ for '{name}': incoming {unit_str} vs existing {ds_unit}."
                            )

                    # Expand and append
                    old_n = ds.shape[0]
                    new_n = old_n + arr.shape[0]
                    ds.resize((new_n,) + ds.shape[1:])
                    ds[old_n:new_n, ...] = arr

                    # Preserve attrs if they were missing
                    if (ds.attrs.get("unit", None) is None) and (unit_str is not None):
                        ds.attrs["unit"] = unit_str
                    if (ds.attrs.get("kind", None) is None) and (kind is not None):
                        ds.attrs["kind"] = kind

                else:
                    # New extendable dataset for this var
                    maxshape = (None,) + arr.shape[1:]
                    ds = g_iter.create_dataset(
                        name,
                        data=arr,
                        maxshape=maxshape,
                        chunks=True,
                        compression=compression,
                        compression_opts=compression_opts,
                        dtype=arr.dtype,
                    )
                    if unit_str is not None:
                        ds.attrs["unit"] = unit_str
                    if kind is not None:
                        ds.attrs["kind"] = kind


def read_data(
    filename: str,
    *,
    # selections
    iter_selection: Iterable[int] | None = None,
    var_selection: Iterable[str] | None = None,
    # how to return time datasets
    times_as: str = "time",        # "time" (default) or "quantity"
    # stacking options
    stack: bool = True,            # <— now defaults to True
    pad_value: float = np.nan,
    return_masks: bool = False,
    # also return raw per-iteration values alongside stacked
    include_by_iter: bool = False,
) -> Dict[str, Any]:
    """
    Read everything from an HDF5 file produced by write_data().

    Returns
    -------
    out : dict
        {
          "const": {name: array/Quantity/Time, ...},

          # With stack=True (default):
          "iter": {var: stacked_array_or_Time/Quantity, ...},
          # shape of each stacked array is (n_iter, Tmax, ...).
          # Iterations with fewer samples are padded with NaN.

          # Optional extras:
          # - "masks": {var: bool array of shape (n_iter, Tmax), True where padded}
          # - "by_iter": {iter_id: {var: array/Quantity/Time, ...}} if include_by_iter=True
        }
    """
    out: Dict[str, Any] = {"const": {}, "iter": {}}

    # ----- load constants and per-iteration raw data -----
    by_iter: Dict[int, Dict[str, Any]] = {}

    with h5py.File(filename, "r") as f:
        # constants
        if "const" in f:
            for name, ds in f["const"].items():
                out["const"][name] = _read_dataset(ds, times_as=times_as)

        # iterations
        if "iter" in f:
            all_iters = []
            for gname in f["iter"].keys():
                if gname.startswith("iter_") and len(gname) == len("iter_00000"):
                    try:
                        all_iters.append(int(gname.split("_")[1]))
                    except Exception:
                        pass
            all_iters.sort()

            if iter_selection is not None:
                want = sorted(set(int(i) for i in iter_selection))
                iters = [i for i in all_iters if i in want]
            else:
                iters = all_iters

            var_filter = set(var_selection) if var_selection is not None else None

            for ii in iters:
                g = f["iter"][f"iter_{ii:05d}"]
                row: Dict[str, Any] = {}
                for dname, ds in g.items():
                    if var_filter is not None and dname not in var_filter:
                        continue
                    row[dname] = _read_dataset(ds, times_as=times_as)
                by_iter[ii] = row

    # No iteration data found
    if not by_iter:
        if include_by_iter:
            out["by_iter"] = {}
        return out

    # Short-circuit if caller asked explicitly not to stack
    if not stack:
        if include_by_iter:
            out["by_iter"] = by_iter
        else:
            # mirror your previous shape: results["iter"][iter_id] -> {var: ...}
            out["iter"] = by_iter
        return out

    # ----- build stacked tensors per variable -----

    # union of variable names across all loaded iterations
    iter_ids = sorted(by_iter)
    var_names = set()
    for ii in iter_ids:
        var_names.update(by_iter[ii].keys())
    var_names = sorted(var_names)

    stacked: Dict[str, Any] = {}
    masks: Dict[str, np.ndarray] = {}

    for var in var_names:
        # Collect rows for all selected iterations; allow missing (treated as length 0)
        rows_numeric: List[np.ndarray] = []
        kinds: List[str | None] = []
        units_seen: List[Any | None] = []
        trailing_shape = None
        stackable = True

        for ii in iter_ids:
            val = by_iter[ii].get(var, None)

            # Missing in this iteration → 0-length row
            if val is None:
                arr = np.empty((0,), dtype=float)
                row_kind = None
                row_unit = None
            else:
                # Normalise to numeric ndarray for stacking and capture kind/unit
                if _is_time_obj(val):                      # astropy.Time
                    arr = np.asarray(val.mjd)
                    row_kind = "time/mjd"
                    row_unit = u.day if u is not None else None
                elif hasattr(val, "unit") and u is not None:  # Quantity
                    arr = np.asarray(val.value)
                    row_kind = "quantity"
                    row_unit = val.unit
                else:
                    arr = np.asarray(val)
                    row_kind = None
                    row_unit = None

                if arr.ndim == 0:
                    arr = arr.reshape(1)

            # Track trailing shape (beyond time axis)
            ts = arr.shape[1:]
            if trailing_shape is None:
                trailing_shape = ts
            elif trailing_shape != ts:
                # Incompatible shapes across iterations → cannot stack this variable
                stackable = False
                break

            rows_numeric.append(arr)
            kinds.append(row_kind)
            units_seen.append(row_unit)

        if not stackable:
            # Skip stacking; leave variable accessible via raw layout if requested
            continue

        # Determine max time length across iterations
        Tmax = max(r.shape[0] for r in rows_numeric)
        n_iter = len(rows_numeric)
        pad_shape = (n_iter, Tmax) + (trailing_shape or ())
        pad = np.full(pad_shape, pad_value, dtype=(rows_numeric[0].dtype if rows_numeric else float))
        mask = np.ones((n_iter, Tmax), dtype=bool)  # True where padded

        # Choose a “reference” unit/kind from the first non-empty row
        ref_unit = None
        ref_kind = None
        for k, (rk, ru) in enumerate(zip(kinds, units_seen)):
            if rows_numeric[k].size > 0:
                ref_kind = rk
                ref_unit = ru
                break

        # Fill rows
        for k, arr in enumerate(rows_numeric):
            t = arr.shape[0]
            if t > 0:
                pad[k, :t, ...] = arr
                mask[k, :t] = False

        # Attach unit or convert to Time as needed
        if ref_kind == "time/mjd" and times_as == "time" and Time is not None:
            # Convert numeric MJD to astropy.Time; NaNs remain as NaN times
            stacked[var] = Time(pad, format="mjd")
        elif ref_kind == "quantity" and (u is not None) and (ref_unit is not None):
            stacked[var] = pad * ref_unit
        else:
            stacked[var] = pad

        if return_masks:
            masks[var] = mask

    # Put results in the requested top-level shape
    out["iter"] = stacked
    if return_masks:
        out["masks"] = masks
    if include_by_iter:
        out["by_iter"] = by_iter

    return out

def analyse_time(times: Any) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """
    Compute timing stats for a single time series or a stacked set.

    Input
    -----
    times : astropy.time.Time | Quantity (MJD) | array-like (MJD)
        Shapes supported:
          (T,)           → one iteration with T samples
          (N, T)         → N iterations, T samples each
          (N, T, 1)      → will be squeezed to (N, T)
        NaNs are allowed (padding) and ignored.

    Output
    ------
    (timestep, iter_durations, total_duration)
        timestep        : mean consecutive spacing across all iterations [s]
        iter_durations  : per-iteration (last - first) duration [s], shape (N,)
        total_duration  : sum(iter_durations) [s]
    """
    # --- normalise to float MJD array -----------------------------------------
    if isinstance(times, Time):
        mjd = np.asarray(times.mjd, dtype=float)
    elif hasattr(times, "unit"):
        try:
            mjd = np.asarray(u.Quantity(times).to(u.day).value, dtype=float)
        except Exception:
            mjd = np.asarray(times, dtype=float)
    else:
        mjd = np.asarray(times, dtype=float)

    # Drop singleton axes so (N, T, 1) → (N, T)
    mjd = np.squeeze(mjd)

    # Ensure (N, T) shape
    if mjd.ndim == 0:
        mjd = mjd.reshape(1, 1)
    elif mjd.ndim == 1:
        mjd = mjd.reshape(1, -1)
    elif mjd.ndim > 2:
        # Flatten everything after the first axis into the time axis.
        mjd = mjd.reshape(mjd.shape[0], -1)

    n_iter, T = mjd.shape
    day_to_sec = u.day.to(u.s)

    # --- collect diffs and durations ------------------------------------------
    all_deltas_sec: list[float] = []
    iter_durs_sec = np.zeros(n_iter, dtype=float)

    for i in range(n_iter):
        row = mjd[i]
        mask = np.isfinite(row)
        if not np.any(mask):
            iter_durs_sec[i] = 0.0
            continue

        t = row[mask]
        # Ensure increasing order before diffs
        t = t[np.argsort(t)]

        if t.size >= 2:
            diffs = np.diff(t) * day_to_sec
            diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
            if diffs.size:
                all_deltas_sec.extend(diffs.tolist())
                iter_durs_sec[i] = (t[-1] - t[0]) * day_to_sec
            else:
                iter_durs_sec[i] = 0.0
        else:
            iter_durs_sec[i] = 0.0

    if len(all_deltas_sec) < 1:
        raise ValueError("Need at least two valid time points globally to compute timestep.")

    timestep        = (np.mean(all_deltas_sec)) * u.s
    iter_durations  = iter_durs_sec * u.s
    total_duration  = np.sum(iter_durs_sec) * u.s
    return timestep, iter_durations, total_duration

from typing import Any
import numpy as np
from astropy import units as u
from astropy.time import Time

def process_integration(
    epfd: Any,
    *,
    integration_period: Any = 2000 * u.s,
    times: Any | None = None,
    timestep: Any | None = None,
    method: str = "rough",               # "rough" | "accurate"
    windowing: str = "sliding",          # "sliding" | "subsequent"
    time_axis: int = 1,                  # which axis is time; typical EPFD shape is (I, T, …)
    base_floor: Any = (10.0**(-50.0)) * (u.W / u.m**2),  # ~ -500 dB(W/m^2), strictly positive
) -> Any:
    """
    Time-average EPFD with a boxcar of ~integration_period along the time axis.

    Usage
    -----
        # Sliding, rough (fast), infer Δt from times:
        epfd_s = process_integration(results["iter"]["epfd_0"],
                                     integration_period=2000*u.s,
                                     times=results["iter"]["times"])

        # Sliding, accurate (valid-only mean with a positive floor):
        epfd_s = process_integration(results["iter"]["epfd_0"],
                                     integration_period=2000*u.s,
                                     times=results["iter"]["times"],
                                     method="accurate")

        # Subsequent (non-overlapping) blocks:
        epfd_b = process_integration(results["iter"]["epfd_0"],
                                     integration_period=2000*u.s,
                                     times=results["iter"]["times"],
                                     windowing="subsequent")

        # If Δt is already known:
        dt, _, _ = analyse_time(results["iter"]["times"])  # Quantity [s]
        epfd_s = process_integration(results["iter"]["epfd_0"],
                                     integration_period=2000*u.s,
                                     timestep=dt)

    Parameters
    ----------
    epfd : array-like or astropy.Quantity
        EPFD values with time along `time_axis`. Must be broadcastable to float.
    integration_period : float or Quantity
        Desired averaging length. If a time Quantity, Δt is used to convert this
        to a sample window. If a plain number and `timestep`/`times` are not given,
        it is treated as a *sample count* directly.
    times : astropy.time.Time or array-like, optional
        Time stamps aligned to `epfd` along the time axis. Used to infer Δt via
        `analyse_time`. Shapes like (I, T) or (I, T, 1) are fine.
    timestep : float or Quantity, optional
        Δt override. If provided, `times` are not required.
    method : {"rough", "accurate"}
        "rough"    → standard boxcar mean (fast). NaNs propagate.
        "accurate" → mean over strictly positive finite samples only; empty
                     windows fall back to `base_floor` (in EPFD’s unit).
    windowing : {"sliding", "subsequent"}
        "sliding"    → overlapping windows; output has same time length as input.
        "subsequent" → non-overlapping K-sample blocks; output time length shrinks
                       to ceil(T/K). The last partial block is included.
    time_axis : int
        Axis index that holds time.
    base_floor : Quantity
        Lower bound used only in "accurate" mode when a block/window has no valid
        samples. Default is ~10^-50 W/m^2.

    Returns
    -------
    smoothed : same type as `epfd`
        Averaged EPFD. Units are preserved when present.

    Notes
    -----
    • Window length K is rounded to the nearest odd integer (≥1) and clamped to T.
    • “Rough” semantics: any NaN in the window makes the windowed mean NaN.
    • “Accurate” semantics: only strictly positive finite samples are averaged.
    """
    # ---------- units & raw values ----------
    has_unit = hasattr(epfd, "unit")
    unit = epfd.unit if has_unit else None
    vals = np.asarray(epfd.value if has_unit else epfd, dtype=np.float64)

    if vals.ndim < 1:
        raise ValueError("`epfd` must have at least one dimension (time axis).")
    if time_axis < 0:
        time_axis = vals.ndim + time_axis
    if not (0 <= time_axis < vals.ndim):
        raise ValueError("`time_axis` is out of range.")
    T = vals.shape[time_axis]
    if T < 1:
        raise ValueError("Time axis has length 0.")

    # ---------- Δt and window length K ----------
    # 1) Δt
    if timestep is not None:
        dt_s = float((timestep * 1.0).to(u.s).value) if hasattr(timestep, "to") else float(timestep)
        if dt_s <= 0.0:
            raise ValueError("`timestep` must be > 0.")
    elif times is not None:
        dt_q, _, _ = analyse_time(times)   # Quantity [s]
        dt_s = float(dt_q.to_value(u.s))
    else:
        dt_s = None  # only valid if integration_period is a sample count

    # 2) K
    if hasattr(integration_period, "to"):
        if dt_s is None:
            raise ValueError("Provide `times` or `timestep` when integration_period has time units.")
        ip_s = float(integration_period.to(u.s).value)
        K = max(1, int(round(ip_s / dt_s)))
    else:
        K = max(1, int(round(float(integration_period))))
    if K % 2 == 0:
        K += 1
    K = min(K, T)

    method = method.lower().strip()
    if method not in {"rough", "accurate"}:
        raise ValueError("`method` must be 'rough' or 'accurate'.")

    windowing = windowing.lower().strip()
    if windowing not in {"sliding", "subsequent"}:
        raise ValueError("`windowing` must be 'sliding' or 'subsequent'.")

    # ---------- helpers for "accurate" ----------
    if method == "accurate":
        valid_mask = np.isfinite(vals) & (vals > 0.0)

        # Effective floor in EPFD's unit; do not raise genuine data
        if has_unit:
            base_floor_num = float(u.Quantity(base_floor).to(unit).value)
        else:
            try:
                base_floor_num = float(base_floor)
            except Exception:
                base_floor_num = 0.0
        if np.any(valid_mask):
            min_valid = float(np.nanmin(vals[valid_mask]))
            eff_floor = max(min(base_floor_num, min_valid), np.finfo(np.float64).tiny)
        else:
            eff_floor = max(base_floor_num, np.finfo(np.float64).tiny)

    # =================================================================
    # SLIDING WINDOWS  → output has same time length as input
    # =================================================================
    if windowing == "sliding":
        if method == "rough":
            # Fast path
            try:
                from scipy.ndimage import uniform_filter1d
                out_vals = uniform_filter1d(vals, size=K, axis=time_axis, mode="nearest")
            except Exception:
                # Fallback: 1D convolution along the time axis
                kernel = np.ones(K, dtype=np.float64) / float(K)
                v = np.moveaxis(vals, time_axis, -1)
                out = np.empty_like(v)
                it = np.nditer(v[..., 0], flags=['multi_index'])
                while not it.finished:
                    idx = it.multi_index
                    out[idx] = np.convolve(v[idx], kernel, mode="same")
                    it.iternext()
                out_vals = np.moveaxis(out, -1, time_axis)
            return (out_vals * unit) if has_unit else out_vals

        # Accurate + sliding: valid-only mean via convolution
        try:
            from scipy.signal import convolve as _conv
            # Build kernel only on the time axis
            kshape = [1] * vals.ndim; kshape[time_axis] = K
            kernel = np.ones(kshape, dtype=np.float64)
            num = _conv(np.where(valid_mask, vals, 0.0), kernel, mode="same", method="auto")
            den = _conv(valid_mask.astype(np.float64), kernel, mode="same", method="auto")
        except Exception:
            # Fallback: per-slice conv along time axis
            v = np.moveaxis(vals, time_axis, -1)
            m = np.moveaxis(valid_mask, time_axis, -1)
            box = np.ones(K, dtype=np.float64)
            s_sum = np.empty_like(v)
            s_cnt = np.empty_like(v)
            it = np.nditer(v[..., 0], flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                row = v[idx]
                msk = m[idx].astype(np.float64)
                s_sum[idx] = np.convolve(np.where(msk > 0.0, row, 0.0), box, mode="same")
                s_cnt[idx] = np.convolve(msk, box, mode="same")
                it.iternext()
            num = np.moveaxis(s_sum, -1, time_axis)
            den = np.moveaxis(s_cnt, -1, time_axis)

        out_vals = np.full_like(num, eff_floor, dtype=np.float64)
        np.divide(num, den, out=out_vals, where=(den > 0.0))
        np.maximum(out_vals, eff_floor, out=out_vals)
        return (out_vals * unit) if has_unit else out_vals

    # =================================================================
    # SUBSEQUENT (NON-OVERLAPPING)  → time length shrinks to ceil(T/K)
    # =================================================================
    # Move time to the last axis for simple slicing; nwin includes the final partial block
    v = np.moveaxis(vals, time_axis, -1)
    nwin = (T + K - 1) // K

    if method == "rough":
        out = np.empty(v.shape[:-1] + (nwin,), dtype=np.float64)
        for w in range(nwin):
            s0 = w * K
            s1 = min(s0 + K, T)
            seg = v[..., s0:s1]
            # Rough semantics: any NaN in the block → NaN result
            out[..., w] = np.mean(seg, axis=-1)
        out_vals = np.moveaxis(out, -1, time_axis)
        return (out_vals * unit) if has_unit else out_vals

    # Accurate + subsequent: valid-only mean per block; empty → floor
    m = np.moveaxis(valid_mask, time_axis, -1).astype(np.float64)
    out = np.empty(v.shape[:-1] + (nwin,), dtype=np.float64)
    for w in range(nwin):
        s0 = w * K
        s1 = min(s0 + K, T)
        seg_v = v[..., s0:s1]
        seg_m = m[..., s0:s1]
        num = np.sum(seg_v * seg_m, axis=-1)
        den = np.sum(seg_m, axis=-1)
        blk = np.full_like(num, eff_floor, dtype=np.float64)
        np.divide(num, den, out=blk, where=(den > 0.0))
        np.maximum(blk, eff_floor, out=blk)
        out[..., w] = blk

    out_vals = np.moveaxis(out, -1, time_axis)
    return (out_vals * unit) if has_unit else out_vals
