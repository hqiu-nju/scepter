"""
Satellite-system modelling helpers for SCEPTer workflows.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>

This module remains part of the GPLv3-licensed SCEPTer project. The author
attribution above does not change the project-wide license.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from astropy import units as u
from astropy.constants import R_earth

if TYPE_CHECKING:
    from scepter.angle_sampler import JointAngleSampler

# ---------------------------------------------------------------------------
# Optional Numba import ------------------------------------------------------
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    from numba import set_num_threads as nb_set_num_threads
    from numba import get_num_threads as nb_get_num_threads
except Exception:
    njit = None
    prange = range
    def nb_set_num_threads(n: int) -> None:
        """No-op because Numba is not available."""
        pass
    def nb_get_num_threads() -> int:
        """Without numba a single-threaded approach would be used"""
        return 1


HAS_NUMBA = njit is not None

_RANDOM_PAIR_C1 = np.uint64(6364136223846793005)
_RANDOM_PAIR_C2 = np.uint64(1442695040888963407)

PURE_REROUTE_CSR_ROW_PTR_KEY = "sat_eligible_csr_row_ptr"
PURE_REROUTE_CSR_SAT_IDX_KEY = "sat_eligible_csr_sat_idx"
PURE_REROUTE_CSR_TIME_COUNT_KEY = "sat_eligible_csr_time_count"
PURE_REROUTE_CSR_CELL_COUNT_KEY = "sat_eligible_csr_cell_count"
PURE_REROUTE_CSR_SAT_COUNT_KEY = "sat_eligible_csr_sat_count"
_PURE_REROUTE_ELIGIBLE_MASK_ENCODINGS = {"dense", "csr", "both"}

# ===========================================================================
# Internal helpers
# ===========================================================================


def _normalise_eligible_mask_encoding(eligible_mask_encoding: str) -> str:
    encoding = str(eligible_mask_encoding).strip().lower()
    if encoding not in _PURE_REROUTE_ELIGIBLE_MASK_ENCODINGS:
        raise ValueError(
            "eligible_mask_encoding must be one of {'dense', 'csr', 'both'}."
        )
    return encoding


def _normalise_cell_active_mask(
    cell_active_mask: Any | None,
    *,
    time_count: int,
    cell_count: int,
    name: str = "cell_active_mask",
) -> np.ndarray | None:
    if cell_active_mask is None:
        return None
    mask = np.asarray(cell_active_mask, dtype=np.bool_)
    time_count_i = int(time_count)
    cell_count_i = int(cell_count)
    if mask.ndim == 1:
        if int(mask.shape[0]) != cell_count_i:
            raise ValueError(
                f"{name} with shape {tuple(mask.shape)!r} must have length {cell_count_i}."
            )
        return np.broadcast_to(mask.reshape(1, cell_count_i), (time_count_i, cell_count_i)).copy()
    if mask.ndim == 2:
        if tuple(mask.shape) != (time_count_i, cell_count_i):
            raise ValueError(
                f"{name} must have shape ({time_count_i}, {cell_count_i}) or ({cell_count_i},); "
                f"got {tuple(mask.shape)!r}."
            )
        return mask.astype(np.bool_, copy=False)
    raise ValueError(
        f"{name} must have shape ({time_count_i}, {cell_count_i}) or ({cell_count_i},); "
        f"got {tuple(mask.shape)!r}."
    )


def _pure_reroute_csr_metadata(
    *,
    time_count: int,
    cell_count: int,
    sat_count: int,
) -> dict[str, int]:
    return {
        PURE_REROUTE_CSR_TIME_COUNT_KEY: int(time_count),
        PURE_REROUTE_CSR_CELL_COUNT_KEY: int(cell_count),
        PURE_REROUTE_CSR_SAT_COUNT_KEY: int(sat_count),
    }


def _pure_reroute_pairs_to_csr_payload(
    time_idx: np.ndarray,
    cell_idx: np.ndarray,
    sat_idx: np.ndarray,
    *,
    time_count: int,
    cell_count: int,
    sat_count: int,
) -> dict[str, Any]:
    time_arr = np.asarray(time_idx, dtype=np.int64).reshape(-1)
    cell_arr = np.asarray(cell_idx, dtype=np.int64).reshape(-1)
    sat_arr = np.asarray(sat_idx, dtype=np.int32).reshape(-1)
    if not (time_arr.size == cell_arr.size == sat_arr.size):
        raise ValueError("time_idx, cell_idx, and sat_idx must have the same length.")
    time_count_i = int(time_count)
    cell_count_i = int(cell_count)
    sat_count_i = int(sat_count)
    if time_count_i < 0 or cell_count_i < 0 or sat_count_i < 0:
        raise ValueError("time_count, cell_count, and sat_count must be non-negative.")
    total_rows = time_count_i * cell_count_i
    if time_arr.size == 0:
        return {
            PURE_REROUTE_CSR_ROW_PTR_KEY: np.zeros(total_rows + 1, dtype=np.int64),
            PURE_REROUTE_CSR_SAT_IDX_KEY: np.empty(0, dtype=np.int32),
            **_pure_reroute_csr_metadata(
                time_count=time_count_i,
                cell_count=cell_count_i,
                sat_count=sat_count_i,
            ),
        }
    if (
        np.any(time_arr < 0)
        or np.any(time_arr >= time_count_i)
        or np.any(cell_arr < 0)
        or np.any(cell_arr >= cell_count_i)
        or np.any(sat_arr < 0)
        or np.any(sat_arr >= sat_count_i)
    ):
        raise ValueError("CSR eligibility indices must lie within the declared shape bounds.")

    row_idx = time_arr * np.int64(cell_count_i) + cell_arr
    order = np.lexsort((sat_arr.astype(np.int64, copy=False), row_idx))
    row_sorted = row_idx[order]
    sat_sorted = sat_arr[order].astype(np.int32, copy=False)
    keep = np.ones(row_sorted.size, dtype=np.bool_)
    if row_sorted.size > 1:
        keep[1:] = (row_sorted[1:] != row_sorted[:-1]) | (sat_sorted[1:] != sat_sorted[:-1])
    row_sorted = row_sorted[keep]
    sat_sorted = sat_sorted[keep]
    counts = np.bincount(row_sorted.astype(np.int64, copy=False), minlength=total_rows).astype(np.int64, copy=False)
    row_ptr = np.empty(total_rows + 1, dtype=np.int64)
    row_ptr[0] = np.int64(0)
    row_ptr[1:] = np.cumsum(counts, dtype=np.int64)
    return {
        PURE_REROUTE_CSR_ROW_PTR_KEY: row_ptr,
        PURE_REROUTE_CSR_SAT_IDX_KEY: sat_sorted.astype(np.int32, copy=False),
        **_pure_reroute_csr_metadata(
            time_count=time_count_i,
            cell_count=cell_count_i,
            sat_count=sat_count_i,
        ),
    }


def _pure_reroute_dense_mask_to_csr_payload(eligible_mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(eligible_mask, dtype=np.bool_)
    if mask.ndim != 3:
        raise ValueError(
            "eligible_mask must have shape (T, C, S) when converting to CSR; "
            f"got {tuple(mask.shape)!r}."
        )
    time_idx, cell_idx, sat_idx = np.nonzero(mask)
    return _pure_reroute_pairs_to_csr_payload(
        time_idx.astype(np.int64, copy=False),
        cell_idx.astype(np.int64, copy=False),
        sat_idx.astype(np.int32, copy=False),
        time_count=int(mask.shape[0]),
        cell_count=int(mask.shape[1]),
        sat_count=int(mask.shape[2]),
    )


def _attach_eligible_mask_outputs(
    result: dict[str, np.ndarray],
    *,
    dense_mask: np.ndarray | None,
    eligible_time: np.ndarray | None,
    eligible_cell: np.ndarray | None,
    eligible_sat: np.ndarray | None,
    time_count: int,
    cell_count: int,
    sat_count: int,
    encoding: str,
) -> None:
    encoding_name = _normalise_eligible_mask_encoding(encoding)
    if encoding_name in {"dense", "both"}:
        if dense_mask is None:
            raise RuntimeError("Dense eligible-mask output requested but no dense mask is available.")
        result["sat_eligible_mask"] = np.asarray(dense_mask, dtype=np.bool_, copy=False)
    if encoding_name in {"csr", "both"}:
        if dense_mask is not None:
            payload = _pure_reroute_dense_mask_to_csr_payload(
                np.asarray(dense_mask, dtype=np.bool_, copy=False)
            )
        else:
            if eligible_time is None or eligible_cell is None or eligible_sat is None:
                raise RuntimeError("CSR eligible-mask output requested but no sparse eligibility indices are available.")
            payload = _pure_reroute_pairs_to_csr_payload(
                np.asarray(eligible_time, dtype=np.int64),
                np.asarray(eligible_cell, dtype=np.int64),
                np.asarray(eligible_sat, dtype=np.int32),
                time_count=int(time_count),
                cell_count=int(cell_count),
                sat_count=int(sat_count),
            )
        result.update(payload)

def _greedy_python(
    order: np.ndarray,      # (T, C·S) arg-sorted flat indices
    vis_flat: np.ndarray,   # (T, C·S) bool visibility, flattened same way
    C: int,                 # Number of cells
    S: int,                 # Number of satellites
    Nco: int,               # How many connections each cell can support
    Nbeam: int,             # Maximum cell connections per satellit
) -> np.ndarray:
    """
    Greedy matcher (pure Python, fallback when Numba unavailable).

    Walks **column-wise** through *order*:  
    column 0 contains the single best candidate of every time slice,  
    column 1 the second-best, and so on.

    Stops early when all cells reach Nco beams *for every* time slice.

    Returns
    -------
    assign : np.ndarray[int32]  (T, C, Nco)
        Filled with satellite indices; unassigned slots are -1.
    """
    T = order.shape[0]

    # Result initialised to -1 → means “no satellite assigned”
    assign = -np.ones((T, C, Nco), dtype=np.int32)

    # Counters track how many beams each cell / satellite already has
    cell_cnt = np.zeros((T, C), dtype=np.int32)  # beams per cell
    sat_cnt  = np.zeros((T, S), dtype=np.int32)  # cells per satellite

    # ------------------------------------------------------------------
    # MAIN LOOP over weight-rank (columns)
    # ------------------------------------------------------------------
    for col in range(order.shape[1]):

        all_times_done = True  # assume finished until proven otherwise

        # Iterate over **time slices** (outer loop small, inner loops cheap)
        for t in range(T):
            # If every cell in slice already has Nco beams, skip slice
            if cell_cnt[t].min() >= Nco:
                continue
            all_times_done = False  # still work to do

            flat_idx = order[t, col]          # best remaining index
            if not vis_flat[t, flat_idx]:     # invisible pair → skip
                continue

            cell = flat_idx // S              # unravel flat → 2-D indices
            sat  = flat_idx % S

            # Capacity test: both sides must still have room
            if cell_cnt[t, cell] < Nco and sat_cnt[t, sat] < Nbeam:
                slot = cell_cnt[t, cell]      # next free slot (0..Nco-1)
                assign[t, cell, slot] = sat   # write assignment
                cell_cnt[t, cell] += 1        # update counters
                sat_cnt[t, sat]   += 1

        if all_times_done:     # nothing left to do in *any* slice
            break

    return assign


# JIT-compile identical logic when Numba available
if njit is not None:
    _greedy_numba = njit(_greedy_python, fastmath=True, cache=True)

    @njit(parallel=True, fastmath=True, cache=True)
    def _unlimited_parallel(rows_weights: np.ndarray,
                            S: int,
                            Nco: int) -> np.ndarray:
        """
        Assign links when *satellites have NO per-satellite beam limit*.

        Very easy mental model
        ----------------------
        Imagine a **big spreadsheet**:

        =====================  ====================================
        Row index              What it represents
        =====================  ====================================
        ``0 … T·C - 1``        Each row = **one cell at one time-step**
        ``Columns (0 … S-1)``  Each column = **one satellite**
        Cell value (float)     Random weight  
                                (smaller number = “I like this sat more”)
                                or **+∞** if satellite is *below horizon*
        =====================  ====================================

        Goal for every row
        ------------------
        Pick up to **Nco** satellites with *finite* weights and the smallest
        numbers (i.e. best random winners).

        Parallel trick
        --------------
        *Rows are totally independent.*  We therefore loop over the rows
        with ``prange`` so that OpenMP can hand out different chunks of the
        spreadsheet to different CPU threads.

        Parameters
        ----------
        rows_weights : ndarray, ``shape = (rows, S)``
            The spreadsheet described above where
            ``rows = time_steps * cells``.
        S : int
            Number of satellites (i.e. number of columns in the sheet).
        Nco : int
            How many satellites each cell *wants* at the same time.

        Returns
        -------
        ndarray, ``shape = (rows, Nco)``, dtype *int32*
            For every row stores the chosen **satellite indices**.
            An entry is **-1** if the cell could not get enough visible
            satellites (e.g. midnight and everyone is below horizon).

        Step-by-step inside the loop
        ----------------------------
        1. **`np.argpartition`** quickly finds the positions of the *Nco*
           smallest weights in *O(S)* time (no full sort).
        2. We **sort just those Nco numbers** so the output order is
           deterministic (useful for testing and reproducibility).
        3. We filter out any index whose weight was +∞ (satellite invisible).
        4. We write the survivors into the output array.
        """
        rows = rows_weights.shape[0]                 # total spreadsheet rows
        out  = -np.ones((rows, Nco), dtype=np.int32) # result initialised to -1

        # --- Outer loop: OpenMP distributes rows among threads -----------
        for r in prange(rows):
            # 1  Pick candidate satellites (indices of smallest weights)
            part  = np.argpartition(rows_weights[r], Nco - 1)[:Nco]

            # 2  Get their actual weights and sort them for nice ordering
            sel_w = rows_weights[r, part]
            ord_  = np.argsort(sel_w)

            # 3  Keep only *finite* weights (visible satellites)
            good  = sel_w[ord_] < np.inf

            # 4  Write the final list (may be shorter than Nco)
            out[r, :np.sum(good)] = part[ord_][good]

        return out
    
    @njit(parallel=True, fastmath=True, cache=True)
    def _greedy_parallel(order: np.ndarray,
                         vis_flat: np.ndarray,
                         C: int, S: int,
                         Nco: int, Nbeam: int) -> np.ndarray:
        """
        Assign links when *each satellite can serve only `Nbeam` cells*.

        Why only the time axis is parallelised
        --------------------------------------
        Within one time-step all cells **compete** for the limited
        *satellite beams*.  Therefore, iterations inside a time slice share
        the arrays that count beams already used, so they **cannot** run in
        parallel without locks.  However, *different* time slices are
        completely isolated → we loop with ``prange`` over `t`.

        Input arrays
        ------------
        * ``order[t]`` – 1-D list of **all** (cell, sat) pairs **sorted by
          weight**, flattened so index
          ``idx = cell * S + sat``.
        * ``vis_flat[t]`` – same flattening but stores ``True`` / ``False``
          for visibility.

        High-level algorithm inside each slice
        --------------------------------------
        1. Walk the sorted list **from best weight to worst**.
        2. For each pair check:
           • Does the *cell* still need a beam?  
           • Does the *satellite* still have spare capacity?
        3. If *yes* to both → record the link and update the counters.
        4. Stop early when every cell has reached `Nco` links.
        """
        T, CS = order.shape
        out = -np.ones((T, C, Nco), dtype=np.int32)   # final answer

        for t in prange(T):                           # OpenMP ⇢ one slice / core
            cell_cnt = np.zeros(C, np.int32)          # beams used by each cell
            sat_cnt  = np.zeros(S, np.int32)          # cells served by sat

            for col in range(CS):                     # walk weight-sorted list
                if cell_cnt.min() >= Nco:             # 👉 every cell full
                    break
                idx = order[t, col]                   # flattened (cell, sat)
                if not vis_flat[t, idx]:              # sat below horizon
                    continue

                cell = idx // S                      # unravel indices
                sat  = idx %  S

                if cell_cnt[cell] < Nco and sat_cnt[sat] < Nbeam:
                    slot = cell_cnt[cell]            # which column to fill
                    out[t, cell, slot] = sat
                    cell_cnt[cell] += 1
                    sat_cnt[sat]   += 1
        return out
else:
    _greedy_numba = _greedy_python

    def _unlimited_parallel(rows_weights: np.ndarray,
                            S: int,
                            Nco: int) -> np.ndarray:          # type: ignore
        """
        Serial fallback for unlimited-beam case (runs if Numba missing)."""
        part = np.argpartition(rows_weights, Nco - 1, axis=-1)[..., :Nco]
        sel  = np.take_along_axis(rows_weights, part, axis=-1)
        ord_ = np.argsort(sel, axis=-1)
        idx  = np.take_along_axis(part, ord_, axis=-1)
        mask = np.take_along_axis(sel, ord_, axis=-1) < np.inf
        out  = -np.ones(rows_weights.shape[:-1] + (Nco,), np.int32)
        out[mask] = idx[mask]
        return out
    
    _greedy_parallel = _greedy_python


def _normalise_selection_strategy(selection_strategy: str | None) -> str:
    strategy = (selection_strategy or "random").lower()
    if strategy in ("random", "rng", "rand"):
        return "random"
    if strategy in ("max_elevation", "highest_elevation", "max_el", "elevation", "elev"):
        return "max_elevation"
    raise ValueError(
        f"Unknown selection_strategy={selection_strategy!r}; supported: 'random', 'max_elevation'"
    )


def _ensure_pairwise_time_axis(arr: np.ndarray, *, name: str) -> tuple[np.ndarray, bool]:
    if arr.ndim == 3:
        return arr[np.newaxis, ...], True
    if arr.ndim == 4:
        return arr, False
    raise ValueError(f"{name} must be a 3-D or 4-D array.")


def _slice_cells_view(
    arr: np.ndarray,
    *,
    name: str,
    full_observer_count: int,
    cell_observer_offset: int,
) -> np.ndarray:
    if arr.shape[1] == full_observer_count:
        return arr[:, cell_observer_offset:, :, :]
    if arr.shape[1] == (full_observer_count - cell_observer_offset):
        return arr
    raise ValueError(
        f"{name} observer dimension {arr.shape[1]} is incompatible with "
        f"full_observer_count={full_observer_count} and cell_observer_offset={cell_observer_offset}."
    )


def _ensure_ras_topo(
    ras_topo: np.ndarray | None,
    *,
    sat_topo_full: np.ndarray,
    cell_observer_offset: int,
    T_local: int,
    n_sats: int,
) -> np.ndarray | None:
    if ras_topo is None:
        if cell_observer_offset < 1:
            return None
        ras_topo_arr = sat_topo_full[:, 0, :, :]
    else:
        ras_topo_arr = np.asarray(ras_topo)
        if ras_topo_arr.ndim == 2:
            ras_topo_arr = ras_topo_arr[np.newaxis, ...]
        elif ras_topo_arr.ndim == 4:
            if ras_topo_arr.shape[1] < 1:
                raise ValueError("ras_topo observer axis must not be empty.")
            ras_topo_arr = ras_topo_arr[:, 0, :, :]
        elif ras_topo_arr.ndim != 3:
            raise ValueError("ras_topo must be a 2-D, 3-D, or 4-D array.")

    return _normalise_ras_topo_array(ras_topo_arr, T_local=T_local, n_sats=n_sats)


def _normalise_ras_topo_array(
    ras_topo: np.ndarray,
    *,
    T_local: int,
    n_sats: int,
) -> np.ndarray:
    ras_topo_arr = np.asarray(ras_topo)
    if ras_topo_arr.ndim == 2:
        ras_topo_arr = ras_topo_arr[np.newaxis, ...]
    elif ras_topo_arr.ndim == 4:
        if ras_topo_arr.shape[1] < 1:
            raise ValueError("ras_topo observer axis must not be empty.")
        ras_topo_arr = ras_topo_arr[:, 0, :, :]
    elif ras_topo_arr.ndim != 3:
        raise ValueError("ras_topo must be a 2-D, 3-D, or 4-D array.")

    if ras_topo_arr.shape[0] != T_local:
        raise ValueError(f"ras_topo time dimension {ras_topo_arr.shape[0]} does not match T={T_local}.")
    if ras_topo_arr.shape[1] != n_sats:
        raise ValueError(f"ras_topo satellite dimension {ras_topo_arr.shape[1]} does not match S={n_sats}.")
    if ras_topo_arr.shape[-1] < 2:
        raise ValueError("ras_topo last axis must contain at least azimuth and elevation.")
    return ras_topo_arr


def _normalise_per_satellite_float(
    value: Any,
    n_sats: int,
    *,
    name: str,
    default_unit: u.Unit,
    dtype: np.dtype,
    fill_value: float | None = None,
) -> np.ndarray:
    if value is None:
        if fill_value is None:
            raise ValueError(f"{name} must not be None.")
        return np.full(n_sats, fill_value, dtype=dtype)

    if hasattr(value, "to_value"):
        arr = np.asarray(u.Quantity(value).to_value(default_unit), dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 0:
        return np.full(n_sats, float(arr), dtype=dtype)

    flat = arr.reshape(-1)
    if flat.size == 1:
        return np.full(n_sats, float(flat[0]), dtype=dtype)
    if flat.size != n_sats:
        raise ValueError(f"{name} must be scalar or have length {n_sats}, got {flat.size}.")
    return flat.astype(dtype, copy=False)


def _normalise_per_satellite_int(
    value: Any,
    n_sats: int,
    *,
    name: str,
    dtype: np.dtype,
    fill_value: int,
) -> np.ndarray:
    if value is None:
        return np.full(n_sats, fill_value, dtype=dtype)

    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(n_sats, int(arr), dtype=dtype)

    flat = arr.reshape(-1)
    if flat.size == 1:
        return np.full(n_sats, int(flat[0]), dtype=dtype)
    if flat.size != n_sats:
        raise ValueError(f"{name} must be scalar or have length {n_sats}, got {flat.size}.")
    return flat.astype(dtype, copy=False)


def _squeeze_time_axis(result: dict[str, np.ndarray], single_time_input: bool) -> dict[str, np.ndarray]:
    if not single_time_input:
        return result

    squeezed: dict[str, np.ndarray] = {}
    for key, value in result.items():
        squeezed[key] = value[0]
    return squeezed


def _normalise_boresight_theta_deg(
    value: float | u.Quantity | None,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    angle = float(u.Quantity(value).to_value(u.deg) if hasattr(value, "to") else value)
    if angle < 0.0:
        raise ValueError(f"{name} must be non-negative or None.")
    return angle


def _normalise_boresight_pointing_arrays(
    azimuth_deg: Any,
    elevation_deg: Any,
    *,
    time_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    az = np.asarray(azimuth_deg, dtype=np.float32)
    el = np.asarray(elevation_deg, dtype=np.float32)
    if az.shape != el.shape:
        raise ValueError("boresight_pointing_azimuth_deg and boresight_pointing_elevation_deg must share the same shape.")
    if az.ndim != 2:
        raise ValueError(
            "boresight_pointing_azimuth_deg and boresight_pointing_elevation_deg must have shape (T, N_sky)."
        )
    if int(az.shape[0]) != int(time_count):
        raise ValueError(
            "Boresight pointing time axis must match the selection time axis; "
            f"expected {int(time_count)}, got {int(az.shape[0])}."
        )
    return az.astype(np.float32, copy=False), el.astype(np.float32, copy=False)


def _normalise_boresight_cell_ids(
    value: Any,
    *,
    cell_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(value, dtype=np.int32).reshape(-1)

    # Empty is valid: after ACTIVE-axis projection a scope resolved via
    # adjacency_layers or radius_km may legitimately affect no active cells.
    if ids.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.zeros(int(cell_count), dtype=np.bool_),
        )

    if np.any(ids < 0) or np.any(ids >= int(cell_count)):
        raise ValueError(
            f"boresight_theta2_cell_ids must lie within [0, {int(cell_count) - 1}]."
        )
    ids_unique = np.unique(ids.astype(np.int32, copy=False))
    mask = np.zeros(int(cell_count), dtype=np.bool_)
    mask[ids_unique] = True
    return ids_unique, mask


def _compute_boresight_satellite_masks_numpy(
    ras_topo: np.ndarray,
    boresight_azimuth_deg: np.ndarray,
    boresight_elevation_deg: np.ndarray,
    *,
    theta1_deg: float | None,
    theta2_deg: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    time_count, sat_count = int(ras_topo.shape[0]), int(ras_topo.shape[1])
    sky_count = int(boresight_azimuth_deg.shape[1])
    full_mask = np.zeros((time_count, sky_count, sat_count), dtype=np.bool_)
    partial_mask = np.zeros_like(full_mask)
    if theta1_deg is None and theta2_deg is None:
        return full_mask, partial_mask

    sat_az_rad = np.deg2rad(np.remainder(ras_topo[..., 0].astype(np.float32, copy=False), np.float32(360.0)))
    sat_el_rad = np.deg2rad(ras_topo[..., 1].astype(np.float32, copy=False))
    tel_az_rad = np.deg2rad(np.remainder(boresight_azimuth_deg.astype(np.float32, copy=False), np.float32(360.0)))
    tel_el_rad = np.deg2rad(boresight_elevation_deg.astype(np.float32, copy=False))

    cos_daz = np.cos(tel_az_rad[:, :, None] - sat_az_rad[:, None, :]).astype(np.float32, copy=False)
    cos_gamma = (
        np.sin(tel_el_rad[:, :, None]).astype(np.float32, copy=False)
        * np.sin(sat_el_rad[:, None, :]).astype(np.float32, copy=False)
        + np.cos(tel_el_rad[:, :, None]).astype(np.float32, copy=False)
        * np.cos(sat_el_rad[:, None, :]).astype(np.float32, copy=False)
        * cos_daz
    ).astype(np.float32, copy=False)
    cos_gamma = np.clip(cos_gamma, np.float32(-1.0), np.float32(1.0))
    separation_deg = np.rad2deg(np.arccos(cos_gamma)).astype(np.float32, copy=False)
    above_horizon = ras_topo[:, None, :, 1].astype(np.float32, copy=False) > np.float32(0.0)

    if theta1_deg is not None:
        full_mask = above_horizon & (separation_deg < np.float32(theta1_deg))
    if theta2_deg is not None:
        partial_mask = above_horizon & (separation_deg < np.float32(theta2_deg))
        if theta1_deg is not None:
            partial_mask &= ~full_mask
    return full_mask, partial_mask


def _splitmix64_numpy(x: np.ndarray | np.uint64) -> np.ndarray:
    z = np.asarray(x, dtype=np.uint64)
    z = z * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    z = z ^ (z >> np.uint64(33))
    z = z * np.uint64(3202034522624059733)
    z = z ^ (z >> np.uint64(29))
    z = z * np.uint64(3935559000370003845)
    return z ^ (z >> np.uint64(32))


def _pair_priority_random_numpy(
    seed_u64: np.uint64,
    row_index: np.ndarray,
    sat_index: np.ndarray,
) -> np.ndarray:
    row_u64 = np.asarray(row_index, dtype=np.uint64)
    sat_u64 = np.asarray(sat_index, dtype=np.uint64)
    key = np.uint64(seed_u64) + (row_u64 + np.uint64(1)) * _RANDOM_PAIR_C1
    key = key + (sat_u64 + np.uint64(1)) * _RANDOM_PAIR_C2
    return _splitmix64_numpy(key)


@dataclass(slots=True)
class SatelliteLinkSelectionLibrary:
    """
    Chunk-stable finite-cap link allocator for step-1 batch loops.

    Parameters
    ----------
    time_count : int
        Number of timesteps represented by this library.
    cell_count : int
        Total number of cell observers in the full batch.
    sat_count : int
        Total number of satellites in the full batch.
    min_elevation_deg : float, Quantity, or array-like, optional
        Minimum serving elevation threshold in degrees. Scalars apply to every
        satellite; one-dimensional arrays must have length ``sat_count``.
    n_links : int, optional
        Number of concurrent serving links retained per cell.
    n_beam : int
        Per-satellite capacity cap applied independently at each timestep.
    strategy : {"random", "max_elevation"}, optional
        Candidate ranking strategy used for the global greedy allocator.
    sat_belt_id_per_sat : array-like, optional
        Integer belt identifier for each satellite.
    beta_max_deg_per_sat : float, Quantity, or array-like, optional
        Per-satellite maximum cone angle ``beta`` in degrees.
    ras_topo : np.ndarray, optional
        RAS-frame satellite geometry with shape ``(T, S, K)``, ``(S, K)``, or
        ``(T, O, S, K)``. Required when ``include_payload=True``.
    cell_active_mask : np.ndarray, optional
        Boolean demand mask aligned with the compact cell axis. Accepted shapes
        are ``(T, C)`` and ``(C,)``. Inactive cells contribute no selected
        links, counts, or eligibility.
    rng : np.random.Generator, int, numpy integer, or None, optional
        Random source used only for ``strategy="random"``. The generator is
        consumed once when the library is initialised so chunk order does not
        affect priorities.
    include_counts : bool, optional
        If ``True``, return per-satellite demand and cone-valid counts.
    include_payload : bool, optional
        If ``True``, return the selected-link payload tensors.
    include_eligible_mask : bool, optional
        If ``True``, store uncapped candidate eligibility for the batch.
    eligible_mask_encoding : {"dense", "csr", "both"}, optional
        Output encoding used when ``include_eligible_mask=True``. ``"dense"``
        returns ``sat_eligible_mask`` with shape ``(T, C, S)``. ``"csr"``
        returns sparse exact-reroute inputs via
        ``sat_eligible_csr_row_ptr``, ``sat_eligible_csr_sat_idx``, and the
        matching ``*_time_count`` / ``*_cell_count`` / ``*_sat_count`` scalar
        metadata. ``"both"`` returns both representations.
    boresight_pointing_azimuth_deg, boresight_pointing_elevation_deg : array-like, optional
        Sampled RAS telescope pointings with shape ``(T, N_sky)``. Required
        only when ``boresight_theta1_deg`` or ``boresight_theta2_deg`` is
        active.
    boresight_theta1_deg, boresight_theta2_deg : float or Quantity, optional
        Optional hard-shutdown and local-exclusion angles in degrees.
        ``None`` disables the corresponding rule.
    boresight_theta2_cell_ids : array-like of int, optional
        Cell-axis indices that define the local ``Theta_2`` exclusion region.
        Required only when ``boresight_theta2_deg`` is active.
    beta_tol_deg : float, optional
        Absolute cone-validation tolerance in degrees.

    Notes
    -----
    Each call to :meth:`add_chunk` registers one contiguous cell chunk for all
    timesteps in the batch. After all chunks are added, :meth:`finalize`
    performs the exact greedy finite-cap allocation and returns the same result
    structure as :func:`select_satellite_links`, always with an explicit time
    axis.
    """

    time_count: int
    cell_count: int
    sat_count: int
    min_elevation_deg: float | u.Quantity | np.ndarray = 30 * u.deg
    n_links: int = 1
    n_beam: int = 1
    strategy: str = "random"
    sat_belt_id_per_sat: np.ndarray | None = None
    beta_max_deg_per_sat: float | u.Quantity | np.ndarray | None = None
    ras_topo: np.ndarray | None = None
    cell_active_mask: np.ndarray | None = None
    rng: np.random.Generator | int | np.integer | None = None
    include_counts: bool = True
    include_payload: bool = True
    include_eligible_mask: bool = False
    eligible_mask_encoding: str = "dense"
    boresight_pointing_azimuth_deg: np.ndarray | None = None
    boresight_pointing_elevation_deg: np.ndarray | None = None
    boresight_theta1_deg: float | u.Quantity | None = None
    boresight_theta2_deg: float | u.Quantity | None = None
    boresight_theta2_cell_ids: np.ndarray | None = None
    beta_tol_deg: float = 1e-3
    sat_min_elev_deg: np.ndarray = field(init=False, repr=False)
    sat_beta_max_deg: np.ndarray = field(init=False, repr=False)
    sat_belt_id: np.ndarray = field(init=False, repr=False)
    strategy_name: str = field(init=False)
    use_cone_check: bool = field(init=False)
    ras_topo_use: np.ndarray | None = field(init=False, repr=False)
    cell_active_mask_use: np.ndarray | None = field(init=False, repr=False)
    seed_u64: np.uint64 = field(init=False, repr=False)
    eligible_mask: np.ndarray | None = field(init=False, repr=False)
    eligible_mask_encoding_name: str = field(init=False)
    boresight_theta1_deg_value: float | None = field(init=False, repr=False)
    boresight_theta2_deg_value: float | None = field(init=False, repr=False)
    boresight_sky_count: int = field(init=False)
    boresight_active: bool = field(init=False)
    boresight_pointing_az_deg_use: np.ndarray | None = field(init=False, repr=False)
    boresight_pointing_el_deg_use: np.ndarray | None = field(init=False, repr=False)
    boresight_theta2_cell_ids_use: np.ndarray | None = field(init=False, repr=False)
    boresight_theta2_cell_mask: np.ndarray | None = field(init=False, repr=False)
    boresight_full_mask: np.ndarray | None = field(init=False, repr=False)
    boresight_partial_mask: np.ndarray | None = field(init=False, repr=False)
    _cells_seen: np.ndarray = field(init=False, repr=False)
    _candidate_time_parts: list[np.ndarray] = field(init=False, repr=False)
    _candidate_cell_parts: list[np.ndarray] = field(init=False, repr=False)
    _candidate_sat_parts: list[np.ndarray] = field(init=False, repr=False)
    _candidate_key_parts: list[np.ndarray] = field(init=False, repr=False)
    _candidate_alpha_parts: list[np.ndarray] | None = field(init=False, repr=False)
    _candidate_beta_parts: list[np.ndarray] | None = field(init=False, repr=False)
    _eligible_time_parts: list[np.ndarray] | None = field(init=False, repr=False)
    _eligible_cell_parts: list[np.ndarray] | None = field(init=False, repr=False)
    _eligible_sat_parts: list[np.ndarray] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.time_count) <= 0:
            raise ValueError("time_count must be positive.")
        if int(self.cell_count) <= 0:
            raise ValueError("cell_count must be positive.")
        if int(self.sat_count) <= 0:
            raise ValueError("sat_count must be positive.")
        if int(self.n_links) <= 0:
            raise ValueError("n_links must be positive.")
        if int(self.n_beam) <= 0:
            raise ValueError("n_beam must be positive.")

        self.time_count = int(self.time_count)
        self.cell_count = int(self.cell_count)
        self.sat_count = int(self.sat_count)
        self.n_links = int(self.n_links)
        self.n_beam = int(self.n_beam)
        self.strategy_name = _normalise_selection_strategy(self.strategy)
        self.eligible_mask_encoding_name = _normalise_eligible_mask_encoding(self.eligible_mask_encoding)
        self.use_cone_check = self.beta_max_deg_per_sat is not None
        self.sat_min_elev_deg = _normalise_per_satellite_float(
            self.min_elevation_deg,
            self.sat_count,
            name="min_elevation_deg",
            default_unit=u.deg,
            dtype=np.float64,
        )
        self.sat_beta_max_deg = _normalise_per_satellite_float(
            self.beta_max_deg_per_sat,
            self.sat_count,
            name="beta_max_deg_per_sat",
            default_unit=u.deg,
            dtype=np.float32,
            fill_value=np.inf,
        )
        self.sat_belt_id = _normalise_per_satellite_int(
            self.sat_belt_id_per_sat,
            self.sat_count,
            name="sat_belt_id_per_sat",
            dtype=np.int16,
            fill_value=-1,
        )
        self.seed_u64 = _seed_from_rng_like(self.rng) if self.strategy_name == "random" else np.uint64(0)

        if self.include_payload:
            if self.ras_topo is None:
                raise ValueError("ras_topo is required when include_payload=True.")
            self.ras_topo_use = _normalise_ras_topo_array(
                np.asarray(self.ras_topo),
                T_local=self.time_count,
                n_sats=self.sat_count,
            )
        elif self.ras_topo is not None:
            self.ras_topo_use = _normalise_ras_topo_array(
                np.asarray(self.ras_topo),
                T_local=self.time_count,
                n_sats=self.sat_count,
            )
        else:
            self.ras_topo_use = None
        self.cell_active_mask_use = _normalise_cell_active_mask(
            self.cell_active_mask,
            time_count=self.time_count,
            cell_count=self.cell_count,
        )

        self.boresight_theta1_deg_value = _normalise_boresight_theta_deg(
            self.boresight_theta1_deg,
            name="boresight_theta1_deg",
        )
        self.boresight_theta2_deg_value = _normalise_boresight_theta_deg(
            self.boresight_theta2_deg,
            name="boresight_theta2_deg",
        )
        if (
            self.boresight_theta1_deg_value is not None
            and self.boresight_theta2_deg_value is not None
            and self.boresight_theta1_deg_value > self.boresight_theta2_deg_value
        ):
            raise ValueError("boresight_theta1_deg must be less than or equal to boresight_theta2_deg.")
        self.boresight_active = (
            self.boresight_theta1_deg_value is not None or self.boresight_theta2_deg_value is not None
        )
        self.boresight_pointing_az_deg_use = None
        self.boresight_pointing_el_deg_use = None
        self.boresight_theta2_cell_ids_use = None
        self.boresight_theta2_cell_mask = None
        self.boresight_full_mask = None
        self.boresight_partial_mask = None
        self.boresight_sky_count = 1
        if self.boresight_active:
            if self.ras_topo_use is None:
                raise ValueError("ras_topo is required when boresight avoidance is active.")
            if self.boresight_pointing_azimuth_deg is None or self.boresight_pointing_elevation_deg is None:
                raise ValueError(
                    "boresight_pointing_azimuth_deg and boresight_pointing_elevation_deg are required "
                    "when boresight avoidance is active."
                )
            (
                self.boresight_pointing_az_deg_use,
                self.boresight_pointing_el_deg_use,
            ) = _normalise_boresight_pointing_arrays(
                self.boresight_pointing_azimuth_deg,
                self.boresight_pointing_elevation_deg,
                time_count=self.time_count,
            )
            self.boresight_sky_count = int(self.boresight_pointing_az_deg_use.shape[1])
            if self.boresight_theta2_deg_value is not None:
                if self.boresight_theta2_cell_ids is None:
                    raise ValueError("boresight_theta2_cell_ids is required when boresight_theta2_deg is active.")
                (
                    self.boresight_theta2_cell_ids_use,
                    self.boresight_theta2_cell_mask,
                ) = _normalise_boresight_cell_ids(
                    self.boresight_theta2_cell_ids,
                    cell_count=self.cell_count,
                )
            self.boresight_full_mask, self.boresight_partial_mask = _compute_boresight_satellite_masks_numpy(
                self.ras_topo_use,
                self.boresight_pointing_az_deg_use,
                self.boresight_pointing_el_deg_use,
                theta1_deg=self.boresight_theta1_deg_value,
                theta2_deg=self.boresight_theta2_deg_value,
            )

        self.eligible_mask = None
        if self.include_eligible_mask and (
            self.eligible_mask_encoding_name in {"dense", "both"} or self.boresight_active
        ):
            self.eligible_mask = np.zeros((self.time_count, self.cell_count, self.sat_count), dtype=np.bool_)

        self._cells_seen = np.zeros(self.cell_count, dtype=np.bool_)
        self._candidate_time_parts = []
        self._candidate_cell_parts = []
        self._candidate_sat_parts = []
        self._candidate_key_parts = []
        if self.include_payload or self.use_cone_check:
            self._candidate_alpha_parts = []
            self._candidate_beta_parts = []
        else:
            self._candidate_alpha_parts = None
            self._candidate_beta_parts = None
        if self.include_eligible_mask and self.eligible_mask_encoding_name in {"csr", "both"} and not self.boresight_active:
            self._eligible_time_parts = []
            self._eligible_cell_parts = []
            self._eligible_sat_parts = []
        else:
            self._eligible_time_parts = None
            self._eligible_cell_parts = None
            self._eligible_sat_parts = None

    def add_chunk(
        self,
        cell_offset: int,
        sat_topo_chunk: np.ndarray,
        *,
        sat_azel: np.ndarray | None = None,
    ) -> None:
        """
        Register one contiguous cell chunk for the batch.

        Parameters
        ----------
        cell_offset : int
            Zero-based offset of the first cell in this chunk within the full
            batch.
        sat_topo_chunk : np.ndarray
            Cell-slice topocentric geometry with shape ``(T, C_chunk, S, K)`` or
            ``(C_chunk, S, K)``.
        sat_azel : np.ndarray, optional
            Satellite-frame geometry aligned with ``sat_topo_chunk``. Required
            when payload gathering, cone checks, or eligible-mask output are
            enabled.
        """
        cell_offset_i = int(cell_offset)
        if cell_offset_i < 0 or cell_offset_i > self.cell_count:
            raise ValueError(f"cell_offset must be in [0, {self.cell_count}], got {cell_offset_i}.")

        sat_topo_full, _ = _ensure_pairwise_time_axis(np.asarray(sat_topo_chunk), name="sat_topo_chunk")
        if sat_topo_full.shape[0] != self.time_count:
            raise ValueError(
                f"sat_topo_chunk time dimension {sat_topo_full.shape[0]} does not match "
                f"time_count={self.time_count}."
            )
        if sat_topo_full.shape[2] != self.sat_count:
            raise ValueError(
                f"sat_topo_chunk satellite dimension {sat_topo_full.shape[2]} does not match "
                f"sat_count={self.sat_count}."
            )
        if sat_topo_full.shape[-1] < 2:
            raise ValueError("sat_topo_chunk last axis must contain at least azimuth and elevation.")

        chunk_cell_count = int(sat_topo_full.shape[1])
        cell_stop = cell_offset_i + chunk_cell_count
        if cell_stop > self.cell_count:
            raise ValueError(
                f"Chunk cell range [{cell_offset_i}, {cell_stop}) exceeds cell_count={self.cell_count}."
            )
        if np.any(self._cells_seen[cell_offset_i:cell_stop]):
            raise ValueError(f"Chunk cell range [{cell_offset_i}, {cell_stop}) overlaps an existing chunk.")
        self._cells_seen[cell_offset_i:cell_stop] = True

        sat_azel_full = None
        if sat_azel is not None:
            sat_azel_full, _ = _ensure_pairwise_time_axis(np.asarray(sat_azel), name="sat_azel")
            if sat_azel_full.shape[:3] != sat_topo_full.shape[:3]:
                raise ValueError("sat_azel must align with sat_topo_chunk over (T, C, S).")
            if sat_azel_full.shape[-1] < 2:
                raise ValueError("sat_azel last axis must contain at least azimuth and theta.")

        if (self.include_payload or self.use_cone_check or self.include_eligible_mask) and sat_azel_full is None:
            raise ValueError(
                "sat_azel is required when include_payload=True, include_eligible_mask=True, "
                "or beta_max_deg_per_sat is provided."
            )

        elev_deg = sat_topo_full[..., 1]
        visible = elev_deg >= self.sat_min_elev_deg[None, None, :]
        if self.cell_active_mask_use is not None:
            visible = visible & self.cell_active_mask_use[:, cell_offset_i:cell_stop, None]

        if self.include_eligible_mask:
            eligible_chunk = visible
            if self.use_cone_check:
                theta_all = np.abs(sat_azel_full[..., 1]).astype(np.float32, copy=False)
                eligible_chunk = eligible_chunk & (
                    theta_all <= (self.sat_beta_max_deg[None, None, :] + np.float32(self.beta_tol_deg))
                )
            if self.eligible_mask is not None:
                self.eligible_mask[:, cell_offset_i:cell_stop, :] = eligible_chunk.astype(np.bool_, copy=False)
            if (
                self._eligible_time_parts is not None
                and self._eligible_cell_parts is not None
                and self._eligible_sat_parts is not None
            ):
                eligible_time, eligible_local_cell, eligible_sat = np.nonzero(eligible_chunk)
                self._eligible_time_parts.append(eligible_time.astype(np.int32, copy=False))
                self._eligible_cell_parts.append(
                    eligible_local_cell.astype(np.int32, copy=False) + np.int32(cell_offset_i)
                )
                self._eligible_sat_parts.append(eligible_sat.astype(np.int32, copy=False))

        if not np.any(visible):
            return

        t_idx, local_cell_idx, sat_idx = np.nonzero(visible)
        global_cell_idx = local_cell_idx.astype(np.int32, copy=False) + np.int32(cell_offset_i)
        global_row_idx = (
            t_idx.astype(np.int64, copy=False) * np.int64(self.cell_count)
            + global_cell_idx.astype(np.int64, copy=False)
        )

        self._candidate_time_parts.append(t_idx.astype(np.int32, copy=False))
        self._candidate_cell_parts.append(global_cell_idx.astype(np.int32, copy=False))
        self._candidate_sat_parts.append(sat_idx.astype(np.int32, copy=False))
        if self.strategy_name == "max_elevation":
            self._candidate_key_parts.append((-elev_deg[visible]).astype(np.float64, copy=False))
        else:
            self._candidate_key_parts.append(_pair_priority_random_numpy(self.seed_u64, global_row_idx, sat_idx))

        if self._candidate_alpha_parts is not None and self._candidate_beta_parts is not None:
            alpha_all = np.remainder(sat_azel_full[..., 0], np.float32(360.0)).astype(np.float32, copy=False)
            beta_all = np.abs(sat_azel_full[..., 1]).astype(np.float32, copy=False)
            self._candidate_alpha_parts.append(alpha_all[visible].astype(np.float32, copy=False))
            self._candidate_beta_parts.append(beta_all[visible].astype(np.float32, copy=False))

    def finalize(self) -> dict[str, np.ndarray]:
        """
        Finalize the exact capped allocation and return explicit-time results.

        Returns
        -------
        dict[str, np.ndarray]
            Same result structure as :func:`select_satellite_links`, with shapes
            ``(T, C, n_links)`` and explicit time axis preserved.
        """
        if not np.all(self._cells_seen):
            missing = np.flatnonzero(~self._cells_seen)
            raise RuntimeError(
                f"Missing cell chunks for {missing.size} cells; first missing index is {int(missing[0])}."
            )

        assignments = np.full((self.time_count, self.cell_count, self.n_links), -1, dtype=np.int32)
        beam_counts_demand = np.zeros((self.time_count, self.sat_count), dtype=np.int32)
        beam_counts_eligible = np.zeros((self.time_count, self.sat_count), dtype=np.int32)

        sat_azimuth = None
        sat_elevation = None
        sat_alpha = None
        sat_beta = None
        sat_belt = None
        cone_ok = None
        if self.include_payload:
            sat_azimuth = np.full(assignments.shape, np.nan, dtype=np.float32)
            sat_elevation = np.full(assignments.shape, np.nan, dtype=np.float32)
            sat_alpha = np.full(assignments.shape, np.nan, dtype=np.float32)
            sat_beta = np.full(assignments.shape, np.nan, dtype=np.float32)
            sat_belt = np.full(assignments.shape, np.int16(-1), dtype=np.int16)
            cone_ok = np.zeros(assignments.shape, dtype=np.bool_)

        if self._candidate_time_parts:
            candidate_time = np.concatenate(self._candidate_time_parts).astype(np.int32, copy=False)
            candidate_cell = np.concatenate(self._candidate_cell_parts).astype(np.int32, copy=False)
            candidate_sat = np.concatenate(self._candidate_sat_parts).astype(np.int32, copy=False)
            candidate_key = np.concatenate(self._candidate_key_parts)
            candidate_alpha = None
            candidate_beta = None
            if self._candidate_alpha_parts is not None and self._candidate_beta_parts is not None:
                candidate_alpha = np.concatenate(self._candidate_alpha_parts).astype(np.float32, copy=False)
                candidate_beta = np.concatenate(self._candidate_beta_parts).astype(np.float32, copy=False)

            order = np.lexsort((candidate_sat, candidate_cell, candidate_key, candidate_time))
            candidate_time = candidate_time[order]
            candidate_cell = candidate_cell[order]
            candidate_sat = candidate_sat[order]
            if candidate_alpha is not None:
                candidate_alpha = candidate_alpha[order]
            if candidate_beta is not None:
                candidate_beta = candidate_beta[order]

            if self.boresight_active:
                return _finalize_boresight_link_selection_library_numpy(
                    self,
                    candidate_time=candidate_time,
                    candidate_cell=candidate_cell,
                    candidate_sat=candidate_sat,
                    candidate_alpha=candidate_alpha,
                    candidate_beta=candidate_beta,
                )

            ras_azimuth_all = None
            ras_elevation_all = None
            if self.ras_topo_use is not None:
                ras_azimuth_all = np.remainder(self.ras_topo_use[..., 0], np.float32(360.0)).astype(
                    np.float32, copy=False
                )
                ras_elevation_all = self.ras_topo_use[..., 1].astype(np.float32, copy=False)

            time_ptr = np.zeros(self.time_count + 1, dtype=np.int64)
            if candidate_time.size > 0:
                time_counts = np.bincount(candidate_time.astype(np.int64, copy=False), minlength=self.time_count)
                time_ptr[1:] = np.cumsum(time_counts, dtype=np.int64)

            for t in range(self.time_count):
                if time_ptr[t] == time_ptr[t + 1]:
                    continue
                cell_counts = np.zeros(self.cell_count, dtype=np.int32)
                filled_cells = 0
                start = int(time_ptr[t])
                stop = int(time_ptr[t + 1])
                for idx in range(start, stop):
                    cell = int(candidate_cell[idx])
                    sat = int(candidate_sat[idx])
                    slot = int(cell_counts[cell])
                    if slot >= self.n_links:
                        continue
                    if int(beam_counts_demand[t, sat]) >= self.n_beam:
                        continue

                    assignments[t, cell, slot] = sat
                    cell_counts[cell] = np.int32(slot + 1)
                    beam_counts_demand[t, sat] += np.int32(1)
                    if slot + 1 == self.n_links:
                        filled_cells += 1

                    if candidate_beta is None:
                        beam_counts_eligible[t, sat] += np.int32(1)
                    else:
                        beta_val = np.float32(candidate_beta[idx])
                        ok = bool(beta_val <= (self.sat_beta_max_deg[sat] + np.float32(self.beta_tol_deg)))
                        if ok:
                            beam_counts_eligible[t, sat] += np.int32(1)
                        if self.include_payload:
                            cone_ok[t, cell, slot] = ok
                            if ok:
                                sat_azimuth[t, cell, slot] = ras_azimuth_all[t, sat]
                                sat_elevation[t, cell, slot] = ras_elevation_all[t, sat]
                                sat_alpha[t, cell, slot] = candidate_alpha[idx]
                                sat_beta[t, cell, slot] = beta_val
                                sat_belt[t, cell, slot] = self.sat_belt_id[sat]
                    if candidate_beta is None and self.include_payload:
                        sat_azimuth[t, cell, slot] = ras_azimuth_all[t, sat]
                        sat_elevation[t, cell, slot] = ras_elevation_all[t, sat]
                        sat_alpha[t, cell, slot] = candidate_alpha[idx]
                        sat_beta[t, cell, slot] = np.float32(candidate_beta[idx]) if candidate_beta is not None else 0.0
                        sat_belt[t, cell, slot] = self.sat_belt_id[sat]
                        cone_ok[t, cell, slot] = True

                    if filled_cells >= self.cell_count:
                        break

        result: dict[str, np.ndarray] = {"assignments": assignments}
        if self.include_payload:
            result["sat_azimuth"] = sat_azimuth
            result["sat_elevation"] = sat_elevation
            result["sat_alpha"] = sat_alpha
            result["sat_beta"] = sat_beta
            result["sat_belt_id"] = sat_belt
            result["cone_ok"] = cone_ok
        if self.include_counts:
            result["sat_beam_counts_demand"] = beam_counts_demand
            result["sat_beam_counts_eligible"] = beam_counts_eligible
        if self.include_eligible_mask:
            _attach_eligible_mask_outputs(
                result,
                dense_mask=self.eligible_mask,
                eligible_time=(
                    np.concatenate(self._eligible_time_parts).astype(np.int32, copy=False)
                    if self._eligible_time_parts
                    else np.empty(0, dtype=np.int32)
                    if self._eligible_time_parts is not None
                    else None
                ),
                eligible_cell=(
                    np.concatenate(self._eligible_cell_parts).astype(np.int32, copy=False)
                    if self._eligible_cell_parts
                    else np.empty(0, dtype=np.int32)
                    if self._eligible_cell_parts is not None
                    else None
                ),
                eligible_sat=(
                    np.concatenate(self._eligible_sat_parts).astype(np.int32, copy=False)
                    if self._eligible_sat_parts
                    else np.empty(0, dtype=np.int32)
                    if self._eligible_sat_parts is not None
                    else None
                ),
                time_count=self.time_count,
                cell_count=self.cell_count,
                sat_count=self.sat_count,
                encoding=self.eligible_mask_encoding_name,
            )
        return result


def _finalize_boresight_link_selection_library_numpy(
    library: SatelliteLinkSelectionLibrary,
    *,
    candidate_time: np.ndarray,
    candidate_cell: np.ndarray,
    candidate_sat: np.ndarray,
    candidate_alpha: np.ndarray | None,
    candidate_beta: np.ndarray | None,
) -> dict[str, np.ndarray]:
    sky_count = int(library.boresight_sky_count)
    assignments = np.full((library.time_count, sky_count, library.cell_count, library.n_links), -1, dtype=np.int32)
    beam_counts_demand = np.zeros((library.time_count, sky_count, library.sat_count), dtype=np.int32)
    beam_counts_eligible = np.zeros((library.time_count, sky_count, library.sat_count), dtype=np.int32)

    sat_azimuth = None
    sat_elevation = None
    sat_alpha = None
    sat_beta = None
    sat_belt = None
    cone_ok = None
    if library.include_payload:
        sat_azimuth = np.full(assignments.shape, np.nan, dtype=np.float32)
        sat_elevation = np.full(assignments.shape, np.nan, dtype=np.float32)
        sat_alpha = np.full(assignments.shape, np.nan, dtype=np.float32)
        sat_beta = np.full(assignments.shape, np.nan, dtype=np.float32)
        sat_belt = np.full(assignments.shape, np.int16(-1), dtype=np.int16)
        cone_ok = np.zeros(assignments.shape, dtype=np.bool_)
    elif library.include_counts and library.use_cone_check:
        cone_ok = np.zeros(assignments.shape, dtype=np.bool_)

    ras_azimuth_all = None
    ras_elevation_all = None
    if library.ras_topo_use is not None:
        ras_azimuth_all = np.remainder(library.ras_topo_use[..., 0], np.float32(360.0)).astype(np.float32, copy=False)
        ras_elevation_all = library.ras_topo_use[..., 1].astype(np.float32, copy=False)

    time_ptr = np.zeros(library.time_count + 1, dtype=np.int64)
    if candidate_time.size > 0:
        time_counts = np.bincount(candidate_time.astype(np.int64, copy=False), minlength=library.time_count)
        time_ptr[1:] = np.cumsum(time_counts, dtype=np.int64)

    theta2_cell_mask = library.boresight_theta2_cell_mask
    full_mask = library.boresight_full_mask
    partial_mask = library.boresight_partial_mask

    for t in range(library.time_count):
        if time_ptr[t] == time_ptr[t + 1]:
            continue
        start = int(time_ptr[t])
        stop = int(time_ptr[t + 1])
        for sky in range(sky_count):
            cell_counts = np.zeros(library.cell_count, dtype=np.int32)
            filled_cells = 0
            full_row = full_mask[t, sky] if full_mask is not None else None
            partial_row = partial_mask[t, sky] if partial_mask is not None else None
            for idx in range(start, stop):
                cell = int(candidate_cell[idx])
                sat = int(candidate_sat[idx])
                if full_row is not None and bool(full_row[sat]):
                    continue
                if (
                    partial_row is not None
                    and theta2_cell_mask is not None
                    and bool(partial_row[sat])
                    and bool(theta2_cell_mask[cell])
                ):
                    continue
                slot = int(cell_counts[cell])
                if slot >= library.n_links:
                    continue
                if int(beam_counts_demand[t, sky, sat]) >= library.n_beam:
                    continue

                assignments[t, sky, cell, slot] = sat
                cell_counts[cell] = np.int32(slot + 1)
                beam_counts_demand[t, sky, sat] += np.int32(1)
                if slot + 1 == library.n_links:
                    filled_cells += 1

                if candidate_beta is None:
                    beam_counts_eligible[t, sky, sat] += np.int32(1)
                    if library.include_payload:
                        sat_azimuth[t, sky, cell, slot] = ras_azimuth_all[t, sat]
                        sat_elevation[t, sky, cell, slot] = ras_elevation_all[t, sat]
                        sat_alpha[t, sky, cell, slot] = candidate_alpha[idx]
                        sat_beta[t, sky, cell, slot] = np.float32(0.0)
                        sat_belt[t, sky, cell, slot] = library.sat_belt_id[sat]
                        cone_ok[t, sky, cell, slot] = True
                else:
                    beta_val = np.float32(candidate_beta[idx])
                    ok = bool(beta_val <= (library.sat_beta_max_deg[sat] + np.float32(library.beta_tol_deg)))
                    if ok:
                        beam_counts_eligible[t, sky, sat] += np.int32(1)
                    if cone_ok is not None:
                        cone_ok[t, sky, cell, slot] = ok
                    if library.include_payload and ok:
                        sat_azimuth[t, sky, cell, slot] = ras_azimuth_all[t, sat]
                        sat_elevation[t, sky, cell, slot] = ras_elevation_all[t, sat]
                        sat_alpha[t, sky, cell, slot] = candidate_alpha[idx]
                        sat_beta[t, sky, cell, slot] = beta_val
                        sat_belt[t, sky, cell, slot] = library.sat_belt_id[sat]

                if filled_cells >= library.cell_count:
                    break

    result: dict[str, np.ndarray] = {"assignments": assignments}
    if library.include_payload:
        result["sat_azimuth"] = sat_azimuth
        result["sat_elevation"] = sat_elevation
        result["sat_alpha"] = sat_alpha
        result["sat_beta"] = sat_beta
        result["sat_belt_id"] = sat_belt
        result["cone_ok"] = cone_ok
    elif cone_ok is not None:
        result["cone_ok"] = cone_ok
    if library.include_counts:
        result["sat_beam_counts_demand"] = beam_counts_demand
        result["sat_beam_counts_eligible"] = beam_counts_eligible
    if library.include_eligible_mask and library.eligible_mask is not None:
        eligible = np.broadcast_to(
            library.eligible_mask[:, np.newaxis, :, :],
            (library.time_count, sky_count, library.cell_count, library.sat_count),
        ).copy()
        if full_mask is not None:
            eligible &= ~full_mask[:, :, np.newaxis, :]
        if partial_mask is not None and theta2_cell_mask is not None:
            eligible &= ~(partial_mask[:, :, np.newaxis, :] & theta2_cell_mask[np.newaxis, np.newaxis, :, np.newaxis])
        result["sat_eligible_mask"] = eligible
    return result


def _seed_from_rng_like(rng: np.random.Generator | int | np.integer | None) -> np.uint64:
    if rng is None:
        local_rng = np.random.default_rng()
        return np.uint64(local_rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    if isinstance(rng, np.random.Generator):
        return np.uint64(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    return np.uint64(int(rng))


def _count_per_satellite_numpy(
    links_tcn: np.ndarray,
    n_sats_full: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    Tt, Nc, Nco_ = links_tcn.shape
    out = np.zeros((Tt, int(n_sats_full)), dtype=np.int32)

    flat_links = links_tcn.reshape(-1)
    if valid_mask is None:
        good = flat_links >= 0
    else:
        if valid_mask.shape != links_tcn.shape:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match links shape {links_tcn.shape}."
            )
        good = (flat_links >= 0) & valid_mask.reshape(-1)

    if not np.any(good):
        return out

    flat_pos = np.nonzero(good)[0].astype(np.int64, copy=False)
    t_idx = flat_pos // (Nc * Nco_)
    sat_idx = flat_links[good].astype(np.int64, copy=False)
    comp = t_idx * int(n_sats_full) + sat_idx
    bc = np.bincount(comp, minlength=Tt * int(n_sats_full))
    out[:] = bc.reshape(Tt, int(n_sats_full)).astype(np.int32, copy=False)
    return out


def _select_topn_numpy_chunk(
    elev_tcs: np.ndarray,
    sat_min_elev_deg: np.ndarray,
    nco: int,
    *,
    selection_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    Tt, Cc, S = elev_tcs.shape
    if nco < 1:
        raise ValueError("nco must be >= 1.")
    if S < 1:
        return np.full((Tt, Cc, nco), -1, dtype=np.int32)

    elig = elev_tcs >= sat_min_elev_deg[None, None, :]

    if selection_mode == "max_elevation":
        scores = -elev_tcs
        scores[~elig] = np.inf
    elif selection_mode == "random":
        scores = rng.random((Tt, Cc, S), dtype=np.float32)
        scores[~elig] = np.float32(np.inf)
    else:
        raise ValueError(f"Unknown selection_mode={selection_mode!r}")

    kth = min(nco - 1, S - 1)
    idx = np.argpartition(scores, kth=kth, axis=2)[:, :, :nco].astype(np.int32, copy=False)
    picked = np.take_along_axis(scores, idx, axis=2)
    return np.where(np.isfinite(picked), idx, np.int32(-1)).astype(np.int32, copy=False)


if HAS_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def _count_per_satellite_numba_nomask(links_tcn: np.ndarray, n_sats_full: int) -> np.ndarray:
        Tt, Nc, Nco_ = links_tcn.shape
        out = np.zeros((Tt, n_sats_full), dtype=np.int32)
        for t in prange(Tt):
            for c in range(Nc):
                for k in range(Nco_):
                    sat = links_tcn[t, c, k]
                    if sat >= 0:
                        out[t, sat] += 1
        return out


    @njit(cache=True, fastmath=True, parallel=True)
    def _count_per_satellite_numba_masked(
        links_tcn: np.ndarray,
        valid_mask: np.ndarray,
        n_sats_full: int,
    ) -> np.ndarray:
        Tt, Nc, Nco_ = links_tcn.shape
        out = np.zeros((Tt, n_sats_full), dtype=np.int32)
        for t in prange(Tt):
            for c in range(Nc):
                for k in range(Nco_):
                    if not valid_mask[t, c, k]:
                        continue
                    sat = links_tcn[t, c, k]
                    if sat >= 0:
                        out[t, sat] += 1
        return out


    @njit(cache=True, fastmath=True, parallel=True)
    def _accumulate_per_satellite_numba_nomask(links_tcn: np.ndarray, out_ts: np.ndarray) -> None:
        Tt, Nc, Nco_ = links_tcn.shape
        for t in prange(Tt):
            row_out = out_ts[t]
            for c in range(Nc):
                for k in range(Nco_):
                    sat = links_tcn[t, c, k]
                    if sat >= 0:
                        row_out[sat] += 1


    @njit(cache=True, fastmath=True, parallel=True)
    def _accumulate_per_satellite_numba_masked(
        links_tcn: np.ndarray,
        valid_mask: np.ndarray,
        out_ts: np.ndarray,
    ) -> None:
        Tt, Nc, Nco_ = links_tcn.shape
        for t in prange(Tt):
            row_out = out_ts[t]
            for c in range(Nc):
                for k in range(Nco_):
                    if not valid_mask[t, c, k]:
                        continue
                    sat = links_tcn[t, c, k]
                    if sat >= 0:
                        row_out[sat] += 1


    @njit(cache=True, fastmath=True, parallel=True)
    def _accumulate_per_satellite_top1_numba_nomask(links_tc: np.ndarray, out_ts: np.ndarray) -> None:
        Tt, Nc = links_tc.shape
        for t in prange(Tt):
            row_out = out_ts[t]
            for c in range(Nc):
                sat = links_tc[t, c]
                if sat >= 0:
                    row_out[sat] += 1


    @njit(cache=True, fastmath=True, parallel=True)
    def _accumulate_per_satellite_top1_numba_masked(
        links_tc: np.ndarray,
        valid_mask_tc: np.ndarray,
        out_ts: np.ndarray,
    ) -> None:
        Tt, Nc = links_tc.shape
        for t in prange(Tt):
            row_out = out_ts[t]
            for c in range(Nc):
                if not valid_mask_tc[t, c]:
                    continue
                sat = links_tc[t, c]
                if sat >= 0:
                    row_out[sat] += 1


    @njit(cache=True, fastmath=True, parallel=True)
    def _select_topn_max_elevation_numba(
        elev_tcs: np.ndarray,
        sat_min_elev_deg: np.ndarray,
        nco: int,
    ) -> np.ndarray:
        Tt, Nc, S = elev_tcs.shape
        out = np.full((Tt, Nc, nco), -1, dtype=np.int32)
        neg_inf = -1.0e300

        for row in prange(Tt * Nc):
            t = row // Nc
            c = row - (t * Nc)

            best_elev = np.empty(nco, dtype=np.float64)
            best_sat = np.empty(nco, dtype=np.int32)
            for k in range(nco):
                best_elev[k] = neg_inf
                best_sat[k] = -1

            for s in range(S):
                e = elev_tcs[t, c, s]
                if e < sat_min_elev_deg[s]:
                    continue

                insert_at = -1
                for k in range(nco):
                    if e > best_elev[k]:
                        insert_at = k
                        break

                if insert_at >= 0:
                    for k in range(nco - 1, insert_at, -1):
                        best_elev[k] = best_elev[k - 1]
                        best_sat[k] = best_sat[k - 1]
                    best_elev[insert_at] = e
                    best_sat[insert_at] = s

            for k in range(nco):
                out[t, c, k] = best_sat[k]

        return out


    @njit(cache=True, fastmath=True, parallel=True)
    def _select_top1_max_elevation_numba(
        elev_tcs: np.ndarray,
        sat_min_elev_deg: np.ndarray,
    ) -> np.ndarray:
        Tt, Nc, S = elev_tcs.shape
        out = np.full((Tt, Nc), -1, dtype=np.int32)
        neg_inf = -1.0e300

        for row in prange(Tt * Nc):
            t = row // Nc
            c = row - (t * Nc)

            best_e = neg_inf
            best_s = -1
            for s in range(S):
                e = elev_tcs[t, c, s]
                if e < sat_min_elev_deg[s]:
                    continue
                if e > best_e:
                    best_e = e
                    best_s = s

            out[t, c] = best_s

        return out


    @njit(cache=True, fastmath=True)
    def _splitmix64(x: np.uint64) -> np.uint64:
        z = x * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        z = z ^ (z >> 33)
        z = z * np.uint64(3202034522624059733)
        z = z ^ (z >> 29)
        z = z * np.uint64(3935559000370003845)
        return z ^ (z >> 32)


    @njit(cache=True, fastmath=True, parallel=True)
    def _select_topn_random_eligible_numba(
        elev_tcs: np.ndarray,
        sat_min_elev_deg: np.ndarray,
        nco: int,
        seed: np.uint64,
    ) -> np.ndarray:
        Tt, Nc, S = elev_tcs.shape
        out = np.full((Tt, Nc, nco), -1, dtype=np.int32)

        c1 = np.uint64(6364136223846793005)
        c2 = np.uint64(1442695040888963407)
        c3 = np.uint64(1140071481932319845)
        c4 = np.uint64(704602925438635313)

        for row in prange(Tt * Nc):
            t = row // Nc
            c = row - (t * Nc)

            chosen = np.full(nco, -1, dtype=np.int32)
            seen = 0

            for s in range(S):
                e = elev_tcs[t, c, s]
                if e < sat_min_elev_deg[s]:
                    continue

                seen += 1
                if seen <= nco:
                    chosen[seen - 1] = s
                else:
                    key = seed + (np.uint64(row + 1) * c1) + (np.uint64(s + 1) * c2)
                    j = int(_splitmix64(key) % np.uint64(seen))
                    if j < nco:
                        chosen[j] = s

            m = nco if seen >= nco else seen
            for i in range(m - 1, 0, -1):
                key = seed + (np.uint64(row + 1) * c3) + (np.uint64(i + 1) * c4)
                j = int(_splitmix64(key) % np.uint64(i + 1))
                tmp = chosen[i]
                chosen[i] = chosen[j]
                chosen[j] = tmp

            for k in range(m):
                out[t, c, k] = chosen[k]

        return out


    @njit(cache=True, fastmath=True, parallel=True)
    def _select_top1_random_eligible_numba(
        elev_tcs: np.ndarray,
        sat_min_elev_deg: np.ndarray,
        seed: np.uint64,
    ) -> np.ndarray:
        Tt, Nc, S = elev_tcs.shape
        out = np.full((Tt, Nc), -1, dtype=np.int32)

        c1 = np.uint64(6364136223846793005)
        c2 = np.uint64(1442695040888963407)

        for row in prange(Tt * Nc):
            t = row // Nc
            c = row - (t * Nc)

            seen = 0
            pick = -1

            for s in range(S):
                e = elev_tcs[t, c, s]
                if e < sat_min_elev_deg[s]:
                    continue

                seen += 1
                key = seed + (np.uint64(row + 1) * c1) + (np.uint64(s + 1) * c2)
                if int(_splitmix64(key) % np.uint64(seen)) == 0:
                    pick = s

            out[t, c] = pick

        return out


    @njit(cache=True, fastmath=True)
    def _norm_azimuth_deg_numba(v: np.float32) -> np.float32:
        if v >= np.float32(360.0) or v < np.float32(0.0):
            v = np.float32(v - np.floor(v / np.float32(360.0)) * np.float32(360.0))
        return v


    @njit(cache=True, fastmath=True, parallel=True)
    def _gather_selected_link_data_numba(
        sat_cell_links: np.ndarray,
        sat_azimuth_all: np.ndarray,
        sat_elevation_all: np.ndarray,
        sat_azel_cells: np.ndarray,
        sat_belt_id: np.ndarray,
        sat_beta_max_deg: np.ndarray,
        beta_tol_deg: np.float32,
    ):
        Tt, Nc, Nco_ = sat_cell_links.shape
        out_sat_az = np.empty((Tt, Nc, Nco_), dtype=np.float32)
        out_sat_el = np.empty((Tt, Nc, Nco_), dtype=np.float32)
        out_alpha = np.empty((Tt, Nc, Nco_), dtype=np.float32)
        out_beta = np.empty((Tt, Nc, Nco_), dtype=np.float32)
        out_belt = np.empty((Tt, Nc, Nco_), dtype=np.int16)
        cone_ok = np.zeros((Tt, Nc, Nco_), dtype=np.bool_)

        nan_f32 = np.float32(np.nan)
        neg_one_i16 = np.int16(-1)

        for row in prange(Tt * Nc):
            t = row // Nc
            c = row - (t * Nc)

            for k in range(Nco_):
                sat = sat_cell_links[t, c, k]
                if sat < 0:
                    out_sat_az[t, c, k] = nan_f32
                    out_sat_el[t, c, k] = nan_f32
                    out_alpha[t, c, k] = nan_f32
                    out_beta[t, c, k] = nan_f32
                    out_belt[t, c, k] = neg_one_i16
                    cone_ok[t, c, k] = False
                    continue

                sat_az = _norm_azimuth_deg_numba(np.float32(sat_azimuth_all[t, sat]))
                sat_el = np.float32(sat_elevation_all[t, sat])
                alpha = _norm_azimuth_deg_numba(np.float32(sat_azel_cells[t, c, sat, 0]))
                theta = np.float32(sat_azel_cells[t, c, sat, 1])
                beta = theta if theta >= np.float32(0.0) else np.float32(-theta)
                ok = beta <= (sat_beta_max_deg[sat] + beta_tol_deg)

                if ok:
                    out_sat_az[t, c, k] = sat_az
                    out_sat_el[t, c, k] = sat_el
                    out_alpha[t, c, k] = alpha
                    out_beta[t, c, k] = beta
                    out_belt[t, c, k] = sat_belt_id[sat]
                    cone_ok[t, c, k] = True
                else:
                    out_sat_az[t, c, k] = nan_f32
                    out_sat_el[t, c, k] = nan_f32
                    out_alpha[t, c, k] = nan_f32
                    out_beta[t, c, k] = nan_f32
                    out_belt[t, c, k] = neg_one_i16
                    cone_ok[t, c, k] = False

        return out_sat_az, out_sat_el, out_alpha, out_beta, out_belt, cone_ok


    @njit(cache=True, fastmath=True, parallel=True)
    def _gather_selected_link_data_top1_numba(
        sat_cell_links_tc: np.ndarray,
        sat_azimuth_all: np.ndarray,
        sat_elevation_all: np.ndarray,
        sat_azel_cells: np.ndarray,
        sat_belt_id: np.ndarray,
        sat_beta_max_deg: np.ndarray,
        beta_tol_deg: np.float32,
    ):
        Tt, Nc = sat_cell_links_tc.shape
        out_sat_az = np.empty((Tt, Nc), dtype=np.float32)
        out_sat_el = np.empty((Tt, Nc), dtype=np.float32)
        out_alpha = np.empty((Tt, Nc), dtype=np.float32)
        out_beta = np.empty((Tt, Nc), dtype=np.float32)
        out_belt = np.empty((Tt, Nc), dtype=np.int16)
        cone_ok = np.zeros((Tt, Nc), dtype=np.bool_)

        nan_f32 = np.float32(np.nan)
        neg_one_i16 = np.int16(-1)

        for row in prange(Tt * Nc):
            t = row // Nc
            c = row - (t * Nc)

            sat = sat_cell_links_tc[t, c]
            if sat < 0:
                out_sat_az[t, c] = nan_f32
                out_sat_el[t, c] = nan_f32
                out_alpha[t, c] = nan_f32
                out_beta[t, c] = nan_f32
                out_belt[t, c] = neg_one_i16
                cone_ok[t, c] = False
                continue

            sat_az = _norm_azimuth_deg_numba(np.float32(sat_azimuth_all[t, sat]))
            sat_el = np.float32(sat_elevation_all[t, sat])
            alpha = _norm_azimuth_deg_numba(np.float32(sat_azel_cells[t, c, sat, 0]))
            theta = np.float32(sat_azel_cells[t, c, sat, 1])
            beta = theta if theta >= np.float32(0.0) else np.float32(-theta)
            ok = beta <= (sat_beta_max_deg[sat] + beta_tol_deg)

            if ok:
                out_sat_az[t, c] = sat_az
                out_sat_el[t, c] = sat_el
                out_alpha[t, c] = alpha
                out_beta[t, c] = beta
                out_belt[t, c] = sat_belt_id[sat]
                cone_ok[t, c] = True
            else:
                out_sat_az[t, c] = nan_f32
                out_sat_el[t, c] = nan_f32
                out_alpha[t, c] = nan_f32
                out_beta[t, c] = nan_f32
                out_belt[t, c] = neg_one_i16
                cone_ok[t, c] = False

        return out_sat_az, out_sat_el, out_alpha, out_beta, out_belt, cone_ok


def _count_per_satellite(
    links_tcn: np.ndarray,
    n_sats_full: int,
    valid_mask: np.ndarray | None = None,
    *,
    prefer_numba: bool = False,
) -> np.ndarray:
    if valid_mask is not None and valid_mask.shape != links_tcn.shape:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match links shape {links_tcn.shape}."
        )

    if HAS_NUMBA and prefer_numba:
        if valid_mask is None:
            return _count_per_satellite_numba_nomask(links_tcn, int(n_sats_full))
        return _count_per_satellite_numba_masked(links_tcn, valid_mask, int(n_sats_full))

    return _count_per_satellite_numpy(links_tcn, n_sats_full, valid_mask=valid_mask)


def warm_up_link_selection_kernels() -> None:
    """Trigger JIT compilation for the rich link-selection kernels."""
    if not HAS_NUMBA:
        return

    _ = _select_topn_max_elevation_numba(
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        1,
    )
    _ = _select_top1_max_elevation_numba(
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )
    _ = _select_topn_max_elevation_numba(
        np.zeros((1, 1, 2), dtype=np.float64),
        np.zeros((2,), dtype=np.float64),
        1,
    )
    _ = _select_top1_max_elevation_numba(
        np.zeros((1, 1, 2), dtype=np.float64),
        np.zeros((2,), dtype=np.float64),
    )
    _ = _select_topn_random_eligible_numba(
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        1,
        np.uint64(1),
    )
    _ = _select_top1_random_eligible_numba(
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        np.uint64(1),
    )
    _ = _select_topn_random_eligible_numba(
        np.zeros((1, 1, 2), dtype=np.float64),
        np.zeros((2,), dtype=np.float64),
        1,
        np.uint64(1),
    )
    _ = _select_top1_random_eligible_numba(
        np.zeros((1, 1, 2), dtype=np.float64),
        np.zeros((2,), dtype=np.float64),
        np.uint64(1),
    )
    _links_dummy = np.asarray([[[0, -1]]], dtype=np.int32)
    _mask_dummy = np.asarray([[[True, False]]], dtype=np.bool_)
    _links_top1_dummy = np.asarray([[0, -1]], dtype=np.int32)
    _mask_top1_dummy = np.asarray([[True, False]], dtype=np.bool_)
    _ = _count_per_satellite_numba_nomask(_links_dummy, 2)
    _ = _count_per_satellite_numba_masked(_links_dummy, _mask_dummy, 2)
    _out_dummy = np.zeros((1, 2), dtype=np.int32)
    _accumulate_per_satellite_numba_nomask(_links_dummy, _out_dummy)
    _accumulate_per_satellite_numba_masked(_links_dummy, _mask_dummy, _out_dummy)
    _accumulate_per_satellite_top1_numba_nomask(_links_top1_dummy, _out_dummy)
    _accumulate_per_satellite_top1_numba_masked(_links_top1_dummy, _mask_top1_dummy, _out_dummy)
    _dummy_links = np.asarray([[[0]]], dtype=np.int32)
    _dummy_links_top1 = np.asarray([[0]], dtype=np.int32)
    _dummy_az = np.asarray([[0.0, 10.0]], dtype=np.float32)
    _dummy_el = np.asarray([[45.0, 50.0]], dtype=np.float32)
    _dummy_azel = np.asarray([[[[30.0, 4.0], [40.0, 5.0]]]], dtype=np.float32)
    _dummy_belt = np.asarray([0, 1], dtype=np.int16)
    _dummy_beta_max = np.asarray([10.0, 10.0], dtype=np.float32)
    _ = _gather_selected_link_data_numba(
        _dummy_links,
        _dummy_az,
        _dummy_el,
        _dummy_azel,
        _dummy_belt,
        _dummy_beta_max,
        np.float32(0.0),
    )
    _ = _gather_selected_link_data_top1_numba(
        _dummy_links_top1,
        _dummy_az,
        _dummy_el,
        _dummy_azel,
        _dummy_belt,
        _dummy_beta_max,
        np.float32(0.0),
    )
# ===========================================================================
# Public API
# ===========================================================================
def set_num_threads(n_threads: int) -> None:
    """
    Set *how many* CPU threads the parallel kernels in this module will use.

    Parameters
    ----------
    n_threads : int
        The desired size of the OpenMP thread pool (must be **≥ 1**).

    What this does behind the curtain
    ---------------------------------
    * If Numba **is installed**, the call forwards to
      :pyfunc:`numba.set_num_threads`, which tells the underlying OpenMP
      runtime how many worker threads it may spawn.  The change is
      **process-wide** and affects all Numba functions compiled with the
      ``parallel=True`` flag (including the kernels below).
    * If Numba is **not installed**, the helper silently accepts the call
      but has no practical effect because the code is already running in
      single-thread mode.

    Notes for the user
    ------------------
    * You can also set the environment variable ``OMP_NUM_THREADS`` **before
      starting Python**; this helper is merely a convenient *in-script*
      override.
    * Asking for more threads than physical CPU cores usually just wastes
      time on context switching, so choose a sensible number (often the
      output of ``os.cpu_count()`` is a good upper limit).
    """
    if n_threads < 1:
        raise ValueError("Thread count must be at least 1.")
    nb_set_num_threads(n_threads)


def select_satellite_links(
    sat_topo: np.ndarray,
    *,
    min_elevation_deg: float | u.Quantity | np.ndarray = 30 * u.deg,
    n_links: int = 1,
    n_beam: int | None = None,
    strategy: str = "random",
    cell_observer_offset: int = 0,
    sat_azel: np.ndarray | None = None,
    beta_max_deg_per_sat: float | u.Quantity | np.ndarray | None = None,
    sat_belt_id_per_sat: np.ndarray | None = None,
    ras_topo: np.ndarray | None = None,
    cell_active_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    include_counts: bool = True,
    include_payload: bool = True,
    include_eligible_mask: bool = False,
    eligible_mask_encoding: str = "dense",
    boresight_pointing_azimuth_deg: np.ndarray | None = None,
    boresight_pointing_elevation_deg: np.ndarray | None = None,
    boresight_theta1_deg: float | u.Quantity | None = None,
    boresight_theta2_deg: float | u.Quantity | None = None,
    boresight_theta2_cell_ids: np.ndarray | None = None,
    beta_tol_deg: float = 1e-3,
    prefer_numba: bool = True,
) -> dict[str, np.ndarray]:
    """
    Select serving satellites and optionally gather rich per-link diagnostics.

    Parameters
    ----------
    sat_topo : np.ndarray
        Satellite-to-observer topocentric geometry. Accepted shapes are
        ``(T, O, S, K)`` and ``(O, S, K)`` where ``T`` is the number of
        timesteps, ``O`` the number of observers, ``S`` the number of
        satellites, and the last axis contains at least azimuth and elevation in
        degrees. When ``cell_observer_offset > 0`` the first
        ``cell_observer_offset`` observers are treated as non-cell observers and
        skipped during link selection.
    min_elevation_deg : float, astropy.units.Quantity, or np.ndarray, optional
        Minimum serving elevation threshold in degrees. Scalars apply to every
        satellite; one-dimensional arrays must have length ``S`` and provide a
        per-satellite threshold.
    n_links : int, optional
        Number of concurrent serving links to retain per cell. The returned
        assignment axis always has length ``n_links`` and uses ``-1`` sentinels
        for unfilled slots.
    n_beam : int or None, optional
        Per-satellite serving-capacity limit applied independently at each
        timestep. ``None`` preserves the existing uncapped row-wise selector.
        Finite values trigger the exact global greedy allocator that resolves
        oversubscription across cells.
    strategy : {"random", "max_elevation"}, optional
        Link ranking strategy. ``"random"`` selects uniformly from the eligible
        satellites for each ``(time, cell)`` row using the supplied random
        generator when ``n_beam is None``. Finite ``n_beam`` uses deterministic
        per-pair priorities so chunked libraries and one-shot runs produce the
        same capped result. ``"max_elevation"`` sorts eligible satellites by
        descending elevation.
    cell_observer_offset : int, optional
        Number of leading observers in ``sat_topo`` that are not candidate
        cells. This is typically ``1`` when the first observer is the RAS site
        and the remaining observers are Earth-grid cells.
    sat_azel : np.ndarray, optional
        Satellite-body-frame pointing geometry aligned with ``sat_topo``.
        Accepted shapes match ``sat_topo`` and the last axis must contain at
        least antenna azimuth and off-axis angle ``theta`` in degrees. Required
        when ``include_payload=True`` or when
        ``beta_max_deg_per_sat`` is provided.
    beta_max_deg_per_sat : float, astropy.units.Quantity, or np.ndarray, optional
        Maximum allowed off-axis angle ``beta`` in degrees for cone validation.
        Scalars apply to every satellite; one-dimensional arrays must have
        length ``S``. When omitted, all selected links pass the cone test.
    sat_belt_id_per_sat : np.ndarray, optional
        Integer belt identifier for each satellite. Scalars broadcast to all
        satellites. Invalid or cone-pruned links use the sentinel ``-1`` in the
        returned payload.
    ras_topo : np.ndarray, optional
        RAS-observer topocentric geometry for the satellites. Accepted shapes
        are ``(T, S, K)``, ``(S, K)``, and ``(T, O, S, K)``. When a full
        observer cube is supplied, only observer index 0 is used. If omitted
        and ``cell_observer_offset >= 1``, the helper falls back to observer 0
        from ``sat_topo``.
    cell_active_mask : np.ndarray, optional
        Boolean demand mask aligned with the compact post-offset cell axis.
        Accepted shapes are ``(T, C)`` and ``(C,)``. Inactive cells are
        treated as having no demand, so assignments stay ``-1`` and
        eligibility/count outputs are suppressed for those rows.
    rng : np.random.Generator, optional
        Random generator used only for ``selection_strategy="random"``. Supply
        a seeded generator for reproducible assignments.
    include_counts : bool, optional
        If ``True``, include per-satellite demand and cone-eligible beam counts
        with shape ``(T, S)``.
    include_payload : bool, optional
        If ``True``, gather the selected link geometry payload
        ``sat_azimuth``, ``sat_elevation``, ``sat_alpha``, ``sat_beta``,
        ``sat_belt_id``, and ``cone_ok`` with shape ``(T, C, n_links)``.
    include_eligible_mask : bool, optional
        If ``True``, include uncapped candidate eligibility for each
        ``(time, cell, satellite)`` pair.
    eligible_mask_encoding : {"dense", "csr", "both"}, optional
        Eligibility output encoding used when ``include_eligible_mask=True``.
        ``"dense"`` returns ``sat_eligible_mask`` with shape ``(T, C, S)``.
        ``"csr"`` returns sparse pure-reroute inputs via
        ``sat_eligible_csr_row_ptr`` with shape ``(T*C + 1,)`` and
        ``sat_eligible_csr_sat_idx`` with shape ``(E,)``, plus scalar metadata
        ``sat_eligible_csr_time_count``, ``sat_eligible_csr_cell_count``, and
        ``sat_eligible_csr_sat_count``. ``"both"`` returns both
        representations.
    boresight_pointing_azimuth_deg, boresight_pointing_elevation_deg : np.ndarray, optional
        Sampled RAS telescope pointings with shape ``(T, N_sky)``. Required
        only when ``boresight_theta1_deg`` or ``boresight_theta2_deg`` is
        active.
    boresight_theta1_deg, boresight_theta2_deg : float or Quantity or None, optional
        Optional hard-shutdown and local-exclusion boresight angles in
        degrees. ``None`` disables the corresponding rule.
    boresight_theta2_cell_ids : np.ndarray, optional
        Integer cell-axis indices affected by ``Theta_2``. The ids refer to
        the post-``cell_observer_offset`` cell ordering.
    beta_tol_deg : float, optional
        Absolute tolerance added to ``beta_max_deg_per_sat`` during cone
        validation.
    prefer_numba : bool, optional
        If ``True`` and Numba kernels are available, use the compiled
        selectors, counters, and gather kernels. If ``False`` the function uses
        the NumPy fallback path.

    Returns
    -------
    dict[str, np.ndarray]
        Result dictionary containing at least:

        ``"assignments"``
            Integer array of shape ``(T, C, n_links)`` or ``(C, n_links)`` for
            single-time input. ``-1`` marks missing links.

        Optional keys are added according to ``include_counts``,
        ``include_payload``, and ``include_eligible_mask``. Sparse eligibility
        outputs use a time-major CSR layout over ``(time, cell)`` rows.

    Raises
    ------
    ValueError
        Raised when the input arrays have incompatible shapes, when
        per-satellite inputs do not have length ``S``, or when the payload
        outputs are requested without the required geometry inputs.

    Notes
    -----
    The random selector preserves deterministic behavior for fixed seeds within
    each backend. The compiled path uses a row-hashed SplitMix64 scheme so that
    repeated runs with the same seeded generator reproduce the same
    ``(time, cell)`` selections without materializing full random score cubes.
    When ``n_beam`` is finite, selection becomes a per-timestep global greedy
    allocation over all visible cell-satellite pairs, and cone checks remain a
    post-selection validity test. Chunked finite-cap workflows should use
    :class:`SatelliteLinkSelectionLibrary` so results stay invariant to chunk
    size and chunk order.
    """
    if n_links <= 0:
        raise ValueError("n_links must be positive.")
    if n_beam is not None and int(n_beam) <= 0:
        raise ValueError("n_beam must be positive or None.")

    sat_topo_full, single_time_input = _ensure_pairwise_time_axis(np.asarray(sat_topo), name="sat_topo")
    if sat_topo_full.shape[-1] < 2:
        raise ValueError("sat_topo last axis must contain at least azimuth and elevation.")
    if cell_observer_offset < 0:
        raise ValueError("cell_observer_offset must be >= 0.")
    if sat_topo_full.shape[1] <= cell_observer_offset:
        raise ValueError("No cell observers remain after applying cell_observer_offset.")

    cells_view = sat_topo_full[:, cell_observer_offset:, :, :]
    T_local, n_cells, n_sats, _ = cells_view.shape
    strategy_name = _normalise_selection_strategy(strategy)
    use_numba = bool(prefer_numba and HAS_NUMBA)
    eligible_mask_encoding_name = _normalise_eligible_mask_encoding(eligible_mask_encoding)
    cell_active_mask_use = _normalise_cell_active_mask(
        cell_active_mask,
        time_count=T_local,
        cell_count=n_cells,
    )

    sat_min_elev_deg = _normalise_per_satellite_float(
        min_elevation_deg,
        n_sats,
        name="min_elevation_deg",
        default_unit=u.deg,
        dtype=np.float64,
    )
    sat_beta_max_deg = _normalise_per_satellite_float(
        beta_max_deg_per_sat,
        n_sats,
        name="beta_max_deg_per_sat",
        default_unit=u.deg,
        dtype=np.float32,
        fill_value=np.inf,
    )
    sat_belt_id = _normalise_per_satellite_int(
        sat_belt_id_per_sat,
        n_sats,
        name="sat_belt_id_per_sat",
        dtype=np.int16,
        fill_value=-1,
    )

    sat_azel_cells = None
    if sat_azel is not None:
        sat_azel_full, _ = _ensure_pairwise_time_axis(np.asarray(sat_azel), name="sat_azel")
        sat_azel_cells = _slice_cells_view(
            sat_azel_full,
            name="sat_azel",
            full_observer_count=sat_topo_full.shape[1],
            cell_observer_offset=cell_observer_offset,
        )
        if sat_azel_cells.shape[:3] != cells_view.shape[:3]:
            raise ValueError("sat_azel must align with the cell slice of sat_topo.")
        if sat_azel_cells.shape[-1] < 2:
            raise ValueError("sat_azel last axis must contain at least azimuth and theta.")

    ras_topo_use = _ensure_ras_topo(
        ras_topo,
        sat_topo_full=sat_topo_full,
        cell_observer_offset=cell_observer_offset,
        T_local=T_local,
        n_sats=n_sats,
    )

    if beta_max_deg_per_sat is not None and sat_azel_cells is None:
        raise ValueError("beta_max_deg_per_sat requires sat_azel.")
    if include_payload and sat_azel_cells is None:
        raise ValueError("include_payload=True requires sat_azel.")
    if include_eligible_mask and sat_azel_cells is None:
        raise ValueError("include_eligible_mask=True requires sat_azel.")
    if include_payload and ras_topo_use is None:
        raise ValueError("include_payload=True requires ras_topo or at least one skipped observer.")

    boresight_theta1_value = _normalise_boresight_theta_deg(
        boresight_theta1_deg,
        name="boresight_theta1_deg",
    )
    boresight_theta2_value = _normalise_boresight_theta_deg(
        boresight_theta2_deg,
        name="boresight_theta2_deg",
    )
    boresight_active = boresight_theta1_value is not None or boresight_theta2_value is not None

    if n_beam is not None or boresight_active:
        finite_library = SatelliteLinkSelectionLibrary(
            time_count=T_local,
            cell_count=n_cells,
            sat_count=n_sats,
            min_elevation_deg=sat_min_elev_deg,
            n_links=int(n_links),
            n_beam=int(n_beam) if n_beam is not None else int(max(1, n_cells * int(n_links))),
            strategy=strategy_name,
            sat_belt_id_per_sat=sat_belt_id,
            beta_max_deg_per_sat=sat_beta_max_deg if beta_max_deg_per_sat is not None else None,
            ras_topo=ras_topo_use,
            cell_active_mask=cell_active_mask_use,
            rng=rng,
            include_counts=include_counts,
            include_payload=include_payload,
            include_eligible_mask=include_eligible_mask,
            eligible_mask_encoding=eligible_mask_encoding_name,
            boresight_pointing_azimuth_deg=boresight_pointing_azimuth_deg,
            boresight_pointing_elevation_deg=boresight_pointing_elevation_deg,
            boresight_theta1_deg=boresight_theta1_value,
            boresight_theta2_deg=boresight_theta2_value,
            boresight_theta2_cell_ids=boresight_theta2_cell_ids,
            beta_tol_deg=float(beta_tol_deg),
        )
        finite_library.add_chunk(0, cells_view, sat_azel=sat_azel_cells)
        return _squeeze_time_axis(finite_library.finalize(), single_time_input)

    local_rng = np.random.default_rng() if rng is None else rng
    elev_deg = cells_view[..., 1]
    elev_deg_for_selection = elev_deg
    if cell_active_mask_use is not None:
        elev_deg_for_selection = elev_deg.copy()
        elev_deg_for_selection[~cell_active_mask_use] = -np.inf

    if strategy_name == "max_elevation":
        if use_numba:
            if n_links == 1:
                sat_top1_chunk = _select_top1_max_elevation_numba(elev_deg_for_selection, sat_min_elev_deg)
                sat_cell_links = sat_top1_chunk[:, :, None]
            else:
                sat_cell_links = _select_topn_max_elevation_numba(elev_deg_for_selection, sat_min_elev_deg, int(n_links))
        else:
            sat_cell_links = _select_topn_numpy_chunk(
                elev_deg_for_selection,
                sat_min_elev_deg,
                int(n_links),
                selection_mode="max_elevation",
                rng=local_rng,
            )
    else:
        if use_numba:
            seed_u64 = np.uint64(local_rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
            if n_links == 1:
                sat_top1_chunk = _select_top1_random_eligible_numba(elev_deg_for_selection, sat_min_elev_deg, seed_u64)
                sat_cell_links = sat_top1_chunk[:, :, None]
            else:
                sat_cell_links = _select_topn_random_eligible_numba(
                    elev_deg_for_selection,
                    sat_min_elev_deg,
                    int(n_links),
                    seed_u64,
                )
        else:
            sat_cell_links = _select_topn_numpy_chunk(
                elev_deg_for_selection,
                sat_min_elev_deg,
                int(n_links),
                selection_mode="random",
                rng=local_rng,
            )
    if cell_active_mask_use is not None:
        sat_cell_links[~cell_active_mask_use] = np.int32(-1)

    result: dict[str, np.ndarray] = {"assignments": sat_cell_links}
    if include_eligible_mask:
        eligible_mask = elev_deg >= sat_min_elev_deg[None, None, :]
        if cell_active_mask_use is not None:
            eligible_mask &= cell_active_mask_use[:, :, None]
        if beta_max_deg_per_sat is not None:
            theta_all = sat_azel_cells[..., 1].astype(np.float32, copy=False)
            beta_all = np.abs(theta_all).astype(np.float32, copy=False)
            eligible_mask &= beta_all <= (
                sat_beta_max_deg[None, None, :] + np.float32(beta_tol_deg)
            )
        _attach_eligible_mask_outputs(
            result,
            dense_mask=eligible_mask.astype(np.bool_, copy=False),
            eligible_time=None,
            eligible_cell=None,
            eligible_sat=None,
            time_count=T_local,
            cell_count=n_cells,
            sat_count=n_sats,
            encoding=eligible_mask_encoding_name,
        )

    beam_counts_demand = np.zeros((T_local, n_sats), dtype=np.int32) if include_counts else None
    beam_counts_eligible = np.zeros((T_local, n_sats), dtype=np.int32) if include_counts else None

    if include_counts:
        if use_numba and n_links == 1:
            _accumulate_per_satellite_top1_numba_nomask(sat_cell_links[:, :, 0], beam_counts_demand)
        elif use_numba:
            _accumulate_per_satellite_numba_nomask(sat_cell_links, beam_counts_demand)
        else:
            beam_counts_demand[:] = _count_per_satellite(
                sat_cell_links,
                n_sats,
                prefer_numba=False,
            )

    if include_payload:
        sat_azimuth_all = np.remainder(ras_topo_use[..., 0], 360.0).astype(np.float32, copy=False)
        sat_elevation_all = ras_topo_use[..., 1].astype(np.float32, copy=False)

        if use_numba and n_links == 1:
            sat_top1_view = sat_cell_links[:, :, 0]
            (
                sat_azimuth,
                sat_elevation,
                sat_alpha,
                sat_beta,
                sat_belt_id_out,
                cone_ok_top1,
            ) = _gather_selected_link_data_top1_numba(
                sat_top1_view,
                sat_azimuth_all,
                sat_elevation_all,
                sat_azel_cells,
                sat_belt_id,
                sat_beta_max_deg,
                np.float32(beta_tol_deg),
            )
            result["sat_azimuth"] = sat_azimuth[:, :, None]
            result["sat_elevation"] = sat_elevation[:, :, None]
            result["sat_alpha"] = sat_alpha[:, :, None]
            result["sat_beta"] = sat_beta[:, :, None]
            result["sat_belt_id"] = sat_belt_id_out[:, :, None]
            result["cone_ok"] = cone_ok_top1[:, :, None]

            if include_counts and beam_counts_eligible is not None:
                _accumulate_per_satellite_top1_numba_masked(
                    sat_top1_view,
                    cone_ok_top1,
                    beam_counts_eligible,
                )
        elif use_numba:
            (
                sat_azimuth,
                sat_elevation,
                sat_alpha,
                sat_beta,
                sat_belt_id_out,
                cone_ok,
            ) = _gather_selected_link_data_numba(
                sat_cell_links,
                sat_azimuth_all,
                sat_elevation_all,
                sat_azel_cells,
                sat_belt_id,
                sat_beta_max_deg,
                np.float32(beta_tol_deg),
            )
            result["sat_azimuth"] = sat_azimuth
            result["sat_elevation"] = sat_elevation
            result["sat_alpha"] = sat_alpha
            result["sat_beta"] = sat_beta
            result["sat_belt_id"] = sat_belt_id_out
            result["cone_ok"] = cone_ok

            if include_counts and beam_counts_eligible is not None:
                _accumulate_per_satellite_numba_masked(sat_cell_links, cone_ok, beam_counts_eligible)
        else:
            no_link_mask = sat_cell_links < 0
            idx_safe = np.where(no_link_mask, np.int32(0), sat_cell_links).astype(np.int32, copy=False)
            sat_antenna_az = sat_azel_cells[..., 0].astype(np.float32, copy=False)
            sat_antenna_theta = sat_azel_cells[..., 1].astype(np.float32, copy=False)

            sat_azimuth = np.take_along_axis(sat_azimuth_all[:, None, :], idx_safe, axis=2).astype(
                np.float32, copy=False
            )
            sat_elevation = np.take_along_axis(sat_elevation_all[:, None, :], idx_safe, axis=2).astype(
                np.float32, copy=False
            )
            sat_azimuth = np.remainder(sat_azimuth, np.float32(360.0)).astype(np.float32, copy=False)
            sat_alpha = np.take_along_axis(sat_antenna_az, idx_safe, axis=2).astype(np.float32, copy=False)
            sat_alpha = np.remainder(sat_alpha, np.float32(360.0)).astype(np.float32, copy=False)
            selected_theta = np.take_along_axis(sat_antenna_theta, idx_safe, axis=2).astype(
                np.float32, copy=False
            )
            sat_beta = np.abs(selected_theta).astype(np.float32, copy=False)
            sat_belt_id_out = np.take_along_axis(
                sat_belt_id[None, None, :],
                idx_safe,
                axis=2,
            ).astype(np.int16, copy=False)
            beta_max = np.take_along_axis(
                sat_beta_max_deg[None, None, :],
                idx_safe,
                axis=2,
            ).astype(np.float32, copy=False)

            cone_ok = (sat_beta <= (beta_max + np.float32(beta_tol_deg))) & (~no_link_mask)
            sat_azimuth[~cone_ok] = np.nan
            sat_elevation[~cone_ok] = np.nan
            sat_alpha[~cone_ok] = np.nan
            sat_beta[~cone_ok] = np.nan
            sat_belt_id_out[~cone_ok] = np.int16(-1)

            result["sat_azimuth"] = sat_azimuth
            result["sat_elevation"] = sat_elevation
            result["sat_alpha"] = sat_alpha
            result["sat_beta"] = sat_beta
            result["sat_belt_id"] = sat_belt_id_out
            result["cone_ok"] = cone_ok

            if include_counts and beam_counts_eligible is not None:
                beam_counts_eligible[:] = _count_per_satellite(
                    sat_cell_links,
                    n_sats,
                    valid_mask=cone_ok,
                    prefer_numba=False,
                )
    elif include_counts and beam_counts_eligible is not None:
        beam_counts_eligible[:] = beam_counts_demand

    if include_counts:
        result["sat_beam_counts_demand"] = beam_counts_demand
        result["sat_beam_counts_eligible"] = beam_counts_eligible

    return _squeeze_time_axis(result, single_time_input)


def summarize_link_selection(
    link_result: dict[str, np.ndarray],
    *,
    n_belts: int | None = None,
) -> dict[str, Any]:
    """
    Derive compact selection statistics from a link-selection result payload.

    Parameters
    ----------
    link_result : dict[str, np.ndarray]
        Result dictionary returned by :func:`select_satellite_links`, or any
        mapping exposing equivalent arrays such as ``assignments``,
        ``sat_beam_counts_demand``, ``sat_beam_counts_eligible``, and optionally
        ``sat_belt_id``.
    n_belts : int | None, optional
        Number of belts expected in the output histogram. When provided and
        ``sat_belt_id`` is available, the returned dictionary includes
        ``belt_hist`` with shape ``(n_belts,)``.

    Returns
    -------
    dict[str, Any]
        Summary dictionary with:

        ``selected_links``
            Total number of non-sentinel link assignments.
        ``cone_ok_links``
            Total number of links that survived cone validation.
        ``frac_ok``
            Ratio ``cone_ok_links / selected_links``. Returns ``1.0`` when
            ``selected_links == 0``.
        ``belt_hist``
            Optional ``np.ndarray[int64]`` of length ``n_belts`` describing
            cone-valid links by belt identifier.

    Raises
    ------
    KeyError
        If the summary cannot derive ``selected_links`` because neither
        ``sat_beam_counts_demand`` nor ``assignments`` is present, or if
        ``n_belts`` is requested without ``sat_belt_id``.

    Notes
    -----
    This helper is intended for lightweight front-end diagnostics. It prefers
    the already-aggregated per-satellite count arrays because summing
    ``(T, S)`` counts is cheaper than rescanning the full ``(T, C, n_links)``
    assignment cube after each propagation chunk. When the counts are absent it
    falls back to the assignment payload so the helper remains usable in tests
    and simpler callers.
    """
    summary: dict[str, Any] = {}

    demand = link_result.get("sat_beam_counts_demand")
    if demand is not None:
        selected_links = int(np.sum(np.asarray(demand), dtype=np.int64))
    else:
        assignments = link_result.get("assignments")
        if assignments is None:
            raise KeyError(
                "link_result must contain either 'sat_beam_counts_demand' or 'assignments'."
            )
        selected_links = int(np.count_nonzero(np.asarray(assignments) >= 0))

    eligible = link_result.get("sat_beam_counts_eligible")
    if eligible is not None:
        cone_ok_links = int(np.sum(np.asarray(eligible), dtype=np.int64))
    else:
        sat_belt_id = link_result.get("sat_belt_id")
        if sat_belt_id is not None:
            cone_ok_links = int(np.count_nonzero(np.asarray(sat_belt_id) >= 0))
        else:
            cone_ok_links = selected_links

    summary["selected_links"] = selected_links
    summary["cone_ok_links"] = cone_ok_links
    summary["frac_ok"] = 1.0 if selected_links == 0 else float(cone_ok_links / selected_links)

    if n_belts is not None:
        sat_belt_id = link_result.get("sat_belt_id")
        if sat_belt_id is None:
            raise KeyError("link_result must contain 'sat_belt_id' when n_belts is provided.")
        belt_ids = np.asarray(sat_belt_id, dtype=np.int64)
        valid_belt_ids = belt_ids[belt_ids >= 0]
        summary["belt_hist"] = np.bincount(valid_belt_ids, minlength=int(n_belts)).astype(
            np.int64, copy=False
        )

    return summary


def _normalise_pure_reroute_mask(
    eligible_mask: Any,
) -> tuple[np.ndarray, bool]:
    mask = np.asarray(eligible_mask, dtype=np.bool_)
    if mask.ndim == 2:
        return mask[np.newaxis, ...], True
    if mask.ndim != 3:
        raise ValueError(
            "eligible_mask must have shape (T, C, S) or (C, S); "
            f"got {mask.shape!r}."
        )
    return mask, False


def _normalise_contiguous_beam_caps(beam_caps: Any) -> np.ndarray:
    caps = np.asarray(beam_caps, dtype=np.int32).reshape(-1)
    if caps.size == 0:
        raise ValueError("beam_caps must be a non-empty one-dimensional array.")
    if int(caps[0]) != 0:
        raise ValueError("beam_caps must start at 0.")
    expected = np.arange(int(caps[0]), int(caps[0]) + caps.size, dtype=np.int32)
    if not np.array_equal(caps, expected):
        raise ValueError("beam_caps must be contiguous integer values.")
    return caps


def _normalise_pure_reroute_input(
    eligible_mask: Any,
) -> tuple[dict[str, Any], bool]:
    if isinstance(eligible_mask, dict) and (
        PURE_REROUTE_CSR_ROW_PTR_KEY in eligible_mask
        and PURE_REROUTE_CSR_SAT_IDX_KEY in eligible_mask
    ):
        row_ptr = np.asarray(eligible_mask[PURE_REROUTE_CSR_ROW_PTR_KEY], dtype=np.int64).reshape(-1)
        sat_idx = np.asarray(eligible_mask[PURE_REROUTE_CSR_SAT_IDX_KEY], dtype=np.int32).reshape(-1)
        try:
            time_count = int(eligible_mask[PURE_REROUTE_CSR_TIME_COUNT_KEY])
            cell_count = int(eligible_mask[PURE_REROUTE_CSR_CELL_COUNT_KEY])
            sat_count = int(eligible_mask[PURE_REROUTE_CSR_SAT_COUNT_KEY])
        except KeyError as exc:
            raise ValueError(
                "CSR pure_reroute input must include time, cell, and satellite counts."
            ) from exc
        if time_count < 0 or cell_count < 0 or sat_count < 0:
            raise ValueError("CSR pure_reroute shape metadata must be non-negative.")
        expected_rows = int(time_count) * int(cell_count) + 1
        if row_ptr.size != expected_rows:
            raise ValueError(
                f"{PURE_REROUTE_CSR_ROW_PTR_KEY} must have length {expected_rows}, got {row_ptr.size}."
            )
        if row_ptr.size == 0 or int(row_ptr[0]) != 0:
            raise ValueError(f"{PURE_REROUTE_CSR_ROW_PTR_KEY} must start at 0.")
        if np.any(row_ptr[1:] < row_ptr[:-1]):
            raise ValueError(f"{PURE_REROUTE_CSR_ROW_PTR_KEY} must be non-decreasing.")
        if int(row_ptr[-1]) != sat_idx.size:
            raise ValueError(
                f"{PURE_REROUTE_CSR_ROW_PTR_KEY} terminal value {int(row_ptr[-1])} does not "
                f"match {PURE_REROUTE_CSR_SAT_IDX_KEY} length {sat_idx.size}."
            )
        if np.any(sat_idx < 0) or np.any(sat_idx >= int(sat_count)):
            raise ValueError(f"{PURE_REROUTE_CSR_SAT_IDX_KEY} contains out-of-range satellite ids.")
        return (
            {
                PURE_REROUTE_CSR_ROW_PTR_KEY: row_ptr.astype(np.int64, copy=False),
                PURE_REROUTE_CSR_SAT_IDX_KEY: sat_idx.astype(np.int32, copy=False),
                **_pure_reroute_csr_metadata(
                    time_count=time_count,
                    cell_count=cell_count,
                    sat_count=sat_count,
                ),
            },
            False,
        )

    mask, single_time = _normalise_pure_reroute_mask(eligible_mask)
    return _pure_reroute_dense_mask_to_csr_payload(mask), single_time


def _compact_pure_reroute_slot_csr(
    row_ptr_full: np.ndarray,
    sat_idx_full: np.ndarray,
    *,
    slot_index: int,
    cell_count: int,
    sat_count: int,
    nco: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    slot_index_i = int(slot_index)
    cell_count_i = int(cell_count)
    row_start = slot_index_i * cell_count_i
    row_stop = row_start + cell_count_i
    slot_row_ptr = np.asarray(row_ptr_full[row_start : row_stop + 1], dtype=np.int64, copy=False)
    degrees = np.diff(slot_row_ptr).astype(np.int32, copy=False)
    active_rows = np.flatnonzero(degrees > 0)
    if active_rows.size == 0:
        return None

    active_degrees = degrees[active_rows].astype(np.int32, copy=False)
    row_ptr = np.empty(active_rows.size + 1, dtype=np.int32)
    row_ptr[0] = np.int32(0)
    row_ptr[1:] = np.cumsum(active_degrees, dtype=np.int32)
    edge_count = int(row_ptr[-1])
    if edge_count <= 0:
        return None

    edge_sat_global = np.empty(edge_count, dtype=np.int32)
    cursor = 0
    for active_row in active_rows.astype(np.int64, copy=False):
        start = int(slot_row_ptr[int(active_row)])
        stop = int(slot_row_ptr[int(active_row) + 1])
        degree = stop - start
        edge_sat_global[cursor : cursor + degree] = np.asarray(
            sat_idx_full[start:stop],
            dtype=np.int32,
            copy=False,
        )
        cursor += degree
    if cursor != edge_count:
        raise RuntimeError("Internal error while compacting pure-reroute CSR slot edges.")

    unique_sats, edge_sat = np.unique(edge_sat_global, return_inverse=True)
    cell_cap = np.minimum(active_degrees, np.int32(int(nco))).astype(np.int32, copy=False)
    total_units = int(np.sum(cell_cap, dtype=np.int64))
    edge_cell = np.repeat(np.arange(active_rows.size, dtype=np.int32), active_degrees)
    return (
        row_ptr,
        edge_cell.astype(np.int32, copy=False),
        np.asarray(edge_sat, dtype=np.int32),
        cell_cap,
        int(unique_sats.size),
        total_units,
    )


def _split_pure_reroute_csr_components(
    row_ptr: np.ndarray,
    edge_cell: np.ndarray,
    edge_sat: np.ndarray,
    cell_cap: np.ndarray,
    sat_count: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]:
    row_ptr_i32 = np.asarray(row_ptr, dtype=np.int32, copy=False)
    edge_cell_i32 = np.asarray(edge_cell, dtype=np.int32, copy=False)
    edge_sat_i32 = np.asarray(edge_sat, dtype=np.int32, copy=False)
    cell_cap_i32 = np.asarray(cell_cap, dtype=np.int32, copy=False)
    row_count = int(cell_cap_i32.size)
    sat_count_i = int(sat_count)
    if row_count <= 0 or sat_count_i <= 0 or edge_sat_i32.size == 0:
        return []

    sat_order = np.argsort(edge_sat_i32, kind="mergesort")
    sat_sorted = edge_sat_i32[sat_order].astype(np.int32, copy=False)
    cell_sorted = edge_cell_i32[sat_order].astype(np.int32, copy=False)
    sat_counts = np.bincount(
        sat_sorted.astype(np.int64, copy=False),
        minlength=sat_count_i,
    ).astype(np.int32, copy=False)
    sat_ptr = np.empty(sat_count_i + 1, dtype=np.int32)
    sat_ptr[0] = np.int32(0)
    sat_ptr[1:] = np.cumsum(sat_counts, dtype=np.int32)
    row_degrees = np.diff(row_ptr_i32).astype(np.int32, copy=False)

    seen_rows = np.zeros(row_count, dtype=np.bool_)
    seen_sats = np.zeros(sat_count_i, dtype=np.bool_)
    components: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]] = []

    for root_row in range(row_count):
        if seen_rows[root_row]:
            continue
        row_stack = [int(root_row)]
        sat_stack: list[int] = []
        row_members: list[int] = []
        sat_members: list[int] = []
        seen_rows[root_row] = True

        while row_stack or sat_stack:
            while row_stack:
                row_idx = row_stack.pop()
                row_members.append(row_idx)
                start = int(row_ptr_i32[row_idx])
                stop = int(row_ptr_i32[row_idx + 1])
                for edge_idx in range(start, stop):
                    sat_idx = int(edge_sat_i32[edge_idx])
                    if not seen_sats[sat_idx]:
                        seen_sats[sat_idx] = True
                        sat_stack.append(sat_idx)
                        sat_members.append(sat_idx)
            while sat_stack:
                sat_idx = sat_stack.pop()
                start = int(sat_ptr[sat_idx])
                stop = int(sat_ptr[sat_idx + 1])
                for pos in range(start, stop):
                    row_idx = int(cell_sorted[pos])
                    if not seen_rows[row_idx]:
                        seen_rows[row_idx] = True
                        row_stack.append(row_idx)

        if not row_members or not sat_members:
            continue

        rows = np.asarray(row_members, dtype=np.int32)
        sats = np.asarray(sat_members, dtype=np.int32)
        rows.sort()
        sats.sort()
        sat_map = np.full(sat_count_i, -1, dtype=np.int32)
        sat_map[sats] = np.arange(sats.size, dtype=np.int32)
        component_degrees = row_degrees[rows].astype(np.int32, copy=False)
        component_row_ptr = np.empty(rows.size + 1, dtype=np.int32)
        component_row_ptr[0] = np.int32(0)
        component_row_ptr[1:] = np.cumsum(component_degrees, dtype=np.int32)
        component_edge_count = int(component_row_ptr[-1])
        component_edge_sat = np.empty(component_edge_count, dtype=np.int32)
        cursor = 0
        for row_idx in rows:
            start = int(row_ptr_i32[int(row_idx)])
            stop = int(row_ptr_i32[int(row_idx) + 1])
            degree = stop - start
            component_edge_sat[cursor : cursor + degree] = sat_map[edge_sat_i32[start:stop]]
            cursor += degree
        component_edge_cell = np.repeat(np.arange(rows.size, dtype=np.int32), component_degrees)
        component_cell_cap = cell_cap_i32[rows].astype(np.int32, copy=False)
        components.append(
            (
                component_row_ptr,
                component_edge_cell.astype(np.int32, copy=False),
                component_edge_sat.astype(np.int32, copy=False),
                component_cell_cap,
                int(sats.size),
                int(np.sum(component_cell_cap, dtype=np.int64)),
            )
        )
    return components


def iter_pure_reroute_component_graphs(
    eligible_mask: Any,
    *,
    nco: int,
) -> Iterator[dict[str, Any]]:
    payload, _single_time = _normalise_pure_reroute_input(eligible_mask)
    row_ptr = np.asarray(payload[PURE_REROUTE_CSR_ROW_PTR_KEY], dtype=np.int64, copy=False)
    sat_idx = np.asarray(payload[PURE_REROUTE_CSR_SAT_IDX_KEY], dtype=np.int32, copy=False)
    time_count = int(payload[PURE_REROUTE_CSR_TIME_COUNT_KEY])
    cell_count = int(payload[PURE_REROUTE_CSR_CELL_COUNT_KEY])
    sat_count = int(payload[PURE_REROUTE_CSR_SAT_COUNT_KEY])
    for slot_index in range(time_count):
        slot_graph = _compact_pure_reroute_slot_csr(
            row_ptr,
            sat_idx,
            slot_index=slot_index,
            cell_count=cell_count,
            sat_count=sat_count,
            nco=int(nco),
        )
        if slot_graph is None:
            continue
        split_components = _split_pure_reroute_csr_components(
            slot_graph[0],
            slot_graph[1],
            slot_graph[2],
            slot_graph[3],
            slot_graph[4],
        )
        for component in split_components:
            yield {
                "slot_index": int(slot_index),
                "row_ptr": component[0],
                "edge_cell": component[1],
                "edge_sat": component[2],
                "cell_cap": component[3],
                "sat_count": int(component[4]),
                "eligible_demand": int(component[5]),
            }


def _build_exact_pure_reroute_graph(
    slot_mask: np.ndarray,
    *,
    nco: int,
) -> tuple[list[list[list[int]]], list[int], list[int], int, list[int]]:
    active_cells = np.flatnonzero(np.any(slot_mask, axis=1))
    active_sats = np.flatnonzero(np.any(slot_mask[active_cells], axis=0)) if active_cells.size else np.empty(0, dtype=np.int32)
    if active_cells.size == 0 or active_sats.size == 0:
        return [], [], [], 0, []

    sat_index_map = np.full(slot_mask.shape[1], -1, dtype=np.int32)
    sat_index_map[active_sats] = np.arange(active_sats.size, dtype=np.int32)

    adjacency: list[list[list[int]]] = []
    cell_caps: list[int] = []
    total_units = 0
    for cell in active_cells.astype(np.int32, copy=False):
        sat_local = sat_index_map[np.flatnonzero(slot_mask[int(cell)])].astype(np.int32, copy=False)
        if sat_local.size == 0:
            continue
        cell_cap = min(int(nco), int(sat_local.size))
        if cell_cap <= 0:
            continue
        adjacency.append([sat_local.tolist() for _ in range(cell_cap)])
        cell_caps.append(cell_cap)
        total_units += cell_cap

    return adjacency, cell_caps, active_sats.astype(int).tolist(), total_units, active_cells.astype(int).tolist()


if HAS_NUMBA:
    @njit(cache=True)
    def _build_exact_pure_reroute_csr_numba(
        slot_mask: np.ndarray,
        nco: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        cell_count = int(slot_mask.shape[0])
        sat_count = int(slot_mask.shape[1])
        cell_degree = np.zeros(cell_count, dtype=np.int32)
        sat_seen = np.zeros(sat_count, dtype=np.uint8)
        active_cell_count = 0
        edge_count = 0

        for cell_idx in range(cell_count):
            degree = 0
            for sat_idx in range(sat_count):
                if slot_mask[cell_idx, sat_idx]:
                    degree += 1
                    sat_seen[sat_idx] = np.uint8(1)
            cell_degree[cell_idx] = np.int32(degree)
            if degree > 0:
                active_cell_count += 1
                edge_count += degree

        if active_cell_count <= 0 or edge_count <= 0:
            return (
                np.zeros(1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                0,
                0,
            )

        sat_index_map = np.full(sat_count, -1, dtype=np.int32)
        active_sat_count = 0
        for sat_idx in range(sat_count):
            if sat_seen[sat_idx] != 0:
                sat_index_map[sat_idx] = np.int32(active_sat_count)
                active_sat_count += 1

        row_ptr = np.empty(active_cell_count + 1, dtype=np.int32)
        edge_cell = np.empty(edge_count, dtype=np.int32)
        edge_sat = np.empty(edge_count, dtype=np.int32)
        cell_cap = np.empty(active_cell_count, dtype=np.int32)

        row_ptr[0] = np.int32(0)
        active_row = 0
        edge_cursor = 0
        total_units = 0
        for cell_idx in range(cell_count):
            degree = int(cell_degree[cell_idx])
            if degree <= 0:
                continue
            cap = degree if degree < int(nco) else int(nco)
            cell_cap[active_row] = np.int32(cap)
            total_units += cap
            for sat_idx in range(sat_count):
                if slot_mask[cell_idx, sat_idx]:
                    edge_cell[edge_cursor] = np.int32(active_row)
                    edge_sat[edge_cursor] = sat_index_map[sat_idx]
                    edge_cursor += 1
            row_ptr[active_row + 1] = np.int32(edge_cursor)
            active_row += 1

        return row_ptr, edge_cell, edge_sat, cell_cap, active_sat_count, total_units


    @njit(cache=True)
    def _pure_reroute_slot_service_curve_exact_numba_from_csr(
        row_ptr: np.ndarray,
        edge_cell: np.ndarray,
        edge_sat: np.ndarray,
        cell_cap: np.ndarray,
        sat_count: int,
        total_units: int,
        beam_caps: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        matched_by_cap = np.zeros(beam_caps.size, dtype=np.int32)
        if total_units <= 0 or sat_count <= 0:
            return matched_by_cap, 0, 0

        cell_count = int(cell_cap.size)
        edge_count = int(edge_sat.size)
        cell_load = np.zeros(cell_count, dtype=np.int32)
        parent_edge_cell = np.empty(cell_count, dtype=np.int32)
        queue = np.empty(cell_count, dtype=np.int32)
        sat_load = np.zeros(sat_count, dtype=np.int32)
        sat_head = np.full(sat_count, -1, dtype=np.int32)
        parent_edge_sat = np.empty(sat_count, dtype=np.int32)
        edge_matched = np.zeros(edge_count, dtype=np.int8)
        edge_next = np.full(edge_count, -1, dtype=np.int32)
        edge_prev = np.full(edge_count, -1, dtype=np.int32)

        total_flow = 0
        required_beam_cap = int(beam_caps[beam_caps.size - 1]) + 1
        current_beam_cap = 0

        for cap_idx in range(beam_caps.size):
            beam_cap = int(beam_caps[cap_idx])
            if beam_cap < current_beam_cap:
                raise ValueError("beam_caps must be sorted ascending.")
            current_beam_cap = beam_cap

            while True:
                for cell_idx in range(cell_count):
                    parent_edge_cell[cell_idx] = np.int32(-2)
                for sat_idx in range(sat_count):
                    parent_edge_sat[sat_idx] = np.int32(-1)

                queue_size = 0
                queue_head = 0
                for cell_idx in range(cell_count):
                    if int(cell_cap[cell_idx]) > int(cell_load[cell_idx]):
                        parent_edge_cell[cell_idx] = np.int32(-1)
                        queue[queue_size] = np.int32(cell_idx)
                        queue_size += 1

                found_sat = -1
                while queue_head < queue_size and found_sat < 0:
                    cell_idx = int(queue[queue_head])
                    queue_head += 1
                    row_start = int(row_ptr[cell_idx])
                    row_stop = int(row_ptr[cell_idx + 1])
                    for edge_idx in range(row_start, row_stop):
                        if int(edge_matched[edge_idx]) != 0:
                            continue
                        sat_idx = int(edge_sat[edge_idx])
                        if int(parent_edge_sat[sat_idx]) != -1:
                            continue
                        parent_edge_sat[sat_idx] = np.int32(edge_idx)
                        if int(sat_load[sat_idx]) < beam_cap:
                            found_sat = sat_idx
                            break

                        matched_edge = int(sat_head[sat_idx])
                        while matched_edge >= 0:
                            matched_cell = int(edge_cell[matched_edge])
                            if int(parent_edge_cell[matched_cell]) == -2:
                                parent_edge_cell[matched_cell] = np.int32(matched_edge)
                                queue[queue_size] = np.int32(matched_cell)
                                queue_size += 1
                            matched_edge = int(edge_next[matched_edge])
                    if found_sat >= 0:
                        break

                if found_sat < 0:
                    break

                current_sat = found_sat
                while True:
                    add_edge = int(parent_edge_sat[current_sat])
                    add_cell = int(edge_cell[add_edge])

                    edge_matched[add_edge] = np.int8(1)
                    old_head = int(sat_head[current_sat])
                    edge_prev[add_edge] = np.int32(-1)
                    edge_next[add_edge] = np.int32(old_head)
                    if old_head >= 0:
                        edge_prev[old_head] = np.int32(add_edge)
                    sat_head[current_sat] = np.int32(add_edge)
                    sat_load[current_sat] = np.int32(int(sat_load[current_sat]) + 1)
                    cell_load[add_cell] = np.int32(int(cell_load[add_cell]) + 1)

                    remove_edge = int(parent_edge_cell[add_cell])
                    if remove_edge < 0:
                        total_flow += 1
                        break

                    previous_sat = int(edge_sat[remove_edge])
                    previous_edge = int(edge_prev[remove_edge])
                    next_edge = int(edge_next[remove_edge])
                    if previous_edge >= 0:
                        edge_next[previous_edge] = np.int32(next_edge)
                    else:
                        sat_head[previous_sat] = np.int32(next_edge)
                    if next_edge >= 0:
                        edge_prev[next_edge] = np.int32(previous_edge)
                    edge_prev[remove_edge] = np.int32(-1)
                    edge_next[remove_edge] = np.int32(-1)
                    edge_matched[remove_edge] = np.int8(0)
                    sat_load[previous_sat] = np.int32(int(sat_load[previous_sat]) - 1)
                    cell_load[add_cell] = np.int32(int(cell_load[add_cell]) - 1)
                    current_sat = previous_sat

            matched_by_cap[cap_idx] = np.int32(total_flow)
            if total_flow >= total_units and required_beam_cap > int(beam_caps[beam_caps.size - 1]):
                required_beam_cap = beam_cap
                for fill_idx in range(cap_idx + 1, beam_caps.size):
                    matched_by_cap[fill_idx] = np.int32(total_flow)
                break

        return matched_by_cap, total_units, required_beam_cap


    @njit(cache=True)
    def _pure_reroute_slot_service_curve_exact_numba(
        slot_mask: np.ndarray,
        nco: int,
        beam_caps: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        row_ptr, edge_cell, edge_sat, cell_cap, sat_count, total_units = _build_exact_pure_reroute_csr_numba(
            slot_mask,
            nco,
        )
        return _pure_reroute_slot_service_curve_exact_numba_from_csr(
            row_ptr,
            edge_cell,
            edge_sat,
            cell_cap,
            sat_count,
            total_units,
            beam_caps,
        )


def _add_flow_edge(
    graph: list[list[list[int]]],
    fr: int,
    to: int,
    cap: int,
) -> int:
    graph[fr].append([to, int(cap), len(graph[to])])
    graph[to].append([fr, 0, len(graph[fr]) - 1])
    return len(graph[fr]) - 1


def _dinic_bfs(
    graph: list[list[list[int]]],
    source: int,
    sink: int,
    level: list[int],
) -> bool:
    for idx in range(len(level)):
        level[idx] = -1
    level[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for to, cap, _rev in graph[node]:
            if cap <= 0 or level[to] >= 0:
                continue
            level[to] = level[node] + 1
            if to == sink:
                return True
            queue.append(to)
    return level[sink] >= 0


def _dinic_dfs(
    graph: list[list[list[int]]],
    node: int,
    sink: int,
    pushed: int,
    level: list[int],
    it: list[int],
) -> int:
    if pushed <= 0:
        return 0
    if node == sink:
        return pushed
    while it[node] < len(graph[node]):
        edge_idx = it[node]
        to, cap, rev = graph[node][edge_idx]
        if cap > 0 and level[to] == level[node] + 1:
            flow = _dinic_dfs(graph, to, sink, min(pushed, cap), level, it)
            if flow > 0:
                graph[node][edge_idx][1] -= flow
                graph[to][rev][1] += flow
                return flow
        it[node] += 1
    return 0


def _dinic_max_flow(
    graph: list[list[list[int]]],
    source: int,
    sink: int,
) -> int:
    flow = 0
    level = [-1] * len(graph)
    while _dinic_bfs(graph, source, sink, level):
        it = [0] * len(graph)
        while True:
            pushed = _dinic_dfs(graph, source, sink, 1 << 30, level, it)
            if pushed <= 0:
                break
            flow += pushed
    return flow


def _pure_reroute_slot_service_curve_exact_from_csr(
    row_ptr: np.ndarray,
    edge_sat: np.ndarray,
    cell_cap: np.ndarray,
    *,
    sat_count: int,
    total_units: int,
    beam_caps: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    matched_by_cap = np.zeros(beam_caps.size, dtype=np.int32)
    if total_units <= 0 or sat_count <= 0 or cell_cap.size == 0:
        return matched_by_cap, 0, 0

    n_cell_nodes = int(cell_cap.size)
    source = 0
    cell_start = 1
    sat_start = cell_start + n_cell_nodes
    sink = sat_start + int(sat_count)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    for cell_idx, cell_cap_i in enumerate(np.asarray(cell_cap, dtype=np.int32, copy=False)):
        cell_node = cell_start + int(cell_idx)
        _add_flow_edge(graph, source, cell_node, int(cell_cap_i))
        start = int(row_ptr[cell_idx])
        stop = int(row_ptr[cell_idx + 1])
        for sat_local in np.asarray(edge_sat[start:stop], dtype=np.int32, copy=False):
            _add_flow_edge(graph, cell_node, sat_start + int(sat_local), 1)

    sat_sink_edges: list[tuple[int, int]] = []
    for sat_local in range(int(sat_count)):
        sat_node = sat_start + int(sat_local)
        edge_idx = _add_flow_edge(graph, sat_node, sink, 0)
        sat_sink_edges.append((sat_node, edge_idx))

    current_beam_cap = 0
    total_flow = 0
    required_beam_cap = int(beam_caps[-1]) + 1
    for idx, cap_val in enumerate(np.asarray(beam_caps, dtype=np.int32, copy=False)):
        cap_i = int(cap_val)
        delta_cap = cap_i - current_beam_cap
        if delta_cap < 0:
            raise ValueError("beam_caps must be sorted ascending.")
        if delta_cap > 0:
            for sat_node, edge_idx in sat_sink_edges:
                graph[sat_node][edge_idx][1] += int(delta_cap)
            current_beam_cap = cap_i
            total_flow += _dinic_max_flow(graph, source, sink)
        matched_by_cap[idx] = np.int32(total_flow)
        if total_flow >= total_units and required_beam_cap > int(beam_caps[-1]):
            required_beam_cap = cap_i
            matched_by_cap[idx + 1 :] = np.int32(total_flow)
            break
    return matched_by_cap, total_units, required_beam_cap


def _pure_reroute_slot_service_curve_exact(
    slot_mask: np.ndarray,
    *,
    nco: int,
    beam_caps: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    row_ptr, edge_cell, edge_sat, cell_cap, sat_count, total_units = _build_exact_pure_reroute_csr_numba.py_func(  # type: ignore[attr-defined]
        np.asarray(slot_mask, dtype=np.bool_, copy=False),
        int(nco),
    ) if HAS_NUMBA else (
        None,
        None,
        None,
        None,
        0,
        0,
    )
    if HAS_NUMBA:
        return _pure_reroute_slot_service_curve_exact_from_csr(
            row_ptr,
            edge_sat,
            cell_cap,
            sat_count=sat_count,
            total_units=total_units,
            beam_caps=np.asarray(beam_caps, dtype=np.int32, copy=False),
        )

    adjacency, cell_caps, _active_sats, total_units, _active_cells = _build_exact_pure_reroute_graph(
        np.asarray(slot_mask, dtype=np.bool_, copy=False),
        nco=nco,
    )
    matched_by_cap = np.zeros(beam_caps.size, dtype=np.int32)
    if total_units <= 0:
        return matched_by_cap, 0, 0

    row_ptr_py = np.empty(len(cell_caps) + 1, dtype=np.int32)
    row_ptr_py[0] = np.int32(0)
    edge_sat_py: list[int] = []
    for idx, sat_lists in enumerate(adjacency):
        seen_sats: set[int] = set()
        for sat_list in sat_lists:
            for sat_local in sat_list:
                sat_local_i = int(sat_local)
                if sat_local_i in seen_sats:
                    continue
                seen_sats.add(sat_local_i)
                edge_sat_py.append(sat_local_i)
        row_ptr_py[idx + 1] = np.int32(len(edge_sat_py))
    return _pure_reroute_slot_service_curve_exact_from_csr(
        row_ptr_py,
        np.asarray(edge_sat_py, dtype=np.int32),
        np.asarray(cell_caps, dtype=np.int32),
        sat_count=int(np.count_nonzero(np.any(slot_mask, axis=0))),
        total_units=total_units,
        beam_caps=np.asarray(beam_caps, dtype=np.int32, copy=False),
    )


def pure_reroute_service_curve(
    eligible_mask: Any,
    *,
    nco: int,
    beam_caps: Any,
) -> dict[str, np.ndarray]:
    """
    Compute exact pure-reroute beam-cap curves from a cell-satellite mask.

    Parameters
    ----------
    eligible_mask : array-like or dict
        Exact-reroute eligibility input. Dense input uses a boolean mask with
        shape ``(T, C, S)`` or ``(C, S)`` where ``True`` means the cell can be
        served by the satellite. Sparse input uses the CSR payload keys
        ``sat_eligible_csr_row_ptr``, ``sat_eligible_csr_sat_idx``,
        ``sat_eligible_csr_time_count``, ``sat_eligible_csr_cell_count``, and
        ``sat_eligible_csr_sat_count``. The eligibility graph is assumed to
        already include any per-satellite belt elevation limits and optional
        cone filtering applied upstream.
    nco : int
        Maximum number of concurrent links requested per cell.
    beam_caps : array-like
        Contiguous beam-cap grid starting at ``0``. The function evaluates the
        exact best achievable service for every cap.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:

        ``eligible_demand``
            Per-slot maximum demand achievable under the eligibility graph,
            shape ``(T,)``.
        ``matched_links``
            Exact per-slot served-link curve under each cap, shape
            ``(T, B)``.
        ``required_beam_cap``
            Smallest beam cap that reaches each slot's eligible demand,
            shape ``(T,)``. Values larger than the evaluated grid are reported
            as ``beam_caps[-1] + 1``.
        ``delta``
            Run-level unmet eligible demand fraction for each cap, shape
            ``(B,)``.
        ``eps``
            Run-level loss-slot probability for each cap, shape ``(B,)``.
        ``tail_risk``
            Alias of ``eps`` for the exact full-coverage pure-reroute policy.
        ``draws_attempted``
            Compatibility diagnostic equal to the total eligible demand summed
            across the evaluated slots.
        ``rounds_used``
            Compatibility diagnostic for downstream callers. The exact solver
            is a one-pass algorithm, so this is ``1`` when any slot is
            evaluated and ``0`` otherwise.

    Raises
    ------
    ValueError
        If ``nco`` is not positive, if the mask shape is invalid, or if
        ``beam_caps`` is not a contiguous integer grid starting at ``0``.

    Notes
    -----
    This helper is an exact ideal lower-bound policy. It is global across all
    eligible satellites and does not impose any additional belt-local routing
    rule beyond the upstream eligibility mask itself. CSR input is preferred
    for large sparse step-1 files because it avoids dense ``(T, C, S)`` graph
    reconstruction before the exact solve.
    """
    if int(nco) <= 0:
        raise ValueError("nco must be positive.")
    payload, single_time = _normalise_pure_reroute_input(eligible_mask)
    caps = _normalise_contiguous_beam_caps(beam_caps)

    time_count = int(payload[PURE_REROUTE_CSR_TIME_COUNT_KEY])
    matched_links = np.zeros((time_count, caps.size), dtype=np.int32)
    eligible_demand = np.zeros(time_count, dtype=np.int32)
    required_beam_cap = np.zeros(time_count, dtype=np.int32)

    for component in iter_pure_reroute_component_graphs(payload, nco=int(nco)):
        if HAS_NUMBA:
            matched_slot, demand_slot, required_slot = _pure_reroute_slot_service_curve_exact_numba_from_csr(
                np.asarray(component["row_ptr"], dtype=np.int32, copy=False),
                np.asarray(component["edge_cell"], dtype=np.int32, copy=False),
                np.asarray(component["edge_sat"], dtype=np.int32, copy=False),
                np.asarray(component["cell_cap"], dtype=np.int32, copy=False),
                int(component["sat_count"]),
                int(component["eligible_demand"]),
                caps,
            )
        else:
            matched_slot, demand_slot, required_slot = _pure_reroute_slot_service_curve_exact_from_csr(
                np.asarray(component["row_ptr"], dtype=np.int32, copy=False),
                np.asarray(component["edge_sat"], dtype=np.int32, copy=False),
                np.asarray(component["cell_cap"], dtype=np.int32, copy=False),
                sat_count=int(component["sat_count"]),
                total_units=int(component["eligible_demand"]),
                beam_caps=caps,
            )
        slot_index = int(component["slot_index"])
        matched_links[slot_index] += matched_slot.astype(np.int32, copy=False)
        eligible_demand[slot_index] = np.int32(int(eligible_demand[slot_index]) + int(demand_slot))
        if int(demand_slot) > 0:
            required_beam_cap[slot_index] = np.int32(
                max(int(required_beam_cap[slot_index]), int(required_slot))
            )

    unmet = eligible_demand[:, None].astype(np.int64, copy=False) - matched_links.astype(np.int64, copy=False)
    unmet = np.maximum(unmet, 0)
    total_eligible_demand = float(np.sum(eligible_demand, dtype=np.int64))
    if total_eligible_demand > 0.0:
        delta = unmet.sum(axis=0, dtype=np.int64).astype(np.float64) / total_eligible_demand
    else:
        delta = np.zeros(caps.size, dtype=np.float64)
    if time_count > 0:
        eps = np.mean(unmet > 0, axis=0, dtype=np.float64)
    else:
        eps = np.zeros(caps.size, dtype=np.float64)

    result = {
        "eligible_demand": eligible_demand,
        "matched_links": matched_links,
        "required_beam_cap": required_beam_cap,
        "delta": delta.astype(np.float64, copy=False),
        "eps": eps.astype(np.float64, copy=False),
        "tail_risk": eps.astype(np.float64, copy=False),
        "draws_attempted": np.int64(np.sum(eligible_demand, dtype=np.int64)),
        "rounds_used": np.int32(1 if time_count > 0 else 0),
    }
    if single_time:
        result["eligible_demand"] = eligible_demand[0]
        result["matched_links"] = matched_links[0]
        result["required_beam_cap"] = required_beam_cap[0]
    return result


SAMPLER_SOURCE_INVALID = np.int8(-1)
SAMPLER_SOURCE_GROUP = np.int8(0)
SAMPLER_SOURCE_BELT = np.int8(1)
SAMPLER_SOURCE_GLOBAL = np.int8(2)

CONDITIONED_TEMPLATE_MODE_PER_SOURCE = "per_source"
CONDITIONED_TEMPLATE_MODE_PER_ROW = "per_row"
CONDITIONED_TEMPLATE_MODE_HYBRID = "hybrid"

_CONDITIONED_TEMPLATE_PER_ROW_ELEMENT_LIMIT = 2_000_000
_CONDITIONED_TEMPLATE_PER_ROW_SCRATCH_LIMIT_BYTES = 512 * 1024 * 1024
_CONDITIONED_TEMPLATE_FLOAT_SCRATCH_ARRAYS = 6
_CONDITIONED_TEMPLATE_INT_SCRATCH_ARRAYS = 1

STREAM_CONDITIONED_CHUNK_SIZE = 8
STREAM_CONDITIONED_MAX_ROUNDS = 4
STREAM_CONDITIONED_RESCUE_CHUNK_SIZE = 32
STREAM_CONDITIONED_MAX_RESCUE_ROUNDS = 4
_STREAM_CONDITIONED_NUMBA_MIN_ROWS = 1024
STEP2_BEAM_PLACEMENT_FORCED_CO_RAS = "forced_co_ras"
STEP2_BEAM_PLACEMENT_EXCLUDE_RAS_RADIUS = "exclude_ras_radius"
_STEP2_BEAM_PLACEMENT_POLICIES = {
    STEP2_BEAM_PLACEMENT_FORCED_CO_RAS,
    STEP2_BEAM_PLACEMENT_EXCLUDE_RAS_RADIUS,
}
_EARTH_RADIUS_M_F64 = float(R_earth.to_value(u.m))


@dataclass(slots=True)
class ConditionedTemplatePlanCpu:
    """
    CPU template-sharing plan for conditioned step-2 beam generation.

    Attributes
    ----------
    mode_requested : str
        Template-sharing mode requested by the caller. One of
        ``"per_source"``, ``"per_row"``, or ``"hybrid"``.
    mode_used : str
        Effective mode after evaluating the hybrid heuristics.
    active_row_count : int
        Number of visible ``(time, satellite)`` rows participating in beam
        generation.
    unit_count : int
        Number of template-sharing units. In ``per_row`` mode this equals
        ``active_row_count``; in ``per_source`` mode it equals the number of
        unique effective conditioning sources.
    pool_size : int
        Number of conditioned angle candidates drawn per unit.
    template_size : int
        Maximum number of mutually separated beams retained per unit.
    estimated_scratch_bytes : int
        Estimated temporary memory required by the per-row template builder.
    flat_shape : tuple[int, int]
        Flattenable ``(time, satellite)`` shape used by the row mapping arrays.
    active_rows : np.ndarray
        Flat indices of rows that are visible above the horizon.
    row_to_unit : np.ndarray
        Mapping from each active row to its template-sharing unit.
    unit_source_kind : np.ndarray
        Effective sampler source kind per unit.
    unit_source_id : np.ndarray
        Effective sampler source identifier per unit.
    unit_beta_max_rad : np.ndarray
        Maximum allowed off-axis angle, in radians, for each unit.
    """

    mode_requested: str
    mode_used: str
    active_row_count: int
    unit_count: int
    pool_size: int
    template_size: int
    estimated_scratch_bytes: int
    flat_shape: tuple[int, int]
    active_rows: np.ndarray
    row_to_unit: np.ndarray
    unit_source_kind: np.ndarray
    unit_source_id: np.ndarray
    unit_beta_max_rad: np.ndarray


def _encode_sampler_source_cpu(kind: np.ndarray, source_id: np.ndarray) -> np.ndarray:
    kind_i64 = np.asarray(kind, dtype=np.int64)
    source_i64 = np.asarray(source_id, dtype=np.int64)
    return (kind_i64 + np.int64(1)) * np.int64(1_000_000_000_000) + source_i64


def _conditioned_template_scratch_bytes_cpu(active_rows: int, pool_size: int) -> int:
    rows = int(max(0, active_rows))
    pool = int(max(0, pool_size))
    float_bytes = rows * pool * _CONDITIONED_TEMPLATE_FLOAT_SCRATCH_ARRAYS * np.dtype(np.float32).itemsize
    int_bytes = rows * pool * _CONDITIONED_TEMPLATE_INT_SCRATCH_ARRAYS * np.dtype(np.int32).itemsize
    return int(float_bytes + int_bytes)


def resolve_conditioned_sampler_sources_cpu(
    sampler: JointAngleSampler,
    sat_azimuth_deg: Any,
    sat_elevation_deg: Any,
    sat_belt_id: Any,
) -> dict[str, np.ndarray]:
    """
    Resolve the conditioned sampling source for each observer-frame row.

    Parameters
    ----------
    sampler : JointAngleSampler
        Preloaded joint-angle sampler with global, belt, and belt-plus-skycell
        reservoirs.
    sat_azimuth_deg : Any
        Satellite azimuth in the observer frame, in degrees. The array is
        interpreted as ``(time, satellite)`` after conversion to
        ``np.float32``.
    sat_elevation_deg : Any
        Satellite elevation in the observer frame, in degrees. Must share the
        same shape as ``sat_azimuth_deg``.
    sat_belt_id : Any
        Integer belt identifier per ``(time, satellite)`` row. Must share the
        same shape as ``sat_azimuth_deg``.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``"skycell_id"``, ``"source_kind"``,
        ``"source_id"``, and ``"valid_mask"``. The returned arrays all share
        the input ``(time, satellite)`` shape. Rows that cannot be mapped to a
        valid conditioned source use ``SAMPLER_SOURCE_INVALID`` and
        ``valid_mask=False``.

    Raises
    ------
    ValueError
        Raised when the input arrays do not share the same shape.

    Notes
    -----
    The fallback chain matches the conditioned step-2 notebook semantics:
    ``(belt, skycell)`` pool first, then belt-only pool, then the global CDF
    sampler.
    """
    az = np.asarray(sat_azimuth_deg, dtype=np.float32)
    el = np.asarray(sat_elevation_deg, dtype=np.float32)
    belt = np.asarray(sat_belt_id, dtype=np.float32)
    if az.shape != el.shape or az.shape != belt.shape:
        raise ValueError("sat_azimuth_deg, sat_elevation_deg, and sat_belt_id must share the same shape.")

    shape = az.shape
    az_flat = az.reshape(-1)
    el_flat = el.reshape(-1)
    belt_flat = belt.reshape(-1)

    sky_flat = np.full(az_flat.size, -1, dtype=np.int32)
    source_kind_flat = np.full(az_flat.size, SAMPLER_SOURCE_INVALID, dtype=np.int8)
    source_id_flat = np.full(az_flat.size, -1, dtype=np.int32)

    valid = np.isfinite(az_flat) & np.isfinite(el_flat) & np.isfinite(belt_flat)
    if np.any(valid):
        belt_i64 = np.rint(belt_flat).astype(np.int64, copy=False)
        belt_is_int = np.abs(belt_flat - belt_i64.astype(np.float32, copy=False)) <= np.float32(1.0e-6)
        in_belt = valid & belt_is_int & (belt_i64 >= 0) & (belt_i64 < int(sampler.n_belts))
        if np.any(in_belt):
            idx_belt = np.flatnonzero(in_belt).astype(np.int64, copy=False)
            sky_vals = sampler._skycell_id_from_observer_angles(
                az_flat[idx_belt],
                el_flat[idx_belt],
            ).astype(np.int32, copy=False)
            sky_flat[idx_belt] = sky_vals

            in_sky = (sky_vals >= 0) & (sky_vals < int(sampler.n_skycells))
            if np.any(in_sky):
                idx_use = idx_belt[in_sky]
                belt_use = belt_i64[idx_use].astype(np.int64, copy=False)
                sky_use = sky_vals[in_sky].astype(np.int64, copy=False)
                group_id = belt_use * np.int64(sampler.n_skycells) + sky_use

                group_raw = np.asarray(sampler.group_raw_counts, dtype=np.int64)
                group_ptr = np.asarray(sampler.group_ptr, dtype=np.int64)
                belt_raw = np.asarray(sampler.belt_raw_counts, dtype=np.int64)
                belt_ptr = np.asarray(sampler.belt_ptr, dtype=np.int64)

                use_group = (
                    (group_raw[group_id] >= int(sampler.cond_min_group_samples))
                    & (group_ptr[group_id + 1] > group_ptr[group_id])
                )
                use_belt = (
                    (~use_group)
                    & (belt_raw[belt_use] >= int(sampler.cond_min_belt_samples))
                    & (belt_ptr[belt_use + 1] > belt_ptr[belt_use])
                )
                use_global = ~(use_group | use_belt)

                if np.any(use_group):
                    rows = idx_use[use_group]
                    source_kind_flat[rows] = SAMPLER_SOURCE_GROUP
                    source_id_flat[rows] = group_id[use_group].astype(np.int32, copy=False)
                if np.any(use_belt):
                    rows = idx_use[use_belt]
                    source_kind_flat[rows] = SAMPLER_SOURCE_BELT
                    source_id_flat[rows] = belt_use[use_belt].astype(np.int32, copy=False)
                if np.any(use_global):
                    rows = idx_use[use_global]
                    source_kind_flat[rows] = SAMPLER_SOURCE_GLOBAL
                    source_id_flat[rows] = np.int32(-1)

    return {
        "skycell_id": sky_flat.reshape(shape),
        "source_kind": source_kind_flat.reshape(shape),
        "source_id": source_id_flat.reshape(shape),
        "valid_mask": (source_kind_flat != SAMPLER_SOURCE_INVALID).reshape(shape),
    }


def sample_conditioned_candidate_batch_cpu(
    sampler: JointAngleSampler,
    source_kind: Any,
    source_id: Any,
    *,
    chunk_size: int = STREAM_CONDITIONED_CHUNK_SIZE,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Draw a small conditioned candidate batch for the requested source rows.

    Parameters
    ----------
    sampler : JointAngleSampler
        Preloaded conditional sampler containing global, belt, and
        belt-plus-skycell reservoirs.
    source_kind : Any
        One-dimensional effective sampler source kind for each requested row.
        Valid values are ``SAMPLER_SOURCE_GROUP``,
        ``SAMPLER_SOURCE_BELT``, and ``SAMPLER_SOURCE_GLOBAL``.
    source_id : Any
        One-dimensional effective source identifier for each requested row.
        Must share the same shape as ``source_kind``.
    chunk_size : int, optional
        Number of candidate draws per requested row.
    rng : np.random.Generator
        Random generator used for deterministic conditioned or global draws.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing ``alpha_rad`` and ``beta_rad`` with shape
        ``(n_rows, chunk_size)``. Invalid rows remain ``NaN`` filled.

    Raises
    ------
    ValueError
        Raised when ``source_kind`` and ``source_id`` do not share the same
        one-dimensional shape, or when ``chunk_size`` is not positive.
    RuntimeError
        Raised when a group or belt source resolves to an empty candidate pool.

    Notes
    -----
    This helper keeps the conditioned fallback logic in the pre-resolved
    ``source_kind`` / ``source_id`` space, so later beam-generation rounds do
    not need to recompute sky-cell IDs.
    """
    source_kind_arr = np.asarray(source_kind, dtype=np.int8).reshape(-1)
    source_id_arr = np.asarray(source_id, dtype=np.int32).reshape(-1)
    if source_kind_arr.shape != source_id_arr.shape:
        raise ValueError("source_kind and source_id must share the same shape.")

    size = int(chunk_size)
    if size <= 0:
        raise ValueError("chunk_size must be positive.")

    row_count = int(source_kind_arr.size)
    alpha_deg = np.full((row_count, size), np.nan, dtype=np.float32)
    beta_deg = np.full((row_count, size), np.nan, dtype=np.float32)

    if row_count == 0:
        return {
            "alpha_rad": alpha_deg,
            "beta_rad": beta_deg,
        }

    valid_kind = source_kind_arr != SAMPLER_SOURCE_INVALID
    if np.any(valid_kind):
        group_ptr = np.asarray(sampler.group_ptr, dtype=np.int64)
        group_beta_pool = np.asarray(sampler.group_beta_pool, dtype=np.float32)
        group_alpha_pool = np.asarray(sampler.group_alpha_pool, dtype=np.float32)
        belt_ptr = np.asarray(sampler.belt_ptr, dtype=np.int64)
        belt_beta_pool = np.asarray(sampler.belt_beta_pool, dtype=np.float32)
        belt_alpha_pool = np.asarray(sampler.belt_alpha_pool, dtype=np.float32)

        for kind_value, ptr, beta_pool, alpha_pool in (
            (SAMPLER_SOURCE_GROUP, group_ptr, group_beta_pool, group_alpha_pool),
            (SAMPLER_SOURCE_BELT, belt_ptr, belt_beta_pool, belt_alpha_pool),
        ):
            rows = np.flatnonzero(valid_kind & (source_kind_arr == kind_value)).astype(np.int64, copy=False)
            if rows.size == 0:
                continue

            ids = source_id_arr[rows].astype(np.int64, copy=False)
            order = np.argsort(ids, kind="mergesort")
            ids_sorted = ids[order]
            rows_sorted = rows[order]
            cuts = np.flatnonzero(np.diff(ids_sorted)) + 1
            starts = np.concatenate(([0], cuts))
            ends = np.concatenate((cuts, [ids_sorted.size]))

            for start, end in zip(starts, ends):
                sid = int(ids_sorted[start])
                p0 = int(ptr[sid])
                p1 = int(ptr[sid + 1])
                if p1 <= p0:
                    raise RuntimeError("Conditioned sampler source resolved an empty candidate pool.")
                rr = rows_sorted[start:end]
                draws = rng.integers(p0, p1, size=(rr.size, size))
                beta_deg[rr, :] = beta_pool[draws]
                alpha_deg[rr, :] = alpha_pool[draws]

        global_rows = np.flatnonzero(valid_kind & (source_kind_arr == SAMPLER_SOURCE_GLOBAL)).astype(np.int64, copy=False)
        if global_rows.size > 0:
            beta_flat, alpha_flat = sampler._sample_unconditional_flat(
                rng,
                int(global_rows.size) * size,
                dtype=np.float32,
            )
            beta_deg[global_rows, :] = beta_flat.reshape(int(global_rows.size), size)
            alpha_deg[global_rows, :] = alpha_flat.reshape(int(global_rows.size), size)

    deg2rad = np.float32(np.pi / 180.0)
    two_pi = np.float32(2.0 * np.pi)
    return {
        "alpha_rad": np.remainder(alpha_deg * deg2rad, two_pi).astype(np.float32, copy=False),
        "beta_rad": (beta_deg * deg2rad).astype(np.float32, copy=False),
    }


def _normalise_step2_beam_placement_policy(policy: str | None) -> str:
    policy_use = str(policy or STEP2_BEAM_PLACEMENT_FORCED_CO_RAS).strip().lower()
    if policy_use not in _STEP2_BEAM_PLACEMENT_POLICIES:
        raise ValueError(
            "beam_placement_policy must be one of "
            f"{sorted(_STEP2_BEAM_PLACEMENT_POLICIES)!r}."
        )
    return policy_use


def _normalise_step2_ras_sat_azel_cpu(
    value: Any,
    *,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 4:
        if int(arr.shape[1]) != 1:
            raise ValueError(
                "ras_sat_azel second axis must be singleton when ndim=4; "
                f"got {tuple(arr.shape)!r}."
            )
        arr = arr[:, 0, :, :]
    elif arr.ndim != 3:
        raise ValueError(
            "ras_sat_azel must have shape (T, 1, S, K) or (T, S, K); "
            f"got {tuple(arr.shape)!r}."
        )
    if tuple(int(v) for v in arr.shape[:2]) != tuple(int(v) for v in expected_shape):
        raise ValueError(
            "ras_sat_azel must align with the step-2 row shape over (T, S); "
            f"expected {expected_shape!r}, got {tuple(int(v) for v in arr.shape[:2])!r}."
        )
    if int(arr.shape[2]) < 2:
        raise ValueError("ras_sat_azel last axis must contain at least azimuth and theta.")
    alpha_rad = np.remainder(arr[..., 0] * np.float32(np.pi / 180.0), np.float32(2.0 * np.pi)).astype(
        np.float32,
        copy=False,
    )
    beta_rad = np.abs(arr[..., 1] * np.float32(np.pi / 180.0)).astype(np.float32, copy=False)
    return alpha_rad, beta_rad


def _ground_intercept_central_angle_rad(
    beta_rad: float,
    orbit_radius_m: float,
    *,
    target_radius_m: float,
) -> float:
    cos_beta = float(np.cos(beta_rad))
    if cos_beta <= 0.0:
        return np.nan
    term = orbit_radius_m * cos_beta
    disc = term * term - (orbit_radius_m * orbit_radius_m - target_radius_m * target_radius_m)
    if disc < 0.0:
        if disc < (-1e-6 * target_radius_m * target_radius_m):
            return np.nan
        disc = 0.0
    d_target_m = term - float(np.sqrt(disc))
    if d_target_m < 0.0:
        return np.nan
    z_coord = orbit_radius_m - d_target_m * cos_beta
    cos_delta = z_coord / target_radius_m
    if cos_delta < -1.0:
        cos_delta = -1.0
    elif cos_delta > 1.0:
        cos_delta = 1.0
    return float(np.arccos(cos_delta))


def _candidate_inside_ras_exclusion_python(
    alpha_c: float,
    beta_c: float,
    ras_alpha: float,
    ras_beta: float,
    orbit_radius_m: float,
    *,
    target_radius_m: float,
    exclusion_radius_m: float,
) -> bool:
    gamma_c = _ground_intercept_central_angle_rad(
        beta_c,
        orbit_radius_m,
        target_radius_m=target_radius_m,
    )
    gamma_ras = _ground_intercept_central_angle_rad(
        ras_beta,
        orbit_radius_m,
        target_radius_m=target_radius_m,
    )
    if not np.isfinite(gamma_c) or not np.isfinite(gamma_ras):
        return True
    cos_sep = float(np.cos(gamma_c) * np.cos(gamma_ras))
    cos_sep += float(np.sin(gamma_c) * np.sin(gamma_ras) * np.cos(alpha_c - ras_alpha))
    sep_rad = float(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
    return (target_radius_m * sep_rad) < exclusion_radius_m


def _accept_conditioned_candidates_python(
    pending_active_idx: np.ndarray,
    beam_idx_active: np.ndarray,
    beam_alpha_active: np.ndarray,
    beam_beta_active: np.ndarray,
    beam_valid_active: np.ndarray,
    candidate_alpha_rad: np.ndarray,
    candidate_beta_rad: np.ndarray,
    beta_max_active: np.ndarray,
    ras_alpha_active: np.ndarray,
    ras_beta_active: np.ndarray,
    orbit_radius_active: np.ndarray,
    *,
    n_beams: int,
    cos_min_sep: float,
    draw_index_base: int,
    apply_ras_exclusion: bool,
    target_radius_m: float,
    exclusion_radius_m: float,
) -> None:
    for pending_idx, row in enumerate(np.asarray(pending_active_idx, dtype=np.int32).tolist()):
        n_acc = int(beam_valid_active[row])
        beta_max = float(beta_max_active[row])
        ras_alpha = float(ras_alpha_active[row])
        ras_beta = float(ras_beta_active[row])
        orbit_radius_m = float(orbit_radius_active[row])
        for cand_idx in range(int(candidate_alpha_rad.shape[1])):
            if n_acc >= int(n_beams):
                break
            alpha_c = float(candidate_alpha_rad[pending_idx, cand_idx])
            beta_c = float(candidate_beta_rad[pending_idx, cand_idx])
            if not np.isfinite(alpha_c) or not np.isfinite(beta_c) or beta_c > beta_max:
                continue
            if apply_ras_exclusion and _candidate_inside_ras_exclusion_python(
                alpha_c,
                beta_c,
                ras_alpha,
                ras_beta,
                orbit_radius_m,
                target_radius_m=target_radius_m,
                exclusion_radius_m=exclusion_radius_m,
            ):
                continue
            ok = True
            for prev in range(n_acc):
                alpha_prev = float(beam_alpha_active[row, prev])
                beta_prev = float(beam_beta_active[row, prev])
                cos_da = float(np.cos(alpha_prev - alpha_c))
                cos_gamma = float(np.cos(beta_prev) * np.cos(beta_c) + np.sin(beta_prev) * np.sin(beta_c) * cos_da)
                if cos_gamma > float(cos_min_sep):
                    ok = False
                    break
            if not ok:
                continue
            beam_idx_active[row, n_acc] = np.int32(draw_index_base + cand_idx)
            beam_alpha_active[row, n_acc] = np.float32(alpha_c)
            beam_beta_active[row, n_acc] = np.float32(beta_c)
            n_acc += 1
        beam_valid_active[row] = np.int16(n_acc)


if njit is not None:
    @njit(cache=True)
    def _ground_intercept_central_angle_rad_numba(
        beta_rad: float,
        orbit_radius_m: float,
        target_radius_m: float,
    ) -> float:
        cos_beta = np.cos(beta_rad)
        if cos_beta <= 0.0:
            return np.nan
        term = orbit_radius_m * cos_beta
        disc = term * term - (orbit_radius_m * orbit_radius_m - target_radius_m * target_radius_m)
        if disc < 0.0:
            if disc < (-1e-6 * target_radius_m * target_radius_m):
                return np.nan
            disc = 0.0
        d_target_m = term - np.sqrt(disc)
        if d_target_m < 0.0:
            return np.nan
        z_coord = orbit_radius_m - d_target_m * cos_beta
        cos_delta = z_coord / target_radius_m
        if cos_delta < -1.0:
            cos_delta = -1.0
        elif cos_delta > 1.0:
            cos_delta = 1.0
        return np.arccos(cos_delta)


    @njit(cache=True)
    def _candidate_inside_ras_exclusion_numba(
        alpha_c: float,
        beta_c: float,
        ras_alpha: float,
        ras_beta: float,
        orbit_radius_m: float,
        target_radius_m: float,
        exclusion_radius_m: float,
    ) -> bool:
        gamma_c = _ground_intercept_central_angle_rad_numba(beta_c, orbit_radius_m, target_radius_m)
        gamma_ras = _ground_intercept_central_angle_rad_numba(ras_beta, orbit_radius_m, target_radius_m)
        if np.isnan(gamma_c) or np.isnan(gamma_ras):
            return True
        cos_sep = np.cos(gamma_c) * np.cos(gamma_ras)
        cos_sep += np.sin(gamma_c) * np.sin(gamma_ras) * np.cos(alpha_c - ras_alpha)
        if cos_sep < -1.0:
            cos_sep = -1.0
        elif cos_sep > 1.0:
            cos_sep = 1.0
        sep_rad = np.arccos(cos_sep)
        return (target_radius_m * sep_rad) < exclusion_radius_m


    @njit(parallel=True, fastmath=True, cache=True)
    def _accept_conditioned_candidates_numba(
        pending_active_idx: np.ndarray,
        beam_idx_active: np.ndarray,
        beam_alpha_active: np.ndarray,
        beam_beta_active: np.ndarray,
        beam_valid_active: np.ndarray,
        candidate_alpha_rad: np.ndarray,
        candidate_beta_rad: np.ndarray,
        beta_max_active: np.ndarray,
        ras_alpha_active: np.ndarray,
        ras_beta_active: np.ndarray,
        orbit_radius_active: np.ndarray,
        n_beams: int,
        cos_min_sep: float,
        draw_index_base: int,
        apply_ras_exclusion: bool,
        target_radius_m: float,
        exclusion_radius_m: float,
    ) -> None:
        for pending_idx in prange(pending_active_idx.size):
            row = int(pending_active_idx[pending_idx])
            n_acc = int(beam_valid_active[row])
            beta_max = float(beta_max_active[row])
            ras_alpha = float(ras_alpha_active[row])
            ras_beta = float(ras_beta_active[row])
            orbit_radius_m = float(orbit_radius_active[row])
            for cand_idx in range(candidate_alpha_rad.shape[1]):
                if n_acc >= n_beams:
                    break
                alpha_c = float(candidate_alpha_rad[pending_idx, cand_idx])
                beta_c = float(candidate_beta_rad[pending_idx, cand_idx])
                if not np.isfinite(alpha_c) or not np.isfinite(beta_c) or beta_c > beta_max:
                    continue
                if apply_ras_exclusion and _candidate_inside_ras_exclusion_numba(
                    alpha_c,
                    beta_c,
                    ras_alpha,
                    ras_beta,
                    orbit_radius_m,
                    target_radius_m,
                    exclusion_radius_m,
                ):
                    continue
                ok = True
                for prev in range(n_acc):
                    alpha_prev = float(beam_alpha_active[row, prev])
                    beta_prev = float(beam_beta_active[row, prev])
                    cos_da = np.cos(alpha_prev - alpha_c)
                    cos_gamma = np.cos(beta_prev) * np.cos(beta_c)
                    cos_gamma += np.sin(beta_prev) * np.sin(beta_c) * cos_da
                    if cos_gamma > cos_min_sep:
                        ok = False
                        break
                if not ok:
                    continue
                beam_idx_active[row, n_acc] = np.int32(draw_index_base + cand_idx)
                beam_alpha_active[row, n_acc] = np.float32(alpha_c)
                beam_beta_active[row, n_acc] = np.float32(beta_c)
                n_acc += 1
            beam_valid_active[row] = np.int16(n_acc)


def _accept_conditioned_candidates_cpu(
    pending_active_idx: np.ndarray,
    beam_idx_active: np.ndarray,
    beam_alpha_active: np.ndarray,
    beam_beta_active: np.ndarray,
    beam_valid_active: np.ndarray,
    candidate_alpha_rad: np.ndarray,
    candidate_beta_rad: np.ndarray,
    beta_max_active: np.ndarray,
    ras_alpha_active: np.ndarray,
    ras_beta_active: np.ndarray,
    orbit_radius_active: np.ndarray,
    *,
    n_beams: int,
    cos_min_sep: float,
    draw_index_base: int,
    apply_ras_exclusion: bool,
    target_radius_m: float,
    exclusion_radius_m: float,
) -> None:
    if njit is not None and int(np.asarray(pending_active_idx).size) >= _STREAM_CONDITIONED_NUMBA_MIN_ROWS:
        _accept_conditioned_candidates_numba(
            np.asarray(pending_active_idx, dtype=np.int32),
            np.asarray(beam_idx_active, dtype=np.int32),
            np.asarray(beam_alpha_active, dtype=np.float32),
            np.asarray(beam_beta_active, dtype=np.float32),
            np.asarray(beam_valid_active, dtype=np.int16),
            np.asarray(candidate_alpha_rad, dtype=np.float32),
            np.asarray(candidate_beta_rad, dtype=np.float32),
            np.asarray(beta_max_active, dtype=np.float32),
            np.asarray(ras_alpha_active, dtype=np.float32),
            np.asarray(ras_beta_active, dtype=np.float32),
            np.asarray(orbit_radius_active, dtype=np.float32),
            int(n_beams),
            float(cos_min_sep),
            int(draw_index_base),
            bool(apply_ras_exclusion),
            float(target_radius_m),
            float(exclusion_radius_m),
        )
        return

    _accept_conditioned_candidates_python(
        pending_active_idx,
        beam_idx_active,
        beam_alpha_active,
        beam_beta_active,
        beam_valid_active,
        candidate_alpha_rad,
        candidate_beta_rad,
        beta_max_active,
        ras_alpha_active,
        ras_beta_active,
        orbit_radius_active,
        n_beams=n_beams,
        cos_min_sep=cos_min_sep,
        draw_index_base=draw_index_base,
        apply_ras_exclusion=apply_ras_exclusion,
        target_radius_m=target_radius_m,
        exclusion_radius_m=exclusion_radius_m,
    )


def fill_conditioned_beams_streaming_cpu(
    sampler: JointAngleSampler,
    source_kind: Any,
    source_id: Any,
    *,
    vis_mask_horizon: Any,
    is_co_sat: Any,
    alpha0_rad: Any,
    beta0_rad: Any,
    beta_max_rad_per_sat: Any,
    n_beams: int,
    cos_min_sep: float,
    seed: int,
    chunk_size: int = STREAM_CONDITIONED_CHUNK_SIZE,
    max_rounds: int = STREAM_CONDITIONED_MAX_ROUNDS,
    rescue_chunk_size: int = STREAM_CONDITIONED_RESCUE_CHUNK_SIZE,
    max_rescue_rounds: int = STREAM_CONDITIONED_MAX_RESCUE_ROUNDS,
    beam_placement_policy: str = STEP2_BEAM_PLACEMENT_FORCED_CO_RAS,
    ras_sat_azel: Any | None = None,
    orbit_radius_m_per_sat: Any | None = None,
    ras_exclusion_radius_km: float | None = None,
    target_alt_km: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Fill step-2 beams by streaming conditioned candidate chunks on CPU.

    Parameters
    ----------
    sampler : JointAngleSampler
        Preloaded conditional joint-angle sampler.
    source_kind, source_id : Any
        Full-shape ``(time, satellite)`` conditioned source tensors, typically
        produced by :func:`resolve_conditioned_sampler_sources_cpu`.
    vis_mask_horizon : Any
        Boolean visibility mask aligned with ``source_kind``.
    is_co_sat : Any
        Boolean mask identifying the serving co-satellite rows.
    alpha0_rad, beta0_rad : Any
        Serving-link beam pointing in radians for each ``(time, satellite)``
        row.
    beta_max_rad_per_sat : Any
        One-dimensional maximum off-axis angle per satellite, in radians.
    n_beams : int
        Number of beams that must be assigned to each visible row.
    cos_min_sep : float
        Cosine of the minimum allowed beam-center separation.
    seed : int
        Deterministic seed for the streaming candidate draws.
    chunk_size : int, optional
        Candidate chunk size used in the normal streaming rounds.
    max_rounds : int, optional
        Maximum number of normal streaming rounds.
    rescue_chunk_size : int, optional
        Larger candidate chunk size used in the rescue rounds.
    max_rescue_rounds : int, optional
        Maximum number of rescue rounds attempted after the normal rounds.
    beam_placement_policy : {"forced_co_ras", "exclude_ras_radius"}, optional
        Step-2 beam-placement policy. ``"forced_co_ras"`` preserves the
        historical behavior where visible co-satellite rows preseed their first
        beam at the RAS station. ``"exclude_ras_radius"`` disables that forced
        beam and rejects any candidate whose exact ground intercept falls
        inside ``ras_exclusion_radius_km`` around the RAS station.
    ras_sat_azel : Any or None, optional
        Exact RAS-station direction in the satellite frame, aligned over
        ``(time, satellite)``. Required for
        ``beam_placement_policy="exclude_ras_radius"``. Accepted shapes are
        ``(T, 1, S, K)`` and ``(T, S, K)``; the first two fields must be
        azimuth and theta in degrees.
    orbit_radius_m_per_sat : Any or None, optional
        One-dimensional orbital-radius array in metres aligned with the
        satellite axis. Required for
        ``beam_placement_policy="exclude_ras_radius"``.
    ras_exclusion_radius_km : float or None, optional
        Forbidden ground-radius around the RAS station used only when
        ``beam_placement_policy="exclude_ras_radius"``.
    target_alt_km : float, optional
        Target-sphere altitude used for the exclusion-zone intercept geometry.
        The default ``0.0`` keeps the exclusion on the Earth surface.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing ``beam_idx``, ``beam_alpha_rad``,
        ``beam_beta_rad``, and ``beam_valid``. Extra scalar diagnostics are
        returned as ``draws_attempted``, ``rounds_used``, and
        ``unfinished_row_count``.

    Raises
    ------
    ValueError
        Raised when the full-shape inputs do not align or when any chunk/beam
        parameter is invalid.
    RuntimeError
        Raised when visible rows remain unfinished after the normal and rescue
        rounds.

    Notes
    -----
    The acceptance rule is identical to the old template path: every accepted
    beam must satisfy the per-satellite ``beta_max`` and must remain at least
    ``theta_sep`` away from all previously accepted beams, including the
    forced CO beam when present. In ``"exclude_ras_radius"`` mode an extra
    exact spherical-Earth intercept test rejects beams whose served point would
    lie inside the requested RAS exclusion zone.
    """
    source_kind_arr = np.asarray(source_kind, dtype=np.int8)
    source_id_arr = np.asarray(source_id, dtype=np.int32)
    vis_arr = np.asarray(vis_mask_horizon, dtype=bool)
    co_arr = np.asarray(is_co_sat, dtype=bool)
    alpha0_arr = np.asarray(alpha0_rad, dtype=np.float32)
    beta0_arr = np.asarray(beta0_rad, dtype=np.float32)

    if (
        source_kind_arr.shape != source_id_arr.shape
        or source_kind_arr.shape != vis_arr.shape
        or source_kind_arr.shape != co_arr.shape
        or source_kind_arr.shape != alpha0_arr.shape
        or source_kind_arr.shape != beta0_arr.shape
    ):
        raise ValueError(
            "source_kind, source_id, vis_mask_horizon, is_co_sat, alpha0_rad, "
            "and beta0_rad must share the same shape."
        )

    beam_count = int(n_beams)
    if beam_count <= 0:
        raise ValueError("n_beams must be positive.")
    if int(chunk_size) <= 0 or int(rescue_chunk_size) <= 0:
        raise ValueError("chunk sizes must be positive.")
    if int(max_rounds) < 0 or int(max_rescue_rounds) < 0:
        raise ValueError("round limits must be non-negative.")
    target_alt_km_value = float(target_alt_km)
    if target_alt_km_value < 0.0:
        raise ValueError("target_alt_km must be non-negative.")

    flat_shape = tuple(int(v) for v in vis_arr.shape)
    flat_rows = int(np.prod(flat_shape, dtype=np.int64))
    sat_count = int(flat_shape[-1])
    beta_max_sat = np.asarray(beta_max_rad_per_sat, dtype=np.float32).reshape(-1)
    if beta_max_sat.size != sat_count:
        raise ValueError("beta_max_rad_per_sat must have length equal to the satellite axis.")
    beam_policy = _normalise_step2_beam_placement_policy(beam_placement_policy)
    use_ras_exclusion = beam_policy == STEP2_BEAM_PLACEMENT_EXCLUDE_RAS_RADIUS
    ras_alpha_row = np.zeros(flat_rows, dtype=np.float32)
    ras_beta_row = np.zeros(flat_rows, dtype=np.float32)
    orbit_radius_row = np.zeros(flat_rows, dtype=np.float32)
    target_radius_m = float(_EARTH_RADIUS_M_F64 + target_alt_km_value * 1000.0)
    exclusion_radius_m = np.float32(0.0)
    if use_ras_exclusion:
        if ras_sat_azel is None:
            raise ValueError("ras_sat_azel is required when beam_placement_policy='exclude_ras_radius'.")
        if orbit_radius_m_per_sat is None:
            raise ValueError("orbit_radius_m_per_sat is required when beam_placement_policy='exclude_ras_radius'.")
        if ras_exclusion_radius_km is None:
            raise ValueError("ras_exclusion_radius_km is required when beam_placement_policy='exclude_ras_radius'.")
        ras_exclusion_radius_value = float(ras_exclusion_radius_km)
        if ras_exclusion_radius_value < 0.0:
            raise ValueError("ras_exclusion_radius_km must be non-negative.")
        ras_alpha_arr, ras_beta_arr = _normalise_step2_ras_sat_azel_cpu(
            ras_sat_azel,
            expected_shape=flat_shape,
        )
        orbit_radius_sat = np.asarray(orbit_radius_m_per_sat, dtype=np.float32).reshape(-1)
        if orbit_radius_sat.size != sat_count:
            raise ValueError("orbit_radius_m_per_sat must have length equal to the satellite axis.")
        ras_alpha_row = ras_alpha_arr.reshape(flat_rows)
        ras_beta_row = ras_beta_arr.reshape(flat_rows)
        orbit_radius_row = np.broadcast_to(orbit_radius_sat[None, :], flat_shape).reshape(flat_rows)
        exclusion_radius_m = np.float32(ras_exclusion_radius_value * 1000.0)

    source_kind_flat = source_kind_arr.reshape(flat_rows)
    source_id_flat = source_id_arr.reshape(flat_rows)
    vis_flat = vis_arr.reshape(flat_rows)
    co_flat = co_arr.reshape(flat_rows)
    alpha0_flat = alpha0_arr.reshape(flat_rows)
    beta0_flat = beta0_arr.reshape(flat_rows)
    beta_max_row = np.broadcast_to(beta_max_sat[None, :], flat_shape).reshape(flat_rows)

    invalid_visible = vis_flat & (source_kind_flat == SAMPLER_SOURCE_INVALID)
    if np.any(invalid_visible):
        raise ValueError("Visible rows contain unresolved conditioned sampler sources.")

    beam_idx_flat = np.full((flat_rows, beam_count), -1, dtype=np.int32)
    beam_alpha_flat = np.full((flat_rows, beam_count), np.nan, dtype=np.float32)
    beam_beta_flat = np.full((flat_rows, beam_count), np.nan, dtype=np.float32)
    beam_valid_flat = np.zeros(flat_rows, dtype=np.int16)

    active_rows = np.flatnonzero(vis_flat).astype(np.int32, copy=False)
    active_row_count = int(active_rows.size)
    if active_row_count == 0:
        return {
            "beam_idx": beam_idx_flat.reshape(flat_shape + (beam_count,)),
            "beam_alpha_rad": beam_alpha_flat.reshape(flat_shape + (beam_count,)),
            "beam_beta_rad": beam_beta_flat.reshape(flat_shape + (beam_count,)),
            "beam_valid": beam_valid_flat.reshape(flat_shape),
            "draws_attempted": np.int64(0),
            "rounds_used": np.int32(0),
            "unfinished_row_count": np.int32(0),
        }

    active_source_kind = source_kind_flat[active_rows].astype(np.int8, copy=False)
    active_source_id = source_id_flat[active_rows].astype(np.int32, copy=False)
    active_is_co = co_flat[active_rows]
    active_alpha0 = alpha0_flat[active_rows].astype(np.float32, copy=False)
    active_beta0 = beta0_flat[active_rows].astype(np.float32, copy=False)
    active_beta_max = beta_max_row[active_rows].astype(np.float32, copy=False)
    active_ras_alpha = ras_alpha_row[active_rows].astype(np.float32, copy=False)
    active_ras_beta = ras_beta_row[active_rows].astype(np.float32, copy=False)
    active_orbit_radius = orbit_radius_row[active_rows].astype(np.float32, copy=False)

    beam_idx_active = np.full((active_row_count, beam_count), -1, dtype=np.int32)
    beam_alpha_active = np.full((active_row_count, beam_count), np.nan, dtype=np.float32)
    beam_beta_active = np.full((active_row_count, beam_count), np.nan, dtype=np.float32)
    beam_valid_active = np.zeros(active_row_count, dtype=np.int16)

    forced = active_is_co & (active_beta0 <= active_beta_max) if beam_policy == STEP2_BEAM_PLACEMENT_FORCED_CO_RAS else np.zeros(
        active_row_count,
        dtype=bool,
    )
    if np.any(forced):
        beam_idx_active[forced, 0] = np.int32(-2)
        beam_alpha_active[forced, 0] = active_alpha0[forced]
        beam_beta_active[forced, 0] = active_beta0[forced]
        beam_valid_active[forced] = np.int16(1)

    rng = np.random.default_rng(int(seed))
    draws_attempted = 0
    rounds_used = 0
    draw_index_base = 0

    for current_chunk_size, current_max_rounds in (
        (int(chunk_size), int(max_rounds)),
        (int(rescue_chunk_size), int(max_rescue_rounds)),
    ):
        for _ in range(current_max_rounds):
            pending_active_idx = np.flatnonzero(beam_valid_active < beam_count).astype(np.int32, copy=False)
            if pending_active_idx.size == 0:
                break

            candidate_batch = sample_conditioned_candidate_batch_cpu(
                sampler,
                active_source_kind[pending_active_idx],
                active_source_id[pending_active_idx],
                chunk_size=current_chunk_size,
                rng=rng,
            )
            _accept_conditioned_candidates_cpu(
                pending_active_idx,
                beam_idx_active,
                beam_alpha_active,
                beam_beta_active,
                beam_valid_active,
                np.asarray(candidate_batch["alpha_rad"], dtype=np.float32),
                np.asarray(candidate_batch["beta_rad"], dtype=np.float32),
                active_beta_max,
                active_ras_alpha,
                active_ras_beta,
                active_orbit_radius,
                n_beams=beam_count,
                cos_min_sep=float(cos_min_sep),
                draw_index_base=draw_index_base,
                apply_ras_exclusion=use_ras_exclusion,
                target_radius_m=float(target_radius_m),
                exclusion_radius_m=float(exclusion_radius_m),
            )
            draws_attempted += int(pending_active_idx.size) * int(current_chunk_size)
            draw_index_base += int(current_chunk_size)
            rounds_used += 1

        if not np.any(beam_valid_active < beam_count):
            break

    unfinished_row_count = int(np.count_nonzero(beam_valid_active < beam_count))
    if unfinished_row_count > 0:
        raise RuntimeError(
            "Streaming conditioned beam generation could not satisfy "
            f"n_beams={beam_count} for {unfinished_row_count} visible rows "
            f"after {draws_attempted} candidate draws."
        )

    beam_idx_flat[active_rows, :] = beam_idx_active
    beam_alpha_flat[active_rows, :] = beam_alpha_active
    beam_beta_flat[active_rows, :] = beam_beta_active
    beam_valid_flat[active_rows] = beam_valid_active
    return {
        "beam_idx": beam_idx_flat.reshape(flat_shape + (beam_count,)),
        "beam_alpha_rad": beam_alpha_flat.reshape(flat_shape + (beam_count,)),
        "beam_beta_rad": beam_beta_flat.reshape(flat_shape + (beam_count,)),
        "beam_valid": beam_valid_flat.reshape(flat_shape),
        "draws_attempted": np.int64(draws_attempted),
        "rounds_used": np.int32(rounds_used),
        "unfinished_row_count": np.int32(0),
    }


def make_conditioned_template_plan_cpu(
    source_kind: Any,
    source_id: Any,
    vis_mask_horizon: Any,
    beta_max_rad_per_sat: Any,
    *,
    mode: str | None = CONDITIONED_TEMPLATE_MODE_HYBRID,
    pool_size: int,
    template_size: int,
) -> ConditionedTemplatePlanCpu:
    """
    Build the conditioned template-sharing plan for a CPU step-2 batch.

    Parameters
    ----------
    source_kind : Any
        Effective conditioned source kind for each ``(time, satellite)`` row.
    source_id : Any
        Effective conditioned source identifier for each row. Must share the
        same shape as ``source_kind``.
    vis_mask_horizon : Any
        Boolean visibility mask for the batch, with the same
        ``(time, satellite)`` shape.
    beta_max_rad_per_sat : Any
        Per-satellite maximum allowed beam off-axis angle in radians. The array
        must be broadcastable to the satellite axis.
    mode : str or None, optional
        Requested template-sharing mode: ``"per_source"``, ``"per_row"``, or
        ``"hybrid"``. ``None`` maps to ``"hybrid"``.
    pool_size : int
        Number of conditioned candidate beams drawn per unit.
    template_size : int
        Maximum number of mutually separated beams retained per unit.

    Returns
    -------
    ConditionedTemplatePlanCpu
        Compact plan object describing the active rows, the effective sharing
        units, and the per-unit beta constraints.

    Raises
    ------
    ValueError
        Raised when the input arrays do not share the same shape, when the
        satellite-axis beta limits have the wrong length, or when ``mode`` is
        unsupported.

    Notes
    -----
    In hybrid mode the helper chooses ``per_row`` only when both the
    ``active_rows * pool_size`` element count and a conservative scratch-memory
    estimate remain below the configured thresholds.
    """
    source_kind_arr = np.asarray(source_kind, dtype=np.int8)
    source_id_arr = np.asarray(source_id, dtype=np.int32)
    vis_arr = np.asarray(vis_mask_horizon, dtype=bool)
    if source_kind_arr.shape != source_id_arr.shape or source_kind_arr.shape != vis_arr.shape:
        raise ValueError("source_kind, source_id, and vis_mask_horizon must share the same shape.")

    flat_shape = tuple(int(v) for v in source_kind_arr.shape)
    sat_count = flat_shape[-1]
    beta_max_per_sat = np.asarray(beta_max_rad_per_sat, dtype=np.float32).reshape(-1)
    if beta_max_per_sat.size != sat_count:
        raise ValueError("beta_max_rad_per_sat must have length equal to the satellite axis.")

    source_kind_flat = source_kind_arr.reshape(-1)
    source_id_flat = source_id_arr.reshape(-1)
    vis_flat = vis_arr.reshape(-1)
    active_rows = np.flatnonzero(vis_flat).astype(np.int32, copy=False)
    active_row_count = int(active_rows.size)

    requested_mode = str(mode or CONDITIONED_TEMPLATE_MODE_HYBRID).lower()
    if requested_mode not in {
        CONDITIONED_TEMPLATE_MODE_PER_SOURCE,
        CONDITIONED_TEMPLATE_MODE_PER_ROW,
        CONDITIONED_TEMPLATE_MODE_HYBRID,
    }:
        raise ValueError(f"Unsupported conditioned template mode: {mode!r}")

    scratch_bytes = _conditioned_template_scratch_bytes_cpu(active_row_count, int(pool_size))
    if requested_mode == CONDITIONED_TEMPLATE_MODE_HYBRID:
        if (
            active_row_count * int(pool_size) <= _CONDITIONED_TEMPLATE_PER_ROW_ELEMENT_LIMIT
            and scratch_bytes <= _CONDITIONED_TEMPLATE_PER_ROW_SCRATCH_LIMIT_BYTES
        ):
            mode_used = CONDITIONED_TEMPLATE_MODE_PER_ROW
        else:
            mode_used = CONDITIONED_TEMPLATE_MODE_PER_SOURCE
    else:
        mode_used = requested_mode

    beta_max_row = np.broadcast_to(beta_max_per_sat[None, :], flat_shape).reshape(-1)

    if active_row_count == 0:
        return ConditionedTemplatePlanCpu(
            mode_requested=requested_mode,
            mode_used=mode_used,
            active_row_count=0,
            unit_count=0,
            pool_size=int(pool_size),
            template_size=int(template_size),
            estimated_scratch_bytes=scratch_bytes,
            flat_shape=flat_shape,
            active_rows=np.empty(0, dtype=np.int32),
            row_to_unit=np.empty(0, dtype=np.int32),
            unit_source_kind=np.empty(0, dtype=np.int8),
            unit_source_id=np.empty(0, dtype=np.int32),
            unit_beta_max_rad=np.empty(0, dtype=np.float32),
        )

    if mode_used == CONDITIONED_TEMPLATE_MODE_PER_ROW:
        row_to_unit = np.arange(active_row_count, dtype=np.int32)
        unit_source_kind = source_kind_flat[active_rows].astype(np.int8, copy=False)
        unit_source_id = source_id_flat[active_rows].astype(np.int32, copy=False)
        unit_beta_max_rad = beta_max_row[active_rows].astype(np.float32, copy=False)
    else:
        active_codes = _encode_sampler_source_cpu(source_kind_flat[active_rows], source_id_flat[active_rows])
        unique_codes, first_idx, inverse = np.unique(
            active_codes,
            return_index=True,
            return_inverse=True,
        )
        unit_rows = active_rows[first_idx]
        row_to_unit = inverse.astype(np.int32, copy=False)
        unit_source_kind = source_kind_flat[unit_rows].astype(np.int8, copy=False)
        unit_source_id = source_id_flat[unit_rows].astype(np.int32, copy=False)
        unit_beta_max_rad = np.zeros(unique_codes.size, dtype=np.float32)
        np.maximum.at(unit_beta_max_rad, row_to_unit, beta_max_row[active_rows].astype(np.float32, copy=False))

    return ConditionedTemplatePlanCpu(
        mode_requested=requested_mode,
        mode_used=mode_used,
        active_row_count=active_row_count,
        unit_count=int(unit_source_kind.size),
        pool_size=int(pool_size),
        template_size=int(template_size),
        estimated_scratch_bytes=scratch_bytes,
        flat_shape=flat_shape,
        active_rows=active_rows,
        row_to_unit=row_to_unit,
        unit_source_kind=unit_source_kind,
        unit_source_id=unit_source_id,
        unit_beta_max_rad=unit_beta_max_rad.astype(np.float32, copy=False),
    )


def sample_conditioned_candidate_pools_cpu(
    sampler: JointAngleSampler,
    source_kind: Any,
    source_id: Any,
    *,
    pool_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Draw conditioned candidate beam pools on CPU.

    Parameters
    ----------
    sampler : JointAngleSampler
        Preloaded joint-angle sampler containing global, belt, and group pools.
    source_kind : Any
        Effective sampler source kind per sharing unit.
    source_id : Any
        Effective sampler source identifier per sharing unit.
    pool_size : int
        Number of candidate beam draws to retain for each sharing unit.
    rng : np.random.Generator
        Random generator used for conditioned or global draws.

    Returns
    -------
    dict[str, np.ndarray]
        Candidate beam pool dictionary with angles in both degrees and radians,
        plus precomputed sine and cosine tables. Every array has shape
        ``(unit_count, pool_size)``.
    """
    source_kind_arr = np.asarray(source_kind, dtype=np.int8).reshape(-1)
    source_id_arr = np.asarray(source_id, dtype=np.int32).reshape(-1)
    unit_count = int(source_kind_arr.size)
    pool_size = int(pool_size)

    beta_deg = np.full((unit_count, pool_size), np.nan, dtype=np.float32)
    alpha_deg = np.full((unit_count, pool_size), np.nan, dtype=np.float32)

    for unit in range(unit_count):
        kind = int(source_kind_arr[unit])
        sid = int(source_id_arr[unit])

        if kind == int(SAMPLER_SOURCE_GROUP):
            if 0 <= sid < int(np.asarray(sampler.group_raw_counts).size):
                p0 = int(sampler.group_ptr[sid])
                p1 = int(sampler.group_ptr[sid + 1])
                if p1 > p0:
                    draws = rng.integers(p0, p1, size=pool_size)
                    beta_deg[unit, :] = np.asarray(sampler.group_beta_pool[draws], dtype=np.float32)
                    alpha_deg[unit, :] = np.asarray(sampler.group_alpha_pool[draws], dtype=np.float32)
        elif kind == int(SAMPLER_SOURCE_BELT):
            if 0 <= sid < int(np.asarray(sampler.belt_raw_counts).size):
                p0 = int(sampler.belt_ptr[sid])
                p1 = int(sampler.belt_ptr[sid + 1])
                if p1 > p0:
                    draws = rng.integers(p0, p1, size=pool_size)
                    beta_deg[unit, :] = np.asarray(sampler.belt_beta_pool[draws], dtype=np.float32)
                    alpha_deg[unit, :] = np.asarray(sampler.belt_alpha_pool[draws], dtype=np.float32)
        elif kind == int(SAMPLER_SOURCE_GLOBAL):
            beta_flat, alpha_flat = sampler._sample_unconditional_flat(
                rng,
                pool_size,
                dtype=np.float32,
            )
            beta_deg[unit, :] = beta_flat
            alpha_deg[unit, :] = alpha_flat

    alpha_rad = np.remainder(alpha_deg * np.float32(np.pi / 180.0), np.float32(2.0 * np.pi)).astype(np.float32, copy=False)
    beta_rad = (beta_deg * np.float32(np.pi / 180.0)).astype(np.float32, copy=False)

    return {
        "beta_deg": beta_deg,
        "alpha_deg": alpha_deg,
        "beta_rad": beta_rad,
        "alpha_rad": alpha_rad,
        "sin_alpha": np.sin(alpha_rad).astype(np.float32, copy=False),
        "cos_alpha": np.cos(alpha_rad).astype(np.float32, copy=False),
        "sin_beta": np.sin(beta_rad).astype(np.float32, copy=False),
        "cos_beta": np.cos(beta_rad).astype(np.float32, copy=False),
    }


def build_conditioned_templates_cpu(
    candidate_pools: dict[str, np.ndarray],
    beta_max_rad_per_unit: Any,
    *,
    template_size: int,
    cos_min_sep: float,
    start_offsets: Any,
) -> dict[str, np.ndarray]:
    """
    Build mutually separated conditioned beam templates on CPU.

    Parameters
    ----------
    candidate_pools : dict[str, np.ndarray]
        Candidate pool dictionary returned by
        :func:`sample_conditioned_candidate_pools_cpu`.
    beta_max_rad_per_unit : Any
        Maximum allowed beam off-axis angle, in radians, for each sharing unit.
    template_size : int
        Number of template slots to retain per unit.
    cos_min_sep : float
        Cosine of the minimum beam separation angle. Candidates with larger
        cosine separation than this threshold are rejected as too close.
    start_offsets : Any
        Per-unit starting offsets used to scan the candidate pools.

    Returns
    -------
    dict[str, np.ndarray]
        Template dictionary containing retained indices, beam angles, trig
        tables, and valid counts for each sharing unit.
    """
    pool_alpha = np.asarray(candidate_pools["alpha_rad"], dtype=np.float32)
    pool_beta = np.asarray(candidate_pools["beta_rad"], dtype=np.float32)
    pool_sina = np.asarray(candidate_pools["sin_alpha"], dtype=np.float32)
    pool_cosa = np.asarray(candidate_pools["cos_alpha"], dtype=np.float32)
    pool_sinb = np.asarray(candidate_pools["sin_beta"], dtype=np.float32)
    pool_cosb = np.asarray(candidate_pools["cos_beta"], dtype=np.float32)
    beta_max_arr = np.asarray(beta_max_rad_per_unit, dtype=np.float32).reshape(-1)
    start_offsets_arr = np.asarray(start_offsets, dtype=np.int32).reshape(-1)

    unit_count = int(pool_alpha.shape[0]) if pool_alpha.ndim == 2 else 0
    pool_size = int(pool_alpha.shape[1]) if unit_count else 0

    template_idx = np.full((unit_count, template_size), -1, dtype=np.int32)
    template_alpha = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_beta = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_sina = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_cosa = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_sinb = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_cosb = np.full((unit_count, template_size), np.nan, dtype=np.float32)
    template_valid = np.zeros(unit_count, dtype=np.int32)

    for unit in range(unit_count):
        beta_max = beta_max_arr[unit]
        start = int(start_offsets_arr[unit]) if start_offsets_arr.size else 0
        for scan_offset in range(pool_size):
            if template_valid[unit] >= template_size:
                break
            pool_pos = int((start + scan_offset) % max(1, pool_size))
            beta_val = pool_beta[unit, pool_pos]
            if not np.isfinite(beta_val) or beta_val > beta_max:
                continue
            ok = True
            for template_slot in range(int(template_valid[unit])):
                cos_da = template_cosa[unit, template_slot] * pool_cosa[unit, pool_pos]
                cos_da += template_sina[unit, template_slot] * pool_sina[unit, pool_pos]
                cos_gamma = template_cosb[unit, template_slot] * pool_cosb[unit, pool_pos]
                cos_gamma += template_sinb[unit, template_slot] * pool_sinb[unit, pool_pos] * cos_da
                if cos_gamma > cos_min_sep:
                    ok = False
                    break
            if not ok:
                continue
            slot = int(template_valid[unit])
            template_idx[unit, slot] = pool_pos
            template_alpha[unit, slot] = pool_alpha[unit, pool_pos]
            template_beta[unit, slot] = beta_val
            template_sina[unit, slot] = pool_sina[unit, pool_pos]
            template_cosa[unit, slot] = pool_cosa[unit, pool_pos]
            template_sinb[unit, slot] = pool_sinb[unit, pool_pos]
            template_cosb[unit, slot] = pool_cosb[unit, pool_pos]
            template_valid[unit] += 1

    return {
        "template_idx": template_idx,
        "template_alpha_rad": template_alpha,
        "template_beta_rad": template_beta,
        "template_sin_alpha": template_sina,
        "template_cos_alpha": template_cosa,
        "template_sin_beta": template_sinb,
        "template_cos_beta": template_cosb,
        "template_valid_count": template_valid,
        "template_start_offsets": start_offsets_arr.astype(np.int32, copy=False),
    }


def assign_conditioned_beams_cpu(
    plan: ConditionedTemplatePlanCpu,
    templates: dict[str, np.ndarray],
    *,
    vis_mask_horizon: Any,
    is_co_sat: Any,
    alpha0_rad: Any,
    beta0_rad: Any,
    sina0: Any,
    cosa0: Any,
    sinb0: Any,
    cosb0: Any,
    beta_max_rad_per_sat: Any,
    n_beams: int,
    cos_min_sep: float,
    start_offsets: Any,
) -> dict[str, np.ndarray]:
    """
    Assign beam slots from conditioned templates to active CPU rows.

    Parameters
    ----------
    plan : ConditionedTemplatePlanCpu
        Template-sharing plan returned by
        :func:`make_conditioned_template_plan_cpu`.
    templates : dict[str, np.ndarray]
        Template dictionary returned by :func:`build_conditioned_templates_cpu`.
    vis_mask_horizon : Any
        Visibility mask for the batch with shape ``(time, satellite)``.
    is_co_sat : Any
        Boolean mask identifying the serving co-satellite row in the same
        shape as ``vis_mask_horizon``.
    alpha0_rad, beta0_rad, sina0, cosa0, sinb0, cosb0 : Any
        Precomputed serving-link beam geometry for the same ``(time, satellite)``
        rows.
    beta_max_rad_per_sat : Any
        Maximum allowed beam off-axis angle, in radians, for each satellite.
    n_beams : int
        Maximum number of beams to assign per row.
    cos_min_sep : float
        Cosine of the minimum beam separation angle.
    start_offsets : Any
        Per-active-row template scan offsets.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing beam indices, beam angles, and valid beam counts.
        The returned arrays have shapes ``(time, satellite, n_beams)`` for the
        beam tables and ``(time, satellite)`` for ``"beam_valid"``.
    """
    flat_shape = tuple(int(v) for v in plan.flat_shape)
    flat_rows = int(np.prod(flat_shape, dtype=np.int64))

    template_counts = np.asarray(templates["template_valid_count"], dtype=np.int32)
    template_idx = np.asarray(templates["template_idx"], dtype=np.int32)
    template_alpha = np.asarray(templates["template_alpha_rad"], dtype=np.float32)
    template_beta = np.asarray(templates["template_beta_rad"], dtype=np.float32)
    template_sina = np.asarray(templates["template_sin_alpha"], dtype=np.float32)
    template_cosa = np.asarray(templates["template_cos_alpha"], dtype=np.float32)
    template_sinb = np.asarray(templates["template_sin_beta"], dtype=np.float32)
    template_cosb = np.asarray(templates["template_cos_beta"], dtype=np.float32)
    row_start_offsets = np.asarray(start_offsets, dtype=np.int32).reshape(-1)

    beam_idx_flat = np.full((flat_rows, n_beams), -1, dtype=np.int32)
    beam_alpha_flat = np.full((flat_rows, n_beams), np.nan, dtype=np.float32)
    beam_beta_flat = np.full((flat_rows, n_beams), np.nan, dtype=np.float32)
    beam_valid_flat = np.zeros(flat_rows, dtype=np.int16)

    beta_max_row = np.broadcast_to(np.asarray(beta_max_rad_per_sat, dtype=np.float32)[None, :], flat_shape).reshape(-1)
    vis_flat = np.asarray(vis_mask_horizon, dtype=bool).reshape(-1)
    co_flat = np.asarray(is_co_sat, dtype=bool).reshape(-1)
    alpha0_flat = np.asarray(alpha0_rad, dtype=np.float32).reshape(-1)
    beta0_flat = np.asarray(beta0_rad, dtype=np.float32).reshape(-1)
    sina0_flat = np.asarray(sina0, dtype=np.float32).reshape(-1)
    cosa0_flat = np.asarray(cosa0, dtype=np.float32).reshape(-1)
    sinb0_flat = np.asarray(sinb0, dtype=np.float32).reshape(-1)
    cosb0_flat = np.asarray(cosb0, dtype=np.float32).reshape(-1)

    for active_row_idx, flat_row in enumerate(plan.active_rows.tolist()):
        if not vis_flat[flat_row]:
            continue
        beta_max = float(beta_max_row[flat_row])
        unit = int(plan.row_to_unit[active_row_idx])
        template_count = int(template_counts[unit])
        if template_count <= 0:
            continue

        n_acc = 0
        if co_flat[flat_row] and beta0_flat[flat_row] <= beta_max:
            beam_idx_flat[flat_row, 0] = -2
            beam_alpha_flat[flat_row, 0] = alpha0_flat[flat_row]
            beam_beta_flat[flat_row, 0] = beta0_flat[flat_row]
            n_acc = 1

        start = int(row_start_offsets[active_row_idx]) if row_start_offsets.size else 0
        for scan_step in range(template_count):
            if n_acc >= n_beams:
                break
            pos = int((start + scan_step) % template_count)
            cand_idx = int(template_idx[unit, pos])
            beta_cand = float(template_beta[unit, pos])
            if cand_idx < 0 or not np.isfinite(beta_cand) or beta_cand > beta_max:
                continue
            if co_flat[flat_row] and n_acc > 0:
                cos_da = cosa0_flat[flat_row] * template_cosa[unit, pos] + sina0_flat[flat_row] * template_sina[unit, pos]
                cos_gamma0 = cosb0_flat[flat_row] * template_cosb[unit, pos]
                cos_gamma0 += sinb0_flat[flat_row] * template_sinb[unit, pos] * cos_da
                if cos_gamma0 > cos_min_sep:
                    continue
            beam_idx_flat[flat_row, n_acc] = cand_idx
            beam_alpha_flat[flat_row, n_acc] = template_alpha[unit, pos]
            beam_beta_flat[flat_row, n_acc] = np.float32(beta_cand)
            n_acc += 1

        beam_valid_flat[flat_row] = np.int16(n_acc)

    return {
        "beam_idx": beam_idx_flat.reshape(flat_shape + (n_beams,)),
        "beam_alpha_rad": beam_alpha_flat.reshape(flat_shape + (n_beams,)),
        "beam_beta_rad": beam_beta_flat.reshape(flat_shape + (n_beams,)),
        "beam_valid": beam_valid_flat.reshape(flat_shape),
    }
