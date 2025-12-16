"""
satsim.py

This module provides supplemental functions to model satellite system behavior.
Author: boris.sorokin <mralin@protonmail.com>
Date: 29-04-2025
"""

import numpy as np
from astropy import units as u
from pycraf.utils import ranged_quantity_input

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

# ===========================================================================
# Internal helpers
# ===========================================================================

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


def compute_sat_cell_links(
    sat_topo: np.ndarray,
    min_elevation: u.Quantity = 30 * u.deg,
    Nco: int = 1,
    cell_observer_offset: int = 1,
    Nbeam: int | None = None,
    BA_mode: bool = False,
    BA_switchoff_angle = None,
    BA_redirection_angle = None,
    BA_redirection_angle_separation = None,
    BA_redirection_avoid_cells = None,
    tel_az = None,
    tel_el = None,
    sat_azel = None,
    selection_strategy: str = "random",
) -> np.ndarray:
    """
    Pair satellites with cells under elevation and capacity constraints.

    Parameters
    ----------
    sat_topo : np.ndarray
        Output of ``cysgp4.propagate_many`` with shape:
            (time, observer, satellite, parameters)
             ^^^^^  optional - if missing, function treats array as single
                      time slice and squeezes time axis on return.

        **Important** - the elevation (degrees) must reside at
        index **1** along the last axis (as used by *cysgp4*).

    min_elevation : Quantity
        Minimum elevation (deg).  Satellite links below this angle are
        considered non-operational and therefore impossible.

    Nco : int
        Desired number of simultaneous satellite links *per cell*.

    cell_observer_offset : int, optional
        Skip this many observers before the first cell.  Example:
        if observer 0 is a protected radio-astronomy station and you only
        want links for grid cells starting at observer 1 ⇒ set to 1.

    Nbeam : int | None
        Maximum number of cells a *single* satellite can serve *per time
        slice*.  ``None`` means “unlimited” (faster path).

    selection_strategy : {"random", "max_elevation"}, optional
        How to rank candidate satellite-cell links before assignment:

        * ``"random"`` (default) - legacy behaviour: each eligible pair
          gets a random weight in [0, 1); lower is “better”.
        * ``"max_elevation"`` - prefer higher elevation:
          within the set of eligible pairs we treat larger elevation
          angles as “better”.

        Case-insensitive; aliases ``"rng"``, ``"rand"`` for random and
        ``"highest_elevation"``, ``"max_el"``, ``"elevation"``, ``"elev"``
        for the elevation-based strategy are also accepted.

    Returns
    -------
    np.ndarray[int32]
        If *sat_topo* contained time axis → shape is
        ``(time, n_cells, Nco)``.  Otherwise ``(n_cells, Nco)``.

        Each entry is the **index of the satellite** chosen for that
        (time, cell, slot).  If no satellite can be assigned, the entry
        is **-1**.

    Raises
    ------
    ValueError
        * sat_topo is not 3- or 4-D
        * Nco ≤ 0
        * Nbeam is 0 or negative

    Notes
    -----
    * Random generator is NumPy default RNG; set a seed **before calling**
      to get reproducible assignments.

          >>> np.random.seed(42)  # reproducible
          >>> compute_sat_cell_links(...)

    * For very large grids the greedy branch may still dominate runtime.
      Numba can JIT the core loop automatically.

    * Memory footprint:
        float32 weights  → 4 bytes * (T·C·S)  
        int32 counters   → 4 bytes * T·(C+S)  
        int32 output     → 4 bytes * T·C·Nco

    Examples
    --------
    Unlimited beams (legacy behaviour)
    >>> links = compute_sat_cell_links(sat_topo, Nco=1, Nbeam=None)

    Beam-limited, every satellite feeds at most 16 cells
    >>> links = compute_sat_cell_links(sat_topo, Nco=2, Nbeam=16)

    Checking result for *t = 0*
    >>> unique, counts = np.unique(
    ...     links[0][links[0] >= 0],     # ignore -1 entries
    ...     return_counts=True
    ... )
    >>> dict(zip(unique, counts))  # satellite → number of cells served
    {0: 16, 1: 15, 2: 16, ...}
    """
    # ------------------------------------------------------------------
    # 0. Input checks
    # ------------------------------------------------------------------
    if Nco <= 0:
        raise ValueError("Nco must be positive.")
    if Nbeam is not None and Nbeam <= 0:
        raise ValueError("Nbeam must be positive or None.")
    if sat_topo.ndim not in (3, 4):
        raise ValueError("sat_topo must be a 3-D or 4-D array.")
    
    # ------------------------------------------------------------------
    # 1. Ensuring having a leading time axis  (shape becomes (T, obs, sat, P))
    # ------------------------------------------------------------------
    single_time_input = sat_topo.ndim == 3
    if single_time_input:
        sat_topo = sat_topo[np.newaxis, ...]

    # ------------------------------------------------------------------
    # 2. Extract only observers that correspond to *cells*
    # ------------------------------------------------------------------
    # Slice away protected stations etc.
    cells_view = sat_topo[:, cell_observer_offset:, :, :]  # (T, cell, sat, P)

    # ------------------------------------------------------------------
    # 3. Compute visibility mask  (True ↔ satellite above elevation limit)
    # ------------------------------------------------------------------
    elevation_deg = cells_view[..., 1]                                 # (T, C, S)
    visibility = elevation_deg > min_elevation.to(u.deg).value         # bool
    
    # ------------------------------------------------------------------
    # 3b. BA-mode ── build visibility / avoidance / redirection masks
    # ------------------------------------------------------------------
    if BA_mode:
    # true_angular_distance_auto returns (T, Sky, C, S)
        # # 3b-1.  Angular distance sky-point  ↔  sat-to-cell direction
        from .gpu_accel import true_angular_distance_auto
        tel_az_local  = tel_az[:, :, np.newaxis, np.newaxis] * u.deg     # (T ,Sky,1 ,1 )
        tel_el_local  = tel_el[:, :, np.newaxis, np.newaxis] * u.deg
        az_from_ras   = cells_view[..., 0][:, np.newaxis, :, :] * u.deg  # (T, 1, C, S)
        el_from_ras   = cells_view[..., 1][:, np.newaxis, :, :] * u.deg

        separation    = true_angular_distance_auto(
                        tel_az_local, tel_el_local, az_from_ras, el_from_ras
                        )                                           # (T, Sky, C, S)
        # 3b-2.  *Switch-off* region  – nothing gets through here
        # this mask shows satellites that should not be used at all
        avoid_mask    = separation <= BA_switchoff_angle           # (T, Sky, C, S)

        # 3b-3.  *Redirection* region  – allowed, but with extra rules
        redir_region = (separation > BA_switchoff_angle) & (separation <= BA_redirection_angle)
        ras_az_sat = sat_azel[:, 0, :, 0][:, np.newaxis, np.newaxis, :]
        ras_el_sat = sat_azel[:, 0, :, 1][:, np.newaxis, np.newaxis, :]

        sat_antenna_az_BA = sat_azel[:,np.newaxis,cell_observer_offset:,:,0]
        sat_antenna_el_BA = sat_azel[:,np.newaxis,cell_observer_offset:,:,1]

        sat2cell_sep = true_angular_distance_auto(
                    ras_az_sat * u.deg,                            # (T, Sky, 1, S)
                    (90-ras_el_sat) * u.deg,   
                    sat_antenna_az_BA * u.deg,       # (T, Sky, C, S)
                    (90-sat_antenna_el_BA) * u.deg
                )                                                       # (T ,Sky,C ,S)
        # 3b-4.  Setting rule that we can only use satellites from redir_region
        # if they don't point too close to the RAS station
        redir_ban = sat2cell_sep <= BA_redirection_angle_separation

        # 3b-5.  preparing block_mask to cancel possibilities rejected by BA logic
        block_mask = avoid_mask | (redir_region & redir_ban)
        
        # # eligible satellite-cell pairs should be above minimum elevation angle,
        # # and not blocked by BA logic
        eligibility   = visibility[:, np.newaxis, :, :] & ~block_mask  # (T, Sky, C, S)

        if BA_redirection_avoid_cells is not None:
            prot = np.atleast_1d(BA_redirection_avoid_cells).astype(int)
            eligibility[:, :, prot, :] = False
        T, Sky, C, S  = eligibility.shape
    else:
        eligibility   = visibility                                  # (T, C, S)
        T, C, S       = eligibility.shape
        Sky = 1

    # ------------------------------------------------------------------
    # 4. Create random *weights* for every (time, cell, sat) candidate
    #    • Visible pairs get  random [0, 1)
    #    • Invisible pairs get +∞ (they will never win)
    #    Stored as float32 to save memory since precision is not needed.
    # ------------------------------------------------------------------
    strategy = (selection_strategy or "random").lower()
    is_random = strategy in ("random", "rng", "rand")
    is_max_el = strategy in (
        "max_elevation", "highest_elevation", "max_el", "elevation", "elev"
    )
    if is_random:
        rng = np.random.default_rng()
        rand_weights = rng.random(size=eligibility.shape, dtype=np.float32)
        weights = np.where(eligibility, rand_weights, np.inf).astype(np.float32)
    elif is_max_el:
        # High elevation ↔ better. We use negative elevation so that
        # argpartition on the smallest values picks highest elevation first.
        if BA_mode:
            # elevation_deg : (T, C, S)  → broadcast to (T, Sky, C, S)
            elev_full = np.broadcast_to(
                elevation_deg[:, np.newaxis, :, :],
                eligibility.shape,
            )
        else:
            elev_full = elevation_deg

        weights = np.where(
            eligibility,
            -elev_full.astype(np.float32),
            np.inf,
        ).astype(np.float32)

    else:
        raise ValueError(
            f"Unknown selection_strategy={selection_strategy!r}; "
            "supported: 'random', 'max_elevation'"
        )

    # Prepare output array:  -1 means “no satellite assigned”
    # If used directly, by default that would link to the last satellite in array, 
    # but if not used as an index directly that would help find those satellites
    # Ensure satellite array will have extra dummy satellite at inf distance
    if BA_mode:
        assignment = -np.ones((T, Sky, C, Nco), dtype=np.int32)
    else:
        assignment = -np.ones((T, C, Nco),      dtype=np.int32)

    # ------------------------------------------------------------------
    # 5. Two different algorithms depending on Nbeam
    # ------------------------------------------------------------------

    # 5a. ***Unlimited beams***  — fastest path (pure NumPy, no Python loop)
    # ------------------------------------------------------------------
    if Nbeam is None:
        # np.argpartition gives indices of the Nco smallest weights per last axis
        part_idx = np.argpartition(weights, Nco - 1, axis=-1)[..., :Nco]    # (T, C, Nco)
        part_w   = np.take_along_axis(weights, part_idx, axis=-1)           # (T, C, Nco)

        # Sort those Nco weights for each cell → deterministic ordering
        order = np.argsort(part_w, axis=-1)                                 # (T, C, Nco)
        idx_sorted = np.take_along_axis(part_idx, order, axis=-1)           # (T, C, Nco)

        # Only accept links where weight is finite
        valid_mask = np.take_along_axis(part_w, order, axis=-1) < np.inf
        assignment[valid_mask] = idx_sorted[valid_mask]
    
    # 5b. ***Beam-limited***  — needs global fairness across cells
    # ------------------------------------------------------------------
    else:
        if BA_mode:
            C_eff = Sky * C 
        else:
            C_eff = C 
        # (i) Flatten  (C, S)  →  (C·S)  so that numpy.sort works row-wise
        flat_weights = weights.reshape(T, C_eff * S)                # (T, C·S)

        # (ii) Sort **once** — ascending weight per time slice
        order = np.argsort(flat_weights, axis=1)                # (T, C·S) int32

        # (iii) Flatten visibility the same way for quick lookup
        vis_flat = eligibility.reshape(T, C_eff * S)                 # (T, C·S) bool

        # (iv) Run greedy assignment engine (Numba if available)
        greedy_fn = _greedy_numba if njit else _greedy_python
        assign = greedy_fn(order, vis_flat, C, S, Nco, Nbeam)

        assignment[...] = assign.reshape((T, Sky, C, Nco)) if BA_mode else assign

    # ------------------------------------------------------------------
    # 6. Squeeze out synthetic time axis if input didn't had it originally
    # ------------------------------------------------------------------
    if single_time_input:
        assignment = assignment[0]  # → shape (C, Nco)

    return assignment

def compute_sat_cell_links_parallel(
    sat_topo: np.ndarray,
    min_elevation: u.Quantity = 30 * u.deg,
    Nco: int = 1,
    cell_observer_offset: int = 1,
    Nbeam: int | None = None,
    BA_mode: bool = False,
    BA_switchoff_angle = None,
    tel_az = None,
    tel_el = None
) -> np.ndarray:
    """
    **Parallel** sibling of :func:`compute_sat_cell_links`.

    Quick recap of the problem
    --------------------------
    * We have *many time-steps* **T**, *many ground cells* **C**, and
      *many satellites* **S**.
    * Each *visible* (cell, sat) pair gets a **random weight** - smaller
      means “pick me first”.
    * Each cell wants **`Nco` links**.
    * Optionally each satellite can feed at most **`Nbeam` cells**.

    How this parallel version works
    -------------------------------
    =======================  ===============================================
    If `Nbeam` is **None**   We *flatten* the two fully-independent axes  
                             (time, cell) into one long list  
                             ``rows = T * C``.  
                             Each row is processed by exactly one thread of
                             :func:`_unlimited_parallel`.  This gives very
                             even load balancing, because there are usually
                             far more rows than CPU cores.
    If `Nbeam` is **given**  Only *time slices* are independent (cells
                             share satellite counters).  
                             We therefore keep the greedy algorithm, but
                             let OpenMP assign whole slices to threads via
                             :func:`_greedy_parallel`.
    =======================  ===============================================

    All other behaviour (inputs, outputs, error handling) is identical to
    the serial function, so you can switch by changing only the function
    name in your code.

    Returns
    -------
    Same ndarray as the serial version: shape ``(T, C, Nco)`` or
    ``(C, Nco)`` if input lacked a time axis; entries are satellite
    indices or **-1** if the cell could not fill that slot.
    """
    # 1. Basic validation (identical to serial routine) -------------------
    if Nco <= 0:
        raise ValueError("Nco must be positive.")
    if Nbeam is not None and Nbeam <= 0:
        raise ValueError("Nbeam must be positive or None.")
    if sat_topo.ndim not in (3, 4):
        raise ValueError("sat_topo must be 3-D or 4-D.")

    # 2. Ensure leading time axis   (adds axis if absent) -----------------
    single_time = sat_topo.ndim == 3
    if single_time:
        sat_topo = sat_topo[np.newaxis, ...]

    # 3. Focus on observers that are *cells* (skip RAS station etc.) ------
    cells_view = sat_topo[:, cell_observer_offset:, :, :]   # (T, C, S, P)

    # 4. Create visibility mask: True if elevation > min_elevation -------
    elev_deg   = cells_view[..., 1]                         # (T, C, S)
    visibility = elev_deg > min_elevation.to(u.deg).value   # bool
    if BA_mode:
        print("BA mode processing")
        from .gpu_accel import true_angular_distance_auto
        separation_angle=true_angular_distance_auto(tel_az,tel_el)
        pass

    T, C, S = visibility.shape                              # unpack dims

    # 5. Generate random weights, +∞ for invisible pairs ------------------
    rng     = np.random.default_rng()
    weights = np.where(
        visibility,
        rng.random(size=visibility.shape, dtype=np.float32),
        np.inf
    ).astype(np.float32)

    # 6. Choose algorithm based on Nbeam ---------------------------------
    if Nbeam is None:                                       # UNLIMITED
        # -- flatten time & cell into one dimension so each row is independent
        rows_out  = _unlimited_parallel(weights.reshape(T * C, S), S, Nco)
        assignment = rows_out.reshape(T, C, Nco)

    else:                                                   # BEAM-LIMITED
        order    = np.argsort(weights.reshape(T, C * S), axis=1)
        vis_flat = visibility.reshape(T, C * S)
        assignment = _greedy_parallel(order, vis_flat, C, S, Nco, Nbeam)

    # 7. Remove synthetic time axis if caller provided 3-D input ----------
    if single_time:
        assignment = assignment[0]

    return assignment

def compute_sat_cell_links_auto(
    sat_topo: np.ndarray,
    min_elevation: u.Quantity = 30 * u.deg,
    Nco: int = 1,
    cell_observer_offset: int = 1,
    Nbeam: int | None = None,
    size_threshold: int = 50*100*720*1000000000000, # temp switch off for 
    *,
    BA_mode: bool = False,
    BA_switchoff_angle = None,
    BA_redirection_angle = None,
    BA_redirection_angle_separation = None,
    BA_redirection_avoid_cells = None,
    tel_az = None,
    tel_el = None,
    sat_azel = None,
    selection_strategy: str = "random",
) -> np.ndarray:
    """
    **Smart wrapper** that selects the fastest implementation.

    Decision rule (simple but effective)
    ------------------------------------
    1. Compute **problem size** = ``time_steps * cells * satellites``.
    2. If that number is **greater** than `size_threshold`
       *and* Numba is available → use the threaded
       :func:`compute_sat_cell_links_parallel`.
    3. Otherwise → use the plain serial
       :func:`compute_sat_cell_links`.

    Parameters
    ----------
    sat_topo, min_elevation, Nco, cell_observer_offset, Nbeam
        Same meaning as in the other two functions.  They are simply
        forwarded unchanged.
    size_threshold : int, optional
        Cut-off value for switching to the parallel path.
        Default ``500 000`` (half a million) was chosen from benchmark
        experience on a typical 8-core laptop.  
        Increase it if you see parallel overhead dominate, decrease it
        if you have many cores.
    selection_strategy : {"random", "max_elevation"}, optional
        Passed through to :func:`compute_sat_cell_links`.  Currently the
        parallel backend supports only the random strategy; if you request
        a different strategy the serial backend is chosen regardless of
        `size_threshold`.

    Returns
    -------
    ndarray
        Exactly the same output you would get from the chosen backend.

    Notes
    -----
    * The check is extremely fast (just multiplies three integers), so
      overhead is negligible.
    * You can still force parallel execution by calling the parallel
      function directly, or adjust the threshold at runtime.
    """
    # -- Determine dimensions *without* copying data ---------------------
    if sat_topo.ndim == 3:
        T, obs, S = sat_topo.shape
        C = obs - cell_observer_offset        # only cells, skip RAS
    elif sat_topo.ndim == 4:
        T, obs, S, _ = sat_topo.shape
        C = obs - cell_observer_offset
    else:
        raise ValueError("sat_topo must be a 3- or 4-D array.")

    problem_size = T * C * S                 # scalar integer

    strategy = (selection_strategy or "random").lower()
    supports_parallel = strategy in ("random", "rng", "rand")

    use_parallel = (
        njit is not None and          # Numba installed ⇒ parallel possible
        problem_size > size_threshold and
        supports_parallel             # only random strategy in parallel path
    )

    if use_parallel:
        return compute_sat_cell_links_parallel(
            sat_topo, min_elevation, Nco,
            cell_observer_offset=cell_observer_offset,
            Nbeam=Nbeam, BA_mode=BA_mode, BA_switchoff_angle=BA_switchoff_angle,
            tel_az=tel_az, tel_el=tel_el
        )
    else:
        return compute_sat_cell_links(
            sat_topo, min_elevation, Nco,
            cell_observer_offset=cell_observer_offset,
            Nbeam=Nbeam, BA_mode=BA_mode, BA_switchoff_angle=BA_switchoff_angle,
            BA_redirection_angle=BA_redirection_angle, BA_redirection_angle_separation=BA_redirection_angle_separation,
            BA_redirection_avoid_cells=BA_redirection_avoid_cells,
            tel_az=tel_az, tel_el=tel_el, sat_azel=sat_azel,            
            selection_strategy=selection_strategy
        )