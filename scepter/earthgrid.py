"""
Earth-grid and Earth-surface geometry helpers for SCEPTer workflows.

It includes routines for calculating the -3 dB antenna footprint on Earth as well
as functions to generate a full hexagon grid on the globe.

Numba is used as an optional accelerator for computational hotspots when
available. When numba is not installed, the pure-Python implementations are used
with identical numerical behavior.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>

This module remains part of the GPLv3-licensed SCEPTer project. The author
attribution above does not change the project-wide license.

Notes
-----
Hexgrid functions are partially based on code developed by Benjamin Winkel,
MPIfR.
"""

import json
import hashlib
import warnings
from collections import OrderedDict, deque
from functools import lru_cache
from importlib.resources import files
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from pycraf import conversions as cnv
from pycraf import geometry, pathprof
from pycraf.utils import ranged_quantity_input
from scipy.spatial import cKDTree
from scepter.antenna import calculate_beamwidth_1d, calculate_beamwidth_2d

try:  # Optional acceleration if numba is installed
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only when numba is absent
    HAS_NUMBA = False

def njit(*args, **kwargs):  # type: ignore[override]
        """Fallback decorator used when numba is unavailable."""

        def wrapper(func):
            return func

        return wrapper


_GEOGRAPHY_MASK_MODE_ALIASES = {
    "land_plus_shoreline_buffer": "land_plus_nearshore_sea",
}
_GEOGRAPHY_MASK_MODES = {"none", "land_only", "land_plus_nearshore_sea"}
_COASTLINE_BACKENDS = {"vendored", "cartopy"}
_HEXGRID_REUSE_SHIFT_PAIRS: dict[int, tuple[int, int]] = {
    1: (1, 0),
    3: (1, 1),
    4: (2, 0),
    7: (2, 1),
    9: (3, 0),
    12: (2, 2),
    13: (3, 1),
    16: (4, 0),
    19: (3, 2),
}
_NATURAL_EARTH_RESOURCE_NAMES = {
    "land": "ne_10m_land.geojson",
    "coastline": "ne_10m_coastline.geojson",
}
_GEOGRAPHY_CLASSIFICATION_CACHE_MAX = 6
_GEOGRAPHY_CLASSIFICATION_CACHE: OrderedDict[
    tuple[Any, ...],
    dict[str, np.ndarray],
] = OrderedDict()


def _array_cache_token(arr: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    arr_use = np.ascontiguousarray(np.asarray(arr))
    digest = hashlib.blake2b(
        arr_use.view(np.uint8),
        digest_size=16,
    ).hexdigest()
    return (tuple(arr_use.shape), str(arr_use.dtype), digest)


def _copy_geography_classification_result(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value).copy()
        for key, value in payload.items()
    }


def _normalise_geography_mask_mode(mode: str | None) -> str:
    mode_normalized = str(mode or "none").strip().lower()
    mode_normalized = _GEOGRAPHY_MASK_MODE_ALIASES.get(mode_normalized, mode_normalized)
    if mode_normalized not in _GEOGRAPHY_MASK_MODES:
        raise ValueError(
            "geography_mask_mode must be one of "
            f"{sorted(_GEOGRAPHY_MASK_MODES | set(_GEOGRAPHY_MASK_MODE_ALIASES))!r}; got {mode!r}."
        )
    return mode_normalized


def _normalise_coastline_backend(backend: str | None) -> str:
    backend_normalized = str(backend or "vendored").strip().lower()
    if backend_normalized not in _COASTLINE_BACKENDS:
        raise ValueError(
            "coastline_backend must be one of "
            f"{sorted(_COASTLINE_BACKENDS)!r}; got {backend!r}."
        )
    return backend_normalized


def _resolve_hexgrid_reuse_shift_pair(reuse_factor: int) -> tuple[int, int]:
    reuse_factor_i = int(reuse_factor)
    if reuse_factor_i not in _HEXGRID_REUSE_SHIFT_PAIRS:
        raise ValueError(
            "reuse_factor must be one of "
            f"{sorted(_HEXGRID_REUSE_SHIFT_PAIRS)!r}; got {reuse_factor!r}."
        )
    return _HEXGRID_REUSE_SHIFT_PAIRS[reuse_factor_i]


def _local_tangent_plane_xy_km(
    longitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
    *,
    ref_lon_deg: float,
    ref_lat_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat points onto a small local tangent plane in kilometres."""
    lon = np.asarray(longitudes_deg, dtype=np.float64)
    lat = np.asarray(latitudes_deg, dtype=np.float64)
    delta_lon = np.deg2rad(lon - float(ref_lon_deg))
    delta_lon = (delta_lon + np.pi) % (2.0 * np.pi) - np.pi
    delta_lat = np.deg2rad(lat - float(ref_lat_deg))
    radius_km = float(R_earth.to_value(u.km))
    x_km = radius_km * np.cos(np.deg2rad(float(ref_lat_deg))) * delta_lon
    y_km = radius_km * delta_lat
    return x_km.astype(np.float64, copy=False), y_km.astype(np.float64, copy=False)


def _local_tangent_plane_lonlat_from_xy_km(
    x_km: np.ndarray,
    y_km: np.ndarray,
    *,
    ref_lon_deg: float,
    ref_lat_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the local tangent-plane projection back to lon/lat degrees."""
    x_arr = np.asarray(x_km, dtype=np.float64)
    y_arr = np.asarray(y_km, dtype=np.float64)
    radius_km = float(R_earth.to_value(u.km))
    ref_lat_rad = np.deg2rad(float(ref_lat_deg))
    cos_ref_lat = max(np.cos(ref_lat_rad), 1.0e-12)
    lat_deg = float(ref_lat_deg) + np.rad2deg(y_arr / radius_km)
    lon_deg = float(ref_lon_deg) + np.rad2deg(x_arr / (radius_km * cos_ref_lat))
    lon_deg = (lon_deg + 180.0) % 360.0 - 180.0
    return lon_deg.astype(np.float64, copy=False), lat_deg.astype(np.float64, copy=False)


def _cube_round_axial(q_float: np.ndarray, r_float: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round fractional axial coordinates using cube-coordinate rounding."""
    q_arr = np.asarray(q_float, dtype=np.float64)
    r_arr = np.asarray(r_float, dtype=np.float64)
    x = q_arr
    z = r_arr
    y = -x - z
    rx = np.rint(x)
    ry = np.rint(y)
    rz = np.rint(z)

    dx = np.abs(rx - x)
    dy = np.abs(ry - y)
    dz = np.abs(rz - z)

    replace_x = (dx > dy) & (dx > dz)
    replace_y = (~replace_x) & (dy > dz)
    replace_z = ~(replace_x | replace_y)

    rx[replace_x] = -ry[replace_x] - rz[replace_x]
    ry[replace_y] = -rx[replace_y] - rz[replace_y]
    rz[replace_z] = -rx[replace_z] - ry[replace_z]
    return rx.astype(np.int32, copy=False), rz.astype(np.int32, copy=False)


def _infer_hexgrid_axial_coordinates(
    x_km: np.ndarray,
    y_km: np.ndarray,
    *,
    point_spacing_km: float,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Infer integer axial coordinates for a locally regular hexgrid."""
    spacing = float(point_spacing_km)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("point_spacing_km must be finite and > 0.")
    sqrt3 = float(np.sqrt(3.0))

    pointy_q = np.asarray(x_km, dtype=np.float64) / spacing - np.asarray(y_km, dtype=np.float64) / (
        sqrt3 * spacing
    )
    pointy_r = (2.0 * np.asarray(y_km, dtype=np.float64)) / (sqrt3 * spacing)
    pointy_q_i, pointy_r_i = _cube_round_axial(pointy_q, pointy_r)
    pointy_x = spacing * (pointy_q_i.astype(np.float64) + 0.5 * pointy_r_i.astype(np.float64))
    pointy_y = spacing * (sqrt3 / 2.0) * pointy_r_i.astype(np.float64)
    pointy_residual = float(
        np.median(
            (np.asarray(x_km, dtype=np.float64) - pointy_x) ** 2
            + (np.asarray(y_km, dtype=np.float64) - pointy_y) ** 2
        )
    )

    flat_q = (2.0 * np.asarray(x_km, dtype=np.float64)) / (sqrt3 * spacing)
    flat_r = np.asarray(y_km, dtype=np.float64) / spacing - 0.5 * flat_q
    flat_q_i, flat_r_i = _cube_round_axial(flat_q, flat_r)
    flat_x = spacing * (sqrt3 / 2.0) * flat_q_i.astype(np.float64)
    flat_y = spacing * (flat_r_i.astype(np.float64) + 0.5 * flat_q_i.astype(np.float64))
    flat_residual = float(
        np.median(
            (np.asarray(x_km, dtype=np.float64) - flat_x) ** 2
            + (np.asarray(y_km, dtype=np.float64) - flat_y) ** 2
        )
    )

    if pointy_residual <= flat_residual:
        return pointy_q_i, pointy_r_i, "pointy", pointy_residual
    return flat_q_i, flat_r_i, "flat", flat_residual


def _estimate_local_hexgrid_spacing_km(
    longitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
    *,
    ref_lon_deg: float,
    ref_lat_deg: float,
) -> float:
    """Estimate the hex-neighbour spacing of a prepared hexgrid near *ref*.

    Projects a local subset (within ~20 deg of the reference) onto a tangent
    plane, queries k=7 (self + 6 hex neighbours), and returns the median
    across cells of the per-cell *median* neighbour distance.  Using median
    of 6 neighbours (rather than only the single nearest) avoids the bias
    caused by non-uniform row/column spacing in the icosahedral grid.
    """
    lons = np.asarray(longitudes_deg, dtype=np.float64).ravel()
    lats = np.asarray(latitudes_deg, dtype=np.float64).ravel()
    n = int(lons.size)
    if n < 2:
        return 1.0

    # Restrict to cells within ~20 deg of the reference to avoid tangent-
    # plane distortion; fall back to the full set if too few remain.
    dlon = np.abs(((lons - float(ref_lon_deg)) + 180.0) % 360.0 - 180.0)
    dlat = np.abs(lats - float(ref_lat_deg))
    local_mask = (dlon < 20.0) & (dlat < 20.0)
    if int(np.count_nonzero(local_mask)) < 20:
        local_mask = np.ones(n, dtype=bool)

    x_km, y_km = _local_tangent_plane_xy_km(
        lons[local_mask],
        lats[local_mask],
        ref_lon_deg=float(ref_lon_deg),
        ref_lat_deg=float(ref_lat_deg),
    )
    coords = np.column_stack([x_km, y_km])
    k = min(7, int(coords.shape[0]))  # self + up to 6 hex neighbours
    if k < 2:
        return 1.0
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=k)
    # distances[:,0] is self (==0), distances[:,1:] are neighbours
    nn = np.asarray(distances[:, 1:], dtype=np.float64)
    finite_mask = np.isfinite(nn) & (nn > 0.0)
    # per-cell median of valid neighbour distances
    nn_masked = np.where(finite_mask, nn, np.nan)
    per_cell_median = np.nanmedian(nn_masked, axis=1)
    valid = np.isfinite(per_cell_median) & (per_cell_median > 0.0)
    if not np.any(valid):
        return 1.0
    return float(np.median(per_cell_median[valid]))


def _reuse_coset_key(q_coord: int, r_coord: int, *, reuse_factor: int, shift_pair: tuple[int, int]) -> tuple[int, int]:
    """Return a stable coset key for one axial cell under the reuse lattice."""
    i_shift, j_shift = (int(shift_pair[0]), int(shift_pair[1]))
    modulus = int(reuse_factor)
    key_a = ((i_shift + j_shift) * int(q_coord) + j_shift * int(r_coord)) % modulus
    key_b = (-j_shift * int(q_coord) + i_shift * int(r_coord)) % modulus
    return int(key_a), int(key_b)


def _enumerate_reuse_cluster_slots(
    *,
    reuse_factor: int,
    shift_pair: tuple[int, int],
    anchor_slot: int,
) -> tuple[dict[tuple[int, int], int], list[dict[str, int]]]:
    coset_to_slot: dict[tuple[int, int], int] = {}
    cluster_representatives: list[dict[str, int]] = []
    axial_queue: deque[tuple[int, int]] = deque([(0, 0)])
    seen_axial: set[tuple[int, int]] = {(0, 0)}
    neighbor_steps = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
    while axial_queue and len(coset_to_slot) < int(reuse_factor):
        q_curr, r_curr = axial_queue.popleft()
        key = _reuse_coset_key(
            q_curr,
            r_curr,
            reuse_factor=int(reuse_factor),
            shift_pair=shift_pair,
        )
        if key not in coset_to_slot:
            slot_id = len(coset_to_slot)
            coset_to_slot[key] = int(slot_id)
            cluster_representatives.append(
                {
                    "slot_id": int((slot_id + int(anchor_slot)) % int(reuse_factor)),
                    "base_slot_id": int(slot_id),
                    "axial_q": int(q_curr),
                    "axial_r": int(r_curr),
                }
            )
        for dq, dr in neighbor_steps:
            candidate = (int(q_curr + dq), int(r_curr + dr))
            if candidate in seen_axial:
                continue
            seen_axial.add(candidate)
            axial_queue.append(candidate)
    return coset_to_slot, cluster_representatives


def _transform_axial_symmetry(
    q_coords: np.ndarray,
    r_coords: np.ndarray,
    *,
    rotation_steps: int,
    reflected: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x_coords = np.asarray(q_coords, dtype=np.int64)
    z_coords = np.asarray(r_coords, dtype=np.int64)
    y_coords = -x_coords - z_coords
    if reflected:
        y_coords, z_coords = z_coords, y_coords
    for _ in range(int(rotation_steps) % 6):
        x_coords, y_coords, z_coords = (-z_coords), (-x_coords), (-y_coords)
    return x_coords.astype(np.int32, copy=False), z_coords.astype(np.int32, copy=False)


def _count_reuse_adjacency_conflicts(
    axial_q: np.ndarray,
    axial_r: np.ndarray,
    slot_ids: np.ndarray,
) -> int:
    neighbors = _build_reuse_neighbors_from_axial(axial_q, axial_r)
    return _count_reuse_adjacency_conflicts_from_neighbors(neighbors, slot_ids)


def _build_reuse_neighbors_from_axial(
    axial_q: np.ndarray,
    axial_r: np.ndarray,
) -> list[list[int]]:
    axial_q_use = np.asarray(axial_q, dtype=np.int32).reshape(-1)
    axial_r_use = np.asarray(axial_r, dtype=np.int32).reshape(-1)
    coord_to_index = {
        (int(q_coord), int(r_coord)): int(index)
        for index, (q_coord, r_coord) in enumerate(
            zip(axial_q_use.tolist(), axial_r_use.tolist())
        )
    }
    neighbors: list[list[int]] = [[] for _ in range(int(axial_q_use.size))]
    for index, (q_coord, r_coord) in enumerate(
        zip(axial_q_use.tolist(), axial_r_use.tolist())
    ):
        for dq, dr in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            neighbor_index = coord_to_index.get((int(q_coord + dq), int(r_coord + dr)))
            if neighbor_index is None:
                continue
            neighbors[int(index)].append(int(neighbor_index))
    return neighbors


def _build_reuse_neighbors_from_xy(
    x_km: np.ndarray,
    y_km: np.ndarray,
    *,
    point_spacing_km: float,
) -> list[list[int]]:
    """Find up to 6 nearest-neighbours within a hexagonal-spacing window.

    Uses ``scipy.spatial.cKDTree`` so the cost is O(N log N) instead of
    the O(N²) pairwise-distance matrix the naive implementation built.
    At ~500 k cells the pairwise path needed 3.57 TiB of RAM for the
    ``(N, N, 2)`` deltas tensor; the kd-tree path stays within a few
    hundred MB.  Falls back to the original O(N²) path for tiny grids
    (where the kd-tree overhead dominates) and when SciPy is
    unavailable.
    """
    point_count = int(np.asarray(x_km).size)
    if point_count != int(np.asarray(y_km).size):
        raise ValueError("x_km and y_km must share a common one-dimensional axis.")
    if point_count < 2:
        return [[] for _ in range(point_count)]
    coords = np.column_stack(
        [
            np.asarray(x_km, dtype=np.float64).reshape(-1),
            np.asarray(y_km, dtype=np.float64).reshape(-1),
        ]
    )
    spacing_value = max(1.0e-6, float(point_spacing_km))
    min_neighbor_distance = 0.45 * spacing_value
    max_neighbor_distance = 1.28 * spacing_value
    neighbors: list[list[int]] = [[] for _ in range(point_count)]

    # For small grids the naive path is actually faster (kd-tree build
    # amortises slowly) and the pairwise matrix only uses ~80 MB at
    # 1000 cells.  Above the threshold the pairwise matrix would
    # exceed host RAM, so switch to the kd-tree path.
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]
        _have_kdtree = True
    except Exception:
        _have_kdtree = False

    if _have_kdtree and point_count > 1500:
        tree = cKDTree(coords)
        # ``query_ball_tree`` returns, for each point, all indices within
        # ``r``.  We filter by the min distance afterwards.
        radius_lists = tree.query_ball_tree(tree, r=max_neighbor_distance)
        for index in range(point_count):
            candidate_ids = np.asarray(radius_lists[index], dtype=np.int64)
            if candidate_ids.size == 0:
                continue
            cand_coords = coords[candidate_ids]
            dists = np.sqrt(np.sum((cand_coords - coords[index]) ** 2, axis=1))
            keep_mask = (dists >= min_neighbor_distance) & (dists <= max_neighbor_distance)
            kept_ids = candidate_ids[keep_mask]
            if kept_ids.size > 6:
                kept_dists = dists[keep_mask]
                order = np.argsort(kept_dists, kind="stable")
                kept_ids = kept_ids[order[:6]]
            neighbors[int(index)] = [int(value) for value in kept_ids.tolist()]
        return neighbors

    # Naive O(N²) path for small grids or when SciPy is unavailable.
    deltas = coords[:, None, :] - coords[None, :, :]
    distances_km = np.sqrt(np.sum(deltas**2, axis=2))
    for index in range(point_count):
        neighbor_ids = np.nonzero(
            (distances_km[index] >= min_neighbor_distance)
            & (distances_km[index] <= max_neighbor_distance)
        )[0]
        if neighbor_ids.size > 6:
            order = np.argsort(distances_km[index, neighbor_ids], kind="stable")
            neighbor_ids = neighbor_ids[order[:6]]
        neighbors[int(index)] = [int(value) for value in neighbor_ids.tolist()]
    return neighbors


def _build_reuse_second_ring_neighbors(
    neighbors: list[list[int]],
) -> list[list[int]]:
    second_ring_neighbors: list[list[int]] = [[] for _ in range(len(neighbors))]
    for index, neighbor_ids in enumerate(neighbors):
        ring_ids: set[int] = set()
        direct_neighbor_ids = set(int(value) for value in neighbor_ids)
        for neighbor in neighbor_ids:
            ring_ids.update(int(value) for value in neighbors[int(neighbor)])
        ring_ids.discard(int(index))
        ring_ids.difference_update(direct_neighbor_ids)
        second_ring_neighbors[int(index)] = sorted(ring_ids)
    return second_ring_neighbors


def _count_reuse_adjacency_conflicts_from_neighbors(
    neighbors: list[list[int]],
    slot_ids: np.ndarray,
) -> int:
    slot_ids_use = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
    conflict_count = 0
    for index, neighbor_ids in enumerate(neighbors):
        for neighbor_index in neighbor_ids:
            if int(neighbor_index) <= int(index):
                continue
            if int(slot_ids_use[index]) == int(slot_ids_use[neighbor_index]):
                conflict_count += 1
    return int(conflict_count)


def _repair_reuse_adjacency_conflicts(
    axial_q: np.ndarray,
    axial_r: np.ndarray,
    slot_ids: np.ndarray,
    *,
    reuse_factor: int,
    anchor_index: int | None,
    anchor_slot: int,
) -> np.ndarray:
    neighbors = _build_reuse_neighbors_from_axial(axial_q, axial_r)
    second_ring_neighbors = _build_reuse_second_ring_neighbors(neighbors)
    return _repair_reuse_adjacency_conflicts_from_neighbors(
        neighbors,
        second_ring_neighbors,
        slot_ids,
        reuse_factor=reuse_factor,
        anchor_index=anchor_index,
        anchor_slot=anchor_slot,
    )


def _repair_reuse_adjacency_conflicts_from_neighbors(
    neighbors: list[list[int]],
    second_ring_neighbors: list[list[int]],
    slot_ids: np.ndarray,
    *,
    reuse_factor: int,
    anchor_index: int | None,
    anchor_slot: int,
) -> np.ndarray:
    slot_ids_out = np.asarray(slot_ids, dtype=np.int32).copy()
    base_slot_ids = slot_ids_out.copy()
    reuse_factor_i = max(1, int(reuse_factor))
    if reuse_factor_i <= 1 or int(slot_ids_out.size) < 2:
        return slot_ids_out
    anchor_index_i = None
    if anchor_index is not None and 0 <= int(anchor_index) < int(slot_ids_out.size):
        anchor_index_i = int(anchor_index)
        slot_ids_out[anchor_index_i] = np.int32(int(anchor_slot) % reuse_factor_i)
        base_slot_ids[anchor_index_i] = np.int32(int(anchor_slot) % reuse_factor_i)
    for _pass in range(96):
        changed = False
        conflict_counts = [
            sum(1 for neighbor in neighbor_ids if int(slot_ids_out[index]) == int(slot_ids_out[neighbor]))
            for index, neighbor_ids in enumerate(neighbors)
        ]
        conflict_indices = [int(index) for index, value in enumerate(conflict_counts) if int(value) > 0]
        if not conflict_indices:
            break
        conflict_indices.sort(
            key=lambda value: (
                value == anchor_index_i,
                -int(conflict_counts[int(value)]),
                -len(neighbors[int(value)]),
                value,
            )
        )
        for index in conflict_indices:
            if anchor_index_i is not None and int(index) == int(anchor_index_i):
                continue
            neighbor_slots = {int(slot_ids_out[neighbor]) for neighbor in neighbors[int(index)]}
            current_slot = int(slot_ids_out[int(index)])
            if current_slot not in neighbor_slots:
                continue
            best_slot = current_slot
            best_score: tuple[int, int, int, int] | None = None
            for candidate_slot in range(reuse_factor_i):
                local_conflicts = sum(
                    1 for neighbor in neighbors[int(index)] if int(slot_ids_out[neighbor]) == int(candidate_slot)
                )
                second_ring_conflicts = sum(
                    1
                    for neighbor in second_ring_neighbors[int(index)]
                    if int(slot_ids_out[neighbor]) == int(candidate_slot)
                )
                score = (
                    int(local_conflicts),
                    int(second_ring_conflicts),
                    0 if int(candidate_slot) == int(base_slot_ids[int(index)]) else 1,
                    0 if int(candidate_slot) == int(current_slot) else 1,
                    int(candidate_slot),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_slot = int(candidate_slot)
                    if score[0] == 0 and score[1] == 0 and score[2] == 0 and score[3] == 0:
                        break
            if int(best_slot) != int(current_slot):
                slot_ids_out[int(index)] = np.int32(int(best_slot))
                changed = True
        if not changed:
            break
    if anchor_index_i is not None:
        slot_ids_out[anchor_index_i] = np.int32(int(anchor_slot) % reuse_factor_i)
    return slot_ids_out


def resolve_frequency_reuse_slots(
    prepared_grid: Mapping[str, Any],
    *,
    reuse_factor: int,
    anchor_slot: int = 0,
) -> dict[str, Any]:
    """
    Resolve deterministic reuse-slot ids for a prepared Earth hexgrid.

    Parameters
    ----------
    prepared_grid : Mapping[str, Any]
        Prepared-grid payload returned by :func:`prepare_active_grid`.
    reuse_factor : int
        Supported hex-cluster reuse factor. Valid values are
        ``{1, 3, 4, 7, 9, 12, 13, 16, 19}``.
    anchor_slot : int, optional
        Slot assigned to the RAS anchor cell. Values wrap modulo
        ``reuse_factor``.

    Returns
    -------
    dict[str, Any]
        Mapping containing slot ids on both the PRE_RAS and ACTIVE axes,
        anchor indices, inferred axial coordinates, and cluster-representative
        metadata suitable for schematic reuse plots.

    Notes
    -----
    The helper infers local axial coordinates on the prepared PRE_RAS hexgrid
    around the RAS anchor cell and then maps those coordinates onto the
    canonical hex reuse lattice for the selected cluster size. When the RAS
    anchor cell is later excluded from service, it still anchors the tiling on
    the PRE_RAS axis and the ACTIVE projection simply omits that cell.
    """
    reuse_factor_i = int(reuse_factor)
    shift_pair = _resolve_hexgrid_reuse_shift_pair(reuse_factor_i)
    pre_lons_q = u.Quantity(prepared_grid["pre_ras_cell_longitudes"], copy=False)
    pre_lats_q = u.Quantity(prepared_grid["pre_ras_cell_latitudes"], copy=False)
    pre_lon_deg = np.asarray(pre_lons_q.to_value(u.deg), dtype=np.float64).reshape(-1)
    pre_lat_deg = np.asarray(pre_lats_q.to_value(u.deg), dtype=np.float64).reshape(-1)
    if pre_lon_deg.size != pre_lat_deg.size:
        raise ValueError("Prepared-grid PRE_RAS longitude/latitude arrays must have the same size.")
    if pre_lon_deg.size < 1:
        raise ValueError("Prepared grid must contain at least one PRE_RAS cell.")

    anchor_pre_ras_index = int(prepared_grid.get("ras_service_cell_index_pre_ras", -1))
    if anchor_pre_ras_index < 0 or anchor_pre_ras_index >= int(pre_lon_deg.size):
        station_lon_deg = float(u.Quantity(prepared_grid["station_lon"]).to_value(u.deg))
        station_lat_deg = float(u.Quantity(prepared_grid["station_lat"]).to_value(u.deg))
        distances_km = _great_circle_distance_km(
            pre_lon_deg * u.deg,
            pre_lat_deg * u.deg,
            station_lat=station_lat_deg * u.deg,
            station_lon=station_lon_deg * u.deg,
        ).to_value(u.km)
        anchor_pre_ras_index = int(np.argmin(np.asarray(distances_km, dtype=np.float64)))

    anchor_lon_deg = float(pre_lon_deg[anchor_pre_ras_index])
    anchor_lat_deg = float(pre_lat_deg[anchor_pre_ras_index])
    point_spacing_km = prepared_grid.get("actual_point_spacing_km") or prepared_grid.get("point_spacing_km")
    if point_spacing_km is None:
        point_spacing_value = _estimate_local_hexgrid_spacing_km(
            pre_lon_deg,
            pre_lat_deg,
            ref_lon_deg=anchor_lon_deg,
            ref_lat_deg=anchor_lat_deg,
        )
    else:
        point_spacing_value = float(point_spacing_km)
    x_km, y_km = _local_tangent_plane_xy_km(
        pre_lon_deg,
        pre_lat_deg,
        ref_lon_deg=anchor_lon_deg,
        ref_lat_deg=anchor_lat_deg,
    )
    axial_q_pre_ras, axial_r_pre_ras, orientation_name, fit_residual_km2 = (
        _infer_hexgrid_axial_coordinates(
            x_km,
            y_km,
            point_spacing_km=float(point_spacing_value),
        )
    )

    anchor_slot_i = int(anchor_slot) % reuse_factor_i
    coset_to_slot, cluster_representatives = _enumerate_reuse_cluster_slots(
        reuse_factor=reuse_factor_i,
        shift_pair=shift_pair,
        anchor_slot=anchor_slot_i,
    )

    if len(coset_to_slot) != reuse_factor_i:
        raise RuntimeError(
            "Failed to enumerate the full reuse cluster; "
            f"expected {reuse_factor_i} slots, got {len(coset_to_slot)}."
        )

    pre_ras_to_active = np.asarray(prepared_grid.get("pre_ras_to_active", np.empty(0, dtype=np.int32)), dtype=np.int32)
    valid_pre_ras_indices = np.nonzero(pre_ras_to_active >= 0)[0]
    active_count_hint = 0
    active_neighbors: list[list[int]] | None = None
    active_second_ring_neighbors: list[list[int]] | None = None
    if valid_pre_ras_indices.size:
        active_grid_longitudes = prepared_grid.get("active_grid_longitudes")
        if active_grid_longitudes is not None:
            active_count_hint = int(u.Quantity(active_grid_longitudes, copy=False).size)
        if active_count_hint <= 0:
            active_count_hint = int(np.max(pre_ras_to_active[valid_pre_ras_indices])) + 1
        active_grid_latitudes = prepared_grid.get("active_grid_latitudes")
        if active_grid_longitudes is not None and active_grid_latitudes is not None and active_count_hint > 0:
            active_lon_deg = np.asarray(
                u.Quantity(active_grid_longitudes, copy=False).to_value(u.deg),
                dtype=np.float64,
            ).reshape(-1)
            active_lat_deg = np.asarray(
                u.Quantity(active_grid_latitudes, copy=False).to_value(u.deg),
                dtype=np.float64,
            ).reshape(-1)
            if active_lon_deg.size == active_lat_deg.size == active_count_hint:
                active_x_km, active_y_km = _local_tangent_plane_xy_km(
                    active_lon_deg,
                    active_lat_deg,
                    ref_lon_deg=anchor_lon_deg,
                    ref_lat_deg=anchor_lat_deg,
                )
                active_neighbors = _build_reuse_neighbors_from_xy(
                    active_x_km,
                    active_y_km,
                    point_spacing_km=float(point_spacing_value),
                )
                active_second_ring_neighbors = _build_reuse_second_ring_neighbors(active_neighbors)

    best_candidate: tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]], bool, int] | None = None
    best_score: tuple[int, int, int] | None = None
    for reflected in (False, True):
        for rotation_steps in range(6):
            axial_q_candidate, axial_r_candidate = _transform_axial_symmetry(
                axial_q_pre_ras,
                axial_r_pre_ras,
                rotation_steps=rotation_steps,
                reflected=reflected,
            )
            pre_ras_slot_ids_candidate = np.empty(pre_lon_deg.shape, dtype=np.int32)
            for idx, (q_coord, r_coord) in enumerate(
                zip(axial_q_candidate.tolist(), axial_r_candidate.tolist())
            ):
                key = _reuse_coset_key(
                    q_coord,
                    r_coord,
                    reuse_factor=reuse_factor_i,
                    shift_pair=shift_pair,
                )
                pre_ras_slot_ids_candidate[idx] = np.int32(
                    (coset_to_slot[key] + anchor_slot_i) % reuse_factor_i
                )
            pre_ras_conflict_count = _count_reuse_adjacency_conflicts(
                axial_q_candidate,
                axial_r_candidate,
                pre_ras_slot_ids_candidate,
            )
            active_conflict_count = int(pre_ras_conflict_count)
            if valid_pre_ras_indices.size and active_count_hint > 0:
                active_slot_ids_candidate = np.full(active_count_hint, -1, dtype=np.int32)
                active_axial_q_candidate = np.full(active_count_hint, 0, dtype=np.int32)
                active_axial_r_candidate = np.full(active_count_hint, 0, dtype=np.int32)
                for pre_ras_index in valid_pre_ras_indices.tolist():
                    active_index = int(pre_ras_to_active[pre_ras_index])
                    active_slot_ids_candidate[active_index] = np.int32(pre_ras_slot_ids_candidate[pre_ras_index])
                    active_axial_q_candidate[active_index] = np.int32(axial_q_candidate[pre_ras_index])
                    active_axial_r_candidate[active_index] = np.int32(axial_r_candidate[pre_ras_index])
                valid_active_mask = active_slot_ids_candidate >= 0
                if active_neighbors is not None and np.all(valid_active_mask):
                    active_conflict_count = int(
                        _count_reuse_adjacency_conflicts_from_neighbors(
                            active_neighbors,
                            active_slot_ids_candidate,
                        )
                    )
                else:
                    active_conflict_count = int(
                        _count_reuse_adjacency_conflicts(
                            active_axial_q_candidate[valid_active_mask],
                            active_axial_r_candidate[valid_active_mask],
                            active_slot_ids_candidate[valid_active_mask],
                        )
                    )
            score = (
                int(active_conflict_count),
                int(pre_ras_conflict_count),
                int(rotation_steps + (6 if reflected else 0)),
            )
            if best_score is not None and score >= best_score:
                continue
            transformed_representatives: list[dict[str, int]] = []
            for representative in cluster_representatives:
                rep_q, rep_r = _transform_axial_symmetry(
                    np.asarray([representative["axial_q"]], dtype=np.int32),
                    np.asarray([representative["axial_r"]], dtype=np.int32),
                    rotation_steps=rotation_steps,
                    reflected=reflected,
                )
                transformed_representatives.append(
                    {
                        **representative,
                        "axial_q": int(rep_q[0]),
                        "axial_r": int(rep_r[0]),
                    }
                )
            best_score = score
            best_candidate = (
                np.asarray(axial_q_candidate, dtype=np.int32),
                np.asarray(axial_r_candidate, dtype=np.int32),
                np.asarray(pre_ras_slot_ids_candidate, dtype=np.int32),
                transformed_representatives,
                bool(reflected),
                int(rotation_steps),
            )
            if active_conflict_count == 0 and pre_ras_conflict_count == 0 and not reflected and rotation_steps == 0:
                break
        if best_score == (0, 0, 0):
            break

    if best_candidate is None:
        raise RuntimeError("Failed to resolve a reuse-slot assignment on the inferred hexgrid.")

    axial_q_pre_ras, axial_r_pre_ras, pre_ras_slot_ids, cluster_representatives, reflected_solution, rotation_steps = best_candidate
    pre_ras_slot_ids = _repair_reuse_adjacency_conflicts(
        axial_q_pre_ras,
        axial_r_pre_ras,
        pre_ras_slot_ids,
        reuse_factor=reuse_factor_i,
        anchor_index=anchor_pre_ras_index,
        anchor_slot=anchor_slot_i,
    )

    pre_ras_conflict_count = int(
        _count_reuse_adjacency_conflicts(axial_q_pre_ras, axial_r_pre_ras, pre_ras_slot_ids)
    )
    active_slot_ids = np.empty((0,), dtype=np.int32)
    active_axial_q = np.empty((0,), dtype=np.int32)
    active_axial_r = np.empty((0,), dtype=np.int32)
    active_conflict_count = pre_ras_conflict_count
    anchor_active_index = int(prepared_grid.get("ras_service_cell_index", -1))
    if pre_ras_to_active.size:
        if valid_pre_ras_indices.size:
            active_count = int(active_count_hint)
            active_slot_ids = np.full(active_count, -1, dtype=np.int32)
            active_axial_q = np.full(active_count, 0, dtype=np.int32)
            active_axial_r = np.full(active_count, 0, dtype=np.int32)
            for pre_ras_index in valid_pre_ras_indices.tolist():
                active_index = int(pre_ras_to_active[pre_ras_index])
                active_slot_ids[active_index] = np.int32(pre_ras_slot_ids[pre_ras_index])
                active_axial_q[active_index] = np.int32(axial_q_pre_ras[pre_ras_index])
                active_axial_r[active_index] = np.int32(axial_r_pre_ras[pre_ras_index])

            valid_active_mask = active_slot_ids >= 0
            compact_active_indices = np.nonzero(valid_active_mask)[0]
            compact_anchor_index = (
                None
                if anchor_active_index < 0 or anchor_active_index >= active_count or (not valid_active_mask[anchor_active_index])
                else int(np.nonzero(compact_active_indices == anchor_active_index)[0][0])
            )
            if active_neighbors is not None and active_second_ring_neighbors is not None and np.all(valid_active_mask):
                repaired_compact_slot_ids = _repair_reuse_adjacency_conflicts_from_neighbors(
                    active_neighbors,
                    active_second_ring_neighbors,
                    active_slot_ids,
                    reuse_factor=reuse_factor_i,
                    anchor_index=anchor_active_index,
                    anchor_slot=anchor_slot_i,
                )
            else:
                repaired_compact_slot_ids = _repair_reuse_adjacency_conflicts(
                    active_axial_q[valid_active_mask],
                    active_axial_r[valid_active_mask],
                    active_slot_ids[valid_active_mask],
                    reuse_factor=reuse_factor_i,
                    anchor_index=compact_anchor_index,
                    anchor_slot=anchor_slot_i,
                )
            active_slot_ids[valid_active_mask] = repaired_compact_slot_ids
            if active_neighbors is not None and np.all(valid_active_mask):
                active_conflict_count = int(
                    _count_reuse_adjacency_conflicts_from_neighbors(active_neighbors, active_slot_ids)
                )
            else:
                active_conflict_count = int(
                    _count_reuse_adjacency_conflicts(
                        active_axial_q[valid_active_mask],
                        active_axial_r[valid_active_mask],
                        active_slot_ids[valid_active_mask],
                    )
                )
            for pre_ras_index in valid_pre_ras_indices.tolist():
                active_index = int(pre_ras_to_active[pre_ras_index])
                pre_ras_slot_ids[pre_ras_index] = np.int32(active_slot_ids[active_index])
    return {
        "reuse_factor": int(reuse_factor_i),
        "anchor_slot": int(anchor_slot_i),
        "anchor_pre_ras_index": int(anchor_pre_ras_index),
        "anchor_active_index": int(anchor_active_index),
        "point_spacing_km_used": float(point_spacing_value),
        "orientation_used": str(orientation_name),
        "orientation_reflected": bool(reflected_solution),
        "orientation_rotation_steps": int(rotation_steps),
        "fit_residual_km2": float(fit_residual_km2),
        "adjacent_same_slot_pair_count": int(active_conflict_count),
        "pre_ras_adjacent_same_slot_pair_count": int(pre_ras_conflict_count),
        "active_adjacent_same_slot_pair_count": int(active_conflict_count),
        "axial_q_pre_ras": np.asarray(axial_q_pre_ras, dtype=np.int32),
        "axial_r_pre_ras": np.asarray(axial_r_pre_ras, dtype=np.int32),
        "axial_q_active": np.asarray(active_axial_q, dtype=np.int32),
        "axial_r_active": np.asarray(active_axial_r, dtype=np.int32),
        "pre_ras_slot_ids": np.asarray(pre_ras_slot_ids, dtype=np.int32),
        "active_slot_ids": np.asarray(active_slot_ids, dtype=np.int32),
        "cluster_representatives": cluster_representatives,
    }


def _require_shapely(*, purpose: str) -> tuple[object, object, object]:
    try:
        import shapely
        from shapely import geometry as shapely_geometry
        from shapely import ops as shapely_ops
    except ImportError as exc:  # pragma: no cover - exercised only when shapely is absent
        raise RuntimeError(
            f"shapely is required for {purpose}. Install it with "
            "'conda install -n scepter-dev -c conda-forge shapely'."
        ) from exc
    return shapely, shapely_geometry, shapely_ops


@lru_cache(maxsize=8)
def _load_natural_earth_geometries(kind: str, backend: str = "vendored") -> tuple[object, ...]:
    """
    Load Natural Earth geometries as Shapely objects.

    Parameters
    ----------
    kind : {"land", "coastline"}
        Dataset to load.
    backend : {"vendored", "cartopy"}, optional
        Geometry source. ``"vendored"`` reads the repo-shipped GeoJSON files.
        ``"cartopy"`` loads the Natural Earth shapefiles through Cartopy.

    Returns
    -------
    tuple of object
        Tuple of Shapely geometries in lon/lat degrees.

    Raises
    ------
    RuntimeError
        Raised when the required optional dependency is unavailable or when the
        requested dataset cannot be loaded.
    ValueError
        Raised when ``kind`` or ``backend`` is unsupported.
    """
    if kind not in _NATURAL_EARTH_RESOURCE_NAMES:
        raise ValueError(
            f"kind must be one of {sorted(_NATURAL_EARTH_RESOURCE_NAMES)!r}; got {kind!r}."
        )
    backend_name = _normalise_coastline_backend(backend)
    _, shapely_geometry, _ = _require_shapely(
        purpose="Natural Earth land/coast geometry loading",
    )

    geometries: list[object] = []
    if backend_name == "vendored":
        resource_path = files("scepter.data").joinpath(_NATURAL_EARTH_RESOURCE_NAMES[kind])
        with resource_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for feature in payload.get("features", []):
            geometry_payload = feature.get("geometry")
            if geometry_payload is None:
                continue
            geometry_obj = shapely_geometry.shape(geometry_payload)
            if not geometry_obj.is_empty:
                geometries.append(geometry_obj)
    else:
        try:
            from cartopy.io import shapereader
        except ImportError as exc:  # pragma: no cover - exercised only when cartopy is absent
            raise RuntimeError(
                "cartopy is required for coastline_backend='cartopy'. Install it with "
                "'conda install -n scepter-dev-full -c conda-forge shapely cartopy'."
            ) from exc

        shapefile_path = shapereader.natural_earth(
            resolution="10m",
            category="physical",
            name="land" if kind == "land" else "coastline",
        )
        for geometry_obj in shapereader.Reader(shapefile_path).geometries():
            if not geometry_obj.is_empty:
                geometries.append(geometry_obj)

    if not geometries:
        raise RuntimeError(
            f"Failed to load any Natural Earth geometries for kind={kind!r} "
            f"and backend={backend_name!r}."
        )
    return tuple(geometries)


def _spherical_central_angle_scalar_impl(R_eff_m: float, R_sat_m: float, alpha_rad: float) -> float:
    """
    Compute central angle gamma = ∠SOE for a ray at angle alpha from nadir.

    Parameters
    ----------
    R_eff_m : float
        Effective Earth radius [m] (spherical model).
    R_sat_m : float
        Satellite radius from Earth center [m] (R_eff_m + altitude).
    alpha_rad : float
        Ray angle from nadir direction at satellite [rad]. Must be >= 0.

    Returns
    -------
    float
        Central angle gamma [rad], or np.nan if the ray misses Earth.
    """
    OS_sq = R_sat_m * R_sat_m
    OE_sq = R_eff_m * R_eff_m
    two_OS_OE = 2.0 * R_sat_m * R_eff_m

    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)

    term_under_sqrt = OE_sq - OS_sq * (sin_alpha * sin_alpha)
    if term_under_sqrt < -1e-11 * OE_sq:
        return np.nan
    if term_under_sqrt < 0.0:
        term_under_sqrt = 0.0

    sqrt_term = np.sqrt(term_under_sqrt)

    slant_range_SE = R_sat_m * cos_alpha - sqrt_term
    if slant_range_SE < 0.0:
        return np.nan

    SE_sq = slant_range_SE * slant_range_SE
    cos_SOE = (OS_sq + OE_sq - SE_sq) / two_OS_OE

    # Manual clamp for numba scalar compatibility (np.clip fails in njit for scalars)
    if cos_SOE < -1.0:
        cos_SOE = -1.0
    elif cos_SOE > 1.0:
        cos_SOE = 1.0

    return float(np.arccos(cos_SOE))



if HAS_NUMBA:
    _spherical_central_angle_scalar = njit(cache=True)(_spherical_central_angle_scalar_impl)
else:
    _spherical_central_angle_scalar = lru_cache(maxsize=32)(_spherical_central_angle_scalar_impl)


def _compute_impact_mask_impl(
    grid_lons_rad: np.ndarray,
    grid_lats_rad: np.ndarray,
    station_lat_rad: float,
    station_lon_rad: float,
    cos_delta_max: float,
) -> np.ndarray:
    """
    Fast impactful mask using cosine-threshold comparisons (no arccos).

    A cell is impactful if:
        δ <= Δ_max   and   δ <= π/2
    which is equivalent to:
        cos(δ) >= cos(Δ_max)  and  cos(δ) >= 0
    """
    sin_station_lat = np.sin(station_lat_rad)
    cos_station_lat = np.cos(station_lat_rad)

    sin_grid_lat = np.sin(grid_lats_rad)
    cos_grid_lat = np.cos(grid_lats_rad)

    delta_lon = grid_lons_rad - station_lon_rad

    # cos(δ) via spherical law of cosines
    cos_delta = (
        sin_station_lat * sin_grid_lat
        + cos_station_lat * cos_grid_lat * np.cos(delta_lon)
    )

    # Numerical safety
    cos_delta = np.clip(cos_delta, -1.0, 1.0)

    # δ <= π/2  <=> cos(δ) >= 0
    return (cos_delta >= cos_delta_max) & (cos_delta >= 0.0)


if HAS_NUMBA:
    _compute_impact_mask = njit(cache=True)(_compute_impact_mask_impl)
else:
    _compute_impact_mask = _compute_impact_mask_impl



# -----------------------------------------------------------------------------
# CALCULATE FOOTPRINT SIZE
# -----------------------------------------------------------------------------
@ranged_quantity_input(
    altitude=(0.001, None, u.km),
    off_nadir_angle=(None, None, u.deg),
    strip_input_units=False,
    allow_none=True,
)
def calculate_footprint_size(
    antenna_gain_func: callable,
    altitude: u.Quantity,
    off_nadir_angle: u.Quantity = 0 * u.deg,
    earth_model: str = "spherical",
    theta: None | u.Quantity = None,
    level_drop: float | u.Quantity = 3.0 * cnv.dB,
    **antenna_pattern_kwargs,
) -> u.Quantity:
    """
    Compute the scan-plane footprint "diameter" (arc length) on Earth's surface
    for a given antenna contour (default: -3 dB).

    CONVENTION
    ----------
    - `theta` is the EDGE HALF-ANGLE to the contour (angular radius from boresight).
    - If `theta is None`, beamwidth is computed via `calculate_beamwidth_1d`, which
      returns FULL beamwidth, and is converted internally to half-angle.

    GEOMETRY (spherical Earth)
    --------------------------
    - beta: boresight off-nadir angle at satellite (signed, in the scan plane).
    - alpha_limb: maximum |alpha| that intersects Earth:
        alpha_limb = asin(R / (R + h))
    - scan-plane contour interval in alpha:
        [beta - theta_edge, beta + theta_edge]
      intersected with Earth-visible interval [-alpha_limb, +alpha_limb].
    - Endpoints are mapped to signed central coordinates:
        s(alpha) = sign(alpha) * gamma(|alpha|)
      where gamma(|alpha|) = ∠SOE central angle for a ray at |alpha|.
    - Footprint diameter (scan-plane arc length):
        L = R * |s_right - s_left|
    """
    if earth_model != "spherical":
        raise ValueError("Only 'spherical' earth_model is currently supported.")
    if not callable(antenna_gain_func):
        raise TypeError("`antenna_gain_func` must be callable.")

    # ---- Convert inputs ONCE to floats (fast, numba-friendly downstream) ----
    R_eff_m = float(R_earth.to_value(u.m))
    h_m = float(altitude.to_value(u.m))
    R_sat_m = R_eff_m + h_m

    beta = float(off_nadir_angle.to_value(u.rad))  # signed ok

    # theta_edge: half-angle in radians (float)
    if theta is None:
        bw_full = calculate_beamwidth_1d(
            antenna_gain_func, level_drop=level_drop, **antenna_pattern_kwargs
        )
        theta_edge = 0.5 * float(bw_full.to_value(u.rad))
    else:
        theta_edge = float(theta.to_value(u.rad))

    if theta_edge < 0.0:
        theta_edge = abs(theta_edge)

    # ---- Limb angle (alpha_limb) ----
    sin_arg = R_eff_m / R_sat_m
    if sin_arg < -1.0:
        sin_arg = -1.0
    elif sin_arg > 1.0:
        sin_arg = 1.0
    alpha_limb = float(np.arcsin(sin_arg))  # [rad]

    # ---- Scan-plane contour interval ----
    a1 = beta - theta_edge
    a2 = beta + theta_edge
    if a1 > a2:
        a1, a2 = a2, a1

    # ---- Intersect with Earth-visible alpha interval ----
    left = max(a1, -alpha_limb)
    right = min(a2, +alpha_limb)

    if left > right:
        warnings.warn("Contour interval does not intersect Earth-visible angles.", UserWarning)
        return np.nan * u.m

    clipped = (left != a1) or (right != a2)
    if clipped:
        warnings.warn("Footprint is truncated by Earth limb in scan plane.", UserWarning)

    # ---- Map alpha endpoints -> signed central coordinate s(alpha) ----
    # gamma(|alpha|) computed via the (optionally numba-jitted) scalar helper.
    def signed_s(alpha_signed: float) -> float:
        sgn = -1.0 if alpha_signed < 0.0 else (1.0 if alpha_signed > 0.0 else 0.0)
        gam = _spherical_central_angle_scalar(R_eff_m, R_sat_m, abs(alpha_signed))
        if np.isnan(gam):
            return np.nan
        return sgn * gam

    s_left = signed_s(left)
    s_right = signed_s(right)

    if np.isnan(s_left) or np.isnan(s_right):
        warnings.warn("Internal error: failed to compute central angle at endpoints.", RuntimeWarning)
        return np.nan * u.m

    total_central_angle = abs(s_right - s_left)  # [rad]
    footprint_diameter_m = R_eff_m * total_central_angle

    return footprint_diameter_m * u.m


# -----------------------------------------------------------------------------
# MAIN FUNCTIONS: GENERATE FULL HEXAGON GRID
# -----------------------------------------------------------------------------
@ranged_quantity_input(
    point_spacing=(1, None, u.km),
    strip_input_units=False,
    allow_none=True
)
def generate_hexgrid_full(point_spacing):
    """
    Generates a full hexagon grid covering Earth based on the specified point 
    spacing (in meters). A spherical triangle is defined and then replicated and 
    rotated to cover the entire globe.
    
    The function uses pycraf's geometry and pathprof modules for spherical 
    coordinate conversion and geoid calculations.
    
    Parameters
    ----------
    point_spacing : float
        Desired spacing (in meters) between grid cell centers.
    
    Returns
    -------
    grid_longitudes : np.ndarray
        Numpy array of longitudes (in degrees) for the hexagon grid points.
    grid_latitudes : np.ndarray
        Numpy array of latitudes (in degrees) for the hexagon grid points.
    grid_spacing : list
        List of grid spacings (in meters) used in the generation process.
    """
    
    # -----------------------------------------------------------------------------
    # HELPER FUNCTIONS: FILL TRIANGLE FUNCTION FOR HEXAGON GRID GENERATION
    # -----------------------------------------------------------------------------
    def _fill_triangle(tri_corners, numpoints_start):
        """
        Computes arrays of longitudes and latitudes for cell centers filling a 
        spherical triangle defined by three Cartesian corner points, and returns 
        the grid spacing used along each row.
        
        Parameters
        ----------
        tri_corners : tuple of array-like
            Tuple (tri_x, tri_y, tri_z) with Cartesian coordinates (in meters) 
            of the triangle corners.
        numpoints_start : int
            Initial number of points along the triangle edge; reduced progressively 
            along the triangle to fill the area.
            
        Returns
        -------
        plons : list of np.ndarray
            List of arrays of longitudes (in degrees) for the grid cell centers.
        plats : list of np.ndarray
            List of arrays of latitudes (in degrees) for the grid cell centers.
        grid_spacing : list
            List of grid spacings (in meters) used for each row in the filled triangle.
        """
        def _process_segment(s_lon, s_lat, e_lon, e_lat, numpoints):
            if np.abs(s_lat) < 1e-5 and np.abs(e_lat) < 1e-5:
                s_lat = e_lat = 1e-5
            d, b, _ = pathprof.geoid_inverse(s_lon * u.deg, s_lat * u.deg,
                                            e_lon * u.deg, e_lat * u.deg)
            spacing = d.to_value(u.m) / (numpoints - 1)
            dvec_seg = np.linspace(0, d.value, numpoints) * u.m
            plon_seg, plat_seg, _ = pathprof.geoid_direct(s_lon * u.deg, s_lat * u.deg, b, dvec_seg[1:])
            return (np.concatenate(([s_lon], plon_seg.to_value(u.deg))),
                    np.concatenate(([s_lat], plat_seg.to_value(u.deg))),
                    spacing)
        tri_x, tri_y, tri_z = tri_corners
        _, tri_phi, tri_theta = geometry.cart_to_sphere(tri_x * u.m, tri_y * u.m, tri_z * u.m)
        
        d1, b1, _ = pathprof.geoid_inverse(tri_phi[1], tri_theta[1],
                                            tri_phi[0], tri_theta[0])
        d3, b3, _ = pathprof.geoid_inverse(tri_phi[2], tri_theta[2],
                                            tri_phi[0], tri_theta[0])
        
        dvec = np.linspace(0, d1.value, numpoints_start) * u.m
        plon1, plat1, _ = pathprof.geoid_direct(tri_phi[1], tri_theta[1], b1, dvec[1:])
        plon3, plat3, _ = pathprof.geoid_direct(tri_phi[2], tri_theta[2], b3, dvec[1:])
        
        plon1 = np.concatenate(([tri_phi[1].to_value(u.deg)], plon1.to_value(u.deg)))
        plat1 = np.concatenate(([tri_theta[1].to_value(u.deg)], plat1.to_value(u.deg)))
        plon3 = np.concatenate(([tri_phi[2].to_value(u.deg)], plon3.to_value(u.deg)))
        plat3 = np.concatenate(([tri_theta[2].to_value(u.deg)], plat3.to_value(u.deg)))
        
        
        
        seg_results = [_process_segment(plon1[idx], plat1[idx], plon3[idx], plat3[idx],
                                        numpoints_start - idx)
                    for idx in range(len(plon1) - 1)]
        plons = [res[0] for res in seg_results]
        plats = [res[1] for res in seg_results]
        grid_spacing = [res[2] for res in seg_results]
        
        plons.append(np.array([tri_phi[0].to_value(u.deg)]))
        plats.append(np.array([tri_theta[0].to_value(u.deg)]))
        
        return plons, plats, grid_spacing
    phi = (np.degrees(
        [0] + [2 * k * np.pi / 5 for k in range(1, 6)] +
        [(2 * k - 1) * np.pi / 5 for k in range(1, 6)] + [0]
    ) + 180) % 360 - 180
    theta = 90. - np.degrees(
        [0] + [np.arctan(2)] * 5 + [np.pi - np.arctan(2)] * 5 + [np.pi]
    )
    x, y, z = geometry.sphere_to_cart(1 * u.m, phi * u.deg, theta * u.deg)
    x, y, z = x.value, y.value, z.value

    d, _, _ = pathprof.geoid_inverse(phi[1] * u.deg, theta[1] * u.deg,
                                     phi[7] * u.deg, theta[7] * u.deg)
    numpoints_start = int(1.125 * d / point_spacing + 0.5) + 1

    plons, plats, grid_spacing = [], [], []
    triangle_configs = [
        ([0, 5, 1], slice(1, -1), slice(0, -1)),
        ([6, 1, 5], slice(0, -1), slice(0, -1)),
        ([1, 6, 7], slice(1, -1), slice(1, None)),
        ([11, 6, 7], slice(0, -1), slice(0, -1)),
    ]
    offsets = np.arange(5) * 72  # Rotation offsets: 0°,72°,144°,216°,288°
    
    for itup, row_sl, col_sl in triangle_configs:
        tri_x, tri_y, tri_z = x[itup], y[itup], z[itup]
        _plons, _plats, _grid_spacing = _fill_triangle((tri_x, tri_y, tri_z), numpoints_start)
        _plons = [p[col_sl] for p in _plons][row_sl]
        _plats = [p[col_sl] for p in _plats][row_sl]
        for row in _plons:
            rotated = ((row[None, :] + offsets[:, None] + 180) % 360) - 180
            plons.append(rotated.flatten())
        for row in _plats:
            plats.extend([row] * 5)
        grid_spacing.append(_grid_spacing)
    
    grid_longitudes = np.concatenate([np.array([0]), np.hstack(plons), np.array([0])])
    grid_latitudes = np.concatenate([np.array([90]), np.hstack(plats), np.array([-90])])

    # --- Deduplicate cells that may share triangle/rotation seam positions ---
    # Round to ~11 m resolution (0.0001 deg) before checking for duplicates;
    # this is far below any realistic cell spacing but catches numerical
    # near-duplicates produced at icosahedral seams.
    _dedup_decimals = 4
    rounded_coords = np.round(
        np.column_stack([grid_longitudes, grid_latitudes]),
        decimals=_dedup_decimals,
    )
    _, unique_indices = np.unique(
        rounded_coords, axis=0, return_index=True,
    )
    unique_indices = np.sort(unique_indices)  # preserve original ordering
    grid_longitudes = grid_longitudes[unique_indices]
    grid_latitudes = grid_latitudes[unique_indices]

    return grid_longitudes * u.deg, grid_latitudes * u.deg, grid_spacing * u.m

@ranged_quantity_input(
    grid_longitudes=(None,None,u.deg),
    grid_latitudes=(None,None,u.deg),
    sat_altitude=(1, None, u.km),
    min_elevation=(0, 90, u.deg),
    station_lat=(None,None,u.deg),
    station_lon=(None,None,u.deg),
    strip_input_units=False,
    allow_none=True
)
def trunc_hexgrid_to_impactful(grid_longitudes, grid_latitudes, sat_altitude, min_elevation, 
                               station_lat, station_lon, station_height=None):
    """
    Truncates a full hexagon grid to only those cells that are potentially served by
    satellites positioned up to the horizon circle (i.e. barely visible) from a given
    radio astronomy station.
    
    For a station at (station_lat, station_lon) with a specified satellite altitude,
    the station's horizon angle is computed as:
    
        θₕ = arccos(R_earth / (R_earth + sat_altitude))
    
    The maximum allowed margin:
    
        γ = arccos((R_earth * cos(min_elevation))/(R_earth + sat_altitude)) - min_elevation
    
    A grid cell at angular separation δ (from the station) is considered impactful if:
    
        δ ≤ θₕ + γ    and    δ ≤ π/2,
    
    ensuring that only cells on the near side of Earth (δ ≤ π/2) and within the combined
    satellite footprint (θₕ + γ) are retained.
    
    Parameters
    ----------
    grid_longitudes : np.ndarray
        Numpy array of longitudes (in degrees) for the full hexagon grid cells.
    grid_latitudes : np.ndarray
        Numpy array of latitudes (in degrees) for the full hexagon grid cells.
    sat_altitude : astropy.units.Quantity
        Satellite altitude above Earth's surface.
    min_elevation : astropy.units.Quantity
        Minimum operational elevation (in degrees) required for service.
    station_lat : astropy.units.Quantity
        Latitude of the radio astronomy station.
    station_lon : astropy.units.Quantity
        Longitude of the radio astronomy station.
    station_height : astropy.units.Quantity
        Station elevation above Earth's surface (provided for completeness, not used at the moment).
        
    Returns
    -------
    mask : np.ndarray of bool
        Boolean mask that, when applied to grid_longitudes and grid_latitudes, retains only
        those grid cells that are impactful.
    """
    # Convert station and grid cell coordinates to radians (floating-point
    # values for compatibility with both Python and numba code paths).
    station_lat_rad = float(station_lat.to(u.rad).value)
    station_lon_rad = float(station_lon.to(u.rad).value)
    grid_lats_rad = grid_latitudes.to(u.rad).value
    grid_lons_rad = grid_longitudes.to(u.rad).value

    # Compute the horizon angle (θₕ) for the station:
    R_earth_m = R_earth.to(u.m).value
    sat_alt_m = sat_altitude.to(u.m).value
    theta_h = np.arccos(R_earth_m / (R_earth_m + sat_alt_m))

    min_elev_rad = float(min_elevation.to(u.rad).value)

    # Simplify the margin γ using trigonometric identities:
    # γ = π/2 - βₘₐₓ - min_elevation = arccos((R_earth * cos(min_elevation))/(R_earth + sat_altitude)) - min_elevation
    gamma = np.arccos((R_earth_m * np.cos(min_elev_rad)) / (R_earth_m + sat_alt_m)) - min_elev_rad

    # A cell is impactful if its angular separation δ is less than or equal to (θₕ + γ)
    # and is on the near side of Earth (δ <= π/2). The calculation is routed through
    # a shared helper so that numba can optionally accelerate the vectorized math.

    # Maximum angular radius of potentially served cells around the station
    delta_max = theta_h + gamma
    cos_delta_max = float(np.cos(delta_max))

    return _compute_impact_mask(
        grid_lons_rad,
        grid_lats_rad,
        station_lat_rad,
        station_lon_rad,
        cos_delta_max,
    )


@ranged_quantity_input(
    alpha=(None, None, u.deg),
    altitude=(0.001, None, u.km),
    strip_input_units=False,
    allow_none=True,
)
def subsatellite_arc_distance(
    alpha: u.Quantity,
    altitude: u.Quantity,
    R: u.Quantity = R_earth,
) -> u.Quantity:
    """
    Compute the ground arc distance from nadir to the ray-Earth intersection.

    Parameters
    ----------
    alpha : astropy.units.Quantity
        Off-nadir angle at the satellite, measured in the scan plane.
    altitude : astropy.units.Quantity
        Satellite altitude above Earth's surface.
    R : astropy.units.Quantity, optional
        Earth radius used for the spherical geometry model.

    Returns
    -------
    astropy.units.Quantity
        Great-circle distance on the Earth surface between the subsatellite
        point and the ground intersection of the ray defined by ``alpha``.

    Notes
    -----
    The computation follows the same spherical-Earth geometry that is used in
    the step-1 notebook diagnostics and in :func:`calculate_footprint_size`.
    """
    alpha_rad = u.Quantity(alpha).to_value(u.rad)
    R_km = u.Quantity(R).to_value(u.km)
    Rs_km = (u.Quantity(R) + u.Quantity(altitude)).to_value(u.km)

    inside = R_km ** 2 - (Rs_km ** 2) * (np.sin(alpha_rad) ** 2)
    inside = np.maximum(inside, 0.0)

    slant = Rs_km * np.cos(alpha_rad) - np.sqrt(inside)
    z_coord = Rs_km - slant * np.cos(alpha_rad)

    cos_delta = np.clip(z_coord / R_km, -1.0, 1.0)
    delta = np.arccos(cos_delta)
    return (R_km * delta) * u.km


_BEAM_SPACING_RULES = {"center_to_contour", "full_footprint_diameter"}
_CONTOUR_DROP_PRESETS_DB = {
    "db3": 3.0,
    "db7": 7.0,
    "db15": 15.0,
}


def _normalise_beam_spacing_rule(rule: str | None) -> str:
    rule_use = str(rule or "center_to_contour").strip().lower()
    if rule_use not in _BEAM_SPACING_RULES:
        raise ValueError(
            "beam_spacing_rule must be one of "
            f"{sorted(_BEAM_SPACING_RULES)!r}."
        )
    return rule_use


def _format_contour_drop_db(drop_db: float) -> str:
    if np.isclose(drop_db, round(drop_db), atol=1e-9):
        return f"{int(round(drop_db))}"
    return f"{float(drop_db):.3f}".rstrip("0").rstrip(".")


def normalize_contour_drop(
    value: str | float | u.Quantity,
    *,
    name: str = "contour_drop",
) -> dict[str, Any]:
    """
    Normalize a contour attenuation drop specification.

    Parameters
    ----------
    value : {"db3", "db7", "db15"} or float or astropy.units.Quantity
        Contour attenuation drop. String presets are mapped to the
        corresponding positive attenuation value in dB. Plain numeric inputs
        are interpreted as a positive attenuation in dB. Quantity inputs are
        converted to ``pycraf.conversions.dB``.
    name : str, optional
        Parameter name used in validation errors.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        ``"drop_db"``
            Positive attenuation drop in dB as a float.
        ``"drop_quantity"``
            Same value as a ``cnv.dB`` quantity.
        ``"label"``
            Human-readable label such as ``"-7 dB"``.
        ``"preset"``
            Normalized preset name when a preset string was supplied, else
            ``None``.

    Raises
    ------
    ValueError
        Raised when the preset name is unsupported or when the resolved drop is
        not strictly positive.
    TypeError
        Raised when the input cannot be interpreted as a numeric attenuation.

    Notes
    -----
    This helper intentionally expects attenuation magnitudes as positive
    quantities, even though users often describe the contour verbally as
    ``-7 dB`` or ``-15 dB``. Negative numeric inputs are therefore rejected to
    avoid ambiguity.
    """
    preset_name: str | None = None
    if isinstance(value, str):
        preset_name = value.strip().lower()
        if preset_name not in _CONTOUR_DROP_PRESETS_DB:
            raise ValueError(
                f"{name} must be one of {sorted(_CONTOUR_DROP_PRESETS_DB)!r}, "
                "a positive float in dB, or a positive dB quantity."
            )
        drop_db = float(_CONTOUR_DROP_PRESETS_DB[preset_name])
    else:
        try:
            if np.isscalar(value) and not hasattr(value, "unit"):
                drop_db = float(value)
            else:
                value_q = u.Quantity(value)
                if value_q.unit == u.dimensionless_unscaled:
                    drop_db = float(value_q.value)
                else:
                    drop_db = float(value_q.to_value(u.dB))
        except Exception as exc:
            raise TypeError(
                f"{name} must be a preset string, a positive float in dB, or "
                "a positive dB quantity."
            ) from exc

    if not np.isfinite(drop_db) or drop_db <= 0.0:
        raise ValueError(f"{name} must resolve to a strictly positive attenuation drop in dB.")

    label = f"-{_format_contour_drop_db(drop_db)} dB"
    return {
        "drop_db": float(drop_db),
        "drop_quantity": float(drop_db) * cnv.dB,
        "label": label,
        "preset": preset_name,
    }


def resolve_contour_half_angle_deg(
    antenna_gain_func: Callable[..., Any],
    *,
    wavelength: u.Quantity,
    contour_drop: str | float | u.Quantity,
    **antenna_pattern_kwargs: Any,
) -> float:
    """Return the half-angle of a contour drop for a configured antenna pattern.

    Dispatches to ``calculate_beamwidth_2d`` which scans both principal
    planes for 2-D patterns (Custom 2-D, M.2101, asymmetric S.1528
    Rec 1.4) and falls back to ``calculate_beamwidth_1d`` for
    axisymmetric patterns. The returned half-angle is the geometric
    mean of the two principal-plane half-widths when the pattern is
    2-D, or the single radial half-width for 1-D.
    """
    drop_info = normalize_contour_drop(contour_drop, name="contour_drop")
    beamwidth = calculate_beamwidth_2d(
        antenna_gain_func,
        level_drop=drop_info["drop_quantity"],
        wavelength=wavelength,
        **antenna_pattern_kwargs,
    )
    return 0.5 * float(beamwidth.to_value(u.deg))


def _ground_spacing_from_beta_float(
    beta_rad: np.ndarray | float,
    delta_beta_rad: float,
    altitude_km: float,
    *,
    beam_spacing_rule: str,
) -> np.ndarray:
    rule = _normalise_beam_spacing_rule(beam_spacing_rule)
    beta_arr = np.asarray(beta_rad, dtype=np.float64)
    delta_beta_val = abs(float(delta_beta_rad))
    radius_km = float(R_earth.to_value(u.km))
    sat_radius_km = radius_km + float(altitude_km)
    alpha_limb = float(np.arcsin(np.clip(radius_km / sat_radius_km, -1.0, 1.0)))

    def _gamma_from_alpha(alpha_abs: np.ndarray) -> np.ndarray:
        ca = np.cos(alpha_abs)
        sa = np.sin(alpha_abs)
        under = radius_km * radius_km - sat_radius_km * sat_radius_km * (sa * sa)
        miss = under < 0.0
        under = np.maximum(under, 0.0)
        slant = sat_radius_km * ca - np.sqrt(under)
        miss |= slant < 0.0
        cos_delta = (sat_radius_km * sat_radius_km + radius_km * radius_km - slant * slant) / (
            2.0 * sat_radius_km * radius_km
        )
        cos_delta = np.clip(cos_delta, -1.0, 1.0)
        delta = np.arccos(cos_delta)
        return np.where(miss, np.nan, delta)

    def _signed_s(alpha_signed: np.ndarray) -> np.ndarray:
        return np.sign(alpha_signed) * _gamma_from_alpha(np.abs(alpha_signed))

    if rule == "center_to_contour":
        alpha_left = np.clip(beta_arr, -alpha_limb, +alpha_limb)
        alpha_right = np.clip(beta_arr + delta_beta_val, -alpha_limb, +alpha_limb)
    else:
        alpha_left = np.clip(beta_arr - delta_beta_val, -alpha_limb, +alpha_limb)
        alpha_right = np.clip(beta_arr + delta_beta_val, -alpha_limb, +alpha_limb)
    return radius_km * np.abs(_signed_s(alpha_right) - _signed_s(alpha_left))


@ranged_quantity_input(
    beta=(None, None, u.deg),
    delta_beta=(None, None, u.deg),
    altitude=(0.001, None, u.km),
    strip_input_units=False,
    allow_none=True,
)
def ground_separation_from_beta(
    beta: u.Quantity,
    delta_beta: u.Quantity,
    altitude: u.Quantity,
    *,
    beam_spacing_rule: str = "center_to_contour",
) -> u.Quantity:
    """
    Compute a scan-plane ground spacing metric for a beam contour.

    Parameters
    ----------
    beta : astropy.units.Quantity
        Reference off-nadir angle of the beam center.
    delta_beta : astropy.units.Quantity
        Contour half-angle used to evaluate the spacing metric.
    altitude : astropy.units.Quantity
        Satellite altitude above Earth's surface.
    beam_spacing_rule : {"center_to_contour", "full_footprint_diameter"}, optional
        Ground-spacing convention. ``"center_to_contour"`` measures the
        distance from the beam center at ``beta`` to the point at
        ``beta + delta_beta``. ``"full_footprint_diameter"`` measures the full
        contour diameter from ``beta - delta_beta`` to ``beta + delta_beta``.

    Returns
    -------
    astropy.units.Quantity
        Absolute great-circle spacing on Earth's surface in the scan plane.
    """
    spacing_km = _ground_spacing_from_beta_float(
        u.Quantity(beta, copy=False).to_value(u.rad),
        float(u.Quantity(delta_beta, copy=False).to_value(u.rad)),
        float(u.Quantity(altitude, copy=False).to_value(u.km)),
        beam_spacing_rule=beam_spacing_rule,
    )
    return spacing_km * u.km


def _coerce_angle_array_deg(value: u.Quantity | np.ndarray, *, name: str) -> np.ndarray:
    arr = u.Quantity(value, copy=False)
    if arr.unit == u.dimensionless_unscaled:
        arr = arr * u.deg
    else:
        arr = arr.to(u.deg)
    out = np.asarray(arr.to_value(u.deg), dtype=np.float64).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return out


def _coerce_scalar_angle_deg(value: u.Quantity | float, *, name: str) -> float:
    arr = u.Quantity(value, copy=False)
    if arr.unit == u.dimensionless_unscaled:
        arr = arr * u.deg
    else:
        arr = arr.to(u.deg)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar.")
    return float(arr.to_value(u.deg))


def _wrap_delta_longitude_rad(delta_lon_rad: np.ndarray) -> np.ndarray:
    return (delta_lon_rad + np.pi) % (2.0 * np.pi) - np.pi


def _great_circle_distance_km(
    longitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
    *,
    ref_lon_deg: float,
    ref_lat_deg: float,
) -> np.ndarray:
    lon_rad = np.deg2rad(np.asarray(longitudes_deg, dtype=np.float64))
    lat_rad = np.deg2rad(np.asarray(latitudes_deg, dtype=np.float64))
    ref_lon_rad = np.deg2rad(float(ref_lon_deg))
    ref_lat_rad = np.deg2rad(float(ref_lat_deg))
    delta_lon = _wrap_delta_longitude_rad(lon_rad - ref_lon_rad)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_ref_lat = np.sin(ref_lat_rad)
    cos_ref_lat = np.cos(ref_lat_rad)
    cos_delta = sin_ref_lat * sin_lat + cos_ref_lat * cos_lat * np.cos(delta_lon)
    return np.arccos(np.clip(cos_delta, -1.0, 1.0)) * float(R_earth.to_value(u.km))


def _local_tangent_offsets_km(
    longitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
    *,
    ref_lon_deg: float,
    ref_lat_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    lon_rad = np.deg2rad(np.asarray(longitudes_deg, dtype=np.float64))
    lat_rad = np.deg2rad(np.asarray(latitudes_deg, dtype=np.float64))
    ref_lon_rad = np.deg2rad(float(ref_lon_deg))
    ref_lat_rad = np.deg2rad(float(ref_lat_deg))
    radius_km = float(R_earth.to_value(u.km))
    delta_lon = _wrap_delta_longitude_rad(lon_rad - ref_lon_rad)
    east_km = radius_km * delta_lon * np.cos(ref_lat_rad)
    north_km = radius_km * (lat_rad - ref_lat_rad)
    return east_km, north_km


def _local_tangent_transformer_km(*, ref_lon_deg: float, ref_lat_deg: float):
    def _transformer(
        x: np.ndarray | float,
        y: np.ndarray | float,
        z: np.ndarray | float | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        east_km, north_km = _local_tangent_offsets_km(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            ref_lon_deg=ref_lon_deg,
            ref_lat_deg=ref_lat_deg,
        )
        if z is None:
            return east_km, north_km
        return east_km, north_km, np.asarray(z, dtype=np.float64)

    return _transformer


@lru_cache(maxsize=8)
def _build_projected_geography_reference(
    ref_lon_deg: float,
    ref_lat_deg: float,
    backend: str = "vendored",
) -> tuple[object, object]:
    shapely, _, shapely_ops = _require_shapely(
        purpose="projected land/coast geometry preparation",
    )
    backend_name = _normalise_coastline_backend(backend)
    transformer = _local_tangent_transformer_km(
        ref_lon_deg=float(ref_lon_deg),
        ref_lat_deg=float(ref_lat_deg),
    )
    land_union = shapely.unary_union(
        [
            shapely.make_valid(shapely_ops.transform(transformer, geom))
            for geom in _load_natural_earth_geometries("land", backend=backend_name)
        ]
    )
    coastline_union = shapely.unary_union(
        [
            shapely.make_valid(shapely_ops.transform(transformer, geom))
            for geom in _load_natural_earth_geometries("coastline", backend=backend_name)
        ]
    )
    return land_union, coastline_union


def _classify_hexgrid_geography(
    cell_longitudes_deg: np.ndarray,
    cell_latitudes_deg: np.ndarray,
    *,
    candidate_mask: np.ndarray,
    station_lon_deg: float,
    station_lat_deg: float,
    geography_mask_mode: str,
    shoreline_buffer_km: float | None,
    coastline_backend: str,
) -> dict[str, np.ndarray]:
    mode_name = _normalise_geography_mask_mode(geography_mask_mode)
    backend_name = _normalise_coastline_backend(coastline_backend)
    candidate_mask_use = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask_use.shape != cell_longitudes_deg.shape:
        raise ValueError("candidate_mask must have the same shape as the cell-coordinate arrays.")

    if mode_name == "none":
        return {
            "geography_mask": np.ones(cell_longitudes_deg.shape, dtype=bool),
            "land_mask": np.zeros(cell_longitudes_deg.shape, dtype=bool),
            "nearshore_sea_mask": np.zeros(cell_longitudes_deg.shape, dtype=bool),
            "shore_distance_km": np.full(cell_longitudes_deg.shape, np.nan, dtype=np.float64),
        }

    shapely, _, _ = _require_shapely(
        purpose="hexgrid land/coast masking",
    )

    shoreline_buffer_value = None
    if mode_name == "land_plus_nearshore_sea":
        if shoreline_buffer_km is None:
            raise ValueError(
                "shoreline_buffer_km is required when geography_mask_mode='land_plus_nearshore_sea'."
            )
        shoreline_buffer_value = float(shoreline_buffer_km)

    candidate_ids = np.flatnonzero(candidate_mask_use)
    geography_mask = np.ones(cell_longitudes_deg.shape, dtype=bool)
    land_mask = np.zeros(cell_longitudes_deg.shape, dtype=bool)
    nearshore_sea_mask = np.zeros(cell_longitudes_deg.shape, dtype=bool)
    shore_distance_km = np.full(cell_longitudes_deg.shape, np.nan, dtype=np.float64)

    if candidate_ids.size == 0:
        return {
            "geography_mask": geography_mask,
            "land_mask": land_mask,
            "nearshore_sea_mask": nearshore_sea_mask,
            "shore_distance_km": shore_distance_km,
        }

    candidate_lon_deg = np.asarray(cell_longitudes_deg[candidate_ids], dtype=np.float64)
    candidate_lat_deg = np.asarray(cell_latitudes_deg[candidate_ids], dtype=np.float64)
    cache_key = (
        mode_name,
        backend_name,
        round(float(station_lon_deg), 6),
        round(float(station_lat_deg), 6),
        None if shoreline_buffer_value is None else round(float(shoreline_buffer_value), 6),
        tuple(candidate_mask_use.shape),
        _array_cache_token(candidate_lon_deg),
        _array_cache_token(candidate_lat_deg),
    )
    cached = _GEOGRAPHY_CLASSIFICATION_CACHE.get(cache_key)
    if cached is not None:
        _GEOGRAPHY_CLASSIFICATION_CACHE.move_to_end(cache_key)
        return _copy_geography_classification_result(cached)

    east_km_sel, north_km_sel = _local_tangent_offsets_km(
        candidate_lon_deg,
        candidate_lat_deg,
        ref_lon_deg=station_lon_deg,
        ref_lat_deg=station_lat_deg,
    )
    points = shapely.points(east_km_sel, north_km_sel)
    land_union, coastline_union = _build_projected_geography_reference(
        float(station_lon_deg),
        float(station_lat_deg),
        backend=backend_name,
    )

    land_mask_sel = np.asarray(shapely.covered_by(points, land_union), dtype=bool)

    land_mask[candidate_ids] = land_mask_sel

    if mode_name == "land_only":
        geography_mask[candidate_ids] = land_mask_sel
    else:
        shore_distance_sel = np.asarray(
            shapely.distance(points, coastline_union),
            dtype=np.float64,
        )
        if shoreline_buffer_value == 0.0:
            shore_distance_store = np.where(land_mask_sel, np.nan, shore_distance_sel)
            shore_distance_km[candidate_ids] = shore_distance_store
            geography_mask[candidate_ids] = land_mask_sel
        elif shoreline_buffer_value > 0.0:
            shore_distance_store = np.where(land_mask_sel, np.nan, shore_distance_sel)
            shore_distance_km[candidate_ids] = shore_distance_store
            nearshore_sel = (~land_mask_sel) & np.isfinite(shore_distance_sel) & (
                shore_distance_sel <= float(shoreline_buffer_value)
            )
            nearshore_sea_mask[candidate_ids] = nearshore_sel
            geography_mask[candidate_ids] = land_mask_sel | nearshore_sel
        else:
            shore_distance_km[candidate_ids] = shore_distance_sel
            erosion_distance_km = abs(float(shoreline_buffer_value))
            eroded_land_sel = land_mask_sel & (
                (~np.isfinite(shore_distance_sel)) | (shore_distance_sel > erosion_distance_km)
            )
            geography_mask[candidate_ids] = eroded_land_sel

    result = {
        "geography_mask": geography_mask,
        "land_mask": land_mask,
        "nearshore_sea_mask": nearshore_sea_mask,
        "shore_distance_km": shore_distance_km,
    }
    _GEOGRAPHY_CLASSIFICATION_CACHE[cache_key] = _copy_geography_classification_result(result)
    while len(_GEOGRAPHY_CLASSIFICATION_CACHE) > _GEOGRAPHY_CLASSIFICATION_CACHE_MAX:
        _GEOGRAPHY_CLASSIFICATION_CACHE.popitem(last=False)
    return result


def _resolve_local_hex_ring_ids(
    cell_longitudes_deg: np.ndarray,
    cell_latitudes_deg: np.ndarray,
    *,
    ras_cell_index: int,
    layers: int,
) -> np.ndarray:
    if layers < 0:
        raise ValueError("layers must be non-negative.")
    if cell_longitudes_deg.size != cell_latitudes_deg.size:
        raise ValueError("cell_longitudes_deg and cell_latitudes_deg must have the same size.")
    if ras_cell_index < 0 or ras_cell_index >= int(cell_longitudes_deg.size):
        raise ValueError(
            f"ras_cell_index must lie within [0, {int(cell_longitudes_deg.size) - 1}]."
        )
    if layers == 0:
        return np.asarray([int(ras_cell_index)], dtype=np.int32)

    ref_lon_deg = float(cell_longitudes_deg[int(ras_cell_index)])
    ref_lat_deg = float(cell_latitudes_deg[int(ras_cell_index)])
    east_km, north_km = _local_tangent_offsets_km(
        cell_longitudes_deg,
        cell_latitudes_deg,
        ref_lon_deg=ref_lon_deg,
        ref_lat_deg=ref_lat_deg,
    )
    points_km = np.column_stack((east_km, north_km))
    if points_km.shape[0] <= 1:
        return np.asarray([int(ras_cell_index)], dtype=np.int32)

    n_points = int(points_km.shape[0])
    k_neighbors = min(7, n_points)
    tree = cKDTree(points_km)
    neighbor_dist, neighbor_idx = tree.query(points_km, k=k_neighbors)
    neighbor_dist = np.asarray(neighbor_dist, dtype=np.float64)
    neighbor_idx = np.asarray(neighbor_idx, dtype=np.int32)
    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]
        neighbor_dist = neighbor_dist[:, None]

    if neighbor_dist.shape[1] <= 1:
        return np.asarray([int(ras_cell_index)], dtype=np.int32)

    nearest_neighbor_dist = neighbor_dist[:, 1]
    nearest_neighbor_dist = nearest_neighbor_dist[np.isfinite(nearest_neighbor_dist) & (nearest_neighbor_dist > 1e-9)]
    if nearest_neighbor_dist.size == 0:
        return np.asarray([int(ras_cell_index)], dtype=np.int32)
    spacing_km = float(np.median(nearest_neighbor_dist))
    max_neighbor_distance_km = spacing_km * 1.35

    # Build a symmetric local hex-neighbor graph from the compact tangent-plane
    # point cloud, then expand rings with a bounded BFS from the RAS cell.
    adjacency: list[set[int]] = [set() for _ in range(n_points)]
    for src_idx in range(n_points):
        for dist_km, dst_idx in zip(neighbor_dist[src_idx], neighbor_idx[src_idx]):
            dst = int(dst_idx)
            if dst == src_idx:
                continue
            if not np.isfinite(dist_km) or float(dist_km) > max_neighbor_distance_km:
                continue
            adjacency[src_idx].add(dst)
            adjacency[dst].add(src_idx)

    visited = np.zeros(n_points, dtype=bool)
    distance = np.full(n_points, -1, dtype=np.int32)
    frontier: deque[int] = deque([int(ras_cell_index)])
    visited[int(ras_cell_index)] = True
    distance[int(ras_cell_index)] = 0

    while frontier:
        current = frontier.popleft()
        current_distance = int(distance[current])
        if current_distance >= layers:
            continue
        for neighbor in adjacency[current]:
            if visited[neighbor]:
                continue
            visited[neighbor] = True
            distance[neighbor] = current_distance + 1
            frontier.append(neighbor)

    selected = np.flatnonzero((distance >= 0) & (distance <= int(layers)))
    if selected.size == 0:
        return np.asarray([int(ras_cell_index)], dtype=np.int32)
    return selected.astype(np.int32, copy=False)


def resolve_ras_hexgrid_cell_ids(
    cell_longitudes: u.Quantity | np.ndarray,
    cell_latitudes: u.Quantity | np.ndarray,
    *,
    station_lat: u.Quantity | float,
    station_lon: u.Quantity | float,
    mode: str,
    ras_cell_index: int | None = None,
    layers: int | None = None,
    radius_km: float | None = None,
) -> np.ndarray:
    """
    Resolve active hexgrid cell ids around the RAS station.

    Parameters
    ----------
    cell_longitudes, cell_latitudes : astropy.units.Quantity or np.ndarray
        One-dimensional active-cell center coordinates. Plain arrays are
        interpreted as degrees.
    station_lat, station_lon : astropy.units.Quantity or float
        RAS-station geodetic coordinates. Plain scalars are interpreted as
        degrees.
    mode : {"none", "adjacency_layers", "radius_km"}
        Resolution mode. ``"none"`` returns an empty selection,
        ``"adjacency_layers"`` returns cells within the requested local
        hex-ring distance of ``ras_cell_index``, and ``"radius_km"`` returns
        cells whose centers lie within ``radius_km`` of the RAS station.
    ras_cell_index : int or None, optional
        Active-grid index of the RAS-containing service cell. Required for
        ``mode="adjacency_layers"``.
    layers : int or None, optional
        Number of adjacency rings to include for
        ``mode="adjacency_layers"``. ``layers=0`` selects only the
        RAS-containing cell.
    radius_km : float or None, optional
        Great-circle radius around the RAS station used when
        ``mode="radius_km"``.

    Returns
    -------
    np.ndarray
        Sorted ``int32`` array of selected active-cell ids. The array may be
        empty when ``mode="none"`` or when the requested radius excludes no
        cells.

    Raises
    ------
    ValueError
        Raised when the inputs are inconsistent or when the requested mode
        lacks its required parameters.

    Notes
    -----
    ``"adjacency_layers"`` uses local tangent-plane coordinates around the
    RAS service cell, builds a symmetrized 6-nearest-neighbor graph on that
    compact point cloud, and returns all cells within the requested
    breadth-first hop count. This is stable on the large prefilter grids used
    by the step1+2 notebooks without requiring an all-pairs neighbor search.
    """
    cell_longitudes_deg = _coerce_angle_array_deg(cell_longitudes, name="cell_longitudes")
    cell_latitudes_deg = _coerce_angle_array_deg(cell_latitudes, name="cell_latitudes")
    if cell_longitudes_deg.size != cell_latitudes_deg.size:
        raise ValueError("cell_longitudes and cell_latitudes must have the same size.")

    mode_normalized = str(mode).strip().lower()
    if mode_normalized == "none":
        return np.empty(0, dtype=np.int32)
    if mode_normalized == "adjacency_layers":
        if ras_cell_index is None:
            raise ValueError("ras_cell_index is required when mode='adjacency_layers'.")
        if layers is None:
            raise ValueError("layers is required when mode='adjacency_layers'.")
        return _resolve_local_hex_ring_ids(
            cell_longitudes_deg,
            cell_latitudes_deg,
            ras_cell_index=int(ras_cell_index),
            layers=int(layers),
        )
    if mode_normalized == "radius_km":
        if radius_km is None:
            raise ValueError("radius_km is required when mode='radius_km'.")
        radius_km_value = float(radius_km)
        if radius_km_value < 0.0:
            raise ValueError("radius_km must be non-negative.")
        station_lat_deg = _coerce_scalar_angle_deg(station_lat, name="station_lat")
        station_lon_deg = _coerce_scalar_angle_deg(station_lon, name="station_lon")
        dist_km = _great_circle_distance_km(
            cell_longitudes_deg,
            cell_latitudes_deg,
            ref_lon_deg=station_lon_deg,
            ref_lat_deg=station_lat_deg,
        )
        return np.flatnonzero(dist_km <= radius_km_value).astype(np.int32, copy=False)
    raise ValueError("mode must be one of {'none', 'adjacency_layers', 'radius_km'}.")


def mask_hexgrid_for_constellation(
    grid_longitudes: u.Quantity | np.ndarray,
    grid_latitudes: u.Quantity | np.ndarray,
    *,
    altitudes: u.Quantity | np.ndarray,
    min_elevations: u.Quantity | np.ndarray,
    inclinations: u.Quantity | np.ndarray,
    station_lat: u.Quantity,
    station_lon: u.Quantity,
    station_height: u.Quantity | None = None,
    latitude_policy: str = "any",
    geography_mask_mode: str = "none",
    shoreline_buffer_km: float | None = None,
    coastline_backend: str = "vendored",
) -> dict[str, u.Quantity | np.ndarray | str]:
    """
    Build a constellation-aware hex-grid mask from per-belt geometry.

    Parameters
    ----------
    grid_longitudes : astropy.units.Quantity or np.ndarray
        Candidate grid-cell longitudes. Plain arrays are interpreted as degrees.
    grid_latitudes : astropy.units.Quantity or np.ndarray
        Candidate grid-cell latitudes. Plain arrays are interpreted as degrees.
    altitudes : astropy.units.Quantity or np.ndarray
        One altitude per belt.
    min_elevations : astropy.units.Quantity or np.ndarray
        One operational minimum elevation per belt.
    inclinations : astropy.units.Quantity or np.ndarray
        One inclination per belt.
    station_lat : astropy.units.Quantity
        Radio-astronomy station latitude.
    station_lon : astropy.units.Quantity
        Radio-astronomy station longitude.
    station_height : astropy.units.Quantity or None, optional
        Station height, passed through to :func:`trunc_hexgrid_to_impactful`
        for API completeness.
    latitude_policy : {"any", "all", "first"}, optional
        How per-belt masks are combined:
        ``"any"`` keeps cells reachable by at least one belt,
        ``"all"`` keeps only cells reachable by every belt, and
        ``"first"`` uses only the first belt.
    geography_mask_mode : {"none", "land_only", "land_plus_nearshore_sea"}, optional
        Optional geography filter applied after the impactful and latitude
        masks. ``"none"`` preserves current behavior, ``"land_only"`` keeps
        only land cells, and ``"land_plus_nearshore_sea"`` applies a signed
        shoreline buffer: ``0`` behaves like ``"land_only"``, positive values
        keep all land cells plus sea cells whose center lies within
        ``shoreline_buffer_km`` of the shoreline, and negative values erode
        the land mask inland by ``abs(shoreline_buffer_km)`` with no sea
        extension.
    shoreline_buffer_km : float or None, optional
        Signed shoreline buffer used only when
        ``geography_mask_mode="land_plus_nearshore_sea"``.
    coastline_backend : {"vendored", "cartopy"}, optional
        Geometry source used for land/coast masking. ``"vendored"`` reads the
        repo-shipped Natural Earth GeoJSON files. ``"cartopy"`` loads Natural
        Earth through Cartopy and may trigger Cartopy's normal data download
        path.

    Returns
    -------
    dict[str, astropy.units.Quantity | np.ndarray]
        Dictionary with keys:

        ``"base_mask"``
            Station-centric impactful mask combined across belts.
        ``"lat_mask"``
            Latitude-feasibility mask from inclination plus service cap.
        ``"combined_mask"``
            Final ``base_mask & lat_mask`` intersection.
        ``"phi_max_per_belt"``
            Maximum reachable latitude for each belt.
        ``"phi_limit"``
            Scalar latitude cap implied by ``latitude_policy``.
        ``"geography_mask"``
            Optional geography keep mask on the full candidate-grid axis.
        ``"land_mask"``
            Land classification on the full candidate-grid axis. Values are
            populated for cells that pass the impactful+latitude prefilter.
        ``"nearshore_sea_mask"``
            Sea-cell subset kept by ``land_plus_nearshore_sea``.
        ``"shore_distance_km"``
            Distance from each candidate cell center to the shoreline in km.
            Values are populated for cells that pass the impactful+latitude
            prefilter.
        ``"geography_mask_mode"``
            Normalized geography mode actually used.
        ``"coastline_backend"``
            Normalized coastline backend actually used.

    Raises
    ------
    ValueError
        If the per-belt arrays are empty, have mismatched lengths, or if
        ``latitude_policy``, ``geography_mask_mode``, or
        ``coastline_backend`` is unsupported.

    Notes
    -----
    For multi-belt constellations this helper uses per-belt union/intersection
    logic instead of the single conservative envelope
    ``max(altitude), min(min_elevation)`` that was previously embedded in the
    step-1 notebook script. Geography classification is computed only for cells
    that already pass the impactful+latitude prefilter.
    """
    grid_longitudes_q = u.Quantity(grid_longitudes, copy=False)
    grid_latitudes_q = u.Quantity(grid_latitudes, copy=False)
    if grid_longitudes_q.unit == u.dimensionless_unscaled:
        grid_longitudes_q = grid_longitudes_q * u.deg
    else:
        grid_longitudes_q = grid_longitudes_q.to(u.deg)
    if grid_latitudes_q.unit == u.dimensionless_unscaled:
        grid_latitudes_q = grid_latitudes_q * u.deg
    else:
        grid_latitudes_q = grid_latitudes_q.to(u.deg)

    altitudes_q = u.Quantity(altitudes).to(u.km)
    min_elevations_q = u.Quantity(min_elevations).to(u.deg)
    inclinations_q = u.Quantity(inclinations).to(u.deg)

    if altitudes_q.size == 0:
        raise ValueError("At least one belt is required.")
    if not (altitudes_q.size == min_elevations_q.size == inclinations_q.size):
        raise ValueError("altitudes, min_elevations, and inclinations must have the same length.")

    base_masks = []
    phi_max_list_deg = []
    R_km = float(R_earth.to_value(u.km))

    for altitude, min_elevation, inclination in zip(altitudes_q, min_elevations_q, inclinations_q):
        base_masks.append(
            trunc_hexgrid_to_impactful(
                grid_longitudes_q,
                grid_latitudes_q,
                altitude,
                min_elevation,
                station_lat,
                station_lon,
                station_height,
            )
        )

        Rs_km = R_km + float(altitude.to_value(u.km))
        min_elev_rad = float(min_elevation.to_value(u.rad))
        inclination_rad = float(inclination.to_value(u.rad))

        beta_b = np.arcsin(np.clip((R_km / Rs_km) * np.cos(min_elev_rad), -1.0, 1.0))
        gamma_max_b = 0.5 * np.pi + min_elev_rad - beta_b
        gamma_horizon = np.arccos(np.clip(R_km / Rs_km, -1.0, 1.0))
        gamma_max_b = float(np.clip(gamma_max_b, 0.0, gamma_horizon))

        phi_max_b = min(inclination_rad + gamma_max_b, 0.5 * np.pi)
        phi_max_list_deg.append(np.degrees(phi_max_b))

    base_masks_stack = np.stack(base_masks, axis=0)
    phi_max_per_belt = u.Quantity(phi_max_list_deg, u.deg)
    phi_cap = 90.0 * u.deg

    policy = (latitude_policy or "any").lower()
    if policy == "any":
        base_mask = np.any(base_masks_stack, axis=0)
        phi_limit = min(phi_max_per_belt.max(), phi_cap)
    elif policy == "all":
        base_mask = np.all(base_masks_stack, axis=0)
        phi_limit = min(phi_max_per_belt.min(), phi_cap)
    elif policy == "first":
        base_mask = base_masks_stack[0]
        phi_limit = min(phi_max_per_belt[0], phi_cap)
    else:
        raise ValueError("latitude_policy must be 'any', 'all', or 'first'.")

    lat_mask = np.abs(grid_latitudes_q) <= phi_limit
    impactful_reachable_mask = base_mask & lat_mask

    geography_info = _classify_hexgrid_geography(
        grid_longitudes_q.to_value(u.deg),
        grid_latitudes_q.to_value(u.deg),
        candidate_mask=impactful_reachable_mask,
        station_lon_deg=_coerce_scalar_angle_deg(station_lon, name="station_lon"),
        station_lat_deg=_coerce_scalar_angle_deg(station_lat, name="station_lat"),
        geography_mask_mode=geography_mask_mode,
        shoreline_buffer_km=shoreline_buffer_km,
        coastline_backend=coastline_backend,
    )
    combined_mask = impactful_reachable_mask & geography_info["geography_mask"]
    geography_mode_name = _normalise_geography_mask_mode(geography_mask_mode)
    coastline_backend_name = _normalise_coastline_backend(coastline_backend)

    return {
        "base_mask": base_mask,
        "lat_mask": lat_mask,
        "combined_mask": combined_mask,
        "phi_max_per_belt": phi_max_per_belt,
        "phi_limit": phi_limit,
        "geography_mask": geography_info["geography_mask"],
        "land_mask": geography_info["land_mask"],
        "nearshore_sea_mask": geography_info["nearshore_sea_mask"],
        "shore_distance_km": geography_info["shore_distance_km"],
        "geography_mask_mode": geography_mode_name,
        "coastline_backend": coastline_backend_name,
    }

def recommend_cell_diameter(
    antenna_gain_func: Callable[..., Any],
    *,
    altitude: u.Quantity,
    min_elevation: u.Quantity,
    wavelength: u.Quantity,
    # --- strategy controls ---
    strategy: str = "random_pointing",
    n_pool_sats: int | None = None,
    n_vis_override: int | None = None,
    vis_count_scale: float = 1.0,
    vis_count_model: str = "poisson",  # "mean" or "poisson"
    # --- footprint contour (indicative/reference) ---
    footprint_drop: str | float | u.Quantity = "db3",
    footprint_theta_edge: u.Quantity | None = None,  # HALF-angle; if None computed from pattern
    # --- operational spacing contour ---
    spacing_drop: str | float | u.Quantity | None = "db15",
    spacing_theta_edge: u.Quantity | None = None,  # HALF-angle; if None computed from pattern
    beam_spacing_rule: str = "center_to_contour",
    # --- recommendation logic ---
    leading_metric: str = "footprint_contour",  # "footprint_contour" | "spacing_contour" | "max_of_both" | "min_of_both"
    cell_quantile: float = 0.5,
    spacing_quantile: float | None = None,
    footprint_quantile: float | None = None,
    enforce_spacing_min: bool = False,
    # --- guardrails ---
    footprint_guard_policy: str = "warn_if_footprint_leading",  # "warn" | "warn_if_footprint_leading" | "off"
    footprint_guard_max_ratio: float = 2.5,
    # --- Monte-Carlo ---
    n_samples: int = 200_000,
    seed: int = 0,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9, 0.95),
    return_samples: bool = False,
    # --- antenna pattern kwargs ---
    **antenna_pattern_kwargs,
) -> dict:
    """
    Recommend a characteristic Earth cell spacing for gridding served locations.

    Parameters
    ----------
    antenna_gain_func : callable
        One-dimensional antenna-gain function compatible with
        :func:`calculate_beamwidth_1d` and :func:`calculate_footprint_size`.
    altitude : astropy.units.Quantity
        Satellite altitude above Earth's surface.
    min_elevation : astropy.units.Quantity
        Operational minimum serving elevation used to derive the visible-beta
        distribution.
    wavelength : astropy.units.Quantity
        Operating wavelength passed through to the antenna pattern helpers.
    strategy : {"random_pointing", "maximum_elevation"}, optional
        Visibility model used to sample the off-nadir beta distribution.
    n_pool_sats, n_vis_override, vis_count_scale, vis_count_model : optional
        Controls for the visible-satellite-count model used by the Monte-Carlo
        beta sampler.
    footprint_drop, footprint_theta_edge : optional
        Footprint contour used for the reference footprint-diameter diagnostic.
        ``footprint_drop`` accepts preset strings such as ``"db3"``,
        positive floats, or positive ``cnv.dB`` quantities. When
        ``footprint_theta_edge`` is omitted, the half-angle is derived from the
        antenna pattern at ``footprint_drop``.
    spacing_drop, spacing_theta_edge, beam_spacing_rule : optional
        Operational spacing contour and geometry convention used for the
        spacing diagnostic. ``beam_spacing_rule="center_to_contour"``
        measures a one-sided center-to-contour spacing, while
        ``"full_footprint_diameter"`` measures the full ground diameter of the
        contour in the scan plane. When ``spacing_drop`` is ``None``, the
        spacing diagnostic is disabled.
    leading_metric : {"footprint_contour", "spacing_contour", "max_of_both", "min_of_both"}, optional
        Metric used to form the final ``recommended_cell_spacing_km``.
    cell_quantile, spacing_quantile, footprint_quantile : float, optional
        Recommendation quantiles on the combined, spacing-only, and
        footprint-only distributions.
    enforce_spacing_min : bool, optional
        If ``True``, enforce the spacing metric as a floor even when the
        leading metric would otherwise choose a smaller recommendation.
    footprint_guard_policy, footprint_guard_max_ratio : optional
        Guardrails for unusually large footprint-driven recommendations.
    n_samples : int, optional
        Monte-Carlo sample count for the beta-distribution estimate.
    seed : int, optional
        Random seed for reproducible beta sampling.
    quantiles : tuple[float, ...], optional
        Quantiles reported in the returned diagnostics.
    return_samples : bool, optional
        If ``True``, include the sampled beta, spacing, and footprint arrays in
        the result dictionary.
    **antenna_pattern_kwargs
        Additional keyword arguments forwarded to the antenna pattern helper.

    Returns
    -------
    dict
        Diagnostic dictionary containing the recommended cell spacing, the
        sampled visibility/beta statistics, quantiles for the footprint and
        spacing metrics, and metadata describing which spacing rule and leading
        metric were applied. When ``return_samples=True``, the dictionary also
        contains raw sample arrays such as ``beta_samples_deg``,
        ``spacing_samples_km``, and ``footprint_samples_km``.

    Raises
    ------
    ValueError
        Raised when the sampling strategy, contour specification, visibility
        model, quantiles, or leading-metric options are inconsistent.
    TypeError
        Raised when ``antenna_gain_func`` is not callable.

    Notes
    -----
    Two distributions are evaluated:

    - Footprint diameter at ``footprint_drop`` in the scan plane.
    - Ground spacing at ``spacing_drop`` under ``beam_spacing_rule``.

    The final recommendation can be driven by either metric via
    ``leading_metric``. A large spacing contour does not automatically mean the
    grid must use that spacing as its cell spacing; by default it remains a
    diagnostic unless selected explicitly through ``leading_metric`` or
    ``enforce_spacing_min``.
    """

    # -----------------------------
    # 0) Validate inputs
    # -----------------------------
    if strategy not in ("random_pointing", "maximum_elevation"):
        raise ValueError("strategy must be 'random_pointing' or 'maximum_elevation'")

    if vis_count_model not in ("mean", "poisson"):
        raise ValueError("vis_count_model must be 'mean' or 'poisson'")

    allowed_metrics = ("footprint_contour", "spacing_contour", "max_of_both", "min_of_both")
    if leading_metric not in allowed_metrics:
        raise ValueError(f"leading_metric must be one of {allowed_metrics}")
    beam_spacing_rule_name = _normalise_beam_spacing_rule(beam_spacing_rule)

    allowed_guard = ("warn", "warn_if_footprint_leading", "off")
    if footprint_guard_policy not in allowed_guard:
        raise ValueError(f"footprint_guard_policy must be one of {allowed_guard}")

    def _check_q(q: float, name: str) -> float:
        if not (0.0 < q < 1.0):
            raise ValueError(f"{name} must be in (0, 1)")
        return float(q)

    cell_quantile = _check_q(cell_quantile, "cell_quantile")
    spacing_q = _check_q(
        spacing_quantile if spacing_quantile is not None else cell_quantile,
        "spacing_quantile",
    )
    fp_q = _check_q(footprint_quantile if footprint_quantile is not None else cell_quantile, "footprint_quantile")

    if not callable(antenna_gain_func):
        raise TypeError("antenna_gain_func must be callable")

    footprint_drop_info = normalize_contour_drop(footprint_drop, name="footprint_drop")
    spacing_drop_info = (
        normalize_contour_drop(spacing_drop, name="spacing_drop")
        if spacing_drop is not None
        else None
    )

    rng = np.random.default_rng(seed)

    # Convert key geometry to floats (fast inner math)
    R_km = float(R_earth.to_value(u.km))
    h_km = float(altitude.to_value(u.km))
    Rs_km = R_km + h_km
    e_min = float(min_elevation.to_value(u.rad))

    # Inject wavelength for beamwidth computations
    antenna_pattern_kwargs = dict(antenna_pattern_kwargs)
    antenna_pattern_kwargs["wavelength"] = wavelength

    # -----------------------------
    # 1) Determine contour half-angles (footprint_theta_edge, spacing_theta_edge)
    # -----------------------------
    if footprint_theta_edge is None:
        bw_full = calculate_beamwidth_1d(
            antenna_gain_func,
            level_drop=footprint_drop_info["drop_quantity"],
            **antenna_pattern_kwargs,
        )
        footprint_theta = 0.5 * float(bw_full.to_value(u.rad))
    else:
        footprint_theta = float(u.Quantity(footprint_theta_edge).to_value(u.rad))
    footprint_theta = abs(footprint_theta)

    spacing_theta = None
    if spacing_drop_info is not None:
        if spacing_theta_edge is None:
            bw_sep_full = calculate_beamwidth_1d(
                antenna_gain_func,
                level_drop=spacing_drop_info["drop_quantity"],
                **antenna_pattern_kwargs,
            )
            spacing_theta = 0.5 * float(bw_sep_full.to_value(u.rad))
        else:
            spacing_theta = float(u.Quantity(spacing_theta_edge).to_value(u.rad))
        spacing_theta = abs(spacing_theta)

    if leading_metric in ("spacing_contour", "max_of_both", "min_of_both") and spacing_theta is None:
        raise ValueError(
            "leading_metric requires spacing but spacing_drop/spacing_theta_edge was not provided."
        )

    # -----------------------------
    # 2) Core spherical geometry helpers (float math)
    # -----------------------------
    ratio = R_km / Rs_km
    ratio = -1.0 if ratio < -1.0 else (1.0 if ratio > 1.0 else ratio)
    alpha_limb = float(np.arcsin(ratio))  # [rad]

    def _elevation_from_gamma(gamma: np.ndarray) -> np.ndarray:
        cg = np.cos(gamma)
        rng_km = np.sqrt(Rs_km * Rs_km + R_km * R_km - 2.0 * R_km * Rs_km * cg)
        s = (Rs_km * cg - R_km) / rng_km
        s = np.clip(s, -1.0, 1.0)
        return np.arcsin(s)

    def _solve_gamma_max() -> float:
        gamma_horizon = float(np.arccos(R_km / Rs_km))
        lo, hi = 0.0, gamma_horizon
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(_elevation_from_gamma(np.array([mid]))[0]) > e_min:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    gamma_max = _solve_gamma_max()
    cos_gmax = float(np.cos(gamma_max))
    beta_max = float(np.arcsin((R_km / Rs_km) * np.cos(e_min)))

    def _gamma_from_alpha(alpha_abs: np.ndarray) -> np.ndarray:
        ca = np.cos(alpha_abs)
        sa = np.sin(alpha_abs)

        under = R_km * R_km - Rs_km * Rs_km * (sa * sa)
        miss = under < 0.0
        under = np.maximum(under, 0.0)

        se = Rs_km * ca - np.sqrt(under)
        miss |= se < 0.0

        cosg = (Rs_km * Rs_km + R_km * R_km - se * se) / (2.0 * Rs_km * R_km)
        cosg = np.clip(cosg, -1.0, 1.0)

        g = np.arccos(cosg)
        g[miss] = np.nan
        return g

    def _signed_s(alpha_signed: np.ndarray) -> np.ndarray:
        return np.sign(alpha_signed) * _gamma_from_alpha(np.abs(alpha_signed))

    # -----------------------------
    # 3) Sample ground locations -> beta distribution
    # -----------------------------
    n_vis_mean_est = None

    if strategy == "random_pointing":
        U = rng.random(int(n_samples))
        cosg = 1.0 - U * (1.0 - cos_gmax)
        gamma = np.arccos(cosg)

    else:
        if n_vis_override is not None:
            N_mean = float(max(1, int(n_vis_override)))
        else:
            if n_pool_sats is None:
                raise ValueError("For maximum_elevation, provide n_pool_sats or n_vis_override.")
            p_cap = (1.0 - cos_gmax) / 2.0
            N_mean = float(max(1.0, float(n_pool_sats) * p_cap * float(vis_count_scale)))
        n_vis_mean_est = N_mean

        if vis_count_model == "mean":
            N = np.full(int(n_samples), int(round(N_mean)), dtype=np.int64)
            N[N < 1] = 1
        else:
            N = rng.poisson(lam=N_mean, size=int(n_samples)).astype(np.int64)
            N[N < 1] = 1

        U = rng.random(int(n_samples))
        Xmin = 1.0 - (1.0 - U) ** (1.0 / N.astype(np.float64))
        cosg = 1.0 - Xmin * (1.0 - cos_gmax)
        gamma = np.arccos(cosg)

    elev = _elevation_from_gamma(gamma)
    sinb = (R_km / Rs_km) * np.cos(elev)
    sinb = np.clip(sinb, -1.0, 1.0)
    beta = np.arcsin(sinb)  # [rad], >=0

    # -----------------------------
    # 4) Footprint diameters at footprint_drop
    # -----------------------------
    a1 = np.clip(beta - footprint_theta, -alpha_limb, +alpha_limb)
    a2 = np.clip(beta + footprint_theta, -alpha_limb, +alpha_limb)

    s1 = _signed_s(a1)
    s2 = _signed_s(a2)

    footprint_km = R_km * np.abs(s2 - s1)
    msk = np.isfinite(footprint_km)
    footprint_ok = footprint_km[msk]
    beta_ok = beta[msk]

    if footprint_ok.size == 0:
        raise RuntimeError("All sampled footprints are invalid. Check inputs / antenna parameters.")

    # -----------------------------
    # 5) Ground spacing metric corresponding to spacing_theta (diagnostic)
    # -----------------------------
    spacing_ok = None
    if spacing_theta is not None:
        spacing_km = _ground_spacing_from_beta_float(
            beta_ok,
            spacing_theta,
            h_km,
            beam_spacing_rule=beam_spacing_rule_name,
        )
        spacing_km = spacing_km[np.isfinite(spacing_km)]
        if spacing_km.size > 0:
            spacing_ok = spacing_km

    # -----------------------------
    # 6) Quantiles + metric-based recommendation
    # -----------------------------
    footprint_quant = {q: float(np.quantile(footprint_ok, q)) for q in quantiles}
    beta_quant = {q: float(np.degrees(np.quantile(beta_ok, q))) for q in quantiles}

    footprint_based_km = float(np.quantile(footprint_ok, fp_q))

    spacing_based_km = None
    spacing_quant = None
    if spacing_ok is not None:
        spacing_quant = {q: float(np.quantile(spacing_ok, q)) for q in quantiles}
        spacing_based_km = float(np.quantile(spacing_ok, spacing_q))

    if leading_metric == "footprint_contour":
        recommended_cell_km = float(footprint_based_km)
        dominant = "footprint_contour"
    elif leading_metric == "spacing_contour":
        recommended_cell_km = float(spacing_based_km)  # type: ignore[arg-type]
        dominant = "spacing_contour"
    elif leading_metric == "max_of_both":
        recommended_cell_km = float(max(footprint_based_km, spacing_based_km))  # type: ignore[arg-type]
        dominant = (
            "spacing_contour"
            if (spacing_based_km is not None and spacing_based_km >= footprint_based_km)
            else "footprint_contour"
        )
    elif leading_metric == "min_of_both":
        recommended_cell_km = float(min(footprint_based_km, spacing_based_km))  # type: ignore[arg-type]
        dominant = (
            "spacing_contour"
            if (spacing_based_km is not None and spacing_based_km <= footprint_based_km)
            else "footprint_contour"
        )
    else:
        raise RuntimeError("Unexpected leading_metric")

    # Optional hard enforcement of grid spacing >= spacing-based spacing
    if (
        enforce_spacing_min
        and spacing_based_km is not None
        and recommended_cell_km < spacing_based_km
    ):
        recommended_cell_km = float(spacing_based_km)
        dominant = "spacing_contour (enforced)"

    # -----------------------------
    # 7) Diagnostics: packing feasibility + guardrails
    # -----------------------------
    footprint_median_km = float(np.quantile(footprint_ok, 0.5))
    ratio_cell_to_fp_med = float(recommended_cell_km / footprint_median_km)

    out: dict = {
        "strategy": strategy,
        "leading_metric": leading_metric,
        "dominant_metric": dominant,
        "enforce_spacing_min": bool(enforce_spacing_min),
        "n_samples_used": int(footprint_ok.size),

        # Geometry
        "gamma_max_deg": float(np.degrees(gamma_max)),
        "beta_max_deg": float(np.degrees(beta_max)),
        "alpha_limb_deg": float(np.degrees(alpha_limb)),

        # Antenna angles (half-angles)
        "footprint_contour_drop_db": float(footprint_drop_info["drop_db"]),
        "footprint_contour_label": str(footprint_drop_info["label"]),
        "footprint_theta_edge_deg": float(np.degrees(footprint_theta)),
        "spacing_contour_drop_db": (
            float(spacing_drop_info["drop_db"]) if spacing_drop_info is not None else None
        ),
        "spacing_contour_label": (
            str(spacing_drop_info["label"]) if spacing_drop_info is not None else None
        ),
        "spacing_theta_edge_deg": float(np.degrees(spacing_theta)) if spacing_theta is not None else None,
        "beam_spacing_rule": beam_spacing_rule_name,

        # Beta stats
        "beta_quantiles_deg": beta_quant,

        # Footprint stats
        "footprint_mean_km": float(np.mean(footprint_ok)),
        "footprint_quantiles_km": footprint_quant,
        "footprint_based_cell_km": float(footprint_based_km),
        "footprint_quantile_used": float(fp_q),

        # Separation stats (if available)
        "spacing_quantiles_km": spacing_quant,
        "spacing_based_cell_km": float(spacing_based_km) if spacing_based_km is not None else None,
        "spacing_quantile_used": float(spacing_q) if spacing_based_km is not None else None,

        # Final recommendation
        "cell_quantile_used": float(cell_quantile),
        "recommended_cell_spacing_km": float(recommended_cell_km),

        # Ratios useful for notebook sanity
        "cell_to_median_footprint_ratio": ratio_cell_to_fp_med,
    }

    if strategy == "maximum_elevation":
        out["n_vis_mean_est"] = float(n_vis_mean_est) if n_vis_mean_est is not None else None
        out["n_pool_sats"] = int(n_pool_sats) if n_pool_sats is not None else None
        out["vis_count_model"] = vis_count_model
        out["vis_count_scale"] = float(vis_count_scale)

    # Packing feasibility if spacing is defined
    if spacing_theta is not None:
        A_cap = 2.0 * np.pi * (1.0 - np.cos(beta_max))
        r = 0.5 * float(spacing_theta)
        A_disk = 2.0 * np.pi * (1.0 - np.cos(r)) if r > 0 else np.inf
        N_upper = (A_cap / A_disk) if A_disk > 0 else np.inf
        ETA_CONS = 0.75
        out["max_beams_upper_est"] = float(N_upper)
        out["max_beams_conservative_est"] = float(ETA_CONS * N_upper)

    # Guardrail warning policy:
    # - "warn": always warn when cell is too big relative to median -3 dB footprint
    # - "warn_if_footprint_leading": warn only when the recommendation is footprint-driven
    # - "off": never warn
    if footprint_guard_policy != "off" and footprint_guard_max_ratio is not None and footprint_guard_max_ratio > 0:
        should_warn = False
        if ratio_cell_to_fp_med > float(footprint_guard_max_ratio):
            if footprint_guard_policy == "warn":
                should_warn = True
            elif footprint_guard_policy == "warn_if_footprint_leading":
                # warn only if footprint is intended to be the driver
                if leading_metric == "footprint_contour":
                    should_warn = True
                # also warn if leading_metric="max_of_both" but footprint dominates
                if leading_metric == "max_of_both" and dominant.startswith("footprint_contour"):
                    should_warn = True

        if should_warn:
            warnings.warn(
                f"Recommended cell spacing ({recommended_cell_km:.2f} km) is > {footprint_guard_max_ratio:.2f}× "
                f"median {footprint_drop_info['label']} footprint ({footprint_median_km:.2f} km). "
                "This may introduce discretisation bias "
                "in link-budget-related statistics.",
                UserWarning,
            )

    # Optional samples
    if return_samples:
        out["beta_samples_deg"] = np.degrees(beta_ok)
        out["footprint_samples_km"] = footprint_ok
        if spacing_ok is not None:
            out["spacing_samples_km"] = spacing_ok

    return out


def _normalise_contour_sampling_strategy(strategy: str) -> tuple[str, str]:
    strategy_name = str(strategy).strip().lower()
    if strategy_name in {"random", "random_pointing"}:
        return "random_pointing", "Random pointing"
    if strategy_name in {"max_elevation", "maximum_elevation"}:
        return "maximum_elevation", "Maximum elevation"
    raise ValueError(
        "strategy must be one of {'random', 'random_pointing', "
        "'max_elevation', 'maximum_elevation'}."
    )


def _leading_metric_summary_label(
    leading_metric: str,
    *,
    footprint_label: str,
    spacing_label: str,
) -> str:
    if leading_metric == "footprint_contour":
        return f"{footprint_label} footprint contour"
    if leading_metric == "spacing_contour":
        return f"{spacing_label} ground spacing contour"
    if leading_metric == "max_of_both":
        return f"max({footprint_label} footprint, {spacing_label} spacing)"
    if leading_metric == "min_of_both":
        return f"min({footprint_label} footprint, {spacing_label} spacing)"
    return leading_metric


def summarize_contour_spacing(
    antenna_gain_func: Callable[..., Any],
    *,
    belt_names: Iterable[str],
    altitudes: u.Quantity | np.ndarray,
    min_elevations: u.Quantity | np.ndarray,
    max_betas: u.Quantity | np.ndarray,
    wavelength: u.Quantity,
    strategy: str,
    indicative_footprint_drop: str | float | u.Quantity = "db3",
    spacing_drop: str | float | u.Quantity = "db7",
    leading_metric: str = "spacing_contour",
    cell_spacing_rule: str = "full_footprint_diameter",
    belt_satellite_counts: Iterable[int] | np.ndarray | None = None,
    vis_count_model: str = "poisson",
    vis_count_scale: float = 1.0,
    cell_quantile: float = 0.5,
    n_samples: int = 200_000,
    seed: int = 0,
    footprint_guard_policy: str = "off",
    **antenna_pattern_kwargs: Any,
) -> dict[str, Any]:
    """
    Build contour diagnostics and the spacing recommendation summary for notebooks.

    Parameters
    ----------
    antenna_gain_func : callable
        One-dimensional antenna gain function passed to
        :func:`calculate_beamwidth_1d`, :func:`calculate_footprint_size`, and
        :func:`recommend_cell_diameter`.
    belt_names : iterable of str
        Belt labels in the same order as the per-belt geometry arrays.
    altitudes, min_elevations, max_betas : astropy.units.Quantity or np.ndarray
        Per-belt altitude, minimum elevation, and operational maximum beta
        values. Plain arrays are interpreted as km/deg/deg respectively.
    wavelength : astropy.units.Quantity
        Operating wavelength used for the antenna-pattern helpers.
    strategy : {"random", "random_pointing", "max_elevation", "maximum_elevation"}
        Visibility strategy used by :func:`recommend_cell_diameter`.
    indicative_footprint_drop, spacing_drop : {"db3", "db7", "db15"} or float or astropy.units.Quantity, optional
        Reference footprint contour and operational spacing contour.
    leading_metric : {"footprint_contour", "spacing_contour", "max_of_both", "min_of_both"}, optional
        Recommendation driver passed through to
        :func:`recommend_cell_diameter`.
    cell_spacing_rule : {"center_to_contour", "full_footprint_diameter"}, optional
        Ground-spacing convention used for the spacing contour.
    belt_satellite_counts : iterable of int or np.ndarray, optional
        Per-belt satellite counts. Required when ``strategy`` resolves to
        ``"maximum_elevation"``.
    vis_count_model, vis_count_scale, cell_quantile, n_samples, seed : optional
        Sampling controls forwarded to :func:`recommend_cell_diameter`.
    footprint_guard_policy : {"warn", "warn_if_footprint_leading", "off"}, optional
        Footprint guardrail policy forwarded to
        :func:`recommend_cell_diameter`.
    **antenna_pattern_kwargs
        Additional pattern arguments. ``wavelength`` is injected explicitly and
        therefore removed from this mapping before forwarding to
        :func:`recommend_cell_diameter`.

    Returns
    -------
    dict[str, Any]
        Dictionary containing per-belt contour angles, footprint diameters,
        recommendation statistics, the selected global cell spacing, and
        notebook-ready ``summary_lines``.

    Raises
    ------
    ValueError
        Raised when the per-belt arrays are inconsistent or when ``strategy``
        requires belt satellite counts that were not provided.
    """
    strategy_name, strategy_label = _normalise_contour_sampling_strategy(strategy)
    cell_spacing_rule_name = _normalise_beam_spacing_rule(cell_spacing_rule)
    indicative_drop_info = normalize_contour_drop(
        indicative_footprint_drop,
        name="indicative_footprint_drop",
    )
    spacing_drop_info = normalize_contour_drop(spacing_drop, name="spacing_drop")

    belt_name_list = [str(name) for name in belt_names]
    altitudes_q = u.Quantity(altitudes, copy=False)
    if altitudes_q.unit == u.dimensionless_unscaled:
        altitudes_q = altitudes_q * u.km
    else:
        altitudes_q = altitudes_q.to(u.km)
    min_elevations_q = u.Quantity(min_elevations, copy=False)
    if min_elevations_q.unit == u.dimensionless_unscaled:
        min_elevations_q = min_elevations_q * u.deg
    else:
        min_elevations_q = min_elevations_q.to(u.deg)
    max_betas_q = u.Quantity(max_betas, copy=False)
    if max_betas_q.unit == u.dimensionless_unscaled:
        max_betas_q = max_betas_q * u.deg
    else:
        max_betas_q = max_betas_q.to(u.deg)

    if not (
        len(belt_name_list) == altitudes_q.size == min_elevations_q.size == max_betas_q.size
    ):
        raise ValueError(
            "belt_names, altitudes, min_elevations, and max_betas must have the same length."
        )

    if strategy_name == "maximum_elevation":
        if belt_satellite_counts is None:
            raise ValueError(
                "belt_satellite_counts is required when strategy='maximum_elevation'."
            )
        belt_satellite_count_arr = np.asarray(list(belt_satellite_counts), dtype=np.int64)
        if belt_satellite_count_arr.size != len(belt_name_list):
            raise ValueError(
                "belt_satellite_counts must have the same length as belt_names."
            )
    else:
        belt_satellite_count_arr = np.zeros(len(belt_name_list), dtype=np.int64)

    pattern_kwargs = dict(antenna_pattern_kwargs)
    pattern_kwargs["wavelength"] = wavelength
    recommend_kwargs = dict(pattern_kwargs)
    recommend_kwargs.pop("wavelength", None)

    # Use 2D-aware beamwidth calculator (handles both 1-D S.1528 and 2-D M.2101)
    indicative_theta_edge = 0.5 * calculate_beamwidth_2d(
        antenna_gain_func,
        level_drop=indicative_drop_info["drop_quantity"],
        **pattern_kwargs,
    )
    spacing_theta_edge = 0.5 * calculate_beamwidth_2d(
        antenna_gain_func,
        level_drop=spacing_drop_info["drop_quantity"],
        **pattern_kwargs,
    )

    indicative_footprint_nadir: list[u.Quantity] = []
    indicative_footprint_edge: list[u.Quantity] = []
    spacing_footprint_nadir: list[u.Quantity] = []
    spacing_footprint_edge: list[u.Quantity] = []
    recommendation_stats: list[dict[str, Any]] = []
    cell_spacing_km_per_belt: list[float] = []

    for idx, _belt_name in enumerate(belt_name_list):
        altitude_q = altitudes_q[idx]
        beta_edge_q = max_betas_q[idx]
        indicative_footprint_nadir.append(
            calculate_footprint_size(
                antenna_gain_func,
                altitude=altitude_q,
                off_nadir_angle=0.0 * u.deg,
                theta=indicative_theta_edge,
                **pattern_kwargs,
            ).to(u.km)
        )
        indicative_footprint_edge.append(
            calculate_footprint_size(
                antenna_gain_func,
                altitude=altitude_q,
                off_nadir_angle=beta_edge_q,
                theta=indicative_theta_edge,
                **pattern_kwargs,
            ).to(u.km)
        )
        spacing_footprint_nadir.append(
            calculate_footprint_size(
                antenna_gain_func,
                altitude=altitude_q,
                off_nadir_angle=0.0 * u.deg,
                theta=spacing_theta_edge,
                **pattern_kwargs,
            ).to(u.km)
        )
        spacing_footprint_edge.append(
            calculate_footprint_size(
                antenna_gain_func,
                altitude=altitude_q,
                off_nadir_angle=beta_edge_q,
                theta=spacing_theta_edge,
                **pattern_kwargs,
            ).to(u.km)
        )

        stats = recommend_cell_diameter(
            antenna_gain_func,
            altitude=altitude_q,
            min_elevation=min_elevations_q[idx],
            wavelength=wavelength,
            strategy=strategy_name,
            n_pool_sats=(
                int(belt_satellite_count_arr[idx])
                if strategy_name == "maximum_elevation"
                else None
            ),
            vis_count_model=vis_count_model,
            vis_count_scale=vis_count_scale,
            footprint_drop=indicative_drop_info["drop_quantity"],
            footprint_theta_edge=indicative_theta_edge,
            spacing_drop=spacing_drop_info["drop_quantity"],
            spacing_theta_edge=spacing_theta_edge,
            beam_spacing_rule=cell_spacing_rule_name,
            leading_metric=leading_metric,
            cell_quantile=cell_quantile,
            footprint_guard_policy=footprint_guard_policy,
            n_samples=n_samples,
            seed=seed,
            **recommend_kwargs,
        )
        recommendation_stats.append(stats)
        cell_spacing_km_per_belt.append(float(stats["recommended_cell_spacing_km"]))

    indicative_footprint_nadir_q = u.Quantity(indicative_footprint_nadir)
    indicative_footprint_edge_q = u.Quantity(indicative_footprint_edge)
    spacing_footprint_nadir_q = u.Quantity(spacing_footprint_nadir)
    spacing_footprint_edge_q = u.Quantity(spacing_footprint_edge)
    selected_cell_spacing_km = float(np.max(cell_spacing_km_per_belt))

    summary_lines: list[str] = [
        "Antenna angular metrics (edge half-angles):",
        (
            f"  Indicative contour ({indicative_drop_info['label']}) = "
            f"{indicative_theta_edge.to_value(u.deg):.6f} deg"
        ),
        (
            f"  Spacing contour ({spacing_drop_info['label']})    = "
            f"{spacing_theta_edge.to_value(u.deg):.6f} deg"
        ),
        "",
        f"{indicative_drop_info['label']} footprint diameter on Earth:",
    ]
    for idx, belt_name in enumerate(belt_name_list):
        summary_lines.append(
            f"  {belt_name}: nadir={indicative_footprint_nadir_q[idx].to_value(u.km):.3f} km, "
            f"edge(β_max)={indicative_footprint_edge_q[idx].to_value(u.km):.3f} km"
        )
    summary_lines.append("")
    summary_lines.append(f"{spacing_drop_info['label']} footprint diameter on Earth:")
    for idx, belt_name in enumerate(belt_name_list):
        summary_lines.append(
            f"  {belt_name}: nadir={spacing_footprint_nadir_q[idx].to_value(u.km):.3f} km, "
            f"edge(β_max)={spacing_footprint_edge_q[idx].to_value(u.km):.3f} km"
        )
    summary_lines.append("")
    summary_lines.append(f"Cell size recommendation — {strategy_label} (pre-simulation):")
    summary_lines.append(
        "  leading_metric = "
        + _leading_metric_summary_label(
            leading_metric,
            footprint_label=indicative_drop_info["label"],
            spacing_label=spacing_drop_info["label"],
        )
    )
    summary_lines.append(f"  spacing_rule = {cell_spacing_rule_name}")
    if strategy_name == "maximum_elevation":
        summary_lines.append(
            "  visibility model = Poisson around mean N_vis derived from n_pool_sats and service cap"
        )
    summary_lines.append("")

    for idx, belt_name in enumerate(belt_name_list):
        stats = recommendation_stats[idx]
        beta_quantiles = stats["beta_quantiles_deg"]
        spacing_quantiles = stats.get("spacing_quantiles_km")
        spacing_geo_p50 = ground_separation_from_beta(
            float(beta_quantiles[0.5]) * u.deg,
            spacing_theta_edge,
            altitudes_q[idx],
            beam_spacing_rule=cell_spacing_rule_name,
        ).to_value(u.km)
        ref_label = (
            "diameter"
            if cell_spacing_rule_name == "full_footprint_diameter"
            else "radius"
        )
        ref_value = (
            spacing_footprint_nadir_q[idx].to_value(u.km)
            if cell_spacing_rule_name == "full_footprint_diameter"
            else 0.5 * spacing_footprint_nadir_q[idx].to_value(u.km)
        )
        summary_lines.append(f"  {belt_name}:")
        if strategy_name == "maximum_elevation" and stats.get("n_vis_mean_est") is not None:
            summary_lines.append(
                f"    N_vis_mean_est ≈ {float(stats['n_vis_mean_est']):.2f}  "
                f"(n_pool_sats={int(belt_satellite_count_arr[idx])})"
            )
        summary_lines.append(
            f"    Recommended cell spacing ({spacing_drop_info['label']}, p50; "
            f"rule={cell_spacing_rule_name}) = {float(stats['recommended_cell_spacing_km']):.3f} km"
        )
        summary_lines.append(
            "    β quantiles (deg): "
            f"p10={float(beta_quantiles[0.1]):.2f}, "
            f"p50={float(beta_quantiles[0.5]):.2f}, "
            f"p90={float(beta_quantiles[0.9]):.2f}, "
            f"p95={float(beta_quantiles[0.95]):.2f}"
        )
        if spacing_quantiles is not None:
            summary_lines.append(
                f"    {spacing_drop_info['label']} ground spacing quantiles (km): "
                f"p10={float(spacing_quantiles[0.1]):.2f}, "
                f"p50={float(spacing_quantiles[0.5]):.2f}, "
                f"p90={float(spacing_quantiles[0.9]):.2f}"
            )
        summary_lines.append(
            "    Footprint diameters (km): "
            f"{indicative_drop_info['label']} nadir={indicative_footprint_nadir_q[idx].to_value(u.km):.2f}, "
            f"edge={indicative_footprint_edge_q[idx].to_value(u.km):.2f} | "
            f"{spacing_drop_info['label']} nadir={spacing_footprint_nadir_q[idx].to_value(u.km):.2f}, "
            f"edge={spacing_footprint_edge_q[idx].to_value(u.km):.2f}"
        )
        summary_lines.append(
            f"    β_p50 (from selection model) = {float(beta_quantiles[0.5]):.2f} deg"
        )
        summary_lines.append(
            f"    spacing_{spacing_drop_info['label']}_p50 (from recommend) = "
            f"{float(stats['recommended_cell_spacing_km']):.2f} km"
        )
        summary_lines.append(
            f"    spacing_{spacing_drop_info['label']}_p50 (geometry check) = "
            f"{spacing_geo_p50:.2f} km  "
            f"(ratio={float(stats['recommended_cell_spacing_km'])/spacing_geo_p50:.3f})"
        )
        summary_lines.append(
            "    Nadir reference (do not expect equality at β>0): "
            f"{spacing_drop_info['label']} diameter={spacing_footprint_nadir_q[idx].to_value(u.km):.2f} km "
            f"-> {ref_label}={ref_value:.2f} km, "
            f"cell/{ref_label}_nadir={float(stats['recommended_cell_spacing_km'])/ref_value:.2f}"
        )
    summary_lines.append(f"Selected cell_km is {selected_cell_spacing_km:.3f}")

    return {
        "summary_lines": summary_lines,
        "strategy": strategy_name,
        "indicative_footprint_drop": indicative_drop_info,
        "spacing_drop": spacing_drop_info,
        "leading_metric": leading_metric,
        "cell_spacing_rule": cell_spacing_rule_name,
        "indicative_theta_edge": indicative_theta_edge,
        "spacing_theta_edge": spacing_theta_edge,
        "indicative_footprint_nadir": indicative_footprint_nadir_q,
        "indicative_footprint_edge": indicative_footprint_edge_q,
        "spacing_footprint_nadir": spacing_footprint_nadir_q,
        "spacing_footprint_edge": spacing_footprint_edge_q,
        "recommendation_stats": recommendation_stats,
        "cell_spacing_km_per_belt": np.asarray(cell_spacing_km_per_belt, dtype=np.float64),
        "selected_cell_spacing_km": selected_cell_spacing_km,
    }


def _resolve_station_service_cell_index(
    cell_longitudes: u.Quantity,
    cell_latitudes: u.Quantity,
    *,
    station_lon: u.Quantity | float,
    station_lat: u.Quantity | float,
) -> int:
    station_lon_deg = _coerce_scalar_angle_deg(station_lon, name="station_lon")
    station_lat_deg = _coerce_scalar_angle_deg(station_lat, name="station_lat")
    cell_lons_deg = u.Quantity(cell_longitudes, copy=False).to_value(u.deg)
    cell_lats_deg = u.Quantity(cell_latitudes, copy=False).to_value(u.deg)
    dlon_deg = (cell_lons_deg - station_lon_deg) * np.cos(np.deg2rad(station_lat_deg))
    dlat_deg = cell_lats_deg - station_lat_deg
    return int(np.argmin(dlon_deg * dlon_deg + dlat_deg * dlat_deg))


def union_orbital_parameters(
    constellations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge orbital parameters from multiple constellations for grid generation.

    Parameters
    ----------
    constellations : list[dict]
        Each dict must contain ``"altitudes_q"``, ``"min_elevations_q"``,
        and ``"inclinations_q"`` (astropy Quantity arrays or NumPy arrays).

    Returns
    -------
    dict[str, Any]
        Merged ``altitudes_q``, ``min_elevations_q``, ``inclinations_q``
        covering all systems' visibility cones.
    """
    if not constellations:
        raise ValueError("At least one constellation is required.")
    if len(constellations) == 1:
        return constellations[0]
    all_alt = []
    all_min_el = []
    all_inc = []
    for c in constellations:
        alt = c["altitudes_q"]
        mel = c["min_elevations_q"]
        inc = c["inclinations_q"]
        if hasattr(alt, "to_value"):
            alt = alt.to_value(u.km)
        if hasattr(mel, "to_value"):
            mel = mel.to_value(u.deg)
        if hasattr(inc, "to_value"):
            inc = inc.to_value(u.deg)
        all_alt.append(np.atleast_1d(np.asarray(alt, dtype=np.float64)))
        all_min_el.append(np.atleast_1d(np.asarray(mel, dtype=np.float64)))
        all_inc.append(np.atleast_1d(np.asarray(inc, dtype=np.float64)))
    return {
        "altitudes_q": np.concatenate(all_alt) * u.km,
        "min_elevations_q": np.concatenate(all_min_el) * u.deg,
        "inclinations_q": np.concatenate(all_inc) * u.deg,
    }


def prepare_active_grid(
    *,
    point_spacing: u.Quantity | float,
    altitudes: u.Quantity | np.ndarray,
    min_elevations: u.Quantity | np.ndarray,
    inclinations: u.Quantity | np.ndarray,
    station_lat: u.Quantity | float,
    station_lon: u.Quantity | float,
    station_height: u.Quantity | None = None,
    latitude_policy: str = "any",
    geography_mask_mode: str = "none",
    shoreline_buffer_km: float | None = None,
    coastline_backend: str = "vendored",
    ras_exclusion_mode: str = "none",
    ras_exclusion_layers: int | None = None,
    ras_exclusion_radius_km: float | None = None,
    ras_pointing_mode: str | None = None,
) -> dict[str, Any]:
    """
    Generate and prepare the notebook-facing Earth grid.

    Parameters
    ----------
    point_spacing : astropy.units.Quantity or float
        Requested global hex-grid spacing. Plain numeric values are interpreted
        as kilometres.
    altitudes, min_elevations, inclinations : astropy.units.Quantity or np.ndarray
        Per-belt geometry arrays passed to
        :func:`mask_hexgrid_for_constellation`.
    station_lat, station_lon : astropy.units.Quantity or float
        RAS-station geodetic coordinates. Plain scalars are interpreted as
        degrees.
    station_height : astropy.units.Quantity or None, optional
        Station height above mean sea level.
    latitude_policy : {"any", "all", "first"}, optional
        Per-belt latitude-feasibility policy.
    geography_mask_mode : {"none", "land_only", "land_plus_nearshore_sea"}, optional
        Optional geography filter applied after the impactful+latitude mask.
        For ``"land_plus_nearshore_sea"``, a zero shoreline buffer behaves
        like ``"land_only"``, positive values keep near-shore sea cells, and
        negative values erode the land mask inland.
    shoreline_buffer_km : float or None, optional
        Signed shoreline buffer used when
        ``geography_mask_mode="land_plus_nearshore_sea"``.
    coastline_backend : {"vendored", "cartopy"}, optional
        Coastline source passed through to
        :func:`mask_hexgrid_for_constellation`.
    ras_exclusion_mode : {"none", "adjacency_layers", "radius_km"}, optional
        RAS-centric cell exclusion mode applied after geography pruning.
    ras_exclusion_layers : int or None, optional
        Number of adjacency layers included when
        ``ras_exclusion_mode="adjacency_layers"``.
    ras_exclusion_radius_km : float or None, optional
        Exclusion radius used when ``ras_exclusion_mode="radius_km"``.
    ras_pointing_mode : {"ras_station", "cell_center"} or None, optional
        Optional report-only label included in ``summary_lines``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the generated full grid, the PREFILTER/PRE_RAS/
        ACTIVE axes, index mappings between them, RAS service-cell indices, and
        notebook-ready ``summary_lines``.
    """
    point_spacing_q = u.Quantity(point_spacing, copy=False)
    if point_spacing_q.unit == u.dimensionless_unscaled:
        point_spacing_q = point_spacing_q * u.km
    else:
        point_spacing_q = point_spacing_q.to(u.km)

    grid_lons, grid_lats, _ = generate_hexgrid_full(point_spacing_q)
    if not hasattr(grid_lons, "unit"):
        grid_lons = np.asarray(grid_lons, dtype=np.float64) * u.deg
    if not hasattr(grid_lats, "unit"):
        grid_lats = np.asarray(grid_lats, dtype=np.float64) * u.deg

    mask_info = mask_hexgrid_for_constellation(
        grid_lons,
        grid_lats,
        altitudes=altitudes,
        min_elevations=min_elevations,
        inclinations=inclinations,
        station_lat=u.Quantity(station_lat),
        station_lon=u.Quantity(station_lon),
        station_height=station_height,
        latitude_policy=latitude_policy,
        geography_mask_mode=geography_mask_mode,
        shoreline_buffer_km=shoreline_buffer_km,
        coastline_backend=coastline_backend,
    )

    base_mask = np.asarray(mask_info["base_mask"], dtype=bool)
    lat_mask = np.asarray(mask_info["lat_mask"], dtype=bool)
    geography_mask = np.asarray(mask_info["geography_mask"], dtype=bool)
    land_mask = np.asarray(mask_info["land_mask"], dtype=bool)
    nearshore_sea_mask = np.asarray(mask_info["nearshore_sea_mask"], dtype=bool)
    shore_distance_km = np.asarray(mask_info["shore_distance_km"], dtype=np.float64)
    prefilter_mask = base_mask & lat_mask

    prefilter_cell_longitudes = grid_lons[prefilter_mask]
    prefilter_cell_latitudes = grid_lats[prefilter_mask]
    if prefilter_cell_longitudes.size < 1:
        raise RuntimeError(
            "No impactful EarthGrid cells remain after the base and latitude masks."
        )

    geography_keep_mask_prefilter = geography_mask[prefilter_mask]
    land_mask_prefilter = land_mask[prefilter_mask]
    nearshore_sea_mask_prefilter = nearshore_sea_mask[prefilter_mask]
    shore_distance_km_prefilter = shore_distance_km[prefilter_mask]

    pre_ras_cell_longitudes = prefilter_cell_longitudes[geography_keep_mask_prefilter]
    pre_ras_cell_latitudes = prefilter_cell_latitudes[geography_keep_mask_prefilter]
    if pre_ras_cell_longitudes.size < 1:
        raise RuntimeError("No EarthGrid cells remain after the geography mask.")

    ras_service_cell_index_prefilter = _resolve_station_service_cell_index(
        prefilter_cell_longitudes,
        prefilter_cell_latitudes,
        station_lon=station_lon,
        station_lat=station_lat,
    )
    ras_exclusion_ids_prefilter = resolve_ras_hexgrid_cell_ids(
        prefilter_cell_longitudes,
        prefilter_cell_latitudes,
        station_lat=station_lat,
        station_lon=station_lon,
        mode=ras_exclusion_mode,
        ras_cell_index=ras_service_cell_index_prefilter,
        layers=ras_exclusion_layers,
        radius_km=ras_exclusion_radius_km,
    )

    prefilter_cell_count = int(prefilter_cell_longitudes.size)
    geography_kept_cell_count = int(np.count_nonzero(geography_keep_mask_prefilter))
    geography_excluded_cell_count = int(prefilter_cell_count - geography_kept_cell_count)
    pre_ras_cell_count = int(pre_ras_cell_longitudes.size)
    land_prefilter_cell_count = (
        int(np.count_nonzero(land_mask_prefilter))
        if str(geography_mask_mode).strip().lower() != "none"
        else 0
    )
    nearshore_sea_prefilter_cell_count = (
        int(np.count_nonzero(nearshore_sea_mask_prefilter))
        if str(geography_mask_mode).strip().lower() != "none"
        else 0
    )

    prefilter_to_pre_ras = np.full(prefilter_cell_count, -1, dtype=np.int32)
    prefilter_to_pre_ras[geography_keep_mask_prefilter] = np.arange(
        pre_ras_cell_count,
        dtype=np.int32,
    )
    ras_exclusion_ids_pre_ras = prefilter_to_pre_ras[ras_exclusion_ids_prefilter]
    ras_exclusion_ids_pre_ras = ras_exclusion_ids_pre_ras[ras_exclusion_ids_pre_ras >= 0]
    ras_exclusion_ids_pre_ras = np.unique(
        ras_exclusion_ids_pre_ras.astype(np.int32, copy=False)
    )

    ras_exclusion_mask_pre_ras = np.zeros(pre_ras_cell_count, dtype=bool)
    ras_exclusion_mask_pre_ras[ras_exclusion_ids_pre_ras] = True
    active_grid_keep_mask_pre_ras = ~ras_exclusion_mask_pre_ras
    if not np.any(active_grid_keep_mask_pre_ras):
        raise RuntimeError(
            "RAS hexgrid exclusion removed all geography-kept EarthGrid cells."
        )

    active_cell_count = int(np.count_nonzero(active_grid_keep_mask_pre_ras))
    pre_ras_to_active = np.full(pre_ras_cell_count, -1, dtype=np.int32)
    pre_ras_to_active[active_grid_keep_mask_pre_ras] = np.arange(
        active_cell_count,
        dtype=np.int32,
    )

    active_grid_longitudes = pre_ras_cell_longitudes[active_grid_keep_mask_pre_ras]
    active_grid_latitudes = pre_ras_cell_latitudes[active_grid_keep_mask_pre_ras]
    ras_service_cell_index_pre_ras = int(prefilter_to_pre_ras[ras_service_cell_index_prefilter])
    ras_service_cell_index = (
        -1
        if ras_service_cell_index_pre_ras < 0
        else int(pre_ras_to_active[ras_service_cell_index_pre_ras])
    )
    ras_service_cell_active = bool(ras_service_cell_index >= 0)
    ras_hex_exclusion_requested_cell_count = int(ras_exclusion_ids_prefilter.size)
    ras_excluded_cell_count = int(np.count_nonzero(ras_exclusion_mask_pre_ras))

    # --- Compute actual median nearest-neighbour spacing from the generated grid ---
    station_lon_deg = float(u.Quantity(station_lon).to_value(u.deg))
    station_lat_deg = float(u.Quantity(station_lat).to_value(u.deg))
    actual_point_spacing_km = _estimate_local_hexgrid_spacing_km(
        np.asarray(pre_ras_cell_longitudes.to_value(u.deg), dtype=np.float64),
        np.asarray(pre_ras_cell_latitudes.to_value(u.deg), dtype=np.float64),
        ref_lon_deg=station_lon_deg,
        ref_lat_deg=station_lat_deg,
    )

    summary_lines = [
        f"Grid spacing (requested)                           = {point_spacing_q.to_value(u.km):.3f} km",
        f"Grid spacing (actual mean-of-6-NN)                 = {actual_point_spacing_km:.3f} km",
        f"Total global cells                                 = {grid_lons.size}",
        f"Impactful cells (base mask)                        = {int(np.count_nonzero(base_mask))}",
        f"Reachable cells (latitude geometry)                = {int(np.count_nonzero(lat_mask))}",
        f"Impactful cells (before geography/RAS exclusion)   = {prefilter_cell_count}",
        f"Impactful cells (after geography exclusion)        = {pre_ras_cell_count}",
        f"Impactful cells (after RAS exclusion)              = {active_cell_count}",
        f"Latitude reachability policy                       = {str(latitude_policy).strip().lower()}",
        f"Latitude limit |φ| (deg)                           = {u.Quantity(mask_info['phi_limit']).to_value(u.deg):.2f}",
        f"Mask belts                                         = {u.Quantity(mask_info['phi_max_per_belt']).size}",
    ]
    geography_mode_name = str(mask_info["geography_mask_mode"])
    if geography_mode_name != "none":
        summary_lines.extend(
            [
                f"Geography mask mode                                = {geography_mode_name!r}",
                f"Coastline backend                                  = {str(mask_info['coastline_backend'])!r}",
                f"Geography-excluded cells                           = {geography_excluded_cell_count}",
                f"Land cells in prefilter                            = {land_prefilter_cell_count}",
                f"Nearshore sea cells in prefilter                   = {nearshore_sea_prefilter_cell_count}",
            ]
        )
        if shoreline_buffer_km is not None:
            summary_lines.append(
                f"Shoreline buffer [km]                              = {float(shoreline_buffer_km):.3f}"
            )
    if ras_pointing_mode is not None:
        summary_lines.append(
            f"RAS pointing mode                                  = {str(ras_pointing_mode)!r}"
        )
    summary_lines.append(
        f"RAS exclusion mode                                 = {str(ras_exclusion_mode)!r}"
    )
    summary_lines.append(
        f"RAS excluded cells                                 = {ras_excluded_cell_count}"
    )
    summary_lines.append(
        f"RAS service cell active after exclusion            = {ras_service_cell_active}"
    )

    return {
        "summary_lines": summary_lines,
        "point_spacing_km": float(point_spacing_q.to_value(u.km)),
        "actual_point_spacing_km": float(actual_point_spacing_km),
        "grid_longitudes": grid_lons,
        "grid_latitudes": grid_lats,
        "base_mask": base_mask,
        "lat_mask": lat_mask,
        "mask_info": mask_info,
        "prefilter_mask": prefilter_mask,
        "prefilter_cell_longitudes": prefilter_cell_longitudes,
        "prefilter_cell_latitudes": prefilter_cell_latitudes,
        "geography_keep_mask_prefilter": geography_keep_mask_prefilter,
        "land_mask_prefilter": land_mask_prefilter,
        "nearshore_sea_mask_prefilter": nearshore_sea_mask_prefilter,
        "shore_distance_km_prefilter": shore_distance_km_prefilter,
        "pre_ras_cell_longitudes": pre_ras_cell_longitudes,
        "pre_ras_cell_latitudes": pre_ras_cell_latitudes,
        "prefilter_to_pre_ras": prefilter_to_pre_ras,
        "ras_exclusion_ids_prefilter": ras_exclusion_ids_prefilter,
        "ras_exclusion_ids_pre_ras": ras_exclusion_ids_pre_ras,
        "ras_exclusion_mask_pre_ras": ras_exclusion_mask_pre_ras,
        "active_grid_keep_mask_pre_ras": active_grid_keep_mask_pre_ras,
        "pre_ras_to_active": pre_ras_to_active,
        "active_grid_longitudes": active_grid_longitudes,
        "active_grid_latitudes": active_grid_latitudes,
        "prefilter_cell_count": prefilter_cell_count,
        "pre_ras_cell_count": pre_ras_cell_count,
        "active_cell_count": active_cell_count,
        "geography_kept_cell_count": geography_kept_cell_count,
        "geography_excluded_cell_count": geography_excluded_cell_count,
        "land_prefilter_cell_count": land_prefilter_cell_count,
        "nearshore_sea_prefilter_cell_count": nearshore_sea_prefilter_cell_count,
        "ras_service_cell_index_prefilter": ras_service_cell_index_prefilter,
        "ras_service_cell_index_pre_ras": ras_service_cell_index_pre_ras,
        "ras_service_cell_index": ras_service_cell_index,
        "ras_service_cell_active": ras_service_cell_active,
        "ras_hex_exclusion_requested_cell_count": ras_hex_exclusion_requested_cell_count,
        "ras_excluded_cell_count": ras_excluded_cell_count,
        "station_lat": u.Quantity(station_lat),
        "station_lon": u.Quantity(station_lon),
        "latitude_policy": str(latitude_policy).strip().lower(),
        "geography_mask_mode": geography_mode_name,
        "coastline_backend": str(mask_info["coastline_backend"]),
        "ras_exclusion_mode": str(ras_exclusion_mode).strip().lower(),
        "ras_exclusion_layers": (
            None if ras_exclusion_layers is None else int(ras_exclusion_layers)
        ),
        "ras_exclusion_radius_km": (
            None if ras_exclusion_radius_km is None else float(ras_exclusion_radius_km)
        ),
    }


def _normalise_theta2_scope_mode(scope_mode: str | None) -> str:
    mode_name = str(scope_mode or "cell_ids").strip().lower()
    if mode_name == "layers":
        mode_name = "adjacency_layers"
    elif mode_name == "radius":
        mode_name = "radius_km"
    if mode_name not in {"cell_ids", "adjacency_layers", "radius_km"}:
        raise ValueError(
            "scope_mode must be one of {'cell_ids', 'adjacency_layers', 'radius_km'}."
        )
    return mode_name


def resolve_theta2_active_cell_ids(
    prepared_grid: Mapping[str, Any],
    *,
    scope_mode: str,
    explicit_ids: Iterable[int] | np.ndarray | None = None,
    layers: int | None = None,
    radius_km: float | None = None,
) -> np.ndarray:
    """
    Resolve Theta_2 boresight-affected Earth-grid cell ids on the ACTIVE axis.

    Parameters
    ----------
    prepared_grid : Mapping[str, Any]
        Prepared-grid result returned by :func:`prepare_active_grid`.
    scope_mode : {"cell_ids", "adjacency_layers", "radius_km"}
        Theta_2 resolution mode. ``"cell_ids"`` validates the explicit ids on
        the ACTIVE axis. The other modes resolve a PREFILTER-axis selection via
        :func:`resolve_ras_hexgrid_cell_ids` and then project it through
        ``PREFILTER -> PRE_RAS -> ACTIVE``.
    explicit_ids : iterable of int or np.ndarray, optional
        Explicit ACTIVE-axis ids used only when ``scope_mode="cell_ids"``.
    layers : int or None, optional
        Adjacency ring depth used when ``scope_mode="adjacency_layers"``.
    radius_km : float or None, optional
        Great-circle radius used when ``scope_mode="radius_km"``.

    Returns
    -------
    np.ndarray
        Sorted ``int32`` array of ACTIVE-axis cell ids. The array may be empty
        when geography masking or RAS exclusion removes the resolved PREFILTER
        scope before the ACTIVE projection stage.
    """
    scope_mode_name = _normalise_theta2_scope_mode(scope_mode)
    active_cell_count = int(prepared_grid["active_cell_count"])

    if scope_mode_name == "cell_ids":
        if explicit_ids is None:
            raise ValueError(
                "explicit_ids must be provided when scope_mode='cell_ids'."
            )
        resolved = np.unique(np.asarray(list(explicit_ids), dtype=np.int32).reshape(-1))
        if np.any((resolved < 0) | (resolved >= active_cell_count)):
            raise ValueError("explicit_ids must reference ACTIVE-axis cell ids.")
    else:
        resolved_prefilter = resolve_ras_hexgrid_cell_ids(
            prepared_grid["prefilter_cell_longitudes"],
            prepared_grid["prefilter_cell_latitudes"],
            station_lat=prepared_grid["station_lat"],
            station_lon=prepared_grid["station_lon"],
            mode=scope_mode_name,
            ras_cell_index=(
                int(prepared_grid["ras_service_cell_index_prefilter"])
                if scope_mode_name == "adjacency_layers"
                else None
            ),
            layers=layers if scope_mode_name == "adjacency_layers" else None,
            radius_km=radius_km if scope_mode_name == "radius_km" else None,
        )
        resolved_pre_ras = np.asarray(prepared_grid["prefilter_to_pre_ras"], dtype=np.int32)[
            resolved_prefilter
        ]
        resolved_pre_ras = resolved_pre_ras[resolved_pre_ras >= 0]
        resolved = np.asarray(prepared_grid["pre_ras_to_active"], dtype=np.int32)[
            resolved_pre_ras
        ]
        resolved = resolved[resolved >= 0]
        resolved = np.unique(resolved.astype(np.int32, copy=False))

    return resolved.astype(np.int32, copy=False)
