"""
earthgrid.py

This module provides functions related to gridding and geometric calculations 
on the Earth's surface, particularly relevant for satellite communications 
and spectrum management.

It includes routines for calculating the -3 dB antenna footprint on Earth as well
as functions to generate a full hexagon grid on the globe.

Numba is used as an optional accelerator for computational hotspots when
available. When numba is not installed, the pure-Python implementations are used
with identical numerical behavior.

Author: boris.sorokin <mralin@protonmail.com>
Hexgrid functions are partially based on code developed by Benjamin Winkel, MPIfR
Date: 01-04-2025
"""

import warnings
from functools import lru_cache

import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from pycraf import conversions as cnv
from pycraf.utils import ranged_quantity_input
from pycraf import geometry, pathprof
from scepter.antenna import calculate_beamwidth_1d

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

def recommend_cell_diameter(
    antenna_gain_func: callable,
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
    # --- footprint contour (-3 dB by default) ---
    level_drop: float | u.Quantity = 3.0 * cnv.dB,
    theta_edge: u.Quantity | None = None,  # HALF-angle; if None computed from pattern
    # --- beam separation diagnostic (-15 dB by default) ---
    beam_sep_drop: float | u.Quantity | None = 15.0 * cnv.dB,
    theta_sep_edge: u.Quantity | None = None,  # HALF-angle; if None computed from pattern
    # --- recommendation logic ---
    leading_metric: str = "footprint_3db",  # "footprint_3db" | "sep_15db" | "max_of_both" | "min_of_both"
    cell_quantile: float = 0.5,
    sep_quantile: float | None = None,
    footprint_quantile: float | None = None,
    enforce_sep_min: bool = False,
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
    Recommend a characteristic Earth cell diameter for gridding served locations.

    Two distributions are computed:
      (A) Footprint diameter distribution at `level_drop` (default: -3 dB) in the scan plane.
      (B) Ground beam-center separation distribution corresponding to `beam_sep_drop`
          (default: -15 dB), Interpretation A.

    The final recommendation can be driven by either metric via `leading_metric`.

    NOTE ABOUT -15 dB:
      A very large ground separation at -15 dB does NOT automatically mean your *grid cell*
      must be that large. You can keep a finer grid (e.g., driven by -3 dB) and enforce the
      -15 dB constraint later when selecting multiple beams per satellite. Therefore, by
      default this function keeps -15 dB as a diagnostic unless explicitly chosen as leading.
    """

    # -----------------------------
    # 0) Validate inputs
    # -----------------------------
    if strategy not in ("random_pointing", "maximum_elevation"):
        raise ValueError("strategy must be 'random_pointing' or 'maximum_elevation'")

    if vis_count_model not in ("mean", "poisson"):
        raise ValueError("vis_count_model must be 'mean' or 'poisson'")

    allowed_metrics = ("footprint_3db", "sep_15db", "max_of_both", "min_of_both")
    if leading_metric not in allowed_metrics:
        raise ValueError(f"leading_metric must be one of {allowed_metrics}")

    allowed_guard = ("warn", "warn_if_footprint_leading", "off")
    if footprint_guard_policy not in allowed_guard:
        raise ValueError(f"footprint_guard_policy must be one of {allowed_guard}")

    def _check_q(q: float, name: str) -> float:
        if not (0.0 < q < 1.0):
            raise ValueError(f"{name} must be in (0, 1)")
        return float(q)

    cell_quantile = _check_q(cell_quantile, "cell_quantile")
    sep_q = _check_q(sep_quantile if sep_quantile is not None else cell_quantile, "sep_quantile")
    fp_q = _check_q(footprint_quantile if footprint_quantile is not None else cell_quantile, "footprint_quantile")

    if not callable(antenna_gain_func):
        raise TypeError("antenna_gain_func must be callable")

    def _db_value(x) -> float:
        try:
            return float(x.to_value(cnv.dB))  # Quantity
        except Exception:
            return float(x)  # plain float

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
    # 1) Determine contour half-angles (theta_edge, theta_sep_edge)
    # -----------------------------
    if theta_edge is None:
        bw_full = calculate_beamwidth_1d(
            antenna_gain_func,
            level_drop=level_drop,
            **antenna_pattern_kwargs,
        )
        theta = 0.5 * float(bw_full.to_value(u.rad))
    else:
        theta = float(theta_edge.to_value(u.rad))
    theta = abs(theta)

    theta_sep = None
    if beam_sep_drop is not None:
        if theta_sep_edge is None:
            bw_sep_full = calculate_beamwidth_1d(
                antenna_gain_func,
                level_drop=beam_sep_drop,
                **antenna_pattern_kwargs,
            )
            theta_sep = 0.5 * float(bw_sep_full.to_value(u.rad))
        else:
            theta_sep = float(theta_sep_edge.to_value(u.rad))
        theta_sep = abs(theta_sep)

    if leading_metric in ("sep_15db", "max_of_both", "min_of_both") and theta_sep is None:
        raise ValueError("leading_metric requires separation but beam_sep_drop/theta_sep_edge was not provided.")

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
    # 4) Footprint diameters at level_drop
    # -----------------------------
    a1 = np.clip(beta - theta, -alpha_limb, +alpha_limb)
    a2 = np.clip(beta + theta, -alpha_limb, +alpha_limb)

    s1 = _signed_s(a1)
    s2 = _signed_s(a2)

    D_km = R_km * np.abs(s2 - s1)
    msk = np.isfinite(D_km)
    D_ok = D_km[msk]
    beta_ok = beta[msk]

    if D_ok.size == 0:
        raise RuntimeError("All sampled footprints are invalid. Check inputs / antenna parameters.")

    # -----------------------------
    # 5) Beam-center separations corresponding to theta_sep (diagnostic)
    # -----------------------------
    sep_ok = None
    if theta_sep is not None:
        a_ref = np.clip(beta_ok, -alpha_limb, +alpha_limb)
        a_next = np.clip(beta_ok + theta_sep, -alpha_limb, +alpha_limb)

        s_ref = _signed_s(a_ref)
        s_next = _signed_s(a_next)

        sep_km = R_km * np.abs(s_next - s_ref)
        sep_km = sep_km[np.isfinite(sep_km)]
        if sep_km.size > 0:
            sep_ok = sep_km

    # -----------------------------
    # 6) Quantiles + metric-based recommendation
    # -----------------------------
    D_quant = {q: float(np.quantile(D_ok, q)) for q in quantiles}
    beta_quant = {q: float(np.degrees(np.quantile(beta_ok, q))) for q in quantiles}

    footprint_based_km = float(np.quantile(D_ok, fp_q))

    sep_based_km = None
    sep_quant = None
    if sep_ok is not None:
        sep_quant = {q: float(np.quantile(sep_ok, q)) for q in quantiles}
        sep_based_km = float(np.quantile(sep_ok, sep_q))

    if leading_metric == "footprint_3db":
        cell_km = float(footprint_based_km)
        dominant = "footprint_3db"
    elif leading_metric == "sep_15db":
        cell_km = float(sep_based_km)  # type: ignore[arg-type]
        dominant = "sep_15db"
    elif leading_metric == "max_of_both":
        cell_km = float(max(footprint_based_km, sep_based_km))  # type: ignore[arg-type]
        dominant = "sep_15db" if (sep_based_km is not None and sep_based_km >= footprint_based_km) else "footprint_3db"
    elif leading_metric == "min_of_both":
        cell_km = float(min(footprint_based_km, sep_based_km))  # type: ignore[arg-type]
        dominant = "sep_15db" if (sep_based_km is not None and sep_based_km <= footprint_based_km) else "footprint_3db"
    else:
        raise RuntimeError("Unexpected leading_metric")

    # Optional hard enforcement of grid spacing >= separation-based spacing
    if enforce_sep_min and sep_based_km is not None and cell_km < sep_based_km:
        cell_km = float(sep_based_km)
        dominant = "sep_15db (enforced)"

    # -----------------------------
    # 7) Diagnostics: packing feasibility + guardrails
    # -----------------------------
    D_med = float(np.quantile(D_ok, 0.5))
    ratio_cell_to_fp_med = float(cell_km / D_med)

    out: dict = {
        "strategy": strategy,
        "leading_metric": leading_metric,
        "dominant_metric": dominant,
        "enforce_sep_min": bool(enforce_sep_min),
        "n_samples_used": int(D_ok.size),

        # Geometry
        "gamma_max_deg": float(np.degrees(gamma_max)),
        "beta_max_deg": float(np.degrees(beta_max)),
        "alpha_limb_deg": float(np.degrees(alpha_limb)),

        # Antenna angles (half-angles)
        "level_drop_db": _db_value(level_drop),
        "theta_edge_deg": float(np.degrees(theta)),
        "beam_sep_drop_db": _db_value(beam_sep_drop) if beam_sep_drop is not None else None,
        "theta_sep_edge_deg": float(np.degrees(theta_sep)) if theta_sep is not None else None,

        # Beta stats
        "beta_quantiles_deg": beta_quant,

        # Footprint stats
        "diameter_mean_km": float(np.mean(D_ok)),
        "diameter_quantiles_km": D_quant,
        "footprint_based_cell_km": float(footprint_based_km),
        "footprint_quantile_used": float(fp_q),

        # Separation stats (if available)
        "beam_center_sep_quantiles_km": sep_quant,
        "sep_based_cell_km": float(sep_based_km) if sep_based_km is not None else None,
        "sep_quantile_used": float(sep_q) if sep_based_km is not None else None,

        # Final recommendation
        "cell_quantile_used": float(cell_quantile),
        "recommended_cell_diameter_km": float(cell_km),

        # Ratios useful for notebook sanity
        "cell_to_median_footprint_ratio": ratio_cell_to_fp_med,
    }

    if strategy == "maximum_elevation":
        out["n_vis_mean_est"] = float(n_vis_mean_est) if n_vis_mean_est is not None else None
        out["n_pool_sats"] = int(n_pool_sats) if n_pool_sats is not None else None
        out["vis_count_model"] = vis_count_model
        out["vis_count_scale"] = float(vis_count_scale)

    # Packing feasibility if separation is defined
    if theta_sep is not None:
        A_cap = 2.0 * np.pi * (1.0 - np.cos(beta_max))
        r = 0.5 * float(theta_sep)
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
                if leading_metric == "footprint_3db":
                    should_warn = True
                # also warn if leading_metric="max_of_both" but footprint dominates
                if leading_metric == "max_of_both" and dominant.startswith("footprint_3db"):
                    should_warn = True

        if should_warn:
            warnings.warn(
                f"Recommended cell ({cell_km:.2f} km) is > {footprint_guard_max_ratio:.2f}× "
                f"median -3 dB footprint ({D_med:.2f} km). This may introduce discretisation bias "
                "in link-budget-related statistics.",
                UserWarning,
            )

    # Optional samples
    if return_samples:
        out["beta_samples_deg"] = np.degrees(beta_ok)
        out["diameter_samples_km"] = D_ok
        if sep_ok is not None:
            out["sep_samples_km"] = sep_ok

    return out


