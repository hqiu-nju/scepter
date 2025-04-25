"""
earthgrid.py

This module provides functions related to gridding and geometric calculations 
on the Earth's surface, particularly relevant for satellite communications 
and spectrum management.

It includes routines for calculating the -3 dB antenna footprint on Earth as well 
as functions to generate a full hexagon grid on the globe.

Author: boris.sorokin <mralin@protonmail.com>
Hexgrid functions are partially based on code developed by Benjamin Winkel, MPIfR
Date: 01-04-2025
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from pycraf import conversions as cnv
from pycraf.utils import ranged_quantity_input
from pycraf import geometry, pathprof
from scepter.antenna import calculate_3dB_angle_1d
from functools import lru_cache


# -----------------------------------------------------------------------------
# MAIN FUNCTIONS: CALCULATE FOOTPRINT SIZE
# -----------------------------------------------------------------------------
@ranged_quantity_input(
    altitude=(0.001, None, u.km),
    off_nadir_angle=(None, None, u.deg),
    strip_input_units=False,
    allow_none=True
)
def calculate_footprint_size(
    antenna_gain_func: callable,
    altitude: u.Quantity,
    off_nadir_angle: u.Quantity = 0 * u.deg,
    earth_model: str = 'spherical',
    **antenna_pattern_kwargs
) -> u.Quantity:
    """
    Computes the physical diameter of the -3 dB antenna footprint on Earth's surface.
    
    The function calculates the beamwidth (using the provided antenna_gain_func) 
    and determines the geometric footprint on Earth based on the satellite altitude 
    and antenna boresight (nadir or off-nadir). If the full beam does not intersect 
    Earth, the footprint is computed up to the Earth's limb.
    
    Parameters
    ----------
    antenna_gain_func : callable
        Function that returns the antenna gain as a function of angle. Used to 
        determine the -3 dB beamwidth.
    altitude : astropy.units.Quantity
        Satellite altitude above Earth's surface.
    off_nadir_angle : astropy.units.Quantity, optional
        Off-nadir angle of the antenna boresight. Defaults to 0° (nadir).
    earth_model : str, optional
        Earth model to use; currently only 'spherical' is supported.
    **antenna_pattern_kwargs : dict
        Additional keyword arguments passed to antenna_gain_func.
        
    Returns
    -------
    footprint_diameter : astropy.units.Quantity
        Calculated diameter (in meters) of the -3 dB footprint on Earth's surface.
        Returns np.nan * u.m if the beam (or its partial edge) does not intersect Earth.
    """
    # -----------------------------------------------------------------------------
    # HELPER FUNCTIONS: CACHED CENTRAL ANGLE CALCULATION
    # -----------------------------------------------------------------------------

    def _calculate_spherical_central_angle(R_eff, R_sat, alpha_from_nadir):
        """
        Calculates the central angle (SOE) between the satellite, Earth's center, 
        and the edge of the antenna beam for a given ray angle from nadir.
        
        Parameters
        ----------
        R_eff : astropy.units.Quantity
            Effective Earth radius.
        R_sat : astropy.units.Quantity
            Distance from Earth's center to the satellite (Earth radius + altitude).
        alpha_from_nadir : astropy.units.Quantity
            Ray angle from the nadir direction.
            
        Returns
        -------
        angle_SOE : astropy.units.Quantity
            Central angle in radians corresponding to the edge of the antenna beam.
            Returns np.nan * u.rad if the ray does not intersect Earth.
        """
        # --- Internal Cached Function  ---
        @lru_cache(maxsize=16)
        def _calc_spherical_central_angle_cached(R_eff_m, R_sat_m, alpha_rad):
            """
            Internal cached function to compute the central angle (in radians) using
            scalar values for the effective Earth radius, satellite distance, and ray angle.
            
            Parameters
            ----------
            R_eff_m : float
                Effective Earth radius in meters.
            R_sat_m : float
                Satellite distance from Earth's center in meters.
            alpha_rad : float
                Ray angle from nadir in radians.
                
            Returns
            -------
            angle_SOE : float
                Central angle in radians. Returns np.nan if the ray does not intersect Earth.
            """
            OS_sq = R_sat_m ** 2
            OE_sq = R_eff_m ** 2
            two_OS_OE = 2 * R_sat_m * R_eff_m

            cos_alpha = np.cos(alpha_rad)
            sin_alpha = np.sin(alpha_rad)
            
            term_under_sqrt = OE_sq - OS_sq * sin_alpha ** 2
            if term_under_sqrt < -1e-11 * OE_sq:
                return np.nan
            sqrt_term = np.sqrt(max(term_under_sqrt, 0.0))
            
            slant_range_SE = R_sat_m * cos_alpha - sqrt_term
            if slant_range_SE < 0:
                return np.nan
            
            SE_sq = slant_range_SE ** 2
            cos_SOE = (OS_sq + OE_sq - SE_sq) / two_OS_OE
            cos_SOE = np.clip(cos_SOE, -1.0, 1.0)
            return np.arccos(cos_SOE)
        
        # Convert quantities to float values in consistent units.
        R_eff_m = R_eff.to(u.m).value
        R_sat_m = R_sat.to(u.m).value
        alpha_rad = alpha_from_nadir.to(u.rad).value
        angle = _calc_spherical_central_angle_cached(R_eff_m, R_sat_m, alpha_rad)
        return angle * u.rad
    
    if not callable(antenna_gain_func):
        raise TypeError("`antenna_gain_func` must be callable.")
    altitude = altitude.to(u.m)
    off_nadir_angle = off_nadir_angle.to(u.deg)
    
    try:
        theta_3dB = calculate_3dB_angle_1d(antenna_gain_func, **antenna_pattern_kwargs)
    except Exception as e:
        warnings.warn(f"Finding -3dB angle failed: {e}", RuntimeWarning)
        return np.nan * u.m

    if earth_model == 'spherical':
        R_eff = R_earth
    else:
        raise ValueError("Only 'spherical' earth_model is implemented.")
    R_sat = R_eff + altitude
    
    sin_max_dev_arg = (R_eff / R_sat).decompose().value
    sin_max_dev_clipped = np.clip(sin_max_dev_arg, -1.0, 1.0)
    max_deviation_angle = np.arcsin(sin_max_dev_clipped) * u.rad
    
    angle_tol = 1e-9 * u.rad
    if off_nadir_angle > max_deviation_angle + theta_3dB + angle_tol:
        warnings.warn("Entire -3dB beam misses Earth.", UserWarning)
        return np.nan * u.m
    if off_nadir_angle > max_deviation_angle + angle_tol:
        warnings.warn("Antenna boresight points beyond Earth limb.", UserWarning)
    
    total_central_angle = np.nan * u.rad
    if np.isclose(off_nadir_angle.value, 0.0, atol=1e-9):
        alpha_edge = theta_3dB
        if alpha_edge > max_deviation_angle + angle_tol:
            warnings.warn("Nadir beam edge misses Earth (theta_3dB > limb angle).", UserWarning)
            return np.nan * u.m
        angle_SOE_edge = _calculate_spherical_central_angle(R_eff, R_sat, alpha_edge)
        if np.isnan(angle_SOE_edge.value):
            warnings.warn("Nadir beam edge ray calculation failed.", RuntimeWarning)
            return np.nan * u.m
        total_central_angle = 2.0 * angle_SOE_edge
    else:
        alpha_1 = off_nadir_angle - theta_3dB
        alpha_2 = off_nadir_angle + theta_3dB
        angle_SOE_1 = _calculate_spherical_central_angle(R_eff, R_sat, alpha_1)
        angle_SOE_2 = _calculate_spherical_central_angle(R_eff, R_sat, alpha_2)
        nan_1 = np.isnan(angle_SOE_1.value)
        nan_2 = np.isnan(angle_SOE_2.value)
        
        if nan_1 and nan_2:
            warnings.warn(f"Both edge rays miss Earth (alpha1={alpha_1:.2f}, alpha2={alpha_2:.2f}).", UserWarning)
            return np.nan * u.m
        elif nan_1 or nan_2:
            angle_SOE_hit = angle_SOE_2 if nan_1 else angle_SOE_1
            alpha_miss = alpha_1 if nan_1 else alpha_2
            angle_SOE_limb = _calculate_spherical_central_angle(R_eff, R_sat, max_deviation_angle)
            if np.isnan(angle_SOE_limb.value):
                warnings.warn("Internal error: Could not calculate limb intersection angle.", RuntimeWarning)
                return np.nan * u.m
            total_central_angle = np.abs(angle_SOE_limb - angle_SOE_hit)
            missing_edge_name = "near" if nan_1 else "far"
            warnings.warn(f"Partial footprint: {missing_edge_name} edge ray (alpha={alpha_miss:.2f}) misses Earth. Size calculated to limb.", UserWarning)
        else:
            total_central_angle = np.abs(angle_SOE_2 - angle_SOE_1)
    
    if np.isnan(total_central_angle.value):
        return np.nan * u.m
    footprint_diameter = R_eff * total_central_angle.to(u.rad).value
    if not footprint_diameter.unit.is_equivalent(u.m):
        raise u.UnitTypeError(f"Calculation resulted in unexpected units: {footprint_diameter.unit}")
    return footprint_diameter

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
    return grid_longitudes, grid_latitudes, grid_spacing

@ranged_quantity_input(
    sat_altitude=(1, None, u.km),
    min_elevation=(0, 90, u.deg),
    strip_input_units=False,
    allow_none=True
)
def trunc_hexgrid_to_impactful(grid_longitudes, grid_latitudes, sat_altitude, min_elevation, 
                               station_lat, station_lon, station_height):
    """
    Truncates a full hexagon grid to only those cells that are potentially served by
    satellites positioned up to the horizon circle (i.e. barely visible) from a given
    radio astronomy station.
    
    For a station at (station_lat, station_lon) with a specified satellite altitude,
    the station’s horizon angle is computed as:
    
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
        Station elevation above Earth's surface (provided for completeness).
        
    Returns
    -------
    mask : np.ndarray of bool
        Boolean mask that, when applied to grid_longitudes and grid_latitudes, retains only
        those grid cells that are impactful.
    """
    # Convert station and grid cell coordinates to radians.
    station_lat_rad = station_lat.to(u.rad).value
    station_lon_rad = station_lon.to(u.rad).value
    grid_lats_rad = np.radians(grid_latitudes)
    grid_lons_rad = np.radians(grid_longitudes)
    
    # Compute angular separation δ (in radians) using the spherical law of cosines.
    cos_delta = (np.sin(station_lat_rad) * np.sin(grid_lats_rad) +
                 np.cos(station_lat_rad) * np.cos(grid_lats_rad) *
                 np.cos(grid_lons_rad - station_lon_rad))
    cos_delta = np.clip(cos_delta, -1.0, 1.0)
    delta = np.arccos(cos_delta)
    
    # Compute the horizon angle (θₕ) for the station:
    R_earth_m = R_earth.to(u.m).value
    sat_alt_m = sat_altitude.to(u.m).value
    theta_h = np.arccos(R_earth_m / (R_earth_m + sat_alt_m))
    
    min_elev_rad = min_elevation.to(u.rad).value
    
    # Simplify the margin γ using trigonometric identities:
    # γ = π/2 - βₘₐₓ - min_elevation = arccos((R_earth * cos(min_elevation))/(R_earth + sat_altitude)) - min_elevation
    gamma = np.arccos((R_earth_m * np.cos(min_elev_rad)) / (R_earth_m + sat_alt_m)) - min_elev_rad
    
    # A cell is impactful if its angular separation δ is less than or equal to (θₕ + γ)
    # and is on the near side of Earth (δ <= π/2).
    mask = (delta <= (theta_h + gamma)) & (delta <= np.pi/2)
    
    return mask
