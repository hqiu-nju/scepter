#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
uvw.py - UVW Coordinate Transformations for Radio Interferometry

This module provides utility functions for converting baseline vectors from the 
local horizontal (Alt-Az / ENU) coordinate system to the standard radio 
interferometry UVW coordinate system following the Thompson, Moran & Swenson 
convention.

Overview
--------
In radio interferometry, the UVW coordinate system is a right-handed Cartesian 
system defined relative to the celestial source being observed:

- **U-axis**: Points toward the East along the celestial equator
- **V-axis**: Points toward the North celestial pole  
- **W-axis**: Points toward the phase tracking center (source direction)

The UVW frame rotates with the celestial sphere, following the source as it 
moves across the sky. This is essential for:

1. **Visibility measurements**: The (u,v) plane samples the Fourier transform 
   of the sky brightness
2. **Fringe tracking**: The w-component determines geometric delay
3. **Image synthesis**: UV coverage determines angular resolution and image quality

Coordinate System Transformations
----------------------------------
The module implements a standard transformation pipeline:

1. **ITRF → ENU** (Earth-fixed to local horizontal):
   Converts Earth-Centered Earth-Fixed (ITRF/ECEF) baseline vectors to the 
   local topocentric East-North-Up frame at the observer location.

2. **ENU → UVW** (Local horizontal to interferometry frame):
   Rotates the local baseline into the UVW frame using the hour angle and 
   declination of the observed source.

The complete transformation from ITRF to UVW is:
    
    baseline_ITRF → (apply observer lat/lon) → baseline_ENU → 
    (apply source HA/Dec) → baseline_UVW

Thompson, Moran & Swenson Convention
-------------------------------------
This module follows the standard rotation matrices from "Interferometry and 
Synthesis in Radio Astronomy" (3rd edition), Chapter 4:

**ITRF to ENU rotation** (equation 4.3):
    | E |   | -sin λ          cos λ         0      | | ΔX |
    | N | = | -sin φ cos λ   -sin φ sin λ   cos φ  | | ΔY |
    | U |   |  cos φ cos λ    cos φ sin λ   sin φ  | | ΔZ |

where λ = longitude, φ = latitude (geodetic)

**ENU to UVW rotation** (equation 4.1):
    | u |   |  sin H           cos H          0      | | E |
    | v | = | -sin δ cos H     sin δ sin H    cos δ  | | N |
    | w |   |  cos δ cos H    -cos δ sin H    sin δ  | | U |

where H = hour angle, δ = declination

Integration with scepter.obs
-----------------------------
This module extends the interferometry capabilities in scepter.obs:

- `scepter.obs.baseline_bearing()` → provides ITRF baseline vectors
- `scepter.obs.baseline_pairs()` → computes all baselines in an array
- `scepter.uvw.itrf_to_enu()` → converts to local topocentric frame
- `scepter.uvw.enu_to_uvw()` → transforms to interferometry UVW frame

The high-level functions (`compute_uvw_from_observers`, `compute_uvw_astropy`, 
`compute_uvw_array`) provide end-to-end pipelines that handle all coordinate 
transformations automatically.

Dependencies
------------
Core functions (numpy-only):
    - numpy: Numerical array operations and linear algebra

Optional dependencies for high-level functions:
    - cysgp4: PyObserver objects for observer locations
    - pycraf: Geospatial coordinate transformations
    - astropy: EarthLocation, Time, LST calculations

Usage Examples
--------------
**Basic transformation using numpy arrays**::

    >>> import numpy as np
    >>> from scepter import uvw
    >>> 
    >>> # Baseline in ENU coordinates (100m east, 0m north, 0m up)
    >>> baseline_enu = np.array([100.0, 0.0, 0.0])
    >>> 
    >>> # Source at RA=0h, Dec=+45°, observed at LST=6h
    >>> ra_rad = 0.0
    >>> lst_rad = np.radians(90.0)  # 6h = 90°
    >>> ha = uvw.hour_angle(ra_rad, lst_rad)
    >>> 
    >>> # Transform to UVW
    >>> baseline_uvw = uvw.enu_to_uvw(ha, np.radians(45.0), baseline_enu)
    >>> print(f"UVW baseline: {baseline_uvw}")

**Using cysgp4 PyObserver objects** (integrates with scepter.obs)::

    >>> from cysgp4 import PyObserver
    >>> from astropy.time import Time
    >>> from scepter import uvw
    >>> 
    >>> # Define two antennas
    >>> ref = PyObserver(21.443, -30.713, 1.0)  # lon, lat (deg), alt (km)
    >>> ant = PyObserver(21.444, -30.713, 1.0)
    >>> 
    >>> # Observe a source at RA=0°, Dec=-30°
    >>> times = Time(['2024-01-01T00:00:00', '2024-01-01T01:00:00'])
    >>> uvw_coords, ha = uvw.compute_uvw_from_observers(
    ...     ref, ant, ra_deg=0.0, dec_deg=-30.0, obs_times=times
    ... )

**Computing all baselines in an array**::

    >>> from scepter import uvw
    >>> 
    >>> # Array of 3 antennas
    >>> antennas = [ant1, ant2, ant3]  # PyObserver objects
    >>> 
    >>> # Compute UVW for all baselines relative to first antenna
    >>> uvw_all, ha = uvw.compute_uvw_array(
    ...     antennas, ra_deg=0.0, dec_deg=45.0, obs_times=times
    ... )
    >>> print(f"Shape: {uvw_all.shape}")  # (3, n_times, 3)

References
----------
- Thompson, A.R., Moran, J.M., and Swenson, G.W. Jr. (2017), 
  "Interferometry and Synthesis in Radio Astronomy", 3rd Edition, Springer
- NRAO CASA documentation on UVW coordinates
- AIPS Memo 117: "The Measurement Equation of a Generic Radio Telescope"

See Also
--------
scepter.obs.baseline_bearing : Calculate ITRF baseline between two antennas
scepter.obs.baseline_pairs : Calculate all baselines in an array
scepter.obs.mod_tau : Calculate geometric delay from baseline

Author: Generated for SCEPTer package
Date Created: 2026-02-06
Version: 0.1
"""

import numpy as np

# Optional imports - these are used by high-level convenience functions
try:
    import cysgp4
    CYSGP4_AVAILABLE = True
except ImportError:
    CYSGP4_AVAILABLE = False

try:
    import pycraf
    from pycraf import geospatial
    PYCRAF_AVAILABLE = True
except ImportError:
    PYCRAF_AVAILABLE = False

try:
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import EarthLocation
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


def hour_angle(ra_rad, lst_rad):
    """
    Compute hour angle from right ascension and local sidereal time.
    
    The hour angle measures the time since the source crossed the local 
    meridian, expressed as an angle. It increases from negative values 
    (before transit) through zero (at transit) to positive values (after transit).
    
    Parameters
    ----------
    ra_rad : float or array-like
        Right ascension of the source (radians)
        Range: [0, 2π)
    lst_rad : float or array-like
        Local sidereal time (radians)
        Range: [0, 2π)
        Can be computed from observer longitude and Greenwich Sidereal Time
    
    Returns
    -------
    hour_angle : float or numpy.ndarray
        Hour angle (radians)
        Range: [-π, π] (wrapped to this range)
        Positive values indicate source has passed meridian (setting)
        Negative values indicate source approaching meridian (rising)
    
    Notes
    -----
    The hour angle is defined as:
        H = LST - RA
    
    This is wrapped to the range [-π, π] for consistency with standard 
    conventions. The wrapping ensures:
    - H = 0: Source on meridian (transiting)
    - H = -π/2: Source 6 hours before transit (rising in east)
    - H = +π/2: Source 6 hours after transit (setting in west)
    
    Hour angle is a crucial quantity in interferometry as it determines:
    - The orientation of the baseline in the UVW frame
    - The rate of fringe rotation
    - The instantaneous UV coverage
    
    For arrays tracking a source, hour angle changes linearly with time:
        dH/dt = 2π rad / (sidereal day) ≈ 7.27×10⁻⁵ rad/s
    
    Examples
    --------
    >>> import numpy as np
    >>> from scepter import uvw
    >>> 
    >>> # Source at RA = 0h, LST = 6h (source 6h past meridian)
    >>> ra = 0.0  # 0h in radians
    >>> lst = np.radians(90.0)  # 6h = 90°
    >>> ha = uvw.hour_angle(ra, lst)
    >>> print(f"Hour angle: {np.degrees(ha):.1f}°")  # 90°
    >>> 
    >>> # Source transiting (on meridian)
    >>> ra = np.radians(45.0)  # 3h
    >>> lst = np.radians(45.0)  # Same as RA
    >>> ha = uvw.hour_angle(ra, lst)
    >>> print(f"Hour angle at transit: {np.degrees(ha):.1f}°")  # 0°
    >>> 
    >>> # Array of times (LST changing, RA constant)
    >>> lst_array = np.linspace(0, 2*np.pi, 100)
    >>> ra_fixed = np.radians(45.0)
    >>> ha_array = uvw.hour_angle(ra_fixed, lst_array)
    
    See Also
    --------
    enu_to_uvw : Use hour angle to rotate ENU to UVW
    compute_uvw_from_observers : Automatically compute hour angle from time
    """
    ha = lst_rad - ra_rad
    # Wrap to [-π, π]
    ha = np.arctan2(np.sin(ha), np.cos(ha))
    return ha


def itrf_to_enu(baseline_itrf, longitude_rad, latitude_rad):
    """
    Convert ITRF baseline vector to local East-North-Up coordinates.
    
    Transforms an Earth-Centered Earth-Fixed (ECEF/ITRF) baseline vector to 
    the local topocentric East-North-Up coordinate system at a specified 
    observer location. This is the first step in converting baselines to 
    the UVW frame for interferometry.
    
    Parameters
    ----------
    baseline_itrf : array-like, shape (..., 3)
        Baseline vector in ITRF Cartesian coordinates (meters)
        Components: [ΔX, ΔY, ΔZ] in ITRF frame
        Can be obtained from scepter.obs.baseline_bearing() or 
        scepter.obs.baseline_pairs()
        Last dimension must be 3 (X, Y, Z components)
    longitude_rad : float
        Observer's geodetic longitude (radians)
        Positive East of Greenwich meridian
        Range: [-π, π] or [0, 2π]
    latitude_rad : float
        Observer's geodetic latitude (radians)
        Positive North of equator
        Range: [-π/2, π/2]
    
    Returns
    -------
    baseline_enu : numpy.ndarray, shape (..., 3)
        Baseline vector in local ENU coordinates (meters)
        Components: [E, N, U]
        - E: East component (positive toward local east)
        - N: North component (positive toward local north)  
        - U: Up component (positive toward local zenith)
        Shape matches input baseline_itrf
    
    Notes
    -----
    The transformation uses the standard rotation matrix from geodesy and 
    radio astronomy (Thompson, Moran & Swenson, equation 4.3):
    
        | E |   | -sin λ          cos λ         0      | | ΔX |
        | N | = | -sin φ cos λ   -sin φ sin λ   cos φ  | | ΔY |
        | U |   |  cos φ cos λ    cos φ sin λ   sin φ  | | ΔZ |
    
    where λ = longitude, φ = latitude (geodetic WGS84).
    
    This rotation:
    1. Accounts for the observer's position on the Earth's surface
    2. Transforms from Earth-fixed to horizon-fixed coordinates
    3. Preserves vector length (orthogonal rotation)
    
    The ITRF (International Terrestrial Reference Frame) is an Earth-centered, 
    Earth-fixed system:
    - Origin: Earth's center of mass
    - Z-axis: Earth's rotation axis (North pole)
    - X-axis: Intersection of equatorial plane and prime meridian
    - Y-axis: 90° East, completes right-handed system
    
    The ENU frame is local to the observer:
    - Origin: Observer's position
    - E-axis: Local tangent pointing East
    - N-axis: Local tangent pointing North
    - U-axis: Local normal (zenith direction)
    
    This function supports broadcasting over multiple baselines and times.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scepter import uvw
    >>> 
    >>> # Baseline in ITRF coordinates (e.g., from baseline_bearing)
    >>> baseline_itrf = np.array([100.0, 0.0, 0.0])  # 100m in X direction
    >>> 
    >>> # Observer at Greenwich, on equator
    >>> lon = 0.0  # Prime meridian
    >>> lat = 0.0  # Equator
    >>> baseline_enu = uvw.itrf_to_enu(baseline_itrf, lon, lat)
    >>> print(f"ENU baseline: {baseline_enu}")
    >>> 
    >>> # Array of baselines (from baseline_pairs)
    >>> baselines_itrf = np.array([
    ...     [100.0, 0.0, 0.0],
    ...     [0.0, 100.0, 0.0],
    ...     [0.0, 0.0, 100.0]
    ... ])
    >>> lon = np.radians(21.443)  # Example: South Africa
    >>> lat = np.radians(-30.713)
    >>> baselines_enu = uvw.itrf_to_enu(baselines_itrf, lon, lat)
    >>> print(f"Shape: {baselines_enu.shape}")  # (3, 3)
    
    See Also
    --------
    enu_to_uvw : Second transformation step to UVW coordinates
    scepter.obs.baseline_bearing : Get ITRF baseline between two antennas
    scepter.obs.baseline_pairs : Get ITRF baselines for antenna array
    compute_uvw_from_observers : Complete pipeline including this transformation
    
    References
    ----------
    Thompson, Moran & Swenson (2017), "Interferometry and Synthesis in Radio 
    Astronomy", 3rd ed., equation 4.3
    """
    baseline_itrf = np.asarray(baseline_itrf)
    
    # Precompute trigonometric functions
    sin_lon = np.sin(longitude_rad)
    cos_lon = np.cos(longitude_rad)
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    
    # Rotation matrix from ITRF to ENU
    # | E |   | -sin λ          cos λ         0      | | ΔX |
    # | N | = | -sin φ cos λ   -sin φ sin λ   cos φ  | | ΔY |
    # | U |   |  cos φ cos λ    cos φ sin λ   sin φ  | | ΔZ |
    R = np.array([
        [-sin_lon,          cos_lon,         0.0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat]
    ])
    
    # Apply rotation: baseline_enu = R @ baseline_itrf
    # Use np.tensordot for proper broadcasting over arbitrary leading dimensions
    baseline_enu = np.tensordot(baseline_itrf, R.T, axes=(-1, -1))
    
    return baseline_enu


def enu_to_uvw(hour_angle, declination, baseline_enu):
    """
    Convert ENU baseline to UVW coordinates for interferometry.
    
    Applies the standard rotation from the local topocentric East-North-Up 
    frame to the celestial UVW coordinate system used in radio interferometry. 
    This transformation depends on the hour angle and declination of the 
    observed source.
    
    Parameters
    ----------
    hour_angle : float or array-like
        Hour angle of the source (radians)
        H = LST - RA (from hour_angle function)
        Shape must be broadcastable with baseline_enu
    declination : float
        Declination of the source (radians)
        Range: [-π/2, π/2]
        Positive for northern hemisphere sources
    baseline_enu : array-like, shape (..., 3)
        Baseline vector in local ENU coordinates (meters)
        Components: [E, N, U]
        From itrf_to_enu() or directly computed
        Last dimension must be 3
    
    Returns
    -------
    baseline_uvw : numpy.ndarray, shape (..., 3)
        Baseline vector in UVW coordinates (meters)
        Components: [u, v, w]
        - u: East component in equatorial plane (perpendicular to source)
        - v: North component (toward celestial pole, perpendicular to source)
        - w: Component toward source (phase tracking center)
        Shape matches broadcasted shape of inputs
    
    Notes
    -----
    The transformation follows Thompson, Moran & Swenson equation 4.1:
    
        | u |   |  sin H           cos H          0      | | E |
        | v | = | -sin δ cos H     sin δ sin H    cos δ  | | N |
        | w |   |  cos δ cos H    -cos δ sin H    sin δ  | | U |
    
    where H = hour angle, δ = declination.
    
    This rotation:
    1. Aligns the w-axis with the source direction
    2. Keeps u in the equatorial plane (perpendicular to w)
    3. Keeps v toward the north celestial pole (perpendicular to u and w)
    
    **Physical interpretation:**
    - **w-component**: Determines geometric delay (longer w → larger delay)
    - **u,v-components**: Sample the Fourier plane (visibility domain)
    - UV distance √(u²+v²): Spatial frequency sampled
    
    **Time dependence:**
    As the Earth rotates, hour angle changes linearly with time, causing 
    the UV coordinates to trace elliptical or circular tracks in the UV plane. 
    This is fundamental to aperture synthesis - different hour angles give 
    different UV coverage.
    
    **Broadcasting support:**
    This function supports full numpy broadcasting, allowing:
    - Single baseline, multiple times (hour_angle array)
    - Multiple baselines, single time
    - Multiple baselines, multiple times (full array processing)
    
    Examples
    --------
    >>> import numpy as np
    >>> from scepter import uvw
    >>> 
    >>> # Single baseline, single time
    >>> baseline_enu = np.array([100.0, 0.0, 0.0])  # 100m East
    >>> ha = np.radians(0.0)   # Source on meridian
    >>> dec = np.radians(45.0)  # 45° North
    >>> baseline_uvw = uvw.enu_to_uvw(ha, dec, baseline_enu)
    >>> print(f"UVW: {baseline_uvw}")
    >>> 
    >>> # Single baseline, track over time (UV track)
    >>> baseline_enu = np.array([100.0, 50.0, 0.0])
    >>> hour_angles = np.linspace(-np.pi/2, np.pi/2, 100)  # -6h to +6h
    >>> dec = np.radians(-30.0)  # Southern source
    >>> # Broadcast: baseline_enu shape (3,), hour_angles shape (100,)
    >>> uvw_track = uvw.enu_to_uvw(hour_angles[:, np.newaxis], dec, baseline_enu)
    >>> print(f"UV track shape: {uvw_track.shape}")  # (100, 3)
    >>> 
    >>> # Multiple baselines, single time
    >>> baselines_enu = np.array([
    ...     [100.0, 0.0, 0.0],
    ...     [0.0, 100.0, 0.0]
    ... ])
    >>> ha = np.radians(30.0)
    >>> uvw_baselines = uvw.enu_to_uvw(ha, dec, baselines_enu)
    >>> print(f"Shape: {uvw_baselines.shape}")  # (2, 3)
    
    See Also
    --------
    hour_angle : Compute hour angle from RA and LST
    itrf_to_enu : First transformation step from ITRF to ENU
    compute_uvw_from_observers : End-to-end pipeline for UVW calculation
    
    References
    ----------
    Thompson, Moran & Swenson (2017), "Interferometry and Synthesis in Radio 
    Astronomy", 3rd ed., equation 4.1, Chapter 4
    """
    baseline_enu = np.asarray(baseline_enu)
    hour_angle = np.asarray(hour_angle)
    
    # Precompute trigonometric functions
    sin_H = np.sin(hour_angle)
    cos_H = np.cos(hour_angle)
    sin_dec = np.sin(declination)
    cos_dec = np.cos(declination)
    
    # Rotation matrix from ENU to UVW
    # | u |   |  sin H           cos H          0      | | E |
    # | v | = | -sin δ cos H     sin δ sin H    cos δ  | | N |
    # | w |   |  cos δ cos H    -cos δ sin H    sin δ  | | U |
    
    # Handle broadcasting: ensure compatible shapes
    # baseline_enu has shape (..., 3)
    # hour_angle may have shape () or (N,) or (N, M, ...)
    # We need to broadcast the rotation matrix properly
    
    # Extract ENU components
    E = baseline_enu[..., 0]
    N = baseline_enu[..., 1]
    U = baseline_enu[..., 2]
    
    # Apply rotation matrix
    u = sin_H * E + cos_H * N
    v = -sin_dec * cos_H * E + sin_dec * sin_H * N + cos_dec * U
    w = cos_dec * cos_H * E - cos_dec * sin_H * N + sin_dec * U
    
    # Stack into UVW array
    # Remove any singleton dimensions that were added by broadcasting
    baseline_uvw = np.stack([u, v, w], axis=-1)
    
    # Squeeze out singleton dimensions if baseline_enu was 1D
    if baseline_enu.ndim == 1:
        baseline_uvw = np.squeeze(baseline_uvw)
    
    return baseline_uvw


def compute_uvw_from_observers(ref_observer, ant_observer, ra_deg, dec_deg, obs_times):
    """
    Compute UVW coordinates from PyObserver objects (cysgp4 integration).
    
    End-to-end pipeline for computing interferometric UVW coordinates from 
    antenna locations defined as cysgp4.PyObserver objects. This integrates 
    seamlessly with the scepter.obs module and handles all coordinate 
    transformations automatically.
    
    Parameters
    ----------
    ref_observer : cysgp4.PyObserver
        Reference antenna/observer location (baseline origin)
        Must have .loc attributes: lon (degrees), lat (degrees), alt (meters)
        Typically the first antenna in an array
    ant_observer : cysgp4.PyObserver
        Target antenna for baseline measurement
        Same requirements as ref_observer
    ra_deg : float
        Right ascension of the observed source (degrees)
        Range: [0, 360)
    dec_deg : float
        Declination of the observed source (degrees)
        Range: [-90, 90]
    obs_times : astropy.time.Time
        Observation times
        Can be scalar or array
        Used to compute local sidereal time
    
    Returns
    -------
    uvw : numpy.ndarray, shape (N_times, 3) or (3,)
        UVW coordinates at each observation time (meters)
        Shape (3,) if obs_times is scalar, (N_times, 3) if array
        Components: [u, v, w]
    hour_angles : numpy.ndarray, shape (N_times,) or scalar
        Hour angles at each observation time (radians)
        Useful for diagnostics and UV track plotting
    
    Raises
    ------
    ImportError
        If cysgp4 or pycraf is not available
        Both are required for this function
    
    Notes
    -----
    **Processing pipeline:**
    
    1. Compute ITRF baseline from observer locations (uses pycraf)
    2. Convert ITRF to ENU using reference observer's lat/lon
    3. Calculate local sidereal time from obs_times (uses astropy)
    4. Compute hour angle from LST and RA
    5. Transform ENU to UVW using hour angle and declination
    
    **Integration with scepter.obs:**
    
    This function is designed to work with the same PyObserver objects used 
    throughout the scepter package:
    
    - scepter.obs.obs_sim uses PyObserver for antenna locations
    - scepter.obs.baseline_bearing works with PyObserver pairs
    - This function extends that infrastructure for UVW calculations
    
    **Time handling:**
    
    Local sidereal time is computed from:
    - Observer's longitude
    - UTC time (from obs_times)
    - Earth's rotation
    
    This accounts for:
    - Longitude offset from Greenwich
    - Precession and nutation (via astropy)
    - Proper sidereal vs. solar time conversion
    
    Examples
    --------
    >>> from cysgp4 import PyObserver
    >>> from astropy.time import Time
    >>> from scepter import uvw
    >>> 
    >>> # Define two antennas (e.g., VLA-like in New Mexico)
    >>> ref = PyObserver(-107.618, 34.079, 2.124)  # lon, lat (deg), alt (km)
    >>> ant = PyObserver(-107.617, 34.079, 2.124)
    >>> 
    >>> # Observe Cygnus A (RA=19h59m, Dec=+40°44')
    >>> ra = (19 + 59/60) * 15  # Convert hours to degrees
    >>> dec = 40 + 44/60
    >>> times = Time('2024-06-21T03:00:00')  # Single time
    >>> 
    >>> uvw_coords, ha = uvw.compute_uvw_from_observers(
    ...     ref, ant, ra_deg=ra, dec_deg=dec, obs_times=times
    ... )
    >>> print(f"UVW: {uvw_coords}")
    >>> print(f"Hour angle: {np.degrees(ha):.2f}°")
    >>> 
    >>> # Track source over 6 hours
    >>> times = Time('2024-06-21T00:00:00') + np.linspace(0, 6, 25) * u.hour
    >>> uvw_track, ha_track = uvw.compute_uvw_from_observers(
    ...     ref, ant, ra_deg=ra, dec_deg=dec, obs_times=times
    ... )
    >>> print(f"UV track shape: {uvw_track.shape}")  # (25, 3)
    >>> 
    >>> # Plot UV coverage
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(uvw_track[:, 0], uvw_track[:, 1], 'o-')
    >>> plt.xlabel('u (m)'); plt.ylabel('v (m)')
    >>> plt.title('UV track')
    >>> plt.axis('equal')
    
    See Also
    --------
    compute_uvw_astropy : Alternative using EarthLocation instead of PyObserver
    compute_uvw_array : Compute UVW for all baselines in an array
    scepter.obs.baseline_bearing : Get ITRF baseline between observers
    hour_angle : Compute hour angle from RA and LST
    itrf_to_enu : ITRF to ENU transformation
    enu_to_uvw : ENU to UVW transformation
    
    References
    ----------
    Thompson, Moran & Swenson (2017), "Interferometry and Synthesis in Radio 
    Astronomy", 3rd ed., Chapter 4
    """
    if not CYSGP4_AVAILABLE:
        raise ImportError(
            "cysgp4 is required for compute_uvw_from_observers. "
            "Install with: pip install cysgp4"
        )
    if not PYCRAF_AVAILABLE:
        raise ImportError(
            "pycraf is required for compute_uvw_from_observers. "
            "Install with: pip install pycraf"
        )
    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required for compute_uvw_from_observers. "
            "Install with: pip install astropy"
        )
    
    # Step 1: Get ITRF baseline using pycraf
    x1, y1, z1 = geospatial.wgs84_to_itrf2008(
        ref_observer.loc.lon * u.deg,
        ref_observer.loc.lat * u.deg,
        ref_observer.loc.alt * u.m
    )
    x2, y2, z2 = geospatial.wgs84_to_itrf2008(
        ant_observer.loc.lon * u.deg,
        ant_observer.loc.lat * u.deg,
        ant_observer.loc.alt * u.m
    )
    
    baseline_itrf = np.array([
        x2.value - x1.value,
        y2.value - y1.value,
        z2.value - z1.value
    ])
    
    # Step 2: Convert ITRF to ENU
    lon_rad = np.radians(ref_observer.loc.lon)
    lat_rad = np.radians(ref_observer.loc.lat)
    baseline_enu = itrf_to_enu(baseline_itrf, lon_rad, lat_rad)
    
    # Step 3: Calculate LST and hour angle
    # Create EarthLocation for LST calculation
    location = EarthLocation(
        lon=ref_observer.loc.lon * u.deg,
        lat=ref_observer.loc.lat * u.deg,
        height=ref_observer.loc.alt * u.m
    )
    
    # Compute LST
    lst = obs_times.sidereal_time('apparent', longitude=location.lon)
    lst_rad = lst.radian
    
    # Convert RA to radians and compute hour angle
    ra_rad = np.radians(ra_deg)
    ha = hour_angle(ra_rad, lst_rad)
    
    # Step 4: Convert ENU to UVW
    dec_rad = np.radians(dec_deg)
    uvw_coords = enu_to_uvw(ha, dec_rad, baseline_enu)
    
    return uvw_coords, ha


def compute_uvw_astropy(ref_location, ant_location, ra_deg, dec_deg, obs_times):
    """
    Compute UVW coordinates from EarthLocation objects (pure astropy).
    
    End-to-end pipeline for computing interferometric UVW coordinates using 
    astropy.coordinates.EarthLocation objects. This provides an alternative to 
    compute_uvw_from_observers that doesn't require pycraf or cysgp4.
    
    Parameters
    ----------
    ref_location : astropy.coordinates.EarthLocation
        Reference antenna/observer location (baseline origin)
        Must have geodetic coordinates (lon, lat, height)
    ant_location : astropy.coordinates.EarthLocation
        Target antenna for baseline measurement
        Same requirements as ref_location
    ra_deg : float
        Right ascension of the observed source (degrees)
        Range: [0, 360)
    dec_deg : float
        Declination of the observed source (degrees)
        Range: [-90, 90]
    obs_times : astropy.time.Time
        Observation times
        Can be scalar or array
        Used to compute local sidereal time
    
    Returns
    -------
    uvw : numpy.ndarray, shape (N_times, 3) or (3,)
        UVW coordinates at each observation time (meters)
        Shape (3,) if obs_times is scalar, (N_times, 3) if array
        Components: [u, v, w]
    hour_angles : numpy.ndarray, shape (N_times,) or scalar
        Hour angles at each observation time (radians)
        Useful for diagnostics and UV track plotting
    
    Raises
    ------
    ImportError
        If astropy is not available
    
    Notes
    -----
    **Difference from compute_uvw_from_observers:**
    
    This function uses EarthLocation objects instead of PyObserver objects, 
    making it suitable for pure astropy workflows without needing pycraf or 
    cysgp4 dependencies.
    
    **Processing pipeline:**
    
    1. Extract ITRF (geocentric) coordinates from EarthLocation
    2. Compute ITRF baseline vector
    3. Convert to ENU using reference location's lat/lon
    4. Calculate hour angle from LST and RA
    5. Transform ENU to UVW
    
    **Coordinate systems:**
    
    EarthLocation stores coordinates in multiple representations:
    - Geodetic: (longitude, latitude, height) - WGS84 ellipsoid
    - Geocentric: (x, y, z) - ITRS Cartesian
    
    This function uses the geocentric (x, y, z) coordinates directly, which 
    are equivalent to ITRF coordinates for our purposes.
    
    Examples
    --------
    >>> from astropy.coordinates import EarthLocation
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from scepter import uvw
    >>> 
    >>> # Define two antennas using EarthLocation
    >>> ref = EarthLocation(lon=-107.618*u.deg, lat=34.079*u.deg, height=2124*u.m)
    >>> ant = EarthLocation(lon=-107.617*u.deg, lat=34.079*u.deg, height=2124*u.m)
    >>> 
    >>> # Observe a source
    >>> times = Time('2024-01-01T00:00:00')
    >>> uvw_coords, ha = uvw.compute_uvw_astropy(
    ...     ref, ant, ra_deg=0.0, dec_deg=45.0, obs_times=times
    ... )
    >>> print(f"UVW: {uvw_coords}")
    >>> 
    >>> # Track over multiple times
    >>> times = Time('2024-01-01T00:00:00') + np.linspace(0, 6, 25) * u.hour
    >>> uvw_track, ha_track = uvw.compute_uvw_astropy(
    ...     ref, ant, ra_deg=0.0, dec_deg=45.0, obs_times=times
    ... )
    >>> print(f"Shape: {uvw_track.shape}")  # (25, 3)
    
    See Also
    --------
    compute_uvw_from_observers : Alternative using PyObserver objects
    compute_uvw_array : Compute UVW for all baselines in an array
    hour_angle : Compute hour angle from RA and LST
    itrf_to_enu : ITRF to ENU transformation
    enu_to_uvw : ENU to UVW transformation
    astropy.coordinates.EarthLocation : Astropy location objects
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required for compute_uvw_astropy. "
            "Install with: pip install astropy"
        )
    
    # Step 1: Get ITRF baseline from EarthLocation objects
    # EarthLocation provides geocentric (x, y, z) directly
    baseline_itrf = np.array([
        (ant_location.x - ref_location.x).to(u.m).value,
        (ant_location.y - ref_location.y).to(u.m).value,
        (ant_location.z - ref_location.z).to(u.m).value
    ])
    
    # Step 2: Convert ITRF to ENU
    lon_rad = ref_location.lon.radian
    lat_rad = ref_location.lat.radian
    baseline_enu = itrf_to_enu(baseline_itrf, lon_rad, lat_rad)
    
    # Step 3: Calculate LST and hour angle
    lst = obs_times.sidereal_time('apparent', longitude=ref_location.lon)
    lst_rad = lst.radian
    
    # Convert RA to radians and compute hour angle
    ra_rad = np.radians(ra_deg)
    ha = hour_angle(ra_rad, lst_rad)
    
    # Step 4: Convert ENU to UVW
    dec_rad = np.radians(dec_deg)
    uvw_coords = enu_to_uvw(ha, dec_rad, baseline_enu)
    
    return uvw_coords, ha


def compute_uvw_array(antennas, ra_deg, dec_deg, obs_times):
    """
    Compute UVW coordinates for all baselines in an antenna array.
    
    Computes interferometric UVW coordinates for all baselines in an array 
    simultaneously, using the first antenna as the reference. This matches 
    the convention used by scepter.obs.baseline_pairs and is efficient for 
    processing entire arrays.
    
    Parameters
    ----------
    antennas : list of cysgp4.PyObserver or astropy.coordinates.EarthLocation
        List of antenna/observer objects
        All must be the same type (either all PyObserver or all EarthLocation)
        First antenna becomes the reference for all baselines
    ra_deg : float
        Right ascension of the observed source (degrees)
        Range: [0, 360)
    dec_deg : float
        Declination of the observed source (degrees)
        Range: [-90, 90]
    obs_times : astropy.time.Time
        Observation times
        Can be scalar or array
    
    Returns
    -------
    uvw_all : numpy.ndarray, shape (N_antennas, N_times, 3) or (N_antennas, 3)
        UVW coordinates for all baselines
        uvw_all[0] is always [0, 0, 0] (reference to itself)
        uvw_all[i] is the UVW baseline from antenna 0 to antenna i
        If obs_times is scalar, shape is (N_antennas, 3)
        If obs_times is array, shape is (N_antennas, N_times, 3)
    hour_angles : numpy.ndarray, shape (N_times,) or scalar
        Hour angles at each observation time (radians)
        Same for all baselines (depends only on time and source position)
    
    Raises
    ------
    ImportError
        If required packages (cysgp4, pycraf, astropy) are not available
        Required packages depend on antenna object type
    ValueError
        If antennas list is empty or contains mixed types
    
    Notes
    -----
    **Baseline convention:**
    
    This function uses the same convention as scepter.obs.baseline_pairs:
    - Reference antenna: antennas[0]
    - All baselines measured from reference to other antennas
    - Baseline 0 (reference to itself) is always [0, 0, 0]
    
    **Array processing:**
    
    For N antennas and M time samples:
    - Computes N baselines (including reference to itself)
    - Each baseline tracked over M times
    - Returns shape (N, M, 3) array of UVW coordinates
    
    **Automatic type detection:**
    
    The function detects whether inputs are PyObserver or EarthLocation objects 
    and calls the appropriate backend (compute_uvw_from_observers or 
    compute_uvw_astropy).
    
    **Memory efficiency:**
    
    For large arrays, this is more efficient than calling compute_uvw_from_observers 
    repeatedly because:
    - Baseline calculations are vectorized
    - Hour angle computed once and reused
    - LST calculation done once for all baselines
    
    Examples
    --------
    >>> from cysgp4 import PyObserver
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from scepter import uvw
    >>> 
    >>> # Create 3-antenna array
    >>> ant1 = PyObserver(21.443, -30.713, 1.0)  # Reference (lon, lat deg, alt km)
    >>> ant2 = PyObserver(21.444, -30.713, 1.0)  # ~100m E
    >>> ant3 = PyObserver(21.443, -30.714, 1.0)  # ~100m N
    >>> antennas = [ant1, ant2, ant3]
    >>> 
    >>> # Single observation time
    >>> times = Time('2024-01-01T00:00:00')
    >>> uvw_all, ha = uvw.compute_uvw_array(
    ...     antennas, ra_deg=0.0, dec_deg=-30.0, obs_times=times
    ... )
    >>> print(f"Shape: {uvw_all.shape}")  # (3, 3)
    >>> print(f"Baseline 0->0: {uvw_all[0]}")  # [0, 0, 0]
    >>> print(f"Baseline 0->1: {uvw_all[1]}")  # Eastward baseline
    >>> print(f"Baseline 0->2: {uvw_all[2]}")  # Northward baseline
    >>> 
    >>> # Track over time
    >>> times = Time('2024-01-01T00:00:00') + np.linspace(0, 6, 25) * u.hour
    >>> uvw_all, ha = uvw.compute_uvw_array(
    ...     antennas, ra_deg=0.0, dec_deg=-30.0, obs_times=times
    ... )
    >>> print(f"Shape: {uvw_all.shape}")  # (3, 25, 3)
    >>> 
    >>> # Plot UV coverage for all baselines
    >>> import matplotlib.pyplot as plt
    >>> for i in range(1, len(antennas)):  # Skip reference baseline
    ...     u_coords = uvw_all[i, :, 0]
    ...     v_coords = uvw_all[i, :, 1]
    ...     plt.plot(u_coords, v_coords, 'o-', label=f'Baseline 0-{i}')
    >>> plt.xlabel('u (m)'); plt.ylabel('v (m)')
    >>> plt.legend(); plt.axis('equal')
    
    See Also
    --------
    compute_uvw_from_observers : Compute single baseline with PyObserver
    compute_uvw_astropy : Compute single baseline with EarthLocation
    scepter.obs.baseline_pairs : Get ITRF baselines for antenna array
    hour_angle : Compute hour angle from RA and LST
    """
    if not antennas:
        raise ValueError("antennas list cannot be empty")
    
    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required for compute_uvw_array. "
            "Install with: pip install astropy"
        )
    
    # Detect antenna type from first element
    first_antenna = antennas[0]
    
    # Determine which backend to use
    if CYSGP4_AVAILABLE and isinstance(first_antenna, cysgp4.PyObserver):
        # Use PyObserver backend
        if not PYCRAF_AVAILABLE:
            raise ImportError(
                "pycraf is required for compute_uvw_array with PyObserver. "
                "Install with: pip install pycraf"
            )
        use_pyobserver = True
    elif ASTROPY_AVAILABLE and isinstance(first_antenna, EarthLocation):
        # Use EarthLocation backend
        use_pyobserver = False
    else:
        raise ValueError(
            "antennas must be a list of cysgp4.PyObserver or "
            "astropy.coordinates.EarthLocation objects"
        )
    
    # Get number of antennas and determine output shape
    n_antennas = len(antennas)
    # Check if obs_times is scalar or array using astropy's isscalar
    is_time_scalar = obs_times.isscalar
    
    if not is_time_scalar:
        n_times = len(obs_times)
        uvw_all = np.zeros((n_antennas, n_times, 3))
    else:
        uvw_all = np.zeros((n_antennas, 3))
    
    # Reference antenna
    ref = antennas[0]
    
    # Compute UVW for each baseline
    for i, ant in enumerate(antennas):
        if i == 0:
            # Reference to itself - already zeros
            continue
        
        if use_pyobserver:
            uvw_coords, ha = compute_uvw_from_observers(
                ref, ant, ra_deg, dec_deg, obs_times
            )
        else:
            uvw_coords, ha = compute_uvw_astropy(
                ref, ant, ra_deg, dec_deg, obs_times
            )
        
        uvw_all[i] = uvw_coords
    
    # Hour angle is the same for all baselines (computed in last iteration)
    # For consistency, we could recompute it, but it's identical for all
    return uvw_all, ha
