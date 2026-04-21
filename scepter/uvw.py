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

The high-level function `compute_uvw` provides an end-to-end pipeline that
handles all coordinate transformations automatically for any number of antennas.

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
    >>> uvw_coords, ha = uvw.compute_uvw(
    ...     [ref, ant], ra_deg=0.0, dec_deg=-30.0, obs_times=times
    ... )

**Computing all baselines in an array**::

    >>> from scepter import uvw
    >>> 
    >>> # Array of 3 antennas
    >>> antennas = [ant1, ant2, ant3]  # PyObserver objects
    >>> 
    >>> # Compute UVW for all baselines relative to first antenna
    >>> uvw_all, ha = uvw.compute_uvw(
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
    baseline_uvw = np.stack([u, v, w], axis=-1)
    
    # Handle broadcasting artifacts: when baseline_enu is 1D, broadcasting with
    # hour_angle arrays may create extra singleton dimensions. Remove them.
    if baseline_enu.ndim == 1 and baseline_uvw.ndim > 2:
        # Squeeze out singleton dimensions except the last one (which is the 3-component UVW)
        baseline_uvw = baseline_uvw.squeeze()
        # Ensure output is at least 1D with last dimension being 3
        if baseline_uvw.ndim == 1 and baseline_uvw.shape[0] == 3:
            # Single UVW coordinate - keep as (3,)
            pass
        elif baseline_uvw.ndim == 1:
            # This shouldn't happen, but safeguard
            baseline_uvw = baseline_uvw.reshape(-1, 3)
    
    return baseline_uvw


def compute_uvw(
    antennas,
    ra_deg: float,
    dec_deg: float,
    obs_times,
    ref_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute UVW coordinates for one or more interferometric baselines.

    Unified replacement for the former ``compute_uvw_from_observers``,
    ``compute_uvw_astropy``, and ``compute_uvw_array`` helpers.  Accepts a
    list of one or more antenna objects and computes all baselines relative
    to ``antennas[ref_index]`` in a single vectorised pass, performing each
    coordinate-frame rotation exactly once regardless of array size:

        ITRF positions (all antennas, batch)
        → ITRF baselines (vectorised subtract)
        → ENU baselines (one rotation matrix applied to all)
        → UVW baselines (one rotation applied to all baselines × all times)

    LST and the ITRF → ENU rotation are computed once for the reference
    antenna and reused for every baseline.

    Parameters
    ----------
    antennas : list of cysgp4.PyObserver or astropy.coordinates.EarthLocation
        One or more antenna locations.  A two-element list ``[ref, ant]`` is
        the minimal case (single baseline).  All elements must be the same
        type.

        *PyObserver* attributes used: ``loc.lon`` (deg), ``loc.lat`` (deg),
        ``loc.alt`` (m).  Requires pycraf.

        *EarthLocation* attributes used: ``.x``, ``.y``, ``.z`` (geocentric
        Cartesian, converted to metres internally).
    ra_deg : float
        Right ascension of the target source (degrees, range [0, 360)).
    dec_deg : float
        Declination of the target source (degrees, range [−90, 90]).
    obs_times : astropy.time.Time
        Observation times.  May be scalar or 1-D array.  Used to derive the
        local sidereal time (LST) and hence the hour angle.
    ref_index : int, optional
        Index of the reference antenna within *antennas* (default 0).
        ``uvw_all[ref_index]`` is always the zero vector.

    Returns
    -------
    uvw_all : numpy.ndarray, shape (N, 3) or (N, T, 3)
        UVW coordinates (metres) for all N antennas (baselines relative to
        ``antennas[ref_index]``).  Shape is ``(N, 3)`` when *obs_times* is a
        scalar ``Time`` and ``(N, T, 3)`` for a length-T time array.

        - Axis 0 — antenna index; ``uvw_all[ref_index]`` is always zero.
        - Axis 1 (array case) — time index.
        - Last axis — UVW components ``[u, v, w]``.
    hour_angles : float or numpy.ndarray, shape (T,)
        Hour angle of the phase centre at each time (radians).  Scalar when
        *obs_times* is scalar, shape ``(T,)`` otherwise.

    Raises
    ------
    ImportError
        If astropy is not installed (always required), or if pycraf is not
        installed when PyObserver inputs are used.
    ValueError
        If *antennas* is empty, contains mixed types, or contains objects of
        an unrecognised type.

    Notes
    -----
    **Coordinate-frame pipeline:**

    1. Collect geocentric ITRF positions for all antennas in a single batch
       call (``pycraf.geospatial.wgs84_to_itrf2008`` for PyObserver inputs;
       ``.x/.y/.z`` extraction for EarthLocation inputs).
    2. Subtract the reference position to obtain all ITRF baselines at once
       (shape ``(N, 3)``; no Python loop).
    3. Apply ``itrf_to_enu`` once to the entire ``(N, 3)`` baseline matrix
       using the reference antenna's geodetic longitude and latitude.
    4. Compute apparent LST and hour angle once for all times.
    5. Apply ``enu_to_uvw`` once via broadcasting:

       - Scalar time → input ``(N, 3)``, output ``(N, 3)``.
       - Array time (T samples) → baselines expanded to ``(N, 1, 3)``,
         hour angle to ``(1, T)``, output ``(N, T, 3)``.

    Examples
    --------
    **Single baseline with PyObserver:**

    >>> from cysgp4 import PyObserver
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from scepter import uvw
    >>>
    >>> ref = PyObserver(-107.618, 34.079, 2.124)  # lon deg, lat deg, alt km
    >>> ant = PyObserver(-107.617, 34.079, 2.124)
    >>> ra = (19 + 59/60) * 15   # Cygnus A RA in degrees
    >>> dec = 40 + 44/60
    >>> times = Time('2024-06-21T03:00:00')
    >>> uvw_coords, ha = uvw.compute_uvw([ref, ant], ra_deg=ra, dec_deg=dec,
    ...                                   obs_times=times)
    >>> print(uvw_coords.shape)   # (2, 3)

    **Three-antenna array tracked over time:**

    >>> ant1 = PyObserver(21.443, -30.713, 1.086)
    >>> ant2 = PyObserver(21.445, -30.713, 1.086)
    >>> ant3 = PyObserver(21.443, -30.712, 1.086)
    >>> times = Time('2024-01-01T00:00:00') + np.linspace(0, 6, 25) * u.hour
    >>> uvw_all, ha = uvw.compute_uvw([ant1, ant2, ant3],
    ...                                ra_deg=0.0, dec_deg=-30.0,
    ...                                obs_times=times)
    >>> print(uvw_all.shape)   # (3, 25, 3)

    **EarthLocation inputs (no pycraf/cysgp4 needed):**

    >>> from astropy.coordinates import EarthLocation
    >>> ref = EarthLocation(lon=-107.618*u.deg, lat=34.079*u.deg, height=2124*u.m)
    >>> ant = EarthLocation(lon=-107.617*u.deg, lat=34.079*u.deg, height=2124*u.m)
    >>> uvw_coords, ha = uvw.compute_uvw([ref, ant], ra_deg=0.0, dec_deg=45.0,
    ...                                   obs_times=Time('2024-01-01T00:00:00'))

    See Also
    --------
    hour_angle : Compute hour angle from RA and LST.
    itrf_to_enu : ITRF → ENU rotation.
    enu_to_uvw : ENU → UVW rotation.
    scepter.obs.baseline_bearing : ITRF baseline between two antennas.
    scepter.obs.baseline_pairs : All ITRF baselines in an array.

    References
    ----------
    Thompson, Moran & Swenson (2017), "Interferometry and Synthesis in Radio
    Astronomy", 3rd ed., Chapter 4.
    """
    if not antennas:
        raise ValueError("antennas list cannot be empty")

    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required for compute_uvw. "
            "Install with: pip install astropy"
        )

    first = antennas[0]
    n = len(antennas)

    # ------------------------------------------------------------------
    # Step 1: collect geocentric ITRF positions for all antennas in batch
    # ------------------------------------------------------------------
    if CYSGP4_AVAILABLE and isinstance(first, cysgp4.PyObserver):
        if not PYCRAF_AVAILABLE:
            raise ImportError(
                "pycraf is required when using PyObserver antennas. "
                "Install with: pip install pycraf"
            )
        lons = np.array([a.loc.lon for a in antennas]) * u.deg
        lats = np.array([a.loc.lat for a in antennas]) * u.deg
        alts = np.array([a.loc.alt for a in antennas]) * u.m
        xs, ys, zs = geospatial.wgs84_to_itrf2008(lons, lats, alts)
        itrf = np.stack([xs.to(u.m).value,
                         ys.to(u.m).value,
                         zs.to(u.m).value], axis=-1)   # (N, 3)
        ref_lon = np.radians(antennas[ref_index].loc.lon)
        ref_lat = np.radians(antennas[ref_index].loc.lat)
        ref_lon_qty = antennas[ref_index].loc.lon * u.deg
    elif ASTROPY_AVAILABLE and isinstance(first, EarthLocation):
        itrf = np.stack([
            np.array([a.x.to(u.m).value for a in antennas]),
            np.array([a.y.to(u.m).value for a in antennas]),
            np.array([a.z.to(u.m).value for a in antennas]),
        ], axis=-1)   # (N, 3)
        ref_lon = antennas[ref_index].lon.radian
        ref_lat = antennas[ref_index].lat.radian
        ref_lon_qty = antennas[ref_index].lon
    else:
        raise ValueError(
            "antennas must be a list of cysgp4.PyObserver or "
            "astropy.coordinates.EarthLocation objects"
        )

    # ------------------------------------------------------------------
    # Step 2: vectorised ITRF baseline subtraction  →  (N, 3)
    # ------------------------------------------------------------------
    baselines_itrf = itrf - itrf[ref_index]

    # ------------------------------------------------------------------
    # Step 3: one ITRF → ENU rotation applied to all baselines
    # ------------------------------------------------------------------
    baselines_enu = itrf_to_enu(baselines_itrf, ref_lon, ref_lat)   # (N, 3)

    # ------------------------------------------------------------------
    # Step 4: compute apparent LST and hour angle once for all times
    # ------------------------------------------------------------------
    lst = obs_times.sidereal_time('apparent', longitude=ref_lon_qty)
    ha = hour_angle(np.radians(ra_deg), lst.radian)

    # ------------------------------------------------------------------
    # Step 5: one ENU → UVW rotation, broadcast over all baselines × times
    # ------------------------------------------------------------------
    dec_rad = np.radians(dec_deg)
    if obs_times.isscalar:
        # ha is scalar; baselines_enu is (N, 3) → output (N, 3)
        uvw_all = enu_to_uvw(ha, dec_rad, baselines_enu)
    else:
        # ha has shape (T,); expand dims for broadcasting
        # baselines_enu: (N, 1, 3) × ha: (1, T) → output (N, T, 3)
        uvw_all = enu_to_uvw(
            ha[np.newaxis, :],
            dec_rad,
            baselines_enu[:, np.newaxis, :],
        )

    return uvw_all, ha
