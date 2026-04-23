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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

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
    from astropy.coordinates import AltAz, EarthLocation, ICRS, SkyCoord
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


def _broadcast_source_tracks(
    ra_deg: float | np.ndarray,
    dec_deg: float | np.ndarray,
    obs_times,
    ref_lon_qty,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Broadcast source coordinates against the observation-time axis.

    Parameters
    ----------
    ra_deg : float or array-like
        Right ascension in degrees. Scalars denote a fixed phase centre; array-
        like inputs must broadcast with *dec_deg*. When *obs_times* is non-
        scalar, the leading axis of any array input must match the number of
        time samples.
    dec_deg : float or array-like
        Declination in degrees. Scalars or arrays broadcastable with *ra_deg*.
    obs_times : astropy.time.Time
        Observation time(s) used to derive local sidereal time.
    ref_lon_qty : astropy.units.Quantity
        Reference longitude passed to ``Time.sidereal_time``.

    Returns
    -------
    ra_arr : numpy.ndarray
        Broadcast right-ascension array in degrees.
    dec_arr : numpy.ndarray
        Broadcast declination array in degrees.
    lst_arr : numpy.ndarray
        Apparent local sidereal time in radians, reshaped so the leading time
        axis broadcasts against ``ra_arr`` / ``dec_arr``.

    Raises
    ------
    ValueError
        If the source-track arrays do not share a compatible leading time axis
        with *obs_times*.
    """
    ra_arr, dec_arr = np.broadcast_arrays(
        np.asarray(ra_deg, dtype=np.float64),
        np.asarray(dec_deg, dtype=np.float64),
    )
    lst = np.asarray(
        obs_times.sidereal_time("apparent", longitude=ref_lon_qty).radian,
        dtype=np.float64,
    )

    if obs_times.isscalar or ra_arr.ndim == 0:
        return ra_arr, dec_arr, lst

    time_size = int(lst.shape[0])
    if ra_arr.shape[0] != time_size:
        raise ValueError(
            "Array-valued ra_deg/dec_deg must use the observation-time axis "
            "as their leading dimension. "
            f"Received leading dimension {ra_arr.shape[0]} for {time_size} "
            "time samples."
        )

    lst_arr = lst.reshape((time_size,) + (1,) * (ra_arr.ndim - 1))
    return ra_arr, dec_arr, lst_arr


def compute_uvw(
    antennas,
    ra_deg: float | np.ndarray,
    dec_deg: float | np.ndarray,
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
    ra_deg : float or array-like
        Right ascension of the target source (degrees, range [0, 360)).

        Scalars describe a fixed phase centre for the full observation.
        Array-valued inputs allow time-varying or multi-source tracks, provided
        they broadcast with *dec_deg*. When *obs_times* is non-scalar, any
        array input must use the observation-time axis as its leading
        dimension. For example:

        - ``(T,)`` for one time-varying phase centre,
        - ``(T, N_sat)`` for per-time satellite tracks.
    dec_deg : float or array-like
        Declination of the target source (degrees, range [−90, 90]). Must be
        scalar or broadcastable with *ra_deg*.
    obs_times : astropy.time.Time
        Observation times.  May be scalar or 1-D array.  Used to derive the
        local sidereal time (LST) and hence the hour angle.
    ref_index : int, optional
        Index of the reference antenna within *antennas* (default 0).
        ``uvw_all[ref_index]`` is always the zero vector.

    Returns
    -------
    uvw_all : numpy.ndarray
        UVW coordinates (metres) for all N antennas (baselines relative to
        ``antennas[ref_index]``). The output shape is
        ``(N,) + source_shape + (3,)`` where ``source_shape`` is:

        - empty for scalar *obs_times* and scalar coordinates,
        - ``(T,)`` for one phase centre sampled over T times,
        - ``(T, N_sat)`` for T samples of N_sat satellite tracks.

        - Axis 0 — antenna index; ``uvw_all[ref_index]`` is always zero.
        - Intermediate axes — time followed by any extra source axes.
        - Last axis — UVW components ``[u, v, w]``.
    hour_angles : float or numpy.ndarray
        Hour angle of the phase centre at each time/source sample (radians).
        Shape matches the broadcast source-coordinate shape described above.

    Raises
    ------
    ImportError
        If astropy is not installed (always required), or if pycraf is not
        installed when PyObserver inputs are used.
    ValueError
        If *antennas* is empty, contains mixed types, contains objects of an
        unrecognised type, or if array-valued source coordinates do not use the
        observation-time axis as their leading dimension.

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
       - Array time plus extra source axes → baselines expanded to
         ``(N, 1, ..., 1, 3)`` and broadcast against the full
         source-coordinate tensor.

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
    ra_arr, dec_arr, lst_radian = _broadcast_source_tracks(
        ra_deg,
        dec_deg,
        obs_times,
        ref_lon_qty,
    )
    ha = hour_angle(np.radians(ra_arr), lst_radian)

    # ------------------------------------------------------------------
    # Step 5: one ENU → UVW rotation, broadcast over all baselines × times
    # ------------------------------------------------------------------
    dec_rad = np.radians(dec_arr)
    if np.ndim(ha) == 0:
        # ha is scalar; baselines_enu is (N, 3) → output (N, 3)
        uvw_all = enu_to_uvw(ha, dec_rad, baselines_enu)
    else:
        # Expand the baseline matrix with one singleton source axis per
        # dimension in the source-coordinate tensor, then broadcast all
        # UVW rotations in a single vectorised call.
        baseline_expanded = baselines_enu[
            (slice(None),) + (np.newaxis,) * np.ndim(ha) + (slice(None),)
        ]
        uvw_all = enu_to_uvw(
            ha[np.newaxis, ...],
            dec_rad,
            baseline_expanded,
        )

    return uvw_all, ha


@dataclass(frozen=True, slots=True)
class AntennaArrayGeometry:
    """
    Parsed telescope-array geometry from an external coordinate file.

    Parameters
    ----------
    antenna_names : tuple of str
        Stable antenna identifiers in file order.
    longitudes_deg : numpy.ndarray, shape (N_ant,)
        Antenna geodetic longitudes in degrees, east-positive.
    latitudes_deg : numpy.ndarray, shape (N_ant,)
        Antenna geodetic latitudes in degrees, north-positive.
    altitudes_m : numpy.ndarray, shape (N_ant,)
        Antenna heights above the reference ellipsoid in metres.
    earth_locations : tuple of astropy.coordinates.EarthLocation
        Per-antenna EarthLocation objects used for UVW calculations.
    pyobservers : numpy.ndarray, shape (N_ant,), dtype object
        Matching ``cysgp4.PyObserver`` objects used for satellite propagation.

    Notes
    -----
    The parsed altitude is always stored in metres. ``pyobservers`` are created
    from the same file by converting altitude to kilometres for the
    ``PyObserver`` constructor, while ``earth_locations`` retain metres for
    Astropy frame transforms.
    """

    antenna_names: tuple[str, ...]
    longitudes_deg: np.ndarray
    latitudes_deg: np.ndarray
    altitudes_m: np.ndarray
    earth_locations: tuple[EarthLocation, ...]
    pyobservers: np.ndarray


@dataclass(frozen=True, slots=True)
class TrackingUvwResult:
    """
    UVW products for one tracked celestial pointing plus propagated satellites.

    Parameters
    ----------
    mjds : numpy.ndarray
        Original observation-time array passed to the builder.
    obs_times : astropy.time.Time
        Flattened UTC observation times derived from *mjds*.
    antenna_names : tuple of str
        Antenna identifiers in baseline axis order.
    satellite_names : tuple of str
        Satellite identifiers in satellite axis order.
    pointing_uvw_m : numpy.ndarray, shape (N_ant, T, 3)
        UVW coordinates for the requested fixed RA/Dec phase centre.
    pointing_hour_angles_rad : numpy.ndarray, shape (T,)
        Hour-angle track for the fixed phase centre.
    satellite_uvw_m : numpy.ndarray, shape (N_ant, T, N_sat, 3)
        UVW coordinates for each propagated satellite, using its time-varying
        ICRS RA/Dec as the phase centre.
    satellite_hour_angles_rad : numpy.ndarray, shape (T, N_sat)
        Satellite-specific hour-angle tracks in radians.
    satellite_ra_deg : numpy.ndarray, shape (T, N_sat)
        Satellite ICRS right ascension tracks in degrees.
    satellite_dec_deg : numpy.ndarray, shape (T, N_sat)
        Satellite ICRS declination tracks in degrees.
    satellite_separation_deg : numpy.ndarray, shape (T, N_sat)
        Angular separation between each satellite and the requested pointing in
        the reference antenna's local AltAz frame.

    Notes
    -----
    All UVW arrays use antenna 0, or the requested reference index, as the
    baseline origin. ``pointing_uvw_m[ref_index]`` and
    ``satellite_uvw_m[ref_index]`` are therefore identically zero.
    """

    mjds: np.ndarray
    obs_times: Time
    antenna_names: tuple[str, ...]
    satellite_names: tuple[str, ...]
    pointing_uvw_m: np.ndarray
    pointing_hour_angles_rad: np.ndarray
    satellite_uvw_m: np.ndarray
    satellite_hour_angles_rad: np.ndarray
    satellite_ra_deg: np.ndarray
    satellite_dec_deg: np.ndarray
    satellite_separation_deg: np.ndarray


_ARRAY_NAME_FIELDS = frozenset({"name", "antenna", "antenna_name", "station", "id"})
_ARRAY_LON_FIELDS = frozenset({"lon", "longitude", "lon_deg", "longitude_deg"})
_ARRAY_LAT_FIELDS = frozenset({"lat", "latitude", "lat_deg", "latitude_deg"})
_ARRAY_ALT_FIELDS = frozenset({"alt", "altitude", "height", "alt_m", "height_m"})


def _normalise_header_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _split_coordinate_fields(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return re.split(r"\s+", line.strip())
    return [field.strip() for field in line.split(delimiter)]


def _token_is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _resolve_array_column(
    header_map: dict[str, int],
    accepted_names: frozenset[str],
    column_label: str,
) -> int:
    for candidate in accepted_names:
        if candidate in header_map:
            return int(header_map[candidate])
    expected = ", ".join(sorted(accepted_names))
    raise ValueError(
        f"Array file is missing a {column_label} column. Expected one of: {expected}."
    )


def load_telescope_array_file(
    array_file: str | Path,
    *,
    altitude_unit: u.UnitBase | None = None,
) -> AntennaArrayGeometry:
    """
    Read an antenna-coordinate text file and build propagation/UVW objects.

    Parameters
    ----------
    array_file : str or pathlib.Path
        Path to the telescope-array definition. Supported formats are:

        - comma-separated text,
        - tab-separated text,
        - whitespace-delimited text.

        Blank lines and ``#`` comments are ignored. The file may either:

        1. include a header row with longitude/latitude/altitude column names,
           optionally plus a name/id column, or
        2. omit the header, in which case the first three columns are assumed to
           be ``lon_deg lat_deg alt`` and the optional fourth column is treated
           as the antenna name.
    altitude_unit : astropy.units.UnitBase, optional
        Physical unit of the altitude column. Defaults to metres. The parsed
        altitude is stored internally in metres and converted to kilometres when
        creating the corresponding ``cysgp4.PyObserver`` objects.

    Returns
    -------
    AntennaArrayGeometry
        Parsed antenna coordinates together with matching ``EarthLocation`` and
        ``PyObserver`` containers.

    Raises
    ------
    ImportError
        If astropy or cysgp4 is unavailable.
    FileNotFoundError
        If *array_file* does not exist.
    ValueError
        If the file is empty, malformed, or does not contain enough columns to
        infer longitude, latitude, and altitude.

    Notes
    -----
    Header names are matched case-insensitively after normalising punctuation.
    Accepted aliases are:

    - longitude: ``lon``, ``longitude``, ``lon_deg``, ``longitude_deg``
    - latitude: ``lat``, ``latitude``, ``lat_deg``, ``latitude_deg``
    - altitude: ``alt``, ``altitude``, ``height``, ``alt_m``, ``height_m``
    - optional name: ``name``, ``antenna``, ``antenna_name``, ``station``, ``id``
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required for load_telescope_array_file. "
            "Install with: pip install astropy"
        )
    if not CYSGP4_AVAILABLE:
        raise ImportError(
            "cysgp4 is required for load_telescope_array_file. "
            "Install with: pip install cysgp4"
        )

    path = Path(array_file)
    if not path.is_file():
        raise FileNotFoundError(f"Telescope array file not found: {path}")

    altitude_unit = u.m if altitude_unit is None else altitude_unit
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)

    if not lines:
        raise ValueError(f"Telescope array file '{path}' is empty.")

    first_line = lines[0]
    if "," in first_line:
        delimiter: str | None = ","
    elif "\t" in first_line:
        delimiter = "\t"
    else:
        delimiter = None

    first_fields = _split_coordinate_fields(first_line, delimiter)
    header_present = len(first_fields) < 3 or not all(
        _token_is_float(token) for token in first_fields[:3]
    )

    antenna_names: list[str] = []
    lons_deg: list[float] = []
    lats_deg: list[float] = []
    alts_m: list[float] = []

    if header_present:
        header_map = {
            _normalise_header_name(field): idx for idx, field in enumerate(first_fields)
        }
        name_idx = next(
            (header_map[field] for field in _ARRAY_NAME_FIELDS if field in header_map),
            None,
        )
        lon_idx = _resolve_array_column(header_map, _ARRAY_LON_FIELDS, "longitude")
        lat_idx = _resolve_array_column(header_map, _ARRAY_LAT_FIELDS, "latitude")
        alt_idx = _resolve_array_column(header_map, _ARRAY_ALT_FIELDS, "altitude")
        data_lines = lines[1:]
    else:
        name_idx = 3
        lon_idx, lat_idx, alt_idx = 0, 1, 2
        data_lines = lines

    if not data_lines:
        raise ValueError(f"Telescope array file '{path}' does not contain any data rows.")

    required_columns = max(lon_idx, lat_idx, alt_idx) + 1
    for row_idx, line in enumerate(data_lines):
        fields = _split_coordinate_fields(line, delimiter)
        if len(fields) < required_columns:
            raise ValueError(
                f"Row {row_idx + 1} in '{path}' has {len(fields)} columns but "
                f"at least {required_columns} are required."
            )
        antenna_names.append(
            fields[name_idx] if name_idx is not None and len(fields) > name_idx else f"ant{row_idx}"
        )
        lons_deg.append(float(fields[lon_idx]))
        lats_deg.append(float(fields[lat_idx]))
        alts_m.append(u.Quantity(float(fields[alt_idx]), altitude_unit).to_value(u.m))

    longitudes = np.asarray(lons_deg, dtype=np.float64)
    latitudes = np.asarray(lats_deg, dtype=np.float64)
    altitudes = np.asarray(alts_m, dtype=np.float64)
    earth_locations = tuple(
        EarthLocation(lon=lon * u.deg, lat=lat * u.deg, height=alt * u.m)
        for lon, lat, alt in zip(longitudes, latitudes, altitudes)
    )
    pyobservers = np.asarray(
        [
            cysgp4.PyObserver(float(lon), float(lat), float(alt / 1000.0))
            for lon, lat, alt in zip(longitudes, latitudes, altitudes)
        ],
        dtype=object,
    )

    return AntennaArrayGeometry(
        antenna_names=tuple(antenna_names),
        longitudes_deg=longitudes,
        latitudes_deg=latitudes,
        altitudes_m=altitudes,
        earth_locations=earth_locations,
        pyobservers=pyobservers,
    )


def load_tle_files(tle_files: str | Path | Sequence[str | Path]) -> np.ndarray:
    """
    Load one or more ASCII TLE files into a single ``PyTle`` object array.

    Parameters
    ----------
    tle_files : str, pathlib.Path, or sequence of either
        One or more plain-text TLE files accepted by
        ``scepter.obs.tle_ascii_to_pytles``.

    Returns
    -------
    numpy.ndarray
        One-dimensional object array of ``cysgp4.PyTle`` instances in file
        order. Multiple files are concatenated.

    Raises
    ------
    ValueError
        If *tle_files* is empty or if no TLEs were loaded.
    """
    if isinstance(tle_files, (str, Path)):
        tle_paths = (tle_files,)
    else:
        tle_paths = tuple(tle_files)

    if len(tle_paths) == 0:
        raise ValueError("tle_files cannot be empty.")

    from . import obs

    batches = [
        np.asarray(obs.tle_ascii_to_pytles(str(path)), dtype=object).ravel()
        for path in tle_paths
    ]
    if any(batch.size == 0 for batch in batches):
        raise ValueError("All TLE files must contain at least one valid TLE block.")

    return np.concatenate(batches)


def _extract_tle_names_from_text(tle_text: str) -> list[str]:
    lines = [
        line.strip()
        for line in tle_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    names: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("1 "):
            if idx + 1 >= len(lines) or not lines[idx + 1].startswith("2 "):
                raise ValueError("Malformed 2-line TLE block encountered while extracting names.")
            satnum = line[2:7].strip()
            names.append(f"sat_{satnum}" if satnum else f"sat{len(names)}")
            idx += 2
            continue
        if idx + 2 < len(lines) and lines[idx + 1].startswith("1 ") and lines[idx + 2].startswith("2 "):
            names.append(line)
            idx += 3
            continue
        idx += 1
    return names


def load_tle_files_with_names(
    tle_files: str | Path | Sequence[str | Path],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    Load ASCII TLE files together with stable satellite names.

    Parameters
    ----------
    tle_files : str, pathlib.Path, or sequence of either
        One or more plain-text TLE files.

    Returns
    -------
    tles : numpy.ndarray
        One-dimensional object array of ``cysgp4.PyTle`` instances.
    names : tuple of str
        Satellite names in the same order as *tles*. Three-line TLE blocks use
        their explicit name line; unnamed two-line blocks fall back to the line
        1 catalogue number.
    """
    if isinstance(tle_files, (str, Path)):
        tle_paths = (tle_files,)
    else:
        tle_paths = tuple(tle_files)

    if len(tle_paths) == 0:
        raise ValueError("tle_files cannot be empty.")

    from . import obs

    all_tles: list[np.ndarray] = []
    all_names: list[str] = []
    for path_like in tle_paths:
        path = Path(path_like)
        tle_text = path.read_text(encoding="utf-8")
        tles = np.asarray(obs.tle_ascii_to_pytles(str(path)), dtype=object).ravel()
        names = _extract_tle_names_from_text(tle_text)
        if len(names) != tles.size:
            names = [f"sat{len(all_names) + idx}" for idx in range(tles.size)]
        all_tles.append(tles)
        all_names.extend(names)

    return np.concatenate(all_tles), tuple(all_names)


def _default_tracking_mjds(
    *,
    epochs: int,
    cadence,
    trange,
    tint,
    startdate,
) -> np.ndarray:
    if not CYSGP4_AVAILABLE:
        raise ImportError(
            "cysgp4 is required for default time-grid generation. "
            "Install with: pip install cysgp4"
        )

    from . import skynet

    startdate = cysgp4.PyDateTime() if startdate is None else startdate
    return skynet.plantime(
        epochs=epochs,
        cadence=cadence,
        trange=trange,
        tint=tint,
        startdate=startdate,
    )


@dataclass(slots=True)
class TrackingUvwBuilder:
    """
    Build UVW products from a tracked RA/Dec, an array file, and TLE catalogs.

    Parameters
    ----------
    array_file : str or pathlib.Path
        Telescope-array coordinate file consumed by
        :func:`load_telescope_array_file`.
    tle_files : str, pathlib.Path, or sequence of either
        One or more ASCII TLE files consumed by :func:`load_tle_files`.
    mjds : numpy.ndarray
        Observation-time array in MJD. The shape is preserved for the returned
        metadata; UVW calculations flatten this to a one-dimensional time axis
        of length ``T = mjds.size``.
    ref_index : int, optional
        Reference antenna index for the UVW origin.
    altitude_unit : astropy.units.UnitBase, optional
        Unit used to interpret the array-file altitude column. Defaults to
        metres.
    d_rx : astropy.units.Quantity, optional
        Receiver diameter. Defaults to ``13.5 m``.
    eta_a_rx : float, optional
        Receiver aperture efficiency. Defaults to ``0.7``.
    freq : astropy.units.Quantity, optional
        Receiver centre frequency. Defaults to ``1420 MHz``.
    bandwidth : astropy.units.Quantity, optional
        Receiver bandwidth. Defaults to ``10 MHz``.
    elevation_limit_deg : float, optional
        Optional mean-elevation filter passed to ``obs.obs_sim.reduce_sats``.
        ``None`` keeps all propagated satellites.
    save_propagation : bool, optional
        If ``True``, let ``obs.obs_sim.populate`` save the propagation cache.
    propagation_save_path : str or pathlib.Path, optional
        Output path used only when ``save_propagation`` is ``True``.
    verbose : bool, optional
        Forwarded to ``obs.obs_sim.populate``.

    Notes
    -----
    The builder uses ``EarthLocation`` objects for the UVW transforms and
    ``PyObserver`` objects for ``cysgp4`` propagation so that the metres-based
    array file remains the single source of truth for antenna altitude.
    Celestial tracking and pointing-to-satellite angular separations are
    delegated to ``obs.obs_sim.sky_track`` and ``obs.obs_sim.sat_separation``
    to avoid duplicating that workflow inside ``uvw.py``.
    """

    array_file: str | Path
    tle_files: str | Path | Sequence[str | Path]
    mjds: np.ndarray
    ref_index: int = 0
    altitude_unit: u.UnitBase | None = None
    d_rx: u.Quantity | None = None
    eta_a_rx: float = 0.7
    freq: u.Quantity | None = None
    bandwidth: u.Quantity | None = None
    elevation_limit_deg: float | None = None
    save_propagation: bool = False
    propagation_save_path: str | Path | None = None
    verbose: bool = False

    def build(self, ra_deg: float, dec_deg: float) -> TrackingUvwResult:
        """
        Generate pointing and satellite UVW tracks for a fixed celestial target.

        Parameters
        ----------
        ra_deg : float
            Right ascension of the tracked phase centre in degrees.
        dec_deg : float
            Declination of the tracked phase centre in degrees.

        Returns
        -------
        TrackingUvwResult
            UVW arrays for the fixed pointing and every propagated satellite.
        """
        if not ASTROPY_AVAILABLE:
            raise ImportError(
                "astropy is required for TrackingUvwBuilder. "
                "Install with: pip install astropy"
            )
        if not CYSGP4_AVAILABLE:
            raise ImportError(
                "cysgp4 is required for TrackingUvwBuilder. "
                "Install with: pip install cysgp4"
            )

        from . import obs, skynet

        geometry = load_telescope_array_file(
            self.array_file,
            altitude_unit=self.altitude_unit,
        )
        tles, satellite_names_all = load_tle_files_with_names(self.tle_files)
        skygrid = skynet.pointgen_S_1586_1(niters=1)
        receiver = obs.receiver_info(
            d_rx=13.5 * u.m if self.d_rx is None else self.d_rx,
            eta_a_rx=self.eta_a_rx,
            pyobs=geometry.pyobservers,
            freq=1420 * u.MHz if self.freq is None else self.freq,
            bandwidth=10 * u.MHz if self.bandwidth is None else self.bandwidth,
        )
        sim = obs.obs_sim(receiver, skygrid, self.mjds)
        save_name = (
            "satellite_info.npz"
            if self.propagation_save_path is None
            else str(self.propagation_save_path)
        )
        sim.populate(
            tles,
            save=self.save_propagation,
            savename=save_name,
            verbose=self.verbose,
        )
        visible_names = satellite_names_all
        if self.elevation_limit_deg is not None:
            sim.reduce_sats(el_limit=self.elevation_limit_deg)
            mask = np.asarray(sim.elevation_mask, dtype=bool)
            visible_names = tuple(name for name, keep in zip(satellite_names_all, mask) if keep)

        obs_times = Time(np.asarray(self.mjds, dtype=np.float64).ravel(), format="mjd", scale="utc")
        sim.sky_track(ra_deg, dec_deg, observer_index=self.ref_index)
        satellite_separation_deg = np.asarray(
            sim.sat_separation(mode="tracking")[self.ref_index, 0, 0, 0, :, :].to_value(u.deg),
            dtype=np.float64,
        )
        antennas = list(geometry.earth_locations)
        pointing_uvw, pointing_ha = compute_uvw(
            antennas,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            obs_times=obs_times,
            ref_index=self.ref_index,
        )

        ref_location = sim.altaz_frame.location
        topo_az = np.asarray(sim.topo_pos_az[self.ref_index, 0, 0, 0, :, :], dtype=np.float64)
        topo_el = np.asarray(sim.topo_pos_el[self.ref_index, 0, 0, 0, :, :], dtype=np.float64)
        sat_altaz = SkyCoord(
            az=topo_az * u.deg,
            alt=topo_el * u.deg,
            frame=AltAz(obstime=obs_times[:, np.newaxis], location=ref_location),
        )
        sat_icrs = sat_altaz.transform_to(ICRS())
        satellite_ra_deg = np.asarray(sat_icrs.ra.deg, dtype=np.float64)
        satellite_dec_deg = np.asarray(sat_icrs.dec.deg, dtype=np.float64)
        satellite_uvw, satellite_ha = compute_uvw(
            antennas,
            ra_deg=satellite_ra_deg,
            dec_deg=satellite_dec_deg,
            obs_times=obs_times,
            ref_index=self.ref_index,
        )
        return TrackingUvwResult(
            mjds=np.asarray(self.mjds, dtype=np.float64),
            obs_times=obs_times,
            antenna_names=geometry.antenna_names,
            satellite_names=visible_names,
            pointing_uvw_m=np.asarray(pointing_uvw, dtype=np.float64),
            pointing_hour_angles_rad=np.asarray(pointing_ha, dtype=np.float64),
            satellite_uvw_m=np.asarray(satellite_uvw, dtype=np.float64),
            satellite_hour_angles_rad=np.asarray(satellite_ha, dtype=np.float64),
            satellite_ra_deg=satellite_ra_deg,
            satellite_dec_deg=satellite_dec_deg,
            satellite_separation_deg=satellite_separation_deg,
        )


def build_tracking_uvw(
    *,
    ra_deg: float,
    dec_deg: float,
    array_file: str | Path,
    tle_files: str | Path | Sequence[str | Path],
    mjds: np.ndarray | None = None,
    epochs: int = 1,
    cadence=None,
    trange=None,
    tint=None,
    startdate=None,
    ref_index: int = 0,
    altitude_unit: u.UnitBase | None = None,
    d_rx: u.Quantity | None = None,
    eta_a_rx: float = 0.7,
    freq: u.Quantity | None = None,
    bandwidth: u.Quantity | None = None,
    elevation_limit_deg: float | None = None,
    save_propagation: bool = False,
    propagation_save_path: str | Path | None = None,
    verbose: bool = False,
) -> TrackingUvwResult:
    """
    Convenience wrapper for :class:`TrackingUvwBuilder`.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Fixed tracking coordinates in ICRS degrees.
    array_file : str or pathlib.Path
        Telescope-array coordinate file.
    tle_files : str, pathlib.Path, or sequence of either
        One or more ASCII TLE files.
    mjds : numpy.ndarray, optional
        Explicit observation-time grid in MJD. If omitted, a default grid is
        generated with ``epochs=1``, ``cadence=24 h``, ``trange=3600 s``, and
        ``tint=1 s`` unless overridden.
    epochs, cadence, trange, tint, startdate
        Parameters forwarded to ``scepter.skynet.plantime`` when *mjds* is not
        supplied.
    ref_index, altitude_unit, d_rx, eta_a_rx, freq, bandwidth,
    elevation_limit_deg, save_propagation, propagation_save_path, verbose
        Passed through to :class:`TrackingUvwBuilder`.

    Returns
    -------
    TrackingUvwResult
        Pointing and satellite UVW products for the requested observation.
    """
    if mjds is None:
        if not ASTROPY_AVAILABLE:
            raise ImportError(
                "astropy is required for build_tracking_uvw. "
                "Install with: pip install astropy"
            )
        cadence = 24 * u.hour if cadence is None else cadence
        trange = 3600 * u.s if trange is None else trange
        tint = 1 * u.s if tint is None else tint
        mjds = _default_tracking_mjds(
            epochs=epochs,
            cadence=cadence,
            trange=trange,
            tint=tint,
            startdate=startdate,
        )

    builder = TrackingUvwBuilder(
        array_file=array_file,
        tle_files=tle_files,
        mjds=np.asarray(mjds, dtype=np.float64),
        ref_index=ref_index,
        altitude_unit=altitude_unit,
        d_rx=d_rx,
        eta_a_rx=eta_a_rx,
        freq=freq,
        bandwidth=bandwidth,
        elevation_limit_deg=elevation_limit_deg,
        save_propagation=save_propagation,
        propagation_save_path=propagation_save_path,
        verbose=verbose,
    )
    return builder.build(ra_deg=ra_deg, dec_deg=dec_deg)
