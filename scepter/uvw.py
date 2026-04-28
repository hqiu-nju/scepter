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

- `scepter.uvw.baseline_bearing()` → provides ITRF baseline vectors
- `scepter.uvw.baseline_pairs()` → computes all baselines in an array
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
scepter.uvw.baseline_bearing : Calculate ITRF baseline between two antennas
scepter.uvw.baseline_pairs : Calculate all baselines in an array
scepter.uvw.geometric_delay_az_el : Calculate geometric delay from Az/El

Author: Generated for SCEPTer package
Date Created: 2026-02-06
Version: 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

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


def _as_antenna_list(antennas) -> list:
    antennas_array = np.atleast_1d(np.asarray(antennas, dtype=object))
    return list(antennas_array.ravel())


def _antenna_itrf_positions_m(antennas) -> np.ndarray:
    antenna_list = _as_antenna_list(antennas)
    if len(antenna_list) == 0:
        raise ValueError("antennas cannot be empty.")

    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required for antenna baseline calculations. "
            "Install with: pip install astropy"
        )

    first = antenna_list[0]
    if CYSGP4_AVAILABLE and isinstance(first, cysgp4.PyObserver):
        if not PYCRAF_AVAILABLE:
            raise ImportError(
                "pycraf is required when using PyObserver antennas. "
                "Install with: pip install pycraf"
            )
        if not all(isinstance(antenna, cysgp4.PyObserver) for antenna in antenna_list):
            raise ValueError("antennas must not mix PyObserver and other location types.")
        lons = np.asarray([antenna.loc.lon for antenna in antenna_list], dtype=np.float64) * u.deg
        lats = np.asarray([antenna.loc.lat for antenna in antenna_list], dtype=np.float64) * u.deg
        alts = np.asarray([antenna.loc.alt for antenna in antenna_list], dtype=np.float64) * u.m
        xs, ys, zs = geospatial.wgs84_to_itrf2008(lons, lats, alts)
        return np.stack(
            [xs.to_value(u.m), ys.to_value(u.m), zs.to_value(u.m)],
            axis=-1,
        )

    if ASTROPY_AVAILABLE and isinstance(first, EarthLocation):
        if not all(isinstance(antenna, EarthLocation) for antenna in antenna_list):
            raise ValueError("antennas must not mix EarthLocation and other location types.")
        return np.stack(
            [
                np.asarray([antenna.x.to_value(u.m) for antenna in antenna_list], dtype=np.float64),
                np.asarray([antenna.y.to_value(u.m) for antenna in antenna_list], dtype=np.float64),
                np.asarray([antenna.z.to_value(u.m) for antenna in antenna_list], dtype=np.float64),
            ],
            axis=-1,
        )

    raise ValueError(
        "antennas must contain cysgp4.PyObserver or astropy.coordinates.EarthLocation objects."
    )


def baseline_bearing(ref, ant) -> tuple[np.ndarray, float]:
    """
    Calculate one ITRF baseline vector and its Euclidean length.

    Parameters
    ----------
    ref : cysgp4.PyObserver or astropy.coordinates.EarthLocation
        Reference antenna used as the baseline origin.
    ant : cysgp4.PyObserver or astropy.coordinates.EarthLocation
        Target antenna. The type must match *ref*.

    Returns
    -------
    bearing : numpy.ndarray, shape (3,)
        ITRF Cartesian baseline vector from *ref* to *ant* in metres.
    distance : float
        Baseline length in metres.

    Raises
    ------
    ImportError
        If Astropy is unavailable, or if PyObserver inputs are used without
        pycraf.
    ValueError
        If the inputs are empty, mixed-type, or unsupported location objects.

    Notes
    -----
    For ``cysgp4.PyObserver`` inputs, this uses the same WGS84 to ITRF2008
    conversion as ``compute_uvw``. For ``EarthLocation`` inputs, geocentric
    ``x``, ``y``, and ``z`` coordinates are used directly.
    """
    bearings, distances = baseline_pairs([ref, ant])
    return np.asarray(bearings[1], dtype=np.float64), float(distances[1])


def baseline_pairs(antennas, ref_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate ITRF baseline vectors from one reference antenna to an array.

    Parameters
    ----------
    antennas : array-like
        One or more ``cysgp4.PyObserver`` or ``astropy.coordinates.EarthLocation``
        objects. All entries must have the same type.
    ref_index : int, optional
        Antenna index used as the baseline origin. Default is 0.

    Returns
    -------
    bearings : numpy.ndarray, shape (N_ant, 3)
        ITRF Cartesian baseline vectors in metres. ``bearings[ref_index]`` is
        the zero vector.
    baselines : numpy.ndarray, shape (N_ant,)
        Euclidean baseline lengths in metres.

    Raises
    ------
    ImportError
        If required coordinate packages are unavailable.
    IndexError
        If *ref_index* does not select an antenna.
    ValueError
        If *antennas* is empty or contains unsupported/mixed location objects.

    Notes
    -----
    This helper intentionally returns baselines relative to one reference
    antenna, matching the historical ``obs.baseline_pairs`` behaviour and the
    baseline axis used by ``compute_uvw``.
    """
    itrf = _antenna_itrf_positions_m(antennas)
    index = int(ref_index)
    if index < 0 or index >= itrf.shape[0]:
        raise IndexError(f"ref_index {index} is out of range for {itrf.shape[0]} antennas.")
    bearings = itrf - itrf[index]
    baselines = np.linalg.norm(bearings, axis=1)
    return bearings, baselines


def _angle_array_to_rad(values, *, input_unit: u.UnitBase) -> np.ndarray:
    if hasattr(values, "to_value"):
        return np.asarray(values.to_value(u.rad), dtype=np.float64)
    return np.asarray(u.Quantity(values, input_unit).to_value(u.rad), dtype=np.float64)


def geometric_delay_az_el(
    baselines_itrf,
    az,
    el,
    ref_lon_rad,
    ref_lat_rad,
    *,
    angle_unit: u.UnitBase = u.deg,
):
    """
    Calculate far-field geometric delay for topocentric Az/El directions.

    Parameters
    ----------
    baselines_itrf : array-like or astropy.Quantity, shape (N_ant, 3)
        Reference-relative ITRF baseline vectors. Non-quantity inputs are
        interpreted as metres. The vector convention is the same as
        ``baseline_pairs``: each row points from the reference antenna to the
        target antenna.
    az : float or array-like or astropy.Quantity
        Topocentric azimuth of the source. Quantity inputs may carry any angle
        unit. Non-quantity inputs use *angle_unit*.
    el : float or array-like or astropy.Quantity
        Topocentric elevation of the source. Quantity inputs may carry any
        angle unit. Non-quantity inputs use *angle_unit*.
    ref_lon_rad : float
        Reference antenna geodetic longitude in radians.
    ref_lat_rad : float
        Reference antenna geodetic latitude in radians.
    angle_unit : astropy.units.UnitBase, optional
        Unit for non-quantity *az* and *el* inputs. Defaults to degrees to
        match SCEPTer sky-grid arrays.

    Returns
    -------
    delay : astropy.Quantity
        Far-field geometric delay in seconds. The shape is ``(N_ant,)`` plus
        the broadcast shape of *az* and *el*.

    Raises
    ------
    ValueError
        If the baseline vectors do not have a final dimension of length 3, or
        if *az* and *el* cannot be broadcast together.

    Notes
    -----
    The calculation uses the UVW coordinate pipeline's local-frame conversion:
    ITRF baselines are rotated to ENU with ``itrf_to_enu`` and then projected
    onto the topocentric unit vector
    ``[east, north, up] = [cos(el) sin(az), cos(el) cos(az), sin(el)]``.

    The sign convention is ``dot(baseline_enu, source_unit) / c``. For a
    distant source this matches the leading path-length difference
    ``(l_ref - l_target) / c`` used by ``baseline_nearfield_delay``.
    """
    if hasattr(baselines_itrf, "to_value"):
        baselines_m = np.asarray(baselines_itrf.to_value(u.m), dtype=np.float64)
    else:
        baselines_m = np.asarray(baselines_itrf, dtype=np.float64)
    if baselines_m.shape[-1] != 3:
        raise ValueError("baselines_itrf must have a final dimension of length 3.")

    baselines_enu = itrf_to_enu(
        baselines_m,
        float(ref_lon_rad),
        float(ref_lat_rad),
    )
    az_rad, el_rad = np.broadcast_arrays(
        _angle_array_to_rad(az, input_unit=angle_unit),
        _angle_array_to_rad(el, input_unit=angle_unit),
    )
    source_enu = np.stack(
        [
            np.cos(el_rad) * np.sin(az_rad),
            np.cos(el_rad) * np.cos(az_rad),
            np.sin(el_rad),
        ],
        axis=-1,
    )
    expanded_baselines = baselines_enu[
        (slice(None),) + (np.newaxis,) * az_rad.ndim + (slice(None),)
    ]
    path_m = np.sum(expanded_baselines * source_enu[np.newaxis, ...], axis=-1)
    return path_m * u.m / (3e8 * u.m / u.s)


def baseline_nearfield_delay(l1, l2, tau):
    """
    Apply near-field path-length correction to a far-field delay.

    Parameters
    ----------
    l1 : astropy.Quantity
        Distance from source to the reference antenna.
    l2 : astropy.Quantity
        Distance from source to each target antenna.
    tau : astropy.Quantity
        Far-field delay to subtract.

    Returns
    -------
    delay : astropy.Quantity
        Corrected delay in seconds, computed as ``(l1 - l2) / c - tau``.

    Raises
    ------
    AttributeError
        If inputs are not Astropy quantities with ``to`` methods.

    Notes
    -----
    This helper is used by ``obs_sim.baselines_nearfield_delays`` for LEO
    satellite fringe calculations where wavefront curvature matters.
    """
    c = 3e8 * u.m / u.s
    return (l1.to(u.m) - l2.to(u.m)) / c - tau


def fringe_attenuation(theta, baseline, bandwidth):
    """
    Estimate finite-bandwidth fringe attenuation for an angular offset.

    Parameters
    ----------
    theta : astropy.Quantity
        Angular offset from phase centre.
    baseline : astropy.Quantity
        Baseline length.
    bandwidth : astropy.Quantity
        Observing bandwidth.

    Returns
    -------
    attenuation : float or numpy.ndarray
        Dimensionless sinc attenuation factor.

    Raises
    ------
    AttributeError
        If any input is not an Astropy quantity with a ``to`` method.

    Notes
    -----
    The model is ``sinc(sin(theta) * baseline * bandwidth / c)`` and matches
    the historical ``obs.fringe_attenuation`` implementation.
    """
    c = 3e8
    theta = theta.to(u.rad).value
    baseline = baseline.to(u.m).value
    bandwidth = bandwidth.to(u.Hz).value
    return np.sinc(np.sin(theta) * baseline * bandwidth / c)


def fringe_response(delay, frequency):
    """
    Calculate monochromatic interferometric fringe response.

    Parameters
    ----------
    delay : astropy.Quantity
        Geometric delay.
    frequency : astropy.Quantity
        Observing frequency.

    Returns
    -------
    response : float or numpy.ndarray
        Dimensionless cosine fringe response.

    Raises
    ------
    AttributeError
        If inputs are not Astropy quantities with ``to`` methods.

    Notes
    -----
    The returned response is ``cos(2*pi*frequency*delay)``.
    """
    delay = delay.to(u.s).value
    frequency = frequency.to(u.Hz).value
    return np.cos(2 * np.pi * frequency * delay)


def _quantity_to_value(value: Any, unit: Any, assumed_unit: str) -> np.ndarray:
    if hasattr(value, "to_value"):
        if not ASTROPY_AVAILABLE:
            raise ImportError(
                f"Astropy is required to convert quantity inputs to {assumed_unit}."
            )
        return np.asarray(value.to_value(unit), dtype=np.float64)
    if hasattr(value, "to"):
        if not ASTROPY_AVAILABLE:
            raise ImportError(
                f"Astropy is required to convert quantity inputs to {assumed_unit}."
            )
        return np.asarray(value.to(unit).value, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _as_hz(value: Any) -> np.ndarray:
    return _quantity_to_value(value, u.Hz if ASTROPY_AVAILABLE else None, "Hz")


def _as_m(value: Any) -> np.ndarray:
    return _quantity_to_value(value, u.m if ASTROPY_AVAILABLE else None, "m")


def _as_s(value: Any) -> np.ndarray:
    return _quantity_to_value(value, u.s if ASTROPY_AVAILABLE else None, "s")


def _broadcast_to_with_trailing_axes(
    value: np.ndarray,
    target_shape: tuple[int, ...],
    label: str,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    try:
        return np.broadcast_to(arr, target_shape)
    except ValueError:
        while arr.ndim < len(target_shape):
            arr = np.expand_dims(arr, axis=-1)
        try:
            return np.broadcast_to(arr, target_shape)
        except ValueError as second_error:
            raise ValueError(
                f"{label} cannot be broadcast to target shape {target_shape}."
            ) from second_error


def pointing_geometric_delay(pointing_uvw_m: np.ndarray) -> np.ndarray:
    """
    Convert tracked-pointing UVW coordinates to applied geometric delays.

    Parameters
    ----------
    pointing_uvw_m : numpy.ndarray, shape (..., 3)
        UVW coordinates for the tracked phase centre in metres. The last axis
        must be ``[u, v, w]``.

    Returns
    -------
    delay_s : numpy.ndarray
        Far-field geometric delay in seconds, computed as ``w / c``.

    Raises
    ------
    ValueError
        If *pointing_uvw_m* does not end with a 3-component UVW axis.

    Notes
    -----
    The sign convention matches :func:`geometric_delay_az_el`, so the result
    can be subtracted from satellite range delays using
    :func:`satellite_geometric_delay`.
    """
    pointing_uvw = _as_m(pointing_uvw_m)
    if pointing_uvw.shape[-1:] != (3,):
        raise ValueError("pointing_uvw_m must end with a 3-component UVW axis.")
    return pointing_uvw[..., 2] / 3e8


def satellite_geometric_delay(
    satellite_distances,
    pointing_delay_s,
    *,
    ref_index: int = 0,
) -> np.ndarray:
    """
    Compute residual satellite delays after subtracting tracking delays.

    Parameters
    ----------
    satellite_distances : astropy.units.Quantity or numpy.ndarray
        Distance from each antenna to each satellite. Quantities are converted
        to metres. Plain numeric values are interpreted as metres. Axis 0 must
        be the antenna/baseline axis, for example ``(N_ant, T, N_sat)``.
    pointing_delay_s : astropy.units.Quantity or numpy.ndarray
        Far-field geometric delays applied for the telescope pointing, normally
        from :func:`pointing_geometric_delay`. Shape must broadcast to
        ``satellite_distances.shape``; missing trailing satellite/source axes
        are inserted automatically.
    ref_index : int, optional
        Reference antenna index. Default is 0.

    Returns
    -------
    delay_s : numpy.ndarray
        Residual delay in seconds with the same shape as
        ``satellite_distances``:

        ``(range_ref - range_ant) / c - pointing_delay``

    Raises
    ------
    IndexError
        If *ref_index* is outside the antenna axis.
    ValueError
        If *satellite_distances* has no antenna axis or if *pointing_delay_s*
        cannot be broadcast to the satellite-distance shape.

    Notes
    -----
    This is the same near-field convention used by
    :func:`baseline_nearfield_delay` and ``obs.obs_sim.baselines_nearfield_delays``.
    It uses actual antenna-to-satellite ranges, so it captures finite-distance
    curvature that is not present in a far-field satellite UVW projection.
    """
    distances_m = _as_m(satellite_distances)
    if distances_m.ndim == 0:
        raise ValueError("satellite_distances must include an antenna axis.")
    index = int(ref_index)
    if index < 0:
        index += distances_m.shape[0]
    if index < 0 or index >= distances_m.shape[0]:
        raise IndexError(
            f"ref_index {ref_index} is out of range for {distances_m.shape[0]} antennas."
        )

    ref_distances_m = np.take(distances_m, index, axis=0)[np.newaxis, ...]
    range_delay_s = (ref_distances_m - distances_m) / 3e8
    pointing_delay = _broadcast_to_with_trailing_axes(
        _as_s(pointing_delay_s),
        range_delay_s.shape,
        "pointing_delay_s",
    )
    return range_delay_s - pointing_delay


def satellite_visibility_phase(
    pointing_uvw_m: np.ndarray,
    satellite_distances,
    frequency,
    *,
    ref_index: int = 0,
    phase_sign: float = -1.0,
    wrap: bool = True,
) -> np.ndarray:
    """
    Compute near-field satellite visibility phase relative to a pointing.

    The phase is derived from actual antenna-to-satellite ranges, then the
    far-field geometric delays applied for the current telescope pointing are
    subtracted:

    ``delay = (range_ref - range_ant) / c - pointing_delay``

    Parameters
    ----------
    pointing_uvw_m : numpy.ndarray, shape (..., 3)
        UVW coordinates for the tracked phase centre in metres. The ``w``
        component is converted to the pointing delay applied by the telescope.
    satellite_distances : astropy.units.Quantity or numpy.ndarray
        Distance from each antenna to each satellite. Quantities are converted
        to metres; plain numeric values are interpreted as metres. A common
        layout is ``(N_ant, T, N_sat)``.
    frequency : astropy.units.Quantity or float or array-like
        Observing frequency. Quantities are converted to hertz. Plain numeric
        values are interpreted as hertz. A scalar returns a phase array with
        shape ``satellite_distances.shape``; an array-valued frequency always
        appends trailing frequency axis/axes.
    ref_index : int, optional
        Reference antenna index for the range-difference calculation.
    phase_sign : float, optional
        Sign convention applied to the phase. The default, ``-1``, follows the
        common radio-interferometry convention ``exp(-2*pi*i*nu*tau)``.
    wrap : bool, optional
        If ``True`` (default), wrap phases to ``[-pi, pi]``.

    Returns
    -------
    phase_rad : numpy.ndarray
        Visibility phase in radians.

    Raises
    ------
    ValueError
        If ``pointing_uvw_m`` does not have a final component axis of length 3
        or if the pointing delays cannot be broadcast against
        ``satellite_distances``.

    Notes
    -----
    This helper assumes the pointing UVW coordinates and satellite distances
    use the same antenna axis and time grid. Use
    ``TrackingUvwResult.satellite_distance_m`` or ``obs_sim.topo_pos_dist *
    u.km`` for range inputs from SCEPTer propagation.
    """
    delay_s = satellite_geometric_delay(
        satellite_distances,
        pointing_geometric_delay(pointing_uvw_m),
        ref_index=ref_index,
    )
    frequency_hz = _as_hz(frequency)
    if np.ndim(frequency_hz) == 0:
        phase = phase_sign * 2.0 * np.pi * frequency_hz * delay_s
    else:
        expand = (np.newaxis,) * np.ndim(frequency_hz)
        phase = phase_sign * 2.0 * np.pi * delay_s[(...,) + expand] * frequency_hz

    if wrap:
        phase = np.angle(np.exp(1j * phase))
    return np.asarray(phase, dtype=np.float64)


def normalised_visibility_amplitude(
    visibilities: np.ndarray,
    *,
    reference: float | None = None,
) -> np.ndarray:
    """
    Convert complex visibilities to a dimensionless amplitude in ``[0, 1]``.

    Parameters
    ----------
    visibilities : numpy.ndarray
        Complex visibility samples. Any shape is accepted.
    reference : float, optional
        Amplitude used for normalisation. If omitted, the largest finite
        absolute value in *visibilities* is used. Empty inputs, all-zero inputs,
        and all-non-finite inputs return zeros.

    Returns
    -------
    amplitude : numpy.ndarray
        ``abs(visibilities) / reference`` clipped to ``[0, 1]``.

    Raises
    ------
    ValueError
        If *reference* is supplied and is negative.

    Notes
    -----
    The British spelling is used to match SCEPTer documentation. The function
    intentionally does not modify phases or replace non-finite complex samples;
    non-finite amplitudes are returned as zero after normalisation.
    """
    amp = np.abs(np.asarray(visibilities))
    finite_amp = np.where(np.isfinite(amp), amp, 0.0)
    if reference is None:
        ref = float(np.max(finite_amp)) if finite_amp.size else 0.0
    else:
        ref = float(reference)
        if ref < 0.0:
            raise ValueError("reference must be non-negative.")

    if ref == 0.0:
        return np.zeros_like(finite_amp, dtype=np.float64)
    return np.clip(finite_amp / ref, 0.0, 1.0)


def simulate_satellite_visibilities(
    pointing_uvw_m: np.ndarray,
    satellite_distances,
    frequency,
    *,
    bandwidth=None,
    channel_samples: int = 1,
    source_amplitude: float | np.ndarray | None = None,
    visibility_mask: np.ndarray | None = None,
    ref_index: int = 0,
    phase_sign: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate complex satellite visibilities from range-delay geometry.

    Parameters
    ----------
    pointing_uvw_m : numpy.ndarray, shape (..., 3)
        UVW coordinates for the tracked phase centre in metres. The ``w``
        component is converted to the pointing delay applied by the telescope.
    satellite_distances : astropy.units.Quantity or numpy.ndarray
        Distance from each antenna to each satellite. Quantities are converted
        to metres; plain numeric values are interpreted as metres. The common
        layout is ``(N_ant, T, N_sat)``.
    frequency : astropy.units.Quantity or float
        Channel centre frequency. Quantities are converted to hertz; plain
        numeric values are interpreted as hertz.
    bandwidth : astropy.units.Quantity or float, optional
        Channel bandwidth. If omitted, visibilities are monochromatic. If
        supplied, phases are sampled linearly across the channel and averaged.
    channel_samples : int, optional
        Number of frequency samples used when *bandwidth* is supplied. Default
        is 1. Values below 1 raise ``ValueError``.
    source_amplitude : float or numpy.ndarray, optional
        Optional relative satellite amplitudes. Values are broadcast against
        the unit complex visibility array. If omitted, unit-amplitude point
        sources are assumed.
    visibility_mask : numpy.ndarray, optional
        Boolean mask broadcastable to the visibility shape. Masked samples are
        set to zero complex visibility and zero normalised amplitude.
    ref_index : int, optional
        Reference antenna index for the range-difference calculation.
    phase_sign : float, optional
        Sign convention passed to :func:`satellite_visibility_phase`.

    Returns
    -------
    visibilities : numpy.ndarray
        Complex visibility samples. Shape follows ``satellite_distances.shape``
        for scalar frequency, with a trailing frequency axis when an array of
        frequencies is supplied and *bandwidth* is omitted.
    phase_rad : numpy.ndarray
        Wrapped visibility phase in radians, derived from
        ``np.angle(visibilities)``.
    normalised_amplitude : numpy.ndarray
        Dimensionless amplitude normalised to the maximum finite visibility
        amplitude in this simulation.

    Raises
    ------
    ValueError
        If *channel_samples* is less than 1, if *frequency* is non-scalar while
        *bandwidth* is supplied, or if amplitudes/masks cannot be broadcast to
        the visibility shape.

    Notes
    -----
    This is a visibility-domain point-source model for satellites. It models
    range-based near-field geometric phase and optional finite-channel
    averaging; it does not include satellite EIRP, receive antenna gain,
    propagation loss, or receiver noise. Use *source_amplitude* to inject
    externally computed relative amplitudes such as gain- or power-weighted
    samples.
    """
    samples = int(channel_samples)
    if samples < 1:
        raise ValueError("channel_samples must be at least 1.")

    if bandwidth is None:
        phase = satellite_visibility_phase(
            pointing_uvw_m,
            satellite_distances,
            frequency,
            ref_index=ref_index,
            phase_sign=phase_sign,
            wrap=False,
        )
        unit_vis = np.exp(1j * phase)
    else:
        centre_hz = _as_hz(frequency)
        if np.ndim(centre_hz) != 0:
            raise ValueError("frequency must be scalar when bandwidth is supplied.")
        bandwidth_hz = _as_hz(bandwidth)
        if np.ndim(bandwidth_hz) != 0:
            raise ValueError("bandwidth must be scalar.")
        if samples == 1:
            freq_samples_hz = np.asarray([float(centre_hz)], dtype=np.float64)
        else:
            freq_samples_hz = np.linspace(
                float(centre_hz) - float(bandwidth_hz) * 0.5,
                float(centre_hz) + float(bandwidth_hz) * 0.5,
                samples,
            )
        phase_samples = satellite_visibility_phase(
            pointing_uvw_m,
            satellite_distances,
            freq_samples_hz,
            ref_index=ref_index,
            phase_sign=phase_sign,
            wrap=False,
        )
        unit_vis = np.mean(np.exp(1j * phase_samples), axis=-1)

    if source_amplitude is None:
        vis = unit_vis
    else:
        amplitude = _broadcast_to_with_trailing_axes(
            np.asarray(source_amplitude, dtype=np.float64),
            unit_vis.shape,
            "source_amplitude",
        )
        vis = unit_vis * amplitude

    if visibility_mask is not None:
        mask = _broadcast_to_with_trailing_axes(
            np.asarray(visibility_mask, dtype=bool),
            vis.shape,
            "visibility_mask",
            dtype=bool,
        )
        vis = np.where(mask, vis, 0.0 + 0.0j)

    vis = np.asarray(vis, dtype=np.complex128)
    return vis, np.angle(vis), normalised_visibility_amplitude(vis)


@dataclass(frozen=True, slots=True)
class VisibilityNpzArchive:
    """
    Complex visibility archive loaded from a SCEPTer NPZ file.

    Parameters
    ----------
    path : pathlib.Path
        Source archive path.
    visibilities : numpy.ndarray
        Complex visibility samples from the ``vis`` key.
    uvw_m : numpy.ndarray
        UVW coordinates in metres from the ``uvw`` key.
    frequency_hz : numpy.ndarray or None
        Frequency array from ``freq_hz`` if present.
    pointing_uvw_m, satellite_uvw_m : numpy.ndarray or None
        Optional pointing and satellite UVW arrays.
    satellite_distance_m : numpy.ndarray or None
        Optional per-antenna satellite ranges in metres.
    phase_rad : numpy.ndarray or None
        Optional wrapped phase samples.
    normalised_amplitude : numpy.ndarray or None
        Optional normalised amplitude samples.
    antenna_names, satellite_names : tuple of str
        Optional names stored as string arrays.
    mjds : numpy.ndarray or None
        Optional observation times in MJD.
    metadata : dict
        Optional JSON metadata stored under ``metadata_json``.

    Notes
    -----
    ``vis`` and ``uvw`` are the required stable keys. Additional arrays are
    intentionally optional so the same reader can consume compact imaging
    products and fuller simulation products.
    """

    path: Path
    visibilities: np.ndarray
    uvw_m: np.ndarray
    frequency_hz: np.ndarray | None
    pointing_uvw_m: np.ndarray | None
    satellite_uvw_m: np.ndarray | None
    satellite_distance_m: np.ndarray | None
    phase_rad: np.ndarray | None
    normalised_amplitude: np.ndarray | None
    antenna_names: tuple[str, ...]
    satellite_names: tuple[str, ...]
    mjds: np.ndarray | None
    metadata: dict[str, Any]


def save_visibility_npz(
    path: str | Path,
    visibilities: np.ndarray,
    uvw_m: np.ndarray,
    *,
    frequency=None,
    pointing_uvw_m: np.ndarray | None = None,
    satellite_uvw_m: np.ndarray | None = None,
    satellite_distance_m: np.ndarray | None = None,
    phase_rad: np.ndarray | None = None,
    normalised_amplitude: np.ndarray | None = None,
    antenna_names: Sequence[str] | None = None,
    satellite_names: Sequence[str] | None = None,
    mjds: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    compressed: bool = True,
) -> Path:
    """
    Save complex visibilities and UVW information to a SCEPTer NPZ archive.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.npz`` path.
    visibilities : numpy.ndarray
        Complex visibility samples. Stored under the stable key ``vis``.
    uvw_m : numpy.ndarray, shape (..., 3)
        UVW coordinates in metres. Stored under the stable key ``uvw``.
    frequency : astropy.units.Quantity or float or array-like, optional
        Observing frequency/frequencies. Quantities are converted to hertz and
        stored under ``freq_hz``; plain numeric values are interpreted as hertz.
    pointing_uvw_m, satellite_uvw_m : numpy.ndarray, optional
        Optional full UVW products stored under their existing descriptive keys.
    satellite_distance_m : numpy.ndarray, optional
        Optional per-antenna satellite ranges in metres.
    phase_rad : numpy.ndarray, optional
        Wrapped visibility phases in radians.
    normalised_amplitude : numpy.ndarray, optional
        Dimensionless normalised amplitudes.
    antenna_names, satellite_names : sequence of str, optional
        Optional string labels stored as Unicode arrays.
    mjds : numpy.ndarray, optional
        Optional observation times in Modified Julian Date.
    metadata : mapping, optional
        JSON-serialisable metadata stored as ``metadata_json``.
    compressed : bool, optional
        If ``True`` (default), use ``numpy.savez_compressed``. Otherwise use
        ``numpy.savez``.

    Returns
    -------
    pathlib.Path
        Path to the written archive.

    Raises
    ------
    ValueError
        If ``uvw_m`` does not end with a 3-component UVW axis, or if metadata
        cannot be serialised as JSON.

    Notes
    -----
    The required archive keys are deliberately short and compatible with the
    existing imaging scripts: ``vis`` for complex visibilities and ``uvw`` for
    coordinates. Optional arrays preserve richer SCEPTer tracking context.
    """
    output_path = Path(path)
    vis = np.asarray(visibilities)
    if not np.iscomplexobj(vis):
        vis = vis.astype(np.complex128)

    uvw_arr = np.asarray(uvw_m, dtype=np.float64)
    if uvw_arr.shape[-1:] != (3,):
        raise ValueError("uvw_m must end with a 3-component UVW axis.")

    payload: dict[str, Any] = {
        "schema_name": np.asarray("scepter_visibility_npz"),
        "schema_version": np.asarray("1"),
        "vis": vis,
        "uvw": uvw_arr,
    }
    if frequency is not None:
        payload["freq_hz"] = _as_hz(frequency)
    if pointing_uvw_m is not None:
        payload["pointing_uvw_m"] = np.asarray(pointing_uvw_m, dtype=np.float64)
    if satellite_uvw_m is not None:
        payload["satellite_uvw_m"] = np.asarray(satellite_uvw_m, dtype=np.float64)
    if satellite_distance_m is not None:
        payload["satellite_distance_m"] = np.asarray(satellite_distance_m, dtype=np.float64)
    if phase_rad is not None:
        payload["phase_rad"] = np.asarray(phase_rad, dtype=np.float64)
    if normalised_amplitude is not None:
        payload["normalised_amplitude"] = np.asarray(normalised_amplitude, dtype=np.float64)
    if antenna_names is not None:
        payload["antenna_names"] = np.asarray(tuple(antenna_names), dtype=str)
    if satellite_names is not None:
        payload["satellite_names"] = np.asarray(tuple(satellite_names), dtype=str)
    if mjds is not None:
        payload["mjds"] = np.asarray(mjds, dtype=np.float64)
    if metadata is not None:
        try:
            metadata_json = json.dumps(dict(metadata), sort_keys=True)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serialisable.") from exc
        payload["metadata_json"] = np.asarray(metadata_json)

    saver = np.savez_compressed if compressed else np.savez
    saver(output_path, **payload)
    return output_path


def load_visibility_npz(path: str | Path) -> VisibilityNpzArchive:
    """
    Read complex visibilities and UVW information from a SCEPTer NPZ archive.

    Parameters
    ----------
    path : str or pathlib.Path
        Input archive produced by :func:`save_visibility_npz` or a compatible
        file containing at least ``vis`` and ``uvw`` arrays.

    Returns
    -------
    VisibilityNpzArchive
        Dataclass containing the required visibility/UVW arrays and any
        optional metadata present in the archive.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If required ``vis`` or ``uvw`` keys are missing.
    ValueError
        If the stored ``uvw`` array does not end with a 3-component axis.

    Notes
    -----
    Loading uses ``allow_pickle=False``. Metadata must therefore be stored as
    numeric arrays, string arrays, or JSON text.
    """
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Visibility archive not found: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as data:
        if "vis" not in data or "uvw" not in data:
            missing = ", ".join(key for key in ("vis", "uvw") if key not in data)
            raise KeyError(f"Visibility archive is missing required key(s): {missing}.")

        uvw_arr = np.asarray(data["uvw"], dtype=np.float64)
        if uvw_arr.shape[-1:] != (3,):
            raise ValueError("Stored uvw array must end with a 3-component UVW axis.")

        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))

        def optional_array(key: str) -> np.ndarray | None:
            return np.asarray(data[key]) if key in data else None

        def optional_names(key: str) -> tuple[str, ...]:
            if key not in data:
                return ()
            return tuple(np.asarray(data[key]).astype(str).tolist())

        return VisibilityNpzArchive(
            path=archive_path,
            visibilities=np.asarray(data["vis"]),
            uvw_m=uvw_arr,
            frequency_hz=optional_array("freq_hz"),
            pointing_uvw_m=optional_array("pointing_uvw_m"),
            satellite_uvw_m=optional_array("satellite_uvw_m"),
            satellite_distance_m=optional_array("satellite_distance_m"),
            phase_rad=optional_array("phase_rad"),
            normalised_amplitude=optional_array("normalised_amplitude"),
            antenna_names=optional_names("antenna_names"),
            satellite_names=optional_names("satellite_names"),
            mjds=optional_array("mjds"),
            metadata=metadata,
        )


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
        Can be obtained from scepter.uvw.baseline_bearing() or
        scepter.uvw.baseline_pairs()
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
    baseline_bearing : Get ITRF baseline between two antennas
    baseline_pairs : Get ITRF baselines for antenna array
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
    baseline_bearing : ITRF baseline between two antennas.
    baseline_pairs : All ITRF baselines in an array.

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
    satellite_distance_m : numpy.ndarray, shape (N_ant, T, N_sat)
        Distance from each antenna to each propagated satellite in metres.
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
    satellite_distance_m: np.ndarray
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
        satellite_distance_m = np.asarray(
            sim.topo_pos_dist[:, 0, 0, 0, :, :],
            dtype=np.float64,
        ) * 1000.0
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
            satellite_distance_m=satellite_distance_m,
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
