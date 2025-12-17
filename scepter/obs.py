#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
obs.py - Observation Simulation Module for Satellite Interference Analysis

This module provides a comprehensive framework for simulating satellite-observer interactions
and analyzing potential radio frequency interference (RFI) in radio astronomy observations.
It combines satellite propagation calculations with antenna gain patterns to model the
impact of satellite constellations on radio telescope observations.

Overview
--------
The module implements a multi-dimensional simulation approach to evaluate satellite emissions
and their impact on radio astronomy observations. It handles:

1. **Satellite Propagation**: Uses cysgp4 for high-accuracy satellite position calculations
2. **Antenna Gain Patterns**: Integrates pycraf antenna models for both transmitters and receivers
3. **Coordinate Transformations**: Handles topocentric, satellite frame, and celestial coordinate systems
4. **Interferometry**: Calculates baseline delays and fringe patterns for array observations
5. **Power Budget Analysis**: Computes received power including path loss and antenna gains

Key Components
--------------
- **transmitter_info**: Class to model satellite transmitter characteristics
- **receiver_info**: Class to model radio telescope receiver properties
- **obs_sim**: Main simulation class orchestrating the observation scenario
- Helper functions for angular separations, baseline calculations, and fringe analysis

Simulation Grid Structure
--------------------------
The simulation operates on a 6-dimensional data cube with the following axes:
    1. Observers/Antennas: Multiple antenna locations (cysgp4 PyObserver objects)
    2. Antenna pointings per grid: Multiple pointing directions for each antenna
    3. Sky grid cells: Spatial grid covering the sky (from skynet.pointgen)
    4. Epochs: Separate observation sessions
    5. Time integrations (nint): Sub-integrations within an observation
    6. Satellites: Number of transmitting satellites in the constellation

This structure enables comprehensive analysis across spatial, temporal, and
configuration dimensions simultaneously.

Coordinate Systems
------------------
The module works with multiple coordinate reference frames:
    - **Topocentric (Az/El)**: Observer-centric horizontal coordinates
    - **Satellite Frame (ZXY)**: Satellite body-fixed coordinates (Z = velocity vector)
    - **ICRS**: International Celestial Reference System (RA/Dec)
    - **ITRF2008**: International Terrestrial Reference Frame for baseline calculations

Dependencies
------------
- cysgp4: Satellite propagation using SGP4/SDP4 models
- pycraf: Radio astronomy protection calculations and antenna patterns
- astropy: Coordinate transformations, time handling, and physical constants
- numpy: Numerical array operations

Usage Example
-------------
    >>> # Create receiver and transmitter objects
    >>> receiver = receiver_info(d_rx=25*u.m, eta_a_rx=0.7, 
    ...                          pyobs=observers, freq=1420*u.MHz,
    ...                          bandwidth=10*u.MHz)
    >>> transmitter = transmitter_info(p_tx_carrier=10*cnv.dBW, 
    ...                                carrier_bandwidth=125*u.kHz,
    ...                                duty_cycle=1.0, d_tx=0.5*u.m,
    ...                                freq=1420*u.MHz)
    >>> 
    >>> # Set up observation simulation
    >>> obs = obs_sim(receiver, skygrid, mjds)
    >>> obs.populate(tles_list)
    >>> obs.sky_track(ra=0*u.deg, dec=45*u.deg)
    >>> 
    >>> # Calculate received power
    >>> transmitter.power_tx(receiver.bandwidth)
    >>> ang_sep = obs.sat_separation(mode='tracking')
    >>> g_rx = receiver.antgain1d(obs.pnt_az, obs.pnt_el, 
    ...                           obs.topo_pos_az, obs.topo_pos_el)
    >>> power = prx_cnv(transmitter.fspl(obs.topo_pos_dist), g_rx)

References
----------
- ITU-R Recommendations for radio astronomy protection (RA.769, S.1586-1)
- SGP4 satellite propagation model
- pycraf documentation: https://bwinkel.github.io/pycraf/

Author: Harry Qiu <hqiu678@outlook.com>
Date Created: 12-03-2024
Version: 0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import pycraf
import cysgp4
from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry
from astropy import units as u, constants as const
from astropy.coordinates import EarthLocation,SkyCoord
from astropy.time import Time
from astropy.coordinates import AltAz, ICRS



def sat_frame_pointing(satf_az,satf_el,beam_el,beam_az):
    """
    Calculate angular separation in satellite reference frame.
    
    Computes the angular separation between the satellite's beam pointing
    direction and the observer's position as seen from the satellite. This
    is crucial for determining the transmitter antenna gain toward the observer.
    
    The calculation uses spherical trigonometry to find the true angular
    distance on the celestial sphere, accounting for the curved geometry.
    
    Parameters
    ----------
    satf_az : float or array-like
        Azimuth of the observer in the satellite reference frame (degrees)
    satf_el : float or array-like
        Elevation of the observer in the satellite reference frame (degrees)
    beam_el : float
        Beam elevation angle in satellite reference frame (degrees)
        ZXY convention: Z = velocity vector, X = perpendicular to velocity
    beam_az : float
        Beam azimuth angle in satellite reference frame (degrees)
    
    Returns
    -------
    ang_sep : astropy.Quantity
        Angular separation between beam pointing and observer (degrees)
    delta_az : float or array-like
        Azimuth difference (degrees)
    delta_el : float or array-like
        Elevation difference (degrees)
    
    Notes
    -----
    The satellite reference frame (ZXY) is body-fixed:
    - Z-axis: Satellite velocity vector (along orbit)
    - X-axis: Perpendicular to velocity in orbital plane
    - Y-axis: Completes right-handed system (radial direction)
    
    This frame is natural for satellite operations as:
    - Z-axis points along the direction of motion
    - X-axis often aligns with solar panels or antennas
    - Y-axis points roughly toward/away from Earth
    
    The angular separation is computed using the haversine formula to avoid
    numerical issues at small angles.
    
    Examples
    --------
    >>> # Observer directly below satellite (nadir)
    >>> ang_sep, daz, del = sat_frame_pointing(
    ...     satf_az=0, satf_el=-90,  # Observer at nadir
    ...     beam_az=0, beam_el=-90    # Beam pointing to nadir
    ... )
    >>> print(f"Separation: {ang_sep}")  # ~0 degrees
    >>> 
    >>> # Beam pointed forward, observer to side
    >>> ang_sep, daz, del = sat_frame_pointing(
    ...     satf_az=90, satf_el=0,   # Observer to side
    ...     beam_az=0, beam_el=0      # Beam forward
    ... )
    >>> print(f"Separation: {ang_sep}")  # ~90 degrees
    
    See Also
    --------
    obs_sim.txbeam_angsep : Apply this calculation in simulation context
    pycraf.geometry.true_angular_distance : Underlying calculation
    """
    # tleprop = sat_info
    # #### obtain coordinates in observation frame and satellite frame
    # # topo_pos_az, topo_pos_el= tleprop['obs_az'], tleprop['obs_el']
    # satf_az, satf_el, satf_dist = tleprop['sat_frame_az'], tleprop['sat_frame_el'], tleprop['sat_frame_dist']


    #### check numpy braodcasting to fix dimensions

    ang_sep=geometry.true_angular_distance(satf_az*u.deg,satf_el*u.deg,beam_az*u.deg,beam_el*u.deg)
    delta_az=satf_az-beam_az
    delta_el=satf_el-beam_el
    return ang_sep,delta_az,delta_el



def baseline_bearing(ref,ant):
    """
    Calculate bearing vector and distance between two antennas.
    
    Computes the baseline vector from a reference antenna to another antenna
    in ITRF2008 Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates. This
    is a fundamental calculation for radio interferometry.
    
    Parameters
    ----------
    ref : cysgp4.PyObserver
        Reference antenna/location (baseline origin)
        Must have .loc attributes: lon (degrees), lat (degrees), alt (meters)
    ant : cysgp4.PyObserver
        Target antenna for baseline measurement
        Must have .loc attributes: lon (degrees), lat (degrees), alt (meters)
    
    Returns
    -------
    bearing : numpy.ndarray, shape (3,)
        Baseline vector in ITRF2008 Cartesian coordinates [X, Y, Z] (meters)
        Points from reference antenna to target antenna
    d : float
        Baseline length (meters)
        Euclidean distance between the two antennas
    
    Notes
    -----
    ITRF2008 (International Terrestrial Reference Frame 2008):
    - Earth-centered, Earth-fixed coordinate system
    - Origin: Earth's center of mass
    - Z-axis: Earth's rotation axis (toward North pole)
    - X-axis: Intersection of equatorial plane and prime meridian
    - Y-axis: 90° East of X-axis
    
    The conversion from WGS84 geodetic coordinates (lon, lat, alt) to
    ITRF2008 Cartesian coordinates accounts for:
    - Earth's ellipsoidal shape (WGS84 ellipsoid)
    - Local radius of curvature
    - Altitude above ellipsoid
    
    Examples
    --------
    >>> from cysgp4 import PyObserver
    >>> 
    >>> # VLA-like baseline (New Mexico)
    >>> ref = PyObserver(lon=-107.618, lat=34.079, alt=2124)
    >>> ant = PyObserver(lon=-107.617, lat=34.079, alt=2124)  # ~1km east
    >>> 
    >>> bearing, distance = baseline_bearing(ref, ant)
    >>> print(f"Baseline vector: {bearing} m")
    >>> print(f"Baseline length: {distance:.1f} m")
    >>> 
    >>> # East-West component
    >>> print(f"East-West: {bearing[0]:.1f} m")
    
    See Also
    --------
    baseline_pairs : Calculate all baselines in an array
    baseline_vector : Project baseline onto source direction
    pycraf.geospatial.wgs84_to_itrf2008 : Coordinate transformation
    """
    ant1 = ref
    ant2 = ant
    x1,y1,z1 = pycraf.geospatial.wgs84_to_itrf2008(ant1.loc.lon*u.deg, ant1.loc.lat*u.deg, ant1.loc.alt*u.m)
    x2,y2,z2 = pycraf.geospatial.wgs84_to_itrf2008(ant2.loc.lon*u.deg, ant2.loc.lat*u.deg, ant2.loc.alt*u.m)



    a1=np.array([x1.value,y1.value,z1.value])
    a2=np.array([x2.value,y2.value,z2.value])
    # print(a1,a2)
    bearing = a2-a1
    d = np.linalg.norm(bearing) # Calculate the distance between the antennas

    
    return bearing, d # Return the distance in meters

def baseline_pairs(antennas):
    """
    Calculate baseline vectors and distances for antenna array.
    
    Computes the baseline vectors from a reference antenna (first in the array)
    to all other antennas in ITRF2008 Cartesian coordinates. This is essential
    for interferometric observations and fringe pattern calculations.
    
    Parameters
    ----------
    antennas : list of cysgp4.PyObserver
        List of antenna/observer objects containing location information
        (longitude, latitude, altitude)
    
    Returns
    -------
    bearings : numpy.ndarray, shape (n_antennas, 3)
        Baseline vectors in ITRF2008 Cartesian coordinates (meters)
        Each row is [X, Y, Z] vector from reference antenna to antenna i
    baselines : numpy.ndarray, shape (n_antennas,)
        Baseline lengths (meters)
        Euclidean norm of each bearing vector
    
    Notes
    -----
    The reference antenna is always antennas[0]. All baselines are measured
    from this reference point.
    
    ITRF2008 is an Earth-centered, Earth-fixed (ECEF) coordinate system:
    - Origin: Earth's center of mass
    - Z-axis: Earth's rotation axis
    - X-axis: Intersection of equatorial plane and Greenwich meridian
    - Y-axis: Completes right-handed system (90° east of X)
    
    For interferometry, baseline vectors are used to calculate:
    - Geometric delays between antennas
    - Fringe patterns
    - UV coverage
    - Array sensitivity
    
    Examples
    --------
    >>> from cysgp4 import PyObserver
    >>> 
    >>> # Create a simple 2-element interferometer
    >>> ant1 = PyObserver(lon=21.443, lat=-30.713, alt=1000)  # Reference
    >>> ant2 = PyObserver(lon=21.444, lat=-30.713, alt=1000)  # 100m east
    >>> 
    >>> bearings, distances = baseline_pairs([ant1, ant2])
    >>> print(f"Baseline vector: {bearings[1]}")  # [dx, dy, dz]
    >>> print(f"Baseline length: {distances[1]:.1f} m")
    
    See Also
    --------
    baseline_bearing : Calculate single baseline between two antennas
    baseline_vector : Project baseline onto source direction
    pycraf.geospatial.wgs84_to_itrf2008 : Coordinate conversion
    """
    n_antennas = len(antennas)
    # Pre-allocate arrays for better performance
    baselines = np.empty(n_antennas, dtype=np.float64)  ### true baseline distance
    bearings = np.empty((n_antennas, 3), dtype=np.float64)
    
    ref = antennas[0]
    for i, ant in enumerate(antennas):
        # Calculate the baseline distance
        bearing, d = baseline_bearing(ref, ant)
        baselines[i] = d
        bearings[i] = bearing

    return bearings, baselines

def baseline_vector(d,az,el,lat):
    """
    Calculate effective baseline projection for a given pointing direction.
    
    Projects a physical baseline vector onto the direction of an astronomical
    source, accounting for the local horizon coordinate system. This is the
    effective baseline that contributes to interferometric fringes for a
    source at the specified azimuth and elevation.
    
    The projection transforms from a baseline length and observer coordinates
    to a 3D effective baseline vector in the local topocentric frame.
    
    Parameters
    ----------
    d : float
        Physical baseline length (meters)
    az : float
        Source azimuth angle (radians)
        Measured from North through East
    el : float
        Source elevation angle (radians)
        Measured from horizon (0) to zenith (π/2)
    lat : float
        Observer's latitude (radians)
        North positive, range: -π/2 to π/2
    
    Returns
    -------
    vector : numpy.ndarray, shape (3,)
        Effective baseline vector in topocentric Cartesian coordinates (meters)
        Components: [x, y, z] where:
        - x: North component
        - y: East component
        - z: Zenith component
    
    Notes
    -----
    The transformation accounts for:
    - Source direction (az, el)
    - Observer's latitude (affects coordinate rotation)
    - Baseline length
    
    The effective baseline determines:
    - Fringe frequency (longer baseline → faster fringes)
    - Spatial resolution (longer baseline → finer resolution)
    - UV coverage point
    
    For a source at the zenith (el = π/2), the effective baseline equals
    the physical baseline length in the zenith direction.
    
    Examples
    --------
    >>> # 1 km baseline, source at 45° elevation, due south, mid-latitude
    >>> import numpy as np
    >>> vec = baseline_vector(
    ...     d=1000,                    # 1 km baseline
    ...     az=np.radians(180),        # South
    ...     el=np.radians(45),         # 45° elevation
    ...     lat=np.radians(30)         # 30° N latitude
    ... )
    >>> print(f"Effective baseline: {vec}")
    >>> print(f"Magnitude: {np.linalg.norm(vec):.1f} m")
    
    See Also
    --------
    mod_tau : Calculate geometric delay from baseline
    baseline_nearfield_delay : Account for near-field effects
    """
    
    
    return d*np.array([np.cos(lat)*np.sin(el)-np.sin(lat)*np.cos(el)*np.cos(az),
    np.cos(el)*np.sin(az),
    np.sin(lat)*np.sin(el)+np.cos(lat)*np.cos(el)*np.cos(az)]) # x,y,z coordinates in meters
    

def mod_tau(az,el,lat,D):
    """
    Calculate geometric delay for astronomical source pointing.
    
    Computes the differential time delay between antenna elements in an
    interferometric array when observing a distant astronomical source.
    This is the classical "far-field" delay calculation for radio interferometry.
    
    The delay arises because wavefronts from a distant source arrive at
    different antennas at slightly different times due to the geometric
    path difference.
    
    Parameters
    ----------
    az : float or array-like
        Source azimuth angle (radians)
    el : float or array-like
        Source elevation angle (radians)
    lat : float
        Observer latitude (radians)
    D : astropy.Quantity
        Baseline length (meters or compatible units)
        Will be converted to meters internally
    
    Returns
    -------
    tau : astropy.Quantity
        Geometric time delay (seconds)
        Shape matches input arrays (supports broadcasting)
    
    Notes
    -----
    The calculation:
    1. Projects baseline onto source direction using baseline_vector
    2. Computes effective baseline length D_eff
    3. Calculates delay: τ = D_eff / c
    
    For a source at angle θ from the baseline direction:
        τ = (D/c) * sin(θ)
    
    This is the "far-field" approximation, valid when:
    - Source distance >> baseline length (astronomical sources)
    - Wavefronts are effectively planar
    
    For satellites (near-field), use baseline_nearfield_delay instead.
    
    Typical delays:
    - 1 km baseline, zenith source: ~3.3 μs
    - 10 km baseline, 30° from zenith: ~16.7 μs
    - 1000 km baseline (VLBI): ~3.3 ms
    
    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> 
    >>> # Simple interferometer
    >>> delay = mod_tau(
    ...     az=np.radians(180),        # South
    ...     el=np.radians(60),         # 60° elevation
    ...     lat=np.radians(30),        # 30° N
    ...     D=1000*u.m                 # 1 km baseline
    ... )
    >>> print(f"Delay: {delay.to(u.us)}")
    >>> 
    >>> # Array of source positions
    >>> az_array = np.linspace(0, 2*np.pi, 100)
    >>> el_array = np.ones(100) * np.radians(45)
    >>> delays = mod_tau(az_array, el_array, np.radians(30), 5000*u.m)
    
    See Also
    --------
    baseline_vector : Baseline projection calculation
    baseline_nearfield_delay : Near-field delay corrections for satellites
    fringe_response : Convert delays to fringe patterns
    """
    c = 3e8 *u.m/u.s  # speed of light in m/s
    baseline = D.to(u.m) # Convert baseline to meters
    new_bearing = baseline_vector(baseline,az,el,lat)
    # print(new_bearing.shape)
    D_eff = np.linalg.norm(new_bearing,axis=0)
    # print(D_eff.shape)    
    return D_eff/c




def baseline_nearfield_delay(l1,l2,tau):
    """
    Calculate near-field delay corrections for satellite observations.
    
    Computes the differential delay for interferometric observations of
    near-field sources (satellites) where the wavefront curvature must be
    considered. This corrects the far-field geometric delay for the finite
    distance to the source.
    
    For distant astronomical sources, wavefronts are effectively planar and
    the delay is purely geometric. For satellites in Low Earth Orbit (LEO),
    wavefronts have significant curvature, causing path length differences
    that depend on the actual distances to each antenna.
    
    Parameters
    ----------
    l1 : astropy.Quantity
        Distance from satellite to reference antenna (km, m, or compatible)
    l2 : astropy.Quantity
        Distance from satellite to target antenna (km, m, or compatible)
    tau : astropy.Quantity
        Far-field geometric delay (seconds)
        Typically from mod_tau()
    
    Returns
    -------
    delay : astropy.Quantity
        Total delay including near-field corrections (seconds)
        delay = (l1 - l2)/c - tau
    
    Notes
    -----
    The total delay consists of:
    1. Path length difference: (l1 - l2)/c
       Direct time difference due to different distances
    2. Geometric correction: -tau
       Adjusts for the far-field approximation
    
    For satellites at ~500 km altitude with baselines of ~1-10 km:
    - Path length difference: ~0.01-0.1 μs
    - Corrections can be significant for long baselines
    
    The near-field effect becomes important when:
        baseline^2 / (8 * satellite_distance) ≈ wavelength
    
    For LEO satellites (500-2000 km) and baselines >100 m, near-field
    corrections are usually necessary for accurate interferometry.
    
    Examples
    --------
    >>> from astropy import units as u
    >>> 
    >>> # LEO satellite observation
    >>> l1 = 550*u.km  # Distance to antenna 1
    >>> l2 = 551*u.km  # Distance to antenna 2 (1 km farther)
    >>> tau = 3.33*u.us  # Far-field delay estimate
    >>> 
    >>> delay = baseline_nearfield_delay(l1, l2, tau)
    >>> print(f"Corrected delay: {delay.to(u.us)}")
    >>> 
    >>> # Compare with far-field approximation
    >>> farfield_delay = tau
    >>> nearfield_delay = (l1 - l2) / (3e8 * u.m/u.s)
    >>> correction = nearfield_delay - farfield_delay - delay
    
    See Also
    --------
    mod_tau : Calculate far-field geometric delay
    obs_sim.baselines_nearfield_delays : Apply to full simulation
    """
    c = 3e8 *u.m/u.s  # speed of light in m/s
    l1 = l1.to(u.m) # Convert distance to meters
    l2 = l2.to(u.m) # Convert distance to meters
    
    return  (l1-l2)/c-tau

def fringe_attenuation(theta, baseline, bandwidth):
    """
    Calculate fringe attenuation due to finite bandwidth.
    
    Computes the amplitude reduction of interferometric fringes caused by
    bandwidth smearing. When a source is offset from the phase center, the
    geometric delay varies across the observing bandwidth, causing partial
    cancellation of the fringe signal.
    
    This effect is critical for wide-field interferometric observations and
    for understanding sensitivity loss away from the pointing center.
    
    Parameters
    ----------
    theta : astropy.Quantity
        Angular offset from phase center (radians, degrees, or compatible)
    baseline : astropy.Quantity
        Baseline length (meters or compatible)
        Can be the projected baseline in the direction of interest
    bandwidth : astropy.Quantity
        Observing bandwidth (Hz or compatible)
    
    Returns
    -------
    attenuation : float or array
        Fringe amplitude attenuation factor (dimensionless)
        Range: 0 to 1, where 1 = no attenuation
        Shape matches input arrays
    
    Notes
    -----
    The attenuation follows a sinc function:
        A(θ) = sinc(sin(θ) * D * Δf / c)
    where:
    - θ: angular offset
    - D: baseline length
    - Δf: bandwidth
    - c: speed of light
    
    Physical interpretation:
    - At phase center (θ=0): A = 1 (no attenuation)
    - First null at: sin(θ) = c / (D * Δf)
    - Attenuation increases with: larger θ, longer baseline, wider bandwidth
    
    This is analogous to the "chromatic aberration" of interferometers.
    
    Practical implications:
    - Limits usable field of view
    - Reduces sensitivity to off-axis sources
    - Important for satellite RFI (often offset from phase center)
    - Can be compensated by "w-projection" in imaging
    
    Examples
    --------
    >>> from astropy import units as u
    >>> 
    >>> # Sensitivity at 10° from phase center
    >>> atten = fringe_attenuation(
    ...     theta=10*u.deg,
    ...     baseline=1000*u.m,
    ...     bandwidth=100*u.MHz
    ... )
    >>> print(f"Attenuation: {atten:.2%}")
    >>> 
    >>> # Field of view estimate (first null)
    >>> theta_null = np.arcsin(3e8 / (1000 * 100e6))
    >>> print(f"First null at: {np.degrees(theta_null):.1f}°")
    >>> 
    >>> # Compare narrow vs wide band
    >>> angles = np.linspace(0, 20, 100) * u.deg
    >>> atten_narrow = fringe_attenuation(angles, 1*u.km, 10*u.MHz)
    >>> atten_wide = fringe_attenuation(angles, 1*u.km, 100*u.MHz)
    
    See Also
    --------
    bw_fringe : Bandwidth-integrated fringe response
    fringe_response : Single-frequency fringe calculation
    """
    c = 3e8  # speed of light in m/s
    theta = theta.to(u.rad).value  # Convert angle to radians
    baseline = baseline.to(u.m).value  # Convert baseline to meters
    bandwidth = bandwidth.to(u.Hz).value  # Convert bandwidth to Hz
    return np.sinc(np.sin(theta)*baseline*bandwidth/c)

def fringe_response(delay,frequency):
    """
    Calculate single-frequency fringe response for interferometry.
    
    Computes the fringe amplitude for a two-element interferometer at a
    single frequency, assuming equal antenna gains. This is the fundamental
    response of an interferometer to a point source.
    
    The fringe pattern oscillates as a function of the geometric delay,
    with the frequency of oscillation determined by the observing frequency.
    
    Parameters
    ----------
    delay : astropy.Quantity
        Geometric delay between antenna elements (seconds)
        Can be scalar or array
    frequency : astropy.Quantity
        Observing frequency (Hz or compatible)
        Can be scalar or array
    
    Returns
    -------
    response : float or array
        Fringe amplitude (dimensionless)
        Range: -1 to +1
        - +1: Constructive interference (in phase)
        - -1: Destructive interference (180° out of phase)
        -  0: 90° phase difference
        Shape matches broadcasted input arrays
    
    Notes
    -----
    The fringe response is:
        R(τ, f) = cos(2π f τ)
    where:
    - τ: geometric delay (time difference between signal arrivals)
    - f: observing frequency
    
    This assumes:
    - Point source
    - Equal antenna gains
    - No noise
    - Monochromatic signal
    
    Physical interpretation:
    - Fringes oscillate as Earth rotates (changing delay)
    - Fringe rate: dR/dt depends on source geometry
    - Fringe spacing: Δτ = 1/f
    
    For realistic observations:
    - Use bw_fringe() for finite bandwidth effects
    - Include antenna gains from receiver_info
    - Add system noise and atmospheric effects
    
    Examples
    --------
    >>> from astropy import units as u
    >>> import numpy as np
    >>> 
    >>> # Single fringe calculation
    >>> response = fringe_response(
    ...     delay=1*u.us,
    ...     frequency=1.4*u.GHz
    ... )
    >>> print(f"Fringe amplitude: {response:.3f}")
    >>> 
    >>> # Fringe pattern vs delay
    >>> delays = np.linspace(0, 10, 1000) * u.us
    >>> fringes = fringe_response(delays, 1420*u.MHz)
    >>> # Plot shows oscillation with period 1/f ≈ 0.7 μs
    >>> 
    >>> # Multi-frequency response
    >>> freqs = np.linspace(1.4, 1.5, 100) * u.GHz
    >>> response_vs_freq = fringe_response(1*u.us, freqs)
    
    See Also
    --------
    bw_fringe : Bandwidth-integrated fringe response
    fringe_attenuation : Bandwidth smearing effects
    obs_sim.sat_fringe : Apply to satellite observations
    """
    delay = delay.to(u.s).value  # Convert delay to seconds
    frequency = frequency.to(u.Hz).value  # Convert frequency to Hz

    return np.cos(2*np.pi*frequency*delay)


def bw_fringe(delays,bwchan,fch1,chan_bin=100):
    """
    Calculate bandwidth-averaged fringe response for interferometry.
    
    Computes the fringe amplitude for a two-element interferometer integrated
    over a finite channel bandwidth. This accounts for bandwidth smearing,
    which reduces fringe amplitude for sources with large geometric delays.
    
    The integration is performed numerically across the channel bandwidth using
    multiple frequency bins, providing an accurate model of the fringe response
    for realistic observing scenarios.
    
    Parameters
    ----------
    delays : numpy.ndarray
        Geometric delay array (seconds), must be 1-D
        Flatten multi-dimensional arrays before calling; reshape output as needed
    bwchan : astropy.Quantity
        Channel bandwidth (Hz, kHz, MHz)
    fch1 : astropy.Quantity
        Channel center frequency (Hz, kHz, MHz)
    chan_bin : int, optional
        Number of frequency bins for integration (default: 100)
        Higher values improve accuracy but increase computation time
    
    Returns
    -------
    response : numpy.ndarray
        Bandwidth-averaged fringe amplitude (dimensionless)
        Shape matches input delays array
        Values range from -1 to +1
    
    Notes
    -----
    The fringe response at a single frequency is:
        R(f) = cos(2π f τ)
    where τ is the geometric delay
    
    The bandwidth-averaged response is:
        <R> = (1/N) Σ cos(2π f_i τ)
    where f_i samples the channel bandwidth
    
    Bandwidth smearing reduces fringe amplitude when:
        Δf * τ ≳ 1
    where Δf is the channel bandwidth
    
    For satellite observations:
    - Near-field effects cause varying delays across the observation
    - Large delays (>1 μs) can cause significant bandwidth smearing
    - Narrow channels (Δf < 1/τ) preserve fringe amplitude
    
    Examples
    --------
    >>> # Calculate fringe for typical radio astronomy setup
    >>> delays = np.array([1e-6, 2e-6, 5e-6])  # 1-5 microseconds
    >>> bw = 10*u.kHz
    >>> freq = 1420*u.MHz
    >>> 
    >>> fringe = bw_fringe(delays, bw, freq, chan_bin=100)
    >>> print(f"Fringe amplitudes: {fringe}")
    >>> 
    >>> # Show bandwidth smearing effect
    >>> delays = np.linspace(0, 10e-6, 100)
    >>> fringes_narrow = bw_fringe(delays, 1*u.kHz, 1420*u.MHz)
    >>> fringes_wide = bw_fringe(delays, 100*u.kHz, 1420*u.MHz)
    >>> # fringes_wide shows more suppression at large delays
    
    See Also
    --------
    fringe_response : Single-frequency fringe calculation
    baseline_nearfield_delay : Calculate geometric delays
    obs_sim.sat_fringe : Apply fringe calculation to simulation
    """
    fch1 = fch1.to(u.kHz).value  # Convert frequency to kHz
    bwchan = bwchan.to(u.kHz).value  # Convert bandwidth to kHz
    # chan_bin=np.int32(bwchan/0.1) ### the bin number
    freq_array= np.linspace(fch1-bwchan*0.5,fch1+bwchan*0.5,chan_bin) *u.kHz # 0.1 kHz resolution
    delays= delays[:,np.newaxis] # add axis to delays
    freq_array = freq_array[np.newaxis,:]
    fringes=fringe_response(delays,freq_array)
    return np.mean(fringes,axis=1)


def prx_cnv(pwr,g_rx, outunit=u.W):
    """
    Calculate received power with receiver gain response.
    
    Combines the incident power with the receiver antenna gain to compute
    the actual received power. This is a key step in the link budget calculation,
    converting power flux density to received power.
    
    The calculation is performed in logarithmic (dB) space for numerical
    stability, then converted to linear units.
    
    Parameters
    ----------
    pwr : astropy.Quantity
        Incident power at the receiver location, typically in dBm or dBW.
        This is usually the output from transmitter_info.fspl()
    g_rx : astropy.Quantity
        Receiver antenna gain in the direction of the source (dBi).
        Typically from receiver_info.antgain1d()
    outunit : astropy.Unit, optional
        Desired output unit for received power (default: u.W)
        Common choices: u.W, u.mW, cnv.dBm, cnv.dBW
    
    Returns
    -------
    p_rx : astropy.Quantity
        Received power in linear space (specified units)
    
    Notes
    -----
    The calculation in dB space:
        P_rx [dB] = P_incident [dB] + G_rx [dBi]
    
    Then converted to linear units:
        P_rx [W] = 10^(P_rx[dB]/10)
    
    This function handles the full link budget:
        P_rx = P_tx + G_tx - FSPL + G_rx
    where the first three terms come from transmitter_info.fspl()
    
    Examples
    --------
    >>> # Complete link budget calculation
    >>> tx = transmitter_info(10*cnv.dBW, 125*u.kHz, 1.0, 0.5*u.m, 1420*u.MHz)
    >>> rx = receiver_info(13.5*u.m, 0.7, observer, 1420*u.MHz, 10*u.MHz)
    >>> 
    >>> # Calculate incident power after FSPL
    >>> tx.power_tx(rx.bandwidth)
    >>> tx.satgain1d(0*u.deg)
    >>> power_incident = tx.fspl(500*u.km)
    >>> 
    >>> # Apply receiver gain
    >>> g_rx = rx.antgain1d(pnt_az, pnt_el, sat_az, sat_el)
    >>> power_received = prx_cnv(power_incident, g_rx, outunit=u.W)
    >>> 
    >>> # Convert to flux density
    >>> A_eff = np.pi * (rx.d_rx/2)**2 * rx.eta_a_rx
    >>> flux = power_received / A_eff / rx.bandwidth
    
    See Also
    --------
    transmitter_info.fspl : Calculate incident power with path loss
    receiver_info.antgain1d : Calculate receiver gain pattern
    pfd_to_Jy : Convert power flux density to Jansky units
    """

    p_db = pwr + g_rx
    p_rx = p_db.to(outunit) ## convert to unit needed
    return p_rx

def pfd_to_Jy(pfd):
    """
    Convert power flux density from dB scale to Jansky units.
    
    Converts power flux density (PFD) from logarithmic decibel units
    (dBW/m²/Hz) to linear Jansky units (Jy), the standard unit in radio
    astronomy for flux density measurements.
    
    The Jansky is defined as:
        1 Jy = 10^-26 W/m²/Hz
    
    This conversion is essential for comparing satellite interference levels
    with astronomical source strengths and sensitivity limits.
    
    Parameters
    ----------
    pfd : float or array-like
        Power flux density in dB scale (dBW, dBW/m²/Hz)
        Can be scalar or numpy array
    
    Returns
    -------
    F_Jy : float or array-like
        Power flux density in Jansky (Jy)
        Shape matches input
    
    Notes
    -----
    Conversion steps:
    1. Convert from dB to linear: P_W = 10^(pfd/10)
    2. Convert to Jansky: F_Jy = P_W / 10^-26
    
    Typical flux density scales in radio astronomy:
    - Sun: ~10^6 Jy (extremely bright)
    - Jupiter: ~1000 Jy
    - Bright radio sources (Cas A, Cyg A): ~1000-10000 Jy
    - Typical quasars: ~1-10 Jy
    - Detection limit for SKA: ~10^-6 Jy (1 μJy)
    
    Satellite interference context:
    - LEO satellites: Can exceed 10^6 Jy at radio telescope
    - Acceptable RFI: Typically <0.1 Jy for most observations
    - Continuum observations: More tolerant than spectral line
    
    Examples
    --------
    >>> # Convert satellite power flux density
    >>> pfd_satellite = -100  # dBW/m²/Hz
    >>> flux = pfd_to_Jy(pfd_satellite)
    >>> print(f"Flux density: {flux:.2e} Jy")
    >>> 
    >>> # Array of values
    >>> import numpy as np
    >>> pfds = np.array([-100, -110, -120, -130])  # dBW/m²/Hz
    >>> fluxes = pfd_to_Jy(pfds)
    >>> print(f"Fluxes: {fluxes}")
    >>> 
    >>> # Compare with astronomical source
    >>> pfd_sat = -95  # dBW/m²/Hz
    >>> flux_sat = pfd_to_Jy(pfd_sat)
    >>> flux_casa = 10000  # Jy, Cassiopeia A
    >>> print(f"Satellite is {flux_sat/flux_casa:.2e} times Cas A")
    
    See Also
    --------
    prx_cnv : Calculate received power with antenna gain
    transmitter_info.fspl : Calculate power flux density at distance
    
    References
    ----------
    - 1 Jansky = 10^-26 W/m²/Hz (definition)
    - ITU-R RA.769: Protection criteria for radio astronomy
    """

    # Convert W/m^2/Hz to W/m^2
    P_W = 10 ** (pfd / 10)


    
    # Define the reference flux density (1 Jy = 10^-26 W/m^2/Hz)
    S_0 = 10**-26    

    # Convert W to Jy
    F_Jy = P_W / S_0 
    
    return F_Jy


class transmitter_info():
    """
    Information holder class for satellite transmitter characteristics.
    
    This class encapsulates all relevant parameters of a satellite transmitter
    including power, frequency, antenna properties, and signal characteristics.
    It provides methods to calculate transmitter gain patterns, power spectral
    density, and path loss for interference analysis.
    
    The transmitter model assumes a simplified satellite downlink with configurable
    parameters for power, duty cycle, and antenna gain patterns. It supports both
    simple 1D gain patterns (based on diameter and frequency) and custom 2D gain
    functions for more complex antenna patterns.
    
    Attributes
    ----------
    carrier : astropy.Quantity
        Transmitted carrier power (typically in dBW or W)
    carrier_bandwidth : astropy.Quantity
        Bandwidth of the carrier signal (Hz)
    duty_cycle : float
        Fraction of time the transmitter is active (0 to 1)
    d_tx : astropy.Quantity
        Diameter of the transmitter antenna aperture (m)
    freq : astropy.Quantity
        Center frequency of the transmission (Hz)
    p_tx : astropy.Quantity
        Calculated transmitted power in observation bandwidth (set by power_tx method)
    g_tx : float
        Transmitter antenna gain in direction of interest (dBi)
    ras_bandwidth : astropy.Quantity
        Radio astronomy service (RAS) observation bandwidth (Hz)
    
    Examples
    --------
    >>> from astropy import units as u
    >>> from pycraf import conversions as cnv
    >>> 
    >>> # Create a transmitter for a typical LEO satellite
    >>> tx = transmitter_info(
    ...     p_tx_carrier=10*cnv.dBW,
    ...     carrier_bandwidth=125*u.kHz,
    ...     duty_cycle=1.0,
    ...     d_tx=0.5*u.m,
    ...     freq=1575.42*u.MHz  # GPS L1
    ... )
    >>> 
    >>> # Calculate power in 10 MHz observation band
    >>> p_tx = tx.power_tx(10*u.MHz)
    >>> 
    >>> # Get gain pattern at 30 degree off-boresight
    >>> gain = tx.satgain1d(30*u.deg)
    >>> 
    >>> # Calculate power at observer after path loss
    >>> power_rx = tx.fspl(1000*u.km)
    
    Notes
    -----
    - The class uses pycraf's antenna patterns for gain calculations
    - Free space path loss (FSPL) includes both distance and frequency effects
    - Power calculations account for duty cycle and bandwidth integration
    
    See Also
    --------
    receiver_info : Receiver/telescope characteristics
    obs_sim : Main observation simulation class
    """

    def __init__(self,p_tx_carrier,carrier_bandwidth,duty_cycle,d_tx,freq):
        """
        Initialize transmitter information object.
        
        Parameters
        ----------
        p_tx_carrier : astropy.Quantity
            Transmitted power of the carrier signal (e.g., 10*u.W or 10*cnv.dBW)
        carrier_bandwidth : astropy.Quantity
            Bandwidth of the carrier signal (Hz)
        duty_cycle : float
            Duty cycle of the signal (0 to 1, where 1 = continuous transmission)
        d_tx : astropy.Quantity
            Diameter of the transmitter antenna aperture (m)
        freq : astropy.Quantity
            Center frequency of the transmission (Hz)
        """
        self.carrier = p_tx_carrier
        self.carrier_bandwidth = carrier_bandwidth
        self.duty_cycle = duty_cycle
        self.d_tx = d_tx
        self.freq = freq
    


    def power_tx(self,ras_bandwidth):
        """
        Calculate the transmitted power integrated over the observation bandwidth.
        
        This method computes the effective transmitted power that falls within the
        radio astronomy service (RAS) observation bandwidth. It accounts for the
        carrier power spectral density, duty cycle, and bandwidth integration.
        
        The calculation follows these steps:
        1. Convert carrier power to power spectral density (W/Hz)
        2. Apply duty cycle factor
        3. Integrate over the RAS observation bandwidth
        4. Convert to dBm for convenience
        
        Parameters
        ----------
        ras_bandwidth : astropy.Quantity
            Radio astronomy observation bandwidth (Hz)
        
        Returns
        -------
        p_tx : astropy.Quantity
            Transmitted power in the observation bandwidth (dBm)
        
        Notes
        -----
        The transmitted power spectral density is calculated as:
            p_tx_nu = (P_carrier / BW_carrier) * duty_cycle
        
        Then integrated over the observation bandwidth:
            P_tx = p_tx_nu * BW_obs
        
        This assumes the transmitter signal overlaps with the observation band.
        
        Examples
        --------
        >>> tx = transmitter_info(10*cnv.dBW, 125*u.kHz, 1.0, 0.5*u.m, 1420*u.MHz)
        >>> power = tx.power_tx(10*u.MHz)
        >>> print(f"Power in 10 MHz band: {power}")
        """
        # Calculate the transmitted power
        self.ras_bandwidth = ras_bandwidth
        p_tx_nu_peak = (
        self.carrier.physical / self.carrier_bandwidth
        ).to(u.W / u.Hz)
        p_tx_nu = p_tx_nu_peak * self.duty_cycle
        p_tx = p_tx_nu.to(u.W / u.Hz) * ras_bandwidth
        self.p_tx = p_tx.to(cnv.dBm)

        return self.p_tx
    def satgain1d(self,phi):
        """
        Calculate 1-D satellite antenna gain pattern.
        
        Retrieves the antenna gain from a basic 1-D gain pattern function based on
        the Sinclair-Levy (FL) pattern model. The gain is calculated as a function
        of the angular separation from the antenna boresight.
        
        This method uses pycraf's fl_pattern function which implements a standard
        parabolic antenna pattern based on ITU recommendations.
        
        Parameters
        ----------
        phi : astropy.Quantity
            Angular separation from antenna boresight (degrees or radians).
            This represents the beam tilt angle of the satellite relative to
            the direction perpendicular to Earth's surface.
        
        Returns
        -------
        G_tx : astropy.Quantity
            Transmitter antenna gain at the specified angle (dBi)
        
        Notes
        -----
        The maximum gain is calculated from the antenna diameter and wavelength:
            G_max = eta * (pi * D / lambda)^2
        where eta is the aperture efficiency (assumed in the pattern function).
        
        The off-axis gain follows the FL (Sinclair-Levy) pattern which includes:
        - Main lobe with 3dB beamwidth
        - Near sidelobes
        - Far sidelobes
        
        Examples
        --------
        >>> tx = transmitter_info(10*cnv.dBW, 125*u.kHz, 1.0, 0.5*u.m, 1420*u.MHz)
        >>> gain_boresight = tx.satgain1d(0*u.deg)
        >>> gain_30deg = tx.satgain1d(30*u.deg)
        >>> print(f"Gain at boresight: {gain_boresight}")
        >>> print(f"Gain at 30°: {gain_30deg}")
        
        See Also
        --------
        custom_gain : For more complex 2D gain patterns
        pycraf.antenna.fl_pattern : Underlying antenna pattern function
        """
        
        wavelength=const.c/self.freq
        gmax=antenna.fl_G_max_from_size(self.d_tx,wavelength)  ## get G_max from diameter and frequency

        # hpbw=antenna.fl_hpbw_from_size(d_tx,wavelength)  ## get hpbw from diameter and frequency
        ### calculate angular separation of satellite to telescope pointing
        flpattern=antenna.fl_pattern(phi,diameter=self.d_tx,wavelength=wavelength,G_max=gmax)
        G_tx=flpattern
        self.g_tx = G_tx
        return G_tx
    def fspl(self,sat_obs_dist,outunit=u.W):
        """
        Calculate power at observer after free space path loss (FSPL).
        
        Computes the received power at the observation point accounting for:
        1. Free space propagation loss
        2. Transmitter antenna gain in the direction of observer
        3. Original transmitted power
        
        The calculation follows the Friis transmission equation in logarithmic form:
            P_rx = P_tx + G_tx - FSPL
        where FSPL = 20*log10(4*pi*d*f/c)
        
        Parameters
        ----------
        sat_obs_dist : float or astropy.Quantity
            Distance between satellite and observer. If float is provided,
            it is assumed to be in meters.
        outunit : astropy.Unit, optional
            Output unit for the received power (default: u.W)
        
        Returns
        -------
        sat_power : astropy.Quantity
            Power at the observer location after path loss (in specified units)
        
        Notes
        -----
        Free space path loss increases with:
        - Distance (20 dB per decade)
        - Frequency (20 dB per decade)
        
        The calculation assumes:
        - Line of sight propagation
        - No atmospheric absorption
        - Far-field conditions (d >> lambda)
        
        Examples
        --------
        >>> tx = transmitter_info(10*cnv.dBW, 125*u.kHz, 1.0, 0.5*u.m, 1420*u.MHz)
        >>> tx.power_tx(10*u.MHz)
        >>> tx.satgain1d(0*u.deg)  # Boresight gain
        >>> power_at_ground = tx.fspl(500*u.km)
        >>> print(f"Received power: {power_at_ground}")
        
        See Also
        --------
        pycraf.conversions.free_space_loss : FSPL calculation function
        """
        FSPL = cnv.free_space_loss(sat_obs_dist,self.freq)
        sat_power= self.p_tx+FSPL+self.g_tx ### in dBm space
        return sat_power
    def custom_gain(self,el,az,gfunc):
        """
        Apply a custom antenna gain pattern function.
        
        Allows the use of custom gain functions for more complex antenna patterns
        beyond the simple 1D model. This is useful for modeling:
        - Phased array antennas
        - Multi-beam systems
        - Specific satellite antenna designs
        - Measured antenna patterns
        
        Parameters
        ----------
        el : float or array-like
            Elevation angle(s) in degrees, measured in the transmitter's
            reference frame
        az : float or array-like
            Azimuth angle(s) in degrees, measured in the transmitter's
            reference frame
        gfunc : callable
            Custom gain function that takes (el, az) as input and returns
            gain in dBi. The function should accept angles in degrees and
            return gain as a float or array.
            Signature: gfunc(el, az) -> gain (dBi)
        
        Returns
        -------
        G_tx : float or array-like
            Transmitter antenna gain at specified direction(s) (dBi)
        
        Examples
        --------
        >>> def my_antenna_pattern(el, az):
        ...     '''Custom phased array pattern'''
        ...     return 20 * np.cos(np.radians(el)) * np.cos(np.radians(az))
        >>> 
        >>> tx = transmitter_info(10*cnv.dBW, 125*u.kHz, 1.0, 0.5*u.m, 1420*u.MHz)
        >>> gain = tx.custom_gain(30, 45, my_antenna_pattern)
        
        Notes
        -----
        The custom function should:
        - Accept numpy arrays for vectorized calculations
        - Return values in dBi
        - Handle edge cases (e.g., elevation > 90°)
        
        See Also
        --------
        satgain1d : Standard 1D gain pattern
        """
        G_tx=gfunc(el,az)
        self.g_tx = G_tx
        return G_tx





class receiver_info():
    """
    Information holder class for radio telescope receiver characteristics.
    
    This class encapsulates the physical and operational parameters of a radio
    telescope receiver system. It handles antenna gain calculations considering
    pointing direction, source location, and antenna beam patterns. The class
    supports both single dish and array configurations.
    
    The receiver model uses standard radio astronomy antenna patterns based on
    ITU-R recommendations, with configurable aperture efficiency and system
    temperature. It can calculate gain patterns for both simple 1D models and
    complex custom functions.
    
    Attributes
    ----------
    d_rx : astropy.Quantity
        Diameter of the receiver telescope aperture (m)
    eta_a_rx : float
        Aperture efficiency of the receiver (dimensionless, 0 to 1)
        Typical values: 0.5-0.7 for radio telescopes
    location : cysgp4.PyObserver or array
        Observer location(s) on Earth (PyObserver object)
    freq : astropy.Quantity
        Center frequency of the receiver band (Hz)
    bandwidth : astropy.Quantity
        Receiver bandwidth (Hz)
    tsys : astropy.Quantity
        System temperature of the receiver (K), default 20 K
    G_rx : astropy.Quantity
        Calculated receiver antenna gain pattern (dBi), set by gain methods
    
    Examples
    --------
    >>> from astropy import units as u
    >>> from cysgp4 import PyObserver
    >>> 
    >>> # Create observer at SKA-Mid location
    >>> observer = PyObserver(lon=21.443888, lat=-30.713055, alt=1000)
    >>> 
    >>> # Create receiver for L-band observations
    >>> rx = receiver_info(
    ...     d_rx=13.5*u.m,           # SKA-Mid dish size
    ...     eta_a_rx=0.7,             # Aperture efficiency
    ...     pyobs=observer,
    ...     freq=1420*u.MHz,          # HI line
    ...     bandwidth=10*u.MHz,
    ...     tsys=25*u.K
    ... )
    >>> 
    >>> # Calculate gain pattern
    >>> gain = rx.antgain1d(
    ...     tp_az=180, tp_el=45,      # Telescope pointing
    ...     sat_obs_az=185, sat_obs_el=50  # Satellite position
    ... )
    
    Notes
    -----
    - The antenna pattern follows ITU-R RA.769 recommendations
    - Gain calculations use pycraf's ras_pattern function
    - System temperature affects sensitivity but not gain patterns
    
    See Also
    --------
    transmitter_info : Transmitter/satellite characteristics
    obs_sim : Main observation simulation class
    pycraf.antenna.ras_pattern : Underlying antenna pattern function
    """
    def __init__(self,d_rx,eta_a_rx,pyobs,freq,bandwidth,tsys=20*u.K):
        """
        Initialize receiver information object.
        
        Parameters
        ----------
        d_rx : astropy.Quantity
            Diameter of the receiver telescope aperture (m)
        eta_a_rx : float
            Aperture efficiency of the receiver telescope (0 to 1)
            Typical values: 0.5-0.7 for parabolic dishes
        pyobs : cysgp4.PyObserver or array of PyObserver
            Observer object(s) containing location information (lon, lat, alt)
        freq : astropy.Quantity
            Center frequency of the receiver band (Hz)
        bandwidth : astropy.Quantity
            Receiver bandwidth (Hz)
        tsys : astropy.Quantity, optional
            System temperature of the receiver (K), default 20 K
        """
        self.d_rx = d_rx
        self.eta_a_rx = eta_a_rx
        self.location = pyobs
        self.freq = freq
        self.bandwidth = bandwidth
        self.tsys = tsys
    
    def antgain1d(self,tp_az,tp_el,sat_obs_az,sat_obs_el):
        """
        Calculate 1-D receiver antenna gain pattern.
        
        Computes the receiver gain using the radio astronomy service (RAS) antenna
        pattern model from pycraf. The gain is calculated based on the angular
        separation between the telescope pointing direction and the source (satellite)
        position.
        
        This method handles array broadcasting for efficient calculation across
        multiple pointings, satellites, and time steps. It uses the standard
        ITU-R RA.769 antenna pattern for radio astronomy antennas.
        
        Parameters
        ----------
        tp_az : float or array-like
            Azimuth of the telescope pointing direction (degrees)
            Range: 0-360°, where 0° = North, 90° = East
        tp_el : float or array-like
            Elevation of the telescope pointing direction (degrees)
            Range: 0-90°, where 0° = horizon, 90° = zenith
        sat_obs_az : float or array-like
            Azimuth of the source/satellite (degrees)
        sat_obs_el : float or array-like
            Elevation of the source/satellite (degrees)
        
        Returns
        -------
        G_rx : astropy.Quantity
            Receiver antenna gain in the direction of the source (dBi)
            Shape matches the broadcasted shape of input arrays
        
        Notes
        -----
        The calculation process:
        1. Compute angular separation between pointing and source using
           spherical trigonometry
        2. Apply the RAS antenna pattern based on aperture size, efficiency,
           and wavelength
        3. Reshape output to match input dimensions
        
        For large arrays, this calculation may take significant time due to:
        - Multiple telescope pointings
        - Multiple satellites
        - Time-series propagation
        
        The RAS pattern includes:
        - Main beam with high gain
        - Near sidelobes (first few lobes)
        - Far sidelobe envelope
        
        Examples
        --------
        >>> rx = receiver_info(13.5*u.m, 0.7, observer, 1420*u.MHz, 10*u.MHz)
        >>> # Single pointing and source
        >>> gain = rx.antgain1d(180, 45, 185, 50)
        >>> print(f"Gain: {gain} dBi")
        >>> 
        >>> # Array of pointings (e.g., drift scan)
        >>> pointings_az = np.linspace(0, 360, 100)
        >>> pointings_el = np.ones(100) * 45
        >>> gains = rx.antgain1d(pointings_az, pointings_el, 185, 50)
        
        See Also
        --------
        custom_gain : For custom antenna pattern functions
        pycraf.antenna.ras_pattern : Underlying RAS pattern function
        pycraf.geometry.true_angular_distance : Angular separation calculation
        """
        print('Obtaining satellite and telescope pointing coordinates, calculation for large arrays may take a while...')
        ang_sep = geometry.true_angular_distance(tp_az*u.deg, tp_el*u.deg, sat_obs_az*u.deg, sat_obs_el *u.deg)
        print('Done. putting angular separation into gain pattern function')
        G_rx = antenna.ras_pattern(
            ang_sep.flatten(), self.d_rx, const.c / self.freq, self.eta_a_rx
            )
        
        self.G_rx = G_rx.reshape(ang_sep.shape)
        return self.G_rx
    
    def custom_gain(self,el_source,az_source,el_receiver,az_receiver,gfunc):
        """
        Apply a custom receiver antenna gain pattern function.
        
        Allows the use of custom gain functions for receiver antennas with
        non-standard patterns. This is useful for:
        - Phased array receivers (e.g., SKA-Low, LOFAR)
        - Multi-feed systems
        - Beam-formed arrays
        - Measured or simulated antenna patterns
        
        Unlike antgain1d, this method passes both source and receiver coordinates
        directly to the custom function, allowing for more complex gain calculations
        that may depend on both pointing and source positions independently.
        
        Parameters
        ----------
        el_source : float or array-like
            Elevation angle of source direction (degrees)
            Measured in topocentric coordinates from horizon
        az_source : float or array-like
            Azimuth angle of source direction (degrees)
            Measured in topocentric coordinates
        el_receiver : float or array-like
            Elevation angle of receiver pointing (degrees)
            Measured in topocentric coordinates from horizon
        az_receiver : float or array-like
            Azimuth angle of receiver pointing (degrees)
            Measured in topocentric coordinates
        gfunc : callable
            Custom gain function with signature:
            gfunc(el_source, az_source, el_receiver, az_receiver) -> gain (dBi)
            Should accept and return numpy arrays for vectorization
        
        Returns
        -------
        G_tx : float or array-like
            Receiver antenna gain at specified directions (dBi)
        
        Examples
        --------
        >>> def my_beam_pattern(el_s, az_s, el_r, az_r):
        ...     '''Custom beam pattern with asymmetric lobes'''
        ...     ang_sep = angular_separation(el_s, az_s, el_r, az_r)
        ...     return 30 - 20 * np.log10(1 + (ang_sep/2)**2)
        >>> 
        >>> rx = receiver_info(13.5*u.m, 0.7, observer, 1420*u.MHz, 10*u.MHz)
        >>> gain = rx.custom_gain(45, 180, 50, 185, my_beam_pattern)
        
        Notes
        -----
        The custom function should:
        - Accept vectorized inputs (numpy arrays)
        - Return gains in dBi
        - Handle coordinate edge cases
        - Be properly normalized to physical gain values
        
        See Also
        --------
        antgain1d : Standard RAS antenna pattern
        transmitter_info.custom_gain : Similar method for transmitters
        """
        G_tx=gfunc(el_source,az_source,el_receiver,az_receiver)
        self.g_tx = G_tx
        return G_tx

class obs_sim():
    """
    Main observation simulation class for satellite interference analysis.
    
    This class orchestrates comprehensive radio frequency interference (RFI)
    simulations for radio astronomy observations affected by satellite
    constellations. It manages the multi-dimensional simulation grid, handles
    satellite propagation, coordinate transformations, and interference
    calculations for both single dishes and interferometric arrays.
    
    The simulation operates on a 6-dimensional data structure:
        1. Observer locations/antennas
        2. Sky grid cells
        3. Antenna pointings per grid cell
        4. Observation epochs
        5. Time integrations within epochs
        6. Satellites in the constellation
    
    This structure enables comprehensive temporal, spatial, and configurational
    analysis of satellite interference scenarios.
    
    Attributes
    ----------
    receiver : receiver_info
        Receiver/telescope characteristics
    location : array of cysgp4.PyObserver
        Observer location(s), reshaped to [location,1,1,1,1,1]
    mjds : array
        Modified Julian Dates for observation times
    grid_az : array
        Azimuth grid for sky coverage (degrees)
    grid_el : array
        Elevation grid for sky coverage (degrees)
    grid_info : array
        Metadata for each grid cell (solid angles, boundaries)
    tles_list : array of PyTle
        Satellite TLE objects for propagation
    sat_info : dict
        Satellite position and coordinate information
    topo_pos_az, topo_pos_el, topo_pos_dist : arrays
        Satellite positions in topocentric coordinates
    satf_az, satf_el, satf_dist : arrays
        Observer positions in satellite reference frame
    pnt_coord : astropy.SkyCoord
        Telescope pointing coordinates
    pnt_az, pnt_el : astropy.Quantity
        Telescope pointing azimuth and elevation
    altaz_frame : astropy.AltAz
        Altitude-azimuth reference frame
    rxang_sep : astropy.Quantity
        Angular separation between telescope pointing and satellites
    txangsep : astropy.Quantity
        Angular separation in satellite reference frame
    bearings : array
        Baseline vectors for interferometry (ITRF coordinates)
    bearing_D : array
        Baseline lengths (meters)
    baseline_delays : array
        Geometric delays for each baseline (seconds)
    fringes : array
        Fringe amplitude patterns
    elevation_mask : array
        Boolean mask for satellites above elevation limit
    
    Examples
    --------
    >>> from scepter import obs, skynet
    >>> 
    >>> # Setup receiver and grid
    >>> receiver = obs.receiver_info(...)
    >>> skygrid = skynet.pointgen_S_1586_1(niters=1)
    >>> mjds = skynet.plantime(...)
    >>> 
    >>> # Create simulation
    >>> sim = obs.obs_sim(receiver, skygrid, mjds)
    >>> 
    >>> # Load or generate satellite data
    >>> sim.populate(tles_list, save=True)
    >>> # or
    >>> sim.load_propagation('satellite_info.npz')
    >>> 
    >>> # Set telescope pointing
    >>> sim.sky_track(ra=0*u.deg, dec=45*u.deg)
    >>> 
    >>> # Calculate interference
    >>> ang_sep = sim.sat_separation(mode='tracking')
    >>> gains = receiver.antgain1d(sim.pnt_az, sim.pnt_el,
    ...                            sim.topo_pos_az, sim.topo_pos_el)
    
    Notes
    -----
    The class supports two main observation modes:
    1. **Tracking mode**: Telescope follows a fixed celestial source
    2. **All-sky mode**: Survey observations with grid-based pointings
    
    For interferometric arrays:
    - Baseline calculations use ITRF2008 coordinates
    - Near-field corrections account for satellite distances
    - Fringe patterns include bandwidth smearing effects
    
    Performance considerations:
    - Large arrays (many antennas) increase computation time
    - Dense time sampling improves accuracy but slows calculation
    - Satellite propagation can be saved/loaded to avoid recomputation
    
    See Also
    --------
    transmitter_info : Satellite transmitter model
    receiver_info : Telescope receiver model
    cysgp4.propagate_many : Satellite propagation engine
    """
    def __init__(self,receiver,skygrid,mjds):
        """
        Initialize observation simulation.
        
        Parameters
        ----------
        receiver : receiver_info
            Receiver/telescope object with antenna and location information
        skygrid : tuple
            Output from skynet.pointgen function containing:
            - tel_az: azimuth grid (degrees)
            - tel_el: elevation grid (degrees)
            - grid_info: cell metadata (solid angles, boundaries)
        mjds : array
            Modified Julian Date array for observation epochs and times.
            Shape determines temporal resolution of simulation.
            Typically from skynet.plantime() function.
        
        Notes
        -----
        The constructor reshapes arrays to support broadcasting across all
        simulation dimensions. The location array becomes shape [N,1,1,1,1,1]
        and grid arrays become [1,M,K,1,1,1] where:
        - N = number of observers/antennas
        - M = number of grid cells
        - K = number of pointings per grid cell
        """

        # self.transmitter = transmitter
        self.receiver = receiver
        # self.ras_bandwidth = receiver.bandwidth
        # self.transmitter.power_tx(self.ras_bandwidth)
        # reformat and reorganise tle array dimension?
        ## in the order of [location,grid cell,antenna pointing per grid, epochs,time,satellite]
        
        self.location = receiver.location[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
        self.mjds = mjds
        tel_az, tel_el, self.grid_info = skygrid
        ### add axis for simulation over time and iterations
        self.grid_az=tel_az[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
        self.grid_el=tel_el[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]

    def azel_track(self,az,el):
        """
        Set fixed azimuth-elevation tracking for the telescope.
        
        Configures the telescope to point at a fixed direction in the local
        horizontal (Alt-Az) coordinate system. Unlike celestial tracking,
        this pointing direction does not account for Earth's rotation, so
        celestial sources will drift through the beam.
        
        This mode is useful for:
        - Satellite tracking at fixed look angles
        - Calibration observations
        - Survey observations with fixed pointing
        - RFI monitoring at specific directions
        
        Parameters
        ----------
        az : float
            Fixed azimuth direction (degrees)
            Range: 0-360°, where 0° = North, 90° = East, 180° = South, 270° = West
        el : float
            Fixed elevation angle (degrees)
            Range: 0-90°, where 0° = horizon, 90° = zenith
        
        Returns
        -------
        pnt_coord : astropy.coordinates.SkyCoord
            Pointing coordinates in Alt-Az frame
        
        Attributes Set
        --------------
        pnt_coord : astropy.SkyCoord
            Pointing direction in Alt-Az frame
        pnt_az : astropy.Quantity
            Fixed azimuth (constant across time)
        pnt_el : astropy.Quantity (stored as .alt)
            Fixed elevation (constant across time)
        altaz_frame : astropy.AltAz
            Local topocentric reference frame
        
        Examples
        --------
        >>> # Point telescope at 45° elevation, due south
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> pointing = sim.azel_track(az=180, el=45)
        >>> 
        >>> # Point at zenith for calibration
        >>> pointing = sim.azel_track(az=0, el=90)
        >>> 
        >>> # Monitor RFI from known satellite pass direction
        >>> pointing = sim.azel_track(az=225, el=30)
        
        Notes
        -----
        Unlike sky_track:
        - Pointing does NOT follow sidereal motion
        - Celestial sources drift at ~15°/hour in azimuth (varies with declination)
        - Useful for non-sidereal tracking or fixed monitoring
        
        The Alt-Az frame is tied to:
        - Observer location (first antenna in the array)
        - Observation time (from mjds)
        
        See Also
        --------
        sky_track : Celestial coordinate tracking (RA/Dec)
        """
        ant1 = self.location.flatten()[0]
        time_2d = Time(self.mjds, format='mjd', scale='utc')
        loc1 = EarthLocation(lat=ant1.loc.lat, lon=ant1.loc.lon, height=ant1.loc.alt)
        altaz = AltAz( obstime=time_2d, location=loc1)
        tel1_pnt = SkyCoord(az,el, unit=(u.deg,u.deg),frame=altaz)
        self.pnt_coord=tel1_pnt
        self.pnt_az, self.pnt_el = tel1_pnt.az, tel1_pnt.alt
        self.altaz_frame=altaz
        return self.pnt_coord

    def sky_track(self,ra,dec,frame='icrs'):
        """
        Set telescope tracking to follow a celestial source.
        
        Configures the telescope to track a fixed position in celestial coordinates
        (RA/Dec). The method transforms celestial coordinates to the topocentric
        Alt-Az frame for each time step, accounting for Earth's rotation and the
        observer's location.
        
        This is the standard observing mode for targeted observations of
        astronomical sources (pulsars, galaxies, HI regions, etc.).
        
        Parameters
        ----------
        ra : float or array-like
            Right ascension of the source (degrees)
            Range: 0-360°, measured eastward from vernal equinox
        dec : float or array-like
            Declination of the source (degrees)
            Range: -90 to +90°, measured from celestial equator
        frame : str or astropy.coordinates frame, optional
            Celestial coordinate frame (default: 'icrs')
            Options: 'icrs', 'fk5', 'galactic', or astropy frame object
            Note: Can also pass astropy AltAz object for direct Az/El tracking
        
        Returns
        -------
        tel1_pnt : astropy.coordinates.SkyCoord
            Telescope pointing in Alt-Az frame at all observation times.
            Can be used to extract azimuth and elevation arrays.
        
        Attributes Set
        --------------
        pnt_coord : astropy.SkyCoord
            Source coordinates in input frame
        pnt_az : astropy.Quantity
            Time-varying azimuth of source (degrees)
        pnt_el : astropy.Quantity
            Time-varying elevation of source (degrees)
        altaz_frame : astropy.AltAz
            Local topocentric frame for coordinate transformations
        
        Examples
        --------
        >>> # Track the Crab Nebula
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> pointing = sim.sky_track(ra=83.633, dec=22.014)  # ICRS
        >>> print(f"Elevation range: {pointing.alt.min():.1f} to {pointing.alt.max():.1f}")
        >>> 
        >>> # Track a Galactic HI region
        >>> sim.sky_track(ra=100.0, dec=0.0, frame='galactic')
        >>> 
        >>> # Direct Az/El tracking (use azel_track instead)
        >>> # sim.azel_track(az=180, el=45)
        
        Notes
        -----
        The transformation accounts for:
        - Precession and nutation
        - Aberration of starlight
        - Observer's location (latitude, longitude, altitude)
        - Time of observation (mjds)
        
        Sources near the horizon (el < 10°) may be affected by:
        - Atmospheric refraction (not corrected here)
        - Increased system temperature
        - Reduced antenna efficiency
        
        See Also
        --------
        azel_track : Fixed azimuth-elevation tracking
        astropy.coordinates.SkyCoord : Coordinate transformation
        """
        # self.pnt_ra = ra
        # self.pnt_dec = dec
        ant1 = self.location.flatten()[0]
        time_2d = Time(self.mjds, format='mjd', scale='utc')
        loc1 = EarthLocation(lat=ant1.loc.lat, lon=ant1.loc.lon, height=ant1.loc.alt)
        altaz = AltAz( obstime=time_2d, location=loc1)
        skycoord_track=SkyCoord(ra,dec, unit=(u.deg,u.deg),frame=frame)
        self.pnt_coord=skycoord_track
        self.altaz_frame=altaz
        tel1_pnt=skycoord_track.transform_to(altaz)
        self.pnt_az, self.pnt_el = tel1_pnt.az, tel1_pnt.alt
        return tel1_pnt

    def load_propagation(self,nparray):
        """
        Load pre-computed satellite propagation data from file.
        
        Loads satellite position and coordinate data that was previously saved
        by the populate() method. This avoids expensive recomputation of
        satellite propagation for repeated analyses of the same scenario.
        
        Parameters
        ----------
        nparray : str
            Path to numpy .npz file containing satellite propagation data
            File should be created by populate() method with save=True
        
        Returns
        -------
        tleprop : numpy structured array
            Loaded propagation data containing all position and coordinate arrays
        
        Attributes Set
        --------------
        sat_info : numpy structured array
            Complete propagation dataset
        topo_pos_az : array
            Topocentric azimuth angles (degrees)
        topo_pos_el : array
            Topocentric elevation angles (degrees)
        topo_pos_dist : array
            Slant distances to satellites (km)
        satf_az : array
            Observer azimuth in satellite frame (degrees)
        satf_el : array
            Observer elevation in satellite frame (degrees)
        satf_dist : array
            Observer distance in satellite frame (km)
        
        Notes
        -----
        The loaded data must match the simulation configuration:
        - Same observer locations
        - Same time array (mjds)
        - Same satellite constellation (TLEs)
        
        Mismatched configurations will lead to incorrect results or errors.
        
        The file format is numpy's compressed .npz format with named arrays:
        - obs_az, obs_el, obs_dist: topocentric coordinates
        - sat_frame_az, sat_frame_el, sat_frame_dist: satellite frame
        
        Examples
        --------
        >>> # First run: compute and save
        >>> sim1 = obs_sim(receiver, skygrid, mjds)
        >>> sim1.populate(tles, save=True, savename='constellation.npz')
        >>> 
        >>> # Later runs: load from file
        >>> sim2 = obs_sim(receiver, skygrid, mjds)
        >>> sim2.load_propagation('constellation.npz')
        >>> # Continue with analysis...
        >>> sim2.sky_track(ra=0, dec=45)
        
        See Also
        --------
        populate : Compute and save propagation data
        """

        tleprop=np.load(nparray,allow_pickle=True)
        self.sat_info = tleprop
        #### obtain coordinates in observation frame and satellite frame
        topo_pos_az, topo_pos_el, topo_pos_dist = tleprop['obs_az'], tleprop['obs_el'], tleprop['obs_dist']
        satf_az, satf_el, satf_dist = tleprop['sat_frame_az'], tleprop['sat_frame_el'] , tleprop['sat_frame_dist']

        self.topo_pos_az = topo_pos_az
        self.topo_pos_el = topo_pos_el
        self.topo_pos_dist = topo_pos_dist
        self.satf_az = satf_az
        self.satf_el = satf_el
        self.satf_dist = satf_dist

    def reduce_sats(self,el_limit=0):
        """
        Filter satellites by elevation angle threshold.
        
        Reduces the satellite dataset by removing satellites that never rise
        above a specified elevation limit. This improves computational efficiency
        by excluding satellites that are below the horizon or at low elevations
        where they have minimal impact or are not visible to the telescope.
        
        The method applies a time-averaged elevation mask, retaining only
        satellites whose mean elevation exceeds the threshold across all
        observation times, grid cells, and pointings.
        
        Parameters
        ----------
        el_limit : float, optional
            Minimum mean elevation angle threshold (degrees)
            Default: 0° (horizon)
            Common values: 10-20° to avoid low-elevation issues
        
        Attributes Modified
        -------------------
        topo_pos_az : array
            Filtered azimuth positions
        topo_pos_el : array
            Filtered elevation positions
        topo_pos_dist : array
            Filtered distances
        satf_az : array
            Filtered satellite frame azimuth
        satf_el : array
            Filtered satellite frame elevation
        satf_dist : array
            Filtered satellite frame distances
        elevation_mask : array (new)
            Boolean mask indicating retained satellites
        
        Notes
        -----
        Benefits of elevation filtering:
        - Reduces memory usage (fewer satellites to track)
        - Speeds up computation (fewer calculations)
        - Removes satellites with low antenna gain (below horizon)
        - Avoids atmospheric effects at low elevations
        
        The threshold is applied to the *mean* elevation across:
        - All time steps
        - All observer locations
        - All sky grid cells
        - All pointing directions
        
        A satellite is retained if its mean elevation exceeds el_limit.
        
        Considerations:
        - Higher thresholds (>20°) may exclude rising/setting passes
        - 0° keeps only satellites above horizon sometime during observation
        - Negative values keep all satellites (no filtering)
        - For RFI analysis, consider main beam response (typically >10°)
        
        Examples
        --------
        >>> # Remove satellites below horizon
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> sim.populate(tles)
        >>> print(f"Initial satellites: {sim.topo_pos_az.shape[-1]}")
        >>> sim.reduce_sats(el_limit=0)
        >>> print(f"After filtering: {sim.topo_pos_az.shape[-1]}")
        >>> 
        >>> # More aggressive filtering
        >>> sim.reduce_sats(el_limit=15)  # Only above 15° mean elevation
        >>> 
        >>> # Check which satellites were kept
        >>> kept_fraction = sim.elevation_mask.sum() / len(sim.elevation_mask)
        >>> print(f"Retained {kept_fraction:.1%} of satellites")
        
        See Also
        --------
        populate : Generate satellite propagation data
        """

        min_el=np.mean(self.topo_pos_el,axis=(0,1,2,3,4))
        mask = min_el>el_limit
        # self.sat_info["obs_az"] = self.sat_info["obs_az"][:,:,:,:,:,mask]
        # self.sat_info["obs_el"] = self.sat_info["obs_el"][:,:,:,:,:,mask]
        # self.sat_info["obs_dist"] = self.sat_info["obs_dist"][:,:,:,:,:,mask]
        # self.sat_info["sat_frame_az"] = self.sat_info["sat_frame_az"][:,:,:,:,:,mask]
        # self.sat_info["sat_frame_el"] = self.sat_info["sat_frame_el"][:,:,:,:,:,mask]
        # self.sat_info["sat_frame_dist"] = self.sat_info["sat_frame_dist"][:,:,:,:,:,mask]

        self.topo_pos_az = self.topo_pos_az[:,:,:,:,:,mask]
        self.topo_pos_el = self.topo_pos_el[:,:,:,:,:,mask]
        self.topo_pos_dist = self.topo_pos_dist[:,:,:,:,:,mask]
        self.satf_az = self.satf_az[:,:,:,:,:,mask]
        self.satf_el = self.satf_el[:,:,:,:,:,mask]
        self.satf_dist = self.satf_dist[:,:,:,:,:,mask]
        self.elevation_mask = mask
        
    def populate(self,tles_list,save=True, savename="satellite_info.npz"):
        """
        Populate the simulation with satellite propagation data.
        
        This method propagates all satellites in the constellation across all
        observation times and observer locations using cysgp4's high-performance
        propagator. It calculates both topocentric (observer frame) and satellite
        frame coordinates for each satellite-observer-time combination.
        
        The propagation results are cached to a numpy file to avoid expensive
        recomputation in subsequent runs.
        
        Parameters
        ----------
        tles_list : list of cysgp4.PyTle
            List of Two-Line Element (TLE) objects for satellites to simulate
        save : bool, optional
            Whether to save propagation results to file (default: True)
        savename : str, optional
            Filename for saved propagation data (default: 'satellite_info.npz')
        
        Returns
        -------
        sat_info : numpy structured array
            Satellite propagation information containing:
            - obs_az: topocentric azimuth (degrees)
            - obs_el: topocentric elevation (degrees)
            - obs_dist: slant distance to satellite (km)
            - sat_frame_az: observer azimuth in satellite frame (degrees)
            - sat_frame_el: observer elevation in satellite frame (degrees)
            - sat_frame_dist: observer distance in satellite frame (km)
        
        Notes
        -----
        The satellite frame orientation uses 'zxy' convention:
        - Z-axis: satellite velocity vector
        - X-axis: perpendicular to velocity in orbital plane
        - Y-axis: completes right-handed system
        
        Computation time scales with:
        - Number of satellites
        - Number of time steps
        - Number of observer locations
        
        For large constellations (>1000 satellites) and dense time sampling,
        expect computation times of several minutes.
        
        Examples
        --------
        >>> from cysgp4 import PyTle
        >>> 
        >>> # Load TLEs for Starlink constellation
        >>> tles = [PyTle(name, line1, line2) for name, line1, line2 in tle_data]
        >>> 
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> sim.populate(tles, save=True, savename='starlink_passes.npz')
        >>> 
        >>> # Later, load without recomputing
        >>> sim2 = obs_sim(receiver, skygrid, mjds)
        >>> sim2.load_propagation('starlink_passes.npz')
        
        See Also
        --------
        load_propagation : Load pre-computed propagation data
        cysgp4.propagate_many : Underlying propagation function
        """
        self.tles_list = tles_list[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]
        observatories = self.location
        mjds = self.mjds
        tles = self.tles_list
        print(observatories.shape,tles.shape,mjds.shape)
        print('Obtaining satellite and time information, propagation for large arrays may take a while...')
        result = cysgp4.propagate_many(mjds,tles,observers=observatories,do_eci_pos=True, do_topo=True, do_obs_pos=True, do_sat_azel=True,sat_frame='zxy') 
        print('Done. Satellite coordinates obtained')
        
        # self.eci_pos = result['eci_pos']
        topo_pos = result['topo']
        sat_azel = result['sat_azel']  ### check cysgp4 for satellite frame orientation description

        # eci_pos_x, eci_pos_y, eci_pos_z = (eci_pos[..., i] for i in range(3))
        self.topo_pos_az, self.topo_pos_el, self.topo_pos_dist, _ = (topo_pos[..., i] for i in range(4))
        
        ### this means azimuth and elevation of the observer, I think the naming is a bit confusing
        self.satf_az, self.satf_el, self.satf_dist = (sat_azel[..., i] for i in range(3))  
        if save == True:
            np.savez(savename,obs_az=self.topo_pos_az,obs_el=self.topo_pos_el,obs_dist=self.topo_pos_dist,sat_frame_az=self.satf_az,sat_frame_el=self.satf_el,sat_frame_dist=self.satf_dist)
        tleprop=np.load(savename,allow_pickle=True)
        self.sat_info = tleprop

    def txbeam_angsep(self,beam_el,beam_az):
        """
        Calculate angular separation in satellite transmitter reference frame.
        
        Computes the angular separation between the satellite's beam pointing
        direction and the observer's position, as measured in the satellite's
        body-fixed reference frame. This determines the transmitter antenna
        gain towards the observer.
        
        Parameters
        ----------
        beam_el : float
            Beam elevation angle in satellite reference frame (degrees)
            ZXY convention: Z = velocity vector
        beam_az : float
            Beam azimuth angle in satellite reference frame (degrees)
        
        Returns
        -------
        txangsep : astropy.Quantity
            Angular separation between beam and observer (degrees)
            Shape matches simulation dimensions
        
        Attributes Set
        --------------
        txangsep : astropy.Quantity
            Stored angular separations
        
        Notes
        -----
        Uses satellite frame coordinates (satf_az, satf_el) set by populate()
        or load_propagation().
        
        The satellite frame (ZXY):
        - Z-axis: Velocity vector (orbital motion direction)
        - X-axis: Perpendicular to velocity in orbital plane
        - Y-axis: Roughly radial (toward/away from Earth)
        
        This is used to determine transmitter gain G_tx(θ) from the
        transmitter_info.satgain1d or custom_gain methods.
        
        Examples
        --------
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> sim.populate(tles)
        >>> 
        >>> # Nadir-pointing beam (toward Earth)
        >>> sep_nadir = sim.txbeam_angsep(beam_el=-90, beam_az=0)
        >>> 
        >>> # Forward-pointing beam
        >>> sep_forward = sim.txbeam_angsep(beam_el=0, beam_az=0)
        
        See Also
        --------
        sat_frame_pointing : Underlying calculation
        transmitter_info.satgain1d : Calculate gain from separation
        """
        self.txangsep,_,_=sat_frame_pointing(self.satf_az,self.satf_el,beam_el,beam_az)
        return self.txangsep
    def sat_separation(self,mode='tracking',pnt_az=None,pnt_el=None):
        """
        Calculate angular separation between telescope pointing and satellites.
        
        Computes the angular distance from the telescope's pointing direction
        to each satellite at each time step. This is crucial for determining
        antenna gain towards satellites and assessing RFI impact.
        
        Supports multiple observation modes with different pointing strategies.
        
        Parameters
        ----------
        mode : str, optional
            Observation mode (default: 'tracking')
            Options:
            - 'tracking': Telescope tracks celestial source (use sky_track or azel_track)
            - 'allsky': Survey mode using sky grid pointings
            - 'pnt': Custom pointing specified by pnt_az and pnt_el
        pnt_az : float or array, optional
            Azimuth for custom pointing (degrees)
            Required if mode='pnt'
        pnt_el : float or array, optional
            Elevation for custom pointing (degrees)
            Required if mode='pnt'
        
        Returns
        -------
        rxang_sep : astropy.Quantity
            Angular separation between pointing and satellites (degrees)
            Shape: matches simulation dimensions
            [locations, grid_cells, pointings, epochs, times, satellites]
        
        Attributes Set
        --------------
        rxang_sep : astropy.Quantity
            Computed angular separations
        
        Notes
        -----
        Angular separation determines:
        - Receiver antenna gain G_rx(θ)
        - Main beam vs sidelobe response
        - Sensitivity to satellite RFI
        - Whether satellite is in field of view
        
        Typical antenna patterns:
        - Main beam: 0-3° from boresight (high gain)
        - Near sidelobes: 3-10° (medium gain)
        - Far sidelobes: >10° (low gain, but many satellites)
        
        The calculation uses spherical trigonometry to account for the
        curved celestial sphere, providing accurate separations even for
        large angles.
        
        Mode details:
        'tracking': Uses self.pnt_az and self.pnt_el set by sky_track or azel_track
        'allsky': Uses grid_az and grid_el for all-sky survey
        'pnt': Uses user-provided pnt_az and pnt_el
        
        Examples
        --------
        >>> # Tracking mode: follow a source
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> sim.populate(tles)
        >>> sim.sky_track(ra=0, dec=45)
        >>> separations = sim.sat_separation(mode='tracking')
        >>> print(f"Separation range: {separations.min():.1f} to {separations.max():.1f}")
        >>> 
        >>> # All-sky survey mode
        >>> separations = sim.sat_separation(mode='allsky')
        >>> 
        >>> # Custom pointing
        >>> separations = sim.sat_separation(mode='pnt', pnt_az=180, pnt_el=60)
        >>> 
        >>> # Find satellites in main beam (within 2°)
        >>> in_beam = separations < 2*u.deg
        >>> n_in_beam = in_beam.sum()
        >>> print(f"{n_in_beam} satellite-time instances in main beam")
        
        See Also
        --------
        sky_track : Set celestial tracking
        azel_track : Set fixed Az/El pointing
        receiver_info.antgain1d : Calculate gain from separations
        pycraf.geometry.true_angular_distance : Underlying calculation
        """
        if mode == 'tracking':
            self.rxang_sep = geometry.true_angular_distance(self.pnt_az, self.pnt_el, self.topo_pos_az*u.deg, self.topo_pos_el *u.deg)
        elif mode == 'allsky':
            self.rxang_sep = geometry.true_angular_distance(self.grid_az*u.deg, self.grid_el*u.deg, self.topo_pos_az*u.deg, self.topo_pos_el *u.deg)
        elif mode == 'pnt':
            self.rxang_sep = geometry.true_angular_distance(pnt_az*u.deg, pnt_el*u.deg, self.topo_pos_az*u.deg, self.topo_pos_el *u.deg)
        return self.rxang_sep

    def create_baselines(self):
        """
        Initialize baseline arrays for interferometric array observations.
        
        Calculates all baseline pairs, vectors, and lengths for the antenna
        array. This is required before performing interferometric fringe
        calculations. The method computes physical baselines in ITRF2008
        coordinates from the observer locations.
        
        Attributes Set
        --------------
        baselines : itertools.combinations object
            All unique antenna pairs (i, j) where i < j
        bearings : array
            Baseline vectors in ITRF2008 coordinates (meters)
            Shape: [n_antennas, 3] for [X, Y, Z] components
        bearing_D : array
            Baseline lengths (meters)
            Reshaped to match simulation dimensions
        
        Notes
        -----
        This method must be called before:
        - baselines_nearfield_delays()
        - sat_fringe()
        - fringe_signal()
        
        The baseline calculations use the reference antenna (first in the array)
        as the origin. All baselines are measured from this reference.
        
        Baseline pairs are generated using itertools.combinations to avoid
        duplicate baselines (i.e., only (i,j) not (j,i) for i < j).
        
        Examples
        --------
        >>> # Setup interferometric array
        >>> observers = [PyObserver(...) for _ in range(4)]  # 4 antennas
        >>> receiver = receiver_info(..., pyobs=np.array(observers), ...)
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> sim.populate(tles)
        >>> 
        >>> # Calculate baselines
        >>> sim.create_baselines()
        >>> print(f"Number of antennas: {len(observers)}")
        >>> print(f"Baseline lengths: {sim.bearing_D}")
        >>> 
        >>> # Continue with fringe calculations
        >>> sim.baselines_nearfield_delays(mode='tracking')
        >>> sim.sat_fringe(bwchan=10*u.MHz, fch1=1420*u.MHz)
        
        See Also
        --------
        baseline_pairs : Underlying calculation
        baselines_nearfield_delays : Calculate geometric delays
        sat_fringe : Calculate fringe patterns
        """
        from itertools import combinations
        antennas = self.receiver.location
        self.baselines = combinations(range(len(antennas)), 2)
        self.bearings, self.bearing_D = baseline_pairs(antennas)
        # self.bearings = self.bearings.reshape(self.location.shape)
        self.bearing_D = self.bearing_D.reshape(self.location.shape)
        # self.delays = mod_tau(self.baselines*u.m)

    def baselines_nearfield_delays(self,mode = 'tracking'):
        """
        Calculate near-field geometric delays for all baselines.
        
        Computes the differential time delays between antennas in an
        interferometric array when observing satellites. Accounts for both
        far-field geometric delays and near-field wavefront curvature effects.
        
        This is essential for accurate interferometric observations of satellites,
        where the near-field corrections can be significant due to the finite
        distance to the source.
        
        Parameters
        ----------
        mode : str, optional
            Observation mode (default: 'tracking')
            Options:
            - 'tracking': Use telescope pointing from sky_track or azel_track
            - 'allsky': Use sky grid pointings
        
        Returns
        -------
        baseline_delays : array
            Geometric delays for each baseline (seconds)
            Shape: matches simulation dimensions
            [locations, grid_cells, pointings, epochs, times, satellites]
        
        Attributes Set
        --------------
        pnt_tau : astropy.Quantity
            Far-field geometric delays for the pointing direction
        baseline_delays : array
            Total delays including near-field corrections
        
        Notes
        -----
        The calculation proceeds in two steps:
        1. Far-field delay: τ_ff = (D · s) / c
           where D is baseline vector, s is source unit vector
        2. Near-field correction: τ_nf = (|r1 - r_sat| - |r2 - r_sat|) / c
           where r1, r2 are antenna positions, r_sat is satellite position
        
        Near-field effects are significant when:
            D² / (8 * d_sat) ≳ λ
        where D = baseline, d_sat = satellite distance, λ = wavelength
        
        For LEO satellites (500-2000 km) and baselines >100 m, near-field
        corrections are typically important.
        
        The method uses:
        - Observer location (latitude) for coordinate transformations
        - Satellite distances from propagation data
        - Pointing direction (from mode parameter)
        - Baseline vectors (from create_baselines)
        
        Examples
        --------
        >>> # Setup and calculate delays
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> sim.populate(tles)
        >>> sim.sky_track(ra=0, dec=45)
        >>> sim.create_baselines()
        >>> 
        >>> # Calculate near-field delays
        >>> delays = sim.baselines_nearfield_delays(mode='tracking')
        >>> print(f"Delay range: {delays.min():.2e} to {delays.max():.2e} seconds")
        >>> 
        >>> # Typical delay for 1 km baseline, 500 km satellite
        >>> # Expected: ~10 nanoseconds to ~10 microseconds
        >>> 
        >>> # All-sky mode
        >>> delays_allsky = sim.baselines_nearfield_delays(mode='allsky')
        
        See Also
        --------
        create_baselines : Initialize baseline arrays
        baseline_nearfield_delay : Underlying delay calculation
        mod_tau : Far-field geometric delay
        sat_fringe : Use delays to calculate fringes
        """


        lat = self.location.flatten()[0].loc.lat
        l1 = self.topo_pos_dist[0][np.newaxis,:]*u.km
        if mode == 'tracking':
            tau1=mod_tau(self.pnt_az,self.pnt_el,lat,self.bearing_D*u.m)
            self.pnt_tau = tau1
            self.baseline_delays = baseline_nearfield_delay(l1,self.topo_pos_dist*u.km,tau=tau1)
        elif mode == 'allsky':
            tau1=mod_tau(self.grid_az,self.grid_el,lat,self.bearing_D*u.m)
            self.pnt_tau = tau1
            self.baseline_delays = baseline_nearfield_delay(l1,self.topo_pos_dist*u.km,tau=tau1)
        else:
            raise ValueError("Invalid mode. Choose 'tracking' or 'allsky'.")
        return self.baseline_delays
    
    def sat_fringe(self,bwchan,fch1,chan_bin=100):
        """
        Calculate fringe patterns for satellite observations.
        
        Computes the bandwidth-averaged fringe response for all baselines,
        satellites, and time steps. Integrates the fringe response across
        the observing bandwidth to account for bandwidth smearing effects.
        
        This method must be called after baselines_nearfield_delays() to
        use the computed geometric delays.
        
        Parameters
        ----------
        bwchan : astropy.Quantity
            Channel bandwidth for integration (Hz, kHz, MHz)
        fch1 : astropy.Quantity
            Channel center frequency (Hz, kHz, MHz)
        chan_bin : int, optional
            Number of frequency bins for bandwidth integration (default: 100)
            Higher values give more accurate results but slower computation
        
        Returns
        -------
        fringes : array
            Fringe amplitude patterns (dimensionless, range: -1 to +1)
            Shape matches simulation dimensions
            Positive values: constructive interference
            Negative values: destructive interference
        
        Attributes Set
        --------------
        fringes : array
            Computed fringe patterns
        
        Notes
        -----
        The fringe response is:
            F(τ, f) = <cos(2π f τ)>_bandwidth
        where τ is the geometric delay, and the average is over the
        channel bandwidth.
        
        Bandwidth smearing reduces fringe amplitude when:
            Δf * τ ≳ 1
        where Δf is the channel bandwidth.
        
        For satellite observations:
        - Near-field delays can be large (microseconds)
        - Wide bandwidths cause significant smearing
        - Narrow channels preserve fringe amplitude better
        
        This method flattens the delay array, computes fringes efficiently,
        then reshapes to the original dimensions.
        
        Examples
        --------
        >>> # Calculate fringes for L-band observation
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> sim.populate(tles)
        >>> sim.sky_track(ra=0, dec=45)
        >>> sim.create_baselines()
        >>> sim.baselines_nearfield_delays(mode='tracking')
        >>> 
        >>> # Compute fringe patterns
        >>> fringes = sim.sat_fringe(
        ...     bwchan=10*u.MHz,
        ...     fch1=1420*u.MHz,
        ...     chan_bin=100
        ... )
        >>> print(f"Fringe amplitude range: {fringes.min():.3f} to {fringes.max():.3f}")
        >>> 
        >>> # Narrow band (less smearing)
        >>> fringes_narrow = sim.sat_fringe(bwchan=100*u.kHz, fch1=1420*u.MHz)
        >>> 
        >>> # Wide band (more smearing)
        >>> fringes_wide = sim.sat_fringe(bwchan=100*u.MHz, fch1=1420*u.MHz)
        
        See Also
        --------
        baselines_nearfield_delays : Calculate delays (must run first)
        bw_fringe : Underlying bandwidth integration
        fringe_signal : Combine fringes with power and gain
        """


        delays= self.baseline_delays.flatten()
        self.fringes=bw_fringe(delays,bwchan,fch1,chan_bin=chan_bin).reshape(self.baseline_delays.shape)

        return self.fringes

    def fringe_signal(self,pwr,g_rx,ant1_idx=0,ant2_idx=1):
        """
        Calculate interferometric power for a specific baseline.
        
        Combines incident power, receiver gain, and fringe patterns to compute
        the correlated power for a two-element baseline. This represents the
        actual interferometric signal that would be measured by a correlator.
        
        The calculation assumes:
        - Equal effective collecting areas
        - Coherent integration
        - Two-element correlation (can extend to more baselines)
        
        Parameters
        ----------
        pwr : float or array
            Incident power at the receivers (linear units, not dB)
            Can be dimensionless for relative calculations (use 1.0)
            Or actual power values from link budget
        g_rx : astropy.Quantity
            Receiver antenna gain in source direction (dBi or dimless)
            From receiver_info.antgain1d() or custom_gain()
        ant1_idx : int, optional
            Index of first antenna in baseline (default: 0)
        ant2_idx : int, optional
            Index of second antenna in baseline (default: 1)
        
        Returns
        -------
        pwr : array
            Correlated power for the baseline
            Shape matches simulation dimensions
            Units: same as input power (if pwr=1, gives fringe attenuation factor)
        
        Attributes Set
        --------------
        fringe_pwr : array
            Stored correlated power
        
        Notes
        -----
        The correlated power is:
            P_corr = √(P₁ * G₁ * F²) * √(P₂ * G₂ * F²)
                   = P * G * F²
        where:
        - P: incident power (assumed equal for both antennas)
        - G: antenna gain (from g_rx)
        - F: fringe amplitude (from sat_fringe)
        
        For equal antennas (typical case):
            P_corr = P * G * F²
        
        The squared fringe term (F²) represents the correlation coefficient.
        
        Usage scenarios:
        1. Attenuation factor: Set pwr=1 to get relative response
        2. Absolute power: Use actual power values from link budget
        3. Multiple baselines: Call repeatedly with different indices
        
        Examples
        --------
        >>> # Complete interferometric RFI calculation
        >>> sim = obs_sim(receiver, skygrid, mjds)
        >>> sim.populate(tles)
        >>> sim.sky_track(ra=0, dec=45)
        >>> 
        >>> # Setup interferometry
        >>> sim.create_baselines()
        >>> sim.baselines_nearfield_delays(mode='tracking')
        >>> sim.sat_fringe(bwchan=10*u.MHz, fch1=1420*u.MHz)
        >>> 
        >>> # Calculate receiver gain
        >>> g_rx = receiver.antgain1d(sim.pnt_az, sim.pnt_el,
        ...                           sim.topo_pos_az, sim.topo_pos_el)
        >>> 
        >>> # Get correlated power for baseline 0-1
        >>> power_01 = sim.fringe_signal(pwr=1.0, g_rx=g_rx, ant1_idx=0, ant2_idx=1)
        >>> 
        >>> # Other baselines
        >>> power_02 = sim.fringe_signal(pwr=1.0, g_rx=g_rx, ant1_idx=0, ant2_idx=2)
        >>> power_12 = sim.fringe_signal(pwr=1.0, g_rx=g_rx, ant1_idx=1, ant2_idx=2)
        
        See Also
        --------
        sat_fringe : Calculate fringe patterns (must run first)
        receiver_info.antgain1d : Calculate receiver gains
        prx_cnv : Calculate incident power
        """
        coherent_v_baselines=np.sqrt(pwr*g_rx.to(cnv.dimless)*(self.fringes**2))
        fringe1 = coherent_v_baselines[ant1_idx]
        fringe2 = coherent_v_baselines[ant2_idx]
        pwr = fringe1*fringe2
        self.fringe_pwr = pwr
        return pwr