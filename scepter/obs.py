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
    calculate the bearing of antenna 2 with respect to antenna 1
    Args:
        ref (object): reference antenna/location PyObserver object
        ant (object): antenna for baseline PyObserver object
    Returns:
        bearing: vector from antenna baseline in cartesian coordinates
        d: baseline vector modulus or baseline length in meters
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
    calculate the effective baseline distance with a confirmed pointing angle of the reference antenna/position
    Args:
        d (float): distance to the antenna in meters
        az (float): azimuth angle in radians
        el (float): elevation angle in radians
        lat (float): latitude of the antenna in radians
    Returns:
        vector: array of the effective baseline vector in cartesian coordinates x,y,z (meters)

    """
    
    
    return d*np.array([np.cos(lat)*np.sin(el)-np.sin(lat)*np.cos(el)*np.cos(az),
    np.cos(el)*np.sin(az),
    np.sin(lat)*np.sin(el)+np.cos(lat)*np.cos(el)*np.cos(az)]) # x,y,z coordinates in meters
    

def mod_tau(az,el,lat,D):
    """
    Calculate the delay difference from astronomical source pointing in seconds for a given angle for large arrays
    Args:
        baseline (quantity): Baseline length in meters etc.
    Returns:
        tau (quantity): delay in seconds
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
    Calculate the delay difference from source pointing in seconds for a given angle
    Args:
        l1 (quantity): distance to the antenna 1 in distance units
        l2 (quantity): distance to the antenna 2 in distance units
        tau (quantity): baseline delay between two antennas in time units
    Returns:
        delay (quantity): delay in seconds
    """
    c = 3e8 *u.m/u.s  # speed of light in m/s
    l1 = l1.to(u.m) # Convert distance to meters
    l2 = l2.to(u.m) # Convert distance to meters
    
    return  (l1-l2)/c-tau

def fringe_attenuation(theta, baseline, bandwidth):
    """
    Calculate the fringe attenuation for a given angle, baseline, frequency, and bandwidth.
    Args:
        theta (quantity): off phase center Angle in radians/degrees etc.
        baseline (quantity): Baseline east-west component in meters etc.
        bandwidth (quantity): Bandwidth in Hz etc.
    """
    c = 3e8  # speed of light in m/s
    theta = theta.to(u.rad).value  # Convert angle to radians
    baseline = baseline.to(u.m).value  # Convert baseline to meters
    bandwidth = bandwidth.to(u.Hz).value  # Convert bandwidth to Hz
    return np.sinc(np.sin(theta)*baseline*bandwidth/c)

def fringe_response(delay,frequency):
    """
    Calculate the fringe response for a given delay and frequency
    based on two element equation integration, assuming equal gain

    Args:
        delay (quantity): delay in seconds
        frequency (quantity): frequency in Hz
    Returns:
        response (quantity): fringe response
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
    '''
    Description: quick function to convert power flux density from dBm to Jansky

    Parameters:
    pfd: float
        power flux density in dB W, dB W/m2/Hz
    frequency_GHz: float

    Returns:
    F_Jy: float
        power flux density in Jansky (Jy), (1 Jy = 10^-26 W/m^2/Hz)
    '''

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
    def __init__(self,d_rx,eta_a_rx,pyobs,freq,bandwidth,tsys=20*u.k):
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
        Description: Load the satellite propagation data from a numpy array file
        Parameters:
        nparray: str
            path to the numpy array file containing the satellite propagation data
        Returns:
        tleprop: numpy array
            numpy array containing the satellite propagation data
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
        '''
        Description: This function reduces the number of satellites in the simulation by applying a limit to the elevation angle
        Parameters:
        el_limit: float
            elevation angle limit (degrees)
        '''

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
        '''
        Description: Calculate the satellite pointing angle separation to the observer in the satellite reference frame

        Parameters:
        beam_el: float
            beam elevation angle in satellite reference frame zxy, where z is the motion vector
        beam_az: float  
            beam azimuth angle in satellite reference frame
        
        Returns:
        txang_sep: float
            angular separation between the satellite pointing and observer in the satellite reference frame
        '''
        self.txangsep,_,_=sat_frame_pointing(self.satf_az,self.satf_el,beam_el,beam_az)
        return self.txangsep
    def sat_separation(self,mode='tracking',pnt_az=None,pnt_el=None):
        '''
        Description: Calculate the satellite angular separation from telescope pointing
        Parameters:
        mode: str
            mode of the simulation, default is 'tracking', other options are 'allsky','pnt'
        Returns:
        rxang_sep: float
            angular separation between the satellite pointing and observer in the telescope reference frame
        '''
        if mode == 'tracking':
            self.rxang_sep = geometry.true_angular_distance(self.pnt_az, self.pnt_el, self.topo_pos_az*u.deg, self.topo_pos_el *u.deg)
        elif mode == 'allsky':
            self.rxang_sep = geometry.true_angular_distance(self.grid_az*u.deg, self.grid_el*u.deg, self.topo_pos_az*u.deg, self.topo_pos_el *u.deg)
        elif mode == 'pnt':
            self.rxang_sep = geometry.true_angular_distance(pnt_az*u.deg, pnt_el*u.deg, self.topo_pos_az*u.deg, self.topo_pos_el *u.deg)
        return self.rxang_sep

    def create_baselines(self):
        '''
        Description: Create the baseline pairs array for fringe simulation
        '''
        from itertools import combinations
        antennas = self.receiver.location
        self.baselines = combinations(range(len(antennas)), 2)
        self.bearings, self.bearing_D = baseline_pairs(antennas)
        # self.bearings = self.bearings.reshape(self.location.shape)
        self.bearing_D = self.bearing_D.reshape(self.location.shape)
        # self.delays = mod_tau(self.baselines*u.m)

    def baselines_nearfield_delays(self,mode = 'tracking'):
        '''
        Description: Calculate the near field delay for the baselines using the satellite positions
        Args:
        mode: str
            mode of the simulation, default is 'tracking', other options are 'allsky'
        '''
        '''

        returns:
        baseline_delays: float
            delay difference for each baseline at each instance of pointing, returns in whole simulation array format 
        '''


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
        Calculate the fringe response for a given delay and frequency
        based on two element equation integration, assuming equal gain.
        the function takes into the frequency settings and does a integration with channel bins over the channel bandwidth

        Args:
            bwchan (quantity): channel bandwidth 
            fch1 (quantity): channel centre frequency
            chan_bin (int): number of channels in the band
        Returns:
            response (float): fringe response
        """


        delays= self.baseline_delays.flatten()
        self.fringes=bw_fringe(delays,bwchan,fch1,chan_bin=chan_bin).reshape(self.baseline_delays.shape)

        return self.fringes

    def fringe_signal(self,pwr,g_rx,ant1_idx=0,ant2_idx=1):
        '''
        Description: Calculate the power of a specifc baseline using the fringes
        Parameters:
        pwr: float
            power of the signal, linear values only, use 1 for attenuation factor calculation
        g_rx: quantity
            receiver gain in source direction during observation (usually cnv.dBi)
        ant1_idx: int
            index of the first antenna in the baseline
        ant2_idx: int
            index of the second antenna in the baseline
        
        Returns:
        pwr: float
            power of the signal
        '''
        coherent_v_baselines=np.sqrt(pwr*g_rx.to(cnv.dimless)*(self.fringes**2))
        fringe1 = coherent_v_baselines[ant1_idx]
        fringe2 = coherent_v_baselines[ant2_idx]
        pwr = fringe1*fringe2
        self.fringe_pwr = pwr
        return pwr