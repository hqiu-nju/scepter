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
    
    Performance notes:
    - Pre-allocates bearing vectors array to minimize memory operations
    - Vectorized baseline length calculation using np.linalg.norm
    
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
    # Pre-allocate bearing vectors array for better performance
    bearings = np.empty((n_antennas, 3), dtype=np.float64)
    
    # Convert reference antenna coordinates
    ref = antennas[0]
    x1, y1, z1 = pycraf.geospatial.wgs84_to_itrf2008(
        ref.loc.lon*u.deg, ref.loc.lat*u.deg, ref.loc.alt*u.m
    )
    ref_coords = np.array([x1.value, y1.value, z1.value])
    
    # Convert all antenna positions and compute bearing vectors
    for i, ant in enumerate(antennas):
        x2, y2, z2 = pycraf.geospatial.wgs84_to_itrf2008(
            ant.loc.lon*u.deg, ant.loc.lat*u.deg, ant.loc.alt*u.m
        )
        bearings[i] = np.array([x2.value, y2.value, z2.value]) - ref_coords
    
    # Vectorized calculation of baseline lengths from bearing vectors
    baselines = np.linalg.norm(bearings, axis=1)

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
    
    Computes the fringe amplitude between two elements integrated
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
    
    Performance optimization:
    - Uses vectorized numpy operations for efficiency
    - Broadcasting is used to minimize memory allocation
    
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
    # Convert to kHz for numerical stability
    fch1_khz = fch1.to(u.kHz).value
    bwchan_khz = bwchan.to(u.kHz).value
    
    # Create frequency array
    freq_array = np.linspace(fch1_khz - bwchan_khz*0.5, 
                             fch1_khz + bwchan_khz*0.5, 
                             chan_bin) * u.kHz
    
    # Reshape for broadcasting: delays as column, frequencies as row
    delays_col = delays[:, np.newaxis]
    freq_row = freq_array[np.newaxis, :]
    
    # Calculate fringe response at all frequencies (vectorized)
    fringes = fringe_response(delays_col, freq_row)
    
    # Average over frequency bins
    return np.mean(fringes, axis=1)

