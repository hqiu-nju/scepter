"""
tleforger.py

This module generates artificial TLEs (Two-Line Element sets) for satellites.

Author: boris.sorokin <mralin@protonmail.com>
Date: 16-01-2025
"""
import numpy as np
from astropy import units as u
from datetime import datetime
from astropy.time import Time
from astropy.constants import GM_earth, R_earth
from cysgp4 import PyTle
from pycraf.utils import ranged_quantity_input

class NamedPyTle(PyTle):
    def __init__(self, name, line1, line2):
        super().__init__(name, line1, line2)
        self.name = name

@ranged_quantity_input(altitude = (0, 384400000, u.m),
                       inclination_deg = (-360, 360, u.deg),
                       raan_deg = (-360, 360, u.deg),
                       argp_deg = (-360, 360, u.deg),
                       anomaly_deg = (-360, 360, u.deg),
                       strip_input_units=True,
                       allow_none=True)
def forge_tle_single(
    sat_name: str = 'Satellite',
    altitude: float = 400000.0 * u.m,
    eccentricity: float = 0.0,
    inclination_deg: float = 90.0 * u.deg,
    raan_deg: float = 0.0 * u.deg,
    argp_deg: float = 0.0 * u.deg,
    anomaly_deg: float = 0.0 * u.deg,
    start_time: Time = Time(datetime(2025, 1, 1, 0, 0, 0)),
    mm_dot: float = 0.0,
    mm_ddot: float = 0.0,
    bstar: float = 0.00
) -> NamedPyTle:
    """
    Forge a TLE for a single satellite from a set of orbital parameters and a reference epoch.

    This function returns a three-line TLE string:
      - Line 0: Satellite name
      - Line 1: TLE metadata (NORAD ID, epoch, ṅ, n̈, BSTAR, etc.)
      - Line 2: Orbital parameters (inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion)

    Parameters
    ----------
        sat_name : str, optional
            The satellite name, which appears on line 0 of the TLE. Default is 'Satellite'.
        altitude : float astropy Quantity (Length), optional
            The orbital altitude above Earth's surface, in meters. Used to compute the mean motion for a (near-)circular orbit.
            Default is 400000.0 m (≈400 km above Earth's surface).
        eccentricity : float, optional
            The orbital eccentricity. Default is 0.0 (circular orbit).
        inclination_deg : float astropy Quantity (Angular distance), optional
            Inclination in degrees. Default is 90.0.
        raan_deg : float astropy Quantity (Angular distance), optional
            Right Ascension of the Ascending Node, in degrees. Default is 0.0.
        argp_deg : float astropy Quantity (Angular distance), optional
            Argument of perigee, in degrees. Default is 0.0.
        anomaly_deg : float astropy Quantity (Angular distance), optional
            Mean anomaly at epoch, in degrees. Default is 0.0.
        start_time : astropy.time.Time, optional
            The reference epoch for the TLE. Default is 2025-01-01 00:00:00 UTC.
        mm_dot : float, optional
            First derivative of mean motion (revs/day²). Default is 0.0.
        mm_ddot : float, optional
            Second derivative of mean motion (revs/day³). This value is stored using
            an 8-character TLE exponential notation. Default is 0.0.
        bstar : float, optional
            BSTAR drag term (earth-radii⁻¹). Also stored using the 8-character TLE
            exponential notation. Default is 0.00.

    Returns
    -------
        NamedPyTle
            A modified PyTle object to also include specific name string.

    Notes
    -----
        1) Internally, mean motion is computed assuming a (near-)circular orbit of
        semi-major axis = R_earth + altitude. Gravity constants (GM_earth)
        and Earth radius (R_earth) are taken from astropy constants.

        2) The second derivative of mean motion (n̈/2 in SGP4) and the BSTAR
        drag term are formatted via a helper function `format_tle_exp()`.
        This uses an 8-character NORAD TLE style with:
            [ sign or space ][ 5-digit mantissa ][ sign exponent ][ 1-digit exponent ]

        Example outputs:
            "+12345-5" or " 12345-5" =  0.12345 x 10⁻⁵ = 1.2345e-6
            "-24500-5"              = -0.24500 x 10⁻⁵ = -2.45e-6

        3) The function enforces correct line lengths (68 characters before checksum).
        A `ValueError` is raised if the line length is unexpected or if
        `mm_ddot` / `bstar` exponent magnitudes exceed ±9.

        4) Checksums for lines 1 and 2 are computed and appended at the end
        (column 69).
    """
    def compute_tle_checksum(tle_line: str) -> int:
        """
        Compute TLE checksum for a single line.
        
        Sums all digits (ignoring '.'), and for each '-', adds +1, then takes mod 10.
        
        Parameters:
        tle_line (str): The TLE line for which to compute the checksum.
        
        Returns:
        int: The computed checksum.
        """
        return sum(int(ch) if ch.isdigit() else 1 if ch == '-' else 0 for ch in tle_line) % 10
    
    def format_tle_exp(value: float) -> str:
        """
        Convert a float to the 7-character TLE 'exponential' format.

        Returns an 8-character string:
            [sign or space][5-digit mantissa][sign exponent][1-digit exponent]

        Columns breakdown:
            [0]:   sign of mantissa ('+' or '-')
            [1..5]: five digits of mantissa (interpreted as 0.xxxxx)
            [6]:    sign of exponent ('+' or '-')
            [7]:    single exponent digit (0..9)

        Examples:
            0.00006796  -> +67960-4  (which TLE reads as 0.67960e-4 = 6.7960e-5)
            1.2345e-6   -> +12345-5
            0.0         -> +00000-0
            -2.45e-5    -> -24500-5
        """
        # Step 1. Determine sign
        sign_mantissa = '-' if value < 0 else ' '

        # Step 2. Working with absolute values fitting the TLE 'exponential' format
        vabs =np.abs(value)
        if vabs < 1.0e-12:
            return f"{sign_mantissa}00000-0"
        
        # Step 3. Find exponent so that 0.1 <= mant < 1
        e = (np.floor(np.log10(vabs)) + 1)
        mant = vabs / np.power(10,e)
        if mant >= 1.0:
            # edge case if x is exactly a power of 10
            mant /= 10.0
            e += 1
        # Step 4. Convert mantissa to 5-digit integer, rounding
        mant_int = int(np.round(mant * 1e5))
        if mant_int == 100000:
            mant_int = 10000
            e += 1
        
        # Step 5. Handle exponent out of range
        if e < -9 or e > 9:
            raise ValueError("Unexpected input parameter. Please check second derivative of mean motion and B*, the drag term, or radiation pressure coefficient")
        
        # Step 6. Build the sign & digit for the exponent
        sign_exp = '+' if e >= 0 else '-'
        exp_digit = abs(e)

        # Step 7. Construct final 8-character string
        return f"{sign_mantissa}{mant_int:05d}{sign_exp}{exp_digit:1.0f}"
    
    # 1. Basic TLE identifiers
    sat_number = 0           # Fake NORAD ID
    classification = "U"
    classification = "U"
    int_desg = "25001A"      # Fake International Designator
    
    # 2. Epoch calculation
    classification = "U"    
    int_desg = "25001A"      # Fake International Designator
    
    # 2. Epoch calculation
    year = start_time.datetime.year
    year_short = year % 100
    int_desg = str(year_short)+"001A"      # Fake International Designator
    
    # 2. Epoch calculation
    start_of_year = Time(start_time.datetime.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0))
    day_of_year = (start_time - start_of_year).to_value('day') + 1  # Day count starts at 1
    epoch_str = f"{year_short:02d}{day_of_year:012.8f}"  # Leading space for epoch
    
    # 3. Mean motion calculation (circular orbit assumption)
    a_m = R_earth.value + altitude
    mean_motion_rad_s = np.sqrt(GM_earth.value / a_m**3)
    mean_motion_rev_day = mean_motion_rad_s * (86400.0 / (2.0 * np.pi))
    
    # 4. Zero placeholders
    mm_dot_string = f"{mm_dot:+.8f}"[:1]+f"{mm_dot:+.8f}"[2:]
    ephemeris_type = 0
    element_set_number = 1
    rev_number = 1  # Dummy revolution number
    
    # Build line 1 (no checksum yet)
    line1 = (f"1 {sat_number:05d}{classification} {int_desg:8} {epoch_str:14} {mm_dot_string} {format_tle_exp(mm_ddot)} {format_tle_exp(bstar)} {ephemeris_type} {element_set_number:4d}")
    
    # Ensure line1 is 68 characters before adding checksum
    if len(line1) != 68:
        raise ValueError("Line 1 is not 68 characters long before adding checksum.")
    
    # Build line 2 (no checksum yet)
    ecc_str = f"{eccentricity:.7f}"[2:]  # Remove '0.' to get 7 digits
    line2 = (
        "2 {:05d} {:8.4f} {:8.4f} {:7} {:8.4f} {:8.4f} {:11.8f}{:05d}"
        .format(
            sat_number, inclination_deg, raan_deg, ecc_str,
            argp_deg, anomaly_deg, mean_motion_rev_day, rev_number
        )
    )
    
    # Ensure line2 is 68 characters before adding checksum
    if len(line2) != 68:
        raise ValueError("Line 2 is not 68 characters long before adding checksum.")
    
    # 5. Compute and append checksums
    line1 += str(compute_tle_checksum(line1))
    line2 += str(compute_tle_checksum(line2))
    
    # 6. Construct the full TLE string
    tle = NamedPyTle(sat_name, line1, line2)  
    return tle


@ranged_quantity_input(RAAN_min = (0, 360, u.deg),
                       RAAN_max = (0, 360, u.deg),
                       altitude = (0, 384400000, u.m),
                       inclination_deg = (-360, 360, u.deg),
                       argp_deg = (-360, 360, u.deg),
                       strip_input_units=False,
                       allow_none=True)
def forge_tle_belt(
    belt = None,
    belt_name: str = "SystemC_Belt_1",
    num_sats_per_plane: int = 40,
    plane_count: int = 18,
    RAAN_min: float = 0 * u.deg,
    RAAN_max: float = 180 * u.deg,
    altitude: float = 1200000.0 * u.m,
    eccentricity: float = 0,
    inclination_deg: float = 87.9 * u.deg,
    argp_deg: float = 0 * u.deg,
    start_time: Time = Time(datetime(2025, 1, 1, 0, 0, 0)),
    mm_dot: float = 0.0,
    mm_ddot: float = 0.0,
    bstar: float = 0.00,
    adjacent_plane_offset: bool = False
) -> np.ndarray:
    """
    Generate a list of TLEs for a constellation (or 'belt') of satellites 
    arranged in multiple orbital planes.

    This function returns a list of three-line TLE strings, each describing 
    one satellite's orbit. By default, it constructs a new belt if the 
    'belt' parameter is None, otherwise it can be extended or adapted 
    to handle an existing belt object.

    Parameters
    ----------
        belt : object or None, optional
            If provided, can be used to modify or extend an existing belt 
            configuration (not implemented in this sample). If None, a new set 
            of TLEs is created using the parameters below. Default is None.
        belt_name : str, optional
            An identifier for the belt. This name is used to build unique 
            satellite names of the form:
            <belt_name>_Plane_<plane_idx>_Satellite_<sat_idx>.
            Default is "SystemC_Belt_1".
        num_sats_per_plane : int, optional
            Number of satellites placed in each orbital plane. Default is 40.
        plane_count : int, optional
            Number of planes in the belt. Default is 18.
        altitude : float astropy Quantity (Length), optional
            Orbital altitude above the Earth's surface in meters. Default is 
            1,200,000 m (approx. 1,200 km).
        RAAN_min : float astropy Quantity (Angular distance), optional
            Minimum Right Ascension of the Ascending Node, in degrees. Default is 0.0.
        RAAN_max : float astropy Quantity (Angular distance), optional
            Maximum Right Ascension of the Ascending Node, in degrees. Default is 180.0.
        eccentricity : float, optional
            Orbital eccentricity for all satellites. Default is 0.0 (circular orbit).
        inclination_deg : float astropy Quantity (Angular distance), optional
            Inclination of all planes in degrees. Default is 87.9.
        argp_deg : float astropy Quantity (Angular distance), optional
            Argument of perigee in degrees for all satellites. Default is 0.0.
        start_time : astropy.time.Time, optional
            Reference epoch (time) for all TLEs. Default is 2025-01-01 00:00:00 UTC.
        mm_dot : float, optional
            First derivative of mean motion (revs/day^2). Default is 0.0.
        mm_ddot : float, optional
            Second derivative of mean motion (revs/day^3), stored in TLE using 
            8-character exponential notation. Default is 0.0.
        bstar : float, optional
            BSTAR drag term (earth-radii^-1), also stored in TLE using 8-character 
            exponential notation. Default is 0.00.
        adjacent_plane_offset : bool, optional
            Some constellations use a checkerboard pattern during deployment to have better coverage. 
            Default is False

    Returns
    -------
        List[NamedPyTle]
            A list of NamedPyTle objects, each describing one satellite.

    Notes
    -----
        1) Within each plane, satellites are spaced in mean anomaly by 
        360.0 / num_sats_per_plane degrees.
        2) The function internally calls `forge_tle_single` for each satellite, 
        which computes TLE line data (NORAD ID, epoch, etc.) and returns 
        a NamedPyTle object as defined above.
        3) If `belt` is not None, this function is currently a stub (`pass`) and 
        does nothing. It's a placeholder for future implementation of belt object.
    """

    if belt is None:
        step_deg = 360.0 / num_sats_per_plane
        raan_deg_step = (RAAN_max - RAAN_min) /plane_count    
        return np.array([
            forge_tle_single(
                sat_name=f"{belt_name}_Plane_{plane_idx+1}_Satellite_{satellite_idx+1}",
                altitude=altitude,
                eccentricity=eccentricity,
                inclination_deg=inclination_deg,
                raan_deg=plane_idx * raan_deg_step,
                argp_deg=argp_deg,
                anomaly_deg=(satellite_idx * step_deg + ((step_deg / 2) if (adjacent_plane_offset and plane_idx % 2 == 1) else 0)) * u.deg,
                start_time=start_time,
                mm_dot=mm_dot,
                mm_ddot=mm_ddot,
                bstar=bstar
            )
            for plane_idx in range(plane_count)
            for satellite_idx in range(num_sats_per_plane)
        ], dtype=object)
    else:
        # Placeholder for future implementation of belt class
        pass