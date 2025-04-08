"""
antenna.py

This module contains a set of antennas that could be used for compatibility calculations.

Author: boris.sorokin <mralin@protonmail.com>
Date: 28-01-2025
Revised 01-04-2025 - added 3dB beamwidth calculation for 1d radial symmetry antenna patterns
"""
import numpy as np
from astropy import units as u
from pycraf import conversions as cnv
from pycraf.utils import ranged_quantity_input
from scipy.optimize import root_scalar


@ranged_quantity_input(offset_angles = (None, None, u.deg), 
                       Gm = (-500, 500, cnv.dBi), 
                       LN = (-30, -15, cnv.dB),
                       LF = (-500, 500, cnv.dBi),
                       LB = (-500, 500, cnv.dBi),
                       D = (0, 1000000, u.m),
                       wavelength = (0, 1000000, u.m),
                       output_unit = (cnv.dBi, cnv.dBi, u.deg),
                       strip_input_units=True,
                       allow_none=True)

def s_1528_rec1_2_pattern(offset_angles, 
                          axis: str = 'major', 
                          Gm: float = None, 
                          LN: float =-15*cnv.dB, 
                          LF: float = 1.8*cnv.dBi, 
                          LB: float = None, 
                          D: float = 1.0*u.m, 
                          wavelength: float = (10.7*u.GHz).to(u.m, equivalencies=u.spectral()), 
                          z: float = 1.0,
                          return_extras: bool = False):
    """
    Calculate the antenna radiation pattern for ITU-R Recommendation S.1528-0 recommends 1.2.

    This function computes the antenna gains based on recommends 1.2 section of ITU-R Recommendation S.1528-0.

    Parameters
    ----------
        offset_angles: array-like quantity: astropy angle
            Off-axis angles in degrees. Shape of this array will be used for gains output.
        axis:  string
            determines which axis to be used. Default is 'major'.
        Gm: float pycraf quantity: dBi, optional
            Maximum gain in the main lobe (dBi). Default is NaN, so Gm will be calculated from other parameters.
        LN: float pycraf quantity: dB, optional
            Near-in-side-lobe level (dB) relative to the peak gain required by the system design.
        LF: float pycraf quantity: dBi, optional
            Far-out side-lobe level (dBi), typically ~0 dBi for ideal patterns. Default is 1.8 [dBi]
        LB: float pycraf quantity: dBi, optional
            Back-lobe level (dBi). Default is NaN, so LB will be calculated from other parameters.
        D: float astropy quantity (Length), optional
            Diameter of the antenna (meters). Default is 1 [m].
        wavelength: float astropy quantity (Length), optional
            Wavelength of the lowest band edge of interest (meters). Default is ~0.028 [m] (10.7 GHz).
        z:  float, optional
            Ratio of major axis to minor axis for elliptical beams. Default is 1.0 (circular beam).
        return_extras: bool, optional
            Whether Gm and psi_b should be returned. Default is false.
        
        Default values allow to recreate Figure 1 from ITU-R Recommendation S.1528-0
    Returns
    -------
        gains: array-like pycraf quantity: dBi
            Gain values (dBi) matching the shape of offset_angles.

        If return_extras is true, there are two more outputs:
        Gm: float pycraf quantity: dBi, optional
            Maximum gain in the main lobe (dBi). Useful for getting pattern relevant to Gmax when it was calculated internally.
        ψ3dB: 
            one-half beamwidth for -3dB level

    Notes
    -----
    The piecewise gain equations are based on ITU-R Recommendation S.1528-0  recommends 1.2. 
    For further details, see the ITU recommendation: https://www.itu.int/rec/R-REC-S.1528-0-200106-I/en.
    """
    if Gm is None:
        efficiency = .60
        Gm = 10*np.log10(efficiency * ((np.pi*D/wavelength)**2))   
        
    
    # Ensure offset_angles is a numpy array for elementwise operations. Also applying abs function as offset angles are assumed to be only positive
    offset_angles = np.abs(np.asarray(offset_angles))
    
    # Compute ψb (one-half the 3 dB beamwidth in degrees)
    psi_b_major = np.sqrt(1200) / (D/wavelength)  # Major axis beamwidth
    psi_b_minor = psi_b_major / z         # Minor axis beamwidth

    # Use the appropriate ψb based on the beam type (major or minor axis)
    if axis == 'major':
        psi_b = psi_b_major
    elif axis == 'minor':
        psi_b = psi_b_minor
    else:
        raise ValueError("Axis must be 'major' or 'minor'")   

    # Constants for LN levels (from Table 1 in the recommendation)
    LN_levels = {
        -15: {"a": 2.58 * np.sqrt(1 - 1.4 * np.log10(z)), "b": 6.32, "alpha": 1.5},
        -20: {"a": 2.58 * np.sqrt(1 - 1.0 * np.log10(z)), "b": 6.32, "alpha": 1.5},
        -25: {"a": 2.58 * np.sqrt(1 - 0.6 * np.log10(z)), "b": 6.32, "alpha": 1.5},
        -30: {"a": 2.58 * np.sqrt(1 - 0.4 * np.log10(z)), "b": 6.32, "alpha": 1.5},
    }

    # Get the constants for the given LN value
    if LN not in LN_levels:
        raise ValueError(f"Unsupported LN value: {LN}. Supported values are -15 dB, -20 dB, -25 dB, -30 dB.")
    a, b, alpha = LN_levels[LN]["a"], LN_levels[LN]["b"], LN_levels[LN]["alpha"]

    # # Compute X and Y thresholds
    X = Gm + LN + 25 * np.log10(b * psi_b)
    Y = b * psi_b * 10**(0.04 * (Gm + LN - LF))

    if LB is None:
        LB = (np.max(15+LN+0.25*Gm+5*np.log10(z),0))

    # Create masks for each condition
    mask1 = offset_angles <= a * psi_b
    mask2 = (offset_angles > a * psi_b) & (offset_angles <= 0.5 * b * psi_b)
    mask3 = (offset_angles > 0.5 * b * psi_b) & (offset_angles <= b * psi_b)
    mask4 = (offset_angles > b * psi_b) & (offset_angles <= Y)
    mask5 = (offset_angles > Y) & (offset_angles <= 90)
    mask6 = (offset_angles > 90) & (offset_angles <= 180)

    
    # Initialize the gains array with NaN values
    gains = np.full_like(offset_angles, np.nan, dtype=np.float64)

    # # Apply the piecewise equations using masks
    gains[mask1] = (Gm - 3 * (offset_angles[mask1] / psi_b)**alpha)
    gains[mask2] = (Gm + LN + 20 * np.log10(z))
    gains[mask3] = (Gm + LN)
    gains[mask4] = (X - 25 * np.log10(offset_angles[mask4]))
    gains[mask5] = LF
    gains[mask6] = LB

    return gains, Gm, psi_b







def calculate_3dB_angle_1d(antenna_gain_func: callable = s_1528_rec1_2_pattern, **antenna_pattern_kwargs) -> u.Quantity:
    """
    Calculates the -3 dB beamwidth angle for a given antenna pattern function with radial symmetry and maximum gain in the main axis direction.
    
    Parameters
    ----------
    antenna_gain_func : callable
        Function that returns the antenna gain given an angle.
    antenna_pattern_kwargs : dict
        Additional keyword arguments for the antenna gain function.
    
    Returns
    -------
    theta_3dB : astropy.units.Quantity
        The -3 dB beamwidth angle in degrees.
    """
    try:
        gain_result = antenna_gain_func(0 * u.deg, **antenna_pattern_kwargs)
        if isinstance(gain_result, (tuple, list)):
            Gmax_dBi = gain_result[0]
        else:
            Gmax_dBi = gain_result
        if not isinstance(Gmax_dBi, u.Quantity):
            Gmax_dBi = Gmax_dBi * cnv.dBi
        Gmax_dBi = Gmax_dBi.to(cnv.dBi)
    except Exception as e:
        raise ValueError(f"Could not get Gmax: {e}") from e

    G_target_dBi = Gmax_dBi - 3.0 * cnv.dB

    def gain_diff(angle_deg_scalar):
        try:
            gain_val = antenna_gain_func(angle_deg_scalar * u.deg, **antenna_pattern_kwargs)
            if isinstance(gain_val, (tuple, list)):
                current_gain_dBi = gain_val[0]
            else:
                current_gain_dBi = gain_val
            if not isinstance(current_gain_dBi, u.Quantity):
                current_gain_dBi = current_gain_dBi * cnv.dBi
            current_gain_dBi = current_gain_dBi.to(cnv.dBi)
            diff = (current_gain_dBi - G_target_dBi).value
            return diff if np.isfinite(diff) else 1e9
        except Exception:
            return 1e9

    try:
        sol = root_scalar(gain_diff, bracket=[1e-9, 90.0], method='brentq')
        if not sol.converged:
            raise RuntimeError(f"-3dB root finding failed: {sol.flag}")
        theta_3dB = sol.root * u.deg
        return theta_3dB
    except Exception as e:
        raise RuntimeError(f"Finding -3dB angle failed: {e}")