"""
antenna.py

This module contains a set of antennas that could be used for compatibility calculations.

Author: boris.sorokin <mralin@protonmail.com>
Date: 28-01-2025
Latest amend date: 01-04-2025 - added Rec 1528 rec 1.4 pattern as amended by ESA
"""
import numpy as np
import scipy.special as sp
from astropy import units as u
from pycraf import conversions as cnv
from pycraf.utils import ranged_quantity_input
from scipy.optimize import root_scalar
from scipy.special import j1


def primary_beam(theta, D, wavelength):
    """
    Calculate the primary beam pattern of a single-dish antenna (Airy pattern), using astropy quantities.

    Parameters:
        theta (Quantity): Angle(s) from beam center (must have angular units)
        D (Quantity): Dish diameter (must have length units)
        wavelength (Quantity): Observing wavelength (must have length units)

    Returns:
        np.ndarray: Primary beam (normalized, max = 1)
    """
    # Ensure quantities have correct units
    theta = theta.to(u.rad)
    D = D.to(u.m)
    wavelength = wavelength.to(u.m)

    x = (np.pi * D / wavelength) * np.sin(theta.value)
    beam = np.ones_like(x)
    nonzero = (x != 0)
    beam[nonzero] = (2 * j1(x[nonzero]) / x[nonzero])**2
    return beam

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

def calculate_3dB_angle_1d(antenna_gain_func: callable = s_1528_rec1_2_pattern, 
                           **antenna_pattern_kwargs) -> u.Quantity:
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
# ------------------------------------------------------------------------
#  ITU-R S.1528  |  Recommends 1-4 : circular-aperture Taylor envelope
# ------------------------------------------------------------------------
@ranged_quantity_input(
    offset_theta=(None, None, u.deg),      # θ can be scalar, vector, or grid
    offset_phi  =(None, None, u.deg),      # φ idem
    SLR         =(None, None, cnv.dB),     # side-lobe ratio     [dB]
    Lr          =(1e-30, 1e30, u.m),       # radial semi-axis    [m]
    Lt          =(1e-30, 1e30, u.m),       # transverse semi-axis[m]
    wavelength  =(1e-30, 1e30, u.m),       # λ                   [m]
    Gmax        =(-1e3, 1e3,  cnv.dBi),    # optional absolute peak
    strip_input_units=True,                # hand floats to body
    allow_none=True)                       # Gmax may be None
def s_1528_rec1_4_pattern_amend(
    offset_theta,
    offset_phi = 0 * u.deg,
    *,
    meshgrid: bool = False,
    SLR: float = 20 * cnv.dB,
    Lr : float = 0.007 * u.m,
    Lt : float = 0.007 * u.m,
    wavelength: float = (10.7 * u.GHz).to(
        u.m, equivalencies=u.spectral()),
    Gmax: float | None = None,
    l: int = 4,
    mu_roots: np.ndarray | None = None,
    return_extras: bool = False):
    """
    Very-detailed (toddler-level) explanation
    -----------------------------------------

    *Purpose*  
        Compute the reference antenna pattern stipulated in ITU-R
        Recommendation **S.1528, Recommends 1-4**.  
        The pattern is a *Taylor* illumination of a **circular
        aperture**, giving symmetrical side-lobes.

    *Main formula (verbal)*  
        *Gain* = ★ 20·log10(  2·J1(u) / u · Π[...]  ) ★  
        - where J1 is a Bessel function and Π is a product that
        shapes the side-lobes.

    *Internal function - u*  

           u = π / λ · √[ (Lr · sinθ · cosφ)² + (Lt · sinθ · sinφ)² ]

        · θ  : polar angle away from beam axis  
        · φ  : azimuth around that axis  
        · Lr : “radius” of the illuminated area seen from the satellite  
        · Lt : “transverse” semi-axis of that area  
        · λ  : wavelength

    *Side-lobe controls*  

        • **SLR** (dB) sets how high the first side-lobe is.  
        • **l**   is the number of side-lobes that we keep in the product.  
        • μᵢ      are the first *l* roots of *J1(π μ)=0*  
                   (we look them up with SciPy unless you supply them).  

        From SLR & l we derive two helper constants, **A** and **σ**,
        exactly as the Recommendation prescribes.

    *Inputs accepted*  
        - θ, φ : any mix of scalars, 1-D vectors, or pre-made 2-D grids;  
        - set *meshgrid=True* when you give *independent* θ- and φ-vectors
          and want the full 2-D pattern;  
        - physical sizes Lr / Lt, wavelength λ, side-lobe ratio SLR… all with
          proper units or bare numbers (decorator takes care of units).

    *Outputs*  
        • always an **Astropy Quantity in dBi** (so it plugs straight into
          the rest of Pycraf);  
        • optional *extras* dict with A, σ, μᵢ, and the computed *u*
          if `return_extras=True`.

    Parameters
    ----------
    offset_theta, offset_phi : float or array_like, **degrees**
        Angular coordinates where you want the gain. They are broadcast
        to the same shape, or turned into a mesh-grid if `meshgrid=True`.
    meshgrid : bool, optional
        Force creation of a full θ×φ grid when θ & φ are 1-D vectors of
        different lengths.
    SLR : float, **dB**, optional
        Side-lobe ratio (main-beam peak minus first side-lobe peak).
        20 dB is the ITU default.
    Lr, Lt : float, **metres**
        Semi-axes of the effective radiating area on the satellite.
    wavelength : float, **metres**
        Usually λ = *c / f* at the lowest band edge (10.7 GHz → ~0.028 m).
    Gmax : float, **dBi**, optional
        Absolute peak gain.  If *None*, the pattern is returned *relative*
        (0 dB at boresight).
    l : int, optional
        Number of side-lobes considered by the Taylor synthesis (≥ 2).
    mu_roots : array_like, optional
        The first *l* roots μᵢ of J₁(π μ)=0.  Computed automatically when
        omitted.
    return_extras : bool, optional
        If *True*, also return a dict with A, σ, μᵢ and the u-array.

    Returns
    -------
    gain : `astropy.units.Quantity`  [dBi]
        Pattern value(s) at the requested (θ, φ).
    extras : dict
        Only when `return_extras` is *True*.
    """
    # ---------------------------------------------------------------
    # 1. Prepare θ and φ arrays so they have *identical* shapes
    # ---------------------------------------------------------------
    th = np.asanyarray(offset_theta, dtype=float)   # keep original shape
    ph = np.asanyarray(offset_phi,  dtype=float)

    # Case A: two different 1-D vectors → make a 2-D mesh-grid
    if ((th.ndim == 1 and ph.ndim == 1 and th.size != ph.size) or meshgrid):
        th, ph = np.meshgrid(th, ph, indexing='ij')  # θ along rows, φ along cols
    else:
        # Case B: scalars, equal-length 1-D vectors, or same-shape grids
        th, ph = np.broadcast_arrays(th, ph)

    # ---------------------------------------------------------------
    # 2. Compute *u* (see equation in the Recommendation)
    # ---------------------------------------------------------------
    sin_th = np.sin(np.deg2rad(th))
    cos_ph = np.cos(np.deg2rad(ph))
    sin_ph = np.sin(np.deg2rad(ph))

    u_val = (np.pi / wavelength) * np.sqrt(
        (Lr * sin_th * cos_ph)**2 + (Lt * sin_th * sin_ph)**2
    )

    # ---------------------------------------------------------------
    # 3. Taylor-synthesis constants  (A, σ, μᵢ)
    # ---------------------------------------------------------------
    A = np.arccosh(10.0**(SLR / 20.0)) / np.pi   # ITU equation

    if mu_roots is None:
        # First l roots of J1(π μ)=0  → SciPy helper gives J1 zeros,
        # then we divide by π.
        mu_roots = sp.jn_zeros(1, l) / np.pi
    mu_roots = np.ascontiguousarray(mu_roots, dtype=float)

    sigma = mu_roots[l-1] / np.sqrt(A*A + (l - 0.5)**2)
    pi2sig2 = (np.pi**2) * (sigma**2)   # pre-multiply for speed

    # ---------------------------------------------------------------
    # 4. 2·J1(u)/u  (with limit 1 when u→0)
    # ---------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = 2.0 * sp.j1(u_val) / u_val
    term1 = np.where(u_val == 0.0, 1.0, term1)

    # ---------------------------------------------------------------
    # 5. Product Π_{i=1}^{l-1}[…]
    #    We keep a tiny, optional Numba JIT inside the function so the
    #    outer namespace stays clean.  If Numba is not installed, we
    #    fall back to vectorised NumPy (still fast for O(10⁴) points).
    # ---------------------------------------------------------------
    try:
        from numba import njit                      # extremely cheap import
        have_numba = True
    except ImportError:                             # pragma: no cover
        have_numba = False

    if have_numba:
        @njit(cache=True, fastmath=True, nogil=True)
        def _prod_numba(u_flat, pi2sig2, A, l, mu):
            """Fast loop in C/LLVM: computes the Taylor product."""
            out = np.ones_like(u_flat)
            for k in range(u_flat.size):
                u  = u_flat[k]
                uu = u * u
                prod = 1.0
                for i in range(1, l):
                    num = 1.0 - uu / (pi2sig2 * (A*A + (i - 0.5)**2))
                    den = 1.0 - (u / (np.pi * mu[i-1]))**2
                    prod *= num / den
                out[k] = prod
            return out

        prod = _prod_numba(u_val.ravel().astype(np.float64),
                           pi2sig2, A, l, mu_roots).reshape(u_val.shape)
    else:
        # -------- vectorised NumPy fallback (still ∼ O(N · l)) --------
        i = np.arange(1, l)[:, None]     # shape (l-1, 1) for broadcasting
        uu = u_val.ravel()[None, :]**2   # shape (1, N)
        num = 1.0 - uu / (pi2sig2 * (A*A + (i - 0.5)**2))
        den = 1.0 - uu / ((np.pi * mu_roots[i-1])**2)
        prod = np.prod(num / den, axis=0).reshape(u_val.shape)

    # ---------------------------------------------------------------
    # 6. Combine terms → gain (linear → dB)
    # ---------------------------------------------------------------
    F     = np.abs(term1 * prod)
    gain  = 20.0 * np.log10(F)           # still a plain ndarray

    # Apply absolute peak if requested
    if Gmax is not None:
        gain += float(Gmax)

    gain_out = gain * cnv.dBi            # attach dBi unit

    # ---------------------------------------------------------------
    # 7. Optionally bundle internal variables for diagnostics
    # ---------------------------------------------------------------
    if return_extras:
        extras = dict(A     = A,
                      sigma = sigma,
                      mu_roots = mu_roots[:l].copy(),
                      u     = u_val.copy())
        return gain_out, extras

    return gain_out
# ------------- end of s_1528_rec1_4_pattern_amend -----------------------