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

# ---------------------------------------------------------------------------
# Optional acceleration flag: we check for Numba **once** at module import to
# avoid repeated import attempts.  Functions can then branch on `_HAVE_NUMBA`
# (cheap boolean) without altering their public API.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - optional dependency
    njit = None
    _HAVE_NUMBA = False


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
                          return_extras: bool = False,
                          use_numba: bool | None = None):
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
        use_numba : bool or None, optional
            When *True*, use a small cached ``numba.njit`` helper for the
            piecewise evaluation provided Numba is installed.  When *False* the
            code uses the NumPy fallback even if Numba is available.  The
            default (*None*) keeps the previous behaviour: auto-enable only when
            Numba was importable at module load (``_HAVE_NUMBA``).
        
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

    Performance notes
    -----------------
    The function is fully vectorised with NumPy for typical scientific use.
    For very large angle grids (O(10⁵) elements) you can set ``use_numba=True``
    to evaluate the piecewise regions via a small ``numba.njit`` helper.  The
    public API and numerical behaviour remain unchanged; if Numba is
    unavailable the routine silently falls back to the NumPy path.  Leaving the
    flag as ``None`` keeps the default auto-enable behaviour when Numba was
    importable at module load.
    """
    if Gm is None:
        efficiency = .60
        Gm = 10*np.log10(efficiency * ((np.pi*D/wavelength)**2))   
        
    
    # Ensure offset_angles is a numpy array for elementwise operations. Also
    # applying abs function as offset angles are assumed to be only positive
    offset_angles = np.abs(np.ascontiguousarray(np.asarray(offset_angles, dtype=float)))
    
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

    # Pre-compute scalar thresholds so both NumPy and (optional) Numba paths
    # work from the same constants.  Keeping them as plain floats means we do
    # not need to worry about units inside the fast path.
    a_psi_b = a * psi_b
    half_b_psi_b = 0.5 * b * psi_b
    b_psi_b = b * psi_b
    ninety = 90.0
    one_eighty = 180.0

    # Initialise the gains array with NaN values; the fast path overwrites
    # every element, while the NumPy path fills regions via boolean masks.
    gains = np.full_like(offset_angles, np.nan, dtype=np.float64)

    # Optional acceleration: fast, compiled scalar loop with identical
    # branching logic.  The availability check happens once at module import
    # (``_HAVE_NUMBA``), so the runtime branch is a cheap boolean test.
    use_numba = _HAVE_NUMBA if use_numba is None else bool(use_numba)

    if use_numba and _HAVE_NUMBA:
        @njit(cache=True, fastmath=True, nogil=True)
        def _piecewise_gain(angles, a_psi_b, half_b_psi_b, b_psi_b,
                            Y, ninety, one_eighty, psi_b, alpha,
                            Gm, LN, LF, LB, X, z):
            out = np.empty_like(angles)
            for idx in range(angles.size):
                ang = angles[idx]
                if ang <= a_psi_b:
                    out[idx] = Gm - 3.0 * (ang / psi_b) ** alpha
                elif ang <= half_b_psi_b:
                    out[idx] = Gm + LN + 20.0 * np.log10(z)
                elif ang <= b_psi_b:
                    out[idx] = Gm + LN
                elif ang <= Y:
                    out[idx] = X - 25.0 * np.log10(ang)
                elif ang <= ninety:
                    out[idx] = LF
                elif ang <= one_eighty:
                    out[idx] = LB
                else:
                    out[idx] = np.nan
            return out

        # Flatten for Numba (1-D contiguous), then reshape back.
        gains_flat = _piecewise_gain(
            offset_angles.ravel().astype(np.float64), a_psi_b,
            half_b_psi_b, b_psi_b, Y, ninety, one_eighty, psi_b,
            alpha, Gm, LN, LF, LB, X, z,
        )
        gains = gains_flat.reshape(offset_angles.shape)

    if np.isnan(gains).any():
        # Create masks for each condition
        mask1 = offset_angles <= a_psi_b
        mask2 = (offset_angles > a_psi_b) & (offset_angles <= half_b_psi_b)
        mask3 = (offset_angles > half_b_psi_b) & (offset_angles <= b_psi_b)
        mask4 = (offset_angles > b_psi_b) & (offset_angles <= Y)
        mask5 = (offset_angles > Y) & (offset_angles <= ninety)
        mask6 = (offset_angles > ninety) & (offset_angles <= one_eighty)

        # Apply the piecewise equations using masks
        gains[mask1] = (Gm - 3 * (offset_angles[mask1] / psi_b)**alpha)
        gains[mask2] = (Gm + LN + 20 * np.log10(z))
        gains[mask3] = (Gm + LN)
        gains[mask4] = (X - 25 * np.log10(offset_angles[mask4]))
        gains[mask5] = LF
        gains[mask6] = LB

    return gains, Gm, psi_b

def calculate_beamwidth_1d(antenna_gain_func: callable = s_1528_rec1_2_pattern,
                           level_drop: float | u.Quantity = 3.0 * cnv.dB, 
                           **antenna_pattern_kwargs) -> u.Quantity:
    """
    Calculates the 1D beamwidth for a given antenna pattern function with
    radial symmetry and maximum gain in the main axis direction, at an 
    arbitrary level drop (default: 3 dB) from the peak.

    Parameters
    ----------
    antenna_gain_func : callable, optional
        Function that returns the antenna gain given an angle.
        It must accept `theta` (astropy.units.Quantity in degrees) as the
        first positional argument and may accept additional keyword 
        arguments passed via `antenna_pattern_kwargs`.
        The function may return either:
        - a single astropy.units.Quantity (gain in dBi), or
        - a tuple/list whose first element is the gain (dBi).
    level_drop : float or astropy.units.Quantity, optional
        Level drop (in dB) relative to the maximum gain at boresight.
        For a -3 dB beamwidth, use 3 or 3 * cnv.dB (default).
        For a -15 dB beamwidth, use 15 or 15 * cnv.dB.
        The sign is ignored (i.e. -3 and 3 give the same result).
    antenna_pattern_kwargs : dict
        Additional keyword arguments forwarded to `antenna_gain_func`.

    Returns
    -------
    theta_bw : astropy.units.Quantity
        The full beamwidth (2 * theta_cross) corresponding to the given
        level drop, in degrees.

    Notes
    -----
    - Assumes the pattern is symmetric around boresight and the main lobe
      maximum is at theta = 0 deg.
    - Uses a 1D root-finding (Brent's method) between 0 and 90 deg to find
      the first crossing with the target level on one side of the main lobe,
      then doubles that angle to obtain the full beamwidth.
    """
    # --- Normalize level_drop to a positive dB quantity ---
    if not isinstance(level_drop, u.Quantity):
        level_drop = np.abs(level_drop) * cnv.dB
    level_drop = np.abs(level_drop.value)*level_drop.unit

    # --- Get maximum gain at boresight (theta = 0 deg) ---
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
        raise ValueError(f"Could not get Gmax from antenna pattern: {e}") from e

    # Target gain = Gmax - level_drop (e.g. Gmax - 3 dB, Gmax - 15 dB, ...)
    G_target_dBi = Gmax_dBi - level_drop

    def gain_diff(angle_deg_scalar: float) -> float:
        """
        Difference between gain(theta) and target level at a scalar angle (deg).
        Used by the root-finder; returns a plain float.
        """
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

            # Protect the root-finder from NaNs / infs
            return diff if np.isfinite(diff) else 1e9

        except Exception:
            # If pattern evaluation fails, return a large value so root-finder
            # avoids this region.
            return 1e9

    # --- Find first crossing with target level and double it ---
    try:
        # You can make this bracket configurable if нужно
        bracket = (1e-9, 90.0)

        sol = root_scalar(gain_diff, bracket=bracket, method='brentq')

        if not sol.converged:
            drop_val = level_drop.to(cnv.dB).value
            raise RuntimeError(
                f"Root finding for -{drop_val:.3g} dB beamwidth failed: {sol.flag}"
            )

        theta_bw = 2 * sol.root * u.deg
        return theta_bw

    except Exception as e:
        drop_val = level_drop.to(cnv.dB).value
        raise RuntimeError(
            f"Finding beamwidth for -{drop_val:.3g} dB level failed: {e}"
        ) from e
    

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
    return_extras: bool = False,
    use_numba: bool | None = None):
    """
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
    use_numba : bool or None, optional
        When *True*, use a tiny cached ``numba.njit`` helper for the Taylor
        product provided Numba is installed.  When *False* the code uses the
        NumPy fallback even if Numba is available.  The default (*None*) keeps
        the previous behaviour: auto-enable only when Numba was importable at
        module load (``_HAVE_NUMBA``).

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
    #    We keep a tiny, optional Numba JIT, enabled only when Numba was
    #    imported at module load (``_HAVE_NUMBA``) and the caller does not
    #    explicitly disable it via ``use_numba=False``.
    # ---------------------------------------------------------------
    use_numba = _HAVE_NUMBA if use_numba is None else bool(use_numba)

    if use_numba and _HAVE_NUMBA:
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