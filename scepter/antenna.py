"""
Compatibility and antenna-pattern helpers for SCEPTer workflows.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>

This module remains part of the GPLv3-licensed SCEPTer project. The author
attribution above does not change the project-wide license.
"""
import importlib
from typing import Any, Callable, Literal

import numpy as np
import scipy.special as sp
from astropy import units as u
from pycraf import conversions as cnv
from pycraf.geometry import true_angular_distance
from pycraf.utils import ranged_quantity_input
from scipy.optimize import root_scalar

# ---------------------------------------------------------------------------
# Optional acceleration flag: keep the normal import path free from Numba so
# GUI and test startup do not eagerly load llvmlite.  We only import Numba the
# first time a caller explicitly opts in via ``use_numba=True`` and then cache
# the result for later auto-enable checks.
# ---------------------------------------------------------------------------
_NUMBA_IMPORT_ATTEMPTED = False
_NUMBA_NJIT: Callable[..., Any] | None = None
_HAVE_NUMBA = False


def _load_numba_njit() -> Callable[..., Any] | None:
    """Import ``numba.njit`` lazily and cache the availability result."""
    global _NUMBA_IMPORT_ATTEMPTED, _NUMBA_NJIT, _HAVE_NUMBA

    if not _NUMBA_IMPORT_ATTEMPTED:
        _NUMBA_IMPORT_ATTEMPTED = True
        try:  # pragma: no cover - optional dependency
            module = importlib.import_module("numba")
        except Exception:  # pragma: no cover - optional dependency
            _NUMBA_NJIT = None
            _HAVE_NUMBA = False
        else:
            candidate = getattr(module, "njit", None)
            _NUMBA_NJIT = candidate if callable(candidate) else None
            _HAVE_NUMBA = _NUMBA_NJIT is not None
    return _NUMBA_NJIT


def pattern_wavelength_cm_from_frequency_mhz(frequency_mhz: float) -> float:
    """Convert an RF frequency in MHz to its wavelength in centimetres."""
    wavelength = (float(frequency_mhz) * u.MHz).to_value(u.cm, equivalencies=u.spectral())
    return float(wavelength)


def resolve_pattern_wavelength_cm(
    *,
    frequency_mhz: float | None,
    pattern_wavelength_cm: float | None,
    derive_from_frequency: bool,
) -> float:
    """Resolve the effective antenna-pattern wavelength in centimetres."""
    if derive_from_frequency:
        if frequency_mhz is None:
            raise ValueError("Frequency is required to derive the pattern wavelength.")
        return pattern_wavelength_cm_from_frequency_mhz(frequency_mhz)
    if pattern_wavelength_cm is None:
        raise ValueError("Pattern wavelength is not configured.")
    return float(pattern_wavelength_cm)


def build_satellite_pattern_spec(
    *,
    antenna_model: str,
    frequency_mhz: float | None,
    pattern_wavelength_cm: float | None,
    derive_pattern_wavelength_from_frequency: bool,
    rec12_gm_dbi: float | None = None,
    rec12_ln_db: float | None = None,
    rec12_z: float | None = None,
    m2101_g_emax_dbi: float | None = None,
    m2101_a_m_db: float | None = None,
    m2101_sla_nu_db: float | None = None,
    m2101_phi_3db_deg: float | None = None,
    m2101_theta_3db_deg: float | None = None,
    m2101_d_h: float | None = None,
    m2101_d_v: float | None = None,
    m2101_n_h: int | None = None,
    m2101_n_v: int | None = None,
    rec14_lt_m: float | None = None,
    rec14_lr_m: float | None = None,
    rec14_l: int | None = None,
    rec14_slr_db: float | None = None,
    rec14_far_sidelobe_start_deg: float | None = None,
    rec14_far_sidelobe_level_dbi: float | None = None,
    rec14_gm_dbi: float | None = None,
    use_numba: bool = False,
) -> tuple[Callable[..., Any], u.Quantity, dict[str, Any]]:
    """Build a pure satellite-pattern spec from normalized S.1528 inputs."""
    wavelength_cm = resolve_pattern_wavelength_cm(
        frequency_mhz=frequency_mhz,
        pattern_wavelength_cm=pattern_wavelength_cm,
        derive_from_frequency=derive_pattern_wavelength_from_frequency,
    )
    wavelength = float(wavelength_cm) * u.cm

    if antenna_model in {"s1528_rec1_2", "s672"}:
        # S.672 Annex 1, Section 1.1 (A1-1.1) uses the same piecewise
        # envelope as S.1528 Rec 1.2.  S.672 calls the near-in sidelobe
        # level "Ls" while S.1528 calls it "LN" — same parameter.
        if rec12_gm_dbi is None or rec12_ln_db is None or rec12_z is None:
            raise ValueError("Pattern inputs (Gm, LN, z) are incomplete.")
        return (
            s_1528_rec1_2_pattern,
            wavelength,
            {
                "Gm": float(rec12_gm_dbi) * cnv.dBi,
                "LN": float(rec12_ln_db) * cnv.dB,
                "z": float(rec12_z),
                "use_numba": bool(use_numba),
            },
        )

    if antenna_model == "s1528_rec1_4":
        required = (
            rec14_lt_m,
            rec14_lr_m,
            rec14_l,
            rec14_slr_db,
            rec14_far_sidelobe_start_deg,
            rec14_far_sidelobe_level_dbi,
        )
        if any(value is None for value in required):
            raise ValueError("REC 1.4 pattern inputs are incomplete.")
        return (
            s_1528_rec1_4_pattern_amend,
            wavelength,
            {
                "Lt": float(rec14_lt_m) * u.m,
                "Lr": float(rec14_lr_m) * u.m,
                "l": int(rec14_l),
                "SLR": float(rec14_slr_db) * cnv.dB,
                "far_sidelobe_start": float(rec14_far_sidelobe_start_deg) * u.deg,
                "far_sidelobe_level": float(rec14_far_sidelobe_level_dbi) * cnv.dBi,
                "Gm": None if rec14_gm_dbi is None else float(rec14_gm_dbi) * cnv.dBi,
                "use_numba": bool(use_numba),
            },
        )

    if antenna_model == "m2101":
        from pycraf.antenna import imt2020_composite_pattern
        m2101_kwargs = kwargs.get("m2101", {}) if "kwargs" in dir() else {}
        # For M.2101, return the pycraf function and the array parameters.
        # The GPU path will use its own kernel; this spec is for CPU-side
        # preview plots only.
        return (
            imt2020_composite_pattern,
            wavelength,
            {
                "G_Emax": float(m2101_g_emax_dbi or 2.0) * cnv.dBi,
                "A_m": float(m2101_a_m_db or 30.0) * cnv.dB,
                "SLA_nu": float(m2101_sla_nu_db or 30.0) * cnv.dB,
                "phi_3db": float(m2101_phi_3db_deg or 120.0) * u.deg,
                "theta_3db": float(m2101_theta_3db_deg or 120.0) * u.deg,
                "d_H": float(m2101_d_h or 0.5) * cnv.dimless,
                "d_V": float(m2101_d_v or 0.5) * cnv.dimless,
                "N_H": int(m2101_n_h or 28),
                "N_V": int(m2101_n_v or 28),
            },
        )

    raise ValueError(f"Unsupported antenna model {antenna_model!r}.")


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
                          Gm: float | None = None,
                          LN: float =-15*cnv.dB,
                          LF: float = 0*cnv.dBi,
                          LB: float = None,
                          D: float | None = None,
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
            default (*None*) reuses cached availability from any earlier
            explicit opt-in and does not trigger a Numba import by itself.
        
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
    flag as ``None`` preserves cached availability from any earlier explicit
    opt-in without importing Numba on the default path.
    """

    efficiency = 0.60

    if (Gm is None) and (D is None):
        # 1) None provided -> default D=1 m, compute Gm
        D = 1.0
        Gm = 10.0 * np.log10(efficiency * (np.pi * (D / wavelength))**2)

    elif Gm is None:
        # 2) D provided -> compute Gm
        Gm = 10.0 * np.log10(efficiency * (np.pi * (D / wavelength))**2)

    elif D is None:
        # 3) Gm provided -> compute D (effective diameter)
        D = (wavelength / np.pi) * np.sqrt((10.0**(0.1 * Gm)) / efficiency)

    else:
        # Both provided -> keep them (optionally validate consistency)
        pass
        
    
    # Ensure offset_angles is a numpy array for elementwise operations. Also
    # applying abs function as offset angles are assumed to be only positive
    offset_angles = np.abs(np.ascontiguousarray(np.asarray(offset_angles, dtype=float)))
    
    # Compute ψb (one-half the 3 dB beamwidth in degrees)
    psi_b_minor = np.sqrt(1200) / (D / wavelength)   # Minor axis beamwidth
    psi_b_major = z * psi_b_minor                    # Major axis beamwidth

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
    # branching logic.  The default path stays pure NumPy; Numba is only
    # loaded after an explicit ``use_numba=True`` request.
    use_numba = _HAVE_NUMBA if use_numba is None else bool(use_numba)
    numba_njit = _load_numba_njit() if use_numba else None

    if numba_njit is not None:
        @numba_njit(cache=True, fastmath=True, nogil=True)
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
                           mode: Literal["first_crossing", "sidelobe_safe"] = "first_crossing",
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
    mode : {"first_crossing", "sidelobe_safe"}, optional
        Beamwidth extraction mode:
        - "first_crossing": first angle where gain reaches or drops below the
          target (current behavior).
        - "sidelobe_safe": smallest angle beyond which gain stays at or below
          the target for the rest of the scan range.
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
    - Uses a 1D search on a 0..90 deg grid and bisection refinement.
    - In "sidelobe_safe" mode, the returned angle is based on the last
      threshold violation in the scan range, i.e. after this point sidelobes
      no longer exceed the requested level.
    """
    # --- Normalize level_drop to a positive dB quantity ---
    if not isinstance(level_drop, u.Quantity):
        level_drop = np.abs(level_drop) * cnv.dB
    level_drop = np.abs(level_drop.value)*level_drop.unit

    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    mode = mode.strip().lower()
    if mode not in ("first_crossing", "sidelobe_safe"):
        raise ValueError(
            "Unsupported mode. Use 'first_crossing' or 'sidelobe_safe'."
        )

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
            diff_arr = np.asarray(
                (current_gain_dBi - G_target_dBi).to_value(cnv.dBi),
                dtype=float,
            )
            if diff_arr.size != 1:
                raise ValueError(
                    "A scalar angle input must produce a scalar gain output."
                )
            diff = float(diff_arr.reshape(-1)[0])

            # Protect the root-finder from NaNs / infs
            return diff if np.isfinite(diff) else 1e9

        except Exception:
            # If pattern evaluation fails, return a large value so root-finder
            # avoids this region.
            return 1e9

    # --- Find FIRST crossing with target level and double it ---
    try:
        max_angle_deg = 90.0
        step_deg = 0.01  # coarse grid; refined by bisection below
        grid_deg = np.arange(0.0, max_angle_deg + step_deg, step_deg)

        # Vectorized evaluation when possible (fast), otherwise fallback to scalar loop.
        try:
            g = antenna_gain_func(grid_deg * u.deg, **antenna_pattern_kwargs)
            if isinstance(g, (tuple, list)):
                g = g[0]
            if not hasattr(g, "unit"):
                g = g * cnv.dBi
            diff = np.asarray(
                g.to_value(cnv.dBi) - G_target_dBi.to_value(cnv.dBi),
                dtype=float,
            )
            if diff.shape != grid_deg.shape:
                raise ValueError(
                    "Vector angle input must produce vector gain output with matching shape."
                )
        except Exception:
            diff = np.array([gain_diff(float(a)) for a in grid_deg], dtype=float)

        finite = np.isfinite(diff)
        if not finite.any():
            raise RuntimeError("No finite gain values in scan grid.")

        if mode == "first_crossing":
            # First angle where we REACH or go BELOW the target.
            mask = finite & (diff <= 0.0)
            if not mask.any():
                raise RuntimeError(
                    "Target level not reached within scan grid (0..90 deg)."
                )

            i0 = int(np.argmax(mask))
            if i0 == 0:
                theta_root_deg = 0.0
            else:
                lo = float(grid_deg[i0 - 1])
                hi = float(grid_deg[i0])

                # Refine the FIRST-crossing boundary with bisection on the
                # predicate (diff <= 0). This also works for flat plateaus.
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if gain_diff(mid) <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                theta_root_deg = hi
        else:  # mode == "sidelobe_safe"
            # Find the LAST angle where gain is above target; the beamwidth
            # boundary is the next crossing after this last violation.
            violation_idx = np.flatnonzero(finite & (diff > 0.0))

            if violation_idx.size == 0:
                theta_root_deg = 0.0
            else:
                i_last = int(violation_idx[-1])
                j = i_last + 1

                while j < grid_deg.size and (not finite[j] or diff[j] > 0.0):
                    j += 1

                if j >= grid_deg.size:
                    raise RuntimeError(
                        "Target attenuation is still violated at scan boundary (90 deg)."
                    )

                lo = float(grid_deg[i_last])
                hi = float(grid_deg[j])

                # Refine the LAST-violation boundary with bisection on the
                # predicate (diff > 0), yielding the first safe angle.
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if gain_diff(mid) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                theta_root_deg = hi

        theta_bw = 2.0 * theta_root_deg * u.deg  # FULL beamwidth
        return theta_bw

    except Exception as e:
        raise RuntimeError(f"Could not find beamwidth for level_drop={level_drop}: {e}")


def calculate_beamwidth_2d(
    antenna_gain_func: Any,
    *,
    level_drop: Any = 3.0 * cnv.dB,
    mode: str = "first_crossing",
    **antenna_pattern_kwargs: Any,
) -> Any:
    """Calculate the effective beamwidth for a 2-D antenna pattern (e.g. M.2101).

    Evaluates the pattern along both principal planes (azimuth at elev=0,
    elevation at az=0) and returns the geometric mean of the two half-power
    widths as the effective 1-D beamwidth.  This is the standard approach
    for link-budget analysis with non-symmetric beam patterns.

    Falls back to ``calculate_beamwidth_1d`` if the pattern accepts only a
    single angular argument.
    """
    import inspect

    # Check if the pattern function has an 'elev' parameter
    try:
        sig = inspect.signature(antenna_gain_func)
        has_elev = "elev" in sig.parameters
    except (TypeError, ValueError):
        has_elev = False

    if not has_elev:
        # 1-D pattern — fall back to the standard calculator
        return calculate_beamwidth_1d(
            antenna_gain_func,
            level_drop=level_drop,
            mode=mode,
            **antenna_pattern_kwargs,
        )

    # --- 2-D pattern (e.g. pycraf imt2020_composite_pattern) ---
    # Signature: func(azim, elev, azim_i, elev_i, ...)
    # azim_i/elev_i are beam steering angles — set to 0 for boresight.
    if not isinstance(level_drop, u.Quantity):
        level_drop = np.abs(level_drop) * cnv.dB
    level_drop = np.abs(level_drop.value) * level_drop.unit

    # Build kwargs with boresight steering, filtering out unsupported keys
    accepted_params = set(sig.parameters.keys())
    kw = {k: v for k, v in antenna_pattern_kwargs.items() if k in accepted_params}
    if "azim_i" not in kw and "azim_i" in accepted_params:
        kw["azim_i"] = 0.0 * u.deg
    if "elev_i" not in kw and "elev_i" in accepted_params:
        kw["elev_i"] = 0.0 * u.deg

    # Get boresight gain
    try:
        g0 = antenna_gain_func(0 * u.deg, elev=0 * u.deg, **kw)
        if isinstance(g0, (tuple, list)):
            g0 = g0[0]
        if not isinstance(g0, u.Quantity):
            g0 = g0 * cnv.dBi
        Gmax_dBi = float(g0.to_value(cnv.dBi))
    except Exception as e:
        raise ValueError(f"Could not evaluate 2-D pattern at boresight: {e}") from e

    target_dBi = Gmax_dBi - float(level_drop.to_value(cnv.dB))

    def _find_crossing(scan_func, max_deg=90.0, step_deg=0.1):
        """Find first angle where gain drops below target along one plane."""
        angles = np.arange(0.0, max_deg + step_deg, step_deg)
        try:
            gains = scan_func(angles)
            if isinstance(gains, (tuple, list)):
                gains = gains[0]
            if hasattr(gains, "to_value"):
                gains_db = np.asarray(gains.to_value(cnv.dBi), dtype=np.float64)
            else:
                gains_db = np.asarray(gains, dtype=np.float64)
        except Exception:
            gains_db = np.array([
                float(scan_func(np.array([a]))[0].to_value(cnv.dBi))
                if hasattr(scan_func(np.array([a]))[0], "to_value")
                else float(scan_func(np.array([a]))[0])
                for a in angles
            ], dtype=np.float64)

        diff = gains_db - target_dBi
        crossings = np.where(np.isfinite(diff) & (diff < 0))[0]
        if len(crossings) == 0:
            return max_deg
        return float(angles[crossings[0]])

    # Scan along azimuth plane (elev=0)
    def _az_scan(az_deg):
        return antenna_gain_func(az_deg * u.deg, elev=0 * u.deg, **kw)

    # Scan along elevation plane (az=0)
    def _el_scan(el_deg):
        return antenna_gain_func(0 * u.deg, elev=el_deg * u.deg, **kw)

    theta_az = _find_crossing(_az_scan)
    theta_el = _find_crossing(_el_scan)

    # Geometric mean of the two half-power half-angles, doubled to full beamwidth
    effective_half_angle = np.sqrt(theta_az * theta_el)
    return 2.0 * effective_half_angle * u.deg

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
    Gm        =(-1e3, 1e3,  cnv.dBi),    # optional absolute peak
    far_sidelobe_start=(0, 180, u.deg),    # optional cutoff angle
    far_sidelobe_level=(-1e3, 1e3, cnv.dBi),  # gain beyond cutoff
    strip_input_units=True,                # hand floats to body
    allow_none=True)                       # Gm may be None
def s_1528_rec1_4_pattern_amend(
    offset_theta,
    offset_phi = 0 * u.deg,
    *,
    meshgrid: bool = False,
    SLR: float = 16.8 * cnv.dB,
    Lr : float = 1.8 * u.m,
    Lt : float = 1.8 * u.m,
    wavelength: float = (10.7 * u.GHz).to(
        u.m, equivalencies=u.spectral()),
    Gm: float | None = None,
    far_sidelobe_start: float | None = None,
    far_sidelobe_level: float | None = None,
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
    Gm : float, **dBi**, optional
        Absolute peak gain.  If *None*, the pattern is returned *relative*
        (0 dB at boresight).
    far_sidelobe_start : float, **degrees**, optional
        Boundary angle where an outer constant-gain region starts.  If set,
        `far_sidelobe_level` must also be set.
    far_sidelobe_level : float, **dBi**, optional
        Gain value enforced for angles at/above `far_sidelobe_start`.
        If set, `far_sidelobe_start` must also be set.
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
        NumPy fallback even if Numba is available.  The default (*None*)
        reuses cached availability from any earlier explicit opt-in and does
        not trigger a Numba import by itself.

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
    #    We keep a tiny, optional Numba JIT that is only loaded after an
    #    explicit ``use_numba=True`` request.  The default path remains NumPy.
    # ---------------------------------------------------------------
    use_numba = _HAVE_NUMBA if use_numba is None else bool(use_numba)
    numba_njit = _load_numba_njit() if use_numba else None

    if numba_njit is not None:
        @numba_njit(cache=True, fastmath=True, nogil=True)
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
    if Gm is not None:
        gain += float(Gm)

    if (far_sidelobe_start is None) ^ (far_sidelobe_level is None):
        raise ValueError(
            "far_sidelobe_start and far_sidelobe_level must be provided together."
        )

    # Optional far-sidelobe clamp:
    # - fast path for purely radial (phi==0) case;
    # - otherwise use spherical separation from boresight via pycraf.
    if far_sidelobe_start is not None:
        far_start = float(far_sidelobe_start)
        far_level = float(far_sidelobe_level)

        if np.allclose(ph, 0.0):
            far_mask = np.abs(th) >= far_start
        else:
            sep = true_angular_distance(
                np.zeros_like(ph) * u.deg,
                np.full_like(th, 90.0) * u.deg,
                ph * u.deg,
                (90.0 - np.abs(th)) * u.deg,
            )
            far_mask = sep.to_value(u.deg) >= far_start

        gain = np.where(far_mask, far_level, gain)

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
