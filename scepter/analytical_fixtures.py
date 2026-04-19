"""Analytical-to-LUT fixture pipeline for custom antenna patterns.

Stage 7 of the 30-stage custom-antenna plan (see ``CLAUDE.md``). This
module samples an analytical antenna pattern onto a user-chosen grid
and returns a :class:`CustomAntennaPattern` — ready to
:func:`dump_custom_pattern` to disk, embed in a project JSON, or feed
directly into the runtime evaluators. Every later-stage test that
wants a ground-truth custom pattern from an ITU formula builds it
through one of the functions here.

Two layers
----------

1. **Pattern-agnostic samplers**.
   :func:`sample_analytical_1d`, :func:`sample_analytical_2d_az_el`, and
   :func:`sample_analytical_2d_theta_phi` take a user-supplied callable
   (``evaluator(angles) -> gain_db``) plus a grid and build the
   :class:`CustomAntennaPattern`. Zero knowledge of any specific ITU
   formula — users can pass ``lambda theta: my_curve(theta)`` and get a
   valid schema-v1 pattern back.

2. **Evaluator factories** for the ITU/3GPP patterns SCEPTer already
   supports natively: RA.1631, S.1528 Rec 1.2, S.1528 Rec 1.4
   (axisymmetric and asymmetric), M.2101, S.672. Each returns a
   ``Callable`` suitable to pass into the corresponding sampler.

Design notes
------------

- The samplers always author ``normalisation="absolute"`` and
  ``peak_gain_source="explicit"``. Users who want a relative LUT or
  an LUT-derived peak can build one on top; the Stage 7 pipeline stays
  minimal so it composes rather than fan-out into a decision tree.
- Grids can be authored non-uniformly (dense near the main lobe, coarse
  in the far sidelobes). The samplers do no resampling — whatever grid
  the user passes ends up in the file verbatim.
- ``Callable`` signatures are vectorised: evaluators receive NumPy
  arrays and return arrays of the same shape. This keeps the sampler
  a single call per pattern rather than a Python loop.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
from astropy import units as u

from scepter.custom_antenna import (
    CustomAntennaPattern,
    GRID_MODE_AZEL,
    GRID_MODE_THETAPHI,
    KIND_1D,
    KIND_2D,
    NORMALISATION_ABSOLUTE,
    PEAK_SOURCE_EXPLICIT,
    FORMAT_VERSION,
)


# ---------------------------------------------------------------------------
# Pattern-agnostic samplers
# ---------------------------------------------------------------------------


def sample_analytical_1d(
    evaluator: Callable[[np.ndarray], np.ndarray],
    theta_grid_deg: np.ndarray,
    *,
    peak_gain_dbi: float,
    meta: Mapping[str, Any] | None = None,
) -> CustomAntennaPattern:
    """Sample a 1-D axisymmetric evaluator onto a θ grid.

    Parameters
    ----------
    evaluator :
        Vectorised callable ``evaluator(theta_deg) -> gain_db``. Must
        accept a NumPy array of degrees and return a NumPy array of dBi
        gains of the same shape.
    theta_grid_deg :
        Non-decreasing grid in degrees. Must start at 0° and extend to
        at least 180° — same rule the loader enforces on saved files
        so the sampler output is round-trippable without resampling.
        Duplicate angles are allowed (see the schema's step-discontinuity
        rule) but the caller is responsible for authoring them
        intentionally.
    peak_gain_dbi :
        Authoritative peak gain in dBi. Stored verbatim with
        ``peak_gain_source="explicit"`` — the LUT maximum is allowed to
        sit below this (e.g. for ITU regulatory masks).
    meta :
        Optional free-form metadata merged into the pattern envelope.
    """
    theta = np.asarray(theta_grid_deg, dtype=np.float64).copy()
    if theta.ndim != 1 or theta.size < 2:
        raise ValueError(
            f"theta_grid_deg must be a 1-D array with ≥ 2 points; got shape {theta.shape}"
        )
    gain = np.asarray(evaluator(theta), dtype=np.float64)
    if gain.shape != theta.shape:
        raise ValueError(
            f"evaluator returned shape {gain.shape}, expected {theta.shape}"
        )
    if not np.all(np.isfinite(gain)):
        raise ValueError("evaluator returned non-finite gain values")
    return CustomAntennaPattern(
        format_version=FORMAT_VERSION,
        kind=KIND_1D,
        normalisation=NORMALISATION_ABSOLUTE,
        peak_gain_source=PEAK_SOURCE_EXPLICIT,
        peak_gain_dbi=float(peak_gain_dbi),
        meta=dict(meta) if meta else {},
        gain_db=gain,
        grid_deg=theta,
    )


def sample_analytical_2d_az_el(
    evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    *,
    peak_gain_dbi: float,
    az_wraps: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> CustomAntennaPattern:
    """Sample a 2-D (az, el) evaluator onto a product grid.

    ``evaluator(az_deg, el_deg)`` receives broadcasted arrays of shape
    ``(N_az, N_el)`` and must return a ``(N_az, N_el)`` array of gains
    in dBi.
    """
    az = np.asarray(az_grid_deg, dtype=np.float64).copy()
    el = np.asarray(el_grid_deg, dtype=np.float64).copy()
    if az.ndim != 1 or az.size < 2 or el.ndim != 1 or el.size < 2:
        raise ValueError(
            "az_grid_deg and el_grid_deg must be 1-D arrays with ≥ 2 points; "
            f"got shapes {az.shape} and {el.shape}"
        )
    az_mesh, el_mesh = np.meshgrid(az, el, indexing="ij")
    gain = np.asarray(evaluator(az_mesh, el_mesh), dtype=np.float64)
    if gain.shape != az_mesh.shape:
        raise ValueError(
            f"evaluator returned shape {gain.shape}, expected {az_mesh.shape}"
        )
    if not np.all(np.isfinite(gain)):
        raise ValueError("evaluator returned non-finite gain values")
    return CustomAntennaPattern(
        format_version=FORMAT_VERSION,
        kind=KIND_2D,
        normalisation=NORMALISATION_ABSOLUTE,
        peak_gain_source=PEAK_SOURCE_EXPLICIT,
        peak_gain_dbi=float(peak_gain_dbi),
        meta=dict(meta) if meta else {},
        gain_db=gain,
        grid_mode=GRID_MODE_AZEL,
        az_grid_deg=az,
        el_grid_deg=el,
        az_wraps=bool(az_wraps),
    )


def sample_analytical_2d_theta_phi(
    evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_grid_deg: np.ndarray,
    phi_grid_deg: np.ndarray,
    *,
    peak_gain_dbi: float,
    phi_wraps: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> CustomAntennaPattern:
    """Sample a 2-D (θ, φ) evaluator onto a product grid.

    ``evaluator(theta_deg, phi_deg)`` receives broadcasted arrays of
    shape ``(N_theta, N_phi)`` and must return a matching gain array
    in dBi.
    """
    theta = np.asarray(theta_grid_deg, dtype=np.float64).copy()
    phi = np.asarray(phi_grid_deg, dtype=np.float64).copy()
    if theta.ndim != 1 or theta.size < 2 or phi.ndim != 1 or phi.size < 2:
        raise ValueError(
            "theta_grid_deg and phi_grid_deg must be 1-D arrays with ≥ 2 points; "
            f"got shapes {theta.shape} and {phi.shape}"
        )
    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing="ij")
    gain = np.asarray(evaluator(theta_mesh, phi_mesh), dtype=np.float64)
    if gain.shape != theta_mesh.shape:
        raise ValueError(
            f"evaluator returned shape {gain.shape}, expected {theta_mesh.shape}"
        )
    if not np.all(np.isfinite(gain)):
        raise ValueError("evaluator returned non-finite gain values")
    return CustomAntennaPattern(
        format_version=FORMAT_VERSION,
        kind=KIND_2D,
        normalisation=NORMALISATION_ABSOLUTE,
        peak_gain_source=PEAK_SOURCE_EXPLICIT,
        peak_gain_dbi=float(peak_gain_dbi),
        meta=dict(meta) if meta else {},
        gain_db=gain,
        grid_mode=GRID_MODE_THETAPHI,
        theta_grid_deg=theta,
        phi_grid_deg=phi,
        phi_wraps=bool(phi_wraps),
    )


# ---------------------------------------------------------------------------
# Evaluator factories for SCEPTer's built-in analytical patterns
# ---------------------------------------------------------------------------


def ra1631_evaluator(
    *, diameter_m: float, wavelength_m: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a vectorised RA.1631 evaluator ``f(theta_deg) -> gain_db``.

    Wraps :func:`pycraf.antenna.ras_pattern` with the diameter and
    wavelength baked in. Returned gains are absolute dBi.
    """
    from pycraf.antenna import ras_pattern
    from pycraf import conversions as cnv

    d = float(diameter_m) * u.m
    wl = float(wavelength_m) * u.m

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        theta_q = np.asarray(theta_deg, dtype=np.float64) * u.deg
        gain = ras_pattern(theta_q, d, wl)
        return np.asarray(gain.to_value(cnv.dBi), dtype=np.float64)

    return _eval


def s1528_rec1_2_evaluator(
    *,
    gm_dbi: float,
    diameter_m: float,
    wavelength_m: float,
    ln_db: float = -15.0,
    lf_dbi: float = 0.0,
    z: float = 1.0,
    axis: str = "major",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a vectorised S.1528 Rec 1.2 evaluator ``f(theta_deg) -> gain_db``.

    Thin wrapper over :func:`scepter.antenna.s_1528_rec1_2_pattern`.
    """
    from scepter.antenna import s_1528_rec1_2_pattern
    from pycraf import conversions as cnv

    gm_q = float(gm_dbi) * cnv.dBi
    d_q = float(diameter_m) * u.m
    wl_q = float(wavelength_m) * u.m
    ln_q = float(ln_db) * cnv.dB
    lf_q = float(lf_dbi) * cnv.dBi

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        theta_q = np.asarray(theta_deg, dtype=np.float64) * u.deg
        gain = s_1528_rec1_2_pattern(
            theta_q,
            axis=axis,
            Gm=gm_q,
            LN=ln_q,
            LF=lf_q,
            D=d_q,
            wavelength=wl_q,
            z=float(z),
        )
        return np.asarray(gain.to_value(cnv.dBi), dtype=np.float64)

    return _eval


# S.672 uses the same envelope formula as S.1528 Rec 1.2 in SCEPTer's
# production dispatch (see scepter/antenna.py:build_satellite_pattern_spec —
# both "s1528_rec1_2" and "s672" route through s_1528_rec1_2_pattern). The
# alias is exposed for user readability; it is a one-line indirection,
# not a re-implementation.
s672_evaluator = s1528_rec1_2_evaluator


def s1528_rec1_4_evaluator(
    *,
    wavelength_m: float,
    lr_m: float,
    lt_m: float,
    slr_db: float,
    gm_db: float | None = None,
    far_sidelobe_start_deg: float | None = None,
    far_sidelobe_level_db: float | None = None,
    l: int = 4,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a vectorised S.1528 Rec 1.4 evaluator ``f(theta_deg, phi_deg) -> gain_db``.

    Always returns a 2-D (θ, φ) evaluator regardless of ``lt``/``lr``
    symmetry — symmetric apertures just give φ-invariant output.
    Callers targeting the 1-D sampler should pass ``lambda t: eval(t, 0.0)``
    or use :func:`s1528_rec1_2_evaluator` directly.
    """
    from scepter.antenna import s_1528_rec1_4_pattern_amend
    from pycraf import conversions as cnv

    wl_q = float(wavelength_m) * u.m
    lr_q = float(lr_m) * u.m
    lt_q = float(lt_m) * u.m
    slr_q = float(slr_db) * cnv.dB
    gm_q = float(gm_db) * cnv.dB if gm_db is not None else None
    fsl_start_q = (
        float(far_sidelobe_start_deg) * u.deg
        if far_sidelobe_start_deg is not None
        else None
    )
    fsl_level_q = (
        float(far_sidelobe_level_db) * cnv.dB
        if far_sidelobe_level_db is not None
        else None
    )

    def _eval(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
        theta_q = np.asarray(theta_deg, dtype=np.float64) * u.deg
        phi_q = np.asarray(phi_deg, dtype=np.float64) * u.deg
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = s_1528_rec1_4_pattern_amend(
                theta_q, phi_q,
                SLR=slr_q, Lr=lr_q, Lt=lt_q, wavelength=wl_q,
                Gm=gm_q,
                far_sidelobe_start=fsl_start_q,
                far_sidelobe_level=fsl_level_q,
                l=int(l),
            )
        arr = np.asarray(gain.to_value(cnv.dB), dtype=np.float64)
        # Bessel / Taylor-product nulls can drive log10(0) → -inf; floor
        # at -200 dB to match the GPU kernel's pattern-null handling.
        return np.where(np.isfinite(arr), arr, -200.0)

    return _eval


def m2101_evaluator(
    *,
    g_emax_dbi: float,
    a_m_db: float,
    sla_nu_db: float,
    phi_3db_deg: float,
    theta_3db_deg: float,
    d_h: float,
    d_v: float,
    n_h: int,
    n_v: int,
    steering_az_deg: float = 0.0,
    steering_el_deg: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a vectorised M.2101 composite-pattern evaluator
    ``f(az_deg, el_deg) -> gain_db``.

    The steering direction defaults to boresight ``(0°, 0°)`` — the
    conventional authoring choice when exporting an M.2101 datasheet
    table: "what does this array look like when pointed at boresight".
    Callers exporting a beam steered elsewhere can override via the
    ``steering_*_deg`` arguments.

    The resulting LUT therefore captures the full composite pattern
    (element × array factor) at the chosen steering; it does *not*
    re-evaluate the AF at runtime. That's the whole point of the custom
    LUT path — the user saves the realised gain surface, not the
    parametric model.
    """
    from pycraf.antenna import imt2020_composite_pattern
    from pycraf import conversions as cnv

    g_emax_q = float(g_emax_dbi) * cnv.dBi
    a_m_q = float(a_m_db) * cnv.dB
    sla_q = float(sla_nu_db) * cnv.dB
    phi3_q = float(phi_3db_deg) * u.deg
    theta3_q = float(theta_3db_deg) * u.deg
    d_h_q = float(d_h) * cnv.dimless
    d_v_q = float(d_v) * cnv.dimless
    steer_az_q = float(steering_az_deg) * u.deg
    steer_el_q = float(steering_el_deg) * u.deg

    def _eval(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        az_q = np.asarray(az_deg, dtype=np.float64) * u.deg
        el_q = np.asarray(el_deg, dtype=np.float64) * u.deg
        with np.errstate(divide="ignore"):
            gain = imt2020_composite_pattern(
                az_q, el_q, steer_az_q, steer_el_q,
                G_Emax=g_emax_q, A_m=a_m_q, SLA_nu=sla_q,
                phi_3db=phi3_q, theta_3db=theta3_q,
                d_H=d_h_q, d_V=d_v_q,
                N_H=int(n_h), N_V=int(n_v),
            )
        arr = np.asarray(gain.to_value(cnv.dB), dtype=np.float64)
        # Array-factor nulls drive log10(0) → -inf. SCEPTer's GPU kernel
        # floors those at -200 dB (see _get_m2101_lut_composite_kernel);
        # mirror that convention here so the sampled LUT matches the
        # runtime evaluator instead of tripping the non-finite guard.
        return np.where(np.isfinite(arr), arr, -200.0)

    return _eval


def f699_evaluator(
    *, g_max_dbi: float, diameter_m: float, wavelength_m: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a vectorised ITU-R F.699 evaluator ``f(theta_deg) -> gain_db``.

    Thin wrapper around :func:`pycraf.antenna.fl_pattern` — the
    F.699 envelope is the peak-gain reference for fixed-satellite-
    service earth stations and line-of-sight point-to-point radio
    relays (e.g. microwave backhaul). Applicable for ``D/λ ≥ 100``;
    for smaller dishes the envelope collapses to a simplified form.
    """
    from pycraf.antenna import fl_pattern
    from pycraf import conversions as cnv

    gmax_q = float(g_max_dbi) * cnv.dBi
    d_q = float(diameter_m) * u.m
    wl_q = float(wavelength_m) * u.m

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        theta_q = np.asarray(theta_deg, dtype=np.float64) * u.deg
        # ``fl_pattern`` returns gain in dBi (log ratio) directly.
        gain = fl_pattern(theta_q, d_q, wl_q, gmax_q)
        return np.asarray(gain.to_value(cnv.dBi), dtype=np.float64)

    return _eval


def f1245_evaluator(
    *, g_max_dbi: float, diameter_m: float, wavelength_m: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """ITU-R F.1245-3 (01/2019) average-sidelobe envelope for
    fixed-service radio relays. More realistic than F.699's peak
    envelope when integrating aggregate interference.

    Correct formula per Rec. ITU-R F.1245-3 Annex 1:
    - ``G_1 = 2 + 15·log₁₀(D/λ)``  (near-sidelobe plateau, both branches)
    - ``φ_m = (20·λ/D)·√(G_max − G_1)``  degrees
    - D/λ > 100:
        * ``φ_r = 15.85·(D/λ)^(−0.6)``
        * far: ``29 − 25·log₁₀(φ)`` for φ_r ≤ φ < 48°
        * back: −13 dBi for 48° ≤ φ ≤ 180°
    - D/λ ≤ 100:
        * ``φ_r = 39.8·(D/λ)^(−0.6)``
        * far: ``39 − 5·log₁₀(D/λ) − 25·log₁₀(φ)`` for φ_r ≤ φ < 48°
        * back: ``−3 − 5·log₁₀(D/λ)`` for 48° ≤ φ ≤ 180°

    Previous SCEPTer implementation had ``G_1 = −3 − 5·log₁₀(D/λ)``
    which is F.699's back-lobe floor, not F.1245's near-sidelobe
    plateau — produced a ~50 dB near-in error for typical D/λ.
    """
    d_over_lam = float(diameter_m) / max(float(wavelength_m), 1.0e-12)
    g_max = float(g_max_dbi)
    log_dovlam = np.log10(max(d_over_lam, 1.0))

    # G_1 is the near-sidelobe plateau — same formula for both
    # large- and small-aperture branches in F.1245-3.
    g_1 = 2.0 + 15.0 * log_dovlam
    phi_m = 20.0 / max(d_over_lam, 1.0) * np.sqrt(max(g_max - g_1, 1.0))

    if d_over_lam > 100.0:
        phi_r = 15.85 * d_over_lam ** -0.6
        sidelobe_far_const = 29.0
        back_const = -13.0
    else:
        # Per ITU-R F.1245-3 Recommends 2: phi_r = 100·λ/D.
        phi_r = 100.0 / max(d_over_lam, 1.0)
        sidelobe_far_const = 39.0 - 5.0 * log_dovlam
        back_const = -3.0 - 5.0 * log_dovlam

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        abs_theta = np.abs(np.asarray(theta_deg, dtype=np.float64))
        main = g_max - 2.5e-3 * (d_over_lam * abs_theta) ** 2
        # Clamp the parabolic branch so it never dives below G_1.
        main = np.maximum(main, g_1)
        far = sidelobe_far_const - 25.0 * np.log10(np.maximum(abs_theta, 1.0e-3))
        g = main
        g = np.where(
            abs_theta >= phi_m,
            np.where(abs_theta < phi_r, g_1, far),
            g,
        )
        g = np.where(abs_theta >= 48.0, back_const, g)
        return g

    return _eval


def s465_evaluator(
    *, g_max_dbi: float, diameter_m: float, wavelength_m: float,
    receive_only: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """ITU-R S.465-6 (01/2010) earth-station reference envelope
    (VSAT / FSS terminals).

    Piecewise envelope:
    - 0 ≤ φ < φ_min: ``G_max``
    - φ_min ≤ φ < 48°: ``32 − 25·log₁₀(φ)`` dBi
    - 48° ≤ φ ≤ 180°: ``−10`` dBi

    ``φ_min`` per S.465-6 text:
    - D/λ ≥ 50: ``max(1°, 100·λ/D)``
    - D/λ < 50 (Note 3): ``max(2°, 114·(D/λ)^(−1.09))``
    - D/λ < 33.3 & receive-only (Note 5): ``2.5°``
    """
    d_over_lam = float(diameter_m) / max(float(wavelength_m), 1.0e-12)
    g_max = float(g_max_dbi)
    # φ_min branching per S.465-6.
    if d_over_lam >= 50.0:
        phi_min = max(1.0, 100.0 / d_over_lam)
    elif receive_only and d_over_lam < 33.3:
        phi_min = 2.5
    else:
        phi_min = max(2.0, 114.0 * d_over_lam ** (-1.09))
    back_floor = -10.0

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        theta = np.abs(np.asarray(theta_deg, dtype=np.float64))
        g = np.full_like(theta, g_max)
        sidelobe = 32.0 - 25.0 * np.log10(np.maximum(theta, 1.0e-3))
        g = np.where(theta >= phi_min, sidelobe, g)
        g = np.where(theta > 48.0, back_floor, g)
        return g

    return _eval


def oneweb_ecc271_satellite_evaluator(
    *,
    g_max_dbi: float = 30.0,
    along_track_axis: str = "elevation",
    beam_along_offset_deg: float = 0.0,
    beam_cross_offset_deg: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """OneWeb satellite Ku-band beam pattern from
    `ECC Report 271 <https://docdb.cept.org/download/3ab9e6bc-0afd/ECC%20Report%20271.pdf>`_
    Annex 1 §A1.1 Table 11 (10.7-12.75 GHz space-to-Earth).

    The pattern is a 2-D asymmetric fan-beam defined via the two
    orthogonal along-track / cross-track cuts. Each cut is given as
    a ``(roll-off, offset-angle)`` table — we linearly interpolate
    the roll-off (in dB) vs offset angle independently along each
    axis, then sum the two roll-offs (separable fan-beam
    assumption — exact for a rectangular-aperture antenna). The
    result is a narrow along-track (~3.3° at −20 dB) × wide
    cross-track (~73.7° at −20 dB) beam with a flat −20 dB back-
    lobe floor.

    Parameters
    ----------
    g_max_dbi :
        Peak boresight gain in dBi. Typical for OneWeb Ku-band
        satellite beams: ~30 dBi.
    along_track_axis :
        Which editor axis (``"elevation"`` or ``"azimuth"``) the
        satellite's along-track direction maps to. Default is
        elevation — matches the common "narrow-in-el, wide-in-az"
        plotting convention in ECC-271.
    beam_along_offset_deg / beam_cross_offset_deg :
        Nadir-offset of the beam centre in the along-track /
        cross-track axis (Table 10 in ECC 271 places the 16 beams
        at rows of −23.5° to +23.5° in the along-track direction).
        0 by default — a centred beam.
    """
    # Roll-off (dB below peak) vs offset angle (degrees), directly
    # transcribed from ECC Report 271 Annex 1 §A1.1 Table 11.
    rolloff_db = np.array([0.0, 0.2, 1.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 20.0])
    along_offset_deg = np.array([0.0, 0.5, 1.0, 1.2, 1.6, 1.95, 2.3, 2.9, 3.3, 180.0])
    cross_offset_deg = np.array([0.0, 6.0, 14.0, 19.0, 27.8, 35.9, 44.2, 52.7, 73.7, 180.0])

    g_max = float(g_max_dbi)

    def _interp_rolloff(abs_offset: np.ndarray, table_offset: np.ndarray) -> np.ndarray:
        """Linear-in-dB interpolation of roll-off vs |offset|."""
        return np.interp(
            np.abs(abs_offset),
            table_offset,
            rolloff_db,
            left=0.0, right=20.0,
        )

    def _eval(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        az = np.asarray(az_deg, dtype=np.float64)
        el = np.asarray(el_deg, dtype=np.float64)
        if str(along_track_axis).lower().startswith("el"):
            along = el - float(beam_along_offset_deg)
            cross = az - float(beam_cross_offset_deg)
        else:
            along = az - float(beam_along_offset_deg)
            cross = el - float(beam_cross_offset_deg)
        r_along = _interp_rolloff(along, along_offset_deg)
        r_cross = _interp_rolloff(cross, cross_offset_deg)
        # Separable fan-beam: sum the two roll-offs in dB (= product
        # in linear). Clamp at −20 dB floor (Table 11's far-sidelobe
        # plateau).
        total = np.minimum(r_along + r_cross, 20.0)
        return g_max - total

    return _eval


# Back-compat alias for older callers. No longer a 1-D wrapper —
# the new evaluator is 2-D per ECC 271 Table 11.
oneweb_ecc271_evaluator = oneweb_ecc271_satellite_evaluator


# ---------------------------------------------------------------------------
# ITU-R S.580-6 (2004) — GSO earth-station design-objective envelope
# ---------------------------------------------------------------------------


def s580_evaluator(
    *, diameter_m: float, wavelength_m: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """ITU-R S.580-6 radiation-diagram design objective for earth
    stations operating with GSO satellites.

    Uses the RR Appendix 8 parabolic main-lobe model
    ``G_max - 0.0025·(D·φ/λ)²`` and delegates to a ``32 - 25·log₁₀(φ)``
    sidelobe envelope (S.465-style) above ``φ_min``.

    Applicable for ``D/λ ≥ 50``.

    Reference
    ---------
    ITU-R S.580-6 (01/2004) §§2-3.
    """
    d_over_lam = float(diameter_m) / max(float(wavelength_m), 1.0e-12)
    g_max = 20.0 * np.log10(max(d_over_lam, 1.0)) + 7.7  # η ≈ 0.6
    phi_min = max(1.0, 100.0 / d_over_lam)

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        phi = np.abs(np.asarray(theta_deg, dtype=np.float64))
        # Main lobe — RR Appendix 8 parabolic rolloff.
        main = g_max - 2.5e-3 * (d_over_lam * phi) ** 2
        sidelobe = 32.0 - 25.0 * np.log10(np.maximum(phi, 1.0e-6))
        g = np.where(phi < phi_min, main, sidelobe)
        g = np.where(phi >= 48.0, -10.0, g)
        return g

    return _eval


# ---------------------------------------------------------------------------
# ITU-R S.1428-1 (2001) — Non-GSO EPFD reference earth-station pattern
# ---------------------------------------------------------------------------


def s1428_evaluator(
    *, diameter_m: float, wavelength_m: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """ITU-R S.1428-1 reference earth-station radiation pattern
    for non-GSO interference assessment (10.7-30 GHz).

    This is the pattern used by the S.1503 EPFD calculation
    methodology. It models the main lobe explicitly (parabolic
    rolloff) and includes back-lobe structure, making it more
    realistic than S.465/S.580 for non-GSO geometry.

    Reference
    ---------
    ITU-R S.1428-1 (02/2001) Annex 1 §1 (D/λ > 100).
    """
    d_over_lam = float(diameter_m) / max(float(wavelength_m), 1.0e-12)
    g_max = 20.0 * np.log10(max(d_over_lam, 1.0)) + 8.4  # η ≈ 0.7
    g_1 = -1.0 + 15.0 * np.log10(max(d_over_lam, 1.0))
    phi_m = 20.0 / max(d_over_lam, 1.0) * np.sqrt(max(g_max - g_1, 0.0))
    phi_r = 15.85 * max(d_over_lam, 1.0) ** -0.6

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        phi = np.abs(np.asarray(theta_deg, dtype=np.float64))
        main = g_max - 2.5e-3 * (d_over_lam * phi) ** 2
        log_phi = np.log10(np.maximum(phi, 1.0e-6))
        g = main
        g = np.where(phi >= phi_m, g_1, g)
        g = np.where(phi >= phi_r, 29.0 - 25.0 * log_phi, g)
        g = np.where(phi >= 10.0, 34.0 - 30.0 * log_phi, g)
        g = np.where(phi >= 34.1, -12.0, g)
        g = np.where(phi >= 80.0, -7.0, g)
        g = np.where(phi >= 120.0, -12.0, g)
        return g

    return _eval


# ---------------------------------------------------------------------------
# ITU-R SA.509-3 (2013) — Space-research / radio-astronomy earth station
# ---------------------------------------------------------------------------


def sa509_evaluator(
    *, diameter_m: float, wavelength_m: float,
    variant: str = "single",
) -> Callable[[np.ndarray], np.ndarray]:
    """ITU-R SA.509-3 earth-station reference antenna radiation
    diagram for the space research and radio astronomy services.

    Two variants are defined:

    - ``variant="single"`` (Pattern 1): single-entry interference,
      uses ``32 - 25·log₁₀(φ)`` sidelobe envelope.
    - ``variant="aggregate"`` (Pattern 2): aggregate interference,
      uses ``29 - 25·log₁₀(φ)`` (3 dB lower sidelobes).

    Reference
    ---------
    ITU-R SA.509-3 (12/2013) Annex 1.
    """
    d_over_lam = float(diameter_m) / max(float(wavelength_m), 1.0e-12)
    g_max = 20.0 * np.log10(max(d_over_lam, 1.0)) + 8.4
    g_1 = -1.0 + 15.0 * np.log10(max(d_over_lam, 1.0))
    phi_m = 20.0 / max(d_over_lam, 1.0) * np.sqrt(max(g_max - g_1, 0.0))
    phi_r = 15.85 * max(d_over_lam, 1.0) ** -0.6

    if variant == "single":
        sidelobe_const = 32.0
        back_floor = -10.0
        back_rear = -10.0
        back_mid = -5.0
    else:
        sidelobe_const = 29.0
        back_floor = -13.0
        back_rear = -13.0
        back_mid = -8.0

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        phi = np.abs(np.asarray(theta_deg, dtype=np.float64))
        main = g_max - 2.5e-3 * (d_over_lam * phi) ** 2
        log_phi = np.log10(np.maximum(phi, 1.0e-6))
        g = main
        g = np.where(phi >= phi_m, g_1, g)
        g = np.where(phi >= phi_r, sidelobe_const - 25.0 * log_phi, g)
        g = np.where(phi >= 48.0, back_floor, g)
        g = np.where(phi >= 80.0, back_mid, g)
        g = np.where(phi >= 120.0, back_rear, g)
        return g

    return _eval


# ---------------------------------------------------------------------------
# ITU-R F.1336-5 (2019) — Omnidirectional base-station antenna
# ---------------------------------------------------------------------------


def f1336_omni_evaluator(
    *, g0_dbi: float, theta_3db_deg: float | None = None,
    k: float = 0.7,
    variant: str = "peak",
) -> Callable[[np.ndarray], np.ndarray]:
    """ITU-R F.1336-5 omnidirectional antenna (elevation-only pattern).

    Models the vertical radiation pattern of a base-station antenna
    with uniform azimuthal coverage (e.g. a collinear array or
    dipole stack). Essential for IMT/5G-into-RAS interference studies.

    Parameters
    ----------
    g0_dbi : peak gain (dBi)
    theta_3db_deg : 3 dB beamwidth in elevation. If None, estimated
        from ``g0_dbi`` via ``107.6·10^(-0.1·G_0)``.
    k : sidelobe adjustment factor (0.7 typical ≤ 3 GHz, 0 improved)
    variant : "peak" for peak-sidelobe envelope, "average" for
        average-sidelobe.

    Reference
    ---------
    ITU-R F.1336-5 (01/2019) Annex 1 §§2.1-2.2.
    """
    g0 = float(g0_dbi)
    if theta_3db_deg is None:
        theta_3 = 107.6 * 10.0 ** (-0.1 * g0)
    else:
        theta_3 = float(theta_3db_deg)
    kv = float(k)

    if variant == "peak":
        log_term = np.log10(max(kv + 1.0, 1.0e-12))
        # theta_4 = theta_3 * sqrt(1 - (1/1.2)*log10(k+1))
        inner = max(1.0 - log_term / 1.2, 0.0)
        theta_4 = theta_3 * np.sqrt(inner)
        sl_offset = -12.0 + 10.0 * log_term
    else:
        log_term = np.log10(max(kv + 1.0, 1.0e-12))
        inner = max(1.25 - log_term / 1.2, 0.0)
        theta_4 = theta_3 * np.sqrt(inner)
        sl_offset = -15.0 + 10.0 * log_term

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta_deg, dtype=np.float64)
        at = np.abs(theta)
        main = g0 - 12.0 * (at / theta_3) ** 2
        ratio = np.maximum(at / theta_3, 1.0e-6)
        sl = g0 + sl_offset + 10.0 * np.log10(ratio ** (-1.5) + kv)
        g = np.where(at < theta_4, main, g0 + sl_offset)
        g = np.where(at >= theta_3, sl, g)
        return g

    return _eval


# ---------------------------------------------------------------------------
# ITU-R F.1336-5 (2019) — Sectoral base-station antenna (2-D)
# ---------------------------------------------------------------------------


def f1336_sectoral_evaluator(
    *, g0_dbi: float,
    phi_3db_deg: float = 65.0,
    theta_3db_deg: float = 8.0,
    k_h: float = 0.7,
    k_v: float = 0.3,
    tilt_deg: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """ITU-R F.1336-5 sectoral antenna (azimuth + elevation pattern).

    Models the 2-D radiation pattern of a panel-type base-station
    antenna with directional coverage in azimuth. The vertical and
    horizontal patterns are combined via an additive model in dB.

    Parameters
    ----------
    g0_dbi : peak gain (dBi)
    phi_3db_deg : horizontal 3 dB beamwidth (degrees)
    theta_3db_deg : vertical 3 dB beamwidth (degrees)
    k_h, k_v : horizontal/vertical sidelobe parameters
    tilt_deg : mechanical/electrical downtilt (positive = below horizon)

    Reference
    ---------
    ITU-R F.1336-5 (01/2019) Annex 1 §3.2.
    """
    g0 = float(g0_dbi)
    phi_3 = float(phi_3db_deg)
    theta_3 = float(theta_3db_deg)

    def _eval(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        az = np.asarray(az_deg, dtype=np.float64)
        el = np.asarray(el_deg, dtype=np.float64) - float(tilt_deg)

        # Horizontal pattern — 12·(φ/φ_3)^2 rolloff with floor.
        x_h = np.abs(az) / phi_3
        g_hr = -np.minimum(12.0 * x_h ** 2, 30.0 + float(k_h))

        # Vertical pattern — same structure as omni elevation.
        x_v = np.abs(el) / theta_3
        g_vr = -np.minimum(12.0 * x_v ** 2, 30.0 + float(k_v))

        # Combined (additive in dB, floored at -(G0 + sidelobe floor))
        g = g0 + np.maximum(g_hr + g_vr, -(g0 + 5.0))
        return g

    return _eval


# ---------------------------------------------------------------------------
# Generic: Uniform circular aperture (Airy / jinc)
# ---------------------------------------------------------------------------


def airy_evaluator(
    *, diameter_m: float, wavelength_m: float,
    efficiency: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Uniform circular aperture diffraction pattern (Airy / jinc).

    The classic ``[2·J₁(u)/u]²`` power pattern of a uniformly
    illuminated circular dish — the physics-based reference before
    any ITU regulatory envelope is applied.

    Parameters
    ----------
    diameter_m : physical aperture diameter (m)
    wavelength_m : operating wavelength (m)
    efficiency : aperture efficiency (0-1, default 1.0 for uniform)
    """
    from scipy.special import j1

    d_over_lam = float(diameter_m) / max(float(wavelength_m), 1.0e-12)
    g_max = 10.0 * np.log10(
        float(efficiency) * (np.pi * d_over_lam) ** 2
    )

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta_deg, dtype=np.float64)
        u = np.pi * d_over_lam * np.sin(np.deg2rad(np.abs(theta)))
        # 2*J1(u)/u → 1 at u=0 (L'Hôpital).
        jinc = np.where(
            np.abs(u) < 1.0e-12,
            1.0,
            2.0 * j1(u) / u,
        )
        power_db = 10.0 * np.log10(np.maximum(jinc ** 2, 1.0e-30))
        return g_max + power_db

    return _eval


# ---------------------------------------------------------------------------
# Generic: Cosine-tapered circular aperture
# ---------------------------------------------------------------------------


def cosine_taper_evaluator(
    *, diameter_m: float, wavelength_m: float,
    taper_order: int = 1,
    efficiency: float = 0.75,
) -> Callable[[np.ndarray], np.ndarray]:
    """Cosine-tapered circular aperture: ``[1 - (r/a)²]^n`` taper.

    Reduces sidelobes at the cost of a wider main beam and lower
    aperture efficiency compared to the uniform (Airy) case.

    - ``taper_order=0``: uniform illumination (≡ Airy, -17.6 dB SLL)
    - ``taper_order=1``: parabolic taper (-24.6 dB SLL, η ≈ 0.75)
    - ``taper_order=2``: strong taper (-30.6 dB SLL, η ≈ 0.56)

    The normalised pattern is ``E(u) = 2^n · n! · J_{n+1}(u) / u^{n+1}``
    where ``u = π·D·sin(θ)/λ``.

    Parameters
    ----------
    diameter_m : physical aperture diameter (m)
    wavelength_m : operating wavelength (m)
    taper_order : cosine taper exponent n (0, 1, or 2)
    efficiency : aperture efficiency (default 0.75 for n=1)
    """
    from scipy.special import jv
    from math import factorial

    n = int(taper_order)
    d_over_lam = float(diameter_m) / max(float(wavelength_m), 1.0e-12)
    g_max = 10.0 * np.log10(
        float(efficiency) * (np.pi * d_over_lam) ** 2
    )
    norm = 2.0 ** n * factorial(n)

    def _eval(theta_deg: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta_deg, dtype=np.float64)
        u = np.pi * d_over_lam * np.sin(np.deg2rad(np.abs(theta)))
        # E(u) = 2^n * n! * J_{n+1}(u) / u^{n+1}, normalised to 1 at u=0.
        field = np.where(
            np.abs(u) < 1.0e-12,
            1.0,
            norm * jv(n + 1, u) / np.maximum(np.abs(u) ** (n + 1), 1.0e-30),
        )
        power_db = 10.0 * np.log10(np.maximum(field ** 2, 1.0e-30))
        return g_max + power_db

    return _eval


# ---------------------------------------------------------------------------
# Generic: Uniform rectangular aperture (sinc × sinc) — 2-D
# ---------------------------------------------------------------------------


def sinc_rect_evaluator(
    *, lx_m: float, lz_m: float, wavelength_m: float,
    efficiency: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Uniform rectangular aperture: separable ``sinc² × sinc²``.

    The simplest 2-D aperture pattern — factorises into two
    independent sinc functions along the horizontal and vertical
    axes. Useful as a generic 2-D reference.

    Parameters
    ----------
    lx_m : horizontal aperture dimension (m)
    lz_m : vertical aperture dimension (m)
    wavelength_m : operating wavelength (m)
    efficiency : aperture efficiency (default 1.0)
    """
    lx_over_lam = float(lx_m) / max(float(wavelength_m), 1.0e-12)
    lz_over_lam = float(lz_m) / max(float(wavelength_m), 1.0e-12)
    g_max = 10.0 * np.log10(
        float(efficiency) * 4.0 * np.pi * lx_over_lam * lz_over_lam
    )

    def _eval(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        az = np.asarray(az_deg, dtype=np.float64)
        el = np.asarray(el_deg, dtype=np.float64)
        ux = np.pi * lx_over_lam * np.sin(np.deg2rad(az))
        uz = np.pi * lz_over_lam * np.sin(np.deg2rad(el))
        sinc_x = np.where(np.abs(ux) < 1.0e-12, 1.0, np.sin(ux) / ux)
        sinc_z = np.where(np.abs(uz) < 1.0e-12, 1.0, np.sin(uz) / uz)
        power_db = 10.0 * np.log10(
            np.maximum(sinc_x ** 2 * sinc_z ** 2, 1.0e-30)
        )
        return g_max + power_db

    return _eval


__all__ = [
    "sample_analytical_1d",
    "sample_analytical_2d_az_el",
    "sample_analytical_2d_theta_phi",
    "ra1631_evaluator",
    "s1528_rec1_2_evaluator",
    "s672_evaluator",
    "s1528_rec1_4_evaluator",
    "m2101_evaluator",
    "f699_evaluator",
    "f1245_evaluator",
    "s465_evaluator",
    "s580_evaluator",
    "s1428_evaluator",
    "sa509_evaluator",
    "f1336_omni_evaluator",
    "f1336_sectoral_evaluator",
    "airy_evaluator",
    "cosine_taper_evaluator",
    "sinc_rect_evaluator",
    "oneweb_ecc271_evaluator",
    "oneweb_ecc271_satellite_evaluator",
]
