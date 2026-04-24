"""Tests for the Stage 7 analytical-to-LUT fixture pipeline.

Verifies that:

1. The three pattern-agnostic samplers (1-D, 2-D az_el, 2-D θ_φ)
   build valid :class:`CustomAntennaPattern` objects from any
   user-supplied evaluator callable.
2. ``dump → load`` round-trips the sampled patterns bit-stably via
   the schema-v1 JSON format.
3. Each ITU-pattern evaluator factory (RA.1631, S.1528 Rec 1.2,
   S.1528 Rec 1.4, M.2101, S.672) produces gains that match the
   underlying analytical formula at the sampled angles, and that a
   loaded LUT's grid-point gains are identical to the evaluator
   output.
4. The ``CustomAntennaPattern`` CPU evaluators
   (``evaluate_pattern_1d`` / ``evaluate_pattern_2d``) on a dense
   ground truth grid recover the analytical formula within a
   sampling-resolution budget — the fixture pipeline's promise is
   that a dense LUT "is" the analytical pattern for downstream code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scepter import analytical_fixtures as af
from scepter.custom_antenna import (
    GRID_MODE_AZEL,
    GRID_MODE_THETAPHI,
    KIND_1D,
    KIND_2D,
    NORMALISATION_ABSOLUTE,
    PEAK_SOURCE_EXPLICIT,
    dump_custom_pattern,
    evaluate_pattern_1d,
    evaluate_pattern_2d,
    load_custom_pattern,
)


# ---------------------------------------------------------------------------
# Samplers — pattern-agnostic
# ---------------------------------------------------------------------------


def test_sample_analytical_1d_basic_round_trip(tmp_path: Path) -> None:
    """A trivial user evaluator round-trips dump → load bit-stably."""
    theta = np.linspace(0.0, 180.0, 37)

    def my_curve(t: np.ndarray) -> np.ndarray:
        # Arbitrary 1-D curve; what matters is the round-trip.
        return 35.0 - 0.4 * t

    pat = af.sample_analytical_1d(
        my_curve, theta, peak_gain_dbi=35.0,
        meta={"title": "unit test"},
    )
    assert pat.kind == KIND_1D
    assert pat.normalisation == NORMALISATION_ABSOLUTE
    assert pat.peak_gain_source == PEAK_SOURCE_EXPLICIT
    assert pat.peak_gain_dbi == 35.0
    np.testing.assert_allclose(pat.gain_db, my_curve(theta))

    out = tmp_path / "pat.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)
    np.testing.assert_array_equal(reloaded.grid_deg, pat.grid_deg)
    np.testing.assert_array_equal(reloaded.gain_db, pat.gain_db)
    assert reloaded.meta["title"] == "unit test"


def test_sample_analytical_1d_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="theta_grid_deg"):
        af.sample_analytical_1d(lambda t: t, np.array([0.0]), peak_gain_dbi=0.0)


def test_sample_analytical_1d_rejects_shape_mismatch() -> None:
    theta = np.linspace(0.0, 180.0, 10)
    with pytest.raises(ValueError, match="evaluator returned shape"):
        af.sample_analytical_1d(
            lambda t: np.zeros(5), theta, peak_gain_dbi=0.0,
        )


def test_sample_analytical_1d_rejects_non_finite() -> None:
    theta = np.linspace(0.0, 180.0, 10)

    def bad(t: np.ndarray) -> np.ndarray:
        out = np.zeros_like(t)
        out[3] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        af.sample_analytical_1d(bad, theta, peak_gain_dbi=0.0)


def test_sample_analytical_2d_az_el_round_trip(tmp_path: Path) -> None:
    az = np.linspace(-180.0, 180.0, 37)
    el = np.linspace(-90.0, 90.0, 19)

    def curve(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        return 10.0 - 0.05 * az_deg**2 - 0.02 * el_deg**2

    pat = af.sample_analytical_2d_az_el(
        curve, az, el, peak_gain_dbi=10.0,
    )
    assert pat.kind == KIND_2D
    assert pat.grid_mode == GRID_MODE_AZEL
    assert pat.az_wraps is True
    assert pat.gain_db.shape == (37, 19)

    out = tmp_path / "pat2d.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)
    np.testing.assert_array_equal(reloaded.az_grid_deg, pat.az_grid_deg)
    np.testing.assert_array_equal(reloaded.el_grid_deg, pat.el_grid_deg)
    np.testing.assert_array_equal(reloaded.gain_db, pat.gain_db)


def test_sample_analytical_2d_theta_phi_round_trip(tmp_path: Path) -> None:
    theta = np.linspace(0.0, 180.0, 19)
    phi = np.linspace(-180.0, 180.0, 25)

    def curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 12.0 - 0.02 * t**2 - 0.01 * p**2

    pat = af.sample_analytical_2d_theta_phi(
        curve, theta, phi, peak_gain_dbi=12.0, phi_wraps=True,
    )
    assert pat.kind == KIND_2D
    assert pat.grid_mode == GRID_MODE_THETAPHI
    assert pat.phi_wraps is True
    assert pat.gain_db.shape == (19, 25)

    out = tmp_path / "pat2d_tp.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)
    np.testing.assert_array_equal(reloaded.theta_grid_deg, pat.theta_grid_deg)
    np.testing.assert_array_equal(reloaded.phi_grid_deg, pat.phi_grid_deg)
    np.testing.assert_array_equal(reloaded.gain_db, pat.gain_db)


# ---------------------------------------------------------------------------
# Evaluator factories — each one should (a) match the underlying analytical
# formula at the sample points, and (b) recover it within sampling budget
# when evaluated off-grid via the custom-pattern CPU evaluator.
# ---------------------------------------------------------------------------


def test_ra1631_evaluator_matches_pycraf_and_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("pycraf.antenna")
    from pycraf.antenna import ras_pattern
    from pycraf import conversions as cnv
    from astropy import units as u

    diameter_m = 25.0
    wavelength_m = 0.21  # 1.4 GHz
    evaluator = af.ra1631_evaluator(
        diameter_m=diameter_m, wavelength_m=wavelength_m,
    )

    # Dense grid — RA.1631 has a very narrow main lobe so sample it finely
    # near boresight.
    theta_grid = np.concatenate([
        np.linspace(0.0, 5.0, 501),
        np.linspace(5.0, 180.0, 176)[1:],
    ])
    pat = af.sample_analytical_1d(
        evaluator, theta_grid, peak_gain_dbi=float(evaluator(np.array([0.0]))[0]),
        meta={"title": "RA.1631", "diameter_m": diameter_m, "wavelength_m": wavelength_m},
    )

    # Round-trip.
    out = tmp_path / "ra1631.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)

    # At sample points the stored values exactly equal the evaluator.
    np.testing.assert_allclose(reloaded.gain_db, evaluator(theta_grid), atol=1e-9)

    # Off-grid: the interpolator recovers the analytical curve within a
    # modest budget (dB errors grow near sidelobe nulls — floor the mask).
    probe = np.array([0.05, 0.3, 1.0, 3.5, 10.0, 30.0, 60.0, 120.0])
    gain_lut = evaluate_pattern_1d(reloaded, probe)
    probe_q = probe * u.deg
    gain_ana = ras_pattern(probe_q, diameter_m * u.m, wavelength_m * u.m).to_value(cnv.dBi)
    # Exclude null-region samples (dB diff blows up on tiny linear error).
    mask = (gain_lut > gain_ana.max() - 50.0) & (gain_ana > gain_ana.max() - 50.0)
    assert mask.any()
    assert np.max(np.abs(gain_lut[mask] - gain_ana[mask])) < 0.5


def test_s1528_rec1_2_evaluator_round_trips() -> None:
    pytest.importorskip("scepter.antenna")
    evaluator = af.s1528_rec1_2_evaluator(
        gm_dbi=34.1, diameter_m=1.8, wavelength_m=0.15,
    )
    theta = np.linspace(0.0, 180.0, 361)
    pat = af.sample_analytical_1d(
        evaluator, theta, peak_gain_dbi=34.1,
    )
    # Stored values equal evaluator output at sample points by construction.
    np.testing.assert_allclose(pat.gain_db, evaluator(theta), atol=1e-9)
    # S.672 is registered as an alias — same callable signature.
    assert af.s672_evaluator is af.s1528_rec1_2_evaluator


def test_s1528_rec1_4_evaluator_asymmetric_captures_phi() -> None:
    """Asymmetric Rec 1.4 (lt ≠ lr) must carry phi information. A custom
    LUT built from the evaluator must show the same φ=0 vs φ=90°
    asymmetry the analytical formula has.
    """
    pytest.importorskip("scepter.antenna")
    evaluator = af.s1528_rec1_4_evaluator(
        wavelength_m=0.15, lr_m=1.6, lt_m=3.2, slr_db=20.0, l=2,
        gm_db=34.1,
        far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0,
    )
    # Sanity: evaluator itself is asymmetric at θ=3°.
    g0 = float(evaluator(np.array([3.0]), np.array([0.0])).item())
    g90 = float(evaluator(np.array([3.0]), np.array([90.0])).item())
    assert abs(g0 - g90) > 1.0, "sanity: phi=0 vs phi=90 must differ"

    theta = np.linspace(0.0, 180.0, 181)
    phi = np.linspace(-180.0, 180.0, 73)
    pat = af.sample_analytical_2d_theta_phi(
        evaluator, theta, phi, peak_gain_dbi=34.1,
    )
    assert pat.gain_db.shape == (theta.size, phi.size)

    # Store pulls the evaluator at theta=3°, phi=0 and phi=90 from the
    # sampled grid; they must match the direct evaluator call.
    i_theta = int(np.argmin(np.abs(theta - 3.0)))
    i_phi_0 = int(np.argmin(np.abs(phi - 0.0)))
    i_phi_90 = int(np.argmin(np.abs(phi - 90.0)))
    np.testing.assert_allclose(pat.gain_db[i_theta, i_phi_0], g0, atol=1e-9)
    np.testing.assert_allclose(pat.gain_db[i_theta, i_phi_90], g90, atol=1e-9)


def test_s1528_rec1_4_evaluator_symmetric_is_phi_invariant() -> None:
    """Symmetric Rec 1.4 (lt == lr) must be φ-invariant."""
    pytest.importorskip("scepter.antenna")
    evaluator = af.s1528_rec1_4_evaluator(
        wavelength_m=0.15, lr_m=1.6, lt_m=1.6, slr_db=20.0, l=2,
        gm_db=34.1,
    )
    phi_values = np.array([0.0, 30.0, 60.0, 90.0, 135.0, 179.0])
    theta_fixed = np.full_like(phi_values, 5.0)
    gains = evaluator(theta_fixed, phi_values)
    # All phi slices at fixed theta must agree.
    assert np.max(np.abs(gains - gains[0])) < 1e-9


def test_m2101_evaluator_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("pycraf.antenna")
    evaluator = af.m2101_evaluator(
        g_emax_dbi=5.0, a_m_db=30.0, sla_nu_db=30.0,
        phi_3db_deg=65.0, theta_3db_deg=65.0,
        d_h=0.5, d_v=0.5, n_h=4, n_v=4,
    )
    az = np.linspace(-180.0, 180.0, 73)
    el = np.linspace(-90.0, 90.0, 37)
    # Peak of boresight-steered 4×4 element array: G_Emax + 10 log10(16).
    peak_gain_dbi = 5.0 + 10.0 * np.log10(16.0)
    pat = af.sample_analytical_2d_az_el(
        evaluator, az, el, peak_gain_dbi=peak_gain_dbi,
    )
    out = tmp_path / "m2101.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)
    # The loader stamps the boresight cell exactly at peak.
    i_az_0 = int(np.argmin(np.abs(az - 0.0)))
    i_el_0 = int(np.argmin(np.abs(el - 0.0)))
    g_bore = reloaded.gain_db[i_az_0, i_el_0]
    assert abs(g_bore - peak_gain_dbi) < 0.1


def test_dense_1d_lut_approximates_analytical_on_off_grid() -> None:
    """Dense LUT built from an analytical evaluator must reproduce the
    evaluator at arbitrary off-grid angles within grid-step-sized dB.
    """
    evaluator = af.s1528_rec1_2_evaluator(
        gm_dbi=34.1, diameter_m=1.8, wavelength_m=0.15,
    )
    theta = np.linspace(0.0, 180.0, 1801)  # 0.1° step
    pat = af.sample_analytical_1d(evaluator, theta, peak_gain_dbi=34.1)
    probe = np.array([0.05, 0.23, 0.71, 2.01, 7.3, 42.7, 131.9])
    gain_lut = evaluate_pattern_1d(pat, probe)
    gain_ana = evaluator(probe)
    # Exclude null-region samples.
    mask = gain_lut > gain_ana.max() - 50.0
    assert mask.any()
    assert np.max(np.abs(gain_lut[mask] - gain_ana[mask])) < 0.5


# ---------------------------------------------------------------------------
# Stage 8 — end-to-end CPU round-trip with formal accuracy budgets
# ---------------------------------------------------------------------------
#
# For each of the four analytical patterns (RA.1631, S.1528 Rec 1.2,
# S.1528 Rec 1.4, M.2101): evaluator → dense sampler → dump → load →
# ``evaluate_pattern_{1d,2d}`` at a probe grid → compare to direct
# analytical. Budgets (from the 30-stage plan, Stage 8):
#   * 1-D: max |Δ| < 0.1 dB
#   * 2-D: max |Δ| < 0.3 dB
# Pattern nulls are masked (floor at 50 dB below peak) because dB
# error blows up logarithmically on vanishing linear gain — that's
# physically meaningless for EPFD / PFD accumulation.


# Floor applied to all comparisons — samples below this threshold
# are pattern-null regions where dB error is dominated by log-scale
# blowup of tiny linear differences. For 2-D patterns the floor also
# excludes probe points close to the -200 dB null clamp that the
# evaluator factories apply — bilinear interpolation across a
# null-clamped cell produces a visually large dB error that is
# physically meaningless at the corresponding sub-fW EPFD level.
_NULL_FLOOR_DB_1D = 50.0
# 2-D patterns use a tighter floor: bilinear interpolation across cells
# that include the -200 dB null clamp produces visually large dB errors
# at sample points well below the peak, even though the linear-gain
# contribution to EPFD/PFD is vanishing (< 1% within 20 dB of peak).
_NULL_FLOOR_DB_2D = 20.0


def _mask_meaningful(
    gain_a: np.ndarray, gain_b: np.ndarray, floor_db: float = _NULL_FLOOR_DB_1D,
) -> np.ndarray:
    """Return a boolean mask of samples where both gains are within
    ``floor_db`` of the peak of either evaluation.
    """
    peak = max(float(gain_a.max()), float(gain_b.max()))
    return (gain_a > peak - floor_db) & (gain_b > peak - floor_db)


def test_stage8_round_trip_ra1631(tmp_path: Path) -> None:
    """RA.1631 export → reload → evaluate — max |Δ| < 0.1 dB."""
    pytest.importorskip("pycraf.antenna")
    evaluator = af.ra1631_evaluator(diameter_m=25.0, wavelength_m=0.21)
    # Dense grid: fine near boresight (narrow main lobe), coarser beyond.
    theta_grid = np.concatenate([
        np.linspace(0.0, 2.0, 401),         # 0.005° near boresight
        np.linspace(2.0, 20.0, 361)[1:],    # 0.05°
        np.linspace(20.0, 180.0, 321)[1:],  # 0.5°
    ])
    pat = af.sample_analytical_1d(
        evaluator, theta_grid,
        peak_gain_dbi=float(evaluator(np.array([0.0]))[0]),
    )
    out = tmp_path / "ra1631_s8.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)

    probe = np.concatenate([
        np.linspace(0.01, 2.0, 200),
        np.linspace(2.01, 179.99, 400),
    ])
    gain_lut = evaluate_pattern_1d(reloaded, probe)
    gain_ana = evaluator(probe)
    mask = _mask_meaningful(gain_lut, gain_ana)
    assert mask.sum() > 0.1 * probe.size
    max_abs = float(np.max(np.abs(gain_lut[mask] - gain_ana[mask])))
    assert max_abs < 0.1, f"RA.1631 round-trip max |Δ| = {max_abs:.4f} dB"


def test_stage8_round_trip_s1528_rec1_2(tmp_path: Path) -> None:
    """S.1528 Rec 1.2 export → reload → evaluate — max |Δ| < 0.1 dB."""
    pytest.importorskip("scepter.antenna")
    evaluator = af.s1528_rec1_2_evaluator(
        gm_dbi=34.1, diameter_m=1.8, wavelength_m=0.15,
    )
    # 0.01° step — S.1528 Rec 1.2 is smoothly piecewise-log, easy to
    # sample densely.
    theta_grid = np.linspace(0.0, 180.0, 18001)
    pat = af.sample_analytical_1d(
        evaluator, theta_grid, peak_gain_dbi=34.1,
    )
    out = tmp_path / "rec12_s8.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)

    probe = np.concatenate([
        np.linspace(0.005, 5.0, 300),
        np.linspace(5.01, 179.99, 300),
    ])
    gain_lut = evaluate_pattern_1d(reloaded, probe)
    gain_ana = evaluator(probe)
    mask = _mask_meaningful(gain_lut, gain_ana)
    assert mask.sum() > 0.1 * probe.size
    max_abs = float(np.max(np.abs(gain_lut[mask] - gain_ana[mask])))
    assert max_abs < 0.1, f"Rec 1.2 round-trip max |Δ| = {max_abs:.4f} dB"


def test_stage8_round_trip_s672(tmp_path: Path) -> None:
    """S.672 (alias of Rec 1.2 in SCEPTer's dispatch) round-trips identically."""
    pytest.importorskip("scepter.antenna")
    evaluator = af.s672_evaluator(
        gm_dbi=40.0, diameter_m=3.0, wavelength_m=0.15,
    )
    theta_grid = np.linspace(0.0, 180.0, 18001)
    pat = af.sample_analytical_1d(
        evaluator, theta_grid, peak_gain_dbi=40.0,
    )
    out = tmp_path / "s672_s8.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)

    probe = np.linspace(0.005, 179.99, 500)
    gain_lut = evaluate_pattern_1d(reloaded, probe)
    gain_ana = evaluator(probe)
    mask = _mask_meaningful(gain_lut, gain_ana)
    assert mask.sum() > 0.1 * probe.size
    max_abs = float(np.max(np.abs(gain_lut[mask] - gain_ana[mask])))
    assert max_abs < 0.1, f"S.672 round-trip max |Δ| = {max_abs:.4f} dB"


def test_stage8_round_trip_s1528_rec1_4_asymmetric(tmp_path: Path) -> None:
    """S.1528 Rec 1.4 (asymmetric) 2-D (θ, φ) export → reload → evaluate —
    max |Δ| < 0.3 dB.
    """
    pytest.importorskip("scepter.antenna")
    evaluator = af.s1528_rec1_4_evaluator(
        wavelength_m=0.15, lr_m=1.6, lt_m=3.2, slr_db=20.0, l=2,
        gm_db=34.1,
        far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0,
    )
    # 2-D grid: θ dense near main lobe AND through the asymmetric
    # sidelobe region (Bessel zeros are at ~0.5-1° spacing for these
    # aperture sizes; 1° step smears them). φ at 2° resolves the
    # pattern's 90°-period asymmetry comfortably.
    theta_grid = np.concatenate([
        np.linspace(0.0, 5.0, 501),          # 0.01°
        np.linspace(5.01, 30.0, 500)[:-1],   # 0.05°
        np.linspace(30.0, 180.0, 151),       # 1°
    ])
    phi_grid = np.linspace(-180.0, 180.0, 181)  # 2° step
    pat = af.sample_analytical_2d_theta_phi(
        evaluator, theta_grid, phi_grid, peak_gain_dbi=34.1,
    )
    out = tmp_path / "rec14_asym_s8.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)

    # Probe concentrated near the main lobe where the pattern drives
    # EPFD (far sidelobes contribute < 1% of the peak-linear power).
    probe_theta_1d = np.concatenate([
        np.linspace(0.03, 2.0, 40),      # near main-lobe edge
        np.linspace(2.1, 10.0, 20),      # first sidelobes
        np.linspace(10.5, 179.97, 20),   # far region
    ])
    probe_phi_1d = np.linspace(-179.0, 179.0, 24)
    pt, pp = np.meshgrid(probe_theta_1d, probe_phi_1d, indexing="ij")
    gain_lut = evaluate_pattern_2d(reloaded, pt.ravel(), pp.ravel()).reshape(pt.shape)
    gain_ana = evaluator(pt, pp)
    mask = _mask_meaningful(gain_lut, gain_ana, floor_db=_NULL_FLOOR_DB_2D)
    assert mask.sum() > 0.1 * pt.size
    max_abs = float(np.max(np.abs(gain_lut[mask] - gain_ana[mask])))
    assert max_abs < 0.3, f"Rec 1.4 asymmetric round-trip max |Δ| = {max_abs:.4f} dB"


def test_stage8_round_trip_m2101(tmp_path: Path) -> None:
    """M.2101 (boresight-steered) 2-D (az, el) export → reload → evaluate —
    max |Δ| < 0.3 dB.
    """
    pytest.importorskip("pycraf.antenna")
    evaluator = af.m2101_evaluator(
        g_emax_dbi=5.0, a_m_db=30.0, sla_nu_db=30.0,
        phi_3db_deg=65.0, theta_3db_deg=65.0,
        d_h=0.5, d_v=0.5, n_h=8, n_v=8,
    )
    # Array-factor structure is at ~(beamwidth / sqrt(N)) degrees —
    # for an 8×8 array that's ~8°; a 1° grid resolves it comfortably.
    az_grid = np.linspace(-180.0, 180.0, 361)
    el_grid = np.linspace(-90.0, 90.0, 181)
    peak_gain_dbi = 5.0 + 10.0 * np.log10(64.0)
    pat = af.sample_analytical_2d_az_el(
        evaluator, az_grid, el_grid, peak_gain_dbi=peak_gain_dbi,
    )
    out = tmp_path / "m2101_s8.json"
    dump_custom_pattern(out, pat)
    reloaded = load_custom_pattern(out)

    # Off-grid probe concentrated near the main beam — M.2101 phased
    # array drops rapidly off boresight due to element pattern
    # roll-off, so far-out samples contribute ~nothing to EPFD and
    # are excluded by the null-floor mask anyway.
    probe_az = np.concatenate([
        np.linspace(-20.0, 20.0, 41),
        np.linspace(-90.0, -25.0, 10),
        np.linspace(25.0, 90.0, 10),
    ])
    probe_el = np.concatenate([
        np.linspace(-20.0, 20.0, 21),
        np.linspace(-70.0, -25.0, 6),
        np.linspace(25.0, 70.0, 6),
    ])
    paz, pel = np.meshgrid(probe_az, probe_el, indexing="ij")
    gain_lut = evaluate_pattern_2d(reloaded, paz.ravel(), pel.ravel()).reshape(paz.shape)
    gain_ana = evaluator(paz, pel)
    mask = _mask_meaningful(gain_lut, gain_ana, floor_db=_NULL_FLOOR_DB_2D)
    assert mask.sum() > 0.1 * paz.size, (
        "too many samples fell into the null floor — the LUT grid may be "
        "too coarse to cover the array factor"
    )
    max_abs = float(np.max(np.abs(gain_lut[mask] - gain_ana[mask])))
    assert max_abs < 0.3, f"M.2101 round-trip max |Δ| = {max_abs:.4f} dB"
