#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the Service & Demand surface-PFD cap.

Covers the K(β) LUT builder, session integration, and (later) the per-beam
and per-satellite-aggregate cap application paths.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth

from scepter import gpu_accel


CUDA_AVAILABLE = gpu_accel.cuda is not None and bool(gpu_accel.cuda.is_available())
GPU_REQUIRED = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")


def _earth_radius_m() -> float:
    return float(R_earth.to_value(u.m))


def _slant_range_m(orbit_radius_m: float, beta_deg: float, earth_r_m: float) -> float:
    """Slant range from a satellite at ``orbit_radius_m`` pointing at
    off-nadir angle ``beta_deg``, to the nearest Earth intersection."""
    beta_rad = math.radians(beta_deg)
    term = orbit_radius_m * math.cos(beta_rad)
    disc = term * term - (orbit_radius_m * orbit_radius_m - earth_r_m * earth_r_m)
    if disc < 0.0:
        return float("nan")
    return term - math.sqrt(disc)


def _psi_horizon_deg(orbit_radius_m: float, earth_r_m: float) -> float:
    return math.degrees(math.asin(earth_r_m / orbit_radius_m))


def _make_narrow_s1528(session):
    """Directive S.1528 pattern (~30 dBi gm), narrow main lobe."""
    wavelength_m = (12.0 * u.GHz).to(u.m, equivalencies=u.spectral()).to_value(u.m)
    return session.prepare_s1528_pattern_context(
        wavelength_m=wavelength_m,
        lt_m=1.6,
        lr_m=1.6,
        slr_db=20.0,
        l=2,
        far_sidelobe_start_deg=90.0,
        far_sidelobe_level_db=-20.0,
        gm_db=48.0,
    )


def _make_narrow_s1528_rec12(session):
    """Directive S.1528 Rec 1.2 piecewise pattern for coverage tests."""
    wavelength_m = (12.0 * u.GHz).to(u.m, equivalencies=u.spectral()).to_value(u.m)
    return session.prepare_s1528_rec12_pattern_context(
        wavelength_m=wavelength_m,
        gm_dbi=38.0,
        ln_db=-20.0,
        z=1.0,
        diameter_m=4.0,
    )


def _make_wide_s1528(session):
    """Low-directivity S.1528 pattern (~10 dBi gm), wide main lobe."""
    wavelength_m = (2.0 * u.GHz).to(u.m, equivalencies=u.spectral()).to_value(u.m)
    return session.prepare_s1528_pattern_context(
        wavelength_m=wavelength_m,
        lt_m=0.15,
        lr_m=0.15,
        slr_db=15.0,
        l=2,
        far_sidelobe_start_deg=90.0,
        far_sidelobe_level_db=-10.0,
        gm_db=10.0,
    )


@GPU_REQUIRED
def test_peak_pfd_lut_beta_zero_matches_analytic_nadir():
    """At β=0, the main-lobe peak is at ψ=0 (nadir); K(0) = 1/(4π·h²)."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3  # 550 km altitude
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.n_beta > 100
        assert lut_ctx.apply_atmosphere is False
        assert len(lut_ctx.shell_keys) == 1

        k_lut_host = lut_ctx.d_k_lut.get()
        assert k_lut_host.shape == (1, lut_ctx.n_beta)

        h = orbit_r_m - earth_r_m
        expected = 1.0 / (4.0 * math.pi * h * h)
        actual = float(k_lut_host[0, 0])
        assert actual == pytest.approx(expected, rel=1.0e-4)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_peak_pfd_lut_narrow_beam_follows_slant_range():
    """Narrow beam: K(β) ≈ 1/(4π·d_target(β)²) because the main lobe wins."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        k_lut_host = lut_ctx.d_k_lut.get()[0]

        # Check a range of β values well inside the horizon
        for beta_deg in (0.5, 5.0, 15.0, 30.0):
            d_m = _slant_range_m(orbit_r_m, beta_deg, earth_r_m)
            expected = 1.0 / (4.0 * math.pi * d_m * d_m)
            beta_idx = int(round(beta_deg / lut_ctx.beta_step_deg))
            actual = float(k_lut_host[beta_idx])
            # Narrow beam: main-lobe peak should match within ~0.1 dB
            ratio_db = 10.0 * math.log10(actual / expected)
            assert abs(ratio_db) < 0.1, (
                f"β={beta_deg}°: actual K={actual:.3e}, expected {expected:.3e} "
                f"(Δ={ratio_db:+.3f} dB)"
            )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_peak_pfd_lut_wide_beam_nadir_wins_at_large_beta():
    """Wide beam at large β: the sidelobe pointing near nadir wins.

    Specifically, for sufficiently small peak directivity, the max PFD on
    Earth occurs near nadir rather than at the main-lobe footprint, because
    the shorter slant range (h vs d_target) more than compensates for the
    reduced sidelobe gain.
    """
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_wide_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        k_lut_host = lut_ctx.d_k_lut.get()[0]

        h = orbit_r_m - earth_r_m
        main_lobe_only_at_60 = 1.0 / (
            4.0 * math.pi
            * _slant_range_m(orbit_r_m, 60.0, earth_r_m) ** 2
        )
        # LUT value must be strictly greater than the pure main-lobe
        # estimate (by construction the LUT includes the nadir sidelobe
        # contribution which outweighs the main lobe here).
        beta_idx = int(round(60.0 / lut_ctx.beta_step_deg))
        actual = float(k_lut_host[beta_idx])
        assert actual > main_lobe_only_at_60, (
            "Wide-beam LUT at β=60° should exceed the pure main-lobe estimate; "
            f"actual {actual:.3e} vs main_lobe {main_lobe_only_at_60:.3e}"
        )

        # And it must not exceed the nadir-per-watt-of-EIRP bound
        # ``1/(4π·h²)`` (the strictest possible upper bound: gain ≤ 1
        # with all pattern normalised by peak).
        nadir_upper_bound = 1.0 / (4.0 * math.pi * h * h)
        assert actual <= nadir_upper_bound * (1.0 + 1.0e-5), (
            f"K({60.0}°)={actual:.3e} exceeds theoretical upper bound "
            f"{nadir_upper_bound:.3e}"
        )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_peak_pfd_lut_is_monotonic_or_smooth():
    """LUT must be continuous (no giant discontinuities between cells)."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        k_lut_host = lut_ctx.d_k_lut.get()[0]
    session.close(reset_device=False)

    k_lut_host = np.asarray(k_lut_host, dtype=np.float64)
    # Work in dB and clip zero-valued tail (beyond horizon)
    valid = k_lut_host > 0.0
    k_db = 10.0 * np.log10(k_lut_host[valid])
    diff_db = np.diff(k_db)
    # Cell-to-cell jumps must not exceed ~2 dB.  Near the horizon the
    # slant range grows steeply so K(β) can move by ~0.5–1 dB per 0.01°
    # β cell legitimately; anything larger would indicate a real glitch.
    assert float(np.max(np.abs(diff_db))) < 2.0, (
        f"LUT has cell-to-cell jump of {float(np.max(np.abs(diff_db))):.3f} dB"
    )


@GPU_REQUIRED
def test_peak_pfd_lut_atmosphere_reduces_k():
    """Turning atmosphere on can only reduce K (never increase it)."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3

        lut_no_atm = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        k_no_atm = lut_no_atm.d_k_lut.get()[0]

        atm_ctx = session.prepare_atmosphere_lut_context(
            frequency_ghz=12.0,
            altitude_km_values=[0.0],
        )
        lut_with_atm = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=atm_ctx,
            target_alt_km=0.0,
        )
        k_with_atm = lut_with_atm.d_k_lut.get()[0]
    session.close(reset_device=False)

    # Pointwise: K_with_atm ≤ K_no_atm (+ small numerical slack)
    assert np.all(k_with_atm <= k_no_atm * (1.0 + 1.0e-5)), (
        "Atmosphere LUT made K larger somewhere — this should be impossible."
    )
    # And at large β (low elevation), atm should meaningfully reduce K
    beta_idx_large = int(round(60.0 / lut_with_atm.beta_step_deg))
    ratio_db = 10.0 * math.log10(
        float(k_with_atm[beta_idx_large]) / float(k_no_atm[beta_idx_large])
    )
    assert ratio_db < -0.001, (
        f"At β=60°, atm LUT should reduce K by at least 0.001 dB; got {ratio_db:+.4f} dB"
    )


@GPU_REQUIRED
def test_peak_pfd_lut_multi_shell_keyed_correctly():
    """Two satellites at different altitudes produce two distinct K rows."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_radii = np.array(
            [earth_r_m + 550.0e3, earth_r_m + 1200.0e3, earth_r_m + 550.0e3],
            dtype=np.float64,
        )
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        k = lut_ctx.d_k_lut.get()
        shell_ids = lut_ctx.d_shell_id_per_sat.get()
    session.close(reset_device=False)

    assert k.shape[0] == 2, f"Expected 2 unique shells, got {k.shape[0]}"
    # Sat 0 and sat 2 share a shell; sat 1 has its own
    assert shell_ids.shape == (3,)
    assert shell_ids[0] == shell_ids[2]
    assert shell_ids[0] != shell_ids[1]

    # K(β=0) at each shell equals 1/(4π·h²)
    for shell_idx, alt_m in ((shell_ids[0], 550.0e3), (shell_ids[1], 1200.0e3)):
        expected = 1.0 / (4.0 * math.pi * alt_m * alt_m)
        assert float(k[shell_idx, 0]) == pytest.approx(expected, rel=1.0e-4)


@GPU_REQUIRED
def test_peak_pfd_lut_rec12_beta_zero_matches_analytic_nadir():
    """LUT builder + lookup work for S.1528 Rec 1.2 patterns (K(0) = 1/(4π·h²))."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528_rec12(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r

        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.n_beta > 100
        k_lut_host = lut_ctx.d_k_lut.get()[0]
        expected = 1.0 / (4.0 * math.pi * h * h)
        assert float(k_lut_host[0]) == pytest.approx(expected, rel=1.0e-3)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_aggregate_cap_rec12_pattern_runs_and_matches_expectations():
    """Aggregate helper works with S.1528 Rec 1.2 patterns.

    Verifies the K-act=1, β=0 case against the analytic nadir peak so a
    regression in how Rec 1.2 is dispatched through
    ``_evaluate_normalised_pattern_cp`` is caught.
    """
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528_rec12(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r

        T, S, K = 1, 1, 1
        beam_alpha = np.zeros((T, S, K), dtype=np.float32)
        beam_beta = np.zeros((T, S, K), dtype=np.float32)
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak = np.full((T, S, K), 500.0, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        kwargs = _agg_helper_kwargs(
            session, ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
        )
        _, peak_pfd_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        peak = float(peak_pfd_cp.get()[0, 0])
    session.close(reset_device=False)

    expected = 500.0 / (4.0 * math.pi * h * h)
    assert peak == pytest.approx(expected, rel=1.0e-3)


@GPU_REQUIRED
def test_peak_pfd_k_lut_custom_1d_matches_native_rec12():
    """Stage 16: a Custom-1D K-LUT built from a Rec 1.2 evaluator
    matches the native Rec 1.2 K-LUT within K-LUT resampling noise.

    End-to-end K-LUT path: ``prepare_peak_pfd_lut_context`` →
    ``_build_peak_pfd_k_lut_cp`` → ``_evaluate_normalised_pattern_cp``
    → ``_evaluate_custom_1d_pattern_cp``. Every Stage-9 "NEEDS UPGRADE"
    consumer along this chain accepts the Custom-1D context.
    """
    import numpy as np
    from scepter import analytical_fixtures as af

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        # Native Rec 1.2 context.
        ctx_native = session.prepare_s1528_rec12_pattern_context(
            wavelength_m=0.15, gm_dbi=34.0, ln_db=-15.0, z=1.0, diameter_m=1.8,
        )
        # Custom-1D sampled densely from the same Rec 1.2 evaluator.
        evaluator = af.s1528_rec1_2_evaluator(
            gm_dbi=34.0, diameter_m=1.8, wavelength_m=0.15,
        )
        theta_grid = np.linspace(0.0, 180.0, 18001)
        pat = af.sample_analytical_1d(evaluator, theta_grid, peak_gain_dbi=34.0)
        ctx_custom = session.prepare_custom_pattern_1d_context(
            pattern=pat, wavelength_m=0.15,
        )

        orbit_radii = np.asarray([6_903_000.0])
        lut_native = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx_native,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        lut_custom = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx_custom,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
    session.close(reset_device=False)

    assert lut_native.is_2d is False
    assert lut_custom.is_2d is False
    assert lut_native.n_beta == lut_custom.n_beta

    k_native = lut_native.d_k_lut.get()
    k_custom = lut_custom.d_k_lut.get()
    assert k_native.shape == k_custom.shape

    # Relative comparison excluding the far-sidelobe floor where K is
    # tiny and relative errors blow up.
    peak = max(float(k_native.max()), float(k_custom.max()))
    mask = (k_native > peak * 1.0e-5) & (k_custom > peak * 1.0e-5)
    assert mask.sum() > 10
    rel = np.abs(k_custom[mask] - k_native[mask]) / np.maximum(k_native[mask], 1.0e-30)
    # Two independent LUT pipelines (analytical piecewise Rec 1.2 on
    # the native side; LUT→LUT resample on the custom side). 2%
    # relative agreement is tight — picks up any structural mistake
    # while absorbing legitimate float32 roundoff across the chain.
    assert float(np.max(rel)) < 2.0e-2, (
        f"Custom-1D K-LUT vs native Rec 1.2 K-LUT: max relative error = "
        f"{float(np.max(rel)):.3e}"
    )


@GPU_REQUIRED
def test_peak_pfd_k_lut_custom_2d_matches_native_asym_s1528():
    """Stage 17: a Custom-2D K-LUT built from an asymmetric S.1528
    Rec 1.4 evaluator matches the native asymmetric-S.1528 1-D K(β)
    LUT built from the same evaluator.

    Both pipelines are α-invariant 1-D K(β) tables (aperture rotates
    with the beam); they should therefore agree to within LUT-
    resample + float32 roundoff.
    """
    import numpy as np
    from scepter import analytical_fixtures as af

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        # Native asymmetric S.1528 Rec 1.4.
        ctx_native = session.prepare_s1528_pattern_context(
            wavelength_m=0.15, lt_m=3.2, lr_m=1.6, slr_db=20.0, l=2,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0, gm_db=34.0,
        )
        assert ctx_native.is_2d

        # Custom-2D sampled densely from the same Rec 1.4 evaluator.
        # Grid-aligned resample step keeps bilinear-of-bilinear exact.
        evaluator = af.s1528_rec1_4_evaluator(
            wavelength_m=0.15, lr_m=1.6, lt_m=3.2, slr_db=20.0, l=2,
            gm_db=34.0,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0,
        )
        theta_grid = np.linspace(0.0, 180.0, 18001)
        phi_grid = np.linspace(-180.0, 180.0, 361)
        pat = af.sample_analytical_2d_theta_phi(
            evaluator, theta_grid, phi_grid, peak_gain_dbi=34.0, phi_wraps=True,
        )
        ctx_custom = session.prepare_custom_pattern_2d_context(
            pattern=pat, wavelength_m=0.15,
            axis0_step_deg=0.01, axis1_step_deg=1.0,
        )

        orbit_radii = np.asarray([6_903_000.0])
        lut_native = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx_native,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        lut_custom = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx_custom,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
    session.close(reset_device=False)

    # Both must be 1-D (α-invariant).
    assert lut_native.is_2d is False
    assert lut_custom.is_2d is False
    assert lut_native.n_beta == lut_custom.n_beta

    k_native = lut_native.d_k_lut.get()
    k_custom = lut_custom.d_k_lut.get()
    peak = max(float(k_native.max()), float(k_custom.max()))
    mask = (k_native > peak * 1.0e-4) & (k_custom > peak * 1.0e-4)
    assert mask.sum() > 10
    rel = np.abs(k_custom[mask] - k_native[mask]) / np.maximum(k_native[mask], 1.0e-30)
    # Same 2-D (ψ, φ) observation sweep; only the pattern evaluator
    # differs (analytical Bessel/Taylor on the native side, bilinear
    # LUT on the custom side).  3% relative agreement is tight enough
    # to catch any dispatch error while absorbing LUT roundoff.
    assert float(np.max(rel)) < 3.0e-2, (
        f"Custom-2D K-LUT vs native asym S.1528 K-LUT: max relative = "
        f"{float(np.max(rel)):.3e}"
    )


@GPU_REQUIRED
def test_stage26_surface_pfd_cap_regression_custom_1d_and_2d():
    """Stage 26: surface-PFD cap regression. A Custom-1D pattern and a
    Custom-2D pattern, both sampled from the same analytical
    evaluator, produce surface-PFD K-LUTs that match the native
    analytical K-LUT within 3 % relative error.

    This is the correctness sibling of the (longer) benchmark
    harness ``benchmark_surface_pfd_cap.py`` — it doesn't measure
    throughput, but it does verify that the cap infrastructure (K-LUT
    build + lookup + cap-factor computation) accepts Custom contexts
    and produces physically equivalent output.
    """
    import numpy as np
    from scepter import analytical_fixtures as af

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        # --- 1-D leg: Rec 1.2 analytical vs Custom-1D of same curve. ---
        ctx_native_1d = session.prepare_s1528_rec12_pattern_context(
            wavelength_m=0.15, gm_dbi=34.0, ln_db=-15.0, z=1.0, diameter_m=1.8,
        )
        ev_1d = af.s1528_rec1_2_evaluator(
            gm_dbi=34.0, diameter_m=1.8, wavelength_m=0.15,
        )
        pat_1d = af.sample_analytical_1d(
            ev_1d, np.linspace(0.0, 180.0, 18001), peak_gain_dbi=34.0,
        )
        ctx_custom_1d = session.prepare_custom_pattern_1d_context(
            pattern=pat_1d, wavelength_m=0.15,
        )
        orbit_radii = np.asarray([6_903_000.0])
        lut_native_1d = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx_native_1d,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None, target_alt_km=0.0,
        )
        lut_custom_1d = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx_custom_1d,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None, target_alt_km=0.0,
        )
        k_native = lut_native_1d.d_k_lut.get()
        k_custom = lut_custom_1d.d_k_lut.get()
        peak = max(float(k_native.max()), float(k_custom.max()))
        mask = (k_native > peak * 1e-5) & (k_custom > peak * 1e-5)
        assert mask.sum() > 10
        rel_1d = np.abs(k_custom[mask] - k_native[mask]) / np.maximum(k_native[mask], 1e-30)
        assert float(np.max(rel_1d)) < 3.0e-2

        # --- 2-D leg: asym S.1528 Rec 1.4 analytical vs Custom-2D. ---
        ctx_native_2d = session.prepare_s1528_pattern_context(
            wavelength_m=0.15, lt_m=3.2, lr_m=1.6, slr_db=20.0, l=2,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0,
            gm_db=34.0,
        )
        ev_2d = af.s1528_rec1_4_evaluator(
            wavelength_m=0.15, lr_m=1.6, lt_m=3.2, slr_db=20.0, l=2,
            gm_db=34.0,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0,
        )
        pat_2d = af.sample_analytical_2d_theta_phi(
            ev_2d,
            np.linspace(0.0, 180.0, 18001),
            np.linspace(-180.0, 180.0, 361),
            peak_gain_dbi=34.0, phi_wraps=True,
        )
        ctx_custom_2d = session.prepare_custom_pattern_2d_context(
            pattern=pat_2d, wavelength_m=0.15,
            axis0_step_deg=0.01, axis1_step_deg=1.0,
        )
        lut_native_2d = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx_native_2d,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None, target_alt_km=0.0,
        )
        lut_custom_2d = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx_custom_2d,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None, target_alt_km=0.0,
        )
        k_native_2d = lut_native_2d.d_k_lut.get()
        k_custom_2d = lut_custom_2d.d_k_lut.get()
        peak_2d = max(float(k_native_2d.max()), float(k_custom_2d.max()))
        mask_2d = (k_native_2d > peak_2d * 1e-4) & (k_custom_2d > peak_2d * 1e-4)
        assert mask_2d.sum() > 10
        rel_2d = np.abs(k_custom_2d[mask_2d] - k_native_2d[mask_2d]) / np.maximum(
            k_native_2d[mask_2d], 1e-30
        )
        assert float(np.max(rel_2d)) < 3.0e-2
    session.close(reset_device=False)


@GPU_REQUIRED
def test_peak_pfd_k_lut_custom_2d_low_level_builder_direct():
    """Stage 17: the Custom-2D K-LUT builder accepts a Custom-2D
    context directly and produces a 1-D K(β) array.

    Complements the session-level test by pinning the low-level
    builder contract.
    """
    import numpy as np
    from scepter import analytical_fixtures as af

    evaluator = af.s1528_rec1_4_evaluator(
        wavelength_m=0.15, lr_m=1.6, lt_m=3.2, slr_db=20.0, l=2,
        gm_db=34.0,
    )
    pat = af.sample_analytical_2d_theta_phi(
        evaluator,
        theta_grid_deg=np.linspace(0.0, 180.0, 361),
        phi_grid_deg=np.linspace(-180.0, 180.0, 73),
        peak_gain_dbi=34.0,
    )
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_custom_pattern_2d_context(
            pattern=pat, wavelength_m=0.15,
        )
        k_cp, beta_step, beta_max, n_beta, psi_horizon = (
            gpu_accel._build_peak_pfd_k_lut_custom2d_cp(
                pattern_context=ctx,
                orbit_radius_m=6_903_000.0,
                earth_radius_m=float(_earth_radius_m()),
                atmosphere_lut_context=None,
                target_alt_km=0.0,
                beta_step_deg=0.5,   # coarser than default so the test is fast
                psi_step_deg=0.25,
                phi_step_deg=5.0,
            )
        )
        k_host = k_cp.get()
    session.close(reset_device=False)
    assert k_host.ndim == 1
    assert k_host.shape == (n_beta,)
    assert n_beta > 0
    assert np.all(np.isfinite(k_host)) and np.all(k_host >= 0.0)
    # K peaks at β=0 for a narrow main-lobe pattern.
    assert float(k_host[0]) == float(k_host.max())


@GPU_REQUIRED
def test_lookup_peak_pfd_k_any_routes_custom_2d_to_1d_lookup():
    """Stage 18: ``_lookup_peak_pfd_k_any_cp`` dispatches on
    ``lut_context.is_2d``. Custom-2D produces a 1-D K(β) LUT (Stage 17
    physics), so the runtime lookup picks ``_lookup_peak_pfd_k_cp`` —
    not the 2-D ``_lookup_peak_pfd_k_2d_cp``. Verifies that α is
    correctly ignored for the Custom-2D case (rotating a beam's
    steering azimuth must not change its K value, same as for
    asymmetric S.1528).
    """
    import cupy as cp
    import numpy as np
    from scepter import analytical_fixtures as af

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        evaluator = af.s1528_rec1_4_evaluator(
            wavelength_m=0.15, lr_m=1.6, lt_m=3.2, slr_db=20.0, l=2,
            gm_db=34.0,
        )
        pat = af.sample_analytical_2d_theta_phi(
            evaluator,
            theta_grid_deg=np.linspace(0.0, 180.0, 361),
            phi_grid_deg=np.linspace(-180.0, 180.0, 73),
            peak_gain_dbi=34.0,
        )
        ctx = session.prepare_custom_pattern_2d_context(
            pattern=pat, wavelength_m=0.15,
        )
        orbit_r = 6_903_000.0
        lut = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.asarray([orbit_r]),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut.is_2d is False  # Stage 17 produces 1-D K(β).

        # Query K at a fixed β for two wildly different α values — the
        # 1-D lookup must return identical K (α-invariance of the
        # aperture-rotates-with-beam physics).
        beta_rad = cp.full((4,), float(np.deg2rad(15.0)), dtype=cp.float32)
        alpha_rad = cp.asarray(
            [0.0, np.deg2rad(30.0), np.deg2rad(90.0), np.deg2rad(170.0)],
            dtype=cp.float32,
        )
        shell_ids = cp.zeros((4,), dtype=cp.int32)
        k_values = gpu_accel._lookup_peak_pfd_k_any_cp(
            lut, alpha_rad, beta_rad, shell_ids,
        ).get()
        assert np.all(k_values > 0)
        # All four must be identical to within float32 roundoff.
        assert np.max(np.abs(k_values - k_values[0])) < 1.0e-7 * float(k_values[0])


@GPU_REQUIRED
def test_peak_pfd_k_lut_custom_2d_builder_rejects_non_custom_2d():
    """Low-level Custom-2D builder rejects other context types with a
    clear error — guard against type confusion."""
    import numpy as np
    from scepter import analytical_fixtures as af

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        rec12_ctx = _make_narrow_s1528_rec12(session)
        with pytest.raises(TypeError, match="GpuCustomPattern2DContext"):
            gpu_accel._build_peak_pfd_k_lut_custom2d_cp(
                pattern_context=rec12_ctx,
                orbit_radius_m=6_903_000.0,
                earth_radius_m=float(_earth_radius_m()),
                atmosphere_lut_context=None,
                target_alt_km=0.0,
            )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_aggregate_cap_asymmetric_s1528_rec14_runs():
    """Aggregate helper works with an asymmetric S.1528 Rec 1.4 pattern.

    Stage 9b regression: before the (θ, φ) branch landed,
    ``_compute_aggregate_surface_pfd_cap_cp`` routed asymmetric
    contexts through ``_evaluate_normalised_pattern_cp`` which would
    raise ``NotImplementedError`` at the 1-D LUT evaluator's Stage-5
    guard. Now the asymmetric branch derives φ per (candidate, beam)
    pair and dispatches the 2-D (θ, φ) LUT. The K=1, β=0 main-lobe
    nadir case should still match the free-space ``EIRP / (4π h²)``
    reference, same as the Rec 1.2 test above.
    """
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_s1528_pattern_context(
            wavelength_m=0.15, lt_m=3.2, lr_m=1.6, slr_db=20.0, l=2,
            far_sidelobe_start_deg=90.0, far_sidelobe_level_db=-20.0, gm_db=34.1,
        )
        assert ctx.is_2d

        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r

        T, S, K = 1, 1, 1
        beam_alpha = np.zeros((T, S, K), dtype=np.float32)
        beam_beta = np.zeros((T, S, K), dtype=np.float32)
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak = np.full((T, S, K), 500.0, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        kwargs = _agg_helper_kwargs(
            session, ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
        )
        _, peak_pfd_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        peak = float(peak_pfd_cp.get()[0, 0])
    session.close(reset_device=False)

    expected = 500.0 / (4.0 * math.pi * h * h)
    # Asymmetric pattern at θ=0 still peaks at G_max (= 1.0 in normalised
    # form), so the aggregate cap peak reduces to the same free-space
    # PFD as the symmetric case.
    assert peak == pytest.approx(expected, rel=1.0e-3)


@GPU_REQUIRED
def test_per_beam_cap_rec12_pattern_full_pipeline():
    """End-to-end cap application with a Rec 1.2 pattern through the
    public ``accumulate_ras_power`` wrapper."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528_rec12(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        ptx_w = 10.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        gmax = 10.0 ** (float(ctx.gm_db) / 10.0)
        peak = (ptx_w * gmax) / (4.0 * math.pi * case["h"] * case["h"])

        result_base = _run_minimal_power(
            session, ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
        )
        cap_limit_dbw_channel = 10.0 * math.log10(0.5 * peak)
        result_capped = _run_minimal_power(
            session, ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
        )
    session.close(reset_device=False)

    pfd_base = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_capped = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    ratio = pfd_capped / pfd_base
    assert_allclose(ratio[pfd_base > 0.0], 0.5, rtol=1.0e-3)


def _make_m2101_8x8(session):
    """Moderate 8×8 phased-array pattern for M.2101 cap tests."""
    return session.prepare_m2101_pattern_context(
        g_emax_db=5.0,
        a_m_db=30.0,
        sla_nu_db=30.0,
        phi_3db_deg=65.0,
        theta_3db_deg=65.0,
        d_h=0.5,
        d_v=0.5,
        n_h=8,
        n_v=8,
        wavelength_m=0.025,
    )


@GPU_REQUIRED
def test_peak_pfd_lut_m2101_builds_2d_lut():
    """M.2101 pattern contexts should build a 2-D ``K(α, β)`` LUT with
    ``is_2d=True`` and a physically reasonable nadir value."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        m2101_ctx = _make_m2101_8x8(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=m2101_ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.is_2d is True
        assert lut_ctx.n_alpha > 1
        assert lut_ctx.n_beta > 1
        # d_k_lut shape: (n_shells=1, n_alpha, n_beta)
        assert tuple(lut_ctx.d_k_lut.shape) == (1, lut_ctx.n_alpha, lut_ctx.n_beta)

        k_lut_host = lut_ctx.d_k_lut.get()
        h = orbit_r_m - earth_r_m
        # With steering at (α=0, β=0) and an 8×8 array pointed broadside,
        # the peak observation direction is exactly nadir: the gain there
        # equals the array peak gain (normalised to 1) and slant range
        # equals ``h``.  So ``K(0, 0)`` must equal ``1/(4π·h²)`` to within
        # a few percent (the 2-D element LUT uses 0.5° bilinear interp).
        expected_nadir = 1.0 / (4.0 * math.pi * h * h)
        alpha_zero_idx = int(round((0.0 - lut_ctx.alpha_min_deg) / lut_ctx.alpha_step_deg))
        actual = float(k_lut_host[0, alpha_zero_idx, 0])
        ratio_db = 10.0 * math.log10(actual / expected_nadir)
        assert abs(ratio_db) < 0.5, (
            f"K(α=0, β=0)={actual:.3e} vs analytic nadir {expected_nadir:.3e} "
            f"(Δ={ratio_db:+.3f} dB)"
        )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_m2101_per_beam_cap_halves_pfd_at_nadir():
    """Full-pipeline per-beam cap with M.2101: cap limit at 50% of the
    analytic peak should halve the per-satellite PFD (β=0 → K=1/(4π h²))."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_m2101_8x8(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.is_2d is True

        # For a nadir-steered 8×8 array, the peak gain is
        # ``g_emax + 10·log10(n_h·n_v)`` — the hot path and the cap use
        # exactly that value for ``gmax_lin``.
        gm_db = float(tx_ctx.g_emax_db) + 10.0 * math.log10(
            float(tx_ctx.n_h) * float(tx_ctx.n_v)
        )
        gmax = 10.0 ** (gm_db / 10.0)
        ptx_w = 10.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        eirp_peak_w = ptx_w * gmax
        peak_pfd_expected = eirp_peak_w / (4.0 * math.pi * case["h"] * case["h"])

        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
        )
        cap_limit_w = 0.5 * peak_pfd_expected
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)
        result_capped = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
        )
    session.close(reset_device=False)

    pfd_base = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_capped = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    assert np.all(pfd_base > 0.0)
    ratio = pfd_capped / pfd_base
    # Allow a few percent tolerance for the 2-D LUT's bilinear interpolation.
    np.testing.assert_allclose(ratio[pfd_base > 0.0], 0.5, rtol=5.0e-3)


@GPU_REQUIRED
def test_m2101_per_satellite_aggregate_cap_full_pipeline():
    """Full-pipeline per-satellite aggregate cap with an M.2101 pattern
    and two coincident nadir beams: aggregate peak = 2 × single-beam
    peak, so a cap at 50% of that halves the output PFD."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_m2101_8x8(session)
        # Two nadir-steered beams on a single satellite.
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r
        case = {
            "sat_topo": np.zeros((1, 1, 3), dtype=np.float32),
            "sat_azel": np.zeros((1, 1, 2), dtype=np.float32),
            "beam_idx": np.zeros((1, 1, 2), dtype=np.int32),
            "beam_alpha_rad": np.zeros((1, 1, 2), dtype=np.float32),
            "beam_beta_rad": np.zeros((1, 1, 2), dtype=np.float32),
            "orbit_radius_m_per_sat": np.array([orbit_r], dtype=np.float32),
            "orbit_r": orbit_r,
            "earth_r": earth_r,
            "h": h,
        }
        case["sat_topo"][0, 0] = [0.0, 90.0, h / 1000.0]
        case["beam_idx"][0, 0] = np.arange(2, dtype=np.int32)

        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        gm_db = float(tx_ctx.g_emax_db) + 10.0 * math.log10(
            float(tx_ctx.n_h) * float(tx_ctx.n_v)
        )
        gmax = 10.0 ** (gm_db / 10.0)
        ptx_w = 5.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        per_beam_peak = (ptx_w * gmax) / (4.0 * math.pi * h * h)
        aggregate_peak = 2.0 * per_beam_peak
        cap_limit_w = 0.5 * aggregate_peak
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)

        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
        )
        result_capped = session.accumulate_ras_power(
            s1528_pattern_context=tx_ctx,
            ras_pattern_context=None,
            sat_topo=case["sat_topo"],
            sat_azel=case["sat_azel"],
            beam_idx=case["beam_idx"],
            beam_alpha_rad=case["beam_alpha_rad"],
            beam_beta_rad=case["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
            observer_alt_km=0.0,
            atmosphere_lut_context=None,
            bandwidth_mhz=1.0,
            power_input_quantity="satellite_ptx",
            pfd0_dbw_m2_mhz=None,
            satellite_ptx_dbw_channel=ptx_dbw_channel,
            target_alt_km=0.0,
            use_ras_station_alt_for_co=False,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
            surface_pfd_cap_mode="per_satellite",
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
    session.close(reset_device=False)

    pfd_base = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_capped = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    assert np.all(pfd_base > 0.0)
    ratio = pfd_capped / pfd_base
    np.testing.assert_allclose(ratio[pfd_base > 0.0], 0.5, rtol=1.0e-2)


@GPU_REQUIRED
def test_peak_pfd_lut_respects_pattern_eval_mode():
    """LUT built with 'analytical' vs 'lut' eval mode must agree within
    ~0.1 dB (the S.1528 LUT is built at 0.0001° resolution, so both paths
    should give nearly identical K values)."""
    import cupy as cp  # local import — GPU_REQUIRED already gates this

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    try:
        with session.activate():
            ctx = _make_narrow_s1528(session)
            earth_r_m = _earth_radius_m()
            orbit_r_m = earth_r_m + 550.0e3

            gpu_accel.set_pattern_eval_mode("lut")
            lut_ctx_fast = session.prepare_peak_pfd_lut_context(
                pattern_context=ctx,
                sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
                atmosphere_lut_context=None,
                target_alt_km=0.0,
            )
            k_fast = cp.asnumpy(lut_ctx_fast.d_k_lut[0])

            gpu_accel.set_pattern_eval_mode("analytical")
            # Drop the cache so the rebuild is visible
            session._peak_pfd_lut_context_cache.clear()
            lut_ctx_exact = session.prepare_peak_pfd_lut_context(
                pattern_context=ctx,
                sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
                atmosphere_lut_context=None,
                target_alt_km=0.0,
            )
            k_exact = cp.asnumpy(lut_ctx_exact.d_k_lut[0])
    finally:
        gpu_accel.set_pattern_eval_mode("lut")
        session.close(reset_device=False)

    valid = (k_fast > 0.0) & (k_exact > 0.0)
    diff_db = 10.0 * np.log10(k_fast[valid]) - 10.0 * np.log10(k_exact[valid])
    assert float(np.max(np.abs(diff_db))) < 0.1, (
        f"LUT/analytical mismatch: max Δ = {float(np.max(np.abs(diff_db))):.4f} dB"
    )


@GPU_REQUIRED
def test_peak_pfd_lookup_interpolation_matches_grid_values():
    """Exact β lookup at grid points should match the underlying LUT row."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        # Sample at exact grid points
        beta_degs = np.array([0.0, 1.0, 5.0, 10.0, 30.0, 40.0], dtype=np.float32)
        beta_rad_cp = cp.asarray(np.radians(beta_degs), dtype=cp.float32)
        shell_ids_cp = cp.zeros_like(beta_rad_cp, dtype=cp.int32)
        k_interp_cp = gpu_accel._lookup_peak_pfd_k_cp(lut_ctx, beta_rad_cp, shell_ids_cp)
        k_interp = cp.asnumpy(k_interp_cp)

        k_lut_host = cp.asnumpy(lut_ctx.d_k_lut[0])
        for beta_deg, k_got in zip(beta_degs.tolist(), k_interp.tolist()):
            beta_idx = int(round(beta_deg / lut_ctx.beta_step_deg))
            assert k_got == pytest.approx(float(k_lut_host[beta_idx]), rel=1.0e-5)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_peak_pfd_lookup_beta_beyond_grid_clamps():
    """β values beyond ``beta_max_deg`` must clamp gracefully, not error."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        huge_beta_cp = cp.asarray([math.radians(89.0)], dtype=cp.float32)
        shell_ids_cp = cp.zeros_like(huge_beta_cp, dtype=cp.int32)
        k_cp = gpu_accel._lookup_peak_pfd_k_cp(lut_ctx, huge_beta_cp, shell_ids_cp)
        val = float(cp.asnumpy(k_cp)[0])
    session.close(reset_device=False)

    # Must be finite and positive
    assert np.isfinite(val) and val > 0.0


# ---------------------------------------------------------------------------
#  Per-beam cap integration tests (hot-path injection in 3-D and 4-D paths)
# ---------------------------------------------------------------------------


def _minimal_3d_case(orbit_alt_m: float = 550.0e3) -> dict[str, np.ndarray]:
    """One-satellite, one-beam, single-timestep 3-D power-kernel fixture.

    Satellite sits directly above the RAS station, beam points at nadir
    (β=0), and the pattern LUT therefore returns ``K(0) = 1/(4π h²)``
    exactly.  All power modes can then be validated analytically.
    """
    earth_r = _earth_radius_m()
    orbit_r = earth_r + orbit_alt_m
    h_km = orbit_alt_m / 1000.0

    sat_topo = np.zeros((1, 1, 3), dtype=np.float32)
    # Sat is straight overhead: az=0, el=90°, range=h (km)
    sat_topo[0, 0] = [0.0, 90.0, h_km]
    sat_azel = np.zeros((1, 1, 2), dtype=np.float32)
    # RAS in sat frame also at (0, 0) ≡ nadir
    sat_azel[0, 0] = [0.0, 0.0]
    beam_idx = np.zeros((1, 1, 1), dtype=np.int32)  # one valid beam
    beam_alpha = np.zeros((1, 1, 1), dtype=np.float32)
    beam_beta = np.zeros((1, 1, 1), dtype=np.float32)  # β=0 (nadir)
    orbit_radius = np.array([orbit_r], dtype=np.float32)
    return {
        "sat_topo": sat_topo,
        "sat_azel": sat_azel,
        "beam_idx": beam_idx,
        "beam_alpha_rad": beam_alpha,
        "beam_beta_rad": beam_beta,
        "orbit_radius_m_per_sat": orbit_radius,
        "orbit_r": orbit_r,
        "earth_r": earth_r,
        "h": orbit_alt_m,
    }


def _run_minimal_power(
    session,
    tx_ctx,
    case,
    *,
    power_input_quantity: str,
    ptx_dbw_channel: float | None = None,
    eirp_dbw_channel: float | None = None,
    target_pfd_dbw_m2_channel: float | None = None,
    peak_pfd_lut_context=None,
    max_surface_pfd_dbw_m2_channel: float | None = None,
    surface_pfd_stats_enabled: bool = False,
) -> dict[str, np.ndarray]:
    return session.accumulate_ras_power(
        s1528_pattern_context=tx_ctx,
        ras_pattern_context=None,
        sat_topo=case["sat_topo"],
        sat_azel=case["sat_azel"],
        beam_idx=case["beam_idx"],
        beam_alpha_rad=case["beam_alpha_rad"],
        beam_beta_rad=case["beam_beta_rad"],
        telescope_azimuth_deg=None,
        telescope_elevation_deg=None,
        orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
        observer_alt_km=0.0,
        atmosphere_lut_context=None,
        bandwidth_mhz=1.0,
        power_input_quantity=power_input_quantity,
        pfd0_dbw_m2_mhz=None,  # disable the legacy default pfd0 fallback
        target_pfd_dbw_m2_channel=target_pfd_dbw_m2_channel,
        satellite_ptx_dbw_channel=ptx_dbw_channel,
        satellite_eirp_dbw_channel=eirp_dbw_channel,
        target_alt_km=0.0,
        use_ras_station_alt_for_co=False,
        peak_pfd_lut_context=peak_pfd_lut_context,
        max_surface_pfd_dbw_m2_channel=max_surface_pfd_dbw_m2_channel,
        surface_pfd_stats_enabled=surface_pfd_stats_enabled,
        include_epfd=False,
        include_prx_total=False,
        include_total_pfd=True,
        include_per_satellite_pfd=True,
        return_device=False,
    )


@GPU_REQUIRED
def test_per_beam_cap_satellite_ptx_halves_pfd_at_beta_zero():
    """satellite_ptx mode, β=0, cap at 50% of analytic peak → PFD halves."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        # Pick a nice Ptx
        ptx_w = 10.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        gmax = 10.0 ** (float(tx_ctx.gm_db) / 10.0)
        eirp_peak_w = ptx_w * gmax
        # K(0) = 1 / (4π h²)
        peak_pfd_expected = eirp_peak_w / (4.0 * math.pi * case["h"] * case["h"])

        # Baseline run (no cap at all)
        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
        )

        # Cap at exactly 50% of the analytic peak
        cap_limit_w = 0.5 * peak_pfd_expected
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)
        result_capped = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
        )
    session.close(reset_device=False)

    pfd_base = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_capped = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    assert np.all(pfd_base > 0.0)
    ratio = pfd_capped / pfd_base
    assert_allclose(ratio, 0.5, rtol=1.0e-4, atol=0.0)


@GPU_REQUIRED
def test_per_beam_cap_satellite_eirp_halves_pfd_at_beta_zero():
    """satellite_eirp mode — same half-PFD semantics as satellite_ptx."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        eirp_w = 40000.0  # 46 dBW, independent of pattern gmax
        eirp_dbw_channel = 10.0 * math.log10(eirp_w)
        # K(0) = 1/(4π h²)
        peak_pfd_expected = eirp_w / (4.0 * math.pi * case["h"] * case["h"])

        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_eirp",
            eirp_dbw_channel=eirp_dbw_channel,
        )
        cap_limit_w = 0.5 * peak_pfd_expected
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)
        result_capped = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_eirp",
            eirp_dbw_channel=eirp_dbw_channel,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
        )
    session.close(reset_device=False)

    pfd_base = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_capped = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    ratio = pfd_capped / pfd_base
    assert_allclose(ratio, 0.5, rtol=1.0e-4, atol=0.0)


@GPU_REQUIRED
def test_per_beam_cap_target_pfd_caps_delivered_pfd_at_beta_zero():
    """target_pfd mode — delivered PFD at the served cell is limited to
    ``limit / K(β)``.  For β=0 and K(0)=1/(4π h²), the relation simplifies
    to: with cap_factor=0.5, the output per-sat PFD halves."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        target_pfd_w = 1.0e-10   # dBW/m² = -100
        target_pfd_dbw_m2_channel = 10.0 * math.log10(target_pfd_w)

        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="target_pfd",
            target_pfd_dbw_m2_channel=target_pfd_dbw_m2_channel,
        )

        # For β=0, cap_factor maps linearly onto the PFD output.
        cap_limit_w = 0.5 * target_pfd_w
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)
        result_capped = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="target_pfd",
            target_pfd_dbw_m2_channel=target_pfd_dbw_m2_channel,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
        )
    session.close(reset_device=False)

    pfd_base = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_capped = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    ratio = pfd_capped / pfd_base
    assert_allclose(ratio, 0.5, rtol=1.0e-4, atol=0.0)


@GPU_REQUIRED
def test_per_beam_cap_loose_limit_matches_no_cap():
    """A cap at +300 dBW/m² (effectively infinite) must produce bitwise-
    identical outputs to a run without the cap kwargs."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        ptx_dbw_channel = 10.0  # 10 W
        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
        )
        result_loose = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=300.0,
        )
    session.close(reset_device=False)

    np.testing.assert_array_equal(
        result_base["PFD_per_sat_RAS_STATION_W_m2"],
        result_loose["PFD_per_sat_RAS_STATION_W_m2"],
    )


@GPU_REQUIRED
def test_per_beam_cap_stats_report_active_cap():
    """When the cap is strict enough to activate, reported stats must be
    non-zero and match the analytic cap depth in dB."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        ptx_w = 10.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        gmax = 10.0 ** (float(tx_ctx.gm_db) / 10.0)
        eirp_peak_w = ptx_w * gmax
        peak_pfd_expected = eirp_peak_w / (4.0 * math.pi * case["h"] * case["h"])

        # 10 dB below the unhinged peak ⇒ cap_factor = 0.1 ⇒ depth = 10 dB
        cap_limit_w = 0.1 * peak_pfd_expected
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)

        result_stats = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
            surface_pfd_stats_enabled=True,
        )
    session.close(reset_device=False)

    assert int(result_stats["surface_pfd_cap_n_beams_capped"]) == 1
    assert float(result_stats["surface_pfd_cap_mean_cap_db"]) == pytest.approx(10.0, abs=0.05)
    assert float(result_stats["surface_pfd_cap_max_cap_db"]) == pytest.approx(10.0, abs=0.05)


@GPU_REQUIRED
def test_per_beam_cap_stats_zero_when_cap_not_active():
    """A loose cap leaves stats at zero (no beams were clamped)."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        result = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=10.0,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=300.0,
            surface_pfd_stats_enabled=True,
        )
    session.close(reset_device=False)

    assert int(result["surface_pfd_cap_n_beams_capped"]) == 0
    assert float(result["surface_pfd_cap_mean_cap_db"]) == 0.0
    assert float(result["surface_pfd_cap_max_cap_db"]) == 0.0


@GPU_REQUIRED
def test_surface_pfd_cap_4d_aggregate_direct_call_without_precomputed_rejected():
    """Direct ``accumulate_ras_power`` calls in the 4-D path still reject
    ``per_satellite`` without a precomputed cap factor.

    The 4-D aggregate cap is supported through
    ``accumulate_direct_epfd_from_link_library`` (which hoists the cap
    factor out of the spectral-slab loop and passes it in via
    ``precomputed_cap_factor_cp``).  Direct kernel callers that bypass
    the fused wrapper need to either compute the cap factor externally
    or fall back to ``per_beam`` mode — the runner refuses to silently
    run without a precomputed factor in that case.
    """
    case = _boresight_avoidance_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=(15 * u.cm).to_value(u.m),
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_context,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        with pytest.raises(NotImplementedError, match=r"per_satellite.*4-D"):
            session.accumulate_ras_power(
                s1528_pattern_context=tx_context,
                ras_pattern_context=None,
                sat_topo=case["ras_topo_full"],
                sat_azel=case["ras_sat_azel_full"],
                beam_idx=beam_result["beam_idx"],
                beam_alpha_rad=beam_result["beam_alpha_rad"],
                beam_beta_rad=beam_result["beam_beta_rad"],
                telescope_azimuth_deg=None,
                telescope_elevation_deg=None,
                orbit_radius_m_per_sat=case["orbit_radius_m"],
                observer_alt_km=1.052,
                atmosphere_lut_context=None,
                pfd0_dbw_m2_mhz=None,
                power_input_quantity="satellite_ptx",
                satellite_ptx_dbw_channel=10.0,
                peak_pfd_lut_context=lut_ctx,
                max_surface_pfd_dbw_m2_channel=-100.0,
                surface_pfd_cap_mode="per_satellite",
                include_total_pfd=True,
                include_epfd=False,
                include_prx_total=False,
                return_device=False,
            )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_per_beam_cap_limit_without_lut_rejected():
    """Supplying a cap limit without a LUT context should raise."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        with pytest.raises(ValueError, match=r"peak_pfd_lut_context is None"):
            session.accumulate_ras_power(
                s1528_pattern_context=tx_ctx,
                ras_pattern_context=None,
                sat_topo=case["sat_topo"],
                sat_azel=case["sat_azel"],
                beam_idx=case["beam_idx"],
                beam_alpha_rad=case["beam_alpha_rad"],
                beam_beta_rad=case["beam_beta_rad"],
                telescope_azimuth_deg=None,
                telescope_elevation_deg=None,
                orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
                observer_alt_km=0.0,
                atmosphere_lut_context=None,
                bandwidth_mhz=1.0,
                power_input_quantity="satellite_ptx",
                satellite_ptx_dbw_channel=10.0,
                target_alt_km=0.0,
                use_ras_station_alt_for_co=False,
                max_surface_pfd_dbw_m2_channel=-100.0,
                include_total_pfd=True,
                include_epfd=False,
                include_prx_total=False,
                return_device=False,
            )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_per_beam_cap_lut_without_limit_rejected():
    """Supplying a LUT context without a limit should also raise."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        with pytest.raises(ValueError, match=r"max_surface_pfd_dbw_m2_\*"):
            session.accumulate_ras_power(
                s1528_pattern_context=tx_ctx,
                ras_pattern_context=None,
                sat_topo=case["sat_topo"],
                sat_azel=case["sat_azel"],
                beam_idx=case["beam_idx"],
                beam_alpha_rad=case["beam_alpha_rad"],
                beam_beta_rad=case["beam_beta_rad"],
                telescope_azimuth_deg=None,
                telescope_elevation_deg=None,
                orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
                observer_alt_km=0.0,
                atmosphere_lut_context=None,
                bandwidth_mhz=1.0,
                power_input_quantity="satellite_ptx",
                satellite_ptx_dbw_channel=10.0,
                target_alt_km=0.0,
                use_ras_station_alt_for_co=False,
                peak_pfd_lut_context=lut_ctx,
                include_total_pfd=True,
                include_epfd=False,
                include_prx_total=False,
                return_device=False,
            )
    session.close(reset_device=False)


def _assert_allclose_wrapper():
    """Provide ``assert_allclose`` from numpy.testing at module scope."""
    from numpy.testing import assert_allclose as _aa
    return _aa


assert_allclose = _assert_allclose_wrapper()


# ---------------------------------------------------------------------------
#  GUI state round-trip (no GPU required)
# ---------------------------------------------------------------------------


def test_service_config_surface_pfd_cap_round_trip():
    """ServiceConfig.to_json_dict → from_json_dict preserves cap fields."""
    try:
        from scepter.scepter_GUI import (
            ServiceConfig,
            _default_service_config,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GUI module not importable: {exc}")

    cfg = _default_service_config()
    cfg.max_surface_pfd_enabled = True
    cfg.max_surface_pfd_dbw_m2_mhz = -140.25
    cfg.max_surface_pfd_dbw_m2_channel = None
    cfg.surface_pfd_cap_mode = "per_satellite"

    payload = cfg.to_json_dict()
    assert payload["max_surface_pfd_enabled"] is True
    assert payload["max_surface_pfd_dbw_m2_mhz"] == pytest.approx(-140.25)
    assert payload["max_surface_pfd_dbw_m2_channel"] is None
    assert payload["surface_pfd_cap_mode"] == "per_satellite"

    restored = ServiceConfig.from_json_dict(payload)
    assert restored.max_surface_pfd_enabled is True
    assert restored.max_surface_pfd_dbw_m2_mhz == pytest.approx(-140.25)
    assert restored.max_surface_pfd_dbw_m2_channel is None
    assert restored.surface_pfd_cap_mode == "per_satellite"


def test_service_config_loads_legacy_payload_without_cap_fields():
    """Projects saved before the cap feature must load as 'cap disabled'."""
    try:
        from scepter.scepter_GUI import (
            ServiceConfig,
            _default_service_config,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GUI module not importable: {exc}")

    legacy_payload = _default_service_config().to_json_dict()
    # Simulate an older save: strip the new cap fields entirely.
    for key in (
        "max_surface_pfd_enabled",
        "max_surface_pfd_dbw_m2_mhz",
        "max_surface_pfd_dbw_m2_channel",
        "surface_pfd_cap_mode",
    ):
        legacy_payload.pop(key, None)

    loaded = ServiceConfig.from_json_dict(legacy_payload)
    assert loaded.max_surface_pfd_enabled is False
    assert loaded.max_surface_pfd_dbw_m2_mhz is None
    assert loaded.max_surface_pfd_dbw_m2_channel is None
    assert loaded.surface_pfd_cap_mode == "per_beam"


def test_runtime_config_surface_pfd_stats_round_trip():
    """RuntimeConfig.to_json_dict/from_json_dict preserves the stats flag."""
    try:
        from scepter.scepter_GUI import (
            RuntimeConfig,
            _default_runtime_config,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GUI module not importable: {exc}")

    rt = _default_runtime_config()
    assert rt.surface_pfd_stats_enabled is False  # default
    rt.surface_pfd_stats_enabled = True

    payload = rt.to_json_dict()
    assert payload["surface_pfd_stats_enabled"] is True
    restored = RuntimeConfig.from_json_dict(payload)
    assert restored.surface_pfd_stats_enabled is True


def test_surface_pfd_cap_stats_round_trip_through_hdf5(tmp_path):
    """End-to-end: push the three cap stat series through the writer
    and read them back to confirm the (n_batches,) time-series layout
    we picked is compatible with ``_write_iteration_batch_owned`` +
    ``read_data``.

    Guards the scenario-layer H5 export path against shape / dtype
    regressions in the writer.  No GPU required — we inject synthetic
    per-batch stat tensors directly via the public writer API.
    """
    from astropy.time import Time
    from scepter import scenario

    filename = tmp_path / "surface_pfd_cap_stats.h5"

    scenario.init_simulation_results(str(filename), write_mode="sync")

    # Three batches, one timestep each, 29 sats — reasonable shapes
    # that match what ``_compute_gpu_direct_epfd_batch_device`` writes
    # for a single-sky non-boresight run.
    for batch_idx, stats in enumerate([
        (0, 0.0, 0.0),       # no cap activation in batch 0
        (7, 4.25, 9.125),    # some cap activation in batch 1
        (29, 2.0, 3.5),      # stricter activation in batch 2
    ]):
        n_capped, mean_db, max_db = stats
        times = Time(
            60030.0 + np.asarray([float(batch_idx)]) / 86400.0,
            format="mjd", scale="utc",
        )
        scenario._write_iteration_batch_owned(
            str(filename),
            iteration=0,
            batch_items=(
                ("times", times),
                (
                    "surface_pfd_cap_n_beams_capped",
                    np.asarray([n_capped], dtype=np.int64),
                ),
                (
                    "surface_pfd_cap_mean_cap_db",
                    np.asarray([mean_db], dtype=np.float32),
                ),
                (
                    "surface_pfd_cap_max_cap_db",
                    np.asarray([max_db], dtype=np.float32),
                ),
            ),
        )

    scenario.flush_writes(str(filename))
    loaded = scenario.read_data(str(filename), stack=False)

    # ``read_data(stack=False)`` returns a top-level dict with an ``iter``
    # subgroup, which itself is a dict keyed by iteration index.  We only
    # wrote iteration 0.
    iter_group = loaded["iter"][0]
    assert "surface_pfd_cap_n_beams_capped" in iter_group
    assert "surface_pfd_cap_mean_cap_db" in iter_group
    assert "surface_pfd_cap_max_cap_db" in iter_group

    n_capped_series = np.asarray(iter_group["surface_pfd_cap_n_beams_capped"]).reshape(-1)
    mean_db_series = np.asarray(iter_group["surface_pfd_cap_mean_cap_db"]).reshape(-1)
    max_db_series = np.asarray(iter_group["surface_pfd_cap_max_cap_db"]).reshape(-1)

    assert n_capped_series.shape[0] == 3
    assert mean_db_series.shape[0] == 3
    assert max_db_series.shape[0] == 3

    assert n_capped_series.tolist() == [0, 7, 29]
    np.testing.assert_allclose(
        mean_db_series, [0.0, 4.25, 2.0], rtol=0.0, atol=1.0e-6,
    )
    np.testing.assert_allclose(
        max_db_series, [0.0, 9.125, 3.5], rtol=0.0, atol=1.0e-6,
    )


@GPU_REQUIRED
def test_power_result_carries_surface_pfd_cap_stats_when_enabled():
    """End-to-end check that cap stats land in the session result dict.

    Ensures the scenario-layer H5 export code path (which just forwards
    ``result["surface_pfd_cap_*"]`` into ``batch_payload``) has the
    keys it expects when ``surface_pfd_stats_enabled=True`` is set on
    the public ``accumulate_ras_power`` API.
    """
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        ptx_w = 10.0
        gmax = 10.0 ** (float(tx_ctx.gm_db) / 10.0)
        peak = (ptx_w * gmax) / (4.0 * math.pi * case["h"] * case["h"])
        cap_limit_dbw_channel = 10.0 * math.log10(0.1 * peak)  # 10 dB below peak

        result = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=10.0 * math.log10(ptx_w),
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
            surface_pfd_stats_enabled=True,
        )
    session.close(reset_device=False)

    # The three stat keys must be present with the expected dtypes
    # (int64 count, float32 dB scalars), packaged as 0-D numpy arrays.
    assert "surface_pfd_cap_n_beams_capped" in result
    assert "surface_pfd_cap_mean_cap_db" in result
    assert "surface_pfd_cap_max_cap_db" in result
    n_capped = np.asarray(result["surface_pfd_cap_n_beams_capped"])
    mean_db = np.asarray(result["surface_pfd_cap_mean_cap_db"])
    max_db = np.asarray(result["surface_pfd_cap_max_cap_db"])
    assert n_capped.dtype == np.int64
    assert mean_db.dtype == np.float32
    assert max_db.dtype == np.float32
    # Scalars shape (0-D ndarray is how ``_finalize_cap_stats`` packs them).
    assert n_capped.shape == ()
    assert mean_db.shape == ()
    assert max_db.shape == ()
    # The cap binds strictly here → exactly 1 beam capped at ~10 dB depth.
    assert int(n_capped) == 1
    assert float(mean_db) == pytest.approx(10.0, abs=0.05)


def test_runtime_config_loads_legacy_payload_without_stats_flag():
    """Projects saved before the reporting feature default to 'stats off'."""
    try:
        from scepter.scepter_GUI import (
            RuntimeConfig,
            _default_runtime_config,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GUI module not importable: {exc}")

    legacy = _default_runtime_config().to_json_dict()
    legacy.pop("surface_pfd_stats_enabled", None)
    loaded = RuntimeConfig.from_json_dict(legacy)
    assert loaded.surface_pfd_stats_enabled is False


def test_service_config_round_trip_preserves_other_fields():
    """Adding cap fields must not break unrelated fields' round-trip."""
    try:
        from scepter.scepter_GUI import (
            ServiceConfig,
            _default_service_config,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GUI module not importable: {exc}")

    cfg = _default_service_config()
    cfg.max_surface_pfd_enabled = True
    cfg.max_surface_pfd_dbw_m2_mhz = -130.0
    cfg.surface_pfd_cap_mode = "per_beam"

    restored = ServiceConfig.from_json_dict(cfg.to_json_dict())
    assert restored.nco == cfg.nco
    assert restored.nbeam == cfg.nbeam
    assert restored.selection_strategy == cfg.selection_strategy
    assert restored.bandwidth_mhz == cfg.bandwidth_mhz
    assert restored.power_input_quantity == cfg.power_input_quantity
    assert restored.power_input_basis == cfg.power_input_basis
    assert restored.target_pfd_dbw_m2_mhz == cfg.target_pfd_dbw_m2_mhz
    assert restored.power_variation_mode == cfg.power_variation_mode


# ---------------------------------------------------------------------------
#  4-D (boresight-avoidance) path coverage
# ---------------------------------------------------------------------------


def _boresight_avoidance_case() -> dict[str, np.ndarray]:
    """Minimal boresight-avoidance fixture shaped like the production path.

    Mirrors ``_boresight_direct_epfd_case`` from ``test_gpu_accel.py`` —
    two satellites, one sky cell, populated beam tables that drive
    ``_accumulate_ras_power_cp``'s 4-D branch.
    """
    earth_r = _earth_radius_m()
    orbit_radius = np.array(
        [earth_r + 525_000.0, earth_r + 530_000.0], dtype=np.float32
    )
    sat_topo = np.zeros((1, 2, 2, 2), dtype=np.float32)
    sat_topo[0, 0, :, 1] = np.array([80.0, 70.0], dtype=np.float32)
    sat_topo[0, 1, :, 1] = np.array([76.0, 75.0], dtype=np.float32)

    sat_azel = np.zeros((1, 2, 2, 2), dtype=np.float32)
    sat_azel[0, 0, :, 0] = np.array([10.0, 90.0], dtype=np.float32)
    sat_azel[0, 1, :, 0] = np.array([20.0, 100.0], dtype=np.float32)
    sat_azel[0, 0, :, 1] = np.array([5.0, 6.0], dtype=np.float32)
    sat_azel[0, 1, :, 1] = np.array([7.0, 8.0], dtype=np.float32)

    ras_topo = np.array([[[10.0, 50.0], [90.0, 50.0]]], dtype=np.float32)
    ras_sat_azel = np.array([[[10.0, 4.0], [90.0, 5.0]]], dtype=np.float32)
    ras_topo_full = np.array(
        [[[10.0, 50.0, 720.0, 0.0], [90.0, 50.0, 730.0, 0.0]]],
        dtype=np.float32,
    )
    ras_sat_azel_full = np.array(
        [[[10.0, 4.0, 0.0], [90.0, 5.0, 0.0]]],
        dtype=np.float32,
    )
    return {
        "sat_topo": sat_topo,
        "sat_azel": sat_azel,
        "ras_topo": ras_topo,
        "ras_sat_azel": ras_sat_azel,
        "ras_topo_full": ras_topo_full,
        "ras_sat_azel_full": ras_sat_azel_full,
        "pointing_az_deg": np.array([[10.0, 90.0]], dtype=np.float32),
        "pointing_el_deg": np.array([[50.0, 50.0]], dtype=np.float32),
        "sat_beta_max": np.array([20.0, 20.0], dtype=np.float32),
        "sat_belt_id": np.array([0, 1], dtype=np.int16),
        "orbit_radius_m": orbit_radius,
    }


@GPU_REQUIRED
def test_per_beam_cap_4d_boresight_avoidance_loose_no_op():
    """4-D boresight-avoidance path: loose cap must preserve the output."""
    case = _boresight_avoidance_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=(15 * u.cm).to_value(u.m),
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_context,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        kw = dict(
            s1528_pattern_context=tx_context,
            ras_pattern_context=None,
            sat_topo=case["ras_topo_full"],
            sat_azel=case["ras_sat_azel_full"],
            beam_idx=beam_result["beam_idx"],
            beam_alpha_rad=beam_result["beam_alpha_rad"],
            beam_beta_rad=beam_result["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m"],
            observer_alt_km=1.052,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=10.0,
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
        result_base = session.accumulate_ras_power(**kw)
        result_loose = session.accumulate_ras_power(
            **kw,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=300.0,
        )
    session.close(reset_device=False)

    np.testing.assert_array_equal(
        result_base["PFD_per_sat_RAS_STATION_W_m2"],
        result_loose["PFD_per_sat_RAS_STATION_W_m2"],
    )
    np.testing.assert_array_equal(
        result_base["PFD_total_RAS_STATION_W_m2"],
        result_loose["PFD_total_RAS_STATION_W_m2"],
    )


# ---------------------------------------------------------------------------
#  Aggregate cap helper unit tests (in-isolation, no hot power kernel)
# ---------------------------------------------------------------------------


def _agg_helper_kwargs(
    session,
    tx_ctx,
    *,
    beam_alpha_host,
    beam_beta_host,
    beam_valid_host,
    eirp_peak_host,
    orbit_radii_host,
    sat_axis_index,
    max_surface_pfd_lin,
    atmosphere_lut_context=None,
    target_alt_km: float = 0.0,
):
    import cupy as cp
    alpha_cp = cp.asarray(beam_alpha_host, dtype=cp.float32)
    beta_cp = cp.asarray(beam_beta_host, dtype=cp.float32)
    valid_cp = cp.asarray(beam_valid_host, dtype=cp.bool_)
    eirp_cp = cp.asarray(eirp_peak_host, dtype=cp.float32)
    orbit_cp = cp.asarray(orbit_radii_host, dtype=cp.float32)
    return dict(
        beam_alpha_cp=alpha_cp,
        beam_beta_cp=beta_cp,
        beam_valid_cp=valid_cp,
        eirp_peak_cp=eirp_cp,
        orbit_radius_per_sat_cp=orbit_cp,
        sat_axis_index=sat_axis_index,
        pattern_context=tx_ctx,
        atmosphere_lut_context=atmosphere_lut_context,
        target_alt_km=target_alt_km,
        max_surface_pfd_lin=max_surface_pfd_lin,
    )


@GPU_REQUIRED
def test_aggregate_cap_single_beam_matches_per_beam_analytic():
    """With one beam at β=0 and a narrow pattern, the aggregate cap
    reduces to the per-beam cap: peak PFD = EIRP × 1/(4π h²)."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r

        T, S, K = 1, 1, 1
        beam_alpha = np.zeros((T, S, K), dtype=np.float32)
        beam_beta = np.zeros((T, S, K), dtype=np.float32)
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak_w = 500.0  # arbitrary peak EIRP
        eirp_peak = np.full((T, S, K), eirp_peak_w, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        expected_peak_pfd = eirp_peak_w / (4.0 * math.pi * h * h)
        cap_limit = 0.5 * expected_peak_pfd  # should halve

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=cap_limit,
        )
        cap_factor_cp, peak_pfd_cp, stats = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        cap_factor = cap_factor_cp.get()
        peak_pfd = peak_pfd_cp.get()
    session.close(reset_device=False)

    assert cap_factor.shape == (1, 1)
    assert peak_pfd.shape == (1, 1)
    assert float(peak_pfd[0, 0]) == pytest.approx(expected_peak_pfd, rel=1.0e-3)
    assert float(cap_factor[0, 0]) == pytest.approx(0.5, rel=1.0e-3)
    assert int(stats["n_capped"]) == 1


@GPU_REQUIRED
def test_aggregate_cap_coincident_beams_scale_linearly():
    """K beams pointing at the same direction give K× the per-beam peak."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r

        T, S, K = 1, 1, 4
        beam_alpha = np.zeros((T, S, K), dtype=np.float32)
        beam_beta = np.zeros((T, S, K), dtype=np.float32)  # all at nadir
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak = np.full((T, S, K), 1000.0, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        single_peak = 1000.0 / (4.0 * math.pi * h * h)
        expected_peak = K * single_peak  # 4× because all beams add at the same point

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,  # no cap, only peak readback
        )
        _, peak_pfd_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        peak_pfd = peak_pfd_cp.get()
    session.close(reset_device=False)

    assert float(peak_pfd[0, 0]) == pytest.approx(expected_peak, rel=1.0e-3)


@GPU_REQUIRED
def test_aggregate_cap_well_separated_beams_do_not_interact():
    """Narrow beams pointing at disjoint footprints: aggregate peak is
    ~max over individual beam peaks, not their sum."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3

        T, S, K = 1, 1, 3
        beam_alpha = np.zeros((T, S, K), dtype=np.float32)
        beam_alpha[0, 0] = np.radians([0.0, 120.0, 240.0]).astype(np.float32)
        beam_beta = np.zeros((T, S, K), dtype=np.float32)
        # 20° off-nadir, three beams at different azimuths
        beam_beta[0, 0] = np.radians([20.0, 20.0, 20.0]).astype(np.float32)
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_per_beam = 100.0
        eirp_peak = np.full((T, S, K), eirp_per_beam, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
        )
        _, peak_pfd_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        peak_pfd_agg = float(peak_pfd_cp.get()[0, 0])

        # Single-beam peak for comparison
        kwargs_single = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha[:, :, :1],
            beam_beta_host=beam_beta[:, :, :1],
            beam_valid_host=beam_valid[:, :, :1],
            eirp_peak_host=eirp_peak[:, :, :1],
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
        )
        _, peak_pfd_single_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs_single)
        peak_pfd_single = float(peak_pfd_single_cp.get()[0, 0])
    session.close(reset_device=False)

    # Well-separated narrow beams: aggregate peak ≈ single-beam peak (within ~0.5 dB)
    ratio_db = 10.0 * math.log10(peak_pfd_agg / peak_pfd_single)
    assert abs(ratio_db) < 0.5, (
        f"Well-separated beams should not interact much; "
        f"aggregate/single ratio = {ratio_db:+.3f} dB (peak_agg={peak_pfd_agg:.3e}, "
        f"peak_single={peak_pfd_single:.3e})"
    )


@GPU_REQUIRED
def test_aggregate_cap_nadir_dominates_for_wide_pattern_off_nadir_beam():
    """Wide pattern + single beam at large β: the nadir sidelobe adds more
    PFD than the beam footprint."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_wide_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3

        T, S, K = 1, 1, 1
        beam_alpha = np.zeros((T, S, K), dtype=np.float32)
        beam_beta = np.radians([[[55.0]]]).astype(np.float32)  # 55° off-nadir
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak = np.full((T, S, K), 100.0, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
        )
        _, peak_pfd_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        peak_agg = float(peak_pfd_cp.get()[0, 0])

        # Main-lobe-only estimate at β=55°
        beta_rad = math.radians(55.0)
        term = orbit_r * math.cos(beta_rad)
        disc = term * term - (orbit_r * orbit_r - earth_r * earth_r)
        d_target = term - math.sqrt(disc)
        main_lobe_pfd = 100.0 / (4.0 * math.pi * d_target * d_target)
    session.close(reset_device=False)

    # Aggregate must exceed the main-lobe-only estimate (nadir sidelobe wins)
    assert peak_agg > main_lobe_pfd, (
        f"Wide pattern at 55° should have the nadir sidelobe dominating; "
        f"peak_agg={peak_agg:.3e} vs main_lobe_only={main_lobe_pfd:.3e}"
    )


def test_cap_disabled_zeroes_stored_thresholds_in_scenario_layer():
    """Regression for iteration 18: when the GUI sets
    ``max_surface_pfd_enabled=False`` but leaves the stored threshold
    value at e.g. ``-83.5 dBW/m²/MHz`` (the GUI preserves the last
    used threshold across toggle changes), the scenario layer
    must zero the threshold to ``None`` before it propagates into
    the power kernel.  Without this zero-out, the kernel raises
    ``ValueError: max_surface_pfd_dbw_m2_* limit supplied but
    peak_pfd_lut_context is None ...``.

    This test asserts the invariant by source inspection — the
    fix is a 3-line conditional at a stable location in
    ``scenario.py``.  Source inspection is slightly brittle but
    catches the most common reintroduction mode (someone
    refactoring the cap gate and removing the early zero-out).
    """
    from pathlib import Path
    from scepter import scenario as _scen
    src = Path(_scen.__file__).read_text(encoding="utf-8")
    # The fix block — look for the exact conditional form that
    # zeroes both thresholds when the cap is disabled.
    assert "if not bool(max_surface_pfd_enabled):" in src, (
        "Iteration-18 cap-disable zero-out conditional is missing "
        "from scenario.py — GUI-serialised threshold values may now "
        "leak into the power kernel when the cap toggle is off."
    )
    # Extract the block text and sanity-check both threshold kwargs
    # are set to None (not numeric zero, which would be a real cap).
    idx = src.index("if not bool(max_surface_pfd_enabled):")
    block = src[idx:idx + 400]
    assert "max_surface_pfd_dbw_m2_mhz = None" in block, (
        "Iteration-18 fix: per-MHz threshold must be reset to None, "
        "not dropped or left untouched."
    )
    assert "max_surface_pfd_dbw_m2_channel = None" in block, (
        "Iteration-18 fix: per-channel threshold must be reset to "
        "None, not dropped or left untouched."
    )
    # Belt-and-braces: "0" or "0.0" must not appear as the reset
    # value — that would be a valid (extremely loose) cap, not
    # 'disabled'.  See the user Q&A recorded in the memory file
    # (feedback_cupy_raw_kernel_contiguity / aggregate_cap_chunking).
    assert "max_surface_pfd_dbw_m2_mhz = 0" not in block
    assert "max_surface_pfd_dbw_m2_channel = 0" not in block


@GPU_REQUIRED
def test_aggregate_cap_chunking_bit_exact_vs_single_pass():
    """Regression test for iteration 19: the ``n_groups``-axis
    chunking path in :func:`_compute_aggregate_surface_pfd_cap_cp`
    must produce bit-exact ``cap_factor``, ``peak_pfd``, and stats
    compared to the single-pass path.

    Chunking is triggered automatically when
    ``n_groups × n_cand × K_act × 4 × 3`` exceeds
    ``chunk_memory_budget_bytes``.  Passing a tiny budget forces
    every group to its own chunk; the expected (single-pass)
    output is obtained with a huge budget.  Each group is
    physics-independent, so correctness requires **bit-exact**
    equality — any tolerance would hide a real bug.
    """
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3

        T, S, K = 3, 6, 4
        rng = np.random.default_rng(1234)
        beam_alpha = rng.uniform(-np.pi, np.pi, (T, S, K)).astype(np.float32)
        beam_beta = rng.uniform(0.0, np.radians(40.0), (T, S, K)).astype(np.float32)
        beam_valid = np.ones((T, S, K), dtype=bool)
        beam_valid[:, :, -1] = rng.random((T, S)) > 0.3
        # Large EIRP so the tight cap below actually triggers.
        eirp_peak = np.full((T, S, K), 1.0e6, dtype=np.float32)
        orbit_radii = np.full((S,), orbit_r, dtype=np.float32)

        kwargs_single = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e-8,  # tight → clamping active
        )
        cap_s_cp, peak_s_cp, stats_s = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            **kwargs_single,
            chunk_memory_budget_bytes=1_000_000_000,
        )
        cap_c_cp, peak_c_cp, stats_c = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            **kwargs_single,
            chunk_memory_budget_bytes=4_096,  # force fine-grained chunking
        )
        cap_s = cap_s_cp.get()
        peak_s = peak_s_cp.get()
        cap_c = cap_c_cp.get()
        peak_c = peak_c_cp.get()
    session.close(reset_device=False)

    np.testing.assert_array_equal(cap_s, cap_c)
    np.testing.assert_array_equal(peak_s, peak_c)
    assert stats_s == stats_c, (
        f"Chunked stats {stats_c} diverge from single-pass {stats_s} "
        "— iteration 19 chunking invariant broken."
    )
    # Sanity: the tight limit should actually have clamped a bunch
    # of groups so this test is not a trivial no-op.
    assert int(stats_s["n_capped"]) > 0


@GPU_REQUIRED
def test_aggregate_cap_loose_limit_no_cap():
    """Very loose limit → cap_factor == 1 for every group, stats are zero."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3

        T, S, K = 2, 3, 2
        rng = np.random.default_rng(42)
        beam_alpha = rng.uniform(-np.pi, np.pi, (T, S, K)).astype(np.float32)
        beam_beta = rng.uniform(0.0, np.radians(40.0), (T, S, K)).astype(np.float32)
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak = np.full((T, S, K), 10.0, dtype=np.float32)
        orbit_radii = np.full((S,), orbit_r, dtype=np.float32)

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,  # ridiculous limit
        )
        cap_factor_cp, _, stats = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        cap_factor = cap_factor_cp.get()
    session.close(reset_device=False)

    assert cap_factor.shape == (T, S)
    assert np.all(cap_factor == 1.0)
    assert int(stats["n_capped"]) == 0
    assert float(stats["cap_db_sum"]) == 0.0
    assert float(stats["cap_db_max"]) == 0.0


@GPU_REQUIRED
def test_aggregate_cap_invalid_beams_are_zeroed():
    """Beams marked invalid must not contribute to the aggregate sum."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r

        T, S, K = 1, 1, 3
        beam_alpha = np.zeros((T, S, K), dtype=np.float32)
        beam_beta = np.zeros((T, S, K), dtype=np.float32)  # all at nadir
        beam_valid = np.array([[[True, False, False]]], dtype=bool)
        eirp_peak = np.array([[[1000.0, 9999.0, 9999.0]]], dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
        )
        _, peak_pfd_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        peak = float(peak_pfd_cp.get()[0, 0])

        expected = 1000.0 / (4.0 * math.pi * h * h)
    session.close(reset_device=False)

    assert peak == pytest.approx(expected, rel=1.0e-3)


@GPU_REQUIRED
def test_aggregate_cap_4d_leading_shape_with_sky_axis():
    """Helper works with a (T, N_sky, S) leading shape and sat_axis=2."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r

        T, N_sky, S, K = 2, 3, 2, 1
        beam_alpha = np.zeros((T, N_sky, S, K), dtype=np.float32)
        beam_beta = np.zeros((T, N_sky, S, K), dtype=np.float32)
        beam_valid = np.ones((T, N_sky, S, K), dtype=bool)
        eirp_peak = np.full((T, N_sky, S, K), 500.0, dtype=np.float32)
        orbit_radii = np.full((S,), orbit_r, dtype=np.float32)

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=2,
            max_surface_pfd_lin=1.0e30,
        )
        cap_factor_cp, peak_pfd_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        cap_factor = cap_factor_cp.get()
        peak_pfd = peak_pfd_cp.get()
    session.close(reset_device=False)

    assert cap_factor.shape == (T, N_sky, S)
    assert peak_pfd.shape == (T, N_sky, S)
    expected_peak = 500.0 / (4.0 * math.pi * h * h)
    assert np.allclose(peak_pfd, expected_peak, rtol=1.0e-3)
    assert np.all(cap_factor == 1.0)


@GPU_REQUIRED
def test_aggregate_cap_atmosphere_reduces_peak():
    """Enabling atmosphere lowers the peak aggregate PFD at every group."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3

        T, S, K = 1, 1, 2
        rng = np.random.default_rng(0)
        beam_alpha = rng.uniform(-np.pi, np.pi, (T, S, K)).astype(np.float32)
        beam_beta = rng.uniform(np.radians(20.0), np.radians(50.0), (T, S, K)).astype(np.float32)
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak = np.full((T, S, K), 100.0, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        atm_ctx = session.prepare_atmosphere_lut_context(
            frequency_ghz=12.0,
            altitude_km_values=[0.0],
        )

        no_atm = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
        )
        with_atm = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
            atmosphere_lut_context=atm_ctx,
        )
        _, peak_no_atm_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**no_atm)
        _, peak_atm_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**with_atm)
        peak_no_atm = peak_no_atm_cp.get()
        peak_atm = peak_atm_cp.get()
    session.close(reset_device=False)

    assert np.all(peak_atm <= peak_no_atm * (1.0 + 1.0e-5))
    # At least one group shows meaningful reduction
    ratio = float(peak_atm[0, 0] / peak_no_atm[0, 0])
    assert ratio < 0.999


@GPU_REQUIRED
def test_aggregate_cap_helper_m2101_single_beam_nadir():
    """Aggregate helper with a single M.2101 beam pointed at nadir should
    compute the analytic peak PFD ``EIRP / (4π·h²)`` and derive a cap
    factor that cuts that peak exactly in half when the limit is 50%."""
    import cupy as cp
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        m2101_ctx = _make_m2101_8x8(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r
        alpha_cp = cp.zeros((1, 1, 1), dtype=cp.float32)
        beta_cp = cp.zeros((1, 1, 1), dtype=cp.float32)
        valid_cp = cp.ones((1, 1, 1), dtype=cp.bool_)
        eirp_w = 10.0
        eirp_cp = cp.full((1, 1, 1), eirp_w, dtype=cp.float32)
        orbit_cp = cp.asarray([orbit_r], dtype=cp.float32)

        # Uncapped — limit well above the analytic peak.
        _, peak_uncapped_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            beam_alpha_cp=alpha_cp,
            beam_beta_cp=beta_cp,
            beam_valid_cp=valid_cp,
            eirp_peak_cp=eirp_cp,
            orbit_radius_per_sat_cp=orbit_cp,
            sat_axis_index=1,
            pattern_context=m2101_ctx,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
            max_surface_pfd_lin=1.0e30,
        )
        expected_peak = eirp_w / (4.0 * math.pi * h * h)
        actual_peak = float(peak_uncapped_cp.get()[0, 0])
        ratio_db = 10.0 * math.log10(actual_peak / expected_peak)
        assert abs(ratio_db) < 0.5, (
            f"M.2101 nadir peak: {actual_peak:.3e} vs {expected_peak:.3e} "
            f"(Δ={ratio_db:+.3f} dB)"
        )

        # Capped at 50% → cap_factor == 0.5.
        cap_factor_cp, _, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            beam_alpha_cp=alpha_cp,
            beam_beta_cp=beta_cp,
            beam_valid_cp=valid_cp,
            eirp_peak_cp=eirp_cp,
            orbit_radius_per_sat_cp=orbit_cp,
            sat_axis_index=1,
            pattern_context=m2101_ctx,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
            max_surface_pfd_lin=0.5 * actual_peak,
        )
        assert float(cap_factor_cp.get()[0, 0]) == pytest.approx(0.5, abs=5.0e-3)
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
#  Aggregate cap full-pipeline integration tests (3-D path, end-to-end)
# ---------------------------------------------------------------------------


def _multi_beam_3d_case(n_beams: int = 2, orbit_alt_m: float = 550.0e3) -> dict[str, np.ndarray]:
    """Single-satellite, multi-beam 3-D fixture for aggregate cap tests."""
    earth_r = _earth_radius_m()
    orbit_r = earth_r + orbit_alt_m
    h_km = orbit_alt_m / 1000.0

    sat_topo = np.zeros((1, 1, 3), dtype=np.float32)
    sat_topo[0, 0] = [0.0, 90.0, h_km]
    sat_azel = np.zeros((1, 1, 2), dtype=np.float32)

    beam_idx = np.arange(n_beams, dtype=np.int32).reshape((1, 1, n_beams))
    beam_alpha = np.zeros((1, 1, n_beams), dtype=np.float32)
    beam_beta = np.zeros((1, 1, n_beams), dtype=np.float32)
    orbit_radius = np.array([orbit_r], dtype=np.float32)
    return {
        "sat_topo": sat_topo,
        "sat_azel": sat_azel,
        "beam_idx": beam_idx,
        "beam_alpha_rad": beam_alpha,
        "beam_beta_rad": beam_beta,
        "orbit_radius_m_per_sat": orbit_radius,
        "orbit_r": orbit_r,
        "earth_r": earth_r,
        "h": orbit_alt_m,
    }


@GPU_REQUIRED
def test_aggregate_cap_3d_ptx_mode_halves_output():
    """Full-pipeline: satellite_ptx mode, 2 coincident beams at nadir.

    Aggregate peak = 2 × per-beam peak; cap at 50% of that should halve
    the output PFD relative to no cap.
    """
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _multi_beam_3d_case(n_beams=2)
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        ptx_w = 5.0  # Ptx per beam
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        gmax = 10.0 ** (float(tx_ctx.gm_db) / 10.0)
        per_beam_peak = (ptx_w * gmax) / (4.0 * math.pi * case["h"] * case["h"])
        aggregate_peak = 2.0 * per_beam_peak
        cap_limit_w = 0.5 * aggregate_peak
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)

        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
        )
        result_capped = session.accumulate_ras_power(
            s1528_pattern_context=tx_ctx,
            ras_pattern_context=None,
            sat_topo=case["sat_topo"],
            sat_azel=case["sat_azel"],
            beam_idx=case["beam_idx"],
            beam_alpha_rad=case["beam_alpha_rad"],
            beam_beta_rad=case["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
            observer_alt_km=0.0,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            bandwidth_mhz=1.0,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=ptx_dbw_channel,
            target_alt_km=0.0,
            use_ras_station_alt_for_co=False,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
            surface_pfd_cap_mode="per_satellite",
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
    session.close(reset_device=False)

    pfd_base = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_capped = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    # Two coincident beams each produce the same PFD, so the base sum
    # is 2 × per_beam_peak × RAS_receive_scale.  The cap scales both
    # beams by 0.5, giving a 0.5× output PFD.
    ratio = pfd_capped / pfd_base
    assert np.allclose(ratio[pfd_base > 0.0], 0.5, rtol=1.0e-3)


@GPU_REQUIRED
def test_aggregate_cap_3d_stats_reflect_satellite_count():
    """Aggregate mode stats count distinct satellites, not beams."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _multi_beam_3d_case(n_beams=3)
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        ptx_w = 5.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        gmax = 10.0 ** (float(tx_ctx.gm_db) / 10.0)
        # 3 coincident beams → 3× per-beam peak; cap 10 dB below → 10 dB depth
        single_peak = (ptx_w * gmax) / (4.0 * math.pi * case["h"] * case["h"])
        agg_peak = 3.0 * single_peak
        cap_limit_w = 0.1 * agg_peak
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)

        result = session.accumulate_ras_power(
            s1528_pattern_context=tx_ctx,
            ras_pattern_context=None,
            sat_topo=case["sat_topo"],
            sat_azel=case["sat_azel"],
            beam_idx=case["beam_idx"],
            beam_alpha_rad=case["beam_alpha_rad"],
            beam_beta_rad=case["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
            observer_alt_km=0.0,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            bandwidth_mhz=1.0,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=ptx_dbw_channel,
            target_alt_km=0.0,
            use_ras_station_alt_for_co=False,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
            surface_pfd_cap_mode="per_satellite",
            surface_pfd_stats_enabled=True,
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
    session.close(reset_device=False)

    # Exactly one satellite is capped (one (t, sat) group), not 3 (one per beam)
    assert int(result["surface_pfd_cap_n_beams_capped"]) == 1
    assert float(result["surface_pfd_cap_mean_cap_db"]) == pytest.approx(10.0, abs=0.05)
    assert float(result["surface_pfd_cap_max_cap_db"]) == pytest.approx(10.0, abs=0.05)


@GPU_REQUIRED
def test_aggregate_cap_3d_loose_matches_no_cap():
    """Aggregate mode with a ridiculously loose limit == no cap."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _multi_beam_3d_case(n_beams=2)
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=10.0,
        )
        result_loose = session.accumulate_ras_power(
            s1528_pattern_context=tx_ctx,
            ras_pattern_context=None,
            sat_topo=case["sat_topo"],
            sat_azel=case["sat_azel"],
            beam_idx=case["beam_idx"],
            beam_alpha_rad=case["beam_alpha_rad"],
            beam_beta_rad=case["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
            observer_alt_km=0.0,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            bandwidth_mhz=1.0,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=10.0,
            target_alt_km=0.0,
            use_ras_station_alt_for_co=False,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=300.0,
            surface_pfd_cap_mode="per_satellite",
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
    session.close(reset_device=False)

    np.testing.assert_array_equal(
        result_base["PFD_per_sat_RAS_STATION_W_m2"],
        result_loose["PFD_per_sat_RAS_STATION_W_m2"],
    )


@GPU_REQUIRED
def test_aggregate_cap_3d_stricter_than_per_beam_when_beams_coincide():
    """Two coincident beams: aggregate cap binds where per-beam cap wouldn't.

    Per-beam cap at the single-beam peak produces no reduction; aggregate
    cap at the same limit cuts each beam to half.
    """
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _multi_beam_3d_case(n_beams=2)
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        ptx_w = 5.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        gmax = 10.0 ** (float(tx_ctx.gm_db) / 10.0)
        single_peak_pfd = (ptx_w * gmax) / (4.0 * math.pi * case["h"] * case["h"])
        # Cap at exactly the single-beam peak
        cap_limit_dbw_channel = 10.0 * math.log10(single_peak_pfd)

        common_kwargs = dict(
            s1528_pattern_context=tx_ctx,
            ras_pattern_context=None,
            sat_topo=case["sat_topo"],
            sat_azel=case["sat_azel"],
            beam_idx=case["beam_idx"],
            beam_alpha_rad=case["beam_alpha_rad"],
            beam_beta_rad=case["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
            observer_alt_km=0.0,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            bandwidth_mhz=1.0,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=ptx_dbw_channel,
            target_alt_km=0.0,
            use_ras_station_alt_for_co=False,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
        result_per_beam = session.accumulate_ras_power(
            **common_kwargs, surface_pfd_cap_mode="per_beam",
        )
        result_aggregate = session.accumulate_ras_power(
            **common_kwargs, surface_pfd_cap_mode="per_satellite",
        )
    session.close(reset_device=False)

    pfd_per_beam = np.asarray(result_per_beam["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_aggregate = np.asarray(result_aggregate["PFD_per_sat_RAS_STATION_W_m2"])
    # Per-beam cap: each beam's individual peak equals the limit exactly
    # → cap_factor = 1 → no reduction at all.
    # Aggregate cap: two coincident beams produce 2× the limit → cap_factor
    # = 0.5 → output halved.
    ratio = pfd_aggregate / pfd_per_beam
    assert np.allclose(ratio[pfd_per_beam > 0.0], 0.5, rtol=1.0e-3)


@GPU_REQUIRED
def test_precomputed_cap_factor_matches_in_kernel_computation():
    """Passing ``precomputed_cap_factor_cp`` into ``accumulate_ras_power``
    must produce the same output as the in-kernel aggregate cap path.

    This guards the fused-path hoist optimisation in
    ``_accumulate_direct_epfd_from_link_library_cp``, which computes the
    cap factor once per batch and forwards it to every spectral-slab call.
    """
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _multi_beam_3d_case(n_beams=2)
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        ptx_w = 5.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        gmax = 10.0 ** (float(tx_ctx.gm_db) / 10.0)
        per_beam_peak = (ptx_w * gmax) / (4.0 * math.pi * case["h"] * case["h"])
        aggregate_peak = 2.0 * per_beam_peak
        cap_limit_w = 0.5 * aggregate_peak
        cap_limit_dbw_channel = 10.0 * math.log10(cap_limit_w)

        common_kwargs = dict(
            s1528_pattern_context=tx_ctx,
            ras_pattern_context=None,
            sat_topo=case["sat_topo"],
            sat_azel=case["sat_azel"],
            beam_idx=case["beam_idx"],
            beam_alpha_rad=case["beam_alpha_rad"],
            beam_beta_rad=case["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
            observer_alt_km=0.0,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            bandwidth_mhz=1.0,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=ptx_dbw_channel,
            target_alt_km=0.0,
            use_ras_station_alt_for_co=False,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
            surface_pfd_cap_mode="per_satellite",
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
        result_in_kernel = session.accumulate_ras_power(**common_kwargs)

        eirp_peak_tensor = cp.full(
            case["beam_alpha_rad"].shape,
            cp.float32(ptx_w * gmax),
            dtype=cp.float32,
        )
        valid_beam_cp = cp.asarray(case["beam_idx"] >= 0, dtype=cp.bool_)
        cap_factor_cp, _, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            beam_alpha_cp=cp.asarray(case["beam_alpha_rad"], dtype=cp.float32),
            beam_beta_cp=cp.asarray(case["beam_beta_rad"], dtype=cp.float32),
            beam_valid_cp=valid_beam_cp,
            eirp_peak_cp=eirp_peak_tensor,
            orbit_radius_per_sat_cp=cp.asarray(
                case["orbit_radius_m_per_sat"], dtype=cp.float32
            ),
            sat_axis_index=1,
            pattern_context=tx_ctx,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
            max_surface_pfd_lin=cap_limit_w,
        )
        result_precomputed = session.accumulate_ras_power(
            **common_kwargs,
            precomputed_cap_factor_cp=cap_factor_cp,
        )
    session.close(reset_device=False)

    np.testing.assert_array_equal(
        result_in_kernel["PFD_per_sat_RAS_STATION_W_m2"],
        result_precomputed["PFD_per_sat_RAS_STATION_W_m2"],
    )
    np.testing.assert_array_equal(
        result_in_kernel["PFD_total_RAS_STATION_W_m2"],
        result_precomputed["PFD_total_RAS_STATION_W_m2"],
    )


@GPU_REQUIRED
def test_precomputed_per_beam_cap_factor_matches_in_kernel_computation():
    """Per-beam hoist: a precomputed ``(T, S, K)`` cap factor must produce
    bitwise-identical output to the in-kernel per-beam path.

    Guards the per-beam hoist inside
    ``_accumulate_direct_epfd_from_link_library_cp`` which computes the
    per-beam cap factor once per batch and forwards it to each spectral
    slab call via the ``precomputed_cap_factor_cp`` kwarg.
    """
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        ptx_w = 10.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)
        gmax = 10.0 ** (float(tx_ctx.gm_db) / 10.0)
        per_beam_peak = (ptx_w * gmax) / (4.0 * math.pi * case["h"] * case["h"])
        cap_limit_dbw_channel = 10.0 * math.log10(0.5 * per_beam_peak)

        common_kwargs = dict(
            s1528_pattern_context=tx_ctx,
            ras_pattern_context=None,
            sat_topo=case["sat_topo"],
            sat_azel=case["sat_azel"],
            beam_idx=case["beam_idx"],
            beam_alpha_rad=case["beam_alpha_rad"],
            beam_beta_rad=case["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
            observer_alt_km=0.0,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            bandwidth_mhz=1.0,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=ptx_dbw_channel,
            target_alt_km=0.0,
            use_ras_station_alt_for_co=False,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=cap_limit_dbw_channel,
            surface_pfd_cap_mode="per_beam",
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
        result_in_kernel = session.accumulate_ras_power(**common_kwargs)

        # Reproduce what the hoist does: compute the (T, S, K) cap
        # factor tensor externally and feed it in via the new
        # per-beam ``precomputed_cap_factor_cp`` path.
        eirp_peak_w = ptx_w * gmax  # scalar because satellite_ptx + no var power
        beam_beta_cp = cp.asarray(case["beam_beta_rad"], dtype=cp.float32)
        shell_per_beam_cp = cp.broadcast_to(
            cp.zeros((int(case["sat_topo"].shape[1]),), dtype=cp.int32)[None, :, None],
            beam_beta_cp.shape,
        )
        k_per_beam_cp = gpu_accel._lookup_peak_pfd_k_cp(
            lut_ctx, beam_beta_cp, shell_per_beam_cp,
        )
        peak_pfd = cp.float32(eirp_peak_w) * k_per_beam_cp
        cap_factor = cp.minimum(
            cp.float32(1.0),
            cp.float32(0.5 * per_beam_peak)
            / cp.maximum(peak_pfd, cp.float32(1.0e-30)),
        )
        valid_mask = cp.asarray(case["beam_idx"] >= 0, dtype=cp.bool_)
        cap_factor = cp.where(valid_mask, cap_factor, cp.float32(1.0))

        result_precomputed = session.accumulate_ras_power(
            **common_kwargs,
            precomputed_cap_factor_cp=cap_factor,
        )
    session.close(reset_device=False)

    np.testing.assert_array_equal(
        result_in_kernel["PFD_per_sat_RAS_STATION_W_m2"],
        result_precomputed["PFD_per_sat_RAS_STATION_W_m2"],
    )
    np.testing.assert_array_equal(
        result_in_kernel["PFD_total_RAS_STATION_W_m2"],
        result_precomputed["PFD_total_RAS_STATION_W_m2"],
    )


@GPU_REQUIRED
def test_precomputed_cap_factor_shape_check():
    """``precomputed_cap_factor_cp`` with wrong shape must raise a clear error."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        case = _minimal_3d_case()
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        bad_cap = cp.ones((2, 3), dtype=cp.float32)
        with pytest.raises(ValueError, match=r"precomputed_cap_factor_cp"):
            session.accumulate_ras_power(
                s1528_pattern_context=tx_ctx,
                ras_pattern_context=None,
                sat_topo=case["sat_topo"],
                sat_azel=case["sat_azel"],
                beam_idx=case["beam_idx"],
                beam_alpha_rad=case["beam_alpha_rad"],
                beam_beta_rad=case["beam_beta_rad"],
                telescope_azimuth_deg=None,
                telescope_elevation_deg=None,
                orbit_radius_m_per_sat=case["orbit_radius_m_per_sat"],
                observer_alt_km=0.0,
                atmosphere_lut_context=None,
                pfd0_dbw_m2_mhz=None,
                bandwidth_mhz=1.0,
                power_input_quantity="satellite_ptx",
                satellite_ptx_dbw_channel=10.0,
                target_alt_km=0.0,
                use_ras_station_alt_for_co=False,
                peak_pfd_lut_context=lut_ctx,
                max_surface_pfd_dbw_m2_channel=-100.0,
                surface_pfd_cap_mode="per_satellite",
                precomputed_cap_factor_cp=bad_cap,
                include_total_pfd=True,
                include_epfd=False,
                include_prx_total=False,
                return_device=False,
            )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_aggregate_cap_large_k_skips_pair_midpoints():
    """K_act > 32 must drop pair midpoints and still return a valid cap.

    Codifies the cutoff introduced to keep the benchmark's ``nbeam=100``
    configuration (a 5151-candidate full set) from blowing the transient
    memory budget.  The reduced candidate set is physically exact when
    no pair of beams has overlapping main lobes.
    """
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3

        T, S, K = 1, 1, 50
        rng = np.random.default_rng(0)
        beam_alpha = rng.uniform(-np.pi, np.pi, (T, S, K)).astype(np.float32)
        beam_beta = rng.uniform(np.radians(2.0), np.radians(20.0), (T, S, K)).astype(np.float32)
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak = np.full((T, S, K), 10.0, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=1.0e30,
        )
        cap_factor_cp, peak_pfd_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
        peak = float(peak_pfd_cp.get()[0, 0])
        cap = float(cap_factor_cp.get()[0, 0])
    session.close(reset_device=False)

    assert cap == 1.0
    assert 0.0 < peak < float("inf")


@GPU_REQUIRED
def test_aggregate_cap_stats_match_analytic_depth():
    """When the cap binds 10 dB below the analytic peak, mean_cap_db ≈ 10."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_narrow_s1528(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r

        T, S, K = 1, 1, 1
        beam_alpha = np.zeros((T, S, K), dtype=np.float32)
        beam_beta = np.zeros((T, S, K), dtype=np.float32)  # nadir
        beam_valid = np.ones((T, S, K), dtype=bool)
        eirp_peak_w = 1000.0
        eirp_peak = np.full((T, S, K), eirp_peak_w, dtype=np.float32)
        orbit_radii = np.array([orbit_r], dtype=np.float32)

        expected_peak = eirp_peak_w / (4.0 * math.pi * h * h)
        cap_limit = 0.1 * expected_peak  # 10 dB below

        kwargs = _agg_helper_kwargs(
            session, tx_ctx,
            beam_alpha_host=beam_alpha,
            beam_beta_host=beam_beta,
            beam_valid_host=beam_valid,
            eirp_peak_host=eirp_peak,
            orbit_radii_host=orbit_radii,
            sat_axis_index=1,
            max_surface_pfd_lin=cap_limit,
        )
        _, _, stats = gpu_accel._compute_aggregate_surface_pfd_cap_cp(**kwargs)
    session.close(reset_device=False)

    assert int(stats["n_capped"]) == 1
    assert float(stats["cap_db_sum"]) == pytest.approx(10.0, abs=0.05)
    assert float(stats["cap_db_max"]) == pytest.approx(10.0, abs=0.05)


@GPU_REQUIRED
def test_4d_aggregate_cap_accepts_precomputed_factor_via_direct_call():
    """Passing a ``(T, N_sky, S)`` per-satellite cap factor into the
    4-D power kernel via ``precomputed_cap_factor_cp`` must work, with
    the factor gathered at ``(active_t, active_sky, active_sat)`` and
    applied to every active beam of the matching satellite.
    """
    import cupy as cp

    case = _boresight_avoidance_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=(15 * u.cm).to_value(u.m),
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_context,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        beam_idx_host = np.asarray(beam_result["beam_idx"])
        T, N_sky, S, _K = beam_idx_host.shape
        # Use a uniform 0.5 cap factor on every (t, sky, sat) → every
        # active beam's emitted EIRP should scale by exactly 0.5.
        cap_factor_cp = cp.full((int(T), int(N_sky), int(S)), 0.5, dtype=cp.float32)

        base_kw = dict(
            s1528_pattern_context=tx_context,
            ras_pattern_context=None,
            sat_topo=case["ras_topo_full"],
            sat_azel=case["ras_sat_azel_full"],
            beam_idx=beam_result["beam_idx"],
            beam_alpha_rad=beam_result["beam_alpha_rad"],
            beam_beta_rad=beam_result["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m"],
            observer_alt_km=1.052,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=10.0,
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
        result_base = session.accumulate_ras_power(**base_kw)
        result_capped = session.accumulate_ras_power(
            **base_kw,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=-300.0,  # value ignored — precomputed wins
            surface_pfd_cap_mode="per_satellite",
            precomputed_cap_factor_cp=cap_factor_cp,
        )
    session.close(reset_device=False)

    base_pfd = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    capped_pfd = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    positive = base_pfd > 0.0
    np.testing.assert_allclose(
        capped_pfd[positive], 0.5 * base_pfd[positive], rtol=1.0e-5,
    )


@GPU_REQUIRED
def test_4d_aggregate_cap_accepts_precomputed_per_beam_factor():
    """Per-beam shape ``(T, N_sky, S, K)`` is also accepted and gathered
    at the active indices."""
    import cupy as cp

    case = _boresight_avoidance_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=(15 * u.cm).to_value(u.m),
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_context,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )

        beam_idx_host = np.asarray(beam_result["beam_idx"])
        T, N_sky, S, K = beam_idx_host.shape
        cap_factor_cp = cp.full(
            (int(T), int(N_sky), int(S), int(K)), 0.25, dtype=cp.float32,
        )

        base_kw = dict(
            s1528_pattern_context=tx_context,
            ras_pattern_context=None,
            sat_topo=case["ras_topo_full"],
            sat_azel=case["ras_sat_azel_full"],
            beam_idx=beam_result["beam_idx"],
            beam_alpha_rad=beam_result["beam_alpha_rad"],
            beam_beta_rad=beam_result["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m"],
            observer_alt_km=1.052,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=10.0,
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
        result_base = session.accumulate_ras_power(**base_kw)
        result_capped = session.accumulate_ras_power(
            **base_kw,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=-300.0,
            surface_pfd_cap_mode="per_beam",
            precomputed_cap_factor_cp=cap_factor_cp,
        )
    session.close(reset_device=False)

    base_pfd = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    capped_pfd = np.asarray(result_capped["PFD_per_sat_RAS_STATION_W_m2"])
    positive = base_pfd > 0.0
    np.testing.assert_allclose(
        capped_pfd[positive], 0.25 * base_pfd[positive], rtol=1.0e-5,
    )


@GPU_REQUIRED
def test_per_beam_cap_4d_boresight_avoidance_strict_scales_output():
    """4-D path: strict cap must monotonically reduce per-sat PFD output
    and leave the stats counters populated."""
    case = _boresight_avoidance_case()
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False)
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        tx_context = session.prepare_s1528_pattern_context(
            wavelength_m=(15 * u.cm).to_value(u.m),
            lt_m=1.6 * u.m,
            lr_m=1.6 * u.m,
            slr_db=20.0,
            l=2,
            far_sidelobe_start_deg=90 * u.deg,
            far_sidelobe_level_db=-20.0,
        )
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_context,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        base_kw = dict(
            s1528_pattern_context=tx_context,
            ras_pattern_context=None,
            sat_topo=case["ras_topo_full"],
            sat_azel=case["ras_sat_azel_full"],
            beam_idx=beam_result["beam_idx"],
            beam_alpha_rad=beam_result["beam_alpha_rad"],
            beam_beta_rad=beam_result["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m"],
            observer_alt_km=1.052,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=10.0,
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
        result_base = session.accumulate_ras_power(**base_kw)
        result_strict = session.accumulate_ras_power(
            **base_kw,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=-300.0,  # essentially cap everything
            surface_pfd_stats_enabled=True,
        )
    session.close(reset_device=False)

    base_pfd = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    strict_pfd = np.asarray(result_strict["PFD_per_sat_RAS_STATION_W_m2"])
    # Strict cap must not increase any entry
    assert np.all(strict_pfd <= base_pfd + 1.0e-30)
    # And where the baseline was positive, the strict path must be
    # strictly smaller than baseline (cap binding everywhere).
    positive_mask = base_pfd > 0.0
    assert np.all(strict_pfd[positive_mask] < base_pfd[positive_mask])

    # Stats counters must be populated
    assert int(result_strict["surface_pfd_cap_n_beams_capped"]) >= 1
    assert float(result_strict["surface_pfd_cap_max_cap_db"]) > 0.0


# ===========================================================================
#  Enhanced atomic & end-to-end tests for the 2026-Q2 follow-ups
#  (per-beam hoist, 4-D aggregate, M.2101 2-D LUT)
# ===========================================================================


# ---------------------------------------------------------------------------
#  _lookup_peak_pfd_k_2d_cp — bilinear interpolation, clamping, α-wrapping
# ---------------------------------------------------------------------------


@GPU_REQUIRED
def test_lookup_2d_grid_points_match_underlying_lut():
    """Exact (α, β) grid points must reproduce the raw LUT values."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.is_2d
        k_lut_host = lut_ctx.d_k_lut.get()  # (1, n_alpha, n_beta)

        # Pick a few grid indices across both axes
        alpha_indices = [0, lut_ctx.n_alpha // 4, lut_ctx.n_alpha // 2]
        beta_indices = [0, 5, lut_ctx.n_beta // 3]
        for ai in alpha_indices:
            for bi in beta_indices:
                alpha_deg = float(lut_ctx.alpha_min_deg + ai * lut_ctx.alpha_step_deg)
                beta_deg = float(bi * lut_ctx.beta_step_deg)
                alpha_rad = cp.asarray([math.radians(alpha_deg)], dtype=cp.float32)
                beta_rad = cp.asarray([math.radians(beta_deg)], dtype=cp.float32)
                shell_id = cp.zeros((1,), dtype=cp.int32)
                k_val = float(gpu_accel._lookup_peak_pfd_k_2d_cp(
                    lut_ctx, alpha_rad, beta_rad, shell_id,
                ).get()[0])
                expected = float(k_lut_host[0, ai, bi])
                assert k_val == pytest.approx(expected, rel=1.0e-4), (
                    f"α_idx={ai} β_idx={bi}: lookup={k_val:.5e} vs LUT={expected:.5e}"
                )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_lookup_2d_bilinear_interpolation_mid_cell():
    """A query at the midpoint of four grid cells must equal their average."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        k_lut_host = lut_ctx.d_k_lut.get()
        ai, bi = 10, 5
        # Query midpoint of cell (ai, bi)-(ai+1, bi+1)
        alpha_mid_deg = lut_ctx.alpha_min_deg + (ai + 0.5) * lut_ctx.alpha_step_deg
        beta_mid_deg = (bi + 0.5) * lut_ctx.beta_step_deg
        alpha_rad = cp.asarray([math.radians(alpha_mid_deg)], dtype=cp.float32)
        beta_rad = cp.asarray([math.radians(beta_mid_deg)], dtype=cp.float32)
        shell_id = cp.zeros((1,), dtype=cp.int32)
        k_interp = float(gpu_accel._lookup_peak_pfd_k_2d_cp(
            lut_ctx, alpha_rad, beta_rad, shell_id,
        ).get()[0])
        # Expected: average of four corners
        expected = 0.25 * (
            float(k_lut_host[0, ai, bi])
            + float(k_lut_host[0, ai + 1, bi])
            + float(k_lut_host[0, ai, bi + 1])
            + float(k_lut_host[0, ai + 1, bi + 1])
        )
        assert k_interp == pytest.approx(expected, rel=1.0e-4)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_lookup_2d_alpha_wrapping():
    """α values outside [-180, 180) must wrap correctly."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        shell_id = cp.zeros((1,), dtype=cp.int32)
        beta_rad = cp.asarray([math.radians(5.0)], dtype=cp.float32)

        # Query at +10° and +370° (should be the same after wrapping)
        k_10 = float(gpu_accel._lookup_peak_pfd_k_2d_cp(
            lut_ctx,
            cp.asarray([math.radians(10.0)], dtype=cp.float32),
            beta_rad, shell_id,
        ).get()[0])
        k_370 = float(gpu_accel._lookup_peak_pfd_k_2d_cp(
            lut_ctx,
            cp.asarray([math.radians(370.0)], dtype=cp.float32),
            beta_rad, shell_id,
        ).get()[0])
        assert k_10 == pytest.approx(k_370, rel=1.0e-4)

        # Same for negative wrap: -350° == +10°
        k_neg350 = float(gpu_accel._lookup_peak_pfd_k_2d_cp(
            lut_ctx,
            cp.asarray([math.radians(-350.0)], dtype=cp.float32),
            beta_rad, shell_id,
        ).get()[0])
        assert k_10 == pytest.approx(k_neg350, rel=1.0e-4)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_lookup_2d_beta_clamp_beyond_horizon():
    """β beyond the grid must clamp to the edge, returning a finite positive K."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        shell_id = cp.zeros((1,), dtype=cp.int32)
        alpha_rad = cp.zeros((1,), dtype=cp.float32)
        huge_beta = cp.asarray([math.radians(89.0)], dtype=cp.float32)
        k_val = float(gpu_accel._lookup_peak_pfd_k_2d_cp(
            lut_ctx, alpha_rad, huge_beta, shell_id,
        ).get()[0])
        assert np.isfinite(k_val) and k_val > 0.0
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
#  _lookup_peak_pfd_k_any_cp — unified dispatch correctness
# ---------------------------------------------------------------------------


@GPU_REQUIRED
def test_lookup_any_dispatches_1d_correctly():
    """_lookup_peak_pfd_k_any_cp with a 1-D LUT must produce bitwise-identical
    results to the direct _lookup_peak_pfd_k_cp call."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_narrow_s1528(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert not lut_ctx.is_2d

        rng = np.random.default_rng(42)
        beta_degs = rng.uniform(0.0, 50.0, size=(20,)).astype(np.float32)
        alpha_degs = rng.uniform(-180.0, 180.0, size=(20,)).astype(np.float32)
        beta_rad = cp.asarray(np.radians(beta_degs), dtype=cp.float32)
        alpha_rad = cp.asarray(np.radians(alpha_degs), dtype=cp.float32)
        shell_ids = cp.zeros((20,), dtype=cp.int32)

        k_direct = gpu_accel._lookup_peak_pfd_k_cp(lut_ctx, beta_rad, shell_ids)
        k_unified = gpu_accel._lookup_peak_pfd_k_any_cp(
            lut_ctx, alpha_rad, beta_rad, shell_ids,
        )
        np.testing.assert_array_equal(cp.asnumpy(k_direct), cp.asnumpy(k_unified))
    session.close(reset_device=False)


@GPU_REQUIRED
def test_lookup_any_dispatches_2d_correctly():
    """_lookup_peak_pfd_k_any_cp with a 2-D LUT must produce bitwise-identical
    results to the direct _lookup_peak_pfd_k_2d_cp call."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.is_2d

        rng = np.random.default_rng(42)
        beta_degs = rng.uniform(0.0, 50.0, size=(20,)).astype(np.float32)
        alpha_degs = rng.uniform(-180.0, 180.0, size=(20,)).astype(np.float32)
        beta_rad = cp.asarray(np.radians(beta_degs), dtype=cp.float32)
        alpha_rad = cp.asarray(np.radians(alpha_degs), dtype=cp.float32)
        shell_ids = cp.zeros((20,), dtype=cp.int32)

        k_direct = gpu_accel._lookup_peak_pfd_k_2d_cp(
            lut_ctx, alpha_rad, beta_rad, shell_ids,
        )
        k_unified = gpu_accel._lookup_peak_pfd_k_any_cp(
            lut_ctx, alpha_rad, beta_rad, shell_ids,
        )
        np.testing.assert_array_equal(cp.asnumpy(k_direct), cp.asnumpy(k_unified))
    session.close(reset_device=False)


@GPU_REQUIRED
def test_lookup_1d_rejects_2d_context():
    """Calling the 1-D lookup with a 2-D context must raise TypeError."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.is_2d
        with pytest.raises(TypeError, match="1-D"):
            gpu_accel._lookup_peak_pfd_k_cp(
                lut_ctx,
                cp.zeros((1,), dtype=cp.float32),
                cp.zeros((1,), dtype=cp.int32),
            )
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
#  _pattern_peak_gain_linear — M.2101 gain formula
# ---------------------------------------------------------------------------


@GPU_REQUIRED
def test_pattern_peak_gain_linear_m2101():
    """M.2101 peak gain = 10^((g_emax + 10·log10(n_h·n_v)) / 10)."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        gmax = gpu_accel._pattern_peak_gain_linear(ctx)
        expected_db = 5.0 + 10.0 * math.log10(64.0)
        expected = 10.0 ** (expected_db / 10.0)
        assert gmax == pytest.approx(expected, rel=1.0e-6)
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
#  M.2101 LUT builder — multi-shell + α asymmetry check
# ---------------------------------------------------------------------------


@GPU_REQUIRED
def test_m2101_lut_multi_shell():
    """Two distinct orbit shells must produce a (2, n_alpha, n_beta) LUT.
    Higher orbit must have lower K(0, 0) because the altitude (hence slant
    range at nadir) is larger."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r_m = _earth_radius_m()
        orbit_radii = np.array(
            [earth_r_m + 500.0e3, earth_r_m + 1200.0e3], dtype=np.float64,
        )
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=orbit_radii,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.is_2d
        assert int(lut_ctx.d_k_lut.shape[0]) == 2
        k_host = lut_ctx.d_k_lut.get()
        alpha_zero_idx = int(round(
            (0.0 - lut_ctx.alpha_min_deg) / lut_ctx.alpha_step_deg
        ))
        k_low = float(k_host[0, alpha_zero_idx, 0])
        k_high = float(k_host[1, alpha_zero_idx, 0])
        assert k_low > k_high, (
            "Lower orbit should have higher K(0, 0) because 1/(4π h²) is larger"
        )
    session.close(reset_device=False)


@GPU_REQUIRED
def test_m2101_lut_alpha_asymmetry():
    """An asymmetric M.2101 array (n_h ≠ n_v) should produce a LUT where
    K(α=0, β) ≠ K(α=90, β) for off-nadir β (the gain pattern is not
    rotationally symmetric)."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = session.prepare_m2101_pattern_context(
            g_emax_db=5.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=65.0, theta_3db_deg=25.0,  # asymmetric beamwidths
            d_h=0.5, d_v=0.5, n_h=16, n_v=4,  # asymmetric array
            wavelength_m=0.025,
        )
        earth_r_m = _earth_radius_m()
        orbit_r_m = earth_r_m + 550.0e3
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=ctx,
            sat_orbit_radius_m_per_sat=np.array([orbit_r_m], dtype=np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        k_host = lut_ctx.d_k_lut.get()
        alpha_0_idx = int(round(
            (0.0 - lut_ctx.alpha_min_deg) / lut_ctx.alpha_step_deg
        ))
        alpha_90_idx = int(round(
            (90.0 - lut_ctx.alpha_min_deg) / lut_ctx.alpha_step_deg
        ))
        # Pick an off-nadir β (e.g. β=10°)
        beta_idx = int(round(10.0 / lut_ctx.beta_step_deg))
        k_at_0 = float(k_host[0, alpha_0_idx, beta_idx])
        k_at_90 = float(k_host[0, alpha_90_idx, beta_idx])
        # They must differ — the array is asymmetric
        ratio_db = abs(10.0 * math.log10(k_at_0 / k_at_90))
        assert ratio_db > 0.1, (
            f"Expected α-asymmetry: K(α=0)={k_at_0:.3e} vs "
            f"K(α=90)={k_at_90:.3e} differ only {ratio_db:.3f} dB"
        )
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
#  Aggregate helper with M.2101 — multi-beam and off-nadir beams
# ---------------------------------------------------------------------------


@GPU_REQUIRED
def test_aggregate_m2101_coincident_beams_scale_linearly():
    """Two coincident nadir beams with M.2101: peak PFD doubles vs one beam."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        orbit_cp = cp.asarray([orbit_r], dtype=cp.float32)
        eirp_w = 10.0

        # One beam at nadir
        _, peak_1_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            beam_alpha_cp=cp.zeros((1, 1, 1), dtype=cp.float32),
            beam_beta_cp=cp.zeros((1, 1, 1), dtype=cp.float32),
            beam_valid_cp=cp.ones((1, 1, 1), dtype=cp.bool_),
            eirp_peak_cp=cp.full((1, 1, 1), eirp_w, dtype=cp.float32),
            orbit_radius_per_sat_cp=orbit_cp,
            sat_axis_index=1,
            pattern_context=ctx,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
            max_surface_pfd_lin=1.0e30,
        )
        # Two identical beams at nadir
        _, peak_2_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            beam_alpha_cp=cp.zeros((1, 1, 2), dtype=cp.float32),
            beam_beta_cp=cp.zeros((1, 1, 2), dtype=cp.float32),
            beam_valid_cp=cp.ones((1, 1, 2), dtype=cp.bool_),
            eirp_peak_cp=cp.full((1, 1, 2), eirp_w, dtype=cp.float32),
            orbit_radius_per_sat_cp=orbit_cp,
            sat_axis_index=1,
            pattern_context=ctx,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
            max_surface_pfd_lin=1.0e30,
        )
        ratio = float(peak_2_cp.get()[0, 0] / peak_1_cp.get()[0, 0])
        assert ratio == pytest.approx(2.0, rel=5.0e-3)
    session.close(reset_device=False)


@GPU_REQUIRED
def test_aggregate_m2101_off_nadir_beam():
    """M.2101 aggregate with a single off-nadir beam (β≠0): peak PFD must
    be positive and less than the nadir case (longer slant range)."""
    import cupy as cp

    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        ctx = _make_m2101_8x8(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        orbit_cp = cp.asarray([orbit_r], dtype=cp.float32)
        eirp_w = 10.0

        # Nadir reference
        _, peak_nadir_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            beam_alpha_cp=cp.zeros((1, 1, 1), dtype=cp.float32),
            beam_beta_cp=cp.zeros((1, 1, 1), dtype=cp.float32),
            beam_valid_cp=cp.ones((1, 1, 1), dtype=cp.bool_),
            eirp_peak_cp=cp.full((1, 1, 1), eirp_w, dtype=cp.float32),
            orbit_radius_per_sat_cp=orbit_cp,
            sat_axis_index=1,
            pattern_context=ctx,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
            max_surface_pfd_lin=1.0e30,
        )
        # Off-nadir beam at α=45°, β=20°
        _, peak_off_cp, _ = gpu_accel._compute_aggregate_surface_pfd_cap_cp(
            beam_alpha_cp=cp.asarray([[[math.radians(45.0)]]], dtype=cp.float32),
            beam_beta_cp=cp.asarray([[[math.radians(20.0)]]], dtype=cp.float32),
            beam_valid_cp=cp.ones((1, 1, 1), dtype=cp.bool_),
            eirp_peak_cp=cp.full((1, 1, 1), eirp_w, dtype=cp.float32),
            orbit_radius_per_sat_cp=orbit_cp,
            sat_axis_index=1,
            pattern_context=ctx,
            atmosphere_lut_context=None,
            target_alt_km=0.0,
            max_surface_pfd_lin=1.0e30,
        )
        peak_nadir = float(peak_nadir_cp.get()[0, 0])
        peak_off = float(peak_off_cp.get()[0, 0])
        assert 0.0 < peak_off < peak_nadir
    session.close(reset_device=False)


# ---------------------------------------------------------------------------
#  M.2101 cap in 4-D boresight-avoidance path (per-beam via hoisted factor)
# ---------------------------------------------------------------------------


@GPU_REQUIRED
def test_m2101_cap_4d_boresight_per_beam_strict_reduces():
    """4-D boresight path with M.2101 per-beam cap: a strict cap must
    strictly reduce every positive PFD entry."""
    case = _boresight_avoidance_case()
    session = gpu_accel.GpuScepterSession(
        compute_dtype=np.float32, sat_frame="xyz", watchdog_enabled=False,
    )
    with session.activate():
        library = session.prepare_satellite_link_selection_library(
            time_count=1,
            cell_count=2,
            sat_count=2,
            min_elevation_deg=0.0,
            n_links=1,
            n_beam=2,
            strategy="max_elevation",
            sat_belt_id_per_sat=case["sat_belt_id"],
            beta_max_deg_per_sat=case["sat_beta_max"],
            ras_topo=case["ras_topo"],
            include_counts=True,
            include_payload=True,
            include_eligible_mask=True,
            boresight_pointing_azimuth_deg=case["pointing_az_deg"],
            boresight_pointing_elevation_deg=case["pointing_el_deg"],
            boresight_theta1_deg=5.0,
        )
        library.add_chunk(0, case["sat_topo"], sat_azel=case["sat_azel"])
        beam_result = library.finalize_direct_epfd_beams(
            ras_cell_index=0,
            ras_sat_azel=case["ras_sat_azel"],
            ras_guard_angle_rad=float(np.deg2rad(2.0)),
            return_device=False,
        )
        tx_ctx = _make_m2101_8x8(session)
        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=case["orbit_radius_m"].astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.is_2d

        base_kw = dict(
            s1528_pattern_context=tx_ctx,
            ras_pattern_context=None,
            sat_topo=case["ras_topo_full"],
            sat_azel=case["ras_sat_azel_full"],
            beam_idx=beam_result["beam_idx"],
            beam_alpha_rad=beam_result["beam_alpha_rad"],
            beam_beta_rad=beam_result["beam_beta_rad"],
            telescope_azimuth_deg=None,
            telescope_elevation_deg=None,
            orbit_radius_m_per_sat=case["orbit_radius_m"],
            observer_alt_km=1.052,
            atmosphere_lut_context=None,
            pfd0_dbw_m2_mhz=None,
            power_input_quantity="satellite_ptx",
            satellite_ptx_dbw_channel=10.0,
            include_epfd=False,
            include_prx_total=False,
            include_total_pfd=True,
            include_per_satellite_pfd=True,
            return_device=False,
        )
        result_base = session.accumulate_ras_power(**base_kw)
        result_strict = session.accumulate_ras_power(
            **base_kw,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=-300.0,
        )
    session.close(reset_device=False)

    base_pfd = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    strict_pfd = np.asarray(result_strict["PFD_per_sat_RAS_STATION_W_m2"])
    positive = base_pfd > 0.0
    assert np.any(positive), "Need at least one positive baseline PFD"
    assert np.all(strict_pfd[positive] < base_pfd[positive])


# ---------------------------------------------------------------------------
#  M.2101 per-beam cap — non-zero steering (off-nadir beam direction)
# ---------------------------------------------------------------------------


@GPU_REQUIRED
def test_m2101_per_beam_cap_off_nadir_steering():
    """M.2101 per-beam cap with a beam steered to β=15°, α=30°:
    the cap should still reduce the PFD when the limit is strict."""
    session = gpu_accel.GpuScepterSession(compute_dtype=np.float32, watchdog_enabled=False)
    with session.activate():
        tx_ctx = _make_m2101_8x8(session)
        earth_r = _earth_radius_m()
        orbit_r = earth_r + 550.0e3
        h = orbit_r - earth_r
        h_km = h / 1000.0

        sat_topo = np.zeros((1, 1, 3), dtype=np.float32)
        sat_topo[0, 0] = [0.0, 90.0, h_km]
        sat_azel = np.zeros((1, 1, 2), dtype=np.float32)

        alpha_rad = math.radians(30.0)
        beta_rad = math.radians(15.0)
        beam_idx = np.zeros((1, 1, 1), dtype=np.int32)
        beam_alpha = np.full((1, 1, 1), alpha_rad, dtype=np.float32)
        beam_beta = np.full((1, 1, 1), beta_rad, dtype=np.float32)
        orbit_radius = np.array([orbit_r], dtype=np.float32)

        case = {
            "sat_topo": sat_topo,
            "sat_azel": sat_azel,
            "beam_idx": beam_idx,
            "beam_alpha_rad": beam_alpha,
            "beam_beta_rad": beam_beta,
            "orbit_radius_m_per_sat": orbit_radius,
        }

        lut_ctx = session.prepare_peak_pfd_lut_context(
            pattern_context=tx_ctx,
            sat_orbit_radius_m_per_sat=orbit_radius.astype(np.float64),
            atmosphere_lut_context=None,
            target_alt_km=0.0,
        )
        assert lut_ctx.is_2d

        ptx_w = 10.0
        ptx_dbw_channel = 10.0 * math.log10(ptx_w)

        result_base = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
        )
        result_strict = _run_minimal_power(
            session, tx_ctx, case,
            power_input_quantity="satellite_ptx",
            ptx_dbw_channel=ptx_dbw_channel,
            peak_pfd_lut_context=lut_ctx,
            max_surface_pfd_dbw_m2_channel=-300.0,
        )
    session.close(reset_device=False)

    pfd_base = np.asarray(result_base["PFD_per_sat_RAS_STATION_W_m2"])
    pfd_strict = np.asarray(result_strict["PFD_per_sat_RAS_STATION_W_m2"])
    positive = pfd_base > 0.0
    assert np.any(positive), "Need positive baseline PFD for the test"
    assert np.all(pfd_strict[positive] < pfd_base[positive])
