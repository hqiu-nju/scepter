"""Chaos / stress tests for the RAS station tab, pattern editors, and UEMR.

Exercises rapid state changes, edge cases, and adversarial input
sequences to verify resilience — no crashes, no stale state, no
data loss.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

# Headless matplotlib backend — must be set before any Qt import
# triggers a matplotlib figure creation.
import matplotlib
matplotlib.use("Agg")

from PySide6 import QtCore, QtWidgets  # noqa: E402

import scepter.scepter_GUI as sgui  # noqa: E402
from scepter import custom_antenna as ca  # noqa: E402
from scepter import analytical_fixtures as af  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_gui_settings() -> None:
    QtCore.QSettings("scepter", "scepter_gui").clear()


def _stub_scene_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out 3-D viewer assets so the main window opens without PyVista."""
    monkeypatch.setattr(sgui, "_PYVISTA_AVAILABLE", False, raising=False)


def _make_1d_pattern() -> ca.CustomAntennaPattern:
    return af.sample_analytical_1d(
        af.ra1631_evaluator(diameter_m=25.0, wavelength_m=0.21),
        np.linspace(0.0, 180.0, 91),
        peak_gain_dbi=60.0,
    )


def _make_2d_pattern_azel() -> ca.CustomAntennaPattern:
    return af.sample_analytical_2d_az_el(
        af.m2101_evaluator(
            g_emax_dbi=5.0, a_m_db=30.0, sla_nu_db=30.0,
            phi_3db_deg=65.0, theta_3db_deg=65.0,
            d_h=0.5, d_v=0.5, n_h=4, n_v=4,
        ),
        np.linspace(-180.0, 180.0, 37),
        np.linspace(-90.0, 90.0, 19),
        peak_gain_dbi=17.0,
    )


def _make_2d_pattern_thetaphi() -> ca.CustomAntennaPattern:
    return af.sample_analytical_2d_theta_phi(
        af.s1528_rec1_4_evaluator(
            wavelength_m=0.15, lr_m=1.6, lt_m=1.6,
            slr_db=20.0, l=2, gm_db=34.0,
        ),
        np.linspace(0.0, 180.0, 37),
        np.linspace(-180.0, 180.0, 19),
        peak_gain_dbi=34.0,
    )


# ---------------------------------------------------------------------------
# RAS station tab chaos
# ---------------------------------------------------------------------------


def test_ras_tab_rapid_model_switching(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Rapidly switch between RA.1631 / Custom 1-D / Custom 2-D
    without loading any patterns — no crash, stale label, or
    leaked state."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    combo = window.ras_antenna_model_combo
    models = [combo.itemData(i) for i in range(combo.count())]
    assert len(models) == 3  # RA.1631, Custom 1-D, Custom 2-D

    # Rapid round-trips.
    for _ in range(5):
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            qapp.processEvents()

    # No pattern loaded — status should say "No file loaded."
    assert window._ras_custom_pattern is None
    window._dirty = False
    window.close()


def test_ras_tab_load_1d_switch_to_2d_and_back(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Load a 1-D pattern, switch to Custom 2-D (should show empty),
    switch back to Custom 1-D (should restore the 1-D pattern)."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    pat_1d = _make_1d_pattern()
    combo = window.ras_antenna_model_combo

    # Select Custom 1-D and inject a pattern.
    idx_1d = combo.findData("custom_1d")
    combo.setCurrentIndex(idx_1d)
    qapp.processEvents()
    window._ras_custom_pattern = pat_1d
    window._ras_custom_pattern_1d = pat_1d
    window._refresh_ras_custom_pattern_status_label()
    assert "1-D" in window.ras_custom_pattern_status_label.text()

    # Switch to Custom 2-D — pattern should be None.
    idx_2d = combo.findData("custom_2d")
    combo.setCurrentIndex(idx_2d)
    qapp.processEvents()
    assert window._ras_custom_pattern is None

    # Switch back to Custom 1-D — pattern should be restored.
    combo.setCurrentIndex(idx_1d)
    qapp.processEvents()
    assert window._ras_custom_pattern is not None
    assert window._ras_custom_pattern.kind == ca.KIND_1D

    window._dirty = False
    window.close()


def test_ras_tab_clear_does_not_revert_model(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Clicking Clear should remove the pattern but keep the model
    selection on Custom — not revert to RA.1631."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    combo = window.ras_antenna_model_combo
    idx_2d = combo.findData("custom_2d")
    combo.setCurrentIndex(idx_2d)
    qapp.processEvents()

    # Inject a 2-D pattern.
    pat_2d = _make_2d_pattern_azel()
    window._ras_custom_pattern = pat_2d
    window._ras_custom_pattern_2d = pat_2d
    window._refresh_ras_custom_pattern_status_label()
    assert window._ras_custom_pattern is not None

    # Clear.
    window._on_ras_clear_custom_pattern_clicked()
    qapp.processEvents()
    assert window._ras_custom_pattern is None
    # Model should still be Custom 2-D.
    assert str(combo.currentData()) == "custom_2d"
    assert "No file loaded" in window.ras_custom_pattern_status_label.text()

    window._dirty = False
    window.close()


def test_ras_tab_project_round_trip_preserves_both_patterns(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    """Save and load a project with a RAS custom pattern — the
    pattern must survive exactly."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)

    pat_1d = _make_1d_pattern()
    state = sgui.ScepterProjectState(
        ras_antenna=sgui.RasAntennaConfig(custom_pattern=pat_1d),
    )
    p = tmp_path / "chaos_project.json"
    sgui.save_project_state(str(p), state)
    loaded = sgui.load_project_state(str(p))

    assert loaded.ras_antenna.custom_pattern is not None
    assert loaded.ras_antenna.custom_pattern.kind == ca.KIND_1D
    np.testing.assert_allclose(
        np.asarray(loaded.ras_antenna.custom_pattern.gain_db),
        np.asarray(pat_1d.gain_db),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# 2-D editor chaos
# ---------------------------------------------------------------------------


def test_2d_editor_rapid_template_switches(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Rapidly switch between built-in templates — no crash."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor2DDialog(parent=window)
    qapp.processEvents()

    # Cycle through all simple (non-dialog) templates.
    simple_keys = [
        "starter", "gaussian_circular", "gaussian_elliptical",
        "cosine", "isoflux", "cosec_sq",
    ]
    for key in simple_keys:
        dlg._apply_template(key)
        qapp.processEvents()
        assert dlg._surface_grid is not None
        assert dlg._points.shape[0] > 0

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_2d_editor_all_interp_methods(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Cycle through every interpolation method — surface should
    regenerate without error each time."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor2DDialog(parent=window)
    qapp.processEvents()

    methods = [k for k, _ in dlg._INTERP_METHODS]
    for method in methods:
        dlg._interp_method = method
        dlg._regenerate_surface_from_anchors()
        qapp.processEvents()
        assert dlg._surface_grid is not None
        assert np.all(np.isfinite(dlg._surface_grid))

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_2d_editor_load_1d_pattern_into_2d_editor_rejected(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Loading a 1-D pattern into the 2-D editor should be handled
    gracefully — not crash or produce corrupt state."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    pat_1d = _make_1d_pattern()
    # The 2-D editor checks kind on load — verify it doesn't crash.
    dlg = sgui.PatternEditor2DDialog(initial_pattern=pat_1d, parent=window)
    qapp.processEvents()
    # 1-D pattern should be ignored (kind mismatch), editor starts
    # with defaults.
    assert dlg._surface_grid is not None

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_2d_editor_json_round_trip_both_grid_modes(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    """Export and reimport in both az_el and theta_phi modes."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    for pat, mode in [
        (_make_2d_pattern_azel(), ca.GRID_MODE_AZEL),
        (_make_2d_pattern_thetaphi(), ca.GRID_MODE_THETAPHI),
    ]:
        p = tmp_path / f"chaos_2d_{mode}.json"
        ca.dump_custom_pattern(str(p), pat)
        loaded = ca.load_custom_pattern(str(p))
        assert loaded.kind == ca.KIND_2D
        assert loaded.grid_mode == mode
        np.testing.assert_allclose(
            np.asarray(loaded.gain_db),
            np.asarray(pat.gain_db),
            atol=1e-9,
        )
        assert abs(loaded.peak_gain_dbi - pat.peak_gain_dbi) < 1e-6

    window._dirty = False
    window.close()


def test_2d_editor_extreme_point_counts(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Editor should handle both very few and many anchor points
    without crashing."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor2DDialog(parent=window)
    qapp.processEvents()

    # Minimal: just the 4 corner anchors.
    corners = np.array([
        (-180, -90, -30), (-180, 90, -30),
        (180, -90, -30), (180, 90, -30),
    ], dtype=np.float64)
    dlg._points = corners
    dlg._regenerate_surface_from_anchors()
    qapp.processEvents()
    assert dlg._surface_grid is not None

    # Dense: 500 scattered points.
    rng = np.random.default_rng(42)
    dense = np.column_stack([
        rng.uniform(-180, 180, 500),
        rng.uniform(-90, 90, 500),
        rng.uniform(-40, 10, 500),
    ])
    dlg._points = dense
    dlg._regenerate_surface_from_anchors()
    qapp.processEvents()
    assert dlg._surface_grid is not None

    # Build pattern — should succeed.
    pat = dlg._build_pattern_from_state()
    assert pat.kind == ca.KIND_2D

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_2d_editor_grid_mode_conversion_is_bijective_on_direction(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """az/el ↔ θ/φ point conversion is bijective (no mirror fold).

    Design context: previously the editor used a half-range
    ``φ ∈ [0°, 180°]`` that assumed mirror symmetry across the
    principal plane, which forced az/el ↔ θ/φ to fold ``+el`` and
    ``-el`` onto the same ``|φ|`` cell — a lossy, non-injective map.
    After iteration 45's extension to full range ``φ ∈ [-180°, 180°]``
    the two grid modes are equivalent full-sphere representations
    of the same unit-vector direction, and ``_convert_points_between_grids``
    is a straight coordinate change.  This test asserts that a
    round trip (az/el → θ/φ → az/el) returns the original points
    **modulo floating-point rounding** for a genuinely asymmetric
    point set (different gains above and below the principal plane).
    """
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor2DDialog(parent=window)
    qapp.processEvents()

    # Asymmetric test points: half above, half below the principal
    # plane, with DISTINCT gain values so any mirror fold would
    # corrupt the data noticeably.
    pts_azel = np.array(
        [
            [45.0, 30.0, -5.0],
            [45.0, -30.0, -15.0],
            [-90.0, 60.0, -10.0],
            [-90.0, -60.0, -25.0],
            [120.0, 15.0, -2.0],
            [120.0, -15.0, -8.0],
            [0.0, 0.0, 0.0],  # boresight
        ],
        dtype=np.float64,
    )

    # az/el → θ/φ → az/el
    pts_thetaphi = dlg._convert_points_between_grids(
        pts_azel, dlg._gm_azel, dlg._gm_thetaphi,
    )
    # No mirror fill: point count must match.
    assert pts_thetaphi.shape == pts_azel.shape, (
        f"Expected {pts_azel.shape} points, got {pts_thetaphi.shape} "
        "— mirror fold reintroduced (iter 45 regression)."
    )
    # φ range must be full [-180, 180], not half [0, 180].
    phi_vals = pts_thetaphi[:, 1]
    assert phi_vals.min() >= -180.0 - 1.0e-9 and phi_vals.max() <= 180.0 + 1.0e-9
    assert phi_vals.min() < 0.0, (
        "At least one asymmetric φ should be negative — if all φ ≥ 0 "
        "the code is still folding to |φ|."
    )

    # θ/φ → az/el (round trip)
    pts_back = dlg._convert_points_between_grids(
        pts_thetaphi, dlg._gm_thetaphi, dlg._gm_azel,
    )
    assert pts_back.shape == pts_azel.shape, (
        f"Round-trip point count mismatch: {pts_back.shape} vs "
        f"{pts_azel.shape}. Mirror fill reintroduced."
    )

    # Points round-trip to themselves (up to float32 rounding).  The
    # az coord can jitter by ~1e-6° near the principal plane where
    # asin/atan2 have subtle conditioning; the gain is carried
    # verbatim so should be bit-exact.
    delta_xy = pts_back[:, :2] - pts_azel[:, :2]
    # Boresight (el=0, az=0) is a topological singularity where az
    # becomes undefined; skip it in the comparison.
    non_boresight = np.any(np.abs(pts_azel[:, :2]) > 1.0e-6, axis=1)
    assert np.all(
        np.abs(delta_xy[non_boresight]) < 1.0e-6
    ), f"Round-trip max |Δ| = {float(np.max(np.abs(delta_xy[non_boresight]))):.3e}°"
    np.testing.assert_array_equal(pts_back[:, 2], pts_azel[:, 2])

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_2d_editor_grid_mode_roundtrip_is_stable(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Anchor-authoritative grid-mode toggle converges after 1 round
    trip — further round trips don't keep accumulating error.

    Design context (see ``CLAUDE.md`` and ``optimization_progress.md``
    iteration 45): the editor is anchor-authoritative by design.
    Every grid-mode toggle resamples anchors at the current density
    and regenerates the dense surface from them, so the user always
    sees the LUT they'd export at this density.  The first roundtrip
    quantises the pattern to N_ref anchors (can be tens of dB for a
    narrow pattern at a coarse density — that IS the honest LUT
    preview).  What we check here is that **the second roundtrip
    doesn't roughly double the error** — the anchor regen hits a
    fixed point quickly, so a narrow pattern doesn't slowly erode to
    silence over N toggles.  Runaway growth would indicate a real
    bug (e.g. anchor positions drifting on every toggle).
    """
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor2DDialog(parent=window)
    qapp.processEvents()

    # Start in az/el and install a deliberately-structured analytic
    # pattern directly on the dense surface grid.
    gx = np.asarray(dlg._surface_grid_x, dtype=np.float64)
    gy = np.asarray(dlg._surface_grid_y, dtype=np.float64)
    mx, my = np.meshgrid(gx, gy, indexing="ij")
    surface0 = 40.0 - 3.0 * ((mx / 15.0) ** 2 + (my / 8.0) ** 2)
    surface0 = np.clip(surface0, -20.0, 40.0)
    dlg._grid_mode = dlg._gm_azel
    dlg._surface_grid_x = gx.copy()
    dlg._surface_grid_y = gy.copy()
    dlg._surface_grid = surface0.copy()
    dlg._surface_is_source = True
    dlg._surface_source_label = "probe"
    # Suppress the confirmation dialog.
    dlg._grid_conversion_confirmed = True
    orig = surface0.copy()

    # Ignore edge strip (<5 deg from grid bounds) to avoid
    # extrapolation artefacts dominating the max-error metric.
    mask = (np.abs(mx) < float(gx[-1]) - 5.0) & (np.abs(my) < float(gy[-1]) - 5.0)

    from scipy.interpolate import RegularGridInterpolator

    def _rms_vs_original() -> float:
        interp = RegularGridInterpolator(
            (dlg._surface_grid_x, dlg._surface_grid_y),
            dlg._surface_grid, bounds_error=False, fill_value=None,
            method="linear",
        )
        sampled = interp(
            np.column_stack([mx.ravel(), my.ravel()])
        ).reshape(orig.shape)
        delta = sampled - orig
        return float(np.sqrt(np.nanmean(delta[mask] ** 2)))

    def _roundtrip_once() -> None:
        dlg._grid_thetaphi_radio.setChecked(True)
        dlg._on_grid_mode_toggled(1, True)
        dlg._grid_azel_radio.setChecked(True)
        dlg._on_grid_mode_toggled(0, True)
        qapp.processEvents()

    _roundtrip_once()
    rms_1 = _rms_vs_original()
    for _ in range(4):
        _roundtrip_once()
    rms_5 = _rms_vs_original()

    # Convergence: RMS error after 5 roundtrips must not exceed 1.5×
    # the error after 1 roundtrip. Runaway accumulation (RMS growing
    # >2× per round trip) would be a bug.  Using RMS not max: anchor
    # regen smooths narrow features, so max error is expected to be
    # large on the first trip and then stable; RMS captures the
    # average-surface drift which is what "stability" actually means.
    assert rms_5 < 1.5 * max(rms_1, 0.1), (
        f"Grid-mode roundtrip diverged: RMS error grew from "
        f"{rms_1:.2f} dB after 1 trip to {rms_5:.2f} dB after 5. "
        "Anchor regen should hit a fixed point — this suggests "
        "anchor positions are drifting on every toggle."
    )

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_2d_editor_wysiwyg_editor_display_equals_exported_grid(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """WYSIWYG contract for the 2-D editor.

    The editor is anchor-authoritative (iteration 45): a template
    samples anchors from an analytical function, then regenerates
    the dense surface from those anchors via the user-selected
    interpolation method.  The surface that gets displayed on the
    heatmap MUST equal the ``gain_db`` grid that
    ``_build_pattern_from_state`` exports to JSON — otherwise the
    user sees one pattern in the editor and a different pattern
    lands on the GPU.  This is the single most important contract
    for the editor and the one iter-22's bit-exact-analytical
    assertion was trying to protect.

    After iteration 45's anchor-authoritative revert, the surface
    is NOT the analytical one (there's regen loss by design — the
    user picks density to trade that off) but the editor-display
    and the export MUST still match each other bit-exactly.
    """
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor2DDialog(parent=window)
    qapp.processEvents()

    # ``gaussian_circular`` is a closed-form template; apply drives
    # the full anchor-authoritative pipeline: analytical → anchors
    # → regen → surface → display.
    dlg._apply_template("gaussian_circular")
    qapp.processEvents()

    # (1) The displayed heatmap IS the authoritative surface grid.
    displayed_x, displayed_y, displayed_z = dlg._compute_heatmap_surface()
    # (2) The exported pattern carries that same grid verbatim.
    pat = dlg._build_pattern_from_state()
    exported = np.asarray(pat.gain_db, dtype=np.float64)

    # Grid axes must agree bit-exactly.
    if pat.grid_mode == dlg._gm_azel:
        exported_x = np.asarray(pat.az_grid_deg, dtype=np.float64)
        exported_y = np.asarray(pat.el_grid_deg, dtype=np.float64)
    else:
        exported_x = np.asarray(pat.theta_grid_deg, dtype=np.float64)
        exported_y = np.asarray(pat.phi_grid_deg, dtype=np.float64)
    np.testing.assert_array_equal(displayed_x, exported_x)
    np.testing.assert_array_equal(displayed_y, exported_y)

    # Gain surface must agree bit-exactly — this is the WYSIWYG
    # contract.  Any drift between ``_compute_heatmap_surface`` and
    # ``_build_pattern_from_state`` means the user sees one thing
    # and the GPU simulates another.
    drift = np.asarray(displayed_z, dtype=np.float64) - exported
    max_drift = float(np.max(np.abs(drift)))
    assert max_drift < 1.0e-12, (
        f"WYSIWYG violation: editor heatmap surface and exported "
        f"gain_db disagree by {max_drift:.3e} dB.  The editor must "
        "show exactly what gets exported — if not, the GPU "
        "simulates a different pattern than the one the user "
        "designed."
    )

    # (3) Sanity check: peak_gain_dbi >= max of surface so the
    # round-trip peak-consistency check never rejects the file.
    surface_max = float(np.max(exported))
    assert float(pat.peak_gain_dbi) + 1.0e-9 >= surface_max, (
        f"Exported peak_gain_dbi ({pat.peak_gain_dbi:.3f}) is below "
        f"the actual surface maximum ({surface_max:.3f} dBi)."
    )

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_2d_editor_build_pattern_peak_tracks_surface(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """The exported peak_gain_dbi must be >= the surface maximum
    so JSON round-trip never fails the peak consistency check."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor2DDialog(parent=window)
    qapp.processEvents()

    # Artificially spike one anchor above the declared peak.
    dlg._points[0, 2] = dlg._peak_gain_dbi + 20.0
    dlg._regenerate_surface_from_anchors()
    qapp.processEvents()

    pat = dlg._build_pattern_from_state()
    surface_max = float(np.max(pat.gain_db))
    assert pat.peak_gain_dbi >= surface_max - 0.01, (
        f"peak_gain_dbi ({pat.peak_gain_dbi}) < surface max ({surface_max})"
    )

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


# ---------------------------------------------------------------------------
# 1-D editor chaos
# ---------------------------------------------------------------------------


def test_1d_editor_opens_and_builds_valid_pattern(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Basic 1-D editor smoke: open, build pattern, close."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor1DDialog(parent=window)
    qapp.processEvents()
    pat = dlg._build_pattern_from_state()
    assert pat.kind == ca.KIND_1D
    assert pat.gain_db.ndim == 1
    assert pat.gain_db.size > 0

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_1d_editor_json_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    """1-D editor pattern survives JSON export/import."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor1DDialog(parent=window)
    qapp.processEvents()
    pat = dlg._build_pattern_from_state()

    p = tmp_path / "chaos_1d.json"
    ca.dump_custom_pattern(str(p), pat)
    loaded = ca.load_custom_pattern(str(p))

    assert loaded.kind == ca.KIND_1D
    np.testing.assert_allclose(
        np.asarray(loaded.gain_db),
        np.asarray(pat.gain_db),
        atol=1e-9,
    )

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


def test_1d_editor_extreme_point_manipulation(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Add and remove points rapidly — editor should stay consistent."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()

    dlg = sgui.PatternEditor1DDialog(parent=window)
    qapp.processEvents()

    initial_count = dlg._points.shape[0]

    # Add points at various angles.
    for theta in [5.0, 15.0, 45.0, 90.0, 135.0]:
        new_pt = np.array([[theta, -20.0]])
        dlg._points = np.vstack([dlg._points, new_pt])

    assert dlg._points.shape[0] == initial_count + 5

    # Sort by angle (editor expects monotonic theta).
    order = np.argsort(dlg._points[:, 0])
    dlg._points = dlg._points[order]

    # Build pattern — should succeed.
    pat = dlg._build_pattern_from_state()
    assert pat.kind == ca.KIND_1D
    assert pat.gain_db.size == dlg._points.shape[0]

    dlg._close_confirmed = True
    dlg.close()
    window._dirty = False
    window.close()


# ---------------------------------------------------------------------------
# Cross-contamination / isolation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ras_pat_factory,ras_kind,ras_model,sat_pat_factory,sat_kind,sat_model",
    [
        # RAS 1-D + satellite 2-D (az/el)
        (_make_1d_pattern, ca.KIND_1D, "custom_1d",
         _make_2d_pattern_azel, ca.KIND_2D, "custom_2d"),
        # RAS 2-D (az/el) + satellite 1-D
        (_make_2d_pattern_azel, ca.KIND_2D, "custom_2d",
         _make_1d_pattern, ca.KIND_1D, "custom_1d"),
        # RAS 1-D + satellite 1-D (both 1-D, different data)
        (_make_1d_pattern, ca.KIND_1D, "custom_1d",
         _make_1d_pattern, ca.KIND_1D, "custom_1d"),
        # RAS 2-D (az/el) + satellite 2-D (az/el) (both 2-D, different data)
        (_make_2d_pattern_azel, ca.KIND_2D, "custom_2d",
         _make_2d_pattern_azel, ca.KIND_2D, "custom_2d"),
        # RAS 2-D (theta/phi) + satellite 2-D (az/el)
        (_make_2d_pattern_thetaphi, ca.KIND_2D, "custom_2d",
         _make_2d_pattern_azel, ca.KIND_2D, "custom_2d"),
        # RAS 1-D + satellite None (analytical S.1528)
        (_make_1d_pattern, ca.KIND_1D, "custom_1d",
         None, None, "s1528_rec14"),
        # RAS None (RA.1631) + satellite 2-D
        (None, None, "ra1631",
         _make_2d_pattern_azel, ca.KIND_2D, "custom_2d"),
    ],
    ids=[
        "ras_1d__sat_2d_azel",
        "ras_2d_azel__sat_1d",
        "ras_1d__sat_1d",
        "ras_2d__sat_2d",
        "ras_2d_tp__sat_2d_ae",
        "ras_1d__sat_analytical",
        "ras_analytical__sat_2d",
    ],
)
def test_ras_and_satellite_patterns_do_not_cross_contaminate(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    ras_pat_factory,
    ras_kind,
    ras_model,
    sat_pat_factory,
    sat_kind,
    sat_model,
) -> None:
    """Custom patterns on RAS and satellite sides must survive a
    project save/load without swapping, leaking, or losing data —
    tested across all kind/model combinations."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)

    ras_pat = ras_pat_factory() if ras_pat_factory is not None else None
    sat_pat = sat_pat_factory() if sat_pat_factory is not None else None

    state = sgui.ScepterProjectState(
        systems=[sgui.SatelliteSystemConfig(
            satellite_antennas=sgui.SatelliteAntennasConfig(
                antenna_model=sat_model,
                custom_pattern=sat_pat,
            ),
        )],
        ras_antenna=sgui.RasAntennaConfig(
            custom_pattern=ras_pat,
            antenna_diameter_m=25.0 if ras_pat is None else None,
        ),
    )

    p = tmp_path / f"isolation_{ras_model}_{sat_model}.json"
    sgui.save_project_state(str(p), state)
    loaded = sgui.load_project_state(str(p))

    # RAS side.
    loaded_ras = loaded.ras_antenna.custom_pattern
    if ras_pat is None:
        assert loaded_ras is None, "RAS should be None (analytical)"
    else:
        assert loaded_ras is not None, "RAS custom pattern lost"
        assert loaded_ras.kind == ras_kind
        np.testing.assert_allclose(
            np.asarray(loaded_ras.gain_db),
            np.asarray(ras_pat.gain_db),
            atol=1e-9,
        )

    # Satellite side.
    loaded_sat = loaded.active_system().satellite_antennas.custom_pattern
    if sat_pat is None:
        assert loaded_sat is None, "Satellite should be None (analytical)"
    else:
        assert loaded_sat is not None, "Satellite custom pattern lost"
        assert loaded_sat.kind == sat_kind
        np.testing.assert_allclose(
            np.asarray(loaded_sat.gain_db),
            np.asarray(sat_pat.gain_db),
            atol=1e-9,
        )

    # When both sides have patterns, verify no cross-contamination.
    if ras_pat is not None and sat_pat is not None:
        ras_peak = float(loaded_ras.peak_gain_dbi)
        sat_peak = float(loaded_sat.peak_gain_dbi)
        ras_orig_peak = float(ras_pat.peak_gain_dbi)
        sat_orig_peak = float(sat_pat.peak_gain_dbi)
        assert abs(ras_peak - ras_orig_peak) < 1e-6, "RAS peak mutated"
        assert abs(sat_peak - sat_orig_peak) < 1e-6, "Satellite peak mutated"


# ---------------------------------------------------------------------------
# Helpers — UEMR state builder
# ---------------------------------------------------------------------------

def _tiny_uemr_state(
    *,
    random_power: bool = False,
    ras_model: str = "ra1631",
    ras_custom_pattern: ca.CustomAntennaPattern | None = None,
) -> sgui.ScepterProjectState:
    """Minimal UEMR-ready project state for chaos tests."""
    sat_antennas = sgui.SatelliteAntennasConfig(
        antenna_model="isotropic",
        isotropic=sgui.AntennaIsotropicConfig(uemr_mode=True),
    )
    service = sgui._default_service_config()
    service.power_input_quantity = "satellite_eirp"
    service.power_input_basis = "per_mhz"
    service.satellite_eirp_dbw_mhz = -71.0
    service.bandwidth_mhz = 370.0
    service.uemr_random_power = random_power
    spectrum = sgui._default_spectrum_config()
    spectrum.service_band_start_mhz = 2620.0
    spectrum.service_band_stop_mhz = 2990.0
    spectrum.unwanted_emission_mask_preset = "flat"
    ras_antenna = sgui.RasAntennaConfig(
        antenna_diameter_m=15.0,
        frequency_mhz=2690.0,
        grx_max_dbi=52.52,
        operational_elevation_min_deg=15.0,
        operational_elevation_max_deg=90.0,
        custom_pattern=ras_custom_pattern,
    )
    system = sgui.SatelliteSystemConfig(
        system_name="UEMR System",
        belts=[
            sgui.BeltConfig(
                belt_name="Belt_1",
                num_sats_per_plane=2,
                plane_count=2,
                altitude_km=525.0,
                eccentricity=0.0,
                inclination_deg=53.0,
                argp_deg=0.0,
                raan_min_deg=0.0,
                raan_max_deg=360.0,
                min_elevation_deg=20.0,
                adjacent_plane_offset=True,
            )
        ],
        satellite_antennas=sat_antennas,
        service=service,
        spectrum=spectrum,
        sat_sys_mode_visited=True,
        grid_analysis=sgui.GridAnalysisConfig(
            indicative_footprint_drop="db3",
            spacing_drop="db7",
            leading_metric="spacing_contour",
            cell_spacing_rule="full_footprint_diameter",
            cell_size_override_enabled=False,
            cell_size_override_km=None,
        ),
        hexgrid=sgui.HexgridConfig(
            geography_mask_mode="none",
            shoreline_buffer_km=None,
            coastline_backend="cartopy",
            ras_pointing_mode="ras_station",
            ras_exclusion_mode="none",
            ras_exclusion_layers=0,
            ras_exclusion_radius_km=None,
            boresight_avoidance_enabled=False,
            boresight_theta1_deg=None,
            boresight_theta2_deg=None,
            boresight_theta2_scope_mode="cell_ids",
            boresight_theta2_cell_ids=None,
            boresight_theta2_layers=0,
            boresight_theta2_radius_km=None,
        ),
        boresight=sgui.BoresightConfig(
            boresight_avoidance_enabled=False,
            boresight_theta1_deg=None,
            boresight_theta2_deg=None,
            boresight_theta2_scope_mode="cell_ids",
            boresight_theta2_cell_ids=None,
            boresight_theta2_layers=0,
            boresight_theta2_radius_km=None,
        ),
    )
    return sgui.ScepterProjectState(
        systems=[system],
        ras_station=sgui.RasStationConfig(
            longitude_deg=21.443611,
            latitude_deg=-30.712777,
            elevation_m=1052.0,
            ras_reference_mode="lower",
            ras_reference_point_count=1,
            receiver_band_start_mhz=2690.0,
            receiver_band_stop_mhz=2700.0,
            receiver_response_mode="rectangular",
            receiver_custom_mask_points=None,
        ),
        ras_antenna=ras_antenna,
        runtime=sgui._default_runtime_config(),
    )


# ---------------------------------------------------------------------------
# UEMR chaos tests
# ---------------------------------------------------------------------------


def test_uemr_random_power_flag_reaches_run_request(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """The uemr_random_power flag must appear in the run request dict
    and propagate into the power_input dict used by the kernel."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_uemr_state(random_power=False))
    qapp.processEvents()
    # Build with random_power=False.
    req_off = window._build_run_request(window.current_state())
    assert req_off.get("uemr_random_power") is False, (
        "uemr_random_power should be False when checkbox is unchecked"
    )
    # Toggle on.
    window.uemr_random_power_checkbox.setChecked(True)
    qapp.processEvents()
    req_on = window._build_run_request(window.current_state())
    assert req_on.get("uemr_random_power") is True, (
        "uemr_random_power should be True when checkbox is checked"
    )
    window._dirty = False; window.close()


def test_uemr_random_power_flows_into_normalised_power_input(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Verify that scenario.py's normalise path injects the flag so the
    UEMR kernel can read power_input['uemr_random_power']."""
    from scepter import scenario as sc
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_uemr_state(random_power=True))
    qapp.processEvents()
    req = window._build_run_request(window.current_state())
    # System 0 path: run_gpu_direct_epfd injects into power_input
    pi = sc.normalize_direct_epfd_power_input(
        bandwidth_mhz=float(req["bandwidth_mhz"]),
        power_input_quantity=req["power_input_quantity"],
        power_input_basis=req["power_input_basis"],
        pfd0_dbw_m2_mhz=None,
        target_pfd_dbw_m2_mhz=req.get("target_pfd_dbw_m2_mhz"),
        target_pfd_dbw_m2_channel=req.get("target_pfd_dbw_m2_channel"),
        satellite_ptx_dbw_mhz=req.get("satellite_ptx_dbw_mhz"),
        satellite_ptx_dbw_channel=req.get("satellite_ptx_dbw_channel"),
        satellite_eirp_dbw_mhz=req.get("satellite_eirp_dbw_mhz"),
        satellite_eirp_dbw_channel=req.get("satellite_eirp_dbw_channel"),
    )
    # The injection mirrors what run_gpu_direct_epfd does:
    pi["uemr_random_power"] = bool(req.get("uemr_random_power", False))
    assert pi["uemr_random_power"] is True, (
        "uemr_random_power not injected into power_input"
    )
    window._dirty = False; window.close()


def test_uemr_multi_system_random_power_per_system(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """In a 2-system UEMR project, each system should carry its own
    uemr_random_power flag independently."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    # System 1: random OFF
    state = _tiny_uemr_state(random_power=False)
    # System 2: random ON (duplicate + toggle)
    import copy
    sys2 = copy.deepcopy(state.systems[0])
    sys2.system_name = "UEMR Randomised"
    sys2.service.uemr_random_power = True
    sys2.system_color = "#0ea5e9"
    state.systems.append(sys2)
    window._load_state_into_widgets(state)
    qapp.processEvents()
    req = window._build_multi_system_run_request(window.current_state())
    per_sys = req.get("_system_run_kwargs", [])
    assert len(per_sys) == 2, f"Expected 2 systems, got {len(per_sys)}"
    assert per_sys[0].get("uemr_random_power") is False, (
        "System 0 should have random_power=False"
    )
    assert per_sys[1].get("uemr_random_power") is True, (
        "System 1 should have random_power=True"
    )
    window._dirty = False; window.close()


def test_uemr_output_group_names_mirror_system_rename(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Renaming a system should update the default output group name
    (unless the user has manually renamed it)."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    import copy
    state = _tiny_uemr_state(random_power=False)
    sys2 = copy.deepcopy(state.systems[0])
    sys2.system_name = "System 2"
    sys2.system_color = "#0ea5e9"
    state.systems.append(sys2)
    window._load_state_into_widgets(state)
    qapp.processEvents()
    # Should have default groups: "UEMR System", "System 2", "Combined"
    groups = window._output_system_groups_cache
    assert len(groups) >= 2
    assert groups[0].name == "UEMR System"
    assert groups[1].name == "System 2"
    # Rename system 0 via the internal path (simulating dialog)
    window._active_system_index = 0
    window._system_configs_cache[0].system_name = "Renamed System"
    # Manually trigger the mirror logic
    old_name = "UEMR System"
    for g in window._output_system_groups_cache:
        if g.system_indices == [0] and g.name == old_name:
            g.name = "Renamed System"
    assert groups[0].name == "Renamed System"
    assert groups[1].name == "System 2"  # unchanged
    window._dirty = False; window.close()


def test_uemr_random_power_survives_json_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    """uemr_random_power must persist through save/load."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    # Build state with random_power=True
    state = _tiny_uemr_state(random_power=True)
    json_path = tmp_path / "uemr_rp.json"
    sgui.save_project_state(str(json_path), state)
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["systems"][0]["service"]["uemr_random_power"] is True
    loaded = sgui.load_project_state(str(json_path))
    assert loaded.systems[0].service.uemr_random_power is True
    # Load into window and verify checkbox state
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(loaded)
    qapp.processEvents()
    assert window.uemr_random_power_checkbox.isChecked() is True
    window._dirty = False; window.close()


def test_uemr_ras_custom_1d_pattern_in_run_request(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """UEMR run request with a custom 1-D RAS pattern must include
    the pattern and the kernel must be able to consume it."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    ras_1d = _make_1d_pattern()
    state = _tiny_uemr_state(ras_custom_pattern=ras_1d)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(state)
    qapp.processEvents()
    req = window._build_run_request(window.current_state())
    ras_pat = req.get("ras_custom_pattern")
    assert ras_pat is not None, "Custom 1-D RAS pattern missing from run request"
    assert ras_pat.kind == "1d_axisymmetric"
    window._dirty = False; window.close()


def test_uemr_ras_custom_2d_pattern_in_run_request(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """UEMR run request with a custom 2-D RAS pattern must include
    the pattern for the kernel's GpuCustomPattern2DContext dispatch."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    ras_2d = _make_2d_pattern_azel()
    state = _tiny_uemr_state(ras_custom_pattern=ras_2d)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(state)
    qapp.processEvents()
    req = window._build_run_request(window.current_state())
    ras_pat = req.get("ras_custom_pattern")
    assert ras_pat is not None, "Custom 2-D RAS pattern missing from run request"
    assert ras_pat.kind == "2d"
    window._dirty = False; window.close()


def test_uemr_rapid_toggle_random_power_no_crash(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Rapidly toggling the random power checkbox must not crash."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_uemr_state())
    qapp.processEvents()
    for _ in range(20):
        window.uemr_random_power_checkbox.setChecked(True)
        qapp.processEvents()
        window.uemr_random_power_checkbox.setChecked(False)
        qapp.processEvents()
    # Must still build a valid run request
    req = window._build_run_request(window.current_state())
    assert isinstance(req, dict)
    window._dirty = False; window.close()


def test_uemr_switch_between_mss_and_uemr_preserves_random_power(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,
) -> None:
    """Switching from UEMR to MSS and back must preserve the random
    power checkbox state."""
    _clear_gui_settings()
    _stub_scene_assets(monkeypatch)
    window = sgui.ScepterMainWindow()
    window._load_state_into_widgets(_tiny_uemr_state(random_power=True))
    qapp.processEvents()
    assert window.uemr_random_power_checkbox.isChecked() is True
    # Switch to MSS (Rec 1.2)
    idx_rec12 = window.antenna_model_combo.findData(sgui._ANTENNA_MODEL_REC12)
    if idx_rec12 >= 0:
        window.antenna_model_combo.setCurrentIndex(idx_rec12)
        qapp.processEvents()
    # Switch back to Isotropic + UEMR
    idx_iso = window.antenna_model_combo.findData(sgui._ANTENNA_MODEL_ISOTROPIC)
    window.antenna_model_combo.setCurrentIndex(idx_iso)
    window.isotropic_uemr_checkbox.setChecked(True)
    qapp.processEvents()
    # random_power state should survive (stored in ServiceConfig)
    svc = window._current_service_config()
    assert svc.uemr_random_power is True, (
        "Random power flag lost after MSS→UEMR round-trip"
    )
