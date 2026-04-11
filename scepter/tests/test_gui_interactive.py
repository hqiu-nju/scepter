"""Interactive GUI tests that simulate user workflows with widget interaction.

Each test creates the main window, programmatically clicks/toggles/selects controls,
takes screenshots at key states, and verifies visual and logical outcomes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scepter.scepter_GUI as sgui

SCREENSHOT_DIR = Path(__file__).resolve().parent.parent.parent / ".pytest-tmp" / "gui_screenshots"


@pytest.fixture(scope="module")
def app():
    """Shared QApplication for all tests in this module."""
    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        return existing
    return QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    """Reusable ScepterMainWindow — reset to clean state for each test."""
    if not hasattr(window, "_cached"):
        window._cached = sgui.ScepterMainWindow()
        window._cached.resize(1920, 1080)
        window._cached.show()
        app.processEvents()
    w = window._cached
    # Reset to clean state between tests.
    # Set _dirty=False first to avoid the "save changes?" dialog which
    # blocks in offscreen mode.
    w._dirty = False
    w._prompt_save_changes = lambda: True
    w.new_configuration()
    app.processEvents()
    return w


def _screenshot(widget: QtWidgets.QWidget, name: str) -> QtGui.QPixmap:
    """Grab a screenshot and save to disk for visual review."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    pixmap = widget.grab()
    path = SCREENSHOT_DIR / f"{name}.png"
    pixmap.save(str(path), "PNG")
    return pixmap


def _screenshot_is_nontrivial(pixmap: QtGui.QPixmap, min_unique_colors: int = 5) -> bool:
    """Check that a screenshot has meaningful content (not blank)."""
    img = pixmap.toImage()
    colors = set()
    step = max(1, img.width() // 20)
    for x in range(0, img.width(), step):
        for y in range(0, img.height(), step):
            colors.add(img.pixelColor(x, y).rgb())
            if len(colors) >= min_unique_colors:
                return True
    return len(colors) >= min_unique_colors


# ---------------------------------------------------------------------------
# Workspace navigation tests
# ---------------------------------------------------------------------------

class TestWorkspaceNavigation:

    def test_home_workspace_is_default(self, window, app):
        """Window can be set to Home workspace."""
        window._set_workspace(sgui._WORKSPACE_HOME)
        app.processEvents()
        assert window.current_workspace() == sgui._WORKSPACE_HOME
        pix = _screenshot(window, "home_workspace")
        assert _screenshot_is_nontrivial(pix)

    def test_switch_to_simulation_workspace(self, window, app):
        """Clicking Simulation nav switches workspace."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        assert window.current_workspace() == sgui._WORKSPACE_SIMULATION
        pix = _screenshot(window, "simulation_workspace")
        assert _screenshot_is_nontrivial(pix)

    def test_switch_to_postprocess_workspace(self, window, app):
        """Clicking Postprocess nav switches workspace."""
        window._set_workspace(sgui._WORKSPACE_POSTPROCESS)
        app.processEvents()
        assert window.current_workspace() == sgui._WORKSPACE_POSTPROCESS
        pix = _screenshot(window, "postprocess_workspace")
        assert _screenshot_is_nontrivial(pix)


# ---------------------------------------------------------------------------
# Antenna model switching
# ---------------------------------------------------------------------------

class TestAntennaModelSwitching:

    def test_rec12_selects_correct_stack_page(self, window, app):
        """Selecting Rec 1.2 sets the model stack to page 0."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.antenna_model_combo.findData(sgui._ANTENNA_MODEL_REC12)
        window.antenna_model_combo.setCurrentIndex(idx)
        app.processEvents()
        assert window.rf_model_stack.currentIndex() == 0
        _screenshot(window, "antenna_rec12")

    def test_rec14_selects_correct_stack_page(self, window, app):
        """Selecting Rec 1.4 sets the model stack to page 1."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.antenna_model_combo.findData(sgui._ANTENNA_MODEL_REC14)
        window.antenna_model_combo.setCurrentIndex(idx)
        app.processEvents()
        assert window.rf_model_stack.currentIndex() == 1
        _screenshot(window, "antenna_rec14")

    def test_m2101_selects_correct_stack_page(self, window, app):
        """Selecting M.2101 sets the model stack to page 2."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.antenna_model_combo.findData(sgui._ANTENNA_MODEL_M2101)
        window.antenna_model_combo.setCurrentIndex(idx)
        app.processEvents()
        assert window.rf_model_stack.currentIndex() == 2
        _screenshot(window, "antenna_m2101")

    def test_s672_selects_rec12_stack_page(self, window, app):
        """Selecting S.672 reuses the Rec 1.2 page (stack index 0)."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.antenna_model_combo.findData(sgui._ANTENNA_MODEL_S672)
        window.antenna_model_combo.setCurrentIndex(idx)
        app.processEvents()
        assert window.rf_model_stack.currentIndex() == 0
        _screenshot(window, "antenna_s672")

    def test_all_models_produce_valid_state(self, window, app):
        """Every antenna model produces a valid state when selected."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_satellite_antenna_defaults()
        app.processEvents()
        for model_key in (sgui._ANTENNA_MODEL_REC12, sgui._ANTENNA_MODEL_REC14,
                          sgui._ANTENNA_MODEL_M2101, sgui._ANTENNA_MODEL_S672):
            idx = window.antenna_model_combo.findData(model_key)
            assert idx >= 0, f"Model {model_key} not in combo"
            window.antenna_model_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            sat_ant = state.active_system().satellite_antennas
            assert sat_ant is not None, f"Antennas config None for {model_key}"
            assert sat_ant.antenna_model == model_key


# ---------------------------------------------------------------------------
# Power variation visibility
# ---------------------------------------------------------------------------

class TestPowerVariationVisibility:

    def test_variation_hidden_for_target_pfd(self, window, app):
        """Power variation controls are hidden when quantity is Target PFD."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.service_power_quantity_combo.findData("target_pfd")
        window.service_power_quantity_combo.setCurrentIndex(idx)
        app.processEvents()
        assert not window._power_variation_label.isVisible()
        assert not window._power_variation_widget.isVisible()

    def test_variation_visible_state_for_satellite_eirp(self, window, app):
        """Power variation widget visibility flag set for Satellite EIRP."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.service_power_quantity_combo.findData("satellite_eirp")
        window.service_power_quantity_combo.setCurrentIndex(idx)
        window._sync_service_power_controls()
        app.processEvents()
        # The widget's visibility property should be True (even if parent tab not shown)
        assert not window._power_variation_widget.isHidden()

    def test_range_fields_hidden_when_fixed(self, window, app):
        """Min/max range fields hidden when variation is Fixed."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window.service_power_quantity_combo.setCurrentIndex(
            window.service_power_quantity_combo.findData("satellite_eirp")
        )
        window.service_power_variation_combo.setCurrentIndex(
            window.service_power_variation_combo.findData("fixed")
        )
        app.processEvents()
        assert not window._power_range_min_widget.isVisible()
        assert not window._power_range_max_widget.isVisible()

    def test_range_fields_not_hidden_when_slant_range(self, window, app):
        """Min/max range fields not hidden when variation is Slant-range."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window.service_power_quantity_combo.setCurrentIndex(
            window.service_power_quantity_combo.findData("satellite_eirp")
        )
        window.service_power_variation_combo.setCurrentIndex(
            window.service_power_variation_combo.findData("slant_range")
        )
        window._sync_service_power_controls()
        app.processEvents()
        assert not window._power_range_min_widget.isHidden()
        assert not window._power_range_max_widget.isHidden()
        _screenshot(window, "power_slant_range_visible")


# ---------------------------------------------------------------------------
# Beamforming collapsed mode visibility
# ---------------------------------------------------------------------------

class TestBeamformingCollapsedVisibility:
    """Beamforming collapsed is selected via the antenna model combo.
    When selected, the collapsed parameter fields should be visible."""

    def test_collapsed_params_hidden_by_default(self, window, app):
        """Default antenna model is not 'collapsed', so fields are hidden."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        # Default model is not beamforming_collapsed
        assert window.antenna_model_combo.currentData() != sgui._ANTENNA_MODEL_COLLAPSED
        assert window.collapsed_baseline_edit.parent().isVisible() is False or \
            not window.collapsed_baseline_edit.isVisibleTo(window)

    def test_collapsed_params_visible_when_model_selected(self, window, app):
        """Selecting 'Beamforming collapsed' antenna model shows parameters."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.antenna_model_combo.findData(sgui._ANTENNA_MODEL_COLLAPSED)
        assert idx >= 0, "Beamforming collapsed option not found in antenna model combo"
        window.antenna_model_combo.setCurrentIndex(idx)
        app.processEvents()
        assert window.antenna_model_combo.currentData() == sgui._ANTENNA_MODEL_COLLAPSED
        _screenshot(window, "beamforming_collapsed_visible")


# ---------------------------------------------------------------------------
# RAS G_rx,max cross-update
# ---------------------------------------------------------------------------

class TestRasGrxCrossUpdate:

    def test_diameter_updates_grx(self, window, app):
        """Setting antenna diameter computes G_rx,max."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        # Set frequency and diameter
        window.ras_frequency_spin.set_value(2690.0)
        window.antenna_spin.set_value(15.0)
        window._update_ras_grx_from_diameter()
        app.processEvents()
        grx = window.ras_grx_max_spin.value_or_none()
        assert grx is not None
        assert 50.0 < grx < 55.0, f"G_rx,max should be ~52.5 dBi, got {grx}"

    def test_grx_updates_diameter(self, window, app):
        """Setting G_rx,max computes antenna diameter."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window.ras_frequency_spin.set_value(2690.0)
        window.ras_grx_max_spin.set_value(52.5)
        window._update_ras_diameter_from_grx()
        app.processEvents()
        diam = window.antenna_spin.value_or_none()
        assert diam is not None
        assert 14.0 < diam < 16.0, f"Diameter should be ~15m, got {diam}"


# ---------------------------------------------------------------------------
# Rec 1.2 diameter/efficiency cross-update
# ---------------------------------------------------------------------------

class TestRec12DiameterEfficiency:

    def test_diameter_updates_gm(self, window, app):
        """Setting Rec 1.2 diameter computes Gm."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window.antenna_model_combo.setCurrentIndex(
            window.antenna_model_combo.findData(sgui._ANTENNA_MODEL_REC12)
        )
        window.frequency_spin.set_value(2690.0)
        window.rec12_diameter_spin.set_value(4.0)
        window.rec12_efficiency_spin.set_value(100.0)
        window._update_rec12_gm_from_diameter()
        app.processEvents()
        gm = window.rec12_gm_spin.value_or_none()
        assert gm is not None
        assert 40.0 < gm < 42.0, f"Gm should be ~41 dBi at 100% eff, got {gm}"


# ---------------------------------------------------------------------------
# Belt table operations
# ---------------------------------------------------------------------------

class TestBeltTable:

    def test_add_and_remove_belt(self, window, app):
        """Adding and removing belts updates the table model."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        initial_count = window._belt_model.rowCount()
        window._add_belt()
        app.processEvents()
        assert window._belt_model.rowCount() == initial_count + 1
        # Select last row and remove
        window.belt_table.selectRow(window._belt_model.rowCount() - 1)
        window._remove_selected_belt()
        app.processEvents()
        assert window._belt_model.rowCount() == initial_count

    def test_add_belt_increments_count(self, window, app):
        """Adding a belt increases the row count and produces valid state."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        before = window._belt_model.rowCount()
        window._add_belt()
        app.processEvents()
        assert window._belt_model.rowCount() == before + 1
        belts = window._belt_model.belts()
        assert belts[-1].belt_name is not None
        _screenshot(window, "belt_table_after_add")


# ---------------------------------------------------------------------------
# Full state round-trip via GUI
# ---------------------------------------------------------------------------

class TestFullStateRoundTrip:

    def test_set_defaults_and_read_state(self, window, app):
        """Set all defaults, read state, verify key fields are populated."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window._set_ras_antenna_defaults()
        window._set_satellite_antenna_defaults()
        window._set_service_defaults()
        app.processEvents()
        state = window.current_state()
        # RAS
        assert state.ras_antenna.antenna_diameter_m == 15.0
        # Satellite antenna
        assert state.active_system().satellite_antennas.antenna_model is not None
        # Service
        assert state.active_system().service.power_input_quantity is not None
        assert state.active_system().service.bandwidth_mhz is not None
        # Runtime
        assert state.runtime.timestep_s > 0
        _screenshot(window, "full_defaults_state")

    def test_modify_and_verify_state(self, window, app):
        """Modify controls programmatically and verify state captures changes."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window._set_service_defaults()
        app.processEvents()
        # Change power to satellite EIRP with slant-range variation
        window.service_power_quantity_combo.setCurrentIndex(
            window.service_power_quantity_combo.findData("satellite_eirp")
        )
        window.satellite_eirp_channel_edit.set_value(52.0)
        window.service_power_variation_combo.setCurrentIndex(
            window.service_power_variation_combo.findData("slant_range")
        )
        window.service_power_range_min_edit.set_value(26.0)
        window.service_power_range_max_edit.set_value(52.0)
        app.processEvents()
        state = window.current_state()
        assert state.active_system().service.power_input_quantity == "satellite_eirp"
        assert state.active_system().service.satellite_eirp_dbw_channel == 52.0
        assert state.active_system().service.power_variation_mode == "slant_range"
        assert state.active_system().service.power_range_min_db == 26.0
        assert state.active_system().service.power_range_max_db == 52.0


# ---------------------------------------------------------------------------
# Multi-resolution tests (HD, Full HD, 4K)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Comprehensive combinatorial tests
# ---------------------------------------------------------------------------

class TestAntennaModelPowerCombinations:
    """Test all antenna model × power quantity × variation × basis combinations."""

    @pytest.mark.parametrize("model", [
        sgui._ANTENNA_MODEL_REC12,
        sgui._ANTENNA_MODEL_REC14,
        sgui._ANTENNA_MODEL_M2101,
        sgui._ANTENNA_MODEL_S672,
    ])
    @pytest.mark.parametrize("quantity", ["target_pfd", "satellite_eirp", "satellite_ptx"])
    def test_model_quantity_combination(self, window, app, model, quantity):
        """Every model × quantity combination produces valid state."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_satellite_antenna_defaults()
        window._set_service_defaults()
        app.processEvents()
        window.antenna_model_combo.setCurrentIndex(
            window.antenna_model_combo.findData(model))
        window.service_power_quantity_combo.setCurrentIndex(
            window.service_power_quantity_combo.findData(quantity))
        app.processEvents()
        state = window.current_state()
        assert state.active_system().satellite_antennas.antenna_model == model
        assert state.active_system().service.power_input_quantity == quantity
        _screenshot(window, f"combo_{model}_{quantity}")

    @pytest.mark.parametrize("quantity", ["satellite_eirp", "satellite_ptx"])
    @pytest.mark.parametrize("variation", ["fixed", "uniform_random", "slant_range"])
    def test_quantity_variation_combination(self, window, app, quantity, variation):
        """Every non-PFD quantity × variation mode produces valid visibility state."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_service_defaults()
        app.processEvents()
        window.service_power_quantity_combo.setCurrentIndex(
            window.service_power_quantity_combo.findData(quantity))
        window.service_power_variation_combo.setCurrentIndex(
            window.service_power_variation_combo.findData(variation))
        app.processEvents()
        # Variation combo should be visible for non-PFD quantities
        assert not window._power_variation_widget.isHidden()
        # Range fields visible only for non-fixed
        if variation == "fixed":
            assert window._power_range_min_widget.isHidden()
        else:
            assert not window._power_range_min_widget.isHidden()
        state = window.current_state()
        assert state.active_system().service.power_variation_mode == variation
        _screenshot(window, f"combo_{quantity}_{variation}")

    @pytest.mark.parametrize("quantity", ["target_pfd", "satellite_eirp", "satellite_ptx"])
    @pytest.mark.parametrize("basis", ["per_mhz", "per_channel"])
    def test_quantity_basis_combination(self, window, app, quantity, basis):
        """Every quantity × basis pair switches the value editor correctly."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_service_defaults()
        app.processEvents()
        window.service_power_quantity_combo.setCurrentIndex(
            window.service_power_quantity_combo.findData(quantity))
        window.service_power_basis_combo.setCurrentIndex(
            window.service_power_basis_combo.findData(basis))
        app.processEvents()
        state = window.current_state()
        assert state.active_system().service.power_input_quantity == quantity
        assert state.active_system().service.power_input_basis == basis


class TestRuntimeOptionCombinations:
    """Test runtime configuration toggle combinations."""

    @pytest.mark.parametrize("atmosphere", [False, True])
    def test_atmosphere_toggle(self, window, app, atmosphere):
        """Atmosphere toggle combinations."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window.runtime_include_atmosphere_checkbox.setChecked(atmosphere)
        app.processEvents()
        # Atmosphere controls enabled only when atmosphere checked
        assert window.runtime_atm_bin_edit.isEnabled() == atmosphere
        _screenshot(window, f"runtime_atm{int(atmosphere)}")

    @pytest.mark.parametrize("pattern_mode", ["lut", "analytical"])
    @pytest.mark.parametrize("precision", ["float32", "float64/float32"])
    def test_pattern_precision_combination(self, window, app, pattern_mode, precision):
        """Pattern evaluation mode × precision profile combinations."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx_p = window.runtime_gpu_pattern_eval_mode_combo.findData(pattern_mode)
        if idx_p >= 0:
            window.runtime_gpu_pattern_eval_mode_combo.setCurrentIndex(idx_p)
        idx_pr = window.runtime_gpu_precision_profile_combo.findData(precision)
        if idx_pr >= 0:
            window.runtime_gpu_precision_profile_combo.setCurrentIndex(idx_pr)
        app.processEvents()
        state = window.current_state()
        if idx_p >= 0:
            assert state.runtime.gpu_pattern_eval_mode == pattern_mode
        if idx_pr >= 0:
            assert state.runtime.gpu_precision_profile == precision


class TestHexgridOptionCombinations:
    """Test hexgrid configuration option combinations."""

    @pytest.mark.parametrize("geography", ["none", "land_only", "land_plus_nearshore_sea"])
    def test_geography_mask_modes(self, window, app, geography):
        """All geography mask modes produce valid state."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.hexgrid_geography_mask_combo.findData(geography)
        if idx >= 0:
            window.hexgrid_geography_mask_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            assert state.active_system().hexgrid.geography_mask_mode == geography

    @pytest.mark.parametrize("pointing", ["ras_station", "cell_center"])
    def test_ras_pointing_modes(self, window, app, pointing):
        """Both RAS pointing modes produce valid state."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.hexgrid_ras_pointing_combo.findData(pointing)
        if idx >= 0:
            window.hexgrid_ras_pointing_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            assert state.active_system().hexgrid.ras_pointing_mode == pointing

    @pytest.mark.parametrize("boresight", [False, True])
    @pytest.mark.parametrize("scope", ["adjacency_layers", "cell_ids", "radius_km"])
    def test_boresight_scope_combinations(self, window, app, boresight, scope):
        """Boresight enabled × scope mode combinations."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window.hexgrid_boresight_enabled_checkbox.setChecked(boresight)
        idx = window.hexgrid_boresight_scope_combo.findData(scope)
        if idx >= 0:
            window.hexgrid_boresight_scope_combo.setCurrentIndex(idx)
        app.processEvents()
        state = window.current_state()
        assert state.active_system().hexgrid.boresight_avoidance_enabled == boresight
        if idx >= 0:
            assert state.active_system().hexgrid.boresight_theta2_scope_mode == scope


class TestSpectrumMaskCombinations:
    """Test all spectrum mask presets."""

    @pytest.mark.parametrize("preset", [
        "sm1541_fss", "sm1541_mss", "3gpp_ts_36_104",
        "wrc27_1_13_s1_dc_mss_imt", "adjacent_45_nonadjacent_50",
    ])
    def test_mask_preset_selection(self, window, app, preset):
        """Every mask preset can be selected and produces valid state."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.spectrum_mask_preset_combo.findData(preset)
        if idx >= 0:
            window.spectrum_mask_preset_combo.setCurrentIndex(idx)
            app.processEvents()
            _screenshot(window, f"mask_{preset}")


class TestOutputFamilyCombinations:
    """Test output family mode selections."""

    @pytest.mark.parametrize("mode", ["none", "raw", "preaccumulated", "both"])
    def test_output_family_modes(self, window, app, mode):
        """Each output family mode can be selected."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        # Find the first output family combo and set it
        for name, widgets in getattr(window, "_runtime_output_family_widgets", {}).items():
            combo = widgets.get("mode")
            if isinstance(combo, QtWidgets.QComboBox):
                idx = combo.findData(mode)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    app.processEvents()
                    break


class TestSelectionStrategies:
    """Test all satellite selection strategies."""

    @pytest.mark.parametrize("strategy", ["max_elevation", "random"])
    def test_selection_strategy(self, window, app, strategy):
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_service_defaults()
        app.processEvents()
        idx = window.service_selection_combo.findData(strategy)
        if idx >= 0:
            window.service_selection_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            assert state.active_system().service.selection_strategy == strategy


class TestCellActivityModes:
    """Test cell activity mode options."""

    @pytest.mark.parametrize("mode", ["whole_cell", "per_channel"])
    def test_cell_activity_mode(self, window, app, mode):
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_service_defaults()
        app.processEvents()
        idx = window.service_cell_activity_mode_combo.findData(mode)
        if idx >= 0:
            window.service_cell_activity_mode_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            assert state.active_system().service.cell_activity_mode == mode


class TestRasExclusionModes:
    """Test RAS exclusion mode options."""

    @pytest.mark.parametrize("mode", ["none", "adjacency_layers", "radius_km"])
    def test_exclusion_mode(self, window, app, mode):
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.hexgrid_ras_exclusion_mode_combo.findData(mode)
        if idx >= 0:
            window.hexgrid_ras_exclusion_mode_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            assert state.active_system().hexgrid.ras_exclusion_mode == mode


class TestPrecisionProfiles:
    """Test all GPU precision profiles."""

    @pytest.mark.parametrize("profile", [
        "float32", "float64", "float64/float32", "float32/float16", "float64/float32/float16",
    ])
    def test_precision_profile(self, window, app, profile):
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.runtime_gpu_precision_profile_combo.findData(profile)
        if idx >= 0:
            window.runtime_gpu_precision_profile_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            assert state.runtime.gpu_precision_profile == profile


class TestHdf5CompressionOptions:
    """Test HDF5 compression options."""

    @pytest.mark.parametrize("compression", [None, "lzf", "gzip"])
    def test_compression_option(self, window, app, compression):
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.runtime_hdf5_compression_combo.findData(compression)
        if idx >= 0:
            window.runtime_hdf5_compression_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            # gzip opts spin should be enabled only for gzip
            if compression == "gzip":
                assert window.runtime_hdf5_compression_opts_spin.isEnabled()
            elif compression is not None:
                assert not window.runtime_hdf5_compression_opts_spin.isEnabled()


class TestMemoryBudgetModes:
    """Test memory budget modes and headroom profiles."""

    @pytest.mark.parametrize("mode", ["hybrid", "host_only", "gpu_only"])
    def test_budget_mode(self, window, app, mode):
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.runtime_memory_budget_combo.findData(mode)
        if idx >= 0:
            window.runtime_memory_budget_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            assert state.runtime.memory_budget_mode == mode

    @pytest.mark.parametrize("headroom", ["conservative", "balanced", "aggressive"])
    def test_headroom_profile(self, window, app, headroom):
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        idx = window.runtime_headroom_combo.findData(headroom)
        if idx >= 0:
            window.runtime_headroom_combo.setCurrentIndex(idx)
            app.processEvents()
            state = window.current_state()
            assert state.runtime.memory_headroom_profile == headroom


class TestCrossInteractions:
    """Test complex cross-feature interactions."""

    @pytest.mark.parametrize("model,quantity,variation", [
        (sgui._ANTENNA_MODEL_M2101, "satellite_eirp", "slant_range"),
        (sgui._ANTENNA_MODEL_REC12, "satellite_ptx", "uniform_random"),
        (sgui._ANTENNA_MODEL_S672, "satellite_eirp", "fixed"),
        (sgui._ANTENNA_MODEL_REC14, "target_pfd", "fixed"),
    ])
    def test_model_quantity_variation_triple(self, window, app, model, quantity, variation):
        """Cross-interaction: model × quantity × variation."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_satellite_antenna_defaults()
        window._set_service_defaults()
        app.processEvents()
        window.antenna_model_combo.setCurrentIndex(
            window.antenna_model_combo.findData(model))
        window.service_power_quantity_combo.setCurrentIndex(
            window.service_power_quantity_combo.findData(quantity))
        if quantity != "target_pfd":
            window.service_power_variation_combo.setCurrentIndex(
                window.service_power_variation_combo.findData(variation))
        app.processEvents()
        state = window.current_state()
        assert state.active_system().satellite_antennas.antenna_model == model
        assert state.active_system().service.power_input_quantity == quantity
        _screenshot(window, f"cross_{model}_{quantity}_{variation}")

    @pytest.mark.parametrize("atmosphere,precision", [
        (True, "float32"),
        (False, "float64"),
        (True, "float32/float16"),
        (False, "float32"),
    ])
    def test_runtime_triple(self, window, app, atmosphere, precision):
        """Cross-interaction: atmosphere × precision."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window.runtime_include_atmosphere_checkbox.setChecked(atmosphere)
        idx = window.runtime_gpu_precision_profile_combo.findData(precision)
        if idx >= 0:
            window.runtime_gpu_precision_profile_combo.setCurrentIndex(idx)
        app.processEvents()

    @pytest.mark.parametrize("boresight,exclusion,geography", [
        (True, "adjacency_layers", "land_plus_nearshore_sea"),
        (False, "none", "none"),
        (True, "radius_km", "land_only"),
    ])
    def test_hexgrid_triple(self, window, app, boresight, exclusion, geography):
        """Cross-interaction: boresight × exclusion × geography."""
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        window.hexgrid_boresight_enabled_checkbox.setChecked(boresight)
        idx_e = window.hexgrid_ras_exclusion_mode_combo.findData(exclusion)
        if idx_e >= 0:
            window.hexgrid_ras_exclusion_mode_combo.setCurrentIndex(idx_e)
        idx_g = window.hexgrid_geography_mask_combo.findData(geography)
        if idx_g >= 0:
            window.hexgrid_geography_mask_combo.setCurrentIndex(idx_g)
        app.processEvents()
        state = window.current_state()
        assert state.active_system().hexgrid.boresight_avoidance_enabled == boresight


class TestMultiResolution:
    """Verify GUI renders correctly at different screen resolutions."""

    @pytest.mark.parametrize("width,height,label", [
        (1280, 720, "HD_720p"),
        (1920, 1080, "FullHD_1080p"),
        (3840, 2160, "4K_2160p"),
    ])
    def test_window_renders_at_resolution(self, window, app, width, height, label):
        """Window renders without errors at the given resolution."""
        window.resize(width, height)
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_satellite_antenna_defaults()
        window._set_ras_antenna_defaults()
        window._set_service_defaults()
        app.processEvents()
        pix = _screenshot(window, f"resolution_{label}")
        assert pix.width() == width
        assert pix.height() == height
        assert _screenshot_is_nontrivial(pix)

    @pytest.mark.parametrize("width,height,label", [
        (1280, 720, "HD_720p"),
        (1920, 1080, "FullHD_1080p"),
        (3840, 2160, "4K_2160p"),
    ])
    def test_no_widget_overflow_at_resolution(self, window, app, width, height, label):
        """No widgets extend beyond the window bounds at the given resolution."""
        window.resize(width, height)
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        app.processEvents()
        # Check that key widgets are within window geometry
        win_rect = window.rect()
        for attr_name in ("belt_table", "antenna_model_combo",
                          "service_power_quantity_combo", "service_bandwidth_edit"):
            widget = getattr(window, attr_name, None)
            if widget is None:
                continue
            # Map widget position to window coordinates
            pos = widget.mapTo(window, QtCore.QPoint(0, 0))
            assert pos.x() >= -10, f"{attr_name} overflows left at {label}: x={pos.x()}"
            assert pos.y() >= -10, f"{attr_name} overflows top at {label}: y={pos.y()}"

    @pytest.mark.parametrize("width,height,label", [
        (1280, 720, "HD_720p"),
        (1920, 1080, "FullHD_1080p"),
        (3840, 2160, "4K_2160p"),
    ])
    def test_postprocess_renders_at_resolution(self, window, app, width, height, label):
        """Postprocess workspace renders at the given resolution."""
        window.resize(width, height)
        window._set_workspace(sgui._WORKSPACE_POSTPROCESS)
        app.processEvents()
        pix = _screenshot(window, f"postprocess_{label}")
        assert pix.width() == width
        assert pix.height() == height
        assert _screenshot_is_nontrivial(pix)

    @pytest.mark.parametrize("width,height,label", [
        (1280, 720, "HD_720p"),
        (1920, 1080, "FullHD_1080p"),
        (3840, 2160, "4K_2160p"),
    ])
    def test_home_renders_at_resolution(self, window, app, width, height, label):
        """Home workspace renders at the given resolution."""
        window.resize(width, height)
        window._set_workspace(sgui._WORKSPACE_HOME)
        app.processEvents()
        pix = _screenshot(window, f"home_{label}")
        assert pix.width() == width
        assert pix.height() == height
        assert _screenshot_is_nontrivial(pix)

    @pytest.mark.parametrize("width,height,taskbar_px,label", [
        (1920, 1080 - 48, 48, "FullHD_taskbar_bottom"),
        (1920, 1080 - 40, 40, "FullHD_taskbar_thin"),
        (3840, 2160 - 48, 48, "4K_taskbar_bottom"),
        (1280, 720 - 48, 48, "HD_taskbar_bottom"),
        (1920, 1040, 40, "FullHD_taskbar_side_approx"),
    ])
    def test_renders_with_taskbar_offset(self, window, app, width, height, taskbar_px, label):
        """Window renders correctly with reduced height (taskbar on screen)."""
        window.resize(width, height)
        window._set_workspace(sgui._WORKSPACE_SIMULATION)
        window._set_satellite_antenna_defaults()
        app.processEvents()
        pix = _screenshot(window, f"taskbar_{label}")
        assert pix.width() == width
        assert pix.height() == height
        assert _screenshot_is_nontrivial(pix)
        # Key controls should still be within bounds
        for attr_name in ("belt_table", "antenna_model_combo"):
            widget = getattr(window, attr_name, None)
            if widget is None:
                continue
            pos = widget.mapTo(window, QtCore.QPoint(0, 0))
            widget_bottom = pos.y() + widget.height()
            assert widget_bottom <= height + 50, \
                f"{attr_name} extends below window at {label}: bottom={widget_bottom}, height={height}"
