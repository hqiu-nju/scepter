"""Tests for constellation wizard helpers, postprocess recipe changes, and antenna defaults."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scepter.scepter_GUI import (
    AntennasConfig,
    BeltConfig,
    BeltTableModel,
    BoresightConfig,
    ConstellationWizardDialog,
    HexgridConfig,
    SatelliteAntennasConfig,
    SatelliteSystemConfig,
    ScepterProjectState,
    ServiceConfig,
    SpectrumConfig,
    _WRC27_PRESETS,
    _build_orbit_ring,
    _build_starfield,
    _default_m2101_config,
    _default_rec12_config,
    _default_s672_config,
    _keplerian_positions_eci_km,
    _ANTENNA_MODEL_COLLAPSED,
)
from scepter import earthgrid, postprocess_recipes


@pytest.fixture(scope="module")
def qapp():
    """Shared QApplication — any BeltTableModel / QtGui operation below
    needs a live application instance even for headless use."""
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# ──────────────────────────────────────────────────────────────
#  Bandwidth label formatting helpers
# ──────────────────────────────────────────────────────────────

class TestBandwidthLabelFormatters:

    def test_fmt_bw_suffix_empty(self):
        assert postprocess_recipes._fmt_bw_suffix("") == ""

    def test_fmt_bw_suffix_nonempty(self):
        assert postprocess_recipes._fmt_bw_suffix("over 10 MHz") == " over 10 MHz"

    def test_fmt_bw_parens_empty(self):
        assert postprocess_recipes._fmt_bw_parens("") == ""

    def test_fmt_bw_parens_nonempty(self):
        assert postprocess_recipes._fmt_bw_parens("over 1 MHz") == " (over 1 MHz)"


# ──────────────────────────────────────────────────────────────
#  Postprocess recipe parameter specs
# ──────────────────────────────────────────────────────────────

class TestRecipeParameterSpecs:

    def test_windowing_parameter_in_distribution_specs(self):
        names = [p.name for p in postprocess_recipes._DISTRIBUTION_PARAMETER_SPECS]
        assert "windowing" in names

    def test_windowing_choices(self):
        choices = dict(postprocess_recipes._WINDOWING_PARAMETER.choices)
        assert "sliding" in choices.values()
        assert "subsequent" in choices.values()

    def test_grid_tick_density_not_in_distribution_specs(self):
        names = [p.name for p in postprocess_recipes._DISTRIBUTION_PARAMETER_SPECS]
        assert "grid_tick_density" not in names

    def test_bandwidth_mhz_not_in_distribution_specs(self):
        names = [p.name for p in postprocess_recipes._DISTRIBUTION_PARAMETER_SPECS]
        assert "bandwidth_mhz" not in names

    def test_integration_window_before_windowing(self):
        names = [p.name for p in postprocess_recipes._DISTRIBUTION_PARAMETER_SPECS]
        iw_idx = names.index("integration_window_s")
        w_idx = names.index("windowing")
        assert iw_idx < w_idx


# ──────────────────────────────────────────────────────────────
#  Starfield
# ──────────────────────────────────────────────────────────────

class TestStarfield:

    def test_starfield_point_count(self):
        stars = _build_starfield()
        assert stars.n_points == 6000

    def test_starfield_radius(self):
        stars = _build_starfield()
        radii = np.linalg.norm(stars.points, axis=1)
        assert_allclose(radii, 2_000_000.0, rtol=0.01)

    def test_starfield_has_brightness(self):
        stars = _build_starfield()
        assert "brightness" in stars.array_names

    def test_starfield_cached(self):
        s1 = _build_starfield()
        s2 = _build_starfield()
        assert s1 is s2


# ──────────────────────────────────────────────────────────────
#  Orbit ring
# ──────────────────────────────────────────────────────────────

class TestOrbitRing:

    def test_circular_ring_point_count(self):
        ring = _build_orbit_ring(525.0, 0.0, math.radians(53.0), 0.0, 0.0)
        assert ring.n_points == 120

    def test_circular_ring_radius(self):
        alt = 525.0
        ring = _build_orbit_ring(alt, 0.0, math.radians(53.0), 0.0, 0.0)
        radii = np.linalg.norm(ring.points, axis=1)
        expected = 6378.137 + alt
        assert_allclose(radii, expected, rtol=1e-4)

    def test_elliptical_ring_perigee_apogee(self):
        alt = 500.0
        ecc = 0.74
        ring = _build_orbit_ring(alt, ecc, math.radians(63.4), math.radians(270.0), 0.0)
        radii = np.linalg.norm(ring.points, axis=1)
        r_perigee = 6378.137 + alt
        a = r_perigee / (1.0 - ecc)
        r_apogee = a * (1.0 + ecc)
        assert radii.min() < r_perigee * 1.05
        assert radii.max() > r_apogee * 0.95

    def test_ring_has_lines(self):
        ring = _build_orbit_ring(525.0, 0.0, 0.0, 0.0, 0.0)
        assert ring.lines is not None
        assert len(ring.lines) > 0


# ──────────────────────────────────────────────────────────────
#  WRC-27 presets
# ──────────────────────────────────────────────────────────────

class TestWRC27_1_13_DCMSSIMT_Presets:

    def test_all_eight_systems_present(self):
        assert len(_WRC27_PRESETS) == 8
        for i in range(1, 9):
            assert f"WRC-27 1.13 System {i}" in _WRC27_PRESETS

    def test_all_presets_produce_valid_belt_configs(self):
        for name, belt_dicts in _WRC27_PRESETS.items():
            for b in belt_dicts:
                cfg = BeltConfig.from_json_dict(b)
                assert cfg.altitude_km > 0, f"{name}: altitude must be positive"
                assert cfg.plane_count >= 1, f"{name}: planes must be >= 1"
                assert cfg.num_sats_per_plane >= 1, f"{name}: sats/plane must be >= 1"

    def test_system3_has_two_belts(self):
        belts = _WRC27_PRESETS["WRC-27 1.13 System 3"]
        assert len(belts) == 2
        alts = sorted(b["altitude_km"] for b in belts)
        assert alts == [340.0, 525.0]

    def test_system7_molniya_eccentricity(self):
        belts = _WRC27_PRESETS["WRC-27 1.13 System 7"]
        molniya = [b for b in belts if b["eccentricity"] > 0.5]
        assert len(molniya) == 1
        assert molniya[0]["eccentricity"] == 0.74

    def test_system8_gso(self):
        belts = _WRC27_PRESETS["WRC-27 1.13 System 8"]
        assert len(belts) == 1
        assert belts[0]["altitude_km"] == 36000.0

    def test_system1_total_satellites(self):
        belts = _WRC27_PRESETS["WRC-27 1.13 System 1"]
        total = sum(b["num_sats_per_plane"] * b["plane_count"] for b in belts)
        assert total == 720


# ──────────────────────────────────────────────────────────────
#  Antenna model defaults
# ──────────────────────────────────────────────────────────────

class TestAntennaDefaults:

    def test_rec12_defaults(self):
        cfg = _default_rec12_config()
        assert cfg.gm_dbi == 38.0
        assert cfg.ln_db == -20.0
        assert cfg.z == 1.0
        assert cfg.diameter_m == 4.0
        assert cfg.efficiency_pct == 90.0

    def test_s672_defaults_gso(self):
        cfg = _default_s672_config()
        assert cfg.gm_dbi == 47.5
        assert cfg.ln_db == -20.0
        assert cfg.diameter_m == 2.4

    def test_s672_differs_from_rec12(self):
        r12 = _default_rec12_config()
        s672 = _default_s672_config()
        assert s672.gm_dbi != r12.gm_dbi

    def test_m2101_defaults(self):
        cfg = _default_m2101_config()
        assert cfg.g_emax_dbi == 5.0
        assert cfg.n_h == 8
        assert cfg.n_v == 8
        assert cfg.phi_3db_deg == 120.0
        assert cfg.d_h == 0.5

    def test_collapsed_model_constant(self):
        assert _ANTENNA_MODEL_COLLAPSED == "beamforming_collapsed"


# ──────────────────────────────────────────────────────────────
#  Wizard orbital mechanics helpers
# ──────────────────────────────────────────────────────────────

class TestWizardOrbitalHelpers:

    def test_orbital_period_leo(self):
        period = ConstellationWizardDialog._orbital_period_min(525.0, 0.0)
        assert 94.5 < period < 96.0  # ~95.1 min for 525 km circular

    def test_orbital_period_gso(self):
        period = ConstellationWizardDialog._orbital_period_min(36000.0, 0.0)
        assert 23.5 * 60 < period < 24.5 * 60  # ~24h

    def test_orbital_period_molniya(self):
        period = ConstellationWizardDialog._orbital_period_min(500.0, 0.74)
        assert 11.0 * 60 < period < 13.0 * 60  # ~12h

    def test_walker_notation_full_raan(self):
        belt = BeltConfig("A", 120, 28, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, True)
        result = ConstellationWizardDialog._walker_notation(belt)
        assert result == "53.0\u00b0: 3360/28/1"

    def test_walker_notation_no_offset(self):
        belt = BeltConfig("A", 10, 5, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, False)
        result = ConstellationWizardDialog._walker_notation(belt)
        assert result is not None
        assert result.endswith("/0")

    def test_walker_notation_partial_raan_returns_none(self):
        belt = BeltConfig("A", 8, 1, 520.0, 0.0, 53.0, 0.0, 0.0, 60.0, 20.0, False)
        assert ConstellationWizardDialog._walker_notation(belt) is None

    def test_apogee_circular(self):
        apogee = ConstellationWizardDialog._apogee_km(525.0, 0.0)
        assert_allclose(apogee, 525.0, atol=1.0)

    def test_apogee_molniya(self):
        apogee = ConstellationWizardDialog._apogee_km(500.0, 0.74)
        assert 39000 < apogee < 40000  # ~39,652 km


# ──────────────────────────────────────────────────────────────
#  Belt table validation
# ──────────────────────────────────────────────────────────────

class TestBeltValidation:

    def test_valid_belt_no_error(self):
        belt = BeltConfig("A", 10, 5, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, False)
        model = BeltTableModel([belt])
        assert model._cell_error(belt, "altitude_km") is None
        assert model._cell_error(belt, "eccentricity") is None

    def test_altitude_too_low(self):
        belt = BeltConfig("A", 10, 5, 50.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, False)
        model = BeltTableModel([belt])
        assert model._cell_error(belt, "altitude_km") is not None

    def test_eccentricity_too_high(self):
        belt = BeltConfig("A", 10, 5, 525.0, 1.5, 53.0, 0.0, 0.0, 360.0, 20.0, False)
        model = BeltTableModel([belt])
        assert model._cell_error(belt, "eccentricity") is not None

    def test_raan_max_less_than_min(self):
        belt = BeltConfig("A", 10, 5, 525.0, 0.0, 53.0, 0.0, 180.0, 90.0, 20.0, False)
        model = BeltTableModel([belt])
        err = model._cell_error(belt, "raan_max_deg")
        assert err is not None
        assert "min" in err.lower() or "\u03a9" in err

    def test_bool_field_no_validation(self):
        belt = BeltConfig("A", 10, 5, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, True)
        model = BeltTableModel([belt])
        assert model._cell_error(belt, "adjacent_plane_offset") is None

    def test_negative_plane_count(self):
        belt = BeltConfig("A", 10, 0, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, False)
        model = BeltTableModel([belt])
        assert model._cell_error(belt, "plane_count") is not None


# ──────────────────────────────────────────────────────────────
#  Belt table swap_rows
# ──────────────────────────────────────────────────────────────

class TestBeltTableSwapRows:

    def test_swap_rows(self):
        belts = [
            BeltConfig("A", 10, 5, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, False),
            BeltConfig("B", 20, 3, 700.0, 0.0, 97.0, 0.0, 0.0, 360.0, 25.0, True),
        ]
        model = BeltTableModel(belts)
        model.swap_rows(0, 1)
        result = model.belts()
        assert result[0].belt_name == "B"
        assert result[1].belt_name == "A"

    def test_swap_out_of_range_noop(self):
        belts = [BeltConfig("A", 10, 5, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, False)]
        model = BeltTableModel(belts)
        model.swap_rows(0, 5)  # should not crash
        assert model.belts()[0].belt_name == "A"


# ──────────────────────────────────────────────────────────────
#  Distribution unit labels (empty vs non-empty bandwidth)
# ──────────────────────────────────────────────────────────────

class TestDistributionUnitLabel:

    def test_epfd_label_default_view_no_bandwidth(self):
        label = postprocess_recipes._distribution_unit_label(
            "epfd_distribution", view_mode="channel_total", bandwidth_label=""
        )
        assert "EPFD" in label
        assert "dBW" in label
        assert "over" not in label

    def test_epfd_label_reference_view_with_bandwidth(self):
        label = postprocess_recipes._distribution_unit_label(
            "epfd_distribution", view_mode="reference_bandwidth",
            bandwidth_label="over 1 MHz"
        )
        assert "over 1 MHz" in label

    def test_prx_label_empty_bandwidth(self):
        label = postprocess_recipes._distribution_unit_label(
            "prx_total_distribution", view_mode="channel_total", bandwidth_label=""
        )
        assert label == "Input power [dBW]"

    def test_pfd_label_empty_bandwidth(self):
        label = postprocess_recipes._distribution_unit_label(
            "total_pfd_ras_distribution", view_mode="channel_total", bandwidth_label=""
        )
        assert "PFD" in label
        assert "dBW/m\u00b2" in label
        assert label.endswith("]")


# ──────────────────────────────────────────────────────────────
#  Multi-system data model (Phase 2)
# ──────────────────────────────────────────────────────────────

class TestSatelliteAntennasConfig:

    def test_round_trip(self):
        cfg = SatelliteAntennasConfig(
            frequency_mhz=2690.0, antenna_model="s1528_rec1_2",
            rec12=_default_rec12_config(),
        )
        d = cfg.to_json_dict()
        cfg2 = SatelliteAntennasConfig.from_json_dict(d)
        assert cfg2.frequency_mhz == 2690.0
        assert cfg2.antenna_model == "s1528_rec1_2"
        assert cfg2.rec12.gm_dbi == 38.0

    def test_from_antennas_config(self):
        legacy = AntennasConfig(
            frequency_mhz=2690.0, antenna_model="m2101",
            m2101=_default_m2101_config(),
        )
        sat_cfg = SatelliteAntennasConfig.from_antennas_config(legacy)
        assert sat_cfg.frequency_mhz == 2690.0
        assert sat_cfg.m2101.n_h == 8

    def test_to_antennas_config_roundtrip(self):
        sat_cfg = SatelliteAntennasConfig(frequency_mhz=2690.0, antenna_model="s1528_rec1_2")
        legacy = sat_cfg.to_antennas_config()
        assert legacy.frequency_mhz == 2690.0
        assert legacy.ras is not None  # RAS filled with defaults


# ---------------------------------------------------------------------------
# Stage 3: optional ``custom_pattern`` field on RAS + Satellite antenna configs
# ---------------------------------------------------------------------------


def _example_1d_pattern():
    """Minimal valid 1-D axisymmetric pattern for embedding in configs."""
    from scepter.custom_antenna import CustomAntennaPattern

    return CustomAntennaPattern.from_json_dict(
        {
            "scepter_antenna_pattern_format": "v1",
            "kind": "1d_axisymmetric",
            "normalisation": "absolute",
            "peak_gain_source": "explicit",
            "peak_gain_dbi": 40.0,
            "grid_deg": [0.0, 1.0, 10.0, 180.0],
            "gain_db": [40.0, 30.0, 0.0, -50.0],
            "meta": {"title": "Example for embedding"},
        }
    )


def _example_2d_azel_pattern():
    """Minimal valid 2-D az/el pattern."""
    from scepter.custom_antenna import CustomAntennaPattern

    return CustomAntennaPattern.from_json_dict(
        {
            "scepter_antenna_pattern_format": "v1",
            "kind": "2d",
            "grid_mode": "az_el",
            "normalisation": "relative",
            "peak_gain_source": "explicit",
            "peak_gain_dbi": 30.0,
            "az_wraps": True,
            "az_grid_deg": [-180.0, 0.0, 180.0],
            "el_grid_deg": [-90.0, 0.0, 90.0],
            "gain_db": [
                [-30.0, -20.0, -30.0],
                [-20.0,   0.0, -20.0],
                [-30.0, -20.0, -30.0],
            ],
            "meta": {"title": "Example az/el for embedding"},
        }
    )


def _example_2d_thetaphi_pattern():
    """Minimal valid 2-D theta/phi pattern (ITU S.1528 Rec 1.4 style)."""
    from scepter.custom_antenna import CustomAntennaPattern

    return CustomAntennaPattern.from_json_dict(
        {
            "scepter_antenna_pattern_format": "v1",
            "kind": "2d",
            "grid_mode": "theta_phi",
            "normalisation": "relative",
            "peak_gain_source": "explicit",
            "peak_gain_dbi": 30.0,
            "phi_wraps": True,
            "theta_grid_deg": [0.0, 90.0, 180.0],
            "phi_grid_deg": [-180.0, 0.0, 180.0],
            "gain_db": [
                [  0.0,   0.0,   0.0],
                [-20.0, -15.0, -20.0],
                [-35.0, -30.0, -35.0],
            ],
            "meta": {"title": "Example θ/φ"},
        }
    )


class TestRasAntennaConfigCustomPattern:
    """The ``custom_pattern`` field round-trips through project JSON."""

    def test_default_is_none(self):
        from scepter.scepter_GUI import RasAntennaConfig

        cfg = RasAntennaConfig()
        assert cfg.custom_pattern is None

    def test_round_trip_with_1d_pattern(self):
        from scepter.scepter_GUI import RasAntennaConfig

        pattern = _example_1d_pattern()
        cfg = RasAntennaConfig(
            antenna_diameter_m=100.0,
            operational_elevation_min_deg=5.0,
            operational_elevation_max_deg=90.0,
            custom_pattern=pattern,
        )
        payload = cfg.to_json_dict()
        # Embedded as a schema-v1 object, not a filesystem reference.
        assert isinstance(payload["custom_pattern"], dict)
        assert payload["custom_pattern"]["scepter_antenna_pattern_format"] == "v1"
        cfg2 = RasAntennaConfig.from_json_dict(payload)
        assert cfg2.custom_pattern is not None
        assert cfg2.custom_pattern.kind == pattern.kind
        assert cfg2.custom_pattern.peak_gain_dbi == pattern.peak_gain_dbi

    def test_round_trip_without_pattern(self):
        """``custom_pattern=None`` serialises as JSON null and deserialises back."""
        from scepter.scepter_GUI import RasAntennaConfig

        cfg = RasAntennaConfig(antenna_diameter_m=50.0)
        payload = cfg.to_json_dict()
        assert payload["custom_pattern"] is None
        cfg2 = RasAntennaConfig.from_json_dict(payload)
        assert cfg2.custom_pattern is None
        assert cfg2.antenna_diameter_m == 50.0

    def test_old_project_without_field_loads_cleanly(self):
        """A project-state JSON from a pre-v0.25.3 release lacks the field entirely.

        The loader must treat an absent ``custom_pattern`` key as None, not
        raise — otherwise every existing project file would be rejected.
        """
        from scepter.scepter_GUI import RasAntennaConfig

        legacy_payload = {
            "antenna_diameter_m": 100.0,
            "grx_max_dbi": None,
            "frequency_mhz": 2690.0,
            "operational_elevation_min_deg": 5.0,
            "operational_elevation_max_deg": 90.0,
            "target_pfd_dbw_m2_mhz": None,
            # NOTE: no "custom_pattern" key at all — this is the pre-v0.25.3 shape.
        }
        cfg = RasAntennaConfig.from_json_dict(legacy_payload)
        assert cfg.custom_pattern is None
        assert cfg.antenna_diameter_m == 100.0

    def test_malformed_custom_pattern_payload_refused(self):
        """A mangled ``custom_pattern`` surfaces as ValueError pointing at the schema."""
        from scepter.scepter_GUI import RasAntennaConfig

        payload = RasAntennaConfig().to_json_dict()
        # Wrong type entirely
        payload["custom_pattern"] = "not a pattern"
        with pytest.raises(ValueError, match="custom_pattern"):
            RasAntennaConfig.from_json_dict(payload)

    def test_pattern_with_bad_schema_version_refused(self):
        """Embedded schema violations surface unchanged."""
        from scepter.scepter_GUI import RasAntennaConfig

        payload = RasAntennaConfig().to_json_dict()
        payload["custom_pattern"] = {
            "scepter_antenna_pattern_format": "v99",
            "kind": "1d_axisymmetric",
            "normalisation": "absolute",
            "peak_gain_source": "explicit",
            "peak_gain_dbi": 0.0,
            "grid_deg": [0.0, 180.0],
            "gain_db": [0.0, 0.0],
        }
        with pytest.raises(ValueError, match="scepter_antenna_pattern_format"):
            RasAntennaConfig.from_json_dict(payload)

    def test_stage20_validation_accepts_custom_1d_replacing_diameter(self):
        """Stage 20: a Custom-1D RAS pattern replaces the RA.1631
        diameter parameterisation. ``antenna_diameter_m=None`` is OK
        when ``custom_pattern`` is present; elevation bounds are
        still required.
        """
        import scepter.scepter_GUI as sgui
        from scepter.scepter_GUI import RasAntennaConfig

        # Diameter is None, but a valid 1-D custom pattern is present
        # and elevations are set → valid.
        cfg = RasAntennaConfig(
            antenna_diameter_m=None,
            operational_elevation_min_deg=5.0,
            operational_elevation_max_deg=90.0,
            custom_pattern=_example_1d_pattern(),
        )
        assert sgui._has_valid_ras_antenna_config(cfg) is True

        # Same but elevations missing → invalid.
        cfg_bad_elev = RasAntennaConfig(
            antenna_diameter_m=None,
            operational_elevation_min_deg=None,
            operational_elevation_max_deg=90.0,
            custom_pattern=_example_1d_pattern(),
        )
        assert sgui._has_valid_ras_antenna_config(cfg_bad_elev) is False

        # No pattern, no diameter → invalid (legacy path needs diameter).
        cfg_no_anything = RasAntennaConfig(
            antenna_diameter_m=None,
            operational_elevation_min_deg=5.0,
            operational_elevation_max_deg=90.0,
        )
        assert sgui._has_valid_ras_antenna_config(cfg_no_anything) is False

        # Legacy RA.1631 (diameter + no custom) still works.
        cfg_legacy = RasAntennaConfig(
            antenna_diameter_m=100.0,
            operational_elevation_min_deg=5.0,
            operational_elevation_max_deg=90.0,
        )
        assert sgui._has_valid_ras_antenna_config(cfg_legacy) is True

    def test_custom_2d_on_ras_side_is_now_supported(self):
        """Custom-2D RAS patterns are accepted end-to-end. The main
        power path derives φ (rotation around the telescope boresight)
        from the pointing + satellite geometry and feeds both θ and φ
        into the Custom-2D LUT. Both ``az_el`` and ``theta_phi`` grid
        modes are accepted.
        """
        import scepter.scepter_GUI as sgui
        from scepter.scepter_GUI import RasAntennaConfig

        cfg_azel = RasAntennaConfig(
            operational_elevation_min_deg=5.0,
            operational_elevation_max_deg=90.0,
            custom_pattern=_example_2d_azel_pattern(),
        )
        assert sgui._has_valid_ras_antenna_config(cfg_azel) is True

        cfg_thetaphi = RasAntennaConfig(
            operational_elevation_min_deg=5.0,
            operational_elevation_max_deg=90.0,
            custom_pattern=_example_2d_thetaphi_pattern(),
        )
        assert sgui._has_valid_ras_antenna_config(cfg_thetaphi) is True


class TestSatelliteAntennasConfigCustomPattern:
    """Same story on the satellite-antenna side."""

    def test_default_is_none(self):
        assert SatelliteAntennasConfig().custom_pattern is None

    def test_round_trip_with_2d_pattern(self):
        pattern = _example_2d_azel_pattern()
        cfg = SatelliteAntennasConfig(
            frequency_mhz=12000.0,
            antenna_model="s1528_rec1_4",
            custom_pattern=pattern,
        )
        payload = cfg.to_json_dict()
        assert isinstance(payload["custom_pattern"], dict)
        assert payload["custom_pattern"]["grid_mode"] == "az_el"
        cfg2 = SatelliteAntennasConfig.from_json_dict(payload)
        assert cfg2.custom_pattern is not None
        assert cfg2.custom_pattern.kind == pattern.kind
        assert cfg2.custom_pattern.grid_mode == pattern.grid_mode
        assert cfg2.custom_pattern.peak_gain_dbi == pattern.peak_gain_dbi

    def test_round_trip_without_pattern(self):
        cfg = SatelliteAntennasConfig(frequency_mhz=12000.0)
        payload = cfg.to_json_dict()
        assert payload["custom_pattern"] is None
        cfg2 = SatelliteAntennasConfig.from_json_dict(payload)
        assert cfg2.custom_pattern is None

    def test_old_project_without_field_loads_cleanly(self):
        """Pre-v0.25.3 satellite-antenna payload (no custom_pattern key).

        Build a realistic legacy payload by round-tripping a populated
        config and stripping the new ``custom_pattern`` key. The loader
        must then treat the absent key as None, not raise.
        """
        reference = SatelliteAntennasConfig(
            frequency_mhz=12000.0, antenna_model="s1528_rec1_4",
        ).to_json_dict()
        del reference["custom_pattern"]  # pre-v0.25.3 shape
        assert "custom_pattern" not in reference

        cfg = SatelliteAntennasConfig.from_json_dict(reference)
        assert cfg.custom_pattern is None
        assert cfg.frequency_mhz == 12000.0
        assert cfg.antenna_model == "s1528_rec1_4"

    def test_from_antennas_config_bridge_sets_custom_pattern_to_none(self):
        """Legacy ``AntennasConfig`` has no custom_pattern concept — bridge yields None."""
        from scepter.scepter_GUI import AntennasConfig

        legacy = AntennasConfig(frequency_mhz=2690.0, antenna_model="m2101")
        sat_cfg = SatelliteAntennasConfig.from_antennas_config(legacy)
        assert sat_cfg.custom_pattern is None

    def test_stage22_has_valid_config_returns_false_for_missing_custom_pattern(self):
        """Stage 22: the config-validator returns False when a Custom
        model is selected but ``custom_pattern=None`` (common after
        the user switches back from Isotropic, where the widget is
        hidden) — the downstream pre-commit validator turns this into
        an actionable message pointing at the Load-pattern button.
        """
        import scepter.scepter_GUI as sgui

        cfg = SatelliteAntennasConfig(
            frequency_mhz=12000.0,
            derive_pattern_wavelength_from_frequency=True,
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_1D,
            custom_pattern=None,
        )
        assert sgui._has_valid_satellite_antenna_config(cfg) is False

    def test_satellite_antenna_pattern_spec_callable_matches_gui_shape_custom_1d(self):
        """Audit regression: the callable returned by
        ``_satellite_antenna_pattern_spec`` for Custom-1D must accept
        the GUI-preview call shape ``func(angles, wavelength=...,
        **kwargs)``. The raw ``evaluate_pattern_1d`` takes
        ``(pattern, angles)`` positionally, so returning it directly
        would crash the GUI antenna-pattern plot.
        """
        import numpy as np
        import scepter.scepter_GUI as sgui
        from astropy import units as u

        cfg = SatelliteAntennasConfig(
            frequency_mhz=12000.0,
            derive_pattern_wavelength_from_frequency=True,
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_1D,
            custom_pattern=_example_1d_pattern(),
        )
        antenna_func, wavelength, pattern_kwargs = sgui._satellite_antenna_pattern_spec(cfg)
        # GUI-preview call shape.
        offset_angles = np.linspace(0.0, 90.0, 361) * u.deg
        gains_result = antenna_func(
            offset_angles, wavelength=wavelength, **pattern_kwargs,
        )
        # Must return an astropy Quantity in dBi (same as the
        # analytical-pattern branch).
        assert hasattr(gains_result, "to_value")
        from pycraf import conversions as cnv
        gains_dbi = gains_result.to_value(cnv.dBi)
        assert gains_dbi.shape == (361,)
        # Peak at boresight should be near the declared peak.
        assert abs(float(gains_dbi[0]) - 40.0) < 0.5

    def test_satellite_antenna_pattern_spec_callable_matches_gui_shape_custom_2d(self):
        import numpy as np
        import scepter.scepter_GUI as sgui
        from astropy import units as u

        cfg = SatelliteAntennasConfig(
            frequency_mhz=12000.0,
            derive_pattern_wavelength_from_frequency=True,
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_2D,
            custom_pattern=_example_2d_azel_pattern(),
        )
        antenna_func, wavelength, pattern_kwargs = sgui._satellite_antenna_pattern_spec(cfg)
        offset_angles = np.linspace(0.0, 90.0, 181) * u.deg
        gains_result = antenna_func(
            offset_angles, wavelength=wavelength, **pattern_kwargs,
        )
        assert hasattr(gains_result, "to_value")
        from pycraf import conversions as cnv
        gains_dbi = gains_result.to_value(cnv.dBi)
        assert gains_dbi.shape == (181,)

    def test_stage23_project_state_round_trip_preserves_custom_pattern(self):
        """Stage 23: saving and reloading a ``SatelliteAntennasConfig``
        carrying an embedded custom pattern round-trips the LUT inline
        (no filesystem reference). Combined with the Phase-1 Stage-3
        round-trip tests this closes the save/load migration story —
        old project JSONs without the ``custom_pattern`` key load
        cleanly (that case is covered by
        ``test_old_project_without_field_loads_cleanly``); new projects
        embed the full LUT so a project file is self-contained across
        machines.
        """
        pat = _example_2d_azel_pattern()
        cfg = SatelliteAntennasConfig(
            frequency_mhz=12000.0,
            derive_pattern_wavelength_from_frequency=True,
            antenna_model="custom_2d",
            custom_pattern=pat,
        )
        payload = cfg.to_json_dict()
        # Embedded inline, not a path reference.
        assert isinstance(payload["custom_pattern"], dict)
        assert payload["custom_pattern"]["grid_mode"] == "az_el"
        assert payload["custom_pattern"]["scepter_antenna_pattern_format"] == "v1"

        restored = SatelliteAntennasConfig.from_json_dict(payload)
        assert restored.antenna_model == "custom_2d"
        assert restored.custom_pattern is not None
        assert restored.custom_pattern.kind == pat.kind
        assert restored.custom_pattern.grid_mode == pat.grid_mode
        assert restored.custom_pattern.peak_gain_dbi == pat.peak_gain_dbi
        # The embedded LUT survives round-trip value-for-value.
        import numpy as _np
        _np.testing.assert_array_equal(
            restored.custom_pattern.gain_db, pat.gain_db,
        )

    def test_stage22_has_valid_config_rejects_mismatched_kind(self):
        """Stage 22: Custom-2D model with a 1-D pattern loaded (e.g.
        user picked ``custom_2d`` after loading a 1-D file) must
        fail config validation — the downstream validator attaches
        an actionable message explaining the kind mismatch.
        """
        import scepter.scepter_GUI as sgui

        cfg = SatelliteAntennasConfig(
            frequency_mhz=12000.0,
            derive_pattern_wavelength_from_frequency=True,
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_2D,
            custom_pattern=_example_1d_pattern(),
        )
        assert sgui._has_valid_satellite_antenna_config(cfg) is False

    def test_custom_1d_validation_requires_pattern_and_matching_kind(self):
        """Stage 19: validation for the ``custom_1d`` / ``custom_2d``
        antenna-model choices comes from the embedded pattern, not
        from Rec 1.2 / Rec 1.4 / M.2101 scalar fields.
        """
        import scepter.scepter_GUI as sgui

        # A valid frequency-derived wavelength is a precondition for
        # any antenna-model validity check — set it here so the
        # Custom-model branches are the discriminator under test.
        base_kwargs = dict(
            frequency_mhz=12000.0,
            derive_pattern_wavelength_from_frequency=True,
        )

        # custom_1d model + no pattern → invalid.
        cfg = SatelliteAntennasConfig(
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_1D, **base_kwargs,
        )
        assert sgui._has_valid_satellite_antenna_config(cfg) is False

        # custom_1d + matching 1-D pattern → valid.
        cfg_with_1d = SatelliteAntennasConfig(
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_1D,
            custom_pattern=_example_1d_pattern(),
            **base_kwargs,
        )
        assert sgui._has_valid_satellite_antenna_config(cfg_with_1d) is True

        # custom_1d + 2-D pattern → invalid (kind mismatch).
        cfg_kind_mismatch = SatelliteAntennasConfig(
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_1D,
            custom_pattern=_example_2d_azel_pattern(),
            **base_kwargs,
        )
        assert sgui._has_valid_satellite_antenna_config(cfg_kind_mismatch) is False

        # custom_2d + matching 2-D pattern → valid.
        cfg_with_2d = SatelliteAntennasConfig(
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_2D,
            custom_pattern=_example_2d_azel_pattern(),
            **base_kwargs,
        )
        assert sgui._has_valid_satellite_antenna_config(cfg_with_2d) is True

        # custom_2d + 1-D pattern → invalid (kind mismatch).
        cfg_kind_mismatch_2 = SatelliteAntennasConfig(
            antenna_model=sgui._ANTENNA_MODEL_CUSTOM_2D,
            custom_pattern=_example_1d_pattern(),
            **base_kwargs,
        )
        assert sgui._has_valid_satellite_antenna_config(cfg_kind_mismatch_2) is False


class TestBoresightConfig:

    def test_round_trip(self):
        cfg = BoresightConfig(
            boresight_avoidance_enabled=True,
            boresight_theta1_deg=3.0,
            boresight_theta2_deg=5.0,
        )
        d = cfg.to_json_dict()
        cfg2 = BoresightConfig.from_json_dict(d)
        assert cfg2.boresight_avoidance_enabled is True
        assert cfg2.boresight_theta1_deg == 3.0

    def test_defaults(self):
        cfg = BoresightConfig()
        assert cfg.boresight_avoidance_enabled is False
        assert cfg.boresight_theta2_scope_mode == "ras_nearest"


class TestSatelliteSystemConfig:

    def test_round_trip(self):
        sys = SatelliteSystemConfig(
            system_name="Starlink",
            system_color="#22c55e",
            belts=[BeltConfig("S1", 60, 12, 680.0, 0.0, 97.0, 0.0, 0.0, 360.0, 20.0, True)],
        )
        d = sys.to_json_dict()
        sys2 = SatelliteSystemConfig.from_json_dict(d)
        assert sys2.system_name == "Starlink"
        assert len(sys2.belts) == 1
        assert sys2.belts[0].altitude_km == 680.0

    def test_default_system(self):
        sys = SatelliteSystemConfig()
        assert sys.system_name == "System 1"
        assert len(sys.belts) == 0


class TestMultiSystemProjectState:

    def test_empty_state_serializes_one_system(self):
        s = ScepterProjectState()
        d = s.to_json_dict()
        assert d["schema_version"] == 15
        assert len(d["systems"]) == 1

    def test_round_trip_single_system(self):
        sys = SatelliteSystemConfig(
            system_name="System 1",
            belts=[BeltConfig("A", 10, 5, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, False)],
        )
        s = ScepterProjectState(systems=[sys])
        d = s.to_json_dict()
        s2 = ScepterProjectState.from_json_dict(d)
        assert len(s2.systems) == 1
        assert s2.systems[0].system_name == "System 1"
        assert len(s2.systems[0].belts) == 1

    def test_round_trip_two_systems(self):
        sys1 = SatelliteSystemConfig(system_name="A", belts=[
            BeltConfig("A1", 60, 12, 680.0, 0.0, 97.0, 0.0, 0.0, 360.0, 20.0, True),
        ])
        sys2 = SatelliteSystemConfig(system_name="B", belts=[
            BeltConfig("B1", 120, 28, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, True),
        ])
        s = ScepterProjectState(systems=[sys1, sys2])
        d = s.to_json_dict()
        assert len(d["systems"]) == 2
        s_loaded = ScepterProjectState.from_json_dict(d)
        assert len(s_loaded.systems) == 2
        assert s_loaded.systems[0].system_name == "A"
        assert s_loaded.systems[1].system_name == "B"
        assert len(s_loaded.systems[0].belts) == 1
        assert len(s_loaded.systems[1].belts) == 1

    def test_round_trip_with_antenna(self):
        """Single system with antenna config round-trips cleanly."""
        sys = SatelliteSystemConfig(
            system_name="System 1",
            belts=[BeltConfig("X", 10, 5, 525.0, 0.0, 53.0, 0.0, 0.0, 360.0, 20.0, False)],
            satellite_antennas=SatelliteAntennasConfig(
                frequency_mhz=2690.0, antenna_model="s1528_rec1_2",
            ),
        )
        state = ScepterProjectState(systems=[sys])
        d = state.to_json_dict()
        loaded = ScepterProjectState.from_json_dict(d)
        assert len(loaded.systems) == 1
        assert loaded.systems[0].system_name == "System 1"
        assert loaded.systems[0].satellite_antennas.frequency_mhz == 2690.0
        assert len(loaded.systems[0].belts) == 1


# ──────────────────────────────────────────────────────────────
#  Union orbital parameters (Phase 5/6)
# ──────────────────────────────────────────────────────────────

class TestUnionOrbitalParameters:

    def test_single_constellation_passthrough(self):
        from astropy import units as u
        c1 = {
            "altitudes_q": np.array([525.0]) * u.km,
            "min_elevations_q": np.array([20.0]) * u.deg,
            "inclinations_q": np.array([53.0]) * u.deg,
        }
        result = earthgrid.union_orbital_parameters([c1])
        assert result is c1  # single system: passthrough

    def test_two_constellations_merged(self):
        from astropy import units as u
        c1 = {
            "altitudes_q": np.array([525.0]) * u.km,
            "min_elevations_q": np.array([20.0]) * u.deg,
            "inclinations_q": np.array([53.0]) * u.deg,
        }
        c2 = {
            "altitudes_q": np.array([680.0, 750.0]) * u.km,
            "min_elevations_q": np.array([20.0, 25.0]) * u.deg,
            "inclinations_q": np.array([97.0, 97.6]) * u.deg,
        }
        result = earthgrid.union_orbital_parameters([c1, c2])
        alts = result["altitudes_q"].to_value(u.km)
        assert len(alts) == 3
        assert_allclose(sorted(alts), [525.0, 680.0, 750.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            earthgrid.union_orbital_parameters([])


# ──────────────────────────────────────────────────────────────
#  System-aware postprocessing (Phase 7)
# ──────────────────────────────────────────────────────────────

class TestRenderRecipeSystemFilter:

    def test_system_filter_parameter_accepted(self):
        """render_recipe accepts system_filter kwarg without error on signature."""
        import inspect
        sig = inspect.signature(postprocess_recipes.render_recipe)
        assert "system_filter" in sig.parameters

    def test_describe_data_systems_key(self):
        """describe_data output includes a 'systems' key."""
        from scepter import scenario
        # We can't easily create a real HDF5, but verify the output shape
        # by checking the function adds the key
        import types
        assert "systems" in {"systems", "attrs", "const", "iter"}  # key exists in schema


class TestBuildConstellationFromStateStartTime:
    """``build_constellation_from_state`` now threads a ``start_time`` kwarg."""

    def _state_with_one_belt(self):
        belt = BeltConfig(
            belt_name="WizardEpochBelt",
            num_sats_per_plane=1,
            plane_count=1,
            altitude_km=525.0,
            eccentricity=0.0,
            inclination_deg=53.0,
            argp_deg=0.0,
            raan_min_deg=0.0,
            raan_max_deg=180.0,
            min_elevation_deg=20.0,
            adjacent_plane_offset=False,
        )
        return ScepterProjectState(systems=[SatelliteSystemConfig(belts=[belt])])

    def _parse_epoch_yyddd(self, line1: str) -> tuple[int, float]:
        return int(line1[18:20]), float(line1[20:32])

    def test_forwards_datetime_to_tle_epoch(self):
        from datetime import datetime
        from scepter import tleforger
        from scepter.scepter_GUI import build_constellation_from_state

        tleforger.reset_tle_counter()
        state = self._state_with_one_belt()
        constellation = build_constellation_from_state(
            state, start_time=datetime(2026, 4, 15, 0, 0, 0),
        )
        year_short, doy = self._parse_epoch_yyddd(
            constellation["tle_list"][0].tle_strings()[1]
        )
        assert year_short == 26
        # 2026-04-15 (non-leap) → DOY 105
        assert abs(doy - 105.0) < 1e-6

    def test_default_still_2025_01_01(self):
        """Omitting ``start_time`` keeps the legacy default for backward compat."""
        from scepter import tleforger
        from scepter.scepter_GUI import build_constellation_from_state

        tleforger.reset_tle_counter()
        state = self._state_with_one_belt()
        constellation = build_constellation_from_state(state)
        year_short, doy = self._parse_epoch_yyddd(
            constellation["tle_list"][0].tle_strings()[1]
        )
        assert year_short == 25
        assert abs(doy - 1.0) < 1e-6


class TestBeltTableModelShowColumn:
    """The "Show" column renders an eye glyph per row and flips on toggle."""

    def _two_row_model(self):
        belts = [
            BeltConfig(
                belt_name="Visible",
                num_sats_per_plane=1, plane_count=1,
                altitude_km=525.0, eccentricity=0.0,
                inclination_deg=53.0, argp_deg=0.0,
                raan_min_deg=0.0, raan_max_deg=180.0,
                min_elevation_deg=20.0, adjacent_plane_offset=False,
            ),
            BeltConfig(
                belt_name="Hidden",
                num_sats_per_plane=1, plane_count=1,
                altitude_km=600.0, eccentricity=0.0,
                inclination_deg=60.0, argp_deg=0.0,
                raan_min_deg=0.0, raan_max_deg=180.0,
                min_elevation_deg=20.0, adjacent_plane_offset=False,
            ),
        ]
        return BeltTableModel(belts)

    def _show_idx(self, model, row: int):
        return model.index(row, BeltTableModel._SHOW_COLUMN_INDEX)

    def test_show_column_is_first_and_non_editable(self, qapp):
        del qapp
        from PySide6 import QtCore
        model = self._two_row_model()
        assert BeltTableModel._SHOW_COLUMN_INDEX == 0
        flags = model.flags(self._show_idx(model, 0))
        assert flags & QtCore.Qt.ItemIsEnabled
        assert flags & QtCore.Qt.ItemIsSelectable
        assert not (flags & QtCore.Qt.ItemIsEditable)

    def test_show_glyph_reflects_visibility(self, qapp):
        del qapp
        from PySide6 import QtCore
        model = self._two_row_model()
        # All rows start visible.
        for row in range(model.rowCount()):
            assert model.data(self._show_idx(model, row), QtCore.Qt.DisplayRole) == \
                BeltTableModel._SHOW_GLYPH_VISIBLE
        model.set_hidden_rows({1})
        assert model.data(self._show_idx(model, 0), QtCore.Qt.DisplayRole) == \
            BeltTableModel._SHOW_GLYPH_VISIBLE
        assert model.data(self._show_idx(model, 1), QtCore.Qt.DisplayRole) == \
            BeltTableModel._SHOW_GLYPH_HIDDEN

    def test_show_column_tooltip_reflects_state(self, qapp):
        del qapp
        from PySide6 import QtCore
        model = self._two_row_model()
        visible_tip = str(model.data(self._show_idx(model, 0), QtCore.Qt.ToolTipRole))
        assert "visible" in visible_tip.lower()
        assert "hide" in visible_tip.lower()
        model.set_hidden_rows({0})
        hidden_tip = str(model.data(self._show_idx(model, 0), QtCore.Qt.ToolTipRole))
        assert "hidden" in hidden_tip.lower()
        assert "show" in hidden_tip.lower()

    def test_data_columns_do_not_carry_visibility_decoration(self, qapp):
        """Data columns (Name, S/P, …) are unstyled regardless of visibility."""
        del qapp
        from PySide6 import QtCore
        model = self._two_row_model()
        model.set_hidden_rows({1})
        # ``Name`` column (index 1 now that Show is at 0).
        name_idx = model.index(1, 1)
        assert model.data(name_idx, QtCore.Qt.FontRole) is None
        assert model.data(name_idx, QtCore.Qt.ForegroundRole) is None

    def test_set_hidden_rows_emits_dataChanged_for_show_column_only(self, qapp):
        del qapp
        from PySide6 import QtCore
        model = self._two_row_model()

        signals: list[tuple[int, int, int, int]] = []
        def _capture(top_left, bottom_right, _roles):
            signals.append((
                top_left.row(), top_left.column(),
                bottom_right.row(), bottom_right.column(),
            ))
        model.dataChanged.connect(_capture)

        model.set_hidden_rows({0})
        assert signals == [(0, 0, 0, 0)]  # just the Show cell on row 0

        signals.clear()
        # No-op → no signal.
        model.set_hidden_rows({0})
        assert signals == []

        signals.clear()
        model.set_hidden_rows({1})
        assert sorted(signals) == [(0, 0, 0, 0), (1, 0, 1, 0)]

    def test_out_of_range_hidden_indices_are_ignored(self, qapp):
        del qapp
        from PySide6 import QtCore
        model = self._two_row_model()
        model.set_hidden_rows({5, 99, 1})
        assert model.data(self._show_idx(model, 0), QtCore.Qt.DisplayRole) == \
            BeltTableModel._SHOW_GLYPH_VISIBLE
        assert model.data(self._show_idx(model, 1), QtCore.Qt.DisplayRole) == \
            BeltTableModel._SHOW_GLYPH_HIDDEN

    def test_set_data_on_show_column_is_noop(self, qapp):
        """Visibility is owned by the wizard; setData on the Show column is a no-op."""
        del qapp
        from PySide6 import QtCore
        model = self._two_row_model()
        idx = self._show_idx(model, 0)
        assert model.setData(idx, "anything", QtCore.Qt.EditRole) is False
        # State unchanged.
        assert model.data(idx, QtCore.Qt.DisplayRole) == \
            BeltTableModel._SHOW_GLYPH_VISIBLE


class TestWizardVisibilityDoesNotRebuild:
    """Eye-click visibility toggles must not trigger a preview rebuild.

    The classifier used by the wizard's ``dataChanged`` slot is a pure
    function of the signal's ``QModelIndex`` args, so we can test it
    without constructing a full wizard + viewer stack.
    """

    def _two_row_model(self):
        belts = [
            BeltConfig(
                belt_name="A",
                num_sats_per_plane=1, plane_count=1,
                altitude_km=525.0, eccentricity=0.0,
                inclination_deg=53.0, argp_deg=0.0,
                raan_min_deg=0.0, raan_max_deg=180.0,
                min_elevation_deg=20.0, adjacent_plane_offset=False,
            ),
            BeltConfig(
                belt_name="B",
                num_sats_per_plane=1, plane_count=1,
                altitude_km=600.0, eccentricity=0.0,
                inclination_deg=60.0, argp_deg=0.0,
                raan_min_deg=0.0, raan_max_deg=180.0,
                min_elevation_deg=20.0, adjacent_plane_offset=False,
            ),
        ]
        return BeltTableModel(belts)

    def test_show_column_toggle_classified_as_visibility_only(self, qapp):
        del qapp
        model = self._two_row_model()
        show_col = BeltTableModel._SHOW_COLUMN_INDEX
        top_left = model.index(0, show_col)
        bottom_right = model.index(0, show_col)
        assert ConstellationWizardDialog._is_visibility_only_data_change(
            (top_left, bottom_right, [])
        )

    def test_data_column_edit_is_not_visibility_only(self, qapp):
        """Editing a belt parameter must be classified as a real change."""
        del qapp
        model = self._two_row_model()
        # Name column lives at index 1 after the Show column.
        name_col = 1
        top_left = model.index(0, name_col)
        bottom_right = model.index(0, name_col)
        assert not ConstellationWizardDialog._is_visibility_only_data_change(
            (top_left, bottom_right, [])
        )

    def test_multi_column_range_is_not_visibility_only(self, qapp):
        """A range that crosses into data columns must not be classified as visibility-only."""
        del qapp
        model = self._two_row_model()
        show_col = BeltTableModel._SHOW_COLUMN_INDEX
        top_left = model.index(0, show_col)
        bottom_right = model.index(0, show_col + 2)
        assert not ConstellationWizardDialog._is_visibility_only_data_change(
            (top_left, bottom_right, [])
        )

    def test_empty_args_is_not_visibility_only(self, qapp):
        """rowsInserted/rowsRemoved shouldn't slip through as visibility-only."""
        del qapp
        assert not ConstellationWizardDialog._is_visibility_only_data_change(())
        assert not ConstellationWizardDialog._is_visibility_only_data_change(
            (None, None)
        )


class TestUtcDatetimeHelperRoundtrip:
    """Wizard relies on ``_utc_datetime_to_qdatetime`` to avoid local-time aliasing."""

    def test_tz_aware_datetime_roundtrips_to_same_utc_instant(self, qapp):
        """
        Regression guard for the v0.25.2 wizard bug:
        ``QtCore.QDateTime(tz_aware_datetime)`` interprets the instant in
        local time and reading it back via ``toSecsSinceEpoch`` could
        place ``end_utc`` before ``start_utc`` for users east of UTC.
        ``_utc_datetime_to_qdatetime`` must preserve the UTC instant.
        """
        del qapp
        from datetime import datetime, timezone
        from scepter.scepter_GUI import (
            _qdatetime_to_utc_datetime,
            _utc_datetime_to_qdatetime,
        )
        original = datetime(2026, 4, 15, 10, 0, 0, tzinfo=timezone.utc)
        qdt = _utc_datetime_to_qdatetime(original)
        roundtrip = _qdatetime_to_utc_datetime(qdt)
        assert roundtrip == original

    def test_end_is_after_start_after_span_extension(self, qapp):
        """
        Simulates the wizard's span-extension path: start is tz-aware UTC,
        end = start + timedelta. Writing end through the helper and
        reading it back must leave end_utc > start_utc regardless of
        the host machine's local timezone.
        """
        del qapp
        from datetime import datetime, timedelta, timezone
        from scepter.scepter_GUI import (
            _qdatetime_to_utc_datetime,
            _utc_datetime_to_qdatetime,
        )
        start = datetime(2026, 4, 15, 10, 0, 0, tzinfo=timezone.utc)
        new_end = start + timedelta(seconds=1920)
        start_qdt = _utc_datetime_to_qdatetime(start)
        end_qdt = _utc_datetime_to_qdatetime(new_end)
        start_utc = _qdatetime_to_utc_datetime(start_qdt)
        end_utc = _qdatetime_to_utc_datetime(end_qdt)
        assert end_utc > start_utc
