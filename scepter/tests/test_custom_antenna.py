"""Unit tests for ``scepter.custom_antenna`` (schema v1, Stage 2).

Every positive case below is a minimal example of the schema documented
in ``scepter.custom_antenna``'s module docstring. Every negative case
targets a specific validation rule from the same module. When a schema
rule changes, update both the module docstring and the matching test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scepter import custom_antenna as ca


# ===========================================================================
# Fixtures — canonical valid payloads
# ===========================================================================


def _minimal_1d_payload() -> dict:
    """Copied from the schema doc's `1d_axisymmetric` minimal example."""
    return {
        "scepter_antenna_pattern_format": "v1",
        "kind": "1d_axisymmetric",
        "normalisation": "absolute",
        "peak_gain_source": "explicit",
        "peak_gain_dbi": 40.0,
        "grid_deg": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 90.0, 120.0, 180.0],
        "gain_db":  [40.0, 38.0, 32.0, 20.0, -5.0, -20.0, -35.0, -45.0, -55.0, -60.0, -70.0],
        "meta": {
            "title": "Example 40 dBi axisymmetric dish",
            "reference": "Synthetic example — not for production use",
            "phi_zero_axis": "N/A (axisymmetric)",
        },
    }


def _minimal_2d_azel_payload() -> dict:
    """Copied from the schema doc's `grid_mode=az_el` minimal example."""
    return {
        "scepter_antenna_pattern_format": "v1",
        "kind": "2d",
        "grid_mode": "az_el",
        "normalisation": "absolute",
        "peak_gain_source": "explicit",
        "peak_gain_dbi": 34.5,
        "az_wraps": True,
        "az_grid_deg": [-180.0, -90.0, -30.0, -10.0, 0.0, 10.0, 30.0, 90.0, 180.0],
        "el_grid_deg": [-90.0, -30.0, -10.0, 0.0, 10.0, 30.0, 90.0],
        "gain_db": [
            [-40.0, -28.0, -18.0, -14.0, -18.0, -28.0, -40.0],
            [-28.0, -15.0,  -5.0,   2.0,  -5.0, -15.0, -28.0],
            [-18.0,  -3.0,  14.0,  24.0,  14.0,  -3.0, -18.0],
            [-14.0,   2.0,  24.0,  34.5,  24.0,   2.0, -14.0],
            [-18.0,  -3.0,  14.0,  24.0,  14.0,  -3.0, -18.0],
            [-28.0, -15.0,  -5.0,   2.0,  -5.0, -15.0, -28.0],
            [-40.0, -28.0, -18.0, -14.0, -18.0, -28.0, -40.0],
            [-28.0, -15.0,  -5.0,   2.0,  -5.0, -15.0, -28.0],
            [-40.0, -28.0, -18.0, -14.0, -18.0, -28.0, -40.0],
        ],
        "meta": {
            "title": "Example 34.5 dBi phased array (az/el table)",
            "reference": "Synthetic example — not for production use",
            "phi_zero_axis": "N/A — az/el grid",
        },
    }


def _minimal_2d_thetaphi_payload() -> dict:
    """Copied from the schema doc's `grid_mode=theta_phi` minimal example."""
    return {
        "scepter_antenna_pattern_format": "v1",
        "kind": "2d",
        "grid_mode": "theta_phi",
        "normalisation": "relative",
        "peak_gain_source": "explicit",
        "peak_gain_dbi": 34.5,
        "phi_wraps": True,
        "theta_grid_deg": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 90.0, 180.0],
        "phi_grid_deg": [-180.0, -90.0, 0.0, 90.0, 180.0],
        "gain_db": [
            [   0.0,    0.0,    0.0,    0.0,    0.0],
            [  -1.0,   -2.5,   -1.0,   -2.5,   -1.0],
            [  -4.0,  -10.0,   -4.0,  -10.0,   -4.0],
            [ -15.0,  -25.0,  -15.0,  -25.0,  -15.0],
            [ -25.0,  -35.0,  -25.0,  -35.0,  -25.0],
            [ -40.0,  -50.0,  -40.0,  -50.0,  -40.0],
            [ -55.0,  -60.0,  -55.0,  -60.0,  -55.0],
            [ -70.0,  -75.0,  -70.0,  -75.0,  -70.0],
        ],
        "meta": {
            "title": "Example asymmetric phased-array (θ/φ table)",
            "reference": "Synthetic example — not for production use",
            "phi_zero_axis": "H-plane (along lr axis)",
        },
    }


def _itu_style_mask_payload() -> dict:
    """The schema doc's worked ITU-style mask example: step jump at 3°.

    Boresight plateau from 0° to 1°, linear slope to 3°, instantaneous
    step from +20 to 0 dBi at 3°, linear slope to 30°, flat back-lobe
    plateau to 180°.
    """
    return {
        "scepter_antenna_pattern_format": "v1",
        "kind": "1d_axisymmetric",
        "normalisation": "absolute",
        "peak_gain_source": "explicit",
        "peak_gain_dbi": 38.0,
        "grid_deg": [0.0, 1.0, 3.0, 3.0, 30.0, 180.0],
        "gain_db":  [38.0, 38.0, 20.0, 0.0, -30.0, -30.0],
        "meta": {"title": "ITU-style mask with step at 3°"},
    }


def _write_payload(tmp_path: Path, payload: dict, name: str = "pattern.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ===========================================================================
# Positive: valid payloads load
# ===========================================================================


class TestLoadValidPayloads:

    def test_1d_minimal_example(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, _minimal_1d_payload()))
        assert pattern.format_version == "v1"
        assert pattern.kind == ca.KIND_1D
        assert pattern.normalisation == ca.NORMALISATION_ABSOLUTE
        assert pattern.peak_gain_source == ca.PEAK_SOURCE_EXPLICIT
        assert pattern.peak_gain_dbi == 40.0
        assert pattern.grid_deg is not None
        assert pattern.gain_db.ndim == 1
        assert pattern.grid_deg.shape == pattern.gain_db.shape == (11,)
        assert pattern.grid_deg.dtype == np.float64
        assert pattern.gain_db.dtype == np.float64
        # 2-D fields must all be inactive for a 1-D pattern.
        assert pattern.grid_mode is None
        assert pattern.az_grid_deg is None
        assert pattern.theta_grid_deg is None

    def test_2d_azel_minimal_example(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(
            _write_payload(tmp_path, _minimal_2d_azel_payload())
        )
        assert pattern.kind == ca.KIND_2D
        assert pattern.grid_mode == ca.GRID_MODE_AZEL
        assert pattern.az_grid_deg is not None
        assert pattern.el_grid_deg is not None
        assert pattern.gain_db.shape == (
            pattern.az_grid_deg.size,
            pattern.el_grid_deg.size,
        )
        assert pattern.az_wraps is True
        # theta/phi fields inactive for az/el mode.
        assert pattern.theta_grid_deg is None
        assert pattern.phi_grid_deg is None
        assert pattern.grid_deg is None

    def test_2d_thetaphi_minimal_example(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(
            _write_payload(tmp_path, _minimal_2d_thetaphi_payload())
        )
        assert pattern.kind == ca.KIND_2D
        assert pattern.grid_mode == ca.GRID_MODE_THETAPHI
        assert pattern.theta_grid_deg is not None
        assert pattern.phi_grid_deg is not None
        assert pattern.gain_db.shape == (
            pattern.theta_grid_deg.size,
            pattern.phi_grid_deg.size,
        )
        assert pattern.phi_wraps is True
        # az/el fields inactive.
        assert pattern.az_grid_deg is None
        assert pattern.el_grid_deg is None
        assert pattern.grid_deg is None

    def test_itu_mask_with_step_jump_loads(self, tmp_path: Path) -> None:
        """Duplicate grid_deg + different gains → step jump, must load."""
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, _itu_style_mask_payload()))
        assert pattern.grid_deg is not None
        # Duplicate angle preserved verbatim (no automatic deduplication).
        assert float(pattern.grid_deg[2]) == 3.0
        assert float(pattern.grid_deg[3]) == 3.0
        # Two different gains at the duplicated angle → step.
        assert float(pattern.gain_db[2]) == 20.0
        assert float(pattern.gain_db[3]) == 0.0


class TestDefaults:
    """Optional fields fall back to the documented defaults."""

    def test_grid_mode_defaults_to_azel(self, tmp_path: Path) -> None:
        payload = _minimal_2d_azel_payload()
        del payload["grid_mode"]
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        assert pattern.grid_mode == ca.GRID_MODE_AZEL

    def test_az_wraps_defaults_to_true(self, tmp_path: Path) -> None:
        payload = _minimal_2d_azel_payload()
        del payload["az_wraps"]
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        assert pattern.az_wraps is True

    def test_phi_wraps_defaults_to_true(self, tmp_path: Path) -> None:
        payload = _minimal_2d_thetaphi_payload()
        del payload["phi_wraps"]
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        assert pattern.phi_wraps is True

    def test_meta_defaults_to_empty_dict(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        del payload["meta"]
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        assert pattern.meta == {}


# ===========================================================================
# Round-trip: load → dump → load
# ===========================================================================


class TestRoundTrip:

    @pytest.mark.parametrize(
        "payload_fn",
        [
            _minimal_1d_payload,
            _minimal_2d_azel_payload,
            _minimal_2d_thetaphi_payload,
            _itu_style_mask_payload,
        ],
        ids=["1d", "2d_azel", "2d_thetaphi", "itu_mask"],
    )
    def test_load_dump_load_preserves_all_fields(
        self, tmp_path: Path, payload_fn: Any
    ) -> None:
        p_in = _write_payload(tmp_path, payload_fn(), name="in.json")
        first = ca.load_custom_pattern(p_in)

        p_out = tmp_path / "out.json"
        ca.dump_custom_pattern(p_out, first)
        second = ca.load_custom_pattern(p_out)

        # Scalar envelope
        assert first.format_version == second.format_version
        assert first.kind == second.kind
        assert first.normalisation == second.normalisation
        assert first.peak_gain_source == second.peak_gain_source
        assert first.peak_gain_dbi == second.peak_gain_dbi
        assert dict(first.meta) == dict(second.meta)
        # 2-D envelope
        assert first.grid_mode == second.grid_mode
        assert first.az_wraps == second.az_wraps
        assert first.phi_wraps == second.phi_wraps
        # Arrays (every array field is either both-None or element-equal)
        for name in (
            "gain_db",
            "grid_deg",
            "az_grid_deg",
            "el_grid_deg",
            "theta_grid_deg",
            "phi_grid_deg",
        ):
            a1 = getattr(first, name)
            a2 = getattr(second, name)
            if a1 is None:
                assert a2 is None, f"{name} must be None after round-trip"
            else:
                np.testing.assert_array_equal(a1, a2)

    def test_dump_is_byte_stable(self, tmp_path: Path) -> None:
        """Two successive dumps of the same pattern produce identical bytes."""
        pattern = ca.load_custom_pattern(
            _write_payload(tmp_path, _minimal_1d_payload())
        )
        out1 = tmp_path / "a.json"
        out2 = tmp_path / "b.json"
        ca.dump_custom_pattern(out1, pattern)
        ca.dump_custom_pattern(out2, pattern)
        assert out1.read_bytes() == out2.read_bytes()

    def test_dump_preserves_unknown_meta_keys(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["meta"] = {
            "title": "T",
            "reference": "R",
            "custom_vendor_field": "hidden value",
            "producer_tool": "tool-X v1.2",
        }
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        out = tmp_path / "out.json"
        ca.dump_custom_pattern(out, pattern)
        dumped = json.loads(out.read_text(encoding="utf-8"))
        assert dumped["meta"]["custom_vendor_field"] == "hidden value"
        assert dumped["meta"]["producer_tool"] == "tool-X v1.2"


# ===========================================================================
# Negative: validation errors
# ===========================================================================


class TestValidationErrors:

    def test_invalid_json_is_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("this is not json at all", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            ca.load_custom_pattern(p)

    def test_top_level_array_is_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "arr.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level JSON"):
            ca.load_custom_pattern(p)

    def test_wrong_format_version(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["scepter_antenna_pattern_format"] = "v99"
        with pytest.raises(ValueError, match="scepter_antenna_pattern_format"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_missing_format_version(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        del payload["scepter_antenna_pattern_format"]
        with pytest.raises(ValueError, match="scepter_antenna_pattern_format"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_unknown_kind(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["kind"] = "spherical_harmonics"
        with pytest.raises(ValueError, match="`kind`"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_unknown_grid_mode(self, tmp_path: Path) -> None:
        payload = _minimal_2d_azel_payload()
        payload["grid_mode"] = "xyz"
        with pytest.raises(ValueError, match="grid_mode"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_unknown_normalisation(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["normalisation"] = "logarithmic"
        with pytest.raises(ValueError, match="normalisation"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_unknown_peak_source(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["peak_gain_source"] = "implicit"
        with pytest.raises(ValueError, match="peak_gain_source"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_grid_not_monotonic(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        # Insert a decreasing step: 5 → 3
        payload["grid_deg"] = [0.0, 1.0, 5.0, 3.0, 180.0]
        payload["gain_db"] = [40.0, 30.0, 20.0, 10.0, -70.0]
        with pytest.raises(ValueError, match="non-decreasing"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_triple_duplicate_grid_value(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["grid_deg"] = [0.0, 3.0, 3.0, 3.0, 180.0]
        payload["gain_db"] = [38.0, 20.0, 10.0, 0.0, -70.0]
        with pytest.raises(ValueError, match="3 or more consecutive duplicate"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_double_duplicate_is_accepted(self, tmp_path: Path) -> None:
        """Counter-test: exactly two duplicates = step jump, accepted."""
        payload = _minimal_1d_payload()
        payload["grid_deg"] = [0.0, 3.0, 3.0, 180.0]
        payload["gain_db"] = [38.0, 20.0, 0.0, -30.0]
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        assert pattern.grid_deg is not None
        assert pattern.grid_deg.size == 4

    def test_1d_grid_does_not_start_at_zero(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["grid_deg"][0] = 0.5
        with pytest.raises(ValueError, match=r"grid_deg\[0\]"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_1d_grid_does_not_reach_180(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["grid_deg"][-1] = 150.0
        with pytest.raises(ValueError, match=r"grid_deg\[-1\]"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_1d_grid_gain_size_mismatch(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["gain_db"] = payload["gain_db"][:-1]
        with pytest.raises(ValueError, match="they must match"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_2d_azel_gain_shape_mismatch(self, tmp_path: Path) -> None:
        payload = _minimal_2d_azel_payload()
        # Drop a column — shape (N_az, N_el-1) instead of (N_az, N_el)
        payload["gain_db"] = [row[:-1] for row in payload["gain_db"]]
        with pytest.raises(ValueError, match="shape"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_2d_az_grid_does_not_cover_minus_180(self, tmp_path: Path) -> None:
        payload = _minimal_2d_azel_payload()
        payload["az_grid_deg"][0] = -170.0
        with pytest.raises(ValueError, match=r"az_grid_deg\[0\]"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_2d_az_grid_does_not_cover_plus_180(self, tmp_path: Path) -> None:
        payload = _minimal_2d_azel_payload()
        payload["az_grid_deg"][-1] = 170.0
        with pytest.raises(ValueError, match=r"az_grid_deg\[-1\]"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_2d_el_grid_does_not_cover_minus_90(self, tmp_path: Path) -> None:
        payload = _minimal_2d_azel_payload()
        payload["el_grid_deg"][0] = -80.0
        with pytest.raises(ValueError, match=r"el_grid_deg\[0\]"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_2d_thetaphi_theta_does_not_start_at_zero(self, tmp_path: Path) -> None:
        payload = _minimal_2d_thetaphi_payload()
        payload["theta_grid_deg"][0] = 0.5
        with pytest.raises(ValueError, match=r"theta_grid_deg\[0\]"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_non_finite_grid_value(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["grid_deg"][3] = float("nan")
        # json.dump with allow_nan=True emits a non-standard "NaN" literal
        # that json.load reads back; the loader catches it at the
        # finiteness check.
        p = tmp_path / "nan.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, allow_nan=True)
        with pytest.raises(ValueError, match="finite"):
            ca.load_custom_pattern(p)

    def test_non_finite_gain_value(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["gain_db"][3] = float("inf")
        p = tmp_path / "inf.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, allow_nan=True)
        with pytest.raises(ValueError, match="finite"):
            ca.load_custom_pattern(p)

    def test_peak_mismatch_over_10_db_is_refused(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        # LUT maximum is 40 dBi; claim 60 dBi → 20 dB mismatch > threshold
        payload["peak_gain_dbi"] = 60.0
        with pytest.raises(ValueError, match="exceeds the 10.0 dB"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_peak_mismatch_under_10_db_is_accepted_in_explicit_mode(
        self, tmp_path: Path
    ) -> None:
        """ITU-mask style: explicit peak above any tabulated sample, < 10 dB."""
        payload = _minimal_1d_payload()
        payload["peak_gain_dbi"] = 46.0  # 6 dB above the LUT max — regulatory-mask like
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        assert pattern.peak_gain_dbi == 46.0

    def test_relative_with_positive_values(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["normalisation"] = "relative"
        # gain_db still has the absolute 40 dBi at the peak — invalid for relative
        with pytest.raises(ValueError, match="relative"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_relative_with_proper_normalisation_loads(self, tmp_path: Path) -> None:
        """Counter-test: re-reference to peak works."""
        payload = _minimal_1d_payload()
        payload["normalisation"] = "relative"
        peak = max(payload["gain_db"])
        payload["gain_db"] = [v - peak for v in payload["gain_db"]]
        # peak_gain_dbi already matches the original dBi peak
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        assert pattern.normalisation == ca.NORMALISATION_RELATIVE
        assert float(np.max(pattern.gain_db)) == pytest.approx(0.0, abs=1e-9)

    def test_meta_must_be_object(self, tmp_path: Path) -> None:
        payload = _minimal_1d_payload()
        payload["meta"] = ["title", "reference"]
        with pytest.raises(ValueError, match="`meta` must be an object"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_az_wraps_must_be_bool(self, tmp_path: Path) -> None:
        payload = _minimal_2d_azel_payload()
        payload["az_wraps"] = "yes"
        with pytest.raises(ValueError, match="az_wraps"):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))


# ===========================================================================
# Warnings
# ===========================================================================


class TestPeakPatternWarnings:

    def test_lut_source_with_mild_mismatch_warns(self, tmp_path: Path) -> None:
        """peak_source=lut with a 5 dB mismatch emits PatternPeakWarning."""
        payload = _minimal_1d_payload()
        payload["peak_gain_source"] = "lut"
        payload["peak_gain_dbi"] = 45.0  # LUT max 40 → 5 dB mismatch, inside the warn band
        with pytest.warns(ca.PatternPeakWarning):
            ca.load_custom_pattern(_write_payload(tmp_path, payload))

    def test_explicit_source_with_mild_mismatch_does_not_warn(
        self, tmp_path: Path, recwarn: pytest.WarningsRecorder
    ) -> None:
        payload = _minimal_1d_payload()
        payload["peak_gain_source"] = "explicit"
        payload["peak_gain_dbi"] = 46.0  # 6 dB mismatch — inside warn band but explicit
        ca.load_custom_pattern(_write_payload(tmp_path, payload))
        peak_warnings = [w for w in recwarn.list if issubclass(w.category, ca.PatternPeakWarning)]
        assert not peak_warnings, "explicit mode must not emit PatternPeakWarning"

    def test_lut_source_with_tiny_mismatch_does_not_warn(
        self, tmp_path: Path, recwarn: pytest.WarningsRecorder
    ) -> None:
        """peak_source=lut with a sub-0.5 dB mismatch stays quiet."""
        payload = _minimal_1d_payload()
        payload["peak_gain_source"] = "lut"
        payload["peak_gain_dbi"] = 40.1  # 0.1 dB mismatch — below warn threshold
        ca.load_custom_pattern(_write_payload(tmp_path, payload))
        peak_warnings = [w for w in recwarn.list if issubclass(w.category, ca.PatternPeakWarning)]
        assert not peak_warnings


# ===========================================================================
# Path error paths
# ===========================================================================


class TestIOErrors:

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cannot read"):
            ca.load_custom_pattern(tmp_path / "does_not_exist.json")


# ===========================================================================
# In-memory dict round-trip (for embedding inside project JSON — Stage 3)
# ===========================================================================


class TestFormatSummary:
    """``format_pattern_summary`` produces a clean, informative text block."""

    def test_1d_summary_headers_and_fields(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, _minimal_1d_payload()))
        out = ca.format_pattern_summary(pattern, path=tmp_path / "p.json")
        # Human-facing header always names the file.
        assert "SCEPTer Custom Antenna Pattern" in out
        assert "p.json" in out
        # Key envelope fields visible.
        assert "Format version" in out
        assert "v1" in out
        assert "1d_axisymmetric" in out
        assert "axisymmetric" in out
        assert "40.000 dBi" in out
        # Grid + gain sections.
        assert "grid_deg" in out
        assert "Tabulated max" in out
        assert "Tabulated min" in out
        assert "Dynamic range" in out

    def test_2d_azel_summary_includes_grid_mode_and_wrap_flag(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(
            _write_payload(tmp_path, _minimal_2d_azel_payload())
        )
        out = ca.format_pattern_summary(pattern)
        assert "grid_mode = az_el" in out
        assert "az_grid_deg" in out
        assert "el_grid_deg" in out
        assert "az_wraps" in out
        assert "True" in out

    def test_2d_thetaphi_summary(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(
            _write_payload(tmp_path, _minimal_2d_thetaphi_payload())
        )
        out = ca.format_pattern_summary(pattern)
        assert "grid_mode = theta_phi" in out
        assert "theta_grid_deg" in out
        assert "phi_grid_deg" in out
        assert "phi_wraps" in out

    def test_summary_reports_step_discontinuity(self, tmp_path: Path) -> None:
        """ITU-style mask with a step at 3° shows up in the sanity observations."""
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, _itu_style_mask_payload()))
        out = ca.format_pattern_summary(pattern)
        assert "Step discontinuity" in out
        assert "3.000°" in out

    def test_summary_reports_peak_vs_lut_delta_in_absolute_mode(
        self, tmp_path: Path
    ) -> None:
        payload = _minimal_1d_payload()
        payload["peak_gain_dbi"] = 46.0  # 6 dB above tabulated max (mild)
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        out = ca.format_pattern_summary(pattern)
        assert "Peak vs LUT max" in out
        assert "6.000 dB" in out
        assert "mild" in out.lower()

    def test_summary_no_observations_for_clean_pattern(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, _minimal_1d_payload()))
        out = ca.format_pattern_summary(pattern)
        # The clean 1-D example has no step discontinuities and no wrap boundary.
        assert "(none — pattern is clean)" in out

    def test_summary_detects_wrap_boundary_mismatch(self, tmp_path: Path) -> None:
        """az_wraps=True with inconsistent boundary rows → human-readable warning."""
        payload = _minimal_2d_azel_payload()
        # Perturb only the +180° column so the wrap boundary disagrees by
        # a couple of dB (still legal, just worth flagging).
        payload["gain_db"][-1] = [
            row + 5.0 for row in payload["gain_db"][-1]
        ]
        # Keep peak_gain_dbi consistent (LUT max still 34.5 at (0,0))
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload))
        out = ca.format_pattern_summary(pattern)
        assert "az_wraps=True but the ±180° boundary values differ" in out


class TestCli:
    """The ``python -m scepter.custom_antenna inspect`` entry point."""

    def test_inspect_ok_path_prints_summary_and_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        p = _write_payload(tmp_path, _minimal_1d_payload())
        rc = ca.main(["inspect", str(p)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "SCEPTer Custom Antenna Pattern" in captured.out
        assert "Tabulated max" in captured.out
        assert captured.err == ""

    def test_inspect_bad_file_returns_nonzero_and_writes_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        payload = _minimal_1d_payload()
        payload["scepter_antenna_pattern_format"] = "v999"
        p = _write_payload(tmp_path, payload)
        rc = ca.main(["inspect", str(p)])
        assert rc == 2
        captured = capsys.readouterr()
        # Error goes to stderr, stdout stays quiet.
        assert "error:" in captured.err
        assert "scepter_antenna_pattern_format" in captured.err
        assert captured.out == ""

    def test_inspect_missing_file_returns_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = ca.main(["inspect", str(tmp_path / "nope.json")])
        assert rc == 2
        captured = capsys.readouterr()
        assert "cannot read" in captured.err

    def test_cli_without_subcommand_prints_help_and_returns_nonzero(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        rc = ca.main([])
        assert rc != 0
        captured = capsys.readouterr()
        # argparse writes help to stdout in this mode.
        assert "inspect" in captured.out.lower()


class TestCliViaSubprocess:
    """Smoke test via real ``python -m scepter.custom_antenna`` invocation."""

    def test_python_dash_m_invocation(self, tmp_path: Path) -> None:
        """Exercise the ``if __name__ == '__main__'`` block."""
        import subprocess
        import sys

        p = _write_payload(tmp_path, _minimal_1d_payload())
        result = subprocess.run(
            [sys.executable, "-m", "scepter.custom_antenna", "inspect", str(p)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "SCEPTer Custom Antenna Pattern" in result.stdout


class TestInMemoryJsonDict:
    """``to_json_dict`` / ``from_json_dict`` mirror the file-based loader.

    These methods let the GUI project-state save/load embed a pattern
    *inline* into the larger project JSON without bouncing through a
    temporary file. They must apply exactly the same validation as
    ``load_custom_pattern``.
    """

    @pytest.mark.parametrize(
        "payload_fn",
        [
            _minimal_1d_payload,
            _minimal_2d_azel_payload,
            _minimal_2d_thetaphi_payload,
            _itu_style_mask_payload,
        ],
        ids=["1d", "2d_azel", "2d_thetaphi", "itu_mask"],
    )
    def test_json_dict_round_trip(self, payload_fn: Any) -> None:
        pattern = ca.CustomAntennaPattern.from_json_dict(payload_fn())
        dumped = pattern.to_json_dict()
        restored = ca.CustomAntennaPattern.from_json_dict(dumped)
        for name in (
            "format_version",
            "kind",
            "normalisation",
            "peak_gain_source",
            "peak_gain_dbi",
            "grid_mode",
            "az_wraps",
            "phi_wraps",
        ):
            assert getattr(pattern, name) == getattr(restored, name), name
        assert dict(pattern.meta) == dict(restored.meta)
        for name in (
            "gain_db",
            "grid_deg",
            "az_grid_deg",
            "el_grid_deg",
            "theta_grid_deg",
            "phi_grid_deg",
        ):
            a1 = getattr(pattern, name)
            a2 = getattr(restored, name)
            if a1 is None:
                assert a2 is None
            else:
                np.testing.assert_array_equal(a1, a2)

    def test_from_json_dict_accepts_mapping(self) -> None:
        """Callers handing a proxy / MappingProxyType instead of a plain dict still work."""
        from types import MappingProxyType

        proxy = MappingProxyType(_minimal_1d_payload())
        pattern = ca.CustomAntennaPattern.from_json_dict(proxy)
        assert pattern.peak_gain_dbi == 40.0

    def test_from_json_dict_reuses_validation(self) -> None:
        """Schema violations fail here just like in load_custom_pattern."""
        payload = _minimal_1d_payload()
        payload["scepter_antenna_pattern_format"] = "v42"
        with pytest.raises(ValueError, match="scepter_antenna_pattern_format"):
            ca.CustomAntennaPattern.from_json_dict(payload)


# ===========================================================================
# Stage 5 — CPU 1-D interpolation kernel
# ===========================================================================


def _ra1631_analytical_dbi(
    phi_deg: np.ndarray,
    *,
    d_over_lambda: float = 166.67,  # 35 m dish at ~20 cm wavelength
    eta_a: float = 0.7,
) -> np.ndarray:
    """Reference ITU-R RA.1631 axisymmetric pattern.

    Pure-NumPy implementation used to validate the 1-D interpolator;
    structurally identical to ``_evaluate_ras_pattern_cp`` in
    ``gpu_accel.py`` but CuPy-free so it runs on the lite env. Defined
    on phi ∈ [0°, 180°] — callers pass non-negative angles.
    """
    phi = np.asarray(phi_deg, dtype=np.float64)
    gmax = 10.0 * np.log10(eta_a * (np.pi * d_over_lambda) ** 2)
    g1 = -1.0 + 15.0 * np.log10(d_over_lambda)
    phi_m = (20.0 / d_over_lambda) * np.sqrt(gmax - g1)
    phi_r = 15.85 * d_over_lambda ** -0.6
    # Avoid log(0) in the sidelobe regions — any φ that falls there
    # is by construction > phi_r > 0.
    phi_safe = np.clip(phi, 1e-9, None)
    main = phi < phi_m
    transition = (phi >= phi_m) & (phi < phi_r)
    side1 = (phi >= phi_r) & (phi < 10.0)
    side2 = (phi >= 10.0) & (phi < 34.1)
    side3 = (phi >= 34.1) & (phi < 80.0)
    side4 = (phi >= 80.0) & (phi < 120.0)
    backlobe = phi >= 120.0
    out = np.zeros_like(phi)
    out[main] = gmax - 2.5e-3 * (d_over_lambda * phi[main]) ** 2
    out[transition] = g1
    out[side1] = 29.0 - 25.0 * np.log10(phi_safe[side1])
    out[side2] = 34.0 - 30.0 * np.log10(phi_safe[side2])
    out[side3] = -12.0
    out[side4] = -7.0
    out[backlobe] = -12.0
    return out


def _build_pattern_1d(
    grid: np.ndarray,
    gain: np.ndarray,
    *,
    normalisation: str = "absolute",
    peak_source: str = "explicit",
    peak_gain_dbi: float | None = None,
) -> "ca.CustomAntennaPattern":
    """Test helper: build a validated 1-D pattern from raw grid + gain arrays."""
    if peak_gain_dbi is None:
        peak_gain_dbi = float(np.max(gain))
    payload = {
        "scepter_antenna_pattern_format": "v1",
        "kind": "1d_axisymmetric",
        "normalisation": normalisation,
        "peak_gain_source": peak_source,
        "peak_gain_dbi": peak_gain_dbi,
        "grid_deg": list(map(float, grid)),
        "gain_db": list(map(float, gain)),
    }
    return ca.CustomAntennaPattern.from_json_dict(payload)


class TestIsAxisymmetricProperty:

    def test_1d_is_axisymmetric(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, _minimal_1d_payload()))
        assert pattern.is_axisymmetric is True

    def test_2d_is_not_axisymmetric(self, tmp_path: Path) -> None:
        for payload_fn in (_minimal_2d_azel_payload, _minimal_2d_thetaphi_payload):
            pattern = ca.load_custom_pattern(_write_payload(tmp_path, payload_fn()))
            assert pattern.is_axisymmetric is False


class TestEvaluatePattern1DBasic:

    def test_exact_grid_points_return_stored_values(self) -> None:
        grid = np.array([0.0, 1.0, 5.0, 30.0, 180.0])
        gain = np.array([40.0, 35.0, 10.0, -20.0, -50.0])
        pattern = _build_pattern_1d(grid, gain)
        out = ca.evaluate_pattern_1d(pattern, grid)
        np.testing.assert_allclose(out, gain, atol=1e-12)

    def test_linear_interpolation_is_exact_between_samples(self) -> None:
        """Input is a straight line in (γ, dB); the interpolator must be
        exact at any sub-grid query point."""
        grid = np.linspace(0.0, 180.0, 19)  # 10° step
        gain = 40.0 - 0.5 * grid  # strictly monotonic linear decay
        pattern = _build_pattern_1d(grid, gain)
        # Pick queries well inside the domain, not on grid nodes.
        queries = np.array([1.7, 2.5, 37.5, 87.5, 172.5])
        expected = 40.0 - 0.5 * queries
        actual = ca.evaluate_pattern_1d(pattern, queries)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_scalar_input_returns_scalar_shape(self) -> None:
        pattern = _build_pattern_1d(
            np.array([0.0, 90.0, 180.0]), np.array([30.0, 0.0, -30.0])
        )
        out = ca.evaluate_pattern_1d(pattern, 45.0)
        assert np.asarray(out).shape == ()
        assert float(out) == pytest.approx(15.0, abs=1e-12)

    def test_multidim_input_preserved(self) -> None:
        pattern = _build_pattern_1d(
            np.array([0.0, 90.0, 180.0]), np.array([30.0, 0.0, -30.0])
        )
        out = ca.evaluate_pattern_1d(pattern, np.array([[45.0, 135.0], [0.0, 180.0]]))
        assert out.shape == (2, 2)
        np.testing.assert_allclose(
            out, np.array([[15.0, -15.0], [30.0, -30.0]]), atol=1e-12
        )

    def test_clipping_below_grid_returns_boundary(self) -> None:
        grid = np.array([0.0, 5.0, 180.0])
        gain = np.array([40.0, 20.0, -30.0])
        pattern = _build_pattern_1d(grid, gain)
        assert float(ca.evaluate_pattern_1d(pattern, -10.0)) == 40.0
        assert float(ca.evaluate_pattern_1d(pattern, -1e-6)) == pytest.approx(40.0, abs=1e-9)

    def test_clipping_above_grid_returns_boundary(self) -> None:
        grid = np.array([0.0, 5.0, 180.0])
        gain = np.array([40.0, 20.0, -30.0])
        pattern = _build_pattern_1d(grid, gain)
        assert float(ca.evaluate_pattern_1d(pattern, 1234.0)) == -30.0

    def test_wrong_kind_raises(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, _minimal_2d_azel_payload()))
        with pytest.raises(ValueError, match="requires kind"):
            ca.evaluate_pattern_1d(pattern, 30.0)


class TestEvaluatePattern1DStepAndPlateau:
    """Step discontinuities + flat plateaus are the schema's distinctive
    features — worth verifying every expected behaviour."""

    def test_flat_plateau_between_equal_gains(self) -> None:
        """Two equal consecutive gains → any query inside returns that gain."""
        grid = np.array([0.0, 5.0, 10.0, 180.0])
        gain = np.array([30.0, -15.0, -15.0, -30.0])  # flat at -15 from 5° to 10°
        pattern = _build_pattern_1d(grid, gain)
        for q in [5.0, 6.0, 7.5, 9.999, 10.0]:
            assert float(ca.evaluate_pattern_1d(pattern, q)) == pytest.approx(-15.0, abs=1e-9)

    def test_step_right_continuous_at_duplicate(self) -> None:
        """At the duplicated angle, the eval returns the right-hand gain."""
        # ITU-style mask-ish: flat 38 to 1°, down to 20 at 3°, jumps to 0 at 3°, down to -30 at 30°
        grid = np.array([0.0, 1.0, 3.0, 3.0, 30.0, 180.0])
        gain = np.array([38.0, 38.0, 20.0, 0.0, -30.0, -30.0])
        pattern = _build_pattern_1d(grid, gain, peak_gain_dbi=38.0)
        # At exactly 3°, right-continuous rule ⇒ the right-side value (0).
        assert float(ca.evaluate_pattern_1d(pattern, 3.0)) == pytest.approx(0.0, abs=1e-9)

    def test_step_just_before_jump_interpolates_on_left_segment(self) -> None:
        grid = np.array([0.0, 1.0, 3.0, 3.0, 30.0, 180.0])
        gain = np.array([38.0, 38.0, 20.0, 0.0, -30.0, -30.0])
        pattern = _build_pattern_1d(grid, gain, peak_gain_dbi=38.0)
        # Just left of 3°: expected ≈ 20 (end of the 1→3 linear slope).
        # Segment (1, 38) → (3, 20). At φ=3-ε, expected = 20 + O(ε).
        out = float(ca.evaluate_pattern_1d(pattern, 3.0 - 1e-6))
        assert abs(out - 20.0) < 1e-3

    def test_step_just_after_jump_interpolates_on_right_segment(self) -> None:
        grid = np.array([0.0, 1.0, 3.0, 3.0, 30.0, 180.0])
        gain = np.array([38.0, 38.0, 20.0, 0.0, -30.0, -30.0])
        pattern = _build_pattern_1d(grid, gain, peak_gain_dbi=38.0)
        # Just right of 3°: segment (3, 0) → (30, -30). At φ=3+ε, expected ≈ 0.
        out = float(ca.evaluate_pattern_1d(pattern, 3.0 + 1e-6))
        assert abs(out - 0.0) < 1e-3

    def test_step_no_interpolation_noise_across_wide_segment(self) -> None:
        """Middle of the post-jump segment still linear-interps cleanly."""
        grid = np.array([0.0, 1.0, 3.0, 3.0, 30.0, 180.0])
        gain = np.array([38.0, 38.0, 20.0, 0.0, -30.0, -30.0])
        pattern = _build_pattern_1d(grid, gain, peak_gain_dbi=38.0)
        # Midpoint of (3, 0) → (30, -30) at φ=16.5. Expected: -15.
        out = float(ca.evaluate_pattern_1d(pattern, 16.5))
        assert out == pytest.approx(-15.0, abs=1e-9)


class TestEvaluatePattern1DNormalisation:

    def test_relative_and_absolute_give_same_absolute_dbi(self) -> None:
        """Same pattern in absolute or relative form ⇒ identical evaluate output."""
        grid = np.array([0.0, 5.0, 30.0, 180.0])
        gain_abs = np.array([40.0, 25.0, -10.0, -50.0])
        gain_rel = gain_abs - 40.0  # shift so peak = 0

        pattern_abs = _build_pattern_1d(
            grid, gain_abs, normalisation="absolute", peak_gain_dbi=40.0,
        )
        pattern_rel = _build_pattern_1d(
            grid, gain_rel, normalisation="relative", peak_gain_dbi=40.0,
        )
        queries = np.array([0.0, 1.0, 7.5, 45.0, 180.0])
        out_abs = ca.evaluate_pattern_1d(pattern_abs, queries)
        out_rel = ca.evaluate_pattern_1d(pattern_rel, queries)
        np.testing.assert_allclose(out_abs, out_rel, atol=1e-12)


class TestEvaluatePattern1DQuadraticResidualBudget:
    """Residual budget on a textbook smooth function with a known bound.

    For a twice-differentiable ``f``, piecewise-linear interpolation on
    a grid of spacing ``Δ`` has max error ``≤ Δ² · max|f''| / 8``. We
    test that theoretical bound is respected, which is the *interpolator*
    guarantee. The RA.1631 test below checks an end-to-end realistic
    use case (ITU pattern + anchor encoding).
    """

    def test_quadratic_residual_matches_interpolation_theory(self) -> None:
        # f(γ) = -0.1·γ² (smooth, single-sign second derivative).
        # max|f''| = 0.2  →  budget at step Δ is (Δ²·0.2)/8.
        step = 0.1
        grid = np.arange(0.0, 180.0 + 0.5 * step, step)
        grid = np.clip(grid, 0.0, 180.0)
        grid = np.unique(grid)
        gain = -0.1 * grid ** 2  # peak 0 at γ=0
        pattern = _build_pattern_1d(grid, gain, peak_gain_dbi=0.0)

        # Query at sub-grid points (offset so we land inside each segment).
        queries = np.arange(step * 0.37, 180.0 - step * 0.37, step * 0.73)
        expected = -0.1 * queries ** 2
        actual = ca.evaluate_pattern_1d(pattern, queries)
        err = np.abs(actual - expected)

        budget = step ** 2 * 0.2 / 8.0 + 1e-12
        assert err.max() <= budget, (
            f"max residual {err.max():.2e} dB exceeds the "
            f"theoretical interp bound {budget:.2e} dB at step={step}°"
        )


class TestEvaluatePattern1DRa1631:
    """RA.1631 end-to-end: sample the analytical pattern with explicit
    anchors at its derivative kinks and step jumps, and verify the
    evaluator stays within the Stage-5 residual budget.

    RA.1631 is piecewise-defined with **C¹ discontinuities** at
    ``phi_m`` / ``phi_r`` / ``10°`` / ``34.1°`` (derivative jumps —
    the value is continuous, the slope isn't) plus two **C⁰
    discontinuities** (step jumps) at ``80°`` and ``120°``. A naive
    dense regular LUT smears each kink over one segment; linear interp
    through that segment disagrees with the analytical value by up to
    ``(Δγ/2) · |Δf'|`` which for RA.1631 can reach tenths of a dB
    even at ``Δγ=0.01°``. That's not an interpolator bug — it's the
    inherent cost of linear-interpolating a function that isn't C¹.

    The correct way to represent such a pattern as a LUT is to
    **place grid points at every kink** so each segment is smooth in
    isolation. This test does exactly that and asserts the documented
    < 0.05 dB budget.
    """

    @staticmethod
    def _ra1631_params(d_over_lambda: float = 166.67, eta_a: float = 0.7):
        gmax = 10.0 * np.log10(eta_a * (np.pi * d_over_lambda) ** 2)
        g1 = -1.0 + 15.0 * np.log10(d_over_lambda)
        phi_m = (20.0 / d_over_lambda) * np.sqrt(gmax - g1)
        phi_r = 15.85 * d_over_lambda ** -0.6
        return gmax, g1, phi_m, phi_r

    def _build_anchored_lut(self, step_deg: float, d_over_lambda: float):
        """Regular step_deg grid + anchor points at every RA.1631 kink.

        Step jumps at 80° and 120° are encoded via duplicate grid
        entries with the analytical left/right values.
        """
        gmax, g1, phi_m, phi_r = self._ra1631_params(d_over_lambda=d_over_lambda)
        base = np.arange(0.0, 180.0 + 0.5 * step_deg, step_deg)
        base = np.clip(base, 0.0, 180.0)
        # Pull in every C¹ kink angle as a grid point; the evaluator
        # then interps within each smooth sub-segment only.
        anchors_c1 = [phi_m, phi_r, 10.0, 34.1]
        grid = np.unique(np.concatenate([base, np.asarray(anchors_c1)]))
        gain = _ra1631_analytical_dbi(
            grid, d_over_lambda=d_over_lambda,
        )
        # Encode the C⁰ step jumps by duplicating 80° and 120° with
        # the correct pre-/post-jump values.
        def _insert_step(grid, gain, at, left, right):
            mask = grid != at
            grid = grid[mask]
            gain = gain[mask]
            idx = int(np.searchsorted(grid, at))
            grid = np.insert(grid, idx, [at, at])
            gain = np.insert(gain, idx, [left, right])
            return grid, gain
        grid, gain = _insert_step(grid, gain, 80.0, -12.0, -7.0)
        grid, gain = _insert_step(grid, gain, 120.0, -7.0, -12.0)
        return grid, gain

    # Step → residual budget pairs. The 0.05 dB figure from the
    # Stage-5 plan is achievable at Δ=0.01° (anchor-encoded LUT, RA.1631
    # main-beam curvature). At Δ=0.1° the same main-beam curvature
    # (|f''| ≈ 5e-3 · (D/λ)² ≈ 139 dB/deg² for the default 35-m / 20-cm
    # dish) saturates the theoretical Δ² / 8 · |f''| bound at ~0.17 dB,
    # so we gate each resolution against its own physical bound.
    @pytest.mark.parametrize("step_deg,budget_db", [(0.1, 0.20), (0.01, 0.05)])
    def test_residual_under_budget_with_anchor_encoded_lut(
        self, step_deg: float, budget_db: float
    ) -> None:
        d_over_lambda = 166.67
        grid, gain = self._build_anchored_lut(step_deg, d_over_lambda)
        pattern = _build_pattern_1d(
            grid, gain,
            peak_source="lut",
            peak_gain_dbi=float(np.max(gain)),
        )

        # Query on a dense offset grid that exercises interp within
        # every sub-segment. Skip exact step-jump angles (80 / 120)
        # so we compare against the analytical value on one side only.
        queries = np.arange(
            step_deg * 0.37, 180.0 - step_deg * 0.37, step_deg * 0.73,
        )
        # Step-jump handling is covered separately; here we compare
        # against the analytical value excluding the exact jump angles.
        keep = (np.abs(queries - 80.0) > 1e-6) & (np.abs(queries - 120.0) > 1e-6)
        queries = queries[keep]
        assert queries.size > 100

        expected = _ra1631_analytical_dbi(queries, d_over_lambda=d_over_lambda)
        actual = ca.evaluate_pattern_1d(pattern, queries)
        err = np.abs(actual - expected)
        assert err.max() <= budget_db, (
            f"max residual {err.max():.4f} dB exceeds {budget_db} dB "
            f"budget at step={step_deg}° (anchor-encoded LUT)"
        )

    def test_ra1631_with_encoded_step_jumps_matches_exact_values(self) -> None:
        """If the user encodes the step jumps as duplicate grid points,
        the evaluator returns the analytical left/right values exactly."""
        # Build a minimal RA.1631 LUT that includes explicit step
        # anchors at 80° and 120°. Resolution doesn't matter much —
        # we just care about the discrete-sample correctness.
        grid = np.array([
            0.0, 0.01, 0.1, 1.0, 10.0, 34.1, 50.0,
            80.0, 80.0,      # step: −12 → −7
            100.0,
            120.0, 120.0,    # step: −7 → −12
            150.0, 180.0,
        ])
        gain_raw = _ra1631_analytical_dbi(grid)
        # Override the duplicated rows so left-side is the pre-jump
        # value (−12 at 80, −7 at 120) and right-side is the
        # post-jump value (−7 at 80, −12 at 120). The analytical
        # helper places φ=80° into the phi >= 80 branch (−7), so we
        # force the left value manually.
        # Index layout after the duplicates are inserted in the
        # ``grid`` list above:
        #   grid[7] = 80.0 (left of jump)
        #   grid[8] = 80.0 (right of jump)
        #   grid[10] = 120.0 (left of jump)
        #   grid[11] = 120.0 (right of jump)
        gain = gain_raw.copy()
        gain[7] = -12.0
        gain[8] = -7.0
        gain[10] = -7.0
        gain[11] = -12.0
        pattern = _build_pattern_1d(
            grid, gain, peak_source="lut", peak_gain_dbi=float(np.max(gain)),
        )
        # At φ = 80° exactly: right-continuous = −7
        assert float(ca.evaluate_pattern_1d(pattern, 80.0)) == pytest.approx(-7.0)
        # Just left of 80°: inside the (50, 80) segment, close to −12
        # (which is both endpoints' gain, so interp is flat −12).
        assert float(ca.evaluate_pattern_1d(pattern, 79.9999)) == pytest.approx(-12.0)
        # At φ = 120° exactly: right-continuous = −12
        assert float(ca.evaluate_pattern_1d(pattern, 120.0)) == pytest.approx(-12.0)
        # Just left of 120°: inside the (100, 120) segment, close to −7.
        assert float(ca.evaluate_pattern_1d(pattern, 119.9999)) == pytest.approx(-7.0)


# ===========================================================================
# Stage 6 — CPU 2-D bilinear interpolation kernel
# ===========================================================================


def _build_pattern_2d_azel(
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    gain: np.ndarray,
    *,
    az_wraps: bool = True,
    normalisation: str = "absolute",
    peak_source: str = "explicit",
    peak_gain_dbi: float | None = None,
) -> "ca.CustomAntennaPattern":
    if peak_gain_dbi is None:
        peak_gain_dbi = float(np.max(gain))
    payload = {
        "scepter_antenna_pattern_format": "v1",
        "kind": "2d",
        "grid_mode": "az_el",
        "normalisation": normalisation,
        "peak_gain_source": peak_source,
        "peak_gain_dbi": peak_gain_dbi,
        "az_wraps": az_wraps,
        "az_grid_deg": [float(v) for v in az_grid],
        "el_grid_deg": [float(v) for v in el_grid],
        "gain_db": [[float(v) for v in row] for row in gain],
    }
    return ca.CustomAntennaPattern.from_json_dict(payload)


def _build_pattern_2d_thetaphi(
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    gain: np.ndarray,
    *,
    phi_wraps: bool = True,
    normalisation: str = "absolute",
    peak_source: str = "explicit",
    peak_gain_dbi: float | None = None,
) -> "ca.CustomAntennaPattern":
    if peak_gain_dbi is None:
        peak_gain_dbi = float(np.max(gain))
    payload = {
        "scepter_antenna_pattern_format": "v1",
        "kind": "2d",
        "grid_mode": "theta_phi",
        "normalisation": normalisation,
        "peak_gain_source": peak_source,
        "peak_gain_dbi": peak_gain_dbi,
        "phi_wraps": phi_wraps,
        "theta_grid_deg": [float(v) for v in theta_grid],
        "phi_grid_deg": [float(v) for v in phi_grid],
        "gain_db": [[float(v) for v in row] for row in gain],
    }
    return ca.CustomAntennaPattern.from_json_dict(payload)


def _anisotropic_gaussian_db(
    axis0_deg: np.ndarray,
    axis1_deg: np.ndarray,
    *,
    a_deg: float = 5.0,
    b_deg: float = 15.0,
) -> np.ndarray:
    """Smooth anisotropic Gaussian (peak 0 dB at origin).

    Used as a stand-in for a realistic ``l_t ≠ l_r`` antenna pattern:
    smooth everywhere (no derivative kinks or step jumps), with
    different curvature along the two axes so bilinear interpolation
    on a regular grid exercises both directions.
    """
    return (
        -10.0 * np.log10(np.e)
        * ((np.asarray(axis0_deg) / a_deg) ** 2 + (np.asarray(axis1_deg) / b_deg) ** 2)
    )


class TestEvaluatePattern2DBasic:
    """Interpolator correctness on canonical inputs."""

    def test_grid_points_return_stored_values(self) -> None:
        az = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])
        el = np.array([-90.0, 0.0, 90.0])
        gain = np.arange(az.size * el.size, dtype=np.float64).reshape(az.size, el.size)
        pattern = _build_pattern_2d_azel(az, el, gain)
        for i, a in enumerate(az):
            for j, e in enumerate(el):
                out = float(ca.evaluate_pattern_2d(pattern, a, e))
                assert out == pytest.approx(gain[i, j], abs=1e-12)

    def test_bilinear_is_exact_on_linear_function(self) -> None:
        """z = a + b·az + c·el is reproduced exactly at any sub-cell query."""
        az = np.linspace(-180.0, 180.0, 13)
        el = np.linspace(-90.0, 90.0, 7)
        AZ, EL = np.meshgrid(az, el, indexing="ij")
        gain = 2.0 + 0.01 * AZ - 0.03 * EL
        pattern = _build_pattern_2d_azel(az, el, gain)

        queries_az = np.array([-150.5, -10.3, 0.0, 37.7, 120.0, 179.9])
        queries_el = np.array([-60.0, -12.5, 0.0, 15.7, 45.0, 85.0])
        AZ_q, EL_q = np.meshgrid(queries_az, queries_el, indexing="ij")
        expected = 2.0 + 0.01 * AZ_q - 0.03 * EL_q
        actual = ca.evaluate_pattern_2d(pattern, AZ_q, EL_q)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_cell_midpoint_returns_four_corner_average(self) -> None:
        """Standard bilinear sanity: at (0.5, 0.5) fractional = mean of 4 corners."""
        az = np.array([-180.0, 0.0, 180.0])
        el = np.array([-90.0, 0.0, 90.0])
        gain = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
            ]
        )
        pattern = _build_pattern_2d_azel(az, el, gain, peak_gain_dbi=32.0)
        # Cell midpoint between (0, 0) and (180, 90) — gain values
        # (21.0, 31.0, 22.0, 32.0) → mean 26.5.
        out = float(ca.evaluate_pattern_2d(pattern, 90.0, 45.0))
        assert out == pytest.approx(26.5, abs=1e-12)

    def test_scalar_input_returns_scalar_shape(self) -> None:
        pattern = _build_pattern_2d_azel(
            np.array([-180.0, 0.0, 180.0]),
            np.array([-90.0, 0.0, 90.0]),
            np.zeros((3, 3)),
            peak_gain_dbi=0.0,
        )
        out = ca.evaluate_pattern_2d(pattern, 45.0, 30.0)
        assert np.asarray(out).shape == ()
        assert float(out) == 0.0

    def test_broadcasting_scalar_and_array(self) -> None:
        pattern = _build_pattern_2d_azel(
            np.array([-180.0, 0.0, 180.0]),
            np.array([-90.0, 0.0, 90.0]),
            np.zeros((3, 3)),
            peak_gain_dbi=0.0,
        )
        # scalar axis0, array axis1 → array-shaped output matching axis1
        out = ca.evaluate_pattern_2d(pattern, 0.0, np.array([-45.0, 0.0, 45.0]))
        assert out.shape == (3,)
        np.testing.assert_allclose(out, np.zeros(3), atol=1e-12)

    def test_wrong_kind_raises(self, tmp_path: Path) -> None:
        pattern = ca.load_custom_pattern(_write_payload(tmp_path, _minimal_1d_payload()))
        with pytest.raises(ValueError, match="requires kind"):
            ca.evaluate_pattern_2d(pattern, 0.0, 0.0)


class TestEvaluatePattern2DBoundaryHandling:
    """Wrap-vs-clip behaviour on each axis."""

    def test_el_axis_always_clips(self) -> None:
        az = np.array([-180.0, 0.0, 180.0])
        el = np.array([-90.0, 0.0, 90.0])
        gain = np.array(
            [
                [-10.0, 0.0, 10.0],
                [-10.0, 0.0, 10.0],
                [-10.0, 0.0, 10.0],
            ]
        )
        pattern = _build_pattern_2d_azel(az, el, gain, peak_gain_dbi=10.0)
        # Query at el = 150° (outside [-90, 90]) must clip to el = 90°
        # → gain 10.0. And el = -999° → clip to -90° → gain -10.0.
        assert float(ca.evaluate_pattern_2d(pattern, 0.0, 150.0)) == 10.0
        assert float(ca.evaluate_pattern_2d(pattern, 0.0, -999.0)) == -10.0

    def test_az_wraps_true_brings_out_of_range_into_grid(self) -> None:
        az = np.array([-180.0, 0.0, 180.0])
        el = np.array([-90.0, 0.0, 90.0])
        # Gain depends only on az: at -180 → 0, at 0 → 10, at 180 → 0.
        gain = np.tile(np.array([0.0, 10.0, 0.0])[:, None], (1, 3))
        pattern = _build_pattern_2d_azel(
            az, el, gain, az_wraps=True, peak_gain_dbi=10.0,
        )
        # Query at az = 540° (= 540 - 360 = 180, same as +180°) → 0.
        assert float(ca.evaluate_pattern_2d(pattern, 540.0, 0.0)) == pytest.approx(0.0)
        # Query at az = -200° (wraps to +160°) → interp between az=0
        # (gain 10) and az=180 (gain 0), fraction = 160/180 ≈ 0.889.
        out = float(ca.evaluate_pattern_2d(pattern, -200.0, 0.0))
        assert out == pytest.approx(10.0 * (1.0 - 160.0 / 180.0), abs=1e-9)

    def test_az_wraps_false_clips(self) -> None:
        az = np.array([-180.0, 0.0, 180.0])
        el = np.array([-90.0, 0.0, 90.0])
        gain = np.tile(np.array([0.0, 10.0, 0.0])[:, None], (1, 3))
        pattern = _build_pattern_2d_azel(
            az, el, gain, az_wraps=False, peak_gain_dbi=10.0,
        )
        # Query at az = 540° must clip to az = 180° → gain 0.
        assert float(ca.evaluate_pattern_2d(pattern, 540.0, 0.0)) == pytest.approx(0.0)
        # Query at az = -999° must clip to az = -180° → gain 0.
        assert float(ca.evaluate_pattern_2d(pattern, -999.0, 0.0)) == pytest.approx(0.0)

    def test_boundary_endpoint_returns_tabulated_value_not_wrapped(self) -> None:
        """Query exactly at +180° must return g[-1], not g[0]."""
        az = np.array([-180.0, 0.0, 180.0])
        el = np.array([-90.0, 0.0, 90.0])
        # Distinct values at -180 and +180 to catch the wrap bug.
        gain = np.array(
            [
                [-1.0, -1.0, -1.0],
                [10.0, 10.0, 10.0],
                [99.0, 99.0, 99.0],
            ]
        )
        pattern = _build_pattern_2d_azel(
            az, el, gain, az_wraps=True, peak_gain_dbi=99.0,
        )
        assert float(ca.evaluate_pattern_2d(pattern, 180.0, 0.0)) == 99.0
        assert float(ca.evaluate_pattern_2d(pattern, -180.0, 0.0)) == -1.0


class TestEvaluatePattern2DStepAndPlateau:
    """Step discontinuities and flat plateaus along either axis."""

    def test_step_along_axis0_is_right_continuous(self) -> None:
        """Duplicate az value encodes a step; query at the jump returns right-side."""
        az = np.array([-180.0, 0.0, 0.0, 180.0])
        el = np.array([-90.0, 0.0, 90.0])
        # Row 1 (az=0 left) vs row 2 (az=0 right) differ.
        gain = np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 20.0, 20.0],   # at az=0 just-left-of-jump
                [-20.0, -20.0, -20.0],  # at az=0 just-right-of-jump
                [0.0, 0.0, 0.0],
            ]
        )
        pattern = _build_pattern_2d_azel(
            az, el, gain, az_wraps=False, peak_gain_dbi=20.0,
        )
        # Exactly at az=0 → right-continuous = -20.
        assert float(ca.evaluate_pattern_2d(pattern, 0.0, 0.0)) == pytest.approx(-20.0)
        # Just-left-of-jump: close to +20 (on the (-180, 0) segment).
        out_left = float(ca.evaluate_pattern_2d(pattern, -1e-6, 0.0))
        assert abs(out_left - 20.0) < 1e-3
        # Just-right-of-jump: close to -20 (on the (0, 180) segment).
        out_right = float(ca.evaluate_pattern_2d(pattern, 1e-6, 0.0))
        assert abs(out_right - (-20.0)) < 1e-3

    def test_step_along_axis1_is_right_continuous(self) -> None:
        """Duplicate el value encodes a step; query at the jump returns right-side."""
        az = np.array([-180.0, 0.0, 180.0])
        el = np.array([-90.0, 0.0, 0.0, 90.0])
        gain = np.array(
            [
                [0.0, 20.0, -20.0, 0.0],
                [0.0, 20.0, -20.0, 0.0],
                [0.0, 20.0, -20.0, 0.0],
            ]
        )
        pattern = _build_pattern_2d_azel(
            az, el, gain, peak_gain_dbi=20.0,
        )
        # Exactly at el=0 → right-continuous = -20.
        assert float(ca.evaluate_pattern_2d(pattern, 0.0, 0.0)) == pytest.approx(-20.0)
        # Just-left-of-jump: close to +20.
        assert abs(
            float(ca.evaluate_pattern_2d(pattern, 0.0, -1e-6)) - 20.0
        ) < 1e-3

    def test_flat_plateau_in_a_cell(self) -> None:
        """Four equal corners → any query in the cell returns that value."""
        az = np.array([-180.0, 0.0, 180.0])
        el = np.array([-90.0, 0.0, 90.0])
        gain = np.full((3, 3), 7.5)
        pattern = _build_pattern_2d_azel(az, el, gain, peak_gain_dbi=7.5)
        for a, e in [(-90.0, -45.0), (0.0, 0.0), (37.5, 63.0), (175.0, 89.0)]:
            out = float(ca.evaluate_pattern_2d(pattern, a, e))
            assert out == pytest.approx(7.5, abs=1e-12)


class TestEvaluatePattern2DThetaPhi:
    """The ``theta_phi`` grid_mode has phi (not theta) as the wrapping axis."""

    def test_thetaphi_grid_points_return_stored_values(self) -> None:
        theta = np.array([0.0, 30.0, 90.0, 180.0])
        phi = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])
        gain = np.arange(theta.size * phi.size, dtype=np.float64).reshape(
            theta.size, phi.size
        )
        pattern = _build_pattern_2d_thetaphi(theta, phi, gain)
        assert float(ca.evaluate_pattern_2d(pattern, 30.0, -90.0)) == pytest.approx(
            gain[1, 1]
        )
        assert float(ca.evaluate_pattern_2d(pattern, 180.0, 180.0)) == pytest.approx(
            gain[3, 4]
        )

    def test_theta_axis_clips_and_phi_axis_wraps(self) -> None:
        theta = np.array([0.0, 90.0, 180.0])
        phi = np.array([-180.0, 0.0, 180.0])
        # Gain only depends on phi: at ±180 → 0, at 0 → 10.
        gain = np.tile(np.array([0.0, 10.0, 0.0]), (3, 1))
        pattern = _build_pattern_2d_thetaphi(
            theta, phi, gain, phi_wraps=True, peak_gain_dbi=10.0,
        )
        # theta below 0 clips; phi above 180 wraps.
        out_theta_clip = float(ca.evaluate_pattern_2d(pattern, -30.0, 0.0))
        assert out_theta_clip == pytest.approx(10.0)
        # phi = 540° wraps to 180° → gain 0.
        out_phi_wrap = float(ca.evaluate_pattern_2d(pattern, 90.0, 540.0))
        assert out_phi_wrap == pytest.approx(0.0)


class TestEvaluatePattern2DNormalisation:

    def test_absolute_and_relative_give_same_absolute_dbi(self) -> None:
        az = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])
        el = np.array([-90.0, -30.0, 0.0, 30.0, 90.0])
        AZ, EL = np.meshgrid(az, el, indexing="ij")
        gain_abs = 30.0 + _anisotropic_gaussian_db(AZ, EL, a_deg=45.0, b_deg=30.0)
        peak = float(np.max(gain_abs))
        gain_rel = gain_abs - peak

        pattern_abs = _build_pattern_2d_azel(
            az, el, gain_abs, normalisation="absolute", peak_gain_dbi=peak,
        )
        pattern_rel = _build_pattern_2d_azel(
            az, el, gain_rel, normalisation="relative", peak_gain_dbi=peak,
        )
        # Query on a random sub-cell grid.
        q_az = np.array([-123.4, -55.5, 0.0, 42.7, 170.0])
        q_el = np.array([-60.0, -15.0, 0.0, 25.3, 85.0])
        QA, QE = np.meshgrid(q_az, q_el, indexing="ij")
        out_abs = ca.evaluate_pattern_2d(pattern_abs, QA, QE)
        out_rel = ca.evaluate_pattern_2d(pattern_rel, QA, QE)
        np.testing.assert_allclose(out_abs, out_rel, atol=1e-10)


class TestEvaluatePattern2DResidualBudget:
    """Asymmetric ``l_t ≠ l_r``-style pattern: bilinear interp error
    stays within the theoretical bound at increasing resolutions."""

    @pytest.mark.parametrize("step_deg", [2.0, 1.0, 0.5])
    def test_anisotropic_gaussian_residual_bound(self, step_deg: float) -> None:
        """Bilinear error bound on a smooth function: ``(Δ²/8) · (|f_xx| + |f_yy|)``.

        Anisotropic Gaussian ``-log10(e) · ((az/A)² + (el/B)²)`` has
        ``|f_xx|_max = 2·log10(e)/A²`` on axis0 and ``2·log10(e)/B²``
        on axis1. A = 5°, B = 15° → f_xx_max ≈ 0.347 dB/deg² along az,
        f_yy_max ≈ 0.0386 dB/deg² along el. The textbook bilinear
        bound is ``(step²/8) · (|f_xx|_max + |f_yy|_max)``.
        """
        a_deg, b_deg = 5.0, 15.0
        az = np.arange(-30.0, 30.0 + 0.5 * step_deg, step_deg)
        el = np.arange(-45.0, 45.0 + 0.5 * step_deg, step_deg)
        AZ, EL = np.meshgrid(az, el, indexing="ij")
        gain = _anisotropic_gaussian_db(AZ, EL, a_deg=a_deg, b_deg=b_deg)
        # Pad grids out to [-180, 180] / [-90, 90] so the loader
        # accepts the schema coverage requirement. Pad the gain with
        # the boundary values (effectively a floor outside the main
        # region — irrelevant for the residual check, which queries
        # only inside the dense region).
        az_full = np.concatenate(([-180.0], az, [180.0]))
        el_full = np.concatenate(([-90.0], el, [90.0]))
        gain_full = np.full((az_full.size, el_full.size), float(np.min(gain)))
        gain_full[1:-1, 1:-1] = gain
        pattern = _build_pattern_2d_azel(
            az_full, el_full, gain_full, peak_gain_dbi=0.0,
        )

        # Query at sub-grid mid-points inside the dense region.
        q_az = np.arange(-25.0 + 0.37 * step_deg, 25.0, step_deg * 0.73)
        q_el = np.arange(-35.0 + 0.37 * step_deg, 35.0, step_deg * 0.73)
        QA, QE = np.meshgrid(q_az, q_el, indexing="ij")
        expected = _anisotropic_gaussian_db(QA, QE, a_deg=a_deg, b_deg=b_deg)
        actual = ca.evaluate_pattern_2d(pattern, QA, QE)

        # The helper uses f = -10·log10(e) · (...) → the actual
        # curvature coefficient is c = 10·log10(e) ≈ 4.343 dB, NOT
        # log10(e) itself — the factor of 10 comes from the "10·log10"
        # convention for the dB scale. |f_xx|_max = 2c/A² etc.
        c = 10.0 * float(np.log10(np.e))
        fxx_max = 2.0 * c / a_deg ** 2
        fyy_max = 2.0 * c / b_deg ** 2
        # Textbook 2-D bilinear error bound for an axis-separable
        # (f_xy = 0) smooth function: (step²/8) · |f_ii|_max per axis,
        # superposed.
        theoretical = (step_deg ** 2 / 8.0) * (fxx_max + fyy_max)
        # 1.5× cushion absorbs (a) the sample-max vs continuum-max
        # discrepancy on a moderately dense query grid and (b) minor
        # boundary effects where the dense region meets the full-range
        # padding.
        tolerance = theoretical * 1.5 + 1e-6

        err = np.abs(actual - expected)
        assert err.max() <= tolerance, (
            f"max residual {err.max():.5f} dB exceeds bilinear bound "
            f"{tolerance:.5f} dB at step={step_deg}° "
            f"(theoretical {theoretical:.5f} dB)"
        )



class TestStage28BundledExamplePatterns:
    """Stage 28: bundled example JSON files under
    scepter/data/custom_patterns/ round-trip through the loader
    cleanly. Keeps the docs "here are some ready-made patterns"
    promise honest — if someone renames a field or bumps the schema
    version these tests catch it before release.
    """

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "custom_patterns"

    def test_example_ra1631_1d_loads_cleanly(self):
        path = self._EXAMPLES_DIR / "example_ra1631_25m_1p4ghz_1d.json"
        assert path.exists(), f"Bundled example missing: {path}"
        pat = ca.load_custom_pattern(path)
        assert pat.kind == ca.KIND_1D
        assert pat.normalisation == ca.NORMALISATION_ABSOLUTE
        assert pat.peak_gain_dbi > 40.0  # 25 m dish at 1.4 GHz peaks ≈ 49 dBi

    def test_example_s1528_rec14_asym_2d_loads_cleanly(self):
        path = self._EXAMPLES_DIR / "example_s1528_rec14_asym_2d.json"
        assert path.exists(), f"Bundled example missing: {path}"
        pat = ca.load_custom_pattern(path)
        assert pat.kind == ca.KIND_2D
        assert pat.grid_mode == ca.GRID_MODE_THETAPHI
        assert pat.peak_gain_dbi == pytest.approx(34.1, abs=0.1)
