from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "w_corrected_image.py"
QUICK_IMG_PATH = REPO_ROOT / "scripts" / "quick_dirty_image_from_uvw.py"


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_coplanar_npz(tmp_path: Path, *, npix: int = 64) -> Path:
    """
    Build a minimal NPZ with coplanar baselines (w = 0) and a unit vis model.
    Uses enough baselines to produce non-trivial UV coverage.
    """
    n_ant = 4
    n_time = 8
    rng = np.random.default_rng(42)

    pointing = np.zeros((n_ant, n_time, 3), dtype=np.float64)
    for i in range(1, n_ant):
        pointing[i, :, 0] = rng.uniform(-50.0, 50.0, n_time)
        pointing[i, :, 1] = rng.uniform(-50.0, 50.0, n_time)
        # w (axis 2) stays 0 → coplanar

    vis = np.ones((n_ant - 1, n_time), dtype=np.complex128)
    out = tmp_path / "coplanar.npz"
    np.savez(out, pointing_uvw_m=pointing, vis=vis, freq_mhz=np.float64(1420.0))
    return out


def _make_nonzero_w_npz(tmp_path: Path) -> Path:
    """
    Build a minimal NPZ where baselines have non-zero w, so the w-correction
    should produce a measurably different image from the standard path.
    """
    n_ant = 4
    n_time = 8
    rng = np.random.default_rng(7)

    pointing = np.zeros((n_ant, n_time, 3), dtype=np.float64)
    for i in range(1, n_ant):
        pointing[i, :, 0] = rng.uniform(-100.0, 100.0, n_time)
        pointing[i, :, 1] = rng.uniform(-100.0, 100.0, n_time)
        pointing[i, :, 2] = rng.uniform(-200.0, 200.0, n_time)  # large w values

    vis = np.ones((n_ant - 1, n_time), dtype=np.complex128)
    out = tmp_path / "nonzero_w.npz"
    np.savez(out, pointing_uvw_m=pointing, vis=vis, freq_mhz=np.float64(1420.0))
    return out


def _import_script(path: Path):
    """Import a script as a module for direct function-level testing.

    Adds the script's parent directory to ``sys.path`` so that sibling
    scripts (e.g. ``quick_dirty_image_from_uvw``) can be resolved by the
    top-level ``from`` imports inside the loaded module.
    """
    import importlib.util

    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_cli(input_npz: Path, output_png: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        str(input_npz),
        "--output", str(output_png),
        "--npix", "64",
        "--fov-deg", "3.0",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# Unit tests: make_w_corrected_dirty_image
# ---------------------------------------------------------------------------

class TestMakeWCorrectedDirtyImage:
    @pytest.fixture(scope="class")
    def mod(self):
        return _import_script(SCRIPT_PATH)

    def test_coplanar_w_corr_matches_standard(self, mod):
        """When all w = 0, w-corrected and standard images should be identical."""
        rng = np.random.default_rng(0)
        n = 50
        uvw = np.zeros((n, 3), dtype=np.float64)
        uvw[:, 0] = rng.uniform(-80.0, 80.0, n)
        uvw[:, 1] = rng.uniform(-80.0, 80.0, n)
        # w remains zero

        vis = np.ones(n, dtype=np.complex128)
        freq_hz = 1420.0e6

        w_corr, std, beam, meta = mod.make_w_corrected_dirty_image(
            uvw, vis, freq_hz,
            npix=64, fov_deg=3.0, n_wplanes=8,
        )

        np.testing.assert_allclose(
            w_corr, std,
            atol=1e-10,
            err_msg="W-corrected and standard images should be identical when w = 0.",
        )

    def test_output_shapes(self, mod):
        rng = np.random.default_rng(1)
        n = 40
        uvw = np.column_stack([
            rng.uniform(-50, 50, n),
            rng.uniform(-50, 50, n),
            rng.uniform(-100, 100, n),
        ])
        vis = np.ones(n, dtype=np.complex128)
        w_corr, std, beam, meta = mod.make_w_corrected_dirty_image(
            uvw, vis, 1420.0e6, npix=32, fov_deg=2.0, n_wplanes=4,
        )
        assert w_corr.shape == (32, 32)
        assert std.shape == (32, 32)
        assert beam.shape == (32, 32)

    def test_metadata_keys_present(self, mod):
        rng = np.random.default_rng(2)
        n = 30
        uvw = np.column_stack([rng.uniform(-30, 30, n), rng.uniform(-30, 30, n), np.zeros(n)])
        vis = np.ones(n, dtype=np.complex128)
        _, _, _, meta = mod.make_w_corrected_dirty_image(uvw, vis, 1420.0e6, npix=32)
        expected_keys = {
            "wavelength_m", "uv_cell_lambda", "n_wplanes_used",
            "w_min_lambda", "w_max_lambda",
            "gridded_sample_count", "dropped_sample_count",
        }
        assert expected_keys <= set(meta.keys())

    def test_nonzero_w_produces_different_image(self, mod):
        """W-corrected image should differ from standard when w is large."""
        rng = np.random.default_rng(3)
        n = 60
        uvw = np.column_stack([
            rng.uniform(-50, 50, n),
            rng.uniform(-50, 50, n),
            rng.uniform(-500, 500, n),  # large w
        ])
        vis = np.ones(n, dtype=np.complex128)

        w_corr, std, _, _ = mod.make_w_corrected_dirty_image(
            uvw, vis, 1420.0e6,
            npix=64, fov_deg=5.0, n_wplanes=16,
        )
        diff_peak = float(np.max(np.abs(w_corr - std)))
        assert diff_peak > 1e-4, (
            f"Expected measurable difference between w-corrected and standard "
            f"images for large w, but max |diff| = {diff_peak}."
        )

    def test_output_is_finite(self, mod):
        rng = np.random.default_rng(4)
        n = 40
        uvw = np.column_stack([
            rng.uniform(-40, 40, n),
            rng.uniform(-40, 40, n),
            rng.uniform(-80, 80, n),
        ])
        vis = np.ones(n, dtype=np.complex128)
        w_corr, std, beam, _ = mod.make_w_corrected_dirty_image(
            uvw, vis, 1420.0e6, npix=32, fov_deg=2.0, n_wplanes=8,
        )
        assert np.isfinite(w_corr).all(), "W-corrected image contains non-finite values."
        assert np.isfinite(std).all(), "Standard image contains non-finite values."
        assert np.isfinite(beam).all(), "Dirty beam contains non-finite values."

    def test_beam_peak_is_unity(self, mod):
        rng = np.random.default_rng(5)
        n = 40
        uvw = np.column_stack([
            rng.uniform(-40, 40, n),
            rng.uniform(-40, 40, n),
            rng.uniform(-80, 80, n),
        ])
        vis = np.ones(n, dtype=np.complex128)
        _, _, beam, _ = mod.make_w_corrected_dirty_image(
            uvw, vis, 1420.0e6, npix=32, fov_deg=2.0, n_wplanes=4,
        )
        assert abs(float(np.max(beam)) - 1.0) < 1e-9, "Beam peak should be normalised to 1."

    def test_n_wplanes_1_matches_standard(self, mod):
        """With n_wplanes=1, the single plane has w_centre = mean(w).
        For w=0, the single-plane result must equal standard exactly."""
        rng = np.random.default_rng(6)
        n = 30
        uvw = np.column_stack([rng.uniform(-40, 40, n), rng.uniform(-40, 40, n), np.zeros(n)])
        vis = np.ones(n, dtype=np.complex128)

        w_corr_1, std, _, _ = mod.make_w_corrected_dirty_image(
            uvw, vis, 1420.0e6, npix=32, fov_deg=2.0, n_wplanes=1,
        )
        np.testing.assert_allclose(w_corr_1, std, atol=1e-10)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_generates_comparison_png(tmp_path: Path):
    npz = _make_nonzero_w_npz(tmp_path)
    out = tmp_path / "out" / "wcorr.png"

    result = _run_cli(npz, out)

    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert out.stat().st_size > 0
    assert "W-planes used" in result.stdout
    assert "Saved comparison figure to" in result.stdout


def test_cli_coplanar_runs_successfully(tmp_path: Path):
    npz = _make_coplanar_npz(tmp_path)
    out = tmp_path / "coplanar.png"

    result = _run_cli(npz, out)

    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_cli_with_beam_panel(tmp_path: Path):
    npz = _make_nonzero_w_npz(tmp_path)
    out = tmp_path / "with_beam.png"

    result = _run_cli(npz, out, extra_args=["--show-beam"])

    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_cli_with_n_correction(tmp_path: Path):
    npz = _make_nonzero_w_npz(tmp_path)
    out = tmp_path / "ncorr.png"

    result = _run_cli(npz, out, extra_args=["--apply-n-correction", "--fov-deg", "5.0"])

    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_cli_reports_w_range(tmp_path: Path):
    npz = _make_nonzero_w_npz(tmp_path)
    out = tmp_path / "wrange.png"

    result = _run_cli(npz, out)

    assert "W range" in result.stdout
    assert "Peak |W-corr − Standard|" in result.stdout


def test_cli_fallback_uvw_key(tmp_path: Path):
    """When pointing_uvw_m is absent, the fallback 'uvw' key should be used."""
    n_ant, n_time = 3, 6
    rng = np.random.default_rng(10)
    uvw = np.zeros((n_ant, n_time, 3), dtype=np.float64)
    uvw[1, :, 0] = rng.uniform(-40, 40, n_time)
    uvw[1, :, 1] = rng.uniform(-40, 40, n_time)
    vis = np.ones((n_ant - 1, n_time), dtype=np.complex128)
    npz = tmp_path / "fallback.npz"
    np.savez(npz, uvw=uvw, vis=vis, freq_mhz=np.float64(1420.0))  # no pointing_uvw_m key

    out = tmp_path / "fallback.png"
    result = _run_cli(npz, out)

    assert result.returncode == 0, result.stderr
    assert out.exists()
