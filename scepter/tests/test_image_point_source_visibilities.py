from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "image_point_source_visibilities.py"


def _import_script(path: Path):
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_source_offsets_place_phase_centre_at_origin() -> None:
    mod = _import_script(SCRIPT_PATH)

    l_deg, m_deg = mod.source_offsets_lm_deg(
        np.asarray([83.633], dtype=np.float64),
        np.asarray([22.014], dtype=np.float64),
        phase_ra_deg=83.633,
        phase_dec_deg=22.014,
    )

    assert_allclose(l_deg, [0.0], atol=1e-12)
    assert_allclose(m_deg, [0.0], atol=1e-12)


def test_cli_images_point_source_visibility_archive(tmp_path: Path) -> None:
    uvw = np.zeros((2, 8, 3), dtype=np.float64)
    uvw[0, :, 0] = np.linspace(-40.0, 40.0, 8)
    uvw[0, :, 1] = np.linspace(35.0, -35.0, 8)
    uvw[1, :, 0] = np.linspace(-55.0, 55.0, 8)
    uvw[1, :, 1] = np.linspace(-20.0, 20.0, 8)
    vis = np.ones((2, 8), dtype=np.complex128)
    input_npz = tmp_path / "point_source_vis.npz"
    np.savez(
        input_npz,
        uvw=uvw,
        vis=vis,
        freq_hz=np.float64(1420.0e6),
        source_names=np.asarray(["phase", "offset"], dtype=str),
        source_ra_deg=np.asarray([83.633, 83.700], dtype=np.float64),
        source_dec_deg=np.asarray([22.014, 22.050], dtype=np.float64),
        phase_ra_deg=np.float64(83.633),
        phase_dec_deg=np.float64(22.014),
    )
    output_png = tmp_path / "image.png"
    output_npz = tmp_path / "image_arrays.npz"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(input_npz),
            "--output",
            str(output_png),
            "--image-npz",
            str(output_npz),
            "--npix",
            "64",
            "--fov-deg",
            "6.0",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Saved point-source image" in result.stdout
    assert output_png.exists()
    assert output_png.stat().st_size > 0
    with np.load(output_npz, allow_pickle=False) as dataset:
        assert dataset["dirty_image"].shape == (64, 64)
        assert dataset["dirty_beam"].shape == (64, 64)
        assert_allclose(dataset["freq_hz"], 1420.0e6)


def test_cli_defaults_read_standard_archive_name(tmp_path: Path) -> None:
    uvw = np.zeros((1, 6, 3), dtype=np.float64)
    uvw[0, :, 0] = np.linspace(-30.0, 30.0, 6)
    uvw[0, :, 1] = np.linspace(20.0, -20.0, 6)
    vis = np.ones((1, 6), dtype=np.complex128)
    np.savez(
        tmp_path / "point_source_vis.npz",
        uvw=uvw,
        vis=vis,
        freq_hz=np.float64(1420.0e6),
        source_names=np.asarray(["phase"], dtype=str),
        source_ra_deg=np.asarray([83.633], dtype=np.float64),
        source_dec_deg=np.asarray([22.014], dtype=np.float64),
        phase_ra_deg=np.float64(83.633),
        phase_dec_deg=np.float64(22.014),
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--npix", "64"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    output_png = tmp_path / "point_source_dirty_image.png"
    assert "Saved point-source image" in result.stdout
    assert output_png.exists()
    assert output_png.stat().st_size > 0
