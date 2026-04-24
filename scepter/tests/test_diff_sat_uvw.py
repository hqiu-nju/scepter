from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diff_sat_uvw.py"


def _build_nominal_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ant = 3
    n_time = 4
    n_sat = 2

    pointing = np.zeros((n_ant, n_time, 3), dtype=np.float64)
    time_axis = np.arange(n_time, dtype=np.float64)

    pointing[1, :, 0] = 20.0 + 2.0 * time_axis
    pointing[1, :, 1] = -5.0 + 0.5 * time_axis
    pointing[2, :, 0] = -15.0 + 1.5 * time_axis
    pointing[2, :, 1] = 10.0 - 0.25 * time_axis

    sat_offsets = np.array([
        [8.0, 2.5, 0.0],
        [-6.0, 3.5, 0.0],
    ])
    satellite = np.repeat(pointing[:, :, np.newaxis, :], n_sat, axis=2)
    for sat_idx in range(n_sat):
        satellite[:, :, sat_idx, :] += sat_offsets[sat_idx]

    vis = np.ones((n_ant - 1, n_time), dtype=np.complex128)
    return pointing, satellite, vis


def _write_npz(path: Path, *, pointing: np.ndarray, satellite: np.ndarray, vis: np.ndarray) -> Path:
    np.savez(
        path,
        pointing_uvw_m=pointing,
        satellite_uvw_m=satellite,
        vis=vis,
        freq_mhz=np.float64(1420.0),
        uvw=pointing,
    )
    return path


def _run_diff_cli(
    input_npz: Path,
    output_png: Path,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cli_args = [
        sys.executable,
        str(SCRIPT_PATH),
        str(input_npz),
        "--output",
        str(output_png),
        "--npix",
        "64",
        "--fov-deg",
        "3.0",
    ]
    if extra_args:
        cli_args.extend(extra_args)

    return subprocess.run(
        cli_args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_generates_three_panel_png(tmp_path: Path):
    pointing, satellite, vis = _build_nominal_arrays()
    input_npz = _write_npz(
        tmp_path / "nominal.npz",
        pointing=pointing,
        satellite=satellite,
        vis=vis,
    )
    output_png = tmp_path / "out" / "diff.png"

    result = _run_diff_cli(input_npz, output_png)

    assert result.returncode == 0, result.stderr
    assert output_png.exists()
    assert output_png.stat().st_size > 0
    assert "No-sat sample count" in result.stdout
    assert "With-sat sample count" in result.stdout
    assert "Differential finite pixels" in result.stdout


def test_cli_rejects_empty_satellite_dimension(tmp_path: Path):
    pointing, _, vis = _build_nominal_arrays()
    empty_satellite = np.zeros((pointing.shape[0], pointing.shape[1], 0, 3), dtype=np.float64)
    input_npz = _write_npz(
        tmp_path / "empty_sat.npz",
        pointing=pointing,
        satellite=empty_satellite,
        vis=vis,
    )
    output_png = tmp_path / "empty_sat.png"

    result = _run_diff_cli(input_npz, output_png)

    assert result.returncode != 0
    assert "must include at least one satellite" in result.stderr


def test_cli_rejects_satellite_shape_mismatch(tmp_path: Path):
    pointing, satellite, vis = _build_nominal_arrays()
    bad_satellite = np.zeros((pointing.shape[0] + 1, pointing.shape[1], satellite.shape[2], 3), dtype=np.float64)
    input_npz = _write_npz(
        tmp_path / "shape_mismatch.npz",
        pointing=pointing,
        satellite=bad_satellite,
        vis=vis,
    )
    output_png = tmp_path / "shape_mismatch.png"

    result = _run_diff_cli(input_npz, output_png)

    assert result.returncode != 0
    assert "leading axes must match" in result.stderr


def test_cli_can_select_satellite_indices(tmp_path: Path):
    pointing, satellite, vis = _build_nominal_arrays()
    input_npz = _write_npz(
        tmp_path / "subset.npz",
        pointing=pointing,
        satellite=satellite,
        vis=vis,
    )
    output_png = tmp_path / "subset.png"

    result = _run_diff_cli(
        input_npz,
        output_png,
        extra_args=["--satellite-indices", "1"],
    )

    assert result.returncode == 0, result.stderr
    assert output_png.exists()
    assert "Selected satellite indices: [1]" in result.stdout
    assert "Selected satellite count : 1" in result.stdout
    assert "No-sat sample count      : 8" in result.stdout
    assert "With-sat sample count    : 16" in result.stdout


def test_cli_rejects_out_of_range_satellite_index(tmp_path: Path):
    pointing, satellite, vis = _build_nominal_arrays()
    input_npz = _write_npz(
        tmp_path / "bad_index.npz",
        pointing=pointing,
        satellite=satellite,
        vis=vis,
    )
    output_png = tmp_path / "bad_index.png"

    result = _run_diff_cli(
        input_npz,
        output_png,
        extra_args=["--satellite-indices", "3"],
    )

    assert result.returncode != 0
    assert "Satellite index out of range" in result.stderr
