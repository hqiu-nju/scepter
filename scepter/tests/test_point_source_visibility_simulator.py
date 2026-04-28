from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from numpy.testing import assert_allclose


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "point_source_visibility_simulator.py"


def _import_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_load_point_source_catalog_accepts_headered_csv(tmp_path: Path) -> None:
    mod = _import_script(SCRIPT_PATH)
    source_file = tmp_path / "sources.csv"
    source_file.write_text(
        "source_name,ra_deg,dec_deg,flux_jy\n"
        "phase,370.0,-30.0,2.5\n",
        encoding="utf-8",
    )

    catalog = mod.load_point_source_catalog(source_file)

    assert catalog.names == ("phase",)
    assert_allclose(catalog.ra_deg, [10.0])
    assert_allclose(catalog.dec_deg, [-30.0])
    assert_allclose(catalog.flux, [2.5])


def test_phase_centre_source_has_constant_real_visibility() -> None:
    mod = _import_script(SCRIPT_PATH)
    antennas = [
        EarthLocation(lon=21.4430 * u.deg, lat=-30.7130 * u.deg, height=1086.0 * u.m),
        EarthLocation(lon=21.4440 * u.deg, lat=-30.7130 * u.deg, height=1086.0 * u.m),
        EarthLocation(lon=21.4430 * u.deg, lat=-30.7120 * u.deg, height=1086.0 * u.m),
    ]
    obs_times = Time(["2025-01-01T00:00:00", "2025-01-01T00:01:00"], scale="utc")
    catalog = mod.PointSourceCatalog(
        names=("phase",),
        ra_deg=np.asarray([83.633], dtype=np.float64),
        dec_deg=np.asarray([22.014], dtype=np.float64),
        flux=np.asarray([3.25], dtype=np.float64),
    )

    result = mod.simulate_point_source_visibilities(
        antennas,
        obs_times,
        phase_ra_deg=83.633,
        phase_dec_deg=22.014,
        catalog=catalog,
        frequency_hz=1420.0e6,
        baseline_mode="all-pairs",
    )

    assert result.baseline_pairs == ((0, 1), (0, 2), (1, 2))
    assert result.uvw_m.shape == (3, 2, 3)
    assert result.vis_per_source.shape == (3, 2, 1)
    assert_allclose(result.phase_rad, 0.0, atol=1e-10)
    assert_allclose(result.vis, 3.25 + 0.0j, atol=1e-10)
    assert_allclose(result.normalised_amplitude, 1.0, atol=1e-12)


def test_resolve_phase_centre_defaults_to_first_source() -> None:
    mod = _import_script(SCRIPT_PATH)
    catalog = mod.PointSourceCatalog(
        names=("first", "second"),
        ra_deg=np.asarray([37.742, 146.905], dtype=np.float64),
        dec_deg=np.asarray([-12.318, 28.441], dtype=np.float64),
        flux=np.asarray([1.20, 0.65], dtype=np.float64),
    )

    ra_deg, dec_deg, source_name = mod.resolve_phase_centre(catalog, None, None)

    assert source_name == "first"
    assert_allclose(ra_deg, 37.742)
    assert_allclose(dec_deg, -12.318)


def test_cli_writes_quicklook_compatible_archive(tmp_path: Path) -> None:
    array_file = tmp_path / "array.csv"
    array_file.write_text(
        "name,lon_deg,lat_deg,alt_m\n"
        "ref,21.4430,-30.7130,1086.0\n"
        "east,21.4440,-30.7130,1086.0\n",
        encoding="utf-8",
    )
    source_file = tmp_path / "sources.txt"
    source_file.write_text(
        "source_name;ra_deg;dec_deg;flux\n"
        "phase;83.633;22.014;1.0\n"
        "offset;84.000;22.200;0.5\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "point_source_vis.npz"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--array-file",
            str(array_file),
            "--source-file",
            str(source_file),
            "--start-time",
            "2025-01-01T00:00:00",
            "--n-times",
            "2",
            "--cadence-sec",
            "60",
            "--freq-mhz",
            "1420.0",
            "--baseline-mode",
            "reference",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Saved point-source visibility archive" in completed.stdout
    with np.load(output_path, allow_pickle=False) as dataset:
        assert "uvw" in dataset.files
        assert "vis" in dataset.files
        assert "freq_hz" in dataset.files
        assert "vis_per_source" in dataset.files
        assert dataset["uvw"].shape == (1, 2, 3)
        assert dataset["vis"].shape == (1, 2)
        assert dataset["vis_per_source"].shape == (1, 2, 2)
        assert dataset["baseline_names"].tolist() == ["ref-east"]
        assert dataset["source_names"].tolist() == ["phase", "offset"]
        assert dataset["phase_source_name"].tolist() == "phase"
        assert_allclose(dataset["phase_ra_deg"], 83.633)
        assert_allclose(dataset["phase_dec_deg"], 22.014)
        assert_allclose(dataset["freq_hz"], 1420.0e6)


def test_cli_defaults_write_archive_in_working_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    output_path = tmp_path / "point_source_vis.npz"
    assert "Saved point-source visibility archive" in completed.stdout
    assert output_path.exists()
    with np.load(output_path, allow_pickle=False) as dataset:
        assert dataset["vis"].shape[1] == 60
        assert dataset["source_names"].shape == (3,)
        assert_allclose(dataset["freq_hz"], 1420.0e6)
