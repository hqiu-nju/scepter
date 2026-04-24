from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "old_uvw_fast_imaging_script"
    / "snapshot_dirty_images_from_uvw.py"
)


def _import_script(path: Path):
    """Import a script module while keeping sibling-script imports resolvable."""
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_source_uvw_prefers_satellite_tensor(tmp_path: Path):
    mod = _import_script(SCRIPT_PATH)

    pointing = np.zeros((3, 4, 3), dtype=np.float64)
    satellite = np.zeros((3, 4, 2, 3), dtype=np.float64)
    archive_path = tmp_path / "tracking_uvw.npz"
    np.savez(
        archive_path,
        pointing_uvw_m=pointing,
        satellite_uvw_m=satellite,
        uvw=pointing,
        vis=np.ones((2, 4), dtype=np.complex128),
        freq_mhz=np.float64(1420.0),
    )

    with np.load(archive_path, allow_pickle=False) as dataset:
        resolved, key = mod._resolve_source_uvw(
            dataset,
            source_uvw_key=None,
            uvw_key="pointing_uvw_m",
            fallback_key="uvw",
        )

    assert key == "satellite_uvw_m"
    assert resolved.shape == (3, 4, 2, 3)


def test_normalise_uvw_sources_matches_generator_layout():
    mod = _import_script(SCRIPT_PATH)

    uvw = np.arange(3 * 5 * 2 * 3, dtype=np.float64).reshape(3, 5, 2, 3)
    uvw_by_source = mod._normalise_uvw_sources(uvw, source_axis=2)

    assert uvw_by_source.shape == (2, 3, 5, 3)
    np.testing.assert_array_equal(uvw_by_source[0], uvw[:, :, 0, :])
    np.testing.assert_array_equal(uvw_by_source[1], uvw[:, :, 1, :])


def test_load_optional_names_requires_matching_source_count(tmp_path: Path):
    mod = _import_script(SCRIPT_PATH)

    archive_path = tmp_path / "names.npz"
    np.savez(
        archive_path,
        satellite_names=np.asarray(["sat_a", "sat_b"], dtype=str),
    )

    with np.load(archive_path, allow_pickle=False) as dataset:
        names = mod._load_optional_names(dataset, "satellite_names", expected_count=2)
        mismatched = mod._load_optional_names(dataset, "satellite_names", expected_count=3)

    assert names == ("sat_a", "sat_b")
    assert mismatched is None