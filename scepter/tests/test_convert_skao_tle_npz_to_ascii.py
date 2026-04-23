from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_skao_tle_npz_to_ascii.py"
EXAMPLE_TLE_TEXT = (
    "ONEWEB-0001\n"
    "1 00001U 24001A   24092.00000000  .00000000  00000+0  00000+0 0  9990\n"
    "2 00001  87.9000 120.0000 0001000  90.0000 270.0000 13.50000000    01\n"
)


def _write_skao_archive(path: Path, tle_text: str = EXAMPLE_TLE_TEXT) -> Path:
    payload = np.array(SimpleNamespace(text=tle_text), dtype=object)
    np.savez(path, payload)
    return path


def test_extract_skao_tle_text_reads_default_archive_payload():
    import importlib.util

    spec = importlib.util.spec_from_file_location("convert_skao_tle_npz_to_ascii", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    archive_path = Path(module.__file__).resolve().parent / ".pytest-extract-temp.npz"
    try:
        _write_skao_archive(archive_path)
        text = module.extract_skao_tle_text(archive_path)
    finally:
        archive_path.unlink(missing_ok=True)

    assert text == EXAMPLE_TLE_TEXT


def test_cli_converts_all_archives_in_directory(tmp_path: Path):
    input_dir = tmp_path / "archive"
    output_dir = tmp_path / "ascii"
    input_dir.mkdir()

    first_archive = _write_skao_archive(input_dir / "20240401_000000.npz")
    second_archive = _write_skao_archive(
        input_dir / "20240401_120000.npz",
        tle_text=EXAMPLE_TLE_TEXT.replace("ONEWEB-0001", "ONEWEB-0002"),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "20240401_000000.tle").read_text(encoding="utf-8") == EXAMPLE_TLE_TEXT
    assert (output_dir / "20240401_120000.tle").read_text(encoding="utf-8") == (
        EXAMPLE_TLE_TEXT.replace("ONEWEB-0001", "ONEWEB-0002")
    )
    assert str(first_archive.with_suffix(".tle").name) in result.stdout
    assert str(second_archive.with_suffix(".tle").name) in result.stdout