from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BORESIGHT_WORKFLOW_NOTEBOOK = REPO_ROOT / "SCEPTer_simulate.ipynb"

ACTIVE_PIPELINE_FILES = (
    BORESIGHT_WORKFLOW_NOTEBOOK,
)

REQUIRED_BORESIGHT_SHARED_CALLS = (
    "summarize_contour_spacing",
    "prepare_active_grid",
    "resolve_theta2_active_cell_ids",
    "plot_cell_status_map",
    "build_observer_layout",
    "run_gpu_direct_epfd",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_active_pipeline_files_no_longer_reference_skao() -> None:
    for path in ACTIVE_PIPELINE_FILES:
        assert "SKAO" not in _read_text(path), f"Found legacy SKAO token in {path.name}"


def test_boresight_notebook_uses_shared_grid_and_gpu_helpers() -> None:
    text = _read_text(BORESIGHT_WORKFLOW_NOTEBOOK)
    for token in REQUIRED_BORESIGHT_SHARED_CALLS:
        assert token in text, f"Missing shared helper {token} in {BORESIGHT_WORKFLOW_NOTEBOOK.name}"
