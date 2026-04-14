from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _compile_notebook_code_cells(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{path.name}#cell-{idx}", "exec")


def test_setup_py_uses_gpl_classifier():
    setup_text = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "GNU General Public License v3 or later" in setup_text
    assert "MIT License" not in setup_text


def test_readme_mentions_core_apis_and_license():
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "GpuScepterSession" in readme_text
    assert "scenario" in readme_text
    assert "gpu_accel" in readme_text
    assert "GPLv3" in readme_text
    assert "environment.yml" in readme_text
    assert "environment-full.yml" in readme_text
    assert "gui.py" in readme_text


@pytest.mark.skip(
    reason="DEPRECATED / pending review: SCEPTer_simulate.ipynb was removed; "
    "test retained for future reinstatement once notebook workflow is redefined."
)
def test_boresight_workflow_notebook_uses_shared_library_workflow():
    notebook_text = (
        REPO_ROOT / "SCEPTer_simulate.ipynb"
    ).read_text(encoding="utf-8")
    for token in (
        "summarize_contour_spacing(",
        "prepare_active_grid(",
        "resolve_theta2_active_cell_ids(",
        "plot_cell_status_map(",
        "scenario.build_observer_layout(",
        "scenario.run_gpu_direct_epfd(",
    ):
        assert token in notebook_text


@pytest.mark.skip(
    reason="DEPRECATED / pending review: SCEPTer_simulate.ipynb was removed; "
    "test retained for future reinstatement once notebook workflow is redefined."
)
def test_boresight_workflow_notebook_code_cells_compile():
    _compile_notebook_code_cells(
        REPO_ROOT / "SCEPTer_simulate.ipynb"
    )


@pytest.mark.parametrize(
    "notebook_name",
    [
        "SCEPTer_simulate.ipynb",
    ],
)
@pytest.mark.skip(
    reason="DEPRECATED / pending review: SCEPTer_simulate.ipynb was removed; "
    "test retained for future reinstatement once notebook workflow is redefined."
)
def test_step1_family_notebooks_code_cells_compile(notebook_name: str):
    _compile_notebook_code_cells(REPO_ROOT / notebook_name)
