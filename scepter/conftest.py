"""
Pytest configuration for package-local SCEPTer tests.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>

This module remains part of the GPLv3-licensed SCEPTer project. The author
attribution above does not change the project-wide license.
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES

    ASTROPY_HEADER = True
except ImportError:  # pragma: no cover
    ASTROPY_HEADER = False


def pytest_configure(config):
    """Enable Astropy-style test headers when the optional plugin is present."""
    if ASTROPY_HEADER:
        config.option.astropy_header = True
        PYTEST_HEADER_MODULES["Astropy"] = "astropy"
        PYTEST_HEADER_MODULES["Matplotlib"] = "matplotlib"
        PYTEST_HEADER_MODULES["NumPy"] = "numpy"

    # Load the rich terminal UI plugin (if rich is installed).
    try:
        from scepter import conftest_rich
        conftest_rich.pytest_configure(config)
    except Exception:
        pass

    # Register markers so pytest doesn't warn about unknown markers.
    config.addinivalue_line(
        "markers",
        "slow_leakage_math: exhaustive channel-subset leakage tests",
    )
    config.addinivalue_line(
        "markers",
        "xdist_group(name): group tests for parallel execution (pytest-xdist)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-group GPU tests for safe parallel execution with pytest-xdist.

    When running with ``-n auto`` (or any ``-n N``), tests that use CUDA
    (identified by file path or skip markers referencing CUDA) are placed
    in the ``gpu`` xdist group so they run sequentially on a single
    worker.  CPU-only tests (leakage math, config parsing, etc.) spread
    across all workers.

    When xdist is not active (``-n0`` or no ``-n``), this is a no-op.
    """
    try:
        worker_count = config.getoption("numprocesses", default=None)
    except (ValueError, AttributeError):
        worker_count = None
    if worker_count is None or worker_count == 0:
        return
    import pytest

    gpu_group = pytest.mark.xdist_group("gpu")
    gpu_test_files = {"test_gpu_accel.py", "test_surface_pfd_cap.py"}
    for item in items:
        # Tests in GPU-heavy test files always need GPU serialization.
        filename = item.path.name if hasattr(item.path, "name") else ""
        if filename in gpu_test_files:
            item.add_marker(gpu_group)
            continue
        # Scenario tests that actually run GPU simulations (long-running)
        # also need serialization; pure-math scenario tests don't.
        if filename == "test_scenario.py":
            # Tests with GPU_REQUIRED marker or tests that call
            # run_gpu_direct_epfd — heuristic: any test whose name
            # contains "gpu" or "direct_epfd" or "boresight" or
            # "session" is likely GPU-bound.
            name_lower = item.name.lower()
            if any(
                kw in name_lower
                for kw in ("gpu", "direct_epfd", "boresight", "session", "propagat")
            ):
                item.add_marker(gpu_group)


def pytest_sessionfinish(session, exitstatus):
    """Clean up the shared basetemp directory after the test session.

    pytest creates per-test directories under ``--basetemp`` but never
    removes the root.  This hook deletes the entire tree once all tests
    have finished so stale fixture files don't accumulate between runs.
    Only runs on the controller process (not on xdist workers).
    """
    # Under pytest-xdist, workers have workerinput; only the controller
    # should clean up.
    if hasattr(session.config, "workerinput"):
        return
    basetemp = session.config.option.basetemp
    if basetemp is None:
        return
    basetemp_path = Path(basetemp).resolve()
    if basetemp_path.exists() and basetemp_path.is_dir():
        try:
            shutil.rmtree(basetemp_path, ignore_errors=True)
        except Exception:
            pass
