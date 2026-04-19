"""Root entry point for the SCEPTer desktop GUI."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import traceback
import warnings

# On Windows, conda environments require Library/bin on PATH for DLL
# loading (BLAS, CUDA, OpenSSL, etc.).  When running outside a
# ``conda activate`` shell (e.g. direct python.exe invocation) these
# DLLs are not discoverable.  Fix this early before any C-extension
# imports.
if sys.platform == "win32":
    _lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(_lib_bin):
        os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_lib_bin)
            except OSError:
                pass
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    _conda_cuda_include = os.path.join(sys.prefix, "Library", "include")
    if os.path.isfile(os.path.join(_conda_cuda_include, "cuda.h")):
        _sys_cuda_path = os.environ.get("CUDA_PATH", "")
        os.environ["CUDA_PATH"] = os.path.join(sys.prefix, "Library")
        if _sys_cuda_path:
            _sys_nvvm_bin = os.path.join(_sys_cuda_path, "nvvm", "bin", "x64")
            if os.path.isdir(_sys_nvvm_bin):
                os.environ["PATH"] = _sys_nvvm_bin + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(_sys_nvvm_bin)
                    except OSError:
                        pass
            _conda_libdevice = os.path.join(sys.prefix, "Library", "nvvm", "libdevice")
            _sys_libdevice = os.path.join(_sys_cuda_path, "nvvm", "libdevice")
            if (
                not os.path.isfile(os.path.join(_conda_libdevice, "libdevice.10.bc"))
                and os.path.isfile(os.path.join(_sys_libdevice, "libdevice.10.bc"))
            ):
                os.environ.setdefault("NUMBA_CUDA_LIBDEVICE", _sys_libdevice)

warnings.filterwarnings(
    "ignore",
    message=r"^pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"shibokensupport\.signature\.parser",
)

from PySide6 import QtCore, QtWidgets


def _load_module_from_path(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _show_startup_error(
    *,
    app: QtWidgets.QApplication,
    splash: QtWidgets.QWidget | None,
    owns_app: bool,
    stage: str,
    exc: Exception,
) -> None:
    """Report a startup failure without leaving the splash loop stranded."""
    if splash is not None:
        try:
            splash.hide()
            splash.close()
        except Exception:
            pass
    detail_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    message_text = str(exc).strip() or exc.__class__.__name__
    dialog = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Critical,
        "SCEPTer Start-up Error",
        f"SCEPTer could not finish starting while {stage}.\n\n{message_text}",
    )
    dialog.setDetailedText(detail_text)
    dialog.setWindowModality(QtCore.Qt.ApplicationModal)
    dialog.setWindowIcon(app.windowIcon())
    dialog.exec()
    if owns_app:
        QtCore.QTimer.singleShot(0, app.quit)


def main() -> int:
    """Launch the desktop GUI and return the Qt exit code.
Y
    The heavy import and window creation are scheduled as deferred event-loop
    callbacks so the splash screen animation stays fluid throughout.  Each
    heavy step is wrapped in a ``QTimer.singleShot(0, ...)`` call which
    yields back to the event loop before executing, giving the animation
    timer a chance to repaint between steps.
    """
    bootstrap = _load_module_from_path(
        "_scepter_gui_bootstrap_launcher", "scepter/gui_bootstrap.py",
    )
    bootstrap.configure_windows_shell_identity()

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])
    icon = bootstrap.load_app_icon()

    splash = None
    if owns_app:
        splash = bootstrap.create_startup_splash(icon)
        splash.set_stage(progress=0.05, status_text="Starting...")
        splash.show()
        bootstrap.apply_windows_window_icon(splash)
        app.processEvents()

    if splash is not None:
        splash.set_stage(progress=0.12, status_text="Configuring application...")
    if owns_app:
        bootstrap.configure_application(app)
    app.setWindowIcon(icon)
    if splash is not None:
        splash.setWindowIcon(icon)
        splash.set_stage(progress=0.20, status_text="Loading core libraries...")
    app.processEvents()

    # ---- Deferred heavy work via event loop callbacks ---------------------
    # Each step yields back to the event loop (via singleShot) so the
    # splash animation timer can fire between steps.  This prevents the
    # OS from marking the window as "not responding".

    _state: dict[str, object] = {
        "splash": splash,
        "icon": icon,
        "app": app,
        "owns_app": owns_app,
        "window": None,
        "startup_failed": False,
    }

    def _handle_startup_failure(stage: str, exc: Exception) -> None:
        if bool(_state.get("startup_failed")):
            return
        _state["startup_failed"] = True
        window = _state.get("window")
        if isinstance(window, QtWidgets.QWidget):
            try:
                window.close()
            except Exception:
                pass
        _show_startup_error(
            app=app,
            splash=_state.get("splash") if isinstance(_state.get("splash"), QtWidgets.QWidget) else None,
            owns_app=bool(_state["owns_app"]),
            stage=stage,
            exc=exc,
        )

    def _step_import_core_deps():
        """Pre-import the heaviest transitive dependencies one at a time."""
        try:
            s = _state["splash"]
            if s is not None:
                s.set_stage(progress=0.25, status_text="Loading NumPy...")
            app.processEvents()
            import numpy  # noqa: F401
            app.processEvents()
            if s is not None:
                s.set_stage(progress=0.30, status_text="Loading Astropy...")
            app.processEvents()
            import astropy  # noqa: F401
            app.processEvents()
            if s is not None:
                s.set_stage(progress=0.35, status_text="Loading Matplotlib...")
            app.processEvents()
            import matplotlib  # noqa: F401
            app.processEvents()
            QtCore.QTimer.singleShot(0, _step_import_scepter)
        except Exception as exc:
            _handle_startup_failure("loading core libraries", exc)

    def _step_import_scepter():
        try:
            s = _state["splash"]
            if s is not None:
                s.set_stage(progress=0.42, status_text="Loading simulation core...")
            app.processEvents()
            import scepter  # noqa: F401
            app.processEvents()
            if s is not None:
                s.set_stage(progress=0.52, status_text="Loading 3D viewer...")
            app.processEvents()
            try:
                import pyvista  # noqa: F401
            except ImportError:
                pass
            app.processEvents()
            QtCore.QTimer.singleShot(0, _step_import_gui)
        except Exception as exc:
            _handle_startup_failure("loading the simulation core", exc)

    def _step_import_gui():
        try:
            s = _state["splash"]
            if s is not None:
                s.set_stage(progress=0.60, status_text="Loading GUI studio...")
            app.processEvents()
            from scepter.scepter_GUI import create_main_window
            _state["create_main_window"] = create_main_window
            if s is not None:
                s.set_stage(progress=0.72, status_text="Building the main window...")
            app.processEvents()
            QtCore.QTimer.singleShot(0, _step_create_window)
        except Exception as exc:
            _handle_startup_failure("loading the GUI studio", exc)

    def _step_create_window():
        try:
            s = _state["splash"]
            app.processEvents()
            create_main_window = _state["create_main_window"]
            window = create_main_window(show=False)
            window.setWindowIcon(_state["icon"])
            _state["window"] = window
            if s is not None:
                s.set_stage(progress=0.88, status_text="Preparing display...")
            app.processEvents()
            QtCore.QTimer.singleShot(0, _step_show_window)
        except Exception as exc:
            _handle_startup_failure("building the main window", exc)

    def _step_show_window():
        try:
            s = _state["splash"]
            window = _state["window"]
            if s is not None:
                s.set_stage(progress=0.95, status_text="Showing the main window...")
            app.processEvents()
            window.showMaximized()
            bootstrap.apply_windows_window_icon(window)
            if s is not None:
                s.set_stage(progress=1.0, status_text="Ready.")
                app.processEvents()
                s.finish_with_min_duration(window)
        except Exception as exc:
            _handle_startup_failure("showing the main window", exc)

    # Kick off the deferred import chain
    QtCore.QTimer.singleShot(0, _step_import_core_deps)

    if not owns_app:
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
