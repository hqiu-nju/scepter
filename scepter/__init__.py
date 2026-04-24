
"""
Package Name
=============

A brief description of the package.

"""

__doc__ = "SCEPTer, Simulating Constellation Emission Patterns for Telescopes"
__version__ = "0.25.3"
__codename__ = "Patterns Strike Back"

# On Windows without PYTHONUTF8=1, ``sys.stdout``/``sys.stderr`` default to
# the ANSI code page (``cp1252`` on most English locales), which cannot
# encode common Unicode glyphs (``→``, ``×``, ``≈``, ``≤``, ...).
# These appear in source-line comments and in some warning/progress text,
# so a stray ``warnings.warn`` / traceback print during a long run surfaces
# as ``UnicodeEncodeError`` and gets wrapped as an opaque stage failure
# (e.g. ``_DirectEpfdStageExecutionError: beam_finalize: 'charmap' codec
# can't encode character '→' ...``). Reconfigure the streams to UTF-8
# with a lossy fallback so reporting never crashes the run. Applied at
# package import so every entry point (GUI, CLI, tests, notebooks)
# benefits — ``gui.py`` does the same thing earlier for the case where
# ``scepter`` hasn't been imported yet.
import os as _os
import sys as _sys
if _sys.platform == "win32":
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
    _lib_bin = _os.path.join(_sys.prefix, "Library", "bin")
    if _os.path.isdir(_lib_bin):
        _os.environ["PATH"] = _lib_bin + _os.pathsep + _os.environ.get("PATH", "")
        if hasattr(_os, "add_dll_directory"):
            try:
                _os.add_dll_directory(_lib_bin)
            except OSError:
                pass
    # CuPy requires CONDA_PREFIX to find the conda CUDA toolkit; without
    # ``conda activate`` this is unset and CuPy crashes with NoneType.
    _os.environ.setdefault("CONDA_PREFIX", _sys.prefix)
    # When the conda env ships its own CUDA headers, point CUDA_PATH there
    # so CuPy's NVRTC compiles against matching headers (avoids version-
    # mismatch errors with a system-installed CUDA toolkit).  We also keep
    # the system toolkit's NVVM directory on PATH so numba can find
    # nvvm64*.dll (not shipped by the conda env by default).
    _conda_cuda_include = _os.path.join(_sys.prefix, "Library", "include")
    if _os.path.isfile(_os.path.join(_conda_cuda_include, "cuda.h")):
        _sys_cuda_path = _os.environ.get("CUDA_PATH", "")
        _os.environ["CUDA_PATH"] = _os.path.join(_sys.prefix, "Library")
        # Preserve access to system NVVM + libdevice for numba-cuda
        if _sys_cuda_path:
            _sys_nvvm_bin = _os.path.join(_sys_cuda_path, "nvvm", "bin", "x64")
            if _os.path.isdir(_sys_nvvm_bin):
                _os.environ["PATH"] = _sys_nvvm_bin + _os.pathsep + _os.environ.get("PATH", "")
                if hasattr(_os, "add_dll_directory"):
                    try:
                        _os.add_dll_directory(_sys_nvvm_bin)
                    except OSError:
                        pass
            # numba-cuda needs libdevice.10.bc; if not in conda env,
            # point numba to the system toolkit's copy.
            _conda_libdevice = _os.path.join(_sys.prefix, "Library", "nvvm", "libdevice")
            _sys_libdevice = _os.path.join(_sys_cuda_path, "nvvm", "libdevice")
            if (
                not _os.path.isfile(_os.path.join(_conda_libdevice, "libdevice.10.bc"))
                and _os.path.isfile(_os.path.join(_sys_libdevice, "libdevice.10.bc"))
            ):
                _os.environ.setdefault("NUMBA_CUDA_LIBDEVICE", _sys_libdevice)

# Import any modules or sub-packages here
from . import obs, skynet

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^pkg_resources is deprecated as an API",
    category=UserWarning,
)

import cysgp4
import pycraf
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# Define any package-level variables or constants here

# Define any package-level functions or classes here



# Optionally, include an __all__ list to specify the public interface of the package

# Optionally, include any initialization code here
# Import the submodule here

# Optionally, add the submodule to the __all__ list
__all__ = ['skynet','obs','tlefinder','tleforger','antenna','earthgrid','satsim','scenario','visualise','gpu_accel','vis','custom_antenna', 'analytical_fixtures', 'custom_antenna_preview']

