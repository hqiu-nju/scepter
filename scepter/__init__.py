
"""
Package Name
=============

A brief description of the package.

"""

__doc__ = "SCEPTer, Simulating Constellation Emission Patterns for Telescopes"
__version__ = "0.25.2"

# On Windows, conda environments require Library/bin on PATH for DLL
# loading (BLAS, CUDA, OpenSSL, etc.).  Fix this before any C-extension
# imports so that ``import scepter`` works without ``conda activate``.
#
# Also override CUDA_PATH when the conda environment ships its own CUDA
# headers, so CuPy's NVRTC compiler uses the conda headers instead of a
# potentially incompatible system-installed CUDA toolkit.
import os as _os
import sys as _sys
if _sys.platform == "win32":
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
__all__ = ['skynet','obs','tlefinder','tleforger','antenna','earthgrid','satsim','scenario','visualise','gpu_accel']

