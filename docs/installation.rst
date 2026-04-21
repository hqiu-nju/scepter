Installation
============

SCEPTer targets Python 3.10 and is distributed in this repository as a standard
Python package plus a notebook-oriented workflow.

Conda environments
------------------

Three Conda environment files are maintained in the repository. All
packages come from the ``conda-forge`` channel.

Standard environment
^^^^^^^^^^^^^^^^^^^^

Use the standard environment for simulation, desktop GUI, and
postprocessing on a machine with an NVIDIA GPU:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate scepter-dev

Full environment
^^^^^^^^^^^^^^^^

The full environment adds Cartopy (geographic map projections) and
Trame (notebook-embedded 3D viewer) on top of the standard set:

.. code-block:: bash

   conda env create -f environment-full.yml
   conda activate scepter-dev-full

CPU-only environment
^^^^^^^^^^^^^^^^^^^^

Use the CPU-only environment on machines without an NVIDIA GPU
(macOS, Linux-without-CUDA, Windows-without-CUDA). The desktop GUI
still imports and supports config editing, HDF5 inspection, and
postprocess recipes; the **Run Simulation** button is disabled because
the engine requires CUDA.

.. code-block:: bash

   conda env create -f environment-cpu.yml
   conda activate scepter-dev-cpu

Core dependencies
-----------------

The package relies primarily on:

- ``astropy`` for units, coordinates, and time handling
- ``cysgp4`` for orbit propagation
- ``pycraf`` for radio astronomy protection utilities and antenna models
- ``numpy`` for array-heavy simulation code

Optional capabilities such as CUDA acceleration, interactive 3D rendering, and
plot export live in the full environment.

Read the Docs build
-------------------

Read the Docs uses ``.readthedocs.yaml`` together with ``docs/conf.py`` and
``docs/requirements.txt``. The documentation pages are written manually so that
the hosted build does not depend on importing every scientific runtime
dependency.
