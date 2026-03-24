Installation
============

SCEPTer targets Python 3.10 and is distributed in this repository as a standard
Python package plus a notebook-oriented workflow.

Conda environments
------------------

Two Conda environment files are maintained in the repository.

Lite environment
^^^^^^^^^^^^^^^^

Use the lite environment for core simulation work and documentation builds:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate scepter-dev

Full environment
^^^^^^^^^^^^^^^^

Use the full environment when you need optional GPU acceleration or the 3D
visualisation stack:

.. code-block:: bash

   conda env create -f environment-full.yml
   conda activate scepter-dev-full

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
