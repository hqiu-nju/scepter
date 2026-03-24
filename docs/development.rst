Development
===========

Repository entry points
-----------------------

The Read the Docs build is driven by three files:

- ``.readthedocs.yaml`` selects the Python version and the Sphinx entry point.
- ``docs/conf.py`` configures the Sphinx project.
- ``docs/index.rst`` defines the table of contents for the published site.

Building the docs locally
-------------------------

The Conda environments include the documentation toolchain. To build the HTML
site locally:

.. code-block:: bash

   conda activate scepter-dev
   sphinx-build -b html docs docs/_build/html

If your existing environment predates the documentation dependencies, refresh it
first:

.. code-block:: bash

   conda env update -n scepter-dev -f environment.yml
   conda activate scepter-dev
   sphinx-build -b html docs docs/_build/html

You can do the same from the full environment by replacing ``scepter-dev`` with
``scepter-dev-full``.

Documentation scope
-------------------

The documentation in this repository is intentionally curated rather than
generated directly from imports. Several runtime modules depend on optional
scientific packages and GPU libraries, so hand-written pages give a more stable
Read the Docs build while still documenting the repository structure and
workflow.

When updating the docs
----------------------

- Keep environment commands copy/paste-ready.
- Prefer short workflow examples over large notebook dumps.
- Document optional GPU and visualisation features as extras, not as baseline
  requirements.
- Update both Conda environment files if documentation tooling changes.
