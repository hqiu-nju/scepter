# SCEPTer
**S**imulating **C**onstellation **E**mission **P**atterns for **Te**lescopes (**r**adio)

This is a modular package to systematically simulate the satellite constellation emissions and measure the EPFD of the observed sky area.

The simulation can be performed using a simulated constellation or from real satellite constellation two line elements (TLEs).
Satellite TLEs can be found on https://celestrak.org/

We use the PyCRAF and cysgp4 packages for many of the base calculations, see requirements.txt for list of dependancies, code was written and tested in Python 3.10

## Development environment

Two Conda environment files are provided so you can choose between a lighter install and the fully accelerated/visualization stack:

- **Lite (default)** – core dependencies only, quickest to create: `conda env create -f environment.yml && conda activate scepter-dev`
- **Full** – includes optional GPU acceleration and visualization packages (`numba`, `cuda-version>=12.0`, `pyvista`, `plotly`, `python-kaleido`, `trame`, `trame-jupyter-extension`, `jupyter-server-proxy`, `trame-vtk`, `trame-vuetify`): `conda env create -f environment-full.yml && conda activate scepter-dev-full`

If you start with the lite environment and later need the optional packages, update it in place with:

```bash
conda env update -n scepter-dev -f environment-full.yml
python -m ipykernel install --user --name scepter-dev
```

All packages are pulled from `conda-forge`, including CUDA-enabled builds where available, so Codespaces or local shells using either environment will align with the expected toolchain.

GitHub Copilot users can refer to `.github/copilot-instructions.md` for project-specific guidance to keep suggestions consistent with the codebase. Agents and collaborators can consult `AGENTS.md` for repository-wide instructions on environment updates and testing expectations.

### Simulation Example Figure

![Simulation Grid](./notebooks/example.png)

## Running the observation simulator

In the obs module, we have the tools for creating an observation simulation that will provide an RFI sky model from satellite emissions.

The simulation is organised through multiple dimensions to give consideration of the telescope pointings, satellite design, constellation beam patterns through a series of time.

scepter operates in a dynamic cube to store the measurements corresponding to each antenna and satellite pairs.

### Simulation grid explanation

Currently, 6 dimensions are used:
1. observers/antennas (cysgp4 pyobserver object)
1. antenna pointings per grid
1. sky grid cells (skygrid pointings using skynet.pointgen)
1. epochs (separate observations)
1. nint, the subintegrations during an observation
1. number of transmitter/satellites

Github Repo: https://github.com/hqiu-nju/scepter
Code will be moving to SKAO Gitlab in the future.

