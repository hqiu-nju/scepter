# SCEPTer

**S**imulating **C**onstellation **E**mission **P**atterns for **Te**lescopes (**r**adio)

SCEPTer is a modular Python toolkit for simulating satellite constellation
emissions and evaluating their impact on radio astronomy observations.  It
computes equivalent power flux density (EPFD) at Radio Astronomy Service (RAS)
stations from large non-geostationary satellite constellations, following
ITU-R methodologies.

## Key capabilities

- **Synthetic constellation generation** from multi-belt orbital definitions
  (altitude, inclination, plane count, satellites per plane)
- **Real-world TLE support** for propagating actual satellite catalogs
  (e.g. from [CelesTrack](https://celestrak.org/))
- **GPU-accelerated direct-EPFD simulation** using CuPy/Numba CUDA kernels
  for high-throughput batch processing
- **ITU-R antenna patterns**: S.1528 Recommends 1.2 and 1.4, M.2101
  phased-array, S.672 (GSO)
- **Frequency reuse and spectrum planning** with configurable channelisation,
  TX emission masks, and RAS receiver response models
- **Surface-PFD cap enforcement** (per-beam and per-satellite aggregate) to
  bound peak power flux density on Earth's surface
- **Boresight-avoidance modelling** with configurable exclusion zones
- **Mixed-precision GPU profiles** (fp64, fp64/fp32, fp32, fp32/fp16) for
  memory/precision trade-offs
- **Desktop GUI** (PySide6) with interactive constellation viewer, hexgrid
  cell-status maps, run monitor, and integrated postprocessing studio
- **Beam-cap (N-beam) sizing analysis** with tail-risk, SLA-delta, and
  SLA-epsilon metrics

## Quick start

### Installation

SCEPTer uses Conda environments.  Two configurations are provided:

**Standard** (simulation + GPU + desktop GUI):

```bash
conda env create -f environment.yml
conda activate scepter-dev
pip install -e .
```

**Full** (standard + Cartopy map projections + Trame 3D remote rendering):

```bash
conda env create -f environment-full.yml
conda activate scepter-dev-full
pip install -e .
```

Both environments include GPU acceleration, the desktop GUI, and the
postprocessing studio.  The full environment adds Cartopy for geographic
map projections and Trame for notebook-embedded 3D viewers.
All packages are sourced from `conda-forge`.

### Running the desktop GUI

```bash
conda activate scepter-dev
python gui.py
```

The GUI starts with an empty workspace.  Add constellation belts, configure a
RAS station, set antenna parameters, and run simulations from the interface.

### Running from notebooks

The simulation and postprocessing workflow notebooks are in the repository root:

- `SCEPTer_simulate.ipynb` — full simulation workflow
- `SCEPTer_postprocess.ipynb` — result analysis and visualization

## GPU requirements

- NVIDIA GPU with CUDA Compute Capability >= 7.0 (Volta or newer)
- Minimum 4 GB VRAM; 8+ GB recommended for large constellations
- System-level NVIDIA CUDA drivers must be installed
- The Conda `environment-full.yml` provides the CUDA toolkit libraries

## Architecture overview

### Simulation pipeline (per batch)

```
propagate (SGP4 → ECI positions)
  → derive_from_eci (geometry: topo, az/el)
    → link_library.add_chunk (cell visibility + candidate pairs)
      → beam_finalize (greedy beam assignment)
        → power (antenna pattern + FSPL + atmosphere → EPFD/PFD)
          → export_copy (histograms + device-to-host transfer)
            → write_enqueue (async HDF5 output)
```

### Core modules

| Module | Purpose |
|--------|---------|
| `scepter.gpu_accel` | GPU session, propagation, pattern evaluation, power accumulation |
| `scepter.scenario` | Simulation runner, scheduler, HDF5 I/O, integration helpers |
| `scepter.scepter_GUI` | Desktop GUI application |
| `scepter.tleforger` | Synthetic TLE constellation generation |
| `scepter.earthgrid` | Hexagonal Earth grid, contour analysis, geography masking |
| `scepter.visualise` | Plotting: CDF/CCDF, hemisphere maps, cell-status maps |
| `scepter.satsim` | CPU satellite link selection and beam assignment |
| `scepter.nbeam` | Beam-cap (N-beam) sizing analysis |
| `scepter.obs` | Observation simulation pipeline |
| `scepter.skynet` | Sky grid generation and scheduling |
| `scepter.antenna` | Antenna pattern definitions |
| `scepter.angle_sampler` | Angular sampling utilities |

### Shared library helpers

These functions are used by both the GUI and the workflow notebooks:

```python
from scepter import earthgrid, scenario, visualise

earthgrid.summarize_contour_spacing(...)
earthgrid.prepare_active_grid(...)
earthgrid.resolve_theta2_active_cell_ids(...)
visualise.plot_cell_status_map(...)
scenario.build_observer_layout(...)
scenario.run_gpu_direct_epfd(...)
```

## Testing

Run the full test suite:

```bash
pytest --basetemp=.pytest-tmp -v
```

Run specific test categories:

```bash
pytest scepter/tests/test_scenario.py -v       # simulation logic
pytest scepter/tests/test_gui.py -v             # GUI (requires PySide6)
pytest scepter/tests/test_gui_interactive.py -v # GUI interaction tests
pytest scepter/tests/test_gui_bootstrap.py -v   # splash screen / icons
```

GPU tests are automatically skipped when no CUDA device is available.

### Parallel testing

```bash
pytest -n auto --dist loadgroup   # auto-detect CPU count
```

GPU tests are grouped to run sequentially on one worker while CPU-only tests
spread across all workers.

## API examples

### Belt definition and TLE generation

```python
from astropy import units as u
from scepter import tleforger

belt_definitions = [
    {
        "belt_name": "LEO_Shell_1",
        "num_sats_per_plane": 120,
        "plane_count": 28,
        "altitude": 525 * u.km,
        "eccentricity": 0.0,
        "inclination_deg": 53.0 * u.deg,
        "argp_deg": 0.0 * u.deg,
        "RAAN_min": 0 * u.deg,
        "RAAN_max": 360 * u.deg,
        "min_elevation": 20 * u.deg,
        "adjacent_plane_offset": True,
    }
]

tleforger.reset_tle_counter()
constellation = tleforger.forge_tle_constellation_from_belt_definitions(
    belt_definitions
)
```

### GPU propagation

```python
import numpy as np
from scepter import gpu_accel

session = gpu_accel.GpuScepterSession(
    compute_dtype=np.float32,
    sat_frame="xyz",
)
sat_ctx = session.prepare_satellite_context(tles, method="vallado")
obs_ctx = session.prepare_observer_context(observers)

with session.activate():
    result = session.propagate_many(
        mjds, sat_ctx,
        observer_context=obs_ctx,
        do_topo=True, do_sat_azel=True,
    )
session.close(reset_device=False)
```

### Direct-EPFD benchmark

```python
from scepter import scenario

summaries = scenario.benchmark_direct_epfd_runs_from_gui_config(
    "benchmark_config.json",
    timestep_values_s=(30.0, 5.0, 1.0),
)
```

## Land/coast masking

The hexgrid supports optional land/coast Earth-grid masking:

- `shapely` is required (included in both environments)
- `cartopy` is optional (full environment only) for Cartopy-backed map
  rendering
- Falls back to plain Matplotlib axes when Cartopy is unavailable

## Repository

- Original: https://github.com/hqiu-nju/scepter
- Active migration target: SKAO GitLab (in progress)

## Contributing

For consistent high-quality changes, read the contributor guidance:

- `AGENTS.md` — repository-wide code quality rules
- `.github/copilot-instructions.md` — AI assistant generation rules

When making dependency changes, keep `environment.yml`, `environment-full.yml`,
`requirements.txt`, and `setup.py` aligned.

## License

SCEPTer is distributed under the **GNU General Public License v3.0 or later**
(GPLv3+).  See `LICENSE` for the full text.

Third-party dependency notices are in `THIRD_PARTY_NOTICES.md`.  Bundled
licence texts are in the `LICENSES/` directory.
