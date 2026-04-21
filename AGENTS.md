# Agent Guidelines for SCEPTer

These instructions apply to the entire repository and are the baseline quality
bar for AI agents and automated helpers.

## 0) Project overview

SCEPTer (**S**atellite **C**onstellation **E**mission **P**attern simulator for
radio **Te**lescopes) models EPFD impact from non-GSO satellite systems on
radio-astronomy stations.  The toolkit covers TLE/orbit generation, antenna
patterns (ITU-R S.1528), propagation, GPU-accelerated EPFD computation, and
post-processing/visualisation.

### Entry points

| Entry point | Purpose |
|---|---|
| `gui.py` | Desktop GUI launcher (PySide6) — bootstrap, splash, then `scepter_GUI` |
| `scepter/scepter_GUI.py` | Main GUI window — constellation editor, RAS geometry, grid analyser, 3-D viewer |
| Jupyter notebooks (`xxx_*.ipynb`, `$4RSA_*.ipynb`) | Analysis / batch simulation workflows |

### Module map (`scepter/`)

| Module | Responsibility |
|---|---|
| `angle_sampler.py` | Sky-grid angle sampling (S.1586) |
| `antenna.py` | Satellite & RAS antenna patterns (S.1528 Rec 1.2 / Rec 1.4, M.2101, S.672, RA.1631), optional Numba |
| `custom_antenna.py` | User-supplied LUT patterns, schema v1 (authoritative format in the module docstring) — loader, CPU evaluators, dump/inspect CLI |
| `analytical_fixtures.py` | Helpers that sample any analytical pattern onto a user-chosen grid and produce a `CustomAntennaPattern` — used by tests and by users who want an LUT from an ITU formula |
| `custom_antenna_preview.py` | Pure-matplotlib figure factory for loaded custom patterns (polar 1-D, heatmap + principal-plane cuts 2-D) |
| `earthgrid.py` | Hex-grid generation, footprint geometry, optional Numba |
| `gpu_accel.py` | GPU-accelerated SGP4, angular distance, CuPy kernels |
| `nbeam.py` | Beam-cap sizing with visibility-aware pooling policies |
| `obs.py` | Observation simulation (propagation + antenna → RFI) |
| `satsim.py` | Link-selection helpers (CPU/GPU variants) |
| `scenario.py` | Scenario orchestration, HDF5 streaming I/O, integration |
| `skynet.py` | Sky-grid generation (S.1586 implementation) |
| `tleforger.py` | Synthetic TLE generation from belt definitions |
| `tlefinder.py` | TLE archive discovery and catalog helpers |
| `visualise.py` | CDF/CCDF, hemispheres, sky maps, animation export |
| `postprocess_recipes.py` | HDF5 inspection and plot recipes |
| `gui_bootstrap.py` | App identity, splash screen (SVG icon via QSvgRenderer, rounded corners, fade-out, non-blocking), icon loading (lightweight; depends on PySide6.QtSvg) |
| `scepter_GUI.py` | Full desktop GUI (~23 k lines) |
| `appinfo.py` | Shared version/branding metadata |

### Assets (`scepter/data/`)

- `satellite_app_icon.svg` / `.ico` — application icon (SVG is the
  high-fidelity source rendered via `QSvgRenderer` in the splash screen;
  ICO is for Windows native taskbar)
- `scepter_brand_mark.svg` — brand mark
- `earth_texture_nasa_flat_earth_8192.jpg` — 3-D viewer Earth texture
- `ne_10m_coastline.geojson`, `ne_10m_land.geojson` — Natural Earth 10 m data

### Environments and testing

| Environment | Python | Scope |
|---|---|---|
| `scepter-dev` | 3.10 | Core simulation + GPU (CuPy, Numba); preferred for benchmarking |
| `scepter-dev-full` | 3.13 | Full stack: PySide6, PyVista, Numba, CuPy (may have issues) |

```bash
conda env create -f environment.yml          # lite
conda env create -f environment-full.yml     # full
conda activate scepter-dev-full
python -m pytest --basetemp=.pytest-tmp -v   # run tests
python gui.py                                 # launch the GUI
```

Tests live in `scepter/tests/`.  GUI tests use `QT_QPA_PLATFORM=offscreen`.
No `pytest-qt` — event-loop management is manual via `QEventLoop`/`QTimer`.
`conftest.py` forces `matplotlib.use("Agg")` for headless plot rendering.

### Core dependencies

astropy (units/time), pycraf (ITU propagation), cysgp4 (SGP4/TLE), shapely
(geometry), PySide6 (Qt GUI), PyVista/PyVistaQt (3-D), h5py (HDF5), Numba
(optional JIT), CuPy (optional GPU arrays).

## 1) Start-up checks

- Read `.github/copilot-instructions.md` before proposing structural or code
  changes.
- Inspect the touched modules and match existing behavior unless a breaking
  change is explicitly requested.
- Keep changes scoped: avoid unrelated refactors in the same edit.

## 1.1) Diff hygiene (avoid no-op churn)

- Make the smallest possible patch for the requested change.
- Do not delete and re-add unchanged lines.
- Avoid file-wide reflow/reformatting when only local edits are needed.
- Preserve existing line endings and surrounding formatting unless a deliberate
  normalization is requested.
- Before finalizing, review `git diff` and remove no-op hunks (especially
  remove/add pairs with identical content).
- Exception: larger, cross-cutting updates are allowed when the user explicitly
  requests methodology-level or architecture-level improvements, or when a
  local patch cannot safely solve the issue. In these cases, state the broader
  scope and rationale before editing.

## 2) Code quality requirements

- Keep all code Python 3.10-compatible.
- Add or maintain type hints for new and modified public functions.
- Keep imports ordered: standard library, third-party, local.
- Use explicit names and avoid hidden side effects.
- Preserve unit-aware interfaces for physical quantities (for example, astropy
  units) and document any conversion assumptions.
- If touching performance-critical code paths, prefer vectorized approaches and
  briefly note performance tradeoffs in the summary.

## 3) Documentation and notebook requirements

- Update docs in the same change when behavior, APIs, or workflows change.
- Keep command examples copy/paste-ready.
- Refer to Conda environments exactly as `scepter-dev` (lite) and
  `scepter-dev-full` (full).
- For simulation-facing docs, include units and expected array shapes where
  relevant.
- For public/user-facing Python APIs, NumPy-style docstrings are the minimum
  bar and should include: short summary, extended description, `Parameters`,
  `Returns`, `Raises`, and `Notes` (plus `Examples` when practical).
- Documentation depth should be pycraf-like or better: explicitly document
  units, expected shapes/dimensionality, defaults, formulas/conventions,
  assumptions, edge cases, and reproducibility-relevant behavior (for example,
  global counters or ordering guarantees).
- Prefer more detail over less: this level is the minimum acceptable baseline,
  not the target ceiling.

## 4) Dependency and environment policy

- If dependencies change, update both `environment.yml` (lite) and
  `environment-full.yml` (full), unless the dependency is intentionally full-only.
- Keep heavy GPU/visualization packages in `environment-full.yml` unless they
  are strictly required for lite workflows.
- When setup commands or options change, update the README development
  environment section in the same edit.

## 5) Validation expectations

- Run targeted validation for the touched behavior (script, module smoke check,
  test, or notebook as appropriate).
- When simulation pipelines are modified, run a representative notebook or
  script path where feasible.
- If validation is skipped, state exactly what was skipped and why in the final
  summary.

## 6) Agent response expectations

- Summaries should include files changed and purpose, validation commands and
  outcomes, and an explicit note of skipped checks or known limitations.

## 7) GPU performance tuning

When optimizing `gpu_accel.py` or `scenario.py`, follow these guidelines:

### Benchmarking

Run `python run_benchmark.py 20` for quick 20-batch tests; use
`python run_benchmark.py full` for production validation.  Results go to
`benchmark_log.txt`.  Batch 0 includes JIT warmup — use "steady state (skip 2
warmup)" avg for meaningful comparisons.  Two configs are provided:
`benchmark_config.json` (standard, no boresight) and
`benchmark_boresight_config.json` (boresight avoidance).

### Memory budgets

Set `gpu_memory_budget_gb` and `host_memory_budget_gb` in config files.
8 GB VRAM machines: use 6/6 GB.  16 GB VRAM machines: 12/12 GB.

### Hot path architecture

Per-batch pipeline: `propagate → derive_from_eci → link_library.add_chunk →
beam_finalize → power → export_copy → write_enqueue`.

Dominant stages (non-boresight): **power** (~36%), **export_copy** (~33%),
**finalize** (~14%).  See `CLAUDE.md` for exact timings and line references.

### CuPy optimization principles

- Array shapes are **small** for non-boresight: `(T=49, S≈180, B≈30)`.
  Python dispatch + CUDA launch overhead (3-10 ms/op) dominates, not bandwidth.
- **Minimize CuPy operation count**: fuse sequential element-wise ops into
  `cp.ElementwiseKernel` or `cp.RawKernel`.
- **Avoid redundant `.astype(cp.float32, copy=False)`** on float32 arrays.
- **Precompute LUTs** for expensive pattern evaluations (S.1528 uses Bessel J1
  + product; a 0.001-deg LUT replaces this with a single interpolation kernel).
- **Avoid `cp.nonzero` + fancy indexing** for filtering — use `cp.where` with
  sentinel values when possible (fewer kernel launches).
- **Use `out=` parameter** for in-place operations (`cp.clip`, etc.).
- Profile with `profile_stages=True` in the benchmark config to get per-stage
  timing breakdowns.

### Key files and functions

| Function | File | Role |
|----------|------|------|
| `_accumulate_ras_power_cp` | `gpu_accel.py:~9852` | Power computation (biggest single stage) |
| `_evaluate_s1528_pattern_cp` | `gpu_accel.py:~9695` | Transmit pattern (uses LUT for phi=None) |
| `_evaluate_ras_pattern_cp` | `gpu_accel.py:~9667` | Receive pattern (fused ElementwiseKernel) |
| `_compute_gpu_direct_epfd_batch_device` | `scenario.py:~6843` | Batch orchestration |
| `run_gpu_direct_epfd` | `scenario.py:~7881` | Top-level iteration driver |
| `GpuScepterSession` | `gpu_accel.py:~11432` | Session lifecycle, caches, activation |
| `prepare_peak_pfd_lut_context` | `gpu_accel.py` | Builds the surface-PFD cap per-beam LUT. 1-D `K(β, shell)` for every pattern whose asymmetry rotates with the beam (S.1528 Rec 1.2, S.1528 Rec 1.4 both symmetric and asymmetric, Custom-1D, Custom-2D, isotropic); 2-D `K(α, β, shell)` for M.2101 whose element pattern is fixed in the sat body frame. |
| `_compute_aggregate_surface_pfd_cap_cp` | `gpu_accel.py` | Loop-free GPU-native per-satellite aggregate cap helper; supports the 3-D and 4-D paths and every TX-pattern family (S.1528 / M.2101 / Custom-1D / Custom-2D). |

### Surface-PFD cap (Service & Demand "Max PFD on Earth surface")

Optional feature that bounds the peak PFD any beam (or any satellite in
aggregate) can deposit on Earth's surface.  Enabled from the GUI
`Service & Demand` tab.  When editing or extending it, keep these invariants:

- The per-beam cap and the aggregate cap must produce identical outputs
  when a satellite has exactly one active beam.  Tested by
  `test_aggregate_cap_single_beam_matches_per_beam_analytic`.
- The cap must honour the run's `include_atmosphere` setting.  The LUT
  builder queries the same atmosphere LUT the hot power path uses.
- Both cap modes in the fused direct-EPFD path are **hoisted** out of the
  spectral-slab loop — the cap factor is computed once per batch and
  passed to each slab call via `precomputed_cap_factor_cp`.  Do not
  re-run the helpers inside a per-slab loop.  `per_satellite` in the
  4-D path is supported **only** via the hoisted precomputed factor;
  direct kernel callers without the hoisted tensor get a clear error.
- Pair midpoints in the candidate set are dropped above `K_act = 32` for
  memory / compute reasons (physically exact for narrow-beam
  constellations).  If you relax this, update
  `test_aggregate_cap_large_k_skips_pair_midpoints`.
- The cap path is **GPU-native with no Python loops**.  Any refactor that
  reintroduces a per-group Python loop is a regression even if the test
  suite passes.
- **1-D `K(β)` LUT** (``GpuPeakPfdLutContext.is_2d == False``) is used
  for every transmit pattern whose asymmetry rotates *with* the beam
  boresight — the full-360° observation sweep of the builder then
  renders the result α-invariant. This covers S.1528 Rec 1.2,
  S.1528 Rec 1.4 (both symmetric and asymmetric), Custom-1D,
  Custom-2D, and isotropic. The asymmetric S.1528 and Custom-2D
  builders use a 2-D `(ψ, φ)` observation sweep so the φ-dependent
  main-lobe peak is captured correctly; the output is still 1-D.
- **2-D `K(α, β)` LUT** (``is_2d == True``) is only used for M.2101 —
  its element pattern is fixed in the sat body frame, so K genuinely
  depends on the beam's steering azimuth.
- All five transmit-pattern families (S.1528 Rec 1.2, S.1528 Rec 1.4
  sym and asym, M.2101, Custom-1D, Custom-2D) and both cap modes
  (per-beam / per-satellite aggregate) are supported in the 3-D and
  4-D boresight-avoidance paths.
