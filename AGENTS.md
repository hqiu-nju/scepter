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

| Module | Responsibility | Key functions/classes |
|---|---|---|
| `__init__.py` | Package metadata, Windows stream/DLL setup, package-level imports | `__version__`, `__codename__`, `__all__` |
| `angle_sampler.py` | Sky-grid angle sampling (S.1586), joint histograms, distribution metrics | `JointAngleSampler`, `joint_hist_metrics`, `sliced_wasserstein` |
| `antenna.py` | Satellite & RAS antenna patterns (S.1528 Rec 1.2 / Rec 1.4, M.2101, S.672, RA.1631), optional Numba | `build_satellite_pattern_spec`, `isotropic_pattern`, `s_1528_rec1_2_pattern`, `s_1528_rec1_4_pattern_amend`, `calculate_beamwidth_1d`, `calculate_beamwidth_2d` |
| `custom_antenna.py` | User-supplied LUT patterns, schema v1 (authoritative format in the module docstring) — loader, CPU evaluators, dump/inspect CLI | `CustomAntennaPattern`, `PatternPeakWarning`, `load_custom_pattern`, `dump_custom_pattern`, `evaluate_pattern_1d`, `evaluate_pattern_2d`, `format_pattern_summary`, `main` |
| `analytical_fixtures.py` | Helpers that sample any analytical pattern onto a user-chosen grid and produce a `CustomAntennaPattern` — used by tests and by users who want an LUT from an ITU formula | `sample_analytical_1d`, `sample_analytical_2d_az_el`, `sample_analytical_2d_theta_phi`, `ra1631_evaluator`, `s1528_rec1_2_evaluator`, `s1528_rec1_4_evaluator`, `m2101_evaluator` |
| `custom_antenna_preview.py` | Pure-matplotlib figure factory for loaded custom patterns (polar 1-D, heatmap + principal-plane cuts 2-D) | `build_custom_pattern_preview_figure` |
| `earthgrid.py` | Hex-grid generation, footprint geometry, geography masks, frequency reuse, optional Numba | `calculate_footprint_size`, `generate_hexgrid_full`, `trunc_hexgrid_to_impactful`, `resolve_frequency_reuse_slots`, `resolve_ras_hexgrid_cell_ids`, `mask_hexgrid_for_constellation`, `recommend_cell_diameter`, `prepare_active_grid` |
| `gpu_accel.py` | GPU-accelerated SGP4, angular distance, CuPy kernels, session/cache lifecycle | `true_angular_distance_gpu`, `true_angular_distance_auto`, `_compute_aggregate_surface_pfd_cap_cp`, `GpuScepterSession`, `GpuScepterSession.prepare_peak_pfd_lut_context` |
| `nbeam.py` | Beam-cap sizing with visibility-aware pooling policies | `BeamCapSizingConfig`, `PolicyMeta`, `PolicyEval`, `StreamingPolicyAccumulator`, `run_beam_cap_sizing`, `plot_tail_risk`, `plot_sla` |
| `obs.py` | Observation simulation (propagation + antenna → RFI) | `receiver_info`, `transmitter_info`, `obs_sim`, `prx_cnv`, `pfd_to_Jy` |
| `satsim.py` | Link-selection helpers (CPU/GPU variants), beam assignment, pure-reroute service curves | `SatelliteLinkSelectionLibrary`, `select_satellite_links`, `summarize_link_selection`, `pure_reroute_service_curve`, `ConditionedTemplatePlanCpu`, `assign_conditioned_beams_cpu` |
| `scenario.py` | Scenario orchestration, HDF5 streaming I/O, integration, direct EPFD drivers | `iter_simulation_batches`, `recommend_observer_chunk_size`, `generate_simulation_batches`, `write_data`, `read_data`, `analyse_time`, `process_integration`, `run_gpu_direct_epfd` |
| `skynet.py` | Sky-grid generation (S.1586 implementation) | `pointgen_S_1586_1`, `pointgen`, `gridmatch`, `plantime`, `plotgrid` |
| `tleforger.py` | Synthetic TLE generation from belt definitions | `reset_tle_counter`, `normalize_and_validate_belt_cfg`, `normalize_and_validate_belt_definitions`, `forge_tle_constellation_from_belt_definitions`, `summarize_constellation_geometry`, `expand_belt_metadata_to_satellites`, `build_satellite_storage_constants`, `forge_tle_single`, `forge_tle_belt` |
| `tlefinder.py` | TLE archive discovery and catalog helpers | `TLEfinder`, `id_locator`, `parse_sgp4info`, `readtlenpz` |
| `uvw.py` | UVW coordinate transforms, geometric delays, visibility simulation, telescope-array/TLE loaders | `hour_angle`, `baseline_bearing`, `baseline_pairs`, `geometric_delay_az_el`, `fringe_attenuation`, `fringe_response`, `pointing_geometric_delay`, `satellite_geometric_delay`, `satellite_visibility_phase`, `normalised_visibility_amplitude`, `simulate_satellite_visibilities`, `VisibilityNpzArchive`, `save_visibility_npz`, `load_visibility_npz`, `itrf_to_enu`, `enu_to_uvw`, `compute_uvw`, `AntennaArrayGeometry`, `TrackingUvwResult`, `load_telescope_array_file`, `load_tle_files`, `load_tle_files_with_names`, `TrackingUvwBuilder`, `build_tracking_uvw` |
| `visualise.py` | CDF/CCDF, hemispheres, sky maps, animation export, cell-status and reuse maps | `plot_cdf_ccdf`, `plot_cdf_ccdf_from_histogram`, `plot_satellite_elevation_pfd_heatmap`, `plot_hemisphere_2D`, `plot_hemisphere_3D`, `satellite_distribution_over_sky`, `plot_cell_status_map`, `plot_frequency_reuse_scheme` |
| `postprocess_recipes.py` | HDF5 inspection and plot recipes | `RecipeParameter`, `PostprocessRecipe`, `default_recipe_parameters`, `normalize_recipe_parameters`, `resolve_recipe_parameter_state`, `recipe_capability`, `recipe_availability`, `inspect_result_file`, `resolve_recipe_source`, `render_recipe` |
| `gui_bootstrap.py` | App identity, splash screen (SVG icon via QSvgRenderer, rounded corners, fade-out, non-blocking), icon loading (lightweight; depends on PySide6.QtSvg) | `configure_application`, `configure_windows_shell_identity`, `apply_windows_window_icon`, `resolve_app_icon_path`, `load_app_icon`, `ScepterSplashScreen`, `build_startup_splash`, `create_startup_splash` |
| `scepter_GUI.py` | Full desktop GUI (~23 k lines) | `ScepterMainWindow`, `ScepterProjectState`, `save_project_state`, `load_project_state`, `build_constellation_from_state`, `validate_project_state`, `build_preview_frames`, `create_main_window`, `launch_gui` |
| `appinfo.py` | Shared version/branding metadata | `format_gui_window_title`, `APP_NAME`, `APP_VERSION`, `APP_CODENAME`, `ABOUT_TEXT` |
| `conftest.py` | Test-suite configuration for headless plotting and GUI/event-loop helpers | pytest fixtures/hooks for tests |
| `conftest_rich.py` | Optional Rich-powered live pytest terminal UI | `_ScepterTestState`, `ScepterRichPlugin`, `pytest_configure` |
| `data/__init__.py` | Package marker for bundled data assets | no public API |
| `tools/__init__.py` | Package marker for maintenance tools | no public API |
| `tools/build_app_icon.py` | Regenerates the multi-resolution Windows `.ico` from the Qt icon renderer | `main` |

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
| `scepter` | 3.10 | Core simulation no cpu; preferred for radio astronomy algorithm testing |
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
