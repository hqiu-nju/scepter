# Agent Guidelines for SCEPTer uvw development

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

| Jupyter notebooks (`xxx_*.ipynb`, `$4RSA_*.ipynb`) | Analysis / batch simulation workflows |
| CLI scripts (scripts/*.py)| example simulations for radio astronomy visibilties |

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


### Environments and testing

| Environment | Python | Scope |
|---|---|---|

| `scepter` | 3.10 | Core simulation no cpu; preferred for radio astronomy algorithm testing |



```bash
conda activate scepter
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