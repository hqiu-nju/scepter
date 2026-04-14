# Changelog

All notable user-facing changes are summarized here at a high level.

## v0.25.1

### UEMR (unwanted emissions) mode

- New per-satellite isotropic emission mode for modelling unwanted
  emissions from internal satellite circuitry that are not coupled to
  the directive antenna. Each visible satellite radiates the
  configured Ptx/EIRP isotropically, once per satellite; the beam
  library, selection strategy, cell activity, coverage tabs,
  boresight avoidance, and surface-PFD cap are all bypassed because
  they have no meaning without beams.
- Enabled via the **Isotropic UEMR mode** checkbox on the Satellite
  Antennas tab (when the antenna model is set to isotropic).
- Toggling UEMR hides irrelevant Service & Demand fields and coverage
  tabs; toggling it off restores them. The transmit emission mask on
  the Spectrum tab still shapes UEMR power across frequency.
- **"Set Service Defaults"** populates UEMR-appropriate starting
  values: EIRP basis, per-MHz, −71 dBW/MHz (CISPR-32 Class B
  quasi-peak at 3 m, 1 MHz measurement BW, ≈ 2 GHz).
- UEMR-specific kernel-required defaults are forced automatically:
  flat (rectangular) transmit emission mask, channel bandwidth =
  service-band width, power basis = per-MHz, power variation = fixed,
  integration cutoff = 50 % of service bandwidth (clipping the mask
  exactly at the service-band edges), reuse factor = 1.
- Sidebar and tab subtitles for **Spectrum** and **Service** switch
  to UEMR-appropriate descriptive copy when UEMR is active.

### Plot polish

- Matplotlib rectangular hemisphere maps no longer draw
  `draw_guides`-style arrow overlays on top of the native axes
  (these only make sense for the polar projection and produced a
  duplicated "arrow + spine" look). Opt in on rect via
  `draw_guides=True`.
- The operational-range lower elevation (e.g. 15°) is now always
  ticked on the y-axis of rectangular hemisphere maps — the default
  locator was hiding the most load-bearing tick in favour of round
  decades.

### Test suite

- Fixed flaky `test_contour_analyser_updates_guidance_immediately`
  that depended on module-level state left by other tests. The test
  now asserts only the robust invariant (analyser completion updates
  the recommended-spacing label).

## v0.25.0

### GPU device selection

- Added GPU device selector dropdown in Review & Run for multi-GPU systems.
- Auto-hides when only one GPU is detected; shows device name and VRAM size for each available GPU.
- `gpu_device_id` persisted in project config and threaded through `run_gpu_direct_epfd`.

### Postprocessing UX

- Source selector now shows "(no data)" suffix on unavailable sources (e.g. "raw (no data)") instead of silently greying them out.
- "auto" source now shows which data path will be used: "auto → preaccumulated" or "auto → raw".
- Fixed: selecting "raw" source for a recipe without raw data no longer crashes with "requires raw data" error; the combo redirects to "auto" and the item is greyed out.

### GUI layout improvements

- Satellite Antennas tab: frequency and antenna parameter panels now use side-by-side layout instead of a single narrow column.
- Antenna model parameters (Rec 1.2, Rec 1.4, M.2101) use adaptive two-column forms when there are 5+ fields, filling available horizontal space.
- Service & Demand tab: "Enforce max PFD on Earth surface" moved to a checkable group box with zero-jitter show/hide of cap fields.
- Service & Demand tab: rebalanced panel widths (1:2 stretch ratio for subscriber demand vs power target).

### Run monitor

- Added explanatory tooltips on all four resource cards (RAM, VRAM, CPU, GPU) explaining what each metric means, including RSS, WDDM VRAM noise, and GPU utilisation interpretation.

## v0.24.0

### Surface-PFD cap: M.2101 phased-array support

- M.2101 patterns now fully supported for both per-beam and per-satellite aggregate cap modes via a 2-D `K(α, β)` LUT indexed by beam steering azimuth and off-nadir angle.
- Bilinear lookup at runtime (`_lookup_peak_pfd_k_2d_cp`) with 360° azimuth wrapping; unified dispatch via `_lookup_peak_pfd_k_any_cp`.
- Aggregate cap helper (`_compute_aggregate_surface_pfd_cap_cp`) extended with M.2101 dispatch using body-frame `(az, el)` coordinates for candidate/beam pattern evaluation.
- Removed all `NotImplementedError` guards for M.2101 cap; all three antenna families (S.1528, S.1528 Rec 1.2, M.2101) work in both cap modes across both 3-D and 4-D paths.

### Surface-PFD cap: 4-D boresight-avoidance support

- `surface_pfd_cap_mode="per_satellite"` now works in the 4-D boresight-avoidance path via hoisted precomputed factor.
- Fused hoist handles `(T, N_sky, S)` aggregate and `(T, N_sky, S, K)` per-beam shapes with correct active-index gather.
- Direct callers without the hoisted tensor get a clear error pointing to the fused wrapper.

### Surface-PFD cap: per-beam spectral-slab hoist

- Per-beam cap factor (`(T, S, K)` or `(T, N_sky, S, K)`) is now computed once per batch and passed to each spectral-slab call via `precomputed_cap_factor_cp`.
- Drops per-beam cap overhead from +6–14% to ≈0% (within measurement noise).

### Surface-PFD cap: D2H sync fusion

- Fused 3 separate `.get()` calls (count, sum, max) into a single `cp.array([...]).get()` for all cap stats collection paths.

## v0.23.0

### GPU kernel fusion pass

- **Fused atmosphere LUT kernel**: replaced ~9 separate CuPy operations per atmosphere lookup with a single `ElementwiseKernel`. Called twice per power slab (station + target), eliminating ~18 kernel launches per slab.
- **Fused TX trig geometry kernel**: replaced ~12 separate CuPy operations (sin/cos/where/clip chain) with a single `ElementwiseKernel` computing `beam_sinb`, `beam_cosb`, and `cos_gamma_tx` from raw satellite az/el and beam α/β.
- **Fused RX trig + RAS pattern**: extended the existing 4-D fused kernel to the 3-D path via broadcast, eliminating ~16 kernel launches for receive-side angular distance + RAS pattern evaluation.
- **Fused beam→ground geometry**: combined slant-range discriminant + target elevation arccos into a single `ElementwiseKernel`, eliminating ~16 separate operations.
- All fusions gate on `pwr_dt == cp.float32` with sequential fallback for fp64 precision profiles.
- Power stage dropped from 193 ms → 78 ms (−60%); wall time from 314 ms → 185 ms (−41%) on the WRC benchmark config.

### Scheduler improvements

- Power-stage byte estimator updated to reflect post-fusion intermediate counts, distinguishing fused (fp32) from sequential (fp64) paths and 3-D from 4-D boresight.
- Added 5% safety margin to visible satellite estimate on top of the headroom profile factor.
- Cap memory pressure accounted for in the scheduler: per-beam mode budgets the hoisted `(T, S, K)` cap factor tensor; per-satellite mode budgets the aggregate helper transient.
- WDDM live-fit false-positive OOM fix (Windows only): 85% floor on the live-fit budget prevents momentary free-memory dips from triggering false OOM events.

## v0.22.0

### Test suite optimisation

- Exhaustive channel-subset leakage tests (14-channel: 169 s → 1.4 s; 12-channel: ~70 s → 1.2 s) replaced with representative subsets covering all structural edge cases.
- Option matrix smoke test (~1008 combinations → ~120 representative combos): 5× faster.
- Full `test_scenario.py` suite: 341 s → 36 s (9.4× speedup).
- Qt memory leak fix: added `sendPostedEvents(None, DeferredDelete)` cleanup fixture preventing ~33 MB/window accumulation across 288 GUI tests.
- Module-scoped asset stubs for GUI tests to avoid per-test monkeypatch overhead.

### Parallel testing support

- `pytest-xdist` configured with GPU-safe auto-grouping: GPU tests serialised on one worker, CPU-only tests spread across all workers.
- Usage: `pytest -n auto --dist loadgroup`.

### Rich terminal UI for tests

- Custom pytest plugin (`conftest_rich.py`) providing live progress table with spinner animation, per-file pass/fail/skip counters, slowest-test ranking, and colour-coded summary.
- Auto-disables under xdist workers, verbose mode, and explicit traceback modes.

### Testing Suite GUI improvements

- Three-state category icons: ⬜ idle → ⏳ queued → ◐◓◑◒ spinning → ✅ pass / ❌ fail.
- Spinner animation (200 ms) for actively running categories.
- Categories finalize when pytest moves to the next file (no more eternal spinners).
- Live test counts shown in description column ("12 done, 1 failed").

## v0.21.0

### Bug fixes

- Fixed `math.isfinite` NameError in GUI `_run_readiness_payload` (replaced with `np.isfinite`; GUI-only validation, not simulation path).
- Fixed M.2101 Astropy Quantity TypeError in `run_gpu_direct_epfd` when GUI state passes degree-unit values to pattern context builder.
- Fixed "Show RAS Antenna Pattern" button disabled when only RAS frequency is set (was incorrectly requiring satellite antenna frequency).
- Fixed `_tiny_state()` test fixture missing RAS receiver band fields causing spectrum validation failures.
- Fixed `test_run_simulation_readiness_rejection` asserting exact error message text that shifted with validation order.
- Fixed `test_multi_system` GPU tests FAILING instead of SKIPPING when CuPy is importable but CUDA is unavailable.
- Fixed `test_replans_before_beam_finalize` depending on actual VRAM availability; now mocks the live memory snapshot.
- Fixed boresight notebook test budgets (1.0/0.25 GB → 4.0/1.0 GB) for 4 GB minimum VRAM target.
- Fixed postprocess source combo allowing selection of unavailable sources.

## v0.20.0

### Multi-satellite-system support

- Introduced `SatelliteSystemConfig` data model: each project can now define multiple satellite systems with independent constellation, antenna, service, spectrum, boresight, and grid settings.
- Added system tab bar in the GUI for switching between systems; right-click to rename, delete, or duplicate a system.
- Per-batch GPU interleaving: multiple systems are processed within each time batch (not sequential full runs), enabling future inter-system interaction constraints for WRC-27 AI 1.16 studies.
- Per-system HDF5 output: each system's raw iterations, preaccumulated distributions, and constants are stored under `/system_N/` groups alongside combined root-level results.
- System output groups (`SystemOutputGroup`): define named subsets of systems for custom combined accumulation and filtering.
- Postprocess system filter: all recipes (CCDF, heatmap, beam statistics, beam cap sizing, time series) respect the active system selection and read from the correct per-system HDF5 group.
- System overlay mode: plot all systems' CCDFs on a single set of axes with per-system colours and a combined curve.

### Antenna patterns

- Added ITU-R M.2101 phased-array antenna pattern with independent horizontal/vertical beamwidths and configurable array dimensions (N_H x N_V).
- Added ITU-R S.672 pattern (routes through Rec 1.2 evaluation).
- Full GPU LUT support for all patterns: S.1528 Rec 1.2 and Rec 1.4 use 1-D LUT interpolation; M.2101 uses a 2-D (phi, theta) LUT with bilinear interpolation.
- Fused GPU kernels: LUT interpolation + dB-to-linear conversion in a single ElementwiseKernel, eliminating per-element dispatch overhead.

### Spectrum plan and reuse

- Full spectrum plan configuration: service band limits, reuse factor, disabled channels, anchor reuse slot, and multi-group power policy.
- Unwanted emission mask support with ITU-R presets and custom mask point entry.
- Spectral integration controls: cutoff basis, cutoff percentage, and TX reference mode.
- Per-channel cell-activity mode alongside whole-cell activation.
- Power variation modes: fixed, uniform random, and slant-range scaled EIRP.

### Mixed-precision GPU profiles

- Five selectable precision profiles controlling dtype independently for propagation, pattern evaluation, and power accumulation stages.
- Profiles range from full float64 to float64/float32/float16 mixed precision; power stage is always at least float32 to avoid FSPL underflow.
- Configurable via Expert mode dropdown or the `gpu_precision_profile` API parameter.

### Boresight avoidance

- Per-system boresight avoidance with configurable theta-1 (antenna half-width) and theta-2 (static redirection) angles.
- Scope modes for theta-2: nearest-RAS, specific cell IDs, geographic radius, and layer-based selection.
- Boresight-affected cells are tracked separately in postprocess visualisations (CCDF corridor, heatmap masks).

### Beam cap sizing analysis

- Integrated beam-cap sizing pipeline (`nbeam` module) with four policies: Simpson pooling, full reroute, no reroute, and pure reroute (exact lower-bound).
- Streaming HDF5 analysis with configurable beam-cap range, SLA targets, and early-stop controls.
- Auto-benchmarking backend selection (CPU vs GPU) for the pure-reroute solver.
- Per-system support via `group_prefix` for reading beam data from per-system HDF5 groups.

### 3D constellation viewer

- Interactive 3D viewer with real-time SGP4 propagation and playback controls (speed, FPS, scrub).
- ECEF and ECI frame modes with smooth Earth rotation and orbit-track rendering.
- Orbit tracks now account for J2 RAAN precession, matching propagated satellite positions.
- Skybox options (off, stars, 4K, 16K textures) and render modes (fast, detailed).
- Satellite picking: click a satellite to inspect its orbital parameters.
- Constellation wizard dialog for interactive belt editing with live 3D preview.
- Eight WRC-27 reference system presets for quick constellation setup.

### Postprocessing

- Eighteen postprocess recipes covering distributions (CCDF), elevation heatmaps, hemisphere maps, beam statistics, beam-cap sizing, and time series.
- Preaccumulated data path for efficient replay of distributions without re-scanning raw HDF5 data.
- Bandwidth view controls: channel-total vs reference-bandwidth scaling with stored-basis awareness (per-MHz, channel-total, RAS-receiver-band).
- Reference-line overlay with configurable protection thresholds, margin display, and custom percentile guides.
- Grid tick density control (normal, dense, sparse) with correct handling of log-scale Y axes.
- Improved grid visibility on light themes across all plot types.
- Removed legacy bandwidth warning (`missing_default`) from the postprocess info panel.

### GPU acceleration

- `GpuScepterSession`: long-lived GPU context manager with resource caching, thread-safety checks, and activation nesting.
- Atmosphere LUT context for elevation-dependent atmospheric loss lookup on the GPU.
- Fused RAS-pattern kernel replacing six sequential `cp.where` calls with a single ElementwiseKernel.
- Redundant `.astype(cp.float32)` cleanup across the power path, reducing per-batch dispatch overhead.
- Histogram optimisation: masked operations replace `nonzero` + fancy-indexing pattern.
- S.1528 pattern LUT: 180,002-element gain table at 0.001-deg resolution, replacing analytical Bessel/product evaluation.

### Grid and hexgrid

- Optional Numba JIT acceleration for grid mapping, skycell ID computation, and histogram accumulation.
- Configurable grid analysis: indicative footprint, spacing rules, leading metric, and cell-size override.
- Geography masking with shoreline buffer and coastline backend selection.

### Project schema

- Project schema version 15 with full multi-system serialisation and derived-state persistence.
- Per-system derived state (analyser/hexgrid signatures) survives project save/reload without re-running coverage analysis.

## v0.11.1

- Smoothed the main workflow across Simulation and Postprocess with better workspace resizing, a global appearance selector, and less intrusive preview-window behavior during theme changes.
- Modernized detached previews with stronger snapping, draggable markers, improved spectrum/reuse inspection, and more consistent antenna/receiver figure styling.
- Clarified spectrum, reuse, and postprocess wording around the RAS receiver band and strengthened layout/theme readability for the plot studio panels.

## v0.11

- Tightened simulation-workspace layout behavior so the maximized workflow, assistant panel, and detached preview windows behave more reliably across themes and desktop/taskbar configurations.
- Refined the RAS, spectrum, and reuse previews with improved theming, better help wiring, clearer spectral terminology, and more interactive inspection tools.
- Hardened reuse-scheme visualization and related grid/reuse UX so the schematic view, anchor-slot controls, and preview behavior are easier to interpret.
- Improved geography-mask handling, shoreline-buffer behavior, and general GUI usability/readability across the simulation workflow.

## v0.10

- Initial public GUI release of SCEPTer.
- Introduced the desktop workflow for configuring RAS studies, orbital/satellite settings, service inputs, previews, GPU runs, and postprocess inspection.
