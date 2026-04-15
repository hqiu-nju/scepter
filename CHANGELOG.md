# Changelog

All notable user-facing changes are summarized here at a high level.

## v0.25.2

### Attributions

- **Third-party notices** — ``THIRD_PARTY_NOTICES.md`` now lists
  ``vtk`` (BSD-3-Clause) explicitly as the 3D renderer underneath
  ``pyvista`` / ``pyvistaqt``, and adds a new *SGP4 propagation —
  algorithmic and implementation references* section with the
  upstream author names, full paper titles, publication venues,
  licence terms, upstream URLs (GitHub repos where applicable),
  and a suggested citation block. Crediting both the analytical
  model and the C/C++ reference implementations SCEPTer validates
  against:
    * **Hoots, F. R. & Roehrich, R. L. (December 1980).** *Models
      for Propagation of NORAD Element Sets.* Project Spacetrack
      Report No. 3, Aerospace Defense Command. Archival copy:
      <https://celestrak.org/NORAD/documentation/spacetrk.pdf>.
    * **Vallado, D. A., Crawford, P., Hujsak, R. & Kelso, T. S.
      (2006, rev. 2012).** *Revisiting Spacetrack Report #3* —
      AIAA 2006-6753. Reference code and errata:
      <https://celestrak.org/publications/AIAA/2006-6753/>. The
      SCEPTer GPU ``"vallado"`` backend is validated against this
      reference.
    * **Warner, Daniel J.** — SGP4 C++ library, Apache-2.0.
      Upstream: <https://github.com/dnwrnr/sgp4>. Wrapped by
      cysgp4 and used as the SCEPTer default propagation backend
      (``method="dwarner"``).
    * **Winkel, B.** — cysgp4 (GPL-3.0-or-later), Cython wrapper
      around Warner's library. Upstream:
      <https://github.com/bwinkel/cysgp4>.
    * **Kelso, T. S.** — CelesTrak TLE/SGP4 reference code,
      errata, and documentation archive:
      <https://celestrak.org>.
  The in-GUI *About SCEPTer* dialog (``scepter/appinfo.py``)
  mirrors the same upstream URLs and licence tags so the
  attribution is visible at runtime.

### Tests

- **UEMR defaults** — ``test_uemr_forces_hidden_defaults`` and
  ``test_uemr_set_defaults_buttons_do_not_touch_hidden_fields`` were
  asserting ``cutoff_percent == 100`` but the actual UEMR baseline
  (integration window coincides with the service-band edges) is
  encoded as ``50`` under the kernel's ``cutoff = span × percent /
  100`` half-width formula. Tests updated to match the real,
  correct semantic ("at service band edge"), with a comment
  spelling out why ``50`` is the right number.
- **New coverage**: tleforger ``start_time`` across all three forge
  helpers (``datetime``, ``astropy.Time``, legacy fallback);
  ``build_constellation_from_state`` forwarding; ``BeltTableModel``
  hidden-row decoration (font, brush, tooltip, narrow
  ``dataChanged`` emission, out-of-range pruning, default-empty
  invariant); wizard UTC-datetime round-trip regression guard;
  ``_interpolate_position_frames`` identity-of-``out`` buffer and
  clamping edge cases; orbit-track vectorized-stack build + rotate
  (GMST + drift fusion in ECEF, drift-only in ECI).

### Bug fixes

- **Constellation Wizard** — orbit-track rings no longer appear offset
  from the satellite markers. Two independent fixes were applied:
  1. The preview builder now forges TLEs with an epoch equal to
     the preview start time instead of the hard-wired
     ``2025-01-01T00:00:00 UTC`` default baked into ``tleforger``.
     When the simulated instant was months past that fixed epoch,
     SGP4's richer J2 model accumulated enough secular drift vs.
     a first-order analytical approximation of the ring to show
     a visible along-track offset.
  2. Ring RAAN and inclination are now derived directly from the
     SGP4-propagated angular momentum (r × v) of a representative
     satellite in each plane, so even if the TLE epoch is far from
     the display time (e.g., externally-supplied TLEs) the ring
     still coincides with the actual orbital plane.
- **Viewer** — per-frame ``Modified()`` calls now invalidate just
  the points ``vtkDataArray`` (``polydata.GetPoints().Modified()``)
  instead of the whole ``vtkPolyData``. The full-polydata bump
  forced VTK to re-derive bounds, normals, and cell-array
  bookkeeping every frame; the points-only bump still triggers
  the GPU coordinate re-upload that the mapper actually needs but
  reuses everything else. Applied to both the satellite-belt
  glyph polydata and the orbit-ring polylines (vectorized
  fast-path and per-ring fallback).
- **Viewer** — render-window AA / smoothing toggles
  (``SetMultiSamples(0)``, ``SetLineSmoothing(False)``,
  ``SetPolygonSmoothing(False)``, ``SetPointSmoothing(False)``)
  are now set explicitly so the wizard never silently inherits a
  per-platform default that would burn fillrate without a visual
  win at this scene scale.
- **Viewer** — per-frame allocation removed from
  ``_interpolate_position_frames``: the linear blend between two
  bracketing frames is now done fully in-place
  (``out = upper - lower; out *= frac; out += lower``) instead of
  via a transient ``frac * upper`` array. For the typical
  3 360-satellite preview that's ~40 KB / frame less GC pressure.
- **Viewer** — ``_apply_scene_frame_transforms`` now skips the
  three ``SetOrientation`` calls when the Earth-rotation angle
  hasn't changed since the previous frame (always the case in
  ECEF view, which holds the angle at 0). The cache is
  invalidated on scene rebuild and on frame-mode change.
- **Constellation Wizard** — orbit-ring rotation per playback frame
  is now vectorized: rings are packed into a contiguous
  ``(N_rings, P, 3)`` ECI stack at build time and each frame
  computes one trig pair plus one broadcast multiply for *all*
  rings instead of looping in Python and recomputing ``cos/sin``
  per ring. Per-ring VTK polydata view buffers are cached too, so
  the only per-ring work left in the playback path is a pair of
  ``np.copyto`` writes and a ``Modified()`` flag flip. The static
  z-component is now written exactly once instead of every frame.
- **Constellation Wizard** — orbit rings now stay locked to satellites
  for the entire animation, not just the initial frame. Each ring's
  RAAN is now propagated forward in time using a per-plane secular
  drift rate extracted from two SGP4 samples (start vs. end of the
  preview window) rather than being held fixed at the build-time
  RAAN. Combined with the build-time r × v initialisation this
  matches SGP4's J2 precession exactly because both the offset and
  the rate come from the same propagator.
- **Constellation Wizard** — default real-time playback target raised
  from 30 s to 120 s, so the initial 64× build covers ~2 h of
  simulated time instead of 32 min and selecting 256× / 1024× from
  the speed combo no longer triggers an immediate rebuild for a
  longer span in the typical case.
- **TLE forger** — ``forge_tle_constellation_from_belt_definitions``,
  ``forge_tle_belt``, and ``forge_tle_single`` now all accept an
  optional ``start_time`` parameter (astropy ``Time`` or Python
  ``datetime``) that threads through to the TLE epoch. Default
  handling has been unified: passing ``None`` (or omitting the
  argument) falls back to ``2025-01-01T00:00:00 UTC`` so existing
  callers are unaffected.
- **UEMR mode** — "Edit Tx Mask" dialog now anchors its in-band
  0-dB points at the **service-band edges** when UEMR is active,
  not at the (UEMR-ignored) Service-tab channel bandwidth. The
  kernel's integration window in UEMR is exactly the service band
  (``cutoff_basis = service_bandwidth`` at 50%), so the mask
  editor's half-width parameter must match — otherwise the fixed
  anchors sit at the wrong offsets and any custom attenuation
  points the user drags in end up relative to a band width the
  kernel never uses. Non-UEMR (directive) systems still get the
  Service-tab channel bandwidth, matching legacy behaviour.
- **UEMR mode** — the transmit unwanted-emission-mask preset is no
  longer forcibly reset to ``flat`` on every UEMR-gating pass.
  ``flat`` is still the auto-selected baseline when no preset has
  been chosen yet (matching the "no out-of-band suppression"
  physical intuition for isotropic circuitry leakage), but an
  explicit user choice — e.g. ``custom`` for a vendor-specific
  roll-off, or any ITU template — is now preserved across
  re-applies triggered by other control changes. "Set Spectrum
  Defaults" in UEMR mode still writes ``flat`` regardless of the
  current selection, so users always have a one-click path back
  to the UEMR baseline.
- **Constellation Wizard** — clicking the eye glyph no longer
  triggers a preview rebuild. Previously *any* ``dataChanged``
  from the belt model (including the visibility-only flip on the
  "Show" column) kicked off the debounced ``_refresh_viewer``
  path, which re-propagated all satellites with SGP4 — turning a
  ~1 ms actor visibility toggle into a second-scale rebuild and
  freezing any in-flight animation. The dataChanged handler now
  classifies the signal: if the change range is confined to the
  Show column it's treated as a render-only update and the
  rebuild is skipped, while real edits (Alt / Planes /
  Eccentricity / …) and row add/remove still rebuild as
  expected.
- **Constellation Wizard** — hidden belts are now marked by a
  dedicated "Show" column at the start of the belt table that
  renders an eye glyph per row (👁 open = visible, 👁⃠ slashed =
  hidden). Clicking the eye directly toggles that belt's
  visibility in the 3D preview — no need to select the row first
  and press a button. The "Hide belt" button still works and
  still flips between "Hide belt"/"Show belt" for the selected
  row. The main-window belt table hides the Show column since it
  has no per-row visibility toggle. Visibility also survives the
  initial "stars → Earth → satellites" build sequence: the
  wizard now pre-sizes the viewer's preview span to match its
  default 64× playback speed before the first build, collapsing
  the legacy "build → flash-rebuild" sequence into a single
  build; the hidden-set is also re-applied on every
  ``preview_build_completed`` as a belt-and-braces guard for
  speed changes after the wizard is already open.
- **Constellation Wizard** — play, reset, and speed-select controls
  now use an expanding size policy with stretch factors instead of
  fixed 32-px widths so their glyphs (▶ / ⏮) and the speed combo
  are no longer truncated on narrow panels. "Hide belt" and "Show
  all" share the remaining space with a larger stretch weight.
- **Constellation Wizard** — fixed `ValueError: End UTC must be later
  than Start UTC.` raised immediately on opening the wizard for users
  whose system timezone differed from UTC. The auto-extend path in
  `_on_wizard_speed_changed` constructed the `end_edit` `QDateTime`
  directly from a tz-aware Python `datetime`, which Qt interpreted as
  local time; reading the widget back via `toSecsSinceEpoch` then
  yielded an instant earlier than start by the local UTC offset. The
  widget is now written through the existing `_utc_datetime_to_qdatetime`
  helper so the stored instant always matches the intended UTC time.

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
