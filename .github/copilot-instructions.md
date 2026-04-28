# Copilot Usage Guidance for SCEPTer

Use this file with `AGENTS.md`. If there is a conflict, follow `AGENTS.md`.

## Project context

SCEPTer models satellite constellation emissions and resulting EPFD impact for
radio telescopes. Most work happens in Python modules under `scepter/`, CLI scripts `scripts/` and in
analysis notebooks.

Core stack:

- Python 3.10
- astropy (units/time)
- pycraf
- cysgp4

Simulation-module highlights (for quick orientation — full details in
`AGENTS.md` and `CLAUDE.md`):

- `scepter/gpu_accel.py` — GPU pipeline (propagation, beam finalize, power).
  Includes the surface-PFD cap LUT builder
  (`prepare_peak_pfd_lut_context`) and aggregate helper
  (`_compute_aggregate_surface_pfd_cap_cp`).
- `scepter/scenario.py` — top-level `run_gpu_direct_epfd` runner, HDF5
  writer, multi-system interleaving.
- `scepter/scepter_GUI.py` — desktop GUI including Service & Demand
  surface-PFD cap controls and Review & Run cap stats reporting toggle.

## Before proposing edits

- Restate the requested change in one short sentence.
- List the files you plan to modify before generating patches.
- If units, coordinate frame, or array-axis semantics are ambiguous, ask a
  clarifying question first.

## Diff hygiene

- Prefer minimal, targeted edits over full-block rewrites.
- Do not remove and immediately re-add unchanged lines.
- Avoid unnecessary whitespace-only or line-wrap-only edits.
- Preserve line endings and existing formatting unless explicitly requested
  otherwise.
- Before finalizing, check `git diff` and clean no-op hunks.
- Exception: when the prompt requests broader methodology/architecture
  improvements, or when a narrow patch would be unsafe, larger coordinated
  changes are allowed. In that case, state intended scope and impacted files
  before generating edits.

## Code generation rules

- Keep generated code Python 3.10-compatible.
- Add type hints to new/modified public APIs.
- Keep import order: standard library, third-party, local.
- Prefer explicit names and small, testable helper functions over long blocks.
- Avoid silent unit stripping or implicit conversions of physical quantities.
- Preserve existing behavior unless the prompt explicitly requests a change.

## Documentation rules

- Update docstrings/README/notebook markdown when behavior or usage changes.
- Keep command examples copy/paste-ready.
- For simulation APIs, document units and expected dimensionality when relevant.
- For public/user-facing Python APIs, treat detailed NumPy-style docstrings as
  the minimum baseline: include summary, extended description, `Parameters`,
  `Returns`, `Raises`, and `Notes` (plus `Examples` where useful).
- Aim for pycraf-level detail or better by documenting units, expected shapes,
  defaults, formulas/conventions, assumptions, edge cases, and
  reproducibility-related behavior.

## Dependency rules

- If dependencies change, update `environment.yml` and `environment-full.yml`
  consistently.
- Keep optional GPU/visualization extras in `environment-full.yml` unless
  needed for core/lite workflows.
- If environment commands change, update the README development environment
  section in the same patch.

## Validation expectations

- Propose or run a targeted validation step for each functional change.
- For simulation pipeline changes, include a representative script/notebook run
  when feasible.
- If validation is skipped, state the reason explicitly.
