# Agent Guidelines for SCEPTer

These instructions apply to the entire repository.

- Read `.github/copilot-instructions.md` before suggesting structural or code changes so AI helpers align with project preferences.
- Keep dependency updates consistent: adjust both `environment.yml` (lite) and `environment-full.yml` (full) when packages change, and refresh the README's development environment section if commands or options shift.
- Stick to Python 3.10-compatible solutions. Place heavy GPU/visualization extras in the full environment unless they are strictly required in the lite setup.
- When editing documentation or notebooks, keep environment commands copy/paste-ready and refer to the environments as `scepter-dev` (lite) and `scepter-dev-full` (full).
- Run targeted tests or illustrative notebooks when modifying simulation pipelines, and note any intentional skips with context in the final summary.
