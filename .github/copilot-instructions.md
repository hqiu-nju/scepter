# Copilot Usage Guidance for SCEPTer

## Project context
SCEPTer simulates satellite constellation emissions and the resulting EPFD for radio telescopes. It relies on Python 3.10, PyCRAF, and cysgp4, and is typically exercised through notebooks and the `scepter` Python package.

## Coding preferences
- Use Python type hints in new or updated code.
- Follow standard library-first imports: standard library, third-party, then local modules.
- Prefer explicit, readable variable names; avoid single-letter names outside of short loops.
- Keep docstrings concise with a one-line summary plus argument/return descriptions when applicable.
- Favor immutable inputs where possible and avoid in-place mutation unless necessary for performance.

## Testing and notebooks
- Where feasible, add or update lightweight tests that illustrate expected behavior for new functions.
- For notebooks, keep cells focused and prefer markdown explanations alongside code so Copilot can suggest context-aware completions.

## Conversation hints
- When Copilot Chat is asked for changes, summarize the intent and list the files it plans to touch before proposing edits.
- If unsure about domain-specific terms (e.g., EPFD), prompt for clarification before generating code.
