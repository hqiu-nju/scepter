"""Custom antenna pattern loader / validator (schema v1).

Implements the ``scepter_antenna_pattern_format="v1"`` schema — this
module's docstring + ``CustomAntennaPattern`` dataclass are the
authoritative definition of that format. This module is pure Python
+ NumPy; it does not depend on PySide6, pyvista, cupy, or any other
heavy runtime, and can be imported on the minimal ``scepter-dev``
environment.

Stage 2 of the custom-antenna-pattern rollout (see the 30-stage plan
in ``CLAUDE.md``). This stage intentionally implements *only* load /
validate / dump. Runtime evaluation (1-D linear / 2-D bilinear
interpolation, GPU upload, coordinate conversion) lives in later
stages.

Public API
----------
- ``CustomAntennaPattern`` — frozen dataclass holding the validated,
  float64 NumPy arrays and the shared metadata envelope.
- ``load_custom_pattern(path)`` — parse + validate, return a
  ``CustomAntennaPattern``. Raises ``ValueError`` with a
  field-quoting message on any schema violation. Emits
  ``PatternPeakWarning`` in the mild-mismatch / undersampled-beam
  band (see the peak-gain semantics section of the schema doc).
- ``dump_custom_pattern(path, pattern)`` — serialise a validated
  pattern back to pretty-printed JSON.

The loader performs **no** normalisation, resampling, coordinate
conversion, or wrap-around averaging — those are runtime concerns.
Every transform that happens here is declared in the schema doc's
"Loader behaviour summary" section.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Schema constants (exported — callers may ``from scepter import custom_antenna
# as ca`` and then compare strings to ``ca.KIND_1D`` etc. to avoid typos).
# ---------------------------------------------------------------------------

FORMAT_VERSION: str = "v1"

KIND_1D: str = "1d_axisymmetric"
KIND_2D: str = "2d"
_VALID_KINDS: tuple[str, ...] = (KIND_1D, KIND_2D)

GRID_MODE_AZEL: str = "az_el"
GRID_MODE_THETAPHI: str = "theta_phi"
_VALID_GRID_MODES: tuple[str, ...] = (GRID_MODE_AZEL, GRID_MODE_THETAPHI)
DEFAULT_GRID_MODE: str = GRID_MODE_AZEL

NORMALISATION_ABSOLUTE: str = "absolute"
NORMALISATION_RELATIVE: str = "relative"
_VALID_NORMALISATIONS: tuple[str, ...] = (
    NORMALISATION_ABSOLUTE,
    NORMALISATION_RELATIVE,
)

PEAK_SOURCE_EXPLICIT: str = "explicit"
PEAK_SOURCE_LUT: str = "lut"
_VALID_PEAK_SOURCES: tuple[str, ...] = (PEAK_SOURCE_EXPLICIT, PEAK_SOURCE_LUT)

# Loader thresholds. ``_PEAK_MISMATCH_REFUSE_DB`` matches the 10 dB figure
# documented in the schema doc's "Peak-gain semantics" section —
# intentionally loose so ITU regulatory masks load cleanly.
_PEAK_MISMATCH_REFUSE_DB: float = 10.0
_PEAK_MISMATCH_WARN_DB: float = 0.5
# Angular slack for boundary checks ("starts at 0°", "extends to ≥180°", ...).
# A 1 µdeg tolerance absorbs float-format noise without letting real
# mistakes slip through.
_ANGLE_TOL_DEG: float = 1.0e-6
# Epsilon for dB comparisons (e.g. the "relative-mode LUT max <= 0 dB" check).
# Numerically equal to ``_ANGLE_TOL_DEG`` but semantically distinct — dB, not
# degrees — so it gets its own name to avoid the unit confusion.
_DB_EPSILON: float = 1.0e-6


class PatternPeakWarning(UserWarning):
    """Emitted when an LUT-derived peak disagrees mildly with the stored peak.

    Mild = outside the ``_PEAK_MISMATCH_WARN_DB`` band but within the
    ``_PEAK_MISMATCH_REFUSE_DB`` band. Larger mismatches raise
    ``ValueError`` instead of warning. See the schema doc for the
    rationale (supports ITU regulatory masks and undersampled
    narrow-beam datasheets).
    """


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomAntennaPattern:
    """Validated custom antenna pattern (schema v1).

    Fields common to both ``kind`` values are always populated; the
    kind-specific fields are ``None`` for the inactive kind. This flat
    layout keeps JSON round-trip trivial and lets downstream code
    branch on ``kind`` / ``grid_mode`` with a single ``if`` ladder.
    """

    # Shared envelope
    format_version: str
    kind: str
    normalisation: str
    peak_gain_source: str
    peak_gain_dbi: float
    meta: Mapping[str, Any]

    # Gain array: 1-D ``(N,)`` for ``kind=1d_axisymmetric``,
    # 2-D ``(N_axis0, N_axis1)`` for ``kind=2d``.
    gain_db: np.ndarray

    # 1-D grid (None for 2-D patterns)
    grid_deg: np.ndarray | None = None

    # 2-D grid envelope (all None for 1-D patterns)
    grid_mode: str | None = None

    # 2-D az_el payload (None for theta_phi / 1-D)
    az_grid_deg: np.ndarray | None = None
    el_grid_deg: np.ndarray | None = None
    az_wraps: bool | None = None

    # 2-D theta_phi payload (None for az_el / 1-D)
    theta_grid_deg: np.ndarray | None = None
    phi_grid_deg: np.ndarray | None = None
    phi_wraps: bool | None = None

    @property
    def is_axisymmetric(self) -> bool:
        """True iff the pattern is 1-D (γ only) — i.e. no φ dependence.

        Stage 5 introduced this flag so downstream runtime code can
        short-circuit the 2-D lookup machinery for axisymmetric
        patterns. It's a pure function of ``kind`` but expressed as
        a property so callers don't have to memorise the kind string.
        """
        return self.kind == KIND_1D

    # -----------------------------------------------------------------
    # JSON-dict round-trip (used by the GUI project-state save/load —
    # the pattern is embedded *inline* into the project JSON, not as a
    # filesystem reference, so projects stay self-contained across
    # machines as required by the Stage 3 design brief).
    # -----------------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Return the schema-v1 JSON-dict representation of this pattern.

        Identical to what ``dump_custom_pattern`` writes to disk, just
        returned in-memory. Safe to embed inside a larger project-state
        dict because every value is a standard JSON type.
        """
        return _to_payload(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "CustomAntennaPattern":
        """Build a validated pattern from the dict form of schema v1.

        Counterpart to :meth:`to_json_dict`. Raises ``ValueError`` on
        any schema violation — the exact same rules that
        :func:`load_custom_pattern` enforces from disk.
        """
        if not isinstance(payload, dict):
            # Accept any Mapping for callers that hand us e.g. an
            # immutable proxy, but normalise to a plain dict for the
            # downstream _from_payload helpers.
            payload = dict(payload)
        return _from_payload(payload)

    def content_fingerprint(self) -> bytes:
        """Return a stable 16-byte content hash.

        Covers every field that affects the gain returned by
        :func:`evaluate_pattern_1d` / :func:`evaluate_pattern_2d` —
        grid arrays, gain array, normalisation / peak semantics, and
        (for 2-D) grid_mode + wrap flags. The ``meta`` free-text
        envelope is *excluded* (changing a title doesn't change
        radiation) as is ``format_version`` (identical across v1).

        Used by the GPU session's Custom-pattern context cache so a
        user can load a pattern, mutate its arrays in place, and
        re-prepare the context without a save-reload round-trip —
        the cache detects the content change automatically.

        The output is a raw ``bytes`` (BLAKE2b-128) so callers can
        use it in dict keys or compare for equality. 128 bits is
        comfortably collision-safe for this use (the session cache
        holds at most a few dozen patterns at a time).
        """
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        h.update(self.kind.encode("utf-8"))
        h.update(self.normalisation.encode("utf-8"))
        h.update(self.peak_gain_source.encode("utf-8"))
        h.update(np.float64(self.peak_gain_dbi).tobytes())
        h.update(np.ascontiguousarray(self.gain_db, dtype=np.float64).tobytes())
        if self.kind == KIND_1D:
            assert self.grid_deg is not None
            h.update(np.ascontiguousarray(self.grid_deg, dtype=np.float64).tobytes())
        else:
            assert self.grid_mode is not None
            h.update(self.grid_mode.encode("utf-8"))
            if self.grid_mode == GRID_MODE_AZEL:
                assert self.az_grid_deg is not None and self.el_grid_deg is not None
                h.update(np.ascontiguousarray(self.az_grid_deg, dtype=np.float64).tobytes())
                h.update(np.ascontiguousarray(self.el_grid_deg, dtype=np.float64).tobytes())
                h.update(bytes([1 if self.az_wraps else 0]))
            else:
                assert self.theta_grid_deg is not None and self.phi_grid_deg is not None
                h.update(np.ascontiguousarray(self.theta_grid_deg, dtype=np.float64).tobytes())
                h.update(np.ascontiguousarray(self.phi_grid_deg, dtype=np.float64).tobytes())
                h.update(bytes([1 if self.phi_wraps else 0]))
        return h.digest()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_custom_pattern(path: str | Path) -> CustomAntennaPattern:
    """Load and validate a v1 custom-antenna-pattern JSON file."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise ValueError(f"{path}: cannot read file ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path}: top-level JSON must be an object, got "
            f"{type(payload).__name__}"
        )
    try:
        return _from_payload(payload)
    except ValueError as exc:
        # Prepend the source path so users know which file is at fault.
        raise ValueError(f"{path}: {exc}") from None


def dump_custom_pattern(
    path: str | Path, pattern: CustomAntennaPattern
) -> None:
    """Serialise a validated pattern back to pretty-printed JSON.

    The output is deterministic (stable key order, 2-space indent,
    float-only numbers) so that ``dump → load → dump`` is byte-stable.
    """
    path = Path(path)
    payload = _to_payload(pattern)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Payload → dataclass
# ---------------------------------------------------------------------------


def _from_payload(payload: dict) -> CustomAntennaPattern:
    _check_format_version(payload)
    kind = _require_choice(payload, "kind", _VALID_KINDS)
    normalisation = _require_choice(payload, "normalisation", _VALID_NORMALISATIONS)
    peak_source = _require_choice(payload, "peak_gain_source", _VALID_PEAK_SOURCES)
    peak_gain_dbi = _require_finite(payload, "peak_gain_dbi")
    meta_raw = payload.get("meta") or {}
    if not isinstance(meta_raw, dict):
        raise ValueError(
            f"`meta` must be an object, got {type(meta_raw).__name__}"
        )
    meta = dict(meta_raw)

    if kind == KIND_1D:
        grid_deg, gain_db = _parse_1d(payload)
        _check_peak_consistency(gain_db, normalisation, peak_source, peak_gain_dbi)
        return CustomAntennaPattern(
            format_version=FORMAT_VERSION,
            kind=kind,
            normalisation=normalisation,
            peak_gain_source=peak_source,
            peak_gain_dbi=peak_gain_dbi,
            meta=meta,
            gain_db=gain_db,
            grid_deg=grid_deg,
        )

    # kind == KIND_2D
    grid_mode = _optional_choice(
        payload, "grid_mode", _VALID_GRID_MODES, default=DEFAULT_GRID_MODE
    )
    if grid_mode == GRID_MODE_AZEL:
        az, el, gain_db, az_wraps = _parse_2d_azel(payload)
        _check_peak_consistency(gain_db, normalisation, peak_source, peak_gain_dbi)
        return CustomAntennaPattern(
            format_version=FORMAT_VERSION,
            kind=kind,
            normalisation=normalisation,
            peak_gain_source=peak_source,
            peak_gain_dbi=peak_gain_dbi,
            meta=meta,
            gain_db=gain_db,
            grid_mode=grid_mode,
            az_grid_deg=az,
            el_grid_deg=el,
            az_wraps=az_wraps,
        )
    # grid_mode == GRID_MODE_THETAPHI
    theta, phi, gain_db, phi_wraps = _parse_2d_thetaphi(payload)
    _check_peak_consistency(gain_db, normalisation, peak_source, peak_gain_dbi)
    return CustomAntennaPattern(
        format_version=FORMAT_VERSION,
        kind=kind,
        normalisation=normalisation,
        peak_gain_source=peak_source,
        peak_gain_dbi=peak_gain_dbi,
        meta=meta,
        gain_db=gain_db,
        grid_mode=grid_mode,
        theta_grid_deg=theta,
        phi_grid_deg=phi,
        phi_wraps=phi_wraps,
    )


# ---------------------------------------------------------------------------
# Dataclass → payload
# ---------------------------------------------------------------------------


def _to_payload(pattern: CustomAntennaPattern) -> dict:
    base: dict[str, Any] = {
        "scepter_antenna_pattern_format": pattern.format_version,
        "kind": pattern.kind,
        "normalisation": pattern.normalisation,
        "peak_gain_source": pattern.peak_gain_source,
        "peak_gain_dbi": float(pattern.peak_gain_dbi),
        "meta": dict(pattern.meta or {}),
    }
    if pattern.kind == KIND_1D:
        assert pattern.grid_deg is not None and pattern.gain_db is not None
        base["grid_deg"] = _list_1d(pattern.grid_deg)
        base["gain_db"] = _list_1d(pattern.gain_db)
        return base

    base["grid_mode"] = pattern.grid_mode
    if pattern.grid_mode == GRID_MODE_AZEL:
        assert pattern.az_grid_deg is not None
        assert pattern.el_grid_deg is not None
        base["az_grid_deg"] = _list_1d(pattern.az_grid_deg)
        base["el_grid_deg"] = _list_1d(pattern.el_grid_deg)
        base["gain_db"] = _list_2d(pattern.gain_db)
        base["az_wraps"] = bool(pattern.az_wraps)
        return base

    assert pattern.grid_mode == GRID_MODE_THETAPHI
    assert pattern.theta_grid_deg is not None
    assert pattern.phi_grid_deg is not None
    base["theta_grid_deg"] = _list_1d(pattern.theta_grid_deg)
    base["phi_grid_deg"] = _list_1d(pattern.phi_grid_deg)
    base["gain_db"] = _list_2d(pattern.gain_db)
    base["phi_wraps"] = bool(pattern.phi_wraps)
    return base


# ---------------------------------------------------------------------------
# Shared envelope checks
# ---------------------------------------------------------------------------


def _check_format_version(payload: dict) -> None:
    fmt = payload.get("scepter_antenna_pattern_format")
    if fmt != FORMAT_VERSION:
        raise ValueError(
            "`scepter_antenna_pattern_format` must be "
            f"{FORMAT_VERSION!r}, got {fmt!r}"
        )


def _require_choice(
    payload: dict, field_name: str, valid: Sequence[str]
) -> str:
    if field_name not in payload:
        raise ValueError(f"`{field_name}` is required")
    value = payload[field_name]
    if not isinstance(value, str) or value not in valid:
        raise ValueError(
            f"`{field_name}` must be one of {list(valid)!r}, got {value!r}"
        )
    return value


def _optional_choice(
    payload: dict, field_name: str, valid: Sequence[str], *, default: str
) -> str:
    if field_name not in payload:
        return default
    value = payload[field_name]
    if not isinstance(value, str) or value not in valid:
        raise ValueError(
            f"`{field_name}` must be one of {list(valid)!r}, got {value!r}"
        )
    return value


def _require_finite(payload: dict, field_name: str) -> float:
    if field_name not in payload:
        raise ValueError(f"`{field_name}` is required")
    raw = payload[field_name]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`{field_name}` must be a finite number, got {raw!r}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"`{field_name}` must be a finite number, got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Array parsing helpers
# ---------------------------------------------------------------------------


def _require_array_1d(payload: dict, field_name: str) -> np.ndarray:
    if field_name not in payload:
        raise ValueError(f"`{field_name}` is required")
    raw = payload[field_name]
    if not isinstance(raw, list):
        raise ValueError(
            f"`{field_name}` must be a 1-D array of numbers, got "
            f"{type(raw).__name__}"
        )
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`{field_name}` contains non-numeric values"
        ) from exc
    if arr.ndim != 1:
        raise ValueError(
            f"`{field_name}` must be 1-D, got shape {arr.shape}"
        )
    if arr.size < 2:
        raise ValueError(
            f"`{field_name}` must contain at least 2 entries"
        )
    _MAX_GRID_ENTRIES = 50_000
    if arr.size > _MAX_GRID_ENTRIES:
        raise ValueError(
            f"`{field_name}` has {arr.size} entries, exceeding the "
            f"maximum allowed size of {_MAX_GRID_ENTRIES}. Reduce "
            f"grid resolution or use a coarser sampling."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`{field_name}` must contain only finite values"
        )
    return arr


def _require_array_2d_shape(
    payload: dict,
    field_name: str,
    expected_shape: tuple[int, int],
    axis_names: tuple[str, str],
) -> np.ndarray:
    if field_name not in payload:
        raise ValueError(f"`{field_name}` is required")
    raw = payload[field_name]
    if not isinstance(raw, list):
        raise ValueError(
            f"`{field_name}` must be a 2-D array (list of lists), got "
            f"{type(raw).__name__}"
        )
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`{field_name}` contains non-numeric values"
        ) from exc
    if arr.ndim != 2:
        raise ValueError(
            f"`{field_name}` must be 2-D, got shape {arr.shape}"
        )
    if arr.shape != expected_shape:
        raise ValueError(
            f"`{field_name}` has shape {arr.shape} but expected "
            f"{expected_shape} from the grids "
            f"({axis_names[0]}={expected_shape[0]}, "
            f"{axis_names[1]}={expected_shape[1]})"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`{field_name}` must contain only finite values"
        )
    return arr


def _check_non_decreasing(arr: np.ndarray, name: str) -> None:
    """Enforce ``arr[i] <= arr[i+1]`` with at most 2 equal consecutive values.

    Duplicate consecutive values are legal and express a step
    discontinuity. Three or more equal consecutive values are
    ambiguous — the schema doc forbids them explicitly.
    """
    diff = np.diff(arr)
    if np.any(diff < 0.0):
        bad_idx = int(np.argmax(diff < 0.0))
        raise ValueError(
            f"`{name}` must be non-decreasing "
            f"(value at index {bad_idx + 1} ({arr[bad_idx + 1]!r}) is less "
            f"than value at index {bad_idx} ({arr[bad_idx]!r}))"
        )
    # Scan for runs of length ≥ 3.
    run_len = 1
    for i in range(1, arr.size):
        if arr[i] == arr[i - 1]:
            run_len += 1
            if run_len >= 3:
                raise ValueError(
                    f"`{name}` has 3 or more consecutive duplicate values "
                    f"near index {i - 2}..{i} (value {arr[i]!r}); the "
                    "schema permits at most 2 consecutive duplicates "
                    "(step discontinuity) — three is ambiguous"
                )
        else:
            run_len = 1


# ---------------------------------------------------------------------------
# 1-D parsing
# ---------------------------------------------------------------------------


def _parse_1d(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    grid_deg = _require_array_1d(payload, "grid_deg")
    gain_db = _require_array_1d(payload, "gain_db")
    if grid_deg.size != gain_db.size:
        raise ValueError(
            f"`grid_deg` has size {grid_deg.size} but `gain_db` has "
            f"size {gain_db.size}; they must match"
        )
    _check_non_decreasing(grid_deg, "grid_deg")
    if abs(float(grid_deg[0])) > _ANGLE_TOL_DEG:
        raise ValueError(
            f"`grid_deg[0]` must be 0.0 (boresight), got {float(grid_deg[0])!r}"
        )
    if float(grid_deg[-1]) < 180.0 - _ANGLE_TOL_DEG:
        raise ValueError(
            f"`grid_deg[-1]` must be >= 180.0 (covers the full polar "
            f"range), got {float(grid_deg[-1])!r}"
        )
    return grid_deg, gain_db


# ---------------------------------------------------------------------------
# 2-D parsing — az/el mode
# ---------------------------------------------------------------------------


def _parse_2d_azel(
    payload: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    az = _require_array_1d(payload, "az_grid_deg")
    el = _require_array_1d(payload, "el_grid_deg")
    _check_non_decreasing(az, "az_grid_deg")
    _check_non_decreasing(el, "el_grid_deg")
    if float(az[0]) > -180.0 + _ANGLE_TOL_DEG:
        raise ValueError(
            f"`az_grid_deg[0]` must be <= -180.0, got {float(az[0])!r}"
        )
    if float(az[-1]) < 180.0 - _ANGLE_TOL_DEG:
        raise ValueError(
            f"`az_grid_deg[-1]` must be >= 180.0, got {float(az[-1])!r}"
        )
    if float(el[0]) > -90.0 + _ANGLE_TOL_DEG:
        raise ValueError(
            f"`el_grid_deg[0]` must be <= -90.0, got {float(el[0])!r}"
        )
    if float(el[-1]) < 90.0 - _ANGLE_TOL_DEG:
        raise ValueError(
            f"`el_grid_deg[-1]` must be >= 90.0, got {float(el[-1])!r}"
        )
    gain_db = _require_array_2d_shape(
        payload, "gain_db", (az.size, el.size), axis_names=("N_az", "N_el")
    )
    az_wraps_raw = payload.get("az_wraps", True)
    if not isinstance(az_wraps_raw, bool):
        raise ValueError(
            f"`az_wraps` must be a boolean, got {az_wraps_raw!r}"
        )
    return az, el, gain_db, bool(az_wraps_raw)


# ---------------------------------------------------------------------------
# 2-D parsing — theta/phi mode
# ---------------------------------------------------------------------------


def _parse_2d_thetaphi(
    payload: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    theta = _require_array_1d(payload, "theta_grid_deg")
    phi = _require_array_1d(payload, "phi_grid_deg")
    _check_non_decreasing(theta, "theta_grid_deg")
    _check_non_decreasing(phi, "phi_grid_deg")
    if abs(float(theta[0])) > _ANGLE_TOL_DEG:
        raise ValueError(
            f"`theta_grid_deg[0]` must be 0.0 (boresight), got "
            f"{float(theta[0])!r}"
        )
    if float(theta[-1]) < 180.0 - _ANGLE_TOL_DEG:
        raise ValueError(
            f"`theta_grid_deg[-1]` must be >= 180.0, got {float(theta[-1])!r}"
        )
    if float(phi[0]) > -180.0 + _ANGLE_TOL_DEG:
        raise ValueError(
            f"`phi_grid_deg[0]` must be <= -180.0, got {float(phi[0])!r}"
        )
    if float(phi[-1]) < 180.0 - _ANGLE_TOL_DEG:
        raise ValueError(
            f"`phi_grid_deg[-1]` must be >= 180.0, got {float(phi[-1])!r}"
        )
    gain_db = _require_array_2d_shape(
        payload,
        "gain_db",
        (theta.size, phi.size),
        axis_names=("N_theta", "N_phi"),
    )
    phi_wraps_raw = payload.get("phi_wraps", True)
    if not isinstance(phi_wraps_raw, bool):
        raise ValueError(
            f"`phi_wraps` must be a boolean, got {phi_wraps_raw!r}"
        )
    return theta, phi, gain_db, bool(phi_wraps_raw)


# ---------------------------------------------------------------------------
# Peak-gain consistency
# ---------------------------------------------------------------------------


def _check_peak_consistency(
    gain_db: np.ndarray,
    normalisation: str,
    peak_source: str,
    peak_gain_dbi: float,
) -> None:
    """Enforce the schema's peak-gain sanity rules.

    - ``normalisation="relative"``: `gain_db` must be <= 0 (peak at
      0 dB). No dBi-vs-LUT-max comparison is possible by definition.
    - ``normalisation="absolute"``: compare the LUT maximum (in dBi)
      against the declared ``peak_gain_dbi``. Refuse on
      ``> _PEAK_MISMATCH_REFUSE_DB`` (10 dB) mismatch regardless of
      ``peak_source``; emit ``PatternPeakWarning`` when
      ``peak_source="lut"`` and the mismatch exceeds the warn band
      (0.5 dB).
    """
    if normalisation == NORMALISATION_RELATIVE:
        lut_max_rel = float(np.max(gain_db))
        if lut_max_rel > _DB_EPSILON:
            raise ValueError(
                "`normalisation=\"relative\"` requires `gain_db` values "
                "<= 0 with the peak at 0 dB; got a LUT maximum of "
                f"{lut_max_rel:.6f} dB. Switch to "
                "`normalisation=\"absolute\"` or re-reference the "
                "array to its maximum."
            )
        return

    # normalisation == NORMALISATION_ABSOLUTE
    lut_max_dbi = float(np.max(gain_db))
    mismatch = abs(lut_max_dbi - peak_gain_dbi)
    if mismatch > _PEAK_MISMATCH_REFUSE_DB:
        raise ValueError(
            f"`peak_gain_dbi` ({peak_gain_dbi:.3f} dBi) and LUT maximum "
            f"({lut_max_dbi:.3f} dBi) differ by {mismatch:.3f} dB, which "
            f"exceeds the {_PEAK_MISMATCH_REFUSE_DB} dB sanity "
            "threshold. Check units and sign conventions."
        )
    if peak_source == PEAK_SOURCE_LUT and mismatch > _PEAK_MISMATCH_WARN_DB:
        warnings.warn(
            f"`peak_gain_source=\"lut\"` and the tabulated maximum "
            f"({lut_max_dbi:.3f} dBi) differs from the declared "
            f"`peak_gain_dbi` ({peak_gain_dbi:.3f} dBi) by "
            f"{mismatch:.3f} dB. Consider "
            "`peak_gain_source=\"explicit\"` — the LUT may be "
            "undersampling a narrow main beam or carrying a "
            "regulatory mask below the achievable peak.",
            PatternPeakWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# NumPy → JSON helpers (for dump)
# ---------------------------------------------------------------------------


def _list_1d(arr: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(arr, dtype=np.float64).ravel()]


def _list_2d(arr: np.ndarray) -> list[list[float]]:
    mat = np.asarray(arr, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(
            f"expected a 2-D gain array for dump, got shape {mat.shape}"
        )
    return [[float(v) for v in row] for row in mat]


__all__ = [
    "CustomAntennaPattern",
    "PatternPeakWarning",
    "FORMAT_VERSION",
    "KIND_1D",
    "KIND_2D",
    "GRID_MODE_AZEL",
    "GRID_MODE_THETAPHI",
    "DEFAULT_GRID_MODE",
    "NORMALISATION_ABSOLUTE",
    "NORMALISATION_RELATIVE",
    "PEAK_SOURCE_EXPLICIT",
    "PEAK_SOURCE_LUT",
    "load_custom_pattern",
    "dump_custom_pattern",
    "format_pattern_summary",
    "evaluate_pattern_1d",
    "evaluate_pattern_2d",
    "main",
]


# ---------------------------------------------------------------------------
# CPU evaluation — 1-D axisymmetric patterns (Stage 5).
#
# This is the reference interpolation path. Later stages add a 2-D
# bilinear evaluator (Stage 6) and GPU counterparts (Stages 11-12); they
# must agree with the output of :func:`evaluate_pattern_1d` to within a
# tight residual budget on the shared test fixtures in Phase 3.
# ---------------------------------------------------------------------------


def evaluate_pattern_1d(
    pattern: "CustomAntennaPattern",
    gamma_deg: "np.ndarray | float",
) -> np.ndarray:
    """Evaluate a 1-D axisymmetric custom pattern at off-boresight angle(s).

    Parameters
    ----------
    pattern :
        A pattern with ``kind == KIND_1D``. ``ValueError`` is raised on
        any other kind so the caller can't accidentally drive a 2-D
        pattern through the 1-D path.
    gamma_deg :
        Query angle(s) in degrees. Scalar or ``ndarray``. The schema's
        1-D grid spans ``[0°, 180°]``; values outside that range are
        clipped to the boundary gain (no extrapolation). Callers that
        genuinely want axisymmetric extension for negative inputs
        should pass ``abs(gamma)`` themselves — we don't second-guess
        the query convention here.

    Returns
    -------
    numpy.ndarray
        Gain in **absolute dBi**. For ``normalisation="relative"``
        patterns the stored relative values are shifted by
        ``peak_gain_dbi`` before return so every consumer sees the
        same basis regardless of how the file was authored. Output
        shape matches ``gamma_deg`` (scalar-in → 0-d array-out).

    Notes
    -----
    Interpolation is piecewise-linear in **dB** vs angle between
    consecutive distinct grid samples. Duplicate grid angles express
    step discontinuities; evaluation at a duplicated angle returns the
    **right-hand** value (right-continuous) — the convention fixed by
    the schema doc. Zero-width segments (duplicate grid points with
    zero span) are handled explicitly to avoid a divide-by-zero.

    This function is pure NumPy and allocation-light — suitable for
    repeated per-frame calls from the runtime. Phase 4 reuses the same
    algorithm on the GPU.
    """
    if pattern.kind != KIND_1D:
        raise ValueError(
            f"evaluate_pattern_1d requires kind={KIND_1D!r}, got {pattern.kind!r}"
        )
    assert pattern.grid_deg is not None, "validated 1-D pattern must carry grid_deg"
    gamma = np.asarray(gamma_deg, dtype=np.float64)
    grid = np.asarray(pattern.grid_deg, dtype=np.float64)
    gain = np.asarray(pattern.gain_db, dtype=np.float64)
    gamma_clipped = np.clip(gamma, float(grid[0]), float(grid[-1]))
    result = _interp_1d_with_steps(grid, gain, gamma_clipped)
    if pattern.normalisation == NORMALISATION_RELATIVE:
        # Relative LUT stores (gain_dBi - peak_gain_dBi); shift back to
        # absolute dBi so the caller doesn't need to know which
        # normalisation the file used.
        result = result + float(pattern.peak_gain_dbi)
    return result


def _interp_1d_with_steps(
    grid: np.ndarray, gain: np.ndarray, gamma: np.ndarray,
) -> np.ndarray:
    """Piecewise-linear interpolation with right-continuous step handling.

    The trick: ``np.searchsorted(grid, gamma, side="right")`` returns
    the insertion index **after** any run of equal grid values. A
    query at a duplicated grid angle therefore lands in the segment
    to the **right** of the jump — exactly the schema's
    right-continuous rule, without any special-case branching.

    Zero-width segments (e.g. the segment between the two duplicated
    entries that encode the jump itself) would give a 0/0 in the
    linear-interp formula; we detect them with ``dx == 0`` and return
    the right-hand gain directly.
    """
    idx_right = np.searchsorted(grid, gamma, side="right")
    # The segment of interest is [grid[idx_upper - 1], grid[idx_upper]].
    # Clamp to the valid range [1, N-1] so queries at the ends still
    # pick a real segment.
    idx_upper = np.clip(idx_right, 1, grid.size - 1)
    idx_lower = idx_upper - 1
    x0 = grid[idx_lower]
    x1 = grid[idx_upper]
    y0 = gain[idx_lower]
    y1 = gain[idx_upper]
    dx = x1 - x0
    zero_width = dx == 0.0
    # np.where short-circuits the division on zero-width segments —
    # we still compute (gamma - x0) / dx in the array, but its
    # (nan / inf) result is masked by the where().
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(zero_width, 0.0, (gamma - x0) / dx)
    return np.where(zero_width, y1, y0 + (y1 - y0) * frac)


# ---------------------------------------------------------------------------
# CPU evaluation — 2-D patterns (Stage 6).
#
# Bilinear interpolation on the tabulated (axis0, axis1) grid with
# right-continuous step handling on either axis. Wrap behaviour on the
# φ / az axis is controlled by the pattern's ``phi_wraps`` / ``az_wraps``
# flag; queries inside the tabulated range are always used as-is
# (including at the ±180° endpoints themselves), queries outside wrap
# modulo the grid period (when wrapping is enabled) or clip to the
# nearest boundary (when not).
# ---------------------------------------------------------------------------


def evaluate_pattern_2d(
    pattern: "CustomAntennaPattern",
    axis0_deg: "np.ndarray | float",
    axis1_deg: "np.ndarray | float",
) -> np.ndarray:
    """Evaluate a 2-D custom pattern at given ``(axis0, axis1)`` query points.

    The meaning of the two query axes follows the pattern's
    ``grid_mode``:

    - ``grid_mode="az_el"`` → ``axis0_deg = az`` (body-frame azimuth,
      ``[-180°, 180°]``), ``axis1_deg = el`` (body-frame elevation,
      ``[-90°, 90°]``).
    - ``grid_mode="theta_phi"`` → ``axis0_deg = theta`` (off-boresight,
      ``[0°, 180°]``), ``axis1_deg = phi`` (rotation around boresight,
      ``[-180°, 180°]``).

    Returns **absolute dBi** regardless of the pattern's stored
    normalisation.  Inputs broadcast together using the standard
    NumPy rules (scalar + array → array; two arrays of compatible
    shape → broadcasted array).

    Raises ``ValueError`` for 1-D patterns — route those through
    :func:`evaluate_pattern_1d` instead.
    """
    if pattern.kind != KIND_2D:
        raise ValueError(
            f"evaluate_pattern_2d requires kind={KIND_2D!r}, got {pattern.kind!r}"
        )
    axis0_query = np.asarray(axis0_deg, dtype=np.float64)
    axis1_query = np.asarray(axis1_deg, dtype=np.float64)
    axis0_query, axis1_query = np.broadcast_arrays(axis0_query, axis1_query)

    axis0_grid, axis1_grid, gain_db, axis0_wraps, axis1_wraps = _resolve_2d_grids(pattern)
    axis0_prepared = _wrap_or_clip(axis0_query, axis0_grid, wraps=axis0_wraps)
    axis1_prepared = _wrap_or_clip(axis1_query, axis1_grid, wraps=axis1_wraps)

    result = _bilinear_with_steps(
        axis0_grid, axis1_grid, gain_db, axis0_prepared, axis1_prepared,
    )
    if pattern.normalisation == NORMALISATION_RELATIVE:
        result = result + float(pattern.peak_gain_dbi)
    return result


def _resolve_2d_grids(
    pattern: "CustomAntennaPattern",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Return ``(axis0_grid, axis1_grid, gain_db, axis0_wraps, axis1_wraps)``.

    For ``grid_mode="az_el"`` only the az axis wraps; for
    ``grid_mode="theta_phi"`` only the phi axis wraps. The
    non-wrapping axis (el or theta) always clips at its boundaries.
    """
    if pattern.grid_mode == GRID_MODE_AZEL:
        assert pattern.az_grid_deg is not None
        assert pattern.el_grid_deg is not None
        return (
            np.asarray(pattern.az_grid_deg, dtype=np.float64),
            np.asarray(pattern.el_grid_deg, dtype=np.float64),
            np.asarray(pattern.gain_db, dtype=np.float64),
            bool(pattern.az_wraps),
            False,
        )
    # grid_mode == GRID_MODE_THETAPHI
    assert pattern.theta_grid_deg is not None
    assert pattern.phi_grid_deg is not None
    return (
        np.asarray(pattern.theta_grid_deg, dtype=np.float64),
        np.asarray(pattern.phi_grid_deg, dtype=np.float64),
        np.asarray(pattern.gain_db, dtype=np.float64),
        False,
        bool(pattern.phi_wraps),
    )


def _wrap_or_clip(
    values: np.ndarray, grid: np.ndarray, *, wraps: bool,
) -> np.ndarray:
    """Bring ``values`` into ``[grid[0], grid[-1]]``.

    Inside that range every query is used as-is — including queries
    exactly at the endpoints (at ``+180°`` with a ``[-180°, 180°]``
    wrap-enabled grid we return ``gain[-1]``, not ``gain[0]``; the
    user explicitly tabulated both). Only queries *outside* the
    tabulated range are wrapped or clipped.
    """
    lo = float(grid[0])
    hi = float(grid[-1])
    if wraps:
        period = hi - lo
        if period <= 0.0:
            return np.clip(values, lo, hi)
        outside = (values < lo) | (values > hi)
        wrapped = lo + np.mod(values - lo, period)
        return np.where(outside, wrapped, values)
    return np.clip(values, lo, hi)


def _bilinear_with_steps(
    axis0_grid: np.ndarray,
    axis1_grid: np.ndarray,
    gain_db: np.ndarray,
    axis0_values: np.ndarray,
    axis1_values: np.ndarray,
) -> np.ndarray:
    """2-D bilinear interpolation with right-continuous step handling.

    Replicates the 1-D algorithm along each axis independently:
    ``searchsorted(side="right")`` locates a segment just right of
    any duplicated angle, so queries at a step-jump angle land in the
    post-jump cell. Zero-width cells (which can appear when a step
    jump is encoded at the trailing grid boundary) fall back to the
    right-side corner value in that axis — matching the 1-D fallback.
    """
    # Segment lookup on each axis (identical to the 1-D case).
    i_upper = np.clip(
        np.searchsorted(axis0_grid, axis0_values, side="right"),
        1,
        axis0_grid.size - 1,
    )
    j_upper = np.clip(
        np.searchsorted(axis1_grid, axis1_values, side="right"),
        1,
        axis1_grid.size - 1,
    )
    i_lower = i_upper - 1
    j_lower = j_upper - 1

    x0 = axis0_grid[i_lower]
    x1 = axis0_grid[i_upper]
    y0 = axis1_grid[j_lower]
    y1 = axis1_grid[j_upper]
    dx = x1 - x0
    dy = y1 - y0

    # Step-jump at the trailing boundary: a zero-width cell in that
    # axis. Fall back to the right-side corner by forcing the
    # interpolation fraction to 1 (not 0 — we want the post-jump row).
    zero_x = dx == 0.0
    zero_y = dy == 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        fx = np.where(zero_x, 1.0, (axis0_values - x0) / dx)
        fy = np.where(zero_y, 1.0, (axis1_values - y0) / dy)

    g00 = gain_db[i_lower, j_lower]
    g01 = gain_db[i_lower, j_upper]
    g10 = gain_db[i_upper, j_lower]
    g11 = gain_db[i_upper, j_upper]

    # Standard bilinear interpolation formula.
    one_minus_fx = 1.0 - fx
    one_minus_fy = 1.0 - fy
    return (
        one_minus_fx * one_minus_fy * g00
        + one_minus_fx * fy * g01
        + fx * one_minus_fy * g10
        + fx * fy * g11
    )


# ---------------------------------------------------------------------------
# CLI smoke tool
# ---------------------------------------------------------------------------
#
# Invoked via ``python -m scepter.custom_antenna inspect <file>``. The main
# consumer is future-stage debugging: users who hit a runtime surprise
# can sanity-check the LUT file without opening the GUI. Output is
# plain text on stdout; load errors and CLI errors go to stderr with a
# non-zero exit code so the tool composes cleanly with shell
# pipelines.


def format_pattern_summary(
    pattern: "CustomAntennaPattern", *, path: str | Path | None = None
) -> str:
    """Render a human-readable summary of a validated pattern.

    Kept as a pure function (no I/O) so tests can assert on the full
    rendered string without spawning a subprocess. The CLI entry
    point below just prints the return value.
    """
    lines: list[str] = []
    header = "SCEPTer Custom Antenna Pattern"
    if path is not None:
        header = f"{header}  —  {Path(path).name}"
    lines.append(header)
    lines.append("─" * max(len(header), 46))

    def row(label: str, value: str) -> None:
        lines.append(f"{label:<20}: {value}")

    row("Format version", pattern.format_version)
    if pattern.kind == KIND_1D:
        row("Kind", f"{pattern.kind} (axisymmetric)")
    else:
        row("Kind", f"{pattern.kind} (grid_mode = {pattern.grid_mode})")
    row("Normalisation", pattern.normalisation)
    row("Peak gain source", pattern.peak_gain_source)
    row("Peak gain", f"{pattern.peak_gain_dbi:.3f} dBi (declared)")
    lines.append("")

    # Grid + gain description
    lines.append("Grid")
    _append_grid_summary(lines, pattern)
    lines.append("")

    lines.append("Gain")
    _append_gain_summary(lines, pattern)
    lines.append("")

    # Sanity observations — step discontinuities, wrap consistency.
    observations = _collect_sanity_observations(pattern)
    lines.append("Sanity observations")
    if observations:
        for item in observations:
            lines.append(f"  • {item}")
    else:
        lines.append("  • (none — pattern is clean)")
    lines.append("")

    meta = dict(pattern.meta or {})
    if meta:
        lines.append("Meta")
        for key in sorted(meta.keys()):
            lines.append(f"  {key:<18}: {meta[key]}")

    return "\n".join(lines).rstrip() + "\n"


def _append_grid_summary(lines: list[str], pattern: "CustomAntennaPattern") -> None:
    def describe_axis(name: str, arr: np.ndarray, unit: str = "°") -> str:
        step_desc = _describe_step(arr)
        return (
            f"  {name:<18}: {arr.size} points  "
            f"[{float(arr[0]):+.3f}, {float(arr[-1]):+.3f}]{unit}  "
            f"{step_desc}"
        )

    if pattern.kind == KIND_1D:
        assert pattern.grid_deg is not None
        lines.append(describe_axis("grid_deg", pattern.grid_deg))
        return
    if pattern.grid_mode == GRID_MODE_AZEL:
        assert pattern.az_grid_deg is not None
        assert pattern.el_grid_deg is not None
        lines.append(describe_axis("az_grid_deg", pattern.az_grid_deg))
        lines.append(describe_axis("el_grid_deg", pattern.el_grid_deg))
        lines.append(f"  {'az_wraps':<18}: {pattern.az_wraps}")
    else:
        assert pattern.theta_grid_deg is not None
        assert pattern.phi_grid_deg is not None
        lines.append(describe_axis("theta_grid_deg", pattern.theta_grid_deg))
        lines.append(describe_axis("phi_grid_deg", pattern.phi_grid_deg))
        lines.append(f"  {'phi_wraps':<18}: {pattern.phi_wraps}")


def _describe_step(arr: np.ndarray) -> str:
    """Classify a grid axis as uniform / near-uniform / varying."""
    if arr.size < 2:
        return "Δ = n/a"
    deltas = np.diff(arr)
    # Drop zero-width segments (step discontinuities) from the uniformity
    # test so a mask with a couple of jumps doesn't look "varying".
    live = deltas[deltas > 0.0]
    if live.size == 0:
        return "Δ = (all-zero grid?)"
    dmin = float(live.min())
    dmax = float(live.max())
    if np.isclose(dmin, dmax, rtol=1e-6, atol=1e-9):
        return f"Δ = {dmin:.3f}° uniform"
    span = float(arr[-1] - arr[0])
    return (
        f"Δ ∈ [{dmin:.3f}°, {dmax:.3f}°] varying "
        f"(mean ≈ {span / (arr.size - 1):.3f}°)"
    )


def _append_gain_summary(lines: list[str], pattern: "CustomAntennaPattern") -> None:
    gain = np.asarray(pattern.gain_db, dtype=np.float64)
    lut_min = float(gain.min())
    lut_max = float(gain.max())
    unit = "dBi" if pattern.normalisation == NORMALISATION_ABSOLUTE else "dB"
    lines.append(f"  {'Tabulated max':<18}: {lut_max:.3f} {unit}")
    lines.append(f"  {'Tabulated min':<18}: {lut_min:.3f} {unit}")
    lines.append(f"  {'Dynamic range':<18}: {lut_max - lut_min:.3f} dB")
    if pattern.normalisation == NORMALISATION_ABSOLUTE:
        mismatch = abs(lut_max - pattern.peak_gain_dbi)
        if mismatch <= _PEAK_MISMATCH_WARN_DB:
            verdict = "within ±0.5 dB"
        elif mismatch <= _PEAK_MISMATCH_REFUSE_DB:
            verdict = (
                f"mild — declared peak sits {pattern.peak_gain_dbi - lut_max:+.3f} dB "
                "relative to tabulated max"
            )
        else:
            verdict = "LARGE — exceeds 10 dB; loader would have refused"
        lines.append(
            f"  {'Peak vs LUT max':<18}: {mismatch:.3f} dB ({verdict})"
        )


def _collect_sanity_observations(pattern: "CustomAntennaPattern") -> list[str]:
    """Generate human-readable observations about the pattern's quirks.

    These complement (not replace) the hard validation rules in the
    loader: the loader has already accepted the file, so every item
    here is informational. Users reading this output are trying to
    understand *why* a particular pattern behaves the way it does.
    """
    observations: list[str] = []

    for axis_name, axis in _iter_grid_axes(pattern):
        duplicates = _find_duplicate_angles(axis)
        if duplicates:
            joined = ", ".join(f"{v:.3f}°" for v in duplicates)
            observations.append(
                f"Step discontinuity on {axis_name} at {joined} "
                "(two points at the same angle with different gains)."
            )

    # az_wraps / phi_wraps boundary consistency: if wraps=True and the
    # tabulated first / last values differ along the wrapped axis, the
    # runtime averages them. Report the mismatch so users know about
    # it before running.
    if pattern.kind == KIND_2D:
        if pattern.grid_mode == GRID_MODE_AZEL and bool(pattern.az_wraps):
            obs = _describe_wrap_boundary(
                pattern.gain_db,
                pattern.az_grid_deg,  # type: ignore[arg-type]
                axis=0,
                axis_name="az",
            )
            if obs is not None:
                observations.append(obs)
        if pattern.grid_mode == GRID_MODE_THETAPHI and bool(pattern.phi_wraps):
            obs = _describe_wrap_boundary(
                pattern.gain_db,
                pattern.phi_grid_deg,  # type: ignore[arg-type]
                axis=1,
                axis_name="phi",
            )
            if obs is not None:
                observations.append(obs)

    if pattern.normalisation == NORMALISATION_RELATIVE:
        max_rel = float(np.asarray(pattern.gain_db).max())
        if abs(max_rel) > _ANGLE_TOL_DEG:
            observations.append(
                f"Relative-normalised LUT maximum is {max_rel:+.6f} dB, "
                "not 0.0 (would have been refused at load)."
            )

    return observations


def _iter_grid_axes(
    pattern: "CustomAntennaPattern",
):
    """Yield ``(axis_name, ndarray)`` pairs for every present grid axis."""
    if pattern.kind == KIND_1D:
        yield "grid_deg", pattern.grid_deg  # type: ignore[misc]
        return
    if pattern.grid_mode == GRID_MODE_AZEL:
        yield "az_grid_deg", pattern.az_grid_deg  # type: ignore[misc]
        yield "el_grid_deg", pattern.el_grid_deg  # type: ignore[misc]
    else:
        yield "theta_grid_deg", pattern.theta_grid_deg  # type: ignore[misc]
        yield "phi_grid_deg", pattern.phi_grid_deg  # type: ignore[misc]


def _find_duplicate_angles(axis: np.ndarray) -> list[float]:
    """Return the list of angles that appear as consecutive duplicates.

    A v1 pattern allows at most 2 consecutive duplicates per angle
    (the loader already enforces the 3+ refusal). We report each
    duplicated angle once.
    """
    angles: list[float] = []
    for i in range(1, axis.size):
        if axis[i] == axis[i - 1]:
            angles.append(float(axis[i]))
    return angles


def _describe_wrap_boundary(
    gain_db: np.ndarray,
    axis_values: np.ndarray,
    *,
    axis: int,
    axis_name: str,
) -> str | None:
    """Return a human note if the wrap-axis endpoints disagree.

    For a ``*_wraps=True`` axis the ±180° boundary values ought to be
    equal (they represent the same direction). Small differences are
    common in real data; we flag anything above 0.5 dB so users know
    the runtime will have to average them.
    """
    if float(axis_values[0]) > -180.0 + _ANGLE_TOL_DEG:
        # Grid doesn't reach -180; not a wrap-boundary situation worth
        # flagging.
        return None
    if float(axis_values[-1]) < 180.0 - _ANGLE_TOL_DEG:
        return None
    slice_first = np.take(gain_db, 0, axis=axis)
    slice_last = np.take(gain_db, -1, axis=axis)
    diff = np.abs(np.asarray(slice_first) - np.asarray(slice_last))
    max_diff = float(diff.max())
    if max_diff <= 0.5:
        return None
    return (
        f"{axis_name}_wraps=True but the ±180° boundary values differ by up to "
        f"{max_diff:.3f} dB; the runtime will average the two sides."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a POSIX-style exit code."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m scepter.custom_antenna",
        description=(
            "Inspect SCEPTer custom antenna pattern (LUT) JSON files. "
            "Prints a human-readable summary with grid extents, "
            "gain range, and sanity observations."
        ),
    )
    sub = parser.add_subparsers(dest="command")
    inspect = sub.add_parser(
        "inspect", help="Print a summary of a schema-v1 pattern file."
    )
    inspect.add_argument(
        "path",
        type=str,
        help="Path to the pattern JSON file.",
    )
    args = parser.parse_args(argv)

    if args.command != "inspect" or args.command is None:
        parser.print_help()
        return 1

    try:
        pattern = load_custom_pattern(args.path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_pattern_summary(pattern, path=args.path))
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess in tests
    import sys

    sys.exit(main())
