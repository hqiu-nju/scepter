#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
point_source_visibility_simulator.py - Far-field RA/Dec point-source visibilities.

Simulate complex interferometric visibilities for fixed celestial point sources
specified in ICRS right ascension and declination coordinates.  The simulator
tracks a user-supplied phase centre, computes UVW coordinates with
``scepter.uvw.compute_uvw``, and writes a SCEPTer-compatible ``.npz`` archive.

The output is directly consumable by quick-look scripts such as
``scripts/quick_dirty_image_from_uvw.py`` through the stable ``uvw``, ``vis``,
and ``freq_hz`` keys.

Source catalog
--------------
The source file may be comma-separated, semicolon-separated, tab-separated, or
whitespace-delimited.  It may include a header with columns equivalent to:

``name, ra_deg, dec_deg, flux``

Headerless files are interpreted as the same four columns.  Flux is treated as
an arbitrary linear amplitude, commonly Jy.

Example
-------
    conda activate scepter-dev
    python scripts/point_source_visibility_simulator.py

Notes
-----
The visibility convention is

``V = sum(flux * exp(phase_sign * 2*pi*i * Δw / wavelength))``

where ``Δw`` is the difference between each source's geometric path projection
and the tracked phase-centre path projection for the same physical baseline.
The default ``phase_sign=-1`` matches the convention used elsewhere in the
SCEPTer UVW helpers.  The model is geometric and far-field only: it does not
include primary-beam attenuation, noise, calibration terms, or spectral index.
By default, the first source in the combined catalog is used as the phase
centre.  Pass ``--phase-ra-deg`` and ``--phase-dec-deg`` together to override
that default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.time import Time

from scepter import uvw


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARRAY_FILE = SCRIPT_DIR / "example_telescope_array.csv"
DEFAULT_SOURCE_FILE = SCRIPT_DIR / "mock_point_sources.csv"
DEFAULT_START_TIME = "2025-01-01T00:00:00"
DEFAULT_N_TIMES = 60
DEFAULT_CADENCE_SEC = 10.0
DEFAULT_FREQ_MHZ = 1420.0
DEFAULT_OUTPUT = Path("point_source_vis.npz")


@dataclass(frozen=True, slots=True)
class PointSourceCatalog:
    """
    Parsed celestial point-source catalog.

    Parameters
    ----------
    names : tuple of str
        Source identifiers in catalog order.
    ra_deg : numpy.ndarray, shape (N_src,)
        ICRS right ascensions in degrees. Values are wrapped into
        ``[0, 360)``.
    dec_deg : numpy.ndarray, shape (N_src,)
        ICRS declinations in degrees.
    flux : numpy.ndarray, shape (N_src,)
        Linear source amplitudes. Values are not unit-converted by this script.

    Notes
    -----
    The simulator treats *flux* as the scalar source brightness multiplying the
    complex geometric fringe term.  Use Jy, normalised amplitudes, or any other
    consistent linear unit according to the downstream analysis.
    """

    names: tuple[str, ...]
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    flux: np.ndarray


@dataclass(frozen=True, slots=True)
class VisibilitySimulationResult:
    """
    In-memory result for a celestial point-source visibility simulation.

    Parameters
    ----------
    vis : numpy.ndarray, shape (N_bl, T)
        Summed complex visibilities over all catalog sources.
    vis_per_source : numpy.ndarray, shape (N_bl, T, N_src)
        Per-source complex visibility contribution before summation.
    uvw_m : numpy.ndarray, shape (N_bl, T, 3)
        Baseline UVW coordinates for the tracked phase centre in metres.
    source_uvw_m : numpy.ndarray, shape (N_bl, T, N_src, 3)
        Baseline UVW coordinates for each source direction in metres.
    phase_rad : numpy.ndarray, shape (N_bl, T, N_src)
        Unwrapped phase used for each per-source contribution.
    normalised_amplitude : numpy.ndarray, shape (N_bl, T)
        ``abs(vis)`` normalised to the maximum finite summed amplitude.
    baseline_pairs : tuple of tuple of int
        Antenna-index pairs aligned with the baseline axis.
    """

    vis: np.ndarray
    vis_per_source: np.ndarray
    uvw_m: np.ndarray
    source_uvw_m: np.ndarray
    phase_rad: np.ndarray
    normalised_amplitude: np.ndarray
    baseline_pairs: tuple[tuple[int, int], ...]


_SOURCE_NAME_FIELDS = frozenset({"name", "source", "source_name", "id"})
_SOURCE_RA_FIELDS = frozenset(
    {"ra", "ra_deg", "right_ascension", "right_ascension_deg"}
)
_SOURCE_DEC_FIELDS = frozenset(
    {"dec", "dec_deg", "declination", "declination_deg"}
)
_SOURCE_FLUX_FIELDS = frozenset({"flux", "flux_jy", "amplitude", "amp", "i"})


def _normalise_header_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _token_is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _detect_delimiter(line: str) -> str | None:
    if "," in line:
        return ","
    if ";" in line:
        return ";"
    if "\t" in line:
        return "\t"
    return None


def _split_fields(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return re.split(r"\s+", line.strip())
    return [field.strip() for field in line.split(delimiter)]


def _resolve_column(
    header_map: dict[str, int],
    accepted_names: frozenset[str],
    label: str,
) -> int:
    for name in accepted_names:
        if name in header_map:
            return int(header_map[name])
    expected = ", ".join(sorted(accepted_names))
    raise ValueError(
        f"Source catalog is missing a {label} column. Expected one of: {expected}."
    )


def load_point_source_catalog(source_file: str | Path) -> PointSourceCatalog:
    """
    Load a celestial point-source catalog from text.

    Parameters
    ----------
    source_file : str or pathlib.Path
        Text catalog with rows containing source name, RA in degrees, Dec in
        degrees, and linear flux/amplitude.  Supported delimiters are comma,
        semicolon, tab, or whitespace.  Blank lines and text after ``#`` are
        ignored.

    Returns
    -------
    PointSourceCatalog
        Parsed source names and numeric source columns.

    Raises
    ------
    FileNotFoundError
        If *source_file* does not exist.
    ValueError
        If the file is empty, malformed, has no valid sources, or contains
        invalid RA/Dec/flux values.

    Notes
    -----
    Headered files may use common aliases such as ``source_name``, ``ra``,
    ``ra_deg``, ``dec``, ``dec_deg``, ``flux``, ``flux_jy``, or ``amplitude``.
    Headerless files must use ``name ra_deg dec_deg flux`` ordering.
    """
    path = Path(source_file)
    if not path.is_file():
        raise FileNotFoundError(f"Point-source catalog not found: {path}")

    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)

    if not lines:
        raise ValueError(f"Point-source catalog '{path}' is empty.")

    delimiter = _detect_delimiter(lines[0])
    first_fields = _split_fields(lines[0], delimiter)
    header_present = len(first_fields) < 4 or not all(
        _token_is_float(token) for token in first_fields[1:4]
    )

    if header_present:
        header_map = {
            _normalise_header_name(field): idx for idx, field in enumerate(first_fields)
        }
        name_idx = next(
            (header_map[field] for field in _SOURCE_NAME_FIELDS if field in header_map),
            None,
        )
        ra_idx = _resolve_column(header_map, _SOURCE_RA_FIELDS, "right ascension")
        dec_idx = _resolve_column(header_map, _SOURCE_DEC_FIELDS, "declination")
        flux_idx = _resolve_column(header_map, _SOURCE_FLUX_FIELDS, "flux")
        data_lines = lines[1:]
    else:
        name_idx, ra_idx, dec_idx, flux_idx = 0, 1, 2, 3
        data_lines = lines

    if not data_lines:
        raise ValueError(f"Point-source catalog '{path}' has no data rows.")

    required_columns = max(ra_idx, dec_idx, flux_idx) + 1
    if name_idx is not None:
        required_columns = max(required_columns, name_idx + 1)

    names: list[str] = []
    ra_values: list[float] = []
    dec_values: list[float] = []
    flux_values: list[float] = []

    for row_idx, line in enumerate(data_lines, start=1):
        fields = _split_fields(line, delimiter)
        if len(fields) < required_columns:
            raise ValueError(
                f"Row {row_idx} in '{path}' has {len(fields)} column(s), "
                f"but at least {required_columns} are required."
            )

        name = fields[name_idx] if name_idx is not None else f"src{row_idx - 1}"
        if not name:
            raise ValueError(f"Source name is empty at row {row_idx} in '{path}'.")

        ra = float(fields[ra_idx]) % 360.0
        dec = float(fields[dec_idx])
        flux = float(fields[flux_idx])
        if not np.isfinite(ra):
            raise ValueError(f"RA is not finite for source '{name}' in '{path}'.")
        if not np.isfinite(dec) or dec < -90.0 or dec > 90.0:
            raise ValueError(
                f"Dec must be finite and within [-90, 90] deg for source '{name}'."
            )
        if not np.isfinite(flux):
            raise ValueError(f"Flux is not finite for source '{name}' in '{path}'.")

        names.append(name)
        ra_values.append(ra)
        dec_values.append(dec)
        flux_values.append(flux)

    return PointSourceCatalog(
        names=tuple(names),
        ra_deg=np.asarray(ra_values, dtype=np.float64),
        dec_deg=np.asarray(dec_values, dtype=np.float64),
        flux=np.asarray(flux_values, dtype=np.float64),
    )


def parse_inline_source(source_spec: str) -> PointSourceCatalog:
    """
    Parse one inline source specification.

    Parameters
    ----------
    source_spec : str
        Source description in ``name,ra_deg,dec_deg,flux`` form.

    Returns
    -------
    PointSourceCatalog
        Single-source catalog.

    Raises
    ------
    ValueError
        If the specification is malformed or contains invalid numeric values.
    """
    fields = [field.strip() for field in source_spec.split(",")]
    if len(fields) != 4:
        raise ValueError(
            "Inline sources must use 'name,ra_deg,dec_deg,flux'. "
            f"Got {source_spec!r}."
        )
    pathless_catalog = PointSourceCatalog(
        names=(fields[0],),
        ra_deg=np.asarray([float(fields[1]) % 360.0], dtype=np.float64),
        dec_deg=np.asarray([float(fields[2])], dtype=np.float64),
        flux=np.asarray([float(fields[3])], dtype=np.float64),
    )
    if not pathless_catalog.names[0]:
        raise ValueError("Inline source name must not be empty.")
    if not np.isfinite(pathless_catalog.ra_deg[0]):
        raise ValueError("Inline source RA must be finite.")
    if (
        not np.isfinite(pathless_catalog.dec_deg[0])
        or not -90.0 <= pathless_catalog.dec_deg[0] <= 90.0
    ):
        raise ValueError("Inline source Dec must be finite and within [-90, 90] deg.")
    if not np.isfinite(pathless_catalog.flux[0]):
        raise ValueError("Inline source flux must be finite.")
    return pathless_catalog


def combine_catalogs(catalogs: Sequence[PointSourceCatalog]) -> PointSourceCatalog:
    """
    Concatenate one or more source catalogs.

    Parameters
    ----------
    catalogs : sequence of PointSourceCatalog
        Catalogs to concatenate in order.

    Returns
    -------
    PointSourceCatalog
        Combined catalog.

    Raises
    ------
    ValueError
        If no catalogs or no sources are supplied.
    """
    if len(catalogs) == 0:
        raise ValueError("At least one source catalog is required.")

    names: list[str] = []
    ra_parts: list[np.ndarray] = []
    dec_parts: list[np.ndarray] = []
    flux_parts: list[np.ndarray] = []
    for catalog in catalogs:
        names.extend(catalog.names)
        ra_parts.append(np.asarray(catalog.ra_deg, dtype=np.float64))
        dec_parts.append(np.asarray(catalog.dec_deg, dtype=np.float64))
        flux_parts.append(np.asarray(catalog.flux, dtype=np.float64))

    if len(names) == 0:
        raise ValueError("At least one source is required.")

    return PointSourceCatalog(
        names=tuple(names),
        ra_deg=np.concatenate(ra_parts),
        dec_deg=np.concatenate(dec_parts),
        flux=np.concatenate(flux_parts),
    )


def resolve_phase_centre(
    catalog: PointSourceCatalog,
    phase_ra_deg: float | None,
    phase_dec_deg: float | None,
) -> tuple[float, float, str]:
    """
    Resolve the tracked phase centre from CLI overrides or the source catalog.

    Parameters
    ----------
    catalog : PointSourceCatalog
        Combined source catalog. The first row is used as the default phase
        centre when no explicit phase-centre coordinates are supplied.
    phase_ra_deg, phase_dec_deg : float or None
        Optional explicit phase-centre coordinates in degrees. They must be
        supplied together.

    Returns
    -------
    tuple[float, float, str]
        ``(ra_deg, dec_deg, source_name)`` for the phase centre. The source
        name is ``"explicit_phase_centre"`` when CLI coordinates are used.

    Raises
    ------
    ValueError
        If the catalog is empty, if only one explicit coordinate is provided,
        or if any resolved coordinate is invalid.
    """
    if len(catalog.names) == 0:
        raise ValueError("At least one source is required to resolve the phase centre.")

    if (phase_ra_deg is None) != (phase_dec_deg is None):
        raise ValueError(
            "Provide both --phase-ra-deg and --phase-dec-deg, or omit both "
            "to use the first source as the phase centre."
        )

    if phase_ra_deg is None and phase_dec_deg is None:
        ra = float(catalog.ra_deg[0]) % 360.0
        dec = float(catalog.dec_deg[0])
        source_name = catalog.names[0]
    else:
        ra = float(phase_ra_deg) % 360.0
        dec = float(phase_dec_deg)
        source_name = "explicit_phase_centre"

    if not np.isfinite(ra):
        raise ValueError("Resolved phase-centre RA must be finite.")
    if not np.isfinite(dec) or dec < -90.0 or dec > 90.0:
        raise ValueError("Resolved phase-centre Dec must be finite and within [-90, 90] deg.")

    return ra, dec, source_name


def build_observation_times(start_time: str, n_times: int, cadence_sec: float) -> Time:
    """
    Build a uniformly sampled UTC observation-time axis.

    Parameters
    ----------
    start_time : str
        UTC start time accepted by ``astropy.time.Time``.
    n_times : int
        Number of time samples. Must be at least one.
    cadence_sec : float
        Spacing between samples in seconds. Must be finite and non-negative.

    Returns
    -------
    astropy.time.Time
        Time array with shape ``(n_times,)``.

    Raises
    ------
    ValueError
        If *n_times* or *cadence_sec* are outside their supported ranges.
    """
    if int(n_times) < 1:
        raise ValueError("n_times must be at least 1.")
    if not np.isfinite(cadence_sec) or cadence_sec < 0.0:
        raise ValueError("cadence_sec must be finite and non-negative.")
    return (
        Time(start_time, scale="utc")
        + np.arange(int(n_times), dtype=np.float64) * cadence_sec * u.s
    )


def build_baseline_pairs(
    n_antennas: int,
    *,
    mode: str,
    ref_index: int,
) -> tuple[tuple[int, int], ...]:
    """
    Build antenna-index baseline pairs.

    Parameters
    ----------
    n_antennas : int
        Number of antennas in the array.
    mode : {"all-pairs", "reference"}
        ``"all-pairs"`` returns every unique ``i < j`` pair. ``"reference"``
        returns only baselines from *ref_index* to every other antenna.
    ref_index : int
        Reference antenna used by ``"reference"`` mode.

    Returns
    -------
    tuple of tuple of int
        Antenna-index pairs aligned with the visibility baseline axis.

    Raises
    ------
    ValueError
        If fewer than two antennas are available or if *mode* / *ref_index* is
        invalid.
    """
    if n_antennas < 2:
        raise ValueError(
            "At least two antennas are required to simulate visibilities."
        )
    if ref_index < 0 or ref_index >= n_antennas:
        raise ValueError(f"ref_index must be in [0, {n_antennas - 1}], got {ref_index}.")
    if mode == "all-pairs":
        return tuple(
            (i, j) for i in range(n_antennas) for j in range(i + 1, n_antennas)
        )
    if mode == "reference":
        return tuple((ref_index, j) for j in range(n_antennas) if j != ref_index)
    raise ValueError(f"Unsupported baseline mode {mode!r}.")


def _baseline_from_antenna_uvw(
    antenna_uvw_m: np.ndarray,
    baseline_pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    uvw_array = np.asarray(antenna_uvw_m, dtype=np.float64)
    if uvw_array.ndim < 2 or uvw_array.shape[-1] != 3:
        raise ValueError(
            f"UVW array must end with a 3-component axis, got {uvw_array.shape!r}."
        )
    if len(baseline_pairs) == 0:
        raise ValueError("At least one baseline pair is required.")

    baselines = [
        uvw_array[j, ...] - uvw_array[i, ...]
        for i, j in baseline_pairs
    ]
    return np.asarray(baselines, dtype=np.float64)


def normalised_visibility_amplitude(visibilities: np.ndarray) -> np.ndarray:
    """
    Normalise finite visibility amplitudes to the maximum finite amplitude.

    Parameters
    ----------
    visibilities : numpy.ndarray
        Complex visibility array of any shape.

    Returns
    -------
    numpy.ndarray
        Real array with the same shape as *visibilities*. Non-finite values are
        treated as zero before normalisation.
    """
    amp = np.abs(np.asarray(visibilities))
    finite_amp = np.where(np.isfinite(amp), amp, 0.0)
    reference = float(np.max(finite_amp)) if finite_amp.size else 0.0
    if reference == 0.0:
        return np.zeros_like(finite_amp, dtype=np.float64)
    return np.clip(finite_amp / reference, 0.0, 1.0)


def simulate_point_source_visibilities(
    antenna_locations: Sequence,
    obs_times: Time,
    phase_ra_deg: float,
    phase_dec_deg: float,
    catalog: PointSourceCatalog,
    frequency_hz: float,
    *,
    baseline_mode: str = "all-pairs",
    ref_index: int = 0,
    phase_sign: float = -1.0,
) -> VisibilitySimulationResult:
    """
    Simulate far-field visibilities for fixed celestial point sources.

    Parameters
    ----------
    antenna_locations : sequence
        Antenna locations accepted by :func:`scepter.uvw.compute_uvw`, normally
        ``astropy.coordinates.EarthLocation`` objects from
        :func:`scepter.uvw.load_telescope_array_file`.
    obs_times : astropy.time.Time
        Observation times. A scalar time is accepted but the returned arrays use
        an explicit length-1 time axis.
    phase_ra_deg, phase_dec_deg : float
        ICRS phase-centre coordinates in degrees.
    catalog : PointSourceCatalog
        Fixed ICRS point-source catalog. Flux values are linear amplitudes.
    frequency_hz : float
        Observing frequency in hertz.
    baseline_mode : {"all-pairs", "reference"}, optional
        Baseline axis layout. Default is every unique antenna pair.
    ref_index : int, optional
        Reference antenna passed to :func:`scepter.uvw.compute_uvw` and used by
        ``baseline_mode="reference"``.
    phase_sign : float, optional
        Phase sign convention. The default ``-1`` computes
        ``exp(-2*pi*i*delta_w/lambda)``.

    Returns
    -------
    VisibilitySimulationResult
        Summed and per-source visibilities plus UVW/phase metadata.

    Raises
    ------
    ValueError
        If coordinates, frequency, antennas, or array shapes are invalid.

    Notes
    -----
    For each physical baseline ``b`` and source direction ``s``, the geometric
    contribution relative to the tracked phase centre ``s0`` is
    ``b · (s - s0)``. Since :func:`scepter.uvw.compute_uvw` stores this
    projection as the ``w`` component for a given source direction, the script
    computes ``delta_w = w_source - w_phase`` and applies the complex fringe
    factor at the requested wavelength.
    """
    n_ant = len(antenna_locations)
    baseline_pairs = build_baseline_pairs(n_ant, mode=baseline_mode, ref_index=ref_index)

    if not np.isfinite(phase_ra_deg):
        raise ValueError("phase_ra_deg must be finite.")
    if not np.isfinite(phase_dec_deg) or phase_dec_deg < -90.0 or phase_dec_deg > 90.0:
        raise ValueError("phase_dec_deg must be finite and within [-90, 90] deg.")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive.")
    if len(catalog.names) == 0:
        raise ValueError("At least one source is required.")

    time_count = 1 if obs_times.isscalar else int(np.asarray(obs_times.mjd).size)
    time_axis = (
        Time([obs_times.iso], scale=obs_times.scale)
        if obs_times.isscalar
        else obs_times
    )

    pointing_uvw_ant_m, _ = uvw.compute_uvw(
        list(antenna_locations),
        ra_deg=float(phase_ra_deg) % 360.0,
        dec_deg=float(phase_dec_deg),
        obs_times=time_axis,
        ref_index=ref_index,
    )
    pointing_uvw_ant_m = np.asarray(pointing_uvw_ant_m, dtype=np.float64)
    if pointing_uvw_ant_m.ndim == 2:
        pointing_uvw_ant_m = pointing_uvw_ant_m[:, np.newaxis, :]

    source_count = len(catalog.names)
    ra_tracks = np.broadcast_to(catalog.ra_deg[np.newaxis, :], (time_count, source_count))
    dec_tracks = np.broadcast_to(catalog.dec_deg[np.newaxis, :], (time_count, source_count))
    source_uvw_ant_m, _ = uvw.compute_uvw(
        list(antenna_locations),
        ra_deg=ra_tracks,
        dec_deg=dec_tracks,
        obs_times=time_axis,
        ref_index=ref_index,
    )
    source_uvw_ant_m = np.asarray(source_uvw_ant_m, dtype=np.float64)
    expected_source_shape = (n_ant, time_count, source_count, 3)
    if source_uvw_ant_m.shape != expected_source_shape:
        raise ValueError(
            "Unexpected source UVW shape from scepter.uvw.compute_uvw. "
            f"Expected {expected_source_shape!r}, got {source_uvw_ant_m.shape!r}."
        )

    pointing_uvw_m = _baseline_from_antenna_uvw(pointing_uvw_ant_m, baseline_pairs)
    source_uvw_m = _baseline_from_antenna_uvw(source_uvw_ant_m, baseline_pairs)
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    delta_w_m = source_uvw_m[..., 2] - pointing_uvw_m[..., 2][:, :, np.newaxis]
    phase_rad = float(phase_sign) * 2.0 * np.pi * delta_w_m / wavelength_m
    vis_per_source = catalog.flux[np.newaxis, np.newaxis, :] * np.exp(1j * phase_rad)
    vis = np.sum(vis_per_source, axis=-1)

    return VisibilitySimulationResult(
        vis=np.asarray(vis, dtype=np.complex128),
        vis_per_source=np.asarray(vis_per_source, dtype=np.complex128),
        uvw_m=pointing_uvw_m,
        source_uvw_m=source_uvw_m,
        phase_rad=np.asarray(phase_rad, dtype=np.float64),
        normalised_amplitude=normalised_visibility_amplitude(vis),
        baseline_pairs=baseline_pairs,
    )


def save_visibility_archive(
    output_path: str | Path,
    result: VisibilitySimulationResult,
    *,
    catalog: PointSourceCatalog,
    antenna_names: Sequence[str],
    obs_times: Time,
    frequency_hz: float,
    phase_ra_deg: float,
    phase_dec_deg: float,
    baseline_mode: str,
    phase_sign: float,
    phase_source_name: str,
    compressed: bool = True,
) -> Path:
    """
    Save a point-source visibility simulation to ``.npz``.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Destination archive path.
    result : VisibilitySimulationResult
        Simulation products.
    catalog : PointSourceCatalog
        Source metadata to store alongside the visibilities.
    antenna_names : sequence of str
        Antenna names aligned with the input array file.
    obs_times : astropy.time.Time
        Observation times aligned with the time axis.
    frequency_hz : float
        Observing frequency in hertz.
    phase_ra_deg, phase_dec_deg : float
        Tracked ICRS phase centre in degrees.
    baseline_mode : str
        Baseline layout label.
    phase_sign : float
        Phase convention sign.
    phase_source_name : str
        Name of the source used as the default phase centre, or
        ``"explicit_phase_centre"`` when CLI override coordinates were used.
    compressed : bool, optional
        If ``True`` (default), use ``numpy.savez_compressed``.

    Returns
    -------
    pathlib.Path
        Written archive path.

    Notes
    -----
    Stable keys are ``uvw`` for phase-centre baseline UVW metres and ``vis``
    for summed complex visibilities. Additional keys retain per-source and
    provenance metadata.
    """
    path = Path(output_path)
    baseline_pairs = np.asarray(result.baseline_pairs, dtype=np.int64)
    baseline_names = np.asarray(
        [f"{antenna_names[i]}-{antenna_names[j]}" for i, j in result.baseline_pairs],
        dtype=str,
    )
    mjds = np.asarray(obs_times.mjd, dtype=np.float64)
    if mjds.ndim == 0:
        mjds = mjds.reshape(1)

    payload = {
        "schema_name": np.asarray("scepter_celestial_point_source_visibility_npz"),
        "schema_version": np.asarray("1"),
        "freq_hz": np.float64(frequency_hz),
        "freq_mhz": np.float64(frequency_hz / 1.0e6),
        "mjds": mjds,
        "phase_ra_deg": np.float64(float(phase_ra_deg) % 360.0),
        "phase_dec_deg": np.float64(phase_dec_deg),
        "phase_source_name": np.asarray(phase_source_name),
        "phase_sign": np.float64(phase_sign),
        "baseline_mode": np.asarray(baseline_mode),
        "antenna_names": np.asarray(tuple(antenna_names), dtype=str),
        "baseline_pairs": baseline_pairs,
        "baseline_names": baseline_names,
        "source_names": np.asarray(catalog.names, dtype=str),
        "source_ra_deg": np.asarray(catalog.ra_deg, dtype=np.float64),
        "source_dec_deg": np.asarray(catalog.dec_deg, dtype=np.float64),
        "source_flux": np.asarray(catalog.flux, dtype=np.float64),
        "uvw": np.asarray(result.uvw_m, dtype=np.float64),
        "source_uvw_m": np.asarray(result.source_uvw_m, dtype=np.float64),
        "vis": np.asarray(result.vis, dtype=np.complex128),
        "vis_per_source": np.asarray(result.vis_per_source, dtype=np.complex128),
        "phase_rad": np.asarray(result.phase_rad, dtype=np.float64),
        "normalised_amplitude": np.asarray(
            result.normalised_amplitude,
            dtype=np.float64,
        ),
    }
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **payload)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate far-field complex visibilities for RA/Dec point sources."
    )
    parser.add_argument(
        "--array-file",
        default=DEFAULT_ARRAY_FILE,
        help=(
            "Telescope array coordinate file accepted by "
            "scepter.uvw.load_telescope_array_file. "
            f"Default: {DEFAULT_ARRAY_FILE}."
        ),
    )
    parser.add_argument(
        "--source-file",
        action="append",
        default=None,
        help=(
            "Point-source catalog file. May be supplied multiple times. "
            "Rows use name, ra_deg, dec_deg, flux. "
            f"Default: {DEFAULT_SOURCE_FILE} when no --source is supplied."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Inline source in 'name,ra_deg,dec_deg,flux' form. May be repeated.",
    )
    parser.add_argument(
        "--phase-ra-deg",
        type=float,
        default=None,
        help="Optional phase-centre RA in degrees. Defaults to the first source RA.",
    )
    parser.add_argument(
        "--phase-dec-deg",
        type=float,
        default=None,
        help="Optional phase-centre Dec in degrees. Defaults to the first source Dec.",
    )
    parser.add_argument(
        "--start-time",
        default=DEFAULT_START_TIME,
        help=f"UTC start time. Default: {DEFAULT_START_TIME}.",
    )
    parser.add_argument(
        "--n-times",
        type=int,
        default=DEFAULT_N_TIMES,
        help=f"Number of time samples. Default: {DEFAULT_N_TIMES}.",
    )
    parser.add_argument(
        "--cadence-sec",
        type=float,
        default=DEFAULT_CADENCE_SEC,
        help=f"Time cadence in seconds. Default: {DEFAULT_CADENCE_SEC:g}.",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=DEFAULT_FREQ_MHZ,
        help=f"Observing frequency in MHz. Default: {DEFAULT_FREQ_MHZ:g}.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output .npz path. Default: {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("all-pairs", "reference"),
        default="all-pairs",
        help="Baseline layout. Default: all-pairs.",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=0,
        help="Reference antenna index for UVW construction and reference baseline mode.",
    )
    parser.add_argument(
        "--altitude-unit",
        default="m",
        help="Unit for the array-file altitude column. Default: m.",
    )
    parser.add_argument(
        "--phase-sign",
        type=float,
        default=-1.0,
        help="Phase sign convention in exp(sign * 2*pi*i*delta_w/lambda). Default: -1.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use numpy.savez instead of numpy.savez_compressed.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    catalogs: list[PointSourceCatalog] = []
    source_files = (
        list(args.source_file)
        if args.source_file is not None
        else ([] if args.source else [DEFAULT_SOURCE_FILE])
    )
    catalogs.extend(load_point_source_catalog(path) for path in source_files)
    catalogs.extend(parse_inline_source(spec) for spec in args.source)
    if len(catalogs) == 0:
        parser.error("Provide at least one --source-file or --source.")
    catalog = combine_catalogs(catalogs)
    phase_ra_deg, phase_dec_deg, phase_source_name = resolve_phase_centre(
        catalog,
        args.phase_ra_deg,
        args.phase_dec_deg,
    )

    geometry = uvw.load_telescope_array_file(
        args.array_file,
        altitude_unit=u.Unit(args.altitude_unit),
    )
    obs_times = build_observation_times(args.start_time, args.n_times, args.cadence_sec)
    frequency_hz = float(args.freq_mhz) * 1.0e6
    result = simulate_point_source_visibilities(
        geometry.earth_locations,
        obs_times,
        phase_ra_deg,
        phase_dec_deg,
        catalog,
        frequency_hz,
        baseline_mode=args.baseline_mode,
        ref_index=args.ref_index,
        phase_sign=args.phase_sign,
    )
    output_path = save_visibility_archive(
        args.output,
        result,
        catalog=catalog,
        antenna_names=geometry.antenna_names,
        obs_times=obs_times,
        frequency_hz=frequency_hz,
        phase_ra_deg=phase_ra_deg,
        phase_dec_deg=phase_dec_deg,
        baseline_mode=args.baseline_mode,
        phase_sign=args.phase_sign,
        phase_source_name=phase_source_name,
        compressed=not args.no_compress,
    )

    print(f"Saved point-source visibility archive to {output_path}")
    print(f"Source count       : {len(catalog.names)}")
    print(f"Antenna count      : {len(geometry.antenna_names)}")
    print(f"Baseline count     : {len(result.baseline_pairs)}")
    print(
        "Phase centre       : "
        f"{phase_source_name} ({phase_ra_deg:.6f}, {phase_dec_deg:.6f}) deg"
    )
    print(f"Time samples       : {result.vis.shape[1]}")
    print(f"UVW shape          : {result.uvw_m.shape}")
    print(f"Summed vis shape   : {result.vis.shape}")
    print(f"Per-source shape   : {result.vis_per_source.shape}")


if __name__ == "__main__":
    main()
