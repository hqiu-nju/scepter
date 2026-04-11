"""
Artificial TLE generation helpers for SCEPTer workflows.

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>

This module remains part of the GPLv3-licensed SCEPTer project. The author
attribution above does not change the project-wide license.
"""
import math
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.constants import GM_earth, R_earth
from cysgp4 import PyTle
from pycraf.utils import ranged_quantity_input

_tle_counter = 0
_COSPAR_PIECES_PER_LAUNCH = 18278


def reset_tle_counter() -> None:
    """
    Reset the internal TLE counter back to zero.
    Should be used for subsequent runs using same environment session to avoid overflow.
    """
    global _tle_counter
    _tle_counter = 0


def _compute_tle_checksum(tle_line: str) -> int:
    """
    Compute the NORAD checksum for one TLE line.

    Digits are summed directly, each '-' contributes +1, and all other
    characters contribute 0. The result is modulo 10.
    """
    return sum(int(ch) if ch.isdigit() else 1 if ch == "-" else 0 for ch in tle_line) % 10


def _format_tle_exp(value: float) -> str:
    """
    Convert a float to TLE 8-character exponential format.

    Format:
        [sign-or-space][5-digit mantissa][sign exponent][1-digit exponent]
    """
    sign_mantissa = "-" if value < 0 else " "
    value_abs = math.fabs(value)
    if value_abs < 1.0e-12:
        return f"{sign_mantissa}00000-0"

    exponent = int(math.floor(math.log10(value_abs)) + 1)
    mantissa = value_abs / (10.0 ** exponent)
    if mantissa >= 1.0:
        # Edge case when the value is an exact power of 10 after rounding.
        mantissa /= 10.0
        exponent += 1

    mantissa_int = int(round(mantissa * 1e5))
    if mantissa_int == 100000:
        mantissa_int = 10000
        exponent += 1

    if exponent < -9 or exponent > 9:
        raise ValueError(
            "Unexpected input parameter. Please check second derivative of mean motion and B*, the drag term, or radiation pressure coefficient"
        )

    sign_exp = "+" if exponent >= 0 else "-"
    return f"{sign_mantissa}{mantissa_int:05d}{sign_exp}{abs(exponent):1d}"


def _index_to_piece(idx: int) -> str:
    """
    Convert a zero-based piece index to COSPAR piece code.

    Valid input range is 0..18277 (inclusive), mapping to A..ZZZ.
    """
    if idx < 0 or idx >= _COSPAR_PIECES_PER_LAUNCH:
        raise ValueError(f"Piece index {idx} out of range (0..{_COSPAR_PIECES_PER_LAUNCH - 1}).")

    # Work with 1-based indexing for the classic A=1..Z=26 conversion.
    n = idx + 1
    letters = []
    while n > 0:
        n -= 1
        letters.append(chr(ord("A") + (n % 26)))
        n //= 26
    return "".join(reversed(letters))


def _build_epoch_string(start_time: Time) -> tuple[int, str]:
    """Return (2-digit year, TLE epoch field) for a given start time."""
    year_short = start_time.datetime.year % 100
    start_of_year = Time(
        start_time.datetime.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    )
    day_of_year = (start_time - start_of_year).to_value("day") + 1.0
    epoch_str = f"{year_short:02d}{day_of_year:012.8f}"
    return year_short, epoch_str


def _compute_mean_motion_rev_day(altitude_m: float, eccentricity: float = 0.0) -> float:
    """Compute mean motion [rev/day] from perigee altitude and eccentricity.

    For circular orbits (e=0), altitude is the orbital height.  For elliptical
    orbits, altitude is the *perigee* height and the semi-major axis is derived
    as ``a = (R_earth + altitude) / (1 - e)``.
    """
    r_perigee_m = R_earth.value + altitude_m
    ecc = float(max(0.0, min(eccentricity, 0.9999)))
    semi_major_axis_m = r_perigee_m / (1.0 - ecc)
    mean_motion_rad_s = math.sqrt(GM_earth.value / semi_major_axis_m**3)
    return mean_motion_rad_s * (86400.0 / (2.0 * math.pi))


def _format_mm_dot(mm_dot: float) -> str:
    """Format first derivative of mean motion for TLE line 1."""
    mm_dot_fmt = f"{mm_dot:+.8f}"
    return mm_dot_fmt[:1] + mm_dot_fmt[2:]


def _to_float_m(value: float | u.Quantity) -> float:
    """Return a length value in meters as plain float."""
    if isinstance(value, u.Quantity):
        return float(value.to_value(u.m))
    return float(value)


def _to_float_deg(value: float | u.Quantity) -> float:
    """Return an angular value in degrees as plain float."""
    if isinstance(value, u.Quantity):
        return float(value.to_value(u.deg))
    return float(value)


def _forge_tle_single_core(
    *,
    sat_name: str,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argp_deg: float,
    anomaly_deg: float,
    year_short: int,
    epoch_str: str,
    mean_motion_rev_day: float,
    mm_dot_string: str,
    mm_ddot_string: str,
    bstar_string: str,
) -> PyTle:
    """
    Fast internal TLE constructor with pre-normalized scalar inputs.
    """
    global _tle_counter
    _tle_counter += 1
    sat_number = _tle_counter  # Artificial NORAD ID for generated catalog.

    if sat_number > 99999:
        raise ValueError(
            f"Cannot assign NORAD ID {sat_number:d}: exceeds 5-digit limit (max 99999)."
        )

    total_idx = sat_number - 1
    piece_idx = total_idx % _COSPAR_PIECES_PER_LAUNCH
    launch_no = (total_idx // _COSPAR_PIECES_PER_LAUNCH) + 1
    if launch_no > 999:
        raise ValueError(
            f"Cannot assign COSPAR launch number {launch_no}: exceeds 3 digits (max 999)."
        )

    piece_str = _index_to_piece(piece_idx)
    int_desg = f"{year_short:02d}{launch_no:03d}{piece_str:>3s}"
    classification = "U"
    ephemeris_type = 0
    element_set_number = 1
    rev_number = 1  # Dummy revolution number.

    line1 = (
        f"1 {sat_number:05d}{classification} {int_desg:8} {epoch_str:14} "
        f"{mm_dot_string} {mm_ddot_string} {bstar_string} "
        f"{ephemeris_type} {element_set_number:4d}"
    )
    if len(line1) != 68:
        raise ValueError("Line 1 is not 68 characters long before adding checksum.")

    ecc_str = f"{eccentricity:.7f}"[2:]  # Remove "0." to get 7 digits.
    line2 = (
        "2 {:05d} {:8.4f} {:8.4f} {:7} {:8.4f} {:8.4f} {:11.8f}{:05d}".format(
            sat_number,
            inclination_deg,
            raan_deg,
            ecc_str,
            argp_deg,
            anomaly_deg,
            mean_motion_rev_day,
            rev_number,
        )
    )
    if len(line2) != 68:
        raise ValueError("Line 2 is not 68 characters long before adding checksum.")

    line1 += str(_compute_tle_checksum(line1))
    line2 += str(_compute_tle_checksum(line2))
    return PyTle(sat_name, line1, line2)


_BELT_REQUIRED_KEYS = {
    "belt_name",
    "num_sats_per_plane",
    "plane_count",
    "altitude",
    "eccentricity",
    "inclination_deg",
    "argp_deg",
    "RAAN_min",
    "RAAN_max",
    "min_elevation",
    "adjacent_plane_offset",
}


def normalize_and_validate_belt_cfg(
    cfg: Mapping[str, Any],
    idx: int | None = None,
) -> dict[str, Any]:
    """
    Normalize one belt-definition dictionary into strict units/types and validate it.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Raw belt configuration.
    idx : int | None, optional
        Optional index used in error messages.

    Returns
    -------
    dict[str, Any]
        Normalized config dictionary ready for `forge_tle_belt`.
    """
    missing = sorted(_BELT_REQUIRED_KEYS - set(cfg.keys()))
    if missing:
        loc = f"belt_definitions[{idx}]" if idx is not None else "belt definition"
        raise KeyError(f"{loc} is missing required keys: {missing}")

    out = dict(cfg)
    out["belt_name"] = str(out["belt_name"])
    out["num_sats_per_plane"] = int(out["num_sats_per_plane"])
    out["plane_count"] = int(out["plane_count"])
    out["altitude"] = u.Quantity(out["altitude"]).to(u.km)
    out["eccentricity"] = float(out["eccentricity"])
    out["inclination_deg"] = u.Quantity(out["inclination_deg"]).to(u.deg)
    out["argp_deg"] = u.Quantity(out["argp_deg"]).to(u.deg)
    out["RAAN_min"] = u.Quantity(out["RAAN_min"]).to(u.deg)
    out["RAAN_max"] = u.Quantity(out["RAAN_max"]).to(u.deg)
    out["min_elevation"] = u.Quantity(out["min_elevation"]).to(u.deg)
    out["adjacent_plane_offset"] = bool(out["adjacent_plane_offset"])

    belt_name = out["belt_name"]
    if out["num_sats_per_plane"] <= 0:
        raise ValueError(f"{belt_name}: num_sats_per_plane must be > 0.")
    if out["plane_count"] <= 0:
        raise ValueError(f"{belt_name}: plane_count must be > 0.")
    if not (0.0 <= out["eccentricity"] < 1.0):
        raise ValueError(f"{belt_name}: eccentricity must satisfy 0 <= e < 1.")
    if out["RAAN_max"] <= out["RAAN_min"]:
        raise ValueError(f"{belt_name}: RAAN_max must be greater than RAAN_min.")
    if out["min_elevation"] < (0.0 * u.deg):
        raise ValueError(f"{belt_name}: min_elevation must be >= 0 deg.")
    if out["altitude"] <= (0.0 * u.km):
        raise ValueError(f"{belt_name}: altitude must be > 0 km.")

    return out


def normalize_and_validate_belt_definitions(
    belt_definitions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Normalize and validate an iterable of belt definitions.

    Parameters
    ----------
    belt_definitions : Iterable[Mapping[str, Any]]
        Input belt-definition dictionaries.

    Returns
    -------
    list[dict[str, Any]]
        Normalized belt definitions in input order.
    """
    normalized = [
        normalize_and_validate_belt_cfg(cfg, idx=idx)
        for idx, cfg in enumerate(belt_definitions)
    ]
    if len(normalized) == 0:
        raise ValueError("At least one belt definition is required.")
    return normalized


def forge_tle_constellation_from_belt_definitions(
    belt_definitions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Forge a full multi-belt constellation and derived geometry metadata.

    This helper is intended for simulation setup where constellation
    configuration arrives as an iterable of belt-definition mappings.
    It performs all normalization/validation, forges one TLE catalog for each
    belt, and then aggregates both per-belt metadata and a flat TLE list.

    Parameters
    ----------
    belt_definitions : Iterable[Mapping[str, Any]]
        Belt configuration dictionaries. Required keys for each mapping are:
        `belt_name`, `num_sats_per_plane`, `plane_count`, `altitude`,
        `eccentricity`, `inclination_deg`, `argp_deg`, `RAAN_min`,
        `RAAN_max`, `min_elevation`, and `adjacent_plane_offset`.
        Unit-bearing angular/length values are expected as astropy quantities.

    Returns
    -------
    dict[str, Any]
        Aggregated result dictionary with the following keys:

        `belt_definitions`
            Normalized belt definitions in input order.
        `belt_names`
            List of belt names in input order.
        `belt_sats`
            `np.ndarray[int64]` with satellite count per belt.
        `altitudes_q`
            `Quantity` array of belt altitudes.
        `inclinations_q`
            `Quantity` array of belt inclinations.
        `min_elevations_q`
            `Quantity` array of operational minimum elevation constraints.
        `max_deviation_angles_q`
            `Quantity` array with geometric off-nadir limits
            `asin(R_earth / (R_earth + altitude))`.
        `max_betas_q`
            `Quantity` array with operational beta limits
            `asin((R_earth / (R_earth + altitude)) * cos(min_elevation))`.
        `tle_list`
            Flat object array of `PyTle`, concatenated belt-by-belt.

    Raises
    ------
    KeyError
        If any belt definition is missing required keys.
    ValueError
        If any normalized value violates validation constraints.
    RuntimeError
        If forged satellite counts do not match expected belt sizes.

    Notes
    -----
    Output ordering is stable and intentionally notebook-friendly:

    - belt-level arrays preserve the input belt-definition order
    - ``tle_list`` is concatenated belt-by-belt in that same order
    - downstream helpers such as :func:`expand_belt_metadata_to_satellites`
      therefore emit per-satellite arrays aligned to the forged TLE order
    """
    belt_definitions_norm = normalize_and_validate_belt_definitions(belt_definitions)

    belt_names: list[str] = []
    belt_sats: list[int] = []
    altitudes: list[u.Quantity] = []
    inclinations: list[u.Quantity] = []
    min_elevations: list[u.Quantity] = []
    max_betas: list[u.Quantity] = []
    max_deviation_angles: list[u.Quantity] = []
    tle_list: list[Any] = []

    for belt_cfg in belt_definitions_norm:
        belt_name = belt_cfg["belt_name"]
        altitude = belt_cfg["altitude"]
        min_elevation = belt_cfg["min_elevation"]

        # Off-nadir envelope implied by Earth geometry.
        sat_radius = R_earth + altitude
        beta_abs_max = np.arcsin(R_earth / sat_radius).to(u.deg)
        beta_max_oper = np.arcsin((R_earth / sat_radius) * np.cos(min_elevation)).to(u.deg)

        belt_tle_list = forge_tle_belt(
            belt_name=belt_name,
            num_sats_per_plane=belt_cfg["num_sats_per_plane"],
            plane_count=belt_cfg["plane_count"],
            RAAN_min=belt_cfg["RAAN_min"],
            RAAN_max=belt_cfg["RAAN_max"],
            altitude=altitude,
            eccentricity=belt_cfg["eccentricity"],
            inclination_deg=belt_cfg["inclination_deg"],
            argp_deg=belt_cfg["argp_deg"],
            adjacent_plane_offset=belt_cfg["adjacent_plane_offset"],
        )

        expected_sats = int(belt_cfg["num_sats_per_plane"]) * int(belt_cfg["plane_count"])
        if len(belt_tle_list) != expected_sats:
            raise RuntimeError(
                f"{belt_name}: forged TLE count mismatch ({len(belt_tle_list)} != expected {expected_sats})."
            )

        belt_names.append(belt_name)
        belt_sats.append(len(belt_tle_list))
        altitudes.append(altitude)
        inclinations.append(belt_cfg["inclination_deg"])
        min_elevations.append(min_elevation)
        max_deviation_angles.append(beta_abs_max)
        max_betas.append(beta_max_oper)
        tle_list.extend(belt_tle_list)

    return {
        "belt_definitions": belt_definitions_norm,
        "belt_names": belt_names,
        "belt_sats": np.asarray(belt_sats, dtype=np.int64),
        "altitudes_q": u.Quantity(altitudes),
        "inclinations_q": u.Quantity(inclinations),
        "min_elevations_q": u.Quantity(min_elevations),
        "max_deviation_angles_q": u.Quantity(max_deviation_angles),
        "max_betas_q": u.Quantity(max_betas),
        "tle_list": np.asarray(tle_list, dtype=object),
    }


def summarize_constellation_geometry(
    constellation: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build notebook-facing summary lines for a forged constellation.

    Parameters
    ----------
    constellation : Mapping[str, Any]
        Result dictionary returned by
        :func:`forge_tle_constellation_from_belt_definitions`.

    Returns
    -------
    dict[str, Any]
        Dictionary containing per-belt slant-range diagnostics plus
        ``summary_lines`` suitable for direct notebook printing.

    Raises
    ------
    KeyError
        Raised when the required forged-constellation keys are missing.
    """
    belt_definitions = constellation["belt_definitions"]
    belt_names = list(constellation["belt_names"])
    belt_sats = np.asarray(constellation["belt_sats"], dtype=np.int64)
    altitudes_q = u.Quantity(constellation["altitudes_q"]).to(u.km)
    inclinations_q = u.Quantity(constellation["inclinations_q"]).to(u.deg)
    min_elevations_q = u.Quantity(constellation["min_elevations_q"]).to(u.deg)
    max_deviation_angles_q = u.Quantity(constellation["max_deviation_angles_q"]).to(u.deg)
    max_betas_q = u.Quantity(constellation["max_betas_q"]).to(u.deg)
    tle_list = np.asarray(constellation["tle_list"], dtype=object)

    slant_range_abs_per_belt = np.sqrt((R_earth + altitudes_q) ** 2 - R_earth**2).to(u.km)
    slant_range_oper_per_belt = (
        -R_earth * np.sin(min_elevations_q)
        + np.sqrt((R_earth + altitudes_q) ** 2 - (R_earth * np.cos(min_elevations_q)) ** 2)
    ).to(u.km)

    idx_abs = int(np.argmax(altitudes_q.to_value(u.km)))
    idx_oper = int(np.argmax(slant_range_oper_per_belt.to_value(u.km)))

    summary_lines = [""]
    summary_lines.append("Belt configuration summary:")
    for idx_cfg, cfg in enumerate(belt_definitions):
        summary_lines.append(
            f"  [{idx_cfg}] {cfg['belt_name']}: "
            f"planes={cfg['plane_count']}, sats/plane={cfg['num_sats_per_plane']}, "
            f"alt={cfg['altitude'].to_value(u.km):.1f} km, "
            f"inc={cfg['inclination_deg'].to_value(u.deg):.2f} deg, "
            f"e={cfg['eccentricity']:.4f}, "
            f"RAAN=[{cfg['RAAN_min'].to_value(u.deg):.1f}, {cfg['RAAN_max'].to_value(u.deg):.1f}] deg, "
            f"min_el={cfg['min_elevation'].to_value(u.deg):.1f} deg, "
            f"adj_offset={cfg['adjacent_plane_offset']}"
        )
    summary_lines.append(f"Total number of satellites: {int(tle_list.size)}")
    summary_lines.append(
        f"Defined belts: {belt_names}\n"
        f"  sats per belt     = {belt_sats.tolist()}\n"
        f"  altitude          = {[float(v) for v in altitudes_q.to_value(u.km)]} km\n"
        f"  inclination       = {[float(v) for v in inclinations_q.to_value(u.deg)]} deg\n"
        f"  min elevation     = {[float(v) for v in min_elevations_q.to_value(u.deg)]} deg\n"
        f"  β_abs_max (e=0°)  = {[float(v) for v in max_deviation_angles_q.to_value(u.deg)]} deg\n"
        f"  β_max (min elev)  = {[float(v) for v in max_betas_q.to_value(u.deg)]} deg"
    )
    summary_lines.append("")
    summary_lines.append("Geometry limits:")
    summary_lines.append(
        f"  Highest altitude belt: {belt_names[idx_abs]} at "
        f"{altitudes_q[idx_abs].to_value(u.km):.2f} km"
    )
    summary_lines.append(
        "    slant_range_abs_max (e=0°) = "
        f"{slant_range_abs_per_belt[idx_abs].to_value(u.km):.2f} km at β_abs_max = "
        f"{max_deviation_angles_q[idx_abs].to_value(u.deg):.2f}°"
    )
    summary_lines.append(
        f"  Operational max (uses min elevation): {belt_names[idx_oper]} at "
        f"{altitudes_q[idx_oper].to_value(u.km):.2f} km"
    )
    summary_lines.append(
        f"    min_elevation = {min_elevations_q[idx_oper].to_value(u.deg):.2f}° ⇒ "
        f"slant_range_max = {slant_range_oper_per_belt[idx_oper].to_value(u.km):.2f} km "
        f"at β_max = {max_betas_q[idx_oper].to_value(u.deg):.2f}°"
    )

    return {
        "summary_lines": summary_lines,
        "slant_range_abs_per_belt": slant_range_abs_per_belt,
        "slant_range_oper_per_belt": slant_range_oper_per_belt,
        "slant_distance_abs_max": slant_range_abs_per_belt[idx_abs],
        "slant_distance_max": slant_range_oper_per_belt[idx_oper],
        "idx_abs": idx_abs,
        "idx_oper": idx_oper,
    }


def expand_belt_metadata_to_satellites(
    constellation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """
    Expand per-belt visibility metadata into per-satellite arrays.

    Parameters
    ----------
    constellation : Mapping[str, Any]
        Result dictionary produced by
        :func:`forge_tle_constellation_from_belt_definitions`. The mapping must
        contain ``"belt_sats"``, ``"min_elevations_q"``, ``"max_betas_q"``,
        and ``"tle_list"``.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary aligned with ``constellation["tle_list"]`` and containing:

        ``"sat_min_elevation_deg"``
            ``np.ndarray[float32]`` with shape ``(S,)``. Operational minimum
            elevation angle for each satellite in degrees.
        ``"sat_beta_max_deg"``
            ``np.ndarray[float32]`` with shape ``(S,)``. Operational off-axis
            cone limit for each satellite in degrees.
        ``"sat_belt_id"``
            ``np.ndarray[int16]`` with shape ``(S,)``. Zero-based belt index
            for each satellite.

    Raises
    ------
    KeyError
        If a required key is missing from ``constellation``.
    RuntimeError
        If the expanded arrays do not cover exactly all satellites from
        ``constellation["tle_list"]``.

    Notes
    -----
    The output order matches the belt-by-belt concatenation order used by
    :func:`forge_tle_constellation_from_belt_definitions`, so the returned
    arrays can be passed directly into link-selection and batch-processing
    routines that expect per-satellite constraints.
    """
    required_keys = ("belt_sats", "min_elevations_q", "max_betas_q", "tle_list")
    missing = [key for key in required_keys if key not in constellation]
    if missing:
        raise KeyError(
            "constellation is missing required keys for satellite expansion: "
            f"{missing}"
        )

    belt_sats = np.asarray(constellation["belt_sats"], dtype=np.int64)
    min_elevations_q = u.Quantity(constellation["min_elevations_q"]).to(u.deg)
    max_betas_q = u.Quantity(constellation["max_betas_q"]).to(u.deg)
    tle_list = np.asarray(constellation["tle_list"], dtype=object)

    n_belts = int(belt_sats.size)
    if min_elevations_q.shape[0] != n_belts or max_betas_q.shape[0] != n_belts:
        raise RuntimeError(
            "Per-belt metadata shape mismatch: belt_sats, min_elevations_q, and "
            "max_betas_q must describe the same number of belts."
        )

    n_sats = int(tle_list.size)
    sat_min_elevation_deg = np.empty(n_sats, dtype=np.float32)
    sat_beta_max_deg = np.empty(n_sats, dtype=np.float32)
    sat_belt_id = np.empty(n_sats, dtype=np.int16)

    offset = 0
    for belt_idx, sat_count_obj in enumerate(belt_sats):
        sat_count = int(sat_count_obj)
        next_offset = offset + sat_count
        sat_min_elevation_deg[offset:next_offset] = np.float32(
            min_elevations_q[belt_idx].to_value(u.deg)
        )
        sat_beta_max_deg[offset:next_offset] = np.float32(
            max_betas_q[belt_idx].to_value(u.deg)
        )
        sat_belt_id[offset:next_offset] = np.int16(belt_idx)
        offset = next_offset

    if offset != n_sats:
        raise RuntimeError(
            f"Per-satellite mapping mismatch: filled {offset}, expected {n_sats}."
        )

    return {
        "sat_min_elevation_deg": sat_min_elevation_deg,
        "sat_beta_max_deg": sat_beta_max_deg,
        "sat_belt_id": sat_belt_id,
    }


def build_satellite_storage_constants(
    satellite_metadata: Mapping[str, Any],
    *,
    orbit_radius_m_per_sat: u.Quantity | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Build typed per-satellite storage constants for notebook-facing HDF5 output.

    Parameters
    ----------
    satellite_metadata : Mapping[str, Any]
        Per-satellite metadata mapping, typically the result of
        :func:`expand_belt_metadata_to_satellites`. The mapping must contain
        ``"sat_belt_id"``, ``"sat_min_elevation_deg"``, and
        ``"sat_beta_max_deg"`` aligned on the full satellite axis.
    orbit_radius_m_per_sat : astropy.units.Quantity or np.ndarray or None, optional
        Optional orbital radius per satellite. Quantity-like inputs are
        converted to metres. Plain numeric arrays are interpreted as metres.

    Returns
    -------
    dict[str, np.ndarray]
        Typed arrays ready for ``storage_constants`` and notebook runner
        arguments. The returned mapping always contains:

        ``"sat_belt_id_per_sat"``
            ``np.ndarray[int16]`` with shape ``(S,)``.
        ``"sat_min_elev_deg_per_sat"``
            ``np.ndarray[float32]`` with shape ``(S,)``.
        ``"sat_beta_max_deg_per_sat"``
            ``np.ndarray[float32]`` with shape ``(S,)``.

        When ``orbit_radius_m_per_sat`` is provided the mapping also contains
        ``"sat_orbit_radius_m_per_sat"`` as ``np.ndarray[float32]`` with shape
        ``(S,)``.

    Raises
    ------
    KeyError
        Raised when a required metadata key is missing.
    ValueError
        Raised when the supplied arrays do not share the same full-satellite
        length.
    """
    required_keys = (
        "sat_belt_id",
        "sat_min_elevation_deg",
        "sat_beta_max_deg",
    )
    missing = [key for key in required_keys if key not in satellite_metadata]
    if missing:
        raise KeyError(
            "satellite_metadata is missing required keys for storage constants: "
            f"{missing}"
        )

    sat_belt_id_per_sat = np.asarray(
        satellite_metadata["sat_belt_id"],
        dtype=np.int16,
    )
    sat_min_elev_deg_per_sat = np.asarray(
        satellite_metadata["sat_min_elevation_deg"],
        dtype=np.float32,
    )
    sat_beta_max_deg_per_sat = np.asarray(
        satellite_metadata["sat_beta_max_deg"],
        dtype=np.float32,
    )

    sat_count = int(sat_belt_id_per_sat.size)
    if (
        int(sat_min_elev_deg_per_sat.size) != sat_count
        or int(sat_beta_max_deg_per_sat.size) != sat_count
    ):
        raise ValueError(
            "satellite_metadata arrays must share the same full-satellite length."
        )

    result = {
        "sat_belt_id_per_sat": sat_belt_id_per_sat,
        "sat_min_elev_deg_per_sat": sat_min_elev_deg_per_sat,
        "sat_beta_max_deg_per_sat": sat_beta_max_deg_per_sat,
    }

    if orbit_radius_m_per_sat is not None:
        orbit_radius_q = u.Quantity(orbit_radius_m_per_sat, copy=False)
        if orbit_radius_q.unit == u.dimensionless_unscaled:
            orbit_radius_q = orbit_radius_q * u.m
        else:
            orbit_radius_q = orbit_radius_q.to(u.m)
        sat_orbit_radius_m_per_sat = np.asarray(
            orbit_radius_q.to_value(u.m),
            dtype=np.float32,
        )
        if int(sat_orbit_radius_m_per_sat.size) != sat_count:
            raise ValueError(
                "orbit_radius_m_per_sat must align with the full satellite axis."
            )
        result["sat_orbit_radius_m_per_sat"] = sat_orbit_radius_m_per_sat

    return result

@ranged_quantity_input(altitude = (0, 384400000, u.m),
                       inclination_deg = (-360, 360, u.deg),
                       raan_deg = (-360, 360, u.deg),
                       argp_deg = (-360, 360, u.deg),
                       anomaly_deg = (-360, 360, u.deg),
                       strip_input_units=True,
                       allow_none=True)
def forge_tle_single(
    sat_name: str = "Satellite",
    altitude: float = 400000.0 * u.m,
    eccentricity: float = 0.0,
    inclination_deg: float = 90.0 * u.deg,
    raan_deg: float = 0.0 * u.deg,
    argp_deg: float = 0.0 * u.deg,
    anomaly_deg: float = 0.0 * u.deg,
    start_time: Time = Time(datetime(2025, 1, 1, 0, 0, 0)),
    mm_dot: float = 0.0,
    mm_ddot: float = 0.0,
    bstar: float = 0.00,
) -> PyTle:
    """
    Forge a TLE for one satellite from orbital parameters and a reference epoch.

    Conceptually, the generated TLE contains:
      - Line 0: satellite name (`sat_name`)
      - Line 1: metadata fields (NORAD ID, epoch, n-dot, n-ddot, BSTAR, etc.)
      - Line 2: orbital elements (inclination, RAAN, eccentricity, argument of
        perigee, mean anomaly, mean motion)

    The function returns a `PyTle` object, which stores the generated lines and
    exposes them through the cysgp4 API.

    Parameters
    ----------
    sat_name : str, optional
        Satellite name placed on TLE line 0.
        Default is `"Satellite"`.
    altitude : astropy.units.Quantity, optional
        Orbital altitude above Earth's surface. Must be a quantity convertible
        to meters. Internally this is converted to a scalar in meters for
        mean-motion computation.
        Default is `400000.0 * u.m`.
    eccentricity : float, optional
        Orbital eccentricity.
        Default is `0.0`.
    inclination_deg : astropy.units.Quantity, optional
        Inclination angle, expected as quantity convertible to degrees.
        Default is `90.0 * u.deg`.
    raan_deg : astropy.units.Quantity, optional
        Right ascension of the ascending node (RAAN), expected as quantity
        convertible to degrees.
        Default is `0.0 * u.deg`.
    argp_deg : astropy.units.Quantity, optional
        Argument of perigee, expected as quantity convertible to degrees.
        Default is `0.0 * u.deg`.
    anomaly_deg : astropy.units.Quantity, optional
        Mean anomaly at epoch, expected as quantity convertible to degrees.
        Default is `0.0 * u.deg`.
    start_time : astropy.time.Time, optional
        Epoch used for the TLE epoch field (`YYDDD.dddddddd` style in line 1).
        Default is `Time(datetime(2025, 1, 1, 0, 0, 0))`.
    mm_dot : float, optional
        First derivative of mean motion, in rev/day^2.
        Default is `0.0`.
    mm_ddot : float, optional
        Second derivative of mean motion, in rev/day^3.
        Encoded using the TLE mantissa/exponent notation.
        Default is `0.0`.
    bstar : float, optional
        BSTAR drag term in earth-radii^-1.
        Encoded using the TLE mantissa/exponent notation.
        Default is `0.0`.

    Returns
    -------
    PyTle
        Generated TLE object for the requested single satellite.

    Raises
    ------
    TypeError
        If unit-bearing arguments are not passed as quantities compatible with
        the `ranged_quantity_input` decorator constraints.
    ValueError
        If generated identifiers exceed TLE field limits (for example NORAD ID
        > 99999, COSPAR launch > 999), if line lengths are invalid before
        checksum insertion, or if exponent encoding for `mm_ddot`/`bstar`
        exceeds the single-digit exponent range.

    Notes
    -----
    1) Mean motion model:
        Mean motion is derived from a circular-orbit approximation with
        semi-major axis `a = R_earth + altitude`, using astropy constants
        `GM_earth` and `R_earth`.

    2) Epoch formatting:
        The epoch field uses the TLE convention `YY + day_of_year`, where the
        day component includes fractional UTC day.

    3) Catalog bookkeeping:
        The function uses module-global `_tle_counter` to assign sequential
        synthetic NORAD IDs and COSPAR piece identifiers.
        Use `reset_tle_counter()` for deterministic repeated runs.

    4) Field encoding:
        `mm_ddot` and `bstar` are formatted via `_format_tle_exp()` into
        8-character TLE exponent fields:
            `[sign-or-space][5 mantissa digits][exp sign][1 exp digit]`

    5) Checksums:
        Checksums for lines 1 and 2 are computed with `_compute_tle_checksum()`
        and appended as column 69.
    """
    year_short, epoch_str = _build_epoch_string(start_time)
    mean_motion_rev_day = _compute_mean_motion_rev_day(float(altitude), eccentricity=float(eccentricity))

    return _forge_tle_single_core(
        sat_name=sat_name,
        eccentricity=eccentricity,
        inclination_deg=float(inclination_deg),
        raan_deg=float(raan_deg),
        argp_deg=float(argp_deg),
        anomaly_deg=float(anomaly_deg),
        year_short=year_short,
        epoch_str=epoch_str,
        mean_motion_rev_day=mean_motion_rev_day,
        mm_dot_string=_format_mm_dot(mm_dot),
        mm_ddot_string=_format_tle_exp(mm_ddot),
        bstar_string=_format_tle_exp(bstar),
    )


@ranged_quantity_input(RAAN_min = (0, 360, u.deg),
                       RAAN_max = (0, 360, u.deg),
                       altitude = (0, 384400000, u.m),
                       inclination_deg = (-360, 360, u.deg),
                       argp_deg = (-360, 360, u.deg),
                       strip_input_units=False,
                       allow_none=True)
def forge_tle_belt(
    belt: Any | None = None,
    belt_name: str = "SystemC_Belt_1",
    num_sats_per_plane: int = 40,
    plane_count: int = 18,
    RAAN_min: float = 0 * u.deg,
    RAAN_max: float = 180 * u.deg,
    altitude: float = 1200000.0 * u.m,
    eccentricity: float = 0,
    inclination_deg: float = 87.9 * u.deg,
    argp_deg: float = 0 * u.deg,
    start_time: Time = Time(datetime(2025, 1, 1, 0, 0, 0)),
    mm_dot: float = 0.0,
    mm_ddot: float = 0.0,
    bstar: float = 0.00,
    adjacent_plane_offset: bool = False,
) -> np.ndarray:
    """
    Generate TLEs for a constellation belt arranged in multiple orbital planes.

    This routine creates `plane_count * num_sats_per_plane` satellites and
    returns them as an object array of `PyTle`.
    Satellite names follow:
        `<belt_name>_Plane_<plane_idx>_Satellite_<sat_idx>`
    where indices are one-based in the emitted names.

    Parameters
    ----------
    belt : object or None, optional
        Placeholder for future belt-object support. Non-`None` values are not
        implemented and raise `NotImplementedError`.
    belt_name : str, optional
        Prefix used for generated satellite names:
        `<belt_name>_Plane_<plane_idx>_Satellite_<sat_idx>`.
    num_sats_per_plane : int, optional
        Number of satellites per plane.
    plane_count : int, optional
        Number of orbital planes.
    altitude : astropy.units.Quantity, optional
        Orbit altitude above Earth.
    RAAN_min : astropy.units.Quantity, optional
        Minimum RAAN for plane distribution.
    RAAN_max : astropy.units.Quantity, optional
        Maximum RAAN for plane distribution.
    eccentricity : float, optional
        Eccentricity shared by all satellites in the belt.
    inclination_deg : astropy.units.Quantity, optional
        Inclination shared by all planes.
    argp_deg : astropy.units.Quantity, optional
        Argument of perigee shared by all satellites.
    start_time : astropy.time.Time, optional
        Common TLE epoch.
    mm_dot : float, optional
        First derivative of mean motion [rev/day^2].
    mm_ddot : float, optional
        Second derivative of mean motion [rev/day^3], TLE encoded.
    bstar : float, optional
        BSTAR drag term [earth-radii^-1], TLE encoded.
    adjacent_plane_offset : bool, optional
        If `True`, odd planes are shifted by half an in-plane spacing.

    Returns
    -------
    np.ndarray
        Object array of `PyTle` with length `plane_count * num_sats_per_plane`.

    Raises
    ------
    NotImplementedError
        If `belt` is not `None`.
    TypeError
        If quantity arguments do not satisfy `ranged_quantity_input`.
    ValueError
        Propagated from single-satellite TLE construction for invalid TLE field
        encoding or identifier overflows.

    Notes
    -----
    1) Plane RAAN distribution:
        `RAAN_min + plane_idx * (RAAN_max - RAAN_min) / plane_count`.

    2) In-plane anomaly distribution:
        `sat_idx * (360 / num_sats_per_plane) + plane_offset`.
        If `adjacent_plane_offset=True`, odd-numbered planes (0-based index
        parity) receive `plane_offset = 0.5 * (360 / num_sats_per_plane)`.

    3) Performance behavior:
        Belt-level invariants (epoch string, mean motion, formatted motion
        derivatives) are computed once and reused for all satellites.
        This avoids repeated decorator and time-conversion overhead in large
        constellation generation.

    4) Counter behavior:
        NORAD/COSPAR assignment continues from the module-global counter across
        calls. Use `reset_tle_counter()` before forging if deterministic catalog
        numbering is required.
    """
    if belt is not None:
        raise NotImplementedError(
            "Belt-object input is not implemented; pass belt=None and use explicit belt parameters."
        )

    step_deg = 360.0 / num_sats_per_plane
    half_step_deg = step_deg / 2.0
    raan_min_deg = _to_float_deg(RAAN_min)
    raan_max_deg = _to_float_deg(RAAN_max)
    raan_deg_step = (raan_max_deg - raan_min_deg) / plane_count
    altitude_m = _to_float_m(altitude)
    inclination_deg_f = _to_float_deg(inclination_deg)
    argp_deg_f = _to_float_deg(argp_deg)
    year_short, epoch_str = _build_epoch_string(start_time)
    mean_motion_rev_day = _compute_mean_motion_rev_day(altitude_m, eccentricity=float(eccentricity))
    mm_dot_string = _format_mm_dot(mm_dot)
    mm_ddot_string = _format_tle_exp(mm_ddot)
    bstar_string = _format_tle_exp(bstar)

    tle_list: list[PyTle] = []
    for plane_idx in range(plane_count):
        raan_deg = raan_min_deg + (plane_idx * raan_deg_step)
        anomaly_offset_deg = half_step_deg if (adjacent_plane_offset and (plane_idx % 2 == 1)) else 0.0

        for satellite_idx in range(num_sats_per_plane):
            anomaly_deg = (satellite_idx * step_deg) + anomaly_offset_deg
            tle_list.append(
                _forge_tle_single_core(
                    sat_name=f"{belt_name}_Plane_{plane_idx+1}_Satellite_{satellite_idx+1}",
                    eccentricity=eccentricity,
                    inclination_deg=inclination_deg_f,
                    raan_deg=raan_deg,
                    argp_deg=argp_deg_f,
                    anomaly_deg=anomaly_deg,
                    year_short=year_short,
                    epoch_str=epoch_str,
                    mean_motion_rev_day=mean_motion_rev_day,
                    mm_dot_string=mm_dot_string,
                    mm_ddot_string=mm_ddot_string,
                    bstar_string=bstar_string,
                )
            )

    return np.asarray(tle_list, dtype=object)
