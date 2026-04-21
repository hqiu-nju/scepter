#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
example_obs_uvw_plane.py - Build UVW products for a tracked source plus TLEs.

This script is a thin CLI wrapper around :func:`scepter.uvw.build_tracking_uvw`.
It reads an antenna-array coordinate file, propagates one or more custom TLE
catalogs, and writes a ``.npz`` bundle containing:

- UVW tracks for the requested fixed RA/Dec pointing,
- a quick-look imaging compatibility bundle (``uvw``, ``vis``, ``freq_mhz``),
- per-satellite UVW tracks using each satellite's time-varying ICRS RA/Dec,
- the corresponding satellite RA/Dec and angular-separation time series.

The compatibility ``vis`` array is a synthetic unit-amplitude point-source
model at the phase centre. It is intended only as a convenience product so
that ``scripts/quick_dirty_image_from_uvw.py`` can form a dirty beam / centred
dirty-image preview directly from the saved example output.

Supported array-file formats
----------------------------
1. Headered CSV / TSV / whitespace text with longitude, latitude, and altitude
   columns. Optional name/id columns are preserved.
2. Headerless text where the first three columns are interpreted as
   ``lon_deg lat_deg alt_m`` and an optional fourth column is used as the name.

Example
-------
    conda activate scepter
    python scripts/example_obs_uvw_plane.py \
        --array-file demo_outputs/meerkat_array.csv \
        --tle-file demo_outputs/oneweb.tle \
        --ra-deg 83.633 \
        --dec-deg 22.014 \
        --output demo_outputs/tracking_uvw.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy import units as u

from scepter import uvw


def _build_unit_point_source_vis(pointing_uvw_m: np.ndarray) -> np.ndarray:
    """
    Build a simple complex visibility model aligned with the pointing UVW array.

    Parameters
    ----------
    pointing_uvw_m : numpy.ndarray, shape (N_ant, T, 3)
        UVW coordinates in metres for a fixed phase centre, with the reference
        antenna stored along axis 0.

    Returns
    -------
    numpy.ndarray, shape (N_ant - 1, T)
        Unit-amplitude complex visibilities for every non-reference baseline.

    Raises
    ------
    ValueError
        If *pointing_uvw_m* does not have shape ``(N_ant, T, 3)`` or if fewer
        than two antennas are available.

    Notes
    -----
    A unit point source at the phase centre has constant complex visibility
    ``V(u, v) = 1 + 0j`` on every measured baseline. The example script stores
    only the non-reference baselines so the output matches the default
    ``(N_ant - 1, T)`` visibility layout accepted by
    ``quick_dirty_image_from_uvw.py``.
    """
    uvw_array = np.asarray(pointing_uvw_m, dtype=np.float64)
    if uvw_array.ndim != 3 or uvw_array.shape[-1] != 3:
        raise ValueError(
            "pointing_uvw_m must have shape (N_ant, T, 3). "
            f"Got {uvw_array.shape!r}."
        )
    if uvw_array.shape[0] < 2:
        raise ValueError("At least two antennas are required to build visibilities.")
    return np.ones(uvw_array.shape[:2], dtype=np.complex128)[1:, :]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create UVW tracks for a fixed RA/Dec pointing and for all "
            "satellites in one or more ASCII TLE files."
        )
    )
    parser.add_argument("--array-file", required=True, help="Path to the antenna coordinate file.")
    parser.add_argument(
        "--tle-file",
        dest="tle_files",
        action="append",
        required=True,
        help="Path to an ASCII TLE file. Repeat to load multiple files.",
    )
    parser.add_argument("--ra-deg", type=float, required=True, help="Tracking right ascension in degrees.")
    parser.add_argument("--dec-deg", type=float, required=True, help="Tracking declination in degrees.")
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=1420.0,
        help="Receiver centre frequency in MHz. Saved to the output archive for quick imaging.",
    )
    parser.add_argument(
        "--output",
        default="tracking_uvw.npz",
        help="Output NPZ file for the UVW products.",
    )
    parser.add_argument(
        "--altitude-unit",
        default="m",
        help="Astropy unit string for the array-file altitude column. Default: m.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of observation epochs when generating the default time grid.",
    )
    parser.add_argument(
        "--cadence-hours",
        type=float,
        default=24.0,
        help="Epoch cadence in hours for the default time grid.",
    )
    parser.add_argument(
        "--trange-seconds",
        type=float,
        default=3600.0,
        help="Observation duration in seconds for the default time grid.",
    )
    parser.add_argument(
        "--tint-seconds",
        type=float,
        default=1.0,
        help="Integration step in seconds for the default time grid.",
    )
    parser.add_argument(
        "--elevation-limit-deg",
        type=float,
        default=None,
        help="Optional mean-elevation filter applied before building satellite UVW tracks.",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=0,
        help="Reference antenna index used as the UVW origin.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose propagation logging from obs.obs_sim.populate.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    freq = args.freq_mhz * u.MHz

    result = uvw.build_tracking_uvw(
        ra_deg=args.ra_deg,
        dec_deg=args.dec_deg,
        array_file=args.array_file,
        tle_files=args.tle_files,
        epochs=args.epochs,
        cadence=args.cadence_hours * u.hour,
        trange=args.trange_seconds * u.s,
        tint=args.tint_seconds * u.s,
        ref_index=args.ref_index,
        altitude_unit=u.Unit(args.altitude_unit),
        freq=freq,
        elevation_limit_deg=args.elevation_limit_deg,
        verbose=args.verbose,
    )
    quicklook_vis = _build_unit_point_source_vis(result.pointing_uvw_m)

    output_path = Path(args.output)
    np.savez(
        output_path,
        mjds=result.mjds,
        antenna_names=np.asarray(result.antenna_names, dtype=str),
        satellite_names=np.asarray(result.satellite_names, dtype=str),
        freq_mhz=np.float64(args.freq_mhz),
        uvw=result.pointing_uvw_m,
        vis=quicklook_vis,
        vis_model=np.asarray("unit_point_source_phase_centre"),
        pointing_uvw_m=result.pointing_uvw_m,
        pointing_hour_angles_rad=result.pointing_hour_angles_rad,
        satellite_uvw_m=result.satellite_uvw_m,
        satellite_hour_angles_rad=result.satellite_hour_angles_rad,
        satellite_ra_deg=result.satellite_ra_deg,
        satellite_dec_deg=result.satellite_dec_deg,
        satellite_separation_deg=result.satellite_separation_deg,
    )

    print(f"Saved UVW products to {output_path}")
    print("Saved quick-imaging compatibility keys: uvw, vis, freq_mhz")
    print(f"Pointing UVW shape     : {result.pointing_uvw_m.shape}")
    print(f"Quick-look vis shape   : {quicklook_vis.shape}")
    print(f"Satellite UVW shape    : {result.satellite_uvw_m.shape}")
    print(f"Satellite RA/Dec shape : {result.satellite_ra_deg.shape}")
    print(f"Antenna count          : {len(result.antenna_names)}")
    print(f"Satellite count        : {len(result.satellite_names)}")


if __name__ == "__main__":
    main()
