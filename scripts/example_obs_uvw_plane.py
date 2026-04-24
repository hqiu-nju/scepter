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

In addition to the ``.npz`` archive, the script always saves two PNG figures:

- ``<output_stem>_array_layout.png`` — antenna positions as East/North offsets
  (km) from the reference antenna.
- ``<output_stem>_radec_fov.png`` — satellite transit tracks plotted as
  tangent-plane offsets (ΔRA·cos Dec, ΔDec) from the pointing in the ICRS
  frame, within the requested FoV radius (default 10°).  Use
  ``--fov-radius-deg`` to change the FoV.

When ``--show-interactive-plots`` is enabled, the script also opens
matplotlib windows that show:

- pointing and satellite tracks in the reference antenna Az/El frame,
- final UV-plane previews for pointing-only and pointing-plus-first-satellite.

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
        --show-interactive-plots \
        --output demo_outputs/tracking_uvw.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, ICRS, SkyCoord

from scepter import uvw


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
_EARTH_RADIUS_KM = 6_371.0


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


def _flatten_uv_samples(uvw_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten UV samples and mirror them with Hermitian conjugate symmetry.

    Parameters
    ----------
    uvw_m : numpy.ndarray
        UVW coordinates with final axis ``(..., 3)`` where indices map to
        ``u, v, w`` in metres.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Flattened and mirrored ``(u, v)`` coordinates in metres.
    """
    uvw_array = np.asarray(uvw_m, dtype=np.float64)
    if uvw_array.ndim < 2 or uvw_array.shape[-1] != 3:
        raise ValueError(f"uvw_m must have a trailing axis of size 3, got {uvw_array.shape!r}.")

    uv_flat = np.reshape(uvw_array, (-1, 3))
    u_samples = uv_flat[:, 0]
    v_samples = uv_flat[:, 1]
    return (
        np.concatenate([u_samples, -u_samples]),
        np.concatenate([v_samples, -v_samples]),
    )


def _show_az_el_tracks_interactive(
    pointing_az_deg: np.ndarray,
    pointing_el_deg: np.ndarray,
    satellite_az_deg: np.ndarray,
    satellite_el_deg: np.ndarray,
    satellite_names: tuple[str, ...],
) -> None:
    """Display pointing and satellite tracks in an interactive Az/El frame."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not available. Skipping interactive Az/El display.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), num="Az/El Tracks")
    ax.plot(pointing_az_deg, pointing_el_deg, color="black", linewidth=2.0, label="Pointing")

    nsat = satellite_az_deg.shape[1] if satellite_az_deg.ndim == 2 else 0
    for sat_idx in range(nsat):
        label = satellite_names[sat_idx] if sat_idx < len(satellite_names) else f"sat{sat_idx}"
        ax.plot(
            satellite_az_deg[:, sat_idx],
            satellite_el_deg[:, sat_idx],
            linewidth=1.0,
            alpha=0.75,
            label=label,
        )

    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-5.0, 95.0)
    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    ax.set_title("Pointing and Satellite Tracks in Az/El")
    ax.grid(True, alpha=0.35, linestyle="--")
    if nsat <= 12:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def _show_uv_preview_interactive(
    pointing_uvw_m: np.ndarray,
    satellite_uvw_m: np.ndarray,
    freq_mhz: float,
    satellite_names: tuple[str, ...],
) -> None:
    """
    Show final UV-plane previews as two subplots.

    The first subplot is pointing-only and the second is pointing plus the
    first available satellite track.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not available. Skipping interactive UV preview display.")
        return

    if satellite_uvw_m.ndim != 4 or satellite_uvw_m.shape[2] == 0:
        print("No satellite UVW tracks available for final UV preview. Skipping second subplot.")
        return

    wavelength_m = SPEED_OF_LIGHT_M_PER_S / (float(freq_mhz) * 1e6)

    u_point, v_point = _flatten_uv_samples(pointing_uvw_m)
    u_sat, v_sat = _flatten_uv_samples(satellite_uvw_m[:, :, 0, :])

    u_point_lambda = u_point / wavelength_m
    v_point_lambda = v_point / wavelength_m
    u_combined_lambda = np.concatenate([u_point, u_sat]) / wavelength_m
    v_combined_lambda = np.concatenate([v_point, v_sat]) / wavelength_m

    sat_name = satellite_names[0] if len(satellite_names) > 0 else "sat0"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), num="Final UV Plane Preview")

    axes[0].scatter(u_point_lambda, v_point_lambda, s=1, alpha=0.35, c="tab:blue", edgecolors="none")
    axes[0].set_title("Pointing UV Plane")
    axes[0].set_xlabel("u [wavelengths]")
    axes[0].set_ylabel("v [wavelengths]")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].scatter(
        u_combined_lambda,
        v_combined_lambda,
        s=1,
        alpha=0.35,
        c="tab:green",
        edgecolors="none",
    )
    axes[1].set_title(f"Pointing + {sat_name}")
    axes[1].set_xlabel("u [wavelengths]")
    axes[1].set_ylabel("v [wavelengths]")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(f"UV Coverage Preview at {freq_mhz:.1f} MHz", fontsize=12)
    fig.tight_layout()
    plt.show()


def _save_array_layout_figure(
    geometry: "uvw.TelescopeArrayGeometry",
    ref_index: int,
    output_path: "Path",
) -> None:
    """
    Save an array-layout PNG showing antenna positions as ENU offsets.

    Parameters
    ----------
    geometry : scepter.uvw.TelescopeArrayGeometry
        Parsed array geometry.
    ref_index : int
        Reference antenna index placed at the ENU origin.
    output_path : pathlib.Path
        Destination file path (PNG).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not available. Skipping array layout figure.")
        return

    lon_deg = geometry.longitudes_deg
    lat_deg = geometry.latitudes_deg
    names = geometry.antenna_names

    lon_ref = lon_deg[ref_index]
    lat_ref = lat_deg[ref_index]
    lat_ref_rad = np.deg2rad(lat_ref)

    east_km = (lon_deg - lon_ref) * np.cos(lat_ref_rad) * _EARTH_RADIUS_KM
    north_km = (lat_deg - lat_ref) * _EARTH_RADIUS_KM

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(east_km, north_km, c="tab:blue", s=40, zorder=3, label="Antenna")
    ax.scatter(
        east_km[ref_index], north_km[ref_index],
        c="tab:red", s=100, marker="*", zorder=4,
        label=f"Ref: {names[ref_index]}",
    )
    if len(names) <= 64:
        for i, name in enumerate(names):
            ax.annotate(
                name, (east_km[i], north_km[i]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=6, alpha=0.8,
            )
    ax.set_xlabel("East offset [km]")
    ax.set_ylabel("North offset [km]")
    ax.set_title("Array Layout (ENU, reference at origin)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved array layout figure to {output_path}")


def _save_radec_fov_figure(
    ra_deg: float,
    dec_deg: float,
    satellite_ra_deg: np.ndarray,
    satellite_dec_deg: np.ndarray,
    satellite_names: tuple[str, ...],
    fov_radius_deg: float,
    output_path: "Path",
) -> None:
    """
    Save a pointing-centred RA/Dec FoV figure with satellite transit tracks.

    Satellite positions are projected onto a gnomonic tangent plane centred
    on the pointing using (ΔRA·cos Dec, ΔDec) offsets in degrees. Only
    epochs within ``fov_radius_deg`` of the pointing centre are rendered.
    The RA axis is displayed with West to the right (increasing RA to the
    left) following the standard radio-astronomy sky-map convention.

    Parameters
    ----------
    ra_deg : float
        Pointing right ascension in degrees (ICRS).
    dec_deg : float
        Pointing declination in degrees (ICRS).
    satellite_ra_deg : numpy.ndarray, shape (T, N_sat)
        Satellite ICRS right ascension tracks in degrees.
    satellite_dec_deg : numpy.ndarray, shape (T, N_sat)
        Satellite ICRS declination tracks in degrees.
    satellite_names : tuple of str
        Satellite identifiers.
    fov_radius_deg : float
        Half-width of the field of view in degrees.
    output_path : pathlib.Path
        Destination file path (PNG).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("Matplotlib is not available. Skipping RA/Dec FoV figure.")
        return

    sat_ra = np.asarray(satellite_ra_deg, dtype=np.float64)   # (T, N_sat)
    sat_dec = np.asarray(satellite_dec_deg, dtype=np.float64) # (T, N_sat)
    dec_ref_rad = np.deg2rad(dec_deg)
    nsat = sat_ra.shape[1] if sat_ra.ndim == 2 else 0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(0.0, 0.0, "+", color="black", markersize=14, markeredgewidth=2,
            zorder=5, label="Pointing")
    ax.add_patch(
        Circle((0, 0), fov_radius_deg, fill=False, linestyle="--",
               color="gray", linewidth=1.0, zorder=2)
    )

    for sat_idx in range(nsat):
        ra_s = sat_ra[:, sat_idx]
        dec_s = sat_dec[:, sat_idx]
        dec_mid_rad = np.deg2rad(0.5 * (dec_deg + dec_s))
        dra_raw = (ra_s - ra_deg + 180.0) % 360.0 - 180.0
        dra_eff = dra_raw * np.cos(dec_mid_rad)
        ddec_eff = dec_s - dec_deg
        mask = np.hypot(dra_eff, ddec_eff) <= fov_radius_deg
        if not np.any(mask):
            continue
        label = satellite_names[sat_idx] if sat_idx < len(satellite_names) else f"sat{sat_idx}"
        (line,) = ax.plot(
            dra_eff[mask], ddec_eff[mask],
            linewidth=0.8, alpha=0.75,
            label=label if nsat <= 12 else None,
        )
        idx_first = int(np.argmax(mask))
        ax.plot(dra_eff[idx_first], ddec_eff[idx_first], "o",
                markersize=3, alpha=0.6, color=line.get_color())

    ax.set_xlim(fov_radius_deg * 1.08, -fov_radius_deg * 1.08)  # RA increases to the left
    ax.set_ylim(-fov_radius_deg * 1.08, fov_radius_deg * 1.08)
    ax.set_xlabel("ΔRA·cos(Dec) [deg]  ← East")
    ax.set_ylabel("ΔDec [deg]")
    ax.set_title(
        f"Satellite Transit Tracks — {fov_radius_deg:.1f}° RA/Dec FoV\n"
        f"Pointing: RA {ra_deg:.3f}°, Dec {dec_deg:.3f}° (ICRS)"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35, linestyle="--")
    if nsat <= 12:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved RA/Dec FoV figure to {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create UVW tracks for a fixed RA/Dec pointing and for all "
            "satellites in one or more ASCII TLE files."
        )
    )
    parser.add_argument('-a',"--array-file", required=True, help="Path to the antenna coordinate file.")
    parser.add_argument(
        "-t","--tle-file",
        dest="tle_files",
        action="append",
        required=True,
        help="Path to an ASCII TLE file. Repeat to load multiple files.",
    )
    parser.add_argument("--ra",dest="ra_deg", type=float, required=True, help="Tracking right ascension in degrees.")
    parser.add_argument("--dec", dest="dec_deg", type=float, required=True, help="Tracking declination in degrees.")
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
    parser.add_argument(
        "--fov-radius-deg",
        type=float,
        default=10.0,
        help="Field-of-view radius in degrees for the saved Az/El FoV figure. Default: 10.",
    )
    parser.add_argument(
        "--show-interactive-plots",
        action="store_true",
        help="Display interactive Az/El and UV preview matplotlib windows.",
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

    satellite_names = result.satellite_names
    satellite_uvw_m = np.asarray(result.satellite_uvw_m, dtype=np.float64)
    satellite_hour_angles_rad = np.asarray(result.satellite_hour_angles_rad, dtype=np.float64)
    satellite_ra_deg = np.asarray(result.satellite_ra_deg, dtype=np.float64)
    satellite_dec_deg = np.asarray(result.satellite_dec_deg, dtype=np.float64)
    satellite_separation_deg = np.asarray(result.satellite_separation_deg, dtype=np.float64)

    geometry = uvw.load_telescope_array_file(
        args.array_file,
        altitude_unit=u.Unit(args.altitude_unit),
    )
    ref_location = geometry.earth_locations[args.ref_index]

    pointing_icrs = SkyCoord(
        ra=np.full(result.obs_times.shape, args.ra_deg, dtype=np.float64) * u.deg,
        dec=np.full(result.obs_times.shape, args.dec_deg, dtype=np.float64) * u.deg,
        frame=ICRS(),
    )
    pointing_altaz = pointing_icrs.transform_to(
        AltAz(obstime=result.obs_times, location=ref_location)
    )

    sat_icrs = SkyCoord(
        ra=satellite_ra_deg * u.deg,
        dec=satellite_dec_deg * u.deg,
        frame=ICRS(),
    )
    sat_altaz = sat_icrs.transform_to(
        AltAz(obstime=result.obs_times[:, np.newaxis], location=ref_location)
    )

    if args.show_interactive_plots:
        print("Showing interactive Az/El track window...")
        _show_az_el_tracks_interactive(
            np.asarray(pointing_altaz.az.deg, dtype=np.float64),
            np.asarray(pointing_altaz.alt.deg, dtype=np.float64),
            np.asarray(sat_altaz.az.deg, dtype=np.float64),
            np.asarray(sat_altaz.alt.deg, dtype=np.float64),
            satellite_names,
        )

    quicklook_vis = _build_unit_point_source_vis(result.pointing_uvw_m)

    output_path = Path(args.output)
    np.savez(
        output_path,
        mjds=result.mjds,
        antenna_names=np.asarray(result.antenna_names, dtype=str),
        satellite_names=np.asarray(satellite_names, dtype=str),
        freq_mhz=np.float64(args.freq_mhz),
        uvw=result.pointing_uvw_m,
        vis=quicklook_vis,
        vis_model=np.asarray("unit_point_source_phase_centre"),
        pointing_uvw_m=result.pointing_uvw_m,
        pointing_hour_angles_rad=result.pointing_hour_angles_rad,
        satellite_uvw_m=satellite_uvw_m,
        satellite_hour_angles_rad=satellite_hour_angles_rad,
        satellite_ra_deg=satellite_ra_deg,
        satellite_dec_deg=satellite_dec_deg,
        satellite_separation_deg=satellite_separation_deg,
    )

    print(f"Saved UVW products to {output_path}")
    print("Saved quick-imaging compatibility keys: uvw, vis, freq_mhz")
    print(f"Pointing UVW shape     : {result.pointing_uvw_m.shape}")
    print(f"Quick-look vis shape   : {quicklook_vis.shape}")
    print(f"Satellite UVW shape    : {satellite_uvw_m.shape}")
    print(f"Satellite RA/Dec shape : {satellite_ra_deg.shape}")
    print(f"Antenna count          : {len(result.antenna_names)}")
    print(f"Satellite count        : {len(satellite_names)}")

    stem = output_path.stem
    out_dir = output_path.parent
    _save_array_layout_figure(
        geometry,
        args.ref_index,
        out_dir / f"{stem}_array_layout.png",
    )
    _save_radec_fov_figure(
        args.ra_deg,
        args.dec_deg,
        satellite_ra_deg,
        satellite_dec_deg,
        satellite_names,
        args.fov_radius_deg,
        out_dir / f"{stem}_radec_fov.png",
    )

    if args.show_interactive_plots:
        print("Showing final interactive UV preview window...")
        _show_uv_preview_interactive(
            result.pointing_uvw_m,
            satellite_uvw_m,
            args.freq_mhz,
            satellite_names,
        )


if __name__ == "__main__":
    main()
