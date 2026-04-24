#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
image_filtered_uvw.py - Dirty-image generation from filtered UVW NPZ products.

This script reads UVW arrays from an NPZ file (for example the
``--output-filtered-npz`` product of ``scripts/plot_uv_plane.py``), builds a
synthetic visibility model from configurable source presets, and forms dirty
images via direct Fourier inversion.

Key features
------------
- Multi-snapshot imaging from one filtered UVW dataset.
- Multiple source presets (phase-centre, off-axis, double, custom list).
- User-defined phase centre shift, either directly in tangent-plane offsets or
  from sky coordinates (RA/Dec).

Examples
--------
Single dirty image with a centred point source:

    conda activate scepter-dev
    python scripts/image_filtered_uvw.py \
        --input scripts/filtered_uvw.npz

Multi-snapshot imaging with two sources and a phase-centre shift:

    python scripts/image_filtered_uvw.py \
        --input scripts/filtered_uvw.npz \
        --uvw-key pointing_uvw_m \
        --snapshots 4 \
        --source-preset double \
        --offset-l-arcmin 6 \
        --offset-m-arcmin -3 \
        --secondary-flux-jy 0.4 \
        --phase-center-shift-l-arcmin 2.0 \
        --phase-center-shift-m-arcmin -1.0 \
        --output-dir scripts/uv_plots

Custom source list with absolute phase-centre coordinates:

    python scripts/image_filtered_uvw.py \
        --input scripts/filtered_uvw.npz \
        --source-preset custom \
        --source "0,0,1.0" \
        --source "5,-2,0.3" \
        --data-phase-center-ra-deg 83.633 \
        --data-phase-center-dec-deg 22.014 \
        --phase-center-ra-deg 83.650 \
        --phase-center-dec-deg 22.100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import os

import numpy as np

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "scepter-mpl-cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
ARC_MIN_TO_RAD = np.pi / (180.0 * 60.0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create dirty images from filtered UVW NPZ arrays."
    )
    parser.add_argument("-i", "--input", required=True, help="Input NPZ path.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./uv_plots",
        help="Directory where images are written. Default: ./uv_plots",
    )
    parser.add_argument(
        "--uvw-key",
        default="pointing_uvw_m",
        choices=("pointing_uvw_m", "satellite_uvw_m"),
        help="UVW dataset key in NPZ. Default: pointing_uvw_m",
    )
    parser.add_argument(
        "--satellite-index",
        type=int,
        default=1,
        help="Satellite index used when --uvw-key satellite_uvw_m. Default: 0",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=None,
        help="Override observing frequency in MHz. Defaults to NPZ freq_mhz.",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=1,
        help="Number of equal-size time snapshots to image. Default: 1",
    )
    parser.add_argument(
        "--npix",
        type=int,
        default=256,
        help="Image size in pixels per axis. Default: 256",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=4.0,
        help="Image field of view width in degrees. Default: 4",
    )
    parser.add_argument(
        "--dft-chunk",
        type=int,
        default=20000,
        help="Sample chunk size used in direct Fourier inversion. Default: 20000",
    )

    parser.add_argument(
        "--source-preset",
        choices=("phase-center", "off-axis", "double", "custom"),
        default="phase-center",
        help=(
            "Synthetic source model preset. Use 'custom' with repeated --source "
            "entries as 'l_arcmin,m_arcmin,flux_jy'."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Custom source entry: 'l_arcmin,m_arcmin,flux_jy'. Repeat as needed.",
    )
    parser.add_argument(
        "--primary-flux-jy",
        type=float,
        default=1.0,
        help="Primary source flux for presets in Jy. Default: 1.0",
    )
    parser.add_argument(
        "--secondary-flux-jy",
        type=float,
        default=0.5,
        help="Secondary source flux for 'double' preset in Jy. Default: 0.5",
    )
    parser.add_argument(
        "--offset-l-arcmin",
        type=float,
        default=6.0,
        help="Source offset in l for off-axis/double presets (arcmin). Default: 6",
    )
    parser.add_argument(
        "--offset-m-arcmin",
        type=float,
        default=0.0,
        help="Source offset in m for off-axis/double presets (arcmin). Default: 0",
    )

    parser.add_argument(
        "--phase-center-shift-l-arcmin",
        type=float,
        default=0.0,
        help="Manual phase-centre shift in l (arcmin). Applied before imaging.",
    )
    parser.add_argument(
        "--phase-center-shift-m-arcmin",
        type=float,
        default=0.0,
        help="Manual phase-centre shift in m (arcmin). Applied before imaging.",
    )
    parser.add_argument(
        "--data-phase-center-ra-deg",
        type=float,
        default=None,
        help="RA of original dataset phase centre in degrees.",
    )
    parser.add_argument(
        "--data-phase-center-dec-deg",
        type=float,
        default=None,
        help="Dec of original dataset phase centre in degrees.",
    )
    parser.add_argument(
        "--phase-center-ra-deg",
        type=float,
        default=None,
        help="Target RA phase centre for imaging in degrees.",
    )
    parser.add_argument(
        "--phase-center-dec-deg",
        type=float,
        default=None,
        help="Target Dec phase centre for imaging in degrees.",
    )
    parser.add_argument(
        "--save-psf",
        action="store_true",
        help="Also save a PSF (dirty beam) image for each snapshot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI. Default: 150",
    )
    return parser


def _wrap_delta_ra_deg(delta_deg: float) -> float:
    """Wrap RA delta to [-180, 180) degrees."""
    return ((delta_deg + 180.0) % 360.0) - 180.0


def _parse_custom_sources(entries: list[str] | None) -> list[tuple[float, float, float]]:
    """Parse repeated --source entries into (l_rad, m_rad, flux_jy)."""
    if not entries:
        raise ValueError("--source-preset custom requires at least one --source entry.")

    parsed: list[tuple[float, float, float]] = []
    for item in entries:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --source '{item}'. Expected 'l_arcmin,m_arcmin,flux_jy'."
            )
        l_arcmin = float(parts[0])
        m_arcmin = float(parts[1])
        flux_jy = float(parts[2])
        parsed.append((l_arcmin * ARC_MIN_TO_RAD, m_arcmin * ARC_MIN_TO_RAD, flux_jy))
    return parsed


def _build_source_components(args: argparse.Namespace) -> list[tuple[float, float, float]]:
    """Create source components as (l_rad, m_rad, flux_jy)."""
    if args.source_preset == "phase-center":
        return [(0.0, 0.0, float(args.primary_flux_jy))]
    if args.source_preset == "off-axis":
        return [
            (
                float(args.offset_l_arcmin) * ARC_MIN_TO_RAD,
                float(args.offset_m_arcmin) * ARC_MIN_TO_RAD,
                float(args.primary_flux_jy),
            )
        ]
    if args.source_preset == "double":
        return [
            (0.0, 0.0, float(args.primary_flux_jy)),
            (
                float(args.offset_l_arcmin) * ARC_MIN_TO_RAD,
                float(args.offset_m_arcmin) * ARC_MIN_TO_RAD,
                float(args.secondary_flux_jy),
            ),
        ]
    return _parse_custom_sources(args.source)


def _extract_track_uvw(data: "np.lib.npyio.NpzFile", uvw_key: str, satellite_index: int) -> np.ndarray:
    """
    Extract UVW as shape (N_ant, T, 3) for imaging.

    Supports:
    - pointing_uvw_m: (N_ant, T, 3)
    - satellite_uvw_m: (N_ant, T, N_sat, 3), (N_sat, N_ant, T, 3),
      (N_sat, T, 3), (T, N_sat, 3)
    """
    if uvw_key not in data:
        raise KeyError(f"Input NPZ does not contain '{uvw_key}'.")

    uvw = np.asarray(data[uvw_key], dtype=np.float64)
    if uvw_key == "pointing_uvw_m":
        if uvw.ndim != 3 or uvw.shape[-1] != 3:
            raise ValueError(
                f"pointing_uvw_m must have shape (N_ant, T, 3), got {uvw.shape!r}."
            )
        return uvw

    if uvw.ndim == 4 and uvw.shape[-1] == 3:
        # Prefer (N_ant, T, N_sat, 3), then (N_sat, N_ant, T, 3).
        if uvw.shape[0] > 1 and uvw.shape[1] > 1:
            if satellite_index < 0 or satellite_index >= uvw.shape[2]:
                raise IndexError(
                    f"satellite-index {satellite_index} out of range [0, {uvw.shape[2] - 1}]."
                )
            return uvw[:, :, satellite_index, :]

        if satellite_index < 0 or satellite_index >= uvw.shape[0]:
            raise IndexError(
                f"satellite-index {satellite_index} out of range [0, {uvw.shape[0] - 1}]."
            )
        return uvw[satellite_index, ...]

    if uvw.ndim == 3 and uvw.shape[-1] == 3:
        # (N_sat, T, 3) or (T, N_sat, 3). Convert to pseudo (N_ant=2, T, 3)
        # by adding a zero reference antenna so baseline extraction still works.
        if satellite_index < 0:
            raise IndexError("satellite-index must be >= 0.")
        if uvw.shape[0] > uvw.shape[1]:
            if satellite_index >= uvw.shape[1]:
                raise IndexError(
                    f"satellite-index {satellite_index} out of range [0, {uvw.shape[1] - 1}]."
                )
            sat_track = uvw[:, satellite_index, :]
        else:
            if satellite_index >= uvw.shape[0]:
                raise IndexError(
                    f"satellite-index {satellite_index} out of range [0, {uvw.shape[0] - 1}]."
                )
            sat_track = uvw[satellite_index, :, :]
        ref = np.zeros_like(sat_track)
        return np.stack([ref, sat_track], axis=0)

    raise ValueError(f"Unsupported {uvw_key} shape: {uvw.shape!r}")


def _uvw_to_uv_lambda(track_uvw_m: np.ndarray, freq_mhz: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert (N_ant, T, 3) UVW to flattened baseline u,v in wavelengths."""
    if track_uvw_m.ndim != 3 or track_uvw_m.shape[-1] != 3:
        raise ValueError(f"Expected UVW shape (N_ant, T, 3), got {track_uvw_m.shape!r}.")
    if track_uvw_m.shape[0] < 2:
        raise ValueError("Need at least two antennas (or one plus reference) to form baselines.")

    wavelength_m = SPEED_OF_LIGHT_M_PER_S / (float(freq_mhz) * 1e6)
    baselines = track_uvw_m[1:, :, :2]
    uv = np.reshape(baselines, (-1, 2))
    u = uv[:, 0] / wavelength_m
    v = uv[:, 1] / wavelength_m
    return u, v


def _build_model_vis(
    u_lambda: np.ndarray,
    v_lambda: np.ndarray,
    source_components: list[tuple[float, float, float]],
) -> np.ndarray:
    """Build synthetic complex visibilities for point-source components."""
    vis = np.zeros_like(u_lambda, dtype=np.complex128)
    for l_rad, m_rad, flux_jy in source_components:
        vis += flux_jy * np.exp(-2j * np.pi * (u_lambda * l_rad + v_lambda * m_rad))
    return vis


def _apply_phase_center_shift(
    vis: np.ndarray,
    u_lambda: np.ndarray,
    v_lambda: np.ndarray,
    shift_l_rad: float,
    shift_m_rad: float,
) -> np.ndarray:
    """Phase-rotate visibilities to a new image phase centre."""
    if shift_l_rad == 0.0 and shift_m_rad == 0.0:
        return vis
    return vis * np.exp(-2j * np.pi * (u_lambda * shift_l_rad + v_lambda * shift_m_rad))


def _phase_center_shift_from_sky(args: argparse.Namespace) -> tuple[float, float]:
    """Return total (l,m) shift in radians from manual and optional RA/Dec inputs."""
    shift_l = float(args.phase_center_shift_l_arcmin) * ARC_MIN_TO_RAD
    shift_m = float(args.phase_center_shift_m_arcmin) * ARC_MIN_TO_RAD

    have_data = args.data_phase_center_ra_deg is not None and args.data_phase_center_dec_deg is not None
    have_target = args.phase_center_ra_deg is not None and args.phase_center_dec_deg is not None

    if have_data and have_target:
        dra_deg = _wrap_delta_ra_deg(
            float(args.phase_center_ra_deg) - float(args.data_phase_center_ra_deg)
        )
        ddec_deg = float(args.phase_center_dec_deg) - float(args.data_phase_center_dec_deg)
        dec0_rad = np.deg2rad(float(args.data_phase_center_dec_deg))
        shift_l += np.deg2rad(dra_deg) * np.cos(dec0_rad)
        shift_m += np.deg2rad(ddec_deg)

    return shift_l, shift_m


def _dirty_image_direct_dft(
    u_lambda: np.ndarray,
    v_lambda: np.ndarray,
    vis: np.ndarray,
    npix: int,
    fov_deg: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form dirty image with direct DFT on an (l,m) grid."""
    fov_rad = np.deg2rad(float(fov_deg))
    axis = np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, int(npix), dtype=np.float64)
    l_grid, m_grid = np.meshgrid(axis, axis, indexing="xy")

    lm_flat = np.column_stack((l_grid.ravel(), m_grid.ravel()))
    image_flat = np.zeros(lm_flat.shape[0], dtype=np.complex128)

    n = int(u_lambda.size)
    if n == 0:
        raise ValueError("No UV samples available for imaging.")

    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        u_chunk = u_lambda[start:stop][:, np.newaxis]
        v_chunk = v_lambda[start:stop][:, np.newaxis]
        vis_chunk = vis[start:stop][:, np.newaxis]

        phase = 2j * np.pi * (u_chunk * lm_flat[:, 0] + v_chunk * lm_flat[:, 1])
        image_flat += np.sum(vis_chunk * np.exp(phase), axis=0)

    image = np.real(image_flat).reshape(int(npix), int(npix)) / float(n)
    return image, axis, axis


def _save_image(
    image: np.ndarray,
    l_axis_rad: np.ndarray,
    m_axis_rad: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Save a dirty-image figure in arcminutes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = [
        np.rad2deg(l_axis_rad[0]) * 60.0,
        np.rad2deg(l_axis_rad[-1]) * 60.0,
        np.rad2deg(m_axis_rad[0]) * 60.0,
        np.rad2deg(m_axis_rad[-1]) * 60.0,
    ]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Dirty intensity [arbitrary]", fontsize=10)
    ax.set_xlabel("l [arcmin]")
    ax.set_ylabel("m [arcmin]")
    ax.set_title(title)
    ax.grid(False)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _snapshot_ranges(n_time: int, snapshots: int) -> list[tuple[int, int]]:
    """Split time samples into snapshot index ranges [start, stop)."""
    if snapshots <= 0:
        raise ValueError("--snapshots must be >= 1")
    if snapshots > n_time:
        snapshots = n_time

    edges = np.linspace(0, n_time, snapshots + 1, dtype=int)
    ranges: list[tuple[int, int]] = []
    for i in range(snapshots):
        start = int(edges[i])
        stop = int(edges[i + 1])
        if stop > start:
            ranges.append((start, stop))
    return ranges


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {input_path}")

    if args.npix <= 0:
        raise ValueError("--npix must be > 0")
    if args.fov_deg <= 0.0:
        raise ValueError("--fov-deg must be > 0")
    if args.dft_chunk <= 0:
        raise ValueError("--dft-chunk must be > 0")

    with np.load(input_path) as data:
        freq_mhz = float(args.freq_mhz) if args.freq_mhz is not None else float(data["freq_mhz"])
        track_uvw = _extract_track_uvw(data, args.uvw_key, args.satellite_index)

    n_ant, n_time, _ = track_uvw.shape
    print(f"Loaded {args.uvw_key} for imaging with shape: {track_uvw.shape}")
    print(f"Frequency: {freq_mhz:.3f} MHz")
    print(f"Using {n_ant - 1} baseline(s) against reference antenna")

    source_components = _build_source_components(args)
    print(f"Source preset: {args.source_preset}")
    print(f"Source components: {len(source_components)}")

    shift_l_rad, shift_m_rad = _phase_center_shift_from_sky(args)
    shift_l_arcmin = np.rad2deg(shift_l_rad) * 60.0
    shift_m_arcmin = np.rad2deg(shift_m_rad) * 60.0
    if shift_l_rad != 0.0 or shift_m_rad != 0.0:
        print(
            "Applied phase-center shift: "
            f"l={shift_l_arcmin:.4f} arcmin, m={shift_m_arcmin:.4f} arcmin"
        )

    ranges = _snapshot_ranges(n_time=n_time, snapshots=int(args.snapshots))
    print(f"Imaging {len(ranges)} snapshot(s).")

    for idx, (start, stop) in enumerate(ranges, start=1):
        uvw_slice = track_uvw[:, start:stop, :]
        u_lambda, v_lambda = _uvw_to_uv_lambda(uvw_slice, freq_mhz=freq_mhz)
        vis = _build_model_vis(u_lambda, v_lambda, source_components)
        vis_shifted = _apply_phase_center_shift(
            vis,
            u_lambda,
            v_lambda,
            shift_l_rad=shift_l_rad,
            shift_m_rad=shift_m_rad,
        )

        image, l_axis_rad, m_axis_rad = _dirty_image_direct_dft(
            u_lambda,
            v_lambda,
            vis_shifted,
            npix=int(args.npix),
            fov_deg=float(args.fov_deg),
            chunk_size=int(args.dft_chunk),
        )

        title = (
            f"Dirty Image | snapshot {idx}/{len(ranges)} | "
            f"t=[{start}:{stop}) | {freq_mhz:.1f} MHz"
        )
        out_path = output_dir / f"dirty_image_snap{idx:03d}.png"
        _save_image(image, l_axis_rad, m_axis_rad, title, out_path, dpi=int(args.dpi))
        print(f"Saved {out_path}")

        if args.save_psf:
            psf, _, _ = _dirty_image_direct_dft(
                u_lambda,
                v_lambda,
                np.ones_like(vis_shifted),
                npix=int(args.npix),
                fov_deg=float(args.fov_deg),
                chunk_size=int(args.dft_chunk),
            )
            psf_title = (
                f"Dirty Beam (PSF) | snapshot {idx}/{len(ranges)} | "
                f"t=[{start}:{stop})"
            )
            psf_path = output_dir / f"dirty_beam_snap{idx:03d}.png"
            _save_image(psf, l_axis_rad, m_axis_rad, psf_title, psf_path, dpi=int(args.dpi))
            print(f"Saved {psf_path}")

    print(f"All images written to: {output_dir}")


if __name__ == "__main__":
    main()