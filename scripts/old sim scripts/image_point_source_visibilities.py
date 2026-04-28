#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
image_point_source_visibilities.py - Image point-source simulator visibilities.

Create a quick-look dirty image from ``scripts/point_source_visibility_simulator.py``
output.  The script reads the simulator's ``uvw`` and ``vis`` arrays, grids the
UV samples, forms a dirty image and dirty beam, and overlays source catalog
positions when the archive contains ``source_ra_deg`` / ``source_dec_deg`` plus
``phase_ra_deg`` / ``phase_dec_deg``.

Example
-------
    conda activate scepter-dev
    python scripts/image_point_source_visibilities.py

Notes
-----
This is a fast diagnostic imager.  It uses nearest-neighbour UV gridding and
does not deconvolve the dirty beam, apply primary-beam terms, or correct the
``w`` term.  Use it to verify that generated visibilities and source geometry
are sensible before moving to a fuller imaging workflow.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "scepter-mpl-cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quick_dirty_image_from_uvw import (  # noqa: E402
    _load_array,
    _load_visibilities,
    flatten_uvw_visibility_samples,
    make_dirty_image,
    resolve_frequency_hz,
)


DEFAULT_INPUT_NPZ = Path("point_source_vis.npz")
DEFAULT_OUTPUT_PNG = Path("point_source_dirty_image.png")
DEFAULT_NPIX = 512
DEFAULT_FOV_DEG = 5.0


def _load_optional_string_array(
    dataset: np.lib.npyio.NpzFile,
    key: str,
) -> tuple[str, ...] | None:
    if key not in dataset.files:
        return None
    values = np.asarray(dataset[key])
    return tuple(str(value) for value in values.reshape(-1))


def _wrap_delta_ra_rad(delta_ra_rad: np.ndarray) -> np.ndarray:
    return (delta_ra_rad + np.pi) % (2.0 * np.pi) - np.pi


def source_offsets_lm_deg(
    source_ra_deg: np.ndarray,
    source_dec_deg: np.ndarray,
    phase_ra_deg: float,
    phase_dec_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project RA/Dec source positions onto phase-centred tangent-plane offsets.

    Parameters
    ----------
    source_ra_deg, source_dec_deg : numpy.ndarray
        ICRS source coordinates in degrees. Arrays must be broadcastable.
    phase_ra_deg, phase_dec_deg : float
        ICRS phase-centre coordinates in degrees.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(l_deg, m_deg)`` tangent-plane direction-cosine offsets converted to
        degrees. Positive ``l`` is toward increasing RA and positive ``m`` is
        toward increasing Dec.

    Raises
    ------
    ValueError
        If the coordinates cannot be broadcast or if a source lies at or behind
        the tangent-plane horizon relative to the phase centre.
    """
    ra_rad, dec_rad = np.broadcast_arrays(
        np.deg2rad(np.asarray(source_ra_deg, dtype=np.float64)),
        np.deg2rad(np.asarray(source_dec_deg, dtype=np.float64)),
    )
    phase_ra_rad = np.deg2rad(float(phase_ra_deg))
    phase_dec_rad = np.deg2rad(float(phase_dec_deg))
    delta_ra = _wrap_delta_ra_rad(ra_rad - phase_ra_rad)

    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec0 = np.sin(phase_dec_rad)
    cos_dec0 = np.cos(phase_dec_rad)
    cos_delta_ra = np.cos(delta_ra)

    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_delta_ra
    if np.any(cos_c <= 0.0):
        raise ValueError(
            "At least one source is outside the visible tangent-plane "
            "hemisphere for the selected phase centre."
        )

    l_rad = cos_dec * np.sin(delta_ra) / cos_c
    m_rad = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_delta_ra) / cos_c
    return np.rad2deg(l_rad), np.rad2deg(m_rad)


def _load_source_overlay(
    dataset: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None:
    required = {"source_ra_deg", "source_dec_deg", "phase_ra_deg", "phase_dec_deg"}
    if not required.issubset(dataset.files):
        return None

    source_ra_deg = np.asarray(dataset["source_ra_deg"], dtype=np.float64)
    source_dec_deg = np.asarray(dataset["source_dec_deg"], dtype=np.float64)
    phase_ra_deg = float(np.asarray(dataset["phase_ra_deg"]).squeeze())
    phase_dec_deg = float(np.asarray(dataset["phase_dec_deg"]).squeeze())
    source_names = _load_optional_string_array(dataset, "source_names")
    if source_names is None or len(source_names) != source_ra_deg.size:
        source_names = tuple(f"src{i}" for i in range(source_ra_deg.size))

    l_deg, m_deg = source_offsets_lm_deg(
        source_ra_deg,
        source_dec_deg,
        phase_ra_deg,
        phase_dec_deg,
    )
    return l_deg, m_deg, source_names


def save_point_source_image_figure(
    dirty_image: np.ndarray,
    dirty_beam: np.ndarray,
    output_path: Path,
    *,
    fov_deg: float,
    title: str,
    source_overlay: tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None = None,
) -> None:
    """
    Save a dirty-image / dirty-beam figure with optional source overlays.

    Parameters
    ----------
    dirty_image, dirty_beam : numpy.ndarray
        Two-dimensional image arrays returned by
        :func:`quick_dirty_image_from_uvw.make_dirty_image`.
    output_path : pathlib.Path
        Destination PNG path.
    fov_deg : float
        Total image field of view in degrees.
    title : str
        Dirty-image panel title.
    source_overlay : tuple or None, optional
        Optional ``(l_deg, m_deg, source_names)`` tuple to overlay on the dirty
        image panel.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent_deg = (-0.5 * fov_deg, 0.5 * fov_deg, -0.5 * fov_deg, 0.5 * fov_deg)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)

    image_artist = axes[0].imshow(
        dirty_image,
        origin="lower",
        extent=extent_deg,
        cmap="inferno",
        interpolation="nearest",
    )
    axes[0].set_title(title)
    axes[0].set_xlabel("l offset [deg]")
    axes[0].set_ylabel("m offset [deg]")
    fig.colorbar(image_artist, ax=axes[0], shrink=0.86, label="Dirty image")

    if source_overlay is not None:
        l_deg, m_deg, source_names = source_overlay
        in_view = (
            (np.abs(l_deg) <= 0.5 * fov_deg)
            & (np.abs(m_deg) <= 0.5 * fov_deg)
        )
        axes[0].scatter(
            l_deg[in_view],
            m_deg[in_view],
            marker="+",
            s=90,
            linewidths=1.8,
            color="white",
            label="catalog sources",
        )
        visible_names = np.asarray(source_names)[in_view]
        for l_val, m_val, name in zip(l_deg[in_view], m_deg[in_view], visible_names):
            axes[0].annotate(
                str(name),
                (float(l_val), float(m_val)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="white",
            )
        if np.any(in_view):
            axes[0].legend(loc="upper right", fontsize=8)

    beam_artist = axes[1].imshow(
        dirty_beam,
        origin="lower",
        extent=extent_deg,
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1].set_title("Dirty beam")
    axes[1].set_xlabel("l offset [deg]")
    axes[1].set_ylabel("m offset [deg]")
    fig.colorbar(beam_artist, ax=axes[1], shrink=0.86, label="Beam response")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_image_arrays(
    output_path: Path,
    *,
    dirty_image: np.ndarray,
    dirty_beam: np.ndarray,
    fov_deg: float,
    frequency_hz: float,
    metadata: dict[str, float | int],
) -> Path:
    """
    Save dirty-image arrays and metadata to ``.npz``.

    Parameters
    ----------
    output_path : pathlib.Path
        Destination archive path.
    dirty_image, dirty_beam : numpy.ndarray
        Image arrays.
    fov_deg : float
        Total image field of view in degrees.
    frequency_hz : float
        Observing frequency in hertz.
    metadata : dict
        Imaging metadata returned by ``make_dirty_image``.

    Returns
    -------
    pathlib.Path
        Written archive path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_name=np.asarray("scepter_point_source_dirty_image_npz"),
        schema_version=np.asarray("1"),
        dirty_image=np.asarray(dirty_image, dtype=np.float64),
        dirty_beam=np.asarray(dirty_beam, dtype=np.float64),
        fov_deg=np.float64(fov_deg),
        freq_hz=np.float64(frequency_hz),
        wavelength_m=np.float64(metadata["wavelength_m"]),
        uv_cell_lambda=np.float64(metadata["uv_cell_lambda"]),
        gridded_sample_count=np.int64(metadata["gridded_sample_count"]),
        dropped_sample_count=np.int64(metadata["dropped_sample_count"]),
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Image visibilities generated by point_source_visibility_simulator.py."
    )
    parser.add_argument(
        "input_npz",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_NPZ,
        help=f"Point-source visibility .npz archive. Default: {DEFAULT_INPUT_NPZ}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help=f"Output PNG path. Default: {DEFAULT_OUTPUT_PNG}.",
    )
    parser.add_argument(
        "--image-npz",
        type=Path,
        default=None,
        help="Optional output .npz path for dirty_image and dirty_beam arrays.",
    )
    parser.add_argument(
        "--npix",
        type=int,
        default=DEFAULT_NPIX,
        help=f"Square image size in pixels. Default: {DEFAULT_NPIX}.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=DEFAULT_FOV_DEG,
        help=f"Total image field of view in degrees. Default: {DEFAULT_FOV_DEG:g}.",
    )
    parser.add_argument(
        "--weighting",
        choices=("natural", "uniform"),
        default="natural",
        help="UV weighting scheme. Default: %(default)s.",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=None,
        help="Frequency override in MHz. Defaults to freq_hz / freq_mhz in the archive.",
    )
    parser.add_argument("--uvw-key", default="uvw", help="UVW array key. Default: %(default)s.")
    parser.add_argument(
        "--vis-key",
        default="vis",
        help="Visibility array key. Default: %(default)s.",
    )
    parser.add_argument(
        "--no-source-overlay",
        action="store_true",
        help="Do not overlay catalog source positions even when metadata is present.",
    )
    parser.add_argument(
        "--no-hermitian-mirror",
        action="store_true",
        help="Do not add conjugate UV samples at (-u, -v).",
    )
    parser.add_argument(
        "--keep-zero-baseline",
        action="store_true",
        help="Keep samples with u=v=0 instead of dropping them.",
    )
    parser.add_argument("--title", default=None, help="Optional dirty-image panel title.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    with np.load(args.input_npz, allow_pickle=False) as dataset:
        uvw_m = _load_array(dataset, args.uvw_key)
        vis = _load_visibilities(
            dataset,
            vis_key=args.vis_key,
            vis_real_key=None,
            vis_imag_key=None,
        )
        frequency_hz = resolve_frequency_hz(dataset, args.freq_mhz)
        source_overlay = None if args.no_source_overlay else _load_source_overlay(dataset)

    uvw_samples_m, vis_samples = flatten_uvw_visibility_samples(
        uvw_m,
        vis,
        drop_zero_baseline=not args.keep_zero_baseline,
    )
    dirty_image, dirty_beam, metadata = make_dirty_image(
        uvw_samples_m,
        vis_samples,
        frequency_hz,
        npix=args.npix,
        fov_deg=args.fov_deg,
        weighting=args.weighting,
        hermitian_mirror=not args.no_hermitian_mirror,
    )

    title = args.title or f"Point-source dirty image: {args.input_npz.stem}"
    save_point_source_image_figure(
        dirty_image,
        dirty_beam,
        args.output,
        fov_deg=args.fov_deg,
        title=title,
        source_overlay=source_overlay,
    )
    if args.image_npz is not None:
        save_image_arrays(
            args.image_npz,
            dirty_image=dirty_image,
            dirty_beam=dirty_beam,
            fov_deg=args.fov_deg,
            frequency_hz=frequency_hz,
            metadata=metadata,
        )

    source_count = 0 if source_overlay is None else len(source_overlay[2])
    print(f"Loaded UVW shape             : {uvw_m.shape}")
    print(f"Loaded visibility shape      : {vis.shape}")
    print(f"Flattened sample count       : {uvw_samples_m.shape[0]}")
    print(f"Observing frequency          : {frequency_hz / 1.0e6:.6f} MHz")
    print(f"Gridded samples kept         : {int(metadata['gridded_sample_count'])}")
    print(f"Samples outside FFT grid     : {int(metadata['dropped_sample_count'])}")
    print(f"Catalog sources overlaid     : {source_count}")
    print(f"Saved point-source image to  : {args.output}")
    if args.image_npz is not None:
        print(f"Saved image arrays to        : {args.image_npz}")


if __name__ == "__main__":
    main()
