#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
quick_dirty_image_from_uvw.py - Quick-look dirty imaging from UVW + visibilities

Builds a fast preview image from interferometric visibility data using the UVW
array convention returned by ``scripts/example_obs_uvw_plane.py``.  The script
expects UVW coordinates in metres and complex visibilities aligned with those
samples, grids the visibilities onto a regular UV plane, and forms a dirty
image via an inverse FFT.

Usage
-----
    conda activate scepter-dev
    python scripts/quick_dirty_image_from_uvw.py uvw_vis_input.npz \\
        --freq-mhz 1420.0 \\
        --output quick_dirty_image.png

Input NPZ format
----------------
The input ``.npz`` file must contain:

- ``uvw`` (default key): UVW coordinates in metres with shape
  ``(N_ant, N_time, 3)``, ``(N_baseline, N_time, 3)``, or ``(N_vis, 3)``.
  This matches the output shape from ``obs_uvw_plane`` / ``scepter.uvw``.
- ``vis`` (default key): complex visibilities aligned with the UVW samples.
  Supported shapes are:

  - ``(N_ant - 1, N_time)`` for the reference-baseline layout produced by the
    UVW example script when antenna 0 is the reference.
  - ``(N_ant, N_time)`` when a placeholder row for antenna 0 is included.
  - ``(N_baseline, N_time)`` or ``(N_vis,)`` when UVW is already baseline-flat.

Optionally the file may also contain ``freq_hz`` or ``freq_mhz``.  If not,
pass ``--freq-mhz`` on the command line.

Notes
-----
- This is a quick-look imager, not a full synthesis-imaging pipeline.
- The script applies nearest-neighbour UV gridding and no deconvolution.
- The ``w`` term, primary beam, bandwidth smearing, and calibration effects are
  ignored.
- By default, Hermitian conjugate UV samples are added automatically because
  the UVW example script plots only one half of the UV plane and mirrors it for
  visualisation.
"""

from __future__ import annotations

import argparse
import os
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


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def _load_array(dataset: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    """Load one array from an ``.npz`` archive and raise a clear error if missing."""
    if key not in dataset.files:
        available = ", ".join(dataset.files) if dataset.files else "<empty>"
        raise KeyError(f"Missing array key {key!r}. Available keys: {available}.")
    return np.asarray(dataset[key])


def _load_visibilities(
    dataset: np.lib.npyio.NpzFile,
    vis_key: str,
    vis_real_key: str | None,
    vis_imag_key: str | None,
) -> np.ndarray:
    """Load complex visibilities from one complex array or real/imag pairs."""
    if vis_real_key is not None or vis_imag_key is not None:
        if vis_real_key is None or vis_imag_key is None:
            raise ValueError(
                "Provide both --vis-real-key and --vis-imag-key, or neither."
            )
        real_part = _load_array(dataset, vis_real_key)
        imag_part = _load_array(dataset, vis_imag_key)
        if real_part.shape != imag_part.shape:
            raise ValueError(
                "Real and imaginary visibility arrays must have the same shape, "
                f"got {real_part.shape!r} and {imag_part.shape!r}."
            )
        return np.asarray(real_part, dtype=np.float64) + 1j * np.asarray(
            imag_part, dtype=np.float64
        )

    vis = _load_array(dataset, vis_key)
    return np.asarray(vis, dtype=np.complex128)


def flatten_uvw_visibility_samples(
    uvw_m: np.ndarray,
    vis: np.ndarray,
    *,
    drop_zero_baseline: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align UVW coordinates with visibility samples and flatten them to 1-D lists.

    Parameters
    ----------
    uvw_m : numpy.ndarray
        UVW coordinates in metres with shape ``(..., 3)``.  The last axis must
        contain ``[u, v, w]`` components.
    vis : numpy.ndarray
        Complex visibilities aligned with the leading UVW axes.  For the
        reference-baseline convention used by the example script, ``vis`` may
        have shape ``(N_ant - 1, N_time)`` while ``uvw_m`` has shape
        ``(N_ant, N_time, 3)``; in that case the reference row
        ``uvw_m[0, ...] == 0`` is skipped automatically.
    drop_zero_baseline : bool, optional
        If ``True``, discard samples with ``u == 0`` and ``v == 0``.  This is
        useful for the example UVW layout because antenna 0 is a bookkeeping
        reference rather than a measured self-correlation baseline.

    Returns
    -------
    uvw_samples_m : numpy.ndarray, shape (N_sample, 3)
        Flattened UVW coordinates in metres.
    vis_samples : numpy.ndarray, shape (N_sample,)
        Flattened complex visibilities aligned with ``uvw_samples_m``.

    Raises
    ------
    ValueError
        If the UVW array does not end in a length-3 axis, if the visibility
        shape cannot be aligned with the UVW leading axes, or if no valid
        samples remain after filtering.

    Notes
    -----
    The function accepts both the example-script output shape
    ``(N_ant, N_time, 3)`` and already flattened UVW shapes such as
    ``(N_vis, 3)``.  All leading axes are flattened in row-major order after
    alignment so that the visibility ordering is preserved.
    """
    uvw_array = np.asarray(uvw_m)
    vis_array = np.asarray(vis, dtype=np.complex128)

    if uvw_array.ndim < 2 or uvw_array.shape[-1] != 3:
        raise ValueError(
            "UVW array must have shape (..., 3) with the last axis storing "
            f"[u, v, w]. Got {uvw_array.shape!r}."
        )

    uvw_for_vis = uvw_array
    uvw_leading_shape = uvw_array.shape[:-1]

    if vis_array.shape != uvw_leading_shape:
        if (
            uvw_array.ndim >= 3
            and uvw_array.shape[0] >= 2
            and vis_array.shape == (uvw_array.shape[0] - 1, *uvw_array.shape[1:-1])
        ):
            uvw_for_vis = uvw_array[1:, ...]
        else:
            raise ValueError(
                "Visibility shape must match the UVW leading axes or the "
                "example-script reference-baseline layout. "
                f"Got UVW shape {uvw_array.shape!r} and vis shape {vis_array.shape!r}."
            )

    uvw_samples_m = np.reshape(uvw_for_vis, (-1, 3)).astype(np.float64, copy=False)
    vis_samples = np.reshape(vis_array, (-1,)).astype(np.complex128, copy=False)

    finite_mask = np.isfinite(uvw_samples_m).all(axis=1)
    finite_mask &= np.isfinite(vis_samples.real)
    finite_mask &= np.isfinite(vis_samples.imag)

    if drop_zero_baseline:
        finite_mask &= np.any(np.abs(uvw_samples_m[:, :2]) > 0.0, axis=1)

    uvw_samples_m = uvw_samples_m[finite_mask]
    vis_samples = vis_samples[finite_mask]

    if uvw_samples_m.size == 0:
        raise ValueError("No valid UVW/visibility samples remain after filtering.")

    return uvw_samples_m, vis_samples


def resolve_frequency_hz(
    dataset: np.lib.npyio.NpzFile,
    cli_freq_mhz: float | None,
) -> float:
    """
    Resolve the observing frequency in Hz from the CLI or the input archive.

    Parameters
    ----------
    dataset : numpy.lib.npyio.NpzFile
        Opened input archive.
    cli_freq_mhz : float or None
        Command-line frequency override in MHz.

    Returns
    -------
    frequency_hz : float
        Observing frequency in Hz.

    Raises
    ------
    ValueError
        If no usable frequency was supplied or if the resolved frequency is not
        strictly positive.
    """
    if cli_freq_mhz is not None:
        frequency_hz = float(cli_freq_mhz) * 1.0e6
    elif "freq_hz" in dataset.files:
        frequency_hz = float(np.asarray(dataset["freq_hz"]).squeeze())
    elif "freq_mhz" in dataset.files:
        frequency_hz = float(np.asarray(dataset["freq_mhz"]).squeeze()) * 1.0e6
    else:
        raise ValueError(
            "No observing frequency found. Pass --freq-mhz or store freq_hz / "
            "freq_mhz in the input .npz file."
        )

    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError(f"Observing frequency must be positive, got {frequency_hz!r}.")

    return frequency_hz


def make_dirty_image(
    uvw_samples_m: np.ndarray,
    vis_samples: np.ndarray,
    frequency_hz: float,
    *,
    npix: int = 512,
    fov_deg: float = 2.0,
    weighting: str = "natural",
    hermitian_mirror: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """
    Form a quick dirty image from irregular UV samples.

    Parameters
    ----------
    uvw_samples_m : numpy.ndarray, shape (N_sample, 3)
        Flattened UVW coordinates in metres.  Only the ``u`` and ``v``
        components are used by this quick-look imager.
    vis_samples : numpy.ndarray, shape (N_sample,)
        Complex visibilities aligned with ``uvw_samples_m``.
    frequency_hz : float
        Observing frequency in Hz.  UV coordinates are converted from metres to
        wavelengths using ``lambda = c / frequency_hz``.
    npix : int, optional
        Output image size.  The dirty image and dirty beam are both returned as
        ``(npix, npix)`` arrays.
    fov_deg : float, optional
        Total image field of view in degrees across each axis.  The image uses
        a square tangent-plane approximation over ``[-fov_deg/2, +fov_deg/2]``
        in both ``l`` and ``m``.
    weighting : {"natural", "uniform"}, optional
        UV-cell weighting strategy.  ``natural`` counts every sample equally.
        ``uniform`` weights each sample by the inverse occupancy of its gridded
        UV cell.
    hermitian_mirror : bool, optional
        If ``True``, add conjugate samples at ``(-u, -v)`` so that a one-sided
        UV track can still produce a real-valued quick-look image.

    Returns
    -------
    dirty_image : numpy.ndarray, shape (npix, npix)
        Real-valued dirty image normalised by the peak of the dirty beam.
    dirty_beam : numpy.ndarray, shape (npix, npix)
        Real-valued dirty beam (synthesised beam) normalised to peak unity.
    metadata : dict[str, float | int]
        Scalar imaging metadata.  Keys are:

        - ``wavelength_m``: observing wavelength in metres.
        - ``uv_cell_lambda``: UV grid spacing in wavelengths.
        - ``gridded_sample_count``: number of UV samples kept after clipping to
          the finite FFT grid.
        - ``dropped_sample_count``: number of UV samples discarded because they
          fall outside the FFT support for the chosen ``npix`` and ``fov_deg``.

    Raises
    ------
    ValueError
        If the inputs are empty, misaligned, or physically invalid.

    Notes
    -----
    This routine deliberately prioritises robustness and speed over imaging
    fidelity.  It uses nearest-neighbour UV gridding followed by an inverse FFT
    and does not apply deconvolution or ``w``-projection.  The output should be
    treated as a preview image suitable for sanity checks and not as a
    publication-quality reconstruction.
    """
    uvw_array = np.asarray(uvw_samples_m, dtype=np.float64)
    vis_array = np.asarray(vis_samples, dtype=np.complex128)

    if uvw_array.ndim != 2 or uvw_array.shape[1] != 3:
        raise ValueError(
            "uvw_samples_m must have shape (N_sample, 3). "
            f"Got {uvw_array.shape!r}."
        )
    if vis_array.ndim != 1 or vis_array.shape[0] != uvw_array.shape[0]:
        raise ValueError(
            "vis_samples must be a 1-D array aligned with uvw_samples_m. "
            f"Got {vis_array.shape!r} for {uvw_array.shape[0]} UVW samples."
        )
    if npix <= 0:
        raise ValueError(f"npix must be positive, got {npix}.")
    if not np.isfinite(fov_deg) or fov_deg <= 0.0:
        raise ValueError(f"fov_deg must be positive, got {fov_deg!r}.")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError(
            f"frequency_hz must be positive and finite, got {frequency_hz!r}."
        )
    if weighting not in {"natural", "uniform"}:
        raise ValueError(
            "weighting must be 'natural' or 'uniform', "
            f"got {weighting!r}."
        )

    wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    u_lambda = uvw_array[:, 0] / wavelength_m
    v_lambda = uvw_array[:, 1] / wavelength_m
    gridded_vis = vis_array

    if hermitian_mirror:
        u_lambda = np.concatenate((u_lambda, -u_lambda))
        v_lambda = np.concatenate((v_lambda, -v_lambda))
        gridded_vis = np.concatenate((gridded_vis, np.conjugate(gridded_vis)))

    fov_rad = np.deg2rad(float(fov_deg))
    uv_cell_lambda = 1.0 / fov_rad
    centre = npix // 2

    u_idx = np.rint(u_lambda / uv_cell_lambda).astype(np.int64) + centre
    v_idx = np.rint(v_lambda / uv_cell_lambda).astype(np.int64) + centre

    in_bounds = (
        (u_idx >= 0)
        & (u_idx < npix)
        & (v_idx >= 0)
        & (v_idx < npix)
    )
    dropped_sample_count = int(np.count_nonzero(~in_bounds))
    if not np.any(in_bounds):
        raise ValueError(
            "All UV samples fall outside the selected FFT grid. Increase --npix "
            "or decrease --fov-deg."
        )

    u_idx = u_idx[in_bounds]
    v_idx = v_idx[in_bounds]
    gridded_vis = gridded_vis[in_bounds]

    sample_weights = np.ones(gridded_vis.shape[0], dtype=np.float64)
    if weighting == "uniform":
        density_grid = np.zeros((npix, npix), dtype=np.float64)
        np.add.at(density_grid, (v_idx, u_idx), 1.0)
        sample_weights = 1.0 / density_grid[v_idx, u_idx]

    uv_grid = np.zeros((npix, npix), dtype=np.complex128)
    sampling_grid = np.zeros((npix, npix), dtype=np.float64)
    np.add.at(uv_grid, (v_idx, u_idx), gridded_vis * sample_weights)
    np.add.at(sampling_grid, (v_idx, u_idx), sample_weights)

    dirty_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid)))
    dirty_beam = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling_grid)))

    beam_peak = float(np.max(dirty_beam.real))
    if beam_peak <= 0.0:
        raise ValueError("Dirty beam peak is non-positive; cannot normalise image.")

    metadata = {
        "wavelength_m": wavelength_m,
        "uv_cell_lambda": uv_cell_lambda,
        "gridded_sample_count": int(gridded_vis.shape[0]),
        "dropped_sample_count": int(dropped_sample_count),
    }
    return dirty_image.real / beam_peak, dirty_beam.real / beam_peak, metadata


def save_dirty_image_figure(
    dirty_image: np.ndarray,
    dirty_beam: np.ndarray,
    output_path: Path,
    *,
    fov_deg: float,
    title: str,
) -> None:
    """Save a two-panel quick-look figure containing the dirty image and beam."""
    extent_deg = (-0.5 * fov_deg, 0.5 * fov_deg, -0.5 * fov_deg, 0.5 * fov_deg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    image_artist = axes[0].imshow(
        dirty_image,
        origin="lower",
        extent=extent_deg,
        cmap="inferno",
        interpolation="nearest",
    )
    axes[0].set_title(title)
    axes[0].set_xlabel("l offset (deg)")
    axes[0].set_ylabel("m offset (deg)")
    fig.colorbar(image_artist, ax=axes[0], shrink=0.86, label="Dirty image")

    beam_artist = axes[1].imshow(
        dirty_beam,
        origin="lower",
        extent=extent_deg,
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1].set_title("Dirty beam")
    axes[1].set_xlabel("l offset (deg)")
    axes[1].set_ylabel("m offset (deg)")
    fig.colorbar(beam_artist, ax=axes[1], shrink=0.86, label="Beam response")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the quick-look imager."""
    parser = argparse.ArgumentParser(
        description=(
            "Make a quick dirty image from UVW coordinates and complex "
            "visibilities stored in an .npz archive."
        )
    )
    parser.add_argument("input_npz", type=Path, help="Input .npz file.")
    parser.add_argument(
        "--uvw-key",
        default="uvw",
        help="NPZ key containing the UVW array in metres. Default: %(default)s.",
    )
    parser.add_argument(
        "--vis-key",
        default="vis",
        help="NPZ key containing the complex visibility array. Default: %(default)s.",
    )
    parser.add_argument(
        "--vis-real-key",
        default=None,
        help="Optional NPZ key for the real visibility component.",
    )
    parser.add_argument(
        "--vis-imag-key",
        default=None,
        help="Optional NPZ key for the imaginary visibility component.",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=None,
        help="Observing frequency in MHz. Overrides freq_hz / freq_mhz in the file.",
    )
    parser.add_argument(
        "--npix",
        type=int,
        default=512,
        help="Square image size in pixels. Default: %(default)s.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=2.0,
        help="Total image field of view in degrees. Default: %(default)s.",
    )
    parser.add_argument(
        "--weighting",
        choices=("natural", "uniform"),
        default="natural",
        help="UV weighting scheme. Default: %(default)s.",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("quick_dirty_image.png"),
        help="Output PNG path. Default: %(default)s.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Defaults to the input file stem.",
    )
    return parser


def main() -> None:
    """Run the quick-look imaging workflow from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args()

    with np.load(args.input_npz, allow_pickle=False) as dataset:
        uvw_m = _load_array(dataset, args.uvw_key)
        vis = _load_visibilities(
            dataset,
            vis_key=args.vis_key,
            vis_real_key=args.vis_real_key,
            vis_imag_key=args.vis_imag_key,
        )
        frequency_hz = resolve_frequency_hz(dataset, args.freq_mhz)

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

    title = args.title or f"Dirty image: {args.input_npz.stem}"
    save_dirty_image_figure(
        dirty_image,
        dirty_beam,
        args.output,
        fov_deg=args.fov_deg,
        title=title,
    )

    print(f"Loaded UVW shape: {uvw_m.shape}")
    print(f"Loaded vis shape: {vis.shape}")
    print(f"Flattened sample count: {uvw_samples_m.shape[0]}")
    print(f"Observing frequency: {frequency_hz / 1.0e6:.6f} MHz")
    print(f"Observing wavelength: {metadata['wavelength_m']:.6f} m")
    print(f"UV grid cell size: {metadata['uv_cell_lambda']:.3f} wavelengths")
    print(f"Gridded samples kept: {int(metadata['gridded_sample_count'])}")
    print(f"Samples dropped outside FFT grid: {int(metadata['dropped_sample_count'])}")
    print(f"Saved quick-look dirty image to {args.output}")


if __name__ == "__main__":
    main()
