#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_uv_plane.py - Visualize UV plane coverage from UVW products.

This script loads UVW data from an NPZ archive (produced by
``scripts/example_obs_uvw_plane.py``) and creates publication-quality figures
showing the UV plane coverage, including:

- A figure of the UV plane for the pointing (fixed source).
- A figure of the combined UV plane signals (pointing + optional satellites).

Usage
-----
    conda activate scepter-dev
    python scripts/plot_uv_plane.py \\
        --input tracking_uvw.npz \\
        --output-dir ./uv_plots \\
        --include-satellites \\
        --num-satellites 2 \\
        --fov-radius-deg 10

Input NPZ format
----------------
The input ``.npz`` file must contain:

- ``pointing_uvw_m``: Pointing UVW coordinates with shape ``(N_ant, N_time, 3)``.
- ``satellite_uvw_m`` (optional): Satellite UVW coordinates with shape
  ``(N_sat, N_time, 3)`` or ``(N_sat, N_ant, N_time, 3)``.
- ``satellite_separation_deg`` (optional): Pointing-to-satellite angular
    separation with shape ``(N_time, N_sat)``. Used by FoV filtering.
- ``freq_mhz``: Receiver centre frequency in MHz.

FoV filtering is enabled by default for satellite plots and keeps only
time instances where ``satellite_separation_deg <= fov_radius_deg``.
Use ``--disable-fov-filter`` to plot all satellite UV samples.

Output
------
PNG figures:

- ``uv_plane_pointing.png``: UV plane for the pointing (tracked source).
- ``uv_plane_satellites.png`` (if satellites are plotted): UV plane using only
    the selected satellites.
- ``uv_plane_combined.png`` (if --include-satellites): Combined UV coverage
  showing both pointing and satellite UV tracks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import os

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "scepter-mpl-cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def _flatten_uv_samples(uvw_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and flatten u, v coordinates from UVW array.

    Parameters
    ----------
    uvw_m : numpy.ndarray
        UVW coordinates with shape (..., 3) or (..., 3, N_ant).
    
    Returns
    -------
    u : numpy.ndarray, shape (N_sample,)
        Flattened u coordinates in metres.
    v : numpy.ndarray, shape (N_sample,)
        Flattened v coordinates in metres.
    """
    uvw_array = np.asarray(uvw_m, dtype=np.float64)
    
    # Flatten all dimensions except the last (which is [u, v, w])
    if uvw_array.ndim < 3:
        raise ValueError(
            f"uvw_m must have at least 3 dimensions (..., 3), got {uvw_array.shape!r}."
        )
    
    # Reshape to (N_sample, 3)
    uvw_flat = np.reshape(uvw_array, (-1, 3))
    
    # Extract u and v, apply Hermitian conjugate symmetry
    u = uvw_flat[:, 0]
    v = uvw_flat[:, 1]
    
    # Mirror to complete the UV plane (add conjugate samples)
    u_mirrored = np.concatenate([u, -u])
    v_mirrored = np.concatenate([v, -v])
    
    return u_mirrored, v_mirrored


def _select_satellites(satellite_uvw_m: np.ndarray, num_satellites: int) -> np.ndarray:
    """
    Select the first ``num_satellites`` entries along the satellite axis.

    Parameters
    ----------
    satellite_uvw_m : numpy.ndarray
        Satellite UVW coordinates with satellite index on axis 0.
    num_satellites : int
        Number of satellites to include.

    Returns
    -------
    numpy.ndarray
        Satellite UVW array sliced to the selected number of satellites.
    """
    if num_satellites <= 0:
        raise ValueError("--num-satellites must be a positive integer.")

    n_available = int(satellite_uvw_m.shape[0])
    n_selected = min(num_satellites, n_available)
    if n_selected < num_satellites:
        print(
            "Warning: requested "
            f"{num_satellites} satellites, but only {n_available} are available. "
            f"Using {n_selected}."
        )

    return satellite_uvw_m[:n_selected, ...]


def _infer_satellite_time_axes(
    satellite_uvw_m: np.ndarray,
    satellite_separation_deg: np.ndarray | None,
) -> tuple[int, int]:
    """Infer satellite and time axis indices from UVW and separation arrays."""
    uvw = np.asarray(satellite_uvw_m)
    if uvw.ndim not in (3, 4) or uvw.shape[-1] != 3:
        raise ValueError(
            "satellite_uvw_m must have shape (N_sat, T, 3), (T, N_sat, 3), "
            "(N_sat, N_ant, T, 3), or (N_ant, T, N_sat, 3). "
            f"Got {uvw.shape!r}."
        )

    if satellite_separation_deg is not None:
        sep = np.asarray(satellite_separation_deg, dtype=np.float64)
        if sep.ndim != 2:
            raise ValueError(
                f"satellite_separation_deg must have shape (T, N_sat), got {sep.shape!r}."
            )
        t_len, s_len = int(sep.shape[0]), int(sep.shape[1])

        if uvw.ndim == 3:
            if uvw.shape[0] == s_len and uvw.shape[1] == t_len:
                return 0, 1
            if uvw.shape[1] == s_len and uvw.shape[0] == t_len:
                return 1, 0
        else:
            # Common 4D layouts.
            if uvw.shape[0] == s_len and uvw.shape[2] == t_len:
                return 0, 2
            if uvw.shape[2] == s_len and uvw.shape[1] == t_len:
                return 2, 1

    # Fallback defaults for legacy layouts.
    if uvw.ndim == 3:
        return 0, 1
    return 0, 2


def _select_satellites_with_optional_separation(
    satellite_uvw_m: np.ndarray,
    satellite_separation_deg: np.ndarray | None,
    num_satellites: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Select first N satellites and keep separation aligned if available."""
    if num_satellites <= 0:
        raise ValueError("--num-satellites must be a positive integer.")

    sat_axis, _ = _infer_satellite_time_axes(satellite_uvw_m, satellite_separation_deg)
    n_available = int(np.asarray(satellite_uvw_m).shape[sat_axis])
    n_selected = min(num_satellites, n_available)
    if n_selected < num_satellites:
        print(
            "Warning: requested "
            f"{num_satellites} satellites, but only {n_available} are available. "
            f"Using {n_selected}."
        )

    satellite_uvw_selected = np.take(
        np.asarray(satellite_uvw_m),
        indices=np.arange(n_selected, dtype=int),
        axis=sat_axis,
    )

    sep_selected: np.ndarray | None = None
    if satellite_separation_deg is not None:
        sep = np.asarray(satellite_separation_deg, dtype=np.float64)
        if sep.ndim == 2 and sep.shape[1] >= n_selected:
            sep_selected = sep[:, :n_selected]
        else:
            print(
                "Warning: satellite_separation_deg is missing or incompatible; "
                "FoV filtering will be skipped."
            )

    return satellite_uvw_selected, sep_selected


def _flatten_satellite_uv_samples(
    satellite_uvw_m: np.ndarray,
    satellite_separation_deg: np.ndarray | None,
    fov_radius_deg: float | None,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """
    Flatten satellite UV samples and optionally apply FoV time filtering.

    Returns
    -------
    u : numpy.ndarray
        Flattened (and mirrored) u samples in metres.
    v : numpy.ndarray
        Flattened (and mirrored) v samples in metres.
    n_in_fov : int | None
        Number of (time, satellite) samples inside FoV before Hermitian
        mirroring. ``None`` when FoV filtering is disabled.
    """
    sat_axis, time_axis = _infer_satellite_time_axes(satellite_uvw_m, satellite_separation_deg)
    uvw = np.asarray(satellite_uvw_m, dtype=np.float64)

    u = np.moveaxis(uvw[..., 0], (time_axis, sat_axis), (-2, -1))
    v = np.moveaxis(uvw[..., 1], (time_axis, sat_axis), (-2, -1))

    n_in_fov: int | None = None
    if fov_radius_deg is not None and satellite_separation_deg is not None:
        sep = np.asarray(satellite_separation_deg, dtype=np.float64)
        if sep.shape != u.shape[-2:]:
            raise ValueError(
                "satellite_separation_deg shape must match UVW (time, satellite) axes. "
                f"Expected {u.shape[-2:]}, got {sep.shape}."
            )
        mask_ts = sep <= float(fov_radius_deg)
        n_in_fov = int(np.count_nonzero(mask_ts))
        mask = np.broadcast_to(mask_ts, u.shape)
        u_flat = u[mask]
        v_flat = v[mask]
    else:
        u_flat = np.reshape(u, (-1,))
        v_flat = np.reshape(v, (-1,))

    return (
        np.concatenate([u_flat, -u_flat]),
        np.concatenate([v_flat, -v_flat]),
        n_in_fov,
    )


def _plot_uv_plane(
    u_samples: np.ndarray,
    v_samples: np.ndarray,
    frequency_mhz: float,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Create and save a UV plane coverage plot.

    Parameters
    ----------
    u_samples : numpy.ndarray, shape (N_sample,)
        u coordinates in metres.
    v_samples : numpy.ndarray, shape (N_sample,)
        v coordinates in metres.
    frequency_mhz : float
        Observing frequency in MHz (used for plot annotation).
    title : str
        Plot title.
    output_path : Path
        Output PNG file path.
    dpi : int
        Figure resolution in dots-per-inch.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to wavelengths
    frequency_hz = frequency_mhz * 1e6
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / frequency_hz
    u_lambda = u_samples / wavelength_m
    v_lambda = v_samples / wavelength_m
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    
    # Plot UV coverage
    ax.scatter(u_lambda, v_lambda, s=1, alpha=0.3, c="blue", edgecolors="none")
    
    # Formatting
    ax.set_xlabel(r"$u$ (wavelengths)", fontsize=12)
    ax.set_ylabel(r"$v$ (wavelengths)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")
    
    # Add frequency annotation
    freq_label = f"Frequency: {frequency_mhz:.1f} MHz (λ = {wavelength_m*1e2:.2f} cm)"
    ax.text(
        0.05, 0.95, freq_label,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )
    
    # Add sample count annotation
    sample_label = f"Samples: {len(u_samples)//2:,} (x2 with conjugate)"
    ax.text(
        0.05, 0.88, sample_label,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )
    
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create UV plane visualizations from UVW NPZ data."
    )
    parser.add_argument(
        '-i',"--input",
        required=True,
        help="Path to the NPZ file from example_obs_uvw_plane.py.",
    )
    parser.add_argument(
        "-o","--output-dir",
        default="./uv_plots",
        help="Output directory for PNG figures. Default: ./uv_plots",
    )
    parser.add_argument(
        "-a","--include-satellites",
        dest="include_satellites",
        action="store_true",
        help="Create an additional combined UV plane figure including satellite data.",
    )
    parser.add_argument(
        "-s","--satellite-only",
        dest="satellite_only",
        action="store_true",
        help="Plot only satellites (skip pointing-only and combined figures).",
    )
    parser.add_argument(
        "-n","--num-satellites",
        type=int,
        default=2,
        help="Number of satellites to include when plotting satellite UV data. Default: 2",
    )
    parser.add_argument(
        "--fov-radius-deg",
        type=float,
        default=10.0,
        help=(
            "Field-of-view radius in degrees used to filter satellite UV points by time. "
            "Default: 10"
        ),
    )
    parser.add_argument(
        "--disable-fov-filter",
        action="store_true",
        help="Disable FoV filtering and plot all satellite UV samples.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output figure DPI. Default: 150",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load NPZ file
    print(f"Loading UVW data from {input_path}...")
    data = np.load(input_path)
    
    # Check for required keys
    if "pointing_uvw_m" not in data:
        raise KeyError("Input NPZ must contain 'pointing_uvw_m'.")
    if "freq_mhz" not in data:
        raise KeyError("Input NPZ must contain 'freq_mhz'.")
    
    frequency_mhz = float(data["freq_mhz"])
    if args.fov_radius_deg <= 0.0:
        raise ValueError("--fov-radius-deg must be greater than zero.")
    
    u_point: np.ndarray | None = None
    v_point: np.ndarray | None = None

    if not args.satellite_only:
        # Plot pointing UV plane
        print("Creating pointing UV plane figure...")
        pointing_uvw = data["pointing_uvw_m"]
        u_point, v_point = _flatten_uv_samples(pointing_uvw)

        _plot_uv_plane(
            u_point,
            v_point,
            frequency_mhz,
            "UV Plane: Pointing (Fixed Source)",
            output_dir / "uv_plane_pointing.png",
            dpi=args.dpi,
        )

    u_sat: np.ndarray | None = None
    v_sat: np.ndarray | None = None
    satellites_requested = args.include_satellites or args.satellite_only
    if satellites_requested:
        if "satellite_uvw_m" not in data:
            if args.satellite_only:
                raise KeyError(
                    "--satellite-only requested but input NPZ does not contain 'satellite_uvw_m'."
                )
            print(
                "Warning: satellite plotting requested but 'satellite_uvw_m' "
                "not in NPZ. Skipping satellite plots."
            )
        else:
            satellite_uvw = np.asarray(data["satellite_uvw_m"])
            satellite_sep = (
                np.asarray(data["satellite_separation_deg"], dtype=np.float64)
                if "satellite_separation_deg" in data
                else None
            )
            satellite_uvw, satellite_sep = _select_satellites_with_optional_separation(
                satellite_uvw,
                satellite_sep,
                args.num_satellites,
            )

            print("Creating satellites-only UV plane figure...")
            fov_radius = None if args.disable_fov_filter else args.fov_radius_deg
            u_sat, v_sat, n_in_fov = _flatten_satellite_uv_samples(
                satellite_uvw,
                satellite_sep,
                fov_radius_deg=fov_radius,
            )
            sat_axis_selected, _ = _infer_satellite_time_axes(satellite_uvw, satellite_sep)
            n_sat_selected = int(satellite_uvw.shape[sat_axis_selected])
            _plot_uv_plane(
                u_sat,
                v_sat,
                frequency_mhz,
                f"UV Plane: Satellites Only (first {n_sat_selected} satellite(s))",
                output_dir / "uv_plane_satellites.png",
                dpi=args.dpi,
            )

            if fov_radius is not None:
                if satellite_sep is None:
                    print(
                        "FoV filter requested but satellite_separation_deg not found in NPZ. "
                        "Plotted all satellite UV samples."
                    )
                else:
                    total_ts = int(np.prod(satellite_sep.shape))
                    print(
                        f"FoV filtering at {fov_radius:.2f} deg kept "
                        f"{n_in_fov}/{total_ts} time-satellite samples "
                        "before conjugate mirroring."
                    )

            if args.include_satellites and not args.satellite_only and u_point is not None and v_point is not None:
                print("Creating combined UV plane figure...")
                # Combine pointing and satellite UV samples
                u_combined = np.concatenate([u_point, u_sat])
                v_combined = np.concatenate([v_point, v_sat])

                _plot_uv_plane(
                    u_combined,
                    v_combined,
                    frequency_mhz,
                    "UV Plane: Combined (Pointing + Satellites)",
                    output_dir / "uv_plane_combined.png",
                    dpi=args.dpi,
                )

            print(f"Satellite UV samples: {len(u_sat)}")

    if u_point is not None:
        print(f"Pointing UV samples: {len(u_point)}")
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
