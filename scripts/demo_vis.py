#!/usr/bin/env python
"""Smoke-test and demo script for ``scepter.vis``.

Run from the repository root:

    python scripts/demo_vis.py

The default run avoids CASA-only operations, so it is safe in the normal
development environment. Pass ``--create-ms`` to exercise Measurement Set
creation when CASA 6 modular packages are installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from scepter import vis


def _demo_antennas() -> list[EarthLocation]:
    """Return a compact three-dish demo array near the SKA-Mid site."""
    return [
        EarthLocation.from_geodetic(21.443611 * u.deg, -30.712777 * u.deg, 1050.0 * u.m),
        EarthLocation.from_geodetic(21.444200 * u.deg, -30.712400 * u.deg, 1051.0 * u.m),
        EarthLocation.from_geodetic(21.442900 * u.deg, -30.713200 * u.deg, 1049.0 * u.m),
    ]


def _validate_antenna_config(config_path: Path, expected_antennas: int) -> None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    station_lines = [line for line in lines if line and not line.startswith("#")]
    if len(station_lines) != expected_antennas:
        raise RuntimeError(
            f"Expected {expected_antennas} antenna rows in {config_path}, "
            f"found {len(station_lines)}."
        )
    for index, line in enumerate(station_lines):
        expected_name = f"ANT{index:02d}"
        if not line.endswith(expected_name):
            raise RuntimeError(
                f"Expected station row {index} to end with {expected_name!r}; got {line!r}."
            )


def run_demo(*, output_dir: Path, create_ms: bool, overwrite_ms: bool, require_casa: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    antennas = _demo_antennas()
    config_path = vis.write_antenna_config(
        antennas,
        output_dir / "scepter_vis_demo.cfg",
        dish_diameter=15.0 * u.m,
        observatory_name="SCEPTER_VIS_DEMO",
    )
    _validate_antenna_config(config_path, expected_antennas=len(antennas))
    print(f"Wrote and validated antenna config: {config_path}")

    spw = vis.SpectralWindow(
        freq_center=1420.40575177 * u.MHz,
        freq_bandwidth=1.0 * u.MHz,
        n_channels=16,
    )
    expected_width = 62.5 * u.kHz
    if abs((spw.channel_width - expected_width).to_value(u.Hz)) > 1e-9:
        raise RuntimeError(f"Unexpected channel width: {spw.channel_width} != {expected_width}")
    print(f"Validated spectral window: {spw.n_channels} channels x {spw.channel_width.to(u.kHz):.3f}")

    if not vis.CASATOOLS_AVAILABLE:
        message = "casatools is not installed; skipping Measurement Set creation."
        if require_casa:
            raise RuntimeError(message)
        print(message)
        return 0

    simulator = vis.VisibilitySimulator(
        antennas=antennas,
        freq_center=spw.freq_center,
        freq_bandwidth=spw.freq_bandwidth,
        n_channels=spw.n_channels,
        integration_time=5.0 * u.s,
        dish_diameter=15.0 * u.m,
        observatory_name="SCEPTER_VIS_DEMO",
    )

    if create_ms:
        ms_path = simulator.create_empty_ms(
            output_dir / "scepter_vis_demo.ms",
            obs_time=Time("2026-01-15T00:00:00", scale="utc"),
            obs_duration=30.0 * u.s,
            phase_center_ra=0.0 * u.deg,
            phase_center_dec=-30.0 * u.deg,
            overwrite=overwrite_ms,
        )
        print(f"Created demo Measurement Set: {ms_path}")
    else:
        print("casatools is available; pass --create-ms to create a demo Measurement Set.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "demo_outputs" / "vis_demo",
        help="Directory for generated demo files.",
    )
    parser.add_argument(
        "--create-ms",
        action="store_true",
        help="Create a tiny Measurement Set if CASA 6 modular packages are installed.",
    )
    parser.add_argument(
        "--overwrite-ms",
        action="store_true",
        help="Replace an existing demo Measurement Set when used with --create-ms.",
    )
    parser.add_argument(
        "--require-casa",
        action="store_true",
        help="Fail instead of skipping when casatools is unavailable.",
    )
    args = parser.parse_args()

    return run_demo(
        output_dir=args.output_dir,
        create_ms=args.create_ms,
        overwrite_ms=args.overwrite_ms,
        require_casa=args.require_casa,
    )


if __name__ == "__main__":
    raise SystemExit(main())
