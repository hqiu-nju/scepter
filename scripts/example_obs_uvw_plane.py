#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
example_obs_uvw_plane.py - UVW-plane coverage from an obs_sim observation

Demonstrates how to compute and plot UVW-plane coordinates for every
antenna in a scepter array across a full observation.  The helper
function ``obs_uvw_plane`` reads the observer locations, time array, and
phase-centre stored in a completed ``obs_sim`` object and returns the
UVW baselines relative to the reference (first) antenna.

Usage
-----
    conda activate scepter-dev
    python scripts/example_obs_uvw_plane.py

Requirements
------------
- numpy
- astropy
- cysgp4
- pycraf
- matplotlib (for the optional UV-coverage plot)
- scepter (scepter.obs, scepter.uvw, scepter.skynet)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import ICRS, AltAz
from astropy.time import Time
from cysgp4 import PyObserver

from scepter import obs, skynet, uvw as _uvw


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

def obs_uvw_plane(
    sim: obs.obs_sim,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute UVW-plane coordinates for all antennas across the full observation.

    Reads the observer locations, time array, and phase-centre tracking
    information stored in a completed ``obs_sim`` object and returns the
    UVW baseline coordinates for every antenna pair relative to the first
    (reference) antenna, sampled at every observation time step.

    The result can be used directly for UV-coverage plots, visibility
    weighting, or fringe-rate analysis.

    Parameters
    ----------
    sim : obs.obs_sim
        A fully initialised ``obs_sim`` instance.  The following attributes
        must have been set before calling this function:

        - ``sim.location`` — observer/antenna array, shape
          ``(N_ant, 1, 1, 1, 1, 1)`` of ``cysgp4.PyObserver`` objects.
          Set by ``obs_sim.__init__``.
        - ``sim.mjds`` — Modified Julian Date array (any shape).  The array
          is flattened to a 1-D time sequence.  Set by ``obs_sim.__init__``.
        - ``sim.pnt_coord`` — phase-centre ``astropy.SkyCoord``.  Set by
          either ``obs_sim.sky_track`` (celestial frame) or
          ``obs_sim.azel_track`` (topocentric AltAz frame).
        - ``sim.altaz_frame`` — ``astropy.coordinates.AltAz`` frame attached
          to the first antenna and the observation times.  Required only when
          ``pnt_coord`` is in an AltAz frame (i.e. after ``azel_track``).

    Returns
    -------
    uvw_all : numpy.ndarray, shape (N_ant, N_times, 3)
        UVW coordinates (metres) for every antenna relative to
        ``antennas[0]`` (the reference).

        - Axis 0 — antenna index (``uvw_all[0]`` is always zero).
        - Axis 1 — time index (flattened ``mjds`` order).
        - Axis 2 — UVW component: ``[u, v, w]``.

    hour_angles : numpy.ndarray, shape (N_times,)
        Hour angle of the phase centre at each time step (radians).
        ``hour_angles[t] = LST(t) - RA_phase_centre``.

    Raises
    ------
    AttributeError
        If ``sim.pnt_coord`` or ``sim.mjds`` have not been set.
    ValueError
        If ``sim.location`` contains no antennas.

    Notes
    -----
    *After* ``sky_track(ra, dec)``:
        ``pnt_coord`` is in a celestial frame and is transformed to ICRS
        directly.

    *After* ``azel_track(az, el)``:
        ``pnt_coord`` is in an AltAz frame.  The AltAz direction is
        converted to ICRS at the first time step and used as a fixed
        phase centre for the whole observation.  For time-varying
        phase centres use ``sky_track`` instead.

    ``sim.mjds`` is always flattened to 1-D before computing UVW so
    that output axis 1 has length ``N_epochs * N_int``.

    The UVW convention follows Thompson, Moran & Swenson (2017), Ch. 4.
    """
    # 1. Extract flat list of PyObserver objects
    antennas = list(sim.location.flatten())
    if len(antennas) == 0:
        raise ValueError(
            "sim.location contains no antennas. "
            "Ensure obs_sim was initialised with a valid receiver_info."
        )

    # 2. Flatten MJD array and build astropy Time sequence
    mjds_flat = np.asarray(sim.mjds).ravel()
    obs_times = Time(mjds_flat, format="mjd", scale="utc")

    # 3. Resolve phase-centre RA / Dec in ICRS degrees
    pnt = sim.pnt_coord
    if isinstance(pnt.frame, AltAz):
        # azel_track: convert fixed AltAz direction to ICRS at first time step
        icrs_pnt = pnt.transform_to(ICRS())
        icrs_pnt_0 = icrs_pnt.flatten()[0]
        ra_deg = float(icrs_pnt_0.ra.deg)
        dec_deg = float(icrs_pnt_0.dec.deg)
    else:
        # sky_track: pnt_coord is already celestial
        icrs_pnt = pnt.transform_to(ICRS())
        if icrs_pnt.isscalar:
            ra_deg = float(icrs_pnt.ra.deg)
            dec_deg = float(icrs_pnt.dec.deg)
        else:
            ra_deg = float(icrs_pnt.ra.deg.flat[0])
            dec_deg = float(icrs_pnt.dec.deg.flat[0])

    # 4. Compute UVW for all baselines
    uvw_all, hour_angles = _uvw.compute_uvw_array(
        antennas,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        obs_times=obs_times,
    )

    return uvw_all, hour_angles


# ---------------------------------------------------------------------------
# Example: three-antenna MeerKAT-like configuration tracking the Crab Nebula
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Array definition  (lon deg, lat deg, alt m)
    # Three antennas near SKA-Mid / MeerKAT site, South Africa
    # ------------------------------------------------------------------
    ant_coords = [
        (21.4430, -30.7130, 1086.0),   # reference
        (21.4445, -30.7130, 1086.0),   # ~130 m East
        (21.4430, -30.7115, 1086.0),   # ~170 m North
    ]
    pyobs_array = np.array(
        [PyObserver(lon=lon, lat=lat, alt=alt) for lon, lat, alt in ant_coords]
    )

    # ------------------------------------------------------------------
    # Receiver definition
    # ------------------------------------------------------------------
    rx = obs.receiver_info(
        d_rx=13.5 * u.m,
        eta_a_rx=0.7,
        pyobs=pyobs_array,
        freq=1420 * u.MHz,
        bandwidth=10 * u.MHz,
    )

    # ------------------------------------------------------------------
    # Time grid: 6 hours centred on 2024-01-01 02:00 UTC, 60-s cadence
    # ------------------------------------------------------------------
    t0 = Time("2024-01-01T02:00:00", scale="utc")
    dt_s = 60.0          # seconds per sample
    n_samples = 360       # 6 hours
    mjds = (t0 + np.arange(n_samples) * dt_s * u.s).mjd

    # ------------------------------------------------------------------
    # Minimal sky grid (1 pointing, 1 cell) — only mjds matters here
    # ------------------------------------------------------------------
    skygrid = skynet.pointgen_S_1586_1(niters=1)

    # ------------------------------------------------------------------
    # Build obs_sim and set phase centre: Crab Nebula (RA=83.633, Dec=22.014)
    # ------------------------------------------------------------------
    sim = obs.obs_sim(rx, skygrid, mjds)
    sim.sky_track(ra=83.633, dec=22.014)   # ICRS degrees

    # ------------------------------------------------------------------
    # Compute UVW plane
    # ------------------------------------------------------------------
    uvw_all, hour_angles = obs_uvw_plane(sim)

    print(f"uvw_all shape : {uvw_all.shape}")   # (3, 360, 3)
    print(f"hour_angles shape: {hour_angles.shape}")  # (360,)
    print(f"Reference baseline (ant 0): max |UVW| = {np.abs(uvw_all[0]).max():.3f} m")
    print(f"Baseline 0→1  u range: {uvw_all[1, :, 0].min():.1f} … {uvw_all[1, :, 0].max():.1f} m")
    print(f"Baseline 0→2  v range: {uvw_all[2, :, 1].min():.1f} … {uvw_all[2, :, 1].max():.1f} m")

    # ------------------------------------------------------------------
    # UV-coverage plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["C1", "C2"]
    for i in range(1, len(ant_coords)):
        u_m = uvw_all[i, :, 0]
        v_m = uvw_all[i, :, 1]
        ax.plot(u_m, v_m, lw=0.8, color=colors[i - 1], label=f"Baseline 0→{i}")
        # Hermitian conjugate (mirror)
        ax.plot(-u_m, -v_m, lw=0.8, color=colors[i - 1], ls="--", alpha=0.5)

    ax.set_xlabel("u  (m)")
    ax.set_ylabel("v  (m)")
    ax.set_title("UV coverage — Crab Nebula, 6 h track")
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    plt.savefig("uvw_coverage.png", dpi=150)
    print("Saved UV coverage plot to uvw_coverage.png")
    plt.show()
