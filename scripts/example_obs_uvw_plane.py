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
    conda activate scepter
    python scripts/example_obs_uvw_plane.py

Requirements
------------
- numpy
- astropy
- cysgp4
- pycraf
- requests
- matplotlib (for the optional UV-coverage plot)
- scepter (scepter.obs, scepter.uvw, scepter.skynet)
"""

from __future__ import annotations

import cysgp4
import requests
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import ICRS, AltAz, EarthLocation, SkyCoord
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
    uvw_all, hour_angles = _uvw.compute_uvw(
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
        (21.4440, -30.7130, 1086.0),   #  East
        (21.4430, -30.7120, 1086.0),   #  North
        (21.4440, -30.7120, 1086.0),
        (21.4440, -30.7122, 1086.0),
        (21.4440, -30.7123, 1086.0),
        (21.4440, -30.7124, 1086.0),
    ]
    pyobs_array = np.array(
        [PyObserver(lon, lat, alt/1000) for lon, lat, alt in ant_coords]
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
    # Time grid: 20 seconds starting from current UTC time, 1-s cadence
    # ------------------------------------------------------------------
    pydt = cysgp4.PyDateTime()   # current UTC date/time
    mjds = skynet.plantime(
        epochs=1,
        cadence=24 * u.hour,
        trange=3600 *1* u.s,
        tint=1 * u.s,
        startdate=pydt,
    )
    n_samples = mjds.size

    # ------------------------------------------------------------------
    # Minimal sky grid (1 pointing, 1 cell) — only mjds matters here
    # ------------------------------------------------------------------
    skygrid = skynet.pointgen_S_1586_1(niters=1)

    # ------------------------------------------------------------------
    # Build obs_sim and set phase centre 
    # ------------------------------------------------------------------
    sim = obs.obs_sim(rx, skygrid, mjds)
    ra=83.633 * u.deg
    dec=50.014 * u.deg  

    # ------------------------------------------------------------------
    # Download 1000 Starlink TLEs from Celestrak and populate obs_sim
    # ------------------------------------------------------------------
    print("Downloading OneWeb TLEs from Celestrak...")
    url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle'
    ctrak_starlink = requests.get(url).text
    tle_list = cysgp4.tle_tuples_from_text(ctrak_starlink) 
    tles = np.array([cysgp4.PyTle(*tle) for tle in tle_list])
    print(f"Propagating {len(tles)} Starlink satellites over {n_samples} s...")
    sim.populate(tles)
    sim.reduce_sats(el_limit=-30)
    
    print(f"Satellites above 10° elevation: {sim.topo_pos_el.shape[-1]}")

    # ------------------------------------------------------------------
    # Compute UVW plane — phase centre
    # ------------------------------------------------------------------
    sim.sky_track(ra.value, dec.value)   # ICRS degrees
    uvw_crab, ha_crab = obs_uvw_plane(sim)

    # ------------------------------------------------------------------
    # Find the two satellites closest (in angle) to the pointing direction.
    # Build UVW arrays for those two satellites, using each satellite's
    # own ICRS RA/Dec as the phase centre.
    # ------------------------------------------------------------------
    pnt_coord = SkyCoord(ra=ra, dec=dec, frame="icrs")
    ref_ant = pyobs_array[0]
    loc_ref = EarthLocation(
        lon=ref_ant.loc.lon * u.deg,
        lat=ref_ant.loc.lat * u.deg,
        height=ref_ant.loc.alt * u.m,
    )
    obs_times_1d = Time(np.asarray(mjds).ravel(), format="mjd", scale="utc")

    az_ts = sim.topo_pos_az[0, 0, 0, 0, :, :]   # (T, N_sat)
    el_ts = sim.topo_pos_el[0, 0, 0, 0, :, :]   # (T, N_sat)
    sat_altaz = SkyCoord(
        az=az_ts * u.deg,
        alt=el_ts * u.deg,
        frame=AltAz(obstime=obs_times_1d[:, np.newaxis], location=loc_ref),
    )
    sat_icrs = sat_altaz.transform_to(ICRS())
    pnt_altaz = pnt_coord.transform_to(AltAz(obstime=obs_times_1d, location=loc_ref))
    print(sat_icrs.shape)  # (T, N_sat)
    # Time-averaged angular separation to the pointing direction.
    separations_deg = sat_altaz.separation(pnt_altaz[:, np.newaxis]).deg
    mean_sep_deg = np.nanmean(separations_deg, axis=0)
    closest_idx = np.argsort(mean_sep_deg)[:2]

    print("Two satellites closest to pointing coordinate:")
    for s in closest_idx:
        print(f"  Sat {int(s)}: mean sep={mean_sep_deg[int(s)]:.3f}°")

    sat1_ra_deg = sat_icrs.ra.deg[:, closest_idx[0]]
    sat1_dec_deg = sat_icrs.dec.deg[:, closest_idx[0]]
    sat2_ra_deg = sat_icrs.ra.deg[:, closest_idx[1]]
    sat2_dec_deg = sat_icrs.dec.deg[:, closest_idx[1]]

    print(
        f"  Sat {int(closest_idx[0])} phase centre track: "
        f"RA=[{sat1_ra_deg.min():.4f}, {sat1_ra_deg.max():.4f}]°, "
        f"Dec=[{sat1_dec_deg.min():.4f}, {sat1_dec_deg.max():.4f}]°"
    )
    print(
        f"  Sat {int(closest_idx[1])} phase centre track: "
        f"RA=[{sat2_ra_deg.min():.4f}, {sat2_ra_deg.max():.4f}]°, "
        f"Dec=[{sat2_dec_deg.min():.4f}, {sat2_dec_deg.max():.4f}]°"
    )

    antennas = list(pyobs_array)
    uvw_sat1, ha_sat1 = _uvw.compute_uvw(
        antennas,
        ra_deg=sat1_ra_deg,
        dec_deg=sat1_dec_deg,
        obs_times=obs_times_1d,
    )
    uvw_sat2, ha_sat2 = _uvw.compute_uvw(
        antennas,
        ra_deg=sat2_ra_deg,
        dec_deg=sat2_dec_deg,
        obs_times=obs_times_1d,
    )

    print(f"uvw_crab shape : {uvw_crab.shape}")
    print(f"uvw_sat1 shape : {uvw_sat1.shape}")
    print(f"uvw_sat2 shape : {uvw_sat2.shape}")
    print(f"Reference baseline (ant 0): max |UVW| = {np.abs(uvw_crab[0]).max():.3f} m")
    print(f"Baseline 0→1  u range (Crab): {uvw_crab[1, :, 0].min():.1f} … {uvw_crab[1, :, 0].max():.1f} m")
    print(f"Baseline 0→2  v range (Crab): {uvw_crab[2, :, 1].min():.1f} … {uvw_crab[2, :, 1].max():.1f} m")

    # ------------------------------------------------------------------
    # W-term comparison across sources
    # ------------------------------------------------------------------
    print("\nW-term comparison (meters):")
    for i in range(1, len(ant_coords)):
        w_crab = uvw_crab[i, :, 2]
        w_sat1 = uvw_sat1[i, :, 2]
        w_sat2 = uvw_sat2[i, :, 2]
        d_w_sat1 = w_sat1 - w_crab
        d_w_sat2 = w_sat2 - w_crab
        print(
            f"  Baseline 0→{i}: "
            f"Crab w=[{w_crab.min():.3f}, {w_crab.max():.3f}], "
            f"sat1 Δw RMS={np.sqrt(np.mean(d_w_sat1**2)):.6f}, "
            f"sat2 Δw RMS={np.sqrt(np.mean(d_w_sat2**2)):.6f}"
        )

    fig_w, axes_w = plt.subplots(len(ant_coords) - 1, 2, figsize=(10, 3.5 * (len(ant_coords) - 1)))
    if len(ant_coords) - 1 == 1:
        axes_w = np.array([axes_w])

    t_sec = (np.asarray(mjds).ravel() - np.asarray(mjds).ravel()[0]) * 86400.0
    for row, i in enumerate(range(1, len(ant_coords))):
        w_crab = uvw_crab[i, :, 2]
        w_sat1 = uvw_sat1[i, :, 2]
        w_sat2 = uvw_sat2[i, :, 2]

        ax_w = axes_w[row, 0]
        ax_dw = axes_w[row, 1]

        ax_w.plot(t_sec, w_crab, label="Crab", lw=1.5)
        ax_w.plot(t_sec, w_sat1, label="Closest sat 1", lw=1.2, ls=":")
        ax_w.plot(t_sec, w_sat2, label="Closest sat 2", lw=1.2, ls="-.")
        ax_w.set_title(f"Baseline 0→{i}: w(t)")
        ax_w.set_xlabel("Time from start (s)")
        ax_w.set_ylabel("w (m)")
        ax_w.legend()

        ax_dw.plot(t_sec, w_sat1 - w_crab, label="sat1 - Crab", lw=1.2)
        ax_dw.plot(t_sec, w_sat2 - w_crab, label="sat2 - Crab", lw=1.2)
        ax_dw.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax_dw.set_title(f"Baseline 0→{i}: Δw(t)")
        ax_dw.set_xlabel("Time from start (s)")
        ax_dw.set_ylabel("Δw (m)")
        ax_dw.legend()

    plt.tight_layout()
    plt.savefig("w_term_comparison.png", dpi=150)
    print("Saved W-term comparison plot to w_term_comparison.png")

    # ------------------------------------------------------------------
    # Array layout plot (local EN offsets from reference antenna)
    # ------------------------------------------------------------------
    lon0, lat0, _ = ant_coords[0]
    lons = np.array([c[0] for c in ant_coords])
    lats = np.array([c[1] for c in ant_coords])
    east_m = (lons - lon0) * np.cos(np.deg2rad(lat0)) * 111320.0
    north_m = (lats - lat0) * 110540.0

    fig_layout, ax_layout = plt.subplots(figsize=(5.8, 5.4))
    ax_layout.scatter(east_m, north_m, s=60, c="C0")
    for i in range(len(ant_coords)):
        ax_layout.text(east_m[i] + 1.0, north_m[i] + 1.0, f"ant{i}", fontsize=9)
    ax_layout.set_xlabel("East offset (m)")
    ax_layout.set_ylabel("North offset (m)")
    ax_layout.set_title("Antenna Array Layout")
    ax_layout.set_aspect("equal")
    ax_layout.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("array_layout.png", dpi=150)
    print("Saved array layout plot to array_layout.png")

    # ------------------------------------------------------------------
    # UV-coverage plots (three sources as subplots in one figure)
    # ------------------------------------------------------------------
    sources = [
        (uvw_crab, "Crab phase centre"),
        (uvw_sat1, "Closest satellite 1"),
        (uvw_sat2, "Closest satellite 2"),
    ]
    fig_uv, axes_uv = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True, sharey=True)

    for ax, (uvw_arr, title) in zip(axes_uv, sources):
        for i in range(1, len(ant_coords)):
            u_m = uvw_arr[i, :, 0]
            v_m = uvw_arr[i, :, 1]
            ax.plot(u_m, v_m, lw=0.9, label=f"B0-{i}")
            # Hermitian conjugate (mirror)
            ax.plot(-u_m, -v_m, lw=0.9, ls="--", alpha=0.5)

        ax.set_title(f"UV coverage - {title}")
        ax.set_xlabel("u (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    axes_uv[0].set_ylabel("v (m)")
    axes_uv[-1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig("uvw_coverage_subplots.png", dpi=150)
    print("Saved UV coverage subplots to uvw_coverage_subplots.png")
    plt.show()


