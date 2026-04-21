#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from cysgp4 import PyObserver
from numpy.testing import assert_allclose

from scepter import obs, skynet, uvw


_TEST_TLE_TEXT = """ISS
1 25544U 98067A   25001.00000000  .00016717  00000+0  10270-3 0  9991
2 25544  51.6421 164.6866 0003884 276.1957 170.2534 15.50057708487109
HST
1 20580U 90037B   25001.00000000  .00000850  00000+0  36266-4 0  9992
2 20580  28.4692 116.4136 0002805  23.6494 336.4611 15.09201640374246
"""


def test_compute_uvw_supports_time_and_satellite_axes() -> None:
    antennas = [
        EarthLocation(lon=21.4430 * u.deg, lat=-30.7130 * u.deg, height=1086.0 * u.m),
        EarthLocation(lon=21.4440 * u.deg, lat=-30.7130 * u.deg, height=1086.0 * u.m),
    ]
    obs_times = Time(["2025-01-01T00:00:00", "2025-01-01T00:01:00"], scale="utc")
    ra_deg = np.array([[83.6330, 84.1000], [83.6400, 84.1100]], dtype=np.float64)
    dec_deg = np.array([[22.0140, 22.5000], [22.0200, 22.5100]], dtype=np.float64)

    uvw_all, hour_angles = uvw.compute_uvw(
        antennas,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        obs_times=obs_times,
    )

    assert uvw_all.shape == (2, 2, 2, 3)
    assert hour_angles.shape == (2, 2)
    assert_allclose(uvw_all[0], 0.0, atol=1e-9)


def test_obs_sim_sky_track_respects_observer_index() -> None:
    observers = np.array(
        [
            PyObserver(0.0, 0.0, 0.0),
            PyObserver(90.0, 0.0, 0.0),
        ],
        dtype=object,
    )
    receiver = obs.receiver_info(
        d_rx=13.5 * u.m,
        eta_a_rx=0.7,
        pyobs=observers,
        freq=1420 * u.MHz,
        bandwidth=10 * u.MHz,
    )
    mjd = Time("2025-01-01T00:00:00", scale="utc").mjd
    mjds = np.array([mjd], dtype=np.float64).reshape(1, 1, 1, 1, 1, 1)
    sim = obs.obs_sim(receiver, skynet.pointgen_S_1586_1(niters=1), mjds)

    pointing_0 = sim.sky_track(ra=0.0, dec=0.0, observer_index=0)
    pointing_1 = sim.sky_track(ra=0.0, dec=0.0, observer_index=1)

    assert sim.tracking_observer_index == 1
    assert not np.allclose(
        np.asarray(pointing_0.az.deg, dtype=np.float64),
        np.asarray(pointing_1.az.deg, dtype=np.float64),
    )


def test_build_tracking_uvw_from_array_and_tle_files(tmp_path) -> None:
    array_file = tmp_path / "array.csv"
    array_file.write_text(
        "name,lon_deg,lat_deg,alt_m\n"
        "ref,21.4430,-30.7130,1086.0\n"
        "east,21.4440,-30.7130,1086.0\n",
        encoding="utf-8",
    )
    tle_file = tmp_path / "catalog.tle"
    tle_file.write_text(_TEST_TLE_TEXT, encoding="utf-8")

    mjd_start = Time("2025-01-01T00:00:00", scale="utc").mjd
    mjds = np.array([mjd_start, mjd_start + 1.0 / 86400.0], dtype=np.float64).reshape(1, 1, 1, 1, 2, 1)

    result = uvw.build_tracking_uvw(
        ra_deg=83.633,
        dec_deg=22.014,
        array_file=array_file,
        tle_files=[tle_file],
        mjds=mjds,
        verbose=False,
    )

    assert result.antenna_names == ("ref", "east")
    assert result.satellite_names == ("ISS", "HST")
    assert result.pointing_uvw_m.shape == (2, 2, 3)
    assert result.pointing_hour_angles_rad.shape == (2,)
    assert result.satellite_uvw_m.shape == (2, 2, 2, 3)
    assert result.satellite_hour_angles_rad.shape == (2, 2)
    assert result.satellite_ra_deg.shape == (2, 2)
    assert result.satellite_dec_deg.shape == (2, 2)
    assert result.satellite_separation_deg.shape == (2, 2)
    assert_allclose(result.pointing_uvw_m[0], 0.0, atol=1e-9)
    assert_allclose(result.satellite_uvw_m[0], 0.0, atol=1e-9)
