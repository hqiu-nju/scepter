# Step-by-Step Guide: `obs.py` Simulation and `uvw.py` Array Products

This guide shows how to:

1. define a telescope array in a text file,
2. load TLEs from either an ASCII catalog (for example from CelesTrak) or an
   SKAO archive `.npz`,
3. run a tracked observation with `scepter.obs.obs_sim`, and
4. generate UVW coordinates for the tracked phase centre and the propagated
   satellites with `scepter.uvw`.

The examples below use a small tracked observation, but the same pattern scales
to larger arrays, denser time grids, and bigger TLE catalogs.

## Prerequisites

Use either the lite or full development environment:

```bash
conda activate scepter-dev
```

or:

```bash
conda activate scepter-dev-full
```

The workflow below uses:

- `astropy`
- `cysgp4`
- `pycraf`
- `numpy`

## 1. Prepare an array file

`scepter.uvw.load_telescope_array_file()` reads a plain-text array definition
and returns both:

- `EarthLocation` objects for Astropy/UVW work, and
- matching `cysgp4.PyObserver` objects for satellite propagation.

The simplest format is a CSV with antenna name, longitude, latitude, and
altitude in metres:

```text
name,lon_deg,lat_deg,alt_m
ref,21.4430,-30.7130,1086.0
east,21.4440,-30.7130,1086.0
north,21.4430,-30.7120,1086.0
```

Notes:

- Longitude is east-positive, in degrees.
- Latitude is north-positive, in degrees.
- Altitude is read in metres by default.
- Headerless whitespace-delimited files are also accepted; in that case the
  first three columns are interpreted as `lon lat alt`, and the optional fourth
  column is the antenna name.

Load the file like this:

```python
from astropy import units as u
from scepter import obs, uvw

geometry = uvw.load_telescope_array_file("array.csv")

print(geometry.antenna_names)
print(geometry.pyobservers.shape)      # (N_ant,)
print(len(geometry.earth_locations))   # N_ant

receiver = obs.receiver_info(
    d_rx=13.5 * u.m,
    eta_a_rx=0.7,
    pyobs="array.csv",
    freq=1420 * u.MHz,
    bandwidth=10 * u.MHz,
)
print(receiver.antenna_names)
```

## 2. Define the observation times

`scepter.skynet.plantime()` builds the MJD grid that `obs.obs_sim` expects.

```python
import cysgp4
from astropy import units as u
from scepter import skynet

mjds = skynet.plantime(
    epochs=1,
    cadence=24 * u.hour,
    trange=10 * u.min,
    tint=10 * u.s,
    startdate=cysgp4.PyDateTime(2025, 1, 1, 0, 0, 0.0),
)

print(mjds.shape)
```

For a tracked run, `plantime()` returns an array shaped like:

`(1, 1, 1, epochs, nint, 1)`

where:

- `epochs` is the number of separate observing windows, and
- `nint` is the number of time samples inside each window.

## 3. Load the TLE catalog

### Option A: ASCII TLE file from CelesTrak or another text source

Use `scepter.obs.tle_ascii_to_pytles()` for plain-text 2-line or 3-line TLE
files:

```python
from scepter import obs

tles = obs.tle_ascii_to_pytles("catalog.tle")
print(tles.shape)
```

This is the right path for files downloaded from sources such as
[CelesTrak](https://celestrak.org/).

### Option B: SKAO archive `.npz`

Use `scepter.tlefinder.readtlenpz()` for SKAO archive files:

```python
from scepter import tlefinder

tles = tlefinder.readtlenpz("20250101_000000.npz")
print(tles.shape)
```

Important: this SKAO `.npz` input format is different from the propagation
cache written by `obs_sim.populate(..., save=True)`. The former is an input TLE
catalog; the latter is an output file containing already-propagated satellite
coordinates.

## 4. Build the receiver and simulator

The receiver uses the array file through `geometry.pyobservers`, while the sky
grid and MJD array define the pointing grid and time axis.

```python
from astropy import units as u
from scepter import obs, skynet

receiver = obs.receiver_info(
    d_rx=13.5 * u.m,
    eta_a_rx=0.7,
    pyobs=geometry.pyobservers,
    freq=1420 * u.MHz,
    bandwidth=10 * u.MHz,
)

skygrid = skynet.pointgen_S_1586_1(niters=1)
sim = obs.obs_sim(receiver, skygrid, mjds)
```

## 5. Propagate the satellites

Propagate the TLE catalog with `obs_sim.populate()`:

```python
sim.populate(
    tles,
    save=True,
    savename="satellite_info.npz",
    verbose=False,
)
```

This populates:

- `sim.topo_pos_az`
- `sim.topo_pos_el`
- `sim.topo_pos_dist`
- `sim.satf_az`
- `sim.satf_el`
- `sim.satf_dist`

For a tracked observation with one epoch grid, the propagation arrays usually
have shape:

`(N_ant, 1, 1, epochs, nint, N_sat)`

If you want to reuse the propagation later without recomputing it:

```python
sim2 = obs.obs_sim(receiver, skygrid, mjds)
sim2.load_propagation("satellite_info.npz")
```

## 6. Track a source and inspect the geometry

Track a fixed ICRS phase centre with `sky_track()`:

```python
sim.sky_track(ra=83.633 * u.deg, dec=22.014 * u.deg)
```

Then compute the angular separation between the pointing direction and every
satellite:

```python
ang_sep = sim.sat_separation(mode="tracking")
print(ang_sep.shape)
```

If you want to discard satellites that remain below an elevation threshold,
filter after propagation:

```python
sim.reduce_sats(el_limit=0.0)
```

## 7. Compute a simple received-power product

`obs.py` separates the transmitter model, the receiver gain, and the final
received-power conversion.

The example below uses boresight transmit gain as a simple upper-bound case. If
you have an explicit satellite beam direction, replace that with a more
realistic `tx.satgain1d(...)` or `sim.txbeam_angsep(...)` workflow.

```python
from pycraf import conversions as cnv

tx = obs.transmitter_info(
    p_tx_carrier=10 * cnv.dBW,
    carrier_bandwidth=125 * u.kHz,
    duty_cycle=1.0,
    d_tx=0.5 * u.m,
    freq=1420 * u.MHz,
)

tx.power_tx(receiver.bandwidth)
tx.satgain1d(0 * u.deg)

g_rx = receiver.antgain1d(
    sim.pnt_az,
    sim.pnt_el,
    sim.topo_pos_az,
    sim.topo_pos_el,
    verbose=False,
)

p_rx = obs.prx_cnv(
    tx.fspl(sim.topo_pos_dist * u.km),
    g_rx,
    outunit=u.W,
)

print(p_rx.shape)
```

At this point you have a full tracked `obs.py` run driven by:

- an external array file,
- an ASCII or SKAO-archive TLE catalog,
- a receiver model,
- propagated satellite geometry, and
- a basic received-power estimate.

## 8. Create UVW products with `uvw.py`

### Fast path: `build_tracking_uvw()` from an array file plus ASCII TLE files

`scepter.uvw.build_tracking_uvw()` is the highest-level helper for generating:

- UVW coordinates for the fixed tracked phase centre, and
- UVW coordinates for each propagated satellite track.

Example:

```python
from scepter import uvw

result = uvw.build_tracking_uvw(
    ra_deg=83.633,
    dec_deg=22.014,
    array_file="array.csv",
    tle_files=["catalog.tle"],
    mjds=mjds,
    ref_index=0,
    elevation_limit_deg=0.0,
    save_propagation=False,
    verbose=False,
)

print(result.antenna_names)
print(result.satellite_names)
print(result.pointing_uvw_m.shape)
print(result.satellite_uvw_m.shape)
```

Key outputs:

| Field | Shape | Units | Meaning |
| --- | --- | --- | --- |
| `pointing_uvw_m` | `(N_ant, T, 3)` | m | UVW coordinates for the fixed RA/Dec phase centre |
| `pointing_hour_angles_rad` | `(T,)` | rad | Hour angle of the fixed phase centre |
| `satellite_uvw_m` | `(N_ant, T, N_sat, 3)` | m | UVW coordinates for each satellite track |
| `satellite_hour_angles_rad` | `(T, N_sat)` | rad | Hour angle of each satellite track |
| `satellite_ra_deg` | `(T, N_sat)` | deg | ICRS right ascension of each propagated satellite |
| `satellite_dec_deg` | `(T, N_sat)` | deg | ICRS declination of each propagated satellite |
| `satellite_separation_deg` | `(T, N_sat)` | deg | Separation between each satellite and the requested phase centre |

Here `T = mjds.size`, because the helper flattens the MJD grid to a single time
axis before calling `compute_uvw()`.

### Lower-level path: `compute_uvw()` directly

If you already have antenna locations and an observation-time array, you can
skip the builder and call `compute_uvw()` directly.

For a fixed phase centre:

```python
import numpy as np
from astropy.time import Time

obs_times = Time(np.asarray(mjds, dtype=np.float64).ravel(), format="mjd", scale="utc")

pointing_uvw_m, pointing_ha = uvw.compute_uvw(
    list(geometry.earth_locations),
    ra_deg=83.633,
    dec_deg=22.014,
    obs_times=obs_times,
    ref_index=0,
)
```

This returns:

- `pointing_uvw_m` with shape `(N_ant, T, 3)`, and
- `pointing_ha` with shape `(T,)`.

## 9. UVW workflow when the TLE source is an SKAO archive `.npz`

`uvw.build_tracking_uvw()` currently accepts ASCII TLE files. If your orbit
input comes from the SKAO archive `.npz` format, use `obs.py` for propagation
and then call `uvw.compute_uvw()` on the propagated satellite tracks.

The pattern is:

```python
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, ICRS, SkyCoord
from astropy.time import Time
from scepter import obs, skynet, tlefinder, uvw

geometry = uvw.load_telescope_array_file("array.csv")
tles = tlefinder.readtlenpz("20250101_000000.npz")

receiver = obs.receiver_info(
    d_rx=13.5 * u.m,
    eta_a_rx=0.7,
    pyobs=geometry.pyobservers,
    freq=1420 * u.MHz,
    bandwidth=10 * u.MHz,
)

skygrid = skynet.pointgen_S_1586_1(niters=1)
sim = obs.obs_sim(receiver, skygrid, mjds)
sim.populate(tles, save=False, verbose=False)
sim.sky_track(ra=83.633 * u.deg, dec=22.014 * u.deg)

obs_times = Time(np.asarray(mjds, dtype=np.float64).ravel(), format="mjd", scale="utc")
antennas = list(geometry.earth_locations)

pointing_uvw_m, pointing_ha = uvw.compute_uvw(
    antennas,
    ra_deg=83.633,
    dec_deg=22.014,
    obs_times=obs_times,
    ref_index=0,
)

ref_location = geometry.earth_locations[0]
topo_az = np.asarray(sim.topo_pos_az[0, 0, 0, 0, :, :], dtype=np.float64)
topo_el = np.asarray(sim.topo_pos_el[0, 0, 0, 0, :, :], dtype=np.float64)

sat_altaz = SkyCoord(
    az=topo_az * u.deg,
    alt=topo_el * u.deg,
    frame=AltAz(obstime=obs_times[:, np.newaxis], location=ref_location),
)
sat_icrs = sat_altaz.transform_to(ICRS())

satellite_uvw_m, satellite_ha = uvw.compute_uvw(
    antennas,
    ra_deg=np.asarray(sat_icrs.ra.deg, dtype=np.float64),
    dec_deg=np.asarray(sat_icrs.dec.deg, dtype=np.float64),
    obs_times=obs_times,
    ref_index=0,
)

print(pointing_uvw_m.shape)   # (N_ant, T, 3)
print(satellite_uvw_m.shape)  # (N_ant, T, N_sat, 3)
```

This is the same sequence used internally by `TrackingUvwBuilder`, except that
the TLE input comes from `tlefinder.readtlenpz()` instead of
`obs.tle_ascii_to_pytles()`.

## 10. Sanity checks

Useful quick checks after running the workflow:

```python
print(geometry.antenna_names)
print(sim.topo_pos_az.shape)
print(sim.topo_pos_el.min(), sim.topo_pos_el.max())
print(result.pointing_uvw_m[0])      # should be all zeros for ref_index=0
print(result.satellite_uvw_m[0])     # same for the reference antenna
```

Expected invariants:

- `pointing_uvw_m[ref_index]` is identically zero.
- `satellite_uvw_m[ref_index]` is identically zero.
- `sim.topo_pos_dist` is in kilometres.
- UVW outputs are in metres.

## 11. Summary

For an `obs.py`-driven tracked simulation:

1. Load the array file with `uvw.load_telescope_array_file()`.
2. Generate `mjds` with `skynet.plantime()`.
3. Load TLEs with either `obs.tle_ascii_to_pytles()` or
   `tlefinder.readtlenpz()`.
4. Create `receiver_info`, `skygrid`, and `obs_sim`.
5. Run `sim.populate(...)`, then `sim.sky_track(...)`.
6. Use `sat_separation()`, `antgain1d()`, `transmitter_info`, and `prx_cnv()`
   for the observation-side analysis.
7. Use `uvw.build_tracking_uvw()` for the ASCII-TLE path, or
   `uvw.compute_uvw()` directly for custom or SKAO-archive workflows.
