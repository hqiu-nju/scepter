Simulation Workflows
====================

SCEPTer is built around a few repeatable analysis patterns. The modules are
fairly independent, but most practical studies follow one of the workflows
below.

Observation simulation
----------------------

Use this workflow when you want a sky-based RFI or EPFD analysis at one or more
radio telescope locations.

1. Create telescope observers with ``cysgp4.PyObserver``.
2. Build a sky grid with ``scepter.skynet.pointgen_S_1586_1`` or
   ``scepter.skynet.pointgen``.
3. Define observation times with ``scepter.skynet.plantime`` or another MJD
   array.
4. Configure the receiver with ``scepter.obs.receiver_info``.
5. Load archived TLEs with ``scepter.tlefinder.TLEfinder`` or generate synthetic
   constellations with ``scepter.tleforger``.
6. Create ``scepter.obs.obs_sim`` and populate the satellite propagation state.
7. Run pointing, separation, gain, and power calculations for the scenario of
   interest.

Typical objects involved:

- ``scepter.obs.receiver_info``
- ``scepter.obs.transmitter_info``
- ``scepter.obs.obs_sim``
- ``scepter.skynet.pointgen_S_1586_1``
- ``scepter.tlefinder.TLEfinder``

Minimal sketch
^^^^^^^^^^^^^^

.. code-block:: python

   from astropy import units as u
   from cysgp4 import PyObserver
   from scepter import obs, skynet

   observer = PyObserver(lon=21.443888, lat=-30.713055, alt=1000)
   receiver = obs.receiver_info(
       d_rx=13.5 * u.m,
       eta_a_rx=0.7,
       pyobs=observer,
       freq=1420 * u.MHz,
       bandwidth=10 * u.MHz,
   )

   skygrid = skynet.pointgen_S_1586_1(niters=1)
   mjds = ...
   sim = obs.obs_sim(receiver, skygrid, mjds)
   sim.populate(tles_list)
   sim.sky_track(ra=0 * u.deg, dec=45 * u.deg)

Constellation geometry and gridding
-----------------------------------

Use the gridding helpers when you need Earth-surface coverage, beam footprints,
or satellite-to-cell matching outside the full observation simulator.

- ``scepter.earthgrid.calculate_footprint_size`` estimates beam impact area on
  the Earth.
- ``scepter.earthgrid.generate_hexgrid_full`` builds global hexagonal sampling
  grids.
- ``scepter.earthgrid.trunc_hexgrid_to_impactful`` trims the grid to cells that
  can be illuminated for a given geometry.
- ``scepter.satsim.compute_sat_cell_links_auto`` assigns visible satellites to
  ground cells with a CPU or Numba-accelerated implementation.

Interferometry and baseline analysis
------------------------------------

The repository contains two layers of interferometry utilities.

- ``scepter.obs`` includes baseline and fringe helpers integrated into the main
  observation simulator.
- ``scepter.array`` exposes baseline, delay, and fringe calculations as a
  standalone utility module.
- ``scepter.uvw`` adds explicit UVW coordinate transforms for radio
  interferometry workflows.

This split is useful when you need baseline products without running the full
sky simulation.

Persisting results
------------------

For longer simulation runs, use ``scepter.scenario`` to batch work and store
outputs:

- ``scepter.scenario.generate_simulation_batches`` partitions long time ranges
- ``scepter.scenario.write_data`` stores results to HDF5
- ``scepter.scenario.read_data`` reconstructs saved arrays with units and time
  metadata
