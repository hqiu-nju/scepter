Module Guide
============

The ``scepter`` package is organised as a set of domain-focused modules rather
than a single monolithic API. The table below is a practical map of the
repository.

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Module
     - Purpose
     - Key objects and functions
   * - ``scepter.obs``
     - Core observation simulation and interference calculations.
     - ``receiver_info``, ``transmitter_info``, ``obs_sim``, ``prx_cnv``,
       ``pfd_to_Jy``
   * - ``scepter.skynet``
     - Upper-hemisphere sky tessellation, observation grids, and time planning.
     - ``pointgen_S_1586_1``, ``pointgen``, ``gridmatch``, ``plantime``,
       ``plotgrid``
   * - ``scepter.antenna``
     - ITU-R satellite and telescope antenna pattern helpers.
     - ``s_1528_rec1_2_pattern``, ``calculate_beamwidth_1d``,
       ``s_1528_rec1_4_pattern_amend``
   * - ``scepter.earthgrid``
     - Earth-surface footprint calculations and hexagonal cell generation.
     - ``calculate_footprint_size``, ``generate_hexgrid_full``,
       ``trunc_hexgrid_to_impactful``, ``recommend_cell_diameter``
   * - ``scepter.satsim``
     - Satellite-to-cell visibility assignment and beam-link bookkeeping.
     - ``compute_sat_cell_links``, ``compute_sat_cell_links_parallel``,
       ``compute_sat_cell_links_auto``
   * - ``scepter.scenario``
     - Batch execution utilities and HDF5 persistence for simulation outputs.
     - ``generate_simulation_batches``, ``write_data``, ``read_data``,
       ``analyse_time``, ``process_integration``
   * - ``scepter.tlefinder``
     - Archived TLE discovery and propagation helpers.
     - ``TLEfinder``, ``id_locator``, ``parse_sgp4info``, ``readtlenpz``
   * - ``scepter.tleforger``
     - Synthetic TLE generation for controlled constellation studies.
     - ``forge_tle_single``, ``forge_tle_belt``, ``reset_tle_counter``
   * - ``scepter.uvw``
     - UVW coordinate transforms for radio interferometry.
     - ``hour_angle``, ``itrf_to_enu``, ``enu_to_uvw``,
       ``compute_uvw``
   * - ``scepter.array``
     - Standalone baseline geometry, delay, and fringe utilities.
     - ``baseline_bearing``, ``baseline_pairs``, ``baseline_vector``,
       ``fringe_attenuation``, ``bw_fringe``
   * - ``scepter.visualise``
     - Plotting utilities for distributions and hemisphere views.
     - ``plot_cdf_ccdf``, ``plot_hemisphere_2D``, ``plot_hemisphere_3D``
   * - ``scepter.angle_sampler``
     - Joint angle sampling, histogram comparison, and distribution metrics.
     - ``JointAngleSampler``, ``joint_hist_metrics``, ``sliced_wasserstein``
   * - ``scepter.gpu_accel``
     - CUDA-based accelerators for large angular-distance and propagation tasks.
     - ``true_angular_distance_gpu_nd``, ``true_angular_distance_auto``,
       ``gpu_topo``, ``gpu_sat_azel``, ``propagate_many_gpu``
   * - ``scepter.gpu_accel_dev``
     - Experimental and validation helpers for GPU acceleration work.
     - ``true_angular_distance_CUDA``, ``_self_test_small``

Suggested reading order
-----------------------

If you are new to the codebase, start with the modules in this order:

1. ``scepter.skynet`` to understand how the sky is discretised.
2. ``scepter.obs`` for the main simulation data model.
3. ``scepter.tlefinder`` or ``scepter.tleforger`` depending on where the orbit
   data comes from.
4. ``scepter.scenario`` for storage and batch execution.
5. ``scepter.visualise`` and ``scepter.uvw`` for downstream analysis products.
