#!/usr/bin/env python3

"""
skynet.py

This is the module for generating the sky grid

Author: Harry Qiu <hqiu678@outlook.com>
Collaborator: Boris Sorokin <mralin@protonmail.com>
Date: 12-03-2024
"""

import cysgp4
from cysgp4 import PyTle, PyObserver
from cysgp4 import get_example_tles, propagate_many
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pycraf
from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry
from astropy import units as u, constants as const
from astropy.time import Time
from astropy.utils.misc import NumpyRNGContext
from functools import lru_cache

def pointgen_S_1586_1(niters=1, rnd_seed=None):
    """
    Generates random pointing directions within sky cells defined in ITU-R S.1586-1 Table 1.

    This function creates `niters` random pointings (azimuth, elevation)
    for each sky cell defined in Recommendation ITU-R S.1586-1, ensuring
    compliance with the cell structure specified in Table 1 of the Recommendation.
    The sampling within each cell ensures a uniform distribution over solid angle.

    All S.1586-1 definitions, calculations, and derived arrays needed for
    sampling are handled internally by a cached helper function (`_get_cached_data`).
    This ensures the setup calculation happens only once per Python session.

    Parameters
    ----------
    niters : int
        The number of random pointings (iterations) to generate *per cell*.
    rnd_seed : int, optional
        A seed for the random number generator for reproducible results.
        If None, the generator is initialized randomly.

    Returns
    -------
    tel_az_deg : numpy.ndarray
        Array of generated azimuth angles in degrees.
        Shape: (niters, n_cells), where n_cells is the total number of
        S.1586-1 cells (exactly 2334).
    tel_el_deg : numpy.ndarray
        Array of generated elevation angles in degrees.
        Shape: (niters, n_cells).
    None : returned for consistency with number of pointgen outputs

    Raises
    ------
    ValueError
        If `niters` is not a positive integer.

    Notes
    -----
    - Internal caching ensures efficient repeated calls.
    - Sampling uses uniform distribution in azimuth and cos(zenith angle).
    - Output angles are in degrees.
    - Cell structure strictly follows Rec. ITU-R S.1586-1, Table 1.
    """
    if not isinstance(niters, int) or niters <= 0:
        raise ValueError("niters must be a positive integer.")

    # --- Internal Cached Function for S.1586-1 Data ---
    @lru_cache(maxsize=1)
    def _get_cached_data():
        """
        Internal helper to calculate and cache S.1586-1 data.

        Defines constants, calculates cell boundaries, reshapes for broadcasting,
        and computes cos(zenith) bounds, strictly following Table 1 of Rec. ITU-R S.1586-1.
        This function's result is cached.

        Returns:
        tuple : (cell_az_low_bc, cell_az_high_bc, z_bound_lower_bc,
                 z_bound_upper_bc, n_total_cells)
        """

        # Define S.1586-1 constants locally, matching Table 1 exactly.
        # Keys are the lower elevation edge of the 3-degree ring.
        # Values are the azimuth step in degrees for that ring.
        s1586_az_steps = {
              0: 3,   3: 3,   6: 3,   9: 3,  12: 3,  15: 3,  18: 3,  21: 3,  24: 3, # Rings 0-27 deg Lower El
             27: 3,  # Ring 27-30 deg Lower El
             30: 4,  33: 4,  36: 4,  39: 4,  42: 4, # Rings 30-45 deg Lower El
             45: 4,  # Ring 45-48 deg Lower El
             48: 5,  51: 5, # Rings 48-54 deg Lower El
             54: 5,  # Ring 54-57 deg Lower El
             57: 6,  60: 6, # Rings 57-63 deg Lower El
             63: 6,  # Ring 63-66 deg Lower El
             66: 8,  # Ring 66-69 deg Lower El
             69: 9,  # Ring 69-72 deg Lower El
             72: 10, # Ring 72-75 deg Lower El
             75: 12, # Ring 75-78 deg Lower El
             78: 18, # Ring 78-81 deg Lower El
             81: 24, # Ring 81-84 deg Lower El
             84: 40, # Ring 84-87 deg Lower El
             87: 120 # Ring 87-90 deg Lower El
        }
        s1586_el_edges_deg = np.arange(0, 90 + 3, 3) # 0, 3, ..., 87, 90
        n_rings = len(s1586_el_edges_deg) - 1 # Should be 30 rings

        # --- Calculate basic cell boundaries ---
        cell_az_low_list = []
        cell_az_high_list = []
        cell_el_low_list = []
        cell_el_high_list = []
        n_total_cells = 0
        calculated_cells_per_ring = [] # For verification

        for i in range(n_rings):
            el_low_deg = s1586_el_edges_deg[i]
            el_high_deg = s1586_el_edges_deg[i+1]
            # Lookup the step size directly from the corrected dictionary
            az_step_deg = s1586_az_steps.get(el_low_deg)
            if az_step_deg is None:
                 # This should not happen now with the complete dictionary
                 raise ValueError(f"Implementation Error: Azimuth step undefined for el_low={el_low_deg} deg.")

            # Check if step divides 360 degrees
            if 360 % az_step_deg != 0:
                # This should not happen with valid S.1586-1 steps
                raise ValueError(f"Invalid Azimuth step {az_step_deg} (does not divide 360) for el={el_low_deg}-{el_high_deg}.")

            n_cells_in_ring = 360 // az_step_deg
            calculated_cells_per_ring.append(n_cells_in_ring) # Store for verification
            az_edges_deg = np.arange(n_cells_in_ring + 1) * az_step_deg

            for j in range(n_cells_in_ring):
                cell_az_low_list.append(az_edges_deg[j])
                cell_az_high_list.append(az_edges_deg[j+1])
                cell_el_low_list.append(el_low_deg)
                cell_el_high_list.append(el_high_deg)
            n_total_cells += n_cells_in_ring

        # --- Verification against Table 1 ---
        # Expected cells per ring from Table 1
        expected_cells_per_ring_table1 = [
            120]*10 + [90]*6 + [72]*3 + [60]*3 + [45, 40, 36, 30, 20, 15, 9, 3]
        if calculated_cells_per_ring != expected_cells_per_ring_table1:
            raise RuntimeError("Implementation Error: Calculated cells per ring do not match S.1586-1 Table 1.")
        if n_total_cells != 2334:
            raise RuntimeError(f"Implementation Error: Calculated total cells ({n_total_cells}) do not match S.1586-1 expected (2334).")
        # --- End Verification ---

        cell_az_low_deg = np.array(cell_az_low_list, dtype=np.float64)
        cell_az_high_deg = np.array(cell_az_high_list, dtype=np.float64)
        cell_el_low_deg = np.array(cell_el_low_list, dtype=np.float64)
        cell_el_high_deg = np.array(cell_el_high_list, dtype=np.float64)

        # --- Reshape boundaries for broadcasting ---
        cell_az_low_bc = cell_az_low_deg[np.newaxis, :]
        cell_az_high_bc = cell_az_high_deg[np.newaxis, :]
        cell_el_low_bc = cell_el_low_deg[np.newaxis, :] # Needed for z calc below
        cell_el_high_bc = cell_el_high_deg[np.newaxis, :] # Needed for z calc below

        # --- Calculate broadcasted cos(zenith) boundaries ---
        z_cos_for_el_low = np.cos(np.radians(90.0 - cell_el_low_bc))
        z_cos_for_el_high = np.cos(np.radians(90.0 - cell_el_high_bc))
        z_bound_upper_bc = np.maximum(z_cos_for_el_low, z_cos_for_el_high)
        z_bound_lower_bc = np.minimum(z_cos_for_el_low, z_cos_for_el_high)

        return (cell_az_low_bc, cell_az_high_bc,
                z_bound_lower_bc, z_bound_upper_bc,
                n_total_cells)
    # --- End of Internal Cached Function ---

    # --- Get Pre-calculated & Cached Data ---
    cell_az_low_bc, cell_az_high_bc, \
    z_bound_lower_bc, z_bound_upper_bc, \
    n_total_cells = _get_cached_data()

    # --- Generate Random Numbers ---
    with NumpyRNGContext(rnd_seed):
        rand_az_uniform = np.random.uniform(0.0, 1.0, size=(niters, n_total_cells))
        rand_z_uniform = np.random.uniform(0.0, 1.0, size=(niters, n_total_cells))

    # --- Vectorized Azimuth Calculation ---
    tel_az_deg = cell_az_low_bc + rand_az_uniform * (cell_az_high_bc - cell_az_low_bc)

    # --- Vectorized Elevation Calculation (via Zenith Angle) ---
    sampled_z = z_bound_lower_bc + rand_z_uniform * (z_bound_upper_bc - z_bound_lower_bc)
    sampled_z = np.clip(sampled_z, -1.0, 1.0)
    tel_el_deg = 90.0 - np.degrees(np.arccos(sampled_z))

    return tel_az_deg, tel_el_deg, None
    
def pointgen(
            niters,
            step_size=3 * u.deg,
            lat_range=(0 * u.deg, 90 * u.deg),
            rnd_seed=None,
            ):
        ### sampling of the sky in equal solid angle
        def sample(niters,low_lon, high_lon, low_lat, high_lat):

            z_low, z_high = np.cos(np.radians(90 - low_lat)), np.cos(np.radians(90 - high_lat))
            az = np.random.uniform(low_lon, high_lon,size=niters)
            el = 90 - np.degrees(np.arccos(
                np.random.uniform(z_low, z_high,size=niters)
                ))
            return az, el

        cell_edges, cell_mids, solid_angles, tel_az, tel_el = [], [], [], [], []

        lat_range = (lat_range[0].to_value(u.deg), lat_range[1].to_value(u.deg))
        ncells_lat = int(
            (lat_range[1] - lat_range[0]) / step_size.to_value(u.deg) + 0.5
            )
        edge_lats = np.linspace(
            lat_range[0], lat_range[1], ncells_lat + 1, endpoint=True
            )
        mid_lats = 0.5 * (edge_lats[1:] + edge_lats[:-1])

        with NumpyRNGContext(rnd_seed):
            for low_lat, mid_lat, high_lat in zip(edge_lats[:-1], mid_lats, edge_lats[1:]):

                ncells_lon = int(360 * np.cos(np.radians(mid_lat)) / step_size.to_value(u.deg) + 0.5)
                edge_lons = np.linspace(0, 360, ncells_lon + 1, endpoint=True)
                mid_lons = 0.5 * (edge_lons[1:] + edge_lons[:-1])

                solid_angle = (edge_lons[1] - edge_lons[0]) * np.degrees(
                    np.sin(np.radians(high_lat)) - np.sin(np.radians(low_lat))
                    )
                for low_lon, mid_lon, high_lon in zip(edge_lons[:-1], mid_lons, edge_lons[1:]):
                    cell_edges.append((low_lon, high_lon, low_lat, high_lat))
                    cell_mids.append((mid_lon, mid_lat))
                    solid_angles.append(solid_angle)
                    cell_tel_az, cell_tel_el = sample(niters, low_lon, high_lon, low_lat, high_lat)
                    tel_az.append(cell_tel_az)
                    tel_el.append(cell_tel_el)

        tel_az = np.array(tel_az).T  # TODO, return u.deg
        tel_el = np.array(tel_el).T

        grid_info = np.column_stack([cell_mids, cell_edges, solid_angles])
        grid_info.dtype = np.dtype([  # TODO, return a QTable
            ('cell_lon', np.float64), ('cell_lat', np.float64),
            ('cell_lon_low', np.float64), ('cell_lon_high', np.float64),
            ('cell_lat_low', np.float64), ('cell_lat_high', np.float64), 
            ('solid_angle', np.float64), 
            ])
        
        return tel_az, tel_el, grid_info[:, 0]

def gridmatch(az,el,grid_info):
    ### get the grid for a fixed list of az and el pointings
    grid_indx=[]
    for i,j,k,l in zip(grid_info['cell_lon_low'],grid_info['cell_lon_high'],grid_info['cell_lat_low'],grid_info['cell_lat_high']):
        azmask=(az >= i) * (az <= j) 
        elmask=(el >= k) * (el <= l)
        mask=azmask & elmask
        grid_indx.append(mask)
    grid_indx=np.array(grid_indx)
    used_grids=np.where(grid_indx.sum(1)>0)[0]
    return used_grids,grid_indx[used_grids]

def plantime(epochs,cadence,trange,tint,startdate=cysgp4.PyDateTime()):
    '''
    Description: This function generates the time grid for the simulation

    Parameters:

    epochs: astropy quantity
        number of time steps
    cadence: astropy quantity
        cadence between epochs
    trange: astropy quantity
        time range of the simulation
    tint: astropy quantity  
        sample integration time of the simulation 
    startdate: cysgp4 PyDateTime object
        start date of the simulation, default cysgp4.PyDateTime() for current date and time

    Returns:
    mjds: numpy array
        a 2d array of time intervals for the simulation, first dimension is the number of epochs, 
        second dimension is the separate time stamps for each integration time sample, in MJD
    '''
    pydt = startdate ## take current date and time
    start_mjd=pydt.mjd  ## get mjd step
    niters = epochs

    start_times_window = cadence

    time_range, time_resol = trange.to_value(u.s), tint.to_value(u.s)  # seconds


    start_times = start_mjd + np.arange(epochs) * start_times_window.to_value(u.day)
    td = np.arange(0, time_range, time_resol) *u.s
    td = td.to_value(u.day)
    mjds = np.array(start_times[np.newaxis,np.newaxis,np.newaxis, :, np.newaxis,np.newaxis] + 
                td[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis])
    return mjds

def plotgrid(val, grid_info,  point_az=[], point_el=[],elmin=30, elmax=85,zlabel='PFD average / cell [dB(W/m2)]',xlabel='Azimuth [deg]',ylabel='Elevation [deg]',azmin=0,azmax=360):
    fig = plt.figure(figsize=(12, 4))
    # val = pfd_avg.to_value(cnv.dB_W_m2)
    vmin, vmax = val.min(), val.max()
    val_norm = (val - vmin) / (vmax - vmin)
    plt.bar(
        grid_info['cell_lon_low'],
        height=grid_info['cell_lat_high'] - grid_info['cell_lat_low'],
        width=grid_info['cell_lon_high'] - grid_info['cell_lon_low'],
        bottom=grid_info['cell_lat_low'],
        color=plt.cm.viridis(val_norm),
        align='edge'
        )
    plt.scatter(point_az,point_el,c='r',s=1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(zlabel)
    plt.ylim(elmin, elmax)
    plt.xlim(azmin,azmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show() ### don't show here just load figure into matplotlib
