"""
satsim.py

This module provides supplemental functions to model satellite system behavior.

Author: boris.sorokin <mralin@protonmail.com>
Date: 16-04-2025
"""

import numpy as np
from astropy import units as u
from pycraf.utils import ranged_quantity_input

@ranged_quantity_input(
    min_elevation=(0, 90, u.deg),
    strip_input_units=False,
    allow_none=False
)
def compute_sat_cell_links(sat_topo, 
                           min_elevation=30*u.deg, 
                           Nco=1, 
                           cell_observer_offset=1):
    """
    Compute the communication matrix for cells by selecting Nco satellites per cell at every timestamp.
    
    This function uses vectorized operations (np.where, np.argpartition, and np.argsort) to assign random
    weights to visible satellites and then selects the Nco satellites with the smallest random weights. 
    Non-visible satellites are masked (set to an infinite weight) so that they are never chosen.

    The function accepts both 4D and 3D inputs for sat_topo. If sat_topo is provided as a 3D array with shape
    (n_observers, n_satellites, n_parameters), it is automatically expanded to a 4D array with a single time step,
    and the output will be returned as a 2D array of shape (n_cells, Nco).

    Parameters
    ----------
    sat_topo : numpy.ndarray
        The propagation result array from cysgp4.propagate_many.
        Expected shape is (n_time, n_observers, n_satellites, n_parameters), or can be provided as a 3D array 
        (n_observers, n_satellites, n_parameters) for a single time instance.
        For each observer, the elevation angle (in degrees) is assumed to be located at index 1 
        along the last axis.
    min_elevation : astropy.units.Quantity
        The minimum elevation threshold (e.g., 30*u.deg) used to consider a satellite as visible.
    Nco : int
        The number of satellites to select per cell.
    cell_observer_offset : int, optional
        The starting observer index for cells. Assumes that indexes [:cell_observer_offset] correspond 
        to the RAS stations and indices starting at cell_observer_offset correspond to the hexagonal cell 
        centers. The default is 1 for the case of one RAS station and cells. Set to 0 if sat_topo was calculated 
        only for cells or adjust according to the number of RAS stations.
    
    Returns
    -------
    communication_all : numpy.ndarray
        If the input sat_topo is 4D, returns a NumPy array of shape (n_time, n_cells, Nco) where each element 
        contains the index of a satellite chosen for communication.
        If the input sat_topo is 3D, returns a NumPy array of shape (n_cells, Nco).
        If fewer than Nco satellites are visible for a given (time, cell), the remaining entries are filled with -1.
        
    Notes
    -----
    This function randomly assigns a weight to each satellite at every timestamp for each cell 
    using np.random.random. Non-visible satellites (i.e. where elevation <= min_elevation) are masked 
    by setting their weight to infinity. The function then uses np.argpartition to efficiently find 
    the indices of the smallest Nco weights, and finally sorts these indices so that the selected 
    satellites are in increasing order (if desired).
    """
    # Check input dimensions; if sat_topo is 3D, add a time axis.
    original_ndim = sat_topo.ndim
    if original_ndim == 3:
        sat_topo = sat_topo[np.newaxis, ...]  # Expand dims to shape (1, n_observers, n_satellites, n_parameters)
    
    # Extract cells' sat_topo data: assume that the first observer corresponds to the RAS station
    # and the remaining observers (indices starting from cell_observer_offset) correspond to cells.
    # This produces an array with shape: (n_time, n_cells, n_satellites, n_parameters)
    cells_sat_topo = sat_topo[:, cell_observer_offset:, :, :]
    
    # Determine satellite visibility based on elevation (index 1) and the minimum threshold.
    visibility = cells_sat_topo[..., 1] > min_elevation.to(u.deg).value  # Shape: (n_time, n_cells, n_satellites)
    
    # Get the dimensions for clarity
    n_time, n_cells, n_sat = visibility.shape
    
    # Generate random weights for every (time, cell, satellite)
    random_weights = np.random.random((n_time, n_cells, n_sat))
    # Mask non-visible satellites by assigning an infinite weight so that they are never chosen.
    random_weights_masked = np.where(visibility, random_weights, np.inf)
    
    # Select the indices corresponding to the Nco smallest weights along the satellite axis.
    # If there are fewer than Nco visible satellites, extra indices will correspond to infinite weights.
    selected_indices_unsorted = np.argpartition(random_weights_masked, kth=Nco-1, axis=-1)[:, :, :Nco]
    
    # Retrieve the corresponding weights for the selected indices.
    selected_weights = np.take_along_axis(random_weights_masked, selected_indices_unsorted, axis=-1)
    # Sort the selected indices along the satellite axis for clarity.
    sort_order = np.argsort(selected_weights, axis=-1)
    selected_indices = np.take_along_axis(selected_indices_unsorted, sort_order, axis=-1)
    
    # Prepare the output array initialized with -1 (to denote invalid selections)
    sat_cell_links = np.full((n_time, n_cells, Nco), fill_value=-1, dtype=int)
    
    # Determine valid indices where the weight is not infinite.
    selected_weights_sorted = np.sort(selected_weights, axis=-1)
    valid_mask = selected_weights_sorted < np.inf  # Boolean mask: True for valid (visible) satellites
    
    # Fill the final communication matrix with valid satellite indices.
    sat_cell_links[valid_mask] = selected_indices[valid_mask]
    
    # If the input was originally 3D, remove the added time dimension.
    if original_ndim == 3:
        sat_cell_links = np.squeeze(sat_cell_links, axis=0)
    
    return sat_cell_links
