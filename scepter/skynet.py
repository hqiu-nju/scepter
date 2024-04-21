#!/usr/bin/env python3

"""
skynet.py

This is the module for generating the sky grid

Author: Harry Qiu <hqiu678@outlook.com>
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

    
def pointgen(
            
            step_size=3 * u.deg,
            lat_range=(0 * u.deg, 90 * u.deg),
            rnd_seed=None,
            ):
        ### sampling of the sky in equal solid angle
        def sample( low_lon, high_lon, low_lat, high_lat):

            z_low, z_high = np.cos(np.radians(90 - low_lat)), np.cos(np.radians(90 - high_lat))
            az = np.random.uniform(low_lon, high_lon)
            el = 90 - np.degrees(np.arccos(
                np.random.uniform(z_low, z_high)
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
                    cell_tel_az, cell_tel_el = sample( low_lon, high_lon, low_lat, high_lat)
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


def plotgrid(val, grid_info,  point_az, point_el,elmin=30, elmax=85):
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
    plt.scatter(point_az,point_el)
    plt.ylim(elmin, elmax)
    plt.show()