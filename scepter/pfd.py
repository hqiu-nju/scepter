#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pfd.py

Call functions for power flux density calculation

Author: Harry Qiu <hqiu678@outlook.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import pycraf
import cysgp4
from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry
from astropy import units as u, constants as const

def sat_gain_func(sat_obs_az, sat_obs_el):
    # Use 0 dBi antenna for the simulations
    
    sat_obs_az, sat_obs_el = np.broadcast_arrays(sat_obs_az, sat_obs_el)
    G_tx = np.zeros(sat_obs_az.shape, dtype=np.float64) * cnv.dBi
    return G_tx

def FPSL(freq, dist):
    # Calculate the free space path loss
    FSPL = cnv.free_space_loss(
        dist * u.km, freq
        ).to(cnv.dB)
    return FSPL

def grx(ang_sep,d_rx,freq,eta_a_rx):
    # this function retrieves the antenna gain at certain angle
    G_rx = antenna.ras_pattern(
        ang_sep, d_rx, const.c / freq, eta_a_rx
        )
    return G_rx

def prx(freq,sat_obs_dist,sat_obs_az,sat_obs_el):
    # Calculate the received power flux density
    FSPL = FPSL(freq, sat_obs_dist)
    G_tx = sat_gain_func(
        sat_obs_az * u.deg,
        sat_obs_el * u.deg,
        )
    
    p_rx = np.sum(p_tx+G_tx+FSPL+G_rx).to_value(u.W)
    return p_rx

