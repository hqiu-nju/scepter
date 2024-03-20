#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pfd.py

Call functions for power flux density calculation

At the moment I will use a dummy isotropic antenna gain with cysgp4 and pycraf 

Author: Harry Qiu <hqiu678@outlook.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import pycraf
import cysgp4
from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry
from astropy import units as u, constants as const

def dummy_gain(sat_obs_az,sat_obs_el,d_tx,freq,tp_az,tp_el):
    '''
    Description: Retrieves gain from basic gain pattern function with angular separation to pointing considered

    Parameters:
    sat_obs_az: float
        azimuth angle of the satellite
    sat_obs_el: float
        elevation angle of the satellite
    d_tx: float
        diameter of the transmitter telescope
    freq: float
        frequency of the signal
    tp_az: float
        azimuth angle of the telescope pointing
    tp_el: float
        elevation angle of the telescope pointing

    Returns:
    G_tx: float
        transmitter gain
    '''
    gmax=antenna.fl_G_max_from_size(d_tx,freq)
    hpbw=antenna.fl_hpbw_from_size(d_tx,freq)  
    ### calculate angular separation of satellite to telescope pointing
    phi=geometry.angular_separation(tp_az,tp_el,sat_obs_az,sat_obs_el) 
    flpattern=antenna.fl_pattern(phi,diameter=d_tx,frequency=freq,G_max=gmax)

    return G_tx


def grx(ang_sep,d_rx,freq,eta_a_rx):
    '''
    Description: This function calculates the receiver gain

    Parameters:
    ang_sep: float
        angular separation between observing direction and telescope pointing
    d_rx: float
        diameter of the receiver telescope
    freq: float
        frequency of the signal
    eta_a_rx: float
        aperture efficiency of the receiver telescope
    '''
    
    G_rx = antenna.ras_pattern(
        ang_sep, d_rx, const.c / freq, eta_a_rx
        )
    return G_rx


def ptx(p_tx_carrier,carrier_bandwidth,duty_cycle,ras_bandwidth):
    # Calculate the transmitted power
    p_tx_nu_peak = (
    p_tx_carrier.physical / carrier_bandwidth
    ).to(u.W / u.Hz)
    p_tx_nu = p_tx_nu_peak * duty_cycle
    p_tx = p_tx_nu.to(u.W / u.Hz) * ras_bandwidth
    p_tx = p_tx_carrier
    return p_tx

def prx(freq,sat_obs_dist,sat_obs_az,sat_obs_el):
    '''
    Description: This function calculates the received power, astropy units needed for all inputs

    Parameters:
    freq: float
        frequency of the signal
    sat_obs_dist: float
        distance between the satellite and the observer 
    sat_obs_az: float
        azimuth angle of the satellite
    sat_obs_el: float
        elevation angle of the satellite
    
    Returns:
    p_rx: float
        received power (Watts)
    '''
    FSPL = cnv.free_space_lost(sat_obs_dist,freq) ### calculate free space path loss
    G_tx = sat_gain_func(  ### calculate transmitter gain
        sat_obs_az * u.deg,
        sat_obs_el * u.deg,
        )
    
    p_rx = np.sum(p_tx+G_tx+FSPL+G_rx).to_value(u.W)
    return p_rx

