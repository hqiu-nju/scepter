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

def dummy_gain_tx(phi,d_tx,freq):
    '''
    Description: Retrieves gain from basic gain pattern function with angular separation to pointing considered

    Parameters:

    phi: float
        beam tilt angle of the satellite (this is a dummy version so just assume 
        tilt angle in reference to satellite perpendicular to earth)
    d_tx: float
        diameter of the transmitter
    freq: float
        frequency of the signal

    Returns:
    G_tx: float
        transmitter gain (dBi)
    '''
    
    wavelength=const.c/freq
    gmax=antenna.fl_G_max_from_size(d_tx,wavelength)  ## get G_max from diameter and frequency

    # hpbw=antenna.fl_hpbw_from_size(d_tx,wavelength)  ## get hpbw from diameter and frequency
    ### calculate angular separation of satellite to telescope pointing
    flpattern=antenna.fl_pattern(phi,diameter=d_tx,wavelength=wavelength,G_max=gmax)
    G_tx=flpattern
    return G_tx


def gain_rx(sat_obs_az,sat_obs_el,tp_az,tp_el,d_rx,freq,eta_a_rx):
    '''
    Description: This function calculates the receiver gain of an model antenna 
    using the ras_pattern function from pycraf.antenna module. 
    I changed wavelength to frequency just for the hack.

    Parameters:
    sat_obs_az: float
        azimuth angle of the satellite
    sat_obs_el: float
        elevation angle of the satellite
    tp_az: float
        azimuth angle of the telescope pointing
    tp_el: float
        elevation angle of the telescope pointing
    d_rx: float
        diameter of the receiver telescope
    freq: float
        frequency of the signal
    eta_a_rx: float
        aperture efficiency of the receiver telescope

    Returns:
    G_rx: float
        receiver gain (dBi)
    '''
    ang_sep = geometry.true_angular_distance(tp_az, tp_el, sat_obs_az, sat_obs_el)
    G_rx = antenna.ras_pattern(
        ang_sep, d_rx, const.c / freq, eta_a_rx
        )
    return G_rx


def power_tx(p_tx_carrier,carrier_bandwidth,duty_cycle,ras_bandwidth):
    '''
    Description: This function calculates the transmitted power, astropy units needed for all inputs

    Parameters:
    p_tx_carrier: float
        transmitted power of the carrier signal
    carrier_bandwidth: float
        bandwidth of the carrier signal
    duty_cycle: float
        duty cycle of the signal
    ras_bandwidth: float
        observation bandwidth
    
    Returns:
    p_tx: float
        transmitted power in dBm
    '''
    # Calculate the transmitted power
    p_tx_nu_peak = (
    p_tx_carrier.physical / carrier_bandwidth
    ).to(u.W / u.Hz)
    p_tx_nu = p_tx_nu_peak * duty_cycle
    p_tx = p_tx_nu.to(u.W / u.Hz) * ras_bandwidth
    p_tx = p_tx_carrier
    return p_tx

def power_rx(freq,d_rx,eta_a_rx,d_tx,p_tx,sat_obs_dist,sat_obs_az,sat_obs_el,tp_az,tp_el):
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
        received power in linear space (W)
    '''
    FSPL = cnv.free_space_loss(sat_obs_dist,freq) ### calculate free space path loss
    phi = geometry.true_angular_distance(  ### calculate transmitter gain
        sat_obs_az ,
        sat_obs_el ,
        tp_az ,
        tp_el 
        )
    G_tx = dummy_gain_tx(phi,d_tx,freq)
    G_rx = gain_rx(sat_obs_az,sat_obs_el,tp_az,tp_el,d_rx,freq,eta_a_rx)
    p_rx = (p_tx+G_tx+FSPL+G_rx).to(u.W) ### convert decibels to linear space
    return p_rx

