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

class transmitter_info():

    def __init__(self,p_tx_carrier,carrier_bandwidth,duty_cycle,d_tx,freq):
        '''
        Description: Information holder class to store transmitter information

        Parameters:

        p_tx_carrier: float
            transmitted power of the carrier signal
        carrier_bandwidth: float
            bandwidth of the carrier signal
        duty_cycle: float 
            duty cycle of the signal
        ras_bandwidth: float
            observation bandwidth
        d_tx: float
            diameter of the transmitter
        freq: float
    
        '''
        self.carrier = p_tx_carrier
        self.carrier_bandwidth = carrier_bandwidth
        self.duty_cycle = duty_cycle
        self.d_tx = d_tx
    


    def power_tx(self,ras_bandwidth):
        '''
        Description: This function calculates the transmitted power, astropy units needed for all inputs

        Parameters:
        ras_bandwidth: float
            observation bandwidth
        
        Returns:
        p_tx: float
            transmitted power in dBm
        '''
        # Calculate the transmitted power
        self.ras_bandwidth = ras_bandwidth
        p_tx_nu_peak = (
        self.p_tx_carrier.physical / self.carrier_bandwidth
        ).to(u.W / u.Hz)
        p_tx_nu = p_tx_nu_peak * self.duty_cycle
        p_tx = p_tx_nu.to(u.W / u.Hz) * ras_bandwidth
        self.p_tx = p_tx

        return p_tx
    def satgain1d(self,phi):
        '''
        Description: Retrieves 1-d gain from basic gain pattern function with angular separation to pointing considered

        Parameters:

        phi: float
            beam tilt angle of the satellite (this is a dummy version so just assume 
            tilt angle in reference to satellite perpendicular to earth)

        Returns:
        G_tx: float
            transmitter gain (dBi)
        '''
        
        wavelength=const.c/self.freq
        gmax=antenna.fl_G_max_from_size(self.d_tx,wavelength)  ## get G_max from diameter and frequency

        # hpbw=antenna.fl_hpbw_from_size(d_tx,wavelength)  ## get hpbw from diameter and frequency
        ### calculate angular separation of satellite to telescope pointing
        flpattern=antenna.fl_pattern(phi,diameter=self.d_tx,wavelength=wavelength,G_max=gmax)
        G_tx=flpattern
        return G_tx
    def power_arrv(self,sat_obs_dist,g_tx,outunit=u.W):
        '''
        Description: The corrected power of the transmitter at the observation end
        
        Parameters:
        sat_obs_dist: float 
            distance between the satellite and the observer
        g_tx: float
            transmitter gain in dBi
    
        Returns:
        sat_power: float
            power of the transmitter at the observation end in dBm
        '''
        FSPL = cnv.free_space_loss(sat_obs_dist,self.freq)
        sat_power= self.p_tx+FSPL+g_tx ### in dBm space
        return sat_power
    def localise(self,pyobs,pydt,tle):
        '''
        Description: This function populates the observer object with satellite information

        Parameters:
        pyobs: cysgp4 observer object
            observer object
        pydt: astropy time object
            time of observation
        tle: cysgp4 tle object
            tle object

        Returns:
        sate_info: cysgp4 Satellite object  
            Satellite class that stores the satellite coordinates and information to the observer object

        '''
        sate_info = cysgp4.PySatellite(tle,pyobs,pydt)
        return sate_info 


class receiver_info():
    def __init__(self,d_rx,eta_a_rx,lat,lon,alt,freq):
        '''
        Description: Information holder class to store receiver information

        Parameters:

        d_rx: float
            diameter of the receiver telescope
        eta_a_rx: float
            aperture efficiency of the receiver telescope
        lat: float
            latitude of the observer
        lon: float
            longitude of the observer
        alt: float
            altitude of the observer
        freq: float
            receiver frequency band
    
        '''
        self.d_rx = d_rx
        self.eta_a_rx = eta_a_rx
        self.location = cysgp4.PyObserver(lat,lon,alt)
        self.freq = freq
    
    def antgain1d(self,phi,tp_az,tp_el,sat_obs_az,sat_obs_el):
        '''
        Description: This function calculates the 1d receiver gain of an model antenna 
        using the ras_pattern function from pycraf.antenna module. 
        I changed wavelength to frequency just for the hack.

        Parameters:
        phi: float
            angular separation between the satellite and the telescope pointing
        tp_az: float
            azimuth of the telescope pointing
        tp_el: float    
            elevation of the telescope pointing
        sat_obs_az: float   
            azimuth of the satellite in the observer reference frame
        sat_obs_el: float
            elevation of the satellite in the observer reference frame
    

        Returns:
        G_rx: float
            receiver gain (dBi)
        '''
        ang_sep = geometry.true_angular_distance(tp_az, tp_el, sat_obs_az, sat_obs_el)
        G_rx = antenna.ras_pattern(
            ang_sep, self.d_rx, const.c / self.freq, self.eta_a_rx
            )
        self.G_rx = G_rx
        return G_rx

class obs_sim():
    def __init__(self,transmitter,receiver,tles_list):
        '''
        Description: simulate observing programme

        Parameters:

        transmitter: transmitter_info object
            transmitter object
        receiver: receiver_info object
            receiver object
        tles_list: list
            list of tle objects (PyTle objects)
        '''

        self.transmitter = transmitter
        self.receiver = receiver
        self.tles_list = tles_list



    def populate(self,mjds,sat_beam_tilt=[0,0]):
        '''
        Description: This function populates the observer object with satellite information

        Parameters:
        tles_list: list
            list of tle objects (PyTle objects)
        mjds: array
            mjd time of observation
        sat_beam: coordinate object
            beam tilt direction of the satellite in the satellite reference frame. no distance needed just az and el

        Returns:
        sat_info: cysgp4 Satellite object  
            Satellite class that stores the satellite coordinates and information to the observer object

        '''
        obs=self.location
        tles=self.tles_list
        sat_info=cysgp4.propagate_many(mjds,tles,observers=obs,do_eci_pos=True, do_topo=True, do_obs_pos=True, do_sat_azel=True)
        self.propagation = sat_info
        return sat_info
    def power_1d(self,tel_az,tel_el):
        '''
        Description: Calculates the received power with the receiver gain response using the dummy 1d gain models

        Parameters:
        '''




def prx_cnv(pwr,g_rx, outunit=u.W):
    '''
    description: Calculates the received power with the receiver gain response. 
    Uses the observing pointing and source coordinates to determine the gain.

    Parameters:
        pwr: float
            power of the signal
        g_rx: float
            receiver gain response function, 2d array, 1d array for ideal beam is also accepted.
        outunit: astropy unit
            unit of the output power
    
    Returns:
        p_rx: float
            received power in linear space (W)
    '''

    p_db = pwr + g_rx
    p_rx = p_db.to(outunit) ## convert to unit needed
    return p_rx

def tx_1d():
    '''
    Description: generate gain calibrated power from transmitter using 1d gain pattern in satellite reference frame,

    Input:
        sat_obs_dist: float
            distance between the satellite and the observer
        sat_obs_az: float
            azimuth of observer from satellite
        sat_obs_el: float
            elevation of observer from satellite
        sat_pointing_az:
            azimuth of the satellite pointing in satellite reference frame
        sat_pointing_el:
            elevation of the satellite pointing in satellite reference frame
        g_tx: float
            1-d transmitter gain function

    '''


# def power_rx(freq,d_rx,eta_a_rx,d_tx,p_tx,sat_obs_dist,sat_obs_az,sat_obs_el,tilt,tp_az,tp_el):
#     '''
#     Description: This function calculates the received power, astropy units needed for all inputs

#     Parameters:
#     freq: float
#         frequency of the signal
#     sat_obs_dist: float
#         distance between the satellite and the observer 
#     sat_obs_az: float
#         azimuth angle of the satellite
#     sat_obs_el: float
#         elevation angle of the satellite
    
#     Returns:
#     p_rx: float
#         received power in linear space (W)
#     '''
#     FSPL = cnv.free_space_loss(sat_obs_dist,freq) ### calculate free space path loss
#     phi = geometry.true_angular_distance(  ### calculate transmitter gain
#         sat_obs_az ,
#         sat_obs_el ,
#         tp_az ,
#         tp_el 
#         )
#     G_tx = dummy_gain_tx(tilt,d_tx,freq)
#     G_rx = gain_rx(sat_obs_az,sat_obs_el,tp_az,tp_el,d_rx,freq,eta_a_rx)
#     p_rx = (p_tx+G_tx+FSPL+G_rx).to(u.W) ### convert decibels to linear space
#     return p_rx

# def dummy_gain(phi,d_tx,freq):
#     '''
#     Description: Retrieves gain from basic gain pattern function with angular separation to pointing considered

#     Parameters:

#     phi: float
#         beam tilt angle of the satellite (this is a dummy version so just assume 
#         tilt angle in reference to satellite perpendicular to earth)
#     d_tx: float
#         diameter of the transmitter
#     freq: float
#         frequency of the signal

#     Returns:
#     G_tx: float
#         transmitter gain (dBi)
#     '''
    
#     wavelength=const.c/freq
#     gmax=antenna.fl_G_max_from_size(d_tx,wavelength)  ## get G_max from diameter and frequency

#     # hpbw=antenna.fl_hpbw_from_size(d_tx,wavelength)  ## get hpbw from diameter and frequency
#     ### calculate angular separation of satellite to telescope pointing
#     flpattern=antenna.fl_pattern(phi,diameter=d_tx,wavelength=wavelength,G_max=gmax)
#     G_tx=flpattern
#     return G_tx