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

        p_tx_carrier: astropy quantity
            transmitted power of the carrier signal
        carrier_bandwidth: astropy quantity
            bandwidth of the carrier signal
        duty_cycle: astropy quantity 
            duty cycle of the signal
        d_tx: astropy quantity
            diameter of the transmitter
        freq: astropy quantity
            frequency of the signal (centre of band)
        '''
        self.carrier = p_tx_carrier
        self.carrier_bandwidth = carrier_bandwidth
        self.duty_cycle = duty_cycle
        self.d_tx = d_tx
        self.freq = freq
    


    def power_tx(self,ras_bandwidth):
        '''
        Description: This function calculates the transmitted power, astropy units needed for all inputs

        Parameters:
        ras_bandwidth: astropy quantity
            observation bandwidth
        
        Returns:
        p_tx: astropy quantity
            transmitted power in dBm
        '''
        # Calculate the transmitted power
        self.ras_bandwidth = ras_bandwidth
        p_tx_nu_peak = (
        self.carrier.physical / self.carrier_bandwidth
        ).to(u.W / u.Hz)
        p_tx_nu = p_tx_nu_peak * self.duty_cycle
        p_tx = p_tx_nu.to(u.W / u.Hz) * ras_bandwidth
        self.p_tx = p_tx

        return p_tx
    def satgain1d(self,phi):
        '''
        Description: Retrieves 1-d gain from basic gain pattern function with angular separation to pointing considered

        Parameters:

        phi: astropy quantity
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



class receiver_info():
    def __init__(self,d_rx,eta_a_rx,pyobs,freq,bandwidth):
        '''
        Description: Information holder class to store receiver information

        Parameters:

        d_rx: float
            diameter of the receiver telescope
        eta_a_rx: float
            aperture efficiency of the receiver telescope
        pyobs: cysgp4 observer object
            observer object
        freq: float
            receiver frequency band
        bandwidth: float
            receiver bandwidth
    
        '''
        self.d_rx = d_rx
        self.eta_a_rx = eta_a_rx
        self.location = pyobs
        self.freq = freq
        self.bandwidth = bandwidth
    
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
        self.ras_bandwidth = receiver.bandwidth
        self.transmitter.power_tx(self.ras_bandwidth)
        self.tles_list = tles_list
        self.location = receiver.location

    def populate(self,mjds):
        '''
        Description: This function populates the observer object with satellite information

        Parameters:
        tles_list: list
            list of tle objects (PyTle objects)
        mjds: array
            mjd time of observation

        Returns:
        sat_info: cysgp4 Satellite object  
            Satellite class that stores the satellite coordinates and information to the observer object

        '''
        obs=self.location
        tles=self.tles_list
        sat_info=cysgp4.propagate_many(mjds,tles,observers=obs,do_eci_pos=True, do_topo=True, do_obs_pos=True, do_sat_azel=True)
        self.propagation = sat_info
        return sat_info




def sat_frame_pointing(sat_info,beam_el,beam_az):
    '''
    Description: Calculate the satellite pointing angle separation to the observer in the satellite reference frame

    Parameters:
    sat_info: obs_sim object
        sat_info from obs_sim class populate function
    beam_el: float
        beam elevation angle in satellite reference frame
    beam_az: float  
        beam azimuth angle in satellite reference frame
    
    Returns:
    ang_sep: float
        angular separation between the satellite pointing and observer in the satellite reference frame
    delta_az: float
        azimuth difference between the satellite pointing and observer in the satellite reference frame
    delta_el: float
        elevation difference between the satellite pointing and observer in the satellite reference frame
    obs_dist: float
        distance between the observer and the satellite
    '''
    result=sat_info
    # eci_pos = result['eci_pos']
    # topo_pos = result['topo']
    sat_azel = result['sat_azel']  ### check cysgp4 for satellite frame orientation description

    # eci_pos_x, eci_pos_y, eci_pos_z = (eci_pos[..., i] for i in range(3))
    # topo_pos_az, topo_pos_el, topo_pos_dist, _ = (topo_pos[..., i] for i in range(4))
    obs_az, obs_el, obs_dist = (sat_azel[..., i] for i in range(3))
    ang_sep=geometry.true_angular_distance(obs_az,obs_el,beam_az,beam_el)
    delta_az=obs_az-beam_az
    delta_el=obs_el-beam_el
    return ang_sep,delta_az,delta_el,obs_dist




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