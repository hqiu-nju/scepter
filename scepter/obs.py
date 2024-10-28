#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
obs.py

Call functions to simulate source and observer interactions

At the moment I will use a dummy isotropic antenna gain with cysgp4 and pycraf 

Author: Harry Qiu <hqiu678@outlook.com>

Date Created: 12-03-2024

Version: 0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import pycraf
import cysgp4
from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry
from astropy import units as u, constants as const
from .skynet import *

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
        self.p_tx = p_tx.to(cnv.dBm)

        return self.p_tx
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
        self.g_tx = G_tx
        return G_tx
    def fspl(self,sat_obs_dist,outunit=u.W):
        ### convert to power flux density values,
        '''
        Description: The corrected power of the transmitter at the observation end after free space path loss
        
        Parameters:
        sat_obs_dist: float or astropy quantity
            distance between the satellite and the observer, float will be assumed to be in m
    
        Returns:
        sat_power: float
            power of the transmitter at the observation end in dBm
        '''
        FSPL = cnv.free_space_loss(sat_obs_dist,self.freq)
        sat_power= self.p_tx+FSPL+self.g_tx ### in dBm space
        return sat_power
    def custom_gain(self,el,az,gfunc):
        '''
        Description: Retrieves gain from basic gain pattern function with angular separation to pointing considered

        Parameters:

        el: float
            elevation angle (deg) of direction in reference to transmitter pointing
        az: float
            azimuth angle (deg) of direction in reference to transmitter pointing
        gfunc: function
            gain function to be used for the satellite, it should only take the directional coordinates as input

        Returns:
        G_tx: float
            transmitter gain (dBi)
        '''
        G_tx=gfunc(el,az)
        self.g_tx = G_tx
        return G_tx





class receiver_info():
    def __init__(self,d_rx,eta_a_rx,pyobs,freq,bandwidth,tsys=20*u.k):
        '''
        Description: Information holder class to store receiver information

        Parameters:

        d_rx: astropy quantity
            diameter of the receiver telescope
        eta_a_rx: astropy quantity
            aperture efficiency of the receiver telescope
        pyobs: cysgp4 observer object
            observer object
        freq: astropy quantity
            receiver frequency band
        bandwidth: astropy quantity
            receiver bandwidth (Hz)
        tsys: astropy quantity
            system temperature of the receiver (K)
        '''
        self.d_rx = d_rx
        self.eta_a_rx = eta_a_rx
        self.location = pyobs
        self.freq = freq
        self.bandwidth = bandwidth
        self.tsys = tsys
    
    def antgain1d(self,tp_az,tp_el,sat_obs_az,sat_obs_el):
        '''
        Description: This function calculates the 1d receiver gain of an model antenna 
        using the ras_pattern function from pycraf.antenna module. 
        I changed wavelength to frequency just for the hack.

        Parameters:
        tp_az: float
            azimuth of the telescope pointing 
        tp_el: float
            elevation of the telescope pointing 
        sat_obs_az: float  
            azimuth of source
        sat_obs_el: float
            elevation of source 
    

        Returns:
        G_rx: float
            receiver gain (dBi)
        '''
        print('Obtaining satellite and telescope pointing coordinates, calculation for large arrays may take a while...')
        ang_sep = geometry.true_angular_distance(tp_az*u.deg, tp_el*u.deg, sat_obs_az*u.deg, sat_obs_el *u.deg)
        print('Done. putting angular separation into gain pattern function')
        G_rx = antenna.ras_pattern(
            ang_sep.flatten(), self.d_rx, const.c / self.freq, self.eta_a_rx
            )
        
        self.G_rx = G_rx.reshape(ang_sep.shape)
        return self.G_rx

class obs_sim():
    def __init__(self,transmitter,receiver,skygrid,mjds):
        '''
        Description: simulate observing programme

        Parameters:

        transmitter: transmitter_info object
            transmitter object
        receiver: receiver_info object
            receiver object
        tles_list: array
            numpy 1-d array of PyTle objects, converts to [np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] format, change after if needed
        skygrid: tuple
            output of the pointgen function from skynet module
        mjds: array
            2-d mjd array of epochs and observation times using skynet.plantime function
        '''

        self.transmitter = transmitter
        self.receiver = receiver
        self.ras_bandwidth = receiver.bandwidth
        self.transmitter.power_tx(self.ras_bandwidth)
        # reformat and reorganise tle array dimension?
        ## in the order of [location,antenna pointing per grid,grid cell, epochs,time,satellite]
        
        self.location = receiver.location[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
        self.mjds = mjds
        tel_az, tel_el, self.grid_info = skygrid
        ### add axis for simulation over time and iterations
        self.tel_az=tel_az[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
        self.tel_el=tel_el[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
    def load_propagation(self,nparray):

        tleprop=np.load(nparray,allow_pickle=True)
        #### obtain coordinates in observation frame and satellite
        obs_az, obs_el, obs_dist = tleprop['obs_az'], tleprop['obs_el'], tleprop['obs_dist']
        sat_frame_az, sat_frame_el= tleprop['sat_frame_az'], tleprop['sat_frame_el']
        


    def populate(self,tles_list):
        '''
        Description: This function populates the observer object with satellite information

        Used the following values from setup:
        tles_list: list
            list of tle objects (PyTle objects)
        mjds: array
            mjd time of observation

        Returns:
        sat_info: cysgp4 Satellite object  
            Satellite class that stores the satellite coordinates and information to the observer object

        '''
        self.tles_list = tles_list[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]
        observatories = self.location
        mjds = self.mjds
        tles = self.tles_list
        print(observatories.shape,tles.shape,mjds.shape)
        print('Obtaining satellite and time information, propagation for large arrays may take a while...')
        result = cysgp4.propagate_many(mjds,tles,observers=observatories,do_eci_pos=True, do_topo=True, do_obs_pos=True, do_sat_azel=True,sat_frame='zxy') 
        print('Done. Satellite coordinates obtained')
        self.sat_info = result
        # self.eci_pos = result['eci_pos']
        topo_pos = result['topo']
        sat_azel = result['sat_azel']  ### check cysgp4 for satellite frame orientation description

        # eci_pos_x, eci_pos_y, eci_pos_z = (eci_pos[..., i] for i in range(3))
        self.topo_pos_az, self.topo_pos_el, self.topo_pos_dist, _ = (topo_pos[..., i] for i in range(4))
        self.obs_az, self.obs_el, self.obs_dist = (sat_azel[..., i] for i in range(3))

    def txbeam_angsep(self,beam_el,beam_az):
        '''
        Description: Calculate the satellite pointing angle separation to the observer in the satellite reference frame

        Parameters:
        beam_el: float
            beam elevation angle in satellite reference frame zxy, where z is the motion vector
        beam_az: float  
            beam azimuth angle in satellite reference frame
        
        Returns:
        ang_sep: float
            angular separation between the satellite pointing and observer in the satellite reference frame
        '''
        result=self.sat_info
        self.angsep=sat_frame_pointing(result,beam_el,beam_az)[0]
        return self.angsep
    def pwr_on_ground(self,gainfunc,corrections,beam_el,beam_az):
        '''
        Description: Calculate the power of the transmitter on the ground

        Returns:
        pfd: float
            power flux density in dBm
        '''


    def g_rx(self,gainfunc):
        '''
        Description: Calculate the receiver gain response function

        Parameters:
        gainfunc: function
            gain function to be used for the receiver, it should take telescope pointing and source coordinates as input (l1,b1,l2,b2)

        Returns:
        g_rx: float
            receiver gain response function, 2d array
        '''
        self.g_rx=gainfunc(self.tel_az,self.tel_el,self.topo_pos_az,self.topo_pos_el)
        return self.g_rx


    
   
    
    # def calcgain1d(self):
    #     '''
    #     Description: Calculate the gain of the transmitter and receiver

    #     Returns:
    #     pfd: float
    #         power flux density in dBm
    #     '''
    #     self.transmitter.satgain1d(self.angsep)
    #     self.receiver.antgain1d(tp_el,tp_az,sat_obs_az,sat_obs_el)


def sat_frame_pointing(sat_info,beam_el,beam_az):
    '''
    Description: Calculate the satellite pointing angle separation to the observer in the satellite reference frame

    Parameters:
    sat_info: obs_sim object
        sat_info from obs_sim class populate function
    beam_el: float
        beam elevation angle in satellite reference frame zxy, where z is the motion vector
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

    #### check numpy braodcasting to fix dimensions

    ang_sep=geometry.true_angular_distance(obs_az*u.deg,obs_el*u.deg,beam_az*u.deg,beam_el*u.deg)
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

def pfd_to_Jy(P_dBm):
    '''
    Description: quick function to convert power flux density from dBm to Jansky

    Parameters:
    P_dBm: float
        power flux density in dBm, dBm/m2/Hz
    frequency_GHz: float

    Returns:
    F_Jy: float
        power flux density in Jansky (Jy), (1 Jy = 10^-26 W/m^2/Hz)
    '''

    # Convert dBm to mW
    P_mW = 10 ** (P_dBm / 10)

    # Convert mW to W
    P_W = P_mW / 1000

    
    # Define the reference flux density (1 Jy = 10^-26 W/m^2/Hz)
    S_0 = 10**-26    

    # Convert W to Jy
    F_Jy = P_W / S_0 
    
    return F_Jy