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
from astropy.coordinates import EarthLocation,SkyCoord
from astropy.time import Time
from astropy.coordinates import AltAz, ICRS



def sat_frame_pointing(satf_az,satf_el,beam_el,beam_az):
    '''
    Description: Calculate the satellite pointing angle separation to the observer in the satellite reference frame

    Parameters:
    satf_az: float
        azimuth of the satellite in the satellite reference frame
    satf_el: float
        elevation of the satellite in the satellite reference frame
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
    # tleprop = sat_info
    # #### obtain coordinates in observation frame and satellite frame
    # # topo_pos_az, topo_pos_el= tleprop['obs_az'], tleprop['obs_el']
    # satf_az, satf_el, satf_dist = tleprop['sat_frame_az'], tleprop['sat_frame_el'], tleprop['sat_frame_dist']


    #### check numpy braodcasting to fix dimensions

    ang_sep=geometry.true_angular_distance(satf_az*u.deg,satf_el*u.deg,beam_az*u.deg,beam_el*u.deg)
    delta_az=satf_az-beam_az
    delta_el=satf_el-beam_el
    return ang_sep,delta_az,delta_el



def baseline_bearing(ref,ant):
    """
    calculate the bearing of antenna 2 with respect to antenna 1
    Args:
        ref (object): reference antenna/location PyObserver object
        ant (object): antenna for baseline PyObserver object
    Returns:
        bearing: vector from antenna baseline in cartesian coordinates
        d: baseline vector modulus or baseline length in meters
    """
    ant1 = ref
    ant2 = ant
    x1,y1,z1 = pycraf.geospatial.wgs84_to_itrf2008(ant1.loc.lon*u.deg, ant1.loc.lat*u.deg, ant1.loc.alt*u.m)
    x2,y2,z2 = pycraf.geospatial.wgs84_to_itrf2008(ant2.loc.lon*u.deg, ant2.loc.lat*u.deg, ant2.loc.alt*u.m)



    a1=np.array([x1.value,y1.value,z1.value])
    a2=np.array([x2.value,y2.value,z2.value])
    # print(a1,a2)
    bearing = a2-a1
    d = np.linalg.norm(bearing) # Calculate the distance between the antennas

    
    return bearing, d # Return the distance in meters

def baseline_pairs(antennas):
    """
    Calculate the baseline pairs for a given set of antennas and returns a mask for the baseline distances
    Args:
        antennas (list): list of antenna objects
    Returns:
        baselines (list): list of baseline pairs
    """

    baselines=[] ### true baseline distance
    bearings=[]
    rabs=[] ### east-west component of the baseline distance
    for i in range(len(antennas)):
        # print(f"Pair: {i} and {j}")
        ant = antennas[i]
        ref = antennas[0]
        # Calculate the baseline distance
        bearing,d = baseline_bearing(ref, ant)
        baselines.append(d)
        bearings.append(bearing)

    return np.array(bearings),np.array(baselines)

def baseline_vector(d,az,el,lat):
    """
    calculate the effective baseline distance with a confirmed pointing angle of the reference antenna/position
    Args:
        d (float): distance to the antenna in meters
        az (float): azimuth angle in radians
        el (float): elevation angle in radians
        lat (float): latitude of the antenna in radians
    Returns:
        vector: array of the effective baseline vector in cartesian coordinates x,y,z (meters)

    """
    
    
    return d*np.array([np.cos(lat)*np.sin(el)-np.sin(lat)*np.cos(el)*np.cos(az),
    np.cos(el)*np.sin(az),
    np.sin(lat)*np.sin(el)+np.cos(lat)*np.cos(el)*np.cos(az)]) # x,y,z coordinates in meters
    

def mod_tau(az,el,lat,D):
    """
    Calculate the delay difference from source pointing in seconds for a given angle for large arrays
    Args:
        baseline (quantity): Baseline length in meters etc.
    Returns:
        tau (quantity): delay in seconds
    """
    c = 3e8 *u.m/u.s  # speed of light in m/s
    baseline = D.to(u.m) # Convert baseline to meters
    new_bearing = baseline_vector(baseline,az,el,lat)
    # print(new_bearing.shape)
    D_eff = np.linalg.norm(new_bearing,axis=0)
    # print(D_eff.shape)    
    return D_eff/c




def baseline_nearfield_delay(l1,l2,tau):
    """
    Calculate the delay difference from source pointing in seconds for a given angle
    Args:
        l1 (quantity): distance to the antenna 1 in distance units
        l2 (quantity): distance to the antenna 2 in distance units
        tau (quantity): baseline delay between two antennas in time units
    Returns:
        delay (quantity): delay in seconds
    """
    c = 3e8 *u.m/u.s  # speed of light in m/s
    l1 = l1.to(u.m) # Convert distance to meters
    l2 = l2.to(u.m) # Convert distance to meters
    
    return  (l1-l2)/c-tau

def fringe_attenuation(theta, baseline, bandwidth):
    """
    Calculate the fringe attenuation for a given angle, baseline, frequency, and bandwidth.
    Args:
        theta (quantity): off phase center Angle in radians/degrees etc.
        baseline (quantity): Baseline east-west component in meters etc.
        bandwidth (quantity): Bandwidth in Hz etc.
    """
    c = 3e8  # speed of light in m/s
    theta = theta.to(u.rad).value  # Convert angle to radians
    baseline = baseline.to(u.m).value  # Convert baseline to meters
    bandwidth = bandwidth.to(u.Hz).value  # Convert bandwidth to Hz
    return np.sinc(np.sin(theta)*baseline*bandwidth/c)

def fringe_response(delay,frequency):
    """
    Calculate the fringe response for a given delay and frequency
    based on two element equation integration, assuming equal gain

    Args:
        delay (quantity): delay in seconds
        frequency (quantity): frequency in Hz
    Returns:
        response (quantity): fringe response
    """
    delay = delay.to(u.s).value  # Convert delay to seconds
    frequency = frequency.to(u.Hz).value  # Convert frequency to Hz

    return np.cos(2*np.pi*frequency*delay)


def bw_fringe(delays,bwchan,fch1,chan_bin=100):
    """
    Calculate the fringe response for a given delay and frequency
    based on two element equation integration, assuming equal gain.
    the function takes into the frequency settings and does a integration with 0.1 kHz resolution
    over the channel bandwidth and number of channels
    delay array must be 1d array, flatten your input and reshape the output.

    Args:
        delays (array): delay array in seconds, 
        bwchan (quantity): channel bandwidth 
        fch1 (quantity): channel centre frequency
        chan_bin (int): number of channels in the band
    Returns:
        response (float): fringe response
    """
    fch1 = fch1.to(u.kHz).value  # Convert frequency to kHz
    bwchan = bwchan.to(u.kHz).value  # Convert bandwidth to kHz
    # chan_bin=np.int32(bwchan/0.1) ### the bin number
    freq_array= np.linspace(fch1-bwchan*0.5,fch1+bwchan*0.5,chan_bin) *u.kHz # 0.1 kHz resolution
    delays= delays[:,np.newaxis] # add axis to delays
    freq_array = freq_array[np.newaxis,:]
    fringes=fringe_response(delays,freq_array)
    return np.mean(fringes,axis=1)


def prx_cnv(pwr,g_rx, outunit=u.W):
    '''
    description: Calculates the received power with the receiver gain response. 
    Uses the observing pointing and source coordinates to determine the gain.

    Parameters:
        pwr: float
            power of the signal (Units in dBm)
        g_rx: float
            receiver gain response function, 2d array, 1d array for ideal beam is also accepted (Units in dBm)
        outunit: astropy unit
            unit of the output power (Default in W)
    
    Returns:
        p_rx: float
            received power in linear space (Units in W)
    '''

    p_db = pwr + g_rx
    p_rx = p_db.to(outunit) ## convert to unit needed
    return p_rx

def pfd_to_Jy(pfd):
    '''
    Description: quick function to convert power flux density from dBm to Jansky

    Parameters:
    pfd: float
        power flux density in dB W, dB W/m2/Hz
    frequency_GHz: float

    Returns:
    F_Jy: float
        power flux density in Jansky (Jy), (1 Jy = 10^-26 W/m^2/Hz)
    '''

    # Convert W/m^2/Hz to W/m^2
    P_W = 10 ** (pfd / 10)


    
    # Define the reference flux density (1 Jy = 10^-26 W/m^2/Hz)
    S_0 = 10**-26    

    # Convert W to Jy
    F_Jy = P_W / S_0 
    
    return F_Jy


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
    def __init__(self,skygrid,mjds):
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

        # self.transmitter = transmitter
        # self.receiver = receiver
        # self.ras_bandwidth = receiver.bandwidth
        # self.transmitter.power_tx(self.ras_bandwidth)
        # reformat and reorganise tle array dimension?
        ## in the order of [location,grid cell,antenna pointing per grid, epochs,time,satellite]
        
        self.location = receiver.location[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
        self.mjds = mjds
        tel_az, tel_el, self.grid_info = skygrid
        ### add axis for simulation over time and iterations
        self.grid_az=tel_az[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
        self.grid_el=tel_el[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
    def sky_track(self,ra,dec,frame='icrs'):

        '''
        Description: This function calculates the azimuth and elevation of a celestial ra dec source from the reference antenna location

        Parameters:
        ra: float/array
            right ascension of the source (degrees)
        dec: float/array
            declination of the source (degrees)
        frame: str/astropy object
            frame of the source coordinates, default is ICRS
            Note: can also input astropy alt az object for az el tracking
        
        Returns:
        tel1_pnt: astropy object
            azimuth and elevation of the source in the telescope reference frame
        '''
        # self.pnt_ra = ra
        # self.pnt_dec = dec
        ant1 = self.location.flatten()[0]
        time_1d = Time(self.mjds.flatten(), format='mjd', scale='utc')
        loc1 = EarthLocation(lat=ant1.loc.lat, lon=ant1.loc.lon, height=ant1.loc.alt)
        altaz = AltAz( obstime=time_1d, location=loc1)
        skycoord_track=SkyCoord(ra,dec, unit=u.deg,frame=frame)
        self.pnt=skycoord_track
        self.altaz_frame=altaz
        tel1_pnt=skycoord_track.transform_to(altaz)
        self.pnt_az, self.pnt_el = tel1_pnt.az[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis], tel1_pnt.alt[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        return tel1_pnt
    def load_propagation(self,nparray):

        tleprop=np.load(nparray,allow_pickle=True)
        self.sat_info = tleprop
        #### obtain coordinates in observation frame and satellite frame
        topo_pos_az, topo_pos_el, topo_pos_dist = tleprop['obs_az'], tleprop['obs_el'], tleprop['obs_dist']
        satf_az, satf_el, satf_dist = tleprop['sat_frame_az'], tleprop['sat_frame_el'] , tleprop['sat_frame_dist']

        self.topo_pos_az = topo_pos_az
        self.topo_pos_el = topo_pos_el
        self.topo_pos_dist = topo_pos_dist
        self.satf_az = satf_az
        self.satf_el = satf_el
        self.satf_dist = satf_dist

    def reduce_sats(self,el_limit=0):
        '''
        Description: This function reduces the number of satellites in the simulation by applying a limit to the elevation angle
        Parameters:
        el_limit: float
            elevation angle limit (degrees)
        '''

        min_el=np.min(self.topo_pos_el,axis=(0,1,2,3,4))
        mask = min_el>el_limit
        # self.sat_info["obs_az"] = self.sat_info["obs_az"][:,:,:,:,:,mask]
        # self.sat_info["obs_el"] = self.sat_info["obs_el"][:,:,:,:,:,mask]
        # self.sat_info["obs_dist"] = self.sat_info["obs_dist"][:,:,:,:,:,mask]
        # self.sat_info["sat_frame_az"] = self.sat_info["sat_frame_az"][:,:,:,:,:,mask]
        # self.sat_info["sat_frame_el"] = self.sat_info["sat_frame_el"][:,:,:,:,:,mask]
        # self.sat_info["sat_frame_dist"] = self.sat_info["sat_frame_dist"][:,:,:,:,:,mask]

        self.topo_pos_az = self.topo_pos_az[:,:,:,:,:,mask]
        self.topo_pos_el = self.topo_pos_el[:,:,:,:,:,mask]
        self.topo_pos_dist = self.topo_pos_dist[:,:,:,:,:,mask]
        self.satf_az = self.satf_az[:,:,:,:,:,mask]
        self.satf_el = self.satf_el[:,:,:,:,:,mask]
        self.satf_dist = self.satf_dist[:,:,:,:,:,mask]
        self.elevation_mask = mask
        
    def populate(self,tles_list,save=True, savename="satellite_info.npz"):
        '''
        Description: This function populates the observer object with satellite information

        Used the following values from setup:
        tles_list: list
            list of tle objects (PyTle objects)
        mjds: array
            mjd time of observation
        save: bool
            save the satellite information to a file to avoid redoing the calculation
        savename: str
            name of the file to save the satellite information
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
        
        # self.eci_pos = result['eci_pos']
        topo_pos = result['topo']
        sat_azel = result['sat_azel']  ### check cysgp4 for satellite frame orientation description

        # eci_pos_x, eci_pos_y, eci_pos_z = (eci_pos[..., i] for i in range(3))
        self.topo_pos_az, self.topo_pos_el, self.topo_pos_dist, _ = (topo_pos[..., i] for i in range(4))
        
        ### this means azimuth and elevation of the observer, I think the naming is a bit confusing
        self.satf_az, self.satf_el, self.satf_dist = (sat_azel[..., i] for i in range(3))  
        if save == True:
            np.savez(savename,obs_az=self.topo_pos_az,obs_el=self.topo_pos_el,obs_dist=self.topo_pos_dist,sat_frame_az=self.satf_az,sat_frame_el=self.satf_el,sat_frame_dist=self.satf_dist)
        tleprop=np.load(savename,allow_pickle=True)
        self.sat_info = tleprop

    def txbeam_angsep(self,beam_el,beam_az,apply_mask=True):
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
        self.angsep=sat_frame_pointing(self.satf_az,self.satf_el,beam_el,beam_az)
        return self.angsep

    def create_baselines(self):
        '''
        Description: Create the baseline pairs array for fringe simulation
        '''
        from itertools import combinations
        antennas = self.receiver.location
        self.baselines = combinations(range(len(antennas)), 2)
        self.bearings, self.bearing_D = baseline_pairs(antennas)
        # self.bearings = self.bearings.reshape(self.location.shape)
        self.bearing_D = self.bearing_D.reshape(self.location.shape)
        # self.delays = mod_tau(self.baselines*u.m)

    def baselines_nearfield_delays(self):
        '''
        Description: Calculate the near field delay for the baselines using the satellite positions

        returns:
        baseline_delays: float
            delay difference for each baseline at each instance of pointing, returns in whole simulation array format 
        '''


        lat = self.location.flatten()[0].loc.lat
        l1 = self.topo_pos_dist[0][np.newaxis,:]*u.km
        tau1=mod_tau(self.pnt_az,self.pnt_el,lat,self.bearing_D*u.m)
        self.pnt_tau = tau1
        self.baseline_delays = baseline_nearfield_delay(l1,self.topo_pos_dist*u.km,tau=tau1)

        return self.baseline_delays
    
    def 

