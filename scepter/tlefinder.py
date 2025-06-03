#!/usr/bin/env python3

"""
tlefinder.py

A smart utility to find TLEs from the SKA Observatory TLE archive and propagate them to the observer location.
For tle npz format, please contact the SKAO spectrum management team.

Author: Harry Qiu <hqiu678@outlook.com>

Date: 12-04-2025
"""

import glob 
from datetime import datetime
import cysgp4
from astropy.time import Time
import numpy as np
import pandas as pd
class TLEfinder():
    def __init__(self, obs, tdir = './'):
        '''
        Class to find and load TLEs from the TLE archive for specific observatory.

        Parameters
        ----------
        obs : list of cysgp4.PyObserver object
            Observer object from cysgp4 of the observatory
        dir : String (default = './')
            Directory where the TLEs are stored, TLEs need to be in datetime format of %Y%m%d_%H%M%S in a npz file



        '''
        self.obs = np.array(obs)
        self.tdir = tdir
        tle_files = np.array(sorted(glob.glob(f'{tdir}/*.npz')))
        self.tle_files = tle_files
        self.filedates = collect_datetimes_from_TLEs(tle_files)
        self.pytles_by_date=[readtlenpz(i) for i in tle_files]

    def mjd_locator(self,mjd):
        '''
        Use mjds to find TLEs in archive and return the satellite positions

        Parameters
        ----------
        mjd : numpy array
            Array of the observation mjds

        Returns
        -------
        bestdate : numpy array
            Array of the best date for the TLEs
        best_tle : numpy array
            array containing tles of constellation
        best_tle_file : numpy array
            array containing tle file names
        '''
        useddates = []


        for i in mjd:
            bestdate = np.argmin(np.abs(self.filedates - i))
            useddates.append(bestdate)


        useddate_idx = np.unique(useddates)
        used_tles = [self.pytles_by_date[i] for i in useddate_idx]
        used_dates = [self.filedates[i] for i in useddate_idx]
        used_tle_files = [self.tle_files[i] for i in useddate_idx]
        check_len = np.array([len(i) for i in used_tles])
        cnts = np.unique(check_len)
        used_tles_array=[]
        setids=[np.where(check_len==i)[0] for i in cnts]
        real_sat_n = np.min(cnts)
        print('Satellite group in TLEs may not be consistent across time in the archive, checking and removing the changed TLEs')
        print(f"Number of TLEs being checked: {len(used_tles)}, with {len(setids)} different lengths")
        ### Find the maximum length of the TLEs
        print(f"Estimate Number of TLEs consistent in the TLE list: {real_sat_n}")


        ### get the satellites that appear in all tles
        real_int_des=np.array([bytes.decode(sattle.int_designator) for sattle in used_tles[0] ])
        for line in used_tles[1:]:
            int_des_array=np.array([bytes.decode(sattle.int_designator) for sattle in line ])
            real_int_des = np.intersect1d(real_int_des,int_des_array)
        ## sanity check that numbers match
        print(f'Intersection method found the true number of satellites appearing in all intervals to be {len(real_int_des)}')
        
        for line in used_tles: 
            int_des_array=[bytes.decode(sattle.int_designator) for sattle in line ]
            intersection = np.intersect1d(int_des_array,real_int_des)
            # print(len(line),len(int_des_array))
            # Find the positions of the extra elements
            remask = np.in1d(int_des_array, intersection)
            # print(sum(remask))
            ### remove the extra elements from the line in best_tle
            used_tles_array.append(line[remask])
        used_tles_array=np.array(used_tles_array)
        bestdate_idx = []
        best_tle = []
        best_tle_file = []
        for i in mjd:
            bestdate = np.argmin(np.abs(used_dates - i))
            bestdate_idx.append(bestdate)
            best_tle.append(used_tles_array[bestdate])
            best_tle_file.append(used_tle_files[bestdate])
        # Find the length of the best tles and check if they are consistent

        return np.array(bestdate_idx),np.array(best_tle),np.array(best_tle_file)

    def run_propagator(self,geteci=False,getsat=True):
        '''
        Run the propagator for the best dates

        Parameters
        ----------
        mjds : numpy array
            array of mjd date
        best_tle : numpy array
            Array of the TLEs, should be 2d array to match bestdates [n_dates,n_tles]
        '''
        self.satinfo = propagate_satellites_from_SKAO_database(self.obs,self.mjds,self.best_tle,geteci=geteci,getsat=getsat)

def id_locator(used_tles,int_designator):
    
    real_int_des=np.array([bytes.decode(sattle.int_designator).split(" ")[0] for sattle in used_tles ])
    mask = (real_int_des == int_designator)
    return np.where(mask)[0],used_tles[mask]

def propagate_satellites_from_SKAO_database(observatory,obs_mjds,pytle,geteci=False,getsat=True):
    '''
    Function to propagate the satellites from the SKAO TLE database
    
    Parameters
    ----------
    observatory : numpy array object
        Observer object from cysgp4 of the observatory
    obs_mjds : numpy array
        Array of the observation mjds
    pytle : numpy array of pytle objects
        Array of the TLEs
    geteci : Boolean (default = False)
        option to get satellite position in ECI frame
    getsat : Boolean (default = True)
        option to get the satellite frame position of observer

    Returns
    -------
    satinfo : numpy array
        Array of the satellite information
    satname : numpy array 
        Array of the satellite names
    '''
    obs_mjds = np.array(obs_mjds)
    observatory =  observatory[:,np.newaxis,np.newaxis]
    obs_mjds = obs_mjds[np.newaxis,:,np.newaxis]
    pytle = pytle[np.newaxis,:,:]
    satinfo=cysgp4.propagate_many(obs_mjds,pytle,observers=observatory,
            do_eci_pos=geteci, do_topo=True, do_obs_pos=False, do_sat_azel=getsat,sat_frame='zxy') 

    return satinfo

def parse_sgp4info(satinfo, frame='sat_azel'):
    '''
    Function to parse the output of cysgp4.propagate_many
    
    Parameters
    ----------
    satinfo : dictionary
        output of cysgp4.propagate_many
    frame : string (default = 'sat_azel')
        frame to parse the output, options are sat_azel, eci_pos, topo
    '''
    obsazel = satinfo[frame]  ### check cysgp4 for satellite frame orientation description
    obs_az, obs_el, obs_dist = (obsazel[..., i] for i in range(3))

    return obs_az, obs_el, obs_dist

def readtlenpz(path,key='arr_0'):
    '''
    Function to read tle file from directory and return the TLEs in pytle
    '''
    tles=np.load(path,allow_pickle=True)
    response=tles[key]
    tle_string=response.item().text
    # print(tle_string)
    tles=np.array(cysgp4.tles_from_text(tle_string))
    ### satellites are in the form of [telescope, n_satellites,mjd]?
    return tles


def collect_datetimes_from_TLEs(tle_files):
    '''
    Function to collect the datetimes of the supplementary TLEs stored in the SKAO TLE folder

    Parameters
    ----------
    tle_files : list
        List of the TLE files in the folder

    Returns
    -------
    datetimes : numpy array
        Array of the datetimes of the TLEs in mjd format
    '''
    
    datetimes = []
    for f in tle_files:
        DT = str.split(str.split(f,'/')[-1],'.')[0]
        dt_date=datetime.strptime(DT,"%Y%m%d_%H%M%S")
        datetimes.append(dt_date)

    return Time(datetimes).mjd

