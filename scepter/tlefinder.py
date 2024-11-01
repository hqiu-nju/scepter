import glob 
from datetime import datetime
import cysgp4
from astropy.time import Time
import numpy as np

class TLEfinder():
    def __init__(self, obs, tdir = './'):
        '''
        Class to find and load TLEs from the TLE archive for specific observatory.

        Parameters
        ----------
        obs : list of cysgp4.PyObserver object
            Observer object from cysgp4 of the observatory
        dir : String (default = './')
            Directory where the TLEs are stored, TLEs need to be in datetime format of %Y%m%d_%H%M%S



        '''
        self.obs = np.array(obs)
        self.tdir = tdir
        tle_files = sorted(glob.glob(f'{tdir}/*.npz'))
        self.tle_files = tle_files
        self.filedates = collect_datetimes_from_TLEs(tle_files)
        self.pytles_by_date=np.array([readtlenpz(i) for i in tle_files])

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
        bestdate_idx = []
        best_tle = []
        best_tle_file = []


        for i in mjd:
            bestdate = np.argmin(np.abs(self.filedates - i))
            bestdate_idx.append(bestdate)
            best_tle.append(self.pytles_by_date[bestdate])
            best_tle_file.append(self.tle_files[bestdate])
        return np.array(bestdate_idx),np.array(best_tle),np.array(best_tle_file)

    def run_propagator(mjds,best_tle,geteci=False,getsat=True):
        '''
        Run the propagator for the best dates

        Parameters
        ----------
        mjds : numpy array
            array of mjd date
        best_tle : numpy array
            Array of the TLEs, should be 2d array to match bestdates [n_dates,n_tles]
        '''
        self.satinfo = propagate_satellites_from_SKAO_database(self.obs,mjds,best_tle,geteci=geteci,getsat=getsat)



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

def readtlenpz(path):
    '''
    Function to read tle file from directory and return the TLEs in pytle
    '''
    tles=np.load(path,allow_pickle=True)
    response=tles['arr_0']
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

