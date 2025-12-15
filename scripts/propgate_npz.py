import numpy as np
import matplotlib.pyplot as plt
import pycraf
import cysgp4
import astropy.coordinates as coord
from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry, satellite
from astropy import units as u, constants as const
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.coordinates import AltAz
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import threading
import time as Time
import argparse
from scepter import skynet, obs, tlefinder


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process TLEs for observations.")
parser.add_argument("filename", type=str, help="Path to the .npz data file.")
parser.add_argument("tle_dir", type=str, help="Directory containing TLE files.")
args = parser.parse_args()

# Define observer parameters
longitude1 = coord.Angle("11d55m11s").deg
latitude1 = coord.Angle("+57d23m37s").deg
elevation1 = 13.2 * u.m
ott1 = cysgp4.PyObserver(longitude1, latitude1, elevation1.to(u.km).value)

longitude2 = coord.Angle("11d55m08s").deg
latitude2 = coord.Angle("+57d23m35s").deg
elevation2 = 13.2 * u.m
ott2 = cysgp4.PyObserver(longitude2, latitude2, elevation2.to(u.km).value)

# Define timer function
def update_timer(start_time, stop_event):
    while not stop_event.is_set():
        elapsed_time = Time.time() - start_time
        print(f"\rElapsed time: {elapsed_time:.2f} seconds", end="")
        Time.sleep(1)
    elapsed_time = Time.time() - start_time
    print(f"\rElapsed time: {elapsed_time:.2f} seconds")

# Automatically generate tlabel using the filename
tlabel = args.filename.split("/")[-1].replace("dataset_", "").replace(".npz", "")

# Load data from the specified file
print(f"Loading data from {args.filename}...")
data_file = np.load(args.filename, allow_pickle=True)
mjd = data_file['mjd']

# Initialize the TLE finder
finder = tlefinder.TLEfinder([ott1], tdir=args.tle_dir)
bdate, btle, btlename = finder.mjd_locator(mjd)

print(f"start propagation with file {btlename[100]}, date {bdate[100]}, mjd {mjd[100]}")
print(f"time dimension is {mjd.shape}, tles dimension is {btle.shape}")

# Start the timer
start_time = Time.time()
stop_event = threading.Event()
timer_thread = threading.Thread(target=update_timer, args=(start_time, stop_event))
print('running propagation models with TLEs for each MJD timestamp, starting timer\n')
timer_thread.start()

# Propagate satellite positions
satinfo = tlefinder.propagate_satellites_from_SKAO_database(finder.obs, mjd, btle)

# Stop the timer
stop_event.set()
timer_thread.join()
print("\nPropagation complete.")

# Obtain coordinates in observation frame and satellite frame
topo_frame_az, topo_frame_el, topo_frame_dist = tlefinder.parse_sgp4info(satinfo, frame='topo')
sat_frame_az, sat_frame_el, sat_frame_dist = tlefinder.parse_sgp4info(satinfo, frame='sat_azel')

# Save the results as an .npz file
output_filename = f'./dataproducts/{tlabel}_sx_positions.npz'
np.savez(output_filename,
         topo_frame_az=topo_frame_az, topo_frame_el=topo_frame_el,
         topo_frame_dist=topo_frame_dist,
         sat_frame_az=sat_frame_az, sat_frame_el=sat_frame_el,
         sat_frame_dist=sat_frame_dist, mjd=mjd)

print(f"Results saved to {output_filename}")