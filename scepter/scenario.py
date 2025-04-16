"""
scenario.py

This module is providing supplemental function to enhance the simulation.

Author: boris.sorokin <mralin@protonmail.com>
Date: 16-04-2025
"""
import os
import h5py
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from pycraf import conversions as cnv
import signal
from contextlib import contextmanager
from datetime import datetime
import threading
import queue
import sys

# Global variable for the current thread count (used by simulation)
current_thread_count = 8

def _timestamp():
    """Return the current time as a formatted timestamp string."""
    return datetime.now().strftime("[%H:%M:%S]")

def set_cysgp4_threads(n, logger=None):
    """
    Update the number of threads used by cysgp4 propagation.

    The value is clamped to [1, 32]. If the cysgp4 module is available, it calls
    cysgp4.set_num_threads(n).

    Parameters
    ----------
    n : int
        The desired thread count.
    logger : callable, optional
        A function that accepts a string (the log message). If provided, this
        function is used to output timestamped messages. If None, fallback to print.
    """
    global current_thread_count
    n = max(1, min(n, 32))
    try:
        import cysgp4
        cysgp4.set_num_threads(n)
    except ImportError:
        msg = f"Warning: cysgp4 module not available."
        if logger:
            logger(msg)
        else:
            print(msg)
    current_thread_count = n
    msg = f"Thread count updated to {current_thread_count}"
    if logger:
        logger(msg)
    else:
        print(msg)

def process_console_command(cmd, IPC=None):
    """
    Process a single command from the interactive console using pattern matching.

    Supported commands:
        - "set threads <number>" : Sets the CPU thread count.
        - "status"               : Prints the current thread count.
        - "exit"                 : Terminates the interactive session.

    If an unrecognized command is entered, an extended help message is displayed.
    All messages are output through the IPC's log() method (if provided) with a timestamp.

    Parameters
    ----------
    cmd : str
        Command entered by the user.
    IPC : InteractiveConsole, optional
        The interactive console instance (to use its log() method).

    Returns
    -------
    str or None
        Returns "exit" if the "exit" command is given; otherwise, None.
    """
    # Define a helper to output messages with timestamp.
    def output(msg):
        if IPC is not None:
            IPC.log(msg)
        else:
            print(f"{_timestamp()} {msg}")

    parts = cmd.strip().split()
    if not parts:
        return None

    match parts:
        case ["set", "threads", number]:
            try:
                n = int(number)
                set_cysgp4_threads(n, logger=IPC.log if IPC is not None else None)
                output(f"Set threads command acknowledged: now using {current_thread_count} threads.\n")
            except ValueError:
                output("Error: Please provide an integer for thread count.\n")
        case ["status"]:
            output(f"Current thread count: {current_thread_count}\n")
        case ["exit"]:
            output("Exiting interactive console and simulation loop...\n")
            return "exit"
        case _:
            help_msg = (
                f"{_timestamp()} [InteractiveConsole] Unknown command.\n\n"
                "Available commands:\n"
                "  set threads <number>\n"
                "      Set the number of CPU threads for simulation. The value is clamped between 1 and 32.\n\n"
                "  status\n"
                "      Display the current number of CPU threads in use.\n\n"
                "  exit\n"
                "      Terminate the interactive session and exit the simulation loop.\n\n"
                "Examples:\n"
                "  >> set threads 16\n"
                "  >> status\n"
                "  >> exit\n"
            )
            output(help_msg)
    return None

class InteractiveConsole:
    """
    A context manager that provides an interactive console via a background thread.

    When entered, the console starts accepting user input via input(">> "),
    and each command is placed into a thread-safe queue. You can call poll_commands()
    to retrieve pending commands. Output (both command acknowledgments and simulation logs)
    is sent using either tqdm.write() or print(), depending on the use_tqdm flag.

    Parameters
    ----------
    use_tqdm : bool, optional
        If True, output messages use tqdm.write(), preserving progress bar placement.
        If False, output is produced via plain print(). Default is True.
    """
    def __init__(self, use_tqdm=True, pbar=None):
        self.use_tqdm = use_tqdm
        if self.use_tqdm:
            try:
                from tqdm import tqdm
                self._output = tqdm.write
            except ImportError:
                self._output = print
        else:
            self._output = print
        self._cmd_queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._console_loop, daemon=True)
        self._pbar = pbar

    def _console_loop(self):
        """
        Runs in a background thread to continuously accept user input.
        Each input line is timestamped and enqueued.
        """
        self._output(f"{_timestamp()} [InteractiveConsole] Ready for commands (>> ).")
        while self._running:
            try:
                if self._pbar is not None:
                   self._pbar.clear()
                cmd = input(">> ")          # Show the prompt and accept input.
                if self._pbar is not None:
                    self._pbar.refresh()
            except EOFError:
                break
            self._output(f"{_timestamp()} [InteractiveConsole] Received command: {cmd}")
            self._cmd_queue.put(cmd)
            if cmd.strip().lower() == "exit":
                break

    def poll_commands(self):
        """
        Retrieve all pending commands from the interactive console.

        Returns
        -------
        list of str
            A list of commands entered by the user.
        """
        cmds = []
        while not self._cmd_queue.empty():
            cmds.append(self._cmd_queue.get())
        return cmds

    def log(self, message):
        """
        Emit a simulation log message via the console output function with a timestamp.

        Parameters
        ----------
        message : str
            The message to be logged.
        """
        self._output(f"{_timestamp()} [Simulation] {message}")

    def __enter__(self):
        """
        Start the interactive console thread upon entering the context.
        
        Returns
        -------
        InteractiveConsole
            The instance itself.
        """
        self._running = True
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Signal the console to terminate and join the background thread.
        """
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        self._output(f"{_timestamp()} [InteractiveConsole] Simulation terminated.")


@contextmanager
def block_interrupts():
    """
    A context manager to temporarily block SIGINT (KeyboardInterrupt).
    This ensures that the enclosed critical section is not interrupted.
    """
    old_handler = signal.getsignal(signal.SIGINT)
    # Ignore SIGINT so that the critical section won't be interrupted.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        # Restore the original SIGINT handler.
        signal.signal(signal.SIGINT, old_handler)

def init_simulation_results(filename):
    """
    Delete the HDF5 file that stores simulation results if it exists.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function deletes the file if it exists. It can be called at the start
    of the simulation to refresh stored results. It's a fancy wrapper for os.remove.
    """
    if os.path.exists(filename):
        os.remove(filename)

def append_simulation_results(filename, times, total_Prx, 
                              selected_sat_antenna_el=None,
                              powerflux_2RAS=None,
                              times_dataset="times", 
                              prx_dataset="total_Prx", 
                              antenna_dataset="selected_satellite_el",
                              powerflux_dataset="powerflux_2RAS"):
    """
    Append simulation results (time stamps, total received power, 
    optionally selected satellite antenna elevation and powerflux_2RAS) 
    to an HDF5 file.

    This function processes the inputs so that:
      - 'times': if the input has an attribute 'mjd' (e.g., an astropy Time object),
         its .mjd values are stored; otherwise the input is converted to a numpy array.
      - 'total_Prx': if the input is an astropy Quantity, it is converted to the unit 
         specified by cnv.dB_W and its .value taken; otherwise a numeric array is used.
      - 'selected_sat_antenna_el': if provided and is a Quantity, its .value is stored.
      - 'powerflux_2RAS': if provided and is a Quantity, it is converted to the unit 
         cnv.dB_W_m2 and its .value taken.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file where data will be stored.
    times : array-like or astropy.time.Time
        Time stamps from the simulation.
    total_Prx : array-like or astropy.Quantity
        Total received power.
    selected_sat_antenna_el : array-like or astropy.Quantity, optional
        Selected satellite antenna elevation.
    powerflux_2RAS : array-like or astropy.Quantity, optional
        Power flux from each satellite, with desired unit cnv.dB_W_m2.
    times_dataset : str, optional
        Dataset name for time stamps. Default is "times".
    prx_dataset : str, optional
        Dataset name for total received power. Default is "total_Prx".
    antenna_dataset : str, optional
        Dataset name for selected satellite antenna elevation. Default is "selected_satellite_el".
    powerflux_dataset : str, optional
        Dataset name for powerflux_2RAS. Default is "powerflux_2RAS".
    
    Notes
    -----
    If the file or datasets do not exist, they are created with an unlimited first dimension.
    Otherwise, the new data are appended along the first (time) dimension.
    Compression (gzip, level 9) is used.
    """
    import gc
    # Process times.
    try:
        times_array = times.mjd
    except AttributeError:
        times_array = np.array(times)
    
    # Process total_Prx.
    if hasattr(total_Prx, "unit"):
        try:
            total_Prx_array = total_Prx.to(cnv.dB_W).value
        except u.UnitConversionError:
            total_Prx_array = total_Prx.value
    else:
        total_Prx_array = np.array(total_Prx)
    
    # Process selected_satellite_el if provided.
    if selected_sat_antenna_el is not None:
        if hasattr(selected_sat_antenna_el, "unit"):
            try:
                antenna_array = selected_sat_antenna_el.to(u.deg).value
            except u.UnitConversionError:
                antenna_array = selected_sat_antenna_el.value
        else:
            antenna_array = np.array(selected_sat_antenna_el)
    # Process powerflux_2RAS if provided.
    if powerflux_2RAS is not None:
        if hasattr(powerflux_2RAS, "unit"):
            try:
                powerflux_array = powerflux_2RAS.to(cnv.dB_W_m2).value
            except u.UnitConversionError:
                powerflux_array = powerflux_2RAS.value
        else:
            powerflux_array = np.array(powerflux_2RAS)
    
    # Determine number of new time samples.
    n_new = times_array.shape[0]
    
    with h5py.File(filename, "a") as f:
        # Append times.
        if times_dataset in f:
            dset_times = f[times_dataset]
            old_len = dset_times.shape[0]
            new_len = old_len + n_new
            dset_times.resize((new_len,))
            dset_times[old_len:new_len] = times_array
        else:
            f.create_dataset(
                times_dataset, data=times_array,
                maxshape=(None,), chunks=True,
                compression="gzip", compression_opts=9
            )
        # Append total_Prx.
        if prx_dataset in f:
            dset_prx = f[prx_dataset]
            old_len = dset_prx.shape[0]
            new_len = old_len + n_new
            dset_prx.resize((new_len,) + dset_prx.shape[1:])
            dset_prx[old_len:new_len, ...] = total_Prx_array
        else:
            new_shape = (None,) + total_Prx_array.shape[1:]
            f.create_dataset(
                prx_dataset, data=total_Prx_array,
                maxshape=new_shape, chunks=True,
                compression="gzip", compression_opts=9
            )
        # Append selected_satellite_el if provided.
        if selected_sat_antenna_el is not None:
            if antenna_dataset in f:
                dset_ant = f[antenna_dataset]
                old_len = dset_ant.shape[0]
                new_len = old_len + n_new
                dset_ant.resize((new_len,) + dset_ant.shape[1:])
                dset_ant[old_len:new_len, ...] = antenna_array
            else:
                new_shape = (None,) + antenna_array.shape[1:]
                f.create_dataset(
                    antenna_dataset, data=antenna_array,
                    maxshape=new_shape, chunks=True,
                    compression="gzip", compression_opts=9
                )
        # Append powerflux_2RAS if provided.
        if powerflux_2RAS is not None:
            if powerflux_dataset in f:
                dset_pf = f[powerflux_dataset]
                old_len = dset_pf.shape[0]
                new_len = old_len + n_new
                dset_pf.resize((new_len,) + dset_pf.shape[1:])
                dset_pf[old_len:new_len, ...] = powerflux_array
            else:
                new_shape = (None,) + powerflux_array.shape[1:]
                f.create_dataset(
                    powerflux_dataset, data=powerflux_array,
                    maxshape=new_shape, chunks=True,
                    compression="gzip", compression_opts=9
                )
            
def read_simulation_results(filename, 
                            times_dataset="times", 
                            prx_dataset="total_Prx", 
                            antenna_dataset="selected_satellite_el",
                            powerflux_dataset="powerflux_2RAS"):
    """
    Read simulation results from the specified HDF5 file and return them in one dictionary.

    For known datasets, appropriate post-processing is performed:
      - "times": Converted from stored MJD values to Python datetime objects.
      - "total_Prx": Converted to an astropy Quantity with unit cnv.dB_W.
      - "selected_satellite_el": If available, reattached with u.deg.
      - "powerflux_2RAS": If available, reattached with the unit cnv.dB_W_m2.
      - Any other datasets are returned as plain NumPy arrays.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing simulation results.
    times_dataset : str, optional
        Dataset name for time stamps. Default is "times".
    prx_dataset : str, optional
        Dataset name for total received power. Default is "total_Prx".
    antenna_dataset : str, optional
        Dataset name for selected satellite antenna elevation. Default is "selected_satellite_el".
    powerflux_dataset : str, optional
        Dataset name for powerflux_2RAS. Default is "powerflux_2RAS".

    Returns
    -------
    results : dict
        A dictionary with keys corresponding to dataset names and values processed as follows:
          - "times": array of Python datetime objects.
          - "total_Prx": astropy Quantity in cnv.dB_W.
          - "selected_satellite_el": astropy Quantity in degrees (if present).
          - "powerflux_2RAS": astropy Quantity in cnv.dB_W_m2 (if present).
          - Any other dataset: returned as a plain NumPy array.
    """

    results = {}
    import h5py
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            data = f[key][:]
            if key == times_dataset:
                # Convert stored MJD values to datetime objects.
                results[key] = Time(data, format="mjd").to_datetime()
            elif key == prx_dataset:
                results[key] = data * cnv.dB_W
            elif key == antenna_dataset:
                results[key] = data * u.deg
            elif key == powerflux_dataset:
                results[key] = data * cnv.dB_W_m2
            else:
                results[key] = data
    return results

def generate_simulation_batches(start_time, end_time, timestep, batch_size):
    """
    Generate batches of simulation times between start_time and end_time.

    Each batch contains up to 'batch_size' timesteps. The function returns a dictionary 
    with keys corresponding to each batch parameter. Specifically, it returns a dictionary
    with:
      - 'batch_start': list of batch start times (astropy Time objects),
      - 'times': list of astropy Time arrays for each batch,
      - 'td': list of TimeDelta arrays for each batch,
      - 'batch_end': list of batch end times (astropy Time objects).
    
    Parameters
    ----------
    start_time : astropy.time.Time
        Global simulation start time.
    end_time : astropy.time.Time
        Global simulation end time.
    timestep : float
        Time step in seconds.
    batch_size : int
        Maximum number of timesteps per batch.

    Returns
    -------
    batches : dict
        A dictionary containing lists for each batch parameter.
    """
    total_duration_sec = (end_time - start_time).sec
    total_steps = int(np.ceil(total_duration_sec / timestep))+1
    
    # Prepare lists to store batch parameters.
    batch_start_list = []
    batch_times_list = []
    batch_td_list = []
    batch_end_list = []
    
    for batch_start_idx in range(0, total_steps, batch_size):
        # Determine batch indices.
        batch_end_idx = min(batch_start_idx + batch_size, total_steps)
        n_steps_in_batch = batch_end_idx - batch_start_idx
        
        # Create the time delta array for the batch.
        batch_td_array = np.arange(0, n_steps_in_batch * timestep, timestep)
        
        # Compute the batch's start time.
        batch_start_time = start_time + TimeDelta(batch_start_idx * timestep, format='sec')
        # Compute the simulation times for the batch.
        batch_times = batch_start_time + TimeDelta(batch_td_array, format='sec')
        # The batch end time is the last time in the batch.
        batch_end_time = batch_times[-1]
        
        # Append results to lists.
        batch_start_list.append(batch_start_time)
        batch_times_list.append(batch_times)
        batch_td_list.append(TimeDelta(batch_td_array, format='sec'))
        batch_end_list.append(batch_end_time)
    
    # Return a dictionary containing all batch information.
    batches = {
        'batch_start': batch_start_list,
        'times': batch_times_list,
        'td': batch_td_list,
        'batch_end': batch_end_list
    }
    
    return batches
