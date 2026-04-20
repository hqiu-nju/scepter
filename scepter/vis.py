#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vis.py - Visibility Simulation and Measurement Set Generation

This module provides a high-level interface for generating simulated radio
interferometric visibilities using CASA's simulation toolkit and saving the
output as a CASA Measurement Set (MS).  It bridges SCEPTer's satellite RFI
simulation infrastructure (``scepter.obs``, ``scepter.uvw``, ``scepter.array``)
with CASA's ``casatools`` / ``casatasks`` packages.

Overview
--------
The module supports two primary workflows:

1. **Empty MS creation** — build a Measurement Set from scratch given an
   antenna configuration, spectral setup, and observing schedule, then
   optionally inject model visibilities (e.g. satellite RFI from
   ``obs_sim``).

2. **Visibility prediction** — use CASA's simulator tool (``sm``) to
   predict visibilities for a component-list sky model or to corrupt them
   with noise and atmospheric effects, producing a realistic MS that can
   be imaged with ``tclean``.

Both workflows accept SCEPTer objects (``cysgp4.PyObserver`` antenna
positions, ``astropy.units.Quantity`` frequencies, ``astropy.time.Time``
observation times) and translate them into CASA-native representations
internally.

Dependencies
------------
Required (CASA 6 modular):
    - casatools  (``simulator``, ``measures``, ``table``, ``quanta``)
    - casatasks  (``simobserve``, optionally ``tclean``)

Core (always available):
    - numpy, astropy (units, time, coordinates)

Optional integration:
    - scepter.obs  — ``obs_sim`` objects for RFI injection
    - scepter.uvw  — UVW recomputation / cross-checks
    - scepter.array — ``baseline_pairs`` for antenna arrays

Usage Examples
--------------
**Create an empty MS for a small array and inject flat-spectrum noise**::

    >>> from scepter.vis import VisibilitySimulator
    >>> from cysgp4 import PyObserver
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>>
    >>> antennas = [
    ...     PyObserver(21.443, -30.713, 1.0),   # lon, lat (deg), alt (km)
    ...     PyObserver(21.444, -30.713, 1.0),
    ...     PyObserver(21.445, -30.714, 1.0),
    ... ]
    >>> vis_sim = VisibilitySimulator(
    ...     antennas=antennas,
    ...     freq_center=1420.0 * u.MHz,
    ...     freq_bandwidth=10.0 * u.MHz,
    ...     n_channels=64,
    ...     integration_time=1.0 * u.s,
    ... )
    >>> vis_sim.create_empty_ms(
    ...     ms_path="test_observation.ms",
    ...     obs_time=Time("2026-01-15T00:00:00"),
    ...     obs_duration=3600.0 * u.s,
    ...     phase_center_ra=0.0 * u.deg,
    ...     phase_center_dec=-30.0 * u.deg,
    ... )

**Predict visibilities with a component sky model**::

    >>> vis_sim.predict(
    ...     ms_path="test_observation.ms",
    ...     component_list="my_sources.cl",
    ... )

**Inject SCEPTer RFI into an existing MS**::

    >>> from scepter.obs import obs_sim
    >>> # ... set up obs_sim with satellite RFI ...
    >>> vis_sim.inject_rfi(
    ...     ms_path="test_observation.ms",
    ...     obs=my_obs_sim,
    ...     column="DATA",
    ... )

References
----------
- CASA 6 documentation: https://casadocs.readthedocs.io/
- CASA simulation tutorial:
  https://casaguides.nrao.edu/index.php/Simulating_Observations_in_CASA
- Thompson, Moran & Swenson, "Interferometry and Synthesis in Radio
  Astronomy", 3rd Ed., Springer (2017)

Authors
-------
- Generated for the SCEPTer project

Date Created: 2026-04-20
Version: 0.1
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from casatools import simulator as casa_simulator
    from casatools import measures as casa_measures
    from casatools import quanta as casa_quanta
    from casatools import table as casa_table

    CASATOOLS_AVAILABLE = True
except ImportError:
    CASATOOLS_AVAILABLE = False

try:
    from casatasks import simobserve

    CASATASKS_AVAILABLE = True
except ImportError:
    CASATASKS_AVAILABLE = False

try:
    import cysgp4

    CYSGP4_AVAILABLE = True
except ImportError:
    CYSGP4_AVAILABLE = False


def _require_casatools() -> None:
    """Raise ``ImportError`` if casatools is not installed."""
    if not CASATOOLS_AVAILABLE:
        raise ImportError(
            "casatools is required for visibility simulation.  "
            "Install CASA 6 modular: pip install casatools casatasks"
        )


# ---------------------------------------------------------------------------
# Helpers: unit / coordinate conversion to CASA-native formats
# ---------------------------------------------------------------------------


def _quantity_to_casa(value: u.Quantity, casa_unit: str) -> str:
    """Convert an astropy Quantity to a CASA quanta string.

    Parameters
    ----------
    value : astropy.units.Quantity
        Physical quantity with units attached.
    casa_unit : str
        Target CASA unit string (e.g. ``"GHz"``, ``"m"``, ``"deg"``).

    Returns
    -------
    str
        CASA-compatible quantity string, e.g. ``"1.42GHz"``.
    """
    converted = value.to(u.Unit(casa_unit)).value
    return f"{converted}{casa_unit}"


def _time_to_casa_epoch(obs_time: Time) -> Dict[str, Any]:
    """Convert an astropy Time to a CASA measures epoch dict.

    Parameters
    ----------
    obs_time : astropy.time.Time
        Observation time.

    Returns
    -------
    dict
        CASA epoch record suitable for ``me.epoch()``.
    """
    _require_casatools()
    me = casa_measures()
    return me.epoch("UTC", f"{obs_time.mjd}d")


def _direction_j2000(ra: u.Quantity, dec: u.Quantity) -> Dict[str, Any]:
    """Build a CASA J2000 direction measure.

    Parameters
    ----------
    ra : astropy.units.Quantity
        Right ascension (angle unit).
    dec : astropy.units.Quantity
        Declination (angle unit).

    Returns
    -------
    dict
        CASA direction record.
    """
    _require_casatools()
    me = casa_measures()
    return me.direction(
        "J2000",
        _quantity_to_casa(ra, "deg"),
        _quantity_to_casa(dec, "deg"),
    )


def _pyobserver_to_earth_location(obs: "cysgp4.PyObserver") -> EarthLocation:
    """Convert a cysgp4 PyObserver to an astropy EarthLocation.

    Parameters
    ----------
    obs : cysgp4.PyObserver
        Observer with ``.loc`` attributes (lon deg, lat deg, alt km).

    Returns
    -------
    astropy.coordinates.EarthLocation
    """
    return EarthLocation.from_geodetic(
        lon=obs.loc.lon * u.deg,
        lat=obs.loc.lat * u.deg,
        height=obs.loc.alt * u.km,
    )


# ---------------------------------------------------------------------------
# Antenna configuration writer
# ---------------------------------------------------------------------------


def write_antenna_config(
    antennas: Sequence,
    config_path: Union[str, Path],
    dish_diameter: u.Quantity = 25.0 * u.m,
    antenna_names: Optional[Sequence[str]] = None,
    observatory_name: str = "SCEPTER_ARRAY",
) -> Path:
    """Write an antenna configuration file in CASA ``.cfg`` format.

    The file lists antenna positions in ITRF Cartesian coordinates (meters),
    one antenna per line, compatible with ``sm.setconfig()``.

    Parameters
    ----------
    antennas : sequence of cysgp4.PyObserver or astropy.coordinates.EarthLocation
        Antenna positions.  ``PyObserver`` objects are transparently converted
        to ``EarthLocation``.
    config_path : str or pathlib.Path
        Output file path (e.g. ``"my_array.cfg"``).
    dish_diameter : astropy.units.Quantity, optional
        Dish diameter for all antennas (default 25 m).
    antenna_names : sequence of str, optional
        Per-antenna names.  Defaults to ``ANT00``, ``ANT01``, …
    observatory_name : str, optional
        Name written in the file header (default ``"SCEPTER_ARRAY"``).

    Returns
    -------
    pathlib.Path
        Resolved path to the written file.

    Raises
    ------
    ValueError
        If *antennas* is empty.
    TypeError
        If antenna type is unsupported.

    Notes
    -----
    The CASA ``.cfg`` format expects one antenna per line with columns::

        X  Y  Z  dish_diameter  antenna_name

    where X, Y, Z are ITRF geocentric Cartesian coordinates in meters.
    """
    config_path = Path(config_path)
    if len(antennas) == 0:
        raise ValueError("At least one antenna is required.")

    locations: list[EarthLocation] = []
    for ant in antennas:
        if CYSGP4_AVAILABLE and isinstance(ant, cysgp4.PyObserver):
            locations.append(_pyobserver_to_earth_location(ant))
        elif isinstance(ant, EarthLocation):
            locations.append(ant)
        else:
            raise TypeError(
                f"Unsupported antenna type {type(ant).__name__}; "
                "expected cysgp4.PyObserver or astropy.coordinates.EarthLocation."
            )

    if antenna_names is None:
        antenna_names = [f"ANT{i:02d}" for i in range(len(locations))]

    diam_m = dish_diameter.to(u.m).value

    with open(config_path, "w") as fh:
        fh.write(f"# observatory={observatory_name}\n")
        fh.write("# coordsys=XYZ\n")
        fh.write("# X Y Z dish_diameter station\n")
        for loc, name in zip(locations, antenna_names):
            x, y, z = loc.geocentric
            fh.write(
                f"{x.to(u.m).value:.6f}  "
                f"{y.to(u.m).value:.6f}  "
                f"{z.to(u.m).value:.6f}  "
                f"{diam_m:.2f}  "
                f"{name}\n"
            )

    logger.info("Wrote antenna config to %s (%d antennas)", config_path, len(locations))
    return config_path.resolve()


# ---------------------------------------------------------------------------
# Core dataclass: spectral window definition
# ---------------------------------------------------------------------------


@dataclass
class SpectralWindow:
    """Description of a single spectral window (SPW).

    Parameters
    ----------
    freq_center : astropy.units.Quantity
        Centre frequency of the spectral window.
    freq_bandwidth : astropy.units.Quantity
        Total bandwidth.
    n_channels : int
        Number of frequency channels.
    stokes : str
        Stokes parameter string (default ``"RR RL LR LL"``).
    spw_name : str
        Name label for the spectral window (default ``"SPW0"``).

    Notes
    -----
    Channel width is derived as ``freq_bandwidth / n_channels``.  The
    channel frequencies span symmetrically about ``freq_center``.
    """

    freq_center: u.Quantity
    freq_bandwidth: u.Quantity
    n_channels: int = 64
    stokes: str = "RR RL LR LL"
    spw_name: str = "SPW0"

    @property
    def channel_width(self) -> u.Quantity:
        """Channel width (derived from bandwidth / n_channels)."""
        return self.freq_bandwidth / self.n_channels


# ---------------------------------------------------------------------------
# Main class: VisibilitySimulator
# ---------------------------------------------------------------------------


class VisibilitySimulator:
    """High-level interface for generating visibilities and Measurement Sets.

    This class wraps CASA's ``casatools.simulator`` to create empty
    Measurement Sets, predict model visibilities, add noise/corruption,
    and inject SCEPTer-computed RFI signals.

    Parameters
    ----------
    antennas : sequence
        Antenna positions as ``cysgp4.PyObserver`` or
        ``astropy.coordinates.EarthLocation`` objects.
    freq_center : astropy.units.Quantity
        Centre frequency of the default spectral window.
    freq_bandwidth : astropy.units.Quantity
        Total bandwidth of the default spectral window.
    n_channels : int, optional
        Number of channels in the default spectral window (default 64).
    integration_time : astropy.units.Quantity, optional
        Correlator integration time (default 1 s).
    dish_diameter : astropy.units.Quantity, optional
        Antenna dish diameter (default 25 m).
    observatory_name : str, optional
        Observatory identifier written into the MS (default
        ``"SCEPTER_ARRAY"``).
    stokes : str, optional
        Stokes polarisation products (default ``"RR RL LR LL"``).
    extra_spws : list of SpectralWindow, optional
        Additional spectral windows beyond the default one.

    Raises
    ------
    ImportError
        If ``casatools`` is not installed.
    ValueError
        If *antennas* is empty.

    Examples
    --------
    >>> from scepter.vis import VisibilitySimulator
    >>> from astropy.coordinates import EarthLocation
    >>> from astropy import units as u
    >>> locs = [
    ...     EarthLocation.from_geodetic(21.44, -30.71, 1000 * u.m),
    ...     EarthLocation.from_geodetic(21.45, -30.71, 1000 * u.m),
    ... ]
    >>> sim = VisibilitySimulator(
    ...     antennas=locs,
    ...     freq_center=1420 * u.MHz,
    ...     freq_bandwidth=10 * u.MHz,
    ... )
    """

    def __init__(
        self,
        antennas: Sequence,
        freq_center: u.Quantity,
        freq_bandwidth: u.Quantity,
        n_channels: int = 64,
        integration_time: u.Quantity = 1.0 * u.s,
        dish_diameter: u.Quantity = 25.0 * u.m,
        observatory_name: str = "SCEPTER_ARRAY",
        stokes: str = "RR RL LR LL",
        extra_spws: Optional[List[SpectralWindow]] = None,
    ) -> None:
        _require_casatools()

        if len(antennas) == 0:
            raise ValueError("At least one antenna is required.")

        self.antennas = list(antennas)
        self.integration_time = integration_time
        self.dish_diameter = dish_diameter
        self.observatory_name = observatory_name

        # Build the default spectral window
        self.spws: List[SpectralWindow] = [
            SpectralWindow(
                freq_center=freq_center,
                freq_bandwidth=freq_bandwidth,
                n_channels=n_channels,
                stokes=stokes,
                spw_name="SPW0",
            )
        ]
        if extra_spws:
            for i, spw in enumerate(extra_spws, start=1):
                if not spw.spw_name or spw.spw_name == "SPW0":
                    spw = SpectralWindow(
                        freq_center=spw.freq_center,
                        freq_bandwidth=spw.freq_bandwidth,
                        n_channels=spw.n_channels,
                        stokes=spw.stokes,
                        spw_name=f"SPW{i}",
                    )
                self.spws.append(spw)

        # Lazily-written antenna config file (cached path)
        self._config_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_config(self, work_dir: Union[str, Path]) -> Path:
        """Write an antenna config file if not already written.

        Parameters
        ----------
        work_dir : str or pathlib.Path
            Directory to place the ``.cfg`` file.

        Returns
        -------
        pathlib.Path
            Path to the configuration file.
        """
        if self._config_path is not None and self._config_path.exists():
            return self._config_path

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = work_dir / f"{self.observatory_name.lower()}.cfg"
        self._config_path = write_antenna_config(
            self.antennas,
            cfg_path,
            dish_diameter=self.dish_diameter,
            observatory_name=self.observatory_name,
        )
        return self._config_path

    # ------------------------------------------------------------------
    # Public API: create an empty Measurement Set
    # ------------------------------------------------------------------

    def create_empty_ms(
        self,
        ms_path: Union[str, Path],
        obs_time: Time,
        obs_duration: u.Quantity,
        phase_center_ra: u.Quantity,
        phase_center_dec: u.Quantity,
        *,
        source_name: str = "SCEPTER_FIELD",
        overwrite: bool = False,
        auto_correlations: bool = False,
    ) -> Path:
        """Create an empty Measurement Set with the configured array.

        The MS contains correct UVW coordinates, timestamps, and spectral
        metadata but all visibilities are initialised to zero.  Use
        :meth:`predict` or :meth:`inject_rfi` to fill data afterwards.

        Parameters
        ----------
        ms_path : str or pathlib.Path
            Output Measurement Set path (directory name, e.g. ``"obs.ms"``).
        obs_time : astropy.time.Time
            Start time of the observation (UTC).
        obs_duration : astropy.units.Quantity
            Total observation duration.
        phase_center_ra : astropy.units.Quantity
            Right ascension of the phase tracking centre (angle).
        phase_center_dec : astropy.units.Quantity
            Declination of the phase tracking centre (angle).
        source_name : str, optional
            Source / field name stored in the MS (default ``"SCEPTER_FIELD"``).
        overwrite : bool, optional
            If *True*, delete any existing MS at *ms_path* before creating
            (default *False*).
        auto_correlations : bool, optional
            Include auto-correlations (default *False*).

        Returns
        -------
        pathlib.Path
            Resolved path to the created Measurement Set.

        Raises
        ------
        FileExistsError
            If *ms_path* already exists and *overwrite* is *False*.
        RuntimeError
            If CASA's simulator tool reports an error.

        Notes
        -----
        Internally this method:

        1. Writes an antenna ``.cfg`` file (ITRF XYZ format).
        2. Opens the CASA ``simulator`` tool.
        3. Configures the antenna layout, spectral windows, feeds, and
           field (phase centre).
        4. Sets the observation time reference and integration time.
        5. Calls ``sm.observe()`` to populate the MS skeleton.
        6. Closes the tool cleanly.

        The generated MS is fully compatible with downstream CASA tasks
        (``tclean``, ``plotms``, ``listobs``, etc.).

        Examples
        --------
        >>> from scepter.vis import VisibilitySimulator
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> sim = VisibilitySimulator(...)
        >>> ms = sim.create_empty_ms(
        ...     ms_path="my_obs.ms",
        ...     obs_time=Time("2026-06-15T04:00:00"),
        ...     obs_duration=7200 * u.s,
        ...     phase_center_ra=180.0 * u.deg,
        ...     phase_center_dec=-45.0 * u.deg,
        ... )
        """
        _require_casatools()
        ms_path = Path(ms_path).resolve()

        if ms_path.exists():
            if overwrite:
                shutil.rmtree(ms_path)
                logger.info("Removed existing MS: %s", ms_path)
            else:
                raise FileExistsError(
                    f"MS already exists: {ms_path}.  Pass overwrite=True to replace."
                )

        # Ensure antenna config file is written
        cfg_path = self._ensure_config(ms_path.parent)

        # --- CASA simulator tool ---
        sm = casa_simulator()
        me = casa_measures()
        qa = casa_quanta()

        try:
            sm.open(str(ms_path))

            # -- Antenna configuration --
            # Read the config file we wrote and set it via setconfig
            sm.setconfig(
                telescopename=self.observatory_name,
                x=self._antenna_x(),
                y=self._antenna_y(),
                z=self._antenna_z(),
                dishdiameter=self._antenna_diameters(),
                mount=["alt-az"] * len(self.antennas),
                antname=self._antenna_names(),
                coordsystem="global",
                referencelocation=me.observatory(self.observatory_name)
                if self.observatory_name in me.obslist()
                else self._reference_position(),
            )

            # -- Spectral windows --
            for spw in self.spws:
                sm.setspwindow(
                    spwname=spw.spw_name,
                    freq=_quantity_to_casa(spw.freq_center, "GHz"),
                    deltafreq=_quantity_to_casa(spw.channel_width, "MHz"),
                    freqresolution=_quantity_to_casa(spw.channel_width, "MHz"),
                    nchannels=spw.n_channels,
                    stokes=spw.stokes,
                )

            # -- Feed (dual circular by default) --
            sm.setfeed("perfect R L", x=[0.0], y=[0.0], pol=[""])

            # -- Field / phase centre --
            direction = _direction_j2000(phase_center_ra, phase_center_dec)
            sm.setfield(
                sourcename=source_name,
                sourcedirection=direction,
            )

            # -- Observation limits --
            sm.setlimits(shadowlimit=0.001, elevationlimit="8.0deg")
            sm.setauto(autocorrwt=1.0 if auto_correlations else 0.0)

            # -- Time reference and integration --
            ref_epoch = _time_to_casa_epoch(obs_time)
            sm.settimes(
                integrationtime=_quantity_to_casa(self.integration_time, "s"),
                usehourangle=False,
                referencetime=ref_epoch,
            )

            # -- Run the observation --
            duration_s = obs_duration.to(u.s).value
            start_s = qa.quantity(0, "s")
            stop_s = qa.quantity(duration_s, "s")

            for spw in self.spws:
                sm.observe(
                    sourcename=source_name,
                    spwname=spw.spw_name,
                    starttime=start_s,
                    stoptime=stop_s,
                )

            logger.info(
                "Created MS: %s  (%d antennas, %d SPWs, %.0f s)",
                ms_path,
                len(self.antennas),
                len(self.spws),
                duration_s,
            )
        finally:
            sm.close()

        return ms_path

    # ------------------------------------------------------------------
    # Public API: predict visibilities from a sky model
    # ------------------------------------------------------------------

    def predict(
        self,
        ms_path: Union[str, Path],
        component_list: Optional[Union[str, Path]] = None,
        sky_model: Optional[Union[str, Path]] = None,
    ) -> None:
        """Predict model visibilities into an existing MS.

        Uses the CASA simulator tool to Fourier-transform a sky model
        (component list or FITS image) and write the result into the
        ``MODEL_DATA`` column of the MS.

        Parameters
        ----------
        ms_path : str or pathlib.Path
            Path to an existing Measurement Set.
        component_list : str or pathlib.Path, optional
            Path to a CASA component list (``.cl``).
        sky_model : str or pathlib.Path, optional
            Path to a FITS sky-model image.

        Raises
        ------
        FileNotFoundError
            If *ms_path* does not exist.
        ValueError
            If neither *component_list* nor *sky_model* is provided.
        RuntimeError
            If CASA's simulator tool reports an error.

        Notes
        -----
        At least one of *component_list* or *sky_model* must be given.
        If both are provided, :func:`sm.predict` applies them
        sequentially—component list first, then the image model is added.

        The predicted visibilities are written to the MS ``MODEL_DATA``
        column (or ``DATA`` if ``MODEL_DATA`` does not exist).
        """
        _require_casatools()
        ms_path = Path(ms_path).resolve()
        if not ms_path.exists():
            raise FileNotFoundError(f"MS not found: {ms_path}")
        if component_list is None and sky_model is None:
            raise ValueError("Provide at least one of component_list or sky_model.")

        sm = casa_simulator()
        try:
            sm.openfromms(str(ms_path))
            sm.setvp(dovp=True, usedefaultvp=True)

            if component_list is not None:
                sm.predict(complist=str(Path(component_list).resolve()))
            if sky_model is not None:
                sm.predict(imagename=[str(Path(sky_model).resolve())])
        finally:
            sm.close()

        logger.info("Predicted visibilities into %s", ms_path)

    # ------------------------------------------------------------------
    # Public API: add noise / corruption
    # ------------------------------------------------------------------

    def corrupt(
        self,
        ms_path: Union[str, Path],
        *,
        add_noise: bool = True,
        noise_mode: str = "simplenoise",
        simplenoise_rms: u.Quantity = 0.1 * u.Jy,
    ) -> None:
        """Add noise or atmospheric corruption to an existing MS.

        Parameters
        ----------
        ms_path : str or pathlib.Path
            Path to an existing Measurement Set.
        add_noise : bool, optional
            Whether to add thermal noise (default *True*).
        noise_mode : str, optional
            Noise mode for ``sm.setnoise()`` (default ``"simplenoise"``).
            Options: ``"simplenoise"``, ``"tsys-atm"``, ``"tsys-manual"``.
        simplenoise_rms : astropy.units.Quantity, optional
            Noise RMS per visibility when *noise_mode* is ``"simplenoise"``
            (default 0.1 Jy).

        Raises
        ------
        FileNotFoundError
            If *ms_path* does not exist.

        Notes
        -----
        This wraps ``sm.setnoise()`` and ``sm.corrupt()`` to inject
        realistic thermal noise.  For atmospheric phase corruption, use
        ``noise_mode="tsys-atm"``.
        """
        _require_casatools()
        ms_path = Path(ms_path).resolve()
        if not ms_path.exists():
            raise FileNotFoundError(f"MS not found: {ms_path}")

        sm = casa_simulator()
        try:
            sm.openfromms(str(ms_path))

            if add_noise:
                sm.setnoise(
                    mode=noise_mode,
                    simplenoise=_quantity_to_casa(simplenoise_rms, "Jy"),
                )

            sm.corrupt()
        finally:
            sm.close()

        logger.info("Applied corruption to %s (noise=%s)", ms_path, add_noise)

    # ------------------------------------------------------------------
    # Public API: inject SCEPTer RFI into an existing MS
    # ------------------------------------------------------------------

    def inject_rfi(
        self,
        ms_path: Union[str, Path],
        visibilities: np.ndarray,
        *,
        column: str = "DATA",
        additive: bool = True,
    ) -> None:
        """Inject externally computed visibilities into an existing MS.

        This is the primary integration point for SCEPTer-simulated
        satellite RFI.  The caller prepares a complex visibility array
        (e.g. from ``obs_sim`` fringe products) and this method writes it
        into the specified MS column.

        Parameters
        ----------
        ms_path : str or pathlib.Path
            Path to an existing Measurement Set.
        visibilities : numpy.ndarray, complex
            Visibility data to inject.  Shape must be
            ``(n_rows, n_channels, n_correlations)`` matching the MS.
            For a single-polarisation injection broadcast to all
            correlations, a shape of ``(n_rows, n_channels)`` is also
            accepted and will be broadcast.
        column : str, optional
            MS column to write into (default ``"DATA"``).  Common choices:
            ``"DATA"``, ``"MODEL_DATA"``, ``"CORRECTED_DATA"``.
        additive : bool, optional
            If *True* (default), add *visibilities* to the existing column
            content.  If *False*, overwrite the column entirely.

        Raises
        ------
        FileNotFoundError
            If *ms_path* does not exist.
        ValueError
            If *visibilities* shape is incompatible with the MS.

        Notes
        -----
        The method opens the MS using ``casatools.table``, reads the
        existing column (if *additive*), adds the supplied array, and
        writes back.  This is safe for concurrent column access but
        callers should avoid concurrent writes to the same column.

        For large MSs (> 10 M rows), consider writing in chunks via
        :meth:`inject_rfi_chunked`.
        """
        _require_casatools()
        ms_path = Path(ms_path).resolve()
        if not ms_path.exists():
            raise FileNotFoundError(f"MS not found: {ms_path}")

        tb = casa_table()
        try:
            tb.open(str(ms_path), nomodify=False)
            n_rows = tb.nrows()

            existing = tb.getcol(column)  # shape (n_corr, n_chan, n_rows)
            # CASA stores columns in Fortran order: (n_corr, n_chan, n_row)
            # Transpose to row-major for user convenience then back.
            existing_T = existing.transpose((2, 1, 0))  # (n_row, n_chan, n_corr)

            vis = np.asarray(visibilities, dtype=np.complex128)

            # Allow (n_row, n_chan) → broadcast to all correlations
            if vis.ndim == 2:
                vis = vis[:, :, np.newaxis]

            if vis.shape[0] != existing_T.shape[0]:
                raise ValueError(
                    f"Row count mismatch: visibilities have {vis.shape[0]} rows, "
                    f"MS has {existing_T.shape[0]} rows."
                )
            if vis.shape[1] != existing_T.shape[1]:
                raise ValueError(
                    f"Channel count mismatch: visibilities have {vis.shape[1]} "
                    f"channels, MS has {existing_T.shape[1]}."
                )

            # Broadcast correlations if needed
            if vis.shape[2] == 1 and existing_T.shape[2] > 1:
                vis = np.broadcast_to(vis, existing_T.shape).copy()

            if additive:
                result = existing_T + vis
            else:
                result = vis

            # Transpose back to CASA order (n_corr, n_chan, n_row)
            tb.putcol(column, result.transpose((2, 1, 0)))
        finally:
            tb.close()

        logger.info(
            "Injected RFI into %s:%s (%d rows, additive=%s)",
            ms_path,
            column,
            n_rows,
            additive,
        )

    # ------------------------------------------------------------------
    # Public API: high-level simobserve wrapper
    # ------------------------------------------------------------------

    def simobserve(
        self,
        sky_model: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        obs_time: Time,
        obs_duration: u.Quantity,
        phase_center_ra: u.Quantity,
        phase_center_dec: u.Quantity,
        total_time: Optional[u.Quantity] = None,
        graphics: str = "none",
        overwrite: bool = False,
    ) -> Path:
        """Run ``casatasks.simobserve`` end-to-end.

        This is a convenience wrapper around ``simobserve()`` that
        automatically writes the antenna configuration file and sets up
        all parameters.

        Parameters
        ----------
        sky_model : str or pathlib.Path
            FITS image of the sky model to observe.
        output_dir : str or pathlib.Path
            Output directory for the simulated MS and images.
        obs_time : astropy.time.Time
            Reference observation time.
        obs_duration : astropy.units.Quantity
            Total integration time.
        phase_center_ra : astropy.units.Quantity
            Phase centre right ascension.
        phase_center_dec : astropy.units.Quantity
            Phase centre declination.
        total_time : astropy.units.Quantity, optional
            Total on-source time (defaults to *obs_duration*).
        graphics : str, optional
            CASA graphics mode (``"none"``, ``"file"``, ``"screen"``).
            Default ``"none"``.
        overwrite : bool, optional
            Overwrite existing outputs (default *False*).

        Returns
        -------
        pathlib.Path
            Path to the output directory containing the MS product.

        Raises
        ------
        ImportError
            If ``casatasks`` is not installed.
        FileNotFoundError
            If *sky_model* does not exist.
        """
        if not CASATASKS_AVAILABLE:
            raise ImportError(
                "casatasks is required for simobserve.  "
                "Install via: pip install casatasks"
            )

        sky_model = Path(sky_model).resolve()
        output_dir = Path(output_dir).resolve()
        if not sky_model.exists():
            raise FileNotFoundError(f"Sky model not found: {sky_model}")

        output_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self._ensure_config(output_dir)

        if total_time is None:
            total_time = obs_duration

        direction_str = (
            f"J2000 "
            f"{phase_center_ra.to(u.deg).value / 15.0:.6f}h "
            f"{phase_center_dec.to(u.deg).value:+.6f}deg"
        )

        simobserve(
            project=str(output_dir / "sim"),
            skymodel=str(sky_model),
            antennalist=str(cfg_path),
            totaltime=_quantity_to_casa(total_time, "s"),
            integration=_quantity_to_casa(self.integration_time, "s"),
            direction=direction_str,
            refdate=obs_time.iso[:10],
            graphics=graphics,
            overwrite=overwrite,
        )

        logger.info("simobserve complete → %s", output_dir)
        return output_dir

    # ------------------------------------------------------------------
    # Public API: read back visibilities from MS
    # ------------------------------------------------------------------

    @staticmethod
    def read_ms_visibilities(
        ms_path: Union[str, Path],
        column: str = "DATA",
    ) -> Dict[str, np.ndarray]:
        """Read visibility data and metadata from a Measurement Set.

        Parameters
        ----------
        ms_path : str or pathlib.Path
            Path to the Measurement Set.
        column : str, optional
            Data column to read (default ``"DATA"``).

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"data"`` : complex128 array, shape ``(n_row, n_chan, n_corr)``
            - ``"uvw"``  : float64 array, shape ``(n_row, 3)``
            - ``"time"`` : float64 array, shape ``(n_row,)`` — MJD seconds
            - ``"antenna1"`` : int array, shape ``(n_row,)``
            - ``"antenna2"`` : int array, shape ``(n_row,)``
            - ``"flag"`` : bool array, same shape as ``"data"``

        Raises
        ------
        FileNotFoundError
            If *ms_path* does not exist.
        """
        _require_casatools()
        ms_path = Path(ms_path).resolve()
        if not ms_path.exists():
            raise FileNotFoundError(f"MS not found: {ms_path}")

        tb = casa_table()
        result: Dict[str, np.ndarray] = {}
        try:
            tb.open(str(ms_path))
            data_raw = tb.getcol(column)  # (n_corr, n_chan, n_row)
            result["data"] = data_raw.transpose((2, 1, 0))  # → (n_row, n_chan, n_corr)

            uvw_raw = tb.getcol("UVW")  # (3, n_row)
            result["uvw"] = uvw_raw.T  # → (n_row, 3)

            result["time"] = tb.getcol("TIME")
            result["antenna1"] = tb.getcol("ANTENNA1")
            result["antenna2"] = tb.getcol("ANTENNA2")

            flag_raw = tb.getcol("FLAG")  # (n_corr, n_chan, n_row)
            result["flag"] = flag_raw.transpose((2, 1, 0))
        finally:
            tb.close()

        logger.info(
            "Read %d rows from %s:%s",
            result["data"].shape[0],
            ms_path,
            column,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: antenna coordinate arrays for sm.setconfig
    # ------------------------------------------------------------------

    def _antenna_locations(self) -> List[EarthLocation]:
        """Return all antennas as EarthLocation list."""
        locs: List[EarthLocation] = []
        for ant in self.antennas:
            if CYSGP4_AVAILABLE and isinstance(ant, cysgp4.PyObserver):
                locs.append(_pyobserver_to_earth_location(ant))
            elif isinstance(ant, EarthLocation):
                locs.append(ant)
            else:
                raise TypeError(f"Unsupported antenna type: {type(ant).__name__}")
        return locs

    def _antenna_x(self) -> List[float]:
        return [loc.geocentric[0].to(u.m).value for loc in self._antenna_locations()]

    def _antenna_y(self) -> List[float]:
        return [loc.geocentric[1].to(u.m).value for loc in self._antenna_locations()]

    def _antenna_z(self) -> List[float]:
        return [loc.geocentric[2].to(u.m).value for loc in self._antenna_locations()]

    def _antenna_diameters(self) -> List[float]:
        d = self.dish_diameter.to(u.m).value
        return [d] * len(self.antennas)

    def _antenna_names(self) -> List[str]:
        return [f"ANT{i:02d}" for i in range(len(self.antennas))]

    def _reference_position(self) -> Dict[str, Any]:
        """Build a CASA position measure from the first antenna."""
        me = casa_measures()
        loc = self._antenna_locations()[0]
        lon, lat, height = loc.geodetic
        return me.position(
            "WGS84",
            _quantity_to_casa(lon, "deg"),
            _quantity_to_casa(lat, "deg"),
            _quantity_to_casa(height, "m"),
        )
