"""Shared HDF5 inspection and plot recipes for notebook/GUI postprocess flows."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import h5py
import math
from pathlib import Path
import re
import warnings
from typing import Any, Mapping, Sequence

from matplotlib.figure import Figure
import numpy as np

from scepter import nbeam, scenario, visualise
from scepter.skynet import pointgen_S_1586_1


SOURCE_AUTO = "auto"
SOURCE_RAW = "raw"
SOURCE_PREACC = "preaccumulated"
SOURCE_MODES = (SOURCE_AUTO, SOURCE_RAW, SOURCE_PREACC)
ENGINE_MATPLOTLIB = "matplotlib"
ENGINE_PLOTLY = "plotly"
ENGINE_MODES = (ENGINE_MATPLOTLIB, ENGINE_PLOTLY)

_STREAM_SLOT_CHUNK = 256
_MAX_MATERIALIZE_BYTES = 8 * 1024**3
_DEFAULT_INTEGRATION_WINDOW_S = 2000.0
_DEFAULT_CCDF_WINDOWING = "sliding"

# Cache for loaded distribution samples.  Keyed on
# (filename, mtime, recipe_id, integrated, integration_window_s, windowing, bw_key).
# Stores (key, values_db_array).  Single-entry cache: evicted when key changes.
_DISTRIBUTION_CACHE: dict[str, tuple[tuple, np.ndarray]] = {}
_DEFAULT_BW_INBAND_MHZ = 5.0
_DEFAULT_WORST_PERCENT = 2.0
_DEFAULT_DATA_LOSS_THRESHOLD_DBW = -98.107
_DEFAULT_S1586_ELEV_RANGE = (15, 90)
_BANDWIDTH_VIEW_CHANNEL_TOTAL = "channel_total"
_BANDWIDTH_VIEW_REFERENCE = "reference_bandwidth"
@dataclass(frozen=True, slots=True)
class RecipeParameter:
    """One GUI-visible plot parameter definition."""

    name: str
    label: str
    kind: str
    default: Any
    description: str
    choices: tuple[tuple[str, Any], ...] = ()
    hidden: bool = False


@dataclass(frozen=True, slots=True)
class PostprocessRecipe:
    """One GUI-visible postprocess recipe."""

    recipe_id: str
    label: str
    category: str
    description: str
    raw_datasets: tuple[str, ...] = ()
    preacc_paths: tuple[str, ...] = ()
    source_modes: tuple[str, ...] = (SOURCE_AUTO, SOURCE_RAW, SOURCE_PREACC)
    raw_only: bool = False
    parameter_specs: tuple[RecipeParameter, ...] = ()


_BOOL_PARAMETER_CHOICES: tuple[tuple[str, Any], ...] = (
    ("Off", False),
    ("On", True),
)

_INTEGRATION_WINDOW_PARAMETER = RecipeParameter(
    name="integration_window_s",
    label="Integration window [s]",
    kind="float",
    default=_DEFAULT_INTEGRATION_WINDOW_S,
    description=(
        "Sliding integration duration in seconds. Larger windows smooth the time series and "
        "usually increase runtime and memory pressure during raw postprocess."
    ),
)

_RAW_HISTOGRAM_BINS_PARAMETER = RecipeParameter(
    name="raw_hist_bins",
    label="Raw histogram bins",
    kind="int",
    default=4096,
    description=(
        "Histogram bins used when reconstructing raw CCDFs. Higher values can sharpen tails, "
        "but they also cost more time and memory."
    ),
)

_REFERENCE_LEVELS_PARAMETER = RecipeParameter(
    name="reference_levels_db",
    label="Reference levels",
    kind="text",
    default="",
    description=(
        "Optional vertical reference levels for CCDF plots, in the current x-axis units. "
        "Use ';' or new lines to separate values, or label a level with 'Damage=-98.1'."
    ),
)

_SHOW_MARGIN_PARAMETER = RecipeParameter(
    name="show_margin",
    label="Show margin notes",
    kind="choice",
    default=False,
    description=(
        "Annotate the CCDF plot with the margin between the selected percentile and each reference level."
    ),
    choices=_BOOL_PARAMETER_CHOICES,
)

_MARGIN_AT_PARAMETER = RecipeParameter(
    name="margin_at",
    label="Margin percentile",
    kind="choice",
    default="p98",
    description="Percentile used when annotating margins against the vertical reference levels.",
    choices=(("P98", "p98"), ("P95", "p95"), ("Custom", "custom")),
)

_CUSTOM_PERCENTILE_PARAMETER = RecipeParameter(
    name="custom_percentile",
    label="Custom percentile [%]",
    kind="float",
    default=99.0,
    description=(
        "Upper-tail percentile when 'Custom' is selected as the margin percentile. "
        "A value of 99 means the largest-value 1%% tail (equivalent to the 99th percentile). "
        "Must be between 50 and 99.99."
    ),
)

_INTEGRATED_PARAMETER = RecipeParameter(
    name="integrated",
    label="Integrated window",
    kind="choice",
    default=False,
    description=(
        "Use the stored instantaneous series directly or integrate it over the configured "
        "time window before plotting."
    ),
    choices=_BOOL_PARAMETER_CHOICES,
)

_WINDOWING_PARAMETER = RecipeParameter(
    name="windowing",
    label="Integration mode",
    kind="choice",
    default="sliding",
    description=(
        "Window selection strategy for time integration. "
        "Sliding: overlapping windows starting at every eligible sample (maximum statistics, correlated). "
        "Subsequent: back-to-back non-overlapping windows (independent samples, fewer points). "
    ),
    choices=(
        ("Sliding", "sliding"),
        ("Subsequent", "subsequent"),
    ),
)

_GRID_DENSITY_PARAMETER = RecipeParameter(
    name="grid_tick_density",
    label="Grid tick density",
    kind="choice",
    default="normal",
    description="Control the density of major grid lines on the plot axes. Dense adds more ticks; sparse reduces visual clutter.",
    choices=(
        ("Sparse", "sparse"),
        ("Normal", "normal"),
        ("Dense", "dense"),
    ),
)

_DISTRIBUTION_PARAMETER_SPECS: tuple[RecipeParameter, ...] = (
    _INTEGRATED_PARAMETER,
    _INTEGRATION_WINDOW_PARAMETER,
    _WINDOWING_PARAMETER,
    RecipeParameter(
        name="bandwidth_view_mode",
        label="Bandwidth view",
        kind="choice",
        default=_BANDWIDTH_VIEW_CHANNEL_TOTAL,
        description=(
            "RAS receiver band (default): show results integrated over the RAS receiver bandwidth. "
            "Reference bandwidth: rescale to a custom bandwidth for RA.769 comparison."
        ),
        choices=(
            ("RAS receiver band", _BANDWIDTH_VIEW_CHANNEL_TOTAL),
            ("Reference bandwidth", _BANDWIDTH_VIEW_REFERENCE),
        ),
    ),
    RecipeParameter(
        name="reference_bandwidth_mhz",
        label="Reference bandwidth [MHz]",
        kind="float",
        default=1.0,
        description="Reference bandwidth used for RA.769-style spectral display and threshold comparison views.",
    ),
    RecipeParameter(
        name="corridor",
        label="Skycell corridor",
        kind="choice",
        default=False,
        description=(
            "Restrict the CCDF display to the S.1586 skycell corridor. Corridor views require "
            "raw sky-resolved samples."
        ),
        choices=_BOOL_PARAMETER_CHOICES,
    ),
    _RAW_HISTOGRAM_BINS_PARAMETER,
    _REFERENCE_LEVELS_PARAMETER,
    _SHOW_MARGIN_PARAMETER,
    _MARGIN_AT_PARAMETER,
    _CUSTOM_PERCENTILE_PARAMETER,
    RecipeParameter(
        name="normalize_by_max_grx",
        label="Normalize by max G_rx",
        kind="choice",
        default=False,
        description=(
            "Shift all values by the maximum RAS receive gain (S.1586 normalization). "
            "When enabled, the displayed quantity is divided by G_rx,max so plots show "
            "the receiver-gain-independent EPFD/PFD."
        ),
        choices=_BOOL_PARAMETER_CHOICES,
    ),
)

_CCDF_REFERENCE_PARAMETER_SPECS: tuple[RecipeParameter, ...] = (
    _REFERENCE_LEVELS_PARAMETER,
    _SHOW_MARGIN_PARAMETER,
    _MARGIN_AT_PARAMETER,
    _CUSTOM_PERCENTILE_PARAMETER,
)

_HEMISPHERE_PROJECTION_PARAMETER = RecipeParameter(
    name="projection",
    label="Projection",
    kind="choice",
    default="polar",
    description="Map projection for 2D hemisphere plots. Polar uses a circular equal-area projection; rectangular uses azimuth vs elevation axes.",
    choices=(
        ("Polar", "polar"),
        ("Rectangular", "rect"),
    ),
)
_HEMISPHERE_SUMMARY_PARAMETER = RecipeParameter(
    name="show_summary_stats",
    label="Show avg/max summary",
    kind="bool",
    default=False,
    description="Show the simple mean and maximum skycell value in a footer below the hemisphere plot.",
)

_HEMISPHERE_PERCENTILE_PARAMETER_SPECS: tuple[RecipeParameter, ...] = (
    _HEMISPHERE_PROJECTION_PARAMETER,
    _INTEGRATED_PARAMETER,
    _INTEGRATION_WINDOW_PARAMETER,
    RecipeParameter(
        name="bandwidth_view_mode",
        label="Bandwidth view",
        kind="choice",
        default=_BANDWIDTH_VIEW_CHANNEL_TOTAL,
        description="Display hemisphere power maps as stored channel totals or in a reference bandwidth.",
        choices=(
            ("RAS receiver band", _BANDWIDTH_VIEW_CHANNEL_TOTAL),
            ("Reference bandwidth", _BANDWIDTH_VIEW_REFERENCE),
        ),
    ),
    RecipeParameter(
        name="bandwidth_mhz",
        label="Channel bandwidth [MHz] (legacy override)",
        kind="float",
        default=_DEFAULT_BW_INBAND_MHZ,
        description=(
            "Override channel bandwidth for legacy files without stored bandwidth metadata. "
            "Modern runs with RAS receiver band do not use this parameter."
        ),
        hidden=True,
    ),
    RecipeParameter(
        name="reference_bandwidth_mhz",
        label="Reference bandwidth [MHz]",
        kind="float",
        default=1.0,
        description="Reference bandwidth used for the alternate spectral display mode.",
    ),
    RecipeParameter(
        name="worst_percent",
        label="Worst percent",
        kind="float",
        default=_DEFAULT_WORST_PERCENT,
        description=(
            "Upper-tail percentile shown in each skycell. A value of 2 renders the 98th "
            "percentile map."
        ),
    ),
    _HEMISPHERE_SUMMARY_PARAMETER,
)

_HEMISPHERE_DATA_LOSS_PARAMETER_SPECS: tuple[RecipeParameter, ...] = (
    _HEMISPHERE_PROJECTION_PARAMETER,
    _INTEGRATED_PARAMETER,
    _INTEGRATION_WINDOW_PARAMETER,
    RecipeParameter(
        name="bandwidth_view_mode",
        label="Bandwidth view",
        kind="choice",
        default=_BANDWIDTH_VIEW_CHANNEL_TOTAL,
        description="Display data-loss comparisons in channel-total or reference-bandwidth form.",
        choices=(
            ("RAS receiver band", _BANDWIDTH_VIEW_CHANNEL_TOTAL),
            ("Reference bandwidth", _BANDWIDTH_VIEW_REFERENCE),
        ),
    ),
    RecipeParameter(
        name="bandwidth_mhz",
        label="Channel bandwidth [MHz] (legacy override)",
        kind="float",
        default=_DEFAULT_BW_INBAND_MHZ,
        description="Override channel bandwidth for legacy files without stored bandwidth metadata.",
        hidden=True,
    ),
    RecipeParameter(
        name="reference_bandwidth_mhz",
        label="Reference bandwidth [MHz]",
        kind="float",
        default=1.0,
        description="Reference bandwidth used for the alternate spectral display mode.",
    ),
    RecipeParameter(
        name="protection_criterion_db",
        label="Loss threshold [dBW]",
        kind="float",
        default=_DEFAULT_DATA_LOSS_THRESHOLD_DBW,
        description=(
            "Threshold used for the skycell data-loss map. Samples above this value count toward "
            "the local exceedance probability."
        ),
    ),
    _HEMISPHERE_SUMMARY_PARAMETER,
    RecipeParameter(
        name="override_colormap_min",
        label="Override colormap min",
        kind="bool",
        default=False,
        description="Enable a fixed lower bound for the data-loss colormap in percent.",
    ),
    RecipeParameter(
        name="colormap_min_pct",
        label="Colormap min [%]",
        kind="float",
        default=0.0,
        description="Lower bound for the data-loss colormap in percent when the override is enabled.",
    ),
    RecipeParameter(
        name="override_colormap_max",
        label="Override colormap max",
        kind="bool",
        default=False,
        description="Enable a fixed upper bound for the data-loss colormap in percent.",
    ),
    RecipeParameter(
        name="colormap_max_pct",
        label="Colormap max [%]",
        kind="float",
        default=100.0,
        description="Upper bound for the data-loss colormap in percent when the override is enabled.",
    ),
)

_RECIPE_PARAMETER_ALIASES: dict[str, dict[str, str]] = {
    # TEMP(summary-toggle-alias): remove after callers migrate to show_summary_stats.
    "hemisphere_data_loss_map": {
        "show_loss_summary_stats": "show_summary_stats",
    },
    # TEMP(summary-toggle-alias): remove after callers migrate to show_summary_stats.
    "hemisphere_data_loss_map_3d": {
        "show_loss_summary_stats": "show_summary_stats",
    },
}

_PRIMARY_POWER_RAW_DATASETS: tuple[str, ...] = (
    "Prx_total_W",
    "EPFD_W_m2",
    "PFD_total_RAS_STATION_W_m2",
)
_POWER_PREACC_FAMILIES: tuple[str, ...] = (
    "prx_total_distribution",
    "epfd_distribution",
    "total_pfd_ras_distribution",
    "per_satellite_pfd_distribution",
    "prx_elevation_heatmap",
    "per_satellite_pfd_elevation_heatmap",
)

_HEATMAP_PARAMETER_SPECS: tuple[RecipeParameter, ...] = (
    RecipeParameter(
        name="bandwidth_view_mode",
        label="Bandwidth view",
        kind="choice",
        default=_BANDWIDTH_VIEW_CHANNEL_TOTAL,
        description="Display the heatmap in channel-total or reference-bandwidth form.",
        choices=(
            ("RAS receiver band", _BANDWIDTH_VIEW_CHANNEL_TOTAL),
            ("Reference bandwidth", _BANDWIDTH_VIEW_REFERENCE),
        ),
    ),
    RecipeParameter(
        name="bandwidth_mhz",
        label="Channel bandwidth [MHz] (legacy override)",
        kind="float",
        default=_DEFAULT_BW_INBAND_MHZ,
        description="Override channel bandwidth for legacy files without stored bandwidth metadata.",
        hidden=True,
    ),
    RecipeParameter(
        name="reference_bandwidth_mhz",
        label="Reference bandwidth [MHz]",
        kind="float",
        default=1.0,
        description="Reference bandwidth used for the alternate spectral display mode.",
    ),
    RecipeParameter(
        name="elevation_bin_step_deg",
        label="Elevation bin step [deg]",
        kind="float",
        default=1.0,
        description=(
            "Elevation resolution used when rebuilding raw heatmaps. Smaller bins add detail but "
            "increase processing time and sparsity."
        ),
    ),
    RecipeParameter(
        name="value_bin_step_db",
        label="Value bin step [dB]",
        kind="float",
        default=0.25,
        description=(
            "Value-axis resolution used when rebuilding raw heatmaps. Smaller bins improve precision "
            "at the cost of more processing and larger histograms."
        ),
    ),
)

_TIME_SERIES_DISPLAY_PARAMETER_SPECS: tuple[RecipeParameter, ...] = (
    RecipeParameter(
        name="max_points",
        label="Max plotted points",
        kind="int",
        default=4000,
        description=(
            "Display-only visual downsampling cap per series. Larger values keep more detail but "
            "can make long time-series plots look like a solid corridor."
        ),
    ),
    RecipeParameter(
        name="smoothing_window_s",
        label="Smoothing window [s]",
        kind="float",
        default=0.0,
        description=(
            "Display-only rolling-mean window in seconds. Set to 0 to keep the raw per-timestep trace."
        ),
    ),
)

_POWER_TIME_SERIES_PARAMETER_SPECS: tuple[RecipeParameter, ...] = (
    RecipeParameter(
        name="bandwidth_view_mode",
        label="Bandwidth view",
        kind="choice",
        default=_BANDWIDTH_VIEW_CHANNEL_TOTAL,
        description="Display the time series in channel-total or reference-bandwidth form.",
        choices=(
            ("RAS receiver band", _BANDWIDTH_VIEW_CHANNEL_TOTAL),
            ("Reference bandwidth", _BANDWIDTH_VIEW_REFERENCE),
        ),
    ),
    RecipeParameter(
        name="bandwidth_mhz",
        label="Channel bandwidth [MHz] (legacy override)",
        kind="float",
        default=_DEFAULT_BW_INBAND_MHZ,
        description="Override channel bandwidth for legacy files without stored bandwidth metadata.",
        hidden=True,
    ),
    RecipeParameter(
        name="reference_bandwidth_mhz",
        label="Reference bandwidth [MHz]",
        kind="float",
        default=1.0,
        description="Reference bandwidth used for the alternate spectral display mode.",
    ),
    *_TIME_SERIES_DISPLAY_PARAMETER_SPECS,
)

_BEAM_OVERVIEW_PARAMETER_SPECS: tuple[RecipeParameter, ...] = (
    RecipeParameter(
        name="show_total_beams",
        label="Service beams",
        kind="bool",
        default=True,
        description="Show demand-serving beams in the network over time.",
    ),
    RecipeParameter(
        name="show_visible_beams",
        label="RAS-visible beams",
        kind="bool",
        default=True,
        description="Show beams from satellites visible at the RAS station over time.",
    ),
    RecipeParameter(
        name="show_beam_demand",
        label="Beam demand",
        kind="bool",
        default=True,
        description="Show the demanded beam-count time series.",
    ),
    RecipeParameter(
        name="show_demand_minus_service",
        label="Demand minus service",
        kind="bool",
        default=True,
        description=(
            "Show beam demand minus network service beams. Positive values indicate "
            "demand shortfall and negative values indicate spare serving capacity."
        ),
    ),
    *_TIME_SERIES_DISPLAY_PARAMETER_SPECS,
)


RECIPES: tuple[PostprocessRecipe, ...] = (
    PostprocessRecipe(
        "prx_total_distribution",
        "Input power CCDF",
        "Distributions",
        "Input-power CCDF with instantaneous/integrated and corridor controls.",
        raw_datasets=("Prx_total_W",),
        preacc_paths=("prx_total_distribution/counts", "prx_total_distribution/edges_dbw"),
        parameter_specs=_DISTRIBUTION_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "epfd_distribution",
        "EPFD CCDF",
        "Distributions",
        "EPFD CCDF with instantaneous/integrated and corridor controls.",
        raw_datasets=("EPFD_W_m2",),
        preacc_paths=("epfd_distribution/counts", "epfd_distribution/edges_dbw"),
        parameter_specs=_DISTRIBUTION_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "total_pfd_ras_distribution",
        "Total PFD CCDF",
        "Distributions",
        "Total PFD at the RAS station with instantaneous/integrated and corridor controls.",
        raw_datasets=("PFD_total_RAS_STATION_W_m2",),
        preacc_paths=("total_pfd_ras_distribution/counts", "total_pfd_ras_distribution/edges_dbw"),
        parameter_specs=_DISTRIBUTION_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "per_satellite_pfd_distribution",
        "Per-satellite PFD CCDF",
        "Distributions",
        "Distribution of instantaneous per-satellite PFD contributions.",
        raw_datasets=("PFD_per_sat_RAS_STATION_W_m2",),
        preacc_paths=(
            "per_satellite_pfd_distribution/counts",
            "per_satellite_pfd_distribution/edges_dbw",
        ),
        parameter_specs=(
            RecipeParameter(
                name="bandwidth_view_mode",
                label="Bandwidth view",
                kind="choice",
                default=_BANDWIDTH_VIEW_CHANNEL_TOTAL,
                description="Display the distribution in channel-total or reference-bandwidth form.",
                choices=(
                    ("Channel total", _BANDWIDTH_VIEW_CHANNEL_TOTAL),
                    ("Reference bandwidth", _BANDWIDTH_VIEW_REFERENCE),
                ),
            ),
            RecipeParameter(
                name="bandwidth_mhz",
                label="Channel bandwidth [MHz] (legacy override)",
                kind="float",
                default=_DEFAULT_BW_INBAND_MHZ,
                description="Override channel bandwidth for legacy files without stored bandwidth metadata.",
                hidden=True,
            ),
            RecipeParameter(
                name="reference_bandwidth_mhz",
                label="Reference bandwidth [MHz]",
                kind="float",
                default=1.0,
                description="Reference bandwidth used for the alternate spectral display mode.",
            ),
            _RAW_HISTOGRAM_BINS_PARAMETER,
            _REFERENCE_LEVELS_PARAMETER,
            _SHOW_MARGIN_PARAMETER,
            _MARGIN_AT_PARAMETER,
            _CUSTOM_PERCENTILE_PARAMETER,
        ),
    ),
    PostprocessRecipe(
        "prx_elevation_heatmap",
        "Prx vs elevation heatmap",
        "Heatmaps",
        "Instantaneous Prx versus satellite elevation at the RAS station.",
        raw_datasets=("Prx_per_sat_RAS_STATION_W", "sat_elevation_RAS_STATION_deg"),
        preacc_paths=(
            "prx_elevation_heatmap/counts",
            "prx_elevation_heatmap/elevation_edges_deg",
            "prx_elevation_heatmap/value_edges_dbw",
        ),
        parameter_specs=_HEATMAP_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "per_satellite_pfd_elevation_heatmap",
        "Per-satellite PFD vs elevation heatmap",
        "Heatmaps",
        "Instantaneous per-satellite PFD contribution versus satellite elevation.",
        raw_datasets=("PFD_per_sat_RAS_STATION_W_m2", "sat_elevation_RAS_STATION_deg"),
        preacc_paths=(
            "per_satellite_pfd_elevation_heatmap/counts",
            "per_satellite_pfd_elevation_heatmap/elevation_edges_deg",
            "per_satellite_pfd_elevation_heatmap/value_edges_dbw",
        ),
        parameter_specs=_HEATMAP_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "hemisphere_percentile_map",
        "Hemisphere percentile map",
        "Hemisphere maps",
        (
            "2D skycell percentile map for the primary power metric. Toggle Integrated window "
            "to switch between instantaneous and sliding-window rendering."
        ),
        raw_datasets=_PRIMARY_POWER_RAW_DATASETS,
        raw_only=True,
        source_modes=(SOURCE_AUTO, SOURCE_RAW),
        parameter_specs=_HEMISPHERE_PERCENTILE_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "hemisphere_data_loss_map",
        "Hemisphere data-loss map",
        "Hemisphere maps",
        (
            "2D skycell data-loss map for the primary power metric. Toggle Integrated window "
            "to switch between instantaneous and sliding-window rendering."
        ),
        raw_datasets=_PRIMARY_POWER_RAW_DATASETS,
        raw_only=True,
        source_modes=(SOURCE_AUTO, SOURCE_RAW),
        parameter_specs=_HEMISPHERE_DATA_LOSS_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "hemisphere_percentile_map_3d",
        "Hemisphere percentile 3D",
        "Hemisphere maps",
        (
            "3D skycell percentile view for the primary power metric. Toggle Integrated window "
            "to switch between instantaneous and sliding-window rendering."
        ),
        raw_datasets=_PRIMARY_POWER_RAW_DATASETS,
        raw_only=True,
        source_modes=(SOURCE_AUTO, SOURCE_RAW),
        parameter_specs=_HEMISPHERE_PERCENTILE_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "hemisphere_data_loss_map_3d",
        "Hemisphere data-loss 3D",
        "Hemisphere maps",
        (
            "3D skycell data-loss view for the primary power metric. Toggle Integrated window "
            "to switch between instantaneous and sliding-window rendering."
        ),
        raw_datasets=_PRIMARY_POWER_RAW_DATASETS,
        raw_only=True,
        source_modes=(SOURCE_AUTO, SOURCE_RAW),
        parameter_specs=_HEMISPHERE_DATA_LOSS_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "beam_count_full_network_ccdf",
        "Full-network beam-count CCDF",
        "Beam statistics",
        "Satellite beam-usage CCDF across the full network.",
        raw_datasets=("sat_beam_counts_used",),
        preacc_paths=("beam_statistics/full_network_count_hist", "beam_statistics/count_edges"),
        parameter_specs=_CCDF_REFERENCE_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "beam_count_visible_ccdf",
        "RAS-visible beam-count CCDF",
        "Beam statistics",
        "Satellite beam-usage CCDF restricted to satellites visible from the RAS station.",
        raw_datasets=("sat_beam_counts_used", "sat_elevation_RAS_STATION_deg"),
        preacc_paths=("beam_statistics/visible_count_hist", "beam_statistics/count_edges"),
        parameter_specs=_CCDF_REFERENCE_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "beam_overview_over_time",
        "Beam overview over time",
        "Beam statistics",
        "Combined beam-over-time view with Service beams, RAS-visible service beams, and Beam demand controls.",
        parameter_specs=_BEAM_OVERVIEW_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "beam_count_total_over_time",
        "Service beams over time",
        "Beam statistics",
        "Total number of demand-serving beams in the network over time.",
        raw_datasets=("sat_beam_counts_used", "times"),
        preacc_paths=("beam_statistics/network_total_beams_over_time",),
        parameter_specs=_TIME_SERIES_DISPLAY_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "beam_count_visible_over_time",
        "RAS-visible service beams over time",
        "Beam statistics",
        "Total number of demand-serving beams from RAS-visible satellites over time.",
        raw_datasets=("sat_beam_counts_used", "sat_elevation_RAS_STATION_deg", "times"),
        preacc_paths=("beam_statistics/visible_total_beams_over_time",),
        parameter_specs=_TIME_SERIES_DISPLAY_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "beam_demand_over_time",
        "Beam demand over time",
        "Beam statistics",
        "Demanded beam-count time series.",
        raw_datasets=("beam_demand_count", "times"),
        preacc_paths=("beam_statistics/beam_demand_over_time",),
        parameter_specs=_TIME_SERIES_DISPLAY_PARAMETER_SPECS,
    ),
    PostprocessRecipe(
        "beam_cap_sizing_analysis",
        "Beam-cap sizing analysis",
        "Capacity planning",
        "Run count-based and exact reroute beam-cap sizing policies on the stored result file.",
        raw_datasets=("sat_beam_counts_used",),
        raw_only=True,
        source_modes=(SOURCE_AUTO, SOURCE_RAW),
        parameter_specs=(
            RecipeParameter("policy_simpson", "Simpson", "bool", True, "Enable the Simpson pooling policy."),
            RecipeParameter("policy_full_reroute", "Full reroute", "bool", True, "Enable the equal-split full-reroute policy."),
            RecipeParameter("policy_no_reroute", "No reroute", "bool", True, "Enable the no-reroute per-satellite baseline."),
            RecipeParameter("policy_pure_reroute", "Pure reroute", "bool", False, "Enable the exact pure-reroute lower bound when eligibility data are present."),
            RecipeParameter("beam_cap_min", "Beam-cap min", "int", 0, "Minimum beam cap to evaluate."),
            RecipeParameter("beam_cap_max", "Beam-cap max", "int", 260, "Maximum beam cap to evaluate."),
            RecipeParameter("nco_override", "Nco override", "int", 0, "Override Nco for the analysis. Use 0 to inherit the stored file value."),
            RecipeParameter("loss_slot_target", "Loss-slot target", "float", 1e-2, "SLA target for epsilon(B)."),
            RecipeParameter("lost_demand_target", "Lost-demand target", "float", 1e-3, "SLA target for delta(B)."),
            RecipeParameter("per_slot_loss_tolerance", "Tail tolerance", "float", 1e-3, "Tail-risk tolerance used when selecting a cap."),
            RecipeParameter(
                "pure_reroute_backend",
                "Pure-reroute backend",
                "choice",
                "auto",
                "Backend used by the exact pure-reroute solver.",
                choices=(("Auto", "auto"), ("CPU", "cpu"), ("GPU", "gpu")),
            ),
            RecipeParameter("max_demand_slots", "Max demand slots", "int", 0, "Optional early-stop cap for demand slots. Use 0 for the full file."),
        ),
    ),
    PostprocessRecipe(
        "total_pfd_over_time",
        "Total PFD over time",
        "Time series",
        "Raw-only total PFD at the RAS station over time.",
        raw_datasets=("PFD_total_RAS_STATION_W_m2", "times"),
        raw_only=True,
        source_modes=(SOURCE_AUTO, SOURCE_RAW),
        parameter_specs=_POWER_TIME_SERIES_PARAMETER_SPECS,
    ),
)

RECIPE_BY_ID = {recipe.recipe_id: recipe for recipe in RECIPES}


def _normalize_render_engine(engine: str | None) -> str:
    value = str(engine or ENGINE_MATPLOTLIB).strip().lower()
    if value in {"mpl", "matplotlib"}:
        return ENGINE_MATPLOTLIB
    if value == ENGINE_PLOTLY:
        return ENGINE_PLOTLY
    raise ValueError(f"Unsupported render engine {engine!r}.")


def _first_iter_datasets(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    iter_meta = meta.get("iter", {}) or {}
    if not iter_meta:
        return {}
    first_iter = min(int(v) for v in iter_meta.keys())
    row = iter_meta.get(first_iter, {}) or {}
    return row.get("datasets", {}) or {}


def _resolve_iter_dataset_name(meta: Mapping[str, Any], name: str) -> str:
    datasets = _first_iter_datasets(meta)
    if str(name) in datasets:
        return str(name)
    return str(name)


def _iter_dataset_names(meta: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for row in (meta.get("iter", {}) or {}).values():
        for name in (row.get("datasets", {}) or {}).keys():
            names.add(str(name))
    return names


def _preacc_dataset_names(meta: Mapping[str, Any]) -> set[str]:
    return {str(name) for name in ((meta.get("preaccumulated", {}) or {}).keys())}


def _dataset_present(meta: Mapping[str, Any], name: str) -> bool:
    actual_name = _resolve_iter_dataset_name(meta, str(name))
    return actual_name in _first_iter_datasets(meta)


def _sky_axis_present(meta: Mapping[str, Any], name: str) -> bool:
    datasets = _first_iter_datasets(meta)
    ds_meta = datasets.get(_resolve_iter_dataset_name(meta, str(name)))
    if not isinstance(ds_meta, Mapping):
        return False
    shape = tuple(ds_meta.get("shape", ()) or ())
    return len(shape) >= 3 and int(shape[1]) == 1 and int(shape[-1]) > 1


def _dataset_shape(meta: Mapping[str, Any], name: str) -> tuple[int, ...] | None:
    datasets = _first_iter_datasets(meta)
    ds_meta = datasets.get(_resolve_iter_dataset_name(meta, str(name)))
    if not isinstance(ds_meta, Mapping):
        return None
    try:
        return tuple(int(v) for v in (ds_meta.get("shape", ()) or ()))
    except Exception:
        return None


def _beam_cap_count_shape_supported(shape: Sequence[int] | None) -> bool:
    if shape is None:
        return False
    return len(tuple(shape)) in {2, 3, 4}


def _beam_cap_eligibility_shape_supported(shape: Sequence[int] | None) -> bool:
    if shape is None:
        return False
    return len(tuple(shape)) in {3, 4}


def _available_primary_power_datasets(meta: Mapping[str, Any]) -> tuple[str, ...]:
    datasets = _iter_dataset_names(meta)
    return tuple(name for name in _PRIMARY_POWER_RAW_DATASETS if name in datasets)


def _available_preacc_family_names(meta: Mapping[str, Any]) -> tuple[str, ...]:
    preacc_names = _preacc_dataset_names(meta)
    families = {
        str(name).split("/", 1)[0]
        for name in preacc_names
        if "/" in str(name)
    }
    return tuple(sorted(families))


def _configured_preacc_power_families(meta: Mapping[str, Any]) -> tuple[str, ...]:
    attrs = meta.get("attrs", {}) or {}
    configured: list[str] = []
    for family_name in _POWER_PREACC_FAMILIES:
        mode = str(attrs.get(f"output_family_{family_name}_mode", "") or "").strip().lower()
        if mode in {SOURCE_PREACC, "both"}:
            configured.append(str(family_name))
    return tuple(configured)


def _available_preacc_power_families(meta: Mapping[str, Any]) -> tuple[str, ...]:
    available = set(_available_preacc_family_names(meta))
    return tuple(
        family_name for family_name in _POWER_PREACC_FAMILIES if family_name in available
    )


def _missing_configured_preacc_power_families(meta: Mapping[str, Any]) -> tuple[str, ...]:
    available = set(_available_preacc_power_families(meta))
    return tuple(
        family_name
        for family_name in _configured_preacc_power_families(meta)
        if family_name not in available
    )


def _recipe_preacc_families(recipe: PostprocessRecipe) -> tuple[str, ...]:
    families = {
        str(path).split("/", 1)[0]
        for path in recipe.preacc_paths
        if "/" in str(path)
    }
    return tuple(sorted(families))


def _missing_configured_preacc_reason(
    meta: Mapping[str, Any],
    recipe: PostprocessRecipe,
) -> str | None:
    missing = set(_missing_configured_preacc_power_families(meta))
    affected = [family_name for family_name in _recipe_preacc_families(recipe) if family_name in missing]
    if not affected:
        return None
    if len(affected) == 1:
        return f"Configured preaccumulated family {affected[0]!r} is missing from this file."
    joined = ", ".join(repr(name) for name in affected)
    return f"Configured preaccumulated families {joined} are missing from this file."


def _primary_power_dataset(
    meta: Mapping[str, Any],
    preferred: str | None = None,
) -> str | None:
    available = _available_primary_power_datasets(meta)
    if not available:
        return None
    preferred_text = str(str(preferred or "").strip())
    if preferred_text in available:
        return preferred_text
    return str(available[0])


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def default_recipe_parameters(recipe_id: str) -> dict[str, Any]:
    recipe = RECIPE_BY_ID[recipe_id]
    return {spec.name: spec.default for spec in recipe.parameter_specs}


def normalize_recipe_parameters(
    recipe_id: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = default_recipe_parameters(recipe_id)
    if params is None:
        return normalized
    recipe = RECIPE_BY_ID[recipe_id]
    by_name = {spec.name: spec for spec in recipe.parameter_specs}
    aliases = _RECIPE_PARAMETER_ALIASES.get(recipe_id, {})
    canonical_params: dict[str, Any] = {}
    for key, value in params.items():
        key_raw = str(key)
        key_use = aliases.get(key_raw, key_raw)
        if key_raw in aliases and key_use in canonical_params:
            continue
        canonical_params[key_use] = value
    for key_use, value in canonical_params.items():
        if key_use not in by_name:
            continue
        spec = by_name[key_use]
        if spec.kind == "choice":
            if spec.choices:
                allowed_values = {choice_value for _label, choice_value in spec.choices}
                normalized[key_use] = value if value in allowed_values else spec.default
            else:
                normalized[key_use] = value
        elif spec.kind == "float":
            try:
                normalized[key_use] = float(value)
            except Exception:
                normalized[key_use] = float(spec.default)
        elif spec.kind == "int":
            try:
                normalized[key_use] = int(round(float(value)))
            except Exception:
                normalized[key_use] = int(spec.default)
        elif spec.kind == "bool":
            normalized[key_use] = _normalize_bool(value, default=bool(spec.default))
        elif spec.kind == "text":
            normalized[key_use] = "" if value is None else str(value)
        else:
            normalized[key_use] = value
    return normalized


def _parse_reference_levels(raw_value: Any) -> list[dict[str, Any]]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    palette = (
        "#ef4444",
        "#f59e0b",
        "#8b5cf6",
        "#0ea5e9",
        "#22c55e",
        "#ec4899",
    )
    reference_lines: list[dict[str, Any]] = []
    for chunk in re.split(r"[;\r\n]+", text):
        item = str(chunk).strip()
        if not item:
            continue
        label = ""
        value_text = item
        for separator in ("=", ":"):
            if separator in item:
                label_part, value_part = item.split(separator, 1)
                if value_part.strip():
                    label = label_part.strip()
                    value_text = value_part.strip()
                    break
        try:
            value = float(value_text)
        except Exception:
            continue
        display_label = label or f"{value:g}"
        reference_lines.append(
            {
                "value": float(value),
                "label": display_label,
                "color": palette[len(reference_lines) % len(palette)],
            }
        )
    return reference_lines


def _reference_plot_kwargs(
    params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_lines = _parse_reference_levels(params.get("reference_levels_db", ""))
    custom_pct_raw = params.get("custom_percentile")
    custom_pct = float(custom_pct_raw) if custom_pct_raw is not None else None
    plot_kwargs = {
        "show_margin": _normalize_bool(params.get("show_margin"), default=False),
        "margin_at": str(params.get("margin_at", "p98") or "p98"),
        "custom_percentile": custom_pct,
    }
    if reference_lines:
        plot_kwargs["prot_value"] = [float(ref["value"]) for ref in reference_lines]
        plot_kwargs["prot_legend"] = [str(ref["label"]) for ref in reference_lines]
        plot_kwargs["prot_colors"] = [str(ref["color"]) for ref in reference_lines]
    return reference_lines, plot_kwargs


def _distribution_family_name(recipe_id: str) -> str | None:
    return {
        "prx_total_distribution": "prx_total_distribution",
        "epfd_distribution": "epfd_distribution",
        "total_pfd_ras_distribution": "total_pfd_ras_distribution",
    }.get(recipe_id)


def _distribution_raw_dataset_name(recipe_id: str) -> str:
    return {
        "prx_total_distribution": "Prx_total_W",
        "epfd_distribution": "EPFD_W_m2",
        "total_pfd_ras_distribution": "PFD_total_RAS_STATION_W_m2",
    }[recipe_id]


def _distribution_label_from_recipe(recipe_id: str) -> str:
    return {
        "prx_total_distribution": "Input power",
        "epfd_distribution": "EPFD",
        "total_pfd_ras_distribution": "Total PFD at RAS station",
    }[recipe_id]


def _fmt_bw_suffix(bandwidth_label: str) -> str:
    """Return ' over X MHz' or '' for title/axis embedding."""
    return f" {bandwidth_label}" if bandwidth_label else ""


def _fmt_bw_parens(bandwidth_label: str) -> str:
    """Return ' (over X MHz)' or '' for title embedding."""
    return f" ({bandwidth_label})" if bandwidth_label else ""


def _distribution_unit_label(recipe_id: str, *, view_mode: str, bandwidth_label: str) -> str:
    bw = _fmt_bw_suffix(bandwidth_label)
    if recipe_id == "prx_total_distribution":
        return f"Input power [dBW{bw}]"
    if recipe_id == "epfd_distribution":
        return f"EPFD [dBW/m\u00b2{bw}]"
    if recipe_id == "total_pfd_ras_distribution":
        return f"PFD [dBW/m\u00b2{bw}]"
    raise KeyError(recipe_id)


def _hemisphere_summary_metric(primary_recipe_id: str, *, mode: str) -> tuple[str, str]:
    """Return the footer label and unit for a hemisphere summary."""
    if mode == "data_loss":
        return "Data loss", "%"
    if primary_recipe_id == "prx_total_distribution":
        return "Input power", "dBW"
    if primary_recipe_id == "epfd_distribution":
        return "EPFD", "dBW/m^2"
    return "PFD", "dBW/m^2"


def _format_hemisphere_summary_value(value: float, *, unit: str) -> str:
    """Format one footer value using the compact unit style used by the GUI."""
    if unit == "%":
        return f"{value:.2f} %"
    return f"{value:.2f} {unit}"


def _format_hemisphere_summary_text(
    *,
    metric_label: str,
    summary_avg_value: float,
    summary_max_value: float,
    summary_unit: str,
) -> str:
    """Build the centered footer text for hemisphere summaries."""
    avg_text = _format_hemisphere_summary_value(summary_avg_value, unit=summary_unit)
    max_text = _format_hemisphere_summary_value(summary_max_value, unit=summary_unit)
    return f"{metric_label} avg: {avg_text} | max: {max_text}"


def _coerce_attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_preacc_group_attrs(
    filename: str | Path,
    group_name: str,
    *,
    system_index: int | None = None,
    group_prefix: str | None = None,
) -> dict[str, Any]:
    with h5py.File(str(filename), "r") as h5:
        if group_prefix is not None:
            root = h5.get(f"{group_prefix}preaccumulated", None)
        elif system_index is not None:
            sys_root = h5.get(f"system_{int(system_index)}")
            if sys_root is not None:
                root = sys_root.get("preaccumulated", None)
            else:
                root = None
        else:
            root = h5.get("preaccumulated", None)
        if root is None or str(group_name) not in root:
            return {}
        group = root[str(group_name)]
        return {str(k): _coerce_attr_value(v) for k, v in group.attrs.items()}


def _read_root_attrs(filename: str | Path) -> dict[str, Any]:
    with h5py.File(str(filename), "r") as h5:
        return {str(k): _coerce_attr_value(v) for k, v in h5.attrs.items()}


def _preacc_zero_leakage_diagnostic_text(
    filename: str | Path,
    *,
    family_name: str,
    sample_count: int,
) -> str | None:
    if int(sample_count) > 0:
        return None
    with h5py.File(str(filename), "r") as h5:
        preacc_root = h5.get("preaccumulated", None)
        family_attrs: dict[str, Any] = {}
        if preacc_root is not None and str(family_name) in preacc_root:
            family_attrs = {
                str(key): _coerce_attr_value(value)
                for key, value in preacc_root[str(family_name)].attrs.items()
            }
        const_group = h5.get("const", None)
        if const_group is None:
            return None
        enabled_channel_indices_ds = const_group.get("spectrum_enabled_channel_index", None)
        slot_group_indices_ds = const_group.get("spectrum_slot_group_channel_index", None)
        slot_group_leakage_ds = const_group.get("spectrum_slot_group_leakage_factor", None)
        cell_leakage_ds = const_group.get("cell_spectral_leakage_factor_active", None)
        cell_group_leakage_ds = const_group.get("cell_group_spectral_leakage_factor_active", None)
        slot_edges_ds = const_group.get("spectrum_slot_edges_mhz", None)
        if slot_group_indices_ds is None or slot_group_leakage_ds is None:
            return None
        slot_group_indices = np.asarray(slot_group_indices_ds[()], dtype=np.int32)
        enabled_channel_indices = (
            np.asarray(enabled_channel_indices_ds[()], dtype=np.int32).reshape(-1)
            if enabled_channel_indices_ds is not None
            else np.empty((0,), dtype=np.int32)
        )
        slot_group_leakage = np.asarray(slot_group_leakage_ds[()], dtype=np.float64)
        cell_leakage = (
            np.asarray(cell_leakage_ds[()], dtype=np.float64)
            if cell_leakage_ds is not None
            else np.empty((0,), dtype=np.float64)
        ).reshape(-1)
        cell_group_leakage = (
            np.asarray(cell_group_leakage_ds[()], dtype=np.float64)
            if cell_group_leakage_ds is not None
            else np.empty((0,), dtype=np.float64)
        ).reshape(-1)
        slot_edges = (
            np.asarray(slot_edges_ds[()], dtype=np.float64)
            if slot_edges_ds is not None
            else np.empty((0,), dtype=np.float64)
        ).reshape(-1)
        slot_all_zero = slot_group_leakage.size > 0 and not np.any(slot_group_leakage > 0.0)
        cell_all_zero = (
            (cell_leakage.size == 0 or not np.any(cell_leakage > 0.0))
            and (cell_group_leakage.size == 0 or not np.any(cell_group_leakage > 0.0))
        )
        if not slot_all_zero or not cell_all_zero:
            return None
        root_attrs = {str(key): _coerce_attr_value(value) for key, value in h5.attrs.items()}
    if enabled_channel_indices.size > 0:
        valid_channel_indices = sorted(
            {
                int(value)
                for value in enabled_channel_indices.tolist()
                if int(value) >= 0
            }
        )
    else:
        valid_channel_indices = sorted(
            {
                int(value)
                for value in slot_group_indices.reshape(-1).tolist()
                if int(value) >= 0
            }
        )
    cause_bits: list[str] = []
    stored_sample_count = family_attrs.get("sample_count")
    if stored_sample_count is not None:
        cause_bits.append(f"stored sample_count={int(stored_sample_count)}")
    reuse_factor = root_attrs.get("reuse_factor")
    if reuse_factor is not None:
        cause_bits.append(f"reuse factor F{int(reuse_factor)}")
    enabled_channel_count = root_attrs.get("enabled_channel_count")
    if enabled_channel_count is not None:
        cause_bits.append(f"enabled_channel_count={int(enabled_channel_count)}")
    else:
        groups_cap = root_attrs.get("channel_groups_per_cell_cap")
        if groups_cap is not None:
            cause_bits.append(f"channel_groups_per_cell_cap={int(groups_cap)}")
    groups_per_cell = root_attrs.get("channel_groups_per_cell")
    if groups_per_cell is not None:
        cause_bits.append(f"channel_groups_per_cell={int(groups_per_cell)}")
    full_channel_count = max(
        0,
        int(slot_edges.size) - 1,
    )
    if valid_channel_indices:
        if len(valid_channel_indices) == 1:
            occupied_text = f"channel {valid_channel_indices[0] + 1}"
        else:
            occupied_text = (
                f"channels {valid_channel_indices[0] + 1}-{valid_channel_indices[-1] + 1}"
            )
        if full_channel_count > 0:
            cause_bits.append(
                f"enabled channel subset uses {occupied_text} of {full_channel_count}"
            )
        else:
            cause_bits.append(f"enabled channel subset uses {occupied_text}")
        service_stop_mhz = root_attrs.get("service_band_stop_mhz")
        if (
            slot_edges.size > valid_channel_indices[-1] + 1
            and service_stop_mhz is not None
        ):
            highest_stop_mhz = float(slot_edges[valid_channel_indices[-1] + 1])
            service_stop_value = float(service_stop_mhz)
            if (
                np.isfinite(highest_stop_mhz)
                and np.isfinite(service_stop_value)
                and highest_stop_mhz < service_stop_value - 1.0e-9
            ):
                cause_bits.append(
                    "no enabled channel reaches the RAS-adjacent service edge at "
                    f"{service_stop_value:.3f} MHz (highest enabled stop {highest_stop_mhz:.3f} MHz)"
                )
    cause_bits.append("stored slot-group and cell leakage factors are all zero")
    return (
        "Histogram contains no positive samples because the stored simulation recorded zero "
        "spectral leakage into the RAS band; "
        + "; ".join(cause_bits)
        + "."
    )


def _resolve_bandwidth_metadata(
    filename: str | Path,
    *,
    family_name: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    family_attrs = _read_preacc_group_attrs(filename, family_name) if family_name is not None else {}
    root_attrs = _read_root_attrs(filename)
    bandwidth_value = family_attrs.get("bandwidth_mhz")
    bandwidth_source = "family"
    if bandwidth_value is None:
        bandwidth_value = root_attrs.get("bandwidth_mhz")
        bandwidth_source = "root"
    missing_source = False
    try:
        bandwidth_mhz = float(bandwidth_value)
        if not np.isfinite(bandwidth_mhz) or bandwidth_mhz <= 0.0:
            raise ValueError
    except Exception:
        bandwidth_mhz = float(_DEFAULT_BW_INBAND_MHZ)
        bandwidth_source = "missing_default"
        missing_source = True

    stored_basis = family_attrs.get("stored_value_basis")
    if stored_basis is None:
        # Check root attribute first — it's authoritative when present
        root_basis = root_attrs.get("stored_power_basis")
        if root_basis is not None and str(root_basis).strip():
            stored_basis = str(root_basis).strip()
        elif family_name is not None:
            # Keep this fallback list in sync with the writer's
            # ``stored_value_basis`` assignments in scenario.py
            # (~L13520 for distributions, ~L13780 for heatmaps).
            # Mismatch means a missing attr on a legacy / externally-
            # produced HDF5 is interpreted as per_mhz — introduces a
            # 10·log10(bandwidth_mhz) error in every displayed value.
            _FAMILY_CHANNEL_TOTAL = {
                # distributions written as channel_total (or
                # ras_receiver_band when a spectrum plan is active,
                # which uses the same rescale path)
                "prx_total_distribution",
                "epfd_distribution",
                "total_pfd_ras_distribution",
                "per_satellite_pfd_distribution",
                # heatmaps
                "prx_elevation_heatmap",
                "per_satellite_pfd_elevation_heatmap",
            }
            if family_name in _FAMILY_CHANNEL_TOTAL:
                stored_basis = "channel_total"
            else:
                stored_basis = "per_mhz"
        elif dataset_name is not None:
            # Every primary power dataset ends up on the same
            # ``stored_power_basis`` rail — "channel_total" when there
            # is no spectrum plan (or "ras_receiver_band" with one,
            # which uses the same rescale path via
            # ``_resolve_bandwidth_view_context``). Keep this set in
            # sync with the canonical name table at scenario.py:~L98.
            _CHANNEL_TOTAL_DATASETS = {
                "EPFD_W_m2",
                "Prx_total_W",
                "Prx_per_sat_RAS_STATION_W",
                "PFD_total_RAS_STATION_W_m2",
                "PFD_per_sat_RAS_STATION_W_m2",
            }
            if _resolve_iter_dataset_name(scenario.describe_data(str(filename)), dataset_name) in _CHANNEL_TOTAL_DATASETS:
                stored_basis = "channel_total"
            else:
                stored_basis = "per_mhz"
        else:
            stored_basis = "per_mhz"
    stored_basis_name = str(stored_basis).strip().lower()
    if stored_basis_name not in {"per_mhz", "channel_total", "ras_receiver_band"}:
        stored_basis_name = "per_mhz"

    return {
        "bandwidth_mhz": float(bandwidth_mhz),
        "bandwidth_source": str(bandwidth_source),
        "stored_basis": str(stored_basis_name),
        "has_ras_receiver_band": bool(
            root_attrs.get("ras_receiver_band_start_mhz") is not None
            and root_attrs.get("ras_receiver_band_stop_mhz") is not None
        ),
        "_root_attrs": dict(root_attrs),
        "missing_source": bool(missing_source),
        "warning_text": (
            "stored bandwidth missing; defaulting to 5 MHz until overridden"
            if missing_source
            else ""
        ),
    }


def _resolve_bandwidth_view_context(
    filename: str | Path,
    *,
    params: Mapping[str, Any],
    family_name: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    metadata = _resolve_bandwidth_metadata(
        filename,
        family_name=family_name,
        dataset_name=dataset_name,
    )
    view_mode = str(
        params.get("bandwidth_view_mode", _BANDWIDTH_VIEW_CHANNEL_TOTAL)
        or _BANDWIDTH_VIEW_CHANNEL_TOTAL
    ).strip().lower()
    if view_mode not in {_BANDWIDTH_VIEW_CHANNEL_TOTAL, _BANDWIDTH_VIEW_REFERENCE}:
        view_mode = _BANDWIDTH_VIEW_CHANNEL_TOTAL
    channel_bandwidth_mhz = float(params.get("bandwidth_mhz", metadata["bandwidth_mhz"]) or metadata["bandwidth_mhz"])
    if not np.isfinite(channel_bandwidth_mhz) or channel_bandwidth_mhz <= 0.0:
        channel_bandwidth_mhz = float(metadata["bandwidth_mhz"])
    reference_bandwidth_mhz = float(params.get("reference_bandwidth_mhz", 1.0) or 1.0)
    if not np.isfinite(reference_bandwidth_mhz) or reference_bandwidth_mhz <= 0.0:
        reference_bandwidth_mhz = 1.0

    # Determine the effective source bandwidth from the stored basis.
    root_attrs = metadata.get("_root_attrs") or {}
    ras_bw_mhz: float | None = None
    ras_start = root_attrs.get("ras_receiver_band_start_mhz")
    ras_stop = root_attrs.get("ras_receiver_band_stop_mhz")
    if ras_start is not None and ras_stop is not None:
        ras_bw_mhz = max(0.001, float(ras_stop) - float(ras_start))

    stored_basis = str(metadata.get("stored_basis", "channel_total"))
    if stored_basis == "ras_receiver_band" and ras_bw_mhz is not None:
        # Spectrum plan was active: stored values are over RAS receiver band
        source_bandwidth_mhz = ras_bw_mhz
    elif stored_basis == "channel_total":
        source_bandwidth_mhz = float(metadata["bandwidth_mhz"])
    else:
        source_bandwidth_mhz = 1.0

    # Determine target bandwidth and label
    if view_mode == _BANDWIDTH_VIEW_CHANNEL_TOTAL:
        # Default view: no rescaling — show as stored.
        # Omit bandwidth from labels: it's always the native storage basis
        # (RAS receiver band when spectrum plan was active, channel otherwise).
        target_bandwidth_mhz = source_bandwidth_mhz
        bandwidth_label = ""
    else:
        target_bandwidth_mhz = float(reference_bandwidth_mhz)
        if reference_bandwidth_mhz >= 1.0:
            bandwidth_label = f"over {reference_bandwidth_mhz:g} MHz"
        elif reference_bandwidth_mhz >= 0.001:
            bandwidth_label = f"over {reference_bandwidth_mhz * 1000.0:g} kHz"
        else:
            bandwidth_label = f"over {reference_bandwidth_mhz * 1e6:g} Hz"

    _src_bw = float(source_bandwidth_mhz)
    _tgt_bw = float(target_bandwidth_mhz)
    if _src_bw <= 0.0 or not math.isfinite(_src_bw) or _tgt_bw <= 0.0 or not math.isfinite(_tgt_bw):
        raise ValueError(
            "source_bandwidth_mhz and target_bandwidth_mhz must be finite and > 0 "
            f"(got source={_src_bw!r}, target={_tgt_bw!r})."
        )
    offset_db = 10.0 * math.log10(_tgt_bw / _src_bw)
    return {
        **metadata,
        "view_mode": str(view_mode),
        "channel_bandwidth_mhz": float(channel_bandwidth_mhz),
        "reference_bandwidth_mhz": float(reference_bandwidth_mhz),
        "target_bandwidth_mhz": float(target_bandwidth_mhz),
        "db_offset_db": float(offset_db),
        "bandwidth_label": str(bandwidth_label),
    }


def _uniform_bin_step_from_edges(edges: np.ndarray | Any) -> float | None:
    arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if int(arr.size) < 2:
        return None
    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs)]
    if int(diffs.size) < 1:
        return None
    step = float(diffs[0])
    atol = max(abs(step) * 1.0e-9, 1.0e-12)
    if np.allclose(diffs, step, rtol=1.0e-9, atol=atol):
        return step
    return float(np.median(diffs))


def resolve_recipe_parameter_state(
    meta: Mapping[str, Any],
    recipe_id: str,
    *,
    source_preference: str = SOURCE_AUTO,
    params: Mapping[str, Any] | None = None,
    filename: str | Path | None = None,
    primary_power_dataset: str | None = None,
) -> dict[str, Any]:
    """Resolve the effective source and any parameter values locked by that source."""
    params_use = normalize_recipe_parameters(recipe_id, params)
    source_used = resolve_recipe_source(
        meta,
        recipe_id,
        source_preference=source_preference,
        params=params_use,
        filename=filename,
        primary_power_dataset=primary_power_dataset,
    )
    locked_parameters: dict[str, Any] = {}
    if source_used == SOURCE_PREACC and filename is not None:
        distribution_family_name = {
            "prx_total_distribution": "prx_total_distribution",
            "epfd_distribution": "epfd_distribution",
            "total_pfd_ras_distribution": "total_pfd_ras_distribution",
            "per_satellite_pfd_distribution": "per_satellite_pfd_distribution",
        }.get(recipe_id)
        if distribution_family_name is not None:
            try:
                edges = np.asarray(
                    _read_preacc(filename, f"{distribution_family_name}/edges_dbw"),
                    dtype=np.float64,
                ).reshape(-1)
            except Exception:
                edges = np.asarray([], dtype=np.float64)
            if int(edges.size) >= 2:
                locked_parameters["raw_hist_bins"] = int(edges.size - 1)
        family_name = {
            "prx_elevation_heatmap": "prx_elevation_heatmap",
            "per_satellite_pfd_elevation_heatmap": "per_satellite_pfd_elevation_heatmap",
        }.get(recipe_id)
        if family_name is not None:
            attrs = _read_preacc_group_attrs(filename, family_name)
            elev_step = attrs.get("elevation_bin_step_deg")
            value_step = attrs.get("value_bin_step_db")
            if elev_step is None:
                elev_step = _uniform_bin_step_from_edges(
                    _read_preacc(filename, f"{family_name}/elevation_edges_deg")
                )
            if value_step is None:
                value_step = _uniform_bin_step_from_edges(
                    _read_preacc(filename, f"{family_name}/value_edges_dbw")
                )
            try:
                if elev_step is not None and np.isfinite(float(elev_step)):
                    locked_parameters["elevation_bin_step_deg"] = float(elev_step)
            except Exception:
                pass
            try:
                if value_step is not None and np.isfinite(float(value_step)):
                    locked_parameters["value_bin_step_db"] = float(value_step)
            except Exception:
                pass
    return {
        "recipe_id": str(recipe_id),
        "source_used": str(source_used),
        "params": params_use,
        "locked_parameters": locked_parameters,
    }


def _integration_skip_message(
    filename: str | Path,
    meta: Mapping[str, Any],
    *,
    integration_window_s: float,
) -> str | None:
    if not _dataset_present(meta, "times"):
        return "Integrated diagnostics require the stored times dataset, which is not available."
    spans: list[float] = []
    for iteration in sorted(int(v) for v in (meta.get("iter", {}) or {}).keys()):
        times = np.asarray(
            scenario.read_dataset_slice(str(filename), name="times", iteration=int(iteration)),
            dtype=np.float64,
        ).reshape(-1)
        spans.append(0.0 if times.size <= 1 else float((times[-1] - times[0]) * 86400.0))
    if not spans:
        return "Integrated diagnostics require iteration times, which were not found."
    available_span_s = float(min(spans))
    if float(integration_window_s) > available_span_s + 1.0e-9:
        return (
            f"Requested integration window {float(integration_window_s):.1f} s exceeds the "
            f"available per-iteration span {available_span_s:.1f} s."
        )
    for iteration in sorted(int(v) for v in (meta.get("iter", {}) or {}).keys()):
        times = np.asarray(
            scenario.read_dataset_slice(str(filename), name="times", iteration=int(iteration)),
            dtype=np.float64,
        ).reshape(-1)
        if times.size < 2:
            continue
        diffs_sec = np.diff(times) * 86400.0
        if diffs_sec.size < 1:
            continue
        last_hold_sec = float(diffs_sec[-1])
        remaining_sec = (float(times[-1]) - np.asarray(times, dtype=np.float64)) * 86400.0 + last_hold_sec
        if np.any(remaining_sec + 1.0e-9 >= float(integration_window_s)):
            return None
    return (
        f"Requested integration window {float(integration_window_s):.1f} s fits within the stored span, "
        "but the available time sampling cannot form a full integration window."
    )


def recipe_capability(
    meta: Mapping[str, Any],
    recipe_id: str,
    *,
    params: Mapping[str, Any] | None = None,
    filename: str | Path | None = None,
    primary_power_dataset: str | None = None,
) -> dict[str, Any]:
    """Return parameter-aware availability and source diagnostics for one recipe."""
    recipe = RECIPE_BY_ID[recipe_id]
    params_use = normalize_recipe_parameters(recipe_id, params)
    raw_names = _iter_dataset_names(meta)
    preacc_names = _preacc_dataset_names(meta)
    missing_configured_preacc_reason = _missing_configured_preacc_reason(meta, recipe)
    raw_available = all(name in raw_names for name in recipe.raw_datasets)
    preacc_available = all(name in preacc_names for name in recipe.preacc_paths)
    blocked_reason = ""
    requires_raw = bool(recipe.raw_only)

    if recipe_id in {
        "prx_total_distribution",
        "epfd_distribution",
        "total_pfd_ras_distribution",
    }:
        raw_dataset = _distribution_raw_dataset_name(recipe_id)
        raw_available = raw_dataset in raw_names
        integrated = _normalize_bool(params_use.get("integrated"), default=False)
        corridor = _normalize_bool(params_use.get("corridor"), default=False)
        integration_window_s = float(
            params_use.get("integration_window_s", _DEFAULT_INTEGRATION_WINDOW_S)
        )
        preacc_available = (
            (not integrated)
            and (not corridor)
            and all(name in preacc_names for name in recipe.preacc_paths)
        )
        requires_raw = integrated or corridor
        if corridor and not raw_available:
            blocked_reason = f"Raw dataset {raw_dataset!r} is not present."
        elif corridor and not _sky_axis_present(meta, raw_dataset):
            raw_available = False
            blocked_reason = f"{raw_dataset!r} does not expose a skycell axis required for corridor plots."
        elif integrated and not raw_available:
            blocked_reason = f"Raw dataset {raw_dataset!r} is not present."
        elif integrated and "times" not in raw_names:
            raw_available = False
            blocked_reason = "Integrated distributions require the stored times dataset."
        elif integrated and filename is not None:
            skip = _integration_skip_message(
                filename,
                meta,
                integration_window_s=integration_window_s,
            )
            if skip is not None:
                raw_available = False
                blocked_reason = skip
        elif requires_raw and not preacc_available:
            blocked_reason = "This parameter combination is raw-only."
        elif not raw_available and not preacc_available:
            if missing_configured_preacc_reason is not None:
                blocked_reason = (
                    f"{missing_configured_preacc_reason} Raw dataset {raw_dataset!r} is not present."
                )
            else:
                blocked_reason = f"Raw dataset {raw_dataset!r} is not present."
    elif recipe_id == "beam_overview_over_time":
        catalog = _beam_overview_series_catalog()
        selected_keys = [
            key for key in catalog.keys() if _normalize_bool(params_use.get(key), default=True)
        ]
        if not selected_keys:
            raw_available = False
            preacc_available = False
            blocked_reason = "Enable at least one beam-over-time curve."
        else:
            raw_available = True
            preacc_available = True
            for key in selected_keys:
                spec = catalog[key]
                raw_available = raw_available and all(name in raw_names for name in spec["raw_datasets"])
                preacc_available = preacc_available and all(name in preacc_names for name in spec["preacc_paths"])
            if not raw_available and not preacc_available:
                blocked_reason = "Selected beam-over-time curves are not all present in this file."
    elif recipe_id == "beam_cap_sizing_analysis":
        requires_raw = True
        preacc_available = False
        count_shape = _dataset_shape(meta, "sat_beam_counts_used")
        dense_eligibility_shape = _dataset_shape(meta, "sat_eligible_mask")
        count_ready = _dataset_present(meta, "sat_beam_counts_used") and _beam_cap_count_shape_supported(count_shape)
        eligibility_ready = (
            _dataset_present(meta, "sat_eligible_mask")
            and _beam_cap_eligibility_shape_supported(dense_eligibility_shape)
        ) or (
            _dataset_present(meta, "sat_eligible_csr_row_ptr")
            and _dataset_present(meta, "sat_eligible_csr_sat_idx")
        )
        any_enabled = any(
            _normalize_bool(params_use.get(name), default=default)
            for name, default in (
                ("policy_simpson", True),
                ("policy_full_reroute", True),
                ("policy_no_reroute", True),
                ("policy_pure_reroute", False),
            )
        )
        pure_enabled = _normalize_bool(params_use.get("policy_pure_reroute"), default=False)
        raw_available = bool(count_ready and any_enabled)
        if not any_enabled:
            raw_available = False
            blocked_reason = "Enable at least one beam-cap policy."
        elif not _dataset_present(meta, "sat_beam_counts_used"):
            raw_available = False
            blocked_reason = "Beam-cap sizing requires the sat_beam_counts_used dataset."
        elif not count_ready:
            raw_available = False
            blocked_reason = (
                "Beam-cap sizing does not support sat_beam_counts_used with shape "
                f"{count_shape!r}. Supported layouts are (T,S), (T,sky,S), and (T,obs,S,sky)."
            )
        elif pure_enabled and not eligibility_ready:
            raw_available = False
            if _dataset_present(meta, "sat_eligible_mask") and not _beam_cap_eligibility_shape_supported(
                dense_eligibility_shape
            ):
                blocked_reason = (
                    "Pure reroute does not support dense sat_eligible_mask with shape "
                    f"{dense_eligibility_shape!r}. Supported dense layouts are (T,C,S) and (T,sky,C,S)."
                )
            else:
                blocked_reason = (
                    "Pure reroute requires sat_eligible_mask or the CSR eligibility payload in the file."
                )
    elif recipe_id in {
        "hemisphere_percentile_map",
        "hemisphere_data_loss_map",
        "hemisphere_percentile_map_3d",
        "hemisphere_data_loss_map_3d",
    }:
        primary = _primary_power_dataset(meta, preferred=primary_power_dataset)
        requires_raw = True
        preacc_available = False
        integrated = _normalize_bool(params_use.get("integrated"), default=False)
        raw_available = primary is not None and _sky_axis_present(meta, primary)
        integration_window_s = float(
            params_use.get("integration_window_s", _DEFAULT_INTEGRATION_WINDOW_S)
        )
        if primary is None:
            blocked_reason = "No primary power raw dataset is available in this file."
        elif not _sky_axis_present(meta, primary):
            raw_available = False
            blocked_reason = f"{primary!r} does not expose a skycell axis."
        elif integrated and "times" not in raw_names:
            raw_available = False
            blocked_reason = "Integrated hemisphere maps require the stored times dataset."
        elif integrated and filename is not None:
            skip = _integration_skip_message(
                filename,
                meta,
                integration_window_s=integration_window_s,
            )
            if skip is not None:
                raw_available = False
                blocked_reason = skip
    elif recipe.raw_only and not raw_available:
        blocked_reason = "This recipe requires raw datasets that are not present."
    elif not raw_available and not preacc_available:
        blocked_reason = "Required raw and preaccumulated datasets are not present."

    if recipe.raw_only or requires_raw:
        available_sources = (SOURCE_RAW,) if raw_available else ()
    else:
        available_sources = tuple(
            source
            for source, ok in (
                (SOURCE_RAW, raw_available),
                (SOURCE_PREACC, preacc_available),
            )
            if ok
        )
    if (
        blocked_reason == "Required raw and preaccumulated datasets are not present."
        and not available_sources
        and missing_configured_preacc_reason is not None
    ):
        blocked_reason = str(missing_configured_preacc_reason)
    if blocked_reason == "" and not available_sources:
        blocked_reason = "This file does not contain the required datasets for this recipe."

    return {
        "recipe_id": recipe.recipe_id,
        "label": recipe.label,
        "category": recipe.category,
        "description": recipe.description,
        "params": params_use,
        "raw_available": bool(raw_available),
        "preaccumulated_available": bool(preacc_available),
        "available_sources": available_sources,
        "raw_only": bool(recipe.raw_only or requires_raw),
        "blocked_reason": str(blocked_reason),
        "requires_raw": bool(requires_raw),
    }


def recipe_availability(meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return GUI-friendly availability rows for all supported recipes."""
    return [recipe_capability(meta, recipe.recipe_id) for recipe in RECIPES]


def inspect_result_file(
    filename: str | Path,
    *,
    primary_power_dataset: str | None = None,
) -> dict[str, Any]:
    """Inspect one result file and augment it with recipe availability."""
    meta = scenario.describe_data(str(filename))
    selected_primary_power = _primary_power_dataset(meta, preferred=primary_power_dataset)
    bandwidth_metadata = (
        _resolve_bandwidth_metadata(filename, dataset_name=selected_primary_power)
        if selected_primary_power is not None
        else _resolve_bandwidth_metadata(filename)
    )
    configured_preacc_power_families = _configured_preacc_power_families(meta)
    available_preacc_power_families = _available_preacc_power_families(meta)
    missing_configured_preacc_power_families = _missing_configured_preacc_power_families(meta)
    postprocess_warnings: list[str] = []
    if missing_configured_preacc_power_families:
        missing_text = ", ".join(str(name) for name in missing_configured_preacc_power_families)
        postprocess_warnings.append(
            "Configured preaccumulated power families are missing from this file: "
            + missing_text
        )
    return {
        "filename": str(filename),
        "meta": meta,
        "recipes": [
            recipe_capability(
                meta,
                recipe.recipe_id,
                filename=filename,
                primary_power_dataset=selected_primary_power,
            )
            for recipe in RECIPES
        ],
        "iter_ids": sorted(int(v) for v in (meta.get("iter", {}) or {}).keys()),
        "primary_power_dataset": selected_primary_power,
        "available_primary_power_datasets": _available_primary_power_datasets(meta),
        "configured_preacc_power_families": configured_preacc_power_families,
        "available_preacc_power_families": available_preacc_power_families,
        "missing_configured_preacc_power_families": missing_configured_preacc_power_families,
        "postprocess_warnings": tuple(postprocess_warnings),
        "bandwidth_metadata": bandwidth_metadata,
        "bandwidth_warning": str(bandwidth_metadata.get("warning_text") or ""),
    }


def resolve_recipe_source(
    meta: Mapping[str, Any],
    recipe_id: str,
    *,
    source_preference: str = SOURCE_AUTO,
    params: Mapping[str, Any] | None = None,
    filename: str | Path | None = None,
    primary_power_dataset: str | None = None,
) -> str:
    """Resolve the effective data source for a recipe."""
    recipe = RECIPE_BY_ID[recipe_id]
    row = recipe_capability(
        meta,
        recipe_id,
        params=params,
        filename=filename,
        primary_power_dataset=primary_power_dataset,
    )
    source_pref = str(source_preference).strip().lower()
    if source_pref not in SOURCE_MODES:
        raise ValueError(f"Unsupported source preference {source_preference!r}.")
    if row["raw_only"]:
        if not row["raw_available"]:
            raise RuntimeError(row["blocked_reason"] or f"Recipe {recipe.label!r} requires raw data.")
        return SOURCE_RAW
    if source_pref == SOURCE_AUTO:
        if row["preaccumulated_available"]:
            return SOURCE_PREACC
        if row["raw_available"]:
            return SOURCE_RAW
        raise RuntimeError(row["blocked_reason"] or f"Recipe {recipe.label!r} is unavailable.")
    if source_pref == SOURCE_PREACC:
        if not row["preaccumulated_available"]:
            raise RuntimeError(
                row["blocked_reason"]
                or f"Recipe {recipe.label!r} requires preaccumulated data that are not present."
            )
        return SOURCE_PREACC
    if not row["raw_available"]:
        raise RuntimeError(
            row["blocked_reason"] or f"Recipe {recipe.label!r} requires raw data that are not present."
        )
    return SOURCE_RAW


def _read_preacc(
    filename: str | Path,
    name: str,
    *,
    system_index: int | None = None,
    group_prefix: str | None = None,
) -> Any:
    if group_prefix is not None:
        path = f"{group_prefix}preaccumulated/{name}"
    elif system_index is not None:
        path = f"system_{int(system_index)}/preaccumulated/{name}"
    else:
        path = f"preaccumulated/{name}"
    return scenario.read_dataset_slice(str(filename), name=path, iteration=None)


def _iteration_ids(filename: str | Path, *, group_prefix: str = "") -> list[int]:
    meta = scenario.describe_data(str(filename))
    if group_prefix:
        # For per-system iter, scan the HDF5 directly
        import h5py as _h5
        iter_root = f"{group_prefix}iter"
        try:
            with _h5.File(str(filename), "r") as f:
                if iter_root not in f:
                    return []
                ids = []
                for gname in f[iter_root].keys():
                    if gname.startswith("iter_"):
                        try:
                            ids.append(int(gname.split("_")[1]))
                        except Exception:
                            pass
                return sorted(ids)
        except Exception:
            return []
    return sorted(int(v) for v in (meta.get("iter", {}) or {}).keys())


def _read_iter_dataset(filename: str | Path, name: str, iteration: int, *, group_prefix: str = "") -> np.ndarray:
    meta = scenario.describe_data(str(filename))
    actual_name = _resolve_iter_dataset_name(meta, str(name))
    path = f"{group_prefix}iter/iter_{int(iteration):05d}/{actual_name}"
    return np.asarray(
        scenario.read_dataset_slice(str(filename), name=path, iteration=None),
    )


def _read_iteration_time_segments(filename: str | Path, *, group_prefix: str = "") -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    for ii in _iteration_ids(filename, group_prefix=group_prefix):
        if group_prefix:
            path = f"{group_prefix}iter/iter_{int(ii):05d}/times"
            times = np.asarray(
                scenario.read_dataset_slice(str(filename), name=path, iteration=None),
                dtype=np.float64,
            ).reshape(-1)
        else:
            times = np.asarray(
                scenario.read_dataset_slice(str(filename), name="times", iteration=ii),
                dtype=np.float64,
            ).reshape(-1)
        if times.size > 0:
            parts.append(times)
    return parts


def _read_all_iteration_times(filename: str | Path) -> np.ndarray:
    parts = _read_iteration_time_segments(filename)
    if not parts:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _relative_iteration_time_segments(filename: str | Path, *, group_prefix: str = "") -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for times in _read_iteration_time_segments(filename, group_prefix=group_prefix):
        rel = (np.asarray(times, dtype=np.float64) - float(times[0])) * 86400.0
        segments.append(np.asarray(rel, dtype=np.float64))
    return segments


def _stack_across_iterations(filename: str | Path, name: str, *, system_index: int | None = None) -> np.ndarray:
    group_prefix = f"system_{int(system_index)}/" if system_index is not None else ""
    parts: list[np.ndarray] = []
    for ii in _iteration_ids(filename, group_prefix=group_prefix):
        arr = _read_iter_dataset(filename, name, ii, group_prefix=group_prefix)
        parts.append(np.asarray(arr))
    if not parts:
        return np.asarray([])
    return np.concatenate(parts, axis=0)


def _open_stream(filename: str | Path, *names: str, system_index: int | None = None) -> dict[str, Any]:
    meta = scenario.describe_data(str(filename))
    actual_names = [_resolve_iter_dataset_name(meta, name) for name in names]
    group_prefix = f"system_{int(system_index)}/" if system_index is not None else ""
    return scenario.read_data(
        str(filename),
        mode="stream",
        var_selection=list(actual_names),
        slot_chunk_size=_STREAM_SLOT_CHUNK,
        prefetch_chunks=2,
        stack=False,
        group_prefix=group_prefix,
    )


def _resolve_raw_stream_dataset_name(meta: Mapping[str, Any], dataset_name: str) -> str:
    """Return the stored iteration dataset name used for raw streaming reads."""
    return str(_resolve_iter_dataset_name(meta, str(dataset_name)))


def _ds_nbytes(meta: Mapping[str, Any], name: str) -> int | None:
    total = 0
    found = False
    actual_name = _resolve_iter_dataset_name(meta, str(name))
    for row in (meta.get("iter", {}) or {}).values():
        ds = (row.get("datasets", {}) or {}).get(actual_name)
        if not isinstance(ds, Mapping):
            continue
        found = True
        total += int(np.prod(ds.get("shape", ()), dtype=np.int64)) * np.dtype(ds["dtype"]).itemsize
    return total if found else None


def _norm_dist(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim >= 2 and arr.shape[-2] == 1:
        arr = np.squeeze(arr, axis=-2)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, arr.shape[-1])


def _norm_sat(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim == 3:
        return np.moveaxis(arr, 1, -1).reshape(-1, arr.shape[1])
    raise RuntimeError(f"Expected (T,S) or sky-resolved per-satellite samples. Got {arr.shape!r}.")


def _lin2db(v: np.ndarray, *, bw_mhz: float | None = None, nonpositive: str = "nan") -> np.ndarray:
    arr = np.array(v, dtype=np.float64, copy=True)
    arr[(~np.isfinite(arr)) | (arr <= 0.0)] = (
        np.finfo(np.float64).tiny if nonpositive == "tiny" else np.nan
    )
    mask = np.isfinite(arr)
    np.log10(arr, out=arr, where=mask)
    arr *= 10.0
    if bw_mhz is not None:
        arr += 10.0 * np.log10(float(bw_mhz))
    return arr


def _positive_db(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr) & (arr > 0.0)
    if not np.any(mask):
        return np.asarray([], dtype=np.float64)
    return 10.0 * np.log10(arr[mask])


def _apply_grid_tick_density(ax: Any, density: str) -> None:
    """Adjust the number of major ticks on both axes based on density preference."""
    from matplotlib import ticker as mpl_ticker
    y_is_log = ax.get_yscale() == "log"
    if density == "dense":
        ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=20, min_n_ticks=10))
        if not y_is_log:
            ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=20, min_n_ticks=10))
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.08, linestyle=":")
    elif density == "sparse":
        ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=5))
        if not y_is_log:
            ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=5))
    # "normal" leaves the default locators unchanged


def _plot_distribution_histogram(
    counts: Any,
    edges: Any,
    *,
    title: str,
    xlabel: str,
    color: str = "#16a34a",
    reference_lines: Sequence[dict[str, Any]] | None = None,
    show_margin: bool = False,
    margin_at: str = "p98",
    custom_percentile: float | None = None,
) -> tuple[Figure, dict[str, Any]]:
    prot_values = [float(ref["value"]) for ref in (reference_lines or [])]
    prot_labels = [str(ref.get("label") or ref["value"]) for ref in (reference_lines or [])]
    prot_colors = [str(ref.get("color") or "#ef4444") for ref in (reference_lines or [])]
    fig, info = visualise.plot_cdf_ccdf_from_histogram(
        counts,
        edges=edges,
        plot_type="ccdf",
        show_two_percent=False,
        y_log=True,
        ccdf_ymin_pct=0.000001,
        title=title,
        x_label=xlabel,
        legend_outside=False,
        prot_value=prot_values or None,
        prot_legend=prot_labels or None,
        prot_colors=prot_colors or None,
        show_margin=bool(show_margin),
        margin_at=str(margin_at),
        figsize=(9.2, 6.0),
        ccdf_color=color,
        return_values=True,
        show=False,
    )
    info = dict(info or {})
    if fig.axes and fig.axes[0].lines:
        info = _set_display_percentiles_from_ccdf_curve(
            info,
            fig.axes[0].lines[0].get_xdata(),
            fig.axes[0].lines[0].get_ydata(),
            custom_percentile=custom_percentile,
        )
    counts_arr = np.asarray(counts, dtype=np.int64).reshape(-1)
    edges_arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if info.get("p95") is None:
        info["p95"] = _histogram_percentile(edges_arr, counts_arr, 95.0)
    if info.get("p98") is None:
        info["p98"] = _histogram_percentile(edges_arr, counts_arr, 98.0)
    if custom_percentile is not None:
        cp_key = f"p{float(custom_percentile):g}"
        if info.get(cp_key) is None:
            info[cp_key] = _histogram_percentile(edges_arr, counts_arr, float(custom_percentile))
    _apply_matplotlib_ccdf_percentile_guide(
        fig,
        info,
        margin_at=margin_at,
        color=color,
        reference_lines=reference_lines,
        show_margin=bool(show_margin),
        custom_percentile=custom_percentile,
    )
    return fig, info


def _plot_distribution_raw(
    values_db: np.ndarray,
    *,
    title: str,
    xlabel: str,
    color: str = "#16a34a",
    corridor: bool = False,
    hist_bins: int = 4096,
    reference_lines: Sequence[dict[str, Any]] | None = None,
    show_margin: bool = False,
    margin_at: str = "p98",
    custom_percentile: float | None = None,
) -> tuple[Figure, dict[str, Any]]:
    values_arr = np.asarray(values_db, dtype=np.float64)
    if values_arr.size < 1:
        raise RuntimeError("No positive finite samples were available for this recipe.")
    if values_arr.ndim == 0:
        values_plot = values_arr.reshape(1, 1)
    elif values_arr.ndim == 1:
        # Raw per-satellite distributions can legitimately collapse to a single
        # 1-D stream. The visualiser still expects a sample-by-cell matrix.
        values_plot = values_arr[:, None]
    else:
        values_plot = values_arr
    prot_values = [float(ref["value"]) for ref in (reference_lines or [])]
    prot_labels = [str(ref.get("label") or ref["value"]) for ref in (reference_lines or [])]
    prot_colors = [str(ref.get("color") or "#ef4444") for ref in (reference_lines or [])]
    fig, info = visualise.plot_cdf_ccdf(
        values_plot,
        plot_type="ccdf",
        cell_axis=-1,
        show_two_percent=False,
        y_log=True,
        ccdf_ymin_pct=0.000001,
        title=title,
        x_label=xlabel,
        ecdf_method="hist",
        hist_bins=max(32, int(hist_bins)),
        assume_finite=True,
        return_values=True,
        show=False,
        show_skycell_corridor=bool(corridor),
        show_skycell_p98_note=bool(corridor),
        legend_outside=bool(corridor),
        prot_value=prot_values or None,
        prot_legend=prot_labels or None,
        prot_colors=prot_colors or None,
        show_margin=bool(show_margin),
        margin_at=str(margin_at),
        figsize=(9.2, 6.0),
        ccdf_color=color,
    )
    info = dict(info or {})
    if fig.axes and fig.axes[0].lines:
        info = _set_display_percentiles_from_ccdf_curve(
            info,
            fig.axes[0].lines[0].get_xdata(),
            fig.axes[0].lines[0].get_ydata(),
            custom_percentile=custom_percentile,
        )
    finite_values = values_arr.reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size > 0:
        if info.get("p95") is None:
            info["p95"] = float(np.percentile(finite_values, 95.0))
        if info.get("p98") is None:
            info["p98"] = float(np.percentile(finite_values, 98.0))
        if custom_percentile is not None:
            cp_key = f"p{float(custom_percentile):g}"
            if info.get(cp_key) is None:
                info[cp_key] = float(np.percentile(finite_values, float(custom_percentile)))
    _apply_matplotlib_ccdf_percentile_guide(
        fig,
        info,
        margin_at=margin_at,
        color=color,
        reference_lines=reference_lines,
        show_margin=bool(show_margin),
        custom_percentile=custom_percentile,
    )
    return fig, info


def _plot_generic_heatmap(
    histogram: np.ndarray,
    *,
    elevation_edges_deg: np.ndarray,
    value_edges_db: np.ndarray,
    title: str,
    ylabel: str,
    colorbar_label: str = "Sample count",
) -> tuple[Figure, dict[str, Any]]:
    hist = np.asarray(histogram, dtype=np.float64)
    positive = hist > 0
    fig, ax = visualise._new_mpl_subplots(figsize=(10.0, 6.8))
    fig.set_facecolor("white")
    ax.set_facecolor("#fbfdff")
    vmin = float(np.min(hist[positive])) if np.any(positive) else 1.0
    vmax = float(np.max(hist)) if np.any(positive) else 1.0
    mesh = ax.pcolormesh(
        np.asarray(elevation_edges_deg, dtype=np.float64),
        np.asarray(value_edges_db, dtype=np.float64),
        hist.T,
        shading="auto",
        cmap="viridis",
        norm=visualise.LogNorm(vmin=vmin, vmax=max(vmin, vmax)),
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.035, fraction=0.05)
    cbar.set_label(colorbar_label)
    ax.set_title(title, pad=16, fontweight="semibold")
    ax.set_xlabel("Satellite elevation at RAS station [deg]")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#78859a", alpha=0.45, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155")
    fig.tight_layout(pad=1.2)
    return fig, {
        "sample_count": int(np.sum(hist, dtype=np.int64)),
        "positive_bin_count": int(np.count_nonzero(positive)),
        "max_bin_count": int(vmax),
    }


def _prepare_raw_value_heatmap(
    values: np.ndarray,
    elevations: np.ndarray,
    *,
    elevation_bin_step_deg: float = 1.0,
    value_bin_step_db: float = 0.25,
    db_offset_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elev = np.asarray(elevations, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(elev) & np.isfinite(vals) & (vals > 0.0)
    if not np.any(mask):
        raise RuntimeError("No positive finite samples were available for this heatmap.")
    elev_f = elev[mask]
    vals_db = 10.0 * np.log10(vals[mask]) + float(db_offset_db)
    elev_edges = np.arange(0.0, 90.0 + float(elevation_bin_step_deg), float(elevation_bin_step_deg))
    v_min = float(np.floor(np.min(vals_db) / value_bin_step_db) * value_bin_step_db)
    v_max = float(np.ceil(np.max(vals_db) / value_bin_step_db) * value_bin_step_db)
    value_edges = np.arange(v_min, v_max + float(value_bin_step_db), float(value_bin_step_db))
    hist, _, _ = np.histogram2d(elev_f, vals_db, bins=(elev_edges, value_edges))
    return hist.astype(np.int64, copy=False), elev_edges, value_edges


def _beam_hist_from_visible_samples(
    counts: np.ndarray,
    visible_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = _beam_counts_samples(counts)
    mask = np.asarray(visible_mask, dtype=bool)
    if mask.shape != samples.shape:
        raise RuntimeError("Visible-satellite mask shape is incompatible with beam counts.")
    visible_samples = np.asarray(samples[mask], dtype=np.int64).reshape(-1)
    if visible_samples.size <= 0:
        return np.zeros(1, dtype=np.int64), np.asarray([0, 1], dtype=np.int64)
    max_count = int(np.max(visible_samples))
    edges = np.arange(0, max_count + 2, dtype=np.int64)
    hist = np.bincount(visible_samples, minlength=max_count + 1).astype(np.int64, copy=False)
    return hist, edges


def _beam_counts_samples(
    counts: np.ndarray,
    *,
    visible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Conservative per-satellite sample view used for histogram / CCDF style
    beam-load plots.

    This helper must not be used to build network-total beam-over-time curves
    for sky-resolved boresight outputs.
    """
    arr = np.asarray(counts)
    if arr.ndim == 2:
        samples = arr
    elif arr.ndim == 3:
        # Conservative per-satellite collapse for (T, sky, S)
        samples = np.max(arr, axis=1)
    elif arr.ndim == 4:
        # Conservative per-satellite collapse for stored boresight shape
        # (T, obs, S, sky). The GUI/raw writer uses obs=1 here.
        samples = np.max(arr, axis=(1, 3))
    else:
        raise RuntimeError(f"Unsupported sat_beam_counts_used shape {arr.shape!r}.")
    samples = np.asarray(samples, dtype=np.int64)
    if visible_mask is not None:
        mask = np.asarray(visible_mask, dtype=bool)
        if mask.ndim == 2 and samples.ndim == 2 and mask.shape == samples.shape:
            samples = np.where(mask, samples, 0)
        else:
            raise RuntimeError("Visible-satellite mask shape is incompatible with beam counts.")
    return samples


def _beam_total_over_time_from_counts(
    counts: np.ndarray,
    *,
    visible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build one physically consistent total-beam time series.

    For sky-resolved boresight counts, sum satellites first for each sky row and
    only then collapse sky with max-over-sky. This avoids the invalid
    max-per-satellite stitch that can exceed beam demand.
    """
    arr = np.asarray(counts)

    if arr.ndim == 2:
        work = np.asarray(arr, dtype=np.float64)
        if visible_mask is not None:
            mask = np.asarray(visible_mask, dtype=bool)
            if mask.shape != work.shape:
                raise RuntimeError("Visible-satellite mask shape is incompatible with beam counts.")
            work = np.where(mask, work, 0.0)
        return np.sum(work, axis=1, dtype=np.float64).reshape(-1)

    if arr.ndim == 3:
        # (T, sky, S)
        work = np.asarray(arr, dtype=np.float64)
        if visible_mask is not None:
            mask = np.asarray(visible_mask, dtype=bool)
            expected = (int(work.shape[0]), int(work.shape[2]))
            if tuple(mask.shape) != expected:
                raise RuntimeError("Visible-satellite mask shape is incompatible with beam counts.")
            work = np.where(mask[:, None, :], work, 0.0)
        sky_totals = np.sum(work, axis=2, dtype=np.float64)
        return np.max(sky_totals, axis=1).reshape(-1)

    if arr.ndim == 4:
        # Stored boresight raw shape: (T, obs, S, sky)
        work = np.asarray(arr, dtype=np.float64)
        work = work[:, 0, :, :] if int(work.shape[1]) == 1 else np.max(work, axis=1)
        if visible_mask is not None:
            mask = np.asarray(visible_mask, dtype=bool)
            expected = (int(work.shape[0]), int(work.shape[1]))
            if tuple(mask.shape) != expected:
                raise RuntimeError("Visible-satellite mask shape is incompatible with beam counts.")
            work = np.where(mask[:, :, None], work, 0.0)
        sky_totals = np.sum(work, axis=1, dtype=np.float64)
        return np.max(sky_totals, axis=1).reshape(-1)

    raise RuntimeError(f"Unsupported sat_beam_counts_used shape {arr.shape!r}.")


def _beam_hist_from_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=np.int64)
    if arr.size == 0:
        raise RuntimeError("No beam-count samples were available.")
    max_count = int(np.max(arr))
    edges = np.arange(0, max_count + 2, dtype=np.int64)
    hist = np.bincount(arr.reshape(-1), minlength=max_count + 1).astype(np.int64, copy=False)
    return hist, edges


def _beam_overview_series_catalog() -> dict[str, dict[str, Any]]:
    return {
        "show_total_beams": {
            "recipe_id": "beam_count_total_over_time",
            "label": "Service beams",
            "color": "#0f766e",
            "raw_datasets": ("sat_beam_counts_used", "times"),
            "preacc_paths": ("beam_statistics/network_total_beams_over_time",),
        },
        "show_visible_beams": {
            "recipe_id": "beam_count_visible_over_time",
            "label": "RAS-visible service beams",
            "color": "#2563eb",
            "raw_datasets": ("sat_beam_counts_used", "sat_elevation_RAS_STATION_deg", "times"),
            "preacc_paths": ("beam_statistics/visible_total_beams_over_time",),
        },
        "show_beam_demand": {
            "recipe_id": "beam_demand_over_time",
            "label": "Beam demand",
            "color": "#dc2626",
            "raw_datasets": ("beam_demand_count", "times"),
            "preacc_paths": ("beam_statistics/beam_demand_over_time",),
        },
        "show_demand_minus_service": {
            "recipe_id": "beam_demand_minus_service_over_time",
            "label": "Demand minus service",
            "color": "#7c3aed",
            "raw_datasets": ("beam_demand_count", "sat_beam_counts_used", "times"),
            "preacc_paths": (
                "beam_statistics/beam_demand_over_time",
                "beam_statistics/network_total_beams_over_time",
            ),
        },
    }


def _normalize_preacc_series(values: np.ndarray | Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _series_segments_from_preacc(
    filename: str | Path,
    values: np.ndarray | Any,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    time_segments = _relative_iteration_time_segments(filename)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return [np.asarray([0.0], dtype=np.float64)], [arr.reshape(1)]
    if not time_segments:
        flat = arr.reshape(-1)
        return [np.arange(flat.size, dtype=np.float64)], [flat]
    if arr.ndim >= 2 and int(arr.shape[0]) == len(time_segments):
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for idx, time_seg in enumerate(time_segments):
            y_seg = np.asarray(arr[idx], dtype=np.float64).reshape(-1)
            seg_len = min(int(time_seg.size), int(y_seg.size))
            if seg_len <= 0:
                continue
            x_parts.append(np.asarray(time_seg[:seg_len], dtype=np.float64))
            y_parts.append(np.asarray(y_seg[:seg_len], dtype=np.float64))
        if x_parts:
            return x_parts, y_parts
    flat = arr.reshape(-1)
    time_total = int(sum(int(seg.size) for seg in time_segments))
    if time_total <= 0:
        return [np.arange(flat.size, dtype=np.float64)], [flat]
    if flat.size < time_total:
        time_segments = [seg for seg in time_segments if seg.size > 0]
        merged = np.concatenate(time_segments, axis=0) if time_segments else np.asarray([], dtype=np.float64)
        seg_len = min(int(flat.size), int(merged.size))
        return [merged[:seg_len]], [flat[:seg_len]]
    x_parts = []
    y_parts = []
    offset = 0
    for time_seg in time_segments:
        seg_len = int(time_seg.size)
        if seg_len <= 0:
            continue
        x_parts.append(np.asarray(time_seg, dtype=np.float64))
        y_parts.append(np.asarray(flat[offset : offset + seg_len], dtype=np.float64))
        offset += seg_len
    if offset < int(flat.size):
        x_parts.append(np.arange(int(flat.size - offset), dtype=np.float64))
        y_parts.append(np.asarray(flat[offset:], dtype=np.float64))
    return x_parts, y_parts


def _join_series_segments(
    x_segments: list[np.ndarray],
    y_segments: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not x_segments or not y_segments:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for idx, (x_seg, y_seg) in enumerate(zip(x_segments, y_segments)):
        x_arr = np.asarray(x_seg, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y_seg, dtype=np.float64).reshape(-1)
        seg_len = min(int(x_arr.size), int(y_arr.size))
        if seg_len <= 0:
            continue
        if idx > 0:
            x_parts.append(np.asarray([np.nan], dtype=np.float64))
            y_parts.append(np.asarray([np.nan], dtype=np.float64))
        x_parts.append(x_arr[:seg_len])
        y_parts.append(y_arr[:seg_len])
    if not x_parts:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def _subtract_series_segments(
    minuend_segments: tuple[list[np.ndarray], list[np.ndarray]],
    subtrahend_segments: tuple[list[np.ndarray], list[np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Subtract one segmented beam-over-time series from another."""
    x_left, y_left = minuend_segments
    x_right, y_right = subtrahend_segments
    if len(x_left) != len(x_right) or len(y_left) != len(y_right):
        raise RuntimeError("Beam overview source segments are incompatible for subtraction.")

    x_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []
    for x_l, y_l, x_r, y_r in zip(x_left, y_left, x_right, y_right):
        x_l_arr = np.asarray(x_l, dtype=np.float64).reshape(-1)
        y_l_arr = np.asarray(y_l, dtype=np.float64).reshape(-1)
        x_r_arr = np.asarray(x_r, dtype=np.float64).reshape(-1)
        y_r_arr = np.asarray(y_r, dtype=np.float64).reshape(-1)
        seg_len = min(int(x_l_arr.size), int(y_l_arr.size), int(x_r_arr.size), int(y_r_arr.size))
        if seg_len <= 0:
            continue
        if not np.allclose(x_l_arr[:seg_len], x_r_arr[:seg_len], equal_nan=True):
            raise RuntimeError("Beam overview time axes are incompatible for subtraction.")
        x_out.append(np.asarray(x_l_arr[:seg_len], dtype=np.float64))
        y_out.append(
            np.asarray(y_l_arr[:seg_len], dtype=np.float64)
            - np.asarray(y_r_arr[:seg_len], dtype=np.float64)
        )
    return x_out, y_out


def _smooth_series_segment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if float(window_s) <= 0.0:
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if int(x_arr.size) < 3 or int(y_arr.size) < 3:
        return x_arr, y_arr
    diffs = np.diff(x_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if int(diffs.size) < 1:
        return x_arr, y_arr
    step_s = float(np.median(diffs))
    if step_s <= 0.0:
        return x_arr, y_arr
    kernel_size = int(max(1, round(float(window_s) / step_s)))
    if kernel_size <= 1:
        return x_arr, y_arr
    kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
    valid_mask = np.isfinite(y_arr)
    filled = np.where(valid_mask, y_arr, 0.0)
    weights = np.where(valid_mask, 1.0, 0.0)
    smooth_num = np.convolve(filled, kernel, mode="same")
    smooth_den = np.convolve(weights, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(smooth_den > 0.0, smooth_num / smooth_den, np.nan)
    return x_arr, smooth


def _downsample_series_segment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    max_points_i = int(max(0, max_points))
    if max_points_i <= 0 or int(x_arr.size) <= max_points_i or int(y_arr.size) <= max_points_i:
        return x_arr, y_arr
    sample_idx = np.linspace(0, int(min(x_arr.size, y_arr.size)) - 1, num=max_points_i, dtype=np.int64)
    sample_idx = np.unique(sample_idx)
    return x_arr[sample_idx], y_arr[sample_idx]


def _prepare_display_series_segments(
    x_segments: Sequence[np.ndarray],
    y_segments: Sequence[np.ndarray],
    *,
    max_points: int = 4000,
    smoothing_window_s: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for x_seg, y_seg in zip(x_segments, y_segments):
        x_use, y_use = _smooth_series_segment(
            x_seg,
            y_seg,
            window_s=float(smoothing_window_s),
        )
        x_use, y_use = _downsample_series_segment(
            x_use,
            y_use,
            max_points=int(max_points),
        )
        if int(x_use.size) <= 0 or int(y_use.size) <= 0:
            continue
        x_parts.append(x_use)
        y_parts.append(y_use)
    return x_parts, y_parts


def _plot_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
    color: str,
) -> tuple[Figure, dict[str, Any]]:
    fig, ax = visualise._new_mpl_subplots(figsize=(9.2, 5.6))
    fig.set_facecolor("white")
    ax.set_facecolor("#fbfdff")
    ax.plot(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), color=color, linewidth=1.8)
    ax.set_title(title, pad=14, fontweight="semibold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#9ca3af", alpha=0.4, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12)
    y_arr = np.asarray(y, dtype=np.float64)
    return fig, {"sample_count": int(np.count_nonzero(np.isfinite(y_arr)))}


def _plot_series_collection(
    series: Sequence[dict[str, Any]],
    *,
    title: str,
    ylabel: str,
) -> tuple[Figure, dict[str, Any]]:
    fig, ax = visualise._new_mpl_subplots(figsize=(9.4, 5.8))
    fig.set_facecolor("white")
    ax.set_facecolor("#fbfdff")
    sample_counts: dict[str, int] = {}
    for row in series:
        x = np.asarray(row["x"], dtype=np.float64)
        y = np.asarray(row["y"], dtype=np.float64)
        ax.plot(
            x,
            y,
            color=str(row["color"]),
            linewidth=1.6,
            label=str(row["label"]),
        )
        sample_counts[str(row["label"])] = int(np.count_nonzero(np.isfinite(y)))
    fig.suptitle(title, y=0.975, fontweight="semibold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#9ca3af", alpha=0.4, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_cols = max(1, min(3, int(len(series))))
    handles, labels = ax.get_legend_handles_labels()
    legend_rows = int(math.ceil(max(1, int(len(series))) / float(legend_cols)))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=legend_cols,
        frameon=True,
        borderaxespad=0.0,
    )
    top = 0.82 - max(0, legend_rows - 1) * 0.06
    fig.subplots_adjust(left=0.12, right=0.98, top=max(0.68, top), bottom=0.12)
    return fig, {"sample_counts": sample_counts}


def _edges_midpoints(edges: np.ndarray | Any) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if int(arr.size) < 2:
        return np.asarray([], dtype=np.float64)
    return 0.5 * (arr[:-1] + arr[1:])


def _histogram_percentile(edges: np.ndarray | Any, counts: np.ndarray | Any, percentile: float) -> float | None:
    edges_arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    counts_arr = np.asarray(counts, dtype=np.int64).reshape(-1)
    total = int(np.sum(counts_arr, dtype=np.int64))
    if int(edges_arr.size) != int(counts_arr.size) + 1 or total <= 0:
        return None
    target = max(0.0, min(100.0, float(percentile))) / 100.0 * float(total)
    cumulative = np.cumsum(counts_arr, dtype=np.int64)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = max(0, min(idx, int(counts_arr.size) - 1))
    bin_count = int(counts_arr[idx])
    if bin_count <= 0:
        return float(edges_arr[idx + 1])
    lower = 0.0 if idx == 0 else float(cumulative[idx - 1])
    frac = (float(target) - lower) / float(bin_count)
    frac = max(0.0, min(1.0, frac))
    return float(edges_arr[idx] + frac * (edges_arr[idx + 1] - edges_arr[idx]))


def _margin_reference_percentile_key(
    margin_at: str,
    custom_percentile: float | None = None,
) -> str:
    key = str(margin_at or "p98").strip().lower()
    if key == "p95":
        return "p95"
    if key == "custom" and custom_percentile is not None:
        return f"p{float(custom_percentile):g}"
    return "p98"


def _margin_reference_percentile_probability(
    margin_at: str,
    custom_percentile: float | None = None,
) -> float:
    """Return the upper-tail probability for the selected margin percentile."""
    key = str(margin_at or "p98").strip().lower()
    if key == "p95":
        return 0.05
    if key == "custom" and custom_percentile is not None:
        return max(0.0001, min(0.5, (100.0 - float(custom_percentile)) / 100.0))
    return 0.02


def _margin_reference_percentile_info(
    info: Mapping[str, Any],
    *,
    margin_at: str,
    custom_percentile: float | None = None,
) -> tuple[str, float | None, float]:
    key = _margin_reference_percentile_key(margin_at, custom_percentile)
    probability = _margin_reference_percentile_probability(margin_at, custom_percentile)
    value_raw = info.get(f"{key}_plot", info.get(key))
    value = None if value_raw is None else float(value_raw)
    return key, value, probability


def _ccdf_display_percentile_point(
    x_values: np.ndarray | Any,
    y_values: np.ndarray | Any,
    *,
    target_probability: float,
    axis_top: float = 1.0,
) -> dict[str, float | bool] | None:
    """Return the upper-tail percentile point as it should appear on the plotted CCDF.

    The GUI uses upper-tail semantics: p95 means the largest-value 5% tail and p98 means
    the largest-value 2% tail. We therefore choose the first displayed x position whose
    staircase reaches or drops below the requested tail probability. If the drawn staircase
    never reaches that probability, the percentile lives on the terminal vertical closure
    at the last x value.
    """

    x_arr = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if int(x_arr.size) == 0 or int(x_arr.size) != int(y_arr.size):
        return None
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(finite):
        return None
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if int(x_arr.size) == 0:
        return None
    target = float(target_probability)
    tol = max(1.0e-12, 1.0e-9 * max(abs(target), abs(float(axis_top)), 1.0))
    crossing = np.flatnonzero(y_arr <= target + tol)
    if int(crossing.size) > 0:
        idx = int(crossing[0])
        x_value = float(x_arr[idx] if idx == 0 else x_arr[idx - 1])
        y_top = float(axis_top if idx == 0 else y_arr[idx - 1])
        y_bottom = float(y_arr[idx])
        return {
            "x": x_value,
            "y": float(target),
            "segment_top": float(max(y_top, y_bottom)),
            "segment_bottom": float(min(y_top, y_bottom)),
            "needs_terminal_closure": False,
        }
    y_last = float(y_arr[-1])
    return {
        "x": float(x_arr[-1]),
        "y": float(target),
        "segment_top": float(max(y_last, target)),
        "segment_bottom": float(min(y_last, target)),
        "needs_terminal_closure": bool(target < y_last - tol),
    }


def _set_display_percentiles_from_ccdf_curve(
    info: Mapping[str, Any] | None,
    x_values: np.ndarray | Any,
    y_values: np.ndarray | Any,
    *,
    percent_scale: bool = False,
    custom_percentile: float | None = None,
) -> dict[str, Any]:
    info_use = dict(info or {})
    axis_top = 100.0 if percent_scale else 1.0
    targets: list[tuple[str, float]] = [
        ("p95", 5.0 if percent_scale else 0.05),
        ("p98", 2.0 if percent_scale else 0.02),
    ]
    if custom_percentile is not None:
        cp_val = float(custom_percentile)
        cp_key = f"p{cp_val:g}"
        cp_target = (100.0 - cp_val) * (1.0 if percent_scale else 0.01)
        targets.append((cp_key, cp_target))
    for key, target in targets:
        point = _ccdf_display_percentile_point(
            x_values,
            y_values,
            target_probability=target,
            axis_top=axis_top,
        )
        if point is None:
            continue
        if key in info_use and f"{key}_raw" not in info_use:
            info_use[f"{key}_raw"] = info_use.get(key)
        info_use[f"{key}_plot"] = float(point["x"])
        info_use[key] = float(point["x"])
    return info_use


def _step_series_value_at_x(
    x_values: np.ndarray | Any,
    y_values: np.ndarray | Any,
    x_ref: float,
) -> float | None:
    x_arr = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if int(x_arr.size) == 0 or int(x_arr.size) != int(y_arr.size):
        return None
    idx = int(np.searchsorted(x_arr, float(x_ref), side="right") - 1)
    idx = max(0, min(idx, int(y_arr.size) - 1))
    value = float(y_arr[idx])
    return value if np.isfinite(value) else None


def _apply_matplotlib_ccdf_percentile_guide(
    fig: Figure,
    info: Mapping[str, Any],
    *,
    margin_at: str,
    color: str,
    reference_lines: Sequence[dict[str, Any]] | None = None,
    show_margin: bool = False,
    custom_percentile: float | None = None,
) -> None:
    if not fig.axes:
        return
    ax = fig.axes[0]
    if ax.lines:
        line = ax.lines[0]
        info = _set_display_percentiles_from_ccdf_curve(
            info,
            line.get_xdata(),
            line.get_ydata(),
            custom_percentile=custom_percentile,
        )
    key, value, probability = _margin_reference_percentile_info(
        info, margin_at=margin_at, custom_percentile=custom_percentile,
    )
    if value is None or not np.isfinite(value):
        return
    point: dict[str, float | bool] | None = None
    if ax.lines:
        line = ax.lines[0]
        point = _ccdf_display_percentile_point(
            line.get_xdata(),
            line.get_ydata(),
            target_probability=float(probability),
        )
    marker_y = float(probability)
    if point is not None and bool(point.get("needs_terminal_closure")):
        floor = max(float(ax.get_ylim()[0]), float(point["segment_bottom"]))
        ax.plot(
            [float(point["x"]), float(point["x"])],
            [float(point["segment_top"]), floor],
            color=str(ax.lines[0].get_color()),
            linewidth=float(ax.lines[0].get_linewidth()),
            alpha=float(ax.lines[0].get_alpha() or 1.0),
            zorder=4,
        )
    ax.axvline(float(value), color=color, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.scatter([float(value)], [float(marker_y)], s=28, color=color, zorder=6)
    x_min, x_max = ax.get_xlim()
    x_span = max(abs(float(x_max) - float(x_min)), 1.0e-9)
    near_right_edge = (float(value) - float(x_min)) / x_span >= 0.78
    ax.annotate(
        f"{key} = {float(value):.3f}",
        xy=(float(value), float(marker_y)),
        xytext=(-6, 8) if near_right_edge else (6, 8),
        textcoords="offset points",
        fontsize=9,
        color=color,
        ha="right" if near_right_edge else "left",
        annotation_clip=False,
    )
    if not show_margin or not reference_lines:
        return
    tail_pct = probability * 100.0
    ref_label = f"{key} ({tail_pct:g}% point)"
    margin_lines: list[str] = []
    for ref in reference_lines:
        ref_value = ref.get("value")
        if ref_value is None or not np.isfinite(float(ref_value)):
            continue
        margin = float(ref_value) - float(value)
        need = max(0.0, float(value) - float(ref_value))
        margin_lines.append(
            f"{str(ref.get('label') or ref_value)}: margin={margin:+.3f}, need={need:.3f}"
        )
    if not margin_lines:
        return
    ax.text(
        0.01,
        0.01,
        f"Margin vs {ref_label}\n" + "\n".join(margin_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cbd5e1", alpha=0.94),
    )


def _plotly_distribution_histogram(
    counts: Any,
    edges: Any,
    *,
    title: str,
    xlabel: str,
    color: str = "#16a34a",
    reference_lines: Sequence[dict[str, Any]] | None = None,
    show_margin: bool = False,
    margin_at: str = "p98",
    custom_percentile: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    import plotly.graph_objects as go

    counts_arr = np.asarray(counts, dtype=np.int64).reshape(-1)
    edges_arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if int(edges_arr.size) != int(counts_arr.size) + 1:
        raise RuntimeError("Histogram edges must have length len(counts) + 1.")
    sample_count = int(np.sum(counts_arr, dtype=np.int64))
    if sample_count <= 0:
        raise RuntimeError("Histogram contains no positive samples.")
    cumulative = np.cumsum(counts_arr, dtype=np.int64)
    cum_before = np.concatenate(([0], cumulative[:-1])).astype(np.float64)
    x = edges_arr[:-1]
    y = 100.0 * (1.0 - (cum_before / float(sample_count)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=2),
            line_shape="vh",
            name="CCDF",
        )
    )
    p95 = _histogram_percentile(edges_arr, counts_arr, 95.0)
    p98 = _histogram_percentile(edges_arr, counts_arr, 98.0)
    seed_info: dict[str, Any] = {"p95": p95, "p98": p98}
    if custom_percentile is not None:
        cp_key = f"p{float(custom_percentile):g}"
        seed_info[cp_key] = _histogram_percentile(edges_arr, counts_arr, float(custom_percentile))
    info_seed = _set_display_percentiles_from_ccdf_curve(
        seed_info,
        x,
        y,
        percent_scale=True,
        custom_percentile=custom_percentile,
    )
    ref_key, ref_value, _probability = _margin_reference_percentile_info(
        info_seed,
        margin_at=margin_at,
        custom_percentile=custom_percentile,
    )
    if ref_value is not None:
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x_span = max(abs(x_max - x_min), 1.0e-9)
        near_right_edge = (float(ref_value) - x_min) / x_span >= 0.78
        point = _ccdf_display_percentile_point(
            x,
            y,
            target_probability=_probability * 100.0,
            axis_top=100.0,
        )
        marker_y = None if point is None else float(point["y"])
        if point is not None and bool(point.get("needs_terminal_closure")):
            fig.add_trace(
                go.Scatter(
                    x=[float(point["x"]), float(point["x"])],
                    y=[float(point["segment_top"]), float(point["segment_bottom"])],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name="CCDF tail",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_vline(
            x=float(ref_value),
            line_color=color,
            line_dash="dot",
            opacity=0.7,
            annotation_text=ref_key.upper(),
            annotation_position="top left" if near_right_edge else "top right",
        )
        if marker_y is not None and np.isfinite(marker_y):
            fig.add_trace(
                go.Scatter(
                    x=[float(ref_value)],
                    y=[float(marker_y)],
                    mode="markers+text",
                    marker=dict(color=color, size=8),
                    text=[f"{ref_key} = {float(ref_value):.3f}"],
                    textposition="middle left" if near_right_edge else "top right",
                    name=ref_key.upper(),
                    showlegend=False,
                    hovertemplate=f"{ref_key}=%{{x:.3f}}<br>CCDF=%{{y:.4g}}%<extra></extra>",
                )
            )
    if reference_lines:
        for ref in reference_lines:
            value = ref.get("value")
            if value is None or not np.isfinite(float(value)):
                continue
            fig.add_vline(
                x=float(value),
                line_color=str(ref.get("color") or "#ef4444"),
                line_dash="dash",
                opacity=0.8,
                annotation_text=str(ref.get("label") or ""),
                annotation_position="top right",
            )
    if show_margin and reference_lines:
        ref_name = str(ref_key)
        ref_value = ref_value
        ref_label = "p98 (2% point)" if ref_name == "p98" else "p95 (5% point)"
        if ref_value is not None and np.isfinite(float(ref_value)):
            margin_lines = []
            for ref in reference_lines:
                value = ref.get("value")
                if value is None or not np.isfinite(float(value)):
                    continue
                margin = float(value) - float(ref_value)
                need = max(0.0, float(ref_value) - float(value))
                margin_lines.append(
                    f"{str(ref.get('label') or value)}: margin={margin:+.3f}, need={need:.3f}"
                )
            if margin_lines:
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.01,
                    xanchor="left",
                    yanchor="bottom",
                    showarrow=False,
                    align="left",
                    bgcolor="rgba(255,255,255,0.94)",
                    bordercolor="#d1d5db",
                    borderwidth=1,
                    text=f"Margin vs {ref_label}<br>" + "<br>".join(margin_lines),
                )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.98),
        template="plotly_white",
        xaxis_title=xlabel,
        yaxis_title="CCDF [%]",
        margin=dict(l=60, r=24, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_yaxes(type="log", range=[math.log10(1.0e-6), 2.0])
    return fig, {
        "sample_count": sample_count,
        **info_seed,
        "ecdf_method_used": "prebinned_hist",
    }


def _plotly_distribution_raw(
    values_db: np.ndarray,
    *,
    title: str,
    xlabel: str,
    color: str = "#16a34a",
    hist_bins: int = 4096,
    reference_lines: Sequence[dict[str, Any]] | None = None,
    show_margin: bool = False,
    margin_at: str = "p98",
    custom_percentile: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    values = np.asarray(values_db, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if int(values.size) < 1:
        raise RuntimeError("No positive finite samples were available for this recipe.")
    counts, edges = np.histogram(values, bins=max(32, int(hist_bins)))
    fig, info = _plotly_distribution_histogram(
        counts,
        edges,
        title=title,
        xlabel=xlabel,
        color=color,
        reference_lines=reference_lines,
        show_margin=show_margin,
        margin_at=margin_at,
        custom_percentile=custom_percentile,
    )
    info = dict(info or {})
    info["ecdf_method_used"] = "hist"
    return fig, info


def _plotly_heatmap(
    histogram: np.ndarray,
    *,
    elevation_edges_deg: np.ndarray,
    value_edges_db: np.ndarray,
    title: str,
    ylabel: str,
    colorbar_label: str = "Sample count",
) -> tuple[Any, dict[str, Any]]:
    import plotly.graph_objects as go

    hist = np.asarray(histogram, dtype=np.float64)
    positive = hist > 0
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=_edges_midpoints(elevation_edges_deg),
                y=_edges_midpoints(value_edges_db),
                z=hist.T,
                colorscale="Viridis",
                colorbar=dict(title=colorbar_label),
                hovertemplate="Elevation %{x:.3f} deg<br>Value %{y:.3f} dB<br>Count %{z}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Satellite elevation at RAS station [deg]",
        yaxis_title=ylabel,
        margin=dict(l=70, r=24, t=70, b=60),
    )
    vmax = float(np.max(hist)) if hist.size > 0 else 0.0
    return fig, {
        "sample_count": int(np.sum(hist, dtype=np.int64)),
        "positive_bin_count": int(np.count_nonzero(positive)),
        "max_bin_count": int(vmax),
    }


def _plotly_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
    color: str,
) -> tuple[Any, dict[str, Any]]:
    import plotly.graph_objects as go

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_arr,
                y=y_arr,
                mode="lines",
                line=dict(color=color, width=2),
                connectgaps=False,
                name=ylabel,
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Time [s]",
        yaxis_title=ylabel,
        margin=dict(l=70, r=24, t=70, b=60),
        showlegend=False,
    )
    return fig, {"sample_count": int(np.count_nonzero(np.isfinite(y_arr)))}


def _plotly_series_collection(
    series: Sequence[dict[str, Any]],
    *,
    title: str,
    ylabel: str,
) -> tuple[Any, dict[str, Any]]:
    import plotly.graph_objects as go

    fig = go.Figure()
    sample_counts: dict[str, int] = {}
    for row in series:
        x_arr = np.asarray(row["x"], dtype=np.float64)
        y_arr = np.asarray(row["y"], dtype=np.float64)
        label = str(row["label"])
        fig.add_trace(
            go.Scatter(
                x=x_arr,
                y=y_arr,
                mode="lines",
                line=dict(color=str(row["color"]), width=2),
                connectgaps=False,
                name=label,
            )
        )
        sample_counts[label] = int(np.count_nonzero(np.isfinite(y_arr)))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.985),
        template="plotly_white",
        xaxis_title="Time [s]",
        yaxis_title=ylabel,
        margin=dict(l=80, r=24, t=112, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.035, xanchor="center", x=0.5),
    )
    return fig, {"sample_counts": sample_counts}


def _load_distribution_samples_db(
    filename: str | Path,
    meta: Mapping[str, Any],
    *,
    recipe_id: str,
    integrated: bool,
    integration_window_s: float,
    windowing: str = _DEFAULT_CCDF_WINDOWING,
    params: Mapping[str, Any],
) -> np.ndarray:
    # --- Cache lookup ---
    try:
        file_mtime = Path(filename).stat().st_mtime
    except OSError:
        file_mtime = 0.0
    bw_view_mode = str(params.get("bandwidth_view_mode", _BANDWIDTH_VIEW_CHANNEL_TOTAL))
    bw_mhz = float(params.get("bandwidth_mhz", _DEFAULT_BW_INBAND_MHZ))
    ref_bw_mhz = float(params.get("reference_bandwidth_mhz", 1.0))
    system_filter_key = params.get("_system_filter")  # set by render_recipe caller
    _raw_sys_idx: int | None = None
    if isinstance(system_filter_key, (int, float)) and system_filter_key is not None:
        _raw_sys_idx = int(system_filter_key)
    cache_key = (
        str(filename), file_mtime, recipe_id, integrated,
        integration_window_s, windowing, bw_view_mode, bw_mhz, ref_bw_mhz,
        system_filter_key,
    )
    cached = _DISTRIBUTION_CACHE.get("last")
    if cached is not None and cached[0] == cache_key:
        return cached[1].copy()
    dataset_name = _distribution_raw_dataset_name(recipe_id)
    stream_dataset_name = _resolve_raw_stream_dataset_name(meta, dataset_name)
    view_context = _resolve_bandwidth_view_context(
        filename,
        params=params,
        dataset_name=dataset_name,
    )
    if integrated:
        skip = _integration_skip_message(
            filename,
            meta,
            integration_window_s=float(integration_window_s),
        )
        if skip is not None:
            raise RuntimeError(skip)
        stream = _open_stream(filename, stream_dataset_name, "times", system_index=_raw_sys_idx)["stream"]
        try:
            arr = scenario.process_integration_stream(
                stream,
                value_key=stream_dataset_name,
                times_key="times",
                windowing=str(windowing),
                integration_period=float(integration_window_s) * scenario.u.s,
            )
        except KeyError as exc:
            raise RuntimeError(
                "Raw postprocess stream could not find stored dataset "
                f"{stream_dataset_name!r} for canonical dataset {dataset_name!r}."
            ) from exc
        arr = _norm_dist(arr).astype(np.float64, copy=False)
        if arr.size == 0 or int(arr.shape[0]) == 0:
            raise RuntimeError(
                f"No integrated {dataset_name} samples were produced even though the requested "
                "window fits within the stored span."
            )
    else:
        est = _ds_nbytes(meta, dataset_name)
        if est is not None and est > _MAX_MATERIALIZE_BYTES:
            raise MemoryError(
                f"{dataset_name} is estimated at {est / (1024**2):.1f} MiB, above the "
                f"materialisation limit of {_MAX_MATERIALIZE_BYTES / (1024**2):.1f} MiB."
            )
        parts: list[np.ndarray] = []
        for chunk in _open_stream(filename, stream_dataset_name, system_index=_raw_sys_idx)["stream"]:
            try:
                values = chunk["data"][stream_dataset_name]
            except KeyError as exc:
                raise RuntimeError(
                    "Raw postprocess stream could not find stored dataset "
                    f"{stream_dataset_name!r} for canonical dataset {dataset_name!r}."
                ) from exc
            parts.append(_norm_dist(values).astype(np.float64, copy=False))
        if not parts:
            raise RuntimeError(f"No {dataset_name} samples were read from the file.")
        arr = np.concatenate(parts, axis=0)
    values_db = _lin2db(
        arr,
        nonpositive="tiny",
    )
    values_db += float(view_context["db_offset_db"])
    # --- Cache store ---
    _DISTRIBUTION_CACHE["last"] = (cache_key, values_db.copy())
    return values_db


def _render_distribution_recipe(
    filename: str | Path,
    meta: Mapping[str, Any],
    *,
    recipe_id: str,
    source: str,
    params: Mapping[str, Any],
    engine: str = ENGINE_MATPLOTLIB,
) -> tuple[Any, dict[str, Any]]:
    integrated = _normalize_bool(params.get("integrated"), default=False)
    corridor = _normalize_bool(params.get("corridor"), default=False)
    view_context = _resolve_bandwidth_view_context(
        filename,
        params=params,
        family_name=_distribution_family_name(recipe_id) if source == SOURCE_PREACC else None,
        dataset_name=_distribution_raw_dataset_name(recipe_id),
    )
    title = (
        f"{'Integrated' if integrated else 'Instantaneous'} "
        f"{_distribution_label_from_recipe(recipe_id)} CCDF"
        f"{_fmt_bw_parens(view_context['bandwidth_label'])}"
        f"{' with skycell corridor' if corridor else ''}"
    )
    xlabel = _distribution_unit_label(
        recipe_id,
        view_mode=str(view_context["view_mode"]),
        bandwidth_label=str(view_context["bandwidth_label"]),
    )
    family_name = _distribution_family_name(recipe_id)
    integration_window_s = float(
        params.get("integration_window_s", _DEFAULT_INTEGRATION_WINDOW_S)
    )
    raw_hist_bins = int(params.get("raw_hist_bins", 4096))
    reference_lines, reference_kwargs = _reference_plot_kwargs(params)
    normalize_grx = _normalize_bool(params.get("normalize_by_max_grx"), default=False)
    grx_max_db_shift = 0.0
    if normalize_grx:
        attrs = meta.get("attrs") or meta  # describe_data nests under "attrs"
        grx_max_dbi = attrs.get("ras_max_gain_dbi")
        # Fallback: compute from stored diameter and wavelength.
        # Custom-RAS runs store ``ras_station_ant_diam_m = NaN`` as
        # "not applicable" (see scenario.py), so require the diameter
        # to be finite before invoking the RA.1631-style derivation —
        # otherwise this silently produces a NaN plot title.
        if grx_max_dbi is None:
            _diam = attrs.get("ras_station_ant_diam_m")
            _wlen = attrs.get("wavelength_m")
            if (
                _diam is not None
                and _wlen is not None
                and np.isfinite(float(_diam))
                and np.isfinite(float(_wlen))
                and float(_wlen) > 0
            ):
                _d_wlen = float(_diam) / float(_wlen)
                grx_max_dbi = 10.0 * np.log10((np.pi * _d_wlen) ** 2)
        if grx_max_dbi is not None and float(grx_max_dbi) != 0.0:
            grx_max_db_shift = -float(grx_max_dbi)
            title += f" (S.1586 norm., G_rx,max={float(grx_max_dbi):.1f} dBi)"
    color = {
        "prx_total_distribution": "#0f766e",
        "epfd_distribution": "#2563eb",
        "total_pfd_ras_distribution": "#7c3aed",
    }[recipe_id]
    use_plotly = _normalize_render_engine(engine) == ENGINE_PLOTLY
    # Resolve per-system filter for preaccumulated reads
    _sys_filt = params.get("_system_filter")
    _preacc_sys_idx: int | None = None
    _preacc_group_prefix: str | None = None
    if isinstance(_sys_filt, str) and _sys_filt not in ("overlay",) and "/" in _sys_filt:
        _preacc_group_prefix = _sys_filt
    elif isinstance(_sys_filt, (int, float)) and _sys_filt is not None:
        _preacc_sys_idx = int(_sys_filt)
    if source == SOURCE_PREACC:
        counts = _read_preacc(filename, f"{family_name}/counts", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
        sample_count = int(np.sum(np.asarray(counts, dtype=np.int64), dtype=np.int64))
        if sample_count <= 0:
            zero_leakage_text = _preacc_zero_leakage_diagnostic_text(
                filename,
                family_name=family_name,
                sample_count=sample_count,
            )
            if zero_leakage_text is not None:
                raise RuntimeError(zero_leakage_text)
        edges = np.asarray(
            _read_preacc(filename, f"{family_name}/edges_dbw", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix),
            dtype=np.float64,
        )
        edges = edges + float(view_context["db_offset_db"]) + grx_max_db_shift
        if use_plotly:
            fig, info = _plotly_distribution_histogram(
                counts,
                edges,
                title=title,
                xlabel=xlabel,
                color=color,
                reference_lines=reference_lines,
                show_margin=bool(reference_kwargs["show_margin"]),
                margin_at=str(reference_kwargs["margin_at"]),
                custom_percentile=reference_kwargs.get("custom_percentile"),
            )
        else:
            fig, info = _plot_distribution_histogram(
                counts,
                edges,
                title=title,
                xlabel=xlabel,
                color=color,
                reference_lines=reference_lines,
                show_margin=bool(reference_kwargs["show_margin"]),
                margin_at=str(reference_kwargs["margin_at"]),
                custom_percentile=reference_kwargs.get("custom_percentile"),
            )
        attrs = _read_preacc_group_attrs(filename, family_name, system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
        info = dict(info or {})
        if attrs:
            info["preaccumulated_attrs"] = attrs
        if view_context["warning_text"]:
            info["bandwidth_warning"] = str(view_context["warning_text"])
        info["bandwidth_view"] = dict(view_context)
        info["reference_lines"] = reference_lines
        _grid = str(params.get("grid_tick_density", "normal"))
        for ax in getattr(fig, "axes", []):
            _apply_grid_tick_density(ax, _grid)
        return fig, info
    windowing = str(params.get("windowing", _DEFAULT_CCDF_WINDOWING))
    values_db = _load_distribution_samples_db(
        filename,
        meta,
        recipe_id=recipe_id,
        integrated=integrated,
        integration_window_s=integration_window_s,
        windowing=windowing,
        params=params,
    )
    if grx_max_db_shift != 0.0:
        values_db = values_db + grx_max_db_shift
    if use_plotly:
        return _plotly_distribution_raw(
            values_db,
            title=title,
            xlabel=xlabel,
            color=color,
            hist_bins=raw_hist_bins,
            reference_lines=reference_lines,
            show_margin=bool(reference_kwargs["show_margin"]),
            margin_at=str(reference_kwargs["margin_at"]),
            custom_percentile=reference_kwargs.get("custom_percentile"),
        )
    fig, info = _plot_distribution_raw(
        values_db,
        title=title,
        xlabel=xlabel,
        color=color,
        corridor=corridor,
        hist_bins=raw_hist_bins,
        reference_lines=reference_lines,
        show_margin=bool(reference_kwargs["show_margin"]),
        margin_at=str(reference_kwargs["margin_at"]),
        custom_percentile=reference_kwargs.get("custom_percentile"),
    )
    info = dict(info or {})
    info["reference_lines"] = reference_lines
    if view_context["warning_text"]:
        info["bandwidth_warning"] = str(view_context["warning_text"])
    info["bandwidth_view"] = dict(view_context)
    _grid = str(params.get("grid_tick_density", "normal"))
    for ax in getattr(fig, "axes", []):
        _apply_grid_tick_density(ax, _grid)
    return fig, info


def _grid_info(n: int) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    n_int = int(n)
    elev = (15, 90) if n_int == 1734 else (0, 90) if n_int == 2334 else None
    if elev is None:
        return None, None
    _az, _el, grid = pointgen_S_1586_1(1, elev_range=elev)
    return grid, elev


def _render_hemisphere_recipe(
    filename: str | Path,
    meta: Mapping[str, Any],
    *,
    recipe_id: str,
    params: Mapping[str, Any],
    primary_power_dataset: str | None = None,
    engine: str = ENGINE_MATPLOTLIB,
) -> tuple[Any, dict[str, Any]]:
    primary = _primary_power_dataset(meta, preferred=primary_power_dataset)
    if primary is None:
        raise RuntimeError("No primary power dataset is available in this file.")
    integrated = _normalize_bool(params.get("integrated"), default=False)
    integration_window_s = float(
        params.get("integration_window_s", _DEFAULT_INTEGRATION_WINDOW_S)
    )
    primary_recipe_id = {
        "Prx_total_W": "prx_total_distribution",
        "EPFD_W_m2": "epfd_distribution",
        "PFD_total_RAS_STATION_W_m2": "total_pfd_ras_distribution",
    }[str(primary)]
    view_context = _resolve_bandwidth_view_context(
        filename,
        params=params,
        dataset_name=primary,
    )
    windowing = str(params.get("windowing", _DEFAULT_CCDF_WINDOWING))
    values_db = _load_distribution_samples_db(
        filename,
        meta,
        recipe_id=primary_recipe_id,
        integrated=integrated,
        integration_window_s=integration_window_s,
        windowing=windowing,
        params=params,
    )
    if values_db.ndim < 2:
        raise RuntimeError(f"{primary!r} does not expose a skycell axis for hemisphere rendering.")
    grid_info, elev_range = _grid_info(values_db.shape[-1])
    if grid_info is None:
        raise RuntimeError(
            f"No built-in S.1586 grid mapping is defined for {int(values_db.shape[-1])} skycells."
        )
    mode_label = "Integrated" if integrated else "Instantaneous"
    is_3d = recipe_id.endswith("_3d")
    mode = "power" if "percentile" in recipe_id else "data_loss"
    common_kwargs: dict[str, Any] = dict(
        grid_info=grid_info,
        elev_range=elev_range or _DEFAULT_S1586_ELEV_RANGE,
        cell_axis=-1,
        engine="plotly" if _normalize_render_engine(engine) == ENGINE_PLOTLY else "mpl",
        show=False,
        return_values=True,
        title=(
            f"{mode_label} {_distribution_label_from_recipe(primary_recipe_id).lower()} "
            f"{'percentile' if mode == 'power' else 'data-loss'} {'3D' if is_3d else 'map'} "
            f"{_fmt_bw_parens(view_context['bandwidth_label']).strip()}"
        ),
    )
    if mode == "power":
        common_kwargs["mode"] = "power"
        common_kwargs["worst_percent"] = float(params.get("worst_percent", _DEFAULT_WORST_PERCENT))
    else:
        common_kwargs["mode"] = "data_loss"
        common_kwargs["protection_criterion"] = float(
            params.get("protection_criterion_db", _DEFAULT_DATA_LOSS_THRESHOLD_DBW)
        ) + float(view_context["db_offset_db"])
        override_colormap_min = _normalize_bool(
            params.get("override_colormap_min"),
            default=False,
        )
        override_colormap_max = _normalize_bool(
            params.get("override_colormap_max"),
            default=False,
        )
        if override_colormap_min and override_colormap_max:
            colormap_min_pct = float(params.get("colormap_min_pct", 0.0) or 0.0)
            colormap_max_pct = float(params.get("colormap_max_pct", 100.0) or 100.0)
            if colormap_max_pct <= colormap_min_pct:
                raise ValueError(
                    "Colormap max [%] must be greater than colormap min [%] when both overrides are enabled."
                )
        if override_colormap_min:
            common_kwargs["vmin"] = float(params.get("colormap_min_pct", 0.0) or 0.0)
        if override_colormap_max:
            common_kwargs["vmax"] = float(params.get("colormap_max_pct", 100.0) or 100.0)
    if is_3d:
        fig, values = visualise.plot_hemisphere_3D(values_db, **common_kwargs)
    else:
        projection = str(params.get("projection", "polar"))
        fig, values = visualise.plot_hemisphere_2D(
            values_db, projection=projection, **common_kwargs,
        )
    summary_metric_label, summary_unit = _hemisphere_summary_metric(
        primary_recipe_id,
        mode=mode,
    )
    summary_avg_value: float | None = None
    summary_max_value: float | None = None
    summary_annotated = False
    if _normalize_bool(
        params.get("show_summary_stats"),
        default=False,
    ):
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            summary_avg_value = float(np.mean(finite_values))
            summary_max_value = float(np.max(finite_values))
            summary_text = _format_hemisphere_summary_text(
                metric_label=summary_metric_label,
                summary_avg_value=summary_avg_value,
                summary_max_value=summary_max_value,
                summary_unit=summary_unit,
            )
            summary_annotated = _annotate_hemisphere_summary_footer(
                fig,
                summary_text=summary_text,
                engine=engine,
            )
    loss_summary_avg_pct = summary_avg_value if mode == "data_loss" else None
    loss_summary_max_pct = summary_max_value if mode == "data_loss" else None
    loss_summary_annotated = summary_annotated if mode == "data_loss" else False
    recipe_label = (
        f"{mode_label} hemisphere {'percentile' if mode == 'power' else 'data-loss'} "
        f"{'3D' if is_3d else 'map'}"
    )
    return fig, {
        "primary_power_dataset": primary,
        "value_shape": tuple(np.asarray(values).shape),
        "mode": mode,
        "integrated": bool(integrated),
        "integrated_window_s": float(integration_window_s),
        "title": common_kwargs["title"],
        "recipe_label": recipe_label,
        "colormap_min_pct": common_kwargs.get("vmin"),
        "colormap_max_pct": common_kwargs.get("vmax"),
        "bandwidth_view": dict(view_context),
        "bandwidth_warning": str(view_context["warning_text"] or ""),
        "summary_avg_value": summary_avg_value,
        "summary_max_value": summary_max_value,
        "summary_metric_label": summary_metric_label,
        "summary_unit": summary_unit,
        "summary_annotated": bool(summary_annotated),
        "loss_summary_avg_pct": loss_summary_avg_pct,
        "loss_summary_max_pct": loss_summary_max_pct,
        "loss_summary_annotated": bool(loss_summary_annotated),
    }


def _series_display_settings(params: Mapping[str, Any]) -> tuple[int, float]:
    return (
        max(0, int(params.get("max_points", 4000) or 0)),
        max(0.0, float(params.get("smoothing_window_s", 0.0) or 0.0)),
    )


def _annotate_hemisphere_summary_footer(
    fig: Any,
    *,
    summary_text: str,
    engine: str,
) -> bool:
    if _normalize_render_engine(engine) == ENGINE_PLOTLY:
        if hasattr(fig, "update_layout"):
            existing_bottom = 0
            layout = getattr(fig, "layout", None)
            margin = getattr(layout, "margin", None)
            margin_bottom = getattr(margin, "b", None)
            if margin_bottom is not None:
                existing_bottom = int(margin_bottom)
            fig.update_layout(margin={"b": max(existing_bottom, 80)})
        fig.add_annotation(
            text=summary_text,
            x=0.5,
            y=0.01,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="bottom",
            font={"size": 12, "color": "#111827"},
            bgcolor="rgba(255, 255, 255, 0.90)",
            bordercolor="rgba(148, 163, 184, 0.55)",
            borderwidth=1,
        )
        return True
    if hasattr(fig, "subplots_adjust") and hasattr(fig, "subplotpars"):
        fig.subplots_adjust(bottom=max(float(fig.subplotpars.bottom), 0.13))
    text_kwargs = {
        "ha": "center",
        "va": "bottom",
        "fontsize": 10.5,
        "color": "#111827",
        "bbox": {
            "boxstyle": "round,pad=0.35",
            "facecolor": (1.0, 1.0, 1.0, 0.90),
            "edgecolor": (148 / 255.0, 163 / 255.0, 184 / 255.0, 0.55),
        },
    }
    fig.text(0.5, 0.02, summary_text, **text_kwargs)
    return True


def _beam_time_series_segments(
    filename: str | Path,
    *,
    recipe_id: str,
    source: str,
    system_index: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    _gp = f"system_{int(system_index)}/" if system_index is not None else ""
    if recipe_id == "beam_count_total_over_time":
        if source == SOURCE_PREACC:
            return _series_segments_from_preacc(
                filename,
                _read_preacc(filename, "beam_statistics/network_total_beams_over_time", system_index=system_index),
            )
        x_segments = _relative_iteration_time_segments(filename)
        y_segments = []
        for ii in _iteration_ids(filename, group_prefix=_gp):
            counts = _read_iter_dataset(filename, "sat_beam_counts_used", ii, group_prefix=_gp)
            y_segments.append(
                _beam_total_over_time_from_counts(counts).reshape(-1)
            )
        return x_segments, y_segments

    if recipe_id == "beam_count_visible_over_time":
        if source == SOURCE_PREACC:
            return _series_segments_from_preacc(
                filename,
                _read_preacc(filename, "beam_statistics/visible_total_beams_over_time", system_index=system_index),
            )
        x_segments = _relative_iteration_time_segments(filename)
        y_segments = []
        for ii in _iteration_ids(filename, group_prefix=_gp):
            counts = _read_iter_dataset(filename, "sat_beam_counts_used", ii, group_prefix=_gp)
            elev = _read_iter_dataset(filename, "sat_elevation_RAS_STATION_deg", ii, group_prefix=_gp)
            vis_mask = np.asarray(elev, dtype=np.float64) > 0.0
            y_segments.append(
                _beam_total_over_time_from_counts(
                    counts,
                    visible_mask=vis_mask,
                ).reshape(-1)
            )
        return x_segments, y_segments
    if recipe_id == "beam_demand_over_time":
        if source == SOURCE_PREACC:
            return _series_segments_from_preacc(
                filename,
                _read_preacc(filename, "beam_statistics/beam_demand_over_time", system_index=system_index),
            )
        x_segments = _relative_iteration_time_segments(filename)
        y_segments = []
        for ii in _iteration_ids(filename, group_prefix=_gp):
            y_segments.append(
                np.asarray(
                    _read_iter_dataset(filename, "beam_demand_count", ii, group_prefix=_gp),
                    dtype=np.float64,
                ).reshape(-1)
            )
        return x_segments, y_segments
    if recipe_id == "beam_demand_minus_service_over_time":
        return _subtract_series_segments(
            _beam_time_series_segments(
                filename,
                recipe_id="beam_demand_over_time",
                source=source,
                system_index=system_index,
            ),
            _beam_time_series_segments(
                filename,
                recipe_id="beam_count_total_over_time",
                source=source,
                system_index=system_index,
            ),
        )
    raise KeyError(recipe_id)


def _render_beam_time_series_recipe(
    filename: str | Path,
    *,
    recipe_id: str,
    source: str,
    params: Mapping[str, Any],
    engine: str,
    system_index: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    max_points, smoothing_window_s = _series_display_settings(params)
    x_segments, y_segments = _beam_time_series_segments(
        filename,
        recipe_id=recipe_id,
        source=source,
        system_index=system_index,
    )
    x_segments, y_segments = _prepare_display_series_segments(
        x_segments,
        y_segments,
        max_points=max_points,
        smoothing_window_s=smoothing_window_s,
    )
    x, y = _join_series_segments(x_segments, y_segments)
    recipe_meta = {
        "beam_count_total_over_time": (
            "Service beams in network over time",
            "Service beams",
            "#0f766e",
        ),
        "beam_count_visible_over_time": (
            "RAS-visible beams over time",
            "Active beams",
            "#2563eb",
        ),
        "beam_demand_over_time": (
            "Beam demand over time",
            "Demanded beams",
            "#dc2626",
        ),
    }[recipe_id]
    title, ylabel, color = recipe_meta
    if _normalize_render_engine(engine) == ENGINE_PLOTLY:
        fig, info = _plotly_series(x, y, title=title, ylabel=ylabel, color=color)
    else:
        fig, info = _plot_series(x, y, title=title, ylabel=ylabel, color=color)
    info = dict(info or {})
    info["max_points"] = int(max_points)
    info["smoothing_window_s"] = float(smoothing_window_s)
    return fig, info


def _render_beam_overview_recipe(
    filename: str | Path,
    *,
    source: str,
    params: Mapping[str, Any],
    engine: str,
    system_index: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    catalog = _beam_overview_series_catalog()
    max_points, smoothing_window_s = _series_display_settings(params)
    selected_keys = [key for key in catalog.keys() if _normalize_bool(params.get(key), default=True)]
    if not selected_keys:
        raise RuntimeError("Enable at least one beam-over-time curve.")
    series_rows: list[dict[str, Any]] = []
    show_zero_reference = False
    for key in selected_keys:
        spec = catalog[key]
        x_segments, y_segments = _beam_time_series_segments(
            filename,
            recipe_id=str(spec["recipe_id"]),
            source=source,
            system_index=system_index,
        )
        x_segments, y_segments = _prepare_display_series_segments(
            x_segments,
            y_segments,
            max_points=max_points,
            smoothing_window_s=smoothing_window_s,
        )
        x, y = _join_series_segments(x_segments, y_segments)
        series_rows.append(
            {
                "label": str(spec["label"]),
                "color": str(spec["color"]),
                "x": x,
                "y": y,
            }
        )
        if str(spec["recipe_id"]) == "beam_demand_minus_service_over_time":
            show_zero_reference = True
    if _normalize_render_engine(engine) == ENGINE_PLOTLY:
        fig, info = _plotly_series_collection(
            series_rows,
            title="Beam overview over time",
            ylabel="Beam count",
        )
        if show_zero_reference:
            fig.add_hline(y=0.0, line_color="#64748b", line_dash="dot", opacity=0.9)
    else:
        fig, info = _plot_series_collection(
            series_rows,
            title="Beam overview over time",
            ylabel="Beam count",
        )
        if show_zero_reference:
            fig.axes[0].axhline(0.0, color="#64748b", linestyle=":", linewidth=1.0, zorder=1)
    info = dict(info or {})
    info["source_used"] = str(source)
    info["selected_series"] = [str(catalog[key]["label"]) for key in selected_keys]
    info["max_points"] = int(max_points)
    info["smoothing_window_s"] = float(smoothing_window_s)
    info["zero_reference_line"] = bool(show_zero_reference)
    return fig, info


def _beam_cap_policy_label(policy_key: str) -> str:
    return {
        "simpson": "Simpson",
        "full_reroute": "Full reroute",
        "no_reroute": "No reroute",
        "pure_reroute": "Pure reroute",
        "belt_sky_temporal": "Belt sky temporal",
    }.get(str(policy_key), str(policy_key).replace("_", " ").title())


def _beam_cap_selected_caps_summary(selected_caps: Mapping[str, Any]) -> str:
    rows = [
        f"{_beam_cap_policy_label(str(policy_key))}: {int(value)}"
        for policy_key, value in selected_caps.items()
        if value is not None
    ]
    if not rows:
        return "Recommended beam caps: unavailable"
    return "Recommended beam caps: " + " | ".join(rows)


def _render_beam_cap_sizing_analysis(
    filename: str | Path,
    *,
    params: Mapping[str, Any],
    engine: str,
    system_index: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    root_attrs = _read_root_attrs(filename)
    enabled_keys = tuple(
        key
        for key, enabled in (
            ("simpson", _normalize_bool(params.get("policy_simpson"), default=True)),
            ("full_reroute", _normalize_bool(params.get("policy_full_reroute"), default=True)),
            ("no_reroute", _normalize_bool(params.get("policy_no_reroute"), default=True)),
            ("pure_reroute", _normalize_bool(params.get("policy_pure_reroute"), default=False)),
        )
        if enabled
    )
    if not enabled_keys:
        raise RuntimeError("Enable at least one beam-cap policy.")
    nco_override = int(params.get("nco_override", 0) or 0)
    # UEMR runs store ``nco="n/a"`` because the concept doesn't apply.
    # The beam-cap sizing recipe has no meaning on a UEMR output — refuse
    # rather than silently coercing the sentinel into a misleading 1.
    _nco_raw = root_attrs.get("nco", 1)
    if isinstance(_nco_raw, (bytes, bytearray)):
        _nco_raw = _nco_raw.decode("utf-8", errors="replace")
    _uemr_flag = bool(root_attrs.get("uemr_mode", False))
    if _uemr_flag or (isinstance(_nco_raw, str) and not _nco_raw.strip().isdigit()):
        raise RuntimeError(
            "Beam-cap sizing is not applicable to UEMR (isotropic per-satellite) "
            "runs — they have no beam library. Pick a different postprocess recipe."
        )
    nco_from_file = int(_nco_raw or 1)
    _beam_group_prefix = f"system_{int(system_index)}/" if system_index is not None else ""
    result = nbeam.run_beam_cap_sizing(
        filename,
        group_prefix=_beam_group_prefix,
        config=nbeam.BeamCapSizingConfig(
            enabled_policy_keys=enabled_keys,
            beam_cap_min=int(params.get("beam_cap_min", 0)),
            beam_cap_max=int(params.get("beam_cap_max", 260)),
            nco=(int(nco_from_file) if nco_override <= 0 else int(nco_override)),
            loss_slot_target=float(params.get("loss_slot_target", 1e-2)),
            lost_demand_target=float(params.get("lost_demand_target", 1e-3)),
            per_slot_loss_tolerance=float(params.get("per_slot_loss_tolerance", 1e-3)),
            pure_reroute_backend=str(params.get("pure_reroute_backend", "auto") or "auto"),
            max_demand_slots=(
                None if int(params.get("max_demand_slots", 0) or 0) <= 0 else int(params.get("max_demand_slots"))
            ),
            emit_progress_output=False,
            save_outputs=False,
            show_plots=False,
            enable_skycell_demand_vis=False,
        ),
    )
    curves = result.get("policy_curves", {}) or {}
    selected_caps = result.get("selected_caps", {}) or {}
    selected_caps_summary = _beam_cap_selected_caps_summary(selected_caps)
    if _normalize_render_engine(engine) == ENGINE_PLOTLY:
        import plotly.graph_objects as go

        fig = go.Figure()
        for policy_key, curve in curves.items():
            cap_grid = np.asarray(curve.get("beam_caps", curve.get("beam_caps_grid", [])), dtype=np.float64)
            if cap_grid.size <= 0:
                cap_grid = np.asarray(result.get("beam_caps", []), dtype=np.float64)
            tail = np.asarray(curve.get("tail_risk_percent", curve.get("tail_risk", [])), dtype=np.float64)
            if cap_grid.size == 0 or tail.size == 0:
                continue
            fig.add_trace(go.Scatter(x=cap_grid, y=tail, mode="lines", name=str(policy_key)))
            selected = selected_caps.get(policy_key)
            if selected is not None:
                fig.add_vline(
                    x=float(selected),
                    line_dash="dot",
                    opacity=0.6,
                    annotation_text=f"{_beam_cap_policy_label(str(policy_key))}: {int(selected)}",
                    annotation_position="top left",
                )
        fig.update_layout(
            title="Beam-cap sizing analysis",
            template="plotly_white",
            xaxis_title="Beam cap",
            yaxis_title="Tail risk / SLA metric",
            margin=dict(l=70, r=24, t=80, b=78),
        )
        return fig, {
            "selected_caps": selected_caps,
            "selected_caps_summary": selected_caps_summary,
            "enabled_policy_keys": list(enabled_keys),
            "run_diagnostics": result.get("run_diagnostics", {}),
        }
    fig, axes = visualise._new_mpl_subplots(3, 1, figsize=(9.6, 9.2))
    titles = (
        ("tail_risk_percent", "tail_risk", "Tail risk [%]"),
        ("delta_percent", "delta", "Delta [%]"),
        ("eps_percent", "eps", "Epsilon [%]"),
    )
    colors = ("#0f766e", "#2563eb", "#dc2626", "#7c3aed")
    for ax, (metric_percent_key, metric_key, ylabel), color in zip(axes, titles, colors):
        for idx, (policy_key, curve) in enumerate(curves.items()):
            cap_grid = np.asarray(curve.get("beam_caps", curve.get("beam_caps_grid", [])), dtype=np.float64)
            if cap_grid.size <= 0:
                cap_grid = np.asarray(result.get("beam_caps", []), dtype=np.float64)
            values = curve.get(metric_percent_key)
            if values is None:
                values = curve.get(metric_key)
                if values is not None and metric_key in {"delta", "eps", "tail_risk"}:
                    values = 100.0 * np.asarray(values, dtype=np.float64)
            values_arr = np.asarray(values if values is not None else [], dtype=np.float64)
            if cap_grid.size == 0 or values_arr.size == 0:
                continue
            ax.plot(cap_grid, values_arr, label=str(policy_key), color=colors[idx % len(colors)], linewidth=1.8)
            selected = selected_caps.get(policy_key)
            if selected is not None:
                ax.axvline(float(selected), color=colors[idx % len(colors)], linestyle=":", linewidth=1.0, alpha=0.75)
                if ax is axes[0]:
                    ax.annotate(
                        f"{_beam_cap_policy_label(str(policy_key))}: {int(selected)}",
                        xy=(float(selected), 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(5, -12 - idx * 14),
                        textcoords="offset points",
                        rotation=90,
                        ha="left",
                        va="top",
                        fontsize=8.5,
                        color=colors[idx % len(colors)],
                    )
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#9ca3af", alpha=0.4, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title("Beam-cap sizing analysis", pad=12, fontweight="semibold")
    axes[-1].set_xlabel("Beam cap")
    axes[0].legend(loc="upper right", frameon=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.11, hspace=0.16)
    return fig, {
        "selected_caps": selected_caps,
        "selected_caps_summary": selected_caps_summary,
        "enabled_policy_keys": list(enabled_keys),
        "run_diagnostics": result.get("run_diagnostics", {}),
        "nco_used": int(nco_from_file) if nco_override <= 0 else int(nco_override),
    }


_OVERLAY_SYSTEM_COLORS = ("#22c55e", "#0ea5e9", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6")


def _render_overlay_distribution(
    filename: str | Path,
    meta: Mapping[str, Any],
    *,
    recipe_id: str,
    source: str,
    params: Mapping[str, Any],
    selected_system_indices: list[int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render per-system CCDF overlay on a single matplotlib figure.

    When ``selected_system_indices`` is a subset, only those systems are
    plotted and their linear-power sum is shown as the combined curve.
    """
    systems_meta = meta.get("systems", []) or []
    if not systems_meta:
        raise RuntimeError("No per-system data found in this file for overlay mode.")

    # Determine which systems to plot
    if selected_system_indices is not None:
        plot_indices = [i for i in selected_system_indices if 0 <= i < len(systems_meta)]
    else:
        plot_indices = list(range(len(systems_meta)))

    from matplotlib.figure import Figure
    fig = Figure(figsize=(8.5, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    label_base = _distribution_label_from_recipe(recipe_id)
    integrated = _normalize_bool(params.get("integrated"), default=False)
    n_sel = len(plot_indices)
    n_tot = len(systems_meta)
    subtitle = (
        "All systems" if n_sel == n_tot
        else f"{n_sel} of {n_tot} systems"
    )
    ax.set_title(
        f"{'Integrated' if integrated else 'Instantaneous'} {label_base} CCDF\n"
        f"System overlay ({subtitle})"
    )
    view_context = _resolve_bandwidth_view_context(
        str(filename), params=params,
        dataset_name=_distribution_raw_dataset_name(recipe_id),
    )
    xlabel = _distribution_unit_label(
        recipe_id, view_mode=str(view_context["view_mode"]),
        bandwidth_label=str(view_context["bandwidth_label"]),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CCDF")
    ax.set_yscale("log")
    ax.grid(True, color="#9ca3af", alpha=0.3, linestyle=":")

    info: dict[str, Any] = {
        "overlay_systems": [],
        "selected_indices": plot_indices,
        "missing_system_indices": [],
        "missing_system_details": [],
    }
    integration_window_s = float(params.get("integration_window_s", _DEFAULT_INTEGRATION_WINDOW_S))
    windowing = str(params.get("windowing", _DEFAULT_CCDF_WINDOWING))

    # If all systems selected, plot combined (root) data as thick white line
    combined_failed = False
    if n_sel == n_tot:
        try:
            combined_params = dict(params)
            combined_params["_system_filter"] = None
            values_db = _load_distribution_samples_db(
                filename, meta, recipe_id=recipe_id,
                integrated=integrated,
                integration_window_s=integration_window_s,
                windowing=windowing,
                params=combined_params,
            )
            sorted_vals = np.sort(values_db)
            ccdf = np.arange(len(sorted_vals), 0, -1, dtype=np.float64) / len(sorted_vals)
            ax.plot(sorted_vals, ccdf, color="#ffffff", linewidth=2.5, alpha=0.9,
                    label="Combined", zorder=10)
        except Exception as _exc_combined:
            combined_failed = True
            info["combined_load_error"] = (
                f"{type(_exc_combined).__name__}: {_exc_combined}"
            )
            warnings.warn(
                f"Overlay CCDF: combined (root) data for recipe {recipe_id!r} "
                f"could not be loaded from {filename!r}: "
                f"{type(_exc_combined).__name__}: {_exc_combined}. "
                "The thick 'Combined' trace will be missing from the plot.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Plot per-system data
    for sys_idx in plot_indices:
        sys_info = systems_meta[sys_idx]
        sys_name = sys_info.get("system_name", f"System {sys_idx + 1}")
        color = _OVERLAY_SYSTEM_COLORS[sys_idx % len(_OVERLAY_SYSTEM_COLORS)]
        info["overlay_systems"].append({
            "system_name": sys_name,
            "system_index": sys_idx,
            "color": color,
        })
        # Try to load per-system data from /system_N/ group
        try:
            sys_params = dict(params)
            sys_params["_system_filter"] = int(sys_idx)
            values_db = _load_distribution_samples_db(
                filename, meta, recipe_id=recipe_id,
                integrated=integrated,
                integration_window_s=integration_window_s,
                windowing=windowing,
                params=sys_params,
            )
            sorted_vals = np.sort(values_db)
            ccdf = np.arange(len(sorted_vals), 0, -1, dtype=np.float64) / len(sorted_vals)
            ax.plot(sorted_vals, ccdf, color=color, linewidth=1.5, alpha=0.85,
                    label=sys_name)
        except Exception as _exc_sys:
            info["missing_system_indices"].append(int(sys_idx))
            info["missing_system_details"].append(
                f"{int(sys_idx)}:{sys_name}:{type(_exc_sys).__name__}:{_exc_sys}"
            )
            warnings.warn(
                f"Overlay CCDF: per-system data for system_{sys_idx} "
                f"({sys_name!r}) could not be loaded from {filename!r}: "
                f"{type(_exc_sys).__name__}: {_exc_sys}. "
                "This system will be missing from the overlay plot.",
                RuntimeWarning,
                stacklevel=2,
            )
    info["combined_failed"] = combined_failed

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", fontsize=8)
    return fig, info


def render_recipe(
    filename: str | Path,
    recipe_id: str,
    *,
    primary_power_dataset: str | None = None,
    source_preference: str = SOURCE_AUTO,
    params: Mapping[str, Any] | None = None,
    engine: str = ENGINE_MATPLOTLIB,
    system_filter: int | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render one supported plot recipe for the GUI.

    Parameters
    ----------
    system_filter : int or str or None, optional
        Multi-system filter.  ``None`` uses the combined (default) datasets.
        An integer selects a specific per-system dataset group.
        ``"overlay"`` plots all systems on the same axes (future).
    """
    filename_use = str(filename)
    recipe = RECIPE_BY_ID[recipe_id]
    meta = scenario.describe_data(filename_use)
    params_use = normalize_recipe_parameters(recipe_id, params)
    params_use["_system_filter"] = system_filter  # for cache key in _load_distribution_samples_db
    # Resolve per-system index or group prefix for preaccumulated reads
    _preacc_sys_idx: int | None = None
    _preacc_group_prefix: str | None = None
    if isinstance(system_filter, str) and system_filter not in ("overlay",) and "/" in system_filter:
        _preacc_group_prefix = system_filter
    elif isinstance(system_filter, (int, float)) and system_filter is not None:
        _preacc_sys_idx = int(system_filter)
    engine_use = _normalize_render_engine(engine)
    source = resolve_recipe_source(
        meta,
        recipe_id,
        primary_power_dataset=primary_power_dataset,
        source_preference=source_preference,
        params=params_use,
        filename=filename_use,
    )

    # Overlay / multi-select mode: render selected systems' CCDFs on one axes
    if (
        (system_filter == "overlay" or isinstance(system_filter, tuple))
        and recipe_id in {"prx_total_distribution", "epfd_distribution", "total_pfd_ras_distribution"}
        and engine_use == ENGINE_MATPLOTLIB
    ):
        selected_indices = (
            list(system_filter) if isinstance(system_filter, tuple)
            else None  # overlay = all systems
        )
        return _render_overlay_distribution(
            filename_use, meta, recipe_id=recipe_id, source=source,
            params=params_use, selected_system_indices=selected_indices,
        )

    if recipe_id in {
        "prx_total_distribution",
        "epfd_distribution",
        "total_pfd_ras_distribution",
    }:
        fig, info = _render_distribution_recipe(
            filename_use,
            meta,
            recipe_id=recipe_id,
            source=source,
            params=params_use,
            engine=engine_use,
        )
    elif recipe_id == "per_satellite_pfd_distribution":
        reference_lines, reference_kwargs = _reference_plot_kwargs(params_use)
        view_context = _resolve_bandwidth_view_context(
            filename_use,
            params=params_use,
            family_name="per_satellite_pfd_distribution" if source == SOURCE_PREACC else None,
            dataset_name="PFD_per_sat_RAS_STATION_W_m2",
        )
        xlabel = f"PFD contribution [dBW/m\u00b2{_fmt_bw_suffix(view_context['bandwidth_label'])}]"
        title = f"Per-satellite PFD contribution CCDF{_fmt_bw_parens(view_context['bandwidth_label'])}"
        if source == SOURCE_PREACC:
            edges = np.asarray(
                _read_preacc(filename_use, "per_satellite_pfd_distribution/edges_dbw", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix),
                dtype=np.float64,
            ) + float(view_context["db_offset_db"])
            if engine_use == ENGINE_PLOTLY:
                fig, info = _plotly_distribution_histogram(
                    _read_preacc(filename_use, "per_satellite_pfd_distribution/counts", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix),
                    edges,
                    title=title,
                    xlabel=xlabel,
                    color="#dc2626",
                    reference_lines=reference_lines,
                    show_margin=bool(reference_kwargs["show_margin"]),
                    margin_at=str(reference_kwargs["margin_at"]),
                    custom_percentile=reference_kwargs.get("custom_percentile"),
                )
            else:
                fig, info = _plot_distribution_histogram(
                    _read_preacc(filename_use, "per_satellite_pfd_distribution/counts", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix),
                    edges,
                    title=title,
                    xlabel=xlabel,
                    color="#dc2626",
                    reference_lines=reference_lines,
                    show_margin=bool(reference_kwargs["show_margin"]),
                    margin_at=str(reference_kwargs["margin_at"]),
                    custom_percentile=reference_kwargs.get("custom_percentile"),
                )
        else:
            vals_db = _positive_db(
                _stack_across_iterations(filename_use, "PFD_per_sat_RAS_STATION_W_m2", system_index=_preacc_sys_idx)
            ) + float(view_context["db_offset_db"])
            if engine_use == ENGINE_PLOTLY:
                fig, info = _plotly_distribution_raw(
                    vals_db,
                    title=title,
                    xlabel=xlabel,
                    color="#dc2626",
                    hist_bins=int(params_use.get("raw_hist_bins", 4096)),
                    reference_lines=reference_lines,
                    show_margin=bool(reference_kwargs["show_margin"]),
                    margin_at=str(reference_kwargs["margin_at"]),
                    custom_percentile=reference_kwargs.get("custom_percentile"),
                )
            else:
                fig, info = _plot_distribution_raw(
                    vals_db,
                    title=title,
                    xlabel=xlabel,
                    color="#dc2626",
                    hist_bins=int(params_use.get("raw_hist_bins", 4096)),
                    reference_lines=reference_lines,
                    show_margin=bool(reference_kwargs["show_margin"]),
                    margin_at=str(reference_kwargs["margin_at"]),
                    custom_percentile=reference_kwargs.get("custom_percentile"),
                )
        info = dict(info or {})
        info["reference_lines"] = reference_lines
        info["bandwidth_view"] = dict(view_context)
        if view_context["warning_text"]:
            info["bandwidth_warning"] = str(view_context["warning_text"])
    elif recipe_id == "prx_elevation_heatmap":
        view_context = _resolve_bandwidth_view_context(
            filename_use,
            params=params_use,
            family_name="prx_elevation_heatmap" if source == SOURCE_PREACC else None,
            dataset_name="Prx_per_sat_RAS_STATION_W",
        )

        if source == SOURCE_PREACC:
            counts = _read_preacc(filename_use, "prx_elevation_heatmap/counts", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
            elev_edges = _read_preacc(filename_use, "prx_elevation_heatmap/elevation_edges_deg", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
            value_edges = np.asarray(
                _read_preacc(filename_use, "prx_elevation_heatmap/value_edges_dbw", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix),
                dtype=np.float64,
            ) + float(view_context["db_offset_db"])
        else:
            values = _stack_across_iterations(filename_use, "Prx_per_sat_RAS_STATION_W", system_index=_preacc_sys_idx)
            elev = _stack_across_iterations(filename_use, "sat_elevation_RAS_STATION_deg", system_index=_preacc_sys_idx)
            counts, elev_edges, value_edges = _prepare_raw_value_heatmap(
                values,
                elev,
                elevation_bin_step_deg=float(params_use.get("elevation_bin_step_deg", 1.0)),
                value_bin_step_db=float(params_use.get("value_bin_step_db", 0.25)),
                db_offset_db=float(view_context["db_offset_db"]),
            )
        if engine_use == ENGINE_PLOTLY:
            fig, info = _plotly_heatmap(
                counts,
                elevation_edges_deg=elev_edges,
                value_edges_db=value_edges,
                title=f"Prx vs elevation heatmap{_fmt_bw_parens(view_context['bandwidth_label'])}",
                ylabel=f"Instantaneous Prx contribution [dBW{_fmt_bw_suffix(view_context['bandwidth_label'])}]",
            )
        else:
            fig, info = _plot_generic_heatmap(
                counts,
                elevation_edges_deg=elev_edges,
                value_edges_db=value_edges,
                title=f"Prx vs elevation heatmap{_fmt_bw_parens(view_context['bandwidth_label'])}",
                ylabel=f"Instantaneous Prx contribution [dBW{_fmt_bw_suffix(view_context['bandwidth_label'])}]",
            )
        info = dict(info or {})
        info["bandwidth_view"] = dict(view_context)
        if view_context["warning_text"]:
            info["bandwidth_warning"] = str(view_context["warning_text"])
    elif recipe_id == "per_satellite_pfd_elevation_heatmap":
        view_context = _resolve_bandwidth_view_context(
            filename_use,
            params=params_use,
            family_name=(
                "per_satellite_pfd_elevation_heatmap" if source == SOURCE_PREACC else None
            ),
            dataset_name="PFD_per_sat_RAS_STATION_W_m2",
        )
        if source == SOURCE_PREACC:
            counts = _read_preacc(filename_use, "per_satellite_pfd_elevation_heatmap/counts", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
            elev_edges = _read_preacc(
                filename_use, "per_satellite_pfd_elevation_heatmap/elevation_edges_deg",
                system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix,
            )
            value_edges = np.asarray(
                _read_preacc(
                    filename_use, "per_satellite_pfd_elevation_heatmap/value_edges_dbw",
                    system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix,
                ),
                dtype=np.float64,
            ) + float(view_context["db_offset_db"])
        else:
            values = _stack_across_iterations(filename_use, "PFD_per_sat_RAS_STATION_W_m2", system_index=_preacc_sys_idx)
            elev = _stack_across_iterations(filename_use, "sat_elevation_RAS_STATION_deg", system_index=_preacc_sys_idx)
            counts, elev_edges, value_edges = _prepare_raw_value_heatmap(
                values,
                elev,
                elevation_bin_step_deg=float(params_use.get("elevation_bin_step_deg", 1.0)),
                value_bin_step_db=float(params_use.get("value_bin_step_db", 0.25)),
                db_offset_db=float(view_context["db_offset_db"]),
            )
        if engine_use == ENGINE_PLOTLY:
            fig, info = _plotly_heatmap(
                counts,
                elevation_edges_deg=elev_edges,
                value_edges_db=value_edges,
                title=f"Per-satellite PFD vs elevation heatmap{_fmt_bw_parens(view_context['bandwidth_label'])}",
                ylabel=f"Instantaneous PFD contribution [dBW/m\u00b2{_fmt_bw_suffix(view_context['bandwidth_label'])}]",
            )
        else:
            fig, info = visualise.plot_satellite_elevation_pfd_heatmap(
                counts,
                elevation_edges_deg=elev_edges,
                pfd_edges_db=value_edges,
                return_values=True,
                show=False,
            )
        info = dict(info or {})
        info["bandwidth_view"] = dict(view_context)
        if view_context["warning_text"]:
            info["bandwidth_warning"] = str(view_context["warning_text"])
    elif recipe_id in {
        "hemisphere_percentile_map",
        "hemisphere_data_loss_map",
        "hemisphere_percentile_map_3d",
        "hemisphere_data_loss_map_3d",
    }:
        fig, info = _render_hemisphere_recipe(
            filename_use,
            meta,
            recipe_id=recipe_id,
            params=params_use,
            primary_power_dataset=primary_power_dataset,
            engine=engine_use,
        )
    elif recipe_id == "beam_count_full_network_ccdf":
        reference_lines, reference_kwargs = _reference_plot_kwargs(params_use)
        if source == SOURCE_PREACC:
            counts = _read_preacc(filename_use, "beam_statistics/full_network_count_hist", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
            edges = _read_preacc(filename_use, "beam_statistics/count_edges", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
        else:
            counts = _stack_across_iterations(filename_use, "sat_beam_counts_used", system_index=_preacc_sys_idx)
            hist, edges = _beam_hist_from_samples(_beam_counts_samples(counts))
            counts = hist
        if engine_use == ENGINE_PLOTLY:
            fig, info = _plotly_distribution_histogram(
                counts,
                edges,
                title="Full-network satellite beam-count CCDF",
                xlabel="Used beams per satellite",
                color="#0f766e",
                reference_lines=reference_lines,
                show_margin=bool(reference_kwargs["show_margin"]),
                margin_at=str(reference_kwargs["margin_at"]),
                custom_percentile=reference_kwargs.get("custom_percentile"),
            )
        else:
            fig, info = _plot_distribution_histogram(
                counts,
                edges,
                title="Full-network satellite beam-count CCDF",
                xlabel="Used beams per satellite",
                color="#0f766e",
                reference_lines=reference_lines,
                show_margin=bool(reference_kwargs["show_margin"]),
                margin_at=str(reference_kwargs["margin_at"]),
                custom_percentile=reference_kwargs.get("custom_percentile"),
            )
        info = dict(info or {})
        info["reference_lines"] = reference_lines
    elif recipe_id == "beam_count_visible_ccdf":
        reference_lines, reference_kwargs = _reference_plot_kwargs(params_use)
        if source == SOURCE_PREACC:
            counts = _read_preacc(filename_use, "beam_statistics/visible_count_hist", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
            edges = _read_preacc(filename_use, "beam_statistics/count_edges", system_index=_preacc_sys_idx, group_prefix=_preacc_group_prefix)
        else:
            counts = _stack_across_iterations(filename_use, "sat_beam_counts_used", system_index=_preacc_sys_idx)
            elev = _stack_across_iterations(filename_use, "sat_elevation_RAS_STATION_deg", system_index=_preacc_sys_idx)
            vis_mask = np.asarray(elev, dtype=np.float64) > 0.0
            hist, edges = _beam_hist_from_visible_samples(counts, vis_mask)
            counts = hist
        if engine_use == ENGINE_PLOTLY:
            fig, info = _plotly_distribution_histogram(
                counts,
                edges,
                title="RAS-visible satellite beam-count CCDF",
                xlabel="Used beams per satellite",
                color="#2563eb",
                reference_lines=reference_lines,
                show_margin=bool(reference_kwargs["show_margin"]),
                margin_at=str(reference_kwargs["margin_at"]),
                custom_percentile=reference_kwargs.get("custom_percentile"),
            )
        else:
            fig, info = _plot_distribution_histogram(
                counts,
                edges,
                title="RAS-visible satellite beam-count CCDF",
                xlabel="Used beams per satellite",
                color="#2563eb",
                reference_lines=reference_lines,
                show_margin=bool(reference_kwargs["show_margin"]),
                margin_at=str(reference_kwargs["margin_at"]),
                custom_percentile=reference_kwargs.get("custom_percentile"),
            )
        info = dict(info or {})
        info["reference_lines"] = reference_lines
    elif recipe_id in {
        "beam_count_total_over_time",
        "beam_count_visible_over_time",
        "beam_demand_over_time",
    }:
        fig, info = _render_beam_time_series_recipe(
            filename_use,
            recipe_id=recipe_id,
            source=source,
            params=params_use,
            engine=engine_use,
            system_index=_preacc_sys_idx,
        )
    elif recipe_id == "beam_overview_over_time":
        fig, info = _render_beam_overview_recipe(
            filename_use,
            source=source,
            params=params_use,
            engine=engine_use,
            system_index=_preacc_sys_idx,
        )
    elif recipe_id == "beam_cap_sizing_analysis":
        fig, info = _render_beam_cap_sizing_analysis(
            filename_use,
            params=params_use,
            engine=engine_use,
            system_index=_preacc_sys_idx,
        )
    elif recipe_id == "total_pfd_over_time":
        _pfd_group_prefix = f"system_{int(_preacc_sys_idx)}/" if _preacc_sys_idx is not None else ""
        view_context = _resolve_bandwidth_view_context(
            filename_use,
            params=params_use,
            dataset_name="PFD_total_RAS_STATION_W_m2",
        )
        x_segments = _relative_iteration_time_segments(filename_use, group_prefix=_pfd_group_prefix)
        y_segments = []
        for ii in _iteration_ids(filename_use, group_prefix=_pfd_group_prefix):
            y_segments.append(
                np.asarray(
                    _read_iter_dataset(filename_use, "PFD_total_RAS_STATION_W_m2", ii, group_prefix=_pfd_group_prefix),
                    dtype=np.float64,
                ).reshape(-1)
            )
        x, y = _join_series_segments(x_segments, y_segments)
        y_db = np.full_like(y, np.nan, dtype=np.float64)
        mask = np.isfinite(y) & (y > 0.0)
        y_db[mask] = 10.0 * np.log10(y[mask]) + float(view_context["db_offset_db"])
        if engine_use == ENGINE_PLOTLY:
            fig, info = _plotly_series(
                x,
                y_db,
                title=f"Total PFD over time{_fmt_bw_parens(view_context['bandwidth_label'])}",
                ylabel=f"Total PFD [dBW/m\u00b2{_fmt_bw_suffix(view_context['bandwidth_label'])}]",
                color="#7c3aed",
            )
        else:
            fig, info = _plot_series(
                x,
                y_db,
                title=f"Total PFD over time{_fmt_bw_parens(view_context['bandwidth_label'])}",
                ylabel=f"Total PFD [dBW/m\u00b2{_fmt_bw_suffix(view_context['bandwidth_label'])}]",
                color="#7c3aed",
            )
        info = dict(info or {})
        info["bandwidth_view"] = dict(view_context)
        if view_context["warning_text"]:
            info["bandwidth_warning"] = str(view_context["warning_text"])
    else:
        raise KeyError(recipe_id)

    info = dict(info or {})
    info["recipe_id"] = recipe_id
    info["recipe_label"] = str(info.get("recipe_label") or recipe.label)
    info["source_used"] = source
    info["filename"] = filename_use
    info["params"] = params_use
    info["engine_used"] = engine_use
    info["capability"] = recipe_capability(
        meta,
        recipe_id,
        params=params_use,
        filename=filename_use,
        primary_power_dataset=primary_power_dataset,
    )
    return fig, info


__all__ = [
    "PostprocessRecipe",
    "RecipeParameter",
    "RECIPES",
    "RECIPE_BY_ID",
    "SOURCE_AUTO",
    "SOURCE_RAW",
    "SOURCE_PREACC",
    "SOURCE_MODES",
    "default_recipe_parameters",
    "inspect_result_file",
    "normalize_recipe_parameters",
    "recipe_availability",
    "recipe_capability",
    "render_recipe",
    "resolve_recipe_source",
    "resolve_recipe_parameter_state",
]
