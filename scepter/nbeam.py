"""Streaming beam-cap sizing with visibility-aware pooling policies."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import plotly.graph_objects as go
try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

from scepter import satsim, scenario
try:
    import numba as nb
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    nb = None
    _HAVE_NUMBA = False

try:
    from scepter.angle_sampler import _skycell_id_s1586, _S1586_N_CELLS
except Exception:  # pragma: no cover
    _skycell_id_s1586 = None
    _S1586_N_CELLS = 2334

try:
    from scepter.visualise import plot_hemisphere_2D, plot_hemisphere_3D
    _HAVE_VISUALISE = True
except Exception:  # pragma: no cover
    plot_hemisphere_2D = None
    plot_hemisphere_3D = None
    _HAVE_VISUALISE = False

S1586_N_CELLS_DEFAULT = int(_S1586_N_CELLS) if _S1586_N_CELLS is not None else 2334
STRICT_GROUP_OFFSET = 10_000_000
FALLBACK_GROUP_OFFSET = 20_000_000
TEMPORAL_GROUP_OFFSET = 30_000_000
_ACTIVE_PROGRESS_BAR_COUNT = 0
_EMIT_PROGRESS_OUTPUT = True


@dataclass(slots=True)
class BeamCapSizingConfig:
    """
    Runtime configuration for the reusable beam-cap sizing pipeline.

    Parameters
    ----------
    max_demand_slots : int or None, optional
        Optional hard stop on processed demand slots. ``None`` scans the full
        file.
    count_var_candidates : tuple[str, ...], optional
        Ordered count-dataset fallback list for legacy count-based policies.
        The first dataset present in every ``/iter/iter_*`` group is used.
    skycell_mode : {"s1586"}, optional
        Receive-side sky-cell mapping used by visibility-aware policies.
    s1586_n_cells : int, optional
        Expected S.1586 sky-cell count for ``skycell_mode="s1586"``.
    beltsky_min_active_sats : int, optional
        Minimum active satellites required before the belt+sky fallback policy
        keeps a strict sky-cell pool instead of falling back to a coarser pool.
    enable_temporal_sky_compat : bool, optional
        Enable the temporal sky-compatibility prepass used by the temporal
        belt+sky policy.
    temporal_window_s : float, optional
        Temporal compatibility window, in seconds.
    temporal_min_edge_count : int, optional
        Minimum repeated co-occurrence count required to keep a temporal edge.
    temporal_prepass_slot_stride : int, optional
        Slot down-sampling factor for the temporal prepass. ``1`` uses every
        slot.
    temporal_prepass_max_demand_slots : int, optional
        Optional limit on demand slots scanned during the temporal prepass.
        ``0`` disables the cap.
    temporal_numba_min_edge_work : int, optional
        Work threshold above which the temporal prepass prefers its Numba path.
    lost_demand_target : float, optional
        SLA target for unmet demand fraction ``delta(B)``.
    loss_slot_target : float, optional
        SLA target for loss-slot probability ``eps(B)``.
    per_slot_loss_tolerance : float, optional
        Tail-risk tolerance used when plotting and selecting beam caps.
    beam_cap_min, beam_cap_max : int, optional
        Inclusive beam-cap range. The evaluated grid is
        ``np.arange(beam_cap_min, beam_cap_max + 1)``.
    read_slot_chunk : int, optional
        Number of time slots streamed from HDF5 per read chunk. This is the
        hard upper bound for each ``pure_reroute`` eligibility read in both
        the auto-backend probe and the main streaming loop. Dense files use
        ``sat_eligible_mask`` while sparse files use the CSR payload
        ``sat_eligible_csr_row_ptr`` / ``sat_eligible_csr_sat_idx``.
    progress_every_demand_slots : int, optional
        Reporting cadence, in processed demand slots, for tqdm postfix refresh
        and plain text fallback progress updates.
    emit_progress_output : bool, optional
        Enable tqdm / plain-text runtime progress output. Disable this in GUI
        paths that should stay quiet while still returning final diagnostics.
    enabled_policy_keys : tuple[str, ...], optional
        Enabled beam-cap curves. Supported keys are ``"simpson"``,
        ``"full_reroute"``, ``"pure_reroute"``, ``"belt"``,
        ``"belt_sky_strict"``, ``"belt_sky_temporal"``, ``"belt_sky_fb"``,
        and ``"no_reroute"``.
    nco : int, optional
        Maximum number of concurrent links requested per cell. This is a real
        input parameter and is required whenever ``pure_reroute`` is enabled.
    pure_reroute_backend : {"auto", "cpu", "gpu"}, optional
        Backend selection for the exact ``pure_reroute`` policy. ``"auto"``
        benchmarks a representative active-slot buffer and keeps the faster
        exact backend for the rest of the run. ``"gpu"`` requires GPU support
        plus CuPy/NVRTC access to matching CUDA toolkit headers.
        ``"cpu"`` forces the CPU exact solver.
    save_outputs : bool, optional
        Save plots and summary artifacts to disk.
    output_dir : str or pathlib.Path, optional
        Root output directory for saved parser artifacts.
    output_prefix : str, optional
        Prefix used when naming timestamped output run directories.
    save_plots_html, save_results_json, save_results_npz : bool, optional
        Individual output toggles for HTML plots, JSON summaries, and NPZ
        curve exports.
    save_interim_outputs : bool, optional
        Enable stable overwrite-only interim snapshots in the run directory.
        Interim cadence is counted in raw scanned timestep slots, not demand
        slots.
    interim_every_slots : int, optional
        Raw-slot cadence for interim snapshots. With chunked scanning, saves
        are chunk-aligned: a snapshot is written after the chunk that first
        reaches or crosses this cadence threshold.
    interim_save_html : bool, optional
        Save interim beam-cap plots to HTML using stable ``*.interim.html``
        filenames that are overwritten in place.
    interim_save_json : bool, optional
        Save an interim JSON summary to ``results_summary.interim.json``.
    interim_include_skycell_vis : bool, optional
        Also save interim skycell-demand visuals. This is disabled by default
        because those visuals are heavier than the beam-cap plots.
    show_plots : bool, optional
        Show generated plots interactively.
    enable_skycell_demand_vis : bool, optional
        Enable demand heatmaps on the receive-side sky grid.
    skycell_demand_vis_engine : {"plotly", "pyvista"}, optional
        Visualization backend for sky-cell demand plots.
    skycell_demand_vis_show : bool, optional
        Show sky-cell demand plots interactively.
    skycell_demand_vis_save_html : bool, optional
        Save sky-cell demand plots to HTML where supported.
    skycell_demand_vis_2d_projection : {"polar", "aitoff"}, optional
        2D projection used for the sky-cell demand plot.
    skycell_demand_normalize_mode : {"sum", "slots"}, optional
        Demand normalization basis for sky-cell demand plots. ``"sum"``
        normalizes by total demand weight. ``"slots"`` normalizes by slot hits.
    skycell_demand_normalize_to_percent : bool, optional
        Express normalized sky-cell demand as percent instead of fraction.
    store_slot_ratio_samples : bool, optional
        Retain per-slot samples for additional diagnostics and exports.
    ccdf_y_floor_percent, sla_y_floor_percent : float, optional
        Lower y-axis floors, in percent, for tail-risk and SLA plots.
    focus_at : float, optional
        Auto-zoom focus location along the x-axis for the selected beam cap.
    ccdf_tail_guard_percent : float, optional
        Tail guard threshold, in percent, used when choosing the CCDF x-range.
    timestep_s : float, optional
        Optional simulated seconds per retained slot, used for reporting only.

    Notes
    -----
    The pipeline supports both legacy count-based policies and the exact
    ``pure_reroute`` lower-bound policy derived from either the legacy dense
    ``sat_eligible_mask`` dataset or the preferred sparse CSR eligibility
    payload. ``pure_reroute`` is global across all eligible satellites,
    including multi-belt files, because belt-specific filtering is assumed to
    already be reflected in the stored eligibility input. ``READ_SLOT_CHUNK``
    remains the single low-memory control for ``pure_reroute``; even
    ``pure_reroute_backend="auto"`` keeps its bounded probe within that slot
    limit.
    The shared output glossary uses these definitions throughout saved JSON,
    plots, wrappers, and notebooks: ``delta(B)`` is total unserved demand
    divided by total processed demand, ``epsilon(B)`` / ``eps(B)`` is the
    fraction of processed demand slots that still fail full service at beam cap
    ``B``, and tail risk is ``P(B_req > B)``, the fraction of processed slots
    whose required beam cap exceeds the tested ``B``. Runtime progress uses
    ``tqdm.auto`` when available and falls back to plain text otherwise.
    """

    max_demand_slots: int | None = None
    count_var_candidates: tuple[str, ...] = (
        "sat_beam_counts_used",
        "sat_beam_counts_eligible",
        "sat_beam_counts_demand",
        "sat_beam_counts",
    )
    skycell_mode: str = "s1586"
    s1586_n_cells: int = S1586_N_CELLS_DEFAULT
    beltsky_min_active_sats: int = 2
    enable_temporal_sky_compat: bool = True
    temporal_window_s: float = 5.0
    temporal_min_edge_count: int = 2
    temporal_prepass_slot_stride: int = 1
    temporal_prepass_max_demand_slots: int = 0
    temporal_numba_min_edge_work: int = 64
    lost_demand_target: float = 1e-3
    loss_slot_target: float = 1e-2
    per_slot_loss_tolerance: float = 1e-3
    beam_cap_min: int = 0
    beam_cap_max: int = 260
    read_slot_chunk: int = 256
    progress_every_demand_slots: int = 100
    emit_progress_output: bool = True
    enabled_policy_keys: tuple[str, ...] = (
        "full_reroute",
        "simpson",
        "belt_sky_temporal",
        "no_reroute",
    )
    nco: int = 1
    pure_reroute_backend: str = "auto"
    save_outputs: bool = True
    output_dir: str | Path = "nbeam_parser"
    output_prefix: str = "nbeam_parser"
    save_plots_html: bool = True
    save_results_json: bool = True
    save_results_npz: bool = True
    save_interim_outputs: bool = False
    interim_every_slots: int = 500
    interim_save_html: bool = True
    interim_save_json: bool = True
    interim_include_skycell_vis: bool = False
    show_plots: bool = True
    enable_skycell_demand_vis: bool = True
    skycell_demand_vis_engine: str = "plotly"
    skycell_demand_vis_show: bool = False
    skycell_demand_vis_save_html: bool = True
    skycell_demand_vis_2d_projection: str = "polar"
    skycell_demand_normalize_mode: str = "sum"
    skycell_demand_normalize_to_percent: bool = True
    store_slot_ratio_samples: bool = False
    ccdf_y_floor_percent: float = 1e-3
    sla_y_floor_percent: float = 1e-4
    focus_at: float = 0.75
    ccdf_tail_guard_percent: float = 1e-2
    timestep_s: float = np.nan


def _resolve_config(
    base_config: BeamCapSizingConfig | None,
    overrides: dict[str, Any],
) -> BeamCapSizingConfig:
    cfg = replace(base_config) if base_config is not None else BeamCapSizingConfig()
    if not overrides:
        return cfg
    valid_fields = {field.name for field in fields(BeamCapSizingConfig)}
    unknown = [key for key in overrides if key not in valid_fields]
    if unknown:
        valid = ", ".join(sorted(valid_fields))
        unknown_txt = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown config override(s): {unknown_txt}. Valid keys: {valid}")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _apply_config_globals(
    cfg: BeamCapSizingConfig,
    *,
    storage_filename: str | Path,
) -> None:
    global STORAGE_FILENAME
    global MAX_DEMAND_SLOTS
    global COUNT_VAR_CANDIDATES
    global SKYCELL_MODE
    global S1586_N_CELLS
    global BELTSKY_MIN_ACTIVE_SATS
    global ENABLE_TEMPORAL_SKY_COMPAT
    global TEMPORAL_WINDOW_S
    global TEMPORAL_MIN_EDGE_COUNT
    global TEMPORAL_PREPASS_SLOT_STRIDE
    global TEMPORAL_PREPASS_MAX_DEMAND_SLOTS
    global TEMPORAL_NUMBA_MIN_EDGE_WORK
    global lost_demand_target
    global loss_slot_target
    global per_slot_loss_tolerance
    global beam_cap_min
    global beam_cap_max
    global beam_caps
    global READ_SLOT_CHUNK
    global PROGRESS_EVERY_DEMAND_SLOTS
    global _EMIT_PROGRESS_OUTPUT
    global ENABLED_POLICY_KEYS
    global SAVE_OUTPUTS
    global OUTPUT_DIR
    global OUTPUT_PREFIX
    global SAVE_PLOTS_HTML
    global SAVE_RESULTS_JSON
    global SAVE_RESULTS_NPZ
    global SAVE_INTERIM_OUTPUTS
    global INTERIM_EVERY_SLOTS
    global INTERIM_SAVE_HTML
    global INTERIM_SAVE_JSON
    global INTERIM_INCLUDE_SKYCELL_VIS
    global SHOW_PLOTS
    global ENABLE_SKYCELL_DEMAND_VIS
    global SKYCELL_DEMAND_VIS_ENGINE
    global SKYCELL_DEMAND_VIS_SHOW
    global SKYCELL_DEMAND_VIS_SAVE_HTML
    global SKYCELL_DEMAND_VIS_2D_PROJECTION
    global SKYCELL_DEMAND_NORMALIZE_MODE
    global SKYCELL_DEMAND_NORMALIZE_TO_PERCENT
    global STORE_SLOT_RATIO_SAMPLES
    global CCDF_Y_FLOOR_PERCENT
    global SLA_Y_FLOOR_PERCENT
    global FOCUS_AT
    global CCDF_TAIL_GUARD_PERCENT
    global _timestep_s

    STORAGE_FILENAME = str(storage_filename)
    MAX_DEMAND_SLOTS = cfg.max_demand_slots
    COUNT_VAR_CANDIDATES = tuple(cfg.count_var_candidates)
    SKYCELL_MODE = str(cfg.skycell_mode)
    S1586_N_CELLS = int(cfg.s1586_n_cells)
    BELTSKY_MIN_ACTIVE_SATS = int(cfg.beltsky_min_active_sats)
    ENABLE_TEMPORAL_SKY_COMPAT = bool(cfg.enable_temporal_sky_compat)
    TEMPORAL_WINDOW_S = float(cfg.temporal_window_s)
    TEMPORAL_MIN_EDGE_COUNT = int(cfg.temporal_min_edge_count)
    TEMPORAL_PREPASS_SLOT_STRIDE = int(cfg.temporal_prepass_slot_stride)
    TEMPORAL_PREPASS_MAX_DEMAND_SLOTS = int(cfg.temporal_prepass_max_demand_slots)
    TEMPORAL_NUMBA_MIN_EDGE_WORK = int(cfg.temporal_numba_min_edge_work)
    lost_demand_target = float(cfg.lost_demand_target)
    loss_slot_target = float(cfg.loss_slot_target)
    per_slot_loss_tolerance = float(cfg.per_slot_loss_tolerance)
    beam_cap_min = int(cfg.beam_cap_min)
    beam_cap_max = int(cfg.beam_cap_max)
    beam_caps = np.arange(beam_cap_min, beam_cap_max + 1, dtype=np.int32)
    READ_SLOT_CHUNK = int(cfg.read_slot_chunk)
    PROGRESS_EVERY_DEMAND_SLOTS = int(cfg.progress_every_demand_slots)
    _EMIT_PROGRESS_OUTPUT = bool(cfg.emit_progress_output)
    ENABLED_POLICY_KEYS = tuple(cfg.enabled_policy_keys)
    SAVE_OUTPUTS = bool(cfg.save_outputs)
    OUTPUT_DIR = str(cfg.output_dir)
    OUTPUT_PREFIX = str(cfg.output_prefix)
    SAVE_PLOTS_HTML = bool(cfg.save_plots_html)
    SAVE_RESULTS_JSON = bool(cfg.save_results_json)
    SAVE_RESULTS_NPZ = bool(cfg.save_results_npz)
    SAVE_INTERIM_OUTPUTS = bool(cfg.save_interim_outputs)
    INTERIM_EVERY_SLOTS = _normalize_positive_int(cfg.interim_every_slots, name="interim_every_slots")
    INTERIM_SAVE_HTML = bool(cfg.interim_save_html)
    INTERIM_SAVE_JSON = bool(cfg.interim_save_json)
    INTERIM_INCLUDE_SKYCELL_VIS = bool(cfg.interim_include_skycell_vis)
    SHOW_PLOTS = bool(cfg.show_plots)
    ENABLE_SKYCELL_DEMAND_VIS = bool(cfg.enable_skycell_demand_vis)
    SKYCELL_DEMAND_VIS_ENGINE = str(cfg.skycell_demand_vis_engine)
    SKYCELL_DEMAND_VIS_SHOW = bool(cfg.skycell_demand_vis_show)
    SKYCELL_DEMAND_VIS_SAVE_HTML = bool(cfg.skycell_demand_vis_save_html)
    SKYCELL_DEMAND_VIS_2D_PROJECTION = str(cfg.skycell_demand_vis_2d_projection)
    SKYCELL_DEMAND_NORMALIZE_MODE = str(cfg.skycell_demand_normalize_mode)
    SKYCELL_DEMAND_NORMALIZE_TO_PERCENT = bool(cfg.skycell_demand_normalize_to_percent)
    STORE_SLOT_RATIO_SAMPLES = bool(cfg.store_slot_ratio_samples)
    CCDF_Y_FLOOR_PERCENT = float(cfg.ccdf_y_floor_percent)
    SLA_Y_FLOOR_PERCENT = float(cfg.sla_y_floor_percent)
    FOCUS_AT = float(cfg.focus_at)
    CCDF_TAIL_GUARD_PERCENT = float(cfg.ccdf_tail_guard_percent)
    _timestep_s = float(cfg.timestep_s)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def style_axes_with_grid(fig: go.Figure) -> None:
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True,
                     gridcolor="rgba(0,0,0,0.15)", gridwidth=1, automargin=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True,
                     gridcolor="rgba(0,0,0,0.15)", gridwidth=1, automargin=True)


def place_legend_below(fig: go.Figure, n_rows: int = 2) -> None:
    if n_rows <= 1:
        y = -0.20
        b = 180
    elif n_rows == 2:
        y = -0.30
        b = 260
    else:
        extra = max(0, int(n_rows) - 3)
        y = -0.42 - 0.08 * float(extra)
        b = 360 + 70 * int(extra)
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="top", y=y, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.92)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
            font=dict(size=14), tracegroupgap=8,
        ),
        margin=dict(b=b),
    )


def add_value_box(fig: go.Figure, lines: list[str], x: float = 1.02, y: float = 0.98) -> None:
    fig.add_annotation(
        xref="paper", yref="paper", x=x, y=y, text="<br>".join(lines), showarrow=False,
        align="left", xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.94)", bordercolor="rgba(0,0,0,0.25)",
        borderwidth=1, borderpad=8, font=dict(size=13, family="Courier New"),
    )


def apply_log_y(fig: go.Figure, y_floor_percent: float, y_max_percent: float, title: str) -> None:
    y0 = max(float(y_floor_percent), 1e-12)
    y1 = max(float(y_max_percent), y0 * 10.0)
    fig.update_yaxes(type="log", range=[np.log10(y0), np.log10(y1)], title_text=title)


def policy_display_name(key: str, label: str) -> str:
    short = {
        "simpson": "Simpson",
        "full_reroute": "Full reroute",
        "pure_reroute": "Pure reroute",
        "belt": "Belt only",
        "belt_sky_strict": "Belt+sky strict",
        "belt_sky_temporal": "Belt+sky temporal",
        "belt_sky_fb": "Belt+sky fallback",
        "no_reroute": "No reroute",
    }
    return short.get(str(key), str(label))


_DELTA_GLOSSARY_TEXT = "delta(B) = total unserved demand divided by total processed demand"
_EPS_GLOSSARY_TEXT = (
    "epsilon(B) / eps(B) = fraction of processed demand slots that still fail full service"
)
_TAIL_RISK_GLOSSARY_TEXT = (
    "tail risk = P(B_req > B), the fraction of processed slots whose required beam cap exceeds B"
)


def _metric_glossary_payload() -> dict[str, dict[str, Any]]:
    return {
        "delta": {
            "symbol": "delta(B)",
            "definition": _DELTA_GLOSSARY_TEXT,
            "json_values": "fraction",
            "plot_values": "percent",
        },
        "epsilon": {
            "symbol": "epsilon(B)",
            "aliases": ["eps(B)"],
            "definition": _EPS_GLOSSARY_TEXT,
            "json_values": "fraction",
            "plot_values": "percent",
        },
        "tail_risk": {
            "symbol": "P(B_req > B)",
            "definition": _TAIL_RISK_GLOSSARY_TEXT,
            "json_values": "percent",
            "plot_values": "percent",
        },
    }


def build_policy_value_lines(evals: list[PolicyEval]) -> list[str]:
    lines = [
        "<b>Selected beam-cap summary</b>",
        "delta = unserved demand fraction",
        "epsilon = failed-demand-slot fraction",
        "Policy            B   Delta    Epsilon",
    ]
    for pe in evals:
        i = pe.selected_idx
        name = policy_display_name(pe.key, pe.label)
        lines.append(f"{name:15s} {pe.selected_b:>3d}  {pe.delta_run[i]:>8.2e}  {pe.eps_run[i]:>8.2e}")
    return lines


def compute_xmax_for_focus(b_focus: int, focus_at: float = 0.75, x_min: int = 0,
                           x_max_cap: int | None = None, extra_right_margin: int = 0, min_span: int = 10) -> int:
    if not (0.05 < float(focus_at) < 0.95):
        raise ValueError("focus_at must be in (0.05, 0.95).")
    xmax = int(np.ceil(max(int(b_focus), 0) / float(focus_at))) + int(extra_right_margin)
    xmax = max(xmax, int(x_min) + int(min_span))
    return min(xmax, int(x_max_cap)) if x_max_cap is not None else xmax


def ccdf_tail_guard_xmax(x_grid: np.ndarray, ccdf_series_percent: list[np.ndarray], tail_guard_percent: float,
                         fallback_xmax: int, x_max_cap: int | None = None) -> int:
    x = np.asarray(x_grid, dtype=np.int32)
    if x.size == 0:
        return int(fallback_xmax)
    below_all = np.ones_like(x, dtype=bool)
    for c in ccdf_series_percent:
        below_all &= (np.asarray(c, dtype=np.float64) < float(tail_guard_percent))
    xmax = int(fallback_xmax) if not np.any(below_all) else max(int(fallback_xmax), int(x[int(np.argmax(below_all))]))
    return min(xmax, int(x_max_cap)) if x_max_cap is not None else xmax


def choose_smallest_beam_cap(beam_caps_grid: np.ndarray, delta_run: np.ndarray, eps_run: np.ndarray,
                             delta_target: float, eps_target: float) -> int:
    ok = (delta_run <= float(delta_target)) & (eps_run <= float(eps_target))
    return int(beam_caps_grid[np.argmax(ok)]) if np.any(ok) else int(beam_caps_grid[-1])


def _normalize_optional_positive_int(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    v = int(value)
    if v <= 0:
        raise ValueError(f"{name} must be None or a positive integer.")
    return v


def _normalize_positive_int(value: int, *, name: str) -> int:
    v = int(value)
    if v <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return v


def _resolve_pure_reroute_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("pure_reroute_backend must be one of {'auto', 'cpu', 'gpu'}.")
    if backend == "gpu" and not _pure_reroute_gpu_backend_available():
        raise RuntimeError(
            "pure_reroute_backend='gpu' requires an available CUDA device and CuPy support."
        )
    return backend


@lru_cache(maxsize=1)
def _load_tqdm():
    """Return `tqdm.auto.tqdm` when available, otherwise ``None``."""
    try:
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        return None


def _progress_write(message: str) -> None:
    """Emit a progress-safe status line, using ``tqdm.write`` when needed."""
    if not _EMIT_PROGRESS_OUTPUT:
        return
    tqdm = _load_tqdm()
    if _ACTIVE_PROGRESS_BAR_COUNT > 0 and tqdm is not None:
        tqdm.write(str(message))
    else:
        print(message)


def _current_process_rss_bytes() -> int | None:
    """Return current process RSS in bytes when psutil is available."""
    if psutil is None:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _format_gib_text(n_bytes: int | None) -> str:
    """Format a byte count as a compact GiB string."""
    if n_bytes is None:
        return "n/a"
    return f"{float(n_bytes) / float(1024 ** 3):.2f} GiB"


def _rss_gib_postfix() -> str | None:
    """Return current RSS formatted for progress postfixes."""
    rss_bytes = _current_process_rss_bytes()
    if rss_bytes is None:
        return None
    return f"{float(rss_bytes) / float(1024 ** 3):.2f}"


class _StageProgress:
    """Optional tqdm-backed stage progress with plain-text fallback."""

    def __init__(
        self,
        *,
        desc: str,
        total: int | None,
        unit: str,
        report_every: int,
    ) -> None:
        self.desc = str(desc)
        self.total = None if total is None else max(0, int(total))
        self.unit = str(unit)
        self.report_every = max(1, int(report_every))
        self._next_report = int(self.report_every)
        self._value = 0
        self._postfix: dict[str, Any] = {}
        self._bar: Any | None = None

        if not _EMIT_PROGRESS_OUTPUT:
            return

        tqdm = _load_tqdm()
        if tqdm is not None:
            global _ACTIVE_PROGRESS_BAR_COUNT
            kwargs: dict[str, Any] = {
                "desc": self.desc,
                "unit": self.unit,
                "leave": False,
                "dynamic_ncols": True,
            }
            if self.total is not None:
                kwargs["total"] = int(self.total)
            self._bar = tqdm(**kwargs)
            _ACTIVE_PROGRESS_BAR_COUNT += 1

    def update(self, delta: int) -> None:
        delta_i = max(0, int(delta))
        if delta_i <= 0:
            return
        self._value += delta_i
        if self._bar is not None:
            self._bar.update(delta_i)

    def set_postfix(self, values: dict[str, Any]) -> None:
        self._postfix = {str(key): value for key, value in values.items() if value is not None}
        if self._bar is not None:
            self._bar.set_postfix(self._postfix, refresh=False)

    def maybe_refresh(self, *, counter: int | None = None, force: bool = False) -> None:
        counter_i = self._value if counter is None else max(0, int(counter))
        if self._bar is not None:
            if force or counter_i >= self._next_report:
                if self._postfix:
                    self._bar.set_postfix(self._postfix, refresh=True)
                elif hasattr(self._bar, "refresh"):
                    self._bar.refresh()
                while self._next_report <= counter_i:
                    self._next_report += self.report_every
            return

        if not (force or counter_i >= self._next_report):
            return

        total_txt = "" if self.total is None else f"/{self.total:,}"
        postfix_txt = ""
        if self._postfix:
            postfix_txt = " | " + ", ".join(f"{key}={value}" for key, value in self._postfix.items())
        _progress_write(f"{self.desc}: {self._value:,}{total_txt} {self.unit}{postfix_txt}")
        while self._next_report <= counter_i:
            self._next_report += self.report_every

    def close(self) -> None:
        if self._bar is not None:
            global _ACTIVE_PROGRESS_BAR_COUNT
            self._bar.close()
            _ACTIVE_PROGRESS_BAR_COUNT = max(0, _ACTIVE_PROGRESS_BAR_COUNT - 1)
            self._bar = None


def _dataset_layout_summary(name: str, ds: h5py.Dataset) -> str:
    """Return a compact one-line summary of dataset layout."""
    shape_txt = tuple(int(v) for v in ds.shape)
    shuffle = bool(getattr(ds, "shuffle", False))
    return (
        f"{name}: shape={shape_txt}, dtype={ds.dtype}, chunks={ds.chunks}, "
        f"compression={ds.compression}, shuffle={shuffle}"
    )


def _dataset_streaming_warnings(
    name: str,
    ds: h5py.Dataset,
    *,
    read_slot_chunk: int,
) -> list[str]:
    """Return human-readable warnings for obviously unfriendly slot streaming."""
    warnings: list[str] = []
    if ds.ndim < 1:
        return warnings
    chunks = ds.chunks
    if chunks is None:
        warnings.append(
            f"{name} is stored contiguously; slot-local reads may still touch large on-disk spans."
        )
        return warnings
    if int(chunks[0]) > int(read_slot_chunk):
        warnings.append(
            f"{name} leading HDF5 chunk spans {int(chunks[0]):,} slots, larger than "
            f"READ_SLOT_CHUNK={int(read_slot_chunk):,}."
        )
    return warnings


def _estimate_pure_reroute_chunk_bytes(
    ds_eligible: h5py.Dataset | None,
    *,
    slot_count: int,
    ds_row_ptr: h5py.Dataset | None = None,
    ds_sat_idx: h5py.Dataset | None = None,
) -> int:
    """Estimate raw bytes for one eligibility-mask chunk."""
    if ds_eligible is not None:
        if ds_eligible.ndim < 3:
            return 0
        return (
            max(0, int(slot_count))
            * int(np.prod(ds_eligible.shape[1:], dtype=np.int64))
            * int(np.dtype(ds_eligible.dtype).itemsize)
        )
    if ds_row_ptr is None or ds_sat_idx is None:
        return 0
    if (
        _PURE_REROUTE_CSR_TIME_COUNT_ATTR not in ds_row_ptr.attrs
        or _PURE_REROUTE_CSR_CELL_COUNT_ATTR not in ds_row_ptr.attrs
    ):
        return 0
    time_count = int(ds_row_ptr.attrs[_PURE_REROUTE_CSR_TIME_COUNT_ATTR])
    cell_count = int(ds_row_ptr.attrs[_PURE_REROUTE_CSR_CELL_COUNT_ATTR])
    total_rows = max(1, time_count * cell_count)
    total_edges = int(ds_row_ptr[-1]) if int(ds_row_ptr.shape[0]) > 0 else 0
    mean_edges_per_row = float(total_edges) / float(total_rows)
    row_bytes = (max(0, int(slot_count)) * cell_count + 1) * int(np.dtype(ds_row_ptr.dtype).itemsize)
    edge_bytes = int(max(0.0, mean_edges_per_row * float(slot_count * cell_count))) * int(np.dtype(ds_sat_idx.dtype).itemsize)
    return int(row_bytes + edge_bytes)


def _pure_reroute_dataset_diagnostics(
    ds_eligible: h5py.Dataset | None = None,
    *,
    read_slot_chunk: int,
    max_demand_slots: int | None,
    ds_row_ptr: h5py.Dataset | None = None,
    ds_sat_idx: h5py.Dataset | None = None,
) -> dict[str, Any]:
    """Summarize low-memory diagnostics for the pure-reroute eligibility mask."""
    probe_slots = min(int(read_slot_chunk), int(_PURE_REROUTE_AUTO_PROBE_MAX_SLOTS))
    if max_demand_slots is not None:
        probe_slots = min(probe_slots, int(max_demand_slots))
    main_chunk_bytes = _estimate_pure_reroute_chunk_bytes(
        ds_eligible,
        slot_count=max(1, int(read_slot_chunk)),
        ds_row_ptr=ds_row_ptr,
        ds_sat_idx=ds_sat_idx,
    )
    probe_chunk_bytes = _estimate_pure_reroute_chunk_bytes(
        ds_eligible,
        slot_count=max(1, int(probe_slots)),
        ds_row_ptr=ds_row_ptr,
        ds_sat_idx=ds_sat_idx,
    )
    warnings: list[str] = []
    if ds_eligible is not None:
        warnings.extend(
            _dataset_streaming_warnings(
                _PURE_REROUTE_DATASET,
                ds_eligible,
                read_slot_chunk=read_slot_chunk,
            )
        )
    elif ds_row_ptr is not None:
        warnings.extend(
            _dataset_streaming_warnings(
                _PURE_REROUTE_CSR_ROW_PTR_DATASET,
                ds_row_ptr,
                read_slot_chunk=max(1, int(read_slot_chunk) * max(1, int(ds_row_ptr.attrs.get(_PURE_REROUTE_CSR_CELL_COUNT_ATTR, 1)))),
            )
        )
    if main_chunk_bytes > 1024 ** 3:
        warnings.append(
            "Estimated raw pure-reroute main chunk exceeds 1 GiB; reduce READ_SLOT_CHUNK for lower memory use."
        )
    if probe_chunk_bytes > 1024 ** 3:
        warnings.append(
            "Estimated raw pure-reroute auto-probe chunk exceeds 1 GiB; reduce READ_SLOT_CHUNK for lower memory use."
        )
    return {
        "probe_slots": max(1, int(probe_slots)),
        "main_chunk_bytes": int(main_chunk_bytes),
        "probe_chunk_bytes": int(probe_chunk_bytes),
        "warnings": warnings,
    }


def _pure_reroute_gpu_backend_available() -> bool:
    try:
        from scepter import gpu_accel
    except Exception:
        return False
    return bool(getattr(gpu_accel, "_has_exact_pure_reroute_gpu_support", lambda: False)())


def _format_pure_reroute_gpu_runtime_error(exc: BaseException) -> str:
    """Return an actionable runtime error message for GPU pure-reroute failures."""
    detail = str(exc).strip()
    detail_lower = detail.lower()
    if "cuda_fp16.h" in detail_lower or "nvrtc" in detail_lower:
        return (
            "pure_reroute GPU solver failed to compile CuPy/NVRTC kernels. "
            "The active environment is missing CUDA toolkit headers required by CuPy "
            "(for example `cuda_fp16.h`). Use `pure_reroute_backend=\"cpu\"` or run "
            "the parser from `scepter-dev-full` with a matching CUDA toolkit install. "
            f"Original error: {detail}"
        )
    return f"pure_reroute GPU solver failed: {detail}"


_PURE_REROUTE_DATASET = "sat_eligible_mask"
_PURE_REROUTE_CSR_ROW_PTR_DATASET = satsim.PURE_REROUTE_CSR_ROW_PTR_KEY
_PURE_REROUTE_CSR_SAT_IDX_DATASET = satsim.PURE_REROUTE_CSR_SAT_IDX_KEY
_PURE_REROUTE_CSR_TIME_COUNT_ATTR = satsim.PURE_REROUTE_CSR_TIME_COUNT_KEY
_PURE_REROUTE_CSR_CELL_COUNT_ATTR = satsim.PURE_REROUTE_CSR_CELL_COUNT_KEY
_PURE_REROUTE_CSR_SAT_COUNT_ATTR = satsim.PURE_REROUTE_CSR_SAT_COUNT_KEY
_PURE_REROUTE_AUTO_PROBE_MIN_SLOTS = 64
_PURE_REROUTE_AUTO_PROBE_MAX_SLOTS = 256
_PURE_REROUTE_AUTO_PROBE_TARGET_EDGES = 1_000_000
_PURE_REROUTE_AUTO_WARMUP_SLOTS = 2
_PURE_REROUTE_AUTO_GPU_WIN_RATIO = 0.90


def _pure_reroute_group_has_csr(group: h5py.Group) -> bool:
    return (
        _PURE_REROUTE_CSR_ROW_PTR_DATASET in group
        and _PURE_REROUTE_CSR_SAT_IDX_DATASET in group
    )


def _pure_reroute_group_time_count(group: h5py.Group) -> int:
    if _pure_reroute_group_has_csr(group):
        ds_row_ptr = group[_PURE_REROUTE_CSR_ROW_PTR_DATASET]
        if _PURE_REROUTE_CSR_TIME_COUNT_ATTR in ds_row_ptr.attrs:
            return int(ds_row_ptr.attrs[_PURE_REROUTE_CSR_TIME_COUNT_ATTR])
        if _PURE_REROUTE_CSR_CELL_COUNT_ATTR not in ds_row_ptr.attrs:
            raise KeyError(
                f"CSR dataset '{_PURE_REROUTE_CSR_ROW_PTR_DATASET}' is missing "
                f"'{_PURE_REROUTE_CSR_CELL_COUNT_ATTR}' metadata."
            )
        cell_count = int(ds_row_ptr.attrs[_PURE_REROUTE_CSR_CELL_COUNT_ATTR])
        if cell_count <= 0:
            raise ValueError("CSR pure-reroute cell_count metadata must be positive.")
        return max(0, (int(ds_row_ptr.shape[0]) - 1) // cell_count)
    return int(group[_PURE_REROUTE_DATASET].shape[0])


def _load_pure_reroute_chunk(
    group: h5py.Group,
    *,
    slot_start: int,
    slot_stop: int,
) -> Any:
    if _pure_reroute_group_has_csr(group):
        ds_row_ptr = group[_PURE_REROUTE_CSR_ROW_PTR_DATASET]
        ds_sat_idx = group[_PURE_REROUTE_CSR_SAT_IDX_DATASET]
        if (
            _PURE_REROUTE_CSR_TIME_COUNT_ATTR not in ds_row_ptr.attrs
            or _PURE_REROUTE_CSR_CELL_COUNT_ATTR not in ds_row_ptr.attrs
            or _PURE_REROUTE_CSR_SAT_COUNT_ATTR not in ds_row_ptr.attrs
        ):
            raise KeyError(
                f"CSR pure-reroute dataset '{_PURE_REROUTE_CSR_ROW_PTR_DATASET}' is missing shape metadata."
            )
        cell_count = int(ds_row_ptr.attrs[_PURE_REROUTE_CSR_CELL_COUNT_ATTR])
        sat_count = int(ds_row_ptr.attrs[_PURE_REROUTE_CSR_SAT_COUNT_ATTR])
        row_start = int(slot_start) * cell_count
        row_stop = int(slot_stop) * cell_count
        row_ptr_slice = np.asarray(ds_row_ptr[row_start : row_stop + 1], dtype=np.int64)
        edge_start = int(row_ptr_slice[0]) if row_ptr_slice.size else 0
        edge_stop = int(row_ptr_slice[-1]) if row_ptr_slice.size else edge_start
        sat_idx_slice = np.asarray(ds_sat_idx[edge_start:edge_stop], dtype=np.int32)
        return {
            _PURE_REROUTE_CSR_ROW_PTR_DATASET: row_ptr_slice - np.int64(edge_start),
            _PURE_REROUTE_CSR_SAT_IDX_DATASET: sat_idx_slice.astype(np.int32, copy=False),
            _PURE_REROUTE_CSR_TIME_COUNT_ATTR: int(slot_stop) - int(slot_start),
            _PURE_REROUTE_CSR_CELL_COUNT_ATTR: cell_count,
            _PURE_REROUTE_CSR_SAT_COUNT_ATTR: sat_count,
        }
    return np.asarray(group[_PURE_REROUTE_DATASET][slot_start:slot_stop], dtype=np.bool_)


def _active_pure_reroute_slot_mask(
    eligible_chunk: np.ndarray,
    count_chunk: np.ndarray | None = None,
) -> np.ndarray:
    eligible = _normalize_dense_pure_reroute_chunk(np.asarray(eligible_chunk, dtype=np.bool_))
    if count_chunk is not None:
        counts = np.asarray(_normalize_beam_cap_count_chunk(count_chunk), dtype=np.int64)
        if counts.shape[0] != eligible.shape[0]:
            raise ValueError("count_chunk and eligible_chunk must share the same slot axis.")
        demand = counts.reshape(counts.shape[0], -1).sum(axis=1, dtype=np.int64)
        return demand > 0
    return np.any(eligible.reshape(eligible.shape[0], -1), axis=1)


def _active_pure_reroute_slot_mask_any(
    eligible_chunk: Any,
    count_chunk: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(eligible_chunk, dict):
        time_count = int(eligible_chunk[_PURE_REROUTE_CSR_TIME_COUNT_ATTR])
        if count_chunk is not None:
            counts = np.asarray(_normalize_beam_cap_count_chunk(count_chunk), dtype=np.int64)
            if counts.shape[0] != time_count:
                raise ValueError("count_chunk and CSR pure-reroute chunk must share the same slot axis.")
            demand = counts.reshape(counts.shape[0], -1).sum(axis=1, dtype=np.int64)
            return demand > 0
        row_ptr = np.asarray(eligible_chunk[_PURE_REROUTE_CSR_ROW_PTR_DATASET], dtype=np.int64, copy=False)
        cell_count = int(eligible_chunk[_PURE_REROUTE_CSR_CELL_COUNT_ATTR])
        slot_offsets = np.arange(0, time_count * cell_count + 1, cell_count, dtype=np.int64)
        return np.diff(row_ptr[slot_offsets]) > 0
    return _active_pure_reroute_slot_mask(
        np.asarray(eligible_chunk, dtype=np.bool_, copy=False),
        count_chunk,
    )


def _pure_reroute_chunk_edge_count(eligible_chunk: Any) -> int:
    if isinstance(eligible_chunk, dict):
        sat_idx = np.asarray(eligible_chunk[_PURE_REROUTE_CSR_SAT_IDX_DATASET], dtype=np.int32, copy=False)
        return int(sat_idx.size)
    return int(np.count_nonzero(np.asarray(eligible_chunk, dtype=np.bool_, copy=False)))


def _filter_pure_reroute_chunk_slots(
    eligible_chunk: Any,
    slot_mask: np.ndarray,
) -> Any:
    slot_mask_arr = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)
    if isinstance(eligible_chunk, dict):
        time_count = int(eligible_chunk[_PURE_REROUTE_CSR_TIME_COUNT_ATTR])
        cell_count = int(eligible_chunk[_PURE_REROUTE_CSR_CELL_COUNT_ATTR])
        sat_count = int(eligible_chunk[_PURE_REROUTE_CSR_SAT_COUNT_ATTR])
        if slot_mask_arr.size != time_count:
            raise ValueError("CSR pure-reroute slot mask does not match the time axis.")
        slot_indices = np.flatnonzero(slot_mask_arr)
        total_rows = int(slot_indices.size) * cell_count
        row_ptr_out = np.empty(total_rows + 1, dtype=np.int64)
        row_ptr_out[0] = np.int64(0)
        sat_parts: list[np.ndarray] = []
        row_ptr_in = np.asarray(eligible_chunk[_PURE_REROUTE_CSR_ROW_PTR_DATASET], dtype=np.int64, copy=False)
        sat_in = np.asarray(eligible_chunk[_PURE_REROUTE_CSR_SAT_IDX_DATASET], dtype=np.int32, copy=False)
        row_cursor = 0
        edge_cursor = 0
        for slot_idx in slot_indices.astype(np.int64, copy=False):
            base = int(slot_idx) * cell_count
            slot_row_ptr = row_ptr_in[base : base + cell_count + 1]
            edge_start = int(slot_row_ptr[0])
            edge_stop = int(slot_row_ptr[-1])
            slot_edges = sat_in[edge_start:edge_stop]
            row_ptr_out[row_cursor + 1 : row_cursor + cell_count + 1] = (
                slot_row_ptr[1:] - np.int64(edge_start) + np.int64(edge_cursor)
            )
            sat_parts.append(np.asarray(slot_edges, dtype=np.int32, copy=False))
            row_cursor += cell_count
            edge_cursor += int(slot_edges.size)
        sat_out = np.concatenate(sat_parts, axis=0).astype(np.int32, copy=False) if sat_parts else np.empty(0, dtype=np.int32)
        return {
            _PURE_REROUTE_CSR_ROW_PTR_DATASET: row_ptr_out,
            _PURE_REROUTE_CSR_SAT_IDX_DATASET: sat_out,
            _PURE_REROUTE_CSR_TIME_COUNT_ATTR: int(slot_indices.size),
            _PURE_REROUTE_CSR_CELL_COUNT_ATTR: cell_count,
            _PURE_REROUTE_CSR_SAT_COUNT_ATTR: sat_count,
        }
    dense = _normalize_dense_pure_reroute_chunk(np.asarray(eligible_chunk, dtype=np.bool_, copy=False))
    return dense[slot_mask_arr]


def _normalize_beam_cap_count_chunk(count_chunk: np.ndarray) -> np.ndarray:
    """
    Normalize supported count layouts to a conservative ``(T, S)`` view.

    Supported layouts:
    - ``(T, S)``
    - ``(T, sky, S)``
    - ``(T, obs, S, sky)``
    """
    arr = np.asarray(count_chunk, dtype=np.int32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return np.asarray(np.max(arr, axis=1), dtype=np.int32)
    if arr.ndim == 4:
        work = arr[:, 0, :, :] if int(arr.shape[1]) == 1 else np.max(arr, axis=1)
        return np.asarray(np.max(work, axis=2), dtype=np.int32)
    raise ValueError(
        "Beam-cap sizing does not support sat_beam_counts_used with shape "
        f"{tuple(arr.shape)!r}. Supported layouts are (T,S), (T,sky,S), and (T,obs,S,sky)."
    )


def _normalize_dense_pure_reroute_chunk(eligible_chunk: np.ndarray) -> np.ndarray:
    """
    Normalize supported dense eligibility layouts to ``(T, C, S)``.

    Supported layouts:
    - ``(T, C, S)``
    - ``(T, sky, C, S)`` -> ``(T, sky*C, S)``
    - singleton observer-style extra axes are squeezed before evaluating the above rules
    """
    arr = np.asarray(eligible_chunk, dtype=np.bool_)
    if arr.ndim < 3:
        raise ValueError(
            "Pure reroute does not support dense sat_eligible_mask with shape "
            f"{tuple(arr.shape)!r}. Supported dense layouts are (T,C,S) and (T,sky,C,S)."
        )
    if arr.ndim == 3:
        return np.asarray(arr, dtype=np.bool_, copy=False)
    if arr.ndim == 4:
        return np.asarray(
            arr.reshape(
                int(arr.shape[0]),
                int(arr.shape[1]) * int(arr.shape[2]),
                int(arr.shape[3]),
            ),
            dtype=np.bool_,
            copy=False,
        )
    squeeze_axes = tuple(
        idx for idx in range(1, arr.ndim - 2) if int(arr.shape[idx]) == 1
    )
    if squeeze_axes:
        arr = np.squeeze(arr, axis=squeeze_axes)
        if arr.ndim == 3:
            return np.asarray(arr, dtype=np.bool_, copy=False)
        if arr.ndim == 4:
            return np.asarray(
                arr.reshape(
                    int(arr.shape[0]),
                    int(arr.shape[1]) * int(arr.shape[2]),
                    int(arr.shape[3]),
                ),
                dtype=np.bool_,
                copy=False,
            )
    raise ValueError(
        "Pure reroute does not support dense sat_eligible_mask with shape "
        f"{tuple(arr.shape)!r}. Supported dense layouts are (T,C,S) and (T,sky,C,S)."
    )


def _concat_pure_reroute_chunks(chunks: list[Any]) -> Any:
    if not chunks:
        return np.empty((0, 0, 0), dtype=np.bool_)
    first = chunks[0]
    if isinstance(first, dict):
        cell_count = int(first[_PURE_REROUTE_CSR_CELL_COUNT_ATTR])
        sat_count = int(first[_PURE_REROUTE_CSR_SAT_COUNT_ATTR])
        total_time = sum(int(chunk[_PURE_REROUTE_CSR_TIME_COUNT_ATTR]) for chunk in chunks)
        total_rows = total_time * cell_count
        row_ptr = np.empty(total_rows + 1, dtype=np.int64)
        row_ptr[0] = np.int64(0)
        sat_parts: list[np.ndarray] = []
        row_cursor = 0
        edge_cursor = 0
        for chunk in chunks:
            chunk_row_ptr = np.asarray(chunk[_PURE_REROUTE_CSR_ROW_PTR_DATASET], dtype=np.int64, copy=False)
            chunk_sat = np.asarray(chunk[_PURE_REROUTE_CSR_SAT_IDX_DATASET], dtype=np.int32, copy=False)
            n_rows = int(chunk_row_ptr.size) - 1
            row_ptr[row_cursor + 1 : row_cursor + n_rows + 1] = chunk_row_ptr[1:] + np.int64(edge_cursor)
            sat_parts.append(chunk_sat.astype(np.int32, copy=False))
            row_cursor += n_rows
            edge_cursor += int(chunk_sat.size)
        sat_idx = np.concatenate(sat_parts, axis=0).astype(np.int32, copy=False) if sat_parts else np.empty(0, dtype=np.int32)
        return {
            _PURE_REROUTE_CSR_ROW_PTR_DATASET: row_ptr,
            _PURE_REROUTE_CSR_SAT_IDX_DATASET: sat_idx,
            _PURE_REROUTE_CSR_TIME_COUNT_ATTR: total_time,
            _PURE_REROUTE_CSR_CELL_COUNT_ATTR: cell_count,
            _PURE_REROUTE_CSR_SAT_COUNT_ATTR: sat_count,
        }
    return np.concatenate(
        [np.asarray(chunk, dtype=np.bool_, copy=False) for chunk in chunks],
        axis=0,
    ).astype(np.bool_, copy=False)


def _collect_pure_reroute_probe_buffer(
    h5: h5py.File,
    iter_names: list[str],
    *,
    count_var: str | None,
    target_slots: int,
    target_edges: int,
    max_slots: int,
    slot_chunk_limit: int | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    iter_root_key: str = "iter",
) -> tuple[Any, int, int]:
    buffers: list[Any] = []
    probe_slots = 0
    probe_edges = 0
    raw_slots_scanned = 0
    max_slots_i = max(1, int(max_slots))
    target_slots_i = max(1, int(target_slots))
    target_edges_i = max(1, int(target_edges))
    slot_chunk_limit_i = max(1, int(slot_chunk_limit if slot_chunk_limit is not None else READ_SLOT_CHUNK))

    for it in iter_names:
        g = h5[iter_root_key][it]
        t_count = _pure_reroute_group_time_count(g)
        ds_count = g[count_var] if count_var is not None else None
        s0 = 0
        while s0 < t_count:
            remaining_slots = max_slots_i - probe_slots
            if remaining_slots <= 0:
                break
            s1 = min(t_count, s0 + min(slot_chunk_limit_i, remaining_slots))
            eligible_chunk = _load_pure_reroute_chunk(g, slot_start=s0, slot_stop=s1)
            count_chunk = np.asarray(ds_count[s0:s1], dtype=np.int32) if ds_count is not None else None
            raw_slots_scanned += int(s1 - s0)
            active_mask = _active_pure_reroute_slot_mask_any(eligible_chunk, count_chunk)
            if not np.any(active_mask):
                if progress_callback is not None:
                    progress_callback(raw_slots_scanned, probe_slots, probe_edges)
                s0 = s1
                continue
            active_chunk = _filter_pure_reroute_chunk_slots(eligible_chunk, active_mask)
            if isinstance(active_chunk, dict):
                active_slots = int(active_chunk[_PURE_REROUTE_CSR_TIME_COUNT_ATTR])
            else:
                active_slots = int(np.asarray(active_chunk, dtype=np.bool_, copy=False).shape[0])
            if active_slots > remaining_slots:
                keep_mask = np.zeros(active_slots, dtype=np.bool_)
                keep_mask[:remaining_slots] = True
                active_chunk = _filter_pure_reroute_chunk_slots(active_chunk, keep_mask)
                active_slots = remaining_slots
            if active_slots <= 0:
                if progress_callback is not None:
                    progress_callback(raw_slots_scanned, probe_slots, probe_edges)
                s0 = s1
                continue
            buffers.append(active_chunk)
            probe_slots += int(active_slots)
            probe_edges += _pure_reroute_chunk_edge_count(active_chunk)
            if progress_callback is not None:
                progress_callback(raw_slots_scanned, probe_slots, probe_edges)
            if probe_slots >= max_slots_i:
                break
            if probe_slots >= target_slots_i and probe_edges >= target_edges_i:
                break
            s0 = s1
        if probe_slots >= max_slots_i or (probe_slots >= target_slots_i and probe_edges >= target_edges_i):
            break

    if not buffers:
        return np.empty((0, 0, 0), dtype=np.bool_), 0, 0
    probe_mask = _concat_pure_reroute_chunks(buffers)
    return probe_mask, probe_slots, probe_edges


def _run_pure_reroute_cpu_solver(
    eligible_chunk: Any,
    *,
    nco: int,
    beam_caps: np.ndarray,
) -> dict[str, Any]:
    return satsim.pure_reroute_service_curve(
        eligible_chunk,
        nco=int(nco),
        beam_caps=beam_caps,
    )


def _run_pure_reroute_gpu_solver(
    session: Any,
    eligible_chunk: Any,
    *,
    nco: int,
    beam_caps: np.ndarray,
) -> dict[str, Any]:
    try:
        return session.pure_reroute_service_curve(
            eligible_chunk,
            nco=int(nco),
            beam_caps=beam_caps,
            return_device=False,
        )
    except Exception as exc:
        raise RuntimeError(_format_pure_reroute_gpu_runtime_error(exc)) from exc


def _preflight_pure_reroute_gpu_solver(session: Any) -> None:
    """Run a tiny exact solve to fail early on broken CuPy/NVRTC setups."""
    _run_pure_reroute_gpu_solver(
        session,
        np.ones((1, 1, 1), dtype=np.bool_),
        nco=1,
        beam_caps=np.arange(2, dtype=np.int32),
    )


def _select_auto_pure_reroute_backend(
    h5: h5py.File,
    iter_names: list[str],
    *,
    count_var: str | None,
    nco: int,
    beam_caps: np.ndarray,
    max_demand_slots: int | None,
    iter_root_key: str = "iter",
) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "pure_reroute_backend_requested": "auto",
        "pure_reroute_backend_selected": "cpu",
        "pure_reroute_probe_slots": 0,
        "pure_reroute_probe_edges": 0,
        "pure_reroute_probe_cpu_s": None,
        "pure_reroute_probe_gpu_s": None,
    }
    if not _pure_reroute_gpu_backend_available():
        return diag

    probe_target_slots = min(int(READ_SLOT_CHUNK), int(_PURE_REROUTE_AUTO_PROBE_MAX_SLOTS))
    max_probe_slots = int(probe_target_slots)
    if max_demand_slots is not None:
        probe_target_slots = min(probe_target_slots, int(max_demand_slots))
        max_probe_slots = min(max_probe_slots, int(max_demand_slots))
    if probe_target_slots <= 0 or max_probe_slots <= 0:
        return diag

    _progress_write(
        "[pure-reroute-probe] Benchmarking exact CPU vs GPU solver "
        f"(slot_limit={int(READ_SLOT_CHUNK):,}, probe_slots={int(probe_target_slots):,})."
    )
    probe_progress = _StageProgress(
        desc="[pure-reroute-probe] active slots",
        total=int(probe_target_slots),
        unit="slot",
        report_every=max(1, int(PROGRESS_EVERY_DEMAND_SLOTS)),
    )
    probe_seen = {"slots": 0}

    def _on_probe_progress(raw_scanned: int, active_slots: int, active_edges: int) -> None:
        delta_slots = max(0, int(active_slots) - int(probe_seen["slots"]))
        if delta_slots > 0:
            probe_progress.update(delta_slots)
            probe_seen["slots"] = int(active_slots)
        probe_progress.set_postfix(
            {
                "raw_scanned": f"{int(raw_scanned):,}",
                "active_slots": f"{int(active_slots):,}",
                "edges": f"{int(active_edges):,}",
                "rss_gib": _rss_gib_postfix(),
            }
        )
        probe_progress.maybe_refresh(counter=int(active_slots), force=True)

    try:
        probe_mask, probe_slots, probe_edges = _collect_pure_reroute_probe_buffer(
            h5,
            iter_names,
            count_var=count_var,
            target_slots=probe_target_slots,
            target_edges=int(_PURE_REROUTE_AUTO_PROBE_TARGET_EDGES),
            max_slots=max_probe_slots,
            slot_chunk_limit=probe_target_slots,
            progress_callback=_on_probe_progress,
            iter_root_key=iter_root_key,
        )
        probe_progress.maybe_refresh(counter=int(probe_slots), force=True)
    finally:
        probe_progress.close()
    diag["pure_reroute_probe_slots"] = int(probe_slots)
    diag["pure_reroute_probe_edges"] = int(probe_edges)
    if probe_slots <= 0:
        return diag

    if isinstance(probe_mask, dict):
        probe_time_count = int(probe_mask[_PURE_REROUTE_CSR_TIME_COUNT_ATTR])
    else:
        probe_time_count = int(np.asarray(probe_mask, dtype=np.bool_, copy=False).shape[0])
    warm_slots = min(int(probe_time_count), int(_PURE_REROUTE_AUTO_WARMUP_SLOTS))
    if warm_slots > 0:
        warm_mask = _filter_pure_reroute_chunk_slots(
            probe_mask,
            np.arange(probe_time_count, dtype=np.int64) < warm_slots,
        )
    else:
        warm_mask = _filter_pure_reroute_chunk_slots(
            probe_mask,
            np.zeros(probe_time_count, dtype=np.bool_),
        )

    try:
        from scepter import gpu_accel
    except Exception:
        return diag

    gpu_session = None
    try:
        gpu_session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
        if warm_slots > 0:
            _run_pure_reroute_cpu_solver(warm_mask, nco=nco, beam_caps=beam_caps)
            _run_pure_reroute_gpu_solver(gpu_session, warm_mask, nco=nco, beam_caps=beam_caps)

        cpu_t0 = time.perf_counter()
        _run_pure_reroute_cpu_solver(probe_mask, nco=nco, beam_caps=beam_caps)
        cpu_elapsed = time.perf_counter() - cpu_t0
        diag["pure_reroute_probe_cpu_s"] = float(cpu_elapsed)

        gpu_t0 = time.perf_counter()
        _run_pure_reroute_gpu_solver(gpu_session, probe_mask, nco=nco, beam_caps=beam_caps)
        gpu_elapsed = time.perf_counter() - gpu_t0
        diag["pure_reroute_probe_gpu_s"] = float(gpu_elapsed)

        if gpu_elapsed <= cpu_elapsed * float(_PURE_REROUTE_AUTO_GPU_WIN_RATIO):
            diag["pure_reroute_backend_selected"] = "gpu"
        else:
            diag["pure_reroute_backend_selected"] = "cpu"
        _progress_write(
            "[pure-reroute-probe] done: "
            f"slots={int(probe_slots):,}, edges={int(probe_edges):,}, "
            f"cpu={diag['pure_reroute_probe_cpu_s']}, gpu={diag['pure_reroute_probe_gpu_s']}, "
            f"selected={diag['pure_reroute_backend_selected']}"
        )
        return diag
    except Exception as exc:
        _progress_write(f"[WARN][pure-reroute-probe] {_format_pure_reroute_gpu_runtime_error(exc)}")
        return diag
    finally:
        if gpu_session is not None:
            gpu_session.close(reset_device=False)


# -----------------------------------------------------------------------------
# Streaming accumulators
# -----------------------------------------------------------------------------
@dataclass
class PolicyMeta:
    key: str
    label: str
    color: str
    dash: str


@dataclass
class PolicyEval:
    key: str
    label: str
    color: str
    dash: str
    delta_run: np.ndarray
    eps_run: np.ndarray
    ccdf_exceed_percent: np.ndarray
    selected_b: int
    selected_idx: int
    slot_max_ratio: np.ndarray | None


class StreamingPolicyAccumulator:
    def __init__(self, beam_caps_grid: np.ndarray, *, tau: float, store_slot_samples: bool = False) -> None:
        b = np.asarray(beam_caps_grid, dtype=np.int64)
        if b.ndim != 1 or b.size == 0:
            raise ValueError("beam_caps_grid must be a non-empty 1D array.")
        expected = np.arange(int(b[0]), int(b[0]) + b.size, dtype=np.int64)
        if not np.array_equal(b, expected) or int(b[0]) != 0:
            raise ValueError("beam_caps_grid must be contiguous and start at 0.")
        self.n_b = b.size
        self.beam_caps = b.astype(np.float64, copy=False)
        self.tail_d_diff = np.zeros(self.n_b + 1, dtype=np.float64)
        self.tail_p_diff = np.zeros(self.n_b + 1, dtype=np.float64)
        self.slot_tail_diff = np.zeros(self.n_b + 1, dtype=np.float64)
        self.ccdf_tail_diff = np.zeros(self.n_b + 1, dtype=np.float64)
        self.tau = float(tau)
        self.store_slot_samples = bool(store_slot_samples)
        self.slot_max_samples: list[float] | None = [] if self.store_slot_samples else None
        self.slot_count = 0

    def _range_add(self, diff: np.ndarray, k: np.ndarray, w: np.ndarray) -> None:
        if k.size == 0:
            return
        diff[0] += float(np.sum(w, dtype=np.float64))
        neg = np.bincount((k + 1).astype(np.int64, copy=False), weights=w, minlength=diff.size)
        diff[:neg.size] -= neg.astype(np.float64, copy=False)

    def add_entries(self, demand_entries: np.ndarray, pool_entries: np.ndarray) -> None:
        d = np.asarray(demand_entries, dtype=np.float64).reshape(-1)
        p = np.asarray(pool_entries, dtype=np.float64).reshape(-1)
        if d.size != p.size:
            raise ValueError("demand_entries and pool_entries must have same size.")
        if d.size == 0:
            return
        r = d / np.maximum(p, 1e-30)
        k = np.ceil(r).astype(np.int64, copy=False) - 1
        k = np.clip(k, -1, self.n_b - 1)
        m = k >= 0
        if np.any(m):
            km = k[m]
            self._range_add(self.tail_d_diff, km, d[m])
            self._range_add(self.tail_p_diff, km, p[m])

    def add_entry(self, demand_entry: float, pool_entry: float) -> None:
        d = float(demand_entry)
        p = float(pool_entry)
        if d <= 0.0:
            return
        r = d / max(p, 1e-30)
        k = int(np.clip(int(np.ceil(r) - 1), -1, self.n_b - 1))
        if k >= 0:
            self.tail_d_diff[0] += d
            self.tail_d_diff[k + 1] -= d
            self.tail_p_diff[0] += p
            self.tail_p_diff[k + 1] -= p

    def add_slot_ratio(self, slot_ratio: float) -> None:
        r = float(slot_ratio)
        if np.isfinite(r):
            k = int(np.clip(int(np.ceil(r) - 1), -1, self.n_b - 1))
            if k >= 0:
                self.slot_tail_diff[0] += 1.0
                self.slot_tail_diff[k + 1] -= 1.0
            # Tail-risk diagnostic uses B_req = ceil((1-tau) * ratio):
            # P(B_req > B) = P(B_req >= B+1), so the same prefix-range trick applies.
            b_req = int(np.ceil((1.0 - self.tau) * r))
            k_req = int(np.clip(b_req - 1, -1, self.n_b - 1))
            if k_req >= 0:
                self.ccdf_tail_diff[0] += 1.0
                self.ccdf_tail_diff[k_req + 1] -= 1.0
            if self.slot_max_samples is not None:
                self.slot_max_samples.append(r)
        else:
            if self.slot_max_samples is not None:
                self.slot_max_samples.append(np.nan)
        self.slot_count += 1

    def finalize(
        self,
        demand_sum: float,
        delta_target: float,
        eps_target: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        tail_d = np.cumsum(self.tail_d_diff, dtype=np.float64)[: self.n_b]
        tail_p = np.cumsum(self.tail_p_diff, dtype=np.float64)[: self.n_b]
        overflow = np.maximum(tail_d - self.beam_caps * tail_p, 0.0)
        delta = overflow / float(max(demand_sum, 1e-30))
        tail_slots = np.cumsum(self.slot_tail_diff, dtype=np.float64)[: self.n_b]
        eps = tail_slots / float(max(self.slot_count, 1))
        ccdf_tail = np.cumsum(self.ccdf_tail_diff, dtype=np.float64)[: self.n_b]
        ccdf_exceed = (ccdf_tail / float(max(self.slot_count, 1))) * 100.0
        b_sel = choose_smallest_beam_cap(self.beam_caps.astype(np.int32, copy=False), delta, eps, delta_target, eps_target)
        idx = int(np.where(self.beam_caps.astype(np.int32, copy=False) == b_sel)[0][0])
        return delta, eps, ccdf_exceed, b_sel, idx


# -----------------------------------------------------------------------------
# Slot/group helpers
# -----------------------------------------------------------------------------
_SAT_KEY_DTYPE = np.dtype([("belt", np.int16), ("az_bits", np.uint32), ("el_bits", np.uint32)])
_SLOT_SAT_KEY_DTYPE = np.dtype(
    [("slot", np.int32), ("belt", np.int16), ("az_bits", np.uint32), ("el_bits", np.uint32)]
)
_SLOT_CELL_BELT_SKY_DTYPE = np.dtype(
    [("slot", np.int32), ("cell", np.int32), ("belt", np.int16), ("sky", np.int32)]
)


@dataclass
class ChunkVisibilityViews:
    slot_offsets: np.ndarray
    belt_valid: np.ndarray
    sky_valid: np.ndarray
    az_bits_valid: np.ndarray | None
    el_bits_valid: np.ndarray | None
    slot_valid: np.ndarray | None
    cell_valid: np.ndarray | None
    sat_slot_offsets: np.ndarray | None
    sat_counts: np.ndarray | None
    sat_belt: np.ndarray | None
    sat_sky: np.ndarray | None


def _iter_names(h5: h5py.File, *, iter_root_key: str = "iter") -> list[str]:
    names = [k for k in h5.get(iter_root_key, {}).keys() if k.startswith("iter_")]
    names.sort()
    return names


def _choose_count_var(h5: h5py.File, iter_names: list[str], candidates: tuple[str, ...], *, iter_root_key: str = "iter") -> str:
    for v in candidates:
        if all(v in h5[iter_root_key][it] for it in iter_names):
            return v
    sample = list(h5[iter_root_key][iter_names[0]].keys()) if iter_names else []
    if "beam_demand_count" in sample:
        raise KeyError(
            "No per-satellite beam-count dataset was found. "
            f"Tried {candidates}. First iter has {sample}. "
            "'beam_demand_count' is demand-per-slot and cannot replace "
            "'sat_beam_counts_used' for count-based sizing."
        )
    raise KeyError(f"No count var found. Tried {candidates}. First iter has {sample}")


def _build_sky_mapper(mode: str) -> tuple[int, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    if str(mode).lower().strip() != "s1586":
        raise ValueError(f"Unsupported SKYCELL_MODE={mode!r}.")
    if _skycell_id_s1586 is None:
        raise RuntimeError("s1586 mode requires scepter.angle_sampler._skycell_id_s1586.")

    def mapper(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        return np.asarray(_skycell_id_s1586(az_deg, el_deg), dtype=np.int32)

    return int(S1586_N_CELLS), mapper


def _build_chunk_visibility_views(
    belt_chunk: np.ndarray,
    az_chunk: np.ndarray,
    el_chunk: np.ndarray,
    sky_mapper: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    need_sat_bits: bool,
    need_slot_cell: bool,
    need_grouped_sats: bool,
) -> ChunkVisibilityViews:
    """
    Build chunk-level flattened valid-link views to avoid repeated per-slot mapping.

    This computes skycell IDs once per chunk for all valid links, then exposes
    slot-wise slices via slot_offsets.
    """
    belt = np.asarray(belt_chunk)
    az = np.asarray(az_chunk, dtype=np.float32)
    el = np.asarray(el_chunk, dtype=np.float32)
    if belt.shape != az.shape or belt.shape != el.shape:
        raise ValueError("belt_chunk, az_chunk, el_chunk must have identical shapes.")
    if belt.ndim < 2:
        raise ValueError("Expected at least 2D [slot, ...] visibility arrays.")

    n_slots = int(belt.shape[0])
    per_slot = int(np.prod(belt.shape[1:], dtype=np.int64))
    belt2 = belt.reshape(n_slots, per_slot)
    az2 = az.reshape(n_slots, per_slot)
    el2 = el.reshape(n_slots, per_slot)

    valid2 = (belt2 >= 0) & np.isfinite(az2) & np.isfinite(el2)
    valid_counts = valid2.sum(axis=1, dtype=np.int64)
    slot_offsets = np.empty(n_slots + 1, dtype=np.int64)
    slot_offsets[0] = 0
    slot_offsets[1:] = np.cumsum(valid_counts, dtype=np.int64)
    n_valid_total = int(slot_offsets[-1])

    if n_valid_total == 0:
        empty_i16 = np.empty(0, dtype=np.int16)
        empty_i32 = np.empty(0, dtype=np.int32)
        empty_u32 = np.empty(0, dtype=np.uint32)
        return ChunkVisibilityViews(
            slot_offsets=slot_offsets,
            belt_valid=empty_i16,
            sky_valid=empty_i32,
            az_bits_valid=empty_u32 if need_sat_bits else None,
            el_bits_valid=empty_u32 if need_sat_bits else None,
            slot_valid=empty_i32 if need_slot_cell else None,
            cell_valid=empty_i32 if need_slot_cell else None,
            sat_slot_offsets=np.zeros(n_slots + 1, dtype=np.int64) if need_grouped_sats else None,
            sat_counts=np.empty(0, dtype=np.float64) if need_grouped_sats else None,
            sat_belt=empty_i32 if need_grouped_sats else None,
            sat_sky=empty_i32 if need_grouped_sats else None,
        )

    valid_flat = valid2.reshape(-1)
    belt_valid = np.asarray(belt2.reshape(-1)[valid_flat], dtype=np.int16, copy=False)
    az_valid = np.asarray(az2.reshape(-1)[valid_flat], dtype=np.float32, copy=False)
    el_valid = np.asarray(el2.reshape(-1)[valid_flat], dtype=np.float32, copy=False)
    sky_valid = np.asarray(sky_mapper(az_valid, el_valid), dtype=np.int32, copy=False)

    need_key_bits = need_sat_bits or need_grouped_sats
    az_bits_valid: np.ndarray | None = None
    el_bits_valid: np.ndarray | None = None
    if need_key_bits:
        az_bits_valid = az_valid.view(np.uint32)
        el_bits_valid = el_valid.view(np.uint32)

    slot_valid: np.ndarray | None = None
    cell_valid: np.ndarray | None = None
    flat_valid_idx: np.ndarray | None = None
    if need_slot_cell or need_grouped_sats:
        flat_valid_idx = np.flatnonzero(valid_flat).astype(np.int64, copy=False)
        slot_valid_all = (flat_valid_idx // np.int64(per_slot)).astype(np.int32, copy=False)
    else:
        slot_valid_all = None
    if need_slot_cell:
        slot_valid = np.asarray(slot_valid_all, dtype=np.int32, copy=False)
        if belt.ndim >= 3:
            n_cells = int(belt.shape[1])
            links_per_cell = max(1, int(per_slot // max(n_cells, 1)))
            cell_valid = ((flat_valid_idx % np.int64(per_slot)) // np.int64(links_per_cell)).astype(np.int32, copy=False)
        elif belt.ndim == 2:
            cell_valid = (flat_valid_idx % np.int64(per_slot)).astype(np.int32, copy=False)
        else:
            cell_valid = np.zeros(n_valid_total, dtype=np.int32)

    sat_slot_offsets: np.ndarray | None = None
    sat_counts: np.ndarray | None = None
    sat_belt: np.ndarray | None = None
    sat_sky: np.ndarray | None = None
    if need_grouped_sats:
        if slot_valid_all is None or az_bits_valid is None or el_bits_valid is None:
            raise RuntimeError("Internal error: grouped satellite views require slot and az/el key arrays.")
        sat_keys = np.empty(n_valid_total, dtype=_SLOT_SAT_KEY_DTYPE)
        sat_keys["slot"] = slot_valid_all
        sat_keys["belt"] = belt_valid
        sat_keys["az_bits"] = az_bits_valid
        sat_keys["el_bits"] = el_bits_valid
        uniq_sat, first_idx, cnt = np.unique(sat_keys, return_index=True, return_counts=True)
        uniq_slots = uniq_sat["slot"].astype(np.int32, copy=False)
        sat_group_counts = np.bincount(uniq_slots.astype(np.int64, copy=False), minlength=n_slots).astype(np.int64, copy=False)
        sat_slot_offsets = np.empty(n_slots + 1, dtype=np.int64)
        sat_slot_offsets[0] = 0
        sat_slot_offsets[1:] = np.cumsum(sat_group_counts, dtype=np.int64)
        sat_counts = cnt.astype(np.float64, copy=False)
        sat_belt = uniq_sat["belt"].astype(np.int64, copy=False)
        sat_sky = sky_valid[first_idx].astype(np.int64, copy=False)

    return ChunkVisibilityViews(
        slot_offsets=slot_offsets,
        belt_valid=belt_valid,
        sky_valid=sky_valid,
        az_bits_valid=az_bits_valid,
        el_bits_valid=el_bits_valid,
        slot_valid=slot_valid,
        cell_valid=cell_valid,
        sat_slot_offsets=sat_slot_offsets,
        sat_counts=sat_counts,
        sat_belt=sat_belt,
        sat_sky=sat_sky,
    )


def _reconstruct_sat_counts_from_views(
    views: ChunkVisibilityViews,
    slot_local_idx: int,
    slot_demand: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, int]:
    """
    Reconstruct per-satellite counts for one slot using chunk-level precomputed views.
    """
    s0 = int(views.slot_offsets[slot_local_idx])
    s1 = int(views.slot_offsets[slot_local_idx + 1])
    n_valid = s1 - s0
    if n_valid <= 0:
        return np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0, float(slot_demand), -1
    if (
        views.sat_slot_offsets is None
        or views.sat_counts is None
        or views.sat_belt is None
        or views.sat_sky is None
    ):
        raise RuntimeError("ChunkVisibilityViews missing grouped satellite views required for reconstruction.")

    g0 = int(views.sat_slot_offsets[slot_local_idx])
    g1 = int(views.sat_slot_offsets[slot_local_idx + 1])
    sat_counts = np.asarray(views.sat_counts[g0:g1], dtype=np.float64, copy=False)
    sat_belt = np.asarray(views.sat_belt[g0:g1], dtype=np.int64, copy=False)
    sat_sky = np.asarray(views.sat_sky[g0:g1], dtype=np.int64, copy=False)
    max_sky = int(np.max(sat_sky)) if np.any(sat_sky >= 0) else -1
    residual = max(0.0, float(slot_demand) - float(np.sum(sat_counts)))
    return sat_counts, sat_belt, sat_sky, int(n_valid), residual, max_sky


if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _edge_keys_cross_numba(now_skies: np.ndarray, past_skies: np.ndarray, belt_id: int, n_sky: int) -> np.ndarray:
        work = int(now_skies.size * past_skies.size)
        out = np.empty(work, dtype=np.int64)
        n = 0
        for i in range(now_skies.size):
            sn = int(now_skies[i])
            for j in range(past_skies.size):
                ps = int(past_skies[j])
                if sn == ps:
                    continue
                if sn < ps:
                    a = sn
                    b = ps
                else:
                    a = ps
                    b = sn
                out[n] = (int(belt_id) * int(n_sky) + int(a)) * int(n_sky) + int(b)
                n += 1
        return out[:n]


def _build_temporal_edge_keys(
    belt_id: int,
    skies_now: np.ndarray,
    past_skies: np.ndarray,
    n_sky: int,
) -> np.ndarray:
    if skies_now.size == 0 or past_skies.size == 0:
        return np.empty(0, dtype=np.int64)
    work = int(skies_now.size * past_skies.size)
    if work <= 0:
        return np.empty(0, dtype=np.int64)

    now_i = np.asarray(skies_now, dtype=np.int32, copy=False)
    past_i = np.asarray(past_skies, dtype=np.int32, copy=False)
    if _HAVE_NUMBA and work >= int(TEMPORAL_NUMBA_MIN_EDGE_WORK):
        return _edge_keys_cross_numba(now_i, past_i, int(belt_id), int(n_sky))
    a = np.minimum(now_i[:, None], past_i[None, :])
    b = np.maximum(now_i[:, None], past_i[None, :])
    m = a != b
    if not np.any(m):
        return np.empty(0, dtype=np.int64)
    return (
        (int(belt_id) * int(n_sky) + a[m].astype(np.int64, copy=False)) * int(n_sky)
        + b[m].astype(np.int64, copy=False)
    )


def _merge_temporal_edge_batches(
    edge_counts: dict[int, int],
    edge_key_batches: list[np.ndarray],
) -> None:
    if not edge_key_batches:
        return
    if len(edge_key_batches) == 1:
        edge_keys = np.asarray(edge_key_batches[0], dtype=np.int64, copy=False)
    else:
        edge_keys = np.concatenate(edge_key_batches).astype(np.int64, copy=False)
    if edge_keys.size == 0:
        return
    uniq, counts = np.unique(edge_keys, return_counts=True)
    for edge_key, count in zip(uniq, counts):
        edge_counts[int(edge_key)] += int(count)


def _prepare_temporal_lookup_dense(comp_map: dict[int, int]) -> np.ndarray:
    if not comp_map:
        return np.empty(0, dtype=np.int64)
    max_node = max(comp_map)
    roots = np.full(int(max_node) + 1, -1, dtype=np.int64)
    for node, root in comp_map.items():
        roots[int(node)] = np.int64(root)
    return roots


def _map_temporal_roots_dense(node_id: np.ndarray, root_lookup: np.ndarray) -> np.ndarray:
    if root_lookup.size == 0:
        return np.asarray(node_id, dtype=np.int64, copy=False)
    node = np.asarray(node_id, dtype=np.int64, copy=False)
    out = node.copy()
    in_range = (node >= 0) & (node < root_lookup.size)
    if np.any(in_range):
        roots = root_lookup[node[in_range]]
        hits = roots >= 0
        if np.any(hits):
            out_idx = np.nonzero(in_range)[0][hits]
            out[out_idx] = roots[hits]
    return out


def _aggregate(sat_counts: np.ndarray, labels: np.ndarray, slot_demand: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if sat_counts.size == 0:
        if slot_demand <= 0.0:
            return np.empty(0), np.empty(0), 0.0, np.empty(0, dtype=np.int64)
        return np.asarray([slot_demand]), np.asarray([1.0]), float(slot_demand), np.asarray([-1], dtype=np.int64)
    x = np.asarray(sat_counts, dtype=np.float64)
    lbl = np.asarray(labels, dtype=np.int64)
    u, inv = np.unique(lbl, return_inverse=True)
    sum_x = np.bincount(inv, weights=x, minlength=u.size).astype(np.float64, copy=False)
    sum_x2 = np.bincount(inv, weights=(x * x), minlength=u.size).astype(np.float64, copy=False)
    a = np.bincount(inv, minlength=u.size).astype(np.float64, copy=False)
    aeff = np.maximum(np.minimum((sum_x * sum_x) / np.maximum(sum_x2, 1e-30), a), 1.0)
    residual = float(slot_demand) - float(np.sum(sum_x))
    if residual > 0.0:
        u = np.concatenate((u, np.asarray([-1], dtype=np.int64)))
        sum_x = np.concatenate((sum_x, np.asarray([residual])))
        aeff = np.concatenate((aeff, np.asarray([1.0])))
    slot_ratio = float(np.max(sum_x / np.maximum(aeff, 1e-30)))
    return sum_x, aeff, slot_ratio, u


def _fallback_labels(
    sat_belt: np.ndarray,
    strict_labels: np.ndarray,
    sky_valid_mask: np.ndarray,
    min_active_sats: int,
) -> np.ndarray:
    """
    Build fallback labels:
    - default group is belt
    - sky-valid rows use strict (belt, sky) only when that strict group has
      at least `min_active_sats` satellites in this slot; otherwise belt.
    """
    if strict_labels.size == 0:
        return np.empty(0, dtype=np.int64)

    out = np.asarray(sat_belt, dtype=np.int64).copy()
    m = np.asarray(sky_valid_mask, dtype=bool)
    if not np.any(m):
        return out

    strict_valid = np.asarray(strict_labels[m], dtype=np.int64)
    u, inv = np.unique(strict_valid, return_inverse=True)
    sat_per_group = np.bincount(inv, minlength=u.size)
    dense_by_u = sat_per_group >= int(min_active_sats)
    dense_rows = dense_by_u[inv]

    strict_dense = strict_valid[dense_rows] + np.int64(FALLBACK_GROUP_OFFSET)
    idx_valid = np.nonzero(m)[0]
    idx_dense = idx_valid[dense_rows]
    out[idx_dense] = strict_dense
    return out


def _estimate_window_slots(times_ds: h5py.Dataset | None, window_s: float) -> int:
    """Estimate integer slot window from times dataset; fallback to global timestep."""
    dt_s = np.nan
    if times_ds is not None:
        try:
            t = np.asarray(times_ds[:], dtype=np.float64).reshape(-1)
            t = t[np.isfinite(t)]
            if t.size >= 2:
                dt_arr = np.diff(t) * 86400.0
                dt_arr = dt_arr[np.isfinite(dt_arr) & (dt_arr > 0.0)]
                if dt_arr.size:
                    dt_s = float(np.median(dt_arr))
        except Exception:
            dt_s = np.nan
    if not np.isfinite(dt_s):
        dt_s = float(_timestep_s) if np.isfinite(_timestep_s) and _timestep_s > 0.0 else 1.0
    return max(1, int(np.ceil(float(window_s) / dt_s)))


def _build_temporal_compatibility_from_h5(
    h5: h5py.File,
    iter_names: list[str],
    count_var: str,
    sky_mapper: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_sky: int,
    *,
    iter_root_key: str = "iter",
) -> tuple[dict[int, int], dict[str, float]]:
    """
    Build per-belt skycell compatibility via cell-local temporal handovers.

    For each (cell, belt), connect skycells observed within TEMPORAL_WINDOW_S.
    Edges with count >= TEMPORAL_MIN_EDGE_COUNT are unioned into compatibility sets.
    """
    if not ENABLE_TEMPORAL_SKY_COMPAT:
        return {}, {"enabled": 0.0}

    _progress_write(
        "[temporal-prepass] Building skycell compatibility graph: "
        f"window={TEMPORAL_WINDOW_S:g}s, min_edge_count={TEMPORAL_MIN_EDGE_COUNT}"
    )
    t0 = time.perf_counter()
    slot_stride = max(1, int(TEMPORAL_PREPASS_SLOT_STRIDE))
    global_cap = _normalize_optional_positive_int(MAX_DEMAND_SLOTS, name="MAX_DEMAND_SLOTS")
    max_demand_slots = int(TEMPORAL_PREPASS_MAX_DEMAND_SLOTS)
    if global_cap is not None:
        max_demand_slots = int(global_cap) if max_demand_slots <= 0 else min(max_demand_slots, int(global_cap))
    raw_slots_total = sum(int(h5[iter_root_key][it][count_var].shape[0]) for it in iter_names)
    if slot_stride > 1:
        _progress_write(f"[temporal-prepass] slot_stride={slot_stride} (sampling every {slot_stride}th demand slot)")
    if max_demand_slots > 0:
        _progress_write(f"[temporal-prepass] max_demand_slots={max_demand_slots} (early-stop prepass)")

    # edge key encoding: ((belt * n_sky + sky_a) * n_sky + sky_b), with sky_a < sky_b
    edge_counts: dict[int, int] = defaultdict(int)
    node_seen: set[int] = set()
    slots_scanned = 0
    slots_with_demand = 0
    stop_early = False
    prepass_progress = _StageProgress(
        desc="[temporal-prepass] raw slots",
        total=int(raw_slots_total),
        unit="slot",
        report_every=max(1, int(PROGRESS_EVERY_DEMAND_SLOTS)),
    )

    for ii, it in enumerate(iter_names, start=1):
        g = h5[iter_root_key][it]
        ds_count = g[count_var]
        ds_belt = g["sat_belt_id"]
        ds_az = g["sat_azimuth"]
        ds_el = g["sat_elevation"]
        ds_times = g["times"] if "times" in g else None
        window_slots = _estimate_window_slots(ds_times, TEMPORAL_WINDOW_S)

        _progress_write(
            f"[temporal-prepass] iter {ii}/{len(iter_names)} {it}: "
            f"window_slots={window_slots}, shape={tuple(ds_count.shape)}"
        )

        # Reset history per iteration to avoid linking unrelated random time shifts.
        history: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        T = int(ds_count.shape[0])
        for s0 in range(0, T, READ_SLOT_CHUNK):
            s1 = min(T, s0 + READ_SLOT_CHUNK)
            c = np.asarray(ds_count[s0:s1], dtype=np.int32)
            d = c.sum(axis=1, dtype=np.int64)
            slots_scanned += int(c.shape[0])
            prepass_progress.update(int(c.shape[0]))
            mask = d > 0
            if not np.any(mask):
                prepass_progress.set_postfix(
                    {
                        "demand_slots": f"{int(slots_with_demand):,}",
                        "edges": f"{len(edge_counts):,}",
                        "rss_gib": _rss_gib_postfix(),
                    }
                )
                prepass_progress.maybe_refresh(counter=int(slots_with_demand))
                continue

            belt_chunk = np.asarray(ds_belt[s0:s1])[mask]
            az_chunk = np.asarray(ds_az[s0:s1])[mask]
            el_chunk = np.asarray(ds_el[s0:s1])[mask]
            raw_slot_idx = (np.nonzero(mask)[0] + s0).astype(np.int64, copy=False)

            if slot_stride > 1:
                m_stride = (raw_slot_idx % int(slot_stride)) == 0
                if not np.any(m_stride):
                    prepass_progress.set_postfix(
                        {
                            "demand_slots": f"{int(slots_with_demand):,}",
                            "edges": f"{len(edge_counts):,}",
                            "rss_gib": _rss_gib_postfix(),
                        }
                    )
                    prepass_progress.maybe_refresh(counter=int(slots_with_demand))
                    continue
                belt_chunk = belt_chunk[m_stride]
                az_chunk = az_chunk[m_stride]
                el_chunk = el_chunk[m_stride]
                raw_slot_idx = raw_slot_idx[m_stride]

            if max_demand_slots > 0:
                remaining = max_demand_slots - slots_with_demand
                if remaining <= 0:
                    stop_early = True
                    break
                if int(belt_chunk.shape[0]) > remaining:
                    keep = int(remaining)
                    belt_chunk = belt_chunk[:keep]
                    az_chunk = az_chunk[:keep]
                    el_chunk = el_chunk[:keep]
                    raw_slot_idx = raw_slot_idx[:keep]
                    stop_early = True

            if belt_chunk.shape[0] == 0:
                prepass_progress.set_postfix(
                    {
                        "demand_slots": f"{int(slots_with_demand):,}",
                        "edges": f"{len(edge_counts):,}",
                        "rss_gib": _rss_gib_postfix(),
                    }
                )
                prepass_progress.maybe_refresh(counter=int(slots_with_demand))
                continue
            slots_with_demand += int(belt_chunk.shape[0])

            views = _build_chunk_visibility_views(
                belt_chunk=belt_chunk,
                az_chunk=az_chunk,
                el_chunk=el_chunk,
                sky_mapper=sky_mapper,
                need_sat_bits=False,
                need_slot_cell=True,
                need_grouped_sats=False,
            )
            if views.slot_valid is None or views.cell_valid is None:
                raise RuntimeError("Internal error: temporal prepass requires slot/cell views.")
            if views.sky_valid.size == 0:
                prepass_progress.set_postfix(
                    {
                        "demand_slots": f"{int(slots_with_demand):,}",
                        "edges": f"{len(edge_counts):,}",
                        "rss_gib": _rss_gib_postfix(),
                    }
                )
                prepass_progress.maybe_refresh(counter=int(slots_with_demand))
                continue

            m_sky = views.sky_valid >= 0
            if not np.any(m_sky):
                prepass_progress.set_postfix(
                    {
                        "demand_slots": f"{int(slots_with_demand):,}",
                        "edges": f"{len(edge_counts):,}",
                        "rss_gib": _rss_gib_postfix(),
                    }
                )
                prepass_progress.maybe_refresh(counter=int(slots_with_demand))
                continue
            triples = np.empty(int(np.count_nonzero(m_sky)), dtype=_SLOT_CELL_BELT_SKY_DTYPE)
            triples["slot"] = views.slot_valid[m_sky]
            triples["cell"] = views.cell_valid[m_sky]
            triples["belt"] = views.belt_valid[m_sky]
            triples["sky"] = views.sky_valid[m_sky]
            uniq = np.unique(triples)
            if uniq.size == 0:
                prepass_progress.set_postfix(
                    {
                        "demand_slots": f"{int(slots_with_demand):,}",
                        "edges": f"{len(edge_counts):,}",
                        "rss_gib": _rss_gib_postfix(),
                    }
                )
                prepass_progress.maybe_refresh(counter=int(slots_with_demand))
                continue

            edge_key_batches: list[np.ndarray] = []
            slot_u = uniq["slot"]
            cell_u = uniq["cell"]
            belt_u = uniq["belt"]
            sky_u = uniq["sky"]
            i0 = 0
            n_u = int(uniq.size)
            while i0 < n_u:
                slot_local = int(slot_u[i0])
                cidx = int(cell_u[i0])
                belt_id = int(belt_u[i0])
                i1 = i0 + 1
                while i1 < n_u and int(slot_u[i1]) == slot_local and int(cell_u[i1]) == cidx and int(belt_u[i1]) == belt_id:
                    i1 += 1

                skies_now = sky_u[i0:i1].astype(np.int32, copy=False)
                if skies_now.size > 0:
                    for s in skies_now:
                        node_seen.add(int(belt_id) * int(n_sky) + int(s))

                    slot_idx = int(raw_slot_idx[slot_local])
                    key = (cidx, belt_id)
                    last_seen = history[key]
                    min_slot = slot_idx - int(window_slots)
                    if last_seen:
                        stale = [sky for sky, seen_slot in last_seen.items() if seen_slot < min_slot]
                        for sky in stale:
                            del last_seen[sky]

                    if last_seen:
                        past_skies = np.fromiter(last_seen.keys(), dtype=np.int32, count=len(last_seen))
                        edge_keys = _build_temporal_edge_keys(
                            belt_id=belt_id,
                            skies_now=skies_now,
                            past_skies=past_skies,
                            n_sky=n_sky,
                        )
                        if edge_keys.size > 0:
                            edge_key_batches.append(edge_keys)
                    for s_now in skies_now:
                        last_seen[int(s_now)] = slot_idx
                i0 = i1

            _merge_temporal_edge_batches(edge_counts, edge_key_batches)

            prepass_progress.set_postfix(
                {
                    "demand_slots": f"{int(slots_with_demand):,}",
                    "edges": f"{len(edge_counts):,}",
                    "rss_gib": _rss_gib_postfix(),
                }
            )
            prepass_progress.maybe_refresh(counter=int(slots_with_demand))
        if stop_early:
            break

    parent: dict[int, int] = {node: node for node in node_seen}
    rank: dict[int, int] = {node: 0 for node in node_seen}

    def _find(x: int) -> int:
        px = parent[x]
        if px != x:
            parent[x] = _find(px)
        return parent[x]

    def _union(a: int, b: int) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra == rb:
            return
        rka = rank[ra]
        rkb = rank[rb]
        if rka < rkb:
            parent[ra] = rb
        elif rka > rkb:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] = rka + 1

    edges_kept = 0
    for edge_key, cnt in edge_counts.items():
        if cnt < int(TEMPORAL_MIN_EDGE_COUNT):
            continue
        tmp = int(edge_key)
        s1 = int(tmp % int(n_sky))
        tmp //= int(n_sky)
        s0 = int(tmp % int(n_sky))
        belt_id = int(tmp // int(n_sky))
        n0 = int(belt_id) * int(n_sky) + int(s0)
        n1 = int(belt_id) * int(n_sky) + int(s1)
        if n0 in parent and n1 in parent:
            _union(n0, n1)
            edges_kept += 1

    comp_map = {node: _find(node) for node in node_seen}
    n_components = len({root for root in comp_map.values()})
    elapsed = time.perf_counter() - t0
    prepass_progress.set_postfix(
        {
            "demand_slots": f"{int(slots_with_demand):,}",
            "edges": f"{len(edge_counts):,}",
            "rss_gib": _rss_gib_postfix(),
        }
    )
    prepass_progress.maybe_refresh(counter=int(slots_with_demand), force=True)
    prepass_progress.close()
    diag = {
        "enabled": 1.0,
        "slots_scanned": float(slots_scanned),
        "slots_with_demand": float(slots_with_demand),
        "slot_stride": float(slot_stride),
        "max_demand_slots": float(max_demand_slots),
        "stopped_early": 1.0 if stop_early else 0.0,
        "numba_available": 1.0 if _HAVE_NUMBA else 0.0,
        "nodes_seen": float(len(node_seen)),
        "edges_seen": float(len(edge_counts)),
        "edges_kept": float(edges_kept),
        "components": float(n_components),
        "elapsed_s": float(elapsed),
    }
    _progress_write(
        "[temporal-prepass] done: "
        f"demand_slots={int(diag['slots_with_demand']):,}, "
        f"nodes={int(diag['nodes_seen']):,}, "
        f"edges_raw={int(diag['edges_seen']):,}, "
        f"edges_kept={int(diag['edges_kept']):,}, "
        f"components={int(diag['components']):,}, "
        f"elapsed={diag['elapsed_s']:.1f}s"
    )
    return comp_map, diag


"""
# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
all_metas = [
    PolicyMeta("simpson", "Simpson pooling", "#1f77b4", "solid"),
    PolicyMeta("full_reroute", "Full reroute (equal split)", "#14b8a6", "dot"),
    PolicyMeta("belt", "Belt-only pooling", "#1f9d55", "dash"),
    PolicyMeta("belt_sky_strict", "Belt+skycell strict", "#f59e0b", "dot"),
    PolicyMeta("belt_sky_temporal", f"Belt+sky temporal ({TEMPORAL_WINDOW_S:g}s)", "#0ea5e9", "dashdot"),
    PolicyMeta("belt_sky_fb", f"Belt+sky fallback (min_sats={BELTSKY_MIN_ACTIVE_SATS})", "#7c3aed", "dash"),
    PolicyMeta("no_reroute", "No reroute (per-satellite)", "#dc2626", "dashdot"),
]
meta_by_key = {m.key: m for m in all_metas}
enabled_keys = []
for k in ENABLED_POLICY_KEYS:
    if k in meta_by_key and k not in enabled_keys:
        enabled_keys.append(k)
unknown_keys = [k for k in ENABLED_POLICY_KEYS if k not in meta_by_key]
if unknown_keys:
    raise ValueError(f"Unknown policy keys in ENABLED_POLICY_KEYS: {unknown_keys}")
if not enabled_keys:
    raise ValueError("ENABLED_POLICY_KEYS is empty after validation.")
metas = [meta_by_key[k] for k in enabled_keys]

need_simpson = "simpson" in enabled_keys
need_full_reroute = "full_reroute" in enabled_keys
need_no_reroute = "no_reroute" in enabled_keys
need_belt = "belt" in enabled_keys
need_strict = "belt_sky_strict" in enabled_keys
need_temporal = "belt_sky_temporal" in enabled_keys
need_fb = "belt_sky_fb" in enabled_keys
need_visibility = need_belt or need_strict or need_temporal or need_fb

acc = {
    m.key: StreamingPolicyAccumulator(
        beam_caps,
        tau=per_slot_loss_tolerance,
        store_slot_samples=STORE_SLOT_RATIO_SAMPLES,
    )
    for m in metas
}

t0 = time.perf_counter()
demand_sum = 0.0
demand_slots = 0
raw_slots_seen = 0
valid_links_total = 0
residual_demand_total = 0.0
missing_context_slots = 0
strict_hotspots: dict[int, float] = defaultdict(float)
max_sky_seen = -1
sky_mappable_links_weight = 0.0
sky_unmappable_links_weight = 0.0
main_demand_slot_cap = _normalize_optional_positive_int(MAX_DEMAND_SLOTS, name="MAX_DEMAND_SLOTS")
stopped_early_main = False
skycell_demand_sum: np.ndarray | None = None
skycell_demand_slot_hits: np.ndarray | None = None

with h5py.File(STORAGE_FILENAME, "r") as h5:
    iter_names = _iter_names(h5)
    if not iter_names:
        raise RuntimeError("No /iter groups found.")
    count_var = _choose_count_var(h5, iter_names, COUNT_VAR_CANDIDATES)
    req_vis = ("sat_belt_id", "sat_azimuth", "sat_elevation")
    n_sky = int(S1586_N_CELLS)
    sky_mapper = None
    if need_visibility:
        if not all(all(v in h5["iter"][it] for v in req_vis) for it in iter_names):
            raise RuntimeError(f"Missing required visibility arrays for selected policies: {req_vis}")
        n_sky, sky_mapper = _build_sky_mapper(SKYCELL_MODE)
        if sky_mapper is None:
            raise RuntimeError("Internal error: failed to initialize sky mapper.")
        if ENABLE_SKYCELL_DEMAND_VIS:
            skycell_demand_sum = np.zeros(n_sky, dtype=np.float64)
            skycell_demand_slot_hits = np.zeros(n_sky, dtype=np.int64)

    raw_total = sum(int(h5["iter"][it][count_var].shape[0]) for it in iter_names)
    print(f"Input file: {STORAGE_FILENAME}")
    print(f"Count source: {count_var}")
    print(f"Enabled policies: {', '.join(enabled_keys)}")
    if need_visibility:
        print(f"S1586 skycell count in mapper: {n_sky}")
    if main_demand_slot_cap is not None:
        print(f"Debug demand-slot cap enabled: {main_demand_slot_cap:,}")
    if need_temporal:
        print(
            f"Temporal edge kernel: {'numba' if _HAVE_NUMBA else 'numpy/python'} "
            f"(numba_min_work={TEMPORAL_NUMBA_MIN_EDGE_WORK})"
        )
    print(f"Iterations: {len(iter_names)} | Raw slots total: {raw_total:,}")
    print(f"Chunk size: {READ_SLOT_CHUNK} slots")

    temporal_comp_map: dict[int, int] = {}
    temporal_diag: dict[str, float] = {"enabled": 0.0}
    if need_temporal and need_visibility and ENABLE_TEMPORAL_SKY_COMPAT:
        if sky_mapper is None:
            raise RuntimeError("Internal error: temporal policy requires sky mapper.")
        temporal_comp_map, temporal_diag = _build_temporal_compatibility_from_h5(
            h5=h5,
            iter_names=iter_names,
            count_var=count_var,
            sky_mapper=sky_mapper,
            n_sky=n_sky,
        )
    temporal_root_lookup = _prepare_temporal_lookup_dense(temporal_comp_map)

    for ii, it in enumerate(iter_names, start=1):
        g = h5["iter"][it]
        ds_count = g[count_var]
        ds_belt = g["sat_belt_id"] if need_visibility else None
        ds_az = g["sat_azimuth"] if need_visibility else None
        ds_el = g["sat_elevation"] if need_visibility else None
        if need_visibility:
            print(f"\n[iter {ii}/{len(iter_names)}] {it}: count_shape={tuple(ds_count.shape)}, link_shape={tuple(ds_belt.shape)}")
        else:
            print(f"\n[iter {ii}/{len(iter_names)}] {it}: count_shape={tuple(ds_count.shape)}")

        T = int(ds_count.shape[0])
        for s0 in range(0, T, READ_SLOT_CHUNK):
            s1 = min(T, s0 + READ_SLOT_CHUNK)
            c = np.asarray(ds_count[s0:s1], dtype=np.int32)
            d = c.sum(axis=1, dtype=np.int64)
            raw_slots_seen += int(c.shape[0])
            mask = d > 0
            if not np.any(mask):
                continue
            c_pos = c[mask]
            d_pos = d[mask].astype(np.float64, copy=False)
            if main_demand_slot_cap is not None:
                remaining = int(main_demand_slot_cap) - int(demand_slots)
                if remaining <= 0:
                    stopped_early_main = True
                    break
                if int(c_pos.shape[0]) > remaining:
                    keep = int(remaining)
                    c_pos = c_pos[:keep]
                    d_pos = d_pos[:keep]
                    stopped_early_main = True
            vis_views: ChunkVisibilityViews | None = None
            if need_visibility:
                if sky_mapper is None:
                    raise RuntimeError("Internal error: visibility policies require sky mapper.")
                belt_pos = np.asarray(ds_belt[s0:s1])[mask]
                az_pos = np.asarray(ds_az[s0:s1])[mask]
                el_pos = np.asarray(ds_el[s0:s1])[mask]
                if main_demand_slot_cap is not None and int(belt_pos.shape[0]) != int(c_pos.shape[0]):
                    keep = int(c_pos.shape[0])
                    belt_pos = belt_pos[:keep]
                    az_pos = az_pos[:keep]
                    el_pos = el_pos[:keep]
                vis_views = _build_chunk_visibility_views(
                    belt_chunk=belt_pos,
                    az_chunk=az_pos,
                    el_chunk=el_pos,
                    sky_mapper=sky_mapper,
                    need_sat_bits=True,
                    need_slot_cell=False,
                    need_grouped_sats=True,
                )

            for k in range(int(c_pos.shape[0])):
                row = c_pos[k]
                d_slot = float(d_pos[k])
                if d_slot <= 0.0:
                    continue
                demand_sum += d_slot
                demand_slots += 1

                need_row_float = need_simpson or need_full_reroute or need_no_reroute
                if need_row_float:
                    row_f = row.astype(np.float64, copy=False)
                    a_slot = float(np.count_nonzero(row_f > 0.0))

                if need_simpson:
                    sum_x2 = float(np.dot(row_f, row_f))
                    aeff = max(min((d_slot * d_slot) / max(sum_x2, 1e-30), a_slot), 1.0)
                    acc["simpson"].add_entries(np.asarray([d_slot]), np.asarray([aeff]))
                    acc["simpson"].add_slot_ratio(d_slot / max(aeff, 1e-30))

                if need_full_reroute:
                    a_pool = max(a_slot, 1.0)
                    acc["full_reroute"].add_entries(np.asarray([d_slot]), np.asarray([a_pool]))
                    acc["full_reroute"].add_slot_ratio(d_slot / max(a_pool, 1e-30))

                if need_no_reroute:
                    x_pos = row_f[row_f > 0]
                    acc["no_reroute"].add_entries(x_pos, np.ones_like(x_pos))
                    acc["no_reroute"].add_slot_ratio(float(np.max(x_pos)) if x_pos.size else 0.0)

                if need_visibility:
                    # reconstruct satellite keys from visibility arrays
                    if vis_views is None:
                        raise RuntimeError("Internal error: missing chunk visibility views.")
                    sat_counts, sat_belt, sat_sky, n_valid, residual, max_sky = _reconstruct_sat_counts_from_views(
                        views=vis_views,
                        slot_local_idx=k,
                        slot_demand=d_slot,
                    )
                    valid_links_total += n_valid
                    residual_demand_total += residual
                    max_sky_seen = max(max_sky_seen, max_sky)
                    if sat_counts.size == 0:
                        missing_context_slots += 1

                    if need_belt:
                        db, pb, rb, _ = _aggregate(sat_counts, sat_belt, d_slot)
                        acc["belt"].add_entries(db, pb)
                        acc["belt"].add_slot_ratio(rb)

                    # Shared strict-core labels if any sky-based policy is enabled.
                    need_any_sky_policy = need_strict or need_temporal or need_fb
                    if need_any_sky_policy:
                        sky_valid = sat_sky >= 0
                        strict_core = sat_belt * np.int64(n_sky) + sat_sky
                        sky_mappable_links_weight += float(np.sum(sat_counts[sky_valid], dtype=np.float64))
                        sky_unmappable_links_weight += float(np.sum(sat_counts[~sky_valid], dtype=np.float64))
                        if skycell_demand_sum is not None and np.any(sky_valid):
                            sky_idx = sat_sky[sky_valid].astype(np.int64, copy=False)
                            np.add.at(skycell_demand_sum, sky_idx, sat_counts[sky_valid])
                            if skycell_demand_slot_hits is not None:
                                sky_unique = np.unique(sky_idx)
                                skycell_demand_slot_hits[sky_unique] += 1

                    if need_strict:
                        strict_lbl = np.asarray(sat_belt, dtype=np.int64).copy()
                        if np.any(sky_valid):
                            strict_lbl[sky_valid] = strict_core[sky_valid] + np.int64(STRICT_GROUP_OFFSET)
                        ds_, ps_, rs_, us_ = _aggregate(sat_counts, strict_lbl, d_slot)
                        acc["belt_sky_strict"].add_entries(ds_, ps_)
                        acc["belt_sky_strict"].add_slot_ratio(rs_)
                        for lbl, val in zip(us_, ds_):
                            strict_hotspots[int(lbl)] += float(val)

                    if need_temporal:
                        # Merge strict sky groups when prepass observed local handovers.
                        temporal_lbl = np.asarray(sat_belt, dtype=np.int64).copy()
                        if np.any(sky_valid):
                            idx_sv = np.nonzero(sky_valid)[0]
                            node_id = sat_belt[idx_sv] * np.int64(n_sky) + sat_sky[idx_sv]
                            root_id = _map_temporal_roots_dense(
                                node_id=node_id,
                                root_lookup=temporal_root_lookup,
                            )
                            temporal_lbl[idx_sv] = root_id + np.int64(TEMPORAL_GROUP_OFFSET)
                        dt_, pt_, rt_, _ = _aggregate(sat_counts, temporal_lbl, d_slot)
                        acc["belt_sky_temporal"].add_entries(dt_, pt_)
                        acc["belt_sky_temporal"].add_slot_ratio(rt_)

                    if need_fb:
                        fb_lbl = _fallback_labels(
                            sat_belt=sat_belt,
                            strict_labels=strict_core,
                            sky_valid_mask=sky_valid,
                            min_active_sats=BELTSKY_MIN_ACTIVE_SATS,
                        )
                        df, pf, rf, _ = _aggregate(sat_counts, fb_lbl, d_slot)
                        acc["belt_sky_fb"].add_entries(df, pf)
                        acc["belt_sky_fb"].add_slot_ratio(rf)

                if demand_slots % PROGRESS_EVERY_DEMAND_SLOTS == 0:
                    elapsed = time.perf_counter() - t0
                    rate = demand_slots / max(elapsed, 1e-9)
                    print(
                        f"[progress] demand_slots={demand_slots:,}, raw_slots_seen={raw_slots_seen:,}, "
                        f"rate={rate:,.1f} slots/s, demand_sum={int(round(demand_sum)):,}"
                    )
            if stopped_early_main:
                break
        if stopped_early_main:
            break

elapsed = time.perf_counter() - t0
print("\nStreaming pass completed.")
print(f"Elapsed: {elapsed:.1f} s")
print(f"Raw slots seen: {raw_slots_seen:,}")
print(f"Demand slots processed: {demand_slots:,}")
if stopped_early_main and main_demand_slot_cap is not None:
    print(f"[debug-cap] Stopped early after reaching MAX_DEMAND_SLOTS={main_demand_slot_cap:,}.")
if np.isfinite(_timestep_s):
    print(f"Retained-slot simulated duration: {demand_slots * _timestep_s:.1f} s")
print(f"Total demand: {int(round(demand_sum)):,}")
if need_visibility:
    print(f"Total valid links in visibility arrays: {valid_links_total:,}")
    print(f"Residual demand (missing/invalid context): {int(round(residual_demand_total)):,}")
    print(f"Slots with fully missing visibility context: {missing_context_slots:,}")
    print(f"Observed max skycell ID: {max_sky_seen} (expected < {S1586_N_CELLS})")
    if max_sky_seen >= S1586_N_CELLS:
        print("[WARN] Observed skycell ID exceeds configured S1586_N_CELLS.")
    sky_total = sky_mappable_links_weight + sky_unmappable_links_weight
    if sky_total > 0.0:
        print(
            "Skycell mapping coverage (link-weighted over reconstructed per-sat counts): "
            f"mappable={100.0 * sky_mappable_links_weight / sky_total:.2f}%, "
            f"unmappable={100.0 * sky_unmappable_links_weight / sky_total:.2f}%"
        )
if float(temporal_diag.get("enabled", 0.0)) > 0.5:
    print(
        "Temporal compatibility prepass summary: "
        f"demand_slots={int(temporal_diag.get('slots_with_demand', 0.0)):,}, "
        f"nodes={int(temporal_diag.get('nodes_seen', 0.0)):,}, "
        f"edges_raw={int(temporal_diag.get('edges_seen', 0.0)):,}, "
        f"edges_kept={int(temporal_diag.get('edges_kept', 0.0)):,}, "
        f"components={int(temporal_diag.get('components', 0.0)):,}, "
        f"window={TEMPORAL_WINDOW_S:g}s, "
        f"stride={int(temporal_diag.get('slot_stride', 1.0))}"
    )

evals: list[PolicyEval] = []
for m in metas:
    delta, eps, ccdf_exceed, b_sel, idx = acc[m.key].finalize(demand_sum, lost_demand_target, loss_slot_target)
    slot_samples: np.ndarray | None = None
    if acc[m.key].slot_max_samples is not None:
        slot_samples = np.asarray(acc[m.key].slot_max_samples, dtype=np.float64)
    evals.append(PolicyEval(m.key, m.label, m.color, m.dash, delta, eps, ccdf_exceed, b_sel, idx, slot_samples))

print("\nSelected beam caps per policy:")
for pe in evals:
    i = pe.selected_idx
    print(f"  {pe.label:42s} B={pe.selected_b:3d} | delta={pe.delta_run[i]:.3e} | eps={pe.eps_run[i]:.3e}")

# Sanity check: no-reroute should typically be the strictest (largest selected B).
if need_no_reroute:
    no_reroute_b = next(pe.selected_b for pe in evals if pe.key == "no_reroute")
    for pe in evals:
        if pe.key != "no_reroute" and pe.selected_b > no_reroute_b:
            print(
                f"[WARN] Policy '{pe.label}' selected B={pe.selected_b} > no-reroute B={no_reroute_b}. "
                "This usually indicates overly conservative grouping or context mismatch."
            )

tot_hot = float(sum(strict_hotspots.values()))
if tot_hot > 0.0:
    print("\nTop strict (belt,skycell) groups by accumulated demand:")
    for gid, val in sorted(strict_hotspots.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        if gid < 0:
            kind = "unknown"
            b = -1
            s = -1
        elif gid >= STRICT_GROUP_OFFSET:
            core = int(gid - STRICT_GROUP_OFFSET)
            kind = "strict"
            b = int(core // n_sky)
            s = int(core % n_sky)
        else:
            kind = "belt-fallback"
            b = int(gid)
            s = -1
        share = 100.0 * float(val) / tot_hot
        print(
            f"  kind={kind:13s} belt={b:2d}, skycell={s:4d}: "
            f"demand={int(round(val)):10d} ({share:5.2f}%)"
        )

tau = float(per_slot_loss_tolerance)
ccdf_series = [np.asarray(pe.ccdf_exceed_percent, dtype=np.float64) for pe in evals]
b_focus = int(max(pe.selected_b for pe in evals))
xmax_focus = compute_xmax_for_focus(b_focus, focus_at=FOCUS_AT, x_min=beam_cap_min, x_max_cap=beam_cap_max, extra_right_margin=2, min_span=15)
xmax = ccdf_tail_guard_xmax(beam_caps, ccdf_series, CCDF_TAIL_GUARD_PERCENT, xmax_focus, beam_cap_max)
x_range = [beam_cap_min, int(xmax)]
print(f"\nAuto-zoom X range: [{x_range[0]}, {x_range[1]}], focus B={b_focus}, tail guard={CCDF_TAIL_GUARD_PERCENT:g}%")


def _maybe_render_skycell_demand_hemisphere(run_dir: Path | None) -> None:
    if not ENABLE_SKYCELL_DEMAND_VIS:
        return
    if not need_visibility:
        _progress_write("[skycell-vis] Skipped: visibility-aware data not enabled for this run.")
        return
    if skycell_demand_sum is None:
        _progress_write("[skycell-vis] Skipped: no skycell demand accumulator available.")
        return
    if skycell_demand_sum.size != 2334:
        _progress_write(
            "[skycell-vis] Skipped: visualise hemisphere helpers currently expect 2334 cells "
            f"(got {skycell_demand_sum.size})."
        )
        return
    if (not _HAVE_VISUALISE) or (plot_hemisphere_2D is None) or (plot_hemisphere_3D is None):
        _progress_write("[skycell-vis] Skipped: scepter.visualise hemisphere plotting functions are unavailable.")
        return

    vals = np.asarray(skycell_demand_sum, dtype=np.float64)
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        _progress_write("[skycell-vis] Skipped: skycell demand data is empty or non-finite.")
        return
    vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), 0.0)
    mode = str(SKYCELL_DEMAND_NORMALIZE_MODE).lower().strip()
    if mode == "sum":
        denom = float(np.sum(vals, dtype=np.float64))
    elif mode == "max":
        denom = float(np.max(vals)) if vals.size else 0.0
    elif mode == "none":
        denom = 1.0
    else:
        raise ValueError("SKYCELL_DEMAND_NORMALIZE_MODE must be one of: 'sum', 'max', 'none'.")
    if denom <= 0.0:
        _progress_write("[skycell-vis] Skipped: cannot normalize relative demand (zero denominator).")
        return

    vals_rel = vals / denom
    if mode == "none":
        value_unit = "raw"
    elif SKYCELL_DEMAND_NORMALIZE_TO_PERCENT:
        vals_rel = vals_rel * 100.0
        value_unit = "%"
    else:
        value_unit = "share"

    data = vals_rel.reshape(1, -1)
    nonzero = int(np.count_nonzero(vals_rel > 0.0))
    if skycell_demand_slot_hits is not None:
        hits = np.asarray(skycell_demand_slot_hits, dtype=np.int64)
        active = hits > 0
        mean_on_active = float(np.mean(vals_rel[active])) if np.any(active) else 0.0
        _progress_write(
            "[skycell-vis] Rendering per-skycell demand maps "
            f"(mode={mode}, nonzero cells={nonzero:,}/{vals_rel.size:,}, "
            f"mean_on_active={mean_on_active:.3f} {value_unit})."
        )
    else:
        _progress_write(
            "[skycell-vis] Rendering per-skycell demand maps "
            f"(mode={mode}, nonzero cells={nonzero:,}/{vals_rel.size:,})."
        )

    save2d = None
    save3d = None
    if SKYCELL_DEMAND_VIS_SAVE_HTML and run_dir is not None:
        save2d = str(run_dir / "skycell_beam_demand_2d.html")
        save3d = str(run_dir / "skycell_beam_demand_3d.html")

    if mode == "sum":
        mode_desc = "share of total demand"
    elif mode == "max":
        mode_desc = "relative to max-cell demand"
    else:
        mode_desc = "raw demand (no normalization)"
    title_2d = (
        "Per-skycell relative demand "
        f"({mode_desc}, unit={value_unit}, slots={demand_slots:,})"
    )
    title_3d = (
        "Per-skycell relative demand 3D "
        f"({mode_desc}, unit={value_unit}, slots={demand_slots:,})"
    )

    try:
        plot_hemisphere_2D(
            data=data,
            mode="power",
            worst_percent=50.0,
            log_mode=False,
            cell_axis=-1,
            cmap="turbo",
            vmin=0.0,
            vmax=float(np.max(vals_rel)) if np.any(np.isfinite(vals_rel)) else 1.0,
            title=title_2d,
            projection=SKYCELL_DEMAND_VIS_2D_PROJECTION,
            engine=SKYCELL_DEMAND_VIS_ENGINE,
            show=SKYCELL_DEMAND_VIS_SHOW,
            save_html=save2d,
        )
        if save2d is not None:
            _progress_write(f"[skycell-vis] Saved 2D hemisphere HTML: {save2d}")
    except Exception as exc:
        _progress_write(f"[skycell-vis] 2D hemisphere plot failed: {exc}")

    try:
        plot_hemisphere_3D(
            data=data,
            mode="power",
            worst_percent=50.0,
            log_mode=False,
            cell_axis=-1,
            cmap="turbo",
            vmin=0.0,
            vmax=float(np.max(vals_rel)) if np.any(np.isfinite(vals_rel)) else 1.0,
            title=title_3d,
            engine=SKYCELL_DEMAND_VIS_ENGINE,
            show=SKYCELL_DEMAND_VIS_SHOW,
            export_html_path=save3d,
        )
        if save3d is not None:
            _progress_write(f"[skycell-vis] Saved 3D hemisphere HTML: {save3d}")
    except Exception as exc:
        _progress_write(f"[skycell-vis] 3D hemisphere plot failed: {exc}")


def _make_output_run_dir(base_dir: str, prefix: str) -> Path:
    out_root = Path(base_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{prefix}_{stamp}"
    idx = 1
    while run_dir.exists():
        run_dir = out_root / f"{prefix}_{stamp}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_outputs(
    fig_tail: go.Figure,
    fig_delta: go.Figure,
    fig_eps: go.Figure,
) -> Path | None:
    if not SAVE_OUTPUTS:
        return None
    run_dir = _make_output_run_dir(OUTPUT_DIR, OUTPUT_PREFIX)

    if SAVE_PLOTS_HTML:
        fig_tail.write_html(str(run_dir / "tail_risk.html"), include_plotlyjs="cdn")
        fig_delta.write_html(str(run_dir / "sla_delta.html"), include_plotlyjs="cdn")
        fig_eps.write_html(str(run_dir / "sla_eps.html"), include_plotlyjs="cdn")

    policy_rows: list[dict[str, object]] = []
    for pe in evals:
        i = pe.selected_idx
        policy_rows.append(
            {
                "key": pe.key,
                "label": pe.label,
                "selected_b": int(pe.selected_b),
                "delta_at_selected": float(pe.delta_run[i]),
                "eps_at_selected": float(pe.eps_run[i]),
                "tail_risk_percent_at_selected": float(pe.ccdf_exceed_percent[i]),
            }
        )

    if SAVE_RESULTS_JSON:
        payload = {
            "storage_filename": STORAGE_FILENAME,
            "count_var_candidates": list(COUNT_VAR_CANDIDATES),
            "enabled_policy_keys": list(enabled_keys),
            "targets": {
                "lost_demand_target": float(lost_demand_target),
                "loss_slot_target": float(loss_slot_target),
                "per_slot_loss_tolerance": float(per_slot_loss_tolerance),
            },
            "beam_caps": {
                "min": int(beam_cap_min),
                "max": int(beam_cap_max),
                "selected_x_range": [int(x_range[0]), int(x_range[1])],
            },
            "run_diagnostics": {
                "elapsed_s": float(elapsed),
                "raw_slots_seen": int(raw_slots_seen),
                "demand_slots": int(demand_slots),
                "max_demand_slots_cap": None if main_demand_slot_cap is None else int(main_demand_slot_cap),
                "stopped_early_main": bool(stopped_early_main),
                "demand_sum": float(demand_sum),
                "need_visibility": bool(need_visibility),
                "valid_links_total": int(valid_links_total),
                "residual_demand_total": float(residual_demand_total),
                "missing_context_slots": int(missing_context_slots),
                "max_sky_seen": int(max_sky_seen),
            },
            "temporal_diag": {k: float(v) for k, v in temporal_diag.items()},
            "policies": policy_rows,
        }
        with (run_dir / "results_summary.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if SAVE_RESULTS_NPZ:
        npz_payload: dict[str, np.ndarray] = {
            "beam_caps": np.asarray(beam_caps, dtype=np.int32),
            "x_range": np.asarray(x_range, dtype=np.int32),
        }
        for pe in evals:
            npz_payload[f"{pe.key}_delta"] = np.asarray(pe.delta_run, dtype=np.float64)
            npz_payload[f"{pe.key}_eps"] = np.asarray(pe.eps_run, dtype=np.float64)
            npz_payload[f"{pe.key}_tail_risk_percent"] = np.asarray(pe.ccdf_exceed_percent, dtype=np.float64)
        np.savez_compressed(run_dir / "results_curves.npz", **npz_payload)

    print(f"Saved outputs to: {run_dir}")
    return run_dir


def plot_tail_risk(show: bool = True) -> go.Figure:
    fig = go.Figure()
    legend_rows = 2 if len(evals) <= 4 else 3 if len(evals) <= 6 else 4
    for pe in evals:
        raw = np.asarray(pe.ccdf_exceed_percent, dtype=np.float64)
        fig.add_trace(go.Scatter(
            x=beam_caps, y=np.clip(raw, CCDF_Y_FLOOR_PERCENT, None), customdata=raw, mode="lines",
            line=dict(width=3, color=pe.color, dash=pe.dash), name=policy_display_name(pe.key, pe.label),
            hovertemplate="B=%{x}<br>P(B_req > B)=%{customdata:.6f}%<extra></extra>",
        ))
        fig.add_vline(x=pe.selected_b, line_width=1, line_dash=pe.dash, line_color=pe.color, opacity=0.65)
    fig.update_layout(template="simple_white", width=1220, height=560, font=dict(family="Times New Roman", size=18),
                      margin=dict(l=80, r=390, t=90, b=170),
                      title=dict(text=f"Per-slot tail risk by policy<br><sup>P(B_req > B), tau={tau:g}</sup>", x=0.02))
    fig.update_xaxes(title_text="Beam cap per satellite, B", range=x_range, title_standoff=0)
    style_axes_with_grid(fig)
    place_legend_below(fig, n_rows=legend_rows)
    apply_log_y(fig, CCDF_Y_FLOOR_PERCENT, 100.0, "Tail risk [%]")
    add_value_box(fig, lines=build_policy_value_lines(evals))
    if show:
        fig.show()
    return fig


def plot_sla(metric: str, target: float, show: bool = True) -> go.Figure:
    m = str(metric).lower().strip()
    if m not in {"delta", "eps"}:
        raise ValueError("metric must be 'delta' or 'eps'.")
    fig = go.Figure()
    legend_rows = 2 if len(evals) <= 4 else 3 if len(evals) <= 6 else 4
    y_max = 100.0 * float(target)
    for pe in evals:
        raw = 100.0 * (pe.delta_run if m == "delta" else pe.eps_run)
        y_max = max(y_max, float(np.max(raw)))
        metric_name = "delta" if m == "delta" else "epsilon"
        fig.add_trace(go.Scatter(
            x=beam_caps, y=np.clip(raw, SLA_Y_FLOOR_PERCENT, None), customdata=raw, mode="lines",
            line=dict(width=3, color=pe.color, dash=pe.dash),
            name=policy_display_name(pe.key, pe.label),
            hovertemplate=f"B=%{{x}}<br>{metric_name}=%{{customdata:.8f}}%<extra></extra>",
        ))
        fig.add_vline(x=pe.selected_b, line_width=1, line_dash=pe.dash, line_color=pe.color, opacity=0.65)
    fig.add_hline(y=100.0 * float(target), line_width=2, line_dash="dot", opacity=0.85)
    title = "Lost-demand fraction (delta)" if m == "delta" else "Loss-slot probability (epsilon)"
    fig.update_layout(template="simple_white", width=1220, height=560, font=dict(family="Times New Roman", size=18),
                      margin=dict(l=80, r=390, t=90, b=170),
                      title=dict(text=f"Run-level {title} vs B<br><sup>target={target:g}</sup>", x=0.02))
    fig.update_xaxes(title_text="Beam cap per satellite, B", range=x_range, title_standoff=0)
    style_axes_with_grid(fig)
    place_legend_below(fig, n_rows=legend_rows)
    apply_log_y(fig, SLA_Y_FLOOR_PERCENT, y_max, f"{title} [%]")
    add_value_box(fig, lines=build_policy_value_lines(evals))
    if show:
        fig.show()
    return fig


fig_tail = plot_tail_risk(show=SHOW_PLOTS)
fig_delta = plot_sla("delta", lost_demand_target, show=SHOW_PLOTS)
fig_eps = plot_sla("eps", loss_slot_target, show=SHOW_PLOTS)
output_run_dir = _save_outputs(fig_tail, fig_delta, fig_eps)
_maybe_render_skycell_demand_hemisphere(output_run_dir)
"""


def _plot_tail_risk(
    evals: list[PolicyEval],
    *,
    x_grid: np.ndarray,
    x_range: list[int],
    tau: float,
    show: bool,
) -> go.Figure:
    fig = go.Figure()
    legend_rows = 2 if len(evals) <= 4 else 3 if len(evals) <= 6 else 4
    for pe in evals:
        raw = np.asarray(pe.ccdf_exceed_percent, dtype=np.float64)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=np.clip(raw, CCDF_Y_FLOOR_PERCENT, None),
                customdata=raw,
                mode="lines",
                line=dict(width=3, color=pe.color, dash=pe.dash),
                name=policy_display_name(pe.key, pe.label),
                hovertemplate=(
                    "B=%{x}<br>"
                    "tail risk: slots needing larger B = %{customdata:.6f}%"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=pe.selected_b, line_width=1, line_dash=pe.dash, line_color=pe.color, opacity=0.65)
    fig.update_layout(
        template="simple_white",
        width=1220,
        height=560,
        font=dict(family="Times New Roman", size=18),
        margin=dict(l=80, r=390, t=90, b=170),
        title=dict(
            text=(
                "Tail risk vs beam cap by policy"
                f"<br><sup>{_TAIL_RISK_GLOSSARY_TEXT}; tau={tau:g}</sup>"
            ),
            x=0.02,
        ),
    )
    fig.update_xaxes(title_text="Beam cap per satellite, B", range=x_range, title_standoff=0)
    style_axes_with_grid(fig)
    place_legend_below(fig, n_rows=legend_rows)
    apply_log_y(fig, CCDF_Y_FLOOR_PERCENT, 100.0, "Tail risk [% of processed slots]")
    add_value_box(fig, lines=build_policy_value_lines(evals))
    if show:
        fig.show()
    return fig


def _plot_sla(
    evals: list[PolicyEval],
    *,
    x_grid: np.ndarray,
    x_range: list[int],
    metric: str,
    target: float,
    show: bool,
) -> go.Figure:
    m = str(metric).lower().strip()
    if m not in {"delta", "eps"}:
        raise ValueError("metric must be 'delta' or 'eps'.")
    fig = go.Figure()
    legend_rows = 2 if len(evals) <= 4 else 3 if len(evals) <= 6 else 4
    y_max = 100.0 * float(target)
    for pe in evals:
        raw = 100.0 * (pe.delta_run if m == "delta" else pe.eps_run)
        y_max = max(y_max, float(np.max(raw)))
        metric_label = (
            "delta: unserved demand share"
            if m == "delta"
            else "epsilon: failed-demand-slot share"
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=np.clip(raw, SLA_Y_FLOOR_PERCENT, None),
                customdata=raw,
                mode="lines",
                line=dict(width=3, color=pe.color, dash=pe.dash),
                name=policy_display_name(pe.key, pe.label),
                hovertemplate=f"B=%{{x}}<br>{metric_label}=%{{customdata:.8f}}%<extra></extra>",
            )
        )
        fig.add_vline(x=pe.selected_b, line_width=1, line_dash=pe.dash, line_color=pe.color, opacity=0.65)
    fig.add_hline(y=100.0 * float(target), line_width=2, line_dash="dot", opacity=0.85)
    title = (
        "Run-level unmet-demand fraction"
        if m == "delta"
        else "Run-level failed-demand-slot fraction"
    )
    subtitle = _DELTA_GLOSSARY_TEXT if m == "delta" else _EPS_GLOSSARY_TEXT
    fig.update_layout(
        template="simple_white",
        width=1220,
        height=560,
        font=dict(family="Times New Roman", size=18),
        margin=dict(l=80, r=390, t=90, b=170),
        title=dict(text=f"{title} vs B<br><sup>{subtitle}; target={target:g}</sup>", x=0.02),
    )
    fig.update_xaxes(title_text="Beam cap per satellite, B", range=x_range, title_standoff=0)
    style_axes_with_grid(fig)
    place_legend_below(fig, n_rows=legend_rows)
    apply_log_y(fig, SLA_Y_FLOOR_PERCENT, y_max, f"{title} [%]")
    add_value_box(fig, lines=build_policy_value_lines(evals))
    if show:
        fig.show()
    return fig


def _make_output_run_dir(base_dir: str | Path, prefix: str) -> Path:
    out_root = Path(base_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{prefix}_{stamp}"
    idx = 1
    while run_dir.exists():
        run_dir = out_root / f"{prefix}_{stamp}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_policy_evals(
    *,
    metas: list[PolicyMeta],
    acc: dict[str, StreamingPolicyAccumulator],
    beam_caps_grid: np.ndarray,
    demand_sum: float,
    pure_demand_sum: float,
    pure_demand_slots: int,
    pure_matched_links_total: np.ndarray,
    pure_fail_slots: np.ndarray,
    pure_required_cap_values: list[np.ndarray],
) -> list[PolicyEval]:
    evals_by_key: dict[str, PolicyEval] = {}
    for meta in metas:
        if meta.key == "pure_reroute":
            if pure_demand_sum > 0.0:
                delta = np.maximum(pure_demand_sum - pure_matched_links_total.astype(np.float64), 0.0) / pure_demand_sum
            else:
                delta = np.zeros(beam_caps_grid.size, dtype=np.float64)
            if pure_demand_slots > 0:
                eps = pure_fail_slots.astype(np.float64) / float(pure_demand_slots)
            else:
                eps = np.zeros(beam_caps_grid.size, dtype=np.float64)
            ccdf_exceed = eps * 100.0
            b_sel = choose_smallest_beam_cap(
                beam_caps_grid.astype(np.int32, copy=False),
                delta,
                eps,
                lost_demand_target,
                loss_slot_target,
            )
            idx = int(np.where(beam_caps_grid.astype(np.int32, copy=False) == b_sel)[0][0])
            slot_samples: np.ndarray | None = None
            if STORE_SLOT_RATIO_SAMPLES and pure_required_cap_values:
                slot_samples = np.concatenate(pure_required_cap_values).astype(np.float64, copy=False)
            evals_by_key[meta.key] = PolicyEval(
                meta.key,
                meta.label,
                meta.color,
                meta.dash,
                delta,
                eps,
                ccdf_exceed,
                b_sel,
                idx,
                slot_samples,
            )
            continue

        delta, eps, ccdf_exceed, b_sel, idx = acc[meta.key].finalize(
            demand_sum,
            lost_demand_target,
            loss_slot_target,
        )
        slot_samples = None
        if acc[meta.key].slot_max_samples is not None:
            slot_samples = np.asarray(acc[meta.key].slot_max_samples, dtype=np.float64)
        evals_by_key[meta.key] = PolicyEval(
            meta.key,
            meta.label,
            meta.color,
            meta.dash,
            delta,
            eps,
            ccdf_exceed,
            b_sel,
            idx,
            slot_samples,
        )
    return [evals_by_key[meta.key] for meta in metas]


def _compute_output_x_range(evals: list[PolicyEval], *, x_grid: np.ndarray) -> list[int]:
    ccdf_series = [np.asarray(pe.ccdf_exceed_percent, dtype=np.float64) for pe in evals]
    b_focus = int(max(pe.selected_b for pe in evals))
    xmax_focus = compute_xmax_for_focus(
        b_focus,
        focus_at=FOCUS_AT,
        x_min=int(x_grid[0]),
        x_max_cap=int(x_grid[-1]),
        extra_right_margin=2,
        min_span=15,
    )
    xmax = ccdf_tail_guard_xmax(x_grid, ccdf_series, CCDF_TAIL_GUARD_PERCENT, xmax_focus, int(x_grid[-1]))
    return [int(x_grid[0]), int(xmax)]


def _build_policy_rows(evals: list[PolicyEval]) -> list[dict[str, object]]:
    policy_rows: list[dict[str, object]] = []
    for pe in evals:
        i = pe.selected_idx
        policy_rows.append(
            {
                "key": pe.key,
                "label": pe.label,
                "selected_b": int(pe.selected_b),
                "delta_at_selected": float(pe.delta_run[i]),
                "eps_at_selected": float(pe.eps_run[i]),
                "tail_risk_percent_at_selected": float(pe.ccdf_exceed_percent[i]),
            }
        )
    return policy_rows

def _build_snapshot_status(
    *,
    is_final: bool,
    raw_slots_seen: int,
    raw_slots_total: int,
    demand_slots: int,
) -> dict[str, Any]:
    progress_fraction = float(raw_slots_seen) / float(raw_slots_total) if raw_slots_total > 0 else 0.0
    return {
        "is_final": bool(is_final),
        "is_partial": not bool(is_final),
        "raw_slots_seen": int(raw_slots_seen),
        "raw_slots_total": int(raw_slots_total),
        "progress_fraction": float(np.clip(progress_fraction, 0.0, 1.0)),
        "demand_slots": int(demand_slots),
    }


def _build_summary_payload(
    *,
    enabled_keys: list[str],
    x_grid: np.ndarray,
    x_range: list[int],
    run_diagnostics: dict[str, Any],
    temporal_diag: dict[str, float],
    extra_summary: dict[str, Any],
    snapshot_status: dict[str, Any],
    evals: list[PolicyEval],
) -> dict[str, Any]:
    return {
        "storage_filename": STORAGE_FILENAME,
        "count_var_candidates": list(COUNT_VAR_CANDIDATES),
        "enabled_policy_keys": list(enabled_keys),
        "targets": {
            "lost_demand_target": float(lost_demand_target),
            "loss_slot_target": float(loss_slot_target),
            "per_slot_loss_tolerance": float(per_slot_loss_tolerance),
        },
        "beam_caps": {
            "min": int(x_grid[0]),
            "max": int(x_grid[-1]),
            "selected_x_range": [int(x_range[0]), int(x_range[1])],
        },
        "snapshot": snapshot_status,
        "metric_glossary": _metric_glossary_payload(),
        "run_diagnostics": run_diagnostics,
        "temporal_diag": {k: float(v) for k, v in temporal_diag.items()},
        "extra_summary": extra_summary,
        "policies": _build_policy_rows(evals),
    }


def _build_snapshot_figures(
    *,
    evals: list[PolicyEval],
    x_grid: np.ndarray,
    x_range: list[int],
    tau: float,
    show: bool,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    fig_tail = _plot_tail_risk(evals, x_grid=x_grid, x_range=x_range, tau=tau, show=show)
    fig_delta = _plot_sla(evals, x_grid=x_grid, x_range=x_range, metric="delta", target=lost_demand_target, show=show)
    fig_eps = _plot_sla(evals, x_grid=x_grid, x_range=x_range, metric="eps", target=loss_slot_target, show=show)
    return fig_tail, fig_delta, fig_eps


def _write_snapshot_artifacts(
    *,
    run_dir: Path,
    fig_tail: go.Figure | None,
    fig_delta: go.Figure | None,
    fig_eps: go.Figure | None,
    save_html: bool,
    save_json: bool,
    save_npz: bool,
    x_grid: np.ndarray,
    x_range: list[int],
    evals: list[PolicyEval],
    summary_payload: dict[str, Any],
    is_final: bool,
) -> None:
    suffix = "" if is_final else ".interim"

    if save_html:
        if fig_tail is None or fig_delta is None or fig_eps is None:
            raise RuntimeError("HTML output requires built figures.")
        fig_tail.write_html(str(run_dir / f"tail_risk{suffix}.html"), include_plotlyjs="cdn")
        fig_delta.write_html(str(run_dir / f"sla_delta{suffix}.html"), include_plotlyjs="cdn")
        fig_eps.write_html(str(run_dir / f"sla_eps{suffix}.html"), include_plotlyjs="cdn")

    if save_json:
        with (run_dir / f"results_summary{suffix}.json").open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

    if save_npz:
        npz_payload: dict[str, np.ndarray] = {
            "beam_caps": np.asarray(x_grid, dtype=np.int32),
            "x_range": np.asarray(x_range, dtype=np.int32),
        }
        for pe in evals:
            npz_payload[f"{pe.key}_delta"] = np.asarray(pe.delta_run, dtype=np.float64)
            npz_payload[f"{pe.key}_eps"] = np.asarray(pe.eps_run, dtype=np.float64)
            npz_payload[f"{pe.key}_tail_risk_percent"] = np.asarray(pe.ccdf_exceed_percent, dtype=np.float64)
        np.savez_compressed(run_dir / "results_curves.npz", **npz_payload)


def _maybe_render_skycell_demand_hemisphere_runtime(
    *,
    run_dir: Path | None,
    need_visibility: bool,
    skycell_demand_sum: np.ndarray | None,
    skycell_demand_slot_hits: np.ndarray | None,
    demand_slots: int,
    file_suffix: str = "",
) -> None:
    if not ENABLE_SKYCELL_DEMAND_VIS:
        return
    if not need_visibility:
        _progress_write("[skycell-vis] Skipped: visibility-aware data not enabled for this run.")
        return
    if skycell_demand_sum is None:
        _progress_write("[skycell-vis] Skipped: no skycell demand accumulator available.")
        return
    if skycell_demand_sum.size != 2334:
        _progress_write(
            "[skycell-vis] Skipped: visualise hemisphere helpers currently expect 2334 cells "
            f"(got {skycell_demand_sum.size})."
        )
        return
    if (not _HAVE_VISUALISE) or (plot_hemisphere_2D is None) or (plot_hemisphere_3D is None):
        _progress_write("[skycell-vis] Skipped: scepter.visualise hemisphere plotting functions are unavailable.")
        return

    vals = np.asarray(skycell_demand_sum, dtype=np.float64)
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        _progress_write("[skycell-vis] Skipped: skycell demand data is empty or non-finite.")
        return
    vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), 0.0)
    mode = str(SKYCELL_DEMAND_NORMALIZE_MODE).lower().strip()
    if mode == "sum":
        denom = float(np.sum(vals, dtype=np.float64))
    elif mode == "max":
        denom = float(np.max(vals)) if vals.size else 0.0
    elif mode == "none":
        denom = 1.0
    else:
        raise ValueError("SKYCELL_DEMAND_NORMALIZE_MODE must be one of: 'sum', 'max', 'none'.")
    if denom <= 0.0:
        _progress_write("[skycell-vis] Skipped: cannot normalize relative demand (zero denominator).")
        return

    vals_rel = vals / denom
    if mode == "none":
        value_unit = "raw"
    elif SKYCELL_DEMAND_NORMALIZE_TO_PERCENT:
        vals_rel = vals_rel * 100.0
        value_unit = "%"
    else:
        value_unit = "share"

    data = vals_rel.reshape(1, -1)
    nonzero = int(np.count_nonzero(vals_rel > 0.0))
    if skycell_demand_slot_hits is not None:
        hits = np.asarray(skycell_demand_slot_hits, dtype=np.int64)
        active = hits > 0
        mean_on_active = float(np.mean(vals_rel[active])) if np.any(active) else 0.0
        _progress_write(
            "[skycell-vis] Rendering per-skycell demand maps "
            f"(mode={mode}, nonzero cells={nonzero:,}/{vals_rel.size:,}, "
            f"mean_on_active={mean_on_active:.3f} {value_unit})."
        )
    else:
        _progress_write(
            "[skycell-vis] Rendering per-skycell demand maps "
            f"(mode={mode}, nonzero cells={nonzero:,}/{vals_rel.size:,})."
        )

    save2d = None
    save3d = None
    if SKYCELL_DEMAND_VIS_SAVE_HTML and run_dir is not None:
        save2d = str(run_dir / f"skycell_beam_demand_2d{file_suffix}.html")
        save3d = str(run_dir / f"skycell_beam_demand_3d{file_suffix}.html")

    if mode == "sum":
        mode_desc = "share of total demand"
    elif mode == "max":
        mode_desc = "relative to max-cell demand"
    else:
        mode_desc = "raw demand (no normalization)"
    title_2d = f"Per-skycell relative demand ({mode_desc}, unit={value_unit}, slots={demand_slots:,})"
    title_3d = f"Per-skycell relative demand 3D ({mode_desc}, unit={value_unit}, slots={demand_slots:,})"

    try:
        plot_hemisphere_2D(
            data=data,
            mode="power",
            worst_percent=50.0,
            log_mode=False,
            cell_axis=-1,
            cmap="turbo",
            vmin=0.0,
            vmax=float(np.max(vals_rel)) if np.any(np.isfinite(vals_rel)) else 1.0,
            title=title_2d,
            projection=SKYCELL_DEMAND_VIS_2D_PROJECTION,
            engine=SKYCELL_DEMAND_VIS_ENGINE,
            show=SKYCELL_DEMAND_VIS_SHOW,
            save_html=save2d,
        )
        if save2d is not None:
            _progress_write(f"[skycell-vis] Saved 2D hemisphere HTML: {save2d}")
    except Exception as exc:
        _progress_write(f"[skycell-vis] 2D hemisphere plot failed: {exc}")

    try:
        plot_hemisphere_3D(
            data=data,
            mode="power",
            worst_percent=50.0,
            log_mode=False,
            cell_axis=-1,
            cmap="turbo",
            vmin=0.0,
            vmax=float(np.max(vals_rel)) if np.any(np.isfinite(vals_rel)) else 1.0,
            title=title_3d,
            engine=SKYCELL_DEMAND_VIS_ENGINE,
            show=SKYCELL_DEMAND_VIS_SHOW,
            export_html_path=save3d,
        )
        if save3d is not None:
            _progress_write(f"[skycell-vis] Saved 3D hemisphere HTML: {save3d}")
    except Exception as exc:
        _progress_write(f"[skycell-vis] 3D hemisphere plot failed: {exc}")


def run_beam_cap_sizing(
    storage_filename: str | Path,
    *,
    config: BeamCapSizingConfig | None = None,
    group_prefix: str = "",
    **overrides: Any,
) -> dict[str, Any]:
    """
    Run the reusable beam-cap sizing pipeline on a streamed HDF5 result file.

    Parameters
    ----------
    storage_filename : str or pathlib.Path
        Input HDF5 file containing ``/iter/iter_*`` groups. Legacy count-based
        policies read the configured count dataset and optional visibility
        arrays. ``pure_reroute`` reads either the legacy dense
        ``sat_eligible_mask`` dataset or the preferred sparse CSR datasets
        ``sat_eligible_csr_row_ptr`` / ``sat_eligible_csr_sat_idx``.
    config : BeamCapSizingConfig, optional
        Base configuration. Values may be overridden with keyword arguments.
        This includes final-output controls, raw-slot interim snapshot cadence,
        and the shared metric glossary used in saved plots and JSON.
    **overrides : Any
        Field-level overrides for :class:`BeamCapSizingConfig`.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the resolved configuration, selected beam caps,
        per-policy curve arrays, runtime diagnostics, and any saved artifact
        paths.

    Raises
    ------
    KeyError
        If an enabled policy requires a dataset that is not present in the
        input file.
    RuntimeError
        If the file has no iteration groups or if required visibility arrays
        are missing for a selected visibility-aware policy.
    ValueError
        If policy keys or config values are invalid.

    Notes
    -----
    ``pure_reroute`` is an exact lower-bound policy on the eligibility graph.
    It is global across all eligible satellites, including multi-belt files,
    because belt-specific elevation constraints are assumed to be already
    reflected in the stored eligibility input. ``pure_reroute_backend="auto"``
    benchmarks a representative active-slot buffer and keeps the faster exact
    backend for the remainder of the run, while ``READ_SLOT_CHUNK`` remains the
    hard upper bound for each eligibility-mask read in both the probe and the
    main streaming loop. Explicit ``pure_reroute_backend="gpu"`` also requires
    CuPy/NVRTC access to matching CUDA toolkit headers. ``delta(B)`` is total
    unserved demand divided by total processed demand, ``epsilon(B)`` /
    ``eps(B)`` is the fraction of processed demand slots that still fail full
    service at beam cap ``B``, and tail risk is ``P(B_req > B)``, the fraction
    of processed slots whose required beam cap exceeds the tested ``B``.
    Runtime progress uses ``tqdm.auto`` when available and falls back to plain
    text otherwise. Interim snapshots, when enabled, are chunk-aligned and
    triggered by raw scanned timestep slots.
    """
    cfg = _resolve_config(config, overrides)
    _apply_config_globals(cfg, storage_filename=storage_filename)

    all_metas = [
        PolicyMeta("simpson", "Simpson pooling", "#1f77b4", "solid"),
        PolicyMeta("full_reroute", "Full reroute (equal split)", "#14b8a6", "dot"),
        PolicyMeta("pure_reroute", "Pure reroute (exact lower bound)", "#111827", "longdash"),
        PolicyMeta("belt", "Belt-only pooling", "#1f9d55", "dash"),
        PolicyMeta("belt_sky_strict", "Belt+skycell strict", "#f59e0b", "dot"),
        PolicyMeta("belt_sky_temporal", f"Belt+sky temporal ({TEMPORAL_WINDOW_S:g}s)", "#0ea5e9", "dashdot"),
        PolicyMeta("belt_sky_fb", f"Belt+sky fallback (min_sats={BELTSKY_MIN_ACTIVE_SATS})", "#7c3aed", "dash"),
        PolicyMeta("no_reroute", "No reroute (per-satellite)", "#dc2626", "dashdot"),
    ]
    meta_by_key = {m.key: m for m in all_metas}
    enabled_keys: list[str] = []
    for key in ENABLED_POLICY_KEYS:
        if key in meta_by_key and key not in enabled_keys:
            enabled_keys.append(key)
    unknown_keys = [key for key in ENABLED_POLICY_KEYS if key not in meta_by_key]
    if unknown_keys:
        raise ValueError(f"Unknown policy keys in ENABLED_POLICY_KEYS: {unknown_keys}")
    if not enabled_keys:
        raise ValueError("ENABLED_POLICY_KEYS is empty after validation.")
    metas = [meta_by_key[key] for key in enabled_keys]

    need_simpson = "simpson" in enabled_keys
    need_full_reroute = "full_reroute" in enabled_keys
    need_pure_reroute = "pure_reroute" in enabled_keys
    need_no_reroute = "no_reroute" in enabled_keys
    need_belt = "belt" in enabled_keys
    need_strict = "belt_sky_strict" in enabled_keys
    need_temporal = "belt_sky_temporal" in enabled_keys
    need_fb = "belt_sky_fb" in enabled_keys
    need_visibility = need_belt or need_strict or need_temporal or need_fb
    need_count_var = need_simpson or need_full_reroute or need_no_reroute or need_visibility

    acc = {
        m.key: StreamingPolicyAccumulator(
            beam_caps,
            tau=per_slot_loss_tolerance,
            store_slot_samples=STORE_SLOT_RATIO_SAMPLES,
        )
        for m in metas
        if m.key != "pure_reroute"
    }

    pure_solver_caps = np.arange(0, int(beam_caps[-1]) + 1, dtype=np.int32)
    pure_grid_idx = beam_caps.astype(np.int64, copy=False)
    pure_backend_requested = _resolve_pure_reroute_backend(cfg.pure_reroute_backend) if need_pure_reroute else "cpu"
    pure_backend = pure_backend_requested if pure_backend_requested != "auto" else "cpu"
    pure_demand_sum = 0.0
    pure_demand_slots = 0
    pure_matched_links_total = np.zeros(beam_caps.size, dtype=np.int64)
    pure_fail_slots = np.zeros(beam_caps.size, dtype=np.int64)
    pure_required_cap_values: list[np.ndarray] = []
    pure_probe_diag: dict[str, Any] = {
        "pure_reroute_backend_requested": pure_backend_requested if need_pure_reroute else None,
        "pure_reroute_backend_selected": pure_backend if need_pure_reroute else None,
        "pure_reroute_probe_slots": 0,
        "pure_reroute_probe_edges": 0,
        "pure_reroute_probe_cpu_s": None,
        "pure_reroute_probe_gpu_s": None,
    }

    gpu_session = None

    t0 = time.perf_counter()
    demand_sum = 0.0
    demand_slots = 0
    raw_slots_seen = 0
    valid_links_total = 0
    residual_demand_total = 0.0
    missing_context_slots = 0
    strict_hotspots: dict[int, float] = defaultdict(float)
    max_sky_seen = -1
    sky_mappable_links_weight = 0.0
    sky_unmappable_links_weight = 0.0
    main_demand_slot_cap = _normalize_optional_positive_int(MAX_DEMAND_SLOTS, name="MAX_DEMAND_SLOTS")
    stopped_early_main = False
    skycell_demand_sum: np.ndarray | None = None
    skycell_demand_slot_hits: np.ndarray | None = None
    temporal_diag: dict[str, float] = {"enabled": 0.0}
    count_var: str | None = None
    n_sky = int(S1586_N_CELLS)
    sky_mapper = None
    raw_total = 0
    output_run_dir: Path | None = None
    interim_snapshots_written = 0
    last_interim_snapshot_raw_slots = -1
    next_interim_raw_slot = int(INTERIM_EVERY_SLOTS)
    main_progress: _StageProgress | None = None
    interim_artifacts_enabled = bool(
        SAVE_INTERIM_OUTPUTS and (INTERIM_SAVE_HTML or INTERIM_SAVE_JSON or INTERIM_INCLUDE_SKYCELL_VIS)
    )

    def _current_run_diagnostics(elapsed_s: float, *, interim_saved: int) -> dict[str, Any]:
        return {
            "elapsed_s": float(elapsed_s),
            "raw_slots_seen": int(raw_slots_seen),
            "demand_slots": int(demand_slots),
            "pure_reroute_demand_slots": int(pure_demand_slots),
            "max_demand_slots_cap": None if main_demand_slot_cap is None else int(main_demand_slot_cap),
            "stopped_early_main": bool(stopped_early_main),
            "demand_sum": float(demand_sum),
            "pure_reroute_demand_sum": float(pure_demand_sum),
            "need_visibility": bool(need_visibility),
            "valid_links_total": int(valid_links_total),
            "residual_demand_total": float(residual_demand_total),
            "missing_context_slots": int(missing_context_slots),
            "max_sky_seen": int(max_sky_seen),
            "pure_reroute_backend_requested": pure_probe_diag["pure_reroute_backend_requested"],
            "pure_reroute_backend_selected": pure_backend if need_pure_reroute else None,
            "pure_reroute_probe_slots": int(pure_probe_diag["pure_reroute_probe_slots"]),
            "pure_reroute_probe_edges": int(pure_probe_diag["pure_reroute_probe_edges"]),
            "pure_reroute_probe_cpu_s": pure_probe_diag["pure_reroute_probe_cpu_s"],
            "pure_reroute_probe_gpu_s": pure_probe_diag["pure_reroute_probe_gpu_s"],
            "interim_snapshots_written": int(interim_saved),
        }

    def _current_extra_summary() -> dict[str, Any]:
        return {
            "pure_reroute_backend": pure_backend if need_pure_reroute else None,
            "nco": int(cfg.nco),
            "output_dir": str(Path(OUTPUT_DIR)),
            "output_run_dir": None if output_run_dir is None else str(output_run_dir),
            "count_var": count_var,
            "pure_reroute_backend_requested": pure_probe_diag["pure_reroute_backend_requested"],
            "save_interim_outputs": bool(SAVE_INTERIM_OUTPUTS),
            "interim_every_slots": int(INTERIM_EVERY_SLOTS),
        }

    def _current_main_progress_postfix(iter_name: str) -> dict[str, str]:
        return {
            "iter": iter_name,
            "demand_slots": f"{int(demand_slots):,}",
            "pure_demand_slots": f"{int(pure_demand_slots):,}" if need_pure_reroute else "n/a",
            "rss_gib": _rss_gib_postfix() or "n/a",
        }

    def _save_interim_snapshot() -> None:
        nonlocal interim_snapshots_written
        nonlocal last_interim_snapshot_raw_slots

        if output_run_dir is None or not interim_artifacts_enabled:
            return

        evals_now = _build_policy_evals(
            metas=metas,
            acc=acc,
            beam_caps_grid=beam_caps,
            demand_sum=demand_sum,
            pure_demand_sum=pure_demand_sum,
            pure_demand_slots=pure_demand_slots,
            pure_matched_links_total=pure_matched_links_total,
            pure_fail_slots=pure_fail_slots,
            pure_required_cap_values=pure_required_cap_values,
        )
        x_range_now = _compute_output_x_range(evals_now, x_grid=beam_caps)
        run_diagnostics_now = _current_run_diagnostics(
            time.perf_counter() - t0,
            interim_saved=interim_snapshots_written + 1,
        )
        summary_payload = _build_summary_payload(
            enabled_keys=enabled_keys,
            x_grid=beam_caps,
            x_range=x_range_now,
            run_diagnostics=run_diagnostics_now,
            temporal_diag=temporal_diag,
            extra_summary=_current_extra_summary(),
            snapshot_status=_build_snapshot_status(
                is_final=False,
                raw_slots_seen=raw_slots_seen,
                raw_slots_total=raw_total,
                demand_slots=demand_slots,
            ),
            evals=evals_now,
        )
        fig_tail_now = None
        fig_delta_now = None
        fig_eps_now = None
        if INTERIM_SAVE_HTML:
            fig_tail_now, fig_delta_now, fig_eps_now = _build_snapshot_figures(
                evals=evals_now,
                x_grid=beam_caps,
                x_range=x_range_now,
                tau=float(per_slot_loss_tolerance),
                show=False,
            )
        _progress_write(
            f"[saving] Writing interim snapshot #{interim_snapshots_written + 1} "
            f"at raw_slots_seen={raw_slots_seen:,}."
        )
        _write_snapshot_artifacts(
            run_dir=output_run_dir,
            fig_tail=fig_tail_now,
            fig_delta=fig_delta_now,
            fig_eps=fig_eps_now,
            save_html=INTERIM_SAVE_HTML,
            save_json=INTERIM_SAVE_JSON,
            save_npz=False,
            x_grid=beam_caps,
            x_range=x_range_now,
            evals=evals_now,
            summary_payload=summary_payload,
            is_final=False,
        )
        if INTERIM_INCLUDE_SKYCELL_VIS:
            _maybe_render_skycell_demand_hemisphere_runtime(
                run_dir=output_run_dir,
                need_visibility=need_visibility,
                skycell_demand_sum=skycell_demand_sum,
                skycell_demand_slot_hits=skycell_demand_slot_hits,
                demand_slots=demand_slots,
                file_suffix=".interim",
            )
        interim_snapshots_written += 1
        last_interim_snapshot_raw_slots = raw_slots_seen
        _progress_write(
            f"[interim] Saved snapshot #{interim_snapshots_written} "
            f"at raw_slots_seen={raw_slots_seen:,} to {output_run_dir}"
        )

    try:
        scenario.flush_writes(STORAGE_FILENAME)
        _iter_root_key = f"{group_prefix}iter" if group_prefix else "iter"
        with h5py.File(STORAGE_FILENAME, "r") as h5:
            iter_names = _iter_names(h5, iter_root_key=_iter_root_key)
            if not iter_names:
                raise RuntimeError(f"No /{_iter_root_key} groups found.")

            if need_count_var:
                count_var = _choose_count_var(h5, iter_names, COUNT_VAR_CANDIDATES, iter_root_key=_iter_root_key)

            req_vis = ("sat_belt_id", "sat_azimuth", "sat_elevation")
            if need_visibility:
                if not all(all(v in h5[_iter_root_key][it] for v in req_vis) for it in iter_names):
                    raise RuntimeError(f"Missing required visibility arrays for selected policies: {req_vis}")
                n_sky, sky_mapper = _build_sky_mapper(SKYCELL_MODE)
                if sky_mapper is None:
                    raise RuntimeError("Internal error: failed to initialize sky mapper.")
                if ENABLE_SKYCELL_DEMAND_VIS:
                    skycell_demand_sum = np.zeros(n_sky, dtype=np.float64)
                    skycell_demand_slot_hits = np.zeros(n_sky, dtype=np.int64)

            if need_pure_reroute and not all(
                (_PURE_REROUTE_DATASET in h5[_iter_root_key][it]) or _pure_reroute_group_has_csr(h5[_iter_root_key][it])
                for it in iter_names
            ):
                raise KeyError(
                    "Enabled policy 'pure_reroute' requires either the dense dataset "
                    f"'{_PURE_REROUTE_DATASET}' or the sparse CSR datasets "
                    f"'{_PURE_REROUTE_CSR_ROW_PTR_DATASET}' / '{_PURE_REROUTE_CSR_SAT_IDX_DATASET}' "
                    f"in every /{_iter_root_key}/iter_* group."
                )

            if need_count_var:
                raw_total = sum(int(h5[_iter_root_key][it][count_var].shape[0]) for it in iter_names)
            else:
                raw_total = sum(_pure_reroute_group_time_count(h5[_iter_root_key][it]) for it in iter_names)
            if SAVE_OUTPUTS or SAVE_INTERIM_OUTPUTS:
                output_run_dir = _make_output_run_dir(OUTPUT_DIR, OUTPUT_PREFIX)

            sample_iter = h5[_iter_root_key][iter_names[0]]
            sample_count_ds = sample_iter[count_var] if need_count_var and count_var is not None else None
            sample_eligible_ds = (
                sample_iter[_PURE_REROUTE_DATASET]
                if need_pure_reroute and _PURE_REROUTE_DATASET in sample_iter
                else None
            )
            sample_eligible_row_ptr_ds = (
                sample_iter[_PURE_REROUTE_CSR_ROW_PTR_DATASET]
                if need_pure_reroute and _pure_reroute_group_has_csr(sample_iter)
                else None
            )
            sample_eligible_sat_idx_ds = (
                sample_iter[_PURE_REROUTE_CSR_SAT_IDX_DATASET]
                if need_pure_reroute and _pure_reroute_group_has_csr(sample_iter)
                else None
            )

            _progress_write("[startup] Preparing beam-cap sizing run.")
            _progress_write(f"[startup] Input file: {STORAGE_FILENAME}")
            if count_var is not None:
                _progress_write(f"[startup] Count source: {count_var}")
            else:
                _progress_write("[startup] Count source: not required for selected policies")
            _progress_write(f"[startup] Enabled policies: {', '.join(enabled_keys)}")
            if need_pure_reroute:
                _progress_write(f"[startup] Pure reroute backend request: {pure_backend_requested}")
            if need_visibility:
                _progress_write(f"[startup] S1586 skycell count in mapper: {n_sky}")
            if main_demand_slot_cap is not None:
                _progress_write(f"[startup] Debug demand-slot cap enabled: {main_demand_slot_cap:,}")
            if need_temporal:
                _progress_write(
                    "[startup] Temporal edge kernel: "
                    f"{'numba' if _HAVE_NUMBA else 'numpy/python'} "
                    f"(numba_min_work={TEMPORAL_NUMBA_MIN_EDGE_WORK})"
                )
            _progress_write(f"[startup] Iterations: {len(iter_names)} | Raw slots total: {raw_total:,}")
            _progress_write(f"[startup] READ_SLOT_CHUNK={READ_SLOT_CHUNK:,}")

            _progress_write("[dataset-inspection] Inspecting HDF5 dataset layout.")
            if sample_count_ds is not None:
                sample_count_shape = tuple(int(v) for v in sample_count_ds.shape)
                if len(sample_count_shape) not in {2, 3, 4}:
                    raise ValueError(
                        "Beam-cap sizing does not support sat_beam_counts_used with shape "
                        f"{sample_count_shape!r}. Supported layouts are (T,S), (T,sky,S), and (T,obs,S,sky)."
                    )
                _progress_write(f"[dataset-inspection] {_dataset_layout_summary(count_var or 'count', sample_count_ds)}")
                for warning in _dataset_streaming_warnings(
                    count_var or "count",
                    sample_count_ds,
                    read_slot_chunk=READ_SLOT_CHUNK,
                ):
                    _progress_write(f"[WARN][dataset-inspection] {warning}")
            if sample_eligible_ds is not None or sample_eligible_row_ptr_ds is not None:
                if sample_eligible_ds is not None:
                    sample_eligible_shape = tuple(int(v) for v in sample_eligible_ds.shape)
                    if len(sample_eligible_shape) not in {3, 4}:
                        raise ValueError(
                            "Pure reroute does not support dense sat_eligible_mask with shape "
                            f"{sample_eligible_shape!r}. Supported dense layouts are (T,C,S) and (T,sky,C,S)."
                        )
                eligible_diag = _pure_reroute_dataset_diagnostics(
                    sample_eligible_ds,
                    read_slot_chunk=READ_SLOT_CHUNK,
                    max_demand_slots=main_demand_slot_cap,
                    ds_row_ptr=sample_eligible_row_ptr_ds,
                    ds_sat_idx=sample_eligible_sat_idx_ds,
                )
                if sample_eligible_ds is not None:
                    _progress_write(
                        f"[dataset-inspection] {_dataset_layout_summary(_PURE_REROUTE_DATASET, sample_eligible_ds)}"
                    )
                if sample_eligible_row_ptr_ds is not None and sample_eligible_sat_idx_ds is not None:
                    _progress_write(
                        f"[dataset-inspection] {_dataset_layout_summary(_PURE_REROUTE_CSR_ROW_PTR_DATASET, sample_eligible_row_ptr_ds)}"
                    )
                    _progress_write(
                        f"[dataset-inspection] {_dataset_layout_summary(_PURE_REROUTE_CSR_SAT_IDX_DATASET, sample_eligible_sat_idx_ds)}"
                    )
                _progress_write(
                    "[dataset-inspection] pure_reroute memory estimate: "
                    f"main_chunk={_format_gib_text(int(eligible_diag['main_chunk_bytes']))}, "
                    f"auto_probe_chunk={_format_gib_text(int(eligible_diag['probe_chunk_bytes']))}"
                )
                for warning in eligible_diag["warnings"]:
                    _progress_write(f"[WARN][dataset-inspection] {warning}")

            if need_pure_reroute and pure_backend_requested == "auto":
                pure_probe_diag = _select_auto_pure_reroute_backend(
                    h5,
                    iter_names,
                    count_var=count_var,
                    nco=int(cfg.nco),
                    beam_caps=pure_solver_caps,
                    max_demand_slots=main_demand_slot_cap,
                    iter_root_key=_iter_root_key,
                )
                pure_backend = str(pure_probe_diag["pure_reroute_backend_selected"])
            if need_pure_reroute and pure_backend == "gpu" and gpu_session is None:
                from scepter import gpu_accel

                gpu_session = gpu_accel.GpuScepterSession(watchdog_enabled=False)
            if need_pure_reroute and pure_backend_requested == "gpu" and gpu_session is not None:
                _progress_write("[startup] Preflighting pure-reroute GPU solver.")
                _preflight_pure_reroute_gpu_solver(gpu_session)

            if need_temporal and need_visibility and ENABLE_TEMPORAL_SKY_COMPAT:
                if count_var is None or sky_mapper is None:
                    raise RuntimeError("Internal error: temporal policy requires count source and sky mapper.")
                temporal_comp_map, temporal_diag = _build_temporal_compatibility_from_h5(
                    h5=h5,
                    iter_names=iter_names,
                    count_var=count_var,
                    sky_mapper=sky_mapper,
                    n_sky=n_sky,
                    iter_root_key=_iter_root_key,
                )
            else:
                temporal_comp_map = {}
            temporal_root_lookup = _prepare_temporal_lookup_dense(temporal_comp_map)
            if need_pure_reroute:
                _progress_write(f"[startup] Pure reroute backend: {pure_backend}")
                if pure_backend_requested == "auto":
                    _progress_write(
                        "[startup] Pure reroute auto-probe: "
                        f"slots={int(pure_probe_diag['pure_reroute_probe_slots'])}, "
                        f"edges={int(pure_probe_diag['pure_reroute_probe_edges'])}, "
                        f"cpu={pure_probe_diag['pure_reroute_probe_cpu_s']}, "
                        f"gpu={pure_probe_diag['pure_reroute_probe_gpu_s']}"
                    )

            _progress_write("[main-stream] Beginning streaming scan.")
            main_progress = _StageProgress(
                desc="[main-stream] raw slots",
                total=int(raw_total),
                unit="slot",
                report_every=max(1, int(PROGRESS_EVERY_DEMAND_SLOTS)),
            )

            for ii, it in enumerate(iter_names, start=1):
                g = h5[_iter_root_key][it]
                ds_count = g[count_var] if need_count_var and count_var is not None else None
                ds_eligible = (
                    g[_PURE_REROUTE_DATASET]
                    if need_pure_reroute and _PURE_REROUTE_DATASET in g
                    else None
                )
                ds_eligible_row_ptr = (
                    g[_PURE_REROUTE_CSR_ROW_PTR_DATASET]
                    if need_pure_reroute and _pure_reroute_group_has_csr(g)
                    else None
                )
                ds_eligible_sat_idx = (
                    g[_PURE_REROUTE_CSR_SAT_IDX_DATASET]
                    if need_pure_reroute and _pure_reroute_group_has_csr(g)
                    else None
                )
                ds_belt = g["sat_belt_id"] if need_visibility else None
                ds_az = g["sat_azimuth"] if need_visibility else None
                ds_el = g["sat_elevation"] if need_visibility else None

                shape_msgs: list[str] = []
                if ds_count is not None:
                    shape_msgs.append(f"count_shape={tuple(ds_count.shape)}")
                if ds_eligible is not None:
                    shape_msgs.append(f"eligible_shape={tuple(ds_eligible.shape)}")
                elif ds_eligible_row_ptr is not None and ds_eligible_sat_idx is not None:
                    csr_shape = (
                        int(ds_eligible_row_ptr.attrs.get(_PURE_REROUTE_CSR_TIME_COUNT_ATTR, 0)),
                        int(ds_eligible_row_ptr.attrs.get(_PURE_REROUTE_CSR_CELL_COUNT_ATTR, 0)),
                        int(ds_eligible_row_ptr.attrs.get(_PURE_REROUTE_CSR_SAT_COUNT_ATTR, 0)),
                    )
                    shape_msgs.append(f"eligible_csr_shape={csr_shape}")
                    shape_msgs.append(
                        f"eligible_csr_edges={int(ds_eligible_row_ptr[-1]) if int(ds_eligible_row_ptr.shape[0]) > 0 else 0}"
                    )
                if ds_belt is not None:
                    shape_msgs.append(f"link_shape={tuple(ds_belt.shape)}")
                _progress_write(f"[main-stream] iter {ii}/{len(iter_names)} {it}: {', '.join(shape_msgs)}")

                t_count = int(ds_count.shape[0]) if ds_count is not None else None
                if ds_eligible is not None or ds_eligible_row_ptr is not None:
                    t_eligible = int(ds_eligible.shape[0]) if ds_eligible is not None else _pure_reroute_group_time_count(g)
                    if t_count is None:
                        t_count = t_eligible
                    elif t_count != t_eligible:
                        raise ValueError(
                            f"Iteration {it!r} has mismatched slot counts between {count_var!r} ({t_count}) "
                            f"and the pure-reroute eligibility input ({t_eligible})."
                        )
                if t_count is None:
                    raise RuntimeError(f"Iteration {it!r} has no datasets required by the selected policies.")

                for s0 in range(0, t_count, READ_SLOT_CHUNK):
                    raw_slots_before_chunk = int(raw_slots_seen)
                    s1 = min(t_count, s0 + READ_SLOT_CHUNK)
                    count_chunk_for_pure: np.ndarray | None = None
                    if ds_count is not None:
                        c = _normalize_beam_cap_count_chunk(
                            np.asarray(ds_count[s0:s1], dtype=np.int32)
                        )
                        count_chunk_for_pure = c
                        d = c.sum(axis=1, dtype=np.int64)
                        raw_slots_seen += int(c.shape[0])
                        mask = d > 0
                        if np.any(mask):
                            c_pos = c[mask]
                            d_pos = d[mask].astype(np.float64, copy=False)
                            if main_demand_slot_cap is not None:
                                remaining = int(main_demand_slot_cap) - int(demand_slots)
                                if remaining <= 0:
                                    stopped_early_main = True
                                    break
                                if int(c_pos.shape[0]) > remaining:
                                    keep = int(remaining)
                                    c_pos = c_pos[:keep]
                                    d_pos = d_pos[:keep]
                                    stopped_early_main = True
                            vis_views: ChunkVisibilityViews | None = None
                            if need_visibility:
                                if sky_mapper is None or ds_belt is None or ds_az is None or ds_el is None:
                                    raise RuntimeError("Internal error: visibility policies require sky mapper and link arrays.")
                                belt_pos = np.asarray(ds_belt[s0:s1])[mask]
                                az_pos = np.asarray(ds_az[s0:s1])[mask]
                                el_pos = np.asarray(ds_el[s0:s1])[mask]
                                if main_demand_slot_cap is not None and int(belt_pos.shape[0]) != int(c_pos.shape[0]):
                                    keep = int(c_pos.shape[0])
                                    belt_pos = belt_pos[:keep]
                                    az_pos = az_pos[:keep]
                                    el_pos = el_pos[:keep]
                                vis_views = _build_chunk_visibility_views(
                                    belt_chunk=belt_pos,
                                    az_chunk=az_pos,
                                    el_chunk=el_pos,
                                    sky_mapper=sky_mapper,
                                    need_sat_bits=True,
                                    need_slot_cell=False,
                                    need_grouped_sats=True,
                                )

                            for k in range(int(c_pos.shape[0])):
                                row = c_pos[k]
                                d_slot = float(d_pos[k])
                                if d_slot <= 0.0:
                                    continue
                                demand_sum += d_slot
                                demand_slots += 1

                                need_row_float = need_simpson or need_full_reroute or need_no_reroute
                                if need_row_float:
                                    row_f = row.astype(np.float64, copy=False)
                                    a_slot = float(np.count_nonzero(row_f > 0.0))

                                if need_simpson:
                                    sum_x2 = float(np.dot(row_f, row_f))
                                    aeff = max(min((d_slot * d_slot) / max(sum_x2, 1e-30), a_slot), 1.0)
                                    acc["simpson"].add_entry(d_slot, aeff)
                                    acc["simpson"].add_slot_ratio(d_slot / max(aeff, 1e-30))

                                if need_full_reroute:
                                    a_pool = max(a_slot, 1.0)
                                    acc["full_reroute"].add_entry(d_slot, a_pool)
                                    acc["full_reroute"].add_slot_ratio(d_slot / max(a_pool, 1e-30))

                                if need_no_reroute:
                                    x_pos = row_f[row_f > 0]
                                    acc["no_reroute"].add_entries(x_pos, np.ones_like(x_pos))
                                    acc["no_reroute"].add_slot_ratio(float(np.max(x_pos)) if x_pos.size else 0.0)

                                if need_visibility:
                                    if vis_views is None:
                                        raise RuntimeError("Internal error: missing chunk visibility views.")
                                    sat_counts, sat_belt, sat_sky, n_valid, residual, max_sky = _reconstruct_sat_counts_from_views(
                                        views=vis_views,
                                        slot_local_idx=k,
                                        slot_demand=d_slot,
                                    )
                                    valid_links_total += n_valid
                                    residual_demand_total += residual
                                    max_sky_seen = max(max_sky_seen, max_sky)
                                    if sat_counts.size == 0:
                                        missing_context_slots += 1

                                    if need_belt:
                                        db, pb, rb, _ = _aggregate(sat_counts, sat_belt, d_slot)
                                        acc["belt"].add_entries(db, pb)
                                        acc["belt"].add_slot_ratio(rb)

                                    need_any_sky_policy = need_strict or need_temporal or need_fb
                                    if need_any_sky_policy:
                                        sky_valid = sat_sky >= 0
                                        strict_core = sat_belt * np.int64(n_sky) + sat_sky
                                        sky_mappable_links_weight += float(np.sum(sat_counts[sky_valid], dtype=np.float64))
                                        sky_unmappable_links_weight += float(np.sum(sat_counts[~sky_valid], dtype=np.float64))
                                        if skycell_demand_sum is not None and np.any(sky_valid):
                                            sky_idx = sat_sky[sky_valid].astype(np.int64, copy=False)
                                            np.add.at(skycell_demand_sum, sky_idx, sat_counts[sky_valid])
                                            if skycell_demand_slot_hits is not None:
                                                sky_unique = np.unique(sky_idx)
                                                skycell_demand_slot_hits[sky_unique] += 1

                                    if need_strict:
                                        strict_lbl = np.asarray(sat_belt, dtype=np.int64).copy()
                                        if np.any(sky_valid):
                                            strict_lbl[sky_valid] = strict_core[sky_valid] + np.int64(STRICT_GROUP_OFFSET)
                                        ds_, ps_, rs_, us_ = _aggregate(sat_counts, strict_lbl, d_slot)
                                        acc["belt_sky_strict"].add_entries(ds_, ps_)
                                        acc["belt_sky_strict"].add_slot_ratio(rs_)
                                        for lbl, val in zip(us_, ds_):
                                            strict_hotspots[int(lbl)] += float(val)

                                    if need_temporal:
                                        temporal_lbl = np.asarray(sat_belt, dtype=np.int64).copy()
                                        if np.any(sky_valid):
                                            idx_sv = np.nonzero(sky_valid)[0]
                                            node_id = sat_belt[idx_sv] * np.int64(n_sky) + sat_sky[idx_sv]
                                            root_id = _map_temporal_roots_dense(
                                                node_id=node_id,
                                                root_lookup=temporal_root_lookup,
                                            )
                                            temporal_lbl[idx_sv] = root_id + np.int64(TEMPORAL_GROUP_OFFSET)
                                        dt_, pt_, rt_, _ = _aggregate(sat_counts, temporal_lbl, d_slot)
                                        acc["belt_sky_temporal"].add_entries(dt_, pt_)
                                        acc["belt_sky_temporal"].add_slot_ratio(rt_)

                                    if need_fb:
                                        fb_lbl = _fallback_labels(
                                            sat_belt=sat_belt,
                                            strict_labels=strict_core,
                                            sky_valid_mask=sky_valid,
                                            min_active_sats=BELTSKY_MIN_ACTIVE_SATS,
                                        )
                                        df, pf, rf, _ = _aggregate(sat_counts, fb_lbl, d_slot)
                                        acc["belt_sky_fb"].add_entries(df, pf)
                                        acc["belt_sky_fb"].add_slot_ratio(rf)

                    if need_pure_reroute and (ds_eligible is not None or ds_eligible_row_ptr is not None):
                        eligible_chunk = _load_pure_reroute_chunk(g, slot_start=s0, slot_stop=s1)
                        if ds_count is None:
                            raw_slots_seen += int(s1 - s0)
                        active_mask = _active_pure_reroute_slot_mask_any(eligible_chunk, count_chunk_for_pure)
                        if not np.any(active_mask):
                            if stopped_early_main:
                                break
                        else:
                            eligible_chunk = _filter_pure_reroute_chunk_slots(eligible_chunk, active_mask)
                            if main_demand_slot_cap is not None:
                                remaining = int(main_demand_slot_cap) - int(pure_demand_slots)
                                if remaining <= 0:
                                    stopped_early_main = True
                                    break
                                chunk_slots = (
                                    int(eligible_chunk[_PURE_REROUTE_CSR_TIME_COUNT_ATTR])
                                    if isinstance(eligible_chunk, dict)
                                    else int(np.asarray(eligible_chunk, dtype=np.bool_, copy=False).shape[0])
                                )
                                if chunk_slots > remaining:
                                    keep_mask = np.zeros(chunk_slots, dtype=np.bool_)
                                    keep_mask[:remaining] = True
                                    eligible_chunk = _filter_pure_reroute_chunk_slots(eligible_chunk, keep_mask)
                                    stopped_early_main = True
                            chunk_has_data = (
                                int(eligible_chunk[_PURE_REROUTE_CSR_TIME_COUNT_ATTR]) > 0
                                if isinstance(eligible_chunk, dict)
                                else int(np.asarray(eligible_chunk, dtype=np.bool_, copy=False).size) > 0
                            )
                            if chunk_has_data:
                                if gpu_session is not None:
                                    try:
                                        pure_result = _run_pure_reroute_gpu_solver(
                                            gpu_session,
                                            eligible_chunk,
                                            nco=int(cfg.nco),
                                            beam_caps=pure_solver_caps,
                                        )
                                    except RuntimeError as exc:
                                        if pure_backend_requested == "auto":
                                            _progress_write(f"[WARN][main-stream] {exc}")
                                            _progress_write(
                                                "[WARN][main-stream] Falling back to the CPU exact solver "
                                                "for the remaining pure_reroute scan."
                                            )
                                            gpu_session.close(reset_device=False)
                                            gpu_session = None
                                            pure_backend = "cpu"
                                            pure_result = _run_pure_reroute_cpu_solver(
                                                eligible_chunk,
                                                nco=int(cfg.nco),
                                                beam_caps=pure_solver_caps,
                                            )
                                        else:
                                            raise
                                else:
                                    pure_result = _run_pure_reroute_cpu_solver(
                                        eligible_chunk,
                                        nco=int(cfg.nco),
                                        beam_caps=pure_solver_caps,
                                    )
                                eligible_demand_chunk = np.asarray(pure_result["eligible_demand"], dtype=np.int64)
                                matched_chunk_full = np.asarray(pure_result["matched_links"], dtype=np.int64)
                                matched_chunk = matched_chunk_full[:, pure_grid_idx]
                                active_mask = eligible_demand_chunk > 0
                                if np.any(active_mask):
                                    eligible_active = eligible_demand_chunk[active_mask]
                                    matched_active = matched_chunk[active_mask]
                                    pure_demand_sum += float(np.sum(eligible_active, dtype=np.int64))
                                    pure_demand_slots += int(np.count_nonzero(active_mask))
                                    pure_matched_links_total += np.sum(matched_active, axis=0, dtype=np.int64)
                                    pure_fail_slots += np.sum(
                                        matched_active < eligible_active[:, None],
                                        axis=0,
                                        dtype=np.int64,
                                    )
                                    if STORE_SLOT_RATIO_SAMPLES:
                                        pure_required_cap_values.append(
                                            np.asarray(pure_result["required_beam_cap"], dtype=np.int32)[active_mask]
                                        )

                    if interim_artifacts_enabled and raw_slots_seen >= next_interim_raw_slot:
                        _save_interim_snapshot()
                        while next_interim_raw_slot <= raw_slots_seen:
                            next_interim_raw_slot += int(INTERIM_EVERY_SLOTS)

                    if main_progress is not None:
                        main_progress.update(int(raw_slots_seen) - int(raw_slots_before_chunk))
                        main_progress.set_postfix(_current_main_progress_postfix(it))
                        main_progress.maybe_refresh(
                            counter=int(demand_slots),
                            force=bool(stopped_early_main or raw_slots_seen >= raw_total),
                        )

                    if stopped_early_main:
                        break
                if stopped_early_main:
                    break
    finally:
        if main_progress is not None:
            main_progress.set_postfix(
                {
                    "demand_slots": f"{int(demand_slots):,}",
                    "pure_demand_slots": f"{int(pure_demand_slots):,}" if need_pure_reroute else "n/a",
                    "rss_gib": _rss_gib_postfix() or "n/a",
                }
            )
            main_progress.maybe_refresh(counter=int(demand_slots), force=True)
            main_progress.close()
        if gpu_session is not None:
            gpu_session.close(reset_device=False)

    elapsed = time.perf_counter() - t0
    _progress_write("[main-stream] Streaming pass completed.")
    _progress_write(f"Elapsed: {elapsed:.1f} s")
    _progress_write(f"Raw slots seen: {raw_slots_seen:,}")
    _progress_write(f"Demand slots processed: {demand_slots:,}")
    if need_pure_reroute:
        _progress_write(f"Pure-reroute eligible demand slots processed: {pure_demand_slots:,}")
    if stopped_early_main and main_demand_slot_cap is not None:
        _progress_write(f"[debug-cap] Stopped early after reaching MAX_DEMAND_SLOTS={main_demand_slot_cap:,}.")
    if np.isfinite(_timestep_s):
        _progress_write(f"Retained-slot simulated duration: {demand_slots * _timestep_s:.1f} s")
    _progress_write(f"Total demand: {int(round(demand_sum)):,}")
    if need_pure_reroute:
        _progress_write(f"Pure-reroute eligible demand: {int(round(pure_demand_sum)):,}")
    if need_visibility:
        _progress_write(f"Total valid links in visibility arrays: {valid_links_total:,}")
        _progress_write(f"Residual demand (missing/invalid context): {int(round(residual_demand_total)):,}")
        _progress_write(f"Slots with fully missing visibility context: {missing_context_slots:,}")
        _progress_write(f"Observed max skycell ID: {max_sky_seen} (expected < {S1586_N_CELLS})")
        if max_sky_seen >= S1586_N_CELLS:
            _progress_write("[WARN] Observed skycell ID exceeds configured S1586_N_CELLS.")
        sky_total = sky_mappable_links_weight + sky_unmappable_links_weight
        if sky_total > 0.0:
            _progress_write(
                "Skycell mapping coverage (link-weighted over reconstructed per-sat counts): "
                f"mappable={100.0 * sky_mappable_links_weight / sky_total:.2f}%, "
                f"unmappable={100.0 * sky_unmappable_links_weight / sky_total:.2f}%"
            )
    if float(temporal_diag.get("enabled", 0.0)) > 0.5:
        _progress_write(
            "Temporal compatibility prepass summary: "
            f"demand_slots={int(temporal_diag.get('slots_with_demand', 0.0)):,}, "
            f"nodes={int(temporal_diag.get('nodes_seen', 0.0)):,}, "
            f"edges_raw={int(temporal_diag.get('edges_seen', 0.0)):,}, "
            f"edges_kept={int(temporal_diag.get('edges_kept', 0.0)):,}, "
            f"components={int(temporal_diag.get('components', 0.0)):,}, "
            f"window={TEMPORAL_WINDOW_S:g}s, "
            f"stride={int(temporal_diag.get('slot_stride', 1.0))}"
        )

    if interim_artifacts_enabled and (not SAVE_OUTPUTS) and last_interim_snapshot_raw_slots != raw_slots_seen:
        _save_interim_snapshot()

    evals = _build_policy_evals(
        metas=metas,
        acc=acc,
        beam_caps_grid=beam_caps,
        demand_sum=demand_sum,
        pure_demand_sum=pure_demand_sum,
        pure_demand_slots=pure_demand_slots,
        pure_matched_links_total=pure_matched_links_total,
        pure_fail_slots=pure_fail_slots,
        pure_required_cap_values=pure_required_cap_values,
    )

    _progress_write("Selected beam caps per policy:")
    for pe in evals:
        i = pe.selected_idx
        _progress_write(f"  {pe.label:42s} B={pe.selected_b:3d} | delta={pe.delta_run[i]:.3e} | eps={pe.eps_run[i]:.3e}")

    if need_no_reroute:
        no_reroute_b = next(pe.selected_b for pe in evals if pe.key == "no_reroute")
        for pe in evals:
            if pe.key != "no_reroute" and pe.selected_b > no_reroute_b:
                _progress_write(
                    f"[WARN] Policy '{pe.label}' selected B={pe.selected_b} > no-reroute B={no_reroute_b}. "
                    "This usually indicates overly conservative grouping or context mismatch."
                )

    tot_hot = float(sum(strict_hotspots.values()))
    if tot_hot > 0.0:
        _progress_write("Top strict (belt,skycell) groups by accumulated demand:")
        for gid, val in sorted(strict_hotspots.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            if gid < 0:
                kind = "unknown"
                b = -1
                s = -1
            elif gid >= STRICT_GROUP_OFFSET:
                core = int(gid - STRICT_GROUP_OFFSET)
                kind = "strict"
                b = int(core // n_sky)
                s = int(core % n_sky)
            else:
                kind = "belt-fallback"
                b = int(gid)
                s = -1
            share = 100.0 * float(val) / tot_hot
            _progress_write(
                f"  kind={kind:13s} belt={b:2d}, skycell={s:4d}: "
                f"demand={int(round(val)):10d} ({share:5.2f}%)"
            )

    tau = float(per_slot_loss_tolerance)
    x_range = _compute_output_x_range(evals, x_grid=beam_caps)
    b_focus = int(max(pe.selected_b for pe in evals))
    _progress_write(
        f"Auto-zoom X range: [{x_range[0]}, {x_range[1]}], "
        f"focus B={b_focus}, tail guard={CCDF_TAIL_GUARD_PERCENT:g}%"
    )

    build_figures = bool(SAVE_OUTPUTS or SHOW_PLOTS)
    fig_tail = None
    fig_delta = None
    fig_eps = None
    if build_figures:
        fig_tail, fig_delta, fig_eps = _build_snapshot_figures(
            evals=evals,
            x_grid=beam_caps,
            x_range=x_range,
            tau=tau,
            show=SHOW_PLOTS,
        )

    run_diagnostics = _current_run_diagnostics(elapsed, interim_saved=interim_snapshots_written)
    extra_summary = _current_extra_summary()
    if SAVE_OUTPUTS and output_run_dir is not None:
        _progress_write("[saving] Writing final parser outputs.")
        final_payload = _build_summary_payload(
            enabled_keys=enabled_keys,
            x_grid=beam_caps,
            x_range=x_range,
            run_diagnostics=run_diagnostics,
            temporal_diag=temporal_diag,
            extra_summary=extra_summary,
            snapshot_status=_build_snapshot_status(
                is_final=True,
                raw_slots_seen=raw_slots_seen,
                raw_slots_total=raw_total,
                demand_slots=demand_slots,
            ),
            evals=evals,
        )
        _write_snapshot_artifacts(
            run_dir=output_run_dir,
            fig_tail=fig_tail,
            fig_delta=fig_delta,
            fig_eps=fig_eps,
            save_html=SAVE_PLOTS_HTML,
            save_json=SAVE_RESULTS_JSON,
            save_npz=SAVE_RESULTS_NPZ,
            x_grid=beam_caps,
            x_range=x_range,
            evals=evals,
            summary_payload=final_payload,
            is_final=True,
        )
        _progress_write(f"Saved outputs to: {output_run_dir}")
    _maybe_render_skycell_demand_hemisphere_runtime(
        run_dir=output_run_dir if SAVE_OUTPUTS else None,
        need_visibility=need_visibility,
        skycell_demand_sum=skycell_demand_sum,
        skycell_demand_slot_hits=skycell_demand_slot_hits,
        demand_slots=demand_slots,
    )

    selected_caps = {pe.key: int(pe.selected_b) for pe in evals}
    curves = {
        pe.key: {
            "delta": np.asarray(pe.delta_run, dtype=np.float64),
            "eps": np.asarray(pe.eps_run, dtype=np.float64),
            "tail_risk_percent": np.asarray(pe.ccdf_exceed_percent, dtype=np.float64),
            "selected_b": int(pe.selected_b),
        }
        for pe in evals
    }
    return {
        "config": asdict(cfg),
        "beam_caps": np.asarray(beam_caps, dtype=np.int32),
        "enabled_policy_keys": list(enabled_keys),
        "selected_caps": selected_caps,
        "policy_curves": curves,
        "policy_evals": evals,
        "run_diagnostics": run_diagnostics,
        "temporal_diag": {k: float(v) for k, v in temporal_diag.items()},
        "output_run_dir": None if output_run_dir is None else str(output_run_dir),
        "fig_tail": fig_tail,
        "fig_delta": fig_delta,
        "fig_eps": fig_eps,
    }


__all__ = [
    "BeamCapSizingConfig",
    "run_beam_cap_sizing",
]
