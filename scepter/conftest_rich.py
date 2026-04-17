"""Rich terminal UI plugin for SCEPTer test suite.

Provides:
- Live progress bar with spinner animation per test file
- Per-category pass/fail/skip counters updated in real time
- Colour-coded status indicators (green pass, red fail, yellow skip)
- Per-file timing breakdown at the end
- Slowest-test ranking in the summary
- Total wall-clock time with a progress percentage

Activated automatically when ``rich`` is importable.  Disabled when
running under xdist workers (each worker would fight for the terminal).

Authors
-------
- Boris Sorokin <boris.sorokin@skao.int> AKA Mralin <mralin@protonmail.com>
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import pytest

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        MofNCompleteColumn,
    )

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ---------------------------------------------------------------------------
#  State container
# ---------------------------------------------------------------------------

class _ScepterTestState:
    def __init__(self) -> None:
        self.total_collected = 0
        self.completed = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = 0
        self.start_time = 0.0

        # Per-file tracking
        self.file_stats: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "time": 0.0}
        )
        self.file_order: list[str] = []

        # Per-test durations for slowest ranking
        self.test_durations: list[tuple[str, float]] = []

        # Current test being run
        self.current_test = ""
        self.current_file = ""

        # Failed test names for summary
        self.failed_tests: list[str] = []


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------

def _status_icon(stats: dict) -> str:
    if stats["failed"] > 0:
        return "[red]FAIL[/red]"
    if stats["passed"] == stats["total"]:
        return "[green]PASS[/green]"
    if stats["skipped"] > 0 and stats["passed"] + stats["skipped"] == stats["total"]:
        return "[green]PASS[/green]"
    return "[yellow]...[/yellow]"


def _build_live_table(state: _ScepterTestState) -> Table:
    """Build the live-updating table shown during test execution."""
    table = Table(
        title="SCEPTer Test Suite",
        title_style="bold cyan",
        show_header=True,
        header_style="bold",
        border_style="dim",
        pad_edge=False,
    )
    table.add_column("", width=4, justify="center")
    table.add_column("File", min_width=30)
    table.add_column("Pass", justify="right", style="green", width=5)
    table.add_column("Fail", justify="right", style="red", width=5)
    table.add_column("Skip", justify="right", style="yellow", width=5)
    table.add_column("Total", justify="right", width=5)
    table.add_column("Time", justify="right", width=8)
    table.add_column("", width=1)

    for filepath in state.file_order:
        stats = state.file_stats[filepath]
        short = Path(filepath).name
        icon = _status_icon(stats)
        done = stats["passed"] + stats["failed"] + stats["skipped"]
        time_str = f"{stats['time']:.1f}s" if stats["time"] > 0 else ""

        # Highlight the currently active file
        if filepath == state.current_file and done < stats["total"]:
            style = "bold"
            spinner = "[cyan]>[/cyan]"
        else:
            style = ""
            spinner = ""

        table.add_row(
            icon,
            Text(short, style=style),
            str(stats["passed"]) if stats["passed"] else "",
            str(stats["failed"]) if stats["failed"] else "",
            str(stats["skipped"]) if stats["skipped"] else "",
            f"{done}/{stats['total']}",
            time_str,
            spinner,
        )

    # Summary row
    elapsed = time.monotonic() - state.start_time
    pct = 100.0 * state.completed / max(1, state.total_collected)
    table.add_section()
    table.add_row(
        "",
        Text("TOTAL", style="bold"),
        str(state.passed),
        str(state.failed) if state.failed else "",
        str(state.skipped) if state.skipped else "",
        f"{state.completed}/{state.total_collected}",
        f"{elapsed:.1f}s",
        f"[cyan]{pct:.0f}%[/cyan]",
    )

    return table


def _build_summary_table(state: _ScepterTestState) -> Table:
    """Build the final summary table shown after all tests complete."""
    elapsed = time.monotonic() - state.start_time

    table = Table(
        title="SCEPTer Test Results",
        title_style="bold cyan",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    table.add_column("", width=6, justify="center")
    table.add_column("File", min_width=30)
    table.add_column("Pass", justify="right", style="green", width=6)
    table.add_column("Fail", justify="right", style="red", width=6)
    table.add_column("Skip", justify="right", style="yellow", width=6)
    table.add_column("Total", justify="right", width=6)
    table.add_column("Time", justify="right", width=10)

    for filepath in state.file_order:
        stats = state.file_stats[filepath]
        short = Path(filepath).name
        icon = _status_icon(stats)
        table.add_row(
            icon,
            short,
            str(stats["passed"]),
            str(stats["failed"]) if stats["failed"] else "-",
            str(stats["skipped"]) if stats["skipped"] else "-",
            str(stats["total"]),
            f"{stats['time']:.2f}s",
        )

    table.add_section()
    status = "[bold green]ALL PASSED[/bold green]" if state.failed == 0 else "[bold red]FAILURES[/bold red]"
    table.add_row(
        status,
        "Total",
        str(state.passed),
        str(state.failed) if state.failed else "-",
        str(state.skipped) if state.skipped else "-",
        str(state.total_collected),
        f"{elapsed:.2f}s",
    )

    return table


# ---------------------------------------------------------------------------
#  Plugin hooks
# ---------------------------------------------------------------------------

class ScepterRichPlugin:
    """Pytest plugin providing rich terminal output for the SCEPTer test suite."""

    def __init__(self) -> None:
        self.console = Console()
        self.state = _ScepterTestState()
        self.live: Live | None = None
        self._file_start_times: dict[str, float] = {}

    def pytest_collection_modifyitems(self, items: list) -> None:
        self.state.total_collected = len(items)
        # Count per-file totals
        for item in items:
            filepath = str(item.path)
            if filepath not in self.state.file_stats:
                self.state.file_order.append(filepath)
            self.state.file_stats[filepath]["total"] += 1

    def pytest_runtestloop(self, session: pytest.Session) -> None:
        self.state.start_time = time.monotonic()

    def pytest_runtest_logstart(self, nodeid: str, location: tuple) -> None:
        filepath = str(location[0]) if location else nodeid.split("::")[0]
        # Resolve to full path
        for fp in self.state.file_order:
            if fp.endswith(filepath) or filepath.endswith(Path(fp).name):
                filepath = fp
                break
        self.state.current_file = filepath
        self.state.current_test = nodeid.split("::")[-1]
        if filepath not in self._file_start_times:
            self._file_start_times[filepath] = time.monotonic()
        if self.live is not None:
            self.live.update(self._render())

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call" and not (report.when == "setup" and report.skipped):
            return

        filepath = str(report.fspath)
        for fp in self.state.file_order:
            if fp.endswith(Path(filepath).name):
                filepath = fp
                break

        stats = self.state.file_stats[filepath]
        if report.passed:
            stats["passed"] += 1
            self.state.passed += 1
        elif report.failed:
            stats["failed"] += 1
            self.state.failed += 1
            self.state.failed_tests.append(report.nodeid)
        elif report.skipped:
            stats["skipped"] += 1
            self.state.skipped += 1

        self.state.completed += 1
        stats["time"] = time.monotonic() - self._file_start_times.get(filepath, time.monotonic())

        if report.when == "call":
            self.state.test_durations.append((report.nodeid, report.duration))

        if self.live is not None:
            self.live.update(self._render())

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.state.start_time = time.monotonic()
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self.live.start()

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def pytest_terminal_summary(self, terminalreporter, exitstatus: int) -> None:
        self.console.print()
        self.console.print(_build_summary_table(self.state))

        # Slowest tests
        if self.state.test_durations:
            slowest = sorted(self.state.test_durations, key=lambda x: -x[1])[:15]
            self.console.print()
            slow_table = Table(
                title="Slowest 15 Tests",
                title_style="bold yellow",
                show_header=True,
                header_style="bold",
                border_style="dim",
            )
            slow_table.add_column("#", width=3, justify="right")
            slow_table.add_column("Test", min_width=50)
            slow_table.add_column("Time", justify="right", width=10)
            for rank, (nodeid, duration) in enumerate(slowest, 1):
                short = nodeid.split("::")[-1]
                file_short = Path(nodeid.split("::")[0]).name
                color = "red" if duration > 10.0 else "yellow" if duration > 3.0 else ""
                slow_table.add_row(
                    str(rank),
                    f"[dim]{file_short}[/dim]::{short}",
                    Text(f"{duration:.2f}s", style=color),
                )
            self.console.print(slow_table)

        # Failed tests
        if self.state.failed_tests:
            self.console.print()
            self.console.print(
                Panel(
                    "\n".join(f"[red]FAIL[/red] {t}" for t in self.state.failed_tests),
                    title="[bold red]Failed Tests[/bold red]",
                    border_style="red",
                )
            )

        elapsed = time.monotonic() - self.state.start_time
        self.console.print()
        if self.state.failed == 0:
            self.console.print(
                f"[bold green]All {self.state.passed} tests passed[/bold green] "
                f"({self.state.skipped} skipped) in {elapsed:.1f}s"
            )
        else:
            self.console.print(
                f"[bold red]{self.state.failed} failed[/bold red], "
                f"{self.state.passed} passed, {self.state.skipped} skipped "
                f"in {elapsed:.1f}s"
            )

    def _render(self):
        return _build_live_table(self.state)


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register the rich plugin when rich is available and we're not an xdist worker."""
    if not HAS_RICH:
        return
    # Don't activate on xdist workers — only the controller needs the UI
    if hasattr(config, "workerinput"):
        return
    # Don't activate if verbose mode is on (would conflict with per-test output)
    if config.option.verbose > 0:
        return
    # Don't activate if --tb=long/short is explicitly set (user wants tracebacks)
    tb = getattr(config.option, "tbstyle", "auto")
    if tb in ("long", "short", "line"):
        return
    config.pluginmanager.register(ScepterRichPlugin(), "scepter-rich")
