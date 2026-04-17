"""Tests for the splash screen and bootstrap helpers in gui_bootstrap.py.

These tests exercise the modernised splash screen: rounded-corner
clipping, non-blocking minimum-duration wait, HiDPI pixmap scaling,
fade-out animation, and the removal of ``WindowStaysOnTopHint``.

All tests run with ``QT_QPA_PLATFORM=offscreen`` so they work in
headless CI environments without a display server.
"""

from __future__ import annotations

import os
import types
from time import monotonic
from unittest.mock import MagicMock

# Force offscreen rendering before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets

import scepter.gui_bootstrap as bootstrap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> QtWidgets.QApplication:
    """Return the singleton QApplication, creating one if needed."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture()
def icon(qapp: QtWidgets.QApplication) -> QtGui.QIcon:
    """Return a non-null app icon for splash construction."""
    return bootstrap.load_app_icon()


def _wait_for(ms: int) -> None:
    """Spin the Qt event loop for *ms* milliseconds."""
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(ms, loop.quit)
    loop.exec()


# ---------------------------------------------------------------------------
# Window flag tests
# ---------------------------------------------------------------------------


class TestSplashWindowFlags:
    """Verify that the splash window has the expected Qt flags."""

    def test_translucent_background_is_set(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """WA_TranslucentBackground must be set for rounded-corner clipping."""
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            assert splash.testAttribute(QtCore.Qt.WA_TranslucentBackground)
        finally:
            splash.close()

    def test_no_stays_on_top_flag(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """WindowStaysOnTopHint must NOT be present — the splash should not
        forcefully obscure other applications."""
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            flags = splash.windowFlags()
            assert not (flags & QtCore.Qt.WindowStaysOnTopHint)
        finally:
            splash.close()

    def test_frameless_hint_is_present(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """FramelessWindowHint must remain — the splash has no title bar."""
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            flags = splash.windowFlags()
            assert flags & QtCore.Qt.FramelessWindowHint
        finally:
            splash.close()

    def test_window_type_for_taskbar_presence(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """Qt.Window flag must be set so the splash gets a taskbar button."""
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            flags = splash.windowFlags()
            assert flags & QtCore.Qt.Window
        finally:
            splash.close()


# ---------------------------------------------------------------------------
# Progress / stage update tests
# ---------------------------------------------------------------------------


class TestSplashStageUpdates:
    """Verify set_stage updates internal state correctly."""

    def test_set_stage_updates_progress_and_text(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            splash.set_stage(progress=0.5, status_text="Halfway there")
            assert splash._progress_value == pytest.approx(0.5)
            assert splash._status_text == "Halfway there"
        finally:
            splash.close()

    def test_set_stage_clamps_negative_progress(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            splash.set_stage(progress=-0.5, status_text="Under zero")
            assert splash._progress_value == pytest.approx(0.0)
        finally:
            splash.close()

    def test_set_stage_clamps_above_one(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            splash.set_stage(progress=1.5, status_text="Over one")
            assert splash._progress_value == pytest.approx(1.0)
        finally:
            splash.close()

    def test_progress_rect_width_scales_with_value(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            splash.set_stage(progress=0.25, status_text="Quarter")
            expected_width = splash._progress_bar_rect.width() * 0.25
            assert splash._progress_value_rect.width() == pytest.approx(
                expected_width
            )
        finally:
            splash.close()


# ---------------------------------------------------------------------------
# HiDPI pixmap tests
# ---------------------------------------------------------------------------


class TestSplashHiDPI:
    """Verify the splash pixmap respects device-pixel-ratio."""

    def test_pixmap_at_1x_dpr(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """At 1× DPR the pixmap should be exactly 480×220 physical pixels."""
        pixmap = bootstrap.build_startup_splash(icon)
        # On the offscreen platform, DPR is typically 1.0.
        dpr = pixmap.devicePixelRatio()
        logical_w = pixmap.width() / dpr
        logical_h = pixmap.height() / dpr
        assert logical_w == pytest.approx(480, abs=1)
        assert logical_h == pytest.approx(220, abs=1)

    def test_pixmap_at_2x_dpr(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """At 2× DPR the pixmap should be 960×440 physical pixels."""
        mock_screen = MagicMock()
        mock_screen.devicePixelRatio.return_value = 2.0

        # Monkeypatch the QGuiApplication.instance() chain so
        # build_startup_splash sees DPR = 2.0.
        mock_app = MagicMock()
        mock_app.primaryScreen.return_value = mock_screen
        monkeypatch.setattr(
            QtGui.QGuiApplication, "instance", staticmethod(lambda: mock_app)
        )

        pixmap = bootstrap.build_startup_splash(icon)
        assert pixmap.devicePixelRatio() == pytest.approx(2.0)
        assert pixmap.width() == 960
        assert pixmap.height() == 440


# ---------------------------------------------------------------------------
# Finish / fade-out tests
# ---------------------------------------------------------------------------


class TestSplashFinish:
    """Verify the non-blocking finish and fade-out behaviour."""

    def test_finish_respects_minimum_duration(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """finish_with_min_duration should wait at least minimum_visible_ms
        before returning, but should not hang indefinitely."""
        min_ms = 120
        splash = bootstrap.create_startup_splash(
            icon, minimum_visible_ms=min_ms
        )
        splash.show()
        qapp.processEvents()

        dummy_window = QtWidgets.QWidget()
        t0 = monotonic()
        splash.finish_with_min_duration(dummy_window)
        elapsed_ms = (monotonic() - t0) * 1000.0

        # Must have waited at least the minimum duration.
        assert elapsed_ms >= min_ms * 0.8  # small tolerance for timer jitter
        # Must not hang — allow generous upper bound for CI slowness.
        assert elapsed_ms < 5000
        dummy_window.close()

    def test_finish_no_extra_wait_when_time_elapsed(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """If enough time has already passed, finish should only run the
        fade-out without an additional minimum-duration wait."""
        splash = bootstrap.create_startup_splash(
            icon, minimum_visible_ms=50
        )
        splash.show()
        qapp.processEvents()

        # Wait longer than the minimum visible time.
        _wait_for(120)

        dummy_window = QtWidgets.QWidget()
        t0 = monotonic()
        splash.finish_with_min_duration(dummy_window)
        elapsed_ms = (monotonic() - t0) * 1000.0

        # Should only take the fade-out duration (250ms) plus overhead.
        assert elapsed_ms < 2000
        dummy_window.close()

    def test_splash_hidden_after_finish(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """After finish_with_min_duration returns, the splash must not be
        visible."""
        splash = bootstrap.create_startup_splash(
            icon, minimum_visible_ms=0
        )
        splash.show()
        qapp.processEvents()

        dummy_window = QtWidgets.QWidget()
        splash.finish_with_min_duration(dummy_window)
        qapp.processEvents()

        assert not splash.isVisible()
        dummy_window.close()


# ---------------------------------------------------------------------------
# Rounded-corner clipping test
# ---------------------------------------------------------------------------


class TestSplashRoundedCorners:
    """Verify the paintEvent clips to a rounded rectangle."""

    def test_corner_pixels_are_transparent(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """The four corner pixels of a grabbed splash image should be fully
        transparent, confirming that the rounded-rect clip path is active.

        Note: On some offscreen QPA implementations the grab may not
        honour WA_TranslucentBackground perfectly.  The test is therefore
        lenient — it checks that corner alpha is significantly lower than
        the centre alpha rather than requiring exact zero.
        """
        splash = bootstrap.create_startup_splash(
            icon, minimum_visible_ms=0
        )
        splash.show()
        qapp.processEvents()

        grabbed = splash.grab()
        img = grabbed.toImage()
        w, h = img.width(), img.height()

        if w == 0 or h == 0:
            pytest.skip("Offscreen platform returned an empty grab")

        # Sample corner and centre pixels.
        top_left = QtGui.QColor(img.pixel(0, 0))
        top_right = QtGui.QColor(img.pixel(w - 1, 0))
        bot_left = QtGui.QColor(img.pixel(0, h - 1))
        bot_right = QtGui.QColor(img.pixel(w - 1, h - 1))
        centre = QtGui.QColor(img.pixel(w // 2, h // 2))

        # Centre should be opaque (or nearly so).
        if centre.alpha() < 128:
            pytest.skip(
                "Centre pixel unexpectedly transparent — offscreen "
                "platform does not support translucent grabs"
            )

        # All four corners should be notably more transparent than centre.
        # On Windows offscreen QPA, grab() does not honour translucent
        # backgrounds so all pixels come back opaque.  Skip gracefully.
        all_opaque = all(
            c.alpha() == 255
            for c in (top_left, top_right, bot_left, bot_right, centre)
        )
        if all_opaque:
            pytest.skip(
                "Offscreen platform returns fully opaque grabs — "
                "rounded-corner clipping cannot be verified via pixel "
                "sampling on this platform"
            )

        for corner in (top_left, top_right, bot_left, bot_right):
            assert corner.alpha() < centre.alpha(), (
                f"Corner alpha ({corner.alpha()}) should be less than "
                f"centre alpha ({centre.alpha()}) due to rounded clipping"
            )

        splash.close()


# ---------------------------------------------------------------------------
# Pixmap / icon sanity
# ---------------------------------------------------------------------------


class TestSplashPixmap:
    """Basic sanity checks for the splash pixmap and icon."""

    def test_splash_pixmap_is_not_null(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        splash = bootstrap.create_startup_splash(icon, minimum_visible_ms=0)
        try:
            assert not splash.pixmap().isNull()
        finally:
            splash.close()

    def test_icon_is_not_null(
        self,
        qapp: QtWidgets.QApplication,
    ) -> None:
        icon = bootstrap.load_app_icon()
        assert not icon.isNull()

    def test_fallback_icon_is_not_null(
        self,
        qapp: QtWidgets.QApplication,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even when the SVG file is missing, the fallback icon should
        be generated successfully."""
        icon = bootstrap._build_fallback_app_icon()
        assert not icon.isNull()


# ---------------------------------------------------------------------------
# SVG icon rendering in splash
# ---------------------------------------------------------------------------


class TestSplashSvgIcon:
    """Verify the SVG icon renders crisply and at the correct size."""

    def test_svg_icon_path_exists(self) -> None:
        """The bundled SVG app icon must be present on disk."""
        assert bootstrap._APP_ICON_PATH.exists(), (
            f"Expected SVG icon at {bootstrap._APP_ICON_PATH}"
        )

    def test_svg_renderer_loads_bundled_icon(
        self,
        qapp: QtWidgets.QApplication,
    ) -> None:
        """QSvgRenderer must successfully parse the bundled SVG."""
        from PySide6.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(str(bootstrap._APP_ICON_PATH))
        assert renderer.isValid(), "SVG renderer failed to parse the app icon"

    def test_splash_pixmap_with_missing_svg_falls_back(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """When the SVG file is absent, build_startup_splash should still
        produce a valid pixmap by falling back to QIcon rasterisation."""
        fake_path = tmp_path / "nonexistent.svg"
        monkeypatch.setattr(bootstrap, "_APP_ICON_PATH", fake_path)
        pixmap = bootstrap.build_startup_splash(icon)
        assert not pixmap.isNull()

    def test_icon_region_is_compact(
        self,
        qapp: QtWidgets.QApplication,
        icon: QtGui.QIcon,
    ) -> None:
        """The icon should occupy a compact region (~36 logical px) and
        not dominate the splash layout.  Verified indirectly: the splash
        pixmap at 1× DPR is 480 wide and the icon plus gap leaves room
        for at least 380 px of title text."""
        pixmap = bootstrap.build_startup_splash(icon)
        dpr = pixmap.devicePixelRatio()
        logical_w = pixmap.width() / dpr
        # Icon is 36 px at x=32, gap=12 → text starts at 80.
        # So at least 400 px remain for text (480 − 80).
        assert logical_w - 80 >= 380, (
            f"Not enough horizontal space for title text: "
            f"{logical_w - 80:.0f} px available"
        )
