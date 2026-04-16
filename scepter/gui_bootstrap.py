"""Lightweight Qt bootstrap helpers for the desktop GUI.

This module provides application identity, Windows shell integration,
branded splash screen with progress feedback, and icon loading helpers.
It is imported early by the top-level ``gui.py`` launcher — before the
heavy ``scepter`` package — so it must stay lightweight.
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os
import sys
from pathlib import Path
from time import monotonic

from PySide6 import QtCore, QtGui, QtWidgets
try:
    from PySide6.QtSvg import QSvgRenderer
except Exception:  # pragma: no cover - optional Qt module
    QSvgRenderer = None

_APP_ICON_PATH = Path(__file__).resolve().parent / "data" / "satellite_app_icon.svg"
_WINDOWS_APP_ICON_PATH = Path(__file__).resolve().parent / "data" / "satellite_app_icon.ico"
_WINDOWS_APP_ID = "org.skao.scepter.gui"
_LINUX_DESKTOP_FILE_NAME = "org.skao.scepter"
_LINUX_RESOURCE_NAME = "scepter"
_APP_ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
_SPLASH_MIN_VISIBLE_MS = 650
_SPLASH_FADE_OUT_MS = 300
_SPLASH_CORNER_RADIUS = 18.0
_SPLASH_ANIM_FPS = 60
_SPLASH_ANIM_INTERVAL_MS = int(1000 / _SPLASH_ANIM_FPS)
_WINDOWS_NATIVE_ICON_HANDLES: list[int] = []


def _load_appinfo_module():
    """Load the lightweight app metadata module without importing ``scepter``."""
    module_name = "_scepter_gui_bootstrap_appinfo"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).resolve().parent / "appinfo.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load app metadata from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


appinfo = _load_appinfo_module()


def _apply_linux_shell_identity(app: QtWidgets.QApplication | None = None) -> None:
    """Apply desktop-file metadata once a QApplication is available."""
    if not sys.platform.startswith("linux"):
        return
    os.environ.setdefault("RESOURCE_NAME", _LINUX_RESOURCE_NAME)
    target_app = QtWidgets.QApplication.instance() if app is None else app
    if target_app is None:
        return
    try:
        target_app.setDesktopFileName(_LINUX_DESKTOP_FILE_NAME)
    except Exception:
        pass


def configure_application(app: QtWidgets.QApplication) -> None:
    """Apply the shared application identity and baseline style."""
    app.setApplicationName(appinfo.GUI_APP_NAME)
    app.setApplicationVersion(appinfo.APP_VERSION)
    app.setOrganizationName("SKAO")
    app.setOrganizationDomain("skao.int")
    app.setStyle("Fusion")
    _apply_linux_shell_identity(app)


def configure_windows_shell_identity(app_id: str = _WINDOWS_APP_ID) -> bool:
    """Apply platform-specific shell identity for taskbar grouping.

    On Windows this sets the explicit AppUserModelID so the taskbar groups
    the GUI correctly.  On Linux this sets the ``WM_CLASS`` / desktop
    file name so GNOME/KDE taskbar and alt-tab show the correct icon.
    """
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
            return True
        except Exception:
            return False
    # Linux / X11 / Wayland: set the desktop entry name and WM_CLASS
    # so the taskbar picks up the correct icon and grouping.
    if sys.platform.startswith("linux"):
        _apply_linux_shell_identity()
        return True
    return False


def _load_windows_hicon(icon_path: Path, size_px: int) -> int:
    if sys.platform != "win32":
        return 0
    try:
        return int(
            ctypes.windll.user32.LoadImageW(
                None, str(icon_path), 1, int(size_px), int(size_px), 0x00000010,
            ) or 0
        )
    except Exception:
        return 0


def apply_windows_window_icon(
    widget: QtWidgets.QWidget | object,
    icon_path: Path | None = None,
) -> bool:
    """Apply the native Windows window/taskbar icon when available."""
    if sys.platform != "win32":
        return False
    resolved_path = resolve_app_icon_path() if icon_path is None else Path(icon_path)
    if resolved_path is None or resolved_path.suffix.lower() != ".ico" or not resolved_path.exists():
        return False
    if not hasattr(widget, "winId"):
        return False
    try:
        hwnd = int(widget.winId())
    except Exception:
        return False
    if hwnd <= 0:
        return False
    user32 = getattr(ctypes, "windll", None)
    if user32 is None or not hasattr(user32, "user32"):
        return False
    user32 = user32.user32
    big_icon = _load_windows_hicon(resolved_path, 256)
    small_icon = _load_windows_hicon(resolved_path, 32)
    if big_icon <= 0 and small_icon <= 0:
        return False
    if big_icon > 0:
        _WINDOWS_NATIVE_ICON_HANDLES.append(int(big_icon))
        try:
            user32.SendMessageW(hwnd, 0x0080, 1, int(big_icon))
        except Exception:
            pass
    if small_icon > 0:
        _WINDOWS_NATIVE_ICON_HANDLES.append(int(small_icon))
        try:
            user32.SendMessageW(hwnd, 0x0080, 0, int(small_icon))
        except Exception:
            pass
    set_class_icon = getattr(user32, "SetClassLongPtrW", None)
    if set_class_icon is None:
        set_class_icon = getattr(user32, "SetClassLongW", None)
    if set_class_icon is not None:
        try:
            if big_icon > 0:
                set_class_icon(hwnd, -14, int(big_icon))
            if small_icon > 0:
                set_class_icon(hwnd, -34, int(small_icon))
        except Exception:
            pass
    return True


def _build_multi_size_icon(base_icon: QtGui.QIcon) -> QtGui.QIcon:
    if base_icon.isNull():
        return base_icon
    icon = QtGui.QIcon()
    for size in _APP_ICON_SIZES:
        pixmap = base_icon.pixmap(int(size), int(size))
        if not pixmap.isNull():
            icon.addPixmap(pixmap)
    return icon if not icon.isNull() else base_icon


# ---------------------------------------------------------------------------
# Multi-tier programmatic icon renderer
# ---------------------------------------------------------------------------
# The icon must read clearly at three very different sizes:
#
#   Tier 3 (16-24 px) — window title bar, small toolbar buttons
#       Only the essential silhouette: orbit arc, horizon curve,
#       satellite as a bright dot, telescope as a dot.
#       Thick strokes, no fine detail, strong contrast.
#
#   Tier 2 (32-64 px) — taskbar, splash logo, alt-tab thumbnail
#       Core composition: orbit arc, horizon, crosshairs,
#       recognisable telescope dish, satellite with visible panels.
#       No star field, no signal waves, no panel detail lines.
#
#   Tier 1 (96-256 px) — about dialog, file explorer, store listing
#       Full showcase: star field, orbit glow, signal waves,
#       antenna, panel detail, Earth fill, border glow.
# ---------------------------------------------------------------------------

def _render_icon_to_pixmap(size_px: int) -> QtGui.QPixmap:
    """Render the SCEPTer app icon at *size_px* with detail appropriate for
    that physical size.  Returns a QPixmap of exactly (size_px × size_px)."""
    s = size_px / 128.0  # scale factor relative to the 128-px canonical size
    tier = 1 if size_px >= 80 else (2 if size_px >= 28 else 3)

    pixmap = QtGui.QPixmap(size_px, size_px)
    pixmap.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pixmap)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

    # --- Background --------------------------------------------------------
    corner_r = 34.0 * s
    # At small sizes, use slightly less rounding so the shape reads as a
    # rounded square rather than a circle.
    if tier == 3:
        corner_r = 22.0 * s
    bg_inset = max(1.0, 8.0 * s)
    bg_size = size_px - 2.0 * bg_inset
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor("#08152A"))
    p.drawRoundedRect(QtCore.QRectF(bg_inset, bg_inset, bg_size, bg_size), corner_r, corner_r)

    # Border (tier 1-2 only)
    if tier <= 2:
        p.setBrush(QtCore.Qt.NoBrush)
        bw = max(0.8, 1.5 * s)
        p.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 50 if tier == 1 else 70), bw))
        p.drawRoundedRect(
            QtCore.QRectF(bg_inset + bw / 2, bg_inset + bw / 2, bg_size - bw, bg_size - bw),
            corner_r - bw / 2, corner_r - bw / 2,
        )

    # --- Star field (tier 1 only) ------------------------------------------
    if tier == 1:
        p.setPen(QtCore.Qt.NoPen)
        for sx, sy, sr, sa in [
            (22, 28, 0.9, 0.50), (42, 18, 0.7, 0.40), (98, 22, 0.8, 0.45),
            (108, 68, 0.7, 0.35), (18, 56, 0.6, 0.30), (88, 34, 0.6, 0.35),
            (34, 38, 0.5, 0.25),
        ]:
            p.setBrush(QtGui.QColor(203, 213, 225, int(255 * sa)))
            p.drawEllipse(QtCore.QPointF(sx * s, sy * s), sr * s, sr * s)

    # --- Orbit arc ---------------------------------------------------------
    orbit = QtGui.QPainterPath()
    orbit.moveTo(24 * s, 76 * s)
    orbit.cubicTo(34 * s, 58 * s, 52 * s, 47 * s, 72 * s, 47 * s)
    orbit.cubicTo(83 * s, 47 * s, 92 * s, 49.5 * s, 99 * s, 53 * s)

    # Glow layer (tier 1 only)
    if tier == 1:
        p.setPen(QtGui.QPen(
            QtGui.QColor(56, 189, 248, 18), max(2.0, 14.0 * s),
            QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
        ))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(orbit)

    # Main orbit stroke
    orbit_w = {1: 7.0, 2: 8.0, 3: 10.0}[tier] * s
    p.setPen(QtGui.QPen(QtGui.QColor("#38BDF8"), orbit_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawPath(orbit)

    # --- Horizon arc -------------------------------------------------------
    horizon = QtGui.QPainterPath()
    horizon.moveTo(28 * s, 90 * s)
    horizon.cubicTo(37 * s, 103 * s, 50 * s, 110 * s, 65 * s, 110 * s)
    horizon.cubicTo(77 * s, 110 * s, 87 * s, 106 * s, 95 * s, 99 * s)
    horizon_w = {1: 5.0, 2: 6.0, 3: 8.0}[tier] * s
    p.setPen(QtGui.QPen(QtGui.QColor("#E2F1FF"), horizon_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
    p.drawPath(horizon)

    # Earth surface fill (tier 1-2)
    if tier <= 2:
        earth = QtGui.QPainterPath()
        earth.moveTo(28 * s, 90 * s)
        earth.cubicTo(37 * s, 103 * s, 50 * s, 110 * s, 65 * s, 110 * s)
        earth.cubicTo(77 * s, 110 * s, 87 * s, 106 * s, 95 * s, 99 * s)
        earth.lineTo(100 * s, 116 * s)
        earth.lineTo(20 * s, 116 * s)
        earth.closeSubpath()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(14, 59, 95, 55 if tier == 1 else 70))
        p.drawPath(earth)

    # --- Crosshairs (tier 1-2 only) ----------------------------------------
    if tier <= 2:
        cw = {1: 3.0, 2: 4.0}[tier] * s
        p.setPen(QtGui.QPen(
            QtGui.QColor(226, 241, 255, 165 if tier == 1 else 210),
            cw, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
        ))
        p.drawLine(QtCore.QPointF(32 * s, 90 * s), QtCore.QPointF(96 * s, 90 * s))
        p.drawLine(QtCore.QPointF(64 * s, 58 * s), QtCore.QPointF(64 * s, 110 * s))

    # --- RAS telescope -----------------------------------------------------
    if tier <= 2:
        # Parabolic dish
        p.save()
        p.translate(64 * s, 51 * s)
        # Pedestal
        ped_w = max(1.0, 2.5 * s)
        p.setPen(QtGui.QPen(QtGui.QColor("#C7D8E8"), ped_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawLine(QtCore.QPointF(0, 4 * s), QtCore.QPointF(0, 8 * s))
        # Dish curve
        dish = QtGui.QPainterPath()
        dish.moveTo(-7.5 * s, 2.5 * s)
        dish.quadTo(-7 * s, -4 * s, 0, -7 * s)
        dish.quadTo(7 * s, -4 * s, 7.5 * s, 2.5 * s)
        dish_w = max(1.0, 2.8 * s)
        p.setPen(QtGui.QPen(QtGui.QColor("#E2F1FF"), dish_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawPath(dish)
        # Feed point
        feed_r = max(0.8, 2.0 * s)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#E2F1FF"))
        p.drawEllipse(QtCore.QPointF(0, -2 * s), feed_r, feed_r)
        p.restore()
    else:
        # Tier 3: telescope as a bright dot
        dot_r = max(1.5, 4.5 * s)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#E2F1FF"))
        p.drawEllipse(QtCore.QPointF(64 * s, 51 * s), dot_r, dot_r)

    # --- Satellite ---------------------------------------------------------
    p.save()
    p.translate(92 * s, 50 * s)
    p.rotate(20)
    p.setPen(QtCore.Qt.NoPen)

    if tier == 3:
        # Minimal: bright cyan dot with tiny wings
        body_r = max(2.0, 5.5 * s)
        p.setBrush(QtGui.QColor("#38BDF8"))
        p.drawEllipse(QtCore.QPointF(0, 0), body_r, body_r)
        # Tiny wing lines for satellite silhouette
        wing_w = max(1.0, 3.0 * s)
        p.setPen(QtGui.QPen(QtGui.QColor("#E0F2FE"), wing_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        span = body_r * 2.5
        p.drawLine(QtCore.QPointF(-span, 0), QtCore.QPointF(-body_r, 0))
        p.drawLine(QtCore.QPointF(body_r, 0), QtCore.QPointF(span, 0))
    elif tier == 2:
        # Medium: body + clear panels, no antenna detail
        body_half = 6.0 * s
        p.setBrush(QtGui.QColor("#38BDF8"))
        p.drawRoundedRect(
            QtCore.QRectF(-body_half, -body_half, body_half * 2, body_half * 2),
            2.5 * s, 2.5 * s,
        )
        p.setBrush(QtGui.QColor("#E0F2FE"))
        p.drawRoundedRect(QtCore.QRectF(-17 * s, -3 * s, 7.5 * s, 6 * s), 1.0 * s, 1.0 * s)
        p.drawRoundedRect(QtCore.QRectF(9.5 * s, -3 * s, 7.5 * s, 6 * s), 1.0 * s, 1.0 * s)
        # Struts
        strut_w = max(0.6, 1.0 * s)
        p.setPen(QtGui.QPen(QtGui.QColor(148, 163, 184, 180), strut_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawLine(QtCore.QPointF(-body_half, 0), QtCore.QPointF(-9.5 * s, 0))
        p.drawLine(QtCore.QPointF(body_half, 0), QtCore.QPointF(9.5 * s, 0))
    else:
        # Full detail: body, antenna, panels with detail lines, struts
        body_half = 6.5 * s
        p.setBrush(QtGui.QColor("#38BDF8"))
        p.drawRoundedRect(
            QtCore.QRectF(-body_half, -body_half, body_half * 2, body_half * 2),
            3 * s, 3 * s,
        )
        # Antenna mast + dish
        p.setBrush(QtGui.QColor("#7DD3FC"))
        p.drawEllipse(QtCore.QPointF(0, -8.5 * s), 1.8 * s, 1.8 * s)
        p.setPen(QtGui.QPen(QtGui.QColor("#7DD3FC"), 1.2 * s, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawLine(QtCore.QPointF(0, -body_half), QtCore.QPointF(0, -8.5 * s))
        p.setPen(QtCore.Qt.NoPen)
        # Solar panels
        p.setBrush(QtGui.QColor("#E0F2FE"))
        p.drawRoundedRect(QtCore.QRectF(-18 * s, -3 * s, 8 * s, 6 * s), 1.2 * s, 1.2 * s)
        p.drawRoundedRect(QtCore.QRectF(10 * s, -3 * s, 8 * s, 6 * s), 1.2 * s, 1.2 * s)
        # Panel center division lines
        p.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 80), 0.5 * s))
        p.drawLine(QtCore.QPointF(-14 * s, -3 * s), QtCore.QPointF(-14 * s, 3 * s))
        p.drawLine(QtCore.QPointF(14 * s, -3 * s), QtCore.QPointF(14 * s, 3 * s))
        # Panel struts
        p.setPen(QtGui.QPen(QtGui.QColor(148, 163, 184, 160), 1.0 * s, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawLine(QtCore.QPointF(-body_half, 0), QtCore.QPointF(-10 * s, 0))
        p.drawLine(QtCore.QPointF(body_half, 0), QtCore.QPointF(10 * s, 0))
    p.restore()

    # --- Signal waves (tier 1 only) ----------------------------------------
    if tier == 1:
        p.setBrush(QtCore.Qt.NoBrush)
        for wdx, wdy, wa, ww in [(82, 55, 46, 1.0), (80, 58, 30, 0.8)]:
            wave = QtGui.QPainterPath()
            wave.moveTo(wdx * s, wdy * s)
            wave.quadTo((wdx - 6) * s, (wdy + 4) * s, (wdx - 10) * s, (wdy + 3) * s)
            p.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, wa), ww * s, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawPath(wave)

    p.end()
    return pixmap


def _build_fallback_app_icon() -> QtGui.QIcon:
    """Build a multi-size icon with tier-appropriate detail at each size.

    Used when the SVG file is missing.  Also used by
    ``_build_multi_size_icon`` cannot produce a good result from the SVG.
    """
    icon = QtGui.QIcon()
    for size in _APP_ICON_SIZES:
        pm = _render_icon_to_pixmap(int(size))
        if not pm.isNull():
            icon.addPixmap(pm)
    return icon


def resolve_app_icon_path() -> Path | None:
    """Return the preferred bundled app icon path for the current platform."""
    if sys.platform == "win32" and _WINDOWS_APP_ICON_PATH.exists():
        return _WINDOWS_APP_ICON_PATH
    if _APP_ICON_PATH.exists():
        return _APP_ICON_PATH
    return None


_APP_ICON_CACHE: QtGui.QIcon | None = None


def load_app_icon() -> QtGui.QIcon:
    """Return a multi-size app icon with detail adapted per size.

    Large sizes (96+) use the SVG when available for maximum crispness.
    Small/medium sizes always use the programmatic tier renderer because
    the SVG's fine details become noise below 64 px.
    """
    global _APP_ICON_CACHE
    if _APP_ICON_CACHE is not None and not _APP_ICON_CACHE.isNull():
        return _APP_ICON_CACHE
    icon = QtGui.QIcon()
    # Use the tier-aware programmatic renderer for every size — it
    # adapts stroke widths, removes fine detail at small sizes, and
    # adds star field / signal waves only at large sizes.
    for size in _APP_ICON_SIZES:
        pm = _render_icon_to_pixmap(int(size))
        if not pm.isNull():
            icon.addPixmap(pm)
    # For the largest sizes, overlay the SVG if available — it has
    # smoother curves than QPainterPath at 256+ px.
    icon_path = resolve_app_icon_path()
    if icon_path is not None and str(icon_path).endswith(".svg") and QSvgRenderer is not None:
        renderer = QSvgRenderer(str(icon_path))
        if renderer.isValid():
            for size in (128, 256):
                svg_pm = QtGui.QPixmap(size, size)
                svg_pm.fill(QtCore.Qt.transparent)
                svg_painter = QtGui.QPainter(svg_pm)
                svg_painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                renderer.render(svg_painter)
                svg_painter.end()
                if not svg_pm.isNull():
                    icon.addPixmap(svg_pm)
    if icon.isNull():
        icon = _build_fallback_app_icon()
    _APP_ICON_CACHE = icon
    return _APP_ICON_CACHE


# ---------------------------------------------------------------------------
# Splash screen
# ---------------------------------------------------------------------------

def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _ease_in_out_quad(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 2.0 * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def _eval_cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate a cubic Bezier curve at parameter *t* in [0, 1]."""
    u = 1.0 - t
    x = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0]
    y = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]
    return (x, y)


def _eval_cubic_bezier_tangent(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Tangent (derivative) of a cubic Bezier at *t*."""
    u = 1.0 - t
    dx = 3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0])
    dy = 3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
    return (dx, dy)


def _eval_orbit_path(
    t_param: float,
    scale: float,
) -> tuple[float, float, float]:
    """Evaluate position and heading angle on the two-segment orbit arc.

    The orbit is composed of two cubic Beziers (matching the SVG icon):
      Segment 1: (24,76) -> controls -> (72,47)    t_param in [0, 0.65]
      Segment 2: (72,47) -> controls -> (99,53)     t_param in [0.65, 1.0]

    Returns (x, y, heading_deg) in scaled icon coordinates.
    """
    # Segment 1 control points (in 128-unit icon space)
    s1_p0 = (24.0, 76.0)
    s1_p1 = (34.0, 58.0)
    s1_p2 = (52.0, 47.0)
    s1_p3 = (72.0, 47.0)
    # Segment 2 control points
    s2_p0 = (72.0, 47.0)
    s2_p1 = (83.0, 47.0)
    s2_p2 = (92.0, 49.5)
    s2_p3 = (99.0, 53.0)

    split = 0.65  # fraction of path length in segment 1
    if t_param <= split:
        local_t = t_param / split
        px, py = _eval_cubic_bezier(s1_p0, s1_p1, s1_p2, s1_p3, local_t)
        dx, dy = _eval_cubic_bezier_tangent(s1_p0, s1_p1, s1_p2, s1_p3, local_t)
    else:
        local_t = (t_param - split) / (1.0 - split)
        px, py = _eval_cubic_bezier(s2_p0, s2_p1, s2_p2, s2_p3, local_t)
        dx, dy = _eval_cubic_bezier_tangent(s2_p0, s2_p1, s2_p2, s2_p3, local_t)

    heading_deg = math.degrees(math.atan2(dy, dx))
    return (px * scale, py * scale, heading_deg)


class ScepterSplashScreen(QtWidgets.QSplashScreen):
    """Animated branded splash screen with staged progress feedback.

    Features:

    * **Non-intrusive** — the splash does not use ``WindowStaysOnTopHint``
      so it will not forcefully obscure other applications.
    * **Never frozen** — a continuous 60 fps animation timer keeps the
      event loop pumping so the OS never marks the window as
      unresponsive.
    * **Animated logo assembly** — orbit rings sweep in, the satellite
      slides into place, text fades up, and a shimmer effect plays
      across the progress bar.
    * **Smooth progress** — progress bar interpolates toward the target
      value with an ease-out curve instead of jumping.
    * **Fade in / fade out** — opacity transitions on show and close.
    """

    def __init__(
        self,
        icon: QtGui.QIcon,
        *,
        minimum_visible_ms: int = _SPLASH_MIN_VISIBLE_MS,
    ) -> None:
        # Build the static background pixmap (gradient + decorative circles).
        bg_pixmap = self._build_background_pixmap()
        super().__init__(
            bg_pixmap,
            QtCore.Qt.Window
            | QtCore.Qt.FramelessWindowHint,
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowIcon(icon)

        self._minimum_visible_ms = max(int(minimum_visible_ms), 0)
        self._shown_monotonic: float | None = None
        self._status_text = "Starting..."
        self._progress_value = 0.0
        self._display_progress = 0.0
        self._progress_bar_rect = QtCore.QRectF(32.0, 184.0, 416.0, 10.0)
        self._progress_value_rect = QtCore.QRectF(32.0, 184.0, 0.0, 10.0)
        self._fade_animation: QtCore.QPropertyAnimation | None = None
        self._icon = icon
        self._svg_renderer: QSvgRenderer | None = None
        if QSvgRenderer is not None and _APP_ICON_PATH.exists():
            r = QSvgRenderer(str(_APP_ICON_PATH))
            if r.isValid():
                self._svg_renderer = r

        # Animation state
        self._anim_start = monotonic()
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setInterval(_SPLASH_ANIM_INTERVAL_MS)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start()

    # -- Static background (no animated elements) ----------------------------

    @staticmethod
    def _build_background_pixmap() -> QtGui.QPixmap:
        dpr = 1.0
        app = QtGui.QGuiApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
            if screen is not None:
                dpr = max(float(screen.devicePixelRatio()), 1.0)
        lw, lh = 480, 220
        pixmap = QtGui.QPixmap(int(lw * dpr), int(lh * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pixmap)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(QtCore.QRectF(0, 0, lw, lh), _SPLASH_CORNER_RADIUS, _SPLASH_CORNER_RADIUS)
        p.setClipPath(clip)
        # Deeper gradient background
        grad = QtGui.QLinearGradient(0, 0, lw * 0.3, lh)
        grad.setColorAt(0.0, QtGui.QColor("#0d1b2e"))
        grad.setColorAt(0.5, QtGui.QColor("#0a1628"))
        grad.setColorAt(1.0, QtGui.QColor("#070e1a"))
        p.fillRect(QtCore.QRectF(0, 0, lw, lh), grad)
        # Subtle nebula-like radial glows
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(14, 165, 233, 18))
        p.drawEllipse(-40, 110, 200, 200)
        p.setBrush(QtGui.QColor(56, 189, 248, 10))
        p.drawEllipse(320, -50, 180, 180)
        # Star field (static)
        _star_data = [
            (45, 25, 1.0, 0.5), (120, 40, 0.8, 0.4), (210, 15, 1.1, 0.45),
            (310, 30, 0.7, 0.35), (380, 55, 0.9, 0.5), (430, 20, 0.8, 0.4),
            (70, 80, 0.6, 0.3), (170, 70, 0.9, 0.45), (260, 45, 0.7, 0.35),
            (350, 75, 1.0, 0.4), (450, 65, 0.8, 0.3), (25, 140, 0.7, 0.25),
            (140, 120, 0.6, 0.3), (290, 130, 0.8, 0.25), (400, 110, 0.7, 0.3),
        ]
        for sx, sy, sr, sa in _star_data:
            p.setBrush(QtGui.QColor(203, 213, 225, int(255 * sa)))
            p.drawEllipse(QtCore.QPointF(sx, sy), sr, sr)
        # Fine border glow inside the rounded rect
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 20), 1.0))
        p.drawRoundedRect(
            QtCore.QRectF(0.5, 0.5, lw - 1.0, lh - 1.0),
            _SPLASH_CORNER_RADIUS - 0.5, _SPLASH_CORNER_RADIUS - 0.5,
        )
        p.end()
        return pixmap

    # -- Qt overrides --------------------------------------------------------

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self._shown_monotonic = monotonic()
        self._anim_start = monotonic()
        self.setWindowOpacity(0.0)
        fade_in = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        fade_in.setDuration(350)
        fade_in.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        super().showEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(QtCore.QRectF(self.rect()), _SPLASH_CORNER_RADIUS, _SPLASH_CORNER_RADIUS)
        painter.setClipPath(clip)
        pm = self.pixmap()
        if not pm.isNull():
            painter.drawPixmap(0, 0, pm)
        self._draw_animated_content(painter)
        painter.end()

    # -- Animation tick (keeps event loop alive) -----------------------------

    def _tick(self) -> None:
        """Called at ~60 fps.  Drives progress interpolation and repaint."""
        target = self._progress_value
        current = self._display_progress
        if abs(target - current) > 0.001:
            self._display_progress = current + (target - current) * 0.15
        else:
            self._display_progress = target
        width = self._progress_bar_rect.width() * self._display_progress
        self._progress_value_rect = QtCore.QRectF(
            self._progress_bar_rect.left(),
            self._progress_bar_rect.top(),
            max(0.0, width),
            self._progress_bar_rect.height(),
        )
        self.update()
        # Pump the event loop so the OS sees the window as responsive
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents, 5)

    # -- Animated content drawing --------------------------------------------

    @staticmethod
    def _preferred_font_family() -> str:
        """Return the best available UI font for the current platform."""
        if sys.platform == "win32":
            return "Segoe UI"
        if sys.platform == "darwin":
            return "SF Pro Display"
        # Linux / other — try common sans-serif fonts
        db = QtGui.QFontDatabase()
        for candidate in ("Inter", "Cantarell", "Noto Sans", "Ubuntu", "DejaVu Sans"):
            if candidate in db.families():
                return candidate
        return "sans-serif"

    def _draw_logo_without_satellite(
        self, painter: QtGui.QPainter, x: float, y: float, size: float,
    ) -> None:
        """Draw the SCEPTer logo elements except the satellite.

        Uses tier-2 detail level (appropriate for the ~44 px splash logo):
        orbit arc, horizon, crosshairs, telescope dish, Earth fill.
        No star field, no signal waves.
        """
        s = size / 128.0
        painter.save()
        painter.translate(x, y)

        # Background rounded rect
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#08152A"))
        painter.drawRoundedRect(QtCore.QRectF(8 * s, 8 * s, 112 * s, 112 * s), 34 * s, 34 * s)

        # Border
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 70), max(0.8, 1.5 * s)))
        painter.drawRoundedRect(QtCore.QRectF(9 * s, 9 * s, 110 * s, 110 * s), 33 * s, 33 * s)

        # Upper orbit arc (cyan, thick for readability at 44px)
        orbit_path = QtGui.QPainterPath()
        orbit_path.moveTo(24 * s, 76 * s)
        orbit_path.cubicTo(34 * s, 58 * s, 52 * s, 47 * s, 72 * s, 47 * s)
        orbit_path.cubicTo(83 * s, 47 * s, 92 * s, 49.5 * s, 99 * s, 53 * s)
        painter.setPen(QtGui.QPen(QtGui.QColor("#38BDF8"), 8.0 * s, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(orbit_path)

        # Lower horizon arc (white)
        horizon_path = QtGui.QPainterPath()
        horizon_path.moveTo(28 * s, 90 * s)
        horizon_path.cubicTo(37 * s, 103 * s, 50 * s, 110 * s, 65 * s, 110 * s)
        horizon_path.cubicTo(77 * s, 110 * s, 87 * s, 106 * s, 95 * s, 99 * s)
        painter.setPen(QtGui.QPen(QtGui.QColor("#E2F1FF"), 6.0 * s, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawPath(horizon_path)

        # Earth surface fill
        earth_fill = QtGui.QPainterPath()
        earth_fill.moveTo(28 * s, 90 * s)
        earth_fill.cubicTo(37 * s, 103 * s, 50 * s, 110 * s, 65 * s, 110 * s)
        earth_fill.cubicTo(77 * s, 110 * s, 87 * s, 106 * s, 95 * s, 99 * s)
        earth_fill.lineTo(100 * s, 116 * s)
        earth_fill.lineTo(20 * s, 116 * s)
        earth_fill.closeSubpath()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(14, 59, 95, 70))
        painter.drawPath(earth_fill)

        # Crosshairs
        cw = max(1.0, 4.0 * s)
        painter.setPen(QtGui.QPen(QtGui.QColor(226, 241, 255, 180), cw, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawLine(QtCore.QPointF(32 * s, 90 * s), QtCore.QPointF(96 * s, 90 * s))
        painter.drawLine(QtCore.QPointF(64 * s, 58 * s), QtCore.QPointF(64 * s, 110 * s))

        # RAS telescope dish
        painter.save()
        painter.translate(64 * s, 51 * s)
        ped_w = max(1.0, 2.5 * s)
        painter.setPen(QtGui.QPen(QtGui.QColor("#C7D8E8"), ped_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawLine(QtCore.QPointF(0, 4 * s), QtCore.QPointF(0, 8 * s))
        dish = QtGui.QPainterPath()
        dish.moveTo(-7.5 * s, 2.5 * s)
        dish.quadTo(-7 * s, -4 * s, 0, -7 * s)
        dish.quadTo(7 * s, -4 * s, 7.5 * s, 2.5 * s)
        dish_w = max(1.0, 2.8 * s)
        painter.setPen(QtGui.QPen(QtGui.QColor("#E2F1FF"), dish_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawPath(dish)
        feed_r = max(0.8, 2.0 * s)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#E2F1FF"))
        painter.drawEllipse(QtCore.QPointF(0, -2 * s), feed_r, feed_r)
        painter.restore()

        painter.restore()

    def _draw_satellite_sprite(
        self,
        painter: QtGui.QPainter,
        x: float,
        y: float,
        heading_deg: float,
        sat_scale: float,
        opacity: float,
    ) -> None:
        """Draw a single satellite at (x, y) with the given heading."""
        if opacity <= 0.01:
            return
        painter.save()
        painter.setOpacity(opacity)
        painter.translate(x, y)
        painter.rotate(heading_deg)
        ss = sat_scale
        painter.setPen(QtCore.Qt.NoPen)
        # Body
        painter.setBrush(QtGui.QColor("#38bdf8"))
        painter.drawRoundedRect(
            QtCore.QRectF(-3.0 * ss, -3.0 * ss, 6.0 * ss, 6.0 * ss),
            1.4 * ss, 1.4 * ss,
        )
        # Solar panels
        painter.setBrush(QtGui.QColor("#E0F2FE"))
        painter.drawRoundedRect(
            QtCore.QRectF(-8.5 * ss, -1.4 * ss, 4.0 * ss, 2.8 * ss),
            0.5 * ss, 0.5 * ss,
        )
        painter.drawRoundedRect(
            QtCore.QRectF(4.5 * ss, -1.4 * ss, 4.0 * ss, 2.8 * ss),
            0.5 * ss, 0.5 * ss,
        )
        # Struts
        painter.setPen(QtGui.QPen(QtGui.QColor(148, 163, 184, 140), 0.4 * ss, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawLine(QtCore.QPointF(-3.0 * ss, 0), QtCore.QPointF(-4.5 * ss, 0))
        painter.drawLine(QtCore.QPointF(3.0 * ss, 0), QtCore.QPointF(4.5 * ss, 0))
        painter.restore()

    def _draw_animated_content(self, painter: QtGui.QPainter) -> None:
        t = monotonic() - self._anim_start
        font_family = self._preferred_font_family()

        # --- Phase 1 (0.0–0.5s): Logo (without satellite) fades in --------
        icon_size = 44.0
        icon_x, icon_y = 30.0, 42.0
        icon_alpha = _ease_out_cubic(min(1.0, t / 0.5))

        painter.save()
        painter.setOpacity(icon_alpha)
        self._draw_logo_without_satellite(painter, icon_x, icon_y, icon_size)
        painter.restore()

        # --- Satellites following the orbit arc path -----------------------
        # Two satellites travel along the cubic bezier orbit arc from left
        # to right, wrapping continuously.  They move in the same direction
        # at slightly different spacings to look like a real constellation.
        scale = icon_size / 128.0
        sat_alpha_f = _ease_out_cubic(min(1.0, t / 0.6))
        travel_period = 4.0  # seconds for a full left→right traverse
        sat_scale = 1.6 * scale

        # Satellite phase offsets (fraction of path apart)
        sat_offsets = [0.0, 0.42]
        for i, offset in enumerate(sat_offsets):
            raw_t = (t / travel_period + offset) % 1.0
            px, py, heading = _eval_orbit_path(raw_t, scale)
            sx = icon_x + px
            sy = icon_y + py
            # Second satellite is slightly smaller and dimmer
            s_scale = sat_scale * (1.0 if i == 0 else 0.75)
            s_alpha = sat_alpha_f * icon_alpha * (1.0 if i == 0 else 0.55)
            self._draw_satellite_sprite(painter, sx, sy, heading, s_scale, s_alpha)

        # --- Phase 2 (0.15–0.6s): Title text slides in from right ----------
        text_left = int(icon_x + icon_size + 14)
        title_t = _ease_out_cubic(max(0.0, min(1.0, (t - 0.15) / 0.45)))
        title_offset = int(30 * (1.0 - title_t))

        painter.save()
        painter.setOpacity(title_t)
        painter.setPen(QtGui.QColor("#f8fafc"))
        title_font = QtGui.QFont(font_family, 24, QtGui.QFont.Bold)
        title_font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 0.5)
        painter.setFont(title_font)
        painter.drawText(
            QtCore.QRect(text_left + title_offset, 42, 380, 44),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "SCEPTer",
        )
        painter.restore()

        # --- Phase 3 (0.35–0.8s): Subtitle fades in below ------------------
        sub_t = _ease_out_cubic(max(0.0, min(1.0, (t - 0.35) / 0.45)))
        sub_offset = int(15 * (1.0 - sub_t))

        painter.save()
        painter.setOpacity(sub_t)
        painter.setPen(QtGui.QColor("#93c5fd"))
        painter.setFont(QtGui.QFont(font_family, 11))
        painter.drawText(
            QtCore.QRect(text_left + sub_offset, 88, 380, 22),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "Simulation Studio",
        )
        painter.restore()

        # --- Version badge (0.5s+) -----------------------------------------
        ver_t = _ease_out_cubic(max(0.0, min(1.0, (t - 0.5) / 0.3)))
        if ver_t > 0.0:
            painter.save()
            painter.setOpacity(ver_t * 0.7)
            badge_font = QtGui.QFont(font_family, 9)
            painter.setFont(badge_font)
            fm = QtGui.QFontMetrics(badge_font)
            tag = appinfo.APP_VERSION_TAG
            tw = fm.horizontalAdvance(tag) + 12
            badge_x = text_left + sub_offset
            badge_y = 112
            badge_rect = QtCore.QRectF(badge_x, badge_y, tw, 18)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(56, 189, 248, 30))
            painter.drawRoundedRect(badge_rect, 4, 4)
            painter.setPen(QtGui.QColor(147, 197, 253, 200))
            painter.drawText(
                badge_rect.toRect(),
                QtCore.Qt.AlignCenter,
                tag,
            )
            painter.restore()

        # --- Decorative orbit arc sweeps across the splash (0.5s+) ---------
        arc_t = max(0.0, min(1.0, (t - 0.5) / 0.8))
        if arc_t > 0.0:
            arc_span = int(arc_t * 280) * 16
            # Double arc for depth
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(
                QtGui.QColor(14, 165, 233, int(30 * _ease_out_cubic(arc_t))),
                1.5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
            ))
            painter.drawArc(QtCore.QRectF(-20.0, 105.0, 520.0, 90.0), 160 * 16, arc_span)
            painter.setPen(QtGui.QPen(
                QtGui.QColor(14, 165, 233, int(15 * _ease_out_cubic(arc_t))),
                0.8, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
            ))
            painter.drawArc(QtCore.QRectF(-30.0, 115.0, 540.0, 80.0), 155 * 16, arc_span)

        # --- Status text (visible after 0.3s) ------------------------------
        status_t = _ease_out_cubic(max(0.0, min(1.0, (t - 0.3) / 0.3)))
        painter.setPen(QtGui.QColor(203, 213, 225, int(255 * status_t)))
        painter.setFont(QtGui.QFont(font_family, 10))
        painter.drawText(
            QtCore.QRectF(32.0, 160.0, 416.0, 20.0),
            int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter),
            str(self._status_text),
        )

        # --- Progress bar ---------------------------------------------------
        bar_t = _ease_out_cubic(max(0.0, min(1.0, (t - 0.4) / 0.3)))
        painter.save()
        painter.setOpacity(bar_t)
        painter.setPen(QtCore.Qt.NoPen)
        # Track
        painter.setBrush(QtGui.QColor(148, 163, 184, 36))
        painter.drawRoundedRect(self._progress_bar_rect, 5.0, 5.0)
        # Fill
        fill_w = self._progress_value_rect.width()
        if fill_w > 0.0:
            # Outer glow
            glow = self._progress_value_rect.adjusted(-1.5, -3.0, 1.5, 3.0)
            painter.setBrush(QtGui.QColor(14, 165, 233, 35))
            painter.drawRoundedRect(glow, 7.0, 7.0)
            # Gradient fill
            bar_grad = QtGui.QLinearGradient(
                self._progress_value_rect.left(), 0,
                self._progress_value_rect.right(), 0,
            )
            bar_grad.setColorAt(0.0, QtGui.QColor("#0369a1"))
            bar_grad.setColorAt(0.5, QtGui.QColor("#0ea5e9"))
            bar_grad.setColorAt(1.0, QtGui.QColor("#38bdf8"))
            painter.setBrush(bar_grad)
            painter.drawRoundedRect(self._progress_value_rect, 5.0, 5.0)
            # Inner highlight line at top of bar
            highlight = self._progress_value_rect.adjusted(2.0, 1.0, -2.0, -self._progress_value_rect.height() + 3.0)
            painter.setBrush(QtGui.QColor(255, 255, 255, 25))
            painter.drawRoundedRect(highlight, 2.0, 2.0)
            # Shimmer highlight sweeping across the bar
            shimmer_period = 2.2
            shimmer_phase = (t % shimmer_period) / shimmer_period
            shimmer_x = self._progress_bar_rect.left() + shimmer_phase * self._progress_bar_rect.width()
            shimmer_w = 70.0
            shimmer_rect = QtCore.QRectF(
                shimmer_x - shimmer_w / 2,
                self._progress_value_rect.top(),
                shimmer_w,
                self._progress_value_rect.height(),
            )
            shimmer_clip = QtGui.QPainterPath()
            shimmer_clip.addRoundedRect(self._progress_value_rect, 5.0, 5.0)
            painter.setClipPath(shimmer_clip)
            shimmer_grad = QtGui.QLinearGradient(
                shimmer_rect.left(), 0, shimmer_rect.right(), 0,
            )
            shimmer_grad.setColorAt(0.0, QtGui.QColor(255, 255, 255, 0))
            shimmer_grad.setColorAt(0.5, QtGui.QColor(255, 255, 255, 50))
            shimmer_grad.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
            painter.setBrush(shimmer_grad)
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRect(shimmer_rect)
        painter.restore()

    # -- Public API ----------------------------------------------------------

    def set_stage(self, *, progress: float, status_text: str) -> None:
        self._progress_value = max(0.0, min(float(progress), 1.0))
        self._display_progress = self._progress_value  # snap to target
        self._status_text = str(status_text)
        self._tick()  # update _progress_value_rect geometry
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def finish_with_min_duration(self, window: QtWidgets.QWidget) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        remaining_ms = 0
        if self._shown_monotonic is not None and self._minimum_visible_ms > 0:
            elapsed_ms = (monotonic() - self._shown_monotonic) * 1000.0
            remaining_ms = max(0, int(self._minimum_visible_ms - elapsed_ms))
        if remaining_ms > 0:
            wait_loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(remaining_ms, wait_loop.quit)
            wait_loop.exec()
        self._anim_timer.stop()
        self._fade_out_and_finish(window)

    def _fade_out_and_finish(self, window: QtWidgets.QWidget) -> None:
        fade_loop = QtCore.QEventLoop()
        anim = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(_SPLASH_FADE_OUT_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        anim.setStartValue(self.windowOpacity())
        anim.setEndValue(0.0)
        anim.finished.connect(fade_loop.quit)
        self._fade_animation = anim
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        fade_loop.exec()
        self.finish(window)


def build_startup_splash(icon: QtGui.QIcon) -> QtGui.QPixmap:
    """Build the background pixmap (for backward compatibility)."""
    return ScepterSplashScreen._build_background_pixmap()


def create_startup_splash(
    icon: QtGui.QIcon,
    *,
    minimum_visible_ms: int = _SPLASH_MIN_VISIBLE_MS,
) -> ScepterSplashScreen:
    """Create the staged startup splash screen used by the top-level launcher."""
    return ScepterSplashScreen(icon, minimum_visible_ms=minimum_visible_ms)


__all__ = [
    "apply_windows_window_icon",
    "build_startup_splash",
    "configure_application",
    "configure_windows_shell_identity",
    "create_startup_splash",
    "load_app_icon",
    "resolve_app_icon_path",
]
