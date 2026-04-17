"""Regenerate the Windows ``.ico`` for the SCEPTer app icon.

Windows' title-bar and taskbar pull the icon via ``LoadImageW`` from
``scepter/data/satellite_app_icon.ico``. A single-resolution ``.ico``
forces Windows to upscale a tiny bitmap whenever it needs a 32- or
256-px version, producing the familiar "blurry taskbar icon" look.

This script renders the programmatic tier-aware icon in
``gui_bootstrap`` at every Windows-standard resolution and packs them
into one multi-resolution ``.ico``. Run it after editing the icon
renderer or whenever the artwork needs to be regenerated.

Requires Pillow (in the full test environment) and PySide6.
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PySide6 import QtCore, QtWidgets

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scepter.gui_bootstrap import _WINDOWS_APP_ICON_PATH, _render_icon_to_pixmap  # noqa: E402

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - tool-only dep
    raise SystemExit(
        "Pillow is required to regenerate the .ico. "
        "Install it or run inside the scepter-dev-test environment."
    ) from exc


_SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    del app  # only needed to make QPixmap usable
    pil_images = []
    for size in _SIZES:
        pm = _render_icon_to_pixmap(int(size))
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.WriteOnly)
        pm.save(buf, "PNG")
        buf.close()
        pil_images.append(Image.open(BytesIO(bytes(buf.data()))).convert("RGBA"))
    target = Path(_WINDOWS_APP_ICON_PATH)
    pil_images[-1].save(
        str(target),
        format="ICO",
        sizes=[(size, size) for size in _SIZES],
    )
    print(f"Wrote {target} ({target.stat().st_size} bytes) — sizes {list(_SIZES)}")


if __name__ == "__main__":
    main()
