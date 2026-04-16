"""Shared application identity metadata for the GUI and notebooks."""

from __future__ import annotations

APP_NAME = "SCEPTer"
GUI_APP_NAME = "SCEPTer GUI"
APP_VERSION = "0.25.2"
APP_VERSION_TAG = f"v{APP_VERSION}"
GUI_WINDOW_TITLE_BASE = f"{GUI_APP_NAME} {APP_VERSION_TAG}"

ABOUT_TITLE = f"About {APP_NAME}"
ABOUT_TEXT = (
    f"{GUI_APP_NAME} {APP_VERSION_TAG}\n\n"
    "SCEPTer — simulating satellite constellation emission patterns for radio telescopes.\n\n"
    "This desktop GUI and the maintained workflow notebooks share the same "
    "simulation and post-processing capabilities for the v0.25.2 release.\n\n"
    "Authors / contacts:\n"
    "  • Boris Sorokin <boris.sorokin@skao.int>  <mralin@protonmail.com>\n"
    "  • Hao (Harry) Qiu (GitHub: https://github.com/hqiu-nju)\n\n"
    "Licence:\n"
    "  GNU General Public License v3.0 or later (GPL-3.0-or-later).\n"
    "  This program is distributed without any warranty, to the extent "
    "permitted by law.\n\n"
    "Third‑party software / data notices (highlights):\n"
    "  • pycraf is GPLv3 software; some functionality may rely on ITU-derived "
    "data that can carry additional terms.\n"
    "  • cysgp4 (GPL-3.0-or-later; https://github.com/bwinkel/cysgp4) "
    "Cython-wraps Daniel Warner's SGP4 C++ library (Apache-2.0; "
    "https://github.com/dnwrnr/sgp4), and ships its own notices for the "
    "upstream Apache-2.0 / BSD-3-Clause attributions.\n"
    "  • The 3D viewer is built on VTK (BSD-3-Clause) via pyvista / "
    "pyvistaqt (MIT).\n"
    "  • The 3D viewer can use a bundled NASA/GSFC Earth texture; reuse and "
    "redistribution remain subject to NASA media usage guidance and local "
    "jurisdiction review.\n"
    "  • Scientific and plotting dependencies are summarised in "
    "THIRD_PARTY_NOTICES.md.\n\n"
    "SGP4 propagation references:\n"
    "  • Hoots, F. R. & Roehrich, R. L. (December 1980). Models for "
    "Propagation of NORAD Element Sets. Project Spacetrack Report No. 3, "
    "Aerospace Defense Command (ADC). Archival copy: "
    "https://celestrak.org/NORAD/documentation/spacetrk.pdf\n"
    "  • Vallado, D. A., Crawford, P., Hujsak, R. & Kelso, T. S. (2006, "
    "rev. 2012). Revisiting Spacetrack Report #3. AIAA/AAS Astrodynamics "
    "Specialist Conference, AIAA 2006-6753. Reference code and errata: "
    "https://celestrak.org/publications/AIAA/2006-6753/\n"
    "  • Warner, Daniel J. — SGP4 C++ library (\"dwarner\"), Apache-2.0. "
    "Upstream: https://github.com/dnwrnr/sgp4  —  wrapped by cysgp4 and "
    "used as the default SCEPTer propagation backend.\n"
    "  • Winkel, B. — cysgp4 (GPL-3.0-or-later), Cython wrapper around "
    "Warner's library. Upstream: https://github.com/bwinkel/cysgp4\n"
    "  • Kelso, T. S. — CelesTrak TLE/SGP4 reference code, errata, and "
    "documentation archive: https://celestrak.org\n\n"
    "See LICENSE, AUTHORS.md, THIRD_PARTY_NOTICES.md, and LICENSES/ for the "
    "full project notices.\n"
)


def format_gui_window_title(document_name: str | None = None, *, dirty: bool = False) -> str:
    """Return the branded main-window title."""
    if document_name:
        suffix = f" - {str(document_name)}"
        if dirty:
            suffix += "*"
        return f"{GUI_WINDOW_TITLE_BASE}{suffix}"
    return GUI_WINDOW_TITLE_BASE


__all__ = [
    "ABOUT_TEXT",
    "ABOUT_TITLE",
    "APP_NAME",
    "APP_VERSION",
    "APP_VERSION_TAG",
    "GUI_APP_NAME",
    "GUI_WINDOW_TITLE_BASE",
    "format_gui_window_title",
]
