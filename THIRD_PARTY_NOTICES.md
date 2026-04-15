# Third-Party Notices for SCEPTer

This file summarises third-party software and data notices relevant to
SCEPTer source distributions and related packaged artefacts.

It is provided for compliance and transparency.  It is not legal advice.

## Project licence

SCEPTer is distributed under the GNU General Public License v3.0 or later
(`GPL-3.0-or-later`).  See `LICENSE` for the project notice and `LICENSES/`
for bundled licence texts.

## Core runtime dependencies

| Package   | Licence        | Notes |
|-----------|----------------|-------|
| `numpy`   | BSD-3-Clause   | |
| `scipy`   | BSD-3-Clause   | |
| `astropy` | BSD-3-Clause   | |
| `matplotlib` | Matplotlib licence (BSD-compatible) | |
| `pandas`  | BSD-3-Clause   | |
| `h5py`    | BSD-3-Clause   | |
| `shapely` | BSD-3-Clause   | |
| `psutil`  | BSD-3-Clause   | |
| `pycraf`  | GPL-3.0        | Some `pycraf` functionality relies on ITU-derived reference data that may be subject to additional terms, including non-commercial-use restrictions.  If you redistribute bundled environments or artefacts that ship `pycraf` and its data payloads, review those terms explicitly. |
| `cysgp4`  | GPL-3.0        | `cysgp4` documents upstream Apache-2.0 and BSD-3-Clause attributions in its own packaged notices. |

## Optional / full-environment dependencies

The full GUI and notebook environment can also include:

| Package     | Licence | Category |
|-------------|---------|----------|
| `numba`     | BSD-2-Clause | GPU acceleration |
| `cupy`      | MIT | GPU acceleration |
| `PySide6`   | LGPL-3.0 / GPL-3.0 | Desktop GUI |
| `pyvista`   | MIT | 3D visualization (wraps VTK) |
| `pyvistaqt` | MIT | 3D visualization |
| `vtk` (Visualization Toolkit) | BSD-3-Clause | 3D rendering engine underlying `pyvista` / `pyvistaqt`. See <https://vtk.org> and the `Copyright.txt` shipped with the VTK Python wheel for the full upstream notice. |
| `plotly`    | MIT | Interactive plots |
| `python-kaleido` | MIT | Plotly image export |
| `cartopy`   | BSD-3-Clause | Map projections |
| `tqdm`      | MPL-2.0 / MIT | Progress bars |
| `ipywidgets` | BSD-3-Clause | Notebook widgets |
| `ipykernel` | BSD-3-Clause | Jupyter kernel |
| `seaborn`   | BSD-3-Clause | Statistical plots |
| `ipympl`    | BSD-3-Clause | Matplotlib in notebooks |
| `trame`     | Apache-2.0 | Remote 3D rendering |
| `trame-vtk` | Apache-2.0 | VTK backend for Trame |
| `trame-vuetify` | Apache-2.0 | UI framework for Trame |

Before redistributing bundled environments or appliance-style builds, verify
the exact licence obligations for the optional dependency set you ship.

## Bundled data files

| File | Source | Notes |
|------|--------|-------|
| `scepter/data/ne_10m_coastline.geojson` | Natural Earth | Public domain.  Derived from naturalearth.com 1:10m coastline dataset. |
| `scepter/data/ne_10m_land.geojson` | Natural Earth | Public domain.  Derived from naturalearth.com 1:10m land polygon dataset. |
| `scepter/data/earth_texture_nasa_flat_earth_8192.jpg` | NASA/GSFC | See below. |
| `scepter/data/satellite_app_icon.svg` | SCEPTer project | GPL-3.0-or-later. |
| `scepter/data/scepter_brand_mark.svg` | SCEPTer project | GPL-3.0-or-later. |
| `scepter/data/satellite_app_icon.ico` | SCEPTer project | GPL-3.0-or-later. |

### NASA Earth texture

- **File**: `scepter/data/earth_texture_nasa_flat_earth_8192.jpg`
- **Source**: NASA Goddard Space Flight Center Scientific Visualization Studio
  (`flat_earth_Largest_still.0330.jpg`, equirectangular Earth texture).
- **Credit**: NASA/GSFC Scientific Visualization Studio; Blue Marble Next
  Generation imagery courtesy of Reto Stockli (NASA/GSFC) and NASA's Earth
  Observatory.
- **Usage note**: Use remains subject to NASA media usage guidance,
  source-credit expectations, and no-endorsement restrictions.  Downstream
  redistributors should verify how NASA material is treated in the
  jurisdictions where they distribute or commercialise the software and its
  bundled assets.

## SGP4 propagation — algorithmic and implementation references

SCEPTer's orbit propagation is carried out by the GPU kernels in
``scepter/gpu_accel.py`` and, on the CPU path, by the ``cysgp4`` wrapper
around Dan Warner's C++ SGP4 library. Both paths expose two selectable
backends, ``"vallado"`` and ``"dwarner"``, and both ultimately derive
from the same publicly documented SGP4 analytical model.

The SCEPTer source files that implement these kernels are GPL-3.0-or-later
(as SCEPTer itself), but the **algorithms** they implement and the
**C/C++ reference code** against which they are validated come from the
upstream works listed below. When you publish scientific results, build
validation comparisons, or redistribute SCEPTer-derived binaries, please
cite and (where applicable) preserve the upstream attributions.

### Original analytical model

**Hoots, F. R., & Roehrich, R. L. (December 1980).**
*Models for Propagation of NORAD Element Sets.*
Project Spacetrack Report No. 3, Aerospace Defense Command
(ADC/DO), United States Air Force. Later compiled, redistributed, and
corrected by T. S. Kelso (CelesTrak).

- Public archival copy:
  <https://celestrak.org/NORAD/documentation/spacetrk.pdf>
- Status: U.S. Government work; effectively public domain. The TLE
  data format and the SGP4 / SDP4 analytical propagators described
  here are the foundation for every SGP4 implementation SCEPTer uses.

### Modern, corrected reference implementation (the ``"vallado"`` backend)

**Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006,
with 2012 errata updates).**
*Revisiting Spacetrack Report #3.*
AIAA/AAS Astrodynamics Specialist Conference, Keystone, Colorado,
21-24 August 2006. Paper AIAA 2006-6753.

- Paper and reference code (C++ / FORTRAN / MATLAB / Pascal):
  <https://celestrak.org/publications/AIAA/2006-6753/>
- Mirror / author page: <https://www.centerforspace.com/> (David Vallado,
  CSSI / Kayhan Space) and <https://celestrak.org/software/vallado-sw.php>.
- Distribution terms: Vallado's reference code is released for free
  reuse with attribution; consult the headers in the distributed
  source files for the exact author statement.
- SCEPTer usage: the GPU ``"vallado"`` backend is a numerical re-
  expression of Vallado's ``sgp4.cpp`` / ``sgp4unit.cpp`` arithmetic
  targeted at CuPy / Numba. The SCEPTer GPU kernels are validated
  against the CPU output of this reference code.

### C++ port used by cysgp4 (the ``"dwarner"`` backend)

**Warner, Daniel J.** — *SGP4 C++ library* ("``dnwrnr/sgp4``").

- Upstream source: <https://github.com/dnwrnr/sgp4>
- Licence: **Apache License 2.0** (see
  ``LICENSE`` in the upstream repository). The Apache-2.0 NOTICE
  requirements apply when redistributing binaries derived from this
  source — including the cysgp4 wheels that SCEPTer depends on.
- SCEPTer usage: the CPU propagation path goes through ``cysgp4``,
  which Cython-wraps Dan Warner's C++ library. The ``"dwarner"``
  method name in ``gpu_accel.py`` / ``cysgp4.propagate_many`` refers
  to this upstream. The SCEPTer GPU ``"dwarner"`` backend mirrors
  Warner's arithmetic on the device and is validated against the CPU
  ``cysgp4``/``dnwrnr`` output.

### cysgp4 — the Cython wrapper

**Winkel, Benjamin.** — *cysgp4: a wrapper of Daniel Warner's SGP4
implementation*.

- Upstream source: <https://github.com/bwinkel/cysgp4>
- Licence: **GPL-3.0-or-later**. The cysgp4 package ships its own
  ``LICENSES/`` directory documenting the Apache-2.0 / BSD-3-Clause
  attributions inherited from the vendored Warner C++ code.
- SCEPTer usage: imported as ``cysgp4`` in ``scepter/scepter_GUI.py``
  (preview builder, constellation wizard) and ``scepter/gpu_accel.py``
  (CPU fallback and validation path).

### TLE conventions, errata, and reference archive

**Kelso, T. S.** — *CelesTrak.*

- <https://celestrak.org/>
- <https://celestrak.org/publications/AIAA/2006-6753/> — co-authored
  code and errata for the Vallado reference implementation.
- <https://celestrak.org/NORAD/documentation/> — SGP4 / TLE
  documentation archive.
- SCEPTer usage: CelesTrak is the authoritative public reference for
  TLE field semantics and SGP4 numerical constants; the TLE strings
  forged by ``scepter.tleforger`` follow CelesTrak's published
  conventions (2-digit year + day-of-year epoch, column-level field
  layout, checksum algorithm).

### How to cite SCEPTer propagation output

When reporting results that depend on the SCEPTer propagation pipeline,
please credit both the analytical model authors (Hoots & Roehrich) and
the implementation authors (Vallado et al. for the ``"vallado"``
backend, Warner for the ``"dwarner"`` backend), together with CelesTrak
/ Kelso for the reference code and errata. A representative citation
block for a SCEPTer-based paper might read:

> Satellite ephemerides were propagated with SCEPTer vX.Y.Z using its
> GPU SGP4 backend (method="dwarner"), which is a CuPy/Numba re-
> expression of Daniel Warner's ``dnwrnr/sgp4`` C++ implementation
> (Apache-2.0; <https://github.com/dnwrnr/sgp4>) and is validated
> against the CPU reference. The underlying analytical model is
> Hoots & Roehrich (1980), Spacetrack Report #3; the corrected
> modern reference is Vallado, Crawford, Hujsak & Kelso (2006),
> AIAA 2006-6753, with CelesTrak-hosted reference code.

## Bundled licence texts

The `LICENSES/` directory includes reference copies of the main licence texts:

- `GPL-3.0-or-later.txt`
- `BSD-3-Clause.txt`
- `BSD-2-Clause.txt`
- `MIT.txt`
- `MPL-2.0.txt`
- `Apache-2.0.txt`

An ITU-specific licence text is not bundled.  Handle any ITU-derived data
notices separately, based on the exact data you redistribute.
