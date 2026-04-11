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
| `pyvista`   | MIT | 3D visualization |
| `pyvistaqt` | MIT | 3D visualization |
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
