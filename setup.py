from setuptools import find_packages, setup
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="scepter",
    version="0.25.2",
    description=(
        "SCEPTer — Simulating Constellation Emission Patterns for Telescopes (radio). "
        "A modular Python toolkit for evaluating satellite constellation EPFD impact "
        "on radio astronomy observations."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SCEPTer contributors",
    author_email="boris.sorokin@skao.int",
    url="https://github.com/hqiu-nju/scepter",
    project_urls={
        "Bug Tracker": "https://github.com/hqiu-nju/scepter/issues",
        "Source Code": "https://github.com/hqiu-nju/scepter",
    },
    license="GPL-3.0-or-later",
    packages=find_packages(
        where=".",
        include=("scepter", "scepter.*"),
        exclude=("scepter.tests", "scepter.tests.*"),
    ),
    include_package_data=True,
    package_data={
        "scepter": [
            "data/*.geojson",
            "data/*.svg",
            "data/*.ico",
            "data/*.jpg",
        ],
    },
    license_files=("LICENSE", "AUTHORS.md", "THIRD_PARTY_NOTICES.md"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "astropy>=5.3",
        "pycraf>=2.0",
        "cysgp4>=0.3",
        "scipy>=1.10",
        "h5py>=3.8",
        "pandas>=2.0",
        "shapely>=2.0",
        "psutil>=5.9",
        "numba>=0.58",
        "cupy>=13.0",
        "PySide6>=6.5",
        "pyvista>=0.42",
        "pyvistaqt>=0.11",
        "plotly>=5.18",
    ],
    extras_require={
        "cartopy": [
            "cartopy>=0.22",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-timeout>=2.0",
        ],
    },
    entry_points={
        "gui_scripts": [
            "scepter-gui=gui:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "satellite",
        "constellation",
        "radio astronomy",
        "RFI",
        "EPFD",
        "simulation",
        "ITU",
        "spectrum management",
    ],
)
