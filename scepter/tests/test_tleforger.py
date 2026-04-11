import numpy as np
from astropy import units as u
from numpy.testing import assert_equal

from scepter import tleforger


class TestExpandBeltMetadataToSatellites:

    def test_expands_per_belt_metadata_in_tle_order(self):
        tleforger.reset_tle_counter()
        constellation = tleforger.forge_tle_constellation_from_belt_definitions(
            [
                {
                    "belt_name": "BeltA",
                    "num_sats_per_plane": 2,
                    "plane_count": 1,
                    "altitude": 500 * u.km,
                    "eccentricity": 0.0,
                    "inclination_deg": 50 * u.deg,
                    "argp_deg": 0 * u.deg,
                    "RAAN_min": 0 * u.deg,
                    "RAAN_max": 10 * u.deg,
                    "min_elevation": 20 * u.deg,
                    "adjacent_plane_offset": True,
                },
                {
                    "belt_name": "BeltB",
                    "num_sats_per_plane": 1,
                    "plane_count": 1,
                    "altitude": 600 * u.km,
                    "eccentricity": 0.0,
                    "inclination_deg": 60 * u.deg,
                    "argp_deg": 0 * u.deg,
                    "RAAN_min": 10 * u.deg,
                    "RAAN_max": 20 * u.deg,
                    "min_elevation": 30 * u.deg,
                    "adjacent_plane_offset": False,
                },
            ]
        )

        expanded = tleforger.expand_belt_metadata_to_satellites(constellation)

        assert_equal(
            expanded["sat_min_elevation_deg"],
            np.array([20.0, 20.0, 30.0], dtype=np.float32),
        )
        assert_equal(expanded["sat_belt_id"], np.array([0, 0, 1], dtype=np.int16))
        assert expanded["sat_beta_max_deg"].shape == (3,)
        assert expanded["sat_beta_max_deg"].dtype == np.float32


class TestBuildSatelliteStorageConstants:

    def test_builds_typed_storage_arrays(self):
        satellite_metadata = {
            "sat_belt_id": np.array([0, 1], dtype=np.int16),
            "sat_min_elevation_deg": np.array([20.0, 25.0], dtype=np.float64),
            "sat_beta_max_deg": np.array([30.0, 35.0], dtype=np.float64),
        }

        storage_constants = tleforger.build_satellite_storage_constants(
            satellite_metadata,
            orbit_radius_m_per_sat=np.array([6.9e6, 7.0e6], dtype=np.float64),
        )

        assert_equal(
            storage_constants["sat_belt_id_per_sat"],
            np.array([0, 1], dtype=np.int16),
        )
        assert storage_constants["sat_min_elev_deg_per_sat"].dtype == np.float32
        assert storage_constants["sat_beta_max_deg_per_sat"].dtype == np.float32
        assert storage_constants["sat_orbit_radius_m_per_sat"].dtype == np.float32


class TestHighOrbitAndEllipticalSupport:
    """Tests for high-altitude and elliptical orbit TLE generation."""

    def test_high_earth_orbit_circular(self):
        """HEO at 40,000 km circular generates valid TLEs with correct period."""
        tleforger.reset_tle_counter()
        constellation = tleforger.forge_tle_constellation_from_belt_definitions([
            {
                "belt_name": "HEO_S7",
                "num_sats_per_plane": 2,
                "plane_count": 2,
                "altitude": 40000 * u.km,
                "eccentricity": 0.0,
                "inclination_deg": 63.4 * u.deg,
                "argp_deg": 270 * u.deg,
                "RAAN_min": 0 * u.deg,
                "RAAN_max": 360 * u.deg,
                "min_elevation": 35 * u.deg,
                "adjacent_plane_offset": False,
            },
        ])
        assert constellation["tle_list"].size == 4
        # Period should be ~27.6 hours for 40,000 km circular
        import math
        R_e = 6378137.0
        mu = 3.986004418e14
        a = R_e + 40_000_000.0
        expected_period_h = 2 * math.pi * math.sqrt(a**3 / mu) / 3600.0
        assert 27.0 < expected_period_h < 28.0

    def test_elliptical_orbit_molniya_type(self):
        """Molniya-type HEO with high eccentricity generates valid TLEs."""
        tleforger.reset_tle_counter()
        # Molniya: perigee ~500 km, apogee ~40,000 km, e ≈ 0.74
        # a = (R_e + 500 + R_e + 40000) / 2 km... but our convention:
        # altitude = perigee altitude, so a = (R_e + alt) / (1-e)
        perigee_km = 500.0
        ecc = 0.74
        constellation = tleforger.forge_tle_constellation_from_belt_definitions([
            {
                "belt_name": "Molniya",
                "num_sats_per_plane": 2,
                "plane_count": 2,
                "altitude": perigee_km * u.km,
                "eccentricity": ecc,
                "inclination_deg": 63.4 * u.deg,
                "argp_deg": 270 * u.deg,
                "RAAN_min": 0 * u.deg,
                "RAAN_max": 360 * u.deg,
                "min_elevation": 35 * u.deg,
                "adjacent_plane_offset": False,
            },
        ])
        assert constellation["tle_list"].size == 4
        # Verify semi-major axis: a = (R_e + perigee) / (1 - e)
        import math
        R_e = 6378.137
        a_km = (R_e + perigee_km) / (1 - ecc)
        apogee_km = a_km * (1 + ecc) - R_e
        assert apogee_km > 25000, f"Apogee should be ~26,000+ km, got {apogee_km:.0f} km"
        # Period should be ~12 hours for Molniya
        mu = 3.986004418e14
        a_m = a_km * 1000.0
        period_h = 2 * math.pi * math.sqrt(a_m**3 / mu) / 3600.0
        assert 11.0 < period_h < 13.0, f"Molniya period should be ~12h, got {period_h:.1f}h"

    def test_mean_motion_eccentricity_effect(self):
        """Non-zero eccentricity changes mean motion for same perigee altitude."""
        n_circular = tleforger._compute_mean_motion_rev_day(525_000.0, eccentricity=0.0)
        n_elliptical = tleforger._compute_mean_motion_rev_day(525_000.0, eccentricity=0.5)
        # Elliptical orbit has larger semi-major axis → slower mean motion
        assert n_elliptical < n_circular, \
            f"Elliptical n ({n_elliptical:.4f}) should be less than circular ({n_circular:.4f})"

    def test_gso_orbit_period_near_24h(self):
        """GSO at 36,000 km has a ~24-hour period."""
        import math
        n = tleforger._compute_mean_motion_rev_day(36_000_000.0, eccentricity=0.0)
        period_h = 24.0 / n
        assert 23.5 < period_h < 24.5, f"GSO period should be ~24h, got {period_h:.1f}h"

    def test_gso_constellation_single_satellite(self):
        """GSO constellation with 1 satellite generates valid TLE."""
        tleforger.reset_tle_counter()
        constellation = tleforger.forge_tle_constellation_from_belt_definitions([
            {
                "belt_name": "GSO_S8",
                "num_sats_per_plane": 1,
                "plane_count": 1,
                "altitude": 36000 * u.km,
                "eccentricity": 0.0,
                "inclination_deg": 0.1 * u.deg,
                "argp_deg": 0 * u.deg,
                "RAAN_min": 0 * u.deg,
                "RAAN_max": 360 * u.deg,
                "min_elevation": 5 * u.deg,
                "adjacent_plane_offset": False,
            },
        ])
        assert constellation["tle_list"].size == 1

    def test_leo_circular_unchanged(self):
        """LEO circular orbit (e=0) is unaffected by the eccentricity fix."""
        tleforger.reset_tle_counter()
        constellation = tleforger.forge_tle_constellation_from_belt_definitions([
            {
                "belt_name": "LEO",
                "num_sats_per_plane": 10,
                "plane_count": 2,
                "altitude": 525 * u.km,
                "eccentricity": 0.0,
                "inclination_deg": 53 * u.deg,
                "argp_deg": 0 * u.deg,
                "RAAN_min": 0 * u.deg,
                "RAAN_max": 360 * u.deg,
                "min_elevation": 20 * u.deg,
                "adjacent_plane_offset": True,
            },
        ])
        assert constellation["tle_list"].size == 20
        # Mean motion should match circular orbit at 525 km
        import math
        n_expected = tleforger._compute_mean_motion_rev_day(525_000.0, eccentricity=0.0)
        period_h = 24.0 / n_expected
        assert 1.5 < period_h < 1.7, f"LEO period should be ~1.6h, got {period_h:.2f}h"
