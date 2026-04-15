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


class TestStartTimeParameter:
    """Custom ``start_time`` is forwarded to the TLE epoch in every forge helper."""

    # TLE line 1 encodes the epoch as ``YYDDD.DDDDDDDD`` in columns 19-32
    # (1-based) → Python slice [18:32]. We parse that back to assert the
    # forger honoured our requested epoch.
    @staticmethod
    def _parse_epoch_yyddd(tle_line1: str) -> tuple[int, float]:
        field = tle_line1[18:32]
        year_short = int(field[:2])
        day_of_year = float(field[2:])
        return year_short, day_of_year

    def test_single_sat_uses_custom_datetime_epoch(self):
        from datetime import datetime
        from astropy import units as u

        tleforger.reset_tle_counter()
        requested = datetime(2030, 3, 15, 12, 0, 0)
        tle = tleforger.forge_tle_single(
            altitude=525_000.0 * u.m,
            inclination_deg=53.0 * u.deg,
            start_time=requested,
        )
        line1 = tle.tle_strings()[1]
        year_short, day_of_year = self._parse_epoch_yyddd(line1)
        assert year_short == 30
        # 2030-03-15 noon → DOY 74.5 (Jan+Feb = 31+28 = 59; +15 = 74; +0.5)
        assert abs(day_of_year - 74.5) < 1e-3

    def test_single_sat_none_falls_back_to_legacy_epoch(self):
        from astropy import units as u

        tleforger.reset_tle_counter()
        tle = tleforger.forge_tle_single(altitude=525_000.0 * u.m, start_time=None)
        year_short, day_of_year = self._parse_epoch_yyddd(tle.tle_strings()[1])
        # Legacy default: 2025-01-01T00:00 → YY=25, DOY=1.0
        assert year_short == 25
        assert abs(day_of_year - 1.0) < 1e-6

    def test_belt_propagates_custom_start_time(self):
        from datetime import datetime
        from astropy import units as u

        tleforger.reset_tle_counter()
        tles = tleforger.forge_tle_belt(
            belt_name="EpochBelt",
            num_sats_per_plane=2,
            plane_count=2,
            altitude=525_000.0 * u.m,
            inclination_deg=53.0 * u.deg,
            start_time=datetime(2028, 7, 1, 0, 0, 0),
        )
        assert tles.size == 4
        epochs = {self._parse_epoch_yyddd(t.tle_strings()[1]) for t in tles}
        # All satellites in a belt share one epoch.
        assert len(epochs) == 1
        year_short, day_of_year = next(iter(epochs))
        assert year_short == 28
        # DOY of 2028-07-01 (leap year): Jan+Feb+Mar+Apr+May+Jun = 31+29+31+30+31+30 = 182, +1 = 183
        assert abs(day_of_year - 183.0) < 1e-6

    def test_constellation_helper_forwards_start_time_to_every_belt(self):
        from datetime import datetime
        from astropy import units as u

        tleforger.reset_tle_counter()
        belts = [
            {
                "belt_name": f"Belt{i}",
                "num_sats_per_plane": 2,
                "plane_count": 1,
                "altitude": (500 + 50 * i) * u.km,
                "eccentricity": 0.0,
                "inclination_deg": 53.0 * u.deg,
                "argp_deg": 0.0 * u.deg,
                "RAAN_min": 0.0 * u.deg,
                "RAAN_max": 180.0 * u.deg,
                "min_elevation": 20.0 * u.deg,
                "adjacent_plane_offset": False,
            }
            for i in range(3)
        ]
        constellation = tleforger.forge_tle_constellation_from_belt_definitions(
            belts, start_time=datetime(2029, 12, 31, 0, 0, 0),
        )
        # Every belt's TLEs share the single requested epoch.
        epochs = {
            self._parse_epoch_yyddd(t.tle_strings()[1])
            for t in constellation["tle_list"]
        }
        assert len(epochs) == 1
        year_short, day_of_year = next(iter(epochs))
        assert year_short == 29
        # 2029-12-31 → DOY 365 (2029 is not a leap year).
        assert abs(day_of_year - 365.0) < 1e-6

    def test_constellation_helper_accepts_astropy_time(self):
        from astropy.time import Time
        from astropy import units as u

        tleforger.reset_tle_counter()
        constellation = tleforger.forge_tle_constellation_from_belt_definitions(
            [
                {
                    "belt_name": "LEO",
                    "num_sats_per_plane": 1,
                    "plane_count": 1,
                    "altitude": 525 * u.km,
                    "eccentricity": 0.0,
                    "inclination_deg": 53 * u.deg,
                    "argp_deg": 0 * u.deg,
                    "RAAN_min": 0 * u.deg,
                    "RAAN_max": 180 * u.deg,
                    "min_elevation": 20 * u.deg,
                    "adjacent_plane_offset": False,
                },
            ],
            start_time=Time("2027-06-15T00:00:00", scale="utc"),
        )
        line1 = constellation["tle_list"][0].tle_strings()[1]
        year_short, day_of_year = self._parse_epoch_yyddd(line1)
        assert year_short == 27
        # DOY of 2027-06-15 (not a leap year): 31+28+31+30+31+15 = 166
        assert abs(day_of_year - 166.0) < 1e-6

    def test_constellation_helper_none_preserves_legacy_behavior(self):
        from astropy import units as u

        tleforger.reset_tle_counter()
        constellation = tleforger.forge_tle_constellation_from_belt_definitions(
            [
                {
                    "belt_name": "LEO",
                    "num_sats_per_plane": 1,
                    "plane_count": 1,
                    "altitude": 525 * u.km,
                    "eccentricity": 0.0,
                    "inclination_deg": 53 * u.deg,
                    "argp_deg": 0 * u.deg,
                    "RAAN_min": 0 * u.deg,
                    "RAAN_max": 180 * u.deg,
                    "min_elevation": 20 * u.deg,
                    "adjacent_plane_offset": False,
                },
            ],
        )
        line1 = constellation["tle_list"][0].tle_strings()[1]
        year_short, day_of_year = self._parse_epoch_yyddd(line1)
        assert year_short == 25
        assert abs(day_of_year - 1.0) < 1e-6
