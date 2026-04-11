#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u

from scepter import antenna


def test_lazy_numba_loader_only_imports_after_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(antenna, "_NUMBA_IMPORT_ATTEMPTED", False)
    monkeypatch.setattr(antenna, "_NUMBA_NJIT", None)
    monkeypatch.setattr(antenna, "_HAVE_NUMBA", False)

    calls: list[str] = []

    def _import_module(name: str, package: str | None = None):
        del package
        calls.append(str(name))
        assert name == "numba"

        class _FakeNumbaModule:
            @staticmethod
            def njit(*args: object, **kwargs: object):
                del args, kwargs

                def _decorate(func):
                    return func

                return _decorate

        return _FakeNumbaModule()

    monkeypatch.setattr(antenna.importlib, "import_module", _import_module)

    default_gain = antenna.s_1528_rec1_2_pattern(np.asarray([0.2, 0.4]) * u.deg)[0]
    assert calls == []
    assert default_gain.shape == (2,)

    accelerated_gain = antenna.s_1528_rec1_2_pattern(
        np.asarray([0.2, 0.4]) * u.deg,
        use_numba=True,
    )[0]
    assert calls == ["numba"]
    assert antenna._HAVE_NUMBA is True
    assert accelerated_gain.shape == (2,)

    cached_gain = antenna.s_1528_rec1_2_pattern(np.asarray([0.2, 0.4]) * u.deg)[0]
    assert calls == ["numba"]
    assert cached_gain.shape == (2,)


def test_s672_routes_to_rec12_pattern():
    """S.672 antenna model routes to the same Rec 1.2 piecewise function."""
    from pycraf import conversions as cnv
    func, wavelength, kwargs = antenna.build_satellite_pattern_spec(
        antenna_model="s672",
        frequency_mhz=2690.0,
        pattern_wavelength_cm=15.0,
        derive_pattern_wavelength_from_frequency=False,
        rec12_gm_dbi=40.0, rec12_ln_db=-20.0, rec12_z=1.0,
    )
    assert func is antenna.s_1528_rec1_2_pattern
    # Verify it produces a valid gain at boresight
    result = func(np.array([0.0]) * u.deg, **kwargs)
    gain_arr = result[0] if isinstance(result, (tuple, list)) else result
    g0 = float(np.asarray(gain_arr).flat[0])
    assert abs(g0 - 40.0) < 0.5, f"S.672 peak gain should be ~40 dBi, got {g0:.1f}"


def test_build_satellite_pattern_spec_all_models():
    """All antenna models produce valid pattern specs without errors."""
    from pycraf import conversions as cnv
    common = dict(
        frequency_mhz=2690.0, pattern_wavelength_cm=15.0,
        derive_pattern_wavelength_from_frequency=False,
        rec12_gm_dbi=38.0, rec12_ln_db=-20.0, rec12_z=1.0,
    )
    for model in ("s1528_rec1_2", "s672"):
        func, wl, kw = antenna.build_satellite_pattern_spec(antenna_model=model, **common)
        assert callable(func), f"Model {model} did not return callable"

    # Rec 1.4
    func14, _, _ = antenna.build_satellite_pattern_spec(
        antenna_model="s1528_rec1_4", **common,
        rec14_lt_m=1.6, rec14_lr_m=1.6, rec14_l=2, rec14_slr_db=20.0,
        rec14_far_sidelobe_start_deg=90.0, rec14_far_sidelobe_level_dbi=-20.0,
    )
    assert callable(func14)

    # M.2101
    func_m, _, _ = antenna.build_satellite_pattern_spec(
        antenna_model="m2101", **common,
        m2101_g_emax_dbi=2.0, m2101_a_m_db=30.0, m2101_sla_nu_db=30.0,
        m2101_phi_3db_deg=120.0, m2101_theta_3db_deg=120.0,
        m2101_d_h=0.5, m2101_d_v=0.5, m2101_n_h=28, m2101_n_v=28,
    )
    assert callable(func_m)
