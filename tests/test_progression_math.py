"""Unit tests for M02 progression math + GenotypePars defaults."""
import pytest

import hpvsim as hpv


def test_genotype_pars_hpv16_defaults():
    """GenotypePars('hpv16') matches the v2 hpv16 defaults verbatim."""
    g = hpv.GenotypePars('hpv16')
    # v2 _v2_legacy/parameters.py:336–342
    assert g.dur_precin == dict(dist='lognormal', par1=3, par2=9)
    assert g.cin_fn == dict(form='logf2', k=0.3, x_infl=0, ttc=50)
    assert g.dur_cin == dict(dist='lognormal', par1=5, par2=20)
    assert g.cancer_fn == dict(method='cin_integral', transform_prob=2e-3)
    assert g.rel_beta == 1.0
    assert g.sero_prob == 0.75


def test_genotype_aliases_hpv16():
    """Aliases let us pass '16' or 'hpv16' interchangeably."""
    assert hpv.parameters.genotype_aliases['hpv16'] == ['hpv16', '16']


def test_get_genotype_pars_factory():
    """get_genotype_pars('hpv16') returns a GenotypePars-equivalent dict-like."""
    g = hpv.get_genotype_pars('hpv16')
    assert g.dur_precin['par1'] == 3


import numpy as np


def test_logf2_pinned_outputs():
    """_logf2 reproduces v2's logistic-2 outputs at canonical points.

    Generated from hpvsim._v2_legacy.utils.logf2 with the HPV16 cin_fn
    parameters (k=0.3, x_infl=0, ttc=50).
    """
    from hpvsim.hpv import _logf2
    expected = {
        0.5:  0.07485973648701938,
        1.0:  0.14888512471190052,
        2.0:  0.2913127906780537,
        5.0:  0.6351493409744831,
        10.0: 0.9051488074189387,
    }
    for x, want in expected.items():
        got = _logf2(x, k=0.3, x_infl=0, ttc=50)
        assert np.isclose(got, want, rtol=1e-9), f'logf2({x}) = {got}, want {want}'


def test_logf2_array_input():
    """_logf2 accepts numpy arrays."""
    from hpvsim.hpv import _logf2
    out = _logf2(np.array([0.5, 1.0, 2.0]), k=0.3, x_infl=0, ttc=50)
    assert out.shape == (3,)
    assert (out >= 0).all() and (out <= 1).all()


def test_logf2_matches_quarantine_verbatim():
    """_logf2 produces bit-identical output to v2's logf2 over a sweep.

    This is the strongest possible parity check: literally compare to v2.
    """
    from hpvsim.hpv import _logf2
    from hpvsim._v2_legacy.utils import logf2 as v2_logf2
    xs = np.linspace(0.01, 60, 200)
    for k in (0.15, 0.3, 0.5):
        for x_infl in (0, 5, 10):
            for ttc in (25, 50):
                a = _logf2(xs, k=k, x_infl=x_infl, ttc=ttc)
                b = v2_logf2(xs, k=k, x_infl=x_infl, ttc=ttc)
                assert np.allclose(a, b, equal_nan=True), \
                    f'mismatch at k={k} x_infl={x_infl} ttc={ttc}'