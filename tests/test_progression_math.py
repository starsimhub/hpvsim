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