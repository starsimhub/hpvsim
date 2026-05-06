"""Unit tests for the cross-immunity Connector and matrix-builder."""
import numpy as np
import pytest

import hpvsim as hpv
from hpvsim.parameters import get_cross_immunity, GENOTYPE_KEYS


def test_genotype_keys_are_canonical_four():
    """GENOTYPE_KEYS pins the M03 4-genotype default ordering."""
    assert GENOTYPE_KEYS == ('hpv16', 'hpv18', 'hi5', 'ohr')


def test_get_cross_immunity_default_shape_and_diagonal():
    """Default cross-immunity matrices are (4, 4) float32 with diagonal == 1.0."""
    m_sus, m_sev = get_cross_immunity()
    for m in (m_sus, m_sev):
        assert m.shape == (4, 4)
        assert m.dtype == np.float32
        assert np.allclose(np.diag(m), 1.0)


def test_get_cross_immunity_default_values():
    """Defaults match v2 scalars: cross_imm_sus_med=0.3, cross_imm_sus_high=0.5,
    cross_imm_sev_med=0.5, cross_imm_sev_high=0.7."""
    m_sus, m_sev = get_cross_immunity()
    keys = ('hpv16', 'hpv18', 'hi5', 'ohr')
    idx = {k: i for i, k in enumerate(keys)}
    # Off-diagonal hpv16<->hpv18 = high (0.5 sus, 0.7 sev); both directions.
    assert m_sus[idx['hpv16'], idx['hpv18']] == pytest.approx(0.5)
    assert m_sus[idx['hpv18'], idx['hpv16']] == pytest.approx(0.5)
    assert m_sev[idx['hpv16'], idx['hpv18']] == pytest.approx(0.7)
    # hpv16 -> hi5 = med
    assert m_sus[idx['hpv16'], idx['hi5']] == pytest.approx(0.3)
    assert m_sev[idx['hpv16'], idx['hi5']] == pytest.approx(0.5)
    # Diagonal forced to 1.0 by convention.
    assert m_sus[idx['hi5'], idx['hi5']] == pytest.approx(1.0)


def test_get_cross_immunity_custom_keys():
    """Caller-supplied genotype ordering controls matrix layout."""
    m_sus, _ = get_cross_immunity(keys=('hi5', 'hpv16'))
    assert m_sus.shape == (2, 2)
    # m_sus[1, 0] is "from hi5 source to hpv16 target" — medium scalar.
    assert m_sus[1, 0] == pytest.approx(0.3)


def test_genotype_pars_imm_init_is_distribution():
    """GenotypePars.imm_init becomes a beta-mean distribution (M03 conversion)."""
    gp = hpv.get_genotype_pars('hpv16')
    assert hasattr(gp.imm_init, 'rvs'), \
        f'imm_init should be a Dist, got {type(gp.imm_init)}'