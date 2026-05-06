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


def test_cross_immunity_connector_collects_hpv_modules():
    """CrossImmunity.init_pre populates _hpv_modules from sim.diseases in registration order."""
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    conn = sim.connectors.crossimmunity   # Starsim auto-snake-cases class name
    assert len(conn._hpv_modules) == 1
    assert conn._hpv_modules[0].genotype == 'hpv16'
    assert conn._genotype_index == {'hpv16': 0}


def test_cross_immunity_connector_default_matrices():
    """If matrices not supplied, init_pre populates from get_cross_immunity for the discovered genotype set."""
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    conn = sim.connectors.crossimmunity   # Starsim copies; check the live instance
    assert conn.cross_imm_sus.shape == (1, 1)
    assert conn.cross_imm_sus.dtype == np.float32
    assert conn.cross_imm_sus[0, 0] == pytest.approx(1.0)


def test_cross_immunity_connector_rejects_off_diagonal_self_immunity():
    """Diagonal entries must be 1.0; init_pre raises otherwise."""
    from hpvsim.connectors import CrossImmunity
    bad = np.array([[0.5]], dtype=np.float32)
    conn = CrossImmunity(cross_imm_sus=bad, cross_imm_sev=bad)
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[conn],
    )
    with pytest.raises(ValueError, match='diagonal'):
        sim.init()


def test_cross_immunity_connector_rejects_shape_mismatch():
    """Matrix dim must match number of HPV modules."""
    from hpvsim.connectors import CrossImmunity
    bad = np.eye(2, dtype=np.float32)
    conn = CrossImmunity(cross_imm_sus=bad, cross_imm_sev=bad)
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[conn],
    )
    with pytest.raises(ValueError, match='shape'):
        sim.init()
