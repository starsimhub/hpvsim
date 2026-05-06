"""Unit tests for the cross-immunity Connector and matrix-builder."""
import numpy as np
import pytest
import starsim as ss

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
    """CrossImmunity.init_pre populates hpv_modules from sim.diseases in registration order."""
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    conn = sim.connectors.crossimmunity   # Starsim auto-snake-cases class name
    assert len(conn.hpv_modules) == 1
    assert conn.hpv_modules[0].genotype == 'hpv16'
    assert conn.genotype_index == {'hpv16': 0}


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


def test_cross_immunity_step_identity_for_single_genotype():
    """1x1 identity matrix: rel_sus = 1 - nab_imm, sev_imm = cell_imm."""
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=10, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    mod = sim.diseases.hpv16
    conn = sim.connectors.crossimmunity
    # Manually set source immunity for half the agents, then step the connector.
    mod.nab_imm.values[:5] = 0.4
    mod.cell_imm.values[:5] = 0.3
    conn.step()
    # Identity multiply: rel_sus = 1 - 0.4 = 0.6 for first five, 1.0 elsewhere.
    rel = np.asarray(mod.rel_sus.values)
    sev = np.asarray(mod.sev_imm.values)
    assert np.allclose(rel[:5], 0.6)
    assert np.allclose(rel[5:], 1.0)
    assert np.allclose(sev[:5], 0.3)
    assert np.allclose(sev[5:], 0.0)


def test_cross_immunity_step_two_genotype_hand_computed():
    """2-genotype case: rel_sus and sev_imm match hand-computed dot product."""
    from hpvsim.connectors import CrossImmunity
    # Cross-immunity: 16->18 = 0.5 sus / 0.7 sev; 18->16 = 0.5 sus / 0.7 sev.
    m_sus = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    m_sev = np.array([[1.0, 0.7], [0.7, 1.0]], dtype=np.float32)
    sim = hpv.Sim(
        n_agents=4, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        connectors=[CrossImmunity(cross_imm_sus=m_sus, cross_imm_sev=m_sev)],
    )
    sim.init()
    h16 = sim.diseases.hpv16
    h18 = sim.diseases.hpv18
    conn = sim.connectors.crossimmunity
    # Agent 0: had hpv16 (nab=0.4, cell=0.3); no hpv18 history.
    h16.nab_imm.values[0] = 0.4
    h16.cell_imm.values[0] = 0.3
    # Agent 1: had hpv18 (nab=0.6, cell=0.5); no hpv16 history.
    h18.nab_imm.values[1] = 0.6
    h18.cell_imm.values[1] = 0.5
    conn.step()
    # Target hpv16, agent 0: sus_imm = 1.0*0.4 + 0.5*0 = 0.4 -> rel_sus = 0.6
    # Target hpv18, agent 0: sus_imm = 0.5*0.4 + 1.0*0 = 0.2 -> rel_sus = 0.8
    # Target hpv16, agent 1: sus_imm = 1.0*0 + 0.5*0.6 = 0.3 -> rel_sus = 0.7
    # Target hpv18, agent 1: sus_imm = 0.5*0 + 1.0*0.6 = 0.6 -> rel_sus = 0.4
    assert h16.rel_sus.values[0] == pytest.approx(0.6, abs=1e-6)
    assert h18.rel_sus.values[0] == pytest.approx(0.8, abs=1e-6)
    assert h16.rel_sus.values[1] == pytest.approx(0.7, abs=1e-6)
    assert h18.rel_sus.values[1] == pytest.approx(0.4, abs=1e-6)
    # sev_imm: target hpv16, agent 0 = 1.0*0.3 + 0.7*0 = 0.3
    #          target hpv18, agent 1 = 0.7*0 + 1.0*0.5 = 0.5
    assert h16.sev_imm.values[0] == pytest.approx(0.3, abs=1e-6)
    assert h18.sev_imm.values[1] == pytest.approx(0.5, abs=1e-6)


def test_cross_immunity_step_clips_to_unit_interval():
    """sus_imm and sev_imm are clipped to [0, 1] (matches v2 np.minimum cap)."""
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=4, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    sim.diseases.hpv16.nab_imm.values[0] = 0.9
    sim.diseases.hpv18.nab_imm.values[0] = 0.9
    sim.connectors.crossimmunity.step()
    # Agent 0: sus_imm to hpv16 = 1*0.9 + 0.5*0.9 = 1.35 -> clipped to 1.0
    # rel_sus = 1 - 1.0 = 0.0
    assert sim.diseases.hpv16.rel_sus.values[0] == pytest.approx(0.0, abs=1e-6)


def test_cross_immunity_step_writes_survive_dead_agents():
    """Connector writes must use Arr __setitem__ via auids, not .values[:].

    With dead agents present (raw.size != auids.size), FloatArr.values returns
    a copy; writing to .values[:] would be silently discarded. Verify writes
    actually land in raw.
    """
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=20, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    mod = sim.diseases.hpv16
    conn = sim.connectors.crossimmunity
    # Kill agents 0 and 1 to force raw.size != auids.size.
    uids_to_kill = ss.uids([0, 1])
    sim.people.ti_dead[uids_to_kill] = -1   # mark as died before current step
    sim.people.alive[uids_to_kill] = False
    sim.people.remove_dead()                 # refresh active-uid arrays
    # Confirm the dead-agent divergence is in place.
    assert mod.rel_sus.raw.size != mod.rel_sus.auids.size
    # Set source immunity for some live agent.
    live_auids = mod.rel_sus.auids
    target_auid = int(live_auids[0])
    mod.nab_imm[ss.uids([target_auid])] = 0.4
    mod.cell_imm[ss.uids([target_auid])] = 0.3
    conn.step()
    # Check the write landed in raw (not just an ephemeral copy).
    assert float(mod.rel_sus.raw[target_auid]) == pytest.approx(0.6, abs=1e-6)
    assert float(mod.sev_imm.raw[target_auid]) == pytest.approx(0.3, abs=1e-6)
