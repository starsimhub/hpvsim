"""Unit tests for hpv.txvx — therapeutic vaccine product."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv
from hpvsim.products import txvx as hpv_txvx


def _four_genotype_sim():
    return hpv.Sim(
        n_agents=200, start=2020, stop=2021, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
    )


def _attach_txvx_and_init(sim, p_instance):
    """Attach the txvx product to a stub treat_num and init.

    Returns the LIVE post-init copy (Sim deep-copies inputs).
    """
    sim.pars['interventions'] = [ss.treat_num(product=p_instance, prob=0.0)]
    sim.init()
    return sim.interventions[0].product


def test_txvx_unknown_name_raises():
    with pytest.raises(ValueError, match='Unknown txvx product name'):
        hpv_txvx(name='nope')


def test_txvx_both_name_and_rel_imm_raises():
    with pytest.raises(ValueError, match='exactly one'):
        hpv_txvx(name='txvx1', rel_imm={'hpv16': 1.0})


def test_txvx1_first_dose_bumps_txvx_imm_on_active_genotypes():
    sim = _four_genotype_sim()
    p = _attach_txvx_and_init(sim, hpv_txvx(name='txvx1'))
    uids = sim.people.alive.uids[:10]
    p.administer(None, uids)
    # All four active genotypes should have at least one agent with txvx_imm > 0
    for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        vals = sim.diseases[g].txvx_imm[uids]
        assert np.all(vals >= 0), f'{g}: txvx_imm must be non-negative'
        assert np.any(vals > 0), f'{g}: at least one agent should have txvx_imm > 0'


def test_txvx_does_not_downgrade():
    """A leaky-draw txvx must not lower an agent's prior txvx_imm."""
    sim = _four_genotype_sim()
    p = _attach_txvx_and_init(sim, hpv_txvx(name='txvx1', sterilizing_p=0.0))
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].txvx_imm[uids] = 0.9
    p.administer(None, uids)
    assert np.all(sim.diseases['hpv16'].txvx_imm[uids] >= 0.9)


def test_txvx2_booster_multiplies_existing():
    sim = _four_genotype_sim()
    booster = _attach_txvx_and_init(sim, hpv_txvx(name='txvx2', imm_boost=1.2))
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].txvx_imm[uids] = 0.5
    booster.administer(None, uids)
    np.testing.assert_allclose(sim.diseases['hpv16'].txvx_imm[uids], 0.6, atol=1e-6)


def test_txvx_inactive_genotype_tolerance():
    """A txvx product targeting an inactive (not-in-sim) genotype skips silently."""
    sim = _four_genotype_sim()
    p = _attach_txvx_and_init(
        sim,
        hpv_txvx(rel_imm={'hpv16': 0.9, 'no_such_genotype': 0.5}),
    )
    uids = sim.people.alive.uids[:3]
    p.administer(None, uids)  # Must not error
    assert np.all(sim.diseases['hpv16'].txvx_imm[uids] > 0)


def test_txvx_empty_uids_noop():
    sim = _four_genotype_sim()
    p = _attach_txvx_and_init(sim, hpv_txvx(name='txvx1'))
    p.administer(None, ss.uids())  # Must not error
