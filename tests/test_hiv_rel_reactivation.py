import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector


def _sim_with_hiv():
    sim = hpv.Sim(n_agents=300, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16], diseases=[hpv.HIV(beta_m2f=0.0)])
    sim.init()
    return sim


def test_hiv_rel_reactivation_default_state():
    """hiv_rel_reactivation defaults to 1.0 when no HIV is seeded (gated no-op)."""
    sim = _sim_with_hiv()
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    uids = sim.people.auids[:5]
    assert np.allclose(conn.hiv_rel_reactivation[uids], 1.0)


def test_hpv_hiv_connector_rel_reactivation_override():
    """rel_reactivation_lo/hi can be overridden independently of the other
    three effects (no more effects= dict requiring every key to be restated),
    and the override actually flows through to the per-agent factor."""
    sim = hpv.Sim(n_agents=300, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16], diseases=[hpv.HIV(beta_m2f=0.0)],
                  connectors=[hpv_hiv_connector(rel_reactivation_lo=5.0, rel_reactivation_hi=7.0)])
    sim.init()
    hivmod = sim.diseases.hiv
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    uid_lo, uid_hi = sim.people.auids[0], sim.people.auids[1]
    hivmod.infected[uid_lo] = True
    hivmod.cd4[uid_lo] = 100.0  # lo stratum
    hivmod.infected[uid_hi] = True
    hivmod.cd4[uid_hi] = 350.0  # hi stratum
    conn.step()
    assert np.isclose(conn.hiv_rel_reactivation[uid_lo], 5.0)
    assert np.isclose(conn.hiv_rel_reactivation[uid_hi], 7.0)
    # Untouched effects stay at class defaults.
    assert np.isclose(conn.pars.rel_sus_lo, 2.2)
