import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector


def _sim_with_hiv():
    sim = hpv.Sim(n_agents=300, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16], diseases=[hpv.HIV(beta_m2f=0.0)])
    sim.init()
    return sim


def test_hiv_connector_lookup():
    """HPV._hiv_connector() returns the registered hpv_hiv_connector."""
    sim = _sim_with_hiv()
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    assert hpvmod._hiv_connector() is conn


def test_hiv_rel_sev_default_state():
    """hiv_rel_sev defaults to 1.0 when no HIV is seeded (gated no-op input)."""
    sim = _sim_with_hiv()
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    uids = sim.people.auids[:5]
    assert np.allclose(conn.hiv_rel_sev[uids], 1.0)


def test_hiv_connector_absent_in_hpv_only_sim():
    """HPV._hiv_connector() returns None when no HIV/connector is present (gating path)."""
    sim = hpv.Sim(n_agents=300, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16])
    sim.init()
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    assert hpvmod._hiv_connector() is None
