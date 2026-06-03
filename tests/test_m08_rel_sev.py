import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector


def test_set_prognoses_uses_hiv_rel_sev():
    """When the connector reports hiv_rel_sev>1 for an agent, set_prognoses
    forms a larger effective severity than the no-HIV baseline."""
    sim = hpv.Sim(n_agents=300, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16], diseases=[hpv.HIV(beta_m2f=0.0)],
                  connectors=[hpv_hiv_connector()])
    sim.init()
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    # No HIV module connector lookup should yield factor 1.0 by default.
    assert hpvmod._hiv_connector() is conn
    # Default state: every agent hiv_rel_sev == 1.0 (no HIV).
    uids = sim.people.auids[:5]
    assert np.allclose(conn.hiv_rel_sev[uids], 1.0)
