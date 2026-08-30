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


def test_hpv_hiv_connector_requires_rel_reactivation_in_custom_effects():
    """A caller-supplied effects dict missing rel_reactivation must raise."""
    import pytest
    incomplete = dict(
        rel_sus={'lt200': 2.0, 'gt200': 2.0},
        rel_sev={'lt200': 1.5, 'gt200': 1.2},
        rel_imm={'lt200': 0.4, 'gt200': 0.8},
        # rel_reactivation deliberately omitted
    )
    with pytest.raises(ValueError, match='rel_reactivation'):
        hpv_hiv_connector(effects=incomplete)
