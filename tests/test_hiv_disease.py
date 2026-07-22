import numpy as np
import hpvsim as hpv
from hpvsim.network import SexualNetwork


def test_hiv_beta_directional_orientation():
    """beta is keyed to the SexualNetwork name as [f2m, m2f] (p1=female, p2=male)."""
    h = hpv.HIV(beta_m2f=0.0035, rel_beta_f2m=0.5)
    sim = hpv.Sim(n_agents=200, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h])
    sim.init()
    net = [n for n in sim.networks.values() if isinstance(n, SexualNetwork)][0]
    beta = dict(sim.diseases.hiv.pars.beta)
    assert net.name in beta
    f2m, m2f = beta[net.name]
    # p1=female so betamap[0]=f2m, betamap[1]=m2f; m2f is the larger direction.
    assert np.isclose(m2f, 0.0035)
    assert np.isclose(f2m, 0.0035 * 0.5)
