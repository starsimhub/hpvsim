"""Integration tests for hpvsim.sim.Sim."""

import starsim as ss

import hpvsim
from hpvsim.sim import Sim
from hpvsim.hpv import HPV
from hpvsim.network import SexualNetwork


def test_sim_constructs_with_defaults():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.5)
    assert sim is not None


def test_sim_init_runs():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.5)
    sim.init()
    assert len(sim.people) == 500


def test_sim_has_two_sexual_network_layers():
    """Default config produces two SexualNetwork instances (m and c).

    v2's default network has only m and c layers; M01 matches that.
    """
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.5)
    sim.init()
    sx_layers = [n for n in sim.networks() if isinstance(n, SexualNetwork)]
    assert len(sx_layers) == 2
    assert {n.layer for n in sx_layers} == {'m', 'c'}


def test_sim_has_one_hpv_disease():
    sim = Sim(location='nigeria', genotype='hpv16', n_agents=500,
              start=2000, stop=2002, dt=0.5)
    sim.init()
    hpv_diseases = [d for d in sim.diseases() if isinstance(d, HPV)]
    assert len(hpv_diseases) == 1
    assert hpv_diseases[0].genotype == 'hpv16'


def test_sim_runs_short_window():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2003, dt=0.5)
    sim.run()
    assert sim.results.hpv16.n_infected[-1] >= 0


def test_sim_pop_scale_computed_from_total_pop():
    """If total_pop is set, pop_scale = total_pop / n_agents."""
    import hpvsim as hpv
    sim = hpv.Sim(n_agents=10_000, total_pop=2_000_000,
                  start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    assert sim.pars.pop_scale == 200.0


def test_sim_pop_scale_default_one_when_total_pop_none():
    """When total_pop is None, pop_scale defaults to 1.0."""
    import hpvsim as hpv
    sim = hpv.Sim(n_agents=1000, start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    assert sim.pars.pop_scale == 1.0