"""Integration tests for hpvsim.sim.Sim."""

import warnings

from hpvsim.sim import Sim
from hpvsim.hpv import HPV
from hpvsim.network import SexualNetwork


def test_sim_constructs_with_defaults():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.25)
    assert sim is not None


def test_sim_init_runs():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.25)
    sim.init()
    assert len(sim.people) == 500


def test_sim_has_one_multilayer_sexual_network():
    """Default config produces a single hpv.SexualNetwork holding both
    partnership layers (m, c).
    """
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.25)
    sim.init()
    sx = [n for n in sim.networks() if isinstance(n, SexualNetwork)]
    assert len(sx) == 1
    assert set(sx[0].layers) == {'m', 'c'}


def test_sim_has_one_hpv_disease():
    sim = Sim(location='nigeria', genotypes=['hpv16'], n_agents=500,
              start=2000, stop=2002, dt=0.25)
    sim.init()
    hpv_diseases = [d for d in sim.diseases() if isinstance(d, HPV)]
    assert len(hpv_diseases) == 1
    assert hpv_diseases[0].genotype == 'hpv16'


def test_sim_runs_short_window():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2003, dt=0.25)
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


def test_end_is_legacy_alias_for_stop():
    """v2 scripts pass ``end=``; it should set ``stop`` and warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim_end = Sim(n_agents=200, start=2000, end=2005, dt=1.0)
    sim_stop = Sim(n_agents=200, start=2000, stop=2005, dt=1.0)
    assert sim_end.pars.stop == sim_stop.pars.stop
    assert any('deprecated alias' in str(w.message) for w in caught), \
        'passing end= should emit a deprecation warning'


def test_stop_without_end_does_not_warn():
    """Using the canonical ``stop=`` must not emit the legacy-alias warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Sim(n_agents=200, start=2000, stop=2005, dt=1.0)
    assert not any('deprecated alias' in str(w.message) for w in caught)