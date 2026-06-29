import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import hpvsim as hpv
import starsim as ss
import numpy as np

def test_ratio_param_and_fine_state_exist():
    assert hpv.__file__.startswith(WT), f'wrong hpvsim loaded: {hpv.__file__}'
    sim = hpv.Sim(location='nigeria', n_agents=500, start=2000, stop=2002,
                  ms_agent_ratio=10)
    sim.init()
    # fine state exists on People, defaults all-False
    assert 'fine' in sim.people.states
    assert not sim.people.fine.values.any()
    # ratio propagated to every genotype HPV module
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            assert int(dis.pars.ms_agent_ratio) == 10

def test_results_are_float_and_scale_weighted_noop_at_ratio1():
    # ratio==1: scale_flows == len, so counts are unchanged but dtype is float.
    sim = hpv.Sim(location='nigeria', n_agents=2000, start=1990, stop=2020,
                  ms_agent_ratio=1, rand_seed=1)
    sim.run()
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            r = dis.results
            assert r.new_cancers.values.dtype == np.float64
            assert r.new_cancer_deaths.values.dtype == np.float64
            # ratio==1 keeps everyone at scale 1.0 → integer-valued counts
            nc = r.new_cancers.values
            assert np.allclose(nc, np.round(nc))
    assert not sim.people.fine.values.any()
    assert np.allclose(sim.people.scale.values, 1.0)
