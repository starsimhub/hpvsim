import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import hpvsim as hpv
import starsim as ss

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
