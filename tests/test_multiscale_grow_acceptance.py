# tests/test_multiscale_grow_acceptance.py
import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import numpy as np
import hpvsim as hpv

def _total_cancers(ratio, seed, n_agents=8000):
    sim = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2030,
                  ms_agent_ratio=ratio, rand_seed=seed)
    sim.run()
    tot = 0.0
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            tot += float(dis.results.new_cancers.values.sum())
    return tot

def test_cancer_incidence_flat_across_ratio():
    seeds = range(8)
    base = np.mean([_total_cancers(1, s) for s in seeds])
    for ratio in (5, 10):
        got = np.mean([_total_cancers(ratio, s) for s in seeds])
        rel = got / base
        assert 0.90 <= rel <= 1.10, f'ratio={ratio}: {rel:.3f} (base={base:.0f})'
