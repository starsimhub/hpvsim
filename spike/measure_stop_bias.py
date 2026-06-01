"""Throwaway: relative bias of total people-space cancers vs simulation stop year.

Tests the 2060-window-truncation hypothesis. With the by-age fix, rescued
cancers get the agent's own LONGER length-biased dur_cin, so cancers initiated
near the window end onset after `stop` and are truncated -- differentially
affecting the ratio=1 vs ratio=12 arms. Extending `stop` should let the cancer
tail complete and shrink the bias.

Reduced sizes (n=12000 both arms, equal n_agents -> equal people-space in this
build) and few seeds for cheapness.

People-space total cancers per sim: new_cancers.sum() * pop_scale.
"""
import numpy as np
import sciris as sc
import hpvsim as hpv

SEEDS = list(range(6))
N = 12000
STOPS = [2060, 2080, 2100]


def total_cancers_people(sim):
    return float(np.asarray(sim.results.hpv16.new_cancers).sum()) * float(sim.pars.pop_scale)


def run_arm(ratio, stop):
    vals = []
    for sd in SEEDS:
        s = hpv.Sim(location='nigeria', genotypes=['hpv16'], start=1990, stop=stop,
                    dt=0.25, total_pop=1e6, n_agents=N, ms_agent_ratio=ratio,
                    rand_seed=sd, verbose=0)
        s.run()
        vals.append(total_cancers_people(s))
    return np.array(vals)


if __name__ == '__main__':
    T = sc.timer()
    print(f'n_agents={N} both arms, {len(SEEDS)} seeds, stops={STOPS}')
    print(f'{"stop":>6} {"single_mean":>14} {"multi_mean":>14} {"rel.bias":>10}')
    results = {}
    for stop in STOPS:
        single = run_arm(1, stop)
        multi = run_arm(12, stop)
        rel = abs(multi.mean() - single.mean()) / single.mean()
        results[stop] = rel
        print(f'{stop:>6} {single.mean():>14.0f} {multi.mean():>14.0f} {rel*100:>9.2f}%',
              flush=True)
    print('\nbias-vs-stop:', {k: f'{v*100:.2f}%' for k, v in results.items()})
    T.toc('TOTAL runtime')
