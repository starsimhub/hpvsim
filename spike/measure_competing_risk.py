"""Throwaway: is the binomial-multiscale cancer under-count caused by competing
background mortality?

Hypothesis: the by-age fix gives rescued cancers their correct LONGER dur_cin,
pushing ti_cancerous later, so more rescued agents die of BACKGROUND causes
(ss.Deaths / AgeMigration removal) before their cancer fires -> permanently lost
cancers, growing with the window.

Test: compare ratio=1 vs ratio=12 cancer-total bias WITH default demographics
vs WITHOUT any demographics (demographics=[] -> no births/deaths/migration, so
the only agent removal is cancer death itself). If the multiscale under-count
vanishes and stops growing with `stop` when background removal is off, competing
risk is confirmed.

Equal n_agents both arms (total_pop does not drive per-agent scale in this
build, so equal n_agents == equal people-space). Relative bias is the signal.
"""
import numpy as np
import sciris as sc
import hpvsim as hpv

SEEDS = list(range(4))
N = 12000
STOPS = [2060, 2100]


def total_cancers_people(sim):
    return float(np.asarray(sim.results.hpv16.new_cancers).sum()) * float(sim.pars.pop_scale)


def run_arm(ratio, stop, demographics):
    vals = []
    for sd in SEEDS:
        kw = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=stop,
                  dt=0.25, total_pop=1e6, n_agents=N, ms_agent_ratio=ratio,
                  rand_seed=sd, verbose=0)
        if demographics is not None:
            kw['demographics'] = demographics
        s = hpv.Sim(**kw)
        s.run()
        vals.append(total_cancers_people(s))
    return np.array(vals)


def bias(stop, demographics):
    single = run_arm(1, stop, demographics)
    multi = run_arm(12, stop, demographics)
    return single.mean(), multi.mean(), abs(multi.mean() - single.mean()) / single.mean()


if __name__ == '__main__':
    T = sc.timer()
    print(f'n_agents={N} both arms, {len(SEEDS)} seeds')
    for label, demog in [('DEFAULT demographics', None), ('NO demographics ([])', [])]:
        print(f'\n--- {label} ---')
        print(f'{"stop":>6} {"single":>12} {"multi":>12} {"rel.bias":>10}')
        for stop in STOPS:
            s, m, b = bias(stop, demog)
            print(f'{stop:>6} {s:>12.0f} {m:>12.0f} {b*100:>9.2f}%', flush=True)
    T.toc('TOTAL runtime')
