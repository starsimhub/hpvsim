"""Throwaway: does v2-spawning multiscale give an UNBIASED MEAN over seeds?

CRN-consistency loss is now accepted. The only remaining question for the
v2-spawning approach (grow placeholder agents per CIN cancer decision, exclude
fine agents from the network) is whether growing agents BIASES the mean cancer
count, or merely DECORRELATES individual seeds (mean unbiased). The mean gate
is a mean-over-independent-seeds comparison, so it doesn't depend on CRN pairing.

Equal n_agents both arms (so same people-space), ratio=1 vs ratio=12. If the
mean matches within noise AND multi has lower variance, v2-spawning works and
CRN loss is the only side effect.
"""
import numpy as np
import sciris as sc
import hpvsim as hpv

SEEDS = list(range(8))
N = 8000
STOP = 2050


def total_cancers_people(sim):
    return float(np.asarray(sim.results.hpv16.new_cancers).sum()) * float(sim.pars.pop_scale)


def arm(ratio):
    vals = []
    for sd in SEEDS:
        s = hpv.Sim(location='nigeria', genotypes=['hpv16'], start=1990, stop=STOP,
                    dt=0.25, total_pop=1e6, n_agents=N, ms_agent_ratio=ratio,
                    rand_seed=sd, verbose=0)
        s.run()
        vals.append(total_cancers_people(s))
    return np.array(vals)


if __name__ == '__main__':
    T = sc.timer()
    print(f'v2-spawning mean check: n_agents={N} both arms, {len(SEEDS)} seeds, stop={STOP}')
    single = arm(1)
    multi = arm(12)
    rel_bias = (multi.mean() - single.mean()) / single.mean()
    print(f'single ratio=1 : mean={single.mean():.0f}  std={single.std(ddof=1):.0f}  CV={single.std(ddof=1)/single.mean():.3f}')
    print(f'multi  ratio=12: mean={multi.mean():.0f}  std={multi.std(ddof=1):.0f}  CV={multi.std(ddof=1)/multi.mean():.3f}')
    print(f'relative bias of multi mean vs single: {rel_bias*100:+.2f}%')
    print(f'variance ratio (multi/single std): {multi.std(ddof=1)/single.std(ddof=1):.3f}')
    T.toc('runtime')
