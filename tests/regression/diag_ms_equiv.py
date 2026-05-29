"""Measurement harness for multiscale internal equivalence (Task 6).

Prints, per ratio, seed-mean over SEEDS: total cancers (people-space),
cum_infections (people-space), n_alive(final).
"""
import sys
import numpy as np
import hpvsim as hpv

# Fast iteration config: small but enough to see bias.
CFG = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2060,
           dt=0.25, total_pop=1e6, verbose=0)


def _cancers_people(sim):
    res = sim.results.hpv16
    return float(np.asarray(res.new_cancers).sum())  # already pop_scale-scaled at finalize


def _cum_inf_people(sim):
    r = sim.results.hpv16
    key = 'cum_infections' if 'cum_infections' in r else 'new_infections'
    if key == 'cum_infections':
        return float(np.asarray(r[key])[-1])
    return float(np.asarray(r[key]).sum())


def _n_alive(sim):
    return float(np.asarray(sim.results.n_alive)[-1])


def measure(n_agents, ratio, seeds):
    canc, inf, alive = [], [], []
    for sd in seeds:
        s = hpv.Sim(n_agents=n_agents, ms_agent_ratio=ratio, rand_seed=sd, **CFG)
        s.run()
        canc.append(_cancers_people(s))
        inf.append(_cum_inf_people(s))
        alive.append(_n_alive(s))
    return np.array(canc), np.array(inf), np.array(alive)


if __name__ == '__main__':
    n_single = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    n_multi = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    ratio = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    nseeds = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    seeds = range(nseeds)

    print(f'single: n={n_single} ratio=1 | multi: n={n_multi} ratio={ratio} | seeds={nseeds}')
    sc, si, sa = measure(n_single, 1, seeds)
    mc, mi, ma = measure(n_multi, ratio, seeds)

    print(f'{"metric":<18}{"single(r=1)":>16}{"multi":>16}{"bias%":>10}')
    for name, s, m in [('cancers', sc, mc), ('cum_inf', si, mi), ('n_alive', sa, ma)]:
        bias = (m.mean() - s.mean()) / s.mean() * 100
        print(f'{name:<18}{s.mean():>16.1f}{m.mean():>16.1f}{bias:>9.1f}%')
    print(f'cancers single std={sc.std(ddof=1):.1f} multi std={mc.std(ddof=1):.1f}')

    # Variance-at-equal-agents probe
    bc, _, _ = measure(n_multi, 1, seeds)
    print(f'equal-agents (n={n_multi}) cancers: r=1 std={bc.std(ddof=1):.1f} '
          f'r={ratio} std={mc.std(ddof=1):.1f}')
