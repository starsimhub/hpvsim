"""Fast bias probe: total cancers (people-space) + scale-weighted mean age at
cancer, single-scale (ratio=1) vs multiscale (ratio=12), equal n_agents.

Small/short for speed. Reports relative bias of the MEAN cancer count and the
shift in scale-weighted mean age-at-cancer (a by-age location check that a
total-count match can hide).
"""
import numpy as np
import sciris as sc
import hpvsim as hpv

SEEDS = list(range(6))
N = 6000
STOP = 2045
RATIO = 12


def run_arm(ratio):
    tot = np.zeros(len(SEEDS))
    mean_age = np.zeros(len(SEEDS))
    for i, sd in enumerate(SEEDS):
        s = hpv.Sim(location='nigeria', genotypes=['hpv16'], start=1990, stop=STOP,
                    dt=0.25, total_pop=1e6, n_agents=N, ms_agent_ratio=ratio,
                    rand_seed=sd, verbose=0)
        s.run()
        r = s.results.hpv16
        nc = np.asarray(r.new_cancers, dtype=float)
        tot[i] = nc.sum() * float(s.pars.pop_scale)
        sa = np.asarray(r.sum_age_at_cancer, dtype=float)
        mean_age[i] = sa.sum() / nc.sum() if nc.sum() > 0 else np.nan
    return tot, mean_age


if __name__ == '__main__':
    T = sc.timer()
    s_tot, s_age = run_arm(1)
    m_tot, m_age = run_arm(RATIO)
    rb = (m_tot.mean() - s_tot.mean()) / s_tot.mean()
    vr = m_tot.std(ddof=1) / s_tot.std(ddof=1)
    print(f'n_agents={N} both arms, stop={STOP}, {len(SEEDS)} seeds, ratio={RATIO}')
    print(f'TOTAL CANCERS  single: mean={s_tot.mean():.0f} std={s_tot.std(ddof=1):.0f} CV={s_tot.std(ddof=1)/s_tot.mean():.3f}')
    print(f'TOTAL CANCERS  multi : mean={m_tot.mean():.0f} std={m_tot.std(ddof=1):.0f} CV={m_tot.std(ddof=1)/m_tot.mean():.3f}')
    print(f'  -> mean rel.bias = {rb*100:+.2f}%   variance ratio (multi/single std) = {vr:.3f}')
    print(f'MEAN AGE@CANCER single={np.nanmean(s_age):.2f}  multi={np.nanmean(m_age):.2f}  shift={np.nanmean(m_age)-np.nanmean(s_age):+.2f} yr')
    T.toc('runtime')
