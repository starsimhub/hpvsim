"""Throwaway measurement: does binomial multiscale distort cancer-BY-AGE?

Compares single-scale (ms_agent_ratio=1) vs binomial multiscale
(ms_agent_ratio=12) on scale-weighted cumulative cancers-by-age and
cancer-deaths-by-age, plus scalar mean age-at-cancer / age-at-cancer-death.

Custom analyzer approach: each step, scan the RAW people arrays (alive+dead
both retained in .raw) for agents whose ti_cancerous / ti_dead_cancer rounds
to the current tick, and accumulate age*scale into fixed 5-yr bins. Scale is
read at the moment of the event (post-multiscale-shrink), so it is the correct
people-space weight for that event. This directly gives scale-weighted
cumulative flows-by-age over the whole run (NOT a prevalent snapshot like
hpv.AgeResults), which is what we need for a flow distribution.
"""
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv

EDGES = np.arange(0., 90., 5.)          # 5-yr bins 0..85, last bin 85+ via 85-90 grab-all
NBINS = len(EDGES) - 1


class ByAgeFlows(ss.Analyzer):
    """Accumulate scale-weighted cancers-by-age and cancer-deaths-by-age."""

    def init_pre(self, sim):
        from hpvsim.hpv import HPV
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        self.cancers_by_age = np.zeros(NBINS)
        self.deaths_by_age = np.zeros(NBINS)
        self.sum_age_cancer = 0.0   # scale-weighted sum of age at cancer
        self.n_cancer = 0.0         # scale-weighted count
        self.sum_age_death = 0.0
        self.n_death = 0.0
        return

    def step(self):
        sim = self.sim
        ti = sim.ti
        ppl = sim.people
        age_raw = np.asarray(ppl.age.raw)
        scale_raw = np.asarray(ppl.scale.raw)
        dt_yr = float(self.t.dt.years if hasattr(self.t.dt, 'years') else self.t.dt)
        for mod in self.hpv_modules:
            tic = np.asarray(mod.ti_cancerous.raw)
            tid = np.asarray(mod.ti_dead_cancer.raw)
            canc_raw = np.asarray(mod.cancerous.raw)

            # Newly cancerous THIS tick. Require the cancerous BoolState set
            # (so we don't double-count agents whose schedule was later
            # overwritten/cancelled by the multiscale reconcile or cross-
            # genotype cancellation). Matches the module's own
            # (cin & ti_cancerous<=ti) transition the step it fires.
            new_canc = (~np.isnan(tic)) & (np.round(tic) == ti) & canc_raw
            if new_canc.any():
                a = age_raw[new_canc]
                w = scale_raw[new_canc]
                h, _ = np.histogram(a, bins=EDGES, weights=w)
                # grab-all the 85+ tail into the last bin
                tail = a >= EDGES[-1]
                h[-1] += w[tail].sum()
                self.cancers_by_age += h
                self.sum_age_cancer += float((a * w).sum())
                self.n_cancer += float(w.sum())

            # Cancer deaths THIS tick. Agent may already be removed from
            # auids but persists in .raw. Detect by ti_dead_cancer rounding
            # to ti; +dt_yr on age to match module's recorded convention.
            new_dead = (~np.isnan(tid)) & (np.round(tid) == ti)
            if new_dead.any():
                a = age_raw[new_dead] + dt_yr
                w = scale_raw[new_dead]
                h, _ = np.histogram(a, bins=EDGES, weights=w)
                tail = a >= EDGES[-1]
                h[-1] += w[tail].sum()
                self.deaths_by_age += h
                self.sum_age_death += float((a * w).sum())
                self.n_death += float(w.sum())
        return


def run_arm(ms_ratio, n_agents, seeds, total_pop=1e6):
    cba = np.zeros((len(seeds), NBINS))
    dba = np.zeros((len(seeds), NBINS))
    mean_age_c = np.zeros(len(seeds))
    mean_age_d = np.zeros(len(seeds))
    tot_c = np.zeros(len(seeds))
    tot_d = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        sim = hpv.Sim(location='nigeria', genotypes=['hpv16'],
                      start=1990, stop=2060, dt=0.25,
                      total_pop=total_pop, n_agents=n_agents,
                      ms_agent_ratio=ms_ratio, rand_seed=seed, verbose=0,
                      analyzers=[ByAgeFlows()])
        sim.run()
        az = [a for a in sim.analyzers.values() if isinstance(a, ByAgeFlows)][0]
        cba[i] = az.cancers_by_age
        dba[i] = az.deaths_by_age
        mean_age_c[i] = az.sum_age_cancer / az.n_cancer if az.n_cancer > 0 else np.nan
        mean_age_d[i] = az.sum_age_death / az.n_death if az.n_death > 0 else np.nan
        tot_c[i] = az.n_cancer
        tot_d[i] = az.n_death
        print(f'  ratio={ms_ratio} seed={seed}: tot_cancer={tot_c[i]:.0f} '
              f'tot_death={tot_d[i]:.0f} mean_age_c={mean_age_c[i]:.2f} '
              f'mean_age_d={mean_age_d[i]:.2f}', flush=True)
    return dict(cba=cba, dba=dba, mean_age_c=mean_age_c, mean_age_d=mean_age_d,
                tot_c=tot_c, tot_d=tot_d)


def report(single, multi):
    bin_lbls = [f'{int(EDGES[i])}-{int(EDGES[i+1])}' for i in range(NBINS - 1)]
    bin_lbls.append(f'{int(EDGES[-2])}+')

    s_mean = single['cba'].mean(0); m_mean = multi['cba'].mean(0)
    s_std = single['cba'].std(0, ddof=1); m_std = multi['cba'].std(0, ddof=1)
    s_cv = np.where(s_mean > 0, s_std / s_mean, np.nan)
    m_cv = np.where(m_mean > 0, m_std / m_mean, np.nan)
    relbias = np.where(s_mean > 0, (m_mean - s_mean) / s_mean, np.nan)

    print('\n===== CANCERS BY AGE (people-space, cumulative over run) =====')
    print(f'{"bin":>6} {"single_mean":>12} {"multi_mean":>12} {"rel.bias":>9} '
          f'{"single_CV":>10} {"multi_CV":>10} {"CV_ratio":>9}')
    for i in range(NBINS):
        cvr = m_cv[i] / s_cv[i] if (s_cv[i] and s_cv[i] > 0) else np.nan
        print(f'{bin_lbls[i]:>6} {s_mean[i]:>12.1f} {m_mean[i]:>12.1f} '
              f'{relbias[i]*100:>8.1f}% {s_cv[i]:>10.3f} {m_cv[i]:>10.3f} '
              f'{cvr:>9.2f}')
    print(f'{"TOTAL":>6} {s_mean.sum():>12.1f} {m_mean.sum():>12.1f} '
          f'{(m_mean.sum()/s_mean.sum()-1)*100:>8.1f}%')

    # deaths
    sd_mean = single['dba'].mean(0); md_mean = multi['dba'].mean(0)
    sd_std = single['dba'].std(0, ddof=1); md_std = multi['dba'].std(0, ddof=1)
    sd_cv = np.where(sd_mean > 0, sd_std / sd_mean, np.nan)
    md_cv = np.where(md_mean > 0, md_std / md_mean, np.nan)
    relbias_d = np.where(sd_mean > 0, (md_mean - sd_mean) / sd_mean, np.nan)
    print('\n===== CANCER DEATHS BY AGE (people-space, cumulative) =====')
    print(f'{"bin":>6} {"single_mean":>12} {"multi_mean":>12} {"rel.bias":>9} '
          f'{"single_CV":>10} {"multi_CV":>10} {"CV_ratio":>9}')
    for i in range(NBINS):
        cvr = md_cv[i] / sd_cv[i] if (sd_cv[i] and sd_cv[i] > 0) else np.nan
        print(f'{bin_lbls[i]:>6} {sd_mean[i]:>12.1f} {md_mean[i]:>12.1f} '
              f'{relbias_d[i]*100:>8.1f}% {sd_cv[i]:>10.3f} {md_cv[i]:>10.3f} '
              f'{cvr:>9.2f}')

    print('\n===== SCALAR LOCATION (scale-weighted mean age) =====')
    print(f'mean age-at-cancer:       single = {np.nanmean(single["mean_age_c"]):.3f} '
          f'(sd {np.nanstd(single["mean_age_c"], ddof=1):.3f})   '
          f'multi = {np.nanmean(multi["mean_age_c"]):.3f} '
          f'(sd {np.nanstd(multi["mean_age_c"], ddof=1):.3f})')
    print(f'mean age-at-cancer-death: single = {np.nanmean(single["mean_age_d"]):.3f} '
          f'(sd {np.nanstd(single["mean_age_d"], ddof=1):.3f})   '
          f'multi = {np.nanmean(multi["mean_age_d"]):.3f} '
          f'(sd {np.nanstd(multi["mean_age_d"], ddof=1):.3f})')

    # summary aggregates
    valid = (s_cv > 0) & np.isfinite(s_cv) & np.isfinite(m_cv) & (s_mean > 5)
    print('\n===== SUMMARY (bins with single_mean>5) =====')
    print(f'mean |rel.bias| cancers      = {np.nanmean(np.abs(relbias[valid]))*100:.1f}%')
    print(f'mean CV single (cancers)     = {np.nanmean(s_cv[valid]):.3f}')
    print(f'mean CV multi  (cancers)     = {np.nanmean(m_cv[valid]):.3f}')
    print(f'mean CV ratio multi/single   = {np.nanmean(m_cv[valid]/s_cv[valid]):.2f}')


if __name__ == '__main__':
    # NOTE: in this build total_pop does NOT drive per-agent scale (single-scale
    # agents have scale=1.0 regardless), so people-space == n_agents. To equate
    # people-space between arms we therefore use the SAME n_agents in both arms;
    # the multiscale variance reduction comes from binomial sub-sampling at the
    # rare cancer decision, not from a coarse/fine population split. (Deviates
    # from the brief's n_single=20000/n_multi=1700, which would NOT be
    # people-space-equivalent in this build — see report.)
    SEEDS = list(range(8))
    N = 12000
    T = sc.timer()
    print(f'SINGLE-SCALE arm: ms_agent_ratio=1, n_agents={N}, {len(SEEDS)} seeds')
    single = run_arm(1, N, SEEDS, total_pop=None)
    print(f'\nMULTISCALE arm: ms_agent_ratio=12, n_agents={N}, {len(SEEDS)} seeds')
    multi = run_arm(12, N, SEEDS, total_pop=None)
    report(single, multi)
    T.toc('TOTAL runtime')
