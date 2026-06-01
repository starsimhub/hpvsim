"""Fig-5-style probe: does multiscale tighten the EVENT-AGE DISTRIBUTION?

Figure 5 boxplots are built from UNWEIGHTED per-cancer-event ages (one sample
per cancer onset; people.scale is ignored). The relevant payoff is therefore
EFFECTIVE SAMPLE SIZE of cancer-pathway events and the seed-to-seed error bar
on a distribution statistic (here: median age at cancer + 25th/75th pctile),
NOT the scale-weighted cancer count.

Run: python spike/quick_dist.py <ratio>
Reports, across seeds: per-seed #events (unweighted), and the across-seed std
of the median / p25 / p75 age-at-cancer (the 'error bar' on the boxplot).
"""
import sys
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv

SEEDS = list(range(5))
N = 6000
STOP = 2045


class CancerAges(ss.Analyzer):
    """Record UNWEIGHTED age at each cancer onset (one sample per event)."""
    def init_pre(self, sim):
        from hpvsim.hpv import HPV
        self.mods = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        self.ages = []

    def step(self):
        sim = self.sim
        age_raw = np.asarray(sim.people.age.raw)
        for m in self.mods:
            new = np.where(np.asarray(m.ti_cancerous.raw) == sim.ti)[0]
            if len(new):
                self.ages.extend(age_raw[new].tolist())


def run_arm(ratio):
    n_events = []
    med = []; p25 = []; p75 = []
    for sd in SEEDS:
        az = CancerAges()
        s = hpv.Sim(location='nigeria', genotypes=['hpv16'], start=1990, stop=STOP,
                    dt=0.25, total_pop=1e6, n_agents=N, ms_agent_ratio=ratio,
                    rand_seed=sd, verbose=0, analyzers=[az])
        s.run()
        a = np.asarray([x for x in s.analyzers['cancerages'].ages if 0 <= x < 90])
        n_events.append(len(a))
        med.append(np.median(a)); p25.append(np.percentile(a, 25)); p75.append(np.percentile(a, 75))
    n_events = np.array(n_events); med = np.array(med); p25 = np.array(p25); p75 = np.array(p75)
    print(f'ratio={ratio}: events/seed mean={n_events.mean():.0f}  '
          f'median-age mean={med.mean():.2f} std={med.std(ddof=1):.3f}  '
          f'p25 mean={p25.mean():.2f} std={p25.std(ddof=1):.3f}  '
          f'p75 mean={p75.mean():.2f} std={p75.std(ddof=1):.3f}')
    return med, p25, p75


if __name__ == '__main__':
    ratio = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    T = sc.timer()
    print(f'n_agents={N}, stop={STOP}, {len(SEEDS)} seeds')
    run_arm(ratio)
    T.toc('runtime')