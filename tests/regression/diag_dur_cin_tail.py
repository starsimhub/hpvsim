"""Confirm the dur_cin long tail drives old-age (55+) cancer in HIV- women.

(1) Percentiles of the calibrated dur_cin lognormal (mean=5x1.043, std=20x1.043
    for hpv16) -- a very long right tail would push some cancers to old ages.
(2) Actual age-at-cancer distribution from the calibrated sim (HIV- females),
    pooled over seeds, with the fraction occurring at 55+.

Run: .venv/Scripts/python.exe tests/regression/diag_dur_cin_tail.py [n_seeds=4] [n_agents=10000]
"""
import sys
from pathlib import Path

import numpy as np
import starsim as ss

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpvsim.hpv import HPV  # noqa: E402
from tests.regression import rwanda_calib as rc  # noqa: E402
from tests.regression.rwanda_calib import build_rwanda_sim  # noqa: E402


def _lognorm_pctiles(mean, std):
    """Percentiles of a lognormal with the given (arithmetic) mean & std,
    matching ss.lognorm_ex's parameterization."""
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    mu = np.log(mean) - sigma ** 2 / 2
    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    from scipy.stats import norm
    return {q: float(np.exp(mu + sigma * norm.ppf(q))) for q in qs}


class _AgeAtCancer(ss.Analyzer):
    def init_pre(self, sim):
        self.hpv = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv = sim.diseases.hiv
        super().init_pre(sim)
        self.age_neg = []
        self.age_pos = []

    def step(self):
        ti = self.sim.ti
        ppl = self.sim.people
        fem = ppl.female.values & ppl.alive.values
        age = ppl.age.values
        pos = self.hiv.infected.values
        newc = np.zeros(fem.shape, bool)
        for m in self.hpv:
            newc |= (m.cancerous.values & (m.ti_cancerous.values == ti))
        # This is a DISTRIBUTION of ages-at-cancer, not a rate, so no scale-
        # weighting is needed at ms_agent_ratio=5 (build_rwanda_sim default):
        # fine multiscale cancer agents are unbiased extra draws from the same
        # age-at-cancer distribution, so pooling them just lowers variance.
        self.age_neg.extend(age[newc & fem & ~pos].tolist())
        self.age_pos.extend(age[newc & fem & pos].tolist())


def main(n_seeds=4, n_agents=10000):
    print('=== (1) calibrated dur_cin lognormal percentiles (years) ===')
    for key in ['hpv16', 'hi5']:
        mean, std = rc._DUR_CIN[key]
        p = _lognorm_pctiles(mean * rc._DUR_CIN_SCALE, std * rc._DUR_CIN_SCALE)
        print(f'  {key}: median={p[0.5]:.1f}  75th={p[0.75]:.1f}  90th={p[0.9]:.1f}'
              f'  95th={p[0.95]:.1f}  99th={p[0.99]:.1f}')

    print(f'\n=== (2) age-at-cancer in calibrated sim ({n_seeds} seeds, n={n_agents}) ===')
    neg, pos = [], []
    for s in range(n_seeds):
        sim = build_rwanda_sim(seed=s, n_agents=n_agents, start=1960, stop=2020)
        pr = _AgeAtCancer()
        sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
        sim.run(verbose=0)
        a = [x for x in sim.analyzers.values() if isinstance(x, _AgeAtCancer)][0]
        neg += a.age_neg; pos += a.age_pos
    neg = np.array(neg); pos = np.array(pos)
    for lbl, arr in [('HIV-', neg), ('HIV+', pos)]:
        if len(arr) == 0:
            print(f'  {lbl}: no cancers'); continue
        print(f'  {lbl} (n={len(arr)}): median age={np.median(arr):.0f}  '
              f'mean={arr.mean():.0f}  90th={np.percentile(arr,90):.0f}  '
              f'%>=55={100*np.mean(arr>=55):.0f}%  %>=65={100*np.mean(arr>=65):.0f}%')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(n_seeds=int(a[0]) if len(a) > 0 else 4,
         n_agents=int(a[1]) if len(a) > 1 else 10000)