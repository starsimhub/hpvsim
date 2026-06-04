"""Verify the ported Rwanda calibration reproduces v2 HPV + cancer levels.

Runs the calibrated incidence-driven Rwanda sim for a few seeds and reports:
  - adult-female (15-49) any-genotype HPV prevalence by year
  - HIV-stratified cervical-cancer incidence per 100k women, pooled across
    seeds over a measurement window (numerator = female cancers, denominator =
    female-years alive by HIV status), reconstructing v2's metric
    cancer_incidence = cancers / n_females_alive * 1e5.

v2 targets (2017): HIV- ~15/100k, HIV+ ~36/100k.

Run: .venv/Scripts/python.exe tests/regression/diag_rwanda_calib_check.py [n_agents] [seeds]
"""
import sys
from pathlib import Path

import numpy as np
import starsim as ss

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.regression.rwanda_calib import build_rwanda_sim  # noqa: E402
from hpvsim.hpv import HPV  # noqa: E402
from hpvsim.hiv import HIV  # noqa: E402


class _Probe(ss.Analyzer):
    def init_pre(self, sim):
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv_module = next(d for d in sim.diseases.values() if isinstance(d, HIV))
        super().init_pre(sim)
        n = len(sim.t.timevec)
        self.n_f_pos = np.zeros(n); self.n_f_neg = np.zeros(n)
        self.canc_f_pos = np.zeros(n); self.canc_f_neg = np.zeros(n)
        self.adult_f = np.zeros(n); self.adult_f_hpv = np.zeros(n)

    def step(self):
        ti = self.sim.ti
        ppl = self.sim.people
        alive = ppl.alive.values
        female = ppl.female.values
        age = ppl.age.values
        pos = self.hiv_module.infected.values
        f_pos = alive & female & pos
        f_neg = alive & female & ~pos
        self.n_f_pos[ti] = f_pos.sum(); self.n_f_neg[ti] = f_neg.sum()
        any_hpv = np.zeros(alive.shape, dtype=bool)
        new_cancer = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            any_hpv |= m.infected.values
            new_cancer |= (m.cancerous.values & (m.ti_cancerous.values == ti))
        self.canc_f_pos[ti] = (new_cancer & f_pos).sum()
        self.canc_f_neg[ti] = (new_cancer & f_neg).sum()
        adult_f = alive & female & (age >= 15) & (age < 50)
        self.adult_f[ti] = adult_f.sum()
        self.adult_f_hpv[ti] = (adult_f & any_hpv).sum()


def _per_year(arr, years, reduce):
    out = {}
    for y in np.unique(years):
        out[int(y)] = reduce(arr[years == y])
    return out


def run(n_agents, seeds, start=1960, stop=2020):
    agg = []
    for seed in range(seeds):
        sim = build_rwanda_sim(seed=seed, n_agents=n_agents, start=start, stop=stop)
        pr = _Probe()
        sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
        sim.run()
        tv = np.asarray(sim.t.timevec, dtype=float)
        years = np.floor(tv).astype(int)[:len(pr.n_f_pos)]
        agg.append((years, pr))
    return agg


if __name__ == '__main__':
    n_agents = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    agg = run(n_agents, seeds)

    print(f'\n=== Adult-female (15-49) any-HPV prevalence, mean over {seeds} seeds ===')
    print(f'{"year":>6} {"prev":>8}')
    for y in (1980, 1990, 2000, 2010, 2017, 2019):
        vals = []
        for years, pr in agg:
            num = _per_year(pr.adult_f_hpv, years, np.sum)
            den = _per_year(pr.adult_f, years, np.sum)
            if y in num and den.get(y, 0) > 0:
                vals.append(num[y] / den[y])
        if vals:
            print(f'{y:>6} {np.mean(vals):>8.3f}')

    # Pooled HIV-stratified incidence over windows.
    print(f'\n=== HIV-stratified cancer incidence /100k (pooled over {seeds} seeds) ===')
    for lo, hi in [(2015, 2019), (2010, 2019), (2005, 2019)]:
        cp = cn = nfp = nfn = 0.0
        for years, pr in agg:
            mask = (years >= lo) & (years <= hi)
            cp += pr.canc_f_pos[mask].sum()
            cn += pr.canc_f_neg[mask].sum()
            # female-years = mean headcount per step * (steps) * dt; pool raw step counts * dt
            nfp += pr.n_f_pos[mask].sum() * 0.25
            nfn += pr.n_f_neg[mask].sum() * 0.25
        inc_p = cp / nfp * 1e5 if nfp else 0.0
        inc_n = cn / nfn * 1e5 if nfn else 0.0
        rr = inc_p / inc_n if inc_n else float('nan')
        print(f'  window {lo}-{hi}: HIV+={inc_p:6.1f}  HIV-={inc_n:6.1f}  RR={rr:4.1f}  '
              f'(cancF+={cp:.0f}, cancF-={cn:.0f})')
    print('\n  v2 target (2017): HIV+ ~36 [28.8,43.7], HIV- ~15 [14.5,16.1], RR~2.4')
