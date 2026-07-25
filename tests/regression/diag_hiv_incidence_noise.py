"""Probe: how measurable is HIV-stratified cancer incidence in the v3 Rwanda sim?

Runs the incidence-driven Rwanda co-infection sim for a couple of seeds and a
couple of population sizes, then reconstructs the v2-style per-year metric

    cancer_incidence_with_hiv[y] = cancers_with_hiv[y] / n_females_with_hiv_alive[y] * 1e5

using the HIVStratifiedResults cancer counts (numerator) and a denominator we
reconstruct here from people + hiv state (the analyzer does NOT yet record the
female-by-HIV-status denominators). Prints the HIV+ female headcount, annual
cancer counts, and the resulting incidence noise around 2017 so we can pick the
measurement window / population / pooling before writing the T13 gate.

Run: .venv/Scripts/python.exe tests/regression/diag_hiv_incidence_noise.py
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hpvsim as hpv  # noqa: E402
from tests.regression.sim_hiv_rwanda import build_rwanda_sim_incidence  # noqa: E402


def _female_hiv_denominators(sim):
    """Reconstruct per-step n_females_{with,no}_hiv_alive from end-state arrays?

    We can't get a per-step denominator from end-state alone, so instead we
    attach a tiny analyzer at run time. Here we just return the analyzer's
    recorded cancers plus a denominator analyzer's recordings.
    """
    raise NotImplementedError


import starsim as ss  # noqa: E402


class _DenomProbe(ss.Analyzer):
    """Record per-step female headcount by HIV status + female cancers by status."""

    def init_pre(self, sim):
        from hpvsim.hpv import HPV
        from hpvsim.hiv import HIV
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv_module = next(d for d in sim.diseases.values() if isinstance(d, HIV))
        super().init_pre(sim)
        n = len(sim.t.timevec)
        self.n_f_pos = np.zeros(n)
        self.n_f_neg = np.zeros(n)
        self.canc_f_pos = np.zeros(n)
        self.canc_f_neg = np.zeros(n)

    def step(self):
        ti = self.sim.ti
        ppl = self.sim.people
        alive = ppl.alive.values
        female = ppl.female.values
        pos = self.hiv_module.infected.values
        f_pos = alive & female & pos
        f_neg = alive & female & ~pos
        self.n_f_pos[ti] = f_pos.sum()
        self.n_f_neg[ti] = f_neg.sum()
        new_cancer = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            new_cancer |= (m.cancerous.values & (m.ti_cancerous.values == ti))
        self.canc_f_pos[ti] = (new_cancer & f_pos).sum()
        self.canc_f_neg[ti] = (new_cancer & f_neg).sum()


def _per_year(arr, dt, reduce):
    spy = int(round(1.0 / dt))
    n_full = len(arr) // spy
    t = arr[:n_full * spy].reshape(n_full, spy)
    return reduce(t, axis=1), n_full


def probe(n_agents, seed):
    sim = build_rwanda_sim_incidence(seed=seed, n_agents=n_agents,
                                     start=1985, stop=2020)
    probe = _DenomProbe()
    sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [probe]
    sim.run()
    dt = float(sim.t.dt)
    start = float(sim.t.start)

    canc_pos_y, ny = _per_year(probe.canc_f_pos, dt, np.sum)
    canc_neg_y, _ = _per_year(probe.canc_f_neg, dt, np.sum)
    nf_pos_y, _ = _per_year(probe.n_f_pos, dt, np.mean)
    nf_neg_y, _ = _per_year(probe.n_f_neg, dt, np.mean)
    years = start + np.arange(ny)

    def inc(c, n):
        return np.where(n > 0, c / n * 1e5, 0.0)

    inc_pos = inc(canc_pos_y, nf_pos_y)
    inc_neg = inc(canc_neg_y, nf_neg_y)
    return dict(years=years, canc_pos=canc_pos_y, canc_neg=canc_neg_y,
                nf_pos=nf_pos_y, nf_neg=nf_neg_y, inc_pos=inc_pos, inc_neg=inc_neg)


if __name__ == '__main__':
    for n_agents in (5000, 20000):
        print('=' * 78)
        print(f'n_agents={n_agents}')
        for seed in (0, 1):
            r = probe(n_agents, seed)
            yi = {int(y): i for i, y in enumerate(r['years'])}
            print(f'  seed={seed}')
            print(f'    {"year":>6} {"nF+":>6} {"nF-":>8} {"cancF+":>7} {"cancF-":>7} '
                  f'{"inc+":>7} {"inc-":>7}')
            for y in (2010, 2013, 2015, 2017, 2019):
                if y in yi:
                    i = yi[y]
                    print(f'    {y:>6} {r["nf_pos"][i]:>6.0f} {r["nf_neg"][i]:>8.0f} '
                          f'{r["canc_pos"][i]:>7.0f} {r["canc_neg"][i]:>7.0f} '
                          f'{r["inc_pos"][i]:>7.1f} {r["inc_neg"][i]:>7.1f}')
            # Pooled over a 2010-2019 window:
            w = [i for y, i in yi.items() if 2010 <= y <= 2019]
            cp, cn = r['canc_pos'][w].sum(), r['canc_neg'][w].sum()
            np_, nn = r['nf_pos'][w].mean(), r['nf_neg'][w].mean()
            n_years = len(w)
            inc_p = cp / (np_ * n_years) * 1e5
            inc_n = cn / (nn * n_years) * 1e5
            print(f'    pooled 2010-2019: incidence+ = {inc_p:.1f}, '
                  f'incidence- = {inc_n:.1f} '
                  f'(cancF+={cp:.0f}, cancF-={cn:.0f}, meanF+={np_:.0f}, meanF-={nn:.0f})')
