"""Diagnose the stubborn HIV+ 55+ cancer bin (model ~0 vs registry 30/100k).

SLATED FOR DELETION IN v3.3 (test cleanup). This is a one-off script from the
v2 -> v3 Rwanda migration, not a test: it is not collected by pytest, it has
no assertions, and several of these run a full Optuna calibration or a
multi-seed sim. They are kept for now because the v3 HIV-HPV parameterization
was derived here and the derivation is worth being able to re-read. Anything
here that should outlive 3.3 -- most likely the CalibProbe-style age-by-HIV
probes, which localizations reimplement -- needs promoting into the package
or into ``tests/`` first.

Hypothesis: the HIV+ 55+ stratum is nearly empty (women infected young in the
1985-2004 pre-ART epidemic largely died before 55), so 30/100k over a tiny
denominator rounds to ~0 cancers -- a demographic/sparsity feature, not a
natural-history calibration fault. This builds the calibrated best-fit and,
for each HIV+ female age bin over the measurement window, reports:

  - women-years (the denominator -- is a55 starved relative to a35/a45?)
  - cancers + implied incidence
  - CD4 composition (<200 vs >=200) and ART coverage (the connector's HIV->HPV
    effect is weaker for CD4>=200 / ART-reconstituted women)

If a55 women-years are tiny vs a45 AND mostly ART/CD4>=200, the zero is
demographic + weak-effect, confirming it is not fixable via the calibration
parameters.

Run: .venv/Scripts/python.exe tests/regression/diag_rwanda_hivpos_a55.py \
         [n_seeds=10] [n_agents=10000] [ncpus=10]
Reads: results/rwanda_calib/best_pars.json
"""
import sys
from pathlib import Path

import numpy as np
import sciris as sc

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import starsim as ss  # noqa: E402
from hpvsim.hpv import HPV  # noqa: E402
from tests.regression import rwanda_calib as rc  # noqa: E402
from tests.regression.calibrate_rwanda import build_sim  # noqa: E402

_AGE_EDGES = [25, 35, 45, 55, 200]
_AGE_LABELS = ['25-35', '35-45', '45-55', '55+']
_WIN = (2010, 2019)
_SCALE = 1e5
_DT = rc._DT


class _Probe(ss.Analyzer):
    def init_pre(self, sim):
        self.hpv = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv = sim.diseases.hiv
        super().init_pre(sim)
        n = len(sim.t.timevec)
        nb = len(_AGE_LABELS)
        self.yr = np.floor(np.asarray(sim.t.timevec, float)).astype(int)[:n]
        self.nf = np.zeros((nb, n))        # HIV+ female headcount
        self.canc = np.zeros((nb, n))      # new cancers, HIV+ female
        self.cd4lt = np.zeros((nb, n))     # CD4<200
        self.cd4ge = np.zeros((nb, n))     # CD4>=200
        self.onart = np.zeros((nb, n))     # on ART

    def step(self):
        ti = self.sim.ti
        ppl = self.sim.people
        fpos = ppl.female.values & ppl.alive.values & self.hiv.infected.values
        age = ppl.age.values
        cd4 = self.hiv.cd4.values
        art = self.hiv.on_art.values
        newc = np.zeros(fpos.shape, bool)
        for m in self.hpv:
            newc |= (m.cancerous.values & (m.ti_cancerous.values == ti))
        for bi, (lo, hi) in enumerate(zip(_AGE_EDGES[:-1], _AGE_EDGES[1:])):
            b = fpos & (age >= lo) & (age < hi)
            self.nf[bi, ti] = b.sum()
            self.canc[bi, ti] = (b & newc).sum()
            self.cd4lt[bi, ti] = (b & (cd4 < 200)).sum()
            self.cd4ge[bi, ti] = (b & (cd4 >= 200)).sum()
            self.onart[bi, ti] = (b & art).sum()


def _run_one(seed, n_agents, start, stop, p):
    sim = build_sim(seed, n_agents, start, stop, **p)
    pr = _Probe()
    sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
    sim.run(verbose=0)
    a = [x for x in sim.analyzers.values() if isinstance(x, _Probe)][0]
    w = (a.yr >= _WIN[0]) & (a.yr <= _WIN[1])
    return {k: getattr(a, k)[:, w].sum(axis=1)
            for k in ('nf', 'canc', 'cd4lt', 'cd4ge', 'onart')}


def main(n_seeds=10, n_agents=10000, ncpus=10, start=1960, stop=2020):
    bp = sc.loadjson(_ROOT / 'results' / 'rwanda_calib' / 'best_pars.json')
    params = dict(bp['best_params'])
    params.setdefault('base_beta', 0.12)
    print(f'Diagnosing HIV+ a55 on best fit (gof={bp.get("best_gof")}). '
          f'{n_seeds} seeds, n={n_agents}, {ncpus} cores, window {_WIN}.')

    argset = [(s, n_agents, start, stop, params) for s in range(n_seeds)]
    res = sc.parallelize(_run_one, iterarg=argset, ncpus=ncpus)
    agg = {k: sum(r[k] for r in res) for k in res[0]}   # (nb,) pooled step-sums

    nyears = _WIN[1] - _WIN[0] + 1
    print(f'\n{"bin":>7}{"women-yrs":>11}{"cancers":>9}{"inc/100k":>10}'
          f'{"%CD4<200":>10}{"%on ART":>9}')
    for bi, lab in enumerate(_AGE_LABELS):
        wy = agg['nf'][bi] * _DT                 # pooled women-years
        c = agg['canc'][bi]
        inc = c / wy * _SCALE if wy > 0 else 0.0
        tot = agg['cd4lt'][bi] + agg['cd4ge'][bi]
        pct_lt = 100 * agg['cd4lt'][bi] / tot if tot > 0 else float('nan')
        pct_art = 100 * agg['onart'][bi] / agg['nf'][bi] if agg['nf'][bi] > 0 else float('nan')
        print(f'{lab:>7}{wy:>11.0f}{c:>9.0f}{inc:>10.1f}{pct_lt:>10.1f}{pct_art:>9.1f}')
    tgt = {'25-35': 15, '35-45': 76, '45-55': 80, '55+': 30}
    print(f'\nregistry HIV+ targets: ' + ', '.join(f'{k}={v}' for k, v in tgt.items()))
    a55_wy = agg['nf'][3] * _DT
    a45_wy = agg['nf'][2] * _DT
    print(f'\na55 women-years = {a55_wy:.0f} vs a45 = {a45_wy:.0f} '
          f'(ratio {a55_wy/a45_wy:.2f}); expected cancers at 30/100k = '
          f'{30/_SCALE*a55_wy:.2f}')
    print('Verdict: if a55 women-years are a small fraction of a45 AND expected '
          'cancers <~1, the zero is demographic sparsity, not a param fault.')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(
        n_seeds=int(a[0]) if len(a) > 0 else 10,
        n_agents=int(a[1]) if len(a) > 1 else 10000,
        ncpus=int(a[2]) if len(a) > 2 else 10,
    )