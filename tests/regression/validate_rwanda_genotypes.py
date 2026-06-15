"""Post-calibration validation: genotype distribution vs Rwanda registry.

The cancer-incidence calibration (calibrate_rwanda.py) fits only the TOTAL
(all-genotype) cancer level/shape/RR. It uses GLOBAL parameter scales that
preserve the genotypes' relative ratios, so it cannot reshape the genotype
mix. This script checks, on the calibrated best-fit, whether the model's
genotype mix matches two INDEPENDENT empirical (tissue-typing) targets:

  cancer  genotype dist (2018):  16=0.55 18=0.17 hi5=0.25 ohr=0.05
                                 (hpvsim_rwanda/data/rwanda_cancer_types.csv)
  precin  genotype dist (2018):  16=0.21 18=0.11 hi5=0.37 ohr=0.31
                                 (hpvsim_rwanda/data/rwanda_precin_types.csv)

If the cancer mix matches, the global-scale calibration was sufficient. If it
is off, that is the trigger to re-calibrate with PER-GENOTYPE parameters
(rel_beta / cin_fn.k) -- the precancer->cancer shift (hi5/ohr dominate CIN but
collapse in cancer; 16 jumps) is the differential-progression signal that would
drive per-genotype transform_prob.

STATE-MAPPING CAVEAT: v3 has BOTH a `cin` and a `precin` state. v2's "precin"
target maps ambiguously; we report the dist for BOTH v3 states so the intended
correspondence can be confirmed. The cancer mapping (`cancerous`) is unambiguous.

Distribution is the female prevalent-count share per genotype, pooled over
seeds across a window around the 2018 target year (proportions are robust to
the window; absorbing `cancerous` accumulates, so a snapshot-window mean is used).

Run: .venv/Scripts/python.exe tests/regression/validate_rwanda_genotypes.py \
         [n_seeds=10] [n_agents=10000] [ncpus=10]
Reads: results/rwanda_calib/best_pars.json  (the calibrated params)
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

_WIN = (2015, 2019)            # window around the 2018 typing target
_GENO_KEYS = rc._GENO_KEYS     # ['hpv16','hpv18','hi5','ohr']

CANCER_DIST = {'hpv16': 0.55, 'hpv18': 0.17, 'hi5': 0.25, 'ohr': 0.05}
PRECIN_DIST = {'hpv16': 0.21, 'hpv18': 0.11, 'hi5': 0.37, 'ohr': 0.31}


class _GProbe(ss.Analyzer):
    """Per-genotype female prevalent counts for cancerous / cin / precin."""
    def init_pre(self, sim):
        self.hpv = {d.name: d for d in sim.diseases.values()
                    if isinstance(d, HPV)}
        super().init_pre(sim)
        n = len(sim.t.timevec)
        self.yr = np.floor(np.asarray(sim.t.timevec, float)).astype(int)[:n]
        self.counts = {st: {g: np.zeros(n) for g in self.hpv}
                       for st in ('cancerous', 'cin', 'precin')}

    def step(self):
        ti = self.sim.ti
        fem = self.sim.people.female.values & self.sim.people.alive.values
        for g, m in self.hpv.items():
            for st in ('cancerous', 'cin', 'precin'):
                self.counts[st][g][ti] = (getattr(m, st).values & fem).sum()


def _run_one(seed, n_agents, start, stop, p):
    sim = build_sim(seed, n_agents, start, stop, **p)
    pr = _GProbe()
    sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
    sim.run(verbose=0)
    a = [x for x in sim.analyzers.values() if isinstance(x, _GProbe)][0]
    w = (a.yr >= _WIN[0]) & (a.yr <= _WIN[1])
    # Window-summed prevalent counts per state/genotype (proportions cancel dt).
    return {st: {g: float(a.counts[st][g][w].sum()) for g in a.counts[st]}
            for st in a.counts}


def _dist(pooled, state):
    tot = sum(pooled[state][g] for g in _GENO_KEYS)
    return {g: (pooled[state][g] / tot if tot > 0 else 0.0) for g in _GENO_KEYS}


def _report(label, model_dist, target):
    l1 = sum(abs(model_dist[g] - target[g]) for g in _GENO_KEYS)
    print(f'\n=== {label} genotype distribution (L1 dist = {l1:.3f}) ===')
    print(f'{"geno":>6}{"model":>9}{"target":>9}{"diff":>8}')
    for g in _GENO_KEYS:
        print(f'{g:>6}{model_dist[g]:>9.3f}{target.get(g,float("nan")):>9.3f}'
              f'{model_dist[g]-target.get(g,0):>+8.3f}')
    return l1


def main(n_seeds=10, n_agents=10000, ncpus=10, start=1960, stop=2020):
    bp = sc.loadjson(_ROOT / 'results' / 'rwanda_calib' / 'best_pars.json')
    params = dict(bp['best_params'])
    params.setdefault('base_beta', 0.12)   # fixed in the calibration
    print(f'Validating best fit (gof={bp.get("best_gof")}, '
          f'{bp.get("n_trials_done")} trials) with params:')
    print('  ' + str({k: round(v, 3) for k, v in params.items()}))
    print(f'{n_seeds} seeds, n_agents={n_agents}, {ncpus} cores, window {_WIN}.')

    argset = [(s, n_agents, start, stop, params) for s in range(n_seeds)]
    results = sc.parallelize(_run_one, iterarg=argset, ncpus=ncpus)
    pooled = {st: {g: sum(r[st][g] for r in results) for g in _GENO_KEYS}
              for st in ('cancerous', 'cin', 'precin')}

    l1_canc = _report('CANCER (state=cancerous)', _dist(pooled, 'cancerous'),
                      CANCER_DIST)
    # v2 "precin" target maps ambiguously -> report both v3 states.
    l1_cin = _report('PRECANCER vs v3 `cin` state', _dist(pooled, 'cin'),
                     PRECIN_DIST)
    l1_pre = _report('PRECANCER vs v3 `precin` state', _dist(pooled, 'precin'),
                     PRECIN_DIST)

    print('\n=== verdict ===')
    print(f'cancer-mix L1 = {l1_canc:.3f}  '
          f'({"MATCH (<0.15): global scales sufficient" if l1_canc < 0.15 else "OFF (>=0.15): consider per-genotype calibration"})')
    print(f'precancer-mix L1: cin={l1_cin:.3f}, precin={l1_pre:.3f} '
          f'(confirm which v3 state the registry `precin` typing corresponds to)')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(
        n_seeds=int(a[0]) if len(a) > 0 else 10,
        n_agents=int(a[1]) if len(a) > 1 else 10000,
        ncpus=int(a[2]) if len(a) > 2 else 10,
    )