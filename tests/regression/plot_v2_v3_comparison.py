"""Compare the published v2 (v2.2.6) Rwanda fit against the calibrated v3 fit,
both vs the registry targets.

v2 published outputs: hpvsim_v23_validation/hpvsim_rwanda/results/v2.2.6_baseline/
  figS2_cancer_incidence_{no,with}_hiv.csv (bins 0-3 = ages 25/35/45/55, boxplot)
  figS2_{cancerous,precin}_genotype_dist.csv (bins 0-3 = 16/18/hi5/ohr, boxplot)
v3: calibrated build_sim (results/rwanda_calib/best_pars.json), pooled over seeds
    with bootstrap 90% CIs (reuses plot_rwanda_calibrated machinery).

4 panels (v2 published / v3 / registry, side by side):
  cancer incidence by age HIV- ; HIV+ ; cancer genotype dist ; precancer dist.

Run: .venv/Scripts/python.exe tests/regression/plot_v2_v3_comparison.py [n_seeds=12] [n_agents=10000]
Saves: results/rwanda_calib/v2_v3_comparison.png  (+ figures/)
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.regression.plot_rwanda_calibrated import (  # noqa: E402
    _run_one, _pool_byage, _boot_rate, _boot_dist, _SCALE, _GK)

_V2 = Path('C:/Users/ryanhu/PycharmProjects/hpvsim_v23_validation/hpvsim_rwanda/'
           'results/v2.2.6_baseline')
_AGE = ['25', '35', '45', '55']
TGT_NEG = np.array([3.13, 11.67, 14.5, 12.0])
TGT_POS = np.array([15.0, 76.0, 80.0, 30.0])
TGT_CANC = np.array([0.55, 0.17, 0.25, 0.05])
TGT_PRECIN = np.array([0.21, 0.11, 0.37, 0.31])


def _v2box(name):
    df = pd.read_csv(_V2 / f'{name}.csv').sort_values('bin')
    return df['med'].values, df['q1'].values, df['q3'].values


def main(n_seeds=12, n_agents=10000, ncpus=10, start=1960, stop=2020):
    # --- v3 calibrated run ---
    bp = sc.loadjson(_ROOT / 'results' / 'rwanda_calib' / 'best_pars.json')
    p = dict(bp['best_params']); p.setdefault('base_beta', 0.12)
    print(f'v3 calibrated (gof={bp.get("best_gof"):.3f}), {n_seeds} seeds, n={n_agents}...')
    res = sc.parallelize(_run_one, iterarg=[(s, n_agents, start, stop, p)
                                            for s in range(n_seeds)], ncpus=ncpus)
    rng = np.random.default_rng(0)
    v3 = {}
    for s, key in (('neg', 'HIV-'), ('pos', 'HIV+')):
        per_c = np.array([r['byc'][s] for r in res]); per_fy = np.array([r['byfy'][s] for r in res])
        lo, hi = _boot_rate(per_c, per_fy, rng)
        v3[key] = (_pool_byage(res, s), lo, hi)
    for gkey, lbl in (('gc', 'canc'), ('gp', 'precin')):
        pc = np.array([[r[gkey][g] for g in _GK] for r in res])
        tot = pc.sum(0); md = tot / tot.sum()
        lo, hi = _boot_dist(pc, rng)
        v3[lbl] = (md, lo, hi)

    # --- v2 published ---
    v2 = {
        'HIV-': _v2box('figS2_cancer_incidence_no_hiv'),
        'HIV+': _v2box('figS2_cancer_incidence_with_hiv'),
        'canc': _v2box('figS2_cancerous_genotype_dist'),
        'precin': _v2box('figS2_precin_genotype_dist'),
    }

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    w = 0.27
    panels = [
        (ax[0, 0], 'HIV-', _AGE, TGT_NEG, 'Cancer incidence by age, HIV- (/100k)'),
        (ax[0, 1], 'HIV+', _AGE, TGT_POS, 'Cancer incidence by age, HIV+ (/100k)'),
        (ax[1, 0], 'canc', _GK, TGT_CANC, 'Cancer genotype distribution'),
        (ax[1, 1], 'precin', _GK, TGT_PRECIN, 'Precancer genotype distribution'),
    ]
    for axi, key, labels, tgt, title in panels:
        x = np.arange(len(labels))
        v2m, v2q1, v2q3 = v2[key]
        v3m, v3lo, v3hi = v3[key]
        axi.bar(x - w, v2m, w, color='#cc6677',
                yerr=[np.maximum(v2m - v2q1, 0), np.maximum(v2q3 - v2m, 0)],
                capsize=3, error_kw=dict(lw=1), label='v2 published (med, IQR)')
        axi.bar(x, v3m, w, color='#4477aa',
                yerr=[np.maximum(v3m - v3lo, 0), np.maximum(v3hi - v3m, 0)],
                capsize=3, error_kw=dict(lw=1), label='v3 calibrated (med, 90% CI)')
        axi.bar(x + w, tgt, w, color='#999999', label='registry target')
        axi.set_xticks(x); axi.set_xticklabels(labels)
        axi.set_title(title); axi.legend(fontsize=8)
        axi.set_ylabel('per 100k' if key in ('HIV-', 'HIV+') else 'fraction')

    fig.suptitle('Rwanda: published v2 (v2.2.6) vs calibrated v3 vs registry', fontsize=14)
    fig.tight_layout()
    out = _ROOT / 'results' / 'rwanda_calib' / 'v2_v3_comparison.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    figdst = _ROOT / 'tests' / 'regression' / 'figures' / 'v2_v3_comparison.png'
    shutil.copy(out, figdst)
    print(f'Saved {out}\n      {figdst}')

    # --- results table ---
    print('\n=== RESULTS: v2 published / v3 calibrated / registry ===')
    for key, labels, tgt in [('HIV-', _AGE, TGT_NEG), ('HIV+', _AGE, TGT_POS),
                             ('canc', _GK, TGT_CANC), ('precin', _GK, TGT_PRECIN)]:
        print(f'\n{key}:')
        v2m = v2[key][0]; v3m = v3[key][0]
        for i, lab in enumerate(labels):
            print(f'  {lab:>5}:  v2={v2m[i]:>7.2f}   v3={v3m[i]:>7.2f}   target={tgt[i]:>6.2f}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(n_seeds=int(a[0]) if len(a) > 0 else 12,
         n_agents=int(a[1]) if len(a) > 1 else 10000)