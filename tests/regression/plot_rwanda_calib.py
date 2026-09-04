"""Generate the v3 Rwanda HIV-HPV calibration figure (FigS2-style).

SLATED FOR DELETION IN v3.3 (test cleanup). This is a one-off script from the
v2 -> v3 Rwanda migration, not a test: it is not collected by pytest, it has
no assertions, and several of these run a full Optuna calibration or a
multi-seed sim. They are kept for now because the v3 HIV-HPV parameterization
was derived here and the derivation is worth being able to re-read. Anything
here that should outlive 3.3 -- most likely the CalibProbe-style age-by-HIV
probes, which localizations reimplement -- needs promoting into the package
or into ``tests/`` first.

Runs N seeds of the calibrated incidence-driven Rwanda co-infection sim
(tests/regression/rwanda_calib.build_rwanda_sim) and plots the headline
validation panels against the published registry / data targets:

  1. Cancer incidence by age, 2017, HIV- women   (vs registry points)
  2. Cancer incidence by age, 2017, HIV+ women   (vs registry points)
  3. Cancer incidence time series: Total / HIV+ / HIV-  (+ 2017 target points)
  4. Adult (15-49) HIV prevalence over time        (vs rwanda/hiv_prevalence.csv)
  5. ART coverage among HIV+ over time
  6. Adult-female any-HPV prevalence by HIV status

Incidence matches v2's definition: cancers / n_females_alive * 1e5 (per year),
restricted by age bin / HIV status as needed. Bands are the inter-seed spread.

Run: .venv/Scripts/python.exe tests/regression/plot_rwanda_calib.py [n_seeds] [n_agents]
Saves: tests/regression/figures/rwanda_calib.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import starsim as ss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hpvsim as hpv  # noqa: E402
from hpvsim.hpv import HPV  # noqa: E402
from tests.regression.rwanda_calib import build_rwanda_sim  # noqa: E402

_DATA = Path(__file__).resolve().parent / 'data'
_FIGDIR = Path(__file__).resolve().parent / 'figures'
_AGE_EDGES = [25, 35, 45, 55, 200]          # 4 published bins
_AGE_LABELS = ['25-35', '35-45', '45-55', '55+']
_SCALE = 1e5
_AGE_TO_LABEL = {25: '25-35', 35: '35-45', 45: '45-55', 55: '55+'}


def _published_target():
    """Published 2017 Rwanda HIV-stratified registry data points.

    Reads the committed CSVs under ``data/`` (real-world published data, not
    model output). Returns aggregate and by-age values keyed by HIV status.
    """
    out = {'aggregate': {}, 'by_age': {}}
    for status in ('no_hiv', 'with_hiv'):
        key = f'cancer_incidence_{status}'
        agg = pd.read_csv(_DATA / f'rwanda_cancer_incidence_{status}.csv')
        out['aggregate'][key] = float(agg['value'].iloc[0])
        by_age = pd.read_csv(_DATA / f'rwanda_cancer_incidence_by_age_{status}.csv')
        out['by_age'][key] = {_AGE_TO_LABEL[int(r.age)]: float(r.value)
                              for r in by_age.itertuples()}
    return out


class CalibProbe(ss.Analyzer):
    def init_pre(self, sim):
        self.hpv = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv = sim.diseases.hiv
        super().init_pre(sim)
        n = len(sim.t.timevec)
        nb = len(_AGE_LABELS)
        self.canc_age = {'pos': np.zeros((nb, n)), 'neg': np.zeros((nb, n))}
        self.nf_age = {'pos': np.zeros((nb, n)), 'neg': np.zeros((nb, n))}
        self.canc = {'pos': np.zeros(n), 'neg': np.zeros(n)}
        self.nf = {'pos': np.zeros(n), 'neg': np.zeros(n)}
        self.n_hiv_1549 = np.zeros(n); self.n_1549 = np.zeros(n)
        self.n_art = np.zeros(n); self.n_hiv = np.zeros(n)
        self.hpv_pos = np.zeros(n); self.hpv_neg = np.zeros(n)
        self.nf_adult = {'pos': np.zeros(n), 'neg': np.zeros(n)}

    def step(self):
        ti = self.sim.ti
        ppl = self.sim.people
        # Scale-weight every count: multiscale fine agents carry scale=1/ratio
        # (build_rwanda_sim defaults ms_agent_ratio=5), so a raw .sum() would
        # over-count fine cancer agents ~5x. w=1.0 everywhere at ratio=1.
        w = ppl.scale.values
        alive = ppl.alive.values
        fem = ppl.female.values & alive
        age = ppl.age.values
        pos = self.hiv.infected.values
        fpos, fneg = fem & pos, fem & ~pos
        # new female cancers this step (current HIV status)
        newc = np.zeros(alive.shape, dtype=bool)
        anyhpv = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv:
            newc |= (m.cancerous.values & (m.ti_cancerous.values == ti))
            anyhpv |= m.infected.values
        # aggregate (all-age female)
        self.canc['pos'][ti] = (w * (newc & fpos)).sum(); self.canc['neg'][ti] = (w * (newc & fneg)).sum()
        self.nf['pos'][ti] = (w * fpos).sum(); self.nf['neg'][ti] = (w * fneg).sum()
        # by age bin
        for bi, (lo, hi) in enumerate(zip(_AGE_EDGES[:-1], _AGE_EDGES[1:])):
            ab = (age >= lo) & (age < hi)
            self.canc_age['pos'][bi, ti] = (w * (newc & fpos & ab)).sum()
            self.canc_age['neg'][bi, ti] = (w * (newc & fneg & ab)).sum()
            self.nf_age['pos'][bi, ti] = (w * (fpos & ab)).sum()
            self.nf_age['neg'][bi, ti] = (w * (fneg & ab)).sum()
        # HIV epidemic (adult 15-49)
        ad = (age >= 15) & (age < 50)
        self.n_hiv_1549[ti] = (w * (alive & ad & pos)).sum(); self.n_1549[ti] = (w * (alive & ad)).sum()
        on_art = self.hiv.on_art.values if hasattr(self.hiv, 'on_art') else np.zeros(alive.shape, bool)
        self.n_art[ti] = (w * (alive & pos & on_art)).sum(); self.n_hiv[ti] = (w * (alive & pos)).sum()
        # HPV prevalence among adult women by HIV status
        adf = fem & ad
        self.hpv_pos[ti] = (w * (adf & pos & anyhpv)).sum(); self.hpv_neg[ti] = (w * (adf & ~pos & anyhpv)).sum()
        self.nf_adult['pos'][ti] = (w * (adf & pos)).sum(); self.nf_adult['neg'][ti] = (w * (adf & ~pos)).sum()


def _years(sim, n):
    tv = np.asarray(sim.t.timevec, dtype=float)
    return np.floor(tv).astype(int)[:n]


def run(n_seeds, n_agents, start=1960, stop=2020):
    out = []
    for seed in range(n_seeds):
        pr = CalibProbe()
        sim = build_rwanda_sim(seed=seed, n_agents=n_agents, start=start, stop=stop)
        sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
        sim.run()
        a = [x for x in sim.analyzers.values() if isinstance(x, CalibProbe)][0]
        a._yr = _years(sim, len(a.canc['pos']))
        out.append(a)
        print(f'  seed {seed} done')
    return out


def _per_year(arr, yr, reduce):
    return np.array([reduce(arr[..., yr == y], axis=-1) if arr.ndim > 1 else reduce(arr[yr == y])
                     for y in np.unique(yr)]), np.unique(yr)


def _byage_window(probes, status, lo=2010, hi=2019):
    """Pooled per-seed incidence per age bin over [lo,hi] -> (nbin, nseed)."""
    nb = len(_AGE_LABELS)
    res = np.zeros((nb, len(probes)))
    for si, a in enumerate(probes):
        w = (a._yr >= lo) & (a._yr <= hi)
        nyear = len(np.unique(a._yr[w]))
        for bi in range(nb):
            c = a.canc_age[status][bi, w].sum()
            nf = a.nf_age[status][bi, w].mean()  # mean headcount
            res[bi, si] = c / (nf * nyear) * _SCALE if nf > 0 else 0.0
    return res


def _ts_incidence(probes, status):
    """Per-year incidence (all-age female) -> (nyear, nseed) + years."""
    years = np.unique(probes[0]._yr)
    res = np.zeros((len(years), len(probes)))
    for si, a in enumerate(probes):
        for yi, y in enumerate(years):
            m = a._yr == y
            c = a.canc[status][m].sum(); nf = a.nf[status][m].mean()
            res[yi, si] = c / nf * _SCALE if nf > 0 else 0.0
    return res, years


def _ts_ratio(probes, num_attr, den_attr):
    years = np.unique(probes[0]._yr)
    res = np.zeros((len(years), len(probes)))
    for si, a in enumerate(probes):
        num = getattr(a, num_attr); den = getattr(a, den_attr)
        for yi, y in enumerate(years):
            m = a._yr == y
            d = den[m].sum()
            res[yi, si] = num[m].sum() / d if d > 0 else 0.0
    return res, years


def _hpv_prev(probes, status):
    years = np.unique(probes[0]._yr)
    res = np.zeros((len(years), len(probes)))
    for si, a in enumerate(probes):
        num = a.hpv_pos if status == 'pos' else a.hpv_neg
        den = a.nf_adult[status]
        for yi, y in enumerate(years):
            m = a._yr == y
            d = den[m].sum()
            res[yi, si] = num[m].sum() / d if d > 0 else 0.0
    return res, years


def main(n_seeds=6, n_agents=15000):
    probes = run(n_seeds, n_agents)
    pub = _published_target()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    x = np.arange(len(_AGE_LABELS))

    # Panels 1-2: by-age incidence vs targets
    for ax, status, lbl in [(axes[0, 0], 'neg', 'HIV-'), (axes[0, 1], 'pos', 'HIV+')]:
        band = _byage_window(probes, status)
        med = np.median(band, axis=1); lo = np.percentile(band, 25, axis=1); hi = np.percentile(band, 75, axis=1)
        ax.bar(x, med, yerr=[med - lo, hi - med], capsize=4, color='#4477aa', alpha=0.7, label='v3 model (median, IQR)')
        tgt = pub['by_age'][f'cancer_incidence_{ "no" if status=="neg" else "with" }_hiv']
        ax.scatter(x, [tgt[l] for l in _AGE_LABELS], color='k', marker='D', s=60, zorder=5, label='registry target')
        ax.set_xticks(x); ax.set_xticklabels(_AGE_LABELS)
        ax.set_title(f'Cancer incidence by age, {lbl} women'); ax.set_ylabel('per 100k'); ax.legend(fontsize=8)

    # Panel 3: time series Total/HIV+/HIV-
    ax = axes[0, 2]
    for status, lbl, col in [('neg', 'HIV-', '#228833'), ('pos', 'HIV+', '#ee6677')]:
        ts, yrs = _ts_incidence(probes, status)
        med = np.median(ts, axis=1)
        ax.plot(yrs, med, color=col, label=lbl)
        ax.fill_between(yrs, np.percentile(ts, 25, axis=1), np.percentile(ts, 75, axis=1), color=col, alpha=0.2)
    ax.scatter([2017], [pub['aggregate']['cancer_incidence_no_hiv']], color='#228833', marker='D', s=60, zorder=5)
    ax.scatter([2017], [pub['aggregate']['cancer_incidence_with_hiv']], color='#ee6677', marker='D', s=60, zorder=5)
    ax.set_title('Cancer incidence over time (D=registry 2017)'); ax.set_ylabel('per 100k'); ax.set_xlim(1990, 2020); ax.legend(fontsize=8)

    # Panel 4: HIV prevalence 15-49 vs target
    ax = axes[1, 0]
    ts, yrs = _ts_ratio(probes, 'n_hiv_1549', 'n_1549')
    ax.plot(yrs, np.median(ts, axis=1), color='#aa3377', label='v3 model')
    ax.fill_between(yrs, np.percentile(ts, 25, axis=1), np.percentile(ts, 75, axis=1), color='#aa3377', alpha=0.2)
    hivt = pd.read_csv(_ROOT / 'hpvsim' / 'data' / 'hiv' / 'rwanda' / 'hiv_prevalence.csv')
    ax.scatter(hivt['year'], hivt['total'], color='k', marker='o', s=20, label='Rwanda data')
    ax.set_title('Adult (15-49) HIV prevalence'); ax.set_xlim(1985, 2020); ax.legend(fontsize=8)

    # Panel 5: ART coverage
    ax = axes[1, 1]
    ts, yrs = _ts_ratio(probes, 'n_art', 'n_hiv')
    ax.plot(yrs, np.median(ts, axis=1), color='#ccbb44')
    ax.fill_between(yrs, np.percentile(ts, 25, axis=1), np.percentile(ts, 75, axis=1), color='#ccbb44', alpha=0.2)
    ax.set_title('ART coverage among HIV+'); ax.set_xlim(2000, 2020); ax.set_ylim(0, 1)

    # Panel 6: HPV prevalence among adult women by HIV
    ax = axes[1, 2]
    for status, lbl, col in [('neg', 'HIV-', '#228833'), ('pos', 'HIV+', '#ee6677')]:
        ts, yrs = _hpv_prev(probes, status)
        ax.plot(yrs, np.median(ts, axis=1), color=col, label=lbl)
        ax.fill_between(yrs, np.percentile(ts, 25, axis=1), np.percentile(ts, 75, axis=1), color=col, alpha=0.2)
    ax.set_title('Adult-female any-HPV prevalence'); ax.set_xlim(1990, 2020); ax.legend(fontsize=8)

    fig.suptitle(f'v3 Rwanda HIV-HPV calibration ({n_seeds} seeds, n={n_agents}) — D=registry target', fontsize=14)
    fig.tight_layout()
    _FIGDIR.mkdir(exist_ok=True)
    outpath = _FIGDIR / 'rwanda_calib.png'
    fig.savefig(outpath, dpi=110, bbox_inches='tight')
    print(f'\nSaved {outpath}')

    # Also print the headline numbers.
    print('\n=== headline (median over seeds) ===')
    for status, lbl in [('neg', 'HIV-'), ('pos', 'HIV+')]:
        band = _byage_window(probes, status)
        agg_band = np.array([ (_ts_incidence(probes, status)[0])[ _ts_incidence(probes, status)[1]==2017 ].mean() ])
        print(f'{lbl} by-age (med): ' + ', '.join(f'{_AGE_LABELS[i]}={np.median(band[i]):.0f}' for i in range(4)))


if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    na = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    main(ns, na)
