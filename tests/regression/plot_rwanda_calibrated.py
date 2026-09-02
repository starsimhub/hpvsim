"""Plot the CALIBRATED v3 Rwanda fit vs the registry data.

Unlike plot_rwanda_calib.py (which uses the ported v2 params), this builds the
sim with the calibrated best-fit params from results/rwanda_calib/best_pars.json
(calibrate_rwanda.build_sim) and overlays the empirical targets:

  panel 1-2  cancer incidence by age, HIV-/HIV+, 2017  (model median+IQR vs registry)
  panel 3    cancer incidence over time, HIV-/HIV+      (+ 2017 registry points)
  panel 4-5  cancer / precancer genotype distribution    (model vs registry bars)
  panel 6    model-vs-target scatter (all 10 incidence points; y=x is perfect)

Run: .venv/Scripts/python.exe tests/regression/plot_rwanda_calibrated.py \
         [n_seeds=12] [n_agents=10000] [ncpus=10]
Saves: results/rwanda_calib/rwanda_calibrated_fit.png
"""
import sys
from pathlib import Path

import numpy as np
import sciris as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import starsim as ss  # noqa: E402
from hpvsim.hpv import HPV  # noqa: E402
from tests.regression import rwanda_calib as rc  # noqa: E402
from tests.regression.calibrate_rwanda import build_sim, TARGETS  # noqa: E402

_AGE_EDGES = [25, 35, 45, 55, 200]
_AGE_LABELS = ['25-35', '35-45', '45-55', '55+']
_WIN = (2010, 2019)
_SCALE = 1e5
_DT = rc._DT
_GK = rc._GENO_KEYS
CANCER_DIST = {'hpv16': 0.55, 'hpv18': 0.17, 'hi5': 0.25, 'ohr': 0.05}
PRECIN_DIST = {'hpv16': 0.21, 'hpv18': 0.11, 'hi5': 0.37, 'ohr': 0.31}


class _Probe(ss.Analyzer):
    def init_pre(self, sim):
        self.hpv = {d.name: d for d in sim.diseases.values() if isinstance(d, HPV)}
        self.hiv = sim.diseases.hiv
        super().init_pre(sim)
        n = len(sim.t.timevec); nb = len(_AGE_LABELS)
        self.yr = np.floor(np.asarray(sim.t.timevec, float)).astype(int)[:n]
        self.ca = {s: np.zeros((nb, n)) for s in ('pos', 'neg')}
        self.na = {s: np.zeros((nb, n)) for s in ('pos', 'neg')}
        self.cg = {s: np.zeros(n) for s in ('pos', 'neg')}
        self.ng = {s: np.zeros(n) for s in ('pos', 'neg')}
        self.gcanc = {g: np.zeros(n) for g in self.hpv}
        self.gprecin = {g: np.zeros(n) for g in self.hpv}

    def step(self):
        ti = self.sim.ti; ppl = self.sim.people
        fem = ppl.female.values & ppl.alive.values
        age = ppl.age.values; pos = self.hiv.infected.values
        newc = np.zeros(fem.shape, bool)
        for m in self.hpv.values():
            newc |= (m.cancerous.values & (m.ti_cancerous.values == ti))
        for s, hm in (('pos', pos), ('neg', ~pos)):
            fs = fem & hm
            self.cg[s][ti] = (newc & fs).sum(); self.ng[s][ti] = fs.sum()
            for bi, (lo, hi) in enumerate(zip(_AGE_EDGES[:-1], _AGE_EDGES[1:])):
                b = fs & (age >= lo) & (age < hi)
                self.ca[s][bi, ti] = (newc & b).sum(); self.na[s][bi, ti] = b.sum()
        for g, m in self.hpv.items():
            self.gcanc[g][ti] = (m.cancerous.values & fem).sum()
            self.gprecin[g][ti] = (m.precin.values & fem).sum()


def _run_one(seed, n_agents, start, stop, p):
    """Return RAW per-seed counts (cancers, female-years) so the central
    estimate can be POOLED across seeds (sum c / sum fy) -- the correct
    estimator for rare events, matching the calibration. Per-seed incidence
    (for the noise overlay) is derived in main()."""
    sim = build_sim(seed, n_agents, start, stop, **p)
    pr = _Probe()
    sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
    sim.run(verbose=0)
    a = [x for x in sim.analyzers.values() if isinstance(x, _Probe)][0]
    w = (a.yr >= _WIN[0]) & (a.yr <= _WIN[1])
    nb = len(_AGE_LABELS)
    byc = {s: np.array([a.ca[s][bi, w].sum() for bi in range(nb)]) for s in ('pos', 'neg')}
    byfy = {s: np.array([a.na[s][bi, w].sum() * _DT for bi in range(nb)]) for s in ('pos', 'neg')}
    years = np.unique(a.yr)
    tsc = {s: np.array([a.cg[s][a.yr == y].sum() for y in years]) for s in ('pos', 'neg')}
    tsfy = {s: np.array([a.ng[s][a.yr == y].sum() * _DT for y in years]) for s in ('pos', 'neg')}
    gc = {g: float(a.gcanc[g][w].sum()) for g in a.gcanc}
    gp = {g: float(a.gprecin[g][w].sum()) for g in a.gprecin}
    return dict(byc=byc, byfy=byfy, years=years, tsc=tsc, tsfy=tsfy, gc=gc, gp=gp)


def _pool_byage(res, s):
    """Pooled by-age incidence (sum cancers / sum female-years) -> (nb,)."""
    c = sum(r['byc'][s] for r in res); fy = sum(r['byfy'][s] for r in res)
    return np.where(fy > 0, c / fy * _SCALE, 0.0)


def _dist(pooled):
    tot = sum(pooled.values())
    return {g: (pooled[g] / tot if tot > 0 else 0.0) for g in _GK}


# Error bars = bootstrap over seeds (resample the n_seeds runs with replacement,
# recompute the POOLED estimate each time) -> CI on the pooled central value.
_NBOOT = 2000
_CI = 90        # percent (5th-95th)


def _boot_rate(per_c, per_fy, rng, nboot=_NBOOT, ci=_CI):
    """Bootstrap CI on a pooled rate (sum c / sum fy * SCALE).
    per_c, per_fy: (nseed, ...) per-seed counts / person-years."""
    n = per_c.shape[0]
    boots = np.empty((nboot,) + per_c.shape[1:])
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        c = per_c[idx].sum(0); fy = per_fy[idx].sum(0)
        boots[b] = np.where(fy > 0, c / fy * _SCALE, 0.0)
    return (np.percentile(boots, (100 - ci) / 2, axis=0),
            np.percentile(boots, 100 - (100 - ci) / 2, axis=0))


def _boot_dist(per_counts, rng, nboot=_NBOOT, ci=_CI):
    """Bootstrap CI on a pooled distribution. per_counts: (nseed, K)."""
    n = per_counts.shape[0]
    boots = np.empty((nboot, per_counts.shape[1]))
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        tot = per_counts[idx].sum(0); s = tot.sum()
        boots[b] = tot / s if s > 0 else np.zeros(per_counts.shape[1])
    return (np.percentile(boots, (100 - ci) / 2, axis=0),
            np.percentile(boots, 100 - (100 - ci) / 2, axis=0))


def main(n_seeds=12, n_agents=10000, ncpus=10, start=1960, stop=2020):
    bp = sc.loadjson(_ROOT / 'results' / 'rwanda_calib' / 'best_pars.json')
    p = dict(bp['best_params']); p.setdefault('base_beta', 0.12)
    print(f'Plotting calibrated fit (gof={bp.get("best_gof"):.3f}), '
          f'{n_seeds} seeds, n={n_agents}, {ncpus} cores.')
    res = sc.parallelize(_run_one, iterarg=[(s, n_agents, start, stop, p)
                                            for s in range(n_seeds)], ncpus=ncpus)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    x = np.arange(len(_AGE_LABELS))
    tgt_byage = {'neg': TARGETS[0:4], 'pos': TARGETS[4:8]}

    # panels 1-2: by-age incidence (bar = pooled; error bar = bootstrap 90% CI)
    for axi, s, lbl in [(ax[0, 0], 'neg', 'HIV-'), (ax[0, 1], 'pos', 'HIV+')]:
        pooled = _pool_byage(res, s)
        per_c = np.array([r['byc'][s] for r in res])
        per_fy = np.array([r['byfy'][s] for r in res])
        lo, hi = _boot_rate(per_c, per_fy, rng)
        axi.bar(x, pooled, color='#4477aa', alpha=0.75,
                yerr=[np.maximum(pooled - lo, 0), np.maximum(hi - pooled, 0)],
                capsize=5, error_kw=dict(ecolor='#22334d', lw=1.3),
                label='v3 calibrated (pooled, 90% CI)')
        axi.scatter(x, tgt_byage[s], color='k', marker='D', s=80, zorder=6,
                    label='registry 2017')
        axi.set_xticks(x); axi.set_xticklabels(_AGE_LABELS)
        axi.set_title(f'Cancer incidence by age, {lbl}'); axi.set_ylabel('per 100k')
        axi.legend(fontsize=8)

    # panel 3: time series (pooled per-year incidence + bootstrap 90% CI band)
    axi = ax[0, 2]; years = res[0]['years']
    for s, lbl, c in [('neg', 'HIV-', '#228833'), ('pos', 'HIV+', '#ee6677')]:
        per_c = np.array([r['tsc'][s] for r in res])
        per_fy = np.array([r['tsfy'][s] for r in res])
        tc = per_c.sum(0); tf = per_fy.sum(0)
        line = np.where(tf > 0, tc / tf * _SCALE, 0.0)
        lo, hi = _boot_rate(per_c, per_fy, rng)
        axi.plot(years, line, color=c, label=lbl, lw=2)
        axi.fill_between(years, lo, hi, color=c, alpha=0.18)
    axi.scatter([2017], [TARGETS[8]], color='#228833', marker='D', s=80, zorder=5)
    axi.scatter([2017], [TARGETS[9]], color='#ee6677', marker='D', s=80, zorder=5)
    axi.set_title('Cancer incidence over time (band=90% CI; D=registry, all-female denom)')
    axi.set_ylabel('per 100k'); axi.set_xlim(1990, 2020); axi.legend(fontsize=8)

    # panels 4-5: genotype distributions (model bar + bootstrap 90% CI)
    w = 0.38
    for axi, key, tgt, lbl in [(ax[1, 0], 'gc', CANCER_DIST, 'Cancer'),
                               (ax[1, 1], 'gp', PRECIN_DIST, 'Precancer (precin)')]:
        per_counts = np.array([[r[key][g] for g in _GK] for r in res])  # (nseed,4)
        tot = per_counts.sum(0); md = tot / tot.sum() if tot.sum() > 0 else tot
        lo, hi = _boot_dist(per_counts, rng)
        axi.bar(x - w / 2, md, w, color='#4477aa', label='v3 calibrated (90% CI)',
                yerr=[np.maximum(md - lo, 0), np.maximum(hi - md, 0)], capsize=4,
                error_kw=dict(ecolor='#22334d', lw=1.3))
        axi.bar(x + w / 2, [tgt[g] for g in _GK], w, color='#999999',
                label='registry 2018')
        axi.set_xticks(x); axi.set_xticklabels(_GK)
        axi.set_title(f'{lbl} genotype distribution'); axi.set_ylabel('fraction')
        axi.legend(fontsize=8)

    # panel 6: model-vs-target scatter (10 points, pooled + bootstrap 90% CI)
    axi = ax[1, 2]
    model_pts, lo_pts, hi_pts = [], [], []
    for s in ('neg', 'pos'):                                      # by-age
        per_c = np.array([r['byc'][s] for r in res])
        per_fy = np.array([r['byfy'][s] for r in res])
        model_pts += _pool_byage(res, s).tolist()
        lo, hi = _boot_rate(per_c, per_fy, rng)
        lo_pts += lo.tolist(); hi_pts += hi.tolist()
    yi = np.where(res[0]['years'] == 2017)[0][0]                  # aggregates
    for s in ('neg', 'pos'):
        per_c = np.array([r['tsc'][s][yi] for r in res])
        per_fy = np.array([r['tsfy'][s][yi] for r in res])
        tf = per_fy.sum()
        model_pts.append(per_c.sum() / tf * _SCALE if tf > 0 else 0.0)
        lo, hi = _boot_rate(per_c, per_fy, rng)
        lo_pts.append(float(lo)); hi_pts.append(float(hi))
    model_pts = np.array(model_pts); lo_pts = np.array(lo_pts); hi_pts = np.array(hi_pts)
    cols = ['#228833'] * 4 + ['#ee6677'] * 4 + ['k', 'k']
    axi.errorbar(TARGETS, model_pts,
                 yerr=[np.maximum(model_pts - lo_pts, 0), np.maximum(hi_pts - model_pts, 0)],
                 fmt='none', ecolor='#888888', lw=1, zorder=4, capsize=3)
    axi.scatter(TARGETS, model_pts, c=cols, s=55, zorder=5)
    lim = max(TARGETS.max(), hi_pts.max()) * 1.1
    axi.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='perfect (y=x)')
    axi.set_xlim(0, lim); axi.set_ylim(0, lim)
    axi.set_xlabel('registry target /100k'); axi.set_ylabel('v3 model /100k (90% CI)')
    axi.set_title('Model vs target (green=HIV-, red=HIV+, blk=agg)')
    axi.legend(fontsize=8)

    fig.suptitle(f'v3 Rwanda CALIBRATED fit vs registry (gof={bp.get("best_gof"):.2f}, '
                 f'{n_seeds} seeds, n={n_agents}; error bars = bootstrap 90% CI)', fontsize=14)
    fig.tight_layout()
    out = _ROOT / 'results' / 'rwanda_calib' / 'rwanda_calibrated_fit.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(n_seeds=int(a[0]) if len(a) > 0 else 12,
         n_agents=int(a[1]) if len(a) > 1 else 10000,
         ncpus=int(a[2]) if len(a) > 2 else 10)