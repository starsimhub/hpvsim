"""Calibrate the v3 Rwanda HIV-HPV sim to the independent registry targets.

WHY THIS EXISTS.  tests/regression/rwanda_calib.py ports v2.3's *published*
parameters, but those were fit against v2's multiscale-BIASED engine (which
scaled cancer down ~4x at ms_agent_ratio=100). v3's engine is multiscale-
UNBIASED (m07-multiscale-ledger), so the ported parameters overshoot badly:
a quick check gives 2015-19 cancer incidence HIV+ ~455 / HIV- ~54 /100k vs
the registry's 33 / 13.1 (RR 8.4 vs 2.5). We therefore re-calibrate v3
directly to the registry data, NOT to any v2 output.

FREE PARAMETERS (7):
  tp_scale            global multiplier on each genotype's cancer_fn
                      .transform_prob (CIN->invasive transformation hazard) --
                      the cancer-LEVEL lever.
  cin_k_scale         global multiplier on each genotype's cin_fn.k (dysplasia
                      progression slope) -- the age-SHAPE lever. Lower k slows
                      progression, pushing cancer to older ages. cancer_fn.k synced.
  dur_cin_scale       global multiplier on dur_cin (mean & std) -- secondary
                      shape/level lever.
  rel_sus_lt200       HIV->HPV susceptibility multiplier, CD4 < 200 -- HIV+
  rel_sus_gt200       HIV->HPV susceptibility multiplier, CD4 in [200,500).
                      These raise HIV+ HPV acquisition; together with rel_sev
                      they drive the HIV+/HIV- cancer RR (by-age target ~5.5).
  rel_sev_lt200       HIV->HPV severity multiplier, CD4 < 200.
  rel_sev_gt200       HIV->HPV severity multiplier, CD4 in [200,500).

FIXED: base_beta = 0.12 (v2's HPV-prevalence calibration; the multiscale bug
corrupted cancer resolution ONLY, not transmission/prevalence, so beta is
uncontaminated -- freeing it against a cancer-only objective just misuses it).

WEIGHTING: the 8 by-age points are weighted 1.0; the 2 aggregate points 0.25
(their all-female denominator is ambiguous vs the registry's adult definition).

TARGETS (10 points, 2017 registry; tests/regression/data/rwanda_cancer_*.csv):
  by-age HIV-  (25/35/45/55):  3.13, 11.67, 14.5, 12   per 100k
  by-age HIV+  (25/35/45/55):  15,   76,    80,   30   per 100k
  aggregate    HIV- / HIV+  :  13.1 / 33               per 100k

GOF: per-point fractional error (compute_gof use_frac=True), summed. Model
incidence is pooled over seeds across a 2010-2019 measurement window (matches
plot_rwanda_calib), reconstructing v2's metric cancers / female-years * 1e5.

RESOURCE CAP: runs n_seeds sims per trial in parallel across NCPUS cores
(default 10 of 20) so half the machine stays free, per the run constraint.

Run:  .venv/Scripts/python.exe tests/regression/calibrate_rwanda.py \
          [total_trials=120] [n_agents=10000] [n_seeds=10] [ncpus=10]
Saves: results/rwanda_calib/  (best_pars.json, trials.csv, study.pkl)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import sciris as sc
import starsim as ss

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hpvsim as hpv  # noqa: E402
from hpvsim.hpv import HPV  # noqa: E402
from hpvsim.hiv import HIV  # noqa: E402
from hpvsim.parameters import get_genotype_pars  # noqa: E402
from hpvsim.cross_genotype import CrossImmunity  # noqa: E402
from hpvsim.hiv import hpv_hiv_connector  # noqa: E402
from hpvsim.calibration import compute_gof  # noqa: E402
from tests.regression import rwanda_calib as rc  # noqa: E402

_AGE_EDGES = [25, 35, 45, 55, 200]
_AGE_LABELS = ['25', '35', '45', '55']         # registry bin lower-bounds
_WIN = (2010, 2019)                            # measurement window (pool cancers)
_SCALE = 1e5

# --- registry targets (per 100k), order MUST match _model_points() ----------
# 4 by-age HIV-, 4 by-age HIV+, then aggregate HIV-, HIV+.
TARGETS = np.array([
    3.13, 11.67, 14.5, 12.0,      # HIV- by age 25/35/45/55
    15.0, 76.0, 80.0, 30.0,       # HIV+ by age 25/35/45/55
    13.1, 33.0,                   # aggregate HIV- / HIV+
])
# The 8 by-age points are age-consistent (trustworthy); the 2 aggregates use an
# all-female denominator that may not match the registry's (likely adult-women)
# definition, so they are down-weighted rather than trusted equally.
WEIGHTS = np.array([1., 1., 1., 1., 1., 1., 1., 1., 0.25, 0.25])


# ---------------------------------------------------------------------------
# Sim construction with the 4 free parameters threaded in
# ---------------------------------------------------------------------------
def _genotype_pars(dur_cin_scale, base_beta, tp_scale, cin_k_scale,
                   dur_cin_std_scale=1.0):
    """rc.rwanda_genotype_pars with free dur_cin_scale, base_beta, a global
    transform_prob multiplier (tp_scale, the cancer-LEVEL lever), a global
    cin_fn.k multiplier (cin_k_scale), and dur_cin_std_scale -- an ADDITIONAL
    multiplier on the dur_cin std only. The default dur_cin std (20yr, ~4x the
    mean) gives a huge right tail (99th pctile ~63yr) that pushes ~40% of HIV-
    cancers to age 55+; shrinking the std (dur_cin_std_scale<1) concentrates
    cancer near the ~45 peak and lets the 55+ bin decline. Per-genotype base
    cin_fn.k is the Rwanda override (hi5/ohr) or the genotype default;
    cancer_fn.k (cin_integral) is kept in sync with cin_fn.k."""
    out = {}
    for key in rc._GENO_KEYS:
        gp = get_genotype_pars(key)
        rb = float(gp.rel_beta)
        d = {'beta': {'sexualnetwork': [base_beta * rb * rc._TRANSF2M,
                                        base_beta * rb * rc._TRANSM2F]}}
        if dur_cin_scale != 1.0 or dur_cin_std_scale != 1.0:
            mean, std = rc._DUR_CIN[key]
            d['dur_cin'] = ss.lognorm_ex(
                mean=ss.years(mean * dur_cin_scale),
                std=ss.years(std * dur_cin_scale * dur_cin_std_scale))
        # cin_fn.k: base = Rwanda override (hi5/ohr) or genotype default,
        # scaled globally by cin_k_scale (age-shape lever).
        base_k = rc._RWANDA_CIN_K.get(key, float(gp.cin_fn['k']))
        k = base_k * cin_k_scale
        cin = dict(gp.cin_fn)
        cin['k'] = k
        d['cin_fn'] = cin
        # cancer_fn: scale transform_prob (level); keep its duplicate k in sync.
        cf = dict(gp.cancer_fn)
        cf['transform_prob'] = cf['transform_prob'] * tp_scale
        cf['k'] = k
        d['cancer_fn'] = cf
        out[key] = d
    return out


def build_sim(seed, n_agents, start, stop, dur_cin_scale, base_beta,
              rel_sev_lt200, rel_sev_gt200, tp_scale, cin_k_scale,
              rel_sus_lt200, rel_sus_gt200, dur_cin_std_scale=1.0):
    """Incidence-driven Rwanda HPV-HIV sim with the calibration params."""
    effects = sc.dcp(rc.RWANDA_HIV_EFFECTS)
    effects['rel_sev'] = {'lt200': rel_sev_lt200, 'gt200': rel_sev_gt200}
    effects['rel_sus'] = {'lt200': rel_sus_lt200, 'gt200': rel_sus_gt200}
    connectors = [
        CrossImmunity(rel_sev_loc=rc.REL_SEV_LOC),
        hpv_hiv_connector(effects=effects),
    ]
    hiv = hpv.HIV.from_location('rwanda', beta_m2f=0.0, init_prev_data=0.0)
    interventions = [
        hpv.hiv_incidence_import.from_location('rwanda'),
        hpv.hiv_art.from_location('rwanda'),
    ]
    return hpv.Sim(
        location='rwanda', rand_seed=seed, n_agents=n_agents,
        start=start, stop=stop, dt=rc._DT,
        genotypes=rc.GENOTYPES,
        genotype_pars=_genotype_pars(dur_cin_scale, base_beta, tp_scale,
                                     cin_k_scale, dur_cin_std_scale),
        init_hpv_dist=rc.RWANDA_INIT_HPV_DIST,
        networks=[rc.make_rwanda_network()],
        connectors=connectors, diseases=[hiv], interventions=interventions,
    )


# ---------------------------------------------------------------------------
# Probe: female cancers + female-years by age bin and HIV status
# ---------------------------------------------------------------------------
class _Probe(ss.Analyzer):
    def init_pre(self, sim):
        self.hpv = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv = next(d for d in sim.diseases.values() if isinstance(d, HIV))
        super().init_pre(sim)
        n = len(sim.t.timevec)
        nb = len(_AGE_LABELS)
        self.canc = {'pos': np.zeros((nb, n)), 'neg': np.zeros((nb, n))}
        self.nf = {'pos': np.zeros((nb, n)), 'neg': np.zeros((nb, n))}
        self.cagg = {'pos': np.zeros(n), 'neg': np.zeros(n)}
        self.nfagg = {'pos': np.zeros(n), 'neg': np.zeros(n)}
        self.yr = np.floor(np.asarray(sim.t.timevec, float)).astype(int)[:n]

    def step(self):
        ti = self.sim.ti
        ppl = self.sim.people
        al = ppl.alive.values
        fem = ppl.female.values & al
        age = ppl.age.values
        pos = self.hiv.infected.values
        newc = np.zeros(al.shape, bool)
        for m in self.hpv:
            newc |= (m.cancerous.values & (m.ti_cancerous.values == ti))
        for status, hivmask in (('pos', pos), ('neg', ~pos)):
            fs = fem & hivmask
            self.cagg[status][ti] = (newc & fs).sum()
            self.nfagg[status][ti] = fs.sum()
            for bi, (lo, hi) in enumerate(zip(_AGE_EDGES[:-1], _AGE_EDGES[1:])):
                ab = (age >= lo) & (age < hi)
                self.canc[status][bi, ti] = (newc & fs & ab).sum()
                self.nf[status][bi, ti] = (fs & ab).sum()


def _run_one(seed, n_agents, start, stop, p):
    """Worker: build+run one sim, return the probe's pooled-window arrays."""
    sim = build_sim(seed, n_agents, start, stop, **p)
    pr = _Probe()
    sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
    sim.run(verbose=0)
    a = [x for x in sim.analyzers.values() if isinstance(x, _Probe)][0]
    w = (a.yr >= _WIN[0]) & (a.yr <= _WIN[1])
    nyear = len(np.unique(a.yr[w]))
    # Return summed cancers and summed female-step-counts per bin/status.
    out = dict(nyear=nyear)
    for status in ('pos', 'neg'):
        out[f'cagg_{status}'] = float(a.cagg[status][w].sum())
        out[f'nfagg_{status}'] = float(a.nfagg[status][w].sum())
        out[f'canc_{status}'] = a.canc[status][:, w].sum(axis=1)        # (nb,)
        out[f'nf_{status}'] = a.nf[status][:, w].sum(axis=1)            # (nb,)
    return out


def _model_points(probes):
    """Pool probes -> 10 incidence rates in TARGETS order. dt cancels in the
    ratio (cancers per female-year), so we use summed step-counts * dt for
    female-years; the per-year mean = sum/nyear, female-years = mean*nyear, so
    incidence = sum_cancers / (sum_nf*dt/nyear * nyear) = sum_c/(sum_nf*dt)."""
    dt = rc._DT
    nyear = probes[0]['nyear']
    pts = []
    for status in ('neg', 'pos'):                 # by-age first
        c = sum(p[f'canc_{status}'] for p in probes)       # (nb,)
        nf = sum(p[f'nf_{status}'] for p in probes)        # (nb,) step-counts
        fy = nf * dt                                        # female-years
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(fy > 0, c / fy * _SCALE, 0.0)
        pts.extend(rate.tolist())
    for status in ('neg', 'pos'):                 # aggregate
        c = sum(p[f'cagg_{status}'] for p in probes)
        nf = sum(p[f'nfagg_{status}'] for p in probes)
        fy = nf * dt
        pts.append(c / fy * _SCALE if fy > 0 else 0.0)
    return np.array(pts)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def make_objective(n_agents, n_seeds, start, stop, ncpus):
    seeds = list(range(n_seeds))

    def objective(trial):
        p = dict(
            tp_scale=trial.suggest_float('tp_scale', 0.10, 1.0),
            cin_k_scale=trial.suggest_float('cin_k_scale', 0.30, 1.2),
            dur_cin_scale=trial.suggest_float('dur_cin_scale', 0.7, 1.3),
            # NOTE: a dur_cin_std_scale lever (shrink the dwell tail) was tested
            # to fix the HIV- 55+ overshoot but REJECTED -- an isolated sweep
            # showed shrinking the std just reshuffles cancer to a45/a25 without
            # fixing a55 (old-age cancer is driven by ongoing HPV acquisition at
            # older ages / the network, not the dwell tail). The build_sim /
            # _genotype_pars plumbing for it remains (defaults to 1.0, inert).
            # HIV+/HIV- differential levers (drive the by-age RR ~5.5 target).
            rel_sus_lt200=trial.suggest_float('rel_sus_lt200', 2.0, 8.0),
            rel_sus_gt200=trial.suggest_float('rel_sus_gt200', 1.0, 5.0),
            rel_sev_lt200=trial.suggest_float('rel_sev_lt200', 1.0, 6.0),
            rel_sev_gt200=trial.suggest_float('rel_sev_gt200', 1.0, 4.0),
            # FIXED: v2-calibrated to HPV prevalence data, which the multiscale
            # bug never touched (it corrupted cancer resolution only). Freeing
            # it in a cancer-only objective just lets it drift / be misused.
            base_beta=0.12,
        )
        argset = [(s, n_agents, start, stop, p) for s in seeds]
        probes = sc.parallelize(_run_one, iterarg=argset, ncpus=ncpus)
        model = _model_points(probes)
        # per-point fractional error, weighted (by-age points trusted over the
        # denominator-ambiguous aggregates), then summed.
        per_point = compute_gof(TARGETS, model, normalize=False, use_frac=True,
                                as_scalar='none')
        gof = float(np.sum(np.asarray(per_point) * WEIGHTS))
        trial.set_user_attr('model', model.tolist())
        return float(gof)

    return objective


def main(total_trials=120, n_agents=10000, n_seeds=10, ncpus=10,
         start=1960, stop=2020, outname='rwanda_calib'):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    outdir = _ROOT / 'results' / outname
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'Rwanda v3 calibration: {total_trials} trials, n_agents={n_agents}, '
          f'{n_seeds} seeds/trial, {ncpus} cores (of {sc.cpu_count()}).')
    print(f'Targets (per 100k): {TARGETS.tolist()}')

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler,
                                study_name='rwanda_v3')
    objective = make_objective(n_agents, n_seeds, start, stop, ncpus)

    t0 = time.time()
    done = {'n': 0}

    def _cb(study, trial):
        done['n'] += 1
        el = time.time() - t0
        b = study.best_trial
        print(f'[{done["n"]:>3}/{total_trials}] trial {trial.number} '
              f'gof={trial.value:.3f}  best={b.value:.3f} '
              f'{ {k: round(v, 3) for k, v in b.params.items()} }  '
              f'({el/done["n"]:.0f}s/trial, {el/60:.1f}min elapsed)', flush=True)
        # Persist incrementally so a long run is inspectable / resumable.
        sc.savejson(outdir / 'best_pars.json', dict(
            best_gof=b.value, best_params=b.params,
            best_model=b.user_attrs.get('model'),
            targets=TARGETS.tolist(), n_trials_done=done['n'],
            n_agents=n_agents, n_seeds=n_seeds))

    study.optimize(objective, n_trials=total_trials, callbacks=[_cb])

    df = study.trials_dataframe()
    df.to_csv(outdir / 'trials.csv', index=False)
    sc.saveobj(outdir / 'study.pkl', study)

    b = study.best_trial
    print('\n=== BEST ===')
    print(f'gof = {b.value:.4f}')
    print(f'params = {b.params}')
    print(f'{"":>14}{"model":>9}{"target":>9}')
    labels = ([f'HIV- a{a}' for a in _AGE_LABELS]
              + [f'HIV+ a{a}' for a in _AGE_LABELS]
              + ['HIV- agg', 'HIV+ agg'])
    for lab, m, t in zip(labels, b.user_attrs['model'], TARGETS):
        print(f'{lab:>14}{m:>9.1f}{t:>9.1f}')
    print(f'\nSaved to {outdir}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(
        total_trials=int(a[0]) if len(a) > 0 else 120,
        n_agents=int(a[1]) if len(a) > 1 else 10000,
        n_seeds=int(a[2]) if len(a) > 2 else 10,
        ncpus=int(a[3]) if len(a) > 3 else 10,
        outname=a[4] if len(a) > 4 else 'rwanda_calib',
    )