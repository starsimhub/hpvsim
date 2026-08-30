"""Per-genotype refinement of the Rwanda calibration.

Builds on calibrate_rwanda.py (incidence-only, gof 1.84) by adding the empirical
genotype-distribution targets and per-genotype levers, to tighten the cancer/
precancer genotype MIX (hi5 was under-, ohr over-represented) WITHOUT breaking
the now-good total level / age-shape / HIV RR fit.

FREE PARAMETERS (12):
  global level/shape/HIV (6):
    tp_scale         global transform_prob multiplier (LEVEL anchor, on hpv16).
    cin_k_scale      global cin_fn.k multiplier (age shape).
    rel_sus_lt200/gt200, rel_sev_lt200/gt200  HIV->HPV effects (RR).
  per-genotype (6), hpv16 = reference (mult 1.0):
    rb_18, rb_hi5, rb_ohr     rel_beta multipliers -- genotype PRESENCE
                              (hi5 needs more circulation).
    tp_rel_18, tp_rel_hi5, tp_rel_ohr   transform_prob multipliers RELATIVE to
                              hpv16 -- precancer->cancer progression per genotype
                              (ohr progresses too readily -> tp_rel_ohr < 1).
FIXED: base_beta=0.12, dur_cin_scale=1.043 (converged neutral value).

TARGETS:
  10 incidence points (by-age + aggregate, HIV+/-, 2017) -- weighted frac error
     (by-age 1.0, aggregate 0.25), as in calibrate_rwanda.
  + cancer genotype dist (4):  16/18/hi5/ohr = .55/.17/.25/.05  (rwanda_cancer_types)
  + precancer (precin) dist (4): 16/18/hi5/ohr = .21/.11/.37/.31 (rwanda_precin_types;
     v3 `precin` state, per the validation mapping)
  genotype terms scored as absolute error on the fractions, scaled by W_GENO.

Run: .venv/Scripts/python.exe tests/regression/calibrate_rwanda_genotype.py \
         [total_trials=180] [n_agents=10000] [n_seeds=10] [ncpus=10]
Saves: results/rwanda_calib_geno/  (best_pars.json, trials.csv, study.pkl)
"""
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
_AGE_LABELS = ['25', '35', '45', '55']
_WIN = (2010, 2019)
_SCALE = 1e5
_DT = rc._DT
_GK = rc._GENO_KEYS                              # ['hpv16','hpv18','hi5','ohr']
DUR_CIN_FIXED = 1.043

# incidence targets + weights (as in calibrate_rwanda)
TARGETS = np.array([3.13, 11.67, 14.5, 12.0, 15.0, 76.0, 80.0, 30.0, 13.1, 33.0])
WEIGHTS = np.array([1., 1., 1., 1., 1., 1., 1., 1., 0.25, 0.25])
# genotype-distribution targets (in _GK order)
CANCER_DIST = np.array([0.55, 0.17, 0.25, 0.05])
PRECIN_DIST = np.array([0.21, 0.11, 0.37, 0.31])
W_GENO = 2.0                                     # weight on the genotype abs-error sum


def _genotype_pars(cin_k_scale, tp_scale, tp_rel, rb_mult):
    """Per-genotype pars with per-genotype rel_beta (rb_mult) and transform_prob
    (tp_scale * tp_rel). tp_rel/rb_mult are dicts; missing key (hpv16) -> 1.0."""
    out = {}
    for key in _GK:
        gp = get_genotype_pars(key)
        rb = float(gp.rel_beta) * rb_mult.get(key, 1.0)
        d = {'beta': {'sexualnetwork': [rc.BASE_BETA * rb * rc._TRANSF2M,
                                        rc.BASE_BETA * rb * rc._TRANSM2F]}}
        mean, std = rc._DUR_CIN[key]
        d['dur_cin'] = ss.lognorm_ex(mean=ss.years(mean * DUR_CIN_FIXED),
                                     std=ss.years(std * DUR_CIN_FIXED))
        base_k = rc._RWANDA_CIN_K.get(key, float(gp.cin_fn['k']))
        k = base_k * cin_k_scale
        cin = dict(gp.cin_fn); cin['k'] = k
        d['cin_fn'] = cin
        cf = dict(gp.cancer_fn)
        cf['transform_prob'] = cf['transform_prob'] * tp_scale * tp_rel.get(key, 1.0)
        cf['k'] = k
        d['cancer_fn'] = cf
        out[key] = d
    return out


def build_sim(seed, n_agents, start, stop, p):
    effects = sc.dcp(rc.RWANDA_HIV_EFFECTS)
    effects['rel_sev'] = {'lt200': p['rel_sev_lt200'], 'gt200': p['rel_sev_gt200']}
    effects['rel_sus'] = {'lt200': p['rel_sus_lt200'], 'gt200': p['rel_sus_gt200']}
    gp = _genotype_pars(
        p['cin_k_scale'], p['tp_scale'],
        {'hpv18': p['tp_rel_18'], 'hi5': p['tp_rel_hi5'], 'ohr': p['tp_rel_ohr']},
        {'hpv18': p['rb_18'], 'hi5': p['rb_hi5'], 'ohr': p['rb_ohr']})
    connectors = [CrossImmunity(rel_sev_loc=rc.REL_SEV_LOC),
                  hpv_hiv_connector(pars=rc.effects_to_connector_pars(effects))]
    hiv = hpv.HIV.from_location('rwanda', beta_m2f=0.0, init_prev_data=0.0)
    interventions = [hpv.hiv_incidence.from_location('rwanda'),
                     hpv.hiv_art.from_location('rwanda')]
    return hpv.Sim(location='rwanda', rand_seed=seed, n_agents=n_agents,
                   start=start, stop=stop, dt=_DT, genotypes=rc.GENOTYPES,
                   genotype_pars=gp, init_hpv_dist=rc.RWANDA_INIT_HPV_DIST,
                   networks=[rc.make_rwanda_network()], connectors=connectors,
                   diseases=[hiv], interventions=interventions)


class _Probe(ss.Analyzer):
    def init_pre(self, sim):
        self.hpv = {d.name: d for d in sim.diseases.values() if isinstance(d, HPV)}
        self.hiv = next(d for d in sim.diseases.values() if isinstance(d, HIV))
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
    sim = build_sim(seed, n_agents, start, stop, p)
    pr = _Probe()
    sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
    sim.run(verbose=0)
    a = [x for x in sim.analyzers.values() if isinstance(x, _Probe)][0]
    w = (a.yr >= _WIN[0]) & (a.yr <= _WIN[1])
    nb = len(_AGE_LABELS)
    out = {}
    for s in ('pos', 'neg'):
        out[f'ca_{s}'] = np.array([a.ca[s][bi, w].sum() for bi in range(nb)])
        out[f'na_{s}'] = np.array([a.na[s][bi, w].sum() for bi in range(nb)])
        out[f'cg_{s}'] = float(a.cg[s][w].sum())
        out[f'ng_{s}'] = float(a.ng[s][w].sum())
    out['gcanc'] = np.array([a.gcanc[g][w].sum() for g in _GK])
    out['gprecin'] = np.array([a.gprecin[g][w].sum() for g in _GK])
    return out


def _pool(res):
    """Pool across seeds -> (incidence[10], cancer_dist[4], precin_dist[4])."""
    inc = []
    for s in ('neg', 'pos'):                          # by-age
        c = sum(r[f'ca_{s}'] for r in res); fy = sum(r[f'na_{s}'] for r in res) * _DT
        inc.extend(np.where(fy > 0, c / fy * _SCALE, 0.0).tolist())
    for s in ('neg', 'pos'):                          # aggregate
        c = sum(r[f'cg_{s}'] for r in res); fy = sum(r[f'ng_{s}'] for r in res) * _DT
        inc.append(c / fy * _SCALE if fy > 0 else 0.0)
    gc = sum(r['gcanc'] for r in res); gp = sum(r['gprecin'] for r in res)
    cdist = gc / gc.sum() if gc.sum() > 0 else np.zeros(4)
    pdist = gp / gp.sum() if gp.sum() > 0 else np.zeros(4)
    return np.array(inc), cdist, pdist


def make_objective(n_agents, n_seeds, start, stop, ncpus):
    seeds = list(range(n_seeds))

    def objective(trial):
        p = dict(
            tp_scale=trial.suggest_float('tp_scale', 0.05, 0.5),
            cin_k_scale=trial.suggest_float('cin_k_scale', 0.3, 1.2),
            rel_sus_lt200=trial.suggest_float('rel_sus_lt200', 2.0, 8.0),
            rel_sus_gt200=trial.suggest_float('rel_sus_gt200', 1.0, 5.0),
            rel_sev_lt200=trial.suggest_float('rel_sev_lt200', 1.0, 6.0),
            rel_sev_gt200=trial.suggest_float('rel_sev_gt200', 1.0, 4.0),
            rb_18=trial.suggest_float('rb_18', 0.4, 3.0),
            rb_hi5=trial.suggest_float('rb_hi5', 0.4, 3.0),
            rb_ohr=trial.suggest_float('rb_ohr', 0.4, 3.0),
            tp_rel_18=trial.suggest_float('tp_rel_18', 0.3, 3.0),
            tp_rel_hi5=trial.suggest_float('tp_rel_hi5', 0.3, 3.0),
            tp_rel_ohr=trial.suggest_float('tp_rel_ohr', 0.3, 3.0),
        )
        res = sc.parallelize(_run_one,
                             iterarg=[(s, n_agents, start, stop, p) for s in seeds],
                             ncpus=ncpus)
        inc, cdist, pdist = _pool(res)
        inc_per = compute_gof(TARGETS, inc, normalize=False, use_frac=True,
                              as_scalar='none')
        inc_gof = float(np.sum(np.asarray(inc_per) * WEIGHTS))
        geno_gof = float(np.sum(np.abs(cdist - CANCER_DIST))
                         + np.sum(np.abs(pdist - PRECIN_DIST)))
        trial.set_user_attr('inc', inc.tolist())
        trial.set_user_attr('cdist', cdist.tolist())
        trial.set_user_attr('pdist', pdist.tolist())
        trial.set_user_attr('inc_gof', inc_gof)
        trial.set_user_attr('geno_gof', geno_gof)
        return inc_gof + W_GENO * geno_gof

    return objective


def main(total_trials=180, n_agents=10000, n_seeds=10, ncpus=10,
         start=1960, stop=2020):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    outdir = _ROOT / 'results' / 'rwanda_calib_geno'
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Rwanda per-genotype refinement: {total_trials} trials, n={n_agents}, '
          f'{n_seeds} seeds, {ncpus} cores. W_GENO={W_GENO}.')

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42),
                                study_name='rwanda_geno')
    objective = make_objective(n_agents, n_seeds, start, stop, ncpus)
    t0 = time.time(); done = {'n': 0}

    def _cb(study, trial):
        done['n'] += 1; b = study.best_trial; el = time.time() - t0
        print(f'[{done["n"]:>3}/{total_trials}] gof={trial.value:.3f} '
              f'best={b.value:.3f} (inc={b.user_attrs["inc_gof"]:.2f} '
              f'geno={b.user_attrs["geno_gof"]:.3f}) '
              f'({el/done["n"]:.0f}s/trial, {el/60:.1f}min)', flush=True)
        sc.savejson(outdir / 'best_pars.json', dict(
            best_gof=b.value, best_params=b.params,
            inc=b.user_attrs['inc'], cdist=b.user_attrs['cdist'],
            pdist=b.user_attrs['pdist'], inc_gof=b.user_attrs['inc_gof'],
            geno_gof=b.user_attrs['geno_gof'], n_trials_done=done['n']))

    study.optimize(objective, n_trials=total_trials, callbacks=[_cb])
    study.trials_dataframe().to_csv(outdir / 'trials.csv', index=False)
    sc.saveobj(outdir / 'study.pkl', study)

    b = study.best_trial
    print('\n=== BEST ===')
    print(f'gof={b.value:.3f}  inc={b.user_attrs["inc_gof"]:.3f}  '
          f'geno={b.user_attrs["geno_gof"]:.3f}\nparams={b.params}')
    print('\ncancer dist  model', [round(x, 3) for x in b.user_attrs['cdist']],
          'target', CANCER_DIST.tolist())
    print('precin dist  model', [round(x, 3) for x in b.user_attrs['pdist']],
          'target', PRECIN_DIST.tolist())
    print(f'Saved to {outdir}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(total_trials=int(a[0]) if len(a) > 0 else 180,
         n_agents=int(a[1]) if len(a) > 1 else 10000,
         n_seeds=int(a[2]) if len(a) > 2 else 10,
         ncpus=int(a[3]) if len(a) > 3 else 10)