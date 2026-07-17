"""Scoped HIV re-fit of the Rwanda HPV-HIV calibration on the GROW multiscale engine.

WHY THIS EXISTS.  The full 7-param calibration (calibrate_rwanda.py) was fit at
ms_agent_ratio=1 on the LEDGER engine. When M08 moved onto the v2-faithful GROW
engine (branch m08-rwanda-on-grow), a 12-seed verification showed:
  - the HPV natural-history side (HIV-) transfers cleanly (agg ~3.5, flat across
    ratios), so tp_scale / cin_k_scale / dur_cin_scale are KEPT FIXED here; and
  - the HIV+ side over-predicts ~2x (agg ~65 vs target 33; gof 2.05 -> 3.0),
    partly because the ledger under-resolved sparse HIV+ cancer and the original
    fit was run at ratio=1 (fit to noise).

So we re-fit ONLY the 4 HIV-differential params, at ms_agent_ratio=5 where the
grow engine resolves HIV+/cancer rare events (each HIV+ cancer-bound agent grows
5 fine agents -> ~5x lower variance per sim). The probe is SCALE-WEIGHTED (fine
agents carry scale 1/ratio); at ratio=1 it reduces to calibrate_rwanda._Probe.

FREE (4):  rel_sus_lt200, rel_sus_gt200, rel_sev_lt200, rel_sev_gt200.
FIXED:     tp_scale, cin_k_scale, dur_cin_scale, base_beta = the rwanda_calib.py
           finalized values; ms_agent_ratio = 5.

TARGETS/WEIGHTS/GOF are identical to calibrate_rwanda.py (10 registry points,
2017; by-age weight 1.0, aggregates 0.25; weighted fractional error). The 5 HIV-
points are ~constant w.r.t. the free params (a ~1.2 gof floor), so minimizing the
total is equivalent to minimizing the HIV+ error while keeping the gof directly
comparable to the verification's 3.0.

Run:  .venv-msgrow/Scripts/python.exe tests/regression/calibrate_rwanda_hiv_grow.py \
          [total_trials=80] [n_agents=10000] [n_seeds=8] [ncpus=8]
Saves: results/rwanda_calib_hiv_grow/  (best_pars.json, trials.csv, study.pkl)
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

import hpvsim as hpv                                    # noqa: E402
from hpvsim.hpv import HPV                              # noqa: E402
from hpvsim.hiv import HIV, hpv_hiv_connector           # noqa: E402
from hpvsim.cross_genotype import CrossImmunity         # noqa: E402
from hpvsim.calibration import compute_gof              # noqa: E402
from tests.regression import rwanda_calib as rc         # noqa: E402
from tests.regression import calibrate_rwanda as cal    # noqa: E402

MS_RATIO = 5  # grow multiscale ratio the re-fit runs at (HIV+ resolution)

# Fixed HPV natural-history scalars (rwanda_calib.py finalized fit; kept, they
# transferred cleanly to grow).
_FIXED = dict(tp_scale=rc._TP_SCALE, cin_k_scale=rc._CIN_K_SCALE,
              dur_cin_scale=rc._DUR_CIN_SCALE, base_beta=rc.BASE_BETA)

# Current (pre-refit) HIV effects, used to seed the search as trial 0.
_SEED_TRIAL = dict(
    rel_sus_lt200=rc.RWANDA_HIV_EFFECTS['rel_sus']['lt200'],
    rel_sus_gt200=rc.RWANDA_HIV_EFFECTS['rel_sus']['gt200'],
    rel_sev_lt200=rc.RWANDA_HIV_EFFECTS['rel_sev']['lt200'],
    rel_sev_gt200=rc.RWANDA_HIV_EFFECTS['rel_sev']['gt200'],
)


def build_sim(seed, n_agents, start, stop, rel_sus_lt200, rel_sus_gt200,
              rel_sev_lt200, rel_sev_gt200):
    """Incidence-driven Rwanda HPV-HIV sim: HPV pars fixed, HIV effects free,
    ms_agent_ratio=MS_RATIO."""
    effects = sc.dcp(rc.RWANDA_HIV_EFFECTS)
    effects['rel_sev'] = {'lt200': rel_sev_lt200, 'gt200': rel_sev_gt200}
    effects['rel_sus'] = {'lt200': rel_sus_lt200, 'gt200': rel_sus_gt200}
    connectors = [CrossImmunity(rel_sev_loc=rc.REL_SEV_LOC),
                  hpv_hiv_connector(effects=effects)]
    hiv = hpv.HIV.from_location('rwanda', beta_m2f=0.0, init_prev_data=0.0)
    interventions = [hpv.hiv_incidence_import.from_location('rwanda'),
                     hpv.hiv_art.from_location('rwanda')]
    return hpv.Sim(
        location='rwanda', rand_seed=seed, n_agents=n_agents,
        start=start, stop=stop, dt=rc._DT, ms_agent_ratio=MS_RATIO,
        genotypes=rc.GENOTYPES,
        genotype_pars=cal._genotype_pars(_FIXED['dur_cin_scale'],
                                         _FIXED['base_beta'], _FIXED['tp_scale'],
                                         _FIXED['cin_k_scale']),
        init_hpv_dist=rc.RWANDA_INIT_HPV_DIST,
        networks=[rc.make_rwanda_network()],
        connectors=connectors, diseases=[hiv], interventions=interventions,
    )


class _WProbe(ss.Analyzer):
    """Scale-weighted twin of calibrate_rwanda._Probe (fine agents count 1/ratio)."""
    def init_pre(self, sim):
        self.hpv = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv = next(d for d in sim.diseases.values() if isinstance(d, HIV))
        super().init_pre(sim)
        n = len(sim.t.timevec)
        nb = len(cal._AGE_LABELS)
        self.canc = {'pos': np.zeros((nb, n)), 'neg': np.zeros((nb, n))}
        self.nf = {'pos': np.zeros((nb, n)), 'neg': np.zeros((nb, n))}
        self.cagg = {'pos': np.zeros(n), 'neg': np.zeros(n)}
        self.nfagg = {'pos': np.zeros(n), 'neg': np.zeros(n)}
        self.yr = np.floor(np.asarray(sim.t.timevec, float)).astype(int)[:n]

    def step(self):
        ti = self.sim.ti
        ppl = self.sim.people
        w = ppl.scale.values
        al = ppl.alive.values
        fem = ppl.female.values & al
        age = ppl.age.values
        pos = self.hiv.infected.values
        newc = np.zeros(al.shape, bool)
        for m in self.hpv:
            newc |= (m.cancerous.values & (m.ti_cancerous.values == ti))
        for status, hivmask in (('pos', pos), ('neg', ~pos)):
            fs = fem & hivmask
            self.cagg[status][ti] = (w * (newc & fs)).sum()
            self.nfagg[status][ti] = (w * fs).sum()
            for bi, (lo, hi) in enumerate(zip(cal._AGE_EDGES[:-1], cal._AGE_EDGES[1:])):
                ab = (age >= lo) & (age < hi)
                self.canc[status][bi, ti] = (w * (newc & fs & ab)).sum()
                self.nf[status][bi, ti] = (w * (fs & ab)).sum()


def _run_one(seed, n_agents, start, stop, p):
    sim = build_sim(seed, n_agents, start, stop, **p)
    pr = _WProbe()
    sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
    sim.run(verbose=0)
    a = [x for x in sim.analyzers.values() if isinstance(x, _WProbe)][0]
    mask = (a.yr >= cal._WIN[0]) & (a.yr <= cal._WIN[1])
    out = dict(nyear=len(np.unique(a.yr[mask])))
    for status in ('pos', 'neg'):
        out[f'cagg_{status}'] = float(a.cagg[status][mask].sum())
        out[f'nfagg_{status}'] = float(a.nfagg[status][mask].sum())
        out[f'canc_{status}'] = a.canc[status][:, mask].sum(axis=1)
        out[f'nf_{status}'] = a.nf[status][:, mask].sum(axis=1)
    return out


def make_objective(n_agents, n_seeds, start, stop, ncpus):
    seeds = list(range(n_seeds))

    def objective(trial):
        p = dict(
            rel_sus_lt200=trial.suggest_float('rel_sus_lt200', 2.0, 8.0),
            rel_sus_gt200=trial.suggest_float('rel_sus_gt200', 1.0, 5.0),
            rel_sev_lt200=trial.suggest_float('rel_sev_lt200', 1.0, 6.0),
            rel_sev_gt200=trial.suggest_float('rel_sev_gt200', 1.0, 4.0),
        )
        argset = [(s, n_agents, start, stop, p) for s in seeds]
        probes = sc.parallelize(_run_one, iterarg=argset, ncpus=ncpus)
        model = cal._model_points(probes)
        per_point = compute_gof(cal.TARGETS, model, normalize=False,
                                use_frac=True, as_scalar='none')
        gof = float(np.sum(np.asarray(per_point) * cal.WEIGHTS))
        trial.set_user_attr('model', model.tolist())
        return gof

    return objective


def main(total_trials=80, n_agents=10000, n_seeds=8, ncpus=8,
         start=1960, stop=2020, outname='rwanda_calib_hiv_grow'):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    outdir = _ROOT / 'results' / outname
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'Rwanda HIV re-fit on GROW (ratio={MS_RATIO}): {total_trials} trials, '
          f'n_agents={n_agents}, {n_seeds} seeds/trial, {ncpus} cores '
          f'(of {sc.cpu_count()}).')
    print(f'FIXED HPV pars: {_FIXED}')
    print(f'Targets (per 100k): {cal.TARGETS.tolist()}')

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler,
                                study_name='rwanda_hiv_grow')
    study.enqueue_trial(_SEED_TRIAL)  # start from the current (pre-refit) point
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
        sc.savejson(outdir / 'best_pars.json', dict(
            best_gof=b.value, best_params=b.params,
            best_model=b.user_attrs.get('model'),
            targets=cal.TARGETS.tolist(), fixed=_FIXED, ms_agent_ratio=MS_RATIO,
            n_trials_done=done['n'], n_agents=n_agents, n_seeds=n_seeds))

    study.optimize(objective, n_trials=total_trials, callbacks=[_cb])

    df = study.trials_dataframe()
    df.to_csv(outdir / 'trials.csv', index=False)
    sc.saveobj(outdir / 'study.pkl', study)

    b = study.best_trial
    print('\n=== BEST ===')
    print(f'gof = {b.value:.4f}')
    print(f'params = {b.params}')
    labels = ([f'HIV- a{a}' for a in cal._AGE_LABELS]
              + [f'HIV+ a{a}' for a in cal._AGE_LABELS]
              + ['HIV- agg', 'HIV+ agg'])
    print(f'{"":>10}{"model":>9}{"target":>9}')
    for lab, m, t in zip(labels, b.user_attrs['model'], cal.TARGETS):
        print(f'{lab:>10}{m:>9.1f}{t:>9.1f}')
    print(f'\nSaved to {outdir}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(
        total_trials=int(a[0]) if len(a) > 0 else 80,
        n_agents=int(a[1]) if len(a) > 1 else 10000,
        n_seeds=int(a[2]) if len(a) > 2 else 8,
        ncpus=int(a[3]) if len(a) > 3 else 8,
        outname=a[4] if len(a) > 4 else 'rwanda_calib_hiv_grow',
    )
