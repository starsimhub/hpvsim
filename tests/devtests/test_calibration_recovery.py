"""Synthetic parameter-recovery gate for hpv.Calibration.

A real 50-trial Optuna study over 20k-agent sims. The Calibration wiring itself
(build_sim routing, eval_fn arithmetic, data validation) is unit-tested cheaply
in ``tests/test_calibration.py``; this asserts that the whole loop can actually
find known parameters.
"""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv


@pytest.mark.slow
def test_parameter_recovery_smoke():
    """Synthetic parameter recovery using the new data= + compute_gof path.

    Generate a target from known calib_pars, then ask Optuna to recover
    them. Plumbing-only gate — asserts each parameter is recovered within
    25% relative error.

    Parameters chosen:
      - hpv16.cin_fn.k: CIN progression severity — strong effect on cancer
        counts.
      - hpv16.cancer_fn.transform_prob: per-step CIN→cancer probability —
        directly scales cancer incidence.
    """
    optuna = pytest.importorskip('optuna')

    edges = np.array([0., 30., 50., 70., 100.])
    snapshot_years = [2010, 2015, 2020, 2025]
    truth = {'hpv16.cin_fn.k': 0.55, 'hpv16.cancer_fn.transform_prob': 0.003}

    def make_sim():
        return hpv.Sim(n_agents=20000, start=1990, stop=2026, dt=1.0,
                       rand_seed=0, genotypes=[16],
                       analyzers=[hpv.by_age(result_args=sc.objdict(
                           cancers=sc.objdict(years=snapshot_years,
                                              edges=edges),
                       ))])

    target_sim = make_sim()
    hpv.calibration.build_sim(target_sim, calib_pars=truth)
    target_sim.run()
    target_ar = [a for a in target_sim.analyzers.values()
                 if isinstance(a, hpv.by_age)][0]
    expected = target_ar.to_dataframe(key='cancers')

    base_sim = make_sim()
    calib_pars = dict(hpv16=dict(
        cin_fn=dict(k=[0.50, 0.20, 0.90]),
        cancer_fn=dict(transform_prob=[0.002, 0.001, 0.005]),
    ))
    calib = hpv.Calibration(
        base_sim,
        calib_pars,
        data={'cancers': expected},
        total_trials=50,
        n_workers=1,
        debug=True,
        reseed=False,
        sampler=optuna.samplers.TPESampler(seed=42),
        die=True,
    )
    calib.calibrate()
    assert calib.calibrated
    best = calib.best_pars
    for name, true_val in truth.items():
        recovered = best[name]
        rel = abs(recovered - true_val) / abs(true_val)
        assert rel <= 0.25, (
            f'Parameter {name!r}: truth={true_val}, '
            f'recovered={recovered}, rel_error={rel:.3f} (>25%)'
        )
