"""Unit tests for hpv.Calibration and helpers."""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss

import hpvsim as hpv


def test_calibration_importable():
    """hpv.Calibration exists at top level and is an ss.Calibration."""
    import starsim as ss
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    calib_pars = dict(beta=dict(low=0.10, high=0.30, guess=0.20))
    calib = hpv.Calibration(sim, calib_pars, total_trials=2, debug=True)
    assert isinstance(calib, ss.Calibration)


def test_build_sim_routes_top_level_pars():
    """A bare-name calib_pars key writes into sim.pars."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    sim.init()
    trial_pars = {'beta': 0.25}
    out = hpv.calibration.build_sim(sim, calib_pars=trial_pars)
    assert out.pars.beta == 0.25


def test_build_sim_routes_per_genotype_pars():
    """A '<genotype>.<...>' calib_pars key writes into that disease's pars."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0,
                  genotypes=[16, 18, 'hi5', 'ohr'])
    sim.init()
    trial_pars = {'hpv16.cin_fn.k': 0.77}
    out = hpv.calibration.build_sim(sim, calib_pars=trial_pars)
    assert out.diseases.hpv16.pars.cin_fn['k'] == 0.77


def test_build_sim_routes_cross_immunity():
    """A 'cross_immunity.<matrix>.<tgt>.<src>' key writes into the connector matrix."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0,
                  genotypes=[16, 18, 'hi5', 'ohr'])
    sim.init()
    trial_pars = {'cross_immunity.cross_imm_sus.hpv16.hpv18': 0.42}
    out = hpv.calibration.build_sim(sim, calib_pars=trial_pars)
    conn = [c for c in out.connectors.values()
            if isinstance(c, hpv.CrossImmunity)][0]
    idx = {m.name: i for i, m in enumerate(conn.hpv_modules)}
    assert conn.cross_imm_sus[idx['hpv16'], idx['hpv18']] == 0.42


def test_build_sim_raises_on_unknown_key():
    """An unrecognized calib_pars key raises ValueError."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    sim.init()
    with pytest.raises(ValueError, match='unrecognized'):
        hpv.calibration.build_sim(sim, calib_pars={'notapar.foo': 1.0})


def test_build_sim_does_not_mutate_base():
    """build_sim mutates the passed sim — ss.Calibration's dcp is the
    no-mutate guarantee. This test confirms we mutate the *passed* sim, not
    something else, and that the test verifies the contract by passing dcp'd
    copies."""
    sim_base = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    sim_base.init()
    original_seed = sim_base.pars.rand_seed
    # ss.Calibration would dcp first; we simulate that here.
    sim_copy = sc.dcp(sim_base)
    hpv.calibration.build_sim(sim_copy, calib_pars={'rand_seed': 999})
    assert sim_base.pars.rand_seed == original_seed
    assert sim_copy.pars.rand_seed == 999


def test_cancer_by_age_factory_returns_one_normal_component_per_age_bin():
    """cancer_by_age returns a list of ss.Normal components, one per age bin.

    Each component's extract_fn returns a 't'-indexed DataFrame with a
    single 'x' column carrying that age bin's counts — the shape that
    ss.Normal.compute_nll consumes.
    """
    edges = np.array([0., 30., 60., 100.])
    age_labels = ['0-30', '30-60', '60+']
    expected = pd.DataFrame(
        [[10, 50, 30]],
        index=pd.Index([2020.0], name='t'),
        columns=age_labels,
    )
    sim = hpv.Sim(n_agents=500, start=2019, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancers=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    ar = sim.analyzers['ageresults']
    full = ar.to_dataframe(key='cancers')

    components = hpv.calibration.cancer_by_age(expected)
    assert len(components) == len(age_labels)
    for comp, age_bin in zip(components, age_labels):
        assert isinstance(comp, ss.Normal)
        assert comp.name == f'cancer_by_age:{age_bin}'
        actual = comp.extract_fn(sim)
        assert actual.index.name == 't'
        assert list(actual.columns) == ['x']
        # The 'x' column carries that age bin's counts from AgeResults.
        assert (actual['x'].values == full[age_bin].values).all()


def test_hpv_prev_by_age_factory_with_counts_returns_betabinomial_per_age_bin():
    """hpv_prev_by_age(expected_x, expected_n) returns ss.BetaBinomial per bin.

    Each component's extract_fn returns a 't'-indexed DataFrame with columns
    ['x', 'n'] for that age bin's positives and totals — the format
    ss.BetaBinomial.compute_nll consumes.
    """
    edges = np.array([0., 30., 60., 100.])
    age_labels = ['0-30', '30-60', '60+']
    expected_x = pd.DataFrame(
        [[5, 20, 8]],
        index=pd.Index([2020.0], name='t'),
        columns=age_labels,
    )
    expected_n = pd.DataFrame(
        [[100, 200, 400]],
        index=pd.Index([2020.0], name='t'),
        columns=age_labels,
    )
    sim = hpv.Sim(n_agents=500, start=2019, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          hpv_prevalence=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    components = hpv.calibration.hpv_prev_by_age(expected_x, expected_n)
    assert len(components) == len(age_labels)
    for comp, age_bin in zip(components, age_labels):
        assert isinstance(comp, ss.BetaBinomial)
        assert comp.name == f'hpv_prev_by_age:{age_bin}'
        actual = comp.extract_fn(sim)
        assert actual.index.name == 't'
        assert list(actual.columns) == ['x', 'n']
        # n should equal the alive count per bin (positive integer).
        assert (actual['n'] > 0).all()
        assert (actual['x'] <= actual['n']).all()


def test_hpv_prev_by_age_factory_returns_one_normal_component_per_age_bin():
    """hpv_prev_by_age returns a list of ss.Normal components, one per age bin."""
    edges = np.array([0., 30., 60., 100.])
    age_labels = ['0-30', '30-60', '60+']
    expected = pd.DataFrame(
        [[0.05, 0.10, 0.02]],
        index=pd.Index([2020.0], name='t'),
        columns=age_labels,
    )
    sim = hpv.Sim(n_agents=500, start=2019, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          hpv_prevalence=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    components = hpv.calibration.hpv_prev_by_age(expected)
    assert len(components) == len(age_labels)
    for comp, age_bin in zip(components, age_labels):
        assert isinstance(comp, ss.Normal)
        assert comp.name == f'hpv_prev_by_age:{age_bin}'
        actual = comp.extract_fn(sim)
        assert actual.index.name == 't'
        assert list(actual.columns) == ['x']


def test_cancer_genotype_dist_factory_returns_dirichlet_with_x_prefixed_columns():
    """cancer_genotype_dist accepts genotype-name columns and emits a
    DirichletMultinomial component with the x_<genotype> column schema that
    ss.DirichletMultinomial.compute_nll consumes. extract_fn returns raw
    counts (not normalized proportions)."""
    expected = pd.DataFrame(
        # Raw counts, not proportions — DirichletMultinomial uses them as
        # multinomial trial outcomes.
        [[70, 15, 10, 5]],
        index=pd.Index([2020.0], name='t'),
        columns=['hpv16', 'hpv18', 'hi5', 'ohr'],
    )
    edges = np.array([0., 100.])
    sim = hpv.Sim(n_agents=500, start=2019, stop=2021, dt=1.0,
                  rand_seed=0,
                  genotypes=[16, 18, 'hi5', 'ohr'],
                  analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancerous_genotype_dist=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    comp = hpv.calibration.cancer_genotype_dist(expected)
    assert isinstance(comp, ss.DirichletMultinomial)
    # Component's expected has x_-prefixed columns.
    expected_cols = ['x_hpv16', 'x_hpv18', 'x_hi5', 'x_ohr']
    assert list(comp.expected.columns) == expected_cols
    actual = comp.extract_fn(sim)
    assert list(actual.index) == list(expected.index)
    assert list(actual.columns) == expected_cols
    # Values are raw counts (non-negative), not proportions.
    assert (actual.values >= 0).all()


@pytest.mark.slow
def test_parameter_recovery_smoke():
    """Synthetic parameter recovery: calibrate to a target generated from
    known calib_pars and assert the best trial recovers each within 25%.

    This is a plumbing gate, not a calibration-quality gate. 50 trials with
    a deterministic Optuna sampler seed and 4 snapshot years (2010, 2015,
    2020, 2025) provide enough signal per ss.Normal component to recover
    both parameters within 25% relative error.

    Parameters chosen:
      - hpv16.cin_fn.k: CIN progression severity — strong monotonic effect on
        cancer counts.
      - hpv16.cancer_fn.transform_prob: per-step CIN→cancer probability —
        directly scales cancer incidence.
    Both are scalars and clearly differentiate simulation outcomes at n=20000.
    """
    optuna = pytest.importorskip('optuna')

    # ----- Generate target -----
    edges = np.array([0., 30., 50., 70., 100.])
    # Multi-year snapshots: each ss.Normal per-bin component sees 4
    # timepoints, giving the Optuna sampler clearer signal across the
    # parameter space.
    snapshot_years = [2010, 2015, 2020, 2025]
    # Two scalar parameters with strong, monotonic signal in cancer counts.
    truth = {'hpv16.cin_fn.k': 0.55, 'hpv16.cancer_fn.transform_prob': 0.003}

    def make_sim():
        # Single genotype (hpv16 only) for faster runs; n_agents=20000 for
        # enough cancer events per bin per snapshot year to provide signal.
        return hpv.Sim(n_agents=20000, start=1990, stop=2026, dt=1.0,
                       rand_seed=0,
                       genotypes=[16],
                       analyzers=[hpv.AgeResults(
                           result_args=sc.objdict(
                               cancers=sc.objdict(years=snapshot_years,
                                                  edges=edges),
                           ),
                       )])

    target_sim = make_sim()
    hpv.calibration.build_sim(target_sim, calib_pars=truth)
    target_sim.run()
    target_ar = [a for a in target_sim.analyzers.values()
                 if isinstance(a, hpv.AgeResults)][0]
    expected = target_ar.to_dataframe(key='cancers')

    # ----- Calibrate -----
    # Use the cancer_by_age factory directly. It returns one ss.Normal
    # component per age bin; Starsim's Calibration sums per-component nll
    # across the four bins to drive Optuna toward the truth values.
    components = hpv.calibration.cancer_by_age(expected)

    base_sim = make_sim()
    calib_pars = {
        'hpv16.cin_fn.k':                 dict(low=0.20, high=0.90, guess=0.50),
        'hpv16.cancer_fn.transform_prob': dict(low=0.001, high=0.005, guess=0.002),
    }
    # reseed=False: keep all trials at rand_seed=0 (same as target_sim)
    # so stochasticity is controlled and trials are comparable to the target.
    calib = hpv.Calibration(
        base_sim,
        calib_pars,
        components=components,
        total_trials=50,
        n_workers=1,
        debug=True,
        reseed=False,
        sampler=optuna.samplers.TPESampler(seed=42),
        die=True,
    )
    calib.calibrate()
    assert calib.calibrated
    best = calib.best_pars   # sc.objdict of best parameter values
    # Recover each parameter within ±25% relative error.
    for name, true_val in truth.items():
        recovered = best[name]
        rel = abs(recovered - true_val) / abs(true_val)
        assert rel <= 0.25, (
            f'Parameter {name!r}: truth={true_val}, '
            f'recovered={recovered}, rel_error={rel:.3f} (>25%)'
        )
