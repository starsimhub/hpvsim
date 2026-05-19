"""Unit tests for hpv.Calibration and helpers."""
import numpy as np
import pandas as pd
import pytest
import sciris as sc

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


def test_cancer_by_age_factory_extract_returns_matching_schema():
    """cancer_by_age factory's extract_fn returns a DataFrame matching expected's schema."""
    edges = np.array([0., 30., 60., 100.])
    age_labels = ['0-30', '30-60', '60+']
    expected = pd.DataFrame(
        [[10, 50, 30]],
        index=pd.Index([2020.0], name='year'),
        columns=age_labels,
    )
    sim = hpv.Sim(n_agents=500, start=2019, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancers=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()

    comp = hpv.calibration.cancer_by_age(expected)
    actual = comp.extract_fn(sim)
    # Same index and columns as expected.
    assert list(actual.index) == list(expected.index)
    assert list(actual.columns) == list(expected.columns)


def test_hpv_prev_by_age_factory_extract_matches_schema():
    edges = np.array([0., 30., 60., 100.])
    age_labels = ['0-30', '30-60', '60+']
    expected = pd.DataFrame(
        [[0.05, 0.10, 0.02]],
        index=pd.Index([2020.0], name='year'),
        columns=age_labels,
    )
    sim = hpv.Sim(n_agents=500, start=2019, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          hpv_prevalence=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    comp = hpv.calibration.hpv_prev_by_age(expected)
    actual = comp.extract_fn(sim)
    assert list(actual.index) == list(expected.index)
    assert list(actual.columns) == list(expected.columns)


def test_cancer_genotype_dist_factory_extract_matches_schema():
    expected = pd.DataFrame(
        [[0.7, 0.15, 0.10, 0.05]],
        index=pd.Index([2020.0], name='year'),
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
    actual = comp.extract_fn(sim)
    assert list(actual.index) == list(expected.index)
    assert list(actual.columns) == list(expected.columns)
