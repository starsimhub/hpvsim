"""Multi-genotype Sim API tests: genotypes= sugar, auto-Connector wiring."""
import numpy as np
import pytest

import hpvsim as hpv


def test_sim_with_explicit_diseases_and_connectors():
    """Explicit diseases= + connectors= path still works (M02 surface)."""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[hpv.CrossImmunity()],
    )
    sim.run()
    assert 'hpv16' in sim.diseases


def test_sim_genotypes_sugar_single():
    """genotypes=[16] auto-instantiates one HPV + a CrossImmunity Connector."""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=[16],
    )
    sim.run()
    assert list(sim.diseases.keys()) == ['hpv16']
    # Connector auto-added and named 'crossimmunity'.
    assert any('crossimmunity' in k.lower() for k in sim.connectors.keys()), \
        f'No CrossImmunity connector found; got {list(sim.connectors.keys())}'


def test_sim_genotypes_sugar_two():
    """genotypes=[16, 18] -> two HPV modules + Connector. (Task 9 enables 4-genotype.)"""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=[16, 18],
    )
    sim.init()
    assert list(sim.diseases.keys()) == ['hpv16', 'hpv18']


def test_sim_rejects_diseases_plus_genotypes():
    """Passing both diseases= and genotypes= raises early."""
    with pytest.raises(ValueError, match='diseases.*genotypes'):
        hpv.Sim(
            n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
            diseases=[hpv.HPV(genotype='hpv16')],
            genotypes=[16],
        )


def test_sim_genotype_pars_override():
    """genotype_pars={'hpv16': {'rel_beta': 1.5}} overrides per-genotype defaults."""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=[16],
        genotype_pars={'hpv16': {'rel_beta': 1.5}},
    )
    sim.init()
    assert float(sim.diseases.hpv16.pars.rel_beta) == pytest.approx(1.5)


def test_four_genotype_sim_runs():
    """End-to-end 4-genotype Sim runs and produces per-genotype results."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=2000, dt=0.5, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
    )
    sim.run()
    for key in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        res = sim.results[key]
        # Each genotype gets cum_infections via Starsim auto-stratification.
        assert 'cum_infections' in res or 'new_infections' in res, \
            f'{key} missing infection results'


def test_genotypes_sugar_matches_explicit_diseases():
    """genotypes=[16,18] == diseases=[HPV(genotype='hpv16'), HPV(genotype='hpv18')]."""
    pars = dict(n_agents=500, location='nigeria',
                start=1990, stop=1995, dt=1.0, rand_seed=0)
    sim_a = hpv.Sim(genotypes=[16, 18], **pars)
    sim_a.run()
    sim_b = hpv.Sim(
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        connectors=[hpv.CrossImmunity()],
        **pars,
    )
    sim_b.run()
    for key in ('hpv16', 'hpv18'):
        a_inf = float(np.asarray(sim_a.results[key].new_infections).sum())
        b_inf = float(np.asarray(sim_b.results[key].new_infections).sum())
        assert a_inf == pytest.approx(b_inf), \
            f'sugar vs explicit drift for {key}: {a_inf} vs {b_inf}'


def test_aggregate_cum_infections_any():
    """AnyGenotypeAggregator.cum_infections_any counts agents ever infected with any genotype."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=1995, dt=1.0, rand_seed=0,
        genotypes=[16, 18],
    )
    sim.run()
    agg = sim.results['anygenotypeaggregator']
    any_cum = float(np.asarray(agg['cum_infections_any']).max())
    h16_cum = float(np.asarray(sim.results['hpv16'].new_infections).sum())
    h18_cum = float(np.asarray(sim.results['hpv18'].new_infections).sum())
    # Boolean-OR is at most the sum (equal iff no co-infections).
    assert any_cum > 0
    assert any_cum <= h16_cum + h18_cum + 1e-6


def test_aggregate_cum_cancers_any():
    """AnyGenotypeAggregator.cum_cancers_any sums per-genotype cum_cancers across genotypes."""
    sim = hpv.Sim(
        n_agents=2000, location='nigeria',
        start=1990, stop=2010, dt=0.5, rand_seed=0,
        genotypes=[16, 18],
    )
    sim.run()
    agg = sim.results['anygenotypeaggregator']
    any_c = float(np.asarray(agg['cum_cancers_any'])[-1])
    sum_c = sum(float(np.asarray(sim.results[k].cum_cancers)[-1])
                for k in ('hpv16', 'hpv18'))
    assert any_c == pytest.approx(sum_c, abs=1e-6)
