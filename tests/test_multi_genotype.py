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