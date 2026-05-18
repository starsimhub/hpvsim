"""Multi-genotype Sim API tests: genotypes= sugar, auto-Connector wiring."""
import numpy as np
import pytest

import hpvsim as hpv


def test_sim_with_explicit_diseases_and_connectors():
    """Explicit diseases= path still works; auto-CrossImmunity is appended."""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
    )
    sim.run()
    assert 'hpv16' in sim.diseases
    assert 'crossimmunity' in sim.connectors


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
    """genotypes=[16,18] with init_seeding='independent' == diseases=[HPV(...)] explicit.

    The 'independent' mode draws from each genotype's own init_prev curve,
    matching the per-HPV default; 'exclusive' (the default) uses coordinated
    seeding so it will differ from a manually-constructed diseases= list.
    """
    pars = dict(n_agents=500, location='nigeria',
                start=1990, stop=1995, dt=1.0, rand_seed=0)
    # Use init_seeding='independent' so both sides draw from the same per-HPV
    # init_prev curves and are directly comparable.
    sim_a = hpv.Sim(genotypes=[16, 18], init_seeding='independent', **pars)
    sim_a.run()
    sim_b = hpv.Sim(
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        **pars,
    )
    sim_b.run()
    for key in ('hpv16', 'hpv18'):
        a_inf = float(np.asarray(sim_a.results[key].new_infections).sum())
        b_inf = float(np.asarray(sim_b.results[key].new_infections).sum())
        assert a_inf == pytest.approx(b_inf), \
            f'sugar vs explicit drift for {key}: {a_inf} vs {b_inf}'


def test_aggregate_cum_infections_any():
    """Aggregate.cum_infections_any counts agents ever infected with any genotype."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=1995, dt=1.0, rand_seed=0,
        genotypes=[16, 18],
    )
    sim.run()
    agg = sim.results['aggregate']
    any_cum = float(np.asarray(agg['cum_infections_any']).max())
    h16_cum = float(np.asarray(sim.results['hpv16'].new_infections).sum())
    h18_cum = float(np.asarray(sim.results['hpv18'].new_infections).sum())
    # Boolean-OR is at most the sum (equal iff no co-infections).
    assert any_cum > 0
    assert any_cum <= h16_cum + h18_cum + 1e-6


def test_aggregate_cum_cancers_any():
    """Aggregate.cum_cancers_any sums per-genotype cum_cancers across genotypes."""
    sim = hpv.Sim(
        n_agents=2000, location='nigeria',
        start=1990, stop=2010, dt=0.5, rand_seed=0,
        genotypes=[16, 18],
    )
    sim.run()
    agg = sim.results['aggregate']
    any_c = float(np.asarray(agg['cum_cancers_any'])[-1])
    sum_c = sum(float(np.asarray(sim.results[k].cum_cancers)[-1])
                for k in ('hpv16', 'hpv18'))
    assert any_c == pytest.approx(sum_c, abs=1e-6)


# ---------------------------------------------------------------------------
# M03 init_seeding tests
# ---------------------------------------------------------------------------

def test_exclusive_init_seeding_matches_v2_semantics():
    """init_seeding='exclusive' (default) gives each agent at most one genotype."""
    sim = hpv.Sim(
        n_agents=5000, location='nigeria',
        start=1990, stop=1991, dt=0.25, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
    )
    sim.init()
    # Per-agent: count how many genotypes are active
    n_inf_per_agent = np.zeros(5000, dtype=int)
    for k in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        n_inf_per_agent += np.asarray(sim.diseases[k].infected.values).astype(int)
    # Exclusive seeding: every agent has 0 or 1 initial infections, never >1
    assert n_inf_per_agent.max() <= 1, \
        f'exclusive seeding produced co-infections: max={n_inf_per_agent.max()}'


def test_independent_init_seeding_allows_coinfection():
    """init_seeding='independent' falls back to per-genotype draws (co-infections allowed)."""
    sim = hpv.Sim(
        n_agents=5000, location='nigeria',
        start=1990, stop=1991, dt=0.25, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
        init_seeding='independent',
    )
    sim.init()
    n_inf_per_agent = np.zeros(5000, dtype=int)
    for k in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        n_inf_per_agent += np.asarray(sim.diseases[k].infected.values).astype(int)
    # Independent seeding: should allow co-infections (max > 1)
    assert n_inf_per_agent.max() >= 2, \
        f'independent seeding had no co-infections: max={n_inf_per_agent.max()}'


def test_init_hpv_dist_weighted():
    """init_hpv_dist allows non-uniform genotype weights at initialisation."""
    # Heavy hpv16 weight; almost all initial infections should be hpv16.
    sim = hpv.Sim(
        n_agents=5000, location='nigeria',
        start=1990, stop=1991, dt=0.25, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
        init_hpv_dist={'hpv16': 0.97, 'hpv18': 0.01, 'hi5': 0.01, 'ohr': 0.01},
    )
    sim.init()
    n16   = int(np.asarray(sim.diseases.hpv16.infected.values).sum())
    n18   = int(np.asarray(sim.diseases.hpv18.infected.values).sum())
    nhi5  = int(np.asarray(sim.diseases.hi5.infected.values).sum())
    nohr  = int(np.asarray(sim.diseases.ohr.infected.values).sum())
    n_total = n16 + n18 + nhi5 + nohr
    if n_total > 0:
        assert n16 / n_total > 0.85, \
            f'hpv16 share should be ~97%, got {n16}/{n_total}={n16/n_total:.2f}'
