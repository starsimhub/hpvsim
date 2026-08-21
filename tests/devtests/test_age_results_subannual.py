"""Sub-annual accumulation gate for by_age.cancer_incidence.

Needs ms_agent_ratio=50 over 5000 agents and 35 years to resolve enough cancers
per single tick for the 4x ratio to be distinguishable from noise, so it lives
here rather than in tests/test_age_results.py.
"""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv


@pytest.mark.slow
def test_cancer_incidence_accumulates_over_full_year_at_subannual_dt():
    """Incidence sums new events across all sub-steps of a calendar year.

    Regression guard for the dt<1 undercount: at dt=0.25 a fully-covered year
    (2019, 4 sub-steps) must report ~4x the incidence of the terminal year
    (2020, only its first tick, since the sim stops at 2020). The old code
    captured a single sub-step per year, giving a ratio near 1. Comparing two
    years within one run isolates the accumulation from dt-dependent dynamics.
    """
    edges = np.array([0., 30., 50., 70., 100.])
    az = hpv.by_age(result_args=sc.objdict(
        cancer_incidence=sc.objdict(years=[2019, 2020], edges=edges)))
    sim = hpv.Sim(n_agents=5000, location='nigeria', genotypes=[16, 18],
                  start=1985, stop=2020, dt=0.25, ms_agent_ratio=50,
                  rand_seed=0, analyzers=[az], verbose=0)
    sim.run()
    ar = sim.analyzers['by_age']
    full_year = float(np.sum(ar.outputs['cancer_incidence'][2019.0]))
    terminal = float(np.sum(ar.outputs['cancer_incidence'][2020.0]))
    assert terminal > 0, 'expected some cancers at the terminal tick'
    assert full_year > 2.5 * terminal, (
        f'full-year 2019 incidence ({full_year:.1f}) should be ~4x the single-tick '
        f'terminal 2020 ({terminal:.1f}); got ratio {full_year / terminal:.2f} '
        f'(a ratio near 1 means the annual accumulation regressed)')
