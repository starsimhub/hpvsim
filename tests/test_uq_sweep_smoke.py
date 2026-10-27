"""Tiny smoke variant of examples/m07_uq_sweep.py so CI catches bit rot.

Uses 2 coverages × 2 seeds. Verifies the make_sim + ss.parallel + quantiles
by scenario pipeline executes and produces non-trivial results — NOT a parity
gate. Real-shape demo lives in examples/m07_uq_sweep.py.
"""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _make_tiny_vx_sim(coverage):
    girls = ss.ppl.female & (ss.ppl.age >= 9) & (ss.ppl.age < 14)
    return hpv.Sim(
        n_agents=500, location='nigeria', genotypes=[16], start=2010, stop=2013, dt=0.25,
        verbose=0, label=f'coverage={coverage:.0%}',
        interventions=ss.vaccinate(hpv.vx('bivalent'), where=girls, coverage=coverage),
    )


def test_demo_pipeline_runs_end_to_end():
    """ss.parallel(*sims, n_runs=) and quantiles by scenario work on hpv.Sim."""
    msim = ss.parallel(_make_tiny_vx_sim(0.3), _make_tiny_vx_sim(0.6), n_runs=2, verbose=0)
    res = msim.results.hpv16.cum_infections
    assert np.shape(res)[0] == 4 # Runs
    q = np.asarray(res.quantiles([0.1, 0.5, 0.9])).reshape(2, 3, -1) # One row per scenario and quantile
    low, mid, high = q[:, 0], q[:, 1], q[:, 2]
    assert mid.shape == (2, np.shape(res)[1]) # Each scenario
    assert (mid[:, -1] >= 0).all() and (high >= low).all()
