"""Tiny smoke variant of examples/m07_uq_sweep.py so CI catches bit rot.

Uses 1 coverage × 2 seeds. Verifies the make_sim + ss.parallel + ss.MultiSim
+ median pipeline executes and produces non-trivial results — NOT a parity
gate. Real-shape demo lives in examples/m07_uq_sweep.py.
"""
import sciris as sc
import starsim as ss
import hpvsim as hpv


def _make_tiny_vx_sim(seed, coverage):
    return hpv.Sim(
        n_agents=500,
        location='nigeria',
        genotypes=[16],
        start=2010,
        stop=2013,
        dt=0.25,
        rand_seed=int(seed),
        verbose=0,
        label=f'coverage={coverage:.0%}|seed={seed}',
        interventions=[hpv.routine_vx(
            product='bivalent', prob=float(coverage),
            age_range=(9, 14), sex='f',
        )],
    )


def test_demo_pipeline_runs_end_to_end():
    """sc.parallelize + ss.parallel + ss.MultiSim.median works on hpv.Sim."""
    iterkwargs = [dict(seed=s, coverage=0.6) for s in range(2)]
    sims = sc.parallelize(_make_tiny_vx_sim, iterkwargs=iterkwargs)
    msim = ss.parallel(*sims, verbose=0)
    sub = ss.MultiSim(list(msim.sims))
    sub.median()
    # MultiSim results use flattened keys (hpv16_cum_infections), not nested.
    cum_inf = sub.results.hpv16_cum_infections
    assert len(cum_inf) > 0
    assert float(cum_inf[-1]) >= 0