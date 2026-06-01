"""Internal-equivalence acceptance gate for multiscale agents (slow).

Two people-space properties of the cancer-pathway ledger, multi-seed:

  * MASS CONSERVATION — total cancers for ``ratio=12`` match ``ratio=1`` at the
    same (large) agent count, on the seed mean (the ledger is an unbiased
    estimator up to its documented competing-risk residual).
  * VARIANCE REDUCTION — at equal agent count, ``ratio=12`` has LOWER seed-to-
    seed variance than ``ratio=1`` (the payoff of resolving the rare cancer
    event at finer granularity), tested in the rare-cancer regime where the
    benefit is detectable (see ``test_multiscale_reduces_variance...``).

Marked ``slow``.
"""
import numpy as np
import pytest
import hpvsim as hpv

# Large-population config for the mass-conservation mean check.
CFG = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2060,
           dt=0.25, total_pop=1e6, verbose=0)
# Rare-cancer config for the variance-reduction check (see that test's
# docstring for why a small population is required to see the effect).
CFG_RARE = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2040,
                dt=0.25, total_pop=1e6, verbose=0)
SEEDS = range(10)


def _total_cancers_people(sim):
    res = sim.results.hpv16
    return float(np.asarray(res.new_cancers).sum())


def _mean_over_seeds(n_agents, ratio, seeds=SEEDS, cfg=CFG):
    vals = []
    for sd in seeds:
        s = hpv.Sim(n_agents=n_agents, ms_agent_ratio=ratio, rand_seed=sd, **cfg)
        s.run()
        vals.append(_total_cancers_people(s))
    return np.array(vals)


@pytest.mark.slow
def test_multiscale_matches_single_scale_mean():
    """Ledger total cancers match single-scale at equal (large) agent count.

    Tolerance 8% (matching the sister gate
    ``test_multiscale_distribution.test_cancer_count_unbiased``). The systematic
    competing-risk residual is small (~-1%); the bound is for seed-batch
    robustness, not to cover a large bias."""
    single = _mean_over_seeds(40000, 1)
    multi = _mean_over_seeds(40000, 12)
    rel_bias = abs(multi.mean() - single.mean()) / single.mean()
    assert rel_bias < 0.08, (
        f'multiscale mean off by {rel_bias:.1%} '
        f'(single={single.mean():.0f}, multi={multi.mean():.0f})'
    )


@pytest.mark.slow
def test_multiscale_reduces_variance_at_equal_agents():
    """At equal agent count, the ledger has lower seed-to-seed variance on the
    total cancer count than single-scale — multiscale's core payoff.

    Because the ledger keeps the population bit-identical across ratio, the
    seed variance decomposes EXACTLY as Var = Var_transmission + Var_resolution,
    with Var_transmission (how many agents ever reach the CIN->cancer decision)
    shared between ratio=1 and ratio=12 and Var_resolution (the rare-event draw)
    reduced ~N-fold by the ledger. So ratio=12's true variance is strictly lower
    by (1-1/N)*Var_resolution.

    Tested in the RARE-cancer regime (small population) because that is where
    the reduction is DETECTABLE at a feasible seed count: when cancer is common
    (e.g. n=40000, or n=4000 over a long window) Var_transmission dominates and
    the resolution term — though still reduced — is a tiny fraction of the
    total, so a 10-seed sample std flickers around 1.0. At n=2000/stop=2040
    cancer is rare enough that the reduction is large and robust (measured
    ms/base std ratio ~0.5-0.65 across independent seed batches)."""
    base = _mean_over_seeds(2000, 1, cfg=CFG_RARE)
    ms = _mean_over_seeds(2000, 12, cfg=CFG_RARE)
    assert ms.std(ddof=1) < base.std(ddof=1), (
        f'multiscale should reduce variance in the rare-cancer regime '
        f'(base std={base.std(ddof=1):.0f}, ms std={ms.std(ddof=1):.0f})'
    )
