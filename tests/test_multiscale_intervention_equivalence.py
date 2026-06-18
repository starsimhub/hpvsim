"""Acceptance gate: multiscale stays correct UNDER screening+treatment.

The data-overlay approach failed exactly here (extras resampled blind to
interventions reported ~105 cancers where the truth was ~0). Fine agents are
real, so screening/treatment act on them; this gate proves it.
"""
import numpy as np
import pytest
import hpvsim as hpv

_CFG = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2040,
            dt=0.25, n_agents=5000, verbose=0)


def _interventions():
    return [
        hpv.routine_screening(product='via', prob=0.9, start_year=2000,
                              age_range=[30, 50], name='screen'),
        hpv.treat_num(name='cin_rx', product='excision', prob=0.9),
    ]


def _cancers(ratio, seed, with_intv):
    intvs = _interventions() if with_intv else []
    s = hpv.Sim(ms_agent_ratio=ratio, rand_seed=seed, interventions=intvs, **_CFG)
    s.run()
    return float(np.asarray(s.results.hpv16.new_cancers).sum())


@pytest.mark.slow
def test_intervention_averts_cancer_is_real():
    """Sanity: the screening+treatment program averts a large, measurable
    fraction of cancers at single scale (so the equivalence test is not vacuous)."""
    base = np.mean([_cancers(1, sd, False) for sd in (0, 1, 2, 3)])
    intv = np.mean([_cancers(1, sd, True) for sd in (0, 1, 2, 3)])
    assert intv < 0.5 * base, f'intervention should avert >50%: base={base:.1f} intv={intv:.1f}'


@pytest.mark.slow
def test_multiscale_matches_single_scale_under_intervention():
    """THE gate: with screening+treatment active, ratio=N realized cancers match
    ratio=1 within tolerance. A treatment-blind implementation (the overlay) fails
    this by a wide margin (it would report the un-averted natural-history count)."""
    seeds = (0, 1, 2, 3)
    one = np.mean([_cancers(1, sd, True) for sd in seeds])
    many = np.mean([_cancers(12, sd, True) for sd in seeds])
    # tolerance = internal-equivalence (~5%) + accepted ~8% residual, widened for
    # the low post-intervention count; an intervention-blind impl misses by >5x.
    assert abs(many - one) <= 0.20 * max(one, 1.0), (
        f'multiscale diverges under intervention: ratio1={one:.2f} ratio12={many:.2f}')
