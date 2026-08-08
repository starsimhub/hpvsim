"""Statistical acceptance gates for the multiscale grow engine.

These three gates run ~72 full sims (many seeds x ratios) and take ~14 min.
The mechanism-level checks — that fine agents are spawned, excluded from the
network, subject to emigration, and reached by interventions — live in
``tests/test_multiscale_grow_unit.py`` and
``tests/test_multiscale_grow_acceptance.py``. What can only be checked
statistically lives here.
"""
import numpy as np
import pytest
import hpvsim as hpv

from tests.test_multiscale_grow_acceptance import _intervention_factory


def _total_cancers(ratio, seed, n_agents=8000):
    sim = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2030,
                  ms_agent_ratio=ratio, rand_seed=seed)
    sim.run()
    tot = 0.0
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            tot += float(dis.results.new_cancers.values.sum())
    return tot


@pytest.mark.slow
def test_cancer_incidence_flat_across_ratio():
    seeds = range(8)
    base = np.mean([_total_cancers(1, s) for s in seeds])
    for ratio in (5, 10):
        got = np.mean([_total_cancers(ratio, s) for s in seeds])
        rel = got / base
        assert 0.90 <= rel <= 1.10, f'ratio={ratio}: {rel:.3f} (base={base:.0f})'


def _averted_fraction(ratio, seed, n_agents=8000):
    """Run paired base vs. screen+treat sims at the given ratio/seed; return averted fraction."""
    base = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2040,
                   ms_agent_ratio=ratio, rand_seed=seed)
    base.run()
    treat = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2040,
                    ms_agent_ratio=ratio, rand_seed=seed,
                    interventions=_intervention_factory())
    treat.run()

    def tot(sim):
        return sum(float(d.results.new_cancers.values.sum())
                   for d in sim.diseases.values() if isinstance(d, hpv.HPV))

    b, t = tot(base), tot(treat)
    return (b - t) / b


@pytest.mark.slow
def test_intervention_equivalence_across_ratio():
    """CENTERPIECE gate: averted cancer fraction must match at ratio=1 vs ratio=10.

    Fine agents are REAL (scale=1/ratio), so a screen+treat program acts on them
    natively. This property could NOT be guaranteed by the abandoned ledger approach
    (which resampled extras without accounting for screening/treatment state).

    Asserts:
    - |av10 - av1| <= 0.05  (ratios agree within 5 pp)
    - av1 > 0.05            (intervention actually averts a non-trivial fraction)
    """
    seeds = range(6)
    av1  = np.mean([_averted_fraction(1,  s) for s in seeds])
    av10 = np.mean([_averted_fraction(10, s) for s in seeds])
    assert av1 > 0.05, (
        f'intervention should avert a non-trivial cancer fraction at ratio=1; got av1={av1:.3f}'
    )
    assert abs(av10 - av1) <= 0.05, (
        f'averted fraction disagrees across ratios: ratio1={av1:.3f}  ratio10={av10:.3f}  '
        f'|diff|={abs(av10-av1):.3f}'
    )


def _mean_age_at_cancer(ratio, seed, n_agents=6000):
    """Run a baseline sim and return mean age at cancer (summed across genotypes)."""
    sim = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2030,
                  ms_agent_ratio=ratio, rand_seed=seed)
    sim.run()
    s_age = 0.0
    n = 0.0
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            s_age += float(dis.results.sum_age_at_cancer.values.sum())
            n += float(dis.results.new_cancers.values.sum())
    return s_age / n if n else np.nan


@pytest.mark.slow
def test_event_age_variance_shrinks_with_ratio():
    """Higher ms_agent_ratio grows more fine cancer agents → tighter mean-age estimator.

    At ratio=1 each coarse agent produces ~1 cancer event; at ratio=10 each coarse
    agent spawns ~10 fine agents so the mean-age-at-cancer estimator is based on ~10x
    more resolved events per seed → cross-seed variance should be lower.

    Asserts: var(ratio=10) < var(ratio=1).
    Reports var values so they can be inspected even if the test fails.
    """
    seeds = range(12)
    ages1  = [_mean_age_at_cancer(1,  s) for s in seeds]
    ages10 = [_mean_age_at_cancer(10, s) for s in seeds]
    # Guard: a seed with zero cancers yields nan (see _mean_age_at_cancer). At
    # n_agents=6000 every seed has hundreds of cancers so this never triggers,
    # but a smaller-n run could — fail with a clear message rather than a vacuous
    # "nan < x is False".
    assert not (np.isnan(ages1).any() or np.isnan(ages10).any()), (
        f'zero-cancer seed produced nan mean-age (raise n_agents): '
        f'nans r1={int(np.isnan(ages1).sum())} r10={int(np.isnan(ages10).sum())}'
    )
    var1  = float(np.var(ages1))
    var10 = float(np.var(ages10))
    mean1  = float(np.nanmean(ages1))
    mean10 = float(np.nanmean(ages10))
    assert var10 < var1, (
        f'variance did NOT shrink: var(ratio=1)={var1:.4f}  var(ratio=10)={var10:.4f}  '
        f'mean_age(ratio=1)={mean1:.2f}  mean_age(ratio=10)={mean10:.2f}'
    )
