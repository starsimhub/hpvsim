# tests/test_multiscale_grow_acceptance.py
import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import numpy as np
import pytest
import hpvsim as hpv

# The three statistical gates below run ~72 full sims (many seeds/ratios) and
# take ~14 min — far over CI's 5-min timeout. They are marked ``slow`` so CI's
# ``pytest -m "not slow"`` run skips them; run them manually before opening a PR
# (or in a nightly job) with ``pytest -m slow tests/test_multiscale_grow_acceptance.py``.
# ``test_interventions_act_on_fine_agents`` below is deliberately NOT slow (one
# ~24s run) so the intervention-coverage mechanism stays covered on every PR.

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


def _intervention_factory():
    """Construct a fresh HPV screen -> colposcopy triage -> excision treatment cascade.

    The M06 canonical screen+treat construction.
    Coverage: routine_screening prob=0.7 (30-50 yo females), triage prob=0.9,
    treatment prob=0.8. Interventions start 2000 and run to sim stop, giving
    ~30+ years of program activity over the 1970-2040 horizon.
    """
    primary = hpv.routine_screening(
        name='primary',
        product='hpv',
        prob=0.7,
        age_range=[30, 50],
        sex='f',
        start_year=2000,
        end_year=2040,
    )
    colpo = hpv.routine_triage(
        name='colpo',
        product='colposcopy',
        prob=0.9,
        eligibility=lambda s: s.interventions['primary'].outcomes['positive'],
        start_year=2000,
        end_year=2040,
    )
    excision = hpv.treat_num(
        name='excision_rx',
        product='excision',
        prob=0.8,
        eligibility=lambda s: s.interventions['colpo'].outcomes['hsil'],
    )
    return [primary, colpo, excision]


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


def test_interventions_act_on_fine_agents():
    """Fine multiscale agents must actually RECEIVE interventions, not just be
    consistent in aggregate.

    The equivalence gate above shows the population-level averted fraction
    matches across ratios, but that could in principle hold even if fine agents
    were skipped and coarse agents over-treated. This asserts the mechanism
    directly: at ratio=10, a real (non-zero) count of FINE agents is screened,
    triaged, AND CIN-treated. Interventions carry no ~fine guard — fine agents
    inherit female/alive/age/cancer state from their source clone, so they are
    eligible natively. (Validated scope: probability-based coverage `prob=`;
    fixed-capacity `max_capacity` is scale-sensitive and out of scope — see the
    design spec §8 coverage-type caveat.)
    """
    sim = hpv.Sim(location='nigeria', n_agents=8000, start=1970, stop=2040,
                  ms_agent_ratio=10, rand_seed=1,
                  interventions=_intervention_factory())
    sim.run()
    ppl = sim.people
    fine = np.asarray(ppl.fine[ppl.auids], dtype=bool)
    assert fine.any(), 'no fine agents were grown — cannot test intervention coverage'

    def n_fine_flagged(intervention_name, flag):
        iv = sim.interventions[intervention_name]
        vals = np.asarray(getattr(iv, flag)[ppl.auids], dtype=bool)
        return int((vals & fine).sum())

    n_screened = n_fine_flagged('primary', 'screened')
    n_triaged  = n_fine_flagged('colpo',   'screened')
    n_treated  = n_fine_flagged('excision_rx', 'cin_treated')
    assert n_screened > 0, f'no fine agent was screened (fine_screened={n_screened})'
    assert n_triaged  > 0, f'no fine agent was triaged (fine_triaged={n_triaged})'
    assert n_treated  > 0, f'no fine agent was CIN-treated (fine_treated={n_treated})'


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
