# tests/test_multiscale_grow_acceptance.py
import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import numpy as np
import hpvsim as hpv

def _total_cancers(ratio, seed, n_agents=8000):
    sim = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2030,
                  ms_agent_ratio=ratio, rand_seed=seed)
    sim.run()
    tot = 0.0
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            tot += float(dis.results.new_cancers.values.sum())
    return tot

def test_cancer_incidence_flat_across_ratio():
    seeds = range(8)
    base = np.mean([_total_cancers(1, s) for s in seeds])
    for ratio in (5, 10):
        got = np.mean([_total_cancers(ratio, s) for s in seeds])
        rel = got / base
        assert 0.90 <= rel <= 1.10, f'ratio={ratio}: {rel:.3f} (base={base:.0f})'


def _intervention_factory():
    """Construct a fresh HPV screen -> colposcopy triage -> excision treatment cascade.

    Copied verbatim from tests/regression/anchor_screen_treat.py (_build_interventions),
    which is the M06 canonical screen+treat construction verified for parity with v2.
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
