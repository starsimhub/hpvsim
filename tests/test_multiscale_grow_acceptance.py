# tests/test_multiscale_grow_acceptance.py
import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import numpy as np
import hpvsim as hpv

# The mechanism check below is deliberately cheap enough to run on every PR.
# The statistical gates it complements — cancer incidence flat across ratio,
# averted fraction equal across ratios, mean-age variance shrinking with ratio —
# need ~72 full sims and live in ``devtests/test_multiscale_grow_gates.py``,
# which reuses ``_intervention_factory`` from this module.

def _intervention_factory():
    """Construct a fresh HPV screen -> colposcopy triage -> excision treatment cascade.

    The canonical screen+treat construction.
    Coverage: routine_screening prob=0.7 (30-50 yo females), triage prob=0.9,
    treatment prob=0.8. Interventions start 2000 and run to 2040, giving 40
    years of program activity over the sims that use them.
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


def test_interventions_act_on_fine_agents():
    """Fine multiscale agents must actually RECEIVE interventions, not just be
    consistent in aggregate.

    The equivalence gate in devtests shows the population-level averted
    fraction matches across ratios, but that could in principle hold even if
    fine agents were skipped and coarse agents over-treated. This asserts the mechanism
    directly: at ratio=10, a real (non-zero) count of FINE agents is screened,
    triaged, AND CIN-treated. Interventions carry no ~fine guard — fine agents
    inherit female/alive/age/cancer state from their source clone, so they are
    eligible natively. (Validated scope: probability-based coverage `prob=`;
    fixed-capacity `max_capacity` is scale-sensitive and out of scope — see the
    design spec §8 coverage-type caveat.)

    Sized for the cheapest run that keeps all three counts comfortably
    non-zero: 4000 agents at dt=0.5 over 2000-2040 screens/triages/treats a few
    hundred fine agents on every seed tried.
    """
    sim = hpv.Sim(location='nigeria', n_agents=4000, start=2000, stop=2040,
                  dt=0.5, ms_agent_ratio=10, rand_seed=1,
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
