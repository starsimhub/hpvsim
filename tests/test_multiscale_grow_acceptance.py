import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import numpy as np
import starsim as ss
import hpvsim as hpv

# Cheap mechanism check only; the ~72-sim statistical gates live in
# devtests/test_multiscale_grow_gates.py, which reuses _intervention_factory.

def _intervention_factory():
    """Construct a fresh HPV screen -> colposcopy triage -> excision treatment cascade.

    The canonical screen+treat construction. Coverage: screening 70% of women
    aged 30-50 each year, triage 90% of those referred, treatment 80% of those
    referred. Screening runs from 2000 to 2040.
    """
    women = ss.ppl.female & (ss.ppl.age >= 30) & (ss.ppl.age < 50)
    primary = ss.screen(hpv.dx('hpv'), where=women, coverage=0.7, interval=ss.years(1), start=2000, stop=2040, name='primary')
    colpo = ss.screen(hpv.dx('colposcopy'), who=primary.positive, coverage=0.9, name='colpo')
    excision = ss.treat(hpv.tx('excision'), who=colpo.hsil & hpv.no_cancer, coverage=0.8, name='excision_rx')
    return [primary, colpo, excision]


def test_interventions_act_on_fine_agents():
    """At ratio=10 a non-zero count of fine agents is screened, triaged and
    CIN-treated.

    Interventions carry no ~fine guard: fine agents inherit
    female/alive/age/cancer state from their source clone, so they are eligible
    natively. Holds for probability-based coverage; fixed-capacity
    ``max_capacity`` is scale-sensitive and untested here. 4000 agents at dt=0.5
    over 2000-2040 keeps all three counts in the hundreds.
    """
    reached = dict(primary=[], colpo=[], excision_rx=[])
    def track(sim): # The agents each step reached (and for treatment, cured)
        for name, uids in reached.items():
            uids.extend(sim.interventions[name].reached('cured' if name == 'excision_rx' else None).tolist())
    sim = hpv.Sim(location='nigeria', n_agents=4000, start=2000, stop=2040,
                  dt=0.5, ms_agent_ratio=10, rand_seed=1,
                  interventions=_intervention_factory(), analyzers=track)
    sim.run()
    ppl = sim.people
    fine = np.asarray(ppl.fine[ppl.auids], dtype=bool)
    assert fine.any(), 'no fine agents were grown — cannot test intervention coverage'

    def n_fine_flagged(name):
        return int(np.isin(ppl.auids[fine], reached[name]).sum())

    n_screened = n_fine_flagged('primary')
    n_triaged  = n_fine_flagged('colpo')
    n_treated  = n_fine_flagged('excision_rx')
    assert n_screened > 0, f'no fine agent was screened (fine_screened={n_screened})'
    assert n_triaged  > 0, f'no fine agent was triaged (fine_triaged={n_triaged})'
    assert n_treated  > 0, f'no fine agent was CIN-treated (fine_treated={n_treated})'
