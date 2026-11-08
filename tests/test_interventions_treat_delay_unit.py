"""A treatment after a delay: ss.treat(when=ss.on(..., delay=...)), as hpv.treat_delay was."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def test_treatment_two_years_after_screening():
    """With delay=2 years, those referred are treated 8 quarterly steps after their screening, if still eligible then."""
    women = ss.person.female & (ss.person.age >= 30) & (ss.person.age < 50)
    screening = ss.screen(hpv.dx('via'), who=women, when=['2021'], coverage=1.0, name='screening')
    treatment = ss.treat(hpv.tx('excision'), when=ss.on(screening.positive, delay=ss.years(2)), coverage=1.0, name='rx')
    sim = hpv.Sim(n_agents=500, start=2020, stop=2025, location='nigeria', genotypes=[16],
                  interventions=[screening, treatment], verbose=0)
    sim.run()
    r = sim.results
    screened = np.flatnonzero(np.asarray(r.screening.delivered))
    treated = np.flatnonzero(np.asarray(r.rx.delivered))
    assert len(screened) == 1 and len(treated) == 1
    assert treated[0] - screened[0] == int(round(2/sim.t.dt.years))
    assert r.rx.delivered.sum() <= r.screening.positive.sum() + 1e-6
