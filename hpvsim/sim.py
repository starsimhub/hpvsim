"""HPVsim convenience Sim wrapper.

``hpv.Sim(location='nigeria', genotype='hpv16')`` instantiates the default
stack — HPV disease module, multi-layer SexualNetwork, ss.Births + ss.Deaths
+ AgeMigration demographics, ss.People with location-specific age pyramid —
and forwards to ``ss.Sim``. Each component is overridable: passing
``diseases=`` / ``networks=`` / ``demographics=`` / ``people=`` short-circuits
the matching default.

Demographics use ss.Births (population-level CBR) rather than ss.Pregnancy
(per-woman ASFR), matching the available data. AgeMigration pins the age
pyramid to the country's UN trajectory each year; without it the active-age
cohorts skew younger than the data target. A future switch to ss.Pregnancy
is possible once age-stratified fertility data is available.

Currently single-genotype; future multi-genotype support replaces the
``genotype=`` kwarg with ``genotypes=[...]``.
"""

import starsim as ss

from .data.country import load_country
from .demographics import AgeMigration
from .hpv import HPV
from .network import SexualNetwork


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location='nigeria', genotype='hpv16',
                 n_agents=10_000, start=1990, stop=2060, dt=0.25,
                 total_pop=None, pars=None, **kwargs):
        country = load_country(location)
        people = kwargs.pop('people', None)
        if people is None:
            people = ss.People(n_agents, age_data=country['age_data'])
        diseases = kwargs.pop('diseases', None)
        if diseases is None:
            diseases = [HPV(genotype=genotype)]
        networks = kwargs.pop('networks', None)
        if networks is None:
            networks = [SexualNetwork(**country['network_pars'])]
        demographics = kwargs.pop('demographics', None)
        if demographics is None:
            demographics = [
                ss.Births(birth_rate=country['birth_rate']),
                ss.Deaths(death_rate=country['death_rate']),
                AgeMigration(),
            ]
        # AgeMigration.init_pre reads sim.location to load country data.
        self.location = location.lower()
        super().__init__(
            start=ss.years(start),
            stop=ss.years(stop),
            dt=ss.years(dt),
            people=people,
            diseases=diseases,
            networks=networks,
            demographics=demographics,
            pars=pars,
            total_pop=total_pop,
            **kwargs,
        )