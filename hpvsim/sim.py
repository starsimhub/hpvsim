"""HPVsim convenience Sim wrapper.

Provides a v2-compatible API: ``hpv.Sim(location='nigeria', genotype='hpv16')``.
Instantiates the four-component default stack (HPV disease module, two
SexualNetwork layers (m, c), ss.Births + ss.Deaths + AgeMigration
demographics, ss.People with location-specific age pyramid) and forwards
to ss.Sim. All defaults are overridable via kwargs (passing ``diseases=`` /
``networks=`` / ``demographics=`` / ``people=`` short-circuits the
convenience wiring).

Demographics: v2 uses a population-level CBR (no per-woman ASFR data); we
match that with ss.Births rather than ss.Pregnancy. AgeMigration pins the
agent age pyramid to the location's UN trajectory each year, matching v2's
``check_migration``; without it, agent populations only grow via births so
the age structure skews younger than the data target. M02+ may switch to
ss.Pregnancy if proper age-stratified fertility data becomes available.

M01: single-genotype only. M03 changes the signature to ``genotypes=[...]``.
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
            networks = [
                SexualNetwork(layer=k, pars=country['network_pars'][k])
                for k in ('m', 'c')
            ]
        demographics = kwargs.pop('demographics', None)
        if demographics is None:
            # hpv.data.load_country produces birth_rate and death_rate in
            # the column shapes ss.Births / ss.Deaths consume by default
            # (Year/CBR for births; Time/AgeGrpStart/Sex/mx with sex labels
            # 'Female'/'Male' and per-1000 rate units for deaths). No
            # metadata override needed — the v2-to-Starsim translation is
            # encapsulated in the adapter.
            #
            # AgeMigration pins the age pyramid to the country's UN
            # trajectory each year (matching v2's check_migration). Without
            # it, the agent population only grows via births so the
            # active-age cohorts skew younger than v2 / data target, which
            # in turn inflates per-step casual-partnership formation.
            demographics = [
                ss.Births(birth_rate=country['birth_rate']),
                ss.Deaths(death_rate=country['death_rate']),
                AgeMigration(),
            ]
        # Store total_pop so init() can compute pop_scale after ss.Sim.__init__
        # wires self.pars.  ss.SimPars.validate_total_pop() (called during
        # sim.init()) will set pop_scale = total_pop / n_agents when total_pop
        # is given, or pop_scale = 1.0 when it is None.
        self._total_pop = total_pop
        # Store location so demographic modules (e.g. AgeMigration) can load
        # country data during their init_pre without needing it passed as a
        # constructor argument.
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