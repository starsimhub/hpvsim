"""HPVsim convenience Sim wrapper.

Provides a v2-compatible API: ``hpv.Sim(location='nigeria', genotype='hpv16')``.
Instantiates the four-component default stack (HPV disease module, two
SexualNetwork layers (m, c), ss.Births + ss.Deaths demographics, ss.People
with location-specific age pyramid) and forwards to ss.Sim. All defaults are
overridable via kwargs (passing ``diseases=`` / ``networks=`` /
``demographics=`` / ``people=`` short-circuits the convenience wiring).

Demographics: v2 uses a population-level CBR (no per-woman ASFR data); we
match that with ss.Births rather than ss.Pregnancy. M02+ may switch to
ss.Pregnancy if proper age-stratified fertility data becomes available.

M01: single-genotype only. M03 changes the signature to ``genotypes=[...]``.
"""

import starsim as ss

from .data.country import load_country
from .hpv import HPV
from .network import SexualNetwork


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location='nigeria', genotype='hpv16',
                 n_agents=10_000, start=1990, stop=2060, dt=0.5,
                 pars=None, **kwargs):
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
            # hpv.data.load_country produces death-rate data with columns
            # (Year, AgeGrp, Sex, Rate); ss.Deaths defaults to UN-style
            # (Time, AgeGrpStart, Sex, mx) and rate_units=1e-3, so we pass
            # matching metadata AND rate_units=1 (v2 mortality data is
            # already fractional, not per-1000).
            # Birth-rate data is [Year, CBR], matching ss.Births defaults
            # (per-1000 — keep the default rate_units=1e-3).
            death_metadata = dict(
                data_cols=dict(year='Year', age='AgeGrp', sex='Sex', value='Rate'),
                sex_keys={'f': 'f', 'm': 'm'},
            )
            demographics = [
                ss.Births(birth_rate=country['birth_rate']),
                ss.Deaths(
                    death_rate=country['death_rate'],
                    rate_units=1,
                    metadata=death_metadata,
                ),
            ]
        super().__init__(
            start=ss.years(start),
            stop=ss.years(stop),
            dt=ss.years(dt),
            people=people,
            diseases=diseases,
            networks=networks,
            demographics=demographics,
            pars=pars,
            **kwargs,
        )