"""HPVsim convenience Sim wrapper.

``hpv.Sim(location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'])`` instantiates
the default stack — one HPV disease module per genotype, multi-layer
SexualNetwork, ss.Births + ss.Deaths + AgeMigration demographics, ss.People
with location-specific age pyramid, plus a CrossImmunity connector — and
forwards to ``ss.Sim``. Each component is overridable: passing ``diseases=``
short-circuits the genotypes-sugar path.
"""

import starsim as ss

from .data.country import load_country
from .demographics import AgeMigration
from .hpv import HPV
from .network import SexualNetwork
from .connectors import CrossImmunity
from .parameters import genotype_aliases


def _normalize_genotype(key):
    """Resolve aliases (16 -> 'hpv16', 'hi5' -> 'hi5') to canonical keys."""
    s = str(key).lower().strip()
    for canonical, aliases in genotype_aliases.items():
        if s == canonical or s in aliases:
            return canonical
    raise ValueError(
        f'Unknown genotype {key!r}; valid: {list(genotype_aliases)}'
    )


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location='nigeria', genotypes=None, genotype_pars=None,
                 n_agents=10_000, start=1990, stop=2060, dt=0.25,
                 total_pop=None, pars=None, **kwargs):
        country = load_country(location)
        people = kwargs.pop('people', None)
        if people is None:
            people = ss.People(n_agents, age_data=country['age_data'])

        diseases = kwargs.pop('diseases', None)
        connectors = kwargs.pop('connectors', None)

        if diseases is not None and genotypes is not None:
            raise ValueError(
                'Pass diseases= OR genotypes=, not both.'
            )

        if diseases is None:
            # Default to single-genotype HPV16 if neither supplied.
            keys = (tuple(_normalize_genotype(g) for g in genotypes)
                    if genotypes is not None else ('hpv16',))
            gpars_overrides = genotype_pars or {}
            diseases = [
                HPV(genotype=k, **gpars_overrides.get(k, {}))
                for k in keys
            ]

        if connectors is None:
            connectors = [CrossImmunity()]

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
            connectors=connectors,
            networks=networks,
            demographics=demographics,
            pars=pars,
            total_pop=total_pop,
            **kwargs,
        )