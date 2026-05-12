"""HPVsim convenience Sim wrapper.

``hpv.Sim(location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'])`` instantiates
the default stack — one HPV disease module per genotype, multi-layer
SexualNetwork, ss.Births + ss.Deaths + AgeMigration demographics, ss.People
with location-specific age pyramid, plus a CrossImmunity connector — and
forwards to ``ss.Sim``. Each component is overridable: passing ``diseases=``
short-circuits the genotypes-sugar path.

Kwargs:
  ``init_seeding`` (str, default ``'exclusive'``):
    ``'exclusive'`` — one Bernoulli per agent using the hpv16 age-banded
    curve as the total HPV prevalence, then exactly one genotype assigned per
    infected agent. No co-infection at initialisation.
    ``'independent'`` — each genotype draws from its own per-genotype
    init_prev curve independently; co-infection at initialisation is possible.

  ``init_hpv_dist`` (dict or None, default ``None``):
    Only used when ``init_seeding='exclusive'``. If ``None``, genotype
    assignment is uniform across active genotypes. If a dict, keys must be
    the resolved canonical genotype names (e.g. ``{'hpv16': 0.6, 'hpv18':
    0.2, 'hi5': 0.1, 'ohr': 0.1}``) and values are weights (need not sum to
    1; normalised internally).
"""

import starsim as ss

from .analyzers import Aggregate
from .data.country import load_country
from .demographics import AgeMigration
from .hpv import HPV, _ExclusiveSeeder
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
                 init_seeding='exclusive', init_hpv_dist=None,
                 n_agents=10_000, start=1990, stop=2060, dt=0.25,
                 total_pop=None, pars=None, **kwargs):
        # Pass start year so the age pyramid matches sim.start (loader
        # defaults to year 2000 with a materially different distribution).
        country = load_country(location, year=int(start))
        people = kwargs.pop('people', None)
        if people is None:
            people = ss.People(n_agents, age_data=country['age_data'])

        diseases = kwargs.pop('diseases', None)
        connectors = kwargs.pop('connectors', None)

        if diseases is not None and genotypes is not None:
            raise ValueError(
                'Pass diseases= OR genotypes=, not both.'
            )

        if init_seeding not in ('exclusive', 'independent'):
            raise ValueError(
                f"init_seeding must be 'exclusive' or 'independent'; got {init_seeding!r}"
            )

        if diseases is None:
            # Default to single-genotype HPV16 if neither supplied.
            keys = (tuple(_normalize_genotype(g) for g in genotypes)
                    if genotypes is not None else ('hpv16',))
            gpars_overrides = genotype_pars or {}

            # Validate init_hpv_dist keys if provided.
            if init_hpv_dist is not None:
                if not isinstance(init_hpv_dist, dict):
                    raise ValueError(
                        f'init_hpv_dist must be a dict or None; got {type(init_hpv_dist)}'
                    )
                dist_keys = set(init_hpv_dist.keys())
                sim_keys = set(keys)
                if dist_keys != sim_keys:
                    raise ValueError(
                        f'init_hpv_dist keys {sorted(dist_keys)} do not match '
                        f'resolved genotype keys {sorted(sim_keys)}'
                    )

            diseases = [HPV(genotype=k, **gpars_overrides.get(k, {})) for k in keys]
            if init_seeding == 'exclusive':
                # 'exclusive': one Bernoulli per agent for any HPV, then one
                # genotype per infected agent via the seeder's per-genotype callback.
                # 'independent' is the no-op path — each HPV's per-genotype init_prev
                # curve drives its own seeding independently.
                self._seeder = _ExclusiveSeeder(
                    genotype_keys=keys, init_hpv_dist=init_hpv_dist
                )
                for d, k in zip(diseases, keys):
                    d.pars.init_prev = ss.bernoulli(p=self._seeder.for_genotype(k))

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

        # Auto-add the any-genotype aggregator unless the caller supplied their
        # own analyzers list (in which case they can add it manually).
        analyzers = kwargs.pop('analyzers', None)
        if analyzers is None:
            analyzers = [Aggregate()]

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
            analyzers=analyzers,
            pars=pars,
            total_pop=total_pop,
            **kwargs,
        )
