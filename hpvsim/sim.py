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

import numpy as np
import starsim as ss

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


class Aggregate(ss.Analyzer):
    """Analyzer that pools per-genotype results into Sim-level *_any aggregates.

    Results are accessible at ``sim.results.aggregate``:
      - ``cum_infections_any`` — per-step sum of new_infections across genotypes,
        cumsum'd. Sum-of-flows: overcounts agents with co-infections.
      - ``cum_cancers_any`` — sum of per-genotype cum_cancers (no double-counting
        since cancer is attributed to a single genotype).
      - ``new_cancer_deaths_any`` — per-step sum of new_cancer_deaths.

    The analyzer is auto-added by ``hpv.Sim`` whenever HPV modules are present.
    ``step()`` captures per-step new_infections; ``finalize_results()`` assembles
    the cumulative aggregates using HPV disease results (available because
    analyzers finalize after disease modules in Starsim's finalization order).
    """

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('cum_infections_any', dtype=int,
                      label='Cumulative agents ever infected (any genotype)'),
            ss.Result('cum_cancers_any', dtype=int,
                      label='Cumulative cancers (any genotype)'),
            ss.Result('new_cancer_deaths_any', dtype=int,
                      label='New cancer deaths (any genotype)'),
        )

    def _hpvs(self):
        return [d for d in self.sim.diseases.values() if isinstance(d, HPV)]

    def step(self):
        """Capture per-step new_infections (needed before they could be overwritten)."""
        ti = self.sim.ti
        hpvs = self._hpvs()
        if not hpvs:
            return
        # Sum-of-flows across genotypes, not boolean-OR — overcounts co-infections.
        self.results['cum_infections_any'][ti] = sum(
            m.results.new_infections[ti] for m in hpvs
        )

    def finalize_results(self):
        """Assemble cumulative aggregates after HPV disease modules have finalized."""
        super().finalize_results()
        hpvs = self._hpvs()
        if not hpvs:
            return
        # Cumulative sum of the per-step counts captured in step().
        self.results['cum_infections_any'][:] = np.cumsum(
            np.asarray(self.results['cum_infections_any'])
        )
        # cum_cancers_any: sum across genotypes. HPV.finalize_results runs
        # before this analyzer, so cum_cancers is already populated.
        self.results['cum_cancers_any'][:] = sum(
            np.asarray(m.results.cum_cancers) for m in hpvs
        )
        self.results['new_cancer_deaths_any'][:] = sum(
            np.asarray(m.results.new_cancer_deaths) for m in hpvs
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
