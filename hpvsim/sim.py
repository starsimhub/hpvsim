"""HPVsim convenience Sim wrapper.

``hpv.Sim(location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'])`` instantiates
the default stack — one HPV disease module per genotype, multi-layer
SexualNetwork, ss.Births + ss.Deaths + AgeMigration demographics, ss.People
with location-specific age pyramid, plus a CrossImmunity connector and an
HPVTotal analyzer — and forwards to ``ss.Sim``.

``connectors=`` and ``analyzers=`` are **append**, not override: user-supplied
modules are added after the auto-defaults (CrossImmunity, the _ExclusiveSeeder
when ``init_seeding='exclusive'``, and HPVTotal). To replace the auto-defaults
entirely, drop down to vanilla ``ss.Sim``.

Other slots (``diseases``, ``networks``, ``demographics``, ``people``) retain
override semantics. ``diseases=`` is mutually exclusive with ``genotypes=``.

Kwargs:
  ``init_seeding`` (str, default ``'exclusive'``):
    ``'exclusive'`` — one Bernoulli per agent using the hpv16 age-banded
    curve as the total HPV prevalence, then exactly one genotype assigned per
    infected agent. No co-infection at initialisation.
    ``'independent'`` — each genotype draws from its own per-genotype
    init_prev curve independently; co-infection at initialisation is possible.

  ``v2_compat_demographics`` (bool, default ``False``):
    When True, activates three v2-compatible demographic conventions:

    1. **Annual-pulse births.** Swaps ``ss.Births`` for ``hpv.AnnualBirths``
       so every year's birth cohort is released as a single pulse at the
       integer-year boundary, matching v2's ``add_births`` / ``dt_demog=1``
       logic.
    2. **Migration jitter disabled.** Passes ``v2_compat=True`` to
       ``AgeMigration`` so immigrants land at exact integer ages (no
       uniform [N, N+1) jitter), matching v2's discrete-cohort structure.
    3. **Initial population age discretization.** After ``ss.People.init_vals``
       samples continuous ages from the UN year-band histogram (each agent
       lands uniformly within its year bin), floors all initial ages to the
       nearest integer. This matches v2's convention of placing the starting
       cohort at exact integer ages.

    All three effects together ensure that every agent entering or starting
    in the sim has a discrete integer age, which aligns the eligibility
    window arithmetic for age-targeted interventions with v2's conventions.
    The default (False) retains v3's continuous-age behaviour.

  ``init_hpv_dist`` (dict or None, default ``None``):
    Only used when ``init_seeding='exclusive'``. If ``None``, genotype
    assignment is uniform across active genotypes. If a dict, keys must be
    the resolved canonical genotype names (e.g. ``{'hpv16': 0.6, 'hpv18':
    0.2, 'hi5': 0.1, 'ohr': 0.1}``) and values are weights (need not sum to
    1; normalised internally).
"""

import numpy as np
import starsim as ss

from .cross_genotype import HPVTotal, CrossImmunity
from .data.country import load_country
from .demographics import AgeMigration, AnnualBirths
from .hpv import HPV, _normalize_genotype
from .network import SexualNetwork
from .seeding import _ExclusiveSeeder


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location='nigeria', genotypes=None, genotype_pars=None,
                 init_seeding='exclusive', init_hpv_dist=None,
                 n_agents=10_000, start=1990, stop=2060, dt=0.25,
                 total_pop=None, pars=None, v2_compat_demographics=False,
                 ms_agent_ratio=None,
                 **kwargs):
        # Pass start year so the age pyramid matches sim.start (loader
        # defaults to year 2000 with a materially different distribution).
        country = load_country(location, year=int(start))
        people = kwargs.pop('people', None)
        if people is None:
            people = ss.People(n_agents, age_data=country['age_data'])

        diseases = kwargs.pop('diseases', None)
        user_connectors = kwargs.pop('connectors', None) or []
        user_analyzers = kwargs.pop('analyzers', None) or []

        if diseases is not None and genotypes is not None:
            raise ValueError(
                'Pass diseases= OR genotypes=, not both.'
            )

        if init_seeding not in ('exclusive', 'independent'):
            raise ValueError(
                f"init_seeding must be 'exclusive' or 'independent'; got {init_seeding!r}"
            )

        auto_connectors = [CrossImmunity()]

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
            if ms_agent_ratio is not None:
                for d in diseases:
                    d.pars.ms_agent_ratio = int(ms_agent_ratio)
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
                # Register so the seeder's Dists go through the standard
                # define_pars -> init_pre -> init_dists lifecycle.
                auto_connectors.append(self._seeder)

        connectors = auto_connectors + user_connectors

        networks = kwargs.pop('networks', None)
        if networks is None:
            networks = [SexualNetwork(**country['network_pars'])]
        demographics = kwargs.pop('demographics', None)
        if demographics is None:
            births_cls = AnnualBirths if v2_compat_demographics else ss.Births
            demographics = [
                births_cls(birth_rate=country['birth_rate']),
                ss.Deaths(death_rate=country['death_rate']),
                AgeMigration(v2_compat=v2_compat_demographics),
            ]

        analyzers = [HPVTotal()] + user_analyzers

        # AgeMigration.init_pre reads sim.location to load country data.
        self.location = location.lower()
        # Stored for use in init() to discretize initial ages.
        self._v2_compat_demographics = v2_compat_demographics
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

    def init(self, **kwargs):
        """Initialize the sim, then discretize initial ages if v2_compat_demographics is set.

        ``ss.People.init_vals()`` samples ages continuously from the UN
        year-band histogram (each agent lands uniformly within its year bin,
        e.g. an agent in the "age 5" bin gets a float in [5, 6)). v2 placed
        agents at exact integer ages. When ``v2_compat_demographics=True``,
        floor all initial agent ages to integers after sampling so the
        starting cohort matches v2's discrete convention.

        Note: ``super().init()`` runs ``SexualNetwork.init_post``, which
        pre-forms one batch of partnerships using debut ages sampled against
        the continuous initial-age distribution. The integer-age floor below
        runs after that pre-form, so the very first pair graph reflects
        continuous ages while every subsequent step sees integer ages. The
        parity gate is statistical and absorbs this transient; v2 has an
        analogous one-off effect at the `make_contacts` step.
        """
        super().init(**kwargs)
        if self._v2_compat_demographics:
            uids = self.people.auids
            self.people.age.raw[uids] = np.floor(self.people.age.raw[uids])
        return self
