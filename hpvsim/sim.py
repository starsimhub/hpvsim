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
    Compatibility flag that forces discrete integer-age demographics. The
    default (False) continuous-age behaviour is preferred; this flag exists
    only to reproduce a bit-for-bit discrete-cohort convention and should
    not be the basis for new work.

    When True, activates three demographic conventions:

    1. **Annual-pulse births.** Swaps ``ss.Births`` for ``hpv.AnnualBirths``
       so every year's birth cohort is released as a single pulse at the
       integer-year boundary.
    2. **Migration jitter disabled.** Passes ``v2_compat=True`` to
       ``AgeMigration`` so immigrants land at exact integer ages (no
       uniform [N, N+1) jitter).
    3. **Initial population age discretization.** After ``ss.People.init_vals``
       samples continuous ages from the UN year-band histogram (each agent
       lands uniformly within its year bin), floors all initial ages to the
       nearest integer, placing the starting cohort at exact integer ages.

    All three effects together ensure that every agent entering or starting
    in the sim has a discrete integer age, which aligns the eligibility-window
    arithmetic for age-targeted interventions to integer boundaries. The
    default (False) retains the continuous-age behaviour.

  ``init_hpv_dist`` (dict or None, default ``None``):
    Only used when ``init_seeding='exclusive'``. If ``None``, genotype
    assignment is uniform across active genotypes. If a dict, keys must be
    the resolved canonical genotype names (e.g. ``{'hpv16': 0.6, 'hpv18':
    0.2, 'hi5': 0.1, 'ohr': 0.1}``) and values are weights (need not sum to
    1; normalised internally).
"""

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import hpvsim as hpv


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location='nigeria', genotypes=None, genotype_pars=None,
                 init_seeding='exclusive', init_hpv_dist=None,
                 n_agents=10_000, start=1990, stop=2060, dt=0.25,
                 total_pop=None, ms_agent_ratio=1, pars=None, v2_compat_demographics=False,
                 **kwargs):
        # Pass start year so the age pyramid matches sim.start (loader
        # defaults to year 2000 with a materially different distribution).
        country = hpv.load_country(location, year=int(start))
        people = kwargs.pop('people', None)
        if people is None:
            people = ss.People(n_agents, age_data=country['age_data'],
                               extra_states=[ss.BoolArr('fine', default=False)])

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

        auto_connectors = [hpv.CrossImmunity()]

        if diseases is None:
            # Default to single-genotype HPV16 if neither supplied.
            keys = (tuple(hpv._normalize_genotype(g) for g in genotypes)
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

            diseases = [hpv.HPV(genotype=k, ms_agent_ratio=ms_agent_ratio,
                            **gpars_overrides.get(k, {})) for k in keys]
            if init_seeding == 'exclusive':
                # 'exclusive': one Bernoulli per agent for any HPV, then one
                # genotype per infected agent via the seeder's per-genotype callback.
                # 'independent' is the no-op path — each HPV's per-genotype init_prev
                # curve drives its own seeding independently.
                self._seeder = hpv._ExclusiveSeeder(
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
            networks = [hpv.SexualNetwork(**country['network_pars'])]
        demographics = kwargs.pop('demographics', None)
        if demographics is None:
            births_cls = hpv.AnnualBirths if v2_compat_demographics else hpv.Births
            # The raw mortality table spans ~1950-2100, but ss.Deaths only ever
            # queries years within the sim window (and otherwise retains the
            # whole frame). Trim to [start, stop] (+/-1yr pad for boundary
            # interpolation) so the module carries only what it uses.
            death_rate = country['death_rate']
            death_rate = death_rate[
                (death_rate['Time'] >= int(start) - 1)
                & (death_rate['Time'] <= int(stop) + 1)
            ]
            demographics = [
                births_cls(birth_rate=country['birth_rate']),
                ss.Deaths(death_rate=death_rate),
                hpv.AgeMigration(v2_compat=v2_compat_demographics),
            ]

        analyzers = [hpv.HPVTotal()] + user_analyzers

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
        e.g. an agent in the "age 5" bin gets a float in [5, 6)). When
        ``v2_compat_demographics=True``, floor all initial agent ages to
        integers after sampling so the starting cohort lands at exact
        integer ages.

        Note: ``super().init()`` runs ``SexualNetwork.init_post``, which
        pre-forms one batch of partnerships using debut ages sampled against
        the continuous initial-age distribution. The integer-age floor below
        runs after that pre-form, so the very first pair graph reflects
        continuous ages while every subsequent step sees integer ages; this
        one-off transient at initialisation is negligible.
        """
        super().init(**kwargs)
        if self._v2_compat_demographics:
            uids = self.people.auids
            self.people.age.raw[uids] = np.floor(self.people.age.raw[uids])
        return self

    def shrink(self, inplace=True, full=True, size_limit=None, base_size=30, die=True):
        """Shrink the sim for saving; skips the per-module size check by default.

        Identical to ``ss.Sim.shrink`` except ``size_limit`` defaults to None
        rather than 1.0. The CrossImmunity connector and HPVTotal analyzer each
        hold references to the shared disease modules; starsim's per-module size
        budget counts those referenced modules against them and raises on a
        multi-genotype sim, even though ``sc.save`` serializes the disease
        modules once and the actual file is small (~1 MB). Dist and
        back-reference shrinking still run — only the (double-counting) size
        check is disabled. Pass ``size_limit=1.0`` to restore it.
        """
        return super().shrink(inplace=inplace, full=full, size_limit=size_limit,
                              base_size=base_size, die=die)

    def to_json(self, filename=None, keys=None, indent=2, verbose=False, **kwargs):
        """Export parameters and results as JSON.

        Works around an upstream Starsim bug: ``ss.Sim.to_json`` does
        ``self.to_df().to_dict()``, but Starsim's own ``Results.to_df`` returns
        an objdict of per-module DataFrames (not a single DataFrame) whenever a
        sim's modules span different time axes — so ``.to_dict()`` raises. This
        hits any mixed-timeline sim; HPVsim triggers it every run because
        ``AgeMigration`` steps annually while the disease modules step
        sub-annually. Here each per-module frame is converted to a dict (a plain
        DataFrame is still handled, for the single-timeline case). Remove this
        override once the upstream ``to_json`` handles the objdict return.

        Args:
            filename (str): if None, return a dict; else write JSON to this path.
            keys (str/list): any of 'pars', 'summary', 'results' (default: all).
            indent (int): JSON indentation when writing to file.
            kwargs (dict): passed to ``sc.jsonify``.
        """
        if keys is None:
            keys = ['pars', 'summary', 'results']
        keys = sc.promotetolist(keys)

        d = sc.objdict()
        for key in keys:
            if key in ('pars', 'parameters'):
                d.pars = self.pars.to_json()
            elif key == 'summary':
                d.summary = (dict(sc.dcp(self.summary)) if self.results_ready
                             else 'Summary not available (Sim has not yet been run)')
            elif key == 'results':
                df = self.to_df()
                if isinstance(df, pd.DataFrame):
                    d.results = df.to_dict()
                else:  # objdict of per-module DataFrames
                    d.results = {k: v.to_dict() for k, v in df.items()}
            else:  # pragma: no cover
                ss.warn(f'Could not convert "{key}" to JSON; continuing...')

        d = sc.jsonify(d, **kwargs)
        if filename is not None:
            sc.savejson(filename=filename, obj=d, indent=indent)
        return d
