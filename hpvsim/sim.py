"""HPVsim convenience Sim wrapper.

``hpv.Sim(location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'])`` instantiates
the default stack — one HPV disease module per genotype, multi-layer
SexualNetwork, ss.Births + ss.Deaths + AgeMigration demographics, ss.People
with location-specific age pyramid, plus a CrossImmunity connector and an
HPVTotal analyzer — and forwards to ``ss.Sim``.

``connectors=`` and ``analyzers=`` are **append**, not override: user-supplied
modules are added after the auto-defaults (CrossImmunity, the _ExclusiveSeeder
when ``init_seeding='exclusive'``, HPVTotal, and — when an ``hpv.HIV`` disease
is present in ``diseases=`` — an ``hpv_hiv_connector`` appended after
CrossImmunity and a ``HIVStratifiedResults`` analyzer). To replace the
auto-defaults entirely, drop down to vanilla ``ss.Sim``.

Other slots (``diseases``, ``networks``, ``demographics``, ``people``) retain
override semantics. ``diseases=`` is mutually exclusive with ``genotypes=``.

The final sim year is ``stop`` (Starsim's name). ``end`` is accepted as a
deprecated v2 alias — if supplied it overrides ``stop`` and emits a warning.

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
import starsim as ss
import hpvsim as hpv

# Explicit imports for symbols referenced by bare name in this module (the HIV
# wiring and disease/connector partition use bare names; the genotype/seeding
# helpers use the ``hpv.`` prefix — both styles coexist post-M08 merge).
from .cross_genotype import HPVTotal, CrossImmunity
from .data.country import load_country
from .demographics import AgeMigration, AnnualBirths, Births
from .hiv import HIV, hpv_hiv_connector, HIVStratifiedResults
from .hpv import HPV, _normalize_genotype
from .network import SexualNetwork
from .seeding import _ExclusiveSeeder


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location=None, genotypes=None, genotype_pars=None,
                 init_seeding='exclusive', init_hpv_dist=None,
                 n_agents=10_000, start=1990, stop=2060, dt=0.25,
                 total_pop=None, ms_agent_ratio=1, pars=None, v2_compat_demographics=False,
                 end=None, datafolder=None, **kwargs):
        # Legacy alias: HPVsim v2 used ``end`` for the final sim year; Starsim
        # renamed it ``stop``. Accept ``end`` so v2 scripts keep running, but
        # nudge toward ``stop``. If given, ``end`` wins over ``stop``.
        if end is not None:
            ss.warn("hpv.Sim: `end` is a deprecated alias for `stop`; use `stop=` instead.")
            stop = end
        # Dispatch on location: string -> load country data (bundled or
        # datafolder); None -> uniform ages, no vitals, location-agnostic
        # network (stisim pattern: no location => no auto pop scaling).
        if location is None:
            country = None
            ss.warn(
                'hpv.Sim: no location supplied; using uniform ages 0-60, no '
                'births/deaths/migration, location-agnostic sexual-network '
                'defaults, pop_scale=1. Pass location= or demographics= to '
                'model a real population, or call hpv.demo() for a canonical '
                'Nigeria sim.'
            )
        else:
            country = hpv.load_country(location, year=int(start), datafolder=datafolder)
            if total_pop is None:
                total_pop = country['age_data']['value'].sum()
        people = kwargs.pop('people', None)
        if people is None:
            age_data = country['age_data'] if country is not None else None
            people = ss.People(n_agents, age_data=age_data,
                               extra_states=[ss.BoolArr('fine', default=False)])

        user_diseases = kwargs.pop('diseases', None) or []
        user_connectors = kwargs.pop('connectors', None) or []
        user_analyzers = kwargs.pop('analyzers', None) or []

        # Partition diseases= by type: HPV genotype modules are built from
        # genotypes= (or supplied directly as HPV instances); any non-HPV
        # disease (e.g. hpv.HIV) is merged in alongside them.
        hpv_instances = [d for d in user_diseases if isinstance(d, HPV)]
        other_diseases = [d for d in user_diseases if not isinstance(d, HPV)]

        if hpv_instances and genotypes is not None:
            raise ValueError(
                'Specify HPV via genotypes= or HPV instances in diseases=, not both.'
            )

        if init_seeding not in ('exclusive', 'independent'):
            raise ValueError(
                f"init_seeding must be 'exclusive' or 'independent'; got {init_seeding!r}"
            )

        # CrossImmunity runs first each step (it overwrites rel_sus). A user
        # may supply a configured CrossImmunity (e.g. a calibrated rel_sev
        # severity scaler); honor it at the front rather than auto-adding a
        # second default one. Any other user connectors keep their order after
        # the auto chain (CrossImmunity -> seeder -> hpv_hiv_connector).
        user_cross = [c for c in user_connectors if isinstance(c, CrossImmunity)]
        user_connectors = [c for c in user_connectors
                           if not isinstance(c, CrossImmunity)]
        auto_connectors = [user_cross[0] if user_cross else CrossImmunity()]

        if hpv_instances:
            hpv_diseases = hpv_instances  # override path; seeder skipped (user-supplied HPV manage their own init_prev)
        else:
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

            hpv_diseases = [HPV(genotype=k, ms_agent_ratio=ms_agent_ratio,
                                **gpars_overrides.get(k, {})) for k in keys]
            if init_seeding == 'exclusive':
                # 'exclusive': one Bernoulli per agent for any HPV, then one
                # genotype per infected agent via the seeder's per-genotype callback.
                # 'independent' is the no-op path — each HPV's per-genotype init_prev
                # curve drives its own seeding independently.
                self._seeder = hpv._ExclusiveSeeder(
                    genotype_keys=keys, init_hpv_dist=init_hpv_dist
                )
                for d, k in zip(hpv_diseases, keys):
                    d.pars.init_prev = ss.bernoulli(p=self._seeder.for_genotype(k))
                # Register so the seeder's Dists go through the standard
                # define_pars -> init_pre -> init_dists lifecycle.
                auto_connectors.append(self._seeder)

        diseases = hpv_diseases + other_diseases
        auto_analyzers = [HPVTotal()]
        # If HIV is present, auto-wire its connector (appended last, so it runs
        # after CrossImmunity, which overwrites rel_sus each step; see
        # hpv_hiv_connector.init_pre) and its stratified-results analyzer.
        # A user may supply their own (e.g. an hpv_hiv_connector with calibrated
        # pars); in that case do NOT auto-add a second one, which would
        # double-apply the rel_sus multiply / double-count the results.
        if any(isinstance(d, HIV) for d in other_diseases):
            if not any(isinstance(c, hpv_hiv_connector) for c in user_connectors):
                auto_connectors.append(hpv_hiv_connector())
            if not any(isinstance(a, HIVStratifiedResults) for a in user_analyzers):
                auto_analyzers.append(HIVStratifiedResults())
        connectors = auto_connectors + user_connectors

        networks = kwargs.pop('networks', None)
        if networks is None:
            # Network calibration is location-agnostic (see hpv.NetworkPars);
            # country['network_pars'] is just NetworkPars() defaults.
            network_pars = country['network_pars'] if country is not None else {}
            networks = [hpv.SexualNetwork(**network_pars)]
        demographics = kwargs.pop('demographics', None)
        if demographics is None:
            if country is None:
                demographics = []  # bare Sim: no births/deaths/migration
            else:
                births_cls = hpv.AnnualBirths if v2_compat_demographics else hpv.Births
                # The raw mortality table spans ~1950-2100, but ss.Deaths only
                # ever queries years within the sim window (and otherwise
                # retains the whole frame). Trim to [start, stop] (+/-1yr pad
                # for boundary interpolation) so the module carries only what
                # it uses.
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

        analyzers = auto_analyzers + user_analyzers

        # AgeMigration.init_pre reads sim.location to load country data.
        self.location = location.lower() if location is not None else None
        # Stored for use in init() to discretize initial ages.
        self._v2_compat_demographics = v2_compat_demographics

        # Split pars into (a) starsim-recognized sim-level keys forwarded to
        # super() so pre-init state (People/timeline) picks them up, and
        # (b) hpv module-level keys (per-genotype pars, network flat pars,
        # connector pars) routed after super().__init__ via hpv.route_pars.
        sim_pars, mod_pars = None, None
        if pars:
            sim_par_keys = set(hpv.SimPars().keys())
            sim_pars = {k: v for k, v in pars.items() if k in sim_par_keys}
            mod_pars = {k: v for k, v in pars.items() if k not in sim_par_keys}

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
            pars=sim_pars,
            total_pop=total_pop,
            **kwargs,
        )
        if mod_pars:
            from .parameters import route_pars
            route_pars(self, mod_pars, verbose=False)

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
