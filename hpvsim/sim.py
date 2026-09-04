"""HPVsim convenience Sim wrapper.

``hpv.Sim(location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'])`` instantiates
the default stack — one HPV disease module per genotype, multi-layer
SexualNetwork, ss.Births + ss.Deaths + AgeMigration demographics, ss.People
with location-specific age pyramid, plus a CrossImmunity connector and an
HPVTotal analyzer — and forwards to ``ss.Sim``.

``connectors=`` and ``analyzers=`` are **append**, not override: user-supplied
modules are added after the auto-defaults (CrossImmunity, the
_ExclusiveSeeder when ``init_seeding='exclusive'``, and the HPVTotal
analyzer). To replace the auto-defaults entirely, drop down to vanilla
``ss.Sim``.

Other slots (``diseases``, ``networks``, ``demographics``, ``people``) retain
override semantics. ``diseases=`` is mutually exclusive with ``genotypes=``.

``model_hiv=True`` (or ``'incidence'``/``'transmission'``) adds HIV
co-infection: ``'incidence'`` (the default under ``True``) imposes a
per-(age,sex,year) incidence curve directly (``hpv.HIV_incidence``) plus a
coverage-based ``sti.ART`` intervention (ART data is mandatory for this mode);
``'transmission'`` drives HIV via network transmission instead
(``hpv.HIV_transmit``), with no auto-added ART. ``hiv_data=`` supplies the
input data (a folder path, see ``hpv.data.load_hiv_data``, or a dict with
``{'incidence', 'art_coverage', 'init_prev'}``); ``hiv_pars=`` overrides the
constructed HIV disease's pars (e.g. ``rel_sus_lo``, ``beta_m2f``). Mutually
exclusive with supplying your own HIV-family disease in ``diseases=`` — build
it yourself via ``hpv.HIV_transmit``/``hpv.HIV_incidence`` in that case, or go
fully manual with a vanilla ``stisim`` ``sti.HIV`` (no HPV-modulation effects
in that case — see ``hpv.HIV``'s docstring).

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
import sciris as sc
import starsim as ss
import hpvsim as hpv

# stisim and the HIV classes are imported lazily in __init__ (optional dependency).
from . import misc
from .cross_genotype import HPVTotal, CrossImmunity
from .data.country import load_country
from .demographics import AgeMigration, AnnualBirths, Births
from .hpv import HPV, _normalize_genotype
from .network import SexualNetwork
from .seeding import _ExclusiveSeeder


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location=None, genotypes=None, genotype_pars=None,
                 init_seeding='exclusive', init_hpv_dist=None,
                 n_agents=10_000, start=1990, stop=2060, dt=0.25,
                 total_pop=None, ms_agent_ratio=1, pars=None, v2_compat_demographics=False,
                 end=None, datafolder=None, model_hiv=None, hiv_data=None, hiv_pars=None,
                 nw_pars=None, imm_pars=None,
                 **kwargs):
        # Legacy v2 alias; ``end`` wins over ``stop`` when supplied.
        if end is not None:
            ss.warn("hpv.Sim: `end` is a deprecated alias for `stop`; use `stop=` instead.")
            stop = end
        # location is a construction-time argument, consumed below before the
        # sim's parameter set exists, so it can't be routed like a normal par.
        # Accept it from pars= anyway (pars= is the documented way to pass
        # parameters) by intercepting it here; a bare location= wins, matching
        # the sc.mergedicts(pars, kwargs) precedence used further down.
        pars = sc.mergedicts(pars)  # Copy, so the caller's dict isn't mutated.
        location = sc.ifelse(location, pars.pop('location', None))
        # location=None: uniform ages, no vitals, no auto pop scaling.
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
        copy_inputs = kwargs.pop('copy_inputs', True)
        sim_data = kwargs.pop('data', None)
        if people is None:
            age_data = country['age_data'] if country is not None else None
            people = ss.People(n_agents, age_data=age_data,
                               extra_states=[ss.BoolArr('fine', default=False)])

        user_diseases = sc.tolist(kwargs.pop('diseases', None))
        user_connectors = sc.tolist(kwargs.pop('connectors', None))
        user_analyzers = sc.tolist(kwargs.pop('analyzers', None))
        user_interventions = sc.tolist(kwargs.pop('interventions', None))

        # Partition diseases=: HPV instances vs everything else (e.g. HIV).
        hpv_instances = [d for d in user_diseases if isinstance(d, HPV)]
        other_diseases = [d for d in user_diseases if not isinstance(d, HPV)]

        if hpv_instances and genotypes is not None:
            raise ValueError(
                'Specify HPV via genotypes= or HPV instances in diseases=, not both.'
            )

        # Autoconstruct HIV and ART if model_hiv=True.
        auto_interventions = []
        if model_hiv:
            sti = misc.require_stisim()
            from .hiv import HIV_incidence, HIV_transmit
            if any(isinstance(d, sti.HIV) for d in other_diseases):
                raise ValueError(
                    'model_hiv= is mutually exclusive with supplying your own HIV '
                    'disease in diseases= -- use diseases= alone in that case.'
                )
            mode = 'incidence' if model_hiv is True else model_hiv
            if mode not in ('incidence', 'transmission'):
                raise ValueError(f"model_hiv must be True, 'incidence', or 'transmission'; got {model_hiv!r}")
            data = hiv_data
            if data is not None and not isinstance(data, dict):
                from .data.hiv import load_hiv_data
                data = load_hiv_data(data)
            if mode == 'incidence':
                # Require both incidence and ART.
                missing = [k for k in ('incidence', 'art_coverage') if not data or k not in data]
                if missing:
                    raise ValueError(
                        f"model_hiv='incidence' (or True) requires hiv_data with {missing}"
                    )
                hiv_disease = HIV_incidence(incidence=data['incidence'],
                                             init_prev_data=data.get('init_prev'), pars=hiv_pars)
                if not any(isinstance(iv, sti.ART) for iv in user_interventions):
                    from .data.hiv import reshape_art_coverage
                    auto_interventions.append(sti.ART(coverage=reshape_art_coverage(data['art_coverage'])))
            else:
                hiv_disease = HIV_transmit(init_prev_data=(data.get('init_prev') if data else None),
                                            pars=hiv_pars)
            other_diseases = other_diseases + [hiv_disease]

        if init_seeding not in ('exclusive', 'independent'):
            raise ValueError(
                f"init_seeding must be 'exclusive' or 'independent'; got {init_seeding!r}"
            )

        # Combine user-provided connectors with the CrossImmunity default.
        user_cross = [c for c in user_connectors if isinstance(c, CrossImmunity)]
        user_connectors = [c for c in user_connectors
                           if not isinstance(c, CrossImmunity)]
        if user_cross and imm_pars:
            raise ValueError(
                'imm_pars= is mutually exclusive with supplying your own '
                'CrossImmunity in connectors= -- pass pars= to your instance directly.'
            )
        auto_connectors = [user_cross[0] if user_cross else CrossImmunity(pars=imm_pars)]

        if hpv_instances:
            hpv_diseases = hpv_instances  # override path; these manage their own init_prev
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
                # One Bernoulli per agent for any HPV, then one genotype each.
                self._seeder = hpv._ExclusiveSeeder(
                    genotype_keys=keys, init_hpv_dist=init_hpv_dist
                )
                for d, k in zip(hpv_diseases, keys):
                    d.pars.init_prev = ss.bernoulli(p=self._seeder.for_genotype(k))
                # Registered so the seeder's Dists get the standard init lifecycle.
                auto_connectors.append(self._seeder)

        diseases = hpv_diseases + other_diseases
        auto_analyzers = [HPVTotal()]
        connectors = auto_connectors + user_connectors

        networks = kwargs.pop('networks', None)
        if networks is None:
            # Network calibration is location-agnostic; these are NetworkPars() defaults.
            network_pars = country['network_pars'] if country is not None else {}
            if nw_pars:
                network_pars = sc.mergedicts(network_pars, nw_pars)
            networks = [hpv.SexualNetwork(**network_pars)]
        elif nw_pars:
            raise ValueError(
                'nw_pars= is mutually exclusive with supplying your own '
                'networks= -- pass pars= to your instance directly.'
            )
        demographics = kwargs.pop('demographics', None)
        if demographics is None:
            if country is None:
                demographics = []  # bare Sim: no births/deaths/migration
            else:
                births_cls = hpv.AnnualBirths if v2_compat_demographics else hpv.Births
                # Trim the ~1950-2100 table to the sim window (+/-1yr interp pad).
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
        interventions = auto_interventions + user_interventions

        # AgeMigration.init_pre reads sim.location to load country data.
        self.location = location.lower() if location is not None else None
        # Stored for use in init() to discretize initial ages.
        self._v2_compat_demographics = v2_compat_demographics

        # Sim-level keys go to super() so pre-init state picks them up; module-level
        # keys are routed after via route_pars. pars= and bare kwargs are equivalent.
        merged = sc.mergedicts(pars, kwargs)
        kwargs = {}
        sim_par_keys = set(hpv.SimPars().keys()) - {'location'}
        sim_pars = {k: v for k, v in merged.items() if k in sim_par_keys}
        mod_pars = {k: v for k, v in merged.items() if k not in sim_par_keys}

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
            interventions=interventions,
            pars=sim_pars,
            total_pop=total_pop,
            copy_inputs=copy_inputs,
            data=sim_data,
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
