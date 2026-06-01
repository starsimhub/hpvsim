"""HPV-specific demographic modules.

``AgeMigration`` pins the agent age pyramid to a target population trajectory.
Sits alongside ``ss.Births`` and ``ss.Deaths`` in ``hpv.Sim``'s default
demographics list.

Algorithm — once per year:
  1. Look up the target age pyramid for the current year (pop_by_age).
  2. Compute ``scale = sim.n_agents / pop_at_sim_start`` (from pop_total).
  3. For each (sex, integer age):
       count_sim    = alive sim agents at this age and sex
       count_target = target_pyramid[sex][age] * scale
       diff         = round(count_target - count_sim)
       if diff > 0:  add ``diff`` immigrants at this age (HPV-naive)
       if diff < 0:  weight-pick ``|diff|`` agents at this age and request
                     their removal (treated as emigration)

Annual cadence is enforced by ``dt=ss.year`` in ``__init__``: ``ss.Loop``
only fires ``step()`` at times in this module's ``t.tvec``, so it runs once
per integer year regardless of sim.dt.

``AnnualBirths`` uses the same annual-cadence trick to match v2's ``add_births``
convention: births fire once per calendar year in a single pulse.
"""
import numpy as np
import pandas as pd
import starsim as ss


__all__ = ['AgeMigration', 'AnnualBirths', 'Level0Births', 'Level0Deaths',
           'Level0People']


class Level0Deaths(ss.Deaths):
    """``ss.Deaths`` whose reported death TALLY is scale-weighted.

    Background deaths still occur per-agent (a multiscale fine agent faces the
    same age-specific mortality — that competing risk is correct and intended);
    only the ``deaths.new`` count is corrected so a fine agent at scale
    ``1/ratio`` contributes its people-space weight instead of a full body
    (otherwise it inflates ~+17% at ratio=12). Re-implements ``ss.Deaths.step``
    verbatim except for the scale-weighted tally, so it draws the identical
    mortality Bernoulli and is bit-identical to ``ss.Deaths`` when all
    scale==1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register as 'deaths' so the mortality dist (_p_death) is seeded
        # identically to ss.Deaths (seed derives from module name) and the
        # results key stays 'deaths'; the class name 'level0deaths' would
        # otherwise shift the stream and break ms_agent_ratio=1 bit-identity.
        self.name = 'deaths'

    def step(self):
        p_death = self.make_p_death()
        self._p_death.set(p=p_death)
        death_uids = self._p_death.filter()
        self.sim.people.request_death(death_uids)
        self.n_deaths = (float(np.asarray(self.sim.people.scale[death_uids]).sum())
                         if len(death_uids) else 0.0)
        return self.n_deaths


class Level0People(ss.People):
    """``ss.People`` whose sim-level demographic body counts are scale-weighted.

    ``People.update_results`` records ``n_alive``, ``new_deaths``,
    ``new_emigrants`` and ``cum_deaths`` as RAW agent counts; multiscale fine
    agents (scale ``1/ratio``) are then counted as whole bodies, inflating these
    sim-level results (measured ~+15% n_alive, ~+25% new_deaths at ratio=12).
    Recompute them in people-space (scale-weighted). Cancer-specific deaths
    (``HPV.new_cancer_deaths``) and the per-genotype epidemiology are corrected
    elsewhere; this covers the all-cause demographic counters.

    Early-returns (leaving the base raw counts untouched) when no agent carries
    a sub-unit scale, so ms_agent_ratio=1 is bit-identical to ``ss.People``.
    """

    def update_results(self):
        super().update_results()
        scale = np.asarray(self.scale)  # .values, aligned to auids
        if (scale == 1.0).all():
            return  # no multiscale agents -> base raw counts are correct
        ti = self.sim.ti
        res = self.sim.results
        res.n_alive[ti] = float((scale * np.asarray(self.alive)).sum())
        res.new_deaths[ti] = float((scale * (np.asarray(self.ti_dead) == ti)).sum())
        res.new_emigrants[ti] = float((scale * (np.asarray(self.ti_removed) == ti)).sum())
        res.cum_deaths[ti] = float(np.sum(res.new_deaths[:ti]))
        return


class _Level0BirthsMixin:
    """Restrict births to ``level0`` (non-fine) bodies, mirroring v2's
    ``this_birth_rate * n_alive_level0`` (``_v2_legacy/people.py:782``).

    ``ss.Births`` realizes births as a per-agent Bernoulli over EVERY alive
    agent. Under multiscale the population also contains fine cancer sub-
    resolution agents (``multiscale_fine``); letting each give a full birth
    inflates the coarse population and transmission. Demographics must count
    BODIES excluding fine agents (v2 ``level0``), NOT scale-weight — a shrunk
    cancer original is 1/ratio of cancer mass but still ONE reproductive body,
    and a cancer split conserves cancer mass, not population, so a scale-weighted
    birth count would be wrong.

    Implementation: drop fine parents from the base's candidate births. The base
    draw is slot-keyed, so non-fine agents' birth outcomes are unchanged by the
    presence of fine agents; we only remove the fine winners. No extra RNG
    stream, so ``ms_agent_ratio=1`` (no fine agents) is bit-identical to the base.
    """

    def get_births(self):
        birth_uids = super().get_births()
        if not len(birth_uids):
            return birth_uids
        from .hpv import multiscale_fine_for  # local import avoids import cycle
        fine = multiscale_fine_for(self.sim, birth_uids)
        return birth_uids[~fine] if fine.any() else birth_uids


class Level0Births(_Level0BirthsMixin, ss.Births):
    """``ss.Births`` that counts only level0 (non-fine) bodies — multiscale-safe
    (see ``_Level0BirthsMixin``); bit-identical to ``ss.Births`` with no fine
    agents present. hpv.Sim's default births class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register under the same module name as ``ss.Births`` so the births
        # ``ss.bernoulli`` is seeded identically (starsim derives the dist seed
        # from the module name). Without this the class swap shifts the birth
        # RNG stream and ms_agent_ratio=1 would NOT be bit-identical to the
        # pre-feature ss.Births default.
        self.name = 'births'


class AgeMigration(ss.Demographics):
    """Age-pyramid pinning to a target population trajectory.

    Fires once per integer year and forces the sim's age x sex composition
    to match the target pyramid by adding immigrants (HPV-naive) or
    requesting removal of emigrants.

    Data can be supplied explicitly via ``pop_total`` / ``pop_by_age``,
    or loaded automatically from the sim's ``location`` parameter.

    Args:
        pop_total  (DataFrame): columns [year, pop_size]. If None, loaded from country data.
        pop_by_age (DataFrame): columns [year, age, male, female]. If None, loaded from
            country data.
    """

    def __init__(self, pars=None, pop_total=None, pop_by_age=None,
                 v2_compat=False, **kwargs):
        # dt=ss.year sets this module's Timeline to annual; ss.Loop only
        # calls step() at times in mod.t.tvec, so it fires once per year
        # regardless of sim.dt.
        super().__init__(dt=ss.year)
        self.update_pars(pars, **kwargs)
        self._pop_total = pop_total                 # [year, pop_size] DataFrame; sets _scale and the data-year window.
        self._pop_by_age = pop_by_age               # [year, age, male, female] DataFrame; the per-step target pyramid.
        # Sim agents per real-world person at sim.start (n_agents / pop_total
        # at start year). Used to scale target_counts each step so the pyramid
        # is matched in agent-space, not person-space.
        self._scale = None
        self._n_imm = 0                             # Immigrants added this step (for results.new_immigrants).
        self._n_emi = 0                             # Emigrants requested this step (for results.new_emigrants).
        self._pop_by_year = None                    # {year: per-year pyramid DataFrame}, built in init_pre.
        # CRN-safe emigrant selection — domain set per call.
        self._emi_select = ss.choice(replace=False)
        # Per-agent Bernoulli for emigrating fine multiscale agents at the
        # band's coarse emigration rate (unbiased; avoids round() under-
        # emigration). Only used when fine agents are present (ms_agent_ratio>1).
        self._fine_emi = ss.bernoulli(p=0.0)
        # Sub-year age jitter for incoming immigrants. Without this, every
        # "year-N" immigrant arrives at exactly age N.0 and the cohort ages
        # in lockstep through age bins, producing discrete pyramid steps
        # rather than smooth transitions. Jittering by uniform(0, 1) means
        # year-N immigrants are spread across [N, N+1) within the year.
        # Skipped when v2_compat=True to match v2's discrete annual-cohort
        # structure (immigrants land at exact integer ages, as add_births does).
        self._age_jitter = ss.uniform(low=0.0, high=1.0)
        # v2_compat: when True, skip age jitter so immigrants land at exact
        # integer ages, matching v2's add_births convention. Pair with
        # AnnualBirths to align both channels to v2's discrete-cohort structure.
        self._v2_compat = v2_compat
        return

    # ---------------------------------------------------------------------- #
    # Lifecycle hooks                                                          #
    # ---------------------------------------------------------------------- #

    def init_pre(self, sim):
        super().init_pre(sim)

        # Resolve sim start year once. ss.years supports float() but not
        # int() directly, so cast through float. Used both for load_country
        # (so age_data samples at the right year) and for _scale below.
        sim_start = sim.pars.start
        sim_start_year = float(
            sim_start.year if hasattr(sim_start, 'year') else sim_start
        )

        # Pull from country data if not explicitly supplied.
        if self._pop_total is None or self._pop_by_age is None:
            from .data.country import load_country
            # hpv.Sim stores location on the sim; fall back to pars if a
            # caller wired it there instead.
            location = (
                getattr(sim, 'location', None)
                or getattr(sim.pars, 'location', None)
            )
            if location is None:
                raise AttributeError(
                    'AgeMigration could not find a location on the sim. '
                    'Either pass pop_total/pop_by_age explicitly, or use '
                    'hpv.Sim which stores sim.location.'
                )
            cd = load_country(location, year=int(sim_start_year))
            if self._pop_total is None:
                self._pop_total = cd['pop_total']
            if self._pop_by_age is None:
                self._pop_by_age = cd['pop_by_age']

        # Scale factor: n_agents / data_population_at_sim_start.
        pt = self._pop_total
        data_pop_at_start = float(
            np.interp(sim_start_year, pt['year'].values, pt['pop_size'].values)
        )
        self._scale = float(sim.pars.n_agents) / data_pop_at_start

        # Group pop_by_age once into a {year: DataFrame} dict so step() can
        # do an O(1) lookup instead of an O(rows) mask + sort every year.
        self._pop_by_year = {
            int(y): grp.sort_values('age')
            for y, grp in self._pop_by_age.groupby('year')
        }
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_immigrants', dtype=int, scale=True,
                      label='New immigrants', auto_plot=False),
            ss.Result('new_emigrants', dtype=int, scale=True,
                      label='New emigrants', auto_plot=False),
        )
        return

    # ---------------------------------------------------------------------- #
    # Step                                                                    #
    # ---------------------------------------------------------------------- #

    def step(self):
        """Pin age x sex pyramid to the target for the current year.

        Called once per integer year (module dt=ss.year gates firing via Loop).
        """
        sim = self.sim
        year = int(sim.t.now('year'))

        self._n_imm = 0
        self._n_emi = 0
        pat_year = self._pop_by_year.get(year)
        if pat_year is None:
            return

        ages_data = pat_year['age'].values.astype(int)

        people = sim.people
        # Snapshot alive UIDs and their attributes at the start of the step.
        # This ensures the boolean masks remain aligned with the UID list even
        # as immigrants are added during the loop.
        snap_uids = people.auids.copy()
        # age is float32 — cast to int for integer-bin lookup; female is already bool.
        ages = people.age[snap_uids].astype(int)
        female = people.female[snap_uids]
        # Multiscale-fine mask (snapshot-aligned). The pyramid is pinned in
        # BODY-count space excluding fine sub-resolution agents (v2's level0
        # count): a coarse agent — including a shrunk cancer original, which
        # stays a real body — counts as 1; a spawned fine cancer agent (level1)
        # counts as 0. NOT scale-weighted: a cancer split conserves cancer mass,
        # not population (the non-cancer sub-resolutions are never materialized),
        # so a scale-weighted count would under-fill the pyramid and over-import
        # real agents. Counting fine agents as whole bodies instead over-fills
        # and emigrates real agents (~-23% transmission). Excluding them — and
        # never emigrating them — is correct and is bit-identical at
        # ms_agent_ratio=1 (no fine agents -> full count). See _multiscale_split.
        from .hpv import multiscale_fine_for  # local import avoids import cycle
        fine = multiscale_fine_for(sim, snap_uids)

        n_imm_total = 0
        n_emi_total = 0

        # Accumulate per-(age, sex) immigrant attributes across the inner loop;
        # concatenated and applied to People in one ``people.grow`` call below.
        # Per-band arrays go in chunks instead of a single grow-per-band so we
        # only pay the People-resize cost once per step.
        imm_age_chunks = []
        imm_female_chunks = []

        for sex_label, sex_mask in (
            ('male',   ~female),
            ('female',  female),
        ):
            sex_is_female = (sex_label == 'female')
            target_counts = pat_year[sex_label].values * self._scale

            for age, target in zip(ages_data, target_counts):
                in_band = sex_mask & (ages == age)
                # Body count EXCLUDING fine sub-resolution agents (v2 level0).
                count_sim = int((in_band & ~fine).sum())
                diff = int(round(target - count_sim))

                if diff > 0:
                    # Under-target: queue ``diff`` immigrants for this band.
                    imm_age_chunks.append(np.full(diff, age, dtype=float))
                    imm_female_chunks.append(np.full(diff, sex_is_female, dtype=bool))
                    n_imm_total += diff
                elif diff < 0:
                    # Over-target: emigrate ``-diff`` COARSE agents from this
                    # band to hit the level0 target.
                    band_uids = snap_uids[in_band & ~fine]
                    self._emigrate(band_uids, n=-diff)
                    n_emi_total += -diff
                    # Emigrate fine sub-resolution agents at the SAME per-capita
                    # rate via a per-agent Bernoulli (self._fine_emi). Without
                    # this, fine cancer agents are never emigrated while single-
                    # scale cancer agents in over-target (old) bands are — so the
                    # fine agents over-survive to cancer onset and the cancer
                    # count is biased high (measured ~+16%). At ms_agent_ratio=1
                    # band_fine is always empty, so _fine_emi is never drawn and
                    # single-scale stays bit-identical (the extra dist does not
                    # shift the other dists' name-derived seeds).
                    band_fine = snap_uids[in_band & fine]
                    if len(band_fine) and count_sim > 0:
                        r = min((-diff) / count_sim, 1.0)
                        self._fine_emi.set(p=r)
                        fine_emig = self._fine_emi.filter(band_fine)
                        if len(fine_emig):
                            self.sim.people.request_removal(fine_emig)

        if n_imm_total > 0:
            # Single People.grow + write per attribute, sized to the total
            # immigration across all (age, sex) bands.
            new_uids = people.grow(n_imm_total)
            ages_at_arrival = np.concatenate(imm_age_chunks)
            # Spread each "year-N" immigrant uniformly across [N, N+1) so the
            # cohort doesn't all transition age bins on the same tick. Uses
            # the per-module CRN dist; ages_at_arrival is the integer
            # lower-bound of each immigrant's year band.
            # Skipped when v2_compat is on — v2's add_births places immigrants
            # at exact integer ages to match the discrete annual birth cohort
            # structure.
            if not self._v2_compat:
                ages_at_arrival = ages_at_arrival + self._age_jitter.rvs(new_uids)
            people.age[new_uids] = ages_at_arrival
            people.female[new_uids] = np.concatenate(imm_female_chunks)

        self._n_imm = n_imm_total
        self._n_emi = n_emi_total
        return

    def update_results(self):
        """Store immigrant/emigrant counts for this timestep."""
        self.results.new_immigrants[self.ti] = self._n_imm
        self.results.new_emigrants[self.ti] = self._n_emi
        return

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _emigrate(self, band_uids, n):
        """Remove n agents from band_uids via ``request_removal``.

        ``request_removal`` is the emigration equivalent of ``request_death``
        — the agent leaves the simulation without being recorded as dead.

        Args:
            band_uids: ss.uids of alive agents in the age x sex band.
            n: number of agents to remove.
        """
        if n <= 0 or len(band_uids) == 0:
            return
        n_pick = min(int(n), len(band_uids))
        self._emi_select.set(a=band_uids)
        chosen_uids = ss.uids(self._emi_select.rvs(n_pick))
        self.sim.people.request_removal(chosen_uids)
        return


class AnnualBirths(_Level0BirthsMixin, ss.Births):
    """Annual-pulse births matching v2's ``add_births`` convention.

    Also restricts births to level0 (non-fine) bodies via ``_Level0BirthsMixin``
    so it is multiscale-safe; bit-identical to plain annual-pulse births when no
    fine agents are present.

    Standard ``ss.Births`` distributes births evenly across all steps. This
    subclass fires a single birth pulse at each integer year boundary by giving
    the module an annual Timeline (``dt=ss.year``). ``ss.Loop`` then only calls
    ``step()`` once per calendar year, and because ``self.t.dt`` equals
    ``ss.years(1)``, ``get_births()`` naturally computes the full annual
    probability rather than a per-quarter fraction.

    With dt=0.25 (4 steps per year) the total number of births over any full
    year is statistically identical between ``ss.Births`` (4 steps × ¼ rate)
    and ``AnnualBirths`` (1 step × full rate). Only the *timing* changes: every
    year's cohort is born on the same calendar step instead of being spread
    across four quarterly sub-cohorts.

    This matches v2's ``add_births`` logic: ``dt_demog=1.0``, fired every
    ``update_freq = int(dt_demog / dt) = 4`` steps at annual boundaries, using
    the full annual crude birth rate scaled by ``dt_demog``.

    Opt-in only — default ``ss.Births`` behavior (continuous births) is
    unchanged. Activate by passing ``demographics=[hpv.AnnualBirths(...), ...]``
    to ``hpv.Sim``.

    Args:
        birth_rate: birth rate data passed through to ``ss.Births``.
        kwargs: forwarded to ``ss.Births.__init__``.

    Example::

        import hpvsim as hpv
        import starsim as ss
        sim = hpv.Sim(
            location='nigeria',
            demographics=[
                hpv.AnnualBirths(),
                ss.Deaths(),
                hpv.AgeMigration(),
            ],
        )
        sim.run()
    """

    def __init__(self, pars=None, **kwargs):
        # Inject dt=ss.year into the pars dict so ss.Module.__init__ receives
        # it via update_pars → Timeline. This sets the module's own Timeline
        # to annual cadence: ss.Loop fires step() only at integer-year
        # boundaries, and self.t.dt == ss.years(1) causes get_births() to
        # compute the full annual birth probability in one pulse.
        import sciris as sc
        pars = sc.mergedicts({'dt': ss.year}, pars)
        super().__init__(pars=pars, **kwargs)