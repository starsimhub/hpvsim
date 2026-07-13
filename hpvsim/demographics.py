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

``AnnualBirths`` uses the same annual-cadence trick to fire births once per
calendar year in a single pulse.
"""
import numpy as np
import pandas as pd
import starsim as ss


__all__ = ['AgeMigration', 'Births', 'AnnualBirths']


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
        # Per-band emigration hazard applied independently to fine multiscale
        # agents (excluded from the pyramid target). p set per band per call.
        self._fine_emi_bern = ss.bernoulli(p=0.0)
        # Sub-year age jitter for incoming immigrants. Without this, every
        # "year-N" immigrant arrives at exactly age N.0 and the cohort ages
        # in lockstep through age bins, producing discrete pyramid steps
        # rather than smooth transitions. Jittering by uniform(0, 1) means
        # year-N immigrants are spread across [N, N+1) within the year.
        # Skipped when v2_compat=True so immigrants land at exact integer ages.
        self._age_jitter = ss.uniform(low=0.0, high=1.0)
        # v2_compat: when True, skip age jitter so immigrants land at exact
        # integer ages. Pair with AnnualBirths for fully discrete integer-age
        # demographics.
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
        # Snapshot alive UIDs at the start of the step, EXCLUDING fine
        # multiscale agents: they are high-resolution cancer stand-ins, not
        # real bodies, so they neither count toward the age x sex pyramid
        # target nor are eligible to emigrate.
        # This ensures the boolean masks remain aligned with the UID list even
        # as immigrants are added during the loop.
        all_alive = people.auids.copy()
        if 'fine' in people.states:
            snap_uids = all_alive[~people.fine[all_alive]]
        else:
            snap_uids = all_alive
        # age is float32 — cast to int for integer-bin lookup; female is already bool.
        ages = people.age[snap_uids].astype(int)
        female = people.female[snap_uids]

        # Fine multiscale agents are excluded from the pyramid count/target
        # above (counting them as whole bodies over-fills cancer-age bands and
        # causes catastrophic over-emigration). But they must still face the
        # SAME per-capita emigration rate as real bodies in their band, or they
        # over-realize cancer relative to single scale (incidence inflates with
        # ms_agent_ratio and horizon — the coarse source can emigrate before its
        # cancer fires, but its fine peers otherwise cannot). Snapshot them
        # separately and apply an independent per-band Bernoulli hazard below.
        has_fine = ('fine' in people.states) and bool(
            np.asarray(people.fine[all_alive], dtype=bool).any())
        if has_fine:
            fine_uids = all_alive[np.asarray(people.fine[all_alive], dtype=bool)]
            fine_ages = people.age[fine_uids].astype(int)
            fine_female = np.asarray(people.female[fine_uids], dtype=bool)

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
                count_sim = int(in_band.sum())
                diff = int(round(target - count_sim))

                if diff > 0:
                    # Under-target: queue ``diff`` immigrants for this band.
                    imm_age_chunks.append(np.full(diff, age, dtype=float))
                    imm_female_chunks.append(np.full(diff, sex_is_female, dtype=bool))
                    n_imm_total += diff
                elif diff < 0:
                    # Over-target: emigrate ``-diff`` agents from this band.
                    band_uids = snap_uids[in_band]
                    self._emigrate(band_uids, n=-diff)
                    n_emi_total += -diff
                    # Apply the same per-capita emigration rate to fine agents in
                    # this (age, sex) band, independently (not counted in target).
                    if has_fine and count_sim > 0:
                        p_band = min((-diff) / count_sim, 1.0)
                        fmask = ((fine_female if sex_is_female else ~fine_female)
                                 & (fine_ages == age))
                        self._emigrate_fine(fine_uids[fmask], p_band)

        if n_imm_total > 0:
            # Single People.grow + write per attribute, sized to the total
            # immigration across all (age, sex) bands.
            new_uids = people.grow(n_imm_total)
            ages_at_arrival = np.concatenate(imm_age_chunks)
            # Spread each "year-N" immigrant uniformly across [N, N+1) so the
            # cohort doesn't all transition age bins on the same tick. Uses
            # the per-module CRN dist; ages_at_arrival is the integer
            # lower-bound of each immigrant's year band.
            # Skipped when v2_compat is on, placing immigrants at exact
            # integer ages.
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

    def _emigrate_fine(self, fine_band_uids, p):
        """Emigrate fine multiscale agents at per-band probability ``p``.

        Independent Bernoulli hazard (not target-count based) so fine agents
        face the same per-capita emigration rate as real bodies in their band
        without being counted in the pyramid target. Only still-pending agents
        (not yet cancerous in any genotype) are removed — already-cancerous fine
        agents are realized/counted, so removing them would not change incidence
        but would disturb cancer-death accounting.
        """
        if p <= 0 or len(fine_band_uids) == 0:
            return
        self._fine_emi_bern.set(p=min(float(p), 1.0))
        chosen = self._fine_emi_bern.filter(fine_band_uids)
        if len(chosen) == 0:
            return
        pending = np.ones(len(chosen), dtype=bool)
        for dis in self.sim.diseases.values():
            canc = getattr(dis, 'cancerous', None)
            if canc is not None:
                pending &= ~np.asarray(canc[chosen], dtype=bool)
        to_remove = chosen[pending]
        if len(to_remove):
            self.sim.people.request_removal(to_remove)
        return


class Births(ss.Births):
    """ss.Births that excludes fine multiscale agents from reproducing.

    Births are an independent per-agent Bernoulli, so dropping fine agents
    from the drawn birth_uids is statistically identical to excluding them
    from the eligible pool; only non-fine (level0) agents reproduce.
    """

    def get_births(self):
        birth_uids = super().get_births()
        ppl = self.sim.people
        if 'fine' in ppl.states:
            birth_uids = birth_uids[~ppl.fine[birth_uids]]
        return birth_uids


class AnnualBirths(Births):
    """Annual-pulse births: one birth cohort per calendar year.

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