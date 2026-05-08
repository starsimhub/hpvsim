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
"""
import numpy as np
import pandas as pd
import starsim as ss


__all__ = ['AgeMigration']


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

    def __init__(self, pars=None, pop_total=None, pop_by_age=None, **kwargs):
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
        self._data_year_min = None                  # Inclusive lower bound of pop_total years; outside this, step() no-ops.
        self._data_year_max = None                  # Inclusive upper bound of pop_total years.
        self._n_imm = 0                             # Immigrants added this step (for results.new_immigrants).
        self._n_emi = 0                             # Emigrants requested this step (for results.new_emigrants).
        # CRN-safe emigrant selection — domain set per call.
        self._emi_select = ss.choice(replace=False)
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

        self._data_year_min = int(pt['year'].min())
        self._data_year_max = int(pt['year'].max())
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

        # Silent-skip if outside data range.
        if year < self._data_year_min or year > self._data_year_max:
            self._n_imm = 0
            self._n_emi = 0
            return

        pat_year = self._pop_by_age[self._pop_by_age['year'] == year]
        if pat_year.empty:
            self._n_imm = 0
            self._n_emi = 0
            return

        pat_year = pat_year.sort_values('age')
        ages_data = pat_year['age'].values.astype(int)

        people = sim.people
        # Snapshot alive UIDs and their attributes at the start of the step.
        # This ensures the boolean masks remain aligned with the UID list even
        # as immigrants are added during the loop.
        snap_uids = people.auids.copy()
        # age is float32 — cast to int for integer-bin lookup; female is already bool.
        ages = people.age[snap_uids].astype(int)
        female = people.female[snap_uids]

        n_imm_total = 0
        n_emi_total = 0

        for sex_label, sex_mask in (
            ('male',   ~female),
            ('female',  female),
        ):
            sex_is_female = (sex_label == 'female')
            target_counts = pat_year[sex_label].values * self._scale

            for age, target in zip(ages_data, target_counts):
                # Agents at this integer age × sex in the snapshot
                in_band = sex_mask & (ages == age)
                count_sim = int(in_band.sum())
                diff = int(round(target - count_sim))

                if diff > 0:
                    self._immigrate(n=diff, age=age, female=sex_is_female)
                    n_imm_total += diff
                elif diff < 0:
                    band_uids = snap_uids[in_band]
                    self._emigrate(band_uids, n=-diff)
                    n_emi_total += -diff

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

    def _immigrate(self, n, age, female):
        """Add n new agents at exact age and sex, HPV-naive.

        ``people.grow(n)`` allocates sequential UIDs and slots. New agents
        inherit the default state for every BoolState (False), so they
        enter HPV-naive.
        """
        if n <= 0:
            return
        people = self.sim.people
        new_uids = people.grow(n)
        people.age[new_uids] = float(age)
        people.female[new_uids] = bool(female)
        return

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