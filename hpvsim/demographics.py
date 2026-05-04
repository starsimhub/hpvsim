"""HPV-specific demographic modules.

M02: AgeMigration — age-pyramid pinning ported from v2's
people.check_migration. Sits alongside ss.Births and ss.Deaths in
hpv.Sim's default demographics list.

Algorithm (lift-and-shift from v2 _v2_legacy/people.py:818-945):
  Each step (annual cadence in v2; respects sim.dt here):
    1. Look up the target age pyramid for the current year (pop_age_trend).
    2. Compute scale = sim.n_agents / pop_at_sim_start (pop_trend).
    3. For each (sex, integer age):
         count_sim     = alive sim agents at this age, this sex
         count_target  = target_pyramid[sex][age] * scale
         diff          = round(count_target - count_sim)
         if diff > 0:  add `diff` immigrants at age (HPV-naive — matches v2)
         if diff < 0:  weight-pick |diff| existing agents at age and
                       request their removal (treated as emigration)
"""
import numpy as np
import pandas as pd
import starsim as ss


__all__ = ['AgeMigration']


class AgeMigration(ss.Demographics):
    """Age-pyramid pinning to a target population trajectory.

    Each timestep, looks up the target age pyramid for the current (integer)
    year and forces the sim's age × sex composition to match by adding
    immigrants (HPV-naive) or removing emigrants.

    Data can be supplied explicitly via ``pop_trend`` / ``pop_age_trend``
    arguments, or loaded automatically from the sim's ``location`` parameter.

    Args:
        pop_trend     (DataFrame): columns [year, pop_size]. If None, loaded from country data.
        pop_age_trend (DataFrame): columns [year, age, male, female]. If None, loaded from
            country data.
    """

    def __init__(self, pars=None, pop_trend=None, pop_age_trend=None, **kwargs):
        super().__init__()
        self.update_pars(pars, **kwargs)
        self._pop_trend = pop_trend
        self._pop_age_trend = pop_age_trend
        self._scale = None
        self._data_year_min = None
        self._data_year_max = None
        self._n_imm = 0
        self._n_emi = 0
        return

    # ---------------------------------------------------------------------- #
    # Lifecycle hooks                                                          #
    # ---------------------------------------------------------------------- #

    def init_pre(self, sim):
        super().init_pre(sim)

        # Pull from country data if not explicitly supplied.
        if self._pop_trend is None or self._pop_age_trend is None:
            from .data.country import load_country
            # hpv.Sim stores location as sim.location; fall back to sim.pars
            # if present (future-proofing) or raise a clear error.
            location = (
                getattr(sim, 'location', None)
                or getattr(sim.pars, 'location', None)
            )
            if location is None:
                raise AttributeError(
                    'AgeMigration could not find a location on the sim. '
                    'Either pass pop_trend/pop_age_trend explicitly, or use '
                    'hpv.Sim which stores sim.location.'
                )
            cd = load_country(location)
            if self._pop_trend is None:
                self._pop_trend = cd['pop_trend']
            if self._pop_age_trend is None:
                self._pop_age_trend = cd['pop_age_trend']

        # Compute the scale factor: n_agents / data_population_at_sim_start.
        # This mirrors v2's ``scale = sim_pop0 / data_pop0``.
        sim_start = sim.pars.start
        # sim.pars.start may be a float year or a date object.
        sim_start_year = float(
            sim_start.year if hasattr(sim_start, 'year') else sim_start
        )
        pt = self._pop_trend
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
        """Pin age × sex pyramid to the target for the current integer year."""
        sim = self.sim
        year = int(sim.t.now('year'))

        # Silent-skip if outside data range (mirrors v2's ``return 0`` guard).
        if year < self._data_year_min or year > self._data_year_max:
            self._n_imm = 0
            self._n_emi = 0
            return

        pat_year = self._pop_age_trend[self._pop_age_trend['year'] == year]
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
        ages = people.age[snap_uids].astype(int)
        female = people.female[snap_uids].astype(bool)

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

        Uses people.grow(n) which allocates sequential UIDs and slots.
        New agents inherit the default state for every BoolState (False),
        matching v2's add_births-based immigrants that are HPV-naive.
        """
        if n <= 0:
            return
        people = self.sim.people
        new_uids = people.grow(n)
        people.age[new_uids] = float(age)
        people.female[new_uids] = bool(female)
        return

    def _emigrate(self, band_uids, n):
        """Remove n agents from band_uids via request_removal.

        request_removal is the emigration equivalent of request_death — the
        agent leaves the simulation without being recorded as dead.

        Args:
            band_uids: ss.uids of alive agents in the age × sex band.
            n: number of agents to remove.
        """
        if n <= 0 or len(band_uids) == 0:
            return
        n_pick = min(int(n), len(band_uids))
        # Random choice without replacement (not CRN-tracked — matches v2's
        # hpu.choose_w which is also not strictly CRN-safe for migration).
        chosen_idx = np.random.choice(len(band_uids), size=n_pick, replace=False)
        chosen_uids = band_uids[chosen_idx]
        self.sim.people.request_removal(chosen_uids)
        return