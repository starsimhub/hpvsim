"""HPVsim analyzers.

M02: AgeResults — minimum-scope age-stratified results, cancer-only.
M03/M04 extend the supported `results` keys to ('cancer', 'cins', 'hpv').
"""
import numpy as np
import starsim as ss


__all__ = ['AgeResults']


_DEFAULT_AGE_BINS = np.arange(0, 105, 5)   # 5-yr bins; last bin is [100, 105)


class AgeResults(ss.Analyzer):
    """Age-stratified result aggregation, cancer-only minimum scope.

    Args:
        results: tuple of result keys to age-stratify. M02 supports
                 ('cancer',). Other keys raise NotImplementedError.
        age_bins: array of bin edges (default: 5-yr bins, 0–100).
        year: scalar or list of report years. Each year produces one row in
              the output array.
    """

    def __init__(self, results=('cancer',), age_bins=None, year=None, **kwargs):
        super().__init__(**kwargs)
        keys = tuple(results) if not isinstance(results, str) else (results,)
        if not set(keys).issubset({'cancer'}):
            raise NotImplementedError(
                f'AgeResults M02 supports cancer only; got {keys!r}.'
            )
        self.results_to_collect = keys
        self.age_bins = np.asarray(age_bins) if age_bins is not None else _DEFAULT_AGE_BINS
        if year is None:
            self.report_years = []
        elif np.isscalar(year):
            self.report_years = [float(year)]
        else:
            self.report_years = [float(y) for y in year]
        self.name = 'age_results'
        self._reported_year_idx = 0
        return

    def init_results(self):
        super().init_results()
        n_years = len(self.report_years)
        n_bins = len(self.age_bins) - 1
        self.define_results(
            ss.Result('cancer_incidence_by_age', shape=(n_years, n_bins),
                      dtype=float, scale=False,
                      label='Cancer incidence by age (per 100k female-yrs)'),
        )
        return

    def step(self):
        sim = self.sim
        if self._reported_year_idx >= len(self.report_years):
            return
        target_year = self.report_years[self._reported_year_idx]
        now_year = float(sim.t.now('year'))
        if now_year < target_year:
            return

        # yearvec maps integer ti indices to float years.
        yearvec = np.asarray(sim.t.yearvec)

        people = sim.people

        # Accumulate cancer counts across all disease modules that have ti_cancerous.
        n_bins = len(self.age_bins) - 1
        total_counts = np.zeros(n_bins, dtype=float)

        for disease in sim.diseases.values():
            if not hasattr(disease, 'ti_cancerous'):
                continue

            ti_arr = np.asarray(disease.ti_cancerous)   # float array, len = max_uids

            # Only consider agents with a finite (non-NaN) ti_cancerous.
            finite_mask = ~np.isnan(ti_arr)
            if not finite_mask.any():
                continue

            # Convert ti index to year for each agent with a scheduled cancer time.
            ti_int = ti_arr[finite_mask].astype(int)
            # Clip to valid yearvec range (shouldn't be necessary but defensive).
            ti_int = np.clip(ti_int, 0, len(yearvec) - 1)
            yr_cancerous = yearvec[ti_int]

            # Window: cancer events in (target_year - 1, target_year].
            window = (yr_cancerous > target_year - 1.0) & (yr_cancerous <= target_year)
            if not window.any():
                continue

            # Get the UIDs / indices of agents in the window.
            # finite_mask indices correspond to agent positions in ti_arr.
            all_indices = np.where(finite_mask)[0]
            window_indices = all_indices[window]
            cancer_uids = ss.uids(window_indices)

            # Get ages of those agents at time of transition (use current age as proxy).
            cancer_ages = np.asarray(people.age[cancer_uids])
            counts, _ = np.histogram(cancer_ages, bins=self.age_bins)
            total_counts += counts.astype(float)

        # Female-years denominator: count alive females per age bin.
        # We use a 1-year window denominator approximation (alive now).
        f_mask = np.asarray(people.alive) & np.asarray(people.female)
        f_ages = np.asarray(people.age)[f_mask]
        denom, _ = np.histogram(f_ages, bins=self.age_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(denom > 0, total_counts / denom * 100_000.0, 0.0)

        self.results.cancer_incidence_by_age[self._reported_year_idx, :] = rate
        self._reported_year_idx += 1
        return