"""HPVsim analyzers built on starsim's ss.Analyzer."""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


__all__ = ['AgeResults']


class AgeResults(ss.Analyzer):
    """Snapshot age-binned simulation outputs at specified years.

    Records counts (e.g. ``cancers``), prevalences (e.g. ``hpv_prevalence``),
    incidences (e.g. ``cancer_incidence``), and per-genotype distributions
    (e.g. ``cancerous_genotype_dist``) at each requested year. Observed
    data and likelihood evaluation live on the ``CalibComponent`` side;
    this class only produces simulation outputs.

    Args:
        result_args (dict): nested dict / objdict where each top-level key
            is a result name (e.g. ``'cancers'``, ``'hpv_prevalence'``,
            ``'cancerous_genotype_dist'``) and the value is a dict with at
            minimum ``years`` (scalar or list) and ``edges`` (array of age
            bin edges).
        die (bool): whether to raise on configuration / validation errors.

    Example::

        import numpy as np, sciris as sc, hpvsim as hpv
        ar = hpv.AgeResults(result_args=sc.objdict(
            cancers=sc.objdict(years=[2015, 2020],
                               edges=np.arange(0, 101, 5)),
        ))
        sim = hpv.Sim(analyzers=[ar])
        sim.run()
        df = ar.to_dataframe(key='cancers')   # t-indexed, age-bin columns
    """

    # Result-name -> per-HPV-module BoolState attribute. Union across modules
    # ("any genotype with this state").
    _COUNT_TO_STATE = {
        'cancers':       'cancerous',
        'n_cancerous':   'cancerous',
        'n_cin':         'cin',
        'n_precin':      'precin',
        'n_infected':    'infected',
        'hpv':           'infected',
    }

    # Result-name -> (BoolState attr, female_only). Prevalence is num/denom
    # where denom is alive (female-only for sex-specific conditions like CIN
    # and cancer; whole population for HPV infection prevalence).
    _PREV_TO_STATE = {
        'hpv_prevalence':       ('infected',  False),
        'cin_prevalence':       ('cin',       True),
        'cancer_prevalence':    ('cancerous', True),
        'precin_prevalence':    ('precin',    True),
    }

    # Result-name -> (event-time attr, in-state attr) on each HPV module.
    # Incidence numerator = agents whose ti_<event> == sim.ti and in-state.
    # Denominator = at-risk alive females (per 100k convention).
    _INC_TO_ATTRS = {
        'cancer_incidence':  ('ti_cancerous', 'cancerous'),
        'cin_incidence':     ('ti_cin',       'cin'),
    }

    # Result-name -> per-HPV-module BoolState attribute used as numerator.
    _TYPE_DIST_TO_STATE = {
        'cancerous_genotype_dist':  'cancerous',
        'cin_genotype_dist':        'cin',
    }

    def __init__(self, result_args=None, die=False, **kwargs):
        super().__init__(**kwargs)
        if result_args is None:
            raise ValueError('AgeResults: result_args is required')
        self.result_args = sc.objdict(result_args)
        self.die = die
        # Per-year per-result output storage; populated by step().
        # Layout: self.outputs[result_key][year] = np.ndarray of length nbins.
        self.outputs = sc.objdict()
        # Populated by init_pre.
        self.hpv_modules = None
        return

    def init_pre(self, sim):
        """Discover HPV modules, validate inputs, and allocate output arrays.

        Sets ``self.hpv_modules`` before ``super().init_pre`` so any setup
        that depends on the module list (e.g. ``init_results``) sees it.
        """
        from .hpv import HPV
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        for rkey, rdict in self.result_args.items():
            rdict.years = np.atleast_1d(rdict.years).astype(float)
            if 'edges' not in rdict or rdict.edges is None:
                raise ValueError(f'AgeResults: result_args[{rkey!r}] missing edges')
            rdict.edges = np.asarray(rdict.edges, dtype=float)
            rdict.bins = rdict.edges[:-1]
            rdict.age_labels = self._make_age_labels(rdict.edges)
            # Each requested year maps to the last sim tick within that
            # calendar year, so annual flows are captured at year-end.
            rdict.year_to_ti = self._resolve_year_ticks(sim, rdict.years)
            # Per-year output arrays:
            #   - Type-distribution: (n_bins, n_genotypes) raw counts.
            #   - Prevalence: (n_bins, 2) — [:, 0] = num, [:, 1] = denom.
            #     Storing both lets BetaBinomial components pull (x, n)
            #     per bin without recomputing.
            #   - Everything else: (n_bins,).
            nbins = len(rdict.bins)
            ng = len(self.hpv_modules)
            if self._is_type_dist(rkey):
                shape = (nbins, ng)
            elif rkey in self._PREV_TO_STATE:
                shape = (nbins, 2)
            else:
                shape = (nbins,)
            self.outputs[rkey] = {float(y): np.zeros(shape) for y in rdict.years}
        return

    @staticmethod
    def _make_age_labels(edges):
        labels = [f'{int(edges[i])}-{int(edges[i+1])}' for i in range(len(edges) - 2)]
        labels.append(f'{int(edges[-2])}+')
        return labels

    @staticmethod
    def _resolve_year_ticks(sim, years):
        """Map calendar years -> timeline tick indices.

        Picks the last tick whose date falls within each calendar year, so
        annual flows accumulated through the year are captured.
        """
        tv_years = np.asarray(sim.timevec.years, dtype=float)
        out = {}
        for y in years:
            mask = (tv_years >= y) & (tv_years < y + 1)
            ticks = np.where(mask)[0]
            if len(ticks) == 0:
                raise ValueError(f'AgeResults: year {y} not in sim timevec '
                                 f'({tv_years[0]} to {tv_years[-1]})')
            out[float(y)] = int(ticks[-1])
        return out

    @classmethod
    def _is_type_dist(cls, rkey):
        return rkey in cls._TYPE_DIST_TO_STATE

    def step(self):
        """At each scheduled year, snapshot age-binned counts."""
        sim = self.sim
        ti = sim.ti
        for rkey, rdict in self.result_args.items():
            # Is this tick the recorded snapshot tick for any of the years?
            year_match = [y for y, ti_y in rdict.year_to_ti.items() if ti_y == ti]
            if not year_match:
                continue
            for year in year_match:
                if rkey in self._COUNT_TO_STATE:
                    self.outputs[rkey][year] = self._bin_count(rdict, attr=self._COUNT_TO_STATE[rkey])
                elif rkey in self._PREV_TO_STATE:
                    attr, female_only = self._PREV_TO_STATE[rkey]
                    self.outputs[rkey][year] = self._bin_prevalence(
                        rdict, attr=attr, female_only=female_only)
                elif rkey in self._INC_TO_ATTRS:
                    date_attr, state_attr = self._INC_TO_ATTRS[rkey]
                    self.outputs[rkey][year] = self._bin_incidence(
                        rdict, date_attr=date_attr, state_attr=state_attr)
                elif rkey in self._TYPE_DIST_TO_STATE:
                    self.outputs[rkey][year] = self._bin_type_distribution(
                        rdict, attr=self._TYPE_DIST_TO_STATE[rkey])
        return

    def _pop_arrays(self):
        """Cache the alive/ages/weights arrays each binning method needs."""
        people = self.sim.people
        alive = people.alive.values
        ages = people.age.values
        scale = getattr(people, 'scale', None)
        weights = scale.values if scale is not None else None
        return people, alive, ages, weights

    @staticmethod
    def _histogram(ages, mask, edges, weights):
        """np.histogram of ages[mask] with optional weights[mask]."""
        w = weights[mask] if weights is not None else None
        counts, _ = np.histogram(ages[mask], bins=edges, weights=w)
        return counts

    def _bin_count(self, rdict, attr):
        """Bin alive-agent count of (union-across-genotypes) BoolState `attr`."""
        _, alive, ages, weights = self._pop_arrays()
        state_any = np.zeros_like(alive)
        for mod in self.hpv_modules:
            state_any |= getattr(mod, attr).values
        return self._histogram(ages, state_any & alive, rdict.edges, weights)

    def _bin_prevalence(self, rdict, attr, female_only=False):
        """Age-bin prevalence as a (nbins, 2) array — [:, 0]=num, [:, 1]=denom.

        Storing (num, denom) per bin lets BetaBinomial components consume
        raw counts directly. ``to_dataframe(key)`` collapses to the ratio
        for callers that want point-in-time prevalences.
        """
        people, alive, ages, weights = self._pop_arrays()
        state_any = np.zeros_like(alive)
        for mod in self.hpv_modules:
            state_any |= getattr(mod, attr).values
        denom_mask = alive & people.female.values if female_only else alive
        num_mask = state_any & denom_mask
        out = np.zeros((len(rdict.bins), 2), dtype=float)
        out[:, 0] = self._histogram(ages, num_mask, rdict.edges, weights)
        out[:, 1] = self._histogram(ages, denom_mask, rdict.edges, weights)
        return out

    def _bin_incidence(self, rdict, date_attr, state_attr):
        """Age-bin new-events-this-year / at-risk-female-denominator (per 100k).

        Numerator: agents whose ti_<event> equals sim.ti and who are in-state.
        At dt=1 this captures one full year of events per snapshot. With
        dt<1 only the final sub-step's events are captured; a multi-substep
        accumulator is a separate enhancement.
        """
        people, alive, ages, weights = self._pop_arrays()
        female = people.female.values
        new_event = np.zeros_like(alive)
        cancerous_any = np.zeros_like(alive)
        for mod in self.hpv_modules:
            ti_arr = getattr(mod, date_attr).values
            state = getattr(mod, state_attr).values
            new_event |= (ti_arr == self.sim.ti) & state
            cancerous_any |= mod.cancerous.values
        at_risk = alive & female & ~cancerous_any
        num = self._histogram(ages, new_event & alive, rdict.edges, weights)
        denom = self._histogram(ages, at_risk, rdict.edges, weights)
        return np.divide(num, denom, out=np.zeros_like(num, dtype=float),
                         where=denom > 0) * 1e5

    def _bin_type_distribution(self, rdict, attr):
        """Per-genotype age-binned raw counts; ``to_dataframe`` normalizes."""
        _, alive, ages, weights = self._pop_arrays()
        out = np.zeros((len(rdict.bins), len(self.hpv_modules)), dtype=float)
        for gi, mod in enumerate(self.hpv_modules):
            mask = getattr(mod, attr).values & alive
            out[:, gi] = self._histogram(ages, mask, rdict.edges, weights)
        return out

    def to_dataframe(self, key, normalize=True):
        """Return outputs for `key` as a DataFrame indexed by year.

        For standard age-binned results: columns are age bin labels.
        For type-distribution results: columns are genotype keys (one row
        per year). With ``normalize=True`` (default) values are proportions
        summing to 1 across genotypes; with ``normalize=False`` values are
        raw counts summed over age bins — the shape that
        ss.DirichletMultinomial.compute_nll consumes.
        """
        if key not in self.outputs:
            raise KeyError(f'AgeResults: no output for key {key!r}; have {list(self.outputs)}')
        rdict = self.result_args[key]
        if self._is_type_dist(key):
            cols = [m.name for m in self.hpv_modules]
            data = {col: [] for col in cols}
            index = []
            for y, arr in self.outputs[key].items():
                index.append(y)
                totals = arr.sum(axis=0)
                if normalize:
                    total_sum = totals.sum()
                    if total_sum > 0:
                        totals = totals / total_sum
                for i, col in enumerate(cols):
                    data[col].append(float(totals[i]))
            return pd.DataFrame(data, index=pd.Index(index, name='t'))
        cols = rdict.age_labels
        rows = []
        index = []
        is_prev = key in self._PREV_TO_STATE
        for y, arr in self.outputs[key].items():
            index.append(y)
            if is_prev:
                # Prev storage is (nbins, 2) = [num, denom]; emit ratio.
                num, denom = arr[:, 0], arr[:, 1]
                ratio = np.divide(num, denom, out=np.zeros_like(num),
                                  where=denom > 0)
                rows.append(ratio.astype(float))
            else:
                rows.append(arr.astype(float))
        return pd.DataFrame(rows, columns=cols,
                            index=pd.Index(index, name='t'))

    def to_xn_per_bin(self, key):
        """Return per-age-bin (x, n) DataFrames for a prevalence-mode result.

        Each value of the returned dict is a 't'-indexed DataFrame with two
        columns: ``x`` (positives) and ``n`` (total). Used by the
        BetaBinomial-style consumers of (positives, totals) per bin.

        Raises if `key` is not a prevalence result.
        """
        if key not in self._PREV_TO_STATE:
            raise ValueError(
                f'AgeResults.to_xn_per_bin: {key!r} is not a prevalence '
                f'result; supported keys are {list(self._PREV_TO_STATE)}'
            )
        rdict = self.result_args[key]
        years = sorted(self.outputs[key].keys())
        result = {}
        for bi, label in enumerate(rdict.age_labels):
            x_vals = [float(self.outputs[key][y][bi, 0]) for y in years]
            n_vals = [float(self.outputs[key][y][bi, 1]) for y in years]
            result[label] = pd.DataFrame(
                {'x': x_vals, 'n': n_vals},
                index=pd.Index(years, name='t'),
            )
        return result
