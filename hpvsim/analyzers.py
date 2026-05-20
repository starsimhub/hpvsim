"""HPVsim analyzers built on starsim's ss.Analyzer.

Currently contains AgeResults (M04 port of v2's age_results). M09 will add
snapshot, age_pyramid, age_causal_infection, and dalys analyzers here.
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


__all__ = ['AgeResults']


class AgeResults(ss.Analyzer):
    """Snapshot age-binned simulation outputs at specified years.

    Faithful port of v2's ``age_results`` analyzer
    (``hpvsim/_v2_legacy/analysis.py:511``) onto ``ss.Analyzer``. Snapshots
    age-binned counts (e.g. cancers), prevalences (e.g. hpv_prevalence),
    incidences, and the type-distribution sub-mode at specified years.

    Args:
        result_args (dict): nested dict / objdict where each top-level key
            is a result name (e.g. ``'cancers'``, ``'hpv_prevalence'``,
            ``'cancerous_genotype_dist'``) and the value is a dict with at
            minimum ``years`` (scalar or list) and ``edges`` (array of age
            bin edges).
        die (bool): whether to raise on configuration / validation errors.

    Deltas from v2 (intentional):
        - No ``datafile`` loading — observed data lives on the
          ``CalibComponent``, not on the analyzer.
        - No ``compute_fit`` / ``mismatch`` — the loss path is the
          ``CalibComponent`` likelihood, not analyzer math.
        - No HIV stratification — M08 will add ``with_hiv`` / ``no_hiv``
          handling when HIV lands.

    Example::

        import numpy as np, sciris as sc, hpvsim as hpv
        ar = hpv.AgeResults(result_args=sc.objdict(
            cancers=sc.objdict(years=[2015, 2020],
                               edges=np.arange(0, 101, 5)),
        ))
        sim = hpv.Sim(analyzers=[ar])
        sim.run()
        df = ar.to_dataframe(key='cancers')   # index=year, columns=age bin labels
    """

    # Result-name -> per-HPV-module BoolState attribute. Union across modules
    # ("any genotype with this state"), matching HPVTotal._UNION_STATES.
    _COUNT_TO_STATE = {
        'cancers':       'cancerous',
        'n_cancerous':   'cancerous',
        'n_cin':         'cin',
        'n_precin':      'precin',
        'n_infected':    'infected',
        'hpv':           'infected',   # alias used in some configs
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
        """Discover HPV modules; allocate output arrays; resolve year -> ti.

        Run before init_results (matches HPVTotal's pattern in
        hpvsim/cross_genotype.py:133).
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
            # Map each requested year to its timeline tick (last tick within
            # that calendar year, matching v2's end-of-year accumulation).
            rdict.year_to_ti = self._resolve_year_ticks(sim, rdict.years)
            # Allocate per-year output arrays:
            #   - Type-distribution: (n_bins, n_genotypes) raw counts.
            #   - Prevalence: (n_bins, 2) — [:, 0] = num, [:, 1] = denom.
            #     Stored separately so factories that want (x, n) per bin
            #     (BetaBinomial) can pull both without recomputing.
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
        timevec = sim.timevec
        # Convert ss.date / pd.Timestamp / float years into floats.
        tv_years = np.array([_as_year(t) for t in timevec], dtype=float)
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

    def _bin_count(self, rdict, attr):
        """Bin alive-agent count of (union-across-genotypes) BoolState `attr`."""
        sim = self.sim
        people = sim.people
        alive = people.alive.values
        state_any = np.zeros_like(alive)
        for mod in self.hpv_modules:
            state_any |= getattr(mod, attr).values
        mask = state_any & alive
        ages = people.age.values[mask]
        weights = getattr(people, 'scale', None)
        if weights is not None:
            weights = weights.values[mask]
        counts, _ = np.histogram(ages, bins=rdict.edges, weights=weights)
        return counts

    def _bin_prevalence(self, rdict, attr, female_only=False):
        """Age-bin prevalence as a (nbins, 2) array — [:, 0]=num, [:, 1]=denom.

        Storing (num, denom) per bin rather than the ratio lets BetaBinomial
        components consume raw counts. to_dataframe(key) collapses to the
        ratio for backward compatibility with prevalence-as-ratio callers.
        """
        sim = self.sim
        people = sim.people
        alive = people.alive.values
        state_any = np.zeros_like(alive)
        for mod in self.hpv_modules:
            state_any |= getattr(mod, attr).values
        denom_mask = alive
        if female_only:
            denom_mask = alive & people.female.values
        ages = people.age.values
        weights = getattr(people, 'scale', None)
        weights = weights.values if weights is not None else None
        num_mask = state_any & denom_mask
        num, _ = np.histogram(ages[num_mask], bins=rdict.edges,
                              weights=(weights[num_mask] if weights is not None else None))
        denom, _ = np.histogram(ages[denom_mask], bins=rdict.edges,
                                weights=(weights[denom_mask] if weights is not None else None))
        out = np.zeros((len(rdict.bins), 2), dtype=float)
        out[:, 0] = num
        out[:, 1] = denom
        return out

    def _bin_incidence(self, rdict, date_attr, state_attr):
        """Age-bin new-events-this-year / at-risk-female-denominator (per 100k).

        Numerator: agents whose ti_<event> equals sim.ti and who are in-state.
        At dt=1 yr this captures one year of events. For dt<1 the snapshot
        captures only the final sub-step's events; the M04 smoke test uses
        dt=1 so this is sufficient. A multi-substep accumulator is a follow-on.
        """
        sim = self.sim
        people = sim.people
        alive = people.alive.values
        female = people.female.values
        # Single pass: union both "new event" and "cancerous_any" across modules.
        new_event = np.zeros_like(alive)
        cancerous_any = np.zeros_like(alive)
        for mod in self.hpv_modules:
            ti_arr = getattr(mod, date_attr).values
            state = getattr(mod, state_attr).values
            new_event |= (ti_arr == sim.ti) & state
            cancerous_any |= mod.cancerous.values
        at_risk = alive & female & ~cancerous_any
        ages = people.age.values
        weights = getattr(people, 'scale', None)
        weights = weights.values if weights is not None else None
        num, _ = np.histogram(ages[new_event & alive], bins=rdict.edges,
                              weights=(weights[new_event & alive] if weights is not None else None))
        denom, _ = np.histogram(ages[at_risk], bins=rdict.edges,
                                weights=(weights[at_risk] if weights is not None else None))
        return np.divide(num, denom, out=np.zeros_like(num, dtype=float),
                         where=denom > 0) * 1e5

    def _bin_type_distribution(self, rdict, attr):
        """Per-genotype age-binned raw counts; to_dataframe normalizes."""
        sim = self.sim
        people = sim.people
        alive = people.alive.values
        ages = people.age.values
        weights = getattr(people, 'scale', None)
        weights = weights.values if weights is not None else None
        nbins = len(rdict.bins)
        ng = len(self.hpv_modules)
        out = np.zeros((nbins, ng), dtype=float)
        for gi, mod in enumerate(self.hpv_modules):
            mask = getattr(mod, attr).values & alive
            counts, _ = np.histogram(ages[mask], bins=rdict.edges,
                                     weights=(weights[mask] if weights is not None else None))
            out[:, gi] = counts
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
        BetaBinomial component path in hpv.calibration.hpv_prev_by_age.

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


def _as_year(t):
    """Convert a starsim timeline entry (ss.date / pd.Timestamp / number) to a float year."""
    if hasattr(t, 'year') and hasattr(t, 'month'):
        # ss.date or pd.Timestamp: use decimal year (Jan 1 = .0, Dec 31 ≈ .997).
        import datetime as dt
        start = dt.datetime(t.year, 1, 1)
        end = dt.datetime(t.year + 1, 1, 1)
        now = dt.datetime(t.year, t.month, getattr(t, 'day', 1))
        return t.year + (now - start).total_seconds() / (end - start).total_seconds()
    return float(t)
