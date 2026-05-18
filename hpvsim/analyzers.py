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
            # Allocate per-year output arrays. Type-distribution mode uses
            # (n_bins, n_genotypes); everything else uses (n_bins,).
            nbins = len(rdict.bins)
            ng = len(self.hpv_modules)
            shape = (nbins, ng) if self._is_type_dist(rkey) else (nbins,)
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

    @staticmethod
    def _is_type_dist(rkey):
        return 'genotype_dist' in rkey

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
                # Other sub-modes handled in later tasks.
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

    def to_dataframe(self, key):
        """Return outputs for `key` as a DataFrame indexed by year.

        For standard age-binned results: columns are age bin labels.
        For type-distribution results: columns are genotype keys (one row per
        year), with values summed over age bins.
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
                for i, col in enumerate(cols):
                    data[col].append(float(totals[i]))
            return pd.DataFrame(data, index=pd.Index(index, name='year'))
        cols = rdict.age_labels
        rows = []
        index = []
        for y, arr in self.outputs[key].items():
            index.append(y)
            rows.append(arr.astype(float))
        return pd.DataFrame(rows, columns=cols,
                            index=pd.Index(index, name='year'))


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
