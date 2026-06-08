"""HPVsim analyzers built on starsim's ss.Analyzer."""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


__all__ = ['AgeResults', 'snapshot', 'age_pyramid', 'age_causal_infection', 'dalys']


def _make_age_labels(edges):
    """['0-5', '5-10', ..., '95+'] from bin edges."""
    labels = [f'{int(edges[i])}-{int(edges[i+1])}' for i in range(len(edges) - 2)]
    labels.append(f'{int(edges[-2])}+')
    return labels


def _histogram(ages, mask, edges, weights):
    """np.histogram of ages[mask] with optional weights[mask]."""
    w = weights[mask] if weights is not None else None
    counts, _ = np.histogram(ages[mask], bins=edges, weights=w)
    return counts


def _resolve_year_ticks(sim, years):
    """Map calendar years -> timeline tick indices.

    Picks the last tick whose date falls within each calendar year, so annual
    flows accumulated through the year are captured (year-end semantics, in
    contrast to _resolve_date_ticks' nearest-tick point-in-time semantics).
    """
    tv_years = np.asarray(sim.timevec.years, dtype=float)
    out = {}
    for y in years:
        mask = (tv_years >= y) & (tv_years < y + 1)
        ticks = np.where(mask)[0]
        if len(ticks) == 0:
            raise ValueError(f'_resolve_year_ticks: year {y} not in sim timevec '
                             f'({tv_years[0]} to {tv_years[-1]})')
        out[float(y)] = int(ticks[-1])
    return out


def _resolve_date_ticks(sim, dates):
    """Map ss.date-coercible inputs -> nearest timeline tick (instantaneous).

    Returns an sc.odict keyed by the resolved ss.date (``sim.timevec[ti]``),
    value = tick index. Point-in-time semantics for snapshots/pyramids, in
    contrast to _resolve_year_ticks' year-end flow capture.
    Out-of-range dates clamp to the nearest endpoint tick. If two inputs
    resolve to the same tick, the later one overwrites the earlier in the
    returned odict.
    """
    tv_years = np.asarray(sim.timevec.years, dtype=float)
    out = sc.odict()
    for d in sc.tolist(dates):
        dd = ss.date(d)
        ti = int(np.argmin(np.abs(tv_years - float(dd.years))))
        out[sim.timevec[ti]] = ti
    return out


class snapshot(ss.Analyzer):
    """Deep-copy ``sim.people`` at requested timepoints.

    Args:
        timepoints: ss.date-coercible scalar/list (year ints, floats, strings,
            or ss.date). Defaults to the sim end date.
        die (bool): raise if a requested timepoint is past the sim end.

    Example::

        snap = hpv.snapshot(timepoints=['2015', 2020])
        sim = hpv.Sim(analyzers=[snap]); sim.run()
        people_2020 = sim.analyzers[0].get(2020)
    """

    def __init__(self, timepoints=None, die=True, **kwargs):
        super().__init__(**kwargs)
        self.timepoints = timepoints
        self.die = die
        self.snapshots = sc.odict()   # ss.date -> deep-copied People
        self._date_to_ti = None       # ss.date -> tick index

    def init_pre(self, sim):
        super().init_pre(sim)
        tps = sc.tolist(self.timepoints) if self.timepoints is not None else [sim.timevec[-1]]
        tv_end = float(np.asarray(sim.timevec.years, dtype=float)[-1])
        kept = []
        for d in tps:
            dd = ss.date(d)
            if float(dd.years) > tv_end + 1e-9:
                if self.die:
                    raise ValueError(f'snapshot: requested {dd} is past sim end {sim.timevec[-1]}')
                continue
            kept.append(dd)
        self._date_to_ti = _resolve_date_ticks(sim, kept)

    def step(self):
        ti = self.sim.ti
        for date, snap_ti in self._date_to_ti.items():
            if snap_ti == ti:
                self.snapshots[date] = sc.dcp(self.sim.people)

    def get(self, key=None):
        """Retrieve a snapshot by ss.date-coercible key (nearest match).

        If key is None, returns the first recorded snapshot.
        """
        keys = list(self.snapshots.keys())
        if not keys:
            raise sc.KeyNotFoundError('snapshot: no snapshots recorded')
        if key is None:
            return self.snapshots[keys[0]]
        target = float(ss.date(key).years)
        yrs = np.array([k.years for k in keys])
        return self.snapshots[keys[int(np.argmin(np.abs(yrs - target)))]]


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
        return _make_age_labels(edges)

    @staticmethod
    def _resolve_year_ticks(sim, years):
        return _resolve_year_ticks(sim, years)

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
        return _histogram(ages, mask, edges, weights)

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


class age_pyramid(ss.Analyzer):
    """Age/sex pyramid (scale-weighted histograms) at requested timepoints.

    Args:
        timepoints: ss.date-coercible scalar/list; defaults to sim end.
        edges: age bin edges; defaults to ``np.linspace(0, 100, 11)``.
        age_labels: optional bin labels; auto-generated if omitted.
        datafile: optional path/dataframe of observed data (stored on
            ``self.data`` for the later plotting layer; not plotted here).
        die (bool): reserved for symmetry with snapshot.

    Output: ``self.age_pyramids`` is an sc.odict keyed by ss.date, each value an
    ``(nbins, 2)`` array with columns [male, female].
    """

    def __init__(self, timepoints=None, edges=None, age_labels=None,
                 datafile=None, die=False, **kwargs):
        super().__init__(**kwargs)
        self.timepoints = timepoints
        self.edges = edges
        self.age_labels = age_labels
        self.datafile = datafile
        self.die = die
        self.data = None
        self.bins = None
        self.age_pyramids = sc.odict()
        self._date_to_ti = None

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.edges is None:
            self.edges = np.linspace(0, 100, 11)
        self.edges = np.asarray(self.edges, dtype=float)
        self.bins = self.edges[:-1]
        if self.age_labels is None:
            self.age_labels = _make_age_labels(self.edges)
        if self.datafile is not None:
            self.data = (pd.read_csv(self.datafile) if isinstance(self.datafile, str)
                         else pd.DataFrame(self.datafile))
        self._date_to_ti = _resolve_date_ticks(sim, self.timepoints
                                                if self.timepoints is not None
                                                else [sim.timevec[-1]])

    def step(self):
        ti = self.sim.ti
        for date, snap_ti in self._date_to_ti.items():
            if snap_ti != ti:
                continue
            people = self.sim.people
            alive = people.alive.values
            ages = people.age.values
            female = people.female.values
            scale = getattr(people, 'scale', None)
            weights = scale.values if scale is not None else None
            out = np.zeros((len(self.bins), 2), dtype=float)
            out[:, 0] = _histogram(ages, alive & ~female, self.edges, weights)
            out[:, 1] = _histogram(ages, alive & female, self.edges, weights)
            self.age_pyramids[date] = out

    def to_dataframe(self):
        """Tidy long-form (date, age_bin, sex, count)."""
        rows = []
        for date, arr in self.age_pyramids.items():
            for bi, label in enumerate(self.age_labels):
                rows.append(dict(date=date, age_bin=label, sex='male',   count=float(arr[bi, 0])))
                rows.append(dict(date=date, age_bin=label, sex='female', count=float(arr[bi, 1])))
        return pd.DataFrame(rows)


class age_causal_infection(ss.Analyzer):
    """Age at causal infection / CIN2+ / cancer, and dwell times, per cancer.

    For each cervical-cancer onset, back-traces to the age at the causal
    (current persistent) HPV infection and at CIN2+, and records the dwell
    times precin (causal->CIN), cin (CIN->cancer), and total. Ledger-aware:
    reads live agents at ms_agent_ratio==1 and the per-module ``_cancer_events``
    ledger (own + extra sub-cancers) at ratio>1.

    Args:
        start: ss.date-coercible; only count cancers at/after this date.
            Defaults to sim start.
    """

    def __init__(self, start=None, **kwargs):
        super().__init__(**kwargs)
        self.start = start

    def init_pre(self, sim):
        from .hpv import HPV
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        self.start_year = (float(ss.date(self.start).years) if self.start is not None
                           else float(sim.timevec.years[0]))
        self.age_causal = []
        self.age_cin = []
        self.age_cancer = []
        self.weights = []
        self.dwelltime = {k: [] for k in ('precin', 'cin', 'total')}
        self._use_ledger = any(int(m.pars.ms_agent_ratio) > 1 for m in self.hpv_modules)

    def _record(self, cancer_age, causal_age, cin_age, weight):
        cancer_age = np.asarray(cancer_age, dtype=float)
        causal_age = np.asarray(causal_age, dtype=float)
        cin_age = np.asarray(cin_age, dtype=float)
        self.age_cancer.extend(cancer_age.tolist())
        self.age_causal.extend(causal_age.tolist())
        self.age_cin.extend(cin_age.tolist())
        self.weights.extend(np.asarray(weight, dtype=float).tolist())
        self.dwelltime['precin'].extend((cin_age - causal_age).tolist())
        self.dwelltime['cin'].extend((cancer_age - cin_age).tolist())
        self.dwelltime['total'].extend((cancer_age - causal_age).tolist())

    def step(self):
        if self._use_ledger:
            return  # events read from the ledger in finalize
        sim = self.sim
        ti = sim.ti
        if float(sim.timevec[ti].years) < self.start_year:
            return
        dt = float(sim.t.dt)
        age_raw = np.asarray(sim.people.age.raw)
        for m in self.hpv_modules:
            new = np.where(np.asarray(m.ti_cancerous.raw) == ti)[0]
            if not len(new):
                continue
            ti_inf = np.asarray(m.ti_infected.raw)[new]
            ti_cin = np.asarray(m.ti_cin.raw)[new]
            ok = np.isfinite(ti_inf) & np.isfinite(ti_cin)
            new, ti_inf, ti_cin = new[ok], ti_inf[ok], ti_cin[ok]
            cur = age_raw[new]
            self._record(cur, cur - (ti - ti_inf) * dt, cur - (ti - ti_cin) * dt,
                         np.ones(len(new)))

    def finalize(self):
        super().finalize()
        if self._use_ledger:
            # Ledger-path note: for EXTRA sub-cancers, `causal` derives from the
            # source agent's infection-age basis in HPV._multiscale_ledger, so the
            # ledger-path age_causal distribution is not identical to the
            # ratio==1 agent path. Cross-ratio comparisons should use age_cancer /
            # age_cin (age_causal stays valid as a within-run distribution).
            for m in self.hpv_modules:
                for (onset_ti, causal, cin_age, cancer_age, _death, w) in m._cancer_events:
                    if float(self.sim.timevec[int(onset_ti)].years) < self.start_year:
                        continue
                    self._record([cancer_age], [causal], [cin_age], [w])
        self.age_causal = np.array(self.age_causal)
        self.age_cin = np.array(self.age_cin)
        self.age_cancer = np.array(self.age_cancer)
        self.weights = np.array(self.weights)
        for k in self.dwelltime:
            self.dwelltime[k] = np.array(self.dwelltime[k])


class dalys(ss.Analyzer):
    """Incidence-based DALYs (YLL + YLD) from cervical cancer, by calendar year.

    YLL and YLD are attributed at the year of cancer onset (incidence-based),
    weighted by ``people.scale``. Ledger-aware: reads live agents at
    ms_agent_ratio==1 and the per-module ``_cancer_events`` ledger at ratio>1.
    Absolute population DALYs require multiplying by ``sim.pars.pop_scale``
    (same convention as v2's per-agent-scale analyzer output).

    Args:
        start: ss.date-coercible; only count onsets at/after this year.
        life_expectancy: reference life expectancy for YLL (default 84;
            pass a country-specific value where available).
    """

    def __init__(self, start=None, life_expectancy=84, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.life_expectancy = life_expectancy
        self.disability_weights = sc.objdict(
            weights=[0.288, 0.049, 0.451, 0.54],     # GBD2017
            time_fraction=[0.05, 0.85, 0.09, 0.01],
        )

    @property
    def av_disutility(self):
        dw = self.disability_weights
        return sum(dw.weights[i] * dw.time_fraction[i] for i in range(len(dw.weights)))

    def init_pre(self, sim):
        from .hpv import HPV
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        tv_years = np.asarray(sim.timevec.years, dtype=float)
        self.start_year = (int(np.floor(ss.date(self.start).years)) if self.start is not None
                           else int(np.floor(tv_years[0])))
        self.end_year = int(np.floor(tv_years[-1]))
        self.years = np.arange(self.start_year, self.end_year + 1)
        n = len(self.years)
        self.yll = np.zeros(n)
        self.yld = np.zeros(n)
        self.dalys = np.zeros(n)
        self._use_ledger = any(int(m.pars.ms_agent_ratio) > 1 for m in self.hpv_modules)

    def _accumulate(self, year, cancer_age, death_age, weight):
        if year < self.start_year or year > self.end_year:
            return
        idx = year - self.start_year
        cancer_age = np.asarray(cancer_age, dtype=float)
        death_age = np.asarray(death_age, dtype=float)
        weight = np.asarray(weight, dtype=float)
        dur = death_age - cancer_age
        self.yld[idx] += float((weight * dur * self.av_disutility).sum())
        years_left = np.maximum(0.0, self.life_expectancy - death_age)
        self.yll[idx] += float((weight * years_left).sum())

    def step(self):
        if self._use_ledger:
            return
        sim = self.sim
        ti = sim.ti
        year = int(np.floor(sim.timevec[ti].years))
        if year < self.start_year:
            return
        dt = float(sim.t.dt)
        age_raw = np.asarray(sim.people.age.raw)
        scale = getattr(sim.people, 'scale', None)
        scale_raw = np.asarray(scale.raw) if scale is not None else None
        for m in self.hpv_modules:
            new = np.where(np.asarray(m.ti_cancerous.raw) == ti)[0]
            if not len(new):
                continue
            cancer_age = age_raw[new]
            ti_dead = np.asarray(m.ti_dead_cancer.raw)[new]
            death_age = cancer_age + (ti_dead - ti) * dt
            w = scale_raw[new] if scale_raw is not None else np.ones(len(new))
            self._accumulate(year, cancer_age, death_age, w)

    def finalize(self):
        super().finalize()
        if self._use_ledger:
            for m in self.hpv_modules:
                for (onset_ti, _causal, _cin, cancer_age, death_age, w) in m._cancer_events:
                    year = int(np.floor(self.sim.timevec[int(onset_ti)].years))
                    self._accumulate(year, [cancer_age], [death_age], [w])
        self.dalys = self.yll + self.yld
