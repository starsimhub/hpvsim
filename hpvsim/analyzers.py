"""HPVsim analyzers built on starsim's ss.Analyzer."""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


__all__ = ['by_age', 'snapshot', 'age_pyramid', 'age_causal_infection',
           'dalys', 'results_by_genotype']


class by_age(ss.Analyzer):
    """Age-binned per-timestep results, stored as ``ss.Result`` per (key, bin).

    Records per-timestep age-binned outputs as one 1D ``ss.Result`` per age
    bin (e.g. ``cancers_20_25``, ``cancers_25_30``, ...). This lets
    starsim's ``finalize_results`` handle ``pop_scale`` multiplication
    automatically for count-shaped results (``scale=True``), and lets each
    per-bin timeseries be annualized via ``ss.Result.annualize()``.

    Supported keys:

      - count (``scale=True``, annualize mean):
        ``n_cancerous``, ``n_cin``, ``n_precin``, ``n_infected``, ``hpv``
      - annual event flows (``scale=True``, annualize sum):
        ``cancers``, ``cins``
      - prevalences (``scale=False``, annualize mean, ratio in [0, 1]):
        ``hpv_prevalence``, ``cin_prevalence``, ``cancer_prevalence``,
        ``precin_prevalence``

    After the sim runs, convenience 2D arrays are populated on the analyzer:
    ``self.cancers`` has shape ``(npts, n_bins)``, ``self.hpv_prevalence``
    likewise. ``to_dataframe(key)`` annualizes and returns a DataFrame
    indexed by year with age-bin columns.

    Args:
        keys: result key (``'cancers'``) or list of keys.
        years: reporting-only filter for ``to_dataframe``. Storage is always
            per-timestep so ``ss.Result.annualize`` can be applied. Scalar
            or list of ints. Default None -> all sim years.
        edges: age bin edges. Default ``np.arange(0, 101, 5)``.

    Example::

        ar = hpv.by_age('cancers', years=2020)
        sim = hpv.Sim(location='nigeria', analyzers=[ar])
        sim.run()
        df = ar.to_dataframe('cancers')          # year x age-bin df
        arr = ar.cancers                         # 2D (npts, n_bins)
        sim.results.by_age.cancers_20_25         # 1D ss.Result
    """

    # Result-name -> per-HPV-module BoolState attribute (union across genotypes).
    # Prevalent stocks: alive-agent count in-state at each timestep.
    _COUNT_KEYS = {
        'n_cancerous':   'cancerous',
        'n_cin':         'cin',
        'n_precin':      'precin',
        'n_infected':    'infected',
        'hpv':           'infected',
    }

    # Demographic counts read directly from ``people.<attr>`` (no HPV
    # module lookup). Result-name -> people-attr, or None to count ``alive``.
    _DEMO_KEYS = {
        'n_alive':   None,
        'n_females': 'female',
        'n_males':   'male',
    }

    # Result-name -> (BoolState attr, female_only). Ratio in [0, 1].
    _PREV_KEYS = {
        'hpv_prevalence':       ('infected',  False),
        'cin_prevalence':       ('cin',       True),
        'cancer_prevalence':    ('cancerous', True),
        'precin_prevalence':    ('precin',    True),
    }

    # Result-name -> (event-time attr, in-state attr). New events at the current
    # tick, alive filter applied. Annualize summarize_by='sum' gives annual
    # event counts in population units (scale=True at Result level).
    _FLOW_KEYS = {
        'cancers':  ('ti_cancerous', 'cancerous'),
        'cins':     ('ti_cin',       'cin'),
    }

    def __init__(self, keys=None, years=None, edges=None, **kwargs):
        super().__init__(**kwargs)
        if keys is None:
            raise ValueError(
                "by_age: pass at least one result key, e.g. hpv.by_age('cancers')")
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        known = set(self._COUNT_KEYS) | set(self._PREV_KEYS) | set(self._FLOW_KEYS)
        unknown = [k for k in self.keys if k not in known]
        if unknown:
            raise ValueError(
                f'by_age: unknown key(s) {unknown}. Available: {sorted(known)}')
        self.years = (None if years is None
                      else sorted(int(y) for y in np.atleast_1d(years).tolist()))
        self.edges = np.asarray(
            edges if edges is not None else np.arange(0, 101, 5), dtype=float)
        self.bin_labels = _make_age_labels(self.edges)
        self.hpv_modules = None  # populated in init_pre

    @staticmethod
    def _result_name(key, lo, hi):
        return f'{key}_{int(lo)}_{int(hi)}'

    def _summarize_by(self, key):
        # FLOW keys annualize by SUM (annual event count). COUNT/PREV by MEAN.
        return 'sum' if key in self._FLOW_KEYS else 'mean'

    def init_pre(self, sim):
        from .hpv import HPV
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)

    def init_results(self):
        super().init_results()
        defs = []
        for key in self.keys:
            scale = key in self._COUNT_KEYS or key in self._FLOW_KEYS
            sby = self._summarize_by(key)
            for lo, hi, label in zip(self.edges[:-1], self.edges[1:], self.bin_labels):
                defs.append(ss.Result(
                    self._result_name(key, lo, hi),
                    dtype=float, scale=scale, summarize_by=sby,
                    label=f'{key} ({label})',
                ))
        self.define_results(*defs)

    def step(self):
        ti = self.sim.ti
        people = self.sim.people
        alive = people.alive.values
        ages = people.age.values
        female = people.female.values
        scale = getattr(people, 'scale', None)
        weights = scale.values if scale is not None else None
        for key in self.keys:
            if key in self._COUNT_KEYS:
                attr = self._COUNT_KEYS[key]
                state_any = np.zeros_like(alive)
                for mod in self.hpv_modules:
                    state_any |= getattr(mod, attr).values
                self._write_bins(key, ti, ages, state_any & alive, weights)
            elif key in self._PREV_KEYS:
                attr, female_only = self._PREV_KEYS[key]
                state_any = np.zeros_like(alive)
                for mod in self.hpv_modules:
                    state_any |= getattr(mod, attr).values
                denom_mask = alive & female if female_only else alive
                num_mask = state_any & denom_mask
                self._write_ratio(key, ti, ages, num_mask, denom_mask, weights)
            elif key in self._FLOW_KEYS:
                date_attr, state_attr = self._FLOW_KEYS[key]
                new_event = np.zeros_like(alive)
                for mod in self.hpv_modules:
                    new_event |= ((getattr(mod, date_attr).values == ti)
                                  & getattr(mod, state_attr).values)
                self._write_bins(key, ti, ages, new_event & alive, weights)

    def _write_bins(self, key, ti, ages, mask, weights):
        w = weights[mask] if weights is not None else None
        counts, _ = np.histogram(ages[mask], bins=self.edges, weights=w)
        for i, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            self.results[self._result_name(key, lo, hi)][ti] = counts[i]

    def _write_ratio(self, key, ti, ages, num_mask, den_mask, weights):
        w_num = weights[num_mask] if weights is not None else None
        w_den = weights[den_mask] if weights is not None else None
        num, _ = np.histogram(ages[num_mask], bins=self.edges, weights=w_num)
        den, _ = np.histogram(ages[den_mask], bins=self.edges, weights=w_den)
        ratio = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        for i, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            self.results[self._result_name(key, lo, hi)][ti] = ratio[i]

    def finalize_results(self):
        super().finalize_results()
        # Convenience 2D arrays: self.<key> is (npts, n_bins), reading from the
        # pop_scale-multiplied per-bin Results.
        for key in self.keys:
            arrs = [self.results[self._result_name(key, lo, hi)].values
                    for lo, hi in zip(self.edges[:-1], self.edges[1:])]
            setattr(self, key, np.stack(arrs, axis=1))

    def to_dataframe(self, key):
        """Annualize per-bin Results and return a year x age-bin DataFrame.

        Method per key type: FLOW keys (`cancers`, `cins`) sum across the
        calendar year; COUNT and PREV keys average. When ``years=`` was
        supplied at construction, only those years are returned.
        """
        if key not in self.keys:
            raise KeyError(f'by_age: no output for key {key!r}; have {self.keys}')
        annual_bins = []
        year_index = None
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            r = self.results[self._result_name(key, lo, hi)].annualize()
            annual_bins.append(np.asarray(r.values))
            if year_index is None:
                year_index = np.floor(np.asarray(r.timevec.years)).astype(int)
        arr = np.stack(annual_bins, axis=1)
        df = pd.DataFrame(arr, columns=self.bin_labels,
                          index=pd.Index([float(y) for y in year_index], name='t'))
        if self.years is not None:
            df = df.loc[df.index.isin([float(y) for y in self.years])]
        return df


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


def _resolve_date_ticks(sim, dates):
    """Map ss.date-coercible inputs -> nearest timeline tick (instantaneous).

    Returns an sc.odict keyed by the resolved ss.date (``sim.timevec[ti]``),
    value = tick index. Point-in-time semantics for snapshots/pyramids.
    Out-of-range dates clamp to the nearest endpoint tick. If two inputs
    resolve to the same tick, the later one overwrites the earlier in the
    returned odict.
    """
    tv_years = sim.timevec.years
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
        people_2020 = sim.analyzers['snapshot'].get(2020)
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
        tv_end = sim.timevec.years[-1]
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


class age_pyramid(ss.Analyzer):
    """Age/sex pyramid (scale-weighted histograms) at requested timepoints.

    Args:
        timepoints: ss.date-coercible scalar/list; defaults to sim end.
        edges: age bin edges; defaults to ``np.linspace(0, 100, 11)``.
        age_labels: optional bin labels; auto-generated if omitted.
        datafile: optional path/dataframe of observed data (stored on
            ``self.data`` for the later plotting layer; not plotted here).
        die (bool): raise if a requested timepoint is past the sim end
            (otherwise it is skipped).

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
        tps = sc.tolist(self.timepoints) if self.timepoints is not None else [sim.timevec[-1]]
        tv_end = sim.timevec.years[-1]
        kept = []
        for d in tps:
            dd = ss.date(d)
            if float(dd.years) > tv_end + 1e-9:
                if self.die:
                    raise ValueError(f'age_pyramid: requested {dd} is past sim end {sim.timevec[-1]}')
                continue
            kept.append(dd)
        self._date_to_ti = _resolve_date_ticks(sim, kept)

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

    def finalize_results(self):
        # age_pyramids stores per-agent people.scale-weighted counts; multiply
        # by sim.pars.pop_scale so the pyramid is in real-population units,
        # matching sim.results.* and by_age FLOW/COUNT semantics.
        super().finalize_results()
        pop_scale = self.sim.pars.pop_scale
        if pop_scale != 1.0:
            for date in self.age_pyramids:
                self.age_pyramids[date] = self.age_pyramids[date] * pop_scale

    def to_dataframe(self):
        """Tidy long-form (date, age_bin, sex, count)."""
        rows = []
        for date, arr in self.age_pyramids.items():
            for bi, label in enumerate(self.age_labels):
                rows.append(dict(date=date, age_bin=label, sex='male',   count=float(arr[bi, 0])))
                rows.append(dict(date=date, age_bin=label, sex='female', count=float(arr[bi, 1])))
        return pd.DataFrame(rows)

    def plot(self, date=None, fig=None):
        """Plot this age pyramid (see hpvsim.plotting.plot_age_pyramid)."""
        from .plotting import plot_age_pyramid
        return plot_age_pyramid(self, date=date, fig=fig)


class age_causal_infection(ss.Analyzer):
    """Age at causal infection / CIN2+ / cancer, and dwell times, per cancer.

    For each cervical-cancer onset, back-traces to the age at the causal
    (current persistent) HPV infection and at CIN2+, and records the dwell
    times precin (causal->CIN), cin (CIN->cancer), and total. Reads live agents
    on the standard code path: on the grow multiscale engine, extra cancers are
    real fine agents in ``sim.people`` (``fine=True``, ``scale=1/ratio``), so
    every cancer is captured at any ``ms_agent_ratio`` and weighted by
    ``people.scale``.

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

    def _record(self, cancer_age, causal_age, cin_age, weight):
        self.age_cancer.extend(cancer_age.tolist())
        self.age_causal.extend(causal_age.tolist())
        self.age_cin.extend(cin_age.tolist())
        self.weights.extend(weight.tolist())
        self.dwelltime['precin'].extend((cin_age - causal_age).tolist())
        self.dwelltime['cin'].extend((cancer_age - cin_age).tolist())
        self.dwelltime['total'].extend((cancer_age - causal_age).tolist())

    def step(self):
        sim = self.sim
        ti = sim.ti
        if sim.t.year < self.start_year:
            return
        dt = sim.t.dt_year
        people = sim.people
        scale = getattr(people, 'scale', None)
        for m in self.hpv_modules:
            # Gate on cancerous, not just ti_cancerous==ti: a scheduled
            # ti_cancerous persists on agents who die of other causes before
            # onset, so the bare time-match overcounts vs realized incidence.
            # BoolArr.uids returns active (alive) agents only, so the alive
            # filter is implicit. On the grow engine, fine agents subject to
            # independent competing mortality are correctly dropped here; this
            # also drops the rare agent who reaches cancer and dies of a
            # competing cause on the same tick (~1%).
            new = ((m.ti_cancerous == ti) & m.cancerous).uids
            if not len(new):
                continue
            ti_inf = m.ti_infected[new]
            ti_cin = m.ti_cin[new]
            ok = np.isfinite(ti_inf) & np.isfinite(ti_cin)
            new, ti_inf, ti_cin = new[ok], ti_inf[ok], ti_cin[ok]
            cur = people.age[new]
            w = people.scale[new] if scale is not None else np.ones(len(new))
            self._record(cur, cur - (ti - ti_inf) * dt, cur - (ti - ti_cin) * dt, w)

    def finalize(self):
        super().finalize()
        self.age_causal = np.array(self.age_causal)
        self.age_cin = np.array(self.age_cin)
        self.age_cancer = np.array(self.age_cancer)
        # weights recorded via _record use per-agent people.scale; multiply by
        # sim.pars.pop_scale so weight.sum() equals real-population cancer
        # count, matching sim.results and by_age FLOW-key semantics.
        pop_scale = self.sim.pars.pop_scale
        self.weights = np.array(self.weights) * pop_scale
        for k in self.dwelltime:
            self.dwelltime[k] = np.array(self.dwelltime[k])

    def plot(self, fig=None):
        """Plot age-at-causal/CIN/cancer histograms (see plotting.plot_age_causal_infection)."""
        from .plotting import plot_age_causal_infection
        return plot_age_causal_infection(self, fig=fig)


class dalys(ss.Analyzer):
    """Incidence-based DALYs (YLL + YLD) from cervical cancer, by calendar year.

    YLL and YLD are attributed at the year of cancer onset (incidence-based),
    weighted by ``people.scale`` and multiplied by ``sim.pars.pop_scale`` at
    finalize so the emitted arrays are in real-population units.  Reads live
    agents on the standard code path: on the grow multiscale engine, extra
    cancers are real fine agents (``scale=1/ratio``) in ``sim.people``, so
    all onsets are captured at any ``ms_agent_ratio``.

    Args:
        start: ss.date-coercible; only count onsets at/after this year.
        life_expectancy: reference life expectancy for YLL (default 84;
            pass a country-specific value where available).
        disability_weights: objdict/dict with ``weights`` and ``time_fraction``
            lists (one entry per cancer stage). Defaults to GBD2017; pass your
            own to use different disability weights.
    """

    # GBD2017 cervical-cancer disability weights and time fractions per stage.
    _DEFAULT_DISABILITY_WEIGHTS = dict(
        weights=[0.288, 0.049, 0.451, 0.54],
        time_fraction=[0.05, 0.85, 0.09, 0.01],
    )

    def __init__(self, start=None, life_expectancy=84, disability_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.life_expectancy = life_expectancy
        self.disability_weights = sc.objdict(
            disability_weights if disability_weights is not None
            else self._DEFAULT_DISABILITY_WEIGHTS
        )

    @property
    def av_disutility(self):
        dw = self.disability_weights
        return sum(dw.weights[i] * dw.time_fraction[i] for i in range(len(dw.weights)))

    def init_pre(self, sim):
        from .hpv import HPV
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        tv_years = sim.timevec.years
        self.start_year = (int(np.floor(ss.date(self.start).years)) if self.start is not None
                           else int(np.floor(tv_years[0])))
        self.end_year = int(np.floor(tv_years[-1]))
        self.years = np.arange(self.start_year, self.end_year + 1)
        n = len(self.years)
        self.yll = np.zeros(n)
        self.yld = np.zeros(n)
        self.dalys = np.zeros(n)

    def _accumulate(self, year, cancer_age, death_age, weight):
        if year < self.start_year or year > self.end_year:
            return
        idx = year - self.start_year
        dur = death_age - cancer_age
        self.yld[idx] += float((weight * dur * self.av_disutility).sum())
        years_left = np.maximum(0.0, self.life_expectancy - death_age)
        self.yll[idx] += float((weight * years_left).sum())

    def step(self):
        sim = self.sim
        ti = sim.ti
        year = int(np.floor(sim.t.year))
        if year < self.start_year:
            return
        dt = sim.t.dt_year
        people = sim.people
        scale = getattr(people, 'scale', None)
        for m in self.hpv_modules:
            # See age_causal_infection.step for the cancerous gate rationale
            # (drops scheduled-but-not-realized onsets and same-tick competing
            # deaths; on grow, fine agents with competing mortality). BoolArr.uids
            # returns active (alive) agents only, so the alive filter is implicit.
            new = ((m.ti_cancerous == ti) & m.cancerous).uids
            if not len(new):
                continue
            # Defensive isfinite guard (symmetric with age_causal_infection):
            # a gated agent always has finite ti_dead_cancer today, but a future
            # non-fatal-cancer or reschedule path could leave it NaN — without
            # this filter a single NaN death_age would poison the whole onset
            # year's YLL/YLD sum.
            ti_dead = m.ti_dead_cancer[new]
            ok = np.isfinite(ti_dead)
            new, ti_dead = new[ok], ti_dead[ok]
            if not len(new):
                continue
            cancer_age = people.age[new]
            death_age = cancer_age + (ti_dead - ti) * dt
            w = people.scale[new] if scale is not None else np.ones(len(new))
            self._accumulate(year, cancer_age, death_age, w)

    def finalize(self):
        super().finalize()
        # yll/yld accumulated with per-agent people.scale weights; multiply
        # by sim.pars.pop_scale so absolute population DALYs match
        # sim.results.* magnitudes (removes the "callers multiply" caveat).
        pop_scale = self.sim.pars.pop_scale
        self.yll *= pop_scale
        self.yld *= pop_scale
        self.dalys = self.yll + self.yld

    def plot(self, fig=None):
        """Plot stacked YLL/YLD over time (see plotting.plot_dalys)."""
        from .plotting import plot_dalys
        return plot_dalys(self, fig=fig)


def results_by_genotype(sim, key='cum_cancers', normalize=False):
    """Stack a per-genotype HPV result into a year-indexed DataFrame.

    Columns are genotype names; index is ``sim.timevec.years``. With
    ``normalize=True``, each row is divided by its total (genotype distribution
    of `key`), leaving all-zero rows as zeros.

    Args:
        sim: a run hpv.Sim.
        key: a result name present on each HPV module (e.g. 'cum_cancers',
            'new_cancers', 'cum_cancer_deaths').
        normalize (bool): row-normalize to a distribution.
    """
    from .hpv import HPV
    mods = [d for d in sim.diseases.values() if isinstance(d, HPV)]
    data = {m.name: np.asarray(m.results[key], dtype=float) for m in mods}
    df = pd.DataFrame(data, index=pd.Index(sim.timevec.years, name='year'))
    if normalize:
        totals = df.sum(axis=1)
        df = df.div(totals.where(totals > 0, 1.0), axis=0)
    return df
