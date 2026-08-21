"""HPVsim plotting helpers built on matplotlib + Starsim's built-in plotting."""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

from .analyzers import by_age, results_by_genotype


__all__ = ['plot_by_age', 'plot_by_genotype', 'plot_type_distribution', 'plot_sim',
           'plot_intervention_impact', 'plot_calibration',
           'plot_age_pyramid', 'plot_age_causal_infection', 'plot_dalys']


class _FigProxy:
    """Adapts a single Axes to the (fig.gca())-based helper interface."""
    def __init__(self, ax):
        self._ax = ax
    def gca(self):
        return self._ax


def _new_fig_ax(fig=None, figsize=(7, 5)):
    """Return (fig, ax): use the supplied fig's current axes, or make one."""
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.gca()
    return fig, ax


def plot_by_age(age_results, key, years=None, kind='line', fig=None, **kwargs):
    """Plot an by_age result as a series per year over age bins.

    Args:
        age_results: a run ``hpv.by_age`` analyzer.
        key: a result name recorded by that analyzer (e.g. 'cancers').
        years: scalar/list of years to plot; default = all recorded years.
        kind: 'line' or 'bar'.
    """
    df = age_results.to_dataframe(key)            # index=year, cols=age-bin labels
    if years is not None:
        df = df.loc[[float(y) for y in sc.tolist(years)]]
    fig, ax = _new_fig_ax(fig)
    x = np.arange(len(df.columns))
    if kind == 'bar':
        kwargs.setdefault('alpha', 0.6)
    for yr, row in df.iterrows():
        if kind == 'bar':
            ax.bar(x, row.values, label=f'{yr:g}', **kwargs)
        else:
            ax.plot(x, row.values, marker='o', label=f'{yr:g}', **kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_xlabel('Age group')
    ax.set_ylabel(key)
    ax.set_title(f'{key} by age')
    ax.legend(title='Year')
    return fig


def plot_by_genotype(sim, key='cum_cancers', normalize=False, fig=None, **kwargs):
    """Overlay a per-genotype result across genotypes over time.

    Lines (one per genotype) by default; stacked areas when ``normalize=True``
    (per-year genotype shares summing to 1).
    """
    df = results_by_genotype(sim, key=key, normalize=normalize)
    fig, ax = _new_fig_ax(fig)
    x = df.index.values
    if normalize:
        ax.stackplot(x, *[df[c].values for c in df.columns], labels=list(df.columns), **kwargs)
    else:
        for c in df.columns:
            ax.plot(x, df[c].values, label=c, **kwargs)
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{key} (share)' if normalize else key)
    ax.set_title(f'{key} by genotype')
    ax.legend(title='Genotype')
    return fig


def plot_type_distribution(source, year=None, key='cum_cancers', fig=None, **kwargs):
    """Bar chart of each genotype's share of `key` at a given year.

    `source` is a run ``hpv.Sim``. Delegates to ``results_by_genotype``. For
    a year with zero cancers the normalized shares are all 0 (not 1).
    """
    df = results_by_genotype(source, key=key, normalize=True)
    if year is None:
        row = df.iloc[-1]
    else:
        yrs = np.asarray(df.index, dtype=float)
        row = df.iloc[int(np.argmin(np.abs(yrs - float(year))))]
    fig, ax = _new_fig_ax(fig)
    x = np.arange(len(row))
    ax.bar(x, row.values, **kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels(list(row.index))
    ax.set_ylabel(f'{key} (share)')
    ax.set_title(f'Genotype distribution ({key})')
    return fig


def _genotype_total_trajectories(obj, key):
    """Return (years, matrix) where matrix is (n_sims, n_years) of `key`
    summed across genotypes. Works for an ss.Sim or an ss.MultiSim."""
    sims = obj.sims if isinstance(obj, ss.MultiSim) else [obj]
    rows = []
    years = None
    for s in sims:
        df = results_by_genotype(s, key=key)
        years = np.asarray(df.index, dtype=float)
        rows.append(df.sum(axis=1).values)
    return years, np.vstack(rows)


def plot_intervention_impact(baseline, scenario, key='cum_cancers',
                             labels=('baseline', 'scenario'), fig=None):
    """Compare a baseline vs an intervention scenario.

    `baseline`/`scenario` are each an ss.Sim or ss.MultiSim. Top panel: median
    `key` trajectory per arm (10/90 band when a MultiSim); bottom panel:
    averted = baseline_median - scenario_median. Raises if the two arms have
    different timevecs.
    """
    yb, mb = _genotype_total_trajectories(baseline, key)
    ys, ms = _genotype_total_trajectories(scenario, key)
    if not np.array_equal(yb, ys):
        raise ValueError('plot_intervention_impact: baseline and scenario have '
                         'different timevecs')
    fig = fig or plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    for y, m, lab in [(yb, mb, labels[0]), (ys, ms, labels[1])]:
        med = np.median(m, axis=0)
        ax1.plot(y, med, label=lab)
        if m.shape[0] > 1:
            ax1.fill_between(y, np.quantile(m, 0.1, axis=0),
                             np.quantile(m, 0.9, axis=0), alpha=0.2)
    averted = np.median(mb, axis=0) - np.median(ms, axis=0)
    ax2.plot(yb, averted, color='k')
    ax2.axhline(0, ls='--', color='grey')
    ax1.set_ylabel(key)
    ax1.set_title('Intervention impact')
    ax1.legend()
    ax2.set_ylabel(f'{key} averted')
    ax2.set_xlabel('Year')
    return fig


_PREV_KEYS = ('hpv_prevalence', 'precin_prevalence', 'cin_prevalence')


def _run_top_n_trials(calib, n):
    """Re-run the top-``n`` trials via ``make_calib_sims`` and return a list
    of per-trial DataFrames shaped like ``calib.eval_kw['data']`` (year
    index + dot-scoped columns). Uses ``extract_fn`` so full sims are not
    pickled back to the parent process."""
    from .calibration import make_calib_sims, _extract_columns
    data = calib.eval_kw['data']
    return make_calib_sims(
        calib, n=n, extract_fn=lambda sim: _extract_columns(sim, data),
    )


def _group_columns(cols):
    """Group standardized calib data columns by (scope, name). Preserves
    within-group column order."""
    groups = {}
    for c in cols:
        parts = c.split('.')
        key = (parts[0], parts[1])
        groups.setdefault(key, []).append(c)
    return groups


def _plot_age_panel(ax, name, cols, expected, actual_list, top_n):
    """Age-stratified panel: x = age bins, y = value. Median + 95% PI band
    across top-N trials; data as diamonds."""
    bin_labels = [c.split('.', 2)[-1] for c in cols]
    x = np.arange(len(bin_labels))
    for yr in expected.index:
        exp = expected.loc[yr, cols].to_numpy(dtype=float)
        stack = np.array([a.loc[yr, cols].to_numpy(dtype=float) for a in actual_list])
        med = np.median(stack, axis=0)
        lo = np.percentile(stack, 2.5, axis=0)
        hi = np.percentile(stack, 97.5, axis=0)
        yr_lab = f' ({int(yr)})' if len(expected.index) > 1 else ''
        ax.fill_between(x, lo, hi, alpha=0.25,
                        label=f'Top-{top_n} 95% PI{yr_lab}')
        ax.plot(x, med, marker='o', lw=2,
                label=f'Top-{top_n} median{yr_lab}')
        ax.scatter(x, exp, marker='d', s=60, color='k', zorder=3,
                   label=f'Data{yr_lab}')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlabel('Age')
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.legend(fontsize=8, loc='best')


def _plot_scalar_ts(ax, name, col, expected, actual_list, top_n):
    """Scalar-per-year panel: x = year, y = value. Timeseries with
    median + 95% PI ribbon; data as diamonds. Single-year targets get an
    errorbar-style point instead of a ribbon."""
    years = expected.index.values
    stack = np.array([a[col].reindex(expected.index).to_numpy(dtype=float)
                      for a in actual_list])
    med = np.median(stack, axis=0)
    lo = np.percentile(stack, 2.5, axis=0)
    hi = np.percentile(stack, 97.5, axis=0)
    if len(years) > 1:
        ax.fill_between(years, lo, hi, alpha=0.25, label=f'Top-{top_n} 95% PI')
        ax.plot(years, med, lw=2, label=f'Top-{top_n} median')
        ax.scatter(years, expected[col].values, marker='d', s=60, color='k',
                   label='Data', zorder=3)
        ax.set_xlim(years.min(), years.max())
    else:
        y0 = float(years[0])
        yerr = np.array([[med[0] - lo[0]], [hi[0] - med[0]]])
        ax.errorbar([y0], [med[0]], yerr=yerr, marker='o', capsize=6, lw=2,
                    label=f'Top-{top_n} median (95% PI)')
        ax.scatter([y0], [expected[col].iloc[0]], marker='d', s=80,
                   color='k', zorder=3, label='Data')
    ax.set_xlabel('Year')
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.legend(fontsize=8, loc='best')


def _plot_genotype_box(ax, name, cols, expected, actual_list, top_n):
    """Per-genotype panel: x = genotype, y = value. Box plot of top-N
    trial distributions (aggregated across any years in the data);
    data as diamonds."""
    genotypes = [c.split('.', 2)[-1] for c in cols]
    boxes = []
    for c in cols:
        vals = np.concatenate([a[c].reindex(expected.index).dropna().to_numpy()
                               for a in actual_list])
        boxes.append(vals)
    x = np.arange(1, len(genotypes) + 1)
    ax.boxplot(boxes, tick_labels=genotypes, showfliers=False, widths=0.5)
    for i, c in enumerate(cols):
        exp_vals = expected[c].dropna().values
        ax.scatter([x[i]] * len(exp_vals), exp_vals, marker='d', s=70,
                   color='k', zorder=3, label='Data' if i == 0 else None)
    ax.set_xlabel('Genotype')
    ax.set_ylabel(name)
    ax.set_title(f'{name} (top-{top_n} trials)')
    ax.legend(fontsize=8, loc='best')


def plot_calibration(calib, top_n=50, fig=None, ncols=None):
    """Data-vs-fit plot for a run ``hpv.Calibration``.

    Auto-inspects ``calib.eval_kw['data']`` and renders one panel per
    (scope, name) target group, overlaying a top-``top_n``-trial model
    ribbon / box on the observed data.

    Panel layout by scope:
      - ``all_hpv.<name>.<bin>``   age-stratified: line + 95% PI band vs.
        data scatter, x = age bins.
      - ``all_hpv.<name>``         scalar: timeseries if multi-year, else
        an errorbar point.
      - ``by_genotype.<name>.<g>`` box plot per genotype vs. data markers.

    Args:
        calib: run ``hpv.Calibration``.
        top_n (int): number of best-mismatch trials to re-run for the
            ribbon (default 50). Clamped to available trials.
        fig: matplotlib Figure to draw into (default: new).
        ncols (int): grid ncols (default: min(n_panels, 3)).
    """
    data = (calib.eval_kw or {}).get('data')
    if data is None or data.empty:
        raise ValueError('plot_calibration: calibration has no target data '
                         "(expected calib.eval_kw['data']).")

    groups = _group_columns(data.columns)
    actual_list = _run_top_n_trials(calib, top_n)
    top_n_actual = len(actual_list)

    n_panels = len(groups)
    ncols = ncols or min(n_panels, 3)
    nrows = -(-n_panels // ncols)
    fig = fig or plt.figure(figsize=(6.0 * ncols, 4.5 * nrows))

    for i, ((scope, name), cols) in enumerate(groups.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        if scope == 'all_hpv':
            is_age = all(len(c.split('.')) == 3 for c in cols)
            if is_age:
                _plot_age_panel(ax, name, cols, data, actual_list, top_n_actual)
            else:
                _plot_scalar_ts(ax, name, cols[0], data, actual_list, top_n_actual)
        elif scope == 'by_genotype':
            _plot_genotype_box(ax, name, cols, data, actual_list, top_n_actual)
        else:
            ax.set_title(f'{scope}.{name} (unknown scope)')
    fig.tight_layout()
    return fig


def plot_age_pyramid(age_pyramid_az, date=None, fig=None):
    """Back-to-back male/female age-pyramid bars from an age_pyramid analyzer."""
    pyr = age_pyramid_az.age_pyramids
    if not len(pyr):
        raise ValueError('plot_age_pyramid: analyzer recorded no pyramids')
    key = list(pyr.keys())[0] if date is None else min(
        pyr.keys(), key=lambda k: abs(k.years - float(ss.date(date).years)))
    arr = pyr[key]                                  # (nbins, 2): male, female
    labels = age_pyramid_az.age_labels
    fig, ax = _new_fig_ax(fig)
    y = np.arange(len(labels))
    ax.barh(y, -arr[:, 0], label='male')
    ax.barh(y, arr[:, 1], label='female')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Count (male left / female right)')
    ax.set_title(f'Age pyramid {key}')
    ax.legend()
    return fig


def plot_age_causal_infection(aci, fig=None):
    """Histograms of age at causal infection / CIN2+ / cancer and the precin /
    cin / total dwell-time distributions (weight-aware)."""
    fig = fig or plt.figure(figsize=(10, 7))
    w = aci.weights if len(aci.weights) == len(aci.age_cancer) else None
    age_series = [('age_causal', aci.age_causal), ('age_cin', aci.age_cin),
                  ('age_cancer', aci.age_cancer)]
    for i, (name, vals) in enumerate(age_series):
        ax = fig.add_subplot(2, 3, i + 1)
        vals = np.asarray(vals, dtype=float)
        if len(vals):
            ax.hist(vals, bins=20, weights=w, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel('Age')
    for j, state in enumerate(('precin', 'cin', 'total')):
        ax = fig.add_subplot(2, 3, j + 4)
        vals = np.asarray(aci.dwelltime[state], dtype=float)
        if len(vals):
            ax.hist(vals, bins=20, weights=w, alpha=0.7, color='#9e1149')
        ax.set_title(f'dwelltime: {state}')
        ax.set_xlabel('Years')
    fig.tight_layout()
    return fig


def plot_dalys(dal, fig=None):
    """Stacked YLL/YLD over the analyzer's year axis."""
    fig, ax = _new_fig_ax(fig)
    ax.bar(dal.years, dal.yld, label='YLD')
    ax.bar(dal.years, dal.yll, bottom=dal.yld, label='YLL')
    ax.set_xlabel('Year')
    ax.set_ylabel('DALYs')
    ax.set_title('DALYs (YLL + YLD)')
    ax.legend()
    return fig


def _find_by_age_or_none(sim):
    """Return the first by_age analyzer on `sim`, or None."""
    for a in sim.analyzers.values():
        if isinstance(a, by_age):
            return a
    return None


def plot_sim(sim, which='default', fig=None, **kwargs):
    """HPV-specific summary figure.

    which='default': 4-panel canonical figure (cumulative cancers over time,
    prevalence by age, cancers by age, genotype distribution). Requires an
    by_age analyzer recording a prevalence key and 'cancers'.
    Any other value (e.g. 'all') delegates to ss.Sim.plot.
    `fig` is honored in both modes (a new figure is created if None).
    Pass a fresh/empty figure: multi-panel helpers add subplots rather than
    reusing existing axes.
    """
    if which != 'default':
        return sim.plot(fig=fig, **kwargs)

    ar = _find_by_age_or_none(sim)
    if ar is None:
        raise ValueError(
            "plot_sim(which='default') needs a by_age analyzer recording "
            "'cancers' and a prevalence key (one of %s). Add e.g. "
            "hpv.by_age(['cancers', 'hpv_prevalence']) to the sim, or call "
            "plot_sim(sim, which='all')." % (_PREV_KEYS,))
    prev_key = next((k for k in _PREV_KEYS if k in ar.keys), None)
    if prev_key is None or 'cancers' not in ar.keys:
        raise ValueError(
            "plot_sim(which='default') needs the by_age analyzer to record "
            "'cancers' and one of %s; found %s." % (_PREV_KEYS, ar.keys))

    fig = fig or plt.figure(figsize=(12, 9))
    # Panel 1: total cumulative cancers over time (summed across genotypes).
    ax1 = fig.add_subplot(2, 2, 1)
    cum = results_by_genotype(sim, key='cum_cancers').sum(axis=1)
    ax1.plot(cum.index.values, cum.values, color='#5f5cd2')
    ax1.set_title('Cumulative cancers')
    ax1.set_xlabel('Year')
    # Panel 2: prevalence by age.
    ax2 = fig.add_subplot(2, 2, 2)
    plot_by_age(ar, prev_key, fig=_FigProxy(ax2))
    # Panel 3: cancers by age.
    ax3 = fig.add_subplot(2, 2, 3)
    plot_by_age(ar, 'cancers', fig=_FigProxy(ax3))
    # Panel 4: genotype distribution of cancers.
    ax4 = fig.add_subplot(2, 2, 4)
    plot_type_distribution(sim, fig=_FigProxy(ax4))
    fig.tight_layout()
    return fig
