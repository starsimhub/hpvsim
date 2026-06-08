"""HPVsim plotting helpers built on matplotlib + Starsim's built-in plotting."""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

from .analyzers import AgeResults, results_by_genotype


__all__ = ['plot_by_age', 'plot_by_genotype', 'plot_type_distribution', 'plot_sim',
           'plot_intervention_impact', 'plot_calibration']


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
    """Plot an AgeResults result as a series per year over age bins.

    Args:
        age_results: a run ``hpv.AgeResults`` analyzer.
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

    `source` may be a run ``hpv.Sim`` (uses ``results_by_genotype``) or an
    ``hpv.AgeResults`` analyzer (uses its type-distribution result `key`, e.g.
    'cancerous_genotype_dist'). `year` defaults to the last recorded year.

    Note: the default ``key='cum_cancers'`` is only valid for the Sim source;
    for an AgeResults source pass a type-distribution key like
    'cancerous_genotype_dist'.
    """
    if isinstance(source, AgeResults):
        df = source.to_dataframe(key, normalize=True)  # index=year, cols=genotypes
    else:
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


def _best_fit_sim(calib):
    """Rebuild and run the sim at the calibration's best parameters
    (mirrors ss.Calibration.plot_final)."""
    pars = sc.dcp(calib.calib_pars)
    for parname, spec in pars.items():
        # Only overwrite 'value' for parameters that Optuna actually sampled
        # (i.e. those present in best_pars). Parameters supplied with a fixed
        # 'value' in the spec are skipped by _sample_from_trial and therefore
        # never appear in study.best_params.
        if parname in calib.best_pars:
            spec['value'] = calib.best_pars[parname]
    sim = calib.build_fn(calib.sim.copy(), calib_pars=pars, **calib.build_kw)
    sim.run()
    if isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]
    return sim


def plot_calibration(calib, sim=None, fig=None):
    """Overlay simulated vs observed for each calibration target (data-vs-fit).

    `calib` is a run ``hpv.Calibration`` (after ``calibrate()``). If `sim` is
    None, the best-fit sim is rebuilt at ``calib.best_pars`` and run once.
    `fig` is honored if provided. Convergence and parameter-distribution views
    are available directly via ``calib.plot_optuna()`` and ``calib.plot_final()``.
    """
    from .calibration import _find_age_results as _find_ar, _extract_actual
    data = (calib.eval_kw or {}).get('data')
    if not data:
        raise ValueError('plot_calibration: calibration has no target data '
                         "(expected calib.eval_kw['data']).")
    if sim is None:
        sim = _best_fit_sim(calib)
    elif isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]
    ar = _find_ar(sim)
    keys = list(data.keys())
    fig = fig or plt.figure(figsize=(5 * len(keys), 4))
    for i, key in enumerate(keys):
        ax = fig.add_subplot(1, len(keys), i + 1)
        expected = data[key]
        actual = _extract_actual(ar, key, expected)
        for col in expected.columns:
            ax.plot(expected.index, expected[col].values, 'o', label=f'data {col}')
            ax.plot(actual.index, actual[col].values, '-', label=f'fit {col}')
        ax.set_title(key)
        ax.set_xlabel('Year')
    fig.tight_layout()
    return fig


def _find_age_results_or_none(sim):
    """Return the first AgeResults analyzer on `sim`, or None."""
    for a in sim.analyzers.values():
        if isinstance(a, AgeResults):
            return a
    return None


def plot_sim(sim, which='default', fig=None, **kwargs):
    """HPV-specific summary figure.

    which='default': 4-panel canonical figure (cumulative cancers over time,
    prevalence by age, cancers by age, genotype distribution). Requires an
    AgeResults analyzer recording a prevalence key and 'cancers'.
    Any other value (e.g. 'all') delegates to ss.Sim.plot.
    `fig` is honored in both modes (a new figure is created if None).
    """
    if which != 'default':
        return sim.plot(fig=fig, **kwargs)

    ar = _find_age_results_or_none(sim)
    if ar is None:
        raise ValueError(
            "plot_sim(which='default') needs an AgeResults analyzer recording "
            "'cancers' and a prevalence key (one of %s). Add e.g. "
            "hpv.AgeResults(result_args=...) to the sim, or call "
            "plot_sim(sim, which='all')." % (_PREV_KEYS,))
    prev_key = next((k for k in _PREV_KEYS if k in ar.outputs), None)
    if prev_key is None or 'cancers' not in ar.outputs:
        raise ValueError(
            "plot_sim(which='default') needs the AgeResults analyzer to record "
            "'cancers' and one of %s; found %s." % (_PREV_KEYS, list(ar.outputs)))

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
