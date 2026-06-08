"""HPVsim plotting helpers built on matplotlib + Starsim's built-in plotting."""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

from .analyzers import AgeResults, results_by_genotype


__all__ = ['plot_by_age', 'plot_by_genotype', 'plot_type_distribution']


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
    ax.set_ylabel('Share of cancers')
    ax.set_title(f'Genotype distribution ({key})')
    return fig
