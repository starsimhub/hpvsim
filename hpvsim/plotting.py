"""HPVsim plotting helpers built on matplotlib + Starsim's built-in plotting."""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

from .analyzers import AgeResults, results_by_genotype


__all__ = ['plot_by_age']


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
