"""Canonical hpvsim example simulations.

Follows the ``hivsim.demo`` pattern (hivsim/sim.py:111-148): each example
is its own submodule providing ``make_sim(**kwargs) -> hpv.Sim``. The
``EXAMPLES`` registry maps a short name to the submodule import path.
``hpv.demo(name=None)`` looks up the entry and delegates.

New examples are added by:
    1. Creating ``hpvsim/examples/<name>/sim.py`` with ``make_sim(**kwargs)``.
    2. Appending ``<name>`` to ``EXAMPLES``.
"""
import sciris as sc


__all__ = ['EXAMPLES', 'demo']


EXAMPLES = ('nigeria',)


def demo(name=None, run=True, plot=True, **kwargs):
    """Create (and optionally run) a canonical hpvsim example sim.

    Args:
        name (str): example key. Defaults to ``'nigeria'`` -- a fully
            configured Nigeria sim (four HPV genotypes, WPP demographics,
            default sexual network). Must be one of ``EXAMPLES``.
        run (bool): whether to run the sim before returning.
        plot (bool): whether to plot after running (only if ``run`` is True).
        **kwargs: forwarded to the example's ``make_sim``.

    Examples::

        import hpvsim as hpv
        hpv.demo()                              # canonical Nigeria; runs + plots
        hpv.demo('nigeria', run=False)          # build only
        hpv.demo('nigeria', n_agents=5_000)     # override make_sim kwargs
    """
    if name is None:
        name = 'nigeria'
    if name not in EXAMPLES:
        raise ValueError(
            f"hpv.demo: unknown example {name!r}; available: {sorted(EXAMPLES)}"
        )
    mod = sc.importbyname(f'hpvsim.examples.{name}.sim')
    sim = mod.make_sim(**kwargs)
    if run:
        sim.run()
        if plot:
            try:
                sim.plot()
            except Exception:
                pass
    return sim
