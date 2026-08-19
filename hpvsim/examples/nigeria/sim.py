"""Canonical Nigeria hpvsim example.

Configured with all four HPV genotypes, UN WPP demographics for Nigeria at
the sim start year, and the default sexual network. Caller kwargs (e.g.
``n_agents``, ``start``, ``stop``, ``rand_seed``) override the defaults
below.
"""
import hpvsim as hpv


__all__ = ['make_sim']


def make_sim(**kwargs):
    """Return an unrun canonical Nigeria hpv.Sim.

    Defaults: ``n_agents=20_000``, ``start=1990``, ``stop=2020``,
    ``dt=0.25``, all four HPV genotypes, ``ms_agent_ratio=100``.
    """
    pars = dict(
        location='nigeria',
        n_agents=20_000,
        start=1990,
        stop=2020,
        dt=0.25,
        genotypes=[16, 18, 'hi5', 'ohr'],
        ms_agent_ratio=100,
    )
    pars.update(kwargs)
    return hpv.Sim(**pars)
