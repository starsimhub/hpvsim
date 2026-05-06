"""HPV simulation parameters.

``SimPars`` subclasses ``ss.SimPars`` with HPV-specific defaults;
``GenotypePars`` holds per-genotype natural-history defaults consumed by the
``HPV(ss.Infection)`` module. ``get_genotype_pars(genotype)`` is the factory
multi-genotype consumers use to look up defaults by name.

Currently ships HPV16 defaults only; future multi-genotype support adds
hpv18 / hi5 / ohr.
"""

import sciris as sc
import starsim as ss


__all__ = ['SimPars', 'GenotypePars', 'get_genotype_pars', 'genotype_aliases']


# Genotype name aliases for user ergonomics.
genotype_aliases = {
    'hpv16': ['hpv16', '16'],
    'hpv18': ['hpv18', '18'],
    'hi5':   ['hi5', 'high-risk-5'],
    'ohr':   ['ohr', 'other-high-risk'],
}


class SimPars(ss.SimPars):
    """HPV-specific defaults on top of ss.SimPars."""

    def __init__(self, **kwargs):
        super().__init__()
        self.n_agents  = 10_000
        self.total_pop = None  # If set, pop_scale = total_pop / n_agents.
        self.pop_scale = None  # Computed at init if total_pop is set.

        self.start     = ss.years(1990)
        self.stop      = ss.years(2060)
        self.dt        = ss.years(0.5)
        self.rand_seed = 0

        self.location = 'nigeria'
        self.verbose = ss.options.verbose

        self.update(kwargs)
        return


class GenotypePars(ss.Pars):
    """Per-genotype natural-history defaults. Currently HPV16 only."""

    def __init__(self, genotype='hpv16', **kwargs):
        super().__init__()
        self.genotype = genotype
        if genotype == 'hpv16':
            self.dur_precin = dict(dist='lognormal', par1=3, par2=9)
            self.cin_fn     = dict(form='logf2', k=0.3, x_infl=0, ttc=50)
            self.dur_cin    = dict(dist='lognormal', par1=5, par2=20)
            self.cancer_fn  = dict(method='cin_integral', transform_prob=2e-3)
            self.rel_beta   = 1.0
            self.sero_prob  = 0.75
        else:
            raise NotImplementedError(
                f'GenotypePars currently supports hpv16 only; got {genotype!r}.'
            )
        self.update(kwargs)
        return


def get_genotype_pars(genotype='hpv16'):
    """Return per-genotype natural-history defaults."""
    return GenotypePars(genotype=genotype)