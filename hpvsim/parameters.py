"""HPV simulation parameters.

``SimPars`` subclasses ``ss.SimPars`` with HPV-specific defaults.
``GenotypePars`` holds per-genotype natural-history defaults (durations,
severity functions, immunity, age risk) consumed by ``HPV(ss.Infection)``;
``get_genotype_pars(genotype)`` is the factory multi-genotype consumers
use to look up defaults by name. Supports the four canonical genotypes:
``hpv16``, ``hpv18``, ``hi5`` (high-risk-5 pool), and ``ohr`` (other
high-risk pool).
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['SimPars', 'GenotypePars', 'get_genotype_pars',
           'get_cross_immunity', 'genotype_aliases', 'GENOTYPE_KEYS']

# Canonical 4-genotype ordering. The Connector uses this order as the
# default when no genotype list is supplied.
GENOTYPE_KEYS = ('hpv16', 'hpv18', 'hi5', 'ohr')

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


def _beta_from_mean_var(mean, var):
    """Build a beta ``ss.Dist`` from (mean, variance).

    Converts to the beta distribution's shape parameters (a, b):
        a = ((1 - mean) / var - 1 / mean) * mean ** 2
        b = a * (1 / mean - 1)
    Valid for mean in (0, 1) and var < mean * (1 - mean).

    Uses the ``ss.beta_dist`` factory rather than a bare
    ``ss.Dist(distname='beta', ...)``: starsim >=3.5 no longer resolves a
    name-only Dist to its scipy object, leaving ``self.dist=None`` so ``.rvs()``
    raises ``AttributeError: 'NoneType' ... 'ppf'``. ``ss.beta_dist`` wires the
    scipy dist explicitly and exists on both 3.4 and 3.5.
    """
    a = ((1 - mean) / var - 1 / mean) * mean ** 2
    b = a * (1 / mean - 1)
    return ss.beta_dist(a=a, b=b)


def _imm_init_dist(mean=0.35, var=0.025):
    """Beta sample for per-clearance humoral-immunity boost.

    Defaults are the calibrated values. Callers wanting a different
    immunity-boost distribution can construct their own and assign it to
    ``GenotypePars.imm_init`` (or pass via ``pars=``) on the per-genotype HPV
    module.
    """
    return _beta_from_mean_var(mean, var)


def _cell_imm_dist(mean=0.25, var=0.025):
    """Beta sample for per-clearance severity-immunity boost (cell_imm_init).

    Defaults are the calibrated values. See ``_imm_init_dist`` for
    how to override per-genotype.
    """
    return _beta_from_mean_var(mean, var)


# Per-genotype natural-history defaults. Duration entries are
# ``(mean, std)`` tuples in years.
_GENOTYPE_DEFAULTS = {
    'hpv16': dict(
        beta=0.25,
        dur_precin_yr=(3.0, 9.0),
        dur_cin_yr=(5.0, 20.0),
        dur_cancer_yr=(8.0, 3.0),
        dur_inf_male_yr=(1.0, 1.0),
        cin_fn=dict(form='logf2', k=0.3, x_infl=0, ttc=50),
        cancer_fn=dict(method='cin_integral', transform_prob=2e-3,
                       form='logf2', k=0.3, x_infl=0, ttc=50),
        age_risk=dict(age=30, risk=2),
        rel_beta=1.0,
        sero_prob=0.75,
        transf2m=1.0,
        transm2f=3.69,
    ),
    'hpv18': dict(
        beta=0.25,
        dur_precin_yr=(2.5, 9.0),
        dur_cin_yr=(5.0, 20.0),
        dur_cancer_yr=(8.0, 3.0),
        dur_inf_male_yr=(1.0, 1.0),
        cin_fn=dict(form='logf2', k=0.25, x_infl=0, ttc=50),
        cancer_fn=dict(method='cin_integral', transform_prob=2e-3,
                       form='logf2', k=0.25, x_infl=0, ttc=50),
        age_risk=dict(age=30, risk=2),
        rel_beta=0.75,
        sero_prob=0.56,
        transf2m=1.0,
        transm2f=3.69,
    ),
    'hi5': dict(
        beta=0.25,
        dur_precin_yr=(2.5, 9.0),
        dur_cin_yr=(4.5, 20.0),
        dur_cancer_yr=(8.0, 3.0),
        dur_inf_male_yr=(1.0, 1.0),
        cin_fn=dict(form='logf2', k=0.2, x_infl=0, ttc=50),
        cancer_fn=dict(method='cin_integral', transform_prob=1.5e-3,
                       form='logf2', k=0.2, x_infl=0, ttc=50),
        age_risk=dict(age=30, risk=2),
        rel_beta=0.9,
        sero_prob=0.60,
        transf2m=1.0,
        transm2f=3.69,
    ),
    'ohr': dict(
        beta=0.25,
        dur_precin_yr=(2.5, 9.0),
        dur_cin_yr=(4.5, 20.0),
        dur_cancer_yr=(8.0, 3.0),
        dur_inf_male_yr=(1.0, 1.0),
        cin_fn=dict(form='logf2', k=0.2, x_infl=0, ttc=50),
        cancer_fn=dict(method='cin_integral', transform_prob=1.5e-3,
                       form='logf2', k=0.2, x_infl=0, ttc=50),
        age_risk=dict(age=30, risk=2),
        rel_beta=0.9,
        sero_prob=0.60,
        transf2m=1.0,
        transm2f=3.69,
    ),
}


def _lognorm_yr(spec):
    """Build an ``ss.lognorm_ex`` from a ``(mean, std)`` tuple in years."""
    mean, std = spec
    return ss.lognorm_ex(mean=ss.years(mean), std=ss.years(std))


class GenotypePars(ss.Pars):
    """Per-genotype natural-history defaults for HPV.

    Each call returns fresh distribution instances so per-genotype RNG
    state stays independent. Supported genotypes: ``hpv16``, ``hpv18``,
    ``hi5``, ``ohr``.

    Notes:
      - ``beta`` is per-sex-act; SexualNetwork applies per-act via
        ``1 - (1-p)**acts``.
      - Female natural-history durations are lognormal in years; males
        clear via ``dur_inf_male`` without entering CIN/cancer.
      - ``cancer_fn`` carries ``cin_fn``'s keys so the ``cin_integral``
        branch can call ``compute_severity_integral`` on the same logf2.
      - ``imm_init`` is sampled per-clearance and feeds ``nab_imm``
        (read by the CrossImmunity Connector).
      - ``age_risk['age']``-and-older women get ``dur_cin`` scaled by
        ``age_risk['risk']``, shifting cancer onset to older ages.
      - ``transf2m`` / ``transm2f`` are sex-directional per-act scalars
        — same across genotypes (act-level, not genotype-level).
    """

    def __init__(self, genotype='hpv16', **kwargs):
        super().__init__()
        if genotype not in _GENOTYPE_DEFAULTS:
            raise ValueError(
                f'GenotypePars supports {GENOTYPE_KEYS}; got {genotype!r}.'
            )
        d = _GENOTYPE_DEFAULTS[genotype]
        self.genotype = genotype
        self.beta = d['beta']
        self.dur_precin = _lognorm_yr(d['dur_precin_yr'])
        self.dur_cin = _lognorm_yr(d['dur_cin_yr'])
        self.dur_cancer = _lognorm_yr(d['dur_cancer_yr'])
        self.dur_inf_male = _lognorm_yr(d['dur_inf_male_yr'])
        self.cin_fn = dict(d['cin_fn'])
        self.cancer_fn = dict(d['cancer_fn'])
        self.imm_init = _imm_init_dist()
        self.cell_imm_init = _cell_imm_dist()
        self.age_risk = dict(d['age_risk'])
        self.rel_beta = d['rel_beta']
        self.sero_prob = d['sero_prob']
        self.transf2m = d['transf2m']
        self.transm2f = d['transm2f']
        self.update(kwargs)
        return


def get_genotype_pars(genotype='hpv16'):
    """Return per-genotype natural-history defaults."""
    return GenotypePars(genotype=genotype)


# Genotypes whose own-immunity is hardcoded to 1.0; hi5/ohr use ``own_imm_hr``.
_FULL_OWN_IMM_KEYS = frozenset({'hpv16', 'hpv18'})

# Pairwise cross-protection clade map. 'high' = hpv16 <-> hpv18 (same clade);
# everything else is 'med'.
_CLADE_HIGH_PAIRS = frozenset({
    ('hpv16', 'hpv18'),
    ('hpv18', 'hpv16'),
})


def _build_cross_matrix(keys, scalar_med, scalar_high, own_imm_hr):
    """Pairwise cross-protection matrix.

    Diagonal: 1.0 for keys in ``_FULL_OWN_IMM_KEYS`` (hpv16, hpv18); else
    ``own_imm_hr`` (0.9 by default).
    Off-diagonal: ``scalar_high`` for clade-high pairs, ``scalar_med`` else.
    """
    n = len(keys)
    m = np.full((n, n), scalar_med, dtype=np.float32)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                m[i, j] = 1.0 if ki in _FULL_OWN_IMM_KEYS else own_imm_hr
            elif (ki, kj) in _CLADE_HIGH_PAIRS:
                m[i, j] = scalar_high
    return m


def get_cross_immunity(keys=None,
                       cross_imm_sus_med=0.3, cross_imm_sus_high=0.5,
                       cross_imm_sev_med=0.5, cross_imm_sev_high=0.7,
                       own_imm_hr=0.9):
    """Build (cross_immunity_sus, cross_immunity_sev) matrices for the given
    genotype ordering.

    Returns a tuple of two ``(n, n)`` float32 arrays. ``keys`` defaults to
    ``GENOTYPE_KEYS``.

    Diagonal: 1.0 for hpv16 and hpv18 (canonical own-immunity); ``own_imm_hr``
    for everyone else.
    """
    if keys is None:
        keys = GENOTYPE_KEYS
    m_sus = _build_cross_matrix(keys, cross_imm_sus_med, cross_imm_sus_high, own_imm_hr)
    m_sev = _build_cross_matrix(keys, cross_imm_sev_med, cross_imm_sev_high, own_imm_hr)
    return m_sus, m_sev
