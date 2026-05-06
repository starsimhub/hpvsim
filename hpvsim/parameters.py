"""HPV simulation parameters.

``SimPars`` subclasses ``ss.SimPars`` with HPV-specific defaults.
``GenotypePars`` holds per-genotype natural-history defaults (durations,
severity functions, immunity, age risk) consumed by ``HPV(ss.Infection)``;
``get_genotype_pars(genotype)`` is the factory multi-genotype consumers
use to look up defaults by name.

Currently ships HPV16 defaults only; future multi-genotype support adds
hpv18 / hi5 / ohr.
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['SimPars', 'GenotypePars', 'get_genotype_pars',
           'get_cross_immunity', 'genotype_aliases', 'GENOTYPE_KEYS']

# Canonical 4-genotype ordering for M03's default Sim. The Connector uses
# this order as the default when no genotype list is supplied.
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


def _imm_init_dist():
    """Beta sample for per-clearance humoral-immunity boost.

    Shape parameters from v2's beta_mean(par1=0.35, par2=0.025).
    """
    a = ((1 - 0.35) / 0.025 - 1 / 0.35) * 0.35 ** 2
    b = a * (1 / 0.35 - 1)
    return ss.Dist(distname='beta', a=a, b=b)


def _cell_imm_dist():
    """Beta sample for per-clearance severity-immunity boost (v2 cell_imm_init).

    Shape parameters from v2's beta_mean(par1=0.25, par2=0.025).
    """
    a = ((1 - 0.25) / 0.025 - 1 / 0.25) * 0.25 ** 2
    b = a * (1 / 0.25 - 1)
    return ss.Dist(distname='beta', a=a, b=b)


class GenotypePars(ss.Pars):
    """Per-genotype natural-history defaults. Currently HPV16 only.

    Each call returns fresh distribution instances so per-genotype RNG
    state stays independent.
    """

    def __init__(self, genotype='hpv16', **kwargs):
        super().__init__()
        self.genotype = genotype
        if genotype == 'hpv16':
            # Per-sex-act probability; SexualNetwork applies per-act via
            # 1 - (1-p)**acts.
            self.beta = 0.25
            # Female natural-history durations (lognormal mean / std in years).
            self.dur_precin = ss.lognorm_ex(mean=ss.years(3.0), std=ss.years(9.0))
            self.dur_cin = ss.lognorm_ex(mean=ss.years(5.0), std=ss.years(20.0))
            self.dur_cancer = ss.lognorm_ex(mean=ss.years(8.0), std=ss.years(3.0))
            # Males clear via this distribution without entering CIN/cancer.
            self.dur_inf_male = ss.lognorm_ex(mean=ss.years(1.0), std=ss.years(1.0))
            # Severity functions consumed by _compute_severity. cancer_fn
            # carries cin_fn's keys so the cin_integral branch can call
            # _compute_severity_integral on the same logf2 internally.
            self.cin_fn = dict(form='logf2', k=0.3, x_infl=0, ttc=50)
            self.cancer_fn = dict(method='cin_integral', transform_prob=2e-3,
                                  form='logf2', k=0.3, x_infl=0, ttc=50)
            # Same-genotype partial permanent immunity. imm_init is sampled
            # per-clearance and feeds nab_imm (M03 Connector source state).
            self.imm_init = _imm_init_dist()
            self.cell_imm_init = _cell_imm_dist()
            # Women aged >= ``age`` get their dur_cin scaled by ``risk``,
            # shifting cancer onset to older ages.
            self.age_risk = dict(age=30, risk=2)
            # Reserved for multi-genotype: per-genotype beta scaler and
            # serology probability. Currently unused by HPV.
            self.rel_beta = 1.0
            self.sero_prob = 0.75
        else:
            raise NotImplementedError(
                f'GenotypePars currently supports hpv16 only; got {genotype!r}.'
            )
        self.update(kwargs)
        return


def get_genotype_pars(genotype='hpv16'):
    """Return per-genotype natural-history defaults."""
    return GenotypePars(genotype=genotype)


# v2 defaults (see hpvsim/_v2_legacy/parameters.py:108-112)
_DEFAULT_CROSS_IMM_SUS_MED = 0.3
_DEFAULT_CROSS_IMM_SUS_HIGH = 0.5
_DEFAULT_CROSS_IMM_SEV_MED = 0.5
_DEFAULT_CROSS_IMM_SEV_HIGH = 0.7

# Pairwise cross-protection clade map. 'high' = hpv16 <-> hpv18; everything
# else is 'med'. Hand-ported from v2's get_cross_immunity (hpvsim/_v2_legacy/
# parameters.py:412-508).
_CLADE_HIGH_PAIRS = frozenset({
    ('hpv16', 'hpv18'),
    ('hpv18', 'hpv16'),
})


def _build_cross_matrix(keys, scalar_med, scalar_high):
    """Pairwise cross-protection matrix; diagonal forced to 1.0."""
    n = len(keys)
    m = np.full((n, n), scalar_med, dtype=np.float32)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                m[i, j] = 1.0
            elif (ki, kj) in _CLADE_HIGH_PAIRS:
                m[i, j] = scalar_high
    return m


def get_cross_immunity(keys=None,
                       cross_imm_sus_med=None, cross_imm_sus_high=None,
                       cross_imm_sev_med=None, cross_imm_sev_high=None):
    """Build (cross_immunity_sus, cross_immunity_sev) matrices for the given
    genotype ordering.

    Returns a tuple of two ``(n, n)`` float32 arrays. ``keys`` defaults to
    ``GENOTYPE_KEYS``. Scalar defaults match v2 (`cross_imm_sus_med=0.3`,
    `cross_imm_sus_high=0.5`, `cross_imm_sev_med=0.5`, `cross_imm_sev_high=0.7`).
    Diagonals are forced to 1.0 by convention.
    """
    if keys is None:
        keys = GENOTYPE_KEYS
    if cross_imm_sus_med is None:  cross_imm_sus_med  = _DEFAULT_CROSS_IMM_SUS_MED
    if cross_imm_sus_high is None: cross_imm_sus_high = _DEFAULT_CROSS_IMM_SUS_HIGH
    if cross_imm_sev_med is None:  cross_imm_sev_med  = _DEFAULT_CROSS_IMM_SEV_MED
    if cross_imm_sev_high is None: cross_imm_sev_high = _DEFAULT_CROSS_IMM_SEV_HIGH
    m_sus = _build_cross_matrix(keys, cross_imm_sus_med, cross_imm_sus_high)
    m_sev = _build_cross_matrix(keys, cross_imm_sev_med, cross_imm_sev_high)
    return m_sus, m_sev
