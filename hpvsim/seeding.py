"""
Initial HPV prevalence

HPVsim 3 seeds each agent from age- and sex-banded curves. Starsim 4 seeds a declared disease
with one share per group, so here each genotype's initial prevalence is the mean of its curve over
the starting population (TODO(v4-rs): seed by the curves once init_prev takes a property of people
that is not a stratification).

    'exclusive'    the 'total' curve, shared between the genotypes (equally, or by init_hpv_dist)
    'independent'  each genotype's own curve
"""

import numpy as np

__all__ = ['init_prev_curves', 'mean_init_prev', 'init_prevs']

# Initial HPV prevalence by age bracket (inclusive lower bounds, last extends to 150) and sex
_INIT_HPV_PREV_AGE_BRACKETS = np.array([12, 17, 24, 34, 44, 64, 80, 150])

_INIT_PREV = {
    'total': {
        'm': np.array([0.0, 0.25, 0.60, 0.25, 0.05, 0.01, 0.0005, 0.0]),
        'f': np.array([0.0, 0.35, 0.70, 0.25, 0.05, 0.01, 0.0005, 0.0]),
    },
    'hpv18': {
        'm': np.array([0.0, 0.15, 0.36, 0.15, 0.03, 0.006, 0.0003, 0.0]),
        'f': np.array([0.0, 0.21, 0.42, 0.15, 0.03, 0.006, 0.0003, 0.0]),
    },
    'hi5': {
        'm': np.array([0.0, 0.10, 0.24, 0.10, 0.02, 0.004, 0.0002, 0.0]),
        'f': np.array([0.0, 0.14, 0.28, 0.10, 0.02, 0.004, 0.0002, 0.0]),
    },
    'ohr': {
        'm': np.array([0.0, 0.10, 0.24, 0.10, 0.02, 0.004, 0.0002, 0.0]),
        'f': np.array([0.0, 0.14, 0.28, 0.10, 0.02, 0.004, 0.0002, 0.0]),
    },
}
_INIT_PREV['hpv16'] = _INIT_PREV['total']


def init_prev_curves(key):
    """ The (female, male) prevalence curves of a genotype, or of 'total', by age bracket """
    c = _INIT_PREV[key]
    return c['f'], c['m']


def mean_init_prev(key, age_data=None):
    """
    The mean of a prevalence curve over a population

    Args:
        key (str): a genotype, or 'total'
        age_data (DataFrame): columns [age, value], the population by single year of age; None for uniform ages 0-60
    """
    if age_data is None:
        ages, weights = np.arange(60) + 0.5, np.ones(60)
    else:
        ages = np.asarray(age_data['age'], dtype=float) + 0.5
        weights = np.asarray(age_data['value'], dtype=float)
    idx = np.clip(np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, ages, side='right'), 0, 7)
    f, m = init_prev_curves(key)
    prev = (f[idx] + m[idx]) / 2
    return float((prev * weights).sum() / weights.sum())


def init_prevs(keys, init_seeding='exclusive', init_hpv_dist=None, age_data=None):
    """ The initial prevalence of each genotype, as {genotype: share} """
    keys = list(keys)
    if init_seeding == 'independent':
        return {k: mean_init_prev(k, age_data) for k in keys}
    elif init_seeding == 'exclusive':
        total = mean_init_prev('total', age_data)
        w = np.ones(len(keys)) if init_hpv_dist is None else np.array([init_hpv_dist[k] for k in keys], dtype=float)
        w = w / w.sum()
        return {k: total * wk for k, wk in zip(keys, w)}
    raise ValueError(f"init_seeding must be 'exclusive' or 'independent'; got {init_seeding!r}")
