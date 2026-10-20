"""
Initial HPV prevalence by age and sex, from HPVsim's curves

`hpv.genotypes(init_prev=hpv.init_prev())` seeds the family from the curve over all genotypes, each
person seeded getting one genotype (HPVsim's 'exclusive' seeding).
"""

import numpy as np

__all__ = ['init_prev']

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


def init_prev(key='total'):
    """
    The share of each person infected at the start, by age and sex, as a function of the sim and the people

    Args:
        key (str): 'total' (any genotype), or a genotype's own curve ('hpv16', 'hpv18', 'hi5', 'ohr')
    """
    curves = _INIT_PREV[key]
    def prev(sim, uids):
        ppl = sim.people
        idx = np.clip(np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, ppl.age[uids], side='right'), 0, len(_INIT_HPV_PREV_AGE_BRACKETS) - 1)
        return np.where(ppl.female[uids], curves['f'][idx], curves['m'][idx])
    return prev
