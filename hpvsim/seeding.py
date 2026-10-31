"""Initial HPV-prevalence curves.

The age-banded init_prev curves and ``_make_init_prev_fn`` sampler are used by
both per-genotype ``HPV`` modules (for ``init_seeding='independent'``) and by
``total_init_prev`` (for ``init_seeding='exclusive'``, which uses the 'total'
curve as the chance of any HPV, before the family assigns a genotype).
"""

import numpy as np

from .network import SexualNetwork


# Initial HPV prevalence by age bracket (inclusive lower bounds, last extends to
# 150) and sex. 'total' seeds 'exclusive' mode; per-genotype seeds 'independent'.
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


def _make_init_prev_fn(key):
    """Return the per-uid init-prev sampler for a given _INIT_PREV key.

    Accepts a genotype key ('hpv16', 'hpv18', 'hi5', 'ohr') for per-genotype
    seeding or 'total' for the total-HPV curve.
    """
    curves = _INIT_PREV[key]
    f_curve = curves['f']
    m_curve = curves['m']

    def _age_stratified(sim, uids):
        age = sim.people.age[uids]
        is_female = sim.people.female[uids]
        bin_idx = np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, age, side='right')
        bin_idx = np.clip(bin_idx, 0, len(f_curve) - 1)
        out = np.zeros(len(uids))
        out[is_female] = f_curve[bin_idx[is_female]]
        out[~is_female] = m_curve[bin_idx[~is_female]]
        return out
    return _age_stratified


def total_init_prev(sim, uids):
    """The chance of any HPV at the start: the 'total' curve, for those past sexual debut.

    For ``init_seeding='exclusive'``, where the family (``hpv.make_family()``) gives each
    person seeded one genotype, so nobody is co-infected at the start.
    """
    p = _make_init_prev_fn('total')(sim, uids)
    net = next((n for n in sim.networks.values() if isinstance(n, SexualNetwork)), None)
    if net is not None:
        net.set_network_states() # Diseases are seeded before networks are initialized, so debut may not be sampled yet
        p = p*net.active(sim.people)[uids]
    return p
