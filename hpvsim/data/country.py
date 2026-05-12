"""Country-data adapter: load location data into Starsim-shaped DataFrames.

Used by ``hpvsim.Sim`` to build People (age pyramid), Births (CBR), Deaths
(age- and sex-specific mortality), and SexualNetwork (debut, partners,
mixing, layer probs, durations, acts). Underlying data lives in
``hpvsim/data/files/`` and is loaded via ``hpvsim.data.loaders`` and the
``hpvsim.parameters`` helpers.

Reshaping summary:
- Age pyramid: (N, 3) ``(age_lower, age_upper, count)`` ndarray with single-
  year resolution -> long DataFrame ``[age, value]``.
- Birth rates: ``[year, cbr]`` per 1000 -> ``[Year, CBR]`` for ``ss.Births``.
- Death rates: per-year per-sex ndarrays -> long DataFrame ``[Time,
  AgeGrpStart, Sex, mx]`` for ``ss.Deaths``.
- Network parameters: 2 layers (m=marital, c=casual). partners and
  cross_layer are stored split by sex; this adapter exposes both sexes
  per layer.
- Distributions: ``{'dist': name, 'par1': ..., 'par2': ...}`` dicts are
  converted to Starsim Dist instances via
  ``hpvsim.migration_utils._v2_dist_to_starsim``.
"""

import numpy as np
import pandas as pd
import starsim as ss

from ..migration_utils import _v2_dist_to_starsim
from . import loaders as _loaders


_KNOWN_LOCATIONS = ['nigeria']


def _default_network_pars(location=None):  # noqa: ARG001  (location reserved for future per-country data)
    """Default network parameters consumed by SexualNetwork construction.

    The ``location`` argument is accepted for API symmetry and future
    per-country extension; current defaults are location-agnostic.

    Keys returned:
        debut, f_cross_layer, m_cross_layer,
        f_partners, m_partners, acts, dur_pship,
        mixing, layer_probs
    """
    debut = dict(
        f=dict(dist='normal', par1=15.0, par2=2.1),
        m=dict(dist='normal', par1=17.6, par2=1.8),
    )
    f_cross_layer = 0.185  # Annual prob of females having concurrent cross-layer relationships
    m_cross_layer = 0.760  # Annual prob of males having concurrent cross-layer relationships

    m_partners = dict(
        m=dict(dist='poisson1', par1=0.01),
        c=dict(dist='poisson1', par1=0.5),
    )
    f_partners = dict(
        m=dict(dist='poisson1', par1=0.01),
        c=dict(dist='poisson', par1=1),
    )
    acts = dict(
        m=dict(dist='neg_binomial', par1=80, par2=40),
        c=dict(dist='neg_binomial', par1=50, par2=5),
    )
    dur_pship = dict(
        m=dict(dist='neg_binomial', par1=80, par2=3),
        c=dict(dist='lognormal', par1=1, par2=2),
    )

    # Age-based act modulation. Marital peaks at 30, casual at 25; both ramp
    # linearly from debut_ratio at debut age to 1.0 at peak, then linearly to
    # retirement_ratio at retirement, then 0 beyond retirement.
    age_act_pars = dict(
        m=dict(peak=30, retirement=100, debut_ratio=0.5, retirement_ratio=0.1),
        c=dict(peak=25, retirement=100, debut_ratio=0.5, retirement_ratio=0.1),
    )

    # Age-mixing matrices (rows = female age band start, cols = male age band).
    mixing = dict(
        m=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .1, .1, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0, .5, .1, .5, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  1, .5, .5, .5, .5, .1,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0, .5,  1,  1, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0, .1,  1,  1,  2,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0, .1, .5,  1,  1,  1,  2, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1],
        ], dtype=float),
        c=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0, .5,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  0, .5,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0,  0, .5,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0,  0, .5,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
        ], dtype=float),
    )
    layer_probs = dict(
        m=np.array([
            [ 0,    5,      10,     15,    20,    25,    30,    35,     40,     45,     50,    55,    60,    65,    70,    75],
            [ 0,    0,  0.0394, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.760, 0.590, 0.344, 0.185, 0.0394],  # Annual prob of females seeking marriage if underpartnered
            [ 0,    0,  0.0394, 0.590, 0.760, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.760, 0.590, 0.344, 0.185, 0.0394], # Annual prob of males seeking marriage if underpartnered
        ], dtype=float),
        c=np.array([
            [ 0,    5,      10,     15,    20,    25,    30,    35,     40,     45,     50,    55,    60,    65,    70,    75],
            [ 0,    0,  0.590, 0.974, 0.998, 0.974, 0.870, 0.870, 0.870, 0.344, 0.0776, 0.0776, 0.0776, 0.0776, 0.0776, 0.0776],  # Annual prob of females seeking casual relationships if underpartnered
            [ 0,    0,  0.590, 0.870, 0.870, 0.870, 0.870, 0.974, 0.998, 0.974, 0.590, 0.344, 0.185, 0.0776, 0.0776, 0.0776],    # Annual prob of males seeking casual relationships if underpartnered
        ], dtype=float),
    )

    return dict(
        debut=debut,
        f_cross_layer=f_cross_layer,
        m_cross_layer=m_cross_layer,
        f_partners=f_partners,
        m_partners=m_partners,
        acts=acts,
        dur_pship=dur_pship,
        age_act_pars=age_act_pars,
        mixing=mixing,
        layer_probs=layer_probs,
    )


def load_country(location, year=None):
    """Return Starsim-shaped data for ``location``.

    Args:
        location (str): country name; must be one of ``_KNOWN_LOCATIONS``.
        year (int): year to load the initial age distribution for.

    Returns:
        dict with keys:
            - 'age_data': DataFrame [age, value]
            - 'birth_rate': DataFrame [Year, CBR]
            - 'death_rate': DataFrame [Time, AgeGrpStart, Sex, mx]
            - 'network_pars': dict ``{'layer_pars': {layer: {...}}, 'debut': {sex: Dist}}``
              consumed by ``hpv.SexualNetwork(**network_pars)``.
            - 'pop_total': DataFrame [year, pop_size] (total population trajectory)
            - 'pop_by_age': DataFrame [year, age, male, female] (age pyramid over time)
    """
    location = location.lower()
    if location not in _KNOWN_LOCATIONS:
        raise ValueError(
            f"Unknown location {location!r}. Supported locations: {_KNOWN_LOCATIONS}."
        )

    return dict(
        age_data=_age_data(location, year=year),
        birth_rate=_birth_rate(location),
        death_rate=_death_rate(location),
        network_pars=_network_pars(location),
        pop_total=_pop_total(location),
        pop_by_age=_pop_by_age(location),
    )


def _age_data(location, year=None):
    """Reshape the age distribution to a (age, value) long-form DataFrame.

    ``get_age_distribution`` returns an (N, 3) ndarray of
    ``(age_lower, age_upper, count)``. The data is at single-year resolution
    (age_upper == age_lower + 1), so age_lower IS the age and we don't need
    to expand bins.

    ``year`` is forwarded to ``_loaders.get_age_distribution``; if omitted,
    the loader defaults to year 2000 with a warning, producing a materially
    different age distribution than the sim-start year would.
    """
    arr = _loaders.get_age_distribution(location=location, year=year)
    return pd.DataFrame({
        'age': arr[:, 0].astype(int),
        'value': arr[:, 2].astype(float),
    })


def _birth_rate(location):
    """Birth rates as [Year, CBR] for ``ss.Births``. CBR is per 1000."""
    raw = _loaders.get_birth_rates(location=location)
    return pd.DataFrame({
        'Year': np.asarray(raw['year'], dtype=int),
        'CBR': np.asarray(raw['cbr'], dtype=float),
    })


def _death_rate(location):
    """Death rates as ``ss.Deaths``-shaped UN-style columns."""
    df = _loaders.map_entries(_loaders.load_file(_loaders.files.death), location)
    df = df[df['Sex'].isin(('Male', 'Female'))]  # drop 'Total'
    return pd.DataFrame({
        'Time': df['Time'].astype(int).values,
        'AgeGrpStart': df['AgeGrpStart'].astype(int).values,
        'Sex': df['Sex'].values,
        'mx': df['mx'].astype(float).values * 1000.0,  # per-1 to per-1000
    })


def _network_pars(location):
    """Build network parameters for ``hpv.SexualNetwork``.

    Returns ``{'layer_pars': {'m': {...}, 'c': {...}}, 'debut': {'f': ...,
    'm': ...}}``. Each layer dict carries: partners, mixing, layer_probs,
    cross_layer, duration, acts. ``debut`` is shared across layers (one
    per-agent sample).
    """
    default_pars = _default_network_pars(location)
    annual = ss.years(1)  # unit shared by all annual probability params

    # ``ss.prob`` lets the network call ``.to_prob(self.t.dt)`` for a
    # dt-correct per-step probability: 1 - exp(-rate*dt) == 1-(1-p)^dt.
    cross_layer_by_sex = {
        'm': ss.prob(default_pars['m_cross_layer'], annual),
        'f': ss.prob(default_pars['f_cross_layer'], annual),
    }

    debut_by_sex = {
        'm': _v2_dist_to_starsim(default_pars['debut']['m']),
        'f': _v2_dist_to_starsim(default_pars['debut']['f']),
    }

    layer_pars = {}
    for layer in ('m', 'c'):
        # layer_probs[layer] is a (3, N) ndarray: row 0 = age-bin lower
        # bounds, row 1 = annual female participation prob, row 2 = annual
        # male participation prob. Split into a per-sex dict so each row
        # carries its annual unit explicitly.
        lp = default_pars['layer_probs'][layer]
        layer_probs = dict(
            bins=lp[0, :],
            f=ss.prob(lp[1, :], annual),
            m=ss.prob(lp[2, :], annual),
        )
        layer_pars[layer] = dict(
            partners={
                'm': _v2_dist_to_starsim(default_pars['m_partners'][layer]),
                'f': _v2_dist_to_starsim(default_pars['f_partners'][layer]),
            },
            mixing=default_pars['mixing'][layer],
            layer_probs=layer_probs,
            cross_layer=cross_layer_by_sex,
            duration=_v2_dist_to_starsim(default_pars['dur_pship'][layer]),
            acts=_v2_dist_to_starsim(default_pars['acts'][layer]),
            age_act_pars=default_pars['age_act_pars'][layer],
        )
    return dict(layer_pars=layer_pars, debut=debut_by_sex)


def _pop_total(location):
    """Total population trajectory: DataFrame [year, pop_size].

    Wraps ``get_total_pop`` which already returns a DataFrame with the canonical
    column names (year, pop_size) scaled to real-world counts.
    """
    df = _loaders.get_total_pop(location=location)
    return pd.DataFrame({
        'year': np.asarray(df['year'], dtype=int),
        'pop_size': np.asarray(df['pop_size'], dtype=float),
    })


def _pop_by_age(location):
    """Age pyramid over time: DataFrame [year, age, male, female].

    Wraps ``get_age_distribution_over_time`` which already renames columns to
    year/age/male/female and scales counts to real-world units (× 1000).
    """
    df = _loaders.get_age_distribution_over_time(location=location)
    return df[['year', 'age', 'male', 'female']].copy()