"""Country-data adapter: wrap v2 hpvsim's location data into Starsim-shaped DataFrames.

Used by hpvsim.Sim to build People (age pyramid), Pregnancy (fertility),
Deaths (mortality), and per-layer SexualNetwork instances. All underlying data
lives in hpvsim/data/files/ and is loaded via the existing hpvsim.data.loaders
module and hpvsim.parameters helpers (which stayed active through the
v2 -> v3 migration).

Notes on v2 -> v3 reshaping:

- Age pyramid: v2's get_age_distribution returns an ndarray of shape (N, 3)
  with columns (age_lower, age_upper, count). We expand bins into a per-year
  long form with columns (age, value).
- Birth rates: v2's get_birth_rates returns a sciris dataframe with columns
  (Location, year, cbr) — a crude birth rate per year. v3 uses ss.Births
  (population-level CBR, matching v2's mechanism) rather than ss.Pregnancy
  (per-woman ASFR), so we just rename columns to [Year, CBR] for the
  ss.Births interface. M02+ may switch to ss.Pregnancy if/when proper ASFR
  data becomes available.
- Death rates: v2's get_death_rates(by_sex=True) returns
  {year: {sex: ndarray(M, 2) with columns (age, rate)}}. We flatten into
  long form (Year, AgeGrp, Sex, Rate).
- Network parameters: v2's "default" network has 2 layers (m, c).
  partners and cross_layer are split by sex in v2 (m_partners/f_partners,
  m_cross_layer/f_cross_layer); we expose both sexes per-layer.
- Distributions: v2 stores distributions as plain dicts like
  ``{'dist': 'poisson1', 'par1': 0.5}`` consumed by hpvsim.utils.sample().
  v3 prefers Starsim Dist instances (sampled via .rvs(uids)). The
  ``_v2_dist_to_starsim`` helper in ``hpvsim.migration_utils`` converts
  between the two for partners, duration, and acts.
"""

import numpy as np
import pandas as pd
import starsim as ss

from .. import parameters as _params
from ..migration_utils import _v2_dist_to_starsim
from . import loaders as _loaders


_KNOWN_LOCATIONS = ['nigeria']  # M01 ships with Nigeria only; expand as needed.


def load_country(location):
    """Return Starsim-shaped data for ``location``.

    Args:
        location (str): country name; must be one of the supported locations
            available via the v2 loaders.

    Returns:
        dict with keys:
            - 'age_data': DataFrame [age, value]
            - 'fertility': DataFrame [Time, AgeGrp, ASFR]
            - 'death_rate': DataFrame [Year, AgeGrp, Sex, Rate]
            - 'network_pars': nested dict {layer: {key: value}}
              with keys: partners, mixing, layer_probs, cross_layer, duration, acts.
    """
    location = location.lower()
    if location not in _KNOWN_LOCATIONS:
        raise ValueError(
            f"Unknown location {location!r}. Supported locations: {_KNOWN_LOCATIONS}."
        )

    return dict(
        age_data=_age_data(location),
        birth_rate=_birth_rate(location),
        death_rate=_death_rate(location),
        network_pars=_network_pars(location),
    )


def _age_data(location):
    """Reshape v2's age distribution to a (age, value) long-form DataFrame.

    ``get_age_distribution`` returns an (N, 3) ndarray of
    ``(age_lower, age_upper, count)``. v2 data is at single-year resolution
    (age_upper == age_lower + 1), so age_lower IS the age and we don't need
    to expand bins.
    """
    arr = _loaders.get_age_distribution(location=location)
    return pd.DataFrame({
        'age': arr[:, 0].astype(int),
        'value': arr[:, 2].astype(float),
    })


def _birth_rate(location):
    """Reshape v2 birth rates into [Year, CBR] long form for ss.Births.

    v2's get_birth_rates returns a sciris dataframe with (Location, year, cbr).
    ss.Births accepts a DataFrame with columns [Year, CBR]; CBR is per 1000
    population per year (the v2 unit), and ss.Births handles the conversion.
    """
    raw = _loaders.get_birth_rates(location=location)
    return pd.DataFrame({
        'Year': np.asarray(raw['year'], dtype=int),
        'CBR': np.asarray(raw['cbr'], dtype=float),
    })


def _death_rate(location):
    """Read v2 death rates as ss.Deaths' default UN-style columns."""
    df = _loaders.map_entries(_loaders.load_file(_loaders.files.death), location)
    df = df[df['Sex'].isin(('Male', 'Female'))]  # drop 'Total'
    return pd.DataFrame({
        'Time': df['Time'].astype(int).values,
        'AgeGrpStart': df['AgeGrpStart'].astype(int).values,
        'Sex': df['Sex'].values,
        'mx': df['mx'].astype(float).values * 1000.0,  # per-1 to per-1000
    })


def _network_pars(location):
    """Build per-layer network parameter dicts.

    v2's "default" network defines 2 layers (m=marital, c=casual).

    Returns ``{'m': {...}, 'c': {...}}`` where each layer dict has:
    partners, mixing, layer_probs, cross_layer, duration, acts, debut.
    """
    default_pars = _params.make_pars(location=location)
    annual = ss.years(1)  # unit shared by all annual probability params

    # Per-sex cross-layer concurrency is an annual probability in v2.
    # ``ss.prob`` lets the network call ``.to_prob(self.t.dt)`` to get a
    # dt-correct per-step probability: 1 - exp(-rate*dt) == 1-(1-p)^dt.
    cross_layer_by_sex = {
        'm': ss.prob(default_pars['m_cross_layer'], annual),
        'f': ss.prob(default_pars['f_cross_layer'], annual),
    }

    debut_by_sex = {
        'm': _v2_dist_to_starsim(default_pars['debut']['m']),
        'f': _v2_dist_to_starsim(default_pars['debut']['f']),
    }

    out = {}
    for layer in ('m', 'c'):
        # v2's layer_probs[layer] is a (3, N) ndarray: row 0 = age-bin lower
        # bounds, row 1 = annual female participation prob, row 2 = annual
        # male participation prob. Split into a dict so the per-sex rows
        # carry their unit (annual) explicitly.
        lp = default_pars['layer_probs'][layer]
        layer_probs = dict(
            bins=np.asarray(lp[0, :]),
            f=ss.prob(np.asarray(lp[1, :]), annual),
            m=ss.prob(np.asarray(lp[2, :]), annual),
        )
        out[layer] = dict(
            partners={
                'm': _v2_dist_to_starsim(default_pars['m_partners'][layer]),
                'f': _v2_dist_to_starsim(default_pars['f_partners'][layer]),
            },
            mixing=default_pars['mixing'][layer],
            layer_probs=layer_probs,
            cross_layer=cross_layer_by_sex,
            duration=_v2_dist_to_starsim(default_pars['dur_pship'][layer]),
            acts=_v2_dist_to_starsim(default_pars['acts'][layer]),
            debut=debut_by_sex,
        )
    return out