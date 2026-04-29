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
  (Location, year, cbr) — i.e. a crude birth rate per year, not an
  age-specific fertility rate (ASFR). To produce ASFR for ss.Pregnancy we
  spread the CBR over reproductive ages (15-49) uniformly. Future work can
  refine this when age-specific data is wired in.
- Death rates: v2's get_death_rates(by_sex=True) returns
  {year: {sex: ndarray(M, 2) with columns (age, rate)}}. We flatten into
  long form (Year, AgeGrp, Sex, Rate).
- Network parameters: v2's "default" network has 2 layers (m, c). M01 ships
  the same two layers — earlier plan drafts assumed a third 'o' (one-off)
  layer based on a misleading comment in v2 parameters.py, but no such
  layer exists in v2 code. partners and cross_layer are split by sex in v2
  (m_partners/f_partners, m_cross_layer/f_cross_layer); we expose both
  sexes per-layer.
- Distributions: v2 stores distributions as plain dicts like
  ``{'dist': 'poisson1', 'par1': 0.5}`` consumed by hpvsim.utils.sample().
  v3 prefers Starsim Dist instances (sampled via .rvs(uids)). The
  ``_v2_dist_to_starsim`` helper below converts between the two for
  partners, duration, and acts.
"""

import numpy as np
import pandas as pd
import starsim as ss

from .. import parameters as _params
from . import loaders as _loaders


class _PoissonShifted(ss.poisson):
    """v2-style 'poisson1' distribution: scipy.stats.poisson with loc=shift.

    v2's ``poisson1`` is "Poisson(rate) + 1" — i.e. every agent has at least
    one partner. scipy.stats.poisson supports a `loc` parameter that shifts
    the support, which we wire through Starsim's sync_pars hook.
    """
    def __init__(self, lam=1.0, shift=1, **kwargs):
        self._shift = shift
        super().__init__(lam=lam, **kwargs)

    def sync_pars(self):
        spars = dict(mu=self._pars.lam, loc=self._shift)
        self.update_dist_pars(spars)
        return spars


def _v2_dist_to_starsim(d):
    """Convert a v2-format distribution dict to a Starsim Dist instance.

    v2 stores distributions as ``{'dist': name, 'par1': p1, 'par2': p2}``;
    Starsim distributions take named parameters and are sampled via
    ``.rvs(uids)``. M01 only needs poisson, poisson1, lognormal, neg_binomial
    (the four used in v2's default Nigeria network pars).
    """
    dist = d['dist']
    par1 = d.get('par1')
    par2 = d.get('par2')
    if dist == 'poisson':
        return ss.poisson(lam=par1)
    if dist == 'poisson1':
        return _PoissonShifted(lam=par1, shift=1)
    if dist in ('lognormal', 'lognorm'):
        # v2 'lognormal' parameters are mean and std of the LOGNORMAL itself;
        # ss.lognorm_ex takes the same parameterization.
        return ss.lognorm_ex(mean=par1, std=par2)
    if dist == 'neg_binomial':
        # v2: par1=mean, par2=k (dispersion).
        # scipy.stats.nbinom: n=number-of-successes, p=success-probability.
        # Mapping: n = k, p = k / (k + mean).
        n_param = par2
        p_param = par2 / (par2 + par1) if (par2 + par1) > 0 else 0.5
        return ss.nbinom(n=n_param, p=p_param)
    if dist == 'normal':
        return ss.normal(loc=par1, scale=par2)
    if dist == 'uniform':
        return ss.uniform(low=par1, high=par2)
    raise ValueError(f'Unsupported v2 distribution: {dist!r}')


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
        fertility=_fertility(location),
        death_rate=_death_rate(location),
        network_pars=_network_pars(location),
    )


def _age_data(location):
    """Reshape v2's age distribution to a (age, value) long-form DataFrame.

    v2 returns an ndarray of shape (N, 3) with columns
    (age_lower, age_upper, count). We expand each [lower, upper) bin into
    one row per integer age, with the bin count distributed evenly.
    """
    arr = _loaders.get_age_distribution(location=location)
    rows = []
    for row in arr:
        age_lower, age_upper, count = float(row[0]), float(row[1]), float(row[2])
        n_ages_in_bin = max(1, int(round(age_upper - age_lower)))
        per_age = count / n_ages_in_bin
        for a in range(int(age_lower), int(age_lower) + n_ages_in_bin):
            rows.append({'age': a, 'value': per_age})
    return pd.DataFrame(rows)


def _fertility(location):
    """Reshape v2 birth rates into [Time, AgeGrp, ASFR] long form.

    v2 returns a sciris dataframe with (Location, year, cbr) per year — a
    crude birth rate (per 1000 population), not age-specific. To match the
    shape ss.Pregnancy(fertility_rate=...) consumes, we spread the CBR over
    reproductive ages (15-49) uniformly and convert from per-1000 to per-1
    by dividing by 1000. Future work can substitute true ASFR data here.
    """
    raw = _loaders.get_birth_rates(location=location)
    # raw is a sciris dataframe; iterate rows.
    repro_lower, repro_upper = 15, 50  # 15-49 inclusive
    repro_ages = list(range(repro_lower, repro_upper))
    rows = []
    for i in range(len(raw)):
        year = int(raw['year'][i])
        cbr = float(raw['cbr'][i])  # per 1000 population
        # Spread evenly across reproductive ages and normalise to per-1.
        asfr = (cbr / 1000.0) / len(repro_ages)
        for age in repro_ages:
            rows.append({'Time': year, 'AgeGrp': age, 'ASFR': asfr})
    return pd.DataFrame(rows)


def _death_rate(location):
    """Reshape v2 death rates into [Year, AgeGrp, Sex, Rate] long form.

    v2 returns ``{year: {sex: ndarray(M, 2) with columns (age, rate)}}``.
    """
    raw = _loaders.get_death_rates(location=location, by_sex=True)
    rows = []
    for year, sex_to_arr in raw.items():
        for sex, arr in sex_to_arr.items():
            arr = np.asarray(arr)
            for age, rate in arr:
                rows.append({
                    'Year': int(year),
                    'AgeGrp': int(age),
                    'Sex': sex,
                    'Rate': float(rate),
                })
    return pd.DataFrame(rows)


def _network_pars(location):
    """Build per-layer network parameter dicts.

    v2's "default" network defines 2 layers (m=marital, c=casual). M01
    matches that — the earlier plan's reference to a third 'o' layer was
    based on a misleading comment in v2 parameters.py; no such layer is
    defined anywhere in v2 code.

    Returns ``{'m': {...}, 'c': {...}}`` where each layer dict has:
    partners, mixing, layer_probs, cross_layer, duration, acts.
    """
    default_pars = _params.make_pars(location=location)

    # Per-sex cross-layer concurrency is a scalar in v2; expose as {sex: val}.
    cross_layer_by_sex = {
        'm': default_pars['m_cross_layer'],
        'f': default_pars['f_cross_layer'],
    }

    out = {}
    for layer in ('m', 'c'):
        out[layer] = dict(
            partners={
                'm': _v2_dist_to_starsim(default_pars['m_partners'][layer]),
                'f': _v2_dist_to_starsim(default_pars['f_partners'][layer]),
            },
            mixing=default_pars['mixing'][layer],
            layer_probs=default_pars['layer_probs'][layer],
            cross_layer=cross_layer_by_sex,
            duration=_v2_dist_to_starsim(default_pars['dur_pship'][layer]),
            acts=_v2_dist_to_starsim(default_pars['acts'][layer]),
        )
    return out