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
- Distributions: ``ss.Dist`` instances (see ``hpv.NetworkPars`` for the
  default set).
"""

import numpy as np
import pandas as pd
import sciris as sc

from ..parameters import NetworkPars
from . import loaders as _loaders


def _default_network_pars(location=None, pars=None, **kwargs):  # noqa: ARG001  (location accepted for API symmetry)
    """Return the flat ``hpv.NetworkPars`` defaults, merged with ``pars`` /
    ``**kwargs`` overrides. Kept as a thin wrapper so callers using the pre-3.0.1
    factory continue to work; new code should call ``hpv.NetworkPars`` directly.

    Network calibration is intentionally location-agnostic (HPVsim ships
    demographic data per country but not network calibration); ``location``
    is accepted for API symmetry.
    """
    return dict(NetworkPars(**sc.mergedicts(pars, kwargs)))


def load_country(location, year=None, datafolder=None):
    """Return Starsim-shaped data for ``location``.

    Any country in the bundled UN WPP 2024 data is valid; see
    ``hpv.data.get_country_aliases()`` for accepted name variants. The
    underlying loaders raise ``ValueError`` with suggestion-based
    diagnostics for unknown names.

    Network calibration is location-agnostic (see
    ``_default_network_pars``). Analysis scripts supply per-country
    network pars as needed.

    Args:
        location (str): country name (case-insensitive).
        year (int): year to load the initial age distribution for.
        datafolder (str/Path): if provided, look for user CSVs named
            ``age_data.csv``, ``birth_rate.csv``, ``death_rate.csv``,
            ``pop_total.csv`` in this folder. Missing files fall back to
            the bundled UN WPP data for ``location`` with a warning; the
            demographic indicator is preserved for any file that IS
            present so users can mix bundled and custom data.

    Returns:
        dict with the same keys as before (age_data, birth_rate, death_rate,
        network_pars, pop_total, pop_by_age). Any indicator whose file is
        missing AND whose bundled fallback also fails is returned as None,
        and the caller (typically ``hpv.Sim``) is responsible for skipping
        the corresponding module.
    """
    location = location.lower()

    return dict(
        age_data=_age_data(location, year=year, datafolder=datafolder),
        birth_rate=_birth_rate(location, datafolder=datafolder),
        death_rate=_death_rate(location, datafolder=datafolder),
        network_pars=_network_pars(location),
        pop_total=_pop_total(location, datafolder=datafolder),
        pop_by_age=_pop_by_age(location),
    )


def _datafile(datafolder, name):
    """Path to a datafolder CSV if it exists; None otherwise."""
    if datafolder is None:
        return None
    path = sc.path(datafolder) / name
    return path if path.exists() else None


def _warn_missing_indicator(indicator, filename, datafolder):
    """Warn that a datafolder indicator is missing (bundle fallback used)."""
    import warnings
    warnings.warn(
        f'hpv.load_country: {indicator} file {filename!r} not found in '
        f'datafolder {sc.path(datafolder)!s}; falling back to bundled UN '
        f'WPP data for the location.',
        stacklevel=3,
    )


def _age_data(location, year=None, datafolder=None):
    """Reshape the age distribution to a (age, value) long-form DataFrame."""
    csv = _datafile(datafolder, 'age_data.csv')
    if datafolder is not None and csv is None:
        _warn_missing_indicator('age_data', 'age_data.csv', datafolder)
    arr = _loaders.get_age_distribution(location=location, year=year,
                                        age_datafile=str(csv) if csv else None)
    return pd.DataFrame({
        'age': arr[:, 0].astype(int),
        'value': arr[:, 2].astype(float),
    })


def _birth_rate(location, datafolder=None):
    """Birth rates as [Year, CBR] for ``ss.Births``. CBR is per 1000."""
    csv = _datafile(datafolder, 'birth_rate.csv')
    if datafolder is not None and csv is None:
        _warn_missing_indicator('birth_rate', 'birth_rate.csv', datafolder)
    raw = _loaders.get_birth_rates(location=location,
                                   birth_datafile=str(csv) if csv else None)
    return pd.DataFrame({
        'Year': np.asarray(raw['year'], dtype=int),
        'CBR': np.asarray(raw['cbr'], dtype=float),
    })


def _death_rate(location, datafolder=None):
    """Death rates as ``ss.Deaths``-shaped UN-style columns."""
    csv = _datafile(datafolder, 'death_rate.csv')
    if csv is not None:
        df = pd.read_csv(csv)
    else:
        if datafolder is not None:
            _warn_missing_indicator('death_rate', 'death_rate.csv', datafolder)
        df = _loaders.map_entries(_loaders.load_file(_loaders.files.death), location)
    df = df[df['Sex'].isin(('Male', 'Female'))]  # drop 'Total'
    return pd.DataFrame({
        'Time': df['Time'].astype(int).values,
        'AgeGrpStart': df['AgeGrpStart'].astype(int).values,
        'Sex': df['Sex'].values,
        'mx': df['mx'].astype(float).values * 1000.0,  # per-1 to per-1000
    })


def _network_pars(location, pars=None, **kwargs):
    """Return the flat ``NetworkPars`` dict for ``location`` with overrides.
    Location is currently unused (see ``_default_network_pars``)."""
    return _default_network_pars(location, pars=pars, **kwargs)


def _pop_total(location, datafolder=None):
    """Total population trajectory: DataFrame [year, pop_size]."""
    csv = _datafile(datafolder, 'pop_total.csv')
    if datafolder is not None and csv is None:
        _warn_missing_indicator('pop_total', 'pop_total.csv', datafolder)
    df = _loaders.get_total_pop(location=location,
                                pop_datafile=str(csv) if csv else None)
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