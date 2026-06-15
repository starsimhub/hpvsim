"""HIV-data adapter: load per-location HIV/ART inputs for the co-infection sim.

Mirrors ``hpvsim.data.country.load_country`` in style. Returns the HIV inputs
used to build a co-infection sim:

- ``art_coverage``: tidy long DataFrame ``[age, sex, year, coverage]`` parsed
  from the by-age ART coverage CSVs (females + males).
- ``init_prev``: a scalar initial HIV prevalence to seed ``sti.HIV`` /
  ``hpv.HIV``. See ``_init_prev`` for the derivation and its caveats.
- ``incidence``: tidy long DataFrame ``[age, sex, year, incidence]`` — the
  per-year HIV acquisition rate among susceptibles by single-year age, sex
  ('f'/'m'), and calendar year. Consumed by the incidence-driven HIV importer
  (``hpv.hiv_incidence_import``).

Data layout: ``hpvsim/data/hiv/<location>/`` holds four location-agnostic
filenames — ``hiv_incidence.csv``, ``art_coverage_by_age_females.csv``,
``art_coverage_by_age_males.csv``, ``hiv_prevalence.csv``. Adding a country is
therefore just dropping in a new ``<location>/`` folder with those four files
and listing it in ``_KNOWN_LOCATIONS``. Data is bundled in-tree so nothing is
imported from a sibling repo at runtime. (Currently only ``'rwanda'`` ships.)
"""

import numpy as np
import pandas as pd
from pathlib import Path

_HIV_DATADIR = Path(__file__).parent / 'hiv'

# v3 sim start year; used to pick the init-prev seed from the prevalence series.
_START_YEAR = 1990

# Locations with a bundled hpvsim/data/hiv/<location>/ data folder.
_KNOWN_LOCATIONS = ['rwanda']


def load_hiv(location):
    """Return the bundled HIV/ART inputs for ``location``.

    Args:
        location (str): country name; must be one of ``_KNOWN_LOCATIONS``
            (i.e. have a ``hpvsim/data/hiv/<location>/`` data folder).

    Returns:
        dict with keys:
            - 'art_coverage': DataFrame ``[age, sex, year, coverage]`` (long).
              ART coverage fraction by single-year age, sex ('f'/'m'), and
              calendar year.
            - 'init_prev': float, an initial HIV prevalence to seed the HIV
              module at the v3 start year (see ``_init_prev``).
            - 'incidence': DataFrame ``[age, sex, year, incidence]`` (long).
              Per-year HIV acquisition rate among susceptibles by single-year
              age, sex ('f'/'m'), and calendar year.
    """
    location = location.lower()
    if location not in _KNOWN_LOCATIONS:
        raise ValueError(
            f"Unknown location {location!r}. Supported locations: {_KNOWN_LOCATIONS}."
        )
    return dict(
        art_coverage=_art_coverage(location),
        init_prev=_init_prev(location),
        incidence=_incidence(location),
    )


def _incidence(location):
    """HIV incidence by age/sex/year as a tidy long DataFrame.

    Reads ``<location>/hiv_incidence.csv`` (long ``Age, Year, Sex, Incidence``;
    Incidence = per-year HIV acquisition rate among susceptibles) and
    normalizes column names/dtypes to ``[age, sex, year, incidence]`` (matching
    the ``art_coverage`` style).
    """
    df = pd.read_csv(_HIV_DATADIR / location / 'hiv_incidence.csv')
    df = df.rename(columns={
        'Age': 'age', 'Year': 'year', 'Sex': 'sex', 'Incidence': 'incidence',
    })
    df['age'] = df['age'].astype(int)
    df['year'] = df['year'].astype(int)
    df['sex'] = df['sex'].astype(str).str.lower().str[0]
    df['incidence'] = df['incidence'].astype(float)
    return df[['age', 'sex', 'year', 'incidence']].reset_index(drop=True)


def _art_coverage(location):
    """ART coverage by age/sex/year as a tidy long DataFrame.

    Reads the two wide by-age CSVs (``age, <year>, ...``), one per sex
    (``art_coverage_by_age_females.csv`` / ``_males.csv``), and melts them
    into ``[age, sex, year, coverage]``.
    """
    frames = []
    for sex, fname in (('f', 'art_coverage_by_age_females.csv'),
                       ('m', 'art_coverage_by_age_males.csv')):
        wide = pd.read_csv(_HIV_DATADIR / location / fname)
        long = wide.melt(id_vars='age', var_name='year', value_name='coverage')
        long['sex'] = sex
        long['year'] = long['year'].astype(int)
        long['age'] = long['age'].astype(int)
        long['coverage'] = long['coverage'].astype(float)
        frames.append(long[['age', 'sex', 'year', 'coverage']])
    return pd.concat(frames, ignore_index=True)


def _init_prev(location):
    """Initial HIV prevalence seed at the v3 start year (1990).

    DERIVATION / CAVEAT: where there is no HIV-prevalence-by-age data source, we
    fall back to a documented interim seed: the aggregate national
    ``hiv_prevalence`` at the start year from ``<location>/hiv_prevalence.csv``,
    applied as a flat adult prevalence. ``sti.HIV.init_prev_data`` accepts this
    scalar and parses it into an ``ss.bernoulli`` over the population. This is an
    INTERIM seed; the by-age/by-sex shape is deferred to calibration.
    """
    fpath = _HIV_DATADIR / location / 'hiv_prevalence.csv'
    df = pd.read_csv(fpath)
    row = df.loc[df['year'] == _START_YEAR]
    if row.empty:
        raise ValueError(
            f"{fpath.name} for {location!r} has no row for start year {_START_YEAR}."
        )
    return float(np.asarray(row['total'])[0])
