"""HIV-data adapter: load country HIV/ART inputs for the co-infection sim.

Mirrors ``hpvsim.data.country.load_country`` in style. Returns the Rwanda HIV
inputs a later task (T10b/T12) consumes to build a co-infection sim:

- ``art_coverage``: tidy long DataFrame ``[age, sex, year, coverage]`` parsed
  from the bundled by-age ART coverage CSVs (females + males).
- ``init_prev``: a scalar initial HIV prevalence to seed ``sti.HIV`` /
  ``hpv.HIV``. See ``_init_prev`` for the derivation and its caveats.

Underlying data lives in ``hpvsim/data/hiv/`` (copied from the Rwanda
validation repo so nothing is imported from a sibling repo at runtime). The
ART-coverage-application intervention (the "shortcut") is a SEPARATE task
(T10b); this module only loads the coverage data, it does not apply it.
"""

import numpy as np
import pandas as pd
from pathlib import Path

_HIV_DATADIR = Path(__file__).parent / 'hiv'

# v3 sim start year; used to pick the init-prev seed from the prevalence series.
_START_YEAR = 1990

_KNOWN_LOCATIONS = ['rwanda']


def load_hiv(location):
    """Return Rwanda HIV/ART inputs for ``location``.

    Args:
        location (str): country name; must be one of ``_KNOWN_LOCATIONS``.
            Only ``'rwanda'`` is supported now.

    Returns:
        dict with keys:
            - 'art_coverage': DataFrame ``[age, sex, year, coverage]`` (long).
              ART coverage fraction by single-year age, sex ('f'/'m'), and
              calendar year (2004-2030).
            - 'init_prev': float, an initial HIV prevalence to seed the HIV
              module at the v3 start year (see ``_init_prev``).
    """
    location = location.lower()
    if location not in _KNOWN_LOCATIONS:
        raise ValueError(
            f"Unknown location {location!r}. Supported locations: {_KNOWN_LOCATIONS}."
        )
    return dict(
        art_coverage=_art_coverage(location),
        init_prev=_init_prev(location),
    )


def _art_coverage(location):  # noqa: ARG001  (single-location for now)
    """ART coverage by age/sex/year as a tidy long DataFrame.

    Reads the two wide by-age CSVs (``age, 2004, ..., 2030``), one per sex,
    and melts them into ``[age, sex, year, coverage]``.
    """
    frames = []
    for sex, fname in (('f', 'rwanda_art_coverage_by_age_females.csv'),
                       ('m', 'rwanda_art_coverage_by_age_males.csv')):
        wide = pd.read_csv(_HIV_DATADIR / fname)
        long = wide.melt(id_vars='age', var_name='year', value_name='coverage')
        long['sex'] = sex
        long['year'] = long['year'].astype(int)
        long['age'] = long['age'].astype(int)
        long['coverage'] = long['coverage'].astype(float)
        frames.append(long[['age', 'sex', 'year', 'coverage']])
    return pd.concat(frames, ignore_index=True)


def _init_prev(location):  # noqa: ARG001  (single-location for now)
    """Initial HIV prevalence seed at the v3 start year (1990).

    DERIVATION / CAVEAT: There is no HIV-prevalence-by-age data source for
    Rwanda, and the cached v2.3.0 baseline does NOT expose a
    ``hiv_prevalence_by_age`` result (its timeseries only carries
    cancer-incidence-by-HIV-status metrics). So we fall back to the documented
    interim seed: the aggregate national ``hiv_prevalence`` at the start year
    (1990) from ``rwanda_hiv_prevalence.csv``, applied as a flat adult
    prevalence. ``sti.HIV.init_prev_data`` accepts this scalar and parses it
    into an ``ss.bernoulli`` over the population. This is an INTERIM seed to be
    refined during calibration (T12); the by-age/by-sex shape is deferred.
    """
    df = pd.read_csv(_HIV_DATADIR / 'rwanda_hiv_prevalence.csv')
    row = df.loc[df['year'] == _START_YEAR]
    if row.empty:
        raise ValueError(
            f"rwanda_hiv_prevalence.csv has no row for start year {_START_YEAR}."
        )
    return float(np.asarray(row['total'])[0])
