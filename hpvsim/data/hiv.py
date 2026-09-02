"""HIV-data loader: reads a user-supplied HIV/ART data folder for the
co-infection sim.

Mirrors ``hpvsim.data.country.load_country``'s ``datafolder=`` style, but with
no bundled fallback -- hpvsim ships no country's HIV data; callers supply
their own. Returns the HIV inputs used to build a co-infection sim:

- ``art_coverage``: tidy long DataFrame ``[age, sex, year, coverage]`` parsed
  from the by-age ART coverage CSVs (females + males).
- ``init_prev``: a scalar initial HIV prevalence to seed ``sti.HIV`` /
  ``hpv.HIV_transmit`` / ``hpv.HIV_incidence``. See ``_init_prev`` for the
  derivation and its caveats.
- ``incidence``: tidy long DataFrame ``[age, sex, year, incidence]`` — the
  per-year HIV acquisition rate among susceptibles by single-year age, sex
  ('f'/'m'), and calendar year. Consumed by ``hpv.HIV_incidence``.

Data layout: a folder holding four fixed filenames — ``hiv_incidence.csv``,
``art_coverage_by_age_females.csv``, ``art_coverage_by_age_males.csv``,
``hiv_prevalence.csv``.
"""

import numpy as np
import pandas as pd
import sciris as sc

# v3 sim start year; used to pick the init-prev seed from the prevalence series.
_START_YEAR = 1990

_REQUIRED_FILES = ('hiv_incidence.csv', 'art_coverage_by_age_females.csv',
                   'art_coverage_by_age_males.csv', 'hiv_prevalence.csv')


def load_hiv_data(datafolder):
    """Load a user-supplied HIV/ART data folder.

    Args:
        datafolder (str/Path): folder containing all four fixed filenames
            listed in ``_REQUIRED_FILES``.

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
    datafolder = sc.path(datafolder)
    missing = [f for f in _REQUIRED_FILES if not (datafolder / f).exists()]
    if missing:
        raise ValueError(f'hpv.data.load_hiv_data: missing {missing} in {datafolder}')
    return dict(
        art_coverage=_art_coverage(datafolder),
        init_prev=_init_prev(datafolder),
        incidence=_incidence(datafolder),
    )


def _incidence(datafolder):
    """HIV incidence by age/sex/year as a tidy long DataFrame.

    Reads ``hiv_incidence.csv`` (long ``Age, Year, Sex, Incidence``;
    Incidence = per-year HIV acquisition rate among susceptibles) and
    normalizes column names/dtypes to ``[age, sex, year, incidence]`` (matching
    the ``art_coverage`` style).
    """
    df = pd.read_csv(datafolder / 'hiv_incidence.csv')
    df = df.rename(columns={
        'Age': 'age', 'Year': 'year', 'Sex': 'sex', 'Incidence': 'incidence',
    })
    df['age'] = df['age'].astype(int)
    df['year'] = df['year'].astype(int)
    df['sex'] = df['sex'].astype(str).str.lower().str[0]
    df['incidence'] = df['incidence'].astype(float)
    return df[['age', 'sex', 'year', 'incidence']].reset_index(drop=True)


def _art_coverage(datafolder):
    """ART coverage by age/sex/year as a tidy long DataFrame.

    Reads the two wide by-age CSVs (``age, <year>, ...``), one per sex
    (``art_coverage_by_age_females.csv`` / ``_males.csv``), and melts them
    into ``[age, sex, year, coverage]``.
    """
    frames = []
    for sex, fname in (('f', 'art_coverage_by_age_females.csv'),
                       ('m', 'art_coverage_by_age_males.csv')):
        wide = pd.read_csv(datafolder / fname)
        long = wide.melt(id_vars='age', var_name='year', value_name='coverage')
        long['sex'] = sex
        long['year'] = long['year'].astype(int)
        long['age'] = long['age'].astype(int)
        long['coverage'] = long['coverage'].astype(float)
        frames.append(long[['age', 'sex', 'year', 'coverage']])
    return pd.concat(frames, ignore_index=True)


def _init_prev(datafolder):
    """Initial HIV prevalence seed at the v3 start year (1990).

    DERIVATION / CAVEAT: where there is no HIV-prevalence-by-age data source, we
    fall back to a documented interim seed: the aggregate national
    ``hiv_prevalence`` at the start year from ``hiv_prevalence.csv``,
    applied as a flat adult prevalence. ``sti.HIV.init_prev_data`` accepts this
    scalar and parses it into an ``ss.bernoulli`` over the population. This is an
    INTERIM seed; the by-age/by-sex shape is deferred to calibration.
    """
    fpath = datafolder / 'hiv_prevalence.csv'
    df = pd.read_csv(fpath)
    row = df.loc[df['year'] == _START_YEAR]
    if row.empty:
        raise ValueError(
            f"{fpath.name} has no row for start year {_START_YEAR}."
        )
    return float(np.asarray(row['total'])[0])


def reshape_art_coverage(df):
    """Reshape a tidy ART-coverage frame into STIsim's stratified format.

    ``load_hiv_data(...)['art_coverage']`` is a long frame with columns
    ``[age, sex, year, coverage]`` (``age`` an age-band lower bound; sex 'f'/'m';
    coverage a fraction of HIV+ in that stratum). ``sti.ART``'s
    stratified-coverage parser expects columns ``Year``, ``Gender``, ``AgeBin``
    (a ``'[lo,hi)'`` string) and a numeric proportion column whose name does NOT
    start with ``n_`` (so it is read as a proportion 'p', not absolute counts).

    Each band's upper edge is taken from the next age present in the frame, so
    any band width works: 5-year bands (0, 5, 10, ... -- the usual
    UNAIDS/Spectrum shape) as well as single years of age. Deriving the edge
    rather than always using ``age + 1`` matters, because unit bins built from
    5-year data would leave ages 1-4, 6-9, ... with no ART coverage at all --
    silently treating a fraction of those who should be on ART. The top band is
    left open-ended (upper edge 150), matching the convention elsewhere in
    hpvsim.

    Coverage is clamped to ``[0, 1]``: a user-supplied curve may carry a few values
    slightly above 1.0 (rounding), and STIsim infers proportion-vs-count from whether the
    max value is <= 1.0 — an un-clamped 1.0001 would flip the whole frame to
    'absolute counts' and treat ~nobody. Clamping keeps it a proportion.
    """
    out = df.rename(columns={'year': 'Year', 'sex': 'Gender', 'coverage': 'p_art'}).copy()
    ages = np.sort(out['age'].unique())
    upper = {int(a): int(ages[i + 1]) if i + 1 < len(ages) else 150
             for i, a in enumerate(ages)}
    out['AgeBin'] = out['age'].map(lambda a: f'[{int(a)},{upper[int(a)]})')
    out['p_art'] = out['p_art'].clip(lower=0.0, upper=1.0)
    return out[['Year', 'Gender', 'AgeBin', 'p_art']]
