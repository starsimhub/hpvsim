"""M08 (T11): v2 Rwanda HIV-stratified cancer-incidence parity TARGET.

Goal: pin the v2.3 Rwanda HIV-stratified baseline that the M08 parity gate
(T13) compares the v3 co-infection model against. This module *extracts* and
*loads* the target; it does NOT calibrate v3 or run the comparison.

WHY a tolerance-band target (not an M03/M05-style multi-seed z-score gate)
-------------------------------------------------------------------------
The M03/M05 gates compare a v3 seed-sweep distribution against a v2 seed-sweep
distribution via ``parity.parity_gate`` (z = Δmean / combined-SE). That needs
the *raw per-seed vectors* on both sides.

The cached v2 Rwanda baseline does NOT store raw per-run vectors. The Rwanda
validation repo's ``run_calibration.py:save_figS2_csvs`` reduces the
``hpv.Calibration`` posterior draws (``analyzer_results`` / ``extra_sim_results``)
to quantile *summaries* before writing CSVs:

  * age-binned metrics  -> ``bin,q1,med,q3,whislo,whishi``   (boxplot stats)
  * time-series metrics -> ``year,metric,med,pi95_low,pi95_high`` (median + 95% PI)

The surviving cached artifacts are therefore a *median + uncertainty band*,
and that band is across *calibration draws* (a posterior), not a fixed seed
sweep. There is no raw per-seed array to feed a z-score gate, and even the
band is not seed-for-seed comparable to a v3 seed sweep. So the right T13
construct is a **tolerance-band check**: v3 (multi-seed median) must land
inside the v2 median +/- band, and near the published 2017 data points.

Cached-file provenance (sibling validation repo, read at extract time only)
---------------------------------------------------------------------------
``<RWANDA>/results/v2.3.0_baseline/scens_timeseries.csv``  (cols:
``scenario,year,metric,value,low,high``) -> the v2.3 *aggregate* 2017
HIV-stratified incidence (Baseline scenario). This is the milestone-aligned
frozen-v2.3 number and is the PRIMARY target.

``<RWANDA>/results/v2.2.6_baseline/figS2_cancer_incidence_{with,no}_hiv.csv``
(cols ``bin,q1,med,q3,whislo,whishi``; bins 0..3 == age groups 25-35, 35-45,
45-55, 55+) -> the *by-age* boxplot target. Only the v2.2.6 cache carries the
figS2 by-age CSVs, so the by-age band is tagged ``source='v2.2.6'``; the
aggregate is ``source='v2.3.0'``.

Published 2017 data points (Rwanda 2017 HIV-stratified registry, ~13.1/100k
HIV- vs ~33/100k HIV+) are copied into ``tests/regression/data/`` so the target
does not depend on the sibling repo at load time.

Outputs
-------
Extract step writes a model-output target to the gitignored baseline dir
``tests/regression_baselines/hiv_baseline_v2.json`` (same convention as
``baseline_v23.py`` -> ``tests/regression_baselines/``; never committed).
The published data points live in committed CSVs under
``tests/regression/data/`` (published data, not model output).

``load_hiv_baseline()`` merges the two: it prefers the extracted cache file
and falls back to the published-only target if the cache file is absent, so
T13 can import a target without the sibling repo present.

Regeneration (only if the cache is unusable -- NOT needed now)
--------------------------------------------------------------
The cached CSVs are directly usable; no fresh v2 run is required for T11.
If a fresh v2.3 baseline is ever needed, regenerate the cache in the FROZEN
v2.3 install (NOT the v3 dev repo), then re-run this extractor:

    # in the Rwanda validation repo, with hpvsim==2.3.x (frozen install):
    "C:/Users/ryanhu/PycharmProjects/hpvsim_v23_frozen/.../python.exe" \
        run_calibration.py --resfolder results/v2.3.0_baseline
    # then, back in this repo (any env):
    python tests/regression/baseline_hiv_v2.py --extract

Do NOT run the multi-decade v2 calibration here; it is far too slow for a gate.
"""

import argparse
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / 'data'
_BASELINE_DIR = _HERE.parent / 'regression_baselines'  # gitignored
_CACHE_JSON = _BASELINE_DIR / 'hiv_baseline_v2.json'

# Sibling validation repo (dev-only; read at --extract time, never at runtime).
_RWANDA = (
    _HERE.parent.parent.parent
    / 'hpvsim_v23_validation' / 'hpvsim_rwanda'
)

# Age-bin index -> published age label (figS2 bins 0..3).
_AGE_LABELS = ['25-35', '35-45', '45-55', '55+']

# Tolerance proxy for the aggregate band when only a point estimate is
# available: +/- this fraction of the value. The cached band is preferred;
# this is the documented fallback.
_DEFAULT_REL_TOL = 0.25


# ---------------------------------------------------------------------------
# Published-data target (committed CSVs; no sibling repo needed)
# ---------------------------------------------------------------------------

def _read_published_point(fname):
    """Read a single-row published incidence CSV -> float value."""
    with open(_DATA_DIR / fname, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return float(rows[0]['value'])


def _read_published_by_age(fname):
    """Read a by-age published incidence CSV -> {age_label: value}."""
    out = {}
    with open(_DATA_DIR / fname, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            age = int(float(row['age']))
            # map start-age (25/35/45/55) to the published bin label
            label = {25: '25-35', 35: '35-45', 45: '45-55', 55: '55+'}[age]
            out[label] = float(row['value'])
    return out


def published_target():
    """Return the published 2017 Rwanda HIV-stratified data points.

    Returns a dict with aggregate and by-age published values (no model band).
    """
    return {
        'aggregate': {
            'cancer_incidence_no_hiv': _read_published_point(
                'rwanda_cancer_incidence_no_hiv.csv'),
            'cancer_incidence_with_hiv': _read_published_point(
                'rwanda_cancer_incidence_with_hiv.csv'),
        },
        'by_age': {
            'cancer_incidence_no_hiv': _read_published_by_age(
                'rwanda_cancer_incidence_by_age_no_hiv.csv'),
            'cancer_incidence_with_hiv': _read_published_by_age(
                'rwanda_cancer_incidence_by_age_with_hiv.csv'),
        },
    }


# ---------------------------------------------------------------------------
# Extract step: read the cached v2 baseline from the sibling repo
# ---------------------------------------------------------------------------

def _extract_aggregate_from_cache(resfolder, year=2017, scenario='Baseline'):
    """Pull med/low/high for the aggregate HIV-stratified metrics at `year`.

    Reads ``<resfolder>/scens_timeseries.csv`` (cols
    scenario,year,metric,value,low,high).
    """
    path = resfolder / 'scens_timeseries.csv'
    out = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['scenario'] != scenario:
                continue
            if abs(float(row['year']) - year) > 0.6:
                continue
            metric = row['metric']
            if metric in ('cancer_incidence_with_hiv', 'cancer_incidence_no_hiv'):
                out[metric] = {
                    'med': float(row['value']),
                    'low': float(row['low']),
                    'high': float(row['high']),
                }
    return out


def _extract_by_age_from_cache(resfolder):
    """Pull boxplot stats per age bin for the HIV-stratified metrics.

    Reads ``<resfolder>/figS2_cancer_incidence_{with,no}_hiv.csv``
    (cols bin,q1,med,q3,whislo,whishi).
    """
    out = {}
    for hiv, metric in (('with', 'cancer_incidence_with_hiv'),
                        ('no', 'cancer_incidence_no_hiv')):
        path = resfolder / f'figS2_cancer_incidence_{hiv}_hiv.csv'
        if not path.exists():
            continue
        per_age = {}
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                bi = int(float(row['bin']))
                label = _AGE_LABELS[bi]
                per_age[label] = {
                    'med': float(row['med']),
                    'low': float(row['q1']),
                    'high': float(row['q3']),
                    'whislo': float(row['whislo']),
                    'whishi': float(row['whishi']),
                }
        out[metric] = per_age
    return out


def extract(rwanda_repo=None, out_path=None,
            agg_folder='v2.3.0_baseline', age_folder='v2.2.6_baseline'):
    """Extract the v2 HIV-stratified target from the cached Rwanda baselines.

    Aggregate band comes from the v2.3.0 cache (milestone-aligned frozen v2.3);
    the by-age boxplot band comes from the v2.2.6 cache (only cache carrying
    the figS2 by-age CSVs). Writes a JSON target to the gitignored baseline dir.
    """
    repo = Path(rwanda_repo) if rwanda_repo else _RWANDA
    out_path = Path(out_path) if out_path else _CACHE_JSON

    agg = _extract_aggregate_from_cache(repo / 'results' / agg_folder)
    by_age = _extract_by_age_from_cache(repo / 'results' / age_folder)

    target = {
        '_provenance': {
            'aggregate_source': agg_folder,
            'by_age_source': age_folder,
            'rwanda_repo': str(repo),
            'note': ('v2 model-output band (median + uncertainty). Aggregate '
                     'is the 2017 Baseline-scenario med/low/high from '
                     'scens_timeseries.csv; by-age is the figS2 boxplot '
                     'q1/med/q3 + whiskers. NOT raw per-seed -> use a '
                     'tolerance-band gate, not a z-score gate.'),
        },
        'aggregate': agg,
        'by_age': by_age,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(target, f, indent=2)
    return target, out_path


# ---------------------------------------------------------------------------
# Loader API (what T13 imports)
# ---------------------------------------------------------------------------

def load_hiv_baseline(cache_path=None):
    """Return the v2 Rwanda HIV-stratified parity target for the T13 gate.

    The returned dict has::

        {
          'aggregate': {
              metric: {'value': float, 'low': float, 'high': float,
                       'published': float, 'source': str},
              ...
          },
          'by_age': {
              metric: {age_label: {'value','low','high','published'?}, ...},
              ...
          },
          'metrics': ['cancer_incidence_no_hiv', 'cancer_incidence_with_hiv'],
        }

    where for each aggregate metric ``value`` is the v2 model median and
    ``[low, high]`` is its uncertainty band (the tolerance band T13 checks the
    v3 multi-seed median against). ``published`` is the 2017 registry datum.

    Prefers the extracted cache JSON (``tests/regression_baselines/
    hiv_baseline_v2.json``). If absent, falls back to a published-only target
    (value == published, band == +/- ``_DEFAULT_REL_TOL``), so the gate can
    import a target without the sibling repo. The fallback path is reported
    via the ``source`` field on each aggregate metric.
    """
    cache_path = Path(cache_path) if cache_path else _CACHE_JSON
    pub = published_target()
    metrics = ['cancer_incidence_no_hiv', 'cancer_incidence_with_hiv']

    cache = None
    if cache_path.exists():
        with open(cache_path, encoding='utf-8') as f:
            cache = json.load(f)

    out = {'aggregate': {}, 'by_age': {}, 'metrics': metrics}

    for metric in metrics:
        pub_val = pub['aggregate'][metric]
        if cache and metric in cache.get('aggregate', {}):
            c = cache['aggregate'][metric]
            out['aggregate'][metric] = {
                'value': c['med'],
                'low': c['low'],
                'high': c['high'],
                'published': pub_val,
                'source': cache['_provenance']['aggregate_source'],
            }
        else:
            out['aggregate'][metric] = {
                'value': pub_val,
                'low': pub_val * (1 - _DEFAULT_REL_TOL),
                'high': pub_val * (1 + _DEFAULT_REL_TOL),
                'published': pub_val,
                'source': 'published_only',
            }

    for metric in metrics:
        pub_age = pub['by_age'][metric]
        out['by_age'][metric] = {}
        cache_age = (cache or {}).get('by_age', {}).get(metric, {})
        for label, pub_val in pub_age.items():
            if label in cache_age:
                c = cache_age[label]
                out['by_age'][metric][label] = {
                    'value': c['med'],
                    'low': c['low'],
                    'high': c['high'],
                    'published': pub_val,
                }
            else:
                out['by_age'][metric][label] = {
                    'value': pub_val,
                    'low': pub_val * (1 - _DEFAULT_REL_TOL),
                    'high': pub_val * (1 + _DEFAULT_REL_TOL),
                    'published': pub_val,
                }

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extract', action='store_true',
                   help='Read the cached v2 Rwanda baseline (sibling repo) and '
                        'write the gitignored target JSON.')
    p.add_argument('--rwanda-repo', default=None,
                   help='Override the hpvsim_rwanda repo path.')
    p.add_argument('--out', default=None, help='Override the output JSON path.')
    args = p.parse_args(argv)

    if args.extract:
        target, out_path = extract(rwanda_repo=args.rwanda_repo,
                                   out_path=args.out)
        print(f'Wrote v2 HIV-stratified target to {out_path}')
        for metric, v in target['aggregate'].items():
            print(f"  {metric}: med={v['med']:.2f} "
                  f"[{v['low']:.2f}, {v['high']:.2f}]")
        return 0

    # Default: load and print the resolved target.
    t = load_hiv_baseline()
    print('Resolved HIV-stratified parity target (aggregate, 2017):')
    for metric in t['metrics']:
        v = t['aggregate'][metric]
        print(f"  {metric}: value={v['value']:.2f} "
              f"[{v['low']:.2f}, {v['high']:.2f}] "
              f"published={v['published']} source={v['source']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
