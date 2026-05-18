"""Compare v2 and v3 multi-seed summaries.

Loads ``v2_seeds.json`` and ``v3_seeds.json`` (each a list of 40-entry
summary dicts), then, for every metric:

  * reports v2 mean ± std (n_v2 seeds) and v3 mean ± std (n_v3 seeds)
  * computes ``z = (v3_mean - v2_mean) / pooled_std`` (a Welch-style
    standardized gap; not a formal t-test, just a direction signal)
  * computes ``v2 percentile in v3 distribution`` — i.e. where v2's mean
    falls in v3's empirical seed distribution. A value near 50 means the
    v2 baseline is squarely typical of v3 seed variability; values near
    0 or 100 mean v3 systematically differs from v2.

Run after both sweeps:

    python tests/regression/multi_seed_v3.py
    "<v2 env>/python.exe" tests/regression/multi_seed_v2.py
    python tests/regression/compare_seeds.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DEFAULT_V2 = Path(__file__).resolve().parent / 'v2_seeds.json'
DEFAULT_V3 = Path(__file__).resolve().parent / 'v3_seeds.json'

GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')
METRICS = (
    'total HPV infections',
    'total cancers',
    'total cancer deaths',
    'mean HPV prevalence (%)',
    'mean cancer incidence (per 100k)',
    'mean age of infection (years)',
    'mean age of cancer (years)',
    'mean age of cancer death (years)',
)


def load(path):
    with open(path) as f:
        return json.load(f)


def stack(rows, key):
    """Return ndarray of values for ``key`` across all seeds."""
    return np.array([float(r[key]) for r in rows if key in r])


def summarize(rows_v2, rows_v3, key):
    a = stack(rows_v2, key)
    b = stack(rows_v3, key)
    if not len(a) or not len(b):
        return None
    out = {
        'v2_mean': a.mean(), 'v2_std': a.std(ddof=1) if len(a) > 1 else 0.0,
        'v3_mean': b.mean(), 'v3_std': b.std(ddof=1) if len(b) > 1 else 0.0,
        'n_v2': len(a), 'n_v3': len(b),
    }
    pooled_std = np.sqrt((out['v2_std']**2 + out['v3_std']**2) / 2.0)
    if pooled_std > 0:
        out['z'] = (out['v3_mean'] - out['v2_mean']) / pooled_std
    else:
        out['z'] = float('nan')
    if out['v2_mean'] != 0:
        out['pct_diff'] = 100 * (out['v3_mean'] - out['v2_mean']) / out['v2_mean']
    else:
        out['pct_diff'] = float('nan')
    # Where does v2 mean fall in v3's seed distribution? Percentile.
    if len(b) > 1:
        out['v2_in_v3_pct'] = float((b < out['v2_mean']).mean() * 100)
    else:
        out['v2_in_v3_pct'] = float('nan')
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--v2', type=Path, default=DEFAULT_V2)
    p.add_argument('--v3', type=Path, default=DEFAULT_V3)
    args = p.parse_args(argv)

    if not args.v2.exists():
        print(f'Missing v2 seed file: {args.v2}'); return 1
    if not args.v3.exists():
        print(f'Missing v3 seed file: {args.v3}'); return 1

    v2 = load(args.v2)
    v3 = load(args.v3)
    print(f'Loaded n_v2={len(v2)} n_v3={len(v3)}')
    print()

    # Aggregate any.* metrics first (most useful).
    print('=== any.* (4-genotype aggregate) ===')
    hdr = (f'{"metric":<35}  {"v2_mean":>10}  {"v2_std":>8}  '
           f'{"v3_mean":>10}  {"v3_std":>8}  {"z":>6}  {"pct_diff":>8}  {"v2_in_v3_pct":>12}')
    print(hdr)
    print('-' * len(hdr))
    for m in METRICS:
        key = f'any.{m}'
        s = summarize(v2, v3, key)
        if s is None:
            continue
        print(f'{m:<35}  {s["v2_mean"]:>10.4g}  {s["v2_std"]:>8.4g}  '
              f'{s["v3_mean"]:>10.4g}  {s["v3_std"]:>8.4g}  '
              f'{s["z"]:>6.2f}  {s["pct_diff"]:>7.1f}%  {s["v2_in_v3_pct"]:>11.1f}%')

    print()
    print('=== Per-genotype, count metrics ===')
    print(hdr)
    print('-' * len(hdr))
    for g in GENOTYPES:
        for m in ('total HPV infections', 'total cancers', 'total cancer deaths'):
            key = f'{g}.{m}'
            s = summarize(v2, v3, key)
            if s is None: continue
            label = f'{g}.{m}'
            print(f'{label:<35}  {s["v2_mean"]:>10.4g}  {s["v2_std"]:>8.4g}  '
                  f'{s["v3_mean"]:>10.4g}  {s["v3_std"]:>8.4g}  '
                  f'{s["z"]:>6.2f}  {s["pct_diff"]:>7.1f}%  {s["v2_in_v3_pct"]:>11.1f}%')

    print()
    print('=== Per-genotype, mean ages ===')
    print(hdr)
    print('-' * len(hdr))
    for g in GENOTYPES:
        for m in ('mean age of infection (years)', 'mean age of cancer (years)',
                  'mean age of cancer death (years)'):
            key = f'{g}.{m}'
            s = summarize(v2, v3, key)
            if s is None: continue
            label = f'{g}.{m}'
            print(f'{label:<35}  {s["v2_mean"]:>10.4g}  {s["v2_std"]:>8.4g}  '
                  f'{s["v3_mean"]:>10.4g}  {s["v3_std"]:>8.4g}  '
                  f'{s["z"]:>6.2f}  {s["pct_diff"]:>7.1f}%  {s["v2_in_v3_pct"]:>11.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())