"""Compare two multi-seed JSON files (pre/post Tier 1 perf changes).

Each JSON is a list of per-seed dicts (output of multi_seed_v3.py). We
compute mean ± std per metric across seeds, then for each metric report
whether the post mean lies inside post's own ±1 SD interval around the
pre mean (i.e. the change is within seed-to-seed noise).

Run:

    python tests/regression/compare_perf_seeds.py \
        tests/regression/v3_seeds_pre_perf.json \
        tests/regression/v3_seeds_post_perf.json
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def load(path):
    with open(path) as f:
        return json.load(f)


def collect(data):
    keys = sorted({k for row in data for k in row if not k.startswith('_')})
    out = {}
    for k in keys:
        vals = [float(row[k]) for row in data if k in row]
        out[k] = vals
    return out


def stats(vals):
    if len(vals) == 0:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('pre', type=Path)
    p.add_argument('post', type=Path)
    p.add_argument('--threshold', type=float, default=1.0,
                   help='Number of pre-SDs of (post_mean - pre_mean) before flagging')
    args = p.parse_args(argv)

    pre = collect(load(args.pre))
    post = collect(load(args.post))

    keys = sorted(set(pre) & set(post))
    print(f'{"metric":<48} {"pre mean":>12} {"pre SD":>10} {"post mean":>12} '
          f'{"post SD":>10} {"delta":>10} {"|d|/pre_SD":>10} flag')
    flagged = []
    for k in keys:
        pre_m, pre_s = stats(pre[k])
        post_m, post_s = stats(post[k])
        delta = post_m - pre_m
        z = (delta / pre_s) if (pre_s and pre_s > 0) else float('nan')
        flag = ''
        if not math.isnan(z) and abs(z) > args.threshold:
            flag = '<-- exceeds threshold'
            flagged.append((k, abs(z)))
        print(f'{k:<48} {pre_m:>12.4g} {pre_s:>10.3g} {post_m:>12.4g} '
              f'{post_s:>10.3g} {delta:>+10.4g} {z:>10.2f} {flag}')

    print()
    print(f'Total metrics: {len(keys)}')
    print(f'Flagged (|delta| > {args.threshold} pre-SD): {len(flagged)}')
    for k, z in sorted(flagged, key=lambda x: -x[1]):
        print(f'  {k}: |z|={z:.2f}')

    # Also report timing if present
    pre_walls = [r.get('_wall_seconds') or r.get('_total_pop') for r in load(args.pre)]
    print()
    print(f'Pre n_seeds: {len(load(args.pre))}')
    print(f'Post n_seeds: {len(load(args.post))}')


if __name__ == '__main__':
    main()