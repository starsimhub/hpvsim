"""HPV16-only multi-seed sweep, v2.3.

Mirror of multi_seed_v3_hpv16.py. Reuses ``baseline_v23.run_and_summarize``
(M02 path, single-genotype HPV16 with the same lifetime-mean-age semantics).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_v23 import PARS, run_and_summarize  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent / 'v2_seeds_hpv16.json'


def run_seed(seed):
    saved = PARS.get('rand_seed')
    PARS['rand_seed'] = int(seed)
    try:
        short, total_pop = run_and_summarize()
    finally:
        PARS['rand_seed'] = saved
    out = dict(short)
    out['_seed'] = int(seed)
    out['_total_pop'] = float(total_pop)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    seeds = list(range(args.start_seed, args.start_seed + args.n))
    results = []
    t0 = time.time()
    for seed in seeds:
        ts = time.time()
        s = run_seed(seed)
        dt = time.time() - ts
        results.append(s)
        print(f'  seed {seed}: done in {dt:.1f}s '
              f'(total HPV infections={s["total HPV infections"]:.0f})')
    total = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {len(results)} hpv16 seed summaries to {args.out} in {total:.1f}s')


if __name__ == '__main__':
    sys.exit(main())