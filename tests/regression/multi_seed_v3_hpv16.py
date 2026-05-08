"""HPV16-only multi-seed sweep, v3.

Same harness as ``multi_seed_v3.py`` but uses the M02 single-genotype anchor
(``anchor_hpv16.PARS``). Used to isolate cross-immunity vs same-genotype
dynamics: if v2 / v3 align here but diverge in the 4-genotype run, the bug
is in cross-immunity or coinfection mechanics.

Run from a v3 env at the repo root:

    python tests/regression/multi_seed_v3_hpv16.py --n 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_hpv16 import PARS, run_and_summarize  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent / 'v3_seeds_hpv16.json'


def run_seed(seed):
    # anchor_hpv16's run_and_summarize uses module-level PARS; mutate it for
    # this seed, capture, restore.
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