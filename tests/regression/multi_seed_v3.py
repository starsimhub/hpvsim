"""Multi-seed sweep of the v3 4-genotype anchor sim.

Runs the M03 anchor with N seeds (default 10) and writes the 40-entry
``build_summary`` dict for each, into a JSON list at ``--out`` (default
``v3_seeds.json`` next to this script). Pair with ``multi_seed_v2.py`` and
``compare_seeds.py`` to test whether the v2 baseline mean falls within v3's
empirical seed-distribution.

Run from a v3 env at the repo root:

    python tests/regression/multi_seed_v3.py
    python tests/regression/multi_seed_v3.py --n 20 --out v3_seeds.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc

import hpvsim as hpv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_4genotype import PARS  # noqa: E402
from short_summary import build_summary  # noqa: E402


DEFAULT_GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')
DEFAULT_OUT = Path(__file__).resolve().parent / 'v3_seeds.json'


def run_seed(seed, genotypes=DEFAULT_GENOTYPES):
    pars = sc.dcp(PARS)
    pars['rand_seed'] = int(seed)
    pars['genotypes'] = list(genotypes)
    sim = hpv.Sim(**pars)
    sim.run()
    summary = build_summary(sim, genotypes)
    summary['_seed'] = int(seed)
    summary['_total_pop'] = float(sim.results['n_alive'][-1])
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10, help='Number of seeds (default 10)')
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--genotypes', type=str, default=None,
                   help='Comma-separated genotype list (default hpv16,hpv18,hi5,ohr)')
    args = p.parse_args(argv)

    if args.genotypes:
        genotypes = tuple(g.strip() for g in args.genotypes.split(','))
    else:
        genotypes = DEFAULT_GENOTYPES

    seeds = list(range(args.start_seed, args.start_seed + args.n))
    results = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        ts = time.time()
        s = run_seed(seed, genotypes=genotypes)
        dt = time.time() - ts
        results.append(s)
        print(f'  seed {seed}: done in {dt:.1f}s '
              f'(any.total HPV infections={s["any.total HPV infections"]:.0f})')
    total = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {len(results)} seed summaries to {args.out} in {total:.1f}s')


if __name__ == '__main__':
    sys.exit(main())