"""Multi-seed sweep of the v2.3 4-genotype anchor sim.

Runs the v2 baseline with N seeds and writes a JSON list of 40-entry
summary dicts (key layout matches ``short_summary.METRIC_KEYS`` and
``v3_seeds.json`` produced by ``multi_seed_v3.py``).

Reuses ``_per_genotype_metrics_v2`` and ``_aggregate_metrics_v2`` from
``baseline_v23.py`` — same pop / config / lifetime-mean-age semantics that
the production v2 baselines use.

Run from a v2.3 env at the repo root:

    "<v2 env>/python.exe" tests/regression/multi_seed_v2.py
    "<v2 env>/python.exe" tests/regression/multi_seed_v2.py --n 20

DO NOT run inside the v3 env.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc
import hpvsim as hpv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_v23 import (  # noqa: E402
    PARS_4GENOTYPE,
    _per_genotype_metrics_v2,
    _aggregate_metrics_v2,
)


DEFAULT_OUT = Path(__file__).resolve().parent / 'v2_seeds.json'
DEFAULT_GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


def run_seed(seed, genotypes=DEFAULT_GENOTYPES):
    pars = sc.dcp(PARS_4GENOTYPE)
    pars['rand_seed'] = int(seed)
    pars['genotypes'] = list(genotypes)
    sim = hpv.Sim(pars)
    sim.run()

    genotype_map = sim.pars['genotype_map']
    genotype_pairs = [(i, k) for i, k in sorted(genotype_map.items())]

    summary = {}
    for gen_idx, gen_key in genotype_pairs:
        per = _per_genotype_metrics_v2(sim, gen_idx, gen_key)
        for k, v in per.items():
            summary[f'{gen_key}.{k}'] = float(v)

    agg = _aggregate_metrics_v2(sim, genotype_pairs)
    for k, v in agg.items():
        summary[f'any.{k}'] = float(v)

    summary['_seed'] = int(seed)
    if 'n_alive' in sim.results:
        summary['_total_pop'] = float(sim.results['n_alive'][-1])
    else:
        summary['_total_pop'] = float(sim.results['pop_size'][-1])
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10)
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
    for seed in seeds:
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