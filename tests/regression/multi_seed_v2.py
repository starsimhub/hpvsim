"""Multi-seed sweep of one of the v2.3 anchor sims for M01/M02/M03.

Runs the v2 baseline for the chosen anchor with N seeds and writes a
JSON list of per-seed summary dicts. The summary key layout matches the
matching v3 builder (short_summary_m01.METRIC_KEYS_M01 for M01,
short_summary.METRIC_KEYS via `<g>.<metric>` + `any.<metric>` for M02 / M03).

Reuses helpers from ``baseline_v23.py`` — same pop / config /
lifetime-mean-age semantics that the production v2 baselines use.

Run from a v2.3 env at the repo root:

    "<v2 env>/python.exe" tests/regression/multi_seed_v2.py --anchor m03_4genotype --n 30
    "<v2 env>/python.exe" tests/regression/multi_seed_v2.py --anchor m02 --n 30
    "<v2 env>/python.exe" tests/regression/multi_seed_v2.py --anchor m01 --n 30

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
    PARS_HPV16,
    PARS_HPV16_TRANSMISSION_ONLY,
    _per_genotype_metrics_v2,
    _aggregate_metrics_v2,
    _summary_v2_m01,
)


DEFAULT_OUT_4GENOTYPE = Path(__file__).resolve().parent / 'v2_seeds_n30.json'
DEFAULT_OUT_M01 = Path(__file__).resolve().parent / 'v2_m01_seeds_n30.json'
DEFAULT_OUT_M02 = Path(__file__).resolve().parent / 'v2_m02_seeds_n30.json'

ANCHORS = {
    'm03_4genotype': dict(
        pars=PARS_4GENOTYPE,
        genotypes=('hpv16', 'hpv18', 'hi5', 'ohr'),
        out=DEFAULT_OUT_4GENOTYPE,
        mode='m03',
    ),
    'm02': dict(
        pars=PARS_HPV16,
        genotypes=('hpv16',),
        out=DEFAULT_OUT_M02,
        mode='m03',
    ),
    'm01': dict(
        pars=PARS_HPV16_TRANSMISSION_ONLY,
        genotypes=('hpv16',),
        out=DEFAULT_OUT_M01,
        mode='m01',
    ),
}


def run_seed(seed, anchor='m03_4genotype'):
    cfg = ANCHORS[anchor]
    pars = sc.dcp(cfg['pars'])
    pars['rand_seed'] = int(seed)
    pars['genotypes'] = list(cfg['genotypes'])
    sim = hpv.Sim(pars)
    sim.run()

    summary = {}
    if cfg['mode'] == 'm03':
        genotype_map = sim.pars['genotype_map']
        genotype_pairs = [(i, k) for i, k in sorted(genotype_map.items())]
        for gen_idx, gen_key in genotype_pairs:
            per = _per_genotype_metrics_v2(sim, gen_idx, gen_key)
            for k, v in per.items():
                summary[f'{gen_key}.{k}'] = float(v)
        agg = _aggregate_metrics_v2(sim, genotype_pairs)
        for k, v in agg.items():
            summary[f'any.{k}'] = float(v)
    elif cfg['mode'] == 'm01':
        genotype_map = sim.pars['genotype_map']
        gen_idx = next((i for i, k in genotype_map.items() if k == 'hpv16'), None)
        if gen_idx is None:
            raise ValueError(f"hpv16 not found in genotype_map: {genotype_map}")
        per = _summary_v2_m01(sim, gen_idx)
        for k, v in per.items():
            summary[k] = float(v)
    else:
        raise ValueError(f"Unknown anchor mode: {cfg['mode']}")

    summary['_seed'] = int(seed)
    if 'n_alive' in sim.results:
        summary['_total_pop'] = float(sim.results['n_alive'][-1])
    else:
        summary['_total_pop'] = float(sim.results['pop_size'][-1])
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--anchor', choices=list(ANCHORS), default='m03_4genotype',
                   help='Anchor to run (m01, m02, or m03_4genotype).')
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=None,
                   help='Override the default output path for this anchor.')
    args = p.parse_args(argv)

    out_path = args.out or ANCHORS[args.anchor]['out']
    seeds = list(range(args.start_seed, args.start_seed + args.n))
    results = []
    t0 = time.time()
    for seed in seeds:
        ts = time.time()
        s = run_seed(seed, anchor=args.anchor)
        dt = time.time() - ts
        results.append(s)
        diag_key = next(iter(k for k in s if k not in ('_seed', '_total_pop')))
        print(f'  seed {seed} ({args.anchor}): done in {dt:.1f}s '
              f'({diag_key}={s[diag_key]:.4g})')
    total = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {len(results)} seed summaries to {out_path} in {total:.1f}s')


if __name__ == '__main__':
    sys.exit(main())