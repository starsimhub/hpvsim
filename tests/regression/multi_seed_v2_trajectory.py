"""Multi-seed sweep of the v2.3 4-genotype anchor trajectories.

Runs the v2 baseline with N seeds and writes a JSON containing the per-seed
trajectories of ``cum_infections_any`` and ``cum_cancers_any``, computed from
v2's ``infections_by_genotype`` / ``cancers_by_genotype`` arrays the same way
``baseline_v23.regen_4genotype`` derives the single-seed trajectory baseline.

Pair with ``tests/test_m03_trajectory_parity.py`` which loads this file and
gates the v3 mean trajectory against the v2 mean trajectory with an SE-aware
z-score check.

Run from a v2.3 env at the repo root:

    "<v2 env>/python.exe" tests/regression/multi_seed_v2_trajectory.py --n 30

DO NOT run inside the v3 env.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sciris as sc
import hpvsim as hpv  # v2.3 here

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_v23 import PARS_4GENOTYPE  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent / 'v2_trajectories.json'


def run_seed(seed):
    """Run one v2 seed and return annual cumulative-aggregate trajectories.

    v2's ``infections_by_genotype`` / ``cancers_by_genotype`` are stored at
    annual cadence (one row per year of the sim, 71 values for 1990-2060).
    We build the matching annual time axis explicitly from ``start`` and
    ``end`` — ``sim.yearvec`` is the per-step axis (~284 entries) and would
    not align with the annual series.
    """
    pars = sc.dcp(PARS_4GENOTYPE)
    pars['rand_seed'] = int(seed)
    sim = hpv.Sim(pars)
    sim.run()
    inf_bg = np.asarray(sim.results['infections_by_genotype'])  # (n_g, n_years)
    can_bg = np.asarray(sim.results['cancers_by_genotype'])     # (n_g, n_years)
    n_years = inf_bg.shape[1]
    start = int(pars['start'])
    annual_time = [float(start + i) for i in range(n_years)]
    return {
        'cum_infections_any': np.cumsum(inf_bg.sum(axis=0)).tolist(),
        'cum_cancers_any':    np.cumsum(can_bg.sum(axis=0)).tolist(),
        'time':               annual_time,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    seeds = list(range(args.start_seed, args.start_seed + args.n))
    time_vec = None
    inf_runs = []
    can_runs = []
    t0 = time.time()
    for seed in seeds:
        ts = time.time()
        run = run_seed(seed)
        if time_vec is None:
            time_vec = run['time']
        inf_runs.append(run['cum_infections_any'])
        can_runs.append(run['cum_cancers_any'])
        dt = time.time() - ts
        print(f'  seed {seed}: done in {dt:.1f}s '
              f'(end cum_infections_any={inf_runs[-1][-1]:.0f})')
    total = time.time() - t0

    payload = {
        'metadata': {
            'hpvsim_version': hpv.__version__,
            'pars': dict(PARS_4GENOTYPE),
            'n_seeds': len(seeds),
            'seeds': seeds,
        },
        'time': time_vec,
        'series': {
            'cum_infections_any': inf_runs,
            'cum_cancers_any':    can_runs,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f)
    print(f'Wrote {len(seeds)} seed trajectories to {args.out} in {total:.1f}s')


if __name__ == '__main__':
    sys.exit(main())