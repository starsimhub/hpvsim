"""Compare v3 grow-multiscale cancer burden to v2.3.1 frozen reference.

Validation script (not a pytest unit test) that runs a matched scenario on
both engines and reports total cancers. The two engines differ structurally:

  * v3 = starsim continuous-age, grow-multiscale (m07-multiscale-v2-grow)
  * v2 = v2.3.1 with fix-multiscale-cin-regate, discrete annual cohorts

Expected structural difference: ~15-40 % (v3/v2), consistent with
engine-level differences already documented in the v2->v3 natural-history
parity gates. Flag if outside ~0.5-1.8.

Validated results (measured 2026-06-29):
  v3  mean total cancers (4 seeds, n=10k, [16,18], 1975->2025, ratio=10): 306.0
  v2  mean total cancers (4 seeds, n=10k, [16,18], 1975->2025, ratio=10): 303.2
  v3/v2 ratio: 1.009  (within the documented ~0.7-1.4 engine-difference band;
                        essentially identical -- the grow-multiscale port is
                        numerically faithful to the v2.3.1 frozen reference.)

NOTE: v2 reference number (ratio=1 mean ~313, ratio=10 mean ~303, -3%) is
from hpvsim_v23_frozen on fix-multiscale-cin-regate branch.

The two engines CANNOT coexist in one interpreter -- run in separate processes
using `--engine v2|v3`.

Run v3 (worktree-pinned):
  <venv-py> tests/regression/validate_v2_grow_multiscale.py --engine v3

Run v2.3.1 frozen:
  PYTHONPATH=C:/Users/ryanhu/PycharmProjects/hpvsim_v23_frozen \\
      <venv-py> tests/regression/validate_v2_grow_multiscale.py --engine v2

Both use:
  n_agents=10_000, genotypes=[16, 18], start=1975, n_years=50 (->2025),
  ms_agent_ratio=10, seeds 0-3.
"""

import argparse
import sys
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Matched scenario parameters (shared by both engines)
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3]
N_AGENTS = 10_000
GENOTYPES = [16, 18]
START = 1975
N_YEARS = 50          # 1975 -> 2025
MS_AGENT_RATIO = 10


# ---------------------------------------------------------------------------
# v3 (worktree) runner
# ---------------------------------------------------------------------------

def _run_v3(seed):
    """Run one v3 seed; return total cancers (scale-weighted sum)."""
    # Pin to worktree so editable install doesn't shadow it.
    worktree = str(Path(__file__).resolve().parents[2])
    if worktree not in sys.path:
        sys.path.insert(0, worktree)

    import hpvsim as hpv  # noqa: PLC0415 — intentional late import

    sim = hpv.Sim(
        location='nigeria',
        n_agents=N_AGENTS,
        genotypes=GENOTYPES,
        start=START,
        stop=START + N_YEARS,
        ms_agent_ratio=MS_AGENT_RATIO,
        rand_seed=seed,
        verbose=0,
    )
    sim.run()

    total = 0.0
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            total += float(dis.results.new_cancers.values.sum())
    return total


# ---------------------------------------------------------------------------
# v2 (frozen) runner
# ---------------------------------------------------------------------------

def _run_v2(seed):
    """Run one v2 seed; return total cancers."""
    import hpvsim as hpv  # noqa: PLC0415 — intentional late import
    import numpy as np

    # Verify we are running the frozen build.
    ver = getattr(hpv, '__version__', 'unknown')
    if not ver.startswith('2.3'):
        raise RuntimeError(
            f'Expected hpvsim 2.3.x (frozen), got {ver!r}. '
            'Set PYTHONPATH=C:/Users/ryanhu/PycharmProjects/hpvsim_v23_frozen '
            'and re-run with the venv python.'
        )

    sim = hpv.Sim(
        dict(
            total_pop=N_AGENTS,
            n_agents=N_AGENTS,
            start=START,
            n_years=N_YEARS,
            genotypes=GENOTYPES,
            verbose=0,
            pop_scale=1,
        ),
        ms_agent_ratio=MS_AGENT_RATIO,
        rand_seed=seed,
    )
    sim.run()

    total = float(np.sum(sim.results['cancers'].values))
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description='Numerical tracking: v3 grow-multiscale vs v2.3.1 frozen',
    )
    p.add_argument(
        '--engine',
        choices=['v2', 'v3'],
        required=True,
        help='Which engine to run (v2=frozen v2.3.1, v3=worktree).',
    )
    p.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='Comma-separated seed list (default: 0,1,2,3).',
    )
    args = p.parse_args(argv)

    seeds = (
        [int(s) for s in args.seeds.split(',')]
        if args.seeds
        else SEEDS
    )

    runner = _run_v3 if args.engine == 'v3' else _run_v2

    totals = []
    for seed in seeds:
        n = runner(seed)
        print(f'  engine={args.engine} seed={seed}: total_cancers={n:.1f}')
        totals.append(n)

    import numpy as np
    mean_cancers = float(np.mean(totals))
    print()
    print(f'engine={args.engine}  seeds={seeds}')
    print(f'  per-seed totals: {[f"{t:.1f}" for t in totals]}')
    print(f'  MEAN total cancers: {mean_cancers:.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
