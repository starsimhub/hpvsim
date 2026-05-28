"""M01 acceptance gate: multi-seed mean parity vs. v2 HPV16-transmission baseline.

Runs N_V3_SEEDS v3 seeds with the M01 anchor PARS, then gates each
short-summary metric on

    z = (v3_mean - v2_mean) / sqrt(v2_SE^2 + v3_SE^2)

and fails any metric with |z| > Z_THRESHOLD. The v2 baseline is the
30-seed sweep at tests/regression/v2_m01_seeds_n30.json (gitignored;
regenerate via ``multi_seed_v2.py --anchor m01 --n 30`` from a v2 env).
"""
import json
from pathlib import Path

import pytest
import sciris as sc

import hpvsim as hpv

from tests.regression.anchor_m01 import PARS
from tests.regression.short_summary_m01 import build_summary_m01
from tests.regression.parity import parity_gate


BASELINE_PATH = Path('tests/regression/v2_m01_seeds_n30.json')
N_V3_SEEDS = 30
Z_THRESHOLD = 3.0

_SKIP_KEYS = frozenset({'_seed', '_total_pop'})


def _run_v3_seeds(n, start_seed=0):
    summaries = []
    for seed in range(start_seed, start_seed + n):
        pars = sc.dcp(PARS)
        pars['rand_seed'] = int(seed)
        sim = hpv.Sim(**pars)
        sim.run()
        summaries.append(build_summary_m01(sim))
    return summaries


@pytest.mark.slow
def test_m01_short_summary_parity():
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v2 M01 baseline at {BASELINE_PATH}. Regenerate via '
            f'`python tests/regression/multi_seed_v2.py --anchor m01 --n 30` '
            f'from a v2 hpvsim env.'
        )
    v2_rows = json.loads(BASELINE_PATH.read_text())
    v3_rows = _run_v3_seeds(N_V3_SEEDS, start_seed=0)

    failures = parity_gate(
        v3_rows, v2_rows, z_threshold=Z_THRESHOLD, skip_keys=_SKIP_KEYS,
    )

    if failures:
        details = '\n'.join(
            f'  {name:<40} z={z:+.2f}' for name, z in failures
        )
        pytest.fail(
            f'M01 mean parity drift exceeds |z|>{Z_THRESHOLD} on '
            f'{len(failures)} metrics (v2 n={len(v2_rows)}, v3 n={len(v3_rows)}):\n{details}'
        )
