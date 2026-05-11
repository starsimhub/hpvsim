"""M03 development gate: multi-seed mean parity vs. v2 4-genotype baseline.

Per-seed CVs for per-genotype count metrics run 14-22% (e.g. hpv18 total
infections SD ~5k on a mean ~30k). The previous single-seed 10% gate
failed on seed=0 specifically because v2's seed-0 baseline is an upper-
tail draw for hpv18; the mean across 30 v2 seeds vs 10 v3 seeds is
statistically indistinguishable (all 40 metrics with |z| < 1).

This test runs ``N_V3_SEEDS`` v3 seeds with the same anchor PARS, then
gates each summary metric on

    z = (v3_mean - v2_mean) / sqrt(v2_SE^2 + v3_SE^2)

and fails any metric with ``|z| > Z_THRESHOLD``. The v2 baseline is the
30-seed sweep at ``tests/regression/v2_seeds_n30.json`` (gitignored;
regenerate via ``multi_seed_v2.py --n 30`` from a v2 hpvsim env).
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv

from tests.regression.anchor_4genotype import PARS
from tests.regression.short_summary import build_summary


BASELINE_PATH = Path('tests/regression/v2_seeds_n30.json')
N_V3_SEEDS = 10
Z_THRESHOLD = 3.0
GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')

# Non-metric fields present in seed-sweep JSON files.
_SKIP_KEYS = frozenset({'_seed', '_total_pop', 'total population'})


def _run_v3_seeds(n, start_seed=0):
    summaries = []
    for seed in range(start_seed, start_seed + n):
        pars = sc.dcp(PARS)
        pars['rand_seed'] = int(seed)
        sim = hpv.Sim(**pars)
        sim.run()
        summaries.append(build_summary(sim, GENOTYPES))
    return summaries


def _mean_se(rows, key):
    vals = np.array([float(r[key]) for r in rows if key in r], dtype=float)
    if vals.size == 0:
        return None
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, se


@pytest.mark.slow
def test_short_summary_parity_4genotype():
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v2 multi-seed baseline at {BASELINE_PATH}. Regenerate via '
            f'`python tests/regression/multi_seed_v2.py --n 30 --out {BASELINE_PATH}` '
            f'from a v2 hpvsim env.'
        )
    v2_rows = json.loads(BASELINE_PATH.read_text())
    v3_rows = _run_v3_seeds(N_V3_SEEDS, start_seed=0)

    metric_keys = sorted((set(v2_rows[0]) & set(v3_rows[0])) - _SKIP_KEYS)

    failures = []
    for key in metric_keys:
        v2_stats = _mean_se(v2_rows, key)
        v3_stats = _mean_se(v3_rows, key)
        if v2_stats is None or v3_stats is None:
            continue
        v2_mean, v2_se = v2_stats
        v3_mean, v3_se = v3_stats
        se_combo = math.sqrt(v2_se ** 2 + v3_se ** 2)
        if se_combo == 0:
            # Both distributions degenerate: pass iff means match exactly.
            if v2_mean != v3_mean:
                failures.append((key, v2_mean, v3_mean, float('inf')))
            continue
        z = (v3_mean - v2_mean) / se_combo
        if abs(z) > Z_THRESHOLD:
            failures.append((key, v2_mean, v3_mean, z))

    if failures:
        details = '\n'.join(
            f'  {k:<50} v2={v2:.4g}  v3={v3:.4g}  z={z:+.2f}'
            for k, v2, v3, z in failures
        )
        pytest.fail(
            f'M03 mean parity drift exceeds |z|>{Z_THRESHOLD} on '
            f'{len(failures)} of {len(metric_keys)} metrics '
            f'(v2 n={len(v2_rows)}, v3 n={len(v3_rows)}):\n{details}'
        )
