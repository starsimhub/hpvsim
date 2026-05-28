"""M06 routine therapeutic vaccination parity gate: multi-seed mean z-score vs v2 baseline.

Same gate pattern as test_m05_vx_routine_parity.py (10 v3 seeds vs 30 v2
seeds; |z| < 3 per metric). Metric set: two txvx scalars at end-of-sim
(n_tx_vaccinated_2060, n_txvx_doses_2060) plus cancer incidence over
[2030, 2060).

Requires the locally-regenerated v2 baseline at
``tests/regression/v2_seeds_n30_txvx.json``. See
``tests/regression/multi_seed_v2_txvx.py`` for how to (re)generate it
(must be run from a v2.3 hpvsim env, not the v3 env).
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests.regression.anchor_txvx_routine import build_v3_sim


BASELINE_PATH = Path(__file__).parent / 'regression' / 'v2_seeds_n30_txvx.json'
N_V3_SEEDS = 10
Z_THRESHOLD = 3.0

_SKIP_KEYS = frozenset({'_seed'})


def _run_v3_seeds(n, start_seed=0):
    summaries = []
    for seed in range(start_seed, start_seed + n):
        sim = build_v3_sim()
        sim.pars['rand_seed'] = int(seed)
        sim.run()
        row = {}

        # --- txvx scalars at end-of-sim (2060) ---
        # BoolArr.sum() and FloatArr.sum() in v3 automatically exclude dead
        # agents — matching v2's alive-masked people.tx_vaccinated[alive].sum().
        txvx = sim.interventions['txvx']

        row['n_tx_vaccinated_2060'] = int(txvx.tx_vaccinated.sum())
        row['n_txvx_doses_2060'] = int(txvx.txvx_doses.sum())

        # --- cancer incidence over [2030, 2060) ---
        # v3 timevec may be a DateArray; convert to float years.
        years = sim.results.timevec
        if hasattr(years, 'years'):
            year_floats = np.asarray(years.years, dtype=float)
        else:
            year_floats = np.asarray(years, dtype=float)
        mask = (year_floats >= 2030) & (year_floats < 2060)

        n_cancers = float(sim.results.hpvtotal.new_cancers[mask].sum())
        pop = sim.results.n_alive[mask]
        # dt in years — mirrors M05's pattern.
        dt = sim.t.dt.years if hasattr(sim.t.dt, 'years') else float(sim.t.dt)
        py = float((pop * dt).sum())
        row['cancer_incidence_2030_2060'] = n_cancers / max(py, 1.0)

        summaries.append(row)
    return summaries


def _mean_se(rows, key):
    vals = np.array([float(r[key]) for r in rows if key in r], dtype=float)
    if vals.size == 0:
        return None
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, se


@pytest.mark.slow
def test_m06_txvx_short_summary_parity():
    """3-metric z-score gate: |z| < 3 across txvx scalars + cancer incidence."""
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'v2 baseline JSON not present — regenerate locally via '
            f'tests/regression/multi_seed_v2_txvx.py from a v2.3 env. '
            f'Expected path: {BASELINE_PATH}'
        )
    v2_rows = json.loads(BASELINE_PATH.read_text())
    v3_rows = _run_v3_seeds(N_V3_SEEDS)

    keys = (set(v2_rows[0].keys()) & set(v3_rows[0].keys())) - _SKIP_KEYS

    failures = []
    for key in sorted(keys):
        v2 = _mean_se(v2_rows, key)
        v3 = _mean_se(v3_rows, key)
        if v2 is None or v3 is None:
            continue
        v2_mean, v2_se = v2
        v3_mean, v3_se = v3
        denom = math.sqrt(v2_se ** 2 + v3_se ** 2)
        if denom == 0:
            if v2_mean != v3_mean:
                failures.append(
                    f'{key}: v3={v3_mean!r} v2={v2_mean!r} (deterministic but unequal)'
                )
            continue
        z = (v3_mean - v2_mean) / denom
        if abs(z) >= Z_THRESHOLD:
            failures.append(
                f'{key}: |z|={abs(z):.2f} (v3={v3_mean:.3g}+/-{v3_se:.2g} vs '
                f'v2={v2_mean:.3g}+/-{v2_se:.2g})'
            )
    assert not failures, '\n'.join([f'Metrics outside |z|<{Z_THRESHOLD}:'] + failures)
