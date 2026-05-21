"""M05 routine-vx trajectory parity gate: per-year z-score vs v2.

For each (year, metric) cell we compute the same z-score as the short-
summary gate, against v2's per-year distribution. Gates each cell at
|z| < 3. This is the strictest shape-check in M05.

Runs on the routine anchor only — the campaign anchor is well-summarised
by the short-summary gate and the trajectory test is the heaviest test
in the suite.

Requires the locally-regenerated v2 baseline at
``tests/regression/v2_seeds_n30_vx_routine.json`` AND that baseline must
include the ``_trajectory`` field (regenerate after Task 12's
multi_seed_v2_vx.py update). See ``tests/regression/README_m05.md``.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import sciris as sc

from tests.regression.anchor_vx_routine import build_v3_sim


BASELINE_PATH = Path(__file__).parent / 'regression' / 'v2_seeds_n30_vx_routine.json'
N_V3_SEEDS = 10
Z_THRESHOLD = 3.0

TRAJECTORY_METRICS = ('new_cancers', 'hpv_total_infections', 'new_vaccinated')


def _v3_trajectory_row(sim):
    """Extract per-year arrays for the trajectory comparison.

    v3 may store timevec as ss.date DateArray; convert to float years
    via .years per starsim-dev-time skill.
    """
    timevec = sim.results.timevec
    year_floats = np.asarray(timevec.years if hasattr(timevec, 'years') else timevec, dtype=float)
    return dict(
        year=year_floats.tolist(),
        new_cancers=list(sim.results['new_cancers']),
        hpv_total_infections=list(sim.results['hpv_total_infections']),
        new_vaccinated=list(sim.results.get('new_vaccinated', [0] * len(year_floats))),
    )


@pytest.mark.slow
def test_m05_routine_trajectory_parity():
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v2 baseline at {BASELINE_PATH}. '
            f'Run tests/regression/multi_seed_v2_vx.py from a v2 env.'
        )
    v2_rows = json.loads(BASELINE_PATH.read_text())
    if '_trajectory' not in v2_rows[0]:
        pytest.skip(
            'v2 baseline lacks `_trajectory` field. Regenerate it after the '
            'Task 12 multi_seed_v2_vx.py update.'
        )

    years = np.array(v2_rows[0]['_trajectory']['year'])
    v2_arrs = {
        m: np.array([r['_trajectory'][m] for r in v2_rows], dtype=float)
        for m in TRAJECTORY_METRICS
    }

    v3_rows = []
    for seed in range(N_V3_SEEDS):
        sim = build_v3_sim()
        sim.pars['rand_seed'] = int(seed)
        sim.run()
        v3_rows.append(_v3_trajectory_row(sim))

    v3_years = np.array(v3_rows[0]['year'])
    assert np.allclose(years, v3_years), (
        f'v3 and v2 year vectors differ: '
        f'v3[{v3_years[0]}..{v3_years[-1]}] vs v2[{years[0]}..{years[-1]}]'
    )

    v3_arrs = {
        m: np.array([r[m] for r in v3_rows], dtype=float)
        for m in TRAJECTORY_METRICS
    }

    failures = []
    for metric in TRAJECTORY_METRICS:
        v2_a = v2_arrs[metric]
        v3_a = v3_arrs[metric]
        v2_mean = v2_a.mean(axis=0)
        v2_se = v2_a.std(axis=0, ddof=1) / math.sqrt(v2_a.shape[0])
        v3_mean = v3_a.mean(axis=0)
        v3_se = v3_a.std(axis=0, ddof=1) / math.sqrt(v3_a.shape[0])
        denom = np.sqrt(v2_se ** 2 + v3_se ** 2)
        z = np.where(denom > 0, (v3_mean - v2_mean) / np.maximum(denom, 1e-12), 0.0)
        bad = np.abs(z) >= Z_THRESHOLD
        if bad.any():
            bad_idx = np.where(bad)[0]
            example_year = years[bad_idx[0]]
            failures.append(
                f'{metric}: {bad.sum()} year(s) with |z|>={Z_THRESHOLD}, '
                f'first at year {example_year:.0f} (|z|={abs(z[bad_idx[0]]):.2f})'
            )
    assert not failures, '\n'.join(['Trajectory cells outside |z|<3:'] + failures)