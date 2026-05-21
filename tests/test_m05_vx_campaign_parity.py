"""M05 campaign-vx parity gate: multi-seed mean z-score vs v2 baseline.

See test_m05_vx_routine_parity.py for the gate description; this test is the campaign-anchor mirror.

Requires the locally-regenerated v2 baseline at
``tests/regression/v2_seeds_n30_vx_campaign.json``. See
``tests/regression/README_m05.md`` for how to (re)generate it.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import sciris as sc

from tests.regression.anchor_vx_campaign import build_v3_sim
from tests.regression.short_summary import build_summary


BASELINE_PATH = Path(__file__).parent / 'regression' / 'v2_seeds_n30_vx_campaign.json'
N_V3_SEEDS = 10
Z_THRESHOLD = 3.0
GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')

_SKIP_KEYS = frozenset({'_seed', '_total_pop', 'total population'})


def _run_v3_seeds(n, start_seed=0):
    summaries = []
    for seed in range(start_seed, start_seed + n):
        sim = build_v3_sim()
        sim.pars['rand_seed'] = int(seed)
        sim.run()
        row = build_summary(sim, GENOTYPES)
        # Pull vaccination scalars off the single intervention (deep-copied
        # by sim.init, so we read from sim.interventions[0] not the original)
        intv = sim.interventions[0]
        row['n_vaccinated_2060'] = int(intv.vaccinated.sum())
        row['n_doses_2060'] = int(intv.n_doses.sum())
        # cancer incidence 2030-2060 — same proxy as the v2 baseline
        years = sim.results.timevec
        # timevec may be a DateArray (ss.date objects). Convert to float years
        # so the (years >= 2030) comparison works (see starsim-dev-time).
        if hasattr(years, 'years'):
            year_floats = np.asarray(years.years, dtype=float)
        else:
            year_floats = np.asarray(years, dtype=float)
        mask = (year_floats >= 2030) & (year_floats < 2060)
        n_cancers = float(sim.results['new_cancers'][mask].sum())
        pop = sim.results['n_alive'][mask]
        dt = sim.t.dt.years if hasattr(sim.t.dt, 'years') else sim.t.dt
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
def test_m05_campaign_short_summary_parity():
    """40-metric z-score gate: |z| < 3 across M03 summary + 3 vx scalars."""
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v2 baseline at {BASELINE_PATH}. '
            f'Generate it via tests/regression/multi_seed_v2_vx.py from a '
            f'v2 hpvsim env (see tests/regression/README_m05.md).'
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
    assert not failures, '\n'.join(['Metrics outside |z|<3:'] + failures)