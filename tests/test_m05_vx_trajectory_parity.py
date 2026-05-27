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
import types
from pathlib import Path

import numpy as np
import pytest
import sciris as sc
import starsim as ss

from tests.regression.anchor_vx_routine import build_v3_sim


class _FirstVaxLogger(ss.Analyzer):
    """Per-step counter for `vaccinated` False→True transitions.

    v3's ``intv.ti_vaccinated`` is rewritten on every administer call
    (including re-doses) AND ``np.asarray(...)`` on the underlying FloatArr
    returns the alive-masked view at access time. Naively histogramming
    ``ti_vaccinated`` therefore (a) loses vaxed agents who later die, and
    (b) shifts re-dose recipients' stamps forward to later years. Both
    distort the per-year ``new_vaccinated`` series in opposite directions.

    v2's ``results['new_vaccinated']`` is a per-step flow counter — set
    once at the first-vax event, never decremented, indifferent to
    re-doses. To match: snapshot ``intv.vaccinated.raw`` pre- and
    post-step, count the actual False→True transitions per step.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_first_vax_per_step = None

    def step(self):
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        intv = None
        for iv in sim.interventions():
            if hasattr(iv, 'vaccinated') and isinstance(iv.vaccinated, ss.BoolArr):
                intv = iv
                break
        if intv is None:
            return
        n_steps = int(sim.t.npts)
        self.n_first_vax_per_step = np.zeros(n_steps, dtype=int)
        original_step = intv.step
        logger = self

        def wrapped_step(self):
            pre = self.vaccinated.raw.copy()
            result = original_step()
            post = self.vaccinated.raw
            ti = int(self.sim.t.ti)
            logger.n_first_vax_per_step[ti] = int(((~pre) & post).sum())
            return result

        intv.step = types.MethodType(wrapped_step, intv)


BASELINE_PATH = Path(__file__).parent / 'regression' / 'v2_seeds_n30_vx_routine.json'
N_V3_SEEDS = 10
Z_THRESHOLD = 3.0

TRAJECTORY_METRICS = ('new_cancers', 'hpv_total_infections', 'new_vaccinated')


def _v3_trajectory_row(sim, intv, first_vax_logger):
    """Per-year arrays for trajectory comparison.

    v3 emits per-step (quarterly) results, v2 emits annual. We downsample v3's
    quarterly per-step counters to annual SUMS so the cadence matches v2's
    trajectory entries.

    new_cancers / hpv_total_infections come from sim.results.hpvtotal;
    new_vaccinated comes from the _FirstVaxLogger analyzer (which records
    first-vax-flow transitions, matching v2's `results['new_vaccinated']`
    semantic).
    """
    timevec = sim.results.timevec
    year_floats = np.asarray(timevec.years if hasattr(timevec, 'years') else timevec, dtype=float)

    # Downsample quarterly per-step counters to annual sums. v3 uses dt=0.25
    # (281 steps for 70-year sim); v2 stores 71 annual entries (70 years + 1
    # for the boundary). Bucket steps by integer floor(year).
    int_years = np.floor(year_floats).astype(int)
    new_cancers_q = np.asarray(sim.results.hpvtotal.new_cancers, dtype=float)
    new_infections_q = np.asarray(sim.results.hpvtotal.new_infections, dtype=float)
    new_vacc_q = (first_vax_logger.n_first_vax_per_step
                  if first_vax_logger.n_first_vax_per_step is not None
                  else np.zeros(len(year_floats), dtype=int)).astype(float)

    annual_years = sorted(np.unique(int_years).tolist())
    annual_new_cancers = []
    annual_new_infections = []
    annual_new_vacc = []
    for y in annual_years:
        bucket = int_years == y
        annual_new_cancers.append(float(new_cancers_q[bucket].sum()))
        annual_new_infections.append(float(new_infections_q[bucket].sum()))
        annual_new_vacc.append(float(new_vacc_q[bucket].sum()))

    return dict(
        year=[float(y) for y in annual_years],
        new_cancers=annual_new_cancers,
        hpv_total_infections=annual_new_infections,
        new_vaccinated=annual_new_vacc,
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
        first_vax_logger = _FirstVaxLogger()
        existing = list(sim.pars.get('analyzers', []) or [])
        sim.pars['analyzers'] = existing + [first_vax_logger]
        sim.run()
        intv = sim.interventions[0]
        v3_rows.append(_v3_trajectory_row(sim, intv, first_vax_logger))

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