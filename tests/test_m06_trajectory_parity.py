"""M06 trajectory parity (screen-treat anchor only).

Uses a per-step BoolArr.raw snapshot to count False -> True transitions,
producing v2-equivalent per-year flow counters from per-intervention
state. Mirrors M05's _FirstVaxLogger.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import starsim as ss

from tests.regression import anchor_screen_treat as anchor


V2_BASELINE = Path(__file__).parent / 'regression' / 'v2_seeds_n30_screen_treat_traj.json'
N_V3_SEEDS = 10
Z_GATE = 3.0


class _FirstEventLogger(ss.Analyzer):
    """Snapshot intervention.<attr>.raw pre/post step; record False->True per year."""

    def __init__(self, intervention_name, attr_name, **kwargs):
        super().__init__(**kwargs)
        self.intervention_name = intervention_name
        self.attr_name = attr_name
        self._prev = None
        self.annual_count = {}  # year_int -> count

    def init_pre(self, sim):
        super().init_pre(sim)
        intv = sim.interventions[self.intervention_name]
        arr = getattr(intv, self.attr_name)
        self._prev = np.asarray(arr.raw).copy()

    def step(self):
        intv = self.sim.interventions[self.intervention_name]
        arr = getattr(intv, self.attr_name)
        cur = np.asarray(arr.raw)
        transitions = int((cur & ~self._prev).sum())
        year_int = int(self.sim.t.now('year'))
        self.annual_count[year_int] = self.annual_count.get(year_int, 0) + transitions
        self._prev = cur.copy()


def _v3_trajectories_one_seed(seed):
    sim = anchor.build_v3_sim()
    sim.pars['rand_seed'] = int(seed)
    sim.analyzers = [
        _FirstEventLogger('primary',     'screened',    name='log_screened'),
        _FirstEventLogger('excision_rx', 'cin_treated', name='log_cin_treated'),
    ]
    sim.run()
    screened_by_year = sim.analyzers['log_screened'].annual_count
    cin_treated_by_year = sim.analyzers['log_cin_treated'].annual_count
    # Annual-bucketed new_cancers from sim.results.hpvtotal
    res = sim.results.hpvtotal
    years_flt = np.asarray(sim.timevec)
    year_int = np.floor(years_flt).astype(int)
    new_cancers_per_step = np.asarray(res['new_cancers'])
    cancers_by_year = {}
    for y in np.unique(year_int):
        mask = year_int == y
        cancers_by_year[int(y)] = float(new_cancers_per_step[mask].sum())
    return dict(
        screened_by_year=screened_by_year,
        cin_treated_by_year=cin_treated_by_year,
        cancers_by_year=cancers_by_year,
    )


@pytest.mark.slow
def test_m06_screen_treat_trajectory_parity():
    if not V2_BASELINE.exists():
        pytest.skip(
            f'v2 trajectory baseline JSON not present at {V2_BASELINE} '
            '— regenerate locally via tests/regression/multi_seed_v2_screen_treat.py'
        )
    v2_rows = json.loads(V2_BASELINE.read_text())
    v3_rows = [_v3_trajectories_one_seed(seed) for seed in range(N_V3_SEEDS)]

    failures = []
    for metric in ('screened_by_year', 'cin_treated_by_year', 'cancers_by_year'):
        years = sorted(int(y) for y in v2_rows[0][metric].keys())
        for y in years:
            v2_vals = np.array([float(r[metric][str(y)]) for r in v2_rows])
            v3_vals = np.array([float(r[metric].get(y, 0.0)) for r in v3_rows])
            n2, n3 = len(v2_vals), len(v3_vals)
            var2, var3 = v2_vals.var(ddof=1), v3_vals.var(ddof=1)
            se = np.sqrt(var2 / n2 + var3 / n3)
            if se == 0:
                z = 0.0 if v2_vals.mean() == v3_vals.mean() else float('inf')
            else:
                z = (v3_vals.mean() - v2_vals.mean()) / se
            if abs(z) >= Z_GATE:
                failures.append((metric, y, z, v2_vals.mean(), v3_vals.mean()))

    if failures:
        msg = '\n'.join(
            f'  {m}@{y}: z={z:+.2f} (v2_mean={mv2:.4g}, v3_mean={mv3:.4g})'
            for m, y, z, mv2, mv3 in failures
        )
        pytest.fail(f'trajectory parity failed at |z| < {Z_GATE}:\n{msg}')
