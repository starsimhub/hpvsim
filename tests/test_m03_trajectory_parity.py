"""M03 capability gate: age-aggregated cancer / infection trajectory parity.

This is the M03 release gate. The threshold is pinned after the first run by
inspecting the empirical drift; the placeholder constants below should be
edited to the chosen pin during execution of this task.
"""
import os
import numpy as np
import pytest
import sciris as sc

from tests.regression.anchor_4genotype import make_sim


TRAJECTORY_BASELINE = 'tests/regression_baselines/anchor_4genotype_trajectory.json'

# THRESHOLD PIN: After Task 16 first run, replace these with empirically chosen
# bounds. Until pinned, the test skips loud.
THRESHOLD_MAX_REL = None  # e.g., 0.15 = 15% max-relative-drift tolerance
THRESHOLD_L2_REL = None   # e.g., 0.10 = 10% L2 norm tolerance


def _l2_rel(a, b):
    """Relative L2 distance: ||a-b||_2 / ||b||_2 (denominator = baseline)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = float(np.linalg.norm(a - b))
    denom = float(np.linalg.norm(b)) or 1e-9
    return num / denom


@pytest.mark.slow
def test_trajectory_parity_4genotype():
    if THRESHOLD_MAX_REL is None or THRESHOLD_L2_REL is None:
        pytest.skip(
            'Trajectory parity threshold not yet pinned. After first run, '
            'set THRESHOLD_MAX_REL and THRESHOLD_L2_REL in this file.'
        )
    if not os.path.exists(TRAJECTORY_BASELINE):
        pytest.skip(f'Trajectory baseline missing at {TRAJECTORY_BASELINE}.')
    v2 = sc.loadjson(TRAJECTORY_BASELINE)
    sim = make_sim()
    sim.run()
    failures = []
    for series in ('cum_cancers_any', 'cum_infections_any'):
        v3_arr = np.asarray(sim.results.aggregate[series])
        v2_arr = np.asarray(v2[series])
        if v3_arr.shape != v2_arr.shape:
            failures.append((series, f'shape {v3_arr.shape} != {v2_arr.shape}'))
            continue
        max_rel = float(np.max(np.abs(v3_arr - v2_arr) /
                                np.maximum(np.abs(v2_arr), 1e-9)))
        l2_rel = _l2_rel(v3_arr, v2_arr)
        if max_rel > THRESHOLD_MAX_REL:
            failures.append((series, f'max_rel={max_rel:.3f} > {THRESHOLD_MAX_REL:.3f}'))
        if l2_rel > THRESHOLD_L2_REL:
            failures.append((series, f'l2_rel={l2_rel:.3f} > {THRESHOLD_L2_REL:.3f}'))
    if failures:
        msg = '\n'.join(f'  {s}: {f}' for s, f in failures)
        pytest.fail(f'M03 trajectory drift exceeded thresholds:\n{msg}')