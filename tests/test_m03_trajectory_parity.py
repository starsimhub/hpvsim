"""M03 capability gate: multi-seed mean-trajectory parity vs. v2 baseline.

For each of ``cum_infections`` and ``cum_cancers``, compares the v3 mean
trajectory across ``N_V3_SEEDS`` runs to the v2 mean trajectory across
N v2 seeds. Dual gate per timestep:

    z[t]   = (v3_mean[t] - v2_mean[t]) / sqrt(v2_SE[t]^2 + v3_SE[t]^2)
    rel[t] = |v3_mean[t] - v2_mean[t]| / |v2_mean[t]|

A timestep fails only when ``|z[t]| > Z_THRESHOLD`` AND ``rel[t] > REL_THRESHOLD``.
The dual gate avoids two failure modes of a pure z gate: with n=30/n=10 seeds
the SE on the mean is so tight that any tiny systematic shift (e.g. ~3% on
end-of-sim cum infections, attributable to v2/v3 per-step transmission
algorithm differences) trips |z|>5, even though the absolute drift is well
inside design tolerance. The rel-threshold mirrors the short-summary's 10%
gate, so this trajectory test fails only on drifts that would also fail the
scalar gate.

Skipped if the v2 multi-seed trajectory baseline is missing; regenerate via
``multi_seed_v2_trajectory.py --n 30`` from a v2 hpvsim env.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv

from tests.regression.anchor_4genotype import PARS


BASELINE_PATH = Path('tests/regression/v2_trajectories_n30.json')
N_V3_SEEDS = 10
Z_THRESHOLD = 5.0
REL_THRESHOLD = 0.10
# Per-step comparisons are unstable when cumulative counts are near zero
# (early sim, cancers especially). Mask out timesteps where v2_mean is below
# this floor.
NONTRIVIAL_FLOOR = 100.0
GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')
SERIES = ('cum_infections', 'cum_cancers')


def _run_v3_seeds(n, start_seed=0):
    """Return dict series_name -> ndarray of shape (n, T_annual).

    v3's ``hpvtotal.cum_*`` are per-step (dt=0.25 → 281 entries for a
    70-year sim). v2's ``infections_by_genotype`` is annual (71 entries),
    where index ``i`` is the cumulative count THROUGH the end of year
    (start+i). To match, downsample v3 with ``full[stride::stride]`` so
    index 0 lands on the first end-of-year boundary (step ``stride``, time
    start+1). Length: 70 — one entry per completed year. v2's array
    extends one year past v3 because v2's ``end=2060`` simulates through
    the end of 2060 while v3's ``stop=2060`` ends at the start of 2060;
    the test trims v2 to v3's length before comparison.
    """
    per_series = {s: [] for s in SERIES}
    for seed in range(start_seed, start_seed + n):
        pars = sc.dcp(PARS)
        pars['rand_seed'] = int(seed)
        sim = hpv.Sim(**pars)
        sim.run()
        dt = float(sim.t.dt)
        stride = max(1, int(round(1.0 / dt)))
        for s in SERIES:
            full = np.asarray(sim.results.hpvtotal[s], dtype=float)
            per_series[s].append(full[stride::stride])
    return {s: np.stack(per_series[s], axis=0) for s in SERIES}


def _mean_se(arr):
    """Mean and SE-of-the-mean along axis 0 (seeds). ``arr`` shape (n, T)."""
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, se


@pytest.mark.slow
def test_trajectory_parity_4genotype():
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v2 multi-seed trajectory baseline at {BASELINE_PATH}. '
            f'Regenerate via '
            f'`python tests/regression/multi_seed_v2_trajectory.py --n 30 --out {BASELINE_PATH}` '
            f'from a v2 hpvsim env.'
        )
    payload = json.loads(BASELINE_PATH.read_text())
    v2_arrays = {s: np.asarray(payload['series'][s], dtype=float) for s in SERIES}

    v3_arrays = _run_v3_seeds(N_V3_SEEDS, start_seed=0)

    failures = []
    for s in SERIES:
        v2_arr = v2_arrays[s]
        v3_arr = v3_arrays[s]
        # v2 extends one year past v3 (see _run_v3_seeds docstring). Trim v2.
        common_T = min(v2_arr.shape[1], v3_arr.shape[1])
        v2_arr = v2_arr[:, :common_T]
        v3_arr = v3_arr[:, :common_T]
        if v3_arr.shape[1] == 0:
            failures.append((s, 'empty trajectory after trim'))
            continue
        v2_mean, v2_se = _mean_se(v2_arr)
        v3_mean, v3_se = _mean_se(v3_arr)
        se_combo = np.sqrt(v2_se ** 2 + v3_se ** 2)
        valid = (np.abs(v2_mean) >= NONTRIVIAL_FLOOR) & (se_combo > 0)
        if not valid.any():
            failures.append((s, f'no timesteps with v2_mean >= {NONTRIVIAL_FLOOR}'))
            continue
        z = (v3_mean[valid] - v2_mean[valid]) / se_combo[valid]
        rel = np.abs(v3_mean[valid] - v2_mean[valid]) / np.abs(v2_mean[valid])
        bad = (np.abs(z) > Z_THRESHOLD) & (rel > REL_THRESHOLD)
        if bad.any():
            # Report the worst offender among the failing timesteps.
            worst_within_bad = int(np.argmax(np.abs(z) * bad))
            t_idx = int(np.where(valid)[0][worst_within_bad])
            failures.append((
                s,
                f'|z|={float(np.abs(z[worst_within_bad])):.2f} and rel='
                f'{float(rel[worst_within_bad]):.2%} at t_idx={t_idx} '
                f'(v2_mean={v2_mean[t_idx]:.1f} +- {v2_se[t_idx]:.1f}, '
                f'v3_mean={v3_mean[t_idx]:.1f} +- {v3_se[t_idx]:.1f})'
            ))

    if failures:
        msg = '\n'.join(f'  {s}: {info}' for s, info in failures)
        n_v2 = v2_arrays[SERIES[0]].shape[0]
        pytest.fail(
            f'M03 trajectory mean drift exceeds |z|>{Z_THRESHOLD} AND '
            f'rel>{REL_THRESHOLD:.0%} (v2 n={n_v2}, v3 n={N_V3_SEEDS}):\n{msg}'
        )