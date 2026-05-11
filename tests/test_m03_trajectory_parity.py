"""M03 capability gate: multi-seed mean-trajectory parity vs. v2 baseline.

For each of ``cum_infections_any`` and ``cum_cancers_any``, compares the v3
mean trajectory across ``N_V3_SEEDS`` runs to the v2 mean trajectory across
N v2 seeds. Per-step z-score:

    z[t] = (v3_mean[t] - v2_mean[t]) / sqrt(v2_SE[t]^2 + v3_SE[t]^2)

Fails any series where ``max_t |z[t]| > Z_THRESHOLD`` on the non-trivial
timesteps (where cumulative count > ``NONTRIVIAL_FLOOR``).

The threshold is looser than the short-summary parity gate (|z|<3) because
this test does ~70 correlated per-year comparisons per series (v3 downsampled
to annual cadence to match v2's ``infections_by_genotype`` cadence), vs. 40
scalar metrics for the short-summary. |z|<5 stays comfortably above the
empirical drift on aggregate metrics (CV ~5% on counts; mean drift <1σ on
all summary metrics in the seed sweep).

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
# Per-step comparisons are unstable when cumulative counts are near zero
# (early sim). Mask out timesteps where v2_mean is below this floor.
NONTRIVIAL_FLOOR = 100.0
GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')
SERIES = ('cum_infections_any', 'cum_cancers_any')


def _run_v3_seeds(n, start_seed=0):
    """Return dict series_name -> ndarray of shape (n, T_annual).

    v3's ``aggregate.cum_*_any`` are per-step (dt=0.25 → ~281 entries for a
    70-year run). v2's ``infections_by_genotype`` is annual (71 entries).
    We downsample v3 to annual end-of-year by slicing every 4th step
    starting at index 0 (which lands on year boundaries: 1990, 1991, ...).
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
            full = np.asarray(sim.results.aggregate[s], dtype=float)
            per_series[s].append(full[::stride])
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
        if v3_arr.shape[1] != v2_arr.shape[1]:
            failures.append(
                (s, f'trajectory length mismatch: v3 T={v3_arr.shape[1]} vs v2 T={v2_arr.shape[1]}')
            )
            continue
        v2_mean, v2_se = _mean_se(v2_arr)
        v3_mean, v3_se = _mean_se(v3_arr)
        se_combo = np.sqrt(v2_se ** 2 + v3_se ** 2)
        valid = (np.abs(v2_mean) >= NONTRIVIAL_FLOOR) & (se_combo > 0)
        if not valid.any():
            failures.append((s, f'no timesteps with v2_mean >= {NONTRIVIAL_FLOOR}'))
            continue
        z = (v3_mean[valid] - v2_mean[valid]) / se_combo[valid]
        i_worst = int(np.argmax(np.abs(z)))
        max_abs_z = float(np.abs(z[i_worst]))
        if max_abs_z > Z_THRESHOLD:
            t_idx = int(np.where(valid)[0][i_worst])
            failures.append((
                s,
                f'max |z|={max_abs_z:.2f} at t_idx={t_idx} '
                f'(v2_mean={v2_mean[t_idx]:.1f} +- {v2_se[t_idx]:.1f}, '
                f'v3_mean={v3_mean[t_idx]:.1f} +- {v3_se[t_idx]:.1f})'
            ))

    if failures:
        msg = '\n'.join(f'  {s}: {info}' for s, info in failures)
        n_v2 = v2_arrays[SERIES[0]].shape[0]
        pytest.fail(
            f'M03 trajectory mean drift exceeds |z|>{Z_THRESHOLD} '
            f'(v2 n={n_v2}, v3 n={N_V3_SEEDS}):\n{msg}'
        )