"""Shared parity-gate helper for multi-seed v3-vs-v2 metric comparison.

The z-score formula is

    z = (v3_mean - v2_mean) / sqrt(v2_SE^2 + v3_SE^2)

where SE is the standard error of the mean across seeds (std with ddof=1
divided by sqrt(n)). A metric fails when |z| >= z_threshold.

Per-seed rows are dicts of {metric_name: value}. Non-metric bookkeeping
keys (e.g. _seed, _total_pop) should be passed via skip_keys.
"""
import math

import numpy as np


def _mean_se(rows, key):
    vals = np.array([float(r[key]) for r in rows if key in r], dtype=float)
    if vals.size == 0:
        return None
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, se


def parity_gate(v3_seeds, v2_seeds, z_threshold=3.0, skip_keys=frozenset()):
    """Return [(metric_name, z)] for metrics exceeding |z| >= z_threshold.

    Args:
        v3_seeds: list of per-seed summary dicts from the v3 run.
        v2_seeds: list of per-seed summary dicts from the v2 baseline.
        z_threshold: failure threshold; metrics with |z| >= threshold fail.
        skip_keys: metric names to ignore (bookkeeping fields).

    Two degenerate-distribution policies:
      - both v2 and v3 have zero spread AND exactly equal means → pass.
      - both have zero spread AND unequal means → fail with z=inf.
    """
    metric_keys = sorted((set(v2_seeds[0]) & set(v3_seeds[0])) - skip_keys)
    failures = []
    for key in metric_keys:
        v2_stats = _mean_se(v2_seeds, key)
        v3_stats = _mean_se(v3_seeds, key)
        if v2_stats is None or v3_stats is None:
            continue
        v2_mean, v2_se = v2_stats
        v3_mean, v3_se = v3_stats
        se_combo = math.sqrt(v2_se ** 2 + v3_se ** 2)
        if se_combo == 0:
            if v2_mean != v3_mean:
                failures.append((key, float('inf')))
            continue
        z = (v3_mean - v2_mean) / se_combo
        if abs(z) >= z_threshold:
            failures.append((key, z))
    return failures
