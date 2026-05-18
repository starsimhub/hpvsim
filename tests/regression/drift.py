"""Per-key drift computation against a reference summary.

Pure-function utility used by the M01/M02 pytest drift gates
(``tests/test_regression.py``). Compares a current run's summary dict
against a stored baseline and flags per-key relative drift above a
threshold. Zero-baseline rows are flagged with ``rel_diff = None``.
"""

THRESHOLD = 0.10  # 10% relative drift


def compute_drift(baseline_summary, current_summary, threshold=THRESHOLD):
    """Compute per-key drift records.

    Args:
        baseline_summary: dict of {key: number} — the stored reference baseline.
        current_summary:  dict of {key: number} — the current run's summary.
        threshold:        relative-drift threshold (default 0.10 = 10%).

    Returns:
        List of dicts with keys: key, baseline, current, abs_diff, rel_diff,
        over_threshold. Keys present in baseline but missing from current are
        skipped. If the baseline value is zero, rel_diff is None and
        over_threshold is True.
    """
    rows = []
    for k in baseline_summary.keys():
        if k not in current_summary:
            continue
        b = float(baseline_summary[k])
        c = float(current_summary[k])
        abs_diff = c - b
        if b == 0:
            rel_diff = None
            over = True
        else:
            rel_diff = abs_diff / b
            over = abs(rel_diff) > threshold
        rows.append({
            'key': k,
            'baseline': b,
            'current': c,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'over_threshold': over,
        })
    return rows