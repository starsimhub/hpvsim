"""Compare a current run of the anchor scenario against a stored v2 baseline.

Prints a per-summary-result drift table. The development gate is informational:
exit code is always 0 regardless of threshold breaches. If no baseline file is
present, prints a notice and exits clean without running the anchor (used by CI;
the anchor-runs check is the pytest smoke test's job).

Usage:
    python tests/regression/compare.py
    python tests/regression/compare.py --baseline path/to/file.json
    python tests/regression/compare.py --threshold 0.05
"""

import argparse
import json
import sys
from pathlib import Path

THRESHOLD = 0.10  # 10% relative drift

# Default baseline location: tests/regression_baselines/anchor.json (gitignored).
DEFAULT_BASELINE = Path(__file__).resolve().parent.parent / 'regression_baselines' / 'anchor.json'


def compute_drift(baseline_summary, current_summary, threshold=THRESHOLD):
    """Compute per-key drift records.

    Args:
        baseline_summary: dict of {key: number} — the stored v2 baseline.
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


def format_table(rows, threshold=THRESHOLD):
    """Format drift rows as a printable table (str)."""
    out = []
    out.append(f'{"key":<40} {"baseline":>12} {"current":>12} {"abs_diff":>12} {"rel_diff":>10} {"over":>6}')
    out.append('-' * 97)
    for r in rows:
        rel = f'{r["rel_diff"]*100:+.2f}%' if r['rel_diff'] is not None else 'n/a'
        flag = 'YES' if r['over_threshold'] else ''
        out.append(
            f'{r["key"]:<40} {r["baseline"]:>12.4g} {r["current"]:>12.4g} '
            f'{r["abs_diff"]:>12.4g} {rel:>10} {flag:>6}'
        )
    n_over = sum(1 for r in rows if r['over_threshold'])
    out.append('')
    out.append(
        f'{n_over}/{len(rows)} keys exceed +/- {threshold*100:.0f}% relative drift '
        f'threshold (informational; exit 0 regardless).'
    )
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description='Compare anchor run against v2 baseline.')
    p.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE,
                   help=f'Baseline JSON path (default: {DEFAULT_BASELINE})')
    p.add_argument('--threshold', type=float, default=THRESHOLD,
                   help='Relative drift threshold (default 0.10).')
    args = p.parse_args(argv)

    if not args.baseline.exists():
        print(f'No baseline at {args.baseline}; skipping diff.')
        print('To generate a baseline, run: python tests/regression/baseline.py')
        return 0

    # Local import: hpvsim is expensive to load. Skip it entirely on the fast
    # no-baseline path, which CI uses as a smoke check.
    _regression_dir = str(Path(__file__).parent)
    if _regression_dir not in sys.path:
        sys.path.insert(0, _regression_dir)
    from anchor import run_and_summarize  # noqa: E402

    with open(args.baseline) as f:
        baseline = json.load(f)
    baseline_summary = baseline['summary']

    short, total_pop = run_and_summarize()
    current_summary = {**{k: float(v) for k, v in short.items()},
                       'total population': total_pop}

    rows = compute_drift(baseline_summary, current_summary, threshold=args.threshold)
    print(format_table(rows, threshold=args.threshold))
    return 0


if __name__ == '__main__':
    sys.exit(main())
