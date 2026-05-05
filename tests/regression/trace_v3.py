"""v3 per-step trace + side-by-side comparison vs v2.

Captures per-step state and per-call compute_severity statistics, writes
v3_trace.csv to the project root, and prints aggregate stats. Pair with
v23_frozen/trace_v2.py (which writes v2_trace.csv) — run both, then
compare.

Run:
    python tests/regression/trace_v3.py [--compare]

If --compare is passed, loads both v2_trace.csv and v3_trace.csv and
prints a per-year side-by-side table.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Add project root so the relative import below works regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import hpvsim as hpv
import hpvsim.hpv as hpv_mod
from tests.regression.anchor_hpv16 import PARS


def _wrap_compute_severity():
    """Monkey-patch _compute_severity to log per-call stats. Returns the log
    list and the original function for restoration.
    """
    log = []
    orig = hpv_mod._compute_severity

    def traced(t, rel_sev=None, pars=None):
        out = orig(t, rel_sev=rel_sev, pars=pars)
        is_cancer = pars is not None and pars.get('method') == 'cin_integral'
        log.append({
            'kind': 'cancer' if is_cancer else 'cin',
            'n': len(t) if hasattr(t, '__len__') else 1,
            'mean_t': float(np.mean(t)),
            'mean_rs': float(np.mean(rel_sev)) if rel_sev is not None else 1.0,
            'mean_p': float(np.mean(out)),
        })
        return out

    hpv_mod._compute_severity = traced
    return log, orig


def _aggregate(calls, key):
    """Length-weighted mean of `key` across `calls`."""
    n = sum(c['n'] for c in calls)
    if n == 0:
        return 0.0
    return sum(c[key] * c['n'] for c in calls) / n


def run_v3_trace(out_path='v3_trace.csv'):
    """Run the M02 anchor sim and write per-year metrics to CSV."""
    log, orig_cs = _wrap_compute_severity()
    try:
        sim = hpv.Sim(**PARS)
        sim.run()
    finally:
        hpv_mod._compute_severity = orig_cs

    res = sim.results.hpv16
    n_quarterly = len(np.asarray(res.prevalence))
    # Map each annual sample year to the closest quarterly index. With
    # quarterly results spanning years [1990, 2060], idx 0 -> 1990 and the
    # last idx -> 2060. Use linspace to get fractional positions.
    qi_per_year = np.linspace(0, n_quarterly - 1, 71).astype(int)

    rows = []
    for i, yr in enumerate(np.linspace(1990, 2060, 71)):
        qi = qi_per_year[i]
        rows.append({
            'year': float(yr),
            'n_alive': float(sim.results['n_alive'][qi]),
            'n_infectious': float(np.asarray(res.n_infected)[qi]),
            'cum_infections': float(np.asarray(res.cum_infections)[qi]) if 'cum_infections' in res else 0.0,
            'n_cin': float(np.asarray(res.n_cin)[qi]),
            'n_cancerous': float(np.asarray(res.n_cancerous)[qi]),
            'hpv_prev': float(np.asarray(res.prevalence)[qi]),
        })

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    cin_calls = [c for c in log if c['kind'] == 'cin']
    cancer_calls = [c for c in log if c['kind'] == 'cancer']
    summary = dict(
        n_cin_calls=len(cin_calls),
        cin_total_n=sum(c['n'] for c in cin_calls),
        cin_mean_t=_aggregate(cin_calls, 'mean_t'),
        cin_mean_rel_sev=_aggregate(cin_calls, 'mean_rs'),
        cin_mean_prob=_aggregate(cin_calls, 'mean_p'),
        n_cancer_calls=len(cancer_calls),
        cancer_total_n=sum(c['n'] for c in cancer_calls),
        cancer_mean_t=_aggregate(cancer_calls, 'mean_t'),
        cancer_mean_rel_sev=_aggregate(cancer_calls, 'mean_rs'),
        cancer_mean_prob=_aggregate(cancer_calls, 'mean_p'),
    )
    print(f'Wrote {out_path} with {len(rows)} annual rows')
    print('\nv3 compute_severity aggregates:')
    print(f'  cin: {summary["n_cin_calls"]} calls, total n={summary["cin_total_n"]}')
    print(f'    weighted mean t={summary["cin_mean_t"]:.3f}y, rel_sev={summary["cin_mean_rel_sev"]:.3f}, '
          f'cin_prob={summary["cin_mean_prob"]*100:.2f}%')
    print(f'  cancer: {summary["n_cancer_calls"]} calls, total n={summary["cancer_total_n"]}')
    print(f'    weighted mean t={summary["cancer_mean_t"]:.3f}y, rel_sev={summary["cancer_mean_rel_sev"]:.3f}, '
          f'cancer_prob={summary["cancer_mean_prob"]*100:.4f}%')
    return rows, summary


def compare_traces(v2_path='v2_trace.csv', v3_path='v3_trace.csv'):
    """Load both CSVs and print per-year side-by-side metrics."""
    def load(p):
        with open(p) as f:
            return list(csv.DictReader(f))
    v2 = load(v2_path)
    v3 = load(v3_path)

    keys = ['n_alive', 'n_infectious', 'hpv_prev']
    print(f'{"year":>6}', end='')
    for k in keys:
        print(f'  {"v2_"+k:>14} {"v3_"+k:>14} {"rel":>8}', end='')
    print()
    indices = [0, 5, 10, 20, 30, 40, 50, 60, 70]
    for i in indices:
        if i >= min(len(v2), len(v3)):
            continue
        yr = float(v2[i]['year'])
        print(f'{yr:>6.0f}', end='')
        for k in keys:
            v2v = float(v2[i][k])
            v3v = float(v3[i][k])
            rel = (v3v - v2v) / max(abs(v2v), 1e-9) * 100
            fmt = '14.3f' if 'prev' in k else '14.0f'
            print(f'  {v2v:{fmt}} {v3v:{fmt}} {rel:+7.1f}%', end='')
        print()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='v3_trace.csv')
    p.add_argument('--compare', action='store_true',
                   help='After tracing, load v2_trace.csv (must exist) '
                        'and print side-by-side comparison')
    args = p.parse_args()
    run_v3_trace(args.out)
    if args.compare:
        v2_path = 'v2_trace.csv'
        if not Path(v2_path).exists():
            print(f'\nNo {v2_path} found; skipping comparison.')
        else:
            print()
            compare_traces(v2_path, args.out)