"""Compare ``v2_trace.csv`` and ``v3_trace.csv`` side-by-side.

Aligns rows on the integer ``ti`` column and prints per-step diffs for the
key reinfection / immunity columns. Useful for finding the timestep at which
v3 starts to drift away from v2.

Run after both traces have been produced:

    python tests/regression/trace_v3.py
    "<v2 env>/python.exe" tests/regression/trace_v2.py
    python tests/regression/compare_traces.py

Default behavior: prints summary for hpv16 at sample timesteps + a final-step
table for all 4 genotypes.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


DEFAULT_V2 = Path(__file__).resolve().parent / 'v2_trace.csv'
DEFAULT_V3 = Path(__file__).resolve().parent / 'v3_trace.csv'

GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


def load(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def to_arrays(rows, cols):
    """Return dict col -> ndarray of float (parsed)."""
    out = {}
    for c in cols:
        if c not in rows[0]:
            out[c] = None
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r[c]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[c] = np.array(vals)
    return out


def pct(v3, v2):
    if v2 == 0:
        return float('nan') if v3 != 0 else 0.0
    return 100.0 * (v3 - v2) / v2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--v2', type=Path, default=DEFAULT_V2)
    p.add_argument('--v3', type=Path, default=DEFAULT_V3)
    p.add_argument('--genotype', default='hpv16')
    p.add_argument('--years', nargs='*', type=int,
                   default=[1990, 1995, 2000, 2010, 2030, 2059],
                   help='Years to spotlight in the per-step diff.')
    args = p.parse_args(argv)

    v2_rows = load(args.v2)
    v3_rows = load(args.v3)

    print(f'v2 rows: {len(v2_rows)}; v3 rows: {len(v3_rows)}')
    print()

    # Index by integer ti.
    v2_by_ti = {int(r['ti']): r for r in v2_rows}
    v3_by_ti = {int(r['ti']): r for r in v3_rows}
    common_ti = sorted(set(v2_by_ti) & set(v3_by_ti))
    if not common_ti:
        print('No overlapping ti between v2 and v3.')
        return 1

    g = args.genotype

    # Columns to spotlight per-step.
    cols = [
        'n_alive', 'n_alive_f',
        f'{g}.n_inf', f'{g}.n_inf_f', f'{g}.n_sus_f', f'{g}.n_immune_f',
        f'{g}.mean_nab_f', f'{g}.mean_nab_f_pos',
        f'{g}.nab_p50', f'{g}.nab_p75', f'{g}.nab_p90', f'{g}.nab_p95',
        f'{g}.mean_rel_sus_f', f'{g}.mean_sev_imm_f',
        f'{g}.new_inf_this_step', f'{g}.first_inf_this_step',
        f'{g}.re_inf_this_step',
    ]

    # Per-year spotlight rows.
    print(f'=== Per-year diff for {g} ===')
    hdr = f'{"year":>6}  {"col":<28}  {"v2":>14}  {"v3":>14}  {"v3-v2":>14}  {"pct":>8}'
    print(hdr)
    print('-' * len(hdr))
    for yr in args.years:
        # Find row closest to year start.
        target_ti = None
        for r in v3_rows:
            if abs(float(r['year']) - yr) < 0.3:
                target_ti = int(r['ti'])
                break
        if target_ti is None or target_ti not in v2_by_ti:
            continue
        v2r = v2_by_ti[target_ti]
        v3r = v3_by_ti[target_ti]
        for c in cols:
            if c not in v2r or c not in v3r:
                continue
            try:
                v2v = float(v2r[c]); v3v = float(v3r[c])
            except (TypeError, ValueError):
                continue
            d = v3v - v2v
            pc = pct(v3v, v2v)
            print(f'{yr:>6}  {c:<28}  {v2v:>14.4g}  {v3v:>14.4g}  {d:>14.4g}  {pc:>7.1f}%')
        print()

    # Cumulative reinfections, all genotypes, final step.
    print('=== Cumulative new/first/re infections (all genotypes) ===')
    print(f'{"genotype":<8}  {"v2_new":>10}  {"v3_new":>10}  {"v2_first":>10}  '
          f'{"v3_first":>10}  {"v2_re":>10}  {"v3_re":>10}')
    for g_ in GENOTYPES:
        cnew = f'{g_}.new_inf_this_step'
        cfirst = f'{g_}.first_inf_this_step'
        cre = f'{g_}.re_inf_this_step'

        def _sum(rows, c):
            t = 0.0
            for r in rows:
                if c in r and r[c] != '':
                    t += float(r[c])
            return t

        print(f'{g_:<8}  {_sum(v2_rows, cnew):>10.0f}  {_sum(v3_rows, cnew):>10.0f}  '
              f'{_sum(v2_rows, cfirst):>10.0f}  {_sum(v3_rows, cfirst):>10.0f}  '
              f'{_sum(v2_rows, cre):>10.0f}  {_sum(v3_rows, cre):>10.0f}')

    # Cumulative diff at the final common ti.
    last_ti = common_ti[-1]
    v2r = v2_by_ti[last_ti]
    v3r = v3_by_ti[last_ti]
    print()
    print(f'=== Final ti={last_ti} (year≈{v3r["year"]}) snapshot, all genotypes ===')
    print(f'{"col":<35}  {"v2":>12}  {"v3":>12}  {"v3-v2":>12}  {"pct":>8}')
    final_cols = ['n_alive', 'n_alive_f']
    for g_ in GENOTYPES:
        final_cols += [
            f'{g_}.n_inf', f'{g_}.n_immune_f',
            f'{g_}.mean_nab_f', f'{g_}.mean_nab_f_pos',
            f'{g_}.mean_rel_sus_f', f'{g_}.mean_sev_imm_f',
        ]
    for c in final_cols:
        if c not in v2r or c not in v3r:
            continue
        try:
            v2v = float(v2r[c]); v3v = float(v3r[c])
        except (TypeError, ValueError):
            continue
        d = v3v - v2v
        pc = pct(v3v, v2v)
        print(f'{c:<35}  {v2v:>12.4g}  {v3v:>12.4g}  {d:>12.4g}  {pc:>7.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())