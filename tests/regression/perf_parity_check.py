"""Capture or compare the 4-genotype anchor short summary for one seed.

Used as a regression check around the Tier 1 perf fixes:

    # Before changes
    python tests/regression/perf_parity_check.py --capture before

    # After changes
    python tests/regression/perf_parity_check.py --compare before

A bit-exact change should print "MATCH" on every key. A behavior-changing
change (e.g. one fewer RNG jump) will show small numeric drift; print the
diff so we can eyeball whether it looks like ordinary stochastic noise.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from anchor_4genotype import run_and_summarize  # noqa: E402


def capture(label):
    t0 = time.time()
    short, total_pop = run_and_summarize()
    dt = time.time() - t0
    out = dict(short)
    out['_total_pop'] = float(total_pop)
    out['_wall_seconds'] = round(dt, 3)
    path = HERE / f'perf_parity_{label}.json'
    with path.open('w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f'[{label}] wall={dt:.2f}s  total_pop={total_pop:.0f}  -> {path}')
    return out


def compare(label):
    cur_path = HERE / f'perf_parity_{label}.json'
    new = capture('current')
    with cur_path.open() as f:
        old = json.load(f)
    print()
    print(f'=== Diff: {cur_path.name} (old) vs current (new) ===')
    keys = sorted(set(old) | set(new))
    n_match = 0
    n_diff = 0
    for k in keys:
        if k.startswith('_wall'):
            continue
        a = old.get(k)
        b = new.get(k)
        if a == b:
            n_match += 1
            continue
        # Numeric? show abs/rel delta
        try:
            af = float(a); bf = float(b)
            if math.isclose(af, bf, rel_tol=0, abs_tol=0):
                n_match += 1
                continue
            abs_d = bf - af
            rel_d = abs_d / af if af else float('nan')
            print(f'  {k:<48} old={af:>12.4g}  new={bf:>12.4g}  delta={abs_d:+.4g} ({rel_d:+.2%})')
        except Exception:
            print(f'  {k:<48} old={a!r}  new={b!r}')
        n_diff += 1
    old_wall = old.get('_wall_seconds')
    new_wall = new.get('_wall_seconds')
    print()
    print(f'  matched: {n_match}  differing: {n_diff}')
    if old_wall is not None and new_wall is not None:
        speed = (old_wall - new_wall) / old_wall * 100
        print(f'  wall: old={old_wall:.2f}s  new={new_wall:.2f}s  delta={new_wall-old_wall:+.2f}s ({speed:+.1f}%)')
    return n_diff == 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['capture', 'compare'])
    p.add_argument('label', help='label of the snapshot file (e.g. before, after_imm)')
    args = p.parse_args(argv)
    if args.mode == 'capture':
        capture(args.label)
    else:
        ok = compare(args.label)
        return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main() or 0)