"""Generate a regression baseline from the currently-installed hpvsim.

Runs the anchor scenario and saves a JSON file containing the per-key
short_summary plus total population, with metadata (hpvsim version, anchor pars).

The output directory is gitignored; baseline files stay local-only.

Usage:
    python tests/regression/baseline.py
    python tests/regression/baseline.py --out path/to/baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

import hpvsim as hpv

# Allow running as `python tests/regression/baseline.py` from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from anchor import PARS, run_and_summarize  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'regression_baselines' / 'anchor.json'


def build_baseline():
    short, total_pop = run_and_summarize()
    return {
        'metadata': {
            'hpvsim_version': hpv.__version__,
            'pars': dict(PARS),  # JSON-serializable: ints, floats, str, list of mixed types
        },
        'summary': {
            **{k: float(v) for k, v in short.items()},
            'total population': total_pop,
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate v2 regression baseline.')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT,
                   help=f'Output JSON path (default: {DEFAULT_OUT}).')
    args = p.parse_args(argv)

    baseline = build_baseline()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f'Wrote baseline to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
