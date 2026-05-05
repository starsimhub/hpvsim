"""Regenerate the M02 v2.3 baseline for the HPV16 anchor scenario.

Run this script INSIDE a Python environment that has hpvsim==2.3.x installed
(e.g. the local hpvsim_v23_frozen clone, or a fresh `pip install hpvsim==2.3`
venv). The v3 active package is NOT used here — only the v2.3 hpvsim API.

The script:
  1. Builds the same 1-genotype HPV16 Nigeria 1990-2060 anchor that v3's
     tests/regression/anchor_hpv16.py runs.
  2. Calls v2's sim.compute_summary() to produce the 8-key short_summary.
  3. Writes the result to tests/regression_baselines/anchor_hpv16.json
     in the same JSON shape that the v3 drift gate expects.

Key v2-vs-v3 syntax differences honored here:
  - v3 PARS uses ``genotype='hpv16'``; v2 expects ``genotypes=['hpv16']``.
  - v3 PARS uses ``stop=2060``; v2 expects ``end=2060``.
  - dt=0.25 matches v2's default sim timestep (``_v2_legacy/parameters.py:61``)
    and v3's M02 anchor.

Usage (from a v2.3 env, with cwd at the repo root):

    python tests/regression/baseline_v23.py
    python tests/regression/baseline_v23.py --out custom/path/anchor.json
"""

import argparse
import json
import sys
from pathlib import Path

import sciris as sc
import hpvsim as hpv  # v2.3 here


# Pinned anchor PARS — must match tests/regression/anchor_hpv16.py:PARS
# at v3 except for v2's API name differences and pop_scale handling.
#
# pop_scale=1 + total_pop=10_000 disables v2's real-world scaling so count
# metrics (total infections, cancers, cancer deaths) come out at the
# agent-level — matching v3's default (where pop_scale defaults to 1.0
# unless total_pop is explicitly set). M01's gen_anchor_hpv16.py used the
# same convention.
PARS = dict(
    n_agents=10_000,
    location='nigeria',
    genotypes=['hpv16'],     # v2 takes a list
    start=1990,
    end=2060,                # v2 calls it 'end', not 'stop'
    dt=0.25,                 # matches v2's default
    rand_seed=0,
    verbose=0,
    pop_scale=1,             # disable real-world scaling (match v3 default)
    total_pop=10_000,        # match n_agents so pop_scale stays 1
    ms_agent_ratio=1,        # disable multiscale dynamic spawning (v2 default
                             # is 10; v3 M02 spec defers multiscale, so v2
                             # baseline must be regenerated with ms_agent_ratio=1
                             # for an apples-to-apples cancer-count comparison.
                             # See _v2_legacy/parameters.py:38 + people.py:280-371.)
)


_EXPECTED_KEYS = (
    'total HPV infections',
    'total cancers',
    'total cancer deaths',
    'mean HPV prevalence (%)',
    'mean cancer incidence (per 100k)',
    'mean age of infection (years)',
    'mean age of cancer (years)',
    'mean age of cancer death (years)',
)


def run_and_summarize():
    """Run the v2.3 anchor sim and return (short_summary_dict, total_pop)."""
    sim = hpv.Sim(sc.dcp(PARS))
    sim.run()

    # v2's compute_summary populates sim.short_summary with the 8 keys
    # listed above (see _v2_legacy/sim.py:1157 -> 1194).
    sim.compute_summary()
    s = dict(sim.short_summary)

    # Sanity-check we got the keys we expect.
    missing = [k for k in _EXPECTED_KEYS if k not in s]
    if missing:
        raise RuntimeError(f'v2 compute_summary missing keys: {missing}; '
                           f'got {list(s.keys())}')

    short = {k: float(s[k]) for k in _EXPECTED_KEYS}

    # Total population at end of sim.
    if 'n_alive' in sim.results:
        total_pop = float(sim.results['n_alive'][-1])
    else:
        total_pop = float(sim.results['pop_size'][-1])

    return short, total_pop


def build_baseline():
    short, total_pop = run_and_summarize()
    return {
        'metadata': {
            'hpvsim_version': hpv.__version__,
            'pars': dict(PARS),
        },
        'summary': {
            **short,
            'total population': total_pop,
        },
    }


DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent / 'regression_baselines' / 'anchor_hpv16.json'
)


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate M02 v2.3 anchor baseline.')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT,
                   help=f'Output JSON path (default: {DEFAULT_OUT}).')
    args = p.parse_args(argv)

    baseline = build_baseline()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f'Wrote v2.3 baseline ({hpv.__version__}) to {args.out}')
    print('Summary:')
    for k, v in baseline['summary'].items():
        print(f'  {k:<40} {v:>12.4g}')
    return 0


if __name__ == '__main__':
    sys.exit(main())