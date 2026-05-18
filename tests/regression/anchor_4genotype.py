"""M03 anchor scenario for the v2 -> v3 migration regression harness.

4-genotype HPV sim, Nigeria, fixed seed, no interventions, no analyzers
(beyond the auto-added HPVTotal analyzer that pools per-genotype results).
Tooling under tests/regression/ (compare.py, baseline_v23.py) imports
``run_and_summarize()`` from here.

Run as a script to print the summary:
    python tests/regression/anchor_4genotype.py
"""

import sys
from pathlib import Path

import sciris as sc

import hpvsim as hpv

# Make sibling short_summary.py importable when this script is invoked
# directly (python tests/regression/anchor_4genotype.py).
# pytest adds the regression package to sys.path so the insert is harmless there.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary  # noqa: E402


# Pinned anchor pars. Do not change without coordinating with regression baselines.
# dt=0.25 matches v2's default sim timestep (declared at _v2_legacy/parameters.py:61).
PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=[16, 18, 'hi5', 'ohr'],
    start=1990,
    stop=2060,
    dt=0.25,
    rand_seed=0,
    verbose=0,
)


def make_sim():
    """Build (but do not run) the M03 anchor sim."""
    return hpv.Sim(**sc.dcp(PARS))


def run_and_summarize():
    """Run the M03 anchor sim and return (short_summary_dict, total_pop).

    Summary is the 40-entry dict from short_summary.build_summary:
    8 metrics x 4 genotypes (32 entries) plus 8 aggregate-across-genotypes
    metrics computed from the HPVTotal analyzer's cum_infections /
    cum_cancers / new_cancer_deaths results.
    """
    sim = make_sim()
    sim.run()
    short = build_summary(sim, genotypes=('hpv16', 'hpv18', 'hi5', 'ohr'))
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop


if __name__ == '__main__':
    short, total_pop = run_and_summarize()
    print('Short summary (M03 4-genotype):')
    for k, v in short.items():
        print(f'  {k:<48} {v:>12.4g}')
    print(f'  {"total population":<48} {total_pop:>12.4g}')