"""Anchor scenario for the v2 -> v3 migration regression harness.

Vanilla 4-genotype HPV sim, Nigeria, fixed seed, no interventions, no analyzers.
This module is the scientific definition of the M0 anchor; tooling under
tests/regression/ (baseline.py, compare.py) imports run_and_summarize() and
make_sim() from here.

Run as a script to print the summary:
    python tests/regression/anchor.py
"""

import sciris as sc
import hpvsim as hpv

# Pinned anchor pars. Do not change without coordinating with regression baselines.
PARS = dict(
    n_agents   = 10e3,
    location   = 'nigeria',
    genotypes  = [16, 18, 'hi5', 'ohr'],
    start      = 1990,
    end        = 2060,
    dt         = 0.5,
    burnin     = 20,
    rand_seed  = 0,
    verbose    = 0,
)


def make_sim():
    """Build (but do not run) the anchor sim."""
    return hpv.Sim(sc.dcp(PARS))


def run_and_summarize():
    """Run the anchor sim and return (short_summary_dict, total_population_float)."""
    sim = make_sim()
    sim.run()
    short = dict(sim.short_summary)
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop


if __name__ == '__main__':
    short, total_pop = run_and_summarize()
    print('Short summary:')
    for k, v in short.items():
        print(f'  {k:<40} {v:>12.4g}')
    print(f'  {"total population":<40} {total_pop:>12.4g}')
