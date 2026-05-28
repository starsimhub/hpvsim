"""M02 anchor scenario: single-genotype HPV16 with full natural history.

Full HPV16 progression (precin/CIN/cancer + cancer death). Nigeria,
1990-2060 to cover the full lifetime cancer signal. This is the M02
acceptance-test target — single-genotype, full natural history.

Pinned anchor pars. Do not change without coordinating with regression baselines.
"""
import sys
from pathlib import Path

import sciris as sc

import hpvsim as hpv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary  # noqa: E402


PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=[16],
    start=1990,
    stop=2060,
    dt=0.25,
    rand_seed=0,
    verbose=0,
)

GENOTYPES = ('hpv16',)


def make_sim():
    """Build (but do not run) the M02 anchor sim."""
    return hpv.Sim(**sc.dcp(PARS))


def run_and_summarize():
    """Run the M02 anchor sim and return (short_summary_dict, total_pop).

    short_summary uses the M03 builder restricted to a single genotype.
    The output has 16 entries: 8 per-genotype (hpv16.*) + 8 aggregate (any.*).
    """
    sim = make_sim()
    sim.run()
    short = build_summary(sim, genotypes=GENOTYPES)
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop


if __name__ == '__main__':
    short, total_pop = run_and_summarize()
    print('Short summary (M02 HPV16 single-genotype):')
    for k, v in short.items():
        print(f'  {k:<48} {v:>12.4g}')
    print(f'  {"total population":<48} {total_pop:>12.4g}')
