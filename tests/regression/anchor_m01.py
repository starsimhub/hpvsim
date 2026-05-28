"""M01 anchor scenario for the v2 -> v3 migration regression harness.

Single-genotype HPV16, transmission only (SIS clearance, no precin/CIN/
cancer progression). Nigeria, 1990-2030 to keep the run short while
still spanning a credible demographic window.

Pinned anchor pars. Do not change without coordinating with regression baselines.
"""
import sys
from pathlib import Path

import sciris as sc

import hpvsim as hpv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary_m01 import build_summary_m01  # noqa: E402


PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=[16],
    start=1990,
    stop=2030,
    dt=0.25,
    rand_seed=0,
    verbose=0,
)


def make_sim():
    """Build (but do not run) the M01 anchor sim."""
    return hpv.Sim(**sc.dcp(PARS))


def run_and_summarize():
    """Run the M01 anchor sim and return the short summary dict."""
    sim = make_sim()
    sim.run()
    return build_summary_m01(sim)


if __name__ == '__main__':
    short = run_and_summarize()
    print('Short summary (M01 HPV16 transmission-only):')
    for k, v in short.items():
        print(f'  {k:<40} {v:>12.4g}')
