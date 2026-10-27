"""M07 demo: vaccination-coverage sweep with multi-seed UQ.

Runs a 4 × 20 (coverage × seed) routine-vx sweep on the M05 Nigeria
4-genotype anchor, reduces each coverage scenario to median + 10/90
quantiles, and plots aggregate cancer trajectories by coverage.

  - make_sim(coverage) builds one labeled sim per scenario
  - ss.parallel(*sims, n_runs=N_SEEDS) runs each with N_SEEDS seeds
  - the MultiSim's results have a run axis first, reduced by scenario
    with .quantiles()

Run from the repo root:
    python examples/m07_uq_sweep.py
Produces:
    m07_uq_sweep.png  (cumulative cancers by coverage, with CIs)
"""
import argparse
from pathlib import Path

import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv


COVERAGES = [0.0, 0.3, 0.6, 0.9]
N_SEEDS = 20

# Nigeria 4-genotype base scenario for the sweep.
VX_PARS = sc.objdict(
    location='nigeria',
    start=1990, stop=2060,
    n_agents=20_000,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    # Integer cohort ages (birth + migration + initial pop) on an annual
    # demographic cadence — keeps per-cohort vaccination clean.
    v2_compat_demographics=True,
)


def make_sim(coverage, **kwargs):
    """Build the sim of one coverage: routine bivalent vaccination of girls as they turn 9, from 2020"""
    vaccination = ss.vaccinate(hpv.vx('bivalent'), where=ss.ppl.female, at_age=9, coverage=coverage, start=2020, name='routine_bivalent_girls')
    return hpv.Sim(**(VX_PARS | kwargs), verbose=0, label=f'coverage={coverage:.0%}', interventions=vaccination)


def main(out_png=Path('m07_uq_sweep.png')):
    sims = [make_sim(c) for c in COVERAGES]
    print(f'Running {len(sims)} scenarios × {N_SEEDS} seeds via ss.parallel...')
    msim = ss.parallel(*sims, n_runs=N_SEEDS, verbose=0)
    _plot_coverage_sweep(msim, out_png)
    print(f'Wrote {out_png}')


def _plot_coverage_sweep(msim, out_png):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    res = msim.results.all_hpv.cum_cancers # Runs by time, from the HPVTotal analyzer
    q = np.asarray(res.quantiles([0.1, 0.5, 0.9])).reshape(len(COVERAGES), 3, -1) # One row per scenario and quantile
    lo, median, hi = q[:, 0], q[:, 1], q[:, 2] # Each scenario by time
    years = np.asarray(msim.sims[0].t.yearvec)
    for i, cov in enumerate(COVERAGES):
        line, = ax.plot(years, median[i], label=f'coverage={cov:.0%}')
        ax.fill_between(years, lo[i], hi[i], alpha=0.2, color=line.get_color())
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative cancers')
    ax.set_title('Cumulative cancers by routine-vx coverage (median + 10/90)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('m07_uq_sweep.png'))
    args = p.parse_args()
    main(out_png=args.out)
