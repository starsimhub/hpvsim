"""M07 demo: vaccination-coverage sweep with multi-seed UQ.

Runs a 4 × 20 (coverage × seed) routine-vx sweep on the M05 Nigeria
4-genotype anchor, reduces each coverage scenario to median + 10/90
quantiles, and plots aggregate cancer trajectories by coverage.

Exercises every M07 verification target:
  - sc.parallelize for parallel sim CREATION
  - ss.parallel for parallel sim EXECUTION
  - ss.MultiSim.median for result REDUCTION
  - make_sim(seed, coverage) builds one labeled sim per scenario point

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


def make_sim(seed, coverage):
    """Build one sim for the (seed, coverage) point.

    Builds a fresh hpv.routine_vx with the swept coverage.

    Sim deep-copies inputs at __init__, so post-run inspection must go
    through sim.interventions, not the local reference.
    """
    return hpv.Sim(
        location=VX_PARS.location,
        start=VX_PARS.start,
        stop=VX_PARS.stop,
        rand_seed=int(seed),
        n_agents=VX_PARS.n_agents,
        genotypes=list(VX_PARS.genotypes),
        v2_compat_demographics=VX_PARS.v2_compat_demographics,
        verbose=0,
        label=f'coverage={coverage:.0%}|seed={seed}',
        interventions=[hpv.routine_vx(
            product='bivalent',
            prob=float(coverage),
            age_range=[9, 10],
            sex='f',
            start_year=2020,
            name='routine_bivalent_girls',
        )],
    )


def main(out_png=Path('m07_uq_sweep.png')):
    iterkwargs = [dict(seed=s, coverage=c)
                  for c in COVERAGES for s in range(N_SEEDS)]
    print(f'Building {len(iterkwargs)} sims via sc.parallelize...')
    sims = sc.parallelize(make_sim, iterkwargs=iterkwargs)

    print(f'Running {len(sims)} sims via ss.parallel...')
    msim = ss.parallel(*sims, verbose=0)

    # Group sims back into per-coverage MultiSims for reduction.
    by_cov = {}
    for cov in COVERAGES:
        tag = f'coverage={cov:.0%}'
        cov_sims = [s for s in msim.sims if s.label.startswith(tag)]
        sub = ss.MultiSim(cov_sims)
        sub.median()
        by_cov[cov] = sub

    _plot_coverage_sweep(by_cov, out_png)
    print(f'Wrote {out_png}')


def _plot_coverage_sweep(by_cov, out_png):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for cov, sub_msim in by_cov.items():
        # The HPVTotal analyzer aggregates per-genotype cancers; results
        # are flattened to top-level keys after .median().
        # Try a few candidate result keys to be robust to API drift.
        candidates = [
            'hpv_cum_cancers',
            'hpv16_cum_cancers',
        ]
        result_key = next(
            (k for k in candidates if hasattr(sub_msim.results, k)), None
        )
        if result_key is None:
            raise RuntimeError(
                f'No expected result key found on reduced msim. '
                f'Tried {candidates}. Available keys: '
                f'{list(sub_msim.results.keys())[:20]}'
            )
        res = getattr(sub_msim.results, result_key)
        tv = sub_msim.sims[0].t.timevec
        # Convert ss.date objects to float years (Starsim 3.x timevec
        # contains ss.date — see starsim-dev-time skill).
        if hasattr(tv[0], 'years'):
            years = np.array([float(y.years) for y in tv])
        else:
            years = np.asarray(tv, dtype=float)
        median = np.asarray(res)
        lo = np.asarray(getattr(res, 'low', median))
        hi = np.asarray(getattr(res, 'high', median))
        line, = ax.plot(years, median, label=f'coverage={cov:.0%}')
        ax.fill_between(years, lo, hi, alpha=0.2, color=line.get_color())
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
