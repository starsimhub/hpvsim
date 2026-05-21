"""Generate the v2 30-seed baselines for the M05 vx anchor scenarios.

Run this ONCE from a separate v2 hpvsim env (NOT the v3 env). Writes two
gitignored JSON files (one per anchor) which the M05 parity tests
consume.

USAGE (from a v2 hpvsim env, NOT the v3 env):

    python tests/regression/multi_seed_v2_vx.py --n 30

Outputs:
    tests/regression/v2_seeds_n30_vx_routine.json
    tests/regression/v2_seeds_n30_vx_campaign.json
"""
import argparse
import json
import sys
from pathlib import Path

# These imports MUST be from a v2 hpvsim environment, not v3.
import hpvsim as hpv  # noqa: I001
import sciris as sc
import numpy as np

# Import the anchor PARS modules from the v3 tree (they are pure-Python).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.regression.anchor_vx_routine import PARS as ROUTINE_PARS  # noqa: E402
from tests.regression.anchor_vx_campaign import PARS as CAMPAIGN_PARS  # noqa: E402
from tests.regression.short_summary import build_summary  # noqa: E402


GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


def _build_v2_intervention(cfg):
    """Convert a serialized intervention spec into a v2 hpvsim object."""
    if cfg.kind == 'routine_vx':
        return hpv.routine_vx(
            product=cfg.product,
            prob=cfg.prob,
            age_range=cfg.age_range,
            sex=cfg.sex,
            start_year=cfg.start_year,
            label=cfg.name,
        )
    if cfg.kind == 'campaign_vx':
        return hpv.campaign_vx(
            product=cfg.product,
            prob=list(cfg.prob),
            age_range=cfg.age_range,
            sex=cfg.sex,
            years=list(cfg.years),
            interpolate=cfg.interpolate,
            label=cfg.name,
        )
    raise ValueError(f'Unknown intervention kind: {cfg.kind!r}')


def _build_v2_sim(pars, seed):
    intervention = _build_v2_intervention(pars.intervention)
    return hpv.Sim(
        location=pars.location,
        start=pars.start, end=pars.stop,
        rand_seed=int(seed),
        n_agents=pars.n_agents,
        genotypes=list(pars.genotypes),
        interventions=[intervention],
    )


def _run_anchor(pars, n_seeds, out_path):
    summaries = []
    for seed in range(n_seeds):
        sim = _build_v2_sim(pars, seed)
        sim.run()
        row = build_summary(sim, GENOTYPES)
        # Vaccination-specific scalars. v2 stores vaccinated / doses on People.
        row['_seed'] = int(seed)
        row['n_vaccinated_2060'] = int(sim.people.vaccinated.sum())
        row['n_doses_2060'] = int(sim.people.doses.sum())
        # Crude post-vaccination cancer incidence proxy: total new cancers
        # in [2030, 2060] / total person-years in that window.
        years = sim.results['year']
        mask = (years >= 2030) & (years < 2060)
        n_cancers = float(sim.results['new_cancers'][mask].sum())
        pop = sim.results['n_alive'][mask]
        py = float((pop * sim['dt']).sum())
        row['cancer_incidence_2030_2060'] = n_cancers / max(py, 1.0)
        # Trajectory baseline: per-year metrics for the parity test in Task 12.
        # Stored as a dict on each seed row so summary + trajectory share one JSON.
        row['_trajectory'] = sc.objdict(
            year=list(sim.results['year']),
            new_cancers=list(sim.results['new_cancers']),
            hpv_total_infections=list(sim.results['hpv_total_infections']),
            new_vaccinated=list(sim.results.get('new_vaccinated', [0] * len(sim.results['year']))),
        )
        summaries.append(row)
        print(f'  seed {seed}: n_vaccinated={row["n_vaccinated_2060"]}')
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f'Wrote {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=30, help='Number of seeds')
    args = parser.parse_args()

    here = Path(__file__).parent
    print(f'Generating routine baseline ({args.n} seeds)...')
    _run_anchor(ROUTINE_PARS, args.n, here / f'v2_seeds_n{args.n}_vx_routine.json')
    print(f'Generating campaign baseline ({args.n} seeds)...')
    _run_anchor(CAMPAIGN_PARS, args.n, here / f'v2_seeds_n{args.n}_vx_campaign.json')


if __name__ == '__main__':
    main()