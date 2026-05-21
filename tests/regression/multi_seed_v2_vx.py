"""Generate the v2 30-seed baselines for the M05 vx anchor scenarios.

Run this ONCE from a separate v2 hpvsim env (NOT the v3 env). Writes two
gitignored JSON files (one per anchor) which the M05 parity tests consume.

USAGE (from a v2.3 env, cwd at hpvsim_v23_frozen):

    .venv-v2/Scripts/python.exe <path>/multi_seed_v2_vx.py --n 30

Outputs:
    tests/regression/v2_seeds_n30_vx_routine.json
    tests/regression/v2_seeds_n30_vx_campaign.json

DO NOT run inside the v3 env.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc
import hpvsim as hpv  # v2.3 here — must be imported before sys.path manipulation

# Import the anchor PARS modules from the v3 tree. They are pure-Python (no v3
# hpvsim import at module level) so they are safe to import in a v2 env.
# The sys.path insert happens AFTER 'import hpvsim' so v2's hpvsim is already
# cached in sys.modules and won't be shadowed by any v3 tree import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tests.regression.anchor_vx_routine import PARS as ROUTINE_PARS   # noqa: E402
from tests.regression.anchor_vx_campaign import PARS as CAMPAIGN_PARS  # noqa: E402

# v2-specific summary extractors — same helpers the M3 baseline generator uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_v23 import (  # noqa: E402
    _per_genotype_metrics_v2,
    _aggregate_metrics_v2,
)

import numpy as np  # noqa: E402  (after sciris/hpvsim to keep import order clean)


# ---------------------------------------------------------------------------
# Intervention construction (v2 API)
# ---------------------------------------------------------------------------

def _build_v2_intervention(cfg):
    """Convert a serialised intervention spec into a v2 hpvsim object.

    v2 uses ``label=`` (not ``name=``) and ``product=`` takes a string directly.
    """
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


# ---------------------------------------------------------------------------
# v2 pars dict translation
# ---------------------------------------------------------------------------

def _build_v2_pars(anchor_pars, seed):
    """Translate an anchor PARS objdict into a v2 hpvsim pars dict."""
    return dict(
        location=anchor_pars.location,
        start=anchor_pars.start,
        end=anchor_pars.stop,          # v2 uses 'end', not 'stop'
        rand_seed=int(seed),
        n_agents=anchor_pars.n_agents, # v2 uses n_agents (confirmed from parameters list)
        genotypes=list(anchor_pars.genotypes),
        verbose=0,
        pop_scale=1,                   # disable real-world scaling (match v3 default)
        total_pop=anchor_pars.n_agents,
        ms_agent_ratio=1,              # disable multiscale dynamic spawning
        eff_condoms=0,                 # disable condom modulation (M03/M05 parity)
    )


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_seed(anchor_pars, seed):
    """Run one seed of a vx anchor scenario and return a summary row dict."""
    pars = _build_v2_pars(anchor_pars, seed)
    intervention = _build_v2_intervention(anchor_pars.intervention)
    sim = hpv.Sim(pars, interventions=[intervention])
    sim.run()

    genotype_map = sim.pars['genotype_map']   # {0: 'hpv16', 1: 'hpv18', ...}
    genotype_pairs = [(i, k) for i, k in sorted(genotype_map.items())]

    # --- Core 8-metric summary, per-genotype + aggregate ---
    # (Follows M3's multi_seed_v2.py pattern exactly.)
    summary = {}
    for gen_idx, gen_key in genotype_pairs:
        per = _per_genotype_metrics_v2(sim, gen_idx, gen_key)
        for k, v in per.items():
            summary[f'{gen_key}.{k}'] = float(v)

    agg = _aggregate_metrics_v2(sim, genotype_pairs)
    for k, v in agg.items():
        summary[f'any.{k}'] = float(v)

    summary['_seed'] = int(seed)
    if 'n_alive' in sim.results:
        summary['_total_pop'] = float(sim.results['n_alive'][-1])
    else:
        summary['_total_pop'] = float(sim.results['pop_size'][-1])

    # --- Vaccination-specific scalars ---
    # v2 stores vaccinated (bool) and doses (int) as per-agent state arrays.
    summary['n_vaccinated_2060'] = int(sim.people.vaccinated.sum())
    summary['n_doses_2060'] = int(sim.people.doses.sum())

    # --- cancer_incidence_2030_2060 ---
    # v2 uses 'cancers' (not 'new_cancers') for the per-step new-cancer flow
    # and 'n_alive' for the living population.
    years = np.asarray(sim.results['year'])
    mask = (years >= 2030) & (years < 2060)
    cancers_series = np.asarray(sim.results['cancers'])
    n_cancers = float(cancers_series[mask].sum())
    pop = np.asarray(sim.results['n_alive'])[mask]
    dt = float(sim.pars['dt'])
    py = float((pop * dt).sum())
    summary['cancer_incidence_2030_2060'] = n_cancers / max(py, 1.0)

    # --- Trajectory field ---
    # v2 key names differ from v3's HPVTotal analyzer:
    #   v2: 'infections' (per-step new HPV infections across all genotypes)
    #   v2: 'cancers'    (per-step new cancers across all genotypes)
    #   v2: 'new_vaccinated' (per-step newly vaccinated count)
    # We map them to the v3-matching names expected by the parity test.
    res = sim.results
    year_list = list(np.asarray(res['year']))
    new_cancers_list = list(np.asarray(res['cancers']))

    # v2's 'infections' is the per-step new-infection flow (across all genotypes)
    # equivalent to v3's hpv_total_infections from the HPVTotal analyzer.
    hpv_total_infections_list = list(np.asarray(res['infections']))

    # v2 has 'new_vaccinated' in results (init_results line 482)
    if 'new_vaccinated' in res:
        new_vaccinated_list = list(np.asarray(res['new_vaccinated']))
    else:
        new_vaccinated_list = [0] * len(year_list)

    summary['_trajectory'] = dict(
        year=year_list,
        new_cancers=new_cancers_list,
        hpv_total_infections=hpv_total_infections_list,
        new_vaccinated=new_vaccinated_list,
    )

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=30,
                   help='Number of seeds (default 30; use 1 for smoke testing)')
    p.add_argument('--start-seed', type=int, default=0,
                   help='First seed index (default 0; for parallelisation)')
    args = p.parse_args(argv)

    here = Path(__file__).resolve().parent
    seeds = list(range(args.start_seed, args.start_seed + args.n))

    anchors = [
        (ROUTINE_PARS,  here / f'v2_seeds_n{args.n}_vx_routine.json',  'routine'),
        (CAMPAIGN_PARS, here / f'v2_seeds_n{args.n}_vx_campaign.json', 'campaign'),
    ]

    for anchor_pars, out_path, label in anchors:
        print(f'Generating {label} baseline ({args.n} seeds)...')
        results = []
        t0 = time.time()
        for seed in seeds:
            ts = time.time()
            row = run_seed(anchor_pars, seed)
            dt = time.time() - ts
            results.append(row)
            print(f'  seed {seed}: n_vaccinated={row["n_vaccinated_2060"]}  '
                  f'any.total HPV infections={row["any.total HPV infections"]:.0f}  '
                  f'({dt:.1f}s)')
        total = time.time() - t0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Wrote {len(results)} seed summaries to {out_path} in {total:.1f}s\n')


if __name__ == '__main__':
    sys.exit(main())