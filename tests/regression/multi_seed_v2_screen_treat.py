"""Generate the v2 30-seed baselines for the M06 screen-treat cascade anchor.

Run this ONCE from a separate v2 hpvsim env (NOT the v3 env). Writes one
gitignored JSON file which the M06 parity tests consume.

USAGE (from a v2.3 env, cwd at hpvsim_v23_frozen):

    .venv-v2/Scripts/python.exe <path>/multi_seed_v2_screen_treat.py --n 30

Output:
    tests/regression/v2_seeds_n30_screen_treat.json

DO NOT run inside the v3 env.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sciris as sc
import hpvsim as hpv  # v2.3 here — must be imported before sys.path manipulation

# ---------------------------------------------------------------------------
# Module-level constants — smoke-test harness reads these without calling main()
# ---------------------------------------------------------------------------
N_SEEDS = 30
HERE = Path(__file__).resolve().parent
OUT = HERE / 'v2_seeds_n30_screen_treat.json'


# ---------------------------------------------------------------------------
# v2 pars dict translation
# ---------------------------------------------------------------------------

def _build_v2_pars(pars, seed):
    """Translate anchor PARS into a v2 hpvsim pars dict."""
    return dict(
        location=pars.location,
        start=pars.start,
        end=pars.stop,          # v2 uses 'end', not 'stop'
        rand_seed=int(seed),
        n_agents=pars.n_agents,
        genotypes=list(pars.genotypes),
        verbose=0,
        pop_scale=1,            # disable real-world scaling (match v3 default)
        total_pop=pars.n_agents,
        ms_agent_ratio=1,       # disable multiscale dynamic spawning
        eff_condoms=0,          # disable condom modulation (M03/M05 parity)
    )


# ---------------------------------------------------------------------------
# v2 cascade construction
# ---------------------------------------------------------------------------

def _build_interventions(pars):
    """Build the three-step v2 cascade from anchor PARS.

    v2 triage eligibility is a lambda that calls sim.get_intervention(label)
    at runtime. The label must match the label= kwarg passed to the screen
    and triage constructors respectively.
    """
    screen_cfg = pars.screen
    primary = hpv.routine_screening(
        product=screen_cfg.product,
        prob=screen_cfg.prob,
        age_range=screen_cfg.age_range,
        start_year=screen_cfg.start_year,
        end_year=screen_cfg.end_year,
        label=screen_cfg.name,
    )

    triage_cfg = pars.triage
    colpo = hpv.routine_triage(
        product=triage_cfg.product,
        prob=triage_cfg.prob,
        eligibility=lambda sim: sim.get_intervention(screen_cfg.name).outcomes['positive'],
        start_year=triage_cfg.start_year,
        end_year=triage_cfg.end_year,
        label=triage_cfg.name,
    )

    treat_cfg = pars.treat
    excision = hpv.treat_num(
        product=treat_cfg.product,
        prob=treat_cfg.prob,
        eligibility=lambda sim: sim.get_intervention(triage_cfg.name).outcomes['hsil'],
        label=treat_cfg.name,
    )

    return [primary, colpo, excision]


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_seed(pars, seed):
    """Run one seed of the screen-treat cascade and return a summary dict."""
    v2_pars = _build_v2_pars(pars, seed)
    interventions = _build_interventions(pars)
    sim = hpv.Sim(v2_pars, interventions=interventions)
    sim.run()

    # M05 lesson 1 (alive-mask): per-agent boolean states must be filtered to
    # alive agents only so they compare apples-to-apples with v3's BoolArr.sum()
    # which automatically excludes dead agents.
    alive = sim.people.alive

    summary = {}
    summary['_seed'] = int(seed)

    # n_screened_2060: cumulative unique people ever screened (alive only)
    summary['n_screened_2060'] = int(sim.people.screened[alive].sum())

    # n_screens_2060: cumulative total screening visits (alive only)
    # v2 tracks this per-agent as people.screens (integer count)
    summary['n_screens_2060'] = int(sim.people.screens[alive].sum())

    # n_cin_treated_2060: cumulative unique people ever treated for CIN (alive only)
    summary['n_cin_treated_2060'] = int(sim.people.cin_treated[alive].sum())

    # n_cin_treatments_2060: cumulative total CIN treatment events (alive only)
    # v2 tracks this per-agent as people.cin_treatments (integer count)
    if hasattr(sim.people, 'cin_treatments'):
        summary['n_cin_treatments_2060'] = int(sim.people.cin_treatments[alive].sum())
    else:
        # Fallback: use cum_cin_treatments from results (not alive-masked, but
        # the best available if the per-agent counter doesn't exist in this v2
        # build). Document the fallback so consumers know.
        summary['n_cin_treatments_2060'] = int(float(sim.results['cum_cin_treatments'][-1]))

    # --- M05 lesson 2 (person-years): cancer incidence over [2030, 2060) ---
    # v2's results are aggregated to ANNUAL cadence (resfreq = 1/dt = 4 per
    # year). Each entry in 'n_alive' represents a one-year snapshot, so the
    # per-entry time contribution to person-years is:
    #   annual_dt = resfreq * dt = 4 * 0.25 = 1.0 year
    # The 'cancers' flow gives per-step new cancers; 'n_alive' gives population
    # at each annual step.
    years = np.asarray(sim.results['year'])
    mask = (years >= 2030) & (years < 2060)

    cancers_series = np.asarray(sim.results['cancers'])
    n_cancers = float(cancers_series[mask].sum())

    pop = np.asarray(sim.results['n_alive'])[mask]
    # M05 lesson 2: annual_dt = resfreq * dt (= 1.0 for annual results)
    annual_dt = float(sim.resfreq) * float(sim.pars['dt'])
    py = float((pop * annual_dt).sum())
    summary['cancer_incidence_2030_2060'] = n_cancers / max(py, 1.0)

    # --- cancer_deaths_2030_2060: total cancer deaths over [2030, 2060) ---
    cancer_deaths_series = np.asarray(sim.results['cancer_deaths'])
    summary['cancer_deaths_2030_2060'] = float(cancer_deaths_series[mask].sum())

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=N_SEEDS,
                   help='Number of seeds (default 30; use 1 for smoke testing)')
    p.add_argument('--start-seed', type=int, default=0,
                   help='First seed index (default 0; for parallelisation)')
    p.add_argument('--out', type=Path, default=OUT,
                   help='Output JSON path')
    args = p.parse_args(argv)

    # Import anchor PARS lazily (safe to import in v2 env — anchor module has
    # no top-level hpvsim import)
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))
    from tests.regression.anchor_screen_treat import PARS  # noqa: E402

    seeds = list(range(args.start_seed, args.start_seed + args.n))
    out_path = args.out

    print(f'Generating screen-treat baseline ({args.n} seeds) -> {out_path}')
    results = []
    t0 = time.time()
    for seed in seeds:
        ts = time.time()
        row = run_seed(PARS, seed)
        dt = time.time() - ts
        results.append(row)
        print(f'  seed {seed}: n_screened={row["n_screened_2060"]}  '
              f'n_cin_treated={row["n_cin_treated_2060"]}  '
              f'cancer_incidence={row["cancer_incidence_2030_2060"]:.2e}  '
              f'({dt:.1f}s)')
    total = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {len(results)} seed summaries to {out_path} in {total:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
