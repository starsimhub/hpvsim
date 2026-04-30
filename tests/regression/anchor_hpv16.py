"""M01 anchor scenario for the v2 -> v3 migration regression harness.

Single-genotype (HPV16) HPV sim, Nigeria, fixed seed, no interventions, no
analyzers. Tooling under tests/regression/ (compare.py) imports
run_and_summarize() from here.

Run as a script to print the summary:
    python tests/regression/anchor_hpv16.py
"""

import numpy as np
import sciris as sc

import hpvsim as hpv

# Pinned anchor pars. Do not change without coordinating with regression baselines.
PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotype='hpv16',
    start=1990,
    stop=2060,
    dt=0.5,
    rand_seed=0,
    verbose=0,
)


def make_sim():
    """Build (but do not run) the M1 anchor sim."""
    return hpv.Sim(**sc.dcp(PARS))


def run_and_summarize():
    """Run the M1 anchor sim and return (short_summary_dict, total_population_float).

    Summary keys:
      - total HPV infections (HPV16 cumulative)
      - mean HPV prevalence (%) (HPV16, mean over the run)
      - mean age of infection (years) (HPV16)
    """
    sim = make_sim()
    sim.run()
    res = sim.results.hpv16

    # Cumulative infections - prefer cum_infections if Starsim provides it,
    # else fall back to summing new_infections.
    if 'cum_infections' in res:
        n_inf = float(res.cum_infections[-1])
    elif 'new_infections' in res:
        n_inf = float(res.new_infections.sum())
    else:
        n_inf = float(res.n_infected.sum())

    mean_prev_pct = 100 * float(res.prevalence.mean())

    # Mean age of first infection: matches v2's per-agent date_infectious
    # semantics (in v2 each agent's first HPV16 infection is recorded
    # once; immunity prevents re-infection). hpv.HPV.set_prognoses
    # accumulates per-step (count, age-sum) of first-time infections.
    age_sum = np.asarray(res.new_first_infections_age_sum)
    age_count = np.asarray(res.new_first_infections_count)
    has_inf = age_count > 0
    if has_inf.any():
        per_step_mean = age_sum[has_inf] / age_count[has_inf]
        mean_age_inf = float(per_step_mean.mean())
    else:
        mean_age_inf = 0.0

    short = {
        'total HPV infections': n_inf,
        'mean HPV prevalence (%)': mean_prev_pct,
        'mean age of infection (years)': mean_age_inf,
    }
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop


if __name__ == '__main__':
    short, total_pop = run_and_summarize()
    print('Short summary:')
    for k, v in short.items():
        print(f'  {k:<40} {v:>12.4g}')
    print(f'  {"total population":<40} {total_pop:>12.4g}')