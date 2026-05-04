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
# dt=0.25 matches v2's default sim timestep (declared at _v2_legacy/parameters.py:61)
# so that v2 baseline regen and v3 runs both use v2's default-driven calibrations.
PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotype='hpv16',
    start=1990,
    stop=2060,
    dt=0.25,
    rand_seed=0,
    verbose=0,
)


def make_sim():
    """Build (but do not run) the M1 anchor sim."""
    return hpv.Sim(**sc.dcp(PARS))


def run_and_summarize():
    """Run the M02 anchor sim and return (short_summary_dict, total_pop).

    Summary keys (matches v2's compute_summary):
      - total HPV infections
      - total cancers
      - total cancer deaths
      - mean HPV prevalence (%)
      - mean cancer incidence (per 100k)
      - mean age of infection (years)
      - mean age of cancer (years)
      - mean age of cancer death (years)
    """
    sim = make_sim()
    sim.run()
    res = sim.results.hpv16
    mod = sim.diseases.hpv16
    dt = float(PARS['dt'])
    pop_scale = float(getattr(sim.pars, 'pop_scale', 1.0) or 1.0)

    # 1. HPV infections (cumulative)
    if 'cum_infections' in res:
        n_inf = float(res.cum_infections[-1])
    elif 'new_infections' in res:
        n_inf = float(res.new_infections.sum())
    else:
        n_inf = float(res.n_infected.sum())

    # 4. Mean HPV prevalence
    mean_prev_pct = 100 * float(res.prevalence.mean())

    # 6. Mean age of first infection (preserved from M01)
    ti_first = mod.ti_first_infection
    ever_first = ti_first.notnan.uids
    if len(ever_first):
        ages_now = np.asarray(sim.people.age[ever_first])
        ti_at_inf = np.asarray(ti_first[ever_first])
        years_since = (float(sim.t.ti) - ti_at_inf) * dt
        mean_age_inf = float((ages_now - years_since).mean())
    else:
        mean_age_inf = 0.0

    # 2. Total cancers — agents whose ti_cancerous was realized during the sim
    #    (ti_cancerous <= ti_now).  Agents with ti_cancerous scheduled beyond
    #    the sim end have their transition pre-computed but never executed, so
    #    they should not count.  Scaled by pop_scale for real-world counts.
    ti_now = float(sim.t.ti)
    all_cancer_uids = mod.ti_cancerous.notnan.uids
    if len(all_cancer_uids):
        ti_cancer_all = np.asarray(mod.ti_cancerous[all_cancer_uids])
        realized_cancer_mask = ti_cancer_all <= ti_now
        realized_cancer_uids = all_cancer_uids[realized_cancer_mask]
        ti_at_cancer = ti_cancer_all[realized_cancer_mask]
    else:
        realized_cancer_uids = all_cancer_uids  # empty
        ti_at_cancer = np.array([], dtype=float)
    n_cancers = float(len(realized_cancer_uids)) * pop_scale

    # 7. Mean age of cancer onset (realized cancer agents only).
    if len(realized_cancer_uids):
        ages_now_c = np.asarray(sim.people.age[realized_cancer_uids])
        # age_now - (ti_now - ti_cancer)*dt recovers age-at-cancer-onset.
        # For agents still alive: exact.
        # For dead agents: people.age is frozen at age-of-death; the subtracted
        # years_since_cancer overshoots slightly (by post-death frozen-age
        # delta). Acceptable approximation for M02 dev-gate; refine if needed.
        years_since_cancer = (ti_now - ti_at_cancer) * dt
        mean_age_cancer = float((ages_now_c - years_since_cancer).mean())
    else:
        mean_age_cancer = 0.0

    # 3. Total cancer deaths — agents whose ti_dead_cancer is in the past
    #    AND who are no longer alive.
    #    Note: starsim removes dead agents from the live pool; after the sim
    #    ends, people.alive reflects only currently-living agents.  For agents
    #    who died from cancer (ti_dead_cancer <= ti_now), request_death fires
    #    step_die which sets alive=False and removes them.  However, if all
    #    cancer agents still have ti_dead_cancer > ti_now at sim end (i.e. the
    #    sim window is shorter than cancer duration), n_cancer_deaths = 0 is
    #    correct — no cancer deaths occurred during the run.
    all_dead_uids = mod.ti_dead_cancer.notnan.uids
    if len(all_dead_uids):
        ti_dead_arr = np.asarray(mod.ti_dead_cancer[all_dead_uids])
        alive_arr = np.asarray(sim.people.alive[all_dead_uids])
        realized = (ti_dead_arr <= ti_now) & (~alive_arr)
        realized_dead_uids = all_dead_uids[realized]
    else:
        realized_dead_uids = all_dead_uids  # empty
    n_cancer_deaths = float(len(realized_dead_uids)) * pop_scale

    # 8. Mean age of cancer death — for dead cancer agents, people.age is
    #    frozen at age-of-death (starsim freezes age when an agent dies).
    if len(realized_dead_uids):
        ages_at_death = np.asarray(sim.people.age[realized_dead_uids])
        mean_age_cancer_death = float(ages_at_death.mean())
    else:
        mean_age_cancer_death = 0.0

    # 5. Mean cancer incidence (per 100k female-years).
    # Total cancer events / total female-alive years × 100k.
    # n_alive counts both sexes; female-years approximation = n_alive/2 × dt.
    n_alive_series = np.asarray(sim.results['n_alive'])
    total_alive_years = float(n_alive_series.sum()) * dt
    female_years = total_alive_years / 2.0
    mean_cancer_incidence = (n_cancers / female_years * 100_000.0) if female_years > 0 else 0.0

    short = {
        'total HPV infections': n_inf,
        'total cancers': n_cancers,
        'total cancer deaths': n_cancer_deaths,
        'mean HPV prevalence (%)': mean_prev_pct,
        'mean cancer incidence (per 100k)': mean_cancer_incidence,
        'mean age of infection (years)': mean_age_inf,
        'mean age of cancer (years)': mean_age_cancer,
        'mean age of cancer death (years)': mean_age_cancer_death,
    }
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop


if __name__ == '__main__':
    short, total_pop = run_and_summarize()
    print('Short summary:')
    for k, v in short.items():
        print(f'  {k:<40} {v:>12.4g}')
    print(f'  {"total population":<40} {total_pop:>12.4g}')