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
    dt = float(PARS['dt'])
    pop_scale = float(getattr(sim.pars, 'pop_scale', 1.0) or 1.0)
    mod = sim.diseases.hpv16

    # 1. HPV infections (cumulative)
    if 'cum_infections' in res:
        n_inf = float(res.cum_infections[-1])
    elif 'new_infections' in res:
        n_inf = float(res.new_infections.sum())
    else:
        n_inf = float(res.n_infected.sum())

    # 4. Mean HPV prevalence
    mean_prev_pct = 100 * float(res.prevalence.mean())

    # 6. Mean age of LATEST infection across alive ever-infected agents.
    #    Uses ``ti_infected`` (overwritten on each new infection) to match
    #    v2's ``people.date_infectious`` semantics in baseline_v23.py.
    ti_latest = mod.ti_infected
    ever_inf = ti_latest.notnan.uids
    if len(ever_inf):
        ages_now = np.asarray(sim.people.age[ever_inf])
        ti_at_inf = np.asarray(ti_latest[ever_inf])
        years_since = (float(sim.t.ti) - ti_at_inf) * dt
        mean_age_inf = float((ages_now - years_since).mean())
    else:
        mean_age_inf = 0.0

    # 2. Total cancers — sum of per-step new-cancer counts emitted by the
    #    cin -> cancerous transition in HPV.step_state.
    new_cancers = np.asarray(res.new_cancers)
    n_cancers_unscaled = float(new_cancers.sum())
    n_cancers = n_cancers_unscaled * pop_scale

    # 7. Mean age of cancer onset = sum(age@onset) / count.
    sum_age_cancer = float(np.asarray(res.sum_age_at_cancer).sum())
    mean_age_cancer = (sum_age_cancer / n_cancers_unscaled) if n_cancers_unscaled > 0 else 0.0

    # 3. Total cancer deaths.
    new_cancer_deaths = np.asarray(res.new_cancer_deaths)
    n_cd_unscaled = float(new_cancer_deaths.sum())
    n_cancer_deaths = n_cd_unscaled * pop_scale

    # 8. Mean age of cancer death = sum(age@death) / count.
    sum_age_cd = float(np.asarray(res.sum_age_at_cancer_death).sum())
    mean_age_cancer_death = (sum_age_cd / n_cd_unscaled) if n_cd_unscaled > 0 else 0.0

    # 5. Mean cancer incidence (per 100k female-years). n_alive counts
    # both sexes; female-years approximation = n_alive/2 * dt.
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