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

    # 6. Mean age of LATEST infection across alive ever-infected agents.
    #    Uses ``ti_infected`` (overwritten on each new infection) to match
    #    v2's ``people.date_infectious`` semantics in baseline_v23.py:
    #    both report the age at the most-recent infection, not the first.
    #    With imm_init=0.35 partial permanent immunity reinfection is rare,
    #    so most agents have ti_infected == ti_first_infection; the
    #    distinction matters mainly for edge cases.
    ti_latest = mod.ti_infected
    ever_inf = ti_latest.notnan.uids
    if len(ever_inf):
        ages_now = np.asarray(sim.people.age[ever_inf])
        ti_at_inf = np.asarray(ti_latest[ever_inf])
        years_since = (float(sim.t.ti) - ti_at_inf) * dt
        mean_age_inf = float((ages_now - years_since).mean())
    else:
        mean_age_inf = 0.0

    # 2. Total cancers — agents who ACTUALLY transitioned to the cancerous
    #    compartment during the sim. Use the lifetime ``ever_cancerous`` flag
    #    rather than ``(ti_cancerous_raw <= ti_now) & finite``: ti_cancerous
    #    is set in set_prognoses for every cancer-bound agent, but the
    #    cin → cancerous transition only fires in step_state if the agent is
    #    still alive (and in cin) when ti_cancerous is reached. Agents who
    #    die of background mortality before their scheduled ti_cancerous
    #    have step_die clear the ``cin`` flag, so the transition never
    #    fires — yet ti_cancerous in .raw retains its scheduled value, which
    #    inflates a naive (ti_cancerous <= ti_now) mask by ~30% (verified
    #    empirically: 743 raw vs 543 actual transitions on the M02 anchor).
    ti_now = float(sim.t.ti)
    ti_cancer_raw = np.asarray(mod.ti_cancerous.raw)
    ti_dead_raw = np.asarray(mod.ti_dead_cancer.raw)
    alive_raw = np.asarray(sim.people.alive.raw)
    ever_cancerous_raw = np.asarray(mod.ever_cancerous.raw)
    cancer_realized_mask = ever_cancerous_raw
    cancer_realized_idx = np.where(cancer_realized_mask)[0]
    ti_at_cancer = ti_cancer_raw[cancer_realized_mask]
    n_cancers = float(cancer_realized_mask.sum()) * pop_scale

    # 7. Mean age of cancer onset (realized cancer agents only).
    #    Same .raw discipline — include dead agents.
    age_raw = np.asarray(sim.people.age.raw)
    if cancer_realized_mask.any():
        ages_now_c = age_raw[cancer_realized_idx]
        # For alive agents: people.age is current age, so years-since-cancer
        # = (ti_now - ti_cancer) * dt and age_at_cancer = current_age - years_since.
        # For dead agents: people.age is frozen at age-of-death, so we use
        # ti_dead_cancer (if cancer-death) or ti_now (proxy for any-cause death,
        # since starsim freezes age on death) as the effective "now". For
        # agents whose ti_dead_cancer is set and in the past, that's the
        # death time; otherwise (background mortality), the agent's age was
        # frozen at some earlier ti — we don't have direct access to the
        # exact death ti without an analyzer, so fall back to ti_now which
        # may slightly underestimate the actual age-at-cancer for those
        # who died of background mortality between cancer onset and sim end.
        effective_ti = np.where(
            alive_raw[cancer_realized_idx],
            ti_now,
            np.where(
                np.isfinite(ti_dead_raw[cancer_realized_idx]),
                ti_dead_raw[cancer_realized_idx],
                ti_now,
            ),
        )
        years_since_cancer = (effective_ti - ti_at_cancer) * dt
        mean_age_cancer = float((ages_now_c - years_since_cancer).mean())
    else:
        mean_age_cancer = 0.0

    # 3. Total cancer deaths — use the lifetime ``dead_cancer`` flag, set
    #    only when the cancer-death pipeline actually fires
    #    (cancerous & ti_dead_cancer <= ti). A naive
    #    (ti_dead_raw <= ti_now & ~alive) mask over-counts by including
    #    cancer-onset agents who died of background mortality after onset
    #    but before their scheduled ti_dead_cancer fired (verified
    #    empirically: ~+25% inflation on the M02 anchor).
    dead_realized_mask = np.asarray(mod.dead_cancer.raw)
    dead_realized_idx = np.where(dead_realized_mask)[0]
    n_cancer_deaths = float(dead_realized_mask.sum()) * pop_scale

    # 8. Mean age of cancer death — people.age is frozen at agent death,
    #    so age.raw[uid] for a dead cancer agent ≈ age at cancer death.
    if dead_realized_mask.any():
        ages_at_death = age_raw[dead_realized_idx]
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