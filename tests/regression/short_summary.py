"""M03 40-entry short-summary builder.

For each genotype produces the 8-metric M02 summary (total HPV infections,
total cancers, total cancer deaths, mean HPV prevalence, mean cancer
incidence, mean ages of infection / cancer / cancer death). Plus an
8-metric ``any.*`` aggregate computed from the Aggregate analyzer's
*_any results (sim.results.aggregate.*).
"""

import numpy as np


METRIC_KEYS = (
    'total HPV infections',
    'total cancers',
    'total cancer deaths',
    'mean HPV prevalence (%)',
    'mean cancer incidence (per 100k)',
    'mean age of infection (years)',
    'mean age of cancer (years)',
    'mean age of cancer death (years)',
)


def _per_genotype_metrics(sim, genotype):
    """Compute the 8-metric M02 summary for one genotype (key into sim.results)."""
    res = sim.results[genotype]
    mod = sim.diseases[genotype]
    dt = float(sim.t.dt)
    pop_scale = float(getattr(sim.pars, 'pop_scale', 1.0) or 1.0)

    new_infections = np.asarray(res.new_infections)
    n_inf_unscaled = float(new_infections.sum())
    n_inf = n_inf_unscaled * pop_scale

    mean_prev_pct = 100 * float(np.asarray(res.prevalence).mean())

    ti_latest = mod.ti_infected
    ever_inf = ti_latest.notnan.uids
    if len(ever_inf):
        ages_now = np.asarray(sim.people.age[ever_inf])
        ti_at_inf = np.asarray(ti_latest[ever_inf])
        years_since = (float(sim.t.ti) - ti_at_inf) * dt
        mean_age_inf = float((ages_now - years_since).mean())
    else:
        mean_age_inf = 0.0

    new_cancers = np.asarray(res.new_cancers)
    n_cancers_unscaled = float(new_cancers.sum())
    n_cancers = n_cancers_unscaled * pop_scale

    sum_age_cancer = float(np.asarray(res.sum_age_at_cancer).sum())
    mean_age_cancer = (sum_age_cancer / n_cancers_unscaled) if n_cancers_unscaled > 0 else 0.0

    new_cancer_deaths = np.asarray(res.new_cancer_deaths)
    n_cd_unscaled = float(new_cancer_deaths.sum())
    n_cancer_deaths = n_cd_unscaled * pop_scale

    sum_age_cd = float(np.asarray(res.sum_age_at_cancer_death).sum())
    mean_age_cancer_death = (sum_age_cd / n_cd_unscaled) if n_cd_unscaled > 0 else 0.0

    n_alive_series = np.asarray(sim.results['n_alive'])
    total_alive_years = float(n_alive_series.sum()) * dt
    female_years = total_alive_years / 2.0
    mean_cancer_incidence = (n_cancers / female_years * 100_000.0) if female_years > 0 else 0.0

    return {
        'total HPV infections': n_inf,
        'total cancers': n_cancers,
        'total cancer deaths': n_cancer_deaths,
        'mean HPV prevalence (%)': mean_prev_pct,
        'mean cancer incidence (per 100k)': mean_cancer_incidence,
        'mean age of infection (years)': mean_age_inf,
        'mean age of cancer (years)': mean_age_cancer,
        'mean age of cancer death (years)': mean_age_cancer_death,
    }


def _aggregate_metrics(sim, genotypes):
    """Compute the 8-metric aggregate from the Aggregate analyzer's *_any results.

    Aggregate results live at ``sim.results.aggregate.*`` (not at the top-level
    ``sim.results.*``). Mean-age metrics are pooled by per-genotype sums and
    counts: mean = (sum across genotypes of sum_age) / (sum across genotypes
    of count).
    """
    dt = float(sim.t.dt)
    pop_scale = float(getattr(sim.pars, 'pop_scale', 1.0) or 1.0)
    agg = sim.results.aggregate

    n_inf_unscaled = float(np.asarray(agg.cum_infections_any)[-1])
    n_inf = n_inf_unscaled * pop_scale

    cum_c = float(np.asarray(agg.cum_cancers_any)[-1])
    n_cancers = cum_c * pop_scale
    n_cancers_unscaled = cum_c

    new_cd_any = np.asarray(agg.new_cancer_deaths_any)
    n_cd_unscaled = float(new_cd_any.sum())
    n_cancer_deaths = n_cd_unscaled * pop_scale

    # Mean prev: average across genotypes' prevalences (approximation).
    prevs = [np.asarray(sim.results[g].prevalence) for g in genotypes]
    mean_prev_pct = 100 * float(np.mean(np.column_stack(prevs)))

    n_alive_series = np.asarray(sim.results['n_alive'])
    total_alive_years = float(n_alive_series.sum()) * dt
    female_years = total_alive_years / 2.0
    mean_cancer_incidence = (n_cancers / female_years * 100_000.0) if female_years > 0 else 0.0

    # Pool per-genotype mean-age sums and counts.
    sum_age_inf_total = 0.0
    n_inf_count_total = 0.0
    sum_age_cancer_total = 0.0
    sum_age_cd_total = 0.0
    n_cd_count_total = 0.0
    for g in genotypes:
        mod = sim.diseases[g]
        ti_latest = mod.ti_infected
        ever_inf = ti_latest.notnan.uids
        if len(ever_inf):
            ages_now = np.asarray(sim.people.age[ever_inf])
            ti_at_inf = np.asarray(ti_latest[ever_inf])
            years_since = (float(sim.t.ti) - ti_at_inf) * dt
            sum_age_inf_total += float((ages_now - years_since).sum())
            n_inf_count_total += float(len(ever_inf))
        sum_age_cancer_total += float(np.asarray(sim.results[g].sum_age_at_cancer).sum())
        sum_age_cd_total += float(np.asarray(sim.results[g].sum_age_at_cancer_death).sum())
        n_cd_count_total += float(np.asarray(sim.results[g].new_cancer_deaths).sum())

    mean_age_inf = (sum_age_inf_total / n_inf_count_total) if n_inf_count_total > 0 else 0.0
    mean_age_cancer = (sum_age_cancer_total / n_cancers_unscaled) if n_cancers_unscaled > 0 else 0.0
    mean_age_cancer_death = (sum_age_cd_total / n_cd_count_total) if n_cd_count_total > 0 else 0.0

    return {
        'total HPV infections': n_inf,
        'total cancers': n_cancers,
        'total cancer deaths': n_cancer_deaths,
        'mean HPV prevalence (%)': mean_prev_pct,
        'mean cancer incidence (per 100k)': mean_cancer_incidence,
        'mean age of infection (years)': mean_age_inf,
        'mean age of cancer (years)': mean_age_cancer,
        'mean age of cancer death (years)': mean_age_cancer_death,
    }


def build_summary(sim, genotypes):
    """Return the per-genotype + aggregate summary dict.

    Keys are ``<genotype>.<metric>`` for per-genotype entries and
    ``any.<metric>`` for aggregate entries. For ``n`` genotypes, returns
    ``8 * (n + 1)`` entries (40 for the 4-genotype anchor).
    """
    out = {}
    for g in genotypes:
        per = _per_genotype_metrics(sim, g)
        for k, v in per.items():
            out[f'{g}.{k}'] = v
    agg = _aggregate_metrics(sim, genotypes)
    for k, v in agg.items():
        out[f'any.{k}'] = v
    return out