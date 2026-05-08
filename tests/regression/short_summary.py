"""M03 40-entry short-summary builder.

For each genotype produces the 8-metric M02 summary (total HPV infections,
total cancers, total cancer deaths, mean HPV prevalence, mean cancer
incidence, mean ages of infection / cancer / cancer death). Plus an
8-metric ``any.*`` aggregate computed from the Aggregate analyzer's
*_any results (sim.results.aggregate.*).

Three measurement formulas mirror v2's exact calculation methods (see
``baseline_v23.py``) so the v3 sim summaries are directly comparable to v2:

  - mean cancer incidence: per-year (cancers / at-risk-female × 100k),
    averaged across years and (for aggregate) across genotypes. v2 reports
    `cancer_incidence_by_genotype` at annual cadence (resfreq=4 with
    dt=0.25), see _v2_legacy/sim.py:1108-1109.
  - mean age of infection / cancer: lifetime reconstruction including both
    alive ever-evented agents (current age − years since event) AND dead-
    of-cancer agents (frozen-at-death age − years between event and death).
    Mirrors _v2_legacy/people.py + baseline_v23._lifetime_mean_age.
  - mean age of cancer death: only actually-died-of-cancer agents.
    people.age is frozen at death so it equals age at cancer death directly.
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


def _earliest_ti_dead_cancer(sim, genotypes):
    """Per-uid earliest cancer-death step across all genotypes.

    Each agent dies of cancer in at most one genotype; this min handles the
    edge case where multiple genotypes scheduled deaths and one fired first.
    Returns an ndarray of length ``n_uids`` with NaN for non-cancer-deaths.
    """
    n = len(sim.people.alive.raw)
    earliest = np.full(n, np.nan)
    for g in genotypes:
        mod = sim.diseases[g]
        td = np.asarray(mod.ti_dead_cancer.raw)
        valid = ~np.isnan(td)
        update = valid & (np.isnan(earliest) | (td < earliest))
        earliest[update] = td[update]
    return earliest


def _lifetime_mean_age_at_event(sim, ti_event_arr, ti_dead_cancer_any):
    """v2-compatible lifetime reconstruction of mean age at event.

    Includes alive ever-evented agents (current age − years since event) AND
    dead-of-cancer agents (frozen-at-death age − years between event and
    cancer-death). Filters to ``ti_event <= end_ti`` so scheduled-but-not-
    yet-realised events (e.g. ``ti_cancerous`` set at CIN onset) don't bias
    the mean upward.

    Mirrors _v2_legacy/people.py logic via baseline_v23._lifetime_mean_age.
    Returns (sum_ages, count) so aggregate can pool across genotypes.
    """
    end_ti = float(sim.t.ti)
    dt = float(sim.t.dt)

    age_arr = np.asarray(sim.people.age.raw)
    alive_arr = np.asarray(sim.people.alive.raw).astype(bool)
    ti_event_arr = np.asarray(ti_event_arr)

    alive_mask = alive_arr & ~np.isnan(ti_event_arr) & (ti_event_arr <= end_ti)
    dead_mask = (
        (~alive_arr)
        & ~np.isnan(ti_event_arr)
        & ~np.isnan(ti_dead_cancer_any)
        & (ti_event_arr <= end_ti)
    )

    ages = []
    if alive_mask.any():
        years_since = (end_ti - ti_event_arr[alive_mask]) * dt
        ages.append(age_arr[alive_mask] - years_since)
    if dead_mask.any():
        years_since_d = (
            ti_dead_cancer_any[dead_mask] - ti_event_arr[dead_mask]
        ) * dt
        ages.append(age_arr[dead_mask] - years_since_d)
    if not ages:
        return 0.0, 0
    all_ages = np.concatenate(ages)
    valid = (all_ages > 0) & (all_ages < 100)
    return float(all_ages[valid].sum()), int(valid.sum())


def _per_year_buckets(per_step_series, dt):
    """Reshape a per-step series into per-year sums (or means).

    Returns a tuple of (per_year_sum, per_year_mean, n_full_years). Drops
    any trailing partial year. v2's resfreq aggregation produces per-year
    results; this mirrors that for v3's per-step results.
    """
    arr = np.asarray(per_step_series)
    steps_per_year = int(round(1.0 / dt))
    n_full = len(arr) // steps_per_year
    if n_full == 0:
        return np.array([]), np.array([]), 0
    truncated = arr[:n_full * steps_per_year].reshape(n_full, steps_per_year)
    return truncated.sum(axis=1), truncated.mean(axis=1), n_full


def _per_genotype_cancer_incidence(sim, genotype):
    """v2-compatible per-genotype cancer incidence: mean of per-year rates.

    Per-year rate = (cancers_in_year / at_risk_females_in_year) * 100k.
    At-risk females approximated as alive/2 (v2 uses alive_f - n_cancerous_f
    but the latter is small relative to alive_f, ≤0.1% in our regime).
    Mean across all full years.
    """
    dt = float(sim.t.dt)
    res = sim.results[genotype]
    new_cancers = np.asarray(res.new_cancers).astype(float)
    n_alive = np.asarray(sim.results['n_alive']).astype(float)

    cancers_per_year, _, n_full = _per_year_buckets(new_cancers, dt)
    _, alive_per_year, _ = _per_year_buckets(n_alive, dt)
    if n_full == 0:
        return 0.0

    female_per_year = alive_per_year / 2.0
    # safedivide: if denom is 0, rate is 0 (v2's safedivide convention)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_per_year = np.where(
            female_per_year > 0,
            cancers_per_year * 100_000.0 / female_per_year,
            0.0,
        )
    return float(rate_per_year.mean())


def _per_genotype_metrics(sim, genotype, genotypes_for_death=None):
    """Compute the 8-metric M02 summary for one genotype (key into sim.results)."""
    res = sim.results[genotype]
    mod = sim.diseases[genotype]
    pop_scale = float(getattr(sim.pars, 'pop_scale', 1.0) or 1.0)

    new_infections = np.asarray(res.new_infections)
    n_inf_unscaled = float(new_infections.sum())
    n_inf = n_inf_unscaled * pop_scale

    mean_prev_pct = 100 * float(np.asarray(res.prevalence).mean())

    new_cancers = np.asarray(res.new_cancers)
    n_cancers_unscaled = float(new_cancers.sum())
    n_cancers = n_cancers_unscaled * pop_scale

    new_cancer_deaths = np.asarray(res.new_cancer_deaths)
    n_cd_unscaled = float(new_cancer_deaths.sum())
    n_cancer_deaths = n_cd_unscaled * pop_scale

    # Cancer death age: people.age is frozen at death, so the per-step
    # accumulator already records age-at-cancer-death directly.
    sum_age_cd = float(np.asarray(res.sum_age_at_cancer_death).sum())
    mean_age_cancer_death = (sum_age_cd / n_cd_unscaled) if n_cd_unscaled > 0 else 0.0

    # Lifetime mean age of infection / cancer onset (v2-compatible).
    if genotypes_for_death is None:
        genotypes_for_death = (genotype,)
    ti_dead_any = _earliest_ti_dead_cancer(sim, genotypes_for_death)
    ti_inf_arr = np.asarray(mod.ti_infected.raw)
    s_inf, n_inf_count = _lifetime_mean_age_at_event(sim, ti_inf_arr, ti_dead_any)
    mean_age_inf = (s_inf / n_inf_count) if n_inf_count > 0 else 0.0
    ti_can_arr = np.asarray(mod.ti_cancerous.raw)
    s_can, n_can_count = _lifetime_mean_age_at_event(sim, ti_can_arr, ti_dead_any)
    mean_age_cancer = (s_can / n_can_count) if n_can_count > 0 else 0.0

    # Mean cancer incidence: per-year rate, averaged across years (v2 cadence).
    mean_cancer_incidence = _per_genotype_cancer_incidence(sim, genotype)

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
    """Compute the 8-metric aggregate across genotypes.

    Aggregate count metrics come from the Aggregate analyzer
    (sim.results.aggregate.cum_*_any). Mean ages pool per-genotype lifetime
    reconstructions. Mean cancer incidence is the mean across genotypes of
    the per-genotype mean-of-per-year rates (matches v2's ``any.mean cancer
    incidence`` aggregation; see _v2_legacy/sim.py:1109).
    """
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

    # Mean cancer incidence: mean across genotypes of per-genotype mean rates.
    per_genotype_rates = [_per_genotype_cancer_incidence(sim, g) for g in genotypes]
    mean_cancer_incidence = (
        float(np.mean(per_genotype_rates)) if per_genotype_rates else 0.0
    )

    # Pool per-genotype lifetime mean-age sums and counts (v2-compatible).
    ti_dead_any = _earliest_ti_dead_cancer(sim, genotypes)
    sum_age_inf_total = 0.0; n_inf_count_total = 0
    sum_age_cancer_total = 0.0; n_can_count_total = 0
    sum_age_cd_total = 0.0; n_cd_count_total = 0.0
    for g in genotypes:
        mod = sim.diseases[g]
        s_inf, c_inf = _lifetime_mean_age_at_event(
            sim, np.asarray(mod.ti_infected.raw), ti_dead_any
        )
        sum_age_inf_total += s_inf; n_inf_count_total += c_inf
        s_can, c_can = _lifetime_mean_age_at_event(
            sim, np.asarray(mod.ti_cancerous.raw), ti_dead_any
        )
        sum_age_cancer_total += s_can; n_can_count_total += c_can
        sum_age_cd_total += float(np.asarray(sim.results[g].sum_age_at_cancer_death).sum())
        n_cd_count_total += float(np.asarray(sim.results[g].new_cancer_deaths).sum())

    mean_age_inf = (sum_age_inf_total / n_inf_count_total) if n_inf_count_total > 0 else 0.0
    mean_age_cancer = (sum_age_cancer_total / n_can_count_total) if n_can_count_total > 0 else 0.0
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
        per = _per_genotype_metrics(sim, g, genotypes_for_death=genotypes)
        for k, v in per.items():
            out[f'{g}.{k}'] = v
    agg = _aggregate_metrics(sim, genotypes)
    for k, v in agg.items():
        out[f'any.{k}'] = v
    return out