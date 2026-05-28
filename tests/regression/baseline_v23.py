"""Regenerate v2.3 baselines for the HPV16 (M02) and 4-genotype (M03) anchor scenarios.

Run this script INSIDE a Python environment that has hpvsim==2.3.x installed
(e.g. the local hpvsim_v23_frozen clone, or a fresh `pip install hpvsim==2.3`
venv). The v3 active package is NOT used here — only the v2.3 hpvsim API.

Two regen entrypoints are provided:

  regen_hpv16()  — M02 single-genotype HPV16 baseline (original).
                   Output: tests/regression_baselines/anchor_hpv16.json

  regen_4genotype() — M03 4-genotype baseline (new, Task 14).
                   Outputs:
                     tests/regression_baselines/anchor_4genotype.json
                       40-entry per-genotype + aggregate summary (8 metrics x
                       4 genotypes + 8 aggregate, keys: ``<g>.<metric>`` and
                       ``any.<metric>``).
                     tests/regression_baselines/anchor_4genotype_trajectory.json
                       time-series of cum_cancers and cum_infections for the
                       trajectory parity test.

Key v2-vs-v3 syntax differences honored here:
  - v3 PARS uses ``genotype='hpv16'``; v2 expects ``genotypes=['hpv16']``.
  - v3 PARS uses ``stop=2060``; v2 expects ``end=2060``.
  - dt=0.25 matches v2's default sim timestep (``_v2_legacy/parameters.py:61``)
    and v3's M02 anchor.
  - pop_scale=1 + total_pop=10_000 disables v2's real-world scaling so count
    metrics (total infections, cancers, cancer deaths) come out at the
    agent-level — matching v3's default (where pop_scale defaults to 1.0).

The PARS dicts here disable v2 features that M03 hasn't yet implemented
(currently: ``eff_condoms=0``). This keeps the v2 baseline and v3 anchor
simulations on equivalent footing — both run with the same effective set
of features active. As M03 implements more v2 features, the corresponding
PARS overrides here can be removed.

Usage (from a v2.3 env, with cwd at the repo root):

    # M02 single-genotype HPV16 baseline (default):
    python tests/regression/baseline_v23.py
    python tests/regression/baseline_v23.py --genotypes hpv16
    python tests/regression/baseline_v23.py --out custom/path/anchor.json

    # M03 4-genotype baseline:
    python tests/regression/baseline_v23.py --genotypes four
    python tests/regression/baseline_v23.py --genotypes four --out custom/out.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sciris as sc
import hpvsim as hpv  # v2.3 here


def _actual_date_dead_v2(people):
    """Per-agent step at which the agent actually died, regardless of cause.

    v2 records cause-specific death dates: date_dead_cancer, date_dead_other,
    date_dead_hiv. An agent dies of at most one cause, so at most one is set
    on any given agent — but we min across all to be defensive in case a
    cause-specific date is left scheduled-in-the-future on an agent who
    died of another cause earlier.

    Returns ndarray of length n_uids with NaN for still-alive agents.
    """
    cols = []
    for attr in ('date_dead_cancer', 'date_dead_other', 'date_dead_hiv'):
        if hasattr(people, attr):
            cols.append(np.asarray(getattr(people, attr), dtype=float))
    if not cols:
        return np.full(len(people.alive), np.nan)
    stacked = np.vstack(cols)
    # All-NaN columns (still-alive agents) intentionally yield NaN; suppress
    # the harmless "All-NaN slice" warning that np.nanmin emits in that case.
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmin(stacked, axis=0)


# ---------------------------------------------------------------------------
# M02 — single-genotype HPV16 anchor PARS
# Pinned anchor PARS — must match tests/regression/anchor_hpv16.py:PARS
# at v3 except for v2's API name differences and pop_scale handling.
# ---------------------------------------------------------------------------

PARS = dict(
    n_agents=10_000,
    location='nigeria',
    genotypes=['hpv16'],     # v2 takes a list
    start=1990,
    end=2060,                # v2 calls it 'end', not 'stop'
    dt=0.25,                 # matches v2's default
    rand_seed=0,
    verbose=0,
    pop_scale=1,             # disable real-world scaling (match v3 default)
    total_pop=10_000,        # match n_agents so pop_scale stays 1
    ms_agent_ratio=1,        # disable multiscale dynamic spawning (v2 default
                             # is 10; v3 M02 spec defers multiscale, so v2
                             # baseline must be regenerated with ms_agent_ratio=1
                             # for an apples-to-apples cancer-count comparison.
                             # See _v2_legacy/parameters.py:38 + people.py:280-371.)
    eff_condoms=0,           # Disable condom modulation entirely (M03 does not yet
                             # implement condoms; setting eff_condoms=0 makes v2's
                             # per-layer condoms[m]=0.01, condoms[c]=0.2 a no-op).
                             # See _v2_legacy/sim.py:806,814 — effective_condoms =
                             # condoms[lkey] * eff_condoms is multiplied into
                             # (1 - .) inside the per-act prob, so eff_condoms=0
                             # makes (1 - 0) = 1 → no reduction.
)


# ---------------------------------------------------------------------------
# M03 — 4-genotype anchor PARS
# Matches tests/regression/anchor_4genotype.py:PARS but with v2 API names.
# Genotype strings: v2 normalises '16' -> 'hpv16', 'hi5hpv' -> 'hi5', etc.
# (see _v2_legacy/parameters.py:268-273). We pass the canonical string forms.
# ---------------------------------------------------------------------------

PARS_4GENOTYPE = dict(
    n_agents=10_000,
    location='nigeria',
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],  # v2 takes a list; 4 genotypes
    start=1990,
    end=2060,                # v2 calls it 'end', not 'stop'
    dt=0.25,                 # matches v2 default (_v2_legacy/parameters.py:61)
    rand_seed=0,
    verbose=0,
    pop_scale=1,             # disable real-world scaling (match v3 default)
    total_pop=10_000,        # match n_agents so pop_scale stays 1
    ms_agent_ratio=1,        # disable multiscale (see PARS comment above)
    eff_condoms=0,           # Disable condom modulation entirely (M03 does not yet
                             # implement condoms; setting eff_condoms=0 makes v2's
                             # per-layer condoms[m]=0.01, condoms[c]=0.2 a no-op).
                             # See _v2_legacy/sim.py:806,814 — effective_condoms =
                             # condoms[lkey] * eff_condoms is multiplied into
                             # (1 - .) inside the per-act prob, so eff_condoms=0
                             # makes (1 - 0) = 1 → no reduction.
)

# ---------------------------------------------------------------------------
# M01 — single-genotype HPV16 transmission-only anchor PARS
# Pinned to match tests/regression/anchor_m01.py:PARS at v3, with v2's API
# differences applied (genotypes is a list; v2 calls 'stop' -> 'end').
# Transmission-only: we do not disable progression knobs explicitly because v2
# does not expose a single "disable progression" toggle; the M01 v2 baseline
# is generated with v2's default progression machinery active. The M01 summary
# only reports infections + prevalence + total population, so the cancer
# machinery's presence does not affect the M01 metrics.
# ---------------------------------------------------------------------------

PARS_HPV16_TRANSMISSION_ONLY = dict(
    n_agents=10_000,
    location='nigeria',
    genotypes=['hpv16'],
    start=1990,
    end=2030,                  # v2 calls it 'end', not 'stop'
    dt=0.25,
    rand_seed=0,
    verbose=0,
    pop_scale=1,
    total_pop=10_000,
    ms_agent_ratio=1,
    eff_condoms=0,
)


# Alias for compatibility with M07 callers that import the M02 anchor PARS
# by its milestone-qualified name.
PARS_HPV16 = PARS


def _summary_v2_m01(sim, gen_idx, gen_key):
    """v2-side M01 summary: 3 transmission-only metrics matching
    short_summary_m01.METRIC_KEYS_M01."""
    import numpy as np
    res = sim.results
    pop_scale = float(sim.pars.get('pop_scale', 1.0) or 1.0)
    inf_arr = np.asarray(res['infections'][gen_idx, :], dtype=float)
    n_inf = float(inf_arr.sum()) * pop_scale
    prev_arr = np.asarray(res['hpv_prevalence'][gen_idx, :], dtype=float)
    mean_prev_pct = 100.0 * float(prev_arr.mean())
    total_pop = float(np.asarray(res['n_alive'])[-1])
    return {
        'total HPV infections': n_inf,
        'mean HPV prevalence (%)': mean_prev_pct,
        'total population': total_pop,
    }


# Canonical genotype key order that v2 stores in genotype_map after normalisation.
# v2 parameters.py:268-273 maps '16' -> 'hpv16', 'hi5hpv' -> 'hi5', etc.
# For these four we pass the canonical strings directly, so the genotype_map
# will be {0: 'hpv16', 1: 'hpv18', 2: 'hi5', 3: 'ohr'}.
_GENOTYPE_KEYS_4 = ('hpv16', 'hpv18', 'hi5', 'ohr')

# Metric keys — must be character-identical to short_summary.METRIC_KEYS so
# that Tasks 15 and 16 can compare v2 JSON against v3 results directly.
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

_EXPECTED_KEYS = METRIC_KEYS  # alias used in M02 path


# ---------------------------------------------------------------------------
# M02 helpers (unchanged from original)
# ---------------------------------------------------------------------------

def run_and_summarize():
    """Run the v2.3 anchor sim and return (short_summary_dict, total_pop)."""
    sim = hpv.Sim(sc.dcp(PARS))
    sim.run()

    # v2's compute_summary populates sim.short_summary with the 8 keys
    # listed above (see _v2_legacy/sim.py:1157 -> 1194).
    sim.compute_summary()
    s = dict(sim.short_summary)

    # Sanity-check we got the keys we expect.
    missing = [k for k in _EXPECTED_KEYS if k not in s]
    if missing:
        raise RuntimeError(f'v2 compute_summary missing keys: {missing}; '
                           f'got {list(s.keys())}')

    short = {k: float(s[k]) for k in _EXPECTED_KEYS}

    # Override age metrics: v2's compute_summary uses compute_age_mean(...,
    # t=-1) for all three — age-weighted mean of NEW events in the LAST sim
    # step only, not a lifetime mean. v3's anchor computes lifetime mean ages
    # from per-agent ti/sum_age accumulators. Recompute the v2 baseline values
    # with matching lifetime semantics for a like-for-like comparison.
    people = sim.people
    end_ti = float(sim.t)
    dt = float(PARS['dt'])
    alive_arr = np.asarray(people.alive).astype(bool)
    age_arr = np.asarray(people.age)
    date_dead_cancer = np.asarray(people.date_dead_cancer)

    # v2 freezes people.age at death. For computing lifetime mean ages at past
    # events (matching v3's sum_age_at_<event> accumulators), we need to include
    # BOTH alive agents (people.age = current) AND dead agents (people.age =
    # age at death, frozen). Reference time differs:
    #   alive:        end_ti
    #   dead-of-cancer: date_dead_cancer (frozen-at-death age was set then)
    # Dead-of-other-causes have no genotype-attributed cancer-onset to count.

    def _lifetime_mean_age_at_event(date_event):
        """Lifetime mean age at the event across alive + dead-of-cancer agents.

        Filters to date_event <= end_ti to exclude agents whose event (e.g.
        date_cancerous) is a scheduled future ti set at CIN diagnosis but not
        yet realised. Without this filter, alive CIN-only agents project ages
        into the future, biasing the mean upward.
        """
        # Alive ever-evented agents whose event has already occurred.
        alive_mask = alive_arr & ~np.isnan(date_event) & (date_event <= end_ti)
        # Dead-of-cancer ever-evented agents whose event has already occurred.
        dead_mask = (~alive_arr) & ~np.isnan(date_event) & ~np.isnan(date_dead_cancer) & (date_event <= end_ti)
        ages = []
        if alive_mask.any():
            years_since = (end_ti - date_event[alive_mask]) * dt
            ages.append(age_arr[alive_mask] - years_since)
        if dead_mask.any():
            years_since_dead = (date_dead_cancer[dead_mask] - date_event[dead_mask]) * dt
            ages.append(age_arr[dead_mask] - years_since_dead)
        if not ages:
            return 0.0
        all_ages = np.concatenate(ages)
        valid = (all_ages > 0) & (all_ages < 100)
        return float(all_ages[valid].mean()) if valid.any() else 0.0

    short['mean age of infection (years)'] = _lifetime_mean_age_at_event(
        np.asarray(people.date_infectious[0])
    )
    short['mean age of cancer (years)'] = _lifetime_mean_age_at_event(
        np.asarray(people.date_cancerous[0])
    )
    # mean age of cancer death: only actual past deaths; people.age is age at death
    actually_died = (~alive_arr) & ~np.isnan(date_dead_cancer) & (date_dead_cancer <= end_ti)
    if actually_died.any():
        ages_at_death = age_arr[actually_died]
        valid_d = (ages_at_death > 0) & (ages_at_death < 100)
        short['mean age of cancer death (years)'] = (
            float(ages_at_death[valid_d].mean()) if valid_d.any() else 0.0
        )
    else:
        short['mean age of cancer death (years)'] = 0.0

    # Total population at end of sim.
    if 'n_alive' in sim.results:
        total_pop = float(sim.results['n_alive'][-1])
    else:
        total_pop = float(sim.results['pop_size'][-1])

    return short, total_pop


def build_baseline():
    short, total_pop = run_and_summarize()
    return {
        'metadata': {
            'hpvsim_version': hpv.__version__,
            'pars': dict(PARS),
        },
        'summary': {
            **short,
            'total population': total_pop,
        },
    }


# ---------------------------------------------------------------------------
# M03 helpers — per-genotype and aggregate metrics from v2's _by_genotype arrays
# ---------------------------------------------------------------------------

def _per_genotype_metrics_v2(sim, gen_idx, gen_key):
    """Compute the 8-metric M02 summary for ONE genotype using v2's results.

    Uses v2's ``_by_genotype`` result arrays (shape ``(n_g, n_t)``) and
    per-agent date arrays for lifetime mean-age computation (since v2's
    ``compute_age_mean`` is aggregate-only, see _v2_legacy/sim.py:1146).

    Args:
        sim      : completed v2 Sim object.
        gen_idx  : integer index of this genotype in genotype_map (0-based).
        gen_key  : string key for this genotype (e.g. 'hpv16').

    Returns:
        dict with exactly the 8 METRIC_KEYS entries.
    """
    res = sim.results
    dt = float(sim.pars['dt'])

    # --- Total infections for this genotype ---
    # infections_by_genotype shape: (n_g, n_t); row gen_idx = new infections
    # per step for this genotype. Sum across time = lifetime count.
    # (see _v2_legacy/sim.py:860-865 for how genotype_flows are accumulated)
    infections_g = np.asarray(res['infections_by_genotype'][gen_idx])
    n_inf = float(infections_g.sum())

    # --- HPV prevalence (%) for this genotype ---
    # hpv_prevalence_by_genotype shape: (n_g, n_t)
    # (_v2_legacy/sim.py:1082: n_infectious_by_genotype / n_alive)
    prev_g = np.asarray(res['hpv_prevalence_by_genotype'][gen_idx])
    mean_prev_pct = 100.0 * float(prev_g.mean())

    # --- Cancers for this genotype ---
    # cancers_by_genotype shape: (n_g, n_t)
    cancers_g = np.asarray(res['cancers_by_genotype'][gen_idx])
    n_cancers = float(cancers_g.sum())

    # --- Cancer deaths attributed to this genotype ---
    # v2 stores cancer_deaths only in aggregate (no _by_genotype variant) and
    # date_dead_cancer has no genotype dim. Empirically date_cancerous[g] is
    # set for more than just the driving genotype, so per-agent attribution
    # over-counts (sum across genotypes >> aggregate). Use proportional
    # allocation: split aggregate cancer_deaths by each genotype's share of
    # aggregate cancers. This is the cleanest deterministic split that
    # preserves the aggregate total.
    agg_cancer_deaths = float(np.asarray(res['cancer_deaths']).sum())
    agg_cancers = float(np.asarray(res['cancers_by_genotype']).sum())
    n_cancer_deaths = (
        (n_cancers / agg_cancers) * agg_cancer_deaths if agg_cancers > 0 else 0.0
    )

    # --- Cancer incidence (per 100k) ---
    # Use per-genotype cancer_incidence_by_genotype (already per-100k, see
    # _v2_legacy/sim.py:1109).  Mean across time steps = mean annual
    # incidence rate.
    cancer_inc_g = np.asarray(res['cancer_incidence_by_genotype'][gen_idx])
    mean_cancer_incidence = float(cancer_inc_g.mean())

    # --- Lifetime mean ages: alive-only reconstruction matching v3 anchor ---
    # v3 anchor uses ti_infected/sum_age_at_cancer/sum_age_at_cancer_death and
    # divides by counts; that's lifetime mean across actual events. For v2
    # parity (people.age is frozen at death), we filter to alive agents for
    # mean-age-of-infection and mean-age-of-cancer (people.age is current),
    # and to actually-died-of-cancer agents for mean-age-of-cancer-death
    # (people.age = age at death, since v2 freezes it at death event).
    people = sim.people
    end_ti = float(sim.t)
    alive_arr = np.asarray(people.alive).astype(bool)
    age_arr = np.asarray(people.age)

    # Actual death time per agent: min of date_dead_cancer / date_dead_other /
    # date_dead_hiv (whichever fired). NaN for still-alive agents. Used below
    # to filter phantom events — scheduled-but-never-realized cancer or
    # infection events on agents who died of another cause first.
    date_dead_any = _actual_date_dead_v2(people)

    def _lifetime_mean_age(date_event):
        """Mean age at event across alive + dead agents whose event fired.

        For alive: ref = end_ti. For dead: ref = date_dead_any (the agent's
        actual death step). Requires date_event <= date_dead_any so phantom
        events (scheduled in set_severity but never realized because the
        agent died of another cause first) are excluded.
        """
        alive_mask = alive_arr & ~np.isnan(date_event) & (date_event <= end_ti)
        dead_mask = (
            (~alive_arr)
            & ~np.isnan(date_event)
            & ~np.isnan(date_dead_any)
            & (date_event <= date_dead_any)
            & (date_event <= end_ti)
        )
        ages = []
        if alive_mask.any():
            years_since = (end_ti - date_event[alive_mask]) * dt
            ages.append(age_arr[alive_mask] - years_since)
        if dead_mask.any():
            years_since_d = (date_dead_any[dead_mask] - date_event[dead_mask]) * dt
            ages.append(age_arr[dead_mask] - years_since_d)
        if not ages:
            return 0.0
        all_ages = np.concatenate(ages)
        valid = (all_ages > 0) & (all_ages < 100)
        return float(all_ages[valid].mean()) if valid.any() else 0.0

    mean_age_inf    = _lifetime_mean_age(np.asarray(people.date_infectious[gen_idx]))
    mean_age_cancer = _lifetime_mean_age(np.asarray(people.date_cancerous[gen_idx]))

    # mean age of cancer death: only agents who actually died of cancer in
    # this genotype. Filter via date_cancerous[gen_idx] set AND not alive AND
    # date_dead_cancer in past. people.age is age at death (frozen).
    date_dead = np.asarray(people.date_dead_cancer)
    date_can_g_for_death = np.asarray(people.date_cancerous[gen_idx])
    actually_died_g = (
        (~alive_arr)
        & ~np.isnan(date_dead)
        & ~np.isnan(date_can_g_for_death)
        & (date_dead <= end_ti)
    )
    if actually_died_g.any():
        ages_at_death = age_arr[actually_died_g]
        valid_d = (ages_at_death > 0) & (ages_at_death < 100)
        mean_age_cancer_death = (
            float(ages_at_death[valid_d].mean()) if valid_d.any() else 0.0
        )
    else:
        mean_age_cancer_death = 0.0

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


def _aggregate_metrics_v2(sim, genotype_keys):
    """Compute the 8-metric aggregate across all genotypes using v2's results.

    v2 has no built-in cum_infections / cum_cancers aggregators (those are v3
    M03 additions, via the HPVTotal analyzer).  We build them from v2's
    ``_by_genotype`` arrays:

      * total HPV infections = infections_by_genotype.sum() (all genotypes, all t)
        This is the correct per-step new-infection count summed across time and
        genotypes.  In the small-co-infection regime it approximates unique
        infections, matching the v3 HPVTotal analyzer's cum_infections (which
        also sums new_infections rather than counting unique agents).
      * total cancers = cancers_by_genotype.sum() (cancer is single-attributed)
      * total cancer deaths = aggregate cancer_deaths.sum() (v2 doesn't stratify
        cancer_deaths by genotype; this is the population-wide total)
      * mean HPV prevalence = mean of per-genotype prevalence series * 100
      * mean cancer incidence = mean of per-genotype cancer_incidence_by_genotype
        (already per-100k in v2, see _v2_legacy/sim.py:1109)
      * mean age metrics = pool per-genotype per-agent date arrays, weighted
        by counts (sum of sum_age / sum of count across genotypes).

    Args:
        sim           : completed v2 Sim object.
        genotype_keys : sequence of (gen_idx, gen_key) tuples in genotype order.
    """
    res = sim.results
    dt = float(sim.pars['dt'])
    people = sim.people
    end_year = float(sim.yearvec[-1])
    start_year = float(sim.pars['start'])

    # --- Totals from _by_genotype flow arrays ---
    inf_bg = np.asarray(res['infections_by_genotype'])    # (n_g, n_t)
    can_bg = np.asarray(res['cancers_by_genotype'])        # (n_g, n_t)

    n_inf         = float(inf_bg.sum())
    n_cancers     = float(can_bg.sum())
    # v2 doesn't stratify cancer_deaths by genotype — use the aggregate.
    n_cancer_deaths = float(np.asarray(res['cancer_deaths']).sum())

    # --- Mean prevalence: average across genotypes' time-series ---
    prev_bg = np.asarray(res['hpv_prevalence_by_genotype'])  # (n_g, n_t)
    mean_prev_pct = 100.0 * float(prev_bg.mean())

    # --- Mean cancer incidence: average across genotypes' time-series ---
    # cancer_incidence_by_genotype is already per-100k (_v2_legacy/sim.py:1109)
    ci_bg = np.asarray(res['cancer_incidence_by_genotype'])  # (n_g, n_t)
    mean_cancer_incidence = float(ci_bg.mean())

    # --- Pool per-genotype per-agent date arrays for mean-age metrics ---
    # Match v3 anchor's lifetime semantics: alive-only for infection/cancer
    # onset (people.age is current); actually-died-of-cancer for cancer death
    # (people.age is frozen at death).
    end_ti = float(sim.t)
    alive_arr = np.asarray(people.alive).astype(bool)
    age_arr = np.asarray(people.age)
    # date_dead is kept for the cancer-death filter below (per-cause check);
    # date_dead_any is the actual death time used in the lifetime accumulator.
    date_dead = np.asarray(people.date_dead_cancer)
    date_dead_any = _actual_date_dead_v2(people)

    sum_age_inf  = 0.0; n_inf_count  = 0
    sum_age_can  = 0.0; n_can_count  = 0
    sum_age_dead = 0.0; n_dead_count = 0

    def _accum_lifetime(date_event):
        """Accumulate ages-at-event across alive + dead agents whose event fired.

        Phantom events (scheduled in set_severity but never realized because
        the agent died of another cause first) are excluded via the
        ``date_event <= date_dead_any`` filter.
        Returns (sum_ages, n_valid).
        """
        alive_mask = alive_arr & ~np.isnan(date_event) & (date_event <= end_ti)
        dead_mask = (
            (~alive_arr)
            & ~np.isnan(date_event)
            & ~np.isnan(date_dead_any)
            & (date_event <= date_dead_any)
            & (date_event <= end_ti)
        )
        ages = []
        if alive_mask.any():
            years_since = (end_ti - date_event[alive_mask]) * dt
            ages.append(age_arr[alive_mask] - years_since)
        if dead_mask.any():
            years_since_d = (date_dead_any[dead_mask] - date_event[dead_mask]) * dt
            ages.append(age_arr[dead_mask] - years_since_d)
        if not ages:
            return 0.0, 0
        all_ages = np.concatenate(ages)
        valid = (all_ages > 0) & (all_ages < 100)
        return float(all_ages[valid].sum()), int(valid.sum())

    for gen_idx, gen_key in genotype_keys:
        s_inf, n_inf_g = _accum_lifetime(np.asarray(people.date_infectious[gen_idx]))
        sum_age_inf += s_inf;  n_inf_count += n_inf_g

        date_can_g = np.asarray(people.date_cancerous[gen_idx])
        s_can, n_can_g = _accum_lifetime(date_can_g)
        sum_age_can += s_can;  n_can_count += n_can_g

        # Mean age of cancer death — actually-died-of-cancer in this genotype.
        # people.age is frozen at death == age at cancer death directly.
        actually_died_g = (
            (~alive_arr)
            & ~np.isnan(date_dead)
            & ~np.isnan(date_can_g)
            & (date_dead <= end_ti)
        )
        if actually_died_g.any():
            ages_at_death = age_arr[actually_died_g]
            valid_d = (ages_at_death > 0) & (ages_at_death < 100)
            sum_age_dead += float(ages_at_death[valid_d].sum())
            n_dead_count += int(valid_d.sum())

    mean_age_inf          = (sum_age_inf  / n_inf_count)  if n_inf_count  > 0 else 0.0
    mean_age_cancer       = (sum_age_can  / n_can_count)  if n_can_count  > 0 else 0.0
    mean_age_cancer_death = (sum_age_dead / n_dead_count) if n_dead_count > 0 else 0.0

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


# ---------------------------------------------------------------------------
# M03 regen entrypoint
# ---------------------------------------------------------------------------

_DEFAULT_OUT_4G = (
    Path(__file__).resolve().parent.parent
    / 'regression_baselines' / 'anchor_4genotype.json'
)
_DEFAULT_TRAJ_4G = (
    Path(__file__).resolve().parent.parent
    / 'regression_baselines' / 'anchor_4genotype_trajectory.json'
)


def regen_4genotype(out=None, traj_out=None):
    """Regenerate the M03 4-genotype v2 baseline and trajectory JSONs.

    Builds PARS_4GENOTYPE sim, runs it, computes the 40-entry per-genotype +
    aggregate summary (8 metrics x 4 genotypes + 8 aggregate), and writes:

      * ``out``      — 40-entry summary JSON (default: anchor_4genotype.json)
      * ``traj_out`` — trajectory JSON with cum_cancers and cum_infections
                       time-series (default: anchor_4genotype_trajectory.json)

    Both files land in ``tests/regression_baselines/`` which is gitignored.

    DO NOT run in the v3 env — this requires hpvsim==2.3.x.
    Tasks 15 and 16 read these files and skip gracefully if absent.
    """
    if out is None:
        out = _DEFAULT_OUT_4G
    if traj_out is None:
        traj_out = _DEFAULT_TRAJ_4G

    sim = hpv.Sim(sc.dcp(PARS_4GENOTYPE))
    sim.run()

    # v2 stores the canonical genotype key order in genotype_map
    # (dict mapping gen_idx -> gen_key string, e.g. {0: 'hpv16', ...}).
    # We rely on insertion order (Python 3.7+) being consistent with the
    # genotypes list we passed in.  See _v2_legacy/parameters.py:358-360.
    genotype_map = sim.pars['genotype_map']   # {0: 'hpv16', 1: 'hpv18', ...}
    genotype_pairs = [(i, k) for i, k in sorted(genotype_map.items())]

    # Build the 40-entry summary dict.
    # Keys: '<gen_key>.<metric>' for per-genotype (32 entries)
    #       'any.<metric>'       for aggregate    (8 entries)
    # Metric strings must match METRIC_KEYS / short_summary.METRIC_KEYS exactly.
    summary = {}
    for gen_idx, gen_key in genotype_pairs:
        per = _per_genotype_metrics_v2(sim, gen_idx, gen_key)
        for k, v in per.items():
            summary[f'{gen_key}.{k}'] = float(v)

    agg = _aggregate_metrics_v2(sim, genotype_pairs)
    for k, v in agg.items():
        summary[f'any.{k}'] = float(v)

    # Sanity-check key count.
    n_g = len(genotype_pairs)
    expected_n = (n_g + 1) * len(METRIC_KEYS)
    if len(summary) != expected_n:
        raise RuntimeError(
            f'Expected {expected_n} summary keys, got {len(summary)}: '
            f'{list(summary.keys())}'
        )

    # Total population at end of sim.
    if 'n_alive' in sim.results:
        total_pop = float(sim.results['n_alive'][-1])
    else:
        total_pop = float(sim.results['pop_size'][-1])

    baseline = {
        'metadata': {
            'hpvsim_version': hpv.__version__,
            'pars': dict(PARS_4GENOTYPE),
            'genotype_map': {str(i): k for i, k in genotype_map.items()},
        },
        'summary': {
            **summary,
            'total population': total_pop,
        },
    }

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f'Wrote v2.3 4-genotype baseline ({hpv.__version__}) to {out}')
    print('Summary (40 entries):')
    for k, v in summary.items():
        print(f'  {k:<56} {v:>12.4g}')

    # --- Trajectory JSON for cross-version comparison ---
    # cum_infections: cumsum of per-step new infections across all genotypes.
    #   We sum infections_by_genotype across genotypes (axis 0) to get per-step
    #   totals, then cumsum across time.  Matches the v3 HPVTotal analyzer's
    #   cum_infections = cumsum(new_infections_summed_across_genotypes).
    # cum_cancers: cumsum of cancers_by_genotype summed across genotypes.
    inf_bg  = np.asarray(sim.results['infections_by_genotype'])  # (n_g, n_t)
    can_bg  = np.asarray(sim.results['cancers_by_genotype'])      # (n_g, n_t)

    cum_infections = np.cumsum(inf_bg.sum(axis=0))
    cum_cancers    = np.cumsum(can_bg.sum(axis=0))

    trajectory = {
        'metadata': {
            'hpvsim_version': hpv.__version__,
            'pars': dict(PARS_4GENOTYPE),
        },
        'time':            sim.yearvec.tolist(),
        'cum_infections':  cum_infections.tolist(),
        'cum_cancers':     cum_cancers.tolist(),
    }

    traj_out = Path(traj_out)
    traj_out.parent.mkdir(parents=True, exist_ok=True)
    with open(traj_out, 'w') as f:
        json.dump(trajectory, f, indent=2)
    print(f'Wrote trajectory baseline to {traj_out}')

    return baseline, trajectory


# ---------------------------------------------------------------------------
# M02 regen entrypoint (original, renamed to regen_hpv16 for symmetry;
# the old function-level name run_and_summarize() is kept as an alias for
# backward compatibility with any callers).
# ---------------------------------------------------------------------------

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent / 'regression_baselines' / 'anchor_hpv16.json'
)


def regen_hpv16(out=None):
    """Regenerate the M02 single-genotype HPV16 v2 baseline JSON.

    Writes to ``tests/regression_baselines/anchor_hpv16.json`` by default.
    """
    if out is None:
        out = DEFAULT_OUT

    baseline = build_baseline()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f'Wrote v2.3 baseline ({hpv.__version__}) to {out}')
    print('Summary:')
    for k, v in baseline['summary'].items():
        print(f'  {k:<40} {v:>12.4g}')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Generate v2.3 anchor baselines (M02 HPV16 or M03 4-genotype).',
    )
    p.add_argument(
        '--genotypes',
        choices=['hpv16', 'four'],
        default='hpv16',
        help=(
            'Which baseline to regenerate.  '
            '"hpv16" (default) = M02 single-genotype HPV16 baseline.  '
            '"four" = M03 4-genotype baseline.'
        ),
    )
    p.add_argument(
        '--out',
        type=Path,
        default=None,
        help=(
            'Primary output JSON path.  '
            'Defaults: hpv16 -> anchor_hpv16.json; four -> anchor_4genotype.json.'
        ),
    )
    p.add_argument(
        '--traj-out',
        type=Path,
        default=None,
        dest='traj_out',
        help=(
            'Trajectory JSON output path (only used with --genotypes four).  '
            'Default: anchor_4genotype_trajectory.json.'
        ),
    )
    args = p.parse_args(argv)

    if args.genotypes == 'four':
        regen_4genotype(out=args.out, traj_out=args.traj_out)
    else:
        regen_hpv16(out=args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())