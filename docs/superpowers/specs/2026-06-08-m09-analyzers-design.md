# M09 (Part 1): Remaining analyzers — design

**Date:** 2026-06-08
**Branch:** `m09-analyzers` (branched off `m07-multiscale-ledger`, not `v3.0-dev`)
**Milestone:** M09 "Remaining analyzers and plotting" (see `MIGRATION_PLAN.md`)

## Scope

M09 bundles two largely independent chunks. This spec covers **Part 1 — the
analyzers**; the plotting layer (`sim.plot()`, by-age/genotype plots,
intervention-impact plots, calibration-result plots) is a deliberate **Part 2**
follow-up design/PR.

**In scope (this PR):**

- Port four v2 analyzers to v3-idiomatic `ss.Analyzer` subclasses in
  `hpvsim/analyzers.py`:
  - `snapshot`
  - `age_pyramid`
  - `age_causal_infection`
  - `dalys` (YLL + YLD)
- Add a convenience **type-distribution results accessor** (genotype
  distribution of cancers) over the already-existing per-module results.
- Enrich the multiscale ledger's per-cancer event record so `dalys` and
  `age_causal_infection` are unbiased under `ms_agent_ratio > 1`.

**Out of scope (deferred to M09 Part 2 — plotting):**

- All matplotlib `.plot()` methods, including `sim.plot()`, plots by age group
  and genotype, intervention-impact plots, calibration-result plots, and the
  `age_pyramid` / `age_causal_infection` plotting helpers.

**Not addressed here:**

- `AgeResults` is already ported (M2/M4) and stays untouched beyond a one-line
  internal helper rename. Its parity gates must remain green.

## Why this base branch

This work starts off `m07-multiscale-ledger` (assuming that PR lands as-is)
rather than `v3.0-dev`, because:

1. The ledger substantially rewrites `hpvsim/hpv.py` (+348 lines). Building M09
   on the pre-ledger base would conflict on merge.
2. The ledger was explicitly built with a `_cancer_events` hook "for the by-event
   distribution analyzer (own cancers + extras)" — i.e. it anticipates exactly
   these analyzers. `age_causal_infection` consumes that hook directly.
3. Agent-iterating analyzers (`dalys`, `age_causal_infection`) would otherwise
   silently undercount under multiscale, because extra cancers exist only as
   side-RNG ledger *data*, not as live `People` agents.

## Approach

Port each analyzer as a clean v3-idiomatic `ss.Analyzer` subclass alongside
`AgeResults` (rejected alternatives: a shared timepoint-snapshot base class —
premature abstraction that risks `AgeResults`' gates; and subclass-delegation to
`_v2_legacy` — more glue than a clean port, plus strip-before-M10 debt).

Shared infra lifted to module-level functions reused by all analyzers:

- `_resolve_date_ticks(sim, dates)` — generalization of the existing
  `_resolve_year_ticks`; maps `ss.date`-coercible inputs to timeline tick
  indices via `sim.timevec`. `AgeResults` keeps calling the (renamed/delegating)
  helper with no behavior change.
- `_histogram(ages, mask, edges, weights)` — already present; lifted to
  module scope.

HPV-module discovery (`[d for d in sim.diseases.values() if isinstance(d, HPV)]`)
is reused from `AgeResults`.

## Time handling — Starsim native (`ss.date`)

v2 keyed snapshots by **date strings** (`'2020'`). v3 drops that custom
convention in favor of Starsim's native time type:

- `timepoints` accepts anything `ss.date(...)` coerces: `2020`, `2020.5`,
  `'2020-06'`, an `ss.date`, or a list. `init_pre` normalizes each via
  `ss.date(...)` and resolves it to the tick whose `sim.timevec` entry matches,
  via `_resolve_date_ticks`.
- Stored snapshots/pyramids are **keyed by the resolved `ss.date`**
  (`self.snapshots[sim.timevec[ti]] = ...`), not a string.
- `get(key=None)` coerces its argument through `ss.date(...)`, so `get(2020)`,
  `get('2020')`, and `get(ss.date(2020))` all resolve to the same entry; bare
  `get()` returns the first.
- `age_causal_infection(start=...)` and `dalys(start=...)` take a single
  `ss.date`-coercible `start` (renamed from v2's `start_year`), compared against
  `sim.timevec` rather than a raw float-year `>=`.

This keeps all four analyzers consuming and emitting the same time objects
`sim.t` / `sim.timevec` use, so the Part-2 plotting layer gets real dates on the
axis (`plt.axvline(ss.date(...))` caveat applies there).

## Per-analyzer design

### `snapshot(timepoints=None, die=True)`

Structural readout — deep-copies `sim.people` at requested ticks.

- `init_pre`: resolve `timepoints` → ticks (default = sim stop date); store
  resolved `ss.date`s.
- `step`: if `sim.ti` is a snapshot tick,
  `self.snapshots[sim.timevec[ti]] = sc.dcp(sim.people)`.
- `finalize`: validate all requested dates were recorded (raise if `die`).
- `get(key=None)`: `ss.date`-coercing retrieval; defaults to first snapshot.

### `age_pyramid(timepoints=None, edges=None, age_labels=None, datafile=None, die=False)`

Age×sex histogram at requested ticks; optional observed datafile for later
data-vs-model plots.

- `init_pre`: resolve ticks; default `edges = np.linspace(0, 100, 11)`; build age
  labels; if `datafile`, load into `self.data` (sciris loader, same schema as
  v2).
- `step`: at each snapshot tick, store an `(nbins, 2)` array — male / female
  scale-weighted age histograms — keyed by `ss.date` in `self.age_pyramids`.
- `to_dataframe()`: tidy long-form `(date, age_bin, sex, count)`. No matplotlib.

### `age_causal_infection(start=None)`

Distribution of age-at-causal-infection, age-at-CIN, age-at-cancer, and dwell
times (`precin` / `cin` / `total`) for cervical-cancer cases. **Ledger-aware**:
generalizes the M07 prototype `_cancerpathwayages`.

- Accumulator lists: `age_causal`, `age_cin`, `age_cancer`, weights, and
  `dwelltime{precin, cin, total}`.
- At `ratio == 1`: read live agents — for each HPV module, select agents with
  `ti_cancerous == sim.ti` **gated on `cancerous & alive`** (a bare time-match
  overcounts agents scheduled for cancer who die of other causes first), compute
  causal/cin/cancer ages from `ti_infected` (the current persistent infection —
  the one that progressed; matches the ledger's own-cancer computation), `ti_cin`,
  current age, all at weight 1.
- At `ratio > 1`: read each module's enriched `_cancer_events` stream (own +
  extra sub-cancers), which already carries the pathway ages and the per-event
  weight.
- Dwell times derived from ages: `precin = cin_age − causal_age`,
  `cin = cancer_age − cin_age`, `total = cancer_age − causal_age`.
- `finalize`: convert lists → arrays.

### `dalys(start=None, life_expectancy=84)`

Incidence-based YLL + YLD, attributed at the **onset year** (matching v2).
**Ledger-aware** via the enriched event record.

- GBD2017 `disability_weights` (weights `[0.288, 0.049, 0.451, 0.54]`,
  time-fractions `[0.05, 0.85, 0.09, 0.01]`) → `av_disutility` property, ported
  verbatim.
- `init_pre`: `si` / `years` from `sim.timevec` years `>= start`; allocate
  `yll` / `yld` / `dalys` arrays by year.
- Consumption follows the same dual-source pattern as `age_causal_infection`
  (the existing `_cancerpathwayages` prototype):
  - `ratio == 1`: read live agents whose `ti_cancerous == sim.ti` (filtered to
    `>= start`), with `cancer_age = age`, `death_age = age + (ti_dead_cancer −
    ti_cancerous)·dt`, weight 1, `onset_ti = sim.ti`.
  - `ratio > 1`: read each module's enriched `_cancer_events` stream (own +
    extras), which carries `(onset_ti, cancer_age, death_age, weight)` directly.
  - For each event `(onset_ti, cancer_age, death_age, weight)`:
    - `dur_cancer = death_age − cancer_age`
    - `yld[year(onset_ti)] += weight · dur_cancer · av_disutility`
    - `yll[year(onset_ti)] += weight · max(0, life_expectancy − death_age)`

  This keeps the `ratio == 1` zero-overhead fast path (no event recording) and
  reuses the ledger only where extras exist.
- `finalize`: `dalys = yll + yld`.

`life_expectancy` defaults to 84 (v2 default); callers should pass
country-specific values where available.

## Multiscale ledger enrichment

To make both ledger-aware analyzers correct under `ms_agent_ratio > 1`, the
ledger's per-cancer event record is **enriched to a single canonical 6-tuple**:

```
_cancer_events: (onset_ti, causal_age, cin_age, cancer_age, death_age, weight)
```

(previously `(causal_age, cin_age, cancer_age, weight)`).

- `death_age` and `onset_ti` are already known at the two append sites
  (`_realize_ledger` for extras — `death_age` is in the internal `_ledger_onset`
  8-tuple; and the own-cancer branch in `step_state` — `ti_dead_cancer` is
  scheduled at onset). This is **propagation, not new logic**, and does not
  change any ledger behavior or population result.
- The `ratio == 1` zero-overhead fast path is preserved: the own-cancer append
  stays gated to `ratio > 1`, and both `dalys` and `age_causal_infection` read
  live agents at `ratio == 1` and `_cancer_events` at `ratio > 1` (matching the
  existing `_cancerpathwayages` prototype). The enrichment therefore only adds
  fields to events that are already recorded at `ratio > 1`.
- The M07 in-test consumer `_cancerpathwayages`
  (`tests/test_multiscale_distribution.py`) unpacks the old 4-tuple; it is
  updated to the new 6-tuple (or retargeted at the production
  `age_causal_infection`). The M07 *equivalence* tests
  (`test_multiscale_equivalence.py`, `test_multiscale.py`) assert on results
  (`new_cancers`, population), not tuple shape, so they are unaffected.

## Type-distribution results accessor

The plan's "Add type-distribution results (genotype distribution of cancers)" is
largely already covered: each HPV module exposes `new_cancers` / `cum_cancers`
per genotype, and `AgeResults` produces age-binned `cancerous_genotype_dist`.

The remaining gap is a **convenience aggregate** so users (and Part-2 plotting)
can get the genotype split without iterating modules:

- Add a thin accessor `results_by_genotype(sim, key='cum_cancers')` that stacks
  the per-module result into a `(n_genotypes, n_timepoints)` array / tidy
  DataFrame with genotype-name columns and optional normalization.
- Pure read of existing `ss.Result`s — no new simulation-time computation, no new
  connector-level result. Keeps per-module results as the single source of truth.
- Sited next to the analyzers (or in `sim.py` if it reads `sim.results`).

## Testing — targeted equivalence

Per the chosen validation rigor (targeted equivalence, not multi-seed z-score
baseline generation):

- **`snapshot`** — structural: requested dates recorded and keyed by `ss.date`;
  `get()` coercion across year / str / `ss.date`; `die` behavior; people
  deep-copied (mutating the sim after a snapshot does not change the snapshot).
- **`age_pyramid`** — structural: bin counts sum to the alive population at the
  tick; male + female split consistent with `people.female`; scale-weighting
  honored; datafile load schema.
- **`dalys`** — numeric: on a single fixed-seed anchor, YLL / YLD / DALYs within
  ±10% of a small v2 reference; `av_disutility` matches the v2 constant exactly;
  onset-year binning correct. Plus a ledger test: total weighted DALYs at
  `ratio > 1` overlap the `ratio == 1` result (multiscale-equivalence).
- **`age_causal_infection`** — numeric: dwell-time and age-at-causal-infection
  distributions overlap v2's on the same anchor (mean / median within tolerance);
  per-genotype causal attribution sums to total new cancers; ledger distribution
  unbiased across ratios (mirrors `test_multiscale_distribution`).
- **type-dist accessor** — unit: stacking shape, genotype-name columns,
  normalization sums to 1.

## Files touched

- `hpvsim/analyzers.py` — add `snapshot`, `age_pyramid`, `age_causal_infection`,
  `dalys`; lift `_resolve_date_ticks` / `_histogram` to module scope; add to
  `__all__`. Possibly the `results_by_genotype` accessor (or in `sim.py`).
- `hpvsim/hpv.py` — enrich `_cancer_events` to the 6-tuple at its two append
  sites (+ the internal `_ledger_onset` propagation already carries `death_age`).
- `hpvsim/__init__.py` — export the new analyzers.
- `tests/test_multiscale_distribution.py` — update the `_cancerpathwayages`
  consumer to the new tuple (or retarget at `age_causal_infection`).
- `tests/test_m09_analyzers.py` (new) — the targeted equivalence tests above.

## Deferred / follow-up

- M09 Part 2 (plotting) — its own spec.
- Any `AgeResults` refactor onto the shared helpers beyond the rename.