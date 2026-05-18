# M04: Calibration Loop — Design

**Date:** 2026-05-18
**Milestone:** M04 (Calibration loop)
**Branch:** `m04-calibration-loop` (off `v3.0-dev`; M04 PR targets `v3.0-dev`)
**Predecessor:** [M03 Multi-genotype and Cross-Immunity](2026-05-06-m03-multi-genotype-cross-immunity-design.md)
**Status:** Drafted; not yet implemented.

---

## Goal

Stand up an end-to-end Optuna-based calibration loop on top of `hpv.Sim`,
backed by a faithful port of v2's `age_results` analyzer for age-stratified
extraction, and prove the plumbing works via a synthetic parameter-recovery
smoke test. Build on `ss.Calibration` rather than v2's `hpv.Calibration`
class; replace v2's `compute_gof` / `compute_fit` math with Starsim's
`CalibComponent` likelihood framework. The full Optuna calibration of India
to convergence (and its comparison against the v2.x India posterior) is
explicitly scoped to a follow-on issue and **not** a merge gate for the M04
PR.

## Scope

**In scope:**

- `hpv.AgeResults(ss.Analyzer)` — faithful port of v2's `age_results`
  (`hpvsim/_v2_legacy/analysis.py:511`) onto `ss.Analyzer`. Snapshots
  age-binned counts / prevalences / incidences and the type-distribution
  sub-mode at specified years. Lives in `hpvsim/analyzers.py`.
- `hpv.Calibration(ss.Calibration)` — thin wrapper that defaults `build_fn`
  to `hpv.calibration.build_sim`, sets `study_name='hpvsim_calibration'`, and
  otherwise inherits.
- `hpv.calibration.build_sim(sim, calib_pars)` — routes flat dotted-key
  `calib_pars` (e.g. `'beta'`, `'hpv16.cin_fn.k'`,
  `'cross_immunity.nab_imm.hpv16.hpv18'`) into `sim.pars`,
  `sim.diseases[<genotype>].pars`, and the `CrossImmunity` connector. Raises
  on unrecognised keys.
- Three `CalibComponent` factories in `hpv.calibration` matching the v2 India
  target shapes: `cancer_by_age` (incident, Normal),
  `hpv_prev_by_age` (prevalent, Beta), `cancer_genotype_dist` (Dirichlet /
  step-containing). Each pre-wires the `extract_fn` against `AgeResults`
  outputs.
- Smoke parameter-recovery test
  (`tests/test_calibration.py::test_parameter_recovery`): set two known
  calib_pars, run sim to freeze an `expected` DataFrame, calibrate from
  broader bounds with `total_trials=50` and a deterministic Optuna sampler
  seed, assert the best trial recovers each known parameter within a
  generous relative tolerance.
- Unit coverage for AgeResults age-binning / multi-year snapshot /
  type-distribution / output schema, plus a v2-parity test against
  `hpvsim/_v2_legacy/analysis.py:511` on matched seeds.
- Unit coverage for `build_sim` routing (top-level / per-genotype /
  cross-immunity / unknown-key-raises / does-not-mutate-base).
- Re-run of M03 trajectory and short-summary parity tests as a regression
  gate — M04 should not move M03's needles.

**Explicitly out of scope (deferred):**

- A real Optuna calibration of India to convergence, including verifying the
  posterior overlaps v2.x's published India calibration. Filed as a
  follow-on issue at PR-open time; tracked against `MIGRATION_PLAN.md`'s
  "Run a full calibration for India" sub-task.
- Ranged-data CalibComponents: v2's India HPV-prevalence CSV stores values
  as `[low, high]` intervals (see `tests/test_data/india_hpv_prevalence.csv`)
  and was consumed via v2's custom `estimator` that distanced to interval
  bounds. Starsim's built-in likelihoods don't ship this shape. Handled by a
  follow-on issue together with the full India run.
- Port of v2's `compute_gof` / `compute_fit` / per-trial `sim.fit` aggregation.
  Replaced by `CalibComponent` likelihoods. Reinterprets the
  `MIGRATION_PLAN.md` M04 sub-task "Port compute_gof() and likelihood
  functions for cancer incidence by age" as "implement the loss using
  Starsim's likelihood framework".
- Plotting helpers (`calib.plot`, `calibration.plot_facet_bootstrap`,
  posterior visualisations). Deferred to M09 plotting.
- v2's `extra_sim_results` glue, custom Optuna sampler wiring,
  multi-database conventions, snapshot-of-sim-pre-trial machinery.
- Location-name normalization and subnational regions. M02's exact-match
  country handling stays. Listed against a later milestone in
  `MIGRATION_PLAN.md`.
- Other analyzers (`snapshot`, `age_pyramid`, `age_causal_infection`,
  `dalys`). Deferred to M09 per `MIGRATION_PLAN.md`.
- Intervention-aware calibration. M05+ once interventions land.
- Multi-country / Scenarios-based calibration sweeps. M07+.

---

## Architecture

### Module layout

```
hpvsim/
  analyzers.py        # new — hpv.AgeResults(ss.Analyzer); will absorb M09's analyzers later
  calibration.py      # new — hpv.Calibration, build_sim, CalibComponent factories
  __init__.py         # exports Calibration, AgeResults at top level
tests/
  test_age_results.py     # new — AgeResults unit + v2 parity
  test_calibration.py     # new — build_sim routing, factories, smoke parameter recovery
```

Naming follows the M03 PascalCase precedent (`HPV`, `CrossImmunity`,
`GenotypePars`). v2's lowercase `age_results` is renamed to
`hpv.AgeResults` on the v3 side.

`hpvsim/analyzers.py` is named for the module's eventual role, not its M04
contents. M04 lands only `AgeResults` there; M09 fills in the rest. One file
holds the surface comfortably while it's small.

### `hpv.AgeResults(ss.Analyzer)`

Faithful port of v2's `age_results` onto `ss.Analyzer`. The public
construction surface stays close to v2:

```python
hpv.AgeResults(
    result_args = sc.objdict(
        cancers                  = sc.objdict(years=[2015, 2020], edges=age_edges),
        hpv_prevalence           = sc.objdict(years=2020,         edges=age_edges),
        cancerous_genotype_dist  = sc.objdict(years=2020,         edges=age_edges,
                                              compute_fn=...),  # type-distribution sub-mode
    ),
)
```

Lifecycle (matches `ss.Analyzer` hooks):

- `init_pre`: validate result keys against a known whitelist; allocate
  per-year per-bin output arrays; resolve year keys to Timeline ticks; hook
  per-genotype `hpv.HPV` modules so the analyzer knows which states/results
  to read for each tracked output.
- `step`: at each scheduled year, snapshot age-binned counts / prevalences /
  incidences from sim state and per-genotype HPV modules.
- `finalize`: compute derived results (rates, distributions); expose
  `to_dataframe()` for `CalibComponent.extract_fn` consumption. The
  returned frame is indexed by year and has age-bin labels as columns
  (or, for the type-distribution sub-mode, genotypes as columns).

The type-distribution sub-mode (e.g. `cancerous_genotype_dist`) sums cancers
by genotype within the age window, matching v2 semantics. Per-age-bin and
per-genotype counts each sum to the total cancers in that age window
(invariant asserted in the unit test).

**Deltas from v2 (intentional, listed because the v2 caller will not work
verbatim):**

- v2's `result_args[k].datafile` (loading observed data into the analyzer
  itself) is dropped. In Starsim's framework observed data lives on the
  `CalibComponent`, not on the analyzer. `hpv.AgeResults` is purely a
  simulation-output snapshotter.
- v2's per-key `gofs` / `mismatch` baked into the analyzer is dropped. The
  loss path is `CalibComponent`'s job.
- Plotting helpers from v2's `age_results.plot()` are not ported. M09.

### `hpv.Calibration(ss.Calibration)`

```python
class Calibration(ss.Calibration):
    def __init__(self, sim, calib_pars, *, build_fn=None, **kwargs):
        if build_fn is None:
            build_fn = build_sim    # hpv.calibration.build_sim
        kwargs.setdefault('study_name', 'hpvsim_calibration')
        super().__init__(sim, calib_pars, build_fn=build_fn, **kwargs)
```

That is the whole class. Every other concern — Optuna study lifecycle,
sqlite db file naming, parallel workers, `n_trials` / `total_trials`
arithmetic, MultiSim integration, `reseed`, error handling — is inherited
from `ss.Calibration` unchanged. The wrapper exists so that users see
`hpv.Calibration` in the top-level namespace and so that the default
`build_fn` and `study_name` don't have to be re-specified at every call
site.

### `hpv.calibration.build_sim(sim, calib_pars)`

`ss.Calibration` invokes `build_fn(sim_copy, calib_pars=trial_pars,
**build_kw)` per trial, expecting back the modified sim. `build_sim` walks
each entry in `calib_pars`, splits on `.`, and dispatches by prefix:

- **No dot** (e.g. `'beta'`) → `sim.pars[key] = value`.
- **`<genotype>.<...>`** where `<genotype>` is a registered genotype key
  (`hpv16`, `hpv18`, `hi5`, `ohr`) → walk the remaining dotted path into
  `sim.diseases[<genotype>].pars[...]`. Leaf assignment uses
  `pars[leaf] = value`; for nested distribution-spec dicts (e.g.
  `dur_precin.par1`) the walker descends one more level.
- **`cross_immunity.<...>`** → walk into the `CrossImmunity` connector's
  matrix entry. `cross_immunity.cross_imm_sus.hpv16.hpv18` writes into the
  `(hpv16, hpv18)` cell of the connector's `cross_imm_sus` matrix (how
  much source-genotype `hpv18`'s `nab_imm` reduces target-genotype `hpv16`'s
  susceptibility). `cross_immunity.cross_imm_sev.<i>.<j>` is the analogous
  path for the severity matrix.
- **Anything else** → raise `ValueError`. No silent fallback — the
  calibration loop must not run on a typo'd key.

`build_sim` does not mutate the base sim. `ss.Calibration` passes a
`sc.dcp` copy in; `build_sim` mutates that copy and returns it. A unit test
asserts the base sim's `pars` and per-genotype `diseases[g].pars` are
unchanged after a `build_sim` call.

### `CalibComponent` factories

Three factories in `hpv.calibration`, one per common HPV target shape:

```python
hpv.calibration.cancer_by_age(
    expected,           # pd.DataFrame indexed by year, age-bin labels as columns
    *,
    likelihood='normal',
    weight=1,
) -> ss.CalibComponent

hpv.calibration.hpv_prev_by_age(
    expected,           # same shape; values in [0,1]
    *,
    likelihood='beta',
    weight=1,
) -> ss.CalibComponent

hpv.calibration.cancer_genotype_dist(
    expected,           # pd.DataFrame indexed by year, genotype keys as columns
    *,
    likelihood='dirichlet',
    weight=1,
) -> ss.CalibComponent
```

Each factory:

1. Validates `expected`'s schema against what `AgeResults.to_dataframe()`
   produces for that result key. Raises on mismatch — early failure beats a
   silently misaligned likelihood.
2. Constructs the `extract_fn` closure that locates the `AgeResults`
   analyzer on `sim` and returns
   `sim.<analyzers attribute>.to_dataframe(key=<...>).loc[expected.index, expected.columns]`.
   The implementation plan pins the exact analyzer key (Starsim's default
   for `class AgeResults` is `ageresults`; the implementation may override
   the analyzer's `name` to `'age_results'` for readability — either is
   fine, the factories use whatever's chosen).
3. Picks `conform`: `'incident'` for `cancer_by_age`, `'prevalent'` for
   `hpv_prev_by_age`, `'step_containing'` for `cancer_genotype_dist`.
4. Returns a fully-wired `ss.CalibComponent`.

The factories are advisory. Users with target shapes outside this set
construct `ss.CalibComponent` directly with their own `extract_fn`. The
factories exist so the common shapes don't make every user re-derive the
`AgeResults` → DataFrame plumbing.

### `calib_pars` shape

Flat dict with dotted keys, value is the Optuna trial spec (Starsim's
existing shape):

```python
calib_pars = {
    'beta':                  dict(low=0.10, high=0.30, guess=0.20),
    'hpv16.dur_precin.par1': dict(low=2.0,  high=6.0,  guess=3.5),
    'hpv16.cin_fn.k':        dict(low=0.2,  high=1.0,  guess=0.5),
    'hpv18.dur_precin.par1': dict(low=2.0,  high=6.0,  guess=3.5),
    'cross_immunity.cross_imm_sus.hpv16.hpv18': dict(low=0.0, high=0.5, guess=0.2),
}
```

This shape plays cleanly with Optuna's flat parameter space and avoids v2's
nested `genotype_pars=dict(hpv16=dict(...))` form that wouldn't translate
trial-by-trial without an intermediate flattener.

### Module exports

`hpvsim/__init__.py` adds:

```python
from .calibration import Calibration
from .analyzers   import AgeResults
```

so `hpv.Calibration` and `hpv.AgeResults` are top-level.
`hpv.calibration.build_sim` and the three factories stay namespaced under
`hpv.calibration` rather than promoted to top-level — they are intermediate
plumbing.

---

## Smoke calibration: synthetic parameter recovery

`tests/test_calibration.py::test_parameter_recovery`. The smoke test
verifies the loop, not calibration quality.

1. **Generate target.** Build a small `hpv.Sim` (~2k agents, 4 genotypes,
   India, 1990–2030, fixed seed). Set two known calib_pars to specific
   values (e.g. `beta=0.20`, `hpv16.cin_fn.k=0.55`). Run, then call
   `AgeResults.to_dataframe(key='cancers')` to freeze cancer-by-age as the
   `expected` DataFrame. `expected` is captured at runtime — no committed
   target file.
2. **Set up calibration.** `hpv.Calibration(sim_base, calib_pars,
   components=[hpv.calibration.cancer_by_age(expected)], total_trials=50,
   sampler=<TPESampler with fixed seed>, n_workers=...)` where `sim_base` is
   the same base sim without the known-par overrides, and `calib_pars`
   brackets the known values:
   - `beta`:                 `[0.10, 0.30]`, guess `0.20`
   - `hpv16.cin_fn.k`:       `[0.20, 1.00]`, guess `0.50`
3. **Assert.**
   - `calibration.run()` completes without exception.
   - `calibration.study.best_trial` is populated.
   - Best trial's `beta` is within ±25% relative error of `0.20`.
   - Best trial's `hpv16.cin_fn.k` is within ±25% relative error of `0.55`.
   - The expected ↔ actual DataFrames produced by the `extract_fn` align on
     schema (same year index, same age-bin columns).
   - AgeResults outputs from two repeat runs at the same seed match (CRN
     sanity).

Tolerance of ±25% is sized for *reliability under 50 trials*, not
statistical strictness. The smoke test is a plumbing gate, not a
calibration-quality gate.

CI cost target: ≤ 90 seconds wall on a typical CI worker. If 50 trials proves
either flaky (parameter recovery failing on >5% of CI runs) or too slow,
trial count and tolerance are tuning knobs the implementation plan can
adjust without re-touching this spec.

---

## Testing strategy

| Tier | Test | Asserts |
|---|---|---|
| AgeResults unit | `tests/test_age_results.py::test_age_binning` | Age-bin edges respected; agents at exact edges land in the upper bin (matches v2 convention) |
| AgeResults unit | `::test_multi_year_snapshot` | `years=[2015, 2020]` produces a snapshot at each year; intervening years not recorded |
| AgeResults unit | `::test_type_distribution_sub_mode` | `cancerous_genotype_dist` sums cancers by genotype within age window; per-genotype counts sum to per-age total |
| AgeResults unit | `::test_to_dataframe_schema` | Output DataFrame's index / columns match what the CalibComponent factories expect (regression anchor for the contract) |
| AgeResults parity | `::test_against_v2_age_results` | Run v3 sim with `AgeResults` and v2 sim with `age_results` on matched seeds and small agent count; assert age-bin counts within ±5% per bin |
| `build_sim` unit | `tests/test_calibration.py::test_build_sim_routing` | Top-level / per-genotype / cross-immunity calib_pars all land at the right address; unknown key raises |
| `build_sim` unit | `::test_build_sim_does_not_mutate_base` | Base sim unchanged after `build_sim` (asserts the dcp contract holds for hpv-shaped sims) |
| Factories unit | `::test_cancer_by_age_factory` | `cancer_by_age` constructs a `CalibComponent` whose `extract_fn` returns a DataFrame matching `expected`'s schema |
| Factories unit | `::test_hpv_prev_by_age_factory`, `::test_cancer_genotype_dist_factory` | Same shape gate for the other two factories |
| Smoke | `::test_parameter_recovery` | End-to-end loop converges on known params |
| Regression | M03 trajectory + short-summary parity tests | Still green — M04 should not move M03's needles |

CI tier separation: AgeResults unit tests and `build_sim` unit tests run on
every commit. The parity test (`test_against_v2_age_results`) and the smoke
calibration run on a slower tier that is still required for merge — same
convention as M03's multi-seed gates.

---

## Branch, PR, and acceptance gates

**Branch.** `m04-calibration-loop` off `v3.0-dev`.

**PR target.** `v3.0-dev`. Single PR for the milestone. Robyn reviews per
RACI.

**Development-tier (per PR, blocking merge):**

- All new tests in the testing-strategy table pass.
- M03 trajectory + short-summary parity tests still green.
- `hpv.Sim().run()` invariant holds (per `MIGRATION_PLAN.md:300`).
- `from hpvsim import Calibration, AgeResults` succeeds.
- `hpv.Calibration(sim, calib_pars).run()` returns a study without
  exception on a tiny config.

**Release-tier (deferred to follow-on, NOT an M04 PR blocker):**

- A full Optuna calibration of India to convergence whose posterior overlaps
  v2.x. Filed as a follow-on issue at PR-open time.

**Follow-on issues filed at M04 PR open:**

- "Run a full India calibration; verify posterior overlap with v2.x"
  (the deferred release-tier acceptance test).
- "Real India data CalibComponents: handle ranged `[low, high]`
  HPV-prevalence data" (needs a custom likelihood or pre-processing step
  Starsim does not ship).
- "AgeResults: optional plotting helpers" — link from M09 plotting spec
  when written, so M09 sees the dependency.

---

## Quarantine policy

`hpvsim/_v2_legacy/analysis.py` and `hpvsim/_v2_legacy/calibration.py` stay
quarantined per `MEMORY.md`'s "Quarantine dependencies" rule. M04's
`hpv.AgeResults` re-implements v2's `age_results` logic in active code
rather than re-exporting or sub-classing it. The v2 parity test
(`test_against_v2_age_results`) is the only test that imports from
`hpvsim/_v2_legacy/` — it is a regression anchor, not a runtime dependency.
M10 deletes the quarantine wholesale.