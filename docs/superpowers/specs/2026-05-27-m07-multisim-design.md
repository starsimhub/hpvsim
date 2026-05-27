# M07: MultiSim and Scenarios — Design

**Date:** 2026-05-27
**Milestone:** M07 (MultiSim and scenarios)
**Branch:** `m07-multisim` (off `m05-vaccination-scenarios`; PR targets `v3.0-dev`)
**Predecessor:** [M05 Vaccination Scenarios](2026-05-20-m05-vaccination-scenarios-design.md)
**Status:** Design — not yet implemented.

## Dependencies and rebase strategy

M07 branches off `m05-vaccination-scenarios` rather than `v3.0-dev` because
the demo (`examples/m07_uq_sweep.py`) and its smoke test
(`tests/test_m07_demo_smoke.py`) consume M05 deliverables: `hpv.routine_vx`,
`hpv.vx`, the Nigeria 4-genotype vaccination anchor at
`tests/regression/anchor_vx_routine.py`, and `products_vx.csv`. The
verification tests (`test_m07_multisim.py`, `test_m07_parallel.py`) and the
M01/M02 parity-gate tests do not depend on M05.

Once M05 merges to `v3.0-dev`, this branch rebases onto `v3.0-dev` and the
M07 PR targets `v3.0-dev` directly. If M07 finishes implementation before
M05 merges, the PR is held in draft against `v3.0-dev` with M05's diff
visible until M05 lands. (Same pattern as M03, which branched off
`m02-natural-history-parity` and rebased onto `v3.0-dev` after M02 merged.)

---

## Goal

Make `hpv.Sim` first-class with Starsim's multi-sim machinery so that users can
quantify uncertainty and compare scenarios using stock Starsim APIs, without any
hpvsim-specific wrapper machinery. After M07, a user can:

1. Run N seeds of any hpvsim configuration via `ss.MultiSim(sim).run()` and
   reduce to median + 10/90 quantiles.
2. Compare scenarios via `ss.parallel(*sims)` with proper sim labeling.
3. Sweep parameters via `sc.parallelize(make_sim, iterkwargs=...)` +
   `ss.parallel(*sims)`.

The M01 and M02 acceptance gates upgrade from single-seed deterministic
comparisons to multi-seed z-score parity gates matching the M03 standard
(30 v3 seeds vs 30 v2 seeds, `|z| < 3` per short-summary metric).

The work that lands in M07 is intentionally small. Starsim already ships
`ss.MultiSim` (with `mean`/`median`/`reduce`/`plot`), `ss.parallel`, and
`sc.parallelize`. M07 ships **zero new production code under `hpvsim/`** — its
deliverables are verification tests, two new milestone parity-gate test
families, one demo example, and a tutorial-section rewrite.

## Scope

**In scope:**

- Verification tests that `ss.MultiSim(hpv.Sim(...))` and
  `ss.parallel(hpv.Sim(...), ...)` produce expected behavior — including seed
  reproducibility (same `rand_seed` → identical results) and the documented
  Starsim `inplace=True` / `copy_inputs=True` semantics.
- New M01 and M02 multi-seed parity gates
  (`tests/test_m01_short_summary_parity.py`,
  `tests/test_m02_short_summary_parity.py`) mirroring the M03 short-summary
  z-score gate. Existing single-seed M01/M02 tests stay as fast smoke gates.
- Shared `parity_gate(v3_seeds, v2_seeds, z_threshold)` helper extracted to
  `tests/regression/parity.py` and consumed by the new M01/M02 tests.
- v2 30-seed baselines for M01 and M02 anchors, generated via an extended
  `tests/regression/multi_seed_v2.py --anchor m0{1,2}` (output gitignored,
  regenerated locally).
- M07 demo: `examples/m07_uq_sweep.py` running a 4×20 (coverage × seed)
  routine-vx sweep on the M05 Nigeria anchor, producing a median + 10/90
  cancer-incidence-by-age plot. A tiny smoke variant of the demo lives in
  `tests/test_m07_demo_smoke.py` so the example doesn't rot.
- Tutorial section in `docs/tutorials/tut_running.qmd`: replace v2's
  `hpv.Scenarios` section with a walk-through of the Starsim-native
  `make_sim(seed, scenario)` + `ss.parallel` pattern, plus a short
  "migrating from v2 Scenarios" subsection.
- `MIGRATION_PLAN.md` M07 sub-task list edited to match the actual scope
  (drop the two "Port" sub-tasks; add the retrofit + demo sub-tasks).

**Explicitly out of scope (deferred / dropped):**

- `hpv.Scenarios`, `hpv.Sweep` — dropped. No shim, no subclass, no
  re-export. Users use Starsim-native `make_sim(seed, scenario)` +
  `ss.parallel` / `sc.parallelize` directly. Validation repos that import
  `hpvsim.Scenarios` migrate as separate post-M07 work, coordinated with
  Robyn per RACI.
- Re-running M03 / M05 parity tests under different UQ thresholds — they
  already pass at `|z| < 3` and `|z| < 5` respectively and remain
  authoritative.
- Refactoring M03 / M05 parity tests onto the new `parity_gate()` helper.
  Tracking-issue follow-up; not gating M07.
- Tightening partnership-network parity beyond M01's wide single-seed
  thresholds — separate M02-followup issue (MIGRATION_PLAN line 131).
  M01's partnership-pattern metrics (`n_pairs`, `mean_dur`, `mixing-cosine`)
  are categorical/distributional and remain outside the z-score framework
  in M07.
- M04 (calibration) and M06 (screen-and-treat) UQ — those milestones own
  their own UQ gates as part of their scopes.
- Any new `hpvsim/run.py` or other v3 production module. Nothing needs to
  live there.
- CRN across truly different parameter values — covered narrowly by the
  same-seed-same-sim test in M07; full cross-scenario CRN audit is a
  follow-up.
- Internal seeding work (`hpvsim/seeding.py`, `_ExclusiveSeeder`). M07 is a
  multi-sim concern, not a per-sim RNG concern.

---

## Architecture

### Module layout

```
hpvsim/
    (no new modules)              # M07 ships ZERO new production code
hpvsim/__init__.py                # no new re-exports — users import
                                  # ss.MultiSim / ss.parallel from starsim
tests/
├── regression/
│   ├── parity.py                 # NEW — shared parity_gate() helper       ~80 LOC
│   ├── anchor_m01.py             # NEW — M01 anchor PARS (1-genotype HPV16) ~40 LOC
│   ├── anchor_m02.py             # NEW — M02 anchor PARS (HPV16 + nat hist) ~40 LOC
│   └── multi_seed_v2.py          # EDITED — add --anchor m01/m02 flag       patch
├── test_m01_short_summary_parity.py   # NEW — 30-seed z-score gate         ~120 LOC
├── test_m02_short_summary_parity.py   # NEW — 30-seed z-score gate         ~120 LOC
├── test_m07_multisim.py               # NEW — verify ss.MultiSim + hpv.Sim  ~80 LOC
├── test_m07_parallel.py               # NEW — verify ss.parallel + RNG     ~80 LOC
└── test_m07_demo_smoke.py             # NEW — tiny smoke variant of demo   ~30 LOC
examples/
└── m07_uq_sweep.py               # NEW — 4×20 coverage sweep demo          ~120 LOC
docs/tutorials/
└── tut_running.qmd               # EDITED — Starsim-native sweep pattern   patch
```

### Why no new hpvsim modules

The MIGRATION_PLAN M07 sub-task list as drafted called for porting v2's
`Scenarios` and `Sweep` classes (~800 LOC combined). Both are dropped:

- Starsim's `ss.parallel(*labeled_sims)` already does v2 `Scenarios`'s job in
  one line, given a `make_sim(seed, scenario)` factory. The factory pattern is
  the documented Starsim idiom (see the `starsim-dev-run` skill reference).
- Per the M07 user directive — "use Starsim's functionality whenever
  possible" — and the implementation convention #3 ("subclass-first
  delegations must have a tracking issue to strip before M10"), shipping a
  shim layer that we know will be deleted in M10 has negative ROI.
- Validation-repo updates are cheap (delete `hpvsim.Scenarios` import,
  replace with three lines of `make_sim` + `ss.parallel`); they're a single
  PR per repo, post-M07.

The MIGRATION_PLAN sub-task list is edited as part of this milestone's commit
to reflect the actual scope.

### Shared parity-gate helper

The z-score formula

```
z = (mean_v3 - mean_v2) / sqrt(SE_v2^2 + SE_v3^2)
```

is currently inline in `tests/test_m03_short_summary_parity.py` and the M05
parity tests. Two new usages (M01, M02) make extraction worthwhile:

```python
# tests/regression/parity.py

def parity_gate(
    v3_seeds: list[dict],
    v2_seeds: list[dict],
    z_threshold: float = 3.0,
    skip_keys: frozenset = frozenset(),
) -> list[tuple[str, float]]:
    """Return [(metric_name, z)] for metrics exceeding |z| >= z_threshold.

    Each *_seeds entry is a dict {metric_name: value} produced by the
    milestone's short_summary builder. SEs are computed across seeds within
    each group.
    """
```

Consumed by the new M01/M02 tests only. M03/M05 keep their inline loops in
M07; a follow-up issue refactors them onto the helper post-merge.

---

## Verification tests

### `tests/test_m07_multisim.py`

Four tests against `ss.MultiSim` + `hpv.Sim`:

1. `test_multisim_n_runs` — `ss.MultiSim(hpv.Sim(...), n_runs=5).run()`
   returns 5 distinct sims; each sim's `short_summary` is finite and
   non-degenerate.
2. `test_multisim_median_reduce` — after `msim.median()`, the per-result
   median + 10/90 quantile arrays have the right shape and are non-negative
   for bounded metrics (cancers, infections, deaths).
3. `test_multisim_mean_reduce` — same shape check for `msim.mean()` as a
   smoke test. The `starsim-dev-run` skill cautions against `.mean()` for
   bounded quantities; we don't recommend it, but it must function.
4. `test_multisim_label_propagation` — labels on individual sims survive
   into `msim.sims[i].label` and post-reduction result indices.

### `tests/test_m07_parallel.py`

Four tests against `ss.parallel` + `hpv.Sim`:

1. `test_parallel_seed_reproducibility` — two `hpv.Sim(rand_seed=42)`
   passed through `ss.parallel` produce **identical** `short_summary`
   per metric. Bit-for-bit equality.
2. `test_parallel_seed_separation` — two `hpv.Sim` with `rand_seed=0` and
   `rand_seed=1` produce different but plausibly distributed results; a
   sanity bound (`|z| < 5` vs M03 anchor's known per-seed CV), not a
   parity gate.
3. `test_parallel_inplace_semantics` — `ss.parallel(s1, s2)` (default
   `inplace=True`) populates `s1.results` and `s2.results`;
   `ss.parallel(s1, s2, inplace=False)` leaves the originals unrun.
4. `test_parallel_copy_inputs_semantics` — a shared
   `hpv.routine_vx(coverage=0.6)` instance passed to two sims yields two
   independent intervention copies; post-run inspection must access
   `sim.interventions[...]`, not the original reference. This is the
   sim-copies-inputs discipline reified as a regression test.

### Test budget

- Verification tests use tiny sims (`n_agents=500`, `dur=2`). Total runtime
  for both files < 30 s. Marked as the existing default (fast) suite.
- M01/M02 parity tests use full milestone anchor pars at 30 seeds. Marked
  `@pytest.mark.slow`, consistent with M03/M05 marking. Picked up by the
  slow CI job.

### Failure-mode notes

- `test_parallel_seed_reproducibility` is the most likely flake source if
  Starsim's RNG handling has subtle in-place state. If it flakes, root-cause
  it rather than relaxing the check.
- Pre-emptive guard: each test constructs `hpv.Sim` fresh (no `sc.dcp`) to
  avoid CRN-state leakage from prior test interactions.

---

## M01 / M02 UQ retrofit

### Anchors

Two new tiny PARS-dict modules mirroring `tests/regression/anchor_4genotype.py`
(M03):

- `tests/regression/anchor_m01.py` — single-genotype HPV16, transmission
  only (no precin/CIN/cancer), Nigeria, 1990–2030. Matches the M01
  sub-task scope.
- `tests/regression/anchor_m02.py` — HPV16 + full natural-history
  progression, Nigeria, 1990–2060. Matches the M02 sub-task scope.

### v2 baseline generation

Extend `tests/regression/multi_seed_v2.py` with `--anchor` accepting
`m01` / `m02` / `m03_4genotype` / `m05_vx_routine` / `m05_vx_campaign`.
Generated baselines live at `tests/regression/v2_m01_seeds_n30.json` and
`tests/regression/v2_m02_seeds_n30.json` (both gitignored).
`tests/regression/README.md` updated with regeneration instructions.

### Gates

- `tests/test_m01_short_summary_parity.py`: 30 v3 seeds via
  `ss.MultiSim(hpv.Sim(**M01_PARS), n_runs=30).run()`. Metrics: total
  infections, mean HPV prevalence, total population. Threshold `|z| < 3`.
- `tests/test_m02_short_summary_parity.py`: 30 v3 seeds on M02 anchor.
  Metrics: full M03 short-summary set restricted to HPV16 (total
  infections, total cancers, total cancer deaths, mean HPV prevalence,
  mean cancer incidence, mean ages of infection / cancer / cancer death,
  total population). Threshold `|z| < 3`.

Both tests call `parity_gate(...)` and `pytest.fail(...)` listing any
failing metrics.

### Existing M01 / M02 tests

`test_partnership_equivalence.py`, `test_natural_history.py`, and the rest
of the single-seed M01/M02 smoke tests are not modified. They keep their
fast-smoke role; the new multi-seed tests become the new acceptance gates.

### Partnership-network metrics

`test_partnership_equivalence.py`'s categorical/distributional metrics
(`n_pairs`, `mean_dur`, `concurrency_max`, `mixing-cosine`) remain in their
current single-seed form. They are not means-with-computable-SEs, so the
z-score formulation does not apply. Tightening their thresholds is tracked
as the separate M02-followup issue (MIGRATION_PLAN line 131).

---

## Demo: vx-coverage sweep

### File

`examples/m07_uq_sweep.py` (~120 LOC).

### Configuration

Reuses M05's Nigeria 4-genotype anchor (`tests/regression/anchor_vx_routine.py:PARS`)
so the demo introduces no new calibration target — it shows how validated M05
machinery extends to UQ + scenario sweep.

### Sweep grid

4 coverage levels × 20 seeds = 80 sims total.

- Coverage levels ∈ `{0.0, 0.3, 0.6, 0.9}` — passed as the `prob` kwarg on
  `hpv.routine_vx` (which forwards to `ss.RoutineDelivery.prob`).
- `rand_seed ∈ range(20)`
- The 0-coverage baseline is realized by setting `prob=0.0` on the
  intervention rather than by dropping the intervention. The sim-construction
  path is identical across the four scenarios; the only difference is the
  `prob` scalar. This keeps the RNG stream cleanly comparable across
  scenarios.

### Script shape

```python
import sciris as sc
import starsim as ss
import hpvsim as hpv
from tests.regression.anchor_vx_routine import PARS as VX_PARS

COVERAGES = [0.0, 0.3, 0.6, 0.9]
N_SEEDS = 20

def make_sim(seed, coverage):
    kwargs = sc.dcp(VX_PARS)
    kwargs['rand_seed'] = seed
    kwargs['label'] = f'coverage={coverage:.0%}'
    kwargs['interventions'] = [hpv.routine_vx(
        product='bivalent', prob=coverage, age_range=(9, 14), sex='f',
    )]
    return hpv.Sim(**kwargs)

if __name__ == '__main__':
    iterkwargs = [dict(seed=s, coverage=c)
                  for c in COVERAGES for s in range(N_SEEDS)]
    sims = sc.parallelize(make_sim, iterkwargs=iterkwargs)
    msim = ss.parallel(*sims)

    by_cov = {c: ss.MultiSim([s for s in msim.sims
                              if s.label == f'coverage={c:.0%}'])
              for c in COVERAGES}
    for sub_msim in by_cov.values():
        sub_msim.median()

    _plot_coverage_sweep(by_cov)
    sc.savefig('m07_uq_sweep.png')
```

### Why this exercises every M07 sub-task

- `sc.parallelize(make_sim, ...)` exercises parallel sim *creation*.
- `ss.parallel(*sims)` exercises parallel sim *execution* across seeds and
  scenarios.
- `ss.MultiSim([...]).median()` exercises result reduction (median + 10/90
  quantiles).
- The `make_sim(seed, coverage)` factory pattern *is* the documented
  Starsim-native replacement for v2's `Scenarios` — this script becomes
  the canonical reference in the tutorial.

### Tutorial section

`docs/tutorials/tut_running.qmd` is edited to:

- Replace the v2 `hpv.Scenarios` example currently embedded there with a
  condensed walk-through of `m07_uq_sweep.py`.
- Add a short "Migrating from v2 Scenarios" subsection mapping
  `hpv.Scenarios(scenarios={'low': {...}, 'high': {...}})` → labeled sims +
  `ss.parallel`.

### CI cost

The demo script itself is not in the test suite — runtime is too long. The
smoke variant `tests/test_m07_demo_smoke.py` (1 coverage × 2 seeds) covers
the example surface so it doesn't bit-rot.

---

## Random-seed semantics

This section pins down the seed semantics M07 depends on. They are Starsim
defaults; M07 verifies them rather than implementing them.

- **Single sim.** `hpv.Sim(rand_seed=k)` is reproducible bit-for-bit on
  identical inputs. Already established by M03's parity gates; M07 changes
  nothing.
- **Replicate runs.** `ss.MultiSim(sim, n_runs=N).run()` assigns
  `rand_seed = base_seed + i` to the i-th replicate, where `base_seed`
  defaults to `sim.pars.rand_seed`. Verified in
  `test_m07_multisim::test_multisim_n_runs` by asserting replicate `i`
  matches a freshly-constructed `hpv.Sim(rand_seed=base_seed + i)`.
- **Scenario comparison.** `ss.parallel(s1, s2, ...)` honors each sim's
  `rand_seed` verbatim — no auto-stepping. Two sims with the same explicit
  `rand_seed` produce identical RNG streams; this is the property
  `test_parallel_seed_reproducibility` exercises.
- **Cross-scenario CRN.** When `make_sim(seed, scenario)` builds N sims per
  scenario sharing the same `seed` set, the same seed across scenarios
  drives correlated streams (paired-seed CRN for variance reduction). M07
  covers the same-seed-same-sim case directly; full CRN-across-scenarios
  audit is a tracking-issue follow-up.

**Nuance worth a docstring, not a test.** Starsim's CRN guarantee holds at
the *distribution-call level*, not the *simulation-step level*. Adding or
removing an `ss.Bernoulli` draw in one variant shifts all downstream draws.
The verification tests' docstrings note this so future maintainers don't
expect bit-equality after adding new modules.

---

## Testing & CI

### New tests

| File | LOC | Markers | Purpose |
|---|---:|---|---|
| `tests/test_m07_multisim.py` | ~80 | fast | n_runs distinctness, median/mean shape, label propagation |
| `tests/test_m07_parallel.py` | ~80 | fast | seed reproducibility, separation, inplace, copy_inputs |
| `tests/test_m07_demo_smoke.py` | ~30 | fast | tiny variant of the demo |
| `tests/test_m01_short_summary_parity.py` | ~120 | slow | 30-seed z-score gate vs v2 M01 baseline |
| `tests/test_m02_short_summary_parity.py` | ~120 | slow | 30-seed z-score gate vs v2 M02 baseline |

### New helpers / data

| File | LOC | Purpose |
|---|---:|---|
| `tests/regression/parity.py` | ~80 | shared `parity_gate()` |
| `tests/regression/anchor_m01.py` | ~40 | M01 anchor PARS dict |
| `tests/regression/anchor_m02.py` | ~40 | M02 anchor PARS dict |

### Existing files edited

- `tests/regression/multi_seed_v2.py` — add `--anchor m01|m02` flag.
  Backward-compatible.
- `tests/regression/README.md` — document the two new anchors + their
  baseline regeneration.
- `docs/tutorials/tut_running.qmd` — replace `Scenarios` section.
- `MIGRATION_PLAN.md` — M07 sub-task list edited to reflect actual scope.

### Existing files NOT touched

- `test_m03_*_parity.py`, `test_m05_vx_*_parity.py` — left alone.
- `test_partnership_equivalence.py`, `test_natural_history.py`, etc. — left
  alone.

### CI

- Fast jobs (default) pick up the three new fast tests automatically;
  combined added wall-clock < 60 s.
- Slow job (existing `@pytest.mark.slow` selection) picks up the two new
  parity tests. No new CI configuration needed.

---

## Acceptance criteria

M07 is done when:

1. All new tests pass locally and in CI.
2. `examples/m07_uq_sweep.py` runs end-to-end on developer hardware and
   produces the cancer-incidence-by-age figure (`m07_uq_sweep.png`).
3. `tests/regression/v2_m01_seeds_n30.json` and `v2_m02_seeds_n30.json` are
   regeneratable from a v2-hpvsim env via `multi_seed_v2.py --anchor m0X`.
4. `docs/tutorials/tut_running.qmd` updated; no `hpv.Scenarios` or
   `hpv.Sweep` references remain in active docs or examples.
5. `MIGRATION_PLAN.md` M07 sub-task list edited to:
   - Drop "Port the `Scenarios` class for parameter sweeps and intervention comparisons."
   - Drop "Port the `Sweep` class for systematic parameter variation."
   - Keep "Verify `ss.MultiSim` works with `hpv.Sim`."
   - Keep "Verify `ss.parallel()` works with proper random-seed handling."
   - Replace the M1–M6 re-run bullet with "Retrofit M01/M02 acceptance tests to multi-seed z-score gates matching the M03 standard; M03/M05 already pass; M04/M06 own their own UQ."
   - Add "Demo: vx-coverage sweep example (`examples/m07_uq_sweep.py`) + tutorial-section rewrite."

## Tracking issues opened during M07 (followups, not gating)

- **Refactor M03/M05 parity tests onto `parity_gate()` helper.** Code
  dedup, no behavior change.
- **Update validation repos (`hpvsim_methods_manuscript`, etc.) to drop
  `hpv.Scenarios` imports.** Coordinated with Robyn per RACI.
- **Tighten partnership-network parity beyond M01 single-seed thresholds.**
  Distinct from M07; remains an M02-followup issue.
- **Audit cross-scenario CRN across truly different parameter values.**
  Out of M07 scope; covered narrowly by M07's same-seed-same-sim test.

## Risks

- **`test_parallel_seed_reproducibility` flake.** Most likely failure mode
  is subtle Starsim RNG in-place state across `ss.parallel` workers.
  Mitigation: fresh `hpv.Sim` per test (no `sc.dcp`); explicit
  `copy_inputs=True` for all comparison sims; if it flakes, root-cause
  rather than relax.
- **v2-baseline drift between regenerations.** Mitigation: lock
  `tests/_legacy/requirements.txt` to the exact v2 release used for M03
  baseline regen; document the lock in the README. (This is an existing
  risk for M03/M05; M07 just inherits it.)

## References

- Starsim multi-sim docs: `starsim-dev-run` skill (Starsim 3.3.4).
- M03 design: [`2026-05-06-m03-multi-genotype-cross-immunity-design.md`](./2026-05-06-m03-multi-genotype-cross-immunity-design.md).
- M05 design: [`2026-05-20-m05-vaccination-scenarios-design.md`](./2026-05-20-m05-vaccination-scenarios-design.md).
- `MIGRATION_PLAN.md` — milestone definitions and conventions.