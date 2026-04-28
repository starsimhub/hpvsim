# HPVsim v3.0 — M0 Foundation: design spec

## Goal

Stand up the regression infrastructure that every later milestone will rely on. M0 is foundation work only — no migration code lands in this milestone, and the package on `v3.0-dev` continues to be v2.x verbatim until M1 begins.

The deliverables are:

1. CI on `v3.0-dev` extended to cover the new regression harness.
2. A committed v2.x baseline-generation script (`tests/regression/baseline.py`) that produces a deterministic baseline locally. Generated baseline files stay local (gitignored), never committed.
3. A committed anchor scenario (`tests/regression/anchor.py`) — vanilla 4-genotype HPV sim, Nigeria, fixed seed, no interventions.
4. A committed comparison script (`tests/regression/compare.py`) that diffs a current run against a locally-stored baseline and reports per-summary-result drift.
5. Documentation in `tests/regression/README.md`, with a pointer added to `tests/README.md`.

## Acceptance test

- CI passes on `v3.0-dev`, including a new smoke test that imports and runs the anchor sim, and a CI step that runs `compare.py` in its no-baseline mode (exit clean).
- A developer can run, locally:
  - `python tests/regression/baseline.py` — generates `tests/regression_baselines/anchor.json` from the currently-installed package.
  - `python tests/regression/compare.py` — runs the anchor against current code, loads the baseline, and prints a per-summary-result drift table.
- The drift report is informational: exit code 0 regardless of threshold breaches; the threshold (±10% relative) is a developer-facing signal, not a CI gate.

## Anchor scenario (pinned)

Lives in `tests/regression/anchor.py`. The module exposes `make_sim()` returning an unrun `hpv.Sim` with these pars, and a `__main__` that runs the sim and prints `sim.short_summary` plus `sim.results['n_alive'][-1]`.

```python
pars = dict(
    n_agents   = 10e3,
    location   = 'nigeria',
    genotypes  = [16, 18, 'hi5', 'ohr'],
    start      = 1990,
    end        = 2060,
    dt         = 0.5,
    burnin     = 20,
    rand_seed  = 0,
    verbose    = 0,
)
```

No interventions, no analyzers. The choice of Nigeria, the 4-genotype set, and the year range mirror Scenario 1 of the existing `tests/generate_v2_baselines.py` (which is being superseded by this harness for M0; that script's vaccination/screening scenarios are deferred until their owning milestones — M4 / M5 — when the corresponding capability scenarios will be added under `tests/regression/`).

## File layout

```
tests/
├── regression/
│   ├── __init__.py
│   ├── anchor.py                # make_sim() + __main__; defines the anchor scenario
│   ├── baseline.py              # CLI: generates baseline JSON from currently-installed package
│   ├── compare.py               # CLI: runs anchor, loads baseline, prints per-key drift table
│   └── README.md                # full detail (anchor pars, usage, drift semantics, gate behavior)
├── regression_baselines/        # gitignored (already excluded in .gitignore)
├── test_regression.py           # pytest: smoke test for anchor + unit tests for compute_drift
└── README.md                    # append a one-paragraph pointer to tests/regression/README.md
```

`anchor.py` is the scientific definition. `baseline.py` and `compare.py` are tooling that imports from `anchor.py`. The pytest file lives at `tests/test_regression.py` (not under `tests/regression/`) because the existing CI invokes `pytest test_*.py -n auto` from the `tests/` directory — that shell glob only picks up `test_*.py` files at the `tests/` root, not in subdirectories. The single `test_regression.py` file holds both the anchor smoke test and the unit tests for `compute_drift`.

## Comparison rules

The set of summary results diffed is pinned to v2.x's `sim.short_summary` plus `total population`. These keys are computed by v2.x on every run and represent the established headline epidemiological metrics:

- `total HPV infections`
- `total cancers`
- `total cancer deaths`
- `mean HPV prevalence (%)`
- `mean cancer incidence (per 100k)`
- `mean age of infection (years)`
- `mean age of cancer (years)`
- `mean age of cancer death (years)`
- `total population` (computed as `sim.results['n_alive'][-1]`)

Drift semantics:

- **Relative drift:** `(new - baseline) / baseline`. ±10% is the informational threshold.
- **Zero-baseline guard:** if a baseline value is zero (not expected for these keys in the anchor scenario, but guarded for safety), the comparison reports absolute drift and flags the row.
- **Output format:** a table with columns `key | baseline | new | abs_diff | rel_diff | over_threshold?`. Always exits 0.
- **No-baseline mode:** if `tests/regression_baselines/anchor.json` is missing, `compare.py` prints `no baseline; skipping diff` and exits 0 *without* running the anchor. The anchor-runs check is the pytest smoke test's job; no-baseline mode is about CLI integrity only. This is the mode CI runs `compare.py` in.

## CI integration

`.github/workflows/tests.yaml` is updated minimally:

- Existing pytest run stays as-is. The new `tests/test_regression.py` is at the `tests/` root, so it is picked up automatically by the existing `pytest test_*.py -n auto` invocation (which globs `test_*.py` from the `tests/` working directory).
- A new step is added after the pytest step that runs `python tests/regression/compare.py` (no-baseline mode). This proves the comparison CLI itself does not bitrot.

No second-venv v2 install. The actual ±10% drift comparison remains a developer-local action.

## Documentation

`tests/regression/README.md` covers, in this order:

1. **What this is.** The regression harness for the v2 → v3 migration. Informational development gate, not a CI gate.
2. **Anchor scenario.** Pars copied from `anchor.py`, justification for Nigeria, justification for the genotype set.
3. **Generating a baseline.** `python tests/regression/baseline.py`. Note that the baseline file is local-only and gitignored. Suggested workflow: generate from a clean v2.3.x environment before starting work, then keep that file as the migration target.
4. **Running the comparison.** `python tests/regression/compare.py`. Output table interpretation.
5. **Drift semantics.** Relative vs. absolute, the ±10% threshold's meaning, why it is informational.
6. **When to refresh the baseline.** Whenever the v2.x reference moves (new patch release of v2.3, post-merge from main into v3.0-dev), or whenever you decide a feature-misalignment-induced drift is the new target.

`tests/README.md` gets a single paragraph appended: "There is also a regression harness under `tests/regression/` used for the v2 → v3 migration. It runs outside the standard pytest flow and is documented in [`tests/regression/README.md`](regression/README.md)."

## Migration plan touch-ups

Two small edits to `MIGRATION_PLAN.md` follow from M0 settling:

1. In §Implementation conventions, item 2 (Dual validation gates), replace the preliminary "summary result" list with the pinned set (the 9 keys above) and remove the `the exact list is pinned in M0` clause now that it is.
2. In M0's Sub-tasks, the "Generate v2.x baseline-generation script" entry should reference `tests/regression/baseline.py` and the corresponding anchor / comparison entries should reference `anchor.py` / `compare.py` so the M0 GitHub issues map cleanly to files.

## Out of scope for M0

- Any migration code in `hpvsim/`. The package on `v3.0-dev` stays v2.x until M1 starts.
- Capability scenarios beyond the anchor (vaccination, screening, etc.) — added by M4 / M5 / etc. under `tests/regression/`.
- A second-venv CI job that installs v2 and runs the diff in CI. Deferred until M1+ code begins to land and manual local diffing becomes painful.
- Migrating or salvaging code from the existing `tests/generate_v2_baselines.py`. M0's anchor is defined fresh against the v2.x public API and uses `sim.short_summary` directly — no helpers from that script are needed. The script remains untracked. Later milestones may revisit it when capability scenarios for vaccination / screening are added under `tests/regression/`.
- Branch-protection rules on `v3.0-dev` (matched to rc2.3 = none, settled in the prior setup plan).
- Fixing automatic download failures (issue #30). This was flagged in the migration plan as "M0 or M9"; deferred to M9 to keep M0 lean.

## Linked documents

- [`MIGRATION_PLAN.md`](../../../MIGRATION_PLAN.md) — overall migration plan, of which this is the M0 deliverable spec.
- [`docs/superpowers/specs/2026-04-23-hpvsim-starsim-port-design.md`](./2026-04-23-hpvsim-starsim-port-design.md) — overall port design spec.
- [`tests/generate_v2_baselines.py`](../../../tests/generate_v2_baselines.py) — pre-existing v2 baseline generator (untracked). Reference; superseded by `tests/regression/` for M0.
- [`hpvsim/regression/pars_v2.3.0.json`](../../../hpvsim/regression/pars_v2.3.0.json) — pre-existing v2.3.0 parameter export (untracked). Useful as a stable parameter snapshot independent of the regression harness.