# Regression harness

This directory holds the v2 → v3 migration regression harness used during the
HPVsim v3.0 port. It runs *outside* the standard pytest flow: the harness's
job is to compare a current run of an anchor scenario against a stored v2
baseline and report per-summary-result drift to the developer.

The harness is deliberately small and informational — it is the **development
gate** described in `MIGRATION_PLAN.md` §Implementation conventions item 2.
It does **not** fail PRs; the **release gate** (overlapping uncertainty
intervals against the analysis-repo suite) is the scientific gate and lives
elsewhere.

## What's here

| File | Role |
|---|---|
| `anchor.py` | Pinned anchor scenario: vanilla 4-genotype HPV sim, Nigeria, seed 0, 1990–2060, no interventions. Exposes `make_sim()` and `run_and_summarize()`; runs as `__main__` for an ad-hoc summary print. |
| `baseline.py` | CLI: runs the anchor, writes a JSON baseline to `../regression_baselines/anchor.json` (gitignored). |
| `compare.py` | CLI: runs the anchor, loads a baseline, prints a per-key drift table. Exits 0 always. No-baseline mode exits without running the anchor. |
| `__init__.py` | Empty; makes this directory an importable package for the pytest smoke test in `tests/test_regression.py`. |

## Anchor scenario

Pinned in `anchor.py:PARS`:

| Par | Value |
|---|---|
| `n_agents` | `10e3` |
| `location` | `'nigeria'` |
| `genotypes` | `[16, 18, 'hi5', 'ohr']` |
| `start` | `1990` |
| `end` | `2060` |
| `dt` | `0.5` |
| `burnin` | `20` |
| `rand_seed` | `0` |
| `verbose` | `0` |

No interventions, no analyzers. Nigeria was chosen because the existing v2.x
multi-scenario script (`tests/generate_v2_baselines.py`) already uses Nigeria
as one of its three locations and Nigeria is well-represented in the
validation-repo suite.

## Generating a baseline

Baselines are local-only and gitignored. The recommended workflow:

1. Check out a clean v2.3.x environment (typically `git checkout rc2.3` and
   `pip install -e .`, or install `hpvsim==2.3.x` from PyPI in a separate venv).
2. Run:

   ```bash
   python tests/regression/baseline.py
   ```

3. The baseline lands at `tests/regression_baselines/anchor.json`. Keep that
   file as your migration target.

The script takes ~30–60s to run the anchor sim. Once the baseline is in place,
return to `v3.0-dev` and use it as the comparison reference.

## Running the comparison

```bash
python tests/regression/compare.py
```

Output: a table of per-key drift, e.g.

```
key                                      baseline      current     abs_diff   rel_diff   over
------------------------------------------------------------------------------------------------
total HPV infections                        12345        12350           5     +0.04%
mean HPV prevalence (%)                      8.2          8.2            0     +0.04%
...

0/9 keys exceed +/- 10% relative drift threshold (informational; exit 0 regardless).
```

Optional flags:

- `--baseline PATH` — diff against a different baseline file.
- `--threshold 0.05` — change the threshold (default 0.10).

## Drift semantics

- **Relative drift:** `(current - baseline) / baseline`. A row is flagged when
  `|rel_diff| > threshold` (default 10%).
- **Zero-baseline guard:** if the baseline value is zero (not expected for any
  pinned key in the anchor scenario, but guarded anyway), the row reports
  absolute drift only and is flagged.
- **The threshold is informational.** A flagged row signals that the developer
  should investigate. Either the change in this PR is the cause and is
  legitimately fixing or breaking equivalence; or the drift is expected
  feature-misalignment that requires a tracking issue per `MIGRATION_PLAN.md`
  §Implementation conventions item 2. The PR is not blocked by drift.

## When to refresh the baseline

- After a new patch release of v2.3 lands on `main` and is merged into
  `v3.0-dev`.
- After an explicit decision that drift introduced by a milestone is the new
  target (i.e., feature-misalignment that has been investigated and accepted).
- Otherwise: don't. Stable baseline = stable signal.

## CI

CI runs:

- The pytest smoke test (`tests/test_regression.py:test_anchor_runs`) which
  imports `anchor.run_and_summarize` and exercises the sim end-to-end.
- `python regression/compare.py` (no-baseline mode) which proves the CLI
  imports and parses arguments cleanly.

Neither step fails on drift. Drift is a developer-local concern.
