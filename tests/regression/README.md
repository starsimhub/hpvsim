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

## M01 1-genotype baseline (`anchor_hpv16.json`)

The M01 anchor (`anchor_hpv16.py`) compares against a v2 hpvsim run configured
with `genotypes=['hpv16']` and otherwise identical pars to the M00 4-genotype
anchor. This baseline is gitignored at:

```
tests/regression_baselines/anchor_hpv16.json
```

### Generating the baseline

In a v2-only environment (a separate venv with v2.3.x hpvsim from PyPI, or a
checkout of the `rc2.3` branch of this repo before the v3 migration):

```python
import json
import sciris as sc
import hpvsim as hpv2  # v2 package

PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=['hpv16'],
    start=1990,
    end=2060,
    dt=0.5,
    burnin=20,
    rand_seed=0,
    verbose=0,
)

sim = hpv2.Sim(sc.dcp(PARS))
sim.run()

# v2 result-key names may differ from v3's hpv.Sim().results.hpv16.* keys.
# Compute the equivalent quantities from v2's underlying time series and
# store them under the v3 key names (see tests/regression/anchor_hpv16.py
# for the exact keys: 'total HPV infections', 'mean HPV prevalence (%)',
# 'mean age of infection (years)').

short = {
    'total HPV infections': float(sim.results['total_infections'].sum()),
    'mean HPV prevalence (%)': 100 * float(sim.results['hpv_prevalence'].mean()),
    'mean age of infection (years)': float(sim.results['mean_age_infection'].mean()),
}
total_pop = float(sim.results['n_alive'][-1])

out = dict(summary={**short, 'total population': total_pop}, pars=PARS)
with open('tests/regression_baselines/anchor_hpv16.json', 'w') as f:
    json.dump(out, f, indent=2)
```

The summary keys **must match v3's `run_and_summarize()` output keys exactly**.
If v2's result names differ, compute the equivalent quantities from v2's
underlying time series and store them under the v3 key names.

### Partnership-equivalence baseline (`partnership_v2.json`)

Supports the M01 acceptance gate (`tests/test_partnership_equivalence.py`,
landing in Task 13). v2 has 2 layers (m, c) — partnership_v2.json uses
both. Recipe:

```python
import json
import numpy as np
import sciris as sc
import hpvsim as hpv2

PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=['hpv16'],
    start=1990,
    end=2015,
    dt=0.5,
    burnin=20,
    rand_seed=0,
    verbose=0,
)

sim = hpv2.Sim(sc.dcp(PARS))
sim.run()

# Capture per-layer (m, c) mixing matrix (16x16 for 5y bins, 0-80,
# female × male), concurrency histogram, and partnership-duration samples.
# v2 stores these on sim.people in layer-keyed structures; consult v2
# internals for exact attribute names.

out = {}
for layer in ('m', 'c'):
    out[layer] = {
        'mixing_matrix': ...,      # 2d list, 16x16, density-normalized
        'concurrency_hist': ...,   # 1d list, indexed by n_concurrent_partners
        'duration_samples': ...,   # 1d list, completed-edge durations in years
    }
with open('tests/regression_baselines/partnership_v2.json', 'w') as f:
    json.dump(out, f)
```

The M01 partnership-equivalence test reads this JSON and runs KS-tests +
bin-wise diff against equivalent quantities produced by v3.
