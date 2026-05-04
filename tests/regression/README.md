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
| `dt` | `0.25` |
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
worktree on the `rc2.3` branch of this repo with v2's hpvsim installed in
that venv). The script below is the verified-working version used for the
initial M01 baseline. Save as `gen_anchor_hpv16.py` in the v2 environment:

```python
import json, os, sys
import numpy as np
import sciris as sc
import hpvsim as hpv2  # v2 package

# Pinned PARS — must mirror the v3 anchor (tests/regression/anchor_hpv16.py).
# pop_scale=1 + total_pop=n_agents disables v2's pop-scaling so absolute
# counts are agent-level (v3 doesn't yet wire pop_scale; that's M02 work).
PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=['hpv16'],
    start=1990,
    end=2060,
    dt=0.25,  # Must match v2's default and the v3 anchor_hpv16.py PARS
    burnin=20,
    rand_seed=0,
    verbose=0,
    pop_scale=1,
    total_pop=10000,
)

sim = hpv2.Sim(sc.dcp(PARS))
sim.run()
res = sim.results

# total HPV infections — sum 'infections' (per-step new infections, 1 gt → 1D)
n_inf = float(np.asarray(res['infections']).sum())

# mean HPV prevalence (%) — mean of 'hpv_prevalence' (instantaneous per-step)
mean_prev_pct = 100 * float(np.asarray(res['hpv_prevalence']).mean())

# mean age of (first) infection — v2 doesn't expose this as a result key;
# compute from sim.people.date_infectious (per-genotype per-agent timestep
# of first HPV acquisition). Subtract from current year-of-birth via age.
people = sim.people
date_inf = np.asarray(people.date_infectious[0])  # shape: (1, n_agents) → row 0
ever_infected = ~np.isnan(date_inf)
if ever_infected.any():
    year_at_inf = PARS['start'] + date_inf[ever_infected] * PARS['dt']
    current_year = PARS['end']
    current_ages = np.asarray(people.age)[ever_infected]
    year_of_birth = current_year - current_ages
    ages_at_inf = year_at_inf - year_of_birth
    valid = (ages_at_inf > 0) & (ages_at_inf < 100)
    mean_age_inf = float(ages_at_inf[valid].mean()) if valid.any() else 0.0
else:
    mean_age_inf = 0.0

total_pop = float(np.asarray(res['n_alive'])[-1])

short = {
    'total HPV infections': n_inf,
    'mean HPV prevalence (%)': mean_prev_pct,
    'mean age of infection (years)': mean_age_inf,
}
print('v2 1-genotype HPV16 anchor summary:')
for k, v in short.items():
    print(f'  {k:<40} {v:>12.4g}')
print(f'  {"total population":<40} {total_pop:>12.4g}')

out = dict(
    summary={**short, 'total population': total_pop},
    pars={k: (v if not isinstance(v, float) else float(v)) for k, v in PARS.items()},
)
target = sys.argv[1] if len(sys.argv) > 1 else (
    'tests/regression_baselines/anchor_hpv16.json'
)
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(target, 'w') as f:
    json.dump(out, f, indent=2)
print(f'Wrote: {target}')
```

The summary keys **must match v3's `run_and_summarize()` output keys exactly**.
If v2's result names differ from those above, compute the equivalent
quantities from v2's underlying time series and store them under the v3 key
names.

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
    dt=0.25,  # Must match v2's default
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

## M02 baseline regeneration

The M02 milestone extends `short_summary` from 3 keys (M01) to 8 keys
covering HPV + CIN/cancer trajectories (`total cancers`, `total cancer
deaths`, `mean cancer incidence (per 100k)`, `mean age of cancer`,
`mean age of cancer death`). M01-era 3-key baselines at
`tests/regression_baselines/anchor_hpv16.json` are incompatible with the
M02 drift gate.

To regenerate against v2.3:

1. **Set up a v2.3 environment** alongside this repo. Two options:
   - Local clone: `C:/Users/ryanhu/PycharmProjects/hpvsim_v23_frozen` (already
     present on the user's machine) — activate that repo's venv.
   - Fresh venv: `python -m venv .v23-venv && .v23-venv/Scripts/pip install
     hpvsim==2.3` (Windows) or `.v23-venv/bin/pip install hpvsim==2.3`
     (Linux/macOS).

2. **Update the baseline-generation script (if needed)** to call v2.3's
   `Sim(...)` API. v2.3's constructor signature differs from v3's — e.g.
   `genotypes=['hpv16']` (a list) vs. v3's `genotype='hpv16'`
   (single). For the M02 1-genotype anchor, pass `genotypes=['hpv16']` to v2.3
   to match the M02 anchor.

3. **Run** the v2.3 baseline-generation script to produce
   `anchor_hpv16.json` with all 8 keys filled in. The summary keys must
   match what `tests/regression/anchor_hpv16.py:run_and_summarize()`
   produces. Use the template provided in the M01 section above as a starting
   point, but compute all 8 M02 keys (cancer counts, ages, incidence rates).

4. **Place the result at** `tests/regression_baselines/anchor_hpv16.json`.
   The path is gitignored by `.gitignore` (`tests/regression_baselines/*.json`)
   so the baseline file stays local-only.

5. **Run the drift gate**:

   ```
   pytest tests/test_regression.py::test_anchor_hpv16_drift -v
   ```

   Expected: PASS within ±10% per metric, OR FAIL with a printed list of
   out-of-tolerance metrics. Per migration convention 2 the gate is
   informational, not auto-blocking — on failure the PR carries either
   a fix or an explicit drift-classification note + tracking issue.

### M02-specific notes

- `total cancer deaths` may be 0 in the M02 anchor scenario (1990–2060)
  because cancer durations average ~8 years and the last cancer-onset events
  fire in the late 2050s, leaving no time for cancer deaths to realize.
  Both v2.3 and v3 should agree on this — verify after regenerating.
- Mean ages of cancer / cancer death rely on starsim freezing `people.age`
  at agent death. If that assumption changes in a future starsim version,
  recompute against the actual people-age semantics.
- **dt must be 0.25** in both v3 and v2 baseline-gen scripts. v2's default
  sim timestep is `dt=0.25` (quarterly), declared at
  `_v2_legacy/parameters.py:61`. The M02 anchor (and any v2 baseline-regen
  script) must use the same `dt=0.25` so both runs use v2's default-driven
  calibrations. If you regenerate the v2 baseline, update the generation
  script's `PARS` dict to `dt=0.25`.
- **AgeMigration fires annually, not every sim step.** v2's
  `check_migration` ran annually via `update_freq = max(1, int(dt_demog/dt))
  = 4` at `dt=0.25` (fired once every 4 sim steps). v3 matches this by
  setting `dt=ss.year` on the `AgeMigration` module constructor, which causes
  `ss.Loop` to call `step()` only at integer-year timesteps regardless of the
  sim's own dt. Per-step immigration/emigration counts in v3 results reflect
  annual totals (1 firing/year), not quarterly (4 firings/year).

## M02 age-cancer capability baseline

Supports the M02 capability gate (`tests/test_natural_history.py::test_m02_capability_age_stratified_cancers`,
landing in Task 16). The baseline file is gitignored at:

```
tests/regression_baselines/m02_age_cancer.json
```

### Generating the baseline

Generate by running, in the v2.3 environment described in the M02 baseline-regeneration section above:

```python
import numpy as np
import json
import sciris as sc
import hpvsim as hpv  # v2.3 here

# Same anchor pars as the regression baseline
pars = dict(
    n_agents=10_000,
    location='nigeria',
    genotypes=['hpv16'],
    start=1990,
    end=2060,
    dt=0.25,  # Must match v2's default and the v3 anchor_hpv16.py PARS
    rand_seed=0,
    verbose=0,
)

# Capture age-stratified cancer incidence at end of sim (year 2059)
sim = hpv.Sim(sc.dcp(pars))
sim.run()

# v2.3 stores cancer_incidence_by_age as a 2D result array
# (time x age_bins). Extract the final year's column (index -1).
# v2's default age bins: 0, 5, 10, ..., 100 (21 bins).
arr = np.asarray(sim.results['cancer_incidence_by_age'])[:, -1]

import os
os.makedirs('tests/regression_baselines', exist_ok=True)
with open('tests/regression_baselines/m02_age_cancer.json', 'w') as f:
    json.dump({'cancer_incidence_by_age': arr.tolist()}, f, indent=2)
print(f'Wrote m02_age_cancer.json with {len(arr)} age bands')
```

The output shape must match `(n_bins,)` and use the same 5-yr age bins as
`hpv.AgeResults` defaults (0, 5, 10, ..., 100, so 21 bins total). The file
is gitignored.
