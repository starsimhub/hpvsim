# Regression harness

This directory holds the v2 → v3 migration regression harness used during the
HPVsim v3.0 port. It compares current v3 runs against stored v2.3 baselines
(per-key drift gates) or against a v2.3 multi-seed sweep (Welch-style z-score
gate over a v3 multi-seed sweep).

The harness is the **development gate** described in `MIGRATION_PLAN.md`
§Implementation conventions item 2. The **release gate** (overlapping
uncertainty intervals against the analysis-repo suite) is the scientific
gate and lives elsewhere.

## What's here

| File | Role |
|---|---|
| `anchor_hpv16.py` | M01 1-genotype HPV16 anchor: `make_sim()` + `run_and_summarize()` + `__main__` runner. Pars pinned to match v2.3 default `dt=0.25`. |
| `anchor_4genotype.py` | M03 4-genotype anchor (`[hpv16, hpv18, hi5, ohr]`): `make_sim()` + `run_and_summarize()` + `__main__` runner. Drives both single-seed parity and the multi-seed sweep. |
| `short_summary.py` | Builds the 40-entry summary dict (8 metrics × 4 genotypes + 8 aggregate-across-genotypes) used by the M03 parity gates. |
| `multi_seed_v3.py` | CLI: runs the M03 anchor across N seeds, writes per-seed summaries to `v3_seeds.json`. |
| `multi_seed_v2.py` | CLI: same idea but runs against a v2.3 hpvsim (must be invoked from a v2-only env). Writes `v2_seeds.json`. |
| `multi_seed_v2_trajectory.py` | v2-side trajectory capture (per-genotype, per-timestep) used by the M03 trajectory parity gate. |
| `compare_seeds.py` | CLI: pairs `v2_seeds.json` × `v3_seeds.json`, prints Welch-style z + v2-mean percentile in v3 distribution. |
| `drift.py` | Pure-function `compute_drift()` used by the M01/M02 per-key drift gates in `tests/test_regression.py`. |
| `time_v2_anchor.py` | Ad-hoc runtime profile against v2.3 (no JSON output; informational). |
| `profile_v3_anchor.py` | Ad-hoc runtime profile of the v3 anchor. |
| `methods_fig1.py` | Methods-paper figure script; not part of the gate. |
| `baseline_v23.py` | Helper for the v2.3 baseline-regeneration workflows documented below. |
| `__init__.py` | Empty; makes this directory an importable package. |

## Pytest gates (in `tests/`)

| Test | Compares | Baseline file |
|---|---|---|
| `test_regression.py::test_compute_drift_*` | Pure unit tests for `drift.py:compute_drift` | (none) |
| `test_regression.py::test_anchor_hpv16_runs` | Tier-2 smoke: M01 anchor runs end-to-end | (none) |
| `test_regression.py::test_anchor_hpv16_drift` (M02) | M02 8-metric drift gate | `anchor_hpv16.json` |
| `test_partnership_equivalence.py` | KS-tests + bin-wise diff on partnership distributions | `partnership_v2.json` |
| `test_natural_history.py::test_m02_capability_age_stratified_cancers` | Age-stratified cancer incidence parity | `m02_age_cancer.json` |
| `test_m03_short_summary_parity.py` | 40-metric z-score gate (10 v3 seeds vs 30 v2 seeds) | `v2_seeds_n30.json` |
| `test_m03_trajectory_parity.py` | Per-genotype trajectory parity | `v2_trajectories_n30.json` |

All baseline files live under `tests/regression_baselines/` and are
gitignored (`tests/regression_baselines/*.json`). Regenerate by following
the per-baseline sections below.

## M03 multi-seed flow

Pinned in `anchor_4genotype.py:PARS` (4 genotypes, Nigeria, seed 0,
1990–2060, `dt=0.25`, no interventions, no analyzers beyond the
auto-added HPVTotal analyzer).

Producing a v3 sweep:

```bash
python tests/regression/multi_seed_v3.py --n 10
```

Producing a v2.3 sweep (from a v2-only env, ~30–60s/seed):

```bash
"<v2 env>/python.exe" tests/regression/multi_seed_v2.py --n 30
```

Diff them ad-hoc with `compare_seeds.py` (prints per-metric z + percentile).
The CI-level gate is the pytest test `test_m03_short_summary_parity.py`,
which runs 10 v3 seeds in-process and compares against the committed
30-seed v2.3 baseline (`v2_seeds_n30.json`) at `|z| < 3` per metric.

## Drift semantics (M01/M02 per-key gates)

- **Relative drift:** `(current - baseline) / baseline`. A row is flagged
  when `|rel_diff| > threshold` (default 10%).
- **Zero-baseline guard:** if the baseline value is zero, the row reports
  absolute drift only and is flagged.
- **The threshold is informational.** A flagged row signals that the
  developer should investigate. Either the change in this PR is the cause
  and is legitimately fixing or breaking equivalence; or the drift is
  expected feature-misalignment that requires a tracking issue per
  `MIGRATION_PLAN.md` §Implementation conventions item 2. The pytest gate
  may pass or fail — read the printout, don't auto-block.

## When to refresh a baseline

- After a new patch release of v2.3 lands and is merged in.
- After an explicit decision that drift introduced by a milestone is the
  new target (i.e., feature-misalignment that has been investigated and
  accepted).
- Otherwise: don't. Stable baseline = stable signal.

## CI

CI runs the full pytest suite (`pytest test_*.py -n auto`). The drift gates
above run in-process and pass / fail like any other test. There is no
separate CLI smoke step.

## M01 1-genotype baseline (`anchor_hpv16.json`)

The M01 anchor (`anchor_hpv16.py`) compares against a v2 hpvsim run
configured with `genotypes=['hpv16']`. This baseline is gitignored at:

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

Supports the M01 acceptance gate (`tests/test_partnership_equivalence.py`).
v2 has 2 layers (m, c) — partnership_v2.json uses both. The working
generation script lives in the v2 frozen worktree at
`hpvsim_v23_frozen/gen_partnership_v2.py`; the PARS it uses are below.

```python
PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=['hpv16'],
    start=1990,
    end=2015,
    dt=0.25,                # Match v2 default (and v3 standard)
    burnin=20,
    rand_seed=0,
    verbose=0,
    pop_scale=1,            # Disable real-world scaling (match v3)
    total_pop=10000,        # Match n_agents so pop_scale stays 1
    ms_agent_ratio=1,       # Disable multiscale dynamic agent spawning. v3
                            # has no multiscale; v2's default ms_agent_ratio=10
                            # spawns level1 sub-agents on cancer events that
                            # otherwise inflate alive-agent slots and
                            # contaminate the comparison.
)
```

Two accounting traps when capturing v2 stats:

1. **`len(people.alive)` is the allocated-slot count, not alive-agent count.**
   The slot array grows via births / migration / multiscale and never shrinks
   when agents die, so dead-agent slots get bucketed into `0 partners` and
   inflate the apparent population. Filter by `np.asarray(people.alive)` to
   get the true alive-agent set, matching v3's
   `_capture_partnership_stats` which uses `people.alive.uids`.
2. **`dur` in v2's `to_df()` is the original-formation duration (years).**
   v3 stores remaining timesteps and reconstructs original via
   `(remaining + elapsed) * dt`. The two are directly comparable.

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

---

## M07 multi-seed anchors and baseline regeneration

M07 adds two additional milestone anchors with multi-seed parity gates,
mirroring M03's pattern but for the M01 (transmission-only HPV16) and M02
(full HPV16 natural history) milestones. **The naming convention follows
the canonical MIGRATION_PLAN milestones**:

| Anchor | PARS file | Summary builder | v2 baseline (gitignored) | v2 regen command |
|---|---|---|---|---|
| M01 | `anchor_m01.py` | `short_summary_m01.py::build_summary_m01` | `v2_m01_seeds_n30.json` | `python tests/regression/multi_seed_v2.py --anchor m01 --n 30` |
| M02 | `anchor_m02.py` | `short_summary.py::build_summary` (genotypes=`('hpv16',)`) | `v2_m02_seeds_n30.json` | `python tests/regression/multi_seed_v2.py --anchor m02 --n 30` |
| M03 (4-genotype) | `anchor_4genotype.py` | `short_summary.py::build_summary` | `v2_seeds_n30.json` | `python tests/regression/multi_seed_v2.py --anchor m03_4genotype --n 30` |
| M05 vx routine | `anchor_vx_routine.py` | (see M05 spec) | `v2_vx_routine_seeds_n30.json` | (see M05 docs) |
| M05 vx campaign | `anchor_vx_campaign.py` | (see M05 spec) | `v2_vx_campaign_seeds_n30.json` | (see M05 docs) |

### Naming-convention note

The earlier "M01 1-genotype baseline" section above documents
`anchor_hpv16.py` under legacy terminology. Under the current MIGRATION_PLAN
naming, that file corresponds to M02 (full HPV16 natural history). The
new M07 anchors (`anchor_m01.py` and `anchor_m02.py`) use the current
naming. Both sets of anchor files coexist — workflows depending on the
legacy `anchor_hpv16.py` continue to work.

### Regenerating from a v2 env

The baseline regen scripts run under hpvsim v2.3 in a separate Python env;
the active v3 env will not work. A local v2 env can be set up as a fresh
worktree of `rc2.3`:

```bash
git worktree add ../hpvsim_v23_frozen rc2.3
cd ../hpvsim_v23_frozen
python -m venv .venv-v23
.venv-v23/bin/pip install -e .
```

Then run, from the *v3* repo root, with the v2 env's Python:

```bash
../hpvsim_v23_frozen/.venv-v23/bin/python tests/regression/multi_seed_v2.py \
    --anchor m01 --n 30 \
    --out tests/regression/v2_m01_seeds_n30.json
```

(Adjust paths for Windows or your local setup.)

### Parity-gate test → baseline mapping

The slow parity-gate tests (marked `@pytest.mark.slow`) skip themselves
if the corresponding baseline JSON is missing. The skip message points at
the regen command for the specific anchor.

| Pytest test | Baseline file expected |
|---|---|
| `test_m01_short_summary_parity.py::test_m01_short_summary_parity` | `v2_m01_seeds_n30.json` |
| `test_m02_short_summary_parity.py::test_m02_short_summary_parity` | `v2_m02_seeds_n30.json` |
| `test_m03_short_summary_parity.py::test_short_summary_parity_4genotype` | `v2_seeds_n30.json` |
| (M05 vx parity tests) | (see M05 spec) |
