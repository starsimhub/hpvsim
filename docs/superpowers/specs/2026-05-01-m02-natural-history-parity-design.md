# M02: Natural History Parity — Design

**Date:** 2026-05-01
**Milestone:** M02 (Natural history parity)
**Branch:** `m02-natural-history-parity` (off `v3.0-dev`, after M01 PR merges)
**Predecessor:** [M01 Basic Transmission Sim](2026-04-28-hpvsim-m1-basic-transmission-design.md)
**Status:** Implemented; see "Post-implementation deltas" below.

---

## Goal

Bring HPV16 natural history into the v3 disease module: precancerous infection
(precin), cervical intraepithelial neoplasia (CIN), invasive cancer, and cancer
death. The single-genotype HPV16 trajectory must match v2.x's HPV16-only run
within the regression-gate tolerance against a v2 1-genotype baseline.

M02 also picks up the auxiliary plumbing required for natural-history results
to be epidemiologically meaningful at multi-decade horizons: population
scaling (`pop_scale` / `total_pop`) and age-specific migration. (The age-
stratified analyzer originally listed here was deferred to the calibration
milestone — see post-implementation deltas.)

This milestone is the headline natural-history work for v3. Multi-genotype
dynamics and cross-immunity are deferred to M03; calibration to M04.

## Scope

**In scope:**
- Disease progression for HPV16 inside `hpv.HPV(ss.Infection)`: precin/CIN/cancer
  states, full-trajectory sampling at infection time, scheduled state
  transitions, cancer-caused death.
- `pop_scale` / `total_pop` plumbing through `hpv.SimPars`.
- `hpv.AgeMigration(ss.Demographics)` — age-specific net-migration adapter
  ported from v2's `_v2_legacy/people.py:check_migration`.
- Slimming `hpvsim/parameters.py` to the starsimhub-conventional shape
  (`SimPars`, `GenotypePars`); migrating remaining v2 content to
  `_v2_legacy/parameters.py` or to data files.
- Auditing `hpvsim/utils.py`: replacing v2 helpers with starsim-native
  equivalents where they exist; quarantining the rest.
- Extended regression coverage: 8-metric `short_summary` parity gate; unit
  tests for vendored progression math.
- Re-running the M01 partnership-equivalence test to confirm the network gates
  still hold (or have tightened) once age-specific migration changes the
  population composition.

**Explicitly out of scope (deferred):**
- Multi-genotype dynamics, cross-immunity (M03).
- Calibration loop (M04).
- Vaccination, screening, treatment, dynamic_pars (M05+).
- Multiscale dynamic agent spawning. v2 spawns extra cancer agents in
  `set_prognoses` to amplify rare events; v3 uses simple multiplicative
  `pop_scale` only. **Tracking issue (M02):** revisit before v3.0.0 if the
  natural-history acceptance test exhibits high-variance tails on cancer
  metrics.
- Network-equivalence tightening beyond M01's 50% gate. M02 only re-measures
  to detect drift. If the gate is still loose post-M02, a follow-up issue is
  filed; production-network tightening + the `LegacyV2SexualNetwork` parity
  test land in a later milestone branch.
- Sex-specific initial prevalence — already implemented in M01
  (`hpvsim/hpv.py:_INIT_HPV_PREV_M`/`_F`). Listed here only to note completion.
- All other analyzers (`snapshot`, `age_pyramid`, `age_causal_infection`,
  `dalys`) — M09.

---

## Architecture

### Disease module: `hpv.HPV(ss.Infection)`

Trajectory-based natural history, lifted directly from v2's
`_v2_legacy/people.py:set_prognoses` algorithm and translated to starsim's
standard `set_prognoses` / `step_state` / `step_die` shape (per
[starsim disease pattern 2: SEIR with exposed state](
https://docs.starsim.org/) — same lifecycle, more compartments). Single
class, single file.

**Why trajectory-based, not per-step Markov:** v2's `cancer_fn = cin_integral`
is a function of the *full* `dur_cin` duration, i.e. an integral over the
trajectory; per-step transition hazards would diverge mathematically from v2.
Trajectory sampling at infection time matches v2 algorithmically.

**New states (`define_states`):**

| Kind | Name | Purpose |
|---|---|---|
| `BoolState` | `precin` | Precancerous infection compartment |
| `BoolState` | `cin` | Cervical intraepithelial neoplasia |
| `BoolState` | `cancerous` | Invasive cancer (no longer infectious) |
| `FloatArr` | `ti_cin` | Scheduled time of CIN onset |
| `FloatArr` | `ti_cancerous` | Scheduled time of invasive cancer onset |
| `FloatArr` | `ti_dead_cancer` | Scheduled time of cancer-caused death |

M01 states preserved: `infected`, `susceptible`, `ti_infected`,
`ti_first_infection`, `ti_clearance`. SIS clearance path remains as the
"cleared without progression" branch.

**New pars (`define_pars`):**

| Par | Source (v2) | Type |
|---|---|---|
| `dur_precin` | `parameters.py:337` lognormal(par1=3, par2=9) | `ss.lognorm_ex` |
| `dur_cin` | `parameters.py:339` lognormal(par1=5, par2=20) | `ss.lognorm_ex` |
| `dur_cancer` | `parameters.py:96` lognormal(par1=8, par2=3) | `ss.lognorm_ex` |
| `cin_fn` | `parameters.py:338` `dict(form='logf2', k=0.3, x_infl=0, ttc=50)` | dict |
| `cancer_fn` | `parameters.py:340` `dict(method='cin_integral', transform_prob=2e-3)` | dict |

**`set_prognoses(uids, sources)` — trajectory sampling:**

```
super().set_prognoses(uids, sources)   # M01: infected/ti_infected/ti_first_infection
1. dur_precin = pars.dur_precin.rvs(uids)
2. precin[uids] = True
3. p_cin = compute_severity(dur_precin, cin_fn)
4. cin_uids = uids subset where bernoulli(p_cin) AND female
5. nocin_uids = uids \ cin_uids                 # incl. all males
   ti_clearance[nocin_uids] = ti + dur_precin[nocin_uids]

For cin_uids:
  ti_cin[cin_uids] = ti + dur_precin[cin_uids]
  dur_cin = pars.dur_cin.rvs(cin_uids)
  p_cancer = compute_severity(dur_cin, cancer_fn)        # cin_integral
  cancer_uids = cin_uids subset where bernoulli(p_cancer)
  nocancer_uids = cin_uids \ cancer_uids
  ti_clearance[nocancer_uids] = ti_cin + dur_cin[nocancer_uids]

  For cancer_uids:
    ti_cancerous = ti_cin + dur_cin
    dur_cancer = pars.dur_cancer.rvs(cancer_uids)
    ti_dead_cancer = ti_cancerous + dur_cancer
```

**Sex specificity:** CIN and cancer are female-only in v2. Males in
`set_prognoses` go directly into the SIS branch and clear after `dur_precin`.

**`step_state()` — execute scheduled transitions:**

```
# (existing M01 SIS clearance path, on infected & not progressing)
cleared = self.infected & self.precin & ~self.cin & ~self.cancerous \
          & (self.ti_clearance <= self.ti)
flip cleared agents back to susceptible; reset precin

# Progression transitions
to_cin = self.precin & (self.ti_cin <= self.ti)
flip: precin = False; cin = True

to_cancerous = self.cin & (self.ti_cancerous <= self.ti)
flip: cin = False; cancerous = True
      infected = False; susceptible = False    # cancer agents not re-infectable
self.rel_trans[to_cancerous] = 0               # and no longer transmitting

to_dead = self.cancerous & (self.ti_dead_cancer <= self.ti)
sim.people.request_death(to_dead)   # standard starsim cancer-death wiring
```

**`step_die(uids)`:** call `super()`, then reset `precin`, `cin`, `cancerous`,
plus M01's `infected`/`susceptible` if not already handled by parent. Required
for any custom `BoolState` per starsim disease pattern 3.

**Cancer-death mechanism:** `sim.people.request_death(uids)` — disease-caused
death goes through People's death pipeline. `ss.Deaths` continues to handle
background mortality independently.

### Vendored progression math

`compute_severity`, `compute_severity_integral`, `logf2`, `cancer_fn ==
cin_integral` — these are HPV-specific math functions, not generic utilities.
They live as private module-level functions inside `hpv.py`. If `hpv.py`
exceeds ~400 LOC during implementation, they get split into a sibling
`hpvsim/_progression.py`; that's a judgment call during execution, not a
gate.

Source: v2's `hpvsim/parameters.py:685` (`compute_severity`),
`hpvsim/utils.py:101` (`logf2`), `hpvsim/utils.py:193` (`transform_prob`).

### Sim parameters: `hpvsim/parameters.py`

Rewrite to the starsimhub-conventional slim shape (mirrors
`stisim/parameters.py`, `fpsim/parameters.py`):

```python
class SimPars(ss.SimPars):
    """HPV-specific defaults on top of ss.SimPars."""
    # n_agents, total_pop, pop_scale, location, start, stop, dt, rand_seed, etc.

class GenotypePars(ss.Pars):
    """Per-genotype natural-history pars.
    M02: HPV16 only. M03 wires hpv18 / hi5 / ohr defaults."""
    # dur_precin, cin_fn, dur_cin, cancer_fn, rel_beta, sero_prob

genotype_aliases = {'hpv16': ['hpv16', '16'], ...}   # carried from v2

def get_genotype_pars(genotype='hpv16') -> GenotypePars:
    """Factory for per-genotype defaults; M03 multi-genotype consumer."""
```

The 845-LOC v2 file's content is partitioned:
- M02-relevant pars (HPV16 progression, sim defaults) → migrate into
  `SimPars` / `GenotypePars`.
- Pure data (regional pars, partner-mixing matrices) → migrate to
  `hpvsim/data/` JSON or CSV files alongside existing country data.
- Future-milestone content (intervention defaults, calibration pars) →
  `_v2_legacy/parameters.py` for porters in M05+.

`GenotypePars` (vs. inlining everything into `HPV.define_pars`) is the right
shape for M03's `hpv.Sim(genotypes=[...])` factory: M03 wants to look up
per-genotype defaults from a structured object, not parse `hpv.HPV.pars`
defaults.

### Utilities: `hpvsim/utils.py`

Audit step: replace v2 helpers with starsim-native equivalents where they
exist; keep what genuinely has no starsim counterpart; quarantine the rest.

| v2 helper | Replacement |
|---|---|
| `hpu.binomial_arr`, `hpu.sample` | `ss.bernoulli`, `ss.lognorm_ex`, `ss.choice` (some already migrated in M01) |
| `hpu.true`, `hpu.false`, `hpu.itrue` | starsim `BoolArr.uids` / `.notnan` patterns |
| `hpu.set_seed` | sim-level seed via `ss.options` / `ss.Sim(rand_seed=...)` |
| `hpu.invlogit`, `hpu.logf2`, `hpu.transform_prob` | colocate with `hpv.py` (HPV-specific math, not generic) |
| Helpers without starsim equivalents | keep in `utils.py` |

Target: `hpvsim/utils.py` ends M02 at <100 LOC.

### Population scaling

`SimPars.total_pop` (target real-world population the agents represent) and
`SimPars.pop_scale = total_pop / n_agents` (computed at init). Applied as a
multiplicative result-scaling factor on cumulative outcomes (cancers, cancer
deaths, infections). No multiscale dynamic agent spawning; that branch of
v2's `set_prognoses` is dropped.

Mirrors stisim's `SimPars`:

```python
self.total_pop = None   # If defined, used for calculating the scale factor
self.pop_scale = None   # How much to scale the population
```

### Demographics: `hpv.AgeMigration(ss.Demographics)`

Lifts v2's `_v2_legacy/people.py:check_migration` algorithm — *age-pyramid
pinning*, not rate-based migration. Each timestep:

1. Look up the target age pyramid for the current year from `pop_age_trend`
   (year × single-age × sex counts in real-world units).
2. Compute `scale = sim.n_agents / pop_at_sim_start` from `pop_trend`
   (year × total-pop trajectory).
3. For each (sex, integer age):
   - `count_sim` = alive sim agents at this exact age × sex.
   - `count_target = age_dist[sex][age] * scale`.
   - `diff = round(count_target - count_sim)`.
   - If `diff > 0`: add `diff` immigrants at that exact age, HPV-naive.
   - If `diff < 0`: weight-pick `|diff|` agents at that age and request
     their death (treated as emigration).

Skips silently when current year is outside `pop_trend`'s range (matches
v2). Immigrants enter HPV-naive — all disease BoolStates default `False`,
which mirrors v2's `add_births` behavior (it only seeds HIV, never HPV
state, when called from `check_migration`).

Required for multi-decade demographic realism when computing cancer
incidence rates. The two data tables (`pop_trend`, `pop_age_trend`) are
already loadable via the v3-active `hpvsim/data/loaders.get_total_pop` and
`get_age_distribution_over_time` helpers; M02 wires them through
`load_country()` alongside the existing `age_data` / `birth_rate` /
`death_rate` keys.

### Analyzer: deferred to calibration milestone

The originally-planned `hpv.AgeResults(ss.Analyzer)` was deferred — see
post-implementation deltas. The full v2 `age_results` surface (multi-key,
`compute_fit`, `reduce`) will be ported as part of the calibration
milestone where its only real consumer lives.

### Public API impact

`hpvsim/__init__.py` already does `from . import parameters` and `from .
import utils`. No active v3 module currently imports symbols from either
(M01's hpv.py only references `parameters.py` in doc comments), so the
parameters/utils slimming has zero internal-refactor surface. Add
`from .parameters import SimPars, GenotypePars` for ergonomics.

---

## Validation gates

Following migration plan §Implementation conventions item 2 (dual validation
gates).

### Development gate (per PR / per merge to milestone branch)

Anchor scenario: same as M01 — `tests/regression/anchor_hpv16.py`,
single-genotype HPV16, Nigeria, fixed seed 0, no interventions, 1990–2060.
With M02 the natural-history machinery is on, so the cancer-related summary
metrics populate non-zero.

**Pinned summary set (8 metrics + total population):** matches v2's
`compute_summary` (`_v2_legacy/sim.py:1179`).

| Metric | M01 value | M02 expectation |
|---|---|---|
| total HPV infections | tracked | tracked |
| total cancers | 0 (not modeled) | non-zero, ±10% vs. v2 baseline |
| total cancer deaths | 0 | non-zero, ±10% vs. v2 baseline |
| mean HPV prevalence (%) | tracked | tracked |
| mean cancer incidence (per 100k) | 0 | non-zero, ±10% vs. v2 baseline |
| mean age of infection (years) | tracked | tracked |
| mean age of cancer (years) | undefined | non-zero, ±10% vs. v2 baseline |
| mean age of cancer death (years) | undefined | non-zero, ±10% vs. v2 baseline |
| total population | tracked | tracked |

**Threshold:** ±10% relative drift per metric. Informational, not
auto-blocking, per migration convention 2 — on failure the PR carries either
a fix or an explicit drift-classification note + tracking issue for
re-convergence.

**Baseline regeneration:** v2 baseline (`tests/regression_baselines/anchor.json`)
regenerated locally against v2.3 by running `tests/regression/baseline.py`
inside the v2.3 environment. Baseline files stay gitignored per migration
plan §Branching and sync strategy.

### Capability test

Originally planned: age-stratified cumulative cancers + cancer deaths at
end-of-sim by 5-year age band, against the v2 baseline. Deferred to the
calibration milestone alongside the AgeResults analyzer it depended on.

### Release gate

Per migration plan: overlapping uncertainty intervals against the
analysis-repo suite. Not exercised at M02 — that's M04+ once calibration
exists. M02's release-gate contribution is "natural history is in place so
the calibration loop has something to optimize."

### Network re-measure (M01 follow-up)

Re-run `tests/test_partnership_equivalence.py` from M01 unchanged. Three
outcomes:

| Outcome | Action |
|---|---|
| Gates still pass at the M01 50% threshold | Document in M02 PR; close out the M02 network-tightening note as "no regression, follow-up not needed yet." |
| Gates tighten on their own (lower drift) | Document the new measurements; consider tightening the threshold in M03 once re-measured under multi-genotype. |
| Gates loosened or fail | Block merge until investigated. The M02 changes that *could* affect network metrics are: (a) age-specific migration alters the pool of pair-eligible agents over time; (b) `parameters.py` refactor — if anything network-related accidentally moved. |

---

## Tests

| Test | Type | Status |
|---|---|---|
| `tests/regression/anchor_hpv16.py` `run_and_summarize()` | Extended (3 → 8 metrics) | Modified |
| `tests/regression/baseline.py` | Unchanged code; rerun against v2.3 env | Unmodified |
| `tests/test_regression.py` | ±10% drift gate, covers extended summary | Unmodified |
| `tests/test_partnership_equivalence.py` | M01 network re-measure | Unmodified |
| `tests/test_natural_history.py` | New — lifecycle smoke (single-agent trajectory through precin → cin → cancerous → dead) | New |
| `tests/test_progression_math.py` | New — unit tests for `compute_severity`, `logf2`, `cin_integral`. Pinned outputs against v2 (one-shot fixture from v2.3 env, checked in) | New |

Demo: `tests/regression/demo_anchor_hpv16.py` extended with CIN-prevalence
and cancer-incidence-by-year trajectories alongside the existing prevalence
plot.

---

## Sub-task ordering

CI must be green at every commit (migration convention 1). Phases below are
ordered to maintain that invariant — plumbing first, then progression on top
of plumbing, then validation last.

**Branching prerequisite:** open M01 PR (`m01-basic-transmission-sim` →
`v3.0-dev`) and wait for merge before starting M02. Then:
`git checkout v3.0-dev && git pull && git checkout -b m02-natural-history-parity`.

| Phase | Sub-task | Purpose |
|---|---|---|
| **A. Slim plumbing** | A1 | Rewrite `parameters.py` → `SimPars` + `GenotypePars`; migrate v2 content to `_v2_legacy/parameters.py` or `data/`. |
| | A2 | Audit `utils.py`; replace with starsim-native helpers; quarantine remainder. |
| **B. Progression** | B1 | Vendor progression-math helpers (`compute_severity`/`logf2`/`cin_integral`) inside `hpv.py`. |
| | B2 | Add new states (`precin`/`cin`/`cancerous` + `ti_*` + `dur_*`) and pars to `hpv.HPV`. |
| | B3 | Implement trajectory sampling in `set_prognoses`. |
| | B4 | Implement progression transitions in `step_state`. |
| | B5 | Update `step_die` to reset all new BoolStates. |
| | B6 | Thread `total_pop` / `pop_scale` through `SimPars`; apply to result-scaling. |
| **C. Demographics** | C1 | `hpv.AgeMigration(ss.Demographics)` adapter. |
| **D. Validation** | D1 | Extend `anchor_hpv16.run_and_summarize()` to 8 metrics. |
| | D2 | Regenerate v2 baseline locally against v2.3 env. |
| | D3 | Add `tests/test_natural_history.py` (lifecycle smoke). |
| | D4 | Add `tests/test_progression_math.py` (unit math). |
| | D5 | Re-run `tests/test_partnership_equivalence.py`; document outcome. |
| **E. Close-out** | E1 | Extend `demo_anchor_hpv16.py` with CIN/cancer trajectories. |
| | E2 | Open M02 PR to `v3.0-dev`. |

---

## Definition of done

- All sub-tasks A1–E2 complete.
- CI green; runnability invariant (`hpv.Sim().run()` works) holds at every
  commit per migration convention 1.
- Anchor regression passes ±10% per metric on the 8-metric `short_summary` +
  total population, OR PR carries an explicit drift-classification note +
  tracking issue per migration convention 2.
- Partnership-equivalence test re-run; outcome documented in PR.
- Tracking issues filed for:
  - Multiscale dynamic-spawning revisit before v3.0.0 (acceptance: assess
    cancer-metric tail variance under M02 acceptance run; if unacceptable,
    reintroduce v2's spawning branch in a later milestone).
  - Any subclass-first delegations introduced (per migration convention 3,
    must strip before M10).
  - M02 network-tightening follow-up if D5 found gates still loose at 50%.
  - Calibration milestone: port the full v2 ``age_results`` surface (multi-
    key, ``compute_fit``, ``reduce``) and reinstate the age-stratified
    cancer capability gate.

---

## Post-implementation deltas

What landed differs from this design in the following ways. The plan doc
captures the per-task narrative; this section is the spec-level diff.

**Same-genotype partial permanent immunity** (added during M02; not in
original scope):
- New pars: `imm_init=0.35` (transmission immunity cap on `rel_sus`),
  `cell_imm_init` (Beta sample, ported from v2 `beta_mean(0.25, 0.025)`),
  `age_risk=dict(age=30, risk=2)` (older-women dur_cin multiplier).
- New states: `rel_sev` (per-agent biological severity baseline, sampled
  from v2's `sev_dist`), `rel_sev_sampled` (init tracker), `sev_imm`
  (severity immunity, max-of-Beta-samples on each clearance).
- `set_prognoses` applies `dur_precin = sample * (1 - sev_imm)` for
  females; `_compute_severity` consumes `rel_sev` separately so the
  effective duration is `dur * (1 - sev_imm) * rel_sev` (two-factor
  product matching v2).
- `step_state` clearance branches reduce `rel_sus` to
  `min(prior, 1 - imm_init)` and accumulate `sev_imm` via
  `np.maximum(prior, new sample)`.

**Male clearance** uses a separate, much shorter distribution
`dur_inf_male` (lognormal mean=1y, std=1y); without it, males stayed
infected far longer than v2 and inflated cancer outcomes via
secondary infections.

**Per-step Result counters** replace lifetime BoolStates for cancer
event tracking. `HPV.step_state` emits, and `finalize_results` rolls up:
- `new_cancers` / `new_cancer_deaths` — realized-event counts per step.
- `cum_cancers` / `cum_cancer_deaths` — cumulative sums (matches the
  starsim `Infection.cum_infections` idiom for calibration consumers).
- `sum_age_at_cancer` / `sum_age_at_cancer_death` — age sums at
  transition; mean = sum / count.

The originally-planned `dur_precin` and `dur_cin` per-agent FloatArrs
were removed: realized durations are recoverable from the existing
`ti_*` timestamps and no consumer required the standalone arrays.

**Single source of truth for HPV16 defaults.** `GenotypePars` (rewritten
to hold starsim-native `ss.Dist` instances rather than v2-shaped dicts)
now carries every par the disease module needs: `beta`, all duration
distributions, `cin_fn`/`cancer_fn`, immunity (`imm_init`,
`cell_imm_init`), and `age_risk`. `HPV.__init__` pulls defaults via
`get_genotype_pars(genotype)`; each call returns fresh `Dist` instances
so per-genotype RNG slots stay independent (forward-compatible with
multi-genotype). `rel_beta` and `sero_prob` are reserved on
`GenotypePars` for future multi-genotype consumers.

**Clearance is one branch, not two.** `step_state` clears precin and CIN
in a single block (`infected & (precin | cin) & ~cancerous &
ti_clearance <= ti`); the compartments are mutually exclusive, so one
beta sample per cleared agent suffices for the `sev_imm` accumulator.

**CRN-safe emigration.** `AgeMigration._emigrate` selects departing
agents via an `ss.choice(replace=False)` distribution so the emigration
draws are reproducible per-seed (not pulled from numpy's global RNG).

**`AgeResults` analyzer deferred to calibration milestone.** The
minimum-scope cancer-only port originally listed as a deliverable was
removed from M02. Rationale: its only consumer was its own test suite —
the smoke tests in ``tests/test_analyzers.py`` and a
capability gate in ``test_natural_history.py`` that depended on a v2
baseline JSON that was never generated. The v2 ``age_results`` surface
in ``_v2_legacy/analysis.py`` is much wider (multi-key results,
``compute_fit``, ``reduce``) and is the calibration loop's primary
entry point; porting it once at calibration time avoids shipping a
partial implementation that would be replaced.

**SexualNetwork refactor:** the multi-network design (separate `m` and
`c` instances) was collapsed into a single `hpv.SexualNetwork` that
carries all layers in one edges table tagged by `layer_id`. `debut` and
`participant` are shared per-agent across layers, and `step()` dissolves
all pairs (single `end_pairs`) before forming new ones per layer.

**Partnership-equivalence gates** were tightened from M01's 50%
placeholder to 10% drift / 0.90 cosine and pass.

---

## Open questions / follow-ups

- **`hpv.py` size:** if it crosses ~400 LOC during B-phase, split progression
  math into `hpvsim/_progression.py`. Judgment call during execution, not a
  pre-commit gate.
- **Cancer-death wiring:** `sim.people.request_death(uids)` is the assumed
  pattern from starsim disease conventions. If implementation discovers a
  cleaner starsim idiom (e.g., setting `ti_dead` on a built-in `ss.Disease`
  field that `step_die` picks up automatically), use that and amend this
  spec.
- **Lognormal parametrization mapping:** v2 declares durations as
  `dict(dist='lognormal', par1=X, par2=Y)` consumed by `hpu.sample`; M01
  uses `ss.lognorm_ex(mean=...)`. The exact translation
  (par1=mean+par2=std vs. par1=μ+par2=σ on the log scale) must be
  confirmed against v2's `hpu.sample` implementation during B2 to keep
  the v2 baseline comparison apples-to-apples. If the v2 parametrization
  is μ/σ on the log scale, use `ss.lognorm_im(meanlog=..., sigmalog=...)`
  instead of `ss.lognorm_ex`.