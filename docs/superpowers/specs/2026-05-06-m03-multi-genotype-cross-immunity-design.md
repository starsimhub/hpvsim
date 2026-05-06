# M03: Multi-genotype and Cross-Immunity — Design

**Date:** 2026-05-06
**Milestone:** M03 (Multi-genotype and cross-immunity)
**Branch:** `m03-multi-genotype-and-cross-immunity` (off `m02-natural-history-parity`; M03 PR targets `v3.0-dev`)
**Predecessor:** [M02 Natural History Parity](2026-05-01-m02-natural-history-parity-design.md)
**Status:** Drafted; not yet implemented.

---

## Goal

Bring HPVsim's four-genotype dynamics with cross-immunity into v3. Replicate
M02's HPV16 natural-history machinery across `[hpv16, hpv18, hi5, ohr]` as four
independent `ss.Disease` instances, and introduce
`hpv.CrossImmunity(ss.Connector)` to compute per-target `rel_sus` and `sev_imm`
from per-source `nab_imm` and `cell_imm` via v2's cross-protection matrices.
The 4-genotype trajectory must match v2.x's 4-genotype Nigeria run within the
regression-gate tolerance — measured per-genotype on the 8-metric `short_summary`
and in aggregate on cancer / infection time series against the M0 4-genotype
anchor.

## Scope

**In scope:**

- Per-genotype HPV instantiation: `hpv.Sim(diseases=[hpv.HPV(genotype='hpv16'), …])`
  primary; `hpv.Sim(genotypes=[16, 18, 'hi5', 'ohr'])` as sugar that constructs
  the four modules from `GenotypePars`.
- Splitting M02's `sev_imm` into a *raw source* state (`ss.FloatArr('cell_imm')`,
  per HPV instance) and a *Connector-derived effective* state (`sev_imm`,
  Connector-written each step). Adding the humoral counterpart
  `ss.FloatArr('nab_imm')` per HPV; `rel_sus` becomes Connector-written.
- `hpv.CrossImmunity(ss.Connector)`: stack-multiply-scatter every step. Reads
  per-genotype `nab_imm` / `cell_imm` columns, applies cross-protection matrices,
  writes per-target `rel_sus` and `sev_imm` on each HPV module. Matrices live on
  the Connector instance, defaults from a hand-translated
  `hpv.parameters.get_cross_immunity()` (see "Quarantine policy" below).
- Per-genotype natural-history params (`rel_beta`, `dur_precin`, `dur_cin`,
  `cin_fn`, `cancer_fn`, `cell_imm_init`, plus a new `imm_init` distribution
  for the humoral side) wired through `GenotypePars` for all four genotypes.
- Per-genotype initial-prevalence seeding: M02's `_INIT_HPV_PREV_F` / `_INIT_HPV_PREV_M`
  generalised to a per-genotype keyed layout. Independent seeding allowed
  (overlapping initial co-infections OK, matching v2).
- Regression coverage: 40-entry `short_summary` parity (32 per-genotype + 8
  aggregate); age-aggregated 4-genotype cancer / infection trajectory parity;
  re-run M02's network and 1-genotype-natural-history tests through the new
  Connector path.
- Local regeneration of the 4-genotype v2 baseline (M0 anchor scenario, Nigeria,
  fixed seed `0`, 1990–2060). Gitignored per M0 baseline convention.

**Explicitly out of scope (deferred):**

- Waning immunity — dropped per `MIGRATION_PLAN.md:30` ("never used in any
  published analysis").
- Age-stratified analyzer (`AgeResults`) — deferred to M04 per commit `b5d9bdf6`.
  Per-genotype age-stratification therefore deferred too; M03's release gate
  uses age-aggregated trajectories only.
- Vaccination / screening / treatment / `dynamic_pars` — M05+.
- All other analyzers (`snapshot`, `age_pyramid`, `age_causal_infection`,
  `dalys`) — M09.
- Multi-genotype partnership-network tightening beyond M02's gates. M03
  re-runs M02's partnership-equivalence test only as a drift check; further
  network tightening is its own milestone-side task.
- Multiscale dynamic agent spawning — same M02 deferral applies.
- Cross-immunity matrix calibration. M03 ports v2's defaults verbatim; whether
  those defaults still fit Nigeria 4-genotype data is an M04 calibration
  concern.

---

## Architecture

### Disease module: `hpv.HPV(ss.Infection)` (M03 changes)

M02's HPV module (`hpvsim/hpv.py:251`) keeps its trajectory-based progression
math unchanged. The immunity layout changes:

| Field | M02 | M03 | Role |
|---|---|---|---|
| `rel_sus` (inherited from `ss.Infection`) | unused (default 1.0) | **Connector-written** each step | Effective per-target susceptibility multiplier; applied at transmission |
| `nab_imm` | absent | **new** `ss.FloatArr`, default `0.0` | Raw humoral immunity to *this* genotype as the source — bumped on clearance from `pars.imm_init.rvs()` |
| `cell_imm` | absent | **new** `ss.FloatArr`, default `0.0` | Raw cell-mediated immunity to *this* genotype as the source — bumped on clearance from `pars.cell_imm_init.rvs()` |
| `sev_imm` | `ss.FloatArr`, written on clearance from `cell_imm_init.rvs()` (`hpv.py:506-509`) | **Connector-written** each step | Effective per-target severity factor consumed by `dur_precin = p.dur_precin.rvs(uids) * (1.0 - sev_imm_uids)` (`hpv.py:416`) |
| `rel_sev` | `ss.FloatArr`, biological baseline sampled once | unchanged | Biological severity baseline; orthogonal to immunity |

The progression-math read sites (`_dispatch_progression`, `step_infect`,
`step_state`) read `self.sev_imm[uids]` and `self.rel_sus[uids]` exactly as in
M02 — they don't care that the writer has moved.

**Clearance-time bookkeeping** (inside `HPV.step_state`, replacing `hpv.py:506-509`):

```python
new_cell = self.pars.cell_imm_init.rvs(cleared)
new_nab  = self.pars.imm_init.rvs(cleared)
self.cell_imm[cleared] = np.maximum(self.cell_imm[cleared], new_cell)
self.nab_imm[cleared]  = np.maximum(self.nab_imm[cleared],  new_nab)
# sev_imm is NOT written here in M03 — Connector overwrites it next step.
```

Each `HPV` instance gains a `genotype` constructor kwarg (string key into
`GenotypePars`). The Disease's `name` is set to the genotype key, so Starsim
auto-stratifies results under `sim.results.hpv16.*`, `sim.results.hpv18.*`, etc.

### Connector: `hpv.CrossImmunity(ss.Connector)`

```python
class CrossImmunity(ss.Connector):
    def __init__(self, cross_imm_sus=None, cross_imm_sev=None, **kwargs):
        super().__init__(**kwargs)
        self.cross_imm_sus = cross_imm_sus  # (n_g, n_g) np.float32; None → default at init
        self.cross_imm_sev = cross_imm_sev
        self._hpv_modules = None            # populated in init_pre
        self._genotype_index = None         # {key: column}
```

**Initialisation (`init_pre`):**

1. Walk `sim.diseases`, collect every `hpv.HPV` instance. Order them by
   registration order. Store as `self._hpv_modules`.
2. Build `self._genotype_index = {m.genotype: i for i, m in enumerate(self._hpv_modules)}`.
3. If `cross_imm_sus` / `cross_imm_sev` is `None`, populate from
   `hpv.parameters.get_cross_immunity()` using the genotype-key ordering from
   step 2.
4. Validate matrix shape `(n_g, n_g)` and that diagonal entries equal 1.0
   (own-genotype = full immunity, by convention; v2 enforces this).
5. Cast to `np.float32` (matches v2's `imm.astype('float32')`).

If the Sim has no HPV modules, the Connector warns once and exits early on
`step`.

**Per-step work (`step`):**

```python
# 1. Stack source-genotype immunity into 2D (n_agents, n_g)
nab  = np.column_stack([m.nab_imm.values  for m in self._hpv_modules])
cell = np.column_stack([m.cell_imm.values for m in self._hpv_modules])

# 2. Apply cross-protection matrices.
#    Convention: row = target genotype, col = source genotype.
#    Effective immunity to target g = sum_k cross[g, k] * source[uid, k]
sus_imm = nab  @ self.cross_imm_sus.T   # (n_agents, n_g)
sev_imm = cell @ self.cross_imm_sev.T   # (n_agents, n_g)

# 3. Clip to [0, 1] (matches v2's np.minimum(.., 1) cap)
np.clip(sus_imm, 0.0, 1.0, out=sus_imm)
np.clip(sev_imm, 0.0, 1.0, out=sev_imm)

# 4. Scatter back to each HPV module
for i, m in enumerate(self._hpv_modules):
    m.rel_sus.values[:] = 1.0 - sus_imm[:, i]
    m.sev_imm.values[:] = sev_imm[:, i]
```

The Connector is a *projection*: it owns no per-agent state, no clearance
bookkeeping, and no genotype configuration beyond the matrix and ordering.

**Sim-step ordering** (Starsim convention):

1. `Disease.step_state` — clearances; HPV writes `nab_imm` / `cell_imm`.
2. `Connector.step` — `CrossImmunity` reads stacked sources, writes per-target
   `rel_sus` / `sev_imm`.
3. `Disease.step_infect` — `rel_sus` correctly applied to new infections.
4. `Disease.step_die`.

This matches v2's `check_immunity` placement in `update_states_pre`.

**Numerical equivalence to v2:** v2 stores immunity as `(n_g, n_agents)` and
dots `(n_g, n_g) @ (n_g, n_agents)`. v3 stores per-disease 1D arrays, stacks
to `(n_agents, n_g)`, and does `(n_agents, n_g) @ (n_g, n_g).T`. Same
matrix-vector products, transposed for column-major layout. Numerically
identical given the same matrix and the same per-clearance source draws.

**Always run**, even when no clearances occurred: the cost is a 4-column ×
N-agent matrix multiply (~200k float ops at typical N), trivial compared to
transmission-step costs. No early-out.

### Single-genotype degenerate path

A 1-genotype Sim (M02's case) goes through the **same** Connector path. The
matrices reduce to `[[1.0]]`, the dot product is the identity, and `sev_imm` is
re-derived each step as the most-recent `cell_imm` value. `rel_sus` becomes
`1.0 - nab_imm`, where M02's behaviour was effectively `rel_sus = 1.0` (no
humoral path). This is **not** a numerical no-op for M02 — M02's 1-genotype
parity test runs through the Connector and may show small drift attributable
to the new humoral path; if so, the drift is captured as an M02-regression
delta.

The auto-instantiated Connector adds ~negligible per-step cost for n=1, but
preserves a single uniform code path.

### Sim wrapper: `hpv.Sim` API surface

```python
# Primary path — explicit Disease list
hpv.Sim(
    diseases=[
        hpv.HPV(genotype='hpv16', rel_beta=1.0),
        hpv.HPV(genotype='hpv18'),
        hpv.HPV(genotype='hi5'),
        hpv.HPV(genotype='ohr'),
    ],
    connectors=[hpv.CrossImmunity()],
)

# Sugar — constructs the four modules from GenotypePars defaults
hpv.Sim(genotypes=[16, 18, 'hi5', 'ohr'])
hpv.Sim(genotypes=[16, 18], genotype_pars={'hpv16': {'rel_beta': 1.2}})
```

Sugar resolution (in `Sim.__init__`):

1. Normalise each entry to a canonical key via an alias table
   (`16 → 'hpv16'`, `'hi5' → 'hi5'`, etc.; ported from v2's `genotype_aliases`).
2. For each key, instantiate
   `hpv.HPV(genotype=key, **GenotypePars[key], **genotype_pars.get(key, {}))`.
3. Auto-instantiate `hpv.CrossImmunity()` if the user didn't pass one (single
   or multi-genotype).

If both `diseases=` and `genotypes=` are passed, error early with a clear
message. M01's existing single-disease path remains valid (it now goes through
the new Connector by default).

### Parameters: `hpvsim/parameters.py` extensions

`GenotypePars` (already at `hpvsim/parameters.py:60` from M02) gains a
canonical genotype set:

```python
GENOTYPE_KEYS = ('hpv16', 'hpv18', 'hi5', 'ohr')
```

Per-genotype defaults populated from v2's
`hpvsim._v2_legacy.parameters.get_genotype_pars()` for `rel_beta`,
`dur_precin`, `dur_cin`, `cin_fn`, `cancer_fn`, `cell_imm_init`, plus a new
`imm_init` distribution (humoral counterpart, sampled into `nab_imm` at
clearance). Values are hand-translated, not delegated.

`hpv.parameters.get_cross_immunity()` is added as a hand-translated port of
v2's cross-protection-matrix builder. Inputs (`cross_imm_med`, `cross_imm_high`,
the genotype→clade table, the matrix-construction logic) reproduce in active
code with no `_v2_legacy` import.

### Initial prevalence

Per-genotype curves move out of `hpvsim/hpv.py` into `hpvsim/parameters.py`
(or a sibling `hpvsim/data/init_prev.py` if the tables are bulky), keyed by
genotype string. HPV16's existing M02 curves migrate verbatim under `'hpv16'`.
Each `HPV(ss.Infection)` reads its own genotype's curve at `init_post`. The
four HPV instances seed independently — overlapping initial co-infections
allowed, matching v2's behaviour.

User overrides flow through the same path as other per-genotype params:

```python
hpv.Sim(
    genotypes=[16, 18, 'hi5', 'ohr'],
    genotype_pars={'hpv16': {'init_prev': custom_curve}},
)
```

At `t=0`, the seeding writes `infected[uid] = True` for genotype X but does
**not** bump `nab_imm` or `cell_imm` (no prior clearance). All immunity arrays
start at 0.0; the Connector's first `step` produces `rel_sus = 1.0` and
`sev_imm = 0.0` everywhere — matching v2's first-step state.

---

## Quarantine policy

Per the standing rule (memory: "Quarantine dependencies — copy v2 logic into
active code rather than re-export from `_v2_legacy/`") and per
`MIGRATION_PLAN.md:264`:

- M03 introduces **zero** new subclass-first delegations to `_v2_legacy/`.
- `get_cross_immunity()` and the per-genotype defaults are hand-translated
  into `hpvsim/parameters.py`. The `_v2_legacy/immunity.py` and
  `_v2_legacy/parameters.py` files remain reference-only.
- Any temptation to import from `_v2_legacy/` during implementation is a flag
  to slow down and finish the port instead.

---

## Results & regression strategy

### Per-genotype results (free from N×Disease)

Each `HPV` instance's `name` is its genotype key, so Starsim auto-stratifies:

```
sim.results.hpv16.cum_infections, .new_infections, .n_susceptible,
                  .new_cancers, .cum_cancers, .new_cancer_deaths, ...
sim.results.hpv18.*  (same)
sim.results.hi5.*
sim.results.ohr.*
```

No per-genotype results plumbing required — falls out of M02's existing
`HPV.define_results()`.

### Sim-level aggregates

Three aggregators added to `hpv.Sim` (or a small helper module):

- `sim.results.hpv.cum_infections_any` — agents ever infected with any
  genotype (boolean OR across module `infected` history; avoids double-counting
  multi-genotype co-infections).
- `sim.results.hpv.cum_cancers_any` — total cancers across genotypes (sum;
  cancer is genotype-attributed, so summing is correct).
- `sim.results.hpv.new_cancer_deaths_any` — same shape, sum.

These are the only Sim-level aggregates added in M03.

### `short_summary` shape (40 entries)

M02's 8-metric `short_summary` (per `MIGRATION_PLAN.md:260` — total HPV
infections, total cancers, total cancer deaths, mean HPV prevalence, mean
cancer incidence, mean ages of infection / cancer / cancer death, plus total
population) becomes:

- **Per-genotype:** 8 metrics × 4 genotypes = 32 entries, keyed
  `hpv16.total_infections`, `hpv18.total_cancers`, etc.
- **Aggregate:** the 8 metrics computed against the `*_any` aggregators
  (8 entries, keyed `any.*`).

Total: 40 summary entries.

### Regression gates

Two gates, mirroring the M02 dual-gate pattern (`MIGRATION_PLAN.md:259-262`):

| Gate | Comparison | Tolerance | Failure action |
|---|---|---|---|
| **Per-genotype `short_summary` parity** (development gate) | v3 4-genotype run vs. v2 4-genotype baseline, all 40 entries | ±10% relative drift | Informational; classify drift or fix per-PR |
| **Age-aggregated trajectory parity** (capability gate) | `cum_cancers_any(t)` and `cum_infections_any(t)` time series, M0 4-genotype anchor scenario | Visual + L2/max-relative-drift threshold (TBD on first run) | Required green for the M03 PR |

The age-aggregated gate is pulled directly from `sim.results.*_any` without
any analyzer. Threshold pre-commitment is deferred to first-run inspection;
once committed, it's pinned in `tests/test_m03_trajectory_parity.py`.

### Test files

- `tests/test_m03_short_summary_parity.py` — per-genotype 32-metric +
  aggregate 8-metric drift table.
- `tests/test_m03_trajectory_parity.py` — time-series cancer / infection
  trajectory comparison against the M0 anchor.
- `tests/test_m03_cross_immunity.py` — unit tests on `CrossImmunity` itself:
  matrix multiplication math against hand-computed expected values; single-
  genotype degenerate identity case; monotonicity (more clearances ⇒ more
  `sus_imm`); shape and diagonal validation.
- M02's `test_natural_history_parity` and the partnership-equivalence test
  re-run against the new Connector path (no new code; parametrise the
  fixture). M02's 1-genotype baseline stays gitignored-but-locally-reproducible
  and acts as the M03 M02-regression sanity gate.

### Baselines

- Regenerate the 4-genotype v2 baseline locally via
  `tests/regression/baseline.py` (M0 anchor scenario: Nigeria, fixed seed `0`,
  1990–2060). Stays gitignored per M0 baseline convention
  (`MIGRATION_PLAN.md:79`).
- M02's 1-genotype baseline stays in place — drives the M02-regression
  sanity test (single-genotype Sim through the new Connector path must still
  match within tolerance).

### Anticipated drift axes (informational)

These are *expected* to show non-zero drift in the first 4-genotype regression
run. Each becomes a tracking issue if it lands outside the ±10% gate, with a
reconvergence plan; none is a blocker if classified as feature-misalignment:

- Per-genotype CIN / cancer counts. Cross-immunity reshapes the
  susceptibility distribution over time, so genotypes that match in aggregate
  may drift individually.
- Mean age of infection per non-HPV16 genotype. Untested in M02; may surface
  RNG-stream or seeding sensitivity that needs flushing out.
- Single-genotype path (M02 baseline) under the Connector. The new humoral
  `nab_imm` → `rel_sus` plumbing was effectively absent in M02; small drift
  on M02's 1-genotype baseline is plausible.

---

## Branch hygiene & sequencing

### Branching

- M03 lives on `m03-multi-genotype-and-cross-immunity`, branched off
  `m02-natural-history-parity` (deviation from `MIGRATION_PLAN.md:271-273`,
  which prescribes "off `v3.0-dev` after the predecessor PR merges").
  Reason: M03 substantively depends on M02 work (`sev_imm` refactor,
  `GenotypePars`, AgeMigration); branching off `v3.0-dev` first would require
  re-implementing M02's foundation.
- **M02-PR drift propagation:** If the M02 PR review surfaces material
  changes, those propagate into M03 by either rebasing M03 onto the updated
  M02 branch, or merging M02 → M03 once the M02 PR lands. The first M03
  commit notes this in its message.
- The M03 PR opens against `v3.0-dev`. Once M02 merges into `v3.0-dev`, M03
  rebases onto the new tip; the PR shows only M03's commits at review time.

### Continuous-runnability invariant

Per `MIGRATION_PLAN.md:258`, every commit on the milestone branch must
satisfy `hpv.Sim().run()`. The structural change at risk is the `sev_imm`
refactor — splitting the M02 single-write into Connector-driven derivation.
Each commit along the way must keep `hpv.Sim().run()` green and M02's
1-genotype parity test passing.

### Suggested commit sequencing (non-binding)

This staging is illustrative and not a contractual structure. The actual
sequence is determined during execution; the only hard requirement is the
runnability invariant above. As a starting reference:

1. Add `nab_imm` and `cell_imm` to `HPV` (default 0.0; not yet read). M02
   `sev_imm` clearance write stays in place. No behaviour change.
2. Add `CrossImmunity` Connector class and the auto-instantiation sugar.
   Single-genotype path opt-in. 1×1 identity multiply ⇒ no behaviour change.
3. Move `sev_imm` write from `HPV.step_state` into the Connector. Connector
   now writes both `rel_sus` and `sev_imm`. Single-genotype identity case
   remains numerically equivalent (modulo the new humoral `rel_sus` path).
4. Wire 4-genotype defaults: extend `GenotypePars`, port
   `get_cross_immunity()`, generalise the initial-prevalence layout.
5. Regression coverage and 4-genotype baseline regeneration; trajectory-gate
   threshold pinned.

### Open / tracked at M03 close (anticipated)

- Per-genotype `short_summary` reconvergence on any genotype that drifts
  >10% — investigated post-M03, fixed as a follow-up commit on `v3.0-dev`
  (not blocking M04).
- Age-stratified per-genotype results — explicitly deferred to M04 with
  `AgeResults`.
- Cross-immunity matrix calibration — M03 ports v2's defaults; whether
  those defaults still fit Nigeria 4-genotype data is an M04 calibration
  concern.

### Out of scope for the M03 PR

Modifying `MIGRATION_PLAN.md` itself, unless M03 surfaces a real plan delta
analogous to M02's "AgeResults deferred" amendment in commit `b5d9bdf6`.
Plan deltas land in their own commit with rationale, not folded into
implementation commits.