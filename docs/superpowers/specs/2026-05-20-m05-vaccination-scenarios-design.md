# M05: Vaccination Scenarios — Design

**Date:** 2026-05-20
**Milestone:** M05 (Vaccination scenarios)
**Branch:** `m05-vaccination-scenarios` (off `m04-calibration-loop`; M05 PR targets `v3.0-dev` once M04 merges)
**Predecessor:** [M04 Calibration Loop](2026-05-18-m04-calibration-loop-design.md)
**Status:** Implemented. See "Post-implementation deltas" below for divergences from this spec discovered during the build.

---

## Goal

Add prophylactic HPV vaccination to v3 by composing Starsim's native
intervention/product framework with one HPV-specific product class and a thin
HPV-specific subclass of `ss.BaseVaccination` that re-exposes v2's targeting
kwargs. A `hpv.Sim` configured with `hpv.routine_vx` or
`hpv.campaign_vx` and a `hpv.vx` product must reproduce v2.x's vaccination-
impact trajectories — cancer incidence and HPV prevalence post-vaccination —
within M03's multi-seed z-score parity gate (`|z| < 3`) on two anchor
scenarios that mirror the headline shapes of `hpvsim_pxv_younger` (routine)
and `hpvsim_1dose` (campaign).

The work that lands in M05 is intentionally small. Starsim already ships
`ss.Product`, `ss.Vx`, `ss.Intervention`, `ss.BaseVaccination`,
`ss.RoutineDelivery`, `ss.CampaignDelivery`, `ss.routine_vx`, `ss.campaign_vx`,
and the per-intervention agent-level state (`vaccinated`, `n_doses`,
`ti_vaccinated`). M05's only new code is (a) an HPV-specific multi-genotype
vaccine product, and (b) a thin subclass of `ss.BaseVaccination` that
re-introduces v2's `age_range`/`sex` targeting kwargs on top of Starsim's
single-callable eligibility hook.

## Scope

**In scope:**

- `hpv.vx(ss.Vx)` product class with per-genotype `rel_imm` table; loaded
  by name from `hpvsim/data/products_vx.csv` (defaults: `bivalent`,
  `quadrivalent`, `nonavalent`) or overridden with an explicit `rel_imm`
  dict. `administer()` applies the all-or-nothing-plus-leaky vaccine model
  per genotype and writes the resulting peak into each active HPV module's
  `nab_imm`. The existing `CrossImmunity(ss.Connector)` (M03) reads
  `nab_imm` each step and produces per-target `rel_sus` — no new connector
  wiring required.
  *(Post-implementation: vaccine immunity now writes to a new per-module
  `vax_imm` state, not `nab_imm`, to prevent the cross-immunity matrix from
  bleeding bivalent protection onto hi5/ohr. See "Post-implementation
  deltas" for the rationale and matching v2 semantics.)*
- `hpv.BaseVaccination(ss.BaseVaccination)` subclass that accepts v2's
  `age_range`, `sex`, and `eligibility` arguments and composes them into a
  single Starsim eligibility callable. Stores the originals for
  introspection (M04 `AgeResults` consumption).
- `hpv.routine_vx(hpv.BaseVaccination, ss.RoutineDelivery)` and
  `hpv.campaign_vx(hpv.BaseVaccination, ss.CampaignDelivery)` empty-body
  leaf classes — they exist so `isinstance(intv, hpv.routine_vx)` works
  and so the `hpv` namespace owns the public types.
- `_compose_eligibility(age_range, sex, extra)` and `_coerce_sex(sex)`
  private helpers in `hpvsim/interventions.py`; the latter implements v2's
  `'f'`/`'m'`/`0`/`1`/list coercion rules.
- `products_vx.csv` moved from `hpvsim/_v2_legacy/data/` into
  `hpvsim/data/` verbatim; loader function `_load_vx_products()` caches it
  at module load.
- Override of `ss.Intervention._parse_product_str` on `hpv.BaseVaccination`
  so `hpv.routine_vx(product='bivalent', ...)` resolves through
  `hpv.vx(name='bivalent')`, matching v2's string-product convention.
- Two regression anchor scenarios (`anchor_vx_routine`, `anchor_vx_campaign`)
  on the M03 Nigeria 4-genotype baseline. v2 baseline regenerated locally
  (gitignored), parity gated at `|z| < 3` on the M03 short-summary metrics
  plus three vaccination-specific summary metrics
  (`n_vaccinated_2060`, `n_doses_2060`, `cancer_incidence_2030_2060`).
- Trajectory-parity gate on the routine anchor only (campaign anchor gets
  the short-summary gate only; trajectory test is the most expensive in
  the suite and the campaign signal is well-summarised by headline metrics).
- Unit tests for `_compose_eligibility`, `_coerce_sex`, CSV lookup, the
  `administer` state-bump, and inactive-genotype tolerance.
- Integration smoke tests verifying routine/campaign interventions fire and
  update per-intervention state; a `test_no_vx_baseline_unchanged` guard
  that asserts a sim without any vx intervention reproduces M03's anchor
  numbers exactly (CRN-stream perturbation guard).

**Explicitly out of scope (deferred):**

- Therapeutic vaccination (`BaseTxVx`, `routine_txvx`, `campaign_txvx`,
  `linked_txvx`) — moved to M06. `linked_txvx` is structurally part of the
  screen-and-treat cascade, and `BaseTxVx` shares its design with the M06
  treatment base classes. The MIGRATION_PLAN sub-task list is updated
  alongside this spec.
- Diagnostic and treatment product classes (`dx`, `tx`) — moved to M06,
  landing with their consumers (screening, `treat_num`/`treat_delay`).
- Waning immunity — plan-wide drop ([`MIGRATION_PLAN.md:30`]).
- `peak_imm` separate-array model — collapses with `nab_imm` since no
  decay; vaccine writes directly to `nab_imm`.
- `imm_boost` multi-dose efficacy escalation. M05's second-dose semantics
  are idempotent (max-of-existing returns the same value); proper booster
  modeling lands post-M05 if needed.
- Sex-specific vaccine efficacy. Uniform across sexes; v2 was too.
- Vaccine-specific analyzers (dose distribution by age cohort, breakthrough
  infection plots) — deferred to M09.
- Migration-guide entries for the vaccination API — deferred to M10.
- Upstream Starsim PR adding `age_range`/`sex` natively to
  `ss.BaseVaccination`. Tracking only; not blocking M05.

---

## Architecture

### Module layout

```
hpvsim/
├── products.py             # NEW — hpv.vx(ss.Vx)                                 ~80 LOC
├── interventions.py        # NEW — hpv.BaseVaccination, routine_vx, campaign_vx
│                           #       + _compose_eligibility(...)                    ~70 LOC
├── data/
│   └── products_vx.csv     # MOVED from _v2_legacy/data/ (verbatim, 24 rows)
hpvsim/__init__.py          # exports: vx, routine_vx, campaign_vx
```

No `Vaccination(ss.Module)` for shared state. No schedule POJOs. No flat
rewrite of `routine_vx` / `campaign_vx`. Per-intervention state lives on
`ss.BaseVaccination` (Starsim's idiomatic location); cross-campaign queries
iterate `sim.interventions`.

### Inheritance graph

```
ss.Module ── ss.Product ── ss.Vx ─────────────────────── hpv.vx

ss.Module ── ss.Intervention ── ss.BaseVaccination ── hpv.BaseVaccination
                                                     │
                              ── ss.RoutineDelivery ─┼─ hpv.routine_vx
                                                     │
                              ── ss.CampaignDelivery ┴─ hpv.campaign_vx
```

`hpv.routine_vx(hpv.BaseVaccination, ss.RoutineDelivery)` and
`hpv.campaign_vx(hpv.BaseVaccination, ss.CampaignDelivery)` mirror Starsim's
`ss.routine_vx(ss.BaseVaccination, ss.RoutineDelivery)` exactly — only the
vaccination-side base is swapped. The MRO chains kwargs cleanly:
`hpv.routine_vx → hpv.BaseVaccination → ss.BaseVaccination → ss.RoutineDelivery → ss.Intervention → ss.Module`.

### Why we are not writing the diamond ourselves

A flat (single-inheritance) intervention design was the initial
recommendation during brainstorming, on the grounds that the diamond would
need M10 strip-out work. Reading Starsim's `interventions.py` revealed that
the diamond is Starsim's committed-to design — `ss.routine_vx` and
`ss.campaign_vx` literally subclass `(BaseVaccination, RoutineDelivery)` and
`(BaseVaccination, CampaignDelivery)`. Using Starsim's classes means we
inherit Starsim's diamond, which is Starsim's responsibility, not ours.
M10 strip-out under migration convention 3 does not apply to upstream
Starsim base classes.

### The `hpv.vx` product class

```python
class vx(ss.Vx):
    """HPV multi-genotype prophylactic vaccine."""
    def __init__(self, name=None, rel_imm=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            name=name,                # 'bivalent' / 'quadrivalent' / 'nonavalent'
            rel_imm=rel_imm,          # dict {genotype: float in [0,1]}
        )
        self._rel_imm = _resolve_vx_pars(self.pars.name, self.pars.rel_imm)
        self._sterilizing_dist = ss.bernoulli(p=0.0)  # re-pointed per genotype

    def administer(self, people, uids):
        if len(uids) == 0:
            return
        for genotype, rel_imm_g in self._rel_imm.items():
            hpv_mod = self._find_genotype_module(genotype)
            if hpv_mod is None:
                continue
            self._sterilizing_dist.set(p=rel_imm_g)
            sterilizing_uids = self._sterilizing_dist.filter(uids)
            peak = np.full(len(uids), rel_imm_g, dtype=float)
            peak[np.isin(uids, sterilizing_uids)] = 1.0
            hpv_mod.nab_imm[uids] = np.maximum(hpv_mod.nab_imm[uids], peak)
```

| Behavior | Mechanism |
|---|---|
| Per-genotype independence (sterilizing-vs-leaky drawn separately per genotype) | Loop over `_rel_imm.items()`, fresh `ss.bernoulli` filter each iteration |
| Cross-immunity propagation | `CrossImmunity(ss.Connector)` reads `nab_imm` next step and updates `rel_sus` |
| Natural-immunity preserving | `np.maximum(existing, peak)` — vaccine never downgrades |
| Inactive-genotype tolerance | Silent skip; 9-valent product in 4-genotype sim only bumps the 4 active genotypes |
| CRN reproducibility | `ss.bernoulli` instance constructed at `__init__`, `.set(p=...)` per genotype, `.filter(uids)` for the draw — matches M03's CRN pattern |
| Second-dose idempotence | Max-of-existing returns the same value; no booster escalation in M05 |

`_resolve_vx_pars(name, override)`:
- Exactly one of `name` / `override` must be provided; both or neither raises `ValueError`.
- If `override`, return verbatim.
- If `name`, look up rows in cached CSV, return `{genotype: rel_imm}` dict.
- Unknown name raises `ValueError` listing valid names.

`_find_genotype_module(genotype)` walks `self.sim.diseases` matching by the
HPV module's genotype attribute. The exact key convention is whatever M03's
`CrossImmunity` connector uses (see `hpvsim/cross_genotype.py`); the
implementer matches that convention rather than introducing a new one.

### The `hpv.BaseVaccination` subclass

```python
class BaseVaccination(ss.BaseVaccination):
    def __init__(self, *args, age_range=None, sex=None, eligibility=None, **kwargs):
        composed = _compose_eligibility(age_range, sex, eligibility)
        super().__init__(*args, eligibility=composed, **kwargs)
        self.age_range = age_range
        self.sex = _coerce_sex(sex)

    def _parse_product_str(self, product):
        return vx(name=product)


class routine_vx(BaseVaccination, ss.RoutineDelivery):
    pass


class campaign_vx(BaseVaccination, ss.CampaignDelivery):
    pass
```

Three responsibilities:
1. Accept v2's `age_range` / `sex` / `eligibility` API and compose into a
   single Starsim eligibility callable.
2. Override `_parse_product_str` so `routine_vx(product='bivalent', ...)`
   resolves via `vx(name='bivalent')`.
3. Provide HPV-namespaced class identity for `isinstance` checks and
   imports.

### Sex coercion (`_coerce_sex`)

Matches v2 conventions, 0=F, 1=M.

| Input | Output |
|---|---|
| `None` | `None` (no sex filter) |
| `'f'` | `{0}` |
| `'m'` | `{1}` |
| `0` or `1` (int) | `{int(sex)}` |
| `['f', 'm']` or `[0, 1]` (list) | `{0, 1}` |
| Anything else | `ValueError` listing valid forms |

### Eligibility composition (`_compose_eligibility`)

```python
def _compose_eligibility(age_range, sex, extra):
    sex_set = _coerce_sex(sex)
    def elig(sim):
        cond = sim.people.alive
        if age_range is not None:
            lo, hi = age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age < hi)
        if sex_set is not None and len(sex_set) == 1:
            (s,) = sex_set
            cond = cond & (sim.people.sex == s)
        if extra is not None:
            cond = cond & _as_boolarr(extra(sim), sim.people)
        return cond.uids
    return elig
```

Re-evaluated each time the intervention fires (Starsim calls
`self.check_eligibility()` at every applicable step), so agents who age into
`age_range` become eligible naturally without explicit refresh.

### Vaccine efficacy model (all-or-nothing + leaky)

Matches v2's `vx.administer` (`hpvsim/_v2_legacy/interventions.py:1455`).
For each vaccinated agent and each genotype `g`:

```
draw      ~ Bernoulli(rel_imm[g])
peak[g]   = 1.0           if draw == 1   (sterilizing immunity)
            rel_imm[g]    if draw == 0   (leaky protection at the floor)
nab_imm[g][agent] = max(existing, peak[g])
```

Per-genotype independence: an agent can be sterilizing-immune to HPV16
(`peak=1.0`) and only leaky-immune to hi5 (`peak=0.5`) from the same dose.
This matches v2's semantics exactly.

The `nab_imm` write flows through the existing `CrossImmunity` connector
the next step, producing per-target `rel_sus` reductions automatically.
M05 does not touch `rel_sus` directly.

### Constructor surface (v2-compatible)

```python
hpv.routine_vx(
    product=hpv.vx(name='bivalent'),
    prob=0.9,
    age_range=[9, 14],
    sex='f',
    start_year=2020,
    name='routine_girls',
)

hpv.campaign_vx(
    product=hpv.vx(name='bivalent'),
    prob=[0.7, 0.5],
    age_range=[9, 30],
    sex='f',
    years=[2020, 2021],
    interpolate=False,
    name='catchup_2020_2021',
)
```

String-product convention also supported:
```python
hpv.routine_vx(product='bivalent', prob=0.9, age_range=[9, 14], ...)
```

---

## Data: `products_vx.csv`

Columns: `name, genotype, rel_imm`. 24 rows covering three products
(`bivalent`, `quadrivalent`, `nonavalent`) across eight genotype keys
(`hpv16, hpv18, hi5, hpv45, hi4, ohr, hr, lr`). Ported verbatim from v2;
no value changes.

v3 sims that run with the standard 4-genotype set (`hpv16, hpv18, hi5, ohr`)
consume only the four matching rows for any given product. The remaining
four rows are silently unused; `hpv.vx._find_genotype_module` returns `None`
for genotypes not present in the sim.

---

## Tests

### Unit tests (`tests/test_m05_vx_unit.py`)

| Test | Verifies |
|---|---|
| `test_compose_eligibility_age` | `age_range` filter produces correct uids on a synthetic sim |
| `test_compose_eligibility_sex_female` | `sex='f'` resolves to sex==0 filter |
| `test_compose_eligibility_sex_male` | `sex='m'` resolves to sex==1 filter |
| `test_compose_eligibility_sex_both` | `sex=['f','m']` produces no sex filter |
| `test_compose_eligibility_extra_callback` | user `eligibility=fn` composes via intersection |
| `test_coerce_sex_invalid_raises` | bad sex values raise `ValueError` |
| `test_vx_csv_lookup_bivalent` | `hpv.vx(name='bivalent')._rel_imm` matches CSV row-for-row |
| `test_vx_unknown_name_raises` | `hpv.vx(name='nope')` raises with valid-names list |
| `test_vx_override_rel_imm` | `hpv.vx(rel_imm={'hpv16': 0.5})` honored over CSV |
| `test_vx_administer_bumps_nab_imm` | small sim, administer to known uids, assert `nab_imm` bumped to expected level under all-or-nothing+leaky |
| `test_vx_administer_inactive_genotype_skipped` | 9-valent product in 4-genotype sim doesn't error |
| `test_vx_administer_no_downgrade` | pre-bumped `nab_imm` (e.g. 0.95 from clearance) not reduced by leaky vaccine (e.g. 0.5) |
| `test_parse_product_str_resolves` | `routine_vx(product='bivalent', ...)` resolves to `hpv.vx(name='bivalent')` |

### Integration smoke tests (`tests/test_m05_vx_integration.py`)

| Test | Verifies |
|---|---|
| `test_routine_vx_smoke` | small sim with `routine_vx` runs; `intervention.vaccinated` flips for targeted agents; `n_doses` increments |
| `test_campaign_vx_smoke` | same for `campaign_vx` |
| `test_no_vx_baseline_unchanged` | sim without any vx intervention produces identical results to M03 anchor (CRN-perturbation guard) |
| `test_routine_vx_age_targeting` | only agents in `age_range` get vaccinated |
| `test_routine_vx_sex_targeting` | only agents matching `sex` get vaccinated |
| `test_vaccine_reduces_susceptibility` | post-vaccination `rel_sus` on each HPV module is reduced for vaccinated agents on the next step (verifies CrossImmunity propagation) |

### Parity tests (slow, on-demand)

```
tests/regression/anchor_vx_routine.py             # NEW — PARS for routine anchor
tests/regression/anchor_vx_campaign.py            # NEW — PARS for campaign anchor
tests/regression/multi_seed_v2_vx.py              # NEW — generates 30-seed v2 baselines
tests/regression/v2_seeds_n30_vx_routine.json     # local-only, gitignored
tests/regression/v2_seeds_n30_vx_campaign.json    # local-only, gitignored
tests/test_m05_vx_routine_parity.py               # short-summary z gate, routine
tests/test_m05_vx_campaign_parity.py              # short-summary z gate, campaign
tests/test_m05_vx_trajectory_parity.py            # trajectory z gate, routine only
```

Each parity test follows M03's pattern (`tests/test_m03_short_summary_parity.py`):
10 v3 seeds × full sim, per-metric z-score against the 30-seed v2 baseline,
gate at `|z| < 3.0`. All marked `@pytest.mark.slow`; CI runs `pytest -m
'not slow'` (matches the `.github/workflows/tests.yaml` M04 commit).

The metric set extends M03's 40-entry short summary with three vaccination-
specific scalars:
- `n_vaccinated_2060` (cumulative first-dose agents at sim end)
- `n_doses_2060` (cumulative doses administered)
- `cancer_incidence_2030_2060` (post-vaccination cancer incidence — the
  actual impact signal)

### Anchor scenarios

**Routine** (`anchor_vx_routine.py`): vanilla M03 Nigeria 4-genotype sim with
one added intervention — bivalent vaccine, girls aged 9–10, 90% annual
coverage, starting 2020. Mirrors `hpvsim_pxv_younger` headline shape.

**Campaign** (`anchor_vx_campaign.py`): vanilla M03 Nigeria 4-genotype sim
with one added intervention — bivalent vaccine, girls aged 9–14, 70% coverage
in each of 2020 and 2021, no inter-year interpolation. Mirrors `hpvsim_1dose`
headline shape.

Both anchors are isolated from the rest of the analysis-repo suite to keep
the parity signal attributable to M05 changes only.

### Pre-PR gate

1. `pytest -m 'not slow'` green on the M5 branch.
2. Both parity tests green locally (`pytest -m slow tests/test_m05_vx_*_parity.py`).
3. Trajectory parity test green locally.
4. M03's tests still green (regression guard).
5. M04's tests still green.
6. Anchor regeneration documented in `tests/regression/README_m05.md`.

---

## Implementation sub-task sequence

A dependency-respecting order for the writing-plans output:

1. Move `products_vx.csv` from `hpvsim/_v2_legacy/data/` to
   `hpvsim/data/`; add `_load_vx_products()` cache helper.
2. Implement `hpv.vx` product class in `hpvsim/products.py`.
3. Unit tests for `hpv.vx` (CSV lookup, override, administer state-bump,
   inactive-genotype skip, no-downgrade).
4. Implement `_coerce_sex` and `_compose_eligibility` helpers.
5. Unit tests for the helpers.
6. Implement `hpv.BaseVaccination`, `hpv.routine_vx`, `hpv.campaign_vx` in
   `hpvsim/interventions.py`.
7. Wire exports in `hpvsim/__init__.py`.
8. Integration smoke tests.
9. Anchor scenario scripts.
10. v2 baseline generation script + regenerate locally from a v2 env.
11. Parity tests (short-summary + trajectory).
12. CI workflow check (confirm `-m 'not slow'` still bounds CI).
13. MIGRATION_PLAN.md edits (move txvx/dx/tx to M06; rewrite M05 sub-tasks).
14. Open the M05 PR against `v3.0-dev` (after M04 merges).

---

## Open risks and mitigations

| Risk | Mitigation |
|---|---|
| **CRN stream perturbation.** New per-vaccinee Bernoulli draws (`_sterilizing_dist`) may share an RNG stream with HPV transmission decisions, drifting the no-vx baseline vs M03 | `test_no_vx_baseline_unchanged` integration test pins this. If it fails, the sterilizing distribution needs its own dedicated `ss.Dist` parent per M03's seeder pattern |
| **`nab_imm` overwrite race.** Vaccine and natural clearance both write to `nab_imm[uids]` on the same step. Order matters | Document and assert via integration test: v3's loop runs interventions after `step_state`, so vaccine reads cleared `nab_imm` and applies `max(existing, peak)`. Test the order explicitly |
| **Genotype key mapping.** `_find_genotype_module` must match the convention M03 settled on (`'hpv16'` vs `'hpv_hpv16'`) | Implementer reads `cross_genotype.py` and uses the same key resolution as `CrossImmunity` |
| **`_parse_product_str` override.** Users may pass `product='bivalent'` (string), which Starsim's base raises `NotImplementedError` for | Override on `hpv.BaseVaccination`; one-line, covered by `test_parse_product_str_resolves` |
| **Trajectory test compute cost.** M03's trajectory test is the most expensive in the suite; adding more vx trajectory tests bloats the slow-test budget | Run trajectory gate on routine anchor only; campaign gets short-summary gate only. Reconsider after first run shows actual budget |
| **AgeResults integration.** AgeResults (M04) was designed for sim-level / module-level results; reading per-intervention state may need a new collector hook | For M05, expose simple sim-level `n_vaccinated_<year>` / `n_doses_<year>` results that the intervention writes to in `step()` (Starsim already does this for the per-intervention counts). Deep AgeResults integration is its own follow-on issue if needed |
| **Multi-intervention name collisions.** Starsim collides same-type interventions without unique `name=`. M05's anchor scenarios use a single intervention so no collision, but a routine + catch-up combo would need distinct names | Documented in `tests/regression/README_m05.md`; not enforced in code (Starsim's existing warning is sufficient) |

---

## Quarantine and active-code policy

Per the user's feedback memory (copy v2 logic into active code; never import
from `_v2_legacy/`), this milestone copies the CSV verbatim into active
code (`hpvsim/data/products_vx.csv`) rather than importing or referencing
the quarantined copy. The quarantined `hpvsim/_v2_legacy/data/products_vx.csv`
remains for reference but is no longer the live source.

M05 does not introduce any new runtime imports from `hpvsim/_v2_legacy/`.

---

## MIGRATION_PLAN.md edits (committed separately, alongside the M5 sub-task work)

**M5 § Status table** — flips to `🟡 In progress; branch
m05-vaccination-scenarios`.

**M5 § Sub-tasks** — rewritten to reflect Starsim-native architecture and
the txvx/dx/tx deferral:

```
### M5: Vaccination scenarios

**Demo:** Run routine + campaign prophylactic vaccination and show cancer
incidence reduction.

**Acceptance test:** Vaccination impact trajectory overlaps v2.x intervals
on `hpvsim_1dose` / `hpvsim_pxv_younger` headline-shape scenarios.

**Sub-tasks:**
- Port `hpv.vx(ss.Vx)` product class — per-genotype `rel_imm` table loaded
  from `hpvsim/data/products_vx.csv`; `administer` applies per-genotype
  all-or-nothing+leaky model and bumps each HPV module's `nab_imm`. Cross-
  immunity propagation is automatic via the existing `CrossImmunity`
  connector.
- Add `hpv.BaseVaccination(ss.BaseVaccination)` subclass adding v2-compatible
  `age_range` / `sex` / `eligibility` constructor args (composed into a single
  Starsim eligibility callable); thin `hpv.routine_vx` and `hpv.campaign_vx`
  leaf classes combining it with Starsim's `RoutineDelivery` / `CampaignDelivery`.
- Move `products_vx.csv` from `hpvsim/_v2_legacy/data/` into active
  `hpvsim/data/`. Default product names: `bivalent`, `quadrivalent`,
  `nonavalent`.
- Add two regression anchors (`anchor_vx_routine`, `anchor_vx_campaign`),
  generator script for v2 baselines, and multi-seed z-score parity gates
  at `|z| < 3` (M03 pattern). Includes a trajectory-parity test on the
  routine anchor.
- Add unit tests for `_compose_eligibility`, `_coerce_sex`, and `hpv.vx`
  product semantics.
- Confirm intervention-level result tracking (`vaccinated`, `n_doses`,
  `ti_vaccinated`) is exposed via the existing `ss.BaseVaccination` state;
  age-stratified consumption uses M04's `AgeResults` analyzer.
```

**M6 § Sub-tasks** — appended:

```
- Port `dx(ss.Product)` diagnostic product class (CSV table maps disease
  state -> result probability). Used by screening interventions.
- Port `tx(ss.Product)` treatment product class. Used by `treat_num` /
  `treat_delay`.
- Port `txvx` therapeutic vaccination: `BaseTxVx` + `routine_txvx` +
  `campaign_txvx` + `linked_txvx`. Moved from M5 because `linked_txvx` is
  structurally part of the screen-and-treat cascade and `BaseTxVx` shares
  its design with the M06 treatment base classes (see M05 spec
  "Scope adjustments" rationale).
- Move `products_tx.csv` and `products_dx.csv` from
  `hpvsim/_v2_legacy/data/` into active `hpvsim/data/`.
```

---

## Post-implementation deltas

To be filled in after implementation lands, documenting any divergences
from this spec discovered during the build. Format follows M03's
post-implementation-deltas section.

- **`hpv.campaign_vx` `init_pre` override added and then removed.** During
  initial implementation, Starsim 3.3.x's `CampaignDelivery.init_pre`
  raised `TypeError` against a DateArray timevec, so `campaign_vx`
  carried a ~30-line `init_pre` override that replicated the timepoint
  interpolation against `sim.timevec.years` (float). The upstream issue
  was fixed in starsim 3.3.4, and commit `f4f39059` removed the override.
  `hpv.campaign_vx` is now the thin diamond leaf the spec originally
  described.

- **`test_no_vx_baseline_unchanged` was rewritten post-implementation.**
  The originally-shipped version compared two no-vx runs against each
  other, which is pure determinism and doesn't guard against M05
  perturbing M03's RNG streams. Post-review fix pins a pre-M05 scalar
  for `hpvtotal['cum_infections'].sum()` and asserts the no-vx run still
  produces it.

- **`hpv.BaseVaccination` stores raw `sex` and raw `eligibility` as
  introspection attributes.** Added `self.sex_raw` (preserves
  user-passed string/list form) and `self.eligibility_raw` (preserves
  the user's callable) alongside the existing `self.age_range` and
  `self.sex` (coerced int-set). M04 `AgeResults` consumers and the
  migration guide can rely on the raw forms.

- **Vaccine-conferred immunity moved from `nab_imm` to a new `vax_imm`
  state.** The original M05 spec landed on "reuse `nab_imm` for vaccine
  immunity" on the belief that v2 also shared one immunity track between
  clearance and vaccination. M05 parity work surfaced that v2 actually
  routes vaccine immunity through a vaccine-specific `imm_source` row of
  `peak_imm` which is NOT included in the cross-protection matrix, so v2
  vaccine immunity does not flow through cross-immunity. v3's
  `CrossImmunity` connector was double-counting cross-protection (the CSV
  per-genotype `rel_imm` AND matrix-amplification from
  `nab_imm[other_genotype]=1.0`), causing an hpv16-only vaccine to reduce
  hi5/hpv18/ohr infections by 20–30%. Post-fix: each HPV module has a
  `vax_imm` FloatArr; vaccine `administer` writes there; `CrossImmunity`
  combines per-target `vax_imm` with the matrix-derived `nab_imm`
  contribution via `rel_sus = (1 - sus_imm_from_nab) * (1 - vax_imm)`
  (independent protection paths). The CSV's per-genotype `rel_imm` table
  is now the complete vaccine cross-protection profile, matching v2
  semantics. Adds regression test
  `test_single_genotype_vaccine_does_not_bleed_to_others` in
  `tests/test_m05_vx_no_cross_bleed.py`.

- **`rel_imm[g]` semantics corrected: cross-protection coefficient, not Bernoulli probability.** v2 has TWO distinct parameters: `imm_init=0.95` (per-agent sterilizing probability, uniform across genotypes — hardcoded in v2's `default_vx`) and `rel_imm[g]` (per-genotype cross-protection coefficient, encoded in v2's cross-immunity matrix as `M[g, vx_source]=rel_imm[g]`). The original M05 implementation conflated these by using `rel_imm[g]` as BOTH the Bernoulli p and the leaky floor, producing `vax_imm[g] = rel_imm[g] * (2 - rel_imm[g])` — overprotecting every non-1.0 genotype (e.g. hi5 50% → 75%, ohr 10% → 19%). Post-fix: `hpv.vx.__init__` accepts `sterilizing_p=0.95` (matches v2's imm_init). `administer` does a single per-agent sterilizing draw at `sterilizing_p`, then writes `vax_imm[g] = rel_imm[g]` for sterilizing agents and `rel_imm[g] * sterilizing_p` for leaky agents. v3's effective per-genotype protection now matches v2's `~0.9975 * rel_imm[g]` to within ~0.25 percentage points.

- **v2 baseline generator counting bugs found and fixed during parity verification.** Two counting asymmetries in `tests/regression/multi_seed_v2_vx.py` were producing spurious gaps:
  - **`n_doses_2060` 13% inflated:** v2 resets `sim.people.vaccinated` to False on death but does NOT reset `sim.people.doses`, so `doses.sum()` counted dead-and-vaccinated agents while `vaccinated.sum()` did not. Fix: apply `sim.people.alive` mask to both. Gap collapsed from ~13% to ~1.3%.
  - **`cancer_incidence_2030_2060` 4.1× inflated:** v2's results aggregate to ANNUAL cadence (`resfreq=4` quarterly steps per annual entry), so each `n_alive` entry represents one year of person-years. The old `dt = sim.pars['dt'] = 0.25` understated person-years by 4×. Fix: `annual_dt = sim.resfreq * sim.pars['dt'] = 1.0`. Closed the gap to within noise.

- **v3 trajectory test downsamples quarterly → annual.** v2 stores trajectory rows at annual cadence (`resfreq=4`, 71 entries for a 70-year sim). v3 stores at quarterly per-step cadence (281 entries). The trajectory parity test now buckets v3's per-step `new_cancers` / `new_infections` / `new_vaccinated` to annual SUMS by `floor(year)` before comparison.

- **`v2_age_compat` shim on `hpv.BaseVaccination` added and then removed.**
  Hypothesizing a v2/v3 step-ordering bias (v2 advances age before
  interventions; Starsim advances age after), commit `5d279128` added an
  opt-in `v2_age_compat` kwarg that evaluated `sim.people.age + dt` against
  `age_range`. Further investigation (see
  `docs/superpowers/specs/2026-05-26-m05-parity-investigation.md`) showed
  the apparent ~30%/year deficit was a plot-metric artifact, not a real
  ordering bias, and that the combination of `v2_compat_demographics=True`
  (annual births + jitter-disabled migration + integer-age initial pop)
  alone closes the gap. Commit `74cede3b` removed the shim. The final
  M05 anchor PARS run with the shim OFF and `v2_compat_demographics=True`.

- **`AgeMigration` jitter disabled under v2_compat.** v3's `AgeMigration` spreads each immigrant's age uniformly across `[N, N+1)` to smooth cohort transitions. This propagates continuous-age distribution through the migration channel even after `AnnualBirths` fixes the births channel — immigrants who arrive as "age 0" can still be aged 0.0–0.99, and 9 years later contribute to the cohort breadth that the eligibility window is trying to align with. Under `v2_compat=True`, the jitter is skipped: immigrants land at exact integer ages, matching v2's `add_births` convention. `hpv.Sim`'s `v2_compat_births` kwarg is renamed to `v2_compat_demographics` to cover both channels under a single flag — it now (1) swaps `ss.Births` for `hpv.AnnualBirths` and (2) passes `v2_compat=True` to `AgeMigration`. M5 anchor PARS updated to use `v2_compat_demographics=True`.

- **`_coerce_sex` renamed to `_cast_sex`** (PR #111 review). The helper that
  normalizes v2-style `sex` input lives in `hpvsim/interventions.py`; the name
  changed but the behavior (returns a set of sex ints `{0}` / `{1}` / `{0,1}`,
  or `None`) is unchanged. Reviewer's alternative `_sex_to_bool` was declined
  because the return is a set, not a bool. Unit tests renamed to match.

- **`_find_genotype_module` promoted to `hpvsim.utils.find_genotype_module`**
  (PR #111 review). What the spec describes as a private method on `hpv.vx`
  is now a module-level `find_genotype_module(sim, genotype)` in
  `hpvsim/utils.py`, so any per-genotype module lookup has one home. Kept in
  `utils` (not `cross_genotype`) so `products` doesn't take a dependency on
  cross-genotype functionality it doesn't need; it late-imports `HPV` to stay
  import-light. Call site became `find_genotype_module(self.sim, genotype)`;
  behavior is unchanged.