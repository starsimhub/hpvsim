# M06: Test-and-Treat Cascade — Design

**Date:** 2026-05-27
**Milestone:** M06 (Screen-and-treat cascade)
**Branch:** `m06-test-and-treat-cascade` (off `v3.0-dev`; PR targets `v3.0-dev` once M05 merges)
**Predecessor:** [M05 Vaccination Scenarios](2026-05-20-m05-vaccination-scenarios-design.md)
**Status:** Spec drafted; implementation not started.

---

## Goal

Add the full screen → triage → treat cascade plus therapeutic vaccination
to v3 by composing Starsim 3.3.4's native intervention/product framework
with HPV-specific product `administer` overrides (per-genotype state
handling) and thin HPV subclasses that re-introduce v2's
`age_range`/`sex` targeting kwargs and HPV-specific eligibility filters
(female adult, cancer-status-matched, post-debut). A `hpv.Sim`
configured with `hpv.routine_screening` + `hpv.routine_triage` +
`hpv.treat_num` (using `hpv.dx` and `hpv.tx` products) must reproduce
v2.x's cervical-cancer-prevention impact trajectories within `|z| < 3` on
the M03 short-summary plus cascade-specific metrics. A second anchor
(`anchor_txvx_routine`) does the same for therapeutic vaccination. The
acceptance test mirrors `hpvsim_methods_manuscript`'s HSP screening
scenario shape.

This is M05's pattern repeated for ~10 classes. Starsim 3.3.4 already
ships `ss.Dx`, `ss.Tx`, `ss.Vx`, `ss.BaseTest`, `ss.BaseScreening`,
`ss.BaseTriage`, `ss.BaseTreatment`, `ss.treat_num`,
`ss.routine_screening`, `ss.campaign_screening`, `ss.routine_triage`,
`ss.campaign_triage`. M06's new code is (a) HPV-specific product classes
that override `administer` for per-genotype HPV state, (b) thin
HPV-specific subclasses that re-introduce v2's targeting/eligibility
kwargs, (c) the txvx family (no upstream equivalents), and (d) the
`treat_delay`, `radiation`, and `dynamic_pars` classes (no upstream
equivalents).

---

## Scope

### In scope (this PR, branched off `v3.0-dev` once M05 merges)

**Products** (`hpvsim/products.py`):

- `hpv.dx(ss.Dx)` — per-genotype multinomial classifier reading
  `module.precin / cin / cancerous / susceptible / latent` BoolStates
  across HPV modules in `sim.diseases.values()`. Loaded by `name` from
  `hpvsim/data/products_dx.csv` (defaults: `via`, `lbc`, `pap`,
  `colposcopy`, `hpv`, `hpv1618`, `hpv_type`, `txvx_assigner`,
  `tx_assigner`). Hierarchy defaults per-product per v2 convention. CSV's
  `latent` rows load verbatim; HPV's new no-op `latent` BoolState
  (default False) means no agent ever classifies as latent — matches
  current v3 natural-history behavior.
- `hpv.tx(ss.Tx)` — per-genotype state-flip treatment. Reads `efficacy`
  from `products_tx.csv` per (state, genotype). For successful
  precin/cin treatment: flips that genotype's `precin`/`cin` to False,
  schedules `ti_clearance = sim.ti + 1`, clears `ti_cin`/`ti_cancerous`
  to NaN. Successful agents' uids returned in `outcomes['successful']`.
  Defaults: `ablation`, `excision`.
- `hpv.txvx(ss.Vx)` — therapeutic vaccine. Two products by name (`txvx1`,
  `txvx2`) plus optional explicit `rel_imm` dict. Per-genotype `rel_imm`
  from `products_txvx.csv`. Bumps a new per-module `txvx_imm` FloatArr
  via the same max-of-existing semantics M05 settled on for `vax_imm`
  (independent protection path; does not flow through the cross-immunity
  matrix). `txvx2` (booster) multiplies by `imm_boost` instead of
  overwriting.
- `hpv.radiation(ss.Product)` — cancer treatment. Extends per-agent
  `ti_dead_cancer` by a configurable duration distribution (default:
  normal mean=18 months, sd=2 months, converted to years at
  construction). Standalone product (no `df`-based config).

**Interventions** (`hpvsim/interventions.py`):

- `hpv.BaseTest(ss.BaseTest)`, `hpv.BaseScreening(ss.BaseScreening)`,
  `hpv.BaseTriage(ss.BaseTriage)` — extend `_compose_eligibility` to
  compose `age_range` + `sex` (default `'f'`) + optional `debut_age`
  filter + user `eligibility` callable into a single Starsim eligibility
  callable. Override `_parse_product_str` so `routine_screening(product='via')`
  resolves through `hpv.dx(name='via')`.
- `hpv.routine_screening(hpv.BaseScreening, ss.RoutineDelivery)` /
  `hpv.campaign_screening(hpv.BaseScreening, ss.CampaignDelivery)` —
  empty-body diamond leaves.
- `hpv.routine_triage(hpv.BaseTriage, ss.RoutineDelivery)` /
  `hpv.campaign_triage(hpv.BaseTriage, ss.CampaignDelivery)` —
  empty-body diamond leaves.
- `hpv.BaseTreatment(ss.BaseTreatment)`, `hpv.treat_num(hpv.BaseTreatment,
  ss.treat_num)`, `hpv.treat_delay(hpv.BaseTreatment)` — adds
  HPV-specific eligibility (female, alive, optionally age-ranged,
  cancer-status-matched via `treat_cancer` flag derived from
  `isinstance(product, hpv.radiation)`). `treat_delay` is a fresh port
  (no Starsim equivalent) using an integer-`ti` scheduler.
- `hpv.BaseTxVx(hpv.BaseTreatment)`, `hpv.routine_txvx`,
  `hpv.campaign_txvx`, `hpv.linked_txvx` — `linked_txvx` is
  `hpv.BaseTxVx` with a `step()` that delegates to `deliver()`
  unconditionally (no own timeline; relies on a user-supplied
  `eligibility=lambda sim: sim.interventions.<screen>.outcomes['positive']`).
- `hpv.dynamic_pars(ss.Intervention)` — standalone class accepting
  `{dotted_path: {years: [...], vals: [...]}}` schedules. On each step,
  sets the resolved parameter to the interpolated (default) or
  stepwise (`interpolate=False`) value for the current year. Dotted-path
  resolution walks `sim.diseases` first, then `sim.interventions`, then
  `sim.pars`.

**HPV module deltas** (`hpvsim/hpv.py`):

- Add `ss.BoolState('latent', default=False)` per-genotype — no-op for
  now; CSV/dx hook ready for future natural-history work.
- Add `ss.FloatArr('txvx_imm', default=0.0)` per-genotype. Written by
  `hpv.txvx.administer`; read by `CrossImmunity`.

**CrossImmunity connector delta** (`hpvsim/cross_genotype.py`):

- Extend the independent-protection combine to include `txvx_imm`:
  `rel_sus = (1 - sus_imm_from_nab) * (1 - vax_imm) * (1 - txvx_imm)`.
  Matches the M05 rationale (vaccine immunity is an independent
  protection path that does not flow through the cross-immunity matrix).

**Data:**

- `products_dx.csv`, `products_tx.csv`, `products_txvx.csv` — already in
  `hpvsim/data/`. M06 adds `_load_dx_products`, `_load_tx_products`,
  `_load_txvx_products` cached helpers paralleling M05's
  `_load_vx_products`. No CSV content changes.

**Tests:**

- Unit tests for each product class, eligibility helper, and CSV loader.
- Integration smoke tests for routine + campaign + linked variants of
  each intervention family, the cascade composition pattern, capacity
  semantics on `treat_num`, timing semantics on `treat_delay`, and
  `dynamic_pars` parameter changes over time.
- `test_no_cascade_baseline_unchanged` CRN-perturbation guard (parallel
  to M05's `test_no_vx_baseline_unchanged`).
- Two regression anchor scenarios (`anchor_screen_treat`,
  `anchor_txvx_routine`) on the M03 Nigeria 4-genotype baseline.
- v2 baseline generation scripts (gitignored 30-seed JSON outputs).
- Short-summary parity tests at `|z| < 3` for both anchors; trajectory
  parity test on the screen+treat anchor only.

### Explicitly out of scope (deferred)

- HIV–HPV interaction effects on screening or treatment efficacy (M08).
- Per-screening-program cohort analyzers (M09).
- Migration-guide entries for the new intervention API (M10).
- Real porting of v2's `latent` reactivation natural history. Today's
  `latent` state is a no-op BoolState added to support the dx CSV
  schema; the post-clearance latency branch and `hpv_reactivation`
  dynamics in `hpvsim/_v2_legacy/sim.py:833-847` and
  `hpvsim/_v2_legacy/people.py:663-677` are tracked as a separate
  natural-history follow-on issue.
- Cervical screen-and-treat scenarios that exercise the
  `txvx_assigner` / `tx_assigner` dx products — those products' CSV
  rows load and unit-test correctly, but the M06 anchor scenarios use
  the canonical HPV → colposcopy → ablation/excision path rather than
  the assigner-routed dual-arm paths.
- Sex-specific cascade behavior. M06 defaults `sex='f'` for screening,
  triage, and treatment (matches v2's `is_female_adult` gate). No
  male-screening anchor.

---

## Architecture

### Module layout

```
hpvsim/
├── products.py            # +hpv.dx, hpv.tx, hpv.txvx, hpv.radiation      ~250 LOC delta
├── interventions.py       # +hpv.BaseTest/Screening/Triage/Treatment/TxVx
│                          #  leaves, +dynamic_pars                         ~350 LOC delta
├── hpv.py                 # +ss.BoolState('latent'), +ss.FloatArr('txvx_imm')  ~6 LOC
├── cross_genotype.py      # +txvx_imm in independent-protection combine    ~5 LOC
└── data/products_{dx,tx,txvx}.csv  # already in place (M05 prep work);
                                    # +loader functions in products.py
```

No new files. `hpvsim/__init__.py` exports get extended with the new
public names.

### Inheritance graph

```
ss.Module ── ss.Product ── ss.Dx ───────────────── hpv.dx
                       └── ss.Tx ───────────────── hpv.tx
                       └── ss.Vx ───────────────── hpv.txvx        # therapeutic
                       └── ss.Product ──────────── hpv.radiation

ss.Module ── ss.Intervention ── ss.BaseTest ── ss.BaseScreening ── hpv.BaseScreening
                                            └── ss.BaseTriage ──── hpv.BaseTriage
                            ── ss.BaseTreatment ────────────────── hpv.BaseTreatment
                                                               │
                                                               ├── ss.treat_num ── hpv.treat_num
                                                               └── (no Starsim) ── hpv.treat_delay
                            ── (no Starsim) ────────────────────── hpv.BaseTxVx (extends hpv.BaseTreatment)
                            ── (no Starsim) ────────────────────── hpv.dynamic_pars

           ── ss.RoutineDelivery ─┬── hpv.routine_screening (hpv.BaseScreening, ss.RoutineDelivery)
                                  ├── hpv.routine_triage    (hpv.BaseTriage,    ss.RoutineDelivery)
                                  └── hpv.routine_txvx      (hpv.BaseTxVx,      ss.RoutineDelivery)
           ── ss.CampaignDelivery ┬── hpv.campaign_screening
                                  ├── hpv.campaign_triage
                                  └── hpv.campaign_txvx
                                                                  hpv.linked_txvx (hpv.BaseTxVx, no delivery base)
```

`linked_txvx` is the odd one — it has no scheduled timepoints. Its
`step()` calls `deliver()` every step, relying entirely on a
user-supplied `eligibility=lambda sim: sim.interventions.<screen>.outcomes['positive']`
to gate when it fires. This is v2's design; it composes cleanly within
the Starsim intervention loop.

### Composition contract: order of operations within a step

`ss.Sim.run_one_step()` executes `step_state` (disease updates) →
`step_intv` (interventions in registration order) → `step()`
(transmission, births, deaths). Within `step_intv`, interventions
execute in the order they were added to `sim.interventions`. Users
compose cascades by ordering:

```python
sim = hpv.Sim(interventions=[
    hpv.routine_screening(name='primary_screen', product='hpv', ...),
    hpv.routine_triage(name='colposcopy', product='colposcopy',
                       eligibility=lambda s: s.interventions.primary_screen.outcomes['positive']),
    hpv.treat_num(name='treat', product='excision',
                  eligibility=lambda s: s.interventions.colposcopy.outcomes['hsil']),
    hpv.linked_txvx(product='txvx1', prob=0.6,
                    eligibility=lambda s: s.interventions.colposcopy.outcomes['lsil']),
])
```

`outcomes` is set inside `deliver()` and read by downstream interventions
later in the same step. **Order is the user's responsibility**, not the
framework's. The contract:

1. An intervention's `outcomes` is replaced (not mutated in place) at
   the start of its own `deliver()`.
2. Downstream interventions read `sim.interventions.<name>.outcomes`
   via attribute lookup, which always resolves to the current binding.
3. An out-of-order registration (e.g. `[treat, triage, screen]`) is
   legal but `treat.outcomes` will be empty on the first step that
   `screen` fires. Documented in the README; not enforced in code.
4. `linked_txvx` requires `eligibility=` (raises `ValueError` if
   missing).

### Per-intervention state vs. people state

Following M05's pattern: per-intervention state lives on the
intervention; cross-program reads walk `sim.interventions`.

| State | Lives on | Why |
|---|---|---|
| `screened` (BoolArr), `screens` (FloatArr), `ti_screened` (FloatArr) | `ss.BaseTest` (already in Starsim) | Per-intervention; cross-program reads via `sim.interventions.<name>.screened` |
| `cin_treated` (BoolArr), `cin_treatments` (FloatArr), `ti_cin_treated` (FloatArr) | `hpv.BaseTreatment` | Per-intervention; matches v2 attribute names |
| `cancer_treated`, `cancer_treatments`, `ti_cancer_treated` | `hpv.BaseTreatment` (only meaningful when `treat_cancer=True`) | Per-intervention |
| `tx_vaccinated`, `txvx_doses`, `ti_tx_vaccinated` | `hpv.BaseTxVx` | Per-intervention |
| `latent` (BoolState), `txvx_imm` (FloatArr) | `hpv.HPV` module (per-genotype) | Disease-module state; consumed by `hpv.dx` and `CrossImmunity` respectively |

`outcomes` is per-intervention (set in `deliver()`, lives until the next
`deliver()` call) — same as v2, same as Starsim.

### HPV-aware product `administer` overrides

This is where Starsim's defaults don't fit and HPV-specific code is
required:

- **`ss.Dx.administer`** iterates `self.diseases` (top-level disease
  names from CSV's `disease` column) and indexes
  `getattr(sim.diseases[disease], state).uids`. v3's HPV is per-genotype
  — `sim.diseases` contains `hpv16`, `hpv18`, `hi5`, `ohr`. The CSV uses
  a `genotype` column (`all` or specific genotype name) instead of a
  `disease` column. `hpv.dx.administer` overrides this to iterate
  per-genotype across HPV modules, classifying via `np.minimum` so the
  hierarchy-first result wins when an agent is positive across multiple
  genotypes (matches v2). When `genotype='all'`, susceptible means
  "susceptible to all genotypes"; non-susceptible means "infected with
  any".
- **`ss.Tx.administer`** flips a single
  `pre_state[uids]=False; post_state[uids]=True`. HPV's cleanup is more
  — clears `ti_cin` / `ti_cancerous` to NaN, schedules
  `ti_clearance = sim.ti + 1`. `hpv.tx.administer` overrides.
- **`ss.Vx.administer`** is unimplemented (`pass`). `hpv.txvx.administer`
  mirrors `hpv.vx.administer` against `txvx_imm` instead of `vax_imm`.
- **`ss.Product`** has no `radiation` analogue. `hpv.radiation.administer`
  writes per-agent `ti_dead_cancer += sample(self.dur)` on each affected
  genotype's cancer trajectory.

### CRN streams

Each intervention's `coverage_dist = ss.bernoulli(p=0)` (already on
`ss.BaseTest` and `ss.BaseTreatment`) and each product's per-state
distribution gets its own `ss.Dist` instance, registered with the sim's
dist registry via the standard Starsim pattern. No shared RNG streams
between cascade interventions. This is critical: M05 found that any
new `ss.Dist` that perturbs the upstream HPV transmission stream
manifests as silent drift in the baseline `cum_infections` /
`cum_cancers`. The `test_no_cascade_baseline_unchanged` test guards
against this.

---

## Per-class skeletons and v2 deltas

The skeletons below show the substantive deltas only. M05's pattern of
"constructor stores v2 kwargs, `_compose_eligibility` composes them,
MRO super().__init__ chains" repeats throughout.

### `hpv.dx`

Per-genotype multinomial classifier:

```python
class dx(ss.Dx):
    """HPV diagnostic product with per-genotype state classification."""

    def __init__(self, name=None, df=None, hierarchy=None, **kwargs):
        df, hierarchy = _resolve_dx_pars(name, df, hierarchy)
        super().__init__(df=df, hierarchy=hierarchy, **kwargs)
        self.name = name
        self._genotypes_in_df = df.genotype.unique()
        self._all_genotype = (len(self._genotypes_in_df) == 1
                              and self._genotypes_in_df[0] == 'all')

    def administer(self, uids, return_format='dict'):
        if len(uids) == 0:
            return {k: ss.uids() for k in self.hierarchy}
        results = np.full(len(uids), self.default_value, dtype=int)
        for state in self.health_states:    # 'susceptible', 'precin', 'cin', 'cancerous', 'latent'
            if self._all_genotype:
                # CSV row is genotype='all' — collapse across all HPV modules once per state.
                these = _state_collapse_across_genotypes(state, uids, self.sim)
                if len(these) == 0:
                    continue
                df_filter = (self.df.state == state) & (self.df.genotype == 'all')
                self._draw_and_min_into(results, uids, these, df_filter)
            else:
                # Per-genotype rows — iterate modules, classify each separately.
                for module in _iter_hpv_modules(self.sim):
                    if module.genotype not in self._genotypes_in_df:
                        continue
                    these = self._state_uids_for(module, state, uids)
                    if len(these) == 0:
                        continue
                    df_filter = (self.df.state == state) & (self.df.genotype == module.genotype)
                    self._draw_and_min_into(results, uids, these, df_filter)
        return _format_dx_output(results, uids, self.hierarchy, return_format)

    def _draw_and_min_into(self, results, uids, these, df_filter):
        probs = [self.df[df_filter & (self.df.result == r)].probability.values[0]
                 for r in self.hierarchy]
        self.result_dist.pars['p'] = probs
        draw = self.result_dist.rvs(these)
        idx_into_results = np.searchsorted(uids, these)
        results[idx_into_results] = np.minimum(draw, results[idx_into_results])
```

Key delta from `ss.Dx`: iteration is over `_iter_hpv_modules(sim)`
instead of `self.diseases`. The CSV's `genotype` column replaces the
`disease` column. The `all` genotype mode mirrors v2's "susceptible to
all / infected with any" semantics — implemented as
`_state_collapse_across_genotypes`. v2 reference:
`hpvsim/_v2_legacy/interventions.py:1285-1333`.

### `hpv.tx`

State-flip with HPV bookkeeping:

```python
class tx(ss.Tx):
    """HPV treatment product — state-flip per (state, genotype) with efficacy draw."""

    def __init__(self, name=None, df=None, **kwargs):
        df = _resolve_tx_pars(name, df)
        super().__init__(df=df, **kwargs)
        self.name = name

    def administer(self, uids, return_format='dict'):
        successful = []
        for state in self.health_states:
            for module in _iter_hpv_modules(self.sim):
                df_filter = (self.df.state == state) & (
                    (self.df.genotype == module.genotype) | (self.df.genotype == 'all')
                )
                rows = self.df[df_filter]
                if len(rows) == 0:
                    continue
                state_arr = getattr(module, state)            # BoolArr
                these = state_arr.uids.intersect(uids)
                if len(these) == 0:
                    continue
                self.efficacy_dist.set(p=float(rows.efficacy.values[0]))
                eff = self.efficacy_dist.filter(these)
                if len(eff) == 0:
                    continue
                successful.extend(eff)
                # State cleanup mirrors v2 hpvsim/_v2_legacy/interventions.py:1387-1391
                module.cin[eff] = False
                module.precin[eff] = False
                module.cancerous[eff] = False                 # only meaningful if state was cancerous
                module.ti_cin[eff] = np.nan
                module.ti_cancerous[eff] = np.nan
                module.ti_clearance[eff] = self.sim.ti + 1
        successful = ss.uids(sorted(set(successful)))
        unsuccessful = ss.uids(np.setdiff1d(uids, successful))
        return {'successful': successful, 'unsuccessful': unsuccessful} if return_format == 'dict' else successful
```

`ti_clearance = sim.ti + 1` matches v2's `date_clearance = people.t + 1`
exactly. v2 also has a commented-out "did they also clear infection?"
branch using `dur_infection`; v2 itself disabled this, so v3 doesn't
re-implement it.

### `hpv.txvx`

Parallel structure to M05's `hpv.vx`:

```python
class txvx(ss.Vx):
    """HPV therapeutic vaccine product."""

    def __init__(self, name=None, rel_imm=None, sterilizing_p=0.95, imm_boost=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(name=name, rel_imm=rel_imm,
                         sterilizing_p=sterilizing_p, imm_boost=imm_boost)
        self.rel_imm = _resolve_txvx_pars(name, rel_imm)
        self._sterilizing_dist = ss.bernoulli(p=0.0)

    def administer(self, people, uids):
        if len(uids) == 0:
            return
        if self.pars.imm_boost is not None:
            for module in _iter_hpv_modules(self.sim):
                module.txvx_imm[uids] *= float(self.pars.imm_boost)   # 2nd dose
            return
        self._sterilizing_dist.set(p=float(self.pars.sterilizing_p))
        sterilizing_uids = self._sterilizing_dist.filter(uids)
        is_sterilizing = np.isin(uids, sterilizing_uids)
        for genotype, rel_imm_g in self.rel_imm.items():
            module = _find_genotype_module(self.sim, genotype)
            if module is None:
                continue
            peak = np.where(is_sterilizing,
                            float(rel_imm_g),
                            float(rel_imm_g) * float(self.pars.sterilizing_p))
            module.txvx_imm[uids] = np.maximum(module.txvx_imm[uids], peak)
```

Architecturally identical to `hpv.vx` — different target state
(`txvx_imm` not `vax_imm`), optional `imm_boost` mode for booster doses
(v2's `txvx2`). `_find_genotype_module` is the existing M05
implementation, moved to a module-private function and shared.

### `hpv.radiation`

```python
class radiation(ss.Product):
    def __init__(self, dur=None, **kwargs):
        super().__init__(**kwargs)
        # v2 default: normal(mean=18 months, sd=2 months). Convert to years.
        self.define_pars(dur=dur or dict(dist='normal', par1=18/12, par2=2/12))
        self._dur_dist = ss.normal(loc=self.pars.dur['par1'], scale=self.pars.dur['par2'])

    def administer(self, uids):
        if len(uids) == 0:
            return ss.uids()
        new_dur = self._dur_dist.rvs(uids)
        dt = self.sim.t.dt
        for module in _iter_hpv_modules(self.sim):
            cancer_uids = module.cancerous.uids.intersect(uids)
            if len(cancer_uids):
                idx = np.isin(uids, cancer_uids)
                module.ti_dead_cancer[cancer_uids] += np.ceil(new_dur[idx] / dt)
        return uids
```

v2 stores duration in months; v3 converts to years at construction. The
integer-timestep extension mirrors v2's
`np.ceil(new_dur_cancer / people.pars['dt'])`.

### `hpv.BaseTest`

HPV-specific default eligibility:

```python
class BaseTest(ss.BaseTest):
    def __init__(self, *args, age_range=None, sex='f', eligibility=None,
                 debut_age=None, **kwargs):
        composed = _compose_screening_eligibility(age_range, sex, eligibility, debut_age)
        super().__init__(*args, eligibility=composed, **kwargs)
        self.age_range = age_range
        self.sex_raw = sex
        self.sex = _coerce_sex(sex)
        self.eligibility_raw = eligibility
        self.debut_age = debut_age

    def _parse_product_str(self, product):
        return dx(name=product)
```

`_compose_screening_eligibility` is a screening-flavored extension of
M05's `_compose_eligibility` — same `age_range`/`sex` composition, plus
an optional `sim.people.age >= debut_age` filter. Default `sex='f'`
matches v2's `is_female_adult`. Triage inherits the same.

### `hpv.BaseTreatment`

HPV-specific cancer-vs-noncancer gate:

```python
class BaseTreatment(ss.BaseTreatment):
    def __init__(self, *args, age_range=None, sex='f', eligibility=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.age_range = age_range
        self.sex_raw = sex
        self.sex = _coerce_sex(sex)
        self.eligibility_user = eligibility
        self.treat_cancer = isinstance(self.product, radiation)
        self.define_states(
            ss.BoolArr('cin_treated'),
            ss.FloatArr('cin_treatments', default=0),
            ss.FloatArr('ti_cin_treated'),
            ss.BoolArr('cancer_treated'),
            ss.FloatArr('cancer_treatments', default=0),
            ss.FloatArr('ti_cancer_treated'),
        )

    def check_eligibility(self):
        sim = self.sim
        cond = sim.people.alive & sim.people.female      # matches M05 helper convention
        if self.age_range is not None:
            lo, hi = self.age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age <= hi)
        any_cancer = _any_genotype_cancer(sim)
        cond = cond & (any_cancer if self.treat_cancer else ~any_cancer)
        if self.eligibility_user is not None:
            cond = cond & _as_boolarr(self.eligibility_user(sim), sim.people)
        return cond.uids

    def step(self):
        treat_uids = super().step()
        if len(treat_uids):
            if self.treat_cancer:
                new = treat_uids[~self.cancer_treated[treat_uids]]
                self.cancer_treated[treat_uids] = True
                self.cancer_treatments[treat_uids] += 1
                self.ti_cancer_treated[treat_uids] = self.sim.ti
                self.results['new_cancer_treated'][self.sim.ti] += len(new)
            else:
                new = treat_uids[~self.cin_treated[treat_uids]]
                self.cin_treated[treat_uids] = True
                self.cin_treatments[treat_uids] += 1
                self.ti_cin_treated[treat_uids] = self.sim.ti
                self.results['new_cin_treated'][self.sim.ti] += len(new)
        return treat_uids
```

`_any_genotype_cancer(sim)` returns a BoolArr OR-ing `module.cancerous`
across all HPV modules — parallel to v2's
`sim.people.cancerous.any(axis=0)`.

### `hpv.treat_delay`

Fresh port (no Starsim equivalent):

```python
class treat_delay(BaseTreatment):
    def __init__(self, delay=None, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay or 0
        self.scheduler = defaultdict(list)

    def add_to_schedule(self):
        accept = self.get_accept_inds()
        if len(accept):
            due_ti = self.sim.ti + int(np.round(self.delay / self.sim.t.dt))
            self.scheduler[due_ti].extend(accept.tolist())

    def get_candidates(self):
        return ss.uids(self.scheduler.pop(self.sim.ti, []))

    def step(self):
        self.add_to_schedule()
        return super().step()
```

Integer-`ti` scheduler keys are the M05-lesson upgrade — v2 used a float
subtraction (`sim.t - self.delay/dt`) that's fragile under timestep
changes; integer-`ti` math is exact.

### `hpv.linked_txvx`

Eligibility-driven, no own timeline:

```python
class linked_txvx(BaseTxVx):
    """TxVx delivery linked to another intervention's outcomes.

    Eligibility must be set; no RoutineDelivery/CampaignDelivery base.
    """
    def __init__(self, *args, eligibility=None, **kwargs):
        if eligibility is None:
            raise ValueError(
                "linked_txvx requires eligibility= "
                "(typically a screen.outcomes['positive'] callback)"
            )
        super().__init__(*args, eligibility=eligibility, **kwargs)
        self.timepoints = None              # No own schedule

    def step(self):
        return self.deliver()               # Fires unconditionally; eligibility gates
```

### `hpv.dynamic_pars`

Orthogonal standalone class:

```python
class dynamic_pars(ss.Intervention):
    """Time-varying parameter editor.

    pars: dict mapping dotted-path (e.g. 'hpv16.beta', 'beta',
    'condom_use.network') to {'years': [...], 'vals': [...]} schedule.
    Interpolated linearly by default, or stepwise if interpolate=False.
    """
    def __init__(self, pars=None, interpolate=True, **kwargs):
        super().__init__(**kwargs)
        self.par_schedules = pars or {}
        self.interpolate = interpolate

    def step(self):
        year = self.sim.t.now('year')
        for dotted_path, schedule in self.par_schedules.items():
            years, vals = np.asarray(schedule['years']), np.asarray(schedule['vals'])
            if self.interpolate:
                val = np.interp(year, years, vals)
            else:
                idx = np.searchsorted(years, year, side='right') - 1
                if idx < 0:
                    continue
                val = vals[idx]
            _set_dotted(self.sim, dotted_path, val)
```

`_set_dotted` walks the path: top-level segment is looked up in
`sim.diseases` first, then `sim.interventions`, then falls back to
`sim.pars`. v2's `dynamic_pars` uses timestep keys; v3 uses `year`
directly so users can write schedules in epoch years without thinking
about `dt`.

### Sex coercion and eligibility composition

Both `_coerce_sex` and `_compose_eligibility` already exist in
`hpvsim/interventions.py` from M05. M06 adds
`_compose_screening_eligibility` (extends M05's helper with the
`debut_age` filter) and `_any_genotype_cancer` (BoolArr OR across HPV
modules) — both are private module-level helpers.

---

## Data: CSV schemas

All three CSVs are already in `hpvsim/data/` (copied from
`hpvsim/_v2_legacy/data/` as part of the M05 prep work). No content
changes in M06.

| CSV | Columns | Rows | Products |
|---|---|---|---|
| `products_dx.csv` | `name, state, genotype, result, probability` | 524 | `via`, `lbc`, `pap`, `colposcopy`, `hpv`, `hpv1618`, `hpv_type`, `txvx_assigner`, `tx_assigner` |
| `products_tx.csv` | `name, state, genotype, efficacy` | 71 | `ablation`, `excision`, `txvx1`, `txvx2` |
| `products_txvx.csv` | `name, genotype, rel_imm` | 17 | `txvx1`, `txvx2` |

`txvx1` and `txvx2` appear in BOTH `products_tx.csv` (which defines
state-flip efficacy per state×genotype for the treatment-effect arm)
and `products_txvx.csv` (which defines per-genotype `rel_imm` for the
immunity arm). v2 used both — `hpv.txvx` reads `products_txvx.csv` for
immunity, and the treatment-effect arm is layered via the same
intervention (the v2 `tx` Product accepts `genotype_pars` from
`products_txvx.csv` in addition to its `df` from `products_tx.csv`).
v3 follows the same dual-CSV convention.

CSV's `latent` state rows in `products_dx.csv` load verbatim. The
no-op `latent` BoolState on HPV ensures no agent classifies as latent
under v3's current natural history; the rows are inert until latency
is ported.

---

## Tests

### Unit tests

| File | Verifies |
|---|---|
| `test_m06_dx_unit.py` | `_load_dx_products` CSV parse; `hpv.dx(name='via').hierarchy == ['positive', 'inadequate', 'negative']`; per-genotype iteration on a synthetic 2-genotype sim returns hierarchy-min results; `genotype='all'` mode collapses correctly (susceptible iff susceptible-to-all, positive iff infected-with-any); `latent` state matches zero agents under current v3 HPV; unknown name raises `ValueError`; `result_dist` is a registered `ss.Dist` instance |
| `test_m06_tx_unit.py` | `hpv.tx(name='ablation')` flips `cin[g]=False` and schedules `ti_clearance=ti+1` for successful agents; unsuccessful agents untouched; `outcomes` dict has disjoint `successful`/`unsuccessful` keys; `efficacy=0` rows produce zero successful agents |
| `test_m06_radiation_unit.py` | `hpv.radiation.administer` extends `ti_dead_cancer` per cancer-positive genotype; doesn't touch non-cancer agents; respects `dt`-to-`ti` conversion |
| `test_m06_txvx_unit.py` | `txvx1` first-dose semantics match `hpv.vx` modulo target state (`txvx_imm` not `vax_imm`); `txvx2` booster multiplies in place; inactive-genotype tolerance; CRN-isolated `_sterilizing_dist` |
| `test_m06_eligibility_unit.py` | `_compose_screening_eligibility(age_range, sex, extra, debut_age)` intersection correct (extends M05's `_compose_eligibility` with optional `debut_age` filter); `_any_genotype_cancer` ORs correctly across HPV modules |
| `test_m06_treat_delay_unit.py` | Schedule keyed by `sim.ti + delay/dt`; agents enqueued at ti=T fire at ti=T+delay_ti; queue cleared after fire; varying `dt` produces correct integer schedules |
| `test_m06_dynamic_pars_unit.py` | Linear interpolation between years; stepwise mode with `interpolate=False`; dotted-path resolution for `sim.diseases.<g>.pars.<p>` / `sim.interventions.<n>.<f>` / `sim.pars.<p>`; out-of-range years use last-known value; unresolvable path raises `KeyError` |
| `test_m06_loaders_unit.py` | All three product CSVs load via `@functools.lru_cache(maxsize=1)`-cached helpers; column-schema check raises `ValueError` on missing columns; products in each CSV match the defaults named in this spec |
| `test_m06_cross_immunity_combine.py` | Three-factor combine: agent with `nab_imm=0`, `vax_imm=0.5`, `txvx_imm=0.5` has `rel_sus = 0.25` |

### Integration smoke tests

| Test | Verifies |
|---|---|
| `test_routine_screening_smoke` | `hpv.routine_screening(product='via', prob=0.5)`; `intv.screened` flips for targeted agents; `n_screened`/`n_dx` results increment per step |
| `test_full_cascade_smoke` | screen → triage → treat composed by ordering and `eligibility` callbacks; each step's `outcomes` is visible to the next intervention in the same step |
| `test_linked_txvx_smoke` | `linked_txvx` with eligibility callback fires only on screen-positive agents; `tx_vaccinated`/`txvx_doses` increment |
| `test_linked_txvx_requires_eligibility` | `hpv.linked_txvx()` without `eligibility=` raises `ValueError` |
| `test_treat_num_capacity` | with `max_capacity=10`, exactly 10 agents per step are treated; remainder stay in queue; queue drains FIFO |
| `test_treat_delay_timing` | `delay=2` years means agents enqueued at year 2025 fire at year 2027 |
| `test_radiation_cancer_only` | `hpv.treat_num(product=hpv.radiation())` sets `treat_cancer=True`; only cancerous agents are eligible; `ti_dead_cancer` extended |
| `test_dynamic_pars_beta_ramp` | `dynamic_pars(pars={'hpv16.beta': {'years':[2020,2030], 'vals':[0.5,0.2]}})` actually changes hpv16.beta over time |
| `test_no_cascade_baseline_unchanged` | sim with no cascade interventions reproduces a pre-M06 scalar (`hpvtotal['cum_infections'].sum()` and `cum_cancers.sum()`) — CRN perturbation guard, mirrors M05's `test_no_vx_baseline_unchanged` |
| `test_cascade_order_dependency` | `[treat, triage, screen]` registration order produces empty `treat.outcomes['successful']` on the first step `screen` fires (legal but order-dependent; documented contract) |

### Parity tests (slow, on-demand)

```
tests/regression/anchor_screen_treat.py            # NEW — PARS for full cascade
tests/regression/anchor_txvx_routine.py            # NEW — PARS for routine txvx
tests/regression/multi_seed_v2_screen_treat.py     # NEW — 30-seed v2 baseline generator
tests/regression/multi_seed_v2_txvx.py             # NEW — 30-seed v2 baseline generator
tests/regression/v2_seeds_n30_screen_treat.json    # local-only, gitignored
tests/regression/v2_seeds_n30_txvx.json            # local-only, gitignored
tests/test_m06_screen_treat_parity.py              # short-summary z gate, screen+treat
tests/test_m06_txvx_parity.py                      # short-summary z gate, txvx
tests/test_m06_trajectory_parity.py                # trajectory z gate, screen+treat only
```

All parity tests follow M03/M05's pattern: 10 v3 seeds × full sim,
per-metric z-score against 30-seed v2 baseline, gate at `|z| < 3.0`.
All marked `@pytest.mark.slow`; CI runs `-m 'not slow'`.

### Cascade-specific summary metrics

Extending M03's 40-entry short summary:

| Metric | Captures |
|---|---|
| `n_screened_2060` | cumulative first-screens at sim end |
| `n_screens_2060` | cumulative screens delivered |
| `n_cin_treated_2060` | cumulative first-CIN-treatments |
| `n_cin_treatments_2060` | cumulative CIN treatments delivered |
| `n_cancer_treated_2060` | cumulative first-cancer-treatments (radiation arm) |
| `cancer_incidence_2030_2060` | post-cascade cancer incidence — the actual prevention signal |
| `cancer_deaths_2030_2060` | post-cascade cancer deaths (radiation impact) |

Plus for the txvx anchor:

| Metric | Captures |
|---|---|
| `n_tx_vaccinated_2060` | cumulative first-doses |
| `n_txvx_doses_2060` | cumulative doses |
| `cancer_incidence_2030_2060` | post-txvx cancer incidence |

### Anchor scenarios

Built on the M03 Nigeria 4-genotype baseline, paralleling M05's anchor
structure. Both anchors set `PARS.v2_compat_demographics = True` (M05
lesson — closes the demographic cohort gap).

- **`anchor_screen_treat`**: HPV primary screening (`product='hpv'`,
  prob=0.7 annual, age_range=[30,50], start_year=2020) → colposcopy
  triage (`product='colposcopy'`, prob=0.9,
  eligibility=screen-positives) → `treat_num(product='excision',
  prob=0.8, eligibility=colposcopy-HSIL)`. Mirrors
  `hpvsim_methods_manuscript`'s HSP scenario shape.
- **`anchor_txvx_routine`**: vanilla M03 Nigeria sim plus one
  `routine_txvx(product='txvx1', prob=0.6, age_range=[25,26],
  start_year=2030)`. Female-only by default.

### Pre-PR gate

1. `pytest -m 'not slow'` green on the M06 branch.
2. Both parity tests green locally
   (`pytest -m slow tests/test_m06_*_parity.py`).
3. Trajectory parity test green locally.
4. M03 + M04 + M05 tests still green (regression guard).
5. Anchor regeneration documented in `tests/regression/README_m06.md`.

---

## M05 lessons codified

Six concrete cross-checks baked into M06's risk + test design, derived
from documented M05 issues:

1. **Counting cadence trap.** v2 baseline generators must apply
   `sim.people.alive` mask to all flow counters (M05 found
   `n_doses_2060` over-counted by 13% from dead-but-vaccinated agents
   — see `tests/regression/multi_seed_v2_vx.py` post-fix).
   `multi_seed_v2_screen_treat.py` and `multi_seed_v2_txvx.py`
   explicitly alive-mask `screened`, `screens`, `cin_treated`,
   `cin_treatments`, `cancer_treated`, `cancer_treatments`,
   `tx_vaccinated`, `txvx_doses` before summing.
2. **Person-years aggregation.** v2's `cancer_incidence_2030_2060`
   baseline used `dt=0.25` but v2 stores annual results so the
   effective person-year denominator is `resfreq * dt = 1.0` — M05
   hit a 4× over-count from this. M06 generator scripts use
   `annual_dt = sim.resfreq * sim.pars['dt']`.
3. **Per-step vs flow counter mismatch.** v3's
   `ti_screened`/`ti_cin_treated` are LAST-event stamps under
   alive-mask — same trap as M05's `ti_vaccinated`. The trajectory
   parity test uses a `_FirstEventLogger` analyzer pattern (snapshot
   `BoolArr.raw` pre/post step, count False→True transitions) to get
   v2-equivalent flow counters from each intervention. Implemented in
   `tests/test_m06_trajectory_parity.py` similar to M05's
   `_FirstVaxLogger`.
4. **CRN perturbation guard.** `test_no_cascade_baseline_unchanged`
   pins a pre-M06 scalar (`hpvtotal['cum_infections'].sum()` and
   `cum_cancers.sum()`). If any new `ss.Dist` instance in M06 shares
   an RNG stream with an HPV transmission decision, this test catches
   it. Each intervention/product constructs its own `ss.Dist`.
5. **Trajectory quarterly→annual downsampling.** v3 stores per-step
   results; v2 stores annual. The trajectory test buckets v3's
   per-step `new_cancers` / `new_cin_treated` / `new_screened` to
   annual sums by `floor(year)` before comparison.
6. **Order-of-operations explicit in spec.** M05's runs had a
   `boundary fire +1 step` rabbit hole driven by ordering ambiguity.
   M06's "Composition contract" section explicitly states user-ordered
   intra-step composition AND warns about `linked_txvx`'s reliance on
   its upstream intervention's `outcomes` being set on the same step.

The `v2_compat_demographics=True` flag (M05 final setting per
[project memory](../../../../memory/project_m05_vx_parity.md)) is the
default in both M06 anchor PARS.

---

## Open risks and mitigations

| Risk | Mitigation |
|---|---|
| `Dx` per-genotype rewrite drifts from `ss.Dx`. Subtle deltas (`np.minimum` semantics over multi-genotype draws, `all` mode collapse) could miss a v2 corner | Unit tests pin every CSV mode (per-genotype, `all`, `susceptible+all`, `infected+all`). Add a parity sub-test: build a synthetic 4-genotype sim where one agent is precin on hpv16 only, run `hpv.dx(name='hpv_type')`, assert the hierarchy result is identical to v2's classification on the same agent state |
| `ti_clearance = sim.ti + 1` from `hpv.tx`. If `step_state` runs at a different point in the loop than the intervention `step()`, the +1 lands wrong | Starsim step order is `step_state → step_intv → step()`, so an intervention writing `ti_clearance=ti+1` at ti=T is read by `step_state` at ti=T+1 correctly. Document in `hpv.tx.administer` and add an integration test that an agent treated at ti=T clears at ti=T+1 |
| `linked_txvx` reads `outcomes` from a same-step upstream — fragile to registration ordering | `linked_txvx.__init__` requires `eligibility=` (raises `ValueError`). Document the ordering contract in the class docstring and add an integration test (`test_cascade_order_dependency`) that an early-registered `linked_txvx` produces zero doses on the first step (correct v2 behaviour, surfaces the contract) |
| `treat_delay` integer-`ti` scheduler vs v2's float subtraction. `round` could create a 1-step drift when `delay/dt` isn't an integer | Unit test pins integer cases (delay=0, 1, 2 with dt=0.25) and a non-integer case (delay=0.5 with dt=0.25 → 2 steps). Document that non-integer delays round to nearest integer step. v2's parity-relevant scenarios use integer-month delays |
| `dynamic_pars` dotted-path resolution: `sim.pars['hpv16.beta']` doesn't work — it's `sim.diseases['hpv16'].pars.beta` | `_set_dotted` resolution order: `sim.diseases` first, then `sim.interventions`, then `sim.pars`. Unit test pins all three branches. Raises `KeyError` with a path hint on miss |
| `txvx_imm` in CrossImmunity combine. M06 introduces a third independent factor on top of M05's two | Update `CrossImmunity` combine to `rel_sus = (1 - sus_imm_from_nab) * (1 - vax_imm) * (1 - txvx_imm)`. Unit test `test_m06_cross_immunity_combine` pins agent with `vax_imm=0.5` and `txvx_imm=0.5` → `rel_sus=0.25` |
| No-op `latent` BoolState bloat — adds an extra state per agent per genotype | ~1 byte × N_agents × N_genotypes (e.g. ~400 kB for 100k agents × 4 genotypes — negligible). Verify with a `Sim.shrink()` size delta in the post-implementation section |
| `outcomes` dict identity. Reassignment vs. in-place mutation matters for downstream `linked_txvx` callbacks | Always `outcomes = {k: ss.uids() for k in hierarchy}` at the start of `deliver()` (not in place). Downstream attribute lookup always resolves to the current binding. Unit test pins dict-identity behavior |
| Multi-intervention name collisions — Starsim collides same-type interventions without unique `name=` | Anchor PARS scripts pass explicit `name=` to every intervention. Documented in `tests/regression/README_m06.md` |
| Parity test compute budget — two anchors + one trajectory test | Reuse M05's compute budget. CI gates `-m 'not slow'`; parity tests stay local-only |

---

## Implementation sub-task sequence

Dependency-respecting order for the writing-plans output (each step
becomes one or more TDD-style tasks):

1. Add `ss.BoolState('latent', default=False)` and
   `ss.FloatArr('txvx_imm', default=0.0)` to `hpv.HPV`. Update
   `CrossImmunity` to include `txvx_imm` in the independent-protection
   combine. Add `test_m06_cross_immunity_combine` unit test.
2. Add `_load_dx_products`, `_load_tx_products`,
   `_load_txvx_products`, and `_resolve_*_pars` helpers in
   `hpvsim/products.py`. Move `_find_genotype_module` and add
   `_iter_hpv_modules` at module level. Unit-test
   (`test_m06_loaders_unit.py`).
3. Implement `hpv.dx`. Unit-test all CSV modes
   (`test_m06_dx_unit.py`).
4. Implement `hpv.tx`. Unit-test state flips, `ti_clearance`
   scheduling, outcomes dict (`test_m06_tx_unit.py`).
5. Implement `hpv.txvx`. Unit-test first-dose and booster behavior
   (`test_m06_txvx_unit.py`).
6. Implement `hpv.radiation`. Unit-test `ti_dead_cancer` extension
   (`test_m06_radiation_unit.py`).
7. Implement `_compose_screening_eligibility` and
   `_any_genotype_cancer` helpers. Unit-test
   (`test_m06_eligibility_unit.py`).
8. Implement `hpv.BaseTest`, `hpv.BaseScreening`, `hpv.BaseTriage`
   plus leaves (`hpv.routine_screening`, `hpv.campaign_screening`,
   `hpv.routine_triage`, `hpv.campaign_triage`). Smoke-test
   composition.
9. Implement `hpv.BaseTreatment`, `hpv.treat_num`, `hpv.treat_delay`.
   Smoke-test capacity (`treat_num`) and timing (`treat_delay`).
10. Implement `hpv.BaseTxVx`, `hpv.routine_txvx`, `hpv.campaign_txvx`,
    `hpv.linked_txvx`. Smoke-test linked-eligibility composition.
11. Implement `hpv.dynamic_pars`. Smoke-test against a beta ramp on
    hpv16.
12. Wire all new public names into `hpvsim/__init__.py`.
13. Author anchor PARS scripts (`anchor_screen_treat.py`,
    `anchor_txvx_routine.py`).
14. Author v2 baseline generators (`multi_seed_v2_screen_treat.py`,
    `multi_seed_v2_txvx.py`) applying M05 lessons 1+2 (alive-mask
    everywhere, `annual_dt = resfreq*dt`).
15. Regenerate 30-seed v2 baselines locally (from a v2 env); gitignored
    JSON outputs.
16. Author short-summary parity tests
    (`test_m06_screen_treat_parity.py`, `test_m06_txvx_parity.py`).
17. Author trajectory parity test (`test_m06_trajectory_parity.py`)
    using the `_FirstEventLogger` pattern (M05 lesson 3).
18. Author `test_no_cascade_baseline_unchanged` CRN-perturbation guard
    and `test_cascade_order_dependency` integration test.
19. CI workflow check (confirm `-m 'not slow'` still bounds CI; no new
    slow tests leak).
20. MIGRATION_PLAN.md edits — flip M06 status to `🟡 In progress;
    branch m06-test-and-treat-cascade`. Update sub-task list to
    reflect the landed shape (Starsim-native diamond, txvx + radiation
    + dynamic_pars all in this PR).
21. Open the M06 PR against `v3.0-dev` after M05 PR merges.

---

## Quarantine and active-code policy

Per the user's standing feedback memory (copy v2 logic into active
code; never import from `_v2_legacy/`), M06 uses the three product
CSVs already in `hpvsim/data/` (copied verbatim as part of the M05 prep
work). The quarantined `hpvsim/_v2_legacy/data/products_{dx,tx,txvx}.csv`
copies remain for porting reference but are no longer the live source.

M06 introduces zero new runtime imports from `hpvsim/_v2_legacy/`. The
HPV-specific `administer` overrides are written from scratch against
the v3 HPV state interface; v2's implementations in
`hpvsim/_v2_legacy/interventions.py:1265-1492` are referenced in code
comments (with file:line citations) where the v2 logic is being
mirrored, per migration convention.

---

## MIGRATION_PLAN.md edits

Committed separately on the M06 branch:

**M6 § Status table** — flips to `🟡 In progress; branch
m06-test-and-treat-cascade`.

**M6 § Sub-tasks** — rewritten to reflect the Starsim-native
architecture and the full-M06-in-one-PR scope:

```
### M6: Screen-and-treat cascade

**Demo:** Run a screen → triage → treat scenario on one country plus
therapeutic vaccination on the same baseline.

**Acceptance test:** Screening + treatment anchor reproduces v2.x intervals
on the `hpvsim_methods_manuscript` HSP scenario shape; txvx anchor reproduces
v2.x intervals on the v2 txvx scenarios. Both gated at `|z| < 3` on the M03
short-summary plus cascade-specific metrics, and trajectory parity on the
screen+treat anchor.

**Sub-tasks:**
- Add `hpv.dx(ss.Dx)` per-genotype diagnostic product with `_load_dx_products`
  loader from `hpvsim/data/products_dx.csv`. Handles `all`/per-genotype CSV
  modes via overridden `administer`.
- Add `hpv.tx(ss.Tx)` per-genotype treatment product with state-flip and
  `ti_clearance = sim.ti + 1` scheduling.
- Add `hpv.txvx(ss.Vx)` therapeutic vaccine product mirroring `hpv.vx`
  architecture; writes to a new per-module `txvx_imm` FloatArr.
- Add `hpv.radiation(ss.Product)` standalone product extending
  `ti_dead_cancer`.
- Add `hpv.BaseTest`, `hpv.BaseScreening`, `hpv.BaseTriage` with HPV
  default eligibility (female + alive + optional debut_age); thin
  diamond leaves `hpv.routine_screening`/`campaign_screening`/
  `routine_triage`/`campaign_triage` combining with Starsim's delivery
  bases.
- Add `hpv.BaseTreatment`, `hpv.treat_num` (extends `ss.treat_num`),
  `hpv.treat_delay` (fresh port, integer-`ti` scheduler).
- Add `hpv.BaseTxVx`, `hpv.routine_txvx`, `hpv.campaign_txvx`,
  `hpv.linked_txvx` (no own timeline; eligibility-driven).
- Add `hpv.dynamic_pars` (year-keyed schedule with dotted-path resolution
  into `sim.diseases` / `sim.interventions` / `sim.pars`).
- Add per-module `latent` BoolState (no-op for now; CSV/dx hook ready)
  and `txvx_imm` FloatArr to `hpv.HPV`; update `CrossImmunity` connector
  to include `txvx_imm` in the independent-protection combine.
- Add two regression anchors (`anchor_screen_treat`, `anchor_txvx_routine`),
  v2 baseline generator scripts, and multi-seed `|z| < 3` parity tests
  (M03 + M05 pattern). Trajectory parity on the screen+treat anchor only.
- Add `test_no_cascade_baseline_unchanged` CRN-perturbation guard.
- Add unit tests for product administer logic, eligibility helpers, and
  loader CSV schemas.
```

---

## Post-implementation deltas

To be filled in after implementation lands, documenting any divergences
from this spec discovered during the build. Format follows M03 and M05.