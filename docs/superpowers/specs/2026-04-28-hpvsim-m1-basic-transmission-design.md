# M01: Basic Transmission Sim — Design

**Date:** 2026-04-28
**Milestone:** M01 (Basic transmission sim)
**Branch:** `m01-basic-transmission-sim` (off `v3.0-dev`)
**Predecessor:** [M00 Foundation](2026-04-28-hpvsim-m0-foundation-design.md)
**Status:** Draft pending implementation plan

---

## Goal

Build the **minimum runnable HPV simulation on Starsim** that ports HPVsim's
sexual network and adds a single-genotype transmission-only HPV disease module.
Demonstrate via aggregate HPV prevalence over time on a Nigeria anchor scenario,
and validate partnership patterns against v2.x within tolerance.

This is an **in-place replacement** of v2's `hpvsim` package, not a coexisting
v3 subpackage. Public API stays at `hpvsim` (e.g., `import hpvsim as hpv`,
`hpv.Sim(...)`, `hpv.HPV(...)`). v2 modules that don't have a v3 equivalent
yet are quarantined to `hpvsim/_v2_legacy/` for porting reference; v2 tests
that exercise removed APIs are quarantined to `tests/_legacy/`. Both
quarantines exist so M02-M09 porters can consult v2 code by line, not by
git archaeology, and so the active package surface only exposes
current-milestone functionality.

## Scope

**In scope:**
- `hpv.HPV(ss.Infection)` — single-genotype (HPV16) transmission-only disease
  module. SIS clearance dynamics; no precin/CIN/cancer.
- `hpv.SexualNetwork(ss.SexualNetwork)` — heterosexual partnership formation
  ported from v2's `population.create_edgelist`. One class instantiated per
  partnership layer (`m`/`c`/`o`).
- `hpv.Sim(ss.Sim)` — convenience wrapper providing v2-compatible
  `hpv.Sim(location=..., genotype=...)` API.
- `hpv.data.load_country()` — thin adapter that wraps v2's existing data
  loaders into Starsim-shaped DataFrames.
- New M01 1-genotype Nigeria HPV16 regression anchor + baseline (sibling to
  M00's 4-genotype anchor).
- Test layering: unit tests, integration smoke, equivalence vs. v2.

**Explicitly out of scope (deferred to later milestones):**
- Multi-genotype + cross-immunity (M03 in renumbered scheme — see Milestone
  Housekeeping below)
- Natural history beyond simple infection→clearance (precin/CIN/cancer; M02)
- Population scaling (`pop_scale`/`total_pop`) (M02)
- Age-specific migration (M02)
- Multi-resolution (`scale`/`level0`/`level1`/`cluster`) — out of scope
  indefinitely; revisit only if needed.
- Vaccination, screening, treatment (M04+)
- HIV–HPV co-infection / MSM networks (M07+)
- `age_results` and other analyzers (M08)
- Plotting beyond what is needed for the demo prevalence trajectory.

---

## Architecture

### Runtime composition

```
hpv.Sim(ss.Sim)                                 ← HPVsim-owned, thin wrapper
├── people = ss.People(n_agents, age_data=df)                ← stock
├── networks = [
│       hpv.SexualNetwork(layer='m', ...),                   ← HPVsim-owned
│       hpv.SexualNetwork(layer='c', ...),                   ← HPVsim-owned
│   ]
├── diseases = [hpv.HPV(genotype='hpv16')]                   ← HPVsim-owned
├── demographics = [
│       ss.Pregnancy(fertility_rate=df),                     ← stock
│       ss.Deaths(death_rate=df),                            ← stock
│   ]
├── connectors = []                            ← none in M01
├── interventions = []                         ← none in M01
└── analyzers = []                             ← none in M01

hpv.data.load_country(location)                ← HPVsim-owned, adapter
   returns dict(age_data, fertility, death_rate, network_pars)
```

### Package layout after M01

```
hpvsim/                          ← in-place v3 package
├── __init__.py                  ← rewritten: exports HPV, SexualNetwork, Sim, data
├── hpv.py                       ← new: HPV(ss.Infection)
├── network.py                   ← new: SexualNetwork(ss.SexualNetwork)
├── sim.py                       ← rewritten: thin Sim(ss.Sim) wrapper
├── parameters.py                ← kept active (used by hpv.data.load_country)
├── version.py                   ← kept
├── settings.py                  ← kept (options used by parameters/defaults)
├── defaults.py                  ← kept (default_int/float, datadir)
├── misc.py                      ← kept (helpers used by parameters)
├── utils.py                     ← kept (numerical helpers — choose_w etc.)
├── data/                        ← kept; adds new country.py adapter
│   ├── __init__.py              ← re-exports load_country
│   ├── country.py               ← new: load_country() function
│   ├── loaders.py               ← v2 loaders (kept; wrapped by adapter)
│   ├── downloaders.py           ← kept (used by loaders)
│   └── files/                   ← v2 data files (kept)
├── regression/                  ← v2 pars JSONs (kept as artifacts)
└── _v2_legacy/                  ← NEW: quarantined v2 modules
    ├── __init__.py              ← empty; "do not import from active code"
    ├── analysis.py              ← moved from hpvsim/
    ├── base.py                  ← moved
    ├── calibration.py           ← moved
    ├── hiv.py                   ← moved
    ├── immunity.py              ← moved
    ├── interventions.py         ← moved
    ├── people.py                ← moved (god-object replaced by ss.People)
    ├── plotting.py              ← moved
    ├── population.py            ← moved (network logic ported to network.py)
    ├── run.py                   ← moved (MultiSim)
    └── sim.py                   ← moved (1395-line v2 Sim, replaced wholesale)

tests/                           ← active tests for current milestone surface
├── test_hpv.py                  ← new
├── test_network.py              ← new
├── test_sim.py                  ← new
├── test_data.py                 ← new
├── test_partnership_equivalence.py  ← new (M01 acceptance gate)
├── test_regression.py           ← modified: M00 4-gt smoke skipped until M03
├── regression/                  ← M00 harness; M01 anchor added
│   ├── anchor.py                ← M00 4-gt; harness still imported but smoke skipped
│   ├── anchor_hpv16.py          ← new: M01 1-gt anchor
│   ├── demo_anchor_hpv16.py     ← new: M01 demo (plots prevalence)
│   ├── compare.py               ← kept
│   ├── baseline.py              ← kept
│   └── README.md                ← updated with M01 baseline procedure
└── _legacy/                     ← NEW: quarantined v2 tests
    └── (all v2 tests that exercise removed APIs)
```

The quarantines (`hpvsim/_v2_legacy/`, `tests/_legacy/`) are **never imported
by active code** — the linting rule is: if you find yourself reaching into
either, lift the necessary v2 code into the active package as part of the
relevant milestone instead. Quarantines exist purely as a porting-reference
convenience; M10 deletes both wholesale.

### Why this shape

- **Rotasim-style multi-genotype design (settled during brainstorm).**
  Future-state: one `hpv.HPV(ss.Infection)` instance per genotype + one
  `hpv.CrossImmunity(ss.Connector)` for cross-protection. M01 builds the
  single-genotype case as a one-of-many; M03 adds the rest. Klebsim's
  `Connector + Type` multi-inheritance pattern was rejected because HPV has no
  reassortment-style shared-state-per-host concern that would justify the extra
  complexity.

- **`ss.SexualNetwork` parent class.** Provides `meta.dur` (partnership
  duration), `meta.acts` (per-pair sex acts), `debut`, `participant`,
  `active(people)`, `available(people, sex)`, `net_beta(...)`, `step()`,
  `end_pairs()` out of the box. We only own `add_pairs()` and the per-layer pars.

- **One class, two instances for layers.** v2's `population.create_edgelist`
  is a single algorithm parameterized by layer (`lno`); the m/c layers differ
  only in distributions and matrices, not in behavior. One `hpv.SexualNetwork`
  class, two instances. (An earlier draft of this spec assumed three layers
  including a one-off layer 'o' based on a misleading comment in v2's
  `parameters.py`; verification of v2 source confirmed only `m` and `c` are
  defined in any v2 network configuration.)

- **No HPVsim subclass of `ss.People`.** All HPV state lives on `hpv.HPV` via
  `define_states`; Starsim auto-aggregates module states onto People. Network
  state (`debut`, `participant`) lives on `hpv.SexualNetwork`. Multi-resolution
  state (`scale`, `level0`/`1`, `cluster`) is deferred indefinitely. Stock
  `ss.People(n_agents, age_data=df)` is sufficient.

- **Stock demographics.** `ss.Pregnancy` and `ss.Deaths` accept pandas
  DataFrames in exactly the shape v2's data files already provide; subclassing
  is unnecessary.

### Performance posture

Per-disease network walk in `ss.Infection.infect()` runs once per disease per
timestep, not vectorized across diseases. For M01's single-genotype case this
is identical to v2's runtime. M03's multi-genotype expansion will pay an
estimated ~2-3x cost vs. v2's pre-vectorized 2D approach; this is the
unavoidable cost of idiomatic Starsim integration. No performance budget for
M01 — profile after M03 if needed.

---

## Components

### `hpv.HPV(ss.Infection)`

Single HPV genotype as a Starsim `Infection`. Transmission-only in M01;
M02 adds natural history (precin, CIN, cancer).

```python
class HPV(ss.Infection):
    def __init__(self, genotype='hpv16', pars=None, **kwargs):
        self.genotype = genotype
        if 'name' not in kwargs:
            kwargs['name'] = genotype
        super().__init__()
        self.define_pars(
            init_prev=ss.bernoulli(p=0.05),     # placeholder, tuned to v2 1-gt anchor
            beta=ss.peryear(0.5),               # placeholder, tuned to v2 1-gt anchor
            dur_inf=ss.lognorm_ex(mean=ss.years(2.0)),  # SIS clearance
        )
        self.update_pars(pars=pars, **kwargs)
        # ss.Infection provides: susceptible, infected, rel_sus, rel_trans, ti_infected
        self.define_states(
            ss.FloatArr('ti_clearance', label='Time of natural clearance'),
        )

    def set_prognoses(self, uids, sources=None):
        super().set_prognoses(uids, sources)
        ti = self.ti
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.ti_clearance[uids] = ti + self.pars.dur_inf.rvs(uids)

    def step_state(self):
        clearing = (self.infected & (self.ti_clearance <= self.ti)).uids
        self.infected[clearing] = False
        self.susceptible[clearing] = True   # SIS — re-infection allowed
```

**Notes:**
- The `genotype` attribute is the duck-type marker M03's
  `hpv.CrossImmunity(ss.Connector)` will use to discover HPV diseases (mirrors
  rotasim's `hasattr(disease, 'G')` discovery).
- M01 uses SIS (clearance returns to susceptible). M02's natural history will
  introduce post-infection states (immune, recovered, precin, etc.) and modify
  this transition.
- Default values for `init_prev`, `beta`, `dur_inf` are placeholders. The
  pinned M01 anchor will use values sourced from v2's
  `get_genotype_pars('hpv16')` to keep the v2 1-gt baseline and v3 M01 sim
  nominally identical.

### `hpv.SexualNetwork(ss.SexualNetwork)`

Lift-and-shift of v2's three-layer sexual network. One class, instantiated
per layer. Inherits scaffolding from `ss.SexualNetwork` (which itself extends
`ss.DynamicNetwork` → `ss.Network`).

```python
class SexualNetwork(ss.SexualNetwork):
    def __init__(self, layer='m', pars=None, **kwargs):
        super().__init__()
        self.layer = layer
        self.define_pars(
            partners=...,           # per-agent desired partner-count distribution
            mixing=...,             # age-mixing matrix
            layer_probs=...,        # participation rates by age and sex
            cross_layer=...,        # cross-layer concurrency proportion (f, m)
            duration=...,           # partnership duration distribution
            acts=...,               # per-pair acts per timestep
        )
        self.update_pars(pars=pars, **kwargs)
        # ss.SexualNetwork already provides: debut (FloatArr), participant
        # (BoolArr), meta.dur, meta.acts on edges.

    def add_pairs(self):
        # Port of v2 hpvsim/population.create_edgelist for one layer:
        # 1. Compute n_partners_elsewhere by iterating sibling hpv.SexualNetwork
        #    instances (filtered by isinstance) — see below.
        # 2. Determine eligible females and males for this layer (active,
        #    underpartnered, concurrency-eligible per cross_layer pars).
        # 3. Sample participation by age and sex.
        # 4. For each cluster, for each female age bin: weight available males
        #    by mixing matrix and select via choose_w.
        # 5. Sample partnership duration from self.pars.duration.
        # 6. self.append(p1=f, p2=m, beta=..., dur=..., acts=...)

    def _n_partners_elsewhere(self):
        """Count how many partnerships each agent has in OTHER hpv.SexualNetwork
        layers, used for cross-layer concurrency eligibility. Returns an int
        array of length n_agents; downstream eligibility check is `n > 0`.
        Returns all zeros if no sibling hpv.SexualNetwork instances exist
        (e.g., a one-layer-only sim)."""
        n = np.zeros(len(self.sim.people), dtype=int)
        for other in self.sim.networks():
            if other is self:
                continue
            if not isinstance(other, SexualNetwork):
                continue
            n[other.edges.p1] += 1
            n[other.edges.p2] += 1
        return n
```

**Notes:**
- Three instances at sim construction:
  ```python
  networks = [
      hpv.SexualNetwork(layer='m', pars=marital_pars),
      hpv.SexualNetwork(layer='c', pars=casual_pars),
  ]
  ```
- The `isinstance(other, hpv.SexualNetwork)` filter in `_n_partners_elsewhere`
  defines the cross-layer concurrency group. A future MSM or maternal network
  using a different class is automatically excluded; a future "sex-work" layer
  authored as another `hpv.SexualNetwork` is automatically included.
- The disease's `beta` becomes a per-network dict
  (`{'m': beta_m, 'c': beta_c, 'o': beta_o}`); `ss.Infection.validate_beta`
  already supports this shape.
- Per-agent lifetime-partner-counter analyzer state (`partners_m`, `_c`, `_o`
  in v2) is intentionally deferred to the analyzers milestone (M08).

### `hpv.Sim(ss.Sim)`

Convenience wrapper providing the v2-compatible
`hpv.Sim(location=..., genotype=...)` API.

```python
class Sim(ss.Sim):
    def __init__(self, location='nigeria', genotype='hpv16',
                 n_agents=10_000, start=1990, stop=2060, dt=0.5,
                 pars=None, **kwargs):
        country = hpv.data.load_country(location)
        people = ss.People(n_agents, age_data=country['age_data'])
        diseases = kwargs.pop('diseases', None) or [HPV(genotype=genotype)]
        networks = kwargs.pop('networks', None) or [
            SexualNetwork(layer=k, pars=country['network_pars'][k])
            for k in ('m', 'c')
        ]
        demographics = kwargs.pop('demographics', None) or [
            ss.Pregnancy(fertility_rate=country['fertility']),
            ss.Deaths(death_rate=country['death_rate']),
        ]
        super().__init__(
            start=ss.years(start), stop=ss.years(stop), dt=ss.years(dt),
            people=people, diseases=diseases, networks=networks,
            demographics=demographics, pars=pars, **kwargs,
        )
```

**Notes:**
- `genotype=` is singular for M01. M03 changes the signature to
  `genotypes=[...]` and adds the cross-immunity connector.
- All defaults are overridable via kwargs; passing custom `diseases=`,
  `networks=`, or `demographics=` short-circuits the M01 conveniences (useful
  for tests).

### `hpv.data.load_country(location)`

Thin adapter wrapping v2's existing data loaders. All values come from v2's
existing data files and helper functions (which stay active in `hpvsim/`,
not quarantined); the adapter just reshapes them. Lives at
`hpvsim/data/country.py` and is re-exported from `hpvsim/data/__init__.py`
so the call site is `hpv.data.load_country(...)`.

```python
def load_country(location):
    """Return Starsim-shaped data for a country.
    Wraps hpvsim.data and hpvsim.parameters loaders that already exist for v2.
    """
    return dict(
        age_data=...,        # pd.DataFrame: age, value (population pyramid)
        fertility=...,       # pd.DataFrame: Time, AgeGrp, ASFR
        death_rate=...,      # pd.DataFrame: Year, AgeGrp, Sex, Rate
        network_pars=dict(
            m=dict(partners=..., mixing=..., layer_probs=..., cross_layer=...,
                   duration=..., acts=...),
            c=dict(...),
            o=dict(...),
        ),
    )
```

**Source mapping (v2 → adapter):**
- `age_data` ← `hpvsim.data.loaders.get_age_distribution(location)` reshaped
  to `[age, value]` columns expected by `ss.People(age_data=...)`.
- `fertility` ← `hpvsim.parameters.get_births_deaths(location)['birth_rates']`
  reshaped to `[Time, AgeGrp, ASFR]` columns expected by `ss.Pregnancy`.
- `death_rate` ← `hpvsim.parameters.get_births_deaths(location)['death_rates']`
  reshaped to `[Year, AgeGrp, Sex, Rate]` columns expected by `ss.Deaths`.
- `network_pars[layer]` ← combination of `hpvsim.parameters.get_mixing()`,
  `pars['layer_probs']`, `pars['partners']`, `pars['dur_pship']`, `pars['acts']`,
  and `pars['cross_layer']` for the named layer.

**Notes:**
- The `...` placeholders in the return dict above are illustrative — the actual
  shapes are pinned by the consuming Starsim modules (`ss.People`,
  `ss.Pregnancy`, `ss.Deaths`) and validated by `hpv.data.load_country` before
  return.
- Network-specific params are grouped per layer under `network_pars` so the
  three `hpv.SexualNetwork` instances each receive a single `pars=` payload.

---

## Data flow

### Sim construction

```
hpv.Sim(location='nigeria', genotype='hpv16', n_agents=10_000,
         start=1990, stop=2060, dt=0.5)
  ↓
hpv.data.load_country('nigeria')              # one read
  ↓ produces dict(age_data, fertility, death_rate, network_pars)
  ↓
ss.People(n_agents, age_data=...)
diseases    = [hpv.HPV(genotype='hpv16')]
networks    = [hpv.SexualNetwork(layer='m', pars=...),
               hpv.SexualNetwork(layer='c', pars=...)]
demographics = [ss.Pregnancy(...), ss.Deaths(...)]
  ↓
super().__init__(...)
  ↓
sim.init()    # Starsim wires modules, allocates state, calls each
              # module's init_pre/init_post
```

### Per-timestep loop

Stock Starsim loop order (from `loop.py`):

```
1. demographics.step()    — ss.Pregnancy adds new agents, ss.Deaths kills
2. diseases.step_state()  — hpv.HPV: infected → susceptible (clearance)
3. connectors.step()      — empty in M01
4. networks.step()        — for each hpv.SexualNetwork (m, c in order):
                                end_pairs()   [stock: duration & death]
                                add_pairs()   [our override: port v2 logic;
                                               iterates sibling hpv.SexualNetwork
                                               instances for cross-layer concurrency]
5. interventions.step()   — empty in M01
6. diseases.step()        — hpv.HPV.step → infect():
                                walk each network's edges, sample
                                β·rel_trans·rel_sus, set_outcomes →
                                set_prognoses (sets infected, schedules clearance)
7. results updates        — stock: n_infected, prevalence, etc.
8. analyzers.step()       — empty in M01
```

### Regression comparison flow

```
# Local-only baseline generation (one-time, before M01 PR is opened):
generate_v2_baseline_hpv16:    Run v2 hpvsim with genotypes=['hpv16'] and
                               same scenario pars as M01 anchor; serialize
                               summary → tests/regression_baselines/anchor_hpv16.json
                               (gitignored, same convention as M00).

# CI / dev-time comparison (every PR):
tests/regression/anchor_hpv16.py    Pinned 1-gt M01 anchor harness.
                                    run_and_summarize() returns same shape as
                                    M00's anchor: short_summary dict + total_pop.

tests/regression/compare.py         Already supports --baseline path. Add
                                    --anchor flag to choose between
                                    anchor.py (M00 4-gt) and
                                    anchor_hpv16.py (M01 1-gt).

tests/test_regression.py            Add test_anchor_hpv16_runs() smoke test
                                    alongside the existing test_anchor_runs().

CI workflow                         No changes; existing smoke-check no-baseline
                                    mode covers M01.
```

### Partnership-equivalence flow

```
test_partnership_equivalence:    Run hpv.Sim through burnin (1990-2010, 20y)
                                  plus observation window (2010-2015, 5y of
                                  steady-state pairs). Capture per-layer:
                                    - age-mixing matrix (binned by 5y)
                                    - concurrency distribution (n_concurrent
                                      partners per agent)
                                    - partnership duration distribution
                                  Compare against same-shape v2 outputs stored
                                  as JSON fixtures in
                                  tests/regression_baselines/partnership_v2.json
                                  (gitignored).
                                  Pass criteria: KS-test p > 0.01 per layer per
                                  metric; mixing-matrix bin-wise relative diff
                                  < 15%.
```

---

## Acceptance gates for M01

1. **Sim runs end-to-end** without errors on the M01 anchor scenario (Nigeria
   1990-2060, dt=0.5, n_agents=10k, HPV16 only, seed 0).
2. **Partnership patterns match v2** within tolerance: age-mixing matrix per
   layer (bin-wise relative diff < 15%), concurrency distribution
   (KS p > 0.01), partnership duration distribution (KS p > 0.01).
3. **HPV prevalence trajectory matches v2's 1-gt run** within informational
   threshold (10% relative drift on summary stats).
4. **Demo notebook/script** plots the M01 anchor's aggregate HPV prevalence
   over time.

---

## Testing strategy

### Tier 1 — unit tests (fast, deterministic)

Run on every CI push; subseconds each.

- `test_hpv_module.py`
  - `test_set_prognoses` — flips states, sets `ti_infected`/`ti_clearance` correctly.
  - `test_step_state_clearance` — agents past `ti_clearance` return to susceptible.
  - `test_genotype_attribute` — instance has `genotype` attribute.
  - `test_init_prev_seeds_correct_count` — Bernoulli seeding tolerance.
- `test_hpv_network.py`
  - `test_single_layer_pairs_form_and_dissolve` — pairs form, then dissolve past `dur`.
  - `test_cross_layer_concurrency_filter` — `isinstance` filter excludes non-`hpv.SexualNetwork` instances.
  - `test_concurrency_eligibility` — partnered-elsewhere agents enter another layer with prob `cross_layer`.
  - `test_age_mixing_assortativity` — sampled pairs reproduce input mixing matrix within sampling tolerance.
- `test_hpv_data.py`
  - `test_load_country_returns_starsim_shapes` — DataFrame shapes match `ss.Pregnancy`/`ss.Deaths` expectations.
  - `test_age_data_pyramid_shape` — monotonic by age, sums sensibly, both sexes.

### Tier 2 — integration smoke (fast, no comparison)

Runs in CI without baseline files. ~10 sec.

- `tests/test_regression.py::test_anchor_hpv16_runs` — sibling to existing `test_anchor_runs`. Constructs M01 anchor, calls `run_and_summarize()`, asserts summary keys present and finite/positive.

### Tier 3 — equivalence vs. v2 (requires local baselines)

Run by developers manually and via the existing `compare.py` CLI. Gated on baseline file present (gitignored).

- `test_anchor_hpv16_drift` — when `tests/regression_baselines/anchor_hpv16.json` exists. Uses `compute_drift()`. Informational 10% threshold; non-failing.
- `test_partnership_equivalence` — when `tests/regression_baselines/partnership_v2.json` exists. KS-tests + mixing-matrix bin-wise diffs. **This is the M01 acceptance gate.**

### Test-data conventions (reused from M00)

- All baseline JSONs live in `tests/regression_baselines/` (gitignored).
- All anchor harnesses live in `tests/regression/` (versioned).
- 1-gt baselines are generated locally by running v2 hpvsim with
  `genotypes=['hpv16']` against the same anchor pars; documented in
  `tests/regression/README.md` alongside the existing M00 instructions.

---

## Error handling

Stock Starsim error semantics for the most part. HPVsim-specific disciplines:

- **`hpv.data.load_country(location)`** — `ValueError` with available
  locations enumerated if `location` is unknown.
- **`hpv.HPV(genotype=...)`** — validate against known list (`hpv16`, `hpv18`,
  `hi5`, `ohr`); reject unknowns at construction.
- **`hpv.SexualNetwork(layer=...)`** — validate `layer` against `m`/`c`/`o`;
  reject unknowns.
- **Cross-layer iteration robustness** — `_n_partners_elsewhere` returns zeros
  if no sibling `hpv.SexualNetwork` instances exist (e.g., one-layer-only sim).
- **Empty population early-step** — handled by stock `ss.SexualNetwork.available()`.

No defensive validation beyond these.

---

## Open items to pin during implementation

These are not architectural decisions — values to be filled in during plan
execution, called out so they don't slip:

1. **M01 anchor's exact pars.** Mirror M00's pinning discipline:
   `location='nigeria'`, `genotype='hpv16'`, `n_agents=10_000`, `start=1990`,
   `stop=2060`, `dt=0.5`, `burnin=20`, `seed=0`. `init_prev`, `beta`, and
   `dur_inf` defaults for HPV16 to be sourced from v2's
   `get_genotype_pars('hpv16')` so the v2 1-gt baseline and the v3 M01 sim use
   the same nominal values.
2. **Tolerance thresholds.** Anchor drift: 10% relative (informational, mirrors
   M00). Partnership equivalence: KS-test p > 0.01 per layer per metric;
   mixing-matrix bin-wise relative diff < 15%. Both will be revisited on first
   run; expect to adjust once we see actual variability.
3. **v2 1-gt baseline generation procedure.** A documented one-shot recipe in
   `tests/regression/README.md`: clone v2 main, install, run v2 with
   `genotypes=['hpv16']` against the anchor pars, capture summary, write JSON
   to `tests/regression_baselines/anchor_hpv16.json`.

---

## Milestone housekeeping

This work happens on `v3.0-dev` *before* the `m01-basic-transmission-sim`
branch is cut, so M01's PR description references the correct downstream
milestone numbers.

### `MIGRATION_PLAN.md` updates

1. Rewrite "M2: Natural history parity" → 1-genotype focus (HPV16
   clearance/CIN/cancer; explicitly excludes multi-genotype + cross-immunity).
2. Insert new section "M3: Multi-genotype and cross-immunity" with sub-tasks:
   - Replicate `hpv.HPV` across `[16, 18, hi5, ohr]`.
   - Add `hpv.CrossImmunity(ss.Connector)`.
   - Add population scaling (`pop_scale` / `total_pop`).
   - Add age-specific migration (from v2's `people.check_migration`).
   - Add `age_results` analyzer at minimum scope.
   - Add regression tests vs. M00's 4-gt anchor.
3. Renumber existing M3-M9 → M4-M10 throughout the document (section headings,
   cross-references).
4. Update M1 sub-task list:
   - Drop "Port People as `hpv.People(ss.People)` using lift-and-shift"
     (multi-resolution deferred indefinitely; stock `ss.People` is sufficient).
   - Change "stored v2 baseline from M0" → "new 1-gt M1 baseline".
5. Add a paragraph to §"Implementation conventions" documenting the in-place
   replacement strategy and the `hpvsim/_v2_legacy/` + `tests/_legacy/`
   quarantine convention (active code never imports from quarantine; quarantine
   is porting reference; both deleted wholesale at M10).

### GitHub reconciliation

1. Rename milestones M03-M09 → M04-M10 (preserving leading zeros for
   alphabetical sorting, per the existing convention).
2. Create new milestone "M03: Multi-genotype and cross-immunity".
3. Move issues currently in (old) M02 that are multi-genotype-coupled
   (cross-immunity, pop_scale, age_results, migration) into the new M03.
4. Sub-task #98 ("Port People as hpv.People(ss.People) using lift-and-shift"):
   close with a comment pointing at the multi-scale-deferred decision.
5. Retitle sub-task #102 ("Tests: HPV prevalence trajectory vs. stored v2
   baseline from M00") → "Tests: HPV prevalence trajectory vs. v2 baseline
   (single-genotype HPV16, M01)".
6. Update tracking issue #95 to reflect new milestone count and structure.

### v2 quarantine moves (during M01 implementation, after housekeeping)

Bulk `git mv` v2 modules that aren't touched by M01 into `hpvsim/_v2_legacy/`,
and v2 tests that exercise removed APIs into `tests/_legacy/`. The complete
list lives in the implementation plan (single dedicated task) so the moves
land in one commit with full git history preserved.

Specific moves:

- `hpvsim/{analysis,base,calibration,hiv,immunity,interventions,people,plotting,population,run,sim}.py`
  → `hpvsim/_v2_legacy/`. Note that the original `hpvsim/sim.py` is preserved
  here before the new (small) `hpvsim/sim.py` is written in place.
- v2 tests under `tests/test_*.py` that import any of the above (or directly
  exercise v2 `Sim` / `People` / partnership APIs that no longer exist) →
  `tests/_legacy/`.
- `tests/regression/anchor.py` (M00 4-genotype anchor) **stays in place** —
  its baseline (`tests/regression_baselines/anchor.json`) is still valid as
  the M03 target. The smoke test `test_anchor_runs` is marked
  `@pytest.mark.skip(reason='Multi-genotype not yet ported; restored in M03')`.

Modules that **stay active** at the top level because the new M01 code
depends on them: `parameters.py`, `defaults.py`, `settings.py`, `misc.py`,
`utils.py`, `version.py`, `data/` (subdirectory). These are utilities or
data plumbing, not v2 entry points, and the new `hpv.data.load_country`
adapter wraps them.

---

## Decisions log (settled during brainstorming, 2026-04-28)

| # | Topic | Decision |
|---|---|---|
| Q1 | M01 disease-module gap | Add minimal `hpv.HPV(ss.Infection)` for HPV16 — transmission only, no nat hx. |
| Q2 | Multi-genotype architecture | Rotasim-style — `hpv.HPV` per genotype + `hpv.CrossImmunity(ss.Connector)`. Klebsim's multi-inheritance pattern rejected. |
| Q3 | Network parent class | `ss.SexualNetwork` (revised from initial `ss.Network` answer after audit). No STIsim dep. |
| Q4 | Demographics | Stock `ss.Pregnancy` + `ss.Deaths` + `ss.People(age_data=...)`, fed by thin `hpv.data.load_country()` adapter. No subclassing of demographic modules. |
| Q5 | M01 baseline | New 1-gt HPV16 Nigeria anchor + baseline (sibling to M00's 4-gt anchor). |
| Q6 | `hpv.People` | Don't build it for M01. Multi-scale logic deferred indefinitely. Stock `ss.People` sufficient. |
| Q7 | Multi-layer network shape | One `hpv.SexualNetwork` class, two instances (m/c). Layers differ only by parameterization. (An earlier draft assumed m/c/o; verification of v2 source confirmed only m and c are defined.) |
| Q8 | Cross-layer concurrency | Each network reads sibling `hpv.SexualNetwork` instances at `add_pairs` time; filtered by `isinstance(other, hpv.SexualNetwork)`. No connector. |
| — | Milestone scope | Split current M02 into M02 (1-gt nat hx) + new M03 (multi-gt + cross-immunity). Renumber M03-M09 → M04-M10. |
| — | Performance | No M01 budget. Estimated ~2-3x overhead in M03 vs. v2; profile after M03. |
| — | Replacement strategy | **In-place replacement** of `hpvsim/`. v2 modules untouched by M01 → `hpvsim/_v2_legacy/`; v2 tests that exercise removed APIs → `tests/_legacy/`. Both quarantines never imported by active code, exist as porting reference, deleted wholesale at M10. |

---

## References

- M00 design: `docs/superpowers/specs/2026-04-28-hpvsim-m0-foundation-design.md`
- Migration plan: `MIGRATION_PLAN.md` (canonical branching strategy lives in §"Branching and sync strategy")
- Starsim source for parent classes: `.venv/Lib/site-packages/starsim/diseases.py` (`Infection` line 109), `.venv/Lib/site-packages/starsim/networks.py` (`SexualNetwork` line 431)
- Multi-strain references: starsimhub/rotasim (`Rotavirus`, `RotaImmunityConnector`, `Sim`), starsimhub/klebsim (`KpnType`, `Kpn`)