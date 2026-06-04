# M08: HIV–HPV Co-infection — Design

**Date:** 2026-06-03
**Milestone:** M08 (HIV–HPV co-infection)
**Branch:** `m08-hiv-hpv-coinfection` (off `v3.0-dev`; PR targets `v3.0-dev`)
**Predecessor:** [M07 MultiSim and Scenarios](2026-05-27-m07-multisim-design.md) — merged to `v3.0-dev` via PR #113.
**Status:** Design approved; implementation not started.

## Dependencies and branch strategy

M08 branches directly off `v3.0-dev` after M07 (PR #113) merged. It does **not**
build on the unscheduled multiscale-agents work (`m07-multiscale-ledger`),
which is an orthogonal performance/scaling feature, explicitly "not a release
blocker" per `MIGRATION_PLAN.md`. M08 has no dependency on multiscale; coupling
a release-gating milestone to an optional feature would only create rebase pain.
Multiscale, if it lands, gets its own PR to `v3.0-dev`.

M08 depends on the full HPV natural-history + intervention stack from M02–M06
(genotype modules, `CrossImmunity`, vaccine/therapeutic-vaccine products) and on
the multi-seed parity-gate harness established in M03/M05/M07.

## Goal

Reproduce HIV-stratified HPV prevalence and cervical-cancer incidence on a
high-HIV-burden setting (Rwanda), using STIsim's **transmission-based** HIV
rather than v2's incidence-based HIV module (a settled scope decision —
`MIGRATION_PLAN.md` line 30). After M08, a user can:

1. Add HIV to any hpvsim configuration by passing `diseases=[hpv.HIV()]` (plus
   `interventions=[sti.ART(...)]`) — the Starsim-idiomatic way, with no bespoke
   constructor flag.
2. Observe HIV+ agents acquiring HPV more readily, progressing to cancer faster,
   and retaining less post-infection/vaccine immunity — the three CD4-stratified
   effects v2 modeled.
3. Read cancer/HPV outcomes stratified by HIV status (and by age × HIV status),
   including the HIV+:HIV− cancer rate ratio.

**Acceptance test:** reproduces `hpvsim_rwanda` outputs with overlapping
uncertainty intervals.

## Scope

**In scope:**

- `hpv.HIV` — thin subclass of `sti.HIV` (STIsim 1.5.0): inherits continuous
  CD4, ART-driven reconstitution, CD4-based mortality; adds a Rwanda
  `init_prev` loader and HPVsim-friendly network-beta targeting.
- `hpv_hiv_connector(ss.Connector)` — the three CD4-stratified HIV→HPV effects.
- `HIVStratifiedResults(ss.Analyzer)` — cross-disease stratified outputs.
- `hpv.Sim` disease-assembly refactor — type-partitioned `diseases=` so HIV is a
  first-class disease alongside genotype-built HPV modules; detection-based
  auto-wiring of the connector and analyzer.
- Rwanda HIV/ART data; incidence-driven HIV reproduction (`hpv.hiv_incidence_import`)
  + the ART coverage shortcut (`hpv.hiv_art`). See Phase 2 for the transmission-β
  deviation rationale.
- Dev-gate anchor + co-infection direction tests; tolerance-band HIV-stratified
  cancer parity (adult-restricted metric); `hpvsim_rwanda` release gate.

**Out of scope:**

- HIV testing / diagnosis cascade (`sti.HIVTest`, `sti.HIVDx`) — v2's Rwanda runs
  did not model a testing cascade; ART coverage was driven directly from a
  schedule.
- Sex-worker / client (FSW) risk-group network structure — STIsim's
  `structuredsexual` risk groups; not present in `hpv.SexualNetwork` and not used
  by v2's age/sex-incidence Rwanda HIV.
- Continuous-CD4 → effect mapping (deferred; see "CD4 representation").
- Dynamic re-evaluation of HPV prognoses on CD4-stratum crossing, *unless* the
  static-prognoses default misses the Rwanda release gate (see "Re-evaluation").

## Approach: phased path to transmission-based HIV (Approach C)

The end state is fixed by scope (transmission-based HIV). M08 reaches it in two
phases to decouple the two independent hard problems — *is the co-infection
coupling correct?* and *is the HIV epidemic calibrated?* — so that when Rwanda
numbers are off, the failing half is identifiable.

### Phase 1 — co-infection mechanics

Build `hpv.HIV`, `hpv_hiv_connector`, `HIVStratifiedResults`, and the Sim
assembly refactor. Validate against a hand-tuned / imposed-prevalence HIV with
unit tests and a small anchor. **Exit criteria:** mechanics correct (each effect
applied at its composition-safe site, directional beta oriented correctly), HIV+
agents show elevated HPV acquisition and cancer in the expected direction, and
the no-HIV byte-identity guard is green.

### Phase 2 — HIV epidemic + parity

**Data audit (2026-06-04).** The Rwanda inputs already exist in the sibling
analysis repo `hpvsim_v23_validation/hpvsim_rwanda` (it is a published v2.3
baseline). Available: ART coverage by age/sex/year
(`rwanda_art_coverage_by_age_{females,males}.csv`, 2004–2030), HIV
incidence/mortality by age, aggregate HIV-prevalence targets (`rwanda_data.csv`),
HIV-stratified cancer targets (2017: ~13.1/100k HIV− vs ~33/100k HIV+), and
**cached v2.2.6/v2.3.0 baselines** under `results/`. The one gap — HIV prevalence
*by age* for `init_prev` — is **derived** from the v2 baseline's
`hiv_prevalence_by_age` result at the v3 start year (no file exists for it).

Copy the needed Rwanda ART/HIV CSVs into `hpvsim/data/`; add `load_hiv('rwanda')`;
calibrate `sti.HIV` beta to the Rwanda HIV prevalence trajectory. The v2 baseline
is reproduced by running the frozen v2.3 install (`hpvsim_v23_frozen`) on the
Rwanda repo — or read directly from the cached baseline. Run the dev-gate parity
tests, then the `hpvsim_rwanda` release gate. Assess whether static prognoses
suffice; add dynamic re-evaluation only if HIV+ cancers miss the gate.

**ART = coverage shortcut (settled 2026-06-04).** v2/Rwanda has no testing
cascade — it assigns ART directly to hit the age/sex/year coverage curve, and the
repo has no testing-coverage data. So M08 marks a data-matched fraction of HIV+
agents on-ART per that curve, rather than using STIsim's `HIVTest → ART` cascade
(which would diverge from the baseline and have nothing to calibrate against).
`sti.HIVTest` stays out of scope.

**HIV epidemic = incidence-driven importer (settled 2026-06-04, supersedes the
transmission-β calibration above).** A trial of transmission-based HIV on Rwanda
(`beta_m2f=0.004`) reproduced the *level* early on but overshot the trajectory
*shape* — peak 7.7% @ 2009 vs the target 4.9% @ 1998, with no post-1998 decline.
A single constant β cannot fix this (Rwanda's decline began ~1998, before ART
scale-up, so it is largely behavior-change-driven); matching it would require
calibrating a time-varying β.

Instead M08 drives HIV with an **incidence importer**, `hpv.hiv_incidence_import`
(an `ss.Intervention`): with HIV `beta_m2f=0` and `init_prev=0`, it imposes the
Rwanda incidence curve (`hiv_incidence_rwanda.csv`) by selecting living
HIV-negative susceptibles at their per-(year, sex, age) force of infection
(`p = 1 - exp(-rate·dt)`) and calling `sti.HIV.set_prognoses(uids)` — which flips
them to infected and wires the full CD4 trajectory. STIsim's CD4 decline, the ART
coverage shortcut, and HIV mortality all run unchanged, as does the HIV→HPV
connector. This is exactly v2's mechanism (v2 was incidence-based), so the
prevalence trajectory matches Rwanda **by construction** and the v2↔v3 parity
comparison is apples-to-apples (HIV trajectory matches by design → any
cancer-by-HIV-status difference isolates the HPV-side port, which is what M08
tests).

*Validation:* adult (15–49) HIV prevalence tracks Rwanda in both shape and level
— peak 5.0% @ 1996 vs target 4.9% @ 1998; 1990–2005 within ~0.5 pp. A modest
tail residual remains (model declines faster post-2010: 1.8% vs 3.0% by 2020).
**The parity metric MUST use the adult (15–49) denominator** — STIsim's all-ages
`prevalence` understates ~2× (an M05-class plot-metric artifact, not a model
error).

**Deviation from the migration plan — flag for Robyn at PR review.** The plan
mandates *transmission-based* HIV (drop v2's incidence-based HIV). The importer
uses STIsim's HIV *disease model* (CD4/ART/mortality/states — not v2's custom
code) but drives infections by imposed incidence rather than simulated network
transmission, so it half-honors the mandate. Rationale: it reproduces Rwanda
without intractable time-varying-β calibration and isolates the M08 deliverable
(the HPV-side co-infection effects). The transmission capability is retained
(`beta_m2f` is still a constructor arg, set to 0 here); transmission-β
calibration — needed for HIV counterfactuals/projections — is documented as
future work, not a blocker for M08's Rwanda reproduction.

**Fallback:** if Phase 2 HIV calibration cannot converge within the milestone
budget, fall back to data-pinned HIV prevalence (Approach B) as an interim,
flagged with a tracking issue to return to transmission-based before v3.0.0.

## Architecture

New active module `hpvsim/hiv.py` (mirroring how `cross_genotype.py` houses the
cross-genotype connector + its companion analyzer) holds all three new
components. The `hpv.Sim` constructor is refactored to assemble HIV as a disease.
HIV transmits over the existing `hpv.SexualNetwork` — no new network.

### Component 1 — `hpv.HIV(sti.HIV)`

Thin subclass. Inherits STIsim's continuous CD4 model (`cd4`, `cd4_nadir`,
`cd4_potential`, `on_art`, logistic ART reconstitution, CD4-based mortality,
acute/latent/falling transmissibility phases) unchanged. The subclass exists only
to:

- Load a Rwanda initial HIV prevalence curve by age (`hiv_init_prev.csv`) through
  the existing `hpv.data` adapter pattern.
- Target its transmission beta at `hpv.SexualNetwork` by name. STIsim's
  `BaseSTI.validate_beta` (`stisim/diseases/sti.py:164`) hard-codes the keys
  `'structuredsexual'`/`'maternal'`/`'msm'` to auto-apply `beta_m2f`/
  `rel_beta_f2m`; our network isn't named that, so `hpv.HIV` sets its per-network
  beta dict explicitly, keyed to `SexualNetwork`'s name, with **directional**
  entries `[p1→p2, p2→p1]` oriented for our convention (**p1 = female,
  p2 = male**, per `network.py:232`) so male→female and female→male HIV betas
  land on the correct direction.

ART is stock `sti.ART`, supplied by the user as an intervention (coverage is
scenario-specific; not auto-added).

### Component 2 — `hpv_hiv_connector(ss.Connector)`

The heart of M08. Follows the `CrossImmunity` pattern (discover modules in
`init_pre`, mutate fields in `step`). Each step it reads `hiv.cd4`/`on_art`, bins
agents into discrete CD4 strata, and computes three per-agent factor arrays —
`hiv_rel_sus`, `hiv_rel_sev`, `hiv_rel_imm` — all defaulting to `1.0` for HIV−
agents.

**CD4 binning.** Discrete strata `lt200` / `gt200` (HIV− = factor 1), matching
v2's source-of-truth (`_v2_legacy/hiv.py:29–44`). The actual multipliers are read
from the v2 pars at implementation rather than hard-coded; v2's values are
approximately:

| Effect | lt200 | gt200 | Meaning |
|---|---|---|---|
| `rel_sus` | ~2.2 | ~2.2 | HIV+ acquire HPV more readily |
| `rel_sev` | ~1.5 | ~1.2 | faster/worse CIN→cancer progression |
| `rel_imm` | ~0.36 | ~0.76 | reduced post-infection/vaccine immunity (multiplier on immunity conferred) |

The binning is isolated in a single `_cd4_stratum(cd4)` helper — the one
"discrete now" seam — so a continuous-CD4 successor swaps it without touching
consumers.

**Where each effect is applied** (the central correctness constraint).
`CrossImmunity` runs first each step and *overwrites* each module's `rel_sus` and
sets `sev_imm` from the cross-genotype matrix. So effects cannot simply mutate
stored state and survive; each is applied at the point that composes correctly:

| Effect | Applied where | Why there |
|---|---|---|
| `rel_sus` (acquisition ↑) | The connector's own `step()`, running **after** `CrossImmunity`, multiplies each module's final `rel_sus`. Only susceptible agents' acquisition is affected — no re-evaluation needed. | `CrossImmunity` overwrites `rel_sus`; the HIV factor must multiply *after* it. |
| `rel_sev` (progression ↑) | Read inside `HPV.set_prognoses` (`hpv.py:241`), combined with the existing cross-genotype `rel_sev` when sampling `dur_precin`/`dur_cin` and computing P(CIN)/P(cancer). | Severity is decided at prognosis time; this is the natural hook. |
| `rel_imm` (immunity ↓) | Applied **at the moment immunity is conferred** — scaling the `nab_imm`/`cell_imm` increment in `HPV.step_state` clearance (`hpv.py:444`) and the `vax_imm`/`txvx_imm` increment in the vaccine products (`products.py`). | Reduces the *stored* value, so `CrossImmunity` then consumes already-reduced immunity with no ordering fight. |

Consumers read the connector's factor arrays; the connector never tries to win an
overwrite war with `CrossImmunity`. All three read sites are gated to **no-op when
no HIV module is present** (factor stays `1.0`) — this is what keeps non-HIV sims
byte-identical.

**Connector ordering** is explicit: the auto-connector list is
`[CrossImmunity, (seeder), hpv_hiv_connector]`, with `hpv_hiv_connector` last so
its `rel_sus` multiply lands after `CrossImmunity`'s overwrite.

**Re-evaluation of already-infected agents (open design point, static default).**
If an agent acquires HPV while healthy (`gt200`, `rel_sev ≈ 1.2`) and later
progresses to AIDS (`lt200`, `rel_sev ≈ 1.5`), v2 *re-sampled* their HPV
prognoses on the stratum crossing.

- **Phase 1 default — static:** trajectory is fixed at infection using the
  stratum in effect at that moment. CRN-clean, simpler, no mid-trajectory
  re-sampling. `rel_sus` (acquisition-only) and `rel_imm` (applied at each
  immunity-gain event) need no re-evaluation regardless; only `rel_sev` is
  affected.
- **Dynamic (v2-faithful, conditional):** re-sample `dur_cin`/P(cancer) when CD4
  crosses 200. Added in Phase 2 **only if** the static version misses overlapping
  intervals on `hpvsim_rwanda`, with the gap documented.

### Component 3 — `HIVStratifiedResults(ss.Analyzer)`

Modeled on `HPVTotal` (`cross_genotype.py`). Records, split by HIV status:

- `cancer_incidence_with_hiv` / `_no_hiv` and the HIV+:HIV− rate ratio.
- `cancer_incidence_by_age_with_hiv` / `_no_hiv`.
- `hpv_prevalence_with_hiv` / `_no_hiv` (and by age).

HIV's own epidemic results (prevalence, incidence, ART coverage, CD4
distribution) come from stock `sti.HIV`/`sti.ART` — the analyzer adds only the
cross-disease stratification HPV needs. Present only when HIV is.

### Component 4 — `hpv.Sim` disease-assembly refactor

Today `diseases=` and `genotypes=` are mutually exclusive (`sim.py:91–94`), and
passing `diseases=` bypasses the entire genotype build (exclusive seeder,
`init_hpv_dist` validation). The refactor **partitions `diseases=` by type** so
HIV is mergeable while the heavily-used HPV-instance override path is preserved
exactly:

```python
user_diseases  = kwargs.pop('diseases', None) or []
hpv_instances  = [d for d in user_diseases if isinstance(d, HPV)]
other_diseases = [d for d in user_diseases if not isinstance(d, HPV)]   # e.g. HIV

if hpv_instances and genotypes is not None:
    raise ValueError('Specify HPV via genotypes= or HPV instances in diseases=, not both.')

if hpv_instances:
    hpv_diseases = hpv_instances          # existing override path — seeder NOT wired (unchanged)
else:
    hpv_diseases = [...auto-build from genotypes + _ExclusiveSeeder...]   # unchanged

diseases = hpv_diseases + other_diseases  # merge HPV modules with HIV/other
```

The mutual-exclusivity guard narrows from "diseases vs genotypes" to
"**HPV-instances** vs genotypes": you cannot specify the HPV set two ways, but
HIV-in-`diseases=` alongside `genotypes=` is now legal.

Connector + analyzer auto-wire on detection, mirroring how `CrossImmunity` is
auto-added for genotypes:

```python
auto_connectors = [CrossImmunity()] + ([seeder] if seeder else [])
if any(isinstance(d, HIV) for d in other_diseases):
    auto_connectors.append(hpv_hiv_connector())
    auto_analyzers.append(HIVStratifiedResults())
```

This preserves the "no uncoupled co-infection" safety: passing `hpv.HIV()` without
the connector cannot happen. **Backward-compat is exact** — for every existing
call `other_diseases` is empty, so the assembled `diseases`/connectors/analyzers
are byte-identical to today, and the override tests pass unchanged.

Resulting usage is plain Starsim:

```python
hpv.Sim(genotypes=[16, 18, 'hi5', 'ohr'], diseases=[hpv.HIV()], interventions=[sti.ART(...)])
```

## Data flow (per step, HIV present)

1. Demographics, network, HIV (`sti.HIV.step`: CD4 update, ART reconstitution,
   HIV transmission over `hpv.SexualNetwork`), HPV genotype modules step.
2. `CrossImmunity.step` sets each genotype's `rel_sus` and `sev_imm` from the
   cross-genotype matrix.
3. `hpv_hiv_connector.step`: bin CD4 → strata; compute `hiv_rel_sus/sev/imm`;
   multiply each module's `rel_sus` by `hiv_rel_sus`.
4. HPV acquisition uses the HIV-adjusted `rel_sus`. New infections' `set_prognoses`
   reads `hiv_rel_sev`. Clearance/vaccination immunity increments are scaled by
   `hiv_rel_imm`.
5. `HIVStratifiedResults.step` records HIV-stratified outcomes.

## Error handling

- Constructor: HPV-instances + `genotypes=` raises (narrowed guard); clear message.
- Connector `init_pre`: if no genotype modules are present alongside HIV, raise
  (HIV-only sims are out of scope for M08); validate the HIV module is discoverable.
- Network beta: directional orientation asserted by a dedicated unit test.
- All HPV-side read sites no-op (factor 1.0) when no HIV module is present.

## Testing

**File layout:**

- `hpvsim/hiv.py` — `HIV`, `hpv_hiv_connector`, `HIVStratifiedResults` (new).
- `hpvsim/sim.py` — type-partitioned assembly + detection-based auto-wiring (refactor).
- `hpvsim/hpv.py` — gated no-op read sites: `set_prognoses` reads `hiv_rel_sev`;
  `step_state` clearance scales the immunity increment by `hiv_rel_imm`.
- `hpvsim/products.py` — vx/txvx `administer` scales `vax_imm`/`txvx_imm` by `hiv_rel_imm`.
- `hpvsim/data/` — `hiv_init_prev.csv`, `hiv_art_coverage.csv`.
- `tests/test_m08_*.py` — unit + integration + no-HIV identity guard.
- `tests/regression/anchor_hiv_hpv.py` + v2 baseline generator + multi-seed parity test.

**Unit tests:**

- `_cd4_stratum` binning boundaries (CD4 199/200/201, HIV−).
- Each of the three factors applied at its correct site (`rel_sus` post-CrossImmunity
  multiply; `rel_sev` read in `set_prognoses`; `rel_imm` scaling the immunity increment).
- Directional HIV beta lands on the correct `[p1→p2, p2→p1]` orientation.
- Type-partitioned constructor: `genotypes=` + `diseases=[HIV]` assembles HPV + HIV +
  auto-wired connector/analyzer; HPV-instances + `genotypes=` raises; HPV-instance
  override path unchanged.

**Integration / regression:**

- Co-infection anchor (`anchor_hiv_hpv`): HIV+ agents show elevated HPV acquisition
  and cancer vs HIV−, in the expected direction.
- **No-HIV byte-identity guard:** an HPV-only sim built the old way vs through the
  refactored constructor produce identical results (protects M01–M07 CRN streams).
- Phase 2: multi-seed z-score parity on the HIV-stratified `short_summary` vs the v2
  Rwanda baseline (`|z| < 3`, loosened to `|z| < 5` per a single residual metric only
  if needed and documented).

## Validation gates

- **Dev gate (per PR):** `anchor_hiv_hpv` + multi-seed z-score on HIV-stratified
  short-summary vs the locally-generated v2 baseline. Informational, not
  auto-blocking, per the dual-validation convention.
- **Release gate:** `hpvsim_rwanda` reproduces with overlapping uncertainty intervals.

## Decisions settled

Items 1–7 settled during brainstorming; 8–10 settled during Phase 2
implementation (2026-06-04) and refine/supersede earlier ones as noted.

1. **CD4 representation:** discrete strata now (lt200/gt200), continuous mapping
   later — isolated behind `_cd4_stratum`.
2. **HIV network:** share `hpv.SexualNetwork` (co-infected agents keep consistent
   partners; reuses M01-validated network). Directional beta wired explicitly.
   (Moot under the incidence-driven approach, decision 9, where `beta=0`.)
3. **HIV intervention scope:** HIV disease + ART only; no testing/diagnosis cascade.
4. **HIV→HPV effects:** all three (`rel_sus`, `rel_sev`, `rel_imm`).
5. **Sequencing:** Approach C — phased path; Phase 1 mechanics, Phase 2 Rwanda.
6. **HIV is a disease, not a flag:** entered via `diseases=`; no bespoke `hiv=` arg.
7. **Re-evaluation:** static prognoses first; dynamic only if Rwanda gate requires it.
8. **ART = coverage shortcut:** `hpv.hiv_art` (an `sti.ART` subclass) diagnoses all
   HIV+ to let `sti.ART` drive the on-ART fraction to the Rwanda age/sex/year
   coverage curve — no `HIVTest` (v2-faithful, data-supported).
9. **HIV epidemic = incidence importer (supersedes the transmission-β path in
   decision 5's Phase 2):** `hpv.hiv_incidence_import` imposes the Rwanda incidence
   curve via `sti.HIV.set_prognoses` with `beta=0`; reproduces the trajectory by
   construction. **Deviates from the plan's transmission-based mandate — flagged for
   Robyn at PR.** Transmission capability retained (`beta_m2f` arg); β calibration
   is future work. See the Phase 2 section for the validation + rationale.
10. **Parity metric is adult-restricted (15–49):** the all-ages denominator
    understates HIV prevalence ~2× (M05-class artifact). T13's tolerance-band gate
    (not a z-score gate — the cached v2 baseline is posterior quantiles, not a seed
    sweep) compares against `load_hiv_baseline()` + the published 2017 points.

## Linked documents

- [`MIGRATION_PLAN.md`](../../../MIGRATION_PLAN.md) — M08 milestone definition.
- [M07 MultiSim and Scenarios](2026-05-27-m07-multisim-design.md) — predecessor.
- `hpvsim/_v2_legacy/hiv.py` — porting reference (CD4-stratified effects, ART);
  not reused, reference only.
- `stisim/diseases/hiv.py`, `hivsim_examples/zimbabwe/` — STIsim HIV + calibration template.