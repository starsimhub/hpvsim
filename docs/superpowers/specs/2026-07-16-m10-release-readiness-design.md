# M10 release readiness — design

**Date:** 2026-07-16
**Source branch:** `m09-analyzers-on-grow` (current furthest-advanced release line)
**Target branch:** `v3.0-dev` → `main` at `v3.0.0`
**Scope:** take the integrated migration line to a published `hpvsim==3.0.0`. Four workstreams: (A) branch integration + scientific validation, (B) strip the v2 quarantine, (C) docs + migration guide, (D) packaging + release. This is a release checklist, not a feature milestone — most items are cleanup, documentation, and verification, not new capability.

---

## 0. Context — what the plan says vs. what is true

`MIGRATION_PLAN.md`'s M10 sub-task list (lines 338–349) and its status table (lines 73–83) are both stale. This spec supersedes them. Audited state as of 2026-07-16:

| Plan item | Reality |
|---|---|
| "M4/M5/M6/M7 in progress, unmerged branches" | **Integrated.** M1–M7 are all merged ancestors of `m09-analyzers-on-grow`, collapsed onto one line via PRs #112/#113/#121. |
| "Switch docs Sphinx → MkDocs/Quarto (#32)" | **Done.** Docs are already Quarto (`docs/_quarto.yml`, `quartodoc`). No Sphinx `conf.py`/`.rst` anywhere. |
| "Strip subclass-first delegations" | **Already clean.** No active module under `hpvsim/` imports the quarantine or delegates to v2. Reduces to deleting the quarantine dirs. |
| "Full analysis-repo validation suite (#64 + …)" | Lives in **6 external repos**, not this repo. One (`hpvsim_rwanda`) is HIV-dependent → gated on M8. |
| version | still `2.3.0` (`hpvsim/version.py:7`) — must bump. |

### The M8 decision (load-bearing for this milestone)

`m08-rwanda-on-grow` (HIV–HPV co-infection + Rwanda calibration) is the **one true fork** off the release spine — a 44-vs-15 divergence from `13864661` that was never merged, and per the branch-restructure decision it is **deliberately off the v3.0 merge path**. This spec confirms that: **HIV–HPV co-infection is deferred to a post-3.0 release.** Consequences for M10:

- The release line is `m09-analyzers-on-grow` — no M8 merge/reconcile is attempted for v3.0.0.
- `hpvsim/hpv.py` and friends ship without the HIV connector; there is no active `hiv` module (the `api/hiv.qmd` reference is a v2 leftover, §C).
- The scientific validation suite for v3.0.0 **excludes `hpvsim_rwanda`** (it requires HIV). Rwanda validation is a gate for the M8 release, not this one.

If the intent has changed and M8 *must* ship in 3.0.0, stop — that reopens a multi-week reconcile (both sides independently evolved the multiscale engine, `_ExclusiveSeeder`, and CRN handling) and is its own milestone, not part of M10.

---

## 1. Branch integration (workstream A, part 1)

Current tips:
- `m09-analyzers-on-grow` (HEAD) — furthest ahead; M1–M7 + grow multiscale + M9 analyzers/plotting.
- `origin/v3.0-dev` = `69404672` (PR #121 merge) — 10 commits behind HEAD (the M9 port).
- local `v3.0-dev` = `12d663db` — **stale, 44 behind**; predates the grow merge.

Actions:
1. **Fast-forward local `v3.0-dev`** to `origin/v3.0-dev` before using it as anything.
2. **Open the M9 PR** (`m09-analyzers-on-grow` → `v3.0-dev`), the last capability PR. After it merges, `v3.0-dev` is the single release line.
3. All subsequent M10 work (B–D) lands on `v3.0-dev` directly (or short-lived `m10-*` branches PR'd into it), per the branching convention.
4. **Final release merge** `v3.0-dev` → `main` happens only at the very end (§5), after the release gate is green.

## 2. Scientific validation (workstream A, part 2)

The release gate (`MIGRATION_PLAN.md` line 383) is **overlapping uncertainty intervals** against the external analysis repos — not bit-exact, not the in-repo dev gate. Canonical list = issue #64.

v3.0.0 gate = 5 non-HIV repos:
- `hpvsim_methods_manuscript`
- `hpvsim_india`
- `hpvsim_1dose`
- `hpvsim_pxv_younger`
- `hpv_faster_kenya`

(`hpvsim_rwanda` deferred with M8, §0.)

Actions:
1. Locate/confirm each repo (siblings of this checkout; `hpvsim_rwanda` is known to be a sibling per project memory). For each: identify its headline figures/outputs and the v2.3 baseline captured under #68–#73 / #82–#87.
2. Run each repo's analysis against the v3 release line; compare against its v2.3 baseline for **interval overlap**. Record per-repo pass/fail + the drift in a validation report committed under `tests/regression/` (e.g. `M10_release_validation.md`).
3. For any repo that does not overlap: classify as (a) fixable regression → fix, or (b) known/expected feature realignment → document with rationale (the grow multiscale engine intentionally tracks a higher, unbiased cancer level than v2.3's biased default — see the multiscale-bias fix; some drift is *correct*).
4. In-repo dev gate (`tests/test_*_parity.py`, `tests/regression/`) must be green on the release line as a precondition — this is cheap and already largely passing.

**Note:** this workstream is the scientific crux and may surface real work. Treat #1–#3 as a discovery task; its output determines whether v3.0.0 is actually releasable or whether specific regressions must be chased first.

## 3. Strip the v2 quarantine (workstream B)

Audit result: **no delegations remain.** No active `hpvsim/` module imports `_v2_legacy`; no delegation TODO markers; the two "delegates" in active code delegate to *starsim* (the v3 target), which is correct. So this is mechanical deletion:

1. `git rm -r hpvsim/_v2_legacy/` (21 modules: `analysis, base, calibration, defaults, hiv, immunity, interventions, misc, parameters, people, plotting, population, run, settings, sim, utils, version` + `data/{downloaders,loaders}` + `__init__`).
2. `git rm -r tests/_legacy/` (16 test modules + `__init__`). Note: the save/load and MultiSim round-trip tests currently live *only* here — port them forward first (§4.3) before deleting.
3. Remove the now-obsolete quarantine note in `hpvsim/__init__.py:6-7` and the banner in `hpvsim/_v2_legacy/__init__.py:5`.
4. Delete orphaned `__pycache__`-only artifacts (`tests/project_validation/`, `tests/_legacy/test_project_validation.py` — no tracked `.py` remains).
5. Run the full test suite after deletion to confirm nothing imported the quarantine transitively (audit says nothing does; verify).

Sequencing constraint: do §3 **after** §4.3 (port the round-trip tests out) and after §2 no longer needs the legacy baselines as a reference.

## 4. Docs + migration guide (workstream C)

### 4.1 Migration guide (real gap — explicit release gate)
Write `docs/migration.qmd` (v2 → v3), linked in `_quarto.yml` and from `overview.md`/`index.md`. Contents:
- API changes: v2 `hpv.Sim(pars_dict)` facade vs. the starsim-native surface; parameter remapping (v2 pars → v3 `ss.Pars`/`GenotypePars`); module renames (`analysis`→`analyzers`, `population`→`network`, removed `base`/`people`/`run`).
- Script-conversion recipes for the common workflows.
- Behavioral deltas users will notice (multiscale cancer level, `init_seeding='exclusive'` default, sex-asymmetric scheduling).
- Leverage the existing `hpvsim/migration_utils.py` (already present — confirm its scope and document/expose it here).

### 4.2 API reference config (build-breaking — fix first)
`docs/_quarto.yml` lists v2 module names in **two** places (nav `contents` lines 88–107 and `quartodoc.sections.contents` lines 156–177). Both:
- **Drop** gone modules: `analysis, base, hiv, immunity, people, population, run`.
- **Add** current active modules missing from the list: `analyzers, cross_genotype, demographics, hpv, network, products, seeding, migration_utils`.
- Keep: `calibration, defaults, interventions, misc, parameters, plotting, settings, sim, utils, data.downloaders, data.loaders`.
- Rebuild `docs` (render script) and confirm the API section generates with no missing-module errors.

### 4.3 Tutorials + changelog
- Re-validate all 7 `docs/tutorials/tut_*.qmd` against v3 output; update any v2-only pars/API. (They use the `import hpvsim as hpv` facade, which largely survives — but outputs and some pars must be checked.)
- Fix stale org links: `institutefordiseasemodeling/hpvsim` → `starsimhub/hpvsim` (binder links in `tut_intro.qmd` etc.).
- `CHANGELOG.md` stops at v2.3.0 — add the v3.0.0 entry summarizing the migration (this feeds `docs/whats-new.md`, which just includes the changelog).
- Add/refresh workflow examples per the plan (calibration, multi-country, vaccination impact). HPV-HIV example is deferred with M8.

## 5. Packaging + release (workstream D)

1. **Version bump** `hpvsim/version.py`: `2.3.0` → `3.0.0`, update `__versiondate__`. (`pyproject.toml` reads it dynamically — no second edit.)
2. **pyproject hygiene:** reconcile `requires-python` (`>=3.9`) with classifiers (3.10–3.12); confirm `starsim>=3.5`, `sciris>=3.0.0` pins; review "Production/Stable" classifier.
3. **Save/load verification (gap):** there is **no active** v3 save/load round-trip test — the only ones are in `tests/_legacy/`. Port `test_save_load_roundtrip` (and the MultiSim variant) forward to `tests/test_misc.py` / `tests/test_run.py` against a v3 `hpv.Sim`; confirm `sc.save`/`sc.load` round-trips an initialized+run sim. Do this **before** §3.2 deletes the legacy tests.
4. **Data files (#12/#30) — investigate, opportunistic:** no code references to either issue were found. `hpvsim/data/downloaders.py` (`quick_download` from `hpvsim/hpvsim_data`, `download_data` from UN WPP2024 via `sc.download`, entry point `hpvsim-download-data`) works. Confirm the download path succeeds on a clean env; only split data files (#12) if load time is a demonstrated problem. Not a release blocker unless download is broken.
5. **Tag + publish (final step):** after A–D green and the release gate (§2) passes — merge `v3.0-dev` → `main`, tag `v3.0.0`, publish to PyPI. This is irreversible and outward-facing: do it only on explicit go-ahead, never as part of an automated pass.

## 6. Sequencing

1. **A.1** fast-forward local `v3.0-dev`; **A.2** merge M9 PR → single release line.
2. **C.2** fix the API config (unblocks docs build) — cheap, do early.
3. **D.3** port save/load round-trip tests forward; **D.1/D.2** version bump + pyproject.
4. **B** delete the quarantine (after D.3); run full suite.
5. **C.1/C.3** migration guide, tutorials, changelog.
6. **A / §2** run the 5-repo scientific validation — the gating discovery task; fix or document drift.
7. **D.5** merge to `main`, tag `v3.0.0`, publish — **only on explicit user go-ahead.**

## 7. Acceptance criteria

- Single release line: `v3.0-dev` contains all M1–M9 work; `hpv.Sim().run()` green (continuous-runnability invariant).
- `hpvsim/_v2_legacy/` and `tests/_legacy/` deleted; full test suite green; save/load round-trip test present and passing on v3.
- Docs build clean on Quarto: API reference matches the real module surface; migration guide published; tutorials run on v3; changelog has a v3.0.0 entry.
- Version = `3.0.0`.
- Scientific release gate: the 5 non-HIV analysis repos reproduce within overlapping intervals (or drift is documented as expected). `hpvsim_rwanda` explicitly deferred with M8.
- `v3.0.0` tagged and on PyPI (final, gated on explicit go-ahead).

## 8. Out of scope (explicitly)

- **HIV–HPV co-infection / M8 / `hpvsim_rwanda` validation** — deferred to a post-3.0 release (§0).
- Any new modeling capability.
- Reconciling the `m08-rwanda-on-grow` fork.
