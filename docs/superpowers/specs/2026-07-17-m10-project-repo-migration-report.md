# M10 — Project-repo v3 migration & output reproduction: consolidated report

Date: 2026-07-17. Branch (hpvsim): `m10-hiv-integration`.

## What was done

1. **HIV integration (prerequisite).** Merged the deferred M08 HIV–HPV work
   (`origin/m08-rwanda-on-grow`, 44 commits) into a new branch
   `m10-hiv-integration` cut from `m10-release-readiness`. 5 conflicts resolved.
   Verified on `.venv-ss35`: full non-slow suite **337 passed / 1 xfailed**; 11
   `test_m08_*` HIV tests green; HIV sims construct, auto-wire, and run;
   `hpvsim/hiv.py` byte-identical to the M08 tip. Committed locally, **not pushed**.
2. **Migrated all six published-analysis repos to the v3 API**, each on a fresh
   `v3-migration` branch off its `main`, one dedicated agent per repo. Verified
   at **reduced scale** against a freshly regenerated **patched v2.3.1**
   reference (the multiscale-CIN-regate fix — the committed v2.3.0 baselines are
   the buggy-multiscale artifact and were not used as truth). Bar = overlapping
   intervals / matching trend (v3 uses a different RNG stream). No calibrations
   were run. Nothing pushed; no hpvsim/starsim source modified.

## Per-repo verification

| Repo | Branch | Verdict | Notes |
|---|---|---|---|
| methods_manuscript | v3-migration (3) | Fig 1 bit-identical; Fig 5 partners/dwelltime PASS | Fig 6 screening ASR trend OK but ~+24% hot → beta re-fit |
| india | v3-migration (4) | Fig 2 natural-history panels PASS | figS2/figS3 need a v3 calibration to regenerate |
| faster_kenya | v3-migration (2) | Intervention machinery correct; relative effects track | Absolute cancers collapse (R0<1) with ported kenya network → refit |
| 1dose | v3-migration (4) | Migration correct; relative structure preserved | Epidemic collapses (R_eff<1) with ported low-connectivity networks → per-country refit |
| pxv_younger | v3-migration (4) | Headline reproduced (infant ≈ coverage-equiv adolescent), rel. effects within ~2–3pp | Baseline burden ~13% high → light refit |
| rwanda (HIV) | v3-migration (3) | HIV epidemic sane; intervention trends/ordering reproduce v2 | Absolute ASR + HIV+/HIV- split off at reduced scale → full-scale + refit |

**Cross-cutting theme:** natural-history and *relative* intervention effects
reproduce well; **absolute cancer levels do not**, consistently. Ported v2
calibrations do not transfer to the unbiased v3 engine — india/pxv over-predict
mildly (~+13–24%), kenya/1dose *collapse* below R=1 on their country-specific
low-connectivity networks, rwanda over-predicts on the fixed engine while the
v3 registry re-fit runs low. This is the expected "v3 was never truly
recalibrated" situation and is a **follow-on calibration task**, not a migration
defect — with one caveat below (the `rel_beta` bug may be contaminating the
collapse diagnoses).

## Calibration follow-ups (do not block migration)

- **Per-country transmission (`beta`/network connectivity) re-fit** for kenya
  and 1dose (collapse), pxv_younger (~+13%), india Fig 6 (~+24%). Fit against
  each location's prevalence/ASR targets on the v3 engine.
- **india figS2/figS3** need a v3 calibration set (no v3-format `*_pars.obj`
  exists); the figure-plot paths work against committed CSVs.
- **Rwanda**: re-confirm at full scale (start 1960, n_agents ≥ 20k, ms=5, many
  seeds); the HIV+ cervical-cancer stratum in particular is noise-dominated at
  reduced scale and was already flagged for a re-fit.
- Re-run the collapse diagnoses **after** fixing the `rel_beta` bug (below) — a
  silently-ignored transmissibility override could be part of the collapse.

## Suspected HPVsim/Starsim bugs (reported, not patched)

Ranked by impact. #1–2 are source-verified; the rest are reproducible footguns.

1. **`rel_beta` (and `transf2m`/`transm2f`) genotype-par overrides are silently
   ignored.** `HPV.__init__` builds the transmission `beta` dict from the
   *default* `gpars` in `define_pars` (`hpvsim/hpv.py:121-124`), then
   `update_pars` (line 144) applies the override to `pars.rel_beta` without
   recomputing `pars.beta`. Result: varying `rel_beta` via `genotype_pars`
   changes nothing. **Confirmed in source**, reported independently by the
   faster_kenya and 1dose agents. High impact — `rel_beta` is a natural
   calibration knob, and the guide's own former example used it. *Fix idea:*
   recompute `beta` in `init_pre` from the resolved pars, or derive `beta`
   lazily.
2. **Unit-less genotype duration silently collapses the epidemic.** Passing a
   duration distribution without `ss.years` (e.g. `ss.lognorm_ex(mean=3,
   std=9)`) is interpreted in timesteps; at `dt=0.25` the infectious window is
   ~4× short and prevalence → ~0 with no error. Reported by faster_kenya, 1dose,
   and pxv_younger. *Fix idea:* validate/raise when a duration TimePar is given
   a unit-less value.
3. **`hpv.dx`/`hpv.tx` built from a `df` get `name=None` and crash** at
   `people.add_module` (`TypeError: attribute name must be string`), with no
   constructor path to name a custom-CSV product (must set `.name` after
   construction). Usability.
4. **`cancer_fn`/`cin_fn` partial override → `KeyError: 'form'`.** These shape
   dicts replace rather than merge; a partial override drops `form/k/x_infl/ttc`.
5. **`ss.ndict` iteration yields keys, not modules** — `for a in sim.analyzers:
   isinstance(a, X)` is silently always-false. Footgun.
6. **Broken shipped example:** `examples/t05_screen_algorithms.py` calls
   `sim.get_intervention(...)` (11 sites), which does not exist in starsim 3.5
   → `AttributeError`. Should be `sim.interventions[name]`. Fix before release.
7. **Benign:** `RuntimeWarning: divide by zero encountered in log`
   (`starsim/time.py`) whenever a network probability entry equals exactly 1.0.
8. **Windows/tooling:** `sc.parallelize` under the v3 venv produced no output on
   Windows (serial works — likely a spawn/`__main__` interaction); and hpvsim's
   startup `⚙` glyph crashes redirected stdout under cp1252 (`PYTHONIOENCODING=utf-8`).

## Migration-guide updates applied (`docs/migration.qmd`)

Expanded genotype-parameters (unit-wrapped durations, the `rel_beta`/`beta`
caveat, whole-dict shape functions); added a **Sexual network** section
(network-level params, `layer_probs` vs cross-layer conventions, `n_rships`
gone, `shrink` drops edges, NaN debut); rewrote **Interventions and analyzers**
with the concrete breakages (name-not-label lookups, no `get_intervention`,
product-module naming, `default_*`→`dx/tx/vx`, vaccine `sterilizing_p`,
`campaign_*` signature/sex changes, `ss.Analyzer` hooks, reserved `results`
name, `age_causal_infection` drop-in); added **HIV–HPV co-infection**; added
**population scaling** (`total_pop` default) and **annual/ASR stop-year**
behavioral notes; added a **Known issues and tips** section.

## Open decisions for the user

- **HIV in the final v3.0.0 tag?** The guide now documents HIV (it is present
  and tested on `m10-hiv-integration`). If HIV is gated out of the final release,
  revert the HIV section.
- Nothing has been pushed. Approve pushes per-repo / for `m10-hiv-integration`
  when ready.
- Leftover `hpvsim_rwanda_v2` git worktree (v2 reference) — remove with
  `git -C hpvsim_rwanda worktree remove ../hpvsim_rwanda_v2` if not needed.
