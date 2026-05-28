# Multiscale feasibility spike — findings

**Date:** 2026-05-28  **Branch:** `m07-multiscale-investigation`  **Status:** throwaway spike (not production)

Spike code: `spike/multiscale_spike.py`. Run with
`PYTHONPATH=$(pwd) python spike/multiscale_spike.py` from the worktree root.
(The worktree needed `hpvsim/data/files/` copied from the main checkout — those
are gitignored downloaded artifacts, absent in a fresh worktree.)

## What the spike tested

A `MultiscaleHPV(hpv.HPV)` subclass that, at `set_prognoses`, splits each
coarse CIN-pathway agent into `ms_ratio` fine agents at relative scale
`1/ms_ratio` via `people.grow()` + state copy. A `CancerScaleAnalyzer` counts
cancer onsets weighted by per-agent `people.scale`.

## Results

| Check | Result |
|---|---|
| **R1 — `grow()` + multi-module state-copy integrity** | ✅ PASS. Population grew 3000→51,255 raw slots; all module arrays (`scale`, `age`, per-genotype `cancerous`) stayed length-consistent; zero NaNs in active ages. `grow()` auto-extends every registered module state. |
| **R2 — scale-weighting reconciles with global `pop_scale`** | ✅ PASS. `Result(scale=True)` multiplies by the scalar `pop_scale = total_pop/n_agents`; per-agent `people.scale` is a *relative* weight (default 1.0). `scale_flows`-style summation (`scale[uids].sum()`) × `pop_scale` gives correct people-space counts. Fine agents at 0.1 sum correctly (358 raw cancer agents → 36.7 agent-equivalents). |
| **R3 — reproducibility under variable agent count** | ✅ PASS. Two same-seed runs were bit-identical (scaled=36.700001, raw=358 both). Starsim CRN is **slot-keyed** and `grow()` assigns new slots = new UIDs, so spawning agents does not perturb existing agents' random streams. |
| **EQ — internal equivalence (cancers in people-space)** | ⚠️ **NOT achieved by the naive mechanism.** See table below. |

### Equivalence controls (6 seeds, total_pop = 1e6)

```
single  N=20000         : mean=49992  CV=0.047  bias=+0.0%
multi   N=20000 ratio=1 : mean=49992  CV=0.047  bias=+0.0%   <- subclass w/ no split == base HPV
multi   N=3000  ratio=1 : mean=46278  CV=0.046  bias=-7.4%   <- low-agent discretization
multi   N=3000  ratio=10: mean=12772  CV=0.111  bias=-74.5%  <- splitting loses ~70% of cancers
```

## Conclusions

**Mechanically feasible.** The three primitives I was unsure about all work on
the v3/Starsim stack: mid-run `grow()` (already used by demographics), automatic
multi-module state extension, and slot-keyed CRN preserving reproducibility.
Per-agent `scale` already exists in Starsim and reconciles cleanly with the
global `pop_scale` multiply, *provided rare-outcome results are switched from
`len(uids)` to `scale[uids].sum()` weighting* (currently `hpv.py` records
`new_cancers = len(to_cancerous)`).

**The hard part is the conservation accounting, not the plumbing.** A naive
"split every first-CIN agent and clone its trajectory" loses ~74% of cancers.
Root cause: agents are **permanently** shrunk to `1/ratio` after their *first*
CIN episode, and clones are excluded from the network — so later-life
reinfection→cancer episodes (when most cancer actually occurs) are
under-weighted and never re-resolved at fine granularity. v2 avoids this by
hooking the **CIN→cancer transition specifically** (not first CIN) and tracking
`level0`/`level1` agents with per-event scale bookkeeping
(`_v2_legacy/people.py:280-369`).

**Implication for the design.** A correct port must replicate v2's
event-level (not episode-level) scaling and its level0/level1 accounting, and
must audit *every* result/analyzer that can include reduced-scale agents
(`n_infected`, `n_cin`, `n_cancerous`, `n_alive`, all `by_age` analyzers) to use
scale-weighting. This is the bulk of the implementation effort and the main
correctness risk; the Starsim integration is not the risk.
