# Multiscale Agents — Feasibility Investigation & Design

**Date:** 2026-05-28
**Feature:** Multiscale agents (per-agent population scaling for rare outcomes)
**Branch:** `m07-multiscale-investigation` (off `m07-multisim`)
**Status:** Feasibility investigation complete. Recommendation: **conditional go** —
implement the faithful v2 port as a future milestone. Not yet scheduled.
**Migration-plan status:** "Multiscale modeling — Unscheduled; low priority; not
a release blocker" (`MIGRATION_PLAN.md:358`).

> **Implementation update (superseded approach below).** This document analyzes
> and recommends the v2-faithful **grow** approach (spawn fine cancer agents at
> `scale = 1/N`). The shipped implementation instead uses a **cancer-pathway
> ledger**: the extra `N−1` sub-cancers per CIN agent are resolved as scheduled
> DATA in a deterministic side-RNG ledger, with NO fine People agents grown. The
> ledger is a read-only overlay, so the population/transmission is bit-identical
> across `ms_agent_ratio` and all the Level0/fine-agent/network-exclusion/
> scale-weighting machinery this document proposes proved unnecessary and was
> removed. The natural-history accounting (length-biased duration resampling,
> competing-risk realization) carries over; only the *substrate* changed (data
> rows instead of agents). See the `_multiscale_ledger`/`_realize_ledger` methods
> in `hpvsim/hpv.py` and `tests/test_multiscale*.py`.

---

## 1. Goal

Port v2 hpvsim's **multiscale agents** technique to the v3/Starsim architecture,
for v2 feature parity. Multiscale lets a coarse population (each agent
representing many people via `pop_scale`) resolve rare outcomes — cervical
cancer and cancer death — at high statistical resolution without simulating the
whole population at fine grain. When a coarse agent progresses toward cancer,
the model spawns `ms_agent_ratio` fine-resolution agents at reduced per-agent
scale, so cancer incidence is estimated from many low-weight agents rather than
one high-weight Bernoulli outcome.

After implementation, a user can set `ms_agent_ratio > 1` on a `hpv.Sim` and get
v2-equivalent multiscale behavior; `ms_agent_ratio = 1` (default) leaves all
behavior unchanged.

### Motivation

v2 feature parity (the migration's reason for this work). Some downstream
analyses use `ms_agent_ratio` to get smooth cancer-incidence curves cheaply; the
v3 port must be able to reproduce that capability. The technique's variance
benefit is established by v2's published use — the open question this
investigation answered is whether it integrates cleanly with Starsim, and what
the v3-specific implementation cost is.

### Acceptance target: internal equivalence (v3 vs v3)

Per the investigation directive, correctness is judged by **internal
equivalence** rather than a v2 baseline:

> A multiscale run (`ms_agent_ratio = N`, few agents) must reproduce a
> single-scale run (`ms_agent_ratio = 1`, many agents) on cancer statistics, in
> people-space, within tolerance — ideally at lower variance per unit compute.

This isolates the technique's correctness without a v2-baseline dependency. (A
v2 cross-check can be added later but is not the gating bar.)

---

## 2. Background: how v2 does it

Source: `hpvsim/_v2_legacy/people.py:280-369`. Key elements:

- Per-agent `scale` array — number of real people each agent represents. Coarse
  agents carry `pop_scale`; fine cancer agents carry `pop_scale / ms_agent_ratio`.
- `level0` (coarse) / `level1` (fine) agent tags. A fine agent never spawns more
  fine agents.
- The split is hooked at the **CIN→cancer decision**, not at infection or first
  CIN. For each coarse CIN agent being evaluated for cancer: the original is
  shrunk to `cancer_scale`; `ms_agent_ratio` *additional independent*
  CIN/cancer draws are made; each cancer success spawns a new fine agent at
  `cancer_scale`, copied from the source with fresh durations.
- All result tallies are **scale-weighted** (`self.scale[inds].sum()`, via
  `scale_flows`), never raw counts.

The net effect: a coarse agent's single high-variance Bernoulli cancer outcome
is replaced by ~`ms_agent_ratio` low-weight independent outcomes, conserving
expected person-mass while cutting cancer-incidence variance by ≈`ms_agent_ratio`.

---

## 3. What the v3/Starsim stack already provides

Established by reading the Starsim 3.3.4 source and the v3 hpvsim port:

- **Per-agent scale.** `starsim/people.py:69` defines
  `ss.FloatArr('scale', default=1.0)` — "the scale factor for the agents
  (multiplied for making results)". `People.scale_flows(inds)` returns
  `self.scale[inds].sum()`.
- **Global population scaling.** `Result(scale=True)` multiplies a result by the
  **scalar** `sim.pars.pop_scale = total_pop / n_agents` at finalize
  (`starsim/sim.py:559-560`). Per-agent `scale` is a *relative* weight on top of
  this; for uniform agents (`scale = 1.0`) the two compose to the usual
  `count × pop_scale`.
- **Mid-run agent creation.** `People.grow(n)` (`starsim/people.py:345`) appends
  agents, auto-extends **every registered module state**, and assigns each new
  agent a fresh slot (`slot = new_uid`). The v3 demographics module already calls
  `people.grow()` mid-run for immigration (`hpvsim/demographics.py:207`), so this
  is a proven path.
- **Slot-keyed CRN.** Starsim's common-random-number streams are keyed by agent
  slot. New agents get new slots, so spawning does not perturb existing agents'
  draws — reproducibility is preserved across a variable agent count.

**Gap:** the v3 `HPV` module records rare outcomes as **raw counts**, not
scale-weighted — e.g. `hpv.py:490` `self.results.new_cancers[ti] = len(to_cancerous)`
and `hpv.py:508` for `new_cancer_deaths`. With uniform scale this is correct
(`len × pop_scale`); with multiscale it is wrong. This is the central code
change (see §6).

---

## 4. Feasibility spike — evidence

A throwaway spike (`spike/multiscale_spike.py`, findings in
`spike/SPIKE_FINDINGS.md`) built a `MultiscaleHPV(hpv.HPV)` prototype and tested
four things on the single-genotype HPV16 anchor:

| Check | Result |
|---|---|
| **Mechanics** — `grow()` + multi-module state copy | ✅ Population grew 3000→51,255 raw slots; all module arrays length-consistent; zero NaNs. |
| **Scale-weighted results + `pop_scale`** | ✅ 358 raw cancer agents at scale 0.1 → 36.7 agent-equivalents → correct people-space count. |
| **Reproducibility (CRN)** | ✅ Two same-seed runs bit-identical. |
| **Internal equivalence** | ⚠️ Naive prototype lost ~74% of cancers — see below. |

Equivalence controls (6 seeds, `total_pop = 1e6`, cancers in people-space):

```
single  N=20000         : mean=49992  CV=0.047  bias=+0.0%
multi   N=20000 ratio=1 : mean=49992  CV=0.047  bias=+0.0%   subclass-with-no-split == base HPV
multi   N=3000  ratio=1 : mean=46278  CV=0.046  bias=-7.4%   low-agent discretization
multi   N=3000  ratio=10: mean=12772  CV=0.111  bias=-74.5%  naive split loses cancers
```

**Interpretation.** The Starsim *plumbing* works; the naive prototype's bias
comes from two accounting shortcuts it took that the real port must not:

1. It split at **first CIN** and **permanently** shrank the agent, so each
   agent's later-life reinfection→cancer episodes (where most cancer occurs)
   were under-weighted and never re-resolved.
2. It cloned trajectories deterministically and excluded clones from the
   network, rather than making v2's independent per-event draws.

v2 avoids both by hooking the **cancer transition specifically** with
`level0`/`level1` per-event bookkeeping. The spike therefore confirms: the
integration is low-risk; the **conservation accounting is the work**.

---

## 5. Recommendation

**Conditional go — implement the faithful v2 port as a future milestone.**

- The technique integrates cleanly with Starsim; no blocking architectural
  obstacle was found.
- The effort is milestone-sized and dominated by (a) faithfully porting v2's
  event-level `level0`/`level1` scaling and (b) a cross-cutting audit of every
  result/analyzer that can include reduced-scale agents. It is **not** a quick
  win — consistent with its "low priority, not a release blocker" status.
- Recommend scheduling it as its own milestone (provisionally **M-multiscale**,
  sequenced after the core natural-history and intervention milestones land),
  gated by the internal-equivalence acceptance test below.

---

## 5a. Framework-level alternative (considered, not chosen)

Starsim ships the multiscale *primitives* — per-agent `scale`
(`people.py:69`), `scale_flows()`, `grow()`, slot-keyed CRN, and
`Result(scale=True)` — but no turnkey multiscale feature and no `split`/
`rescale` operation. Two pieces of this work are genuinely generic and would
fit at the Starsim framework level: (1) a `people.split(uids, ratio)` operation
(shrink scale, grow copies, copy all module states, tag fine, fresh CRN), and
(2) scale-weighted result counting by convention (the reason hpvsim must audit
every `len()` → `scale.sum()` is that Starsim modules count raw agents). There
is precedent — Covasim, Starsim's predecessor, shipped multiscale as a
first-class feature.

**Decision: keep the implementation in hpvsim.** The framework route does not
remove the genuinely hard part (the disease-specific conservation accounting,
§4) and it adds an upstream-contribution + version-dependency coordination cost
on a feature that is low priority here. The mechanism is the architecturally
right candidate to upstream *later* if it proves broadly useful, but this port
ships self-contained in hpvsim first. (Revisit if a second Starsim-based model
needs the same capability.)

## 6. Proposed architecture (for the full port)

### 6.1 Parameters

- Add `ms_agent_ratio` (int, default `1`) to `SimPars` / forwarded to each `HPV`
  module. `1` = feature off, zero behavior change. This is the v2 parameter
  name, preserved for parity.
- `pop_scale` already exists as `sim.pars.pop_scale` (derived from `total_pop`).
  No new top-level scaling parameter is needed; per-agent `scale` is the
  relative weight on top of it.

### 6.2 Spawning hook (faithful to v2)

In `HPV`, hook the **CIN→cancer decision** (the `_cancer_bern` evaluation in
`set_prognoses`, `hpv.py:346-373`), not infection or first CIN:

- For each coarse (`scale == pop-relative 1.0`) CIN agent being evaluated:
  shrink its scale to `1.0 / ms_agent_ratio`.
- Draw `ms_agent_ratio` additional **independent** CIN/cancer trajectories
  (fresh `dur_cin`, `rel_sev`, cancer Bernoulli), mirroring
  `_v2_legacy/people.py:299-324`.
- For each extra cancer success, `people.grow(1)` a fine agent at scale
  `1.0 / ms_agent_ratio`, copy the source's demographic + natural-history state,
  and set its independently-sampled `ti_cancerous` / `ti_dead_cancer`.
- Tag fine agents so they never re-split (v2's `level1`). Implement as a
  `BoolArr('multiscale_fine')` on `People` or a per-module marker; a fine agent
  is excluded from the split candidate set.

Fine cancer agents do not transmit (`rel_trans = 0` once cancerous, as today)
and need not join the sexual network — consistent with v2, where cancer agents
are post-infectious. This must be verified, not assumed (see §8 risks).

### 6.3 Scale-weighted results — the cross-cutting change

Every result or analyzer that can tally agents whose `scale` may differ from the
coarse baseline must switch from `len(uids)` to `people.scale[uids].sum()`
(i.e. `scale_flows`). Inventory to audit (non-exhaustive; the milestone produces
the complete list):

- `HPV`: `new_cancers`, `new_cancer_deaths`, `sum_age_at_cancer`,
  `sum_age_at_cancer_death`, `n_cancerous`, and any `n_*` compartment counts that
  reduced-scale agents can occupy.
- `HPVTotal` aggregator and `cum_*` derivations.
- `AgeResults` / `by_age` analyzers (`analyzers.py`) — already weight some
  histograms by `scale` in v2; confirm v3 parity.
- Population/demographic results that count fine agents (`n_alive`, prevalence
  denominators) — fine agents inflate raw alive counts but sum correctly by
  scale.

Because `Result(scale=True)` already multiplies by the scalar `pop_scale`, the
per-agent weighting must produce **agent-equivalent** counts (coarse agent = 1.0,
`ms_agent_ratio` fine agents summing to 1.0), so the existing `pop_scale`
multiply stays correct and is not double-applied.

### 6.4 Module layout

```
hpvsim/
  hpv.py            # EDITED — ms split hook in set_prognoses; scale-weighted
                    #          result tallies in step_state
  parameters.py     # EDITED — add ms_agent_ratio (default 1)
  analyzers.py      # EDITED — scale-weight by_age cancer histograms
  cross_genotype.py # EDITED if HPVTotal aggregation needs scale-weighting
  sim.py            # EDITED — forward ms_agent_ratio to HPV modules
tests/
  test_multiscale_equivalence.py   # NEW — internal-equivalence gate
  regression/anchor_*              # reuse existing cancer-bearing anchors
```

---

## 7. Acceptance criteria

The feature is done when:

1. `ms_agent_ratio = 1` produces **bit-identical** results to the current code
   (no-op path; verified in test).
2. **Internal-equivalence gate** passes: for a cancer-bearing anchor
   (`anchor_hpv16` and `anchor_4genotype`), `ms_agent_ratio = N` at reduced
   `n_agents` reproduces `ms_agent_ratio = 1` at full `n_agents` on the cancer
   short-summary metrics (total cancers, cancer deaths, mean cancer incidence,
   mean age at cancer/death) within an agreed tolerance — multi-seed, in
   people-space. The spike's control harness (single vs ratio=1 vs ratio=N) is
   the template.
3. The multiscale run demonstrates **lower variance** on cancer incidence than a
   single-scale run at the same agent count (the efficiency claim).
4. Every audited result/analyzer is scale-weighted; the audit list is documented.
5. Multi-genotype correctness: splitting in one genotype module does not corrupt
   other genotype modules' arrays (the spike showed `grow()` extends them; the
   milestone must verify *values* stay correct, including the cross-genotype
   cancer-cancellation in `_cancel_other_genotype_progression_for`).

---

## 8. Risks

- **Conservation accounting (primary).** Demonstrated by the spike: a wrong hook
  point or scale bookkeeping silently biases cancer counts by tens of percent.
  Mitigation: port v2's event-level algorithm faithfully; gate on the
  internal-equivalence test with the no-split bit-identity check as a guardrail.
- **Result-audit completeness.** Missing one scaled result yields a subtle bias
  only visible in that metric. Mitigation: enumerate every `scale=True` result
  and every analyzer histogram; add per-metric equivalence assertions.
- **Network interaction of fine agents.** Fine cancer agents are post-infectious
  and likely need no network edges, but mid-run `grow()` agents must not break
  `SexualNetwork` invariants. Mitigation: explicit test; follow how births enter
  the network.
- **Multi-genotype state on spawn.** A fine agent spawned by one genotype module
  must have coherent state in *all* genotype modules. Mitigation: set the new
  agent's state across modules as v2 does (`states_to_set` loop), with a
  multi-genotype equivalence test.
- **Memory growth on long runs.** `grow()` only appends; the agent array grows
  with cumulative cancers (spike: 3000→51k slots over 70 years at ratio=10).
  Bounded by cumulative rare events, but worth measuring and noting.

---

## 9. Out of scope

- Multiscaling any outcome other than cancer/cancer-death (v2 only multiscaled
  the cancer transition).
- HIV-coinfection multiscale interactions (v2 `_v2_legacy/hiv.py` references
  `scale`); deferred until the v3 HIV module exists.
- A v2-baseline parity gate (internal equivalence is the chosen bar; v2
  cross-check is an optional follow-up).
- Dynamic agent *removal* / pruning of spent fine agents (memory optimization);
  deferred unless long-run memory proves problematic.

---

## 10. References

- v2 implementation: `hpvsim/_v2_legacy/people.py:280-369` (spawning),
  `:102-106` (`scale_flows`), `:142-143` (`scale` init).
- v2 concept devtests: `tests/devtests/multiscale_concept2.py`,
  `tests/devtests/test_multiscale.py`.
- v3 cancer pathway: `hpvsim/hpv.py:235-373` (`set_prognoses`), `:408-510`
  (`step_state`).
- Starsim scaling: `starsim/people.py:69` (`scale`), `:345` (`grow`);
  `starsim/sim.py:559-560` (`pop_scale` multiply).
- v3 mid-run growth precedent: `hpvsim/demographics.py:207`.
- Spike: `spike/multiscale_spike.py`, `spike/SPIKE_FINDINGS.md`.
- Migration-plan status: `MIGRATION_PLAN.md:31,358`.
