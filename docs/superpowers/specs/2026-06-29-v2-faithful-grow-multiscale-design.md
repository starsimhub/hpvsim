# v2-faithful grow-real-agents multiscale (hand-built in hpvsim)

**Date:** 2026-06-29
**Branch:** `m07-multiscale-v2-grow` (off `v3.0-dev`)
**Status:** IMPLEMENTED — all 12 plan tasks complete; all acceptance gates pass
(ratio==1 bit-identical; incidence flat rel 0.99/1.01 across {1,5,10};
intervention equivalence; 6.4× event-age variance reduction, unbiased;
v2.3.1 numerical tracking ratio 1.009). See
`docs/superpowers/plans/2026-06-29-v2-faithful-grow-multiscale.md` and
`.superpowers/sdd/progress.md`. Not yet PR'd (standalone comparison branch;
multiscale is Unscheduled in `MIGRATION_PLAN.md`).

## 1. Goal

Reproduce HPVsim v2.3.1's `ms_agent_ratio` multiscale — the **grow-real-agents**
mechanism — on the v3.0-dev (starsim) engine, implemented **entirely in
hpvsim** using only generic starsim-main primitives. This is a deliberately
distinct third implementation, to be compared against the two existing v3
approaches:

- the **ledger / overlay** (m08 / `m07-multiscale-ledger`): no fine agents;
  schedules extra sub-cancers as results-only DATA. Intervention-blind.
- the **native-starsim** port (`m07-multiscale-investigation` + the
  `feat/spawn-fine` starsim branch): `people.split` / `spawn_fine`,
  `epi_weight`, scale-weighted counting, fine-agent competing-risk death — all
  living in the starsim framework.

This third version moves **none** of the multiscale machinery into starsim. It
assumes starsim-main has zero multiscale functionality and rebuilds v2's
hand-built layer in hpvsim on top of these generic starsim-main primitives only:

- `people.grow(n)` → append `n` agents, return their uids (used by births)
- `people.scale` (`ss.FloatArr`, default 1.0) → per-agent result weight
- `people.scale_flows(uids)` → `scale[uids].sum()` (scale-weighted count helper)
- `people.request_removal(uids)` / `request_death(uids)` → agent removal

**CRN safety is explicitly out of scope.** No slot-keyed reproducibility across
scenarios is required; within-run reproducibility from the global `rand_seed`
is sufficient. We may use plain numpy RNG / starsim dists freely.

### Reference source of truth

`hpvsim_v23_frozen` @ branch `fix-multiscale-cin-regate`
(`hpvsim/people.py::set_severity`, commit `c58e9f2d`). This is the **corrected**
v2.3.1 multiscale (CIN-conditional extra `dur_precin` + `age_risk` on extra
`dur_cin`), NOT the un-fixed `_v2_legacy` copy in this repo. The corrected gate
removes the ~3–4× cancer-incidence deflation; we port the corrected behavior.

## 2. Background: the two engines differ in WHERE cancer is decided

| | v2.3.1 (reference) | v3.0-dev (target engine) |
|---|---|---|
| Cancer decision | at **CIN onset**, inside `set_severity` | at **infection**, inside `set_prognoses` (pre-schedules `ti_cin`/`ti_cancerous`/`ti_dead_cancer`) |
| Multiscale grow | in `set_severity`, at the CIN→cancer draw | (new) in `set_prognoses`, at the cancer draw (step 5b) |
| Result counts | `scale_flows` (scale preset to `pop_scale`) | bare `len()` + scalar `pop_scale` multiply at finalize |
| Per-agent scale | `scale[:] = pop_scale` | `scale` default 1.0; scalar `pop_scale` applied once at `finalize_results` |

Because v3 pre-schedules the whole trajectory at infection, the grow happens at
infection too. **Consequence:** fine agents are born at infection (not at CIN
onset like v2) and therefore share the base agent's *full* lifecourse window —
which is internally consistent with the v3 engine, where the base agent is also
"decided" at infection. This is an accepted, principled deviation forced by the
host engine; it is not a behavior the multiscale layer introduces on its own.

## 3. Scale convention

v3 keeps `people.scale == 1.0` and applies population scaling via the scalar
`pop_scale` at `finalize_results` (multiplies every `scale=True` result once).
We preserve that model and use per-agent `scale` purely as a **multiscale
sub-weight**:

```
cancer_scale = 1.0 / ratio        # NOT pop_scale/ratio (pop_scale stays scalar)
```

- Normal agent: `scale = 1.0`.
- Transforming base agent (drew its own cancer): shrunk to `scale = 1/ratio`.
- Grown fine cancer agent: `scale = 1/ratio`.

Net of the finalize `pop_scale` multiply: a single-scale cancer counts as
`pop_scale`; a fine cancer counts as `pop_scale/ratio`; the base + (ratio-1)
expected fine cancers sum to ≈ `pop_scale` — mass-consistent with single scale.

At `ratio == 1`: `cancer_scale == 1.0`, no `scale`/`fine` writes, no grow → the
entire path is a **bit-identical no-op** vs the current single-scale engine.

## 4. New per-agent state

Add a People-level boolean `fine` (v2's `level1`; `level0 == ~fine`), default
`False`. Registered as an extra People state in hpvsim's People setup
(`defaults.py` / wherever hpvsim defines its People states). It is the single
flag that drives all faithful exclusions in §6.

## 5. Fine-agent grow (full v2-exact clone)

Inserted into `HPV.set_prognoses`, immediately after `cancer_uids` /
`nocancer_uids` are determined (current step 5b). No-op when
`int(pars.ms_agent_ratio) <= 1`.

Mirrors `set_severity` (reference lines 288–378):

1. **Shrink base.** For the base agents that drew their own cancer
   (`cancer_uids`), set `people.scale[cancer_uids] = 1/ratio`.
2. **Draw extras for every CIN reacher.** Over all `cin_uids` (= v2's `inds`),
   build the `(n, ratio-1)` extra arrays:
   - `extra_dur_cin = dur_cin_sample * age_mod[:,None]` (same `age_risk`
     modifier as the base — reference's `extra_age_mod`).
   - `extra_dur_precin`: rejection-sample until each passes the precin→CIN gate
     (`compute_severity(..., cin_fn)`), i.e. the **CIN-conditional /
     length-biased** distribution (reference's `while pending.any()` loop). This
     is the corrected gate; every extra is CIN, hence cancer-eligible.
   - `extra_cancer_bools = binomial(compute_severity(extra_dur_cin, cancer_fn))`,
     then `*= level0[inds, None]` so existing fine agents never spawn more fine
     agents.
3. **Grow + full clone.** For the total `n_new = extra_cancer_bools.sum()`
   successes, `new_uids = people.grow(n_new)`, then **clone every per-agent Arr**
   from the broadcast source uids to `new_uids`. "Every per-agent Arr" = the v3
   analogue of v2's `states_to_set`:
   - all People states (age, female, scale, debut, participant, immunity-shared
     fields, …),
   - **every** HPV genotype module's states (susceptible/infected/precin/cin/
     cancerous, all `ti_*`, all `dur_*`, nab_imm/cell_imm/rel_trans, …),
   - the CrossImmunity connector's per-agent states (`rel_sev`, sev_imm inputs).
   A shared helper enumerates clonable per-agent Arrs across People + modules +
   connectors so the clone stays exhaustive as states evolve. The helper also
   **resets lifecycle states** (`ti_dead`, `ti_removed`, `alive`) on the new uids
   to fresh-agent defaults, so a clone never inherits its source's death/removal
   schedule (a clone is a new agent, not a continuation).
4. **Override the fine agents' identity + this-genotype trajectory.** After the
   clone: `fine=True`, `scale=1/ratio`; then set THIS module's fresh cancer-bound
   schedule from the extra draws (`ti_infected=ti`, `precin=True`,
   `infected=True`, `susceptible=False`, `dur_precin/dur_cin/dur_infection`,
   `ti_cin = ti + randround(new_dur_precin)`, and force cancer:
   `ti_cancerous = ti_cin + randround(new_dur_cin)`,
   `ti_dead_cancer = ti_cancerous + randround(dur_cancer)`). This reproduces the
   reference's post-grow block (lines 343–409) plus its `cancer_prob_arr=1`
   forcing.

**Cross-genotype double-cancer** is already prevented: the fine agent's cloned
*other*-genotype cancer schedules are cancelled by the existing
`_cancel_other_genotype_progression_for` the moment any one genotype fires
cancerous for that uid (mirrors v2 `check_cancer`). The `level0` guard in step 2
prevents fine agents from spawning further fine agents.

**Rounding / RNG.** Reuse `_randround` for schedule rounding (CRN-safe rounding
is harmless and already present); the extra-trajectory draws come from **dedicated
`ss.Dist` streams registered on the module in `__init__`** (`_extra_dur_precin`
and `_extra_dur_cin` as unitless-years `ss.lognorm_ex` mirroring the live
duration dists; `_extra_cin_unif` and `_extra_cancer_unif` as `ss.random`). These
are separate RNG streams from the live natural-history dists, so growing fine
agents never consumes randomness the real agents draw from — that isolation is
what keeps `ratio==1` bit-identical and incidence flat across ratios. They are
drawn by size (non-CRN; cross-scenario CRN is out of scope) and reuse
`ss.lognorm_ex`'s external→internal parameterization. (An earlier iteration
hand-rolled this with a `crc32`/magic-constant seed feeding `np.random.default_rng`
plus an inline lognormal ex→im conversion; that was replaced by the dedicated
dists to drop the bespoke seed hash and the duplicated distribution math.)

## 6. Faithful exclusions

Fine (`level1`) agents in v2: **excluded** from the sexual network and from
driving births; **subjected** to background ("other-cause") death. We replicate
exactly:

- **Network.** Override `hpvsim.network.SexualNetwork.active()` (or the
  eligibility expression in `add_pairs`) to AND with `~people.fine`. Fine agents
  never enter edges → never acquire or transmit. (Reference: v2 `is_active`
  requires `level0`, `base.py:1328`.)
- **Births.** A thin hpvsim `Births` subclass (and matching change in
  `AnnualBirths`) computes the birth count from the **non-fine** alive count
  (v2's `n_alive_level0`, `people.py:782`). Newborns are `fine=False`,
  `scale=1.0`. Wire it into `sim.py`'s default demographics stack.
- **Emigration.** Fine agents are excluded from the `AgeMigration` pyramid
  **count/target** (counting them as whole bodies over-fills cancer-age bands and
  causes catastrophic over-emigration), BUT they must still face the **same
  per-capita emigration rate** as real bodies in their band, or they over-realize
  cancer relative to single scale (the coarse source can emigrate before its
  cancer fires, but its fine peers otherwise cannot → incidence inflates with
  `ms_agent_ratio` and horizon, +12% at ratio=10). `AgeMigration` therefore
  applies an **independent per-band Bernoulli hazard** (`_emigrate_fine`,
  `p = band emigrants / band count`) to fine agents, removing pending
  (not-yet-cancerous) successes. (Root-caused during Task 9; the original
  "exclude fine from emigration entirely" was the bug. v2's bias test never
  exposed this — it runs `total_pop==n_agents` with minimal migration.)
- **Background death.** **No change** to `ss.Deaths` — it already applies
  age/sex mortality to all alive agents, so fine agents face it, matching v2
  `apply_death_rates` (which runs over everyone alive with no `level` filter,
  `people.py:705–748`). This is also the competing-risk-correct behavior. When a
  fine agent dies of any cause, starsim's removal pipeline + `HPV.step_die`
  clears its flags.

## 7. Scale-weighted result counting

Because starsim-main `Arr.count()` is plain `np.count_nonzero` and
`finalize_results` only applies the **scalar** `pop_scale`, per-agent `scale`
reaches results only if hpvsim weights tallies itself. Audit and convert **every
population-count HPV result** to `scale_flows`-weighted, **`dtype=float`**:

- **Flows:** `new_cancers[ti] = scale_flows(to_cancerous)`,
  `new_cancer_deaths[ti] = scale_flows(to_dead)` (and any other event flows).
- **Age accumulators:** `sum_age_at_cancer[ti] = Σ(age[to_cancerous]·scale)`,
  likewise for death; the mean-age denominator becomes `scale_flows`, so event-
  age means/distributions are correctly weighted (Fig 5).
- **Stocks:** `n_precin` / `n_cin` / `n_cancerous` / `n_infected` and any
  by-age/by-sex stock tallies → `scale_flows` over the masked uids instead of
  `.count()` / `len()`.
- **HIV-stratified + analyzer results** that count people → same treatment
  (relevant once this lands under the HIV/Rwanda stack, but the engine change is
  here).

**Int-truncation gotcha:** a `1/ratio` write into an `ss.Result(dtype=int)`
truncates to 0 each step, silently biasing every scaled result low. All
scale-weighted results MUST be `dtype=float`. (Verified historically against a
real `ss.Result(dtype=int)`, not a bare numpy array.)

At `ratio == 1` every `scale == 1.0`, so `scale_flows == len` and all of the
above are bit-identical to the current `len()`-based tallies.

## 8. Acceptance gates

1. **ratio==1 bit-identical.** Full single-scale regression (cancer counts,
   event ages, prevalence, all stocks) identical to the pre-change engine. This
   is the gating no-op test.
2. **Incidence flat across ratio.** Cancer incidence statistically flat across
   `ms_agent_ratio ∈ {1, 5, 10}` (target ≈ ±1–2%, the band the v2.3.1 fix
   achieved), powered with enough agents/seeds.
3. **Multiscale × intervention equivalence (centerpiece).** Under a
   screen+treat (and txvx) program, cancers averted at `ratio>1` match `ratio=1`
   within tolerance — because fine agents are REAL and are screened/treated
   natively. This is the property the ledger structurally could not deliver and
   is the main reason to grow real agents. Interventions carry **no `~fine`
   guard** — `check_eligibility` (screening/triage/treatment/txvx) selects on
   `female`/`alive`/`age_range`/cancer-status, all of which fine agents inherit
   from their source clone, so fine agents are eligible automatically. Verified
   empirically (ratio=10, M06 cascade): fine agents are screened, triaged, and
   CIN-treated in proportion to their presence in the cancer-pathway pool.

   **Coverage-type caveat (validated scope).** Equivalence is established for
   **probability-based** coverage (`prob=`), which is a per-agent Bernoulli and
   therefore scale-invariant. **Fixed-capacity** coverage (`treat_num` with
   `max_capacity=N`, or a campaign delivering a fixed number of doses) is NOT
   validated equivalent and is expected to be scale-sensitive: it selects `N`
   agents from an eligible pool inflated by sub-scaled fine agents, and its
   `new_*_treated` counters use a plain `len()` that counts a fine agent as a
   whole body. A scale-aware capacity (draw against `scale_flows`, or weight the
   count) would be required before using fixed-capacity programs under `ratio>1`.
4. **Variance reduction.** Cancer event-age error bars shrink as `ratio`
   increases (the original methods-Fig-5 motivation), at fixed base agents.
5. **v2.3.1 numerical tracking.** Cancer totals / incidence track the
   `hpvsim_v23_frozen` `fix-multiscale-cin-regate` reference for a matched
   scenario, within the v2→v3 engine-difference tolerance already established by
   the natural-history parity gates.

## 9. Files touched (anticipated)

- `hpvsim/hpv.py` — grow block in `set_prognoses`; `scale_flows` counting in
  `step_state` / results; float result dtypes.
- `hpvsim/defaults.py` — `fine` People state; `ms_agent_ratio` default/coercion.
- `hpvsim/parameters.py` — `ms_agent_ratio` parameter.
- `hpvsim/network.py` — `active()` excludes `fine`.
- `hpvsim/demographics.py` — `Births`/`AnnualBirths` non-fine denominator;
  `AgeMigration` emigration excludes `fine`.
- `hpvsim/sim.py` — wire the fine-aware `Births` into the default stack;
  `ms_agent_ratio` plumbing.
- A per-agent-Arr clone helper in `hpv.py` (enumerates clonable Arrs across
  People + all disease modules + connectors).
- Tests: ratio==1 regression, incidence-flat, intervention-equivalence,
  variance-reduction, v2.3.1 tracking.

## 10. Out of scope / non-goals

- CRN / cross-scenario reproducibility (explicitly waived).
- Any change to starsim. All multiscale logic lives in hpvsim.
- Dynamic per-agent rescaling beyond the one-time cancer-scale shrink.
- Reconciling/merging with the ledger or native-starsim approaches; this is a
  standalone third implementation for comparison.
