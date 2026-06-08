# Multiscale agents in HPVsim v3

How HPVsim resolves rare cervical-cancer outcomes at high statistical resolution
without simulating the whole population at fine grain — what v2 did, the three
v3 approaches that were tried, and why the shipped **ledger** implementation is
the one that reproduces v2's functionality on the Starsim architecture.

- **Code:** `hpvsim/hpv.py` (`HPV._multiscale_ledger`, `HPV._realize_ledger`,
  `HPV._sources_available`, `HPV.step_state`).
- **Tests:** `tests/test_multiscale.py`, `tests/test_multiscale_distribution.py`,
  `tests/test_multiscale_equivalence.py`.
- **Feasibility/design history:** `docs/superpowers/specs/2026-05-28-multiscale-agents-design.md`.

---

## 1. What multiscale is for

HPVsim models a coarse population: each simulated agent stands for many real
people via a global `pop_scale = total_pop / n_agents`. Cervical cancer is a
**rare** terminal outcome of a long natural history
(`susceptible → precin → CIN → cancer → death`, over decades). At a tractable
agent count, very few agents ever reach cancer, so any quantity computed from
cancer events is noisy.

The specific deliverable that motivated the v3 work is **Figure 5 of the methods
manuscript** (originally produced with v2.2.6): the *event-age distributions*
of three points on the cancer causal pathway, back-traced from each cancer
event:

1. **causal HPV infection age** (when the infection that led to cancer was
   acquired),
2. **CIN2+ age** (when it progressed to high-grade lesion),
3. **cancer age** (invasive cancer onset).

The figure is rendered **single-seed**, so its boxplots/quantiles are only as
well-resolved as the number of cancer events in one run. Multiscale exists to
raise the effective sample size of these rare events so the distributions are
sharply resolved — **without biasing them** and without changing the
population-level epidemiology.

**Goal, stated precisely.** For a fixed agent count, multiscale should:

- **(G1) Resolve more cancer-pathway events per run** — ideally `ratio×` more
  cancer onsets, each carrying its own causal/CIN2+/cancer ages — so the Fig-5
  distributions have more samples.
- **(G2) Keep those distributions unbiased** — the *shape* of each event-age
  distribution must match a single-scale run.
- **(G3) Keep the total cancer count (and cancer deaths) unbiased** — the extra
  resolution must conserve cancer mass in expectation.
- **(G4) Reduce Monte-Carlo variance** of rare-event estimators at equal agent
  count.
- **(G5) Leave the rest of the model alone** — transmission, demographics, and
  all non-cancer results should be unaffected.

Multiscale is scoped to the **cancer stage only**. Causal-infection age is set
by transmission and is shared across a coarse agent's sub-resolutions, so it is
not independently tightened (documented scope); it is still required to be
*unbiased*.

---

## 2. How v2 multiscaling worked

Source: `hpvsim/_v2_legacy/people.py` (`set_severity`, ~lines 280–369; births at
~line 782). v2 used **multiscale agents**: extra fine-grained People grown at
the moment of the CIN→cancer decision.

Key quantities:

```
n_extra      = pars['ms_agent_ratio']        # e.g. 12
cancer_scale = pop_scale / n_extra           # weight of each cancer sub-agent
```

Every agent carries a per-agent `scale` (its people-space weight) and two tags,
`level0` (a full-weight coarse agent) and `level1` (a fine sub-resolution agent).
All result counting is **scale-weighted**: `scale_flows(inds) = scale[inds].sum()`.

At the CIN→cancer decision, for the set of CIN-reaching women `inds`:

1. **Resolve the originals.** Draw each woman's own cancer outcome from
   `cancer_prob = compute_severity(dur_cin, rel_sev, cancer_fn)`. The women who
   transform (`cancer_inds`) have their weight **shrunk** from `pop_scale` to
   `cancer_scale = pop_scale / n_extra`. They stay `level0` (still real bodies,
   still demographically counted).

2. **Spawn extras.** For each of the `inds`, build `n_extra` columns of *extra*
   sub-resolutions. Each extra **resamples its own** `dur_precin` and `dur_cin`,
   re-rolls the precin→CIN gate (`extra_cin_bools`) and then the CIN→cancer gate
   (`extra_cancer_bools`), with cancer zeroed where the CIN gate failed. Extras
   are only spawned for the agents that are themselves `level0`
   (`extra_cancer_bools *= level0[inds]`) — a fine agent never spawns more fine
   agents.

3. **Grow the cancer successes.** Only the extras that drew cancer are
   materialized via `people._grow(...)`. Each new agent copies its source's
   state, is tagged `level1` (and `level0 = False`), gets `scale = cancer_scale`,
   and is assigned its resampled durations so it progresses to cancer on its own
   independent timeline.

Net effect: each coarse CIN agent that would have produced one full-weight
cancer instead produces up to `n_extra` cancer events, **each weighted
`pop_scale / n_extra`**. In expectation the cancer mass is conserved
(`E[k/n_extra] = p_cancer`), but the cancer-pathway events are resolved at
`n_extra×` finer granularity, each with independently resampled onset ages —
delivering (G1)–(G4).

To keep the population honest (G5), v2 made demographics **level0-aware**:
births use `this_birth_rate * n_alive_level0` (only full bodies reproduce), and
death/flow counts are scale-weighted. Fine cancer agents are post-transmission
(cancer agents do not transmit), so they do not inflate the epidemic.

**Why this is hard to port verbatim to v3/Starsim.** Starsim's
common-random-number (CRN) scheme is **slot-keyed**: each agent's random draws
are determined by its slot/uid. `people.grow()` allocates new monotonic
uids/slots, and a module's distribution seeds are derived from the **module
name**. Growing agents mid-run therefore (a) consumes fresh slots whose draws
are decorrelated from the single-scale run, and (b) makes the fine agents fully
fledged members of `ss.People` — they are seen by `ss.Births` (per-agent
Bernoulli over *all* alive), `ss.Deaths`, the `AgeMigration` pyramid-pinning,
and the sexual network unless every one of those is taught to special-case them.
v2 owned its entire `People`/`Sim`, so "teach demographics about level0/level1"
was a local change; in v3 it means subclassing or guarding multiple framework
modules.

---

## 3. The three v3 approaches

Three implementations were built and evaluated. All share the same natural-history
math (`compute_severity`, the duration distributions) and the same scoping
(cancer stage only); they differ in **how the extra resolution is represented**.

### 3.1 Approach A — binomial-on-original (no extra agents)

**Mechanism.** Don't grow anything. At the CIN→cancer decision, the coarse agent
(standing for `N = ms_agent_ratio` people-space individuals) resolves its cancer
outcome as `k ~ Binomial(N, p_cancer)` and carries weight `k/N`. The single
agent's `new_cancers` contribution is `k/N · pop_scale` instead of a 0/1
Bernoulli.

**Pros.**

- Trivially **unbiased count** (`E[k/N] = p_cancer`) and **conserves mass**.
- **No agents grown** → population/transmission/demographics completely
  untouched (G5 for free), bit-identical at `ratio=1`.
- Reduces the **per-decision** sampling variance from `p(1−p)` to `p(1−p)/N`.

**Cons / limiting factors.**

- **Fails the primary goal (G1/G2).** The `k` sub-cancers all belong to one
  agent with one age, one CIN time, one causal-infection time. There are **no
  new event-age samples** — the Fig-5 distributions get *no* extra resolution.
  This is the disqualifying limitation: it tightens a *count*, not a
  *distribution*.
- **Variance benefit is regime-dependent (G4).** The seed-to-seed variance of
  the *total* cancer count is dominated by the transmission process (how many
  agents ever reach the CIN decision), which is identical across `ratio` at
  equal agent count. The rare-event term that binomial resolution shrinks is a
  negligible fraction of total variance unless cancer is genuinely rare (small
  population). At `n=4000` over a long horizon the measured std ratio flickers
  around 1.0.

This approach survives as the `test_multiscale_equivalence.py` gate (it is still
a correct, unbiased estimator), but it does not deliver Fig 5.

### 3.2 Approach B — grow fine agents (faithful v2 port)

**Mechanism.** Reproduce v2 directly: at the CIN→cancer decision, shrink the
cancer-drawing originals to `1/ratio` and `people.grow()` `ratio−1` extra fine
cancer agents per coarse agent, each with resampled, length-biased durations and
its own future cancer-onset timeline. Tag them so demographics/network exclude
them.

To make this work on Starsim it required a substantial support layer:

- `Level0Births` / `Level0Deaths` / `Level0People` — scale-weighted or
  level0-restricted births, death tallies, and `n_alive`/`new_deaths`/
  `new_emigrants` so fine agents don't inflate sim-level counts.
- `AgeMigration` fine handling — pin the pyramid on the level0 body count and
  emigrate fine agents at the band's per-capita rate (`_fine_emi`) so they don't
  over-survive.
- Sexual-network exclusion of fine agents.
- `HPV.update_results` recomputing every per-genotype `n_<state>` / `prevalence`
  / `new_infections` in people-space.
- Careful `self.name` preservation so the swapped demographic classes keep
  v2/stock RNG seeds and stay bit-identical at `ratio=1`.

**Pros.**

- **Delivers G1–G4 directly**, exactly as v2: each fine agent is an independent
  natural-history realization with its own onset ages, so the distributions
  tighten and gain `ratio×` samples; each fine agent independently experiences
  background mortality/emigration, so the **competing risk is faithful**.
- Conceptually faithful to the validated v2 technique.

**Cons / limiting factors.**

- **Demographic coupling is the core problem.** Fine agents are real bodies to
  the framework. Without the full Level0 layer they inflate births and get
  emigrated/born against, perturbing transmission (measured up to ~−23% to ~+33%
  swings before the fixes; a documented ~+4% residual cancer count remained).
- **CRN decorrelation.** Growing/removing agents shifts slot allocation, so a
  `ratio>1` run is **not** bit-identical to single-scale and the transmission
  realization drifts between ratios — adding across-seed noise that can exceed
  the sampling-variance gain at non-rare configs.
- **Large, invasive footprint.** It touches `demographics.py`, `sim.py`,
  `network.py`, and `hpv.py`, and every one of those special cases is a place a
  future change can silently break the population invariants.

### 3.3 Approach C — cancer-pathway ledger (shipped)

**Mechanism.** Resolve the extra sub-cancers as **scheduled data**, not as
agents. The coarse agent lives an ordinary single-scale life; its **own** cancer
(sub-resolution 0) drives the population and is counted at weight `1/ratio`. The
other `ratio−1` sub-resolutions per CIN agent are resolved with an **independent
side RNG** and recorded into a **ledger** keyed by future onset/death timestep;
they are realized into the cancer *results* (and into the event-age list the
Fig-5 analyzer reads) when their scheduled step arrives — overlaid on a
population that never grows, shrinks, or reweights.

This is described in detail in §4.

**Pros.**

- **Read-only overlay → population bit-identical across `ratio` (G5, strongly).**
  Because no agent state is grown/shrunk/reweighted, transmission, demographics,
  and every non-cancer result are *identical* regardless of `ms_agent_ratio`.
  This is what makes the entire Level0/fine-agent/network-exclusion layer
  **unnecessary** — it was all reverted to stock.
- **Delivers G1/G2:** each extra resamples its own length-biased durations, so
  it is an independent cancer/CIN2+ event-age sample — `~ratio×` more events,
  distributions unbiased.
- **Delivers G4 cleanly:** because the population is shared across ratios, the
  dominant transmission-variance term cancels exactly between `ratio=1` and
  `ratio>1`, leaving only the (reduced) rare-event resolution noise — a strict
  variance reduction (visible robustly in the rare-cancer regime).
- **Reproducible and process-stable:** the side stream is seeded from
  `(rand_seed, timestep, genotype)` via `crc32`, independent of the slot CRN, so
  it can't perturb the population's draws.
- **Small footprint:** lives almost entirely in `hpv.py`.

**Cons / limiting factors.**

- **Competing risk is modeled, not native.** The extras are not agents, so they
  don't *experience* background mortality/emigration; the ledger approximates it
  by tying each extra to its **source agent's** fate (see §4.3). This is exact
  in expectation for background death + emigration but is an approximation in
  its tails (≈−1% residual count, within the across-seed noise floor).
- **Cross-genotype competition is reconstructed, not native** (see §4.4): a
  small multi-genotype residual remains (~+4% total over single-genotype's ~−1%)
  because competition is resolved by earliest realized onset rather than the
  full severity-coupled dynamics single-scale uses.
- **Per-event ages are continuous; own-cancer ages come from discrete agent
  timesteps** — a sub-`dt` representation difference (immaterial for year-binned
  distributions, but a real asymmetry).

---

## 4. The ledger in detail — reproducing v2's functionality

### 4.1 Scheduling (`HPV._multiscale_ledger`)

Called from `set_prognoses` for every CIN-reaching agent (`cin_uids`), at the
infection step. For `ratio > 1`, for each CIN agent it resolves `m = ratio − 1`
extra sub-individuals:

- **Side RNG.** `seed = (rand_seed·C1 + ti·C2 + crc32(genotype)) & 0x7FFFFFFF`,
  `rng = np.random.default_rng(seed)`. Independent of the slot CRN, reproducible,
  process-stable (`crc32`, not `hash()`).
- **Stable sub-resolution index.** Each extra gets `sub_idx ∈ {1..ratio−1}`
  within its source; `(source uid, sub_idx)` identifies one sub-individual and
  is the cross-genotype competition key (§4.4).
- **Length-biased pre-CIN duration (rejection sampling).** A single-scale
  CIN-reacher's `dur_precin` is *length-biased* — it is the precin that passed
  the CIN gate. The ledger reproduces that distribution exactly by
  **rejection-sampling**: redraw `dur_precin` until it passes
  `f_cin = compute_severity(dpre, rel_sev, cin_fn)`. This both length-biases the
  onset (so cancer-onset truncation matches single-scale → unbiased count) and
  makes every extra an **independent CIN2+-age sample** (an earlier "single draw,
  else fall back to the source's precin" left ~3/4 of extras *copying* the
  source's CIN age, adding no independent information and not tightening the
  CIN2+ distribution).
- **Cancer draw.** Resample `dur_cin`, draw cancer at
  `f_cancer = compute_severity(dcin, rel_sev, cancer_fn)`, keep the successes
  (length-biased `dur_cin` → correct, diversified cancer-onset ages).
- **Record.** For each kept extra, compute continuous `causal`/`cin`/`cancer`/
  `death` ages and integer `onset_ti`/`death_ti`, and append
  `(uid, sub_idx, ages…, death_ti, weight)` to `self._ledger_onset[onset_ti]`.
  Weight = `source_scale / ratio`, captured here while the source is alive (the
  same `w_own · people.scale` convention the own cancer uses; uniform `1/ratio`
  today since `people.scale == 1`).

This mirrors v2 step-for-step — resample durations, re-gate CIN, re-draw cancer,
keep successes — but writes **data rows** instead of growing People.

### 4.2 Realization and counting (`HPV._realize_ledger`, `step_state`)

Each step, after the agent's own natural-history transitions:

- The agent's **own** cancer (sub-resolution 0) is tallied at weight `1/ratio`
  (`new_cancers[ti] = w_own · scale.sum()`), and its pathway ages are recorded
  to `_cancer_events` (the analyzer's data source).
- `_realize_ledger(ti)` pops `_ledger_onset[ti]` and `_ledger_death[ti]` and, for each
  available, un-claimed extra, adds its weight to `new_cancers` /
  `new_cancer_deaths` and its ages to `_cancer_events`. Events whose ti falls
  past the sim window are never popped → correctly truncated.

At `ratio=1` the ledger is empty and `w_own = 1`, so results are **bit-identical**
to the pre-feature code. Mass conservation per CIN agent:
`own f(d)/ratio + (ratio−1)·E[f(dur′)]/ratio = p̄` (the single-scale rate).

### 4.3 Competing risk (`HPV._sources_available`)

A scheduled cancer should only be counted if the individual survives background
mortality / emigration to its onset (in single-scale, an agent that dies of
something else first never reaches cancer). Since extras aren't agents, the
ledger uses the **source agent's fate as a shared proxy**:

```
available = source alive
            OR (source died AND it became cancerous in some genotype at/before death)
```

The crucial subtlety: the source's **own cancer death is excluded** from the
competing risk. A coarse agent represents `ratio` *different* people — the source
dying of its own cancer says nothing about whether a *sibling* sub-individual
(who got cancer independently) is alive, whereas the source's *background* death
or *emigration* **is** a correctly-rated sample of the hazard the siblings face.
Without this exclusion, late-onset extras of cancer-drawing sources are
over-suppressed (this was a ~−5% count bias; excluding the own-cancer death
brought it to ~−1%).

The gate is evaluated only over the small set of source uids for the current
step (not the whole, growing population array), and is the single source of the
competing-risk logic for both onset and death realization.

### 4.4 Cross-genotype competition (`_ms_cancer_claims`)

A real person gets cancer at most once (cancer is terminal). With independent
per-genotype ledgers, a sub-individual co-infected in two genotypes could be
counted in both. The agent itself (sub-resolution 0) is already arbitrated by
`_cancel_other_genotype_progression_for` (which clears other genotypes' pending
cancer when one fires). For the **extras**, a sim-shared registry
`sim._ms_cancer_claims` keyed `(uid, sub_idx)` records the first realized cancer
for each sub-individual; a later cancer for an already-claimed sub-individual —
whether in another genotype or a later reinfection episode of the same genotype
— is suppressed. This mirrors single-scale's "first cancer wins, person dies"
per sub-individual.

This is correct for the *same* sub-individual; it does **not** reproduce the full
severity-coupled cross-genotype dynamics (which genotype wins depends on coupled
severities, not just earliest realized onset), so a small multi-genotype residual
remains.

---

## 5. Verification and equivalence

| Property | Gate | Result |
|---|---|---|
| (G5) Population unaffected | `test_ledger_population_bit_identical_across_ratio` | new_infections / prevalence / n_infected / n_alive **bit-identical** ratio=1 vs 12 |
| `ratio=1` reproduces pre-feature | `test_ratio_one_is_bit_identical` | exact |
| (G1) More event samples | `test_more_cancer_event_samples` | **~7.9×** more cancer-onset samples (>3× required) |
| (G2) Distribution unbiased | `test_distribution_unbiased`, `test_causal_infection_unbiased` | cancer/CIN2+/causal median shift < 2 yr |
| (G2/G4) CIN2+ tightening | `test_cin2plus_distribution_tighter` | across-seed median std reduced (rejection-sampled precin) |
| (G3) Count unbiased (single-genotype) | `test_cancer_count_unbiased` | **~−1%**, within noise (< 8% gate) |
| (G4) Variance reduction | `test_multiscale_reduces_variance_at_equal_agents` | std ratio ~0.5–0.65 in the rare-cancer regime |
| (G3) Multi-genotype total | `test_multigenotype_total_cancer_unbiased` | **~+4%** (< 10% gate) |
| Multi-genotype split | `test_multigenotype_split_bounded` | hpv16 share shift bounded |

The two headline residuals — **~−1%** single-genotype count and **~+4%**
multi-genotype total — are documented limitations of the competing-risk proxy
(§4.3) and the reconstructed cross-genotype competition (§4.4), respectively.
Neither skews the event-age *distributions* (onset age is `dur_cin`-dominated,
not survival-dominated), so the Fig-5 deliverable is unaffected.

---

## 6. Approach comparison

| | A: binomial-on-original | B: grow fine agents (v2 port) | C: ledger (shipped) |
|---|---|---|---|
| Extra resolution as | weight on the original agent | grown People | scheduled data rows |
| More event-age samples (G1) | ❌ none | ✅ yes | ✅ yes (~7.9×) |
| Distributions unbiased (G2) | n/a (no samples) | ✅ | ✅ (<2 yr median shift) |
| Count unbiased (G3) | ✅ exact | ✅ (~+4% residual) | ✅ (~−1% single / ~+4% multi) |
| Variance reduction (G4) | regime-dependent | ✅ but CRN-decorrelated | ✅ robust (shared population) |
| Population untouched (G5) | ✅ exact | ❌ needs Level0 layer | ✅ bit-identical across ratio |
| Competing risk | n/a | native (per fine agent) | modeled (source-fate proxy) |
| Cross-genotype | n/a | native (real agents) | reconstructed (claims registry) |
| Footprint | tiny | large (4 modules) | small (`hpv.py`) |
| Bit-identical at ratio=1 | ✅ | ✅ (seed-preserved) | ✅ |
| Bit-identical population at ratio>1 | ✅ | ❌ | ✅ |

---

## 7. Limiting factors, summarized

- **Approach A** cannot resolve distributions at all (one age per agent) — it is
  a count estimator, not an event-resolution technique. Its variance gain is
  only visible when cancer is genuinely rare.
- **Approach B** is limited by Starsim's agent-centric framework: fine agents are
  real bodies, so faithful demographics requires a large, fragile special-case
  layer, and growing agents decorrelates the CRN so `ratio>1` is no longer the
  same population as single-scale.
- **Approach C** is limited by the fact that the extras are *not* agents: the two
  things real agents get for free — independent competing risk and native
  cross-genotype cancer competition — must be **modeled**, leaving small,
  documented residuals (~−1% single-genotype count; ~+4% multi-genotype total).
  The win is that everything *else* (the entire population) is provably
  unaffected, which is exactly the property the manuscript Fig-5 use case needs.

The ledger was chosen because it is the only approach that delivers the primary
goal (G1/G2 — more, unbiased event samples) **and** the strong population
guarantee (G5 — bit-identical across ratio), at a small code footprint, with the
remaining inaccuracies confined to count-level residuals that do not affect the
distributions the feature exists to produce.
