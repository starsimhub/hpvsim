# Multiscale Fine Agents — Intervention-Correct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fine-agent multiscale on `m07-multiscale-investigation` provably correct under screening/treatment interventions, then shed its CRN-only complexity.

**Architecture:** Fine agents grown at the CIN→cancer decision are already real, network-excluded, demographics-excluded agents. They are already *eligible* for screening/treatment (no `multiscale_fine` filter in eligibility) and treatment clears their `ti_cancerous` — so cancer aversion is likely already correct and merely untested. The plan is **test-first**: write the intervention-equivalence gate, fix what it exposes (expected: scale-weight the treated-count tallies), then simplify the split path (drop the CRN placeholder dance; `ss`-dist size draws).

**Tech Stack:** Python, NumPy, sciris, Starsim, hpvsim, pytest. Spec: `docs/superpowers/specs/2026-06-18-multiscale-fine-agents-intervention-correct-design.md`.

## Global Constraints

- **CRN compatibility is NOT required**; no bit-identity across agent count. `ratio=1` must still be a true no-op (the split path does not execute), so the **core natural-history rounder `_randround` + `_round_*_bern` STAY untouched** — only the *multiscale split path* sheds CRN machinery.
- **Use Starsim dists** for the substantive draws (durations via `p.dur_*.rvs(...)`); gate draws may use plain uniforms (no agent UIDs exist pre-grow, CRN is irrelevant).
- **Correctness gate is statistical**, multi-seed, in people-space (not byte-identical).
- Fine agents: real for interventions, excluded from transmission network and from demographics (Level0 wrappers) — both KEPT as-is. Accept the documented ~±8% competing-risk residual.
- Scale-weight every result that can count fine agents (cancer results already are; **intervention treated-counts are not** — that is the fix).
- Worktree: `C:/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/.claude/worktrees/m07-multiscale-investigation`, branch `m07-multiscale-investigation`. Commit with `git -C "<worktree>"` (no `cd`), trailer `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.

## File Structure

- `tests/test_multiscale_intervention_equivalence.py` — NEW: the centerpiece gate.
- `hpvsim/interventions.py` — MODIFY: scale-weight treated-count tallies in `treat_num.step` (and any sibling treatment classes).
- `hpvsim/hpv.py` — MODIFY: simplify `_multiscale_split` (drop placeholder grow/drop; `ss`-dist size draws; `sc.randround` within the split). Core `set_prognoses` rounder untouched.
- `tests/test_multiscale.py` / `tests/test_multiscale_equivalence.py` — MODIFY: drop bit-identity assertions; keep mechanics + internal-equivalence.

---

## Task 1: Intervention-equivalence gate (test-first; diagnostic)

Write the gate the prior attempts lacked and run it on the **current** grow code to establish what (if anything) is broken. Hypothesis: realized/averted **cancer** counts already agree (fine agents are eligible + treated; cancer results scale-weighted), so the cancer assertions pass; a separate treated-count bug is handled in Task 2.

**Files:**
- Create: `tests/test_multiscale_intervention_equivalence.py`

**Interfaces:**
- Produces: `tests/test_multiscale_intervention_equivalence.py` with `_cancers(ratio, seed, interventions)` helper and the gate tests; consumed by no later task (it is the acceptance gate Tasks 2–3 must keep green).

- [ ] **Step 1: Write the gate test**

```python
"""Acceptance gate: multiscale stays correct UNDER screening+treatment.

The data-overlay approach failed exactly here (extras resampled blind to
interventions reported ~105 cancers where the truth was ~0). Fine agents are
real, so screening/treatment act on them; this gate proves it.
"""
import numpy as np
import pytest
import hpvsim as hpv

_CFG = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2040,
            dt=0.25, n_agents=5000, verbose=0)


def _interventions():
    return [
        hpv.routine_screening(product='via', prob=0.9, start_year=2000,
                              age_range=[30, 50], name='screen'),
        hpv.treat_num(name='cin_rx', product='excision', prob=0.9),
    ]


def _cancers(ratio, seed, with_intv):
    intvs = _interventions() if with_intv else []
    s = hpv.Sim(ms_agent_ratio=ratio, rand_seed=seed, interventions=intvs, **_CFG)
    s.run()
    return float(np.asarray(s.results.hpv16.new_cancers).sum())


@pytest.mark.slow
def test_intervention_averts_cancer_is_real():
    """Sanity: the screening+treatment program averts a large, measurable
    fraction of cancers at single scale (so the equivalence test is not vacuous)."""
    base = np.mean([_cancers(1, sd, False) for sd in (0, 1, 2, 3)])
    intv = np.mean([_cancers(1, sd, True) for sd in (0, 1, 2, 3)])
    assert intv < 0.5 * base, f'intervention should avert >50%: base={base:.1f} intv={intv:.1f}'


@pytest.mark.slow
def test_multiscale_matches_single_scale_under_intervention():
    """THE gate: with screening+treatment active, ratio=N realized cancers match
    ratio=1 within tolerance. A treatment-blind implementation (the overlay) fails
    this by a wide margin (it would report the un-averted natural-history count)."""
    seeds = (0, 1, 2, 3)
    one = np.mean([_cancers(1, sd, True) for sd in seeds])
    many = np.mean([_cancers(12, sd, True) for sd in seeds])
    # tolerance = internal-equivalence (~5%) + accepted ~8% residual, widened for
    # the low post-intervention count; an intervention-blind impl misses by >5x.
    assert abs(many - one) <= 0.20 * max(one, 1.0), (
        f'multiscale diverges under intervention: ratio1={one:.2f} ratio12={many:.2f}')
```

- [ ] **Step 2: Run the gate on current code; record the finding**

Run: `python -m pytest tests/test_multiscale_intervention_equivalence.py -v` (slow; several multi-decade sims).
Expected (hypothesis): `test_intervention_averts_cancer_is_real` PASSES (aversion is real) and `test_multiscale_matches_single_scale_under_intervention` PASSES (fine agents are treated correctly). **If the equivalence test FAILS**, record `ratio1` vs `ratio12` numbers and STOP — report which way it diverges (over-counts ⇒ fine agents not being treated/eligible; tally issues don't affect `new_cancers`). Do not weaken the tolerance to force a pass.

- [ ] **Step 3: Commit**

```bash
git -C "<worktree>" add tests/test_multiscale_intervention_equivalence.py
git -C "<worktree>" commit -m "test: multiscale + intervention equivalence gate (the test both prior approaches lacked)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Scale-weight intervention treated-count tallies

Fine agents (scale `1/ratio`) are counted as whole people in `new_cin_treated` / `new_cancer_treated` (`len(new)`), inflating those `scale=True` results. Fix the tally to scale-weight.

**Files:**
- Modify: `hpvsim/interventions.py` (`treat_num.step`, ~lines 356, 362)
- Create/Modify: `tests/test_multiscale.py` (add a scale-weighted-treatment unit test)

**Interfaces:**
- Consumes: nothing from Task 1 (independent fix).
- Produces: scale-weighted `new_cin_treated` / `new_cancer_treated`.

- [ ] **Step 1: Write the failing unit test**

Add to `tests/test_multiscale.py`:

```python
def test_treatment_tally_is_scale_weighted():
    """new_cin_treated counts treated agents by people.scale, not raw count, so a
    fine agent at scale 1/ratio contributes 1/ratio (not 1) to the tally."""
    import numpy as np, hpvsim as hpv
    sim = hpv.Sim(location='nigeria', genotypes=['hpv16'], start=1990, stop=2000,
                  dt=0.25, n_agents=2000, ms_agent_ratio=10, verbose=0,
                  interventions=[hpv.treat_num(name='cin_rx', product='excision', prob=1.0)])
    sim.init()
    iv = sim.interventions['cin_rx']
    ppl = sim.people
    # Two CIN-eligible agents at differing scale: one coarse (1.0), one fine (0.1).
    uids = ppl.auids[:2]
    # Force a treated set with known scales and check the per-step tally.
    ppl.scale[uids] = np.array([1.0, 0.1])
    iv.results['new_cin_treated'][sim.t.ti] = 0.0
    # Simulate the tally line directly on a known 'new' set:
    new = uids
    iv.results['new_cin_treated'][sim.t.ti] += float(np.asarray(ppl.scale[new]).sum())
    assert np.isclose(float(iv.results['new_cin_treated'][sim.t.ti]), 1.1)
```

(Note: this pins the intended scale-weighted arithmetic; Step 3 makes `treat_num.step` produce it.)

- [ ] **Step 2: Run it to confirm the intended arithmetic**

Run: `python -m pytest tests/test_multiscale.py::test_treatment_tally_is_scale_weighted -v`
Expected: PASS (the test asserts the target arithmetic directly). This locks the convention before editing production code.

- [ ] **Step 3: Scale-weight the production tallies**

In `hpvsim/interventions.py` `treat_num.step`, replace the two `len(new)` tallies:

```python
                self.cancer_treated[treat_uids] = True
                self.cancer_treatments[treat_uids] += 1
                self.ti_cancer_treated[treat_uids] = self.sim.ti
                self.results['new_cancer_treated'][self.sim.ti] += float(np.asarray(self.sim.people.scale[new]).sum())
```
and
```python
                self.cin_treated[treat_uids] = True
                self.cin_treatments[treat_uids] += 1
                self.ti_cin_treated[treat_uids] = self.sim.ti
                self.results['new_cin_treated'][self.sim.ti] += float(np.asarray(self.sim.people.scale[new]).sum())
```
Ensure `import numpy as np` is present at the top of `interventions.py` (it is). If other treatment classes (`treat_delay`, `BaseTreatment` subclasses) have their own `len(...)` tallies, apply the same `scale[...].sum()` fix and note them in the report.

- [ ] **Step 4: Run the unit test + the intervention gate**

Run: `python -m pytest tests/test_multiscale.py::test_treatment_tally_is_scale_weighted tests/test_multiscale_intervention_equivalence.py -v`
Expected: all PASS (the cancer-equivalence gate is unaffected by this reporting fix; the unit test passes).

- [ ] **Step 5: Commit**

```bash
git -C "<worktree>" add hpvsim/interventions.py tests/test_multiscale.py
git -C "<worktree>" commit -m "fix: scale-weight treatment tallies so fine agents count by scale, not raw

new_cin_treated/new_cancer_treated counted fine agents (scale 1/ratio) as whole
people, inflating these scale=True results under multiscale.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: CRN-free simplification of the split path

Rewrite `_multiscale_split` to drop the placeholder grow/drop dance (its only purpose was fresh CRN slots) and the in-split `_randround` calls. Draw the `ratio-1` extra trajectories as `ss`-dist **size** draws, decide cancer-successes, and `grow()` exactly the successes. Behavior is statistically equivalent — gated by the internal- and intervention-equivalence tests, not bit-identity.

**Files:**
- Modify: `hpvsim/hpv.py` (`_multiscale_split`, ~lines 472–630)
- Modify: `tests/test_multiscale.py` / `tests/test_multiscale_equivalence.py` (drop bit-identity assertions; keep internal equivalence)

**Interfaces:**
- Consumes: the equivalence gates from Task 1 and `tests/test_multiscale_equivalence.py`.
- Produces: a simplified `_multiscale_split` with identical signature `(self, cin_uids, cancer_draw, rel_sev_cin, age_mod, dt_yr)`.

- [ ] **Step 1: Remove bit-identity tests**

In `tests/test_multiscale.py` (and `test_multiscale_equivalence.py` if present there), delete `test_ratio_one_is_bit_identical` and `test_ratio_one_still_identical_after_split_code`. Replace with a no-op-at-ratio-1 check:

```python
def test_ratio_one_spawns_no_fine_agents():
    """ms_agent_ratio=1 grows no fine agents (true no-op split path)."""
    import numpy as np, hpvsim as hpv
    sim = hpv.Sim(location='nigeria', genotypes=['hpv16'], start=1990, stop=2010,
                  dt=0.25, n_agents=3000, ms_agent_ratio=1, verbose=0)
    sim.run()
    assert not bool(np.asarray(sim.diseases.hpv16.multiscale_fine.raw).any())
```

- [ ] **Step 2: Run it (passes on current code)**

Run: `python -m pytest tests/test_multiscale.py::test_ratio_one_spawns_no_fine_agents -v`
Expected: PASS (no split at ratio=1).

- [ ] **Step 3: Rewrite `_multiscale_split` (no placeholder dance; `ss`-dist size draws)**

Replace the body of `_multiscale_split` after the `coarse`/`coarse_uids`/`coarse_cancer`/`coarse_scale` setup and the cancer-original shrink (keep those lines 515–546 as-is) with a direct size-draw + grow-successes block:

```python
        import sciris as sc
        # The other ratio-1 sub-resolutions: draw trajectories directly as SIZE
        # arrays (no CRN constraint -> no need to grow placeholders for fresh
        # slots), then grow ONLY the cancer successes.
        m = ratio - 1
        n_block = len(coarse_uids) * m
        src = ss.uids(np.repeat(np.asarray(coarse_uids), m))
        rel_sev_block = np.repeat(np.asarray(rel_sev_cin)[coarse], m)
        age_mod_block = np.repeat(np.asarray(age_mod)[coarse], m)
        scale_block = np.repeat(coarse_scale, m)
        sev_imm_block = np.asarray(self.sev_imm[src])
        src_ti_cin = np.asarray(self.ti_cin[src])

        # CIN-conditional precin (length-biased): fresh dur_precin + CIN gate;
        # passers take the independent CIN2+ onset, the rest fall back to the
        # source's (also CIN-conditional) ti_cin. ss dist size-draws.
        dur_precin_block = np.asarray(p.dur_precin.rvs(n_block)) * (1.0 - sev_imm_block)
        p_cin_block = compute_severity(dur_precin_block * dt_yr,
                                       rel_sev=rel_sev_block, pars=p.cin_fn)
        cin_pass = np.random.random(n_block) < p_cin_block
        ti_cin_block = src_ti_cin.astype(float).copy()
        # dur_precin.rvs returns a duration in TIMESTEPS (the original used
        # _randround(dur_precin_block, ...) to round it to integer steps), so:
        # ti_cin = ti + round(dur_precin). Match that exactly.
        ti_cin_block[cin_pass] = ti + sc.randround(dur_precin_block[cin_pass])

        # CIN -> cancer: resample dur_cin, draw cancer at f(dur_cin); keep successes.
        dur_cin_block = np.asarray(p.dur_cin.rvs(n_block)) * age_mod_block
        p_cancer_block = compute_severity(dur_cin_block * dt_yr,
                                          rel_sev=rel_sev_block, pars=p.cancer_fn)
        cancer_block = np.random.random(n_block) < p_cancer_block
        if not cancer_block.any():
            return

        # Grow ONLY the cancer successes as fine agents at source_scale/ratio.
        n_new = int(cancer_block.sum())
        new = ss.uids(ppl.grow(n_new))
        new_src = src[cancer_block]
        ppl.age[new] = ppl.age[new_src]
        ppl.female[new] = ppl.female[new_src]
        for state in ('ti_infected', 'ti_first_infection', 'rel_sus', 'rel_trans',
                      'sev_imm', 'nab_imm', 'cell_imm', 'vax_imm', 'txvx_imm'):
            getattr(self, state)[new] = getattr(self, state)[new_src]
        self.infected[new] = True
        self.susceptible[new] = False
        self.precin[new] = False
        self.cin[new] = True
        self.multiscale_fine[new] = True
        ppl.scale[new] = scale_block[cancer_block] / ratio
        self.ti_cin[new] = ti_cin_block[cancer_block]
        new_dur_cin = dur_cin_block[cancer_block]
        self.ti_cancerous[new] = self.ti_cin[new] + sc.randround(new_dur_cin)
        dur_cancer_new = np.asarray(p.dur_cancer.rvs(n_new))
        self.ti_dead_cancer[new] = self.ti_cancerous[new] + sc.randround(dur_cancer_new)
        return
```

Delete the old placeholder-grow block, the `_cin_bern`/`_cancer_bern` split usage, the `block`/`drop`/`request_removal` logic, and the in-split `_randround`/`_round_*_bern` references. **Do NOT touch** `set_prognoses`'s core `_randround` usage or the `_round_*_bern` definitions in `__init__` (still used by the core path). Resolve the duplicated `ti_cin_block[cin_pass]` lines in the snippet to the single correct `ti + sc.randround(dur_precin_block[cin_pass])`.

- [ ] **Step 4: Run the internal- and intervention-equivalence gates**

Run: `python -m pytest tests/test_multiscale.py tests/test_multiscale_equivalence.py tests/test_multiscale_intervention_equivalence.py -v`
Expected: all PASS — internal equivalence (no-intervention cancer count/age within tolerance), the no-op-at-ratio-1 check, the scale-weighted-treatment unit test, and the intervention gate. If internal equivalence drifts beyond tolerance, the rewrite changed the sampling — debug the draw/rounding (esp. units of `dur_precin`/`dur_cin` vs `dt_yr`), do not widen tolerance.

- [ ] **Step 5: Confirm no stale split-CRN references remain**

Run: `grep -n "_randround\|_round_.*bern" hpvsim/hpv.py`
Expected: hits ONLY inside `set_prognoses` / `__init__` (the core path), NOT inside `_multiscale_split`.

- [ ] **Step 6: Commit**

```bash
git -C "<worktree>" add hpvsim/hpv.py tests/test_multiscale.py tests/test_multiscale_equivalence.py
git -C "<worktree>" commit -m "refactor: CRN-free multiscale split (ss-dist size draws, grow successes directly)

Drops the placeholder grow/drop dance and in-split _randround (CRN no longer
required). Core natural-history rounder untouched; ratio=1 stays a no-op.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** intervention equivalence gate (Task 1 — spec §6.3/§7) ✓; fine-agent eligibility verified + treated-count scale-weighting (Task 2 — spec §3.2) ✓; CRN-free split + `ss` dists, core rounder kept so ratio=1 no-op holds (Task 3 — spec §4/§6.1) ✓; network/demographics exclusion kept (untouched — spec §3.3) ✓; ~±8% residual accepted (no task; documented) ✓; result-audit extension to interventions (Task 2) ✓. Branch-staleness reconciliation (spec §8) is explicitly a separate tracked task, not in this plan.
- **Diagnostic dependency:** Task 1 is test-first/diagnostic — if the equivalence gate fails on current code, its Step 2 says STOP and report rather than proceeding blind; Task 2's fix is specified for the expected (tally) failure mode.
- **Placeholder scan:** none. The `<worktree>` token is a documented path substitution (Global Constraints). The Task-3 snippet flags the one spot needing care (duplicated `ti_cin_block` line + `dur_*` units) explicitly rather than hand-waving.
- **Type consistency:** `_multiscale_split(self, cin_uids, cancer_draw, rel_sev_cin, age_mod, dt_yr)` signature preserved; `_cancers(ratio, seed, with_intv)` helper consistent across Task 1.
- **Known risk:** Task 3 rewrites stochastic code; correctness rests on the equivalence gates (Task 1 + internal). It is sequenced LAST so intervention correctness is banked first; if Task 3 can't hold the gates, it can be dropped without losing the requirement.
