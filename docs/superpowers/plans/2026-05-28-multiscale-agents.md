# Multiscale Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port v2 hpvsim's multiscale-agents technique (`ms_agent_ratio`) to the v3/Starsim `HPV` module so rare cancer/death outcomes are resolved at high statistical resolution at low agent counts, gated by an internal-equivalence test.

**Architecture:** Add a per-module `ms_agent_ratio` parameter (default `1` = off). At the CIN→cancer decision in `HPV.set_prognoses`, shrink cancer-bound coarse agents to relative scale `1/ratio` and spawn `ratio-1` independent fine cancer agents via `people.grow()` (fresh slots ⇒ independent CRN draws). Switch every rare-outcome result tally from raw `len()` counts to `people.scale`-weighted sums so the existing global `pop_scale` multiply stays correct.

**Tech Stack:** Python, Starsim 3.3.4 (`ss.Infection`, `ss.FloatArr`/`BoolArr`, `people.grow()`, slot-keyed CRN, `Result(scale=True)`), pytest. Design doc: `docs/superpowers/specs/2026-05-28-multiscale-agents-design.md`.

---

## Key facts the implementer must know

- **Per-agent scale is relative.** `people.scale` defaults to `1.0`. Results with `scale=True` are multiplied by the *scalar* `sim.pars.pop_scale = total_pop/n_agents` at finalize (`starsim/sim.py:559-560`). So multiscale code must keep per-agent scale as an *agent-equivalent* weight: a coarse agent = `1.0`; `ratio` fine agents must sum to `1.0`. Never multiply by `pop_scale` yourself.
- **CRN is slot-keyed.** Drawing the same `ss.bernoulli` twice for the *same* uid in one timestep returns the *same* value. To get independent "extra" draws you must `grow()` new agents first (they get fresh slots = new uids) and draw on the *new* uids.
- **`grow()` auto-extends every registered module state** and updates `auids`/slots (`starsim/people.py:345-380`). Demographics already calls it mid-run (`hpvsim/demographics.py:207`).
- **`AgeResults` already scale-weights** its histograms (`hpvsim/analyzers.py:182-190`) — no change needed there; just verify in a test.
- **Run tests against the worktree code**, not the editable install pointing at the main checkout. A fresh worktree also lacks `hpvsim/data/files/` (gitignored downloads); copy it from the main checkout once: `cp -r ../../../hpvsim/data/files hpvsim/data/files` (relative to the worktree root), then prefix test runs with `PYTHONPATH=$(pwd)`. Confirm with `PYTHONPATH=$(pwd) python -c "import hpvsim,os;print(os.path.dirname(hpvsim.__file__))"`.
- **Test idiom:** `import hpvsim as hpv`, plain pytest functions, `hpv.Sim(...)`. Mark long equivalence tests `@pytest.mark.slow`.

## File structure

- Modify `hpvsim/hpv.py` — add `ms_agent_ratio` par; add `_multiscale_split()`; call it in `set_prognoses`; scale-weight cancer/death tallies in `step_state`.
- Modify `hpvsim/sim.py` — accept `ms_agent_ratio=` kwarg and forward to auto-built `HPV` modules.
- Modify `hpvsim/cross_genotype.py` — scale-weight `HPVTotal` `n_alive`, `prevalence`, `cum_infections_unique`, and unique-state sums.
- Create `tests/test_multiscale.py` — no-op identity, scale-weighting unit, mechanics, multi-genotype, variance.
- Create `tests/test_multiscale_equivalence.py` — the `@pytest.mark.slow` internal-equivalence acceptance gate.
- Modify `MIGRATION_PLAN.md`, `CHANGELOG.md` — status + changelog.

The per-agent "fine" marker lives on `People` as a registered `BoolArr` added by `HPV` (so it survives `grow()` and is shared across genotype modules).

---

## Task 1: Add `ms_agent_ratio` parameter (no-op identity)

**Files:**
- Modify: `hpvsim/hpv.py` (`HPV.__init__` `define_pars`, ~line 71; `define_states`, ~line 99)
- Modify: `hpvsim/sim.py` (`Sim.__init__`, ~line 68 signature and ~line 116 auto-build)
- Test: `tests/test_multiscale.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiscale.py
"""Multiscale-agents feature tests."""
import numpy as np
import pytest
import hpvsim as hpv

ANCHOR = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2030,
              dt=0.25, rand_seed=0, verbose=0)


def test_ms_agent_ratio_defaults_to_one():
    """ms_agent_ratio defaults to 1 and is exposed on the HPV module pars."""
    mod = hpv.HPV(genotype='hpv16')
    assert int(mod.pars.ms_agent_ratio) == 1


def test_ms_agent_ratio_forwarded_from_sim():
    """hpv.Sim(ms_agent_ratio=N) forwards to every auto-built genotype module."""
    sim = hpv.Sim(n_agents=200, ms_agent_ratio=5, **ANCHOR)
    sim.init()
    assert int(sim.diseases.hpv16.pars.ms_agent_ratio) == 5


def test_ratio_one_is_bit_identical():
    """ms_agent_ratio=1 must reproduce the pre-feature results bit-for-bit."""
    base = hpv.Sim(n_agents=2000, **ANCHOR); base.run()
    one  = hpv.Sim(n_agents=2000, ms_agent_ratio=1, **ANCHOR); one.run()
    a = np.asarray(base.results.hpv16.new_cancers)
    b = np.asarray(one.results.hpv16.new_cancers)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py -v`
Expected: FAIL — `ms_agent_ratio` not in pars / `Sim.__init__` rejects the kwarg.

- [ ] **Step 3: Add the parameter, the fine-marker state, and sim forwarding**

In `hpvsim/hpv.py`, inside `HPV.__init__`'s `self.define_pars(...)` call (after `sero_prob=gpars.sero_prob,`, ~line 94) add:

```python
            # Multiscale: number of fine cancer agents per coarse agent at the
            # CIN->cancer decision. 1 = feature off (no splitting).
            ms_agent_ratio=1,
```

In the same file, inside `self.define_states(...)` (after the `txvx_imm` FloatArr, ~line 136) add:

```python
            # v2 level0/level1 tag: True for fine agents spawned by multiscale
            # splitting. A fine agent never re-splits. People-level (shared
            # across genotype modules); registered here so grow() extends it.
            ss.BoolArr('multiscale_fine', default=False),
```

In `hpvsim/sim.py`, add `ms_agent_ratio=None` to the `Sim.__init__` signature (line ~68, alongside `genotype_pars=None`). Then in the auto-build branch where modules are created (line 116), forward it:

```python
            diseases = [HPV(genotype=k, **gpars_overrides.get(k, {})) for k in keys]
            if ms_agent_ratio is not None:
                for d in diseases:
                    d.pars.ms_agent_ratio = int(ms_agent_ratio)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py hpvsim/sim.py tests/test_multiscale.py
git commit -m "multiscale: add ms_agent_ratio param + multiscale_fine marker (no-op)"
```

---

## Task 2: Scale-weight cancer/death tallies in `HPV.step_state`

**Files:**
- Modify: `hpvsim/hpv.py` (`step_state`, lines 489-491 and 507-509)
- Test: `tests/test_multiscale.py`

**Why:** `step_state` currently records `new_cancers = len(to_cancerous)` (raw count). With per-agent scale this is wrong. Switch to `people.scale[uids].sum()` so a fine agent at scale `1/ratio` contributes proportionally. `sum_age_at_cancer` must likewise weight ages by scale so the mean-age derivation (`sum_age / count`) stays correct.

- [ ] **Step 1: Write the failing test**

```python
def test_cancer_count_is_scale_weighted():
    """new_cancers weights by people.scale, not raw agent count."""
    sim = hpv.Sim(n_agents=3000, **ANCHOR)
    sim.init()
    mod = sim.diseases.hpv16
    ppl = sim.people

    # Force two agents into the CIN->cancerous transition this step at
    # known scales, and verify the recorded tally is the scale sum (1.5),
    # not the raw count (2).
    uids = ppl.auids[:2]
    ppl.scale[uids] = np.array([1.0, 0.5])
    mod.cin[uids] = True
    mod.cancerous[uids] = False
    mod.ti_cancerous[uids] = sim.t.ti
    mod.ti_clearance[uids] = np.nan
    mod.infected[uids] = True
    mod.step_state()
    ti = sim.t.ti
    assert np.isclose(float(mod.results.new_cancers[ti]), 1.5)
    # age tally also scale-weighted: sum(age*scale)
    expected_age = float((np.asarray(ppl.age[uids]) * np.array([1.0, 0.5])).sum())
    assert np.isclose(float(mod.results.sum_age_at_cancer[ti]), expected_age)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py::test_cancer_count_is_scale_weighted -v`
Expected: FAIL — `new_cancers[ti]` is `2.0` (raw count), `sum_age_at_cancer` unweighted.

- [ ] **Step 3: Scale-weight the cancer tally (and death tally)**

In `hpvsim/hpv.py` `step_state`, replace the cancer-onset recording block (currently lines 489-491):

```python
            ages_at_cancer = self.sim.people.age[to_cancerous]
            self.results.new_cancers[ti] = len(to_cancerous)
            self.results.sum_age_at_cancer[ti] = float(ages_at_cancer.sum())
```

with:

```python
            scale = self.sim.people.scale[to_cancerous]
            ages_at_cancer = self.sim.people.age[to_cancerous]
            self.results.new_cancers[ti] = float((scale).sum())
            self.results.sum_age_at_cancer[ti] = float((ages_at_cancer * scale).sum())
```

Replace the cancer-death recording block (currently lines 507-509):

```python
            ages_at_death = self.sim.people.age[to_dead] + dt_yr
            self.results.new_cancer_deaths[ti] = len(to_dead)
            self.results.sum_age_at_cancer_death[ti] = float(ages_at_death.sum())
```

with:

```python
            scale_d = self.sim.people.scale[to_dead]
            ages_at_death = self.sim.people.age[to_dead] + dt_yr
            self.results.new_cancer_deaths[ti] = float((scale_d).sum())
            self.results.sum_age_at_cancer_death[ti] = float((ages_at_death * scale_d).sum())
```

Note: `new_cancers`/`new_cancer_deaths` are declared `dtype=int` (`hpv.py:190-193`). Change those two `Result(...)` declarations to `dtype=float` so fractional scale weights are not truncated. Leave `cum_*` as-is (they cumsum the now-float arrays; change them to `dtype=float` too).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py -v`
Expected: PASS. The Task-1 `test_ratio_one_is_bit_identical` must still pass (with uniform scale=1.0, `scale.sum() == len`).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_multiscale.py
git commit -m "multiscale: scale-weight cancer/death tallies in step_state"
```

---

## Task 3: Scale-weight `HPVTotal` population/prevalence results

**Files:**
- Modify: `hpvsim/cross_genotype.py` (`HPVTotal.step`, lines ~245-272)
- Test: `tests/test_multiscale.py`

**Why:** `HPVTotal.step` uses raw `int(alive.sum())` and `int(u.sum())` for `n_alive`, `prevalence`, `cum_infections_unique`, and per-state unique sums. Fine agents inflate raw counts; these must weight by `people.scale`.

**Note on the actual code:** `HPVTotal.step` (cross_genotype.py:233-272) works in `.values` space (boolean masks aligned to `people.alive.values`), and `n_alive` is a *local variable*, not a result. The scale-sensitive *results* are the `_UNION_STATES` counts (`n_infected`, `n_susceptible`, `n_precin`, `n_cin`, `n_cancerous`, ...), `n_susceptible`, and `cum_infections_unique`. Weight by `people.scale.values` (same `.values` space as the masks).

- [ ] **Step 1: Write the failing test**

```python
def test_hpvtotal_counts_are_scale_weighted():
    """HPVTotal union counts weight by people.scale: halving every agent's
    scale halves the counts."""
    sim = hpv.Sim(n_agents=1000, **ANCHOR)
    sim.init()
    ppl = sim.people
    tot = [a for a in sim.analyzers.values()
           if a.__class__.__name__ == 'HPVTotal'][0]
    ti = sim.t.ti
    ppl.scale[ppl.auids] = 1.0
    tot.step()
    base_ns = float(tot.results['n_susceptible'][ti])
    ppl.scale[ppl.auids] = 0.5
    tot.step()
    half_ns = float(tot.results['n_susceptible'][ti])
    assert base_ns > 0
    assert np.isclose(half_ns, 0.5 * base_ns, rtol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py::test_hpvtotal_counts_are_scale_weighted -v`
Expected: FAIL — `n_susceptible` is a raw count (unchanged when scale halves).

- [ ] **Step 3: Scale-weight HPVTotal counts**

In `hpvsim/cross_genotype.py` `HPVTotal.step`, introduce `scale_vals = self.sim.people.scale.values` (aligned with `alive = people.alive.values`) and replace each raw `int(...)`-count with a scale-weighted sum:

```python
        scale_vals = people.scale.values
        alive = people.alive.values
        n_alive = float((scale_vals * alive).sum())
        if n_alive == 0:
            return
        union_arrays = {}
        for key, attr in self._UNION_STATES.items():
            u = np.zeros(alive.shape, dtype=bool)
            for m in hpvs:
                u |= getattr(m, attr).values
            u &= alive
            union_arrays[key] = u
            self.results[key][ti] = float((scale_vals * u).sum())
        n_inf = float((scale_vals * union_arrays['n_infected']).sum())
        self.results['n_susceptible'][ti] = n_alive - n_inf
        self.results['prevalence'][ti] = n_inf / n_alive
        ...
        self.results['cum_infections_unique'][ti] = float((scale_vals * ever_infected).sum())
```

`prevalence` keeps its formula (`n_inf / n_alive`) — a ratio, so scale-invariant under uniform scale. Change the affected `Result` dtypes from `int` to `float` in `HPVTotal.define_results`: the `_UNION_STATES` results (defined with `dtype=src.dtype` ~line 221 → use `dtype=float`), `n_susceptible` (~line 224), and `cum_infections_unique` (~line 229).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py -v`
Expected: PASS. Verify Task-1 identity test still passes (uniform scale ⇒ unchanged).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/cross_genotype.py tests/test_multiscale.py
git commit -m "multiscale: scale-weight HPVTotal n_alive/prevalence/unique counts"
```

---

## Task 4: Core split — spawn fine cancer agents at the CIN→cancer decision

**Files:**
- Modify: `hpvsim/hpv.py` (add `_multiscale_split`; call it in `set_prognoses` after the cancer draw, ~line 351)
- Test: `tests/test_multiscale.py`

**Algorithm (CRN-correct port of `_v2_legacy/people.py:288-369`):** For each coarse CIN agent, replace its single cancer Bernoulli (weight 1.0) with `ratio` independent draws each at weight `1/ratio`. The original is one sub-draw; the other `ratio-1` are realized by *growing* fresh agents (new slots ⇒ independent CRN), copying the source's natural-history trajectory, and drawing their cancer outcome on their *new* uids. Conservation: `ratio` agents × `1/ratio` scale = `1.0` (the coarse agent's original weight).

- [ ] **Step 1: Write the failing mechanics test**

```python
def test_split_conserves_scale_mass_and_marks_fine():
    """Splitting shrinks the coarse cancer agent and adds fine agents whose
    summed scale conserves the original 1.0 weight; fines are tagged."""
    sim = hpv.Sim(n_agents=5000, ms_agent_ratio=10, **ANCHOR)
    sim.run()
    ppl = sim.people
    # Any agent that was split is tagged fine and carries scale ~ 1/ratio.
    fine = ppl.multiscale_fine.uids
    assert len(fine) > 0, 'expected some fine agents over a 40-year run'
    assert np.all(np.asarray(ppl.scale[fine]) < 0.9999)


def test_split_is_reproducible():
    """Same seed twice -> identical scaled cancer totals under splitting."""
    def total(seed):
        s = hpv.Sim(n_agents=5000, ms_agent_ratio=10,
                    **{**ANCHOR, 'rand_seed': seed}); s.run()
        return float(np.asarray(s.results.hpv16.new_cancers).sum())
    assert total(7) == total(7)


def test_split_preserves_array_integrity():
    """All module/people arrays stay length-consistent after growth; no NaN ages."""
    sim = hpv.Sim(n_agents=5000, ms_agent_ratio=10, **ANCHOR); sim.run()
    ppl = sim.people
    assert len(ppl.age.raw) == len(ppl.scale.raw) == len(sim.diseases.hpv16.cancerous.raw)
    assert int(np.isnan(np.asarray(ppl.age[ppl.auids])).sum()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py -k split -v`
Expected: FAIL — no splitting yet (`multiscale_fine` empty).

- [ ] **Step 3: Implement `_multiscale_split` and call it**

Add this method to `HPV` in `hpvsim/hpv.py` (place after `set_prognoses`):

```python
    def _multiscale_split(self, cin_uids, cancer_draw, rel_sev_cin, age_mod, dt_yr):
        """Spawn fine cancer agents at the CIN->cancer decision (v2 multiscale).

        For each coarse CIN agent: shrink it to 1/ratio and make ratio-1
        independent extra cancer draws on freshly-grown agents (new slots ->
        independent CRN). Conserves expected cancer mass; resolves cancer at
        ratio-finer granularity. Returns (cancer_uids, dur_cancer_uids) for the
        FULL set (originals + new fine cancer agents) so the caller schedules
        ti_cancerous/ti_dead_cancer for all of them uniformly.
        """
        ratio = int(self.pars.ms_agent_ratio)
        p = self.pars
        ppl = self.sim.people

        # Only coarse agents split; a fine agent never re-splits. NOTE:
        # multiscale_fine is a PER-MODULE state (self.multiscale_fine), not a
        # People-level array — there is no ppl.multiscale_fine.
        coarse = ~np.asarray(self.multiscale_fine[cin_uids])
        coarse_uids = cin_uids[coarse]
        if ratio <= 1 or len(coarse_uids) == 0:
            return cin_uids[cancer_draw]  # unchanged: original cancer set

        # 1. Shrink ALL coarse CIN agents to 1/ratio (each now represents one
        #    of the ratio sub-draws of its population), and tag them fine.
        ppl.scale[coarse_uids] = ppl.scale[coarse_uids] / ratio
        self.multiscale_fine[coarse_uids] = True

        # 2. Grow (ratio-1) extra agents per coarse CIN agent. Fresh uids ->
        #    independent CRN draws below.
        src = ss.uids(np.repeat(np.asarray(coarse_uids), ratio - 1))
        new = ss.uids(ppl.grow(len(src)))

        # 3. Copy demographic identity + clearance-immunity context.
        ppl.age[new] = ppl.age[src]
        ppl.female[new] = ppl.female[src]
        ppl.scale[new] = ppl.scale[src]          # the shrunk 1/ratio value
        self.multiscale_fine[new] = True
        for nm in ('susceptible', 'infected', 'precin', 'cin', 'ti_infected',
                   'ti_cin', 'ti_first_infection', 'sev_imm', 'nab_imm',
                   'cell_imm', 'vax_imm', 'txvx_imm', 'rel_sus'):
            getattr(self, nm)[new] = getattr(self, nm)[src]
        self.rel_trans[new] = self.rel_trans[src]

        # 4. Independent cancer draw for each new agent (on NEW uids -> CRN
        #    independence). p_cancer recomputed from a fresh dur_cin sample.
        rel_sev_new = np.repeat(np.asarray(rel_sev_cin)[coarse], ratio - 1)
        age_mod_new = np.repeat(np.asarray(age_mod)[coarse], ratio - 1)
        dur_cin_new = p.dur_cin.rvs(new) * age_mod_new
        p_cancer_new = compute_severity(dur_cin_new * dt_yr,
                                        rel_sev=rel_sev_new, pars=p.cancer_fn)
        self._cancer_bern.set(p=p_cancer_new)
        cancer_new = self._cancer_bern.rvs(new)

        # 5. New agents that did NOT draw cancer are non-cancer infected
        #    sub-agents: they clear after dur_cin (no cancer). Schedule
        #    clearance and leave them in the population at 1/ratio scale.
        nocancer_new = new[~cancer_new]
        if len(nocancer_new):
            self.ti_clearance[nocancer_new] = (
                self.ti_cin[nocancer_new] + self._randround(
                    dur_cin_new[~cancer_new], nocancer_new,
                    self._round_clear_cin_bern))

        # 6. Return the union of original cancer-drawers + new cancer-drawers;
        #    the caller schedules their cancer timeline.
        orig_cancer = cin_uids[cancer_draw]
        return orig_cancer.concat(new[cancer_new]) if hasattr(orig_cancer, 'concat') \
            else ss.uids(np.concatenate([np.asarray(orig_cancer),
                                         np.asarray(new[cancer_new])]))
```

Then in `set_prognoses`, replace the cancer-scheduling tail. Currently (lines 348-373):

```python
        self._cancer_bern.set(p=p_cancer)
        cancer_draw = self._cancer_bern.rvs(cin_uids)
        cancer_uids = cin_uids[cancer_draw]
        nocancer_uids = cin_uids[~cancer_draw]
        # 5a. Clear after CIN (no cancer).
        self.ti_clearance[nocancer_uids] = (...)
        # 5b. Progression to cancer.
        if len(cancer_uids) == 0:
            return
        self.ti_cancerous[cancer_uids] = (...)
        dur_cancer = p.dur_cancer.rvs(cancer_uids)
        self.ti_dead_cancer[cancer_uids] = (...)
```

Change so the no-cancer clearance for ORIGINALS happens first (unchanged), then call `_multiscale_split` to obtain the full cancer set, then schedule cancer for that full set:

```python
        self._cancer_bern.set(p=p_cancer)
        cancer_draw = self._cancer_bern.rvs(cin_uids)
        nocancer_uids = cin_uids[~cancer_draw]

        # 5a. Clear after CIN (no cancer) — originals only.
        self.ti_clearance[nocancer_uids] = (
            self.ti_cin[nocancer_uids] + self._randround(
                dur_cin[~cancer_draw], nocancer_uids, self._round_clear_cin_bern,
            )
        )

        # 5b. Multiscale: shrink coarse cancer agents and spawn fine ones.
        #     With ms_agent_ratio==1 this returns cin_uids[cancer_draw] unchanged.
        cancer_uids = self._multiscale_split(
            cin_uids, cancer_draw, rel_sev_cin, age_mod, dt_yr)
        if len(cancer_uids) == 0:
            return

        # 5c. Progression to cancer for the full cancer set.
        self.ti_cancerous[cancer_uids] = (
            self.ti_cin[cancer_uids] + self._randround(
                dur_cin_full(self, cancer_uids), cancer_uids, self._round_cancer_bern,
            )
        )
        dur_cancer = p.dur_cancer.rvs(cancer_uids)
        self.ti_dead_cancer[cancer_uids] = (
            self.ti_cancerous[cancer_uids] + self._randround(
                dur_cancer, cancer_uids, self._round_dead_bern,
            )
        )
```

Because `ti_cancerous` for new agents needs a `dur_cin`-based offset and the originals already have one computed, simplify by scheduling `ti_cancerous` for new fine agents *inside* `_multiscale_split` (set `self.ti_cancerous[new[cancer_new]] = self.ti_cin[new[cancer_new]] + self._randround(dur_cin_new[cancer_new], new[cancer_new], self._round_cancer_bern)`), and have `set_prognoses` schedule only the originals as before. Replace the placeholder `dur_cin_full(...)` accordingly: keep the original block scheduling `ti_cancerous`/`ti_dead_cancer` for `cin_uids[cancer_draw]`, and let `_multiscale_split` schedule the same for its new cancer agents. (The implementer should choose whichever keeps both groups scheduled exactly once; the equivalence test in Task 6 is the oracle.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py -k split -v`
Expected: PASS (3 mechanics tests). Re-run the whole file; Task-1 identity test must still pass (`ratio=1` returns early in `_multiscale_split`).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_multiscale.py
git commit -m "multiscale: spawn fine cancer agents at CIN->cancer decision"
```

---

## Task 4b: Exclude multiscale-fine agents from transmission & demographics

**Added after Task 4 implementation revealed a plan gap:** spawned fine agents
copy `infected`/`susceptible`/`rel_trans` and are NOT excluded from the sexual
network or demographics, so they transmit and seed extra infections → ~2× cancer
over-count across ratios. v2 prevented this by operating the network and births
only on `level0` (coarse) agents (`_v2_legacy/people.py:319,354,782,865,940`).
This task mirrors that gating. `multiscale_fine` is per-HPV-module, so the
"fine in ANY genotype" view is the union across HPV modules.

**Files:**
- Modify: `hpvsim/network.py` (`SexualNetwork.set_network_states`, ~line 150)
- Modify: `hpvsim/demographics.py` (`AnnualBirths`, `AgeMigration` alive denominators) — only if measurement shows a material leak
- Test: `tests/test_multiscale.py`

- [ ] **Step 1: Add a fine-agent union helper.** In `hpvsim/hpv.py`, add a module-level function:

```python
def multiscale_fine_for(sim, uids):
    """Boolean array (aligned with `uids`): True where the agent is a fine
    multiscale agent in ANY HPV genotype module. multiscale_fine is per-module,
    so this unions across modules. Duck-typed (hasattr) to avoid imports."""
    fine = np.zeros(len(uids), dtype=bool)
    for m in sim.diseases.values():
        if hasattr(m, 'multiscale_fine'):
            fine |= np.asarray(m.multiscale_fine[uids])
    return fine
```

- [ ] **Step 2: Write the failing test** (cross-ratio conservation precursor to Task 6). Append to `tests/test_multiscale.py`:

```python
def test_split_does_not_inflate_infections_via_network():
    """Fine agents must not transmit: total infections should be ~scale-
    invariant across ms_agent_ratio (within noise), not inflated by ratio."""
    cfg = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2030,
               dt=0.25, total_pop=1e6, verbose=0)
    def cum_inf(ratio, seed):
        s = hpv.Sim(n_agents=5000, ms_agent_ratio=ratio, rand_seed=seed, **cfg)
        s.run()
        r = s.results.hpv16
        key = 'cum_infections' if 'cum_infections' in r else 'new_infections'
        v = float(np.asarray(r[key]).sum()) if key == 'new_infections' \
            else float(r[key][-1])
        return v * float(s.pars.pop_scale)
    base = np.mean([cum_inf(1, sd) for sd in range(4)])
    ms   = np.mean([cum_inf(10, sd) for sd in range(4)])
    assert abs(ms - base) / base < 0.10, f'infections inflated: {ms:.0f} vs {base:.0f}'
```

- [ ] **Step 3: Run to verify it fails** (fine agents currently transmit): `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py::test_split_does_not_inflate_infections_via_network -v` → FAIL (ms >> base).

- [ ] **Step 4: Exclude fine agents from the network.** In `hpvsim/network.py` `set_network_states`, after `unset = (~self.participant).uids` and the empty-check, drop fine agents so they never become participants (and thus never form partnerships or transmit):

```python
        from .hpv import multiscale_fine_for  # local import avoids cycle
        if len(unset):
            unset = unset[~multiscale_fine_for(self.sim, unset)]
        if not len(unset):
            return
```

(Place this right after the existing `if not len(unset): return`. Fine agents are spawned during disease `set_prognoses`, AFTER `network.step` has run that timestep, so they have `participant=False` and will simply never be activated.)

- [ ] **Step 5: Run the conservation test + measure demographics.** Run the new test. If it now passes, demographics leakage is immaterial — skip Step 6. If infections still inflate, measure births: print `n_alive` and `new_infections` across ratios, and proceed to Step 6.

- [ ] **Step 6 (conditional): Exclude fine agents from demographics denominators.** If Step 5 shows a material leak, gate `AnnualBirths`/`AgeMigration` alive-counts on `~multiscale_fine_for(...)` (mirror v2's `n_alive_level0`/`alive_level0`). Note: the default `hpv.Sim` uses generic `ss.Births`/`ss.Deaths`; if those leak, the equivalence test (Task 6) should set `v2_compat_demographics=True` to use the gateable `AnnualBirths`. Document whichever path you take.

- [ ] **Step 7: Run regression + commit.** `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py tests/test_natural_history.py -v` (all pass; the `ratio=1` no-op is unaffected since the union mask is all-False when nothing is split).

```bash
git add hpvsim/network.py hpvsim/hpv.py tests/test_multiscale.py
git commit -m "multiscale: exclude fine agents from network (+demographics if needed)"
```

---

## Task 5: Multi-genotype correctness

**Files:**
- Test: `tests/test_multiscale.py`

**Why:** A fine agent spawned by one genotype module must have coherent state in *all* genotype modules, and the cross-genotype cancer-cancellation (`_cancel_other_genotype_progression_for`, `hpv.py:375`) must still hold.

- [ ] **Step 1: Write the failing test**

```python
def test_multigenotype_split_keeps_modules_consistent():
    """With 2 genotypes + splitting, all module arrays match people length and
    no agent is cancerous in two genotypes at once."""
    sim = hpv.Sim(n_agents=4000, genotypes=['hpv16', 'hpv18'],
                  ms_agent_ratio=8, location='nigeria',
                  start=1990, stop=2030, dt=0.25, rand_seed=1, verbose=0)
    sim.run()
    n = len(sim.people.age.raw)
    g16, g18 = sim.diseases.hpv16, sim.diseases.hpv18
    assert len(g16.cancerous.raw) == len(g18.cancerous.raw) == n
    both = np.asarray(g16.cancerous.raw) & np.asarray(g18.cancerous.raw)
    assert both.sum() == 0, 'no agent may have invasive cancer in two genotypes'
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py::test_multigenotype_split_keeps_modules_consistent -v`
Expected: If it fails, the split is leaving inconsistent cross-module state — fix by ensuring `grow()` happens once per step and the cancer-cancellation runs for new agents too. If it passes immediately, keep it as a regression guard.

- [ ] **Step 3: Fix if failing**

If the dual-cancer assertion fails, ensure `_cancel_other_genotype_progression_for` is invoked for newly-spawned cancer agents in `step_state` (it already runs on `to_cancerous`, which will include fine agents once they transition — verify, and if a fine agent is spawned already-cancerous in two genotypes, gate splitting to one genotype per agent per step). Also change the re-split guard in `_multiscale_split` from `self.multiscale_fine[...]` to the cross-module union (`multiscale_fine_for(self.sim, ...)` added in Task 4b) so a genotype does not re-split an agent already made fine by another genotype (avoids compounding scale shrink).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py::test_multigenotype_split_keeps_modules_consistent -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_multiscale.py
git commit -m "multiscale: multi-genotype consistency under splitting"
```

---

## Task 6: Internal-equivalence acceptance gate

**Files:**
- Create: `tests/test_multiscale_equivalence.py`

**Why:** This is the acceptance bar from the spec: `ms_agent_ratio=N` at few agents reproduces `ms_agent_ratio=1` at many agents on cancer statistics, in people-space, multi-seed. Reuses the spike's control structure (`spike/multiscale_spike.py`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiscale_equivalence.py
"""Internal-equivalence acceptance gate for multiscale agents (slow)."""
import numpy as np
import pytest
import hpvsim as hpv

CFG = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2060,
           dt=0.25, total_pop=1e6, verbose=0)
SEEDS = range(10)


def _total_cancers_people(sim):
    res = sim.results.hpv16
    return float(np.asarray(res.new_cancers).sum()) * float(sim.pars.pop_scale)


def _mean_over_seeds(n_agents, ratio):
    vals = []
    for sd in SEEDS:
        s = hpv.Sim(n_agents=n_agents, ms_agent_ratio=ratio, rand_seed=sd, **CFG)
        s.run()
        vals.append(_total_cancers_people(s))
    return np.array(vals)


@pytest.mark.slow
def test_multiscale_matches_single_scale_mean():
    """Multiscale (N=4000, ratio=12) reproduces single-scale (N=40000) total
    cancers in people-space within 5% on the seed mean."""
    single = _mean_over_seeds(40000, 1)
    multi  = _mean_over_seeds(4000, 12)
    rel_bias = abs(multi.mean() - single.mean()) / single.mean()
    assert rel_bias < 0.05, f'multiscale mean off by {rel_bias:.1%}'


@pytest.mark.slow
def test_multiscale_reduces_variance_at_equal_agents():
    """At equal agent count, ratio>1 lowers cancer-incidence variance vs ratio=1."""
    base = _mean_over_seeds(4000, 1)
    ms   = _mean_over_seeds(4000, 12)
    assert ms.std(ddof=1) < base.std(ddof=1), 'multiscale should reduce variance'
```

- [ ] **Step 2: Run the gate (expect bias > 5% initially)**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale_equivalence.py -m slow -v`
Expected: Likely FAIL on `test_multiscale_matches_single_scale_mean` until the Task-4 accounting is tuned (this is the known hard part from the spike).

- [ ] **Step 3: Debug accounting to pass the gate**

This is the central correctness work. Use these levers, re-running the gate after each:
- **Extras resample their own `p_cancer`.** Task 4 (ratified) currently reuses the source agent's `p_cancer` for the spawned extras instead of resampling a fresh `dur_cin`→`p_cancer` per extra (v2 resamples). If equivalence is biased, switch the extras to resample their own `dur_cin` (already sampled as `dur_cin_new`) → `p_cancer` (already done) — verify the extra draw uses the per-extra probability, not the source's. This is the first lever to try.
- **Confirm Task 4b network/demographics exclusion is in effect** (fine agents must not transmit); the `test_split_does_not_inflate_infections_via_network` gate should already pass before attempting cancer-total equivalence.
- Verify conservation: instrument a single run to print summed `people.scale` over CIN/cancer agents vs the unsplit baseline; the per-coarse-agent cancer mass must equal `ratio × (1/ratio) × P(cancer)`.
- Check the `ti_cancerous` scheduling for new fine agents is applied exactly once (Task-4 note).
- Confirm new fine agents are not double-counted by `HPVTotal` or re-split on a later step (the `multiscale_fine` guard).
- Compare against the spike's controls (`spike/SPIKE_FINDINGS.md`): the `ratio=1` low-agent case should sit within a few percent; only the splitting should change variance, not the mean.

- [ ] **Step 4: Run the gate to verify it passes**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale_equivalence.py -m slow -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_multiscale_equivalence.py hpvsim/hpv.py
git commit -m "multiscale: internal-equivalence acceptance gate (+ accounting fixes)"
```

---

## Task 7: Docs, changelog, migration-plan status

**Files:**
- Modify: `MIGRATION_PLAN.md` (lines ~31, ~358)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update MIGRATION_PLAN status**

Change the two "Multiscale modeling — Unscheduled; low priority" lines (`MIGRATION_PLAN.md:31` and `:358`) to reflect that multiscale is now implemented via `ms_agent_ratio`, referencing the spec and this plan.

- [ ] **Step 2: Add a CHANGELOG entry**

Add an entry describing the new `ms_agent_ratio` parameter, scale-weighted cancer/death/population results, and the internal-equivalence gate.

- [ ] **Step 3: Verify full suite**

Run: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale.py tests/test_natural_history.py -v`
and the slow gate: `PYTHONPATH=$(pwd) python -m pytest tests/test_multiscale_equivalence.py -m slow -v`
Expected: all PASS; no regression in natural-history smoke tests.

- [ ] **Step 4: Commit**

```bash
git add MIGRATION_PLAN.md CHANGELOG.md
git commit -m "multiscale: document feature, update migration-plan status"
```

---

## Self-review notes

- **Spec coverage:** §6.1 param → Task 1; §6.2 spawning hook → Task 4; §6.3 results audit → Tasks 2 (HPV tallies), 3 (HPVTotal), and verified-already for `AgeResults`; §7 acceptance criteria 1 (no-op identity) → Task 1, criterion 2 (equivalence) → Task 6, criterion 3 (variance) → Task 6, criterion 4 (audit) → Tasks 2/3, criterion 5 (multi-genotype) → Task 5; §8 risks (accounting, audit, multi-genotype, memory) → Tasks 6, 2/3, 5, and the memory note in the spec.
- **Known soft spot:** Task 4's exact `ti_cancerous` scheduling for new fine agents and Task 6's accounting tuning are the genuine implementation risk identified by the spike; the equivalence gate (Task 6) is the oracle that proves them correct. The plan deliberately front-loads the cheap, certain changes (Tasks 1-3) before the hard accounting (Tasks 4, 6).
- **Type consistency:** `ms_agent_ratio` (int), `multiscale_fine` (BoolArr), `new_cancers`/`new_cancer_deaths` (float after Task 2) used consistently across tasks.
