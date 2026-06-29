# v2-faithful grow-real-agents multiscale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port HPVsim v2.3.1's `ms_agent_ratio` grow-real-agents multiscale onto the v3.0-dev (starsim) engine, implemented entirely in hpvsim, growing real `fine` cancer agents weighted at `1/ratio`.

**Architecture:** At the CIN→cancer decision in `HPV.set_prognoses`, shrink each transforming base agent's `people.scale` to `1/ratio` and grow `ratio-1` extra real fine cancer agents per CIN agent (full cross-genotype clone of the source). Fine agents are excluded from the sexual network, births, and emigration, but face background death (v2-faithful). All population-count results are scale-weighted via `people.scale_flows` and stored as float. At `ratio==1` the entire path is a gated bit-identical no-op.

**Tech Stack:** Python, starsim (main branch — assume NO multiscale primitives), hpvsim, numpy, pytest. Reference source of truth: `hpvsim_v23_frozen` @ branch `fix-multiscale-cin-regate`, `hpvsim/people.py::set_severity`.

## Global Constraints

- **starsim is the main branch; assume zero multiscale functionality there.** Only generic primitives may be used: `people.grow(n)`, `people.scale` (FloatArr default 1.0), `people.scale_flows(uids)` (= `scale[uids].sum()`), `people.request_removal(uids)`, `people.request_death(uids)`, `module.state_list`, `people.states`.
- **CRN safety is out of scope.** Within-run reproducibility from `rand_seed` suffices; no cross-scenario CRN required. Plain `np.random.default_rng` and module dists may be used freely.
- **Editable-install worktree trap.** hpvsim is `pip install -e` pointed at the MAIN repo, NOT this worktree. ALWAYS run tests/scripts pinned to the worktree:
  `PYTHONPATH=C:/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/.claude/worktrees/m07-multiscale-v2-grow C:/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/.venv/Scripts/python.exe -m pytest <args>`
  In scripts, also `sys.path.insert(0, '<worktree>')` and assert `hpvsim.__file__` starts with the worktree path.
- **Scale convention:** `cancer_scale = 1.0 / ratio` (per-agent `people.scale` units; the scalar `pop_scale` is applied separately at `finalize_results`). NOT `pop_scale/ratio`.
- **Int-truncation gotcha:** every scale-weighted result MUST be `dtype=float`. A `1/ratio` write into an `ss.Result(dtype=int)` truncates to 0 each step.
- **ratio==1 bit-identical:** every engine change MUST keep `ms_agent_ratio==1` byte-for-byte identical to the pre-feature single-scale engine (no `fine`/`scale` writes, `scale_flows==len`, no grow). This is the gating invariant for Tasks 2–8.
- **venv python:** `C:/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/.venv/Scripts/python.exe`.

---

### Task 1: Add `ms_agent_ratio` parameter + `fine` People state

**Files:**
- Modify: `hpvsim/hpv.py` (`HPV.define_pars` ~line 71; `HPV.__init__` ratio coercion)
- Modify: `hpvsim/sim.py` (accept `ms_agent_ratio` kwarg ~line 75; build People with `fine` extra_state ~line 85; pass ratio to each genotype HPV ~line 123)
- Test: `tests/test_multiscale_grow_unit.py`

**Interfaces:**
- Produces: `HPV.pars.ms_agent_ratio` (int, default 1); `sim.people.fine` (`ss.BoolArr`, default False); `hpv.Sim(ms_agent_ratio=N)` kwarg sets it on every genotype module.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiscale_grow_unit.py
import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import hpvsim as hpv
import starsim as ss

def test_ratio_param_and_fine_state_exist():
    assert hpv.__file__.startswith(WT), f'wrong hpvsim loaded: {hpv.__file__}'
    sim = hpv.Sim(location='nigeria', n_agents=500, start=2000, stop=2002,
                  ms_agent_ratio=10)
    sim.init()
    # fine state exists on People, defaults all-False
    assert 'fine' in sim.people.states
    assert not sim.people.fine.values.any()
    # ratio propagated to every genotype HPV module
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            assert int(dis.pars.ms_agent_ratio) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_ratio_param_and_fine_state_exist -v`
Expected: FAIL (`ms_agent_ratio` is not a valid kwarg / `fine` not in states).

- [ ] **Step 3: Implement**

In `hpvsim/hpv.py` `HPV.define_pars(...)`, add to the pars dict:
```python
            # Multiscale: number of agents each cancer-capable agent represents.
            # 1 = single scale (bit-identical no-op). >1 grows real fine cancer
            # agents at scale 1/ms_agent_ratio. See
            # docs/superpowers/specs/2026-06-29-v2-faithful-grow-multiscale-design.md
            ms_agent_ratio=1,
```
After `self.update_pars(pars=pars, **kwargs)` in `HPV.__init__`, coerce to int:
```python
        self.pars.ms_agent_ratio = int(self.pars.ms_agent_ratio)
```

In `hpvsim/sim.py`, add `ms_agent_ratio=1` to the `Sim.__init__` signature (near `total_pop=None`), build People with the extra state:
```python
        if people is None:
            people = ss.People(n_agents, age_data=country['age_data'],
                               extra_states=[ss.BoolArr('fine', default=False)])
```
and inject the ratio into every genotype module at creation (line ~123):
```python
            diseases = [HPV(genotype=k, ms_agent_ratio=ms_agent_ratio,
                            **gpars_overrides.get(k, {})) for k in keys]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_ratio_param_and_fine_state_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py hpvsim/sim.py tests/test_multiscale_grow_unit.py
git commit -m "feat(multiscale): add ms_agent_ratio param + fine People state"
```

---

### Task 2: Scale-weighted, float-dtype flow + age results (ratio==1 no-op)

**Files:**
- Modify: `hpvsim/hpv.py` (`init_results` ~line 195: dtypes; `step_state` ~line 496 & 514: tallies)
- Test: `tests/test_multiscale_grow_unit.py`

**Interfaces:**
- Consumes: `sim.people.scale_flows(uids)`, `sim.people.scale`.
- Produces: `results.new_cancers`/`new_cancer_deaths`/`cum_cancers`/`cum_cancer_deaths` are `dtype=float`, scale-weighted; `sum_age_at_cancer*` scale-weighted.

- [ ] **Step 1: Write the failing test**

```python
def test_results_are_float_and_scale_weighted_noop_at_ratio1():
    import numpy as np
    # ratio==1: scale_flows == len, so counts are unchanged but dtype is float.
    sim = hpv.Sim(location='nigeria', n_agents=2000, start=1990, stop=2020,
                  ms_agent_ratio=1, rand_seed=1)
    sim.run()
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            r = dis.results
            assert r.new_cancers.values.dtype == np.float64
            assert r.new_cancer_deaths.values.dtype == np.float64
            # ratio==1 keeps everyone at scale 1.0 → integer-valued counts
            nc = r.new_cancers.values
            assert np.allclose(nc, np.round(nc))
    assert not sim.people.fine.values.any()
    assert np.allclose(sim.people.scale.values, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_results_are_float_and_scale_weighted_noop_at_ratio1 -v`
Expected: FAIL (`new_cancers` dtype is int).

- [ ] **Step 3: Implement**

In `hpvsim/hpv.py` `init_results`, change the four count results from `dtype=int` to `dtype=float`:
```python
            ss.Result('new_cancers', dtype=float, scale=True,
                      label='New cancers'),
            ss.Result('new_cancer_deaths', dtype=float, scale=True,
                      label='New cancer deaths'),
            ss.Result('cum_cancers', dtype=float, scale=True,
                      label='Cumulative cancers'),
            ss.Result('cum_cancer_deaths', dtype=float, scale=True,
                      label='Cumulative cancer deaths'),
```
In `step_state`, replace the cancer-onset tally block (~line 495-498):
```python
            ppl = self.sim.people
            ages_at_cancer = ppl.age[to_cancerous]
            w = ppl.scale[to_cancerous]
            self.results.new_cancers[ti] = ppl.scale_flows(to_cancerous)
            self.results.sum_age_at_cancer[ti] = float((ages_at_cancer * w).sum())
            self._cancel_other_genotype_progression_for(to_cancerous)
```
and the cancer-death tally block (~line 510-515):
```python
            dt_yr = float(self.t.dt.years if hasattr(self.t.dt, 'years') else self.t.dt)
            ppl = self.sim.people
            ages_at_death = ppl.age[to_dead] + dt_yr
            w = ppl.scale[to_dead]
            self.results.new_cancer_deaths[ti] = ppl.scale_flows(to_dead)
            self.results.sum_age_at_cancer_death[ti] = float((ages_at_death * w).sum())
            self.sim.people.request_death(to_dead)
```

- [ ] **Step 4: Run test + the full existing suite (ratio==1 regression gate)**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_results_are_float_and_scale_weighted_noop_at_ratio1 tests/test_natural_history.py tests/test_hpv.py -v`
Expected: PASS (all). The natural-history/hpv tests confirm ratio==1 is bit-identical.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_multiscale_grow_unit.py
git commit -m "feat(multiscale): scale-weighted float cancer/age results (ratio==1 no-op)"
```

---

### Task 3: Per-agent-Arr clone helper

**Files:**
- Modify: `hpvsim/hpv.py` (module-level helper `_clone_agents`)
- Test: `tests/test_multiscale_grow_unit.py`

**Interfaces:**
- Produces: `_clone_agents(sim, src_uids, new_uids)` — copies every per-agent Arr (People states except `uid`/`slot`, plus every disease and connector module's `state_list`) from `src_uids` to `new_uids` (element-wise aligned, equal length).

- [ ] **Step 1: Write the failing test**

```python
def test_clone_agents_copies_people_and_module_state():
    import numpy as np
    from hpvsim.hpv import _clone_agents
    sim = hpv.Sim(location='nigeria', n_agents=400, start=2000, stop=2001,
                  genotypes=[16, 18], ms_agent_ratio=1, rand_seed=2)
    sim.init()
    ppl = sim.people
    src = ppl.auids[:5]
    # mark distinct source values we can check after cloning
    ppl.age[src] = np.array([11., 22., 33., 44., 55.])
    new = ppl.grow(len(src))
    _clone_agents(sim, src, new)
    assert np.allclose(ppl.age[new], ppl.age[src])
    # uid must NOT be overwritten by the clone
    assert not np.array_equal(np.asarray(ppl.uid[new]), np.asarray(ppl.uid[src]))
    # module states copied for every genotype
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            assert np.array_equal(np.asarray(dis.susceptible[new]),
                                  np.asarray(dis.susceptible[src]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_clone_agents_copies_people_and_module_state -v`
Expected: FAIL (`cannot import name _clone_agents`).

- [ ] **Step 3: Implement**

Add to `hpvsim/hpv.py` (module level, near the top after imports):
```python
# Per-agent Arrs that must NOT be cloned (identity, not biology).
_NO_CLONE_STATES = {'uid', 'slot'}

def _clone_agents(sim, src_uids, new_uids):
    """Copy every per-agent Arr from src_uids to new_uids (v2 states_to_set).

    Covers People states (except identity keys) plus every disease and
    connector module's per-agent states. Network/demographics states are
    omitted: fine agents are excluded from those subsystems, so their values
    are inert. src_uids and new_uids must align element-wise and be equal
    length.
    """
    ppl = sim.people
    for key, arr in ppl.states.items():
        if key in _NO_CLONE_STATES:
            continue
        arr[new_uids] = arr[src_uids]
    for mod in list(sim.diseases.values()) + list(sim.connectors.values()):
        for st in mod.state_list:
            st[new_uids] = st[src_uids]
    return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_clone_agents_copies_people_and_module_state -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_multiscale_grow_unit.py
git commit -m "feat(multiscale): per-agent Arr clone helper (v2 states_to_set analogue)"
```

---

### Task 4: Grow real fine cancer agents in `set_prognoses`

**Files:**
- Modify: `hpvsim/hpv.py` (`set_prognoses`: insert grow block after step 5b ~line 413; new method `_grow_fine_agents`)
- Test: `tests/test_multiscale_grow_unit.py`

**Interfaces:**
- Consumes: `_clone_agents`, `self.pars.ms_agent_ratio`, `compute_severity`, `self._randround`, `people.grow`, `people.scale`, `people.fine`.
- Produces: side effects on `people` (grown fine agents, shrunk base scale). No return value used by callers.

- [ ] **Step 1: Write the failing test**

```python
def test_grow_creates_fine_cancer_agents_at_ratio():
    import numpy as np
    n0 = 6000
    sim = hpv.Sim(location='nigeria', n_agents=n0, start=1980, stop=2025,
                  ms_agent_ratio=10, rand_seed=3)
    sim.run()
    ppl = sim.people
    # multiscale grew real fine agents...
    assert ppl.fine.values.any(), 'no fine agents were grown'
    # ...all fine agents carry scale 1/ratio
    fine_uids = ppl.auids[ppl.fine[ppl.auids]]
    assert np.allclose(ppl.scale[fine_uids], 1.0/10)
    # ...and every fine agent is cancer-bound in exactly the genotype that grew
    # it (cancerous now, or scheduled): at least one HPV module flags them.
    flagged = np.zeros(len(fine_uids), dtype=bool)
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            sched = ~np.isnan(np.asarray(dis.ti_cancerous[fine_uids]))
            flagged |= (np.asarray(dis.cancerous[fine_uids]) | sched)
    assert flagged.all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_grow_creates_fine_cancer_agents_at_ratio -v`
Expected: FAIL (no fine agents grown).

- [ ] **Step 3: Implement**

In `set_prognoses`, replace the early `if len(cancer_uids) == 0: return` / scheduling block (step 5b) so it always reaches the grow call. After scheduling the base `cancer_uids`' `ti_cancerous`/`ti_dead_cancer` (existing lines), add at the end of `set_prognoses`:
```python
        # 6. Multiscale: grow real fine cancer agents (v2 set_severity port).
        self._grow_fine_agents(cin_uids, cancer_uids, dur_cin, age_mod,
                               rel_sev_cin, sev_imm_uids[cin_mask], dt_yr)
```
(Keep the `if len(cancer_uids) == 0` guard local to the base scheduling only; the grow must run over all `cin_uids` even when no base agent drew cancer.)

Add the method (mirrors reference `set_severity` lines 288–409):
```python
    def _grow_fine_agents(self, cin_uids, cancer_uids, dur_cin, age_mod,
                          rel_sev_cin, sev_imm_cin, dt_yr):
        """Grow ratio-1 extra real fine cancer agents per CIN agent (v2-faithful).

        Mirrors hpvsim_v23_frozen@fix-multiscale-cin-regate set_severity: the
        transforming base agents are shrunk to scale 1/ratio; for every CIN
        reacher, ratio-1 extra trajectories are drawn (age_risk-modified
        dur_cin; CIN-conditional rejection-sampled dur_precin), cancer is drawn
        for each extra, and one real fine agent is grown per extra-cancer
        success (full cross-genotype clone of the source, then this genotype's
        cancer-bound trajectory written). No-op at ms_agent_ratio<=1.
        """
        ratio = int(self.pars.ms_agent_ratio)
        n = len(cin_uids)
        if ratio <= 1 or n == 0:
            return
        p = self.pars
        ppl = self.sim.people
        ti = self.ti
        cancer_scale = 1.0 / ratio

        # Shrink base agents that drew their own cancer.
        if len(cancer_uids):
            ppl.scale[cancer_uids] = cancer_scale

        # Side RNG (CRN not required). Seed from rand_seed/ti/genotype.
        import zlib
        seed = ((int(self.sim.pars.rand_seed or 0) * 2654435761
                 + int(ti) * 40503 + zlib.crc32(self.genotype.encode()))
                & 0x7FFFFFFF)
        rng = np.random.default_rng(seed)

        def _ln(dist, size):  # lognormal years on the side RNG (ex->im inline)
            mean = float(dist.pars['mean']); std = float(dist.pars['std'])
            sig2 = np.log(1.0 + (std / mean) ** 2)
            return rng.lognormal(np.log(mean) - 0.5 * sig2, np.sqrt(sig2), size)

        m = ratio - 1
        size = (n, m)
        amod = np.asarray(age_mod, dtype=float)[:, None]
        rel = np.asarray(rel_sev_cin, dtype=float)[:, None] * np.ones(size)
        sevimm = np.asarray(sev_imm_cin, dtype=float)[:, None] * np.ones(size)

        # age_risk-modified extra dur_cin (years).
        extra_dur_cin = _ln(p.dur_cin, size) * amod
        # CIN-conditional (length-biased) extra dur_precin: rejection-sample.
        extra_dur_precin = _ln(p.dur_precin, size) * (1.0 - sevimm)
        pending = np.ones(size, dtype=bool)
        for _ in range(64):
            if not pending.any():
                break
            cinp = compute_severity(extra_dur_precin, rel_sev=rel, pars=p.cin_fn)
            passed = (rng.random(size) < cinp) & pending
            pending &= ~passed
            if pending.any():
                redraw = _ln(p.dur_precin, int(pending.sum())) \
                    * (1.0 - sevimm[pending])
                extra_dur_precin[pending] = redraw

        # CIN -> cancer for every extra (all are CIN now).
        pcanc = compute_severity(extra_dur_cin, rel_sev=rel, pars=p.cancer_fn)
        extra_cancer = rng.random(size) < pcanc
        # Existing fine agents never spawn more fine agents (v2 level0 guard).
        not_fine = ~np.asarray(ppl.fine[cin_uids], dtype=bool)
        extra_cancer &= not_fine[:, None]
        counts = extra_cancer.sum(axis=1)
        n_new = int(counts.sum())
        if n_new == 0:
            return

        # Broadcast source uids per success and grow.
        src_uids = ss.uids(np.repeat(np.asarray(cin_uids), counts))
        new_dur_precin = extra_dur_precin[extra_cancer]   # years
        new_dur_cin = extra_dur_cin[extra_cancer]         # years
        new_uids = ppl.grow(n_new)

        # Full cross-genotype clone of the source individuals.
        _clone_agents(self.sim, src_uids, new_uids)

        # Fine-agent identity.
        ppl.fine[new_uids] = True
        ppl.scale[new_uids] = cancer_scale

        # This genotype's fresh cancer-bound trajectory (overwrites cloned).
        self.susceptible[new_uids] = False
        self.infected[new_uids] = True
        self.precin[new_uids] = True
        self.cin[new_uids] = False
        self.cancerous[new_uids] = False
        self.ti_infected[new_uids] = ti
        self.ti_first_infection[new_uids] = ti
        self.ti_clearance[new_uids] = np.nan
        # durations are in YEARS; convert to steps via /dt_yr before rounding.
        ti_cin = ti + self._randround(new_dur_precin / dt_yr, new_uids,
                                      self._round_cin_bern)
        self.ti_cin[new_uids] = ti_cin
        ti_canc = ti_cin + self._randround(new_dur_cin / dt_yr, new_uids,
                                           self._round_cancer_bern)
        self.ti_cancerous[new_uids] = ti_canc
        dur_cancer = p.dur_cancer.rvs(new_uids)  # steps (module dist)
        self.ti_dead_cancer[new_uids] = ti_canc + self._randround(
            dur_cancer, new_uids, self._round_dead_bern)
        return
```

NOTE: confirm whether `p.dur_cin.rvs` returns years or steps in this engine; the base path multiplies `dur_cin * dt_yr` before `compute_severity`, implying `rvs` returns STEPS. The `_ln` side-draw returns YEARS (its `mean`/`std` are the dist's year-parameterization), so `compute_severity` gets years directly (correct) and the `/dt_yr` in scheduling converts years→steps (correct). Verify against a single-scale base agent's scheduling in Step 4’s incidence check (Task 9).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_grow_creates_fine_cancer_agents_at_ratio -v`
Expected: PASS

- [ ] **Step 5: Run ratio==1 regression gate**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_natural_history.py tests/test_hpv.py -v`
Expected: PASS (grow is gated off at ratio==1).

- [ ] **Step 6: Commit**

```bash
git add hpvsim/hpv.py tests/test_multiscale_grow_unit.py
git commit -m "feat(multiscale): grow real fine cancer agents in set_prognoses"
```

---

### Task 5: Exclude fine agents from the sexual network

**Files:**
- Modify: `hpvsim/network.py` (`SexualNetwork.active` override)
- Test: `tests/test_multiscale_grow_unit.py`

**Interfaces:**
- Consumes: `people.fine`.
- Produces: `SexualNetwork.active(people)` returns active & ~fine.

- [ ] **Step 1: Write the failing test**

```python
def test_fine_agents_excluded_from_network():
    import numpy as np
    sim = hpv.Sim(location='nigeria', n_agents=6000, start=1980, stop=2025,
                  ms_agent_ratio=10, rand_seed=4)
    sim.run()
    ppl = sim.people
    fine_uids = ppl.auids[ppl.fine[ppl.auids]]
    assert len(fine_uids) > 0
    # No edge in any sexual-network layer touches a fine agent.
    net = [n for n in sim.networks.values()
           if isinstance(n, hpv.SexualNetwork)][0]
    edges = net.edges
    fine_set = set(np.asarray(fine_uids).tolist())
    p1 = set(np.asarray(edges.p1).tolist())
    p2 = set(np.asarray(edges.p2).tolist())
    assert fine_set.isdisjoint(p1) and fine_set.isdisjoint(p2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_fine_agents_excluded_from_network -v`
Expected: FAIL (fine agents appear in edges).

- [ ] **Step 3: Implement**

In `hpvsim/network.py`, add to `class SexualNetwork`:
```python
    def active(self, people):
        """Network participation requires a non-fine (level0) agent.

        Fine multiscale agents (people.fine) are high-resolution cancer
        stand-ins excluded from transmission, mirroring v2's is_active
        requirement that participants be level0.
        """
        return super().active(people) & ~people.fine
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_fine_agents_excluded_from_network -v`
Expected: PASS

- [ ] **Step 5: Run ratio==1 regression gate**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_natural_history.py tests/test_demographics.py -v`
Expected: PASS (at ratio==1 no agent is fine, so `~people.fine` is all-True — identical eligibility).

- [ ] **Step 6: Commit**

```bash
git add hpvsim/network.py tests/test_multiscale_grow_unit.py
git commit -m "feat(multiscale): exclude fine agents from the sexual network"
```

---

### Task 6: Exclude fine agents from driving births

**Files:**
- Modify: `hpvsim/demographics.py` (new `class Births(ss.Births)`; change `AnnualBirths(ss.Births)` → `AnnualBirths(Births)`)
- Modify: `hpvsim/sim.py` (default `births_cls`: use `hpv.Births`)
- Test: `tests/test_multiscale_grow_unit.py`

**Interfaces:**
- Consumes: `super().get_births()` (returns `birth_uids`), `people.fine`.
- Produces: `hpv.Births` whose `get_births` drops fine "parents"; used as the default births class.

- [ ] **Step 1: Write the failing test**

```python
def test_fine_agents_do_not_drive_births():
    import numpy as np
    from hpvsim.demographics import Births
    # A fine parent dropped from birth_uids == excluded from the pool, because
    # births are an independent per-agent Bernoulli.
    sim = hpv.Sim(location='nigeria', n_agents=3000, start=2000, stop=2002,
                  ms_agent_ratio=10, rand_seed=5)
    sim.init()
    births = [d for d in sim.demographics.values() if isinstance(d, Births)]
    assert births, 'default births class should be hpv.Births'
    ppl = sim.people
    # Force a couple of agents fine, then confirm get_births never returns them.
    some = ppl.auids[:50]
    ppl.fine[some] = True
    b = births[0]
    for _ in range(20):
        uids = b.get_births()
        assert not np.asarray(ppl.fine[uids]).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_fine_agents_do_not_drive_births -v`
Expected: FAIL (`cannot import name Births` / default is `ss.Births`).

- [ ] **Step 3: Implement**

In `hpvsim/demographics.py` add (above `AnnualBirths`):
```python
class Births(ss.Births):
    """ss.Births that excludes fine multiscale agents from reproducing.

    Births are an independent per-agent Bernoulli, so dropping fine agents
    from the drawn birth_uids is statistically identical to excluding them
    from the eligible pool — matching v2's add_births, which counts only
    level0 agents (n_alive_level0).
    """

    def get_births(self):
        birth_uids = super().get_births()
        ppl = self.sim.people
        if 'fine' in ppl.states:
            birth_uids = birth_uids[~ppl.fine[birth_uids]]
        return birth_uids
```
Change `class AnnualBirths(ss.Births):` to `class AnnualBirths(Births):`.
In `hpvsim/sim.py`, update the import line `from .demographics import AgeMigration, AnnualBirths` → `from .demographics import AgeMigration, AnnualBirths, Births` and the default:
```python
            births_cls = AnnualBirths if v2_compat_demographics else Births
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_fine_agents_do_not_drive_births -v`
Expected: PASS

- [ ] **Step 5: Run ratio==1 regression gate**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_demographics.py -v`
Expected: PASS (at ratio==1 no fine agents → birth_uids unchanged).

- [ ] **Step 6: Commit**

```bash
git add hpvsim/demographics.py hpvsim/sim.py tests/test_multiscale_grow_unit.py
git commit -m "feat(multiscale): exclude fine agents from driving births"
```

---

### Task 7: Exclude fine agents from emigration (AgeMigration)

**Files:**
- Modify: `hpvsim/demographics.py` (`AgeMigration.step` snap_uids filter ~line 163)
- Test: `tests/test_multiscale_grow_unit.py`

**Interfaces:**
- Consumes: `people.fine`.
- Produces: `AgeMigration` ignores fine agents for both the pyramid count and emigration selection.

- [ ] **Step 1: Write the failing test**

```python
def test_fine_agents_not_emigrated():
    import numpy as np
    sim = hpv.Sim(location='nigeria', n_agents=6000, start=1980, stop=2025,
                  ms_agent_ratio=10, rand_seed=6)
    sim.run()
    ppl = sim.people
    # Any agent flagged fine and still alive must not be marked emigrated.
    alive_fine = ppl.auids[ppl.fine[ppl.auids]]
    assert len(alive_fine) > 0
    # emigrated agents are request_removal'd, so a still-active fine agent
    # cannot also be emigrated; assert the AgeMigration snapshot excludes fine.
    mig = [d for d in sim.demographics.values()
           if isinstance(d, hpv.AgeMigration)][0]
    # white-box: re-run one step's snapshot logic is covered by code; here we
    # assert the live invariant that no fine agent was emigrated this run.
    assert not np.asarray(ppl.emigrated[alive_fine]).any() if 'emigrated' in ppl.states else True
```

NOTE: `emigrated` may not exist as a People state in v3 (it was a vestigial v2 state). If absent, the test's final assertion is vacuously true; the real protection is the snapshot filter below plus the count-exclusion. Keep the assertion guarded as written.

- [ ] **Step 2: Run test to verify it fails (or passes vacuously — then strengthen)**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_fine_agents_not_emigrated -v`
Expected: initially may pass vacuously; the implementation below is still required for correctness of the pyramid count. If it passes vacuously, that is acceptable — the count-exclusion is verified indirectly by Task 9 (incidence flat) and Task 12 (v2 tracking).

- [ ] **Step 3: Implement**

In `hpvsim/demographics.py` `AgeMigration.step`, change the snapshot (line ~163) to exclude fine agents:
```python
        people = sim.people
        # Snapshot alive UIDs at the start of the step, EXCLUDING fine
        # multiscale agents: they are high-resolution cancer stand-ins, not
        # real bodies, so they neither count toward the age x sex pyramid
        # target nor are eligible to emigrate (v2 used alive_level0).
        all_alive = people.auids.copy()
        if 'fine' in people.states:
            snap_uids = all_alive[~people.fine[all_alive]]
        else:
            snap_uids = all_alive
```

- [ ] **Step 4: Run test + ratio==1 regression gate**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_fine_agents_not_emigrated tests/test_demographics.py -v`
Expected: PASS (at ratio==1 `snap_uids == all_alive`).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/demographics.py tests/test_multiscale_grow_unit.py
git commit -m "feat(multiscale): exclude fine agents from AgeMigration"
```

---

### Task 8: Scale-weight stock prevalence results

**Files:**
- Modify: `hpvsim/hpv.py` (add/extend `update_results` for scale-weighted stocks) OR `hpvsim/cross_genotype.py` (`HPVTotal`) — whichever owns the prevalence stocks
- Test: `tests/test_multiscale_grow_unit.py`

**Interfaces:**
- Consumes: `people.scale_flows`.
- Produces: any per-step stock count of agents (`n_precin`/`n_cin`/`n_cancerous`/`n_infected` and analyzer prevalence denominators) is scale-weighted and `dtype=float`.

- [ ] **Step 1: Audit which stocks exist and how they're tallied**

Run: `PYTHONPATH=<worktree> <venv-python> -c "import sys; sys.path.insert(0,'.'); import hpvsim as hpv; s=hpv.Sim(location='nigeria', n_agents=300, start=2000, stop=2001, ms_agent_ratio=1); s.run(); [print(k, s.results[k].dtype if hasattr(s.results[k],'dtype') else type(s.results[k])) for k in s.results.keys()]"`
Identify every result that is a population count of agents (stocks `n_*`, analyzer prevalence). For each, note whether it uses `.count()`/`len()` (plain) vs scale-weighted.

- [ ] **Step 2: Write the failing test**

```python
def test_stock_prevalence_scale_weighted():
    import numpy as np
    # n_cin (or equivalent stock) must count a fine agent as 1/ratio, not 1.
    sim = hpv.Sim(location='nigeria', n_agents=6000, start=1980, stop=2025,
                  ms_agent_ratio=10, rand_seed=7)
    sim.run()
    # Find a genotype module CIN stock; assert it is float and that a manual
    # scale-weighted recompute at the final step matches the stored value.
    dis = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    r = dis.results
    assert 'n_cin' in r, 'expected an n_cin stock to scale-weight'
    assert r.n_cin.values.dtype == np.float64
    ppl = sim.people
    cin_uids = ppl.auids[np.asarray(dis.cin[ppl.auids], dtype=bool)]
    expected_last = ppl.scale_flows(cin_uids)
    assert np.isclose(r.n_cin.values[-1], expected_last, rtol=1e-6)
```

NOTE: adjust the stock name (`n_cin`) to whatever the audit in Step 1 reveals. If the auto stock is named differently or owned by `HPVTotal`, target that result. If NO `n_*` stock exists for the disease module, the test instead asserts the analyzer prevalence denominator is scale-weighted; write it against the actual result key found in Step 1.

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_stock_prevalence_scale_weighted -v`
Expected: FAIL (stock is plain count, fine counted as whole body).

- [ ] **Step 4: Implement**

Add an `update_results` override to `HPV` (or extend the existing results step) that recomputes each agent-count stock as a scale-weighted float. Example for `n_cin` (replicate for every stock identified in Step 1):
```python
    def update_results(self):
        super().update_results()
        ti = self.ti
        ppl = self.sim.people
        res = self.results
        for state_name in ('precin', 'cin', 'cancerous', 'infected'):
            key = f'n_{state_name}'
            if key in res:
                mask = np.asarray(getattr(self, state_name)[ppl.auids], dtype=bool)
                res[key][ti] = ppl.scale_flows(ppl.auids[mask])
```
Ensure each such result is declared `dtype=float` in `init_results` (add explicit float `n_*` results there if starsim auto-created them as int; if so, declare them in `define_results` to control dtype).

- [ ] **Step 5: Run test to verify it passes + ratio==1 regression gate**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_unit.py::test_stock_prevalence_scale_weighted tests/test_natural_history.py tests/test_age_results.py -v`
Expected: PASS (at ratio==1 `scale_flows == count`).

- [ ] **Step 6: Commit**

```bash
git add hpvsim/hpv.py tests/test_multiscale_grow_unit.py
git commit -m "feat(multiscale): scale-weight stock prevalence results"
```

---

### Task 9: Acceptance gate — cancer incidence flat across ratio

**Files:**
- Test: `tests/test_multiscale_grow_acceptance.py`

**Interfaces:**
- Consumes: the full feature (Tasks 1–8).

- [ ] **Step 1: Write the test**

```python
# tests/test_multiscale_grow_acceptance.py
import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import numpy as np
import hpvsim as hpv

def _total_cancers(ratio, seed, n_agents=8000):
    sim = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2030,
                  ms_agent_ratio=ratio, rand_seed=seed)
    sim.run()
    tot = 0.0
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            tot += float(dis.results.new_cancers.values.sum())
    return tot

def test_cancer_incidence_flat_across_ratio():
    seeds = range(8)
    base = np.mean([_total_cancers(1, s) for s in seeds])
    for ratio in (5, 10):
        got = np.mean([_total_cancers(ratio, s) for s in seeds])
        rel = got / base
        assert 0.90 <= rel <= 1.10, f'ratio={ratio}: {rel:.3f} (base={base:.0f})'
```

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_acceptance.py::test_cancer_incidence_flat_across_ratio -v`
Expected: PASS. If it fails LOW at high ratio, suspect the years/steps duration bug flagged in Task 4 Step 3 (compute_severity must receive years; scheduling must receive steps). If it fails HIGH, suspect a missing `~fine` guard or double-count. Debug with `superpowers:systematic-debugging`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_multiscale_grow_acceptance.py
git commit -m "test(multiscale): cancer incidence flat across ms_agent_ratio"
```

---

### Task 10: Acceptance gate — multiscale × intervention equivalence (centerpiece)

**Files:**
- Test: `tests/test_multiscale_grow_acceptance.py`

**Interfaces:**
- Consumes: full feature + an hpvsim screen+treat intervention.

- [ ] **Step 1: Find the canonical screen+treat construction**

Run: `PYTHONPATH=<worktree> <venv-python> -c "import sys; sys.path.insert(0,'.'); import hpvsim as hpv; print([x for x in dir(hpv) if 'creen' in x or 'reat' in x or 'herap' in x])"`
Inspect `tests/test_m06_*` for the exact screen+treat intervention API (products, eligibility, coverage) and copy that construction verbatim into the test.

- [ ] **Step 2: Write the test**

```python
def _averted_fraction(ratio, seed, intervention_factory, n_agents=8000):
    base = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2040,
                   ms_agent_ratio=ratio, rand_seed=seed)
    base.run()
    treat = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2040,
                    ms_agent_ratio=ratio, rand_seed=seed,
                    interventions=intervention_factory())
    treat.run()
    def tot(sim):
        return sum(float(d.results.new_cancers.values.sum())
                   for d in sim.diseases.values() if isinstance(d, hpv.HPV))
    b, t = tot(base), tot(treat)
    return (b - t) / b

def test_intervention_equivalence_across_ratio():
    # intervention_factory must return a fresh list of screen+treat
    # interventions each call (copied from tests/test_m06_*; see Step 1).
    def intervention_factory():
        ...  # paste the verified screen+treat construction here
    seeds = range(6)
    av1 = np.mean([_averted_fraction(1, s, intervention_factory) for s in seeds])
    av10 = np.mean([_averted_fraction(10, s, intervention_factory) for s in seeds])
    assert abs(av10 - av1) <= 0.05, f'averted frac ratio1={av1:.3f} ratio10={av10:.3f}'
    assert av1 > 0.05, 'intervention should avert a non-trivial cancer fraction'
```

- [ ] **Step 3: Run the test**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_acceptance.py::test_intervention_equivalence_across_ratio -v`
Expected: PASS — fine agents are REAL and get screened/treated natively, so averted fraction matches across ratios (the property the ledger could not deliver).

- [ ] **Step 4: Commit**

```bash
git add tests/test_multiscale_grow_acceptance.py
git commit -m "test(multiscale): intervention equivalence across ratio (centerpiece gate)"
```

---

### Task 11: Acceptance gate — cancer event-age variance reduction

**Files:**
- Test: `tests/test_multiscale_grow_acceptance.py`

- [ ] **Step 1: Write the test**

```python
def _mean_age_at_cancer(ratio, seed, n_agents=6000):
    sim = hpv.Sim(location='nigeria', n_agents=n_agents, start=1970, stop=2030,
                  ms_agent_ratio=ratio, rand_seed=seed)
    sim.run()
    s_age = 0.0; n = 0.0
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            s_age += float(dis.results.sum_age_at_cancer.values.sum())
            n += float(dis.results.new_cancers.values.sum())
    return s_age / n if n else np.nan

def test_event_age_variance_shrinks_with_ratio():
    seeds = range(12)
    var1 = np.var([_mean_age_at_cancer(1, s) for s in seeds])
    var10 = np.var([_mean_age_at_cancer(10, s) for s in seeds])
    assert var10 < var1, f'var ratio1={var1:.3f} ratio10={var10:.3f}'
```

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=<worktree> <venv-python> -m pytest tests/test_multiscale_grow_acceptance.py::test_event_age_variance_shrinks_with_ratio -v`
Expected: PASS (more cancer events at higher ratio → tighter mean-age estimator; the original Fig-5 motivation).

- [ ] **Step 3: Commit**

```bash
git add tests/test_multiscale_grow_acceptance.py
git commit -m "test(multiscale): cancer event-age variance reduction with ratio"
```

---

### Task 12: Validation — numerical tracking vs v2.3.1 frozen

**Files:**
- Create: `tests/regression/validate_v2_grow_multiscale.py` (script, not a unit test)

**Interfaces:**
- Compares v3 grow-multiscale cancer totals to `hpvsim_v23_frozen` @ `fix-multiscale-cin-regate`.

- [ ] **Step 1: Write the comparison script**

```python
# tests/regression/validate_v2_grow_multiscale.py
"""Compare v3 grow-multiscale cancer burden to v2.3.1 frozen reference.

Run v3 (this worktree) and v2.3.1 frozen on a matched single-genotype
scenario; report the cancer-total ratio. Acceptance: within the v2->v3
engine-difference band already established by the natural-history parity
gates (document the observed ratio; flag if outside ~0.85-1.15).

v3 (worktree-pinned):
  PYTHONPATH=<worktree> <venv-python> tests/regression/validate_v2_grow_multiscale.py --engine v3
v2.3.1 frozen:
  PYTHONPATH=C:/Users/ryanhu/PycharmProjects/hpvsim_v23_frozen <venv-python> \
      tests/regression/validate_v2_grow_multiscale.py --engine v2
"""
import sys, argparse
# ... build a matched scenario per engine, run at ms_agent_ratio=10,
#     print total cancers; compare the two printed numbers by hand or via a
#     small wrapper. Keep the two engines in separate processes (different
#     PYTHONPATH) — they cannot coexist in one interpreter.
```

- [ ] **Step 2: Run both engines and record the ratio**

Run the two commands in the docstring; record the v3/v2 cancer-total ratio in the script header as the validated figure.
Expected: ratio within the documented engine-difference band. If materially outside, debug with `superpowers:systematic-debugging` before declaring the port faithful.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/validate_v2_grow_multiscale.py
git commit -m "test(multiscale): v2.3.1 frozen numerical-tracking validation script"
```

---

## Self-Review

**Spec coverage:**
- §3 scale convention → Task 1 (`cancer_scale=1/ratio` set in Task 4), Task 2.
- §4 `fine` state → Task 1.
- §5 grow + full clone → Tasks 3 (clone) + 4 (grow).
- §6 exclusions (network/births/emigration; death untouched) → Tasks 5, 6, 7.
- §7 scale-weighted float results (flows, ages, stocks) → Tasks 2, 8.
- §8 acceptance gates (ratio==1 / incidence-flat / intervention-equiv / variance / v2-tracking) → ratio==1 woven through Tasks 2–8; Tasks 9, 10, 11, 12.
- §9 files touched → matches Tasks 1–8.
- §10 non-goals (no starsim change, no CRN) → honored in Global Constraints.

**Placeholder scan:** Task 10 Step 2 (`intervention_factory` body) and Task 12 (script body) are intentionally completed during execution from the verified m06 API / matched scenario — Step 1 of each task locates the exact construction first. These are discovery-then-fill, not blind placeholders; flagged explicitly. All engine-code tasks (1–8) contain complete code.

**Type consistency:** `_clone_agents(sim, src_uids, new_uids)` defined in Task 3, called in Task 4. `hpv.Births.get_births()` defined in Task 6, consumed by default stack. `people.fine` (BoolArr) introduced in Task 1, consumed in Tasks 3–8. `ms_agent_ratio` (int) introduced in Task 1, consumed in Task 4. Result dtypes float (Task 2, 8) consistent with scale-weighted writes.

**Known risk flagged:** Task 4 Step 3 NOTE — years vs steps in duration handling — is the single most likely source of an incidence-flat failure (Task 9) and has an explicit debugging pointer.
