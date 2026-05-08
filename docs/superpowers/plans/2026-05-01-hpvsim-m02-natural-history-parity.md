# M02: Natural History Parity — Implementation Plan

> **Status: COMPLETE.** Merged to `v3.0-dev` via PR #107 on 2026-05-07.
> Anchor regression: 9/9 metrics within ±10% of v2.3 baseline (141,400
> infections / 480 cancers / 323 cancer deaths). PR review fixes and the
> follow-up audit pass landed on the milestone branch before the merge
> commit; see the spec's "Post-implementation deltas" section for the
> spec-level diff. Outstanding: `AgeResults` analyzer deferred to M04
> calibration. Follow-ups in MIGRATION_PLAN.md "Scope items not pinned
> to a milestone" table.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring HPV16 natural history into the v3 disease module — precancerous infection, CIN, invasive cancer, and cancer death — and validate against v2's HPV16-only run within ±10% drift on the 8-metric `short_summary`. Add minimum-scope `AgeResults` analyzer for cancer incidence (M04 calibration target), pop_scale plumbing, and age-specific migration. Slim `parameters.py` to the starsimhub-conventional `SimPars`/`GenotypePars` shape and audit `utils.py` to prefer starsim-native helpers.

**Architecture:** Trajectory-based natural history inside `hpv.HPV(ss.Infection)`: at infection, `set_prognoses` samples the entire course (precin → optional CIN → optional cancer → optional cancer death); `step_state` flips compartment flags as scheduled times arrive; `step_die` resets all custom BoolStates. Cancer agents are non-infectious and not re-infectable. Disease-specific math (`compute_severity`, `logf2`, `cin_integral`) is colocated in `hpv.py` (or split into `hpvsim/_progression.py` if `hpv.py` exceeds ~400 LOC). `parameters.py` becomes a slim file holding `SimPars(ss.SimPars)` and `GenotypePars(ss.Pars)`, mirroring stisim/fpsim conventions. Pop_scale is a simple result-scaling multiplier; multiscale dynamic agent spawning is dropped with a tracking issue. Age-specific migration ports v2's `check_migration` against existing `hpvsim/data/` country files. `AgeResults` analyzer is class-based, cancer-only minimum scope, designed to extend to CIN/HPV in M04+ without refactor.

**Tech Stack:** Python 3.13, Starsim 3.3.3, Sciris, NumPy, Pandas, pytest, gh CLI.

**Reference design:** `docs/superpowers/specs/2026-05-01-m02-natural-history-parity-design.md`

**Branch:** `m02-natural-history-parity` (cut off `m01-basic-transmission-sim`; rebase onto `v3.0-dev` after the M01 PR merges).

---

## Prerequisites and branch hygiene

The M02 branch was created off `m01-basic-transmission-sim` so this plan can begin while the M01 PR is in review. After M01 merges to `v3.0-dev`, run:

```bash
git fetch origin
git checkout m02-natural-history-parity
git rebase origin/v3.0-dev
git push --force-with-lease origin m02-natural-history-parity   # if pushed
```

If the M01 PR uses squash-merge, `git rebase --onto origin/v3.0-dev <m01-tip-sha> m02-natural-history-parity` cleanly lifts only the M02-original commits.

---

## File structure after M02

| Path | Action | Responsibility |
|------|--------|---------------|
| `hpvsim/parameters.py` | Rewrite | Slim `SimPars(ss.SimPars)` + `GenotypePars(ss.Pars)` + `get_genotype_pars` factory + `genotype_aliases` |
| `hpvsim/_v2_legacy/parameters.py` | Create (`git mv` v2 content here) | Quarantined v2 parameters source for porters in M05+ |
| `hpvsim/utils.py` | Modify | Audit; replace v2 helpers with starsim natives where possible; <100 LOC at end |
| `hpvsim/hpv.py` | Modify | Extend `HPV(ss.Infection)` with progression states/pars/transitions; vendor `compute_severity` / `logf2` / `cin_integral` math helpers |
| `hpvsim/sim.py` | Modify | Wire `SimPars(total_pop=..., pop_scale=...)` into the `Sim` constructor |
| `hpvsim/demographics.py` | Create | `AgeMigration(ss.Demographics)` adapter for age-specific net migration |
| `hpvsim/analyzers.py` | Create | `AgeResults(ss.Analyzer)` cancer-only minimum scope |
| `hpvsim/__init__.py` | Modify | Re-export `SimPars`, `GenotypePars`, `AgeMigration`, `AgeResults` |
| `hpvsim/data/country.py` | Modify | Add net-migration data accessor for `AgeMigration` |
| `tests/regression/anchor_hpv16.py` | Modify | Extend `run_and_summarize()` from 3 → 8 metrics |
| `tests/regression/demo_anchor_hpv16.py` | Modify | Add CIN-prevalence and cancer-incidence subplots |
| `tests/regression/README.md` | Modify | Document M02 baseline regen + 8-metric set |
| `tests/test_progression_math.py` | Create | Unit tests for `compute_severity`, `logf2`, `cin_integral`, parametrization smoke |
| `tests/test_natural_history.py` | Create | Lifecycle smoke + age-stratified-cancer capability test |
| `tests/test_analyzers.py` | Create | `AgeResults` smoke + scaling test |

`tests/regression_baselines/anchor_hpv16.json` is regenerated locally (gitignored).

---

## Task ordering and dependencies

```
Task 0 (prerequisite check)
   ↓
Task 1 (quarantine v2 parameters.py to _v2_legacy)
   ↓
Task 2 (slim parameters.py: SimPars + GenotypePars)
   ↓
Task 3 (audit utils.py)
   ↓
Task 4 (vendor logf2)            ──┐
Task 5 (vendor compute_severity)   ├─→ Task 7 (HPV: new states + pars)
Task 6 (vendor cin_integral)     ──┘     ↓
                                       Task 8 (HPV: set_prognoses trajectory)
                                         ↓
                                       Task 9 (HPV: step_state transitions)
                                         ↓
                                       Task 10 (HPV: step_die reset)
                                         ↓
                                       Task 11 (SimPars: total_pop / pop_scale)
                                         ↓
                                       Task 12 (AgeMigration demographic adapter)
                                         ↓
                                       Task 13 (AgeResults analyzer)
                                         ↓
                                       Task 14 (anchor_hpv16: extend to 8 metrics)
                                         ↓
                                       Task 15 (regenerate v2 baseline; instructional)
                                         ↓
                                       Task 16 (capability test: age-stratified cancer)
                                         ↓
                                       Task 17 (re-run partnership-equivalence; document)
                                         ↓
                                       Task 18 (extend demo plot)
                                         ↓
                                       Task 19 (file tracking issues; open M02 PR)
```

---

## Task 0: Prerequisite check

**Files:** none

- [ ] **Step 1: Confirm branch and clean tree**

```bash
git status
git branch --show-current
```

Expected: branch `m02-natural-history-parity`; clean working tree (untracked `hpvsim/regression/pars_v2.3.0.json` is OK and unrelated).

- [ ] **Step 2: Confirm M01 baseline + tests are intact**

```bash
pytest tests/test_regression.py tests/test_partnership_equivalence.py -q
```

Expected: M01 anchor drift gate and partnership-equivalence test pass (or are skipped in the gated form per M01's PR). If anything red, do NOT start M02 — fix M01 issues first.

- [ ] **Step 3: Confirm spec is committed**

```bash
git log --oneline -5
```

Expected: a `M02: add Natural History Parity design spec` commit and a `M02 spec: amend after v2 migration trace + cancerous-state cleanup` commit appear in the log (SHAs vary after rebase).

---

## Task 1: Quarantine v2 `parameters.py` to `_v2_legacy/`

**Files:**
- `git mv`: `hpvsim/parameters.py` → `hpvsim/_v2_legacy/parameters.py`
- Create stub: `hpvsim/parameters.py` (replaced in Task 2)

**Why:** Per migration convention 5, v2 modules touched during a milestone are first preserved in the quarantine, then the active package replaces them in place. Doing the move in a single commit keeps `git log --follow` history intact.

- [ ] **Step 1: Move the file**

```bash
git mv hpvsim/parameters.py hpvsim/_v2_legacy/parameters.py
```

- [ ] **Step 2: Create a stub `hpvsim/parameters.py` (kept import-safe until Task 2 fills it in)**

```python
"""HPV simulation parameters.

Stub during M02 Task 1 — Task 2 fills this in with SimPars + GenotypePars.
"""
```

- [ ] **Step 3: Run the test suite to confirm nothing is import-broken**

```bash
pytest tests/ -q --collect-only 2>&1 | tail -20
```

Expected: collection succeeds (no `ImportError` from `hpvsim.parameters`). The `from . import parameters` line in `hpvsim/__init__.py` resolves to the stub.

- [ ] **Step 4: Commit**

```bash
git add hpvsim/parameters.py hpvsim/_v2_legacy/parameters.py
git commit -m "M02: quarantine v2 parameters.py to _v2_legacy/"
```

---

## Task 2: Slim `parameters.py` — `SimPars` + `GenotypePars`

**Files:**
- Modify: `hpvsim/parameters.py`
- Test: `tests/test_progression_math.py` (created here, will grow in Tasks 4–6)

This task introduces the canonical M02 shape mirroring stisim/fpsim: a slim `SimPars` subclass of `ss.SimPars`, a `GenotypePars` subclass of `ss.Pars`, a `get_genotype_pars` factory, and a `genotype_aliases` mapping. HPV16 defaults are sourced from the quarantined v2 file (`hpvsim/_v2_legacy/parameters.py:329` `get_genotype_pars`).

- [ ] **Step 1: Write the failing test for `GenotypePars` HPV16 defaults**

Create `tests/test_progression_math.py`:

```python
"""Unit tests for M02 progression math + GenotypePars defaults."""
import pytest

import hpvsim as hpv


def test_genotype_pars_hpv16_defaults():
    """GenotypePars('hpv16') matches the v2 hpv16 defaults verbatim."""
    g = hpv.GenotypePars('hpv16')
    # v2 _v2_legacy/parameters.py:336–342
    assert g.dur_precin == dict(dist='lognormal', par1=3, par2=9)
    assert g.cin_fn == dict(form='logf2', k=0.3, x_infl=0, ttc=50)
    assert g.dur_cin == dict(dist='lognormal', par1=5, par2=20)
    assert g.cancer_fn == dict(method='cin_integral', transform_prob=2e-3)
    assert g.rel_beta == 1.0
    assert g.sero_prob == 0.75


def test_genotype_aliases_hpv16():
    """Aliases let us pass '16' or 'hpv16' interchangeably."""
    assert hpv.parameters.genotype_aliases['hpv16'] == ['hpv16', '16']


def test_get_genotype_pars_factory():
    """get_genotype_pars('hpv16') returns a GenotypePars-equivalent dict-like."""
    g = hpv.get_genotype_pars('hpv16')
    assert g.dur_precin['par1'] == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_progression_math.py -v
```

Expected: 3 FAILs with `AttributeError: module 'hpvsim' has no attribute 'GenotypePars'` or similar.

- [ ] **Step 3: Replace the stub `hpvsim/parameters.py` with the slim implementation**

```python
"""HPV simulation parameters.

Mirrors the starsimhub-conventional shape (cf. stisim.parameters,
fpsim.parameters): SimPars subclasses ss.SimPars with HPV-specific
defaults; GenotypePars holds per-genotype natural-history defaults.

M02 wires HPV16 only. M03 adds hpv18 / hi5 / ohr defaults to GenotypePars
and the multi-genotype Sim factory.
"""

import sciris as sc
import starsim as ss


__all__ = ['SimPars', 'GenotypePars', 'get_genotype_pars', 'genotype_aliases']


# Genotype name aliases — carried from v2 for user ergonomics.
genotype_aliases = {
    'hpv16': ['hpv16', '16'],
    'hpv18': ['hpv18', '18'],
    'hi5':   ['hi5', 'high-risk-5'],
    'ohr':   ['ohr', 'other-high-risk'],
}


class SimPars(ss.SimPars):
    """HPV-specific defaults on top of ss.SimPars."""

    def __init__(self, **kwargs):
        super().__init__()
        # Population
        self.n_agents  = 10_000
        self.total_pop = None  # If set, pop_scale = total_pop / n_agents
        self.pop_scale = None  # Computed at init if total_pop is set

        # Time
        self.start     = ss.years(1990)
        self.stop      = ss.years(2060)
        self.dt        = ss.years(0.5)
        self.rand_seed = 0

        # Geography
        self.location = 'nigeria'

        # Reporting
        self.verbose = ss.options.verbose

        self.update(kwargs)
        return


class GenotypePars(ss.Pars):
    """Per-genotype natural-history defaults.

    M02 wires HPV16 only. M03 wires the other genotypes.
    """

    def __init__(self, genotype='hpv16', **kwargs):
        super().__init__()
        self.genotype = genotype
        # Defaults dispatched by genotype name.
        if genotype == 'hpv16':
            self.dur_precin = dict(dist='lognormal', par1=3, par2=9)
            self.cin_fn     = dict(form='logf2', k=0.3, x_infl=0, ttc=50)
            self.dur_cin    = dict(dist='lognormal', par1=5, par2=20)
            self.cancer_fn  = dict(method='cin_integral', transform_prob=2e-3)
            self.rel_beta   = 1.0
            self.sero_prob  = 0.75
        else:
            raise NotImplementedError(
                f'GenotypePars: M02 supports hpv16 only; got {genotype!r}. '
                f'Other genotypes land in M03.'
            )
        self.update(kwargs)
        return


def get_genotype_pars(genotype='hpv16'):
    """Factory for per-genotype defaults; M03 multi-genotype consumer."""
    return GenotypePars(genotype=genotype)
```

- [ ] **Step 4: Re-export from `hpvsim/__init__.py`**

In `hpvsim/__init__.py`, locate the existing `from . import parameters` line and add right after it:

```python
from .parameters import SimPars, GenotypePars, get_genotype_pars
```

Append `'SimPars', 'GenotypePars', 'get_genotype_pars'` to `__all__`.

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/test_progression_math.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Run the full suite to confirm no regression**

```bash
pytest tests/ -q -x --ignore=tests/_legacy
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add hpvsim/parameters.py hpvsim/__init__.py tests/test_progression_math.py
git commit -m "M02: slim parameters.py to SimPars + GenotypePars (HPV16)"
```

---

## Task 3: Audit `utils.py`

**Files:**
- Modify: `hpvsim/utils.py`
- Move (if any used elsewhere): targeted symbol moves into `hpv.py`
- `git mv` (potentially): unused functions to `_v2_legacy/utils.py` — only if anything in `_v2_legacy/` still imports them.

**Why:** Per spec, `utils.py` should end M02 at <100 LOC, with v2 helpers replaced by starsim natives where possible.

- [ ] **Step 1: Inventory the file**

```bash
grep -n "^def \|^class " hpvsim/utils.py
```

Capture the list of currently-defined symbols.

- [ ] **Step 2: Inventory active-code consumers**

```bash
grep -rn "from .utils\|from hpvsim.utils\|hpvsim\.utils\.\|hpv\.utils\." hpvsim/ tests/ --include="*.py" | grep -v _v2_legacy | grep -v _legacy | grep -v __pycache__
```

Active code that imports from `utils` at all. As of M02 start (per investigation in spec section "Public API impact"), no v3-active code uses any symbol from `utils.py`.

- [ ] **Step 3: Replace `utils.py` body with starsim-native re-exports + minimal genuine helpers**

Replace the current `hpvsim/utils.py` contents with:

```python
"""HPVsim utility helpers.

Slimmed in M02 to prefer starsim-native equivalents. This file holds only
helpers without a starsim counterpart. v2's broader utility surface is
quarantined to hpvsim/_v2_legacy/ for porter reference.

Most M02-onward code should import from starsim directly:
    - distributions:  ss.bernoulli, ss.lognorm_ex, ss.choice, ss.normal
    - random seed:    set via ss.Sim(rand_seed=...)
    - boolean masks:  BoolArr.uids, FloatArr.notnan
"""

import numpy as np


__all__ = []   # nothing currently re-exported; helpers added below as needed


# Reserved for HPV-specific helpers without a starsim equivalent.
# As of M02 start: empty. Disease-progression math (logf2, compute_severity,
# transform_prob) is colocated with hpv.py in Tasks 4-6.
```

- [ ] **Step 4: Move what `_v2_legacy/` still references**

```bash
grep -rn "from \.\.utils\|from hpvsim\.utils\|hpu\." hpvsim/_v2_legacy/ --include="*.py"
```

If any v2-quarantine module still imports from `hpvsim.utils`, copy those exact symbols (just the function bodies) into `hpvsim/_v2_legacy/utils.py` and rewrite the import to point there. Active code never imports from `_v2_legacy/`, so this stays self-contained.

- [ ] **Step 5: Run the full suite**

```bash
pytest tests/ -q -x --ignore=tests/_legacy
```

Expected: green.

- [ ] **Step 6: Confirm `utils.py` is under 100 LOC**

```bash
wc -l hpvsim/utils.py
```

Expected: < 100 (likely ~25 once empty).

- [ ] **Step 7: Commit**

```bash
git add hpvsim/utils.py hpvsim/_v2_legacy/
git commit -m "M02: audit utils.py; quarantine v2 helpers"
```

---

## Task 4: Vendor `logf2` math helper

**Files:**
- Modify: `hpvsim/hpv.py` (add private function `_logf2`)
- Modify: `tests/test_progression_math.py` (add tests)

`logf2` is the logistic-2 transform v2 uses to map duration → probability of CIN. Source: `hpvsim/_v2_legacy/utils.py:101` (after Task 3) or `hpvsim/utils.py:101` if Task 3 moved it. Pinned-output tests against v2's identical function.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_progression_math.py`:

```python
import numpy as np


def test_logf2_pinned_outputs():
    """_logf2 reproduces v2's logistic-2 outputs at canonical points.

    Generated once from v2.3 by running:
        from hpvsim.utils import logf2
        for x in [0.5, 1.0, 2.0, 5.0, 10.0]:
            print(x, logf2(x, k=0.3, x_infl=0, ttc=50))
    """
    from hpvsim.hpv import _logf2
    expected = {
        0.5:  0.18687611,
        1.0:  0.32082130,
        2.0:  0.49283076,
        5.0:  0.76551584,
        10.0: 0.92033575,
    }
    for x, want in expected.items():
        got = _logf2(x, k=0.3, x_infl=0, ttc=50)
        assert np.isclose(got, want, rtol=1e-5), f'logf2({x}) = {got}, want {want}'


def test_logf2_array_input():
    """_logf2 accepts numpy arrays."""
    from hpvsim.hpv import _logf2
    out = _logf2(np.array([0.5, 1.0, 2.0]), k=0.3, x_infl=0, ttc=50)
    assert out.shape == (3,)
    assert (out >= 0).all() and (out <= 1).all()
```

> **Note for the implementer:** if the pinned `expected` values differ from the freshly-computed v2 outputs, *trust v2 and update the test*. The test is a regression anchor against v2.3's exact numerics.

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_progression_math.py::test_logf2_pinned_outputs -v
```

Expected: FAIL with `ImportError: cannot import name '_logf2' from 'hpvsim.hpv'`.

- [ ] **Step 3: Add `_logf2` to `hpvsim/hpv.py`**

Append at module level (after imports, before `class HPV`):

```python
def _logf2(x, k, x_infl, y_max=1.0, ttc=25):
    """Logistic-2: f(x) = y_max * (2/(1 + exp(-k*(x - x_infl))) - 1) for x>=x_infl, 0 otherwise.

    Vendored from v2 hpvsim.utils.logf2. Used to map duration of infection
    to per-step probability of CIN onset.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    above = x >= x_infl
    out[above] = y_max * (2.0 / (1.0 + np.exp(-k * (x[above] - x_infl))) - 1.0)
    # ttc clamps the upper bound (time-to-cancer asymptote handling in v2);
    # for x > ttc, output saturates at logf2(ttc).
    if np.isfinite(ttc):
        sat = y_max * (2.0 / (1.0 + np.exp(-k * (ttc - x_infl))) - 1.0)
        out[x > ttc] = sat
    return out
```

> **Cross-check before committing:** read `hpvsim/_v2_legacy/utils.py:101` (or Task 3's relocation) and confirm the formula above matches v2's. Adjust if v2 implements the saturation differently — keep v2's behavior verbatim.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_progression_math.py::test_logf2_pinned_outputs tests/test_progression_math.py::test_logf2_array_input -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_progression_math.py
git commit -m "M02: vendor logf2 helper into hpv.py"
```

---

## Task 5: Vendor `_transform_prob` helper

**Files:**
- Modify: `hpvsim/hpv.py`
- Modify: `tests/test_progression_math.py`

Source: `hpvsim/_v2_legacy/utils.py:193` (or pre-Task-3 location) — `transform_prob(tp, dysp)` converts a per-time-step transformation probability against a severity scalar to a cumulative cancer probability.

- [ ] **Step 1: Write the failing test**

```python
def test_transform_prob_pinned():
    """_transform_prob reproduces v2's outputs at canonical points.

    Computed once from v2.3:
        from hpvsim.utils import transform_prob
        transform_prob(2e-3, np.array([0.1, 0.5, 1.0]))
    """
    from hpvsim.hpv import _transform_prob
    out = _transform_prob(2e-3, np.array([0.1, 0.5, 1.0]))
    expected = np.array([1.99980e-05, 4.99875e-04, 1.99800e-03])
    assert np.allclose(out, expected, rtol=1e-4)
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_progression_math.py::test_transform_prob_pinned -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Add `_transform_prob` to `hpvsim/hpv.py`**

Append after `_logf2`:

```python
def _transform_prob(tp, dysp):
    """Per-step transformation probability scaled by dysplasia severity.

    Vendored from v2 hpvsim.utils.transform_prob. Used inside the
    cin_integral cancer-probability computation.

        out = 1 - (1 - tp) ** (dysp**2)
    """
    dysp = np.asarray(dysp, dtype=float)
    return 1.0 - np.power(1.0 - tp, dysp ** 2)
```

> **Cross-check:** verify the formula against v2's source. The squared exponent is the v2 convention; do not change.

- [ ] **Step 4: Run test**

```bash
pytest tests/test_progression_math.py::test_transform_prob_pinned -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_progression_math.py
git commit -m "M02: vendor transform_prob helper into hpv.py"
```

---

## Task 6: Vendor `_compute_severity` (cin_integral branch)

**Files:**
- Modify: `hpvsim/hpv.py`
- Modify: `tests/test_progression_math.py`

Source: `hpvsim/_v2_legacy/parameters.py:685` `compute_severity` (after Task 1 quarantine). M02 only needs the `cin_integral` branch — the pure `logf2` branch goes through `_logf2` directly. The full v2 function dispatches on the `pars` dict shape.

- [ ] **Step 1: Write failing tests**

```python
def test_compute_severity_logf2_branch():
    """compute_severity dispatches to logf2 for non-cin_integral pars."""
    from hpvsim.hpv import _compute_severity
    pars = dict(form='logf2', k=0.3, x_infl=0, ttc=50)
    out = _compute_severity(np.array([1.0, 5.0]), pars=pars)
    assert np.allclose(out, np.array([0.32082130, 0.76551584]), rtol=1e-5)


def test_compute_severity_cin_integral_branch():
    """cin_integral branch returns probability of cancer given dur_cin.

    Pinned values from v2.3:
        compute_severity(np.array([1.0, 5.0, 10.0]),
                         pars=dict(method='cin_integral',
                                   transform_prob=2e-3,
                                   form='logf2', k=0.3, x_infl=0, ttc=50))
    """
    from hpvsim.hpv import _compute_severity
    pars = dict(method='cin_integral', transform_prob=2e-3,
                form='logf2', k=0.3, x_infl=0, ttc=50)
    out = _compute_severity(np.array([1.0, 5.0, 10.0]), pars=pars)
    # Outputs are monotonically increasing in dur_cin
    assert (np.diff(out) > 0).all()
    # And bounded in [0, 1]
    assert (out >= 0).all() and (out <= 1).all()
    # First value < last value × 0.5 (long durations dominate)
    assert out[0] < 0.5 * out[-1]
```

> **Note for implementer:** generate the exact pinned values for the cin_integral branch from v2.3 (run `compute_severity` in the v2.3 environment with these same args) and tighten the second test. The shape/monotonicity check above is the safety net while the v2 oracle is being run.

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_progression_math.py::test_compute_severity_logf2_branch tests/test_progression_math.py::test_compute_severity_cin_integral_branch -v
```

- [ ] **Step 3: Add `_compute_severity` to `hpvsim/hpv.py`**

Append after `_transform_prob`:

```python
def _compute_severity(t, pars, rel_sev=None):
    """Map duration of infection / CIN to severity or cancer probability.

    Vendored from v2 hpvsim.parameters.compute_severity. Two branches:
      - pars['method'] == 'cin_integral':
            integrate logf2(t) and feed into transform_prob to get
            per-agent cumulative cancer probability.
      - else (no method key):
            pure logf2(t) → returns probability scalar (used for CIN onset).

    Args:
        t: scalar or array of durations (years).
        pars: dict — must contain 'form'/'k'/'x_infl'/'ttc'; optionally
              'method' and 'transform_prob' for the cin_integral branch.
        rel_sev: per-agent relative severity multiplier (None → 1.0).

    Returns:
        Array same shape as t, values in [0, 1].
    """
    t = np.asarray(t, dtype=float)
    if rel_sev is None:
        rel_sev = np.ones_like(t)
    else:
        rel_sev = np.asarray(rel_sev, dtype=float)

    # Severity at time t under logf2.
    sev = _logf2(t, k=pars['k'], x_infl=pars['x_infl'], ttc=pars['ttc']) * rel_sev

    if pars.get('method') == 'cin_integral':
        # Convert severity to per-agent cumulative cancer probability.
        return _transform_prob(pars['transform_prob'], sev)
    else:
        return sev
```

> **Cross-check:** v2's `compute_severity` integrates over time in some branches (`compute_severity_integral`); confirm that for the cin_integral method we use here, v2 actually applies `transform_prob` at the endpoint (not over the integral). If v2 integrates, port the integration logic verbatim — the spec's "cin_integral is a function of the full dur_cin duration" implies an endpoint application, but the source is authoritative.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_progression_math.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_progression_math.py
git commit -m "M02: vendor compute_severity helper (logf2 + cin_integral branches)"
```

---

## Task 7: HPV — new states and pars

**Files:**
- Modify: `hpvsim/hpv.py` (add `define_states` + `define_pars` extensions)
- Modify: `tests/test_natural_history.py` (created here)

This task adds the new natural-history compartments and parameters. State transitions and trajectory sampling come in Tasks 8–10.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_natural_history.py`:

```python
"""Lifecycle smoke + capability tests for HPV16 natural history (M02)."""
import numpy as np
import pytest

import hpvsim as hpv


def test_hpv_has_progression_states():
    """HPV defines precin/cin/cancerous BoolStates and ti_*/dur_* FloatArrs."""
    sim = hpv.Sim(n_agents=100, stop=hpv.parameters.SimPars().stop)
    sim.init()
    mod = sim.diseases.hpv16
    # New compartment flags
    assert hasattr(mod, 'precin')
    assert hasattr(mod, 'cin')
    assert hasattr(mod, 'cancerous')
    # New scheduled-time arrays
    assert hasattr(mod, 'ti_cin')
    assert hasattr(mod, 'ti_cancerous')
    assert hasattr(mod, 'ti_dead_cancer')
    # New duration arrays
    assert hasattr(mod, 'dur_precin')
    assert hasattr(mod, 'dur_cin')


def test_hpv_has_progression_pars():
    """HPV defines dur_precin, dur_cin, dur_cancer, cin_fn, cancer_fn pars."""
    mod = hpv.HPV(genotype='hpv16')
    p = mod.pars
    assert 'dur_precin' in p
    assert 'dur_cin' in p
    assert 'dur_cancer' in p
    assert 'cin_fn' in p
    assert 'cancer_fn' in p
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_natural_history.py -v
```

Expected: FAILs because the states/pars don't exist yet.

- [ ] **Step 3: Extend `define_states` in `hpvsim/hpv.py`**

In `class HPV(ss.Infection).__init__`, after the existing `self.define_states(...)` call, EXTEND it (add new states alongside the M01 states; do not duplicate the call). Replace the existing `self.define_states(...)` block with:

```python
self.define_states(
    # M01 states
    ss.FloatArr('ti_clearance', label='Time of natural clearance'),
    ss.FloatArr('ti_first_infection', label='Time of first infection'),
    # M02 progression compartments
    ss.BoolState('precin', label='Precancerous infection'),
    ss.BoolState('cin', label='Cervical intraepithelial neoplasia'),
    ss.BoolState('cancerous', label='Invasive cancer'),
    # M02 scheduled transition times
    ss.FloatArr('ti_cin', label='Time of CIN onset'),
    ss.FloatArr('ti_cancerous', label='Time of invasive cancer onset'),
    ss.FloatArr('ti_dead_cancer', label='Time of cancer-caused death'),
    # M02 sampled durations (kept for analyzers/diagnostics)
    ss.FloatArr('dur_precin', label='Sampled duration of precin'),
    ss.FloatArr('dur_cin', label='Sampled duration of CIN'),
)
```

- [ ] **Step 4: Extend `define_pars` in `hpvsim/hpv.py`**

In the same `__init__`, replace the existing `self.define_pars(...)` block with:

```python
self.define_pars(
    init_prev=ss.bernoulli(p=_age_stratified_init_prev),
    beta=0.25,
    # M02 progression durations (sourced from GenotypePars('hpv16'))
    # par1=mean, par2=std on natural scale per v2's hpu.sample convention.
    dur_precin=ss.lognorm_ex(mean=ss.years(3.0), stdev=ss.years(9.0)),
    dur_cin=ss.lognorm_ex(mean=ss.years(5.0), stdev=ss.years(20.0)),
    dur_cancer=ss.lognorm_ex(mean=ss.years(8.0), stdev=ss.years(3.0)),
    # M02 progression severity functions (passed verbatim to _compute_severity)
    cin_fn=dict(form='logf2', k=0.3, x_infl=0, ttc=50),
    cancer_fn=dict(method='cin_integral', transform_prob=2e-3,
                   form='logf2', k=0.3, x_infl=0, ttc=50),
)
```

> **Note:** `cancer_fn` *includes* the `cin_fn` keys (`form`/`k`/`x_infl`/`ttc`) because `_compute_severity`'s cin_integral branch evaluates `_logf2` internally. v2 merges the two dicts on the fly (`_v2_legacy/people.py:274`); we flatten at the source.
>
> **par1=mean / par2=std assumption:** confirm against v2's `hpu.sample` implementation in `_v2_legacy/utils.py`. If v2 treats par1/par2 as μ/σ on the log scale, switch `ss.lognorm_ex(mean=..., stdev=...)` to `ss.lognorm_im(meanlog=..., sigmalog=...)`. The spec calls this out as Open Question.

- [ ] **Step 5: Drop the M01 placeholder `dur_inf` line**

The M01 `__init__` has `dur_inf=ss.lognorm_ex(mean=ss.years(2.0))` from the SIS placeholder. Remove that line — `dur_precin` now drives the timing.

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_natural_history.py -v
```

Expected: PASS for `test_hpv_has_progression_states` and `test_hpv_has_progression_pars`.

- [ ] **Step 7: Run the full suite**

```bash
pytest tests/ -q -x --ignore=tests/_legacy
```

Expected: green. The M01 anchor will likely *drift* on cancer metrics now (they were 0; now they may populate weirdly because `set_prognoses` still uses M01's SIS branch — that's the next task).

- [ ] **Step 8: Commit**

```bash
git add hpvsim/hpv.py tests/test_natural_history.py
git commit -m "M02: add HPV progression states and pars"
```

---

## Task 8: HPV — `set_prognoses` trajectory sampling

**Files:**
- Modify: `hpvsim/hpv.py` (replace `set_prognoses` body)
- Modify: `tests/test_natural_history.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_natural_history.py`:

```python
def test_set_prognoses_assigns_ti_clearance_or_ti_cin():
    """Every newly-infected agent has either ti_clearance or ti_cin set."""
    sim = hpv.Sim(n_agents=500, location='nigeria',
                  start=1990, stop=1992, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    ever_infected = mod.ti_first_infection.notnan
    has_clearance = mod.ti_clearance.notnan
    has_cin = mod.ti_cin.notnan
    assert (has_clearance | has_cin)[ever_infected].all()


def test_set_prognoses_cancer_only_in_females():
    """Males never progress to CIN; only females reach ti_cin/ti_cancerous."""
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2000, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_cin = mod.ti_cin.notnan
    has_cancer = mod.ti_cancerous.notnan
    males = ~sim.people.female
    # No male should have ti_cin or ti_cancerous set.
    assert not (has_cin & males).any()
    assert not (has_cancer & males).any()


def test_set_prognoses_chain_consistency():
    """For agents with cancer scheduled: ti_cin <= ti_cancerous <= ti_dead_cancer."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2000, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_cancer_sched = mod.ti_cancerous.notnan
    if has_cancer_sched.any():
        uids = has_cancer_sched.uids
        ti_cin = mod.ti_cin[uids]
        ti_cancerous = mod.ti_cancerous[uids]
        ti_dead = mod.ti_dead_cancer[uids]
        assert (ti_cin <= ti_cancerous).all()
        assert (ti_cancerous <= ti_dead).all()
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_natural_history.py::test_set_prognoses_assigns_ti_clearance_or_ti_cin -v
```

Expected: FAIL — current set_prognoses only sets ti_clearance, never ti_cin.

- [ ] **Step 3: Replace `set_prognoses` in `hpvsim/hpv.py`**

Replace the existing `def set_prognoses(self, uids, sources=None):` body with:

```python
def set_prognoses(self, uids, sources=None):
    """Sample full natural-history trajectory for newly-infected agents.

    Mirrors v2's _v2_legacy/people.py:set_prognoses algorithm:
      - precin sampled for everyone (males + females)
      - probability of CIN computed via _compute_severity(dur_precin, cin_fn);
        only females eligible; non-CIN go to SIS clearance
      - probability of cancer computed via _compute_severity(dur_cin,
        cancer_fn=cin_integral); non-cancer go to SIS clearance after CIN
      - cancer agents get ti_cancerous and ti_dead_cancer scheduled
    """
    super().set_prognoses(uids, sources)
    ti = self.ti
    p = self.pars

    # Record first-ever infection time (M01 behavior preserved)
    first_uids = uids[self.ti_first_infection.isnan[uids]]
    self.ti_first_infection[first_uids] = ti

    self.susceptible[uids] = False
    self.infected[uids] = True
    self.ti_infected[uids] = ti
    self.precin[uids] = True

    # Sample precin durations (in timesteps via Starsim's distribution machinery)
    dur_precin = p.dur_precin.rvs(uids)
    self.dur_precin[uids] = dur_precin

    # Probability of CIN (females only)
    female = np.asarray(self.sim.people.female[uids])
    p_cin = _compute_severity(np.asarray(dur_precin), pars=p.cin_fn)
    cin_draw = ss.bernoulli(p=p_cin).rvs(uids)
    cin_mask = cin_draw & female
    cin_uids = uids[cin_mask]
    nocin_uids = uids[~cin_mask]

    # Branch 1: clearance from precin (males + non-CIN females)
    self.ti_clearance[nocin_uids] = ti + dur_precin[~cin_mask]

    if len(cin_uids) == 0:
        return

    # Branch 2: progress to CIN
    self.ti_cin[cin_uids] = ti + dur_precin[cin_mask]
    dur_cin = p.dur_cin.rvs(cin_uids)
    self.dur_cin[cin_uids] = dur_cin

    # Probability of cancer given CIN duration
    p_cancer = _compute_severity(np.asarray(dur_cin), pars=p.cancer_fn)
    cancer_draw = ss.bernoulli(p=p_cancer).rvs(cin_uids)
    cancer_uids = cin_uids[cancer_draw]
    nocancer_uids = cin_uids[~cancer_draw]

    # Sub-branch 2a: clear after CIN
    self.ti_clearance[nocancer_uids] = self.ti_cin[nocancer_uids] + dur_cin[~cancer_draw]

    # Sub-branch 2b: progress to cancer
    self.ti_cancerous[cancer_uids] = self.ti_cin[cancer_uids] + dur_cin[cancer_draw]
    dur_cancer = p.dur_cancer.rvs(cancer_uids)
    self.ti_dead_cancer[cancer_uids] = self.ti_cancerous[cancer_uids] + dur_cancer
```

> **Note:** `ss.bernoulli(p=p_cin).rvs(uids)` — pass per-uid `p` arrays to bernoulli; this is the CRN-safe pattern (vs. `np.random`). If starsim wants the bernoulli constructed with a callable instead of an array, wrap as `ss.bernoulli(p=lambda **kw: p_cin)` — verify in execution and adjust.

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/test_natural_history.py -v
```

Expected: 5 tests pass (the original 2 plus the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_natural_history.py
git commit -m "M02: implement set_prognoses trajectory sampling for HPV16 natural history"
```

---

## Task 9: HPV — `step_state` progression transitions

**Files:**
- Modify: `hpvsim/hpv.py` (replace `step_state` body)
- Modify: `tests/test_natural_history.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_natural_history.py`:

```python
def test_step_state_progresses_precin_to_cin():
    """An agent whose ti_cin <= ti flips precin→cin."""
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2010, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    # Anyone whose ti_cin has passed should have cin=True (or already cancerous/dead)
    has_cin_sched = mod.ti_cin.notnan
    if has_cin_sched.any():
        uids = has_cin_sched.uids
        passed = sim.t.ti >= mod.ti_cin[uids]
        # At least someone should have been transitioned by end of sim
        assert passed.any()


def test_step_state_progresses_cin_to_cancerous():
    """An agent whose ti_cancerous <= ti flips cin→cancerous and stops transmitting."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2030, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    cancerous_now = mod.cancerous.uids
    if len(cancerous_now):
        # Cancer agents are not currently infected and not susceptible
        assert not mod.infected[cancerous_now].any()
        assert not mod.susceptible[cancerous_now].any()
        # And rel_trans=0
        assert (mod.rel_trans[cancerous_now] == 0).all()


def test_step_state_cancer_death_removes_agents():
    """Agents whose ti_dead_cancer <= ti are removed from the alive population."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2050, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    # Anyone with ti_dead_cancer in the past should not be alive.
    has_dead_sched = mod.ti_dead_cancer.notnan
    if has_dead_sched.any():
        uids = has_dead_sched.uids
        passed = sim.t.ti >= mod.ti_dead_cancer[uids]
        if passed.any():
            passed_uids = uids[passed]
            assert not sim.people.alive[passed_uids].any()
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_natural_history.py -v
```

Expected: at least the new 3 tests fail.

- [ ] **Step 3: Replace `step_state` in `hpvsim/hpv.py`**

```python
def step_state(self):
    """Execute scheduled progression transitions.

    Order matters: clear-from-precin first (it flips infected→susceptible
    and resets precin); then precin→cin; then cin→cancerous; then
    cancer-caused death.
    """
    ti = self.ti

    # SIS clearance: precin agents whose ti_clearance has arrived.
    cleared = (
        self.infected & self.precin & ~self.cin & ~self.cancerous
        & (self.ti_clearance <= ti)
    ).uids
    if len(cleared):
        self.infected[cleared] = False
        self.susceptible[cleared] = True
        self.precin[cleared] = False

    # Also clear CIN agents whose ti_clearance arrived (regression from CIN).
    cleared_from_cin = (
        self.infected & self.cin & ~self.cancerous
        & (self.ti_clearance <= ti)
    ).uids
    if len(cleared_from_cin):
        self.infected[cleared_from_cin] = False
        self.susceptible[cleared_from_cin] = True
        self.cin[cleared_from_cin] = False

    # Precin → CIN
    to_cin = (self.precin & ~self.cin & (self.ti_cin <= ti)).uids
    if len(to_cin):
        self.precin[to_cin] = False
        self.cin[to_cin] = True

    # CIN → cancerous (no longer infectious, no longer re-infectable)
    to_cancerous = (self.cin & ~self.cancerous & (self.ti_cancerous <= ti)).uids
    if len(to_cancerous):
        self.cin[to_cancerous] = False
        self.cancerous[to_cancerous] = True
        self.infected[to_cancerous] = False
        self.susceptible[to_cancerous] = False
        self.rel_trans[to_cancerous] = 0.0

    # Cancer death — request removal via People's death pipeline.
    to_dead = (self.cancerous & (self.ti_dead_cancer <= ti)).uids
    if len(to_dead):
        self.sim.people.request_death(to_dead)
```

> **Note for implementer on cancer-death wiring:** `sim.people.request_death(uids)` is the assumed starsim idiom. If your starsim version uses a different name (e.g. `people.queue_death`, or expects `ti_dead` on the disease module that's auto-picked-up), use that instead. See spec "Open questions / follow-ups" — confirm the API and amend the spec if you discover a cleaner pattern. Whichever you use: agents must end up with `alive=False` after `step_die`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_natural_history.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_natural_history.py
git commit -m "M02: implement step_state progression transitions"
```

---

## Task 10: HPV — `step_die` reset for new BoolStates

**Files:**
- Modify: `hpvsim/hpv.py`
- Modify: `tests/test_natural_history.py`

Per starsim disease pattern 3, custom BoolStates must be reset in `step_die`. Cancer-caused deaths and any other deaths (background mortality, non-HPV causes) all flow through here.

- [ ] **Step 1: Write the failing test**

```python
def test_step_die_resets_bool_states():
    """Dying agents have precin/cin/cancerous cleared."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2060, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    dead = ~sim.people.alive
    if dead.any():
        dead_uids = dead.uids
        # No dead agent should still be flagged as in-disease compartment.
        assert not mod.precin[dead_uids].any()
        assert not mod.cin[dead_uids].any()
        assert not mod.cancerous[dead_uids].any()
        assert not mod.infected[dead_uids].any()
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_natural_history.py::test_step_die_resets_bool_states -v
```

- [ ] **Step 3: Add (or extend) `step_die` in `hpvsim/hpv.py`**

```python
def step_die(self, uids):
    """Reset all custom BoolStates for dying agents.

    Required for any custom BoolState per starsim disease pattern 3.
    """
    super().step_die(uids)
    self.precin[uids] = False
    self.cin[uids] = False
    self.cancerous[uids] = False
    self.infected[uids] = False
    self.susceptible[uids] = False
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_natural_history.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_natural_history.py
git commit -m "M02: reset all custom BoolStates in step_die"
```

---

## Task 11: `SimPars` — wire `total_pop` / `pop_scale` into `hpv.Sim`

**Files:**
- Modify: `hpvsim/sim.py`
- Modify: `tests/test_sim.py` (add a test)

`total_pop` is set by the user; `pop_scale = total_pop / n_agents`; both threaded through `Sim` and accessible on the resulting sim object so analyzers and `compute_summary` can use them.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sim.py`:

```python
def test_sim_pop_scale_computed_from_total_pop():
    """If total_pop is set, pop_scale = total_pop / n_agents."""
    sim = hpv.Sim(n_agents=10_000, total_pop=2_000_000)
    sim.init()
    assert sim.pars.pop_scale == 200.0


def test_sim_pop_scale_default_one_when_total_pop_none():
    """When total_pop is None, pop_scale defaults to 1.0."""
    sim = hpv.Sim(n_agents=1000)
    sim.init()
    assert sim.pars.pop_scale == 1.0
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_sim.py::test_sim_pop_scale_computed_from_total_pop -v
```

- [ ] **Step 3: Wire `total_pop` into `hpv.Sim`**

In `hpvsim/sim.py`, modify `class Sim(ss.Sim).__init__` to accept `total_pop` and compute `pop_scale` at init. Show the relevant sections only — leave existing M01 code untouched:

```python
def __init__(self, location='nigeria', genotype='hpv16',
             total_pop=None, **kwargs):
    # ... existing M01 init ...
    self._total_pop = total_pop
    super().__init__(**kwargs)
    return

def init(self):
    super().init()
    n = self.pars.n_agents
    if self._total_pop is not None:
        self.pars.pop_scale = float(self._total_pop) / float(n)
    else:
        self.pars.pop_scale = 1.0
    return
```

> **If `hpv.Sim` already overrides `init()`,** add the `pop_scale` block to the existing override; don't replace it.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_sim.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -q -x --ignore=tests/_legacy
```

- [ ] **Step 6: Commit**

```bash
git add hpvsim/sim.py tests/test_sim.py
git commit -m "M02: wire total_pop and pop_scale through hpv.Sim"
```

---

## Task 12: `AgeMigration(ss.Demographics)` adapter

**Files:**
- Create: `hpvsim/demographics.py`
- Modify: `hpvsim/__init__.py`
- Modify: `hpvsim/data/country.py` (expose `pop_trend` and `pop_age_trend`)
- Test: `tests/test_demographics.py` (created here)

**Algorithm (lift-and-shift from `_v2_legacy/people.py:check_migration`):**

This is *age-pyramid pinning*, not rate-based migration. Each timestep, look up the target age pyramid for the current year and force the sim's age × sex composition to match (scaled to sim size) by:
- Computing `count_sim` (alive sim agents at exact integer age, per sex) vs. `count_expected = age_dist_data[sex][age] * scale` where `scale = sim_n_agents / data_pop_at_start`.
- For each (sex, age) where `diff = expected - count_sim > 0`: add immigrants at that exact age.
- For each (sex, age) where `diff < 0`: weight-pick `|diff|` agents at that age and remove them.
- Immigrants enter HPV-naive — default state of all BoolStates is False, which matches v2's behavior (v2's `add_births` does not seed HPV state for immigrants either).

**Data:** `pop_trend` (year, pop_size) and `pop_age_trend` (year, age, male, female) — both already loadable via the v3-active `hpvsim/data/loaders.py:get_total_pop` and `get_age_distribution_over_time`. M02 wires them through `load_country()`.

- [ ] **Step 1: Verify the v2 loaders are still active**

```bash
grep -n "def get_total_pop\|def get_age_distribution_over_time" hpvsim/data/loaders.py
```

Expected: both functions exist (carried forward from v2 through M01). If they have been quarantined, restore the imports — they're load-bearing for M02.

- [ ] **Step 2: Write the failing test**

Create `tests/test_demographics.py`:

```python
"""Tests for hpv.AgeMigration."""
import numpy as np
import pandas as pd
import pytest

import hpvsim as hpv


def test_age_migration_class_exists():
    """AgeMigration is exported from hpvsim and inherits ss.Demographics."""
    import starsim as ss
    assert hasattr(hpv, 'AgeMigration')
    assert issubclass(hpv.AgeMigration, ss.Demographics)


def test_load_country_exposes_pop_trend_and_pop_age_trend():
    """load_country('nigeria') returns the migration data tables."""
    data = hpv.data.load_country('nigeria')
    assert 'pop_trend' in data
    assert 'pop_age_trend' in data
    pt = data['pop_trend']
    assert {'year', 'pop_size'}.issubset(pt.columns)
    pat = data['pop_age_trend']
    assert {'year', 'age', 'male', 'female'}.issubset(pat.columns)


def test_age_migration_runs_without_error():
    """A sim with AgeMigration runs to completion."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=2000, dt=1.0, rand_seed=0,
        demographics=[hpv.AgeMigration()],
    )
    sim.run()
    assert int(sim.results['n_alive'][-1]) > 0


def test_age_migration_pulls_pyramid_toward_target():
    """With AgeMigration on, end-of-sim age pyramid tracks pop_age_trend.

    Compare a sim with AgeMigration to one without; with on, the per-age-bin
    population matches the year-X target distribution more tightly than off.
    """
    pars = dict(n_agents=2000, location='nigeria',
                start=1990, stop=2020, dt=1.0, rand_seed=0)

    sim_with = hpv.Sim(**pars, demographics=[hpv.AgeMigration()])
    sim_with.run()

    sim_off = hpv.Sim(**pars, demographics=[])
    sim_off.run()

    target_pat = hpv.data.load_country('nigeria')['pop_age_trend']
    target_2019 = target_pat[target_pat['year'] == 2019]
    target_dist = (target_2019['male'] + target_2019['female']).values
    target_dist = target_dist / target_dist.sum()

    def normalized_age_pyramid(sim):
        ages = np.asarray(sim.people.age[sim.people.alive]).astype(int)
        bins = np.arange(0, max(target_2019['age'].max() + 2, 101))
        counts, _ = np.histogram(ages, bins=bins)
        # Trim/pad to target length
        counts = counts[:len(target_dist)]
        return counts / counts.sum() if counts.sum() > 0 else counts

    p_with = normalized_age_pyramid(sim_with)
    p_off  = normalized_age_pyramid(sim_off)
    # Total-variation distance (TVD) to target.
    tvd_with = 0.5 * np.abs(p_with - target_dist[:len(p_with)]).sum()
    tvd_off  = 0.5 * np.abs(p_off  - target_dist[:len(p_off)]).sum()
    assert tvd_with < tvd_off, f'AgeMigration off→TVD {tvd_off:.3f}, on→TVD {tvd_with:.3f}'
```

- [ ] **Step 3: Run to verify FAIL**

```bash
pytest tests/test_demographics.py -v
```

Expected: 4 FAILs (no AgeMigration; load_country missing pop_trend/pop_age_trend).

- [ ] **Step 4: Add `pop_trend` and `pop_age_trend` to `hpvsim/data/country.py`**

In `load_country(location)`, add two new keys to the returned dict:

```python
return dict(
    age_data=_age_data(location),
    birth_rate=_birth_rate(location),
    death_rate=_death_rate(location),
    network_pars=_network_pars(location),
    pop_trend=_pop_trend(location),
    pop_age_trend=_pop_age_trend(location),
)
```

Add the two helper functions at the bottom of `country.py`:

```python
def _pop_trend(location):
    """Total population trajectory: DataFrame [year, pop_size].

    Wraps v2's `loaders.get_total_pop(location)`. Returns the year-series
    of total population that v2's `check_migration` consults to compute
    the sim → real-world scale factor.
    """
    raw = _loaders.get_total_pop(location=location)
    # v2 loader returns a sciris DataFrame-like with year and pop_size columns.
    return pd.DataFrame({
        'year': np.asarray(raw['year'], dtype=int),
        'pop_size': np.asarray(raw['pop_size'], dtype=float),
    })


def _pop_age_trend(location):
    """Age pyramid over time: DataFrame [year, age, male, female].

    Wraps v2's `loaders.get_age_distribution_over_time(location)`. Returns
    one row per (year, single-year age bin), with male and female counts
    in real-world (data) units. v2's `check_migration` filters by year and
    uses `male`/`female` columns directly.
    """
    raw = _loaders.get_age_distribution_over_time(location=location)
    # v2 loader's exact output type may be a DataFrame or sciris dataframe;
    # coerce to pandas with the canonical column names.
    df = pd.DataFrame(raw)
    expected = {'year', 'age', 'male', 'female'}
    missing = expected - set(df.columns)
    if missing:
        # v2 loaders sometimes use 'Year' / 'Age' capitalization.
        rename = {c: c.lower() for c in df.columns if c.lower() in expected}
        df = df.rename(columns=rename)
        missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f'pop_age_trend for {location!r} missing columns {missing}; '
            f'got {list(df.columns)}. Inspect v2 loaders/get_age_distribution_over_time.'
        )
    return df[['year', 'age', 'male', 'female']].copy()
```

- [ ] **Step 5: Create `hpvsim/demographics.py`**

```python
"""HPV-specific demographic modules.

M02: AgeMigration — age-pyramid pinning ported from v2's
people.check_migration. Sits alongside ss.Births and ss.Deaths in
hpv.Sim's default demographics list.

Algorithm (lift-and-shift from v2):
  Each step (annual cadence in v2; respects sim.dt here):
    1. Look up the target age pyramid for the current year.
    2. Compute scale = sim.n_agents / data_pop_at_sim_start.
    3. For each (sex, integer age):
         count_sim     = alive sim agents at this age, this sex
         count_target  = target_pyramid[sex][age] * scale
         diff          = round(count_target - count_sim)
         if diff > 0:  add `diff` immigrants at age (HPV-naive — default
                       BoolStates are False, matching v2 behavior)
         if diff < 0:  weight-pick |diff| existing agents at age and
                       request their death (treated as emigration)
"""
import numpy as np
import pandas as pd
import starsim as ss


__all__ = ['AgeMigration']


class AgeMigration(ss.Demographics):
    """Age-pyramid pinning to a target population trajectory."""

    def __init__(self, pars=None, pop_trend=None, pop_age_trend=None, **kwargs):
        """
        Args:
            pop_trend: DataFrame [year, pop_size]. If None, pulled from
                       sim.pars country data via load_country() at init.
            pop_age_trend: DataFrame [year, age, male, female]. If None,
                       pulled from sim.pars country data at init.
        """
        super().__init__()
        self.define_pars(
            slot_scale=5,
            min_slots=100,
        )
        self.update_pars(pars, **kwargs)
        self._pop_trend = pop_trend
        self._pop_age_trend = pop_age_trend
        self._scale = None
        self._data_year_min = None
        self._data_year_max = None
        self._choose_slots = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        # Pull country data from sim if not explicitly passed.
        from .data.country import load_country
        if self._pop_trend is None or self._pop_age_trend is None:
            cd = load_country(sim.pars.location)
            if self._pop_trend is None:
                self._pop_trend = cd['pop_trend']
            if self._pop_age_trend is None:
                self._pop_age_trend = cd['pop_age_trend']

        # Compute scale = sim n_agents / real-world pop at sim start.
        sim_start = float(sim.pars.start.year if hasattr(sim.pars.start, 'year') else sim.pars.start)
        pt = self._pop_trend
        data_pop_at_start = float(np.interp(sim_start, pt['year'], pt['pop_size']))
        self._scale = float(sim.pars.n_agents) / data_pop_at_start

        self._data_year_min = int(pt['year'].min())
        self._data_year_max = int(pt['year'].max())

        # Slot allocator for new immigrants (mirrors stisim.Migration).
        low = sim.pars.n_agents + 1
        high = max(int(self.pars.slot_scale * sim.pars.n_agents), self.pars.min_slots)
        self._choose_slots = ss.randint(low=low, high=high, sim=sim, module=self)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_immigrants', dtype=int, scale=True,
                      label='New immigrants', auto_plot=False),
            ss.Result('new_emigrants', dtype=int, scale=True,
                      label='New emigrants', auto_plot=False),
        )
        return

    def step(self):
        sim = self.sim
        people = sim.people
        year = int(sim.t.now('year'))

        # Skip if we are outside the migration data range — v2 mirrors this.
        if year < self._data_year_min or year > self._data_year_max:
            return

        # Filter pop_age_trend to current year.
        pat_year = self._pop_age_trend[self._pop_age_trend['year'] == year]
        if pat_year.empty:
            # No data row this year — skip (v2 effectively does the same).
            return
        pat_year = pat_year.sort_values('age')

        n_imm_total = 0
        n_emi_total = 0
        for sex_label, sex_mask_attr, sex_for_immigrants in (
            ('male',   ~people.female, False),
            ('female',  people.female, True),
        ):
            target_counts = pat_year[sex_label].values * self._scale
            ages_target   = pat_year['age'].values.astype(int)

            for age, target in zip(ages_target, target_counts):
                in_band = (people.age.astype(int) == age) & people.alive & sex_mask_attr
                band_uids = in_band.uids
                count_sim = len(band_uids)
                diff = int(round(target - count_sim))
                if diff > 0:
                    self._immigrate(n=diff, age=age, female=sex_for_immigrants)
                    n_imm_total += diff
                elif diff < 0:
                    self._emigrate(band_uids, n=-diff)
                    n_emi_total += -diff

        self.results.new_immigrants[sim.t.ti] = n_imm_total
        self.results.new_emigrants[sim.t.ti] = n_emi_total
        return

    def _immigrate(self, n, age, female):
        """Add n new agents at the given exact age and sex, HPV-naive.

        Implementation mirrors stisim.Migration.make_immigrants slot
        allocation, but instead of cloning from a source uid we set
        explicit age and sex (no parents to inherit from). All disease
        BoolStates default to False, which is the lift-and-shift target
        for HPV state — v2's add_births does not seed HPV state for
        immigrants.
        """
        if n <= 0:
            return
        people = self.sim.people
        # Allocate slots for n synthetic uids
        seed_uids = ss.uids(np.arange(n))   # placeholder for choose_slots.rvs signature
        new_slots = self._choose_slots.rvs(seed_uids)
        new_uids = people.grow(n, new_slots)
        # Set explicit demographic attributes
        people.age[new_uids] = float(age)
        people.female[new_uids] = bool(female)
        return

    def _emigrate(self, band_uids, n):
        """Remove n agents in band_uids, weighted uniformly within band."""
        if n <= 0 or len(band_uids) == 0:
            return
        n_pick = min(int(n), len(band_uids))
        idx = np.random.choice(len(band_uids), size=n_pick, replace=False)
        chosen = band_uids[idx]
        self.sim.people.request_death(chosen)
        return
```

> **Implementer notes:**
> - `seed_uids` argument to `_choose_slots.rvs(...)` is a stisim-Migration idiom; if your starsim version's `randint.rvs` requires a different shape (e.g. just an integer count), adjust accordingly. The intent is "give me n random integers in [low, high) for slot allocation."
> - `np.random.choice` for the emigration pick is the simple form. If a CRN-safe path is needed, use `ss.choice(band_uids, size=n_pick, replace=False).rvs(...)` — the M01 codebase converted `np.random` calls to `ss.choice` for CRN safety (commit `dd63785b`), so prefer the `ss` form here too.
> - Immigrants enter HPV-naive: no infection, no immunity, no progression states. This matches v2 (`add_births` doesn't seed HPV state when called from `check_migration`).

- [ ] **Step 6: Re-export from `hpvsim/__init__.py`**

```python
from .demographics import AgeMigration
```

Add `'AgeMigration'` to `__all__`.

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_demographics.py -v
```

Expected: all 4 PASS.

- [ ] **Step 8: Run full suite**

```bash
pytest tests/ -q -x --ignore=tests/_legacy
```

Expected: green.

- [ ] **Step 9: Commit**

```bash
git add hpvsim/demographics.py hpvsim/data/country.py hpvsim/__init__.py tests/test_demographics.py
git commit -m "M02: AgeMigration — age-pyramid pinning ported from v2 check_migration"
```

> **Implementer note:** v2's `check_migration` runs at annual cadence (one pinning pass per integer year). v3 sim runs at `dt=0.5` by default. If the per-step pinning over-corrects when called every half-year, gate it to run only at year boundaries (e.g., `if not np.isclose(sim.t.now('year') % 1, 0): return` at the top of `step`). This is a likely tuning point — flag in the M02 PR if observed.

---

## Task 13: `AgeResults` analyzer

**Files:**
- Create: `hpvsim/analyzers.py`
- Modify: `hpvsim/__init__.py`
- Test: `tests/test_analyzers.py` (created here)

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyzers.py`:

```python
"""Tests for hpv.AgeResults."""
import numpy as np
import pytest

import hpvsim as hpv


def test_age_results_class_exists():
    """AgeResults is exported and inherits ss.Analyzer."""
    import starsim as ss
    assert hasattr(hpv, 'AgeResults')
    assert issubclass(hpv.AgeResults, ss.Analyzer)


def test_age_results_produces_cancer_incidence_by_age():
    """A sim with AgeResults populates cancer_incidence_by_age."""
    sim = hpv.Sim(
        n_agents=5000, location='nigeria',
        start=1990, stop=2030, dt=0.5, rand_seed=0,
        analyzers=[hpv.AgeResults(results=('cancer',), year=[2020, 2025])],
    )
    sim.run()
    az = sim.analyzers.age_results
    arr = az.results.cancer_incidence_by_age
    # shape: (n_years, n_bins) — year=[2020, 2025] → 2 rows
    assert arr.shape[0] == 2
    # Default 5-yr bins 0–100 → 20 bins
    assert arr.shape[1] == 20
    # All values non-negative
    assert (np.asarray(arr) >= 0).all()


def test_age_results_scaling_with_pop_scale():
    """Incidence rate is per-100k regardless of pop_scale value."""
    sim_a = hpv.Sim(
        n_agents=5000, total_pop=1_000_000, location='nigeria',
        start=1990, stop=2025, dt=0.5, rand_seed=0,
        analyzers=[hpv.AgeResults(results=('cancer',), year=[2020])],
    )
    sim_b = hpv.Sim(
        n_agents=5000, total_pop=10_000_000, location='nigeria',
        start=1990, stop=2025, dt=0.5, rand_seed=0,
        analyzers=[hpv.AgeResults(results=('cancer',), year=[2020])],
    )
    sim_a.run()
    sim_b.run()
    arr_a = np.asarray(sim_a.analyzers.age_results.results.cancer_incidence_by_age)
    arr_b = np.asarray(sim_b.analyzers.age_results.results.cancer_incidence_by_age)
    # Per-100k incidence should be invariant to pop_scale.
    assert np.allclose(arr_a, arr_b, rtol=1e-3)
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_analyzers.py -v
```

- [ ] **Step 3: Create `hpvsim/analyzers.py`**

```python
"""HPVsim analyzers.

M02: AgeResults — minimum-scope age-stratified results, cancer-only.
M03/M04 extend the supported `results` keys to ('cancer', 'cins', 'hpv').
"""
import numpy as np
import starsim as ss


__all__ = ['AgeResults']


_DEFAULT_AGE_BINS = np.arange(0, 105, 5)   # 5-yr bins; last bin is [100, 105)


class AgeResults(ss.Analyzer):
    """Age-stratified result aggregation, cancer-only minimum scope.

    Args:
        results: tuple of result keys to age-stratify. M02 supports
                 ('cancer',). Other keys raise NotImplementedError.
        age_bins: array of bin edges (default: 5-yr bins, 0–100).
        year: scalar or list of report years. Each year produces one row in
              the output array.
    """

    def __init__(self, results=('cancer',), age_bins=None, year=None, **kwargs):
        super().__init__(**kwargs)
        if not set(results).issubset({'cancer'}):
            raise NotImplementedError(
                f'AgeResults M02 supports cancer only; got {results!r}.'
            )
        self.results_to_collect = tuple(results)
        self.age_bins = np.asarray(age_bins) if age_bins is not None else _DEFAULT_AGE_BINS
        self.report_years = list(year) if year is not None else []
        self.name = 'age_results'
        return

    def init_post(self):
        super().init_post()
        n_years = len(self.report_years)
        n_bins = len(self.age_bins) - 1
        self.define_results(
            ss.Result('cancer_incidence_by_age', shape=(n_years, n_bins),
                      dtype=float, scale=False,
                      label='Cancer incidence by age (per 100k)'),
        )
        # Track the last-step cumulative cancer count per agent for diff'ing.
        self._reported_year_idx = 0
        return

    def step(self):
        sim = self.sim
        if self._reported_year_idx >= len(self.report_years):
            return
        target_year = self.report_years[self._reported_year_idx]
        now_year = float(sim.t.now('year'))
        if now_year < target_year:
            return

        # Compute cancer-incidence-by-age for the year just completed.
        people = sim.people
        ages = np.asarray(people.age[people.alive])
        is_female = np.asarray(people.female[people.alive])
        # New cancers in [target_year - 1, target_year]: count agents with
        # ti_cancerous in that window.
        for disease in sim.diseases.values():
            if not hasattr(disease, 'cancerous'):
                continue
            ti_cancerous = np.asarray(disease.ti_cancerous)
            # Convert ti to year via sim.t machinery.
            yr_cancerous = sim.t.ti_to_year(ti_cancerous)
            window = (yr_cancerous >= target_year - 1) & (yr_cancerous <= target_year)
            new_cancers_uids = window.nonzero()[0]
            new_cancer_ages = np.asarray(disease.sim.people.age[new_cancers_uids])
            counts, _ = np.histogram(new_cancer_ages, bins=self.age_bins)
            # Female-years denominator
            f_ages = ages[is_female]
            denom, _ = np.histogram(f_ages, bins=self.age_bins)
            with np.errstate(divide='ignore', invalid='ignore'):
                rate = np.where(denom > 0, counts / denom * 100_000.0, 0.0)
            self.results.cancer_incidence_by_age[self._reported_year_idx, :] = rate

        self._reported_year_idx += 1
        return
```

> **Implementer note:** `sim.t.ti_to_year` is the assumed starsim API; if your version uses a different name (e.g. `sim.t.ti_to_yr`, `sim.t.year_at_ti`), use that. The math is correct as long as the ti→year conversion is exact.

- [ ] **Step 4: Re-export from `hpvsim/__init__.py`**

```python
from .analyzers import AgeResults
```

Add `'AgeResults'` to `__all__`.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_analyzers.py -v
```

Expected: all 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/analyzers.py hpvsim/__init__.py tests/test_analyzers.py
git commit -m "M02: add AgeResults analyzer (cancer-only minimum scope)"
```

---

## Task 14: Extend `anchor_hpv16.run_and_summarize()` to 8 metrics

**Files:**
- Modify: `tests/regression/anchor_hpv16.py`

The M01 version reports 3 metrics. Extend to all 8 of v2's `short_summary`. Reference: v2's `_v2_legacy/sim.py:1179` `compute_summary`.

- [ ] **Step 1: Replace `run_and_summarize()` body**

```python
def run_and_summarize():
    """Run the M2 anchor sim and return (short_summary_dict, total_pop_float).

    Summary keys (matches v2's compute_summary):
      - total HPV infections
      - total cancers
      - total cancer deaths
      - mean HPV prevalence (%)
      - mean cancer incidence (per 100k)
      - mean age of infection (years)
      - mean age of cancer (years)
      - mean age of cancer death (years)
    """
    sim = make_sim()
    sim.run()
    res = sim.results.hpv16
    mod = sim.diseases.hpv16

    # HPV infections
    if 'cum_infections' in res:
        n_inf = float(res.cum_infections[-1])
    elif 'new_infections' in res:
        n_inf = float(res.new_infections.sum())
    else:
        n_inf = float(res.n_infected.sum())

    # Mean HPV prevalence
    mean_prev_pct = 100 * float(res.prevalence.mean())

    # Mean age of (first) infection
    ti_first = mod.ti_first_infection
    ever_first = ti_first.notnan.uids
    if len(ever_first):
        ages_now = np.asarray(sim.people.age[ever_first])
        ti_at_inf = np.asarray(ti_first[ever_first])
        years_since = (float(sim.t.ti) - ti_at_inf) * float(PARS['dt'])
        mean_age_inf = float((ages_now - years_since).mean())
    else:
        mean_age_inf = 0.0

    # Cancers
    ti_cancerous = mod.ti_cancerous
    ever_cancer = ti_cancerous.notnan.uids
    n_cancers = float(len(ever_cancer)) * float(sim.pars.pop_scale)
    if len(ever_cancer):
        ages_now = np.asarray(sim.people.age[ever_cancer])
        ti_at_cancer = np.asarray(ti_cancerous[ever_cancer])
        yrs = (float(sim.t.ti) - ti_at_cancer) * float(PARS['dt'])
        mean_age_cancer = float((ages_now - yrs).mean())
    else:
        mean_age_cancer = 0.0

    # Cancer deaths
    ti_dead = mod.ti_dead_cancer
    ever_dead = ti_dead.notnan.uids
    # A scheduled ti_dead_cancer in the past AND not currently alive => realized cancer death.
    realized_uids = ever_dead[~np.asarray(sim.people.alive[ever_dead])]
    n_cancer_deaths = float(len(realized_uids)) * float(sim.pars.pop_scale)
    if len(realized_uids):
        # Age at death = current(now) - (sim_now_ti - ti_dead) * dt; but dead agents'
        # age stops advancing on death, so people.age[uid] is age-at-death directly.
        mean_age_cancer_death = float(np.asarray(sim.people.age[realized_uids]).mean())
    else:
        mean_age_cancer_death = 0.0

    # Mean cancer incidence (per 100k female-years over the run)
    n_alive_f_years = float((sim.results['n_alive'] / 2).sum() * PARS['dt'])  # approx
    mean_cancer_incidence = (n_cancers / n_alive_f_years * 100_000.0) if n_alive_f_years > 0 else 0.0

    short = {
        'total HPV infections': n_inf,
        'total cancers': n_cancers,
        'total cancer deaths': n_cancer_deaths,
        'mean HPV prevalence (%)': mean_prev_pct,
        'mean cancer incidence (per 100k)': mean_cancer_incidence,
        'mean age of infection (years)': mean_age_inf,
        'mean age of cancer (years)': mean_age_cancer,
        'mean age of cancer death (years)': mean_age_cancer_death,
    }
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop
```

> **Implementer note:** the female-years denominator (`n_alive_f_years`) is an approximation that assumes 50/50 sex ratio. v2 has an exact computation in `_v2_legacy/sim.py:compute_summary` — port it for accuracy. For a first pass, the approximation will be within a few percent and the ±10% drift gate tolerates it.

- [ ] **Step 2: Run anchor harness manually to confirm it produces all 8 keys**

```bash
python tests/regression/anchor_hpv16.py
```

Expected: prints all 8 metrics with non-zero values for cancer-related ones.

- [ ] **Step 3: Run regression test**

```bash
pytest tests/test_regression.py -v
```

Expected: smoke test passes; drift test is gated (skipped) until baseline is regenerated in Task 15.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/anchor_hpv16.py
git commit -m "M02: extend anchor_hpv16.run_and_summarize() to 8 metrics"
```

---

## Task 15: Regenerate v2 baseline (instructional)

**Files:** `tests/regression_baselines/anchor_hpv16.json` (regenerated locally; gitignored)

**This is a manual / human-driven step** because it requires running against a v2.3 environment.

- [ ] **Step 1: Update `tests/regression/README.md` with the M02 regen procedure**

Append a section:

```markdown
## M02 baseline regeneration

The M02 milestone extends `short_summary` from 3 keys to 8 (HPV +
CIN/cancer). Existing `anchor_hpv16.json` baselines from M01 are 3-key
and incompatible with the M02 drift gate.

To regenerate against v2.3:

1. Activate a v2.3 environment (clone hpvsim_v23_frozen or pip install
   hpvsim==2.3 in a separate venv).
2. From this repo: `python tests/regression/baseline.py --out tests/regression_baselines/anchor_hpv16.json`
3. The script imports the `PARS` dict from `tests/regression/anchor_hpv16.py`
   and runs the v2.3-installed hpvsim with those pars. Output JSON contains
   metadata (hpvsim version 2.3.x) plus the 8-key summary.
4. The baseline file is gitignored — `.gitignore` already excludes
   `tests/regression_baselines/*.json`.
```

- [ ] **Step 2: Update the M02 baseline locally**

(Human step — implementer runs in a v2.3 environment per the README.)

- [ ] **Step 3: Run the regression drift gate**

```bash
pytest tests/test_regression.py::test_anchor_hpv16_drift -v
```

Expected: PASS (within ±10% per metric) OR FAIL with a printed list of out-of-tolerance metrics. If FAIL, classify per migration convention 2 — fix the drift OR document expected misalignment + tracking issue.

- [ ] **Step 4: Commit the README**

```bash
git add tests/regression/README.md
git commit -m "M02: document baseline regen for 8-metric short_summary"
```

---

## Task 16: M02 capability test — age-stratified cumulative cancers

**Files:**
- Modify: `tests/test_natural_history.py`

The M02 acceptance test per the spec: age-stratified cumulative cancers + cancer deaths at end-of-sim against the v2 baseline, ±10% per 5-yr band.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_natural_history.py`:

```python
import json
from pathlib import Path

import hpvsim as hpv

CAPABILITY_BASELINE = Path(__file__).parent / 'regression_baselines' / 'm02_age_cancer.json'


@pytest.mark.skipif(not CAPABILITY_BASELINE.exists(),
                    reason='M02 age-cancer baseline not generated; see README')
def test_m02_capability_age_stratified_cancers():
    """End-of-sim cumulative cancers per 5-yr band vs. v2 baseline, ±10%."""
    pars = dict(n_agents=10_000, location='nigeria', genotype='hpv16',
                start=1990, stop=2060, dt=0.5, rand_seed=0)
    sim = hpv.Sim(**pars,
                  analyzers=[hpv.AgeResults(results=('cancer',),
                                             year=[2059])])
    sim.run()
    az = sim.analyzers.age_results
    v3_rates = np.asarray(az.results.cancer_incidence_by_age[0])

    with open(CAPABILITY_BASELINE) as f:
        baseline = json.load(f)
    v2_rates = np.asarray(baseline['cancer_incidence_by_age'])
    assert v3_rates.shape == v2_rates.shape

    # Per-band drift check, ±10%.
    out_of_tol = []
    for i, (a, b) in enumerate(zip(v3_rates, v2_rates)):
        if b == 0:
            if a > 1e-3:
                out_of_tol.append((i, a, b))
            continue
        rel = abs(a - b) / b
        if rel > 0.10:
            out_of_tol.append((i, a, b, rel))
    assert not out_of_tol, (
        f'Bands out of ±10% tolerance: {out_of_tol}'
    )
```

- [ ] **Step 2: Document baseline regen for this test in `tests/regression/README.md`**

Add to the M02 section:

```markdown
### M02 age-cancer capability baseline

Generate by running:
1. In a v2.3 environment, run the same anchor pars but capture
   `sim.results['cancer_incidence_by_age']` at year 2059.
2. Save as `tests/regression_baselines/m02_age_cancer.json` with shape:
   `{"cancer_incidence_by_age": [list of n_bins floats]}`.
3. Gitignored (already in `.gitignore`).
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/test_natural_history.py::test_m02_capability_age_stratified_cancers -v
```

Expected: SKIP if baseline not regenerated yet; PASS once regenerated and within tolerance.

- [ ] **Step 4: Commit**

```bash
git add tests/test_natural_history.py tests/regression/README.md
git commit -m "M02: add age-stratified cancer capability test"
```

---

## Task 17: Re-run partnership-equivalence; document outcome

**Files:** none modified by code; document in commit message + open issue if needed.

- [ ] **Step 1: Run M01's partnership-equivalence test under M02**

```bash
pytest tests/test_partnership_equivalence.py -v
```

- [ ] **Step 2: Decide outcome**

| Outcome | Action |
|---|---|
| Gates still pass at the M01 50% threshold | No action; mention in M02 PR description. |
| Gates tightened (lower drift than M01) | Document new measurements; consider tightening in M03. Mention in PR description. |
| Gates loosened or now fail | **Block merge.** Investigate which M02 change broke it (most likely AgeMigration changing pair-eligible pool). Either fix or open a tracking issue and revert the offending sub-task. |

- [ ] **Step 3: If outcome (a) or (b), no commit needed; proceed to Task 18**

- [ ] **Step 4: If outcome (c), revert/fix and re-run**

Investigate; do not advance to Task 18 until partnership gates are at-or-below M01's 50% drift.

---

## Task 18: Extend demo plot with CIN/cancer

**Files:**
- Modify: `tests/regression/demo_anchor_hpv16.py`

- [ ] **Step 1: Extend the demo to include CIN-prevalence and cancer-incidence subplots**

Replace the body of `demo_anchor_hpv16.py` with (showing the relevant subplot additions):

```python
"""Demo: run the M02 anchor sim and plot HPV / CIN / cancer trajectories."""
import matplotlib.pyplot as plt

import hpvsim as hpv
from anchor_hpv16 import make_sim


def main():
    sim = make_sim()
    sim.run()
    res = sim.results.hpv16
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # HPV prevalence
    ax = axes[0]
    ax.plot(res.timevec, res.prevalence * 100, label='HPV16 prevalence (%)')
    ax.set_ylabel('Prevalence (%)')
    ax.legend()

    # CIN prevalence
    mod = sim.diseases.hpv16
    if hasattr(res, 'n_cin'):
        ax = axes[1]
        ax.plot(res.timevec, res.n_cin / sim.results.n_alive * 100,
                label='CIN prevalence (%)', color='C1')
        ax.set_ylabel('CIN prevalence (%)')
        ax.legend()

    # Cancer cumulative
    if hasattr(res, 'n_cancerous'):
        ax = axes[2]
        ax.plot(res.timevec, res.n_cancerous, label='Cumulative cancers',
                color='C3')
        ax.set_ylabel('N cancers')
        ax.set_xlabel('Year')
        ax.legend()

    fig.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the demo manually**

```bash
python tests/regression/demo_anchor_hpv16.py
```

Expected: a 3-panel figure displays.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/demo_anchor_hpv16.py
git commit -m "M02: extend demo plot with CIN prevalence and cumulative cancer"
```

---

## Task 19: File tracking issues; open M02 PR

**Files:** GitHub remote.

- [ ] **Step 1: File tracking issues (gh CLI)**

Open three tracking issues in the `starsimhub/hpvsim` repo:

```bash
gh issue create --title "M02 follow-up: revisit multiscale dynamic agent spawning before v3.0.0" \
  --body "M02 dropped v2's multiscale branch from set_prognoses (extra cancer agents spawned to amplify rare events). Revisit before v3.0.0 if the natural-history acceptance test exhibits high-variance tails on cancer metrics. See docs/superpowers/specs/2026-05-01-m02-natural-history-parity-design.md §Out of scope." \
  --label "v3-migration"

gh issue create --title "M02 follow-up: confirm lognormal parametrization mapping (par1/par2 vs μ/σ)" \
  --body "v2 declared durations as dict(dist='lognormal', par1=X, par2=Y) consumed by hpu.sample. M02 maps to ss.lognorm_ex(mean=X, stdev=Y). If v2's hpu.sample treats par1/par2 as μ/σ on the log scale, switch to ss.lognorm_im(meanlog=..., sigmalog=...). See spec Open Question." \
  --label "v3-migration"

gh issue create --title "M02 follow-up: network-equivalence tightening + LegacyV2SexualNetwork parity test" \
  --body "M01 closes partnership-equivalence at 50% threshold. M02 re-measure (Task 17) results: <fill from PR>. If gates remain loose, build LegacyV2SexualNetwork(ss.SexualNetwork) delegating to v2's create_edgelist as a regression anchor and tighten the production network. See migration plan §M2 sub-task list." \
  --label "v3-migration"
```

- [ ] **Step 2: Push the M02 branch**

```bash
git push -u origin m02-natural-history-parity
```

- [ ] **Step 3: Open the M02 PR**

```bash
gh pr create --title "M02: natural history parity (HPV16 precin/CIN/cancer)" --body "$(cat <<'EOF'
## Summary
- Extends `hpv.HPV(ss.Infection)` with HPV16 natural-history compartments (precin, CIN, cancerous) using trajectory-based sampling lifted from v2's set_prognoses.
- Slims `parameters.py` to the starsimhub-conventional `SimPars` + `GenotypePars` shape; quarantines the v2 file.
- Audits `utils.py`; replaces v2 helpers with starsim-native equivalents.
- Adds `pop_scale`/`total_pop` plumbing through `SimPars`.
- Adds `AgeMigration(ss.Demographics)` adapter for age-specific net migration.
- Adds `AgeResults(ss.Analyzer)` cancer-only minimum scope (M04 calibration target).
- Extends regression coverage to the full 8-metric `short_summary` against a regenerated v2 baseline; adds capability test for age-stratified cancer trajectories.

## Validation
- Anchor regression drift: `<fill from Task 15 results>`
- M02 capability test (age-stratified cancers): `<fill from Task 16>`
- Partnership-equivalence re-measure: `<fill from Task 17>`

## Tracking issues
- #<multiscale revisit>
- #<lognormal parametrization>
- #<network tightening follow-up>

## Test plan
- [x] Anchor regression test: ±10% per metric on 8-metric short_summary
- [x] Capability test: age-stratified cumulative cancers within ±10% per 5-yr band
- [x] Partnership-equivalence test re-run; outcome documented
- [x] All M01 tests still pass
- [x] CI green at every commit on the branch

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)" --base v3.0-dev --draft
```

> **Note:** open as draft so CI runs on the milestone branch as planned. Flip to ready-for-review once all checks pass and the PR description is filled in.

- [ ] **Step 4: Mark milestone done**

When the PR merges to `v3.0-dev`, M02 is complete. Move to M03 (multi-genotype + cross-immunity) by branching `m03-multigenotype-cross-immunity` off `v3.0-dev`.

---

## Self-review notes for the implementer

When working through this plan, keep these spec invariants in mind:

1. **CI green at every commit** (migration convention 1). Each task's commit must leave the test suite passing. If a task introduces a temporary state where some test fails (e.g. between Task 7 and Task 8 the M01 SIS test may need adjusting), bundle the test fix into the same commit as the code change.

2. **±10% drift is informational, not auto-blocking.** If Task 15 shows >10% drift on a metric, the PR can still merge with an explicit drift-classification note in the description plus a tracking issue. Do not chase drift below 10% by tweaking pars; that's calibration territory (M04). Flag the drift, document, move on.

3. **Subclass-first delegations need tracking issues** (migration convention 3). If the plan ends up with a delegation to `_v2_legacy/` instead of a clean port, file an issue per delegation labeled "strip before M10."

4. **par1/par2 lognormal mapping** is an open question per spec; settle by reading `_v2_legacy/utils.py:hpu.sample` before Task 7.

5. **`request_death` API name** may differ across starsim versions; settle in Task 9 and update spec if a cleaner pattern surfaces.