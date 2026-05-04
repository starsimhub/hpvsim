# M01: Basic Transmission Sim — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In-place replacement of v2 `hpvsim` with the minimum runnable HPV sim on Starsim — single-genotype HPV16 transmission-only disease module + ported v2 sexual network + thin `Sim` wrapper + data adapter — and validate against v2 via a new 1-genotype Nigeria regression baseline and partnership-pattern equivalence tests.

**Architecture:** `hpvsim/` is rewritten in place — new entry points (`HPV`, `SexualNetwork`, `Sim`, `data.load_country`) replace the v2 implementations. v2 modules untouched by M01 are quarantined to `hpvsim/_v2_legacy/`; v2 tests that exercise removed APIs go to `tests/_legacy/`. Quarantines are never imported by active code; M10 deletes them wholesale. Rotasim-style multi-genotype design (one `HPV(ss.Infection)` per genotype, instantiated with N=1 for M01); cross-immunity connector deferred to M03. Two `SexualNetwork(ss.SexualNetwork)` instances (one per v2 layer m/c); cross-layer concurrency at `add_pairs` time via `isinstance` filtering of sibling networks. (An earlier draft assumed three layers including 'o' based on a misleading comment in v2's parameters.py; verification confirmed only m and c exist in v2.)

**Tech Stack:** Python 3.13, Starsim 3.3.3, Sciris, NumPy, Pandas, pytest, gh CLI.

**Reference design:** `docs/superpowers/specs/2026-04-28-hpvsim-m1-basic-transmission-design.md`

**Branch:** `m01-basic-transmission-sim` (cut from `v3.0-dev` after Task 0 housekeeping commits land).

---

## File structure after M01

| Path | Action | Responsibility |
|------|--------|---------------|
| `MIGRATION_PLAN.md` | Modify | M02/M03 split; renumber M03–M09 → M04–M10; add quarantine convention to §"Implementation conventions" |
| `hpvsim/__init__.py` | Rewrite | Export only `HPV`, `SexualNetwork`, `Sim`, `data`, plus version/options |
| `hpvsim/hpv.py` | Create | `HPV(ss.Infection)` single-genotype disease module |
| `hpvsim/network.py` | Create | `SexualNetwork(ss.SexualNetwork)` partnership network |
| `hpvsim/sim.py` | Replace | Old 1395-line file moves to `_v2_legacy/`; new tiny `Sim(ss.Sim)` wrapper takes its place |
| `hpvsim/data/__init__.py` | Modify | Re-export new `load_country` |
| `hpvsim/data/country.py` | Create | `load_country(location)` adapter wrapping v2 loaders |
| `hpvsim/_v2_legacy/__init__.py` | Create | Empty marker; "quarantine — not imported by active code" |
| `hpvsim/_v2_legacy/{analysis,base,calibration,hiv,immunity,interventions,people,plotting,population,run,sim}.py` | `git mv` | Quarantined v2 modules |
| `hpvsim/{parameters,defaults,settings,misc,utils,version}.py`, `hpvsim/data/{loaders,downloaders}.py`, `hpvsim/data/files/`, `hpvsim/regression/` | Keep | Utilities, data plumbing, v2 pars artifacts; not user-facing API |
| `tests/test_data.py` | Create | Tests for `load_country` |
| `tests/test_hpv.py` | Create | Tests for `HPV` (functional via minimal sim) |
| `tests/test_network.py` | Create | Tests for `SexualNetwork` (formation, dissolution, concurrency, mixing) |
| `tests/test_sim.py` | Create | Integration tests for `Sim` |
| `tests/test_partnership_equivalence.py` | Create | M01 acceptance gate vs. v2 partnership patterns |
| `tests/regression/anchor_hpv16.py` | Create | M01 1-genotype Nigeria HPV16 anchor harness |
| `tests/regression/demo_anchor_hpv16.py` | Create | M01 demo: runs anchor and plots aggregate prevalence |
| `tests/regression/README.md` | Modify | Document M01 1-gt baseline generation procedure |
| `tests/test_regression.py` | Modify | Add `test_anchor_hpv16_runs` + gated `test_anchor_hpv16_drift`; mark `test_anchor_runs` (M00 4-gt) skipped until M03 |
| `tests/regression/anchor.py` | Keep | M00 4-gt anchor — unchanged; harness still imports it but smoke is skipped |
| `tests/_legacy/__init__.py` | Create | Empty marker |
| `tests/_legacy/test_*.py` | `git mv` | All currently-broken v2 tests (everything except `test_regression.py`) |

`tests/regression_baselines/anchor_hpv16.json` and `tests/regression_baselines/partnership_v2.json` are gitignored local artifacts.

---

## Task ordering and dependencies

```
Task 0 (housekeeping on v3.0-dev)
   ↓
Task 1 (cut m01 branch)
   ↓
Task 2 (quarantine v2 modules + tests; rewrite hpvsim/__init__.py minimally)
   ↓
Task 3 (data adapter)  ──────────┐
Task 4 (HPV module)     ─────────┤
Task 5 (SexualNetwork scaffold)  ┤
Task 6 (SexualNetwork add_pairs) ┤
   ↓                              ├─→ Task 7 (Sim wrapper) ─→ Task 8 (final hpvsim/__init__.py)
                                  │      ↓
                                  │   Task 9 (anchor harness) ─→ Task 10 (smoke + M00-skip) ─→ Task 12 (drift, gated)
                                  │      ↓
                                  │   Task 11 (README baseline procedure)
                                  │      ↓
                                  └─→ Task 13 (partnership equivalence) ─→ Task 14 (demo script)
                                                                              ↓
                                                                          Task 15 (verify, push, PR)
```

---

## Task 0: Milestone housekeeping on `v3.0-dev`

**Branch:** `v3.0-dev` (NOT yet on `m01-basic-transmission-sim`)

**Why this is first:** The M01 spec depends on the milestone split (M02 = 1-gt nat hx, new M03 = multi-gt + cross-immunity, renumber M03–M09 → M04–M10). The plan and GitHub state must reflect this before the M01 branch cuts so M01's PR description can reference correct downstream milestone numbers.

**Files:**
- Modify: `MIGRATION_PLAN.md`
- Remote: GitHub milestones and issues

- [ ] **Step 1: Verify clean working tree on `v3.0-dev`**

```bash
git status
git branch --show-current
```

Expected: `v3.0-dev`, clean working tree (untracked files outside MIGRATION_PLAN.md are fine).

- [ ] **Step 2: Edit `MIGRATION_PLAN.md` — rewrite "M2: Natural history parity" to single-genotype**

Replace the current M2 section's body. Keep heading `### M2: Natural history parity`.

```markdown
### M2: Natural history parity

**Demo:** Show single-genotype (HPV16) natural history — clearance, precin, CIN, and cancer progression — matching v2.x.

**Acceptance test:** HPV → CIN → cancer dynamics for HPV16 match v2.x's HPV16-only run within calibration tolerance, against a v2 1-genotype baseline.

**Sub-tasks:**
- Port disease progression for HPV16 into `hpv.HPV(ss.Infection)` — add precin/CIN/cancer states, port duration distributions and progression functions from v2 `parameters.get_genotype_pars('hpv16')`.
- Port `age_results` analyzer at minimum scope (enough for calibration to consume in M4).
- Port `pop_scale` / `total_pop`.
- Port age-specific migration (from v2 `people.check_migration`).
- Add tests: HPV16 CIN/cancer trajectories match v2 1-genotype baseline.
```

- [ ] **Step 3: Edit `MIGRATION_PLAN.md` — insert new "M3: Multi-genotype and cross-immunity"**

Insert immediately after the (rewritten) M2 section:

```markdown
### M3: Multi-genotype and cross-immunity

**Demo:** 4-genotype HPV sim with cross-immunity matching v2's 4-genotype Nigeria baseline.

**Acceptance test:** Age-stratified HPV / CIN / cancer incidence by genotype overlaps v2.x intervals against the M0 4-genotype baseline.

**Sub-tasks:**
- Replicate `hpv.HPV(ss.Infection)` across all four genotypes `[16, 18, hi5, ohr]`; auto-instantiate via `hpv.Sim(genotypes=[...])` API.
- Add `hpv.CrossImmunity(ss.Connector)` implementing v2's cross-protection matrix.
- Wire genotype-specific natural history params (`rel_beta`, `dur_precin`, `dur_cin`, `cin_fn`, `cancer_fn`) per genotype.
- Add tests: 4-genotype prevalence + CIN + cancer trajectories match the M0 stored 4-genotype baseline.
```

- [ ] **Step 4: Edit `MIGRATION_PLAN.md` — renumber M3–M9 → M4–M10**

Renumber the remaining `### M3: ...` through `### M9: ...` sections to `### M4:` through `### M10:`. Update body-text cross-references (e.g., "deferred to M9" → "deferred to M10" if pointing at Release Readiness).

Verify:

```bash
grep -nE "^### M[0-9]+:" MIGRATION_PLAN.md
```

Expected: ten section headings, M0 through M10, in order.

- [ ] **Step 5: Edit `MIGRATION_PLAN.md` — update M1 sub-task list**

In `### M1: Basic transmission sim`, replace sub-tasks with:

```markdown
**Sub-tasks:**
- Port HPVsim's custom sexual network as `hpv.SexualNetwork(ss.SexualNetwork)` using lift-and-shift; one class, two instances (m/c); cross-layer concurrency via sibling iteration.
- Add a minimal single-genotype `hpv.HPV(ss.Infection)` for HPV16 — transmission only, SIS clearance, no precin/CIN/cancer.
- Replace `hpvsim.Sim` with thin `hpv.Sim(ss.Sim)` wrapper; rely on stock `ss.People`, `ss.Pregnancy`, `ss.Deaths`. Add `hpv.data.load_country()` adapter.
- Quarantine v2 modules untouched by M01 to `hpvsim/_v2_legacy/` and v2 tests to `tests/_legacy/`.
- Add tests for partnership pattern equivalence (age-mixing matrix, concurrency distribution, partnership duration distribution) vs. v2.x.
- Add tests for HPV prevalence trajectory vs. a new M1 1-genotype HPV16 baseline.
- Multi-resolution state (`scale`/`level0`/`level1`/`cluster`) deferred indefinitely; stock `ss.People` is sufficient.
```

- [ ] **Step 6: Edit `MIGRATION_PLAN.md` — add quarantine paragraph to §"Implementation conventions"**

Find the `## Implementation conventions` section and append:

```markdown
- **In-place replacement, with quarantines.** v3 work replaces `hpvsim/` in place. v2 modules untouched by the current milestone are moved to `hpvsim/_v2_legacy/`; v2 tests that exercise removed APIs are moved to `tests/_legacy/`. Active code never imports from either quarantine — quarantines exist purely as a porting reference. M10 deletes both wholesale.
```

- [ ] **Step 7: Update §"Branching and sync strategy" milestone references**

Search for any `M9` references that mean Release Readiness:

```bash
grep -nE "M9|merges into.*M[0-9]" MIGRATION_PLAN.md
```

Update each Release-Readiness reference from M9 to M10.

- [ ] **Step 8: Commit `MIGRATION_PLAN.md`**

```bash
git add MIGRATION_PLAN.md
git commit -m "$(cat <<'EOF'
Split M2 into M2 (1-gt nat hx) + new M3 (multi-gt + cross-immunity)

Rewrites M2 as single-genotype HPV16 natural history parity. Inserts new
M3 for multi-genotype + cross-immunity work (replicating HPV16 across the
four genotypes, adding hpv.CrossImmunity(ss.Connector), genotype-specific
natural history params). Renumbers M3-M9 to M4-M10. Updates M1 sub-task
list to reflect: rotasim-style direction, ss.SexualNetwork parent, three
instances per layer, isinstance-filtered cross-layer concurrency, no
hpv.People (multi-scale deferred), in-place replacement with v2
quarantine, and a new 1-gt M1 baseline. Adds quarantine convention to
Implementation conventions section.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9: GitHub — list current milestones**

```bash
gh api repos/:owner/:repo/milestones --paginate -q '.[] | "\(.number) \(.title)"'
```

Capture the output — you'll need each milestone's numeric `number` field for the rename API calls below.

- [ ] **Step 10: GitHub — rename milestones M03–M09 → M04–M10**

For each milestone titled `M03:` through `M09:` (in descending order to avoid name collisions during rename — rename M09 first, then M08, etc.):

```bash
# Example for one milestone — substitute MILESTONE_NUMBER and NEW_TITLE.
# Repeat for M09→M10, M08→M09, M07→M08, M06→M07, M05→M06, M04→M05, M03→M04.
gh api -X PATCH repos/:owner/:repo/milestones/MILESTONE_NUMBER \
  -f title="NEW_TITLE"
```

After: `gh api repos/:owner/:repo/milestones --paginate -q '.[] | .title' | sort` should show M00, M01, M02, M04, M05, M06, M07, M08, M09, M10 (no M03 yet).

- [ ] **Step 11: GitHub — create new "M03: Multi-genotype and cross-immunity"**

```bash
gh api -X POST repos/:owner/:repo/milestones \
  -f title="M03: Multi-genotype and cross-immunity" \
  -f description="Replicate hpv.HPV across all four genotypes [16, 18, hi5, ohr]; add hpv.CrossImmunity(ss.Connector); wire genotype-specific natural history params; regression vs. M00 4-gt baseline. See MIGRATION_PLAN.md §M3 for sub-task detail." \
  -f state=open
```

- [ ] **Step 12: GitHub — move multi-genotype-coupled issues from old M2 to new M3**

```bash
gh issue list --milestone "M02: Natural history parity" --state open --json number,title
```

For each issue whose scope is multi-genotype-coupled (cross-immunity, pop_scale, age_results analyzer, age-specific migration), move:

```bash
gh issue edit ISSUE_NUMBER --milestone "M03: Multi-genotype and cross-immunity"
```

Issues that remain in M02 should be HPV16-only natural history concerns. If unsure, leave in M02.

- [ ] **Step 13: GitHub — close M01 sub-task #98 with deferral note**

```bash
gh issue close 98 -c "Closing as not-needed for M01. Multi-resolution state (scale, level0/level1, cluster) is deferred indefinitely; stock ss.People is sufficient for M01-M03. See M01 design spec §Open items / §Out of scope."
```

- [ ] **Step 14: GitHub — retitle M01 sub-task #102**

```bash
gh issue edit 102 --title "Tests: HPV prevalence trajectory vs. v2 baseline (single-genotype HPV16, M01)"
```

- [ ] **Step 15: GitHub — update tracking issue #95**

```bash
gh issue view 95 --json body -q '.body' > /tmp/issue95.md
```

Edit `/tmp/issue95.md` to reflect the ten-milestone structure (M00–M10) with M02/M03 split. Then:

```bash
gh issue edit 95 --body-file /tmp/issue95.md
```

- [ ] **Step 16: Push `v3.0-dev` to origin**

```bash
git push origin v3.0-dev
```

---

## Task 1: Cut M01 branch

**Files:** none (branch operation only).

- [ ] **Step 1: Pull latest `v3.0-dev`**

```bash
git checkout v3.0-dev
git pull
```

- [ ] **Step 2: Cut the M01 branch**

```bash
git checkout -b m01-basic-transmission-sim
```

Expected: switched to `m01-basic-transmission-sim` based on the latest `v3.0-dev` (with Task 0 housekeeping included).

- [ ] **Step 3: Confirm**

```bash
git branch --show-current
git log --oneline -3
```

Expected: branch is `m01-basic-transmission-sim`; latest commit is the housekeeping commit from Task 0.

No commit; this task is a branch creation only.

---

## Task 2: Quarantine v2 modules and tests; minimal `hpvsim/__init__.py`

**Files:**
- `git mv` 11 files: `hpvsim/{analysis,base,calibration,hiv,immunity,interventions,people,plotting,population,run,sim}.py` → `hpvsim/_v2_legacy/`
- `git mv` v2 tests except `test_regression.py` → `tests/_legacy/`
- Create: `hpvsim/_v2_legacy/__init__.py`
- Create: `tests/_legacy/__init__.py`
- Modify: `hpvsim/__init__.py` (rewrite to minimal active surface)
- Modify: `tests/test_regression.py` (mark `test_anchor_runs` as skip)

This is one large commit — bulk moves preserve git history per file. Subsequent tasks restore active functionality.

- [ ] **Step 1: Create the two quarantine package markers**

```bash
mkdir -p hpvsim/_v2_legacy
mkdir -p tests/_legacy
```

Create `hpvsim/_v2_legacy/__init__.py`:

```python
"""Quarantine for v2 hpvsim modules awaiting port to v3 (Starsim).

DO NOT IMPORT FROM ACTIVE CODE. This package exists as a porting reference
during the v2 -> v3 migration. Each milestone (M02-M09) ports the relevant
modules out of here and into the active package surface. M10 deletes this
package wholesale.
"""
```

Create `tests/_legacy/__init__.py`:

```python
"""Quarantine for v2 hpvsim tests that exercise removed APIs."""
```

- [ ] **Step 2: Bulk-move v2 modules to quarantine**

```bash
git mv hpvsim/analysis.py       hpvsim/_v2_legacy/analysis.py
git mv hpvsim/base.py           hpvsim/_v2_legacy/base.py
git mv hpvsim/calibration.py    hpvsim/_v2_legacy/calibration.py
git mv hpvsim/hiv.py            hpvsim/_v2_legacy/hiv.py
git mv hpvsim/immunity.py       hpvsim/_v2_legacy/immunity.py
git mv hpvsim/interventions.py  hpvsim/_v2_legacy/interventions.py
git mv hpvsim/people.py         hpvsim/_v2_legacy/people.py
git mv hpvsim/plotting.py       hpvsim/_v2_legacy/plotting.py
git mv hpvsim/population.py     hpvsim/_v2_legacy/population.py
git mv hpvsim/run.py            hpvsim/_v2_legacy/run.py
git mv hpvsim/sim.py            hpvsim/_v2_legacy/sim.py
```

Verify:

```bash
ls hpvsim/_v2_legacy/
ls hpvsim/*.py
```

Expected: `_v2_legacy/` contains the 11 quarantined files plus `__init__.py`. `hpvsim/*.py` shows only `__init__.py`, `defaults.py`, `misc.py`, `parameters.py`, `settings.py`, `utils.py`, `version.py`.

- [ ] **Step 3: Bulk-move v2 tests to quarantine (except `test_regression.py`)**

```bash
# List all v2 test files first to confirm what's about to move:
ls tests/test_*.py
```

Move all `tests/test_*.py` except `test_regression.py`:

```bash
for f in tests/test_*.py; do
  base=$(basename "$f")
  if [ "$base" != "test_regression.py" ]; then
    git mv "$f" "tests/_legacy/$base"
  fi
done
```

Verify:

```bash
ls tests/test_*.py
ls tests/_legacy/
```

Expected: `tests/test_regression.py` is the only file matching `tests/test_*.py`; `tests/_legacy/` contains all the moved v2 test files.

- [ ] **Step 4: Rewrite `hpvsim/__init__.py` to minimal active surface**

Replace `hpvsim/__init__.py` content with:

```python
"""hpvsim — HPV simulation tools (Starsim-based).

This is the v3 package, in-place replacement of the legacy v2 package.
Public API expands as milestones land:

    M01: Sim, HPV, SexualNetwork, data            (this milestone)
    M02: + natural history (CIN, cancer)
    M03: + multi-genotype, cross-immunity
    M04: + calibration
    M05: + interventions
    ...

v2 modules awaiting port live in hpvsim/_v2_legacy/. Active code MUST NOT
import from there.
"""

import sciris as sc

# Stable utility imports — these modules stayed active through the migration:
from .version import __version__, __versiondate__, __license__
from .settings import options
from .defaults import datadir, default_int, default_float, get_default_plots
from . import data
from . import parameters
from . import misc
from . import utils

# M01 public API — populated as components land in Tasks 3-7. Imports are
# commented out for now and uncommented in Task 8 (final hpvsim/__init__.py).
# from .hpv import HPV
# from .network import SexualNetwork
# from .sim import Sim

del sc
```

- [ ] **Step 5: Mark M00 4-gt anchor smoke test as skipped**

In `tests/test_regression.py`, modify `test_anchor_runs` by inserting a `pytest.mark.skip` decorator above it.

Read the current `tests/test_regression.py`:

```bash
cat tests/test_regression.py | head -30
```

Apply this edit: change

```python
def test_anchor_runs():
```

to

```python
import pytest as _pytest_for_skip


@_pytest_for_skip.mark.skip(
    reason='Multi-genotype not yet ported to v3; restored in M03 when '
           'genotypes=[16, 18, hi5, ohr] is supported again.'
)
def test_anchor_runs():
```

(The aliased `_pytest_for_skip` is just to avoid colliding with any existing `pytest` import. If `pytest` is already imported at the top of the file, just use `@pytest.mark.skip(...)` instead.)

- [ ] **Step 6: Verify imports of remaining active modules work**

```bash
python -c "import hpvsim; print('imports OK; version:', hpvsim.__version__)"
python -c "import hpvsim.data; import hpvsim.parameters; import hpvsim.utils; print('utility imports OK')"
```

Expected: both prints succeed without `ImportError` or `ModuleNotFoundError`.

If `hpvsim.parameters` fails because it imports something now in `_v2_legacy/`, you'll see `ImportError`. Fix by editing `hpvsim/parameters.py` to import the missing dependency from `._v2_legacy.X` (leaky quarantine — acceptable as a temporary workaround; M02 cleanup inlines what's needed). The current scan (per `grep -nE "^from \.|^import" hpvsim/parameters.py`) showed parameters.py only imports `settings`, `misc`, `utils`, `defaults`, `data` — all stay active — so this should not be needed, but flag if it does happen.

- [ ] **Step 7: Verify pytest can collect the active test**

```bash
pytest --collect-only tests/test_regression.py
```

Expected: collection succeeds. `test_anchor_runs` shows up but will be skipped at runtime; `test_compute_drift_*` tests collect normally. There should be no collection errors.

If `tests/_legacy/test_*.py` shows collection errors when `pytest --collect-only tests/` is run, add a `tests/conftest.py` (or modify the existing one) with:

```python
collect_ignore_glob = ['_legacy/*']
```

This prevents pytest from collecting the quarantined tests at all.

- [ ] **Step 8: Commit**

```bash
git add hpvsim/_v2_legacy tests/_legacy hpvsim/__init__.py tests/test_regression.py
# If conftest.py was added or modified:
# git add tests/conftest.py
git commit -m "$(cat <<'EOF'
Quarantine v2 modules to hpvsim/_v2_legacy and v2 tests to tests/_legacy

Bulk git-mv of 11 v2 modules (analysis, base, calibration, hiv, immunity,
interventions, people, plotting, population, run, sim) into _v2_legacy/.
Bulk git-mv of all v2 test_*.py files except test_regression.py into
tests/_legacy/.

Rewrites hpvsim/__init__.py to a minimal active surface (utilities only;
M01 entry points wired in Task 8). Marks the M00 4-genotype anchor smoke
(test_anchor_runs) as skipped until M03 since multi-genotype is no longer
supported until that milestone.

The quarantines are reference-only: active code does not import from them.
M10 deletes both wholesale.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `hpv.data.load_country` adapter

**Files:**
- Create: `hpvsim/data/country.py`
- Modify: `hpvsim/data/__init__.py` (re-export `load_country`)
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data.py`:

```python
"""Tests for hpvsim.data.country.load_country — country-data adapter."""

import pandas as pd

import hpvsim


def test_load_country_returns_expected_keys():
    """load_country returns a dict with exactly the expected top-level keys."""
    out = hpvsim.data.load_country('nigeria')
    expected = {'age_data', 'fertility', 'death_rate', 'network_pars'}
    assert set(out.keys()) == expected, f'unexpected keys: {set(out.keys())}'


def test_age_data_shape():
    """age_data is a DataFrame with the columns ss.People accepts."""
    out = hpvsim.data.load_country('nigeria')
    df = out['age_data']
    assert isinstance(df, pd.DataFrame)
    assert {'age', 'value'}.issubset(df.columns), f'columns: {df.columns.tolist()}'
    assert len(df) > 0
    assert (df['value'] >= 0).all()


def test_fertility_shape():
    """fertility is a DataFrame ss.Pregnancy(fertility_rate=...) accepts."""
    out = hpvsim.data.load_country('nigeria')
    df = out['fertility']
    assert isinstance(df, pd.DataFrame)
    assert {'Time', 'AgeGrp', 'ASFR'}.issubset(df.columns), f'columns: {df.columns.tolist()}'
    assert len(df) > 0
    assert (df['ASFR'] >= 0).all()


def test_death_rate_shape():
    """death_rate is a DataFrame ss.Deaths(death_rate=...) accepts."""
    out = hpvsim.data.load_country('nigeria')
    df = out['death_rate']
    assert isinstance(df, pd.DataFrame)
    assert {'Year', 'AgeGrp', 'Sex', 'Rate'}.issubset(df.columns), f'columns: {df.columns.tolist()}'
    assert len(df) > 0
    assert (df['Rate'] >= 0).all()


def test_network_pars_per_layer():
    """network_pars contains entries for the two v2 layers m/c."""
    out = hpvsim.data.load_country('nigeria')
    np_pars = out['network_pars']
    assert set(np_pars.keys()) == {'m', 'c'}, f'layers: {list(np_pars.keys())}'
    expected = {'partners', 'mixing', 'layer_probs', 'cross_layer', 'duration', 'acts'}
    for layer, layer_pars in np_pars.items():
        assert expected.issubset(layer_pars.keys()), \
            f'layer {layer} missing keys: {expected - set(layer_pars.keys())}'


def test_unknown_location_raises():
    """Unknown location raises ValueError listing supported locations."""
    import pytest
    with pytest.raises(ValueError, match='nigeria'):
        hpvsim.data.load_country('atlantis')
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: failures — `AttributeError: module 'hpvsim.data' has no attribute 'load_country'`.

- [ ] **Step 3: Implement `load_country`**

Create `hpvsim/data/country.py`:

```python
"""Country-data adapter: wrap v2 hpvsim's location data into Starsim-shaped DataFrames.

Used by hpvsim.Sim to build People (age pyramid), Pregnancy (fertility),
Deaths (mortality), and SexualNetwork (per-layer partnership pars). All
underlying data lives in hpvsim/data/files/ and is loaded via the existing
hpvsim.data.loaders module and hpvsim.parameters helpers (which stayed
active through the v2 -> v3 migration).
"""

import pandas as pd

from .. import parameters as _params
from . import loaders as _loaders


_KNOWN_LOCATIONS = ['nigeria']  # M01 ships with Nigeria only; expand as needed.


def load_country(location):
    """Return Starsim-shaped data for ``location``.

    Args:
        location (str): country name; must be one of the supported locations
            available via the v2 loaders.

    Returns:
        dict with keys:
            - 'age_data': DataFrame [age, value]
            - 'fertility': DataFrame [Time, AgeGrp, ASFR]
            - 'death_rate': DataFrame [Year, AgeGrp, Sex, Rate]
            - 'network_pars': nested dict {layer: {key: value}}
              with keys: partners, mixing, layer_probs, cross_layer, duration, acts.
    """
    location = location.lower()
    if location not in _KNOWN_LOCATIONS:
        raise ValueError(
            f"Unknown location {location!r}. Supported locations: {_KNOWN_LOCATIONS}."
        )

    return dict(
        age_data=_age_data(location),
        fertility=_fertility(location),
        death_rate=_death_rate(location),
        network_pars=_network_pars(location),
    )


def _age_data(location):
    """Reshape v2's age distribution to (age, value) DataFrame."""
    ages = _loaders.get_age_distribution(location=location)
    rows = []
    for age_lower, age_upper, prob in ages:
        n_ages_in_bin = max(1, int(age_upper - age_lower))
        per_age = prob / n_ages_in_bin
        for a in range(int(age_lower), int(age_upper)):
            rows.append({'age': a, 'value': per_age})
    return pd.DataFrame(rows)


def _fertility(location):
    """Reshape v2 birth rates into [Time, AgeGrp, ASFR] long form."""
    raw = _loaders.get_birth_rates(location=location)
    rows = []
    for year, age_to_asfr in raw.items():
        for age, asfr in age_to_asfr.items():
            rows.append({'Time': year, 'AgeGrp': age, 'ASFR': asfr})
    return pd.DataFrame(rows)


def _death_rate(location):
    """Reshape v2 death rates into [Year, AgeGrp, Sex, Rate] long form."""
    raw = _loaders.get_death_rates(location=location, by_sex=True)
    rows = []
    for year, sex_to_age in raw.items():
        for sex, age_to_rate in sex_to_age.items():
            for age, rate in age_to_rate.items():
                rows.append({'Year': year, 'AgeGrp': age, 'Sex': sex, 'Rate': rate})
    return pd.DataFrame(rows)


def _network_pars(location):
    """Build per-layer network parameter dicts using v2 helpers + defaults.

    Returns {'m': {...}, 'c': {...}, 'o': {...}} where each layer dict has:
    partners, mixing, layer_probs, cross_layer, duration, acts.
    """
    default_pars = _params.make_pars(location=location)
    mixing = _params.get_mixing(network=default_pars.get('network', None))

    out = {}
    for layer in ('m', 'c'):
        out[layer] = dict(
            partners=default_pars['partners'][layer],
            mixing=mixing[layer],
            layer_probs=default_pars['layer_probs'][layer],
            cross_layer=default_pars['cross_layer'][layer],
            duration=default_pars['dur_pship'][layer],
            acts=default_pars['acts'][layer],
        )
    return out
```

- [ ] **Step 4: Re-export `load_country` from `hpvsim/data/__init__.py`**

Read the current `hpvsim/data/__init__.py`:

```bash
cat hpvsim/data/__init__.py
```

Append (or insert near the top, depending on existing structure):

```python
from .country import load_country

__all__ = list(set(globals().get('__all__', [])) | {'load_country'})
```

If `__all__` is not already used in this file, simpler:

```python
from .country import load_country
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: all six tests pass.

If a test fails because v2's data shape differs from what `_fertility`/`_death_rate`/`_age_data` assumed, inspect:

```bash
python -c "import hpvsim.data.loaders as L; import pprint; pprint.pp(L.get_birth_rates(location='nigeria'))"
```

and adjust the reshape logic accordingly.

If `_network_pars` fails because `parameters.make_pars` does not return the expected dict structure (e.g., the key is `partners_n` not `partners`, or `cross_layer` is a top-level dict not a per-layer dict), inspect:

```bash
python -c "import hpvsim.parameters as P; import sciris as sc; pars = P.make_pars(location='nigeria'); print(list(pars.keys())); print('partners:', pars.get('partners'))"
```

and adjust key names accordingly.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/data/country.py hpvsim/data/__init__.py tests/test_data.py
git commit -m "$(cat <<'EOF'
Add hpvsim.data.load_country adapter

Wraps v2 loaders (get_age_distribution, get_birth_rates, get_death_rates,
parameters.make_pars, parameters.get_mixing) and reshapes output into
Starsim-shaped pandas DataFrames consumed by ss.People, ss.Pregnancy,
ss.Deaths, and per-layer SexualNetwork instances. Nigeria is the only
supported location for M01.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `hpv.HPV(ss.Infection)` disease module

**Files:**
- Create: `hpvsim/hpv.py`
- Create: `tests/test_hpv.py`

**Note on testing approach:** `HPV` cannot be unit-tested in true isolation — it requires a `Sim` context. Tests instantiate a minimal `ss.Sim` with `ss.RandomNet` (the new `SexualNetwork` lands in Task 5-6) and run a few timesteps.

- [ ] **Step 1: Write failing tests**

Create `tests/test_hpv.py`:

```python
"""Functional tests for hpvsim.hpv.HPV via a minimal ss.Sim."""

import numpy as np
import pytest
import starsim as ss

from hpvsim.hpv import HPV


def _minimal_sim(genotype='hpv16', n_agents=1000, init_prev=0.05, beta=0.3,
                 dur_years=2.0, n_steps=4):
    """Build a minimal Sim with HPV(genotype=...) and a stock random network."""
    hpv = HPV(
        genotype=genotype,
        init_prev=ss.bernoulli(p=init_prev),
        beta=ss.peryear(beta),
        dur_inf=ss.lognorm_ex(mean=ss.years(dur_years)),
    )
    sim = ss.Sim(
        diseases=hpv,
        networks='random',
        n_agents=n_agents,
        dur=ss.years(n_steps * 0.5),
        dt=ss.years(0.5),
        verbose=0,
    )
    return sim, hpv


def test_genotype_attribute_set():
    hpv = HPV(genotype='hpv16')
    assert hpv.genotype == 'hpv16'


def test_unknown_genotype_rejected():
    with pytest.raises(ValueError, match='hpv16'):
        HPV(genotype='hpv99')


def test_init_prev_seeds_initial_cases():
    """init_prev=0.05 + n_agents=1000 yields ~50 initial cases (Bernoulli ±5σ)."""
    sim, hpv = _minimal_sim(init_prev=0.05, n_agents=1000)
    sim.init()
    n_initial = int(hpv.infected.sum())
    expected = 0.05 * 1000
    sigma = (0.05 * 0.95 * 1000) ** 0.5
    assert abs(n_initial - expected) < 5 * sigma, \
        f'initial cases {n_initial} far from expected {expected:.0f} ± {sigma:.1f}'


def test_set_prognoses_flips_state():
    """set_prognoses moves agents from susceptible to infected, sets ti_clearance > ti."""
    sim, hpv = _minimal_sim()
    sim.init()
    sus_uids = hpv.susceptible.uids
    assert len(sus_uids) >= 3
    target = sus_uids[:3]
    hpv.set_prognoses(target, sources=None)
    assert (~hpv.susceptible[target]).all()
    assert hpv.infected[target].all()
    assert (hpv.ti_clearance[target] > hpv.ti).all()


def test_step_state_clears_at_ti_clearance():
    """An agent whose ti_clearance is reached returns to susceptible (SIS)."""
    sim, hpv = _minimal_sim(init_prev=0.0)
    sim.init()
    target = ss.uids([0])
    hpv.set_prognoses(target, sources=None)
    hpv.ti_clearance[target] = hpv.ti
    hpv.step_state()
    assert hpv.susceptible[target].all()
    assert (~hpv.infected[target]).all()


def test_runs_a_few_timesteps():
    """End-to-end: minimal sim with random net runs without error."""
    sim, hpv = _minimal_sim(init_prev=0.05, n_agents=500, n_steps=4)
    sim.run()
    assert 'n_infected' in sim.results.hpv16
    assert sim.results.hpv16.n_infected[-1] >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hpv.py -v
```

Expected: failures with `ModuleNotFoundError: No module named 'hpvsim.hpv'`.

- [ ] **Step 3: Implement `HPV`**

Create `hpvsim/hpv.py`:

```python
"""HPV genotype as a Starsim Infection.

M01: single-genotype, transmission-only with SIS clearance (no precin/CIN/cancer).
M02 will add natural-history states (precin, CIN, cancer) and override step_state.
M03 will instantiate one HPV per genotype and add a CrossImmunity connector.
"""

import starsim as ss


_KNOWN_GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


class HPV(ss.Infection):
    """Single-genotype HPV disease module.

    The ``genotype`` attribute identifies which strain this instance models
    and is the duck-type marker M03's ``hpv.CrossImmunity`` connector will
    use to discover HPV diseases (mirrors rotasim's
    ``hasattr(disease, 'G')`` pattern).
    """

    def __init__(self, genotype='hpv16', pars=None, **kwargs):
        if genotype not in _KNOWN_GENOTYPES:
            raise ValueError(
                f'Unknown genotype {genotype!r}. Known: {list(_KNOWN_GENOTYPES)}.'
            )
        self.genotype = genotype
        if 'name' not in kwargs:
            kwargs['name'] = genotype
        super().__init__()
        self.define_pars(
            init_prev=ss.bernoulli(p=0.05),
            beta=ss.peryear(0.5),
            dur_inf=ss.lognorm_ex(mean=ss.years(2.0)),
        )
        self.update_pars(pars=pars, **kwargs)
        # ss.Infection already provides: susceptible, infected, rel_sus,
        # rel_trans, ti_infected. We add ti_clearance for SIS dynamics.
        self.define_states(
            ss.FloatArr('ti_clearance', label='Time of natural clearance'),
        )

    def set_prognoses(self, uids, sources=None):
        """Mark uids as infected; schedule clearance per dur_inf."""
        super().set_prognoses(uids, sources)
        ti = self.ti
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.ti_clearance[uids] = ti + self.pars.dur_inf.rvs(uids)

    def step_state(self):
        """SIS: agents past ti_clearance return to susceptible."""
        clearing = (self.infected & (self.ti_clearance <= self.ti)).uids
        if len(clearing):
            self.infected[clearing] = False
            self.susceptible[clearing] = True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hpv.py -v
```

Expected: all six tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_hpv.py
git commit -m "$(cat <<'EOF'
Add hpvsim.hpv.HPV(ss.Infection) — single-genotype transmission module

M01 scope: SIS clearance, no natural history. The genotype attribute is
the duck-type marker for M03's cross-immunity connector. Functional tests
exercise set_prognoses, step_state clearance, init_prev seeding, unknown-
genotype rejection, and end-to-end run via a minimal ss.Sim with
ss.RandomNet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `hpv.SexualNetwork` scaffold + cross-layer helper

**Files:**
- Create: `hpvsim/network.py`
- Create: `tests/test_network.py`

This task implements the class shell and the `_n_partners_elsewhere` helper but leaves `add_pairs` as a no-op. Task 6 ports the pair-formation algorithm. Splitting these allows independent verification of the cross-layer-filter behavior before tackling the full algorithm.

- [ ] **Step 1: Write failing tests for class scaffold + cross-layer filter**

Create `tests/test_network.py`:

```python
"""Tests for hpvsim.network.SexualNetwork."""

import numpy as np
import pytest
import starsim as ss

from hpvsim.network import SexualNetwork


def test_known_layers_accepted():
    for layer in ('m', 'c'):
        net = SexualNetwork(layer=layer)
        assert net.layer == layer


def test_unknown_layer_rejected():
    with pytest.raises(ValueError, match="m.*c.*o"):
        SexualNetwork(layer='x')


def test_n_partners_elsewhere_with_no_siblings_returns_zeros():
    """One-layer-only sim: helper returns all zeros."""
    net = SexualNetwork(layer='m')
    sim = ss.Sim(networks=[net], n_agents=200, diseases=None,
                 dur=ss.years(1), dt=ss.years(0.5), verbose=0)
    sim.init()
    n = net._n_partners_elsewhere()
    assert n.shape == (len(sim.people),)
    assert (n == 0).all()


def test_n_partners_elsewhere_filters_non_hpv_networks():
    """An ss.RandomNet sibling should NOT be counted."""
    hpv_m = SexualNetwork(layer='m')
    rand = ss.RandomNet(n_contacts=5)
    sim = ss.Sim(networks=[hpv_m, rand], n_agents=200, diseases=None,
                 dur=ss.years(1), dt=ss.years(0.5), verbose=0)
    sim.init()
    sim.step()
    n = hpv_m._n_partners_elsewhere()
    assert (n == 0).all(), \
        f'isinstance filter failed — {n.sum()} non-zero entries from non-hpv siblings'


def test_n_partners_elsewhere_counts_sibling_hpv_networks():
    """A sibling SexualNetwork instance contributes its edge endpoints."""
    hpv_m = SexualNetwork(layer='m')
    hpv_c = SexualNetwork(layer='c')
    sim = ss.Sim(networks=[hpv_m, hpv_c], n_agents=200, diseases=None,
                 dur=ss.years(1), dt=ss.years(0.5), verbose=0)
    sim.init()
    hpv_c.append(p1=ss.uids([0, 1]), p2=ss.uids([2, 3]),
                 beta=np.array([1.0, 1.0]),
                 dur=np.array([10.0, 10.0]))
    n = hpv_m._n_partners_elsewhere()
    assert n[0] == 1 and n[1] == 1 and n[2] == 1 and n[3] == 1
    assert n[4:].sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_network.py -v
```

Expected: failures with `ModuleNotFoundError: No module named 'hpvsim.network'`.

- [ ] **Step 3: Implement `SexualNetwork` scaffold**

Create `hpvsim/network.py`:

```python
"""HPVsim sexual partnership network.

Lift-and-shift of v2 hpvsim's three-layer (marital/casual/one-off) sexual
network. One class instantiated three times, one per layer; cross-layer
concurrency resolved at add_pairs time via isinstance-filtered iteration of
sibling networks. Inherits scaffolding (debut, participant, duration tracking,
end_pairs, net_beta) from ss.SexualNetwork.

Task 5 (this commit): class scaffold + cross-layer helper.
Task 6: port v2's create_edgelist into add_pairs.
"""

import numpy as np
import starsim as ss


_KNOWN_LAYERS = ('m', 'c')


class SexualNetwork(ss.SexualNetwork):
    """One layer of HPVsim's heterosexual partnership network.

    Args:
        layer: one of 'm' (marital/long-term), 'c' (casual).
        pars: dict of layer parameters; see hpvsim.data.load_country for
            the expected shape (partners, mixing, layer_probs, cross_layer,
            duration, acts).
    """

    def __init__(self, layer='m', pars=None, **kwargs):
        if layer not in _KNOWN_LAYERS:
            raise ValueError(
                f'Unknown layer {layer!r}. Known: {list(_KNOWN_LAYERS)}.'
            )
        self.layer = layer
        kwargs.setdefault('name', layer)
        super().__init__()
        self.define_pars(
            partners=ss.poisson(lam=1),
            mixing=None,
            layer_probs=None,
            cross_layer=0.0,
            duration=ss.lognorm_ex(mean=ss.years(5)),
            acts=ss.poisson(lam=ss.freqperyear(50)),
        )
        self.update_pars(pars=pars, **kwargs)

    def _n_partners_elsewhere(self):
        """Count current partnerships each agent has in OTHER hpv.SexualNetwork
        layers. Used by add_pairs (Task 6) for cross-layer concurrency
        eligibility.

        Returns:
            np.ndarray of int, shape (n_agents,). Returns all zeros if no
            sibling SexualNetwork instances exist.
        """
        n = np.zeros(len(self.sim.people), dtype=int)
        for other in self.sim.networks():
            if other is self:
                continue
            if not isinstance(other, SexualNetwork):
                continue
            if len(other) == 0:
                continue
            n[other.edges.p1] += 1
            n[other.edges.p2] += 1
        return n

    def add_pairs(self):
        """Pair-formation logic — implemented in Task 6."""
        return
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_network.py -v
```

Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/network.py tests/test_network.py
git commit -m "$(cat <<'EOF'
Add hpvsim.network.SexualNetwork scaffold with cross-layer helper

Class shell, layer validation, _n_partners_elsewhere with isinstance
filter for cross-layer concurrency. add_pairs is a no-op; Task 6 ports
v2's create_edgelist algorithm. Tests verify the isinstance filter
correctly excludes non-SexualNetwork siblings (e.g., ss.RandomNet) and
correctly counts sibling SexualNetwork edges.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Port `add_pairs` from v2's `create_edgelist`

**Files:**
- Modify: `hpvsim/network.py` (replace `add_pairs` no-op)
- Modify: `tests/test_network.py` (add behavioral tests)

**Reference:** v2 source for the algorithm being ported lives at `hpvsim/_v2_legacy/population.py`, function `create_edgelist`, lines ~281-379. The implementer should **read v2's `create_edgelist` end-to-end and port it faithfully**, with the four adaptations explicitly listed below.

### Adaptations from v2 → v3

1. `lno` (layer index) → `self.layer` (cosmetic only; no arithmetic).
2. `current_partners[lno, :] < partners` → per-uid count of own edges in `self.edges` compared against `self._partners_target` (sampled once at first add_pairs call from `self.pars.partners`).
3. `current_partners[other_layers, :].any(axis=0)` → `self._n_partners_elsewhere() > 0`.
4. `current_partners` updates → `self.append(p1=..., p2=..., beta=..., dur=..., acts=...)`.

### v2 algorithm structure (line refs are v2's `create_edgelist`)

```
1. Compute eligible females and males (active ∧ underpartnered ∧ concurrency-ok)
2. Filter participants by age- and sex-specific layer_probs
3. Bin participants by age (np.digitize on layer_probs[0])
4. Shuffle clusters, then for each cluster:
       For each female age bin (shuffled order):
           Compute weighted male preferences: m_probs * mixing[:, age_bin+1] * cluster_mixing
           Sample males with replacement=False using the weights
           Append to (f, m), mark males as taken
5. Sample partnership durations and per-pair acts; self.append(...)
```

- [ ] **Step 1: Add behavioral tests**

Append to `tests/test_network.py`:

```python
# --- add_pairs behavior tests (Task 6) ----------------------------------------

import sciris as sc

import hpvsim


def _layered_sim(layers=('m', 'c'), n_agents=2000, n_steps=4):
    """Build a Sim with two SexualNetwork instances configured from Nigeria data."""
    country = hpvsim.data.load_country('nigeria')
    networks = [SexualNetwork(layer=k, pars=country['network_pars'][k])
                for k in layers]
    sim = ss.Sim(
        networks=networks, n_agents=n_agents, diseases=None,
        dur=ss.years(n_steps * 0.5), dt=ss.years(0.5),
        rand_seed=0, verbose=0,
    )
    return sim


def test_pairs_form_after_a_few_steps():
    sim = _layered_sim()
    sim.run()
    for net in sim.networks():
        if isinstance(net, SexualNetwork):
            assert len(net) > 0, f'layer {net.layer} formed no pairs'


def test_pairs_dissolve_via_stock_end_pairs():
    sim = _layered_sim(layers=('c',), n_steps=20)
    sim.run()
    net = sim.networks()[0]
    assert (net.edges.dur > 0).all() or len(net) == 0


def test_pair_endpoints_are_male_female():
    sim = _layered_sim()
    sim.run()
    people = sim.people
    for net in sim.networks():
        if not isinstance(net, SexualNetwork):
            continue
        if len(net) == 0:
            continue
        f_at_p1 = people.female[net.edges.p1]
        f_at_p2 = people.female[net.edges.p2]
        assert (f_at_p1 ^ f_at_p2).all(), \
            f'layer {net.layer} has same-sex pairs'


def test_cross_layer_concurrency_filter():
    """With cross_layer=0, no agent should appear in both m and c."""
    country = hpvsim.data.load_country('nigeria')
    nets = []
    for k in ('m', 'c'):
        pars = sc.dcp(country['network_pars'][k])
        pars['cross_layer'] = (0.0, 0.0)
        nets.append(SexualNetwork(layer=k, pars=pars))
    sim = ss.Sim(networks=nets, n_agents=2000, diseases=None,
                 dur=ss.years(2), dt=ss.years(0.5), rand_seed=0, verbose=0)
    sim.run()
    m_net, c_net = nets
    if len(m_net) == 0 or len(c_net) == 0:
        return
    m_members = set(m_net.members.tolist())
    c_members = set(c_net.members.tolist())
    overlap = m_members & c_members
    assert len(overlap) / len(m_members | c_members) < 0.01, \
        f'cross_layer=0 violated: {len(overlap)} agents in both layers'


def test_age_mixing_assortativity():
    """Sampled pairs reproduce expected age-mixing concentration on/near the diagonal."""
    sim = _layered_sim(layers=('m',), n_agents=5000, n_steps=10)
    sim.run()
    net = sim.networks()[0]
    if len(net) < 100:
        return
    people = sim.people
    f_at_p1 = people.female[net.edges.p1]
    f_uids = np.where(f_at_p1, net.edges.p1, net.edges.p2)
    m_uids = np.where(f_at_p1, net.edges.p2, net.edges.p1)
    f_ages = people.age[f_uids]
    m_ages = people.age[m_uids]
    bins = np.arange(0, 81, 5)
    f_bins = np.digitize(f_ages, bins) - 1
    m_bins = np.digitize(m_ages, bins) - 1
    n_bins = len(bins) - 1
    obs = np.zeros((n_bins, n_bins))
    for fb, mb in zip(f_bins, m_bins):
        if 0 <= fb < n_bins and 0 <= mb < n_bins:
            obs[fb, mb] += 1
    if obs.sum() == 0:
        return
    obs_d = obs / obs.sum()
    diag = np.trace(obs_d) + np.trace(obs_d, offset=1) + np.trace(obs_d, offset=-1)
    far_off = obs_d[0, -1] + obs_d[-1, 0]
    assert diag > 5 * far_off, \
        f'mixing not assortative: diag {diag:.3f} not >> far {far_off:.3f}'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_network.py -v
```

Expected: the new behavioral tests fail (no pairs form because `add_pairs` is still a no-op). Task-5 scaffold tests still pass.

- [ ] **Step 3: Port `add_pairs`**

Open `hpvsim/_v2_legacy/population.py` and read `create_edgelist` end-to-end. Replace the `add_pairs` no-op in `hpvsim/network.py` with the port. Also add the helper `_participation_filter` as a module-level function.

```python
    def add_pairs(self):
        """Form new pairs in this layer. Port of v2 create_edgelist
        (hpvsim/_v2_legacy/population.py:281-379) with four adaptations:
          - lno → self.layer (cosmetic only)
          - current_partners[other_layers] → self._n_partners_elsewhere() > 0
          - current_partners[lno, :] count → count of own edges per uid
          - current_partners updates → self.append(...)
        """
        sim = self.sim
        people = sim.people
        pars = self.pars

        n_agents = len(people)
        is_female = people.female
        active = self.active(people)

        if not hasattr(self, '_partners_target'):
            self._partners_target = pars.partners.rvs(np.arange(n_agents))

        n_in_self = np.zeros(n_agents, dtype=int)
        if len(self):
            np.add.at(n_in_self, self.edges.p1, 1)
            np.add.at(n_in_self, self.edges.p2, 1)
        underpartnered = n_in_self < self._partners_target

        n_elsewhere = self._n_partners_elsewhere()
        has_other = n_elsewhere > 0

        f_cross_prop, m_cross_prop = pars.cross_layer
        f_with_other = (has_other & is_female).uids
        m_with_other = (has_other & ~is_female).uids
        rng = np.random.default_rng(int(self.t.ti))
        f_cross_eligible = f_with_other[rng.random(len(f_with_other)) < f_cross_prop]
        m_cross_eligible = m_with_other[rng.random(len(m_with_other)) < m_cross_prop]

        cross_layer_bools = np.zeros(n_agents, dtype=bool)
        cross_layer_bools[f_cross_eligible] = True
        cross_layer_bools[m_cross_eligible] = True

        f_eligible = active & is_female & underpartnered & (~has_other | cross_layer_bools)
        m_eligible = active & ~is_female & underpartnered & (~has_other | cross_layer_bools)

        layer_probs = pars.layer_probs
        bins = layer_probs[0, :]

        m_eligible_inds = np.where(m_eligible)[0]
        m_participants = _participation_filter(
            m_eligible_inds, people.age, layer_probs[2, :], bins, rng
        )

        cluster = getattr(people, 'cluster', None)
        if cluster is None:
            cluster = np.zeros(n_agents, dtype=int)
        cluster_range = np.unique(cluster)
        rng.shuffle(cluster_range)

        add_mixing = getattr(pars, 'add_mixing', np.ones((len(cluster_range), len(cluster_range))))

        m_probs = np.ones(n_agents)
        f_out = []
        m_out = []

        if len(m_participants):
            age_bins_m = np.digitize(people.age[m_participants], bins=bins) - 1
        else:
            age_bins_m = np.array([], dtype=int)

        for cl in cluster_range:
            f_eligible_in_cluster = np.where(f_eligible & (cluster == cl))[0]
            f_cl = _participation_filter(
                f_eligible_in_cluster, people.age, layer_probs[1, :], bins, rng
            )
            if len(f_cl) == 0:
                continue
            age_bins_f = np.digitize(people.age[f_cl], bins=bins) - 1
            bin_range_f, males_needed = np.unique(age_bins_f, return_counts=True)
            bin_order = np.arange(len(bin_range_f))
            rng.shuffle(bin_order)
            for ab, nm in zip(bin_range_f[bin_order], males_needed[bin_order]):
                if ab + 1 >= pars.mixing.shape[1]:
                    continue
                male_dist = pars.mixing[:, ab + 1]
                this_weighting = (
                    m_probs[m_participants] * male_dist[age_bins_m]
                    * add_mixing[cl, cluster[m_participants]]
                )
                if this_weighting.sum() == 0:
                    continue
                males_nonzero = np.where(this_weighting > 0)[0]
                w = this_weighting[males_nonzero]
                f_inds = f_cl[age_bins_f == ab]
                if nm > len(males_nonzero):
                    f_selected = rng.choice(f_inds, size=len(males_nonzero), replace=False)
                    nm = len(f_selected)
                else:
                    f_selected = f_inds
                m_selected = m_participants[
                    males_nonzero[rng.choice(
                        len(males_nonzero), size=nm, replace=False, p=w / w.sum()
                    )]
                ]
                m_probs[m_selected] = 0
                f_out.append(f_selected)
                m_out.append(m_selected)

        if not f_out:
            return
        f_arr = np.concatenate(f_out)
        m_arr = np.concatenate(m_out)

        n_new = len(f_arr)
        if isinstance(pars.duration, ss.Dist):
            dur = pars.duration.rvs(np.arange(n_new))
        else:
            dur = np.full(n_new, float(pars.duration))
        if isinstance(pars.acts, ss.Dist):
            acts = pars.acts.rvs(np.arange(n_new))
        else:
            acts = np.full(n_new, float(pars.acts))

        beta = np.ones(n_new)
        self.append(
            p1=ss.uids(f_arr.astype(int)),
            p2=ss.uids(m_arr.astype(int)),
            beta=beta,
            dur=dur,
            acts=acts,
        )


def _participation_filter(eligible_inds, age, layer_probs_per_age, bins, rng):
    """Filter eligible_inds by age-bin-specific participation rates.
    Replicates v2 hpvsim.utils.participation_filter behavior."""
    if len(eligible_inds) == 0:
        return np.array([], dtype=int)
    age_bins = np.digitize(age[eligible_inds], bins=bins) - 1
    rates = np.where(
        (age_bins >= 0) & (age_bins < len(layer_probs_per_age)),
        layer_probs_per_age[np.clip(age_bins, 0, len(layer_probs_per_age) - 1)],
        0.0,
    )
    keep = rng.random(len(eligible_inds)) < rates
    return eligible_inds[keep]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_network.py -v
```

Expected: all behavioral tests pass.

**Common failure modes:**
- `test_pairs_form_after_a_few_steps`: zero pairs forming → check `pars.layer_probs` and `pars.mixing` shapes via `print(country['network_pars']['m']['mixing'].shape)`.
- `test_pair_endpoints_are_male_female`: same-sex pairs → verify f/m assignment matches v2 (line 376 of v2 create_edgelist always puts females in p1).
- `test_cross_layer_concurrency_filter`: too much overlap → verify `pars.cross_layer` tuple unpacking matches v2's `f_cross_layer`/`m_cross_layer` semantics.
- `test_age_mixing_assortativity`: weak diagonal → verify `pars.mixing[:, ab + 1]` axis indexing matches v2 line 357.

Iterate until green.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/network.py tests/test_network.py
git commit -m "$(cat <<'EOF'
Port v2 create_edgelist into hpvsim.network.SexualNetwork.add_pairs

Lift-and-shift of v2 hpvsim/_v2_legacy/population.py:create_edgelist with
four adaptations: lno → self.layer, current_partners[other] →
_n_partners_elsewhere(), current_partners[lno] count → own edge count,
current_partners updates → self.append. Behavioral tests verify pair
formation, M/F endpoints, cross-layer concurrency at cross_layer=0, and
age-mixing assortativity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `hpv.Sim(ss.Sim)` convenience wrapper

**Files:**
- Create: `hpvsim/sim.py` (replaces the file moved to `_v2_legacy/sim.py` in Task 2)
- Create: `tests/test_sim.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sim.py`:

```python
"""Integration tests for hpvsim.sim.Sim."""

import starsim as ss

import hpvsim
from hpvsim.sim import Sim
from hpvsim.hpv import HPV
from hpvsim.network import SexualNetwork


def test_sim_constructs_with_defaults():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.5)
    assert sim is not None


def test_sim_init_runs():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.5)
    sim.init()
    assert len(sim.people) == 500


def test_sim_has_three_sexual_network_layers():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.5)
    sim.init()
    sx_layers = [n for n in sim.networks() if isinstance(n, SexualNetwork)]
    assert len(sx_layers) == 3
    assert {n.layer for n in sx_layers} == {'m', 'c'}


def test_sim_has_one_hpv_disease():
    sim = Sim(location='nigeria', genotype='hpv16', n_agents=500,
              start=2000, stop=2002, dt=0.5)
    sim.init()
    hpv_diseases = [d for d in sim.diseases() if isinstance(d, HPV)]
    assert len(hpv_diseases) == 1
    assert hpv_diseases[0].genotype == 'hpv16'


def test_sim_runs_short_window():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2003, dt=0.5)
    sim.run()
    assert sim.results.hpv16.n_infected[-1] >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sim.py -v
```

Expected: failures with `ModuleNotFoundError: No module named 'hpvsim.sim'`.

- [ ] **Step 3: Implement `Sim`**

Create `hpvsim/sim.py`:

```python
"""HPVsim convenience Sim wrapper.

Provides a v2-compatible API: ``hpv.Sim(location='nigeria', genotype='hpv16')``.
Instantiates the four-component default stack (HPV disease module, three
SexualNetwork layers, ss.Pregnancy + ss.Deaths demographics, ss.People with
location-specific age pyramid) and forwards to ss.Sim. All defaults are
overridable via kwargs (passing ``diseases=`` / ``networks=`` /
``demographics=`` short-circuits the convenience wiring).

M01: single-genotype only. M03 changes the signature to ``genotypes=[...]``.
"""

import starsim as ss

from .data.country import load_country
from .hpv import HPV
from .network import SexualNetwork


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location='nigeria', genotype='hpv16',
                 n_agents=10_000, start=1990, stop=2060, dt=0.5,
                 pars=None, **kwargs):
        country = load_country(location)
        people = kwargs.pop('people', None) or ss.People(
            n_agents, age_data=country['age_data']
        )
        diseases = kwargs.pop('diseases', None) or [HPV(genotype=genotype)]
        networks = kwargs.pop('networks', None) or [
            SexualNetwork(layer=k, pars=country['network_pars'][k])
            for k in ('m', 'c')
        ]
        demographics = kwargs.pop('demographics', None) or [
            ss.Pregnancy(fertility_rate=country['fertility']),
            ss.Deaths(death_rate=country['death_rate']),
        ]
        super().__init__(
            start=ss.years(start),
            stop=ss.years(stop),
            dt=ss.years(dt),
            people=people,
            diseases=diseases,
            networks=networks,
            demographics=demographics,
            pars=pars,
            **kwargs,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sim.py -v
```

Expected: all five tests pass.

**If `validate_beta` errors about network/beta key mismatch**, the disease's beta needs to be a per-network dict. Edit `hpvsim/hpv.py` to change the default to:

```python
beta=ss.peryear({'m': 0.5, 'c': 0.3, 'o': 0.1})
```

and update the test in `tests/test_hpv.py::_minimal_sim` if needed (the random net case may need a different beta shape).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/sim.py tests/test_sim.py
git commit -m "$(cat <<'EOF'
Add hpvsim.sim.Sim(ss.Sim) convenience wrapper

In-place replacement for the (now quarantined) v2 1395-line Sim. Composes
HPV disease module, two SexualNetwork layers (m/c), stock ss.Pregnancy
+ ss.Deaths demographics, and ss.People with location-specific age pyramid
via hpvsim.data.load_country. All defaults overridable via kwargs (passing
diseases/networks/demographics short-circuits the convenience wiring).
M01 supports a single genotype; M03 will switch to genotypes=[...].

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire final `hpvsim/__init__.py` exports

**Files:**
- Modify: `hpvsim/__init__.py`

- [ ] **Step 1: Update package init**

Replace `hpvsim/__init__.py` content with:

```python
"""hpvsim — HPV simulation tools (Starsim-based).

Public API:
    import hpvsim as hpv
    sim = hpv.Sim(location='nigeria', genotype='hpv16')
    sim.run()

Modules:
    hpv.HPV              — single-genotype HPV disease module
    hpv.SexualNetwork    — heterosexual partnership network
    hpv.Sim              — convenience Sim wrapper
    hpv.data             — country-data adapter (load_country)

v2 modules awaiting port live in hpvsim/_v2_legacy/. Active code MUST NOT
import from there.
"""

import sciris as sc

from .version import __version__, __versiondate__, __license__
from .settings import options
from .defaults import datadir, default_int, default_float, get_default_plots
from . import data
from . import parameters
from . import misc
from . import utils

from .hpv import HPV
from .network import SexualNetwork
from .sim import Sim

__all__ = [
    'HPV', 'SexualNetwork', 'Sim', 'data',
    'options', 'datadir', '__version__',
]

del sc
```

- [ ] **Step 2: Verify exports**

```bash
python -c "import hpvsim; print(hpvsim.HPV, hpvsim.SexualNetwork, hpvsim.Sim, hpvsim.data.load_country)"
```

Expected: prints four class/function repr strings, no errors.

- [ ] **Step 3: Run the full active test suite**

```bash
pytest tests/test_hpv.py tests/test_network.py tests/test_sim.py tests/test_data.py tests/test_regression.py -v
```

Expected: all tests pass; `test_anchor_runs` is skipped per the Task-2 mark.

- [ ] **Step 4: Commit**

```bash
git add hpvsim/__init__.py
git commit -m "$(cat <<'EOF'
Wire hpvsim public API exports — HPV, SexualNetwork, Sim, data

Final M01 surface: HPV, SexualNetwork, Sim, data are now importable
directly from hpvsim. Stable utility imports (settings.options,
defaults.datadir, parameters, misc, utils) remain.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: M01 anchor harness

**Files:**
- Create: `tests/regression/anchor_hpv16.py`

- [ ] **Step 1: Create the anchor module**

```python
"""M01 anchor scenario for the v2 -> v3 migration regression harness.

Single-genotype (HPV16) HPV sim, Nigeria, fixed seed, no interventions, no
analyzers. Tooling under tests/regression/ (compare.py) imports
run_and_summarize() from here.

Run as a script to print the summary:
    python tests/regression/anchor_hpv16.py
"""

import sciris as sc

import hpvsim as hpv

# Pinned anchor pars. Do not change without coordinating with regression baselines.
PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotype='hpv16',
    start=1990,
    stop=2060,
    dt=0.5,
    rand_seed=0,
    verbose=0,
)


def make_sim():
    """Build (but do not run) the M1 anchor sim."""
    return hpv.Sim(**sc.dcp(PARS))


def run_and_summarize():
    """Run the M1 anchor sim and return (short_summary_dict, total_population_float)."""
    sim = make_sim()
    sim.run()
    res = sim.results.hpv16

    n_inf = float(res.cum_infections[-1]) if 'cum_infections' in res else float(res.n_infected.sum())
    mean_prev_pct = 100 * float(res.prevalence.mean())

    hpv_mod = sim.diseases.hpv16
    infected_agents = (~hpv_mod.ti_infected.isnan).uids
    if len(infected_agents):
        ages_at_inf = sim.people.age[infected_agents] - (
            float(sim.t.now) - hpv_mod.ti_infected[infected_agents] * float(sim.t.dt)
        )
        mean_age_inf = float(ages_at_inf.mean())
    else:
        mean_age_inf = 0.0

    short = {
        'total HPV infections': n_inf,
        'mean HPV prevalence (%)': mean_prev_pct,
        'mean age of infection (years)': mean_age_inf,
    }
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop


if __name__ == '__main__':
    short, total_pop = run_and_summarize()
    print('Short summary:')
    for k, v in short.items():
        print(f'  {k:<40} {v:>12.4g}')
    print(f'  {"total population":<40} {total_pop:>12.4g}')
```

- [ ] **Step 2: Run as a script**

```bash
python tests/regression/anchor_hpv16.py
```

Expected: prints a 4-row summary with finite values. Runtime ~30-60s.

If it errors on a result-key naming mismatch (e.g., `sim.results.hpv16` vs. `sim.results['hpv16']`), inspect `sim.results.keys()` and adjust accordingly.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/anchor_hpv16.py
git commit -m "$(cat <<'EOF'
Add M01 1-genotype HPV16 anchor harness

Pinned single-genotype Nigeria scenario for the v2 -> v3 regression
harness. Mirrors M00's anchor.py structure (PARS dict, make_sim,
run_and_summarize) with genotype='hpv16' and an M01-appropriate summary
set (total HPV infections, mean HPV prevalence %, mean age of infection).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: M01 anchor smoke test

**Files:**
- Modify: `tests/test_regression.py`

- [ ] **Step 1: Append smoke test**

Add to the end of `tests/test_regression.py`:

```python
# --- M01 anchor smoke (added in Task 10) --------------------------------------

from regression.anchor_hpv16 import run_and_summarize as run_anchor_hpv16  # noqa: E402


def test_anchor_hpv16_runs():
    short, total_pop = run_anchor_hpv16()
    expected_keys = {
        'total HPV infections',
        'mean HPV prevalence (%)',
        'mean age of infection (years)',
    }
    missing = expected_keys - set(short.keys())
    assert not missing, f'short_summary missing keys: {missing}'
    assert total_pop > 0, f'total_pop should be positive, got {total_pop}'
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_regression.py::test_anchor_hpv16_runs -v
```

Expected: passes (~30-60s runtime).

- [ ] **Step 3: Commit**

```bash
git add tests/test_regression.py
git commit -m "$(cat <<'EOF'
Add test_anchor_hpv16_runs smoke test

Tier-2 smoke (no baseline file required). Asserts the M01 anchor harness
produces a non-empty summary with expected keys and a positive total
population.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Document M01 1-genotype baseline procedure

**Files:**
- Modify: `tests/regression/README.md`

- [ ] **Step 1: Read the existing README**

```bash
cat tests/regression/README.md
```

- [ ] **Step 2: Append M01 section**

Append to `tests/regression/README.md`:

```markdown

## M01 1-genotype baseline (`anchor_hpv16.json`)

The M01 anchor (`anchor_hpv16.py`) compares against a v2 hpvsim run configured
with `genotypes=['hpv16']` and otherwise identical pars to the M00 4-genotype
anchor. This baseline is gitignored at:

```
tests/regression_baselines/anchor_hpv16.json
```

### Generating the baseline

In a v2-only environment (e.g., a separate venv with v2.3.x hpvsim from PyPI,
or a checkout of the `rc2.3` branch of this repo before the v3 migration):

```python
import json
import sciris as sc
import hpvsim as hpv2  # v2 package

PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=['hpv16'],
    start=1990,
    end=2060,
    dt=0.5,
    burnin=20,
    rand_seed=0,
    verbose=0,
)

sim = hpv2.Sim(sc.dcp(PARS))
sim.run()

short = {
    'total HPV infections': float(sim.results['total_infections'].sum()),
    'mean HPV prevalence (%)': 100 * float(sim.results['hpv_prevalence'].mean()),
    'mean age of infection (years)': float(sim.results['mean_age_infection'].mean()),
}
total_pop = float(sim.results['n_alive'][-1])

out = dict(summary={**short, 'total population': total_pop}, pars=PARS)
with open('tests/regression_baselines/anchor_hpv16.json', 'w') as f:
    json.dump(out, f, indent=2)
```

The summary keys **must match v3's `run_and_summarize()` output keys exactly**
— see `tests/regression/anchor_hpv16.py`. If v2's result names differ, compute
the equivalent quantities from v2's underlying time series and store them
under the v3 key names.

### Partnership-equivalence baseline (`partnership_v2.json`)

Supports the M01 acceptance gate (`tests/test_partnership_equivalence.py`).

```python
import json
import numpy as np
import sciris as sc
import hpvsim as hpv2

PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=['hpv16'],
    start=1990,
    end=2015,
    dt=0.5,
    burnin=20,
    rand_seed=0,
    verbose=0,
)

sim = hpv2.Sim(sc.dcp(PARS))
sim.run()

# Capture per-layer mixing matrix (16x16 for 5y bins, 0-80, female × male),
# concurrency histogram, and partnership-duration samples. v2 stores these
# on sim.people in layer-keyed structures; consult v2 internals for exact
# attribute names.

out = {}
for layer in ('m', 'c'):
    out[layer] = {
        'mixing_matrix': ...,      # 2d list, 16x16, density-normalized
        'concurrency_hist': ...,   # 1d list, indexed by n_concurrent_partners
        'duration_samples': ...,   # 1d list, completed-edge durations in years
    }
with open('tests/regression_baselines/partnership_v2.json', 'w') as f:
    json.dump(out, f)
```

The M01 partnership-equivalence test reads this JSON and runs KS-tests +
bin-wise diff against equivalent quantities produced by v3.
```

- [ ] **Step 3: Commit**

```bash
git add tests/regression/README.md
git commit -m "$(cat <<'EOF'
Document M01 1-gt baseline generation procedure

Adds an M01 section to tests/regression/README.md with the v2 hpvsim
recipe for generating anchor_hpv16.json (anchor drift baseline) and
partnership_v2.json (partnership-equivalence acceptance gate). Both
baselines remain local-only / gitignored, following the M00 convention.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Tier-3 anchor drift test (gated on baseline file)

**Files:**
- Modify: `tests/test_regression.py`

- [ ] **Step 1: Append gated drift test**

Add to `tests/test_regression.py`:

```python
# --- M01 anchor drift (added in Task 12) --------------------------------------

import json
from pathlib import Path


_BASELINE_HPV16 = Path(__file__).resolve().parent / 'regression_baselines' / 'anchor_hpv16.json'


@pytest.mark.skipif(not _BASELINE_HPV16.exists(),
                    reason='anchor_hpv16.json baseline not present (gitignored; '
                           'see tests/regression/README.md for generation procedure)')
def test_anchor_hpv16_drift():
    """Gated drift test against v2 1-genotype baseline. Informational threshold
    (10% relative drift); does NOT fail the build."""
    short, total_pop = run_anchor_hpv16()
    current_summary = {**{k: float(v) for k, v in short.items()},
                       'total population': total_pop}

    with open(_BASELINE_HPV16) as f:
        baseline = json.load(f)
    rows = compute_drift(baseline['summary'], current_summary, threshold=0.10)
    n_over = sum(1 for r in rows if r['over_threshold'])
    print(f'\nM01 anchor drift: {n_over}/{len(rows)} keys exceed 10% relative drift')
    for r in rows:
        rel = f"{r['rel_diff']*100:+.2f}%" if r['rel_diff'] is not None else 'n/a'
        flag = 'YES' if r['over_threshold'] else ''
        print(f'  {r["key"]:<40} {r["baseline"]:>12.4g} {r["current"]:>12.4g} {rel:>10} {flag}')
    assert True
```

(Make sure `pytest` is imported at the top of the file. If `pytest` isn't already imported, the existing `_pytest_for_skip` alias from Task 2 can be reused, or add `import pytest` at the top.)

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_regression.py::test_anchor_hpv16_drift -v
```

Expected: skipped with the message about the baseline being absent.

- [ ] **Step 3: Commit**

```bash
git add tests/test_regression.py
git commit -m "$(cat <<'EOF'
Add gated M01 anchor drift test against v2 1-gt baseline

Tier-3 informational drift test (10% relative threshold, non-failing).
Skipped when tests/regression_baselines/anchor_hpv16.json is absent.
Same convention as M00's drift test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Partnership-equivalence acceptance test

**Files:**
- Create: `tests/test_partnership_equivalence.py`

This is the **M01 acceptance gate**.

- [ ] **Step 1: Create the test file**

Create `tests/test_partnership_equivalence.py`:

```python
"""M01 acceptance gate: partnership patterns vs. v2.x.

Per-layer comparison of:
  - age-mixing matrix (5y bins, 0-80, female × male)
  - concurrency distribution (n_concurrent_partners histogram)
  - partnership duration distribution

Against v2 baselines stored in tests/regression_baselines/partnership_v2.json
(gitignored; generation procedure documented in tests/regression/README.md).

Pass criteria:
  - mixing-matrix bin-wise relative diff < 15% (per non-zero bin)
  - concurrency: KS-test p > 0.01
  - duration: KS-test p > 0.01
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import hpvsim as hpv
from hpvsim.network import SexualNetwork


_BASELINE_PARTNERSHIP = Path(__file__).resolve().parent / 'regression_baselines' / 'partnership_v2.json'

PARS = dict(
    location='nigeria',
    genotype='hpv16',
    n_agents=10_000,
    start=1990,
    stop=2015,
    dt=0.5,
    rand_seed=0,
    verbose=0,
)


def _capture_partnership_stats(sim):
    """Return per-layer dict with mixing_matrix, concurrency_hist, duration_samples."""
    people = sim.people
    out = {}
    for net in sim.networks():
        if not isinstance(net, SexualNetwork):
            continue
        if len(net) == 0:
            out[net.layer] = dict(
                mixing_matrix=np.zeros((16, 16)).tolist(),
                concurrency_hist=[len(people), 0, 0, 0, 0],
                duration_samples=[],
            )
            continue
        f_at_p1 = people.female[net.edges.p1]
        f_uids = np.where(f_at_p1, net.edges.p1, net.edges.p2)
        m_uids = np.where(f_at_p1, net.edges.p2, net.edges.p1)
        bins = np.arange(0, 81, 5)
        f_bins = np.digitize(people.age[f_uids], bins) - 1
        m_bins = np.digitize(people.age[m_uids], bins) - 1
        n_bins = len(bins) - 1
        mat = np.zeros((n_bins, n_bins))
        for fb, mb in zip(f_bins, m_bins):
            if 0 <= fb < n_bins and 0 <= mb < n_bins:
                mat[fb, mb] += 1
        if mat.sum() > 0:
            mat = mat / mat.sum()

        n_per_agent = np.zeros(len(people), dtype=int)
        np.add.at(n_per_agent, net.edges.p1, 1)
        np.add.at(n_per_agent, net.edges.p2, 1)
        max_k = max(5, n_per_agent.max() + 1)
        conc_hist = np.bincount(n_per_agent, minlength=max_k).tolist()

        durations = net.edges.dur.tolist() if hasattr(net.edges, 'dur') else []

        out[net.layer] = dict(
            mixing_matrix=mat.tolist(),
            concurrency_hist=conc_hist,
            duration_samples=durations,
        )
    return out


@pytest.fixture(scope='module')
def v3_stats():
    sim = hpv.Sim(**PARS)
    sim.run()
    return _capture_partnership_stats(sim)


@pytest.fixture(scope='module')
def v2_stats():
    if not _BASELINE_PARTNERSHIP.exists():
        pytest.skip(
            'partnership_v2.json baseline not present (gitignored; '
            'see tests/regression/README.md for generation procedure)'
        )
    with open(_BASELINE_PARTNERSHIP) as f:
        return json.load(f)


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_age_mixing_matrix(v3_stats, v2_stats, layer):
    v3 = np.array(v3_stats[layer]['mixing_matrix'])
    v2 = np.array(v2_stats[layer]['mixing_matrix'])
    assert v3.shape == v2.shape
    nonzero = v2 > 0.001
    if nonzero.sum() == 0:
        return
    rel_diff = np.abs(v3[nonzero] - v2[nonzero]) / v2[nonzero]
    max_diff = rel_diff.max()
    assert max_diff < 0.15, \
        f'layer {layer} mixing matrix max bin-wise rel diff {max_diff:.3f} >= 0.15'


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_concurrency_distribution(v3_stats, v2_stats, layer):
    v3_hist = np.array(v3_stats[layer]['concurrency_hist'])
    v2_hist = np.array(v2_stats[layer]['concurrency_hist'])
    v3_samples = np.repeat(np.arange(len(v3_hist)), v3_hist)
    v2_samples = np.repeat(np.arange(len(v2_hist)), v2_hist)
    if len(v3_samples) == 0 or len(v2_samples) == 0:
        return
    ks_stat, p_value = stats.ks_2samp(v3_samples, v2_samples)
    assert p_value > 0.01, \
        f'layer {layer} concurrency KS p={p_value:.4f} <= 0.01'


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_duration_distribution(v3_stats, v2_stats, layer):
    v3_dur = np.array(v3_stats[layer]['duration_samples'])
    v2_dur = np.array(v2_stats[layer]['duration_samples'])
    if len(v3_dur) < 30 or len(v2_dur) < 30:
        pytest.skip(f'layer {layer}: too few duration samples for KS-test')
    ks_stat, p_value = stats.ks_2samp(v3_dur, v2_dur)
    assert p_value > 0.01, \
        f'layer {layer} duration KS p={p_value:.4f} <= 0.01'
```

- [ ] **Step 2: Run the test (will skip if baseline absent)**

```bash
pytest tests/test_partnership_equivalence.py -v
```

Expected: 9 tests skipped (3 layers × 3 metrics) when the baseline file is absent.

- [ ] **Step 3: Commit**

```bash
git add tests/test_partnership_equivalence.py
git commit -m "$(cat <<'EOF'
Add M01 partnership-equivalence acceptance test

Per-layer comparison of mixing matrix (15% bin-wise rel diff), concurrency
distribution (KS p > 0.01), and partnership duration distribution
(KS p > 0.01) against v2 fixtures stored in
tests/regression_baselines/partnership_v2.json (gitignored). M01
acceptance gate; gating on the baseline file's presence keeps CI green
when absent and forces a real comparison locally once generated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: M01 demo script

**Files:**
- Create: `tests/regression/demo_anchor_hpv16.py`

- [ ] **Step 1: Create the demo script**

```python
"""M01 demo: run the 1-genotype HPV16 anchor and plot aggregate prevalence.

Visible artifact of the M01 milestone (acceptance gate #4 per the M01
design spec).

Run with:
    python tests/regression/demo_anchor_hpv16.py
"""

import matplotlib.pyplot as plt

import hpvsim as hpv
from regression.anchor_hpv16 import PARS


def main():
    sim = hpv.Sim(**PARS)
    sim.run()

    yearvec = sim.t.yearvec
    prev = sim.results.hpv16.prevalence

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(yearvec, prev * 100, label='HPV16 prevalence (M01 anchor)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('M01 demo: HPV16 prevalence trajectory, Nigeria 1990-2060')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the demo**

```bash
python tests/regression/demo_anchor_hpv16.py
```

Expected: matplotlib window opens with a smooth prevalence curve over 1990-2060. Close to exit.

If running headless: `MPLBACKEND=Agg python tests/regression/demo_anchor_hpv16.py` — script constructs the figure and exits without display; verify no exceptions.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/demo_anchor_hpv16.py
git commit -m "$(cat <<'EOF'
Add M01 demo script — plot HPV16 prevalence trajectory

Runs the pinned anchor and renders aggregate HPV16 prevalence over
1990-2060 (Nigeria, single genotype). Acceptance gate #4 per the M01
design spec. Imports PARS from anchor_hpv16.py to ensure demo and anchor
remain identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Final verification + push + open M01 PR

**Files:** none (housekeeping only).

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected:
- All M01 tests pass.
- `test_anchor_runs` (M00 4-gt) skips with reason about M03.
- Gated tests (`test_anchor_hpv16_drift`, partnership-equivalence) skip cleanly.
- Quarantined tests in `tests/_legacy/` are not collected (per the conftest ignore added in Task 2).

- [ ] **Step 2: Smoke-test the public API end-to-end**

```bash
python -c "
import hpvsim as hpv
sim = hpv.Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.5)
sim.run()
print('OK', float(sim.results.hpv16.prevalence[-1]))
"
```

Expected: `OK <some-float>`.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin m01-basic-transmission-sim
```

- [ ] **Step 4: Open the M01 PR**

```bash
gh pr create --base v3.0-dev \
  --title "M01: Basic transmission sim — single-genotype HPV16 + ported network" \
  --body "$(cat <<'EOF'
## Summary

In-place replacement of v2 `hpvsim` with the minimum runnable HPV sim on Starsim. Implements M01 of the v2 → v3 migration:

- `hpvsim.HPV(ss.Infection)` — single-genotype HPV16 transmission-only disease module (SIS clearance; CIN/cancer ship in M02).
- `hpvsim.SexualNetwork(ss.SexualNetwork)` — lift-and-shift of v2's two-layer (m/c) heterosexual network. One class, two instances; cross-layer concurrency resolved at `add_pairs` time via `isinstance`-filtered iteration of sibling networks.
- `hpvsim.Sim(ss.Sim)` — convenience wrapper providing the v2-compatible `hpv.Sim(location=..., genotype=...)` API.
- `hpvsim.data.load_country()` — adapter wrapping v2's data loaders into Starsim-shaped DataFrames.

v2 modules untouched by M01 are quarantined to `hpvsim/_v2_legacy/`; v2 tests that exercise removed APIs go to `tests/_legacy/`. Quarantines are never imported by active code; M10 deletes them wholesale.

## Test plan

- [ ] Unit / functional tests pass: `pytest tests/test_hpv.py tests/test_network.py tests/test_sim.py tests/test_data.py -v`
- [ ] M01 anchor smoke passes: `pytest tests/test_regression.py::test_anchor_hpv16_runs -v` (~30-60s)
- [ ] Full test suite passes (M00 4-gt smoke skipped until M03): `pytest tests/ -v`
- [ ] Demo script renders prevalence trajectory: `python tests/regression/demo_anchor_hpv16.py`
- [ ] (Reviewer) Generate v2 1-gt baselines locally per `tests/regression/README.md` and confirm gated tests pass:
  - [ ] `pytest tests/test_regression.py::test_anchor_hpv16_drift -v`
  - [ ] `pytest tests/test_partnership_equivalence.py -v`

## Decisions log

See `docs/superpowers/specs/2026-04-28-hpvsim-m1-basic-transmission-design.md` §"Decisions log" for the architectural choices settled during the M01 brainstorm: rotasim-style multi-genotype direction, `ss.SexualNetwork` parent class, two instances (m/c, matching v2's actual default network), isinstance-filtered cross-layer concurrency, no `hpv.People` (multi-scale deferred), in-place replacement with v2 quarantine, milestone split (M02 1-gt nat hx, new M03 multi-gt + cross-immunity).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Self-review checklist

Run through after writing the plan; before declaring complete:

1. **Spec coverage:**
   - Architecture diagram → Tasks 1-8 (every component lands; quarantine in Task 2; final wiring in Task 8).
   - Components: HPV → Task 4; SexualNetwork → Tasks 5-6; Sim → Task 7; data adapter → Task 3; package init → Tasks 2 (minimal) + 8 (final).
   - Data flow → Tasks 7-13.
   - Acceptance gates: (1) sim runs end-to-end → Task 10 smoke; (2) partnership equivalence → Task 13; (3) prevalence trajectory matches v2 → Task 12 (gated); (4) demo plots prevalence → Task 14.
   - Testing strategy (Tier 1/2/3) → Tier 1 in Tasks 3-7; Tier 2 in Task 10; Tier 3 in Tasks 12-13.
   - Error handling → covered by validation in Tasks 3-5.
   - Open items pinned → anchor pars in Task 9; tolerance thresholds in Task 13; baseline procedure in Task 11.
   - Milestone housekeeping → Task 0.
   - In-place replacement / quarantine → Task 2.

2. **Placeholder scan:** No "TBD", no "implement later", no "add appropriate error handling". The references to v2's `_v2_legacy/population.py:281-379` in Task 6 are *source pointers* for the port, not placeholders for unwritten plan content.

3. **Type consistency:** `HPV.genotype` (Task 4) is referenced by Tasks 7, 9, 10; `SexualNetwork.layer` (Task 5) by Tasks 6, 7, 13; `_n_partners_elsewhere` (Task 5) used by Task 6; `data.load_country` shape (Task 3) consumed by Task 7. Test imports use the final `import hpvsim` / `from hpvsim.X import Y` form, consistent with the `__init__.py` exports landed in Task 8 (and the minimal init in Task 2 for the early tasks).

---

## Execution choice

Plan complete and saved to `docs/superpowers/plans/2026-04-29-hpvsim-m1-basic-transmission.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, two-stage review (spec compliance + code quality) between tasks, fast iteration in this session.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
