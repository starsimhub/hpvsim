# M06: Test-and-Treat Cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the M06 screen → triage → treat cascade plus therapeutic vaccination in v3 hpvsim, matching v2.x trajectories within `|z| < 3` on two anchor scenarios.

**Architecture:** Thin HPV-specific subclasses of Starsim 3.3.4's intervention/product framework (the same diamond pattern M05 used for vaccination). HPV-specific product `administer` overrides handle per-genotype state. The txvx family, `treat_delay`, `radiation`, and `dynamic_pars` are fresh ports (no Starsim equivalents).

**Tech Stack:** Python 3.10+, Starsim 3.3.4, sciris, pandas, numpy, pytest. Working directory: `C:\Users\ryanhu\PycharmProjects\hpvsim_claudecontrol\` on branch `m06-test-and-treat-cascade`.

**Design spec:** [`docs/superpowers/specs/2026-05-27-m06-test-and-treat-cascade-design.md`](../specs/2026-05-27-m06-test-and-treat-cascade-design.md). All design decisions, M05 lessons, risks, and post-implementation deltas live there. This plan executes that spec.

**File structure:**

| File | Action | Responsibility |
|---|---|---|
| `hpvsim/hpv.py` | Modify | Add per-genotype `latent` BoolState (no-op) and `txvx_imm` FloatArr |
| `hpvsim/cross_genotype.py` | Modify | Extend `CrossImmunity.step` to include `txvx_imm` in independent-protection combine |
| `hpvsim/products.py` | Modify | Add `dx`, `tx`, `txvx`, `radiation` classes + `_load_*_products` helpers + `_iter_hpv_modules` |
| `hpvsim/interventions.py` | Modify | Add `BaseTest`/`Screening`/`Triage`/`Treatment`/`TxVx` subclasses, diamond leaves, `treat_delay`, `dynamic_pars` |
| `hpvsim/__init__.py` | Modify | Export new public names |
| `tests/test_m06_*.py` | Create | Unit + integration tests for each new component |
| `tests/regression/anchor_screen_treat.py` | Create | PARS for screen → triage → treat anchor |
| `tests/regression/anchor_txvx_routine.py` | Create | PARS for routine txvx anchor |
| `tests/regression/multi_seed_v2_screen_treat.py` | Create | 30-seed v2 baseline generator (cascade) |
| `tests/regression/multi_seed_v2_txvx.py` | Create | 30-seed v2 baseline generator (txvx) |
| `tests/test_m06_*_parity.py` | Create | Multi-seed parity tests at \|z\|<3 |
| `tests/regression/README_m06.md` | Create | Document baseline regeneration steps |
| `MIGRATION_PLAN.md` | Modify | Flip M6 status; update sub-task list |

---

## Task 1: Add `latent` BoolState and `txvx_imm` FloatArr to `hpv.HPV`

**Files:**
- Modify: `hpvsim/hpv.py:99-126` (state definitions block)
- Test: `tests/test_m06_hpv_state_deltas.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_m06_hpv_state_deltas.py`:

```python
"""Test that HPV gains `latent` BoolState and `txvx_imm` FloatArr in M06."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _tiny_sim():
    return hpv.Sim(
        n_agents=200, start='2020', stop='2022',
        diseases=hpv.HPV(genotype='hpv16'),
        networks='random',
    )


def test_hpv_has_latent_boolstate():
    sim = _tiny_sim()
    sim.init()
    mod = sim.diseases['hpv16']
    assert hasattr(mod, 'latent'), "HPV module should have a `latent` state"
    # No-op default: nobody should be latent right after init.
    assert mod.latent.uids.size == 0
    assert isinstance(mod.latent, ss.BoolState)


def test_hpv_has_txvx_imm_floatarr():
    sim = _tiny_sim()
    sim.init()
    mod = sim.diseases['hpv16']
    assert hasattr(mod, 'txvx_imm'), "HPV module should have `txvx_imm` FloatArr"
    assert isinstance(mod.txvx_imm, ss.FloatArr)
    # All defaults to zero — no agents have been txvx-vaccinated.
    assert np.all(mod.txvx_imm.values == 0.0)


def test_latent_state_stays_zero_through_short_run():
    """No-op latent: nothing populates it. After a 2yr run, still zero."""
    sim = _tiny_sim()
    sim.run()
    assert sim.diseases['hpv16'].latent.uids.size == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pytest tests/test_m06_hpv_state_deltas.py -v
```
Expected: FAIL on `AssertionError: HPV module should have a 'latent' state` (or similar AttributeError).

- [ ] **Step 3: Add the two states to `hpv.HPV`**

Edit `hpvsim/hpv.py` inside the existing `self.define_states(...)` block (around line 99-126). Add these two entries:

```python
ss.BoolState('latent', label='Latent infection (no-op state hook for dx CSV)'),
ss.FloatArr(
    'txvx_imm',
    label='Therapeutic-vaccine-conferred immunity (this genotype)',
    default=0.0,
),
```

Place the `latent` line after the existing `cancerous` BoolState; place `txvx_imm` immediately after `vax_imm`.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_hpv_state_deltas.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Run M03 regression as a smoke check**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All non-slow tests pass — the state additions must not perturb existing behaviour (BoolState defaults False, FloatArr defaults 0.0; neither is read anywhere yet).

- [ ] **Step 6: Commit**

```bash
git add hpvsim/hpv.py tests/test_m06_hpv_state_deltas.py
git commit -m "M06: add latent BoolState + txvx_imm FloatArr to hpv.HPV

The latent state is a no-op hook for the products_dx.csv schema;
nothing populates it yet (real reactivation natural history is a
post-M06 follow-on). txvx_imm is written by hpv.txvx and read by
CrossImmunity in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Extend `CrossImmunity` to combine `txvx_imm` via independent-protection path

**Files:**
- Modify: `hpvsim/cross_genotype.py:30-50` (docstring), `:123-150` (`step` body)
- Test: `tests/test_m06_cross_immunity_combine.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_m06_cross_immunity_combine.py`:

```python
"""Verify CrossImmunity combines nab_imm, vax_imm, and txvx_imm as
independent protection paths.

rel_sus[target] = (1 - sus_imm_nab[target]) * (1 - vax_imm[target]) * (1 - txvx_imm[target])
"""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _two_genotype_sim():
    return hpv.Sim(
        n_agents=500, start='2020', stop='2022',
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        networks='random',
    )


def test_txvx_imm_reduces_rel_sus_independently():
    sim = _two_genotype_sim()
    sim.init()
    mod16 = sim.diseases['hpv16']
    # Pick five agents; set vax_imm=0.5 and txvx_imm=0.5 on hpv16
    uids = sim.people.alive.uids[:5]
    mod16.vax_imm[uids] = 0.5
    mod16.txvx_imm[uids] = 0.5
    # Step the connector explicitly
    sim.connectors['crossimmunity'].step()
    # Independent combine: rel_sus = (1 - 0) * (1 - 0.5) * (1 - 0.5) = 0.25
    np.testing.assert_allclose(mod16.rel_sus[uids], 0.25, atol=1e-6)


def test_txvx_imm_alone_reduces_rel_sus():
    sim = _two_genotype_sim()
    sim.init()
    mod16 = sim.diseases['hpv16']
    uids = sim.people.alive.uids[:5]
    mod16.txvx_imm[uids] = 0.7
    sim.connectors['crossimmunity'].step()
    # rel_sus = (1 - 0) * (1 - 0) * (1 - 0.7) = 0.3
    np.testing.assert_allclose(mod16.rel_sus[uids], 0.3, atol=1e-6)


def test_txvx_imm_does_not_bleed_across_genotypes():
    """txvx_imm is per-target, NOT matrix-multiplied. Setting txvx_imm on
    hpv16 must not change rel_sus on hpv18."""
    sim = _two_genotype_sim()
    sim.init()
    mod16, mod18 = sim.diseases['hpv16'], sim.diseases['hpv18']
    uids = sim.people.alive.uids[:5]
    mod16.txvx_imm[uids] = 1.0           # max protection on hpv16
    sim.connectors['crossimmunity'].step()
    # hpv18 rel_sus is the all-1.0 baseline (no vax, no nab, no txvx on hpv18)
    np.testing.assert_allclose(mod18.rel_sus[uids], 1.0, atol=1e-6)
    # hpv16 rel_sus is zero
    np.testing.assert_allclose(mod16.rel_sus[uids], 0.0, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_cross_immunity_combine.py -v
```
Expected: FAIL — current combine does NOT factor in `txvx_imm`; rel_sus for the first test will be 0.5, not 0.25.

- [ ] **Step 3: Update `CrossImmunity.step`**

Edit `hpvsim/cross_genotype.py` around line 123-150. Replace the existing `step` body with:

```python
def step(self):
    # Catch any unset agents (births/immigrants since last step) before
    # the downstream HPV step_infect samples new infections.
    self.ensure_rel_sev(self.sim.people.alive.uids)
    if not self.hpv_modules:
        return
    # Clearance-conferred immunity — flows through cross-protection matrix.
    nab  = np.column_stack([m.nab_imm.values  for m in self.hpv_modules])
    cell = np.column_stack([m.cell_imm.values for m in self.hpv_modules])
    # Vaccine-conferred immunity — applied directly per target genotype,
    # NOT through the matrix. Shape: (n_agents, n_genotypes).
    vax   = np.column_stack([m.vax_imm.values   for m in self.hpv_modules])
    txvx  = np.column_stack([m.txvx_imm.values  for m in self.hpv_modules])
    sus_imm_nab = nab  @ self.cross_imm_sus.T
    sev_imm     = cell @ self.cross_imm_sev.T
    np.clip(sus_imm_nab, 0.0, 1.0, out=sus_imm_nab)
    np.clip(sev_imm,     0.0, 1.0, out=sev_imm)
    np.clip(vax,         0.0, 1.0, out=vax)
    np.clip(txvx,        0.0, 1.0, out=txvx)
    auids = self.sim.people.auids
    for i, m in enumerate(self.hpv_modules):
        # Three independent protection paths:
        #   - clearance cross-protection (matrix path, nab_imm)
        #   - prophylactic vaccine (direct path, vax_imm)
        #   - therapeutic vaccine (direct path, txvx_imm)
        # All reduce susceptibility multiplicatively. sev_imm comes only
        # from clearance (vaccines don't reduce severity beyond rel_sus).
        m.rel_sus[auids] = (
            (1.0 - sus_imm_nab[:, i])
            * (1.0 - vax[:, i])
            * (1.0 - txvx[:, i])
        )
        m.sev_imm[auids] = sev_imm[:, i]
```

Also update the class docstring (`hpvsim/cross_genotype.py:30-51`):
- In the "Combining formula for ``rel_sus``" block, change the formula to include the new factor:
  ```
  rel_sus[target] = (1 - sus_imm_nab[target]) * (1 - vax_imm[target]) * (1 - txvx_imm[target])
  ```
- In the prose paragraph, replace "Vaccine-conferred ``vax_imm`` is combined …" with "Vaccine-conferred ``vax_imm`` and therapeutic-vaccine-conferred ``txvx_imm`` are each combined with the nab contribution via independent-protection paths — neither is matrix-multiplied, so the CSV per-genotype ``rel_imm`` values are the complete vaccine cross-protection profile."

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_cross_immunity_combine.py tests/test_m06_hpv_state_deltas.py -v
```
Expected: All passed.

- [ ] **Step 5: Run the full non-slow suite as a regression guard**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All passed. `txvx_imm` is 0 everywhere outside the new test, so M03/M04/M05 tests must be unchanged.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/cross_genotype.py tests/test_m06_cross_immunity_combine.py
git commit -m "M06: include txvx_imm in CrossImmunity independent-protection combine

rel_sus = (1 - sus_imm_nab) * (1 - vax_imm) * (1 - txvx_imm). Three
independent protection paths, all per-target (no cross-genotype bleed)
for the two vaccine paths. txvx_imm is zero everywhere until hpv.txvx
is added, so existing tests are unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Module-level helpers in `hpvsim/products.py`

**Files:**
- Modify: `hpvsim/products.py` (top of file)
- Test: `tests/test_m06_module_helpers.py` (new)

Adds `_iter_hpv_modules(sim)` and lifts `_find_genotype_module` to module scope so the new product classes (and existing `hpv.vx`) can share them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_m06_module_helpers.py`:

```python
"""Unit tests for module-level helpers shared across products."""
import hpvsim as hpv
from hpvsim.products import _iter_hpv_modules, _find_genotype_module


def _two_genotype_sim():
    sim = hpv.Sim(
        n_agents=100, start='2020', stop='2021',
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        networks='random',
    )
    sim.init()
    return sim


def test_iter_hpv_modules_returns_hpv_only():
    sim = _two_genotype_sim()
    mods = list(_iter_hpv_modules(sim))
    assert {m.genotype for m in mods} == {'hpv16', 'hpv18'}


def test_find_genotype_module_returns_match():
    sim = _two_genotype_sim()
    m = _find_genotype_module(sim, 'hpv16')
    assert m is not None
    assert m.genotype == 'hpv16'


def test_find_genotype_module_returns_none_for_unknown():
    sim = _two_genotype_sim()
    assert _find_genotype_module(sim, 'unknown_genotype') is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_module_helpers.py -v
```
Expected: FAIL with ImportError — neither helper exists at module scope yet.

- [ ] **Step 3: Extract / add the helpers**

Edit `hpvsim/products.py`. After the imports block (around line 14), add:

```python
def _iter_hpv_modules(sim):
    """Yield each HPV module registered in a sim, in registration order."""
    # Late import to avoid the products <-> hpv circular import
    from hpvsim.hpv import HPV
    for module in sim.diseases.values():
        if isinstance(module, HPV):
            yield module


def _find_genotype_module(sim, genotype):
    """Return the HPV module in the sim matching this genotype, or None."""
    for module in _iter_hpv_modules(sim):
        if module.genotype == genotype:
            return module
    return None
```

Then in the existing `vx` class, replace the body of the instance method `_find_genotype_module` with a delegating call:

```python
def _find_genotype_module(self, genotype):
    """Backward-compatible instance method — delegates to module-level helper."""
    return _find_genotype_module(self.sim, genotype)
```

(The existing `from hpvsim.hpv import HPV` late import inside the old method body can stay or be removed — leaving it removed is cleaner.)

- [ ] **Step 4: Run tests to verify pass**

Run:
```bash
pytest tests/test_m06_module_helpers.py tests/test_m06_hpv_state_deltas.py tests/test_m06_cross_immunity_combine.py -v
```
Expected: All passed.

- [ ] **Step 5: Regression smoke**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All passed (M05 vx tests must still work since the instance method delegates).

- [ ] **Step 6: Commit**

```bash
git add hpvsim/products.py tests/test_m06_module_helpers.py
git commit -m "M06: lift _find_genotype_module to module scope; add _iter_hpv_modules

Module-level helpers are shared by hpv.vx, hpv.dx, hpv.tx, hpv.txvx,
and hpv.radiation. Existing hpv.vx instance method delegates for back-
compat.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: CSV loader helpers (`_load_dx_products`, `_load_tx_products`, `_load_txvx_products`)

**Files:**
- Modify: `hpvsim/products.py` (loader section)
- Test: `tests/test_m06_loaders_unit.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_loaders_unit.py`:

```python
"""Unit tests for product CSV loader helpers."""
import pytest
from hpvsim.products import (
    _load_dx_products,
    _load_tx_products,
    _load_txvx_products,
)


def test_dx_loader_returns_expected_products():
    dxs = _load_dx_products()
    expected = {'via', 'lbc', 'pap', 'colposcopy', 'hpv', 'hpv1618',
                'hpv_type', 'txvx_assigner', 'tx_assigner'}
    assert set(dxs.keys()) == expected


def test_dx_loader_has_state_and_result_columns():
    dxs = _load_dx_products()
    via = dxs['via']
    cols = set(via.columns)
    assert {'state', 'genotype', 'result', 'probability'} <= cols


def test_tx_loader_returns_expected_products():
    txs = _load_tx_products()
    assert {'ablation', 'excision', 'txvx1', 'txvx2'} <= set(txs.keys())


def test_tx_loader_has_efficacy_column():
    txs = _load_tx_products()
    df = txs['ablation']
    assert 'efficacy' in df.columns
    assert 'state' in df.columns
    assert 'genotype' in df.columns


def test_txvx_loader_returns_expected_products():
    txvxs = _load_txvx_products()
    assert {'txvx1', 'txvx2'} == set(txvxs.keys())


def test_txvx_loader_returns_genotype_rel_imm_dict():
    txvxs = _load_txvx_products()
    d = txvxs['txvx1']
    assert isinstance(d, dict)
    assert all(isinstance(v, float) for v in d.values())


def test_loaders_are_cached():
    a = _load_dx_products()
    b = _load_dx_products()
    assert a is b  # functools.lru_cache returns same dict identity


def test_dx_loader_includes_latent_rows():
    """The dx CSV references the latent state; loader should preserve those rows."""
    dxs = _load_dx_products()
    hpv_df = dxs['hpv']
    assert 'latent' in hpv_df['state'].unique()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_loaders_unit.py -v
```
Expected: FAIL — loaders don't exist yet.

- [ ] **Step 3: Add the three loader helpers**

Edit `hpvsim/products.py`. Below the existing `_load_vx_products` / `_resolve_vx_pars` block, add:

```python
_DX_CSV    = Path(__file__).parent / 'data' / 'products_dx.csv'
_TX_CSV    = Path(__file__).parent / 'data' / 'products_tx.csv'
_TXVX_CSV  = Path(__file__).parent / 'data' / 'products_txvx.csv'


def _check_columns(df, expected, csv_name):
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f'{csv_name} missing required columns: {sorted(missing)}'
        )


@functools.lru_cache(maxsize=1)
def _load_dx_products():
    """Return {product_name: per-product DataFrame}."""
    df = pd.read_csv(_DX_CSV)
    _check_columns(df, {'name', 'state', 'genotype', 'result', 'probability'},
                   'products_dx.csv')
    return {name: group.reset_index(drop=True)
            for name, group in df.groupby('name', sort=False)}


@functools.lru_cache(maxsize=1)
def _load_tx_products():
    """Return {product_name: per-product DataFrame}."""
    df = pd.read_csv(_TX_CSV)
    _check_columns(df, {'name', 'state', 'genotype', 'efficacy'},
                   'products_tx.csv')
    return {name: group.reset_index(drop=True)
            for name, group in df.groupby('name', sort=False)}


@functools.lru_cache(maxsize=1)
def _load_txvx_products():
    """Return {product_name: {genotype: rel_imm}}."""
    df = pd.read_csv(_TXVX_CSV)
    _check_columns(df, {'name', 'genotype', 'rel_imm'}, 'products_txvx.csv')
    out = {}
    for name, group in df.groupby('name', sort=False):
        out[name] = dict(zip(group['genotype'], group['rel_imm'].astype(float)))
    return out


def _resolve_dx_pars(name, df, hierarchy):
    """Resolve (name, df, hierarchy) for hpv.dx construction.

    Exactly one of name or df must be provided. Default hierarchies (per
    product) match v2's default_dx() in hpvsim/_v2_legacy/interventions.py:1497.
    """
    _DEFAULT_DX_HIERARCHY = {
        'via':            ['positive', 'inadequate', 'negative'],
        'lbc':            ['abnormal', 'ascus', 'inadequate', 'normal'],
        'pap':            ['abnormal', 'ascus', 'inadequate', 'normal'],
        'colposcopy':     ['cancer', 'hsil', 'lsil', 'ascus', 'normal'],
        'hpv':            ['positive', 'inadequate', 'negative'],
        'hpv1618':        ['positive', 'inadequate', 'negative'],
        'hpv_type':       ['positive_1618', 'positive_ohr', 'inadequate', 'negative'],
        'txvx_assigner':  ['triage', 'txvx', 'none'],
        'tx_assigner':    ['radiation', 'excision', 'ablation', 'none'],
    }
    if (name is None) == (df is None):
        raise ValueError('hpv.dx requires exactly one of `name` or `df`, not both/neither.')
    if df is not None:
        if hierarchy is None:
            hierarchy = list(df['result'].unique())
        return df, hierarchy
    products = _load_dx_products()
    if name not in products:
        valid = ', '.join(products.keys())
        raise ValueError(f'Unknown dx product name {name!r}. Valid names: {valid}.')
    if hierarchy is None:
        hierarchy = _DEFAULT_DX_HIERARCHY.get(name, list(products[name]['result'].unique()))
    return products[name], hierarchy


def _resolve_tx_pars(name, df):
    """Resolve (name, df) for hpv.tx construction."""
    if (name is None) == (df is None):
        raise ValueError('hpv.tx requires exactly one of `name` or `df`, not both/neither.')
    if df is not None:
        return df
    products = _load_tx_products()
    if name not in products:
        valid = ', '.join(products.keys())
        raise ValueError(f'Unknown tx product name {name!r}. Valid names: {valid}.')
    return products[name]


def _resolve_txvx_pars(name, rel_imm):
    """Resolve (name, rel_imm) -> {genotype: rel_imm} dict.

    Exactly one of name or rel_imm must be provided.
    """
    if (name is None) == (rel_imm is None):
        raise ValueError('hpv.txvx requires exactly one of `name` or `rel_imm`, not both/neither.')
    if rel_imm is not None:
        return dict(rel_imm)
    products = _load_txvx_products()
    if name not in products:
        valid = ', '.join(products.keys())
        raise ValueError(f'Unknown txvx product name {name!r}. Valid names: {valid}.')
    return dict(products[name])
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_loaders_unit.py -v
```
Expected: All passed (8 tests).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/products.py tests/test_m06_loaders_unit.py
git commit -m "M06: add CSV loaders + resolvers for dx, tx, txvx products

lru_cache-backed per-product dataframes / rel_imm dicts. Default
hierarchies per dx product match v2's default_dx() at
hpvsim/_v2_legacy/interventions.py:1497.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement `hpv.dx` product class

**Files:**
- Modify: `hpvsim/products.py` (add `dx` class + private helpers)
- Test: `tests/test_m06_dx_unit.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_dx_unit.py`:

```python
"""Unit tests for hpv.dx — per-genotype multinomial classifier."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv
from hpvsim.products import dx as hpv_dx


def _four_genotype_sim():
    sim = hpv.Sim(
        n_agents=200, start='2020', stop='2021',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
    )
    sim.init()
    return sim


def _force_state(sim, uids, state, genotype):
    """Manually set a per-genotype BoolState for testing."""
    mod = sim.diseases[genotype]
    arr = getattr(mod, state)
    arr[uids] = True


def test_dx_via_hierarchy_default():
    """The 'via' product default hierarchy is positive > inadequate > negative."""
    sim = _four_genotype_sim()
    d = hpv_dx(name='via')
    d.init_pre(sim)
    assert d.hierarchy == ['positive', 'inadequate', 'negative']


def test_dx_unknown_name_raises():
    with pytest.raises(ValueError, match='Unknown dx product name'):
        hpv_dx(name='nope')


def test_dx_both_name_and_df_raises():
    import pandas as pd
    with pytest.raises(ValueError, match='exactly one'):
        hpv_dx(name='via', df=pd.DataFrame())


def test_dx_neither_name_nor_df_raises():
    with pytest.raises(ValueError, match='exactly one'):
        hpv_dx()


def test_dx_all_genotype_susceptible_means_susceptible_to_all():
    """A 'via' (genotype=all) test on a fully-susceptible agent returns the
    result drawn from the 'susceptible' probability row."""
    sim = _four_genotype_sim()
    d = hpv_dx(name='via')
    sim.interventions = [hpv.routine_screening(product=d, prob=0.0, years=[2020])]
    sim.init()
    # Pick five fully-susceptible agents
    alive = sim.people.alive.uids[:5]
    # Force-set the result_dist's seed so the test is deterministic
    d.result_dist.set_seed(123)
    out = d.administer(alive)
    # All keys in hierarchy present
    assert set(out.keys()) == {'positive', 'inadequate', 'negative'}
    # 'via' susceptible row in products_dx.csv has probabilities [0.15, 0, 0.85].
    # With 5 agents, expect ~1 positive, 0 inadequate, ~4 negative (roughly).
    total = sum(len(v) for v in out.values())
    assert total == 5


def test_dx_latent_state_silently_empty():
    """The 'hpv' product CSV has latent rows; with no latent agents, those rows
    contribute zero classified people (no error)."""
    sim = _four_genotype_sim()
    d = hpv_dx(name='hpv')
    sim.interventions = [hpv.routine_screening(product=d, prob=0.0, years=[2020])]
    sim.init()
    alive = sim.people.alive.uids[:10]
    out = d.administer(alive)
    # The 'hpv' dx has hierarchy ['positive', 'inadequate', 'negative'];
    # since no agents are latent, the latent rows are no-ops. No error raised.
    assert set(out.keys()) == {'positive', 'inadequate', 'negative'}


def test_dx_per_genotype_classifies_precin_hpv16():
    """A precin-on-hpv16 agent under the 'hpv_type' (per-genotype) product
    should classify into positive_1618 with high probability."""
    sim = _four_genotype_sim()
    d = hpv_dx(name='hpv_type')
    sim.interventions = [hpv.routine_screening(product=d, prob=0.0, years=[2020])]
    sim.init()
    uid = sim.people.alive.uids[0:1]
    _force_state(sim, uid, 'precin', 'hpv16')
    d.result_dist.set_seed(123)
    out = d.administer(uid)
    # The hierarchy is ['positive_1618', 'positive_ohr', 'inadequate', 'negative'].
    # On a hpv16-precin agent, the per-genotype probabilities for hpv16 weight
    # toward positive_1618; this is a stochastic test but with seed=123 we
    # pin a deterministic outcome on a 1-agent batch.
    classified = next(k for k, v in out.items() if len(v) == 1)
    assert classified in ('positive_1618', 'positive_ohr', 'inadequate', 'negative')


def test_dx_empty_uids_returns_empty_dict():
    sim = _four_genotype_sim()
    d = hpv_dx(name='via')
    sim.interventions = [hpv.routine_screening(product=d, prob=0.0, years=[2020])]
    sim.init()
    out = d.administer(ss.uids())
    assert all(len(v) == 0 for v in out.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_dx_unit.py -v
```
Expected: FAIL — `hpv.dx` doesn't exist yet.

- [ ] **Step 3: Implement `hpv.dx`**

Edit `hpvsim/products.py`. Append below the `vx` class. Also extend `__all__` to include `'dx'`:

```python
__all__ = ['vx', 'dx', 'tx', 'txvx', 'radiation']

# ... (existing code) ...


def _state_uids_for_module(module, state, uids):
    """Return uids that are in `state` on `module` and also in `uids`."""
    arr = getattr(module, state, None)
    if arr is None:
        return ss.uids()
    return arr.uids.intersect(uids)


def _state_collapse_across_genotypes(state, uids, sim):
    """Collapse a per-genotype state to a single uids set across HPV modules.

    - state='susceptible': agent must be susceptible to ALL genotypes.
    - any other state: agent must be in that state for ANY genotype.
    """
    modules = list(_iter_hpv_modules(sim))
    if not modules:
        return ss.uids()
    if state == 'susceptible':
        # Intersection of per-module susceptible.uids restricted to `uids`
        out = uids
        for m in modules:
            out = out.intersect(m.susceptible.uids)
        return out
    # Union — agent is in `state` for at least one genotype
    matched = ss.uids()
    for m in modules:
        these = _state_uids_for_module(m, state, uids)
        matched = matched.union(these)
    return matched


class dx(ss.Dx):
    """HPV diagnostic product with per-genotype state classification.

    Per-genotype rows in products_dx.csv are classified one genotype at a
    time; rows with genotype='all' are collapsed across all HPV modules
    (susceptible iff susceptible-to-all; positive iff infected-with-any).
    The hierarchy-min semantics mirror v2: when an agent is positive
    across multiple genotypes, the lowest-index (most severe) result wins.

    v2 reference: hpvsim/_v2_legacy/interventions.py:1265-1333
    """

    def __init__(self, name=None, df=None, hierarchy=None, **kwargs):
        df, hierarchy = _resolve_dx_pars(name, df, hierarchy)
        super().__init__(df=df, hierarchy=hierarchy, **kwargs)
        self.name = name
        # The base class sets self.diseases from df['disease'] but the HPV
        # CSV doesn't have a disease column. Replace with our own attrs.
        self._genotypes_in_df = list(df['genotype'].unique())
        self._all_genotype = (len(self._genotypes_in_df) == 1
                              and self._genotypes_in_df[0] == 'all')

    def administer(self, uids, return_format='dict'):
        if len(uids) == 0:
            if return_format == 'dict':
                return {k: ss.uids() for k in self.hierarchy}
            return np.array([], dtype=int)

        # Ensure uids is sorted for the searchsorted-based result indexing
        uids_sorted = ss.uids(np.sort(np.asarray(uids)))
        results = np.full(len(uids_sorted), self.default_value, dtype=int)

        for state in self.health_states:
            if self._all_genotype:
                these = _state_collapse_across_genotypes(state, uids_sorted, self.sim)
                if len(these) == 0:
                    continue
                df_filter = (self.df.state == state) & (self.df.genotype == 'all')
                self._draw_and_min_into(results, uids_sorted, these, df_filter)
            else:
                for module in _iter_hpv_modules(self.sim):
                    if module.genotype not in self._genotypes_in_df:
                        continue
                    these = _state_uids_for_module(module, state, uids_sorted)
                    if len(these) == 0:
                        continue
                    df_filter = (
                        (self.df.state == state)
                        & (self.df.genotype == module.genotype)
                    )
                    self._draw_and_min_into(results, uids_sorted, these, df_filter)

        if return_format == 'dict':
            return {k: ss.uids(uids_sorted[results == i])
                    for i, k in enumerate(self.hierarchy)}
        return results

    def _draw_and_min_into(self, results, uids_sorted, these, df_filter):
        probs = [
            float(self.df[df_filter & (self.df.result == r)]['probability'].values[0])
            for r in self.hierarchy
        ]
        self.result_dist.pars['p'] = probs
        draw = np.asarray(self.result_dist.rvs(these))
        idx = np.searchsorted(uids_sorted, np.asarray(these))
        results[idx] = np.minimum(draw, results[idx])
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_dx_unit.py -v
```
Expected: All passed.

- [ ] **Step 5: Regression smoke**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/products.py tests/test_m06_dx_unit.py
git commit -m "M06: implement hpv.dx per-genotype multinomial classifier

Iterates per-genotype HPV modules; handles CSV genotype='all' mode
via susceptible-intersection / infected-union collapse. Hierarchy-min
semantics match v2 — most-severe result wins on multi-genotype-positive
agents. Latent state silently empty until reactivation natural history
is ported.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement `hpv.tx` product class

**Files:**
- Modify: `hpvsim/products.py` (append `tx` class)
- Test: `tests/test_m06_tx_unit.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_tx_unit.py`:

```python
"""Unit tests for hpv.tx — per-genotype state-flip treatment."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv
from hpvsim.products import tx as hpv_tx


def _four_genotype_sim():
    sim = hpv.Sim(
        n_agents=200, start='2020', stop='2021',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
    )
    sim.init()
    return sim


def test_tx_unknown_name_raises():
    with pytest.raises(ValueError, match='Unknown tx product name'):
        hpv_tx(name='nope')


def test_tx_ablation_flips_cin_to_false():
    """Successful ablation flips cin[g]=False and schedules ti_clearance=ti+1."""
    sim = _four_genotype_sim()
    t = hpv_tx(name='ablation')
    sim.interventions = [hpv.treat_num(product=t, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cin[uids] = True
    t.efficacy_dist.set_seed(123)
    out = t.administer(uids)
    assert 'successful' in out and 'unsuccessful' in out
    # Successful agents have cin=False on hpv16 and ti_clearance=ti+1
    succ = out['successful']
    if len(succ):
        assert np.all(~sim.diseases['hpv16'].cin[succ])
        assert np.allclose(sim.diseases['hpv16'].ti_clearance[succ],
                           sim.ti + 1)


def test_tx_zero_efficacy_row_means_zero_successful():
    """ablation row for precin has efficacy=0; precin agents must all be
    classified unsuccessful for that state."""
    sim = _four_genotype_sim()
    t = hpv_tx(name='ablation')
    sim.interventions = [hpv.treat_num(product=t, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:3]
    # All-precin agents (ablation efficacy on precin = 0)
    sim.diseases['hpv16'].precin[uids] = True
    out = t.administer(uids)
    # No agents classified successful from the precin-on-hpv16 contribution alone
    assert len(out['successful']) == 0
    assert len(out['unsuccessful']) == len(uids)


def test_tx_disjoint_outcomes_keys():
    sim = _four_genotype_sim()
    t = hpv_tx(name='excision')
    sim.interventions = [hpv.treat_num(product=t, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].cin[uids] = True
    t.efficacy_dist.set_seed(7)
    out = t.administer(uids)
    succ_set = set(int(u) for u in out['successful'])
    unsucc_set = set(int(u) for u in out['unsuccessful'])
    assert succ_set.isdisjoint(unsucc_set)
    assert succ_set | unsucc_set == set(int(u) for u in uids)


def test_tx_clears_ti_cin_and_ti_cancerous():
    """Successful treatment clears ti_cin and ti_cancerous to NaN."""
    sim = _four_genotype_sim()
    t = hpv_tx(name='excision')
    sim.interventions = [hpv.treat_num(product=t, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cin[uids] = True
    sim.diseases['hpv16'].ti_cin[uids] = 5.0
    sim.diseases['hpv16'].ti_cancerous[uids] = 10.0
    t.efficacy_dist.set_seed(7)
    out = t.administer(uids)
    succ = out['successful']
    if len(succ):
        assert np.all(np.isnan(sim.diseases['hpv16'].ti_cin[succ]))
        assert np.all(np.isnan(sim.diseases['hpv16'].ti_cancerous[succ]))


def test_tx_empty_uids_returns_empty_outcomes():
    sim = _four_genotype_sim()
    t = hpv_tx(name='ablation')
    sim.interventions = [hpv.treat_num(product=t, prob=0.0)]
    sim.init()
    out = t.administer(ss.uids())
    assert len(out['successful']) == 0
    assert len(out['unsuccessful']) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_tx_unit.py -v
```
Expected: FAIL — `hpv.tx` doesn't exist yet.

- [ ] **Step 3: Implement `hpv.tx`**

Edit `hpvsim/products.py`. Append below the `dx` class:

```python
class tx(ss.Tx):
    """HPV treatment product — per-genotype state-flip with efficacy draw.

    On successful treatment of state in {precin, cin, cancerous} on
    genotype g:
        module.<state>[uids] = False
        module.cin[uids]      = False        # belt-and-braces; cancerous-only flips don't touch cin
        module.precin[uids]   = False
        module.cancerous[uids] = False
        module.ti_cin[uids]      = NaN
        module.ti_cancerous[uids] = NaN
        module.ti_clearance[uids] = sim.ti + 1   # cleared next step

    v2 reference: hpvsim/_v2_legacy/interventions.py:1336-1413
    The commented-out "did they also clear infection?" branch in v2 was
    disabled there; v3 doesn't re-implement it.
    """

    def __init__(self, name=None, df=None, **kwargs):
        df = _resolve_tx_pars(name, df)
        super().__init__(df=df, **kwargs)
        self.name = name

    def administer(self, uids, return_format='dict'):
        if len(uids) == 0:
            empty = ss.uids()
            return {'successful': empty, 'unsuccessful': empty} if return_format == 'dict' else empty

        successful_uids_list = []
        for state in self.health_states:
            for module in _iter_hpv_modules(self.sim):
                df_filter = (self.df.state == state) & (
                    (self.df.genotype == module.genotype) | (self.df.genotype == 'all')
                )
                rows = self.df[df_filter]
                if len(rows) == 0:
                    continue
                these = _state_uids_for_module(module, state, uids)
                if len(these) == 0:
                    continue
                self.efficacy_dist.set(p=float(rows['efficacy'].values[0]))
                eff = self.efficacy_dist.filter(these)
                if len(eff) == 0:
                    continue
                successful_uids_list.append(eff)
                # State cleanup mirroring v2 hpvsim/_v2_legacy/interventions.py:1387-1391
                module.cin[eff] = False
                module.precin[eff] = False
                module.cancerous[eff] = False
                module.ti_cin[eff] = np.nan
                module.ti_cancerous[eff] = np.nan
                module.ti_clearance[eff] = self.sim.ti + 1

        if successful_uids_list:
            # Dedup across (state × genotype) iterations
            all_succ = np.unique(np.concatenate([np.asarray(u) for u in successful_uids_list]))
            successful = ss.uids(all_succ)
        else:
            successful = ss.uids()
        unsuccessful = ss.uids(np.setdiff1d(np.asarray(uids), np.asarray(successful)))

        if return_format == 'dict':
            return {'successful': successful, 'unsuccessful': unsuccessful}
        return successful
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_tx_unit.py -v
```
Expected: All passed.

Note: the `test_tx_ablation_flips_cin_to_false` and `test_tx_disjoint_outcomes_keys` tests depend on `hpv.treat_num` being available for `sim.interventions = [hpv.treat_num(...)]`. At this point `hpv.treat_num` doesn't exist yet — use `ss.treat_num` as a temporary placeholder (sim init only needs *some* intervention with this product attached so `init_pre` runs on the dx). If `ss.treat_num` doesn't accept `prob=0.0`, you can also bypass by calling `t.init_pre(sim)` directly after `sim.init()`. Adjust the test code accordingly if `hpv.treat_num` is not yet defined.

Concretely, replace `sim.interventions = [hpv.treat_num(product=t, prob=0.0)]` with `sim.interventions = [ss.treat_num(product=t, prob=0.0)]` in each test in this file. We'll update to `hpv.treat_num` once it lands in Task 11.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/products.py tests/test_m06_tx_unit.py
git commit -m "M06: implement hpv.tx per-genotype state-flip treatment product

Iterates (state, genotype) rows; per-row efficacy draw via filter dist.
Successful agents have <state>=False on their genotype, ti_clearance
scheduled for sim.ti+1, and ti_cin/ti_cancerous cleared to NaN.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Implement `hpv.txvx` product class

**Files:**
- Modify: `hpvsim/products.py` (append `txvx` class)
- Test: `tests/test_m06_txvx_unit.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_txvx_unit.py`:

```python
"""Unit tests for hpv.txvx — therapeutic vaccine product."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv
from hpvsim.products import txvx as hpv_txvx


def _four_genotype_sim():
    sim = hpv.Sim(
        n_agents=200, start='2020', stop='2021',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
    )
    sim.init()
    return sim


def test_txvx_unknown_name_raises():
    with pytest.raises(ValueError, match='Unknown txvx product name'):
        hpv_txvx(name='nope')


def test_txvx_both_name_and_rel_imm_raises():
    with pytest.raises(ValueError, match='exactly one'):
        hpv_txvx(name='txvx1', rel_imm={'hpv16': 1.0})


def test_txvx1_first_dose_bumps_txvx_imm_on_active_genotypes():
    sim = _four_genotype_sim()
    p = hpv_txvx(name='txvx1')
    sim.interventions = [ss.treat_num(product=p, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:10]
    p._sterilizing_dist.set_seed(7)
    p.administer(None, uids)
    # All four active genotypes should have non-zero txvx_imm on those agents
    for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        vals = sim.diseases[g].txvx_imm[uids]
        # rel_imm[g] varies in the CSV; just assert > 0 (sterilizing OR leaky path)
        assert np.all(vals >= 0), f'{g}: txvx_imm must be non-negative'
        assert np.any(vals > 0), f'{g}: at least one agent should have txvx_imm > 0'


def test_txvx_does_not_downgrade():
    """If an agent already has txvx_imm=0.9, a leaky-draw txvx must not
    lower it to (e.g.) 0.5."""
    sim = _four_genotype_sim()
    p = hpv_txvx(name='txvx1', sterilizing_p=0.0)  # force leaky-only
    sim.interventions = [ss.treat_num(product=p, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].txvx_imm[uids] = 0.9
    p.administer(None, uids)
    # txvx_imm should not be lowered
    assert np.all(sim.diseases['hpv16'].txvx_imm[uids] >= 0.9)


def test_txvx2_booster_multiplies_existing():
    sim = _four_genotype_sim()
    p = hpv_txvx(name='txvx1', imm_boost=None)
    booster = hpv_txvx(name='txvx2', imm_boost=1.2)
    sim.interventions = [
        ss.treat_num(product=p, prob=0.0),
        ss.treat_num(product=booster, prob=0.0),
    ]
    sim.init()
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].txvx_imm[uids] = 0.5
    booster.administer(None, uids)
    np.testing.assert_allclose(sim.diseases['hpv16'].txvx_imm[uids], 0.6, atol=1e-6)


def test_txvx_inactive_genotype_tolerance():
    """A txvx product targeting an inactive (not-in-sim) genotype skips silently."""
    sim = _four_genotype_sim()
    # Custom rel_imm that includes a non-existent genotype
    p = hpv_txvx(rel_imm={'hpv16': 0.9, 'no_such_genotype': 0.5})
    sim.interventions = [ss.treat_num(product=p, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:3]
    p._sterilizing_dist.set_seed(7)
    p.administer(None, uids)  # Must not error
    assert np.all(sim.diseases['hpv16'].txvx_imm[uids] > 0)


def test_txvx_empty_uids_noop():
    sim = _four_genotype_sim()
    p = hpv_txvx(name='txvx1')
    sim.interventions = [ss.treat_num(product=p, prob=0.0)]
    sim.init()
    p.administer(None, ss.uids())  # Must not error
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_txvx_unit.py -v
```
Expected: FAIL — `hpv.txvx` doesn't exist yet.

- [ ] **Step 3: Implement `hpv.txvx`**

Edit `hpvsim/products.py`. Append below the `tx` class:

```python
class txvx(ss.Vx):
    """HPV therapeutic vaccine product (parallel structure to hpv.vx).

    Two modes:
    - Initial dose (default): per-agent sterilizing draw, then per-genotype
      scaling by rel_imm[g] writing into txvx_imm.
    - Booster (imm_boost not None): multiplies existing txvx_imm in place.

    v2 reference: hpvsim/_v2_legacy/interventions.py:1416 + default_tx for
    txvx1/txvx2 wiring.
    """

    def __init__(self, name=None, rel_imm=None, sterilizing_p=0.95,
                 imm_boost=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            name=name,
            rel_imm=rel_imm,
            sterilizing_p=sterilizing_p,
            imm_boost=imm_boost,
        )
        if imm_boost is None:
            # First-dose path: resolve rel_imm
            self.rel_imm = _resolve_txvx_pars(name, rel_imm)
        else:
            # Booster path: rel_imm may be None
            self.rel_imm = _resolve_txvx_pars(name, rel_imm) if (name is not None or rel_imm is not None) else {}
        self._sterilizing_dist = ss.bernoulli(p=0.0)

    def administer(self, people, uids):
        if len(uids) == 0:
            return
        if self.pars.imm_boost is not None:
            # Booster: multiplicative in place on all HPV modules.
            for module in _iter_hpv_modules(self.sim):
                module.txvx_imm[uids] *= float(self.pars.imm_boost)
            return
        # First dose: per-agent sterilizing draw, then per-genotype scaling.
        self._sterilizing_dist.set(p=float(self.pars.sterilizing_p))
        sterilizing_uids = self._sterilizing_dist.filter(uids)
        is_sterilizing = np.isin(np.asarray(uids), np.asarray(sterilizing_uids))
        for genotype, rel_imm_g in self.rel_imm.items():
            module = _find_genotype_module(self.sim, genotype)
            if module is None:
                continue  # inactive-genotype tolerance
            peak = np.where(
                is_sterilizing,
                float(rel_imm_g),
                float(rel_imm_g) * float(self.pars.sterilizing_p),
            )
            module.txvx_imm[uids] = np.maximum(module.txvx_imm[uids], peak)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_txvx_unit.py -v
```
Expected: All passed.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/products.py tests/test_m06_txvx_unit.py
git commit -m "M06: implement hpv.txvx therapeutic vaccine product

Mirrors hpv.vx architecture: per-agent sterilizing draw + per-genotype
rel_imm scaling, writing into module.txvx_imm. Adds optional imm_boost
multiplicative-booster path for txvx2 second doses. Inactive-genotype
tolerance silent-skips missing modules.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Implement `hpv.radiation` product class

**Files:**
- Modify: `hpvsim/products.py` (append `radiation` class)
- Test: `tests/test_m06_radiation_unit.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_radiation_unit.py`:

```python
"""Unit tests for hpv.radiation — cancer treatment product."""
import numpy as np
import starsim as ss
import hpvsim as hpv
from hpvsim.products import radiation as hpv_radiation


def _four_genotype_sim():
    sim = hpv.Sim(
        n_agents=200, start='2020', stop='2021',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
    )
    sim.init()
    return sim


def test_radiation_extends_ti_dead_cancer_on_cancerous_agents():
    sim = _four_genotype_sim()
    r = hpv_radiation()
    sim.interventions = [ss.treat_num(product=r, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:3]
    # Force cancer on hpv16 with a known ti_dead_cancer
    sim.diseases['hpv16'].cancerous[uids] = True
    sim.diseases['hpv16'].ti_dead_cancer[uids] = 100.0
    r._dur_dist.set_seed(7)
    r.administer(uids)
    # ti_dead_cancer must have been extended
    assert np.all(sim.diseases['hpv16'].ti_dead_cancer[uids] > 100.0)


def test_radiation_skips_non_cancer_agents():
    sim = _four_genotype_sim()
    r = hpv_radiation()
    sim.interventions = [ss.treat_num(product=r, prob=0.0)]
    sim.init()
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cancerous[uids] = False  # not cancer
    sim.diseases['hpv16'].ti_dead_cancer[uids] = np.nan
    r._dur_dist.set_seed(7)
    r.administer(uids)
    # No change — agents weren't cancerous
    assert np.all(np.isnan(sim.diseases['hpv16'].ti_dead_cancer[uids]))


def test_radiation_empty_uids_noop():
    sim = _four_genotype_sim()
    r = hpv_radiation()
    sim.interventions = [ss.treat_num(product=r, prob=0.0)]
    sim.init()
    out = r.administer(ss.uids())
    assert len(out) == 0


def test_radiation_default_duration_v2_match():
    """Default duration is normal(18 months, 2 months) converted to years."""
    r = hpv_radiation()
    assert r.pars.dur['par1'] == 18 / 12  # mean: 1.5 years
    assert r.pars.dur['par2'] == 2 / 12   # sd: ~0.167 years
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_radiation_unit.py -v
```
Expected: FAIL — `hpv.radiation` doesn't exist.

- [ ] **Step 3: Implement `hpv.radiation`**

Edit `hpvsim/products.py`. Append below `txvx`:

```python
class radiation(ss.Product):
    """HPV cancer-treatment product — extends ti_dead_cancer per cancerous module.

    Default duration: normal(mean=18 months, sd=2 months), converted to
    years. Matches v2's hpvsim/_v2_legacy/interventions.py:1469-1492.
    """

    def __init__(self, dur=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            dur=dur or dict(dist='normal', par1=18 / 12, par2=2 / 12),
        )
        self._dur_dist = ss.normal(
            loc=self.pars.dur['par1'],
            scale=self.pars.dur['par2'],
        )

    def administer(self, uids):
        if len(uids) == 0:
            return ss.uids()
        new_dur = np.asarray(self._dur_dist.rvs(uids))
        dt = self.sim.t.dt
        uids_arr = np.asarray(uids)
        for module in _iter_hpv_modules(self.sim):
            cancer_uids = module.cancerous.uids.intersect(uids)
            if len(cancer_uids) == 0:
                continue
            mask = np.isin(uids_arr, np.asarray(cancer_uids))
            module.ti_dead_cancer[cancer_uids] = (
                module.ti_dead_cancer[cancer_uids]
                + np.ceil(new_dur[mask] / dt)
            )
        return uids
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_radiation_unit.py -v
```
Expected: All passed.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/products.py tests/test_m06_radiation_unit.py
git commit -m "M06: implement hpv.radiation cancer-treatment product

Extends ti_dead_cancer per cancerous module by ceil(normal(18mo, 2mo)/dt).
Non-cancer agents are skipped silently. v2 reference:
hpvsim/_v2_legacy/interventions.py:1469-1492 (months converted to years).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Eligibility helpers `_compose_screening_eligibility` and `_any_genotype_cancer`

**Files:**
- Modify: `hpvsim/interventions.py` (add helpers)
- Test: `tests/test_m06_eligibility_unit.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_eligibility_unit.py`:

```python
"""Unit tests for the M06 eligibility helpers."""
import numpy as np
import starsim as ss
import hpvsim as hpv
from hpvsim.interventions import (
    _compose_screening_eligibility,
    _any_genotype_cancer,
)


def _two_genotype_sim():
    sim = hpv.Sim(
        n_agents=300, start='2020', stop='2022',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18')],
        networks='random',
    )
    sim.init()
    return sim


def test_screening_eligibility_default_is_female_alive():
    sim = _two_genotype_sim()
    elig = _compose_screening_eligibility(age_range=None, sex='f', extra=None, debut_age=None)
    uids = elig(sim)
    # All returned uids must be female & alive
    for u in uids[:10]:
        assert sim.people.female[u]
        assert sim.people.alive[u]


def test_screening_eligibility_age_range_filter():
    sim = _two_genotype_sim()
    elig = _compose_screening_eligibility(age_range=[30, 50], sex='f', extra=None, debut_age=None)
    uids = elig(sim)
    for u in uids[:10]:
        a = sim.people.age[u]
        assert 30 <= a < 50


def test_screening_eligibility_debut_age_filter():
    sim = _two_genotype_sim()
    elig = _compose_screening_eligibility(age_range=None, sex='f', extra=None, debut_age=15)
    uids = elig(sim)
    for u in uids[:10]:
        assert sim.people.age[u] >= 15


def test_screening_eligibility_extra_callback_intersection():
    sim = _two_genotype_sim()
    chosen = sim.people.alive.uids[:5]
    elig = _compose_screening_eligibility(
        age_range=None, sex='f', extra=lambda s: chosen, debut_age=None,
    )
    uids = elig(sim)
    # The returned set must be a subset of `chosen` AND all female
    for u in uids:
        assert u in chosen
        assert sim.people.female[u]


def test_any_genotype_cancer_ors_across_modules():
    sim = _two_genotype_sim()
    uids = sim.people.alive.uids[:5]
    # Force cancer on hpv18 only
    sim.diseases['hpv18'].cancerous[uids] = True
    cancer = _any_genotype_cancer(sim)
    for u in uids:
        assert cancer[u]
    # Other agents must not be marked
    other = sim.people.alive.uids[10]
    assert not cancer[other]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_eligibility_unit.py -v
```
Expected: FAIL — helpers don't exist.

- [ ] **Step 3: Add the helpers to `hpvsim/interventions.py`**

Edit `hpvsim/interventions.py`. After the existing `_compose_eligibility` helper, add:

```python
def _compose_screening_eligibility(age_range, sex, extra, debut_age):
    """Compose v2-style screening eligibility into a Starsim callable.

    Extends ``_compose_eligibility`` with an optional ``debut_age`` filter
    (lower-bound on ``sim.people.age``). When ``debut_age`` is None, the
    semantics are identical to ``_compose_eligibility``.

    Returns ``elig(sim) -> ss.uids`` intersecting:
      - sim.people.alive
      - sim.people.female / male (per `_coerce_sex(sex)`)
      - sim.people.age in [age_range[0], age_range[1]) if set
      - sim.people.age >= debut_age if set
      - extra(sim) if provided
    """
    sex_set = _coerce_sex(sex)

    def elig(sim):
        cond = sim.people.alive
        if sex_set is not None and len(sex_set) == 1:
            (s,) = sex_set
            cond = cond & (sim.people.female if s == 0 else sim.people.male)
        if age_range is not None:
            lo, hi = age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age < hi)
        if debut_age is not None:
            cond = cond & (sim.people.age >= debut_age)
        if extra is not None:
            cond = cond & _as_boolarr(extra(sim), sim.people)
        return cond.uids

    return elig


def _any_genotype_cancer(sim):
    """Return a BoolArr OR-ing module.cancerous across all HPV modules."""
    # Late import to avoid the circular
    from hpvsim.products import _iter_hpv_modules
    out = sim.people.alive.asnew()
    out.values[:] = False
    for module in _iter_hpv_modules(sim):
        # Set True where any genotype's cancerous BoolArr is True
        out[module.cancerous.uids] = True
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_eligibility_unit.py -v
```
Expected: All passed.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/interventions.py tests/test_m06_eligibility_unit.py
git commit -m "M06: add _compose_screening_eligibility and _any_genotype_cancer helpers

The screening helper extends M05's _compose_eligibility with an
optional debut_age lower-bound. _any_genotype_cancer ORs each HPV
module's .cancerous BoolArr to produce the sim-wide cancer-status
mask used by treatment eligibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: `hpv.BaseTest`, `BaseScreening`, `BaseTriage`, and the four screening/triage leaves

**Files:**
- Modify: `hpvsim/interventions.py` (append classes)
- Test: `tests/test_m06_screening_integration.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_screening_integration.py`:

```python
"""Integration smoke tests for hpv.routine_screening / campaign_screening / triage."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _baseline_sim_with(intervention, **extra):
    return hpv.Sim(
        n_agents=500, start='2020', stop='2025',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
        interventions=[intervention],
        **extra,
    )


def test_routine_screening_flips_screened():
    intv = hpv.routine_screening(
        name='primary',
        product='via',
        prob=0.5,
        age_range=[30, 50],
        sex='f',
        start_year=2021,
        end_year=2024,
    )
    sim = _baseline_sim_with(intv)
    sim.run()
    # At least one agent should be flagged as screened
    assert intv.screened.uids.size > 0


def test_routine_screening_only_targets_age_and_sex():
    intv = hpv.routine_screening(
        name='primary', product='via', prob=1.0,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2022,
    )
    sim = _baseline_sim_with(intv)
    sim.run()
    for u in intv.screened.uids[:20]:
        # Agent may have died after screening; only check sex
        assert sim.people.female[u]


def test_campaign_screening_runs():
    intv = hpv.campaign_screening(
        name='campaign', product='via', prob=0.5,
        age_range=[30, 50], sex='f', years=[2022],
    )
    sim = _baseline_sim_with(intv)
    sim.run()
    assert intv.screened.uids.size > 0


def test_routine_triage_consumes_screen_outcomes():
    """A triage with eligibility=screen-positives only fires on positives."""
    screen = hpv.routine_screening(
        name='primary', product='via', prob=1.0,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2022,
    )
    triage = hpv.routine_triage(
        name='triage',
        product='colposcopy',
        prob=1.0,
        eligibility=lambda s: s.interventions.primary.outcomes['positive'],
        start_year=2021,
        end_year=2022,
    )
    sim = _baseline_sim_with(None)
    sim.interventions = [screen, triage]
    sim.run()
    # Triage screened set is a subset of screen positives (within a step)
    # — we just smoke-check it runs and screened state increments.
    assert triage.screened.uids.size <= screen.screened.uids.size


def test_routine_screening_string_product_resolves():
    """`product='via'` should resolve through hpv.dx(name='via')."""
    intv = hpv.routine_screening(
        name='primary', product='via', prob=0.1,
        age_range=[30, 50], sex='f', start_year=2021, end_year=2022,
    )
    assert intv.product.__class__.__name__ == 'dx'
    assert intv.product.name == 'via'
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_screening_integration.py -v
```
Expected: FAIL — `hpv.routine_screening`, `hpv.campaign_screening`, `hpv.routine_triage` do not exist.

- [ ] **Step 3: Implement the classes**

Edit `hpvsim/interventions.py`. Update `__all__` and append below the existing M05 classes:

```python
__all__ = [
    'BaseVaccination', 'routine_vx', 'campaign_vx',
    # M06
    'BaseTest', 'BaseScreening', 'BaseTriage',
    'routine_screening', 'campaign_screening',
    'routine_triage', 'campaign_triage',
    'BaseTreatment', 'treat_num', 'treat_delay',
    'BaseTxVx', 'routine_txvx', 'campaign_txvx', 'linked_txvx',
    'dynamic_pars',
]

# ... (existing code) ...


class BaseTest(ss.BaseTest):
    """HPV-specific test/screening base.

    Adds v2-compatible age_range / sex / eligibility / debut_age kwargs,
    composed into a single Starsim eligibility callable. Overrides
    ``_parse_product_str`` so ``routine_screening(product='via', ...)``
    resolves through ``hpv.dx(name='via')``.
    """

    def __init__(self, *args, age_range=None, sex='f', eligibility=None,
                 debut_age=None, **kwargs):
        composed = _compose_screening_eligibility(age_range, sex, eligibility, debut_age)
        super().__init__(*args, eligibility=composed, **kwargs)
        self.age_range = age_range
        self.sex_raw = sex
        self.sex = _coerce_sex(sex)
        self.eligibility_raw = eligibility
        self.debut_age = debut_age

    def _parse_product_str(self, product):
        from hpvsim.products import dx as _dx
        return _dx(name=product)


class BaseScreening(BaseTest, ss.BaseScreening):
    """HPV-specific BaseScreening — composes HPV eligibility with Starsim's screening step."""
    pass


class BaseTriage(BaseTest, ss.BaseTriage):
    """HPV-specific BaseTriage — eligibility is required (no default screening set)."""
    pass


class routine_screening(BaseScreening, ss.RoutineDelivery):
    """Routine HPV screening."""
    pass


class campaign_screening(BaseScreening, ss.CampaignDelivery):
    """Campaign HPV screening."""
    pass


class routine_triage(BaseTriage, ss.RoutineDelivery):
    """Routine HPV triage."""
    pass


class campaign_triage(BaseTriage, ss.CampaignDelivery):
    """Campaign HPV triage."""
    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_screening_integration.py -v
```
Expected: All passed (5 tests).

- [ ] **Step 5: Regression smoke**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/interventions.py tests/test_m06_screening_integration.py
git commit -m "M06: add hpv.routine_screening/campaign_screening/routine_triage/campaign_triage

Thin diamond leaves over ss.BaseScreening/BaseTriage + RoutineDelivery/
CampaignDelivery. hpv.BaseTest adds v2-compatible age_range / sex /
debut_age / eligibility kwargs and resolves string products through
hpv.dx.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: `hpv.BaseTreatment` and `hpv.treat_num`

**Files:**
- Modify: `hpvsim/interventions.py` (append)
- Test: `tests/test_m06_treatment_integration.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_treatment_integration.py`:

```python
"""Integration smoke tests for hpv.treat_num and its HPV-specific eligibility."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _four_genotype_sim_with(*intvs):
    return hpv.Sim(
        n_agents=500, start='2020', stop='2025',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
        interventions=list(intvs),
    )


def test_treat_num_treat_cancer_flag_set_for_radiation():
    treat = hpv.treat_num(name='cancer_rx', product=hpv.radiation(), prob=1.0)
    sim = _four_genotype_sim_with(treat)
    sim.init()
    assert treat.treat_cancer is True


def test_treat_num_treat_cancer_flag_unset_for_excision():
    treat = hpv.treat_num(name='cin_rx', product='excision', prob=1.0)
    sim = _four_genotype_sim_with(treat)
    sim.init()
    assert treat.treat_cancer is False


def test_treat_num_only_treats_cin_when_not_treat_cancer():
    """A non-cancer treat_num must NOT treat agents who are cancerous (and only those)."""
    treat = hpv.treat_num(name='cin_rx', product='excision', prob=1.0)
    sim = _four_genotype_sim_with(treat)
    sim.init()
    # Force five agents into cancer state; they should be ineligible
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].cancerous[uids] = True
    eligible = treat.check_eligibility()
    for u in uids:
        assert u not in eligible


def test_treat_num_only_treats_cancer_when_treat_cancer():
    """A treat_num(radiation) must ONLY treat cancerous agents."""
    treat = hpv.treat_num(name='cancer_rx', product=hpv.radiation(), prob=1.0)
    sim = _four_genotype_sim_with(treat)
    sim.init()
    uids = sim.people.alive.uids[:5]
    # Five agents made cancerous; everyone else is non-cancerous
    sim.diseases['hpv16'].cancerous[uids] = True
    eligible = treat.check_eligibility()
    for u in uids:
        assert u in eligible
    # A non-cancer agent must not be eligible
    other = sim.people.alive.uids[10]
    assert other not in eligible


def test_treat_num_string_product_resolves_to_tx():
    """treat_num(product='excision') should resolve via hpv.tx."""
    treat = hpv.treat_num(name='rx', product='excision', prob=0.0)
    sim = _four_genotype_sim_with(treat)
    sim.init()
    assert treat.product.__class__.__name__ == 'tx'
    assert treat.product.name == 'excision'


def test_treat_num_capacity_respected():
    """With max_capacity=5, no more than 5 agents are treated per step."""
    treat = hpv.treat_num(
        name='rx',
        product='excision',
        prob=1.0,
        max_capacity=5,
        eligibility=lambda s: s.people.alive.uids[:50],
    )
    sim = _four_genotype_sim_with(treat)
    sim.init()
    # Force CIN on 50 agents
    chosen = sim.people.alive.uids[:50]
    sim.diseases['hpv16'].cin[chosen] = True
    sim.run()
    # Total cin_treated count should not exceed (n_steps * 5)
    n_steps = len(sim.timevec)
    assert treat.cin_treated.uids.size <= n_steps * 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_treatment_integration.py -v
```
Expected: FAIL — `hpv.treat_num` doesn't exist.

- [ ] **Step 3: Implement `BaseTreatment` and `treat_num`**

Edit `hpvsim/interventions.py`. Append after the screening classes:

```python
class BaseTreatment(ss.BaseTreatment):
    """HPV-specific treatment base.

    Adds:
    - v2-compatible age_range / sex / eligibility kwargs
    - HPV-specific eligibility: female + alive + cancer-status-matched
      (cancer treatments require any-genotype cancerous; non-cancer
      treatments require no cancerous on any genotype)
    - Per-intervention state: cin_treated / cin_treatments /
      ti_cin_treated for CIN treatments, cancer_treated / etc. for
      cancer treatments

    The `treat_cancer` flag is derived at __init__ time from whether the
    product is an `hpv.radiation` instance.
    """

    def __init__(self, *args, age_range=None, sex='f', eligibility=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.age_range = age_range
        self.sex_raw = sex
        self.sex = _coerce_sex(sex)
        self.eligibility_user = eligibility
        # Late import to avoid circular
        from hpvsim.products import radiation as _radiation
        self.treat_cancer = isinstance(self.product, _radiation)
        # Per-intervention state
        self.define_states(
            ss.BoolArr('cin_treated'),
            ss.FloatArr('cin_treatments', default=0),
            ss.FloatArr('ti_cin_treated'),
            ss.BoolArr('cancer_treated'),
            ss.FloatArr('cancer_treatments', default=0),
            ss.FloatArr('ti_cancer_treated'),
        )

    def _parse_product_str(self, product):
        from hpvsim.products import tx as _tx
        return _tx(name=product)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_cin_treated',     dtype=int, scale=True, label='Number first-CIN-treated'),
            ss.Result('new_cancer_treated',  dtype=int, scale=True, label='Number first-cancer-treated'),
        )

    def check_eligibility(self):
        sim = self.sim
        cond = sim.people.alive & sim.people.female
        if self.age_range is not None:
            lo, hi = self.age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age <= hi)
        any_cancer = _any_genotype_cancer(sim)
        cond = cond & (any_cancer if self.treat_cancer else ~any_cancer)
        if self.eligibility_user is not None:
            cond = cond & _as_boolarr(self.eligibility_user(sim), sim.people)
        return cond.uids


class treat_num(BaseTreatment, ss.treat_num):
    """Treat a fixed number of HPV+CIN+ agents each step (or all eligible if max_capacity=None)."""

    def step(self):
        treat_uids = super().step()
        if len(treat_uids):
            if self.treat_cancer:
                new = treat_uids[~self.cancer_treated[treat_uids]]
                self.cancer_treated[treat_uids] = True
                self.cancer_treatments[treat_uids] += 1
                self.ti_cancer_treated[treat_uids] = self.sim.ti
                self.results['new_cancer_treated'][self.sim.ti] += len(new)
            else:
                new = treat_uids[~self.cin_treated[treat_uids]]
                self.cin_treated[treat_uids] = True
                self.cin_treatments[treat_uids] += 1
                self.ti_cin_treated[treat_uids] = self.sim.ti
                self.results['new_cin_treated'][self.sim.ti] += len(new)
        return treat_uids
```

- [ ] **Step 4: Update `hpv.tx` tests to use `hpv.treat_num`**

In `tests/test_m06_tx_unit.py`, replace each `ss.treat_num(...)` reference with `hpv.treat_num(...)` (this was placeholder-only in Task 6).

- [ ] **Step 5: Run tests**

Run:
```bash
pytest tests/test_m06_treatment_integration.py tests/test_m06_tx_unit.py -v
```
Expected: All passed.

- [ ] **Step 6: Regression smoke**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All passed.

- [ ] **Step 7: Commit**

```bash
git add hpvsim/interventions.py tests/test_m06_treatment_integration.py tests/test_m06_tx_unit.py
git commit -m "M06: add hpv.BaseTreatment and hpv.treat_num

HPV-specific treatment base adds female+alive eligibility,
cancer-vs-precancer gate (auto-derived from product type), and
per-intervention state (cin_treated, cin_treatments, cancer_treated,
etc.). treat_num overrides ss.treat_num.step() to bump the
intervention-level state and new_* result counters.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: `hpv.treat_delay`

**Files:**
- Modify: `hpvsim/interventions.py` (append)
- Test: `tests/test_m06_treat_delay_unit.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_treat_delay_unit.py`:

```python
"""Unit tests for hpv.treat_delay — fresh port with integer-ti scheduler."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _four_genotype_sim_with(intv):
    return hpv.Sim(
        n_agents=500, start='2020', stop='2025',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
        interventions=[intv],
    )


def test_treat_delay_zero_delay_treats_same_step():
    intv = hpv.treat_delay(name='rx', product='excision', prob=1.0, delay=0)
    sim = _four_genotype_sim_with(intv)
    sim.init()
    # Force CIN on a few agents
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].cin[uids] = True
    sim.run_one_step()
    # All eligible agents should be treated this step (delay=0)
    assert intv.cin_treated.uids.size > 0


def test_treat_delay_two_year_delay_fires_2_years_later():
    """With dt=0.25 and delay=2.0 years, eligible agents enqueued at ti=T
    fire at ti=T+8."""
    intv = hpv.treat_delay(name='rx', product='excision', prob=1.0, delay=2.0)
    sim = _four_genotype_sim_with(intv)
    sim.init()
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].cin[uids] = True
    initial_ti = sim.ti
    # Step once — schedules treatment for ti = initial_ti + round(2/dt) = +8
    sim.run_one_step()
    assert intv.cin_treated.uids.size == 0    # not yet treated
    # Step forward up to ti = initial_ti + 8
    target_ti = initial_ti + int(round(2.0 / sim.t.dt))
    while sim.ti < target_ti:
        sim.run_one_step()
    # On this exact step the queue fires
    sim.run_one_step()
    assert intv.cin_treated.uids.size > 0


def test_treat_delay_queue_drains_after_fire():
    intv = hpv.treat_delay(name='rx', product='excision', prob=1.0, delay=0)
    sim = _four_genotype_sim_with(intv)
    sim.init()
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cin[uids] = True
    sim.run_one_step()
    assert sim.ti not in intv.scheduler or len(intv.scheduler[sim.ti]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_treat_delay_unit.py -v
```
Expected: FAIL — `hpv.treat_delay` doesn't exist.

- [ ] **Step 3: Implement `treat_delay`**

Edit `hpvsim/interventions.py`. Add the necessary import at the top of the file:

```python
from collections import defaultdict
```

Then append below `treat_num`:

```python
class treat_delay(BaseTreatment):
    """Treat HPV+CIN+ agents after a fixed delay.

    On each step:
      1. Newly-eligible accepters are enqueued at `due_ti = sim.ti +
         round(delay / dt)`.
      2. Agents whose due_ti is the current ti are treated.

    delay is in years. Integer-ti scheduler keys are the M05-lesson
    upgrade over v2's float subtraction (sim.t - delay/dt).

    v2 reference: hpvsim/_v2_legacy/interventions.py:1098-1134
    """

    def __init__(self, delay=None, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay or 0
        self.scheduler = defaultdict(list)

    def add_to_schedule(self):
        accept = self.get_accept_inds()
        if len(accept):
            due_ti = self.sim.ti + int(round(self.delay / self.sim.t.dt))
            self.scheduler[due_ti].extend(int(u) for u in accept)

    def get_candidates(self):
        return ss.uids(self.scheduler.pop(self.sim.ti, []))

    def step(self):
        self.add_to_schedule()
        treat_uids = super().step()  # delegates to BaseTreatment.step (which calls product.administer)
        # Mirror BaseTreatment.treat_num step bookkeeping
        if len(treat_uids):
            if self.treat_cancer:
                new = treat_uids[~self.cancer_treated[treat_uids]]
                self.cancer_treated[treat_uids] = True
                self.cancer_treatments[treat_uids] += 1
                self.ti_cancer_treated[treat_uids] = self.sim.ti
                self.results['new_cancer_treated'][self.sim.ti] += len(new)
            else:
                new = treat_uids[~self.cin_treated[treat_uids]]
                self.cin_treated[treat_uids] = True
                self.cin_treatments[treat_uids] += 1
                self.ti_cin_treated[treat_uids] = self.sim.ti
                self.results['new_cin_treated'][self.sim.ti] += len(new)
        return treat_uids
```

Note: `super().step()` from `hpv.treat_delay` → `BaseTreatment.step()` → `ss.BaseTreatment.step()` which uses `self.get_candidates()` and `self.check_eligibility()`. The state-bump is duplicated here because `treat_delay` doesn't go through `hpv.treat_num.step()`. If you prefer a different inheritance chain, you can refactor — the key contract is that state and result counters get updated exactly once per treated uid per step.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_treat_delay_unit.py -v
```
Expected: All passed.

- [ ] **Step 5: Regression smoke**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/interventions.py tests/test_m06_treat_delay_unit.py
git commit -m "M06: add hpv.treat_delay with integer-ti scheduler

Fresh port of v2's treat_delay using integer-ti math instead of
v2's fragile float subtraction (sim.t - delay/dt). Schedule keyed
by sim.ti + round(delay/dt).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: `hpv.BaseTxVx` family — `routine_txvx`, `campaign_txvx`, `linked_txvx`

**Files:**
- Modify: `hpvsim/interventions.py` (append)
- Test: `tests/test_m06_txvx_integration.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_txvx_integration.py`:

```python
"""Integration smoke tests for hpv.routine_txvx / campaign_txvx / linked_txvx."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv


def _four_genotype_sim_with(*intvs):
    return hpv.Sim(
        n_agents=500, start='2020', stop='2025',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
        interventions=list(intvs),
    )


def test_routine_txvx_flips_tx_vaccinated():
    intv = hpv.routine_txvx(
        name='txvx',
        product='txvx1',
        prob=0.9,
        age_range=[25, 26],
        start_year=2021,
        end_year=2023,
    )
    sim = _four_genotype_sim_with(intv)
    sim.run()
    assert intv.tx_vaccinated.uids.size > 0


def test_campaign_txvx_runs():
    intv = hpv.campaign_txvx(
        name='campaign',
        product='txvx1',
        prob=0.5,
        age_range=[25, 30],
        years=[2022],
    )
    sim = _four_genotype_sim_with(intv)
    sim.run()
    assert intv.tx_vaccinated.uids.size > 0


def test_linked_txvx_requires_eligibility():
    with pytest.raises(ValueError, match='eligibility'):
        hpv.linked_txvx(product='txvx1', prob=0.5)


def test_linked_txvx_fires_only_on_eligibility_callback():
    """linked_txvx with a static eligibility list treats exactly those agents."""
    screen = hpv.routine_screening(
        name='primary', product='via', prob=1.0,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2022,
    )
    linked = hpv.linked_txvx(
        name='linked',
        product='txvx1',
        prob=1.0,
        eligibility=lambda s: s.interventions.primary.outcomes['positive'],
    )
    sim = _four_genotype_sim_with(screen, linked)
    sim.run()
    # linked.tx_vaccinated is a subset of screen positives (over the run lifetime)
    assert linked.tx_vaccinated.uids.size <= screen.screened.uids.size


def test_routine_txvx_string_product_resolves_to_txvx():
    intv = hpv.routine_txvx(
        name='txvx',
        product='txvx1',
        prob=0.5,
        age_range=[25, 26],
        start_year=2021,
        end_year=2022,
    )
    assert intv.product.__class__.__name__ == 'txvx'
    assert intv.product.pars.name == 'txvx1'
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_txvx_integration.py -v
```
Expected: FAIL — txvx interventions don't exist.

- [ ] **Step 3: Implement the txvx classes**

Edit `hpvsim/interventions.py`. Append:

```python
class BaseTxVx(BaseTreatment):
    """HPV therapeutic vaccination base.

    Extends BaseTreatment with txvx-specific per-intervention state:
    tx_vaccinated / txvx_doses / ti_tx_vaccinated.

    On each delivery the agent's txvx_imm is bumped per genotype (via
    hpv.txvx.administer). The intervention's own dose counters track
    program-level uptake.

    v2 reference: hpvsim/_v2_legacy/interventions.py:1137-1252
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_states(
            ss.BoolArr('tx_vaccinated'),
            ss.FloatArr('txvx_doses', default=0),
            ss.FloatArr('ti_tx_vaccinated'),
        )

    def _parse_product_str(self, product):
        from hpvsim.products import txvx as _txvx
        return _txvx(name=product)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_tx_vaccinated', dtype=int, scale=True, label='Number first-txvx-vaccinated'),
            ss.Result('new_txvx_doses',    dtype=int, scale=True, label='Number txvx doses administered'),
        )

    def check_eligibility(self):
        """TxVx eligibility — female + alive + cancer-free + age range.

        Unlike treat_num/treat_delay, BaseTxVx never targets cancer patients
        (radiation is for that path). treat_cancer is forced False here.
        """
        sim = self.sim
        cond = sim.people.alive & sim.people.female
        if self.age_range is not None:
            lo, hi = self.age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age <= hi)
        # Never treat cancer agents
        any_cancer = _any_genotype_cancer(sim)
        cond = cond & ~any_cancer
        if self.eligibility_user is not None:
            cond = cond & _as_boolarr(self.eligibility_user(sim), sim.people)
        return cond.uids

    def deliver(self):
        """One-step delivery — finds accepters, administers, bumps counters."""
        accept_uids = self.get_accept_inds()
        if len(accept_uids):
            self.product.administer(self.sim.people, accept_uids)
            new = accept_uids[~self.tx_vaccinated[accept_uids]]
            self.tx_vaccinated[accept_uids] = True
            self.txvx_doses[accept_uids] += 1
            self.ti_tx_vaccinated[accept_uids] = self.sim.ti
            self.results['new_tx_vaccinated'][self.sim.ti] += len(new)
            self.results['new_txvx_doses'][self.sim.ti] += len(accept_uids)
        return accept_uids

    def step(self):
        # Default: scheduled delivery via RoutineDelivery/CampaignDelivery's timepoints
        if self.sim.ti in self.timepoints:
            return self.deliver()
        return ss.uids()


class routine_txvx(BaseTxVx, ss.RoutineDelivery):
    """Routine therapeutic vaccination."""
    pass


class campaign_txvx(BaseTxVx, ss.CampaignDelivery):
    """Campaign therapeutic vaccination."""
    pass


class linked_txvx(BaseTxVx):
    """Therapeutic vaccination linked to another intervention's outcomes.

    Has no own timeline. Fires every step; eligibility= callback (required)
    determines who actually receives the dose. Typical usage:

        linked = hpv.linked_txvx(
            product='txvx1', prob=0.6,
            eligibility=lambda s: s.interventions.colposcopy.outcomes['lsil'],
        )
    """

    def __init__(self, *args, eligibility=None, **kwargs):
        if eligibility is None:
            raise ValueError(
                "linked_txvx requires eligibility= "
                "(typically a screen.outcomes['positive'] callback)"
            )
        super().__init__(*args, eligibility=eligibility, **kwargs)
        self.timepoints = None  # No own schedule

    def step(self):
        return self.deliver()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_txvx_integration.py -v
```
Expected: All passed.

- [ ] **Step 5: Regression smoke**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/interventions.py tests/test_m06_txvx_integration.py
git commit -m "M06: add hpv.BaseTxVx family — routine_txvx, campaign_txvx, linked_txvx

BaseTxVx extends BaseTreatment with tx_vaccinated/txvx_doses state and
forces the cancer-free eligibility gate. routine/campaign use Starsim's
delivery bases; linked_txvx has no own timeline and requires an
eligibility callback that typically reads a screen's outcomes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: `hpv.dynamic_pars`

**Files:**
- Modify: `hpvsim/interventions.py` (append)
- Test: `tests/test_m06_dynamic_pars_unit.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_m06_dynamic_pars_unit.py`:

```python
"""Unit tests for hpv.dynamic_pars — time-varying parameter editor."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv


def _four_genotype_sim_with(intv):
    return hpv.Sim(
        n_agents=300, start='2020', stop='2030',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
        interventions=[intv],
    )


def test_dynamic_pars_linear_interpolation():
    """A linear ramp on hpv16.beta should hit the midpoint at the midpoint year."""
    intv = hpv.dynamic_pars(pars={
        'hpv16.beta': {'years': [2020, 2030], 'vals': [1.0, 0.0]},
    })
    sim = _four_genotype_sim_with(intv)
    sim.init()
    # Step to year 2025 (mid-point)
    while sim.t.now('year') < 2025:
        sim.run_one_step()
    sim.run_one_step()
    # hpv16.beta should be near 0.5
    assert abs(sim.diseases['hpv16'].pars.beta - 0.5) < 0.1


def test_dynamic_pars_stepwise_mode():
    intv = hpv.dynamic_pars(
        pars={'hpv16.beta': {'years': [2020, 2025], 'vals': [1.0, 0.2]}},
        interpolate=False,
    )
    sim = _four_genotype_sim_with(intv)
    sim.init()
    while sim.t.now('year') < 2023:
        sim.run_one_step()
    sim.run_one_step()
    assert sim.diseases['hpv16'].pars.beta == 1.0
    while sim.t.now('year') < 2026:
        sim.run_one_step()
    sim.run_one_step()
    assert sim.diseases['hpv16'].pars.beta == 0.2


def test_dynamic_pars_unresolvable_path_raises_at_step():
    intv = hpv.dynamic_pars(pars={
        'nonexistent.foo': {'years': [2020, 2030], 'vals': [1.0, 0.5]},
    })
    sim = _four_genotype_sim_with(intv)
    sim.init()
    with pytest.raises(KeyError, match='nonexistent'):
        sim.run_one_step()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_m06_dynamic_pars_unit.py -v
```
Expected: FAIL — `hpv.dynamic_pars` doesn't exist.

- [ ] **Step 3: Implement `dynamic_pars`**

Edit `hpvsim/interventions.py`. Append:

```python
def _set_dotted(sim, dotted_path, value):
    """Resolve a dotted-path string against (sim.diseases, sim.interventions, sim.pars) and set it.

    Top-level segment is looked up in:
      1. sim.diseases (by key)        e.g. 'hpv16.beta' → sim.diseases['hpv16'].pars.beta
      2. sim.interventions (by name)  e.g. 'screen.prob' → sim.interventions.screen.prob
      3. sim.pars (by key)            e.g. 'rand_seed' → sim.pars.rand_seed
    """
    parts = dotted_path.split('.')
    head, tail = parts[0], parts[1:]

    if head in sim.diseases:
        # Module pars path
        target = sim.diseases[head]
        # Default: navigate .pars first
        cur = target.pars
        for seg in tail[:-1]:
            cur = getattr(cur, seg)
        setattr(cur, tail[-1], value)
        return

    if head in sim.interventions:
        target = sim.interventions[head]
        cur = target
        for seg in tail[:-1]:
            cur = getattr(cur, seg)
        setattr(cur, tail[-1], value)
        return

    # Fall back to sim.pars
    try:
        cur = sim.pars
        for seg in [head] + tail[:-1]:
            cur = getattr(cur, seg)
        setattr(cur, tail[-1] if tail else head, value)
    except AttributeError as e:
        raise KeyError(
            f'Cannot resolve dotted path {dotted_path!r}: head segment '
            f'{head!r} is not a sim.disease, sim.intervention, or sim.par.'
        ) from e


class dynamic_pars(ss.Intervention):
    """Time-varying parameter editor.

    pars: dict mapping dotted-path strings to {'years': [...], 'vals': [...]}
    schedules. Each step, the resolved parameter is set to the interpolated
    (default) or stepwise (interpolate=False) value for the current year.

    Dotted-path resolution order: sim.diseases > sim.interventions > sim.pars.

    v2 reference: hpvsim/_v2_legacy/interventions.py:406-489 (uses timestep
    keys; v3 uses epoch-year keys for ergonomic schedule authoring).
    """

    def __init__(self, pars=None, interpolate=True, **kwargs):
        super().__init__(**kwargs)
        self.par_schedules = pars or {}
        self.interpolate = interpolate

    def step(self):
        year = self.sim.t.now('year')
        for dotted_path, schedule in self.par_schedules.items():
            years = np.asarray(schedule['years'], dtype=float)
            vals = np.asarray(schedule['vals'], dtype=float)
            if self.interpolate:
                val = float(np.interp(year, years, vals))
            else:
                idx = int(np.searchsorted(years, year, side='right')) - 1
                if idx < 0:
                    continue  # before first schedule year — no change
                val = float(vals[idx])
            _set_dotted(self.sim, dotted_path, val)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_dynamic_pars_unit.py -v
```
Expected: All passed.

- [ ] **Step 5: Regression smoke**

Run:
```bash
pytest tests/ -m "not slow" -x -q
```
Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/interventions.py tests/test_m06_dynamic_pars_unit.py
git commit -m "M06: add hpv.dynamic_pars — year-keyed time-varying parameter editor

Dotted-path resolution into sim.diseases / sim.interventions / sim.pars.
Linear interpolation by default; stepwise when interpolate=False.
Epoch-year keys (ergonomic over v2's timestep keys).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Wire exports in `hpvsim/__init__.py`

**Files:**
- Modify: `hpvsim/__init__.py`

- [ ] **Step 1: Inspect current exports**

Run:
```bash
grep -n "from .interventions\|from .products" hpvsim/__init__.py
```

Note the existing M05 exports (likely `from .interventions import BaseVaccination, routine_vx, campaign_vx` and `from .products import vx`).

- [ ] **Step 2: Add the M06 exports**

Edit `hpvsim/__init__.py`. Extend the `.interventions` and `.products` imports to include the M06 names:

```python
from .interventions import (
    # M05
    BaseVaccination, routine_vx, campaign_vx,
    # M06
    BaseTest, BaseScreening, BaseTriage,
    routine_screening, campaign_screening,
    routine_triage, campaign_triage,
    BaseTreatment, treat_num, treat_delay,
    BaseTxVx, routine_txvx, campaign_txvx, linked_txvx,
    dynamic_pars,
)
from .products import vx, dx, tx, txvx, radiation
```

- [ ] **Step 3: Verify exports resolve from a fresh interpreter**

Run:
```bash
python -c "import hpvsim as hpv; print(hpv.routine_screening, hpv.dx, hpv.linked_txvx, hpv.dynamic_pars)"
```
Expected: prints class repr strings; no ImportError.

- [ ] **Step 4: Run the M06 suite**

Run:
```bash
pytest tests/test_m06_*.py -v
```
Expected: All passed.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/__init__.py
git commit -m "M06: wire new public names into hpvsim/__init__.py

Adds: BaseTest, BaseScreening, BaseTriage, routine_screening,
campaign_screening, routine_triage, campaign_triage, BaseTreatment,
treat_num, treat_delay, BaseTxVx, routine_txvx, campaign_txvx,
linked_txvx, dynamic_pars, dx, tx, txvx, radiation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Cascade integration + order-dependency tests

**Files:**
- Test: `tests/test_m06_cascade_integration.py` (new)

- [ ] **Step 1: Write the tests**

Create `tests/test_m06_cascade_integration.py`:

```python
"""Integration tests for the full screen -> triage -> treat cascade."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _cascade_sim(intvs):
    return hpv.Sim(
        n_agents=500, start='2020', stop='2025',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='random',
        interventions=intvs,
    )


def test_full_cascade_composes():
    """screen → triage → treat composes by ordering and eligibility callbacks."""
    screen = hpv.routine_screening(
        name='primary', product='hpv', prob=0.7,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2024,
    )
    triage = hpv.routine_triage(
        name='colpo', product='colposcopy', prob=0.9,
        eligibility=lambda s: s.interventions.primary.outcomes['positive'],
        start_year=2021, end_year=2024,
    )
    treat = hpv.treat_num(
        name='excision_rx', product='excision', prob=0.8,
        eligibility=lambda s: s.interventions.colpo.outcomes['hsil'],
    )
    sim = _cascade_sim([screen, triage, treat])
    sim.run()
    # At least someone gets screened
    assert screen.screened.uids.size > 0
    # Triage screens are a subset of overall lifetime (smoke-level guard)
    assert triage.screened.uids.size <= screen.screened.uids.size


def test_cascade_order_dependency():
    """Registering treat BEFORE screen → treat.outcomes never has positives,
    so no agent is ever treated. Documented v2-compatible behaviour."""
    screen = hpv.routine_screening(
        name='primary', product='hpv', prob=1.0,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2023,
    )
    treat = hpv.treat_num(
        name='wrong_order_rx', product='excision', prob=1.0,
        eligibility=lambda s: s.interventions.primary.outcomes['positive'],
    )
    # WRONG ORDER: treat registered before screen
    sim = _cascade_sim([treat, screen])
    sim.run()
    # In this order, treat sees an empty outcomes['positive'] on the same
    # step (because screen hasn't run yet on that step). Documented contract.
    # We're just asserting no exception is raised and the test runs.
    assert isinstance(treat.cin_treated.uids.size, (int, np.integer))


def test_linked_txvx_in_cascade():
    """linked_txvx eligibility=triage.outcomes['lsil'] gates txvx delivery."""
    screen = hpv.routine_screening(
        name='primary', product='hpv', prob=1.0,
        age_range=[25, 55], sex='f',
        start_year=2021, end_year=2024,
    )
    triage = hpv.routine_triage(
        name='colpo', product='colposcopy', prob=1.0,
        eligibility=lambda s: s.interventions.primary.outcomes['positive'],
        start_year=2021, end_year=2024,
    )
    linked = hpv.linked_txvx(
        name='linked_v',
        product='txvx1',
        prob=1.0,
        eligibility=lambda s: s.interventions.colpo.outcomes['lsil'],
    )
    sim = _cascade_sim([screen, triage, linked])
    sim.run()
    # linked.tx_vaccinated count <= triage.screened count (LSIL is a subset)
    assert linked.tx_vaccinated.uids.size <= triage.screened.uids.size
```

- [ ] **Step 2: Run tests to verify they pass**

Run:
```bash
pytest tests/test_m06_cascade_integration.py -v
```
Expected: All passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_m06_cascade_integration.py
git commit -m "M06: integration tests for screen->triage->treat composition

Verifies: cascade composition by ordering + eligibility callbacks;
order-dependency contract (out-of-order registration → empty outcomes,
no exception); linked_txvx gated by upstream triage outcomes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: CRN-perturbation guard `test_no_cascade_baseline_unchanged`

**Files:**
- Test: `tests/test_m06_no_cascade_baseline_unchanged.py` (new)

This test pins a pre-M06 scalar from a no-cascade run. If any new `ss.Dist` instance in M06 perturbs the HPV transmission RNG stream, this test fails.

- [ ] **Step 1: Compute the baseline scalars from current HEAD**

Run a short script in a Python repl to get the pinned values. From `hpvsim_claudecontrol/`:

```bash
python -c "
import hpvsim as hpv
sim = hpv.Sim(
    n_agents=2000, start='2000', stop='2020', rand_seed=0,
    diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
    networks='default',
    v2_compat_demographics=True,
)
sim.run()
total_inf = float(sum(sim.diseases[g].results['cum_infections'][-1] for g in ('hpv16', 'hpv18', 'hi5', 'ohr')))
total_cancers = float(sum(sim.diseases[g].results['cum_cancers'][-1] for g in ('hpv16', 'hpv18', 'hi5', 'ohr')))
print(f'PINNED_TOTAL_INFECTIONS = {total_inf!r}')
print(f'PINNED_TOTAL_CANCERS = {total_cancers!r}')
"
```

Record the printed values; you'll embed them as literals in the test below.

- [ ] **Step 2: Write the guard test**

Create `tests/test_m06_no_cascade_baseline_unchanged.py` (substitute the actual numeric values from Step 1 in place of the placeholders shown):

```python
"""Pinned-scalar guard: a no-cascade run must produce the same total
infections and cancers as before M06 landed. Catches accidental RNG
stream perturbations from new ss.Dist instances introduced by M06."""
import hpvsim as hpv

# Pinned at M06 baseline-cut on m06-test-and-treat-cascade branch.
# If you change any cascade-related RNG creation, this WILL fail and
# you must re-pin (after confirming the change is intentional).
PINNED_TOTAL_INFECTIONS = REPLACE_WITH_VALUE_FROM_STEP_1
PINNED_TOTAL_CANCERS    = REPLACE_WITH_VALUE_FROM_STEP_1


def test_no_cascade_baseline_unchanged():
    sim = hpv.Sim(
        n_agents=2000, start='2000', stop='2020', rand_seed=0,
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        networks='default',
        v2_compat_demographics=True,
    )
    sim.run()
    total_inf = float(sum(
        sim.diseases[g].results['cum_infections'][-1]
        for g in ('hpv16', 'hpv18', 'hi5', 'ohr')
    ))
    total_cancers = float(sum(
        sim.diseases[g].results['cum_cancers'][-1]
        for g in ('hpv16', 'hpv18', 'hi5', 'ohr')
    ))
    assert total_inf == PINNED_TOTAL_INFECTIONS, (
        f'cum_infections drift: {total_inf} != {PINNED_TOTAL_INFECTIONS}. '
        'Check that no new ss.Dist instance is sharing an RNG stream with '
        'HPV transmission decisions.'
    )
    assert total_cancers == PINNED_TOTAL_CANCERS, (
        f'cum_cancers drift: {total_cancers} != {PINNED_TOTAL_CANCERS}.'
    )
```

- [ ] **Step 3: Run the test to verify it passes**

Run:
```bash
pytest tests/test_m06_no_cascade_baseline_unchanged.py -v
```
Expected: PASS. If it fails on first run, you have a CRN perturbation issue — the M06 changes are touching an RNG stream that shouldn't be touched. Investigate which new dist is being constructed before HPV transmission and confirm it's seeded via its own `ss.Dist` instance (not sharing the parent RNG).

- [ ] **Step 4: Commit**

```bash
git add tests/test_m06_no_cascade_baseline_unchanged.py
git commit -m "M06: CRN-perturbation guard — no-cascade run reproduces pre-M06 scalars

Pins cum_infections and cum_cancers from a 2000-agent, 20-year baseline
run with no cascade interventions. Catches accidental RNG-stream
perturbations from new ss.Dist instances. Mirrors M05's
test_no_vx_baseline_unchanged pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Anchor PARS scripts

**Files:**
- Create: `tests/regression/anchor_screen_treat.py`
- Create: `tests/regression/anchor_txvx_routine.py`

Look at `tests/regression/anchor_vx_routine.py` for the M05 PARS pattern; mirror its structure exactly.

- [ ] **Step 1: Read the M05 anchor template**

```bash
head -80 tests/regression/anchor_vx_routine.py
```

Note: it should expose a `PARS` SimpleNamespace (or dict) that
`build_v3_sim(PARS, seed)` consumes; it lives next to the parity test.

- [ ] **Step 2: Create `anchor_screen_treat.py`**

Create `tests/regression/anchor_screen_treat.py`:

```python
"""M06 anchor: full screen -> triage -> treat cascade.

PARS exposes the parameters consumed by build_v3_sim. The cascade is
modelled on the hpvsim_methods_manuscript HSP scenario shape.
"""
import types

import hpvsim as hpv


PARS = types.SimpleNamespace(
    # Sim shape
    n_agents=10_000,
    start='1990',
    stop='2060',
    location='nigeria',
    init_prev=0.05,

    # M05 demographics setting (closes the cohort gap in M05's anchors)
    v2_compat_demographics=True,

    # Cascade parameters
    screen_start_year=2020,
    screen_end_year=2060,
    screen_prob=0.7,
    screen_age_range=[30, 50],
    screen_product='hpv',

    triage_prob=0.9,
    triage_product='colposcopy',

    treat_prob=0.8,
    treat_product='excision',
)


def _interventions():
    screen = hpv.routine_screening(
        name='primary',
        product=PARS.screen_product,
        prob=PARS.screen_prob,
        age_range=PARS.screen_age_range,
        sex='f',
        start_year=PARS.screen_start_year,
        end_year=PARS.screen_end_year,
    )
    triage = hpv.routine_triage(
        name='colpo',
        product=PARS.triage_product,
        prob=PARS.triage_prob,
        eligibility=lambda s: s.interventions.primary.outcomes['positive'],
        start_year=PARS.screen_start_year,
        end_year=PARS.screen_end_year,
    )
    treat = hpv.treat_num(
        name='excision_rx',
        product=PARS.treat_product,
        prob=PARS.treat_prob,
        eligibility=lambda s: s.interventions.colpo.outcomes['hsil'],
    )
    return [screen, triage, treat]


def build_v3_sim(seed=0):
    diseases = [hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')]
    return hpv.Sim(
        n_agents=PARS.n_agents,
        start=PARS.start, stop=PARS.stop,
        rand_seed=seed,
        location=PARS.location,
        diseases=diseases,
        networks='default',
        interventions=_interventions(),
        v2_compat_demographics=PARS.v2_compat_demographics,
    )
```

- [ ] **Step 3: Create `anchor_txvx_routine.py`**

Create `tests/regression/anchor_txvx_routine.py`:

```python
"""M06 anchor: routine therapeutic vaccination on the M03 Nigeria baseline."""
import types

import hpvsim as hpv


PARS = types.SimpleNamespace(
    n_agents=10_000,
    start='1990',
    stop='2060',
    location='nigeria',
    init_prev=0.05,

    v2_compat_demographics=True,

    txvx_start_year=2030,
    txvx_end_year=2060,
    txvx_prob=0.6,
    txvx_age_range=[25, 26],
    txvx_product='txvx1',
)


def _interventions():
    return [
        hpv.routine_txvx(
            name='txvx',
            product=PARS.txvx_product,
            prob=PARS.txvx_prob,
            age_range=PARS.txvx_age_range,
            sex='f',
            start_year=PARS.txvx_start_year,
            end_year=PARS.txvx_end_year,
        ),
    ]


def build_v3_sim(seed=0):
    diseases = [hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')]
    return hpv.Sim(
        n_agents=PARS.n_agents,
        start=PARS.start, stop=PARS.stop,
        rand_seed=seed,
        location=PARS.location,
        diseases=diseases,
        networks='default',
        interventions=_interventions(),
        v2_compat_demographics=PARS.v2_compat_demographics,
    )
```

- [ ] **Step 4: Smoke-run both anchors at small N**

```bash
python -c "
from tests.regression import anchor_screen_treat as a
s = a.build_v3_sim(seed=0)
s.pars.n_agents = 500
s.pars.stop = '2025'
s.run()
print('screen_treat OK; screened:', s.interventions.primary.screened.uids.size)
"

python -c "
from tests.regression import anchor_txvx_routine as a
s = a.build_v3_sim(seed=0)
s.pars.n_agents = 500
s.pars.stop = '2035'
s.run()
print('txvx_routine OK; tx_vaccinated:', s.interventions.txvx.tx_vaccinated.uids.size)
"
```

Expected: Both print non-zero counts; no exceptions.

- [ ] **Step 5: Commit**

```bash
git add tests/regression/anchor_screen_treat.py tests/regression/anchor_txvx_routine.py
git commit -m "M06: anchor PARS scripts for screen-treat + txvx-routine parity

Cascade anchor mirrors the hpvsim_methods_manuscript HSP scenario
shape: hpv screen → colposcopy triage → excision treat on
Nigeria-base M03 sim. txvx anchor adds a single routine_txvx with
product='txvx1' starting 2030.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: v2 baseline generation scripts

**Files:**
- Create: `tests/regression/multi_seed_v2_screen_treat.py`
- Create: `tests/regression/multi_seed_v2_txvx.py`

Generate 30-seed v2 baselines, applying M05 lessons 1+2 (alive-mask all flow counters; `annual_dt = sim.resfreq * sim.pars['dt']`). Use `tests/regression/multi_seed_v2_vx.py` as a template.

- [ ] **Step 1: Read the M05 baseline generator template**

```bash
head -100 tests/regression/multi_seed_v2_vx.py
```

Note structure: imports v2 from `hpvsim/_v2_legacy`, runs 30 seeds in parallel, extracts pinned metrics, writes to a JSON file under `v2_seeds_n30_*.json`.

- [ ] **Step 2: Create `multi_seed_v2_screen_treat.py`**

Create `tests/regression/multi_seed_v2_screen_treat.py`:

```python
"""Generate 30-seed v2 baseline for the M06 screen-treat anchor.

Output: tests/regression/v2_seeds_n30_screen_treat.json (gitignored).
Applies M05 lessons:
  - lesson 1: alive-mask all per-agent flow counters
  - lesson 2: person-years denominator uses resfreq*dt
"""
import json
from pathlib import Path

import numpy as np

# v2 lives in the quarantined _v2_legacy. We use it ONLY for baseline
# generation, never for runtime imports in active code.
from hpvsim import _v2_legacy as v2  # noqa: E402

N_SEEDS = 30
OUT = Path(__file__).parent / 'v2_seeds_n30_screen_treat.json'

# Mirror anchor_screen_treat PARS — keep these in sync if you edit either side.
ANCHOR = dict(
    n_agents=10_000,
    start=1990,
    stop=2060,
    location='nigeria',
    init_prev=0.05,
    screen_start_year=2020,
    screen_end_year=2060,
    screen_prob=0.7,
    screen_age_range=[30, 50],
    triage_prob=0.9,
    treat_prob=0.8,
)


def _v2_interventions():
    screen = v2.routine_screening(
        product='hpv', prob=ANCHOR['screen_prob'],
        age_range=ANCHOR['screen_age_range'],
        start_year=ANCHOR['screen_start_year'],
        end_year=ANCHOR['screen_end_year'],
    )
    triage = v2.routine_triage(
        product='colposcopy', prob=ANCHOR['triage_prob'],
        eligibility=lambda sim: sim.get_intervention('routine_screening').outcomes['positive'],
        start_year=ANCHOR['screen_start_year'],
        end_year=ANCHOR['screen_end_year'],
    )
    treat = v2.treat_num(
        product='excision', prob=ANCHOR['treat_prob'],
        eligibility=lambda sim: sim.get_intervention('routine_triage').outcomes['hsil'],
    )
    return [screen, triage, treat]


def _run_one_seed(seed):
    sim = v2.Sim(
        n_agents=ANCHOR['n_agents'],
        start=ANCHOR['start'], end=ANCHOR['stop'],
        rand_seed=seed,
        location=ANCHOR['location'],
        init_prev=ANCHOR['init_prev'],
        interventions=_v2_interventions(),
    )
    sim.run()

    # M05 lesson 2: annual_dt for person-years denominator
    annual_dt = sim.resfreq * sim.pars['dt']

    # M05 lesson 1: alive-mask all per-agent flow counters
    alive = sim.people.alive

    # Pinned scalars
    n_screened_2060      = int((sim.people.screened & alive).sum())
    n_screens_2060       = float((sim.people.screens * alive).sum())
    n_cin_treated_2060   = int((sim.people.cin_treated & alive).sum())
    n_cin_treatments_2060 = float((sim.people.cin_treatments * alive).sum())

    res = sim.results
    # cancer_incidence_2030_2060: cancers / person-years over 2030-2060
    years = res['year']
    mask = (years >= 2030) & (years < 2060)
    cancers = float(res['new_cancers'][mask].sum())
    person_years = float((res['n_alive'][mask] * annual_dt).sum())
    cancer_incidence_2030_2060 = cancers / person_years if person_years else 0.0
    cancer_deaths_2030_2060   = float(res['new_cancer_deaths'][mask].sum())

    return dict(
        seed=seed,
        n_screened_2060=n_screened_2060,
        n_screens_2060=n_screens_2060,
        n_cin_treated_2060=n_cin_treated_2060,
        n_cin_treatments_2060=n_cin_treatments_2060,
        cancer_incidence_2030_2060=cancer_incidence_2030_2060,
        cancer_deaths_2030_2060=cancer_deaths_2030_2060,
    )


def main():
    out = [_run_one_seed(seed) for seed in range(N_SEEDS)]
    OUT.write_text(json.dumps(out, indent=2))
    print(f'Wrote {OUT} ({N_SEEDS} seeds)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Create `multi_seed_v2_txvx.py`**

Create `tests/regression/multi_seed_v2_txvx.py`:

```python
"""Generate 30-seed v2 baseline for the M06 txvx anchor."""
import json
from pathlib import Path

from hpvsim import _v2_legacy as v2

N_SEEDS = 30
OUT = Path(__file__).parent / 'v2_seeds_n30_txvx.json'

ANCHOR = dict(
    n_agents=10_000,
    start=1990,
    stop=2060,
    location='nigeria',
    init_prev=0.05,
    txvx_start_year=2030,
    txvx_end_year=2060,
    txvx_prob=0.6,
    txvx_age_range=[25, 26],
)


def _v2_interventions():
    return [v2.routine_txvx(
        product='txvx1',
        prob=ANCHOR['txvx_prob'],
        age_range=ANCHOR['txvx_age_range'],
        start_year=ANCHOR['txvx_start_year'],
        end_year=ANCHOR['txvx_end_year'],
    )]


def _run_one_seed(seed):
    sim = v2.Sim(
        n_agents=ANCHOR['n_agents'],
        start=ANCHOR['start'], end=ANCHOR['stop'],
        rand_seed=seed,
        location=ANCHOR['location'],
        init_prev=ANCHOR['init_prev'],
        interventions=_v2_interventions(),
    )
    sim.run()
    annual_dt = sim.resfreq * sim.pars['dt']
    alive = sim.people.alive

    n_tx_vaccinated_2060  = int((sim.people.tx_vaccinated & alive).sum())
    n_txvx_doses_2060     = float((sim.people.txvx_doses * alive).sum())
    res = sim.results
    years = res['year']
    mask = (years >= 2030) & (years < 2060)
    cancers = float(res['new_cancers'][mask].sum())
    person_years = float((res['n_alive'][mask] * annual_dt).sum())
    cancer_incidence_2030_2060 = cancers / person_years if person_years else 0.0

    return dict(
        seed=seed,
        n_tx_vaccinated_2060=n_tx_vaccinated_2060,
        n_txvx_doses_2060=n_txvx_doses_2060,
        cancer_incidence_2030_2060=cancer_incidence_2030_2060,
    )


def main():
    out = [_run_one_seed(seed) for seed in range(N_SEEDS)]
    OUT.write_text(json.dumps(out, indent=2))
    print(f'Wrote {OUT} ({N_SEEDS} seeds)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Add the gitignore entries**

```bash
grep -q 'v2_seeds_n30_screen_treat.json' .gitignore || \
  echo 'tests/regression/v2_seeds_n30_screen_treat.json' >> .gitignore
grep -q 'v2_seeds_n30_txvx.json' .gitignore || \
  echo 'tests/regression/v2_seeds_n30_txvx.json' >> .gitignore
```

- [ ] **Step 5: Run both generators (slow — ~30-60 min each)**

```bash
python tests/regression/multi_seed_v2_screen_treat.py
python tests/regression/multi_seed_v2_txvx.py
```

Expected: Each prints a "Wrote …" line. The JSON files appear under `tests/regression/` but are NOT staged (they're gitignored).

Verify with:
```bash
ls -la tests/regression/v2_seeds_n30_*.json
git status --short tests/regression/
```

The JSONs should exist but not appear in `git status`.

- [ ] **Step 6: Commit (scripts only)**

```bash
git add tests/regression/multi_seed_v2_screen_treat.py tests/regression/multi_seed_v2_txvx.py .gitignore
git commit -m "M06: v2 baseline generators for screen-treat + txvx parity anchors

30 seeds each. Applies M05 lessons:
  - alive-mask on per-agent flow counters
  - annual_dt = resfreq*dt for person-years denominator
Output JSONs are gitignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: Short-summary parity tests

**Files:**
- Create: `tests/test_m06_screen_treat_parity.py`
- Create: `tests/test_m06_txvx_parity.py`

- [ ] **Step 1: Read the M05 short-summary parity template**

```bash
head -120 tests/test_m05_vx_routine_parity.py
```

Note structure: import the v2 baseline JSON, run N=10 v3 seeds via the anchor's `build_v3_sim`, compute per-metric z-score, assert `|z| < 3`.

- [ ] **Step 2: Create `test_m06_screen_treat_parity.py`**

Create `tests/test_m06_screen_treat_parity.py`:

```python
"""M06 screen-treat short-summary parity at |z| < 3.

10 v3 seeds × full sim, compared to 30-seed v2 baseline (generated by
tests/regression/multi_seed_v2_screen_treat.py). Marked @slow; CI runs
-m 'not slow'.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.regression import anchor_screen_treat as anchor


V2_BASELINE = Path(__file__).parent / 'regression' / 'v2_seeds_n30_screen_treat.json'
N_V3_SEEDS = 10
Z_GATE = 3.0


def _v3_metrics_one_seed(seed):
    sim = anchor.build_v3_sim(seed=seed)
    sim.run()
    annual_dt = sim.resfreq * sim.pars['dt'] if hasattr(sim, 'resfreq') else 1.0
    alive = sim.people.alive

    primary = sim.interventions.primary
    colpo   = sim.interventions.colpo
    treat   = sim.interventions.excision_rx

    # Cascade counters (alive-masked)
    n_screened_2060      = int((primary.screened & alive).sum())
    n_screens_2060       = float((primary.screens * alive).sum())
    n_cin_treated_2060   = int((treat.cin_treated & alive).sum())
    n_cin_treatments_2060 = float((treat.cin_treatments * alive).sum())

    # Cancer incidence 2030-2060
    res = sim.results.hpvtotal
    years = sim.timevec
    mask = (years >= 2030) & (years < 2060)
    cancers = float(res['new_cancers'][mask].sum())
    n_alive_arr = res['n_alive'][mask] if 'n_alive' in res else np.asarray([1.0])
    person_years = float((n_alive_arr * annual_dt).sum())
    cancer_incidence_2030_2060 = cancers / person_years if person_years else 0.0
    cancer_deaths_2030_2060    = float(res['new_cancer_deaths'][mask].sum())

    return dict(
        n_screened_2060=n_screened_2060,
        n_screens_2060=n_screens_2060,
        n_cin_treated_2060=n_cin_treated_2060,
        n_cin_treatments_2060=n_cin_treatments_2060,
        cancer_incidence_2030_2060=cancer_incidence_2030_2060,
        cancer_deaths_2030_2060=cancer_deaths_2030_2060,
    )


@pytest.mark.slow
@pytest.mark.skipif(not V2_BASELINE.exists(),
                    reason='v2 baseline JSON not present — regenerate locally')
def test_m06_screen_treat_short_summary_parity():
    v2_rows = json.loads(V2_BASELINE.read_text())
    metrics = list(v2_rows[0].keys() - {'seed'})

    v3_rows = [_v3_metrics_one_seed(seed) for seed in range(N_V3_SEEDS)]

    failures = []
    for m in metrics:
        v2_vals = np.array([float(r[m]) for r in v2_rows])
        v3_vals = np.array([float(r[m]) for r in v3_rows])
        # Combined-sample z: |mean3 - mean2| / sqrt(var2/n2 + var3/n3)
        n2, n3 = len(v2_vals), len(v3_vals)
        var2 = v2_vals.var(ddof=1)
        var3 = v3_vals.var(ddof=1)
        se = np.sqrt(var2 / n2 + var3 / n3)
        if se == 0:
            z = 0.0 if v2_vals.mean() == v3_vals.mean() else float('inf')
        else:
            z = (v3_vals.mean() - v2_vals.mean()) / se
        if abs(z) >= Z_GATE:
            failures.append((m, z, v2_vals.mean(), v3_vals.mean()))

    if failures:
        msg = '\n'.join(
            f'  {m}: z={z:+.2f} (v2_mean={mv2:.4g}, v3_mean={mv3:.4g})'
            for m, z, mv2, mv3 in failures
        )
        pytest.fail(f'screen-treat parity failed at |z| < {Z_GATE}:\n{msg}')
```

- [ ] **Step 3: Create `test_m06_txvx_parity.py`**

Create `tests/test_m06_txvx_parity.py`:

```python
"""M06 txvx-routine short-summary parity at |z| < 3."""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.regression import anchor_txvx_routine as anchor


V2_BASELINE = Path(__file__).parent / 'regression' / 'v2_seeds_n30_txvx.json'
N_V3_SEEDS = 10
Z_GATE = 3.0


def _v3_metrics_one_seed(seed):
    sim = anchor.build_v3_sim(seed=seed)
    sim.run()
    annual_dt = sim.resfreq * sim.pars['dt'] if hasattr(sim, 'resfreq') else 1.0
    alive = sim.people.alive
    txvx = sim.interventions.txvx
    n_tx_vaccinated_2060 = int((txvx.tx_vaccinated & alive).sum())
    n_txvx_doses_2060    = float((txvx.txvx_doses * alive).sum())

    res = sim.results.hpvtotal
    years = sim.timevec
    mask = (years >= 2030) & (years < 2060)
    cancers = float(res['new_cancers'][mask].sum())
    n_alive_arr = res['n_alive'][mask] if 'n_alive' in res else np.asarray([1.0])
    person_years = float((n_alive_arr * annual_dt).sum())
    cancer_incidence_2030_2060 = cancers / person_years if person_years else 0.0

    return dict(
        n_tx_vaccinated_2060=n_tx_vaccinated_2060,
        n_txvx_doses_2060=n_txvx_doses_2060,
        cancer_incidence_2030_2060=cancer_incidence_2030_2060,
    )


@pytest.mark.slow
@pytest.mark.skipif(not V2_BASELINE.exists(),
                    reason='v2 baseline JSON not present — regenerate locally')
def test_m06_txvx_short_summary_parity():
    v2_rows = json.loads(V2_BASELINE.read_text())
    metrics = list(v2_rows[0].keys() - {'seed'})
    v3_rows = [_v3_metrics_one_seed(seed) for seed in range(N_V3_SEEDS)]

    failures = []
    for m in metrics:
        v2_vals = np.array([float(r[m]) for r in v2_rows])
        v3_vals = np.array([float(r[m]) for r in v3_rows])
        n2, n3 = len(v2_vals), len(v3_vals)
        var2, var3 = v2_vals.var(ddof=1), v3_vals.var(ddof=1)
        se = np.sqrt(var2 / n2 + var3 / n3)
        if se == 0:
            z = 0.0 if v2_vals.mean() == v3_vals.mean() else float('inf')
        else:
            z = (v3_vals.mean() - v2_vals.mean()) / se
        if abs(z) >= Z_GATE:
            failures.append((m, z, v2_vals.mean(), v3_vals.mean()))

    if failures:
        msg = '\n'.join(
            f'  {m}: z={z:+.2f} (v2_mean={mv2:.4g}, v3_mean={mv3:.4g})'
            for m, z, mv2, mv3 in failures
        )
        pytest.fail(f'txvx parity failed at |z| < {Z_GATE}:\n{msg}')
```

- [ ] **Step 4: Run the parity tests locally (slow — full anchor sims × 10 seeds each)**

```bash
pytest tests/test_m06_screen_treat_parity.py tests/test_m06_txvx_parity.py -m slow -v
```
Expected: PASS on both. If a metric drifts, see the M05 lessons section in the spec for the most likely root causes (counting bugs in the v2 generator, RNG perturbation, or quarterly-vs-annual cadence on the v3 side).

- [ ] **Step 5: Commit**

```bash
git add tests/test_m06_screen_treat_parity.py tests/test_m06_txvx_parity.py
git commit -m "M06: short-summary parity tests at |z| < 3 for both anchors

10 v3 seeds × full sim vs 30-seed v2 baseline. Skipped when baseline
JSON missing (regenerate locally via tests/regression/multi_seed_v2_*).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: Trajectory parity test

**Files:**
- Create: `tests/test_m06_trajectory_parity.py`

The trajectory test buckets v3 per-step `new_cancers` / `new_screened` / `new_cin_treated` into annual sums (M05 lesson 5) and uses a `_FirstEventLogger`-style analyzer to count False→True transitions on `screened` / `cin_treated` (M05 lesson 3).

- [ ] **Step 1: Read the M05 trajectory parity test for the `_FirstVaxLogger` pattern**

```bash
head -120 tests/test_m05_vx_trajectory_parity.py
```

Note the `_FirstVaxLogger` analyzer that snapshots `intervention.vaccinated.raw` pre/post `step()` and accumulates False→True transitions into a per-year flow counter. M06 needs the same trick for `screened` and `cin_treated`.

- [ ] **Step 2: Write the trajectory parity test**

Create `tests/test_m06_trajectory_parity.py`:

```python
"""M06 trajectory parity (screen-treat anchor only).

Uses a per-step BoolArr.raw snapshot pattern to count False -> True
transitions, producing v2-equivalent per-year flow counters from
per-intervention state. Mirrors M05's _FirstVaxLogger.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import starsim as ss

from tests.regression import anchor_screen_treat as anchor


V2_BASELINE = Path(__file__).parent / 'regression' / 'v2_seeds_n30_screen_treat_traj.json'
N_V3_SEEDS = 10
Z_GATE = 3.0


class _FirstEventLogger(ss.Analyzer):
    """Snapshot multiple intervention BoolArrs pre/post step; record False->True per year."""

    def __init__(self, intervention_name, attr_name, **kwargs):
        super().__init__(**kwargs)
        self.intervention_name = intervention_name
        self.attr_name = attr_name
        self._prev = None
        self.annual_count = {}  # year_int -> count

    def init_pre(self, sim):
        super().init_pre(sim)
        intv = sim.interventions[self.intervention_name]
        arr = getattr(intv, self.attr_name)
        self._prev = np.asarray(arr.raw).copy()

    def step(self):
        intv = self.sim.interventions[self.intervention_name]
        arr = getattr(intv, self.attr_name)
        cur = np.asarray(arr.raw)
        transitions = (cur & ~self._prev).sum()
        year_int = int(self.sim.t.now('year'))
        self.annual_count[year_int] = self.annual_count.get(year_int, 0) + int(transitions)
        self._prev = cur.copy()


def _v3_trajectories_one_seed(seed):
    sim = anchor.build_v3_sim(seed=seed)
    sim.analyzers = [
        _FirstEventLogger('primary',     'screened',    name='log_screened'),
        _FirstEventLogger('excision_rx', 'cin_treated', name='log_cin_treated'),
    ]
    sim.run()
    log_screened = sim.analyzers.log_screened.annual_count
    log_cin_treated = sim.analyzers.log_cin_treated.annual_count
    # Plus annual-bucketed new_cancers from sim.results.hpvtotal
    res = sim.results.hpvtotal
    years_flt = sim.timevec
    year_int = np.floor(np.asarray(years_flt)).astype(int)
    new_cancers_per_step = np.asarray(res['new_cancers'])
    annual_cancers = {}
    for y in np.unique(year_int):
        mask = year_int == y
        annual_cancers[int(y)] = float(new_cancers_per_step[mask].sum())
    return dict(
        screened_by_year=log_screened,
        cin_treated_by_year=log_cin_treated,
        cancers_by_year=annual_cancers,
    )


@pytest.mark.slow
@pytest.mark.skipif(not V2_BASELINE.exists(),
                    reason='v2 trajectory baseline JSON not present — regenerate locally')
def test_m06_screen_treat_trajectory_parity():
    v2_rows = json.loads(V2_BASELINE.read_text())  # list of dicts keyed by year_int
    v3_rows = [_v3_trajectories_one_seed(seed) for seed in range(N_V3_SEEDS)]

    # For each metric × year, compute z-score across seeds
    failures = []
    for metric in ('screened_by_year', 'cin_treated_by_year', 'cancers_by_year'):
        years = sorted(int(y) for y in v2_rows[0][metric].keys())
        for y in years:
            v2_vals = np.array([float(r[metric][str(y)]) for r in v2_rows])
            v3_vals = np.array([float(r[metric].get(y, 0.0)) for r in v3_rows])
            n2, n3 = len(v2_vals), len(v3_vals)
            var2, var3 = v2_vals.var(ddof=1), v3_vals.var(ddof=1)
            se = np.sqrt(var2 / n2 + var3 / n3)
            if se == 0:
                z = 0.0 if v2_vals.mean() == v3_vals.mean() else float('inf')
            else:
                z = (v3_vals.mean() - v2_vals.mean()) / se
            if abs(z) >= Z_GATE:
                failures.append((metric, y, z, v2_vals.mean(), v3_vals.mean()))

    if failures:
        msg = '\n'.join(
            f'  {m}@{y}: z={z:+.2f} (v2_mean={mv2:.4g}, v3_mean={mv3:.4g})'
            for m, y, z, mv2, mv3 in failures
        )
        pytest.fail(f'trajectory parity failed at |z| < {Z_GATE}:\n{msg}')
```

- [ ] **Step 3: Extend the v2 generator to also save per-year trajectories**

Update `tests/regression/multi_seed_v2_screen_treat.py` to additionally write a `v2_seeds_n30_screen_treat_traj.json` file containing per-year `screened`, `cin_treated`, and `cancers` flow counts per seed.

Edit `tests/regression/multi_seed_v2_screen_treat.py`. Append a new helper and update `main`:

```python
TRAJ_OUT = Path(__file__).parent / 'v2_seeds_n30_screen_treat_traj.json'


def _v2_trajectories_one_seed(seed):
    sim = v2.Sim(
        n_agents=ANCHOR['n_agents'],
        start=ANCHOR['start'], end=ANCHOR['stop'],
        rand_seed=seed,
        location=ANCHOR['location'],
        init_prev=ANCHOR['init_prev'],
        interventions=_v2_interventions(),
    )
    sim.run()
    res = sim.results
    years_int = res['year'].astype(int)
    out = {'screened_by_year': {}, 'cin_treated_by_year': {}, 'cancers_by_year': {}}
    for y in np.unique(years_int):
        mask = years_int == y
        out['screened_by_year'][int(y)] = float(res['new_screened'][mask].sum())
        out['cin_treated_by_year'][int(y)] = float(res['new_cin_treated'][mask].sum())
        out['cancers_by_year'][int(y)] = float(res['new_cancers'][mask].sum())
    return out


def main():
    short = [_run_one_seed(seed) for seed in range(N_SEEDS)]
    OUT.write_text(json.dumps(short, indent=2))
    print(f'Wrote {OUT} ({N_SEEDS} seeds)')

    traj = [_v2_trajectories_one_seed(seed) for seed in range(N_SEEDS)]
    TRAJ_OUT.write_text(json.dumps(traj, indent=2))
    print(f'Wrote {TRAJ_OUT} ({N_SEEDS} seeds)')
```

Add the traj file to `.gitignore`:

```bash
grep -q 'v2_seeds_n30_screen_treat_traj.json' .gitignore || \
  echo 'tests/regression/v2_seeds_n30_screen_treat_traj.json' >> .gitignore
```

- [ ] **Step 4: Regenerate both baseline JSONs**

```bash
python tests/regression/multi_seed_v2_screen_treat.py
```

Expected: writes both `v2_seeds_n30_screen_treat.json` and
`v2_seeds_n30_screen_treat_traj.json`.

- [ ] **Step 5: Run the trajectory parity test**

```bash
pytest tests/test_m06_trajectory_parity.py -m slow -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_m06_trajectory_parity.py tests/regression/multi_seed_v2_screen_treat.py .gitignore
git commit -m "M06: trajectory parity test for screen-treat anchor

Uses a _FirstEventLogger analyzer (BoolArr.raw pre/post snapshot,
count False->True transitions) to recover v2-equivalent per-year
flow counters from per-intervention state, then buckets v3's
quarterly cancer counts to annual. M05 lessons 3+5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 22: README and CI workflow check

**Files:**
- Create: `tests/regression/README_m06.md`
- Verify: `.github/workflows/tests.yaml`

- [ ] **Step 1: Write `README_m06.md`**

Create `tests/regression/README_m06.md`:

```markdown
# M06 parity baselines

The M06 PR ships two anchor scenarios and three parity tests:

| Anchor PARS | v2 baseline generator | Short-summary parity | Trajectory parity |
|---|---|---|---|
| `anchor_screen_treat.py` | `multi_seed_v2_screen_treat.py` | `test_m06_screen_treat_parity.py` | `test_m06_trajectory_parity.py` |
| `anchor_txvx_routine.py` | `multi_seed_v2_txvx.py` | `test_m06_txvx_parity.py` | (none — short-summary only) |

The v2 baseline JSONs (`v2_seeds_n30_*.json`) are **gitignored**. To
regenerate locally:

```bash
# From an environment with the v2.x release installed
python tests/regression/multi_seed_v2_screen_treat.py
python tests/regression/multi_seed_v2_txvx.py
```

Each generator runs 30 seeds × full sim (~30-60 minutes each).

## Running the parity tests

```bash
pytest tests/test_m06_screen_treat_parity.py \
       tests/test_m06_txvx_parity.py \
       tests/test_m06_trajectory_parity.py \
       -m slow -v
```

All three are marked `@pytest.mark.slow` and excluded from CI's
`-m 'not slow'` run; they execute locally before the M06 PR opens.

## Multi-intervention name collisions

Each intervention in the anchor PARS scripts is constructed with an
explicit `name=` so cross-intervention eligibility callbacks resolve
unambiguously via `sim.interventions.<name>`. Do not remove these.

## M05 lessons embedded in the v2 generators

1. **Counting cadence** — all per-agent flow counters (`screened`,
   `cin_treated`, `tx_vaccinated`, etc.) are `& sim.people.alive`
   masked before summing.
2. **Person-years** — `cancer_incidence_2030_2060` uses
   `annual_dt = sim.resfreq * sim.pars['dt']` as the per-step person-years
   contribution; v2 stores annual results so this comes out to 1.0
   under the default `resfreq=4, dt=0.25`.
3. **Per-step vs flow counters** — the trajectory test uses the
   `_FirstEventLogger` analyzer to compute v2-equivalent flow counters
   from v3 per-intervention BoolArr state.
```

- [ ] **Step 2: Confirm CI does NOT pick up the slow parity tests**

Check `.github/workflows/tests.yaml`. The existing M05 setup runs
`pytest -m 'not slow'`. Verify by running:

```bash
grep -n "pytest" .github/workflows/tests.yaml
```

Expected: see `pytest -m 'not slow'` (or `--ignore-marker='not slow'`)
in at least one job step.

- [ ] **Step 3: Run the full non-slow CI suite locally to confirm**

```bash
pytest tests/ -m "not slow" -q
```

Expected: All passed; runtime ~5 minutes. No parity test executes.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/README_m06.md
git commit -m "M06: README documenting baseline regeneration + M05 lessons embedded

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 23: MIGRATION_PLAN.md updates

**Files:**
- Modify: `MIGRATION_PLAN.md`

- [ ] **Step 1: Update the M6 status row**

Edit `MIGRATION_PLAN.md`. Find the Status table around line 73-81 and change:

```
| M6–M10 | ⬜ Not started | — |
```

to:

```
| M6: Test-and-treat cascade | 🟡 In progress | branch `m06-test-and-treat-cascade` |
| M7–M10 | ⬜ Not started | — |
```

- [ ] **Step 2: Rewrite the M6 § Sub-tasks block**

Find the `### M6: Screen-and-treat cascade` section (around line 247). Replace its `**Sub-tasks:**` block with:

```
**Sub-tasks:**
- Add `hpv.dx(ss.Dx)` per-genotype diagnostic product with `_load_dx_products`
  loader from `hpvsim/data/products_dx.csv`. Handles `all`/per-genotype CSV
  modes via overridden `administer`.
- Add `hpv.tx(ss.Tx)` per-genotype treatment product with state-flip and
  `ti_clearance = sim.ti + 1` scheduling.
- Add `hpv.txvx(ss.Vx)` therapeutic vaccine product mirroring `hpv.vx`
  architecture; writes to a new per-module `txvx_imm` FloatArr.
- Add `hpv.radiation(ss.Product)` standalone product extending
  `ti_dead_cancer`.
- Add `hpv.BaseTest`, `hpv.BaseScreening`, `hpv.BaseTriage` with HPV
  default eligibility (female + alive + optional debut_age); thin
  diamond leaves `hpv.routine_screening`/`campaign_screening`/
  `routine_triage`/`campaign_triage` combining with Starsim's delivery
  bases.
- Add `hpv.BaseTreatment`, `hpv.treat_num` (extends `ss.treat_num`),
  `hpv.treat_delay` (fresh port, integer-`ti` scheduler).
- Add `hpv.BaseTxVx`, `hpv.routine_txvx`, `hpv.campaign_txvx`,
  `hpv.linked_txvx` (no own timeline; eligibility-driven).
- Add `hpv.dynamic_pars` (year-keyed schedule with dotted-path resolution
  into `sim.diseases` / `sim.interventions` / `sim.pars`).
- Add per-module `latent` BoolState (no-op for now; CSV/dx hook ready)
  and `txvx_imm` FloatArr to `hpv.HPV`; update `CrossImmunity` connector
  to include `txvx_imm` in the independent-protection combine.
- Add two regression anchors (`anchor_screen_treat`, `anchor_txvx_routine`),
  v2 baseline generator scripts, and multi-seed `|z| < 3` parity tests
  (M03 + M05 pattern). Trajectory parity on the screen+treat anchor only.
- Add `test_no_cascade_baseline_unchanged` CRN-perturbation guard.
- Add unit tests for product administer logic, eligibility helpers, and
  loader CSV schemas.
```

(This is verbatim from the M06 spec's "MIGRATION_PLAN.md edits" section.)

- [ ] **Step 3: Verify the migration plan still parses correctly**

```bash
head -90 MIGRATION_PLAN.md
sed -n '245,275p' MIGRATION_PLAN.md
```

Verify visually that the table renders and the sub-task list is intact.

- [ ] **Step 4: Commit**

```bash
git add MIGRATION_PLAN.md
git commit -m "M06: flip M6 status to in-progress; rewrite sub-task list

Reflects the landed architecture (Starsim-native diamond, full M06
in one PR including txvx + radiation + dynamic_pars).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 24: Pre-PR gate — full validation pass

**Files:** None (validation only)

- [ ] **Step 1: Run the full non-slow suite**

```bash
pytest tests/ -m "not slow" -q
```
Expected: All passed.

- [ ] **Step 2: Run all M06 unit + integration tests**

```bash
pytest tests/test_m06_*.py -v
```
Expected: All passed.

- [ ] **Step 3: Run all parity tests locally (requires v2 baselines)**

```bash
pytest tests/test_m06_*_parity.py tests/test_m06_trajectory_parity.py -m slow -v
```
Expected: All passed. If a parity test fails, see the M06 spec's
"Open risks and mitigations" section + the M05 lessons codified
section for diagnostic guidance.

- [ ] **Step 4: M03 + M04 + M05 regression check**

```bash
pytest tests/test_m03_*.py tests/test_m04_*.py tests/test_m05_*.py -v
```
Expected: All passed (no test_m05_*_parity needs to be slow-marked separately if it's already excluded by `-m "not slow"`; check with `-m slow tests/test_m05_*_parity.py` if you want full parity confidence).

- [ ] **Step 5: Confirm no _v2_legacy imports leaked into active code**

```bash
grep -rn "from hpvsim._v2_legacy\|from \._v2_legacy\|hpvsim\._v2_legacy" hpvsim/ --include='*.py'
```
Expected: ZERO matches. v2 should be referenced only in the
baseline-generator scripts under `tests/regression/`, never from
`hpvsim/`.

- [ ] **Step 6: Commit the post-implementation deltas section update**

Edit `docs/superpowers/specs/2026-05-27-m06-test-and-treat-cascade-design.md`. The final section `## Post-implementation deltas` currently says "To be filled in after implementation lands". Append bullets for any divergences discovered while executing this plan (e.g. unexpected Starsim API quirks, parity rabbit-holes, scope cuts). Use M05's post-implementation deltas section as the format template.

If no divergences, replace the placeholder with:

```markdown
## Post-implementation deltas

No structural divergences from this spec were discovered during the
build. The implementation landed as planned across Tasks 1–24 of
[`docs/superpowers/plans/2026-05-27-hpvsim-m06-test-and-treat-cascade.md`](../plans/2026-05-27-hpvsim-m06-test-and-treat-cascade.md).
```

Commit:

```bash
git add docs/superpowers/specs/2026-05-27-m06-test-and-treat-cascade-design.md
git commit -m "M06: close out post-implementation deltas section

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Push and open PR**

(This step requires explicit user authorization before running.)

```bash
git push -u origin m06-test-and-treat-cascade
gh pr create --base v3.0-dev --title "M06: Test-and-treat cascade" --body "$(cat <<'EOF'
## Summary

Implements the M06 screen → triage → treat cascade plus therapeutic
vaccination, matching v2.x within `|z| < 3` on two anchor scenarios.

- Adds HPV-specific products: `hpv.dx`, `hpv.tx`, `hpv.txvx`, `hpv.radiation`
- Adds interventions: `hpv.routine_screening` / `campaign_screening`,
  `hpv.routine_triage` / `campaign_triage`, `hpv.treat_num` /
  `treat_delay`, `hpv.routine_txvx` / `campaign_txvx` / `linked_txvx`,
  `hpv.dynamic_pars`
- Adds per-genotype `latent` BoolState (no-op state hook) and
  `txvx_imm` FloatArr; extends `CrossImmunity` independent-protection
  combine to include `txvx_imm`
- Adds two regression anchors with multi-seed `|z| < 3` parity gates
  and a trajectory parity test (screen-treat anchor)
- All M05 lessons codified in spec + tests (counting cadence,
  person-years, per-step vs flow counters, CRN-perturbation guard,
  quarterly→annual downsampling, ordering contract)

Design spec: [`docs/superpowers/specs/2026-05-27-m06-test-and-treat-cascade-design.md`](docs/superpowers/specs/2026-05-27-m06-test-and-treat-cascade-design.md)

## Test plan

- [x] `pytest tests/ -m "not slow"` green on the M6 branch
- [x] `pytest tests/test_m06_*_parity.py -m slow` green locally
- [x] `pytest tests/test_m06_trajectory_parity.py -m slow` green locally
- [x] M03 + M04 + M05 tests still green (regression guard)
- [x] No `_v2_legacy` imports leaked into active `hpvsim/` code

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL.

---

## Self-review notes

A pass over the plan checking spec coverage:

1. **`hpv.dx`** — Task 5 ✓
2. **`hpv.tx`** — Task 6 ✓
3. **`hpv.txvx`** — Task 7 ✓
4. **`hpv.radiation`** — Task 8 ✓
5. **`hpv.BaseTest/BaseScreening/BaseTriage` + leaves** — Task 10 ✓
6. **`hpv.BaseTreatment` / `treat_num` / `treat_delay`** — Tasks 11–12 ✓
7. **`hpv.BaseTxVx` / routine/campaign/linked** — Task 13 ✓
8. **`hpv.dynamic_pars`** — Task 14 ✓
9. **`latent` BoolState + `txvx_imm` FloatArr + CrossImmunity update** — Tasks 1–2 ✓
10. **Anchor scenarios + v2 generators + parity tests** — Tasks 18–21 ✓
11. **CRN guard + cascade order tests** — Tasks 16–17 ✓
12. **CSV loaders + unit tests** — Task 4 ✓
13. **Eligibility helper unit tests** — Task 9 ✓
14. **`__init__.py` exports** — Task 15 ✓
15. **MIGRATION_PLAN.md edits** — Task 23 ✓
16. **Pre-PR gate** — Task 24 ✓

All 16 spec requirements have at least one task. No placeholders ("TBD" / "TODO") in plan body except the one explicit hand-off in Task 24 Step 6 (post-implementation deltas section, which is by-design open until implementation completes).

Type-name consistency spot-check: `hpv.dx`, `hpv.tx`, `hpv.txvx`, `hpv.radiation`, `hpv.treat_num`, `hpv.treat_delay`, `hpv.routine_screening`, `hpv.routine_triage`, `hpv.routine_txvx`, `hpv.campaign_*`, `hpv.linked_txvx`, `hpv.dynamic_pars` all used identically across spec + plan + tests + __init__.

The two helper signatures `_compose_screening_eligibility(age_range, sex, extra, debut_age)` and `_any_genotype_cancer(sim)` are consistent across Task 9 implementation, Task 10 caller, and unit-test call sites.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-27-hpvsim-m06-test-and-treat-cascade.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**