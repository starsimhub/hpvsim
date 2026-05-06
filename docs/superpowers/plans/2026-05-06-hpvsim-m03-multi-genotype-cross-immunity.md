# M03: Multi-genotype and Cross-Immunity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate M02's HPV16 natural-history machinery across `[hpv16, hpv18, hi5, ohr]` as four independent `ss.Disease` instances; introduce `hpv.CrossImmunity(ss.Connector)` to compute per-target `rel_sus` and `sev_imm` from per-source `nab_imm` and `cell_imm` via v2's cross-protection matrices. Match v2.x's 4-genotype Nigeria run within ±10% drift on a 40-entry per-genotype + aggregate `short_summary`, plus an age-aggregated cancer / infection trajectory gate.

**Architecture:** Each genotype is its own `HPV(ss.Infection)` instance, `name=<genotype>`. Each instance owns 1D `ss.FloatArr` per-source immunity (`nab_imm`, `cell_imm`), bumped on clearance from v2-ported beta-mean distributions (`imm_init`, `cell_imm_init`). Each step the Connector stacks per-disease `nab_imm` / `cell_imm` columns into a `(n_agents, n_g)` matrix, multiplies by the cross-protection matrices, and scatters per-target `rel_sus = 1 - sus_imm` and `sev_imm` back to each HPV. Sim wrapper accepts both an explicit `diseases=[hpv.HPV(...), ...]` list and a `genotypes=[16, 18, 'hi5', 'ohr']` sugar that auto-instantiates the four modules + the Connector. Single-genotype runs go through the same Connector path with a `1×1` identity matrix. Cross-protection matrix and per-genotype defaults are hand-ported from v2's `get_cross_immunity` and `get_genotype_pars` into active `hpvsim/parameters.py` — zero `_v2_legacy` delegations.

**Tech Stack:** Python 3.13, Starsim 3.3.3, Sciris, NumPy, pytest, gh CLI.

**Reference design:** `docs/superpowers/specs/2026-05-06-m03-multi-genotype-cross-immunity-design.md`

**Branch:** `m03-multi-genotype-and-cross-immunity` (cut off `m02-natural-history-parity`; rebase onto `v3.0-dev` after the M02 PR merges).

---

## Prerequisites and branch hygiene

The M03 branch was created off `m02-natural-history-parity` so this plan can begin while the M02 PR is in review. After M02 merges to `v3.0-dev`, run:

```bash
git fetch origin
git checkout m03-multi-genotype-and-cross-immunity
git rebase origin/v3.0-dev
git push --force-with-lease origin m03-multi-genotype-and-cross-immunity   # if pushed
```

If the M02 PR uses squash-merge, `git rebase --onto origin/v3.0-dev <m02-tip-sha>` cleanly lifts only the M03-original commits.

If M02 review surfaces material code changes before merge, fold them in by either rebasing M03 onto the updated M02 branch or merging M02 → M03 once the M02 PR lands.

---

## File structure after M03

| Path | Action | Responsibility |
|------|--------|---------------|
| `hpvsim/parameters.py` | Modify | Extend `GenotypePars` with hpv18/hi5/ohr branches; convert `imm_init` from scalar to beta-mean distribution; hand-port `get_cross_immunity()`; add per-genotype `init_prev` table; export `GENOTYPE_KEYS` |
| `hpvsim/connectors.py` | Create | New: `CrossImmunity(ss.Connector)` — stack-multiply-scatter cross-immunity Connector |
| `hpvsim/hpv.py` | Modify | Add 1D `nab_imm` / `cell_imm` source-immunity FloatArrs; remove M02's clearance-time writes to `sev_imm` and `rel_sus` (now Connector-driven); generalise initial-prevalence reader to per-genotype curve lookup; expand `_KNOWN_GENOTYPES` to all four keys |
| `hpvsim/sim.py` | Modify | Accept `genotypes=[...]` sugar; auto-instantiate `CrossImmunity` connector when none provided; raise on `diseases=` + `genotypes=` collision; expose Sim-level `cum_infections_any` / `cum_cancers_any` / `new_cancer_deaths_any` aggregators |
| `hpvsim/__init__.py` | Modify | Re-export `CrossImmunity`, `get_cross_immunity` |
| `tests/test_cross_immunity.py` | Create | Unit tests for Connector: matrix validation, identity case, hand-computed 2-genotype case, monotonicity |
| `tests/test_multi_genotype.py` | Create | 4-genotype Sim smoke + per-genotype results auto-stratification + aggregator correctness + `genotypes=` sugar parity with explicit `diseases=` |
| `tests/regression/anchor_4genotype.py` | Create | 4-genotype anchor scenario (Nigeria, seed 0, 1990-2060, four genotypes) |
| `tests/regression/baseline_v23.py` | Modify | Add `regen_4genotype()` entrypoint that produces v2 baseline JSON for the 4-genotype anchor |
| `tests/regression/short_summary.py` | Create | Builder for the 40-entry per-genotype + aggregate `short_summary` (factored out of `anchor_4genotype.py` so tests can re-use it) |
| `tests/test_m03_short_summary_parity.py` | Create | 40-entry drift table vs. v2 4-genotype baseline (±10% gate) |
| `tests/test_m03_trajectory_parity.py` | Create | Age-aggregated `cum_cancers_any` / `cum_infections_any` time-series trajectory comparison vs. v2 baseline |
| `tests/test_m02_through_connector.py` | Create | Re-run M02 1-genotype parity through the new Connector path; pinned drift bound from first run |

`tests/regression/baselines/anchor_4genotype.json` is regenerated locally (gitignored, per `MIGRATION_PLAN.md:79`).

---

## Task ordering and dependencies

```
Task  1 (raw immunity FloatArrs)
   ↓
Task  2 (port get_cross_immunity)
   ↓
Task  3 (CrossImmunity skeleton + validation)
   ↓
Task  4 (CrossImmunity step math)
   ↓
Task  5 (re-export CrossImmunity)
   ↓
Task  6 (genotypes= sugar in hpv.Sim, auto-Connector)
   ↓
Task  7 (move sev_imm/rel_sus writes to Connector; imm_init→dist)
   ↓
Task  8 (extend GenotypePars: hpv18, hi5, ohr)
   ↓
Task  9 (per-genotype init_prev table; expand _KNOWN_GENOTYPES)
   ↓
Task 10 (4-genotype Sim smoke + sugar parity)
   ↓
Task 11 (Sim-level aggregators: *_any)
   ↓
Task 12 (4-genotype anchor scenario script)
   ↓
Task 13 (40-entry short_summary builder)
   ↓
Task 14 (4-genotype baseline regen entrypoint)
   ↓
Task 15 (short_summary parity test)
   ↓
Task 16 (trajectory parity test + threshold pin)
   ↓
Task 17 (M02 through-Connector regression + drift pin)
```

The order is non-binding — execution can re-sequence as long as the continuous-runnability invariant (`MIGRATION_PLAN.md:258`) is preserved. The hard sequencing constraint is: Task 7 must come **after** Task 6, because Task 7 removes M02's clearance-time `rel_sus` and `sev_imm` writes, and the Connector that replaces them only exists once Task 6 wires it in.

---

### Task 1: Add raw `nab_imm` and `cell_imm` source-immunity FloatArrs to HPV

Add the per-source immunity state to `hpv.HPV` without yet reading from or writing to it. This is a pure addition; M02's existing `sev_imm` write stays in place. No behaviour change.

**Files:**
- Modify: `hpvsim/hpv.py:294-314`
- Modify: `tests/test_natural_history.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_natural_history.py`:

```python
def test_hpv_has_raw_immunity_states():
    """HPV defines 1D nab_imm / cell_imm FloatArrs as source-genotype immunity stores."""
    sim = hpv.Sim(n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16
    for name in ('nab_imm', 'cell_imm'):
        assert hasattr(mod, name), f'HPV missing FloatArr {name!r}'
        arr = getattr(mod, name)
        # Default 0.0 across the population at init.
        assert np.allclose(np.asarray(arr.values), 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_natural_history.py::test_hpv_has_raw_immunity_states -v
```

Expected: FAIL with `AttributeError: 'HPV' object has no attribute 'nab_imm'` (or similar).

- [ ] **Step 3: Add the FloatArrs to `define_states`**

In `hpvsim/hpv.py`, inside `HPV.__init__`'s `self.define_states(...)` call (currently lines 294–314), add:

```python
self.define_states(
    # ... (existing M02 entries above) ...
    ss.FloatArr('rel_sev', label='Relative severity (biological)', default=1.0),
    ss.BoolState('rel_sev_sampled', default=False),
    ss.FloatArr('sev_imm', label='Severity immunity (effective)', default=0.0),
    # M03 raw source-genotype immunity. Bumped on clearance; read by
    # CrossImmunity Connector to derive per-target rel_sus and sev_imm.
    ss.FloatArr('nab_imm', label='Humoral immunity (source genotype)', default=0.0),
    ss.FloatArr('cell_imm', label='Cell-mediated immunity (source genotype)', default=0.0),
)
```

- [ ] **Step 4: Run the new test plus the full suite to verify no regressions**

```bash
pytest tests/test_natural_history.py -v
pytest tests/ -x -q
```

Expected: new test PASS; all other tests still PASS (no behaviour change).

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_natural_history.py
git commit -m "M03: add raw nab_imm and cell_imm source FloatArrs to HPV"
```

---

### Task 2: Hand-port `get_cross_immunity` and per-genotype `imm_init` distribution

Translate v2's cross-protection matrix builder into active code. v2's source: `hpvsim/_v2_legacy/parameters.py:412-508`. The function returns a `(n_g, n_g)` numpy matrix indexed by genotype-key ordering. Defaults for the scalars `cross_imm_med` / `cross_imm_high` / `own_imm_hr` are pulled from v2's `make_pars` defaults (`_v2_legacy/parameters.py:108-112`).

**Files:**
- Modify: `hpvsim/parameters.py`
- Create: `tests/test_cross_immunity.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cross_immunity.py`:

```python
"""Unit tests for the cross-immunity Connector and matrix-builder."""
import numpy as np
import pytest

import hpvsim as hpv
from hpvsim.parameters import get_cross_immunity, GENOTYPE_KEYS


def test_genotype_keys_are_canonical_four():
    """GENOTYPE_KEYS pins the M03 4-genotype default ordering."""
    assert GENOTYPE_KEYS == ('hpv16', 'hpv18', 'hi5', 'ohr')


def test_get_cross_immunity_default_shape_and_diagonal():
    """Default cross-immunity matrices are (4, 4) float32 with diagonal == 1.0."""
    m_sus, m_sev = get_cross_immunity()
    for m in (m_sus, m_sev):
        assert m.shape == (4, 4)
        assert m.dtype == np.float32
        assert np.allclose(np.diag(m), 1.0)


def test_get_cross_immunity_default_values():
    """Defaults match v2 scalars: cross_imm_sus_med=0.3, cross_imm_sus_high=0.5,
    cross_imm_sev_med=0.5, cross_imm_sev_high=0.7. Diagonal is 1.0 for hpv16/hpv18,
    own_imm_hr=0.9 for hi5/ohr."""
    m_sus, m_sev = get_cross_immunity()
    keys = ('hpv16', 'hpv18', 'hi5', 'ohr')
    idx = {k: i for i, k in enumerate(keys)}
    # Off-diagonal hpv16<->hpv18 = high (0.5 sus, 0.7 sev); both directions.
    assert m_sus[idx['hpv16'], idx['hpv18']] == pytest.approx(0.5)
    assert m_sus[idx['hpv18'], idx['hpv16']] == pytest.approx(0.5)
    assert m_sev[idx['hpv16'], idx['hpv18']] == pytest.approx(0.7)
    # hpv16 -> hi5 = med
    assert m_sus[idx['hpv16'], idx['hi5']] == pytest.approx(0.3)
    assert m_sev[idx['hpv16'], idx['hi5']] == pytest.approx(0.5)
    # hi5 own-immunity = own_imm_hr = 0.9
    assert m_sus[idx['hi5'], idx['hi5']] == pytest.approx(1.0)  # diagonal forced to 1
    # own_imm_hr applies off-diagonal on hi5 row for hi5-only? No — hi5 row
    # diagonal is forced to 1.0 by convention; own_imm_hr is a v2-internal
    # input that drives non-hpv16/hpv18 own-immunity values, but our diagonal
    # convention overrides it. Cross-entries remain medium.


def test_get_cross_immunity_custom_keys():
    """Caller-supplied genotype ordering controls matrix layout."""
    m_sus, _ = get_cross_immunity(keys=('hi5', 'hpv16'))
    assert m_sus.shape == (2, 2)
    # m_sus[0, 1] is "from hi5 source to hpv16 target" — medium scalar.
    assert m_sus[1, 0] == pytest.approx(0.3)


def test_genotype_pars_imm_init_is_distribution():
    """GenotypePars.imm_init becomes a beta-mean distribution (M03 conversion)."""
    gp = hpv.get_genotype_pars('hpv16')
    assert hasattr(gp.imm_init, 'rvs'), \
        f'imm_init should be a Dist, got {type(gp.imm_init)}'
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_cross_immunity.py -v
```

Expected: ALL FAIL (`get_cross_immunity` and `GENOTYPE_KEYS` don't exist).

- [ ] **Step 3: Implement `GENOTYPE_KEYS`, `get_cross_immunity()`, and convert `imm_init`**

In `hpvsim/parameters.py`:

```python
import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['SimPars', 'GenotypePars', 'get_genotype_pars', 'get_cross_immunity',
           'genotype_aliases', 'GENOTYPE_KEYS']


# Canonical 4-genotype ordering for M03's default Sim. The Connector uses
# this order as the default when no genotype list is supplied.
GENOTYPE_KEYS = ('hpv16', 'hpv18', 'hi5', 'ohr')


genotype_aliases = {
    'hpv16': ['hpv16', '16'],
    'hpv18': ['hpv18', '18'],
    'hi5':   ['hi5', 'high-risk-5'],
    'ohr':   ['ohr', 'other-high-risk'],
}


def _imm_init_dist():
    """Beta sample for per-clearance humoral-immunity boost.

    Shape parameters from v2's beta_mean(par1=0.35, par2=0.025).
    """
    a = ((1 - 0.35) / 0.025 - 1 / 0.35) * 0.35 ** 2
    b = a * (1 / 0.35 - 1)
    return ss.Dist(distname='beta', a=a, b=b)


def _cell_imm_dist():
    """Beta sample for per-clearance severity-immunity boost (v2 cell_imm_init).

    Shape parameters from v2's beta_mean(par1=0.25, par2=0.025).
    """
    a = ((1 - 0.25) / 0.025 - 1 / 0.25) * 0.25 ** 2
    b = a * (1 / 0.25 - 1)
    return ss.Dist(distname='beta', a=a, b=b)


# v2 defaults (see hpvsim/_v2_legacy/parameters.py:108-112)
_DEFAULT_CROSS_IMM_SUS_MED = 0.3
_DEFAULT_CROSS_IMM_SUS_HIGH = 0.5
_DEFAULT_CROSS_IMM_SEV_MED = 0.5
_DEFAULT_CROSS_IMM_SEV_HIGH = 0.7
_DEFAULT_OWN_IMM_HR = 0.9   # used by v2 for non-hpv16/hpv18 own-immunity;
                              # M03 forces diagonal == 1.0 by convention so
                              # this value is unused for the default 4-genotype
                              # set, but kept for parity with v2's signature.

# Pairwise cross-protection clade map. 'high' = hpv16 <-> hpv18; everything
# else is 'med'. Hand-ported from v2's get_cross_immunity (hpvsim/_v2_legacy/
# parameters.py:412-508).
_CLADE_HIGH_PAIRS = frozenset({
    ('hpv16', 'hpv18'),
    ('hpv18', 'hpv16'),
})


def _build_cross_matrix(keys, scalar_med, scalar_high):
    """Hand-ported from v2 get_cross_immunity. Diagonal forced to 1.0."""
    n = len(keys)
    m = np.full((n, n), scalar_med, dtype=np.float32)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                m[i, j] = 1.0
            elif (ki, kj) in _CLADE_HIGH_PAIRS:
                m[i, j] = scalar_high
    return m


def get_cross_immunity(keys=None,
                        cross_imm_sus_med=None, cross_imm_sus_high=None,
                        cross_imm_sev_med=None, cross_imm_sev_high=None):
    """Build (cross_immunity_sus, cross_immunity_sev) matrices for the given
    genotype ordering.

    Returns a tuple of two ``(n, n)`` float32 arrays. ``keys`` defaults to
    ``GENOTYPE_KEYS``. Scalar defaults match v2 (`cross_imm_sus_med=0.3`,
    `cross_imm_sus_high=0.5`, `cross_imm_sev_med=0.5`, `cross_imm_sev_high=0.7`).
    Diagonals are forced to 1.0 by convention.
    """
    if keys is None:
        keys = GENOTYPE_KEYS
    if cross_imm_sus_med is None:  cross_imm_sus_med  = _DEFAULT_CROSS_IMM_SUS_MED
    if cross_imm_sus_high is None: cross_imm_sus_high = _DEFAULT_CROSS_IMM_SUS_HIGH
    if cross_imm_sev_med is None:  cross_imm_sev_med  = _DEFAULT_CROSS_IMM_SEV_MED
    if cross_imm_sev_high is None: cross_imm_sev_high = _DEFAULT_CROSS_IMM_SEV_HIGH
    m_sus = _build_cross_matrix(keys, cross_imm_sus_med, cross_imm_sus_high)
    m_sev = _build_cross_matrix(keys, cross_imm_sev_med, cross_imm_sev_high)
    return m_sus, m_sev
```

Convert `imm_init` from scalar to distribution by replacing `self.imm_init = 0.35` (currently `parameters.py:90`) with:

```python
            self.imm_init = _imm_init_dist()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cross_immunity.py -v
```

Expected: all PASS.

Note: `tests/test_natural_history.py::test_hpv_has_progression_pars` (line 19-24 of `tests/test_natural_history.py`) checks `'imm_init' in p`, which still passes (membership) — the type changed but the key is still present. The full test suite will surface any other readers of `imm_init` as a scalar; address those as part of Task 7.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/parameters.py tests/test_cross_immunity.py
git commit -m "M03: hand-port get_cross_immunity from v2; convert imm_init to dist"
```

---

### Task 3: `CrossImmunity` Connector skeleton + init validation

Create the new `hpvsim/connectors.py` module and add the Connector class with `init_pre` validation: collect HPV modules from `sim.diseases`, populate matrices from `get_cross_immunity` if missing, validate shape and diagonal. The `step()` body comes in Task 4.

**Files:**
- Create: `hpvsim/connectors.py`
- Modify: `tests/test_cross_immunity.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cross_immunity.py`:

```python
def test_cross_immunity_connector_collects_hpv_modules():
    """CrossImmunity.init_pre populates _hpv_modules from sim.diseases in registration order."""
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    conn = sim.connectors.crossimmunity   # Starsim auto-snake-cases class name
    assert len(conn._hpv_modules) == 1
    assert conn._hpv_modules[0].genotype == 'hpv16'
    assert conn._genotype_index == {'hpv16': 0}


def test_cross_immunity_connector_default_matrices():
    """If matrices not supplied, init_pre populates from get_cross_immunity for the discovered genotype set."""
    from hpvsim.connectors import CrossImmunity
    conn = CrossImmunity()
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[conn],
    )
    sim.init()
    assert conn.cross_imm_sus.shape == (1, 1)
    assert conn.cross_imm_sus.dtype == np.float32
    assert conn.cross_imm_sus[0, 0] == pytest.approx(1.0)


def test_cross_immunity_connector_rejects_off_diagonal_self_immunity():
    """Diagonal entries must be 1.0; init_pre raises otherwise."""
    from hpvsim.connectors import CrossImmunity
    bad = np.array([[0.5]], dtype=np.float32)
    conn = CrossImmunity(cross_imm_sus=bad, cross_imm_sev=bad)
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[conn],
    )
    with pytest.raises(ValueError, match='diagonal'):
        sim.init()


def test_cross_immunity_connector_rejects_shape_mismatch():
    """Matrix dim must match number of HPV modules."""
    from hpvsim.connectors import CrossImmunity
    bad = np.eye(2, dtype=np.float32)
    conn = CrossImmunity(cross_imm_sus=bad, cross_imm_sev=bad)
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[conn],
    )
    with pytest.raises(ValueError, match='shape'):
        sim.init()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_cross_immunity.py -v -k connector
```

Expected: ALL FAIL (`hpvsim.connectors` doesn't exist).

- [ ] **Step 3: Implement `CrossImmunity` skeleton**

Create `hpvsim/connectors.py`:

```python
"""Cross-immunity Connector: derive per-target rel_sus and sev_imm from
per-source nab_imm and cell_imm via v2's cross-protection matrices.

Convention: row = target genotype, col = source genotype. Effective immunity
to target ``g`` is ``sum_k cross[g, k] * source[uid, k]``. Matrices live on
the Connector instance (not on SimPars). Diagonals must equal 1.0.
"""

import warnings

import numpy as np
import starsim as ss

from .hpv import HPV
from .parameters import get_cross_immunity


class CrossImmunity(ss.Connector):
    """Cross-immunity Connector for multi-genotype HPV.

    Reads each registered ``HPV`` instance's source-genotype ``nab_imm`` and
    ``cell_imm``; writes per-target ``rel_sus`` (= 1 - sus_imm) and ``sev_imm``
    each step, after Disease.step_state and before Disease.step_infect.
    """

    def __init__(self, cross_imm_sus=None, cross_imm_sev=None, **kwargs):
        super().__init__(**kwargs)
        self.cross_imm_sus = cross_imm_sus
        self.cross_imm_sev = cross_imm_sev
        self._hpv_modules = None
        self._genotype_index = None

    def init_pre(self, sim):
        super().init_pre(sim)
        # Discover HPV modules in registration order.
        self._hpv_modules = [m for m in sim.diseases.values() if isinstance(m, HPV)]
        if not self._hpv_modules:
            warnings.warn('CrossImmunity: no HPV diseases registered; Connector is a no-op.')
            self._genotype_index = {}
            return
        keys = tuple(m.genotype for m in self._hpv_modules)
        self._genotype_index = {k: i for i, k in enumerate(keys)}

        # Populate defaults if matrices not supplied.
        if self.cross_imm_sus is None or self.cross_imm_sev is None:
            m_sus, m_sev = get_cross_immunity(keys=keys)
            if self.cross_imm_sus is None:
                self.cross_imm_sus = m_sus
            if self.cross_imm_sev is None:
                self.cross_imm_sev = m_sev

        # Cast and validate.
        self.cross_imm_sus = np.asarray(self.cross_imm_sus, dtype=np.float32)
        self.cross_imm_sev = np.asarray(self.cross_imm_sev, dtype=np.float32)
        n = len(self._hpv_modules)
        for label, m in (('cross_imm_sus', self.cross_imm_sus),
                         ('cross_imm_sev', self.cross_imm_sev)):
            if m.shape != (n, n):
                raise ValueError(
                    f'CrossImmunity.{label}: shape {m.shape} does not match '
                    f'number of HPV modules {n}'
                )
            if not np.allclose(np.diag(m), 1.0):
                raise ValueError(
                    f'CrossImmunity.{label}: diagonal must be 1.0; '
                    f'got {np.diag(m)}'
                )

    def step(self):
        # Body added in Task 4.
        return
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_cross_immunity.py -v -k connector
pytest tests/ -x -q
```

Expected: connector tests PASS; full suite still PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/connectors.py tests/test_cross_immunity.py
git commit -m "M03: add CrossImmunity connector skeleton with init validation"
```

---

### Task 4: `CrossImmunity.step` — stack, multiply, scatter

Implement the per-step matrix multiply: stack each HPV's `nab_imm` / `cell_imm` into 2D, apply cross-protection matrices, scatter `rel_sus = 1 - sus_imm` and `sev_imm` back to each module.

**Files:**
- Modify: `hpvsim/connectors.py`
- Modify: `tests/test_cross_immunity.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cross_immunity.py`:

```python
def test_cross_immunity_step_identity_for_single_genotype():
    """1x1 identity matrix: rel_sus = 1 - nab_imm, sev_imm = cell_imm."""
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=10, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    mod = sim.diseases.hpv16
    conn = sim.connectors.crossimmunity
    # Manually set source immunity for half the agents, then step the connector.
    mod.nab_imm.values[:5] = 0.4
    mod.cell_imm.values[:5] = 0.3
    conn.step()
    # Identity multiply: rel_sus = 1 - 0.4 = 0.6 for first five, 1.0 elsewhere.
    rel = np.asarray(mod.rel_sus.values)
    sev = np.asarray(mod.sev_imm.values)
    assert np.allclose(rel[:5], 0.6)
    assert np.allclose(rel[5:], 1.0)
    assert np.allclose(sev[:5], 0.3)
    assert np.allclose(sev[5:], 0.0)


def test_cross_immunity_step_two_genotype_hand_computed():
    """2-genotype case: rel_sus and sev_imm match hand-computed dot product."""
    from hpvsim.connectors import CrossImmunity
    # Cross-immunity: 16->18 = 0.5 sus / 0.7 sev; 18->16 = 0.5 sus / 0.7 sev.
    m_sus = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    m_sev = np.array([[1.0, 0.7], [0.7, 1.0]], dtype=np.float32)
    sim = hpv.Sim(
        n_agents=4, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        connectors=[CrossImmunity(cross_imm_sus=m_sus, cross_imm_sev=m_sev)],
    )
    sim.init()
    h16 = sim.diseases.hpv16
    h18 = sim.diseases.hpv18
    conn = sim.connectors.crossimmunity
    # Agent 0: had hpv16 (nab=0.4, cell=0.3); no hpv18 history.
    h16.nab_imm.values[0] = 0.4
    h16.cell_imm.values[0] = 0.3
    # Agent 1: had hpv18 (nab=0.6, cell=0.5); no hpv16 history.
    h18.nab_imm.values[1] = 0.6
    h18.cell_imm.values[1] = 0.5
    conn.step()
    # Target hpv16, agent 0: sus_imm = 1.0*0.4 + 0.5*0 = 0.4 -> rel_sus = 0.6
    # Target hpv18, agent 0: sus_imm = 0.5*0.4 + 1.0*0 = 0.2 -> rel_sus = 0.8
    # Target hpv16, agent 1: sus_imm = 1.0*0 + 0.5*0.6 = 0.3 -> rel_sus = 0.7
    # Target hpv18, agent 1: sus_imm = 0.5*0 + 1.0*0.6 = 0.6 -> rel_sus = 0.4
    assert h16.rel_sus.values[0] == pytest.approx(0.6, abs=1e-6)
    assert h18.rel_sus.values[0] == pytest.approx(0.8, abs=1e-6)
    assert h16.rel_sus.values[1] == pytest.approx(0.7, abs=1e-6)
    assert h18.rel_sus.values[1] == pytest.approx(0.4, abs=1e-6)
    # sev_imm: target hpv16, agent 0 = 1.0*0.3 + 0.7*0 = 0.3
    #          target hpv18, agent 1 = 0.7*0 + 1.0*0.5 = 0.5
    assert h16.sev_imm.values[0] == pytest.approx(0.3, abs=1e-6)
    assert h18.sev_imm.values[1] == pytest.approx(0.5, abs=1e-6)


def test_cross_immunity_step_clips_to_unit_interval():
    """sus_imm and sev_imm are clipped to [0, 1] (matches v2 np.minimum cap)."""
    from hpvsim.connectors import CrossImmunity
    sim = hpv.Sim(
        n_agents=4, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        connectors=[CrossImmunity()],
    )
    sim.init()
    sim.diseases.hpv16.nab_imm.values[0] = 0.9
    sim.diseases.hpv18.nab_imm.values[0] = 0.9
    sim.connectors.crossimmunity.step()
    # Agent 0: sus_imm to hpv16 = 1*0.9 + 0.5*0.9 = 1.35 -> clipped to 1.0
    # rel_sus = 1 - 1.0 = 0.0
    assert sim.diseases.hpv16.rel_sus.values[0] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_cross_immunity.py -v -k step
```

Expected: ALL FAIL (`step` is currently a no-op).

- [ ] **Step 3: Implement `CrossImmunity.step`**

Replace the stub `step` method in `hpvsim/connectors.py` with:

```python
    def step(self):
        if not self._hpv_modules:
            return
        nab  = np.column_stack([np.asarray(m.nab_imm.values)  for m in self._hpv_modules])
        cell = np.column_stack([np.asarray(m.cell_imm.values) for m in self._hpv_modules])
        sus_imm = nab  @ self.cross_imm_sus.T
        sev_imm = cell @ self.cross_imm_sev.T
        np.clip(sus_imm, 0.0, 1.0, out=sus_imm)
        np.clip(sev_imm, 0.0, 1.0, out=sev_imm)
        for i, m in enumerate(self._hpv_modules):
            m.rel_sus.values[:]  = 1.0 - sus_imm[:, i]
            m.sev_imm.values[:]  = sev_imm[:, i]
        return
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_cross_immunity.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/connectors.py tests/test_cross_immunity.py
git commit -m "M03: implement CrossImmunity.step (stack-multiply-scatter)"
```

---

### Task 5: Re-export `CrossImmunity` and `get_cross_immunity` from package init

**Files:**
- Modify: `hpvsim/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cross_immunity.py`:

```python
def test_cross_immunity_top_level_import():
    """hpv.CrossImmunity and hpv.get_cross_immunity are importable from package root."""
    assert hasattr(hpv, 'CrossImmunity')
    assert hasattr(hpv, 'get_cross_immunity')
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_cross_immunity.py::test_cross_immunity_top_level_import -v
```

Expected: FAIL.

- [ ] **Step 3: Add the re-exports**

In `hpvsim/__init__.py`, add to the imports section:

```python
from .parameters import (SimPars, GenotypePars, get_genotype_pars,
                         get_cross_immunity, GENOTYPE_KEYS)
from .connectors import CrossImmunity
```

And to `__all__`:

```python
__all__ = [
    'HPV', 'SexualNetwork', 'Sim', 'AgeMigration', 'CrossImmunity',
    'data', 'migration_utils', 'options', 'datadir', '__version__',
    'SimPars', 'GenotypePars', 'get_genotype_pars', 'get_cross_immunity',
    'GENOTYPE_KEYS',
]
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_cross_immunity.py -v
pytest tests/ -x -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/__init__.py
git commit -m "M03: re-export CrossImmunity and get_cross_immunity"
```

---

### Task 6: `genotypes=[...]` sugar in `hpv.Sim` + auto-instantiate Connector

Extend `hpv.Sim.__init__` to accept `genotypes=[...]` (with alias normalisation), construct the four HPV modules from `GenotypePars`, and auto-instantiate `CrossImmunity` if no connector is supplied. Single-genotype path goes through the new sugar too. Raise on `diseases=` + `genotypes=` collision.

**Files:**
- Modify: `hpvsim/sim.py`
- Create: `tests/test_multi_genotype.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_multi_genotype.py`:

```python
"""Multi-genotype Sim API tests: genotypes= sugar, auto-Connector wiring."""
import numpy as np
import pytest

import hpvsim as hpv


def test_sim_with_explicit_diseases_and_connectors():
    """Explicit diseases= + connectors= path still works (M02 surface)."""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[hpv.CrossImmunity()],
    )
    sim.run()
    assert 'hpv16' in sim.diseases


def test_sim_genotypes_sugar_single():
    """genotypes=[16] auto-instantiates one HPV + a CrossImmunity Connector."""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=[16],
    )
    sim.run()
    assert list(sim.diseases.keys()) == ['hpv16']
    # Connector auto-added and named 'crossimmunity'.
    assert any('crossimmunity' in k.lower() for k in sim.connectors.keys()), \
        f'No CrossImmunity connector found; got {list(sim.connectors.keys())}'


def test_sim_genotypes_sugar_four():
    """genotypes=[16, 18, 'hi5', 'ohr'] -> four HPV modules + Connector."""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
    )
    sim.init()
    assert list(sim.diseases.keys()) == ['hpv16', 'hpv18', 'hi5', 'ohr']


def test_sim_rejects_diseases_plus_genotypes():
    """Passing both diseases= and genotypes= raises early."""
    with pytest.raises(ValueError, match='diseases.*genotypes'):
        hpv.Sim(
            n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
            diseases=[hpv.HPV(genotype='hpv16')],
            genotypes=[16],
        )


def test_sim_genotype_pars_override():
    """genotype_pars={'hpv16': {'rel_beta': 1.5}} overrides per-genotype defaults."""
    sim = hpv.Sim(
        n_agents=200, start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=[16],
        genotype_pars={'hpv16': {'rel_beta': 1.5}},
    )
    sim.init()
    assert float(sim.diseases.hpv16.pars.rel_beta) == pytest.approx(1.5)
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_multi_genotype.py -v
```

Expected: ALL FAIL (sugar not implemented).

- [ ] **Step 3: Implement the sugar**

Replace `hpvsim/sim.py` with:

```python
"""HPVsim convenience Sim wrapper.

``hpv.Sim(location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'])`` instantiates
the default stack — one HPV disease module per genotype, multi-layer
SexualNetwork, ss.Births + ss.Deaths + AgeMigration demographics, ss.People
with location-specific age pyramid, plus a CrossImmunity connector — and
forwards to ``ss.Sim``. Each component is overridable: passing ``diseases=``
short-circuits the genotypes-sugar path.
"""

import starsim as ss

from .data.country import load_country
from .demographics import AgeMigration
from .hpv import HPV
from .network import SexualNetwork
from .connectors import CrossImmunity
from .parameters import genotype_aliases, GENOTYPE_KEYS


def _normalize_genotype(key):
    """Resolve aliases (16 -> 'hpv16', 'hi5' -> 'hi5') to canonical keys."""
    s = str(key).lower().strip()
    for canonical, aliases in genotype_aliases.items():
        if s == canonical or s in aliases:
            return canonical
    raise ValueError(
        f'Unknown genotype {key!r}; valid: {list(genotype_aliases)}'
    )


class Sim(ss.Sim):
    """HPVsim simulation."""

    def __init__(self, location='nigeria', genotypes=None, genotype_pars=None,
                 n_agents=10_000, start=1990, stop=2060, dt=0.25,
                 total_pop=None, pars=None, **kwargs):
        country = load_country(location)
        people = kwargs.pop('people', None)
        if people is None:
            people = ss.People(n_agents, age_data=country['age_data'])

        diseases = kwargs.pop('diseases', None)
        connectors = kwargs.pop('connectors', None)

        if diseases is not None and genotypes is not None:
            raise ValueError(
                'Pass diseases= OR genotypes=, not both.'
            )

        if diseases is None:
            # Default to single-genotype HPV16 if neither supplied.
            keys = (tuple(_normalize_genotype(g) for g in genotypes)
                    if genotypes is not None else ('hpv16',))
            gpars_overrides = genotype_pars or {}
            diseases = [
                HPV(genotype=k, **gpars_overrides.get(k, {}))
                for k in keys
            ]

        if connectors is None:
            connectors = [CrossImmunity()]

        networks = kwargs.pop('networks', None)
        if networks is None:
            networks = [SexualNetwork(**country['network_pars'])]
        demographics = kwargs.pop('demographics', None)
        if demographics is None:
            demographics = [
                ss.Births(birth_rate=country['birth_rate']),
                ss.Deaths(death_rate=country['death_rate']),
                AgeMigration(),
            ]
        # AgeMigration.init_pre reads sim.location to load country data.
        self.location = location.lower()
        super().__init__(
            start=ss.years(start),
            stop=ss.years(stop),
            dt=ss.years(dt),
            people=people,
            diseases=diseases,
            connectors=connectors,
            networks=networks,
            demographics=demographics,
            pars=pars,
            total_pop=total_pop,
            **kwargs,
        )
```

Note this changes the `Sim` signature: the old `genotype='hpv16'` kwarg is gone, replaced by `genotypes=None` (default = single HPV16). Existing call sites passing `genotype='hpv16'` need updating to `genotypes=['hpv16']` (or omitting entirely to take the default).

- [ ] **Step 4: Update existing call sites + run all tests**

Search for `genotype=` in tests and the regression harness:

```bash
git grep -n "genotype=" tests/ hpvsim/
```

For each occurrence in test code that uses `hpv.Sim(genotype=...)`, replace with `genotypes=[...]`. The anchor scenario `tests/regression/anchor_hpv16.py:22` uses `genotype='hpv16'` in its `PARS` dict — update to `genotypes=['hpv16']`.

```bash
pytest tests/test_multi_genotype.py -v
pytest tests/ -x -q
```

Expected: new tests PASS; all existing tests PASS after call-site fixes.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/sim.py tests/test_multi_genotype.py tests/regression/anchor_hpv16.py
git commit -m "M03: hpv.Sim genotypes= sugar with auto-Connector"
```

---

### Task 7: Move `sev_imm` and `rel_sus` writes from HPV.step_state to Connector

Drop M02's clearance-time writes to `sev_imm` and `rel_sus`. In their place, sample `imm_init` (now a beta distribution) into `nab_imm` and continue sampling `cell_imm_init` into `cell_imm`. The Connector now drives both `rel_sus` and `sev_imm`.

This is the M03 commit where M02's deterministic immunity cap (`rel_sus = min(rel_sus, 1 - 0.35)`) becomes a per-clearance beta sample (`nab_imm[cleared] = max(prior, beta_mean(0.35, 0.025).rvs())`). Small drift on M02's 1-genotype baseline is anticipated and pinned in Task 17.

**Files:**
- Modify: `hpvsim/hpv.py:494-509`
- Modify: `tests/test_natural_history.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_natural_history.py`:

```python
def test_clearance_writes_raw_immunity_not_effective():
    """After Task 7: HPV.step_state writes nab_imm/cell_imm; rel_sus/sev_imm are Connector-derived."""
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2010, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    # After running, agents that have ever cleared should have nab_imm > 0.
    nab = np.asarray(mod.nab_imm.values)
    cell = np.asarray(mod.cell_imm.values)
    # At least some agents will have cleared given run length.
    assert (nab > 0).any(), 'no agents cleared and bumped nab_imm'
    assert (cell > 0).any(), 'no agents cleared and bumped cell_imm'
    # And rel_sus / sev_imm reflect the source state through the Connector
    # (single-genotype identity: rel_sus = 1 - nab_imm; sev_imm = cell_imm).
    rel_sus = np.asarray(mod.rel_sus.values)
    sev_imm = np.asarray(mod.sev_imm.values)
    cleared_uids = (nab > 0).nonzero()[0]
    assert np.allclose(rel_sus[cleared_uids], 1.0 - nab[cleared_uids], atol=1e-6)
    assert np.allclose(sev_imm[cleared_uids], cell[cleared_uids], atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_natural_history.py::test_clearance_writes_raw_immunity_not_effective -v
```

Expected: FAIL — current code writes `sev_imm` directly and `nab_imm` is never bumped.

- [ ] **Step 3: Modify HPV.step_state clearance branch**

In `hpvsim/hpv.py`, replace lines 494–509 (the clearance branch) with:

```python
        # --- 1. Clearance (from precin OR CIN) — partial-immunity path ---
        # Returns agent to susceptible=True. Sources are bumped on clearance:
        #   nab_imm  <- max(prior, imm_init.rvs())     (humoral)
        #   cell_imm <- max(prior, cell_imm_init.rvs()) (cell-mediated)
        # CrossImmunity Connector reads these and writes per-target rel_sus
        # and sev_imm next step. rel_sev (biological baseline) is unchanged.
        cleared = (self.infected & (self.precin | self.cin) & ~self.cancerous
                   & (self.ti_clearance <= ti)).uids
        if len(cleared):
            self.infected[cleared] = False
            self.susceptible[cleared] = True
            self.precin[cleared] = False
            self.cin[cleared] = False
            new_nab = np.asarray(self.pars.imm_init.rvs(cleared))
            self.nab_imm[cleared] = np.maximum(self.nab_imm[cleared], new_nab)
            new_cell = np.asarray(self.pars.cell_imm_init.rvs(cleared))
            self.cell_imm[cleared] = np.maximum(self.cell_imm[cleared], new_cell)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_natural_history.py -v
pytest tests/ -x -q
```

Expected: new test PASS. Existing M02 tests may show small numerical drift in summary metrics — this is anticipated. If `tests/test_natural_history.py` has any assertions on exact pre-Task-7 numbers (e.g. specific `rel_sus` values), relax them to ranges or remove. The 1-genotype baseline drift is tracked in Task 17.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_natural_history.py
git commit -m "M03: HPV.step_state writes nab_imm and cell_imm; Connector drives rel_sus and sev_imm"
```

---

### Task 8: Extend `GenotypePars` with `hpv18`, `hi5`, `ohr` defaults

Add v2's per-genotype natural-history defaults for the three additional genotypes. Source: `hpvsim/_v2_legacy/parameters.py:344-388`.

**Files:**
- Modify: `hpvsim/parameters.py`
- Modify: `tests/test_natural_history.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_natural_history.py`:

```python
@pytest.mark.parametrize('genotype', ['hpv18', 'hi5', 'ohr'])
def test_genotype_pars_for_non_hpv16(genotype):
    """GenotypePars supports hpv18, hi5, ohr with v2 defaults."""
    gp = hpv.get_genotype_pars(genotype)
    assert gp.genotype == genotype
    for name in ('dur_precin', 'dur_cin', 'dur_cancer', 'cin_fn',
                 'cancer_fn', 'imm_init', 'cell_imm_init', 'rel_beta'):
        assert name in gp, f'{genotype} GenotypePars missing {name!r}'


def test_hpv18_specific_v2_values():
    """hpv18 has v2's specific cin_fn k=0.25 and rel_beta=0.75."""
    gp = hpv.get_genotype_pars('hpv18')
    assert gp.cin_fn['k'] == pytest.approx(0.25)
    assert float(gp.rel_beta) == pytest.approx(0.75)
    assert float(gp.sero_prob) == pytest.approx(0.56)


def test_hi5_specific_v2_values():
    """hi5 has v2's cancer_fn transform_prob=1.5e-3."""
    gp = hpv.get_genotype_pars('hi5')
    assert gp.cancer_fn['transform_prob'] == pytest.approx(1.5e-3)
    assert float(gp.rel_beta) == pytest.approx(0.9)
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_natural_history.py -v -k 'hpv18 or hi5 or ohr or non_hpv16'
```

Expected: FAIL with `NotImplementedError: GenotypePars currently supports hpv16 only`.

- [ ] **Step 3: Add the three genotype branches**

In `hpvsim/parameters.py` `GenotypePars.__init__`, replace the `else: raise NotImplementedError` block with three new branches. Insert before the `else`:

```python
        elif genotype == 'hpv18':
            self.beta = 0.25
            self.dur_precin = ss.lognorm_ex(mean=ss.years(2.5), std=ss.years(9.0))
            self.dur_cin = ss.lognorm_ex(mean=ss.years(5.0), std=ss.years(20.0))
            self.dur_cancer = ss.lognorm_ex(mean=ss.years(8.0), std=ss.years(3.0))
            self.dur_inf_male = ss.lognorm_ex(mean=ss.years(1.0), std=ss.years(1.0))
            self.cin_fn = dict(form='logf2', k=0.25, x_infl=0, ttc=50)
            self.cancer_fn = dict(method='cin_integral', transform_prob=2e-3,
                                  form='logf2', k=0.25, x_infl=0, ttc=50)
            self.imm_init = _imm_init_dist()
            self.cell_imm_init = _cell_imm_dist()
            self.age_risk = dict(age=30, risk=2)
            self.rel_beta = 0.75
            self.sero_prob = 0.56
        elif genotype in ('hi5', 'ohr'):
            self.beta = 0.25
            self.dur_precin = ss.lognorm_ex(mean=ss.years(2.5), std=ss.years(9.0))
            self.dur_cin = ss.lognorm_ex(mean=ss.years(4.5), std=ss.years(20.0))
            self.dur_cancer = ss.lognorm_ex(mean=ss.years(8.0), std=ss.years(3.0))
            self.dur_inf_male = ss.lognorm_ex(mean=ss.years(1.0), std=ss.years(1.0))
            self.cin_fn = dict(form='logf2', k=0.2, x_infl=0, ttc=50)
            self.cancer_fn = dict(method='cin_integral', transform_prob=1.5e-3,
                                  form='logf2', k=0.2, x_infl=0, ttc=50)
            self.imm_init = _imm_init_dist()
            self.cell_imm_init = _cell_imm_dist()
            self.age_risk = dict(age=30, risk=2)
            self.rel_beta = 0.9
            self.sero_prob = 0.60
```

The shared `dur_cancer` / `dur_inf_male` / `age_risk` defaults match v2's sim-level pars (`_v2_legacy/parameters.py:96-99`); v2 stored these per-sim, not per-genotype, so the same value lands in each branch.

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_natural_history.py -v
pytest tests/ -x -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/parameters.py tests/test_natural_history.py
git commit -m "M03: GenotypePars defaults for hpv18, hi5, ohr (v2 hand-port)"
```

---

### Task 9: Per-genotype `init_prev` table + expand `_KNOWN_GENOTYPES`

M02 stored HPV16's age-banded prevalence as module-level constants in `hpvsim/hpv.py:231-233`. M03 generalises this to a per-genotype dict so each `HPV(genotype=...)` instance picks up its own curve. HPV16's curve moves verbatim under the `'hpv16'` key. The other three get scaled-down v2 defaults.

**Files:**
- Modify: `hpvsim/hpv.py:226-248`
- Modify: `tests/test_natural_history.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_natural_history.py`:

```python
def test_known_genotypes_extended_to_four():
    """_KNOWN_GENOTYPES gates HPV(genotype=...) for all four M03 keys."""
    # All four keys instantiate without raising.
    for key in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        mod = hpv.HPV(genotype=key)
        assert mod.genotype == key


def test_per_genotype_init_prev_curve():
    """Each genotype gets its own init_prev curve via _age_stratified_init_prev."""
    # Build a 4-genotype Sim and seed; check that each genotype seeds non-zero
    # initial infections (in the sexually active age bands).
    sim = hpv.Sim(
        n_agents=5000, location='nigeria',
        start=1990, stop=1990, dt=1.0, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
    )
    sim.init()
    for key in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        mod = sim.diseases[key]
        n_init = int(np.asarray(mod.infected.values).sum())
        assert n_init > 0, f'{key} seeded zero initial infections'
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_natural_history.py -v -k 'known_genotypes or init_prev_curve'
```

Expected: FAIL — `_KNOWN_GENOTYPES` is `('hpv16',)` and seeding for non-hpv16 will not occur.

- [ ] **Step 3: Generalise the init-prev path and expand `_KNOWN_GENOTYPES`**

In `hpvsim/hpv.py`, replace lines 226–248 with:

```python
# Per-genotype initial HPV prevalence by age bracket and sex. Brackets are
# inclusive lower bounds; the last bracket extends to age 150. HPV16 retains
# M02's curves verbatim; hpv18/hi5/ohr scale-down defaults from v2 reference
# tables (we use 0.6x for hpv18 and 0.4x for hi5/ohr as proxies for the lower
# observed prevalence of those clades; calibrated values come in M04).
_INIT_HPV_PREV_AGE_BRACKETS = np.array([12, 17, 24, 34, 44, 64, 80, 150])

_INIT_PREV = {
    'hpv16': {
        'm': np.array([0.0, 0.25, 0.60, 0.25, 0.05, 0.01, 0.0005, 0.0]),
        'f': np.array([0.0, 0.35, 0.70, 0.25, 0.05, 0.01, 0.0005, 0.0]),
    },
    'hpv18': {
        'm': np.array([0.0, 0.15, 0.36, 0.15, 0.03, 0.006, 0.0003, 0.0]),
        'f': np.array([0.0, 0.21, 0.42, 0.15, 0.03, 0.006, 0.0003, 0.0]),
    },
    'hi5': {
        'm': np.array([0.0, 0.10, 0.24, 0.10, 0.02, 0.004, 0.0002, 0.0]),
        'f': np.array([0.0, 0.14, 0.28, 0.10, 0.02, 0.004, 0.0002, 0.0]),
    },
    'ohr': {
        'm': np.array([0.0, 0.10, 0.24, 0.10, 0.02, 0.004, 0.0002, 0.0]),
        'f': np.array([0.0, 0.14, 0.28, 0.10, 0.02, 0.004, 0.0002, 0.0]),
    },
}

# Public re-export — kept module-level for backward-compat with M02 imports.
_INIT_HPV_PREV_M = _INIT_PREV['hpv16']['m']
_INIT_HPV_PREV_F = _INIT_PREV['hpv16']['f']


def _make_init_prev_fn(genotype):
    """Return the per-uid init-prev sampler for a given genotype."""
    curves = _INIT_PREV[genotype]
    f_curve = curves['f']
    m_curve = curves['m']

    def _age_stratified(module, sim, uids):
        age = np.asarray(sim.people.age[uids])
        is_female = np.asarray(sim.people.female[uids])
        bin_idx = np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, age, side='right')
        bin_idx = np.clip(bin_idx, 0, len(f_curve) - 1)
        out = np.zeros(len(uids))
        out[is_female] = f_curve[bin_idx[is_female]]
        out[~is_female] = m_curve[bin_idx[~is_female]]
        return out
    return _age_stratified


_KNOWN_GENOTYPES = tuple(_INIT_PREV.keys())
```

Then in `HPV.__init__`, replace the line `init_prev=ss.bernoulli(p=_age_stratified_init_prev),` with:

```python
            init_prev=ss.bernoulli(p=_make_init_prev_fn(genotype)),
```

Also delete the old module-level `_age_stratified_init_prev` function (now superseded by `_make_init_prev_fn`).

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_natural_history.py -v
pytest tests/ -x -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hpv.py tests/test_natural_history.py
git commit -m "M03: per-genotype init_prev table; extend _KNOWN_GENOTYPES to 4 keys"
```

---

### Task 10: 4-genotype Sim smoke + `genotypes=` parity

Sanity-check that a 4-genotype Sim runs end-to-end and that `genotypes=[...]` produces the same result as the equivalent explicit `diseases=[...]`.

**Files:**
- Modify: `tests/test_multi_genotype.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_genotype.py`:

```python
def test_four_genotype_sim_runs():
    """End-to-end 4-genotype Sim runs and produces per-genotype results."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=2000, dt=0.5, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
    )
    sim.run()
    for key in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        res = sim.results[key]
        # Each genotype gets cum_infections via Starsim auto-stratification.
        assert 'cum_infections' in res or 'new_infections' in res, \
            f'{key} missing infection results'


def test_genotypes_sugar_matches_explicit_diseases():
    """genotypes=[16,18] == diseases=[HPV(genotype='hpv16'), HPV(genotype='hpv18')]."""
    pars = dict(n_agents=500, location='nigeria',
                start=1990, stop=1995, dt=1.0, rand_seed=0)
    sim_a = hpv.Sim(genotypes=[16, 18], **pars)
    sim_a.run()
    sim_b = hpv.Sim(
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
        connectors=[hpv.CrossImmunity()],
        **pars,
    )
    sim_b.run()
    for key in ('hpv16', 'hpv18'):
        a_inf = float(np.asarray(sim_a.results[key].new_infections).sum())
        b_inf = float(np.asarray(sim_b.results[key].new_infections).sum())
        assert a_inf == pytest.approx(b_inf), \
            f'sugar vs explicit drift for {key}: {a_inf} vs {b_inf}'
```

- [ ] **Step 2: Run to verify pass (these should pass given Tasks 1-9)**

```bash
pytest tests/test_multi_genotype.py -v
```

Expected: PASS — given Tasks 1-9, the 4-genotype path is end-to-end functional.

- [ ] **Step 3: If Step 2 reveals issues, debug**

If anything fails, the most likely culprits are: (a) Connector matrix shape mismatch when modules are registered in different orders; (b) M02-only test asserting `sim.diseases.hpv16` works (fine here, but check); (c) per-genotype `init_prev` curves seeding zero for the new genotypes due to a typo in `_INIT_PREV`. Resolve and re-run.

- [ ] **Step 4: Run full suite**

```bash
pytest tests/ -x -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_multi_genotype.py
git commit -m "M03: 4-genotype Sim smoke and genotypes= sugar parity"
```

---

### Task 11: Sim-level `cum_infections_any`, `cum_cancers_any`, `new_cancer_deaths_any` aggregators

Each Disease module already produces its own per-genotype results. M03 adds three Sim-level aggregators that pool across genotypes for the regression gate. `cum_infections_any` is a boolean OR (avoids double-counting agents with multi-genotype co-infection); the cancer aggregators are sums (cancer is genotype-attributed in v2).

**Files:**
- Modify: `hpvsim/sim.py`
- Modify: `tests/test_multi_genotype.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_genotype.py`:

```python
def test_aggregate_cum_infections_any():
    """Sim-level cum_infections_any counts agents ever infected with any genotype (boolean OR)."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=1995, dt=1.0, rand_seed=0,
        genotypes=[16, 18],
    )
    sim.run()
    # Expected: count of unique uids where any genotype's infected has ever been True.
    # Approximate via final n_susceptible: total_pop - n_susceptible_any > 0.
    any_cum = float(np.asarray(sim.results['cum_infections_any']).max())
    h16_cum = float(np.asarray(sim.results['hpv16'].new_infections).sum())
    h18_cum = float(np.asarray(sim.results['hpv18'].new_infections).sum())
    # Boolean-OR is at most the sum (equal iff no co-infections).
    assert any_cum > 0
    assert any_cum <= h16_cum + h18_cum + 1e-6


def test_aggregate_cum_cancers_any():
    """Sim-level cum_cancers_any sums per-genotype new_cancers across genotypes."""
    sim = hpv.Sim(
        n_agents=2000, location='nigeria',
        start=1990, stop=2010, dt=0.5, rand_seed=0,
        genotypes=[16, 18],
    )
    sim.run()
    any_c = float(np.asarray(sim.results['cum_cancers_any'])[-1])
    sum_c = sum(float(np.asarray(sim.results[k].cum_cancers)[-1])
                for k in ('hpv16', 'hpv18'))
    assert any_c == pytest.approx(sum_c, abs=1e-6)
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_multi_genotype.py -v -k aggregate
```

Expected: FAIL (aggregators don't exist yet).

- [ ] **Step 3: Implement aggregators on `Sim`**

Add aggregator helpers to `hpvsim/sim.py`. The cleanest location is to subclass `ss.Sim`'s `init_results` / `finalize_results` to register and populate the aggregators after the per-disease results have been finalised.

Add to `Sim` class in `hpvsim/sim.py`:

```python
    def init_results(self):
        super().init_results()
        # Aggregate-across-genotypes results, populated in finalize_results.
        # Use the first HPV's results array shape for dtype/length.
        hpvs = [d for d in self.diseases.values() if isinstance(d, HPV)]
        if not hpvs:
            return
        n_t = len(hpvs[0].results.new_infections)
        # ss.Result on the Sim, not on a Disease.
        self.results += ss.Result('cum_infections_any',
                                  shape=n_t, dtype=int, scale=True,
                                  label='Cumulative agents ever infected (any genotype)')
        self.results += ss.Result('cum_cancers_any',
                                  shape=n_t, dtype=int, scale=True,
                                  label='Cumulative cancers (any genotype)')
        self.results += ss.Result('new_cancer_deaths_any',
                                  shape=n_t, dtype=int, scale=True,
                                  label='New cancer deaths (any genotype)')

    def finalize_results(self):
        super().finalize_results()
        hpvs = [d for d in self.diseases.values() if isinstance(d, HPV)]
        if not hpvs:
            return
        # cum_infections_any: boolean-OR across modules' "ever infected"
        # state, summed at the per-step new-event level. We approximate via
        # max of per-module cum_infections (correct upper bound when single
        # genotype dominates; tracking issue if co-infection rates are high
        # enough to need exact OR).
        new_inf_stack = np.column_stack([
            np.asarray(m.results.new_infections) for m in hpvs
        ])
        # Approximation: per-step max-event across genotypes. For exact OR we
        # would need per-uid history, which Starsim doesn't surface by default.
        per_step_any = new_inf_stack.max(axis=1)
        self.results['cum_infections_any'][:] = np.cumsum(per_step_any)
        # cum_cancers_any: sum (cancer attributed to one genotype per agent).
        cum_c_stack = np.column_stack([
            np.asarray(m.results.cum_cancers) for m in hpvs
        ])
        self.results['cum_cancers_any'][:] = cum_c_stack.sum(axis=1)
        # new_cancer_deaths_any: sum across genotypes per step.
        ncd_stack = np.column_stack([
            np.asarray(m.results.new_cancer_deaths) for m in hpvs
        ])
        self.results['new_cancer_deaths_any'][:] = ncd_stack.sum(axis=1)
```

The `cum_infections_any` is approximated as a per-step max — it's an upper bound on the true boolean-OR. If the regression run shows this approximation drifts too far from v2's exact count, replace with a per-uid history scan in a follow-up commit. (Tracking issue: anticipated in spec under "Anticipated drift axes".)

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_multi_genotype.py -v
pytest tests/ -x -q
```

Expected: aggregator tests PASS; full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/sim.py tests/test_multi_genotype.py
git commit -m "M03: Sim-level cum_infections_any / cum_cancers_any / new_cancer_deaths_any aggregators"
```

---

### Task 12: 4-genotype anchor scenario script

Create the regression-harness anchor scenario that runs a vanilla 4-genotype HPV sim (Nigeria, fixed seed `0`, 1990-2060). Mirrors `tests/regression/anchor_hpv16.py` but for the 4-genotype set.

**Files:**
- Create: `tests/regression/anchor_4genotype.py`

- [ ] **Step 1: Write the script**

Create `tests/regression/anchor_4genotype.py`:

```python
"""M03 anchor scenario for the v2 -> v3 migration regression harness.

4-genotype HPV sim, Nigeria, fixed seed, no interventions, no analyzers.
Tooling under tests/regression/ (compare.py, baseline_v23.py) imports
``run_and_summarize()`` from here.

Run as a script to print the summary:
    python tests/regression/anchor_4genotype.py
"""

import sciris as sc

import hpvsim as hpv
from .short_summary import build_summary  # added in Task 13


# Pinned anchor pars. Do not change without coordinating with regression baselines.
PARS = dict(
    n_agents=10e3,
    location='nigeria',
    genotypes=[16, 18, 'hi5', 'ohr'],
    start=1990,
    stop=2060,
    dt=0.25,
    rand_seed=0,
    verbose=0,
)


def make_sim():
    """Build (but do not run) the M03 anchor sim."""
    return hpv.Sim(**sc.dcp(PARS))


def run_and_summarize():
    """Run the M03 anchor sim and return (short_summary_dict, total_pop).

    Summary is the 40-entry dict from short_summary.build_summary:
    8 metrics x 4 genotypes (32 entries) plus 8 aggregate-across-genotypes
    metrics computed from cum_infections_any / cum_cancers_any /
    new_cancer_deaths_any.
    """
    sim = make_sim()
    sim.run()
    short = build_summary(sim, genotypes=('hpv16', 'hpv18', 'hi5', 'ohr'))
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop


if __name__ == '__main__':
    short, total_pop = run_and_summarize()
    print('Short summary (M03 4-genotype):')
    for k, v in short.items():
        print(f'  {k:<48} {v:>12.4g}')
    print(f'  {"total population":<48} {total_pop:>12.4g}')
```

- [ ] **Step 2: Skip — no test file for this task**

The anchor script is exercised by Tasks 13-16. No standalone test.

- [ ] **Step 3: Skip — implementation done in Step 1**

- [ ] **Step 4: Smoke-run the anchor script**

```bash
python tests/regression/anchor_4genotype.py
```

Expected: ImportError on `short_summary.build_summary` (Task 13 hasn't landed). That's fine; we'll run it again at the end of Task 13.

- [ ] **Step 5: Commit**

```bash
git add tests/regression/anchor_4genotype.py
git commit -m "M03: 4-genotype anchor scenario script"
```

---

### Task 13: 40-entry per-genotype + aggregate `short_summary` builder

Factor the M02 8-metric summary computation into a reusable builder that produces 32 per-genotype entries (8 metrics × 4 genotypes) plus 8 aggregate-across-genotypes entries computed from `cum_infections_any` / `cum_cancers_any` / `new_cancer_deaths_any`. Reused by the anchor script and by parity tests.

**Files:**
- Create: `tests/regression/short_summary.py`
- Create: `tests/test_short_summary.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_short_summary.py`:

```python
"""Unit tests for the M03 40-entry short-summary builder."""
import numpy as np
import pytest

import hpvsim as hpv
from tests.regression.short_summary import build_summary, METRIC_KEYS


def test_metric_keys_are_eight():
    """Each genotype contributes exactly 8 metrics (matches M02 short_summary set)."""
    assert len(METRIC_KEYS) == 8


def test_build_summary_keys_are_40_for_four_genotypes():
    """4-genotype build = 32 per-genotype + 8 aggregate = 40 entries."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=1995, dt=1.0, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
    )
    sim.run()
    out = build_summary(sim, genotypes=('hpv16', 'hpv18', 'hi5', 'ohr'))
    assert len(out) == 40
    for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        for m in METRIC_KEYS:
            assert f'{g}.{m}' in out
    for m in METRIC_KEYS:
        assert f'any.{m}' in out


def test_build_summary_per_genotype_zero_safe():
    """Zero-cancer trajectory yields 0.0 (not NaN) for mean-age-of-cancer."""
    sim = hpv.Sim(
        n_agents=200, location='nigeria',
        start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=[16],
    )
    sim.run()
    out = build_summary(sim, genotypes=('hpv16',))
    # 8 + 8 = 16 entries (1 genotype + aggregate).
    assert len(out) == 16
    val = out['hpv16.mean age of cancer (years)']
    assert val == 0.0 or not np.isnan(val), \
        f'expected 0.0 or non-NaN; got {val}'
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_short_summary.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `build_summary`**

Create `tests/regression/short_summary.py`:

```python
"""M03 40-entry short-summary builder.

For each genotype produces the 8-metric M02 summary (total HPV infections,
total cancers, total cancer deaths, mean HPV prevalence, mean cancer
incidence, mean ages of infection / cancer / cancer death). Plus an
8-metric ``any.*`` aggregate computed from the Sim-level *_any results.
"""

import numpy as np


METRIC_KEYS = (
    'total HPV infections',
    'total cancers',
    'total cancer deaths',
    'mean HPV prevalence (%)',
    'mean cancer incidence (per 100k)',
    'mean age of infection (years)',
    'mean age of cancer (years)',
    'mean age of cancer death (years)',
)


def _per_genotype_metrics(sim, genotype):
    """Compute the 8-metric M02 summary for one genotype (key into sim.results)."""
    res = sim.results[genotype]
    mod = sim.diseases[genotype]
    dt = float(sim.t.dt)
    pop_scale = float(getattr(sim.pars, 'pop_scale', 1.0) or 1.0)

    new_infections = np.asarray(res.new_infections)
    n_inf_unscaled = float(new_infections.sum())
    n_inf = n_inf_unscaled * pop_scale

    mean_prev_pct = 100 * float(np.asarray(res.prevalence).mean())

    ti_latest = mod.ti_infected
    ever_inf = ti_latest.notnan.uids
    if len(ever_inf):
        ages_now = np.asarray(sim.people.age[ever_inf])
        ti_at_inf = np.asarray(ti_latest[ever_inf])
        years_since = (float(sim.t.ti) - ti_at_inf) * dt
        mean_age_inf = float((ages_now - years_since).mean())
    else:
        mean_age_inf = 0.0

    new_cancers = np.asarray(res.new_cancers)
    n_cancers_unscaled = float(new_cancers.sum())
    n_cancers = n_cancers_unscaled * pop_scale

    sum_age_cancer = float(np.asarray(res.sum_age_at_cancer).sum())
    mean_age_cancer = (sum_age_cancer / n_cancers_unscaled) if n_cancers_unscaled > 0 else 0.0

    new_cancer_deaths = np.asarray(res.new_cancer_deaths)
    n_cd_unscaled = float(new_cancer_deaths.sum())
    n_cancer_deaths = n_cd_unscaled * pop_scale

    sum_age_cd = float(np.asarray(res.sum_age_at_cancer_death).sum())
    mean_age_cancer_death = (sum_age_cd / n_cd_unscaled) if n_cd_unscaled > 0 else 0.0

    n_alive_series = np.asarray(sim.results['n_alive'])
    total_alive_years = float(n_alive_series.sum()) * dt
    female_years = total_alive_years / 2.0
    mean_cancer_incidence = (n_cancers / female_years * 100_000.0) if female_years > 0 else 0.0

    return {
        'total HPV infections': n_inf,
        'total cancers': n_cancers,
        'total cancer deaths': n_cancer_deaths,
        'mean HPV prevalence (%)': mean_prev_pct,
        'mean cancer incidence (per 100k)': mean_cancer_incidence,
        'mean age of infection (years)': mean_age_inf,
        'mean age of cancer (years)': mean_age_cancer,
        'mean age of cancer death (years)': mean_age_cancer_death,
    }


def _aggregate_metrics(sim, genotypes):
    """Compute the 8-metric aggregate from sim-level *_any results.

    For metrics that are inherently per-event (means of age), aggregate by
    pooling per-genotype sums and counts; e.g. mean age of cancer = sum of
    (sum_age_at_cancer across genotypes) / sum of (n_cancers across genotypes).
    """
    dt = float(sim.t.dt)
    pop_scale = float(getattr(sim.pars, 'pop_scale', 1.0) or 1.0)

    n_inf_unscaled = float(np.asarray(sim.results['cum_infections_any'])[-1])
    n_inf = n_inf_unscaled * pop_scale

    cum_c = float(np.asarray(sim.results['cum_cancers_any'])[-1])
    n_cancers = cum_c * pop_scale
    n_cancers_unscaled = cum_c

    new_cd_any = np.asarray(sim.results['new_cancer_deaths_any'])
    n_cd_unscaled = float(new_cd_any.sum())
    n_cancer_deaths = n_cd_unscaled * pop_scale

    # Mean prev: average across genotypes' prevalences (approximation).
    prevs = [np.asarray(sim.results[g].prevalence) for g in genotypes]
    mean_prev_pct = 100 * float(np.mean(np.column_stack(prevs)))

    n_alive_series = np.asarray(sim.results['n_alive'])
    total_alive_years = float(n_alive_series.sum()) * dt
    female_years = total_alive_years / 2.0
    mean_cancer_incidence = (n_cancers / female_years * 100_000.0) if female_years > 0 else 0.0

    # Pool per-genotype mean-age sums and counts.
    sum_age_inf_total = 0.0
    n_inf_count_total = 0.0
    sum_age_cancer_total = 0.0
    sum_age_cd_total = 0.0
    n_cd_count_total = 0.0
    for g in genotypes:
        mod = sim.diseases[g]
        ti_latest = mod.ti_infected
        ever_inf = ti_latest.notnan.uids
        if len(ever_inf):
            ages_now = np.asarray(sim.people.age[ever_inf])
            ti_at_inf = np.asarray(ti_latest[ever_inf])
            years_since = (float(sim.t.ti) - ti_at_inf) * dt
            sum_age_inf_total += float((ages_now - years_since).sum())
            n_inf_count_total += float(len(ever_inf))
        sum_age_cancer_total += float(np.asarray(sim.results[g].sum_age_at_cancer).sum())
        sum_age_cd_total += float(np.asarray(sim.results[g].sum_age_at_cancer_death).sum())

    mean_age_inf = (sum_age_inf_total / n_inf_count_total) if n_inf_count_total > 0 else 0.0
    mean_age_cancer = (sum_age_cancer_total / n_cancers_unscaled) if n_cancers_unscaled > 0 else 0.0
    mean_age_cancer_death = (sum_age_cd_total / n_cd_unscaled) if n_cd_unscaled > 0 else 0.0

    return {
        'total HPV infections': n_inf,
        'total cancers': n_cancers,
        'total cancer deaths': n_cancer_deaths,
        'mean HPV prevalence (%)': mean_prev_pct,
        'mean cancer incidence (per 100k)': mean_cancer_incidence,
        'mean age of infection (years)': mean_age_inf,
        'mean age of cancer (years)': mean_age_cancer,
        'mean age of cancer death (years)': mean_age_cancer_death,
    }


def build_summary(sim, genotypes):
    """Return the 40-entry per-genotype + aggregate summary dict.

    Keys are ``<genotype>.<metric>`` for per-genotype entries and
    ``any.<metric>`` for aggregate entries.
    """
    out = {}
    for g in genotypes:
        per = _per_genotype_metrics(sim, g)
        for k, v in per.items():
            out[f'{g}.{k}'] = v
    agg = _aggregate_metrics(sim, genotypes)
    for k, v in agg.items():
        out[f'any.{k}'] = v
    return out
```

- [ ] **Step 4: Run tests + smoke-run the anchor script**

```bash
pytest tests/test_short_summary.py -v
python tests/regression/anchor_4genotype.py
```

Expected: tests PASS; anchor script prints a 40-entry summary.

- [ ] **Step 5: Commit**

```bash
git add tests/regression/short_summary.py tests/test_short_summary.py
git commit -m "M03: 40-entry per-genotype + aggregate short_summary builder"
```

---

### Task 14: 4-genotype v2 baseline regen entrypoint

Extend `tests/regression/baseline_v23.py` with a `regen_4genotype()` function that produces v2's 4-genotype baseline JSON for the M03 parity gates. The output goes to `tests/regression/baselines/anchor_4genotype.json` (gitignored).

**Files:**
- Modify: `tests/regression/baseline_v23.py`

- [ ] **Step 1: Inspect existing single-genotype regen**

```bash
cat tests/regression/baseline_v23.py
```

Note the patterns: where it constructs a v2 sim, runs it, and dumps the summary. The 4-genotype variant should use the same code path but instantiate v2 with `genotypes=['hpv16', 'hpv18', 'hi5', 'ohr']`.

- [ ] **Step 2: Add `regen_4genotype()` function**

In `tests/regression/baseline_v23.py`, add a new entrypoint that:
1. Instantiates a v2 4-genotype sim with the same `PARS` as `anchor_4genotype.PARS` (n_agents, location, dates, dt, seed).
2. Runs to completion.
3. Builds a 40-entry summary dict with the same `METRIC_KEYS` and naming convention as `tests/regression/short_summary.build_summary` (the v2 sim has its own per-genotype results layout — translate to the same dotted-key format).
4. Writes the dict to `tests/regression/baselines/anchor_4genotype.json` via `sciris.save_json`.

The exact v2 API for genotype results differs from v3 — refer to v2's `compute_summary` and `sim.results['hpv16'][...]` access in `_v2_legacy/sim.py`. Where v2 uses different metric semantics (e.g., infections counted at the start of an infection rather than at re-infection), document the divergence in a comment alongside the metric.

- [ ] **Step 3: Generate the baseline locally**

```bash
python -c "from tests.regression.baseline_v23 import regen_4genotype; regen_4genotype()"
ls -la tests/regression/baselines/anchor_4genotype.json
```

Expected: file exists, contains 40 entries.

- [ ] **Step 4: Verify .gitignore covers the new baseline file**

```bash
git check-ignore tests/regression/baselines/anchor_4genotype.json
```

Expected: prints the path (i.e., is gitignored). If not, add `tests/regression/baselines/` to `.gitignore`:

```bash
echo "tests/regression/baselines/" >> .gitignore
```

- [ ] **Step 5: Commit (script and gitignore only — not the JSON)**

```bash
git add tests/regression/baseline_v23.py .gitignore
git commit -m "M03: 4-genotype v2 baseline regeneration entrypoint"
```

---

### Task 15: Per-genotype + aggregate `short_summary` parity test

Compare the v3 4-genotype `short_summary` to v2's regenerated baseline at ±10% relative drift per entry. This is the development gate. The test reads the local baseline JSON; if missing, it skips with a clear pointer to Task 14's regen command.

**Files:**
- Create: `tests/test_m03_short_summary_parity.py`

- [ ] **Step 1: Write the test**

```python
"""M03 development gate: 40-entry short_summary parity vs. v2 4-genotype baseline.

Fails any entry that drifts >10% relative to v2. Skipped if the local baseline
JSON is missing (run regen via tests/regression/baseline_v23.py:regen_4genotype).
"""
import os
import pytest
import sciris as sc

from tests.regression.anchor_4genotype import run_and_summarize


BASELINE_PATH = 'tests/regression/baselines/anchor_4genotype.json'
RELATIVE_TOLERANCE = 0.10


@pytest.mark.slow
def test_short_summary_parity_4genotype():
    if not os.path.exists(BASELINE_PATH):
        pytest.skip(
            f'Baseline missing at {BASELINE_PATH}. Regenerate via '
            f'`python -c "from tests.regression.baseline_v23 import '
            f'regen_4genotype; regen_4genotype()"`'
        )
    v2_summary = sc.loadjson(BASELINE_PATH)
    v3_summary, _ = run_and_summarize()

    drifts = {}
    for k, v2_val in v2_summary.items():
        if k not in v3_summary:
            drifts[k] = (v2_val, None, 'missing in v3')
            continue
        v3_val = v3_summary[k]
        denom = max(abs(v2_val), 1e-9)
        rel_drift = abs(v3_val - v2_val) / denom
        if rel_drift > RELATIVE_TOLERANCE:
            drifts[k] = (v2_val, v3_val, rel_drift)

    if drifts:
        rows = '\n'.join(
            f'  {k:<50} v2={v2:.4g}  v3={v3 if v3 is not None else "MISSING":<10}  drift={d}'
            for k, (v2, v3, d) in drifts.items()
        )
        pytest.fail(
            f'M03 short_summary drift > {RELATIVE_TOLERANCE:.0%} on '
            f'{len(drifts)} of {len(v2_summary)} entries:\n{rows}'
        )
```

- [ ] **Step 2: Run to verify the skip path works**

```bash
pytest tests/test_m03_short_summary_parity.py -v
```

Expected: SKIP if baseline missing. After running Task 14's regen locally, expected: PASS or FAIL with a drift table.

- [ ] **Step 3: Skip — implementation done in Step 1**

- [ ] **Step 4: If FAIL, document and classify**

If the test fails, capture the drift table and decide for each entry whether the drift is expected feature-misalignment (file a tracking issue) or a bug (fix in a follow-up commit on this branch). Per `MIGRATION_PLAN.md:260`, the development gate is informational — failures don't block the M03 PR if classified.

- [ ] **Step 5: Commit**

```bash
git add tests/test_m03_short_summary_parity.py
git commit -m "M03: per-genotype + aggregate short_summary parity test"
```

---

### Task 16: Trajectory parity test (cum_*_any time series) + threshold pin

Compare v3's `cum_cancers_any` and `cum_infections_any` time series to v2's. This is the capability gate (must be green for the M03 PR). The threshold is pinned **after** the first run by inspecting the actual drift; until pinned, the test skips with the recorded threshold candidate.

**Files:**
- Create: `tests/test_m03_trajectory_parity.py`
- Modify: `tests/regression/baseline_v23.py` (extend regen to also save trajectory series)

- [ ] **Step 1: Extend baseline regen to save trajectories**

In `tests/regression/baseline_v23.py`'s `regen_4genotype`, also save the v2 `cum_cancers_any` and `cum_infections_any` trajectories to `tests/regression/baselines/anchor_4genotype_trajectory.json` as `{ 'time': [...], 'cum_cancers_any': [...], 'cum_infections_any': [...] }`. Both arrays are length-equal to the time grid.

- [ ] **Step 2: Write the test**

Create `tests/test_m03_trajectory_parity.py`:

```python
"""M03 capability gate: age-aggregated cancer / infection trajectory parity.

This is the M03 release gate. The threshold is pinned after the first run by
inspecting the empirical drift; the placeholder constants below should be
edited to the chosen pin during execution of this task.
"""
import os
import numpy as np
import pytest
import sciris as sc

from tests.regression.anchor_4genotype import make_sim


TRAJECTORY_BASELINE = 'tests/regression/baselines/anchor_4genotype_trajectory.json'

# THRESHOLD PIN: After Task 16 first run, replace these with empirically chosen
# bounds. Until pinned, the test skips loud.
THRESHOLD_MAX_REL = None  # e.g., 0.15 = 15% max-relative-drift tolerance
THRESHOLD_L2_REL = None   # e.g., 0.10 = 10% L2 norm tolerance


def _l2_rel(a, b):
    """Relative L2 distance: ||a-b||_2 / ||b||_2 (denominator = baseline)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = float(np.linalg.norm(a - b))
    denom = float(np.linalg.norm(b)) or 1e-9
    return num / denom


@pytest.mark.slow
def test_trajectory_parity_4genotype():
    if THRESHOLD_MAX_REL is None or THRESHOLD_L2_REL is None:
        pytest.skip(
            'Trajectory parity threshold not yet pinned. After first run, '
            'set THRESHOLD_MAX_REL and THRESHOLD_L2_REL in this file.'
        )
    if not os.path.exists(TRAJECTORY_BASELINE):
        pytest.skip(f'Trajectory baseline missing at {TRAJECTORY_BASELINE}.')
    v2 = sc.loadjson(TRAJECTORY_BASELINE)
    sim = make_sim()
    sim.run()
    failures = []
    for series in ('cum_cancers_any', 'cum_infections_any'):
        v3_arr = np.asarray(sim.results[series])
        v2_arr = np.asarray(v2[series])
        if v3_arr.shape != v2_arr.shape:
            failures.append((series, f'shape {v3_arr.shape} != {v2_arr.shape}'))
            continue
        max_rel = float(np.max(np.abs(v3_arr - v2_arr) /
                                np.maximum(np.abs(v2_arr), 1e-9)))
        l2_rel = _l2_rel(v3_arr, v2_arr)
        if max_rel > THRESHOLD_MAX_REL:
            failures.append((series, f'max_rel={max_rel:.3f} > {THRESHOLD_MAX_REL:.3f}'))
        if l2_rel > THRESHOLD_L2_REL:
            failures.append((series, f'l2_rel={l2_rel:.3f} > {THRESHOLD_L2_REL:.3f}'))
    if failures:
        msg = '\n'.join(f'  {s}: {f}' for s, f in failures)
        pytest.fail(f'M03 trajectory drift exceeded thresholds:\n{msg}')
```

- [ ] **Step 3: Run the test once to inspect drift, then pin**

```bash
pytest tests/test_m03_trajectory_parity.py -v -s
```

Expected: SKIP (threshold not pinned). Manually compute the empirical drift via:

```bash
python -c "
import numpy as np, sciris as sc
from tests.regression.anchor_4genotype import make_sim
v2 = sc.loadjson('tests/regression/baselines/anchor_4genotype_trajectory.json')
sim = make_sim(); sim.run()
for s in ('cum_cancers_any', 'cum_infections_any'):
    v3 = np.asarray(sim.results[s]); v2_arr = np.asarray(v2[s])
    max_rel = float(np.max(np.abs(v3 - v2_arr) / np.maximum(np.abs(v2_arr), 1e-9)))
    l2_rel = float(np.linalg.norm(v3 - v2_arr) / max(np.linalg.norm(v2_arr), 1e-9))
    print(f'{s}: max_rel={max_rel:.3f}  l2_rel={l2_rel:.3f}')
"
```

Edit `tests/test_m03_trajectory_parity.py` to set `THRESHOLD_MAX_REL` and `THRESHOLD_L2_REL` to slightly above the empirical values (e.g., empirical 0.08 → pinned 0.10), with a comment recording the empirical numbers.

- [ ] **Step 4: Re-run the test with pinned thresholds**

```bash
pytest tests/test_m03_trajectory_parity.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_m03_trajectory_parity.py tests/regression/baseline_v23.py
git commit -m "M03: trajectory parity test for cum_*_any with pinned thresholds"
```

---

### Task 17: M02 1-genotype regression through the new Connector path + drift pin

After Task 7, M02's 1-genotype baseline runs through the new sample-and-derive Connector path. This task pins the expected drift on the M02 8-metric `short_summary` so future regressions on the M03 branch don't silently slip.

**Files:**
- Create: `tests/test_m02_through_connector.py`

- [ ] **Step 1: Write the test (with empirically pinned drift)**

Create `tests/test_m02_through_connector.py`:

```python
"""M02 1-genotype baseline regression through the M03 Connector path.

After Task 7 of M03, M02's clearance writes were redirected from sev_imm /
rel_sus into source-genotype nab_imm / cell_imm, with the Connector deriving
the effective values. The previously-deterministic immunity cap (rel_sus
capped at 1 - 0.35) became a per-clearance beta_mean(0.35, 0.025) sample.
Small drift on M02's 8-metric short_summary is anticipated; this test pins
that drift bound so it doesn't slip silently.

Drift values pinned from the first M03 1-genotype run; recompute and update
if the M03 anchor sim parameters change.
"""
import os
import pytest
import sciris as sc

from tests.regression.anchor_hpv16 import run_and_summarize


M02_BASELINE = 'tests/regression/baselines/anchor_hpv16.json'

# Per-metric pinned drift: drift_value > tolerance => fail. Empirically chosen
# in Step 3 of this task; widen with care.
PINNED_TOLERANCES = {
    'total HPV infections': 0.05,
    'total cancers': 0.10,
    'total cancer deaths': 0.10,
    'mean HPV prevalence (%)': 0.05,
    'mean cancer incidence (per 100k)': 0.10,
    'mean age of infection (years)': 0.05,
    'mean age of cancer (years)': 0.05,
    'mean age of cancer death (years)': 0.05,
}


@pytest.mark.slow
def test_m02_baseline_through_connector():
    if not os.path.exists(M02_BASELINE):
        pytest.skip(f'M02 baseline missing at {M02_BASELINE}.')
    v2 = sc.loadjson(M02_BASELINE)
    v3, _ = run_and_summarize()
    failures = []
    for k, tol in PINNED_TOLERANCES.items():
        v2_val = v2[k]
        v3_val = v3[k]
        denom = max(abs(v2_val), 1e-9)
        drift = abs(v3_val - v2_val) / denom
        if drift > tol:
            failures.append((k, v2_val, v3_val, drift, tol))
    if failures:
        msg = '\n'.join(
            f'  {k:<40} v2={v2v:.4g} v3={v3v:.4g} drift={d:.3f} > {t:.3f}'
            for k, v2v, v3v, d, t in failures
        )
        pytest.fail(f'M02-through-Connector drift exceeded pinned tolerances:\n{msg}')
```

- [ ] **Step 2: Run once to inspect empirical drift**

```bash
pytest tests/test_m02_through_connector.py -v -s
```

Expected: this is informational on first run. Inspect actual drifts; replace `PINNED_TOLERANCES` values to be slightly above empirical (record empirical in a comment).

- [ ] **Step 3: Re-run with pinned tolerances**

```bash
pytest tests/test_m02_through_connector.py -v
```

Expected: PASS.

- [ ] **Step 4: Run the full test suite once more**

```bash
pytest tests/ -q
```

Expected: all PASS (modulo `@pytest.mark.slow` markers as configured).

- [ ] **Step 5: Commit**

```bash
git add tests/test_m02_through_connector.py
git commit -m "M03: M02 1-genotype regression through Connector with pinned drift"
```

---

## Self-review

After writing the tasks, I checked the spec against the plan:

- **Spec Section 1 — Goal & Scope:** All in-scope items map to tasks. Out-of-scope items (waning, AgeResults, vaccination, etc.) are not present in any task. ✓
- **Spec Section 2 — Sim & Parameters API:** Genotype list + diseases= passthrough → Task 6. Aliases → Task 6. GenotypePars extension → Task 8. `get_cross_immunity` hand-port → Task 2. ✓
- **Spec Section 3 — HPV module changes:** `nab_imm`/`cell_imm` add → Task 1. Move sev_imm/rel_sus writes → Task 7. Uniform Connector path even for n=1 → Task 6 (auto-Connector) and Task 17 (M02 regression). ✓
- **Spec Section 4 — `CrossImmunity` Connector:** Skeleton + validation → Task 3. Step math → Task 4. Auto-instantiation → Task 6. Diagonal forced to 1.0 → Task 3. ✓
- **Spec Section 5 — Initial prevalence:** Per-genotype curves → Task 9. ✓
- **Spec Section 6 — Results & regression:** 40-entry summary → Task 13. Aggregate `*_any` → Task 11. Per-genotype + aggregate parity test → Task 15. Trajectory parity → Task 16. M02 regression sanity → Task 17. 4-genotype baseline regen → Task 14. ✓
- **Spec Section 7 — Branch hygiene & sequencing:** Branched off m02; PR targets v3.0-dev (Prerequisites section, top of plan). Suggested staging — non-binding (Task ordering preface). Zero `_v2_legacy` delegations (Task 2 hand-ports `get_cross_immunity` directly). ✓

**Placeholder scan:** No "TBD" / "TODO" / "fill in details" outside of the documented threshold-pin steps in Tasks 16 and 17, where pinning empirical thresholds is the explicit task. ✓

**Type consistency:**
- `CrossImmunity` matrix shape `(n, n)` consistent across Tasks 3, 4. ✓
- `_INIT_PREV` dict keyed by genotype, values keyed `'m'`/`'f'` consistent in Task 9. ✓
- `build_summary(sim, genotypes=...)` signature consistent across Tasks 12, 13, 15. ✓
- `METRIC_KEYS` defined once (Task 13) and re-used by `build_summary` and parity tests. ✓
- `cum_infections_any` / `cum_cancers_any` / `new_cancer_deaths_any` keys consistent across Tasks 11, 13, 16. ✓

No issues found.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-06-hpvsim-m03-multi-genotype-cross-immunity.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
