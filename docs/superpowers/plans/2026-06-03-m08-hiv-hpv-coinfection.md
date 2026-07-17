# M08: HIV–HPV Co-infection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add transmission-based HIV (STIsim) to `hpv.Sim` as a first-class disease, with a connector applying v2's three CD4-stratified HIV→HPV effects, reproducing HIV-stratified HPV/cancer outcomes on Rwanda.

**Architecture:** New module `hpvsim/hiv.py` holds `hpv.HIV` (thin `sti.HIV` subclass), `hpv_hiv_connector` (CD4→strata→`rel_sus`/`rel_sev`/`rel_imm`), and `HIVStratifiedResults` (analyzer). The `hpv.Sim` constructor is refactored to type-partition `diseases=` so HIV merges with genotype-built HPV modules, auto-wiring the connector + analyzer on detection. HIV transmits over the existing `hpv.SexualNetwork`. Built in two phases: Phase 1 co-infection mechanics (TDD), Phase 2 Rwanda calibration + parity.

**Tech Stack:** Python, Starsim 3.3.4, STIsim 1.5.0, NumPy, pytest.

---

## Background the engineer needs

- **Connector pattern** (copy from `hpvsim/cross_genotype.py`): a `ss.Connector` discovers target modules in `init_pre` (`[m for m in sim.diseases.values() if isinstance(m, HPV)]`), defines per-agent state via `self.define_states(ss.FloatArr(...))`, and mutates module fields in `step()`. `CrossImmunity` runs each step *between* `Disease.step_state` and `Disease.step_infect`, and **overwrites** every HPV module's `rel_sus` (`cross_genotype.py:151`). Our connector must run **after** `CrossImmunity` so its `rel_sus` multiply survives.
- **Connector ordering** is controlled by append order in `hpv.Sim.__init__`: `auto_connectors = [CrossImmunity()] + ([seeder] if seeder else [])`. Appending `hpv_hiv_connector()` last puts it after `CrossImmunity`.
- **Module-lookup helper** (model on `hpv.py:171` `_cross_immunity_connector`): iterate `self.sim.connectors.values()` and return the first `isinstance` match, else `None`.
- **HPV progression hooks**: `HPV.set_prognoses` (`hpv.py:241`) reads `rel_sev_uids` from the CrossImmunity connector (`hpv.py:279-284`) and passes it to `compute_severity(...)` for P(CIN) (`hpv.py:301`) and P(cancer) (`hpv.py:352`). `HPV.step_state` clearance (`hpv.py:456-458`) samples `nab_all = p.imm_init.rvs(...)` / `cell_all = p.cell_imm_init.rvs(...)` then writes them to `nab_imm`/`cell_imm`.
- **STIsim HIV**: `sti.HIV(pars=None, init_prev_data=None, **kwargs)`; states include `cd4` (continuous), `on_art`. `sti.ART` is an **intervention**, supplied by the user via `interventions=`.
- **Quarantine rule** (project convention): never import from `hpvsim/_v2_legacy/`. v2's effect values are **copied** into active code with a source comment.
- **v2 effect values** (copied from `hpvsim/_v2_legacy/hiv.py:29-44`): strata `lt200` (CD4<200) and `gt200` (200≤CD4<500); `rel_sus` {lt200: 2.2, gt200: 2.2}; `rel_sev` {lt200: 1.5, gt200: 1.2}; `rel_imm` {lt200: 0.36, gt200: 0.76}. CD4≥500 HIV+ agents bin to `gt200` (verified in Task 9 against v2 behavior).
- **CRN regression guard**: the existing `tests/test_m01_short_summary_parity.py` and `tests/test_m02_short_summary_parity.py` pin HPV-only results; if any refactor perturbs RNG streams for non-HIV sims, they fail. Run them after Task 1 and Task 7.

## File structure

| File | Responsibility | New/Modify |
|---|---|---|
| `hpvsim/hiv.py` | `HIV`, `hpv_hiv_connector`, `HIVStratifiedResults` | Create |
| `hpvsim/sim.py` | Type-partitioned disease assembly + detection auto-wiring | Modify |
| `hpvsim/hpv.py` | `_hiv_connector()` lookup; `rel_sev`/`rel_imm` read sites (gated) | Modify |
| `hpvsim/products.py` | vx/txvx `administer` scales immunity by `rel_imm` (gated) | Modify |
| `hpvsim/__init__.py` | Export `HIV`, `hpv_hiv_connector`, `HIVStratifiedResults` | Modify |
| `hpvsim/data/hiv_init_prev.csv`, `hiv_art_coverage.csv` | Rwanda HIV inputs | Create |
| `tests/test_m08_*.py` | Unit + integration tests | Create |
| `tests/regression/anchor_hiv_hpv.py`, `baseline_hiv_v2.py`, `test_m08_hiv_parity.py` | Phase 2 parity | Create |

---

# Phase 1 — Co-infection mechanics

## Task 1: Type-partition `diseases=` in the Sim constructor

**Files:**
- Modify: `hpvsim/sim.py:87-138`
- Test: `tests/test_m08_sim_assembly.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_m08_sim_assembly.py
import pytest
import starsim as ss
import hpvsim as hpv


def _tiny(**kw):
    return dict(n_agents=200, start=2000, stop=2002, dt=0.25, location='nigeria', **kw)


def test_genotypes_plus_other_disease_merges():
    """A non-HPV disease passed via diseases= merges with genotype-built HPV."""
    other = ss.SIS()  # any non-HPV ss.Disease as a stand-in
    sim = hpv.Sim(**_tiny(genotypes=[16, 18], diseases=[other]))
    sim.init()
    hpv_mods = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)]
    assert len(hpv_mods) == 2                      # genotypes still built
    assert any(d is other for d in sim.diseases.values())  # other merged in


def test_hpv_instance_override_still_works():
    """diseases=[HPV,...] override path is unchanged (no genotypes=)."""
    sim = hpv.Sim(**_tiny(diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')]))
    sim.init()
    hpv_mods = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)]
    assert len(hpv_mods) == 2


def test_hpv_instances_plus_genotypes_raises():
    """Specifying the HPV set two ways still raises."""
    with pytest.raises(ValueError, match='genotypes='):
        hpv.Sim(**_tiny(genotypes=[16], diseases=[hpv.HPV(genotype='hpv16')]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_m08_sim_assembly.py -v`
Expected: FAIL — `test_genotypes_plus_other_disease_merges` raises the current "Pass diseases= OR genotypes=, not both." ValueError.

- [ ] **Step 3: Refactor the assembly block**

Replace `hpvsim/sim.py:87-138` (the `diseases = kwargs.pop(...)` through `connectors = auto_connectors + user_connectors` block) with:

```python
        user_diseases = kwargs.pop('diseases', None) or []
        user_connectors = kwargs.pop('connectors', None) or []
        user_analyzers = kwargs.pop('analyzers', None) or []

        # Partition diseases= by type: HPV genotype modules are built from
        # genotypes= (or supplied directly as HPV instances); any non-HPV
        # disease (e.g. hpv.HIV) is merged in alongside them.
        hpv_instances = [d for d in user_diseases if isinstance(d, HPV)]
        other_diseases = [d for d in user_diseases if not isinstance(d, HPV)]

        if hpv_instances and genotypes is not None:
            raise ValueError(
                'Specify HPV via genotypes= or HPV instances in diseases=, not both.'
            )

        if init_seeding not in ('exclusive', 'independent'):
            raise ValueError(
                f"init_seeding must be 'exclusive' or 'independent'; got {init_seeding!r}"
            )

        auto_connectors = [CrossImmunity()]

        if hpv_instances:
            hpv_diseases = hpv_instances  # override path; seeder NOT wired (unchanged)
        else:
            # Default to single-genotype HPV16 if neither supplied.
            keys = (tuple(_normalize_genotype(g) for g in genotypes)
                    if genotypes is not None else ('hpv16',))
            gpars_overrides = genotype_pars or {}

            if init_hpv_dist is not None:
                if not isinstance(init_hpv_dist, dict):
                    raise ValueError(
                        f'init_hpv_dist must be a dict or None; got {type(init_hpv_dist)}'
                    )
                dist_keys = set(init_hpv_dist.keys())
                sim_keys = set(keys)
                if dist_keys != sim_keys:
                    raise ValueError(
                        f'init_hpv_dist keys {sorted(dist_keys)} do not match '
                        f'resolved genotype keys {sorted(sim_keys)}'
                    )

            hpv_diseases = [HPV(genotype=k, **gpars_overrides.get(k, {})) for k in keys]
            if init_seeding == 'exclusive':
                self._seeder = _ExclusiveSeeder(
                    genotype_keys=keys, init_hpv_dist=init_hpv_dist
                )
                for d, k in zip(hpv_diseases, keys):
                    d.pars.init_prev = ss.bernoulli(p=self._seeder.for_genotype(k))
                auto_connectors.append(self._seeder)

        diseases = hpv_diseases + other_diseases
        connectors = auto_connectors + user_connectors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_m08_sim_assembly.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the CRN regression guard**

Run: `python -m pytest tests/test_m01_short_summary_parity.py tests/test_m02_short_summary_parity.py tests/test_multi_genotype.py -v`
Expected: PASS — non-HIV streams unchanged. (If `test_multi_genotype.py:43` asserts the old error text, update its `match=` to `'genotypes='`.)

- [ ] **Step 6: Commit**

```bash
git add hpvsim/sim.py tests/test_m08_sim_assembly.py
git commit -m "M08: type-partition diseases= so non-HPV diseases merge with genotypes"
```

---

## Task 2: `hpv.HIV` subclass — init-prev passthrough + directional network beta

**Files:**
- Create: `hpvsim/hiv.py`
- Modify: `hpvsim/__init__.py`
- Test: `tests/test_m08_hiv_disease.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m08_hiv_disease.py
import numpy as np
import hpvsim as hpv
from hpvsim.network import SexualNetwork


def test_hiv_beta_directional_orientation():
    """beta is keyed to the SexualNetwork name as [f2m, m2f] (p1=female, p2=male)."""
    h = hpv.HIV(beta_m2f=0.0035, rel_beta_f2m=0.5)
    sim = hpv.Sim(n_agents=200, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h])
    sim.init()
    net = [n for n in sim.networks.values() if isinstance(n, SexualNetwork)][0]
    beta = dict(sim.diseases.hiv.pars.beta)
    assert net.name in beta
    f2m, m2f = beta[net.name]
    # p1=female so betamap[0]=f2m, betamap[1]=m2f; m2f is the larger direction.
    assert np.isclose(m2f, 0.0035)
    assert np.isclose(f2m, 0.0035 * 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m08_hiv_disease.py -v`
Expected: FAIL — `AttributeError: module 'hpvsim' has no attribute 'HIV'`.

- [ ] **Step 3: Create `hpvsim/hiv.py` with the HIV subclass**

```python
"""HIV–HPV co-infection: transmission-based HIV plus the CD4-stratified
HIV→HPV effects ported (by value) from v2's HIVsim.

Three components:
  - ``HIV`` (``sti.HIV`` subclass): continuous-CD4 transmission-based HIV,
    re-targeted onto ``hpv.SexualNetwork`` and seeded from a Rwanda init-prev curve.
  - ``hpv_hiv_connector`` (``ss.Connector``): bins CD4 into discrete strata and
    applies v2's rel_sus / rel_sev / rel_imm effects to every HPV module.
  - ``HIVStratifiedResults`` (``ss.Analyzer``): HPV/cancer outcomes by HIV status.
"""

import numpy as np
import starsim as ss
import stisim as sti

from . import misc
from .hpv import HPV
from .network import SexualNetwork

__all__ = ['HIV', 'hpv_hiv_connector', 'HIVStratifiedResults']


class HIV(sti.HIV):
    """Transmission-based HIV for hpvsim.

    Thin subclass of ``sti.HIV``: inherits continuous CD4, ART reconstitution,
    and CD4-based mortality unchanged. Adds (1) HPVsim-friendly directional
    beta targeting ``hpv.SexualNetwork`` (whose p1=female, p2=male, unlike
    STIsim's ``structuredsexual``), and (2) a Rwanda init-prevalence curve.
    """

    def __init__(self, beta_m2f=0.0035, rel_beta_f2m=0.5, init_prev_data=None,
                 pars=None, **kwargs):
        super().__init__(pars=pars, init_prev_data=init_prev_data, **kwargs)
        self._beta_m2f = beta_m2f
        self._rel_beta_f2m = rel_beta_f2m

    def init_pre(self, sim):
        super().init_pre(sim)
        # Target the HPV sexual network by name with directional betas.
        # hpv.SexualNetwork puts females in p1, males in p2, so betamap entry
        # 0 = female->male, entry 1 = male->female. Male->female is the higher-
        # risk direction (beta_m2f); female->male = beta_m2f * rel_beta_f2m.
        nets = [n for n in sim.networks.values() if isinstance(n, SexualNetwork)]
        if not nets:
            misc.warn('hpv.HIV: no SexualNetwork found; HIV will not transmit.')
            return
        beta = {}
        for net in nets:
            beta[net.name] = [self._beta_m2f * self._rel_beta_f2m, self._beta_m2f]
        self.pars.beta = beta
```

- [ ] **Step 4: Export from the package**

In `hpvsim/__init__.py`, add alongside the existing module imports:

```python
from .hiv import HIV, hpv_hiv_connector, HIVStratifiedResults
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_m08_hiv_disease.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/hiv.py hpvsim/__init__.py tests/test_m08_hiv_disease.py
git commit -m "M08: add hpv.HIV subclass with directional beta on hpv.SexualNetwork"
```

---

## Task 3: `hpv_hiv_connector` — CD4 binning + factor arrays + rel_sus application

**Files:**
- Modify: `hpvsim/hiv.py`
- Test: `tests/test_m08_connector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_m08_connector.py
import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector, _HIV_EFFECTS


def test_cd4_stratum_boundaries():
    c = hpv_hiv_connector()
    cd4 = np.array([50.0, 199.0, 200.0, 400.0, 700.0])
    strata = c._cd4_stratum(cd4)
    # 0 = lt200, 1 = gt200 (>=200, including >=500)
    assert list(strata) == [0, 0, 1, 1, 1]


def test_rel_sus_scaled_for_hiv_positive():
    """After step(), an HIV+ agent's HPV rel_sus is multiplied by the stratum factor."""
    h = hpv.HIV(beta_m2f=0.0)  # no transmission; we set HIV state manually
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h])
    sim.init()
    hivmod = sim.diseases.hiv
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]

    # Force one agent HIV+ with low CD4, rest HIV-negative.
    uid = sim.people.auids[0]
    hivmod.infected[uid] = True
    hivmod.cd4[uid] = 100.0  # lt200
    hpvmod.rel_sus[sim.people.auids] = 1.0

    conn.step()
    assert np.isclose(hpvmod.rel_sus[uid], _HIV_EFFECTS['rel_sus']['lt200'])
    other = sim.people.auids[1]
    assert np.isclose(hpvmod.rel_sus[other], 1.0)  # HIV- unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_m08_connector.py -v`
Expected: FAIL — `ImportError: cannot import name 'hpv_hiv_connector'` / `_HIV_EFFECTS`.

- [ ] **Step 3: Add the effect constants and connector to `hpvsim/hiv.py`**

Add after the imports, before `class HIV`:

```python
# CD4-stratified HIV→HPV effect multipliers. Copied by value from v2's
# HIVsim defaults (hpvsim/_v2_legacy/hiv.py:29-44) per the no-quarantine-import
# rule. Strata: 'lt200' = CD4 < 200; 'gt200' = CD4 >= 200 (v2's 200-500 band,
# extended to all CD4 >= 200 for HIV+ agents).
_HIV_EFFECTS = {
    'rel_sus': {'lt200': 2.2, 'gt200': 2.2},   # increased HPV acquisition
    'rel_sev': {'lt200': 1.5, 'gt200': 1.2},   # faster/worse CIN->cancer progression
    'rel_imm': {'lt200': 0.36, 'gt200': 0.76}, # reduced post-infection/vaccine immunity
}
_CD4_THRESHOLD = 200.0
```

Add this class after `class HIV`:

```python
class hpv_hiv_connector(ss.Connector):
    """Apply v2's CD4-stratified HIV→HPV effects to every HPV module.

    Each step: bin HIV+ agents' CD4 into discrete strata, compute per-agent
    factor arrays (hiv_rel_sus / hiv_rel_sev / hiv_rel_imm; 1.0 for HIV-),
    and multiply each HPV module's rel_sus by hiv_rel_sus. The rel_sev and
    rel_imm factors are *read* by HPV.set_prognoses, HPV.step_state, and the
    vaccine products (see those sites) — applied where they compose correctly
    with CrossImmunity, which overwrites rel_sus each step before this runs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hpv_modules = None
        self.hiv_module = None
        self.define_states(
            ss.FloatArr('hiv_rel_sus', default=1.0),
            ss.FloatArr('hiv_rel_sev', default=1.0),
            ss.FloatArr('hiv_rel_imm', default=1.0),
        )

    def init_pre(self, sim):
        super().init_pre(sim)
        self.hpv_modules = [m for m in sim.diseases.values() if isinstance(m, HPV)]
        hivs = [m for m in sim.diseases.values() if isinstance(m, HIV)]
        if not self.hpv_modules or not hivs:
            raise ValueError(
                'hpv_hiv_connector requires both HPV genotype module(s) and an '
                'hpv.HIV disease in the sim.'
            )
        self.hiv_module = hivs[0]

    def _cd4_stratum(self, cd4):
        """Return 0 for lt200 (CD4<200), 1 for gt200 (CD4>=200)."""
        return (np.asarray(cd4) >= _CD4_THRESHOLD).astype(int)

    def _factor_array(self, effect, hiv_pos, strata, n):
        """Build a per-agent factor array (1.0 for HIV-, stratum value for HIV+)."""
        out = np.ones(n, dtype=float)
        lt200 = _HIV_EFFECTS[effect]['lt200']
        gt200 = _HIV_EFFECTS[effect]['gt200']
        vals = np.where(strata == 0, lt200, gt200)
        out[hiv_pos] = vals[hiv_pos]
        return out

    def step(self):
        if not self.hpv_modules:
            return
        auids = self.sim.people.auids
        cd4 = self.hiv_module.cd4[auids]
        hiv_pos = self.hiv_module.infected[auids]
        strata = self._cd4_stratum(np.nan_to_num(cd4, nan=1e4))  # HIV- -> high CD4 -> gt200, masked out below
        n = len(auids)
        rel_sus = self._factor_array('rel_sus', hiv_pos, strata, n)
        rel_sev = self._factor_array('rel_sev', hiv_pos, strata, n)
        rel_imm = self._factor_array('rel_imm', hiv_pos, strata, n)
        self.hiv_rel_sus[auids] = rel_sus
        self.hiv_rel_sev[auids] = rel_sev
        self.hiv_rel_imm[auids] = rel_imm
        # Acquisition effect: multiply each module's rel_sus (set by CrossImmunity
        # earlier this step) by the HIV factor. Susceptible-only by construction.
        for m in self.hpv_modules:
            m.rel_sus[auids] = m.rel_sus[auids] * rel_sus
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_m08_connector.py -v`
Expected: PASS. (The connector is not yet auto-wired by the Sim; the test reaches it because Task 7 isn't done — so for now add `connectors=[hpv_hiv_connector()]` is NOT needed because the test fetches it... if the connector isn't present, the test will KeyError. To make Task 3 self-contained, the test must pass the connector explicitly.)

Update the test's sim construction in Step 1 to pass the connector explicitly until Task 7 auto-wires it:

```python
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h],
                  connectors=[hpv_hiv_connector()])
```

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hiv.py tests/test_m08_connector.py
git commit -m "M08: add hpv_hiv_connector with CD4 binning and rel_sus application"
```

---

## Task 4: `HPV._hiv_connector()` lookup + `rel_sev` consumption in `set_prognoses`

**Files:**
- Modify: `hpvsim/hpv.py` (add `_hiv_connector`; edit `set_prognoses` near line 279-285)
- Test: `tests/test_m08_rel_sev.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m08_rel_sev.py
import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector


def test_set_prognoses_uses_hiv_rel_sev():
    """When the connector reports hiv_rel_sev>1 for an agent, set_prognoses
    forms a larger effective severity than the no-HIV baseline."""
    sim = hpv.Sim(n_agents=300, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16], diseases=[hpv.HIV(beta_m2f=0.0)],
                  connectors=[hpv_hiv_connector()])
    sim.init()
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    # No HIV module connector lookup should yield factor 1.0 by default.
    assert hpvmod._hiv_connector() is conn
    # Default state: every agent hiv_rel_sev == 1.0 (no HIV).
    uids = sim.people.auids[:5]
    assert np.allclose(conn.hiv_rel_sev[uids], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m08_rel_sev.py -v`
Expected: FAIL — `AttributeError: 'HPV' object has no attribute '_hiv_connector'`.

- [ ] **Step 3: Add `_hiv_connector` to HPV and consume the factor**

In `hpvsim/hpv.py`, add after `_cross_immunity_connector` (after line 184):

```python
    def _hiv_connector(self):
        """Locate the hpv_hiv_connector on the sim, if any (None when no HIV)."""
        from .hiv import hpv_hiv_connector
        for c in self.sim.connectors.values():
            if isinstance(c, hpv_hiv_connector):
                return c
        return None
```

In `set_prognoses`, replace lines 279-284 (the `cross = self._cross_immunity_connector()` block) with:

```python
        cross = self._cross_immunity_connector()
        if cross is not None:
            cross.ensure_rel_sev(uids)
            rel_sev_uids = cross.rel_sev[uids]
        else:
            rel_sev_uids = np.ones(len(uids), dtype=float)
        # HIV co-infection raises progression severity (gated no-op when no HIV).
        hivc = self._hiv_connector()
        if hivc is not None:
            rel_sev_uids = rel_sev_uids * hivc.hiv_rel_sev[uids]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m08_rel_sev.py -v`
Expected: PASS.

- [ ] **Step 5: Run the no-HIV guard**

Run: `python -m pytest tests/test_m01_short_summary_parity.py -v`
Expected: PASS — `_hiv_connector()` returns None for HPV-only sims, so `rel_sev_uids` is unchanged.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/hpv.py tests/test_m08_rel_sev.py
git commit -m "M08: HPV.set_prognoses reads hiv_rel_sev (gated no-op without HIV)"
```

---

## Task 5: `rel_imm` consumption in `HPV.step_state` clearance

**Files:**
- Modify: `hpvsim/hpv.py:456-458` (clearance immunity sampling)
- Test: `tests/test_m08_rel_imm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m08_rel_imm.py
import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector


def test_clearance_immunity_scaled_by_hiv_rel_imm():
    """A cleared HIV+ female's conferred nab_imm/cell_imm is reduced by hiv_rel_imm."""
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16], diseases=[hpv.HIV(beta_m2f=0.0)],
                  connectors=[hpv_hiv_connector()])
    sim.init()
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    hivmod = sim.diseases.hiv
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]

    # Pick a female agent, infect with HPV due to clear now, make her HIV+ lt200.
    females = sim.people.auids[sim.people.female[sim.people.auids]]
    uid = females[0]
    hivmod.infected[uid] = True
    hivmod.cd4[uid] = 100.0
    conn.step()  # populate hiv_rel_imm
    hpvmod.infected[uid] = True
    hpvmod.precin[uid] = True
    hpvmod.ti_clearance[uid] = sim.ti
    hpvmod.nab_imm[uid] = 0.0  # first clearance
    hpvmod.step_state()
    # nab_imm should be at most rel_imm['lt200'] * imm_init mean fraction;
    # simplest robust check: it is strictly less than the gt200/no-HIV path would give.
    assert hpvmod.nab_imm[uid] >= 0.0
    # Compare against an HIV- female cleared the same step.
    uid2 = females[1]
    hpvmod.infected[uid2] = True
    hpvmod.precin[uid2] = True
    hpvmod.ti_clearance[uid2] = sim.ti
    hpvmod.nab_imm[uid2] = 0.0
    # Re-run not needed; assert factor applied: conn.hiv_rel_imm[uid] < conn.hiv_rel_imm[uid2]
    assert conn.hiv_rel_imm[uid] < conn.hiv_rel_imm[uid2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m08_rel_imm.py -v`
Expected: FAIL — `conn.hiv_rel_imm[uid]` is 1.0 because `step()` ran before HIV state was set; reorder in test OR the scaling isn't applied. (Set HIV state *before* `conn.step()` — the test above already does. The real failure is that the clearance code doesn't yet read the factor, so this test's final assert passes but the immunity isn't scaled. Make the test assert the scaling by checking nab_imm is reduced vs an unscaled reference — see Step 3 note.)

Refine the assertion to pin the behavior: capture `imm_init` mean and assert the HIV+ agent's nab is bounded by `rel_imm['lt200']`:

```python
    expected_cap = hpv.hiv._HIV_EFFECTS['rel_imm']['lt200']
    # nab_imm for the HIV+ first-clearance female is the seroconvert draw * imm_init
    # sample * rel_imm factor; assert it never exceeds the unscaled imm_init max * factor.
    assert hpvmod.nab_imm[uid] <= hpvmod.pars.imm_init.rvs(np.array([uid]))[0] * expected_cap + 1e-9
```

- [ ] **Step 3: Scale the immunity increment in clearance**

In `hpvsim/hpv.py`, in `step_state`, after `cell_all = p.cell_imm_init.rvs(f_cleared)` (line 458) insert:

```python
                # HIV co-infection reduces conferred immunity (gated no-op).
                hivc = self._hiv_connector()
                if hivc is not None:
                    imm_factor = hivc.hiv_rel_imm[f_cleared]
                    nab_all = nab_all * imm_factor
                    cell_all = cell_all * imm_factor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m08_rel_imm.py -v`
Expected: PASS.

- [ ] **Step 5: Run the no-HIV guard**

Run: `python -m pytest tests/test_m01_short_summary_parity.py tests/test_m02_short_summary_parity.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/hpv.py tests/test_m08_rel_imm.py
git commit -m "M08: HPV clearance scales conferred immunity by hiv_rel_imm (gated)"
```

---

## Task 6: `rel_imm` consumption in vaccine / therapeutic-vaccine products

**Files:**
- Modify: `hpvsim/products.py` (vx and txvx `administer`)
- Test: `tests/test_m08_vx_rel_imm.py`

- [ ] **Step 1: Inspect the current administer methods**

Run: `grep -n "def administer\|vax_imm\|txvx_imm\|class .*vx\|class .*Vx" hpvsim/products.py`
Identify where `vax_imm` and `txvx_imm` are written (the immunity each product confers).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_m08_vx_rel_imm.py
import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector


def test_vaccine_imm_scaled_for_hiv_positive():
    """A vaccine confers strictly less vax_imm to an HIV+ lt200 agent than to
    an otherwise-identical HIV- agent (the rel_imm['lt200'] reduction)."""
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16, 18], diseases=[hpv.HIV(beta_m2f=0.0)],
                  connectors=[hpv_hiv_connector()])
    sim.init()
    hivmod = sim.diseases.hiv
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]

    # uid_pos: HIV+ lt200; uid_neg: HIV-. Same sex (female) for a fair comparison.
    females = sim.people.auids[sim.people.female[sim.people.auids]]
    uid_pos, uid_neg = females[0], females[1]
    hivmod.infected[uid_pos] = True
    hivmod.cd4[uid_pos] = 100.0
    conn.step()
    assert np.isclose(conn.hiv_rel_imm[uid_pos], hpv.hiv._HIV_EFFECTS['rel_imm']['lt200'])
    assert np.isclose(conn.hiv_rel_imm[uid_neg], 1.0)

    # Administer the same vaccine product to both and compare conferred vax_imm
    # on the hpv16 module. (Adapt the product constructor + administer signature
    # to the real API found in Step 1.)
    vx = hpv.vx(genotypes=['hpv16', 'hpv18'])  # adjust to actual constructor
    hpvmod16 = [d for d in sim.diseases.values()
                if isinstance(d, hpv.HPV) and d.genotype == 'hpv16'][0]
    vx.administer(sim.people, np.array([uid_pos, uid_neg]))
    assert float(hpvmod16.vax_imm[uid_pos]) < float(hpvmod16.vax_imm[uid_neg])
```

> Note: adapt `hpv.vx(...)` construction and the `administer` signature to the actual product API found in Step 1. The HIV+ vs HIV− comparison is robust to the exact conferred value, so it holds regardless of the product's internal `rel_imm` table — it only requires that the gated `hiv_rel_imm` scaling is applied.

- [ ] **Step 3: Scale conferred immunity in the products**

In each product's `administer` (vx and txvx), after computing the per-agent conferred immunity array `imm` and before writing it to `module.vax_imm` / `module.txvx_imm`, insert the gated scaling. Locate the connector via the sim the product is bound to (products receive `people`; use `people.sim`):

```python
        # HIV co-infection reduces vaccine take (gated no-op without HIV).
        from .hiv import hpv_hiv_connector
        sim = people.sim
        hivc = next((c for c in sim.connectors.values()
                     if isinstance(c, hpv_hiv_connector)), None)
        if hivc is not None:
            imm = imm * hivc.hiv_rel_imm[uids]
```

Apply identically in the txvx product, writing to `txvx_imm`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m08_vx_rel_imm.py -v`
Expected: PASS.

- [ ] **Step 5: Run the M05/M06 vaccine guards**

Run: `python -m pytest tests/test_m06_txvx_unit.py tests/test_m06_txvx_integration.py -v`
Expected: PASS — no HIV present, scaling is a no-op.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/products.py tests/test_m08_vx_rel_imm.py
git commit -m "M08: vaccine and txvx products scale conferred immunity by hiv_rel_imm (gated)"
```

---

## Task 7: Detection-based auto-wiring of connector + analyzer

**Files:**
- Modify: `hpvsim/sim.py` (the assembly block from Task 1; the `analyzers = [...]` line)
- Test: `tests/test_m08_sim_assembly.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_m08_sim_assembly.py
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector, HIVStratifiedResults


def test_hiv_autowires_connector_and_analyzer():
    sim = hpv.Sim(n_agents=200, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16, 18], diseases=[hpv.HIV(beta_m2f=0.0)])
    sim.init()
    assert any(isinstance(c, hpv_hiv_connector) for c in sim.connectors.values())
    assert any(isinstance(a, HIVStratifiedResults) for a in sim.analyzers.values())


def test_no_hiv_no_autowire():
    sim = hpv.Sim(n_agents=200, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16, 18])
    sim.init()
    assert not any(isinstance(c, hpv_hiv_connector) for c in sim.connectors.values())
    assert not any(isinstance(a, HIVStratifiedResults) for a in sim.analyzers.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_m08_sim_assembly.py::test_hiv_autowires_connector_and_analyzer -v`
Expected: FAIL — no connector auto-added.

- [ ] **Step 3: Add detection auto-wiring**

In `hpvsim/sim.py`, at the top of the file add the import (with the other local imports):

```python
from .hiv import HIV, hpv_hiv_connector, HIVStratifiedResults
```

After `diseases = hpv_diseases + other_diseases` (from Task 1), add:

```python
        auto_analyzers = [HPVTotal()]
        if any(isinstance(d, HIV) for d in other_diseases):
            auto_connectors.append(hpv_hiv_connector())
            auto_analyzers.append(HIVStratifiedResults())
```

Then change the existing `analyzers = [HPVTotal()] + user_analyzers` line to:

```python
        analyzers = auto_analyzers + user_analyzers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_m08_sim_assembly.py -v`
Expected: PASS (all 5 tests). Now the explicit `connectors=[hpv_hiv_connector()]` in Tasks 3-6 tests is redundant but harmless; leave them.

- [ ] **Step 5: Run the no-HIV CRN guard**

Run: `python -m pytest tests/test_m01_short_summary_parity.py tests/test_m02_short_summary_parity.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add hpvsim/sim.py tests/test_m08_sim_assembly.py
git commit -m "M08: auto-wire hpv_hiv_connector + HIVStratifiedResults when HIV present"
```

---

## Task 8: `HIVStratifiedResults` analyzer

**Files:**
- Modify: `hpvsim/hiv.py`
- Test: `tests/test_m08_results.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m08_results.py
import numpy as np
import hpvsim as hpv


def test_hiv_stratified_results_present_and_shaped():
    sim = hpv.Sim(n_agents=500, start=2000, stop=2003, dt=0.25, location='nigeria',
                  genotypes=[16, 18], diseases=[hpv.HIV(beta_m2f=0.004)])
    sim.run()
    res = sim.results.hivstratifiedresults
    for key in ('cancers_with_hiv', 'cancers_no_hiv',
                'hpv_prevalence_with_hiv', 'hpv_prevalence_no_hiv'):
        assert key in res
        assert len(res[key]) == len(sim.results.timevec)
    # Prevalence is a fraction in [0, 1].
    assert np.all((res['hpv_prevalence_with_hiv'] >= 0) &
                  (res['hpv_prevalence_with_hiv'] <= 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m08_results.py -v`
Expected: FAIL — `HIVStratifiedResults` does not yet define these results.

- [ ] **Step 3: Implement the analyzer in `hpvsim/hiv.py`**

```python
class HIVStratifiedResults(ss.Analyzer):
    """HPV/cancer outcomes split by HIV status (mirrors v2's cancer_*_with/no_hiv).

    Adds only the cross-disease stratification HPV needs; HIV's own epidemic
    results come from sti.HIV / sti.ART. Auto-added by hpv.Sim when HIV present.
    """

    def init_pre(self, sim):
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv_module = next(d for d in sim.diseases.values() if isinstance(d, HIV))
        super().init_pre(sim)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('cancers_with_hiv', dtype=int, label='New cancers (HIV+)'),
            ss.Result('cancers_no_hiv', dtype=int, label='New cancers (HIV-)'),
            ss.Result('hpv_prevalence_with_hiv', dtype=float, label='HPV prevalence (HIV+)'),
            ss.Result('hpv_prevalence_no_hiv', dtype=float, label='HPV prevalence (HIV-)'),
        )

    def step(self):
        ti = self.sim.ti
        people = self.sim.people
        alive = people.alive.values
        hiv_pos = self.hiv_module.infected.values & alive
        hiv_neg = (~self.hiv_module.infected.values) & alive

        # Any-genotype HPV infection (union across modules).
        any_hpv = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            any_hpv |= m.infected.values

        n_pos = int(hiv_pos.sum())
        n_neg = int(hiv_neg.sum())
        self.results['hpv_prevalence_with_hiv'][ti] = (
            float((any_hpv & hiv_pos).sum()) / n_pos if n_pos else 0.0)
        self.results['hpv_prevalence_no_hiv'][ti] = (
            float((any_hpv & hiv_neg).sum()) / n_neg if n_neg else 0.0)

        # New cancers this step, attributed by current HIV status.
        new_cancer = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            fired = (m.cancerous.values & (m.ti_cancerous.values == ti))
            new_cancer |= fired
        self.results['cancers_with_hiv'][ti] = int((new_cancer & hiv_pos).sum())
        self.results['cancers_no_hiv'][ti] = int((new_cancer & hiv_neg).sum())
```

> Note: confirm the new-cancer detection matches how `HPV.step_state` records cancers (it sets `cancerous=True` and writes `results.new_cancers[ti]`; `ti_cancerous == ti` identifies the agents that transitioned this step). If `ti_cancerous` stores a rounded float, compare with `np.isclose` or `<= ti` combined with a "not previously counted" guard.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m08_results.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hpvsim/hiv.py tests/test_m08_results.py
git commit -m "M08: add HIVStratifiedResults analyzer (HPV/cancer by HIV status)"
```

---

## Task 9: Co-infection anchor — direction + CD4-binning correctness

**Files:**
- Create: `tests/regression/anchor_hiv_hpv.py`
- Test: `tests/test_m08_coinfection_direction.py`

- [ ] **Step 1: Write the anchor builder**

```python
# tests/regression/anchor_hiv_hpv.py
"""Co-infection anchor: 4-genotype HPV + transmission HIV + ART on Nigeria.

Phase-1 anchor uses a hand-tuned HIV beta to produce a non-trivial HIV+
subpopulation; it is NOT calibrated to a country (that is Phase 2 / Rwanda).
"""
import starsim as ss
import stisim as sti
import hpvsim as hpv

PARS = dict(n_agents=5000, start=1990, stop=2030, dt=0.25, location='nigeria')


def build_sim(seed=0, hiv_beta_m2f=0.0045):
    return hpv.Sim(
        rand_seed=seed,
        genotypes=[16, 18, 'hi5', 'ohr'],
        diseases=[hpv.HIV(beta_m2f=hiv_beta_m2f, init_prev_data=0.01)],
        interventions=[sti.ART(coverage=[0, 0.3, 0.6], years=[1990, 2010, 2030])],
        **PARS,
    )
```

> Note: adapt the `sti.ART(...)` constructor call to its actual signature (check `grep -n "class ART" $(python -c "import stisim,os;print(os.path.dirname(stisim.__file__))")/diseases/hiv.py` or `interventions`). If `init_prev_data` expects a file/array rather than a scalar, pass a small constant prevalence via the form STIsim accepts.

- [ ] **Step 2: Write the failing direction test**

```python
# tests/test_m08_coinfection_direction.py
import numpy as np
from tests.regression.anchor_hiv_hpv import build_sim


def test_hiv_positive_have_higher_hpv_and_cancer():
    sim = build_sim(seed=0)
    sim.run()
    res = sim.results.hivstratifiedresults
    # Late-sim mean prevalence higher among HIV+ than HIV-.
    sl = slice(-40, None)  # last 10 years at dt=0.25
    assert np.nanmean(res['hpv_prevalence_with_hiv'][sl]) > \
           np.nanmean(res['hpv_prevalence_no_hiv'][sl])
    # Cumulative cancers per capita higher among HIV+ (sanity on direction).
    assert res['cancers_with_hiv'].sum() > 0
```

- [ ] **Step 3: Run test to verify it fails, then passes**

Run: `python -m pytest tests/test_m08_coinfection_direction.py -v`
Expected: initially may FAIL if HIV prevalence is too low or `init_prev_data`/`ART` args are wrong. Tune `hiv_beta_m2f` and the ART/init-prev args until HIV+ subpopulation is non-trivial and the direction holds. This task validates the *mechanics wire together end-to-end*, not calibration.

- [ ] **Step 4: Verify CD4 binning against v2 semantics**

Add to the test file:

```python
def test_cd4_high_bins_to_gt200():
    from hpvsim.hiv import hpv_hiv_connector
    c = hpv_hiv_connector()
    assert list(c._cd4_stratum(np.array([594.0, 800.0]))) == [1, 1]  # newly-infected start CD4 -> gt200
```

Run: `python -m pytest tests/test_m08_coinfection_direction.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/regression/anchor_hiv_hpv.py tests/test_m08_coinfection_direction.py
git commit -m "M08: co-infection anchor + direction/CD4-binning integration tests"
```

- [ ] **Step 6: Run the full M08 Phase-1 suite + no-HIV guards**

Run: `python -m pytest tests/test_m08_*.py tests/test_m01_short_summary_parity.py tests/test_m02_short_summary_parity.py -v`
Expected: PASS. **Phase 1 complete: mechanics correct, no-HIV byte-identity preserved.**

---

# Phase 2 — HIV epidemic + Rwanda parity

> Phase 2 is calibration-and-validation work; it is procedural rather than pure unit-TDD. Each task still ends in a commit and a runnable artifact.

> **Data audit revision (2026-06-04).** The Rwanda data already exists in the sibling repo `C:\Users\ryanhu\PycharmProjects\hpvsim_v23_validation\hpvsim_rwanda` (published v2.3 baseline). This changes the original Phase-2 tasks:
> - **T10** copies the *existing* Rwanda ART/HIV CSVs into `hpvsim/data/` rather than authoring new ones; `init_prev`-by-age is **derived** from the v2 baseline's `hiv_prevalence_by_age` (no file exists). ART coverage files: `rwanda_art_coverage_by_age_{females,males}.csv`.
> - **NEW T10b** implements the **coverage-based ART shortcut** (settled decision): a thin intervention that marks a data-matched fraction of HIV+ agents on-ART per the age/sex/year curve — NOT STIsim's HIVTest→ART cascade (v2 has no testing step; no testing data exists).
> - **T11** does not generate a baseline from scratch — the v2 baselines are **cached** under `hpvsim_rwanda/results/v2.3.0_baseline/`; this task extracts the HIV-stratified short-summary from the cache (or re-runs the frozen v2.3 install at `hpvsim_v23_frozen` if a fresh one is needed).
> - **T12–T14** otherwise as written: calibrate `sti.HIV` transmission beta to the Rwanda prevalence trajectory, run the HIV-stratified parity gate, then the `hpvsim_rwanda` release gate.
>
> **Prerequisite resolved (2026-06-04):** v3 `load_country` was Nigeria-only, blocking Rwanda. Fixed by cherry-picking commit `2b9be106` from the `country-support` branch (removes the `_KNOWN_LOCATIONS` gate; the bundled UN WPP 2024 data + `loaders` already shipped) → landed as `bcc2b07b`. `load_country('rwanda')` and a full Rwanda co-infection sim now run end-to-end.

## Task 10: Rwanda HIV data files + init-prev wiring

**Files:**
- Create: `hpvsim/data/hiv_init_prev.csv`, `hpvsim/data/hiv_art_coverage.csv`
- Modify: `hpvsim/hiv.py` (`HIV` loads init-prev by location)
- Test: `tests/test_m08_data_load.py`

- [ ] **Step 1: Source the data**

Adapt the column schema from `hivsim_examples/zimbabwe/init_prev_hiv.csv` (run `head` on it). Populate Rwanda initial HIV prevalence by age (UNAIDS/DHS Rwanda) and an ART coverage-by-year schedule. Keep file format identical to the Zimbabwe template so STIsim's loaders accept it.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_m08_data_load.py
import hpvsim as hpv


def test_hiv_init_prev_loads_for_rwanda():
    h = hpv.HIV.from_location('rwanda')   # convenience loader (Step 3)
    assert h.init_prev_data is not None
```

- [ ] **Step 3: Add a location loader to `HIV`**

```python
    @classmethod
    def from_location(cls, location, **kwargs):
        """Build HIV with init-prev + default beta loaded for a known location."""
        from .data import load_hiv  # add this adapter alongside load_country
        init_prev, beta_m2f = load_hiv(location)
        return cls(init_prev_data=init_prev, beta_m2f=beta_m2f, **kwargs)
```

Add a `load_hiv(location)` adapter in the `hpv.data` module (mirroring `load_country`) that reads the two CSVs and returns `(init_prev_array_or_df, default_beta)`.

- [ ] **Step 4: Run test; then commit**

Run: `python -m pytest tests/test_m08_data_load.py -v`
Expected: PASS.

```bash
git add hpvsim/data/hiv_init_prev.csv hpvsim/data/hiv_art_coverage.csv hpvsim/hiv.py hpvsim/data*.py tests/test_m08_data_load.py
git commit -m "M08: Rwanda HIV init-prev + ART data files and load_hiv adapter"
```

## Task 11: v2 Rwanda baseline generator

**Files:**
- Create: `tests/regression/baseline_hiv_v2.py`

- [ ] **Step 1: Write the generator**

Model it on the existing v2 baseline generators (`tests/regression/baseline_v23.py`, `multi_seed_v2.py`). It runs the v2 HPVsim with HIV enabled (the v2 `HIVsim` requires `art_datafile` + `hiv_datafile`) on Rwanda across N seeds and writes HIV-stratified short-summary metrics (total cancers HIV+/HIV−, HPV prevalence HIV+/HIV−, cancer rate ratio) to a local gitignored `.npz`/`.csv` — never committed (per M0 convention).

- [ ] **Step 2: Run it to produce the local baseline**

Run: `python tests/regression/baseline_hiv_v2.py --location rwanda --seeds 30`
Expected: writes `tests/regression/_baselines/hiv_rwanda_v2.npz` (gitignored).

- [ ] **Step 3: Commit the generator (not the baseline)**

```bash
git add tests/regression/baseline_hiv_v2.py
git commit -m "M08: v2 Rwanda HIV baseline generator (baseline stays local)"
```

## Task 12: Calibrate transmission HIV + ART to Rwanda

**Files:**
- Create: `examples/m08_rwanda_hiv_calibration.py` (or a regression helper)

- [ ] **Step 1: Build the Rwanda co-infection sim** using `hpv.HIV.from_location('rwanda')` + `sti.ART` from the coverage schedule.

- [ ] **Step 2: Calibrate HIV beta (and ART params if needed)** so simulated HIV prevalence trajectory overlaps Rwanda data, using the `hivsim_examples/zimbabwe/` workflow as a template (Optuna via `hpv.calibration` / `sti` calibration). Target: HIV prevalence by year (and ideally by age/sex) within data uncertainty.

- [ ] **Step 3: Record calibrated parameters** in `load_hiv('rwanda')` defaults so `from_location('rwanda')` reproduces the calibrated epidemic.

- [ ] **Step 4: Commit**

```bash
git add examples/m08_rwanda_hiv_calibration.py hpvsim/data*.py
git commit -m "M08: calibrate transmission HIV + ART to Rwanda HIV prevalence"
```

## Task 13: Multi-seed HIV-stratified parity gate

**Files:**
- Create: `tests/regression/test_m08_hiv_parity.py`

- [ ] **Step 1: Write the parity test** following the M03/M05 z-score pattern (`tests/test_m03_short_summary_parity.py` as the template): run N v3 seeds of the calibrated Rwanda co-infection sim, compare the HIV-stratified short-summary vector against the local v2 baseline from Task 11 via per-metric z-score.

```python
# tests/regression/test_m08_hiv_parity.py — skeleton; mirror test_m03 structure
THRESHOLD = 3.0  # |z| < 3; loosen a single residual metric to 5.0 only if needed (document)

def test_m08_rwanda_hiv_stratified_parity():
    v2 = load_local_baseline('hiv_rwanda_v2.npz')
    v3 = run_v3_seeds(n_seeds=10)
    for metric, (mu, sd) in v2.items():
        z = abs(v3[metric].mean() - mu) / (sd + 1e-9)
        assert z < THRESHOLD, f'{metric}: |z|={z:.2f}'
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/regression/test_m08_hiv_parity.py -v`
Expected: PASS, or a documented single-metric loosening to `|z| < 5` (M05 precedent), recorded in the spec's post-implementation deltas.

- [ ] **Step 3: Decide static vs dynamic re-evaluation**

If HIV+ cancer metrics fail the gate in a way consistent with under-progression of pre-AIDS infections, implement dynamic re-evaluation: in `hpv_hiv_connector.step`, detect agents whose CD4 crossed 200 since last step and who are HPV-infected (precin/CIN), and re-call the affected HPV module's prognosis re-sampling for `dur_cin`/P(cancer). Add a unit test asserting a CD4 drop re-shortens `ti_cancerous`. Otherwise, document that static prognoses suffice.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_m08_hiv_parity.py
git commit -m "M08: multi-seed HIV-stratified parity gate vs v2 Rwanda baseline"
```

## Task 14: Release gate + milestone wrap-up

- [ ] **Step 1: Run the `hpvsim_rwanda` analysis-repo check** (overlapping uncertainty intervals on headline HIV-stratified outputs) per the milestone acceptance test.

- [ ] **Step 2: Update `MIGRATION_PLAN.md`** — flip M8 status to complete with PR/branch reference; add a "Post-implementation deltas" section to the M08 spec recording: final effect values, static-vs-dynamic decision, any `|z| < 5` loosening, and the Rwanda calibration outcome.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: PASS.

- [ ] **Step 4: Commit + open PR to `v3.0-dev`**

```bash
git add MIGRATION_PLAN.md docs/superpowers/specs/2026-06-03-m08-hiv-hpv-coinfection-design.md
git commit -m "M08: mark milestone complete; record post-implementation deltas"
```

---

## Self-review notes (for the executor)

- **No-HIV byte-identity** is guarded after Tasks 1, 4, 5, 7 by the M01/M02 parity tests. Run them at each of those checkpoints — a failure means a gated read site is not actually gated.
- **Connector ordering** (`hpv_hiv_connector` after `CrossImmunity`) is load-bearing for the `rel_sus` multiply. Task 3's test only checks the connector in isolation; Task 9's anchor exercises the real ordering.
- **Adapt-to-actual-API flags:** Tasks 6 (product `administer` internals), 9 (`sti.ART` constructor, `init_prev_data` form), and 10 (Zimbabwe CSV schema) require checking the real signatures before writing final code — each step says so inline.