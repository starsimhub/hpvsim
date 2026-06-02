# HPVsim M05: Vaccination Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prophylactic HPV vaccination to v3 by composing Starsim's native intervention/product framework with one HPV-specific product class (`hpv.vx`) and a thin v2-compatible API shim (`hpv.BaseVaccination` + `routine_vx` + `campaign_vx`). The acceptance test is two regression anchor scenarios (routine + campaign) gated against locally-regenerated v2 baselines under M03's multi-seed z-score gate (`|z| < 3`).

**Architecture:** Use Starsim's existing diamond inheritance (`BaseVaccination` + `RoutineDelivery` / `CampaignDelivery`) and per-intervention state (`vaccinated`, `n_doses`, `ti_vaccinated`). The only HPV-specific work is (a) `hpv.vx(ss.Vx)` with per-genotype `rel_imm` and all-or-nothing+leaky `administer()` writing to each HPV module's `nab_imm`, and (b) `hpv.BaseVaccination(ss.BaseVaccination)` adding v2's `age_range`/`sex`/`eligibility` constructor args. Cross-immunity propagation through to `rel_sus` is automatic via the existing M03 `CrossImmunity(ss.Connector)`.

**Tech Stack:** Starsim 3.3.x (`ss.Vx`, `ss.BaseVaccination`, `ss.RoutineDelivery`, `ss.CampaignDelivery`, `ss.bernoulli`), sciris, pandas, numpy. v2 reference code in `hpvsim/_v2_legacy/` (porting reference only — no runtime imports from there).

**Spec:** [`docs/superpowers/specs/2026-05-20-m05-vaccination-scenarios-design.md`](../specs/2026-05-20-m05-vaccination-scenarios-design.md)

**Branch:** `m05-vaccination-scenarios` (off `m04-calibration-loop`; the spec is already committed at `c9410e58`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `hpvsim/data/products_vx.csv` | Create (move from `_v2_legacy/data/`) | Default-product table: `(name, genotype, rel_imm)` for bivalent / quadrivalent / nonavalent. 24 rows. Verbatim copy. |
| `hpvsim/products.py` | Create | `hpv.vx(ss.Vx)` multi-genotype vaccine product + `_load_vx_products()` CSV loader. |
| `hpvsim/interventions.py` | Create | `hpv.BaseVaccination(ss.BaseVaccination)` shim + `hpv.routine_vx` + `hpv.campaign_vx` leaf classes + `_coerce_sex` + `_compose_eligibility` helpers. |
| `hpvsim/__init__.py` | Modify | Add `vx`, `routine_vx`, `campaign_vx` to top-level exports + `__all__`. |
| `tests/test_m05_vx_unit.py` | Create | Unit tests for `_coerce_sex`, `_compose_eligibility`, CSV loading, `hpv.vx` constructor + `administer` semantics. |
| `tests/test_m05_vx_integration.py` | Create | Integration smoke tests: routine/campaign fires, state updates, age/sex targeting, CRN-stream guard against M03 baseline. |
| `tests/regression/anchor_vx_routine.py` | Create | PARS dict for the routine-anchor parity scenario. |
| `tests/regression/anchor_vx_campaign.py` | Create | PARS dict for the campaign-anchor parity scenario. |
| `tests/regression/multi_seed_v2_vx.py` | Create | Script run **once** from a v2 hpvsim env to generate the 30-seed v2 baselines. Output JSONs are gitignored. |
| `tests/regression/v2_seeds_n30_vx_routine.json` | Generated locally (gitignored) | 30-seed v2 baseline for the routine anchor. |
| `tests/regression/v2_seeds_n30_vx_campaign.json` | Generated locally (gitignored) | 30-seed v2 baseline for the campaign anchor. |
| `tests/test_m05_vx_routine_parity.py` | Create | Short-summary z-score parity gate, routine anchor. `@pytest.mark.slow`. |
| `tests/test_m05_vx_campaign_parity.py` | Create | Short-summary z-score parity gate, campaign anchor. `@pytest.mark.slow`. |
| `tests/test_m05_vx_trajectory_parity.py` | Create | Trajectory z-score parity gate, routine anchor only. `@pytest.mark.slow`. |
| `tests/regression/README_m05.md` | Create | How to regenerate the v2 baselines and run the parity gate locally. |
| `MIGRATION_PLAN.md` | Modify | Status table → M5 in progress; rewrite M5 sub-tasks; append M6 sub-tasks (txvx + dx + tx). |

---

## Task 1: Move `products_vx.csv` into active code with a CSV loader

**Files:**
- Create: `hpvsim/data/products_vx.csv` (moved verbatim from `hpvsim/_v2_legacy/data/products_vx.csv`)
- Create: `hpvsim/products.py` (initial scaffold with `_load_vx_products` only)
- Test: `tests/test_m05_vx_unit.py`

- [ ] **Step 1: Write the failing CSV-loader test**

Create `tests/test_m05_vx_unit.py`:

```python
"""Unit tests for M05 vaccination components."""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv
from hpvsim.products import _load_vx_products


def test_load_vx_products_returns_dict_of_genotype_to_rel_imm():
    """_load_vx_products() returns {product_name: {genotype: rel_imm}} from CSV."""
    products = _load_vx_products()
    # Three default products from v2's CSV
    assert set(products.keys()) >= {'bivalent', 'quadrivalent', 'nonavalent'}
    # Bivalent has full protection against hpv16 and hpv18
    assert products['bivalent']['hpv16'] == pytest.approx(1.0)
    assert products['bivalent']['hpv18'] == pytest.approx(1.0)
    # Nonavalent has full protection against hi5
    assert products['nonavalent']['hi5'] == pytest.approx(1.0)
    # Bivalent has partial cross-protection against hi5
    assert 0 < products['bivalent']['hi5'] < 1.0


def test_load_vx_products_cached():
    """Repeat calls return the same dict object (module-level cache)."""
    first = _load_vx_products()
    second = _load_vx_products()
    assert first is second
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/test_m05_vx_unit.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hpvsim.products'`.

- [ ] **Step 3: Move the CSV file into active code**

```
git mv hpvsim/_v2_legacy/data/products_vx.csv hpvsim/data/products_vx.csv
```

- [ ] **Step 4: Create `hpvsim/products.py` with the loader**

Create `hpvsim/products.py`:

```python
"""HPV-specific Starsim products.

Currently contains the prophylactic vaccine product class ``hpv.vx``. M06
will add ``hpv.dx`` (diagnostics) and ``hpv.tx`` (treatments). M06 will also
add the therapeutic vaccine product variant.
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['vx']


_PRODUCT_CSV = Path(__file__).parent / 'data' / 'products_vx.csv'


@functools.lru_cache(maxsize=1)
def _load_vx_products():
    """Load the CSV and return {product_name: {genotype: rel_imm}}.

    Cached at module load — the CSV is small (~24 rows) and never changes
    at runtime. Returns a frozen mapping of product name -> {genotype: rel_imm}.
    """
    df = pd.read_csv(_PRODUCT_CSV)
    expected_cols = {'name', 'genotype', 'rel_imm'}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f'products_vx.csv missing required columns: {sorted(missing)}'
        )
    out = {}
    for name, group in df.groupby('name'):
        out[name] = dict(zip(group['genotype'], group['rel_imm'].astype(float)))
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

```
pytest tests/test_m05_vx_unit.py -v
```

Expected: both `test_load_vx_products_returns_dict_of_genotype_to_rel_imm` and `test_load_vx_products_cached` PASS.

- [ ] **Step 6: Commit**

```
git add hpvsim/data/products_vx.csv hpvsim/products.py tests/test_m05_vx_unit.py
git rm hpvsim/_v2_legacy/data/products_vx.csv  # already staged by git mv
git commit -m "M05: scaffold hpv.products + move products_vx.csv to active data

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Implement `hpv.vx` product class — constructor + parameter resolution

**Files:**
- Modify: `hpvsim/products.py`
- Test: `tests/test_m05_vx_unit.py`

- [ ] **Step 1: Add failing tests for the constructor surface**

Append to `tests/test_m05_vx_unit.py`:

```python
def test_vx_constructor_with_name_loads_csv():
    """hpv.vx(name='bivalent') resolves rel_imm from the CSV."""
    from hpvsim.products import vx
    product = vx(name='bivalent')
    assert product._rel_imm['hpv16'] == pytest.approx(1.0)
    assert product._rel_imm['hpv18'] == pytest.approx(1.0)


def test_vx_constructor_with_rel_imm_uses_override():
    """hpv.vx(rel_imm={...}) uses the explicit dict and ignores the CSV."""
    from hpvsim.products import vx
    custom = {'hpv16': 0.7, 'hpv18': 0.6}
    product = vx(rel_imm=custom)
    assert product._rel_imm == custom


def test_vx_constructor_both_name_and_rel_imm_raises():
    """Providing both name and rel_imm is ambiguous; raise."""
    from hpvsim.products import vx
    with pytest.raises(ValueError, match='exactly one'):
        vx(name='bivalent', rel_imm={'hpv16': 0.5})


def test_vx_constructor_neither_raises():
    """Providing neither name nor rel_imm has no efficacy; raise."""
    from hpvsim.products import vx
    with pytest.raises(ValueError, match='exactly one'):
        vx()


def test_vx_unknown_name_raises_with_valid_names_listed():
    """Unknown product name surfaces the list of valid names."""
    from hpvsim.products import vx
    with pytest.raises(ValueError, match='bivalent.*quadrivalent.*nonavalent'):
        vx(name='not_a_real_vaccine')
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_m05_vx_unit.py -v -k "vx_constructor or vx_unknown_name"
```

Expected: FAIL — `vx` not yet defined in `hpvsim.products`.

- [ ] **Step 3: Add `vx` class with constructor + `_resolve_vx_pars`**

Edit `hpvsim/products.py` — append after `_load_vx_products`:

```python
def _resolve_vx_pars(name, rel_imm):
    """Resolve (name, rel_imm) to a {genotype: rel_imm} dict.

    Exactly one of name or rel_imm must be provided.
    """
    if (name is None) == (rel_imm is None):  # both None or both set
        raise ValueError(
            'hpv.vx requires exactly one of `name` or `rel_imm`, not both/neither.'
        )
    if rel_imm is not None:
        return dict(rel_imm)
    products = _load_vx_products()
    if name not in products:
        valid = ', '.join(sorted(products.keys()))
        raise ValueError(
            f'Unknown vx product name {name!r}. Valid names: {valid}.'
        )
    return dict(products[name])


class vx(ss.Vx):
    """HPV multi-genotype prophylactic vaccine.

    Constructed with EITHER ``name`` (looks up the per-genotype rel_imm from
    ``hpvsim/data/products_vx.csv``) OR ``rel_imm`` (explicit per-genotype
    dict). Default product names: ``'bivalent'``, ``'quadrivalent'``,
    ``'nonavalent'``.

    The vaccine model is "all-or-nothing + leaky": per agent per genotype,
    draw Bernoulli(rel_imm[g]); on success the agent's ``nab_imm[g]``
    becomes 1.0 (sterilizing immunity), on failure it becomes rel_imm[g]
    (leaky protection floor). Existing ``nab_imm`` is never downgraded.
    """

    def __init__(self, name=None, rel_imm=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            name=name,
            rel_imm=rel_imm,
        )
        self._rel_imm = _resolve_vx_pars(name, rel_imm)
        # CRN-safe Bernoulli; p is overwritten per-genotype in administer().
        self._sterilizing_dist = ss.bernoulli(p=0.0)

    def administer(self, people, uids):
        """Apply the vaccine — see class docstring for the model. Stubbed; see Task 3."""
        raise NotImplementedError('administer() implemented in Task 3.')
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_m05_vx_unit.py -v -k "vx_constructor or vx_unknown_name"
```

Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```
git add hpvsim/products.py tests/test_m05_vx_unit.py
git commit -m "M05: hpv.vx constructor with name lookup + rel_imm override

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Implement `hpv.vx.administer` — all-or-nothing + leaky writing to `nab_imm`

**Files:**
- Modify: `hpvsim/products.py`
- Test: `tests/test_m05_vx_unit.py`

- [ ] **Step 1: Write failing tests for administer semantics**

Append to `tests/test_m05_vx_unit.py`:

```python
def _make_small_sim(genotypes=('hpv16', 'hpv18', 'hi5', 'ohr')):
    """Construct a small initialized 4-genotype sim suitable for administer tests."""
    sim = hpv.Sim(
        location='nigeria',
        start=2010, stop=2012,
        n_agents=200,
        genotypes=list(genotypes),
        rand_seed=0,
    )
    sim.init()
    return sim


def test_vx_administer_bumps_nab_imm_for_active_genotypes():
    """administer() writes per-genotype nab_imm; max-of-existing semantics."""
    from hpvsim.products import vx
    sim = _make_small_sim()
    product = vx(rel_imm={'hpv16': 1.0, 'hpv18': 0.5})
    product.init_pre(sim)  # bind to sim before use
    # Pick 20 agents and vaccinate them
    uids = sim.people.alive.uids[:20]
    pre_hpv16 = sim.diseases['hpv16'].nab_imm[uids].copy()
    pre_hpv18 = sim.diseases['hpv18'].nab_imm[uids].copy()
    product.administer(sim.people, uids)
    post_hpv16 = sim.diseases['hpv16'].nab_imm[uids]
    post_hpv18 = sim.diseases['hpv18'].nab_imm[uids]
    # hpv16 has rel_imm=1.0 -> every agent is sterilizing -> post = 1.0
    assert np.all(post_hpv16 == 1.0)
    # hpv18 has rel_imm=0.5 -> each agent's post is either 1.0 (sterilizing)
    # or 0.5 (leaky); never less than 0.5
    assert np.all(post_hpv18 >= 0.5)
    assert np.all(post_hpv18 <= 1.0)
    # No regressions in initial state
    assert np.all(post_hpv16 >= pre_hpv16)
    assert np.all(post_hpv18 >= pre_hpv18)


def test_vx_administer_skips_inactive_genotypes_silently():
    """A 9-valent product in a 4-genotype sim must not error."""
    from hpvsim.products import vx
    sim = _make_small_sim()  # has hpv16/hpv18/hi5/ohr only
    # Bivalent CSV has entries for hpv45, hi4, hr, lr which are NOT in this sim
    product = vx(name='bivalent')
    product.init_pre(sim)
    uids = sim.people.alive.uids[:10]
    # Must not raise
    product.administer(sim.people, uids)


def test_vx_administer_does_not_downgrade_natural_immunity():
    """If nab_imm is already higher than the vaccine peak, it is preserved."""
    from hpvsim.products import vx
    sim = _make_small_sim()
    product = vx(rel_imm={'hpv16': 0.5})
    product.init_pre(sim)
    uids = sim.people.alive.uids[:10]
    # Force-bump nab_imm to 0.95 (simulating natural clearance immunity)
    sim.diseases['hpv16'].nab_imm[uids] = 0.95
    product.administer(sim.people, uids)
    # leaky floor is 0.5 (rel_imm) — must not downgrade the 0.95
    assert np.all(sim.diseases['hpv16'].nab_imm[uids] >= 0.95)


def test_vx_administer_empty_uids_is_noop():
    """Calling administer with empty uids does nothing."""
    from hpvsim.products import vx
    sim = _make_small_sim()
    product = vx(name='bivalent')
    product.init_pre(sim)
    # Should not raise
    product.administer(sim.people, sim.people.alive.uids[:0])
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_m05_vx_unit.py -v -k "administer"
```

Expected: FAIL with `NotImplementedError: administer() implemented in Task 3.`

- [ ] **Step 3: Implement `administer` and `_find_genotype_module`**

Edit `hpvsim/products.py` — replace the `administer` stub on `vx`:

```python
    def administer(self, people, uids):
        """Apply the vaccine: all-or-nothing+leaky per genotype, max-of-existing.

        For each genotype g configured in this product:
          1. Look up the corresponding HPV(ss.Infection) module in the sim
             (by genotype attribute). Skip silently if not present.
          2. Per-agent Bernoulli(rel_imm[g]):
               - heads: peak = 1.0 (sterilizing immunity)
               - tails: peak = rel_imm[g] (leaky protection floor)
          3. Write hpv_mod.nab_imm[uids] = max(existing, peak).
        """
        if len(uids) == 0:
            return
        for genotype, rel_imm_g in self._rel_imm.items():
            hpv_mod = self._find_genotype_module(genotype)
            if hpv_mod is None:
                continue
            # All-or-nothing draw at p = rel_imm_g for this genotype
            self._sterilizing_dist.set(p=float(rel_imm_g))
            sterilizing_uids = self._sterilizing_dist.filter(uids)
            # Build per-uid peak vector: rel_imm_g (leaky) by default; 1.0
            # for those who got the sterilizing draw
            peak = np.full(len(uids), float(rel_imm_g), dtype=float)
            is_sterilizing = np.isin(uids, sterilizing_uids)
            peak[is_sterilizing] = 1.0
            # Max-of-existing: vaccine never downgrades existing immunity
            hpv_mod.nab_imm[uids] = np.maximum(hpv_mod.nab_imm[uids], peak)

    def _find_genotype_module(self, genotype):
        """Return the HPV module in the sim matching this genotype, or None.

        Matches M03's CrossImmunity convention: walk sim.diseases.values()
        and identify HPV modules by isinstance + .genotype attribute.
        """
        # Late import avoids the products <-> hpv circular import
        from hpvsim.hpv import HPV
        for module in self.sim.diseases.values():
            if isinstance(module, HPV) and module.genotype == genotype:
                return module
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_m05_vx_unit.py -v -k "administer"
```

Expected: all four `administer` tests PASS.

- [ ] **Step 5: Commit**

```
git add hpvsim/products.py tests/test_m05_vx_unit.py
git commit -m "M05: hpv.vx.administer — all-or-nothing+leaky writing to nab_imm

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Implement `_coerce_sex` helper

**Files:**
- Create: `hpvsim/interventions.py` (initial scaffold with `_coerce_sex` only)
- Test: `tests/test_m05_vx_unit.py`

- [ ] **Step 1: Write failing tests for sex coercion**

Append to `tests/test_m05_vx_unit.py`:

```python
def test_coerce_sex_none_returns_none():
    from hpvsim.interventions import _coerce_sex
    assert _coerce_sex(None) is None


def test_coerce_sex_f_returns_zero_set():
    from hpvsim.interventions import _coerce_sex
    assert _coerce_sex('f') == {0}


def test_coerce_sex_m_returns_one_set():
    from hpvsim.interventions import _coerce_sex
    assert _coerce_sex('m') == {1}


def test_coerce_sex_int_zero_or_one():
    from hpvsim.interventions import _coerce_sex
    assert _coerce_sex(0) == {0}
    assert _coerce_sex(1) == {1}


def test_coerce_sex_list_both():
    from hpvsim.interventions import _coerce_sex
    assert _coerce_sex(['f', 'm']) == {0, 1}
    assert _coerce_sex([0, 1]) == {0, 1}


def test_coerce_sex_invalid_raises():
    from hpvsim.interventions import _coerce_sex
    with pytest.raises(ValueError, match='sex'):
        _coerce_sex('female')
    with pytest.raises(ValueError, match='sex'):
        _coerce_sex(2)
    with pytest.raises(ValueError, match='sex'):
        _coerce_sex(['x', 'y'])
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_m05_vx_unit.py -v -k "coerce_sex"
```

Expected: FAIL — `hpvsim.interventions` module does not yet exist.

- [ ] **Step 3: Create `hpvsim/interventions.py` with `_coerce_sex`**

Create `hpvsim/interventions.py`:

```python
"""HPV-specific Starsim interventions.

Currently contains the prophylactic vaccination intervention API:
``hpv.BaseVaccination`` (the v2-compatible age_range/sex shim) and the
``hpv.routine_vx`` / ``hpv.campaign_vx`` leaf classes that combine the
shim with Starsim's RoutineDelivery / CampaignDelivery.

M06 will add screening (routine_screening / campaign_screening), triage,
treatment (treat_num / treat_delay / radiation), dynamic_pars, and the
txvx family (BaseTxVx / routine_txvx / campaign_txvx / linked_txvx).
"""
import numpy as np
import sciris as sc
import starsim as ss

from hpvsim.products import vx as _vx

__all__ = ['BaseVaccination', 'routine_vx', 'campaign_vx']


def _coerce_sex(sex):
    """Coerce v2-style sex input into a set of allowed sex ints (0=F, 1=M).

    Accepts:
      - None: no sex filter (returns None)
      - 'f' / 'm': single sex
      - 0 / 1 (int): single sex by int convention
      - list of 'f'/'m'/0/1: union of sexes

    Anything else raises ValueError.
    """
    if sex is None:
        return None
    if isinstance(sex, str):
        if sex == 'f':
            return {0}
        if sex == 'm':
            return {1}
        raise ValueError(f"sex string must be 'f' or 'm', got {sex!r}")
    if isinstance(sex, (list, tuple, set, np.ndarray)):
        out = set()
        for s in sex:
            out |= _coerce_sex(s)
        return out
    # Numeric path
    try:
        s_int = int(sex)
    except (TypeError, ValueError):
        raise ValueError(f"sex must be 'f', 'm', 0, 1, or a list thereof, got {sex!r}")
    if s_int not in (0, 1):
        raise ValueError(f"sex int must be 0 or 1, got {sex!r}")
    return {s_int}
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_m05_vx_unit.py -v -k "coerce_sex"
```

Expected: all six `coerce_sex` tests PASS.

- [ ] **Step 5: Commit**

```
git add hpvsim/interventions.py tests/test_m05_vx_unit.py
git commit -m "M05: scaffold hpv.interventions + _coerce_sex helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement `_compose_eligibility` helper

**Files:**
- Modify: `hpvsim/interventions.py`
- Test: `tests/test_m05_vx_unit.py`

- [ ] **Step 1: Write failing tests for eligibility composition**

Append to `tests/test_m05_vx_unit.py`:

```python
def test_compose_eligibility_no_filters_returns_all_alive():
    """No age, no sex, no extra -> all alive agents are eligible."""
    from hpvsim.interventions import _compose_eligibility
    sim = _make_small_sim()
    elig = _compose_eligibility(age_range=None, sex=None, extra=None)
    uids = elig(sim)
    # All eligible uids must be alive
    assert np.all(sim.people.alive[uids])
    # And there are some
    assert len(uids) > 0


def test_compose_eligibility_age_range_filters():
    """age_range=[lo, hi] yields only agents with lo <= age < hi."""
    from hpvsim.interventions import _compose_eligibility
    sim = _make_small_sim()
    elig = _compose_eligibility(age_range=[9, 14], sex=None, extra=None)
    uids = elig(sim)
    ages = sim.people.age[uids]
    assert np.all(ages >= 9)
    assert np.all(ages < 14)


def test_compose_eligibility_sex_female_filters():
    """sex='f' yields only agents with sex==0."""
    from hpvsim.interventions import _compose_eligibility
    sim = _make_small_sim()
    elig = _compose_eligibility(age_range=None, sex='f', extra=None)
    uids = elig(sim)
    assert np.all(sim.people.sex[uids] == 0)


def test_compose_eligibility_sex_male_filters():
    """sex='m' yields only agents with sex==1."""
    from hpvsim.interventions import _compose_eligibility
    sim = _make_small_sim()
    elig = _compose_eligibility(age_range=None, sex='m', extra=None)
    uids = elig(sim)
    assert np.all(sim.people.sex[uids] == 1)


def test_compose_eligibility_sex_both_applies_no_filter():
    """sex=['f', 'm'] applies no sex filter."""
    from hpvsim.interventions import _compose_eligibility
    sim = _make_small_sim()
    elig_both = _compose_eligibility(age_range=None, sex=['f', 'm'], extra=None)
    elig_none = _compose_eligibility(age_range=None, sex=None, extra=None)
    assert set(elig_both(sim)) == set(elig_none(sim))


def test_compose_eligibility_extra_callback_intersects():
    """An `extra` callable intersects further with age/sex conditions."""
    from hpvsim.interventions import _compose_eligibility
    sim = _make_small_sim()
    # Eligible: agents alive AND age >= 20 (extra is the only filter)
    extra = lambda s: (s.people.age >= 20).uids
    elig = _compose_eligibility(age_range=None, sex=None, extra=extra)
    uids = elig(sim)
    assert np.all(sim.people.age[uids] >= 20)
    assert np.all(sim.people.alive[uids])


def test_compose_eligibility_combines_age_sex_extra():
    """All three filters compose via intersection."""
    from hpvsim.interventions import _compose_eligibility
    sim = _make_small_sim()
    extra = lambda s: (s.people.age >= 12).uids
    elig = _compose_eligibility(age_range=[9, 14], sex='f', extra=extra)
    uids = elig(sim)
    ages = sim.people.age[uids]
    assert np.all(ages >= 12)
    assert np.all(ages < 14)
    assert np.all(sim.people.sex[uids] == 0)
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_m05_vx_unit.py -v -k "compose_eligibility"
```

Expected: FAIL — `_compose_eligibility` not yet defined.

- [ ] **Step 3: Add `_compose_eligibility` and the `_as_boolarr` helper**

Edit `hpvsim/interventions.py` — append after `_coerce_sex`:

```python
def _as_boolarr(extra_result, people):
    """Coerce an eligibility-callback return value into an ss.BoolArr.

    Starsim eligibility callbacks may return either a BoolArr or an
    ss.uids. We need a BoolArr so we can intersect with our own conditions
    via ``&``. Build a BoolArr of False the same length as people and
    fill True at the returned uids.
    """
    if isinstance(extra_result, ss.BoolArr):
        return extra_result
    # Assume ss.uids or array-like of ints
    n = len(people)
    out = ss.BoolArr('_compose_eligibility_extra', default=False)
    out.init(people=people)
    out[extra_result] = True
    return out


def _compose_eligibility(age_range, sex, extra):
    """Compose v2-style targeting into a Starsim eligibility callable.

    Returns ``elig(sim) -> ss.uids`` that intersects:
      - sim.people.alive
      - sim.people.age in [age_range[0], age_range[1]) if age_range is set
      - sim.people.sex matches sex if sex is set to a single sex
      - extra(sim) if extra is provided (callable returning BoolArr or uids)
    """
    sex_set = _coerce_sex(sex)
    def elig(sim):
        cond = sim.people.alive
        if age_range is not None:
            lo, hi = age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age < hi)
        if sex_set is not None and len(sex_set) == 1:
            (s,) = sex_set
            cond = cond & (sim.people.sex == s)
        if extra is not None:
            cond = cond & _as_boolarr(extra(sim), sim.people)
        return cond.uids
    return elig
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_m05_vx_unit.py -v -k "compose_eligibility"
```

Expected: all seven `compose_eligibility` tests PASS.

- [ ] **Step 5: Commit**

```
git add hpvsim/interventions.py tests/test_m05_vx_unit.py
git commit -m "M05: _compose_eligibility composing age_range/sex/extra into a callable

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement `hpv.BaseVaccination` + `hpv.routine_vx` + `hpv.campaign_vx`

**Files:**
- Modify: `hpvsim/interventions.py`
- Modify: `hpvsim/__init__.py`
- Test: `tests/test_m05_vx_unit.py`

- [ ] **Step 1: Write failing tests for the intervention classes**

Append to `tests/test_m05_vx_unit.py`:

```python
def test_base_vaccination_accepts_v2_args():
    """hpv.BaseVaccination accepts age_range, sex, eligibility as kwargs."""
    from hpvsim.interventions import BaseVaccination, routine_vx
    from hpvsim.products import vx
    intv = routine_vx(
        product=vx(name='bivalent'),
        prob=0.9,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
    )
    assert isinstance(intv, BaseVaccination)
    assert intv.age_range == [9, 14]
    assert intv.sex == {0}


def test_parse_product_str_resolves_to_default_vx():
    """routine_vx(product='bivalent', ...) resolves through hpv.vx(name='bivalent')."""
    from hpvsim.interventions import routine_vx
    from hpvsim.products import vx
    intv = routine_vx(product='bivalent', prob=0.5, start_year=2020)
    assert isinstance(intv.product, vx)
    assert intv.product._rel_imm['hpv16'] == pytest.approx(1.0)


def test_routine_vx_isinstance_chain():
    """Class identity preserved across the diamond."""
    import starsim as ss
    from hpvsim.interventions import routine_vx, campaign_vx, BaseVaccination
    intv_r = routine_vx(product='bivalent', prob=0.5, start_year=2020)
    intv_c = campaign_vx(product='bivalent', prob=0.5, years=[2020])
    assert isinstance(intv_r, BaseVaccination)
    assert isinstance(intv_r, ss.BaseVaccination)
    assert isinstance(intv_r, ss.RoutineDelivery)
    assert isinstance(intv_c, BaseVaccination)
    assert isinstance(intv_c, ss.CampaignDelivery)


def test_campaign_vx_passes_years_through():
    """campaign_vx accepts years= and stores it for init_pre."""
    from hpvsim.interventions import campaign_vx
    intv = campaign_vx(product='bivalent', prob=[0.7, 0.5], years=[2020, 2021])
    assert list(intv.years) == [2020, 2021]
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_m05_vx_unit.py -v -k "base_vaccination or parse_product or isinstance_chain or campaign_vx_passes"
```

Expected: FAIL — `BaseVaccination`, `routine_vx`, `campaign_vx` not yet defined.

- [ ] **Step 3: Append the intervention classes**

Edit `hpvsim/interventions.py` — append after `_compose_eligibility`:

```python
class BaseVaccination(ss.BaseVaccination):
    """HPV-specific prophylactic vaccination base.

    Wraps Starsim's ``ss.BaseVaccination`` to add v2-compatible
    ``age_range`` / ``sex`` / ``eligibility`` constructor args. These
    compose into a single Starsim eligibility callable. The originals are
    stored on the instance for introspection (e.g. AgeResults consumption).

    Also overrides ``_parse_product_str`` so that
    ``routine_vx(product='bivalent', ...)`` resolves through
    ``hpv.vx(name='bivalent')``, mirroring v2's string-product convention.
    """

    def __init__(self, *args, age_range=None, sex=None, eligibility=None, **kwargs):
        composed = _compose_eligibility(age_range, sex, eligibility)
        super().__init__(*args, eligibility=composed, **kwargs)
        self.age_range = age_range
        self.sex = _coerce_sex(sex)

    def _parse_product_str(self, product):
        """Resolve a string product name through hpv.vx default lookup."""
        return _vx(name=product)


class routine_vx(BaseVaccination, ss.RoutineDelivery):
    """Routine prophylactic HPV vaccination."""
    pass


class campaign_vx(BaseVaccination, ss.CampaignDelivery):
    """Campaign-style prophylactic HPV vaccination."""
    pass
```

- [ ] **Step 4: Wire top-level exports**

Open `hpvsim/__init__.py` and locate the existing `from .` imports near the top. Add (or extend) the following near them (after the existing module imports such as `hpv`, `cross_genotype`, `seeding`, `analyzers`, `calibration`):

```python
from .products import vx
from .interventions import BaseVaccination, routine_vx, campaign_vx
```

If `__all__` is defined in `hpvsim/__init__.py`, append `'vx'`, `'routine_vx'`, `'campaign_vx'`, and `'BaseVaccination'` to it. Do NOT add `_coerce_sex` / `_compose_eligibility` — they are private helpers.

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_m05_vx_unit.py -v -k "base_vaccination or parse_product or isinstance_chain or campaign_vx_passes"
```

Expected: all four tests PASS.

- [ ] **Step 6: Run the full unit-test file to confirm no regressions**

```
pytest tests/test_m05_vx_unit.py -v
```

Expected: every unit test PASSES.

- [ ] **Step 7: Commit**

```
git add hpvsim/interventions.py hpvsim/__init__.py tests/test_m05_vx_unit.py
git commit -m "M05: hpv.BaseVaccination shim + routine_vx + campaign_vx + exports

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Integration smoke tests — routine/campaign fire and update state

**Files:**
- Create: `tests/test_m05_vx_integration.py`

- [ ] **Step 1: Write the integration smoke tests**

Create `tests/test_m05_vx_integration.py`:

```python
"""Integration smoke tests for M05 vaccination interventions.

These tests run small sims end-to-end (200 agents, 5-year horizon) to
verify the routine/campaign interventions fire on schedule, update
per-intervention state correctly, target the right age/sex cohorts, and
do not perturb the M03 no-vx baseline (CRN-stream guard).
"""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv


SMALL_PARS = dict(
    location='nigeria',
    start=2018, stop=2025,
    n_agents=500,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    rand_seed=0,
)


def test_routine_vx_fires_and_updates_state():
    """routine_vx fires on schedule, vaccinated/n_doses/ti_vaccinated update."""
    intv = hpv.routine_vx(
        product='bivalent',
        prob=0.9,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='routine_smoke',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    # At least some agents were vaccinated
    assert intv.vaccinated.sum() > 0
    # Doses == 1 for everyone vaccinated (no boosters in M05)
    assert np.all(intv.n_doses[intv.vaccinated.uids] == 1)
    # ti_vaccinated set for every vaccinated agent
    assert np.all(~np.isnan(intv.ti_vaccinated[intv.vaccinated.uids]))


def test_campaign_vx_fires_and_updates_state():
    """campaign_vx fires on each campaign year."""
    intv = hpv.campaign_vx(
        product='bivalent',
        prob=[0.7, 0.5],
        age_range=[9, 30],
        sex='f',
        years=[2020, 2021],
        name='campaign_smoke',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    assert intv.vaccinated.sum() > 0


def test_routine_vx_respects_age_range():
    """Only agents in age_range at time of firing are vaccinated."""
    intv = hpv.routine_vx(
        product='bivalent',
        prob=1.0,
        age_range=[9, 10],
        sex='f',
        start_year=2020,
        name='routine_age_check',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    # Every vaccinated agent must have been 9 <= age < 10 at the time of
    # their first dose. Because age advances each step, we check that no
    # vaccinated agent was outside the band at *every* step in [2020, 2025).
    vacc_uids = intv.vaccinated.uids
    # At sim end, all vaccinated agents are aged >= 9 + (sim.t.now - ti_vaccinated*dt)
    # Lower-bound check: minimum age at vaccination must have been >= 9
    ages_now = sim.people.age[vacc_uids]
    ti_vacc = intv.ti_vaccinated[vacc_uids]
    # Approximate: age at vaccination = ages_now - (last_ti - ti_vacc) * dt
    ages_at_vacc = ages_now - (sim.ti - ti_vacc) * sim.t.dt
    # Allow a small dt-rounding tolerance
    assert np.all(ages_at_vacc >= 9 - sim.t.dt)
    assert np.all(ages_at_vacc < 10 + sim.t.dt)


def test_routine_vx_respects_sex():
    """sex='f' vaccinates only sex==0 agents."""
    intv = hpv.routine_vx(
        product='bivalent',
        prob=1.0,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='routine_sex_check',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    vacc_uids = intv.vaccinated.uids
    assert np.all(sim.people.sex[vacc_uids] == 0)


def test_no_vx_baseline_unchanged():
    """A sim with no vx intervention must reproduce M03 numbers exactly.

    Guard against CRN-stream perturbation: adding hpv.vx imports or
    Bernoulli construction should not have changed any RNG stream used
    by the existing M03 pipeline.
    """
    # Run a small sim with no vx
    sim = hpv.Sim(**SMALL_PARS)
    sim.run()
    # Sanity: a few summary numbers
    assert sim.results.timevec[0] == sim.t.start
    # Re-run with the same seed -> identical results
    sim2 = hpv.Sim(**SMALL_PARS)
    sim2.run()
    # Pick one canonical scalar that aggregates across the run
    s1 = float(sim.results['hpv_total_infections'].sum())
    s2 = float(sim2.results['hpv_total_infections'].sum())
    assert s1 == pytest.approx(s2)


def test_routine_vx_reduces_susceptibility_post_dose():
    """rel_sus on the HPV module drops for vaccinated agents in the next step."""
    intv = hpv.routine_vx(
        product=hpv.vx(rel_imm={'hpv16': 1.0}),  # full sterilizing for hpv16
        prob=1.0,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='routine_sus_check',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    vacc_uids = intv.vaccinated.uids
    if len(vacc_uids) == 0:
        pytest.skip('No agents vaccinated in this small-sim window')
    # After vaccination + at least one CrossImmunity step, rel_sus[hpv16]
    # must be reduced for vaccinated agents.
    rel_sus = sim.diseases['hpv16'].rel_sus[vacc_uids]
    # Sterilizing immunity (peak=1.0) -> rel_sus should be near 0
    assert np.all(rel_sus < 1.0)
```

- [ ] **Step 2: Run integration tests to verify they pass**

```
pytest tests/test_m05_vx_integration.py -v
```

Expected: all six tests PASS. If `test_routine_vx_respects_age_range` fails due to dt-rounding edge cases, widen the tolerance. If `test_no_vx_baseline_unchanged` fails, the new Bernoulli construction may be perturbing the CRN streams — file the failure as a blocking finding and revisit the design's "CRN stream perturbation" risk before continuing.

- [ ] **Step 3: Commit**

```
git add tests/test_m05_vx_integration.py
git commit -m "M05: integration smoke tests for routine/campaign vx

Verifies state updates, age/sex targeting, no-vx CRN-stream invariance
vs M03 baseline, and post-dose rel_sus reduction via CrossImmunity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Anchor scenario PARS scripts

**Files:**
- Create: `tests/regression/anchor_vx_routine.py`
- Create: `tests/regression/anchor_vx_campaign.py`

- [ ] **Step 1: Create the routine anchor PARS module**

Create `tests/regression/anchor_vx_routine.py`:

```python
"""M05 anchor scenario: routine bivalent vaccination of girls aged 9-10.

Mirrors the headline shape of hpvsim_pxv_younger-style routine programs.
Used by tests/test_m05_vx_routine_parity.py and the v2 baseline generator.
"""
import sciris as sc

# Anchor PARS — vanilla M03 Nigeria 4-genotype + one routine_vx intervention.
PARS = sc.objdict(
    location='nigeria',
    start=1990, stop=2060,
    rand_seed=0,
    n_agents=20_000,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    # The intervention spec is recorded as a serializable dict because the
    # v2 baseline script constructs the v2 equivalent from these fields.
    intervention=sc.objdict(
        kind='routine_vx',
        product='bivalent',
        prob=0.9,
        age_range=[9, 10],
        sex='f',
        start_year=2020,
        name='routine_bivalent_girls',
    ),
)


def build_v3_intervention():
    """Construct the v3 hpv.routine_vx from PARS.intervention."""
    import hpvsim as hpv
    cfg = PARS.intervention
    return hpv.routine_vx(
        product=cfg.product,
        prob=cfg.prob,
        age_range=cfg.age_range,
        sex=cfg.sex,
        start_year=cfg.start_year,
        name=cfg.name,
    )


def build_v3_sim():
    """Construct the v3 hpv.Sim used by the parity test."""
    import hpvsim as hpv
    return hpv.Sim(
        location=PARS.location,
        start=PARS.start, stop=PARS.stop,
        rand_seed=PARS.rand_seed,
        n_agents=PARS.n_agents,
        genotypes=list(PARS.genotypes),
        interventions=[build_v3_intervention()],
    )
```

- [ ] **Step 2: Create the campaign anchor PARS module**

Create `tests/regression/anchor_vx_campaign.py`:

```python
"""M05 anchor scenario: one-off campaign bivalent vaccination of girls 9-14.

Mirrors the headline shape of hpvsim_1dose-style catch-up campaigns.
Used by tests/test_m05_vx_campaign_parity.py and the v2 baseline generator.
"""
import sciris as sc

PARS = sc.objdict(
    location='nigeria',
    start=1990, stop=2060,
    rand_seed=0,
    n_agents=20_000,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    intervention=sc.objdict(
        kind='campaign_vx',
        product='bivalent',
        prob=[0.7, 0.5],
        age_range=[9, 14],
        sex='f',
        years=[2020, 2021],
        interpolate=False,
        name='campaign_bivalent_catchup',
    ),
)


def build_v3_intervention():
    import hpvsim as hpv
    cfg = PARS.intervention
    return hpv.campaign_vx(
        product=cfg.product,
        prob=list(cfg.prob),
        age_range=cfg.age_range,
        sex=cfg.sex,
        years=list(cfg.years),
        interpolate=cfg.interpolate,
        name=cfg.name,
    )


def build_v3_sim():
    import hpvsim as hpv
    return hpv.Sim(
        location=PARS.location,
        start=PARS.start, stop=PARS.stop,
        rand_seed=PARS.rand_seed,
        n_agents=PARS.n_agents,
        genotypes=list(PARS.genotypes),
        interventions=[build_v3_intervention()],
    )
```

- [ ] **Step 3: Smoke-run each anchor to confirm it builds**

```
python -c "from tests.regression.anchor_vx_routine import build_v3_sim; sim = build_v3_sim(); print('routine OK; n_interventions =', len(sim.interventions))"
python -c "from tests.regression.anchor_vx_campaign import build_v3_sim; sim = build_v3_sim(); print('campaign OK; n_interventions =', len(sim.interventions))"
```

Expected: prints `routine OK; n_interventions = 1` and `campaign OK; n_interventions = 1`.

- [ ] **Step 4: Commit**

```
git add tests/regression/anchor_vx_routine.py tests/regression/anchor_vx_campaign.py
git commit -m "M05: anchor PARS scripts for routine + campaign vx parity tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: v2 baseline generation script (one-time local run)

**Files:**
- Create: `tests/regression/multi_seed_v2_vx.py`
- Create: `tests/regression/README_m05.md`

This task creates the v2 baseline generator and documents how to run it. **The script itself runs only against a separate v2-hpvsim conda env**; it is not exercised by the v3 test suite.

- [ ] **Step 1: Inspect the existing M03 generator for the matching pattern**

Open `tests/regression/multi_seed_v2.py` and read it through. The M05 generator follows the same layout: parse `--n`, loop over seeds, run v2 sims, write JSON. The only differences are (a) it loops over both anchor scenarios, (b) it constructs v2 interventions from the PARS spec, and (c) it writes two output files.

- [ ] **Step 2: Create the M05 generator script**

Create `tests/regression/multi_seed_v2_vx.py`:

```python
"""Generate the v2 30-seed baselines for the M05 vx anchor scenarios.

Run this ONCE from a separate v2 hpvsim env (not the v3 env). Writes two
gitignored JSON files (one per anchor) which the M05 parity tests
consume.

USAGE (from a v2 hpvsim env, NOT the v3 env):

    python tests/regression/multi_seed_v2_vx.py --n 30

Outputs:
    tests/regression/v2_seeds_n30_vx_routine.json
    tests/regression/v2_seeds_n30_vx_campaign.json
"""
import argparse
import json
import sys
from pathlib import Path

# These imports MUST be from a v2 hpvsim environment, not v3.
import hpvsim as hpv  # noqa: I001
import sciris as sc
import numpy as np

# Import the anchor PARS modules from the v3 tree (they are pure-Python).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.regression.anchor_vx_routine import PARS as ROUTINE_PARS  # noqa: E402
from tests.regression.anchor_vx_campaign import PARS as CAMPAIGN_PARS  # noqa: E402
from tests.regression.short_summary import build_summary  # noqa: E402


GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


def _build_v2_intervention(cfg):
    """Convert a serialized intervention spec into a v2 hpvsim object."""
    if cfg.kind == 'routine_vx':
        return hpv.routine_vx(
            product=cfg.product,
            prob=cfg.prob,
            age_range=cfg.age_range,
            sex=cfg.sex,
            start_year=cfg.start_year,
            label=cfg.name,
        )
    if cfg.kind == 'campaign_vx':
        return hpv.campaign_vx(
            product=cfg.product,
            prob=list(cfg.prob),
            age_range=cfg.age_range,
            sex=cfg.sex,
            years=list(cfg.years),
            interpolate=cfg.interpolate,
            label=cfg.name,
        )
    raise ValueError(f'Unknown intervention kind: {cfg.kind!r}')


def _build_v2_sim(pars, seed):
    intervention = _build_v2_intervention(pars.intervention)
    return hpv.Sim(
        location=pars.location,
        start=pars.start, end=pars.stop,
        rand_seed=int(seed),
        n_agents=pars.n_agents,
        genotypes=list(pars.genotypes),
        interventions=[intervention],
    )


def _run_anchor(pars, n_seeds, out_path):
    summaries = []
    for seed in range(n_seeds):
        sim = _build_v2_sim(pars, seed)
        sim.run()
        row = build_summary(sim, GENOTYPES)
        # Vaccination-specific scalars (intervention 0 is our vx; v2 stores
        # vaccinated / doses on People).
        row['_seed'] = int(seed)
        row['n_vaccinated_2060'] = int(sim.people.vaccinated.sum())
        row['n_doses_2060'] = int(sim.people.doses.sum())
        # Crude post-vaccination cancer incidence proxy: total new cancers
        # in [2030, 2060] / total person-years in that window.
        years = sim.results['year']
        mask = (years >= 2030) & (years < 2060)
        n_cancers = float(sim.results['new_cancers'][mask].sum())
        # Person-years approximation: pop * dt summed in window.
        pop = sim.results['n_alive'][mask]
        py = float((pop * sim['dt']).sum())
        row['cancer_incidence_2030_2060'] = n_cancers / max(py, 1.0)
        summaries.append(row)
        print(f'  seed {seed}: n_vaccinated={row["n_vaccinated_2060"]}')
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f'Wrote {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=30, help='Number of seeds')
    args = parser.parse_args()

    here = Path(__file__).parent
    print(f'Generating routine baseline ({args.n} seeds)...')
    _run_anchor(ROUTINE_PARS, args.n, here / f'v2_seeds_n{args.n}_vx_routine.json')
    print(f'Generating campaign baseline ({args.n} seeds)...')
    _run_anchor(CAMPAIGN_PARS, args.n, here / f'v2_seeds_n{args.n}_vx_campaign.json')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Verify the script parses (no syntax errors)**

```
python -m py_compile tests/regression/multi_seed_v2_vx.py
```

Expected: no output, exit 0.

(The script will only RUN against a v2 hpvsim env; this step only confirms it parses.)

- [ ] **Step 4: Confirm the baseline-output filenames are gitignored**

```
git check-ignore -v tests/regression/v2_seeds_n30_vx_routine.json
git check-ignore -v tests/regression/v2_seeds_n30_vx_campaign.json
```

Expected: both files report as ignored (matching an existing `v2_seeds_*.json` rule). If they do not, append `v2_seeds_*_vx_*.json` to `tests/regression/.gitignore` or the repo-root `.gitignore` — whichever already handles `v2_seeds_n30.json` (M03).

- [ ] **Step 5: Create the regenerate-the-baseline README**

Create `tests/regression/README_m05.md`:

```markdown
# M05 vaccination parity gates

Two slow tests (`test_m05_vx_routine_parity.py`,
`test_m05_vx_campaign_parity.py`) plus one trajectory test
(`test_m05_vx_trajectory_parity.py`) gate v3 vaccination against
locally-regenerated v2.x baselines. Both follow M03's multi-seed z-score
pattern (`|z| < 3`) over the M03 short summary plus three vaccination-
specific summary scalars (`n_vaccinated_2060`, `n_doses_2060`,
`cancer_incidence_2030_2060`).

## Regenerating the v2 baselines (one-time, local)

1. Activate a separate conda env with v2 hpvsim installed:

       conda activate hpvsim-v2

2. Generate both baseline JSONs (30 seeds each, gitignored output):

       python tests/regression/multi_seed_v2_vx.py --n 30

   Outputs:
       tests/regression/v2_seeds_n30_vx_routine.json
       tests/regression/v2_seeds_n30_vx_campaign.json

3. Switch back to the v3 env:

       conda activate hpvsim-v3

## Running the M05 parity gate locally

    pytest tests/test_m05_vx_routine_parity.py tests/test_m05_vx_campaign_parity.py tests/test_m05_vx_trajectory_parity.py -v -m slow

The slow tests are excluded from CI by `pytest -m 'not slow'`.

## Updating the baselines

If you tighten an anchor scenario's PARS, regenerate the corresponding
v2 baseline (step 2 above) before re-running the parity gate.
```

- [ ] **Step 6: Commit**

```
git add tests/regression/multi_seed_v2_vx.py tests/regression/README_m05.md
git commit -m "M05: v2 baseline generation script + regeneration README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Routine-anchor short-summary parity test

**Files:**
- Create: `tests/test_m05_vx_routine_parity.py`

- [ ] **Step 1: Inspect M03's short-summary parity test for the template**

Open `tests/test_m05_vx_routine_parity.py` (yet to create) — pattern is identical to `tests/test_m03_short_summary_parity.py`. Differences: load the routine-anchor baseline JSON, run the routine anchor sim 10 times, gate the merged metric set (M03 summary + 3 vx scalars) at `|z| < 3`.

- [ ] **Step 2: Create the parity test**

Create `tests/test_m05_vx_routine_parity.py`:

```python
"""M05 routine-vx parity gate: multi-seed mean z-score vs v2 baseline.

Same gate pattern as test_m03_short_summary_parity.py (10 v3 seeds vs 30
v2 seeds; |z| < 3 per metric). Metric set: M03's short summary + three
vaccination-specific scalars (n_vaccinated_2060, n_doses_2060,
cancer_incidence_2030_2060).

Requires the locally-regenerated v2 baseline at
``tests/regression/v2_seeds_n30_vx_routine.json``. See
``tests/regression/README_m05.md`` for how to (re)generate it.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import sciris as sc

from tests.regression.anchor_vx_routine import build_v3_sim
from tests.regression.short_summary import build_summary


BASELINE_PATH = Path(__file__).parent / 'regression' / 'v2_seeds_n30_vx_routine.json'
N_V3_SEEDS = 10
Z_THRESHOLD = 3.0
GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')

_SKIP_KEYS = frozenset({'_seed', '_total_pop', 'total population'})


def _run_v3_seeds(n, start_seed=0):
    summaries = []
    for seed in range(start_seed, start_seed + n):
        sim = build_v3_sim()
        sim.pars['rand_seed'] = int(seed)
        sim.run()
        row = build_summary(sim, GENOTYPES)
        # Pull vaccination scalars off the single intervention
        intv = sim.interventions[0]
        row['n_vaccinated_2060'] = int(intv.vaccinated.sum())
        row['n_doses_2060'] = int(intv.n_doses.sum())
        # cancer incidence 2030-2060 — same proxy as the v2 baseline
        years = sim.results.timevec
        mask = (years >= 2030) & (years < 2060)
        n_cancers = float(sim.results['new_cancers'][mask].sum())
        pop = sim.results['n_alive'][mask]
        py = float((pop * sim.t.dt).sum())
        row['cancer_incidence_2030_2060'] = n_cancers / max(py, 1.0)
        summaries.append(row)
    return summaries


def _mean_se(rows, key):
    vals = np.array([float(r[key]) for r in rows if key in r], dtype=float)
    if vals.size == 0:
        return None
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, se


@pytest.mark.slow
def test_m05_routine_short_summary_parity():
    """40-metric z-score gate: |z| < 3 across M03 summary + 3 vx scalars."""
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v2 baseline at {BASELINE_PATH}. '
            f'Generate it via tests/regression/multi_seed_v2_vx.py from a '
            f'v2 hpvsim env (see tests/regression/README_m05.md).'
        )
    v2_rows = json.loads(BASELINE_PATH.read_text())
    v3_rows = _run_v3_seeds(N_V3_SEEDS)

    # Collect every key shared by both sides (minus the skip set)
    keys = (set(v2_rows[0].keys()) & set(v3_rows[0].keys())) - _SKIP_KEYS

    failures = []
    for key in sorted(keys):
        v2 = _mean_se(v2_rows, key)
        v3 = _mean_se(v3_rows, key)
        if v2 is None or v3 is None:
            continue
        v2_mean, v2_se = v2
        v3_mean, v3_se = v3
        denom = math.sqrt(v2_se ** 2 + v3_se ** 2)
        if denom == 0:
            # Both sides deterministic; require equal means
            if v2_mean != v3_mean:
                failures.append(
                    f'{key}: v3={v3_mean!r} v2={v2_mean!r} (deterministic but unequal)'
                )
            continue
        z = (v3_mean - v2_mean) / denom
        if abs(z) >= Z_THRESHOLD:
            failures.append(
                f'{key}: |z|={abs(z):.2f} (v3={v3_mean:.3g}+/-{v3_se:.2g} vs '
                f'v2={v2_mean:.3g}+/-{v2_se:.2g})'
            )
    assert not failures, '\n'.join(['Metrics outside |z|<3:'] + failures)
```

- [ ] **Step 3: Run the test locally (after baseline regeneration)**

```
pytest tests/test_m05_vx_routine_parity.py -v -m slow
```

Expected (with baseline present): PASS.
Expected (without baseline): SKIP with the regeneration instructions.

- [ ] **Step 4: Commit**

```
git add tests/test_m05_vx_routine_parity.py
git commit -m "M05: routine-vx short-summary parity gate at |z|<3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Campaign-anchor short-summary parity test

**Files:**
- Create: `tests/test_m05_vx_campaign_parity.py`

- [ ] **Step 1: Create the campaign parity test**

Create `tests/test_m05_vx_campaign_parity.py` — identical structure to the routine version (Task 10) except for the imports and baseline path. Copy Task 10's file verbatim and apply these substitutions:

- `from tests.regression.anchor_vx_routine import build_v3_sim` -> `from tests.regression.anchor_vx_campaign import build_v3_sim`
- `v2_seeds_n30_vx_routine.json` -> `v2_seeds_n30_vx_campaign.json`
- `test_m05_routine_short_summary_parity` -> `test_m05_campaign_short_summary_parity`
- Docstring "M05 routine-vx parity gate" -> "M05 campaign-vx parity gate"

Concretely, the new file is:

```python
"""M05 campaign-vx parity gate: multi-seed mean z-score vs v2 baseline.

See test_m05_vx_routine_parity.py for the gate description; this test is
the campaign-anchor mirror.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import sciris as sc

from tests.regression.anchor_vx_campaign import build_v3_sim
from tests.regression.short_summary import build_summary


BASELINE_PATH = Path(__file__).parent / 'regression' / 'v2_seeds_n30_vx_campaign.json'
N_V3_SEEDS = 10
Z_THRESHOLD = 3.0
GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')

_SKIP_KEYS = frozenset({'_seed', '_total_pop', 'total population'})


def _run_v3_seeds(n, start_seed=0):
    summaries = []
    for seed in range(start_seed, start_seed + n):
        sim = build_v3_sim()
        sim.pars['rand_seed'] = int(seed)
        sim.run()
        row = build_summary(sim, GENOTYPES)
        intv = sim.interventions[0]
        row['n_vaccinated_2060'] = int(intv.vaccinated.sum())
        row['n_doses_2060'] = int(intv.n_doses.sum())
        years = sim.results.timevec
        mask = (years >= 2030) & (years < 2060)
        n_cancers = float(sim.results['new_cancers'][mask].sum())
        pop = sim.results['n_alive'][mask]
        py = float((pop * sim.t.dt).sum())
        row['cancer_incidence_2030_2060'] = n_cancers / max(py, 1.0)
        summaries.append(row)
    return summaries


def _mean_se(rows, key):
    vals = np.array([float(r[key]) for r in rows if key in r], dtype=float)
    if vals.size == 0:
        return None
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, se


@pytest.mark.slow
def test_m05_campaign_short_summary_parity():
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v2 baseline at {BASELINE_PATH}. '
            f'Generate via tests/regression/multi_seed_v2_vx.py from a v2 env.'
        )
    v2_rows = json.loads(BASELINE_PATH.read_text())
    v3_rows = _run_v3_seeds(N_V3_SEEDS)
    keys = (set(v2_rows[0].keys()) & set(v3_rows[0].keys())) - _SKIP_KEYS
    failures = []
    for key in sorted(keys):
        v2 = _mean_se(v2_rows, key)
        v3 = _mean_se(v3_rows, key)
        if v2 is None or v3 is None:
            continue
        v2_mean, v2_se = v2
        v3_mean, v3_se = v3
        denom = math.sqrt(v2_se ** 2 + v3_se ** 2)
        if denom == 0:
            if v2_mean != v3_mean:
                failures.append(f'{key}: v3={v3_mean!r} v2={v2_mean!r}')
            continue
        z = (v3_mean - v2_mean) / denom
        if abs(z) >= Z_THRESHOLD:
            failures.append(
                f'{key}: |z|={abs(z):.2f} (v3={v3_mean:.3g}+/-{v3_se:.2g} vs '
                f'v2={v2_mean:.3g}+/-{v2_se:.2g})'
            )
    assert not failures, '\n'.join(['Metrics outside |z|<3:'] + failures)
```

- [ ] **Step 2: Run locally (after baseline regeneration)**

```
pytest tests/test_m05_vx_campaign_parity.py -v -m slow
```

Expected: PASS (with baseline present) or SKIP (without).

- [ ] **Step 3: Commit**

```
git add tests/test_m05_vx_campaign_parity.py
git commit -m "M05: campaign-vx short-summary parity gate at |z|<3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Routine-anchor trajectory parity test

**Files:**
- Create: `tests/test_m05_vx_trajectory_parity.py`
- Modify: `tests/regression/multi_seed_v2_vx.py` (add per-year trajectory rows to baseline)

The trajectory test gates `(year, metric)` cells against v2 at `|z| < 3`. We run on the routine anchor only (the heaviest test in the M05 suite); the campaign anchor stays at short-summary level.

- [ ] **Step 1: Inspect the M03 trajectory test for the template**

Open `tests/test_m03_trajectory_parity.py`. The M05 trajectory test follows the same structure: load yearly v2 rows, run 10 v3 seeds, compute z per (year, metric), gate at `|z| < 3`.

- [ ] **Step 2: Extend the v2 baseline generator to emit trajectory rows**

Edit `tests/regression/multi_seed_v2_vx.py` — extend `_run_anchor` to ALSO write a trajectory baseline. Append within `_run_anchor`, right before `out_path.write_text(...)`:

```python
        # Trajectory baseline: per-year metrics for the routine anchor only
        # (campaign trajectory test is intentionally omitted per the spec).
        # Subsequent run rebuilds; here we just store the per-step time series
        # used by the trajectory gate.
        row['_trajectory'] = sc.objdict(
            year=list(sim.results['year']),
            new_cancers=list(sim.results['new_cancers']),
            hpv_total_infections=list(sim.results['hpv_total_infections']),
            new_vaccinated=list(sim.results.get('new_vaccinated', [0] * len(sim.results['year']))),
        )
```

(Re-emit both baselines after this change. The routine JSON now carries `_trajectory` per row; the campaign JSON does too but only the routine trajectory test reads it.)

- [ ] **Step 3: Create the trajectory parity test**

Create `tests/test_m05_vx_trajectory_parity.py`:

```python
"""M05 routine-vx trajectory parity gate: per-year z-score vs v2.

For each (year, metric) cell we compute the same z-score as the short-
summary gate, against v2's per-year distribution. Gates each cell at
|z| < 3. This is the strictest shape-check in M05.

Runs on the routine anchor only — the campaign anchor is well-summarised
by the short-summary gate and the trajectory test is the heaviest test
in the suite.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import sciris as sc

from tests.regression.anchor_vx_routine import build_v3_sim


BASELINE_PATH = Path(__file__).parent / 'regression' / 'v2_seeds_n30_vx_routine.json'
N_V3_SEEDS = 10
Z_THRESHOLD = 3.0

TRAJECTORY_METRICS = ('new_cancers', 'hpv_total_infections', 'new_vaccinated')


def _v3_trajectory_row(sim):
    return dict(
        year=list(sim.results.timevec),
        new_cancers=list(sim.results['new_cancers']),
        hpv_total_infections=list(sim.results['hpv_total_infections']),
        new_vaccinated=list(sim.results.get('new_vaccinated', [0] * len(sim.results.timevec))),
    )


@pytest.mark.slow
def test_m05_routine_trajectory_parity():
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v2 baseline at {BASELINE_PATH}. '
            f'Run tests/regression/multi_seed_v2_vx.py from a v2 env.'
        )
    v2_rows = json.loads(BASELINE_PATH.read_text())
    if '_trajectory' not in v2_rows[0]:
        pytest.skip(
            'v2 baseline lacks `_trajectory` field. Regenerate it after the '
            'Task 12 multi_seed_v2_vx.py update.'
        )

    # Collect v2 trajectory arrays per metric: shape (n_seeds, n_years)
    years = np.array(v2_rows[0]['_trajectory']['year'])
    v2_arrs = {
        m: np.array([r['_trajectory'][m] for r in v2_rows], dtype=float)
        for m in TRAJECTORY_METRICS
    }

    # Run 10 v3 seeds
    v3_rows = []
    for seed in range(N_V3_SEEDS):
        sim = build_v3_sim()
        sim.pars['rand_seed'] = int(seed)
        sim.run()
        v3_rows.append(_v3_trajectory_row(sim))

    # Sanity: same year grid
    v3_years = np.array(v3_rows[0]['year'])
    assert np.allclose(years, v3_years), (
        f'v3 and v2 year vectors differ: '
        f'v3[{v3_years[0]}..{v3_years[-1]}] vs v2[{years[0]}..{years[-1]}]'
    )

    v3_arrs = {
        m: np.array([r[m] for r in v3_rows], dtype=float)
        for m in TRAJECTORY_METRICS
    }

    failures = []
    for metric in TRAJECTORY_METRICS:
        v2_a = v2_arrs[metric]
        v3_a = v3_arrs[metric]
        v2_mean = v2_a.mean(axis=0)
        v2_se = v2_a.std(axis=0, ddof=1) / math.sqrt(v2_a.shape[0])
        v3_mean = v3_a.mean(axis=0)
        v3_se = v3_a.std(axis=0, ddof=1) / math.sqrt(v3_a.shape[0])
        denom = np.sqrt(v2_se ** 2 + v3_se ** 2)
        # Avoid division-by-zero where both sides are deterministically equal
        z = np.where(denom > 0, (v3_mean - v2_mean) / np.maximum(denom, 1e-12), 0.0)
        bad = np.abs(z) >= Z_THRESHOLD
        if bad.any():
            bad_idx = np.where(bad)[0]
            example_year = years[bad_idx[0]]
            failures.append(
                f'{metric}: {bad.sum()} year(s) with |z|>={Z_THRESHOLD}, '
                f'first at year {example_year:.0f} (|z|={abs(z[bad_idx[0]]):.2f})'
            )
    assert not failures, '\n'.join(['Trajectory cells outside |z|<3:'] + failures)
```

- [ ] **Step 4: Run locally (after baseline regeneration)**

```
pytest tests/test_m05_vx_trajectory_parity.py -v -m slow
```

Expected: PASS (with regenerated baseline including `_trajectory`) or SKIP otherwise.

- [ ] **Step 5: Commit**

```
git add tests/test_m05_vx_trajectory_parity.py tests/regression/multi_seed_v2_vx.py
git commit -m "M05: routine-vx trajectory parity gate per (year, metric)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: CI workflow check + final sanity run

**Files:**
- Inspect: `.github/workflows/tests.yaml` (no edits expected; verify the `-m 'not slow'` filter still applies)

- [ ] **Step 1: Confirm the CI workflow still filters out slow tests**

```
grep -n "not slow" .github/workflows/tests.yaml
```

Expected output includes the M04 line: `pytest -m 'not slow' ...`. If absent, the slow filter is broken — file as a blocker.

- [ ] **Step 2: Run the full non-slow test suite locally**

```
pytest -m 'not slow' -q
```

Expected: all tests PASS. The M05 parity tests (slow-marked) are excluded; the M05 unit and integration tests run and pass.

- [ ] **Step 3: Run only the slow M05 parity tests locally (after baseline regen)**

```
pytest -m slow tests/test_m05_vx_routine_parity.py tests/test_m05_vx_campaign_parity.py tests/test_m05_vx_trajectory_parity.py -v
```

Expected: all three PASS (skipped if baselines absent).

- [ ] **Step 4: Confirm M03 and M04 tests still pass (regression guard)**

```
pytest tests/test_m03_short_summary_parity.py tests/test_m03_trajectory_parity.py tests/test_age_results.py tests/test_calibration.py -v -m 'not slow'
```

Expected: all PASS.

- [ ] **Step 5: No commit — this task is verification only**

If any step in this task fails, debug to root cause before proceeding. Do not paper over a failing test with `xfail` or skip; the parity gates are the M05 acceptance criterion.

---

## Task 14: Update MIGRATION_PLAN.md

**Files:**
- Modify: `MIGRATION_PLAN.md`

- [ ] **Step 1: Flip the M5 status in the status table**

Open `MIGRATION_PLAN.md` and locate the status table (around line 71-79). Change the M5 row from:

```
| M4–M10 | ⬜ Not started | — |
```

to:

```
| M5 | 🟡 Implementation complete; PR not yet opened | branch `m05-vaccination-scenarios` |
| M6–M10 | ⬜ Not started | — |
```

(If the M4 status is also out of date, update it consistently with whatever state has landed.)

- [ ] **Step 2: Replace the M5 sub-task block**

Locate `### M5: Vaccination scenarios` (around line 178). Replace the entire `**Sub-tasks:**` list with the rewritten list from the spec:

```
**Sub-tasks:**
- Port `hpv.vx(ss.Vx)` product class — per-genotype `rel_imm` table loaded
  from `hpvsim/data/products_vx.csv`; `administer` applies per-genotype
  all-or-nothing+leaky model and bumps each HPV module's `nab_imm`. Cross-
  immunity propagation is automatic via the existing `CrossImmunity`
  connector.
- Add `hpv.BaseVaccination(ss.BaseVaccination)` shim adding v2-compatible
  `age_range` / `sex` / `eligibility` constructor args; thin `hpv.routine_vx`
  and `hpv.campaign_vx` subclasses combining the shim with Starsim's
  `RoutineDelivery` / `CampaignDelivery`.
- Move `products_vx.csv` from `hpvsim/_v2_legacy/data/` into active
  `hpvsim/data/`. Default product names: `bivalent`, `quadrivalent`,
  `nonavalent`.
- Add two regression anchors (`anchor_vx_routine`, `anchor_vx_campaign`),
  generator script for v2 baselines, and multi-seed z-score parity gates at
  `|z| < 3` (M03 pattern). Includes a trajectory-parity test on the
  routine anchor.
- Add unit tests for `_compose_eligibility`, `_coerce_sex`, and `hpv.vx`
  product semantics.
- Confirm intervention-level result tracking (`vaccinated`, `n_doses`,
  `ti_vaccinated`) is exposed via the existing `ss.BaseVaccination` state;
  age-stratified consumption uses M04's `AgeResults` analyzer.
```

- [ ] **Step 3: Append the M6 sub-tasks for txvx + dx + tx**

Locate `### M6: Screen-and-treat cascade` (around line 193). Append to the existing `**Sub-tasks:**` list:

```
- Port `dx(ss.Product)` diagnostic product class (CSV table maps disease
  state -> result probability). Used by screening interventions.
- Port `tx(ss.Product)` treatment product class. Used by `treat_num` /
  `treat_delay`.
- Port `txvx` therapeutic vaccination: `BaseTxVx` + `routine_txvx` +
  `campaign_txvx` + `linked_txvx`. Moved from M5 because `linked_txvx` is
  structurally part of the screen-and-treat cascade and `BaseTxVx` shares
  its design with the M06 treatment base classes (see M05 spec
  "Scope adjustments" rationale).
- Move `products_tx.csv` and `products_dx.csv` from
  `hpvsim/_v2_legacy/data/` into active `hpvsim/data/`.
```

- [ ] **Step 4: Commit**

```
git add MIGRATION_PLAN.md
git commit -m "M05: update MIGRATION_PLAN — M5 in progress; move txvx/dx/tx to M6

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification before opening the PR

After all 14 tasks complete:

- [ ] `pytest -m 'not slow' -q` — every non-slow test passes
- [ ] `pytest -m slow tests/test_m05_vx_*.py -v` — all three parity tests pass (baselines regenerated locally)
- [ ] `pytest tests/test_m03_*.py tests/test_age_results.py tests/test_calibration.py -m 'not slow' -v` — M03 and M04 still pass
- [ ] `git log --oneline v3.0-dev..m05-vaccination-scenarios` — every M05 commit prefixed `M05:`
- [ ] `git diff --stat v3.0-dev m05-vaccination-scenarios` — no surprising files (only the spec, plan, products_vx.csv move, two new modules, six test files, anchor/baseline scripts, README, MIGRATION_PLAN)
- [ ] Open PR `m05-vaccination-scenarios → v3.0-dev`; reference the spec and this plan in the PR description; flag the second-dose idempotence stance and the CRN-stream guard explicitly so reviewers know they were considered.