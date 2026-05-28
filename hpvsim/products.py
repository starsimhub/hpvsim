"""HPV-specific Starsim products.

Contains:
  - hpv.vx: prophylactic vaccine product (M05)
  - hpv.dx: per-genotype multinomial diagnostic classifier (M06)

M06 will additionally add hpv.tx (treatment), hpv.txvx (therapeutic
vaccine), and hpv.radiation (cancer treatment) — see plan
docs/superpowers/plans/2026-05-27-hpvsim-m06-test-and-treat-cascade.md.
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import starsim as ss

__all__ = ['vx', 'dx', 'tx', 'txvx', 'radiation']

_PRODUCT_CSV = Path(__file__).parent / 'data' / 'products_vx.csv'


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


@functools.lru_cache(maxsize=1)
def _load_vx_products():
    """Load the CSV and return a mapping of product name -> {genotype: rel_imm}.

    Cached at module load — the CSV is small (~24 rows) and never changes
    at runtime.
    """
    df = pd.read_csv(_PRODUCT_CSV)
    expected_cols = {'name', 'genotype', 'rel_imm'}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f'products_vx.csv missing required columns: {sorted(missing)}'
        )
    out = {}
    for name, group in df.groupby('name', sort=False):
        out[name] = dict(zip(group['genotype'], group['rel_imm'].astype(float)))
    return out


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
        valid = ', '.join(products.keys())
        raise ValueError(
            f'Unknown vx product name {name!r}. Valid names: {valid}.'
        )
    return dict(products[name])


_DX_CSV   = Path(__file__).parent / 'data' / 'products_dx.csv'
_TX_CSV   = Path(__file__).parent / 'data' / 'products_tx.csv'
_TXVX_CSV = Path(__file__).parent / 'data' / 'products_txvx.csv'


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


class vx(ss.Vx):
    """HPV multi-genotype prophylactic vaccine.

    Constructed with EITHER ``name`` (looks up the per-genotype rel_imm from
    ``hpvsim/data/products_vx.csv``) OR ``rel_imm`` (explicit per-genotype
    dict). Default product names: ``'bivalent'``, ``'quadrivalent'``,
    ``'nonavalent'``.

    The vaccine model mirrors v2's two-parameter architecture:

    - ``sterilizing_p`` (default 0.95): per-agent Bernoulli probability of
      sterilizing immunity, drawn ONCE per agent (not per genotype). Matches
      v2's hardcoded ``imm_init=0.95`` in ``default_vx``.
    - ``rel_imm[g]`` from the CSV: per-genotype cross-protection coefficient.
      Sterilizing agents receive ``vax_imm[g] = rel_imm[g]``; leaky agents
      receive ``vax_imm[g] = rel_imm[g] * sterilizing_p``.

    The effective per-genotype protection is approximately
    ``0.9975 * rel_imm[g]``, matching v2 to within ~0.25 percentage points.
    Existing ``vax_imm`` is never downgraded (max-of-existing semantics).

    Vaccine immunity is written to ``vax_imm`` (NOT ``nab_imm``). The
    ``CrossImmunity`` connector applies ``vax_imm`` directly per-genotype
    without flowing it through the cross-immunity matrix, so the CSV's
    per-genotype ``rel_imm`` values are the complete vaccine cross-protection
    profile — matching v2 semantics where vaccine immunity does not amplify
    into cross-protection against non-target genotypes.
    """

    def __init__(self, name=None, rel_imm=None, sterilizing_p=0.95, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            name=name,
            rel_imm=rel_imm,
            sterilizing_p=sterilizing_p,
        )
        self.rel_imm = _resolve_vx_pars(name, rel_imm)
        # CRN-safe Bernoulli; p set per administer call.
        self._sterilizing_dist = ss.bernoulli(p=0.0)

    def administer(self, people, uids):
        """Apply the vaccine: per-agent all-or-nothing sterilizing draw,
        scaled per genotype by the CSV's rel_imm cross-protection coefficient.

        Mirrors v2's architecture: a single per-agent sterilizing Bernoulli at
        ``sterilizing_p`` (default 0.95, matching v2's hardcoded imm_init=0.95),
        then per-genotype scaling by ``rel_imm[g]`` from products_vx.csv. v2
        encodes the per-genotype effect via the cross-immunity matrix coefficient
        M[g, vx_source] = rel_imm[g]; v3 encodes it directly into vax_imm[g] using
        ``rel_imm[g]`` as a multiplicative scalar on the per-agent peak.

        For each vaccinated agent:
          - Sterilizing fate is drawn once at p=sterilizing_p (NOT per-genotype).
          - For each genotype g:
              - Sterilizing agents:       vax_imm[g] = rel_imm[g]
              - Non-sterilizing (leaky):  vax_imm[g] = rel_imm[g] * sterilizing_p
        Max-of-existing prevents vaccine from downgrading prior immunity.
        """
        if len(uids) == 0:
            return
        # Single sterilizing draw per agent (NOT per genotype) — matches v2
        self._sterilizing_dist.set(p=float(self.pars.sterilizing_p))
        sterilizing_uids = self._sterilizing_dist.filter(uids)
        is_sterilizing = np.isin(uids, sterilizing_uids)
        for genotype, rel_imm_g in self.rel_imm.items():
            hpv_mod = self._find_genotype_module(genotype)
            if hpv_mod is None:
                continue
            # Sterilizing agents get rel_imm[g]; leaky get rel_imm[g] * sterilizing_p
            peak = np.where(
                is_sterilizing,
                float(rel_imm_g),
                float(rel_imm_g) * float(self.pars.sterilizing_p),
            )
            hpv_mod.vax_imm[uids] = np.maximum(hpv_mod.vax_imm[uids], peak)

    def _find_genotype_module(self, genotype):
        """Return the HPV module in self.sim matching this genotype, or None.

        Backward-compatible instance method — delegates to the module-level
        helper of the same name. Kept on hpv.vx for callers that hold a vx
        product instance rather than a sim reference.
        """
        return _find_genotype_module(self.sim, genotype)


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
        resolved_df, resolved_hierarchy = _resolve_dx_pars(name, df, hierarchy)
        # ss.Dx.__init__ accesses df.disease.unique() which HPV CSVs don't have.
        # Add a temporary stub column so the base init succeeds, then overwrite
        # self.diseases with our HPV-specific genotype attribute.
        df_for_base = resolved_df.copy()
        if 'disease' not in df_for_base.columns:
            df_for_base['disease'] = '_hpv_stub'
        super().__init__(df=df_for_base, hierarchy=resolved_hierarchy, **kwargs)
        # ss.Dx populates self.diseases from the stub; we don't use that attribute
        # — hpv.dx routes through _iter_hpv_modules instead. Clear it to avoid
        # confusing downstream introspection.
        self.diseases = None
        # Store the original (no-stub) df for our own administer logic
        self.df = resolved_df
        self.name = name
        # The base sets self.diseases from df['disease'] but HPV CSV uses
        # 'genotype' instead. Replace with our own attrs.
        self._genotypes_in_df = list(resolved_df['genotype'].unique())
        self._all_genotype = (len(self._genotypes_in_df) == 1
                              and self._genotypes_in_df[0] == 'all')

    def administer(self, uids, return_format='dict'):
        if len(uids) == 0:
            if return_format == 'dict':
                return {k: ss.uids() for k in self.hierarchy}
            return np.array([], dtype=int)

        # Normalize input to ss.uids so downstream .intersect / .union ops work
        # regardless of whether the caller passed an ss.uids or a plain ndarray.
        uids = ss.uids(uids)
        # uid-keyed Series so we can update with .loc[these_uids] directly,
        # mirroring ss.Dx.administer. Hierarchy-min semantics: most-severe
        # result (lowest hierarchy index) wins on multi-genotype-positive agents.
        results = pd.Series(self.default_value, index=uids)

        for state in self.health_states:
            if self._all_genotype:
                these = _state_collapse_across_genotypes(state, uids, self.sim)
                if len(these) == 0:
                    continue
                df_filter = (self.df.state == state) & (self.df.genotype == 'all')
                self._draw_and_min_into(results, these, df_filter)
            else:
                for module in _iter_hpv_modules(self.sim):
                    if module.genotype not in self._genotypes_in_df:
                        continue
                    these = _state_uids_for_module(module, state, uids)
                    if len(these) == 0:
                        continue
                    df_filter = (
                        (self.df.state == state)
                        & (self.df.genotype == module.genotype)
                    )
                    self._draw_and_min_into(results, these, df_filter)

        if return_format == 'dict':
            return {k: ss.uids(results.index[results == i].to_numpy())
                    for i, k in enumerate(self.hierarchy)}
        return results.to_numpy(dtype=int)

    def _draw_and_min_into(self, results, these, df_filter):
        probs = [
            float(self.df[df_filter & (self.df.result == r)]['probability'].values[0])
            for r in self.hierarchy
        ]
        self.result_dist.pars['p'] = probs
        draw = self.result_dist.rvs(these)
        results.loc[these] = np.minimum(draw, results.loc[these])


class tx(ss.Tx):
    """HPV treatment product — per-genotype state-flip with efficacy draw.

    On successful treatment of state in {precin, cin, cancerous} on
    genotype g:
        module.<state>[uids] = False
        module.cin[uids]      = False
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
        # ss.Tx.__init__ accesses df.disease.unique() — HPV CSVs use 'genotype'
        # instead. Add a temporary stub column so the base init succeeds, then
        # restore the original df and clear the base-set self.diseases attribute.
        df_for_base = df.copy()
        if 'disease' not in df_for_base.columns:
            df_for_base['disease'] = '_hpv_stub'
        super().__init__(df=df_for_base, **kwargs)
        # ss.Tx populates self.diseases from the stub; hpv.tx routes through
        # _iter_hpv_modules instead — clear it to avoid confusing introspection.
        self.diseases = None
        # Restore the original (no-stub) df for our own administer logic.
        self.df = df
        self.name = name

    def administer(self, uids, return_format='dict'):
        if len(uids) == 0:
            empty = ss.uids()
            return {'successful': empty, 'unsuccessful': empty} if return_format == 'dict' else empty

        successful_uids_list = []
        for state in self.health_states:
            for module in _iter_hpv_modules(self.sim):
                # Match rows by state and genotype ('all' matches every module)
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
            successful = ss.uids(np.unique(np.concatenate(successful_uids_list)))
        else:
            successful = ss.uids()
        unsuccessful = ss.uids(np.setdiff1d(uids, successful))

        if return_format == 'dict':
            return {'successful': successful, 'unsuccessful': unsuccessful}
        return successful


class txvx(ss.Vx):
    """HPV therapeutic vaccine product (parallel structure to hpv.vx).

    Two modes:
    - Initial dose (default): per-agent sterilizing draw at sterilizing_p,
      then per-genotype scaling by rel_imm[g] writing into txvx_imm.
    - Booster (imm_boost not None): multiplies existing txvx_imm in place.

    v2 reference: hpvsim/_v2_legacy/interventions.py:1416-1466 + default_tx
    wiring for txvx1/txvx2.
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
        # Mirror hpv.tx / hpv.dx: set the module-level name to the product
        # name so that sim.people.add_module() doesn't collide with an
        # intervention or another product that also uses class name 'txvx'.
        if name is not None:
            self.name = name
        if imm_boost is None:
            # First-dose path requires resolved rel_imm
            self.rel_imm = _resolve_txvx_pars(name, rel_imm)
        else:
            # Booster path: rel_imm is optional (in-place multiply doesn't need it)
            if name is not None or rel_imm is not None:
                self.rel_imm = _resolve_txvx_pars(name, rel_imm)
            else:
                self.rel_imm = {}
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
        is_sterilizing = np.isin(uids, sterilizing_uids)
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
        dt = self.sim.t.dt
        for module in _iter_hpv_modules(self.sim):
            cancer_uids = module.cancerous.uids.intersect(uids)
            if len(cancer_uids) == 0:
                continue
            # Draw per-module on the cancer subset. Avoids the alignment trap
            # of pre-drawing on `uids` and then trying to index by mask (which
            # gives uids-order durations) while writing by cancer_uids
            # (which is sorted order — `intersect` does not preserve uids-order).
            new_dur = self._dur_dist.rvs(cancer_uids)
            module.ti_dead_cancer[cancer_uids] += np.ceil(new_dur / dt)
        return uids

