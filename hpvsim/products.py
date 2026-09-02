"""HPV-specific Starsim products.

Contains:
  - hpv.vx: prophylactic vaccine product
  - hpv.dx: per-genotype multinomial diagnostic classifier
  - hpv.tx: per-genotype state-flip treatment product
  - hpv.txvx: therapeutic vaccine product
  - hpv.radiation: cancer treatment product
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import starsim as ss

from hpvsim.utils import find_genotype_module, iter_hpv_modules

from . import misc

__all__ = ['vx', 'dx', 'tx', 'txvx', 'radiation']

_PRODUCT_CSV = Path(__file__).parent / 'data' / 'products_vx.csv'


def _hiv_rel_imm_factor(sim):
    """Return the HIV module's full ``hiv_rel_imm`` FloatArr, or None.

    Returns the HIV disease's ``hiv_rel_imm`` FloatArr (caller indexes by the
    relevant uid subset; value is 1.0 for HIV- agents) when an ``hpv.HIV``
    disease is registered in the sim, else ``None`` so callers cleanly no-op
    without HIV. Returns None when stisim isn't installed.
    """
    if not getattr(sim, 'diseases', None):
        return None
    hivm = misc.hiv_module(sim)
    return None if hivm is None else hivm.hiv_rel_imm


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

    Exactly one of name or df must be provided. Default hierarchies are
    defined per product below.
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

    In practice almost all callers use ``name`` — the named products in the CSV
    cover the real-world vaccines. The explicit ``rel_imm`` dict is an
    escape hatch for ad-hoc / experimental products (e.g. a hypothetical
    vaccine, or sensitivity sweeps over cross-protection coefficients) and is
    rarely needed.

    The vaccine model has two parameters:

    - ``sterilizing_p`` (default 0.95): per-agent Bernoulli probability of
      sterilizing immunity, drawn ONCE per agent (not per genotype).
    - ``rel_imm[g]`` from the CSV: per-genotype cross-protection coefficient.
      Sterilizing agents receive ``vax_imm[g] = rel_imm[g]``; leaky agents
      receive ``vax_imm[g] = rel_imm[g] * sterilizing_p``.

    The effective per-genotype protection is approximately
    ``0.9975 * rel_imm[g]``. Existing ``vax_imm`` is never downgraded
    (max-of-existing semantics).

    Vaccine immunity is written to ``vax_imm`` (NOT ``nab_imm``). The
    ``CrossImmunity`` connector applies ``vax_imm`` directly per-genotype
    without flowing it through the cross-immunity matrix, so the CSV's
    per-genotype ``rel_imm`` values are the complete vaccine cross-protection
    profile: vaccine immunity does not amplify into cross-protection against
    non-target genotypes.
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

        A single per-agent sterilizing Bernoulli at ``sterilizing_p``
        (default 0.95), then per-genotype scaling by ``rel_imm[g]`` from
        products_vx.csv. ``rel_imm[g]`` is applied directly to ``vax_imm[g]``
        as a multiplicative scalar on the per-agent peak.

        For each vaccinated agent:
          - Sterilizing fate is drawn once at p=sterilizing_p (NOT per-genotype).
          - For each genotype g:
              - Sterilizing agents:       vax_imm[g] = rel_imm[g]
              - Non-sterilizing (leaky):  vax_imm[g] = rel_imm[g] * sterilizing_p
        Max-of-existing prevents vaccine from downgrading prior immunity.
        """
        if len(uids) == 0:
            return
        # HIV co-infection reduces vaccine take (gated no-op without HIV).
        rel_imm_hiv = _hiv_rel_imm_factor(self.sim)
        # Single sterilizing draw per agent, not per genotype.
        self._sterilizing_dist.set(p=float(self.pars.sterilizing_p))
        sterilizing_uids, leaky_uids = self._sterilizing_dist.filter(uids, both=True)
        for genotype, rel_imm_g in self.rel_imm.items():
            hpv_mod = find_genotype_module(self.sim, genotype)
            if hpv_mod is None:
                continue
            # Sterilizing agents get rel_imm[g]; leaky get rel_imm[g] * sterilizing_p
            ster_peak = float(rel_imm_g)
            leaky_peak = ster_peak * float(self.pars.sterilizing_p)
            if rel_imm_hiv is None:
                ster_vals = ster_peak
                leaky_vals = leaky_peak
            else:
                ster_vals = ster_peak * rel_imm_hiv[sterilizing_uids]
                leaky_vals = leaky_peak * rel_imm_hiv[leaky_uids]
            hpv_mod.vax_imm[sterilizing_uids] = np.maximum(hpv_mod.vax_imm[sterilizing_uids], ster_vals)
            hpv_mod.vax_imm[leaky_uids] = np.maximum(hpv_mod.vax_imm[leaky_uids], leaky_vals)


def _state_uids_for_module(module, state, uids):
    """Return uids that are in `state` on `module` and also in `uids`."""
    arr = getattr(module, state, None)
    if arr is None:
        return ss.uids()
    return arr.uids.intersect(uids)


def _state_uids_across_genotypes(state, uids, sim):
    """Collapse a per-genotype state to a single uids set across HPV modules.

    - state='susceptible': agent must be susceptible to ALL genotypes.
    - any other state: agent must be in that state for ANY genotype.
    """
    modules = list(iter_hpv_modules(sim))
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
    Hierarchy-min semantics: when an agent is positive across multiple
    genotypes, the lowest-index (most severe) result wins.
    """

    def __init__(self, name=None, df=None, hierarchy=None, **kwargs):
        resolved_df, resolved_hierarchy = _resolve_dx_pars(name, df, hierarchy)
        # ss.Dx.__init__ needs df.disease, which HPV CSVs lack; stub it out.
        df_for_base = resolved_df.copy()
        if 'disease' not in df_for_base.columns:
            df_for_base['disease'] = '_hpv_stub'
        super().__init__(df=df_for_base, hierarchy=resolved_hierarchy, **kwargs)
        # Unused: hpv.dx routes through iter_hpv_modules, not self.diseases.
        self.diseases = None
        # Store the original (no-stub) df for our own administer logic
        self.df = resolved_df
        # Module name = product name; a df-built product keeps the class default
        # ('dx'), since a None name breaks people.add_module() at sim.init().
        if name is not None:
            self.name = name
        self._genotypes_in_df = list(resolved_df['genotype'].unique())
        self._all_genotype = (len(self._genotypes_in_df) == 1
                              and self._genotypes_in_df[0] == 'all')

    def administer(self, uids, return_format='dict'):
        if len(uids) == 0:
            if return_format == 'dict':
                return {k: ss.uids() for k in self.hierarchy}
            return np.array([], dtype=int)

        # Normalize so .intersect / .union work on a plain-ndarray input.
        uids = ss.uids(uids)
        # uid-keyed for .loc updates; hierarchy-min means lowest index wins.
        results = pd.Series(self.default_value, index=uids)

        for state in self.health_states:
            if self._all_genotype:
                these = _state_uids_across_genotypes(state, uids, self.sim)
                if len(these) == 0:
                    continue
                df_filter = (self.df.state == state) & (self.df.genotype == 'all')
                self._draw_and_min_into(results, these, df_filter)
            else:
                for module in iter_hpv_modules(self.sim):
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
        module.ti_clearance[uids] = module.ti + 1   # cleared next module step
        module.to_latent[uids]    = False           # clears to susceptible
    """

    def __init__(self, name=None, df=None, **kwargs):
        df = _resolve_tx_pars(name, df)
        # ss.Tx.__init__ needs df.disease, which HPV CSVs lack; stub it out.
        df_for_base = df.copy()
        if 'disease' not in df_for_base.columns:
            df_for_base['disease'] = '_hpv_stub'
        super().__init__(df=df_for_base, **kwargs)
        # Unused: hpv.tx routes through iter_hpv_modules, not self.diseases.
        self.diseases = None
        # Restore the original (no-stub) df for our own administer logic.
        self.df = df
        # Module name = product name; a df-built product keeps the class default
        # ('tx'), since a None name breaks people.add_module() at sim.init().
        if name is not None:
            self.name = name

    def administer(self, uids, return_format='dict'):
        if len(uids) == 0:
            empty = ss.uids()
            return {'successful': empty, 'unsuccessful': empty} if return_format == 'dict' else empty

        successful_uids_list = []
        for state in self.health_states:
            for module in iter_hpv_modules(self.sim):
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
                # State cleanup: clear all dysplasia state on the treated genotype
                module.cin[eff] = False
                module.precin[eff] = False
                module.cancerous[eff] = False
                module.ti_cin[eff] = np.nan
                module.ti_cancerous[eff] = np.nan
                # ti_clearance is read against the HPV module's own ti, not ours.
                module.ti_clearance[eff] = module.ti + 1
                # Cancel any pending latency roll: treatment-induced clearance
                # sends the agent to susceptible, not into latency. No-op at the
                # default hpv_control_prob=0, where to_latent is never set.
                module.to_latent[eff] = False

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
        # Avoids an add_module() collision with an intervention also named 'txvx'.
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
            # Booster: multiplicative in place. No HIV scaling — the first dose
            # was already scaled, so applying it again would double-count.
            for module in iter_hpv_modules(self.sim):
                module.txvx_imm[uids] *= float(self.pars.imm_boost)
            return
        # First dose: per-agent sterilizing draw (rvs returns bools in uids-order).
        self._sterilizing_dist.set(p=float(self.pars.sterilizing_p))
        is_sterilizing = self._sterilizing_dist.rvs(uids)
        # HIV reduces vaccine take (gated no-op); uids-order matches peak below.
        rel_imm_hiv = _hiv_rel_imm_factor(self.sim)
        hiv_scale = 1.0 if rel_imm_hiv is None else rel_imm_hiv[uids]
        for genotype, rel_imm_g in self.rel_imm.items():
            module = find_genotype_module(self.sim, genotype)
            if module is None:
                continue  # inactive-genotype tolerance
            peak = np.where(
                is_sterilizing,
                float(rel_imm_g),
                float(rel_imm_g) * float(self.pars.sterilizing_p),
            ) * hiv_scale
            module.txvx_imm[uids] = np.maximum(module.txvx_imm[uids], peak)


class radiation(ss.Product):
    """HPV cancer-treatment product — extends ti_dead_cancer per cancerous module.

    Default duration: normal(mean=18 months, sd=2 months), converted to
    years at construction.
    """

    def __init__(self, dur=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            dur=dur or dict(dist='normal', par1=18 / 12, par2=2 / 12),
        )
        # Placeholder; loc/scale are re-pointed on each administer call so
        # post-init mutation of self.pars.dur is honored.
        self._dur_dist = ss.normal(loc=0.0, scale=1.0)

    def administer(self, uids):
        if len(uids) == 0:
            return ss.uids()
        self._dur_dist.set(loc=float(self.pars.dur['par1']),
                           scale=float(self.pars.dur['par2']))
        # dt_year, not t.dt: arithmetic on the freq object drops the time unit.
        dt_year = self.sim.t.dt_year
        for module in iter_hpv_modules(self.sim):
            cancer_uids = module.cancerous.uids.intersect(uids)
            if len(cancer_uids) == 0:
                continue
            # Draw on cancer_uids: intersect returns sorted order, not uids-order.
            new_dur = self._dur_dist.rvs(cancer_uids)
            module.ti_dead_cancer[cancer_uids] += np.ceil(new_dur / dt_year)
        return uids

