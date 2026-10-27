"""HPV-specific Starsim products, delivered by ``ss.vaccinate()``, ``ss.screen()``, and ``ss.treat()``.

Contains:
  - hpv.vx: prophylactic vaccine product
  - hpv.dx: per-genotype multinomial diagnostic classifier
  - hpv.tx: per-genotype state-flip treatment product
  - hpv.txvx: therapeutic vaccine product
  - hpv.radiation: cancer treatment product

Each is a product written as code (an ``ss.Product`` with an ``administer()``), since the
genotypes are diseases written as code, whose states v4's ``ss.Diagnostic`` and ``ss.Treatment``
cannot key their results by (see ``v4_port_oct26.md``, HP70). Each returns its outcomes from
``administer()``, so they are results and events of the delivery (e.g. ``screening.positive``).
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

from hpvsim.utils import find_genotype_module, iter_hpv_modules

from . import misc

__all__ = ['vx', 'dx', 'tx', 'txvx', 'radiation']

_PRODUCT_CSV = Path(__file__).parent / 'data' / 'products_vx.csv'


def _hiv_rel_imm_factor(sim):
    """Return HIV's full ``hiv_rel_imm`` FloatArr, or None.

    Returns the ``hiv_rel_imm`` FloatArr of ``hpv.HIVEffects`` (caller indexes
    by the relevant uid subset; value is 1.0 for HIV- agents) when the sim has
    one, else ``None`` so callers cleanly no-op without HIV. Returns None when
    stisim isn't installed.
    """
    if not getattr(sim, 'connectors', None):
        return None
    effects = misc.hiv_effects(sim)
    return None if effects is None else effects.hiv_rel_imm


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

    At least one of ``name`` or ``df`` must be provided. When both are
    supplied, ``df`` wins for the row table and ``name`` is kept as the
    module name — this is how multiple df-built dx products coexist in a
    single sim without colliding on the class-default name.
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
    if name is None and df is None:
        raise ValueError('hpv.dx requires at least one of `name` or `df`.')
    if df is not None:
        if hierarchy is None:
            hierarchy = _DEFAULT_DX_HIERARCHY.get(name, list(df['result'].unique()))
        return df, hierarchy
    products = _load_dx_products()
    if name not in products:
        valid = ', '.join(products.keys())
        raise ValueError(f'Unknown dx product name {name!r}. Valid names: {valid}.')
    if hierarchy is None:
        hierarchy = _DEFAULT_DX_HIERARCHY.get(name, list(products[name]['result'].unique()))
    return products[name], hierarchy


def _resolve_tx_pars(name, df):
    """Resolve (name, df) for hpv.tx construction.

    At least one of ``name`` or ``df`` must be provided. When both are
    supplied, ``df`` wins for the row table and ``name`` is kept as the
    module name — this is how multiple df-built tx products coexist in a
    single sim without colliding on the class-default name.
    """
    if name is None and df is None:
        raise ValueError('hpv.tx requires at least one of `name` or `df`.')
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


def _default_name(product, name):
    """Name a product after the product it looks up (e.g. 'via'), as its default name: if a delivery has the same name
    (``ablation = ss.treat(hpv.tx('ablation'))``), the sim renames the product (``ablation_product``); ``module_name=`` names it for good"""
    if name is not None and product._default_name:
        product.name = product.label = name
    return


class vx(ss.Product):
    """HPV multi-genotype prophylactic vaccine.

    Constructed with EITHER ``name`` (looks up the per-genotype rel_imm from
    ``hpvsim/data/products_vx.csv``) OR ``rel_imm`` (explicit per-genotype
    dict). Default product names: ``'bivalent'``, ``'quadrivalent'``,
    ``'nonavalent'``.

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
    profile.

    Examples::

        bivalent = hpv.vx('bivalent')
        routine = ss.vaccinate(bivalent, where=ss.ppl.female, at_age=9, coverage=0.9, start=2020)
    """

    def __init__(self, name=None, rel_imm=None, sterilizing_p=0.95, module_name=None, **kwargs):
        super().__init__(name=module_name, **kwargs)
        _default_name(self, name)
        self.define_pars(sterilizing_p=sterilizing_p)
        self.rel_imm = _resolve_vx_pars(name, rel_imm)
        self.sterilizing = ss.bernoulli(p=self.pars.sterilizing_p) # One draw per agent, not per genotype
        return

    def administer(self, uids):
        """Per-agent all-or-nothing sterilizing draw, scaled per genotype by ``rel_imm``; HIV reduces it"""
        rel_imm_hiv = _hiv_rel_imm_factor(self.sim)
        self.sterilizing.set(p=float(self.pars.sterilizing_p))
        sterilizing_uids, leaky_uids = self.sterilizing.filter(uids, both=True)
        for genotype, rel_imm_g in self.rel_imm.items():
            hpv_mod = find_genotype_module(self.sim, genotype)
            if hpv_mod is None:
                continue
            for these, peak in [(sterilizing_uids, rel_imm_g), (leaky_uids, rel_imm_g*self.pars.sterilizing_p)]:
                vals = peak if rel_imm_hiv is None else peak*rel_imm_hiv[these]
                hpv_mod.vax_imm[these] = np.maximum(hpv_mod.vax_imm[these], vals)
        return


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
    matched = ss.uids()
    for m in modules:
        matched = matched.union(_state_uids_for_module(m, state, uids))
    return matched


class dx(ss.Product):
    """HPV diagnostic product with per-genotype state classification.

    Each person tested gets one result, from the probability of each result
    given the state she is in (``hpvsim/data/products_dx.csv``). Per-genotype
    rows are classified one genotype at a time; rows with genotype='all' are
    collapsed across all HPV modules (susceptible iff susceptible to all;
    any other state iff in it for any genotype). When a person qualifies for
    several rows, the most severe result (the first in ``hierarchy``) wins.

    The results are the outcomes of its delivery, e.g. ``screening.positive``,
    or ``colpo.hsil`` for colposcopy; a result whose probability is 0 in every
    state (``inadequate``, in the shipped products) is left out.

    Examples::

        via = ss.screen(hpv.dx('via'), where=women, coverage=0.6, interval=ss.years(1))
        ablation = ss.treat(hpv.tx('ablation'), who=via.positive & hpv.no_cancer)
    """

    def __init__(self, name=None, df=None, hierarchy=None, module_name=None, **kwargs):
        df, hierarchy = _resolve_dx_pars(name, df, hierarchy)
        super().__init__(name=module_name, **kwargs)
        _default_name(self, name)
        used = set(df.loc[df.probability > 0, 'result'])
        self.hierarchy = [r for r in hierarchy[:-1] if r in used] + hierarchy[-1:] # Most severe first; the last is everyone else
        self.df = df
        self.health_states = list(df.state.unique())
        self.result_dist = ss.choice(a=len(self.hierarchy))
        genotypes = list(df.genotype.unique())
        self._all_genotype = genotypes == ['all']
        self._genotypes = genotypes
        return

    def outcome_names(self):
        return list(self.__dict__.get('hierarchy', []))

    def administer(self, uids):
        results = pd.Series(len(self.hierarchy) - 1, index=uids) # The least severe, unless a row says otherwise
        for state in self.health_states:
            if self._all_genotype:
                rows = [(_state_uids_across_genotypes(state, uids, self.sim), 'all')]
            else:
                rows = [(_state_uids_for_module(m, state, uids), m.genotype) for m in iter_hpv_modules(self.sim) if m.genotype in self._genotypes]
            for these, genotype in rows:
                if len(these):
                    sub = self.df[(self.df.state == state) & (self.df.genotype == genotype)]
                    probs = np.array([sub.probability[sub.result == r].sum() for r in self.hierarchy])
                    self.result_dist.set(p=probs/probs.sum())
                    results.loc[these] = np.minimum(self.result_dist.rvs(these), results.loc[these])
        return {k: ss.uids(results.index[results == i].to_numpy()) for i, k in enumerate(self.hierarchy)}


class tx(ss.Product):
    """HPV treatment product: per-genotype state-flip with an efficacy draw.

    For each genotype a person has in a state the product treats
    (``hpvsim/data/products_tx.csv``), one efficacy draw; on success the
    genotype's precin/CIN/cancer is cleared (she is susceptible to it again,
    from its next step). Its outcomes, as for ``ss.Treatment``, are ``cured``
    (cleared of at least one genotype), ``failed`` (in a treated state, and
    cleared of none), and ``unneeded`` (in no state it treats).

    Examples::

        ablation = ss.treat(hpv.tx('ablation'), who=screening.positive & hpv.no_cancer, coverage=0.9)
    """

    def __init__(self, name=None, df=None, module_name=None, **kwargs):
        df = _resolve_tx_pars(name, df)
        super().__init__(name=module_name, **kwargs)
        _default_name(self, name)
        self.df = df
        self.health_states = list(df.state.unique())
        self.efficacy_dist = ss.bernoulli(p=0)
        return

    def outcome_names(self):
        return ['cured', 'failed', 'unneeded']

    def administer(self, uids):
        cured = []
        treated = []
        for state in self.health_states:
            for module in iter_hpv_modules(self.sim):
                rows = self.df[(self.df.state == state) & ((self.df.genotype == module.genotype) | (self.df.genotype == 'all'))]
                these = _state_uids_for_module(module, state, uids)
                if len(rows) == 0 or len(these) == 0:
                    continue
                treated.append(these)
                self.efficacy_dist.set(p=float(rows['efficacy'].values[0]))
                eff = self.efficacy_dist.filter(these)
                if len(eff):
                    cured.append(eff)
                    module.clear_infection(eff)
        cured = ss.uids(np.unique(np.concatenate(cured))) if cured else ss.uids()
        treated = ss.uids(np.unique(np.concatenate(treated))) if treated else ss.uids()
        return dict(cured=cured, failed=treated.remove(cured), unneeded=ss.uids(uids).remove(treated))


class txvx(tx):
    """HPV therapeutic vaccine: clears infection/lesions, then confers immunity.

    A treatment product, not a prophylactic. It flips per-genotype disease
    state through the same state x genotype efficacy table as ``hpv.tx``
    (the ``txvx1``/``txvx2`` rows of ``products_tx.csv``), and additionally
    confers severity immunity via ``imm_init``/``imm_boost``, scaled per
    target genotype by ``rel_imm`` (``products_txvx.csv``).

    Named products reproduce the v2 defaults: ``txvx1`` is a first dose
    conferring ``beta_mean(0.35, 0.025)`` immunity, ``txvx2`` a booster
    multiplying existing immunity by 1.5.
    """

    #: v2 default immunity per named product: (imm_init, imm_boost).
    _DEFAULTS = {
        'txvx1': (ss.beta_mean(mean=0.35, var=0.025), None),
        'txvx2': (None, 1.5),
    }

    def __init__(self, name=None, df=None, rel_imm=None, imm_init=None,
                 imm_boost=None, module_name=None, **kwargs):
        if name is None and df is None:
            name = 'txvx1'
        default_name = name if name in self._DEFAULTS else 'txvx1'
        if imm_init is None and imm_boost is None:
            imm_init, imm_boost = self._DEFAULTS[default_name]
        if imm_init is not None and imm_boost is not None:
            raise ValueError('hpv.txvx takes at most one of `imm_init` or `imm_boost`.')
        super().__init__(name=name, df=df, module_name=module_name, **kwargs)
        self.imm_init = sc.dcp(imm_init)
        self.imm_boost = imm_boost
        # Per-target-genotype scaling of the conferred immunity. Defaults to the
        # named product's row in products_txvx.csv; 1.0 for anything unlisted.
        if rel_imm is None:
            rel_imm = _load_txvx_products().get(default_name)
        self.rel_imm = dict(rel_imm) if rel_imm is not None else {}

    def administer(self, uids):
        """Clear disease state as ``hpv.tx`` does, then confer immunity on everyone dosed (as v2 did)"""
        out = super().administer(uids)
        base = self.imm_init.rvs(uids) if self.imm_init is not None else None # One draw per agent, scaled per target genotype
        for module in iter_hpv_modules(self.sim):
            scale = float(self.rel_imm.get(module.genotype, 1.0))
            if base is not None:
                module.txvx_sev_imm[uids] = base*scale
            else:
                module.txvx_sev_imm[uids] = module.txvx_sev_imm[uids]*self.imm_boost
        return out


class radiation(ss.Product):
    """HPV cancer-treatment product: extends ``ti_dead_cancer`` per cancerous genotype.

    Default duration: normal(mean=18 months, sd=2 months).

    Examples::

        radiation = ss.treat(hpv.radiation(), who=screening.positive & hpv.has_cancer)
    """

    def __init__(self, dur=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            dur=dur or ss.normal(ss.years(1.5), ss.years(2/12)),
        )
        return

    def administer(self, uids):
        for module in iter_hpv_modules(self.sim):
            cancer_uids = module.cancerous.uids.intersect(uids)
            if len(cancer_uids):
                module.ti_dead_cancer[cancer_uids] += np.ceil(self.pars.dur.rvs(cancer_uids)) # In the module's steps
        return
