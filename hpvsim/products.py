"""HPV-specific Starsim products.

Currently contains the prophylactic vaccine product class ``hpv.vx``. M06
will add ``hpv.dx`` (diagnostics) and ``hpv.tx`` (treatments). M06 will also
add the therapeutic vaccine product variant.
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import starsim as ss

__all__ = ['vx']

_PRODUCT_CSV = Path(__file__).parent / 'data' / 'products_vx.csv'


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
        self.rel_imm = _resolve_vx_pars(name, rel_imm)
        # CRN-safe Bernoulli; p is overwritten per-genotype in administer().
        # Initialized via the standard intervention -> product.init_pre ->
        # sim.init_dists() chain, matching the ss.Dx / ss.Tx pattern.
        self._sterilizing_dist = ss.bernoulli(p=0.0)

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
        for genotype, rel_imm_g in self.rel_imm.items():
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