"""HPV-specific Starsim products.

Currently contains the prophylactic vaccine product class ``hpv.vx``. M06
will add ``hpv.dx`` (diagnostics) and ``hpv.tx`` (treatments). M06 will also
add the therapeutic vaccine product variant.
"""
import functools
from pathlib import Path

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
        self._sterilizing_dist = ss.bernoulli(p=0.0)

    def administer(self, people, uids):
        """Apply the vaccine — see class docstring for the model."""
        raise NotImplementedError('hpv.vx.administer() is not yet implemented.')