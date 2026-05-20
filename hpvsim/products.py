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