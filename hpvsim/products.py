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

from hpvsim.utils import find_genotype_module

__all__ = ['vx']

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
        # Single sterilizing draw per agent (NOT per genotype).
        self._sterilizing_dist.set(p=float(self.pars.sterilizing_p))
        sterilizing_uids, leaky_uids = self._sterilizing_dist.filter(uids, both=True)
        for genotype, rel_imm_g in self.rel_imm.items():
            hpv_mod = find_genotype_module(self.sim, genotype)
            if hpv_mod is None:
                continue
            # Sterilizing agents get rel_imm[g]; leaky get rel_imm[g] * sterilizing_p
            ster_peak = float(rel_imm_g)
            leaky_peak = ster_peak * float(self.pars.sterilizing_p)
            hpv_mod.vax_imm[sterilizing_uids] = np.maximum(hpv_mod.vax_imm[sterilizing_uids], ster_peak)
            hpv_mod.vax_imm[leaky_uids] = np.maximum(hpv_mod.vax_imm[leaky_uids], leaky_peak)

    def _find_genotype_module(self, genotype):
        """Return the HPV module in self.sim matching this genotype, or None.

        Backward-compatible instance method — delegates to the module-level
        helper of the same name. Kept on hpv.vx for callers that hold a vx
        product instance rather than a sim reference.
        """
        return _find_genotype_module(self.sim, genotype)
