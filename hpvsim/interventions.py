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

__all__ = []


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
    try:
        s_int = int(sex)
    except (TypeError, ValueError):
        raise ValueError(f"sex must be 'f', 'm', 0, 1, or a list thereof, got {sex!r}")
    if s_int not in (0, 1):
        raise ValueError(f"sex int must be 0 or 1, got {sex!r}")
    return {s_int}