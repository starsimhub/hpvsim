"""HIV–HPV co-infection: transmission-based HIV plus the CD4-stratified
HIV→HPV effects ported (by value) from v2's HIVsim.

Three components:
  - ``HIV`` (``sti.HIV`` subclass): continuous-CD4 transmission-based HIV,
    re-targeted onto ``hpv.SexualNetwork`` and seeded from a Rwanda init-prev curve.
  - ``hpv_hiv_connector`` (``ss.Connector``): bins CD4 into discrete strata and
    applies v2's rel_sus / rel_sev / rel_imm effects to every HPV module.
  - ``HIVStratifiedResults`` (``ss.Analyzer``): HPV/cancer outcomes by HIV status.
"""

import starsim as ss
import stisim as sti

from . import misc
from .network import SexualNetwork

__all__ = ['HIV', 'hpv_hiv_connector', 'HIVStratifiedResults']


class HIV(sti.HIV):
    """Transmission-based HIV for hpvsim.

    Thin subclass of ``sti.HIV``: inherits continuous CD4, ART reconstitution,
    and CD4-based mortality unchanged. Adds (1) HPVsim-friendly directional
    beta targeting ``hpv.SexualNetwork`` (whose p1=female, p2=male, unlike
    STIsim's ``structuredsexual``), and (2) a Rwanda init-prevalence curve.

    Note: ``beta_m2f`` / ``rel_beta_f2m`` are taken as constructor arguments and
    applied directly to ``pars.beta`` in ``init_pre``. STIsim's same-named
    ``pars.beta_m2f`` / ``pars.rel_beta_f2m`` are NOT used here — its
    ``validate_beta`` only applies them to a network named ``'structuredsexual'``,
    which hpvsim does not use. Set the transmission rate via the constructor args,
    not via ``pars``.
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
        # So beta[net.name][0] = f2m (smaller), beta[net.name][1] = m2f = beta_m2f (larger).
        nets = [n for n in sim.networks.values() if isinstance(n, SexualNetwork)]
        if not nets:
            misc.warn('hpv.HIV: no SexualNetwork found; HIV will not transmit.')
            return
        beta = {}
        for net in nets:
            beta[net.name] = [self._beta_m2f * self._rel_beta_f2m, self._beta_m2f]
        self.pars.beta = beta


class hpv_hiv_connector(ss.Connector):
    """Placeholder — implemented in M08 Task 3."""
    pass


class HIVStratifiedResults(ss.Analyzer):
    """Placeholder — implemented in M08 Task 8."""
    pass
