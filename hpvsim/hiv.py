"""HIV–HPV co-infection: transmission-based HIV plus the CD4-stratified
HIV→HPV effects ported (by value) from v2's HIVsim.

Three components:
  - ``HIV`` (``sti.HIV`` subclass): continuous-CD4 transmission-based HIV,
    re-targeted onto ``hpv.SexualNetwork`` and seeded from a Rwanda init-prev curve.
  - ``hpv_hiv_connector`` (``ss.Connector``): bins CD4 into discrete strata and
    applies v2's rel_sus / rel_sev / rel_imm effects to every HPV module.
  - ``HIVStratifiedResults`` (``ss.Analyzer``): HPV/cancer outcomes by HIV status.
"""

import numpy as np
import starsim as ss
import stisim as sti

from . import misc
from .hpv import HPV
from .network import SexualNetwork

__all__ = ['HIV', 'hpv_hiv_connector', 'HIVStratifiedResults']


# CD4-stratified HIV→HPV effect multipliers. Copied by value from v2's
# HIVsim defaults (hpvsim/_v2_legacy/hiv.py:29-44) per the no-quarantine-import
# rule. Strata: 'lt200' = CD4 < 200; 'gt200' = CD4 >= 200 (v2's 200-500 band,
# extended to all CD4 >= 200 for HIV+ agents). Agents with CD4 > 500 (newly
# infected in v2's model) had no multiplier in v2; here they receive the gt200 factor.
_HIV_EFFECTS = {
    'rel_sus': {'lt200': 2.2, 'gt200': 2.2},   # increased HPV acquisition
    'rel_sev': {'lt200': 1.5, 'gt200': 1.2},   # faster/worse CIN->cancer progression
    'rel_imm': {'lt200': 0.36, 'gt200': 0.76}, # reduced post-infection/vaccine immunity
}
_CD4_THRESHOLD = 200.0


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
    """Apply v2's CD4-stratified HIV→HPV effects to every HPV module.

    Each step: bin HIV+ agents' CD4 into discrete strata, compute per-agent
    factor arrays (hiv_rel_sus / hiv_rel_sev / hiv_rel_imm; 1.0 for HIV-),
    and multiply each HPV module's rel_sus by hiv_rel_sus. The rel_sev and
    rel_imm factors are *read* by HPV.set_prognoses, HPV.step_state, and the
    vaccine products (see those sites) — applied where they compose correctly
    with CrossImmunity, which overwrites rel_sus each step before this runs.
    Must be registered AFTER CrossImmunity in the connectors list.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hpv_modules = None
        self.hiv_module = None
        self.define_states(
            ss.FloatArr('hiv_rel_sus', default=1.0),
            ss.FloatArr('hiv_rel_sev', default=1.0),
            ss.FloatArr('hiv_rel_imm', default=1.0),
        )

    def init_pre(self, sim):
        super().init_pre(sim)
        self.hpv_modules = [m for m in sim.diseases.values() if isinstance(m, HPV)]
        hivs = [m for m in sim.diseases.values() if isinstance(m, HIV)]
        if not self.hpv_modules or not hivs:
            raise ValueError(
                'hpv_hiv_connector requires both HPV genotype module(s) and an '
                'hpv.HIV disease in the sim.'
            )
        self.hiv_module = hivs[0]
        # This connector must run AFTER CrossImmunity, which overwrites rel_sus
        # each step; otherwise the HIV rel_sus factor is silently discarded.
        from .cross_genotype import CrossImmunity
        conns = list(sim.connectors.values())
        xi = next((i for i, c in enumerate(conns) if isinstance(c, CrossImmunity)), None)
        si = next((i for i, c in enumerate(conns) if c is self), None)
        if xi is not None and si is not None and si < xi:
            misc.warn('hpv_hiv_connector is registered before CrossImmunity; the '
                      'HIV rel_sus effect will be overwritten. Register it after '
                      'CrossImmunity.')

    def _cd4_stratum(self, cd4):
        """Return 0 for lt200 (CD4<200), 1 for gt200 (CD4>=200)."""
        return (np.asarray(cd4) >= _CD4_THRESHOLD).astype(int)

    def _factor_array(self, effect, hiv_pos, strata, n):
        """Build a per-agent factor array (1.0 for HIV-, stratum value for HIV+)."""
        out = np.ones(n, dtype=float)
        lt200 = _HIV_EFFECTS[effect]['lt200']
        gt200 = _HIV_EFFECTS[effect]['gt200']
        vals = np.where(strata == 0, lt200, gt200)
        out[hiv_pos] = vals[hiv_pos]
        return out

    def step(self):
        if not self.hpv_modules:
            return
        auids = self.sim.people.auids
        cd4 = np.asarray(self.hiv_module.cd4[auids])
        # Effects apply only to agents who are HIV+ AND have an initialized CD4.
        # HIV- agents have NaN cd4 (never initialized); an HIV+ agent whose cd4
        # is still NaN (pre-init edge case) is treated as neutral (factor 1.0)
        # rather than silently binned into a stratum.
        hiv_pos = np.asarray(self.hiv_module.infected[auids], dtype=bool) & ~np.isnan(cd4)
        strata = self._cd4_stratum(np.nan_to_num(cd4, nan=1e4))
        n = len(auids)
        rel_sus = self._factor_array('rel_sus', hiv_pos, strata, n)
        rel_sev = self._factor_array('rel_sev', hiv_pos, strata, n)
        rel_imm = self._factor_array('rel_imm', hiv_pos, strata, n)
        self.hiv_rel_sus[auids] = rel_sus
        self.hiv_rel_sev[auids] = rel_sev
        self.hiv_rel_imm[auids] = rel_imm
        # Acquisition effect: multiply each module's rel_sus (set by CrossImmunity
        # earlier this step) by the HIV factor. rel_sus is written for all agents,
        # but Starsim only samples it for susceptibles during step_infect.
        for m in self.hpv_modules:
            m.rel_sus[auids] = m.rel_sus[auids] * rel_sus


class HIVStratifiedResults(ss.Analyzer):
    """Placeholder — implemented in M08 Task 8."""
    pass
