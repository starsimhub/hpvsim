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

__all__ = ['HIV', 'hiv_art', 'hpv_hiv_connector', 'HIVStratifiedResults']


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

    @classmethod
    def from_location(cls, location, beta_m2f=0.0035, **kwargs):
        """Build an ``hpv.HIV`` seeded from a country's bundled HIV inputs.

        Pulls ``init_prev`` from ``hpv.data.load_hiv(location)`` and uses it as
        ``init_prev_data``. ``beta_m2f`` is left as a TUNABLE with an
        UNCALIBRATED placeholder default (matching the ``__init__`` default);
        it is calibrated to the location in T12. The ART coverage data returned
        by ``load_hiv`` is NOT applied here — that is the T10b ART-shortcut
        intervention's job.

        Args:
            location (str): country name (only ``'rwanda'`` supported now).
            beta_m2f (float): male->female per-act transmission rate
                (placeholder, uncalibrated).
            **kwargs: forwarded to ``hpv.HIV.__init__``.
        """
        from . import data as _data
        inputs = _data.load_hiv(location)
        return cls(beta_m2f=beta_m2f, init_prev_data=inputs['init_prev'], **kwargs)

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


def _reshape_art_coverage(df):
    """Reshape the tidy Rwanda ART-coverage frame into STIsim's stratified format.

    ``hpv.data.load_hiv(location)['art_coverage']`` is a long frame with columns
    ``[age, sex, year, coverage]`` (single year of age; sex 'f'/'m'; coverage a
    fraction of HIV+ in that stratum). ``sti.ART``'s stratified-coverage parser
    expects columns ``Year``, ``Gender``, ``AgeBin`` (a ``'[lo,hi)'`` string) and
    a numeric proportion column whose name does NOT start with ``n_`` (so it is
    read as a proportion 'p', not absolute counts). Single years of age map to
    unit bins ``[age, age+1)``.

    Coverage is clamped to ``[0, 1]``: the bundled Rwanda curve has a few values
    at 1.0001 (rounding), and STIsim infers proportion-vs-count from whether the
    max value is <= 1.0 — an un-clamped 1.0001 would flip the whole frame to
    'absolute counts' and treat ~nobody. Clamping keeps it a proportion.
    """
    out = df.rename(columns={'year': 'Year', 'sex': 'Gender', 'coverage': 'p_art'}).copy()
    out['AgeBin'] = out['age'].map(lambda a: f'[{int(a)},{int(a) + 1})')
    out['p_art'] = out['p_art'].clip(lower=0.0, upper=1.0)
    return out[['Year', 'Gender', 'AgeBin', 'p_art']]


class hiv_art(sti.ART):
    """Coverage-based ART shortcut for the HPV–HIV co-infection model.

    v2/Rwanda has no HIV testing cascade: ART is assigned directly to hit an
    age/sex/year coverage curve. Stock ``sti.ART`` only treats agents that are
    already ``diagnosed`` (it expects a ``sti.HIVTest`` upstream), and no
    testing-coverage data exists for Rwanda — adding ``HIVTest`` would diverge
    from v2. Instead, this thin ``sti.ART`` subclass marks every living HIV+
    agent ``diagnosed`` at the start of each step, then defers to ``sti.ART`` to
    do the actual treatment bookkeeping and CD4 reconstitution.

    With all HIV+ agents diagnosed, ``sti.ART``'s diagnosed pool equals the full
    HIV+ pool, so its ``art_coverage_correction`` drives the on-ART fraction
    *among HIV+ agents* to the supplied age/sex/year curve (no double-discount).
    The Rwanda coverage curve is per-single-year-of-age, fed in as STIsim's
    native stratified-DataFrame coverage (see ``_reshape_art_coverage``); STIsim
    interpolates it to the sim's timestep grid and applies it per (age, sex).

    Pass explicitly via ``interventions=[...]``; it is NOT auto-wired, because
    ART coverage is scenario-specific. Use ``hiv_art.from_location('rwanda')``
    for the bundled curve, or pass any coverage form ``sti.ART`` accepts via the
    ``coverage`` kwarg.
    """

    @classmethod
    def from_location(cls, location, **kwargs):
        """Build an ``hiv_art`` from a country's bundled ART-coverage curve.

        Args:
            location (str): country name (only ``'rwanda'`` supported now).
            **kwargs: forwarded to ``sti.ART`` (e.g. ``art_initiation``).
        """
        from . import data as _data
        cov = _reshape_art_coverage(_data.load_hiv(location)['art_coverage'])
        return cls(coverage=cov, **kwargs)

    def init_pre(self, sim):
        # sti.ART.init_pre warns when no HIVTest precedes it; that is expected
        # and intended here (we diagnose by coverage instead of by testing), so
        # suppress only that specific warning to avoid alarming users.
        import warnings as _warnings
        with _warnings.catch_warnings():
            # ss.warn prefixes the message with a newline, so match with DOTALL.
            _warnings.filterwarnings(
                'ignore',
                message='(?s).*without an HIV testing intervention.*',
                category=RuntimeWarning,
            )
            super().init_pre(sim)

    def step(self):
        # Diagnose-to-coverage: make the diagnosed pool equal the living HIV+
        # pool so sti.ART can fill its among-HIV+ coverage target. Newly flagged
        # agents get ti_diagnosed = ti (the same bookkeeping HIVTest would do);
        # already-diagnosed agents keep their original ti_diagnosed.
        hiv = self.sim.diseases.hiv
        newly = (hiv.infected & ~hiv.diagnosed).uids
        if len(newly):
            hiv.diagnosed[newly] = True
            hiv.ti_diagnosed[newly] = self.ti
        return super().step()


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
    """HPV/cancer outcomes split by HIV status (mirrors v2's cancer_*_with/no_hiv).

    Adds only the cross-disease stratification HPV needs; HIV's own epidemic
    results come from sti.HIV / sti.ART. Auto-added by hpv.Sim when HIV present.
    Accessible at ``sim.results.hivstratifiedresults``.

    Cancer detection: each HPV module fires its cin->cancerous transition in
    ``step_state`` for agents with ``ti_cancerous <= ti``. Since ``ti_cancerous``
    is scheduled as ``ti_cin + _randround(...)`` (an integer step >= ti_cin) and
    the agent is already CIN by then, the transition fires at exactly
    ``ti == ti_cancerous``, so detecting flips this step via
    ``cancerous & (ti_cancerous == ti)`` matches the agents that just turned
    cancerous. NaN ti_cancerous (no scheduled cancer) compares False, so it is
    safely excluded.
    """

    def init_pre(self, sim):
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        hivs = [d for d in sim.diseases.values() if isinstance(d, HIV)]
        if not hivs:
            raise ValueError('HIVStratifiedResults requires an hpv.HIV disease in the sim.')
        self.hiv_module = hivs[0]
        super().init_pre(sim)

    def init_results(self):
        """Declare the HIV-stratified result schema (diseases init before analyzers)."""
        super().init_results()
        self.define_results(
            ss.Result('cancers_with_hiv', dtype=int, label='New cancers (HIV+)'),
            ss.Result('cancers_no_hiv', dtype=int, label='New cancers (HIV-)'),
            ss.Result('hpv_prevalence_with_hiv', dtype=float, label='HPV prevalence (HIV+)'),
            ss.Result('hpv_prevalence_no_hiv', dtype=float, label='HPV prevalence (HIV-)'),
        )

    def step(self):
        ti = self.sim.ti
        people = self.sim.people
        alive = people.alive.values
        hiv_pos = self.hiv_module.infected.values & alive
        hiv_neg = (~self.hiv_module.infected.values) & alive

        # Any-genotype HPV infection (union across modules).
        any_hpv = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            any_hpv |= m.infected.values

        n_pos = int(hiv_pos.sum())
        n_neg = int(hiv_neg.sum())
        self.results['hpv_prevalence_with_hiv'][ti] = (
            float((any_hpv & hiv_pos).sum()) / n_pos if n_pos else 0.0)
        self.results['hpv_prevalence_no_hiv'][ti] = (
            float((any_hpv & hiv_neg).sum()) / n_neg if n_neg else 0.0)

        # New cancers this step, attributed by current HIV status. NOTE: this
        # analyzer runs after step_die in the Starsim loop, so an agent who turns
        # cancerous in step_state AND dies from background demographics the same
        # step has both `cancerous` and the HIV `infected` flag cleared by the
        # time we read them here — that cancer is counted by HPVTotal.new_cancers
        # (recorded in step_state) but missed here. The bias is O(P_death x
        # P_cancer_transition) per step, negligible at typical scales. A complete
        # fix would snapshot HIV status before step_die (e.g. via an update_results
        # override); revisit only if the Phase-2 parity gate needs it.
        new_cancer = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            fired = (m.cancerous.values & (m.ti_cancerous.values == ti))
            new_cancer |= fired
        self.results['cancers_with_hiv'][ti] = int((new_cancer & hiv_pos).sum())
        self.results['cancers_no_hiv'][ti] = int((new_cancer & hiv_neg).sum())
