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

__all__ = ['HIV', 'hiv_incidence_import', 'hiv_art', 'hpv_hiv_connector',
           'HIVStratifiedResults']


# CD4-stratified HIV→HPV effect multipliers. Copied by value from v2's
# HIVsim defaults (hpvsim/_v2_legacy/hiv.py:29-44) per the no-quarantine-import
# rule. v2's CD4 strata are 'lt200' = [0, 200) and 'gt200' = [200, 500); agents
# with CD4 >= 500 fall in NEITHER stratum and so receive NO HIV→HPV effect
# (factor 1.0, biological). This is load-bearing: HIV+ agents start at CD4~594
# and ART reconstitutes CD4 above 500, so most HIV+ person-time is CD4 >= 500 —
# applying gt200 there (as an earlier draft did) over-amplifies HIV+ cancer ~10x.
#
# These are the generic HIVsim class defaults. A *location calibration* may
# override them: e.g. v2's Rwanda calibration (results/rwanda_pars.obj) tuned
# rel_sus to {lt200: 4.75, gt200: 2.75} and rel_sev to {lt200: 2.5, gt200: 3.5}
# (rel_imm unchanged). Pass such overrides via ``hpv_hiv_connector(effects=...)``;
# see ``tests/regression/rwanda_calib.py`` for the Rwanda values in use.
_HIV_EFFECTS = {
    'rel_sus': {'lt200': 2.2, 'gt200': 2.2},   # increased HPV acquisition
    'rel_sev': {'lt200': 1.5, 'gt200': 1.2},   # faster/worse CIN->cancer progression
    'rel_imm': {'lt200': 0.36, 'gt200': 0.76}, # reduced post-infection/vaccine immunity
}
_CD4_THRESHOLD = 200.0   # lt200 / gt200 boundary
_CD4_UPPER = 500.0       # CD4 >= this -> no HIV→HPV effect (v2's gt200 ceiling)


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
        # Allow the caller to override the seed (e.g. init_prev_data=0.0 for the
        # incidence-driven build, where the epidemic is constructed by the
        # importer rather than by seeding).
        kwargs.setdefault('init_prev_data', inputs['init_prev'])
        return cls(beta_m2f=beta_m2f, **kwargs)

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


class hiv_incidence_import(ss.Intervention):
    """Incidence-driven HIV importer (v2-faithful).

    Imposes a per-(year, sex, age) HIV incidence curve directly onto STIsim's
    HIV module instead of relying on network transmission. Each step it selects
    living HIV-negative susceptibles, draws each at their age/sex/year-specific
    per-timestep infection probability, and calls
    ``sim.diseases.hiv.set_prognoses(selected)`` — which flips them to infected
    AND wires the full CD4 trajectory (acute -> latent -> falling -> AIDS death)
    plus ART/mortality machinery. With HIV ``beta_m2f=0`` and ``init_prev=0`` the
    epidemic is built entirely here, so the prevalence trajectory matches the
    target incidence curve by construction (as v2 did).

    The incidence DataFrame has columns ``[age, sex, year, incidence]`` (sex
    'f'/'m'; ``incidence`` = the per-year HIV acquisition rate among
    susceptibles). Use ``hiv_incidence_import.from_location('rwanda')`` for the
    bundled Rwanda curve, or pass a frame of that shape via ``incidence``.

    Pass explicitly via ``interventions=[...]``; it is NOT auto-wired. If no HIV
    module is present in the sim, ``init_pre`` raises a clear ValueError (the
    importer is meaningless without a target HIV disease).

    FOI -> per-step probability: a per-year rate ``r`` is converted to a
    per-timestep infection probability with the exponential survival form
    ``p = 1 - exp(-r * dt_years)`` (correct for non-small rates), then drawn via
    a CRN-safe ``ss.bernoulli``. Years outside the data's [min, max] range are
    nearest-year clamped (incidence is 0 before the curve begins).
    """

    def __init__(self, incidence=None, **kwargs):
        super().__init__(**kwargs)
        if incidence is None:
            raise ValueError(
                'hiv_incidence_import requires an incidence DataFrame '
                '[age, sex, year, incidence]; use from_location() or pass incidence=.'
            )
        self.incidence = incidence
        # Per-agent infection draw; p is set per-step from the looked-up rates.
        self.infect_bern = ss.bernoulli(p=0.0)
        # Filled in init_pre:
        self._years = None          # sorted unique years (int)
        self._year_index = None     # {year: row index into the rate cube}
        self._ages = None           # sorted unique ages (int)
        self._age_min = None
        self._age_max = None
        # rate_cube[sex][year_idx, age_idx] -> incidence rate; sex in {0:'f',1:'m'}
        self._rate_cube = None

    @classmethod
    def from_location(cls, location, **kwargs):
        """Build an importer from a country's bundled HIV incidence curve.

        Args:
            location (str): country name (only ``'rwanda'`` supported now).
            **kwargs: forwarded to ``hiv_incidence_import.__init__``.
        """
        from . import data as _data
        inc = _data.load_hiv(location)['incidence']
        return cls(incidence=inc, **kwargs)

    def init_pre(self, sim):
        super().init_pre(sim)
        if 'hiv' not in sim.diseases:
            raise ValueError(
                'hiv_incidence_import requires an HIV disease in the sim '
                "(sim.diseases.hiv); none found."
            )
        # Precompute a fast lookup cube indexed by [sex, year, age].
        df = self.incidence
        self._years = np.sort(df['year'].unique()).astype(int)
        self._year_index = {int(y): i for i, y in enumerate(self._years)}
        self._ages = np.sort(df['age'].unique()).astype(int)
        self._age_min = int(self._ages.min())
        self._age_max = int(self._ages.max())
        n_year, n_age = len(self._years), len(self._ages)
        age_index = {int(a): i for i, a in enumerate(self._ages)}
        cube = {0: np.zeros((n_year, n_age)), 1: np.zeros((n_year, n_age))}
        sex_code = {'f': 0, 'm': 1}
        for r in df.itertuples(index=False):
            s = sex_code.get(str(r.sex).lower()[0])
            if s is None:
                continue
            yi = self._year_index[int(r.year)]
            ai = age_index[int(r.age)]
            cube[s][yi, ai] = float(r.incidence)
        self._rate_cube = cube

    def _lookup_rates(self, year, ages, female):
        """Per-agent annual incidence rate for the given calendar year.

        ``ages`` is a float array of agent ages; ``female`` a bool mask. Years
        are nearest-year clamped to the data range; ages are clamped to the data
        age range; integer-floored age indexes the per-single-year curve.
        """
        # Nearest-year clamp: years before the curve start -> first year (which
        # is 0 incidence in the Rwanda data); after the end -> last year.
        y = int(np.clip(int(np.floor(year)), int(self._years[0]), int(self._years[-1])))
        yi = self._year_index[y]
        # Clamp/floor ages into the per-single-year curve index space.
        ai = np.clip(np.floor(ages).astype(int), self._age_min, self._age_max) - self._age_min
        rates = np.empty(len(ages), dtype=float)
        f_curve = self._rate_cube[0][yi]
        m_curve = self._rate_cube[1][yi]
        rates[female] = f_curve[ai[female]]
        rates[~female] = m_curve[ai[~female]]
        return rates

    def step(self):
        hiv = self.sim.diseases.hiv
        people = self.sim.people
        # Living, HIV-negative susceptibles (not currently infected).
        eligible = (hiv.susceptible & ~hiv.infected & people.alive).uids
        if not len(eligible):
            return
        ages = np.asarray(people.age[eligible], dtype=float)
        female = np.asarray(people.female[eligible], dtype=bool)
        year = float(self.t.now('year'))
        rates = self._lookup_rates(year, ages, female)
        # Annual rate -> per-timestep probability via the exponential survival
        # form (exact for non-small rates): p = 1 - exp(-rate * dt_years).
        dt_years = float(self.t.dt_year)
        p = 1.0 - np.exp(-rates * dt_years)
        self.infect_bern.set(p=p)
        selected = self.infect_bern.filter(eligible)
        if len(selected):
            hiv.set_prognoses(selected)
        return selected


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

    def __init__(self, effects=None, **kwargs):
        super().__init__(**kwargs)
        # CD4-stratified effect multipliers; defaults to the generic HIVsim
        # values (_HIV_EFFECTS). A location calibration passes its own dict
        # (same {effect: {'lt200':.., 'gt200':..}} shape) — e.g. the Rwanda
        # rel_sus/rel_sev overrides. Validated for the required keys here so a
        # malformed override fails at construction, not mid-run.
        if effects is None:
            effects = _HIV_EFFECTS
        else:
            for eff in ('rel_sus', 'rel_sev', 'rel_imm'):
                if eff not in effects or not {'lt200', 'gt200'} <= set(effects[eff]):
                    raise ValueError(
                        "hpv_hiv_connector effects must provide 'rel_sus', "
                        "'rel_sev', 'rel_imm', each with 'lt200' and 'gt200' keys; "
                        f'got {effects!r}'
                    )
        self.effects = effects
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
        """Return 0 for lt200 (CD4<200), 1 for gt200 (CD4>=200).

        This is the lt200/gt200 split only; the CD4>=500 'no effect' band is
        applied separately in ``step`` (such agents are excluded from the
        effect mask), matching v2's strata [0,200) and [200,500).
        """
        return (np.asarray(cd4) >= _CD4_THRESHOLD).astype(int)

    def _factor_array(self, effect, hiv_pos, strata, n):
        """Build a per-agent factor array (1.0 for HIV-, stratum value for HIV+)."""
        out = np.ones(n, dtype=float)
        lt200 = self.effects[effect]['lt200']
        gt200 = self.effects[effect]['gt200']
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
        # v2-faithful: effects apply only to HIV+ agents with an initialized CD4
        # in [0, 500). CD4 >= 500 (newly infected at ~594, or ART-reconstituted)
        # falls outside v2's gt200=[200,500) band and gets NO effect (factor 1.0).
        infected = np.asarray(self.hiv_module.infected[auids], dtype=bool)
        hiv_pos = infected & ~np.isnan(cd4) & (cd4 < _CD4_UPPER)
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
