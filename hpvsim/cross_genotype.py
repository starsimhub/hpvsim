"""Cross-genotype coordination for multi-genotype HPV.

Two modules that operate across HPV genotypes:

  - ``CrossImmunity`` (``ss.Connector``): per-step, reads each registered
    HPV's ``nab_imm`` / ``cell_imm`` and writes per-target ``rel_sus`` /
    ``sev_imm`` via a cross-protection matrix. Runs between
    ``Disease.step_state`` and ``Disease.step_infect``. Convention: row =
    target genotype, col = source genotype. Effective immunity to target
    ``g`` is ``sum_k cross[g, k] * source[uid, k]``. Diagonals must lie in
    [0, 1].

  - ``HPVTotal`` (``ss.Analyzer``): post-hoc, pools per-genotype results
    into Sim-level totals. Auto-added by ``hpv.Sim`` whenever HPV modules
    are present; accessible at ``sim.results.all_hpv`` (the analyzer's ``name``
    defaults to ``'all_hpv'``).
"""

import numpy as np
import starsim as ss

from . import misc
from .hpv import HPV
from .parameters import GENOTYPE_KEYS


__all__ = ['CrossImmunity', 'HPVTotal']


# Genotypes whose own-immunity is hardcoded to 1.0; other keys use ``own_imm_hr``.
_FULL_OWN_IMM_KEYS = frozenset({'hpv16', 'hpv18'})

# Pairwise cross-protection clade map. 'high' = hpv16 <-> hpv18 (same clade);
# everything else is 'med'.
_CLADE_HIGH_PAIRS = frozenset({
    ('hpv16', 'hpv18'),
    ('hpv18', 'hpv16'),
})


def _build_cross_matrix(keys, scalar_med, scalar_high, own_imm_hr):
    """One (n, n) cross-protection matrix. Diagonal: 1.0 for keys in
    ``_FULL_OWN_IMM_KEYS`` (hpv16, hpv18); else ``own_imm_hr``. Off-diagonal:
    ``scalar_high`` for clade-high pairs, ``scalar_med`` otherwise."""
    n = len(keys)
    m = np.full((n, n), scalar_med, dtype=np.float32)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                m[i, j] = 1.0 if ki in _FULL_OWN_IMM_KEYS else own_imm_hr
            elif (ki, kj) in _CLADE_HIGH_PAIRS:
                m[i, j] = scalar_high
    return m


class CrossImmunity(ss.Connector):
    """Cross-immunity + shared HPV-agent state for multi-genotype HPV.

    Per-step, reads each registered ``HPV`` instance's clearance-conferred
    ``nab_imm`` / ``cell_imm`` and writes per-target ``sev_imm`` and a
    nab-based susceptibility reduction via cross-protection matrices.
    Vaccine-conferred ``vax_imm`` and therapeutic-vaccine-conferred
    ``txvx_imm`` are each combined with the nab contribution via
    independent-protection paths — neither is matrix-multiplied, so the
    CSV per-genotype ``rel_imm`` values are the complete vaccine
    cross-protection profile.

    Combining formula for ``rel_sus``:
        sus_imm_nab[target] = sum_k cross_imm_sus[target, k] * nab_imm[uid, k]
        rel_sus[target]     = (1 - sus_imm_nab[target]) * (1 - vax_imm[target]) * (1 - txvx_imm[target])

    Also owns per-agent ``rel_sev`` — an intrinsic biological severity
    scaler sampled once per agent and shared across every genotype's
    progression, so each agent has a single intrinsic progression speed.
    It lives on the connector rather than per HPV module so that all
    genotypes read the same per-agent draw from ``set_prognoses``.
    """

    def __init__(self, cross_imm_sus=None, cross_imm_sev=None, pars=None, **kwargs):
        super().__init__()
        # User-supplied matrices override the auto-built defaults; auto-built
        # matrices are constructed at init_pre from the scalar med/high pars.
        self.cross_imm_sus = cross_imm_sus
        self.cross_imm_sev = cross_imm_sev
        self.hpv_modules = None
        self.genotype_index = None
        self.define_states(
            # Per-agent biological severity scaler, shared across all HPV
            # genotypes. Sampled once on first need via _ensure_rel_sev.
            ss.FloatArr('rel_sev', label='Relative severity (biological)', default=1.0),
            ss.BoolState('rel_sev_sampled', default=False),
        )
        self.define_pars(
            # Folded normal via abs() in _ensure_rel_sev. Default loc=1, scale=0.2
            # has < 1e-6 negative tail so it's effectively a positive-truncated
            # normal; calibration may lower loc (e.g. to normal(0.87, 0.2)).
            rel_sev=ss.normal(loc=1.0, scale=0.2),
            # Scalar medium/high cross-immunity used by get_cross_immunity to
            # build the sus/sev matrices when explicit matrices aren't passed.
            cross_imm_sus_med=0.3,
            cross_imm_sus_high=0.5,
            cross_imm_sev_med=0.5,
            cross_imm_sev_high=0.7,
            own_imm_hr=0.9,
        )
        self.update_pars(pars=pars, **kwargs)

    def make_cross_immunity(self, keys=None):
        """Return ``(m_sus, m_sev)`` matrices built from ``self.pars`` scalars
        (``cross_imm_sus_med/high``, ``cross_imm_sev_med/high``, ``own_imm_hr``).
        ``keys`` defaults to ``GENOTYPE_KEYS``.
        """
        if keys is None:
            keys = GENOTYPE_KEYS
        own = self.pars.own_imm_hr
        m_sus = _build_cross_matrix(keys, self.pars.cross_imm_sus_med,
                                    self.pars.cross_imm_sus_high, own)
        m_sev = _build_cross_matrix(keys, self.pars.cross_imm_sev_med,
                                    self.pars.cross_imm_sev_high, own)
        return m_sus, m_sev

    def init_pre(self, sim):
        super().init_pre(sim)
        # Discover HPV modules in registration order.
        self.hpv_modules = [m for m in sim.diseases.values() if isinstance(m, HPV)]
        if not self.hpv_modules:
            misc.warn('CrossImmunity: no HPV diseases registered; Connector is a no-op.')
            self.genotype_index = {}
            return
        keys = tuple(m.genotype for m in self.hpv_modules)
        self.genotype_index = {k: i for i, k in enumerate(keys)}

        # Populate defaults if matrices not supplied.
        if self.cross_imm_sus is None or self.cross_imm_sev is None:
            m_sus, m_sev = self.make_cross_immunity(keys=keys)
            if self.cross_imm_sus is None:
                self.cross_imm_sus = m_sus
            if self.cross_imm_sev is None:
                self.cross_imm_sev = m_sev

        # Cast and validate.
        self.cross_imm_sus = np.asarray(self.cross_imm_sus, dtype=np.float32)
        self.cross_imm_sev = np.asarray(self.cross_imm_sev, dtype=np.float32)
        n = len(self.hpv_modules)
        for label, m in (('cross_imm_sus', self.cross_imm_sus),
                         ('cross_imm_sev', self.cross_imm_sev)):
            if m.shape != (n, n):
                raise ValueError(
                    f'CrossImmunity.{label}: shape {m.shape} does not match '
                    f'number of HPV modules {n}'
                )
            diag = np.diag(m)
            if (diag < 0).any() or (diag > 1).any():
                raise ValueError(
                    f'CrossImmunity.{label}: diagonal entries must be in '
                    f'[0, 1]; got {diag}'
                )

    def ensure_rel_sev(self, uids):
        """Sample ``rel_sev`` for any of ``uids`` that don't have a sample yet.

        Called from each HPV module's ``set_prognoses`` so that each agent
        is sampled exactly once, at first need, regardless of module init
        order. Subsequent calls for the same uids are no-ops.
        """
        if len(uids) == 0:
            return
        unset_mask = ~self.rel_sev_sampled[uids]
        if not unset_mask.any():
            return
        unset = uids[unset_mask]
        self.rel_sev[unset] = np.abs(self.pars.rel_sev.rvs(unset))
        self.rel_sev_sampled[unset] = True

    def step(self):
        # Catch any unset agents (births/immigrants since last step) before
        # the downstream HPV step_infect samples new infections.
        self.ensure_rel_sev(self.sim.people.alive.uids)
        if not self.hpv_modules:
            return
        # Clearance-conferred immunity — flows through cross-protection matrix.
        nab  = np.column_stack([m.nab_imm.values  for m in self.hpv_modules])
        cell = np.column_stack([m.cell_imm.values for m in self.hpv_modules])
        # Vaccine-conferred immunity — applied directly per target genotype,
        # NOT through the matrix. Shape: (n_agents, n_genotypes).
        vax   = np.column_stack([m.vax_imm.values   for m in self.hpv_modules])
        txvx  = np.column_stack([m.txvx_imm.values  for m in self.hpv_modules])
        sus_imm_nab = nab  @ self.cross_imm_sus.T
        sev_imm     = cell @ self.cross_imm_sev.T
        np.clip(sus_imm_nab, 0.0, 1.0, out=sus_imm_nab)
        np.clip(sev_imm,     0.0, 1.0, out=sev_imm)
        np.clip(vax,         0.0, 1.0, out=vax)
        np.clip(txvx,        0.0, 1.0, out=txvx)
        auids = self.sim.people.auids
        for i, m in enumerate(self.hpv_modules):
            # Three independent protection paths:
            #   - clearance cross-protection (matrix path, nab_imm)
            #   - prophylactic vaccine (direct path, vax_imm)
            #   - therapeutic vaccine (direct path, txvx_imm)
            # All reduce susceptibility multiplicatively. sev_imm comes only
            # from clearance (vaccines don't reduce severity beyond rel_sus).
            m.rel_sus[auids] = m.rel_sus[auids] * (
                (1.0 - sus_imm_nab[:, i])
                * (1.0 - vax[:, i])
                * (1.0 - txvx[:, i])
            )
            m.sev_imm[auids] = sev_imm[:, i]


class HPVTotal(ss.Analyzer):
    """Analyzer that pools per-genotype HPV results into Sim-level totals.

    Schema is mirrored from the per-genotype HPV modules at init time, so
    HPVTotal automatically gains a matching ``all_hpv.<metric>`` entry for
    each per-genotype result. Three aggregation strategies are applied:

      - **People-level union** for per-agent state counts listed in
        ``_UNION_STATES``: boolean OR across each module's BoolState array,
        then counted (an agent infected with any genotype counts once).
      - **Custom derivation** for results that need it: ``n_susceptible``
        (= n_alive - n_infected), ``prevalence`` (= n_infected / n_alive),
        and the extra ``cum_infections_unique`` (people-level cumulative
        unique-agent count, complementing the sum-of-flows ``cum_infections``).
      - **Element-wise sum** across module result arrays for everything
        else, computed in ``finalize_results``. Cancer flows/cumulatives
        are exact under this (cancer is attributed to one genotype per
        agent); infection flows overcount co-infections by design.

    Auto-added by ``hpv.Sim`` whenever HPV modules are present.
    """

    def __init__(self, *args, **kwargs):
        # Results land under ``sim.results[self.name]``. Default the name to
        # 'all_hpv' so the pooled totals read as
        # ``sim.results.all_hpv.cum_infections`` (cleaner than the class name's
        # 'hpvtotal'). Note 'hpv' itself is taken — the HPV DNA screening test
        # is a product module named 'hpv', so an 'hpv' analyzer would collide.
        kwargs.setdefault('name', 'all_hpv')
        super().__init__(*args, **kwargs)

    # Per-agent state counts aggregated by boolean OR across modules.
    # Maps result key on the HPV module -> BoolState attribute name.
    _UNION_STATES = {
        'n_infected':  'infected',
        'n_precin':    'precin',
        'n_cin':       'cin',
        'n_cancerous': 'cancerous',
        'n_latent':    'latent',
    }

    # WHO 2000 World Standard Population weights per 5-year age band (0-4
    # through 100+). Sum = 100_035 (rounding artifact vs the nominal
    # 100_000; ASR normalizes by weights.sum() so the total is harmless).
    # Bin 21 (100+) uses an open-ended upper edge of 150.
    WHO2000_5YR_EDGES = np.array(
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
         80, 85, 90, 95, 100, 150], dtype=float)
    WHO2000_5YR_WEIGHTS = np.array(
        [8860, 8690, 8600, 8470, 8220, 7930, 7610, 7150, 6590, 6040, 5370,
         4550, 3720, 2960, 2210, 1520, 910, 440, 150, 40, 5], dtype=float)

    @classmethod
    def who2000_weights_for_edges(cls, edges):
        """Aggregate WHO 2000 5-year weights to arbitrary bin ``edges``.
        Each output bin ``[edges[i], edges[i+1])`` sums the 5-year weights
        whose age midpoint falls within it."""
        edges = np.asarray(edges, dtype=float)
        midpoints = (cls.WHO2000_5YR_EDGES[:-1] + cls.WHO2000_5YR_EDGES[1:]) / 2.0
        out = np.zeros(len(edges) - 1, dtype=float)
        for i in range(len(edges) - 1):
            mask = (midpoints >= edges[i]) & (midpoints < edges[i + 1])
            out[i] = cls.WHO2000_5YR_WEIGHTS[mask].sum()
        return out

    @classmethod
    def compute_asr(cls, cancers_by_age, n_female_by_age, edges):
        """Age-standardized cancer incidence per 100k person-years (WHO 2000).

        Args:
            cancers_by_age: per-bin annual cancer counts.
            n_female_by_age: per-bin female population at risk.
            edges: bin edges (len = counts + 1); weights aggregate to
                these bins from ``WHO2000_5YR_WEIGHTS``.
        """
        weights = cls.who2000_weights_for_edges(edges)
        cancers = np.asarray(cancers_by_age, dtype=float)
        n = np.asarray(n_female_by_age, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.where(n > 0, cancers / n * 1e5, 0.0)
        return float(np.sum(rates * weights) / weights.sum())

    # HPV result keys we override with custom derivation, not naive sum/union.
    _DERIVED = ('n_susceptible', 'prevalence')

    # HPV result keys not aggregated (bookkeeping artifacts).
    _SKIP = ('timevec',)

    def init_pre(self, sim):
        """Discover HPV modules once at init; mirrors CrossImmunity's pattern.

        Set ``hpv_modules``/``hiv_module`` before the ``super().init_pre``
        call, since starsim's ``Module.init_pre`` invokes ``self.init_results``
        which reads them.
        """
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv_module = misc.hiv_module(sim)
        super().init_pre(sim)
        # Allocate per-ti WHO2000-binned histograms (rows = ti, cols = 5-year
        # bins). Populated in step() and consumed by finalize_results() to
        # derive per-year `asr_cancer_incidence`. See compute_asr / WHO2000_*.
        n_pts = len(sim.timevec)
        n_bins = len(self.WHO2000_5YR_EDGES) - 1
        self._cancers_by_who_bin       = np.zeros((n_pts, n_bins), dtype=float)
        self._cancer_deaths_by_who_bin = np.zeros((n_pts, n_bins), dtype=float)
        self._females_by_who_bin       = np.zeros((n_pts, n_bins), dtype=float)

    def init_results(self):
        """Mirror schema from per-genotype HPV results + add derived/extras.

        Diseases initialize before analyzers in Starsim's setup, so the
        per-genotype HPV result definitions (dtype + label) are available
        as a template when this runs.

        The union-based stocks (n_infected, n_precin, n_cin, n_cancerous) are
        declared as ``dtype=float`` so that ``step()`` can write scale-weighted
        values (fine agents carry ``scale=1/ratio`` and must count as 1/ratio).
        """
        super().init_results()
        hpvs = self.hpv_modules
        if not hpvs:
            return
        template = hpvs[0].results
        defs = []
        for key, src in template.items():
            if key in self._SKIP or key in self._DERIVED:
                continue
            # Union-based stocks are written as scale-weighted floats in
            # step(); all other results mirror the per-genotype dtype.
            result_dtype = float if key in self._UNION_STATES else src.dtype
            defs.append(ss.Result(key, dtype=result_dtype,
                                  label=f'{src.label} (any genotype)'))
        # Derived results (computed in step()).
        defs.append(ss.Result('n_susceptible', dtype=float,
                              label='Currently uninfected with any genotype'))
        # prevalence is a ratio in [0,1]; scale=False so finalize doesn't
        # multiply it by pop_scale.
        defs.append(ss.Result('prevalence', dtype=float, scale=False,
                              label='Prevalence of any HPV genotype'))
        # Extra result with no per-genotype counterpart.
        defs.append(ss.Result('cum_infections_unique', dtype=int,
                              label='Cumulative agents ever infected with any genotype'))
        # Age-standardized cancer incidence / mortality (WHO 2000, per 100k
        # person-years). Populated in finalize_results from per-ti
        # WHO2000-binned histograms; per-ti positions carry the annualized
        # value of that timestep's year (so every ti within a calendar year
        # reads the same ASR).
        defs.append(ss.Result('asr_cancer_incidence', dtype=float, scale=False,
                              label='Age-standardized cervical cancer incidence '
                                    '(WHO 2000, per 100,000 person-years)'))
        defs.append(ss.Result('asr_cancer_mortality', dtype=float, scale=False,
                              label='Age-standardized cervical cancer mortality '
                                    '(WHO 2000, per 100,000 person-years)'))
        if self.hiv_module is not None:
            defs.append(ss.Result('cancers_with_hiv', dtype=float, label='New cancers (HIV+)'))
            defs.append(ss.Result('cancers_no_hiv', dtype=float, label='New cancers (HIV-)'))
            defs.append(ss.Result('cancer_incidence_with_hiv', dtype=float, scale=False,
                                  label='Cancer incidence per 100k (HIV+)'))
            defs.append(ss.Result('cancer_incidence_no_hiv', dtype=float, scale=False,
                                  label='Cancer incidence per 100k (HIV-)'))
            defs.append(ss.Result('cancer_rate_ratio', dtype=float, scale=False,
                                  label='Cancer incidence rate ratio (HIV+/HIV-)'))
        self.define_results(*defs)

    def step(self):
        """Capture per-step union-based state counts and derived results.

        Element-wise sums for the remaining results are computed in
        ``finalize_results`` once each module's full per-step array exists.

        Union stocks (n_infected, n_precin, n_cin, n_cancerous) are written as
        scale-weighted floats: ``people.scale_flows(uids_in_state)`` so that
        fine agents (scale=1/ratio) count as 1/ratio, not 1. Derived results
        (n_susceptible, prevalence) are computed from scale-weighted totals.
        """
        ti = self.ti
        hpvs = self.hpv_modules
        if not hpvs:
            return
        people = self.sim.people
        # auids contains all currently-tracked agents (alive + recently dead
        # pending removal). alive.uids filters to those marked alive.
        alive_uids = people.alive.uids
        n_alive_sw = people.scale_flows(alive_uids)
        if n_alive_sw == 0:
            return
        # Per-agent state unions across modules, restricted to alive agents.
        # `.values` is the auid-indexed active-agent view; the alive mask still
        # matters because auids holds agents who died this step (remove_dead
        # runs after analyzers). The mask is invariant across states, so hoist.
        auids = people.auids
        alive_mask = people.alive.values
        union_arrays = {}
        for key, attr in self._UNION_STATES.items():
            # Boolean union across all HPV modules, filtered to alive agents.
            in_state = np.zeros(len(auids), dtype=bool)
            for m in hpvs:
                in_state |= getattr(m, attr).values
            in_state &= alive_mask
            uids_in = auids[in_state]
            union_arrays[key] = uids_in
            # Scale-weighted count: fine agents count as 1/ratio, not 1.
            self.results[key][ti] = (
                people.scale_flows(uids_in) if len(uids_in) > 0 else 0.0
            )
        # Derived from scale-weighted n_infected.
        sw_inf = float(self.results['n_infected'][ti])
        self.results['n_susceptible'][ti] = n_alive_sw - sw_inf
        self.results['prevalence'][ti] = sw_inf / n_alive_sw
        # Cumulative unique: agents whose ti_first_infection has fired on
        # any genotype (including init-seeded), among those still alive.
        # `.values` is auid-indexed, but auids still holds agents who died
        # this step (remove_dead runs after analyzers), so mask by alive.
        ever_infected = np.zeros_like(people.alive.values)
        for m in hpvs:
            ever_infected |= np.isfinite(m.ti_first_infection.values)
        ever_infected &= people.alive.values
        self.results['cum_infections_unique'][ti] = int(ever_infected.sum())

        # Populate WHO2000-binned histograms used to derive per-year ASR
        # (incidence + mortality) in finalize_results.
        self._accumulate_asr_histograms(ti, people, hpvs)

    def _accumulate_asr_histograms(self, ti, people, hpvs):
        """Fill per-ti WHO2000 5-year-bin histograms for alive-females,
        new cancer events, and new cancer-death events. Aggregated to
        per-year ASR in ``finalize_results`` via ``compute_asr``."""
        ages = people.age.values
        female_mask = people.female.values
        alive_bool = people.alive.values
        edges = self.WHO2000_5YR_EDGES
        w = getattr(people, 'scale', None)
        # Denominator: currently alive females per bin (scale-weighted).
        alive_female = alive_bool & female_mask
        wts_af = w.values[alive_female] if w is not None else None
        self._females_by_who_bin[ti, :] = np.histogram(
            ages[alive_female], bins=edges, weights=wts_af)[0]
        # Numerator: cancers realized this step, across genotypes.
        new_cancer = np.zeros_like(alive_bool)
        for m in hpvs:
            new_cancer |= ((m.ti_cancerous.values == ti) & m.cancerous.values)
        new_cancer &= alive_bool
        wts_nc = w.values[new_cancer] if w is not None else None
        self._cancers_by_who_bin[ti, :] = np.histogram(
            ages[new_cancer], bins=edges, weights=wts_nc)[0]
        if self.hiv_module is not None:
            self._update_hiv_cancer_results(ti, people, new_cancer)
        # Numerator: cancer deaths realized this step, across genotypes.
        # ``ti_dead_cancer`` is set at scheduled cancer-death time; the agent
        # is still marked alive at ti (remove_dead runs after analyzers).
        new_cd = np.zeros_like(alive_bool)
        for m in hpvs:
            new_cd |= (m.ti_dead_cancer.values == ti)
        new_cd &= alive_bool
        wts_cd = w.values[new_cd] if w is not None else None
        self._cancer_deaths_by_who_bin[ti, :] = np.histogram(
            ages[new_cd], bins=edges, weights=wts_cd)[0]

    def _update_hiv_cancer_results(self, ti, people, new_cancer):
        """HIV-stratified new-cancer counts/incidence/rate ratio, from the
        same ``new_cancer`` mask used for the ASR histogram. Runs at this
        module's own (sim-cadence) ti, so no risk of a faster HIV clock
        double-counting the same event across multiple HIV sub-ticks."""
        alive = people.alive.values
        scale = people.scale.values
        hiv_pos = self.hiv_module.infected.values & alive
        hiv_neg = (~self.hiv_module.infected.values) & alive
        n_pos = float((hiv_pos * scale).sum())
        n_neg = float((hiv_neg * scale).sum())
        cancers_with_hiv = float(((new_cancer & hiv_pos) * scale).sum())
        cancers_no_hiv = float(((new_cancer & hiv_neg) * scale).sum())
        self.results['cancers_with_hiv'][ti] = cancers_with_hiv
        self.results['cancers_no_hiv'][ti] = cancers_no_hiv
        inc_with_hiv = cancers_with_hiv / n_pos * 1e5 if n_pos else 0.0
        inc_no_hiv = cancers_no_hiv / n_neg * 1e5 if n_neg else 0.0
        self.results['cancer_incidence_with_hiv'][ti] = inc_with_hiv
        self.results['cancer_incidence_no_hiv'][ti] = inc_no_hiv
        self.results['cancer_rate_ratio'][ti] = inc_with_hiv / inc_no_hiv if inc_no_hiv else 0.0

    def finalize_results(self):
        """Sum across modules for all results not handled by step()."""
        super().finalize_results()
        hpvs = self.hpv_modules
        if not hpvs:
            return
        handled_in_step = (set(self._UNION_STATES)
                           | set(self._DERIVED)
                           | {'cum_infections_unique',
                              'asr_cancer_incidence', 'asr_cancer_mortality',
                              'timevec'})
        template = hpvs[0].results
        for key in template.keys():
            if key in self._SKIP or key in handled_in_step:
                continue
            self.results[key][:] = sum(m.results[key] for m in hpvs)
        self._finalize_asr()

    def _finalize_asr(self):
        """Aggregate per-ti WHO2000-binned cancer / cancer-death counts to
        annual (sum) and alive-female counts to annual (mean), then write
        annualized ``asr_cancer_incidence`` / ``asr_cancer_mortality`` to
        every ti within each calendar year (all ti in year Y hold Y's ASR)."""
        edges = self.WHO2000_5YR_EDGES
        years = np.asarray(self.sim.timevec.years).astype(int)
        inc = np.zeros(len(years), dtype=float)
        mort = np.zeros(len(years), dtype=float)
        for y in np.unique(years):
            mask = years == y
            n_female_year = self._females_by_who_bin[mask].mean(axis=0)
            cancers_year = self._cancers_by_who_bin[mask].sum(axis=0)
            cd_year = self._cancer_deaths_by_who_bin[mask].sum(axis=0)
            inc[mask] = self.compute_asr(cancers_year, n_female_year, edges)
            mort[mask] = self.compute_asr(cd_year, n_female_year, edges)
        self.results['asr_cancer_incidence'][:] = inc
        self.results['asr_cancer_mortality'][:] = mort