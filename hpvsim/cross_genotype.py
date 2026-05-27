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
    are present; accessible at ``sim.results.hpvtotal``.
"""

import numpy as np
import starsim as ss

from . import misc
from .hpv import HPV
from .parameters import get_cross_immunity


__all__ = ['CrossImmunity', 'HPVTotal']


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
    scaler that v2 stores once per agent and uses across every genotype's
    progression. Per-module storage in v3 would give each genotype its
    own independent rel_sev draw for the same agent, breaking the v2-
    implied "intrinsic progression speed" correlation. Owning rel_sev
    here keeps the v2 semantic — sampled once per agent, read by every
    HPV module's ``set_prognoses``.
    """

    def __init__(self, cross_imm_sus=None, cross_imm_sev=None, **kwargs):
        super().__init__(**kwargs)
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
        # Folded normal via abs() in _ensure_rel_sev; with loc=1.0, scale=0.2
        # the negative tail mass is < 1e-6 so the practical distribution is
        # equivalent to v2's normal_pos(1, 0.2).
        self._rel_sev_dist = ss.normal(loc=1.0, scale=0.2)

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
            m_sus, m_sev = get_cross_immunity(keys=keys)
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

        Called from each HPV module's ``set_prognoses`` so the v2 semantic
        — sampled once per agent at first need — holds regardless of
        module init order. Subsequent calls for the same uids are no-ops.
        """
        if len(uids) == 0:
            return
        unset_mask = ~self.rel_sev_sampled[uids]
        if not unset_mask.any():
            return
        unset = uids[unset_mask]
        self.rel_sev[unset] = np.abs(self._rel_sev_dist.rvs(unset))
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
            m.rel_sus[auids] = (
                (1.0 - sus_imm_nab[:, i])
                * (1.0 - vax[:, i])
                * (1.0 - txvx[:, i])
            )
            m.sev_imm[auids] = sev_imm[:, i]


class HPVTotal(ss.Analyzer):
    """Analyzer that pools per-genotype HPV results into Sim-level totals.

    Schema is mirrored from the per-genotype HPV modules at init time, so
    HPVTotal automatically gains a matching ``hpvtotal.<metric>`` entry for
    each per-genotype result. Three aggregation strategies are applied:

      - **People-level union** for per-agent state counts listed in
        ``_UNION_STATES``: boolean OR across each module's BoolState array,
        then counted. Matches v2's ``any_hpv_prevalence`` semantics.
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

    # Per-agent state counts aggregated by boolean OR across modules.
    # Maps result key on the HPV module -> BoolState attribute name.
    _UNION_STATES = {
        'n_infected':  'infected',
        'n_precin':    'precin',
        'n_cin':       'cin',
        'n_cancerous': 'cancerous',
    }

    # HPV result keys we override with custom derivation, not naive sum/union.
    _DERIVED = ('n_susceptible', 'prevalence')

    # HPV result keys not aggregated (bookkeeping artifacts).
    _SKIP = ('timevec',)

    def init_pre(self, sim):
        """Discover HPV modules once at init; mirrors CrossImmunity's pattern.

        Set ``hpv_modules`` before the ``super().init_pre`` call, since starsim's
        ``Module.init_pre`` invokes ``self.init_results`` which reads this.
        """
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)

    def init_results(self):
        """Mirror schema from per-genotype HPV results + add derived/extras.

        Diseases initialize before analyzers in Starsim's setup, so the
        per-genotype HPV result definitions (dtype + label) are available
        as a template when this runs.
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
            defs.append(ss.Result(key, dtype=src.dtype,
                                  label=f'{src.label} (any genotype)'))
        # Derived results (computed in step()).
        defs.append(ss.Result('n_susceptible', dtype=int,
                              label='Currently uninfected with any genotype'))
        defs.append(ss.Result('prevalence', dtype=float,
                              label='Prevalence of any HPV genotype'))
        # Extra result with no per-genotype counterpart.
        defs.append(ss.Result('cum_infections_unique', dtype=int,
                              label='Cumulative agents ever infected with any genotype'))
        self.define_results(*defs)

    def step(self):
        """Capture per-step union-based state counts and derived results.

        Element-wise sums for the remaining results are computed in
        ``finalize_results`` once each module's full per-step array exists.
        """
        ti = self.sim.ti
        hpvs = self.hpv_modules
        if not hpvs:
            return
        people = self.sim.people
        # auids includes dead-but-not-yet-removed agents, so the alive mask
        # (not len(auids)) is the right denominator for prevalence and the
        # right filter for current-state counts.
        alive = people.alive.values
        n_alive = int(alive.sum())
        if n_alive == 0:
            return
        # Per-agent state unions across modules, masked to alive agents.
        union_arrays = {}
        for key, attr in self._UNION_STATES.items():
            u = np.zeros(alive.shape, dtype=bool)
            for m in hpvs:
                u |= getattr(m, attr).values
            u &= alive
            union_arrays[key] = u
            self.results[key][ti] = int(u.sum())
        # Derived from n_infected.
        n_inf = int(union_arrays['n_infected'].sum())
        self.results['n_susceptible'][ti] = n_alive - n_inf
        self.results['prevalence'][ti] = n_inf / n_alive
        # Cumulative unique: agents whose ti_first_infection has fired on
        # any genotype (including init-seeded). Filter to alive agents so the
        # count reflects currently-visible ever-infected agents. Once an
        # agent dies and is removed from auids, they drop out of this count.
        ever_infected = np.zeros(alive.shape, dtype=bool)
        for m in hpvs:
            ever_infected |= np.isfinite(m.ti_first_infection.values)
        ever_infected &= alive
        self.results['cum_infections_unique'][ti] = int(ever_infected.sum())

    def finalize_results(self):
        """Sum across modules for all results not handled by step()."""
        super().finalize_results()
        hpvs = self.hpv_modules
        if not hpvs:
            return
        handled_in_step = (set(self._UNION_STATES)
                           | set(self._DERIVED)
                           | {'cum_infections_unique', 'timevec'})
        template = hpvs[0].results
        for key in template.keys():
            if key in self._SKIP or key in handled_in_step:
                continue
            self.results[key][:] = sum(m.results[key] for m in hpvs)