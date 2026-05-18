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
    """Cross-immunity Connector for multi-genotype HPV.

    Reads each registered ``HPV`` instance's source-genotype ``nab_imm`` and
    ``cell_imm``; writes per-target ``rel_sus`` (= 1 - sus_imm) and ``sev_imm``
    each step, after Disease.step_state and before Disease.step_infect.
    """

    def __init__(self, cross_imm_sus=None, cross_imm_sev=None, **kwargs):
        super().__init__(**kwargs)
        self.cross_imm_sus = cross_imm_sus
        self.cross_imm_sev = cross_imm_sev
        self.hpv_modules = None
        self.genotype_index = None

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

    def step(self):
        if not self.hpv_modules:
            return
        nab  = np.column_stack([m.nab_imm.values  for m in self.hpv_modules])
        cell = np.column_stack([m.cell_imm.values for m in self.hpv_modules])
        sus_imm = nab  @ self.cross_imm_sus.T
        sev_imm = cell @ self.cross_imm_sev.T
        np.clip(sus_imm, 0.0, 1.0, out=sus_imm)
        np.clip(sev_imm, 0.0, 1.0, out=sev_imm)
        auids = self.sim.people.auids
        for i, m in enumerate(self.hpv_modules):
            m.rel_sus[auids] = 1.0 - sus_imm[:, i]
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
    _SKIP = ('timevec', 'n_rel_sev_sampled')

    def _hpvs(self):
        return [d for d in self.sim.diseases.values() if isinstance(d, HPV)]

    def init_results(self):
        """Mirror schema from per-genotype HPV results + add derived/extras.

        Diseases initialize before analyzers in Starsim's setup, so the
        per-genotype HPV result definitions (dtype + label) are available
        as a template when this runs.
        """
        super().init_results()
        hpvs = self._hpvs()
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
        hpvs = self._hpvs()
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
        hpvs = self._hpvs()
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