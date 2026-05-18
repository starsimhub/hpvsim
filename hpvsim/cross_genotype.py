"""Cross-genotype coordination for multi-genotype HPV.

Two modules that operate across HPV genotypes:

  - ``CrossImmunity`` (``ss.Connector``): per-step, reads each registered
    HPV's ``nab_imm`` / ``cell_imm`` and writes per-target ``rel_sus`` /
    ``sev_imm`` via a cross-protection matrix. Runs between
    ``Disease.step_state`` and ``Disease.step_infect``. Convention: row =
    target genotype, col = source genotype. Effective immunity to target
    ``g`` is ``sum_k cross[g, k] * source[uid, k]``. Diagonals must lie in
    [0, 1].

  - ``Aggregate`` (``ss.Analyzer``): post-hoc, pools per-genotype results
    into Sim-level ``*_any`` aggregates. Auto-added by ``hpv.Sim`` whenever
    HPV modules are present; accessible at ``sim.results.aggregate``.
"""

import numpy as np
import starsim as ss

from . import misc
from .hpv import HPV
from .parameters import get_cross_immunity


__all__ = ['CrossImmunity', 'Aggregate']


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


class Aggregate(ss.Analyzer):
    """Analyzer that pools per-genotype results into Sim-level *_any aggregates.

    Results are accessible at ``sim.results.aggregate``:
      - ``cum_infections_any`` — per-step sum of new_infections across genotypes,
        cumsum'd. Sum-of-flows: overcounts agents with co-infections.
      - ``cum_cancers_any`` — sum of per-genotype cum_cancers (no double-counting
        since cancer is attributed to a single genotype).
      - ``new_cancer_deaths_any`` — per-step sum of new_cancer_deaths.

    The analyzer is auto-added by ``hpv.Sim`` whenever HPV modules are present.
    ``step()`` captures per-step new_infections; ``finalize_results()`` assembles
    the cumulative aggregates using HPV disease results (available because
    analyzers finalize after disease modules in Starsim's finalization order).
    """

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('cum_infections_any', dtype=int,
                      label='Cumulative agents ever infected (any genotype)'),
            ss.Result('cum_cancers_any', dtype=int,
                      label='Cumulative cancers (any genotype)'),
            ss.Result('new_cancer_deaths_any', dtype=int,
                      label='New cancer deaths (any genotype)'),
        )

    def _hpvs(self):
        return [d for d in self.sim.diseases.values() if isinstance(d, HPV)]

    def step(self):
        """Capture per-step new_infections (needed before they could be overwritten)."""
        ti = self.sim.ti
        hpvs = self._hpvs()
        if not hpvs:
            return
        # Sum-of-flows across genotypes, not boolean-OR — overcounts co-infections.
        self.results['cum_infections_any'][ti] = sum(
            m.results.new_infections[ti] for m in hpvs
        )

    def finalize_results(self):
        """Assemble cumulative aggregates after HPV disease modules have finalized."""
        super().finalize_results()
        hpvs = self._hpvs()
        if not hpvs:
            return
        # Cumulative sum of the per-step counts captured in step().
        self.results['cum_infections_any'][:] = np.cumsum(
            np.asarray(self.results['cum_infections_any'])
        )
        # cum_cancers_any: sum across genotypes. HPV.finalize_results runs
        # before this analyzer, so cum_cancers is already populated.
        self.results['cum_cancers_any'][:] = sum(
            np.asarray(m.results.cum_cancers) for m in hpvs
        )
        self.results['new_cancer_deaths_any'][:] = sum(
            np.asarray(m.results.new_cancer_deaths) for m in hpvs
        )