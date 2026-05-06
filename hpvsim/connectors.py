"""Cross-immunity Connector: derive per-target rel_sus and sev_imm from
per-source nab_imm and cell_imm via v2's cross-protection matrices.

Convention: row = target genotype, col = source genotype. Effective immunity
to target ``g`` is ``sum_k cross[g, k] * source[uid, k]``. Matrices live on
the Connector instance (not on SimPars). Diagonals must equal 1.0.
"""

import numpy as np
import starsim as ss

from . import misc
from .hpv import HPV
from .parameters import get_cross_immunity


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
            if not np.allclose(np.diag(m), 1.0):
                raise ValueError(
                    f'CrossImmunity.{label}: diagonal must be 1.0; '
                    f'got {np.diag(m)}'
                )

    def step(self):
        # Body added in Task 4.
        return