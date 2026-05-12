"""HPVsim analyzers.

Currently exposes ``Aggregate``, which pools per-genotype HPV results into
Sim-level ``*_any`` aggregates. Auto-added by ``hpv.Sim`` whenever HPV
modules are present; accessible at ``sim.results.aggregate``.
"""

import numpy as np
import starsim as ss

from .hpv import HPV


__all__ = ['Aggregate']


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