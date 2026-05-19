"""HPVsim calibration — thin wrapper around ss.Calibration + helpers.

Provides:
    - hpv.Calibration: ss.Calibration subclass with HPV-aware defaults.
    - build_sim: default build_fn that routes flat dotted-key calib_pars to
      sim.pars, sim.diseases[<genotype>].pars, or the CrossImmunity connector.
    - CalibComponent factories for the three common HPV target shapes:
      cancer_by_age, hpv_prev_by_age, cancer_genotype_dist.
"""
import sciris as sc
import starsim as ss


__all__ = ['Calibration', 'build_sim',
           'cancer_by_age', 'hpv_prev_by_age', 'cancer_genotype_dist']


class Calibration(ss.Calibration):
    """HPVsim calibration. Delegates to ss.Calibration with HPV-aware defaults.

    Default build_fn is hpv.calibration.build_sim, which routes flat
    dotted-key calib_pars (e.g. 'beta', 'hpv16.cin_fn.k',
    'cross_immunity.cross_imm_sus.hpv16.hpv18') to the right address.
    """

    def __init__(self, sim, calib_pars, *, build_fn=None, **kwargs):
        if build_fn is None:
            build_fn = build_sim
        kwargs.setdefault('study_name', 'hpvsim_calibration')
        super().__init__(sim, calib_pars, build_fn=build_fn, **kwargs)


def build_sim(sim, calib_pars, **kwargs):
    """Default build_fn for hpv.Calibration. Implementation in Task 7."""
    raise NotImplementedError('build_sim — implemented in Task 7')


def cancer_by_age(expected, *, likelihood='normal', weight=1):
    """Implementation in Task 8."""
    raise NotImplementedError('cancer_by_age — implemented in Task 8')


def hpv_prev_by_age(expected, *, likelihood='beta', weight=1):
    """Implementation in Task 8."""
    raise NotImplementedError('hpv_prev_by_age — implemented in Task 8')


def cancer_genotype_dist(expected, *, likelihood='dirichlet', weight=1):
    """Implementation in Task 8."""
    raise NotImplementedError('cancer_genotype_dist — implemented in Task 8')