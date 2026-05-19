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
    """Apply calib_pars to a (copy of) sim and return it.

    calib_pars is a flat dict with dotted-key paths. Routing rules:
      - No dot: writes to sim.pars[key].
      - '<genotype>.<...>': writes to sim.diseases[<genotype>].pars[...].
      - 'cross_immunity.<matrix>.<tgt>.<src>': writes a cell into the
        CrossImmunity connector's named matrix.
      - Anything else: raises ValueError.

    ss.Calibration passes a sc.dcp(sim) in per trial, so we mutate freely.
    """
    from .hpv import HPV
    from .cross_genotype import CrossImmunity

    # Discover registered genotype keys (names on each HPV disease module).
    hpv_keys = {d.name for d in sim.diseases.values() if isinstance(d, HPV)}

    for key, value in calib_pars.items():
        parts = key.split('.')
        if len(parts) == 1:
            # Top-level sim par.
            sim.pars[parts[0]] = value
        elif parts[0] in hpv_keys:
            # Per-genotype par: walk into sim.diseases[<g>].pars[...].
            target = sim.diseases[parts[0]].pars
            for p in parts[1:-1]:
                target = target[p]
            target[parts[-1]] = value
        elif parts[0] == 'cross_immunity':
            # cross_immunity.<matrix>.<tgt>.<src>
            if len(parts) != 4:
                raise ValueError(
                    f'build_sim: cross_immunity key must be of the form '
                    f'cross_immunity.<matrix>.<tgt>.<src>; got {key!r}')
            _, matrix_name, tgt, src = parts
            connectors = [c for c in sim.connectors.values()
                          if isinstance(c, CrossImmunity)]
            if not connectors:
                raise ValueError(
                    f'build_sim: cross_immunity key {key!r} requires a '
                    f'CrossImmunity connector on the sim')
            conn = connectors[0]
            idx = {m.name: i for i, m in enumerate(conn.hpv_modules)}
            i, j = idx[tgt], idx[src]   # matrix is [target, source]
            getattr(conn, matrix_name)[i, j] = value
        else:
            raise ValueError(
                f'build_sim: unrecognized calib_par key {key!r}. '
                f'Expected a bare sim par name, a <genotype>.<...> path '
                f'(genotypes: {sorted(hpv_keys)}), or cross_immunity.<...>.')
    return sim


def cancer_by_age(expected, *, likelihood='normal', weight=1):
    """Implementation in Task 8."""
    raise NotImplementedError('cancer_by_age — implemented in Task 8')


def hpv_prev_by_age(expected, *, likelihood='beta', weight=1):
    """Implementation in Task 8."""
    raise NotImplementedError('hpv_prev_by_age — implemented in Task 8')


def cancer_genotype_dist(expected, *, likelihood='dirichlet', weight=1):
    """Implementation in Task 8."""
    raise NotImplementedError('cancer_genotype_dist — implemented in Task 8')