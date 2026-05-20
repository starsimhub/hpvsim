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

    # ss.Calibration deep-copies an uninitialized sim before calling build_fn.
    # Support both initialized sims (sim.diseases is an ndict) and
    # uninitialized sims (disease modules live in sim.pars['diseases'] list).
    if hasattr(sim, 'diseases'):
        # Post-init: diseases and connectors are ndict attributes on sim.
        disease_lookup = {d.name: d for d in sim.diseases.values()
                          if isinstance(d, HPV)}
        connector_list = [c for c in sim.connectors.values()
                          if isinstance(c, CrossImmunity)]
    else:
        # Pre-init: modules are lists in sim.pars.
        disease_lookup = {d.name: d
                          for d in sim.pars.get('diseases', [])
                          if isinstance(d, HPV)}
        connector_list = [c for c in sim.pars.get('connectors', [])
                          if isinstance(c, CrossImmunity)]

    hpv_keys = set(disease_lookup.keys())

    for key, value in calib_pars.items():
        # ss.Calibration._sample_from_trial passes each entry as a spec dict
        # {'low':..., 'high':..., 'value': <sampled_float>, 'path':..., ...}.
        # Extract the actual scalar when that shape is present.
        if isinstance(value, dict) and 'value' in value:
            value = value['value']
        parts = key.split('.')
        if len(parts) == 1:
            # Top-level sim par.
            sim.pars[parts[0]] = value
        elif parts[0] in hpv_keys:
            # Per-genotype par: walk into disease.pars[...].
            target = disease_lookup[parts[0]].pars
            for p in parts[1:-1]:
                target = target[p]
            # Special case: pars.beta is stored as a per-network dict
            # {'sexualnetwork': [f2m, m2f]}. If the caller supplies a scalar,
            # scale all entries proportionally (preserving the F→M / M→F ratio).
            final_key = parts[-1]
            if (final_key == 'beta' and sc.isnumber(value)
                    and isinstance(target.get(final_key), dict)):
                old_beta = target[final_key]
                first_entry = next(iter(old_beta.values()))
                old_ref = first_entry[0] if isinstance(first_entry, list) else first_entry
                if old_ref == 0:
                    scale = 1.0
                else:
                    scale = value / old_ref
                target[final_key] = {
                    net: ([v[0] * scale, v[1] * scale] if isinstance(v, list)
                          else v * scale)
                    for net, v in old_beta.items()
                }
            else:
                target[final_key] = value
        elif parts[0] == 'cross_immunity':
            # cross_immunity.<matrix>.<tgt>.<src>
            if len(parts) != 4:
                raise ValueError(
                    f'build_sim: cross_immunity key must be of the form '
                    f'cross_immunity.<matrix>.<tgt>.<src>; got {key!r}')
            _, matrix_name, tgt, src = parts
            if not connector_list:
                raise ValueError(
                    f'build_sim: cross_immunity key {key!r} requires a '
                    f'CrossImmunity connector on the sim')
            conn = connector_list[0]
            idx = {m.name: i for i, m in enumerate(conn.hpv_modules)}
            i, j = idx[tgt], idx[src]   # matrix is [target, source]
            getattr(conn, matrix_name)[i, j] = value
        else:
            raise ValueError(
                f'build_sim: unrecognized calib_par key {key!r}. '
                f'Expected a bare sim par name, a <genotype>.<...> path '
                f'(genotypes: {sorted(hpv_keys)}), or cross_immunity.<...>.')
    return sim


def _find_age_results(sim):
    """Locate the AgeResults analyzer on the sim, regardless of its
    name/key. Raises if there isn't exactly one."""
    from .analyzers import AgeResults
    matches = [a for a in sim.analyzers.values() if isinstance(a, AgeResults)]
    if len(matches) != 1:
        raise ValueError(
            f'CalibComponent extract: expected exactly one AgeResults '
            f'analyzer on sim; found {len(matches)}')
    return matches[0]


def _make_extract_fn(result_key, expected):
    """Build a closure that pulls AgeResults[result_key] in expected's schema."""
    def extract_fn(sim):
        ar = _find_age_results(sim)
        df = ar.to_dataframe(key=result_key)
        # Align on expected's index/columns; missing rows/cols => KeyError,
        # which surfaces schema mismatches at evaluation time.
        return df.loc[expected.index, expected.columns]
    return extract_fn


def _validate_age_schema(expected, sim_template):
    """expected must have a 't'-named index and string column labels."""
    if expected.index.name != 't':
        raise ValueError(
            f'expected.index.name must be \'t\'; got {expected.index.name!r}')
    if not all(isinstance(c, str) for c in expected.columns):
        raise ValueError(
            f'expected.columns must be strings (age-bin labels); '
            f'got {list(expected.columns)}')


def cancer_by_age(expected, *, weight=1):
    """CalibComponent for age-binned cancer counts (stock snapshot, Normal likelihood).

    AgeResults snapshots the count of agents with cancerous=True at each
    requested year. This is a prevalence (stock), not an incident flow.
    conform='step_containing' picks the sim timestep that contains the
    target year, which is appropriate for point-in-time counts.
    """
    _validate_age_schema(expected, None)
    return ss.Normal(
        name='cancer_by_age',
        expected=expected,
        extract_fn=_make_extract_fn('cancers', expected),
        conform='step_containing',
        weight=weight,
    )


def hpv_prev_by_age(expected, *, weight=1):
    """CalibComponent for age-binned HPV prevalence (prevalent, Beta-Binomial likelihood)."""
    _validate_age_schema(expected, None)
    return ss.BetaBinomial(
        name='hpv_prev_by_age',
        expected=expected,
        extract_fn=_make_extract_fn('hpv_prevalence', expected),
        conform='step_containing',
        weight=weight,
    )


def cancer_genotype_dist(expected, *, weight=1):
    """CalibComponent for the cancer-genotype distribution (Dirichlet-Multinomial likelihood)."""
    # Type-dist factory: columns are genotype keys, not age labels.
    if expected.index.name != 't':
        raise ValueError(
            f'expected.index.name must be \'t\'; got {expected.index.name!r}')
    return ss.DirichletMultinomial(
        name='cancer_genotype_dist',
        expected=expected,
        extract_fn=_make_extract_fn('cancerous_genotype_dist', expected),
        conform='step_containing',
        weight=weight,
    )