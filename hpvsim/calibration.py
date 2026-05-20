"""HPVsim calibration — thin wrapper around ss.Calibration + helpers.

Provides:
    - hpv.Calibration: ss.Calibration subclass with HPV-aware defaults.
    - build_sim: default build_fn that routes flat dotted-key calib_pars to
      sim.pars, sim.diseases[<genotype>].pars, or the CrossImmunity connector.
    - CalibComponent factories for the three common HPV target shapes:
      cancer_by_age, hpv_prev_by_age, cancer_genotype_dist.
"""
import pandas as pd
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


def _make_age_bin_extract_fn(result_key, age_bin):
    """Build a closure that extracts one age bin as a 't'-indexed
    DataFrame with a single 'x' column. Used for per-age-bin Normal
    components: Starsim's Normal.compute_nll merges on 't' and reads
    rep['x_e']/rep['x_a'], so each component must produce single-channel
    'x' data."""
    def extract_fn(sim):
        ar = _find_age_results(sim)
        df = ar.to_dataframe(key=result_key)
        return pd.DataFrame({'x': df[age_bin].values}, index=df.index)
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


def _per_age_bin_normal_components(expected, *, result_key, name_prefix, weight,
                                   sigma2_floor=1.0):
    """Build one ss.Normal component per age-bin column in `expected`.

    Starsim's Normal.compute_nll merges expected/actual on 't' and reads
    'x_e'/'x_a' — so each component must produce single-channel 'x' data.
    With one component per age bin, an N-bin DataFrame becomes N components,
    each carrying a 't'-indexed 'x' column with that bin's values.

    Each component has its sigma2 set explicitly using a Poisson-like
    approximation (variance ≈ mean, floored at sigma2_floor). The default
    auto-compute in ss.Normal (sigma2 = SSE/N over expected values) is
    degenerate with a single timepoint — sigma2 collapses to (e-a)^2 and
    yields NaN log-likelihoods when e==a exactly.
    """
    _validate_age_schema(expected, None)
    components = []
    for age_bin in expected.columns:
        col = expected[age_bin].values.astype(float)
        sub_expected = pd.DataFrame({'x': col}, index=expected.index)
        # Poisson-like sigma2: variance ≈ mean for count data, floored to
        # avoid sigma2=0 when expected counts are zero in this bin.
        sigma2 = max(float(col.mean()), sigma2_floor)
        components.append(ss.Normal(
            name=f'{name_prefix}:{age_bin}',
            expected=sub_expected,
            extract_fn=_make_age_bin_extract_fn(result_key, age_bin),
            conform='step_containing',
            weight=weight,
            sigma2=sigma2,
        ))
    return components


def cancer_by_age(expected, *, weight=1):
    """List of ss.Normal components for age-binned cancer counts.

    AgeResults snapshots the count of agents with cancerous=True at each
    requested year (point-in-time stock). conform='step_containing' picks
    the sim timestep that contains the target year.

    Returns one component per age-bin column in `expected`; each component
    carries that bin's counts as a single 't'-indexed 'x' column.
    """
    return _per_age_bin_normal_components(
        expected, result_key='cancers', name_prefix='cancer_by_age',
        weight=weight,
    )


def hpv_prev_by_age(expected, expected_n=None, *, weight=1):
    """Components for age-binned HPV prevalence — Normal or BetaBinomial.

    Two modes, picked by the shape of the inputs:

    - **Ratio (Normal):** call as ``hpv_prev_by_age(expected_ratio)``.
      ``expected_ratio`` is a DataFrame with `t`-named index and age-bin
      columns of prevalence values in [0,1]. Returns one ss.Normal per
      age bin, comparing simulated ratios to observed ratios. Use when
      target data is reported as prevalences only (no denominator).

    - **Counts (BetaBinomial):** call as
      ``hpv_prev_by_age(expected_x, expected_n)``. Both DataFrames share
      the same `t`-indexed schema; cells of ``expected_x`` are positives
      and cells of ``expected_n`` are totals per age bin. Returns one
      ss.BetaBinomial per age bin. Use when target data is reported as
      raw counts (positives + sample size), which gives a proper count
      likelihood instead of a Gaussian on the ratio.

    The BetaBinomial path reads simulated (x, n) per bin from
    ``AgeResults.to_xn_per_bin('hpv_prevalence')``.
    """
    if expected_n is None:
        return _per_age_bin_normal_components(
            expected, result_key='hpv_prevalence',
            name_prefix='hpv_prev_by_age', weight=weight,
        )
    _validate_age_schema(expected, None)
    _validate_age_schema(expected_n, None)
    if list(expected.columns) != list(expected_n.columns):
        raise ValueError(
            f'hpv_prev_by_age: expected and expected_n must share columns; '
            f'got {list(expected.columns)} vs {list(expected_n.columns)}'
        )
    if list(expected.index) != list(expected_n.index):
        raise ValueError(
            f'hpv_prev_by_age: expected and expected_n must share index; '
            f'got {list(expected.index)} vs {list(expected_n.index)}'
        )
    components = []
    for age_bin in expected.columns:
        sub_expected = pd.DataFrame(
            {
                'x': expected[age_bin].values.astype(float),
                'n': expected_n[age_bin].values.astype(float),
            },
            index=expected.index,
        )
        components.append(ss.BetaBinomial(
            name=f'hpv_prev_by_age:{age_bin}',
            expected=sub_expected,
            extract_fn=_make_prev_xn_extract_fn(age_bin),
            conform='step_containing',
            weight=weight,
        ))
    return components


def _make_prev_xn_extract_fn(age_bin):
    """Closure: pull (x, n) per timepoint for one age bin of hpv_prevalence."""
    def extract_fn(sim):
        ar = _find_age_results(sim)
        xn = ar.to_xn_per_bin('hpv_prevalence')
        return xn[age_bin]
    return extract_fn


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