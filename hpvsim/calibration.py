"""HPVsim calibration: the goodness of fit to HPVsim's data, for ``sim.fit()``.

Parameters to fit are given with bounds (``ss.par``), and ``sim.fit()`` fits
them; this module turns HPVsim's calibration data into the function of the
sim it minimizes::

    sim = hpv.Sim(location='nigeria', genotypes=[16, 18], data=['nigeria_cancer_cases.csv'],
                  pars=dict(beta=ss.par(0.25, bounds=[0.15, 0.35])))
    fit = sim.fit(gof=hpv.calibration.gof, method='optuna', n_trials=100)

``hpv.Sim(data=...)`` loads the data (``hpv.data.loaders.load_calib_data``),
adds the analyzers it needs, and runs the sim past its last year.

Provides:
    - gof: the weighted mismatch of a run sim to its data.
    - data_analyzers: the analyzers the data needs.
    - compute_gof: goodness-of-fit between actual and predicted arrays.
    - build_sim: routes flat dotted-key pars to sim.pars,
      sim.diseases[<genotype>].pars, or the CrossImmunity connector.

Standardized ``data`` DataFrame: index='t' (float years), columns dot-scoped:
    - ``all_hpv.<name>``           scalar-per-year pooled target; looked up
      on ``sim.results.all_hpv`` (HPVTotal) at eval time
      (e.g. ``all_hpv.asr_cancer_incidence``).
    - ``all_hpv.<name>.<bin>``     age-stratified pooled target; looked up
      on the auto-attached ``all_hpv_by_age`` ``by_age`` analyzer
      (e.g. ``all_hpv.cancers.0-15``).
    - ``by_genotype.<name>.<g>``   per-genotype distribution target;
      computed at eval time via ``hpv.results_by_genotype``.
"""
import numpy as np
import pandas as pd
import sciris as sc

from .analyzers import by_age, results_by_genotype
from .parameters import route_pars as build_sim
from .data.loaders import load_calib_data


__all__ = ['gof', 'data_analyzers', 'build_sim', 'compute_gof', 'default_eval_fn']


def gof(sim, weights=None, gof_kwargs=None):
    """The weighted mismatch of a run sim to its calibration data (``hpv.Sim(data=...)``), for ``sim.fit(gof=hpv.calibration.gof)``

    Args:
        sim (hpv.Sim): a run sim made with ``data=``
        weights (dict): the weight of each column (default 1); with weights, give
            ``gof=functools.partial(hpv.calibration.gof, weights=...)``
        gof_kwargs (dict): passed to ``compute_gof()``

    Returns:
        The sum over the data's columns of ``compute_gof()`` times its weight
    """
    data = getattr(sim, 'calib_data', None)
    if data is None:
        errormsg = 'hpv.calibration.gof needs a sim made with calibration data, e.g. hpv.Sim(..., data=["cancer_cases.csv"])'
        raise ValueError(errormsg)
    return default_eval_fn(sim, data=data, weights=weights, gof_kwargs=gof_kwargs)


def data_analyzers(data, analyzers=()):
    """The analyzers that calibration data needs that are not among ``analyzers`` (``all_hpv_by_age``, for targets by age)

    Args:
        data (DataFrame): standardized calibration data (see ``hpv.data.loaders.load_calib_data``)
        analyzers (list): the analyzers the sim has already
    """
    existing = list(analyzers)
    _setup_analyzers(existing, data)
    return existing[len(list(analyzers)):]


def compute_gof(actual, predicted, normalize=True, use_frac=False,
                use_squared=False, as_scalar='none', eps=1e-9,
                skestimator=None, estimator=None, **kwargs):
    """Goodness-of-fit between two arrays — normalized absolute error by default.

    Mean squared error is ``normalize=False, use_squared=True, as_scalar='mean'``.

    Args:
        actual:      array of observed (data) points.
        predicted:   array of model points; same shape as ``actual``.
        normalize:   if True, divide errors by ``max(|actual|)``.
        use_frac:    if True, divide each error by ``max(actual, predicted) + eps``
                     instead of normalizing by the global max.
        use_squared: if True, square the per-point errors.
        as_scalar:   collapse to a scalar via ``'sum'`` / ``'mean'`` / ``'median'``;
                     ``'none'`` returns the per-point array.
        eps:         small constant guarding the ``use_frac`` denominator.
        skestimator: scikit-learn metric name (e.g. ``'mean_squared_error'``).
        estimator:   user-supplied callable ``(actual, predicted, **kwargs)``.
        kwargs:      forwarded to the scikit-learn / custom estimator.
    """
    actual = np.array(actual, dtype=float, copy=True)
    predicted = np.array(predicted, dtype=float, copy=True)

    if skestimator is not None:
        import sklearn.metrics as sm
        return getattr(sm, skestimator)(actual, predicted, **kwargs)

    if estimator is not None:
        return estimator(actual, predicted, **kwargs)

    gofs = np.abs(actual - predicted)

    if normalize and not use_frac:
        actual_max = np.abs(actual).max()
        if actual_max > 0:
            gofs = gofs / actual_max

    if use_frac:
        if (actual < 0).any() or (predicted < 0).any():
            # Fractional error is ill-defined on negatives; keep absolute error.
            pass
        else:
            maxvals = np.maximum(actual, predicted) + eps
            gofs = gofs / maxvals

    if use_squared:
        gofs = gofs ** 2

    if as_scalar == 'sum':
        return float(np.sum(gofs))
    if as_scalar == 'mean':
        return float(np.mean(gofs))
    if as_scalar == 'median':
        return float(np.median(gofs))
    return gofs


# ---------------------------------------------------------------------------
# Data-key scoping + analyzer-attachment helpers.
# ---------------------------------------------------------------------------

ALL_HPV = 'all_hpv'                # column prefix: pooled target
BY_GENOTYPE = 'by_genotype'        # column prefix: per-genotype distribution
ALL_HPV_BY_AGE = 'all_hpv_by_age'  # by_age analyzer name for pooled age-stratified targets

# Genotype-stratified target name -> (per-HPV stock result key, normalize).
_GENOTYPE_DIST_MAP = {
    'precin_genotype_dist':    ('precin',    True),
    'cin_genotype_dist':       ('cin',       True),
    'cancerous_genotype_dist': ('cancerous', True),
}


def _parse_column(col):
    """Parse a standardized column name into (scope, name, subkey).

    Two accepted forms (dot-scoped, 2 or 3 levels):
      ``'all_hpv.<name>'``            scalar; subkey=None. Looked up on
                                       ``sim.results.all_hpv`` (HPVTotal).
      ``'all_hpv.<name>.<bin>'``      age-stratified; looked up on the
                                       ``all_hpv_by_age`` analyzer.
      ``'by_genotype.<name>.<g>'``    per-genotype; computed via
                                       ``results_by_genotype``.
    """
    parts = col.split('.')
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(
        f'hpv.calibration.gof: unrecognized column format {col!r}; expected '
        f'"<scope>.<name>" or "<scope>.<name>.<subkey>".')


def _parse_bin_label(label):
    """``'0-15'`` -> ``(0.0, 15.0)``; ``'85+'`` -> ``(85.0, 150.0)``."""
    if label.endswith('+'):
        return (float(label.rstrip('+')), 150.0)
    lo_str, hi_str = label.split('-', 1)
    return (float(lo_str), float(hi_str))


def _edges_from_labels(labels):
    """Set of contiguous by_age bin labels -> sorted edges array."""
    ranges = sorted(_parse_bin_label(l) for l in labels)
    lows = [r[0] for r in ranges]
    highs = [r[1] for r in ranges]
    for i in range(len(ranges) - 1):
        if highs[i] != lows[i + 1]:
            raise ValueError(
                f'hpv.calibration.gof: age bin labels not contiguous: {sorted(labels)}')
    return np.array(lows + [highs[-1]], dtype=float)


def _setup_analyzers(analyzers, data):
    """Inspect ``data`` columns and append any analyzers missing from the list ``analyzers``.

    - Age-stratified pooled columns ``all_hpv.<name>.<bin>`` require a
      ``by_age`` named ``all_hpv_by_age`` with matching edges + years;
      created if absent, edges-validated if present.
    - Scalar ``all_hpv.<name>`` columns are read from HPVTotal at eval time
      (no attachment needed).
    - ``by_genotype.<name>.<g>`` columns compute via ``results_by_genotype``
      at eval time (no attachment needed).
    """
    age_result_names = []          # preserves order of appearance
    age_labels_by_result = {}      # name -> set of bin labels
    unknown = []
    for col in data.columns:
        try:
            scope, name, subkey = _parse_column(col)
        except ValueError:
            unknown.append(col)
            continue
        if scope == ALL_HPV and subkey is not None:
            if name not in age_result_names:
                age_result_names.append(name)
            age_labels_by_result.setdefault(name, set()).add(subkey)
        elif scope == ALL_HPV and subkey is None:
            pass  # HPVTotal result -- no attachment
        elif scope == BY_GENOTYPE:
            if name not in _GENOTYPE_DIST_MAP:
                raise ValueError(
                    f'hpv.calibration.gof: unknown by_genotype target {name!r} '
                    f'(from column {col!r}); known: {sorted(_GENOTYPE_DIST_MAP)}.')
        else:
            unknown.append(col)
    if unknown:
        raise ValueError(f'hpv.Sim(data=...): unrecognized data columns: {unknown}')

    if age_result_names:
        # Targets sharing a bin-label set share a by_age analyzer; distinct
        # schemes get all_hpv_by_age, all_hpv_by_age_1, ...
        groups = {}  # bin-set key -> list of result names in insertion order
        for name in age_result_names:
            key = frozenset(age_labels_by_result[name])
            groups.setdefault(key, []).append(name)
        years = sorted(float(y) for y in data.index)
        for i, (labels, names) in enumerate(groups.items()):
            edges = _edges_from_labels(labels)
            aname = ALL_HPV_BY_AGE if i == 0 else f'{ALL_HPV_BY_AGE}_{i}'
            _get_or_create_all_hpv_by_age(analyzers, names, years, edges, name=aname)
    return


def _get_or_create_all_hpv_by_age(analyzers, result_keys, years, edges, name=ALL_HPV_BY_AGE):
    """Return a ``by_age`` analyzer with the given ``name`` in ``analyzers`` (create +
    append if absent; edges-validated if present). Multiple analyzers can
    coexist under distinct names (``all_hpv_by_age``, ``all_hpv_by_age_1``, ...)
    for targets that need different bin schemes."""
    for a in analyzers:
        if getattr(a, 'name', None) == name:
            if not np.array_equal(a.edges, edges):
                raise ValueError(
                    f'hpv.calibration.gof: existing {name} analyzer has '
                    f'edges {list(a.edges)}, incompatible with data-derived '
                    f'edges {list(edges)}.')
            return a
    ar = by_age(result_keys, years=years, edges=edges, name=name)
    analyzers.append(ar)
    return ar


def _extract_columns(sim, data):
    """Return a DataFrame same-shape as ``data`` with sim-side values per
    column. Extraction is cached per underlying source (by_age analyzer,
    results_by_genotype table, HPVTotal result) so multi-subkey columns
    sharing a base don't re-extract.
    """
    tv_years = np.asarray(sim.timevec.years).astype(int)
    all_hpv_results = sim.results['all_hpv'] if 'all_hpv' in sim.results else None
    # result name -> analyzer, across all all_hpv_by_age* analyzers.
    result_to_analyzer = {}
    for a in sim.analyzers.values():
        aname = getattr(a, 'name', None) or ''
        if isinstance(a, by_age) and (aname == ALL_HPV_BY_AGE
                                       or aname.startswith(f'{ALL_HPV_BY_AGE}_')):
            for k in a.keys:
                result_to_analyzer[k] = a
    cache = {}
    out = pd.DataFrame(np.nan, index=data.index.copy(), columns=data.columns)
    for col in data.columns:
        scope, name, subkey = _parse_column(col)
        if scope == ALL_HPV and subkey is not None:
            key = ('by_age', name)
            if key not in cache:
                cache[key] = result_to_analyzer[name].to_dataframe(name)
            out[col] = cache[key][subkey].reindex(data.index)
        elif scope == ALL_HPV and subkey is None:
            if all_hpv_results is None or name not in all_hpv_results:
                raise KeyError(
                    f'hpv.calibration.gof: column {col!r} needs '
                    f'sim.results.all_hpv.{name}; not found.')
            vals = np.asarray(all_hpv_results[name])
            for year in data.index:
                matches = np.where(tv_years == int(float(year)))[0]
                if len(matches):
                    out.at[year, col] = float(vals[matches[0]])
        elif scope == BY_GENOTYPE:
            key = ('by_genotype', name)
            if key not in cache:
                sim_key, normalize = _GENOTYPE_DIST_MAP[name]
                df = results_by_genotype(sim, key=sim_key, normalize=normalize)
                df.index = df.index.astype(float)
                df.columns = [str(c) for c in df.columns]
                cache[key] = df
            out[col] = cache[key][subkey].reindex(data.index)
    return out


def default_eval_fn(sim, data, weights=None, gof_kwargs=None):
    """Weighted sum of ``compute_gof`` across each column of ``data``.

    ``data`` is a wide DataFrame (index='t', dot-scoped columns). For each
    column, sim-side values are extracted via ``_extract_columns``, NaN
    cells in ``data`` are skipped, and per-column mismatches are weighted
    by ``weights.get(column_name, 1.0)`` before summing.
    """
    weights = weights or {}
    gof_kwargs = dict(gof_kwargs or {})
    gof_kwargs.setdefault('as_scalar', 'sum')
    actual = _extract_columns(sim, data)
    total = 0.0
    for col in data.columns:
        expected_col = data[col].dropna()
        if len(expected_col) == 0:
            continue
        actual_col = actual[col].loc[expected_col.index]
        if actual_col.isna().any():
            missing = expected_col.index[actual_col.isna()].tolist()
            raise KeyError(
                f'hpv.calibration.gof: sim produced no value for {col!r} at '
                f'years {missing}; expand sim.stop or update data.')
        mismatch = compute_gof(
            np.asarray(expected_col.values, dtype=float),
            np.asarray(actual_col.values, dtype=float),
            **gof_kwargs,
        )
        total += float(mismatch) * float(weights.get(col, 1.0))
    return total

