"""HPVsim calibration — ss.Calibration subclass with a weighted-gof eval.

Provides:
    - hpv.Calibration: ss.Calibration subclass that takes a ``data`` dict of
      observed-target DataFrames and computes a single weighted mismatch
      using ``compute_gof`` (normalized absolute error by default).
    - compute_gof: goodness-of-fit between actual and predicted arrays.
    - build_sim: default build_fn that routes flat dotted-key calib_pars to
      sim.pars, sim.diseases[<genotype>].pars, or the CrossImmunity connector.

The default eval_fn pulls each target's simulated values out of the
``by_age`` analyzer, aligns on (year, column), then sums
``compute_gof`` over the flattened (year × column) values. Per-target
``weights`` scale each result's mismatch before summing.
"""
import tempfile

import numpy as np
import optuna as op
import pandas as pd
import sciris as sc
import starsim as ss

from .analyzers import by_age


__all__ = ['Calibration', 'build_sim', 'compute_gof', 'default_eval_fn']


class Calibration(ss.Calibration):
    """HPVsim calibration. Delegates to ss.Calibration with HPV-aware defaults.

    Two entry points to specify the fit target:

    - ``data={'cancers': df, ...}``: dict of 't'-indexed DataFrames keyed by
      ``by_age`` result name. The default eval_fn extracts the matching
      simulated values, aligns on (year, column), and sums
      ``compute_gof`` across all rows/columns, scaled by per-key
      ``weights`` (a mean-absolute-error mismatch).
    - ``components=[...]`` or a custom ``eval_fn``: standard ss.Calibration
      paths, unchanged.

    Default build_fn is hpv.calibration.build_sim, which routes flat
    dotted-key calib_pars (e.g. 'beta', 'hpv16.cin_fn.k',
    'cross_immunity.cross_imm_sus.hpv16.hpv18') to the right address.
    """

    def __init__(self, sim, calib_pars, *, data=None, weights=None,
                 gof_kwargs=None, build_fn=None, eval_fn=None, eval_kw=None,
                 **kwargs):
        if build_fn is None:
            build_fn = build_sim
        # Give each calibration its own Optuna study database, in a temp dir, so
        # multiple hpv.Calibration runs in one session (or the test suite) do not
        # share or leak trials through a single database in the cwd. Callers can
        # still pass study_name/db_name explicitly (e.g. continue_db resume).
        if 'study_name' not in kwargs and 'db_name' not in kwargs:
            kwargs['study_name'] = 'hpvsim_calibration'
            kwargs['db_name'] = str(sc.path(tempfile.mkdtemp()) / 'hpvsim_calibration.db')

        if data is not None:
            if eval_fn is not None:
                raise ValueError(
                    'hpv.Calibration: pass either data= or eval_fn=, not both.')
            self._validate_data(data)
            eval_fn = default_eval_fn
            eval_kw = sc.mergedicts(eval_kw, dict(
                data=data,
                weights=weights or {},
                gof_kwargs=gof_kwargs or {},
            ))

        super().__init__(sim, calib_pars, build_fn=build_fn,
                         eval_fn=eval_fn, eval_kw=eval_kw, **kwargs)

    def worker(self):
        """Run a single worker.

        Mirrors ``stisim.Calibration.worker``: wraps ``study.optimize`` in a
        try/except so a single worker's SQLite-lock error (or any other
        transient Optuna storage failure) does not propagate through
        Optuna's own error handler and trip its
        ``assert False, 'Should not reach.'``, taking the whole run down.
        Upstream ``ss.Calibration.worker`` calls ``study.optimize`` bare.
        """
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name, sampler=self.run_args.sampler)
        try:
            output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials, callbacks=None)
        except Exception as E:
            print(f'Worker failed with error: {E}')
            output = None
        return output

    @staticmethod
    def _validate_data(data):
        """Each value must be a DataFrame whose index is named 't'."""
        if not isinstance(data, dict):
            raise TypeError(
                f'hpv.Calibration.data must be a dict; got {type(data).__name__}')
        known = (set(by_age._COUNT_KEYS) | set(by_age._PREV_KEYS)
                 | set(by_age._FLOW_KEYS))
        for key, df in data.items():
            if key not in known:
                raise ValueError(
                    f'hpv.Calibration.data: unknown result key {key!r}; '
                    f'must be one of {sorted(known)}')
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f'hpv.Calibration.data[{key!r}] must be a DataFrame; '
                    f'got {type(df).__name__}')
            if df.index.name != 't':
                raise ValueError(
                    f'hpv.Calibration.data[{key!r}].index.name must be \'t\'; '
                    f'got {df.index.name!r}')


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
            # Fractional error on negative quantities is ill-defined; fall
            # back to absolute error rather than producing nonsense.
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


def _find_by_age(sim):
    """Locate the by_age analyzer on the sim, regardless of its
    name/key. Raises if there isn't exactly one."""
    matches = [a for a in sim.analyzers.values() if isinstance(a, by_age)]
    if len(matches) != 1:
        raise ValueError(
            f'hpv.Calibration: expected exactly one by_age analyzer on '
            f'the sim; found {len(matches)}')
    return matches[0]


def _extract_actual(ar, key, expected):
    """Pull `key` from by_age, aligned to expected's (index, columns)."""
    actual = ar.to_dataframe(key)
    missing_rows = [t for t in expected.index if t not in actual.index]
    missing_cols = [c for c in expected.columns if c not in actual.columns]
    if missing_rows or missing_cols:
        raise KeyError(
            f'hpv.Calibration eval: data[{key!r}] references rows '
            f'{missing_rows} / columns {missing_cols} not produced by '
            f'by_age; available rows={list(actual.index)}, '
            f'columns={list(actual.columns)}')
    return actual.loc[expected.index, expected.columns]


def default_eval_fn(sim, data, weights=None, gof_kwargs=None):
    """Default eval_fn: weighted sum of compute_gof across each data target.

    For each ``(key, expected_df)`` in ``data``, pull the simulated values
    from the sim's ``by_age`` analyzer aligned on ``(index, columns)``,
    flatten both, call ``compute_gof(as_scalar='sum')``, then multiply by
    ``weights.get(key, 1.0)``. Returns the total as a single float —
    smaller is better.
    """
    weights = weights or {}
    gof_kwargs = dict(gof_kwargs or {})
    # Default to a scalar-sum gof unless the caller already specified.
    gof_kwargs.setdefault('as_scalar', 'sum')
    ar = _find_by_age(sim)
    total = 0.0
    for key, expected in data.items():
        actual = _extract_actual(ar, key, expected)
        mismatch = compute_gof(
            np.asarray(expected.values, dtype=float).ravel(),
            np.asarray(actual.values, dtype=float).ravel(),
            **gof_kwargs,
        )
        total += float(mismatch) * float(weights.get(key, 1.0))
    return total