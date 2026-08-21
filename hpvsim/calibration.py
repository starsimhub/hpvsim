"""HPVsim calibration — ss.Calibration subclass with a weighted-gof eval.

Provides:
    - hpv.Calibration: ss.Calibration subclass that takes ``data`` (a
      standardized wide DataFrame, a dict of pre-scoped DataFrames, or a
      list of CSV paths / long-format DataFrames — all normalized via
      ``hpv.data.loaders.load_calib_data``) and computes a single weighted
      mismatch using ``compute_gof``.
    - compute_gof: goodness-of-fit between actual and predicted arrays.
    - build_sim: default build_fn that routes flat dotted-key calib_pars to
      sim.pars, sim.diseases[<genotype>].pars, or the CrossImmunity connector.

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
import tempfile

import numpy as np
import optuna as op
import pandas as pd
import sciris as sc
import starsim as ss

from .analyzers import by_age, results_by_genotype
from .parameters import route_pars as build_sim
from .data.loaders import load_calib_data


__all__ = ['Calibration', 'build_sim', 'compute_gof', 'default_eval_fn',
           'make_calib_sims']


_LIST_SPEC_KEYS = ('guess', 'low', 'high', 'step')


def _prepare_calib_pars(calib_pars):
    """Flatten a nested-by-scope calib_pars dict and convert list leaves to
    Optuna spec dicts. Nested scopes are collapsed with ``sc.flattendict``
    (dot-separated). Leaves must be ``[best, low, high, step]`` lists (step
    optional) -- Optuna spec dicts already at leaves would over-flatten.
    """
    if any('.' in k for k in calib_pars):
        bad = sorted(k for k in calib_pars if '.' in k)
        raise ValueError(
            f'hpv.Calibration: flat dotted-key calib_pars no longer supported '
            f'(got {bad!r}). Use nested dict form, e.g. '
            f'hi5=dict(cin_fn=dict(k=[best, low, high, step])).'
        )
    # Detect Optuna spec dicts at leaves before sc.flattendict over-descends
    # into them and produces confusing "leaf 'x.low'" errors.
    def _check(d, path=()):
        for k, v in d.items():
            if isinstance(v, dict):
                if v.keys() & {'low', 'high', 'guess', 'value', 'step'}:
                    raise ValueError(
                        f'hpv.Calibration: leaf {".".join((*path, k))!r} is a '
                        f'dict {v!r}; use the list form [best, low, high, step].'
                    )
                _check(v, (*path, k))
    _check(calib_pars)
    flat = sc.flattendict(calib_pars, sep='.')
    spec = {}
    for key, val in flat.items():
        if isinstance(val, (list, tuple)):
            if not 3 <= len(val) <= 4:
                raise ValueError(
                    f'hpv.Calibration: leaf {key!r} must be [best, low, high] '
                    f'or [best, low, high, step]; got {val!r}.'
                )
            spec[key] = dict(zip(_LIST_SPEC_KEYS, val))
        else:
            raise ValueError(
                f'hpv.Calibration: leaf {key!r} must be a list '
                f'[best, low, high, step]; got {type(val).__name__}.'
            )
    return spec


class Calibration(ss.Calibration):
    """HPVsim calibration. Delegates to ss.Calibration with HPV-aware defaults.

    Three entry points to specify the fit target:

    - ``datafiles=['data/foo_cancer_cases.csv', ...]``: list of long-format
      CSV paths (columns ``year,name,[age,sex,genotype,]value``). The loader
      dispatches on stratification columns: files with an ``age`` column go
      to a shared ``by_age`` analyzer named ``'calib_by_age'`` (attached to
      ``sim.pars.analyzers`` if not already there); ``by_age`` result keys
      then live under the scoped data key ``'calib_by_age.<name>'``. Ages
      and years are derived from the data itself. Genotype-stratified files
      (no ``age`` column) are reserved for ``'calib_by_genotype.<name>'``
      but not yet wired up (``NotImplementedError``).
    - ``data={'calib_by_age.cancers': df, ...}``: pre-built dict of
      't'-indexed DataFrames. Keys are ``'<analyzer_name>.<result_key>'``
      (scoped) or a bare result key that unambiguously matches a single
      ``by_age`` analyzer on the sim. The default eval_fn extracts each,
      aligns on ``(index, columns)``, and sums ``compute_gof`` across all
      cells, scaled by per-key ``weights``.
    - ``components=[...]`` or a custom ``eval_fn``: standard ss.Calibration
      paths, unchanged.

    ``calib_pars`` is a nested dict grouped by scope, with **list** leaves
    ``[best, low, high, step]`` (step optional). ``sc.flattendict`` collapses
    the scope tree to ``ss.Calibration``'s flat dotted-key form::

        calib_pars = dict(
            beta=[0.2, 0.1, 0.34, 0.02],
            network=dict(m_partners_casual=[0.5, 0.1, 0.9, 0.05]),
            hi5=dict(cin_fn=dict(k=[0.15, 0.1, 0.25, 0.02])),
        )

    Default build_fn is ``hpv.calibration.build_sim`` (``route_pars``); it
    accepts the flattened dotted keys and routes them.
    """

    def __init__(self, sim, calib_pars, *, data=None,
                 weights=None, gof_kwargs=None, build_fn=None,
                 eval_fn=None, eval_kw=None, **kwargs):
        if build_fn is None:
            build_fn = build_sim

        if calib_pars is not None:
            calib_pars = _prepare_calib_pars(calib_pars)

        # Give each calibration its own Optuna study database, in a temp dir, so
        # multiple hpv.Calibration runs in one session (or the test suite) do not
        # share or leak trials through a single database in the cwd. Callers can
        # still pass study_name/db_name explicitly (e.g. continue_db resume).
        if 'study_name' not in kwargs and 'db_name' not in kwargs:
            kwargs['study_name'] = 'hpvsim_calibration'
            kwargs['db_name'] = str(sc.path(tempfile.mkdtemp()) / 'hpvsim_calibration.db')

        # Default storage: JournalStorage (Optuna 4.x). SQLite (ss.Calibration
        # default) uses a global write lock that serializes every trial commit;
        # under ~32+ concurrent workers this deadlocks. JournalStorage is
        # Optuna's recommended backend for distributed / high-worker-count
        # optimization -- append-only per-process journals.
        if 'storage' not in kwargs:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
            journal_dir = sc.path(tempfile.mkdtemp())
            journal_path = journal_dir / 'hpvsim_calibration.log'
            kwargs['storage'] = JournalStorage(JournalFileBackend(str(journal_path)))

        # ss.Calibration defaults to reseed=True, which resamples rand_seed
        # from [0, 1_000_000] on every trial as if it were a calibrated par.
        # For HPV/cancer this is nearly always wrong: cancer is a rare event,
        # per-agent stochastic variance is large, and the resulting mismatch
        # surface is dominated by seed noise -- Optuna picks the luckiest
        # seed rather than the best parameters (see
        # https://github.com/starsimhub/hpvsim/pull/... for the pathology).
        # Override to False; callers who want per-trial reseed pass explicitly.
        kwargs.setdefault('reseed', False)

        if data is not None:
            if eval_fn is not None:
                raise ValueError(
                    'hpv.Calibration: pass either data= or eval_fn=, not both.')
            data = load_calib_data(data)
            _setup_analyzers(sim, data)
            eval_fn = default_eval_fn
            eval_kw = sc.mergedicts(eval_kw, dict(
                data=data,
                weights=weights or {},
                gof_kwargs=gof_kwargs or {},
            ))

        super().__init__(sim, calib_pars, build_fn=build_fn,
                         eval_fn=eval_fn, eval_kw=eval_kw, **kwargs)

    def remove_db(self):
        """Skip cleanup when ``self.run_args.storage`` isn't a string URL
        (parent's ``'sqlite' in storage`` check errors on a Storage object).
        Each hpv.Calibration gets its own tempdir per instance, so there's
        no cross-run pollution to worry about."""
        if isinstance(self.run_args.storage, str):
            return super().remove_db()
        return

    def shrink(self, n_results=100):
        """Return a lightweight ``sc.objdict`` with the top-``n_results``
        trials (by mismatch) + the metadata needed to replot / rebuild sims.

        Use for committing calibration results to source control: a full
        1000-5000-trial ``calib.obj`` is many MB (Optuna study, tempdir
        storage refs, per-trial state); the shrunken version drops all of
        that while remaining a drop-in for ``hpv.plot_calibration`` and
        ``utils.run_top_n``.

        Preserved attributes (accessed by plotting/rebuild code paths):
          - ``df``          : top-N rows of the trial DataFrame, sorted by
                              mismatch ascending.
          - ``best_pars``   : the best-fit par dict.
          - ``eval_kw``     : the eval kwargs (holds ``data`` for plotting).
          - ``calib_pars``  : the Optuna spec dict.
          - ``build_fn``    : how to reconstruct a sim from a par set.
          - ``build_kw``    : build_fn kwargs (defaults to ``{}``).
          - ``sim``         : base sim template.
        """
        cal = sc.objdict()
        n = min(n_results, len(self.df))
        cal.df = self.df.nsmallest(n, 'mismatch').reset_index(drop=True)
        cal.best_pars = self.best_pars
        cal.eval_kw = self.eval_kw
        cal.calib_pars = self.calib_pars
        cal.build_fn = self.build_fn
        cal.build_kw = self.build_kw or {}
        cal.sim = self.sim
        return cal

    def worker(self):
        """Run a single worker.

        Mirrors ``stisim.Calibration.worker``: wraps ``study.optimize`` in a
        try/except so a single worker's transient Optuna storage failure
        does not propagate through Optuna's own error handler and trip its
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


# ---------------------------------------------------------------------------
# Data-key scoping (standardized column-name conventions produced by
# hpv.data.loaders.load_calib_data) + analyzer-attachment helpers.
# ---------------------------------------------------------------------------

ALL_HPV = 'all_hpv'                # column prefix: pooled target
BY_GENOTYPE = 'by_genotype'        # column prefix: per-genotype distribution
ALL_HPV_BY_AGE = 'all_hpv_by_age'  # by_age analyzer name for pooled age-stratified targets

# Genotype-stratified target name -> (per-HPV result key, normalize).
# Stock-based (matches v2 `{state}_genotype_dist = n_{state}_by_genotype /
# totals`; see v2.2.6 hpvsim/sim.py:1112).
_GENOTYPE_DIST_MAP = {
    'precin_genotype_dist':    ('n_precin',    True),
    'cin_genotype_dist':       ('n_cin',       True),
    'cancerous_genotype_dist': ('n_cancerous', True),
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
        f'hpv.Calibration: unrecognized column format {col!r}; expected '
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
                f'hpv.Calibration: age bin labels not contiguous: {sorted(labels)}')
    return np.array(lows + [highs[-1]], dtype=float)


def _setup_analyzers(sim, data):
    """Inspect ``data`` columns and attach any analyzers the sim is missing.

    - Age-stratified pooled columns ``all_hpv.<name>.<bin>`` require a
      ``by_age`` named ``all_hpv_by_age`` with matching edges + years;
      created if absent, edges-validated if present.
    - Scalar ``all_hpv.<name>`` columns are read from HPVTotal at eval time
      (no attachment needed).
    - ``by_genotype.<name>.<g>`` columns compute via ``results_by_genotype``
      at eval time (no attachment needed).

    ``sim.pars.stop`` is auto-extended past the latest data year (+1 for
    partial-year margin) so callers don't redefine sim horizon.
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
                    f'hpv.Calibration: unknown by_genotype target {name!r} '
                    f'(from column {col!r}); known: {sorted(_GENOTYPE_DIST_MAP)}.')
        else:
            unknown.append(col)
    if unknown:
        raise ValueError(f'hpv.Calibration: unrecognized data columns: {unknown}')

    if age_result_names:
        # Group result names by their bin-label set. Targets that share bins
        # go on the same by_age analyzer; distinct bin schemes each get their
        # own (all_hpv_by_age, all_hpv_by_age_1, ...). Preserves the ordered
        # appearance of names in the data columns.
        groups = {}  # bin-set key -> list of result names in insertion order
        for name in age_result_names:
            key = frozenset(age_labels_by_result[name])
            groups.setdefault(key, []).append(name)
        years = sorted(float(y) for y in data.index)
        for i, (labels, names) in enumerate(groups.items()):
            edges = _edges_from_labels(labels)
            aname = ALL_HPV_BY_AGE if i == 0 else f'{ALL_HPV_BY_AGE}_{i}'
            _get_or_create_all_hpv_by_age(sim, names, years, edges, name=aname)

    if len(data):
        target_stop = int(float(data.index.max())) + 1
        cur_stop = sim.pars.get('stop')
        if cur_stop is None or float(cur_stop) < target_stop:
            sim.pars.stop = target_stop


def _get_or_create_all_hpv_by_age(sim, result_keys, years, edges, name=ALL_HPV_BY_AGE):
    """Return a ``by_age`` analyzer with the given ``name`` on ``sim`` (create +
    attach if absent; edges-validated if present). Multiple analyzers can
    coexist under distinct names (``all_hpv_by_age``, ``all_hpv_by_age_1``, ...)
    for targets that need different bin schemes."""
    existing = sim.pars.get('analyzers', []) or []
    for a in existing:
        if getattr(a, 'name', None) == name:
            if not np.array_equal(a.edges, edges):
                raise ValueError(
                    f'hpv.Calibration: existing {name} analyzer has '
                    f'edges {list(a.edges)}, incompatible with data-derived '
                    f'edges {list(edges)}.')
            return a
    ar = by_age(result_keys, years=years, edges=edges, name=name)
    sim.pars.analyzers = list(existing) + [ar]
    return ar


def _extract_columns(sim, data):
    """Return a DataFrame same-shape as ``data`` with sim-side values per
    column. Extraction is cached per underlying source (by_age analyzer,
    results_by_genotype table, HPVTotal result) so multi-subkey columns
    sharing a base don't re-extract.
    """
    tv_years = np.asarray(sim.timevec.years).astype(int)
    all_hpv_results = sim.results['all_hpv'] if 'all_hpv' in sim.results else None
    # Build a name -> analyzer map across all all_hpv_by_age* analyzers.
    # A single scheme uses just 'all_hpv_by_age'; multi-scheme calibrations
    # also have 'all_hpv_by_age_1', '_2', ... (one per distinct bin set).
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
                    f'hpv.Calibration eval: column {col!r} needs '
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


def _make_calib_sim_worker(pars, calib_pars, build_fn, build_kw, base_sim,
                           sim_kwargs, analyzers, extract_fn):
    """sc.parallelize worker for ``make_calib_sims``: apply this trial's
    ``pars`` to a deep-copied ``base_sim``, extend analyzers, run, then
    return the sim (or ``extract_fn(sim)`` if provided)."""
    spec = sc.dcp(calib_pars)
    for parname, s in spec.items():
        if parname in pars:
            s['value'] = pars[parname]
    sim = sc.dcp(base_sim)
    for k, v in (sim_kwargs or {}).items():
        setattr(sim.pars, k, v)
    if analyzers is not None:
        extra = analyzers() if callable(analyzers) else sc.dcp(analyzers)
        existing = list(sim.pars.get('analyzers', []) or [])
        sim.pars.analyzers = existing + list(extra)
    sim = build_fn(sim, calib_pars=spec, **build_kw)
    if isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]
    sim.run()
    return extract_fn(sim) if extract_fn is not None else sim


def make_calib_sims(calib, n=50, sim_kwargs=None, analyzers=None,
                    extract_fn=None, n_workers=None):
    """Rerun the top-``n`` trials from a ``hpv.Calibration`` in parallel.

    A ``hpv.Calibration`` stores per-trial ``mismatch`` + eval-column values,
    not per-trial sim results. To inspect any other result (e.g. the
    ``asr_cancer_incidence`` trajectory, or a custom by_age analyzer output),
    rerun the top-``n`` best-fit trials with this helper and read
    ``sim.results`` / ``sim.analyzers`` from the returned sims.

    Args:
        calib: hpv.Calibration (or the shrunk sc.objdict from
            ``calib.shrink()`` — both expose ``df``, ``calib_pars``,
            ``build_fn``, ``build_kw``, ``sim``).
        n (int): number of best-mismatch trials to rerun; clamped to
            ``len(calib.df)``.
        sim_kwargs (dict): overrides applied to ``sim.pars`` before
            ``build_fn`` (e.g. ``dict(stop=2045)`` to project past the
            calibration window).
        analyzers: zero-arg callable returning a fresh list of Analyzer
            instances (recommended so each subprocess gets its own state),
            OR a list of instances (deep-copied per worker). Appended to
            any analyzers the Calibration already attached to the base sim.
        extract_fn (callable): if provided, worker returns
            ``extract_fn(sim)`` instead of the full sim. Strongly recommended
            — see the note below.
        n_workers (int): defaults to ``min(n, cpu_count())``.

    Returns:
        list of length ``min(n, len(calib.df))``: run sims if ``extract_fn``
        is None, else the extract_fn outputs.

    Note:
        Full sims are memory-heavy (each holds people arrays, edge tables,
        results, analyzers). For ``n=50`` this can be many GB in the parent
        process after ``sc.parallelize`` pickles them back. Standard
        practice: pass an ``extract_fn`` that returns a small dict /
        DataFrame / array (use ``ss.Result.to_df`` or ``ss.Result.annualize``
        for time-series results), then commit that extracted output rather
        than the raw sims.
    """
    n = min(n, len(calib.df))
    top = calib.df.nsmallest(n, 'mismatch')
    # Optuna leaks 'rand_seed' when reseed=True; not a calibratable model par.
    par_cols = [c for c in top.columns if c not in ('index', 'mismatch', 'rand_seed')]
    par_sets = [{c: row[c] for c in par_cols} for _, row in top.iterrows()]
    if n_workers is None:
        n_workers = min(len(par_sets), sc.cpu_count())
    return sc.parallelize(
        _make_calib_sim_worker,
        iterkwargs=[{'pars': p} for p in par_sets],
        kwargs=dict(
            calib_pars=calib.calib_pars,
            build_fn=calib.build_fn,
            build_kw=calib.build_kw or {},
            base_sim=calib.sim,
            sim_kwargs=sim_kwargs,
            analyzers=analyzers,
            extract_fn=extract_fn,
        ),
        ncpus=n_workers, serial=False,
    )


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
                f'hpv.Calibration eval: sim produced no value for {col!r} at '
                f'years {missing}; expand sim.stop or update data.')
        mismatch = compute_gof(
            np.asarray(expected_col.values, dtype=float),
            np.asarray(actual_col.values, dtype=float),
            **gof_kwargs,
        )
        total += float(mismatch) * float(weights.get(col, 1.0))
    return total

