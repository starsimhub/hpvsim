"""
Compare two sets of project validation baselines.

Loads baseline and current JSON results for each project, compares key time
series metrics, and prints a summary table with PASS/WARN/FAIL status.

Usage:
    python compare_baselines.py --baseline baselines/v2_main --current baselines/v2_branch
"""

import argparse
import numpy as np
import sciris as sc
import pylab as pl


# Metrics to compare
compare_keys = [
    'infections',
    'cancers',
    'cancer_deaths',
    'hpv_incidence',
    'cancer_incidence',
    'hpv_prevalence',
    'n_alive',
]

# Tolerances
PASS_RTOL = 0.15
PASS_ATOL = 5.0
WARN_RTOL = 0.30


def _is_rate(key):
    """Return True if the metric is a per-capita rate (use mean), False if a count (use sum)."""
    return key in ('hpv_incidence', 'cancer_incidence', 'hpv_prevalence', 'cancer_mortality')


def compute_diffs(baseline_vals, current_vals):
    """Compute RMSE and normalized RMSE between two time series.

    Returns:
        nrmse (float): RMSE divided by the baseline mean (0 = perfect match, lower is better)
        rmse (float): root mean squared error in original units
    """
    base = np.array(baseline_vals, dtype=float)
    curr = np.array(current_vals, dtype=float)

    rmse = float(np.sqrt(np.mean((base - curr) ** 2)))

    # Normalize by baseline mean to get a scale-free measure
    base_mean = np.mean(np.abs(base))
    if base_mean > 1e-10:
        nrmse = rmse / base_mean
    else:
        nrmse = 0.0

    return nrmse, rmse


def summarize_values(vals, key):
    """Return a scalar summary of a time series: mean for rates, sum for counts."""
    arr = np.array(vals, dtype=float)
    if _is_rate(key):
        return float(np.nanmean(arr))
    else:
        return float(np.nansum(arr))


def classify(nrmse, rmse):
    """Classify a comparison as PASS, WARN, or FAIL based on NRMSE."""
    if rmse <= PASS_ATOL or nrmse <= PASS_RTOL:
        return 'PASS'
    elif nrmse <= WARN_RTOL:
        return 'WARN'
    else:
        return 'FAIL'


def compare_baselines(baseline_dir, current_dir):
    """Compare all project baselines between two directories."""
    baseline_path = sc.path(baseline_dir)
    current_path = sc.path(current_dir)

    if not baseline_path.exists():
        print(f'ERROR: Baseline directory not found: {baseline_path}')
        return
    if not current_path.exists():
        print(f'ERROR: Current directory not found: {current_path}')
        return

    # Find matching project JSON files
    baseline_files = sorted(baseline_path.glob('*.json'))
    current_files = sorted(current_path.glob('*.json'))

    baseline_names = {f.stem: f for f in baseline_files}
    current_names = {f.stem: f for f in current_files}

    common_projects = sorted(set(baseline_names.keys()) & set(current_names.keys()))

    if not common_projects:
        print('No matching project files found between directories.')
        print(f'  Baseline: {sorted(baseline_names.keys())}')
        print(f'  Current:  {sorted(current_names.keys())}')
        return

    print(f'Comparing baselines:')
    print(f'  Baseline: {baseline_path}')
    print(f'  Current:  {current_path}')
    print(f'  Projects: {common_projects}')
    print(f'  Values:   sum for counts, mean for rates')
    print()

    # Table header
    header = (f'{"Project":<30} {"Metric":<22} '
              f'{"Baseline":>14} {"Current":>14} '
              f'{"NRMSE":>10} {"RMSE":>14} {"Status":>8}')
    separator = '-' * len(header)
    print(header)
    print(separator)

    n_pass = 0
    n_warn = 0
    n_fail = 0
    n_error = 0

    for project_name in common_projects:
        baseline_data = sc.loadjson(str(baseline_names[project_name]))
        current_data = sc.loadjson(str(current_names[project_name]))

        # Find matching run labels
        common_labels = sorted(set(baseline_data.keys()) & set(current_data.keys()))

        for run_label in common_labels:
            base_run = baseline_data[run_label]
            curr_run = current_data[run_label]

            # Skip runs with errors
            if 'error' in base_run or 'error' in curr_run:
                print(f'{run_label:<30} {"(error)":<22} {"":>14} {"":>14} {"":>10} {"":>14} {"ERROR":>8}')
                n_error += 1
                continue

            base_ts = base_run.get('time_series', {})
            curr_ts = curr_run.get('time_series', {})

            for key in compare_keys:
                if key not in base_ts or key not in curr_ts:
                    continue

                base_vals = base_ts[key]
                curr_vals = curr_ts[key]

                # Handle scalar vs list
                if not isinstance(base_vals, list):
                    base_vals = [base_vals]
                if not isinstance(curr_vals, list):
                    curr_vals = [curr_vals]

                # Skip if lengths differ
                if len(base_vals) != len(curr_vals):
                    print(f'{run_label:<30} {key:<22} {"":>14} {"":>14} {"len!=":>10} {"":>14} {"FAIL":>8}')
                    n_fail += 1
                    continue

                nrmse, rmse = compute_diffs(base_vals, curr_vals)
                status = classify(nrmse, rmse)

                if status == 'PASS':
                    n_pass += 1
                elif status == 'WARN':
                    n_warn += 1
                else:
                    n_fail += 1

                # Summarize raw values: mean for rates, sum for counts
                base_summary = summarize_values(base_vals, key)
                curr_summary = summarize_values(curr_vals, key)

                # Format: use scientific notation for large counts, fixed for rates
                if _is_rate(key):
                    base_str = f'{base_summary:>14.4f}'
                    curr_str = f'{curr_summary:>14.4f}'
                else:
                    base_str = f'{base_summary:>14.0f}'
                    curr_str = f'{curr_summary:>14.0f}'

                print(f'{run_label:<30} {key:<22} {base_str} {curr_str} {nrmse:>10.4f} {rmse:>14.2f} {status:>8}')

    # Summary
    print(separator)
    n_total = n_pass + n_warn + n_fail + n_error
    print(f'\nSummary: {n_total} comparisons')
    print(f'  PASS:  {n_pass}')
    print(f'  WARN:  {n_warn}')
    print(f'  FAIL:  {n_fail}')
    print(f'  ERROR: {n_error}')

    if n_fail > 0:
        print(f'\nRESULT: FAIL ({n_fail} failures)')
    elif n_warn > 0:
        print(f'\nRESULT: WARN ({n_warn} warnings)')
    else:
        print(f'\nRESULT: PASS (all comparisons within tolerance)')


def plot_baselines(baseline_dir, current_dir, seed=0, baseline_label='v2_main', current_label='v2_branch'):
    """Plot time series comparisons between two baseline sets.

    Creates one figure per project with subplots for each metric, showing
    baseline vs current time series overlaid.

    Args:
        baseline_dir (str): path to baseline JSON directory
        current_dir (str): path to current JSON directory
        seed (int): which seed to plot (0, 1, or 2)
        baseline_label (str): label for the baseline series
        current_label (str): label for the current series
    """
    baseline_path = sc.path(baseline_dir)
    current_path = sc.path(current_dir)

    baseline_files = sorted(baseline_path.glob('*.json'))
    current_files = sorted(current_path.glob('*.json'))
    baseline_names = {f.stem: f for f in baseline_files}
    current_names = {f.stem: f for f in current_files}
    common_projects = sorted(set(baseline_names.keys()) & set(current_names.keys()))

    for project_name in common_projects:
        baseline_data = sc.loadjson(str(baseline_names[project_name]))
        current_data = sc.loadjson(str(current_names[project_name]))

        run_label = f'{project_name}_seed{seed}'
        if run_label not in baseline_data or run_label not in current_data:
            print(f'Skipping {project_name}: seed {seed} not found')
            continue

        base_run = baseline_data[run_label]
        curr_run = current_data[run_label]
        if 'error' in base_run or 'error' in curr_run:
            print(f'Skipping {project_name}: run has errors')
            continue

        base_ts = base_run.get('time_series', {})
        curr_ts = curr_run.get('time_series', {})
        base_years = np.array(base_run.get('year', []))
        curr_years = np.array(curr_run.get('year', []))

        # Filter to metrics that exist in both
        keys = [k for k in compare_keys if k in base_ts and k in curr_ts]
        n_keys = len(keys)
        if n_keys == 0:
            continue

        ncols = min(n_keys, 4)
        nrows = int(np.ceil(n_keys / ncols))
        fig, axes = pl.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()
        fig.suptitle(f'{project_name}  (seed {seed})', fontsize=14, fontweight='bold')

        for i, key in enumerate(keys):
            ax = axes[i]
            base_vals = np.array(base_ts[key], dtype=float)
            curr_vals = np.array(curr_ts[key], dtype=float)

            ax.plot(base_years[:len(base_vals)], base_vals, label=baseline_label, lw=1.5, alpha=0.8)
            ax.plot(curr_years[:len(curr_vals)], curr_vals, label=current_label, lw=1.5, alpha=0.8, ls='--')

            # Compute status for the title
            if len(base_vals) == len(curr_vals):
                nrmse, rmse = compute_diffs(base_vals, curr_vals)
                status = classify(nrmse, rmse)
                ax.set_title(f'{key}  [{status}, NRMSE={nrmse:.2f}]', fontsize=11)
            else:
                ax.set_title(key, fontsize=11)

            ax.set_xlabel('Year')
            ax.legend(fontsize=8)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 4))

        # Hide unused axes
        for j in range(n_keys, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()

    pl.show()


if __name__ == '__main__':
    basedir = sc.thisdir(__file__)

    parser = argparse.ArgumentParser(description='Compare two sets of project validation baselines')
    parser.add_argument('--baseline', type=str, default=str(sc.path(basedir) / 'baselines' / 'v2_main'),
                        help='Path to baseline directory (default: baselines/v2_main)')
    parser.add_argument('--current', type=str, default=str(sc.path(basedir) / 'baselines' / 'v2_branch'),
                        help='Path to current directory (default: baselines/v2_branch)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--seed', type=int, default=0, help='Which seed to plot (default: 0)')
    args = parser.parse_args()

    compare_baselines(args.baseline, args.current)

    if not args.no_plot:
        baseline_label = sc.path(args.baseline).name
        current_label = sc.path(args.current).name
        plot_baselines(args.baseline, args.current, seed=args.seed,
                       baseline_label=baseline_label, current_label=current_label)
