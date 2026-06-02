"""M01 short-summary builder (transmission-only, no precin/CIN/cancer).

M01 ports only the SIS transmission core; cancer-related metrics from
the M02/M03 short_summary do not apply here.
"""
import numpy as np


METRIC_KEYS_M01 = (
    'total HPV infections',
    'mean HPV prevalence (%)',
    'total population',
)


def build_summary_m01(sim, genotype='hpv16'):
    """Return the 3-entry M01 summary as a flat dict."""
    res = sim.results[genotype]
    pop_scale = float(getattr(sim.pars, 'pop_scale', 1.0) or 1.0)

    n_inf = float(np.asarray(res.new_infections).sum()) * pop_scale
    mean_prev_pct = 100.0 * float(np.asarray(res.prevalence).mean())
    total_pop = float(np.asarray(sim.results['n_alive'])[-1])

    return {
        'total HPV infections': n_inf,
        'mean HPV prevalence (%)': mean_prev_pct,
        'total population': total_pop,
    }
