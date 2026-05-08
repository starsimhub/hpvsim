"""M02 reproduction of methods-paper Fig 1 (natural history), HPV16-only.

Single-genotype port of ``plot_nh_simple`` from
https://github.com/hpvsim/hpvsim_methods_manuscript/blob/main/plot_fig1.py.

The manuscript script loops over four genotypes (hpv16, hpv18, hi5, ohr);
M02 ships HPV16 only, so this version produces the same 2x2 figure with a
single genotype. Adding genotypes in M03+ is a one-line edit to
``GENOTYPES``.

The math is identical to the manuscript: v3's ``GenotypePars.cancer_fn``
already carries the cin_fn keys (form, k, x_infl, ttc), so passing
``gp.cancer_fn`` to ``compute_severity`` produces the same values as the
manuscript's ``sc.mergedicts(cin_fn, cancer_fn)`` call.

Run with:
    python tests/regression/methods_fig1_hpv16.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
from scipy.stats import lognorm

from hpvsim.parameters import get_genotype_pars
from hpvsim.utils import compute_severity


GENOTYPES = ('hpv16',)
# Canonical four-genotype slot order; HPV16 keeps its index-0 colour even
# while M02 only iterates over hpv16. ``sc.gridcolors`` matches the
# manuscript's ``plot_fig1.py``.
_CANONICAL_GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')
_PALETTE = sc.gridcolors(len(_CANONICAL_GENOTYPES))
GENOTYPE_LABELS = {'hpv16': 'HPV16', 'hpv18': 'HPV18', 'hi5': 'Hi5', 'ohr': 'OHR'}
GENOTYPE_COLORS = {g: _PALETTE[i] for i, g in enumerate(_CANONICAL_GENOTYPES)}


def _lognorm_params(par1, par2):
    """(mean, std) -> scipy lognorm (shape, scale). Matches the manuscript helper."""
    mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))
    sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))
    return sigma, np.exp(mean)


def _dur_mean_std_years(dist):
    """Pull (mean, std) in years from an ``ss.lognorm_ex`` instance."""
    mean = dist.pars.mean
    std = dist.pars.std
    return float(getattr(mean, 'years', mean)), float(getattr(std, 'years', std))


def main():
    # Match the manuscript's ``ut.set_font(size=16)``. Libertinus Sans
    # falls back to the system default if not installed; the default font
    # (DejaVu Sans) is wider, so figsize is bumped from the manuscript's
    # (11, 9) to (12, 10) so titles still fit at fontsize=16.
    sc.options(fontsize=16)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    dt = 0.25
    this_precinx = np.arange(dt, 15 + dt, dt)
    years = np.arange(1, 16, 1)
    this_cinx = np.arange(dt, 30 + dt, dt)

    width = 0.2
    multiplier = 0
    for gi, gtype in enumerate(GENOTYPES):
        gp = get_genotype_pars(gtype)
        color = GENOTYPE_COLORS[gtype]
        label = GENOTYPE_LABELS[gtype]

        # Panel A: durations of infection (bar chart at integer years).
        precin_mean, precin_std = _dur_mean_std_years(gp.dur_precin)
        sigma, scale = _lognorm_params(precin_mean, precin_std)
        rv = lognorm(sigma, 0, scale)
        offset = width * multiplier
        axes[0].bar(years + offset - width / 3, rv.pdf(years),
                    color=color, lw=2, label=label, width=width)
        multiplier += 1

        # Panel B: probability of CIN by infection duration.
        dysp = compute_severity(this_precinx, rel_sev=None, pars=gp.cin_fn)
        axes[1].plot(this_precinx, dysp, color=color, lw=2, label=gtype.upper())

        # Panel C: distribution of CIN durations.
        cin_mean, cin_std = _dur_mean_std_years(gp.dur_cin)
        sigma, scale = _lognorm_params(cin_mean, cin_std)
        rv = lognorm(sigma, 0, scale)
        axes[2].plot(this_cinx, rv.pdf(this_cinx),
                     color=color, lw=2, label=label)

        # Panel D: probability of cancer by CIN duration. v3's cancer_fn
        # already carries cin_fn's keys, so passing it directly matches
        # the manuscript's mergedicts(cin_fn, cancer_fn) call.
        cancer = compute_severity(this_cinx, rel_sev=None, pars=gp.cancer_fn)
        axes[3].plot(this_cinx, cancer, color=color, lw=2, label=gtype.upper())

    axes[0].set_ylabel("")
    axes[0].grid()
    axes[0].set_xlabel("Duration of infection (years)")
    axes[0].set_title("(A) Probability of persistance")
    axes[0].legend(frameon=False)

    axes[1].set_ylabel("Probability of CIN")
    axes[1].set_xlabel("Duration of infection (years)")
    axes[1].set_title("(B) Probability that an infection of at least\n"
                      "X years will lead to high-grade lesions")
    axes[1].set_ylim([0, 1])
    axes[1].grid()

    axes[2].set_ylabel("")
    axes[2].grid()
    axes[2].set_xlabel("Duration of high-grade lesions (years)")
    axes[2].set_title("(C) Distribution of high-grade lesion duration")
    axes[2].legend(frameon=False)

    axes[3].set_ylim([0, 1])
    axes[3].grid()
    axes[3].set_xlabel("Duration of CIN (years)")
    axes[3].set_title("(D) Probability that a high-grade lesion of \n"
                      "at least X years will eventuate in cancer")

    fig.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    main()