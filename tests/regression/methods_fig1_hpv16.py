"""M02 reproduction of methods-paper Fig 1 (natural history), HPV16-only.

Validates the M02 single-genotype natural-history pipeline by visualising
the per-genotype duration distributions and progression-probability curves
against the parameters in ``hpvsim.parameters.GenotypePars``, plus the
final-outcome shares observed in a single-genotype anchor sim.

Adapted from the v2 ``tests/devtests/test_new_progs.py:make_fig1`` to fit
M02's scope:
  - Single genotype (HPV16) instead of {hpv16, hpv18, hi5, ohr}
  - Single CIN compartment (no CIN1 / CIN2 / CIN3 stratification or
    peak-severity colour bands)
  - Final-outcome panel computed from the M02 sim (not a v2-specific helper)

Multi-genotype panels (and CIN-grade stratification) return when those
features land in M03+; the script is structured so adding genotypes is a
one-line edit to ``GENOTYPES`` below.

Run with:
    python tests/regression/methods_fig1_hpv16.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

import hpvsim as hpv
from hpvsim.hpv import _compute_severity
from hpvsim.parameters import get_genotype_pars

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_hpv16 import PARS  # noqa: E402


GENOTYPES = ('hpv16',)
GENOTYPE_COLORS = {'hpv16': 'C0', 'hpv18': 'C1', 'hi5': 'C2', 'ohr': 'C3'}


def _lognorm_pdf(x, mean, std):
    """PDF of a lognormal parameterised by (mean, std) — matches ``ss.lognorm_ex``."""
    var = std ** 2
    sigma2 = np.log(1.0 + var / mean ** 2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    return lognorm.pdf(x, s=sigma, scale=np.exp(mu))


def _dur_mean_std_years(dist):
    """Pull (mean, std) in years from an ``ss.lognorm_ex`` instance."""
    mean = dist.pars.mean
    std = dist.pars.std
    return float(getattr(mean, 'years', mean)), float(getattr(std, 'years', std))


def _outcome_shares(sim):
    """Worst-progression shares for *currently-alive* ever-infected women.

    Each woman is classified by the worst stage she's reached so far. We
    restrict to alive women because dead women have their compartment
    flags reset in ``step_die`` and the schedule timestamps don't tell
    us whether the transition fired before background mortality. Restricting
    to survivors biases against late-life cancer (women who died of cancer
    are excluded), so the cancer share here is a *lower bound*.
    """
    mod = sim.diseases.hpv16
    people = sim.people
    alive = people.alive.raw.astype(bool)
    female = people.female.raw.astype(bool)
    ever_inf = (~mod.ti_first_infection.isnan.raw) & female & alive

    # Active compartment flags are valid for alive agents.
    in_cin = mod.cin.raw.astype(bool) & alive
    in_cancer = mod.cancerous.raw.astype(bool) & alive
    # "Reached CIN" includes alive agents currently in CIN, plus alive agents
    # currently in cancer (they passed through CIN), plus alive agents who
    # cleared from CIN (ti_cin set & ti_clearance set & ti_clearance was
    # after ti_cin).
    cleared_from_cin = (
        ever_inf
        & (~mod.ti_cin.isnan.raw)
        & (~mod.ti_clearance.isnan.raw)
        & (mod.ti_clearance.raw >= mod.ti_cin.raw)
    )
    ever_cin = ever_inf & (in_cin | in_cancer | cleared_from_cin)
    ever_cancer = ever_inf & in_cancer

    n = int(ever_inf.sum())
    if n == 0:
        return dict(none=0., cin=0., cancer=0.)
    return dict(
        none=float((ever_inf & ~ever_cin).sum()) / n,
        cin=float((ever_cin & ~ever_cancer).sum()) / n,
        cancer=float(ever_cancer.sum()) / n,
    )


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Top-left: pre-CIN duration PDFs ---
    # x starts at 0.5 — the lognormal mode for these high-variance fits is
    # very close to 0, so plotting from 0.01 makes the rest of the body
    # invisible in print scale.
    ax = axes[0, 0]
    x_precin = np.linspace(0.5, 30, 400)
    for g in GENOTYPES:
        gp = get_genotype_pars(g)
        mean, std = _dur_mean_std_years(gp.dur_precin)
        ax.plot(x_precin, _lognorm_pdf(x_precin, mean, std),
                color=GENOTYPE_COLORS[g], lw=2, label=g.upper())
    ax.set_xlabel('Pre-CIN duration (years)')
    ax.set_title('Distribution of infection durations\nprior to CIN or clearance')
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Top-middle: P(CIN | duration) ---
    ax = axes[0, 1]
    for g in GENOTYPES:
        gp = get_genotype_pars(g)
        p_cin = _compute_severity(x_precin, rel_sev=None, pars=gp.cin_fn)
        ax.plot(x_precin, p_cin, color=GENOTYPE_COLORS[g], lw=2, label=g.upper())
    ax.set_xlabel('Pre-CIN duration (years)')
    ax.set_ylabel('P(progress to CIN)')
    ax.set_title('Probability of progressing to CIN\nby pre-CIN duration')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Top-right: CIN duration PDFs ---
    ax = axes[0, 2]
    x_cin = np.linspace(0.5, 30, 400)
    for g in GENOTYPES:
        gp = get_genotype_pars(g)
        mean, std = _dur_mean_std_years(gp.dur_cin)
        ax.plot(x_cin, _lognorm_pdf(x_cin, mean, std),
                color=GENOTYPE_COLORS[g], lw=2, label=g.upper())
    ax.set_xlabel('CIN duration (years)')
    ax.set_title('Distribution of CIN durations\nprior to cancer or clearance')
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Bottom-left: P(cancer | CIN duration) ---
    ax = axes[1, 0]
    for g in GENOTYPES:
        gp = get_genotype_pars(g)
        p_cancer = _compute_severity(x_cin, rel_sev=None, pars=gp.cancer_fn)
        ax.plot(x_cin, p_cancer, color=GENOTYPE_COLORS[g], lw=2, label=g.upper())
    ax.set_xlabel('CIN duration (years)')
    ax.set_ylabel('P(progress to cancer)')
    ax.set_title('Probability of progressing to cancer\nby CIN duration')
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Bottom-middle: simulated worst-progression shares ---
    ax = axes[1, 1]
    sim = hpv.Sim(**PARS)
    sim.run()
    shares = _outcome_shares(sim)
    labels = ('None\n(precin clear)', 'CIN\n(no cancer)', 'Cancer')
    keys = ('none', 'cin', 'cancer')
    colors = ('#9ec5e8', '#f4a460', '#d97070')
    ax.bar(labels, [shares[k] for k in keys], color=colors, edgecolor='k')
    ax.set_ylabel('Share of ever-infected women')
    ax.set_title(f'Eventual outcomes for women\n(HPV16 anchor sim, n={sim.pars.n_agents})')
    ax.grid(axis='y', alpha=0.3)
    for i, k in enumerate(keys):
        ax.text(i, shares[k] + 0.01, f'{shares[k]:.1%}', ha='center', fontsize=10)

    # --- Bottom-right: legend / scope notes ---
    ax = axes[1, 2]
    ax.axis('off')
    notes = (
        'M02 scope notes\n'
        '─────────────────────\n'
        'Single genotype: HPV16 only.\n'
        'Genotypes hpv18 / hi5 / ohr land in M03+.\n\n'
        'Single CIN compartment — no CIN1 / CIN2 /\n'
        'CIN3 stratification or peak-severity bands.\n\n'
        'Outcome shares restricted to currently-alive\n'
        'ever-infected women (post-mortem flags are\n'
        'reset by step_die). Cancer share is a lower\n'
        'bound — women who died of cancer are excluded.\n\n'
        f'Anchor: {PARS["location"]}, '
        f'{int(getattr(PARS["start"], "years", PARS["start"]))}–'
        f'{int(getattr(PARS["stop"], "years", PARS["stop"]))}, '
        f'seed={PARS["rand_seed"]}.\n'
    )
    ax.text(0.0, 1.0, notes, ha='left', va='top', family='monospace', fontsize=11)

    fig.suptitle('M02 natural-history confirmation (Fig 1 reproduction, HPV16)',
                 fontsize=14)
    fig.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    main()