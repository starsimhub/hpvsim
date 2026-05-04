"""M02 demo: run the 1-genotype HPV16 anchor and plot natural history trajectory.

Extends M01 demo with CIN prevalence and cancer incidence visualization.
Visible artifact of the M02 milestone (acceptance gate #4 per the M02
design spec).

Run with:
    python tests/regression/demo_anchor_hpv16.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

import hpvsim as hpv

# Make sibling anchor_hpv16.py importable when this script is invoked
# directly (python tests/regression/demo_anchor_hpv16.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_hpv16 import PARS  # noqa: E402


def main():
    sim = hpv.Sim(**PARS)
    sim.run()

    yearvec = sim.t.yearvec
    res = sim.results.hpv16
    mod = sim.diseases.hpv16

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Panel 1: HPV prevalence
    ax = axes[0]
    ax.plot(yearvec, res.prevalence * 100, color='C0',
            label='HPV16 prevalence')
    ax.set_ylabel('HPV16 prevalence (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: CIN prevalence
    ax = axes[1]
    n_cin_arr = res.n_cin if 'n_cin' in res else None
    if n_cin_arr is not None:
        n_alive_arr = sim.results['n_alive']
        cin_prev = (n_cin_arr / n_alive_arr) * 100
        ax.plot(yearvec, cin_prev, color='C1', label='CIN prevalence')
    ax.set_ylabel('CIN prevalence (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: cumulative cancers
    ax = axes[2]
    n_canc_arr = res.n_cancerous if 'n_cancerous' in res else None
    if n_canc_arr is not None:
        ax.plot(yearvec, n_canc_arr, color='C3', label='N currently cancerous')
    ax.set_ylabel('N cancerous')
    ax.set_xlabel('Year')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('M02 demo: HPV16 / CIN / cancer trajectory, Nigeria 1990-2060')
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    main()