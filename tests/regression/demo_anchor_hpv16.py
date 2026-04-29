"""M01 demo: run the 1-genotype HPV16 anchor and plot aggregate prevalence.

Visible artifact of the M01 milestone (acceptance gate #4 per the M01
design spec).

Run with:
    python tests/regression/demo_anchor_hpv16.py
"""

import matplotlib.pyplot as plt

import hpvsim as hpv
from regression.anchor_hpv16 import PARS


def main():
    sim = hpv.Sim(**PARS)
    sim.run()

    yearvec = sim.t.yearvec
    prev = sim.results.hpv16.prevalence

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(yearvec, prev * 100, label='HPV16 prevalence (M01 anchor)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('M01 demo: HPV16 prevalence trajectory, Nigeria 1990-2060')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    main()