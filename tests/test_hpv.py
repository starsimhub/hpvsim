"""
Simple sim tests
"""

# Imports
import sciris as sc
import starsim as ss
import hpvsim as hpv
import stisim as sti
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import os


def test_hpv(debug=True):

    sim = hpv.Sim()
    sim.run()

    fig, axes = pl.subplots(nrows=3, figsize=(12, 7))
    to_plot_by_genotype = ["n_infected", "n_cin"]
    to_plot_total = ["cancer_incidence"]
    i = 0
    for resname in to_plot_by_genotype:
        for genotype in ["hpv16", "hpv18"]:
            ax = axes[i]
            y = sim.results[genotype][resname].values[15 * 12 :]
            x = sim.results[genotype]["timevec"][15 * 12 :]
            ax.plot(x, y, label=genotype)
        ax.set_title(resname)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        i += 1
    for to_plot in to_plot_total:
        ax = axes[i]
        y = sim.results.total[to_plot].values[15 * 12 :]
        x = sim.results.total["timevec"][15 * 12 :]
        ax.plot(x, y)
        ax.set_title(to_plot)
        ax.set_ylim(bottom=0)
    for ax in axes:
        sc.SIticks(ax=ax)

    fig.tight_layout()

    fig, axes = pl.subplots(nrows=2, figsize=(10, 10))
    ages = [15, 25, 35, 45, 55]
    for i, age in enumerate(ages):
        resname = f"prevalence_{age}_{age+9}"
        y = sim.results.genotype_connector[resname].values
        axes[0].bar(i, y[-1])

    axes[0].set_xticks(range(len(ages)))
    axes[0].set_xticklabels([f"{age}-{age+9}" for age in ages])
    axes[0].set_ylabel("HPV Prevalence")
    sc.SIticks(ax=axes[0])
    for genotype in ["hpv16", "hpv18"]:
        y = sim.results[genotype]["prevalence"].values
        x = sim.results[genotype]["timevec"]
        axes[1].plot(x, y, label=genotype)
    axes[1].legend(frameon=False)
    axes[1].set_ylabel("HPV Prevalence")
    fig.tight_layout()
    pl.show()

    print("done")


def test_screening(debug=False):

    treat_eligible = lambda sim: sim.interventions["screen"].screen_results["positive"]

    scenarios = {
        "no_screen": [],
        "screen": [
            hpv.screen(
                modules=[super_connector],
                start_year=2000,
                pars=dict(p_seek_care=ss.bernoulli(p=0.5)),
                label="screen",
            ),
            hpv.treat(
                modules=[hpv16, hpv18],
                start_year=2000,
                eligibility=treat_eligible,
                label="treat",
            ),
        ],
    }
    sims = []
    for scen_label, scen_int in scenarios.items():
        sim = hpv.Sim(interventions=scen_int, label=scen_label)
        sims.append(sim)

    ss.parallel(sims)

    return sims


def make_hiv(hiv_pars=None):
    """Make HIV arguments for sim"""
    hiv_pars = sc.mergedicts(
        dict(
            beta_m2f=0.01,  # pulled from calibration
            eff_condom=0.95,
            rel_init_prev=5.0,
        ),
        hiv_pars,
    )
    hiv = sti.HIV(
        pars=hiv_pars,
        init_prev_data=pd.read_csv(
            os.path.join(hpv.root, "tests", "test_data", "init_prev_hiv.csv")
        ),
    )
    return hiv


def test_hiv_hpv(debug=False):
    hiv = make_hiv()

    # Make sim
    sim = hpv.Sim(diseases=hiv, connectors=con, analyzers=an)
    sim.run()

    colors = sc.gridcolors(2)
    hiv_hpv_results = sim.results.hiv_hpv_results
    fig, axes = pl.subplots(nrows=2, figsize=(10, 10))
    ages = [15, 25, 35, 45, 55]
    for i, age in enumerate(ages):
        for j, hiv_res in enumerate(["hiv_neg", "hiv_pos"]):
            resname = f"hpv_prevalence_{hiv_res}_{age}_{age+9}"
            y = hiv_hpv_results[resname].values
            if i == 0:
                axes[0].scatter(i, y[-1], color=colors[j], label=hiv_res)
            else:
                axes[0].scatter(i, y[-1], color=colors[j])

    axes[0].set_xticks(range(len(ages)))
    axes[0].set_xticklabels([f"{age}-{age+9}" for age in ages])
    axes[0].set_ylabel("HPV Prevalence")
    axes[0].legend(frameon=False)
    sc.SIticks(ax=axes[0])
    for j, hiv_res in enumerate(["hiv_neg", "hiv_pos"]):
        resname = f"hpv_prevalence_{hiv_res}"
        y = hiv_hpv_results[resname].values
        x = hiv_hpv_results["timevec"]
        axes[1].plot(x, y, label=hiv_res)
    axes[1].legend(frameon=False)
    axes[1].set_ylabel("HPV Prevalence")
    fig.tight_layout()

    pl.show()


if __name__ == "__main__":

    T = sc.tic()
    test_hpv()
    # test_hiv_hpv()
    # test_screening()

    sc.toc(T)
    print("Done.")
