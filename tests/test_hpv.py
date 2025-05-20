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
import matplotlib as pl
import matplotlib.ticker as mtick


def test_hpv(debug=False):
    n_agents = [5e4, 1e3][debug]
    start = 1970
    stop = 2025
    total_pop = {1970: 5.203e6, 1980: 7.05e6, 1990: 9980999, 2000: 11.83e6}[start]
    ppl = ss.People(
        n_agents,
    )
    hpv_pars = dict(
        beta=dict(structuredsexual=[1, 1], maternal=[0, 0]),
    )
    hpv16 = hpv.HPV(genotype="16", pars=hpv_pars)
    hpv18 = hpv.HPV(genotype="18", pars=hpv_pars)
    maternal = ss.MaternalNet(unit="month")
    structuredsexual = sti.FastStructuredSexual(
        unit="month",
        prop_f0=0.8,
        prop_f2=0.05,
        prop_m0=0.65,
        f1_conc=0.05,
        f2_conc=0.25,
        m1_conc=0.15,
        m2_conc=0.3,
        p_pair_form=0.6,
    )
    nets = ss.ndict(structuredsexual, maternal)
    dem = []
    dis = [hpv16, hpv18]
    con = [hpv.genotype_connector(genotypes=["hpv16", "hpv18"])]
    an = []

    pregnancy_pars = {
        "fertility_rate": pd.read_csv("test_data/asfr_zim.csv"),
        "unit": "month",
        "dt": 1,
    }
    pregnancy = ss.Pregnancy(pars=pregnancy_pars)
    death_rates = {"death_rate": pd.read_csv("test_data/deaths_zim.csv")}
    death = ss.Deaths(death_rates)

    dem += [pregnancy, death]

    # Make sim
    sim_args = dict(
        unit="month",
        dt=1,
        start=ss.date(start),
        stop=ss.date(stop),
        people=ppl,
        total_pop=total_pop,
        diseases=dis,
        networks=nets,
        demographics=dem,
        connectors=con,
        analyzers=an,
    )

    # RUN
    sim = ss.Sim(**sim_args)
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
        y = sim.results.genotype_connector[to_plot].values[15 * 12 :]
        x = sim.results.genotype_connector["timevec"][15 * 12 :]
        ax.plot(x, y)
        ax.set_title(to_plot)
        ax.set_ylim(bottom=0)
    for ax in axes:
        sc.SIticks(ax=ax)

    fig.tight_layout()
    pl.savefig(f"{figdir}/summary.png", dpi=100)

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
    pl.savefig(f"{figdir}/hpv_prevalence.png", dpi=100)

    print("done")


def test_screening(debug=False):
    n_agents = [1e4, 1e3][debug]
    start = 1990
    stop = 2030
    total_pop = {1970: 5.203e6, 1980: 7.05e6, 1990: 9980999, 2000: 11.83e6}[start]
    ppl = ss.People(
        n_agents,
    )
    hpv_pars = dict(
        beta=dict(structuredsexual=[1, 1], maternal=[0, 0]),
    )
    hpv16 = hpv.HPV(genotype="16", pars=hpv_pars)
    hpv18 = hpv.HPV(genotype="18", pars=hpv_pars)
    maternal = ss.MaternalNet(unit="month")
    structuredsexual = sti.FastStructuredSexual(
        unit="month",
        prop_f0=0.8,
        prop_f2=0.05,
        prop_m0=0.65,
        f1_conc=0.05,
        f2_conc=0.25,
        m1_conc=0.15,
        m2_conc=0.3,
        p_pair_form=0.6,
    )
    nets = ss.ndict(structuredsexual, maternal)
    dem = []
    dis = [hpv16, hpv18]
    super_connector = hpv.genotype_connector(genotypes=["hpv16", "hpv18"])
    con = [super_connector]
    an = []

    pregnancy_pars = {
        "fertility_rate": pd.read_csv("test_data/asfr_zim.csv"),
        "unit": "month",
        "dt": 1,
    }
    pregnancy = ss.Pregnancy(pars=pregnancy_pars)
    death_rates = {"death_rate": pd.read_csv("test_data/deaths_zim.csv")}
    death = ss.Deaths(death_rates)

    dem += [pregnancy, death]
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

        # Make sim
        sim_args = dict(
            unit="month",
            dt=1,
            start=ss.date(start),
            stop=ss.date(stop),
            people=ppl,
            total_pop=total_pop,
            diseases=dis,
            networks=nets,
            demographics=dem,
            connectors=con,
            analyzers=an,
            interventions=scen_int,
            label=scen_label,
        )

        # RUN
        sim = ss.Sim(**sim_args)
        sims.append(sim)

    ss.parallel(sims)

    return


def test_performance(debug=False):
    n_agents = [5e3, 1e3][debug]
    start = 1970
    stop = 2030
    total_pop = {1970: 5.203e6, 1980: 7.05e6, 1990: 9980999, 2000: 11.83e6}[start]
    ppl = ss.People(
        n_agents,
    )
    hpv_pars = dict(
        beta=dict(structuredsexual=[1, 1], maternal=[0, 0]),
    )
    hpv16 = hpv.HPV(genotype="16", pars=hpv_pars)
    hpv18 = hpv.HPV(genotype="18", pars=hpv_pars)
    maternal = ss.MaternalNet(unit="month")
    structuredsexual = sti.FastStructuredSexual(
        unit="month",
        dt=3,
        prop_f0=0.8,
        prop_f2=0.05,
        prop_m0=0.65,
        f1_conc=0.05,
        f2_conc=0.25,
        m1_conc=0.15,
        m2_conc=0.3,
        p_pair_form=0.6,
    )
    nets = ss.ndict(structuredsexual, maternal)
    dem = []
    dis = [hpv16, hpv18]
    con = [hpv.genotype_connector(genotypes=["hpv16", "hpv18"])]
    an = []

    pregnancy_pars = {
        "fertility_rate": pd.read_csv("test_data/asfr_zim.csv"),
        "unit": "month",
        "dt": 3,
    }
    pregnancy = ss.Pregnancy(pars=pregnancy_pars)
    death_rates = {"death_rate": pd.read_csv("test_data/deaths_zim.csv")}
    death = ss.Deaths(death_rates)

    dem += [pregnancy, death]

    # Make sim
    sim_args = dict(
        unit="month",
        dt=3,
        start=ss.date(start),
        stop=ss.date(stop),
        people=ppl,
        total_pop=total_pop,
        diseases=dis,
        networks=nets,
        demographics=dem,
        connectors=con,
        analyzers=an,
    )

    # RUN
    sim = ss.Sim(**sim_args)
    sim.run()


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
        init_prev_data=pd.read_csv("test_data/init_prev_hiv.csv"),
    )
    return hiv


def test_hiv_hpv(debug=False):
    n_agents = [1e4, 1e3][debug]
    start = 1970
    stop = 2030
    total_pop = {1970: 5.203e6, 1980: 7.05e6, 1990: 9980999, 2000: 11.83e6}[start]
    ppl = ss.People(
        n_agents,
    )
    hpv_pars = dict(
        beta=dict(structuredsexual=[1, 1], maternal=[0, 0]),
    )
    hpv16 = hpv.HPV(genotype="16", pars=hpv_pars)
    hpv18 = hpv.HPV(genotype="18", pars=hpv_pars)
    hiv = make_hiv()
    maternal = ss.MaternalNet(unit="month")
    structuredsexual = sti.FastStructuredSexual(
        unit="month",
        dt=3,
        prop_f0=0.8,
        prop_f2=0.05,
        prop_m0=0.65,
        f1_conc=0.05,
        f2_conc=0.25,
        m1_conc=0.15,
        m2_conc=0.3,
        p_pair_form=0.6,
    )
    nets = ss.ndict(structuredsexual, maternal)
    dem = []
    dis = [hpv16, hpv18, hiv]
    hpv_super_connector = hpv.genotype_connector(genotypes=["hpv16", "hpv18"])
    con = [hpv_super_connector, hpv.hpv_hiv_connector(hiv=hiv, hpv=hpv_super_connector)]
    hpv_hiv_analyzer = hpv.hiv_hpv_results(hiv=hiv, hpv=hpv_super_connector)
    an = [hpv_hiv_analyzer]

    pregnancy_pars = {
        "fertility_rate": pd.read_csv("test_data/asfr_zim.csv"),
        "unit": "month",
        "dt": 3,
    }
    pregnancy = ss.Pregnancy(pars=pregnancy_pars)
    death_rates = {"death_rate": pd.read_csv("test_data/deaths_zim.csv")}
    death = ss.Deaths(death_rates)

    dem += [pregnancy, death]

    # Make sim
    sim_args = dict(
        unit="month",
        dt=3,
        start=ss.date(start),
        stop=ss.date(stop),
        people=ppl,
        total_pop=total_pop,
        diseases=dis,
        networks=nets,
        demographics=dem,
        connectors=con,
        analyzers=an,
    )

    # RUN
    sim = ss.Sim(**sim_args)
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
    pl.savefig(f"{figdir}/hpv_hiv_prevalence.png", dpi=100)


if __name__ == "__main__":

    T = sc.tic()
    # test_hpv()

    # test_hiv_hpv()

    # test_performance()

    test_screening()

    sc.toc(T)
    print("Done.")
