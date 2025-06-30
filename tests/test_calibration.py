import starsim as ss
import sciris as sc
import stisim as sti
import hpvsim as hpv
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import numpy as np
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

datadir = os.path.join(hpv.root, "tests", "test_data")


# Run settings
debug = True  # If True, this will do smaller runs that can be run locally for debugging
n_trials = [10, 2][debug]  # How many trials to run for calibration
n_workers = [None, 1][debug]  # How many cores to use
# storage = ["mysql://hpvsim_user@localhost/hpvsim_db", None][debug]  # Storage for calibrations
storage = None
continue_db = False  # Whether to continue from an existing database (if True, will load the previous state of the db)


def make_sim():
    """
    Make an HPV simulation using Starsim

    Returns
    -------
    Sim
        A single simulation that has been configured, but not initialized or run.
    """
    debug = False
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

    maternal = ss.MaternalNet(unit="month")
    structuredsexual = sti.FastStructuredSexual(
        unit="month",
        dt=1,
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
    hpv_super_connector = hpv.genotype_connector(genotypes=["hpv16", "hpv18"])
    con = [hpv_super_connector]
    an = []

    pregnancy_pars = {
        "fertility_rate": pd.read_csv(f"{datadir}/asfr_zim.csv"),
        "unit": "month",
        "dt": 1,
    }
    pregnancy = ss.Pregnancy(pars=pregnancy_pars)
    death_rates = {"death_rate": pd.read_csv(f"{datadir}/deaths_zim.csv")}
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
        verbose=-1,
    )

    # RUN
    sim = ss.Sim(**sim_args)
    return sim


def modify_sim(sim, calib_pars, rand_seed=0):
    """
    Modify the given simulation with the calibration parameters and random seed.

    Parameters
    ----------
    sim : Sim
        The simulation to modify.
    calib_pars : dict
        The calibration parameters to apply, note that the parameter values to use are stored in "value".
    rand_seed : int
        The random seed to use for the simulation.

    Returns
    -------
    Sim
        The modified simulation.
    """

    diseases = ss.ndict(sim.pars.diseases)
    nets = ss.ndict(sim.pars.networks)

    for k, pars in calib_pars.items():
        v = pars["value"]

        mod = k.split("_")[0]  # Get the module name from the key
        if mod in diseases:
            hpv_mod = diseases[mod]  # Get the disease object
            k = k.replace(f"{mod}_", "")  # As above
            if "beta" in k:
                hpv_mod.pars[k] = ss.rate_prob(
                    v, unit="month"
                )  # Interpret beta as a rate per month to be automatically converted to a probability of transmission per time step.
            elif "cin_fn" in k:
                k = k.replace("cin_fn_", "")  # As above
                hpv_mod.pars["cin_fn"][k] = v
            else:
                hpv_mod.pars[k] = v
        elif mod == "nw":  # Network parameters
            nw = nets[
                "structuredsexual"
            ]  # Assuming we are modifying the structuredsexual network here
            k = k.replace(f"{mod}_", "")  # As above
            if "pair_form" in k:
                nw.pars[k].set(v)
            else:
                nw.pars[k] = v
        else:
            raise ValueError(f"Unknown module {mod} for parameter {k}.")

    sim.pars["rand_seed"] = rand_seed

    return sim


def make_components():

    # Start with cancers by age
    # Prepare the data
    calib_data_cancers_by_age = pd.read_csv(f"{datadir}/calib_data_cancers_by_age.csv")
    calib_data_cancers_by_age["t"] = pd.to_datetime(
        calib_data_cancers_by_age["t"], format="%Y"
    )  # Ensure time is in datetime format
    calib_data_cancers_by_age["t1"] = pd.to_datetime(
        calib_data_cancers_by_age["t1"], format="%Y"
    )  # Ensure time is in datetime format

    cancer_data = calib_data_cancers_by_age.set_index(["t", "t1", "age_group"])

    def extract_cancer_data(sim):
        cancer_ages = ["20_34", "35_49", "50_64", "65_79"]
        dfs = sc.autolist()
        for age in cancer_ages:
            df = pd.DataFrame(
                {
                    "x": sim.results.genotype_connector[
                        f"cancers_{age}"
                    ],  # Number of events
                    "n": sim.results.genotype_connector[f"sus_{age}"]
                    * sim.t.dt_year,  # Person-years of exposure
                    "t": sim.results.timevec,
                    "age_group": age,
                }
            )
            dfs.append(df)
        df = pd.concat(dfs)
        df = df.set_index(["t", "age_group"])

        return df

    cancers_by_age = ss.GammaPoisson(
        name="Cancers by age",
        conform="incident",
        expected=cancer_data,
        extract_fn=extract_cancer_data,
    )

    # Now HPV prevalence

    hpv_prevalence = ss.Binomial(
        name="HPV prevalence 18-49",
        conform="prevalent",
        expected=pd.DataFrame(
            {
                "x": [1987],  # 24.7% HPV prevalence amongst 18-49 year old women
                "n": [8110],
            },
            index=pd.Index([ss.date(d) for d in ["2009-12-31"]], name="t"),
        ),
        extract_fn=lambda sim: pd.DataFrame(
            {
                "x": sim.results.genotype_connector.n_hpv_18_49,
                "n": sim.results.genotype_connector.n_pop_18_49,
            },
            index=pd.Index(sim.results.timevec, name="t"),
        ),
    )

    return [cancers_by_age, hpv_prevalence]


def run_calibration(n_trials=None, n_workers=None, do_save=True, resdir="results"):

    # Define the calibration parameters
    calib_pars = dict(
        nw_prop_f0=dict(low=0.55, high=0.9, guess=0.85),  # network
        nw_f1_conc=dict(low=0.01, high=0.2, guess=0.01),
        nw_m1_conc=dict(low=0.01, high=0.2, guess=0.01),
        nw_p_pair_form=dict(low=0.4, high=0.9, guess=0.5),
        hpv16_beta=dict(low=0.01, high=0.1, guess=0.05),
        hpv18_beta=dict(low=0.01, high=0.1, guess=0.05),
        hpv16_cin_fn_k=dict(low=0.1, high=0.5, guess=0.3),
        hpv18_cin_fn_k=dict(low=0.1, high=0.5, guess=0.35),
    )

    # Make the sim, define data
    sim = make_sim()

    components = make_components()

    class SaveResults:
        def __init__(self, components, resdir):
            self.comps = components
            self.resdir = resdir
            return

        def __call__(self, study, trial):
            for comp in self.comps:
                actual = comp.actual
                actual["trial"] = (
                    trial.number
                )  # Add the trial number to the actuals for reference
                actual["nll"] = (
                    comp.nll
                )  # groupby should preserve the order when computing nll, so results *should* align here

                file_path = os.path.join(
                    self.resdir, f"tmp_{comp.name}_{trial.number}.csv"
                )
                actual.to_csv(file_path)  # Save to a temporary file for debugging
            return

    callback = SaveResults(components, resdir)

    # Make the calibration object
    calib = ss.Calibration(
        calib_pars=calib_pars,
        build_fn=modify_sim,
        sim=sim,
        components=components,
        total_trials=n_trials,
        n_workers=n_workers,
        die=True,
        reseed=False,
        storage=storage,
        keep_db=True,
        continue_db=continue_db,
        callbacks=[callback],
    )

    # Calibrate!
    calib.calibrate(load=True)
    sc.saveobj(os.path.join(resdir, "calib.obj"), calib)

    print(f"Best pars are {calib.best_pars}")

    return sim, calib


def plot_result(component, results, ax, by_age=True):

    ex = component.expected
    if component.name == "Cancers by age":
        ex["cancer_incidence"] = 100000 * ex["x"] / ex["n"]
        to_plot = "cancer_incidence"
    elif component.name == "HPV prevalence 18-49":
        ex["hpv_prevalence"] = ex["x"] / ex["n"]
        to_plot = "hpv_prevalence"
    if by_age:
        sns.boxplot(data=results, x="age_group", y=to_plot, ax=ax)
    else:
        sns.boxplot(data=results, x="t", y=to_plot, ax=ax)
    ax.scatter(x=range(len(ex)), y=ex[to_plot].values)
    ax.set_title(component.name)
    return


if __name__ == "__main__":

    filestem = "apr8"
    resdir = os.path.join("tests/results", filestem)
    os.makedirs(resdir, exist_ok=True)  # Ensure the figures directory exists

    run_args = [
        "run_calibration",
        # "plot_calibration",
    ]
    if "run_calibration" in run_args:
        sim, calib = run_calibration(
            n_trials=n_trials, n_workers=n_workers, resdir=resdir
        )

        calib.check_fit(do_plot=False)
        figs = calib.plot(bootstrap=False)
        for i, fig in enumerate(figs):
            if isinstance(fig, list):
                for j, f in enumerate(fig):
                    f.savefig(os.path.join(resdir, f"Component_{i}_{j}.pdf"))
            else:
                fig.savefig(os.path.join(resdir, f"Component_{i}.pdf"))

        # Optuna plots
        for method in [
            "plot_optimization_history",
            "plot_param_importances",
            "plot_slice",
        ]:
            figs = calib.plot_optuna(method)
            plt.savefig(os.path.join(resdir, f"optuna_{method}.png"), dpi=600)

    elif "plot_calibration" in run_args:
        # Load the calibration results
        # calib = sc.loadobj(os.path.join(resdir, "calib.obj"))
        # pars_df = calib.to_df()
        # res_to_plot = 10
        components = make_components()
        fig, axv = plt.subplots(nrows=1, ncols=len(components), figsize=(12, 6))
        for i, c in enumerate(components):
            ax = axv[i]
            ret = []
            for fn in glob.glob(os.path.join(resdir, f"tmp_{c.name}_*.csv")):
                df = pd.read_csv(fn)
                if c.name == "Cancers by age":
                    by_age = True
                    df["cancer_incidence"] = 100000 * df["x"] / df["n"]
                elif c.name == "HPV prevalence 18-49":
                    by_age = False
                    df["hpv_prevalence"] = df["x"] / df["n"]
                ret.append(df)
            result = pd.concat(ret, ignore_index=True)
            # result = result[result["trial"].isin(list(pars_df[:res_to_plot].index))]
            plot_result(c, result, ax, by_age=by_age)

            file_path = os.path.join(resdir, f"{c.name}.csv")
            result.to_csv(file_path)  # Save to a file for future use
        fig.tight_layout(pad=2.0, h_pad=2.0)
        plt.savefig(os.path.join(resdir, f"calib.png"), dpi=600)

    # calib = sc.loadobj(os.path.join(resdir, 'calib.obj'))
    # results = sc.loadobj(os.path.join(resdir, 'results.obj'))
