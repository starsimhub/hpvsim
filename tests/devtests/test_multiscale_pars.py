import sciris as sc
import pylab as pl
import hpvsim as hpv

rerun = False
filename = "multiscale_test.df"

T = sc.timer()

if rerun:
    # Define the parameters
    pars = dict(
        total_pop=10e3,  # Population size
        start=1980,  # Starting year
        n_years=50,  # Number of years to simulate
        genotypes=[16, 18],  # Include the two genotypes of greatest general interest
        verbose=0,
        rel_init_prev=4.0,
        use_multiscale=True,
    )

    debug = 0
    repeats = [10, 3][debug]
    ms_agent_ratios = [[1, 3, 10, 30, 100], [1, 3, 10, 30]][debug]
    n_agents = [
        [100, 200, 500, 1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3],
        [100, 200, 500, 1000],
    ][debug]

    # Run the sims -- not parallelized to collect timings
    data = []
    count = 0
    for n in n_agents:
        for ms in ms_agent_ratios:
            for r in range(repeats):
                count += 1
                label = f"n={n} ms={ms} r={r}"
                sc.heading(
                    f"Running {count} of {len(n_agents)*len(ms_agent_ratios)*repeats}: {label}"
                )
                sim = hpv.Sim(
                    pars, rand_seed=r, n_agents=n, ms_agent_ratio=ms, label=label
                )
                T.tic()
                sim.run()
                sim.time = T.tocout()
                row = dict(
                    n=n,
                    ms=ms,
                    seed=r,
                    time=sim.time,
                    n_agents=len(sim.people),
                    infs=sim.results.total_infections.values.sum(),
                    deaths=sim.results.total_cancer_deaths.values.sum(),
                )
                data.append(row)
                print(f"Time: {sim.time:0.2f} s")

    df = sc.dataframe(data)
    sc.save(filename, df)

else:
    df = sc.load(filename)


# %% Analysis

g = df.groupby(["n", "ms"])
gm = g.mean()
gs = g.std()
gc = g.std() / g.mean()

sc.heading("Means")
print(gm)
sc.heading("STDs")
print(gs)
sc.heading("CoVs")
print(gc)

quantity = ["infs", "deaths"][1]
fig = pl.figure()
index = gc.reset_index()
colors = sc.vectocolor(pl.log(index["ms"].values), cmap="parula")
sizes = 2 * (index["n"].values) ** (1 / 2)
pl.scatter(
    gm["time"].values,
    gc[quantity].values,
    lw=0,
    marker="o",
    c=colors,
    s=sizes,
    alpha=0.7,
)
pl.gca().set_xscale("log")
pl.gca().set_yscale("log")
pl.xlabel("Time per simulation (s)")
pl.ylabel(f"Coefficient of variation in {quantity}")
pl.title("Dot color = multiscale agent ratio\nDot size = number of agents")
pl.show()


total = T.timings[:].sum()
print(f"Done: {total:0.0f} s")
