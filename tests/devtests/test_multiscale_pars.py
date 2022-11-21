import sciris as sc
import pylab as pl
import hpvsim as hpv

# Define the parameters
pars = dict(
    n_agents      = 5e3,       # Population size
    start         = 1980,       # Starting year
    n_years       = 50,         # Number of years to simulate
    genotypes     = [16, 18],   # Include the two genotypes of greatest general interest
    verbose = 0,
    rel_init_prev = 4.0,
    use_multiscale = True,
)

repeats = 3
ms_agent_ratios = [1, 3, 10, 30, 100]#, 300, 1000]
n_agents = [100, 300, 1000]#, 3000, 10000]

# Create the sim
sims = []


data = []
count = 0
for n in n_agents:
    for ms in ms_agent_ratios:
        for r in range(repeats):
            count += 1
            label = f'n={n} ms={ms} r={r}'
            sc.heading(f'Running {count} of {len(n_agents)*len(ms_agent_ratios)*repeats}: {label}')
            sim = hpv.Sim(pars, rand_seed=r, n_agents=n, ms_agent_ratio=ms, label=label)
            T = sc.timer()
            sim.run()
            sim.time = T.tocout()
            row = [n, ms, r, sim.time, sim.results.total_infections.values.sum(), sim.results.total_cancer_deaths.values.sum()]
            data.append(row)

df = sc.dataframe(columns=['n', 'ms', 'r', 'time', 'infs', 'deaths'], data=data)

g = df.groupby(['n', 'ms'])
print(g.mean())
print(g.std())
print(g.std()/g.mean())

#%% Plot sims
# for key in ['n_alive', 'n_total_infected', 'total_hpv_prevalence']:

#     fig = pl.figure()
#     dtcols = sc.vectocolor(-pl.log(dts))
#     for i,s in enumerate(msim.sims):
#         res = s.results
#         pl.plot(res.year, res[key], label=s.label, c=dtcols[i], lw=3, alpha=0.7)
#     pl.title(key)
    
#     pl.legend()
