import sciris as sc
import pylab as pl
import hpvsim as hpv

# Define the parameters
pars = dict(
    n_agents      = 5e3,       # Population size
    start         = 1980,       # Starting year
    n_years       = 50,         # Number of years to simulate
    rand_seed     = 2,          # Set a non-default seed
    genotypes     = [16, 18],   # Include the two genotypes of greatest general interest
    rel_init_prev = 4.0,
)

# Create the sim
sims = []
dts = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
for dt in dts:
    sim = hpv.Sim(pars, dt=dt, label=f'dt={dt}')
    sims.append(sim)
msim = hpv.parallel(sims)
msim.plot()

#%% Plot sims
for key in ['n_alive', 'n_total_infected', 'total_hpv_prevalence']:

    fig = pl.figure()
    dtcols = sc.vectocolor(-pl.log(dts))
    for i,s in enumerate(msim.sims):
        res = s.results
        pl.plot(res.year, res[key], label=s.label, c=dtcols[i], lw=3, alpha=0.7)
    pl.title(key)
    
    pl.legend()
