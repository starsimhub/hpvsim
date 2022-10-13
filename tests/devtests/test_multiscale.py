'''
Test multiscale assumptions
''' 

import sciris as sc
import hpvsim as hpv

T = sc.timer()


repeats = 2

pars = dict(
    n_agents  = 1e3,
    start     = 1975,
    n_years   = 50,
    burnin    = 25,
    genotypes = [16,18],
    verbose = -1,
)

sims = []
for use_multiscale in [False, True]:
    for r in range(repeats):
        label = ['default', 'multiscale'][use_multiscale]
        label += f'{r}'
        sim = hpv.Sim(pars, use_multiscale=use_multiscale, label=label, rand_seed=r)
        sims.append(sim)


msim = hpv.parallel(sims)
df = msim.compare(output=True, show_match=True)
print(df.to_string())

T.toc('Done')