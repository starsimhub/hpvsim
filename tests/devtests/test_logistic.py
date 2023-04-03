'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
import hpvsim.parameters as hppar
import pylab as pl

def logf2(x, x_infl, k):
    l_asymp = -1/(1+np.exp(k*x_infl))
    return l_asymp + 1/( 1 + np.exp(-k*(x-x_infl)))

def get_asymptotes(x_infl, k, ttc=25, s=1):
    term1 = (1 + np.exp(k*(x_infl-ttc)))**s
    term2 = (1 + np.exp(k*x_infl))**s
    u_asymp_num = term1*(1-term2)
    u_asymp_denom = term1 - term2
    u_asymp = u_asymp_num / u_asymp_denom
    l_asymp = term1 / (term1 - term2)
    return l_asymp, u_asymp

def logf3(x, x_infl, k, ttc=25, s=1):
    l_asymp, u_asymp = get_asymptotes(x_infl, k, ttc, s)
    return np.minimum(1, l_asymp + (u_asymp-l_asymp)/(1+np.exp(k*(x_infl-x)))**s)


def transform_prob(tp,dysp):
    '''
    Returns transformation probability given % of dysplastic cells
    '''

    return 1-np.power(1-tp, dysp*100)


#%% Run as a script
if __name__ == '__main__':

    # Start timing
    T = sc.tic()


    colors = sc.gridcolors(10)
    t = np.arange(0,30,0.1) # Array of years
    rel_sev = 1

    multical_i=f'results/india_{mc_filename}_pars.obj'
    pars_n = {'k': 0.241455, 'x_infl': 12.0686, 's': 1, 'ttc': 6.04416}
    pars_i = {'k': 0.279961, 'x_infl': 5.86349, 's': 1, 'ttc': 9.5469}
    apars_n = {'form': 'logf3', 'k': 0.241455, 'x_infl': 12.0686, 's': 1, 'ttc': 6.04416}
    apars_i = {'form': 'logf3', 'k': 0.279961, 'x_infl': 5.86349, 's': 1, 'ttc': 9.5469}
    tp_n = 3.53262e-05
    tp_i = 1.8846e-05

    n_samples = 20

    fig, axes = pl.subplots(2, 2, figsize=(12, 12))

    ax = axes[0,0]
    ax.plot(t, logf3(t * rel_sev, **pars_i), color=colors[1], label=f'India', lw=2)
    ax.plot(t, logf3(t * rel_sev, **pars_n), color=colors[2], label=f'Nigeria', lw=2)
    ax.set_title(f'Severity')

    ax = axes[1,0]
    cum_dysp_i = hppar.compute_severity_integral(t, rel_sev=rel_sev, pars=apars_i)
    cum_dysp_n = hppar.compute_severity_integral(t, rel_sev=rel_sev, pars=apars_n)
    ax.plot(t, cum_dysp_i, color=colors[1], lw=2)
    ax.plot(t, cum_dysp_n, color=colors[2], lw=2)
    ax.set_title(f'Cumulative severity')

    ax= axes[0,1]
    ax.plot(t, transform_prob(tp_i, cum_dysp_i), color=colors[1])
    ax.plot(t, transform_prob(tp_n, cum_dysp_n), color=colors[2])
    ax.set_title(f'Probability of transformation')

    axes[0,0].legend(frameon=False)
    pl.show()

    sc.toc(T)
    print('Done.')
