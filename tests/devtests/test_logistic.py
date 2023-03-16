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
    x_infl = 15 # Fix point of inflection
    ttc = 25
    tp = 3 / 1e4
    s = 1
    rel_sev=1
    rel_sev_sd = 0.15
    k = 0.4
    # Try different growth rates:
    # karr = np.linspace(0.001, 1, 6)

    n_samples = 20

    ks = [0.25, 0.35, 0.45]
    fig, axes = pl.subplots(3, len(ks), figsize=(12, 12))

    for ik, k in enumerate(ks):
        rel_sevs = hpv.utils.sample(**dict(dist='normal_pos', par1=1.0, par2=rel_sev_sd),
                                    size=n_samples)  # Distribution to draw individual level severity scale factors
        # for ittc, ttc in enumerate([5, 10, 25]):
        for isn, s in enumerate([0.5, 1, 1.5]):
            ax = axes[0,ik]
            for irs, rel_sev in enumerate(rel_sevs):
                dysp = logf3(t * rel_sev, x_infl, k, ttc, s)
                if irs == 0:
                    ax.plot(t, dysp, color=colors[isn], label=f's={s}', lw=0.5)
                else:
                    ax.plot(t, dysp, color=colors[isn], lw=0.5)
            ax.set_title(f'Severity, sev rate={k}')
            ax.legend()

            for rel_sev in rel_sevs:
                ax = axes[1, ik]
                t_step = 0.25
                t_sequence = np.arange(0, 200, t_step)
                timesteps = t_sequence / t_step
                cumdysp_arr = np.cumsum(logf3(t_sequence, x_infl, k, ttc, s)) * t_step
                cum_dysp = hppar.compute_severity_integral(t / t_step, rel_sev=rel_sev, pars=dict(form='cumsum'),
                                                           cumdysp=cumdysp_arr)

                ax.plot(t, cum_dysp, color=colors[isn], lw=0.5)
                ax.set_title(f'Cumulative severity, sev rate={k}')

                ax= axes[2, ik]
                y = transform_prob(tp, cum_dysp)
                ax.plot(t, y, color=colors[isn])
                ax.set_title(f'Probability of transformation, sev rate={k}')

    # axes[0].legend(frameon=False)
    pl.show()

    sc.toc(T)
    print('Done.')
