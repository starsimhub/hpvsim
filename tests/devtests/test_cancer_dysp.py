"""
Script to explore linking cancer probability to dysplasia

"""

# Imports
import numpy as np
import sciris as sc
import hpvsim as hpv
import hpvsim.utils as hpu
import matplotlib.pyplot as plt
from scipy.stats import lognorm, weibull_min


def lognorm_params(par1, par2):
    """
    Given the mean and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
    sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution

    scale = np.exp(mean)
    shape = sigma
    return shape, scale


def logf1(x, k):
    '''
    The concave part of a logistic function, with point of inflexion at 0,0
    and upper asymptote at 1. Accepts 1 parameter which determines the growth rate.
    '''
    return (2 / (1 + np.exp(-k * x))) - 1


def logf2(x, x_infl, k):
    '''
    Logistic function, constrained to pass through 0,0 and with upper asymptote
    at 1. Accepts 2 parameters: growth rate and point of inflexion.
    '''
    l_asymp = -1/(1+np.exp(k*x_infl))
    return l_asymp + 1/( 1 + np.exp(-k*(x-x_infl)))

def set_font(size=None, font='Libertinus Sans'):
    ''' Set a custom font '''
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return


#%% Functions
def run_calcs():

    # Group genotypes
    genotypes = ['hpv16', 'hpv18', 'hrhpv']
    sim = hpv.Sim(genotypes=genotypes)
    sim.initialize()

    # Get parameters
    ng = sim['n_genotypes']
    genotype_map = sim['genotype_map']

    # Get parameters
    genotype_pars = sim['genotype_pars']
    genotype_pars['hpv16']['cancer_prob'] = 0.0005

    # Shorten duration names
    dur_episomal = [genotype_pars[genotype_map[g]]['dur_episomal'] for g in range(ng)]
    trans_rate = [genotype_pars[genotype_map[g]]['trans_rate'] for g in range(ng)]
    trans_infl = [genotype_pars[genotype_map[g]]['trans_infl'] for g in range(ng)]
    prog_rate = [genotype_pars[genotype_map[g]]['prog_rate'] for g in range(ng)]
    prog_rate_sd = [genotype_pars[genotype_map[g]]['prog_rate_sd'] for g in range(ng)]
    prog_infl = [genotype_pars[genotype_map[g]]['prog_infl'] for g in range(ng)]
    cancer_probs = [genotype_pars[genotype_map[g]]['cancer_prob'] for g in range(ng)]
    clearance_decay = [genotype_pars[genotype_map[g]]['clearance_decay'] for g in range(ng)]
    init_clearance_prob = [genotype_pars[genotype_map[g]]['init_clearance_prob'] for g in range(ng)]

    set_font(size=20)
    colors = sc.gridcolors(ng)
    cmap = plt.cm.Oranges([0.25, 0.5, 0.75, 1])
    fig, ax = plt.subplot_mosaic('AB;CD;EF', figsize=(16, 20))

    ####################
    # Panel A and C
    ####################

    x = np.linspace(0.01, 10, 200)  # Make an array of durations 0-15 years
    glabels = ['HPV16', 'HPV18', 'HRHPV']
    dysp_shares = []
    gtypes = []
    igi = 0.01  # Define the integration interval
    longx = sc.inclusiverange(0.01, 20, igi)  # Initialize a LONG array of years

    # Loop over genotypes, plot each one
    for gi, gtype in enumerate(genotypes):
        sigma, scale = lognorm_params(dur_episomal[gi]['par1'], dur_episomal[gi]['par2'])
        rv = lognorm(sigma, 0, scale)
        aa = np.diff(rv.cdf(longx))  # Calculate the probability that a woman will have a pre-dysplasia duration in any of the subintervals of time spanning 0-25 years
        bb = logf2(longx, trans_infl[gi], trans_rate[gi])[1:]  # Calculate the probablity of her developing dysplasia for a given duration
        dysp_shares.append(np.dot(aa,bb))  # Convolve the two above calculations to determine the probability of her developing dysplasia overall
        gtypes.append(gtype)  # Store genotype names for labeling
        ax['A'].plot(x, rv.pdf(x), color=colors[gi], lw=2, label=glabels[gi])
        ax['C'].plot(x, logf2(x, trans_infl[gi], trans_rate[gi]), color=colors[gi], lw=2, label=gtype.upper())

    bottom = np.zeros(ng)
    ax['E'].bar(np.arange(1, ng + 1), 1-np.array(dysp_shares), color='grey', bottom=bottom, label='Episomal')
    ax['E'].bar(np.arange(1, ng + 1), np.array(dysp_shares), color=cmap[0], bottom=1-np.array(dysp_shares), label='Transforming')
    ax['E'].set_xticks(np.arange(1, ng + 1))
    ax['E'].set_xticklabels(glabels)
    ax['E'].set_ylabel("")
    ax['E'].set_ylabel("Distribution of infection outcomes")

    # Axis labeling and other settings
    ax['C'].set_xlabel("Total duration of episomal infection (years)")
    for axn in ['A', 'C']:
        ax[axn].set_ylabel("")
        ax[axn].grid()

    ax['A'].set_ylabel("Density")
    ax['C'].set_ylabel("Cumulative probability of transformation")
    ax['A'].set_xlabel("Total duration of episomal infection (years)")

    ax['A'].legend(fontsize=20, frameon=True)
    ax['E'].legend(fontsize=20, frameon=True, loc='lower right')


    ####################
    # Make plots
    ####################

    thisx = np.linspace(0.01, 35, 100)
    n_samples = 10

    def cancer_prob(cp,dysp): return 1-np.power(1-cp, dysp*100)

    def clearance_prob(cp_adj, cp,dysp): return cp_adj*(1-(1 - np.power(1 - cp, dysp * 100)))

    # Durations and severity of dysplasia
    for gi, gtype in enumerate(genotypes):
        ax['B'].plot(thisx, logf2(thisx, prog_infl[gi], prog_rate[gi]), color=colors[gi], lw=3, label=gtype.upper())
        for smpl in range(n_samples):
            pr = hpu.sample(dist='normal_pos', par1=prog_rate[gi], par2=prog_rate_sd[gi])
            ax['B'].plot(thisx, logf2(thisx, prog_infl[gi], pr), color=colors[gi], lw=1, alpha=0.5, label=gtype.upper())

        cp = cancer_prob(cancer_probs[gi], logf2(thisx, prog_infl[gi], prog_rate[gi]))
        clear_p = clearance_prob(init_clearance_prob[gi], clearance_decay[gi], logf2(thisx, prog_infl[gi], prog_rate[gi]))
        ax['D'].plot(thisx, cp, color=colors[gi], label=gtype.upper())
        ax['D'].plot(thisx, clear_p, color=colors[gi], ls='--', label=gtype.upper())

    ax['B'].set_ylabel("")
    ax['B'].grid()
    ax['B'].set_ylim([0, 1])
    ax['B'].set_xlabel("Years with transforming infection")
    ax['B'].set_ylabel("% of cells transformed")

    ax['D'].set_ylabel("Probability of invasion or clearance")
    ax['D'].set_xlabel("Years with transforming infection")
    ax['D'].grid()
    h, l = ax['D'].get_legend_handles_labels()

    ax['D'].legend([h[0], h[1]], ['Cervical cancer invasion', 'HPV clearance'], loc='upper right')

    ####################
    # Panel F
    ####################

    # Now determine outcomes for those with transformation

    longx = np.linspace(1, 40, 40)
    n_samples = 1000
    cancers = dict()
    cinshares, cancershares = [], []
    for gi, gtype in enumerate(genotypes):
        cancers[gtype] = 0
        pr = hpu.sample(dist='normal_pos', par1=prog_rate[gi], par2=prog_rate_sd[gi], size=n_samples)
        for x in longx:
            cp = cancer_prob(cancer_probs[gi], logf2(x, prog_infl[gi], pr))
            has_cancer = hpu.n_binomial(cp, len(cp))
            cancer_inds = hpu.true(has_cancer)
            cancers[gtype] += len(cancer_inds)
            pr = pr[~has_cancer]
            cp = clearance_prob(init_clearance_prob[gi], clearance_decay[gi], logf2(x, prog_infl[gi], pr))
            clears_hpv = hpu.n_binomial(cp, len(cp))
            pr = pr[~clears_hpv]

        cancer_share = cancers[gtype]/n_samples
        cin_share = 1 - cancer_share
        cinshares.append(cin_share)
        cancershares.append(cancer_share)

    # This section calculates the overall share of outcomes for people infected with each genotype
    bottom = np.zeros(ng)
    all_shares = [cinshares,
                  cancershares
                  ]

    for gn, grade in enumerate(['Pre-cancer', 'Cervical cancer']):
        ydata = np.array(all_shares[gn])
        color = cmap[gn + 1, :]
        ax['F'].bar(np.arange(1, ng + 1), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata

    ax['F'].set_xticks(np.arange(1, ng + 1))
    ax['F'].set_xticklabels(glabels)
    ax['F'].set_ylabel("")
    ax['F'].set_ylabel("Distribution of transformation outcomes")
    ax['F'].legend(fontsize=20, frameon=True, loc='lower right')
    fig.tight_layout()
    plt.savefig(f"AA_cells.png", dpi=100)


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    run_calcs()

    sc.toc(T)
    print('Done.')
