"""
Testing a new algorithm for computing progression probabilities
"""

# Imports
import numpy as np
import sciris as sc
import pandas as pd
import hpvsim as hpv
import hpvsim.utils as hpu
import matplotlib.pyplot as plt
from scipy.stats import lognorm


# Create sim to get baseline prognoses parameters
sim = hpv.Sim(genotypes='all')
# sim = hpv.Sim(genotypes=[16,18,31,33,35,51,52,56,58])
sim.initialize()

# Get parameters
ng = sim['n_genotypes']
genotype_pars = sim['genotype_pars']
genotype_map = sim['genotype_map']
cancer_thresh = 0.99


# Shorten duration names
dur_precin = [genotype_pars[genotype_map[g]]['dur_precin'] for g in range(ng)]
dur_dysp = [genotype_pars[genotype_map[g]]['dur_dysp'] for g in range(ng)]
dysp_rate = [genotype_pars[genotype_map[g]]['dysp_rate'] for g in range(ng)]
prog_rate = [genotype_pars[genotype_map[g]]['prog_rate'] for g in range(ng)]
prog_rate_sd = [genotype_pars[genotype_map[g]]['prog_rate_sd'] for g in range(ng)]


#%% Helper functions
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
    at 1. Accepts 2 parameters: growth rate and point of inlfexion.
    '''
    l_asymp = -1/(1+np.exp(k*x_infl))
    return l_asymp + 1/( 1 + np.exp(-k*(x-x_infl)))


# Figure settings
font_size = 30
sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
sc.options(font='Libertinus Sans')
plt.rcParams['font.size'] = font_size
colors = sc.gridcolors(ng)
x = np.linspace(0.01, 2, 200)

#%% Preliminary calculations (all to be moved within an analyzer? or sim method?)
#
###### Share of women who develop of detectable dysplasia by genotype
shares = []
gtypes = []
longx = np.linspace(0.01, 12, 1000)

for g in range(ng):
    sigma, scale = lognorm_params(dur_precin[g]['par1'], dur_precin[g]['par2'])
    rv = lognorm(sigma, 0, scale)
    aa = np.diff(rv.cdf(longx))
    bb = logf1(longx, dysp_rate[g])[1:]
    shares.append(np.dot(aa, bb))
    gtypes.append(genotype_map[g].replace('hpv',''))
    # gtypes.append(genotype_map[g].upper())


###### Distribution of eventual outcomes for women by genotype
noneshares, cin1shares, cin2shares, cin3shares, cancershares = [], [], [], [], []
longx = np.linspace(0.01, 21, 1000)
for g in range(ng):
    sigma, scale = lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2'])
    rv = lognorm(sigma, 0, scale)
    dd = logf1(longx, prog_rate[g])

    indcin1 = sc.findinds(dd<.33)[-1]
    if (dd>.33).any():
        indcin2 = sc.findinds((dd>.33)&(dd<.67))[-1]
    else:
        indcin2 = indcin1
    if (dd>.67).any():
        indcin3 = sc.findinds((dd>.67)&(dd<cancer_thresh))[-1]
    else:
        indcin3 = indcin2
    if (dd>cancer_thresh).any():
        indcancer = sc.findinds(dd>cancer_thresh)[-1]
    else:
        indcancer = indcin3

    noneshares.append(1 - shares[g])
    cin1shares.append(((rv.cdf(longx[indcin1])-rv.cdf(longx[0]))*shares[g]))
    cin2shares.append(((rv.cdf(longx[indcin2])-rv.cdf(longx[indcin1]))*shares[g]))
    cin3shares.append(((rv.cdf(longx[indcin3])-rv.cdf(longx[indcin2]))*shares[g]))
    cancershares.append(((rv.cdf(longx[indcancer])-rv.cdf(longx[indcin3]))*shares[g]))

######## Outcomes by duration of infection and genotype
n_samples = 10e3

# create dataframes
data = {}
years = np.arange(1,13)
cin1_shares, cin2_shares, cin3_shares, cancer_shares = [], [], [], []
all_years = []
all_genotypes = []
for g in range(ng):
    sigma, scale = lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2'])
    r = lognorm(sigma, 0, scale)

    for year in years:
        mean_peaks = logf1(year, prog_rate[g])
        peaks = logf1(year, hpu.sample(dist='normal', par1=prog_rate[g], par2=prog_rate_sd[g], size=n_samples))
        cin1_shares.append(sum(peaks<0.33)/n_samples)
        cin2_shares.append(sum((peaks>0.33)&(peaks<0.67))/n_samples)
        cin3_shares.append(sum((peaks>0.67)&(peaks<cancer_thresh))/n_samples)
        cancer_shares.append(sum(peaks>cancer_thresh)/n_samples)
        all_years.append(year)
        # all_genotypes.append(genotype_map[g].upper())
        all_genotypes.append(genotype_map[g].replace('hpv',''))
data = {'Year':all_years, 'Genotype':all_genotypes, 'CIN1':cin1_shares, 'CIN2':cin2_shares, 'CIN3':cin3_shares, 'Cancer': cancer_shares}
sharesdf = pd.DataFrame(data)

HR = ['hpv16', 'hpv18']
OHR = ['hpv31', 'hpv33', 'hpv35', 'hpv45', 'hpv51', 'hpv52', 'hpv56', 'hpv58']
LR = ['hpv6', 'hpv11']
alltypes = [HR, OHR, LR]


################################################################################
# BEGIN FIGURE WITH PRECIN DISTRIBUTIONS
################################################################################
def make_precinfig():

    fig, ax = plt.subplots(2, 3, figsize=(24, 12))

    pn = 0

    # Output table
    table  = ' Type : % no dysp\n'

    for ai, gtypes in enumerate(alltypes):
        for gtype in gtypes:
            sigma, scale = lognorm_params(genotype_pars[gtype]['dur_precin']['par1'], genotype_pars[gtype]['dur_precin']['par2'])
            rv = lognorm(sigma, 0, scale)
            ax[0,ai].plot(x, rv.pdf(x), color=colors[pn], lw=2, label=gtype.upper())
            ax[1,ai].plot(x, logf1(x, genotype_pars[gtype]['dysp_rate']), color=colors[pn], lw=2, label=gtype.upper())
            table += f"{gtype.upper().rjust(5)}: {100-sum(np.diff(rv.cdf(x))*logf1(x, genotype_pars[gtype]['dysp_rate'])[1:])*100:.0f}\n"
            pn += 1

        ax[0,ai].legend(fontsize=18)
        ax[1,ai].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")
        for row in [0,1]:
            ax[row,ai].set_ylabel("")
            ax[row,ai].grid()
            ax[row,ai].set_xticks([0,0.5,1.0,1.5,2.0])
            ax[row,ai].set_xticklabels([0,6,12,18,24])
        # ax[0,ai].get_yaxis().set_ticks([])
        ax[1,ai].set_ylim([0,.99])
        ax[0,ai].set_ylim([0,1.8])
    ax[0,0].set_ylabel("Frequency")
    ax[1,0].set_ylabel("Probability of developing\ndysplasia")

    plt.figtext(0.06, 0.94, 'A', fontweight='bold', fontsize=30)
    plt.figtext(0.375, 0.94, 'B', fontweight='bold', fontsize=30)
    plt.figtext(0.69, 0.94, 'C', fontweight='bold', fontsize=30)
    plt.figtext(0.06, 0.51, 'D', fontweight='bold', fontsize=30)
    plt.figtext(0.375, 0.51, 'E', fontweight='bold', fontsize=30)
    plt.figtext(0.69, 0.51, 'F', fontweight='bold', fontsize=30)

    fig.tight_layout()
    plt.savefig("precin_dists.png", dpi=100)
    print(table)


################################################################################
# BEGIN FIGURE WITH CIN EVOLUTION
################################################################################
def make_cinfig():

    ###### Relationship between duration of dysplasia and clinical severity
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    pn = 0
    thisx = np.linspace(0.01, 20, 100)

    cmap = plt.cm.Oranges([0.25,0.5,0.75,1])
    n_samples = 10

    # Durations
    for ai, gtypes in enumerate([HR, OHR]):
        for gtype in gtypes:
            sigma, scale = lognorm_params(genotype_pars[gtype]['dur_dysp']['par1'], genotype_pars[gtype]['dur_dysp']['par2'])
            rv = lognorm(sigma, 0, scale)
            ax[ai].plot(thisx, rv.pdf(thisx), color=colors[pn], lw=2, label=gtype.upper())
            pn+=1
        ax[ai].set_ylabel("")
        ax[ai].legend(fontsize=18)
        ax[ai].set_ylim([0, 1.49])
        ax[ai].grid()
        ax[ai].set_ylabel("Frequency")
        ax[ai].set_xlabel("Duration of dysplasia prior to\nregression/cancer (years)")

    # Severity
    myg = 'hpv16'
    ax[2].plot(thisx, logf1(thisx, genotype_pars[myg]['prog_rate']), color='k', lw=2)
    for year in range(1, 21):# Plot variation
        peaks = logf1(year, hpu.sample(dist='normal', par1=genotype_pars[myg]['prog_rate'],
                                       par2=genotype_pars[myg]['prog_rate_sd'], size=n_samples))
        ax[2].plot([year]*n_samples, peaks, color='k', lw=0, marker='o', alpha=0.5)

    ax[2].set_xlabel("Duration of dysplasia prior to\nregression/cancer (years)")
    ax[2].set_ylabel("Clinical severity")
    ax[2].axhline(y=0.33, ls=':', c='k')
    ax[2].axhline(y=0.67, ls=':', c='k')
    ax[2].axhline(y=cancer_thresh, ls=':', c='k')
    ax[2].axhspan(0, 0.33, color=cmap[0], alpha=.4)
    ax[2].axhspan(0.33, 0.67, color=cmap[1], alpha=.4)
    ax[2].axhspan(0.67, cancer_thresh, color=cmap[2], alpha=.4)
    ax[2].axhspan(cancer_thresh, 1, color=cmap[3], alpha=.4)
    ax[2].text(-0.3, 0.08, 'CIN1', fontsize=30, rotation=90)
    ax[2].text(-0.3, 0.4, 'CIN2', fontsize=30, rotation=90)
    ax[2].text(-0.3, 0.73, 'CIN3', fontsize=30, rotation=90)


    plt.figtext(0.045, 0.94, 'A', fontweight='bold', fontsize=30)
    plt.figtext(0.375, 0.94, 'B', fontweight='bold', fontsize=30)
    plt.figtext(0.7, 0.94, 'C', fontweight='bold', fontsize=30)

    fig.tight_layout()
    plt.savefig("cin_prog.png", dpi=100)


################################################################################
# BEGIN OUTCOMES FIG
################################################################################
def make_outcomefig():

    fig, ax = plt.subplots(1, 2, figsize=(26, 8))
    cmap = plt.cm.Oranges([0.25,0.5,0.75,1])

    ###### Share of women who develop each CIN grade
    loc_array = np.array([-6,-5,-4,-3,-2,-1,1,2,3,4,5,6])
    w = 0.07
    for y in years:
        la = loc_array[y - 1] * w + np.sign(loc_array[y - 1])*(-1)*w/2
        bottom = np.zeros(ng)
        for gn, grade in enumerate(['CIN1', 'CIN2', 'CIN3', 'Cancer']):
            ydata = sharesdf[sharesdf['Year']==y][grade]
            ax[0].bar(np.arange(1,ng+1)+la, ydata, width=w, color=cmap[gn], bottom=bottom, edgecolor='k', label=grade);
            bottom = bottom + ydata

    # ax[1,1].legend()
    ax[0].set_title("Share of women with dysplasia\nby clinical grade and duration")
    ax[0].set_xlabel("")
    ax[0].set_xticks(np.arange(ng) + 1)
    ax[0].set_xticklabels(gtypes,fontsize=30)


    ##### Final outcomes for women
    bottom = np.zeros(ng+1)
    all_shares = [noneshares+[sum([j*1/ng for j in noneshares])],
                  cin1shares+[sum([j*1/ng for j in cin1shares])],
                  cin2shares+[sum([j*1/ng for j in cin2shares])],
                  cin3shares+[sum([j*1/ng for j in cin3shares])],
                  cancershares+[sum([j*1/ng for j in cancershares])],
                  ]
    for gn,grade in enumerate(['None', 'CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        if len(ydata.shape)>1: ydata = ydata[:,0]
        color = cmap[gn-1] if gn>0 else 'gray'
        ax[1].bar(np.arange(1,ng+2), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata
    ax[1].set_xticks(np.arange(ng+1) + 1)
    ax[1].set_xticklabels(gtypes+['Average'],fontsize=30)
    ax[1].set_ylabel("")
    ax[1].set_title("Eventual outcomes for women\n")
    ax[1].legend(bbox_to_anchor =(0.5, 1.07),loc='upper center',fontsize=20,ncol=5,frameon=False)
    # ax[1].legend(bbox_to_anchor =(1.2, 1.),loc='upper center',fontsize=30,ncol=1,frameon=False)


    plt.figtext(0.04, 0.85, 'A', fontweight='bold', fontsize=30)
    plt.figtext(0.51, 0.85, 'B', fontweight='bold', fontsize=30)

    fig.tight_layout()
    plt.savefig("cin_outcomes.png", dpi=100)



################################################################################
# BEGIN FIGURE 1
################################################################################
def make_fig1():
    fig, ax = plt.subplots(2, 3, figsize=(24, 12))

    ################################################################################
    # Pre-dysplasia dynamics
    ################################################################################

    ###### Distributions
    for g in range(ng):
        sigma, scale = lognorm_params(dur_precin[g]['par1'], dur_precin[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0,0].plot(x, rv.pdf(x), color=colors[g], lw=2, label=genotype_map[g].upper())
    ax[0,0].legend()
    ax[0,0].set_xlabel("Pre-dysplasia/control duration")
    ax[0,0].set_ylabel("")
    ax[0,0].grid()
    ax[0,0].set_title("Distribution of infection durations\nprior to dysplasia or control")


    ###### Relationship between durations and probability of detectable dysplasia
    for g in range(ng):
        ax[0,1].plot(x, logf1(x, dysp_rate[g]), color=colors[g], lw=2)
    ax[0,1].set_xlabel("Pre-dysplasia/control duration")
    ax[0,1].set_ylabel("")
    ax[0,1].grid()
    ax[0,1].legend(fontsize=30, frameon=False)
    ax[0,1].set_title("Probability of developing\ndysplasia by duration")


    ###### Distributions post-dysplasia
    thisx = np.linspace(0.01, 20, 100)
    for g in range(ng):
        sigma, scale = lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0,2].plot(thisx, rv.pdf(thisx), color=colors[g], lw=2, label=genotype_map[g].upper())
    ax[0,2].set_xlabel("Duration of dysplasia")
    ax[0,2].set_ylabel("")
    ax[0,2].grid()
    ax[0,2].set_title("Distribution of dysplasia durations\nprior to cancer/control")



    ################################################################################
    # Post-dysplasia dynamics
    ################################################################################

    ###### Relationship between durations peak clinical severity
    cmap = plt.cm.Oranges([0.25,0.5,0.75,1])
    n_samples = 10
    for g in range(ng):
        ax[1,0].plot(thisx, logf1(thisx, prog_rate[g]), color=colors[g], lw=2, label=genotype_map[g].upper())
        # Plot variation
        for year in range(1, 21):
            peaks = logf1(year, hpu.sample(dist='normal', par1=prog_rate[g], par2=prog_rate_sd[g], size=n_samples))
            ax[1, 0].plot([year] * n_samples, peaks, color=colors[g], lw=0, marker='o', alpha=0.5)

    ax[1,0].set_xlabel("Duration of dysplasia")
    ax[1,0].set_ylabel("")
    ax[1,0].grid(axis='x')
    ax[1,0].set_title("Mean peak clinical severity")
    ax[1,0].get_yaxis().set_ticks([])
    ax[1,0].axhline(y=0.33, ls=':', c='k')
    ax[1,0].axhline(y=0.67, ls=':', c='k')
    ax[1,0].axhline(y=cancer_thresh, ls=':', c='k')
    ax[1,0].axhspan(0, 0.33, color=cmap[0],alpha=.4)
    ax[1,0].axhspan(0.33, 0.67, color=cmap[1],alpha=.4)
    ax[1,0].axhspan(0.67, cancer_thresh, color=cmap[2],alpha=.4)
    ax[1,0].axhspan(cancer_thresh, 1, color=cmap[3],alpha=.4)
    ax[1,0].text(-0.3, 0.08, 'CIN1', fontsize=30, rotation=90)
    ax[1,0].text(-0.3, 0.4, 'CIN2', fontsize=30, rotation=90)
    ax[1,0].text(-0.3, 0.73, 'CIN3', fontsize=30, rotation=90)

    ###### Share of women who develop each CIN grade
    loc_array = np.array([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10])
    w = 0.04
    for y in years:
        la = loc_array[y - 1] * w + np.sign(loc_array[y - 1])*(-1)*w/2
        bottom = np.zeros(ng)
        for gn, grade in enumerate(['CIN1', 'CIN2', 'CIN3', 'Cancer']):
            ydata = sharesdf[sharesdf['Year']==y][grade]
            ax[1,1].bar(np.arange(1,ng+1)+la, ydata, width=w, color=cmap[gn], bottom=bottom, edgecolor='k', label=grade);
            bottom = bottom + ydata

    # ax[1,1].legend()
    ax[1,1].set_title("Share of women with dysplasia\nby clinical grade and duration")
    ax[1,1].set_xlabel("")
    ax[1,1].set_xticks(np.arange(ng) + 1)
    ax[1,1].set_xticklabels(gtypes)


    ##### Final outcomes for women
    bottom = np.zeros(ng+1)
    all_shares = [noneshares+[sum([j*.33 for j in noneshares])],
                  cin1shares+[sum([j*.33 for j in cin1shares])],
                  cin2shares+[sum([j*.33 for j in cin2shares])],
                  cin3shares+[sum([j*.33 for j in cin3shares])],
                  cancershares+[sum([j*.33 for j in cancershares])],
                  ]
    for gn,grade in enumerate(['None', 'CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        if len(ydata.shape)>1: ydata = ydata[:,0]
        color = cmap[gn-1] if gn>0 else 'gray'
        ax[1,2].bar(np.arange(1,ng+2), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata
    ax[1,2].set_xticks(np.arange(ng+1) + 1)
    ax[1,2].set_xticklabels(gtypes+['Average'])
    ax[1,2].set_ylabel("")
    ax[1,2].set_title("Eventual outcomes for women\n")
    ax[1,2].legend(bbox_to_anchor =(1.2, 1.),loc='upper center',fontsize=30,ncol=1,frameon=False)

    fig.tight_layout()
    plt.savefig("progressions-1.png", dpi=100)



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    make_precinfig()
    make_cinfig()
    # make_outcomefig()
    # make_fig1()

    sc.toc(T)
    print('Done.')
