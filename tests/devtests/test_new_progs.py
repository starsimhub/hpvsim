"""
Testing a new algorithm for computing progression probabilities
"""

import numpy as np
import sciris as sc
import os
import sys
import hpvsim as hpv
import hpvsim.utils as hpu
import matplotlib.pyplot as plt

# Create sim to get baseline prognoses parameters
hpv16 = hpv.genotype('HPV16')
hpv18 = hpv.genotype('HPV18')
hpv6 = hpv.genotype('HPV6')
hpv31 = hpv.genotype('HPV31')
sim = hpv.Sim(genotypes=[hpv16,hpv18,hpv6,hpv31])
sim.initialize()

# Create plot of durations pre-dysplasia
ng = sim['n_genotypes']
progs = sim['prognoses']
genotype_pars = sim['genotype_pars']
genotype_map = sim['genotype_map']
durpars = [genotype_pars[genotype_map[g]]['dur'] for g in genotype_map]

# Prognoses to fit to
prognoses = dict(
        duration_cutoffs  = np.array([0,       1,          2,          3,          4,          5,          10]),      # Duration cutoffs (lower limits)
        seroconvert_probs = np.array([0.25,    0.5,        0.95,       1.0,        1.0,        1.0,        1.0]),    # Probability of seroconverting given duration of infection
        cin1_probs        = np.array([0.015,   0.3655,     0.86800,    1.0,        1.0,        1.0,        1.0]),   # Conditional probability of developing CIN1 given HPV infection
        cin2_probs        = np.array([0.020,   0.0287,     0.0305,     0.06427,    0.1659,     0.3011,     0.4483]),   # Conditional probability of developing CIN2 given CIN1, derived from Harvard model calibration
        cin3_probs        = np.array([0.007,   0.0097,     0.0102,     0.0219,     0.0586,     0.112,      0.1779]),   # Conditional probability of developing CIN3 given CIN2, derived from Harvard model calibration
        cancer_probs      = np.array([0.002,   0.003,      0.0564,     0.1569,     0.2908,     0.3111,     0.5586]),   # Conditional probability of developing cancer given CIN3, derived from Harvard model calibration
        )

# Make figure with the durations
def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale

from scipy.stats import lognorm
import matplotlib.pyplot as plt
font_size = 26
font_family = 'Libertinus Sans'
plt.rcParams['font.size'] = font_size
plt.rcParams['font.family'] = font_family

fig, ax = plt.subplots(2, 3, figsize=(24, 12))
x = np.linspace(0.01, 7, 700)
colors = sc.gridcolors(ng)

################################################################################
# Pre-dysplasia dynamics
################################################################################

###### Distributions
for g in range(ng):
    sigma, scale = lognorm_params(durpars[g]['none']['par1'], durpars[g]['none']['par2'])
    rv = lognorm(sigma, 0, scale)
    ax[0,0].plot(x, rv.cdf(x), color=colors[g], lw=2, label=genotype_map[g].upper())
    # ax[1].plot(x, rv.cdf(x), color=colors[g], lw=2, label=genotype_map[g].upper())
ax[0,0].legend()
ax[0,0].set_xlabel("Pre-dysplasia/clearance duration")
ax[0,0].set_ylabel("")
ax[0,0].grid()
ax[0,0].set_title("Distribution of infection durations\nprior to detectable dysplasia/control")

# Map durations pre-dysplasia to the probability of dysplasia beginning
def mean_peak_fn(x, k):
    '''
    Define a function to link the duration of dysplasia prior to control/integration
    to the peak dysplasia prior to control/integration.
    Currently this is modeled as the concave part of a logistic function
    '''
    return (2 / (1 + np.exp(-k * x))) - 1

###### Relationship between durations and probability of detectable dysplasia
xx = prognoses['duration_cutoffs']
yy = prognoses['cin1_probs']
for g in range(ng):
    ax[0,1].plot(x, mean_peak_fn(x, genotype_pars[genotype_map[g]]['dysp_rate']), color=colors[g], lw=2, label=genotype_map[g].upper())
ax[0,1].plot(xx[:-1], yy[:-1], 'ko', label="Values from Jane's model")
ax[0,1].set_xlabel("Pre-dysplasia/clearance duration")
ax[0,1].set_ylabel("")
ax[0,1].grid()
ax[0,1].set_title("Probability of developing\ndetectable dysplasia by duration")

###### Share of women who develop of detectable dysplasia
for g in range(ng):
    yvals = rv.pdf(x) * mean_peak_fn(x, genotype_pars[genotype_map[g]]['dysp_rate'])
    ax[0,2].plot(x, yvals, color=colors[g], lw=2)
ax[0,2].set_xlabel("Pre-dysplasia/clearance duration")
ax[0,2].set_ylabel("")
ax[0,2].grid()
ax[0,2].set_title("Share of women who develop\ndetectable dysplasia")

################################################################################
# Post-dysplasia dynamics
################################################################################

###### Distributions
for g in range(ng):
    sigma, scale = lognorm_params(durpars[g]['dys']['par1'], durpars[g]['dys']['par2'])
    rv = lognorm(sigma, 0, scale)
    ax[1,0].plot(x, rv.pdf(x), color=colors[g], lw=2, label=genotype_map[g].upper())
    # ax[1].plot(x, rv.cdf(x), color=colors[g], lw=2, label=genotype_map[g].upper())
ax[1,0].set_xlabel("Post-dysplasia duration")
ax[1,0].set_ylabel("")
ax[1,0].grid()
ax[1,0].set_title("Distribution of dysplasia durations\nprior to integration/control")


###### Relationship between durations and probability of detectable dysplasia
cmap = plt.cm.Oranges([0.33,0.67,1])
for g in range(ng):
    ax[1,1].plot(x, mean_peak_fn(x, genotype_pars[genotype_map[g]]['prog_rate']), color=colors[g], lw=2, label=genotype_map[g].upper())

# Plot variation for HPV18 only
peaks = np.minimum(1, hpu.sample(dist='lognormal', par1=mean_peaks, par2=(1 - mean_peaks) ** 2))  # Evaluate peak dysplasia, which is a proxy for the clinical classification

ax[1,1].set_xlabel("Post-dysplasia duration")
ax[1,1].set_ylabel("")
ax[1,1].grid(axis='x')
ax[1,1].set_title("Mean peak clinical severity by duration")
ax[1,1].get_yaxis().set_ticks([])
ax[1,1].axhline(y=0.33, ls=':', c='k')
ax[1,1].axhline(y=0.67, ls=':', c='k')
ax[1,1].axhspan(0, 0.33, color=cmap[0],alpha=.4)
ax[1,1].axhspan(0.33, 0.67, color=cmap[1],alpha=.4)
ax[1,1].axhspan(0.67, 1, color=cmap[2],alpha=.4)
ax[1,1].text(6, 0.1, 'CIN1')
ax[1,1].text(6, 0.45, 'CIN2')
ax[1,1].text(6, 0.8, 'CIN3')


# ###### Share of women who develop each CIN grade
# for g in range(ng):
#     ax[1,2].plot(x, mean_peak_fn(rv.cdf(x), genotype_pars[genotype_map[g]]['dysp_rate']), color=colors[g], lw=2)
# ax[1,2].set_xlabel("Duration of infection prior to detectable dysplasia")
# ax[1,2].set_ylabel("")
# ax[1,2].grid()
# ax[1,2].set_title("Share of women who develop\ndetectable dysplasia")


fig.tight_layout()
plt.savefig("progressions.png", dpi=100)


# # gtypes = np.random.randint(0,ng,100) # Make 100 people infected with one of these genotypes
#
# # Sample some durations
# dur_none = [hpu.sample(**durpars[g]['none'],size=1000) for g in range(ng)]
# dur_cin1 = [hpu.sample(**durpars[g]['cin1'],size=1000) for g in range(ng)]
# dur_cin2 = [hpu.sample(**durpars[g]['cin2'],size=1000) for g in range(ng)]
# dur_cin3 = [hpu.sample(**durpars[g]['cin3'],size=1000) for g in range(ng)]
#
# # Use the durations to create dysplasia curves
# def f2(x, k, x0):
#     ''' 2 parameter logistic function '''
#     return 1 / (1. + np.exp(-k * (x - x0)))
#
# progprobs = [f2(dur_none[g], 2.5, 1.2) for g in range(ng)]
# np.histogram(progprobs[0], bins=np.linspace(0,1,11)) # Plot
# is_cin1 = hpu.binomial_arr(progprobs[0]) # Figure out how many actually progress
#
# # Output mean duration of infection for those who don't progress
# # Output time to progression for those who do progress
#
# # Potential problem here - what if it's quick to progress? i.e., the duration
# # of time with no dyplasia is short, but we draw progression probabilities based
# # on the duration. Maybe we want to draw the peak and the time to peak independently
#
# # Draw duration of overall infection - genotype specific
# # Draw peak dysplasia (related to duration??) - yes, related to duration
# # Draw growth rate or time to peak (related to either of the above??)
# peak_dys = hpu.sample(dist='beta', par1=1, par2=1, size=100) # Figure out a distribution st the distribution of people who go to CIN1/2/3 is about right
# np.histogram(peak_dys, bins=np.linspace(0,1,4))

# ##################################################
# # 1. Define distributions for the duration of infection prior to (a) clearance or (b) dysplasia, whichever occurs first
# dur_none = sc.objdict()
# dur_none.hpv16 = {'dist': 'lognormal', 'par1': 1, 'par2': 1} # Slow-growing oncogenic genotype
# dur_none.hpv18 = {'dist': 'lognormal', 'par1': 1, 'par2': 1} # Fast-growing oncogenic genotype
# dur_none.hpv6 = {'dist': 'lognormal', 'par1': 1, 'par2': 0.1} # Fast-clearing non-oncogenic genotype
#
# # 2. Define distributions for the duration of infection with dysplasia prior to (a) control or (b) integration & progression to cancer
# ttp = sc.objdict()
# ttp.hpv16 = {'dist': 'lognormal', 'par1': 4, 'par2': 2} # Slow-growing oncogenic genotype
# ttp.hpv18 = {'dist': 'lognormal', 'par1': 3, 'par2': 2} # Fast-growing oncogenic genotype
# ttp.hpv6 = {'dist': 'lognormal', 'par1': 1, 'par2': 1} # Non-oncogenic genotype
#
# # 2. Define the relative speed of progression
# rel_prog = sc.objdict()
# rel_prog.hpv16 = 1
# rel_prog.hpv18 = 1.5
# rel_prog.hpv6 = 0.5
#
# # 3. Define a function to link the duration of non-integrated/controlled infection to the peak dysplasia prior to control/integration
# def mean_peak_fn(x, k):
#     ''' Concave part of logistic function '''
#     return (2 / (1 + np.exp(-k * x))) - 1
#
# # Sample prognoses for people infected with each genotype
# ng = len(ttp)
# n_agents = 100
# durations = []
# mean_peaks = []
# peaks = []
# cin1_probs, cin2_probs, cin3_probs = [], [], []
# is_cin2, is_cin3 = [], []
# for g in range(ng):
#     # Evaluate duration of viral infection without dysplasia
#     dur_no_dys  = hpu.sample(**dur_none[g], size=n_agents) # Draw durations of infection prior to dysplasia
#     prob_dys    = mean_peak_fn(dur_no_dys, rel_prog[g]) # Probability of establishing a persistent infection
#     persistent  = hpu.binomial_arr(prob_dys) # Boolean array of persistent infections
#     pers_inds   = hpu.true(persistent) # Indices of those with persistent infections
#     cin1_probs.append(len(pers_inds)/n_agents) # Probabilities of progressing to CIN1, aka of dysplasia beginning
#
#     # For people with dysplasia, evaluate duration of dysplasia prior to (a) control or (b) integration & progression to cancer
#     these_durations = hpu.sample(**ttp[g], size=len(pers_inds))
#     these_mean_peaks = mean_peak_fn(these_durations, rel_prog[g])
#     these_peaks = np.minimum(1, hpu.sample(dist='lognormal', par1=these_mean_peaks, par2=(1-these_mean_peaks)**2))
#     durations.append(these_durations)
#     mean_peaks.append(these_mean_peaks)
#     peaks.append(these_peaks)
#
#     # Get probabilities of progressing to CIN2/3
#     is_cin2.append(hpu.true(peaks[g]>.33))
#     is_cin3.append(hpu.true(peaks[g]>.66))
#     cin2_probs.append(len(is_cin2[-1])/n_agents)
#     cin3_probs.append(len(is_cin3[-1])/n_agents)
#

#
# do_plot=0
# if do_plot:
#     fig, ax = plt.subplots(1, 1)
#     colors = sc.gridcolors(ng)
#     for g in range(ng):
#         yvals = hpu.sample(**ttp[g], size=100e3)
#         bins = np.linspace(0,10,101)
#         yy,xx = np.histogram(yvals, bins=bins)
#         ax.plot(xx[:-1],yy,color=colors[g])
#     plt.show()
#
#
#
# fig, ax = plt.subplots(1, 1)
# colors = sc.gridcolors(2)
# for g in range(2):
#     yvals = hpu.sample(**ttp[g], size=100e3)
#     bins = np.linspace(0,10,101)
#     yy,xx = np.histogram(yvals, bins=bins)
#     ax.plot(xx[:-1],yy/1e3,color=colors[g])
#
# plt.show()
#
# fig, ax = plt.subplots(1, 1)
# xfull = np.linspace(0,15,151)
# kvals = [0.2, 0.3, 0.4, 0.5, 1, 2, 4]
# colors = sc.gridcolors(len(kvals))
# for i,k in enumerate(kvals):
#     yfull = f2(xfull, k)
#     ax.plot(xfull, yfull, color=colors[i], label=f'k={k}')
# ax.axhline(y=0.33, ls=':', c='k')
# ax.axhline(y=0.67, ls=':', c='k')
# plt.legend()
# plt.show()
#
# # 3. Draw individual peak
# mean_peaks = f2(yvals, 1)
# individual_peaks = hpu.sample(dist='lognormal', par1=mean_peaks, par2=0.01)
#

if __name__ == '__main__':

    plot_weibull=0
    plot_gompertz_samples=0


    if plot_weibull:
        from scipy.stats import weibull_min
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        c = 6.25 # Fixed
        lam_range = np.linspace(4,10,7) # Time to peak
        cols = plt.cm.Oranges(lam_range/10)
        max_x = 15 # For plotting
        x = np.linspace(0, max_x, 100) # x axis

        # Plot dysplastic cell growth for different durations
        for i,lam in enumerate(lam_range):
            rv = weibull_min(c, loc=0, scale=lam)
            sf = (lam/c)**3 # Scale factor to adjust peak height
            ax.plot(x, sf*rv.pdf(x), color=cols[i], lw=2,
                    label=f'Time to peak = {lam} years')

        ax.axhline(y=0.33, ls=':', c='k')
        ax.axhline(y=0.67, ls=':', c='k')
        ax.legend(loc='best', frameon=False)
        ax.set_ylabel('Cell dysplasia proportion')
        ax.set_xlabel('Duration of infection')
        ax.set_ylim([0, 1])
        plt.show()


    if plot_gompertz_samples:
        from scipy.stats import gompertz
        import matplotlib.pyplot as plt
        import sciris as sc

        fig, ax = plt.subplots(1, 1)
        c = 0.01 # Shape parameter - fixed for now
        times_to_peak = np.linspace(1,10,4) # Durations of infection
        n_peak_samples = 10
        cols = sc.gridcolors(4)
        max_x = 15 # For plotting
        x = np.linspace(0, max_x, 100) # x axis

        def mean_peak(x, k=1/3, x0=5, max_x=0.8):
            ''' Logistic relationship between duration and mean peak dysplasia '''
            return max_x/(1+np.exp(-k*(x-x0)))

        def lognormal(par1, par2=0.1, size=1):
            ''' Lognormal distribution for peak dysplasia '''
            mean = np.log(par1**2 / np.sqrt(par2**2 + par1**2))  # Computes the mean of the underlying normal distribution
            sigma = np.sqrt(np.log(par2**2 / par1**2 + 1))  # Computes sigma for the underlying normal distribution
            samples = np.random.lognormal(mean=mean, sigma=sigma, size=size)
            return samples

        # Plot cell dysplasia for different durations
        for i,time in enumerate(times_to_peak):
            scale = time / np.log(1 / c)
            rv = gompertz(c, loc=0, scale=scale)
            default_peak = rv.pdf(time)
            this_mean_peak = mean_peak(time)
            mean_sf = this_mean_peak / default_peak
            ax.plot(x, mean_sf * rv.pdf(x), color=cols[i], lw=2, label=f'Time to peak = {time} years')
            for ps in range(n_peak_samples):
                sf = lognormal(this_mean_peak)/default_peak
                ax.plot(x, sf*rv.pdf(x), color=cols[i], lw=1, alpha=0.5)

        ax.axhline(y=0.33, ls=':', c='k')
        ax.axhline(y=0.67, ls=':', c='k')
        ax.legend(loc='best', frameon=False)
        ax.set_ylabel('Cell dysplasia proportion')
        ax.set_xlabel('Duration of infection')
        ax.set_ylim([0, 1])
        plt.show()
