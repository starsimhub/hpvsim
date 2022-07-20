"""
Testing a new algorithm for computing progression probabilities
"""

import numpy as np
import sciris as sc
import os
import sys
import pandas as pd
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
    ax[0,0].plot(x, rv.pdf(x), color=colors[g], lw=2, label=genotype_map[g].upper())
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
ax[0,2].set_title("Distribution of women who develop\ndetectable dysplasia")

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


###### Relationship between durations peak clinical severity
cmap = plt.cm.Oranges([0.33,0.67,1])
n_samples = 10
for g in range(ng):
    ax[1,1].plot(x, mean_peak_fn(x, genotype_pars[genotype_map[g]]['prog_rate']), color=colors[g], lw=2, label=genotype_map[g].upper())

    if g<2:
        # Plot variation
        for year in range(1,7):
            mean_peaks = mean_peak_fn(year, genotype_pars[genotype_map[g]]['prog_rate'])
            peaks = np.minimum(1, hpu.sample(dist='lognormal', par1=mean_peaks, par2=(1 - mean_peaks) ** 2, size=n_samples))
            ax[1,1].plot([year]*n_samples, peaks, color=colors[g], lw=0, marker='o', alpha=0.5)

ax[1,1].set_xlabel("Post-dysplasia duration")
ax[1,1].set_ylabel("")
ax[1,1].grid(axis='x')
ax[1,1].set_title("Mean peak clinical severity by duration\n(dots show individual variation)")
ax[1,1].get_yaxis().set_ticks([])
ax[1,1].axhline(y=0.33, ls=':', c='k')
ax[1,1].axhline(y=0.67, ls=':', c='k')
ax[1,1].axhspan(0, 0.33, color=cmap[0],alpha=.4)
ax[1,1].axhspan(0.33, 0.67, color=cmap[1],alpha=.4)
ax[1,1].axhspan(0.67, 1, color=cmap[2],alpha=.4)
ax[1,1].text(6, 0.1, 'CIN1')
ax[1,1].text(6, 0.45, 'CIN2')
ax[1,1].text(6, 0.8, 'CIN3')


###### Share of women who develop each CIN grade
n_samples = 10e3

# create dataframes
data = {}
years = np.arange(1,12)
n_grades=3
cin1_shares, cin2_shares, cin3_shares = [], [], []
all_years = []
all_genotypes = []
dfs = []
for g in range(ng):
    for year in years:
        mean_peaks = mean_peak_fn(year, genotype_pars[genotype_map[g]]['prog_rate'])
        peaks = np.minimum(1, hpu.sample(dist='lognormal', par1=mean_peaks, par2=(1 - mean_peaks) ** 2, size=n_samples))
        cin1_shares.append(sum(peaks<0.33)/n_samples)
        cin2_shares.append(sum((peaks>0.33)&(peaks<0.67))/n_samples)
        cin3_shares.append(sum(peaks>0.67)/n_samples)
        all_years.append(year)
        all_genotypes.append(genotype_map[g].upper())
data = {'Year':all_years, 'Genotype':all_genotypes, 'CIN1':cin1_shares, 'CIN2':cin2_shares, 'CIN3':cin3_shares}
df = pd.DataFrame(data)

plotdf = df[df["Genotype"]=="HPV16"]
bottom = np.zeros(len(years))
for gn,grade in enumerate(['CIN1', 'CIN2', 'CIN3']):
    ydata = np.array(plotdf[grade])
    ax[1,2].bar(years, ydata, color=cmap[gn], bottom=bottom, alpha=.4, label=grade);
    bottom = bottom + ydata
ax[1,2].legend()
ax[1,2].set_title("Share of women with HPV16\ndysplasia by clinical grade")
ax[1,2].set_xlabel("Duration post-dysplasia")

fig.tight_layout()
plt.savefig("progressions.png", dpi=100)
