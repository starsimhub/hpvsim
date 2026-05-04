'''
Numerical utilities for v2 hpvsim (quarantine copy).

This file is a verbatim copy of the pre-M02 hpvsim/utils.py kept here so that
_v2_legacy modules (which do ``from . import utils as hpu``) continue to work
without importing from the active hpvsim.utils surface.

DO NOT import this file from active (non-_v2_legacy) code.
'''

import numpy as np
import sciris as sc
from scipy.stats import norm


__all__ = []


def unique(arr):
    '''
    Find the unique elements and counts in an array.
    Equivalent to np.unique(return_counts=True) but ~5x faster, and
    only works for arrays of positive integers.
    '''
    counts = np.bincount(arr.ravel())
    unique = np.flatnonzero(counts)
    counts = counts[unique]
    return unique, counts


def isin(arr, search_inds):
    ''' Find search_inds in arr. Like np.isin() but faster '''
    n = len(arr)
    result = np.full(n, False)
    set_search_inds = set(search_inds)
    for i in range(n):
        if arr[i] in set_search_inds:
            result[i] = True
    return result


def findinds(arr, vals):
    ''' Finds indices of vals in arr, accounting for repeats '''
    return isin(arr, vals).nonzero()[0]


def find_contacts(p1, p2, inds):  # pragma: no cover
    """
    Numba for Layer.find_contacts()

    A set is returned here rather than a sorted array so that custom tracing interventions can efficiently
    add extra people. For a version with sorting by default, see Layer.find_contacts(). Indices must be
    an int64 array since this is what's returned by true() etc. functions by default.
    """
    pairing_partners = set()
    inds = set(inds)
    for i in range(len(p1)):
        if p1[i] in inds:
            pairing_partners.add(p2[i])
        if p2[i] in inds:
            pairing_partners.add(p1[i])
    return pairing_partners


def logf1(x, k, ttc=25):
    '''
    Logistic function passing through (0,0) and (ttc,1).
    Accepts 1 parameter which determines the growth rate.
    '''
    return logf3(x, k, 0, 1, ttc=ttc)


def get_asymptotes(k, x_infl, s=1, y_max=1, ttc=25):
    '''
    Get upper asymptotes for logistic functions
    '''
    term1 = (1 + np.exp(k*(x_infl-ttc)))**s
    term2 = (1 + np.exp(k*x_infl))**s
    u_asymp_num = y_max*term1*(1-term2)
    u_asymp_denom = term1 - term2
    u_asymp = u_asymp_num / u_asymp_denom
    l_asymp = y_max * term1 / (term1 - term2)
    return l_asymp, u_asymp


def logf3(x, k, x_infl, s=1, y_max=1, ttc=25):
    '''
    Logistic function passing through (0,0) and (ttc,y_max).
    This version is derived from the 5-parameter version here: https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
    However, since it's constrained to pass through 2 points, there are 3 free parameters remaining.
    Args:
         k: growth rate
         x_infl: a location parameter
         s: asymmetry parameter
         ttc (time to cancer): x value for which the curve passes through 1.
    '''
    l_asymp, u_asymp = get_asymptotes(k, x_infl, s=1, y_max=y_max, ttc=ttc)
    return np.minimum(1, l_asymp + (u_asymp-l_asymp)/(1+np.exp(k*(x_infl-x)))**s)


def logf2(x, k, x_infl, y_max=1, ttc=25):
    '''
    Logistic function constrained to pass through (0,0) and (ttc,y_max). s=1 fixed.
    '''
    return logf3(x, k, x_infl, s=1, y_max=y_max, ttc=ttc)


def linear(x, slope, b=0):
    '''
    Linear function
    '''
    return b + slope*x


def invlogf3(y, k, x_infl, s, ttc=25):
    '''
    Inverse of logf3; see definition there for arguments
    '''
    l_asymp, u_asymp = get_asymptotes(k, x_infl, s, ttc)
    part1 = np.log((u_asymp-l_asymp)/(y-l_asymp))/s
    part2 = np.log(np.exp(part1)-1)
    final = 1/k * (k*x_infl - part2)
    return final


def invlogf2(y, k, x_infl, ttc=25):
    '''
    Inverse of logf2; see definition there for arguments
    '''
    return invlogf3(y, k, x_infl, 1, ttc=ttc)


def invlogf1(y, k, ttc=25):
    '''
    The inverse of the concave part of a logistic function.
    '''
    return invlogf3(y, k, 0, 1, ttc=ttc)


def indef_int_logf2(x, k, x_infl, ttc=25, y_max=1):
    '''
    Indefinite integral of logf2; see definition there for arguments
    '''
    t1 = 1 + np.exp(k*(x_infl-ttc))
    t2 = 1 + np.exp(k*x_infl)
    integ = np.log(np.exp(k*(x_infl-x)) + 1) / k + x
    result = y_max/(t1-t2)*(1-t1*t2*integ)
    return result


def intlogf2(upper, k, x_infl, ttc=25, y_max=1):
    '''
    Integral of logf2 between 0 and the limit given by upper
    '''
    exceeding_ttc_inds = (upper > ttc).nonzero()
    lims_to_find = np.minimum(ttc, upper)

    val_at_0 = indef_int_logf2(0, k, x_infl, ttc)
    val_at_lim = indef_int_logf2(lims_to_find, k, x_infl, ttc)
    integral = val_at_lim - val_at_0

    excess_integral = upper[exceeding_ttc_inds] - ttc
    integral[exceeding_ttc_inds] += excess_integral

    return integral


def indef_int_logf1(x, k, ttc=25):
    '''
    Indefinite integral of logf1; see definition there for arguments
    '''
    return indef_int_logf2(x, k, 0, ttc=ttc)


def intlogf1(upper, k, ttc=25):
    '''
    Integral of logf1 between 0 and the limit given by upper
    '''
    return intlogf2(upper, k, 0, ttc=ttc)


def transform_prob(tp, dysp):
    '''
    Returns transformation probability given dysplasia.
    '''
    return 1-np.power(1-tp, 0.5*((dysp)**3)*100)


def logn_percentiles_to_pars(x1, p1, x2, p2):
    """
    Find the parameters of a lognormal distribution where:
        P(X < p1) = x1
        P(X < p2) = x2
    """
    x1 = np.log(x1)
    x2 = np.log(x2)
    p1ppf = norm.ppf(p1)
    p2ppf = norm.ppf(p2)
    s = (x2 - x1) / (p2ppf - p1ppf)
    mean = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)
    scale = np.exp(mean)
    return s, scale


__all__ += ['sample', 'get_pdf', 'set_seed']


def sample(dist=None, par1=None, par2=None, size=None, **kwargs):
    '''
    Draw a sample from the distribution specified by the input.
    '''
    choices = [
        'uniform', 'normal', 'normal_pos', 'normal_int',
        'lognormal', 'lognormal_int', 'poisson', 'poisson1',
        'neg_binomial', 'neg_binomial1', 'beta', 'gamma',
    ]

    if size is not None and not isinstance(size, tuple):
        size = int(size)

    if   dist in ['unif', 'uniform']: samples = np.random.uniform(low=par1, high=par2, size=size, **kwargs)
    elif dist in ['norm', 'normal']:  samples = np.random.normal(loc=par1, scale=par2, size=size, **kwargs)
    elif dist == 'normal_pos':        samples = np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs))
    elif dist == 'normal_int':        samples = np.round(np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs)))
    elif dist == 'poisson':           samples = n_poisson(rate=par1, n=size, **kwargs)
    elif dist == 'poisson1':          samples = n_poisson(rate=par1, n=size, **kwargs)+1
    elif dist == 'neg_binomial':      samples = n_neg_binomial(rate=par1, dispersion=par2, n=size, **kwargs)
    elif dist == 'neg_binomial1':     samples = n_neg_binomial(rate=par1, dispersion=par2, n=size, **kwargs)+1
    elif dist == 'beta':              samples = np.random.beta(a=par1, b=par2, size=size, **kwargs)
    elif dist == 'gamma':             samples = np.random.gamma(shape=par1, scale=par2, size=size, **kwargs)
    elif dist in ['lognorm', 'lognormal', 'lognorm_int', 'lognormal_int']:
        if (sc.isnumber(par1) and par1>0) or (sc.checktype(par1, 'arraylike') and (par1>0).all()):
            mean  = np.log(par1**2 / np.sqrt(par2**2 + par1**2))
            sigma = np.sqrt(np.log(par2**2/par1**2 + 1))
            samples = np.random.lognormal(mean=mean, sigma=sigma, size=size, **kwargs)
        else:
            samples = np.zeros(size)
        if '_int' in dist:
            samples = np.round(samples)
    elif dist == 'beta_mean':
        a       = ((1 - par1)/par2 - 1/par1) * par1**2
        b       = a * (1 / par1 - 1)
        samples = np.random.beta(a=a, b=b, size=size, **kwargs)
    else:
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {sc.newlinejoin(choices)}'
        raise NotImplementedError(errormsg)

    return samples


def get_pdf(dist=None, par1=None, par2=None):
    '''
    Return a probability density function for the specified distribution.
    '''
    import scipy.stats as sps

    choices = ['none', 'uniform', 'lognormal']

    if dist in ['None', 'none', None]:
        return None
    elif dist == 'uniform':
        pdf = sps.uniform(loc=par1, scale=par2)
    elif dist == 'lognormal':
        mean  = np.log(par1**2 / np.sqrt(par2 + par1**2))
        sigma = np.sqrt(np.log(par2/par1**2 + 1))
        pdf   = sps.lognorm(sigma, loc=-0.5, scale=np.exp(mean))
    else:
        choicestr = '\n'.join(choices)
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {choicestr}'
        raise NotImplementedError(errormsg)

    return pdf


def set_seed(seed=None):
    '''
    Reset the random seed.
    '''
    if seed is not None:
        seed = int(seed)
    np.random.seed(seed)
    return


__all__ += ['n_binomial', 'binomial_filter', 'binomial_arr', 'n_multinomial',
            'poisson', 'n_poisson', 'n_neg_binomial', 'choose', 'choose_r', 'choose_w',
            'participation_filter']


def n_binomial(prob, n):
    ''' Perform multiple binomial (Bernoulli) trials '''
    return np.random.random(n) < prob


def binomial_filter(prob, arr):
    ''' Binomial filter — return elements of arr that pass coin flip '''
    return arr[(np.random.random(len(arr)) < prob).nonzero()[0]]


def binomial_arr(prob_arr):
    ''' Binomial trials each with different probabilities '''
    return np.random.random(prob_arr.shape) < prob_arr


def n_multinomial(probs, n):
    ''' An array of multinomial trials '''
    return np.searchsorted(np.cumsum(probs), np.random.random(n))


def poisson(rate):
    ''' A Poisson trial '''
    return np.random.poisson(rate, 1)[0]


def n_poisson(rate, n):
    ''' An array of Poisson trials '''
    return np.random.poisson(rate, n)


def n_neg_binomial(rate, dispersion, n, step=1):
    ''' An array of negative binomial trials '''
    nbn_n = dispersion
    nbn_p = dispersion/(rate/step + dispersion)
    samples = np.random.negative_binomial(n=nbn_n, p=nbn_p, size=n)*step
    return samples


def choose(max_n, n):
    ''' Choose a subset of items without replacement '''
    return np.random.choice(max_n, n, replace=False)


def choose_r(max_n, n):
    ''' Choose a subset of items with replacement '''
    return np.random.choice(max_n, n, replace=True)


def choose_w(probs, n, unique=True):
    ''' Choose n items each with a probability from the distribution probs '''
    probs = np.array(probs)
    n_choices = len(probs)
    n_samples = int(n)
    probs_sum = probs.sum()
    if probs_sum:
        probs = probs/probs_sum
    else:
        probs = np.ones(n_choices)/n_choices
    return np.random.choice(n_choices, n_samples, p=probs, replace=not(unique))


def participation_filter(inds, age, layer_probs, bins):
    '''
    Apply age-specific participation filter to eligible individuals.
    '''
    age_bins = np.digitize(age[inds], bins=bins) - 1
    bin_range = np.unique(age_bins)
    participating_inds = np.array([], dtype=int)
    for ab in bin_range:
        these_contacts = binomial_filter(layer_probs[ab], inds[age_bins == ab])
        participating_inds = np.append(participating_inds, these_contacts)
    return participating_inds


__all__ += ['true', 'false', 'defined', 'undefined',
            'itrue', 'ifalse', 'idefined', 'iundefined',
            'itruei', 'ifalsei', 'idefinedi', 'iundefinedi',
            'dtround', 'find_cutoff']


def true(arr):
    ''' Returns the indices of the values of the array that are true '''
    return arr.nonzero()[-1]


def false(arr):
    ''' Returns the indices of the values of the array that are false '''
    return np.logical_not(arr).nonzero()[-1]


def defined(arr):
    ''' Returns the indices of the values of the array that are not-nan '''
    return (~np.isnan(arr)).nonzero()[-1]


def undefined(arr):
    ''' Returns the indices of the values of the array that are nan '''
    return np.isnan(arr).nonzero()[-1]


def itrue(arr, inds):
    ''' Returns the indices that are true in the array '''
    return inds[arr]


def ifalse(arr, inds):
    ''' Returns the indices that are false in the array '''
    return inds[np.logical_not(arr)]


def idefined(arr, inds):
    ''' Returns the indices that are defined in the array '''
    return inds[~np.isnan(arr)]


def iundefined(arr, inds):
    ''' Returns the indices that are undefined in the array '''
    return inds[np.isnan(arr)]


def itruei(arr, inds):
    ''' Returns the indices that are true in the array -- indices[true[indices]] '''
    return inds[arr[inds]]


def ifalsei(arr, inds):
    ''' Returns the indices that are false in the array -- indices[false[indices]] '''
    return inds[np.logical_not(arr[inds])]


def idefinedi(arr, inds):
    ''' Returns the indices that are defined in the array -- indices[defined[indices]] '''
    return inds[~np.isnan(arr[inds])]


def iundefinedi(arr, inds):
    ''' Returns the indices that are undefined in the array -- indices[undefined[indices]] '''
    return inds[np.isnan(arr[inds])]


def dtround(arr, dt, ceil=True):
    ''' Rounds the values in the array to the nearest timestep '''
    if ceil:
        return np.ceil(arr * (1/dt)) / (1/dt)
    else:
        return np.round(arr * (1/dt)) / (1/dt)


def find_cutoff(duration_cutoffs, duration):
    ''' Find which duration bin each ind belongs to '''
    return np.nonzero(duration_cutoffs <= duration)[0][-1]