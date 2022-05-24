'''
Numerical utilities for running hpvsim.
'''

#%% Housekeeping

import numba as nb # For faster computations
import numpy as np # For numerics
import random # Used only for resetting the seed
import sciris as sc # For additional utilities
from .settings import options as hpo # To set options
from . import defaults as hpd # To set default types


# What functions are externally visible -- note, this gets populated in each section below
__all__ = []

# Set dtypes -- note, these cannot be changed after import since Numba functions are precompiled
nbbool  = nb.bool_
nbint   = hpd.nbint
nbfloat = hpd.nbfloat

# Specify whether to allow parallel Numba calculation -- 10% faster for safe and 20% faster for random, but the random number stream becomes nondeterministic for the latter
safe_opts = [1, '1', 'safe']
full_opts = [2, '2', 'full']
safe_parallel = hpo.numba_parallel in safe_opts + full_opts
rand_parallel = hpo.numba_parallel in full_opts
if hpo.numba_parallel not in [0, 1, 2, '0', '1', '2', 'none', 'safe', 'full']:
    errormsg = f'Numba parallel must be "none", "safe", or "full", not "{hpo.numba_parallel}"'
    raise ValueError(errormsg)
cache = hpo.numba_cache # Turning this off can help switching parallelization options


#%% The core functions 

@nb.njit(              (nbbool[:,:],    nbbool[:,:],    nbbool[:]), cache=cache, parallel=safe_parallel)
def get_sources_targets(inf,           sus,            sex):
    ''' Get indices of sources, i.e. people with current infections '''
    f_sus = (sus * ~sex).nonzero()
    m_sus = (sus * sex).nonzero()
    f_inf = (inf * ~sex).nonzero()
    m_inf = (inf * sex).nonzero()
    return f_inf, m_inf, f_sus, m_sus


@nb.njit(parallel=safe_parallel)
def pair_lookup_vals(contacts_array, people_inds, genotypes, n):
    lookup = np.empty(n, nbfloat)
    lookup.fill(np.nan)
    lookup[people_inds[::-1]] = genotypes[::-1]
    res_val = lookup[contacts_array]
    mask = ~np.isnan(res_val)
    return mask, res_val

@nb.njit(parallel=safe_parallel)
def pair_lookup(contacts_array, people_inds, n):
    lookup = np.full(n, False)
    lookup[people_inds[::-1]] = True
    res_val = lookup[contacts_array]
    return res_val

@nb.njit((nbint[:], nb.int64[:]), cache=cache, parallel=safe_parallel)
def isin( arr,      search_inds):
    ''' Find search_inds in arr. Like np.isin() but faster '''
    n = len(arr)
    result = np.full(n, False)
    set_search_inds = set(search_inds)
    for i in nb.prange(n): 
        if arr[i] in set_search_inds:
            result[i] = True
    return result

@nb.njit(   (nbint[:],  nb.int64[:]), cache=cache, parallel=safe_parallel)
def findinds(arr,       vals):
    ''' Finds indices of vals in arr, accounting for repeats '''
    return isin(arr,vals).nonzero()[0]


@nb.njit()
def get_discordant_pairs(f_inf_inds, m_inf_inds, f_sus_inds, m_sus_inds, f, m, n):
    '''
    Construct discordant partnerships
    '''
    f_source_pships, f_genotypes = pair_lookup_vals(f, f_inf_inds[1], f_inf_inds[0], n) # Pull out the indices of partnerships in which the female is infected, as well as the genotypes
    m_source_pships, m_genotypes = pair_lookup_vals(m, m_inf_inds[1], m_inf_inds[0], n) # Pull out the indices of partnerships in which the female is infected, as well as the genotypes
    f_sus_pships = pair_lookup(f, f_sus_inds[1], n) # Pull out the indices of partnerships in which the female is susceptible, as well as the genotypes
    m_sus_pships = pair_lookup(m, m_sus_inds[1], n) # ... same thing for males
    f_genotypes = f_genotypes[(~np.isnan(f_genotypes)*m_sus_pships).nonzero()[0]].astype(hpd.default_int) # Now get the actual genotypes
    m_genotypes = m_genotypes[(~np.isnan(m_genotypes)*f_sus_pships).nonzero()[0]].astype(hpd.default_int) # ... and again for males
    f_source_pships = f_source_pships * m_sus_pships # Remove partnerships where both partners have an infection with the same genotype
    m_source_pships = m_source_pships * f_sus_pships # ... same thing for males
    f_source_inds = f_source_pships.nonzero()[0] # Indices of partnerships where the female has an infection
    m_source_inds = m_source_pships.nonzero()[0] # Indices of partnerships where the male has an infection and the female does not
    return f_source_inds, f_genotypes, m_source_inds, m_genotypes


@nb.njit(             (nbfloat[:],  nbint[:]), cache=cache, parallel=safe_parallel)
def compute_infections(betas,       targets):
    '''
    Compute who infects whom
    '''
    # Determine transmissions
    transmissions   = (np.random.random(len(betas)) < betas).nonzero()[0] # Apply probabilities to determine partnerships in which transmission occurred
    target_inds     = targets[transmissions] # Extract indices of those who got infected
    return target_inds


@nb.njit(          (nbfloat[:,:],   nbint,  nbint[:,:],  nbint[:],  nbfloat[:], nbfloat[:,:]), cache=cache)
def update_immunity(imm,            t,      t_imm_event, inds,      imm_kin,    peak_imm):
    '''
    Step immunity levels forward in time
    '''
    ss              = t_imm_event[:, inds].shape
    t_since_boost   = (t - t_imm_event[:,inds]).ravel()
    current_imm     = imm_kin[t_since_boost].reshape(ss) # Get people's current level of immunity
    imm[:,inds]     = current_imm*peak_imm[:,inds] # Set immunity relative to peak
    return imm


@nb.njit((nbint[:], nbint[:], nb.int64[:]), cache=cache)
def find_contacts(p1, p2, inds): # pragma: no cover
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



#%% Sampling and seed methods

__all__ += ['sample', 'get_pdf', 'set_seed']


def sample(dist=None, par1=None, par2=None, size=None, **kwargs):
    '''
    Draw a sample from the distribution specified by the input. The available
    distributions are:

    - 'uniform'       : uniform distribution from low=par1 to high=par2; mean is equal to (par1+par2)/2
    - 'normal'        : normal distribution with mean=par1 and std=par2
    - 'lognormal'     : lognormal distribution with mean=par1 and std=par2 (parameters are for the lognormal distribution, *not* the underlying normal distribution)
    - 'normal_pos'    : right-sided normal distribution (i.e. only positive values), with mean=par1 and std=par2 *of the underlying normal distribution*
    - 'normal_int'    : normal distribution with mean=par1 and std=par2, returns only integer values
    - 'lognormal_int' : lognormal distribution with mean=par1 and std=par2, returns only integer values
    - 'poisson'       : Poisson distribution with rate=par1 (par2 is not used); mean and variance are equal to par1
    - 'neg_binomial'  : negative binomial distribution with mean=par1 and k=par2; converges to Poisson with k=∞
    - 'beta'          : beta distribution with alpha=par1 and beta=par2;

    Args:
        dist (str):   the distribution to sample from
        par1 (float): the "main" distribution parameter (e.g. mean)
        par2 (float): the "secondary" distribution parameter (e.g. std)
        size (int):   the number of samples (default=1)
        kwargs (dict): passed to individual sampling functions

    Returns:
        A length N array of samples

    **Examples**::

        hp.sample() # returns Unif(0,1)
        hp.sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)
        hp.sample(dist='lognormal_int', par1=5, par2=3) # returns a lognormally distributed set of values with mean 5 and std 3

    Notes:
        Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
        function assumes the user wants to specify the mean and std of the lognormal distribution.

        Negative binomial distributions are parameterized with reference to the mean and dispersion parameter k
        (see: https://en.wikipedia.org/wiki/Negative_binomial_distribution). The r parameter of the underlying
        distribution is then calculated from the desired mean and k. For a small mean (~1), a dispersion parameter
        of ∞ corresponds to the variance and standard deviation being equal to the mean (i.e., Poisson). For a
        large mean (e.g. >100), a dispersion parameter of 1 corresponds to the standard deviation being equal to
        the mean.
    '''

    # Some of these have aliases, but these are the "official" names
    choices = [
        'uniform',
        'normal',
        'normal_pos',
        'normal_int',
        'lognormal',
        'lognormal_int',
        'poisson',
        'neg_binomial',
        'beta',
    ]

    # Ensure it's an integer
    if size is not None:
        size = int(size)

    # Compute distribution parameters and draw samples
    # NB, if adding a new distribution, also add to choices above
    if   dist in ['unif', 'uniform']: samples = np.random.uniform(low=par1, high=par2, size=size, **kwargs)
    elif dist in ['norm', 'normal']:  samples = np.random.normal(loc=par1, scale=par2, size=size, **kwargs)
    elif dist == 'normal_pos':        samples = np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs))
    elif dist == 'normal_int':        samples = np.round(np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs)))
    elif dist == 'poisson':           samples = n_poisson(rate=par1, n=size, **kwargs) # Use Numba version below for speed
    elif dist == 'neg_binomial':      samples = n_neg_binomial(rate=par1, dispersion=par2, n=size, **kwargs) # Use custom version below
    elif dist == 'beta':              samples = np.random.beta(a=par1, b=par2, size=size, **kwargs)
    elif dist in ['lognorm', 'lognormal', 'lognorm_int', 'lognormal_int']:
        if par1>0:
            mean  = np.log(par1**2 / np.sqrt(par2**2 + par1**2)) # Computes the mean of the underlying normal distribution
            sigma = np.sqrt(np.log(par2**2/par1**2 + 1)) # Computes sigma for the underlying normal distribution
            samples = np.random.lognormal(mean=mean, sigma=sigma, size=size, **kwargs)
        else:
            samples = np.zeros(size)
        if '_int' in dist:
            samples = np.round(samples)
    else:
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {sc.newlinejoin(choices)}'
        raise NotImplementedError(errormsg)

    return samples



def get_pdf(dist=None, par1=None, par2=None):
    '''
    Return a probability density function for the specified distribution. This
    is used for example by test_num to retrieve the distribution of times from
    symptom-to-swab for testing. For example, for Washington State, these values
    are dist='lognormal', par1=10, par2=170.
    '''
    import scipy.stats as sps # Import here since slow

    choices = [
        'none',
        'uniform',
        'lognormal',
    ]

    if dist in ['None', 'none', None]:
        return None
    elif dist == 'uniform':
        pdf = sps.uniform(loc=par1, scale=par2)
    elif dist == 'lognormal':
        mean  = np.log(par1**2 / np.sqrt(par2 + par1**2)) # Computes the mean of the underlying normal distribution
        sigma = np.sqrt(np.log(par2/par1**2 + 1)) # Computes sigma for the underlying normal distribution
        pdf   = sps.lognorm(sigma, loc=-0.5, scale=np.exp(mean))
    else:
        choicestr = '\n'.join(choices)
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {choicestr}'
        raise NotImplementedError(errormsg)

    return pdf


def set_seed(seed=None):
    '''
    Reset the random seed -- complicated because of Numba, which requires special
    syntax to reset the seed. This function also resets Python's built-in random
    number generated.

    Args:
        seed (int): the random seed
    '''

    @nb.njit((nbint,), cache=cache)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed) # If None, reinitializes it
    if seed is None: # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)
    random.seed(seed) # Finally, reset Python's built-in random number generator, just in case (used by SynthPops)

    return


#%% Probabilities -- mostly not jitted since performance gain is minimal

__all__ += ['n_binomial', 'binomial_filter', 'binomial_arr', 'n_multinomial',
            'poisson', 'n_poisson', 'n_neg_binomial', 'choose', 'choose_r', 'choose_w']

def n_binomial(prob, n):
    '''
    Perform multiple binomial (Bernolli) trials

    Args:
        prob (float): probability of each trial succeeding
        n (int): number of trials (size of array)

    Returns:
        Boolean array of which trials succeeded

    **Example**::

        outcomes = hp.n_binomial(0.5, 100) # Perform 100 coin-flips
    '''
    return np.random.random(n) < prob


def binomial_filter(prob, arr): # No speed gain from Numba
    '''
    Binomial "filter" -- the same as n_binomial, except return
    the elements of arr that succeeded.

    Args:
        prob (float): probability of each trial succeeding
        arr (array): the array to be filtered

    Returns:
        Subset of array for which trials succeeded

    **Example**::

        inds = hp.binomial_filter(0.5, np.arange(20)**2) # Return which values out of the (arbitrary) array passed the coin flip
    '''
    return arr[(np.random.random(len(arr)) < prob).nonzero()[0]]


def binomial_arr(prob_arr):
    '''
    Binomial (Bernoulli) trials each with different probabilities.

    Args:
        prob_arr (array): array of probabilities

    Returns:
         Boolean array of which trials on the input array succeeded

    **Example**::

        outcomes = hp.binomial_arr([0.1, 0.1, 0.2, 0.2, 0.8, 0.8]) # Perform 6 trials with different probabilities
    '''
    return np.random.random(len(prob_arr)) < prob_arr


def n_multinomial(probs, n): # No speed gain from Numba
    '''
    An array of multinomial trials.

    Args:
        probs (array): probability of each outcome, which usually should sum to 1
        n (int): number of trials

    Returns:
        Array of integer outcomes

    **Example**::

        outcomes = hp.multinomial(np.ones(6)/6.0, 50)+1 # Return 50 die-rolls
    '''
    return np.searchsorted(np.cumsum(probs), np.random.random(n))


@nb.njit((nbfloat,), cache=cache, parallel=rand_parallel) # Numba hugely increases performance
def poisson(rate):
    '''
    A Poisson trial.

    Args:
        rate (float): the rate of the Poisson process

    **Example**::

        outcome = hp.poisson(100) # Single Poisson trial with mean 100
    '''
    return np.random.poisson(rate, 1)[0]


@nb.njit((nbfloat, nbint), cache=cache, parallel=rand_parallel) # Numba hugely increases performance
def n_poisson(rate, n):
    '''
    An array of Poisson trials.

    Args:
        rate (float): the rate of the Poisson process (mean)
        n (int): number of trials

    **Example**::

        outcomes = hp.n_poisson(100, 20) # 20 Poisson trials with mean 100
    '''
    return np.random.poisson(rate, n)


def n_neg_binomial(rate, dispersion, n, step=1): # Numba not used due to incompatible implementation
    '''
    An array of negative binomial trials. See hp.sample() for more explanation.

    Args:
        rate (float): the rate of the process (mean, same as Poisson)
        dispersion (float):  dispersion parameter; lower is more dispersion, i.e. 0 = infinite, ∞ = Poisson
        n (int): number of trials
        step (float): the step size to use if non-integer outputs are desired

    **Example**::

        outcomes = hp.n_neg_binomial(100, 1, 50) # 50 negative binomial trials with mean 100 and dispersion roughly equal to mean (large-mean limit)
        outcomes = hp.n_neg_binomial(1, 100, 20) # 20 negative binomial trials with mean 1 and dispersion still roughly equal to mean (approximately Poisson)
    '''
    nbn_n = dispersion
    nbn_p = dispersion/(rate/step + dispersion)
    samples = np.random.negative_binomial(n=nbn_n, p=nbn_p, size=n)*step
    return samples


@nb.njit((nbint, nbint), cache=cache) # Numba hugely increases performance
def choose(max_n, n):
    '''
    Choose a subset of items (e.g., people) without replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        choices = hp.choose(5, 2) # choose 2 out of 5 people with equal probability (without repeats)
    '''
    return np.random.choice(max_n, n, replace=False)


@nb.njit((nbint, nbint), cache=cache) # Numba hugely increases performance
def choose_r(max_n, n):
    '''
    Choose a subset of items (e.g., people), with replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        choices = hp.choose_r(5, 10) # choose 10 out of 5 people with equal probability (with repeats)
    '''
    return np.random.choice(max_n, n, replace=True)


def choose_w(probs, n, unique=True): # No performance gain from Numba
    '''
    Choose n items (e.g. people), each with a probability from the distribution probs.

    Args:
        probs (array): list of probabilities, should sum to 1
        n (int): number of samples to choose
        unique (bool): whether or not to ensure unique indices

    **Example**::

        choices = hp.choose_w([0.2, 0.5, 0.1, 0.1, 0.1], 2) # choose 2 out of 5 people with nonequal probability.
    '''
    probs = np.array(probs)
    n_choices = len(probs)
    n_samples = int(n)
    probs_sum = probs.sum()
    if probs_sum: # Weight is nonzero, rescale
        probs = probs/probs_sum
    else: # Weights are all zero, choose uniformly
        probs = np.ones(n_choices)/n_choices
    return np.random.choice(n_choices, n_samples, p=probs, replace=not(unique))



#%% Simple array operations

__all__ += ['true',   'false',   'defined',   'undefined',
            'itrue',  'ifalse',  'idefined',  'iundefined',
            'itruei', 'ifalsei', 'idefinedi', 'iundefinedi',
            'dtround', 'find_cutoff']


def true(arr):
    '''
    Returns the indices of the values of the array that are true: just an alias
    for arr.nonzero()[0].

    Args:
        arr (array): any array

    **Example**::

        inds = hp.true(np.array([1,0,0,1,1,0,1])) # Returns array([0, 3, 4, 6])
    '''
    return arr.nonzero()[0]


def false(arr):
    '''
    Returns the indices of the values of the array that are false.

    Args:
        arr (array): any array

    **Example**::

        inds = hp.false(np.array([1,0,0,1,1,0,1]))
    '''
    return np.logical_not(arr).nonzero()[0]


def defined(arr):
    '''
    Returns the indices of the values of the array that are not-nan.

    Args:
        arr (array): any array

    **Example**::

        inds = hp.defined(np.array([1,np.nan,0,np.nan,1,0,1]))
    '''
    return (~np.isnan(arr)).nonzero()[0]


def undefined(arr):
    '''
    Returns the indices of the values of the array that are not-nan.

    Args:
        arr (array): any array

    **Example**::

        inds = hp.defined(np.array([1,np.nan,0,np.nan,1,0,1]))
    '''
    return np.isnan(arr).nonzero()[0]


def itrue(arr, inds):
    '''
    Returns the indices that are true in the array -- name is short for indices[true]

    Args:
        arr (array): a Boolean array, used as a filter
        inds (array): any other array (usually, an array of indices) of the same size

    **Example**::

        inds = hp.itrue(np.array([True,False,True,True]), inds=np.array([5,22,47,93]))
    '''
    return inds[arr]


def ifalse(arr, inds):
    '''
    Returns the indices that are true in the array -- name is short for indices[false]

    Args:
        arr (array): a Boolean array, used as a filter
        inds (array): any other array (usually, an array of indices) of the same size

    **Example**::

        inds = hp.ifalse(np.array([True,False,True,True]), inds=np.array([5,22,47,93]))
    '''
    return inds[np.logical_not(arr)]


def idefined(arr, inds):
    '''
    Returns the indices that are defined in the array -- name is short for indices[defined]

    Args:
        arr (array): any array, used as a filter
        inds (array): any other array (usually, an array of indices) of the same size

    **Example**::

        inds = hp.idefined(np.array([3,np.nan,np.nan,4]), inds=np.array([5,22,47,93]))
    '''
    return inds[~np.isnan(arr)]


def iundefined(arr, inds):
    '''
    Returns the indices that are undefined in the array -- name is short for indices[undefined]

    Args:
        arr (array): any array, used as a filter
        inds (array): any other array (usually, an array of indices) of the same size

    **Example**::

        inds = hp.iundefined(np.array([3,np.nan,np.nan,4]), inds=np.array([5,22,47,93]))
    '''
    return inds[np.isnan(arr)]



def itruei(arr, inds):
    '''
    Returns the indices that are true in the array -- name is short for indices[true[indices]]

    Args:
        arr (array): a Boolean array, used as a filter
        inds (array): an array of indices for the original array

    **Example**::

        inds = hp.itruei(np.array([True,False,True,True,False,False,True,False]), inds=np.array([0,1,3,5]))
    '''
    return inds[arr[inds]]


def ifalsei(arr, inds):
    '''
    Returns the indices that are false in the array -- name is short for indices[false[indices]]

    Args:
        arr (array): a Boolean array, used as a filter
        inds (array): an array of indices for the original array

    **Example**::

        inds = hp.ifalsei(np.array([True,False,True,True,False,False,True,False]), inds=np.array([0,1,3,5]))
    '''
    return inds[np.logical_not(arr[inds])]


def idefinedi(arr, inds):
    '''
    Returns the indices that are defined in the array -- name is short for indices[defined[indices]]

    Args:
        arr (array): any array, used as a filter
        inds (array): an array of indices for the original array

    **Example**::

        inds = hp.idefinedi(np.array([4,np.nan,0,np.nan,np.nan,4,7,4,np.nan]), inds=np.array([0,1,3,5]))
    '''
    return inds[~np.isnan(arr[inds])]


def iundefinedi(arr, inds):
    '''
    Returns the indices that are undefined in the array -- name is short for indices[defined[indices]]

    Args:
        arr (array): any array, used as a filter
        inds (array): an array of indices for the original array

    **Example**::

        inds = hp.iundefinedi(np.array([4,np.nan,0,np.nan,np.nan,4,7,4,np.nan]), inds=np.array([0,1,3,5]))
    '''
    return inds[np.isnan(arr[inds])]


def dtround(arr, dt, ceil=True):
    '''
    Rounds the values in the array to the nearest timestep

    Args:
        arr (array): any array
        dt  (float): float, usually representing a timestep in years

    **Example**::

        dtround = hp.dtround(np.array([0.23,0.61,20.53])) # Returns array([0.2, 0.6, 20.6])
        dtround = hp.dtround(np.array([0.23,0.61,20.53]),ceil=True) # Returns array([0.4, 0.8, 20.6])
    '''
    if ceil:
        return np.ceil(arr * (1/dt)) / (1/dt)
    else:
        return np.round(arr * (1/dt)) / (1/dt)


def find_cutoff(duration_cutoffs, duration):
    '''
    Find which duration bin each ind belongs to.
    '''
    return np.nonzero(duration_cutoffs <= duration)[0][-1]  # Index of the duration bin to use