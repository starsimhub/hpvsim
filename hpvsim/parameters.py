"""
Set parameters
"""
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv


__all__ = ['SimPars', 'make_sim_pars', 'HPVPars', 'make_hpv_pars', 'make_genotype_pars', 'NetworkPars', 'make_network_pars', 'ImmPars', 'make_imm_pars', 'make_immunity_matrix']


class SimPars(ss.SimPars):
    """
    Subclass of Starsim's SimPars with defaults for HPV simulations. Refer to
    Starsim's SimPars for more information on the parameters.
    """
    def __init__(self, **kwargs):

        # Initialize the parent class
        super().__init__()

        # General parameters
        self.label   = ''  # The label of the simulation
        self.verbose = ss.options.verbose  # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)

        # Population parameters
        self.n_agents  = 10e3  # Number of agents
        self.total_pop = None  # If defined, used for calculating the scale factor
        self.pop_scale = None  # How much to scale the population

        # Simulation parameters
        self.unit      = 'month'    # The time unit to use; options are 'year' (default), 'day', 'week', 'month', or 'none'
        self.start     = ss.date(1990)  # Start of the simulation
        self.stop      = ss.date(2030)  # End of the simulation
        self.dur       = None   # Duration of time to run, if stop isn't specified
        self.dt        = 3      # Timestep (in units of self.unit)
        self.rand_seed = 1      # Random seed; if None, don't reset

        # Demographic parameters
        self.birth_rate = 20
        self.death_rate = 15
        self.use_aging  = True  # True if demographics, false otherwise

        # Disease parameters
        self.genotypes = [16, 18]  # HPV genotypes to include in the simulation; can be a list of integers or strings

        # Misc other parameters and settings
        # World Standard Population, used to calculate age-standardised rates (ASR) of incidence
        self.age_bin_edges = np.array([0,   5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75, 80, 85, 100])
        self.standard_pop_weights = np.array([.12, .10, .09, .09, .08, .08, .06, .06, .06, .06, .05, .04, .04, .03, .02, .01, 0.005, 0.005, 0])
        self.standard_pop = np.array([self.age_bin_edges, self.standard_pop_weights])

        # Update with any supplied parameter values and generate things that need to be generated
        self.update(kwargs)
        return


class HPVPars(ss.Pars):
    """
    Subclass of Starsim's SimPars with defaults for HPV simulations. Refer to
    Starsim's SimPars for more information on the parameters.
    """
    def __init__(self, genotype=None, **kwargs):

        # Initialize the parent class
        super().__init__()

        # Initial conditions
        self.init_prev = ss.bernoulli(p=0.05)

        # Transmission parameters
        self.beta = 0  # Constructed by the STI class using the m2f and f2m parameters
        self.beta_m2f = 1
        self.rel_beta_f2m = 0.27
        self.beta_m2c = 0
        self.eff_condom = 0.5

        # Disease progression parameters
        self.dur_cancer = ss.lognorm_ex(ss.dur(8, "year"), ss.dur(3, "year"))
        self.dur_infection_male = ss.lognorm_ex(ss.dur(1, "year"), ss.dur(1, "year"))
        self.inf_imm = hpv.beta_mean(par1=0.35, par2=0.025)  # nAB-like immunity to reinfection with this genotype
        self.cell_imm = hpv.beta_mean(par1=0.25, par2=0.025)  # T-cell-like immunity following reinfection

        # Genotype-specific parameters
        self.rel_beta = 1
        self.dur_precin = None    # Set for individual genotypes by derived classes
        self.cin_fn = None        # Set for individual genotypes by derived classes
        self.dur_cin = None       # Set for individual genotypes by derived classes
        self.cancer_fn = None       # Set for individual genotypes by derived classes
        self.cin_prob = ss.bernoulli(p=0)     # placeholder, gets reset
        self.cancer_prob = ss.bernoulli(p=0)  # placeholder, gets reset
        self.sero_prob = ss.bernoulli(p=0.0)

        # Latency parameters
        self.p_control = ss.bernoulli(p=0.0)  # Probability of controlling HPV infection
        self.p_reactivate = ss.bernoulli(p=0.005)  # Per-timestep probability of reactivating; unused unless p_control>0

        # Update the values above with genotype values
        if genotype is not None:
            gen_kwargs = make_genotype_pars(genotype)
            self.update(gen_kwargs)

        self.include_care = False  # Turn off default STI module behavior of storing care outcomes (TODO reconsider)

        # Update with any supplied parameter values
        self.update(kwargs)
        return


class GenotypePars(ss.Pars):
    def __init__(self, **kwargs):
        super().__init__()
        return


def make_genotype_pars(gkey=None):
    genotype_pars = sc.objdict(
        hpv16=dict(
            rel_beta=1.0,
            dur_precin=ss.lognorm_ex(ss.dur(3, "year"), ss.dur(9, "year")),
            dur_cin=ss.lognorm_ex(ss.dur(5, "year"), ss.dur(20, "year")),
            cin_fn=sc.objdict(k=0.3, ttc=50),
            cancer_fn=sc.objdict(method='cin_integral', transform_prob=2e-3), # Map CIN duration to cancer probability
            sero_prob = ss.bernoulli(p=0.75),
        ),
        hpv18=dict(
            rel_beta=0.75,
            dur_precin=ss.lognorm_ex(ss.dur(2.5, "year"), ss.dur(9, "year")),
            dur_cin=ss.lognorm_ex(ss.dur(5, "year"), ss.dur(20, "year")),
            cin_fn=sc.objdict(k=0.25, ttc=50),
            cancer_fn=sc.objdict(method='cin_integral', transform_prob=2e-3),  # Map CIN duration to cancer probability
            sero_prob = ss.bernoulli(p=0.56),
        ),
        hi5=dict(
            rel_beta=0.9,
            dur_precin=ss.lognorm_ex(ss.dur(2.5, "year"), ss.dur(9, "year")),
            dur_cin=ss.lognorm_ex(ss.dur(4.5, "year"), ss.dur(20, "year")),
            cin_fn=sc.objdict(k=0.2, ttc=50),
            cancer_fn=sc.objdict(method='cin_integral', transform_prob=1.5e-3),  # Map CIN duration to cancer probability
            sero_prob = ss.bernoulli(p=0.6),
        ),
        ohr=dict(
            rel_beta=0.9,
            dur_precin=ss.lognorm_ex(ss.dur(2.5, "year"), ss.dur(9, "year")),
            dur_cin=ss.lognorm_ex(ss.dur(4.5, "year"), ss.dur(20, "year")),
            cin_fn=sc.objdict(k=0.2, ttc=50),
            cancer_fn=sc.objdict(method='cin_integral', transform_prob=1.5e-3),  # Map CIN duration to cancer probability
            sero_prob = ss.bernoulli(p=0.6),
        ),
    )
    if gkey is None:
        return genotype_pars
    else:
        if gkey not in genotype_pars:
            raise ValueError(f"Unknown genotype key: {gkey}. Available keys are {list(genotype_pars.keys())}.")
        return genotype_pars[gkey]


class NetworkPars(ss.Pars):
    def __init__(self, **kwargs):
        self.update(kwargs)
        return


class ImmPars(ss.Pars):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_imm_sus_med = 0.3
        self.cross_imm_sus_high = 0.5
        self.cross_imm_sev_med = 0.5
        self.cross_imm_sev_high = 0.7
        self.sus_imm_matrix = None
        self.sev_imm_matrix = None

        # Remove / overwrite default values that are not relevant for HPV connector
        # TODO reconsider
        self.include_care = False
        self.log = False
        self.init_prev = None

        self.update(kwargs)
        return


def make_sim_pars(**kwargs):
    """ Shortcut for making a new instance of SimPars """
    return SimPars(**kwargs)


def make_hpv_pars(**kwargs):
    """ Shortcut for making a new instance of SimPars """
    return HPVPars(**kwargs)


def make_network_pars(**kwargs):
    """ Shortcut for making a new instance of NetworkPars """
    return NetworkPars(**kwargs)


def make_imm_pars(**kwargs):
    """ Shortcut for making a new instance of NetworkPars """
    return ImmPars(**kwargs)


def make_immunity_matrix(gnames=None, cross_imm_med=0.3, cross_imm_high=0.5, as_matrix=True):
    """
    Create a cross-immunity function based on the provided parameters.
    """
    imm_dict = dict(
        # All values based roughly on https://academic.oup.com/jnci/article/112/10/1030/5753954 or assumptions
        hpv16=dict(
            hpv16=1.0,  # Default for own-immunity
            hpv18=cross_imm_high,
            hi5=cross_imm_med,
            ohr=cross_imm_med,
            hr=cross_imm_med,
            lr=cross_imm_med,
        ),
        hpv18=dict(
            hpv16=cross_imm_high,
            hpv18=1.0,  # Default for own-immunity
            hi5=cross_imm_med,
            ohr=cross_imm_med,
            hr=cross_imm_med,
            lr=cross_imm_med,
        ),
        hi5=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hi5=1,
            ohr=cross_imm_med,
            hr=cross_imm_med,
            lr=cross_imm_med,
        ),
        ohr=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hi5=cross_imm_med,
            ohr=1,
            hr=cross_imm_med,
            lr=cross_imm_med,
        ),
        lr=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hi5=cross_imm_med,
            ohr=cross_imm_med,
            hr=cross_imm_med,
            lr=1,
        ),
    )

    if gnames is not None:
        # If gnames is provided, filter the default_pars to only include those genotypes
        imm_dict = {g: {h: imm_dict[g][h] for h in gnames} for g in gnames}

    if as_matrix:
        # Convert the dictionary to a matrix format
        keys = imm_dict.keys()
        return np.array([[imm_dict[row][col] for col in keys] for row in keys])
    else:
        # Return the dictionary format
        return imm_dict
