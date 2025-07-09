"""
Set parameters
"""
import numpy as np
import starsim as ss


__all__ = ['SimPars', 'make_sim_pars', 'HPVPars', 'make_hpv_pars', 'NetworkPars', 'make_network_pars', 'ImmPars', 'make_imm_pars']


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
        self.dur       = None   # Duration of time to run, if stop isn't specified (default 50 steps of self.unit)
        self.dt        = 3      # Timestep (in units of self.unit)
        self.rand_seed = 1      # Random seed; if None, don't reset

        # Demographic parameters
        self.birth_rate = 20
        self.death_rate = 15
        self.use_aging  = True  # True if demographics, false otherwise

        # Disease parameters
        self.genotypes = [16, 18]  # HPV genotypes to include in the simulation; can be a list of integers or strings

        # Update with any supplied parameter values and generate things that need to be generated
        self.update(kwargs)
        return


class HPVPars(ss.Pars):
    """
    Subclass of Starsim's SimPars with defaults for HPV simulations. Refer to
    Starsim's SimPars for more information on the parameters.
    """
    def __init__(self, **kwargs):

        # Initialize the parent class
        super().__init__()

        # Initial conditions
        self.init_prev = ss.bernoulli(p=0.2)

        # Transmission parameters
        self.beta = None  # Constructed by the STI class using the m2f and f2m parameters
        self.beta_m2f = 1
        self.rel_beta_f2m = 0.27
        self.beta_m2c = 0
        self.eff_condom = 0.5

        # Disease progression parameters
        self.dur_cancer = ss.lognorm_ex(ss.dur(8, "year"), ss.dur(3, "year"))
        self.dur_infection_male = ss.lognorm_ex(ss.dur(1, "year"), ss.dur(1, "year"))
        self.sero_prob = ss.bernoulli(p=0.75)
        self.init_imm = 1
        self.init_cell_imm = 1

        # Genotype-specific parameters
        self.rel_beta = 1
        self.dur_precin = None    # Set for individual genotypes by derived classes
        self.cin_fn = None        # Set for individual genotypes by derived classes
        self.dur_cin = None       # Set for individual genotypes by derived classes
        self.cancer_fn = None       # Set for individual genotypes by derived classes
        self.cin_prob = ss.bernoulli(p=0)     # placeholder, gets reset
        self.cancer_prob = ss.bernoulli(p=0)  # placeholder, gets reset

        self.include_care = False  # Temporary...

        # Update with any supplied parameter values
        self.update(kwargs)
        return


class NetworkPars(ss.Pars):
    def __init__(self, **kwargs):
        self.update(kwargs)
        return


class ImmPars(ss.Pars):
    def __init__(self, **kwargs):
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
    return NetworkPars(**kwargs)