"""
Create a simulation for running the HPV module.
"""

import starsim as ss
import sciris as sc
import pandas as pd
import hpvsim as hpv
import stisim as sti
import numpy as np
from .data import loaders as hpdata


__all__ = ["Sim"]


class Sim(ss.Sim):
    """
    Custom simulation class for HPV module, inheriting from starsim.Sim
    """
    def __init__(self, pars=None, sim_pars=None, hpv_pars=None, nw_pars=None, imm_pars=None,
                 datafolder=None, location=None,
                 label=None, people=None, demographics=None, diseases=None, networks=None,
                 interventions=None, analyzers=None, connectors=None, **kwargs):

        # Inputs and defaults
        self.location = location
        self.datafolder = datafolder
        self.hpv_pars = None    # Parameters for the HPV modules - processed later
        self.nw_pars = None     # Parameters for the networks - processed later
        self.imm_pars = None    # Parameters for cross-immunity, used in the HPV connector
        self.pars = None        # Parameters for the simulation - processed later
        self.genotypes = ss.ndict()   # Set during init, after processing the genotypes and diseases

        # Call the constructor of the parent class WITHOUT pars or module args
        super().__init__(pars=None, label=label)
        self.pars = hpv.make_sim_pars()  # Make default parameters using values from parameters.py

        # Separate the parameters, storing sim pars now and saving module pars to process in init
        sim_kwargs = dict(label=label, people=people, demographics=demographics, diseases=diseases, networks=networks,
                    interventions=interventions, analyzers=analyzers, connectors=connectors)
        sim_kwargs = {key: val for key, val in sim_kwargs.items() if val is not None}
        updated_pars = self.separate_pars(pars, sim_pars, hpv_pars, nw_pars, imm_pars, sim_kwargs, **kwargs)
        self.pars.update(updated_pars)

        return

    def separate_pars(self, pars=None, sim_pars=None, hpv_pars=None, nw_pars=None, imm_pars=None, sim_kwargs=None, **kwargs):
        """
        Create a nested dict of parameters that get passed to Sim constructor and the component modules
        Prioritization:
            - If any key appears in both pars and *_pars, the value from *_pars will be used.
            - If any key appears in both pars and kwargs, the value from kwargs will be used.
        """
        # Marge in pars and kwargs
        all_pars = sc.mergedicts(pars, sim_pars, hpv_pars, nw_pars, imm_pars, sim_kwargs, kwargs)
        all_pars = self.remap_pars(all_pars)  # Remap any v2 parameters to v3 names

        # Deal with sim pars
        user_sim_pars = {k: v for k, v in all_pars.items() if k in self.pars.keys()}
        for k in user_sim_pars: all_pars.pop(k)
        sim_pars = sc.mergedicts(user_sim_pars, sim_pars, _copy=True)  # Don't merge with defaults, those are set above
        # if sim_pars.get('start'): sim_pars['start'] = ss.date(sim_pars['start'])
        # if sim_pars.get('stop'): sim_pars['stop'] = ss.date(sim_pars['stop'])

        # Deal with HPV pars. Don't merge in defaults yet, this is done
        # during process_genotypes to get the genotype information.
        default_hpv_pars = hpv.make_hpv_pars()
        user_hpv_pars = {}
        for k, v in all_pars.items():
            if k in default_hpv_pars.keys(): user_hpv_pars[k] = v  # Just set
            if sc.checktype(v, dict):  # See whether it contains HPV pars
                user_hpv_pars[k] = {gk: gv for gk, gv in v.items() if gk in default_hpv_pars}
        for k in user_hpv_pars: all_pars.pop(k)
        hpv_pars = sc.mergedicts(user_hpv_pars, hpv_pars, _copy=True)

        # Deal with network pars
        default_nw_pars = hpv.make_network_pars()
        user_nw_pars = {k: v for k, v in all_pars.items() if k in default_nw_pars.keys()}
        for k in user_nw_pars: all_pars.pop(k)
        nw_pars = sc.mergedicts(default_nw_pars, user_nw_pars, nw_pars, _copy=True)

        # Deal with immunity pars
        default_imm_pars = hpv.make_imm_pars()
        user_imm_pars = {k: v for k, v in all_pars.items() if k in default_imm_pars.keys()}
        for k in user_imm_pars: all_pars.pop(k)
        imm_pars = sc.mergedicts(default_imm_pars, user_imm_pars, imm_pars, _copy=True)

        # Raise an exception if there are any leftover pars
        if all_pars:
            raise ValueError(f'Unrecognized parameters: {all_pars.keys()}. Refer to parameters.py for parameters.')

        # Store the parameters for the modules - thse will be fed into the modules during init
        self.hpv_pars = hpv_pars    # Parameters for the HPV modules
        self.nw_pars = nw_pars      # Parameters for the networks
        self.imm_pars = imm_pars    # Parameters for cross-immunity, used in the HPV connector

        return sim_pars

    @staticmethod
    def remap_pars(pars):
        """
        Remap any v2 parameters to v3 names, for backwards compatibility.
        """
        if 'start_year' in pars:
            pars['start'] = pars.pop('start_year')
        if 'end_year' in pars:
            pars['stop'] = pars.pop('end_year')
        if 'seed' in pars:
            pars['rand_seed'] = pars.pop('seed')
        if 'beta' in pars and sc.isnumber(pars['beta']):
            pars['beta_m2f'] = pars.pop('beta')
        if 'location' in pars:
            pars['demographics'] = pars.pop('location')
        return pars

    def init(self, force=False, **kwargs):
        """
        Perform all initializations for the sim
        """
        # Process the genotypes and HPV connector
        genotypes, hpv_connector = self.process_genotypes()
        self.pars['diseases'] += genotypes
        if hpv_connector is not None: self.pars['connectors'] += hpv_connector

        # Process the network
        self.pars['networks'] = self.process_network()

        # Process the demographics
        demographics, people, total_pop = self.process_demographics()
        self.pars['demographics'] += demographics
        self.pars['people'] = people
        self.pars['total_pop'] = total_pop

        super().init(force=force, **kwargs)  # Call the parent init method

        return self

    def process_genotypes(self):
        """
        Process the genotypes to create HPV modules.
        If genotypes are provided, they will be used; otherwise, default to HPV16 and HPV18.
        """
        # Genotypes may be provided in various forms; process them here
        self.pars['genotypes'] = sc.tolist(self.pars['genotypes'])  # Ensure it's a list
        genotypes = sc.autolist()

        # Get the definitive dict of parameters that can be used to construct an HPV module
        # Sort the self.hpv_pars dict into things that can be used for all genotypes, and things that
        # vary by genotype.
        default_hpv_pars = hpv.make_hpv_pars()
        hpv_main_pars = {k: v for k, v in self.hpv_pars.items() if k in default_hpv_pars}
        remaining_pars = {k: v for k, v in self.hpv_pars.items() if k not in default_hpv_pars}

        # Check that the remaining parameters are keyed by genotype, remapping them if needed
        g_options, g_mapping = hpv.get_genotype_choices()
        genotype_pars = {}
        for gparname, gpardict in remaining_pars.items():
            if gparname not in g_mapping.keys():
                raise ValueError(f'Parameters for genotype {gparname} were provided, but this is not an inbuilt genotype')
            else:
                genotype_pars[g_mapping[gparname]] = gpardict

        # Construct or interpret the genotypes from the pars
        for gtype in self.pars['genotypes']:

            if sc.isnumber(gtype): gtype = f'hpv{gtype}'  # Convert e.g. 16 to hpv16

            # If it's a string, convert to a module
            if sc.checktype(gtype, str):
                if gtype not in g_options.keys():
                    errormsg = f'Genotype {gtype} is not one of the inbuilt options.'
                    raise ValueError(errormsg)

                # See if any parameters have been provided for this genotype
                this_gtype_pars = {}
                if gtype in genotype_pars.keys():
                    this_gtype_pars = genotype_pars[gtype]
                hpv_pars = sc.mergedicts(hpv_main_pars, this_gtype_pars)
                genotypes += hpv.make_hpv(genotype=gtype, hpv_pars=hpv_pars)

            elif isinstance(gtype, hpv.Genotype):
                genotypes += gtype
            else:
                raise ValueError(f"Invalid genotype type: {type(gtype)}. Must be str, int, or hpv.HPV.")

        # Store genotypes
        self.genotypes = ss.ndict(genotypes)

        # See if there's a connector added, and add one if not
        # TODO, improve this
        if isinstance(self.pars['connectors'], hpv.HPV): hpv_connector = None
        elif sc.isiterable(self.pars['connectors']):
            hpv_con = [c for c in self.pars['connectors'] if isinstance(c, hpv.HPV)]
            if not hpv_con:
                hpv_connector = hpv.HPV(genotypes=self.genotypes, pars=self.imm_pars)
            else:
                hpv_connector = None
        elif self.pars['connectors'] is None:
            hpv_connector = hpv.HPV(genotypes=self.genotypes, pars=self.imm_pars)

        return genotypes, hpv_connector

    def process_network(self):
        """
        Process the network parameters to create a network module.
        If networks are provided, they will be used; otherwise, use default network (usual case)
        """
        if len(self.pars['networks']):
            # If networks are provided, use them directly
            networks = self.pars['networks']

        else:
            # If no networks are provided, create them based on the network parameters
            networks = ss.ndict(sti.StructuredSexual(pars=self.nw_pars))
        return networks

    def process_demographics(self):
        """ Process the location to create people and demographics if not provided. """

        # If it's a string, do lots of work
        if sc.checktype(self.pars['demographics'], str):
            location = self.pars.pop('demographics')
            self.pars['demographics'] = ss.ndict()
            birth_rates, death_rates = self.get_births_deaths(location)

            # Create modules for births and deaths
            births = ss.Births(birth_rate=birth_rates, metadata=dict(data_cols=dict(year='year', value='cbr')))
            deaths = ss.Deaths(death_rate=death_rates)

            try:
                age_data = hpdata.get_age_distribution(location, year=self.pars.start.year)
                pop_trend = hpdata.get_total_pop(location)
                pop_age_trend = hpdata.get_age_distribution_over_time(location)
                total_pop = int(age_data.value.sum())*1e3
            except ValueError as E:
                warnmsg = f'Could not load age data for requested location "{location}" ({str(E)}); using default'
                raise ValueError(warnmsg) from E

            people = ss.People(self.pars.n_agents, age_data=age_data)
            demographics = [births, deaths]

        else:
            demographics = self.pars['demographics']
            people = self.pars['people']
            total_pop = self.pars['total_pop']

        return demographics, people, total_pop

    @staticmethod
    def get_births_deaths(location, verbose=1, by_sex=True, overall=False):
        '''
        Get mortality and fertility data by location if provided

        Args:
            location (str):  location
            verbose (bool):  whether to print progress
            by_sex   (bool): whether to get sex-specific death rates (default true)
            overall  (bool): whether to get overall values ie not disaggregated by sex (default false)

        Returns:
            lx (dict): dictionary keyed by sex, storing arrays of lx - the number of people who survive to age x
            birth_rates (arr): array of crude birth rates by year
        '''

        if verbose:
            print(f'Loading location-specific demographic data for "{location}"')
        try:
            death_rates = hpdata.get_death_rates(location=location, by_sex=by_sex, overall=overall)
            birth_rates = hpdata.get_birth_rates(location=location)
            return birth_rates, death_rates
        except ValueError as E:
            warnmsg = f'Could not load demographic data for requested location "{location}" ({str(E)})'
            print(warnmsg)
