'''
Defines classes and methods for hiv natural history
'''

import numpy as np
import sciris as sc
import pandas as pd
from . import utils as hpu
from . import defaults as hpd
from . import base as hpb
from .data import loaders as hpdata


class HIVsim(hpb.ParsObj):
    '''
        A class based around performing operations on a self.pars dict.
        '''

    def __init__(self, sim, location, art_datafile, hiv_datafile, hiv_pars=None):
        pars = self.load_data(location=location, hiv_datafile=hiv_datafile, art_datafile=art_datafile)

        # Define default parameters, can be overwritten by hiv_pars
        pars['hiv_pars'] = {
            'rel_sus': 2.2,  # Increased risk of acquiring HPV
            'rel_hiv_sev_infl': 0.5,  # Speed up growth of disease severity
            'reactivation_prob': 3, # Unused for now, TODO: add in rel_reactivation to make functional
        }

        if hiv_pars is not None:
            pars = sc.mergedicts(pars, hiv_pars)
        self.update_pars(pars, create=True)
        self.init_results(sim)

        return


    def init_results(self, sim):

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = hpb.Result(*args, **kwargs, npts=sim.res_npts)
            return output

        self.resfreq = sim.resfreq
        # Initialize storage
        results = sc.objdict()

        na = len(sim['age_bins']) - 1  # Number of age bins

        results['hiv_infections'] = init_res('Number HIV infections')
        results['hiv_infections_by_age'] = init_res('Number HIV infections by age', n_rows=na)
        results['n_hiv'] = init_res('Number living with HIV')
        results['n_hiv_by_age'] = init_res('Number living with HIV by age', n_rows=na)
        results['hiv_prevalence'] = init_res('HIV prevalence')
        results['hiv_prevalence_by_age'] = init_res('HIV prevalence by age', n_rows=na)
        results['hiv_incidence'] = init_res('HIV incidence')
        results['hiv_incidence_by_age'] = init_res('HIV incidence by age', n_rows=na)
        self.results = results
        return

    # %% HIV methods

    def set_hiv_prognoses(self, people, inds, year=None):
        ''' Set HIV outcomes (for now only ART) '''

        art_cov = self['art_adherence']  # Shorten

        # Extract index of current year
        all_years = np.array(list(art_cov.keys()))
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]

        # Figure out which age bin people belong to
        age_bins = art_cov[nearest_year][0, :]
        age_inds = np.digitize(people.age[inds], age_bins)

        # Apply ART coverage by age to people
        art_covs = art_cov[nearest_year][1, :]
        art_adherence = art_covs[age_inds]
        people.art_adherence[inds] = art_adherence
        people.rel_sev_infl[inds] = (1 - art_adherence) * self['hiv_pars']['rel_hiv_sev_infl']
        people.rel_sus[inds] = (1 - art_adherence) * self['hiv_pars']['rel_sus']
        return

    def apply(self, people, year=None):
        '''
        Wrapper method that checks for new HIV infections, updates prognoses, etc.
        '''

        new_infection_inds = self.new_hiv_infections(people, year) # Newly acquired HIV infections
        self.set_hiv_prognoses(people, new_infection_inds, year=year)  # Set ART adherence for those with HIV
        self.update_hpv_progs(people, new_infection_inds) # Update any HPV prognoses
        new_infections = people.scale_flows(new_infection_inds) # Return scaled number of infections
        return new_infections

    def new_hiv_infections(self, people, year=None):
        '''Apply HIV infection rates to population'''
        hiv_pars = self['infection_rates']
        all_years = np.array(list(hiv_pars.keys()))
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]
        hiv_year = hiv_pars[nearest_year]
        dt = people.pars['dt']

        hiv_probs = np.zeros(len(people), dtype=hpd.default_float)
        for sk in ['f', 'm']:
            hiv_year_sex = hiv_year[sk]
            age_bins = hiv_year_sex[:, 0]
            hiv_rates = hiv_year_sex[:, 1] * dt
            mf_inds = people.is_female if sk == 'f' else people.is_male
            mf_inds *= people.alive  # Only include people alive
            age_inds = np.digitize(people.age[mf_inds], age_bins)
            hiv_probs[mf_inds] = hiv_rates[age_inds]
        hiv_probs[people.hiv] = 0  # not at risk if already infected

        # Get indices of people who acquire HIV
        hiv_inds = hpu.true(hpu.binomial_arr(hiv_probs))
        people.hiv[hiv_inds] = True

        if people.t % self.resfreq == self.resfreq - 1:
            # Update stock and flows
            idx = int(people.t / self.resfreq)
            self.results['hiv_infections'][idx] = people.scale_flows(hiv_inds)
            self.results[f'hiv_infections_by_age'][:, idx] = np.histogram(people.age[hiv_inds], bins=people.age_bins, weights=people.scale[hiv_inds])[0]
            self.results['n_hiv'][idx] = people.count('hiv')
            hivinds = hpu.true(people['hiv'])
            self.results[f'n_hiv_by_age'][:, idx] = np.histogram(people.age[hivinds], bins=people.age_bins, weights=people.scale[hivinds])[0]

        return hiv_inds

    def update_hpv_progs(self, people, hiv_inds):
        dt = people.pars['dt']
        for g in range(people.pars['n_genotypes']):
            gpars = people.pars['genotype_pars'][people.pars['genotype_map'][g]]
            hpv_inds = hpu.itruei((people.is_female & people.episomal[g, :]), hiv_inds)  # Women with HIV who have episomal HPV
            if len(hpv_inds):  # Reevaluate these women's severity markers and determine whether they will develop cellular changes
                people.set_severity_pars(hpv_inds, g, gpars)
                people.set_severity(hpv_inds, g, gpars, dt)
        return

    def get_hiv_data(self, location=None, hiv_datafile=None, art_datafile=None, verbose=False):
        '''
        Load HIV incidence and art coverage data, if provided
        ART adherance calculations use life expectancy data to infer lifetime average coverage
        rates for people in different age buckets. To give an example, suppose that ART coverage
        over 2010-2020 is given by:
            art_coverage = [0.23,0.3,0.38,0.43,0.48,0.52,0.57,0.61,0.65,0.68,0.72]
        The average ART adherence in 2010 will be higher for younger cohorts than older ones.
        Someone expected to die within a year would be given an average lifetime ART adherence
        value of 0.23, whereas someone expected to survive >10 years would be given a value of 0.506.

        Args:
            location (str): must be provided if you want to run with HIV dynamics
            hiv_datafile (str):  must be provided if you want to run with HIV dynamics
            art_datafile (str):  must be provided if you want to run with HIV dynamics
            verbose (bool):  whether to print progress

        Returns:
            hiv_inc (dict): dictionary keyed by sex, storing arrays of HIV incidence over time by age
            art_cov (dict): dictionary keyed by sex, storing arrays of ART coverage over time by age
            life_expectancy (dict): dictionary storing life expectancy over time by age
        '''

        if hiv_datafile is None and art_datafile is None:
            hiv_incidence_rates, art_adherence = None, None

        else:

            # Load data
            life_exp = self.get_life_expectancy(location=location,
                                           verbose=verbose)  # Load the life expectancy data (needed for ART adherance calcs)
            df_inc = pd.read_csv(hiv_datafile)  # HIV incidence
            df_art = pd.read_csv(art_datafile)  # ART coverage

            # Process HIV and ART data
            sex_keys = ['Male', 'Female']
            sex_key_map = {'Male': 'm', 'Female': 'f'}

            ## Start with incidence file
            years = df_inc['Year'].unique()
            hiv_incidence_rates = dict()

            # Processing
            for year in years:
                hiv_incidence_rates[year] = dict()
                for sk in sex_keys:
                    sk_out = sex_key_map[sk]
                    hiv_incidence_rates[year][sk_out] = np.concatenate(
                        [
                            np.array(df_inc[(df_inc['Year'] == year) & (df_inc['Sex'] == sk_out)][['Age', 'Incidence']],
                                     dtype=hpd.default_float),
                            np.array([[150, 0]])  # Add another entry so that all older age groups are covered
                        ]
                    )

            # Now compute ART adherence over time/age
            art_adherence = dict()
            years = df_art['Year'].values
            for i, year in enumerate(years):

                # Use the incidence file to determine which age groups we want to calculate ART coverage for
                ages_inc = hiv_incidence_rates[year]['m'][:, 0]  # Read in the age groups we have HIV incidence data for
                ages_ex = life_exp[year]['m'][:, 0]  # Age groups available in life expectancy file
                ages = np.intersect1d(ages_inc, ages_ex)  # Age groups we want to calculate ART coverage for

                # Initialize age-specific ART coverage dict and start filling it in
                cov = np.zeros(len(ages), dtype=hpd.default_float)
                for j, age in enumerate(ages):
                    idx = np.where(life_exp[year]['f'][:, 0] == age)[0]  # Finding life expectancy for this age group/year
                    this_life_exp = life_exp[year]['f'][idx, 1]  # Pull out value
                    last_year = int(year + this_life_exp)  # Figure out the year in which this age cohort is expected to die
                    year_ind = sc.findnearest(years,
                                              last_year)  # Get as close to the above year as possible within the data
                    if year_ind > i:  # Either take the mean of ART coverage from now up until the year of death
                        cov[j] = np.mean(df_art[i:year_ind]['ART Coverage'].values)
                    else:  # Or, just use ART overage in this year
                        cov[j] = df_art.iloc[year_ind]['ART Coverage']

                art_adherence[year] = np.array([ages, cov])

        return hiv_incidence_rates, art_adherence

    def load_data(self, location=None, hiv_datafile=None, art_datafile=None):
        ''' Load any data files that are used to create additional parameters, if provided '''
        hiv_data = sc.objdict()
        hiv_data.infection_rates, hiv_data.art_adherence = self.get_hiv_data(location=location, hiv_datafile=hiv_datafile, art_datafile=art_datafile)
        return hiv_data

    def get_life_expectancy(self, location, verbose=False):
        '''
        Get life expectancy data by location
        life_expectancy (dict): dictionary storing life expectancy over time by age
        '''
        if location is not None:
            if verbose:
                print(f'Loading location-specific life expectancy data for "{location}" - needed for HIV runs')
            try:
                life_expectancy = hpdata.get_life_expectancy(location=location)
                return life_expectancy
            except ValueError as E:
                errormsg = f'Could not load HIV data for requested location "{location}" ({str(E)})'
                raise NotImplementedError(errormsg)
        else:
            raise NotImplementedError('Cannot load HIV data without a specified location')

    def finalize(self, sim):
        '''
        Compute prevalence, incidence.
        '''
        res = self.results
        simres = sim.results

        # Compute HIV incidence and prevalence
        def safedivide(num, denom):
            ''' Define a variation on sc.safedivide that respects shape of numerator '''
            answer = np.zeros_like(num)
            fill_inds = (denom != 0).nonzero()
            if len(num.shape) == len(denom.shape):
                answer[fill_inds] = num[fill_inds] / denom[fill_inds]
            else:
                answer[:, fill_inds] = num[:, fill_inds] / denom[fill_inds]
            return answer

        self.results['hiv_prevalence_by_age'][:] = safedivide(res['n_hiv_by_age'][:], simres['n_alive_by_age'][:])
        self.results['hiv_incidence'][:] = sc.safedivide(res['hiv_infections'][:], (simres['n_alive'][:] - res['n_hiv'][:]))
        self.results['hiv_incidence_by_age'][:] = sc.safedivide(res['hiv_infections_by_age'][:], (simres['n_alive_by_age'][:] - res['n_hiv_by_age'][:]))
        self.results['hiv_prevalence'][:] = sc.safedivide(res['n_hiv'][:], simres['n_alive'][:])
        return