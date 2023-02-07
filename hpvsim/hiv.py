'''
Defines classes and methods for hiv natural history
'''

import numpy as np
import sciris as sc
from collections.abc import Iterable
from . import utils as hpu
from . import defaults as hpd
from . import parameters as hppar
from . import interventions as hpi


# %% HIV methods

def set_hiv_prognoses(people, inds, year=None):
    ''' Set HIV outcomes (for now only ART) '''

    art_cov = people.hiv_pars.art_adherence  # Shorten

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
    people.rel_sev_infl[inds] = (1-art_adherence)*people.pars['hiv_pars']['rel_hiv_sev_infl']

    return

def apply_hiv_rates(people, year=None):
    '''
    Apply HIV infection rates to population
    '''
    hiv_pars = people.hiv_pars.infection_rates
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

    # Update prognoses for those with HIV
    if len(hiv_inds):

        set_hiv_prognoses(people, hiv_inds, year=year)  # Set ART adherence for those with HIV

        for g in range(people.pars['n_genotypes']):
            gpars = people.pars['genotype_pars'][people.pars['genotype_map'][g]]
            hpv_inds = hpu.itruei((people.is_female & people.episomal[g, :]), hiv_inds)  # Women with HIV who have episomal HPV
            if len(hpv_inds):  # Reevaluate these women's severity markers and determine whether they will develop cellular changes
                people.set_sev_rates(hpv_inds, g, gpars)
                people.set_sev_outcomes(hpv_inds, g, dt)

    return people.scale_flows(hiv_inds)
