'''
Load data
'''

#%% Housekeeping
import numpy as np
import sciris as sc
from . import country_age_data    as cad
import re

__all__ = ['get_country_aliases', 'map_entries', 'show_locations', 'get_age_distribution', 'get_death_rates']


def get_country_aliases():
    ''' Define aliases for countries with odd names in the data '''
    country_mappings = {
       'Bolivia':        'Bolivia (Plurinational State of)',
       'Burkina':        'Burkina Faso',
       'Cape Verde':     'Cabo Verdeo',
       'Hong Kong':      'China, Hong Kong Special Administrative Region',
       'Macao':          'China, Macao Special Administrative Region',
       "Cote d'Ivore":   'Côte d’Ivoire',
       "Ivory Coast":    'Côte d’Ivoire',
       'DRC':            'Democratic Republic of the Congo',
       'Iran':           'Iran (Islamic Republic of)',
       'Laos':           "Lao People's Democratic Republic",
       'Micronesia':     'Micronesia (Federated States of)',
       'Korea':          'Republic of Korea',
       'South Korea':    'Republic of Korea',
       'Moldova':        'Republic of Moldova',
       'Russia':         'Russian Federation',
       'Palestine':      'State of Palestine',
       'Syria':          'Syrian Arab Republic',
       'Taiwan':         'Taiwan Province of China',
       'Macedonia':      'The former Yugoslav Republic of Macedonia',
       'UK':             'United Kingdom of Great Britain and Northern Ireland',
       'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
       'Tanzania':       'United Republic of Tanzania',
       'USA':            'United States of America',
       'United States':  'United States of America',
       'Venezuela':      'Venezuela (Bolivarian Republic of)',
       'Vietnam':        'Viet Nam',
        }

    return country_mappings # Convert to lowercase


def map_entries(json, location):
    '''
    Find a match between the JSON file and the provided location(s).

    Args:
        json (list or dict): the data being loaded
        location (list or str): the list of locations to pull from
    '''

    # The data have slightly different formats: list of dicts or just a dict
    countries = [key.lower() for key in json.keys()]

    # Set parameters
    if location is None:
        location = countries
    else:
        location = sc.promotetolist(location)

    # Define a mapping for common mistakes
    mapping = get_country_aliases()
    mapping = {key.lower(): val.lower() for key, val in mapping.items()}

    entries = {}
    for loc in location:
        lloc = loc.lower()
        if lloc not in countries and lloc in mapping:
            lloc = mapping[lloc]
        try:
            ind = countries.index(lloc)
            entry = list(json.values())[ind]
            entries[loc] = entry
        except ValueError as E:
            import traceback;
            traceback.print_exc();
            import pdb;
            pdb.set_trace()
            suggestions = sc.suggest(loc, countries, n=4)
            if suggestions:
                errormsg = f'Location "{loc}" not recognized, did you mean {suggestions}? ({str(E)})'
            else:
                errormsg = f'Location "{loc}" not recognized ({str(E)})'
            raise ValueError(errormsg)

    return entries


def show_locations(location=None, output=False):
    '''
    Print a list of available locations.

    Args:
        location (str): if provided, only check if this location is in the list
        output (bool): whether to return the list (else print)

    **Examples**::

        cv.data.show_locations() # Print a list of valid locations
        cv.data.show_locations('lithuania') # Check if Lithuania is a valid location
        cv.data.show_locations('Viet-Nam') # Check if Viet-Nam is a valid location
    '''
    country_json   = sc.dcp(cad.data)
    aliases        = get_country_aliases()

    age_data       = sc.mergedicts(country_json, aliases) # Countries will overwrite states, e.g. Georgia

    loclist = sc.objdict()
    loclist.age_distributions = sorted(list(age_data.keys()))

    if location is not None:
        age_available = location.lower() in [v.lower() for v in loclist.age_distributions]
        age_sugg = ''
        hh_sugg = ''
        age_sugg = f'(closest match: {sc.suggest(location, loclist.age_distributions)})' if not age_available else ''
        print(f'For location "{location}":')
        print(f'  Population age distribution is available: {age_available} {age_sugg}')
        return

    if output:
        return loclist
    else:
        print(f'There are {len(loclist.age_distributions)} age distributions and {len(loclist.household_size_distributions)} household size distributions.')
        print('\nList of available locations (case insensitive):\n')
        sc.pp(loclist)
        return


def get_age_distribution(location=None):
    '''
    Load age distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        age_data (array): Numpy array of age distributions, or dict if multiple locations
    '''

    # Load the raw data
    json   = sc.dcp(cad.data)
    entries = map_entries(json, location)

    max_age = 99
    result = {}
    for loc,age_distribution in entries.items():
        total_pop = sum(list(age_distribution.values()))
        local_pop = []

        for age, age_pop in age_distribution.items():
            if age[-1] == '+':
                val = [int(age[:-1]), max_age, age_pop/total_pop]
            else:
                ages = age.split('-')
                val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
            local_pop.append(val)
        result[loc] = np.array(local_pop)

    if len(result) == 1:
        result = list(result.values())[0]

    return result


def get_death_rates(location=None, by_sex=True, overall=False):
    '''
    Load death rates for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for
        by_sex (bool): whether to rates by sex
        overall (bool): whether to load total rate


    Returns:
        death_rates (dict): death rates by age and sex
    '''
    # Load the raw data
    try:
        df = sc.load('../data/lx.obj')
    except ValueError as E:
        errormsg = f'Could not locate datafile with age-specific death rates by country. Please run data/get_death_data.py first.'
        raise ValueError(errormsg)

    age_groups = df['dim.AGEGROUP'].unique()
    df = df.set_index(['dim.COUNTRY', 'dim.SEX', 'dim.AGEGROUP'])
    dd = df.groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict()
    raw_lx = map_entries(dd,location)[location]['Value']

    sex_keys = []
    if by_sex: sex_keys += ['Male','Female']
    if overall: sex_keys += ['Both sexes']
    sex_key_map = {'Male':'m', 'Female':'f', 'Both sexes': 'tot'}
 
    max_age = 99
    result = dict()

    # Processing
    for sk in sex_keys:
        sk_out = sex_key_map[sk]
        result[sk_out] = []
        for age in age_groups:
            if (sk, age) in raw_lx.keys():
                this_lx = float(raw_lx[(sk, age)])
                if age[2] == '+':
                    val = [int(age[:2]), max_age, this_lx]
                elif age[0] == '<':
                    val = [0, int(age[1]), this_lx]
                else:
                    ages = re.split('-',age[:-6]) # Remove the 'years' part of the string
                    val = [int(ages[0]), int(ages[1]), this_lx]
                result[sk_out].append(val)
        result[sk_out] = np.array(result[sk_out])
        result[sk_out] = result[sk_out][result[sk_out][:, 0].argsort()]

    return result


def get_birth_rates(location=None):
    '''
    Load crude birth rates for a given country

    Args:
        location (str or list): name of the country to load the birth rates for

    Returns:
        birth_rates (arr): years and crude birth rates
    '''
    # Load the raw data
    try:
        birth_rate_data = sc.load('../data/birth_rates.obj')
    except ValueError as E:
        errormsg = f'Could not locate datafile with birth rates by country. Please run data/get_birth_data.py first.'
        raise ValueError(errormsg)

    standardized = map_entries(birth_rate_data, location)
    birth_rates, years = standardized[location], birth_rate_data['Year']
    return np.array([years, birth_rates])

