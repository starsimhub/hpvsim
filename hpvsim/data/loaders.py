'''
Load data
'''

#%% Housekeeping
import numpy as np
import sciris as sc
from .. import misc as hpm

__all__ = ['get_country_aliases', 'map_entries', 'get_age_distribution', 'get_death_rates', 'get_birth_rates']


thisdir = sc.path(sc.thisdir())
files = sc.objdict()
files.age_dist = 'populations.obj'
files.birth = 'birth_rates.obj'
files.death = 'mx.obj'

for k,v in files.items():
    files[k] = thisdir / v


def load_file(filename):
    ''' Load a data file from the local data folder '''
    path = sc.path(sc.thisdir()) / filename
    obj = sc.load(path)
    return obj


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


def map_entries(json, location, df=None):
    '''
    Find a match between the JSON file and the provided location(s).

    Args:
        json (list or dict): the data being loaded
        location (list or str): the list of locations to pull from
    '''

    # The data have slightly different formats: list of dicts or just a dict
    if sc.checktype(json, dict):
        countries = [key.lower() for key in json.keys()]
    elif sc.checktype(json, 'listlike'):
        countries = [l.lower() for l in json]

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
            suggestions = sc.suggest(loc, countries, n=4)
            if suggestions:
                errormsg = f'Location "{loc}" not recognized, did you mean {suggestions}? ({str(E)})'
            else:
                errormsg = f'Location "{loc}" not recognized ({str(E)})'
            raise ValueError(errormsg)

    return entries


def get_age_distribution(location=None, year=None, total_pop_file=None):
    '''
    Load age distribution for a given country or countries.

    Args:
        location (str or list): name of the country to load the age distribution for
        year (int): year to load the age distribution for
        total_pop_file (str): optional filepath to save total population size for every year

    Returns:
        age_data (array): Numpy array of age distributions, or dict if multiple locations
    '''

    # Load the raw data
    try:
        df = load_file(files.age_dist)
    except Exception as E:
        errormsg = 'Could not locate datafile with population sizes by country. Please run data/get_data.py first.'
        raise ValueError(errormsg) from E

    # Handle year
    if year is None:
        warnmsg = 'No year provided for the initial population age distribution, using 2000 by default'
        hpm.warn(warnmsg)
        year = 2000

    # Extract the age distribution for the given location and year
    full_df = map_entries(df, location)[location]
    raw_df = full_df[full_df["Time"] == year]

    # Pull out the data
    result = np.array([raw_df["AgeGrpStart"],raw_df["AgeGrpStart"]+1,raw_df["PopTotal"]*1e3]).T # Data are stored in thousands

    # Optinally save total population sizes for calibration/plotting purposes
    if total_pop_file is not None:
        dd = full_df.groupby("Time").sum()["PopTotal"]
        dd = dd * 1e3
        dd = dd.astype(int)
        dd = dd.rename("n_alive")
        dd = dd.rename_axis("year")
        dd.to_csv(total_pop_file)

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
        df = load_file(files.death)
    except Exception as E:
        errormsg = 'Could not locate datafile with age-specific death rates by country. Please run data/get_data.py first.'
        raise ValueError(errormsg) from E

    raw_df = map_entries(df, location)[location]

    sex_keys = []
    if by_sex: sex_keys += ['Male', 'Female']
    if overall: sex_keys += ['Both sexes']
    sex_key_map = {'Male': 'm', 'Female': 'f', 'Both sexes': 'tot'}

    # max_age = 99
    # age_groups = raw_df['AgeGrpStart'].unique()
    years = raw_df['Time'].unique()
    result = dict()

    # Processing
    for year in years:
        result[year] = dict()
        for sk in sex_keys:
            sk_out = sex_key_map[sk]
            result[year][sk_out] = np.array(raw_df[(raw_df['Time']==year) & (raw_df['Sex']== sk)][['AgeGrpStart','mx']])
            result[year][sk_out] = result[year][sk_out][result[year][sk_out][:, 0].argsort()]

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
        birth_rate_data = load_file(files.birth)
    except Exception as E:
        errormsg = 'Could not locate datafile with birth rates by country. Please run data/get_data.py first.'
        raise ValueError(errormsg) from E

    standardized = map_entries(birth_rate_data, location)
    birth_rates, years = standardized[location], birth_rate_data['years']
    birth_rates, inds = sc.sanitize(birth_rates, returninds=True)
    years = years[inds]
    return np.array([years, birth_rates])