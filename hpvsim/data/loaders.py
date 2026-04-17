'''
Load data
'''

#%% Housekeeping
import numpy as np
import pandas as pd
import sciris as sc
import unicodedata
import re


__all__ = ['get_country_aliases', 'map_entries', 'get_age_distribution', 'get_age_distribution_over_time', 'get_total_pop', 'get_death_rates',
           'get_birth_rates']


thisdir = sc.thispath(__file__)
filesdir = thisdir / 'files'
files = sc.objdict()
files.metadata = 'metadata.json'
files.age_dist = 'populations.obj'
files.age_dist_sex = 'populations_by_sex.obj'
files.birth = 'birth_rates.obj'
files.death = 'mx.obj'

download_tip = 'Please run hpv.download_data() first.'

# Cache data as a dict
cache = dict()

for k,v in files.items():
    files[k] = filesdir / v


def sanitizestr(string=None, alphanumeric=True, nospaces=True, asciify=True, lower=True, spacechar='_', symchar='_'):
    ''' Remove all non-printable characters from a string -- to be moved to Sciris eventually '''
    string = str(string)
    if asciify:
        string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode()
    if nospaces:
        string = string.replace(' ', spacechar)
    if lower:
        string = string.lower()
    if alphanumeric:
        string = re.sub('[^0-9a-zA-Z ]', symchar, string)
    return string


def load_file(path):
    ''' Load a data file from the local data folder -- but store in memory if already loaded '''
    strpath = str(path)
    if strpath not in cache:
        obj = sc.load(path)
        cache[strpath] = obj
    else:
        obj = cache[strpath]
    return obj


def get_country_aliases():
    ''' Define aliases for countries with odd names in the data '''
    country_mappings = {
       'Bolivia':        'Bolivia (Plurinational State of)',
       'Burkina':        'Burkina Faso',
       'Cape Verde':     'Cabo Verdeo',
       'Hong Kong':      'China, Hong Kong Special Administrative Region',
       'Macao':          'China, Macao Special Administrative Region',
       "Cote d'Ivoire":  "Côte d'Ivoire",
       "Cote dIvoire":   "Côte d'Ivoire",
       "Ivory Coast":    "Côte d'Ivoire",
       'DRC':            'Democratic Republic of the Congo',
       'Congo':          'Congo, Rep.',
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
    elif sc.checktype(json, pd.DataFrame):
        countries = [l.lower() for l in np.unique(json.Country.values)]

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

    # If a single entry, just return it
    if len(entries) == 1:
        entries = entries[loc]

    return entries


def get_age_distribution(location=None, year=None, total_pop_file=None, age_datafile=None):
    '''
    Load age distribution for a given country or countries.

    Args:
        location (str): name of the country to load the age distribution for
        year (int): year to load the age distribution for
        total_pop_file (str): optional filepath to save total population size for every year

    Returns:
        age_data (array): Numpy array of age distributions, or dict if multiple locations
    '''

    # Load the raw data
    if age_datafile is None:
        try:
            df = load_file(files.age_dist)
        except Exception as E:
            errormsg = f'Could not locate datafile with population sizes by country. {download_tip}'
            raise ValueError(errormsg) from E

        # Handle year
        if year is None:
            from .. import misc as hpm # Not needed except for this warning
            warnmsg = 'No year provided for the initial population age distribution, using 2000 by default'
            hpm.warn(warnmsg)
            year = 2000

        # Extract the age distribution for the given location and year
        full_df = map_entries(df, location)
        raw_df = full_df[full_df["Time"] == year]

    else:
        raw_df = pd.read_csv(age_datafile)

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


def get_age_distribution_over_time(location=None, popage_datafile=None):
    '''
    Load age distribution for a given country or countries over time.

    Args:
        location (str): name of the country to load the age distribution for

    Returns:
        age_data (dataframe): Pandas dataframe with age distribution over time
    '''

    # Load the raw data
    if popage_datafile is None:
        try:
            df = load_file(files.age_dist_sex)
        except Exception as E:
            errormsg = f'Could not locate datafile with population sizes over time. {download_tip}'
            raise ValueError(errormsg) from E
        full_df = map_entries(df, location)
    else:
        full_df = pd.read_csv(popage_datafile)

    result = full_df.rename(columns={'Time':'year', 'AgeGrpStart': 'age', 'PopMale': 'male', 'PopFemale': 'female'})
    result['male'] *= 1e3 # reported as per 1,000
    result['female'] *= 1e3 # reported as per 1,000

    return result


def get_total_pop(location=None, pop_datafile=None):
    '''
    Load total population for a given country or countries.

    Args:
        location (str or list): name of the country to load the total population for

    Returns:
        pop_data (dataframe): Dataframe of year and pop_size columns
    '''

    # Load the raw data
    if pop_datafile is None:
        try:
            df = load_file(files.age_dist)
        except Exception as E:
            errormsg = f'Could not locate datafile with total population sizes by country. {download_tip}'
            raise ValueError(errormsg) from E

        # Extract the age distribution for the given location and year
        full_df = map_entries(df, location)
        dd = full_df.groupby("Time").sum(numeric_only=True)["PopTotal"]

    else:
        dd = pd.read_csv(pop_datafile)

    dd = dd * 1e3
    df = sc.dataframe(dd).reset_index().rename(columns={'Time':'year', 'PopTotal':'pop_size'})
    return df


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
        errormsg = f'Could not locate datafile with age-specific death rates by country. {download_tip}'
        raise ValueError(errormsg) from E

    raw_df = map_entries(df, location)

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
        errormsg = f'Could not locate datafile with birth rates by country. {download_tip}'
        raise ValueError(errormsg) from E

    raw_df = map_entries(birth_rate_data, location)
    raw_df = raw_df.dropna()
    df = sc.dataframe(raw_df).reset_index().rename(columns={'Time':'year', 'CBR':'cbr'})

    return df

