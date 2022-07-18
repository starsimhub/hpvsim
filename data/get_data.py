'''
Download data needed for HPVsim
'''

import os
import sys
import zipfile
from urllib import request
import wbgapi as wb
import numpy as np
import pandas as pd
import sciris as sc

# Set parameters
age_stem = 'WPP2022_Population1JanuaryBySingleAgeSex_Medium_'
death_stem = 'WPP2022_Life_Table_Abridged_Medium_'
base_url = 'https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/'
years = ['1950-2021', '2022-2100']


def get_UN_data(label='', file_stem=None, outfile=None, columns=None, force=None, tidy=None):
    ''' Download data from UN Population Division '''
    if force is None: force = False
    if tidy  is None: tidy  = True

    sc.heading(f'Downloading {label} data...')
    T = sc.timer()
    dfs = []

    # Download data if it's not already in the directory
    for year in years:
        url = f'{base_url}{file_stem}{year}.zip'
        local_path = f'{file_stem}{year}.csv'
        if force or not os.path.exists(local_path):
            print(f'Downloading from {url}, this may take a while...')
            filehandle, _ = request.urlretrieve(url)
            zip_file_object = zipfile.ZipFile(filehandle, 'r')
            zip_file_object.extractall()
        else:
            print(f'Skipping {local_path}, already downloaded')

        # Extract the parts used in the model and save
        df = pd.read_csv(local_path)
        df = df[columns]
        dfs.append(df)
        if tidy:
            print(f'Removing {local_path}')
            os.remove(local_path)
        T.toctic(label=f'Done with {year}')

    df = pd.concat(dfs)
    dd = {l:df[df["Location"]==l] for l in df["Location"].unique()}
    sc.save(outfile, dd)

    T.toc(doprint=False)
    print(f'Done with {label}: took {T.timings[:].sum():0.1f} s.')

    return dd


def get_age_data(force=None, tidy=None):
    ''' Import population sizes by age from UNPD '''
    columns = ["Location", "Time", "AgeGrpStart", "PopTotal"]
    outfile = 'mx.obj'
    kw = dict(label='age', file_stem=age_stem, outfile=outfile, columns=columns, force=force, tidy=tidy)
    return get_UN_data(**kw)


def get_death_data(force=None, tidy=None):
    ''' Import age-specific death rates and population distributions from UNPD '''
    columns = ["Location", "Time", "Sex", "AgeGrpStart", "mx"]
    outfile = 'populations.obj'
    kw = dict(label='death', file_stem=death_stem, outfile=outfile, columns=columns, force=force, tidy=tidy)
    return get_UN_data(**kw)


def get_birth_data(start=1960, end=2020):
    ''' Import crude birth rates from WB '''
    sc.heading('Downloading World Bank birth rate data...')
    T = sc.timer()
    birth_rates = wb.data.DataFrame('SP.DYN.CBRT.IN', time=range(start,end), labels=True, skipAggs=True).reset_index()
    d = dict()
    for country in birth_rates['Country'].unique():
        d[country] = birth_rates.loc[(birth_rates['Country']==country)].values[0,2:]
    d['years'] = np.arange(start, end)
    sc.saveobj('birth_rates.obj',d)
    T.toc(label='Done with birth data')
    return d


def parallel_downloader(which):
    ''' Function for use with a parallel download function '''
    if which in ['age', 'ages']:
        get_age_data()
    if which in ['birth', 'births']:
        get_birth_data()
    if which in ['death', 'deaths']:
        get_death_data()
    return



if __name__ == '__main__':

    T = sc.timer()

    if len(sys.argv) > 1:
        which = sys.argv[1]
        if which not in ['all', 'age', 'ages', 'birth', 'births', 'death', 'deaths']:
            errormsg = f'Invalid selection "{which}": must be all, ages, births, or deaths'
            raise ValueError(errormsg)
    else:
        which = 'all'

    if which == 'all':
        which = ['age', 'birth', 'death']

    sc.parallelize(parallel_downloader, which)

    T.toc('Done downloading data for HPVsim')