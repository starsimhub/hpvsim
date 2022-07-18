'''
Download data needed for HPVsim
'''

import os
import sys
import zipfile
import urllib.request as urllib
import wbgapi as wb
import numpy as np
import pandas as pd
import sciris as sc

# Set parameters
age_stem = 'WPP2022_Population1JanuaryBySingleAgeSex_Medium_'
death_stem = 'WPP2022_Life_Table_Abridged_Medium_'
base_url = 'https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/'
years = ['1950-2021', '2022-2100']


def get_UN_data(file_stem, force=None, tidy=None):
    ''' Download data from UN Population Division '''
    if force is None: force = False
    if tidy  is None: tidy  = True

    sc.heading('Downloading UN data...')
    T = sc.timer()
    dfs = []

    # Download data if it's not already in the directory
    for year in years:
        url = f'{base_url}{file_stem}{year}.zip'
        local_path = f'{file_stem}{year}.csv'
        if force or not os.path.exists(local_path):
            print(f'Downloading from {url}, this may take a while...')
            filehandle, _ = urllib.urlretrieve(url)
            zip_file_object = zipfile.ZipFile(filehandle, 'r')
            zip_file_object.extractall()
        else:
            print(f'Skipping {local_path}, already downloaded')

        # Extract the parts used in the model and save
        df = pd.read_csv(local_path)
        df = df[["Location", "Time", "AgeGrpStart", "PopTotal"]]
        dfs.append(df)
        if tidy:
            os.remove(local_path)
        T.toctic(label=f'Done with {year}')

    df = pd.concat(dfs)
    dd = {l:df[df["Location"]==l] for l in df["Location"].unique()}
    sc.save('populations.obj',dd)

    print(f'Done: took {T.timings[:].sum():0.1f} s.')

    return dd


def get_age_data(force=None, tidy=None):
    ''' Import population sizes by age from UNPD '''
    return get_UN_data(file_stem=age_stem, force=force, tidy=tidy)


def get_death_data(force=None, tidy=None):
    ''' Import age-specific death rates and population distributions from UNPD '''
    return get_UN_data(file_stem=death_stem, force=force, tidy=tidy)


def get_birth_data(start=1960, end=2020):
    ''' Import crude birth rates from WB '''
    sc.heading('Downloading WB birth rate data...')
    birth_rates = wb.data.DataFrame('SP.DYN.CBRT.IN', time=range(start,end), labels=True, skipAggs=True).reset_index()
    d = dict()
    for country in birth_rates['Country'].unique():
        d[country] = birth_rates.loc[(birth_rates['Country']==country)].values[0,2:]
    d['years'] = np.arange(start, end)
    sc.saveobj('birth_rates.obj',d)
    return d


if __name__ == '__main__':

    if len(sys.argv) > 1:
        which = sys.argv[1]
        if which not in ['all', 'age', 'ages', 'birth', 'births', 'death', 'deaths']:
            errormsg = f'Invalid selection "{which}": must be all, ages, births, or deaths'
            raise ValueError(errormsg)
    else:
        which = 'all'

    if which in ['all', 'age', 'ages']:
        get_age_data()
    if which in ['all', 'birth', 'births']:
        get_birth_data()
    if which in ['all', 'death', 'deaths']:
        get_death_data()