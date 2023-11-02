'''
Download data needed for HPVsim.

Typically, this is done automatically: on load, HPVsim checks if the data are already
downloaded, and if not, downloads them using the quick_download() function. The
"slow download" functions supply the files that are usually zipped and stored in
a separate repository, hpvsim_data.

To ensure the data is updated, update the data_version parameter below.

Running this file as a script will remove and then re-download all data.
'''

import os
import sys
import numpy as np
import pandas as pd
import sciris as sc
from hpvsim.data import loaders as ld

# Set parameters
data_version = '1.3' # Data version
data_file = f'hpvsim_data_v{data_version}.zip'
quick_url = f'https://github.com/amath-idm/hpvsim_data/blob/main/{data_file}?raw=true'
age_stem = 'WPP2022_Population1JanuaryBySingleAgeSex_Medium_'
death_stem = 'WPP2022_Life_Table_Abridged_Medium_'
base_url = 'https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/'
years = ['1950-2021', '2022-2100']


__all__ = ['get_data', 'quick_download', 'check_downloaded', 'remove_data']


# Define here to optionally be overwritten
filesdir = ld.filesdir

def set_filesdir(path):
    ''' Used to change the file folder '''
    global filesdir
    orig = filesdir
    filesdir = path
    print(f'Done: filesdir reset from {orig} to {filesdir}')
    return


def get_UN_data(label='', file_stem=None, outfile=None, columns=None, force=None, tidy=None):
    ''' Download data from UN Population Division '''
    if force is None: force = False
    if tidy  is None: tidy  = True

    sc.heading(f'Getting {label} data...')
    T = sc.timer()
    dfs = []

    # Download data if it's not already in the directory
    for year in years:
        url = f'{base_url}{file_stem}{year}.zip'
        local_base = filesdir/f'{file_stem}{year}'
        local_zip = f'{local_base}.zip'
        local_csv = f'{local_base}.csv'
        if force or not os.path.exists(local_csv):
            print(f'\nDownloading from {url}, this may take a while...')
            sc.download(url, filename=local_zip)
            sc.unzip(local_zip, outfolder=filesdir)
        else:
            print(f'Skipping {local_csv}, already downloaded')

        # Extract the parts used in the model and save
        df = pd.read_csv(local_csv, usecols=columns)
        dfs.append(df)
        if tidy:
            print(f'Removing {local_base}')
            sc.rmpath(local_zip, die=False)
            sc.rmpath(local_csv, die=False)
        T.toctic(label=f'  Done with {label} for {year}')

    # Parse by location
    df = pd.concat(dfs)
    dd = sc.objdict({l:d for l,d in df.groupby('Location')})
    assert dd[0][columns[-1]].dtype != object, "Last column should be numeric type, not mixed or string type"
        
    sc.save(filesdir/outfile, dd)
    T.toc(f'Done with {label}')

    return dd


def get_age_data(force=None, tidy=None):
    ''' Import population sizes by age from UNPD '''
    columns = ["Location", "Time", "AgeGrpStart", "PopTotal"]
    outfile = 'populations.obj'
    kw = dict(label='age', file_stem=age_stem, outfile=outfile, columns=columns, force=force, tidy=tidy)
    return get_UN_data(**kw)


def get_death_data(force=None, tidy=None):
    ''' Import age-specific death rates and population distributions from UNPD '''
    columns = ["Location", "Time", "Sex", "AgeGrpStart", "mx"]
    outfile = 'mx.obj'
    kw = dict(label='death', file_stem=death_stem, outfile=outfile, columns=columns, force=force, tidy=tidy)
    return get_UN_data(**kw)


def get_ex_data(force=None, tidy=None):
    ''' Import age-specific life expectancy and population distributions from UNPD '''
    columns = ["Location", "Time", "Sex", "AgeGrpStart", "ex"]
    outfile = 'ex.obj'
    kw = dict(label='ex', file_stem=death_stem, outfile=outfile, columns=columns, force=force, tidy=tidy)
    return get_UN_data(**kw)


def get_birth_data(start=1960, end=2020, force=None, tidy=None):
    ''' Import crude birth rates from WB '''
    sc.heading('Downloading World Bank birth rate data...')
    try:
        import wbgapi as wb
    except Exception as E:
        errormsg = 'Could not import wbgapi: cannot download raw data'
        raise ModuleNotFoundError(errormsg) from E
    T = sc.timer()
    birth_rates = wb.data.DataFrame('SP.DYN.CBRT.IN', time=range(start,end), labels=True, skipAggs=True).reset_index()
    d = dict()
    for country in birth_rates['Country'].unique():
        d[country] = birth_rates.loc[(birth_rates['Country']==country)].values[0,3:]
        d[country] = d[country].astype(float) # Loaded as an object otherwise!
    d['years'] = np.arange(start, end)
    sc.save(filesdir/'birth_rates.obj', d)
    
    if tidy:
        print(f'Removing {local_path}')
        sc.rmpath(local_path, die=False)
            
    T.toc(label='Done with birth data')
    return d


def parallel_downloader(which, **kwargs):
    ''' Function for use with a parallel download function '''
    if which in ['age', 'ages']:
        get_age_data(**kwargs)
    if which in ['birth', 'births']:
        get_birth_data(**kwargs)
    if which in ['death', 'deaths']:
        get_death_data(**kwargs)
    if which in ['life_expectancy', 'ex']:
        get_ex_data(**kwargs)
    return


def get_data(serial=False, **kwargs):
    ''' Download data in parallel '''
    sc.heading('Downloading HPVsim data, please be patient...')
    T = sc.timer()

    if len(sys.argv) > 1:
        which = sys.argv[1]
        if which not in ['all', 'age', 'ages', 'birth', 'births', 'death', 'deaths']:
            errormsg = f'Invalid selection "{which}": must be all, ages, births, or deaths'
            raise ValueError(errormsg)
    else:
        which = 'all'

    if which == 'all':
        which = ['age', 'birth', 'death', 'life_expectancy']

    # Actually download
    sc.parallelize(parallel_downloader, which, kwargs=kwargs, serial=serial)
    T.toc('Done downloading data for HPVsim')

    return


def quick_download(verbose=True, init=False):
    ''' Download pre-processed data files '''
    if verbose:
        sc.heading('Downloading preprocessed HPVsim data')
        if init:
            print('Note: this automatic download only happens once, when HPVsim is first run.\n\n')
    filepath = sc.makefilepath(filesdir / f'tmp_{data_file}.zip')
    sc.download(url=quick_url, filename=filepath, convert=False, verbose=verbose)
    sc.unzip(filepath, outfolder=filesdir)
    sc.rmpath(filepath)
    if verbose:
        print('\nData downloaded.')
    return


def check_downloaded(verbose=1, check_version=True):
    '''
    Check if data is downloaded. Note: to update data, update the date here and
    in data/files/metadata.json.

    Args:
        verbose (int): detail to print (0 = none, 1 = reason for failure, 2 = everything)
        check_version (bool): whether to treat a version mismatch as a failure
    '''

    # Do file checks
    exists = dict()
    for key,fn in ld.files.items():
        exists[key] = os.path.exists(fn)
        if verbose>1:
            print(f'HPVsim data: checking {fn}: {exists[key]}')
    ok = all(list(exists.values()))
    if not ok and verbose:
        print(f'HPVsim data: at least one file missing: {exists}')
    elif ok and verbose>1:
        print('HPVsim data: all files exist')

    # Do version check (if files exist)
    if ok and check_version:
        metadata = sc.loadjson(ld.files.metadata)
        match = metadata['version'] == data_version
        if verbose:
            if not match and verbose:
                print(f'HPVsim data: versions do not match ({metadata["version"]} != {data_version})')
            elif match and verbose>1:
                print(f'HPVsim data: versions match ({data_version})')
        ok = ok and match

    return ok


def remove_data(verbose=True, **kwargs):
    ''' Remove downloaded data; arguments passed to sc.rmpath() '''
    if verbose: sc.heading('Removing HPVsim data files')
    for key,fn in ld.files.items():
        sc.rmpath(fn, verbose=verbose, **kwargs)
    if verbose: print('Data files removed.')
    return


if __name__ == '__main__':
    
    ans = input('Are you sure you want to remove and redownload data? y/[n] ')
    if ans == 'y':
        remove_data()
        get_data()
        check_downloaded()