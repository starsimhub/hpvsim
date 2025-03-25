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
import pandas as pd
import sciris as sc
ld = sc.importbypath(sc.thispath(__file__) / 'loaders.py') # To avoid circular HPVsim import

# Set parameters
data_version = '1.4' # Data version
data_file = f'hpvsim_data_v{data_version}.zip'
quick_url = f'https://github.com/hpvsim/hpvsim_data/blob/main/{data_file}?raw=true'

base_url = 'https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/'
indicators_stem = 'WPP2024_Demographic_Indicators_Medium'
age_stem = 'WPP2024_Population1JanuaryBySingleAgeSex_Medium_'
death_stem = 'WPP2024_Life_Table_Abridged_Medium_'
years = ['1950-2023', '2024-2100']
ext = '.csv.gz'

__all__ = ['download_data', 'quick_download', 'check_downloaded', 'remove_data']


# Define here to optionally be overwritten
filesdir = ld.filesdir

def make_paths():
    """ Create all file paths and download URLs """
    paths = sc.autolist()
    paths += indicators_stem + ext
    for stem in [age_stem, death_stem]:
        for year_pair in years:
            paths += stem + year_pair + ext
    out = sc.odict({p:base_url+p for p in paths})
    return out

def set_filesdir(path):
    ''' Used to change the file folder '''
    global filesdir
    orig = filesdir
    filesdir = path
    print(f'Done: filesdir reset from {orig} to {filesdir}')
    return


def get_UN_data(label='', file_stem=None, outfile=None, years=years, excludes=[':', '('],
                columns=None, force=None, tidy=None, verbose=True):
    ''' Download data from UN Population Division; remove entries with ":" or "(" in the name (not true countries) '''
    if force is None: force = False
    if tidy  is None: tidy  = True

    if verbose:
        print(f'Downloading {label} data...\n')
    T = sc.timer()
    dfs = []

    # Download data if it's not already in the directory
    for year_pair in years:
        url = f'{base_url}{file_stem}{year_pair}{ext}'
        local_base = filesdir/f'{file_stem}{year_pair}'
        local_zip = f'{local_base}{ext}'
        if force or not os.path.exists(local_zip):
            if verbose:
                sc.printgreen(f'Downloading {url}...')
            sc.download(url, filename=local_zip, verbose=False)
        else:
            if verbose:
                print(f'Skipping {local_zip}, already downloaded')

        # Extract the parts used in the model and save
        df = pd.read_csv(local_zip, usecols=columns)
        dfs.append(df)
        if tidy:
            sc.rmpath(local_zip, die=False, verbose=verbose)
        if verbose:
            T.toctic(label=f'  Done with "{label}" for {year_pair}')
            print()

    # Parse by location
    df = pd.concat(dfs)
    dd = {l:d for l,d in df.groupby('Location')}

    # Filter to exclude certain names
    if excludes is not None:
        excludes = sc.tolist(excludes)
        for ex in excludes:
            dd = {k:v for k,v in dd.items() if ex not in k}

    # Convert to objdict and double check
    dd = sc.objdict(dd)
    assert dd[0][columns[-1]].dtype != object, "Last column should be numeric type, not mixed or string type"

    sc.save(filesdir/outfile, dd)
    if verbose:
        if verbose:
            T.toc(doprint=False)
            print(f'Done with {label}: {T.sum():n}')

    return dd


def get_age_data(**kw):
    ''' Download population sizes by age from UNPD '''
    columns = ["Location", "Time", "AgeGrpStart", "PopTotal"]
    outfile = 'populations.obj'
    kw = kw | dict(label='age', file_stem=age_stem, outfile=outfile, columns=columns)
    return get_UN_data(**kw)


def get_age_sex_data(**kw):
    ''' Download population sizes by age from UNPD '''
    columns = ["Location", "Time", "AgeGrpStart", "PopMale", "PopFemale"]
    outfile = 'populations_by_sex.obj'
    kw = kw | dict(label='age', file_stem=age_stem, outfile=outfile, columns=columns)
    return get_UN_data(**kw)


def get_death_data(**kw):
    ''' Download age-specific death rates and population distributions from UNPD '''
    columns = ["Location", "Time", "Sex", "AgeGrpStart", "mx"]
    outfile = 'mx.obj'
    kw = kw | dict(label='death', file_stem=death_stem, outfile=outfile, columns=columns)
    return get_UN_data(**kw)


def get_birth_data(**kw):
    ''' Download crude birth rates UNPD '''
    columns = ["Location", "Time", "CBR"]
    outfile = 'birth_rates.obj'
    years = [''] # Unlike other data sources, this does not have a year range
    kw = kw | dict(label='birth', years=years, file_stem=indicators_stem, outfile=outfile, columns=columns)
    return get_UN_data(**kw)


def downloader(which, **kwargs):
    ''' Function for use with a parallel download function '''
    sc.heading(f'Working on {which}')
    if which in ['age', 'ages']:
        get_age_data(**kwargs)
        get_age_sex_data(**kwargs)
    if which in ['birth', 'births']:
        get_birth_data(**kwargs)
    if which in ['death', 'deaths']:
        get_death_data(**kwargs)
    return


def download_data(serial=False, **kwargs):
    ''' Download data in parallel '''
    sc.heading('Downloading HPVsim data manually, please be patient...')
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

    # Actually download
    if serial:
        for key in which:
            downloader(key, **kwargs)
    else:
        sc.parallelize(downloader, which, kwargs=kwargs)
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
        if key != 'metadata':
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
        try:
            metadata = sc.loadjson(ld.files.metadata)
        except Exception as E:
            print(f'Warning: metadata not available; not checking version:\n{E}')
            return ok
        match = metadata['version'] == data_version
        if verbose:
            if not match and verbose:
                print(f'HPVsim data: versions do not match ({metadata["version"]} != {data_version})')
            elif match and verbose>1:
                print(f'HPVsim data: versions match ({data_version})')
        ok = ok and match

    return ok


def remove_data(verbose=True, die=False, **kwargs):
    ''' Remove downloaded data; arguments passed to sc.rmpath() '''
    if verbose: sc.heading('Removing HPVsim data files')
    for key,fn in ld.files.items():
        sc.rmpath(fn, verbose=verbose, die=die, **kwargs)
    if verbose: print('Data files removed.')
    return


if __name__ == '__main__':

    ans = input('Are you sure you want to remove and redownload data? y/[n] ')
    if ans == 'y':
        remove_data()
        download_data(serial=True) # TEMP
        check_downloaded()