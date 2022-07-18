''' Import population sizes by age from UNPD '''


import os
import zipfile
import urllib.request as urllib
import sciris as sc
import pandas as pd

local_stem = 'WPP2022_Population1JanuaryBySingleAgeSex_Medium_'
url_stem = f'https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/{local_stem}'
years = ['1950-2021', '2022-2100']

def get_age_data(force=False, tidy=True):
    sc.heading('Downloading age data...')
    T = sc.timer()
    dfs = []

    # Download data if it's not already in the directory
    for year in years:
        url = f'{url_stem}{year}.zip'
        local_path = f'{local_stem}{year}.csv'
        if force or not os.path.exists(local_path):
            print(f'Downloading from {url}, this may take a while...')
            filehandle, _ = urllib.urlretrieve(url)
            zip_file_object = zipfile.ZipFile(filehandle, 'r')
            zip_file_object.extractall()

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

if __name__ == '__main__':
    dd = get_age_data()