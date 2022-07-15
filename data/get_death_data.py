''' Import age-specific death rates and population distributions from UNPD '''

import sciris as sc
import numpy as np
import os
import urllib.request as urllib
import zipfile
import pandas as pd

# Filepaths
urls = ["https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2022_Life_Table_Abridged_Medium_1950-2021.zip",
        "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2022_Life_Table_Abridged_Medium_2022-2100.zip"]
local_paths = ["WPP2022_Life_Table_Abridged_Medium_1950-2021.csv",
               "WPP2022_Life_Table_Abridged_Medium_2022-2100.csv"]

# Download data if it's not already in the directory
for url, local_path in zip(urls, local_paths):
    if not os.path.exists(local_path):
        print(f'Downloading from {url}, this may take a while...')
        filehandle, _ = urllib.urlretrieve(url)
        zip_file_object = zipfile.ZipFile(filehandle, 'r')
        zip_file_object.extractall()

# Extract the parts used in the model and save
df1 = pd.read_csv(local_paths[0])
df2 = pd.read_csv(local_paths[1])
df11 = df1[["Location", "Time", "Sex", "AgeGrpStart", "mx"]]
df22 = df2[["Location", "Time", "Sex", "AgeGrpStart", "mx"]]
df = pd.concat([df11, df22])
dd = {l:df[df["Location"]==l] for l in df["Location"].unique()}
sc.save('mx.obj',dd)
