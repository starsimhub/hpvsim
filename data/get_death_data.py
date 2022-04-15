''' Fetch and save age-specific death rates for all countries '''


from pandas.io.json import json_normalize   
import urllib.request
import json 
import sciris as sc

url = "http://apps.who.int/gho/athena/api/GHO/LIFE_0000000029.json?profile=simple&filter=COUNTRY:*;YEAR:2019"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}) # To avoid 403 errors
with urllib.request.urlopen(req) as req:
    data = json.loads(req.read().decode())
df = json_normalize(data['fact'])
sc.saveobj('data/age_specific_death_rates.obj',df)
