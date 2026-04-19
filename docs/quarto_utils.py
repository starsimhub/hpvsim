"""
Utilities for HPVsim Quarto docs (version file, quartodoc, notebook cleanup).
"""

import os
import sys
import importlib
import subprocess
import sciris as sc
import hpvsim as hpv

default_folders = ['tutorials'] # Folders with notebooks
temp_patterns = ['**/my-*.*', '**/example*.*'] # Temporary files to be removed
temp_items = ['user_guide/results'] # Temporary files/folders to be removed

def run(cmd):
    """Verbose version of subprocess.run."""
    sc.printgreen(f'\n> {cmd}\n')
    return subprocess.run(cmd, check=True, shell=True)


def update_version(pkg=hpv):
    sc.heading('Updating version...')
    return sc.saveyaml('_variables.yml', dict(version=pkg.__version__, versiondate=pkg.__versiondate__))


def build_api_docs():
    sc.heading('Building API documentation...')
    return run('python -m quartodoc build')


def customize_aliases(mod_name='hpvsim', json_path='objects.json'):
    """
    Add aliases so links can use hpvsim.ClassName as well as submodule paths.
    """
    sc.heading('Customizing aliases ...')
    mod = importlib.import_module(mod_name)
    mod_items = dir(mod)
    json_data = sc.loadjson(json_path)
    items = json_data['items']
    names = [item['name'] for item in items]
    print(f'  Loaded {len(json_data["items"])} items')
    dups = []
    for item in items:
        parts = item['name'].split('.')
        if len(parts) < 3 or parts[0] != mod_name:
            continue
        objname = parts[2]
        if objname in mod_items:
            remainder = '.'.join(parts[2:])
            alias = f'{mod_name}.{remainder}'
            if alias not in names:
                dup = sc.dcp(item)
                dup['name'] = alias
                dups.append(dup)
    items.extend(dups)
    sc.savejson(json_path, json_data)
    print(f'  Saved {len(json_data["items"])} items')


def build_interlinks():
    sc.heading('Building docs links...')
    return run('python -m quartodoc interlinks')


def clean_outputs(folders=None, sleep=3, patterns=None):
    """ Clears outputs from notebooks """
    sc.heading('Cleaning outputs ...')
    if folders is None:
        folders = default_folders
    if patterns is None:
        patterns = temp_patterns
    filenames = sc.dcp(temp_items)
    for pattern in patterns:
        for folder in folders:
            filenames += sc.getfilelist(folder=folder, pattern=pattern, recursive=True)
    if len(filenames):
        print(f'Deleting: {sc.newlinejoin(filenames)}\nin {sleep} seconds')
        sc.timedsleep(sleep)
        for filename in filenames:
            sc.rmpath(filename, verbose=True, die=False)
    else:
        print('No files found to clean')
    return


if __name__ == '__main__':

    if 'pre' in sys.argv:
        sc.heading('Starting Quarto docs build', divider='★')
        update_version()
        build_api_docs()
        customize_aliases()
        build_interlinks()

    elif 'post' in sys.argv:
        clean_outputs()

    elif len(sys.argv) > 1:
        errormsg = f'Argument must be "pre" or "post", not {sys.argv}'
        raise ValueError(errormsg)
    else:
        raise ValueError('Run with pre or post as argv')
