"""
Utilities for HPVsim Quarto docs (version file, quartodoc, notebook cleanup).
"""

import os
import sys
import importlib
import subprocess
import sciris as sc
import hpvsim as hpv

default_folders = ['tutorials']
temp_patterns = ['**/my-*.*', '**/example*.*']
temp_items = []


def run(cmd):
    """Verbose version of subprocess.run."""
    sc.printgreen(f'\n> {cmd}\n')
    return subprocess.run(cmd, check=True, shell=True)


@sc.timer('Update version')
def update_version(pkg=hpv):
    sc.heading('Updating docs version number...')
    filename = '_variables.yml'
    data = dict(version=pkg.__version__, versiondate=pkg.__versiondate__)
    orig = sc.loadyaml(filename)
    if data != orig:
        sc.saveyaml(filename, data)
        print('Version updated to:', data)
    else:
        print('Version already correct:', orig)


@sc.timer('Build API docs')
def build_api_docs():
    sc.heading('Building API documentation...')
    return run('python -m quartodoc build')


@sc.timer('Customize aliases')
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


@sc.timer('Build interlinks')
def build_interlinks():
    sc.heading('Building docs links...')
    return run('python -m quartodoc interlinks')


@sc.timer('Clean outputs')
def clean_outputs(folders=None, sleep=3, patterns=None):
    """Clear temporary files produced during notebook runs."""
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
        raise ValueError(f'Argument must be "pre" or "post", not {sys.argv}')
    else:
        raise ValueError('Run with pre or post as argv')
