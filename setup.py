import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'hpvsim', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.rst'), "r") as f:
    long_description = f.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

setup(
    name="hpvsim",
    version=version,
    author="Robyn Stuart, Jamie Cohen, Cliff Kerr, Romesh Abeysuriya, Mariah Boudreau, Daniel Klein, Hao Hu",
    author_email="robyn.stuart@gatesfoundation.org",
    description="HPVsim: Human Papillomavirus Simulator",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='http://hpvsim.org',
    keywords=["HPV", "agent-based model", "simulation"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'pandas>=1.4.0', 
        'sciris>=2.0.4',
        'matplotlib',
        'seaborn',
        'optuna',
    ],
)