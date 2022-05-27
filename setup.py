from setuptools import setup, find_packages
from setuptools.extension import Extension
import hpvsim.version

with open("README.md", "r") as fh:
    long_description = fh.read()
    ext_name = "hpvsim"

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().split("\n")

setup(
    name=ext_name,
    version=hpvsim.version.__version__,
    author="Jamie Cohen",
    author_email="jcohen@idmod.org",
    description="IDM's agent-based HPV transmission model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amath-idm/hpvsim",
    packages=setuptools.find_packages(exclude=["data"]),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons International",
        "Operating System :: OS Independent",
    ]
)
