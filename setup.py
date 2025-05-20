from setuptools import setup, find_packages

setup(
    name="hpvsim",
    version="3.0.0",
    author="Jamie Cohen, Robyn Stuart, Daniel Klein, Cliff Kerr",
    author_email="jamie.cohen@gatesfoundation.org",
    description="HPVsim: A model for HPV transmission and disease progression within the Starsim framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/starsimhub/hpvsim_starsim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "starsim>=2.2.0",
        "stisim",
        "optuna",
    ],
)
