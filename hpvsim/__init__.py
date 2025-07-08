from .version import __version__, __versiondate__, __license__

from .hpv import *
from .parameters import *
from .connectors import *
from .distributions import *
from .interventions import *
from .analyzers import *
from .utils import *
from .sim import *

# Assign the root folder
import sciris as sc

root = sc.thispath(__file__).parent
# data = root/'data'

# Import the version and print the license
print(__license__)

# Double-check key requirements -- should match setup.py
sc.require(
    ["starsim>=2.2.0", "sciris>=3.1.6", "pandas>=2.0.0", "stisim", "scipy"],
    message=f"The following dependencies for vagisim {__version__} were not met: <MISSING>.",
)
del sc  # Clean up namespace
