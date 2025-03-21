# This ensures SCIRIS_NUM_THREADS is utilized
import sciris as sc

# Import HPVsim
from .version import __version__, __versiondate__, __license__
from .settings      import *
from .defaults      import *
from .misc          import *
from .parameters    import *
from .utils         import *
from .plotting      import *
from .base          import *
from .people        import *
from .population    import *
from .interventions import *
from .immunity      import *
from .analysis      import *
from .sim           import *
from .run           import *
from .calibration   import *

# Import the version and print the license unless verbosity is disabled, via e.g. os.environ['HPVSIM_VERBOSE'] = 0
if settings.options.verbose:
    print(__license__)

# Import data and check
from . import data
if not data.check_downloaded():
    try:
        data.quick_download(init=True)
    except Exception as E1:
        try:
            print(f'Quick download failed ({str(E1)}), trying manual download ...')
            data.download_data(serial=True)
        except:
            errormsg = f"Warning: couldn't download data:\n\n{sc.traceback()}\nProceeding anyway..."
            print(errormsg)

# Set the root directory for the codebase
rootdir = sc.thispath(__file__).parent