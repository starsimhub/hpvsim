from .version import __version__, __versiondate__
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

# Import data and check
from . import data
if not data.check_downloaded():
    try:
        data.quick_download(init=True)
    except:
        import sciris as sc
        errormsg = f"Warning: couldn't download data:\n\n{sc.traceback()}\nProceeding anyway..."
        print(errormsg)
