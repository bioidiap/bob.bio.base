from .utils import *
from . import database
from . import preprocessor
from . import extractor
from . import algorithm
from . import tools
from . import grid # only one file, not complete directory
from . import annotator

from . import script
from . import test

def get_config():
  """Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
