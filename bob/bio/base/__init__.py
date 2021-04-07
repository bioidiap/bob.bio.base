from .utils import *
from . import database
from . import preprocessor
from . import extractor
from . import algorithm
from . import annotator
from . import pipelines

from . import script
from . import test
from . import score


def get_config():
    """Returns a string containing the configuration information.
  """
    import bob.extension

    return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
