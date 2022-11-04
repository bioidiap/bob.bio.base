# isort: skip_file
from .utils import *  # noqa: F401,F403
from . import database  # noqa: F401
from . import preprocessor  # noqa: F401
from . import extractor  # noqa: F401
from . import algorithm  # noqa: F401
from . import annotator  # noqa: F401
from . import pipelines  # noqa: F401

from . import script  # noqa: F401
from . import score  # noqa: F401


def get_config():
    """Returns a string containing the configuration information."""
    import bob.extension

    return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
