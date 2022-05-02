# isort: skip_file
from . import database  # noqa: F401
from .utils import *  # noqa: F401,F403
from . import (  # noqa: F401
    algorithm,
    annotator,
    extractor,
    pipelines,
    preprocessor,
    score,
    script,
    test,
)


def get_config():
    """Returns a string containing the configuration information."""
    import bob.extension

    return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
