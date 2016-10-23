from .Extractor import Extractor
from .Linearize import Linearize

# to fix sphinx warnings of not being able to find classes, when path is shortened
Extractor.__module__ = "bob.bio.base.extractor"
Linearize.__module__ = "bob.bio.base.extractor"

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
