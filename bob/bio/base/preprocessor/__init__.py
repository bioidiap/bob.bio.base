from .Preprocessor import Preprocessor
from .Filename import Filename

# to fix sphinx warnings of not being able to find classes, when path is shortened
Preprocessor.__module__ = "bob.bio.base.preprocessor"
Filename.__module__ = "bob.bio.base.preprocessor"

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
