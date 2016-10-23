from .Algorithm import Algorithm
from .Distance import Distance
from .PCA import PCA
from .LDA import LDA
from .PLDA import PLDA
from .BIC import BIC

# to fix sphinx warnings of not being able to find classes, when path is shortened
Algorithm.__module__ = "bob.bio.base.algorithm"
Distance.__module__ = "bob.bio.base.algorithm"
PCA.__module__ = "bob.bio.base.algorithm"
LDA.__module__ = "bob.bio.base.algorithm"
PLDA.__module__ = "bob.bio.base.algorithm"
BIC.__module__ = "bob.bio.base.algorithm"

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
