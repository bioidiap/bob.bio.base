from .Algorithm import Algorithm
from .Distance import Distance
from .PCA import PCA
from .LDA import LDA
from .PLDA import PLDA
from .BIC import BIC

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
