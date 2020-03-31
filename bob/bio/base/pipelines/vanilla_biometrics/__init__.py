# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path

from .pipeline import VanillaBiometrics, dask_vanilla_biometrics

__path__ = extend_path(__path__, __name__)
