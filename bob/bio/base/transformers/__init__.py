# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from collections import defaultdict
def split_X_by_y(X, y):    
    training_data = defaultdict(list)
    for x1, y1 in zip(X, y):
        training_data[y1].append(x1)
    training_data = list(training_data.values())
    return training_data



from .preprocessor import PreprocessorTransformer
from .extractor import ExtractorTransformer
from .algorithm import AlgorithmTransformer
