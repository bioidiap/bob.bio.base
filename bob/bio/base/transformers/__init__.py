# isort: skip_file
from collections import defaultdict


def split_X_by_y(X, y):
    training_data = defaultdict(list)
    for x1, y1 in zip(X, y):
        training_data[y1].append(x1)
    training_data = list(training_data.values())
    return training_data


from .preprocessor import PreprocessorTransformer
from .extractor import ExtractorTransformer
from .preprocessing import ReferenceIdEncoder

# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened.

    Parameters
    ----------
    *args
        An iterable of objects to modify

    Resolves `Sphinx referencing issues
    <https://github.com/sphinx-doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    PreprocessorTransformer,
    ExtractorTransformer,
    ReferenceIdEncoder,
)

__all__ = [_ for _ in dir() if not _.startswith("_")]
