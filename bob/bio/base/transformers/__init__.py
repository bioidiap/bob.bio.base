from collections import defaultdict
import os
import logging

logger = logging.getLogger(__name__)


def split_X_by_y(X, y):
    training_data = defaultdict(list)
    for x1, y1 in zip(X, y):
        training_data[y1].append(x1)
    training_data = list(training_data.values())
    return training_data


class _LazyModelLoader:
    def __init__(self, model_path=None, load_method_name="load", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model_loaded = model_path is None
        self.load_method_name = load_method_name

    def load(self):
        if self.model_loaded:
            return True
        # if the model exists, load it instead
        if os.path.isfile(self.model_path):
            logger.info("Found a checkpoint for {self}. Loading ...")
            getattr(self.instance, self.load_method_name)(self.model_path)
            self.model_loaded = True

        return self.model_loaded


from .preprocessor import (
    TransformerPreprocessor,
    SamplePreprocessor,
    CheckpointPreprocessor,
)

from .extractor import (
    TransformerExtractor,
    SampleExtractor,
    CheckpointExtractor,
)
from .algorithm import (
    TransformerAlgorithm,
    SampleAlgorithm,
    CheckpointAlgorithm,
)


def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened.

    Parameters
    ----------
    *args
        The objects that you want sphinx to beleive that are defined here.

    Resolves `Sphinx referencing issues <https//github.com/sphinx-
    doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    TransformerPreprocessor,
    SamplePreprocessor,
    CheckpointPreprocessor,
    TransformerExtractor,
    SampleExtractor,
    CheckpointExtractor,
    TransformerAlgorithm,
    SampleAlgorithm,
    CheckpointAlgorithm,
)

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
