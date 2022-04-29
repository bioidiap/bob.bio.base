import numpy

from bob.bio.base.database import BioFile
from bob.bio.base.preprocessor import Preprocessor

numpy.random.seed(10)


class DummyPreprocessor(Preprocessor):
    def __init__(self, return_none=False, probability_of_none=1, **kwargs):
        Preprocessor.__init__(self)
        self.return_none = return_none
        self.probability_of_none = probability_of_none

    def __call__(self, data, annotation):
        """Does nothing, simply converts the data type of the data, ignoring any annotation."""
        if self.return_none:
            return numpy.random.choice(
                [None, data],
                p=[self.probability_of_none, 1 - self.probability_of_none],
            )

        return data


preprocessor = DummyPreprocessor()


class DummyPreprocessorMetadata(DummyPreprocessor):
    def __call__(self, data, annotation, metadata=None):
        """Does nothing, simply converts the data type of the data, ignoring any annotation."""
        assert isinstance(metadata, BioFile)
        return super(DummyPreprocessorMetadata, self).__call__(data, annotation)


preprocessor_metadata = DummyPreprocessorMetadata()
