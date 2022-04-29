import numpy

import bob.bio.base

from bob.bio.base.database import BioFile
from bob.bio.base.extractor import Extractor

_data = [0.0, 1.0, 2.0, 3.0, 4.0]


class DummyExtractor(Extractor):
    def __init__(self, **kwargs):
        Extractor.__init__(self, requires_training=True)
        self.model = False

    def train(self, train_data, extractor_file):
        assert isinstance(train_data, list)
        bob.bio.base.save(_data, extractor_file)

    def load(self, extractor_file):
        data = bob.bio.base.load(extractor_file)
        assert (_data == data).all()
        self.model = True

    def __call__(self, data):
        """Does nothing, simply converts the data type of the data, ignoring any annotation."""
        assert self.model
        return data.astype(numpy.float).flatten()


extractor = DummyExtractor()


class DummyExtractorMetadata(DummyExtractor):
    def __call__(self, data, metadata=None):
        """Does nothing, simply converts the data type of the data, ignoring any annotation."""
        assert isinstance(metadata, BioFile)
        return super(DummyExtractorMetadata, self).__call__(data)


extractor_metadata = DummyExtractorMetadata()
