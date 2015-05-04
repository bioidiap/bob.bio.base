import bob.bio.base
import numpy

_data = [0., 1., 2., 3., 4.]

class DummyExtractor (bob.bio.base.extractor.Extractor):
  def __init__(self):
    bob.bio.base.extractor.Extractor.__init__(self, requires_training=True)
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
