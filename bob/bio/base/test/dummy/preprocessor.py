
import bob.bio.base

class DummyPreprocessor (bob.bio.base.preprocessor.Preprocessor):
  def __init__(self):
    bob.bio.base.preprocessor.Preprocessor.__init__(self)

  def __call__(self, data, annotation):
    """Does nothing, simply converts the data type of the data, ignoring any annotation."""
    return data

preprocessor = DummyPreprocessor()
