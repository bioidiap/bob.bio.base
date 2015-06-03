from bob.bio.base.preprocessor import Preprocessor

class DummyPreprocessor (Preprocessor):
  def __init__(self, **kwargs):
    Preprocessor.__init__(self)

  def __call__(self, data, annotation):
    """Does nothing, simply converts the data type of the data, ignoring any annotation."""
    return data

preprocessor = DummyPreprocessor()
