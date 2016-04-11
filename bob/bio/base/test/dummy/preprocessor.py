from bob.bio.base.preprocessor import Preprocessor

class DummyPreprocessor (Preprocessor):
  def __init__(self, return_none=False, **kwargs):
    Preprocessor.__init__(self)
    self.return_none = return_none

  def __call__(self, data, annotation):
    """Does nothing, simply converts the data type of the data, ignoring any annotation."""
    if self.return_none:
      return None
    return data

preprocessor = DummyPreprocessor()
