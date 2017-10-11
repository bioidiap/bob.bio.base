from ..utils.processors import SequentialProcessor, ParallelProcessor
from .Preprocessor import Preprocessor


class SequentialPreprocessor(SequentialProcessor, Preprocessor):
  __doc__ = SequentialProcessor.__doc__

  def __init__(self, processors, **kwargs):
    min_preprocessed_file_size = 1000
    try:
      min_preprocessed_file_size = min(
          (p.min_preprocessed_file_size for p in processors))
    except AttributeError:
      pass

    SequentialProcessor.__init__(self, processors)
    Preprocessor.__init__(
        self, min_preprocessed_file_size=min_preprocessed_file_size, **kwargs)

  def __call__(self, data, annotations):
    return super(SequentialPreprocessor, self).__call__(
        data, annotations=annotations)


class ParallelPreprocessor(ParallelProcessor, Preprocessor):
  __doc__ = ParallelProcessor.__doc__

  def __init__(self, processors, **kwargs):
    min_preprocessed_file_size = 1000
    try:
      min_preprocessed_file_size = min(
          (p.min_preprocessed_file_size for p in processors))
    except AttributeError:
      pass

    ParallelProcessor.__init__(self, processors)
    Preprocessor.__init__(
        self, min_preprocessed_file_size=min_preprocessed_file_size, **kwargs)

  def __call__(self, data, annotations):
    return super(ParallelPreprocessor, self).__call__(
        data, annotations=annotations)


class CallablePreprocessor(Preprocessor):
  """A simple preprocessor that takes a callable and applies that callable to
  the input.

  Attributes
  ----------
  callable : object
      Anything that is callable. It will be used as a preprocessor in
      bob.bio.base.
  """

  def __init__(self, callable, **kwargs):
    super(CallablePreprocessor, self).__init__(**kwargs)
    self.callable = callable

  def __call__(self, data, annotations):
    return self.callable(data)
