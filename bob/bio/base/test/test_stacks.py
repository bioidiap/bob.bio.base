from functools import partial
import numpy as np
from bob.bio.base.utils.processors import (
    SequentialProcessor, ParallelProcessor)
from bob.bio.base.preprocessor import (
    SequentialPreprocessor, ParallelPreprocessor, CallablePreprocessor)
from bob.bio.base.extractor import (
    SequentialExtractor, ParallelExtractor, CallableExtractor)

DATA = [0, 1, 2, 3, 4]
PROCESSORS = [partial(np.power, 2), np.mean]
SEQ_DATA = PROCESSORS[1](PROCESSORS[0](DATA))
PAR_DATA = (PROCESSORS[0](DATA), PROCESSORS[1](DATA))


def test_processors():
  proc = SequentialProcessor(PROCESSORS)
  data = proc(DATA)
  assert np.allclose(data, SEQ_DATA)

  proc = ParallelProcessor(PROCESSORS)
  data = proc(DATA)
  assert all(np.allclose(x1, x2) for x1, x2 in zip(data, PAR_DATA))


def test_preprocessors():
  processors = [CallablePreprocessor(p, False) for p in PROCESSORS]
  proc = SequentialPreprocessor(processors)
  data = proc(DATA, None)
  assert np.allclose(data, SEQ_DATA)

  proc = ParallelPreprocessor(processors)
  data = proc(DATA, None)
  assert all(np.allclose(x1, x2) for x1, x2 in zip(data, PAR_DATA))


def test_extractors():
  processors = [CallableExtractor(p) for p in PROCESSORS]
  proc = SequentialExtractor(processors)
  data = proc(DATA)
  assert np.allclose(data, SEQ_DATA)

  proc = ParallelExtractor(processors)
  data = proc(DATA)
  assert all(np.allclose(x1, x2) for x1, x2 in zip(data, PAR_DATA))
