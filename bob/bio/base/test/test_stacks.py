from functools import partial
import numpy as np
import tempfile
from bob.bio.base.preprocessor import (
    SequentialPreprocessor, ParallelPreprocessor, CallablePreprocessor)
from bob.bio.base.extractor import (
    SequentialExtractor, ParallelExtractor, CallableExtractor)
from bob.bio.base.test.dummy.extractor import extractor as dummy_extractor

DATA = [0, 1, 2, 3, 4]
PROCESSORS = [partial(np.power, 2), np.mean]
SEQ_DATA = PROCESSORS[1](PROCESSORS[0](DATA))
PAR_DATA = (PROCESSORS[0](DATA), PROCESSORS[1](DATA))


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
  proc.load(None)
  data = proc(DATA)
  assert np.allclose(data, SEQ_DATA)

  proc = ParallelExtractor(processors)
  proc.load(None)
  data = proc(DATA)
  assert all(np.allclose(x1, x2) for x1, x2 in zip(data, PAR_DATA))


def test_sequential_trainable_extractors():
  processors = [CallableExtractor(p) for p in PROCESSORS] + [dummy_extractor]
  proc = SequentialExtractor(processors)
  with tempfile.NamedTemporaryFile(suffix='.hdf5') as f:
    proc.train(DATA, f.name)
    proc.load(f.name)
  data = proc(DATA)
  assert np.allclose(data, SEQ_DATA)


def test_parallel_trainable_extractors():
  processors = [CallableExtractor(p) for p in PROCESSORS] + [dummy_extractor]
  proc = ParallelExtractor(processors)
  with tempfile.NamedTemporaryFile(suffix='.hdf5') as f:
    proc.train(DATA, f.name)
    proc.load(f.name)
  data = proc(np.array(DATA))
  assert all(np.allclose(x1, x2) for x1, x2 in zip(data, PAR_DATA))
