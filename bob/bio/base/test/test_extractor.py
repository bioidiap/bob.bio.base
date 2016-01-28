import bob.bio.base
import bob.io.base.test_utils
import os
import numpy

from . import utils

def test_linearize():
  # load extractor
  extractor = bob.bio.base.load_resource("linearize", "extractor", preferred_package = 'bob.bio.base')

  # generate input
  data = utils.random_training_set((10,10), 1)[0]
  assert len(data.shape) == 2

  # extract features
  extracted = extractor(data)
  assert len(extracted.shape) == 1
  assert extracted.shape[0] == data.shape[0] * data.shape[1]
  assert extracted.dtype == data.dtype

  # test IO
  filename = bob.io.base.test_utils.temporary_filename()
  try:
    extractor.write_feature(extracted, filename)
    extracted2 = extractor.read_feature(filename)

    assert (extracted == extracted2).all()

  finally:
    if os.path.exists(filename):
      os.remove(filename)

  # extract with dtype
  extractor = bob.bio.base.extractor.Linearize(dtype=numpy.complex128)
  extracted = extractor(data)
  assert len(extracted.shape) == 1
  assert extracted.shape[0] == data.shape[0] * data.shape[1]
  assert extracted.dtype == numpy.complex128
