import bob.bio.base
import bob.learn.linear
import pkg_resources
import os

import bob.io.base.test_utils

from . import utils

def test_resources():
  # loading by resource
  cls = bob.bio.base.load_resource("pca", "algorithm")
  assert isinstance (cls, bob.bio.base.algorithm.PCA)

  # loading by configuration file
  cls = bob.bio.base.load_resource(pkg_resources.resource_filename("bob.bio.base.config.algorithm", "pca.py"), "algorithm")
  assert isinstance (cls, bob.bio.base.algorithm.PCA)

  # loading by instatiation
  cls = bob.bio.base.load_resource("bob.bio.base.algorithm.PCA(10, distance_function=scipy.spatial.distance.euclidean)", "algorithm", imports=['bob.bio.base', 'scipy.spatial'])
  assert isinstance (cls, bob.bio.base.algorithm.PCA)


def test_io():
  # Test that bob.bio.base.load and save works as expected
  filename = bob.io.base.test_utils.temporary_filename()

  try:
    # with simple data
    d = utils.random_training_set((10), 1)
    bob.io.base.save(d, filename)
    d2 = bob.io.base.load(filename)
    assert (d==d2).all()

    # with complex data
    m = bob.learn.linear.Machine(20,20)
    bob.bio.base.save(m, filename)
    m2 = bob.learn.linear.Machine(bob.io.base.HDF5File(filename))
    assert m == m2

    # compressed
    bob.bio.base.save_compressed(d, filename, create_link=True)
    assert os.path.exists(filename)
    assert os.path.islink(filename+".tar.bz2")
    os.remove(filename+".tar.bz2")
    d3 = bob.bio.base.load_compressed(filename)
    assert (d==d3).all()

    # compressed with complex data
    bob.bio.base.save_compressed(m, filename, compression_type="", create_link=False)
    assert os.path.exists(filename)
    assert not os.path.exists(filename+".tar.bz2")
    hdf5 = bob.bio.base.open_compressed(filename, compression_type="")
    m3 = bob.learn.linear.Machine(hdf5)
    bob.bio.base.close_compressed(filename, hdf5, compression_type="", create_link=False)
    assert m == m2

  finally:
    # cleanup
    if os.path.exists(filename):
      os.remove(filename)
