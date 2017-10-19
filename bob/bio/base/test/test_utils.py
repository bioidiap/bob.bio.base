import bob.bio.base
import bob.learn.linear
import pkg_resources
import os
import numpy
import nose
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

  # get list of extensions
  extensions = bob.bio.base.extensions()
  assert isinstance(extensions, list)
  assert 'bob.bio.base' in extensions


def test_grid():
  # try to load the grid configurations
  g = bob.bio.base.load_resource("grid", "grid")
  assert not g.is_local()
  g = bob.bio.base.load_resource("demanding", "grid")
  assert not g.is_local()

  g = bob.bio.base.load_resource("local-p4", "grid")
  assert g.is_local()
  assert g.number_of_parallel_processes == 4
  g = bob.bio.base.load_resource("local-p8", "grid")
  assert g.is_local()
  assert g.number_of_parallel_processes == 8
  g = bob.bio.base.load_resource("local-p16", "grid")
  assert g.is_local()
  assert g.number_of_parallel_processes == 16


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


def test_io_vstack():

  paths = [1, 2, 3, 4, 5]

  def oracle(reader, paths):
    return numpy.vstack([reader(p) for p in paths])

  def reader_same_size_C(path):
    return numpy.arange(10).reshape(5, 2)

  def reader_different_size_C(path):
    return numpy.arange(2 * path).reshape(path, 2)

  def reader_same_size_F(path):
    return numpy.asfortranarray(numpy.arange(10).reshape(5, 2))

  def reader_different_size_F(path):
    return numpy.asfortranarray(numpy.arange(2 * path).reshape(path, 2))

  def reader_same_size_C2(path):
    return numpy.arange(30).reshape(5, 2, 3)

  def reader_different_size_C2(path):
    return numpy.arange(6 * path).reshape(path, 2, 3)

  def reader_same_size_F2(path):
    return numpy.asfortranarray(numpy.arange(30).reshape(5, 2, 3))

  def reader_different_size_F2(path):
    return numpy.asfortranarray(numpy.arange(6 * path).reshape(path, 2, 3))

  def reader_wrong_size(path):
    return numpy.arange(2 * path).reshape(2, path)

  # when same_size is False
  for reader in [
      reader_different_size_C,
      reader_different_size_F,
      reader_same_size_C,
      reader_same_size_F,
      reader_different_size_C2,
      reader_different_size_F2,
      reader_same_size_C2,
      reader_same_size_F2,
  ]:
    numpy.all(bob.bio.base.vstack_features(reader, paths) ==
              oracle(reader, paths))

  # when same_size is True
  for reader in [
      reader_same_size_C,
      reader_same_size_F,
      reader_same_size_C2,
      reader_same_size_F2,
  ]:
    numpy.all(bob.bio.base.vstack_features(reader, paths, True) ==
              oracle(reader, paths))

  with nose.tools.assert_raises(AssertionError):
    bob.bio.base.vstack_features(reader_wrong_size, paths)

  # test actual files
  paths = [bob.io.base.test_utils.temporary_filename(),
           bob.io.base.test_utils.temporary_filename(),
           bob.io.base.test_utils.temporary_filename()]
  try:
    # try different readers:
    for reader in [
        reader_different_size_C,
        reader_different_size_F,
        reader_same_size_C,
        reader_same_size_F,
        reader_different_size_C2,
        reader_different_size_F2,
        reader_same_size_C2,
        reader_same_size_F2,
    ]:
      # save some data in files
      for i, path in enumerate(paths):
        bob.bio.base.save(reader(i + 1), path)
      # test when all data is present
      reference = oracle(bob.bio.base.load, paths)
      numpy.all(bob.bio.base.vstack_features(bob.bio.base.load, paths) ==
                reference)
      # delete the first one
      os.remove(paths[0])
      reference = oracle(bob.bio.base.load, paths[1:])
      target = bob.bio.base.vstack_features(bob.bio.base.load, paths, False,
                                            True)
      numpy.all(target == reference)
      # save back first one and delete second one
      bob.bio.base.save(reader(1), paths[0])
      os.remove(paths[1])
      reference = oracle(bob.bio.base.load, paths[:1] + paths[2:])
      target = bob.bio.base.vstack_features(bob.bio.base.load, paths, False,
                                            True)
      numpy.all(target == reference)
      # Check if RuntimeError is raised when one of the files is missing and
      # allow_missing_files if False
      with nose.tools.assert_raises(RuntimeError):
        bob.bio.base.vstack_features(bob.bio.base.load, paths)
      # Check if ValueError is raised.
      with nose.tools.assert_raises(ValueError):
        bob.bio.base.vstack_features(bob.bio.base.load, paths, True, True)
  finally:
    try:
      for path in paths:
        os.remove(path)
    except Exception:
      pass


def test_sampling():
  # test selection of elements
  indices = bob.bio.base.selected_indices(100, 10)
  assert indices == list(range(5, 100, 10))

  indices = bob.bio.base.selected_indices(100, 300)
  assert indices == range(100)

  indices = bob.bio.base.selected_indices(100, None)
  assert indices == range(100)

  array = numpy.arange(100)
  elements = bob.bio.base.selected_elements(array, 10)
  assert (elements - numpy.arange(5, 100, 10) == 0.).all()

  elements = bob.bio.base.selected_elements(array, 200)
  assert (elements - numpy.arange(100) == 0.).all()

  elements = bob.bio.base.selected_elements(array, None)
  assert (elements - numpy.arange(100) == 0.).all()
