import os
import tempfile
import tarfile
import collections  # this is needed for the sphinx documentation
import functools  # this is needed for the sphinx documentation
import numpy
import logging
logger = logging.getLogger("bob.bio.base")

from .. import database
import bob.io.base


def filter_missing_files(file_names, split_by_client=False, allow_missing_files=True):
  """This function filters out files that do not exist, but only if ``allow_missing_files`` is set to ``True``, otherwise the list of ``file_names`` is returned unaltered."""

  if not allow_missing_files:
    return file_names

  if split_by_client:
    # filter out missing files and empty clients
    existing_files = [
        [f for f in client_files if os.path.exists(f)] for client_files in file_names]
    existing_files = [
        client_files for client_files in existing_files if client_files]
  else:
    # filter out missing files
    existing_files = [f for f in file_names if os.path.exists(f)]
  return existing_files


def filter_none(data, split_by_client=False):
  """This function filters out ``None`` values from the given list (or list of lists, when ``split_by_client`` is enabled)."""

  if split_by_client:
    # filter out missing files and empty clients
    existing_data = [[d for d in client_data if d is not None]
                     for client_data in data]
    existing_data = [
        client_data for client_data in existing_data if client_data]
  else:
    # filter out missing files
    existing_data = [d for d in data if d is not None]
  return existing_data


def check_file(filename, force, expected_file_size=1):
  """Checks if the file with the given ``filename`` exists and has size greater or equal to ``expected_file_size``.
  If the file is to small, **or** if the ``force`` option is set to ``True``, the file is removed.
  This function returns ``True`` is the file exists (and has not been removed), otherwise ``False``"""
  if os.path.exists(filename):
    if force or os.path.getsize(filename) < expected_file_size:
      logger.debug("  .. Removing old file '%s'.", filename)
      os.remove(filename)
      return False
    else:
      return True
  return False


def read_original_data(biofile, directory, extension):
  """This function reads the original data using the given ``biofile`` instance.
  It simply calls ``load(directory, extension)`` from :py:class:`bob.bio.base.database.BioFile` or one of its derivatives.

  Parameters
  ----------

  ``biofile`` : :py:class:`bob.bio.base.database.BioFile` or one of its derivatives
    The file to read the original data.

  ``directory`` : str
    The base directory of the database.

  ``extension`` : str or ``None``
    The extension of the original data.
    Might be ``None`` if the ``biofile`` itself has the extension stored.

  Returns
  -------

  object:
    Whatver ``biofile.load`` returns; usually a :py:class:`numpy.ndarray`
  """
  assert isinstance(biofile, database.BioFile)
  return biofile.load(directory, extension)


def load(file):
  """Loads data from file. The given file might be an HDF5 file open for reading or a string."""
  if isinstance(file, bob.io.base.HDF5File):
    return file.read("array")
  else:
    return bob.io.base.load(file)


def save(data, file, compression=0):
  """Saves the data to file using HDF5. The given file might be an HDF5 file open for writing, or a string.
  If the given data contains a ``save`` method, this method is called with the given HDF5 file.
  Otherwise the data is written to the HDF5 file using the given compression."""
  f = file if isinstance(file, bob.io.base.HDF5File) else bob.io.base.HDF5File(file, 'w')
  if hasattr(data, 'save'):
    data.save(f)
  else:
    f.set("array", data, compression=compression)


def open_compressed(filename, open_flag='r', compression_type='bz2'):
  """Opens a compressed HDF5File with the given opening flags.
  For the 'r' flag, the given compressed file will be extracted to a local space.
  For 'w', an empty HDF5File is created.
  In any case, the opened HDF5File is returned, which needs to be closed using the close_compressed() function.
  """
  # create temporary HDF5 file name
  hdf5_file_name = tempfile.mkstemp('.hdf5', 'bob_')[1]

  if open_flag == 'r':
    # extract the HDF5 file from the given file name into a temporary file name
    tar = tarfile.open(filename, mode="r:" + compression_type)
    memory_file = tar.extractfile(tar.next())
    real_file = open(hdf5_file_name, 'wb')
    real_file.write(memory_file.read())
    del memory_file
    real_file.close()
    tar.close()

  return bob.io.base.HDF5File(hdf5_file_name, open_flag)


def close_compressed(filename, hdf5_file, compression_type='bz2', create_link=False):
  """Closes the compressed hdf5_file that was opened with open_compressed.
  When the file was opened for writing (using the 'w' flag in open_compressed), the created HDF5 file is compressed into the given file name.
  To be able to read the data using the real tools, a link with the correct extension might is created, when create_link is set to True.
  """
  hdf5_file_name = hdf5_file.filename
  is_writable = hdf5_file.writable
  hdf5_file.close()

  if is_writable:
    # create compressed tar file
    tar = tarfile.open(filename, mode="w:" + compression_type)
    tar.add(hdf5_file_name, os.path.basename(filename))
    tar.close()

  if create_link:
    extension = {'': '.tar', 'bz2': '.tar.bz2',
                 'gz': 'tar.gz'}[compression_type]
    link_file = filename + extension
    if not os.path.exists(link_file):
      os.symlink(os.path.basename(filename), link_file)

  # clean up locally generated files
  os.remove(hdf5_file_name)


def load_compressed(filename, compression_type='bz2'):
  """Extracts the data to a temporary HDF5 file using HDF5 and reads its contents.
  Note that, though the file name is .hdf5, it contains compressed data!
  Accepted compression types are 'gz', 'bz2', ''"""
  # read from compressed HDF5
  hdf5 = open_compressed(filename, 'r')
  data = hdf5.read("array")
  close_compressed(filename, hdf5)

  return data


def save_compressed(data, filename, compression_type='bz2', create_link=False):
  """Saves the data to a temporary file using HDF5.
  Afterwards, the file is compressed using the given compression method and saved using the given file name.
  Note that, though the file name will be .hdf5, it will contain compressed data!
  Accepted compression types are 'gz', 'bz2', ''"""
  # write to compressed HDF5 file
  hdf5 = open_compressed(filename, 'w')
  save(data, hdf5)
  close_compressed(filename, hdf5, compression_type, create_link)


def _generate_features(reader, paths, same_size=False,
                       allow_missing_files=False):
  """Load and stack features in a memory efficient way. This function is meant
  to be used inside :py:func:`vstack_features`.

  Parameters
  ----------
  reader : ``collections.Callable``
      See the documentation of :py:func:`vstack_features`.
  paths : ``collections.Iterable``
      See the documentation of :py:func:`vstack_features`.
  same_size : :obj:`bool`, optional
      See the documentation of :py:func:`vstack_features`.
  allow_missing_files : :obj:`bool`, optional
      See the documentation of :py:func:`vstack_features`.

  Yields
  ------
  object
      The first object returned is a tuple of :py:class:`numpy.dtype` of
      features and the shape of the first feature. The rest of objects are
      the actual values in features. The features are returned in C order.
  """

  shape_determined = False
  for i, path in enumerate(paths):
    if allow_missing_files and not os.path.isfile(path):
      logger.debug("... File %s, that does not exist, has been ignored.", path)
      continue

    feature = numpy.atleast_2d(reader(path))
    feature = numpy.ascontiguousarray(feature)
    if not shape_determined:
      shape_determined = True
      dtype = feature.dtype
      shape = list(feature.shape)
      yield (dtype, shape)
    else:
      # make sure all features have the same shape and dtype
      if same_size:
        assert shape == list(feature.shape)
      else:
        assert shape[1:] == list(feature.shape[1:])
      assert dtype == feature.dtype

    for value in feature.flat:
      yield value


def vstack_features(reader, paths, same_size=False, allow_missing_files=False):
  """Stacks all features in a memory efficient way.

  Parameters
  ----------
  reader : ``collections.Callable``
      The function to load the features. The function should only take one
      argument being the path to the features. Use
      :any:`functools.partial` to accommodate your reader to this format.
      The features returned by ``reader`` are expected to have the same
      :py:class:`numpy.dtype` and the same shape except for their first
      dimension. First dimension is should correspond to the number of samples.
  paths : ``collections.Iterable``
      An iterable of paths to iterate on. Whatever is inside path is given to
      ``reader`` so they do not need to be necessarily paths to actual files.
      If ``same_size`` is ``True``, ``len(paths)`` must be valid.
  same_size : :obj:`bool`, optional
      If ``True``, it assumes that arrays inside all the paths are the same
      shape. If you know the features are the same size in all paths, set this
      to ``True`` to improve the performance.
  allow_missing_files : :obj:`bool`, optional
      If ``True``, it assumes that the items inside paths are actual files and
      ignores the ones that do not exist.

  Returns
  -------
  numpy.ndarray
      The read features with the shape (n_samples, \*features_shape[1:]).

  Raises
  ------
  ValueError
      If both same_size and allow_missing_files are ``True``.

  Examples
  --------
  This function in a simple way is equivalent to calling
  ``numpy.vstack(reader(p) for p in paths)``.

  >>> import numpy
  >>> from bob.bio.base import vstack_features
  >>> def reader(path):
  ...     # in each file, there are 5 samples and features are 2 dimensional.
  ...     return numpy.arange(10).reshape(5,2)
  >>> paths = ['path1', 'path2']
  >>> all_features = vstack_features(reader, paths)
  >>> all_features
  array([[0, 1],
         [2, 3],
         [4, 5],
         [6, 7],
         [8, 9],
         [0, 1],
         [2, 3],
         [4, 5],
         [6, 7],
         [8, 9]])
  >>> all_features_with_more_memory = numpy.vstack(reader(p) for p in paths)
  >>> numpy.allclose(all_features, all_features_with_more_memory)
  True

  You can allocate the array at once to improve the performance if you know
  that all features in paths have the same shape and you know the total number
  of the paths:

  >>> vstack_features(reader, paths, same_size=True)
  array([[0, 1],
         [2, 3],
         [4, 5],
         [6, 7],
         [8, 9],
         [0, 1],
         [2, 3],
         [4, 5],
         [6, 7],
         [8, 9]])

  """
  if same_size and allow_missing_files:
    raise ValueError("Both same_size and allow_missing_files cannot be True at"
                     " the same time.")
  iterable = _generate_features(reader, paths, same_size, allow_missing_files)
  dtype, shape = next(iterable)
  if same_size:
    total_size = int(len(paths) * numpy.prod(shape))
    all_features = numpy.fromiter(iterable, dtype, total_size)
  else:
    all_features = numpy.fromiter(iterable, dtype)

  # the shape is assumed to be (n_samples, ...) it can be (5, 2) or (5, 3, 4).
  shape = list(shape)
  shape[0] = -1
  return numpy.reshape(all_features, shape, order='C')
