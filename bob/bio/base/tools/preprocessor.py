import bob.io.base
import os

import logging
import inspect
logger = logging.getLogger("bob.bio.base")

from .FileSelector import FileSelector
from .. import utils


def preprocess(preprocessor, groups = None, indices = None, allow_missing_files = False, force = False):
  """Preprocesses the original data of the database with the given preprocessor.

  The given ``preprocessor`` is used to preprocess all data required for the current experiment.
  It writes the preprocessed data into the directory specified by the :py:class:`bob.bio.base.tools.FileSelector`.
  By default, if target files already exist, they are not re-created.

  **Parameters:**

  preprocessor : py:class:`bob.bio.base.preprocessor.Preprocessor` or derived
    The preprocessor, which should be applied to all data.

  groups : some of ``('world', 'dev', 'eval')`` or ``None``
    The list of groups, for which the data should be preprocessed.

  indices : (int, int) or None
    If specified, only the data for the given index range ``range(begin, end)`` should be preprocessed.
    This is usually given, when parallel threads are executed.

  allow_missing_files : bool
    If set to ``True``, files for which the preprocessor returns ``None`` are silently ignored.

  force : bool
    If given, files are regenerated, even if they already exist.
  """
  if not preprocessor.writes_data:
    # The preprocessor does not write anything, so no need to call it
    logger.info("Skipping preprocessing as preprocessor does not write any data")
    return

  # the file selector object
  fs = FileSelector.instance()

  # get the file lists
  data_files = fs.original_data_list(groups=groups)
  original_directory, original_extension = fs.original_directory_and_extension()
  preprocessed_data_files = fs.preprocessed_data_list(groups=groups)

  if utils.is_argument_available("metadata", preprocessor.__call__):
    metadata = fs.original_data_list(groups=groups)
  else:
    metadata = None

  # select a subset of keys to iterate
  if indices is not None:
    index_range = range(indices[0], indices[1])
    logger.info("- Preprocessing: splitting of index range %s", str(indices))
  else:
    index_range = range(len(data_files))

  logger.info("- Preprocessing: processing %d data files from directory '%s' to directory '%s'", len(index_range), fs.directories['original'], fs.directories['preprocessed'])

  # read annotation files
  annotation_list = fs.annotation_list(groups=groups)

  # iterate over the selected files  
  for i in index_range:
    preprocessed_data_file = preprocessed_data_files[i]
    file_object = data_files[i]
    file_name = file_object.make_path(original_directory, original_extension)

    # check for existence
    if not utils.check_file(preprocessed_data_file, force,
                            preprocessor.min_preprocessed_file_size):
      logger.debug("... Processing original data file '%s' (%d/%d)", file_name,
          index_range.index(i)+1, len(index_range))

      data = preprocessor.read_original_data(file_object, original_directory, original_extension)
      # create output directory before reading the data file (is sometimes required, when relative directories are specified, especially, including a .. somewhere)
      bob.io.base.create_directories_safe(os.path.dirname(preprocessed_data_file))

      # get the annotations; might be None
      annotations = fs.get_annotations(annotation_list[i])

      # call the preprocessor
      if metadata is None:
        preprocessed_data = preprocessor(data, annotations)
      else:
        preprocessed_data = preprocessor(data, annotations, metadata=metadata[i])

      if preprocessed_data is None:
        if allow_missing_files:
          logger.debug("... Processing original data file '%s' was not successful", file_name)
          continue
        else:
          raise RuntimeError("Preprocessing of file '%s' was not successful" % file_name)

      # write the data
      preprocessor.write_data(preprocessed_data, preprocessed_data_file)

    else:
      logger.debug("... Skipping original data file '%s' since preprocessed data '%s' exists", file_name, preprocessed_data_file)    


def read_preprocessed_data(file_names, preprocessor, split_by_client = False, allow_missing_files = False):
  """read_preprocessed_data(file_names, preprocessor, split_by_client = False) -> preprocessed

  Reads the preprocessed data from ``file_names`` using the given preprocessor.
  If ``split_by_client`` is set to ``True``, it is assumed that the ``file_names`` are already sorted by client.

  **Parameters:**

  file_names : [str] or [[str]]
    A list of names of files to be read.
    If ``split_by_client = True``, file names are supposed to be split into groups.

  preprocessor : py:class:`bob.bio.base.preprocessor.Preprocessor` or derived
    The preprocessor, which can read the preprocessed data.

  split_by_client : bool
    Indicates if the given ``file_names`` are split into groups.

  allow_missing_files : bool
    If set to ``True``, preprocessed data files that are not found are silently ignored.

  **Returns:**

  preprocessed : [object] or [[object]]
    The list of preprocessed data, in the same order as in the ``file_names``.
  """
  file_names = utils.filter_missing_files(file_names, split_by_client, allow_missing_files and preprocessor.writes_data)

  if split_by_client:
    preprocessed = [[preprocessor.read_data(f) for f in client_files] for client_files in file_names]
  else:
    preprocessed = [preprocessor.read_data(f) for f in file_names]
  return utils.filter_none(preprocessed, split_by_client)
