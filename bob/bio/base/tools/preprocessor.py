import bob.io.base
import os

import logging
logger = logging.getLogger("bob.bio.base")

from .FileSelector import FileSelector
from .. import utils

def preprocess(preprocessor, groups=None, indices=None, force=False):
  """Preprocesses the original data of the database with the given preprocessor."""
  # the file selector object
  fs = FileSelector.instance()

  # get the file lists
  data_files = fs.original_data_list(groups=groups)
  preprocessed_data_files = fs.preprocessed_data_list(groups=groups)

  # select a subset of keys to iterate
  if indices != None:
    index_range = range(indices[0], indices[1])
    logger.info("- Preprocessing: splitting of index range %s", str(indices))
  else:
    index_range = range(len(data_files))

  logger.info("- Preprocessing: processing %d data files from directory '%s' to directory '%s'", len(index_range), fs.directories['original'], fs.directories['preprocessed'])

  # read annotation files
  annotation_list = fs.annotation_list(groups=groups)

  # iterate over the selected files
  for i in index_range:
    preprocessed_data_file = str(preprocessed_data_files[i])

    # check for existence
    if not utils.check_file(preprocessed_data_file, force, 1000):
      file_name = data_files[i]
      data = preprocessor.read_original_data(file_name)

      # get the annotations; might be None
      annotations = fs.get_annotations(annotation_list[i])

      # call the preprocessor
      preprocessed_data = preprocessor(data, annotations)
      if preprocessed_data is None:
        logger.error("Preprocessing of file %s was not successful", str(file_name))

      # write the data
      bob.io.base.create_directories_safe(os.path.dirname(preprocessed_data_file))
      preprocessor.write_data(preprocessed_data, preprocessed_data_file)


def read_preprocessed_data(file_names, preprocessor, split_by_client=False):
  """Reads the preprocessed data from ``file_names`` using the given preprocessor.
  If ``split_by_client`` is set to ``True``, it is assumed that the ``file_names`` are already sorted by client.
  """
  if split_by_client:
    return [[preprocessor.read_data(str(f)) for f in client_files] for client_files in file_names]
  else:
    return [preprocessor.read_data(str(f)) for f in file_names]
