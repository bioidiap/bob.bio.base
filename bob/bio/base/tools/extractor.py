import bob.io.base
import os

import logging
logger = logging.getLogger("bob.bio.base")

from .FileSelector import FileSelector
from .preprocessor import read_preprocessed_data
from .. import utils

def train_extractor(extractor, preprocessor, force = False):
  """Trains the feature extractor using preprocessed data of the ``'world'`` group, if the feature extractor requires training.

  This function should only be called, when the ``extractor`` actually requires training.
  The given ``extractor`` is trained using preprocessed data.
  It writes the extractor to the file specified by the :py:class:`bob.bio.base.tools.FileSelector`.
  By default, if the target file already exist, it is not re-created.

  **Parameters:**

  extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
    The extractor to be trained.

  preprocessor : py:class:`bob.bio.base.preprocessor.Preprocessor` or derived
    The preprocessor, used for reading the preprocessed data.

  force : bool
    If given, the extractor file is regenerated, even if it already exists.
  """

  if not extractor.requires_training:
    logger.warn("The train_extractor function should not have been called, since the extractor does not need training.")
    return

  # the file selector object
  fs = FileSelector.instance()
  # the file to write
  if utils.check_file(fs.extractor_file, force, 1000):
    logger.info("- Extraction: extractor '%s' already exists.", fs.extractor_file)
  else:
    # read training files
    train_files = fs.training_list('preprocessed', 'train_extractor', arrange_by_client = extractor.split_training_data_by_client)
    train_data = read_preprocessed_data(train_files, preprocessor, extractor.split_training_data_by_client)
    if extractor.split_training_data_by_client:
      logger.info("- Extraction: training extractor '%s' using %d identities:", fs.extractor_file, len(train_files))
    else:
      logger.info("- Extraction: training extractor '%s' using %d training files:", fs.extractor_file, len(train_files))
    # train model
    bob.io.base.create_directories_safe(os.path.dirname(fs.extractor_file))
    extractor.train(train_data, fs.extractor_file)



def extract(extractor, preprocessor, groups=None, indices = None, force = False):
  """Extracts features from the preprocessed data using the given extractor.

  The given ``extractor`` is used to extract all features required for the current experiment.
  It writes the extracted data into the directory specified by the :py:class:`bob.bio.base.tools.FileSelector`.
  By default, if target files already exist, they are not re-created.

  The preprocessor is only used to load the data in a coherent way.

  **Parameters:**

  extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
    The extractor, used for extracting and writing the features.

  preprocessor : py:class:`bob.bio.base.preprocessor.Preprocessor` or derived
    The preprocessor, used for reading the preprocessed data.

  groups : some of ``('world', 'dev', 'eval')`` or ``None``
    The list of groups, for which the data should be extracted.

  indices : (int, int) or None
    If specified, only the features for the given index range ``range(begin, end)`` should be extracted.
    This is usually given, when parallel threads are executed.

  force : bool
    If given, files are regenerated, even if they already exist.
  """
  # the file selector object
  fs = FileSelector.instance()
  extractor.load(fs.extractor_file)
  data_files = fs.preprocessed_data_list(groups=groups)
  feature_files = fs.feature_list(groups=groups)

  # select a subset of indices to iterate
  if indices is not None:
    index_range = range(indices[0], indices[1])
    logger.info("- Extraction: splitting of index range %s" % str(indices))
  else:
    index_range = range(len(data_files))

  logger.info("- Extraction: extracting %d features from directory '%s' to directory '%s'", len(index_range), fs.directories['preprocessed'], fs.directories['extracted'])
  for i in index_range:
    data_file = data_files[i]
    feature_file = feature_files[i]

    if not utils.check_file(feature_file, force, 1000):
      logger.debug("... Extracting features for data file '%s'", data_file)
      # load data
      data = preprocessor.read_data(data_file)
      # extract feature
      feature = extractor(data)
      # write feature
      bob.io.base.create_directories_safe(os.path.dirname(feature_file))
      extractor.write_feature(feature, feature_file)
    else:
      logger.debug("... Skipping preprocessed data '%s' since feature file '%s' exists", data_file, feature_file)


def read_features(file_names, extractor, split_by_client = False):
  """read_features(file_names, extractor, split_by_client = False) -> extracted

  Reads the extracted features from ``file_names`` using the given ``extractor``.
  If ``split_by_client`` is set to ``True``, it is assumed that the ``file_names`` are already sorted by client.

  **Parameters:**

  file_names : [str] or [[str]]
    A list of names of files to be read.
    If ``split_by_client = True``, file names are supposed to be split into groups.

  extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
    The extractor, used for reading the extracted features.

  split_by_client : bool
    Indicates if the given ``file_names`` are split into groups.

  **Returns:**

  extracted : [object] or [[object]]
    The list of extracted features, in the same order as in the ``file_names``.
  """
  if split_by_client:
    return [[extractor.read_feature(f) for f in client_files] for client_files in file_names]
  else:
    return [extractor.read_feature(f) for f in file_names]
