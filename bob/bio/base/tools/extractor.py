import bob.io.base
import os

import logging
logger = logging.getLogger("bob.bio.base")

from .FileSelector import FileSelector
from .preprocessor import read_preprocessed_data
from .. import utils

def train_extractor(extractor, preprocessor, force = False):
  """Trains the feature extractor using preprocessed data of the 'world' set, if the feature extractor requires training."""
  if extractor.requires_training:
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



def extract(extractor, preprocessor, groups=None, indices = None, force=False):
  """Extracts the features from the preprocessed data using the given extractor."""
  # the file selector object
  fs = FileSelector.instance()
  extractor.load(fs.extractor_file)
  data_files = fs.preprocessed_data_list(groups=groups)
  feature_files = fs.feature_list(groups=groups)

  # select a subset of indices to iterate
  if indices != None:
    index_range = range(indices[0], indices[1])
    logger.info("- Extraction: splitting of index range %s" % str(indices))
  else:
    index_range = range(len(data_files))

  logger.info("- Extraction: extracting %d features from directory '%s' to directory '%s'", len(index_range), fs.preprocessed_directory, fs.extracted_directory)
  for i in index_range:
    data_file = str(data_files[i])
    feature_file = str(feature_files[i])

    if not utils.check_file(feature_file, force, 1000):
      # load data
      data = preprocessor.read_data(data_file)
      # extract feature
      feature = extractor(data)
      # write feature
      bob.io.base.create_directories_safe(os.path.dirname(feature_file))
      extractor.write_feature(feature, feature_file)

def read_features(file_names, extractor, split_by_client=False):
  """Reads the features from ``file_names`` using the given ``extractor``.
  If ``split_by_client`` is set to ``True``, it is assumed that the ``file_names`` are already sorted by client.
  """
  if split_by_client:
    return [[extractor.read_feature(str(f)) for f in client_files] for client_files in file_names]
  else:
    return [extractor.read_feature(str(f)) for f in file_names]
