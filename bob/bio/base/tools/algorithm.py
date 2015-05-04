import bob.io.base
import os

import logging
logger = logging.getLogger("bob.bio.base")

from .FileSelector import FileSelector
from .extractor import read_features
from .. import utils


def train_projector(algorithm, extractor, force=False):
  """Train the feature projector with the extracted features of the world group."""
  if algorithm.requires_projector_training:
    # the file selector object
    fs = FileSelector.instance()

    if utils.check_file(fs.projector_file, force, 1000):
      logger.info("- Projection: projector '%s' already exists.", fs.projector_file)
    else:
      bob.io.base.create_directories_safe(os.path.dirname(fs.projector_file))
      # train projector
      logger.info("- Projection: loading training data")
      train_files = fs.training_list('extracted', 'train_projector', arrange_by_client = algorithm.split_training_features_by_client)
      train_features = read_features(train_files, extractor, algorithm.split_training_features_by_client)
      if algorithm.split_training_features_by_client:
        logger.info("- Projection: training projector '%s' using %d identities: ", fs.projector_file, len(train_files))
      else:
        logger.info("- Projection: training projector '%s' using %d training files: ", fs.projector_file, len(train_files))

      # perform training
      algorithm.train_projector(train_features, fs.projector_file)



def project(algorithm, extractor, groups = None, indices = None, force=False):
  """Projects the features for all files of the database."""
  # load the projector file
  if algorithm.performs_projection:
    # the file selector object
    fs = FileSelector.instance()

    # load the projector
    algorithm.load_projector(fs.projector_file)

    feature_files = fs.feature_list(groups=groups)
    projected_files = fs.projected_list(groups=groups)

    # select a subset of indices to iterate
    if indices != None:
      index_range = range(indices[0], indices[1])
      logger.info("- Projection: splitting of index range %s", str(indices))
    else:
      index_range = range(len(feature_files))

    logger.info("- Projection: projecting %d features from directory '%s' to directory '%s'", len(index_range), fs.extracted_directory, fs.projected_directory)
    # extract the features
    for i in index_range:
      feature_file = str(feature_files[i])
      projected_file = str(projected_files[i])

      if not utils.check_file(projected_file, force, 1000):
        # load feature
        feature = extractor.read_feature(feature_file)
        # project feature
        projected = algorithm.project(feature)
        # write it
        bob.io.base.create_directories_safe(os.path.dirname(projected_file))
        algorithm.write_feature(projected, projected_file)



def train_enroller(algorithm, extractor, force=False):
  """Trains the model enroller using the extracted or projected features, depending on your setup of the agorithm."""
  if algorithm.requires_enroller_training:
    # the file selector object
    fs = FileSelector.instance()

    if utils.check_file(fs.enroller_file, force, 1000):
      logger.info("- Enrollment: enroller '%s' already exists.", fs.enroller_file)
    else:
      # define the tool that is required to read the features
      reader = algorithm if algorithm.use_projected_features_for_enrollment else extractor
      bob.io.base.create_directories_safe(os.path.dirname(fs.enroller_file))

      # first, load the projector
      algorithm.load_projector(fs.projector_file)

      # load training data
      train_files = fs.training_list('projected' if algorithm.use_projected_features_for_enrollment else 'extracted', 'train_enroller', arrange_by_client = True)
      logger.info("- Enrollment: loading %d enroller training files", len(train_files))
      train_features = read_features(train_files, reader, True)

      # perform training
      logger.info("- Enrollment: training enroller '%s' using %d identities: ", fs.enroller_file, len(train_features))
      algorithm.train_enroller(train_features, fs.enroller_file)



def enroll(algorithm, extractor, compute_zt_norm, indices = None, groups = ['dev', 'eval'], types = ['N', 'T'], force=False):
  """Enroll the models for 'dev' and 'eval' groups, for both models and T-Norm-models.
     This function uses the extracted or projected features to compute the models,
     depending on your setup of the base class Algorithm."""

  # the file selector object
  fs = FileSelector.instance()
  # read the projector file, if needed
  algorithm.load_projector(fs.projector_file)
  # read the model enrollment file
  algorithm.load_enroller(fs.enroller_file)

  # which tool to use to read the features...
  reader = algorithm if algorithm.use_projected_features_for_enrollment else extractor

  # Create Models
  if 'N' in types:
    for group in groups:
      model_ids = fs.model_ids(group)

      if indices != None:
        model_ids = model_ids[indices[0]:indices[1]]
        logger.info("- Enrollment: splitting of index range %s", str(indices))

      logger.info("- Enrollment: enrolling models of group '%s'", group)
      for model_id in model_ids:
        # Path to the model
        model_file = str(fs.model_file(model_id, group))

        # Removes old file if required
        if not utils.check_file(model_file, force, 1000):
          enroll_files = fs.enroll_files(model_id, group, 'projected' if algorithm.use_projected_features_for_enrollment else 'extracted')

          # load all files into memory
          enroll_features = [reader.read_feature(str(enroll_file)) for enroll_file in enroll_files]

          model = algorithm.enroll(enroll_features)
          # save the model
          bob.io.base.create_directories_safe(os.path.dirname(model_file))
          algorithm.write_model(model, model_file)

  # T-Norm-Models
  if 'T' in types and compute_zt_norm:
    for group in groups:
      t_model_ids = fs.t_model_ids(group)

      if indices != None:
        t_model_ids = t_model_ids[indices[0]:indices[1]]
        logger.info("- Enrollment: splitting of index range %s", str(indices))

      logger.info("- Enrollment: enrolling T-models of group '%s'", group)
      for t_model_id in t_model_ids:
        # Path to the model
        t_model_file = str(fs.t_model_file(t_model_id, group))

        # Removes old file if required
        if not utils.check_file(t_model_file, force, 1000):
          t_enroll_files = fs.t_enroll_files(t_model_id, group, 'projected' if algorithm.use_projected_features_for_enrollment else 'extracted')

          # load all files into memory
          t_enroll_features = [reader.read_feature(str(t_enroll_file)) for t_enroll_file in t_enroll_files]

          t_model = algorithm.enroll(t_enroll_features)
          # save model
          bob.io.base.create_directories_safe(os.path.dirname(t_model_file))
          algorithm.write_model(t_model, t_model_file)
