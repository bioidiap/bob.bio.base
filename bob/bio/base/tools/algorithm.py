import bob.io.base
import os

import logging
logger = logging.getLogger("bob.bio.base")

from .FileSelector import FileSelector
from .extractor import read_features
from .. import utils


def train_projector(algorithm, extractor, force = False):
  """Trains the feature projector using extracted features of the ``'world'`` group, if the algorithm requires projector training.

  This function should only be called, when the ``algorithm`` actually requires projector training.
  The projector of the given ``algorithm`` is trained using extracted features.
  It writes the projector to the file specified by the :py:class:`bob.bio.base.tools.FileSelector`.
  By default, if the target file already exist, it is not re-created.

  **Parameters:**

  algorithm : py:class:`bob.bio.base.algorithm.Algorithm` or derived
    The algorithm, in which the projector should be trained.

  extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
    The extractor, used for reading the training data.

  force : bool
    If given, the projector file is regenerated, even if it already exists.
  """
  if not algorithm.requires_projector_training:
    logger.warn("The train_projector function should not have been called, since the algorithm does not need projector training.")
    return

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



def project(algorithm, extractor, groups = None, indices = None, force = False):
  """Projects the features for all files of the database.

  The given ``algorithm`` is used to project all features required for the current experiment.
  It writes the projected data into the directory specified by the :py:class:`bob.bio.base.tools.FileSelector`.
  By default, if target files already exist, they are not re-created.

  The extractor is only used to load the data in a coherent way.

  **Parameters:**

  algorithm : py:class:`bob.bio.base.algorithm.Algorithm` or derived
    The algorithm, used for projecting features and writing them to file.

  extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
    The extractor, used for reading the extracted features, which should be projected.

  groups : some of ``('world', 'dev', 'eval')`` or ``None``
    The list of groups, for which the data should be projected.

  indices : (int, int) or None
    If specified, only the features for the given index range ``range(begin, end)`` should be projected.
    This is usually given, when parallel threads are executed.

  force : bool
    If given, files are regenerated, even if they already exist.
  """
  if not algorithm.performs_projection:
    logger.warn("The project function should not have been called, since the algorithm does not perform projection.")
    return

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

  logger.info("- Projection: projecting %d features from directory '%s' to directory '%s'", len(index_range), fs.directories['extracted'], fs.directories['projected'])
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



def train_enroller(algorithm, extractor, force = False):
  """Trains the model enroller using the extracted or projected features, depending on your setup of the algorithm.

  This function should only be called, when the ``algorithm`` actually requires enroller training.
  The enroller of the given ``algorithm`` is trained using extracted or projected features.
  It writes the enroller to the file specified by the :py:class:`bob.bio.base.tools.FileSelector`.
  By default, if the target file already exist, it is not re-created.

  **Parameters:**

  algorithm : py:class:`bob.bio.base.algorithm.Algorithm` or derived
    The algorithm, in which the enroller should be trained.
    It is assured that the projector file is read (if required) before the enroller training is started.

  extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
    The extractor, used for reading the training data, if unprojected features are used for enroller training.

  force : bool
    If given, the enroller file is regenerated, even if it already exists.
  """
  if not algorithm.requires_enroller_training:
    logger.warn("The train_enroller function should not have been called, since the algorithm does not need enroller training.")
    return

  # the file selector object
  fs = FileSelector.instance()

  if utils.check_file(fs.enroller_file, force, 1000):
    logger.info("- Enrollment: enroller '%s' already exists.", fs.enroller_file)
  else:
    # define the tool that is required to read the features
    reader = algorithm if algorithm.use_projected_features_for_enrollment else extractor
    bob.io.base.create_directories_safe(os.path.dirname(fs.enroller_file))

    # first, load the projector
    if algorithm.requires_projector_training:
      algorithm.load_projector(fs.projector_file)

    # load training data
    train_files = fs.training_list('projected' if algorithm.use_projected_features_for_enrollment else 'extracted', 'train_enroller', arrange_by_client = True)
    logger.info("- Enrollment: loading %d enroller training files", len(train_files))
    train_features = read_features(train_files, reader, True)

    # perform training
    logger.info("- Enrollment: training enroller '%s' using %d identities: ", fs.enroller_file, len(train_features))
    algorithm.train_enroller(train_features, fs.enroller_file)



def enroll(algorithm, extractor, compute_zt_norm, indices = None, groups = ['dev', 'eval'], types = ['N', 'T'], force = False):
  """Enroll the models for the given groups, eventually for both models and T-Norm-models.
     This function uses the extracted or projected features to compute the models, depending on your setup of the given ``algorithm``.

  The given ``algorithm`` is used to enroll all models required for the current experiment.
  It writes the models into the directories specified by the :py:class:`bob.bio.base.tools.FileSelector`.
  By default, if target files already exist, they are not re-created.

  The extractor is only used to load features in a coherent way.

  **Parameters:**

  algorithm : py:class:`bob.bio.base.algorithm.Algorithm` or derived
    The algorithm, used for enrolling model and writing them to file.

  extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
    The extractor, used for reading the extracted features, if the algorithm enrolls models from unprojected data.

  compute_zt_norm : bool
    If set to ``True`` and `'T'`` is part of the ``types``, also T-norm models are extracted.

  indices : (int, int) or None
    If specified, only the models for the given index range ``range(begin, end)`` should be enrolled.
    This is usually given, when parallel threads are executed.

  groups : some of ``('dev', 'eval')``
    The list of groups, for which models should be enrolled.

  force : bool
    If given, files are regenerated, even if they already exist.
  """
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
