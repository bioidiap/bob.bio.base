import bob.io.base
import os

import logging
logger = logging.getLogger("bob.bio.base")

from .FileSelector import FileSelector
from .extractor import read_features
from .. import utils


def train_projector(algorithm, extractor, allow_missing_files = False, force = False):
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

  allow_missing_files : bool
    If set to ``True``, extracted files that are not found are silently ignored during training.

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
    train_features = read_features(train_files, extractor, algorithm.split_training_features_by_client, allow_missing_files)
    if algorithm.split_training_features_by_client:
      logger.info("- Projection: training projector '%s' using %d identities: ", fs.projector_file, len(train_files))
    else:
      logger.info("- Projection: training projector '%s' using %d training files: ", fs.projector_file, len(train_files))

    # perform training
    algorithm.train_projector(train_features, fs.projector_file)



def project(algorithm, extractor, groups = None, indices = None, allow_missing_files = False, force = False):
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

  allow_missing_files : bool
    If set to ``True``, extracted files that are not found are silently ignored.

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
  if indices is not None:
    index_range = range(indices[0], indices[1])
    logger.info("- Projection: splitting of index range %s", str(indices))
  else:
    index_range = range(len(feature_files))

  logger.info("- Projection: projecting %d features from directory '%s' to directory '%s'", len(index_range), fs.directories['extracted'], fs.directories['projected'])
  # extract the features
  for i in index_range:
    feature_file = feature_files[i]
    projected_file = projected_files[i]

    if not os.path.exists(feature_file):
      if allow_missing_files:
        logger.debug("... Cannot find extracted feature file %s; skipping", feature_file)
        continue
      else:
        logger.error("Cannot find extracted feature file %s", feature_file)


    if not utils.check_file(projected_file, force, 1000):
      logger.debug("... Projecting features for file '%s'", feature_file)
      # create output directory before reading the data file (is sometimes required, when relative directories are specified, especially, including a .. somewhere)
      bob.io.base.create_directories_safe(os.path.dirname(projected_file))
      # load feature
      feature = extractor.read_feature(feature_file)
      # project feature
      projected = algorithm.project(feature)

      if projected is None:
        if allow_missing_files:
          logger.debug("... Projection for extracted file %s failed; skipping", feature_file)
          continue
        else:
          logger.error("Projection of file '%s' was not successful", feature_file)
        continue

      # write it
      algorithm.write_feature(projected, projected_file)

    else:
      logger.debug("... Skipping feature file '%s' since projected file '%s' exists", feature_file, projected_file)



def train_enroller(algorithm, extractor, allow_missing_files = False, force = False):
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

  allow_missing_files : bool
    If set to ``True``, extracted files that are not found are silently ignored during training.

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
    train_features = read_features(train_files, reader, True, allow_missing_files)

    # perform training
    logger.info("- Enrollment: training enroller '%s' using %d identities", fs.enroller_file, len(train_features))
    algorithm.train_enroller(train_features, fs.enroller_file)



def enroll(algorithm, extractor, compute_zt_norm, indices = None, groups = ['dev', 'eval'], types = ['N', 'T'], allow_missing_files = False, force = False):
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

  allow_missing_files : bool
    If set to ``True``, extracted or ptojected files that are not found are silently ignored.
    If none of the enroll files are found, no model file will be written.

  force : bool
    If given, files are regenerated, even if they already exist.
  """
  # the file selector object
  fs = FileSelector.instance()
  # read the projector file, if needed
  if algorithm.requires_projector_training:
    algorithm.load_projector(fs.projector_file)
  # read the model enrollment file
  algorithm.load_enroller(fs.enroller_file)

  # which tool to use to read the features...
  reader = algorithm if algorithm.use_projected_features_for_enrollment else extractor

  # Create Models
  if 'N' in types:
    for group in groups:
      model_ids = fs.model_ids(group)

      if indices is not None:
        model_ids = model_ids[indices[0]:indices[1]]
        logger.info("- Enrollment: splitting of index range %s", str(indices))

      logger.info("- Enrollment: enrolling models of group '%s'", group)
      for model_id in model_ids:
        # Path to the model
        model_file = fs.model_file(model_id, group)

        # Removes old file if required
        if not utils.check_file(model_file, force, 1000):
          enroll_files = fs.enroll_files(model_id, group, 'projected' if algorithm.use_projected_features_for_enrollment else 'extracted')

          if allow_missing_files:
            enroll_files = utils.filter_missing_files(enroll_files)
            if not enroll_files:
              logger.debug("... Skipping model file %s since no feature file could be found", model_file)
              continue

          logger.debug("... Enrolling model from %d features to file '%s'", len(enroll_files), model_file)
          bob.io.base.create_directories_safe(os.path.dirname(model_file))

          # load all files into memory
          enroll_features = [reader.read_feature(enroll_file) for enroll_file in enroll_files]

          model = algorithm.enroll(enroll_features)

          if model is None:
            if allow_missing_files:
              logger.debug("... Enrollment for model %s failed; skipping", model_id)
              continue
            else:
              logger.error("Enrollemnt of model '%s' was not successful", model_id)
            continue

          # save the model
          algorithm.write_model(model, model_file)

        else:
          logger.debug("... Skipping model file '%s' since it exists", model_file)


  # T-Norm-Models
  if 'T' in types and compute_zt_norm:
    for group in groups:
      t_model_ids = fs.t_model_ids(group)

      if indices is not None:
        t_model_ids = t_model_ids[indices[0]:indices[1]]
        logger.info("- Enrollment: splitting of index range %s", str(indices))

      logger.info("- Enrollment: enrolling T-models of group '%s'", group)
      for t_model_id in t_model_ids:
        # Path to the model
        t_model_file = fs.t_model_file(t_model_id, group)

        # Removes old file if required
        if not utils.check_file(t_model_file, force, 1000):
          t_enroll_files = fs.t_enroll_files(t_model_id, group, 'projected' if algorithm.use_projected_features_for_enrollment else 'extracted')

          if allow_missing_files:
            t_enroll_files = utils.filter_missing_files(t_enroll_files)
            if not t_enroll_files:
              logger.debug("... Skipping T-model file %s since no feature file could be found", t_model_file)
              continue

          logger.debug("... Enrolling T-model from %d features to file '%s'", len(t_enroll_files), t_model_file)
          bob.io.base.create_directories_safe(os.path.dirname(t_model_file))

          # load all files into memory
          t_enroll_features = [reader.read_feature(t_enroll_file) for t_enroll_file in t_enroll_files]

          t_model = algorithm.enroll(t_enroll_features)

          if t_model is None:
            if allow_missing_files:
              logger.debug("... Enrollment for T-model %s failed; skipping", t_model_id)
              continue
            else:
              logger.error("Enrollemnt of T-model '%s' was not successful", t_model_id)
            continue

          # save model
          algorithm.write_model(t_model, t_model_file)
        else:
          logger.debug("... Skipping T-model file '%s' since it exists", t_model_file)
