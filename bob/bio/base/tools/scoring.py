import bob.io.base
import bob.learn.em
import bob.learn.linear
import bob.measure
import numpy
import os, sys
import tarfile

import logging
logger = logging.getLogger("bob.bio.base")

from .FileSelector import FileSelector
from .extractor import read_features
from .. import utils

def _scores(algorithm, model, probes, allow_missing_files):
  """Compute scores for the given model and a list of probes.
  """
  # the file selector object
  fs = FileSelector.instance()
  # the scores to be computed; initialized with NaN
  scores = numpy.ones((1,len(probes)), numpy.float64) * numpy.nan

  if allow_missing_files and model is None:
    # if we have no model, all scores are undefined
    return scores

  # Loops over the probe sets
  for i, probe_element in enumerate(probes):
    if fs.uses_probe_file_sets():
      assert isinstance(probe_element, list)
      # filter missing files
      if allow_missing_files:
        probe_element = utils.filter_missing_files(probe_element)
        if not probe_element:
          # we keep the NaN score
          continue
      # read probe from probe_set
      probe = [algorithm.read_probe(probe_file) for probe_file in probe_element]
      # compute score
      scores[0,i] = algorithm.score_for_multiple_probes(model, probe)
    else:
      if allow_missing_files and not os.path.exists(probe_element):
        # we keep the NaN score
        continue
      # read probe
      probe = algorithm.read_probe(probe_element)
      # compute score
      scores[0,i] = algorithm.score(model, probe)
  # Returns the scores
  return scores


def _open_to_read(score_file):
  """Checks for the existence of the normal and the compressed version of the file, and calls :py:func:`bob.measure.load.open_file` for the existing one."""
  if not os.path.exists(score_file):
    score_file += '.tar.bz2'
    if not os.path.exists(score_file):
      raise IOError("The score file '%s' cannot be found. Aborting!" % score_file)
  return bob.measure.load.open_file(score_file)


def _open_to_write(score_file, write_compressed):
  """Opens the given score file for writing. If write_compressed is set to ``True``, a file-like structure is returned."""
  bob.io.base.create_directories_safe(os.path.dirname(score_file))
  if write_compressed:
    if sys.version_info[0] <= 2:
      import StringIO
      f = StringIO.StringIO()
    else:
      import io
      f = io.BytesIO()
    score_file += '.tar.bz2'
  else:
    f = open(score_file, 'w')

  return f

def _write(f, data, write_compressed):
  """Writes the given data to file, after converting it to the required type."""
  if write_compressed:
    if sys.version_info[0] > 2:
      data = str.encode(data)

  f.write(data)

def _close_written(score_file, f, write_compressed):
  """Closes the file f that was opened with :py:func:`_open_to_read`"""
  if write_compressed:
    f.seek(0)
    tarinfo = tarfile.TarInfo(os.path.basename(score_file))
    tarinfo.size = len(f.buf if sys.version_info[0] <= 2 else f.getbuffer())
    tar = tarfile.open(score_file, 'w')
    tar.addfile(tarinfo, f)
    tar.close()
  # close the file
  f.close()


def _save_scores(score_file, scores, probe_objects, client_id, write_compressed):
  """Saves the scores of one model into a text file that can be interpreted by :py:func:`bob.measure.load.split_four_column`."""
  assert len(probe_objects) == scores.shape[1]

  # open file for writing
  f = _open_to_write(score_file, write_compressed)

  # write scores in four-column format as string
  for i, probe_object in enumerate(probe_objects):
    _write(f, "%s %s %s %3.8f\n" % (str(client_id), str(probe_object.client_id), str(probe_object.path), scores[0,i]), write_compressed)

  _close_written(score_file, f, write_compressed)


def _scores_a(algorithm, model_ids, group, compute_zt_norm, force, write_compressed, allow_missing_files):
  """Computes A scores for the models with the given model_ids. If ``compute_zt_norm = False``, these are the only scores that are actually computed."""
  # the file selector object
  fs = FileSelector.instance()

  if compute_zt_norm:
    logger.info("- Scoring: computing score matrix A for group '%s'", group)
  else:
    logger.info("- Scoring: computing scores for group '%s'", group)

  # Computes the raw scores for each model
  for model_id in model_ids:
    # test if the file is already there
    score_file = fs.a_file(model_id, group) if compute_zt_norm else fs.no_norm_file(model_id, group)
    if utils.check_file(score_file, force):
      logger.warn("Score file '%s' already exists.", score_file)
    else:
      # get probe files that are required for this model
      current_probe_objects = fs.probe_objects_for_model(model_id, group)
      model_file = fs.model_file(model_id, group)
      if allow_missing_files and not os.path.exists(model_file):
        model = None
      else:
        model = algorithm.read_model(model_file)
      # get the probe files
      current_probe_files = fs.get_paths(current_probe_objects, 'projected' if algorithm.performs_projection else 'extracted')
      # compute scores
      a = _scores(algorithm, model, current_probe_files, allow_missing_files)

      if compute_zt_norm:
        # write A matrix only when you want to compute zt norm afterwards
        bob.io.base.save(a, fs.a_file(model_id, group), True)

      # Save scores to text file
      _save_scores(fs.no_norm_file(model_id, group), a, current_probe_objects, fs.client_id(model_id, group), write_compressed)


def _scores_b(algorithm, model_ids, group, force, allow_missing_files):
  """Computes B scores for the given model ids."""
  # the file selector object
  fs = FileSelector.instance()

  # probe files:
  z_probe_objects = fs.z_probe_objects(group)
  z_probe_files = fs.get_paths(z_probe_objects, 'projected' if algorithm.performs_projection else 'extracted')

  logger.info("- Scoring: computing score matrix B for group '%s'", group)

  # Loads the models
  for model_id in model_ids:
    # test if the file is already there
    score_file = fs.b_file(model_id, group)
    if utils.check_file(score_file, force):
      logger.warn("Score file '%s' already exists.", score_file)
    else:
      model_file = fs.model_file(model_id, group)
      if allow_missing_files and not os.path.exists(model_file):
        model = None
      else:
        model = algorithm.read_model(model_file)
      b = _scores(algorithm, model, z_probe_files, allow_missing_files)
      bob.io.base.save(b, score_file, True)

def _scores_c(algorithm, t_model_ids, group, force, allow_missing_files):
  """Computes C scores for the given t-norm model ids."""
  # the file selector object
  fs = FileSelector.instance()

  # probe files:
  probe_objects = fs.probe_objects(group)
  probe_files = fs.get_paths(probe_objects, 'projected' if algorithm.performs_projection else 'extracted')

  logger.info("- Scoring: computing score matrix C for group '%s'", group)

  # Computes the raw scores for the T-Norm model
  for t_model_id in t_model_ids:
    # test if the file is already there
    score_file = fs.c_file(t_model_id, group)
    if utils.check_file(score_file, force):
      logger.warn("Score file '%s' already exists.", score_file)
    else:
      t_model_file = fs.t_model_file(t_model_id, group)
      if allow_missing_files and not os.path.exists(t_model_file):
        t_model = None
      else:
        t_model = algorithm.read_model(t_model_file)
      c = _scores(algorithm, t_model, probe_files, allow_missing_files)
      bob.io.base.save(c, score_file, True)

def _scores_d(algorithm, t_model_ids, group, force, allow_missing_files):
  """Computes D scores for the given t-norm model ids. Both the D matrix and the D-samevalue matrix are written."""
  # the file selector object
  fs = FileSelector.instance()

  # probe files:
  z_probe_objects = fs.z_probe_objects(group)
  z_probe_files = fs.get_paths(z_probe_objects, 'projected' if algorithm.performs_projection else 'extracted')

  logger.info("- Scoring: computing score matrix D for group '%s'", group)

  # Gets the Z-Norm impostor samples
  z_probe_ids = [z_probe_object.client_id for z_probe_object in z_probe_objects]

  # Loads the T-Norm models
  for t_model_id in t_model_ids:
    # test if the file is already there
    score_file = fs.d_file(t_model_id, group)
    same_score_file = fs.d_same_value_file(t_model_id, group)
    if utils.check_file(score_file, force) and utils.check_file(same_score_file, force):
      logger.warn("score files '%s' and '%s' already exist.", score_file, same_score_file)
    else:
      t_model_file = fs.t_model_file(t_model_id, group)
      if allow_missing_files and not os.path.exists(t_model_file):
        t_model = None
      else:
        t_model = algorithm.read_model(t_model_file)
      d = _scores(algorithm, t_model, z_probe_files, allow_missing_files)
      bob.io.base.save(d, score_file, True)

      t_client_id = [fs.client_id(t_model_id, group, True)]
      d_same_value_tm = bob.learn.em.ztnorm_same_value(t_client_id, z_probe_ids)
      bob.io.base.save(d_same_value_tm, same_score_file, True)


def compute_scores(algorithm, compute_zt_norm, indices = None, groups = ['dev', 'eval'], types = ['A', 'B', 'C', 'D'], write_compressed = False, allow_missing_files = False, force = False):
  """Computes the scores for the given groups.

  This function computes all scores for the experiment, and writes them to files, one per model.
  When ``compute_zt_norm`` is enabled, scores are computed for all four matrices, i.e. A: normal scores; B: Z-norm scores; C: T-norm scores; D: ZT-norm scores and ZT-samevalue scores.
  By default, scores are computed for both groups ``'dev'`` and ``'eval'``.

  **Parameters:**

  algorithm : py:class:`bob.bio.base.algorithm.Algorithm` or derived
    The algorithm, used for enrolling model and writing them to file.

  compute_zt_norm : bool
    If set to ``True``, also ZT-norm scores are computed.

  indices : (int, int) or None
    If specified, scores are computed only for the models in the given index range ``range(begin, end)``.
    This is usually given, when parallel threads are executed.

    .. note:: The probe files are not limited by the ``indices``.

  groups : some of ``('dev', 'eval')``
    The list of groups, for which scores should be computed.

  types : some of ``['A', 'B', 'C', 'D']``
    A list of score types to be computed.
    If ``compute_zt_norm = False``, only the ``'A'`` scores are computed.

  write_compressed : bool
    If enabled, score files are compressed as ``.tar.bz2`` files.

  allow_missing_files : bool
    If set to ``True``, model and probe files that are not found will produce ``NaN`` scores.

  force : bool
    If given, score files are regenerated, even if they already exist.
  """
  # the file selector object
  fs = FileSelector.instance()

  # load the projector and the enroller, if needed
  if algorithm.performs_projection:
    algorithm.load_projector(fs.projector_file)
  algorithm.load_enroller(fs.enroller_file)

  for group in groups:
    # get model ids
    model_ids = fs.model_ids(group)
    if indices is not None:
      model_ids = model_ids[indices[0]:indices[1]]
      logger.info("- Scoring: splitting of index range %s", str(indices))
    if compute_zt_norm:
      t_model_ids = fs.t_model_ids(group)
      if indices is not None:
        t_model_ids = t_model_ids[indices[0]:indices[1]]

    # compute A scores
    if 'A' in types:
      _scores_a(algorithm, model_ids, group, compute_zt_norm, force, write_compressed, allow_missing_files)

    if compute_zt_norm:
      # compute B scores
      if 'B' in types:
        _scores_b(algorithm, model_ids, group, force, allow_missing_files)

      # compute C scores
      if 'C' in types:
        _scores_c(algorithm, t_model_ids, group, force, allow_missing_files)

      # compute D scores
      if 'D' in types:
        _scores_d(algorithm, t_model_ids, group, force, allow_missing_files)



def _c_matrix_split_for_model(selected_probe_objects, all_probe_objects, all_c_scores):
  """Helper function to sub-select the c-scores in case not all probe files were used to compute A scores."""
  c_scores_for_model = numpy.empty((all_c_scores.shape[0], len(selected_probe_objects)), numpy.float64)
  selected_index = 0
  for all_index in range(len(all_probe_objects)):
    if selected_index < len(selected_probe_objects) and selected_probe_objects[selected_index].id == all_probe_objects[all_index].id:
      c_scores_for_model[:,selected_index] = all_c_scores[:,all_index]
      selected_index += 1
  assert selected_index == len(selected_probe_objects)

  # return the split database
  return c_scores_for_model

def _scores_c_normalize(model_ids, t_model_ids, group):
  """Compute normalized probe scores using T-model scores."""
  # the file selector object
  fs = FileSelector.instance()

  # read all tmodel scores
  c_for_all = None
  for t_model_id in t_model_ids:
    tmp = bob.io.base.load(fs.c_file(t_model_id, group))
    if c_for_all is None:
      c_for_all = tmp
    else:
      c_for_all = numpy.vstack((c_for_all, tmp))

  # iterate over all models and generate C matrices for that specific model
  all_probe_objects = fs.probe_objects(group)
  for model_id in model_ids:
    # select the correct probe files for the current model
    probe_objects_for_model = fs.probe_objects_for_model(model_id, group)
    c_matrix_for_model = _c_matrix_split_for_model(probe_objects_for_model, all_probe_objects, c_for_all)
    # Save C matrix to file
    bob.io.base.save(c_matrix_for_model, fs.c_file_for_model(model_id, group))

def _scores_d_normalize(t_model_ids, group):
  """Compute normalized D scores for the given T-model ids"""
  # the file selector object
  fs = FileSelector.instance()

  # initialize D and D_same_value matrices
  d_for_all = None
  d_same_value = None
  for t_model_id in t_model_ids:
    tmp = bob.io.base.load(fs.d_file(t_model_id, group))
    tmp2 = bob.io.base.load(fs.d_same_value_file(t_model_id, group))
    if d_for_all is None and d_same_value is None:
      d_for_all = tmp
      d_same_value = tmp2
    else:
      d_for_all = numpy.vstack((d_for_all, tmp))
      d_same_value = numpy.vstack((d_same_value, tmp2))

  # Saves to files
  bob.io.base.save(d_for_all, fs.d_matrix_file(group))
  bob.io.base.save(d_same_value, fs.d_same_value_matrix_file(group))



def zt_norm(groups = ['dev', 'eval'], write_compressed = False, allow_missing_files = False):
  """Computes ZT-Norm using the previously generated A, B, C, D and D-samevalue matrix files.

  This function computes the ZT-norm scores for all model ids for all desired groups and writes them into files defined by the :py:class:`bob.bio.base.tools.FileSelector`.
  It loads the A, B, C, D and D-samevalue matrix files that need to be computed beforehand.

  **Parameters:**

  groups : some of ``('dev', 'eval')``
    The list of groups, for which ZT-norm should be applied.

  write_compressed : bool
    If enabled, score files are compressed as ``.tar.bz2`` files.

  allow_missing_files : bool
    Currently, this option is only provided for completeness.
    ``NaN`` scores are not yet handled correctly.
  """
  # the file selector object
  fs = FileSelector.instance()

  for group in groups:
    logger.info("- Scoring: computing ZT-norm for group '%s'", group)
    # list of models
    model_ids = fs.model_ids(group)
    t_model_ids = fs.t_model_ids(group)

    # first, normalize C and D scores
    _scores_c_normalize(model_ids, t_model_ids, group)
    # and normalize it
    _scores_d_normalize(t_model_ids, group)

    # load D matrices only once
    d = bob.io.base.load(fs.d_matrix_file(group))
    d_same_value = bob.io.base.load(fs.d_same_value_matrix_file(group)).astype(bool)
    error_log_done = False
    # Loops over the model ids
    for model_id in model_ids:
      # Loads probe files to get information about the type of access
      probe_objects = fs.probe_objects_for_model(model_id, group)

      # Loads A, B, and C matrices for current model id
      a = bob.io.base.load(fs.a_file(model_id, group))
      b = bob.io.base.load(fs.b_file(model_id, group))
      c = bob.io.base.load(fs.c_file_for_model(model_id, group))

      # compute zt scores
      if allow_missing_files:
        # TODO: handle NaN scores, i.e., when allow_missing_files is enabled
        if not error_log_done and any(numpy.any(numpy.isnan(x)) for x in (a,b,c,d,d_same_value)):
          logger.error("There are NaN scores inside one of the score files for group %s; ZT-Norm will not work", group)
          error_log_done = True

      zt_scores = bob.learn.em.ztnorm(a, b, c, d, d_same_value)

      # Saves to text file
      _save_scores(fs.zt_norm_file(model_id, group), zt_scores, probe_objects, fs.client_id(model_id, group), write_compressed)



def _concat(score_files, output, write_compressed):
  """Concatenates a list of score files into a single score file."""
  f = _open_to_write(output, write_compressed)

  # Concatenates the scores
  for score_file in score_files:
    i = _open_to_read(score_file)
    f.write(i.read())

  _close_written(output, f, write_compressed)



def concatenate(compute_zt_norm, groups = ['dev', 'eval'], write_compressed = False):
  """Concatenates all results into one (or two) score files per group.

  Score files, which were generated per model, are concatenated into a single score file, which can be interpreter by :py:func:`bob.measure.load.split_four_column`.
  The score files are always re-computed, regardless if they exist or not.

  **Parameters:**

  compute_zt_norm : bool
    If set to ``True``, also score files for ZT-norm are concatenated.

  groups : some of ``('dev', 'eval')``
    The list of groups, for which score files should be concatenated.

  write_compressed : bool
    If enabled, concatenated score files are compressed as ``.tar.bz2`` files.
  """
  # the file selector object
  fs = FileSelector.instance()
  for group in groups:
    logger.info("- Scoring: concatenating score files for group '%s'", group)
    # (sorted) list of models
    model_files = [fs.no_norm_file(model_id, group) for model_id in fs.model_ids(group)]
    result_file = fs.no_norm_result_file(group)
    _concat(model_files, result_file, write_compressed)
    logger.info("- Scoring: wrote score file '%s'", result_file)

    if compute_zt_norm:
      model_files = [fs.zt_norm_file(model_id, group) for model_id in fs.model_ids(group)]
      result_file = fs.zt_norm_result_file(group)
      _concat(model_files, result_file, write_compressed)
      logger.info("- Scoring: wrote score file '%s'", result_file)


def calibrate(compute_zt_norm, groups = ['dev', 'eval'], prior = 0.5, write_compressed = False):
  """Calibrates the score files by learning a linear calibration from the dev files (first element of the groups) and executing the on all groups.

  This function is intended to compute the calibration parameters on the scores of the development set using the :py:class:`bob.learn.linear.CGLogRegTrainer`.
  Afterward, both the scores of the development and evaluation sets are calibrated and written to file.
  For ZT-norm scores, the calibration is performed independently, if enabled.
  The names of the calibrated score files that should be written are obtained from the :py:class:`bob.bio.base.tools.FileSelector`.

  .. note::
     All ``NaN`` scores in the development set are silently ignored.
     This might raise an error, if **all** scores are ``NaN``.

  **Parameters:**

  compute_zt_norm : bool
    If set to ``True``, also score files for ZT-norm are calibrated.

  groups : some of ``('dev', 'eval')``
    The list of groups, for which score files should be calibrated.
    The first of the given groups is used to train the logistic regression parameters, while the calibration is performed for all given groups.

  prior : float
    Whatever :py:class:`bob.learn.linear.CGLogRegTrainer` takes as a ``prior``.

  write_compressed : bool
    If enabled, calibrated score files are compressed as ``.tar.bz2`` files.
  """
  # the file selector object
  fs = FileSelector.instance()
  # read score files of the first group (assuming that the first group is 'dev')
  norms = ['nonorm', 'ztnorm'] if compute_zt_norm else ["nonorm"]
  for norm in norms:
    training_score_file = fs.no_norm_result_file(groups[0]) if norm == 'nonorm' else fs.zt_norm_result_file(groups[0]) if norm == 'ztnorm' else None

    # create a LLR trainer
    logger.info(" - Calibration: Training calibration for type %s from group %s", norm, groups[0])
    llr_trainer = bob.learn.linear.CGLogRegTrainer(prior, 1e-16, 100000)

    training_scores = list(bob.measure.load.split_four_column(training_score_file))
    for i in (0,1):
      h = numpy.array(training_scores[i])
      # remove NaN's
      h = h[~numpy.isnan(h)]
      training_scores[i] = h[:,numpy.newaxis]
    # train the LLR
    llr_machine = llr_trainer.train(training_scores[0], training_scores[1])
    del training_scores
    logger.debug("   ... Resulting calibration parameters: shift = %f, scale = %f", llr_machine.biases[0], llr_machine.weights[0,0])

    # now, apply it to all groups
    for group in groups:
      score_file = fs.no_norm_result_file(group) if norm == 'nonorm' else fs.zt_norm_result_file(group) if norm is 'ztnorm' else None
      calibrated_file = fs.calibrated_score_file(group, norm == 'ztnorm')

      logger.info(" - Calibration: calibrating scores from '%s' to '%s'", score_file, calibrated_file)

      # iterate through the score file and calibrate scores
      scores = bob.measure.load.four_column(_open_to_read(score_file))

      f = _open_to_write(calibrated_file, write_compressed)

      for line in scores:
        assert len(line) == 4, "The line %s of score file %s cannot be interpreted" % (line, score_file)
        calibrated_score = llr_machine([line[3]])
        f.write('%s %s %s %3.8f\n' % (line[0], line[1], line[2], calibrated_score[0]))
      _close_written(calibrated_file, f, write_compressed)
