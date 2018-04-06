#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 23 May 2011 16:23:05 CEST

"""A set of utilities to load score files with different formats.
"""

import numpy
import csv
import tarfile
import os
import sys

import logging
logger = logging.getLogger('bob.bio.base')


def open_file(filename, mode='rt'):
  """Opens the given score file for reading.

  Score files might be raw text files, or a tar-file including a single score
  file inside.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The name of the score file to
      open, or a file-like object open for reading. If a file name is given,
      the according file might be a raw text file or a (compressed) tar file
      containing a raw text file.


  Returns:


    ``file-like``: A read-only file-like object as it would be returned by
    :py:func:`open`.

  """

  if not isinstance(filename, str) and hasattr(filename, 'read'):
    # It seems that this is an open file
    return filename

  if not os.path.isfile(filename):
    raise IOError("Score file '%s' does not exist." % filename)
  if not tarfile.is_tarfile(filename):
    return open(filename, mode)

  # open the tar file for reading
  tar = tarfile.open(filename, 'r')
  # get the first file in the tar file
  tar_info = tar.next()
  while tar_info is not None and not tar_info.isfile():
    tar_info = tar.next()
  # check that one file was found in the archive
  if tar_info is None:
    raise IOError(
        "The given file is a .tar file, but it does not contain any file.")

  # open the file for reading
  return tar.extractfile(tar_info)


def four_column(filename):
  """Loads a score set from a single file and yield its lines

  Loads a score set from a single file and yield its lines (to avoid loading
  the score file at once into memory).  This function verifies that all fields
  are correctly placed and contain valid fields.  The score file must contain
  the following information in each line:

  .. code-block:: text

     claimed_id real_id test_label score


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Yields:

    str: The claimed identity -- the client name of the model that was used in
    the comparison

    str: The real identity -- the client name of the probe that was used in the
    comparison

    str: A label of the probe -- usually the probe file name, or the probe id

    float: The result of the comparison of the model and the probe

  """
  return _iterate_score_file(filename)


def split_four_column(filename):
  """Loads a score set from a single file and splits the scores

  Loads a score set from a single file and splits the scores between negatives
  and positives. The score file has to respect the 4 column format as defined
  in the method :py:func:`four_column`.

  This method avoids loading and allocating memory for the strings present in
  the file. We only keep the scores.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    array: negatives, 1D float array containing the list of scores, for which
      the ``claimed_id`` and the ``real_id`` are different
      (see :py:func:`four_column`)

    array: positives, 1D float array containing the list of scores, for which
      the ``claimed_id`` and the ``real_id`` are identical
      (see :py:func:`four_column`)

  """

  score_lines = four_column(filename)
  return _split_scores(score_lines, 1)


def cmc_four_column(filename):
  """Loads scores to compute CMC curves from a file in four column format.

  The four column file needs to be in the same format as described in
  :py:func:`four_column`, and the ``test_label`` (column 3) has to contain the
  test/probe file name or a probe id.

  This function returns a list of tuples.  For each probe file, the tuple
  consists of a list of negative scores and a list of positive scores.
  Usually, the list of positive scores should contain only one element, but
  more are allowed.  The result of this function can directly be passed to,
  e.g., the :py:func:`bob.measure.cmc` function.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    :any:`list`: A list of tuples, where each tuple contains the
    ``negative`` and ``positive`` scores for one probe of the database. Both
    ``negatives`` and ``positives`` can be either an 1D
    :py:class:`numpy.ndarray` of type ``float``, or ``None``.

  """

  score_lines = four_column(filename)
  return _split_cmc_scores(score_lines, 1)


def five_column(filename):
  """Loads a score set from a single file and yield its lines

  Loads a score set from a single file and yield its lines (to avoid loading
  the score file at once into memory).  This function verifies that all fields
  are correctly placed and contain valid fields.  The score file must contain
  the following information in each line:

  .. code-block:: text

     claimed_id model_label real_id test_label score


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Yields:

    str: The claimed identity -- the client name of the model that was used in
    the comparison

    str: A label for the model -- usually the model file name, or the model id

    str: The real identity -- the client name of the probe that was used in the
    comparison

    str: A label of the probe -- usually the probe file name, or the probe id

    float: The result of the comparison of the model and the probe

  """

  return _iterate_score_file(filename)


def split_five_column(filename):
  """Loads a score set from a single file and splits the scores

  Loads a score set from a single file in five column format and splits the
  scores between negatives and positives. The score file has to respect the 5
  column format as defined in the method :py:func:`five_column`.

  This method avoids loading and allocating memory for the strings present in
  the file. We only keep the scores.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    array: negatives, 1D float array containing the list of scores, for which
      the ``claimed_id`` and the ``real_id`` are different
      (see :py:func:`four_column`)

    array: positives, 1D float array containing the list of scores, for which
      the ``claimed_id`` and the ``real_id`` are identical
      (see :py:func:`four_column`)

  """

  score_lines = four_column(filename)
  return _split_scores(score_lines, 2)


def cmc_five_column(filename):
  """Loads scores to compute CMC curves from a file in five column format.

  The five column file needs to be in the same format as described in
  :py:func:`five_column`, and the ``test_label`` (column 4) has to contain the
  test/probe file name or a probe id.

  This function returns a list of tuples.  For each probe file, the tuple
  consists of a list of negative scores and a list of positive scores.
  Usually, the list of positive scores should contain only one element, but
  more are allowed.  The result of this function can directly be passed to,
  e.g., the :py:func:`bob.measure.cmc` function.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    :any:`list`: A list of tuples, where each tuple contains the
    ``negative`` and ``positive`` scores for one probe of the database.

  """
  score_lines = four_column(filename)
  return _split_cmc_scores(score_lines, 2)


def scores(filename, ncolumns=None):
  """scores(filename, ncolumns=None) -> tuple

  Loads the scores from the given score file and yield its lines.
  Depending on the score file format, four or five elements are yielded, see
  :py:func:`bob.bio.base.score.load.four_column` and
  :py:func:`bob.bio.base.score.load.five_column` for details.

  Parameters:

  filename:  :py:class:`str`, ``file-like``:
    The file object that will be opened with :py:func:`open_file` containing the scores.

  ncolumns: any
    ignored

  Yields:

  tuple:
    see :py:func:`bob.bio.base.score.load.four_column` or
    :py:func:`bob.bio.base.score.load.five_column`
  """
  return _iterate_score_file(filename)


def split(filename, ncolumns=None):
  """split(filename, ncolumns=None) -> negatives, positives

  Loads the scores from the given score file and splits them into positives and negatives.

  Depending on the score file format, it calls see :py:func:`split_four_column` 
  and :py:func:`split_five_column` for details.

  Parameters:

  filename:  :py:class:`str`, ``file-like``:
    The file object that will be opened with :py:func:`open_file` containing the scores.

  ncolumns: int or ``None``
    If specified to be ``4`` or ``5``, the score file will be assumed to be in the given format.
    If not specified, the score file format will be estimated automatically

  Returns:

  negatives: 1D :py:class:`numpy.ndarray` of type float
    This array contains the list of scores, for which the ``claimed_id`` and the ``real_id`` are different (see :py:func:`four_column`)

  positives: 1D :py:class:`numpy.ndarray` of type float
    This array contains the list of scores, for which the ``claimed_id`` and the ``real_id`` are identical (see :py:func:`four_column`)

  """
  ncolumns = _estimate_score_file_format(filename, ncolumns)
  if ncolumns == 4:
    return split_four_column(filename)
  else:
    assert ncolumns == 5
    return split_five_column(filename)


def cmc(filename, ncolumns=None):
  """cmc(filename, ncolumns=None) -> list

  Loads scores to compute CMC curves.

  Depending on the score file format, it calls see
  :py:func:`bob.bio.base.score.load.cmc_four_column` and
  `:py:func:`bob.bio.base.score.load.cmc_five_column` for details.

  Parameters:

    filename (:py:class:`str` or ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.

    ncolumns: (:py:class:`int`, Optional): If specified to be ``4`` or ``5``,
      the score file will be assumed to be in the given format.  If not
      specified, the score file format will be estimated automatically

  Returns:

  :any:`list`: [(neg,pos)] A list of tuples, where each tuple contains the
  ``negative`` and ``positive`` scores for one probe of the database.

  """
  ncolumns = _estimate_score_file_format(filename, ncolumns)
  if ncolumns == 4:
    return cmc_four_column(filename)
  else:
    assert ncolumns == 5
    return cmc_five_column(filename)

def load_score(filename, ncolumns=None, minimal=False, **kwargs):
  """Load scores using numpy.loadtxt and return the data as a numpy array.

  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.

    ncolumns (:py:class:`int`, optional): 4, 5 or None (the default),
      specifying the number of columns in the score file. If None is provided,
      the number of columns will be guessed.

    minimal (:py:class:`bool`, optional): If True, only loads ``claimed_id``, ``real_id``,
      and ``scores``.

    **kwargs: Keyword arguments passed to :py:func:`numpy.genfromtxt`


  Returns:

    array: An array which contains not only the actual scores but also the
    ``claimed_id``, ``real_id``, ``test_label`` and ``['model_label']``

  """

  def convertfunc(x):
    return x

  ncolumns = _estimate_score_file_format(filename, ncolumns)

  usecols = kwargs.pop('usecols', None)
  if ncolumns == 4:
    names = ('claimed_id', 'real_id', 'test_label', 'score')
    converters = {
        0: convertfunc,
        1: convertfunc,
        2: convertfunc,
        3: float}
    if minimal:
      usecols = (0, 1, 3)

  elif ncolumns == 5:
    names = ('claimed_id', 'model_label', 'real_id', 'test_label', 'score')
    converters = {
        0: convertfunc,
        1: convertfunc,
        2: convertfunc,
        3: convertfunc,
        4: float}
    if minimal:
      usecols = (0, 2, 4)
  else:
    raise ValueError("ncolumns of 4 and 5 are supported only.")

  score_lines = numpy.genfromtxt(
      open_file(filename, mode='rb'), dtype=None, names=names,
      converters=converters, invalid_raise=True, usecols=usecols, **kwargs)
  new_dtype = []
  for name in score_lines.dtype.names[:-1]:
    new_dtype.append((name, str(score_lines.dtype[name]).replace('S', 'U')))
  new_dtype.append(('score', float))
  score_lines = numpy.array(score_lines, new_dtype)
  return score_lines

def load_files(filenames, func_load):
    """Load a list of score files and return a list of tuples of (neg, pos)

    Parameters
    ----------

    filenames : :any:`list`
        list of file paths
    func_load :
        function that can read files in the list

    Returns
    -------

    :any:`list`: [(neg,pos)] A list of tuples, where each tuple contains the
    ``negative`` and ``positive`` sceach system/probee.

    """
    if filenames is None:
        return None
    res = []
    for filepath in filenames:
        try:
            tmp = func_load(filepath)
            if isinstance(tmp, list):
                res += func_load(filepath)
            else:
                res.append(tmp)
        except:
            raise
    return res

def get_negatives_positives(score_lines):
  """Take the output of load_score and return negatives and positives.  This
  function aims to replace split_four_column and split_five_column but takes a
  different input. It's up to you to use which one.
  """

  pos_mask = score_lines['claimed_id'] == score_lines['real_id']
  positives = score_lines['score'][pos_mask]
  negatives = score_lines['score'][numpy.logical_not(pos_mask)]
  return (negatives, positives)


def get_negatives_positives_from_file(filename, **kwargs):
  """Loads the scores first efficiently and then calls
  get_negatives_positives"""
  score_lines = load_score(filename, minimal=True, **kwargs)
  return get_negatives_positives(score_lines)


def get_negatives_positives_all(score_lines_list):
  """Take a list of outputs of load_score and return stacked negatives and
  positives.
  """

  negatives, positives = [], []
  for score_lines in score_lines_list:
    neg_pos = get_negatives_positives(score_lines)
    negatives.append(neg_pos[0])
    positives.append(neg_pos[1])
  negatives = numpy.vstack(negatives).T
  positives = numpy.vstack(positives).T
  return (negatives, positives)


def get_all_scores(score_lines_list):
  """Take a list of outputs of load_score and return stacked scores"""

  return numpy.vstack([score_lines['score']
                       for score_lines in score_lines_list]).T


def dump_score(filename, score_lines):
  """Dump scores that were loaded using :py:func:`load_score`
  The number of columns is automatically detected.
  """

  if len(score_lines.dtype) == 5:
    fmt = '%s %s %s %s %.9f'
  elif len(score_lines.dtype) == 4:
    fmt = '%s %s %s %.9f'
  else:
    raise ValueError("Only scores with 4 and 5 columns are supported.")
  numpy.savetxt(filename, score_lines, fmt=fmt)


def _estimate_score_file_format(filename, ncolumns=None):
  """Estimates the score file format from the given score file.
  If ``ncolumns`` is in ``(4,5)``, then ``ncolumns`` is returned instead.
  """
  if ncolumns in (4, 5):
    return ncolumns

  f = open_file(filename, 'rb')
  try:
    line = f.readline()
    ncolumns = len(line.split())
  except Exception:
    logger.warn('Could not guess the number of columns in file: {}. '
                'Assuming 4 column format.'.format(filename))
    ncolumns = 4
  finally:
    f.close()
  return ncolumns


def _iterate_score_file(filename):
  """Opens the score file for reading and yields the score file line by line in a tuple/list.

  The last element of the line (which is the score) will be transformed to float, the other elements will be str
  """
  opened = open_file(filename, 'rb')
  if sys.version_info.major > 2:
    import io
    if not isinstance(opened, io.TextIOWrapper):
      opened = io.TextIOWrapper(opened, newline="")

  reader = csv.reader(opened, delimiter=' ')
  for splits in reader:
    splits[-1] = float(splits[-1])
    yield splits


def _split_scores(score_lines, real_id_index, claimed_id_index=0, score_index=-1):
  """Take the output of :py:func:`four_column` or :py:func:`five_column` and return negatives and positives.
  """
  positives, negatives = [], []
  for line in score_lines:
    which = positives if line[claimed_id_index] == line[
        real_id_index] else negatives
    which.append(line[score_index])

  return (numpy.array(negatives), numpy.array(positives))


def _split_cmc_scores(score_lines, real_id_index, probe_name_index=None, claimed_id_index=0, score_index=-1):
  """Takes the output of :py:func:`four_column` or :py:func:`five_column` and return cmc scores.
  """
  if probe_name_index is None:
    probe_name_index = real_id_index + 1
  # extract positives and negatives
  pos_dict = {}
  neg_dict = {}
  # read four column list
  for line in score_lines:
    which = pos_dict if line[claimed_id_index] == line[
        real_id_index] else neg_dict
    probe_name = line[probe_name_index]
    # append score
    if probe_name not in which:
      which[probe_name] = []
    which[probe_name].append(line[score_index])

  # convert to lists of tuples of ndarrays (or None)
  probe_names = sorted(set(neg_dict.keys()).union(set(pos_dict.keys())))
  # get all scores in the desired format
  return [(
      numpy.array(neg_dict[probe_name],
                  numpy.float64) if probe_name in neg_dict else None,
      numpy.array(pos_dict[probe_name],
                  numpy.float64) if probe_name in pos_dict else None
  ) for probe_name in probe_names]
