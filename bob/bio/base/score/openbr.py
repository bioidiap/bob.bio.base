#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""This file includes functionality to convert between Bob's four column or
   five column score files and the Matrix files used in OpenBR."""


import numpy
import sys
import logging
logger = logging.getLogger("bob.measure")

from .load import open_file, four_column, five_column


def write_matrix(
        score_file,
        matrix_file,
        mask_file,
        model_names=None,
        probe_names=None,
        score_file_format='4column',
        gallery_file_name='unknown-gallery.lst',
        probe_file_name='unknown-probe.lst',
        search=None):
  """Writes the OpenBR matrix and mask files (version 2), given a score file.

  If gallery and probe names are provided, the matrices in both files will be
  sorted by gallery and probe names.  Otherwise, the order will be the same as
  given in the score file.

  If ``search`` is given (as an integer), the resulting matrix files will be in
  the *search* format, keeping the given number of gallery scores with the
  highest values for each probe.

  .. warning::

     When provided with a 4-column score file, this function will work only, if
     there is only a single model id for each client.

  Parameters:

    score_file (str): The 4 or 5 column style score file written by bob.

    matrix_file (str): The OpenBR matrix file that should be written.
      Usually, the file name extension is ``.mtx``

    mask_file (str): The OpenBR mask file that should be written.
      The mask file defines, which values are positives, negatives or to be
      ignored.  Usually, the file name extension is ``.mask``

    model_names (:py:class:`str`, optional): If given, the matrix will be
      written in the same order as the given model names.  The model names must
      be identical with the second column in the 5-column ``score_file``.

      .. note::

         If the score file is in four column format, the model_names must be
         the client ids stored in the first column.  In this case, there might
         be only a single model per client

      Only the scores of the given models will be considered.

    probe_names (:py:class:`list`, optional): A list of strings. If given,
      the matrix will be written in the same order as the given probe names
      (the ``path`` of the probe).  The probe names are identical to the third
      column of the 4-column (or the fourth column of the 5-column)
      ``score_file``.  Only the scores of the given probe names will be
      considered in this case.

    score_file_format (:py:class:`str`, optional): One of ``('4column',
      '5column')``. The format, in which the ``score_file`` is; defaults to
      ``'4column'``

    gallery_file_name (:py:class:`str`, optional): The name of the gallery file
      that will be written in the header of the OpenBR files.

    probe_file_name (:py:class:`str`, optional): The name of the probe file that
      will be written in the header of the OpenBR files.

    search (:py:class:`int`, optional): If given, the scores will be sorted per
      probe, keeping the specified number of highest scores.  If the given
      number is higher than the models, ``NaN`` values will be added, and the
      mask will contain ``0x00`` values.

  """

  def _write_matrix(filename, matrix):
    # Helper function to write a matrix file as required by OpenBR
    with open(filename, 'wb') as f:
      # write the first four lines
      header = "S2\n%s\n%s\nM%s %d %d " % (
          gallery_file_name, probe_file_name, 'B' if matrix.dtype == numpy.uint8 else 'F', matrix.shape[0], matrix.shape[1])
      footer = "\n"
      if sys.version_info[0] > 2:
        header, footer = header.encode('utf-8'), footer.encode('utf-8')
      f.write(header)
      # write magic number
      numpy.array(0x12345678, numpy.int32).tofile(f)
      f.write(footer)
      # write the matrix
      matrix.tofile(f)

  # define read functions, and which information should be read
  read_function = {'4column': four_column,
                   '5column': five_column}[score_file_format]
  offset = {'4column': 0, '5column': 1}[score_file_format]

  # first, read the score file and estimate model and probe names, if not given
  if model_names is None or probe_names is None:
    model_names, probe_names = [], []
    model_set, probe_set = set(), set()

    # read the score file
    for line in read_function(score_file):
      model, probe = line[offset], line[2 + offset]
      if model not in model_set:
        model_names.append(model)
        model_set.add(model)
      if probe not in probe_set:
        probe_names.append(probe)
        probe_set.add(probe)

  if search is None:
    # create a shortcut to get indices for client and probe subset (to
    # increase speed)
    model_dict, probe_dict = {}, {}
    for i, m in enumerate(model_names):
      model_dict[m] = i
    for i, p in enumerate(probe_names):
      probe_dict[p] = i

    # create the matrices in the desired size
    matrix = numpy.ndarray((len(probe_names), len(model_names)), numpy.float32)
    matrix[:] = numpy.nan
    mask = numpy.zeros(matrix.shape, numpy.uint8)

    # now, iterate through the score file and fill in the matrix
    for line in read_function(score_file):
      client, model, id, probe, score = line[0], line[offset], line[
          1 + offset], line[2 + offset], line[3 + offset]

      assert model in model_dict, "model " + model + " unknown"
      assert probe in probe_dict, "probe " + probe + " unknown"

      model_index = model_dict[model]
      probe_index = probe_dict[probe]

      # check, if we have already written something into that matrix element
      if mask[probe_index, model_index]:
        logger.warn("Overwriting existing matrix '%f' element of client '%s' and probe '%s' with '%f'", matrix[
                    probe_index, model_index], client, probe, score)

      matrix[probe_index, model_index] = score
      mask[probe_index, model_index] = 0xff if client == id else 0x7f

  else:
    # get the correct search parameter, if negative
    if search < 0:
      search = len(model_names)

    # create the matrices in the desired size
    matrix = numpy.ndarray((len(probe_names), search), numpy.float32)
    matrix[:] = numpy.nan
    mask = numpy.zeros(matrix.shape, numpy.uint8)

    # get the scores, sorted by probe
    scores = {}
    for line in read_function(score_file):
      client, model, id, probe, score = line[0], line[offset], line[
          1 + offset], line[2 + offset], line[3 + offset]

      if probe not in scores:
        scores[probe] = []
      scores[probe].append((score, 0xff if client == id else 0x7f))

    # go ahead and sort the scores per probe
    sorted_scores = {}
    for k, v in scores.items():
      sorted_scores[k] = sorted(v, key=lambda x: x[0], reverse=True)

    # now, write matrix
    for p, probe in enumerate(probe_names):
      if probe in scores:
        for m in range(min(search, len(sorted_scores[probe]))):
          matrix[p, m], mask[p, m] = sorted_scores[probe][m]

  # OK, now finally write the file in the desired format
  _write_matrix(mask_file, mask)
  _write_matrix(matrix_file, matrix)


def write_score_file(
    matrix_file,
    mask_file,
    score_file,
    models_ids=None,
    probes_ids=None,
    model_names=None,
    probe_names=None,
    score_file_format='4column',
    replace_nan=None
):
  """Writes the Bob score file in the desired format from OpenBR files.

  Writes a Bob score file in the desired format (four or five column), given
  the OpenBR matrix and mask files.

  In principle, the score file can be written based on the matrix and mask
  files, and the format suffice the requirements to compute CMC curves.
  However, the contents of the score files can be adapted.  If given, the
  ``models_ids`` and ``probes_ids`` define the **client ids** of model and
  probe, and they have to be in the same order as used to compute the OpenBR
  matrix.  The ``model_names`` and ``probe_names`` define the **paths** of
  model and probe, and they should be in the same order as the ids.

  In rare cases, the OpenBR matrix contains NaN values, which Bob's score files
  cannot handle.  You can use the ``replace_nan`` parameter to decide, what to
  do with these values.  By default (``None``), these values are ignored, i.e.,
  not written into the score file.  This is, what OpenBR is doing as well.
  However, you can also set ``replace_nan`` to any value, which will be written
  instead of the NaN values.


  Parameters:

    matrix_file (str): The OpenBR matrix file that should be read. Usually, the
      file name extension is ``.mtx``

    mask_file (str): The OpenBR mask file that should be read. Usually, the
      file name extension is ``.mask``

    score_file (str): Path to the 4 or 5 column style score file that should be
      written.

    models_ids (:py:class:`list`, optional): A list of strings with the client
      ids of the models that will be written in the first column of the score
      file.  If given, the size must be identical to the number of models
      (gallery templates) in the OpenBR files.  If not given, client ids of the
      model will be identical to the **gallery index** in the matrix file.

    probes_ids (:py:class:`list`, optional): A list of strings with the client
      ids of the probes that will be written in the second/third column of the
      four/five column score file.  If given, the size must be identical to the
      number of probe templates in the OpenBR files.  It will be checked that
      the OpenBR mask fits to the model/probe client ids.  If not given, the
      probe ids will be estimated automatically, i.e., to fit the OpenBR
      matrix.

    model_names (:py:class:`list`, optional): A list of strings with the model
      path written in the second column of the five column score file. If not
      given, the model index in the OpenBR file will be used.

      .. note::

         This entry is ignored in the four column score file format.

    probe_names (:py:class:`list`, optional): A list of probe path to be
      written in the third/fourth column in the four/five column score file. If
      given, the size must be identical to the number of probe templates in the
      OpenBR files. If not given, the probe index in the OpenBR file will be
      used.

    score_file_format (:py:class:`str`, optional): One of ``('4column',
      '5column')``. The format, in which the ``score_file`` is; defaults to
      ``'4column'``

    replace_nan (:py:class:`float`, optional): If NaN values are encountered in
      the OpenBR matrix (which are not ignored due to the mask being non-NULL),
      this value will be written instead. If ``None``, the values will not be
      written in the score file at all.

  """

  def _read_matrix(filename):
    py3 = sys.version_info[0] >= 3
    # Helper function to read a matrix file as written by OpenBR
    with open(filename, 'rb') as f:
      # get version
      header = f.readline()
      if py3:
        header = header.decode("utf-8")
      assert header[:2] == "S2"
      # skip gallery and probe files
      f.readline()
      f.readline()
      # read size and type of matrix
      size = f.readline()
      if py3:
        size = size.decode("utf-8")
      splits = size.rstrip().split()
      # TODO: check the endianess of the magic number stored in split[3]
      assert splits[0][0] == 'M'
      w, h = int(splits[1]), int(splits[2])
      # read matrix data
      data = numpy.fromfile(
          f, dtype={'B': numpy.uint8, 'F': numpy.float32}[splits[0][1]])
      assert data.shape[0] == w * h
      data.shape = (w, h)
    return data

  # check parameters
  if score_file_format not in ("4column", "5column"):
    raise ValueError(
        "The given score file format %s is not known; choose one of ('4column', '5column')" % score_file_format)
  # get type of score file
  four_col = score_file_format == "4column"

  # read the two matrices
  scores = _read_matrix(matrix_file)
  mask = _read_matrix(mask_file)

  # generate the id lists, if not given
  if models_ids is None:
    models_ids = [str(g + 1) for g in range(mask.shape[1])]
  assert len(models_ids) == mask.shape[1]

  if probes_ids is None:
    probes_ids = []
    # iterate over all probes
    for p in range(mask.shape[0]):
      # get indices, where model and probe id should be identical
      equal_indices = numpy.where(mask[p] == 0xff)
      if len(equal_indices):
        # model id found, use the first one
        probes_ids.append(models_ids[equal_indices[0][0]])
      else:
        # no model found; add non-existing id
        probes_ids.append("unknown")
  else:
    assert len(probes_ids) == mask.shape[0]
    # check that the probes client ids are in the correct order
    for p in range(mask.shape[0]):
      for g in range(mask.shape[1]):
        if mask[p, g] == 0x7f:
          if models_ids[g] == probes_ids[p]:
            raise ValueError("The probe id %s with index %d should not be identical to model id %s with index %d" % (
                probes_ids[p], p, models_ids[g], g))
        elif mask[p, g] == 0xff:
          if models_ids[g] != probes_ids[p]:
            raise ValueError("The probe id %s with index %d should be identical to model id %s with index %d" % (
                probes_ids[p], p, models_ids[g], g))

  # generate model and probe names, if not given
  if not four_col and model_names is None:
    model_names = [str(g + 1) for g in range(mask.shape[1])]
  if probe_names is None:
    probe_names = [str(p + 1) for p in range(mask.shape[0])]

  # iterate through the files and write scores
  with open(score_file, 'w') as f:
    for g in range(mask.shape[1]):
      for p in range(mask.shape[0]):
        if mask[p, g]:
          score = scores[p, g]
          # handle NaN values
          if numpy.isnan(score):
            if replace_nan is None:
              continue
            score = replace_nan
          # write score file
          if four_col:
            f.write("%s %s %s %3.8f\n" %
                    (models_ids[g], probes_ids[p], probe_names[p], score))
          else:
            f.write("%s %s %s %s %3.8f\n" % (models_ids[g], model_names[
                    g], probes_ids[p], probe_names[p], score))
