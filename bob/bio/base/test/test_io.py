#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 11 Dec 15:14:08 2013 CET
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the IO functionality of bob.bio.base.score."""

import numpy
import tempfile
import os
import shutil
import pkg_resources

from .. import score


def test_load_scores():
  # This function tests the IO functionality of loading score files in
  # different ways

  load_functions = {'4col': score.four_column,
                    '5col': score.five_column}
  cols = {'4col': 4, '5col': 5}

  for variant in cols:
    normal_score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/dev-%s.txt' % variant)
    normal_scores = list(load_functions[variant](normal_score_file))

    assert len(normal_scores) == 910
    assert all(len(s) == cols[variant] for s in normal_scores)

    # read the compressed score file
    compressed_score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/dev-%s.tar.gz' % variant)
    compressed_scores = list(load_functions[variant](compressed_score_file))

    assert len(compressed_scores) == len(normal_scores)
    assert all(len(c) == cols[variant] for c in compressed_scores)
    assert all(c[i] == s[i] for c, s in zip(compressed_scores,
                                            normal_scores) for i in range(cols[variant]))

    # Use auto-estimated score file contents
    # read score file in normal way
    normal_scores = list(score.scores(normal_score_file))

    assert len(normal_scores) == 910
    assert all(len(s) == cols[variant] for s in normal_scores)

    # read the compressed score file
    compressed_scores = list(score.scores(compressed_score_file))

    assert len(compressed_scores) == len(normal_scores)
    assert all(len(c) == cols[variant] for c in compressed_scores)
    assert all(c[i] == s[i] for c, s in zip(compressed_scores,
                                            normal_scores) for i in range(cols[variant]))


def test_split_scores():
  # This function tests the IO functionality of loading score files in
  # different ways

  split_functions = {'4col': score.split_four_column,
                     '5col': score.split_five_column}
  cols = {'4col': 4, '5col': 5}

  for variant in cols:
    # read score file in normal way
    normal_score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/dev-%s.txt' % variant)
    negatives, positives = split_functions[variant](normal_score_file)

    assert len(negatives) == 520, len(negatives)
    assert len(positives) == 390, len(positives)

    # read the compressed score file
    compressed_score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/dev-%s.tar.gz' % variant)
    negatives, positives = split_functions[variant](compressed_score_file)

    assert len(negatives) == 520, len(negatives)
    assert len(positives) == 390, len(positives)

    # Use auto-estimated score file contents
    # read score file in normal way
    negatives, positives = score.split(normal_score_file)

    assert len(negatives) == 520, len(negatives)
    assert len(positives) == 390, len(positives)

    # read the compressed score file
    negatives, positives = score.split(compressed_score_file)

    assert len(negatives) == 520, len(negatives)
    assert len(positives) == 390, len(positives)


def test_load_score():
  # This function tests the IO functionality of loading score files in
  # different ways

  scores = []
  cols = {'4col': 4, '5col': 5}

  for variant in cols:
    compressed_score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/dev-%s.tar.gz' % variant)
    normal_score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/dev-%s.txt' % variant)
    normal_scores = score.load_score(
        normal_score_file, cols[variant])

    assert len(normal_scores) == 910
    assert len(normal_scores.dtype) == cols[variant]

    # read the compressed score file
    compressed_score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/dev-%s.tar.gz' % variant)
    compressed_scores = score.load_score(
        compressed_score_file, cols[variant])

    assert len(compressed_scores) == len(normal_scores)
    assert len(compressed_scores.dtype) == cols[variant]
    for name in normal_scores.dtype.names:
      assert all(normal_scores[name] == compressed_scores[name])

    # test minimal loading
    minimal_scores = score.load_score(
        normal_score_file, minimal=True)
    assert len(minimal_scores) == 910
    assert len(minimal_scores.dtype) == 3
    assert minimal_scores.dtype.names == ('claimed_id', 'real_id', 'score')


def test_dump_score():
  # This function tests the IO functionality of dumping score files

  scores = []
  cols = {'4col': 4, '5col': 5}

  for variant in cols:
    # read score file
    normal_score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/dev-%s.txt' % variant)
    normal_scores = score.load_score(
        normal_score_file, cols[variant])

    with tempfile.TemporaryFile() as f:
      score.dump_score(f, normal_scores)
      f.seek(0)
      loaded_scores = score.load_score(f, cols[variant])

    for name in normal_scores.dtype.names:
      assert all(normal_scores[name] == loaded_scores[name])


def _check_binary_identical(name1, name2):
  # see: http://www.peterbe.com/plog/using-md5-to-check-equality-between-files
  from hashlib import md5
  # tests if two files are binary identical
  with open(name1, 'rb') as f1:
    with open(name2, 'rb') as f2:
      assert md5(f1.read()).digest() == md5(f2.read()).digest()


def test_openbr_verify():
  # This function tests that the conversion to the OpenBR verify file works
  # as expected
  temp_dir = tempfile.mkdtemp(prefix='bob_test')

  # define output files
  openbr_extensions = ('.mtx', '.mask')
  matrix_file, mask_file = [os.path.join(
      temp_dir, "scores%s") % ext for ext in openbr_extensions]

  try:
    for variant in ('4col', '5col'):
      # get score file
      score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/scores-cmc-%s.txt' % variant)

      # first round, do not define keyword arguments -- let the file get the
      # gallery and probe ids automatically
      kwargs = {}
      for i in range(2):
        # get the files by automatically obtaining the identities
        score.write_matrix(
            score_file, matrix_file, mask_file, score_file_format="%sumn" % variant, **kwargs)

        assert os.path.isfile(matrix_file) and os.path.isfile(mask_file)

        # check that they are binary identical to the reference files (which
        # are tested to work and give the same results with OpenBR)
        matrix_ref, mask_ref = [pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/scores%s' % ext) for ext in openbr_extensions]
        _check_binary_identical(matrix_file, matrix_ref)
        _check_binary_identical(mask_file, mask_ref)

        # define new kwargs for second round, i.e., define model and probe names
        # these names are identical to what is found in the score file, which
        # in turn comes from the AT&T database
        model_type = {"4col": "%d", "5col": "s%d"}[variant]
        dev_ids = (3, 4, 7, 8, 9, 13, 15, 18, 19, 22, 23,
                   25, 28, 30, 31, 32, 35, 37, 38, 40)
        kwargs['model_names'] = [model_type % c for c in dev_ids]
        kwargs['probe_names'] = ["s%d/%d" %
                                 (c, i) for c in dev_ids for i in (1, 3, 6, 8, 10)]

  finally:
    shutil.rmtree(temp_dir)


def test_openbr_search():
  # This function tests that the conversion to the OpenBR search file works
  # as expected
  temp_dir = tempfile.mkdtemp(prefix='bob_test')

  # define output files
  openbr_extensions = ('.mtx', '.mask')
  matrix_file, mask_file = [os.path.join(
      temp_dir, "search%s") % ext for ext in openbr_extensions]

  try:
    for variant in ('4col', '5col'):
      # get score file
      score_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/scores-cmc-%s.txt' % variant)

      # first round, do not define keyword arguments -- let the file get the
      # gallery and probe ids automatically
      kwargs = {}
      for i in range(2):
        # get the files by automatically obtaining the identities
        score.write_matrix(
            score_file, matrix_file, mask_file, score_file_format="%sumn" % variant, search=50, **kwargs)

        assert os.path.isfile(matrix_file) and os.path.isfile(mask_file)

        # check that they are binary identical to the reference files (which
        # are tested to work and give the same results with OpenBR)
        matrix_ref, mask_ref = [pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/search%s' % ext) for ext in openbr_extensions]
        _check_binary_identical(matrix_file, matrix_ref)
        _check_binary_identical(mask_file, mask_ref)

        # define new kwargs for second round, i.e., define model and probe names
        # these names are identical to what is found in the score file, which
        # in turn comes from the AT&T database
        model_type = {"4col": "%d", "5col": "s%d"}[variant]
        dev_ids = (3, 4, 7, 8, 9, 13, 15, 18, 19, 22, 23,
                   25, 28, 30, 31, 32, 35, 37, 38, 40)
        kwargs['model_names'] = [model_type % c for c in dev_ids]
        kwargs['probe_names'] = ["s%d/%d" %
                                 (c, i) for c in dev_ids for i in (1, 3, 6, 8, 10)]

  finally:
    shutil.rmtree(temp_dir)


def test_from_openbr():
  # This function tests that the conversion from the OpenBR matrices work as
  # expected
  temp_dir = tempfile.mkdtemp(prefix='bob_test')

  # define input files
  openbr_extensions = ('.mtx', '.mask')
  matrix_file, mask_file = [pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/scores%s' % ext) for ext in openbr_extensions]

  score_file = os.path.join(temp_dir, "scores")
  load_functions = {'4col': score.four_column,
                    '5col': score.five_column}

  try:
    for variant in load_functions:
      # first, do not define keyword arguments -- let the file get the model
      # and probe ids being created automatically
      score.write_score_file(
          matrix_file, mask_file, score_file, score_file_format="%sumn" % variant)
      assert os.path.exists(score_file)
      # read the score file with bobs functionality
      columns = list(load_functions[variant](score_file))

      # check the contents
      assert len(columns) == 2000

      # now, generate model and probe names and ids
      model_type = {"4col": "%d", "5col": "s%d"}[variant]
      dev_ids = (3, 4, 7, 8, 9, 13, 15, 18, 19, 22, 23,
                 25, 28, 30, 31, 32, 35, 37, 38, 40)
      model_names = ["s%d" % c for c in dev_ids]
      probe_names = ["s%d/%d" % (c, i)
                     for c in dev_ids for i in (1, 3, 6, 8, 10)]
      models_ids = ["%d" % c for c in dev_ids]
      probes_ids = ["%d" % c for c in dev_ids for i in (1, 3, 6, 8, 10)]

      score.write_score_file(matrix_file, mask_file, score_file, models_ids=models_ids, probes_ids=probes_ids,
                                          model_names=model_names, probe_names=probe_names, score_file_format="%sumn" % variant)

      # check that we re-generated the bob score file
      reference_file = pkg_resources.resource_filename(
        'bob.bio.base.test', 'data/scores-cmc-%s.txt' % variant)

      # assert that we can (almost) reproduce the score file
      # ... read both files
      columns = list(load_functions[variant](score_file))
      reference = list(load_functions[variant](reference_file))
      assert len(columns) == len(reference)
      for i in range(len(columns)):
        for j in range(len(columns[i]) - 1):
          # check that the model and probe names are fine
          assert columns[i][j] == reference[i][j], str(
              columns[i]) + " != " + str(reference[i])
        # check that the score is close (OpenBR write scores in float32
        # precision only)
        assert abs(columns[i][-1] - numpy.float32(reference[i][-1])
                   ) <= 1e-8, str(columns[i][-1]) + " != " + str(reference[i][-1])

        #assert numpy.isclose(columns[i][-1], reference[i][-1], atol = 1e-3, rtol=1e-8), str(columns[i][-1]) + " != " + str(reference[i][-1])
        assert numpy.allclose(columns[i][-1], reference[i][-1], atol=1e-3,
                              rtol=1e-8), str(columns[i][-1]) + " != " + str(reference[i][-1])

  finally:
    shutil.rmtree(temp_dir)
