

from __future__ import print_function

import bob.measure

import os
import sys
import shutil
import tempfile
import numpy

import bob.io.base.test_utils
import bob.io.image
import bob.bio.base
from . import utils

from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_reference = False



dummy_dir = pkg_resources.resource_filename('bob.bio.base', 'test/dummy')
data_dir = pkg_resources.resource_filename('bob.bio.base', 'test/data')

def _verify(parameters, test_dir, sub_dir, ref_modifier="", score_modifier=('scores','')):
  from bob.bio.base.script.verify import main
  try:
    main([sys.argv[0]] + parameters)

    # assert that the score file exists
    score_files = [os.path.join(test_dir, sub_dir, 'Default', norm, '%s-dev%s'%score_modifier) for norm in ('nonorm',  'ztnorm')]
    assert os.path.exists(score_files[0]), "Score file %s does not exist" % score_files[0]
    assert os.path.exists(score_files[1]), "Score file %s does not exist" % score_files[1]

    # also assert that the scores are still the same -- though they have no real meaning
    reference_files = [os.path.join(data_dir, 'scores-%s%s-dev'%(norm, ref_modifier)) for norm in ('nonorm',  'ztnorm')]

    if regenerate_reference:
      for i in (0,1):
        shutil.copy(score_files[i], reference_files[i])

    for i in (0,1):
      d = []
      # read reference and new data
      for score_file in (score_files[i], reference_files[i]):
        f = bob.measure.load.open_file(score_file)
        d_ = []
        for line in f:
          if isinstance(line, bytes): line = line.decode('utf-8')
          d_.append(line.rstrip().split())
        d.append(numpy.array(d_))

      assert d[0].shape == d[1].shape
      # assert that the data order is still correct
      assert (d[0][:,0:3] == d[1][:, 0:3]).all()
      # assert that the values are OK
      assert numpy.allclose(d[0][:,3].astype(float), d[1][:,3].astype(float), 1e-5)

  finally:
    shutil.rmtree(test_dir)


def test_verify_local():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', os.path.join(dummy_dir, 'database.py'),
      '-p', os.path.join(dummy_dir, 'preprocessor.py'),
      '-e', os.path.join(dummy_dir, 'extractor.py'),
      '-a', os.path.join(dummy_dir, 'algorithm.py'),
      '--zt-norm',
      '-s', 'test_local',
      '--temp-directory', test_dir,
      '--result-directory', test_dir
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_local')


def test_verify_resources():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy',
      '-a', 'dummy',
      '--zt-norm',
      '-s', 'test_resource',
      '--temp-directory', test_dir,
      '--result-directory', test_dir
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_resource')


def test_verify_commandline():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'bob.bio.base.test.dummy.database.DummyDatabase()',
      '-p', 'bob.bio.base.test.dummy.preprocessor.DummyPreprocessor()',
      '-e', 'bob.bio.base.test.dummy.extractor.DummyExtractor()',
      '-a', 'bob.bio.base.test.dummy.algorithm.DummyAlgorithm()',
      '--zt-norm',
      '-s', 'test_commandline',
      '--temp-directory', test_dir,
      '--result-directory', test_dir
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_commandline')


@utils.grid_available
def test_verify_parallel():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  test_database = os.path.join(test_dir, "submitted.sql3")

  # define dummy parameters
  parameters = [
      '-d', os.path.join(dummy_dir, 'database.py'),
      '-p', 'dummy',
      '-e', 'bob.bio.base.test.dummy.extractor.DummyExtractor()',
      '-a', 'dummy',
      '--zt-norm',
      '-s', 'test_parallel',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '-g', 'bob.bio.base.grid.Grid(grid = "local", number_of_parallel_processes = 2, scheduler_sleep_time = 0.1)', '-G', test_database, '--run-local-scheduler', '-R',
      '--import', 'bob.io.image'
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_parallel')


def test_verify_compressed():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy',
      '-a', 'dummy',
      '--zt-norm',
      '-s', 'test_compressed',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--write-compressed-score-files'
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_compressed', score_modifier=('scores', '.tar.bz2'))


def test_verify_calibrate():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy',
      '-a', 'dummy',
      '--zt-norm',
      '-s', 'test_calibrate',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--calibrate-scores'
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_calibrate', '-calibrated', score_modifier=('calibrated', ''))


def test_verify_fileset():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', os.path.join(dummy_dir, 'database.py'),
      '-p', 'dummy',
      '-e', 'bob.bio.base.test.dummy.extractor.DummyExtractor()',
      '-a', 'dummy',
      '--zt-norm',
      '-s', 'test_fileset',
      '--temp-directory', test_dir,
      '--result-directory', test_dir
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_fileset', ref_modifier="-fileset")



def test_verify_filelist():
  try:
    import bob.db.verification.filelist
  except ImportError:
    raise SkipTest("Skipping test since bob.db.verification.filelist is not available")
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', os.path.join(dummy_dir, 'filelist.py'),
      '-p', 'dummy',
      '-e', 'dummy',
      '-a', 'dummy',
      '--zt-norm',
      '-s', 'test_filelist',
      '--temp-directory', test_dir,
      '--result-directory', test_dir
  ]

  print (bob.bio.base.tools.command_line(parameters))

  try:
    from bob.bio.base.script.verify import main
    main([sys.argv[0]] + parameters)

    # assert that the score file exists
    score_files = [os.path.join(test_dir, 'test_filelist', 'None', norm, 'scores-dev') for norm in ('nonorm', 'ztnorm')]
    assert os.path.exists(score_files[0]), "Score file %s does not exist" % score_files[0]
    assert os.path.exists(score_files[1]), "Score file %s does not exist" % score_files[1]

    # assert that the scores are are identical (might be in a different order, though
    reference_files = [os.path.join(data_dir, 'scores-%s-dev' % norm) for norm in ('nonorm', 'ztnorm')]

    for i in (0,1):
      # load scores
      a1, b1 = bob.measure.load.split_four_column(score_files[i])
      a2, b2 = bob.measure.load.split_four_column(reference_files[i])
      # sort scores
      a1 = sorted(a1); a2 = sorted(a2); b1 = sorted(b1); b2 = sorted(b2)

      # assert that scores are almost equal
      for i in range(len(a1)):
        abs(a1[i] - a2[i]) < 1e-6
      for i in range(len(b1)):
        abs(b1[i] - b2[i]) < 1e-6

  finally:
    shutil.rmtree(test_dir)



"""
def test11_baselines_api(self):
  self.grid_available()
  # test that all of the baselines would execute
  from facereclib.script.baselines import available_databases, all_algorithms, main

  for database in available_databases:
    parameters = [sys.argv[0], '-d', database, '--dry-run']
    main(parameters)
    parameters.append('-g')
    main(parameters)
    parameters.extend(['-e', 'HTER'])
    main(parameters)

  for algorithm in all_algorithms:
    parameters = [sys.argv[0], '-a', algorithm, '--dry-run']
    main(parameters)
    parameters.append('-g')
    main(parameters)
    parameters.extend(['-e', 'HTER'])
    main(parameters)


def test15_evaluate(self):
  # tests our 'evaluate' script using the reference files
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  reference_files = ('scores-nonorm-dev', 'scores-ztnorm-dev')
  plots = [os.path.join(test_dir, '%s.pdf')%f for f in ['roc', 'cmc', 'det']]
  parameters = [
    '--dev-files', reference_files[0], reference_files[1],
    '--eval-files', reference_files[0], reference_files[1],
    '--directory', os.path.join(base_dir, 'scripts'),
    '--legends', 'no norm', 'ZT norm',
    '--criterion', 'HTER',
    '--roc', plots[0],
    '--det', plots[1],
    '--cmc', plots[2],
  ]

  # execute the script
  from facereclib.script.evaluate import main
  main(parameters)
  for i in range(3):
    self.assertTrue(os.path.exists(plots[i]))
    os.remove(plots[i])
  os.rmdir(test_dir)


def test16_collect_results(self):
  # simply test that the collect_results script works
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  from facereclib.script.collect_results import main
  main(['--directory', test_dir, '--sort', '--sort-key', 'dir', '--criterion', 'FAR', '--self-test'])
  os.rmdir(test_dir)


def test21_parameter_script(self):
  self.grid_available()
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # tests that the parameter_test.py script works properly

  # first test without grid option
  parameters = [
      sys.argv[0],
      '-c', os.path.join(base_dir, 'scripts', 'parameter_Test.py'),
      '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
      '-f', 'lgbphs',
      '-b', 'test_p',
      '-s', '.',
      '-T', test_dir,
      '-R', test_dir,
      '--', '--dry-run',
  ]
  from facereclib.script.parameter_test import main
  main(parameters)

  # number of jobs should be 12
  self.assertEqual(facereclib.script.parameter_test.task_count, 12)
  # but no job in the grid
  self.assertEqual(facereclib.script.parameter_test.job_count, 0)

  # now, in the grid...
  parameters = [
      sys.argv[0],
      '-c', os.path.join(base_dir, 'scripts', 'parameter_Test.py'),
      '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
      '-f', 'lgbphs',
      '-b', 'test_p',
      '-i', '.',
      '-s', '.',
      '-T', test_dir,
      '-R', test_dir,
      '-g', 'grid',
      '--', '--dry-run',
  ]
  main(parameters)

  # number of jobs should be 12
  self.assertEqual(facereclib.script.parameter_test.task_count, 12)
  # number of jobs in the grid: 36 (including best possible re-use of files; minus preprocessing)
  self.assertEqual(facereclib.script.parameter_test.job_count, 36)

  shutil.rmtree(test_dir)
"""
