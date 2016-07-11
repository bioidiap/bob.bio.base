

from __future__ import print_function

import bob.measure

import os
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
    main(parameters)

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

      assert not os.path.exists(os.path.join(test_dir, 'submitted.sql3'))

  finally:
    shutil.rmtree(test_dir)


def test_verify_config():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', os.path.join(dummy_dir, 'database.py'),
      '-p', os.path.join(dummy_dir, 'preprocessor.py'),
      '-e', os.path.join(dummy_dir, 'extractor.py'),
      '-a', os.path.join(dummy_dir, 'algorithm.py'),
      '--zt-norm',
      '-vs', 'test_config',
      '--temp-directory', test_dir,
      '--result-directory', test_dir
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_config')


def test_verify_resources():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy',
      '-a', 'dummy',
      '--zt-norm',
      '--allow-missing-files',
      '-vs', 'test_resource',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.base'
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
      '-vs', 'test_commandline',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--imports', 'bob.bio.base.test.dummy'
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
      '--allow-missing-files',
      '-vs', 'test_parallel',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '-g', 'bob.bio.base.grid.Grid(grid_type = "local", number_of_parallel_processes = 2, scheduler_sleep_time = 0.1)',
      '-G', test_database, '--run-local-scheduler', '--stop-on-failure',
      '-D', 'success',
      '--imports', 'bob.io.image', 'bob.bio.base.test.dummy',
      '--preferred-package', 'bob.bio.base'
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
      '-vs', 'test_compressed',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--write-compressed-score-files',
      '--preferred-package', 'bob.bio.base'
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
      '-vs', 'test_calibrate',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--calibrate-scores',
      '--preferred-package', 'bob.bio.base'
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_calibrate', '-calibrated', score_modifier=('calibrated', ''))


def test_verify_fileset():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', os.path.join(dummy_dir, 'fileset.py'),
      '-p', 'dummy',
      '-e', 'bob.bio.base.test.dummy.extractor.DummyExtractor()',
      '-a', 'dummy',
      '--zt-norm',
      '-vs', 'test_fileset',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.base',
      '--imports', 'bob.bio.base.test.dummy'
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
      '-vs', 'test_filelist',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.base'
  ]

  print (bob.bio.base.tools.command_line(parameters))

  try:
    from bob.bio.base.script.verify import main
    main(parameters)

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


def test_verify_missing():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'bob.bio.base.test.dummy.preprocessor.DummyPreprocessor(return_none=True)',
      '-e', 'dummy',
      '-a', 'dummy',
      '--zt-norm',
      '--allow-missing-files',
      '-vs', 'test_missing',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.base',
      '--imports', 'bob.bio.base.test.dummy'
  ]

  print (bob.bio.base.tools.command_line(parameters))

  try:
    from bob.bio.base.script.verify import main
    main(parameters)

    # assert that the score file exists
    score_files = [os.path.join(test_dir, 'test_missing', 'Default', norm, 'scores-dev') for norm in ('nonorm', 'ztnorm')]
    assert os.path.exists(score_files[0]), "Score file %s does not exist" % score_files[0]
    assert os.path.exists(score_files[1]), "Score file %s does not exist" % score_files[1]

    # assert that all scores are NaN

    for i in (0,1):
      # load scores
      a, b = bob.measure.load.split_four_column(score_files[i])

      assert numpy.all(numpy.isnan(a))
      assert numpy.all(numpy.isnan(b))

  finally:
    shutil.rmtree(test_dir)


def test_fusion():
  # tests that the fuse_scores script is doing something useful
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  reference_files = [os.path.join(data_dir, s) for s in ('scores-nonorm-dev', 'scores-ztnorm-dev')]
  output_files = [os.path.join(test_dir, s) for s in ("fused-dev", "fused-eval")]
  parameters = [
    '--dev-files', reference_files[0], reference_files[1],
    '--eval-files', reference_files[0], reference_files[1],
    '--fused-dev-file', output_files[0],
    '--fused-eval-file', output_files[1],
    '--max-iterations', '100',
    '--convergence-threshold', '1e-4',
    '-v'
  ]

  # execute the script
  from bob.bio.base.script.fuse_scores import main
  try:
    main(parameters)

    # assert that we can read the two files, and that they contain the same number of lines as the original file
    for i in (0,1):
      assert os.path.exists(output_files[i])
      r = bob.measure.load.four_column(reference_files[i])
      o = bob.measure.load.four_column(output_files[i])
      assert len(list(r)) == len(list(o))
  finally:
    shutil.rmtree(test_dir)



def test_evaluate():
  # tests our 'evaluate' script using the reference files
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  reference_files = ('scores-nonorm-dev', 'scores-ztnorm-dev')
  plots = [os.path.join(test_dir, '%s.pdf')%f for f in ['roc', 'cmc', 'det']]
  parameters = [
    '--dev-files', reference_files[0], reference_files[1],
    '--eval-files', reference_files[0], reference_files[1],
    '--directory', os.path.join(data_dir),
    '--legends', 'no norm', 'ZT norm',
    '--criterion', 'HTER',
    '--roc', plots[0],
    '--det', plots[1],
    '--cmc', plots[2],
    '--rr',
    '--thresholds', '5000', '0',
    '-v',
  ]

  # execute the script
  from bob.bio.base.script.evaluate import main
  main(parameters)
  for i in range(3):
    assert os.path.exists(plots[i])
    os.remove(plots[i])
  os.rmdir(test_dir)


def test_resources():
  # simply test that the collect_results script works
  from bob.bio.base.script.resources import resources, databases
  with utils.Quiet():
    resources(['--types', 'database', 'preprocessor', 'extractor', 'algorithm', 'grid', '--details', '--packages', 'bob.bio.base'])
    databases([])


def test_collect_results():
  # simply test that the collect_results script works
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  try:
    from bob.bio.base.script.collect_results import main
    main(['--directory', test_dir, '--sort', '--sort-key', 'dir', '--criterion', 'FAR', '--self-test'])
  finally:
    if os.path.exists(test_dir):
      os.rmdir(test_dir)


@utils.grid_available
def test_grid_search():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # tests that the parameter_test.py script works properly

  try:
    # first test without grid option
    parameters = [
        '-c', os.path.join(dummy_dir, 'grid_search.py'),
        '-d', 'dummy',
        '-e', 'dummy',
        '-s', 'test_grid_search',
        '-T', test_dir,
        '-R', test_dir,
        '-v',
        '--', '--dry-run',
        '--preferred-package', 'bob.bio.base'
    ]
    from bob.bio.base.script.grid_search import main
    with utils.Quiet():
      main(parameters)

    # number of jobs should be 12
    assert bob.bio.base.script.grid_search.task_count == 6
    # but no job in the grid
    assert bob.bio.base.script.grid_search.job_count == 0

    # now, in the grid...
    parameters = [
        '-c', os.path.join(dummy_dir, 'grid_search.py'),
        '-d', 'dummy',
        '-s', 'test_grid_search',
        '-i', '.',
        '-G', test_dir,
        '-T', test_dir,
        '-R', test_dir,
        '-w', 'Job.txt',
        '-g', 'grid',
        '-v',
        '--', '--dry-run',
        '--preferred-package', 'bob.bio.base'
    ]
    with utils.Quiet():
      main(parameters)

    # number of jobs should be 12
    assert bob.bio.base.script.grid_search.task_count == 6
    # number of jobs in the grid: 36 (including best possible re-use of files; minus preprocessing)
    assert bob.bio.base.script.grid_search.job_count == 30

    # and now, finally run locally
    parameters = [
        '-c', os.path.join(dummy_dir, 'grid_search.py'),
        '-d', 'dummy',
        '-s', 'test_grid_search',
        '-G', test_dir,
        '-T', test_dir,
        '-R', test_dir,
        '-w', 'Job.txt',
        '-l', '4', '-L', '-1', '-v',
        '--', '--imports', 'bob.io.image',
        '--dry-run',
        '--preferred-package', 'bob.bio.base'
    ]
    with utils.Quiet():
      main(parameters)

    # number of jobs should be 12
    assert bob.bio.base.script.grid_search.task_count == 6
    # number of jobs in the grid: 36 (including best possible re-use of files; minus preprocessing)
    assert bob.bio.base.script.grid_search.job_count == 0

  finally:
    shutil.rmtree(test_dir)


def test_scripts():
  # Tests the bin/preprocess.py, bin/extract.py, bin/enroll.py and bin/score.py scripts
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  data_file = os.path.join(test_dir, "data.hdf5")
  annotation_file = os.path.join(test_dir, "annotatations.txt")
  preprocessed_file = os.path.join(test_dir, "preprocessed.hdf5")
  preprocessed_image = os.path.join(test_dir, "preprocessed.png")
  extractor_file = os.path.join(test_dir, "extractor.hdf5")
  extracted_file = os.path.join(test_dir, "extracted.hdf5")
  projector_file = os.path.join(test_dir, "projector.hdf5")
  enroller_file = os.path.join(test_dir, "enroller.hdf5")
  model_file = os.path.join(test_dir, "model.hdf5")

  # tests that the parameter_test.py script works properly
  try:
    # create test data
    test_data = utils.random_array((20,20), 0., 255., seed=84)
    test_data[0,0] = 0.
    test_data[19,19] = 255.
    bob.io.base.save(test_data, data_file)
    with open(annotation_file, 'w') as a:
      a.write("leye 100 200\nreye 100 100")

    extractor = bob.bio.base.load_resource("dummy", "extractor")
    extractor.train([], extractor_file)

    algorithm = bob.bio.base.load_resource("dummy", "algorithm")
    algorithm.train_projector([], projector_file)
    algorithm.train_enroller([], enroller_file)

    from bob.bio.base.script.preprocess import main as preprocess
    from bob.bio.base.script.extract import main as extract
    from bob.bio.base.script.enroll import main as enroll
    from bob.bio.base.script.score import main as score

    # preprocessing
    parameters = [
        '-i', data_file,
        '-a', annotation_file,
        '-p', 'dummy',
        '-o', preprocessed_file,
        '-c', preprocessed_image,
        '-v',
    ]
    preprocess(parameters)

    assert os.path.isfile(preprocessed_file)
    assert os.path.isfile(preprocessed_image)
    assert numpy.allclose(bob.io.base.load(preprocessed_file), test_data)
    assert numpy.allclose(bob.io.base.load(preprocessed_image), test_data, rtol=1., atol=1.)

    # feature extraction
    parameters = [
        '-i', preprocessed_file,
        '-p', 'dummy',
        '-e', 'dummy',
        '-E', extractor_file,
        '-o', extracted_file,
        '-v',
    ]
    extract(parameters)

    assert os.path.isfile(extracted_file)
    assert numpy.allclose(bob.io.base.load(extracted_file), test_data.flatten())

    # enrollment
    parameters = [
        '-i', extracted_file, extracted_file,
        '-e', 'dummy',
        '-a', 'dummy',
        '-P', projector_file,
        '-E', enroller_file,
        '-o', model_file,
        '-v',
    ]
    enroll(parameters)

    assert os.path.isfile(model_file)
    assert numpy.allclose(bob.io.base.load(model_file), test_data.flatten())

    # scoring
    parameters = [
        '-m', model_file, model_file,
        '-p', extracted_file, extracted_file,
        '-a', 'dummy',
        '-P', projector_file,
        '-E', enroller_file,
        '-v',
    ]
    with utils.Quiet():
      score(parameters)

  finally:
    shutil.rmtree(test_dir)
