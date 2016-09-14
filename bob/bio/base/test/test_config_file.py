#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''Tests for the configuration-file command line options'''

import os
import shutil
import tempfile

from ..script.verify import parse_arguments


def tmp_file(contents):
  '''Generates a temporary configuration file with the contents on the input'''

  retval = tempfile.NamedTemporaryFile('w')
  retval.write('\n'.join(contents) + '\n')
  retval.flush()
  return retval


def check_parameters(args_file, args_cmdline):
  '''Checks parameters generated from a configuration file or command-line
  are as similar they can be'''

  from bob.bio.base.test.dummy.database import DummyDatabase
  assert isinstance(args_file.database, DummyDatabase)
  assert isinstance(args_cmdline.database, DummyDatabase)
  from bob.bio.base.test.dummy.preprocessor import DummyPreprocessor
  assert isinstance(args_file.preprocessor, DummyPreprocessor)
  assert isinstance(args_cmdline.preprocessor, DummyPreprocessor)
  from bob.bio.base.test.dummy.extractor import DummyExtractor
  assert isinstance(args_file.extractor, DummyExtractor)
  assert isinstance(args_cmdline.extractor, DummyExtractor)
  from bob.bio.base.test.dummy.algorithm import DummyAlgorithm
  assert isinstance(args_file.algorithm, DummyAlgorithm)
  assert isinstance(args_cmdline.algorithm, DummyAlgorithm)

  # elements checked otherwise or not comparable between the two settings
  skip_check = (
      'configuration_file',
      'imports',
      'database',
      'preprocessor',
      'extractor',
      'algorithm',
      )

  for attr in [k for k in dir(args_file) if not k.startswith('_')]:
    if attr in skip_check: continue
    assert hasattr(args_cmdline, attr)
    attr_cmdline = getattr(args_cmdline, attr)
    attr_file = getattr(args_file, attr)
    if (isinstance(attr_file, (bool, str, int, list))) or (attr_file is None):
      assert attr_cmdline == attr_file, '(%s) %r != %r' % \
          (attr, attr_cmdline, attr_file)
    else:
      assert False, '(%s) %r == %r?' % (attr, attr_cmdline, attr_file)


def test_basic():

  test_dir = None
  test_config_file = None
  try:
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    test_config_file = tmp_file([
      'from bob.bio.base.test.dummy.database import database',
      'from bob.bio.base.test.dummy.preprocessor import preprocessor',
      'from bob.bio.base.test.dummy.extractor import extractor',
      'from bob.bio.base.test.dummy.algorithm import algorithm',
      'zt_norm = True',
      'verbose = 1',
      'sub_directory = "test_config"',
      'temp_directory = "%s"' % test_dir,
      'result_directory = "%s"' % test_dir,
      ])

    args = parse_arguments([test_config_file.name])

    assert args.zt_norm is True
    assert args.verbose == 1
    assert args.sub_directory.endswith('test_config')
    assert args.temp_directory.startswith(test_dir)
    assert args.result_directory.startswith(test_dir)
    assert args.allow_missing_files is False

    from bob.bio.base.test.dummy.database import DummyDatabase
    assert isinstance(args.database, DummyDatabase)
    from bob.bio.base.test.dummy.preprocessor import DummyPreprocessor
    assert isinstance(args.preprocessor, DummyPreprocessor)
    from bob.bio.base.test.dummy.extractor import DummyExtractor
    assert isinstance(args.extractor, DummyExtractor)
    from bob.bio.base.test.dummy.algorithm import DummyAlgorithm
    assert isinstance(args.algorithm, DummyAlgorithm)

  finally:
    if test_dir: shutil.rmtree(test_dir)
    if test_config_file: del test_config_file


def test_compare_to_cmdline_basic():

  test_dir = None
  test_config_file = None
  try:
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    test_config_file = tmp_file([
      'from bob.bio.base.test.dummy.database import database',
      'from bob.bio.base.test.dummy.preprocessor import preprocessor',
      'from bob.bio.base.test.dummy.extractor import extractor',
      'from bob.bio.base.test.dummy.algorithm import algorithm',
      'zt_norm = True',
      'verbose = 1',
      'sub_directory = "test_config"',
      'temp_directory = "%s"' % test_dir,
      'result_directory = "%s"' % test_dir,
      ])

    args_file = parse_arguments([test_config_file.name])

    # now do the same with command-line arguments, ensure result is equal
    args_cmdline = parse_arguments([
      '-d', 'bob.bio.base.test.dummy.database.DummyDatabase()',
      '-p', 'bob.bio.base.test.dummy.preprocessor.DummyPreprocessor()',
      '-e', 'bob.bio.base.test.dummy.extractor.DummyExtractor()',
      '-a', 'bob.bio.base.test.dummy.algorithm.DummyAlgorithm()',
      '--zt-norm',
      '-vs', 'test_config',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--imports', 'bob.bio.base.test.dummy',
      ])

    check_parameters(args_file, args_cmdline)

  finally:
    if test_dir: shutil.rmtree(test_dir)
    if test_config_file: del test_config_file


def test_compare_to_cmdline_resources():

  test_dir = None
  test_config_file = None
  try:
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    test_config_file = tmp_file([
      'database = "dummy"',
      'preprocessor = "dummy"',
      'extractor = "dummy"',
      'algorithm = "dummy"',
      'zt_norm = True',
      'allow_missing_files = True',
      'verbose = 1',
      'sub_directory = "test_config"',
      'temp_directory = "%s"' % test_dir,
      'result_directory = "%s"' % test_dir,
      'preferred_package = "bob.bio.base"',
      ])

    args_file = parse_arguments([test_config_file.name])

    # now do the same with command-line arguments, ensure result is equal
    args_cmdline = parse_arguments([
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy',
      '-a', 'dummy',
      '--zt-norm',
      '--allow-missing-files',
      '-vs', 'test_config',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.base',
      ])

    check_parameters(args_file, args_cmdline)

  finally:
    if test_dir: shutil.rmtree(test_dir)
    if test_config_file: del test_config_file


def test_compare_to_cmdline_skip():

  test_dir = None
  test_config_file = None
  try:
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    test_config_file = tmp_file([
      'database = "dummy"',
      'preprocessor = "dummy"',
      'extractor = "dummy"',
      'skip_preprocessing = True',
      'skip_extraction = True',
      'algorithm = "dummy"',
      'zt_norm = True',
      'allow_missing_files = True',
      'verbose = 1',
      'sub_directory = "test_config"',
      'temp_directory = "%s"' % test_dir,
      'result_directory = "%s"' % test_dir,
      'preferred_package = "bob.bio.base"',
      ])

    args_file = parse_arguments([test_config_file.name])

    # now do the same with command-line arguments, ensure result is equal
    args_cmdline = parse_arguments([
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy',
      '-a', 'dummy',
      '--zt-norm',
      '--allow-missing-files',
      '--skip-preprocessing',
      '--skip-extraction',
      '-vs', 'test_config',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.base',
      ])

    check_parameters(args_file, args_cmdline)

  finally:
    if test_dir: shutil.rmtree(test_dir)
    if test_config_file: del test_config_file


def test_from_resource():

  test_dir = None

  try:
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    args = parse_arguments(['dummy'])

    assert args.sub_directory.endswith('test_dummy')
    assert args.allow_missing_files is False
    assert args.zt_norm is True
    assert args.verbose == 1

    from bob.bio.base.test.dummy.database import DummyDatabase
    assert isinstance(args.database, DummyDatabase)
    from bob.bio.base.test.dummy.preprocessor import DummyPreprocessor
    assert isinstance(args.preprocessor, DummyPreprocessor)
    from bob.bio.base.test.dummy.extractor import DummyExtractor
    assert isinstance(args.extractor, DummyExtractor)
    from bob.bio.base.test.dummy.algorithm import DummyAlgorithm
    assert isinstance(args.algorithm, DummyAlgorithm)

  finally:
    if test_dir: shutil.rmtree(test_dir)


def test_from_module():

  test_dir = None

  try:
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    args = parse_arguments(['bob.bio.base.test.dummy.config'])

    assert args.sub_directory.endswith('test_dummy')
    assert args.allow_missing_files is False
    assert args.zt_norm is True
    assert args.verbose == 1

    from bob.bio.base.test.dummy.database import DummyDatabase
    assert isinstance(args.database, DummyDatabase)
    from bob.bio.base.test.dummy.preprocessor import DummyPreprocessor
    assert isinstance(args.preprocessor, DummyPreprocessor)
    from bob.bio.base.test.dummy.extractor import DummyExtractor
    assert isinstance(args.extractor, DummyExtractor)
    from bob.bio.base.test.dummy.algorithm import DummyAlgorithm
    assert isinstance(args.algorithm, DummyAlgorithm)

  finally:
    if test_dir: shutil.rmtree(test_dir)


def test_order():

  test_dir = None

  try:
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    args = parse_arguments(['dummy', 'dummy2'])

    assert args.sub_directory.endswith('test_dummy2')
    assert args.allow_missing_files is False
    assert args.zt_norm is True
    assert args.verbose == 2

    from bob.bio.base.test.dummy.database import DummyDatabase
    assert isinstance(args.database, DummyDatabase)
    from bob.bio.base.test.dummy.preprocessor import DummyPreprocessor
    assert isinstance(args.preprocessor, DummyPreprocessor)
    from bob.bio.base.test.dummy.extractor import DummyExtractor
    assert isinstance(args.extractor, DummyExtractor)
    from bob.bio.base.test.dummy.algorithm import DummyAlgorithm
    assert isinstance(args.algorithm, DummyAlgorithm)

  finally:
    if test_dir: shutil.rmtree(test_dir)


def test_order_inverse():

  test_dir = None

  try:
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    args = parse_arguments(['dummy2', 'dummy'])

    assert args.sub_directory.endswith('test_dummy')
    assert args.allow_missing_files is False
    assert args.zt_norm is True
    assert args.verbose == 1

    from bob.bio.base.test.dummy.database import DummyDatabase
    assert isinstance(args.database, DummyDatabase)
    from bob.bio.base.test.dummy.preprocessor import DummyPreprocessor
    assert isinstance(args.preprocessor, DummyPreprocessor)
    from bob.bio.base.test.dummy.extractor import DummyExtractor
    assert isinstance(args.extractor, DummyExtractor)
    from bob.bio.base.test.dummy.algorithm import DummyAlgorithm
    assert isinstance(args.algorithm, DummyAlgorithm)

  finally:
    if test_dir: shutil.rmtree(test_dir)
