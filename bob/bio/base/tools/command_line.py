import argparse
import os
import sys
import socket

import bob.core
logger = bob.core.log.setup("bob.bio.base")

from .. import utils
from . import FileSelector
from .. import database

"""Execute biometric recognition algorithms on a certain biometric database.
"""

def is_idiap():
  return os.path.isdir("/idiap") and "USER" in os.environ

def command_line_parser(description=__doc__, exclude_resources_from=[]):
  """command_line_parser(description=__doc__, exclude_resources_from=[]) -> parsers

  Creates an :py:class:`argparse.ArgumentParser` object that includes the minimum set of command line options (which is not so few).
  The ``description`` can be overwritten, but has a (small) default.

  Included in the parser, several groups are defined.
  Each group specifies a set of command line options.
  For the configurations, registered resources are listed, which can be limited by the ``exclude_resources_from`` list of extensions.

  It returns a dictionary, containing the parser object itself (in the ``'main'`` keyword), and a list of command line groups.

  **Parameters:**

  description : str
    The documentation of the script.

  exclude_resources_from : [str]
    A list of extension packages, for which resources should not be listed.

  **Returns:**

  parsers : dict
    A dictionary of parser groups, with the main parser under the 'main' key.
    Feel free to add more options to any of the parser groups.
  """
  parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

  #######################################################################################
  ############## options that are required to be specified #######################
  config_group = parser.add_argument_group('\nParameters defining the experiment. Most of these parameters can be a registered resource, a configuration file, or even a string that defines a newly created object')
  config_group.add_argument('-d', '--database', metavar = 'x', nargs = '+', required = True,
      help = 'Database and the protocol; registered databases are: %s' % utils.resource_keys('database', exclude_resources_from))
  config_group.add_argument('-p', '--preprocessor', metavar = 'x', nargs = '+', required = True,
      help = 'Data preprocessing; registered preprocessors are: %s' % utils.resource_keys('preprocessor', exclude_resources_from))
  config_group.add_argument('-e', '--extractor', metavar = 'x', nargs = '+', required = True,
      help = 'Feature extraction; registered feature extractors are: %s' % utils.resource_keys('extractor', exclude_resources_from))
  config_group.add_argument('-a', '--algorithm', metavar = 'x', nargs = '+', required = True,
      help = 'Biometric recognition; registered algorithms are: %s' % utils.resource_keys('algorithm', exclude_resources_from))
  config_group.add_argument('-g', '--grid', metavar = 'x', nargs = '+',
      help = 'Configuration for the grid setup; if not specified, the commands are executed sequentially on the local machine.')
  config_group.add_argument('-I', '--imports', metavar = 'LIB', nargs = '+', default = ['bob.bio.base'],
      help = 'If one of your configuration files is an actual command, please specify the lists of required libraries (imports) to execute this command')
  config_group.add_argument('-W', '--preferred-package', metavar = 'LIB',
      help = 'If resources with identical names are defined in several packages, prefer the one from the given package')
  config_group.add_argument('-s', '--sub-directory', metavar = 'DIR', required = True,
      help = 'The sub-directory where the files of the current experiment should be stored. Please specify a directory name with a name describing your experiment')
  config_group.add_argument('--groups', metavar = 'GROUP', nargs = '+', default = ['dev'],
      help = "The groups (i.e., 'dev', 'eval') for which the models and scores should be generated; by default, only the 'dev' group is evaluated")
  config_group.add_argument('-P', '--protocol', metavar='PROTOCOL',
      help = 'Overwrite the protocol that is stored in the database by the given one (might not by applicable for all databases).')

  #######################################################################################
  ############## options to modify default directories or file names ####################

  # directories differ between idiap and extern
  temp = "/idiap/temp/%s/database-name/sub-directory" % os.environ["USER"] if is_idiap() else "temp"
  results = "/idiap/user/%s/database-name/sub-directory" % os.environ["USER"] if is_idiap() else "results"
  database_replacement = "%s/.bob_bio_databases.txt" % os.environ["HOME"]

  dir_group = parser.add_argument_group('\nDirectories that can be changed according to your requirements')
  dir_group.add_argument('-T', '--temp-directory', metavar = 'DIR',
      help = 'The directory for temporary files, default is: %s.' % temp)
  dir_group.add_argument('-R', '--result-directory', metavar = 'DIR',
      help = 'The directory for resulting score files, default is: %s.' % results)

  file_group = parser.add_argument_group('\nName (maybe including a path relative to the --temp-directory, if not specified otherwise) of files that will be generated. Note that not all files will be used by all algorithms')
  file_group.add_argument('--extractor-file', metavar = 'FILE', default = 'Extractor.hdf5',
      help = 'Name of the file to write the feature extractor into.')
  file_group.add_argument('--projector-file', metavar = 'FILE', default = 'Projector.hdf5',
      help = 'Name of the file to write the feature projector into.')
  file_group.add_argument('--enroller-file' , metavar = 'FILE', default = 'Enroller.hdf5',
      help = 'Name of the file to write the model enroller into.')
  file_group.add_argument('-G', '--gridtk-database-file', metavar = 'FILE', default = 'submitted.sql3',
      help = 'The database file in which the submitted jobs will be written; relative to the current directory (only valid with the --grid option).')
  file_group.add_argument('--experiment-info-file', metavar = 'FILE', default = 'Experiment.info',
      help = 'The file where the configuration of all parts of the experiments are written; relative to te --result-directory.')
  file_group.add_argument('-D', '--database-directories-file', metavar = 'FILE', default = database_replacement,
      help = 'An optional file, where database directories are stored (to avoid changing the database configurations)')

  sub_dir_group = parser.add_argument_group('\nSubdirectories of certain parts of the tool chain. You can specify directories in case you want to reuse parts of the experiments (e.g. extracted features) in other experiments. Please note that these directories are relative to the --temp-directory, but you can also specify absolute paths')
  sub_dir_group.add_argument('--preprocessed-directory', metavar = 'DIR', default = 'preprocessed',
      help = 'Name of the directory of the preprocessed data.')
  sub_dir_group.add_argument('--extracted-directory', metavar = 'DIR', default = 'extracted',
      help = 'Name of the directory of the extracted features.')
  sub_dir_group.add_argument('--projected-directory', metavar = 'DIR', default = 'projected',
      help = 'Name of the directory where the projected data should be stored.')
  sub_dir_group.add_argument('--model-directories', metavar = 'DIR', nargs = '+', default = ['models', 'tmodels'],
      help = 'Name of the directory where the models (and T-Norm models) should be stored')
  sub_dir_group.add_argument('--score-directories', metavar = 'DIR', nargs = '+', default = ['nonorm', 'ztnorm'],
      help = 'Name of the directory (relative to --result-directory) where to write the results to')
  sub_dir_group.add_argument('--zt-directories', metavar = 'DIR', nargs = 5, default = ['zt_norm_A', 'zt_norm_B', 'zt_norm_C', 'zt_norm_D', 'zt_norm_D_sameValue'],
      help = 'Name of the directories (of --temp-directory) where to write the ZT-norm values; only used with --zt-norm')
  sub_dir_group.add_argument('--grid-log-directory', metavar = 'DIR', default = 'gridtk_logs',
      help = 'Name of the directory (relative to --temp-directory) where to log files are written; only used with --grid')

  flag_group = parser.add_argument_group('\nFlags that change the behavior of the experiment')
  bob.core.log.add_command_line_option(flag_group)
  flag_group.add_argument('-q', '--dry-run', action='store_true',
      help = 'Only report the commands that will be executed, but do not execute them.')
  flag_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist')
  flag_group.add_argument('-Z', '--write-compressed-score-files', action='store_true',
      help = 'Writes score files which are compressed with tar.bz2.')
  flag_group.add_argument('-S', '--stop-on-failure', action='store_true',
      help = 'Try to recursively stop the dependent jobs from the SGE grid queue, when a job failed')
  flag_group.add_argument('-X', '--external-dependencies', type=int, default = [], nargs='+',
      help = 'The jobs submitted to the grid have dependencies on the given job ids.')
  flag_group.add_argument('-B', '--timer', choices=('real', 'system', 'user'), nargs = '*',
      help = 'Measure and report the time required by the execution of the tool chain (only on local machine)')
  flag_group.add_argument('-L', '--run-local-scheduler', action='store_true',
      help = 'Starts the local scheduler after submitting the jobs to the local queue (by default, local jobs must be started by hand, e.g., using ./bin/jman --local -vv run-scheduler -x)')
  flag_group.add_argument('-N', '--nice', type=int, default=10,
      help = 'Runs the local scheduler with the given nice value')
  flag_group.add_argument('-D', '--delete-jobs-finished-with-status', choices = ('all', 'failure', 'success'),
      help = 'If selected, local scheduler jobs that finished with the given status are deleted from the --gridtk-database-file; otherwise the jobs remain in the database')
  flag_group.add_argument('-c', '--calibrate-scores', action='store_true',
      help = 'Performs score calibration after the scores are computed.')
  flag_group.add_argument('-z', '--zt-norm', action='store_true',
      help = 'Enable the computation of ZT norms')
  flag_group.add_argument('-A', '--allow-missing-files', action='store_true',
      help = "If given, missing files will not stop the processing; this is helpful if not all files of the database can be processed; missing scores will be NaN.")
  flag_group.add_argument('-r', '--parallel', type=int,
      help = 'This flag is a shortcut for running the commands on the local machine with the given amount of parallel threads; equivalent to --grid bob.bio.base.grid.Grid("local", number_of_parallel_threads=X) --run-local-scheduler --stop-on-failure.')

  flag_group.add_argument('-t', '--environment', dest='env', nargs='*', default=[], help='Passes specific environment variables to the job.')

  return {
    'main' : parser,
    'config' : config_group,
    'dir' : dir_group,
    'sub-dir' : sub_dir_group,
    'file' : file_group,
    'flag' : flag_group
  }



def initialize(parsers, command_line_parameters = None, skips = []):
  """initialize(parsers, command_line_parameters = None, skips = []) -> args

  Parses the command line and arranges the arguments accordingly.
  Afterward, it loads the resources for the database, preprocessor, extractor, algorithm and grid (if specified), and stores the results into the returned args.

  This function also initializes the :py:class:`FileSelector` instance by arranging the directories and files according to the command line parameters.

  If the ``skips`` are given, an '--execute-only' parameter is added to the parser, according skips are selected.

  **Parameters:**

  parsers : dict
    The dictionary of command line parsers, as returned from :py:func:`command_line_parser`.
    Additional arguments might have been added.

  command_line_parameters : [str] or None
    The command line parameters that should be interpreted.
    By default, the parameters specified by the user on command line are considered.

  skips : [str]
    A list of possible ``--skip-...`` options to be added and evaluated automatically.

  **Returns:**

  args : namespace
    A namespace of arguments as read from the command line.

    .. note:: The database, preprocessor, extractor, algorithm and grid (if specified) are actual instances of the according classes.
  """

  # execute-only
  if skips is not None:
    #######################################################################################
    ################# options for skipping parts of the toolchain #########################
    skip_group = parsers['main'].add_argument_group('\nFlags that allow to skip certain parts of the experiments. This does only make sense when the generated files are already there (e.g. when reusing parts of other experiments)')
    for skip in skips:
      skip_group.add_argument('--skip-%s' % skip, action='store_true', help = 'Skip the %s step.' % skip)
    skip_group.add_argument('-o', '--execute-only', nargs = '+', choices = skips, help = 'If specified, executes only the given parts of the tool chain.')

  args = parsers['main'].parse_args(command_line_parameters)

  # evaluate skips
  if skips is not None and args.execute_only is not None:
    for skip in skips:
      if skip not in args.execute_only:
        exec("args.skip_%s = True" % (skip.replace("-", "_")))

  if args.parallel is not None:
    args.grid = ['bob.bio.base.grid.Grid("local", number_of_parallel_processes = %d)' % args.parallel]
    args.run_local_scheduler = True
    args.stop_on_failure = True

  # logging
  bob.core.log.set_verbosity_level(logger, args.verbose)

  # timer
  if args.timer is not None and not len(args.timer):
    args.timer = ('real', 'system', 'user')

  # load configuration resources
  args.database = utils.load_resource(' '.join(args.database), 'database', imports = args.imports, preferred_package = args.preferred_package)
  args.preprocessor = utils.load_resource(' '.join(args.preprocessor), 'preprocessor', imports = args.imports, preferred_package = args.preferred_package)
  args.extractor = utils.load_resource(' '.join(args.extractor), 'extractor', imports = args.imports, preferred_package = args.preferred_package)
  args.algorithm = utils.load_resource(' '.join(args.algorithm), 'algorithm', imports = args.imports, preferred_package = args.preferred_package)
  if args.grid is not None:
    args.grid = utils.load_resource(' '.join(args.grid), 'grid', imports = args.imports, preferred_package = args.preferred_package)

  # set base directories
  if args.temp_directory is None:
    args.temp_directory = "/idiap/temp/%s/%s" % (os.environ["USER"], args.database.name) if is_idiap() else "temp"
  if args.result_directory is None:
    args.result_directory = "/idiap/user/%s/%s" % (os.environ["USER"], args.database.name) if is_idiap() else "results"

  args.temp_directory = os.path.join(args.temp_directory, args.sub_directory)
  args.result_directory = os.path.join(args.result_directory, args.sub_directory)
  args.grid_log_directory = os.path.join(args.temp_directory, args.grid_log_directory)


  # protocol command line override
  if args.protocol is not None:
    args.database.protocol = args.protocol

  protocol = 'None' if args.database.protocol is None else args.database.protocol

  # result files
  args.info_file = os.path.join(args.result_directory, protocol, args.experiment_info_file)

  # sub-directorues that depend on the database
  extractor_sub_dir = protocol if args.database.training_depends_on_protocol and args.extractor.requires_training else '.'
  projector_sub_dir = protocol if args.database.training_depends_on_protocol and args.algorithm.requires_projector_training else extractor_sub_dir
  enroller_sub_dir = protocol if args.database.training_depends_on_protocol and args.algorithm.requires_enroller_training else projector_sub_dir
  model_sub_dir = protocol if args.database.models_depend_on_protocol else enroller_sub_dir

  # Database directories, which should be automatically replaced
  if isinstance(args.database, database.DatabaseBob):
    args.database.replace_directories(args.database_directories_file)

  # initialize the file selector
  FileSelector.create(
    database = args.database,
    extractor_file = os.path.join(args.temp_directory, extractor_sub_dir, args.extractor_file),
    projector_file = os.path.join(args.temp_directory, projector_sub_dir, args.projector_file),
    enroller_file = os.path.join(args.temp_directory, enroller_sub_dir, args.enroller_file),

    preprocessed_directory = os.path.join(args.temp_directory, args.preprocessed_directory),
    extracted_directory = os.path.join(args.temp_directory, extractor_sub_dir, args.extracted_directory),
    projected_directory = os.path.join(args.temp_directory, projector_sub_dir, args.projected_directory),
    model_directories = [os.path.join(args.temp_directory, model_sub_dir, m) for m in args.model_directories],
    score_directories = [os.path.join(args.result_directory, protocol, z) for z in args.score_directories],
    zt_score_directories = [os.path.join(args.temp_directory, protocol, s) for s in args.zt_directories],
    compressed_extension = '.tar.bz2' if args.write_compressed_score_files else '',
    default_extension = '.hdf5',
  )

  return args


def groups(args):
  """groups(args) -> groups

  Returns the groups, for which the files must be preprocessed, and features must be extracted and projected.
  This function should be used in order to eliminate the training files (the ``'world'`` group), when no training is required in this experiment.

  **Parameters:**

  args : namespace
    The interpreted command line arguments as returned by the :py:func:`initialize` function.

  **Returns:**

  groups : [str]
    A list of groups, for which data needs to be treated.
  """
  groups = args.groups[:]
  if args.extractor.requires_training or args.algorithm.requires_projector_training or args.algorithm.requires_enroller_training:
    groups.append('world')
  return groups


def command_line(cmdline):
  """command_line(cmdline) -> str

  Converts the given options to a string that can be executed in a terminal.
  Parameters are enclosed into ``'...'`` quotes so that the command line can interpret them (e.g., if they contain spaces or special characters).

  **Parameters:**

  cmdline : [str]
    A list of command line options to be converted into a string.

  **Returns:**

  str : str
    The command line string that can be copy-pasted into the terminal.
  """
  c = ""
  for cmd in cmdline:
    if cmd[0] in '/-':
      c += "%s " % cmd
    else:
      c += "'%s' " % cmd
  return c


def write_info(args, command_line_parameters, executable):
  """Writes information about the current experimental setup into a file specified on command line.

  **Parameters:**

  args : namespace
    The interpreted command line arguments as returned by the :py:func:`initialize` function.

  command_line_parameters : [str] or ``None``
    The command line parameters that have been interpreted.
    If ``None``, the parameters specified by the user on command line are considered.

  executable : str
    The name of the executable (such as ``'./bin/verify.py'``) that is used to run the experiments.
  """
  if command_line_parameters is None:
    command_line_parameters = sys.argv[1:]
  # write configuration
  try:
    bob.io.base.create_directories_safe(os.path.dirname(args.info_file))
    f = open(args.info_file, 'w')
    f.write("Command line:\n")
    f.write(command_line([executable] + command_line_parameters) + "\n\n")
    f.write("Host: %s\n" % socket.gethostname())
    f.write("Configuration:\n")
    f.write("Database:\n%s\n\n" % args.database)
    f.write("Preprocessor:\n%s\n\n" % args.preprocessor)
    f.write("Extractor:\n%s\n\n" % args.extractor)
    f.write("Algorithm:\n%s\n\n" % args.algorithm)
  except IOError:
    logger.error("Could not write the experimental setup into file '%s'", args.info_file)
