#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
from __future__ import print_function

from . import verify

import argparse, os, sys
import copy # for deep copies of dictionaries
from .. import utils, tools
from ..tools import is_idiap

import bob.core
logger = bob.core.log.setup("bob.bio.base")

# the configuration read from config file
global configuration
# the place holder key given on command line
global place_holder_key
# the extracted command line arguments
global args
# the job ids as returned by the call to the verify function
global job_ids
# first fake job id (useful for the --dry-run option)
global fake_job_id
fake_job_id = 0
# the number of grid jobs that are executed
global job_count
# the total number of experiments run
global task_count
# the directories, where score files will be generated
global score_directories


# The different steps of the processing chain.
# Use these keywords to change parameters of the specific part
steps = ['preprocess', 'extract', 'project', 'enroll', 'score']


def command_line_options(command_line_parameters):
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-c', '--configuration-file', required = True,
      help = 'The file containing the information what parameters you want to have tested.')

  parser.add_argument('-k', '--place-holder-key', default = '#',
      help = 'The place holder key that starts the place holders which will be replaced.')

  parser.add_argument('-d', '--database', required = True,
      help = 'The database that you want to execute the experiments on.')

  parser.add_argument('-P', '--protocol',
      help = 'The protocol that you want to use (if not specified, the default protocol for the database is used).')

  parser.add_argument('-s', '--sub-directory', required = True,
      help = 'The sub-directory where the files of the current experiment should be stored. Please specify a directory name with a name describing your experiment.')

  parser.add_argument('-p', '--preprocessor',
      help = "The preprocessing to be used (will overwrite the 'preprocessor' in the configuration file)")

  parser.add_argument('-e', '--extractor',
      help = "The features to be extracted (will overwrite the 'extractor' in the configuration file)")

  parser.add_argument('-a', '--algorithm',
      help = "The recognition algorithms to be employed (will overwrite the 'algorithm' in the configuration file)")

  parser.add_argument('-g', '--grid',
      help = 'The SGE grid configuration')

  parser.add_argument('-l', '--parallel', type=int,
      help = 'Run the algorithms in parallel on the local machine, using the given number of parallel threads')

  parser.add_argument('-L', '--gridtk-database-split-level', metavar='LEVEL', type=int, default=-1,
      help = 'Split the gridtk databases after the following level -1 - never split; 0 - preprocess; 1 - extract; 2 -- project; 3 -- enroll; 4 -- score;')

  parser.add_argument('-x', '--executable', metavar='X',
      help = '(optional) The executable to be executed instead of bob/bio/base/verify.py (which is taken *always* from bob.bio.base, not from the bin directory)')

  parser.add_argument('-R', '--result-directory', metavar='DIR',
      help = 'The directory where to write the resulting score files to.')

  parser.add_argument('-T', '--temp-directory', metavar='DIR',
      help = 'The directory where to write temporary files into.')

  parser.add_argument('-i', '--preprocessed-directory', metavar='DIR',
      help = '(optional) The directory where to read the already preprocessed data from (no preprocessing is performed in this case).')

  parser.add_argument('-G', '--gridtk-database-directory', metavar='DIR', default = 'grid_db',
      help = 'Directory where the submitted.sql3 files should be written into (will create sub-directories on need)')

  parser.add_argument('-w', '--write-commands',
      help = '(optional) The file name where to write the calls into (will not write the dependencies, though)')

  parser.add_argument('-q', '--dry-run', action='store_true',
      help = 'Just write the commands to console and mimic dependencies, but do not execute the commands')

  parser.add_argument('-j', '--skip-when-existent', action='store_true',
      help = 'Skip the submission/execution of jobs when the result directory already exists')

  parser.add_argument('-N', '--replace-variable',
      help = 'Use the given variable instead of the "replace" keyword in the configuration file')

  parser.add_argument('parameters', nargs = argparse.REMAINDER,
      help = "Parameters directly passed to the verify.py script. Use -- to separate this parameters from the parameters of this script. See './bin/verify.py --help' for a complete list of options.")

  bob.core.log.add_command_line_option(parser)

  global args
  args = parser.parse_args(command_line_parameters)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  # set base directories
  if args.temp_directory is None:
    args.temp_directory = "/idiap/temp/%s/grid_search" % os.environ["USER"] if is_idiap() else "temp/grid_search"
  if args.result_directory is None:
    args.result_directory = "/idiap/user/%s/grid_search" % os.environ["USER"] if is_idiap() else "results/grid_search"


  if args.executable:
    global verify
    verify = __import__('importlib').import_module(args.executable)




def extract_values(replacements, indices):
  """Extracts the value dictionary from the given dictionary of replacements"""
  extracted_values = {}
  for place_holder in replacements.keys():
    # get all occurrences of the place holder key
    parts = place_holder.split(place_holder_key)
    # only one part -> no place holder key found -> no strings to be extracted
    if len(parts) == 1:
      continue

    keys = [part[:1] for part in parts[1:]]

    value_index = indices[place_holder]

    entries = replacements[place_holder]
    entry_key = sorted(entries.keys())[value_index]

    # check that the keys are unique
    for key in keys:
      if key in extracted_values:
        raise ValueError("The replacement key '%s' was defined multiple times. Please use each key only once."%key)

    # extract values
    if len(keys) == 1:
      extracted_values[keys[0]] = entries[entry_key]

    else:
      for i in range(len(keys)):
        extracted_values[keys[i]] = entries[entry_key][i]

  return extracted_values


def replace(string, replacements):
  """Replaces the place holders in the given string with the according values from the values dictionary."""
  # get all occurrences of the place holder key
  parts = string.split(place_holder_key)
  # only one part -> no place holder key found -> return the whole string
  if len(parts) == 1:
    return string

  keys = [part[:1] for part in parts[1:]]

  retval = parts[0]
  for i in range(0, len(keys)):
    # replace the place holder by the desired string and add the remaining of the command
    retval += str(replacements[keys[i]]) + str(parts[i+1][1:])

  return retval


def create_command_line(replacements):
  """Creates the parameters for the function call that will be given to the verify script."""
  # get the values to be replaced with
  values = {}
  for key in configuration.replace:
    values.update(extract_values(configuration.replace[key], replacements))
  # replace the place holders with the values
  call = ['--database', args.database]
  if args.protocol:
    call += ['--protocol', args.protocol]
  call += ['--temp-directory', args.temp_directory, '--result-directory', args.result_directory]
  return call + [
      '--preprocessor', replace(configuration.preprocessor, values),
      '--extractor', replace(configuration.extractor, values),
      '--algorithm', replace(configuration.algorithm, values),
      '--imports'
  ] + configuration.imports



# Parts that could be skipped when the dependecies are on the indexed level
skips = [[''],
         ['--skip-preprocessing'],
         ['--skip-extractor-training', '--skip-extraction'],
         ['--skip-projector-training', '--skip-projection'],
         ['--skip-enroller-training', '--skip-enrollment']
        ]

# The keywords to parse the job ids to get the according dependencies right
dependency_keys  = ['DUMMY', 'preprocess', 'extract', 'project', 'enroll']


def directory_parameters(directories):
  """This function generates the verify parameters that define the directories, where the data is stored.
  The directories are set such that data is reused whenever possible, but disjoint if needed."""
  def _join_dirs(index, subdir):
    # collect sub-directories
    dirs = []
    for i in range(index+1):
      dirs += directories[steps[i]]
    if not dirs:
      return subdir
    else:
      dir = dirs[0]
      for d in dirs[1:]:
        dir = os.path.join(dir, d)
      return os.path.join(dir, subdir)

  global args
  parameters = []

  # add directory parameters
  # - preprocessing
  if args.preprocessed_directory:
    parameters += ['--preprocessed-directory', os.path.join(args.preprocessed_directory, _join_dirs(0, 'preprocessed'))] + skips[1]
  else:
    parameters += ['--preprocessed-directory', _join_dirs(0, 'preprocessed')]

  # - feature extraction
  parameters += ['--extracted-directory', _join_dirs(1, 'extracted'), '--extractor-file', _join_dirs(1, 'Extractor.hdf5')]

  # - feature projection
  parameters += ['--projected-directory', _join_dirs(2, 'projected'), '--projector-file', _join_dirs(2, 'Projector.hdf5')]

  # - model enrollment
  parameters += ['--model-directories', _join_dirs(3, 'N-Models'), _join_dirs(3, 'T-Models'), '--enroller-file', _join_dirs(3, 'Enroller.hdf5')]

  # - scoring
  parameters += ['--score-directories', _join_dirs(4, 'nonorm'), _join_dirs(4, 'ztnorm')]

  # the sub-dorectory, given on command line
  parameters += ['--sub-directory', args.sub_directory]

  global score_directories
  score_directories.append(_join_dirs(4, '.'))

  # grid database
  if args.grid is not None or args.parallel is not None:
    # we get one database per preprocessing job (all others might have job inter-dependencies)
    parameters += ['--gridtk-database-file', os.path.join(args.gridtk_database_directory, _join_dirs(args.gridtk_database_split_level, 'submitted.sql3'))]

  return parameters


def check_requirements(replacements):
  # check if the requirement are met
  global configuration
  values = {}
  for key in configuration.replace:
    # check that the key is one of the known steps
    if key not in steps:
      raise ValueError("The step '%s' defined in the configuration file '%s' is unknown; choose one of %s" % (key, args.configuration_file, steps))
    values.update(extract_values(configuration.replace[key], replacements))
  for requirement in configuration.requirements:
    test = replace(requirement, values)
    while not isinstance(test, bool):
      test = eval(test)
    if not test:
      return False
  return True


def execute_dependent_task(command_line, directories, dependency_level):
  # add other command line arguments
  if args.grid:
    command_line += ['--grid', args.grid, '--stop-on-failure']
  if args.parallel:
    command_line += ['--parallel', str(args.parallel)]

  if args.verbose:
    command_line += ['-' + 'v'*args.verbose]

  # create directory parameters
  command_line += directory_parameters(directories)

  # add skip parameters according to the dependency level
  for i in range(1, dependency_level+1):
    command_line += skips[i]

  if args.parameters is not None:
    command_line += args.parameters[1:]

  # write the command to file?
  if args.write_commands:
    index = command_line.index('--gridtk-database-file')
    command_file = os.path.join(os.path.dirname(command_line[index+1]), args.write_commands)
    bob.io.base.create_directories_safe(os.path.dirname(command_file))
    with open(command_file, 'w') as f:
      f.write('bin/verify.py ')
      for p in command_line:
        f.write(p + ' ')
      f.close()
    logger.info("Wrote command line into file '%s'", command_file)

  # extract dependencies
  global job_ids
  dependencies = []
  for k in sorted(job_ids.keys()):
    for i in range(1, dependency_level+1):
      if k.find(dependency_keys[i]) != -1:
        dependencies.append(job_ids[k])

  # add dependencies
  if dependencies:
    command_line += ['--external-dependencies'] + [str(d) for d in sorted(list(set(dependencies)))]

  # execute the command
  new_job_ids = {}
  try:
    verif_args = verify.parse_arguments(command_line)
    result_dirs = [os.path.join(verif_args.result_directory, verif_args.database.protocol, verif_args.score_directories[i]) for i in ((0,1) if verif_args.zt_norm else (0,))]
    if not args.skip_when_existent or not all(os.path.exists(result_dir) for result_dir in result_dirs):
      # get the command line parameter for the result directory
      if args.dry_run:
        if args.verbose:
          print ("Would have executed job", tools.command_line(command_line))
      else:
        # execute the verification experiment
        global fake_job_id
        new_job_ids = verify.verify(verif_args, command_line, external_fake_job_id = fake_job_id)
    else:
      logger.info("Skipping execution of %s since result directories '%s' already exists", tools.command_line(command_line), result_dirs)

  except Exception as e:
    logger.error("The execution of job was rejected!\n%s\n Reason:\n%s", tools.command_line(command_line), e)

  # some statistics
  global job_count, task_count
  job_count += len(new_job_ids)
  task_count += 1
  fake_job_id += 100
  job_ids.update(new_job_ids)


def create_recursive(replace_dict, step_index, directories, dependency_level, keys=[]):
  """Iterates through all the keywords and replaces all place holders with all keywords in a defined order."""

  # check if we are at the lowest level
  if step_index == len(steps):
    # create a call and execute it
    if check_requirements(replace_dict):
      execute_dependent_task(create_command_line(replace_dict), directories, dependency_level)
  else:
    if steps[step_index] not in directories:
      directories[steps[step_index]] = []

    # we are at another level
    if steps[step_index] not in configuration.replace.keys():
      # nothing to be replaced here, so just go to the next level
      create_recursive(replace_dict, step_index+1, directories, dependency_level)
    else:
      # iterate through the keys
      if keys == []:
        # call this function recursively by defining the set of keys that we need
        create_recursive(replace_dict, step_index, directories, dependency_level, keys = sorted(configuration.replace[steps[step_index]].keys()))
      else:
        # create a deep copy of the replacement dict to be able to modify it
        replace_dict_copy = copy.deepcopy(replace_dict)
        directories_copy = copy.deepcopy(directories)
        # iterate over all replacements for the first of the keys
        key = keys[0]
        replacement_directories = sorted(configuration.replace[steps[step_index]][key])
        directories_copy[steps[step_index]].append("")
        new_dependency_level = dependency_level
        for replacement_index in range(len(replacement_directories)):
          # increase the counter of the current replacement
          replace_dict_copy[key] = replacement_index
          directories_copy[steps[step_index]][-1] = replacement_directories[replacement_index]
          # call the function recursively
          if len(keys) == 1:
            # we have to go to the next level
            create_recursive(replace_dict_copy, step_index+1, directories_copy, new_dependency_level)
          else:
            # we have to subtract the keys
            create_recursive(replace_dict_copy, step_index, directories_copy, new_dependency_level, keys = keys[1:])
          new_dependency_level = step_index


def main(command_line_parameters = None):
  """Main entry point for the parameter test. Try --help to see the parameters that can be specified."""

  global task_count, job_count, job_ids, score_directories
  job_count = 0
  task_count = 0
  job_ids = {}
  score_directories = []

  command_line_options(command_line_parameters)

  global configuration, place_holder_key
  configuration = utils.read_config_file(args.configuration_file)
  place_holder_key = args.place_holder_key

  if args.preprocessor:
    configuration.preprocessor = args.preprocessor
  if args.extractor:
    configuration.extractor = args.extractor
  if args.algorithm:
    configuration.algorithm = args.algorithm

  if args.replace_variable is not None:
    exec("configuration.replace = configuration.%s" % args.replace_variable)

  for attribute in ('preprocessor', 'extractor', 'algorithm'):
    if not hasattr(configuration, attribute):
      raise ValueError("The given configuration file '%s' does not contain the required attribute '%s', and it was not given on command line either" %(args.configuration_file, attribute))

  # extract the dictionary of replacements from the configuration
  if not hasattr(configuration, 'replace'):
    raise ValueError("Please define a set of replacements using the 'replace' keyword.")
  if not hasattr(configuration, 'imports'):
    configuration.imports = ['bob.bio.base']
    logger.info("No 'imports' specified in configuration file '%s' -> using default %s", args.configuration_file, configuration.imports)

  if not hasattr(configuration, 'requirements'):
    configuration.requirements = []

  replace_dict = {}
  for step, replacements in configuration.replace.items():
    for key in replacements.keys():
      if key in replace_dict:
        raise ValueError("The replacement key '%s' was defined multiple times. Please use each key only once.")
      # we always start with index 0.
      replace_dict[key] = 0

  # now, iterate through the list of replacements and create the according calls
  create_recursive(replace_dict, step_index = 0, directories = {}, dependency_level = 0)

  # finally, write some information about the
  if args.grid is not None:
    logger.info("The number of executed tasks is: %d, which are split up into %d jobs that are executed in the grid" , task_count, job_count)

  if args.parallel is not None:
    logger.info("The total amount of finsihed tasks is: %d", task_count)

  return score_directories
