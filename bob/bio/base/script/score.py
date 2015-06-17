"""This script can be used to compute scores between a list of enrolled models and a list of probe files.
"""

from __future__ import print_function

import argparse
import bob.core
logger = bob.core.log.setup("bob.bio.base")

import bob.bio.base


def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-a', '--algorithm', metavar = 'x', nargs = '+', required = True, help = 'Biometric recognition; registered algorithms are: %s' % bob.bio.base.resource_keys('algorithm'))
  parser.add_argument('-P', '--projector-file', metavar = 'FILE', help = 'The pre-trained extractor file, if the algorithm performs projection')
  parser.add_argument('-E', '--enroller-file' , metavar = 'FILE', help = 'The pre-trained enroller file, if the extractor requires enroller training')
  parser.add_argument('-m', '--model-files', metavar = 'MODEL', nargs='+', required = True, help = "A list of enrolled model files")
  parser.add_argument('-p', '--probe-files', metavar = 'PROBE', nargs='+', required = True, help = "A list of extracted feature files used as probes")

  # add verbose option
  bob.core.log.add_command_line_option(parser)
  # parse arguments
  args = parser.parse_args(command_line_parameters)
  # set verbosity level
  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args


def main(command_line_parameters=None):
  """Preprocesses the given image with the given preprocessor."""
  args = command_line_arguments(command_line_parameters)

  logger.debug("Loading algorithm")
  algorithm = bob.bio.base.load_resource(' '.join(args.algorithm), "algorithm")
  if algorithm.requires_projector_training:
    if args.projector_file is None:
      raise ValueError("The desired algorithm requires a pre-trained projector file, but it was not specified")
    algorithm.load_projector(args.projector_file)

  if algorithm.requires_enroller_training:
    if args.enroller_file is None:
      raise ValueError("The desired algorithm requires a pre-trained enroller file, but it was not specified")
    algorithm.load_enroller(args.enroller_file)

  models, probes = {}, {}
  logger.debug("Loading %d models", len(args.model_files))
  for m in args.model_files: models[m] = algorithm.read_model(m)
  logger.debug("Loading %d probes", len(args.probe_files))
  for p in args.probe_files: probes[p] = algorithm.read_probe(p)
  if algorithm.performs_projection:
    logger.debug("Projecting %d probes", len(args.probe_files))
    for p in probes: probes[p] = algorithm.project(probes[p])

  logger.info("Computing scores")
  for p in args.probe_files:
    for m in args.model_files:
      print("Score between model '%s' and probe '%s' is %3.8f" % (m, p, algorithm.score(models[m], probes[p])))
