"""This script can be used to enroll a model from several features using the given algorithm.
"""

import argparse
import bob.core
logger = bob.core.log.setup("bob.bio.base")

import bob.bio.base


def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-a', '--algorithm', metavar = 'x', nargs = '+', required = True, help = 'Biometric recognition; registered algorithms are: %s' % bob.bio.base.resource_keys('algorithm'))
  parser.add_argument('-e', '--extractor', metavar = 'x', nargs = '+', required = True, help = 'Feature extraction; registered feature extractors are: %s' % bob.bio.base.resource_keys('extractor'))
  parser.add_argument('-P', '--projector-file', metavar = 'FILE', help = 'The pre-trained extractor file, if the algorithm performs projection')
  parser.add_argument('-E', '--enroller-file', metavar = 'FILE', help = 'The pre-trained enroller file, if the extractor requires enroller training')
  parser.add_argument('-i', '--input-files', metavar = 'FEATURE', nargs='+', required = True, help = "A list of feature files to enroll the model from")
  parser.add_argument('-o', '--output-file', metavar = 'MODEL', default = 'model.hdf5', help = "The file to write the enrolled model into (should be of type HDF5)")

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

  logger.debug("Loading extractor")
  extractor = bob.bio.base.load_resource(' '.join(args.extractor), "extractor")

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

  logger.debug("Loading %d features for enrollment", len(args.input_files))
  features = [extractor.read_feature(f) for f in args.input_files]
  if algorithm.use_projected_features_for_enrollment:
    logger.debug("Projecting enrollment features")
    features = [algorithm.project(f) for f in features]

  logger.debug("Enrolling model")
  model = algorithm.enroll(features)
  algorithm.write_model(model, args.output_file)
  logger.info("Wrote model to file '%s'", args.output_file)
