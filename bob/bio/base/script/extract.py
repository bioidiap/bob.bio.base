"""This script can be used to extract features using the given extractor from the given preprocessed image.
"""

import argparse
import bob.core
logger = bob.core.log.setup("bob.bio.base")

import bob.bio.base


def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-e', '--extractor', metavar = 'x', nargs = '+', required = True, help = 'Feature extraction; registered feature extractors are: %s' % bob.bio.base.resource_keys('extractor'))
  parser.add_argument('-E', '--extractor-file', metavar = 'FILE', help = "The pre-trained extractor file, if the extractor requires training")
  parser.add_argument('-p', '--preprocessor', metavar = 'x', nargs = '+', required = True, help = 'Data preprocessing; registered preprocessors are: %s' % bob.bio.base.resource_keys('preprocessor'))
  parser.add_argument('-i', '--input-file', metavar = 'PREPROCESSED', required = True, help = "The preprocessed data file to read.")
  parser.add_argument('-o', '--output-file', metavar = 'FEATURE', default = 'extracted.hdf5', help = "The file to write the extracted features into (should be of type HDF5)")

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

  logger.debug("Loading preprocessor")
  preprocessor = bob.bio.base.load_resource(' '.join(args.preprocessor), "preprocessor")
  logger.debug("Loading extractor")
  extractor = bob.bio.base.load_resource(' '.join(args.extractor), "extractor")
  if extractor.requires_training:
    if args.extractor_file is None:
      raise ValueError("The desired extractor requires a pre-trained extractor file, but it was not specified")
    extractor.load(args.extractor_file)

  logger.debug("Loading preprocessed data from file '%s'", args.input_file)
  preprocessed = preprocessor.read_data(args.input_file)

  logger.info("Extracting features")
  extracted = extractor(preprocessed)
  extractor.write_feature(extracted, args.output_file)
  logger.info("Wrote extracted features to file '%s'", args.output_file)
