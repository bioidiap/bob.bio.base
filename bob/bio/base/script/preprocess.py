"""This script can be used to preprocess a single data file with a given preprocessor.
"""

import argparse
import bob.core
logger = bob.core.log.setup("bob.bio.base")

import bob.bio.base
import bob.db.verification.utils
import numpy

import bob.core
import bob.io.base
import bob.io.image


def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-p', '--preprocessor', metavar = 'x', nargs = '+', required = True, help = 'Data preprocessing; registered preprocessors are: %s' % bob.bio.base.resource_keys('preprocessor'))
  parser.add_argument('-i', '--input-file', metavar = 'FILE', required = True, help = "The data file to be preprocessed.")
#  parser.add_argument('-a', '--annotations', nargs='+', help = "Key=value-pairs for the annotations")
  parser.add_argument('-a', '--annotation-file', metavar = 'FILE', help = "The annotation file for the given data file, if applicable and/or available; currently the only supported format is the 'named' annotation format.")
  parser.add_argument('-o', '--output-file', metavar = 'PREPROCESSED', default = 'preprocessed.hdf5', help = "Write the preprocessed data into this file (should be of type HDF5)")
  parser.add_argument('-c', '--convert-as-image', metavar = 'IMAGE', help = "Write the preprocessed data into this image file, converting it to an image, if possible")

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

  logger.debug("Loading input data from file '%s'%s", args.input_file, " and '%s'" % args.annotation_file if args.annotation_file is not None else "")
  data = preprocessor.read_original_data(args.input_file)
  annotations = bob.db.verification.utils.read_annotation_file(args.annotation_file, 'named') if args.annotation_file is not None else None

  logger.info("Preprocessing data")
  preprocessed = preprocessor(data, annotations)
  preprocessor.write_data(preprocessed, args.output_file)
  logger.info("Wrote preprocessed data to file '%s'", args.output_file)

  if args.convert_as_image is not None:
    converted = bob.core.convert(preprocessed, 'uint8', dest_range=(0,255), source_range=(numpy.min(preprocessed), numpy.max(preprocessed)))
    bob.io.base.save(converted, args.convert_as_image)
    logger.info("Wrote preprocessed data to image file '%s'", args.convert_as_image)
