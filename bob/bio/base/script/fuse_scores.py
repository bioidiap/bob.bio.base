#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
# Elie El Khoury <elie.khoury@idiap.ch>
# Manuel Guenther <siebenkopf@googlemail.com>
# Mon 13 Jul 11:55:34 CEST 2015
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""This script fuses scores from various systems, from a score file in four or five column format.

Note: The score file has to contain the exact probe file names as the 3rd (4column) or 4th (5column) column.
The resulting fused score files will be written in 4 column format.
"""



import bob, os, sys
import bob.learn.linear

import bob.core
logger = bob.core.log.setup("bob.bio.base")

def parse_command_line(command_line_options):
  """Parse the program options"""

  usage = 'usage: %s [arguments]' % os.path.basename(sys.argv[0])

  import argparse
  parser = argparse.ArgumentParser(usage=usage, description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # This option is not normally shown to the user...
  parser.add_argument('-d', '--dev-files', required=True, nargs='+', help = "A list of score files of the development set.")
  parser.add_argument('-e', '--eval-files', nargs='+', help = "A list of score files of the evaluation set; if given it must be the same number of files as the --dev-files.")
  parser.add_argument('-f', '--fused-dev-file', required = True, help = 'The fused development score file in 4 column format.')
  parser.add_argument('-g', '--fused-eval-file', help = 'The fused evaluation score file in 4 column format.')
  parser.add_argument('-p', '--parser', default = '4column', choices = ('4column', '5column'),  help = "The style of the resulting score files. The default fits to the usual output of score files.")

  parser.add_argument('-m', '--max-iterations', type=int, default=10000, help = "Select the maximum number of iterations for the LLR training")
  parser.add_argument('-t', '--convergence-threshold', type=float, default=1e-10, help = "Select the convergence threshold for the LLR training")
  parser.add_argument('-n', '--no-whitening', action="store_true", help = "If given, disable the score mean/std-normalization prior to fusion (this is not recommended)")

  # enable logging
  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_options)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  if args.eval_files is not None and len(args.eval_files) != len(args.dev_files):
    raise ValueError("When --eval-files are specified, there need to be exactly one eval file for each dev file")

  if args.eval_files is not None and args.fused_eval_file is None:
    raise ValueError("When --eval-files are specified, the --fused-eval-file needs to be given, too")

  return args


def main(command_line_options = None):
  """Score Fusion using Logistic regression"""
  args = parse_command_line(command_line_options)

  # read data
  n_systems = len(args.dev_files)
  for i in range(n_systems):
    if not os.path.isfile(args.dev_files[i]): raise IOError("The given score file does not exist")

  # collect training data from development sets
  data = []
  for i in range(n_systems):
    logger.info("Loading development set score file '%s'", args.dev_files[i])
    # pythonic way: create inline dictionary "{...}", index with desired value "[...]", execute function "(...)"
    data.append({'4column' : bob.measure.load.split_four_column, '5column' : bob.measure.load.split_five_column}[args.parser](args.dev_files[i]))
  import numpy

  trainer = bob.learn.linear.CGLogRegTrainer(0.5, args.convergence_threshold, args.max_iterations, mean_std_norm=not args.no_whitening)
  data_neg = numpy.vstack([data[k][0] for k in range(n_systems)]).T
  data_pos = numpy.vstack([data[k][1] for k in range(n_systems)]).T
  machine = trainer.train(data_neg, data_pos)

  # fuse development scores
  gen_data_dev = []
  for i in range(n_systems):
    logger.info("Loading development set score file '%s'", args.dev_files[i])
    gen_data_dev.append({'4column' : bob.measure.load.four_column, '5column' : bob.measure.load.five_column}[args.parser](args.dev_files[i]))

  logger.info("Writing fused development set score file '%s'", args.fused_dev_file)
  outf = open(args.fused_dev_file, 'w')
  for line in gen_data_dev[0]:
    claimed_id = line[0]
    real_id = line[-3]
    test_label = line[-2]
    scores= [ line[-1] ]
    for n in range(1, n_systems):
      scores.append(next(gen_data_dev[n])[-1])
    scores = numpy.array([scores], dtype=numpy.float64)
    s_fused = machine.forward(scores)[0,0]
    line = claimed_id + " " + real_id + " " + test_label + " "  + str(s_fused) + "\n"
    outf.write(line)

  # fuse evaluation scores
  if args.eval_files is not None:
    gen_data_eval = []
    for i in range(n_systems):
      logger.info("Loading evaluation set score file '%s'", args.eval_files[i])
      gen_data_eval.append({'4column' : bob.measure.load.four_column, '5column' : bob.measure.load.five_column}[args.parser](args.eval_files[i]))

    logger.info("Writing fused evaluation set score file '%s'", args.fused_eval_file)
    outf = open(args.fused_eval_file, 'w')
    for line in gen_data_eval[0]:
      claimed_id = line[0]
      real_id = line[-3]
      test_label = line[-2]
      scores= [ line[-1] ]
      for n in range(1, n_systems):
        scores.append(next(gen_data_eval[n])[-1])
      scores = numpy.array([scores], dtype=numpy.float64)
      s_fused = machine.forward(scores)[0,0]
      line = claimed_id + " " + real_id + " " + test_label + " "  + str(s_fused) + "\n"
      outf.write(line)
