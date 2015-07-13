#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
# Elie El Khoury <elie.khoury@idiap.ch>
#Mon 13 Jul 11:55:34 CEST 2015
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

"""This script fuses scores from various systems,
from a score file in four or five column format.

Note: The score file has to contain the exact probe file names as the 3rd (4column) or 4th (5column) column.
"""



import bob, os, sys
import bob.learn.linear

def parse_command_line(command_line_options):
  """Parse the program options"""

  usage = 'usage: %s [arguments]' % os.path.basename(sys.argv[0])

  import argparse
  parser = argparse.ArgumentParser(usage=usage, description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # This option is not normally shown to the user...
  parser.add_argument('--self-test', action = 'store_true', help = argparse.SUPPRESS)
  parser.add_argument('-d', '--dev-files', required=True, nargs='+', help = "A list of score files of the development set.")
  parser.add_argument('-e', '--eval-files', nargs='+', help = "A list of score files of the evaluation set; if given it must be the same number of files as the --dev-files.")
  parser.add_argument('-f', '--score-fused-dev-file', required = True, help = 'The calibrated development score file in 4 or 5 column format to calibrate.')
  parser.add_argument('-g', '--score-fused-eval-file', help = 'The calibrated evaluation score file in 4 or 5 column format to calibrate.')
  parser.add_argument('-p', '--parser', default = '4column', choices = ('4column', '5column'),  help="The style of the resulting score files. The default fits to the usual output of score files.")

  args = parser.parse_args(command_line_options)

  return args

def main(command_line_options = None):
  """Score Fusion using logistic regresssion"""
  args = parse_command_line(command_line_options)

  # read data
  n_systems = len(args.dev_files)
  for i in range(n_systems):
    if not os.path.isfile(args.dev_files[i]): raise IOError("The given score file does not exist")
  # pythonic way: create inline dictionary "{...}", index with desired value "[...]", execute function "(...)"
  data = []
  for i in range(n_systems):
    data.append({'4column' : bob.measure.load.split_four_column, '5column' : bob.measure.load.split_five_column}[args.parser](args.dev_files[i]))
  import numpy

  data_neg = numpy.vstack([data[k][0] for k in range(n_systems)]).T.copy()
  data_pos = numpy.vstack([data[k][1] for k in range(n_systems)]).T.copy()
  trainer = bob.learn.linear.CGLogRegTrainer(0.5, 1e-10, 10000)
  machine = trainer.train(data_neg, data_pos)

  # fuse development scores
  gen_data_dev = []
  for i in range(n_systems):
    gen_data_dev.append({'4column' : bob.measure.load.four_column, '5column' : bob.measure.load.five_column}[args.parser](args.dev_files[i]))

  outf = open(args.score_fused_dev_file, 'w')
  for line in gen_data_dev[0]:
    claimed_id = line[0]
    real_id = line[-3]
    test_label = line[-2]
    scores= [ line[-1] ]
    for n in range(1, n_systems): 
      scores.append(gen_data_dev[n].next()[-1])
    scores = numpy.array([scores], dtype=numpy.float64)
    s_fused = machine.forward(scores)[0,0]  
    line = claimed_id + " " + real_id + " " + test_label + " "  + str(s_fused) + "\n"
    outf.write(line)

  # fuse evaluation scores
  if args.eval_files is not None:
    if len(args.dev_files) != len(args.eval_files):
      logger.error("The number of --dev-files (%d) and --eval-files (%d) are not identical", len(args.dev_files), len(args.eval_files))
    
    gen_data_eval = []
    for i in range(n_systems):
      gen_data_eval.append({'4column' : bob.measure.load.four_column, '5column' : bob.measure.load.five_column}[args.parser](args.eval_files[i]))
      
    outf = open(args.score_fused_eval_file, 'w')
    for line in gen_data_eval[0]:
      claimed_id = line[0]
      real_id = line[-3]
      test_label = line[-2]
      scores= [ line[-1] ] 
      for n in range(1, n_systems): 
        scores.append(gen_data_eval[n].next()[-1])
      scores = numpy.array([scores], dtype=numpy.float64)
      s_fused = machine.forward(scores)[0,0]  
      line = claimed_id + " " + real_id + " " + test_label + " "  + str(s_fused) + "\n"
      outf.write(line)
  return 0

if __name__ == '__main__':
  main(sys.argv[1:])
