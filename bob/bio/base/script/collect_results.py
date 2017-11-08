#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
# Tue Jul 2 14:52:49 CEST 2013

from __future__ import print_function

"""
This script parses through the given directory, collects all results of
verification experiments that are stored in file with the given file name.
It supports the split into development and test set of the data, as well as
ZT-normalized scores.

All result files are parsed and evaluated. For each directory, the following
information are given in columns:

  * The Equal Error Rate of the development set
  * The Equal Error Rate of the development set after ZT-Normalization
  * The Half Total Error Rate of the evaluation set
  * The Half Total Error Rate of the evaluation set after ZT-Normalization
  * The sub-directory where the scores can be found

The measure type of the development set can be changed to compute "HTER" or
"FAR" thresholds instead, using the --criterion option.
"""


import sys, os,  glob
import argparse
import numpy

import bob.measure
import bob.core
logger = bob.core.log.setup("bob.bio.base")

def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-d', '--devel-name', dest="dev", default="scores-dev", help = "Name of the file containing the development scores")
  parser.add_argument('-e', '--eval-name', dest="eval", default="scores-eval", help = "Name of the file containing the evaluation scores")
  parser.add_argument('-D', '--directory', default=".", help = "The directory where the results should be collected from; might include search patterns as '*'.")
  parser.add_argument('-r', '--rank', type=int, default=1, help = "The rank for which to compute RR and DIR")
  parser.add_argument('-f', '--far-threshold', type=float, default=0.001, help = "The FAR threshold to be used with criterion FAR and DIR")

  parser.add_argument('-n', '--nonorm-dir', dest="nonorm", default="nonorm", help = "Directory where the unnormalized scores are found")
  parser.add_argument('-z', '--ztnorm-dir', dest="ztnorm", default = "ztnorm", help = "Directory where the normalized scores are found")
  parser.add_argument('-s', '--sort', action='store_true', help = "Sort the results")
  parser.add_argument('-k', '--sort-key', dest='key', default = 'nonorm-dev', choices= ('nonorm-dev','nonorm-eval','ztnorm-dev','ztnorm-eval','dir'),
      help = "Sort the results according to the given key")
  parser.add_argument('-c', '--criterion', dest='criterion', default = 'EER', choices = ('EER', 'HTER', 'FAR', 'RR', 'DIR'),
      help = "Minimize the threshold on the development set according to the given criterion")

  parser.add_argument('-o', '--output', help = "Name of the output file that will contain the EER/HTER scores")

  parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)

  bob.core.log.add_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args

class Result:
  """Class for collecting the results of one experiment."""
  def __init__(self, dir, args):
    self.dir = dir
    self.m_args = args
    self.nonorm_dev = None
    self.nonorm_eval = None
    self.ztnorm_dev = None
    self.ztnorm_eval = None

  def _calculate(self, dev_file, eval_file = None):
    """Calculates the EER and HTER or FRR based on the threshold criterion."""
    if self.m_args.criterion in ("RR", "DIR"):
      scores_dev = bob.measure.load.cmc(dev_file)
      if eval_file is not None:
        scores_eval = bob.measure.load.cmc(eval_file)

      if self.m_args.criterion == "DIR":
        # get negatives without positives
        negatives = [max(neg) for neg, pos in scores_dev if (pos is None or not numpy.array(pos).size) and neg is not None]
        if not negatives:
          raise ValueError("There need to be at least one pair with only negative scores")
        threshold = bob.measure.far_threshold(negatives, [], self.m_args.far_threshold)
        DIR_dev = bob.measure.detection_identification_rate(scores_dev, threshold, self.m_args.rank)
        if eval_file is not None:
          # re-compute the threshold for eval file
          negatives = [max(neg) for neg, pos in scores_eval if (pos is None or not numpy.array(pos).size) and neg is not None]
          if not negatives:
            raise ValueError("There need to be at least one pair with only negative scores")
          threshold = bob.measure.far_threshold(negatives, [], self.m_args.far_threshold)
          DIR_eval = bob.measure.detection_identification_rate(scores_eval, threshold, self.m_args.rank)
        else:
          DIR_eval = None
        return (DIR_dev, DIR_eval)

      else:
        # Recognition Rate
        RR_dev = bob.measure.recognition_rate(scores_dev)
        RR_eval = None if eval_file is None else bob.measure.recognition_rate(scores_eval)
        return (RR_dev, RR_eval)

    else:

      dev_neg, dev_pos = bob.measure.load.split(dev_file)

      # switch which threshold function to use
      if self.m_args.criterion == 'EER':
        threshold = bob.measure.far_threshold(dev_neg, dev_pos)
      elif self.m_args.criterion == 'HTER':
        threshold = bob.measure.min_hter_threshold(dev_neg, dev_pos)
      elif self.m_args.criterion == 'FAR':
        threshold = bob.measure.far_threshold(dev_neg, dev_pos, self.m_args.far_threshold)
      else:
        raise NotImplementedError("Criterion %s is not yet implemented", self.m_args.criterion)

      # compute far and frr for the given threshold
      dev_far, dev_frr = bob.measure.farfrr(dev_neg, dev_pos, threshold)
      dev_hter = (dev_far + dev_frr)/2.0

      if eval_file:
        eval_neg, eval_pos = bob.measure.load.split(eval_file)
        eval_far, eval_frr = bob.measure.farfrr(eval_neg, eval_pos, threshold)
        eval_hter = (eval_far + eval_frr)/2.0
      else:
        eval_hter = None
        eval_frr = None

      if self.m_args.criterion == 'FAR':
        return (dev_frr, eval_frr)
      else:
        return (dev_hter, eval_hter)

  def nonorm(self, dev_file, eval_file = None):
    self.nonorm_dev, self.nonorm_eval = self._calculate(dev_file, eval_file)

  def ztnorm(self, dev_file, eval_file = None):
    self.ztnorm_dev, self.ztnorm_eval = self._calculate(dev_file, eval_file)

  def valid(self):
    return any(a is not None for a in [self.nonorm_dev, self.ztnorm_dev, self.nonorm_eval, self.ztnorm_eval])

  def __str__(self):
    str = ""
    for v in [self.nonorm_dev, self.ztnorm_dev, self.nonorm_eval, self.ztnorm_eval]:
      if v is not None:
        val = "% 2.3f%%"%(v*100)
      else:
        val = "None"
      cnt = 16-len(val)
      str += " "*cnt + val
    str += "        %s"%self.dir
    return str[5:]


def add_results(args, nonorm, ztnorm = None):
  """Adds results of the given nonorm and ztnorm directories."""
  r = Result(os.path.dirname(nonorm).replace(args.directory+"/", ""), args)
  logger.info("Adding results from directory '%s'", r.dir)

  # check if the results files are there
  dev_file = os.path.join(nonorm, args.dev)
  eval_file = os.path.join(nonorm, args.eval)
  if os.path.isfile(dev_file):
    if os.path.isfile(eval_file):
      r.nonorm(dev_file, eval_file)
    else:
      r.nonorm(dev_file)

  if ztnorm:
    dev_file = os.path.join(ztnorm, args.dev)
    eval_file = os.path.join(ztnorm, args.eval)
    if os.path.isfile(dev_file):
      if os.path.isfile(eval_file):
        r.ztnorm(dev_file, eval_file)
      else:
        r.ztnorm(dev_file)

  return r


def recurse(args, path):
  """Recurse the directory structure and collect all results that are stored in the desired file names."""
  dir_list = os.listdir(path)
  results = []

  # check if the score directories are included in the current path
  if args.nonorm in dir_list or args.nonorm == '.':
    if args.ztnorm in dir_list or args.ztnorm == '.':
      return results + [add_results(args, os.path.join(path, args.nonorm), os.path.join(path, args.ztnorm))]
    else:
      return results + [add_results(args, os.path.join(path, args.nonorm))]

  for e in dir_list:
    real_path = os.path.join(path, e)
    if os.path.isdir(real_path):
      r = recurse(args, real_path)
      if r is not None:
        results += r

  return results
  

def table(results):
  """Generates a table containing all results in a nice format."""
  A = " "*2 + 'dev  nonorm'+ " "*5 + 'dev  ztnorm' + " "*6 + 'eval nonorm' + " "*4 + 'eval ztnorm' + " "*12 + 'directory\n'
  A += "-"*100+"\n"
  for r in results:
    if r.valid():
      A += str(r) + "\n"
  return A


def main(command_line_parameters = None):
  """Iterates through the desired directory and collects all result files."""
  args = command_line_arguments(command_line_parameters)

  results = []
  # collect results
  directories = glob.glob(args.directory)
  for directory in directories:
    r = recurse(args, directory)
    if r is not None:
      results += r

  # sort results if desired
  if args.sort:
    import operator
    results.sort(key=operator.attrgetter(args.key.replace('-','_')))

  # print the results
  if args.self_test:
    table(results)
  elif args.output:
    f = open(args.output, "w")
    f.writelines(table(results))
    f.close()
  else:
    print (table(results))
