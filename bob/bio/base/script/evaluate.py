#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""This script evaluates the given score files and computes EER, HTER.
It also is able to plot CMC and ROC curves.
You can set the environment variable BOB_NO_STYLE_CHANGES to any value to avoid
this script from changing the matplotlib style values. """

from __future__ import print_function

# matplotlib stuff
import matplotlib
from matplotlib import pyplot
pyplot.switch_backend('pdf')  # switch to non-X backend
from matplotlib.backends.backend_pdf import PdfPages

# import bob.measure after matplotlib, so that it cannot define the backend

import argparse
import numpy
import math
import os

import bob.measure
from .. import score


if not os.environ.get('BOB_NO_STYLE_CHANGES'):
  # make the fig size smaller so that everything becomes bigger
  matplotlib.rc('figure', figsize=(4, 3))


import bob.core
logger = bob.core.log.setup("bob.bio.base")


def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-d', '--dev-files', required=True, nargs='+', help = "A list of score files of the development set.")
  parser.add_argument('-e', '--eval-files', nargs='+', help = "A list of score files of the evaluation set; if given it must be the same number of files as the --dev-files.")

  parser.add_argument('-s', '--directory', default = '.', help = "A directory, where to find the --dev-files and the --eval-files")

  parser.add_argument('-c', '--criterion', choices = ('EER', 'HTER', 'FAR'), help = "If given, the threshold of the development set will be computed with this criterion.")
  parser.add_argument('-f', '--far-value', type=float, default=0.001, help = "The FAR value for which to evaluate (only for --criterion FAR)")
  parser.add_argument('-x', '--cllr', action = 'store_true', help = "If given, Cllr and minCllr will be computed.")
  parser.add_argument('-m', '--mindcf', action = 'store_true', help = "If given, minDCF will be computed.")
  parser.add_argument('--cost', default=0.99,  help='Cost for FAR in minDCF')
  parser.add_argument('-r', '--rr', action = 'store_true', help = "If given, the Recognition Rate will be computed.")
  parser.add_argument('-o', '--rank', type=int, default=1, help = "The rank for which to plot the DIR curve")
  parser.add_argument('-t', '--thresholds', type=float, nargs='+', help = "If given, the Recognition Rate will incorporate an Open Set handling, rejecting all scores that are below the given threshold; when multiple thresholds are given, they are applied in the same order as the --dev-files.")
  parser.add_argument('-l', '--legends', nargs='+', help = "A list of legend strings used for ROC, CMC and DET plots; if given, must be the same number than --dev-files.")
  parser.add_argument('-F', '--legend-font-size', type=int, default=10, help = "Set the font size of the legends.")
  parser.add_argument('-P', '--legend-position', type=int, help = "Set the font size of the legends.")
  parser.add_argument('-T', '--title', nargs = '+', help = "Overwrite the default title of the plot for development (and evaluation) set")
  parser.add_argument('-R', '--roc', help = "If given, ROC curves will be plotted into the given pdf file.")
  parser.add_argument('-D', '--det', help = "If given, DET curves will be plotted into the given pdf file.")
  parser.add_argument('-C', '--cmc', help = "If given, CMC curves will be plotted into the given pdf file.")
  parser.add_argument('-O', '--dir', help = "If given, DIR curves will be plotted into the given pdf file; This is an open-set measure, which cannot be applied to closed set score files.")
  parser.add_argument('-E', '--epc', help = "If given, EPC curves will be plotted into the given pdf file. For this plot --eval-files is mandatory.")
  parser.add_argument('-M', '--min-far-value', type=float, default=1e-4, help = "Select the minimum FAR value used in ROC plots; should be a power of 10.")
  parser.add_argument('-L', '--far-line-at', type=float, help = "If given, draw a veritcal line at this FAR value in the ROC plots.")

  # add verbose option
  bob.core.log.add_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  # set verbosity level
  bob.core.log.set_verbosity_level(logger, args.verbose)

  # some sanity checks:
  for f in args.dev_files + (args.eval_files or []):
    real_file = os.path.join(args.directory, f)
    if not os.path.exists(real_file):
      raise ValueError("The provided score file '%s' does not exist" % real_file)

  if args.eval_files is not None and len(args.dev_files) != len(args.eval_files):
    logger.error("The number of --dev-files (%d) and --eval-files (%d) are not identical", len(args.dev_files), len(args.eval_files))

  # update legends when they are not specified on command line
  if args.legends is None:
    args.legends = [f.replace('_', '-') for f in args.dev_files]
    logger.warn("Legends are not specified; using legends estimated from --dev-files: %s", args.legends)

  # check that the legends have the same length as the dev-files
  if len(args.dev_files) != len(args.legends):
    logger.error("The number of --dev-files (%d) and --legends (%d) are not identical", len(args.dev_files), len(args.legends))

  if args.thresholds is not None:
    if len(args.thresholds) == 1:
      args.thresholds = args.thresholds * len(args.dev_files)
    elif len(args.thresholds) != len(args.dev_files):
      logger.error("If given, the number of --thresholds imust be either 1, or the same as --dev-files (%d), but it is %d", len(args.dev_files), len(args.thresholds))
  else:
    args.thresholds = [None] * len(args.dev_files)

  if args.title is not None:
    if args.eval_files is None and len(args.title) != 1:
      logger.warning("Ignoring the title for the evaluation set, as no evaluation set is given")
    if args.eval_files is not None and len(args.title) < 2:
      logger.error("The title for the evaluation set is not specified")

  return args

def _add_far_labels(min_far):
  # compute and apply tick marks
  assert min_far > 0
  ticks = [min_far]
  while ticks[-1] < 1.: ticks.append(ticks[-1] * 10.)
  pyplot.xticks(ticks)
  pyplot.axis([min_far, 1., -0.01, 1.01])



def _plot_roc(frrs, colors, labels, title, fontsize=10, position=None, farfrrs=None, min_far=None):
  if position is None: position = 'lower right'
  figure = pyplot.figure()

  # plot FAR and CAR for each algorithm
  for i in range(len(frrs)):
    pyplot.semilogx([f for f in frrs[i][0]], [1. - f for f in frrs[i][1]], color=colors[i], label=labels[i])
    if isinstance(farfrrs, list):
      pyplot.plot(farfrrs[i][0], (1.-farfrrs[i][1]), 'o', color=colors[i], markeredgecolor=colors[i])

  # plot vertical bar, if desired
  if farfrrs is not None:
    if isinstance(farfrrs, float):
      pyplot.plot([farfrrs,farfrrs],[0.,1.], "--", color='black')
    else:
      pyplot.plot([x[0] for x in farfrrs], [(1.-x[1]) for x in farfrrs], '--', color='black')

  _add_far_labels(min_far)

  # set label, legend and title
  pyplot.xlabel('FMR')
  pyplot.ylabel('1 - FNMR')
  pyplot.grid(True, color=(0.6,0.6,0.6))
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure


def _plot_det(dets, colors, labels, title, fontsize=10, position=None):
  if position is None: position = 'upper right'
  # open new page for current plot
  figure = pyplot.figure(figsize=(matplotlib.rcParams['figure.figsize'][0],
                                  matplotlib.rcParams['figure.figsize'][0] * 0.975))
  pyplot.grid(True)

  # plot the DET curves
  for i in range(len(dets)):
    pyplot.plot(dets[i][0], dets[i][1], color=colors[i], label=labels[i])

  # change axes accordingly
  det_list = [0.0002, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95]
  ticks = [bob.measure.ppndf(d) for d in det_list]
  labels = [("%.5f" % d).rstrip('0').rstrip('.') for d in det_list]
  pyplot.xticks(ticks, [l if i % 2 else "" for i,l in enumerate(labels)])
  pyplot.yticks(ticks, labels)
  pyplot.axis((ticks[0], ticks[-1], ticks[0], ticks[-1]))

  pyplot.xlabel('FMR')
  pyplot.ylabel('FNMR')
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure


def _plot_cmc(cmcs, colors, labels, title, fontsize=10, position=None):
  if position is None: position = 'lower right'
  # open new page for current plot
  figure = pyplot.figure()

  max_R = 0
  # plot the CMC curves
  for i in range(len(cmcs)):
    probs = bob.measure.cmc(cmcs[i])
    R = len(probs)
    pyplot.semilogx(range(1, R+1), probs, figure=figure, color=colors[i], label=labels[i])
    max_R = max(R, max_R)

  # change axes accordingly
  ticks = [int(t) for t in pyplot.xticks()[0]]
  pyplot.xlabel('Rank')
  pyplot.ylabel('Probability')
  pyplot.xticks(ticks, [str(t) for t in ticks])
  pyplot.axis([0, max_R, -0.01, 1.01])
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure


def _plot_dir(cmc_scores, far_values, rank, colors, labels, title, fontsize=10, position=None):
  if position is None: position = 'lower right'
  # open new page for current plot
  figure = pyplot.figure()

  # for each probe, for which no positives exists, get the highest negative
  # score; and sort them to compute the FAR thresholds
  for i, cmcs in enumerate(cmc_scores):
    negatives = sorted(max(neg) for neg, pos in cmcs if (pos is None or not numpy.array(pos).size) and neg is not None)
    if not negatives:
      raise ValueError("There need to be at least one pair with only negative scores")

    # compute thresholds based on FAR values
    thresholds = [bob.measure.far_threshold(negatives, [], v, True) for v in far_values]

    # compute detection and identification rate based on the thresholds for
    # the given rank
    rates = [bob.measure.detection_identification_rate(cmcs, t, rank) for t in thresholds]

    # plot DIR curve
    pyplot.semilogx(far_values, rates, figure=figure, color=colors[i], label=labels[i])

  # finalize plot
  _add_far_labels(far_values[0])

  pyplot.xlabel('FAR')
  pyplot.ylabel('DIR')
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure


def _plot_epc(scores_dev, scores_eval, colors, labels, title, fontsize=10, position=None):
  if position is None: position = 'upper center'
  # open new page for current plot
  figure = pyplot.figure()

  # plot the DET curves
  for i in range(len(scores_dev)):
    x,y = bob.measure.epc(scores_dev[i][0], scores_dev[i][1], scores_eval[i][0], scores_eval[i][1], 100)
    pyplot.plot(x, y, color=colors[i], label=labels[i])

  # change axes accordingly
  pyplot.xlabel('alpha')
  pyplot.ylabel('HTER')
  pyplot.title(title)
  pyplot.axis([-0.01, 1.01, -0.01, 0.51])
  pyplot.grid(True)
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure


def remove_nan(scores):
    """removes the NaNs from the scores"""
    nans = numpy.isnan(scores)
    sum_nans = sum(nans)
    total = len(scores)
    return scores[numpy.where(~nans)], sum_nans, total


def get_fta(scores):
    """calculates the Failure To Acquire (FtA) rate"""
    fta_sum, fta_total = 0, 0
    neg, sum_nans, total = remove_nan(scores[0])
    fta_sum += sum_nans
    fta_total += total
    pos, sum_nans, total = remove_nan(scores[1])
    fta_sum += sum_nans
    fta_total += total
    return (neg, pos, fta_sum * 100 / float(fta_total))


def main(command_line_parameters=None):
  """Reads score files, computes error measures and plots curves."""

  args = command_line_arguments(command_line_parameters)

  # get some colors for plotting
  if len(args.dev_files) > 10:
    cmap = pyplot.cm.get_cmap(name='magma')
    colors = [cmap(i) for i in numpy.linspace(0, 1.0, len(args.dev_files) + 1)]
  else:
    # matplotlib 2.0 default color cycler list: Vega category10 palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

  if args.criterion or args.roc or args.det or args.epc or args.cllr or args.mindcf:

    # First, read the score files
    logger.info("Loading %d score files of the development set", len(args.dev_files))
    scores_dev = [score.split(os.path.join(args.directory, f)) for f in args.dev_files]
    # remove nans
    scores_dev = [get_fta(s) for s in scores_dev]

    if args.eval_files:
      logger.info("Loading %d score files of the evaluation set", len(args.eval_files))
      scores_eval = [score.split(os.path.join(args.directory, f)) for f in args.eval_files]
      # remove nans
      scores_eval = [get_fta(s) for s in scores_eval]


    if args.criterion:
      logger.info("Computing %s on the development " % args.criterion + ("and HTER on the evaluation set" if args.eval_files else "set"))
      for i in range(len(scores_dev)):
        # compute threshold on development set
        if args.criterion == 'FAR':
          threshold = bob.measure.far_threshold(scores_dev[i][0], scores_dev[i][1], args.far_value/100.)
        else:
          threshold = {'EER': bob.measure.eer_threshold, 'HTER' : bob.measure.min_hter_threshold} [args.criterion](scores_dev[i][0], scores_dev[i][1])
        # apply threshold to development set
        far, frr = bob.measure.farfrr(scores_dev[i][0], scores_dev[i][1], threshold)
        if args.criterion == 'FAR':
          print("The FRR at FAR=%.1E of the development set of '%s' is %2.3f%% (CAR: %2.3f%%)" % (args.far_value, args.legends[i], frr * 100., 100.*(1-frr)))
        else:
          print("The %s of the development set of '%s' is %2.3f%%" % (args.criterion, args.legends[i], (far + frr) * 50.)) # / 2 * 100%
        if args.eval_files:
          # apply threshold to evaluation set
          far, frr = bob.measure.farfrr(scores_eval[i][0], scores_eval[i][1], threshold)
          if args.criterion == 'FAR':
            print("The FRR of the evaluation set of '%s' is %2.3f%% (CAR: %2.3f%%)" % (args.legends[i], frr * 100., 100.*(1-frr))) # / 2 * 100%
          else:
            print("The HTER of the evaluation set of '%s' is %2.3f%%" % (args.legends[i], (far + frr) * 50.)) # / 2 * 100%


    if args.mindcf:
      logger.info("Computing minDCF on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      for i in range(len(scores_dev)):
        # compute threshold on development set
        threshold = bob.measure.min_weighted_error_rate_threshold(scores_dev[i][0], scores_dev[i][1], args.cost)
        # apply threshold to development set
        far, frr = bob.measure.farfrr(scores_dev[i][0], scores_dev[i][1], threshold)
        print("The minDCF of the development set of '%s' is %2.3f%%" % (args.legends[i], (args.cost * far + (1-args.cost) * frr) * 100. ))
        if args.eval_files:
          # compute threshold on evaluation set
          threshold = bob.measure.min_weighted_error_rate_threshold(scores_eval[i][0], scores_eval[i][1], args.cost)
          # apply threshold to evaluation set
          far, frr = bob.measure.farfrr(scores_eval[i][0], scores_eval[i][1], threshold)
          print("The minDCF of the evaluation set of '%s' is %2.3f%%" % (args.legends[i], (args.cost * far + (1-args.cost) * frr) * 100. ))


    if args.cllr:
      logger.info("Computing Cllr and minCllr on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      for i in range(len(scores_dev)):
        cllr = bob.measure.calibration.cllr(scores_dev[i][0], scores_dev[i][1])
        min_cllr = bob.measure.calibration.min_cllr(scores_dev[i][0], scores_dev[i][1])
        print("Calibration performance on development set of '%s' is Cllr %1.5f and minCllr %1.5f " % (args.legends[i], cllr, min_cllr))
        if args.eval_files:
          cllr = bob.measure.calibration.cllr(scores_eval[i][0], scores_eval[i][1])
          min_cllr = bob.measure.calibration.min_cllr(scores_eval[i][0], scores_eval[i][1])
          print("Calibration performance on evaluation set of '%s' is Cllr %1.5f and minCllr %1.5f" % (args.legends[i], cllr, min_cllr))


    if args.roc:
      logger.info("Computing CAR curves on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      min_far = int(math.floor(math.log(args.min_far_value, 10)))
      fars = [math.pow(10., i * 0.25) for i in range(min_far * 4, 0)] + [1.]
      frrs_dev = [bob.measure.roc_for_far(scores[0], scores[1], fars) for scores in scores_dev]
      if args.eval_files:
        frrs_eval = [bob.measure.roc_for_far(scores[0], scores[1], fars) for scores in scores_eval]

      logger.info("Plotting ROC curves to file '%s'", args.roc)
      try:
        # create a multi-page PDF for the ROC curve
        pdf = PdfPages(args.roc)
        # create a separate figure for dev and eval
        pdf.savefig(_plot_roc(frrs_dev, colors, args.legends, args.title[0] if args.title is not None else "ROC for development set", args.legend_font_size, args.legend_position, args.far_line_at, min_far=args.min_far_value), bbox_inches='tight')
        del frrs_dev
        if args.eval_files:
          if args.far_line_at is not None:
            farfrrs = []
            for i in range(len(scores_dev)):
              threshold = bob.measure.far_threshold(scores_dev[i][0], scores_dev[i][1], args.far_line_at)
              farfrrs.append(bob.measure.farfrr(scores_eval[i][0], scores_eval[i][1], threshold))
          else:
            farfrrs = None
          pdf.savefig(_plot_roc(frrs_eval, colors, args.legends, args.title[1] if args.title is not None else "ROC for evaluation set", args.legend_font_size, args.legend_position, farfrrs, min_far=args.min_far_value), bbox_inches='tight')
          del frrs_eval
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of ROC curves, the following exception occured:\n%s" % e)

    if args.det:
      logger.info("Computing DET curves on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      dets_dev = [bob.measure.det(scores[0], scores[1], 1000) for scores in scores_dev]
      if args.eval_files:
        dets_eval = [bob.measure.det(scores[0], scores[1], 1000) for scores in scores_eval]

      logger.info("Plotting DET curves to file '%s'", args.det)
      try:
        # create a multi-page PDF for the DET curve
        pdf = PdfPages(args.det)
        # create a separate figure for dev and eval
        pdf.savefig(_plot_det(dets_dev, colors, args.legends, args.title[0] if args.title is not None else "DET for development set", args.legend_font_size, args.legend_position), bbox_inches='tight')
        del dets_dev
        if args.eval_files:
          pdf.savefig(_plot_det(dets_eval, colors, args.legends, args.title[1] if args.title is not None else "DET for evaluation set", args.legend_font_size, args.legend_position), bbox_inches='tight')
          del dets_eval
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of DET curves, the following exception occured:\n%s" % e)


    if args.epc:
      logger.info("Plotting EPC curves to file '%s'", args.epc)

      if not args.eval_files:
        raise ValueError("To plot the EPC curve the evaluation scores are necessary. Please, set it with the --eval-files option.")

      try:
        # create a multi-page PDF for the EPC curve
        pdf = PdfPages(args.epc)
        pdf.savefig(_plot_epc(scores_dev, scores_eval, colors, args.legends, args.title[0] if args.title is not None else "" , args.legend_font_size, args.legend_position), bbox_inches='tight')
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of EPC curves, the following exception occured:\n%s" % e)



  if args.cmc or args.rr or args.dir:
    logger.info("Loading CMC data on the development " + ("and on the evaluation set" if args.eval_files else "set"))
    cmcs_dev = [score.cmc(os.path.join(args.directory, f)) for f in args.dev_files]
    if args.eval_files:
      cmcs_eval = [score.cmc(os.path.join(args.directory, f)) for f in args.eval_files]

    if args.cmc:
      logger.info("Plotting CMC curves to file '%s'", args.cmc)
      try:
        # create a multi-page PDF for the CMC curve
        pdf = PdfPages(args.cmc)
        # create a separate figure for dev and eval
        pdf.savefig(_plot_cmc(cmcs_dev, colors, args.legends, args.title[0] if args.title is not None else "CMC curve for development set", args.legend_font_size, args.legend_position), bbox_inches='tight')
        if args.eval_files:
          pdf.savefig(_plot_cmc(cmcs_eval, colors, args.legends, args.title[1] if args.title is not None else "CMC curve for evaluation set", args.legend_font_size, args.legend_position), bbox_inches='tight')
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of CMC curves, the following exception occured:\n%s\nUsually this happens when the label contains characters that LaTeX cannot parse." % e)

    if args.rr:
      logger.info("Computing recognition rate on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      for i in range(len(cmcs_dev)):
        rr = bob.measure.recognition_rate(cmcs_dev[i], args.thresholds[i])
        print("The Recognition Rate of the development set of '%s' is %2.3f%%" % (args.legends[i], rr * 100.))
        if args.eval_files:
          rr = bob.measure.recognition_rate(cmcs_eval[i], args.thresholds[i])
          print("The Recognition Rate of the development set of '%s' is %2.3f%%" % (args.legends[i], rr * 100.))

    if args.dir:
      # compute false alarm values to evaluate
      min_far = int(math.floor(math.log(args.min_far_value, 10)))
      fars = [math.pow(10., i * 0.25) for i in range(min_far * 4, 0)] + [1.]
      logger.info("Plotting DIR curves to file '%s'", args.dir)
      try:
        # create a multi-page PDF for the DIR curve
        pdf = PdfPages(args.dir)
        # create a separate figure for dev and eval
        pdf.savefig(_plot_dir(cmcs_dev, fars, args.rank, colors, args.legends, args.title[0] if args.title is not None else "DIR curve for development set", args.legend_font_size, args.legend_position), bbox_inches='tight')
        if args.eval_files:
          pdf.savefig(_plot_dir(cmcs_eval, fars, args.rank, colors, args.legends, args.title[1] if args.title is not None else "DIR curve for evaluation set", args.legend_font_size, args.legend_position), bbox_inches='tight')
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of DIR curves, the following exception occured:\n%s" % e)
