"""The click-based vulnerability analysis commands.
"""

import os
import csv
import numpy
import click
from click.types import FLOAT
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import (
    verbosity_option, bool_option, list_float_option
)
from bob.core import random, log
from bob.io.base import create_directories_safe
from bob.bio.base.score.load import split_csv_vuln
from . import vuln_figure as figure

logger = log.setup(__name__)
NUM_GENUINE_ACCESS = 5000
NUM_ZEIMPOSTORS = 5000
NUM_PA = 5000
NUM_NEG_PA = 5000


def fnmr_at_option(dflt=' ', **kwargs):
  '''Get option to draw const FNMR lines'''
  return list_float_option(
      name='fnmr', short_name='fnmr',
      desc='If given, draw horizontal lines at the given FNMR position. '
      'Your values must be separated with a comma (,) without space. '
      'This option works in ROC and DET curves.',
      nitems=None, dflt=dflt, **kwargs
  )


def gen_score_distr(mean_gen, mean_zei, mean_pa, mean_npa, sigma_gen=1,
                    sigma_zei=1, sigma_pa=1, sigma_npa=1):
  mt = random.mt19937()  # initialise the random number generator

  genuine_generator = random.normal(numpy.float32, mean_gen, sigma_gen)
  zei_generator = random.normal(numpy.float32, mean_zei, sigma_zei)
  pa_generator = random.normal(numpy.float32, mean_pa, sigma_pa)
  neg_pa_generator = random.normal(numpy.float32, mean_npa, sigma_npa)

  genuine_scores = [genuine_generator(mt) for i in range(NUM_GENUINE_ACCESS)]
  zei_scores = [zei_generator(mt) for i in range(NUM_ZEIMPOSTORS)]
  pa_scores = [pa_generator(mt) for i in range(NUM_PA)]
  neg_pa_scores = [neg_pa_generator(mt) for i in range(NUM_NEG_PA)]

  return genuine_scores, zei_scores, pa_scores, neg_pa_scores


def write_scores_to_file(neg_licit, pos_licit, neg_spoof, pos_spoof, filename):
  """Writes score distributions into a CSV score file. For the format of
    the score files, please refer to Bob's documentation.

  Parameters
  ----------
  neg : array_like
      Scores for negative samples.
  pos : array_like
      Scores for positive samples.
  filename : str
      The path to write the score to.
  """
  logger.info(f"Creating score file '{filename}'")
  create_directories_safe(os.path.dirname(filename))
  with open(filename, 'wt') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["bio_ref_subject_id", "probe_reference_id", "probe_attack_type", "score"])
    for score in neg_licit:
      csv_writer.writerow(["x", "y", "licit", score])
    for score in pos_licit:
      csv_writer.writerow(["x", "x", "licit", score])
    for score in neg_spoof:
      csv_writer.writerow(["x", "y", "pai", score])
    for score in pos_spoof:
      csv_writer.writerow(["x", "x", "pai", score])


@click.command()
@click.argument('outdir')
@click.option('-mg', '--mean-gen', default=7, type=FLOAT, show_default=True)
@click.option('-mz', '--mean-zei', default=3, type=FLOAT, show_default=True)
@click.option('-mp', '--mean-pa', default=5, type=FLOAT, show_default=True)
@click.option('-mnp', '--mean-npa', default=3, type=FLOAT, show_default=True)
@verbosity_option()
def gen(outdir, mean_gen, mean_zei, mean_pa, mean_npa, **kwargs):
  """Generate random scores.
  Generates random scores for three types of verification attempts:
  genuine users, zero-effort impostors and spoofing attacks and writes them
  into 4-column score files for so called licit and spoof scenario. The
  scores are generated using Gaussian distribution whose mean is an input
  parameter. The generated scores can be used as hypothetical datasets.
  """
  # Generate the data
  genuine_dev, zei_dev, pa_dev, npa_dev = gen_score_distr(
      mean_gen, mean_zei, mean_pa, mean_npa)
  genuine_eval, zei_eval, pa_eval, npa_eval = gen_score_distr(
      mean_gen, mean_zei, mean_pa, mean_npa)

  # Write the data into files
  write_scores_to_file(zei_dev, genuine_dev, npa_dev, pa_dev,
                       os.path.join(outdir, 'scores-dev.csv'))
  write_scores_to_file(zei_eval, genuine_eval, npa_eval, pa_eval,
                       os.path.join(outdir, 'scores-eval.csv'))


@click.command()
@common_options.scores_argument(min_arg=2, nargs=-1)
@common_options.output_plot_file_option(default_out='roc.pdf')
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_loc_option(dflt='upper-right')
@common_options.title_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.figsize_option(dflt=None)
@common_options.min_far_option()
@common_options.axes_val_option()
@verbosity_option()
@common_options.x_rotation_option(dflt=45)
@common_options.x_label_option()
@common_options.y_label_option()
@click.option('--real-data/--no-real-data', default=True, show_default=True,
              help='If False, will annotate the plots hypothetically, instead '
              'of with real data values of the calculated error rates.')
@fnmr_at_option()
@click.pass_context
def roc(ctx, scores, real_data, **kwargs):
  """Plot ROC

  You need to provide 2 scores
  files for each vulnerability system in this order:

  \b
  * licit scores
  * spoof scores

  Examples:
      $ bob vuln roc -v licit-scores spoof-scores

      $ bob vuln roc -v scores-{licit,spoof}
  """
  process = figure.RocVuln(ctx, scores, True, split_csv_vuln, real_data, False)
  process.run()


@click.command()
@common_options.scores_argument(min_arg=1, nargs=-1)
@common_options.output_plot_file_option(default_out='det.pdf')
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_loc_option(dflt='upper-right')
@common_options.title_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.figsize_option(dflt=None)
@verbosity_option()
@common_options.axes_val_option(dflt='0.01,95,0.01,95')
@common_options.x_rotation_option(dflt=45)
@common_options.x_label_option()
@common_options.y_label_option()
@click.option('--real-data/--no-real-data', default=True, show_default=True,
              help='If False, will annotate the plots hypothetically, instead '
              'of with real data values of the calculated error rates.')
@fnmr_at_option()
@click.pass_context
def det(ctx, scores, real_data, **kwargs):
  """Plot DET


  You need to provide 2 scores
  files for each vulnerability system in this order:

  \b
  * licit scores
  * spoof scores

  Examples:
      $ bob vuln det -v licit-scores spoof-scores

      $ bob vuln det -v scores-{licit,spoof}
  """
  process = figure.DetVuln(ctx, scores, True, split_csv_vuln, real_data, False)
  process.run()


@click.command()
@common_options.scores_argument(min_arg=1, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='epc.pdf')
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_loc_option()
@common_options.title_option()
@common_options.const_layout_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.figsize_option(dflt=None)
@common_options.style_option()
@common_options.bool_option(
    'iapmr', 'I', 'Whether to plot the IAPMR related lines or not.', True
)
@common_options.style_option()
@verbosity_option()
@click.pass_context
def epc(ctx, scores, **kwargs):
  """Plot EPC (expected performance curve):

  You need to provide 4 score
  files for each biometric system in this order:

  \b
  * licit development scores
  * licit evaluation scores
  * spoof development scores
  * spoof evaluation scores

  See :ref:`bob.pad.base.vulnerability` in the documentation for a guide on
  vulnerability analysis.

  Examples:
      $ bob vuln epc -v dev-scores eval-scores

      $ bob vuln epc -v -o my_epc.pdf dev-scores1 eval-scores1

      $ bob vuln epc -v {licit,spoof}/scores-{dev,eval}
  """
  process = figure.Epc(ctx, scores, True, split_csv_vuln)
  process.run()


@click.command()
@common_options.scores_argument(min_arg=1, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='epsc.pdf')
@common_options.titles_option()
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_ncols_option()
@common_options.const_layout_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.figsize_option(dflt='5,3')
@common_options.style_option()
@common_options.bool_option(
    'wer', 'w', 'Whether to plot the WER related lines or not.', True
)
@common_options.bool_option(
    'three-d', 'D', 'If true, generate 3D plots. You need to turn off '
    'wer or iapmr when using this option.', False
)
@common_options.bool_option(
    'iapmr', 'I', 'Whether to plot the IAPMR related lines or not.', True
)
@click.option('-c', '--criteria', default="eer", show_default=True,
              help='Criteria for threshold selection',
              type=click.Choice(('eer', 'min-hter')))
@click.option('-vp', '--var-param', default="omega", show_default=True,
              help='Name of the varying parameter',
              type=click.Choice(('omega', 'beta')))
@list_float_option(name='fixed-params', short_name='fp', dflt='0.5',
                   desc='Values of the fixed parameter, separated by commas')
@click.option('-s', '--sampling', default=5, show_default=True,
              help='Sampling of the EPSC 3D surface', type=click.INT)
@verbosity_option()
@click.pass_context
def epsc(ctx, scores, criteria, var_param, three_d, sampling,
         **kwargs):
  """Plot EPSC (expected performance spoofing curve):

  You need to provide 4 score
  files for each biometric system in this order:

  \b
  * licit development scores
  * licit evaluation scores
  * spoof development scores
  * spoof evaluation scores

  See :ref:`bob.pad.base.vulnerability` in the documentation for a guide on
  vulnerability analysis.

  Note that when using 3D plots with option ``--three-d``, you cannot plot
  both WER and IAPMR on the same figure (which is possible in 2D).

  Examples:
      $ bob vuln epsc -v -o my_epsc.pdf dev-scores1 eval-scores1

      $ bob vuln epsc -v -D {licit,spoof}/scores-{dev,eval}
  """
  fixed_params = ctx.meta.get('fixed_params', [0.5])
  if three_d:
    if (ctx.meta['wer'] and ctx.meta['iapmr']):
      logger.info('Cannot plot both WER and IAPMR in 3D. Will turn IAPMR off.')
      ctx.meta['iapmr'] = False
    ctx.meta['sampling'] = sampling
    process = figure.Epsc3D(
        ctx, scores, True, split_csv_vuln,
        criteria, var_param, fixed_params
    )
  else:
    process = figure.Epsc(
        ctx, scores, True, split_csv_vuln,
        criteria, var_param, fixed_params
    )
  process.run()


@click.command()
@common_options.scores_argument(nargs=-1, min_arg=1)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.thresholds_option()
@common_options.print_filenames_option(dflt=False)
@bool_option(
    'iapmr-line', 'I', 'Whether to plot the IAPMR related lines or not.', True
)
@bool_option(
    'real-data', 'R',
    'If False, will annotate the plots hypothetically, instead '
    'of with real data values of the calculated error rates.', True
)
@common_options.titles_option()
@common_options.const_layout_option()
@common_options.figsize_option(dflt=None)
@common_options.subplot_option()
@common_options.legend_ncols_option()
@common_options.style_option()
@common_options.hide_dev_option()
@common_options.eval_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, evaluation, **kwargs):
  '''Vulnerability analysis distributions.

  Plots the histogram of score distributions. You need to provide 2 or 4 score
  files for each biometric system in this order.
  When evaluation scores are provided, you must use the ``--eval`` option.

  \b
  * licit development scores
  * (optional) licit evaluation scores
  * spoof development scores
  * (optional) spoof evaluation scores

  See :ref:`bob.pad.base.vulnerability` in the documentation for a guide on
  vulnerability analysis.


  By default, when eval-scores are given, only eval-scores histograms are
  displayed with threshold line
  computed from dev-scores.

  Examples:

      $ bob vuln hist -v results/scores-dev.csv

      $ bob vuln hist -e -v results/scores-dev.csv results/scores-eval.csv

      $ bob vuln hist -e -v results/scores-{dev,eval}.csv
  '''
  process = figure.HistVuln(ctx, scores, evaluation, split_csv_vuln)
  process.run()


@click.command()
@common_options.scores_argument(min_arg=1, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='fmr_iapmr.pdf')
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_loc_option()
@common_options.title_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.figsize_option()
@verbosity_option()
@common_options.axes_val_option()
@common_options.x_rotation_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.semilogx_option()
@click.pass_context
def fmr_iapmr(ctx, scores, **kwargs):
  """Plot FMR vs IAPMR

  You need to provide 4 scores
  files for each vuln system in this order:

  \b
  * licit development scores
  * licit evaluation scores
  * spoof development scores
  * spoof evaluation scores

  Examples:
      $ bob vuln fmr_iapmr -v dev-scores eval-scores

      $ bob vuln fmr_iapmr -v {licit,spoof}/scores-{dev,eval}
  """
  process = figure.FmrIapmr(ctx, scores, True, split_csv_vuln)
  process.run()
