''' Click commands for ``bob.bio.base`` '''

import click
import bob.bio.base.script.figure as bio_figure
import bob.measure.script.figure as measure_figure
from ..score import load
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import (verbosity_option,
                                                open_file_mode_option)


def rank_option(**kwargs):
    '''Get option for rank parameter'''
    def custom_rank_option(func):
        def callback(ctx, param, value):
            value = 1 if value < 0 else value
            ctx.meta['rank'] = value
            return value
        return click.option(
            '-rk', '--rank', type=click.INT, default=1,
            help='Provide rank for the command',
            callback=callback, show_default=True, **kwargs)(func)
    return custom_rank_option

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.table_option()
@common_options.eval_option()
@common_options.output_log_metric_option()
@common_options.criterion_option(['eer', 'min-hter', 'far', 'mindcf', 'cllr', 'rr'])
@common_options.cost_option()
@common_options.thresholds_option()
@common_options.far_option()
@common_options.legends_option()
@open_file_mode_option()
@verbosity_option()
@click.pass_context
def metrics(ctx, scores, evaluation, **kargs):
    """Prints a single output line that contains all info for a given
    criterion (eer, min-hter, far, mindcf, cllr, rr).

    You need to provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev-scores
    are used, the flag `--no-evaluation` must be used.
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    Resulting table format can be changer using the `--tablefmt`. Default
    formats are `rst` when output in the terminal and `latex` when
    written in a log file (see `--log`)

    Examples:
        $ bob bio metrics dev-scores

        $ bob bio metrics --no-evaluation dev-scores1 dev-scores2

        $ bob bio metrics -l results.txt dev-scores1 eval-scores1

        $ bob bio metrics {dev,eval}-scores1 {dev,eval}-scores2
    """
    if 'criterion' in ctx.meta and ctx.meta['criterion'] == 'rr':
        process = bio_figure.Metrics(ctx, scores, evaluation, load.cmc)
    else:
        process = bio_figure.Metrics(ctx, scores, evaluation, load.split)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.title_option()
@common_options.legends_option()
@common_options.sep_dev_eval_option()
@common_options.output_plot_file_option(default_out='roc.pdf')
@common_options.eval_option()
@common_options.points_curve_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=[1e-4, 1, 1e-4, 1])
@common_options.x_rotation_option()
@common_options.lines_at_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option()
@common_options.min_far_option()
@verbosity_option()
@click.pass_context
def roc(ctx, scores, evaluation, **kargs):
    """Plot ROC (receiver operating characteristic) curve:
    The plot will represent the false match rate on the horizontal axis and the
    false non match rate on the vertical axis.  The values for the axis will be
    computed using :py:func:`bob.measure.roc`.

    You need to provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev-scores
    are used, the flag `--no-evaluation` must be used.
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    Examples:
        $ bob bio roc dev-scores

        $ bob bio roc dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob bio roc -o my_roc.pdf dev-scores1 eval-scores1
    """
    process = bio_figure.Roc(ctx, scores, evaluation, load.split)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.title_option()
@common_options.output_plot_file_option(default_out='det.pdf')
@common_options.legends_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.sep_dev_eval_option()
@common_options.eval_option()
@common_options.axes_val_option(dflt=[0.01, 95, 0.01, 95])
@common_options.x_rotation_option(dflt=45)
@common_options.points_curve_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option()
@common_options.lines_at_option()
@common_options.min_far_option()
@verbosity_option()
@click.pass_context
def det(ctx, scores, evaluation, **kargs):
    """Plot DET (detection error trade-off) curve:
    modified ROC curve which plots error rates on both axes
    (false positives on the x-axis and false negatives on the y-axis)

    You need to provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev-scores
    are used, the flag `--no-evaluation` must be used.
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    Examples:
        $ bob bio det dev-scores

        $ bob bio det dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob bio det -o my_det.pdf dev-scores1 eval-scores1
    """
    process = bio_figure.Det(ctx, scores, evaluation, load.split)
    process.run()

@click.command()
@common_options.scores_argument(min_arg=1, force_eval=True, nargs=-1)
@common_options.title_option()
@common_options.output_plot_file_option(default_out='epc.pdf')
@common_options.legends_option()
@common_options.points_curve_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option()
@verbosity_option()
@click.pass_context
def epc(ctx, scores, **kargs):
    """Plot EPC (expected performance curve):
    plots the error rate on the eval set depending on a threshold selected
    a-priori on the development set and accounts for varying relative cost
    in [0; 1] of FPR and FNR when calculating the threshold.

    You need to provide one or more development score and eval file(s)
    for each experiment. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    Examples:
        $ bob bio epc dev-scores eval-scores

        $ bob bio epc -o my_epc.pdf dev-scores1 eval-scores1
    """
    process = measure_figure.Epc(ctx, scores, True, load.split)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.title_option()
@common_options.legends_option()
@common_options.sep_dev_eval_option()
@common_options.output_plot_file_option(default_out='cmc.pdf')
@common_options.eval_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=None)
@common_options.x_rotation_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option()
@verbosity_option()
@click.pass_context
def cmc(ctx, scores, evaluation, **kargs):
    """Plot CMC (cumulative match characteristic curve):
    graphical presentation of results of an identification task eval,
    plotting rank values on the x-axis and the probability of correct identification
    at or below that rank on the y-axis. The values for the axis will be
    computed using :py:func:`bob.measure.cmc`.

    You need to provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev-scores
    are used, the flag `--no-evaluation` must be used.
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.


    Examples:
        $ bob bio cmc dev-scores

        $ bob bio cmc dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob bio cmc -o my_roc.pdf dev-scores1 eval-scores1
    """
    process = bio_figure.Cmc(ctx, scores, evaluation, load.cmc)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.title_option()
@common_options.legends_option()
@common_options.sep_dev_eval_option()
@common_options.output_plot_file_option(default_out='cmc.pdf')
@common_options.eval_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=None)
@common_options.x_rotation_option()
@rank_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option()
@verbosity_option()
@click.pass_context
def dir(ctx, scores, evaluation, **kargs):
    """Plots the Detection & Identification Rate curve over the FAR

    This curve is designed to be used in an open set identification protocol, and
    defined in Chapter 14.1 of [LiJain2005]_.  It requires to have at least one
    open set probe item, i.e., with no corresponding gallery, such that the
    positives for that pair are ``None``.

    The detection and identification curve first computes FAR thresholds based on
    the out-of-set probe scores (negative scores).  For each probe item, the
    **maximum** negative score is used.  Then, it plots the detection and
    identification rates for those thresholds, which are based on the in-set
    probe scores only. See [LiJain2005]_ for more details.

    .. [LiJain2005] **Stan Li and Anil K. Jain**, *Handbook of Face Recognition*, Springer, 2005

    You need to provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev-scores
    are used, the flag `--no-evaluation` must be used.
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    Examples:
        $ bob bio dir dev-scores

        $ bob bio dir dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob bio dir -o my_roc.pdf dev-scores1 eval-scores1
    """
    process = bio_figure.Dir(ctx, scores, evaluation, load.cmc)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.title_option()
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.eval_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.thresholds_option()
@common_options.const_layout_option()
@common_options.print_filenames_option()
@common_options.legends_option()
@common_options.style_option()
@common_options.figsize_option(dflt=None)
@common_options.subplot_option()
@common_options.legend_ncols_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, evaluation, **kwargs):
    """ Plots histograms of positive and negatives along with threshold
    criterion.

    You need to provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev-scores
    are used, the flag `--no-evaluation` must be used.
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    By default, when eval-scores are given, only eval-scores histograms are
    displayed with threshold line
    computed from dev-scores. If you want to display dev-scores distributions
    as well, use ``--show-dev`` option.

    Examples:
        $ bob bio hist dev-scores

        $ bob bio hist dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob bio hist --criterion --show-dev min-hter dev-scores1 eval-scores1
    """
    process = bio_figure.Hist(ctx, scores, evaluation, load.split)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.legends_option()
@common_options.sep_dev_eval_option()
@common_options.table_option()
@common_options.eval_option()
@common_options.output_log_metric_option()
@common_options.output_plot_file_option(default_out='eval_plots.pdf')
@common_options.points_curve_option()
@common_options.lines_at_option()
@common_options.cost_option()
@rank_option()
@common_options.far_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.figsize_option()
@verbosity_option()
@click.pass_context
def evaluate(ctx, scores, evaluation, **kwargs):
    '''Evalutes score file, runs error analysis on score sets and plot curves.

    \b
    1. Computes the threshold using either EER, min. HTER or FAR value
       criteria on development set scores
    2. Applies the above threshold on eval set scores to compute the HTER, if a
       eval-score set is provided
    3. Computes Cllr and minCllr and minDCF
    3. Reports error metrics in the console or in a log file
    4. Plots ROC, EPC, DET, score distributions
       curves to a multi-page PDF file

    You need to provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev-scores
    are used, the flag `--no-evaluation` must be used.
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    You need to provide 2 score files for each biometric system in this order:

    \b
    * development scores
    * evaluation scores

    Examples:
        $ bob bio evaluate dev-scores

        $ bob bio evaluate -l metrics.txt -o my_plots.pdf dev-scores eval-scores

        $ bob bio evaluate -o my_plots.pdf /path/to/syst-{1,2,3}/{dev,eval}-scores
    '''
    log_str = ''
    if 'log' in ctx.meta and ctx.meta['log'] is not None:
        log_str = ' %s' % ctx.meta['log']

    # first time erase if existing file
    ctx.meta['open_mode'] = 'w'
    click.echo("Computing metrics with EER%s..." % log_str)
    ctx.meta['criterion'] = 'eer'  # no criterion passed to evaluate
    ctx.invoke(metrics, scores=scores, evaluation=evaluation)
    # other times, appends the content
    ctx.meta['open_mode'] = 'a'
    click.echo("Computing metrics with min-HTER%s..." % log_str)
    ctx.meta['criterion'] = 'min-hter'  # no criterion passed in evaluate
    ctx.invoke(metrics, scores=scores, evaluation=evaluation)
    if 'far_value' in ctx.meta and ctx.meta['far_value'] is not None:
        click.echo("Computing metrics with FAR=%f%s..." %\
        (ctx.meta['far_value'], log_str))
        ctx.meta['criterion'] = 'far'  # no criterio % n passed in evaluate
        ctx.invoke(metrics, scores=scores, evaluation=evaluation)

    click.echo("Computing minDCF%s..." % log_str)
    ctx.meta['criterion'] = 'mindcf'  # no criterion passed in evaluate
    ctx.invoke(metrics, scores=scores, evaluation=evaluation)

    click.echo("Computing  Cllr and minCllr%s..." % log_str)
    ctx.meta['criterion'] = 'cllr'  # no criterion passed in evaluate
    ctx.invoke(metrics, scores=scores, evaluation=evaluation)

    # avoid closing pdf file before all figures are plotted
    ctx.meta['closef'] = False

    if evaluation:
        click.echo("Starting evaluate with dev and eval scores...")
    else:
        click.echo("Starting evaluate with dev scores only...")

    click.echo("Generating ROC in %s..." % ctx.meta['output'])
    ctx.forward(roc) # use class defaults plot settings

    click.echo("Generating DET in %s..." % ctx.meta['output'])
    ctx.forward(det) # use class defaults plot settings

    if evaluation:
        click.echo("Generating EPC in %s..." % ctx.meta['output'])
        ctx.forward(epc) # use class defaults plot settings

    # the last one closes the file
    ctx.meta['closef'] = True
    click.echo("Generating score histograms in %s..." % ctx.meta['output'])
    ctx.meta['criterion'] = 'eer'  # no criterion passed in evaluate
    ctx.forward(hist)

    click.echo("Evaluate successfully completed!")
