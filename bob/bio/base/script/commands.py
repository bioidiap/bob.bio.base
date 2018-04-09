''' Click commands for ``bob.bio.base`` '''


import click
import bob.bio.base.script.figure as bio_figure
import bob.measure.script.figure as measure_figure
from ..score import load
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import verbosity_option

FUNC_SPLIT = lambda x: load.load_files(x, load.split)
FUNC_CMC = lambda x: load.load_files(x, load.cmc)

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.table_option()
@common_options.test_option()
@common_options.open_file_mode_option()
@common_options.output_plot_metric_option()
@common_options.criterion_option(['eer', 'hter', 'far', 'mindcf', 'cllr', 'rr'])
@common_options.cost_option()
@common_options.thresholds_option()
@common_options.far_option()
@verbosity_option()
@click.pass_context
def metrics(ctx, scores, test, **kargs):
    """Prints a single output line that contains all info for a given
    criterion (eer or hter).

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    Resulting table format can be changer using the `--tablefmt`. Default
    formats are `rst` when output in the terminal and `latex` when
    written in a log file (see `--log`)

    Examples:
        $ bob bio metrics dev-scores

        $ bob bio metrics --test -l results.txt dev-scores1 test-scores1

        $ bob bio metrics --test {dev,test}-scores1 {dev,test}-scores2
    """
    if 'criter' in ctx.meta and ctx.meta['criter'] == 'rr':
        process = bio_figure.Metrics(ctx, scores, test, FUNC_CMC)
    else:
        process = bio_figure.Metrics(ctx, scores, test, FUNC_SPLIT)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.sep_dev_test_option()
@common_options.output_plot_file_option(default_out='roc.pdf')
@common_options.test_option()
@common_options.points_curve_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=[1e-4, 1, 1e-4, 1])
@common_options.axis_fontsize_option()
@common_options.x_rotation_option()
@common_options.fmr_line_at_option()
@verbosity_option()
@click.pass_context
def roc(ctx, scores, test, **kargs):
    """Plot ROC (receiver operating characteristic) curve:
    The plot will represent the false match rate on the horizontal axis and the
    false non match rate on the vertical axis.  The values for the axis will be
    computed using :py:func:`bob.measure.roc`.

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.


    Examples:
        $ bob bio roc dev-scores

        $ bob bio roc --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob bio roc --test -o my_roc.pdf dev-scores1 test-scores1
    """
    process = measure_figure.Roc(ctx, scores, test, FUNC_SPLIT)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='det.pdf')
@common_options.titles_option()
@common_options.sep_dev_test_option()
@common_options.test_option()
@common_options.axis_fontsize_option(dflt=6)
@common_options.axes_val_option(dflt=[0.01, 95, 0.01, 95])
@common_options.x_rotation_option(dflt=45)
@common_options.points_curve_option()
@verbosity_option()
@click.pass_context
def det(ctx, scores, test, **kargs):
    """Plot DET (detection error trade-off) curve:
    modified ROC curve which plots error rates on both axes
    (false positives on the x-axis and false negatives on the y-axis)

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.


    Examples:
        $ bob bio det dev-scores

        $ bob bio det --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob bio det --test -o my_det.pdf dev-scores1 test-scores1
    """
    process = measure_figure.Det(ctx, scores, test, FUNC_SPLIT)
    process.run()

@click.command()
@common_options.scores_argument(test_mandatory=True, nargs=-1)
@common_options.output_plot_file_option(default_out='epc.pdf')
@common_options.titles_option()
@common_options.points_curve_option()
@common_options.axis_fontsize_option()
@verbosity_option()
@click.pass_context
def epc(ctx, scores, **kargs):
    """Plot EPC (expected performance curve):
    plots the error rate on the test set depending on a threshold selected
    a-priori on the development set and accounts for varying relative cost
    in [0; 1] of FPR and FNR when calculating the threshold.

    You need provide one or more development score and test file(s)
    for each experiment. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    Examples:
        $ bob bio epc dev-scores test-scores

        $ bob bio epc -o my_epc.pdf dev-scores1 test-scores1
    """
    process = measure_figure.Epc(ctx, scores, True, FUNC_SPLIT)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.test_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.axis_fontsize_option()
@common_options.thresholds_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, test, **kwargs):
    """ Plots histograms of positive and negatives along with threshold
    criterion.

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.

    Examples:
        $ bob bio hist dev-scores

        $ bob bio hist --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob bio hist --test --criter hter dev-scores1 test-scores1
    """
    process = measure_figure.Hist(ctx, scores, test, FUNC_SPLIT)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.sep_dev_test_option()
@common_options.output_plot_file_option(default_out='cmc.pdf')
@common_options.test_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=None)
@common_options.axis_fontsize_option()
@common_options.x_rotation_option()
@verbosity_option()
@click.pass_context
def cmc(ctx, scores, test, **kargs):
    """Plot CMC (cumulative match characteristic curve):
    graphical presentation of results of an identification task test,
    plotting rank values on the x-axis and the probability of correct identification
    at or below that rank on the y-axis. The values for the axis will be
    computed using :py:func:`bob.measure.cmc`.

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.


    Examples:
        $ bob bio cmc dev-scores

        $ bob bio cmc --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob bio cmc --test -o my_roc.pdf dev-scores1 test-scores1
    """
    process = bio_figure.Cmc(ctx, scores, test, FUNC_CMC)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.sep_dev_test_option()
@common_options.output_plot_file_option(default_out='cmc.pdf')
@common_options.test_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=None)
@common_options.axis_fontsize_option()
@common_options.x_rotation_option()
@common_options.rank_option()
@verbosity_option()
@click.pass_context
def dic(ctx, scores, test, **kargs):
    """Plots the Detection & Identification curve over the FAR

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

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.


    Examples:
        $ bob bio dic dev-scores

        $ bob bio dic --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob bio dic --test -o my_roc.pdf dev-scores1 test-scores1
    """
    process = bio_figure.Dic(ctx, scores, test, FUNC_CMC)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.test_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.axis_fontsize_option()
@common_options.thresholds_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, test, **kwargs):
    """ Plots histograms of positive and negatives along with threshold
    criterion.

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.

    Examples:
        $ bob bio hist dev-scores

        $ bob bio hist --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob bio hist --test --criter hter dev-scores1 test-scores1
    """
    process = measure_figure.Hist(ctx, scores, test, FUNC_SPLIT)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.sep_dev_test_option()
@common_options.table_option()
@common_options.test_option()
@common_options.output_plot_metric_option()
@common_options.output_plot_file_option(default_out='eval_plots.pdf')
@common_options.points_curve_option()
@common_options.fmr_line_at_option()
@common_options.cost_option()
@common_options.rank_option()
@common_options.cmc_option()
@common_options.bool_option(
    'metrics', 'M', 'If set, computes table of threshold with EER, HTER (and '
    'FAR, if ``--far-value`` provided.)'
)
@common_options.far_option()
@common_options.bool_option(
    'cllr', 'x', 'If given, Cllr and minCllr will be computed.'
)
@common_options.bool_option(
    'mindcf', 'm', 'If given, minDCF will be computed.'
)
@common_options.bool_option(
    'rr', 'r', 'If given, the Recognition Rate will be computed.'
)
@common_options.bool_option(
    'hist', 'H', 'If given, score histograms will be generated.'
)
@common_options.bool_option(
    'roc', 'R', 'If given, ROC will be generated.'
)
@common_options.bool_option(
    'det', 'D', 'If given, DET will be generated.'
)
@common_options.bool_option(
    'epc', 'E', 'If given, EPC will be generated.'
)
@common_options.bool_option(
    'dic', 'O', 'If given, DIC will be generated.'
)
@verbosity_option()
@click.pass_context
def evaluate(ctx, scores, test, **kwargs):
    '''Evalutes score file, runs error analysis on score sets and plot curves.

    \b
    1. Computes the threshold using either EER, min. HTER or FAR value
       criteria on development set scores
    2. Applies the above threshold on test set scores to compute the HTER, if a
       test-score set is provided
    3. Computes Cllr and minCllr, minDCF, and recognition rate (if cmc scores
       provided)
    3. Reports error metrics in the console or in a log file
    4. Plots ROC, EPC, DET, score distributions, CMC (if cmc) and DIC (if cmc)
       curves to a multi-page PDF file


    You need to provide 2 score files for each biometric system in this order:

    \b
    * development scores
    * evaluation scores

    Examples:
        $ bob bio evaluate dev-scores

        $ bob bio evaluate -t -l metrics.txt -o my_plots.pdf dev-scores test-scores
    '''
    log_str=''
    if 'log' in ctx.meta and ctx.meta['log'] is not None:
        log_str = ' %s' % ctx.meta['log']

    if ctx.meta['metrics']:
        # first time erase if existing file
        ctx.meta['open_mode'] = 'w'
        click.echo("Computing metrics with EER%s..." % log_str)
        ctx.meta['criter'] = 'eer'  # no criterion passed to evaluate
        ctx.invoke(metrics, scores=scores, test=test)
        # other times, appends the content
        ctx.meta['open_mode'] = 'a'
        click.echo("Computing metrics with HTER%s..." % log_str)
        ctx.meta['criter'] = 'hter'  # no criterion passed in evaluate
        ctx.invoke(metrics, scores=scores, test=test)
        if 'far_value' in ctx.meta and ctx.meta['far_value'] is not None:
            click.echo("Computing metrics with FAR=%f%s..." %\
                       (ctx.meta['far_value'], log_str))
            ctx.meta['criter'] = 'far'  # no criterio % n passed in evaluate
            ctx.invoke(metrics, scores=scores, test=test)

    if ctx.meta['mindcf']:
        click.echo("Computing minDCF%s..." % log_str)
        ctx.meta['criter'] = 'mindcf'  # no criterion passed in evaluate
        ctx.invoke(metrics, scores=scores, test=test)

    if ctx.meta['cllr']:
        click.echo("Computing  Cllr and minCllr%s..." % log_str)
        ctx.meta['criter'] = 'cllr'  # no criterion passed in evaluate
        ctx.invoke(metrics, scores=scores, test=test)

    if ctx.meta['rr']:
        click.echo("Computing  recognition rate%s..." % log_str)
        ctx.meta['criter'] = 'rr'  # no criterion passed in evaluate
        ctx.invoke(metrics, scores=scores, test=test)

    # avoid closing pdf file before all figures are plotted
    ctx.meta['closef'] = False

    if test:
        click.echo("Starting evaluate with dev and test scores...")
    else:
        click.echo("Starting evaluate with dev scores only...")

    if ctx.meta['roc']:
        click.echo("Generating ROC in %s..." % ctx.meta['output'])
        ctx.forward(roc) # use class defaults plot settings

    if ctx.meta['det']:
        click.echo("Generating DET in %s..." % ctx.meta['output'])
        ctx.forward(det) # use class defaults plot settings

    if test and ctx.meta['epc']:
        click.echo("Generating EPC in %s..." % ctx.meta['output'])
        ctx.forward(epc) # use class defaults plot settings

    if ctx.meta['cmc']:
        click.echo("Generating CMC in %s..." % ctx.meta['output'])
        ctx.forward(cmc) # use class defaults plot settings

    if ctx.meta['dic']:
        click.echo("Generating DIC in %s..." % ctx.meta['output'])
        ctx.forward(dic) # use class defaults plot settings

    # the last one closes the file
    if ctx.meta['hist']:
        click.echo("Generating score histograms in %s..." % ctx.meta['output'])
        ctx.meta['criter'] = 'hter'  # no criterion passed in evaluate
        ctx.forward(hist)
    ctx.meta['closef'] = True
    #just to make sure pdf is closed
    if 'PdfPages' in ctx.meta:
        ctx.meta['PdfPages'].close()

    click.echo("Evaluate successfully completed!")
