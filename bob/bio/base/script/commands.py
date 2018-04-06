''' Click commands for ``bob.bio.base`` '''


import click
from ..score import load
from bob.measure.script import figure
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import verbosity_option


FUNC_SPLIT = lambda x: load.load_files(x, load.split)


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.table_option()
@common_options.test_option()
@common_options.open_file_mode_option()
@common_options.output_plot_metric_option()
@common_options.criterion_option()
@common_options.threshold_option()
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
    process = figure.Metrics(ctx, scores, test, FUNC_SPLIT)
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
    process = figure.Roc(ctx, scores, test, FUNC_SPLIT)
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
    process = figure.Det(ctx, scores, test, FUNC_SPLIT)
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
    process = figure.Epc(ctx, scores, True, FUNC_SPLIT)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.test_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.axis_fontsize_option()
@common_options.threshold_option()
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
    process = figure.Hist(ctx, scores, test, FUNC_SPLIT)
    process.run()
