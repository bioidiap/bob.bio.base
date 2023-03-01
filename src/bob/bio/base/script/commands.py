""" Click commands for ``bob.bio.base`` """
import logging

import click

from clapper.click import verbosity_option

import bob.measure.script.figure as measure_figure

from bob.measure.script import common_options

from ..score import load
from . import figure as bio_figure

logger = logging.getLogger(__name__)

SCORE_FORMAT = (
    "Files must be 4- or 5- columns format, see "
    ":py:func:`bob.bio.base.score.load.four_column` and "
    ":py:func:`bob.bio.base.score.load.five_column` for details."
)
CRITERIA = ("eer", "min-hter", "far", "mindcf", "cllr", "rr")


def rank_option(**kwargs):
    """Get option for rank parameter"""

    def custom_rank_option(func):
        def callback(ctx, param, value):
            value = 1 if value < 0 else value
            ctx.meta["rank"] = value
            return value

        return click.option(
            "-rk",
            "--rank",
            type=click.INT,
            default=1,
            help="Provide rank for the command",
            callback=callback,
            show_default=True,
            **kwargs,
        )(func)

    return custom_rank_option


@common_options.metrics_command(
    common_options.METRICS_HELP.format(
        names="FtA, FAR, FRR, FMR, FNMR, HTER",
        criteria=CRITERIA,
        score_format=SCORE_FORMAT,
        hter_note="Note that FAR = FMR * (1 - FtA), FRR = FtA + FNMR * (1 - FtA) "
        "and HTER = (FMR + FNMR) / 2",
        command="bob bio metrics",
    ),
    criteria=CRITERIA,
)
@common_options.cost_option()
def metrics(ctx, scores, evaluation, **kwargs):
    if "criterion" in ctx.meta and ctx.meta["criterion"] == "rr":
        process = bio_figure.Metrics(ctx, scores, evaluation, load.cmc)
    else:
        process = bio_figure.Metrics(ctx, scores, evaluation, load.split)
    process.run()


@common_options.roc_command(
    common_options.ROC_HELP.format(
        score_format=SCORE_FORMAT, command="bob bio roc"
    )
)
def roc(ctx, scores, evaluation, **kwargs):
    process = bio_figure.Roc(ctx, scores, evaluation, load.split)
    process.run()


@common_options.det_command(
    common_options.DET_HELP.format(
        score_format=SCORE_FORMAT, command="bob bio det"
    )
)
def det(ctx, scores, evaluation, **kwargs):
    process = bio_figure.Det(ctx, scores, evaluation, load.split)
    process.run()


@common_options.epc_command(
    common_options.EPC_HELP.format(
        score_format=SCORE_FORMAT, command="bob bio epc"
    )
)
def epc(ctx, scores, **kwargs):
    process = measure_figure.Epc(ctx, scores, True, load.split)
    process.run()


@common_options.hist_command(
    common_options.HIST_HELP.format(
        score_format=SCORE_FORMAT, command="bob bio hist"
    )
)
def hist(ctx, scores, evaluation, **kwargs):
    process = bio_figure.Hist(ctx, scores, evaluation, load.split)
    process.run()


@common_options.evaluate_command(
    common_options.EVALUATE_HELP.format(
        score_format=SCORE_FORMAT, command="bob bio evaluate"
    ),
    criteria=CRITERIA,
)
@common_options.cost_option()
def evaluate(ctx, scores, evaluation, **kwargs):
    common_options.evaluate_flow(
        ctx, scores, evaluation, metrics, roc, det, epc, hist, **kwargs
    )


@common_options.multi_metrics_command(
    common_options.MULTI_METRICS_HELP.format(
        names="FtA, FAR, FRR, FMR, FNMR, HTER",
        criteria=CRITERIA,
        score_format=SCORE_FORMAT,
        command="bob bio multi-metrics",
    ),
    criteria=CRITERIA,
)
def multi_metrics(ctx, scores, evaluation, protocols_number, **kwargs):
    ctx.meta["min_arg"] = protocols_number * (2 if evaluation else 1)
    process = bio_figure.MultiMetrics(ctx, scores, evaluation, load.split)
    process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.legends_option()
@common_options.sep_dev_eval_option()
@common_options.output_plot_file_option(default_out="cmc.pdf")
@common_options.eval_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=None)
@common_options.x_rotation_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option()
@verbosity_option(logger=logger)
@click.pass_context
def cmc(ctx, scores, evaluation, **kwargs):
    """Plot CMC (cumulative match characteristic curve).
    graphical presentation of results of an identification task eval, plotting
    rank values on the x-axis and the probability of correct identification at
    or below that rank on the y-axis. The values for the axis will be computed
    using :py:func:`bob.measure.cmc`.

    You need to provide one or more development score file(s) for each
    experiment. You can also provide eval files along with dev files. If
    eval-scores are used, the flag `--eval` must be used. is required
    in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.


    Examples:
        $ bob bio cmc -v dev-scores

        $ bob bio cmc -v dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob bio cmc -v -o my_roc.pdf dev-scores1 eval-scores1
    """
    process = bio_figure.Cmc(ctx, scores, evaluation, load.cmc)
    process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.legends_option()
@common_options.sep_dev_eval_option()
@common_options.output_plot_file_option(default_out="dir.pdf")
@common_options.eval_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=None)
@common_options.x_rotation_option()
@rank_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option()
@common_options.min_far_option()
@verbosity_option(logger=logger)
@click.pass_context
def dir(ctx, scores, evaluation, **kwargs):
    """Plots the Detection & Identification Rate curve over the FAR.

    This curve is designed to be used in an open set identification protocol,
    and defined in Chapter 14.1 of [LiJain2005]_.  It requires to have at least
    one open set probe item, i.e., with no corresponding gallery, such that the
    positives for that pair are ``None``.

    The detection and identification curve first computes FAR thresholds based
    on the out-of-set probe scores (negative scores).  For each probe item, the
    **maximum** negative score is used.  Then, it plots the detection and
    identification rates for those thresholds, which are based on the in-set
    probe scores only. See [LiJain2005]_ for more details.

    .. [LiJain2005] **Stan Li and Anil K. Jain**, *Handbook of Face Recognition*, Springer, 2005

    You need to provide one or more development score file(s) for each
    experiment. You can also provide eval files along with dev files. If
    eval-scores are used, the flag `--eval` must be used. is required
    in that case. Files must be 4- or 5- columns format, see
    :py:func:`bob.bio.base.score.load.four_column` and
    :py:func:`bob.bio.base.score.load.five_column` for details.

    Examples:
        $ bob bio dir -e -v dev-scores

        $ bob bio dir -v dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob bio dir -v -o my_roc.pdf dev-scores1 eval-scores1
    """
    process = bio_figure.Dir(ctx, scores, evaluation, load.cmc)
    process.run()
