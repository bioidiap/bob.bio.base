"""The click-based vulnerability analysis commands.
"""

import csv
import functools
import logging
import os

import click

from click.types import FLOAT
from numpy import random

from bob.bio.base.score.load import split_csv_vuln
from bob.io.base import create_directories_safe
from bob.measure.script import common_options

from . import vuln_figure as figure

logger = logging.getLogger(__name__)


def vuln_plot_options(
    docstring,
    plot_output_default="vuln_plot.pdf",
    legend_loc_default="best",
    axes_lim_default=None,
    figsize_default="4,3",
    force_eval=False,
    x_label_rotation_default=0,
):
    def custom_options_command(func):
        func.__doc__ = docstring

        def eval_if_not_forced(force_eval):
            def decorator(f):
                if not force_eval:
                    return common_options.eval_option()(
                        common_options.sep_dev_eval_option()(
                            common_options.hide_dev_option()(f)
                        )
                    )
                else:
                    return f

            return decorator

        @click.command()
        @common_options.scores_argument(
            min_arg=1, force_eval=force_eval, nargs=-1
        )
        @eval_if_not_forced(force_eval)
        @common_options.legends_option()
        @common_options.no_legend_option()
        @common_options.legend_ncols_option()
        @common_options.legend_loc_option(dflt=legend_loc_default)
        @common_options.output_plot_file_option(default_out=plot_output_default)
        @common_options.lines_at_option(dflt=" ")
        @common_options.axes_val_option(dflt=axes_lim_default)
        @common_options.x_rotation_option(dflt=x_label_rotation_default)
        @common_options.x_label_option()
        @common_options.y_label_option()
        @common_options.points_curve_option()
        @common_options.const_layout_option()
        @common_options.figsize_option(dflt=figsize_default)
        @common_options.style_option()
        @common_options.linestyles_option()
        @common_options.alpha_option()
        @common_options.verbosity_option()
        @click.pass_context
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        return wrapper

    return custom_options_command


def real_data_option(**kwargs):
    """Option to choose if input data is real or generated"""
    return common_options.bool_option(
        name="real-data",
        short_name="R",
        desc="If False, will annotate the plots hypothetically, instead "
        "of with real data values of the calculated error rates.",
        dflt=True,
        **kwargs,
    )


def fnmr_at_option(dflt=" ", **kwargs):
    """Get option to draw const FNMR lines"""
    return common_options.list_float_option(
        name="fnmr",
        short_name="fnmr",
        desc="If given, draw horizontal lines at the given FNMR position. "
        "Your values must be separated with a comma (,) without space. "
        "This option works in ROC and DET curves.",
        nitems=None,
        dflt=dflt,
        **kwargs,
    )


def gen_score_distr(
    mean_gen,
    mean_zei,
    mean_pa,
    sigma_gen=1,
    sigma_zei=1,
    sigma_pa=1,
    num_gen=5000,
    num_zei=5000,
    num_pa=5000,
):
    # initialise the random number generator
    mt = random.RandomState(0)

    genuine_scores = mt.normal(loc=mean_gen, scale=sigma_gen, size=num_gen)
    zei_scores = mt.normal(loc=mean_zei, scale=sigma_zei, size=num_zei)
    pa_scores = mt.normal(loc=mean_pa, scale=sigma_pa, size=num_pa)

    return genuine_scores, zei_scores, pa_scores


def write_scores_to_file(neg_licit, pos_licit, spoof, filename):
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
    with open(filename, "wt") as f:
        csv_writer = csv.writer(f)
        # Write the header
        csv_writer.writerow(
            [
                "bio_ref_reference_id",
                "probe_reference_id",
                "probe_key",
                "probe_attack_type",
                "score",
            ]
        )
        for score in neg_licit:
            csv_writer.writerow(["x", "y", "0", None, score])
        for score in pos_licit:
            csv_writer.writerow(["x", "x", "0", None, score])
        for score in spoof:
            csv_writer.writerow(["x", "y", "0", "pai", score])


@click.command()
@click.argument("outdir")
@click.option("-mg", "--mean-gen", default=7, type=FLOAT, show_default=True)
@click.option("-mz", "--mean-zei", default=3, type=FLOAT, show_default=True)
@click.option("-mp", "--mean-pa", default=5, type=FLOAT, show_default=True)
@common_options.verbosity_option()
def gen(outdir, mean_gen, mean_zei, mean_pa, **kwargs):
    """Generate random scores.
    Generates random scores for three types of verification attempts:
    genuine users, zero-effort impostors and spoofing attacks and writes them
    into CSV score files for so called licit and spoof scenario. The
    scores are generated using Gaussian distribution whose mean is an input
    parameter. The generated scores can be used as hypothetical datasets.
    """
    # Generate the data
    genuine_dev, zei_dev, pa_dev = gen_score_distr(mean_gen, mean_zei, mean_pa)
    genuine_eval, zei_eval, pa_eval = gen_score_distr(
        mean_gen, mean_zei, mean_pa
    )

    # Write the data into files
    write_scores_to_file(
        zei_dev, genuine_dev, pa_dev, os.path.join(outdir, "scores-dev.csv")
    )
    write_scores_to_file(
        zei_eval, genuine_eval, pa_eval, os.path.join(outdir, "scores-eval.csv")
    )


@common_options.metrics_command(
    docstring="""Extracts different statistical values from scores distributions

    Prints a table that contains different metrics to assess the performance of a
    biometric system against zero-effort impostor and presentation attacks.

    The CSV score files must contain an `attack-type` column, in addition to the
    "regular" biometric scores columns (`bio_ref_reference_id`,
    `probe_reference_id`, and `score`).

    Examples:

        $ bob vuln metrics -v scores-dev.csv

        $ bob vuln metrics -v -e scores-{dev,eval}.csv
  """
)
@common_options.cost_option()
def metrics(ctx, scores, evaluation, **kwargs):
    process = figure.Metrics(ctx, scores, evaluation, split_csv_vuln)
    process.run()


@vuln_plot_options(
    docstring="""Plots the ROC for vulnerability analysis

    You need to provide 1 or 2 (with `--eval`) score
    files for each vulnerability system in this order:

    \b
    * dev scores
    * [eval scores]

    The CSV score files must contain an `attack-type` column, in addition to the
    "regular" biometric scores columns (`bio_ref_reference_id`,
    `probe_reference_id`, and `score`).

    Examples:

        $ bob vuln roc -v -o roc.pdf scores.csv

        $ bob vuln roc -v -e scores-{dev,eval}.csv
  """,
    plot_output_default="vuln_roc.pdf",
)
@common_options.title_option()
@common_options.min_far_option()
@common_options.tpr_option(dflt=True)
@common_options.semilogx_option(dflt=True)
@fnmr_at_option()
@real_data_option()
def roc(ctx, scores, evaluation, real_data, **kwargs):
    process = figure.RocVuln(
        ctx, scores, evaluation, split_csv_vuln, real_data, False
    )
    process.run()


@vuln_plot_options(
    docstring="""Plots the DET for vulnerability analysis

    You need to provide 1 or 2 (with `--eval`) score
    files for each vulnerability system in this order:

    \b
    * dev scores
    * [eval scores]

    The CSV score files must contain an `attack-type` column, in addition to the
    "regular" biometric scores columns (`bio_ref_reference_id`,
    `probe_reference_id`, and `score`).

    See :ref:`bob.bio.base.vulnerability` in the documentation for a guide on
    vulnerability analysis.

    Examples:

        $ bob vuln det -v -o det.pdf scores.csv

        $ bob vuln det -v -e scores-{dev,eval}.csv
  """,
    plot_output_default="vuln_det.pdf",
    legend_loc_default="upper-right",
    axes_lim_default="0.01,95,0.01,95",
    figsize_default="6,4",
    x_label_rotation_default=45,
)
@common_options.title_option()
@real_data_option()
@fnmr_at_option()
def det(ctx, scores, evaluation, real_data, **kwargs):
    process = figure.DetVuln(
        ctx, scores, evaluation, split_csv_vuln, real_data, False
    )
    process.run()


@vuln_plot_options(
    docstring="""Plots the EPC for vulnerability analysis

    You need to provide 2 score files for each vulnerability system in this order:

    \b
    * dev scores
    * eval scores

    The CSV score files must contain an `attack-type` column, in addition to the
    "regular" biometric scores columns (`bio_ref_reference_id`,
    `probe_reference_id`, and `score`).

    See :ref:`bob.bio.base.vulnerability` in the documentation for a guide on
    vulnerability analysis.

    Examples:

        $ bob vuln epc -v scores-dev.csv scores-eval.csv

        $ bob vuln epc -v -o epc.pdf scores-{dev,eval}.csv
  """,
    plot_output_default="vuln_epc.pdf",
    force_eval=True,
    legend_loc_default="upper-center",
)
@common_options.title_option()
@common_options.bool_option(
    "iapmr", "I", "Whether to plot the IAPMR related lines or not.", True
)
def epc(ctx, scores, **kwargs):
    process = figure.Epc(ctx, scores, True, split_csv_vuln)
    process.run()


@vuln_plot_options(
    docstring="""Plots the EPSC for vulnerability analysis

    Plots the Expected Performance Spoofing Curve.

    Note that when using 3D plots with option ``--three-d``, you cannot plot
    both WER and IAPMR on the same figure (which is possible in 2D).

    You need to provide 2 score files for each vulnerability system in this order:

    \b
    * dev scores
    * eval scores

    The CSV score files must contain an `attack-type` column, in addition to the
    "regular" biometric scores columns (`bio_ref_reference_id`,
    `probe_reference_id`, and `score`).

    See :ref:`bob.bio.base.vulnerability` in the documentation for a guide on
    vulnerability analysis.

    Examples:

        $ bob vuln epsc -v scores-dev.csv scores-eval.csv

        $ bob vuln epsc -v -o epsc.pdf scores-{dev,eval}.csv

        $ bob vuln epsc -v -D -o epsc_3D.pdf scores-{dev,eval}.csv
  """,
    plot_output_default="vuln_epc.pdf",
    force_eval=True,
    figsize_default="5,3",
)
@common_options.titles_option()
@common_options.bool_option(
    "wer", "w", "Whether to plot the WER related lines or not.", True
)
@common_options.bool_option(
    "iapmr", "I", "Whether to plot the IAPMR related lines or not.", True
)
@common_options.bool_option(
    "three-d",
    "D",
    "If true, generate 3D plots. You need to turn off "
    "wer or iapmr when using this option.",
    False,
)
@click.option(
    "-c",
    "--criteria",
    default="eer",
    show_default=True,
    help="Criteria for threshold selection",
    type=click.Choice(("eer", "min-hter")),
)
@click.option(
    "-vp",
    "--var-param",
    default="omega",
    show_default=True,
    help="Name of the varying parameter",
    type=click.Choice(("omega", "beta")),
)
@common_options.list_float_option(
    name="fixed-params",
    short_name="fp",
    dflt="0.5",
    desc="Values of the fixed parameter, separated by commas",
)
@click.option(
    "-s",
    "--sampling",
    default=5,
    show_default=True,
    help="Sampling of the EPSC 3D surface",
    type=click.INT,
)
def epsc(ctx, scores, criteria, var_param, three_d, sampling, **kwargs):
    if three_d:
        if ctx.meta["wer"] and ctx.meta["iapmr"]:
            logger.info(
                "Cannot plot both WER and IAPMR in 3D. Will turn IAPMR off."
            )
            ctx.meta["iapmr"] = False
        ctx.meta["sampling"] = sampling
        process = figure.Epsc3D(
            ctx, scores, split_csv_vuln, criteria, var_param, **kwargs
        )
    else:
        process = figure.Epsc(
            ctx, scores, split_csv_vuln, criteria, var_param, **kwargs
        )
    process.run()


@vuln_plot_options(
    docstring="""Vulnerability analysis score distribution histograms.

    Plots the histogram of score distributions. You need to provide 1 or 2 score
    files for each biometric system in this order:

    \b
    * development scores
    * [evaluation scores]

    When evaluation scores are provided, you must use the ``--eval`` option.

    The CSV score files must contain an `attack-type` column, in addition to the
    "regular" biometric scores columns (`bio_ref_reference_id`,
    `probe_reference_id`, and `score`).

    See :ref:`bob.bio.base.vulnerability` in the documentation for a guide on
    vulnerability analysis.

    When eval-scores are given, eval-scores histograms are displayed with the
    threshold line computed from dev-scores.

    Examples:

        $ bob vuln hist -v -o hist.pdf results/scores-dev.csv

        $ bob vuln hist -e -v results/scores-dev.csv results/scores-eval.csv

        $ bob vuln hist -e -v results/scores-{dev,eval}.csv
  """,
    plot_output_default="vuln_hist.pdf",
)
@common_options.titles_option()
@common_options.n_bins_option()
@common_options.thresholds_option()
@common_options.print_filenames_option(dflt=False)
@common_options.bool_option(
    "iapmr-line", "I", "Whether to plot the IAPMR related lines or not.", True
)
@real_data_option()
@common_options.subplot_option()
@common_options.criterion_option()
def hist(ctx, scores, evaluation, **kwargs):
    process = figure.HistVuln(ctx, scores, evaluation, split_csv_vuln)
    process.run()


@vuln_plot_options(
    docstring="""Plots the FMR vs IAPMR for vulnerability analysis

    You need to provide 1 or 2 (with `--eval`) score
    files for each vulnerability system in this order:

    \b
    * dev scores
    * [eval scores]

    The CSV score files must contain an `attack-type` column, in addition to the
    "regular" biometric scores columns (`bio_ref_reference_id`,
    `probe_reference_id`, and `score`).

    Examples:

        $ bob vuln fmr_iapmr -v -o fmr_iapmr.pdf scores.csv

        $ bob vuln fmr_iapmr -v -e scores-{dev,eval}.csv
  """,
    plot_output_default="vuln_roc.pdf",
    force_eval=True,
)
@common_options.title_option()
@common_options.semilogx_option()
def fmr_iapmr(ctx, scores, **kwargs):
    process = figure.FmrIapmr(ctx, scores, True, split_csv_vuln)
    process.run()


@common_options.evaluate_command(
    common_options.EVALUATE_HELP.format(
        score_format=(
            "Files must be in CSV format, with the `bio_ref_reference_id`, "
            "`probe_references_id`, `score`, and `attack_type` columns."
        ),
        command="bob vuln evaluate",
    ),
    criteria=("eer", "min-hter", "far"),
)
def evaluate(ctx, scores, evaluation, **kwargs):
    # open_mode is always 'write' in this command.
    ctx.meta["open_mode"] = "w"
    criterion = ctx.meta.get("criterion")
    if criterion is not None:
        click.echo(f"Computing metrics with {criterion}...")
        ctx.invoke(metrics, scores=scores, evaluation=evaluation)
        if ctx.meta.get("log") is not None:
            click.echo(f"[metrics] => {ctx.meta['log']}")

    ctx.meta["lines_at"] = None

    # Avoid closing pdf file before all figures are plotted
    ctx.meta["closef"] = False
    if evaluation:
        click.echo("Starting evaluate with dev and eval scores...")
    else:
        click.echo("Starting evaluate with dev scores only...")
    click.echo("Plotting FMR vs IAPMR for bob vuln evaluate...")
    ctx.forward(fmr_iapmr)  # uses class defaults plot settings
    click.echo("Plotting ROC for bob vuln evaluate...")
    ctx.forward(roc)  # uses class defaults plot settings
    click.echo("Plotting DET for bob vuln evaluate...")
    ctx.forward(det)  # uses class defaults plot settings
    if evaluation:
        click.echo("Plotting EPSC for bob vuln evaluate...")
        ctx.forward(epsc)  # uses class defaults plot settings
    # Mark the last plot to close the output file
    ctx.meta["closef"] = True
    click.echo("Plotting score histograms for bob vuln evaluate...")
    ctx.forward(hist)
    click.echo("Evaluate successfully completed!")
    click.echo(f"[plots] => {ctx.meta['output']}")
