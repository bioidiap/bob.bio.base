"""Generate random scores.
"""
import csv
import logging
import os

import click
import numpy

from clapper.click import verbosity_option

logger = logging.getLogger(__name__)


def gen_score_distr(
    mean_neg,
    mean_pos,
    sigma_neg=10,
    sigma_pos=10,
    n_neg=5000,
    n_pos=5000,
    seed=0,
):
    """Generate scores from normal distributions

    Parameters
    ----------
    mean_neg : float
        Mean for negative scores
    mean_pos : float
        Mean for positive scores
    sigma_neg : float
        STDev for negative scores
    sigma_pos : float
        STDev for positive scores
    n_pos: int
        The number of positive scores generated
    n_neg: int
        The number of negative scores generated
    seed: int
        A value to initialize the Random Number generator. Giving the same
        value (or not specifying 'seed') on two different calls will generate
        the same lists of scores.

    Returns
    -------
    neg_scores : :any:`list`
        Negatives scores
    pos_scores : :any:`list`
        Positive scores
    """

    logger.debug("Initializing RNG.")
    numpy.random.seed(seed)

    logger.info(f"Generating {n_neg} negative and {n_pos} positive scores.")

    neg_scores = numpy.random.normal(loc=mean_neg, scale=sigma_neg, size=n_neg)
    pos_scores = numpy.random.normal(loc=mean_pos, scale=sigma_pos, size=n_pos)

    return neg_scores, pos_scores


def write_scores_to_file(
    neg,
    pos,
    filename,
    n_subjects=5,
    n_probes_per_subject=5,
    n_unknown_subjects=0,
    neg_unknown=None,
    to_csv=True,
    five_col=False,
    metadata={"meta0": "data0", "meta1": "data1"},
):
    """Writes score distributions

    Parameters
    ----------
    neg : :py:class:`numpy.ndarray`
        Scores for negative samples.
    pos : :py:class:`numpy.ndarray`
        Scores for positive samples.
    filename : str
        The path to write the score to.
    n_subjects: int
        Number of different subjects
    n_probes_per_subject: int
        Number of different samples used as probe for each subject
    n_unknown_subjects: int
        The number of unknown (no registered model) subjects
    neg_unknown: None or list
        The of unknown subjects scores
    to_csv: bool
        Use the CSV format, else the legacy 4 or 5 columns format.
    five_col : bool
        If 5-colum format, else 4-column
    """
    logger.debug(f"Creating result directories ('{filename}').")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    s_subjects = ["x%d" % i for i in range(n_subjects)]

    logger.debug("Writing scores to files.")

    with open(filename, "wt") as f:
        if to_csv:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ["bio_ref_subject_id", "probe_subject_id", "key", "score"]
                + list(metadata.keys())
            )
        # Generate one line per probe (unless "--force-count" specified)
        logger.debug("Writing positive scores.")
        for i, score in enumerate(pos):
            s_name = s_subjects[int(i / n_probes_per_subject) % n_subjects]
            s_five = " " if not five_col else " d" + s_name + " "
            probe_id = "%s_%d" % (s_name, i % n_probes_per_subject)
            if to_csv:
                csv_writer.writerow(
                    [s_name, s_name, probe_id, score] + list(metadata.values())
                )
            else:
                f.write(
                    "%s%s%s %s %f\n" % (s_name, s_five, s_name, probe_id, score)
                )

        # Generate one line per probe against each ref (unless "--force-count" specified)
        logger.debug("Writing negative scores.")
        for i, score in enumerate(neg):
            n_impostors = n_subjects - 1
            ref = s_subjects[
                int(i / n_probes_per_subject / n_impostors) % n_subjects
            ]
            impostors = [s for s in s_subjects if s != ref]  # ignore pos
            probe = impostors[int(i / n_probes_per_subject) % n_impostors]
            s_five = " " if not five_col else " d" + ref
            probe_id = "%s_%d" % (probe, i % n_probes_per_subject)
            if to_csv:
                csv_writer.writerow(
                    [ref, probe, probe_id, score] + list(metadata.values())
                )
            else:
                f.write(
                    "%s%s%s %s %f\n" % (ref, s_five, probe, probe_id, score)
                )

        logger.debug("Writing unknown scores.")
        if neg_unknown is not None:
            s_unknown_subjects = ["u%d" % i for i in range(n_unknown_subjects)]
            for i, score in enumerate(neg_unknown):
                ref = s_subjects[
                    int(i / n_probes_per_subject / n_unknown_subjects)
                    % n_subjects
                ]
                probe = s_unknown_subjects[
                    int(i / n_probes_per_subject) % n_unknown_subjects
                ]
                s_five = " " if not five_col else " d" + ref + " "
                probe_id = "%s_%d" % (probe, i % n_probes_per_subject)
                if to_csv:
                    csv_writer.writerow(
                        [ref, probe, probe_id, score] + list(metadata.values())
                    )
                else:
                    f.write(
                        "%s%s%s %s %f\n" % (ref, s_five, probe, probe_id, score)
                    )


@click.command(
    epilog="""
Scores generation examples:

Output 'scores-dev.csv' and 'scores-eval.csv' in a new folder 'generated_scores/':

  $ bob bio gen ./generated_scores

Output scores similar to a system evaluated on the AT&T dataset dev group:

  $ bob bio gen -s 20 -p 5 ./generated_scores

Output a given number of scores in each file:

  $ bob bio gen -f --n-neg 500 --n-pos 100 ./generated_scores

Include unknown subjects scores:

  $ bob bio gen -s 5 -u 2 ./generated_scores

Change the mean and standard deviation of the scores distributions:

  $ bob bio gen -mm 1 -sp 0.3 -mnm -1 -sn 0.5 ./generated_scores

You can observe the distributions histograms in a pdf file with:

  $ bob bio hist -e ./generated_scores/scores-{dev,eval}.csv -o hist_gen.pdf
"""
)
@click.argument("outdir")
@click.option(
    "-mm",
    "--mean-match",
    default=10,
    type=click.FLOAT,
    show_default=True,
    help="Mean for the positive scores distribution",
)
@click.option(
    "-mnm",
    "--mean-non-match",
    default=-10,
    type=click.FLOAT,
    show_default=True,
    help="Mean for the negative scores distribution",
)
@click.option(
    "-p",
    "--n-probes-per-subject",
    default=5,
    type=click.INT,
    show_default=True,
    help="Number of probes per subject",
)
@click.option(
    "-s",
    "--n-subjects",
    default=50,
    type=click.INT,
    show_default=True,
    help="Number of subjects",
)
@click.option(
    "-sp",
    "--sigma-positive",
    default=10,
    type=click.FLOAT,
    show_default=True,
    help="Variance for the positive score distributions",
)
@click.option(
    "-sn",
    "--sigma-negative",
    default=10,
    type=click.FLOAT,
    show_default=True,
    help="Variance for the negative score distributions",
)
@click.option(
    "-u",
    "--n-unknown-subjects",
    default=0,
    type=click.INT,
    show_default=True,
    help="Number of unknown subjects (useful for open-set plots)",
)
@click.option(
    "-f",
    "--force-count",
    "force_count",
    is_flag=True,
    help="Use --n-pos and --n-neg amounts instead of the subject and sample counts",
)
@click.option(
    "--n-pos",
    "n_pos",
    default=5000,
    type=click.INT,
    show_default=True,
    help="Number of Positive verifications (number of lines in the file)",
)
@click.option(
    "--n-neg",
    "n_neg",
    default=5000,
    type=click.INT,
    show_default=True,
    help="Number of Negative verifications (number of lines in the file)",
)
@click.option(
    "--n-unk",
    "n_unk",
    default=5000,
    type=click.INT,
    show_default=True,
    help="Number of Unknown verifications (number of lines in the file)",
)
@click.option("--csv/--legacy", default=True, show_default=True)
@click.option("--five-col/--four-col", default=False, show_default=True)
@verbosity_option(logger=logger)
def gen(
    outdir,
    mean_match,
    mean_non_match,
    n_probes_per_subject,
    n_subjects,
    sigma_positive,
    sigma_negative,
    n_unknown_subjects,
    csv,
    five_col,
    force_count,
    n_pos,
    n_neg,
    n_unk,
    **kwargs,
):
    """Generate random scores.

    Generates random scores in 4col or 5col format. The scores are generated
    using Gaussian distribution whose mean and variance are an input
    parameter. The generated scores can be used as hypothetical datasets.

    This command generates scores relative to the number of subjects and
    probes per subjects, unless the -f flag is set. In that case, the --n-pos
    and --n-neg options are used as number of genuine and impostor
    comparisons.
    """

    # Compute the number of verifications needed
    if force_count:
        neg_count, pos_count, unknown_count = n_neg, n_pos, n_unk
    else:
        # One reference (model), and `n_probes_per_subject` probes per subject
        neg_count = n_subjects * n_probes_per_subject * (n_subjects - 1)
        pos_count = n_probes_per_subject * n_subjects
        unknown_count = n_unknown_subjects * n_subjects * n_probes_per_subject

    # Generate the data
    logger.info("Generating dev scores.")
    neg_dev, pos_dev = gen_score_distr(
        mean_non_match,
        mean_match,
        sigma_negative,
        sigma_positive,
        n_neg=neg_count,
        n_pos=pos_count,
        seed=0,
    )
    logger.info("Generating eval scores.")
    neg_eval, pos_eval = gen_score_distr(
        mean_non_match,
        mean_match,
        sigma_negative,
        sigma_positive,
        n_neg=neg_count,
        n_pos=pos_count,
        seed=1,
    )

    # For simplicity I will use the same distribution for dev-eval
    if n_unknown_subjects:
        logger.info("Generating unknown scores.")
        neg_unknown, _ = gen_score_distr(
            mean_non_match,
            mean_match,
            sigma_negative,
            sigma_positive,
            n_neg=unknown_count,
            n_pos=0,
            seed=2,
        )
    else:
        neg_unknown = None

    # Write the data into files
    logger.info("Saving results.")
    write_scores_to_file(
        neg_dev,
        pos_dev,
        os.path.join(outdir, "scores-dev.csv"),
        n_subjects,
        n_probes_per_subject,
        n_unknown_subjects,
        neg_unknown,
        csv,
        five_col,
    )

    write_scores_to_file(
        neg_eval,
        pos_eval,
        os.path.join(outdir, "scores-eval.csv"),
        n_subjects,
        n_probes_per_subject,
        n_unknown_subjects,
        neg_unknown,
        csv,
        five_col,
    )
