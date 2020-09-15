"""Generate random scores.
"""
import os
import logging
import numpy
import random
import click
from click.types import FLOAT
from bob.extension.scripts.click_helper import verbosity_option
import bob.core
from bob.io.base import create_directories_safe
from bob.measure.script import common_options

logger = logging.getLogger(__name__)

NUM_NEG = 5000
NUM_POS = 5000


def gen_score_distr(mean_neg, mean_pos, sigma_neg=10, sigma_pos=10, n_neg=NUM_NEG, n_pos=NUM_POS):
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

    Returns
    -------
    neg_scores : :any:`list`
        Negatives scores
    pos_scores : :any:`list`
        Positive scores
    """

    logger.debug("Initializing RNG.")
    mt = bob.core.random.mt19937()  # initialise the random number generator

    neg_generator = bob.core.random.normal(numpy.float32, mean_neg, sigma_neg)
    pos_generator = bob.core.random.normal(numpy.float32, mean_pos, sigma_pos)

    logger.info(f"Generating {n_neg} negative and {n_pos} positive scores.")

    neg_scores = [neg_generator(mt) for _ in range(n_neg)]
    pos_scores = [pos_generator(mt) for _ in range(n_pos)]

    return neg_scores, pos_scores


def write_scores_to_file(neg, pos, filename, n_subjects=5, n_probes_per_subject=5,
                         n_unknown_subjects=0, neg_unknown=None, five_col=False):
    """ Writes score distributions

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
    five_col : bool
        If 5-colum format, else 4-column
    """
    logger.debug(f"Creating result directories ('{filename}').")
    create_directories_safe(os.path.dirname(filename))
    s_subjects = ['x%d' % i for i in range(n_subjects)]

    logger.debug("Writing scores to files.")

    with open(filename, 'wt') as f:
        # Generate one line per probe (unless "--force-count" specified)
        logger.debug("Writing positive scores.")
        for i,score in enumerate(pos):
            s_name = s_subjects[int(i/n_probes_per_subject) % n_subjects]
            s_five = ' ' if not five_col else ' d' + s_name + ' '
            probe_id = "%s_%d" %(s_name, i%n_probes_per_subject)
            f.write('%s%s%s %s %f\n' % (s_name, s_five, s_name, probe_id, score))

        # Generate one line per probe against each ref (unless "--force-count" specified)
        logger.debug("Writing negative scores.")
        for i,score in enumerate(neg):
            n_impostors = n_subjects - 1
            ref = s_subjects[int(i/n_probes_per_subject/n_impostors) % n_subjects]
            impostors = [s for s in s_subjects if s!=ref] # ignore pos
            probe = impostors[int(i/n_probes_per_subject) % n_impostors]
            s_five = ' ' if not five_col else ' d' + ref
            probe_id = "%s_%d" %(probe, i%n_probes_per_subject)
            f.write('%s%s%s %s %f\n' % (ref, s_five, probe, probe_id, score))

        logger.debug("Writing unknown scores.")
        if neg_unknown is not None:
            s_unknown_subjects = ['u%d' % i for i in range(n_unknown_subjects)]
            for i,score in enumerate(neg_unknown):
                ref = s_subjects[int(i/n_probes_per_subject/n_unknown_subjects)%n_subjects]
                probe = s_unknown_subjects[int(i/n_probes_per_subject) % n_unknown_subjects]
                s_five = ' ' if not five_col else ' d' + ref + ' '
                probe_id = "%s_%d" %(probe, i%n_probes_per_subject)
                f.write('%s%s%s %s %f\n' % (ref, s_five, probe, probe_id, score))


@click.command()
@click.argument('outdir')
@click.option('-mm', '--mean-match', default=10, type=FLOAT, show_default=True,\
              help="Mean for the positive scores distribution")
@click.option('-mnm', '--mean-non-match', default=-10, type=FLOAT, show_default=True,\
             help="Mean for the negative scores distribution")
@click.option('-p', '--n-probes-per-subject', default=5, type=click.INT, show_default=True,\
              help="Number of probes per subject")
@click.option('-s', '--n-subjects', default=5, type=click.INT, show_default=True,\
              help="Number of subjects")
@click.option('-sp', '--sigma-positive', default=10, type=click.FLOAT, show_default=True,\
              help="Variance for the positive score distributions")
@click.option('-sn', '--sigma-negative', default=10, type=click.FLOAT, show_default=True,\
              help="Variance for the negative score distributions")
@click.option('-u', '--n-unknown-subjects', default=0, type=click.INT, show_default=True,\
              help="Number of unknown subjects (useful for openset plots)")
@click.option("-f", "--force-count", "force_count", is_flag=True,
              help="Use --n-pos and --n-neg amounts instead of the subject and sample counts")
@click.option("--n-pos", "n_pos", default=5000, type=click.INT, show_default=True,\
              help="Number of Positive verifications (number of lines in the file)")
@click.option("--n-neg", "n_neg", default=5000, type=click.INT, show_default=True,\
              help="Number of Negative verifications (number of lines in the file)")
@click.option("--n-unk", "n_unk", default=5000, type=click.INT, show_default=True,\
              help="Number of Unknown verifications (number of lines in the file)")
@click.option('--five-col/--four-col', default=False, show_default=True)
@verbosity_option()
def gen(outdir, mean_match, mean_non_match, n_probes_per_subject, n_subjects,
        sigma_positive, sigma_negative, n_unknown_subjects, five_col,
        force_count, n_pos, n_neg, n_unk, **kwargs):
    """Generate random scores.

    Generates random scores in 4col or 5col format. The scores are generated
    using Gaussian distribution whose mean and variance are an input parameter.
    The generated scores can be used as hypothetical datasets.

    This command generates scores relative to the number of subjects and
    samples per subjects, unless the -f flag is set. In that case, the --n-pos
    and --n-neg options are used as number of genuine and impostor comparisons.
    """

    # Compute the number of verifications needed
    if force_count:
        neg_count, pos_count, unknown_count = n_neg, n_pos, n_unk
    else:
        # One reference (model), and `n_probes_per_subject` probes per subject
        neg_count = n_subjects * n_probes_per_subject * (n_subjects-1)
        pos_count = n_probes_per_subject * n_subjects
        unknown_count = n_unknown_subjects*n_subjects*n_probes_per_subject

    # Generate the data
    logger.info("Generating dev scores.")
    neg_dev, pos_dev = gen_score_distr(mean_non_match, mean_match,
                                       sigma_negative, sigma_positive,
                                       n_neg=neg_count, n_pos=pos_count)
    logger.info("Generating eval scores.")
    neg_eval, pos_eval = gen_score_distr(mean_non_match, mean_match,
                                         sigma_negative, sigma_positive,
                                         n_neg=neg_count, n_pos=pos_count)

    # For simplicity I will use the same distribution for dev-eval
    if n_unknown_subjects:
        logger.info("Generating unknown scores.")
        neg_unknown,_ = gen_score_distr(mean_non_match, mean_match,
                                        sigma_negative, sigma_positive,
                                        n_neg=unknown_count, n_pos=0)
    else:
        neg_unknown = None

    # Write the data into files
    logger.info("Saving results.")
    write_scores_to_file(neg_dev, pos_dev,
                         os.path.join(outdir, 'scores-dev'),
                         n_subjects, n_probes_per_subject,
                         n_unknown_subjects, neg_unknown, five_col)

    write_scores_to_file(neg_eval, pos_eval,
                         os.path.join(outdir, 'scores-eval'),
                         n_subjects, n_probes_per_subject,
                         n_unknown_subjects, neg_unknown, five_col)

