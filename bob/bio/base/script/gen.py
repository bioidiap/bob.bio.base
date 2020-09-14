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


def gen_score_distr(mean_neg, mean_pos, sigma_neg=10, sigma_pos=10):
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

    Returns
    -------
    neg_scores : :any:`list`
        Negatives scores
    pos_scores : :any:`list`
        Positive scores
    """
    mt = bob.core.random.mt19937()  # initialise the random number generator

    neg_generator = bob.core.random.normal(numpy.float32, mean_neg, sigma_neg)
    pos_generator = bob.core.random.normal(numpy.float32, mean_pos, sigma_pos)

    neg_scores = [neg_generator(mt) for _ in range(NUM_NEG)]
    pos_scores = [pos_generator(mt) for _ in range(NUM_POS)]

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
    n_sys : int
        Number of different systems
    five_col : bool
        If 5-colum format, else 4-column
    """
    create_directories_safe(os.path.dirname(filename))
    s_subjects = ['x%d' % i for i in range(n_subjects)]

    with open(filename, 'wt') as f:
        for i in pos:
            s_name = random.choice(s_subjects)
            s_five = ' ' if not five_col else ' d' + \
                random.choice(s_subjects) + ' '
            probe_id = "%s_%d" %(s_name, random.randint(0, n_probes_per_subject-1))
            f.write('%s%s%s %s %f\n' % (s_name, s_five, s_name, probe_id, i))
        for i in neg:
            s_names = random.sample(s_subjects, 2)
            s_five = ' ' if not five_col else ' d' + \
                random.choice(s_names) + ' '
            probe_id = "%s_%d" %(s_names[1], random.randint(0, n_probes_per_subject-1))
            f.write('%s%s%s %s %f\n' % (s_names[0], s_five, s_names[1], probe_id, i))

        if neg_unknown is not None:
            s_unknown_subjects = ['u%d' % i for i in range(n_unknown_subjects)]
            for i in neg_unknown:
                s_name = random.choice(s_subjects)
                s_name_probe = random.choice(s_unknown_subjects)
                s_five = ' ' if not five_col else ' d' + \
                    random.choice(s_subjects) + ' '
                probe_id = "%s_%d" %(s_name_probe, random.randint(0, n_probes_per_subject-1))
                f.write('%s%s%s %s %f\n' % (s_name, s_five, s_name_probe, probe_id, i))


@click.command()
@click.argument('outdir')
@click.option('-mm', '--mean-match', default=10, type=FLOAT, show_default=True,\
              help="Mean for the positive scores distribution")
@click.option('-mnm', '--mean-non-match', default=-10, type=FLOAT, show_default=True,\
             help="Mean for the negative scores distribution")
@click.option('--n-probes-per-subjects', default=5, type=click.INT, show_default=True,\
              help="Number of probes per subject")
@click.option('-s', '--n-subjects', default=5, type=click.INT, show_default=True,\
              help="Number of subjects")
@click.option('-p', '--sigma-positive', default=10, type=click.FLOAT, show_default=True,\
              help="Variance for the positive score distributions")
@click.option('-n', '--sigma-negative', default=10, type=click.FLOAT, show_default=True,\
              help="Variance for the negative score distributions")
@click.option('-u', '--n-unknown-subjects', default=0, type=click.INT, show_default=True,\
              help="Number of unknown subjects (useful for openset plots)")
@click.option('--five-col/--four-col', default=False, show_default=True)
@verbosity_option()
def gen(outdir, mean_match, mean_non_match, n_probes_per_subjects, n_subjects,\
        sigma_positive, sigma_negative, n_unknown_subjects,  five_col, **kwargs):
    """Generate random scores.
    Generates random scores in 4col or 5col format. The scores are generated
    using Gaussian distribution whose mean is an input
    parameter. The generated scores can be used as hypothetical datasets.
    """
    # Generate the data
    neg_dev, pos_dev = gen_score_distr(mean_non_match, mean_match, sigma_negative, sigma_positive)
    neg_eval, pos_eval = gen_score_distr(mean_non_match, mean_match, sigma_negative, sigma_positive)

    # For simplicity I will use the same distribution for dev-eval
    if n_unknown_subjects:
        neg_unknown,_ = gen_score_distr(mean_non_match, mean_match, sigma_negative, sigma_positive)
    else:
        neg_unknown = None

    # Write the data into files
    write_scores_to_file(neg_dev, pos_dev,
                         os.path.join(outdir, 'scores-dev'),
                         n_subjects, n_probes_per_subjects,
                         n_unknown_subjects, neg_unknown, five_col)

    write_scores_to_file(neg_eval, pos_eval,
                         os.path.join(outdir, 'scores-eval'),
                         n_subjects, n_probes_per_subjects,
                         n_unknown_subjects, neg_unknown, five_col)

