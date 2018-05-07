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
    pos_scores = [pos_generator(mt) for _ in range(NUM_NEG)]

    return neg_scores, pos_scores


def write_scores_to_file(neg, pos, filename, n_sys=1, five_col=False):
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
    s_names = ['s%d' % i for i in range(n_sys)]
    with open(filename, 'wt') as f:
        for i in pos:
            s_name = random.choice(s_names)
            s_five = ' ' if not five_col else ' d' + \
                random.choice(s_names) + ' '
            f.write('x%sx %s %f\n' % (s_five, s_name, i))
        for i in neg:
            s_name = random.choice(s_names)
            s_five = ' ' if not five_col else ' d' + \
                random.choice(s_names) + ' '
            f.write('x%sy %s %f\n' % (s_five, s_name, i))


@click.command()
@click.argument('outdir')
@click.option('-mm', '--mean-match', default=10, type=FLOAT, show_default=True)
@click.option('-mnm', '--mean-non-match', default=-10, type=FLOAT, show_default=True)
@click.option('-n', '--n-sys', default=1, type=click.INT, show_default=True)
@click.option('--five-col/--four-col', default=False, show_default=True)
@verbosity_option()
def gen(outdir, mean_match, mean_non_match, n_sys, five_col):
    """Generate random scores.
    Generates random scores in 4col or 5col format. The scores are generated
    using Gaussian distribution whose mean is an input
    parameter. The generated scores can be used as hypothetical datasets.
    """
    # Generate the data
    neg_dev, pos_dev = gen_score_distr(mean_non_match, mean_match)
    neg_eval, pos_eval = gen_score_distr(mean_non_match, mean_match)

    # Write the data into files
    write_scores_to_file(neg_dev, pos_dev,
                         os.path.join(outdir, 'scores-dev'), n_sys, five_col)
    write_scores_to_file(neg_eval, pos_eval,
                         os.path.join(outdir, 'scores-eval'), n_sys, five_col)
