"""Sorts score files based on their score value
"""
import click
import logging
import numpy
from bob.bio.base.score.load import load_score, dump_score
from bob.extension.scripts.click_helper import verbosity_option, log_parameters

logger = logging.getLogger(__name__)


@click.command(
    epilog="""\b
Examples:

  $ bob bio sort -vvv /path/to/scores
"""
)
@click.argument(
    "score_paths",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=True),
    nargs=-1,
)
@verbosity_option()
def sort(score_paths, **kwargs):
    """Sorts score files based on their score values

    The conversion happens in-place; backup your scores before using this script
    """
    log_parameters(logger)

    for path in score_paths:
        logger.info("Sorting: %s", path)
        scores = load_score(path)
        scores = scores[numpy.argsort(scores["score"])]
        dump_score(path, scores)
