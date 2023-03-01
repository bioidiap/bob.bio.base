"""Sorts score files based on their score value
"""
import logging

import click
import numpy

from clapper.click import log_parameters, verbosity_option

from bob.bio.base.score.load import dump_score, load_score

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
@verbosity_option(logger=logger)
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
