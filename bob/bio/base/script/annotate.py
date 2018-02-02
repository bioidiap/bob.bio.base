"""A script to help annotate databases.
"""
import logging
import json
import click
from os.path import dirname, isfile
from bob.extension.scripts.click_helper import (
    verbosity_option, Command, Option)
from bob.io.base import create_directories_safe
from bob.bio.base.tools.grid import indices

logger = logging.getLogger(__name__)


@click.command(entry_point_group='bob.bio.config', cls=Command)
@click.option('--database', '-d', required=True, cls=Option,
              entry_point_group='bob.bio.database')
@click.option('--annotator', '-a', required=True, cls=Option,
              entry_point_group='bob.bio.annotator')
@click.option('--output-dir', '-o', required=True, cls=Option)
@click.option('--force', '-f', is_flag=True, cls=Option)
@click.option('--array', type=click.INT, default=1,)
@verbosity_option(cls=Option)
def annotate(database, annotator, output_dir, force, array, **kwargs):
    """Annotates a database.
    The annotations are written in text file (json) format which can be read
    back using :any:`bob.db.base.read_annotation_file` (annotation_type='json')

    \b
    Parameters
    ----------
    database : :any:`bob.bio.database`
        The database that you want to annotate. Can be a ``bob.bio.database``
        entry point or a path to a Python file which contains a variable
        named `database`.
    annotator : callable
        A function that takes the database and a sample (biofile) of the
        database and returns the annotations in a dictionary. Can be a
        ``bob.bio.annotator`` entry point or a path to a Python file which
        contains a variable named `annotator`.
    output_dir : str
        The directory to save the annotations.
    force : bool, optional
        Wether to overwrite existing annotations.
    array : int, optional
        Use this option alongside gridtk to submit this script as an array job.
    verbose : int, optional
        Increases verbosity (see help for --verbose).

    \b
    [CONFIG]...            Configuration files. It is possible to pass one or
                           several Python files (or names of ``bob.bio.config``
                           entry points) which contain the parameters listed
                           above as Python variables. The options through the
                           command-line (see below) will override the values of
                           configuration files.
    """
    logger.debug('database: %s', database)
    logger.debug('annotator: %s', annotator)
    logger.debug('force: %s', force)
    logger.debug('output_dir: %s', output_dir)
    logger.debug('array: %s', array)
    logger.debug('kwargs: %s', kwargs)

    biofiles = database.objects(groups=None, protocol=database.protocol)
    biofiles = sorted(biofiles)

    if array > 1:
        start, end = indices(biofiles, array)
        biofiles = biofiles[start:end]

    total = len(biofiles)
    logger.info("Saving annotations in %s", output_dir)
    logger.info("Annotating %d samples ...", total)

    for i, biofile in enumerate(biofiles):
        outpath = biofile.make_path(output_dir, '.json')
        if isfile(outpath):
            if force:
                logger.debug("Overwriting the annotations file `%s'", outpath)
            else:
                logger.debug("The annotation `%s' already exists", outpath)
                continue

        logger.info(
            "Extracting annotations for sample %d out of %d", i + 1, total)
        data = annotator.read_original_data(
            biofile, database.original_directory, database.original_extension)
        annot = annotator(data)

        create_directories_safe(dirname(outpath))
        with open(outpath, 'w') as f:
            json.dump(annot, f, indent=1, allow_nan=False)
