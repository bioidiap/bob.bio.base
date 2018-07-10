"""A script to help annotate databases.
"""
import logging
import json
import click
from os.path import dirname, isfile, expanduser
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption, log_parameters)
from bob.io.base import create_directories_safe
from bob.bio.base.tools.grid import indices

logger = logging.getLogger(__name__)


ANNOTATE_EPILOG = '''\b
Examples:

  $ bob bio annotate -vvv -d <database> -a <annotator> -o /tmp/annotations
  $ jman submit --array 64 -- bob bio annotate ... --array 64
'''


@click.command(entry_point_group='bob.bio.config', cls=ConfigCommand,
               epilog=ANNOTATE_EPILOG)
@click.option('--database', '-d', required=True, cls=ResourceOption,
              entry_point_group='bob.bio.database',
              help='''The database that you want to annotate.''')
@click.option('--annotator', '-a', required=True, cls=ResourceOption,
              entry_point_group='bob.bio.annotator',
              help='A callable that takes the database and a sample (biofile) '
              'of the database and returns the annotations in a dictionary.')
@click.option('--output-dir', '-o', required=True, cls=ResourceOption,
              help='The directory to save the annotations.')
@click.option('--force', '-f', is_flag=True, cls=ResourceOption,
              help='Whether to overwrite existing annotations.')
@click.option('--array', type=click.INT, default=1, cls=ResourceOption,
              help='Use this option alongside gridtk to submit this script as '
              'an array job.')
@click.option('--database-directories-file', cls=ResourceOption,
              default=expanduser('~/.bob_bio_databases.txt'),
              help='(Deprecated) To support loading of old databases.')
@verbosity_option(cls=ResourceOption)
def annotate(database, annotator, output_dir, force, array,
             database_directories_file, **kwargs):
    """Annotates a database.

    The annotations are written in text file (json) format which can be read
    back using :any:`bob.db.base.read_annotation_file` (annotation_type='json')
    """
    log_parameters(logger)

    # Some databases need their original_directory to be replaced
    database.replace_directories(database_directories_file)

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
                logger.info("Overwriting the annotations file `%s'", outpath)
            else:
                logger.info("The annotation `%s' already exists", outpath)
                continue

        logger.info(
            "Extracting annotations for sample %d out of %d: %s", i + 1, total,
            outpath)
        data = annotator.read_original_data(
            biofile, database.original_directory, database.original_extension)
        annot = annotator(data)

        create_directories_safe(dirname(outpath))
        with open(outpath, 'w') as f:
            json.dump(annot, f, indent=1, allow_nan=False)
