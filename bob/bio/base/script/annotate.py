"""A script to help annotate databases.
"""
import logging
import json
import click
import functools
from os.path import dirname, isfile, expanduser
from os import makedirs
from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    log_parameters,
)

logger = logging.getLogger(__name__)


def annotate_common_options(func):
    @click.option(
        "--annotator",
        "-a",
        required=True,
        cls=ResourceOption,
        entry_point_group="bob.bio.annotator",
        help="A callable that takes the database and a sample (biofile) "
        "of the database and returns the annotations in a dictionary.",
    )
    @click.option(
        "--output-dir",
        "-o",
        required=True,
        cls=ResourceOption,
        help="The directory to save the annotations.",
    )
    @click.option(
        "--force",
        "-f",
        is_flag=True,
        cls=ResourceOption,
        help="Whether to overwrite existing annotations.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        return func(*args, **kwds)

    return wrapper


@click.command(
    entry_point_group="bob.bio.config",
    cls=ConfigCommand,
    epilog="""\b
Examples:

  $ bob bio annotate -vvv -d <database> -a <annotator> -o /tmp/annotations
""",
)
@click.option(
    "--database",
    "-d",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.bio.database",
    help="""The database that you want to annotate.""",
)
@annotate_common_options
@click.option(
    "--database-directories-file",
    cls=ResourceOption,
    default=expanduser("~/.bob_bio_databases.txt"),
    help="(Deprecated) To support loading of old databases.",
)
@verbosity_option(cls=ResourceOption)
def annotate(
    database, annotator, output_dir, force, database_directories_file, **kwargs
):
    """Annotates a database.

    The annotations are written in text file (json) format which can be read
    back using :any:`bob.db.base.read_annotation_file` (annotation_type='json')
    """
    log_parameters(logger)

    # Some databases need their original_directory to be replaced
    database.replace_directories(database_directories_file)

    biofiles = database.objects(groups=None, protocol=database.protocol)
    samples = sorted(biofiles)

    def reader(biofile):
        return annotator.read_original_data(
            biofile, database.original_directory, database.original_extension
        )

    def make_path(biofile, output_dir):
        return biofile.make_path(output_dir, ".json")

    return annotate_generic(samples, reader, make_path, annotator, output_dir, force)


@click.command(
    entry_point_group="bob.bio.config",
    cls=ConfigCommand,
    epilog="""\b
Examples:

  $ bob bio annotate-samples -vvv config.py -a <annotator> -o /tmp/annotations

You have to define samples, reader, and make_path in a python file (config.py) as in
examples.
""",
)
@click.option(
    "--samples",
    required=True,
    cls=ResourceOption,
    help="A list of all samples that you want to annotate. The list must be sorted or "
    "deterministic in consequent calls. This is needed so that this script works "
    "correctly on the grid.",
)
@click.option(
    "--reader",
    required=True,
    cls=ResourceOption,
    help="A function with the signature of ``data = reader(sample)`` which takes a "
    "sample and returns the loaded data. The data is given to the annotator.",
)
@click.option(
    "--make-path",
    required=True,
    cls=ResourceOption,
    help="A function with the signature of ``path = make_path(sample, output_dir)`` "
    "which takes a sample and output_dir and returns the unique path for that sample "
    "to be saved in output_dir. The extension of the path must be '.json'.",
)
@annotate_common_options
@verbosity_option(cls=ResourceOption)
def annotate_samples(
    samples, reader, make_path, annotator, output_dir, force, **kwargs
):
    """Annotates a list of samples.

    This command is very similar to ``bob bio annotate`` except that it works without a
    database interface. You only need to provide a list of **sorted** samples to be
    annotated and two functions::

        def reader(sample):
            # load data from sample here
            # for example:
            data = bob.io.base.load(sample)
            # data will be given to the annotator
            return data

        def make_path(sample, output_dir):
            # create a unique path for this sample in the output_dir
            # for example:
            return os.path.join(output_dir, str(sample) + ".json")

    Please note that your samples must be a list and must be sorted!
    """
    log_parameters(logger, ignore=("samples",))
    logger.debug("len(samples): %d", len(samples))
    return annotate_generic(samples, reader, make_path, annotator, output_dir, force)


def annotate_generic(samples, reader, make_path, annotator, output_dir, force):
    total = len(samples)
    logger.info("Saving annotations in %s", output_dir)
    logger.info("Annotating %d samples ...", total)

    for i, sample in enumerate(samples):
        outpath = make_path(sample, output_dir)
        if not outpath.endswith(".json"):
            outpath += ".json"

        if isfile(outpath):
            if force:
                logger.info("Overwriting the annotations file `%s'", outpath)
            else:
                logger.info("The annotation `%s' already exists", outpath)
                continue

        logger.info(
            "Extracting annotations for sample %d out of %d: %s", i + 1, total, outpath
        )
        data = reader(sample)
        annot = annotator(data)

        makedirs(dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(annot, f, indent=1, allow_nan=False)
