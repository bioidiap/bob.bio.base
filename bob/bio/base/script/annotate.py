"""A script to help annotate databases.
"""
import logging
import click
import functools
from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    log_parameters,
)
from bob.pipelines import wrap, ToDaskBag
logger = logging.getLogger(__name__)

def annotate_common_options(func):
    @click.option(
        "--annotator",
        "-a",
        required=True,
        cls=ResourceOption,
        entry_point_group="bob.bio.annotator",
        help="A Transformer instance that takes a series of sample and returns "
        "the modified samples with annotations as a dictionary.",
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
    @click.option(
        "--dask-client",
        "-l",
        "dask_client",
        entry_point_group="dask.client",
        help="Dask client for the execution of the pipeline. If not specified, "
            "uses a single threaded, local Dask Client.",
        cls=ResourceOption,
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
    help="Biometric Database (class that implements the methods: "
        "`background_model_samples`, `references` and `probes`).",
)
@click.option(
    "--groups",
    "-g",
    multiple=True,
    default=["dev", "eval"],
    show_default=True,
    help="Biometric Database group that will be annotated.",
)
@annotate_common_options
@verbosity_option(cls=ResourceOption)
def annotate(
    database, groups, annotator, output_dir, force, dask_client, **kwargs
):
    """Annotates a database.

    The annotations are written in text file (json) format which can be read
    back using :any:`bob.db.base.read_annotation_file` (annotation_type='json')
    """
    log_parameters(logger)

    # Allows passing of Sample objects as parameters
    annotator = wrap(["annotated_sample"], annotator)

    # Will save the annotations in the `data` fields to a json file
    annotator = wrap(
        bases=["checkpoint_annotations"],
        annotator=annotator,
        annotations_dir=output_dir,
        force=force,
    )

    # Allows reception of Dask Bags
    annotator = wrap(["dask"], annotator)

    # Transformer that splits the samples into several Dask Bags
    to_dask_bags = ToDaskBag()


    logger.debug("Retrieving background model samples from database.")
    background_model_samples = database.background_model_samples()

    logger.debug("Retrieving references and probes samples from database.")
    references_samplesets = []
    probes_samplesets = []
    for group in groups:
        references_samplesets.extend(database.references(group=group))
        probes_samplesets.extend(database.probes(group=group))

    # Unravels all samples in one list (no SampleSets)
    samples = background_model_samples
    samples.extend([sample for r in references_samplesets for sample in r.samples])
    samples.extend([sample for p in probes_samplesets for sample in p.samples])

    # Sets the scheduler to local if no dask_client is specified
    if dask_client is not None:
        scheduler=dask_client
    else:
        scheduler="single-threaded"

    logger.info(f"Saving annotations in {output_dir}.")
    logger.info(f"Annotating {len(samples)} samples...")
    dask_bags = to_dask_bags.transform(samples)
    annotator.transform(dask_bags).compute(scheduler=scheduler)

    if dask_client is not None:
        logger.info("Shutdown workers...")
        dask_client.shutdown()
    logger.info("Done.")


@click.command(
    entry_point_group="bob.bio.config",
    cls=ConfigCommand,
    epilog="""\b
Examples:

  $ bob bio annotate-samples -vvv config.py -a <annotator> -o /tmp/annotations

You have to define ``samples`` in a python file (config.py) as in examples.
""",
)
@click.option(
    "--samples",
    entry_point_group="bob.bio.config",
    required=True,
    cls=ResourceOption,
    help="A list of all samples that you want to annotate.",
)
@annotate_common_options
@verbosity_option(cls=ResourceOption)
def annotate_samples(
    samples, annotator, output_dir, force, dask_client, **kwargs
):
    """Annotates a list of samples.

    This command is very similar to ``bob bio annotate`` except that it works
    without a database interface.
    """
    log_parameters(logger, ignore=("samples",))

    # Allows passing of Sample objects as parameters
    annotator = wrap(["annotated_sample"], annotator)

    # Will save the annotations in the `data` fields to a json file
    annotator = wrap(
        bases=["checkpoint_annotations"],
        annotator=annotator,
        annotations_dir=output_dir,
        force=force,
    )

    # Allows reception of Dask Bags
    annotator = wrap(["dask"], annotator)

    # Transformer that splits the samples into several Dask Bags
    to_dask_bags = ToDaskBag()

    if dask_client is not None:
        scheduler=dask_client
    else:
        scheduler="single-threaded"

    logger.info(f"Saving annotations in {output_dir}")
    logger.info(f"Annotating {len(samples)} samples...")
    dask_bags = to_dask_bags.transform(samples)
    annotator.transform(dask_bags).compute(scheduler=scheduler)

    if dask_client is not None:
        logger.info("Shutdown workers...")
        dask_client.shutdown()
    logger.info("Done.")
