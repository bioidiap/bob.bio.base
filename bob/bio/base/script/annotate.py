"""A script to help annotate databases.
"""
import logging
import click
import json
import functools
from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    log_parameters,
)
from bob.pipelines import wrap, ToDaskBag, DelayedSample
logger = logging.getLogger(__name__)

def save_json(data, path):
    """
    Saves a dictionnary ``data`` in a json file at ``path``.
    """
    with open(path, "w") as f:
        json.dump(data, f)

def load_json(path):
    """
    Returns a dictionnary from a json file at ``path``.
    """
    with open(path, "r") as f:
        return json.load(f)

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
    database, groups, annotator, output_dir, dask_client, **kwargs
):
    """Annotates a database.

    The annotations are written in text file (json) format which can be read
    back using :any:`bob.db.base.read_annotation_file` (annotation_type='json')
    """
    log_parameters(logger)

    # Allows passing of Sample objects as parameters
    annotator = wrap(["sample"], annotator, output_attribute="annotations")

    # Will save the annotations in the `data` fields to a json file
    annotator = wrap(
        bases=["checkpoint"],
        estimator=annotator,
        features_dir=output_dir,
        extension=".json",
        save_func=save_json,
        load_func=load_json,
        sample_attribute="annotations",
    )

    # Allows reception of Dask Bags
    annotator = wrap(["dask"], annotator)

    # Transformer that splits the samples into several Dask Bags
    to_dask_bags = ToDaskBag(npartitions=50)


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
    samples.extend([
        sample
        for r in references_samplesets
        for sample in r.samples
    ])
    samples.extend([
        sample
        for p in probes_samplesets
        for sample in p.samples
    ])

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

You have to define ``samples``, ``reader``, and ``make_key`` in python files
(config.py) as in examples.
""",
)
@click.option(
    "--samples",
    entry_point_group="bob.bio.config",
    required=True,
    cls=ResourceOption,
    help="A list of all samples that you want to annotate. They will be passed "
    "as is to the ``reader`` and ``make-key`` functions.",
)
@click.option(
    "--reader",
    required=True,
    cls=ResourceOption,
    help="A function with the signature of ``data = reader(sample)`` which "
    "takes a sample and returns the loaded data. The returned data is given to "
    "the annotator.",
)
@click.option(
    "--make-key",
    required=True,
    cls=ResourceOption,
    help="A function with the signature of ``key = make_key(sample)`` which "
    "takes a sample and returns a unique str identifier for that sample that "
    "will be use to save it in output_dir. ``key`` generally is the relative "
    "path to a sample's file from the dataset's root directory.",
)
@annotate_common_options
@verbosity_option(cls=ResourceOption)
def annotate_samples(
    samples, reader, make_key, annotator, output_dir, dask_client, **kwargs
):
    """Annotates a list of samples.

    This command is very similar to ``bob bio annotate`` except that it works
    without a database interface. You must provide a list of samples as well as
    two functions:

        def reader(sample):
            # Loads data from a sample.
            # for example:
            data = bob.io.base.load(sample)
            # data will be given to the annotator
            return data

        def make_key(sample):
            # Creates a unique str identifier for this sample.
            # for example:
            return str(sample)
    """
    log_parameters(logger, ignore=("samples",))

    # Allows passing of Sample objects as parameters
    annotator = wrap(["sample"], annotator, output_attribute="annotations")

    # Will save the annotations in the `data` fields to a json file
    annotator = wrap(
        bases=["checkpoint"],
        estimator=annotator,
        features_dir=output_dir,
        extension=".json",
        save_func=save_json,
        load_func=load_json,
        sample_attribute="annotations",
    )

    # Allows reception of Dask Bags
    annotator = wrap(["dask"], annotator)

    # Transformer that splits the samples into several Dask Bags
    to_dask_bags = ToDaskBag(npartitions=50)

    if dask_client is not None:
        scheduler=dask_client
    else:
        scheduler="single-threaded"

    # Converts samples into a list of DelayedSample objects
    samples_obj = [
        DelayedSample(
            load=functools.partial(reader,s),
            key=make_key(s),
        )
        for s in samples
    ]
    # Splits the samples list into bags
    dask_bags = to_dask_bags.transform(samples_obj)

    logger.info(f"Saving annotations in {output_dir}")
    logger.info(f"Annotating {len(samples_obj)} samples...")
    annotator.transform(dask_bags).compute(scheduler=scheduler)

    if dask_client is not None:
        logger.info("Shutdown workers...")
        dask_client.shutdown()
    logger.info("Done.")
