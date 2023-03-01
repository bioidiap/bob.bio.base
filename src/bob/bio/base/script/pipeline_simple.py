#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes biometric pipeline"""

import logging

import click

from clapper.click import ConfigCommand, ResourceOption, verbosity_option

from bob.pipelines.distributed import VALID_DASK_CLIENT_STRINGS

logger = logging.getLogger(__name__)


EPILOG = """\b

Command line examples\n
-----------------------

$ bob bio pipeline simple -vv DATABASE PIPELINE

See the help of the CONFIG argument on top of this help message
for a list of available configurations.

It is possible to provide database and pipeline through a configuration file.
Generate an example configuration file with:

$ bob bio pipeline simple --dump-config my_experiment.py

and execute it with:

$ bob bio pipeline simple -vv my_experiment.py

my_experiment.py must contain the following elements:

   >>> transformer = ... # A scikit-learn pipeline wrapped with bob.pipelines' SampleWrapper\n
   >>> algorithm   = ... # `An BioAlgorithm`\n
   >>> pipeline = PipelineSimple(transformer,algorithm)\n
   >>> database = .... # Biometric Database (class that implements the methods: `background_model_samples`, `references` and `probes`)"
\b"""


@click.command(
    name="simple",
    entry_point_group="bob.bio.config",
    cls=ConfigCommand,
    epilog=EPILOG,
)
@click.option(
    "--pipeline",
    "-p",
    required=True,
    entry_point_group="bob.bio.pipeline",
    help="The simplest pipeline possible composed of a scikit-learn Pipeline and a BioAlgorithm",
    cls=ResourceOption,
)
@click.option(
    "--database",
    "-d",
    entry_point_group="bob.bio.database",
    required=True,
    help="Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)",
    cls=ResourceOption,
)
@click.option(
    "--dask-client",
    "-l",
    entry_point_group="dask.client",
    string_exceptions=VALID_DASK_CLIENT_STRINGS,
    default="single-threaded",
    help="Dask client for the execution of the pipeline.",
    cls=ResourceOption,
)
@click.option(
    "--group",
    "-g",
    "groups",
    type=click.Choice(["dev", "eval"]),
    multiple=True,
    default=("dev",),
    help="If given, this value will limit the experiments belonging to a particular protocolar group",
    cls=ResourceOption,
)
@click.option(
    "--output",
    "-o",
    show_default=True,
    default="results",
    help="Name of output directory where output scores will be saved.",
    cls=ResourceOption,
)
@click.option(
    "--write-metadata-scores/--write-column-scores",
    "-meta/-nmeta",
    default=True,
    help="If set, all the scores will be written with all their metadata using the `CSVScoreWriter`",
    cls=ResourceOption,
)
@click.option(
    "--memory",
    "-m",
    is_flag=True,
    help="If set, it will run the experiment keeping all objects on memory with nothing checkpointed. If not set, checkpoints will be saved in `--output`.",
    cls=ResourceOption,
)
@click.option(
    "--checkpoint-dir",
    "-c",
    show_default=True,
    default=None,
    help="Name of output directory where the checkpoints will be saved. In case --checkpoint is set, checkpoints will be saved in this directory.",
    cls=ResourceOption,
)
@click.option(
    "--dask-partition-size",
    "-s",
    help="If using Dask, this option defines the max size of each dask.bag.partition. "
    "Use this option if the current heuristic that sets this value doesn't suit your experiment. "
    "(https://docs.dask.org/en/latest/bag-api.html?highlight=partition_size#dask.bag.from_sequence).",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@click.option(
    "--dask-n-partitions",
    "-n",
    help="If using Dask, this option defines a fixed number of dask.bag.partition for "
    "each set of data. Use this option if the current heuristic that sets this value "
    "doesn't suit your experiment."
    "(https://docs.dask.org/en/latest/bag-api.html?highlight=partition_size#dask.bag.from_sequence).",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@click.option(
    "--dask-n-workers",
    "-w",
    help="If using Dask, this option defines the number of workers to start your experiment. "
    "Dask automatically scales up/down the number of workers due to the current load of tasks to be solved. "
    "Use this option if the current amount of workers set to start an experiment doesn't suit you.",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="If set, it will force generate all the checkpoints of an experiment. This option doesn't work if `--memory` is set",
    cls=ResourceOption,
)
@click.option(
    "--no-dask",
    is_flag=True,
    help="If set, it will not use Dask to run the experiment.",
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption, logger=logger)
def pipeline_simple(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    memory,
    checkpoint_dir,
    dask_partition_size,
    dask_n_workers,
    dask_n_partitions,
    force,
    no_dask,
    **kwargs,
):
    """Runs the simplest biometrics pipeline.

    Such pipeline consists into two major components.
    The first component consists of a scikit-learn `Pipeline`,
    where a sequence of transformations of the input data
    is defined.
    The second component is a `BioAlgorithm` that defines the primitives
    `enroll` and `score`

    With those two components any Biometric Experiment can be done.
    A Biometric experiment consists of three sub-pipelines and
    they are defined below:

    Sub-pipeline 1:\n
    ---------------

    Training background model.
    Some biometric algorithms demands the training of background model, for instance a neural network.

    \b
    This pipeline runs: `Pipeline.fit(DATA_FOR_FIT)`



    \b

    Sub-pipeline 2:\n
    ---------------

    Creation of biometric references: This is a standard step in a biometric pipelines.
    Given a set of samples of one identity, create a biometric reference (a.k.a template) for sub identity.


    \b
    raw_data --> preprocessing >> feature extraction >> enroll(background_model) --> biometric_reference

    This pipeline runs: `BioAlgorithm.enroll(Pipeline.transform(DATA_ENROLL))` >> biometric_references


    Sub-pipeline 3:\n
    ---------------

    Probing: This is another standard step in biometric pipelines.
    Given one sample and one biometric reference, computes a score.
    Such score has different meanings depending on the scoring method your biometric algorithm uses.
    It's out of scope to explain in a help message to explain what scoring is for different biometric algorithms.

    This pipeline runs: `BioAlgorithm.score(Pipeline.transform(DATA_SCORE, biometric_references))` >> biometric_references

    .. Note::
        Refrain from calling this function directly from a script. Prefer
        :py:func:`bob.bio.base.pipelines.execute_pipeline_simple`
        instead.


    Using Dask
    ----------

    This pipeline is intended to work with Dask to split the load of work between
    processes on a machine or workers on a distributed grid system. By default, the
    local machine is used in single-threaded mode. However, by specifying the
    `--dask-client` option, you specify a Dask Client.

    When using multiple workers, a few things have to be considered:
    - The number of partitions in the data.
    - The number of workers to process the data.

    Ideally, (and this is the default behavior) you want to split all the data between
    many available workers, and all the workers work at the same time on all the data.
    But the number of workers may be limited, or one partition of data may be filling
    the memory of one worker. Moreover, having many small tasks (by splitting the data
    into many partitions) is not recommended as the scheduler will then spend more time
    organizing and communicating with the workers.

    To solve speed or memory issues, options are available to split the data
    differently (`--dask-n-partitions` or `--dask-partition-size`). If you encounter
    memory issues on a worker, try augmenting the number of partitions, and if your
    scheduler is not keeping up, try reducing that number.
    """
    from bob.bio.base.pipelines import execute_pipeline_simple

    if no_dask:
        dask_client = None

    checkpoint = not memory

    logger.debug("Executing PipelineSimple with:")
    logger.debug(f"pipeline: {pipeline}")
    logger.debug(f"  transformer: {pipeline.transformer}")
    logger.debug(f"  biometric_algorithm: {pipeline.biometric_algorithm}")
    logger.debug(f"database: {database}")

    execute_pipeline_simple(
        pipeline=pipeline,
        database=database,
        dask_client=dask_client,
        groups=groups,
        output=output,
        write_metadata_scores=write_metadata_scores,
        checkpoint=checkpoint,
        dask_partition_size=dask_partition_size,
        dask_n_partitions=dask_n_partitions,
        dask_n_workers=dask_n_workers,
        checkpoint_dir=checkpoint_dir,
        force=force,
    )

    logger.info("Experiment finished !")
