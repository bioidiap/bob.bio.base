#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes only the train part of a biometric pipeline"""

import logging

import click

from bob.extension.scripts.click_helper import (
    ConfigCommand,
    ResourceOption,
    verbosity_option,
)
from bob.pipelines.distributed import VALID_DASK_CLIENT_STRINGS

logger = logging.getLogger(__name__)


EPILOG = """\b

Command line examples\n
-----------------------

$ bob bio pipeline train -vv DATABASE PIPELINE

See the help of the CONFIG argument on top of this help message
for a list of available configurations.

It is possible to provide database and pipeline through a configuration file.
Generate an example configuration file with:

$ bob bio pipeline train --dump-config my_experiment.py

and execute it with:

$ bob bio pipeline train -vv my_experiment.py

my_experiment.py must contain the following elements:

   >>> transformer = ... # A scikit-learn pipeline wrapped with bob.pipelines' SampleWrapper\n
   >>> pipeline = PipelineTrain(transformer)\n
   >>> database = .... # Biometric Database (class that implements the methods: `background_model_samples`, `references` and `probes`)"
\b"""


@click.command(
    name="train",
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
    "--output",
    "-o",
    show_default=True,
    default="results",
    help="Name of output directory where output files will be saved.",
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
    help="Name of output directory where the checkpoints will be saved. In case --memory is not set, checkpoints will be saved in this directory.",
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
@click.option(
    "--split-training",
    is_flag=True,
    help="EXPERIMENTAL. Splits the training set in partitions and trains the pipeline in multiple steps.",
    cls=ResourceOption,
)
@click.option(
    "--n-splits",
    default=3,
    help="EXPERIMENTAL. Number of partitions to split the training set in. "
    "Each partition will be trained in a separate step.",
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def pipeline_train(
    pipeline,
    database,
    dask_client,
    output,
    memory,
    checkpoint_dir,
    dask_partition_size,
    dask_n_workers,
    dask_n_partitions,
    force,
    no_dask,
    split_training,
    n_splits,
    **kwargs,
):
    """Runs the training part of a biometrics pipeline.

    Such pipeline consists only of one component, contrary to the ``simple`` pipeline.
    The component is a scikit-learn `Pipeline`,
    where a sequence of transformations of the input data
    is defined.

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

    from bob.bio.base.pipelines import execute_pipeline_train

    if no_dask:
        dask_client = None

    checkpoint = not memory

    logger.debug("Executing PipelineSimple with:")
    logger.debug(f"pipeline: {pipeline}")
    logger.debug(f"  transformer: {pipeline.transformer}")
    logger.debug(f"  biometric_algorithm: {pipeline.biometric_algorithm}")
    logger.debug(f"database: {database}")

    execute_pipeline_train(
        pipeline=pipeline,
        database=database,
        dask_client=dask_client,
        output=output,
        checkpoint=checkpoint,
        dask_partition_size=dask_partition_size,
        dask_n_partitions=dask_n_partitions,
        dask_n_workers=dask_n_workers,
        checkpoint_dir=checkpoint_dir,
        force=force,
        split_training=split_training,
        n_splits=n_splits,
    )

    logger.info(f"Experiment finished ! ({output=})")
