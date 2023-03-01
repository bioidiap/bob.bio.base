#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes biometric pipeline"""

import logging

import click

from clapper.click import ConfigCommand, ResourceOption, verbosity_option

from bob.pipelines.distributed import VALID_DASK_CLIENT_STRINGS

from .pipeline_simple import EPILOG as _SIMPLE_EPILOG

logger = logging.getLogger(__name__)


EPILOG = _SIMPLE_EPILOG.replace(
    "bob bio pipeline simple", "bob bio pipeline score-norm"
)


@click.command(
    name="score-norm",
    entry_point_group="bob.bio.config",
    cls=ConfigCommand,
    epilog=EPILOG,
)
@click.option(
    "--pipeline",
    "-p",
    entry_point_group="bob.bio.pipeline",
    required=True,
    help="PipelineSimple composed of a scikit-learn Pipeline and a BioAlgorithm",
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
    "-o",
    "--output",
    show_default=True,
    default="results",
    help="Name of output directory where output scores will be saved. In case --checkpoint is set, checkpoints will be saved in this directory.",
    cls=ResourceOption,
)
@click.option(
    "--write-metadata-scores/--write-column-scores",
    "-m/-nm",
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
    "--dask-partition-size",
    "-s",
    help="If using Dask, this option defines the size of each dask.bag.partition."
    "Use this option if the current heuristic that sets this value doesn't suit your experiment."
    "(https://docs.dask.org/en/latest/bag-api.html?highlight=partition_size#dask.bag.from_sequence).",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@click.option(
    "--dask-n-workers",
    "-n",
    help="If using Dask, this option defines the number of workers to start your experiment."
    "Dask automatically scales up/down the number of workers due to the current load of tasks to be solved."
    "Use this option if the current amount of workers set to start an experiment doesn't suit you.",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@click.option(
    "-c",
    "--checkpoint-dir",
    show_default=True,
    default=None,
    help="Name of output directory where the checkpoints will be saved. In case --checkpoint is set, checkpoints will be saved in this directory.",
    cls=ResourceOption,
)
@click.option(
    "--top-norm",
    "-t",
    is_flag=True,
    help="If set, it will do the top-norm.",
    cls=ResourceOption,
)
@click.option(
    "--top-norm-score-fraction",
    default=1.0,
    type=float,
    help="Sets the percentage of samples used for t-norm and z-norm. Sometimes you don't want to use all the t/z samples for normalization",
    cls=ResourceOption,
)
@click.option(
    "--score-normalization-type",
    "-nt",
    type=click.Choice(["znorm", "tnorm"]),
    multiple=False,
    default="znorm",
    help="Type of normalization",
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
def pipeline_score_norm(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    memory,
    dask_partition_size,
    dask_n_workers,
    checkpoint_dir,
    top_norm,
    top_norm_score_fraction,
    score_normalization_type,
    force,
    no_dask,
    **kwargs,
):
    """Runs the PipelineSimple with score normalization strategies

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

    """
    from bob.bio.base.pipelines.entry_points import execute_pipeline_score_norm

    logger.debug("Executing PipelineScoreNorm")

    if no_dask:
        dask_client = None

    checkpoint = not memory

    execute_pipeline_score_norm(
        pipeline,
        database,
        dask_client,
        groups,
        output,
        write_metadata_scores,
        checkpoint,
        dask_partition_size,
        dask_n_workers,
        checkpoint_dir,
        top_norm,
        top_norm_score_fraction,
        score_normalization_type,
        force,
    )

    logger.info("Experiment finished !")
