#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes biometric pipeline"""

import click

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ResourceOption,
    ConfigCommand,
)

import logging
import os
import itertools
import dask.bag
from bob.bio.base.pipelines.vanilla_biometrics import (
    VanillaBiometricsPipeline,
    BioAlgorithmCheckpointWrapper,
    BioAlgorithmDaskWrapper,
    checkpoint_vanilla_biometrics,
    dask_vanilla_biometrics,
    dask_get_partition_size,
    FourColumnsScoreWriter,
    CSVScoreWriter,
    BioAlgorithmLegacy,
    is_checkpointed,
)
from dask.delayed import Delayed
import pkg_resources
from bob.extension.config import load as chain_load
from bob.pipelines.utils import isinstance_nested
from bob.bio.base.utils import get_resource_filename


logger = logging.getLogger(__name__)


EPILOG = """\b


 Command line examples\n
 -----------------------


 $ bob pipelines vanilla-biometrics -p my_experiment.py -vv


 my_experiment.py must contain the following elements:

   >>> transformer = ... # A scikit-learn pipeline\n
   >>> algorithm   = ... # `An BioAlgorithm`\n
   >>> pipeline = VanillaBiometricsPipeline(transformer,algorithm)\n
   >>> database = .... # Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)"

\b

"""


def compute_scores(result, dask_client):
    if isinstance(result, Delayed) or isinstance(result, dask.bag.Bag):
        if dask_client is not None:
            result = result.compute(scheduler=dask_client)
        else:
            logger.warning("`dask_client` not set. Your pipeline will run locally")
            result = result.compute(scheduler="single-threaded")
    return result


def post_process_scores(pipeline, scores, path):
    writed_scores = pipeline.write_scores(scores)
    return pipeline.post_process(writed_scores, path)


def load_database_pipeline(database, pipeline):
    # It's necessary to chain load 2 resources together
    pipeline_config = get_resource_filename(pipeline, "bob.bio.pipeline")

    if database is None:
        vanilla_pipeline = chain_load([pipeline_config])
        if hasattr(vanilla_pipeline, "database"):
            return vanilla_pipeline.database, vanilla_pipeline.pipeline
        else:
            raise ValueError(
                "Database was not set. Please look in `bob bio pipelines vanilla-biometrics --help` for more information"
            )
    else:
        database_config = get_resource_filename(database, "bob.bio.database")
        vanilla_pipeline = chain_load([database_config, pipeline_config])
        return vanilla_pipeline.database, vanilla_pipeline.pipeline


@click.command(
    entry_point_group="bob.bio.config", cls=ConfigCommand, epilog=EPILOG,
)
@click.option(
    "--pipeline",
    "-p",
    required=True,
    entry_point_group="bob.bio.pipeline"
    help="Vanilla biometrics pipeline composed of a scikit-learn Pipeline and a BioAlgorithm",
    cls=ResourceOption,
)
@click.option(
    "--database",
    "-d",
    entry_point_group="bob.bio.database"
    required=False,
    help="Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)",
    cls=ResourceOption,
)
@click.option(
    "--dask-client",
    "-l",
    required=False,
    entry_point_group="dask.client",
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
    "--write-metadata-scores",
    "-m",
    is_flag=True,
    help="If set, all the scores will be written with all its metadata using the `CSVScoreWriter`",
    cls=ResourceOption,
)
@click.option(
    "--checkpoint",
    "-c",
    is_flag=True,
    help="If set, it will checkpoint all steps of the pipeline. Checkpoints will be saved in `--output`.",
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def vanilla_biometrics(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    checkpoint,
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
    Some biometric algorithms demands the training of background model, for instance, PCA/LDA matrix or a Neural networks.

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

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    # Picking the resources
    # database, pipeline = load_database_pipeline(database, pipeline)

    # if dask_client is not None:
    #     dask_client = chain_load([dask_client]).dask_client

    if write_metadata_scores:
        pipeline.score_writer = CSVScoreWriter(os.path.join(output, "./tmp"))
    else:
        pipeline.score_writer = FourColumnsScoreWriter(os.path.join(output, "./tmp"))

    # Check if it's already checkpointed
    if checkpoint and not is_checkpointed(pipeline):
        pipeline = checkpoint_vanilla_biometrics(pipeline, output)

    background_model_samples = database.background_model_samples()

    for group in groups:

        score_file_name = os.path.join(output, f"scores-{group}")
        biometric_references = database.references(group=group)
        probes = database.probes(group=group)

        if dask_client is not None and not isinstance_nested(
            pipeline.biometric_algorithm, "biometric_algorithm", BioAlgorithmDaskWrapper
        ):

            n_objects = max(
                len(background_model_samples), len(biometric_references), len(probes)
            )
            pipeline = dask_vanilla_biometrics(
                pipeline,
                partition_size=dask_get_partition_size(dask_client.cluster, n_objects),
            )

        logger.info(f"Running vanilla biometrics for group {group}")
        allow_scoring_with_all_biometric_references = (
            database.allow_scoring_with_all_biometric_references
            if hasattr(database, "allow_scoring_with_all_biometric_references")
            else False
        )

        result = pipeline(
            background_model_samples,
            biometric_references,
            probes,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        post_processed_scores = post_process_scores(pipeline, result, score_file_name)
        _ = compute_scores(post_processed_scores, dask_client)

    if dask_client is not None:
        dask_client.shutdown()
