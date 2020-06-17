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
)
from dask.delayed import Delayed
import pkg_resources
from bob.extension.config import load as chain_load
from bob.pipelines.utils import isinstance_nested


logger = logging.getLogger(__name__)


def get_resource_filename(resource_name, group):
    """
    Get the file name of a resource.


    Parameters
    ----------
        resource_name: str
            Name of the resource to be searched
        
        group: str
            Entry point group

    Return
    ------
        filename: str
            The entrypoint file name

    """

    # Check if it's already a path
    if os.path.exists(resource_name):
        return resource_name

    # If it's a resource get the path of this resource
    resources = [r for r in pkg_resources.iter_entry_points(group)]

    # if resource_name not in [r.name for r in resources]:
    #    raise ValueError(f"Resource not found: `{resource_name}`")

    for r in resources:
        if r.name == resource_name:
            resource = r
            break
    else:
        raise ValueError(f"Resource not found: `{resource_name}`")

    # TODO: This get the root path only
    #        I don't know how to get the filename
    return (
        pkg_resources.resource_filename(
            resource.module_name, resource.module_name.split(".")[-1]
        )
        + ".py"
    )


EPILOG = """\b


 Command line examples\n
 -----------------------


 $ bob pipelines vanilla-biometrics my_experiment.py -vv


 my_experiment.py must contain the following elements:

 >>> preprocessor = my_preprocessor() \n
 >>> extractor = my_extractor() \n
 >>> algorithm = my_algorithm() \n
 >>> checkpoints = EXPLAIN CHECKPOINTING \n

\b


Look at the following example

 $ bob pipelines vanilla-biometrics ./bob/pipelines/config/distributed/sge_iobig_16cores.py \
                                    ./bob/pipelines/config/database/mobio_male.py \
                                    ./bob/pipelines/config/baselines/facecrop_pca.py

\b



TODO: Work out this help

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


@click.command(
    entry_point_group="bob.bio.pipeline.config", cls=ConfigCommand, epilog=EPILOG,
)
@click.option(
    "--pipeline", "-p", required=True, help="Vanilla biometrics pipeline",
)
@click.option(
    "--database",
    "-d",
    required=True,
    help="Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)",
)
@click.option(
    "--dask-client",
    "-l",
    required=False,
    help="Dask client for the execution of the pipeline.",
)
@click.option(
    "--group",
    "-g",
    "groups",
    type=click.Choice(["dev", "eval"]),
    multiple=True,
    default=("dev",),
    help="If given, this value will limit the experiments belonging to a particular protocolar group",
)
@click.option(
    "-o",
    "--output",
    show_default=True,
    default="results",
    help="Name of output directory",
)
@verbosity_option(cls=ResourceOption)
def vanilla_biometrics(pipeline, database, dask_client, groups, output, **kwargs):
    """Runs the simplest biometrics pipeline.

    Such pipeline consists into three sub-pipelines.
    In all of them, given raw data as input it does the following steps:

    Sub-pipeline 1:\n
    ---------------

    Training background model. Some biometric algorithms demands the training of background model, for instance, PCA/LDA matrix or a Neural networks. This sub-pipeline handles that and it consists of 3 steps:

    \b
    raw_data --> preprocessing >> feature extraction >> train background model --> background_model



    \b

    Sub-pipeline 2:\n
    ---------------

    Creation of biometric references: This is a standard step in a biometric pipelines.
    Given a set of samples of one identity, create a biometric reference (a.k.a template) for sub identity. This sub-pipeline handles that in 3 steps and they are the following:

    \b
    raw_data --> preprocessing >> feature extraction >> enroll(background_model) --> biometric_reference

    Note that this sub-pipeline depends on the previous one



    Sub-pipeline 3:\n
    ---------------


    Probing: This is another standard step in biometric pipelines. Given one sample and one biometric reference, computes a score. Such score has different meanings depending on the scoring method your biometric algorithm uses. It's out of scope to explain in a help message to explain what scoring is for different biometric algorithms.


    raw_data --> preprocessing >> feature extraction >> probe(biometric_reference, background_model) --> score

    Note that this sub-pipeline depends on the two previous ones


    """

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    # It's necessary to chain load 2 resources together
    pipeline_config = get_resource_filename(pipeline, "bob.bio.pipeline")
    database_config = get_resource_filename(database, "bob.bio.database")
    vanilla_pipeline = chain_load([database_config, pipeline_config])
    dask_client = chain_load([dask_client]).dask_client

    # Picking the resources
    database = vanilla_pipeline.database
    pipeline = vanilla_pipeline.pipeline

    # Check if it's already checkpointed
    if not isinstance_nested(
        pipeline.biometric_algorithm,
        "biometric_algorithm",
        BioAlgorithmCheckpointWrapper,
    ):
        pipeline = checkpoint_vanilla_biometrics(pipeline, output)

    background_model_samples = database.background_model_samples()
    for group in groups:

        score_file_name = os.path.join(output, f"scores-{group}")
        biometric_references = database.references(group=group)
        probes = database.probes(group=group)

        if dask_client is not None and not isinstance_nested(
            pipeline.biometric_algorithm, "biometric_algorithm", BioAlgorithmDaskWrapper
        ):

            n_objects = (
                len(background_model_samples) + len(biometric_references) + len(probes)
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
