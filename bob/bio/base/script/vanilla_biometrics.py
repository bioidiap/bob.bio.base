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
)


logger = logging.getLogger(__name__)


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


@click.command(
    entry_point_group="bob.pipelines.config", cls=ConfigCommand, epilog=EPILOG,
)
@click.option(
    "--pipeline",
    "-p",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.pipelines.pipeline",
    help="Feature extraction algorithm",
)
@click.option(
    "--database",
    "-d",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.bio.database",  # This should be linked to bob.bio.base
    help="Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)",
)
@click.option(
    "--dask-client",
    "-l",
    required=False,
    cls=ResourceOption,
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

    for group in groups:

        score_file_name = os.path.join(output, f"scores-{group}.txt")
        biometric_references = database.references(group=group)

        logger.info(f"Running vanilla biometrics for group {group}")

        allow_scoring_with_all_biometric_references = (
            database.allow_scoring_with_all_biometric_references
            if hasattr(database, "allow_scoring_with_all_biometric_references")
            else False
        )

        result = pipeline(
            database.background_model_samples(),
            biometric_references,
            database.probes(group=group),
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        if isinstance(result, dask.bag.core.Bag):
            if dask_client is not None:
                result = result.compute(scheduler=dask_client)
            else:
                logger.warning(
                    "`dask_client` not set. Your pipeline will run locally"
                )
                result = result.compute(scheduler="single-threaded")

        # Check if there's a score writer hooked in
        if isinstance(pipeline.biometric_algorithm, BioAlgorithmCheckpointWrapper):
            pipeline.biometric_algorithm.score_writer.concatenate_write_scores(result, score_file_name)
        else:
            # Flatting out the list
            result = itertools.chain(*result)
            for probe in result:
                for sample in probe.samples:
                    f.write(sample.data)

    if dask_client is not None:
        dask_client.shutdown()
