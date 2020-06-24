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
    ZTNormPipeline,
    ZTNormDaskWrapper,
    ZTNormCheckpointWrapper,
    checkpoint_vanilla_biometrics,
    dask_vanilla_biometrics,
    dask_get_partition_size,
    FourColumnsScoreWriter,
    CSVScoreWriter,
    BioAlgorithmLegacy,
)
from dask.delayed import Delayed
from bob.bio.base.utils import get_resource_filename
from bob.extension.config import load as chain_load
from bob.pipelines.utils import isinstance_nested
from .vanilla_biometrics import (
    compute_scores,
    post_process_scores,
    load_database_pipeline,
)
import copy

logger = logging.getLogger(__name__)


EPILOG = """\b


 Command line examples\n
 -----------------------


 $ bob pipelines vanilla-biometrics -p my_experiment.py -vv


 my_experiment.py must contain the following elements:

   >>> transformer = ... # A scikit-learn pipeline
   >>> algorithm   = ... # `An BioAlgorithm`
   >>> pipeline = VanillaBiometricsPipeline(transformer,algorithm)
   >>> database = .... # Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)" 

\b


"""


@click.command(
    entry_point_group="bob.pipelines.config", cls=ConfigCommand, epilog=EPILOG,
)
@click.option(
    "--pipeline",
    "-p",
    required=True,
    help="Vanilla biometrics pipeline composed of a scikit-learn Pipeline and a BioAlgorithm",
)
@click.option(
    "--database",
    "-d",
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
@click.option(
    "--consider-genuines",
    is_flag=True,
    help="If set, will consider genuine scores in the ZT score normalization",
)
@click.option(
    "--write-metadata-scores",
    "-m",
    is_flag=True,
    help="If set, all the scores will be written with all its metadata",
)
@click.option(
    "--ztnorm-cohort-proportion",
    default=1.0,
    type=float,
    help="Sets the percentage of samples used for t-norm and z-norm. Sometimes you don't want to use all the t/z samples for normalization",
)
@click.option(
    "--checkpoint",
    "-c",
    is_flag=True,
    help="If set, it will checkpoint all steps of the pipeline",
)
@verbosity_option(cls=ResourceOption)
def vanilla_biometrics_ztnorm(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    consider_genuines,
    write_metadata_scores,
    ztnorm_cohort_proportion,
    checkpoint,
    **kwargs,
):
    """Runs the the vanilla-biometrics with ZT-Norm like score normalizations.

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

    def _merge_references_ztnorm(biometric_references, probes, zprobes, treferences):
        treferences_sub = [t.subject for t in treferences]
        biometric_references_sub = [t.subject for t in biometric_references]

        for i in range(len(zprobes)):
            probes[i].references += treferences_sub

        for i in range(len(zprobes)):
            zprobes[i].references = biometric_references_sub + treferences_sub

        return probes, zprobes

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    # It's necessary to chain load 2 resources together
    # Picking the resources
    database, pipeline = load_database_pipeline(database, pipeline)

    if dask_client is not None:
        dask_client = chain_load([dask_client]).dask_client

    if write_metadata_scores:
        pipeline.score_writer = CSVScoreWriter(os.path.join(output, "./tmp"))
    else:
        pipeline.score_writer = FourColumnsScoreWriter(os.path.join(output, "./tmp"))

    # Check if it's already checkpointed
    if checkpoint and (
        not isinstance_nested(
            pipeline, "biometric_algorithm", BioAlgorithmCheckpointWrapper,
        )
        and not isinstance_nested(pipeline, "biometric_algorithm", BioAlgorithmLegacy)
    ):
        pipeline = checkpoint_vanilla_biometrics(pipeline, output)

    # Patching the pipeline in case of ZNorm and checkpointing it
    pipeline = ZTNormPipeline(pipeline)
    pipeline.ztnorm_solver = ZTNormCheckpointWrapper(
        pipeline.ztnorm_solver, os.path.join(output, "normed-scores")
    )

    background_model_samples = database.background_model_samples()
    zprobes = database.zprobes(proportion=ztnorm_cohort_proportion)
    treferences = database.treferences(proportion=ztnorm_cohort_proportion)
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

        if consider_genuines:
            z_probes_cpy = copy.deepcopy(zprobes)
            zprobes += copy.deepcopy(treferences)
            treferences += z_probes_cpy

        probes, zprobes = _merge_references_ztnorm(
            biometric_references, probes, zprobes, treferences
        )

        (
            raw_scores,
            z_normed_scores,
            t_normed_scores,
            zt_normed_scores,
            s_normed_scores,
        ) = pipeline(
            background_model_samples,
            biometric_references,
            probes,
            zprobes,
            treferences,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        def _build_filename(score_file_name, suffix):
            return os.path.join(score_file_name, suffix)

        # Running RAW_SCORES
        raw_scores = post_process_scores(
            pipeline, raw_scores, _build_filename(score_file_name, "raw_scores")
        )

        _ = compute_scores(raw_scores, dask_client)

        # Z-SCORES
        z_normed_scores = post_process_scores(
            pipeline,
            z_normed_scores,
            _build_filename(score_file_name, "z_normed_scores"),
        )
        _ = compute_scores(z_normed_scores, dask_client)

        # T-SCORES
        t_normed_scores = post_process_scores(
            pipeline,
            t_normed_scores,
            _build_filename(score_file_name, "t_normed_scores"),
        )
        _ = compute_scores(t_normed_scores, dask_client)

        # S-SCORES
        s_normed_scores = post_process_scores(
            pipeline,
            s_normed_scores,
            _build_filename(score_file_name, "s_normed_scores"),
        )
        _ = compute_scores(s_normed_scores, dask_client)

        # ZT-SCORES
        zt_normed_scores = post_process_scores(
            pipeline,
            zt_normed_scores,
            _build_filename(score_file_name, "zt_normed_scores"),
        )
        _ = compute_scores(zt_normed_scores, dask_client)

    if dask_client is not None:
        dask_client.shutdown()