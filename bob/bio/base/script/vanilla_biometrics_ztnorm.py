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
    CSVScoreWriter
)
from dask.delayed import Delayed
from bob.bio.base.utils import get_resource_filename
from bob.extension.config import load as chain_load
from bob.pipelines.utils import isinstance_nested
from .vanilla_biometrics import compute_scores, post_process_scores
import copy

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
@click.option(
    "--consider-genuines",
    is_flag=True,
    help="If set, will consider genuine scores in the ZT score normalization",
)
@click.option(
    "--write-metadata-scores", "-m",
    is_flag=True,
    help="If set, all the scores will be written with all its metadata",
)
@click.option("--ztnorm-cohort-proportion", default=1., type=float, 
    help="Sets the percentage of samples used for t-norm and z-norm. Sometimes you don't want to use all the t/z samples for normalization")
@verbosity_option(cls=ResourceOption)
def vanilla_biometrics_ztnorm(
    pipeline, database, dask_client, groups, output, consider_genuines, write_metadata_scores, ztnorm_cohort_proportion, **kwargs
):
    """Runs the simplest biometrics pipeline under ZTNorm.

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
    pipeline_config = get_resource_filename(pipeline, "bob.bio.pipeline")
    database_config = get_resource_filename(database, "bob.bio.database")
    vanilla_pipeline = chain_load([database_config, pipeline_config])
    if dask_client is not None:
        dask_client = chain_load([dask_client]).dask_client

    # Picking the resources
    database = vanilla_pipeline.database
    pipeline = vanilla_pipeline.pipeline

    if write_metadata_scores:
        pipeline.score_writer = CSVScoreWriter(os.path.join(output,"./tmp"))
    else:
        pipeline.score_writer = FourColumnsScoreWriter(os.path.join(output,"./tmp"))


    # Check if it's already checkpointed
    if not isinstance_nested(
        pipeline.biometric_algorithm,
        "biometric_algorithm",
        BioAlgorithmCheckpointWrapper,
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
        
        raw_scores, z_normed_scores, t_normed_scores, zt_normed_scores, s_normed_scores = pipeline(
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
