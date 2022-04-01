import logging
import os

import dask.bag
from bob.bio.base.pipelines import BioAlgorithmDaskWrapper
from bob.bio.base.pipelines import CSVScoreWriter
from bob.bio.base.pipelines import FourColumnsScoreWriter
from bob.bio.base.pipelines import (
    checkpoint_score_normalization_pipeline,
    dask_score_normalization_pipeline,
    PipelineScoreNorm,
    ZNormScores,
    TNormScores,
)
from bob.bio.base.pipelines import checkpoint_pipeline_simple
from bob.bio.base.pipelines import dask_pipeline_simple
from bob.bio.base.pipelines import is_checkpointed
from bob.pipelines.utils import isinstance_nested, is_estimator_stateless
from dask.delayed import Delayed
from bob.pipelines.distributed import dask_get_partition_size

logger = logging.getLogger(__name__)


def compute_scores(result, dask_client):
    if isinstance(result, Delayed) or isinstance(result, dask.bag.Bag):
        if dask_client is not None:
            result = result.compute(scheduler=dask_client)
        else:
            logger.warning("`dask_client` not set. Your pipeline will run locally")
            result = result.compute(scheduler="single-threaded")
    return result


def post_process_scores(pipeline, scores, path):
    written_scores = pipeline.write_scores(scores)
    return pipeline.post_process(written_scores, path)


def execute_pipeline_simple(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    checkpoint,
    dask_partition_size,
    dask_n_workers,
    checkpoint_dir=None,
    force=False,
    **kwargs,
):
    """
    Function that executes the PipelineSimple.

    This is called when using the ``bob bio pipeline simple``
    command.

    This is also callable from a script without fear of interrupting the running
    Dask instance, allowing chaining multiple experiments while keeping the
    workers alive.

    Parameters
    ----------

    pipeline: Instance of :py:class:`bob.bio.base.pipelines.PipelineSimple`
        A constructed PipelineSimple object.

    database: Instance of :py:class:`bob.bio.base.pipelines.abstract_class.Database`
        A database interface instance

    dask_client: instance of :py:class:`dask.distributed.Client` or ``None``
        A Dask client instance used to run the experiment in parallel on multiple
        machines, or locally.
        Basic configs can be found in ``bob.pipelines.config.distributed``.

    groups: list of str
        Groups of the dataset that will be requested from the database interface.

    output: str
        Path where the scores will be saved.

    write_metadata_scores: bool
        Use the CSVScoreWriter instead of the FourColumnScoreWriter when True.

    checkpoint: bool
        Whether checkpoint files will be created for every step of the pipelines.

    checkpoint_dir: str
        If `checkpoint` is set, this path will be used to save the checkpoints.
        If `None`, the content of `output` will be used.

    force: bool
        If set, it will force generate all the checkpoints of an experiment. This option doesn't work if `--memory` is set
    """
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    # Setting the `checkpoint_dir`
    if checkpoint_dir is None:
        checkpoint_dir = output
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Scores are written on `output`
    if write_metadata_scores:
        pipeline.score_writer = CSVScoreWriter(os.path.join(output, "./tmp"))
    else:
        pipeline.score_writer = FourColumnsScoreWriter(os.path.join(output, "./tmp"))

    # Checkpoint if it's already checkpointed
    if checkpoint and not is_checkpointed(pipeline):
        hash_fn = database.hash_fn if hasattr(database, "hash_fn") else None
        pipeline = checkpoint_pipeline_simple(
            pipeline, checkpoint_dir, hash_fn=hash_fn, force=force
        )

    # Load the background model samples only if the transformer requires fitting
    if all([is_estimator_stateless(step) for step in pipeline.transformer]):
        background_model_samples = []
    else:
        background_model_samples = database.background_model_samples()

    for group in groups:

        score_file_name = os.path.join(
            output, f"scores-{group}" + (".csv" if write_metadata_scores else "")
        )
        biometric_references = database.references(group=group)
        probes = database.probes(group=group)

        # If there's no data to be processed, continue
        if len(biometric_references) == 0 or len(probes) == 0:
            logger.warning(
                f"Current dataset ({database}) does not have `{group}` set. The experiment will not be executed."
            )
            continue

        if dask_client is not None and not isinstance_nested(
            pipeline.biometric_algorithm, "biometric_algorithm", BioAlgorithmDaskWrapper
        ):
            # Scaling up
            if dask_n_workers is not None and not isinstance(dask_client, str):
                dask_client.cluster.scale(dask_n_workers)

            n_objects = max(
                len(background_model_samples), len(biometric_references), len(probes)
            )
            partition_size = None
            if not isinstance(dask_client, str):
                partition_size = dask_get_partition_size(dask_client.cluster, n_objects)
            if dask_partition_size is not None:
                partition_size = dask_partition_size

            pipeline = dask_pipeline_simple(
                pipeline,
                partition_size=partition_size,
            )

        logger.info(f"Running the PipelineSimple for group {group}")
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


def execute_pipeline_score_norm(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    checkpoint,
    dask_partition_size,
    dask_n_workers,
    checkpoint_dir=None,
    top_norm=False,
    top_norm_score_fraction=0.8,
    score_normalization_type="znorm",
    force=False,
    **kwargs,
):
    """
    Function that extends the capabilities of the PipelineSimple to run score normalization.

    This is called when using the ``bob bio pipeline score-norm`` command.

    This is also callable from a script without fear of interrupting the running
    Dask instance, allowing chaining multiple experiments while keeping the
    workers alive.

    Parameters
    ----------

    pipeline: Instance of :py:class:`bob.bio.base.pipelines.PipelineSimple`
        A constructed PipelineSimple object.

    database: Instance of :py:class:`bob.bio.base.pipelines.abstract_class.Database`
        A database interface instance

    dask_client: instance of :py:class:`dask.distributed.Client` or ``None``
        A Dask client instance used to run the experiment in parallel on multiple machines, or locally. Basic configs can be found in ``bob.pipelines.config.distributed``.

    groups: list of str
        Groups of the dataset that will be requested from the database interface.

    output: str
        Path where the results and checkpoints will be saved to.

    write_metadata_scores: bool
        Use the CSVScoreWriter instead of the FourColumnScoreWriter when True.

    checkpoint: bool
        Whether checkpoint files will be created for every step of the pipelines.

    dask_partition_size: int
        If using Dask, this option defines the size of each dask.bag.partition. Use this option if the current heuristic that sets this value doesn't suit your experiment. (https://docs.dask.org/en/latest/bag-api.html?highlight=partition_size#dask.bag.from_sequence).

    dask_n_workers: int
        If using Dask, this option defines the number of workers to start your experiment. Dask automatically scales up/down the number of workers due to the current load of tasks to be solved. Use this option if the current amount of workers set to start an experiment doesn't suit you.

    top_norm: bool

    top_norm_score_fraction: float
        Sets the percentage of samples used for t-norm and z-norm. Sometimes you don't want to use all the t/z samples for normalization

    checkpoint_dir: str
        If `checkpoint` is set, this path will be used to save the checkpoints.
        If `None`, the content of `output` will be used.

    """

    # def _merge_references_ztnorm(biometric_references, probes, zprobes, treferences):
    #    treferences_sub = [t.reference_id for t in treferences]
    #    biometric_references_sub = [t.reference_id for t in biometric_references]
    #    for i in range(len(probes)):
    #        probes[i].references += treferences_sub

    #    for i in range(len(zprobes)):
    #        zprobes[i].references = biometric_references_sub + treferences_sub

    #    return probes, zprobes

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    # Setting the `checkpoint_dir`
    if checkpoint_dir is None:
        checkpoint_dir = output
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Scores are written on `output`
    if write_metadata_scores:
        pipeline.score_writer = CSVScoreWriter(os.path.join(output, "./tmp"))
    else:
        pipeline.score_writer = FourColumnsScoreWriter(os.path.join(output, "./tmp"))

    # Check if it's already checkpointed
    if checkpoint and not is_checkpointed(pipeline):
        pipeline = checkpoint_pipeline_simple(pipeline, checkpoint_dir, force=force)

    ## PICKING THE TYPE OF POST-PROCESSING
    if score_normalization_type == "znorm":
        post_processor = ZNormScores(
            pipeline=pipeline,
            top_norm=top_norm,
            top_norm_score_fraction=top_norm_score_fraction,
        )
    elif score_normalization_type == "tnorm":
        post_processor = TNormScores(
            pipeline=pipeline,
            top_norm=top_norm,
            top_norm_score_fraction=top_norm_score_fraction,
        )
    else:
        raise ValueError(
            f"score_normalization_type {score_normalization_type} is not valid"
        )

    pipeline = ScoreNormalizationPipeline(
        pipeline, post_processor, CSVScoreWriter(os.path.join(output, "./tmp"))
    )

    if checkpoint:

        # checkpoint_score_normalization_pipeline,
        # dask_score_normalization_pipeline,

        pipeline = checkpoint_score_normalization_pipeline(
            pipeline, os.path.join(checkpoint_dir, f"{score_normalization_type}-scores")
        )

    background_model_samples = database.background_model_samples()

    # treferences = database.treferences(proportion=ztnorm_cohort_proportion)
    for group in groups:

        if score_normalization_type == "znorm":
            score_normalization_samples = database.zprobes(group=group)
        elif score_normalization_type == "tnorm":
            score_normalization_samples = database.treferences()

        score_file_name = os.path.join(output, f"scores-{group}")

        biometric_references = database.references(group=group)
        probes = database.probes(group=group)

        # If there's no data to be processed, continue
        if len(biometric_references) == 0 or len(probes) == 0:
            logger.warning(
                f"Current dataset ({database}) does not have `{group}` set. The experiment will not be executed."
            )
            continue

        if dask_client is not None and not isinstance_nested(
            pipeline.biometric_algorithm, "biometric_algorithm", BioAlgorithmDaskWrapper
        ):
            # Scaling up
            if dask_n_workers is not None and not isinstance(dask_client, str):
                dask_client.cluster.scale(dask_n_workers)

            n_objects = max(
                len(background_model_samples), len(biometric_references), len(probes)
            )
            partition_size = None
            if not isinstance(dask_client, str):
                partition_size = dask_get_partition_size(dask_client.cluster, n_objects)
            if dask_partition_size is not None:
                partition_size = dask_partition_size

            pipeline = dask_pipeline_simple(
                pipeline,
                partition_size=partition_size,
            )

        logger.info(f"Running PipelineSimple for group {group}")
        allow_scoring_with_all_biometric_references = (
            database.allow_scoring_with_all_biometric_references
            if hasattr(database, "allow_scoring_with_all_biometric_references")
            else False
        )

        # if consider_genuines:
        #    z_probes_cpy = copy.deepcopy(zprobes)
        #    zprobes += copy.deepcopy(treferences)
        #    treferences += z_probes_cpy

        # probes, zprobes = _merge_references_ztnorm(
        # biometric_references, probes, zprobes, treferences
        # )

        # probes, score_normalization_samples = _merge_references_ztnorm(
        #    biometric_references,
        #    probes,
        #    score_normalization_samples,
        #    score_normalization_samples,
        # )

        (raw_scores, score_normed_scores,) = pipeline(
            background_model_samples,
            biometric_references,
            probes,
            score_normalization_samples,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        def _build_filename(score_file_name, suffix):
            return os.path.join(score_file_name, suffix)

        # Running RAW_SCORES

        raw_scores = post_process_scores(
            pipeline, raw_scores, _build_filename(score_file_name, "raw_scores.csv")
        )
        _ = compute_scores(raw_scores, dask_client)

        # Z-SCORES
        score_normed_scores = post_process_scores(
            pipeline,
            score_normed_scores,
            _build_filename(score_file_name, f"{score_normalization_type}.csv"),
        )
        _ = compute_scores(score_normed_scores, dask_client)

        # T-SCORES
        """
        t_normed_scores = post_process_scores(
            pipeline,
            t_normed_scores,
            _build_filename(score_file_name, "t_normed_scores.csv"),
        )
        _ = compute_scores(t_normed_scores, dask_client)

        # S-SCORES
        s_normed_scores = post_process_scores(
            pipeline,
            s_normed_scores,
            _build_filename(score_file_name, "s_normed_scores.csv"),
        )
        _ = compute_scores(s_normed_scores, dask_client)

        # ZT-SCORES
        zt_normed_scores = post_process_scores(
            pipeline,
            zt_normed_scores,
            _build_filename(score_file_name, "zt_normed_scores.csv"),
        )
        _ = compute_scores(zt_normed_scores, dask_client)
        """
