import logging
import os

import dask.bag

from dask.delayed import Delayed

from bob.bio.base.pipelines import (
    BioAlgDaskWrapper,
    CSVScoreWriter,
    FourColumnsScoreWriter,
    PipelineScoreNorm,
    TNormScores,
    ZNormScores,
    checkpoint_pipeline_simple,
    dask_bio_pipeline,
    is_biopipeline_checkpointed,
)
from bob.pipelines import estimator_requires_fit, is_instance_nested, wrap
from bob.pipelines.distributed import dask_get_partition_size
from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster

logger = logging.getLogger(__name__)


def compute_scores(result, dask_client):
    if isinstance(result, Delayed) or isinstance(result, dask.bag.Bag):
        if dask_client is not None:
            result = result.compute(scheduler=dask_client)
        else:
            logger.warning(
                "`dask_client` not set. Your pipeline will run locally"
            )
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
    dask_n_partitions,
    dask_partition_size,
    dask_n_workers,
    checkpoint_dir=None,
    force=False,
):
    """
    Function that executes the PipelineSimple.

    This is called when using the ``bob bio pipeline simple``
    command.

    This is also callable from a script without fear of interrupting the running
    Dask instance, allowing chaining multiple experiments while keeping the
    workers alive.

    When using Dask, something to keep in mind is that we want to split our data and
    processing time on multiple workers. There is no recipe to make everything work on
    any system. So if you encounter some balancing error (a few of all the available
    workers actually working while the rest waits, or the scheduler being overloaded
    trying to organise millions of tiny tasks), you can specify ``dask_n_partitions``
    or ``dask_partition_size``.
    The first will try to split any set of data into a number of chunks (ideally, we
    would want one per worker), and the second creates similar-sized partitions in each
    set.
    If the memory on the workers is not sufficient, try reducing the size of the
    partitions (or increasing the number of partitions).

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

    dask_n_partitions: int or None
        Specifies a number of partitions to split the data into.

    dask_partition_size: int or None
        Specifies a data partition size when using dask. Ignored when dask_n_partitions
        is set.

    dask_n_workers: int or None
        Sets the starting number of Dask workers. Does not prevent Dask from requesting
        more or releasing workers depending on load.

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
        pipeline.score_writer = FourColumnsScoreWriter(
            os.path.join(output, "./tmp")
        )

    # Checkpoint if it's already checkpointed
    if checkpoint and not is_biopipeline_checkpointed(pipeline):
        hash_fn = database.hash_fn if hasattr(database, "hash_fn") else None
        pipeline = checkpoint_pipeline_simple(
            pipeline, checkpoint_dir, hash_fn=hash_fn, force=force
        )

    # Load the background model samples only if the transformer requires fitting
    if estimator_requires_fit(pipeline.transformer):
        background_model_samples = database.background_model_samples()
    else:
        background_model_samples = []

    for group in groups:

        score_file_name = os.path.join(
            output,
            f"scores-{group}" + (".csv" if write_metadata_scores else ""),
        )
        biometric_references = database.references(group=group)
        probes = database.probes(group=group)

        # If there's no data to be processed, continue
        if len(biometric_references) == 0 or len(probes) == 0:
            logger.warning(
                f"Current dataset ({database}) does not have `{group}` set. The experiment will not be executed."
            )
            continue

        if dask_client is not None and not is_instance_nested(
            pipeline.biometric_algorithm,
            "biometric_algorithm",
            BioAlgDaskWrapper,
        ):
            # Scaling up
            if dask_n_workers is not None and not isinstance(dask_client, str):
                dask_client.cluster.scale(dask_n_workers)

            # Data partitioning.
            # - Too many small partitions: the scheduler takes more time scheduling
            #   than the computations.
            # - Too few big partitions: We don't use all the available workers and thus
            #   run slower.
            if dask_partition_size is not None:
                logger.debug(
                    f"Splitting data with fixed size partitions: {dask_partition_size}."
                )
                pipeline = dask_bio_pipeline(
                    pipeline,
                    partition_size=dask_partition_size,
                )
            elif dask_n_partitions is not None or dask_n_workers is not None:
                # Divide each Set in a user-defined number of partitions
                logger.debug("Splitting data with fixed number of partitions.")
                pipeline = dask_bio_pipeline(
                    pipeline,
                    npartitions=dask_n_partitions or dask_n_workers,
                )
            else:
                # Split in max_jobs partitions or revert to the default behavior of
                # dask.Bag from_sequence: partition_size = 100
                n_jobs = None
                if not isinstance(dask_client, str) and isinstance(
                    dask_client.cluster, SGEMultipleQueuesCluster
                ):
                    logger.debug(
                        "Splitting data according to the number of available workers."
                    )
                    n_jobs = dask_client.cluster.sge_job_spec["default"][
                        "max_jobs"
                    ]
                    logger.debug(f"{n_jobs} partitions will be created.")
                pipeline = dask_bio_pipeline(pipeline, npartitions=n_jobs)

        logger.info(f"Running the PipelineSimple for group {group}")
        score_all_vs_all = (
            database.score_all_vs_all
            if hasattr(database, "score_all_vs_all")
            else False
        )

        result = pipeline(
            background_model_samples,
            biometric_references,
            probes,
            score_all_vs_all=score_all_vs_all,
        )

        post_processed_scores = post_process_scores(
            pipeline, result, score_file_name
        )
        compute_scores(post_processed_scores, dask_client)


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
        pipeline.score_writer = FourColumnsScoreWriter(
            os.path.join(output, "./tmp")
        )

    # Check if it's already checkpointed
    if checkpoint and not is_biopipeline_checkpointed(pipeline):
        pipeline = checkpoint_pipeline_simple(
            pipeline, checkpoint_dir, force=force
        )

    # PICKING THE TYPE OF POST-PROCESSING
    if score_normalization_type == "znorm":
        post_processor = ZNormScores(
            top_norm=top_norm,
            top_norm_score_fraction=top_norm_score_fraction,
        )
    elif score_normalization_type == "tnorm":
        post_processor = TNormScores(
            top_norm=top_norm,
            top_norm_score_fraction=top_norm_score_fraction,
        )
    else:
        raise ValueError(
            f"score_normalization_type {score_normalization_type} is not valid"
        )

    if checkpoint and not is_biopipeline_checkpointed(post_processor):
        model_path = os.path.join(
            checkpoint_dir,
            f"{score_normalization_type}-scores",
            "norm",
            "stats.pkl",
        )
        # we cannot checkpoint "features" because sample.keys are not unique.
        post_processor = wrap(
            ["checkpoint"], post_processor, model_path=model_path, force=force
        )

    pipeline = PipelineScoreNorm(pipeline, post_processor)

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

        if dask_client is not None and not is_instance_nested(
            pipeline.biometric_algorithm,
            "biometric_algorithm",
            BioAlgDaskWrapper,
        ):
            # Scaling up
            if dask_n_workers is not None and not isinstance(dask_client, str):
                dask_client.cluster.scale(dask_n_workers)

            n_objects = max(
                len(background_model_samples),
                len(biometric_references),
                len(probes),
            )
            partition_size = None
            if not isinstance(dask_client, str):
                partition_size = dask_get_partition_size(
                    dask_client.cluster, n_objects
                )
            if dask_partition_size is not None:
                partition_size = dask_partition_size

            pipeline = dask_bio_pipeline(
                pipeline,
                partition_size=partition_size,
            )

        logger.info(f"Running PipelineSimple for group {group}")
        score_all_vs_all = (
            database.score_all_vs_all
            if hasattr(database, "score_all_vs_all")
            else False
        )

        (raw_scores, score_normed_scores,) = pipeline(
            background_model_samples,
            biometric_references,
            probes,
            score_normalization_samples,
            score_all_vs_all=score_all_vs_all,
        )

        def _build_filename(score_file_name, suffix):
            return os.path.join(score_file_name, suffix)

        # Running RAW_SCORES

        raw_scores = post_process_scores(
            pipeline,
            raw_scores,
            _build_filename(score_file_name, "raw_scores.csv"),
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
