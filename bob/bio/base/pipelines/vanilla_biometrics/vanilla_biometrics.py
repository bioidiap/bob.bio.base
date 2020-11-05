import logging
import os
import dask.bag
from dask.delayed import Delayed
from bob.bio.base.pipelines.vanilla_biometrics import (
    BioAlgorithmDaskWrapper,
    checkpoint_vanilla_biometrics,
    dask_vanilla_biometrics,
    dask_get_partition_size,
    FourColumnsScoreWriter,
    CSVScoreWriter,
    is_checkpointed,
)
from bob.pipelines.utils import isinstance_nested


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



def execute_vanilla_biometrics(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    checkpoint,
    **kwargs,
):
    """
    Function that executes the Vanilla Biometrics pipeline.

    This is called when using the ``bob bio pipelines vanilla-biometrics``
    command.

    This is also callable from a script without fear of interrupting the running
    Dask instance, allowing chaining multiple experiments while keeping the
    workers alive.

    Parameters
    ----------

    pipeline: Instance of :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline`
        A constructed vanilla-biometrics pipeline.

    database: Instance of :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.abstract_class.Database`
        A database interface instance

    dask_client: instance of :py:class:`dask.distributed.Client` or ``None``
        A Dask client instance used to run the experiment in parallel on multiple
        machines, or locally.
        Basic configs can be found in ``bob.pipelines.config.distributed``.

    groups: list of str
        Groups of the dataset that will be requested from the database interface.

    output: str
        Path where the results and checkpoints will be saved to.

    write_metadata_scores: bool
        Use the CSVScoreWriter instead of the FourColumnScoreWriter when True.

    checkpoint: bool
        Whether checkpoint files will be created for every step of the pipelines.
    """
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

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
