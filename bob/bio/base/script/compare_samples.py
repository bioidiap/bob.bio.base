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
import bob.io.base
import bob.io.image

import logging
import os
import itertools
import dask.bag

# from bob.bio.base.pipelines.vanilla_biometrics import (
#    VanillaBiometricsPipeline,
#    BioAlgorithmCheckpointWrapper,
#    BioAlgorithmDaskWrapper,
#    checkpoint_vanilla_biometrics,
#    dask_vanilla_biometrics,
#    dask_get_partition_size,
#    FourColumnsScoreWriter,
#    CSVScoreWriter,
# )
# from dask.delayed import Delayed
# import pkg_resources
from bob.extension.config import load as chain_load
from bob.pipelines.utils import isinstance_nested
from bob.bio.base.utils import get_resource_filename
from .vanilla_biometrics import compute_scores, load_database_pipeline
from bob.pipelines import Sample, SampleSet


logger = logging.getLogger(__name__)


EPILOG = """\b


 Command line examples\n
 -----------------------


"""


@click.command(epilog=EPILOG)
@click.argument("samples", nargs=-1)
@click.option(
    "--pipeline",
    "-p",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.bio.pipeline",
    help="Vanilla biometrics pipeline composed of a scikit-learn Pipeline and a BioAlgorithm",
)
@click.option(
    "--dask-client",
    "-l",
    required=False,
    cls=ResourceOption,
    help="Dask client for the execution of the pipeline.",
)
@verbosity_option(cls=ResourceOption)
def compare_samples(
    samples, pipeline, dask_client, **kwargs,
):
    """Compare several samples all vs all using one vanilla biometrics pipeline

    """

    if len(samples) == 1:
        raise ValueError(
            "It's necessary to have at least two samples for the comparison"
        )

    sample_sets = [
        SampleSet([Sample(bob.io.base.load(s), key=str(i))], key=str(i))
        for i, s in enumerate(samples)
    ]

    import ipdb; ipdb.set_trace()
    for e in sample_sets:
        biometric_references = pipeline.create_biometric_reference([e])
        scores = pipeline.compute_scores(biometric_references, sample_sets)
        pass

    #    B = bob.io.base.load(p)
    #    pipeline.biometric_algorithm

    if dask_client is not None:
        dask_client.shutdown()
