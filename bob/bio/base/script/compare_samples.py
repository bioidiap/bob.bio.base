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
from tabulate import tabulate
from bob.bio.base.pipelines.vanilla_biometrics import dask_vanilla_biometrics
from bob.pipelines import SampleSet, DelayedSample
import functools


EPILOG = """\n


 Command line examples\n
 -----------------------

    >>> bob bio compare-samples ./imgs/1.png ./imgs/2.png -p inception_resnetv2_msceleb \n
    \n
    \n

    All vs All comparison \n
    -------------------  ------------------- \n
    ./imgs/1.png         ./imgs/2.png         \n
    0.0                  -0.5430189337666903  \n
    -0.5430189337666903  0.0                  \n
    -------------------  -------------------  \n

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
    entry_point_group="bob.pipeline.dask_client",
    help="Dask client for the execution of the pipeline.",
)
@verbosity_option()
def compare_samples(samples, pipeline, dask_client, verbose):
    """Compare several samples in a All vs All fashion.
    """
    if len(samples) == 1:
        raise ValueError(
            "It's necessary to have at least two samples for the comparison"
        )

    sample_sets = [
        SampleSet(
            [DelayedSample(functools.partial(bob.io.base.load, s), key=str(s))],
            key=str(s),
            biometric_id=str(i),
        )
        for i, s in enumerate(samples)
    ]
    if dask_client is not None:
        pipeline = dask_vanilla_biometrics(pipeline)

    table = [[s for s in samples]]
    biometric_references = pipeline.create_biometric_reference(sample_sets)
    scores, _ = pipeline.compute_scores(sample_sets, biometric_references)
    if dask_client is not None:
        scores = scores.compute(scheduler=dask_client)
    for sset in scores:
        table.append([str(s.data) for s in sset])

    print(f"All vs All comparison")
    print(tabulate(table))

    if dask_client is not None:
        dask_client.shutdown()
