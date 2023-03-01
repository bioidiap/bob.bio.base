#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes biometric pipeline"""

import functools
import logging

from typing import List

import click
import dask.distributed

from clapper.click import ResourceOption, verbosity_option
from tabulate import tabulate

import bob.io.base

from bob.bio.base.pipelines import PipelineSimple, dask_bio_pipeline
from bob.pipelines import DelayedSample, Sample, SampleSet

logger = logging.getLogger(__name__)

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
    entry_point_group="dask.client",
    help="Dask client for the execution of the pipeline.",
)
@verbosity_option(logger=logger)
def compare_samples(
    samples: List[Sample],
    pipeline: PipelineSimple,
    dask_client: dask.distributed.Client,
    verbose: int,
):
    """Compare several samples in a All vs All fashion."""
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
        pipeline = dask_bio_pipeline(pipeline)

    table = [[s for s in samples]]
    enroll_templates = pipeline.enroll_templates(sample_sets)
    probe_templates = pipeline.probe_templates(sample_sets)
    scores = pipeline.compute_scores(
        probe_templates, enroll_templates, score_all_vs_all=True
    )
    if dask_client is not None:
        scores = scores.compute(scheduler=dask_client)
    for sset in scores:
        table.append([str(s.data) for s in sset])

    print("All vs All comparison")
    print(tabulate(table))

    if dask_client is not None:
        dask_client.shutdown()
