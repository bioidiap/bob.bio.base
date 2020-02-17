#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes biometric pipeline"""

import os
import functools

import click

from bob.extension.scripts.click_helper import verbosity_option, ResourceOption, ConfigCommand


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
    entry_point_group='bob.pipelines.config', cls=ConfigCommand,
    epilog=EPILOG,
)
@click.option(
    "--preprocessor",
    "-p",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.pipelines.preprocessors",  # This should be linked to bob.bio.base
    help="Data preprocessing algorithm",
)
@click.option(
    "--extractor",
    "-e",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.pipelines.extractor",  # This should be linked to bob.bio.base
    help="Feature extraction algorithm",
)
@click.option(
    "--algorithm",
    "-a",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.pipelines.biometric_algorithm",  # This should be linked to bob.bio.base
    help="Biometric Algorithm (class that implements the methods: `fit`, `enroll` and `score`)",
)
@click.option(
    "--database",
    "-d",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.pipelines.database",  # This should be linked to bob.bio.base
    help="Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)",
)
@click.option(
    "--dask-client",
    "-l",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.pipelines.client",  # This should be linked to bob.bio.base
    help="Dask client for the execution of the pipeline.",
)
@click.option(
    "--checkpointing", "-c", is_flag=True, help="Save checkpoints in this experiment?"
)
@click.option(
    "--group",
    "-g",
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
def vanilla_biometrics(
    preprocessor,
    extractor,
    algorithm,
    database,
    dask_client,
    checkpointing,
    group,
    output,
    **kwargs
):
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

    # Always turn-on the checkpointing
    checkpointing = True

    # Chooses the pipeline to run
    from bob.bio.base.pipelines.vanilla_biometrics import biometric_pipeline

    if not os.path.exists(output):
        os.makedirs(output)
 
    if checkpointing:
        checkpoints = {
            "background": {
                "preprocessor": os.path.join(output, "background", "preprocessed"),
                "extractor": os.path.join(output, "background", "extracted"),
                # at least, the next stage must be provided!
                "model": os.path.join(output, "background", "model"),
            },
            "references": {
                "preprocessor": os.path.join(output, "references", "preprocessed"),
                "extractor": os.path.join(output, "references", "extracted"),
                "enrolled": os.path.join(output, "references", "enrolled"),
            },
            "probes": {
                "preprocessor": os.path.join(output, "probes", "preprocessed"),
                "extractor": os.path.join(output, "probes", "extracted"),
            },
        }


    # Defines the processing pipeline for loading samples
    # Can add any number of steps!
    pipeline = [("preprocessor",preprocessor),
                ("extractor", extractor)]

    # Mechanism that loads samples
    # from ..bob_bio.blocks import SampleLoader
    from bob.bio.base.pipelines.annotated_blocks import SampleLoaderAnnotated as SampleLoader
    loader = SampleLoader(pipeline)

    for g in group:

        result = biometric_pipeline(
            database.background_model_samples(),
            database.references(group=g),
            database.probes(group=g),
            loader,
            algorithm,
            npartitions=len(dask_client.cluster.workers),
            checkpoints=checkpoints,
        )

        # result.visualize(os.path.join(output, "graph.pdf"), rankdir="LR")
        result = result.compute(scheduler=dask_client)
        for probe in result:
            for reference in probe.samples:
                print(reference.subject, probe.subject, probe.path, reference.data)

    dask_client.shutdown()
