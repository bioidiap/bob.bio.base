#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes biometric pipeline"""

import os
import functools

import click

from bob.extension.scripts.click_helper import verbosity_option, ResourceOption, ConfigCommand
from bob.pipelines.sample import DelayedSample, Sample

import logging
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
    entry_point_group='bob.pipelines.config', cls=ConfigCommand,
    epilog=EPILOG,
)
@click.option(
    "--extractor",
    "-e",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.bio.extractor",  # This should be linked to bob.bio.base
    help="Feature extraction algorithm",
)
@click.option(
    "--algorithm",
    "-a",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.bio.algorithm",  # This should be linked to bob.bio.base
    help="Biometric Algorithm (class that implements the methods: `fit`, `enroll` and `score`)",
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
    entry_point_group="bob.pipelines.client",  # This should be linked to bob.bio.base
    help="Dask client for the execution of the pipeline.",
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
    extractor,
    algorithm,
    database,
    dask_client,
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
    from bob.bio.base.pipelines.vanilla_biometrics.pipeline import biometric_pipeline

    if not os.path.exists(output):
        os.makedirs(output)
 
    for g in group:

        with open(os.path.join(output,f"scores-{g}"), "w") as f:
            biometric_references = database.references(group=g)

            result = biometric_pipeline(
                database.background_model_samples(),
                biometric_references,
                database.probes(group=g),
                extractor,
                algorithm,
                
            )
        
            if dask_client is not None:
                result = result.compute(scheduler=dask_client)
                result = result.compute(scheduler="single-threaded")

            #import ipdb; ipdb.set_trace()
            for probe in result:
                for sample in probe.samples:
                    
                    if isinstance(sample, Sample):
                        line = "{0} {1} {2} {3}\n".format(sample.key, probe.key, probe.path, sample.data)
                        f.write(line)
                    elif isinstance(sample, DelayedSample):
                        lines = sample.load().readlines()
                        f.writelines(lines)
                    else:
                        raise TypeError("The output of the pipeline is not writeble")

    #dask_client.shutdown()



@click.command()
@click.argument("output-file")
@verbosity_option(cls=ResourceOption)
def vanilla_biometrics_template(output_file, **kwargs):
    """
    Generate an template configuration file for the vanilla biometrics pipeline
    """

    import bob.io.base

    path = os.path.dirname(output_file)
    logger.info(f"Writting template configuration file in {path}")
    bob.io.base.create_directories_safe(path)

    template = '''

# Client dask. Look at https://gitlab.idiap.ch/bob/bob.pipelines/tree/master/bob/pipelines/config/distributed to find proper dask clients.
# You don't need to necessary instantiate a dask client yourself. You can simply pipe those config files

dask_client = my_client


preprocessor = my_preprocessor


extractor = my_extractor


algorithm = my_algorithm


database = my_database
    
'''

    open(output_file, "w").write(template)
