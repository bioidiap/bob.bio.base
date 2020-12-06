#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes biometric pipeline"""

import logging

import click
from bob.bio.base.pipelines.vanilla_biometrics import execute_vanilla_biometrics
from bob.extension.scripts.click_helper import ConfigCommand
from bob.extension.scripts.click_helper import ResourceOption
from bob.extension.scripts.click_helper import verbosity_option

from bob.pipelines.distributed import VALID_DASK_CLIENT_STRINGS

logger = logging.getLogger(__name__)


EPILOG = """\b


 Command line examples\n
 -----------------------

$ bob pipelines vanilla-biometrics DATABASE PIPELINE -vv

 Check out all PIPELINE available by running:
  `resource.py --types pipeline`
\b

  and all available databases by running:
  `resource.py --types database`

\b

It is possible to do it via configuration file

 $ bob pipelines vanilla-biometrics -p my_experiment.py -vv


 my_experiment.py must contain the following elements:

   >>> transformer = ... # A scikit-learn pipeline\n
   >>> algorithm   = ... # `An BioAlgorithm`\n
   >>> pipeline = VanillaBiometricsPipeline(transformer,algorithm)\n
   >>> database = .... # Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)"

\b


"""


@click.command(
    entry_point_group="bob.bio.config", cls=ConfigCommand, epilog=EPILOG,
)
@click.option(
    "--pipeline",
    "-p",
    required=True,
    entry_point_group="bob.bio.pipeline",
    help="Vanilla biometrics pipeline composed of a scikit-learn Pipeline and a BioAlgorithm",
    cls=ResourceOption,
)
@click.option(
    "--database",
    "-d",
    entry_point_group="bob.bio.database",
    required=True,
    help="Biometric Database connector (class that implements the methods: `background_model_samples`, `references` and `probes`)",
    cls=ResourceOption,
)
@click.option(
    "--dask-client",
    "-l",
    entry_point_group="dask.client",
    string_exceptions=VALID_DASK_CLIENT_STRINGS,
    default="single-threaded",
    help="Dask client for the execution of the pipeline.",
    cls=ResourceOption,
)
@click.option(
    "--group",
    "-g",
    "groups",
    type=click.Choice(["dev", "eval"]),
    multiple=True,
    default=("dev",),
    help="If given, this value will limit the experiments belonging to a particular protocolar group",
    cls=ResourceOption,
)
@click.option(
    "-o",
    "--output",
    show_default=True,
    default="results",
    help="Name of output directory where output scores will be saved. In case --checkpoint is set, checkpoints will be saved in this directory.",
    cls=ResourceOption,
)
@click.option(
    "--write-metadata-scores",
    "-m",
    is_flag=True,
    help="If set, all the scores will be written with all its metadata using the `CSVScoreWriter`",
    cls=ResourceOption,
)
@click.option(
    "--checkpoint",
    "-c",
    is_flag=True,
    help="If set, it will checkpoint all steps of the pipeline. Checkpoints will be saved in `--output`.",
    cls=ResourceOption,
)
@click.option(
    "--dask-partition-size",
    "-s",
    help="If using Dask, this option defines the size of each dask.bag.partition."
    "Use this option if the current heuristic that sets this value doesn't suit your experiment."
    "(https://docs.dask.org/en/latest/bag-api.html?highlight=partition_size#dask.bag.from_sequence).",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@click.option(
    "--dask-n-workers",
    "-n",
    help="If using Dask, this option defines the number of workers to start your experiment."
    "Dask automatically scales up/down the number of workers due to the current load of tasks to be solved."
    "Use this option if the current amount of workers set to start an experiment doesn't suit you.",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def vanilla_biometrics(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    checkpoint,
    dask_partition_size,
    dask_n_workers,
    **kwargs,
):
    """Runs the simplest biometrics pipeline.

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

    .. Note::
        Refrain from calling this function directly from a script. Prefer
        :py:func:`~bob.bio.base.pipelines.vanilla_biometrics.execute_vanilla_biometrics`
        instead.

    """
    logger.debug("Executing Vanilla-biometrics")
    execute_vanilla_biometrics(
        pipeline,
        database,
        dask_client,
        groups,
        output,
        write_metadata_scores,
        checkpoint,
        dask_partition_size,
        dask_n_workers,
        **kwargs,
    )

    logger.info("Experiment finished !")
