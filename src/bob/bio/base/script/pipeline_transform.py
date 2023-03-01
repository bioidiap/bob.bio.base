#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


import logging

import click

from clapper.click import ConfigCommand, ResourceOption, verbosity_option

from bob.pipelines import is_pipeline_wrapped
from bob.pipelines.distributed import (
    VALID_DASK_CLIENT_STRINGS,
    dask_get_partition_size,
)

logger = logging.getLogger(__name__)
from bob.pipelines.wrappers import CheckpointWrapper, DaskWrapper, wrap

EPILOG = """\b
Command line examples\n
---------------------\n


Follow below an example on how to extract arcface features from a database:\n

`bob bio transform my_database iresnet100 -vv`\n\n

To "dask" the execution of the pipeline, you can use the `--dask-client` option.\n
In the example below we show how to use the `--dask-client` option to start a dask cluster on SGE.\n

`bob bio transform my_database iresnet100 --dask-client sge -vv`\n\n

\b

Creating my own transformer\n
---------------------------\n

This command accepts configuration file as input.\n
For example, if you desire to customize your transformer, you can use the following configuration file:\n
\b\b


```py\n
from sklearn.base import BaseEstimator, TransformerMixin \n
from sklearn.pipeline import make_pipeline  \n
from bob.pipelines import wrap \n

class MyTransformer(TransformerMixin, BaseEstimator): \n
    def _more_tags(self): \n
        return {"requires_fit": False} \n

    def transform(self, X): \n
        # do something \n
        return X \n

transformer = wrap(["sample"],make_pipeline(MyTransformer())) \n
```

Then, you can use above configuration file to run the command:

\b
`bob bio pipelines transform my_database my_transformer.py --dask-client sge -vv`

\b\b

Leveraging from FunctionTransformer\n
-----------------------------------\n

You can also benefit from `FunctionTransformer` to create a transformer

```py \n
from sklearn.preprocessing import FunctionTransformer \n
from sklearn.pipeline import make_pipeline \n
from bob.pipelines import wrap \n

\b

def my_transformer(X): \n
    # do something \n
    return X \n


transformer = wrap(["sample"],make_pipeline(FunctionTransformer(my_transformer))) \n
```

Then, you can use above configuration file to run the command:\n

`bob bio pipelines transform my_database my_transformer.py --dask-client sge -vv`


\b\b
"""


@click.command(
    name="transform",
    entry_point_group="bob.bio.config",
    cls=ConfigCommand,
    epilog=EPILOG,
)
@click.option(
    "--transformer",
    "-t",
    required=True,
    entry_point_group="bob.bio.transformer",
    help="A scikit-learn Pipeline containing the set of transformations",
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
    "-c",
    "--checkpoint-dir",
    show_default=True,
    default="./checkpoints",
    help="Name of output directory where the checkpoints will be saved.",
    cls=ResourceOption,
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="If set, it will force generate all the checkpoints of an experiment. This option doesn't work if `--memory` is set",
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
@verbosity_option(cls=ResourceOption, logger=logger)
def pipeline_transform(
    transformer,
    database,
    dask_client,
    checkpoint_dir,
    force,
    dask_partition_size,
    **kwargs,
):
    """
    This CLI command will execute a pipeline (a scikit learn Pipeline)  on a given database.

    This command can be used, for example, to extract features, face-crops, preprocess audio files and so on.

    """

    logger.info(f"Transforming samples from {database}")

    # save_func = bob.io.base.save

    # Idiap SETUP. This avoids having directories with more than 1000 files/directories
    hash_fn = database.hash_fn if hasattr(database, "hash_fn") else None

    # If NONE of the items are checkpointed, we checkpoint them all
    if not any(is_pipeline_wrapped(transformer, CheckpointWrapper)):
        logger.info("Checkpointing it")
        transformer = wrap(
            ["checkpoint"],
            transformer,
            features_dir=checkpoint_dir,
            hash_fn=hash_fn,
            force=force,
        )
    else:
        # If there is only one item that is checkpointed, we don't need to wrap the pipeline
        logger.warning(
            f"{transformer}"
            f"The pipeline contains elements that are already checkpointed."
            "Hence, we are not checkpointing them again."
        )

    # Fetching all samples
    samples = database.all_samples()

    # The number of dasked elements has to be the number of
    # elements in the pipeline - 1 (the ToDaskBag doesn't count)
    dasked_elements = is_pipeline_wrapped(transformer, DaskWrapper)

    if any(dasked_elements):
        logger.warning(
            "The pipeline is already dasked, hence, we are not dasking it again."
        )
    else:
        if not isinstance(dask_client, str):
            dask_partition_size = (
                dask_get_partition_size(
                    dask_client.cluster, len(samples), lower_bound=200
                )
                if dask_partition_size is None
                else dask_partition_size
            )

        logger.info(
            f"Dask wrapping it with partition size {dask_partition_size}"
        )
        transformer = wrap(
            ["dask"], transformer, partition_size=dask_partition_size
        )

    transformer.transform(samples).compute(
        scheduler="single-threaded" if dask_client is None else dask_client
    )

    logger.info("Transformation finished !")
