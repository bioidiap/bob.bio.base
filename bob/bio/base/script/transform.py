#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from ast import In
import logging
from sched import scheduler

import click
from bob.bio.base.pipelines.vanilla_biometrics import execute_vanilla_biometrics
from bob.extension.scripts.click_helper import ConfigCommand
from bob.extension.scripts.click_helper import ResourceOption
from bob.extension.scripts.click_helper import verbosity_option

from bob.pipelines.distributed import VALID_DASK_CLIENT_STRINGS

from bob.pipelines.distributed import dask_get_partition_size

from bob.pipelines.utils import isinstance_nested, is_estimator_wrapped

logger = logging.getLogger(__name__)
from bob.pipelines.wrappers import (
    wrap,
    SampleWrapper,
    CheckpointWrapper,
    DaskWrapper,
)


import bob.io.base
import bob.io.image


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
For example, if you desire to customize your transfomer, you can use the following configuration file:\n
\b\b


```py\n
from sklearn.base import BaseEstimator, TransformerMixin \n
from sklearn.pipeline import make_pipeline  \n

class MyTransformer(TransformerMixin, BaseEstimator): \n
    def _more_tags(self): \n
        return {"stateless": True, "requires_fit": False} \n
    
        def transform(self, X): \n
        # do something \n
        return X \n

transformer = make_pipeline(MyTransformer()) \n
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

\b

def my_transformer(X): \n
    # do something \n
    return X \n


transformer = make_pipeline(FunctionTransformer(my_transformer)) \n
```

Then, you can use above configuration file to run the command:\n

`bob bio pipelines transform my_database my_transformer.py --dask-client sge -vv`


\b\b
"""


@click.command(
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
    "-e",
    "--file-extension",
    required=True,
    help="File extension of the output files.",
    cls=ResourceOption,
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="If set, it will force generate all the checkpoints of an experiment. This option doesn't work if `--memory` is set",
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def transform(
    transformer,
    database,
    dask_client,
    checkpoint_dir,
    file_extension,
    force,
    **kwargs,
):
    """
    This CLI command will execute a pipeline (a scikit learn Pipeline)  on a given database.

    This command can be used, for example, to extract features, face-crops, preprocess audio files and so on.

    """

    logger.info(f"Transforming samples from {database}")

    save_func = bob.io.base.save

    # Idiap SETUP. This avoids having directories with more than 1000 files/directories
    hash_fn = database.hash_fn if hasattr(database, "hash_fn") else None

    # Checking if the transformer is Sample wrapped
    # In this way we can use any transfomer from skimage, or other package
    # that implements the sklearn API
    if not is_estimator_wrapped(transformer, SampleWrapper):
        logger.info(f"Sample wrapping it")
        transformer = wrap(["sample"], transformer)

    # Checkpoint if it's already checkpointed
    if not is_estimator_wrapped(transformer, CheckpointWrapper):
        logger.info(f"Checkpointing it")
        transformer = wrap(
            ["checkpoint"],
            transformer,
            features_dir=checkpoint_dir,
            extension=file_extension,
            save_func=save_func,
            hash_fn=hash_fn,
            force=force,
        )

    # Fetching all samples
    samples = database.all_samples()

    # Dasking it if it's not already dasked
    if dask_client is not None and not is_estimator_wrapped(transformer, DaskWrapper):
        partition_size = 200
        if not isinstance(dask_client, str):
            partition_size = dask_get_partition_size(
                dask_client.cluster, len(samples), lower_bound=200
            )
        logger.info(f"Dask wrapping it with partition size {partition_size}")
        transformer = wrap(["dask"], transformer, partition_size=partition_size)

    transformer.transform(samples).compute(
        scheduler="single-threaded" if dask_client is None else dask_client
    )

    logger.info("Transformation finished !")
