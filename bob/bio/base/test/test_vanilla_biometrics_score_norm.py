#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.pipelines import Sample, SampleSet, DelayedSample
import os
import numpy as np
import tempfile
from sklearn.pipeline import make_pipeline
from bob.bio.base.wrappers import wrap_bob_legacy

from bob.bio.base.test.test_transformers import (
    FakePreprocesor,
    FakeExtractor,
    FakeAlgorithm,
)
from bob.bio.base.test.test_vanilla_biometrics import DummyDatabase, _make_transformer


from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
    ZTNormVanillaBiometricsPipeline,
    BioAlgorithmCheckpointWrapper,
    dask_vanilla_biometrics,
    BioAlgorithmLegacy,
)

import bob.pipelines as mario
import uuid
import shutil
import itertools


def test_znorm_on_memory():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(with_dask):

            database = DummyDatabase(one_d=False)

            transformer = _make_transformer(dir_name)

            biometric_algorithm = Distance()

            vanilla_biometrics_pipeline = ZTNormVanillaBiometricsPipeline(
                VanillaBiometricsPipeline(transformer, biometric_algorithm)
            )

            if with_dask:
                vanilla_biometrics_pipeline = dask_vanilla_biometrics(
                    vanilla_biometrics_pipeline, npartitions=2
                )

            scores = vanilla_biometrics_pipeline(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                database.zprobes(),
                database.treferences(),
                allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
            )

            if with_dask:
                scores = scores.compute(scheduler="single-threaded")

            assert len(scores) == 10

        run_pipeline(False)
        #run_pipeline(False)  # Testing checkpoint
        # shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        # os.makedirs(dir_name, exist_ok=True)
        # run_pipeline(True)
        # run_pipeline(True)  # Testing checkpoint
