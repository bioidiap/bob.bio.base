#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""A few checks at the Verification Filelist database.
"""

import os
import bob.io.base
import bob.io.base.test_utils
from bob.bio.base.database import CSVDatasetDevEval, CSVToSampleLoader
import nose.tools
from bob.pipelines import DelayedSample, SampleSet
import numpy as np
from .utils import atnt_database_directory
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap

example_dir = os.path.realpath(
    bob.io.base.test_utils.datafile(".", __name__, "data/example_csv_filelist")
)
atnt_protocol_path = os.path.realpath(
    bob.io.base.test_utils.datafile(".", __name__, "data/atnt")
)


def check_all_true(list_of_something, something):
    """
    Assert if list of `Something` contains `Something`
    """
    return np.alltrue([isinstance(s, something) for s in list_of_something])


def test_csv_file_list_dev_only():

    dataset = CSVDatasetDevEval(example_dir, "protocol_only_dev")
    assert len(dataset.background_model_samples()) == 8
    assert check_all_true(dataset.background_model_samples(), DelayedSample)

    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.probes()) == 10
    assert check_all_true(dataset.probes(), SampleSet)


def test_csv_file_list_dev_only_metadata():

    dataset = CSVDatasetDevEval(example_dir, "protocol_only_dev_metadata")
    assert len(dataset.background_model_samples()) == 8
    assert check_all_true(dataset.background_model_samples(), DelayedSample)
    assert np.alltrue(
        ["METADATA_1" in s.__dict__ for s in dataset.background_model_samples()]
    )
    assert np.alltrue(
        ["METADATA_2" in s.__dict__ for s in dataset.background_model_samples()]
    )

    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)
    assert np.alltrue(["METADATA_1" in s.__dict__ for s in dataset.references()])
    assert np.alltrue(["METADATA_2" in s.__dict__ for s in dataset.references()])

    assert len(dataset.probes()) == 10
    assert check_all_true(dataset.probes(), SampleSet)
    assert np.alltrue(["METADATA_1" in s.__dict__ for s in dataset.probes()])
    assert np.alltrue(["METADATA_2" in s.__dict__ for s in dataset.probes()])
    assert np.alltrue(["references" in s.__dict__ for s in dataset.probes()])


def test_csv_file_list_dev_eval():

    dataset = CSVDatasetDevEval(example_dir, "protocol_dev_eval")
    assert len(dataset.background_model_samples()) == 8
    assert check_all_true(dataset.background_model_samples(), DelayedSample)

    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.probes()) == 10
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.references(group="eval")) == 6
    assert check_all_true(dataset.references(group="eval"), SampleSet)

    assert len(dataset.probes(group="eval")) == 13
    assert check_all_true(dataset.probes(group="eval"), SampleSet)


def test_csv_file_list_atnt():

    dataset = CSVDatasetDevEval(atnt_protocol_path, "idiap_protocol")
    assert len(dataset.background_model_samples()) == 200
    assert len(dataset.references()) == 20
    assert len(dataset.probes()) == 100


def test_atnt_experiment():
    def load(path):
        import bob.io.image

        return bob.io.base.load(path)

    def linearize(X):
        X = np.asarray(X)
        return np.reshape(X, (X.shape[0], -1))

    dataset = CSVDatasetDevEval(
        dataset_protocol_path=atnt_protocol_path,
        protocol_name="idiap_protocol",
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=load,
            dataset_original_directory=atnt_database_directory(),
            extension=".pgm",
        ),
    )

    #### Testing it in a real recognition systems
    transformer = wrap(["sample"], make_pipeline(FunctionTransformer(linearize)))

    vanilla_biometrics_pipeline = VanillaBiometricsPipeline(transformer, Distance())

    scores = vanilla_biometrics_pipeline(
        dataset.background_model_samples(),
        dataset.references(),
        dataset.probes(),
    )

    assert len(scores)==100
    assert np.alltrue([len(s)==20] for s in scores)