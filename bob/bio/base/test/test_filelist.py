#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""A few checks at the Verification Filelist database.
"""

import os
import bob.io.base.test_utils
from bob.bio.base.database import CSVDatasetDevEval
import nose.tools
from bob.pipelines import DelayedSample, SampleSet
import numpy as np

example_dir = os.path.realpath(
    bob.io.base.test_utils.datafile(".", __name__, "data/example_csv_filelist")
)
atnt_dir = os.path.realpath(bob.io.base.test_utils.datafile(".", __name__, "data/atnt"))


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
    assert np.alltrue(['METADATA_1' in s.__dict__ for s in dataset.background_model_samples()])
    assert np.alltrue(['METADATA_2' in s.__dict__ for s in dataset.background_model_samples()])    


    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)
    assert np.alltrue(['METADATA_1' in s.__dict__ for s in dataset.references()])
    assert np.alltrue(['METADATA_2' in s.__dict__ for s in dataset.references()])    


    assert len(dataset.probes()) == 10
    assert check_all_true(dataset.probes(), SampleSet)
    assert np.alltrue(['METADATA_1' in s.__dict__ for s in dataset.probes()])
    assert np.alltrue(['METADATA_2' in s.__dict__ for s in dataset.probes()])    


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

    dataset = CSVDatasetDevEval(atnt_dir, "idiap_protocol")
    assert len(dataset.background_model_samples()) == 200
    assert len(dataset.references()) == 20
    assert len(dataset.probes()) == 100
