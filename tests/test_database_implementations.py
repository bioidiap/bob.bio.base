#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>
# Wed 20 July 16:20:12 CEST 2016
#

"""
Very simple tests for Implementations
"""

from pathlib import Path

import bob.bio.base

from bob.bio.base.config.dummy.database import database as dummy_database
from bob.bio.base.database import CSVDatabase
from bob.pipelines import DelayedSample, SampleSet

DATA_DIR = Path(__file__).parent / "data"


def test_all_samples():
    all_samples = dummy_database.all_samples(groups=None)
    assert len(all_samples) == 400
    assert all([isinstance(s, DelayedSample) for s in all_samples])
    assert len(dummy_database.all_samples(groups=["train"])) == 200
    assert len(dummy_database.all_samples(groups=["dev"])) == 200
    assert len(dummy_database.all_samples(groups=[])) == 400


def test_atnt():
    database = bob.bio.base.load_resource(
        "atnt", "database", preferred_package="bob.bio.base"
    )
    train_set = database.background_model_samples()
    assert len(train_set) > 0
    assert isinstance(train_set[0], DelayedSample)
    references = database.references()
    assert len(references) > 0
    references_sset = references[0]
    assert isinstance(references_sset, SampleSet)
    assert hasattr(references_sset, "key")
    assert hasattr(references_sset, "subject_id")
    assert hasattr(references_sset, "template_id")
    references_sample = references_sset.samples[0]
    assert isinstance(references_sample, DelayedSample)
    assert hasattr(references_sample, "key")
    probes = database.probes()
    assert len(probes) > 0
    assert isinstance(probes[0], SampleSet)
    assert isinstance(probes[0].samples[0], DelayedSample)
    all_samples = database.all_samples()
    assert len(all_samples) > 0
    assert isinstance(all_samples[0], DelayedSample)


def test_metadata():
    local_protocol_definition_path = DATA_DIR / "example_csv_filelist"
    database = CSVDatabase(
        name="dummy_metadata",
        protocol="protocol_only_dev_metadata",
        dataset_protocols_path=local_protocol_definition_path,
        templates_metadata=["subject_metadata"],
    )
    references_sset = database.references()[0]
    assert hasattr(references_sset, "subject_metadata")
    references_sample = references_sset.samples[0]
    assert hasattr(references_sample, "sample_metadata")
    probes_sset = database.probes()[0]
    assert hasattr(probes_sset, "subject_metadata")
    probes_sample = probes_sset.samples[0]
    assert hasattr(probes_sample, "sample_metadata")
