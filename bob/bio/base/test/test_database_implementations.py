#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>
# Wed 20 July 16:20:12 CEST 2016
#

"""
Very simple tests for Implementations
"""

import bob.bio.base

from bob.bio.base.test.dummy.database import database as dummy_database
from bob.pipelines import DelayedSample


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
    assert len(database.background_model_samples()) > 0
    assert len(database.references()) > 0
    assert len(database.probes()) > 0
    assert len(database.all_samples()) > 0
