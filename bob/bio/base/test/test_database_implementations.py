#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>
# Wed 20 July 16:20:12 CEST 2016
#

"""
Very simple tests for Implementations
"""

import os
from bob.bio.base.database import BioDatabase, ZTBioDatabase
from bob.bio.base.test.dummy.database import database as dummy_database
from bob.pipelines import DelayedSample

def check_database(database, groups=('dev',), protocol=None, training_depends=False, models_depend=False, skip_train=False, check_zt=False):
    database_legacy = database.database
    assert isinstance(database_legacy, BioDatabase)

    # load the directories
    if 'HOME' in os.environ:
        database_legacy.replace_directories(os.path.join(os.environ['HOME'], '.bob_bio_databases.txt'))

    if protocol:
        database_legacy.protocol = protocol
    if protocol is None:
        protocol = database_legacy.protocol

    assert len(database_legacy.all_files(add_zt_files=check_zt)) > 0
    if not skip_train:
        assert len(database_legacy.training_files('train_extractor')) > 0
        assert len(database_legacy.arrange_by_client(database_legacy.training_files('train_enroller'))) > 0

    for group in groups:
        model_ids = database_legacy.model_ids_with_protocol(group, protocol=protocol)
        assert len(model_ids) > 0
        assert database_legacy.client_id_from_model_id(model_ids[0], group) is not None
        assert len(database_legacy.enroll_files(model_ids[0], group)) > 0
        assert len(database_legacy.probe_files(model_ids[0], group)) > 0

    assert database_legacy.training_depends_on_protocol == training_depends
    assert database_legacy.models_depend_on_protocol == models_depend


def check_database_zt(database, groups=('dev', 'eval'), protocol=None, training_depends=False, models_depend=False):
    database_legacy = database.database
    check_database(database, groups, protocol, training_depends, models_depend, check_zt=True)
    assert isinstance(database_legacy, ZTBioDatabase)
    for group in groups:
        t_model_ids = database_legacy.t_model_ids(group)
        assert len(t_model_ids) > 0
        assert database_legacy.client_id_from_model_id(t_model_ids[0], group) is not None
        assert len(database_legacy.t_enroll_files(t_model_ids[0], group)) > 0
        assert len(database_legacy.z_probe_files(group)) > 0

def test_all_samples():
    all_samples = dummy_database.all_samples(groups=None)
    assert len(all_samples) == 400
    assert all([isinstance(s, DelayedSample) for s in all_samples])
    assert len(dummy_database.all_samples(groups=["train"])) == 200
    assert len(dummy_database.all_samples(groups=["dev"])) == 200
    assert len(dummy_database.all_samples(groups=[])) == 400
