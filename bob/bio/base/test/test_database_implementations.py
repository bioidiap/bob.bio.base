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


def check_database(database, groups=('dev',), protocol=None, training_depends=False, models_depend=False, skip_train=False):
    assert isinstance(database, BioDatabase)

    # load the directories
    if 'HOME' in os.environ:
        database.replace_directories(os.path.join(os.environ['HOME'], '.bob_bio_databases.txt'))

    if protocol: database.protocol = protocol
    if protocol is None: protocol = database.protocol

    assert len(database.all_files()) > 0
    if not skip_train:
        assert len(database.training_files('train_extractor')) > 0
        assert len(database.arrange_by_client(database.training_files('train_enroller'))) > 0

    for group in groups:
        model_ids = database.model_ids_with_protocol(group, protocol=protocol)
        assert len(model_ids) > 0
        assert database.client_id_from_model_id(model_ids[0]) is not None
        assert len(database.enroll_files(model_ids[0], group)) > 0
        assert len(database.probe_files(model_ids[0], group)) > 0

    assert database.training_depends_on_protocol == training_depends
    assert database.models_depend_on_protocol == models_depend


def check_database_zt(database, groups=('dev', 'eval'), protocol=None, training_depends=False, models_depend=False):
    check_database(database, groups, protocol, training_depends, models_depend)
    assert isinstance(database, ZTBioDatabase)
    for group in groups:
        t_model_ids = database.t_model_ids(group)
        assert len(t_model_ids) > 0
        assert database.client_id_from_model_id(t_model_ids[0]) is not None
        assert len(database.t_enroll_files(t_model_ids[0], group)) > 0
        assert len(database.z_probe_files(group)) > 0


