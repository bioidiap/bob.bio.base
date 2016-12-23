#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""A few checks at the Verification Filelist database.
"""

import os
import bob.io.base.test_utils
from bob.bio.base.database import FileListBioDatabase


example_dir = os.path.realpath(bob.io.base.test_utils.datafile('.', __name__, 'data/example_fielist'))


def test_query():
    db = FileListBioDatabase(example_dir, 'test', use_dense_probe_file_list=False)

    assert len(db.groups()) == 5  # 5 groups (dev, eval, world, optional_world_1, optional_world_2)

    assert len(db.client_ids()) == 6  # 6 client ids for world, dev and eval
    assert len(db.client_ids(groups='world')) == 2  # 2 client ids for world
    assert len(db.client_ids(groups='optional_world_1')) == 2  # 2 client ids for optional world 1
    assert len(db.client_ids(groups='optional_world_2')) == 2  # 2 client ids for optional world 2
    assert len(db.client_ids(groups='dev')) == 2  # 2 client ids for dev
    assert len(db.client_ids(groups='eval')) == 2  # 2 client ids for eval

    assert len(db.tclient_ids()) == 2  # 2 client ids for T-Norm score normalization
    assert len(db.zclient_ids()) == 2  # 2 client ids for Z-Norm score normalization

    assert len(db.model_ids_with_protocol()) == 6  # 6 model ids for world, dev and eval
    assert len(db.model_ids_with_protocol(groups='world')) == 2  # 2 model ids for world
    assert len(db.model_ids_with_protocol(groups='optional_world_1')) == 2  # 2 model ids for optional world 1
    assert len(db.model_ids_with_protocol(groups='optional_world_2')) == 2  # 2 model ids for optional world 2
    assert len(db.model_ids_with_protocol(groups='dev')) == 2  # 2 model ids for dev
    assert len(db.model_ids_with_protocol(groups='eval')) == 2  # 2 model ids for eval

    assert len(db.tmodel_ids_with_protocol()) == 2  # 2 model ids for T-Norm score normalization

    assert len(db.objects(groups='world')) == 8  # 8 samples in the world set

    assert len(db.objects(groups='dev', purposes='enroll')) == 8  # 8 samples for enrollment in the dev set
    assert len(db.objects(groups='dev', purposes='enroll',
                          model_ids='3')) == 4  # 4 samples for to enroll model '3' in the dev set
    assert len(db.objects(groups='dev', purposes='enroll',
                          model_ids='7')) == 0  # 0 samples for enrolling model '7' (it is a T-Norm model)
    assert len(db.objects(groups='dev', purposes='probe')) == 8  # 8 samples as probes in the dev set
    assert len(
        db.objects(groups='dev', purposes='probe', classes='client')) == 8  # 8 samples as client probes in the dev set
    assert len(db.objects(groups='dev', purposes='probe',
                          classes='impostor')) == 4  # 4 samples as impostor probes in the dev set

    assert len(db.tobjects(groups='dev')) == 8  # 8 samples for enrolling T-norm models
    assert len(db.tobjects(groups='dev', model_ids='7')) == 4  # 4 samples for enrolling T-norm model '7'
    assert len(
        db.tobjects(groups='dev', model_ids='3')) == 0  # 0 samples for enrolling T-norm model '3' (no T-Norm model)
    assert len(db.zobjects(groups='dev')) == 8  # 8 samples for Z-norm impostor accesses

    assert db.client_id_from_model_id('1', group=None) == '1'
    assert db.client_id_from_model_id('3', group=None) == '3'
    assert db.client_id_from_model_id('6', group=None) == '6'
    assert db.client_id_from_t_model_id('7', group=None) == '7'


def test_query_protocol():
    db = FileListBioDatabase(os.path.dirname(example_dir), 'test', protocol='example_fielist', use_dense_probe_file_list=False)

    assert len(db.groups()) == 5  # 5 groups (dev, eval, world, optional_world_1, optional_world_2)

    assert len(db.client_ids()) == 6  # 6 client ids for world, dev and eval
    assert len(db.client_ids(groups='world', )) == 2  # 2 client ids for world
    assert len(db.client_ids(groups='optional_world_1', )) == 2  # 2 client ids for optional world 1
    assert len(db.client_ids(groups='optional_world_2', )) == 2  # 2 client ids for optional world 2
    assert len(db.client_ids(groups='dev', )) == 2  # 2 client ids for dev
    assert len(db.client_ids(groups='eval', )) == 2  # 2 client ids for eval

    assert len(db.tclient_ids()) == 2  # 2 client ids for T-Norm score normalization
    assert len(db.zclient_ids()) == 2  # 2 client ids for Z-Norm score normalization

    assert len(db.model_ids_with_protocol()) == 6  # 6 model ids for world, dev and eval
    assert len(db.model_ids_with_protocol(groups='world', )) == 2  # 2 model ids for world
    assert len(db.model_ids_with_protocol(groups='optional_world_1', )) == 2  # 2 model ids for optional world 1
    assert len(db.model_ids_with_protocol(groups='optional_world_2', )) == 2  # 2 model ids for optional world 2
    assert len(db.model_ids_with_protocol(groups='dev', )) == 2  # 2 model ids for dev
    assert len(db.model_ids_with_protocol(groups='eval', )) == 2  # 2 model ids for eval

    assert len(db.tmodel_ids_with_protocol()) == 2  # 2 model ids for T-Norm score normalization

    assert len(db.objects(groups='world', )) == 8  # 8 samples in the world set

    assert len(db.objects(groups='dev', purposes='enroll', )) == 8  # 8 samples for enrollment in the dev set
    assert len(db.objects(groups='dev', purposes='enroll', model_ids='3',
                          )) == 4  # 4 samples for to enroll model '3' in the dev set
    assert len(db.objects(groups='dev', purposes='enroll', model_ids='7',
                          )) == 0  # 0 samples for enrolling model '7' (it is a T-Norm model)
    assert len(db.objects(groups='dev', purposes='probe', )) == 8  # 8 samples as probes in the dev set
    assert len(db.objects(groups='dev', purposes='probe', classes='client',
                          )) == 8  # 8 samples as client probes in the dev set
    assert len(db.objects(groups='dev', purposes='probe', classes='impostor',
                          )) == 4  # 4 samples as impostor probes in the dev set

    assert len(db.tobjects(groups='dev', )) == 8  # 8 samples for enrolling T-norm models
    assert len(db.tobjects(groups='dev', model_ids='7', )) == 4  # 4 samples for enrolling T-norm model '7'
    assert len(db.tobjects(groups='dev', model_ids='3',
                           )) == 0  # 0 samples for enrolling T-norm model '3' (no T-Norm model)
    assert len(db.zobjects(groups='dev')) == 8  # 8 samples for Z-norm impostor accesses

    assert db.client_id_from_model_id('1', group=None) == '1'
    assert db.client_id_from_model_id('3', group=None) == '3'
    assert db.client_id_from_model_id('6', group=None) == '6'
    assert db.client_id_from_t_model_id('7', group=None) == '7'


def test_query_dense():
    db = FileListBioDatabase(example_dir, 'test', probes_filename='for_probes.lst')

    assert len(db.objects(groups='world')) == 8  # 8 samples in the world set

    assert len(db.objects(groups='dev', purposes='enroll')) == 8  # 8 samples for enrollment in the dev set
    assert len(db.objects(groups='dev', purposes='probe')) == 8  # 8 samples as probes in the dev set


def test_annotation():
    db = FileListBioDatabase(example_dir, 'test', use_dense_probe_file_list=False,
                             annotation_directory=example_dir, annotation_type='named')
    f = [o for o in db.objects() if o.path == "data/model4_session1_sample2"][0]
    annots = db.annotations(f)

    assert annots is not None
    assert 'key1' in annots
    assert 'key2' in annots
    assert annots['key1'] == (20, 10)
    assert annots['key2'] == (40, 30)


def test_multiple_extensions():
    # check that the old behavior still works
    db = FileListBioDatabase(example_dir, 'test', use_dense_probe_file_list=False,
                             original_directory=example_dir, original_extension='.pos')
    file = bob.bio.base.database.BioFile(4, "data/model4_session1_sample2", "data/model4_session1_sample2")
    file_name = db.original_file_name(file, True)
    assert file_name == os.path.join(example_dir, file.path + '.pos')

    # check that the new behavior works as well
    db = FileListBioDatabase(example_dir, 'test', use_dense_probe_file_list=False,
                             original_directory=example_dir, original_extension=['.jpg', '.pos'])
    file_name = db.original_file_name(file)
    assert file_name == os.path.join(example_dir, file.path + '.pos')

    file = bob.bio.base.database.BioFile(4, "data/model4_session1_sample1", "data/model4_session1_sample1")
    try:
        file_name = db.original_file_name(file, False)
        raised = False
    except IOError as e:
        raised = True

    assert raised


def test_driver_api():
    from bob.db.base.script.dbmanage import main
    assert main(('bio_filelist dumplist --list-directory=%s --self-test' % example_dir).split()) == 0
    assert main((
                'bio_filelist dumplist --list-directory=%s --purpose=enroll --group=dev --class=client --self-test' % example_dir).split()) == 0
    assert main(('bio_filelist checkfiles --list-directory=%s --self-test' % example_dir).split()) == 0
