#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
from nose.plugins.skip import SkipTest

import bob.bio.base

import pkg_resources
dummy_dir = pkg_resources.resource_filename('bob.bio.base', 'test/dummy')

def test_verification_filelist():
  try:
    db1 = bob.bio.base.load_resource(os.path.join(dummy_dir, 'database.py'), 'database')
  except Exception as e:
    raise SkipTest("This test is skipped since the atnt database is not available.")
  try:
    db2 = bob.bio.base.load_resource(os.path.join(dummy_dir, 'filelist.py'), 'database')
  except Exception as e:
    raise SkipTest("This test is skipped since the verification.filelist database is not available.")
  # The test of the verification.filelist database is a bit different.
  # here, we test the output of two different ways of querying the AT&T database
  # where actually both ways are uncommon...

  # assure that different kind of queries result in the same file lists
  assert set([str(id) for id in db1.model_ids()]) ==  set(db2.model_ids())
  assert set([str(id) for id in db1.t_model_ids()]) == set(db2.t_model_ids())

  def _check_files(f1, f2):
    assert set([file.path for file in f1]) == set([file.path for file in f2])

  _check_files(db1.all_files(), db2.all_files())
  _check_files(db1.training_files('train_extractor'), db2.training_files('train_extractor'))
  _check_files(db1.enroll_files(model_id=22), db2.enroll_files(model_id='22'))
  _check_files(db1.probe_files(model_id=22), db2.probe_files(model_id='22'))

  _check_files(db1.t_enroll_files(t_model_id=22), db2.t_enroll_files(t_model_id='22'))
  _check_files(db1.z_probe_files(), db2.z_probe_files())

  f1 = db1.all_files()[0]
  f2 = [f for f in db2.all_files() if f.path == f1.path][0]

  assert f1.make_path(directory='xx', extension='.yy') == f2.make_path(directory='xx', extension='.yy')

  m1 = sorted([str(id) for id in db1.model_ids()])[0]
  m2 = sorted([str(id) for id in db2.model_ids()])[0]
  assert str(db1.client_id_from_model_id(m1)) == db2.client_id_from_model_id(m2)
