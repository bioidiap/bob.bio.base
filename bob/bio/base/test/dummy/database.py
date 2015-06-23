import bob.db.atnt
import os

from bob.bio.base.database import DatabaseBob, DatabaseBobZT
from bob.bio.base.test.utils import atnt_database_directory

class DummyDatabase (DatabaseBobZT):

  def __init__(self):
    # call base class constructor with useful parameters
    DatabaseBobZT.__init__(
        self,
        database = bob.db.atnt.Database(
            original_directory = atnt_database_directory()
        ),
        name = 'test',
        check_original_files_for_existence = True,
        training_depends_on_protocol = False,
        models_depend_on_protocol = False
    )

  def all_files(self, groups = ['dev']):
    return DatabaseBob.all_files(self, groups)

  def t_model_ids(self, group = 'dev'):
    return self.model_ids(group)

  def t_enroll_files(self, t_model_id, group = 'dev'):
    return self.enroll_files(t_model_id, group)

  def z_probe_files(self, group = 'dev'):
    return self.probe_files(None, group)

database = DummyDatabase()
