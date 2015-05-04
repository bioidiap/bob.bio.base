import bob.db.atnt
import bob.bio.base
import os

class DummyDatabase (bob.bio.base.database.DatabaseBobZT):

  def __init__(self):
    # call base class constructor with useful parameters
    bob.bio.base.database.DatabaseBobZT.__init__(
        self,
        database = bob.db.atnt.Database(
            original_directory = bob.bio.base.test.utils.atnt_database_directory()
        ),
        name = 'test',
        check_original_files_for_existence = True,
        training_depends_on_protocol = False,
        models_depend_on_protocol = False
    )


  def all_files(self, groups = ['dev']):
    return bob.bio.base.database.DatabaseBob.all_files(self, groups)


  def t_model_ids(self, group = 'dev'):
    return self.model_ids(group)


  def t_enroll_files(self, model_id, group = 'dev'):
    return self.enroll_files(model_id, group)


  def z_probe_files(self, group = 'dev'):
    return self.probe_files(None, group)

database = DummyDatabase()
