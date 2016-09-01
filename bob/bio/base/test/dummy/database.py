from bob.bio.db import ZTBioDatabase, AtntBioDatabase
from bob.bio.base.test.utils import atnt_database_directory


class DummyDatabase(ZTBioDatabase):

    def __init__(self):
        # call base class constructor with useful parameters
        super(DummyDatabase, self).__init__(
            name='test',
            original_directory=atnt_database_directory(),
            original_extension='.pgm',
            check_original_files_for_existence=True,
            training_depends_on_protocol=False,
            models_depend_on_protocol=False
        )
        self.__db = AtntBioDatabase()

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids_with_protocol(groups, protocol)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        return self.__db.objects(groups, protocol, purposes, model_ids, **kwargs)

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        return []

    def zobjects(self, groups=None, protocol=None, **kwargs):
        return []

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        return self.__db.model_ids_with_protocol(groups, protocol)

    def t_enroll_files(self, t_model_id, group='dev'):
        return self.enroll_files(t_model_id, group)

    def z_probe_files(self, group='dev'):
        return self.probe_files(None, group)

database = DummyDatabase()
