from bob.bio.base.database import ZTBioDatabase
from bob.bio.base.database.file import BioFile
from bob.bio.base.test.utils import atnt_database_directory
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector

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
        import bob.db.atnt
        self._db = bob.db.atnt.Database()

    def _make_bio(self, files):
      return [BioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in files]

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self._db.model_ids(groups, protocol)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        return self._make_bio(self._db.objects(model_ids, groups, purposes, protocol, **kwargs))

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        return []

    def zobjects(self, groups=None, protocol=None, **kwargs):
        return []

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        return self._db.model_ids(groups)

    def t_enroll_files(self, t_model_id, group='dev'):
        return self.enroll_files(t_model_id, group)

    def z_probe_files(self, group='dev'):
        return self.probe_files(None, group)

    def annotations(self, file):
        return None

    def groups(self, protocol=None):
        return self._db.groups(protocol)


database = DatabaseConnector(DummyDatabase())
