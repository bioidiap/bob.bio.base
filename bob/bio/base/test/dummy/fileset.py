from bob.bio.base.database import ZTBioDatabase, BioFileSet, BioFile
from bob.bio.base.test.utils import atnt_database_directory


class DummyDatabase(ZTBioDatabase):

    def __init__(self):
        # call base class constructor with useful parameters
        super(DummyDatabase, self).__init__(
            name='test_fileset',
            original_directory=atnt_database_directory(),
            original_extension='.pgm',
            check_original_files_for_existence=True,
            training_depends_on_protocol=False,
            models_depend_on_protocol=False
        )
        import bob.db.atnt
        self._db = bob.db.atnt.Database()

    def uses_probe_file_sets(self):
        return True

    def _make_bio(self, files):
      return [BioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in files]

    def object_sets(self, groups='dev', protocol=None, purposes=None, model_ids=None):
        """Returns the list of probe File objects (for the given model id, if given)."""
        files = self.arrange_by_client(self.sort(self.objects(protocol=None, groups=groups, purposes=purposes)))
        # arrange files by clients
        file_sets = [BioFileSet(client_files[0].client_id, client_files) for client_files in files]
        return file_sets

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

    def z_probe_file_sets(self, group='dev'):
        return self.probe_file_sets(None, group)

    def annotations(self, file):
        return None


database = DummyDatabase()
