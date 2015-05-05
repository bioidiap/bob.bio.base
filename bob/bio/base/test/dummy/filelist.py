import bob.db.verification.filelist
from bob.bio.base.database import DatabaseFileList
from bob.bio.base.test.utils import atnt_database_directory
import pkg_resources

database = DatabaseFileList(
  database = bob.db.verification.filelist.Database(
    base_dir = pkg_resources.resource_filename('bob.bio.base.test', 'data/atnt'),
    original_directory = atnt_database_directory(),
    original_extension = ".pgm",
    dev_subdir = '.',
    eval_subdir = '.',
    world_filename = 'world.lst',
    models_filename = 'models.lst',
    probes_filename = 'probes.lst',
    tnorm_filename = 'models.lst',
    znorm_filename = 'probes.lst',
    keep_read_lists_in_memory = True
  ),
  name = 'test_filelist',
  protocol = None,
  check_original_files_for_existence = True,
  training_depends_on_protocol = False,
  models_depend_on_protocol = False
)
