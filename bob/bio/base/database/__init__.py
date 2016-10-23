from .file import BioFile
from .file import BioFileSet
from .database import BioDatabase
from .database import ZTBioDatabase

# to fix sphinx warnings of not being able to find classes, when path is shortened
BioFile.__module__ = "bob.bio.base.database"
BioFileSet.__module__ = "bob.bio.base.database"
BioDatabase.__module__ = "bob.bio.base.database"
ZTBioDatabase.__module__ = "bob.bio.base.database"

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
