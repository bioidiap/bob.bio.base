# isort: skip_file
from .csv_dataset import (
    CSVDataset,
    CSVToSampleLoaderBiometrics,
    CSVDatasetCrossValidation,
    LSTToSampleLoader,
    CSVDatasetZTNorm,
)
from .file import BioFile
from .file import BioFileSet
from .database import BioDatabase
from .database import ZTBioDatabase
from .filelist import FileListBioDatabase
from . import filelist  # noqa: F401
from .atnt import AtntBioDatabase  # noqa: F401


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is shortened.
    Parameters:

      *args: An iterable of objects to modify

    Resolves `Sphinx referencing issues
    <https://github.com/sphinx-doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    BioFile,
    BioFileSet,
    BioDatabase,
    ZTBioDatabase,
    CSVDataset,
    CSVToSampleLoaderBiometrics,
    CSVDatasetCrossValidation,
    FileListBioDatabase,
    LSTToSampleLoader,
    CSVDatasetZTNorm,
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
