# isort: skip_file
from .file import BioFile, BioFileSet
# import BioFile first to avoid circular import
from .database import BioDatabase, ZTBioDatabase
from .csv_dataset import (
    CSVDataset,
    CSVDatasetCrossValidation,
    CSVDatasetZTNorm,
    CSVToSampleLoaderBiometrics,
    LSTToSampleLoader,
)
from .atnt import AtntBioDatabase  # noqa: F401
from . import filelist  # noqa: F401
from .filelist import FileListBioDatabase


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
    CSVDatasetZTNorm,
    CSVToSampleLoaderBiometrics,
    CSVDatasetCrossValidation,
    FileListBioDatabase,
    LSTToSampleLoader,
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
