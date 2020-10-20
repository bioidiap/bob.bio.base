import bob.db.base
from bob.db.base.annotations import read_annotation_file
from bob.pipelines.sample import _ReprMixin

class BioFile(bob.db.base.File, _ReprMixin):
    """
    A simple base class that defines basic properties of File object for the use
    in verification experiments

    Attributes
    ----------
    client_id : str or int
        The id of the client this file belongs to.
        Its type depends on your implementation.
        If you use an SQL database, this should be an SQL type like Integer or
        String.
    path : object
        see :py:class:`bob.db.base.File` constructor
    file_id : object
        see :py:class:`bob.db.base.File` constructor
    original_directory : str or None
        The path to the original directory of the file
    original_extension : str or None
        The extension of the original files. This attribute is deprecated.
        Please try to include the extension in the ``path`` attribute
    annotation_directory : str or None
        The path to the directory of the annotations
    annotation_extension : str or None
        The extension of annotation files. Default is ``.json``
    annotation_type : str or None
        The type of the annotation file, see
        :any:`bob.db.base.annotations.read_annotation_file`. Default is
        ``json``.
    """

    def __init__(
        self,
        client_id,
        path,
        file_id=None,
        original_directory=None,
        original_extension=None,
        annotation_directory=None,
        annotation_extension=None,
        annotation_type=None,
        **kwargs,
    ):
        super(BioFile, self).__init__(path, file_id, **kwargs)

        # just copy the information
        self.client_id = client_id
        """The id of the client, to which this file belongs to."""
        self.original_directory = original_directory
        self.original_extension = original_extension
        self.annotation_directory = annotation_directory
        self.annotation_extension = annotation_extension or ".json"
        self.annotation_type = annotation_type or "json"

    def load(self):
        """Loads the data at the specified location and using the given extension.
        Override it if you need to load differently.

        Returns
        -------
        object
            The loaded data (normally :py:class:`numpy.ndarray`).
        """
        # get the path
        path = self.make_path(
            self.original_directory or "", self.original_extension or ""
        )
        return bob.io.base.load(path)

    @property
    def annotations(self):
        path = self.make_path(self.annotation_directory or "", self.annotation_extension or "")
        return read_annotation_file(path, annotation_type=self.annotation_type)


class BioFileSet(BioFile):
    """This class defines the minimum interface of a set of database files that needs to be exported.
    Use this class, whenever the database provides several files that belong to the same probe.
    Each file set has an id, and a list of associated files, which are of
    type :py:class:`bob.bio.base.database.BioFile` of the same client.
    The file set id can be anything hashable, but needs to be unique all over the database.

    Parameters
    ----------

    file_set_id : str or int
        A unique ID that identifies the file set.

    files : [:py:class:`bob.bio.base.database.BioFile`]
        A non-empty list of BioFile objects that should be stored inside this file.
        All files of that list need to have the same client ID.
    """

    def __init__(self, file_set_id, files, path=None, **kwargs):
        # don't accept empty file lists
        assert len(files), "Cannot create an empty BioFileSet"

        # call base class constructor
        super(BioFileSet, self).__init__(
            files[0].client_id,
            "+".join(f.path for f in files) if path is None else path,
            file_set_id,
            **kwargs,
        )

        # check that all files come from the same client
        assert all(f.client_id == self.client_id for f in files)

        # The list of files contained in this set
        self.files = files

    def __lt__(self, other):
        """Defines an order between file sets by using the order of the file set ids."""
        # compare two BioFile set objects by comparing their IDs
        return self.id < other.id
