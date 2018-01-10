#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.db.base


class BioFile(bob.db.base.File):
    """A simple base class that defines basic properties of File object for the use in verification experiments

    Parameters
    ----------

    client_id : object
      The id of the client this file belongs to.
      Its type depends on your implementation.
      If you use an SQL database, this should be an SQL type like Integer or String.

    file_id : object
      see :py:class:`bob.db.base.File` constructor

    path : object
      see :py:class:`bob.db.base.File` constructor
    """

    def __init__(self, client_id, path, file_id=None, **kwargs):
        super(BioFile, self).__init__(path, file_id, **kwargs)

        # just copy the information
        self.client_id = client_id
        """The id of the client, to which this file belongs to."""


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
            file_set_id, **kwargs)

        # check that all files come from the same client
        assert all(f.client_id == self.client_id for f in files)

        # The list of files contained in this set
        self.files = files

    def __lt__(self, other):
        """Defines an order between file sets by using the order of the file set ids."""
        # compare two BioFile set objects by comparing their IDs
        return self.id < other.id
