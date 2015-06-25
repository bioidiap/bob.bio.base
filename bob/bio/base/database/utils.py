class File:
  """This class defines the minimum interface of a database file that needs to be exported.

  Each file has a path, an id and an associated client (aka. identity, person, user).
  Usually, this file is part of a database, which has a common directory for all files.
  The path of this file is usually *relative* to that common directory, and it is usually stored *without* filename extension.
  The file id can be anything hashable, but needs to be unique all over the database.
  The client id can be anything hashable, but needs to be identical for different files of the same client, and different between clients.

  **Parameters:**

  file_id : str or int
    A unique ID that identifies the file.
    This ID might be identical to the ``path``, though integral IDs perform faster.

  client_id : str or int
    A unique ID that identifies the client (user) to which this file belongs.
    This ID might be the name of the person, though integral IDs perform faster.

  path : str
    The file path of the file, which is relative to the common database directory, and without filename extension.
  """

  def __init__(self, file_id, client_id, path):
    # The **unique** id of the file
    self.id = file_id
    # The id of the client that is attached to the file
    self.client_id = client_id
    # The **relative** path of the file according to the base directory of the database, without file extension
    self.path = path

  def __lt__(self, other):
    """Defines an order between files by using the order of the file ids."""
    # compare two File objects by comparing their IDs
    return self.id < other.id


class FileSet:
  """This class defines the minimum interface of a set of database files that needs to be exported.

  Use this class, whenever the database provides several files that belong to the same probe.

  Each file set has an id, an associated client (aka. identity, person, user), and a list of associated files.
  Usually, these files are part of a database, which has a common directory for all files.
  The paths of this file set are usually *relative* to that common directory, and they are usually stored *without* filename extension.
  The file id can be anything hashable, but needs to be unique all over the database.
  The client id can be anything hashable, but needs to be identical for different files of the same client, and different between clients.

  **Parameters:**

  file_id : str or int
    A unique ID that identifies the file.
    This ID might be identical to the ``path``, though integral IDs perform faster.

  client_id : str or int
    A unique ID that identifies the client (user) to which this file belongs.
    This ID might be the name of the person, though integral IDs perform faster.

  path : str
    The file path of the file, which is relative to the common database directory, and without filename extension.
  """

  def __init__(self, file_set_id, client_id, file_set_name):
    # The **unique** id of the file set
    self.id = file_set_id
    # The id of the client that is attached to the file
    self.client_id = client_id
    # A name of the file set
    self.path = file_set_name
    # The list of files contained in this set
    self.files = []

  def __lt__(self, other):
    # compare two File set objects by comparing their IDs
    return self.id < other.id
