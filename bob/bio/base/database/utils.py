class File:
  """This class defines the minimum interface of a file that needs to be exported"""

  def __init__(self, file_id, client_id, path):
    # The **unique** id of the file
    self.id = file_id
    # The id of the client that is attached to the file
    self.client_id = client_id
    # The **relative** path of the file according to the base directory of the database, without file extension
    self.path = path

  def __lt__(self, other):
    # compare two File objects by comparing their IDs
    return self.id < other.id


class FileSet:
  """This class defines the minimum interface of a file set that needs to be exported"""

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
