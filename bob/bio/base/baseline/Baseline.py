#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
from .. import resource_keys, load_resource


def search_preprocessor(db_name, keys):
  """
  Wrapper that searches for preprocessors for specific databases.
  If not found, the default preprocessor is returned
  """
  for k in keys:
    if db_name.startswith(k):
      return k
  else:
    return "default"


def get_available_databases():
  """
  Get all the available databases through the database entry-points
  """

  available_databases = dict()
  all_databases = resource_keys('database', strip=[])
  for database in all_databases:
    try:
      database_entry_point = load_resource(database, 'database')

      available_databases[database] = dict()

      # Checking if the database has data for the ZT normalization
      available_databases[database]["has_zt"] = hasattr(database_entry_point, "zobjects") and hasattr(database_entry_point, "tobjects")
      available_databases[database]["groups"] = []
      # Searching for database groups
      try:
        groups = list(database_entry_point.groups())
        for g in ["dev", "eval"]:
          available_databases[database]["groups"] += [g] if g in groups else []
      except Exception:
        # In case the method groups is not implemented
        available_databases[database]["groups"] = ["dev"]
    except Exception:
      pass
  return available_databases


class Baseline(object):
  """
  Base class to define baselines

  A Baseline is composed by the triplet
  :any:`bob.bio.base.preprocessor.Preprocessor`,
  :any:`bob.bio.base.extractor.Extractor`, and
  :any:`bob.bio.base.algorithm.Algorithm`

  Attributes
  ----------
  name : str
    Name of the baseline. This name will be displayed in the command line
    interface.
  preprocessors : dict
    Dictionary containing all possible preprocessors
  extractor : str
    Registered resource or a config file containing the feature extractor
  algorithm : str
     Registered resource or a config file containing the algorithm
  """

  def __init__(self, name, preprocessors, extractor, algorithm, **kwargs):
    super(Baseline, self).__init__(**kwargs)
    self.name = name
    self.preprocessors = preprocessors
    self.extractor = extractor
    self.algorithm = algorithm
