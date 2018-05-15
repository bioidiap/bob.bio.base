from .Baseline import Baseline
import bob.bio.base

 
def get_available_databases():
    """
    Get all the available databases through the database entry-points
    """
    
    available_databases = dict()
    all_databases = bob.bio.base.resource_keys('database', strip=[])
    for database in all_databases:        
        try:               
            database_entry_point = bob.bio.base.load_resource(database, 'database')

            available_databases[database] = dict()

            # Checking if the database has data for the ZT normalization
            available_databases[database]["has_zt"] = hasattr(database_entry_point, "zobjects") and hasattr(database_entry_point, "tobjects")
            available_databases[database]["groups"] = []
            # Searching for database groups
            try:
                groups = list(database_entry_point.groups())
                for g in ["dev", "eval"]:
                    available_databases[database]["groups"] += [g] if g in groups else []
            except:
                # In case the method groups is not implemented
                available_databases[database]["groups"] = ["dev"]
        except:
            pass
    return available_databases


def get_config():
  """Returns a string containing the configuration information.
  """

  import bob.extension
  return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
