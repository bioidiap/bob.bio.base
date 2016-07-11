#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu Oct 25 10:05:55 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import print_function

import imp
import os
import pkg_resources

import sys
if sys.version_info[0] == 2:
  from string import letters as ascii_letters
else:
  from string import ascii_letters

import logging
logger = logging.getLogger("bob.bio.base")


#: Keywords for which resources are defined.
valid_keywords = ('database', 'preprocessor', 'extractor', 'algorithm', 'grid')

def read_config_file(filename, keyword = None):
  """read_config_file(filename, keyword = None) -> config

  Use this function to read the given configuration file.
  If a keyword is specified, only the configuration according to this keyword is returned.
  Otherwise a dictionary of the configurations read from the configuration file is returned.

  **Parameters:**

  filename : str
    The name of the configuration file to read.

  keyword : str or ``None``
    If specified, only the contents of the variable with the given name is returned.
    If ``None``, the whole configuration is returned (a local namespace)

  **Returns:**

  config : object or namespace
    If ``keyword`` is specified, the object inside the configuration with the given name is returned.
    Otherwise, the whole configuration is returned (as a local namespace).
  """

  if not os.path.exists(filename):
    raise IOError("The given configuration file '%s' could not be found" % filename)

  import string
  import random
  tmp_config = "".join(random.sample(ascii_letters, 10))
  config = imp.load_source(tmp_config, filename)

  if not keyword:
    return config

  if not hasattr(config, keyword):
    raise ImportError("The desired keyword '%s' does not exist in your configuration file '%s'." %(keyword, filename))

  return eval('config.' + keyword)


def _get_entry_points(keyword, strip = [], package_prefix='bob.bio.'):
  """Returns the list of entry points for registered resources with the given keyword."""
  return  [entry_point for entry_point in pkg_resources.iter_entry_points(package_prefix + keyword) if not entry_point.name.startswith(tuple(strip))]


def load_resource(resource, keyword, imports = ['bob.bio.base'], package_prefix='bob.bio.', preferred_package=None):
  """load_resource(resource, keyword, imports = ['bob.bio.base'], package_prefix='bob.bio.', preferred_package = None) -> resource

  Loads the given resource that is registered with the given keyword.
  The resource can be:

  1. a resource as defined in the setup.py
  2. a configuration file
  3. a string defining the construction of an object. If imports are required for the construction of this object, they can be given as list of strings.

  **Parameters:**

  resource : str
    Any string interpretable as a resource (see above).

  keyword : str
    A valid resource keyword, can be one of :py:attr:`valid_keywords`.

  imports : [str]
    A list of strings defining which modules to import, when constructing new objects (option 3).

  package_prefix : str
    Package namespace, in which we search for entry points, e.g., ``bob.bio``.

  preferred_package : str or ``None``
    When several resources with the same name are found in different packages (e.g., in different ``bob.bio`` or other packages), this specifies the preferred package to load the resource from.
    If not specified, the extension that is **not** from ``bob.bio`` is selected.

  **Returns:**

  resource : object
    The resulting resource object is returned, either read from file or resource, or created newly.
  """

  # first, look if the resource is a file name
  if os.path.isfile(resource):
    return read_config_file(resource, keyword)

  if keyword not in valid_keywords:
    raise ValueError("The given keyword '%s' is not valid. Please use one of %s!" % (str(keyword), str(valid_keywords)))

  # now, we check if the resource is registered as an entry point in the resource files
  entry_points = [entry_point for entry_point in _get_entry_points(keyword, package_prefix=package_prefix) if entry_point.name == resource]

  if len(entry_points):
    if len(entry_points) == 1:
      return entry_points[0].load()
    else:
      # TODO: extract current package name and use this one, if possible

      # Now: check if there are only two entry points, and one is from the bob.bio.base, then use the other one
      index = -1
      if preferred_package is not None:
        for i,p in enumerate(entry_points):
          if p.dist.project_name == preferred_package:
            index = i
            break

      if index == -1:
        # by default, use the first one that is not from bob.bio
        for i,p in enumerate(entry_points):
          if not p.dist.project_name.startswith(package_prefix):
            index = i
            break

      if index != -1:
        logger.debug("RESOURCES: Using the resource '%s' from '%s', and ignoring the one from '%s'", resource, entry_points[index].module_name, entry_points[1-index].module_name)
        return entry_points[index].load()
      else:
        logger.warn("Under the desired name '%s', there are multiple entry points defined, we return the first one: %s", resource, [entry_point.module_name for entry_point in entry_points])
        return entry_points[0].load()


  # if the resource is neither a config file nor an entry point,
  # just execute it as a command
  try:
    # first, execute all import commands that are required
    for i in imports:
      exec ("import %s"%i)
    # now, evaluate the resource (re-evaluate if the resource is still a string)
    while isinstance(resource, str):
      resource = eval(resource)
    return resource

  except Exception as e:
    raise ImportError("The given command line option '%s' is neither a resource for a '%s', nor an existing configuration file, nor could be interpreted as a command (error: %s)"%(resource, keyword, str(e)))


def extensions(keywords=valid_keywords, package_prefix='bob.bio.'):
  """extensions(keywords=valid_keywords, package_prefix='bob.bio.') -> extensions

  Returns a list of packages that define extensions using the given keywords.

  **Parameters:**

  keywords : [str]
    A list of keywords to load entry points for.
    Defaults to all :py:attr:`valid_keywords`.

  package_prefix : str
    Package namespace, in which we search for entry points, e.g., ``bob.bio``.
  """
  entry_points = [entry_point for keyword in keywords for entry_point in _get_entry_points(keyword, package_prefix=package_prefix)]
  return sorted(list(set(entry_point.dist.project_name for entry_point in entry_points)))


def resource_keys(keyword, exclude_packages=[], package_prefix='bob.bio.', strip=['dummy']):
  """Reads and returns all resources that are registered with the given keyword.
  Entry points from the given ``exclude_packages`` are ignored."""
  return sorted([entry_point.name for entry_point in
                 _get_entry_points(keyword, strip=strip, package_prefix=package_prefix)
                 if entry_point.dist.project_name not in exclude_packages])


def list_resources(keyword, strip=['dummy'], package_prefix='bob.bio.', verbose=False, packages=None):
  """Returns a string containing a detailed list of resources that are registered with the given keyword."""
  if keyword not in valid_keywords:
    raise ValueError("The given keyword '%s' is not valid. Please use one of %s!" % (str(keyword), str(valid_keywords)))

  entry_points = _get_entry_points(keyword, strip, package_prefix=package_prefix)
  last_dist = None
  retval = ""
  length = max(len(entry_point.name) for entry_point in entry_points) if entry_points else 1

  if packages is not None:
    entry_points = [entry_point for entry_point in entry_points if entry_point.dist.project_name in packages]

  for entry_point in sorted(entry_points, key=lambda p: (p.dist.project_name, p.name)):
    if last_dist != str(entry_point.dist):
      retval += "\n- %s @ %s: \n" % (str(entry_point.dist), str(entry_point.dist.location))
      last_dist = str(entry_point.dist)

    if len(entry_point.attrs):
      retval += "  + %s --> %s: %s\n" % (entry_point.name + " "*(length - len(entry_point.name)), entry_point.module_name, entry_point.attrs[0])
    else:
      retval += "  + %s --> %s\n" % (entry_point.name + " "*(length - len(entry_point.name)), entry_point.module_name)
    if verbose:
      retval += "    ==> " + str(entry_point.load()) + "\n\n"

  return retval


def database_directories(strip=['dummy'], replacements = None, package_prefix='bob.bio.'):
  """Returns a dictionary of original directories for all registered databases."""
  entry_points = _get_entry_points('database', strip, package_prefix=package_prefix)

  dirs = {}
  for entry_point in sorted(entry_points, key=lambda entry_point: entry_point.name):
    try:
      db = load_resource(entry_point.name, 'database')
      db.replace_directories(replacements)
      dirs[entry_point.name] = [db.original_directory]
      if db.annotation_directory is not None:
        dirs[entry_point.name].append(db.annotation_directory)
    except (AttributeError, ValueError, ImportError):
      pass

  return dirs
