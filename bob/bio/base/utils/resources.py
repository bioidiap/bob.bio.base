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


valid_keywords = ('database', 'preprocessor', 'extractor', 'algorithm', 'grid')


def read_config_file(filename, keyword = None):
  """Use this function to read the given configuration file.
  If a keyword is specified, only the configuration according to this keyword is returned.
  Otherwise a dictionary of the configurations read from the configuration file is returned."""

  if not os.path.exists(filename):
    raise IOError("The given configuration file '%s' could not be found" % file)

  import string
  import random
  tmp_config = "".join(random.sample(ascii_letters, 10))
  config = imp.load_source(tmp_config, filename)

  if not keyword:
    return config

  if not hasattr(config, keyword):
    raise ImportError("The desired keyword '%s' does not exist in your configuration file '%s'." %(keyword, filename))

  return eval('config.' + keyword)


def _get_entry_points(keyword, strip = []):
  """Returns the list of entry points for registered resources with the given keyword."""
  return  [entry_point for entry_point in pkg_resources.iter_entry_points('bob.bio.' + keyword) if not entry_point.name.startswith(tuple(strip))]


def load_resource(resource, keyword, imports = ['bob.bio.base'], preferred_distribution = None):
  """Loads the given resource that is registered with the given keyword.
  The resource can be:

    * a resource as defined in the setup.py
    * a configuration file
    * a string defining the construction of an object. If imports are required for the construction of this object, they can be given as list of strings.

  In any case, the resulting resource object is returned.
  """

  # first, look if the resource is a file name
  if os.path.isfile(resource):
    return read_config_file(resource, keyword)

  if keyword not in valid_keywords:
    raise ValueError("The given keyword '%s' is not valid. Please use one of %s!" % (str(keyword), str(valid_keywords)))

  # now, we check if the resource is registered as an entry point in the resource files
  entry_points = [entry_point for entry_point in _get_entry_points(keyword) if entry_point.name == resource]

  if len(entry_points):
    if len(entry_points) == 1:
      return entry_points[0].load()
    else:
      # TODO: extract current package name and use this one, if possible

      # Now: check if there are only two entry points, and one is from the bob.bio.base, then use the other one
      index = -1
      if preferred_distribution:
        for i,p in enumerate(entry_points):
          if p.dist.project_name == preferred_distribution: index = i

      if index == -1:
        if len(entry_points) == 2:
          if entry_points[0].dist.project_name == 'bob.bio.base': index = 1
          elif entry_points[1].dist.project_name == 'bob.bio.base': index = 0

      if index != -1:
        logger.info("RESOURCES: Using the resource '%s' from '%s', and ignoring the one from '%s'", resource, entry_points[index].module_name, entry_points[1-index].module_name)
        return entry_points[index].load()
      else:
        raise ImportError("Under the desired name '%s', there are multiple entry points defined: %s" %(resource, [entry_point.module_name for entry_point in entry_points]))

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


def read_file_resource(resource, keyword):
  """Treats the given resource as a file and reads its configuration"""
  # first, look if the resource is a file name
  if os.path.isfile(resource):
    # load it without the keyword -> all entries of the resource file are read
    return read_config_file(resource)

  if keyword not in valid_keywords:
    raise ValueError("The given keyword '%s' is not valid. Please use one of %s!" % (str(keyword), str(valid_keywords)))

  entry_points = [entry_point for entry_point in _get_entry_points(keyword) if entry_point.name == resource]

  if not len(entry_points):
    raise ImportError("The given option '%s' is neither a resource, nor an existing configuration file for resource type '%s'"%(resource, keyword))

  if len(entry_points) == 1:
    return entry_points[0].load()
  else:
    # TODO: extract current package name and use this one, if possible

    # Now: check if there are only two entry points, and one is from the bob.bio.base, then use the other one
    index = -1
    if len(entry_points) == 2:
      if entry_points[0].dist.project_name == 'bob.bio.base': index = 1
      elif entry_points[1].dist.project_name == 'bob.bio.base': index = 0

    if index != -1:
      logger.info("RESOURCES: Using the resource '%s' from '%s', and ignoring the one from '%s'" %(resource, entry_points[index].module_name, entry_points[1-index].module_name))
      return entry_points[index].load()
    else:
      raise ImportError("Under the desired name '%s', there are multiple entry points defined: %s" %(resource, [entry_point.module_name for entry_point in entry_points]))


def extensions(keywords=valid_keywords):
  """Returns a list of packages that define extensions using the given keywords, which default to all keywords."""
  entry_points = [entry_point for keyword in keywords for entry_point in _get_entry_points(keyword)]
  return sorted(list(set(entry_point.dist.project_name for entry_point in entry_points)))


def resource_keys(keyword, exclude_packages=[], strip=['dummy']):
  """Reads and returns all resources that are registered with the given keyword.
  Entry points from the given ``exclude_packages`` are ignored."""
  return sorted([entry_point.name for entry_point in _get_entry_points(keyword, strip) if entry_point.dist.project_name not in exclude_packages])


def list_resources(keyword, strip=['dummy']):
  """Returns a string containing a detailed list of resources that are registered with the given keyword."""
  if keyword not in valid_keywords:
    raise ValueError("The given keyword '%s' is not valid. Please use one of %s!" % (str(keyword), str(valid_keywords)))

  entry_points = _get_entry_points(keyword, strip)
  last_dist = None
  retval = ""
  length = max(len(entry_point.name) for entry_point in entry_points)

  for entry_point in sorted(entry_points):
    if last_dist != str(entry_point.dist):
      retval += "\n- %s: \n" % str(entry_point.dist)
      last_dist = str(entry_point.dist)

    if len(entry_point.attrs):
      retval += "  + %s --> %s: %s\n" % (entry_point.name + " "*(length - len(entry_point.name)), entry_point.module_name, entry_point.attrs[0])
    else:
      retval += "  + %s --> %s\n" % (entry_point.name + " "*(length - len(entry_point.name)), entry_point.module_name)
  return retval


def database_directories(strip=['dummy'], replacements = None):
  """Returns a dictionary of original directories for all registered databases."""
  entry_points = _get_entry_points('database', strip)

  dirs = {}
  for entry_point in sorted(entry_points):
    try:
      db = load_resource(entry_point.name, 'database')
      db.replace_directories(replacements)
      dirs[entry_point.name] = [db.original_directory]
#      import ipdb; ipdb.set_trace()
      if db.annotation_directory is not None:
        dirs[entry_point.name].append(db.annotation_directory)
    except (AttributeError, ValueError):
      pass

  return dirs
