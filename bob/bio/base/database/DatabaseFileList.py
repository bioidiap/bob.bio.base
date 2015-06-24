#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Wed Oct  3 10:31:51 CEST 2012
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


from .DatabaseBob import DatabaseBobZT
import bob.db.verification.filelist

class DatabaseFileList (DatabaseBobZT):
  """This class can be used whenever you have a database that uses the Bob filelist database interface, which is defined in :py:class:`bob.db.verification.filelist.Database`

  **Parameters:**

  database : a :py:class:`bob.db.verification.filelist.Database`
    The database instance that provides the actual interface.

  kwargs : ``key=value`` pairs
    The arguments of the :py:class:`DatabaseBobZT` or :py:class:`DatabaseBob` base class constructors.

    .. note:: Usually, the ``name``, ``protocol``, ``training_depends_on_protocol`` and ``models_depend_on_protocol`` keyword parameters of the base class constructor need to be specified.
  """

  def __init__(
      self,
      database,  # The bob database that is used
      **kwargs  # The default parameters of the base class
  ):

    DatabaseBobZT.__init__(
        self,
        database = database,
        **kwargs
    )

    assert isinstance(database, bob.db.verification.filelist.Database)


  def all_files(self, groups = ['dev']):
    """all_files(groups=None) -> files

    Returns all files of the database, respecting the current protocol.
    If the current protocol is ``'None'``, ``None`` will be used instead.
    When the underlying file list database provides files for ZT score normalization, these files are returned as well.
    The files can be limited using the ``all_files_options`` in the constructor.

    **Parameters:**

    groups : some of ``('world', 'dev', 'eval')`` or ``None``
      The groups to get the data for.
      If ``None``, data for all groups is returned.

    **Returns:**

    files : [:py:class:`bob.db.verification.filelist.File`]
      The sorted and unique list of all files of the database.
    """
    protocol = self.protocol if self.protocol != 'None' else None
    files = self.database.objects(protocol = protocol, groups = groups, **self.all_files_options)

    # add all files that belong to the ZT-norm
    for group in groups:
      if group == 'world': continue
      if self.database.implements_zt(protocol = protocol, groups = group):
        files += self.database.tobjects(protocol = protocol, groups = group, model_ids = None)
        files += self.database.zobjects(protocol = protocol, groups = group, **self.z_probe_options)
    return self.sort(files)


  def uses_probe_file_sets(self):
    """File sets are not (yet) supported in the :py:class:`bob.db.verification.filelist.Database`, so this function returns ``False`` throughout."""
    return False


  def model_ids(self, group = 'dev'):
    """model_ids(group = 'dev') -> ids

    Returns a list of model ids for the given group, respecting the current protocol.
    If the current protocol is ``'None'``, ``None`` will be used instead.

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the model ids for.

    **Returns:**

    ids : [str]
      The list of (unique) model ids for the given group.
    """
    return sorted(self.database.model_ids(protocol = self.protocol if self.protocol != 'None' else None, groups = group))


  def client_id_from_model_id(self, model_id, group = 'dev'):
    """client_id_from_model_id(model_id, group = 'dev') -> client_id

    Uses :py:meth:`bob.db.verification.filelist.Database.get_client_id_from_model_id` to retrieve the client id for the given model id.
    If the current protocol is ``'None'``, ``None`` will be used instead.

    **Parameters:**

    model_id : str
      A unique ID that identifies the model for the client.

    group : one of ``('dev', 'eval')``
      The group to get the client ids for.

    **Returns:**

    client_id : str
      A unique ID that identifies the client, to which the model belongs.
    """
    return self.database.get_client_id_from_model_id(model_id, groups = group, protocol = self.protocol if self.protocol != 'None' else None)


  def client_id_from_t_model_id(self, t_model_id, group = 'dev'):
    """client_id_from_t_model_idt_(model_id, group = 'dev') -> client_id

    Uses :py:meth:`bob.db.verification.filelist.Database.get_client_id_from_t_model_id` to retrieve the client id for the T-norm given model id.
    If the current protocol is ``'None'``, ``None`` will be used instead.

    **Parameters:**

    t_model_id : str
      A unique ID that identifies the T-Norm model.

    group : one of ``('dev', 'eval')``
      The group to get the client ids for.

    **Returns:**

    client_id : str
      A unique ID that identifies the client, to which the T-Norm model belongs.
    """
    return self.database.get_client_id_from_tmodel_id(t_model_id, groups = group, protocol = self.protocol if self.protocol != 'None' else None)


  def t_model_ids(self, group = 'dev'):
    """t_model_ids(group = 'dev') -> ids

    Returns a list of model ids of T-Norm models for the given group, respecting the current protocol.
    If the current protocol is ``'None'``, ``None`` will be used instead.

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the model ids for.

    **Returns:**

    ids : [int] or [str]
      The list of (unique) model ids for T-Norm models of the given group.
    """
    return sorted(self.database.tmodel_ids(protocol = self.protocol if self.protocol != 'None' else None, groups = group))
