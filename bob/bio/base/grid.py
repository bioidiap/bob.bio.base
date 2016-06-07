#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Tue Oct  2 12:12:39 CEST 2012
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


PREDEFINED_QUEUES = {
  'default'     : {},
  '2G'          : {'queue' : 'all.q',  'memfree' : '2G'},
  '4G'          : {'queue' : 'all.q',  'memfree' : '4G'},
  '4G-io-big'   : {'queue' : 'q1d',  'memfree' : '4G', 'io_big' : True},
  '8G'          : {'queue' : 'q1d',  'memfree' : '8G'},
  '8G-io-big'   : {'queue' : 'q1d',  'memfree' : '8G', 'io_big' : True},
  '16G'         : {'queue' : 'q1dm', 'memfree' : '16G', 'pe_opt' : 'pe_mth 2', 'hvmem' : '8G'},
  '16G-io-big'  : {'queue' : 'q1dm', 'memfree' : '16G', 'pe_opt' : 'pe_mth 2', 'hvmem' : '8G', 'io_big' : True},
  '32G'         : {'queue' : 'q1dm', 'memfree' : '32G', 'pe_opt' : 'pe_mth 4', 'hvmem' : '8G', 'io_big' : True},
  '64G'         : {'queue' : 'q1dm', 'memfree' : '64G', 'pe_opt' : 'pe_mth 8', 'hvmem' : '8G', 'io_big' : True},
  'Week'        : {'queue' : 'q1wm', 'memfree' : '32G', 'pe_opt' : 'pe_mth 4', 'hvmem' : '8G'},
  'GPU'         : {'queue' : 'gpu'}
}

from . import utils

class Grid:
  """This class is defining the options that are required to submit parallel jobs to the SGE grid, or jobs to the local queue.

  If the given ``grid_type`` is ``'sge'`` (the default), this configuration is set up to submit algorithms to the SGE grid.
  In this setup, specific SGE queues can be specified for different steps of the tool chain, and different numbers of parallel processes can be specified for each step.
  Currently, only the SGE at Idiap_ is tested and supported, for other SGE's we do not assure compatibility.

  If the given ``grid_type`` is ``'local'``, this configuration is set up to run using a local scheduler on a single machine.
  In this case, only the ``number_of_parallel_processes`` and ``scheduler_sleep_time`` options will be taken into account.

  **Parameters:**

  grid_type : one of ``('sge', 'local')``
    The type of submission system, which should be used.
    Currently, only sge and local submissions are supported.

  number_of_preprocessing_jobs, number_of_extraction_jobs, number_of_projection_jobs, number_of_enrollment_jobs, number_of_scoring_jobs : int
    Only valid if ``grid_type = 'sge'``.
    The number of parallel processes that should be executed for preprocessing, extraction, projection, enrollment or scoring.

  training_queue, preprocessing_queue, extraction_queue, projection_queue, enrollment_queue, scoring_queue : str or dict
    Only valid if ``grid_type = 'sge'``.
    SGE queues that should be used for training, preprocessing, extraction, projection, enrollment or scoring.
    The queue can be defined using a dictionary of keywords that will directly passed to the :py:func:`gridtk.tools.qsub` function, or one of our :py:data:`PREDEFINED_QUEUES`, which are adapted for Idiap_.

  number_of_parallel_processes : int
    Only valid if ``grid_type = 'local'``.
    The number of parallel processes, with which the preprocessing, extraction, projection, enrollment and scoring should be executed.

  scheduler_sleep_time : float
    The time (in seconds) that the local scheduler will sleep between its iterations.
  """

  def __init__(
    self,
    # grid type, currently supported 'local' and 'sge'
    grid_type = 'sge',
    # parameters for the splitting of jobs into array jobs; ignored by the local scheduler
    number_of_preprocessing_jobs = 32,
    number_of_extraction_jobs = 32,
    number_of_projection_jobs = 32,
    number_of_enrollment_jobs = 32,
    number_of_scoring_jobs = 32,

    # queue setup for the SGE grid (only used if grid = 'sge', the default)
    training_queue = '8G',
    preprocessing_queue = 'default',
    extraction_queue = 'default',
    projection_queue = 'default',
    enrollment_queue = 'default',
    scoring_queue = 'default',

    # setup of the local submission and execution of job (only used if grid = 'local')
    number_of_parallel_processes = 1,
    scheduler_sleep_time = 1.0 # sleep time for scheduler in seconds
  ):

    self.grid_type = grid_type
    if self.is_local():
      self._kwargs = dict(grid_type=grid_type, number_of_parallel_processes=number_of_parallel_processes, scheduler_sleep_time=scheduler_sleep_time)
    else:
      self._kwargs = dict(
          grid_type=grid_type,
          number_of_preprocessing_jobs=number_of_preprocessing_jobs, number_of_extraction_jobs=number_of_extraction_jobs, number_of_projection_jobs=number_of_projection_jobs, number_of_enrollment_jobs=number_of_enrollment_jobs,
          training_queue=training_queue, preprocessing_queue=preprocessing_queue, extraction_queue=extraction_queue, projection_queue=projection_queue, enrollment_queue=enrollment_queue, scoring_queue=scoring_queue
      )


    # the numbers
    if self.is_local():
      self.number_of_preprocessing_jobs = number_of_parallel_processes
      self.number_of_extraction_jobs = number_of_parallel_processes
      self.number_of_projection_jobs = number_of_parallel_processes
      self.number_of_enrollment_jobs = number_of_parallel_processes
      self.number_of_scoring_jobs = number_of_parallel_processes
    else:
      self.number_of_preprocessing_jobs = number_of_preprocessing_jobs
      self.number_of_extraction_jobs = number_of_extraction_jobs
      self.number_of_projection_jobs = number_of_projection_jobs
      self.number_of_enrollment_jobs = number_of_enrollment_jobs
      self.number_of_scoring_jobs = number_of_scoring_jobs

    # the queues
    self.training_queue = self.queue(training_queue)
    self.preprocessing_queue = self.queue(preprocessing_queue)
    self.extraction_queue = self.queue(extraction_queue)
    self.projection_queue = self.queue(projection_queue)
    self.enrollment_queue = self.queue(enrollment_queue)
    self.scoring_queue = self.queue(scoring_queue)
    # the local setup
    self.number_of_parallel_processes = number_of_parallel_processes
    self.scheduler_sleep_time = scheduler_sleep_time


  def __str__(self):
    """Converts this grid configuration into a string, which contains the complete set of parameters."""
    return utils.pretty_print(self, self._kwargs)


  def queue(self, params):
    """queue(params) -> dict

    This helper function translates the given queue parameters to grid options.
    When the given ``params`` are a dictionary already, they are simply returned.
    If ``params`` is a string, the :py:data:`PREDEFINED_QUEUES` are indexed with them.
    If ``params`` is ``None``, or the ``grid_type`` is ``'local'``, an empty dictionary is returned.
    """
    if self.is_local():
      return {}
    if isinstance(params, str) and params in PREDEFINED_QUEUES:
      return PREDEFINED_QUEUES[params]
    elif isinstance(params, dict):
      return params
    elif params is None:
      return {}
    else:
      raise ValueError("The given queue parameters '%s' are not in the predefined queues and neither a dictionary with values." % str(params))


  def is_local(self):
    """Returns whether this grid setup should use the local submission or the SGE grid."""
    return self.grid_type == 'local'

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
