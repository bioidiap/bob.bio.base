from __future__ import print_function

import sys
import os
import math
from .. import grid
from .command_line import command_line

import bob.core
import logging
logger = logging.getLogger("bob.bio.base")

def indices(list_to_split, number_of_parallel_jobs, task_id=None):
  """This function returns the first and last index for the files for the current job ID.
     If no job id is set (e.g., because a sub-job is executed locally), it simply returns all indices."""

  if number_of_parallel_jobs is None or number_of_parallel_jobs == 1:
    return None

  # test if the 'SEG_TASK_ID' environment is set
  sge_task_id = os.getenv('SGE_TASK_ID') if task_id is None else task_id
  if sge_task_id is None:
    # task id is not set, so this function is not called from a grid job
    # hence, we process the whole list
    return (0,len(list_to_split))
  else:
    job_id = int(sge_task_id) - 1
    # compute number of files to be executed
    number_of_objects_per_job = int(math.ceil(float(len(list_to_split) / float(number_of_parallel_jobs))))
    start = job_id * number_of_objects_per_job
    end = min((job_id + 1) * number_of_objects_per_job, len(list_to_split))
    return (start, end)


class GridSubmission:
  def __init__(self, args, command_line_parameters, executable = 'verify.py', first_fake_job_id = 0):
    # find, where the executable is installed
    import bob.extension

    if command_line_parameters is None:
      command_line_parameters = sys.argv[1:]

    executables = bob.extension.find_executable(executable, prefixes = [os.path.dirname(sys.argv[0]), 'bin'])
    if not len(executables):
      raise IOError("Could not find the '%s' executable." % executable)
    executable = executables[0]
    assert os.path.isfile(executable)
    self.executable = executable

    if args.grid is not None:
      assert isinstance(args.grid, grid.Grid)
      
      if(hasattr(args,'env')):
        self.env = args.env #Fetching the enviroment variable
      else:
        self.env = None

      # find, where jman is installed
      jmans = bob.extension.find_executable('jman', prefixes = ['bin'])
      if not len(jmans):
        raise IOError("Could not find the 'jman' executable. Have you installed GridTK?")
      jman = jmans[0]
      assert os.path.isfile(jman)

      self.args = args
      self.command_line = [p for p in command_line_parameters if not p.startswith('--skip') and p not in ('-q', '--dry-run')]
      self.fake_job_id = first_fake_job_id

      import gridtk
      # setup logger
      bob.core.log.set_verbosity_level(bob.core.log.setup("gridtk"), min(args.verbose,2))
      Manager = gridtk.local.JobManagerLocal if args.grid.is_local() else gridtk.sge.JobManagerSGE
      self.job_manager = Manager(database = args.gridtk_database_file, wrapper_script=jman)
      self.submitted_job_ids = []


  def submit(self, command, number_of_parallel_jobs = 1, dependencies=[], name = None, **kwargs):
    """Submit a grid job with the given command, which is added to the default command line.
    If the name is not given, it will take the second parameter of the ``command`` as name.
    """
    dependencies = dependencies + self.args.external_dependencies

    # create the command to be executed
    cmd = [self.executable] + self.command_line
    cmd += command.split()

    # if no job name is specified, create one
    if name is None:
      name = command.split()[1]
    # generate log directory
    log_dir = os.path.join(self.args.grid_log_directory, name)

    # generate job array
    if number_of_parallel_jobs > 1:
      array = (1,number_of_parallel_jobs,1)
    else:
      array = None

    # submit the job to the job manager
    if not self.args.dry_run:
      if(self.env is not None):
        kwargs['env'] = self.env

      job_id = self.job_manager.submit(
          command_line = cmd,
          name = name,
          array = array,
          dependencies = dependencies,
          log_dir = log_dir,
          stop_on_failure = self.args.stop_on_failure,
          **kwargs
      )
      logger.info("submitted: job '%s' with id '%d' and dependencies '%s'" % (name, job_id, dependencies))
      self.submitted_job_ids.append(job_id)
      return job_id
    else:
      self.fake_job_id += 1
      print ('would have submitted job', name, 'with id', self.fake_job_id, 'with parameters', kwargs, end='')
      if array:
        print (' using', array[1], 'parallel jobs', end='')
      print (' as:', command_line(cmd), '\nwith dependencies', dependencies)
      return self.fake_job_id


  def execute_local(self):
    """Starts the local deamon and waits until it has finished."""
    logger.info("Starting jman deamon to run the jobs on the local machine.")
    failures = self.job_manager.run_scheduler(job_ids=self.submitted_job_ids, parallel_jobs=self.args.grid.number_of_parallel_processes, sleep_time=self.args.grid.scheduler_sleep_time, die_when_finished=True, nice=self.args.nice)
    if failures:
      logger.error("The jobs with the following IDS did not finish successfully: '%s'.", ', '.join([str(f) for f in failures]))
      self.job_manager.report(job_ids = failures[:1], output=False)

    # delete the jobs that we have added
    if self.args.delete_jobs_finished_with_status is not None:
      logger.info("Deleting jman jobs that we have added")
      status = ('success', 'failure') if self.args.delete_jobs_finished_with_status == 'all' else (self.args.delete_jobs_finished_with_status,)
      self.job_manager.delete(job_ids=self.submitted_job_ids, status=status)
