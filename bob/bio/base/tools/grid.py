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
