import bob.bio.base

# define the queue using all the default parameters
grid = bob.bio.base.grid.Grid(
  grid_type = 'local',
  number_of_parallel_processes = 4
)


# define a queue that is highly parallelized
grid_p8 = bob.bio.base.grid.Grid(
  grid_type = 'local',
  number_of_parallel_processes = 8
)

# define a queue that is highly parallelized
grid_p16 = bob.bio.base.grid.Grid(
  grid_type = 'local',
  number_of_parallel_processes = 16
)
