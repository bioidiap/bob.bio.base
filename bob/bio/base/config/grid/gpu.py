import bob.bio.base

# define a queue with demanding parameters
grid = bob.bio.base.grid.Grid(
  training_queue = 'GPU',
  # preprocessing
  preprocessing_queue = '4G',
  # feature extraction
  extraction_queue = 'GPU',
  # feature projection
  projection_queue = '4G',
  # model enrollment
  enrollment_queue = '4G',
  # scoring
  scoring_queue = '4G'
)
