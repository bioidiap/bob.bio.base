import bob.bio.base

# define a queue with demanding parameters
grid = bob.bio.base.grid.Grid(
  training_queue = '32G',
  # preprocessing
  preprocessing_queue = '4G-io-big',
  # feature extraction
  extraction_queue = '8G-io-big',
  # feature projection
  projection_queue = '8G-io-big',
  # model enrollment
  enrollment_queue = '8G-io-big',
  # scoring
  scoring_queue = '8G-io-big'
)
