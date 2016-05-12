import bob.bio.base

from . import utils

def test_filename():
  # load extractor
  preprocessor = bob.bio.base.load_resource("filename", "preprocessor", preferred_package = 'bob.bio.base')
  assert isinstance(preprocessor, bob.bio.base.preprocessor.Preprocessor)
  assert isinstance(preprocessor, bob.bio.base.preprocessor.Filename)

  # try to load the original image
  assert preprocessor.read_original_data("/any/path") is None

  # try to process
  assert preprocessor(None, None) == 1

  # try to write
  preprocessor.write_data(None, "/any/path")

  # read a file
  assert preprocessor.read_data("/any/file.name") == "/any/file"
