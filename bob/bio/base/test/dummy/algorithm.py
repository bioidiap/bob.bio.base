import scipy.spatial
import bob.io.base

from bob.bio.base.algorithm import Algorithm

_data = [5., 6., 7., 8., 9.]

class DummyAlgorithm (Algorithm):
  """This class is used to test all the possible functions of the tool chain, but it does basically nothing."""

  def __init__(self, **kwargs):
    """Generates a test value that is read and written"""

    # call base class constructor registering that this tool performs everything.
    Algorithm.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = True
    )

  def _test(self, file_name):
    """Simply tests that the read data is consistent"""
    data = bob.io.base.load(file_name)
    assert (_data == data).all()

  def train_projector(self, train_files, projector_file):
    """Does not train the projector, but writes some file"""
    # save something
    bob.io.base.save(_data, projector_file)

  def load_projector(self, projector_file):
    """Loads the test value from file and compares it with the desired one"""
    self._test(projector_file)

  def project(self, feature):
    """Just returns the feature since this dummy implementation does not really project the data"""
    return feature

  def train_enroller(self, train_files, enroller_file):
    """Does not train the projector, but writes some file"""
    # save something
    bob.io.base.save(_data, enroller_file)

  def load_enroller(self, enroller_file):
    """Loads the test value from file and compares it with the desired one"""
    self._test(enroller_file)

  def enroll(self, enroll_features):
    """Returns the first feature as the model"""
    assert len(enroll_features)
    # just return the first feature
    return enroll_features[0]

  def score(self, model, probe):
    """Returns the Euclidean distance between model and probe"""
    return scipy.spatial.distance.euclidean(model, probe)

algorithm = DummyAlgorithm()
