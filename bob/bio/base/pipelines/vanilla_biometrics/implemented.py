import scipy.spatial.distance
from sklearn.utils.validation import check_array
import numpy
from .abstract_classes import BioAlgorithm
from .mixins import BioAlgCheckpointMixin


class Distance(BioAlgorithm):
    def __init__(self, distance_function=scipy.spatial.distance.euclidean, factor=-1):

        self.distance_function = distance_function
        self.factor = factor

    def enroll(self, enroll_features):
        """enroll(enroll_features) -> model

        Enrolls the model by storing all given input vectors.

        Parameters:
        -----------

        ``enroll_features`` : [:py:class:`numpy.ndarray`]
          The list of projected features to enroll the model from.

        Returns:
        --------

        ``model`` : 2D :py:class:`numpy.ndarray`
          The enrolled model.
        """

        enroll_features = check_array(enroll_features, allow_nd=True)

        return numpy.mean(enroll_features, axis=0)

    def score(self, model, probe):
        """score(model, probe) -> float

        Computes the distance of the model to the probe using the distance function specified in the constructor.

        Parameters:
        -----------

        ``model`` : 2D :py:class:`numpy.ndarray`
          The model storing all enrollment features

        ``probe`` : :py:class:`numpy.ndarray`
          The probe feature vector

        Returns:
        --------

        ``score`` : float
          A similarity value between ``model`` and ``probe``
        """

        probe = probe.flatten()
        # return the negative distance (as a similarity measure)
        return self.factor * self.distance_function(model, probe)


class CheckpointDistance(BioAlgCheckpointMixin, Distance):
    pass
