import numpy as np
import scipy.spatial.distance

from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_array

from .abstract_classes import BioAlgorithm


class Distance(BioAlgorithm):
    def __init__(
        self,
        distance_function=scipy.spatial.distance.cosine,
        factor=-1,
        average_on_enroll=True,
        **kwargs
    ):
        """

        Parameters
        ----------

        ``distance_function``:
          function to be used to measure the distance of probe and model features.

        ``factor``: float
          a coefficient which is multiplied to distance (after distance_function) to find score between probe and model features.

        ``average_on_enroll``: bool:
          [this flag is useful when there are multiple samples to be enrolled for each user]
          If True, the average of (multiple available) features of each user is calculated in the enrollment, and then the score
            for each probe is calculated comparing "probe" vs "average of reference features".
          If False, all the available reference features are enrolled for the user, and then the score for each probe is calculated
            using the average of scores (score of comparing "probe" vs each of the multiple available reference features for the
            enrolled user).
        """
        super().__init__(**kwargs)
        self.distance_function = distance_function
        self.factor = factor
        self.average_on_enroll = average_on_enroll

    def _make_2d(self, X):
        """
        This function will make sure that the inputs are ndim=2 before enrollment and scoring.

        For instance, when the source is `VideoLikeContainer` the input of `enroll:enroll_features` and  `score:probes` are
        [`VideoLikeContainer`, ....].
        The concatenation of them makes and array of `ZxNxD`. Hence we need to stack them in `Z`.

        """
        if X.ndim == 3:
            return np.vstack(X)
        elif X.ndim == 1:
            return np.expand_dims(X, axis=0)
        else:
            return X

    def enroll(self, enroll_features):
        """enroll(enroll_features) -> model

        Enrolls the model by storing all given input vectors.

        Parameters
        ----------

        ``enroll_features`` : [:py:class:`numpy.ndarray`]
          The list of projected features to enroll the model from.

        Returns
        -------

        ``model`` : 2D :py:class:`numpy.ndarray`
          The enrolled model.
        """

        enroll_features = check_array(
            enroll_features, allow_nd=True, ensure_2d=True
        )

        enroll_features = self._make_2d(enroll_features)

        # This avoids some possible mistakes in the feature extraction
        # That dumps vectors in the format `Nx1xd`
        assert enroll_features.ndim == 2

        return (
            np.mean(enroll_features, axis=0)
            if self.average_on_enroll
            else enroll_features
        )

    def score(self, biometric_reference, data):
        """score(model, probe) -> float

        Computes the distance of the model to the probe using the distance function specified in the constructor.

        Parameters
        ----------

        ``model`` : 2D :py:class:`numpy.ndarray`
          The model storing all enrollment features

        ``probe`` : :py:class:`numpy.ndarray`
          The probe feature vector

        Returns
        -------

        ``score`` : float
          A similarity value between ``model`` and ``probe``

        """

        # We have to do this `check_array` because we can
        # have other array formats that are not necessarily numpy arrays but extensions of it
        data = check_array(data, allow_nd=True, ensure_2d=False)

        data = self._make_2d(data)

        assert data.ndim == 2

        # return the negative distance (as a similarity measure)
        scores = self.factor * self.distance_function(biometric_reference, data)

        return scores if self.average_on_enroll else np.mean(scores)

    def score_multiple_biometric_references(self, biometric_references, data):

        # We have to do this `check_array` because we can
        # have other array formats that are not necessarily numpy arrays but extensions of it
        data = check_array(data, allow_nd=True, ensure_2d=False)

        data = self._make_2d(data)

        assert data.ndim == 2

        if self.average_on_enroll:
            references_stacked = np.vstack(biometric_references)
            scores = self.factor * cdist(
                references_stacked, data, self.distance_function
            )

        else:
            scores = [
                self.factor
                * cdist(
                    np.vstack(reference), data, self.distance_function
                ).flatten()
                for reference in biometric_references
            ]

        return scores
