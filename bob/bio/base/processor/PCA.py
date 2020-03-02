#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.bio.base.pipelines.vanilla_biometrics.blocks import VanillaBiometricsAlgoritm
import sklearn.decomposition
from scipy.spatial.distance import euclidean
import numpy

import logging

logger = logging.getLogger("bob.bio.base")


class PCA(VanillaBiometricsAlgoritm):
    """Performs a principal component analysis (PCA) on the given data.

  This algorithm computes a PCA projection (:py:class:`bob.learn.linear.PCATrainer`) on the given training features, projects the features to eigenspace and computes the distance of two projected features in eigenspace.
  For example, the eigenface algorithm as proposed by [TP91]_ can be run with this class.

  **Parameters:**

  subspace_dimension : int or float
    If specified as ``int``, defines the number of eigenvectors used in the PCA projection matrix.
    If specified as ``float`` (between 0 and 1), the number of eigenvectors is calculated such that the given percentage of variance is kept.

  distance_function : function
    A function taking two parameters and returns a float.
    If ``uses_variances`` is set to ``True``, the function is provided with a third parameter, which is the vector of variances (aka. eigenvalues).

  svd_solver: std
    The way to solve the eigen value problem

  factor: float
     Multiplication factor used for the scoring stage

  kwargs : ``key=value`` pairs
    A list of keyword arguments directly passed to the :py:class:`Algorithm` base class constructor.
  """

    def __init__(
        self,
        subspace_dimension,  # if int, number of subspace dimensions; if float, percentage of variance to keep
        distance_function=euclidean,
        svd_solver="auto",
        factor=-1,
        **kwargs,  # parameters directly sent to the base class
    ):

        # call base class constructor and register that the algorithm performs a projection
        super(PCA, self).__init__(performs_projection=True)

        self.subspace_dim = subspace_dimension
        self.distance_function = distance_function
        self.svd_solver = svd_solver
        self.factor = -1

    def fit(self, samplesets, checkpoints):
        """
        This method should implement the sub-pipeline 0 of the Vanilla Biometrics Pipeline :ref:`_vanilla-pipeline-0`.

        It represents the training of background models that an algorithm may need.

        Parameters
        ----------

            samplesets: :py:class:`bob.pipelines.sample.sample.SampleSet`
                         Set of samples used to train a background model


            checkpoint: str
                If provided, must the path leading to a location where this
                model should be saved at (complete path without extension) -
                currently, it needs to be provided because of existing
                serialization requirements (see bob/bob.io.base#106), but
                checkpointing will still work as expected.
         
        """

        pca = sklearn.decomposition.PCA(self.subspace_dim, svd_solver=self.svd_solver)
        samples_array = self._stack_samples_2_ndarray(samplesets)
        logger.info(
            "Training PCA with samples of shape {0}".format(samples_array.shape)
        )
        pca.fit(samples_array)

        # TODO: save the shit

        return pca

    def project_one_sample(self, background_model, data):
        if data.ndim == 1:
            return background_model.transform(data.reshape(1, -1))

        return background_model.transform(data)

    def enroll_one_sample(self, data):
        return numpy.mean(data, axis=0)

    def score_one_sample(self, biometric_reference, data):
        """It handles the score computation for one sample

        Parameters
        ----------

            biometric_reference : list
                Biometric reference to be compared

            data : list
                Data to be compared

        Returns
        -------

            scores : list
                For each sample in a probe, returns as many scores as there are
                samples in the probe, together with the probe's and the
                relevant reference's subject identifiers.

        """

        return self.factor * self.distance_function(biometric_reference, data)
