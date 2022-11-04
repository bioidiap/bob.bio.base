"""Interface between the lower level GMM classes and the Algorithm Transformer.

Implements the enroll and score methods using the low level GMM implementation.

This adds the notions of models, probes, enrollment, and scores to GMM.
"""


import copy
import logging

from typing import Callable, Union

import dask.array as da
import numpy as np

from h5py import File as HDF5File

from bob.bio.base.pipelines import BioAlgorithm
from bob.learn.em import GMMMachine, GMMStats, linear_scoring

logger = logging.getLogger(__name__)


def check_data_dim(data, expected_ndim):
    """Stacks the features into a matrix of shape (n_samples, n_features) or
    labels into shape of (n_samples,) if the input data is not like that already

    Parameters
    ----------
    data : array-like
        features or labels
    expected_ndim : int
        expected number of dimensions of the data

    Returns
    -------
    stacked_data : array-like
        stacked features or labels if needed
    """
    if expected_ndim not in (1, 2):
        raise ValueError(
            f"expected_ndim must be 1 or 2 but got {expected_ndim}"
        )

    if expected_ndim == 1:
        stack_function = np.concatenate
    else:
        stack_function = np.vstack

    if data[0].ndim == expected_ndim:
        return stack_function(data)

    return data


class GMM(GMMMachine, BioAlgorithm):
    """Algorithm for computing UBM and Gaussian Mixture Models of the features.

    Features must be normalized to zero mean and unit standard deviation.

    Models are MAP GMM machines trained from a UBM on the enrollment feature set.

    The UBM is a ML GMM machine trained on the training feature set.

    Probes are GMM statistics of features projected on the UBM.
    """

    def __init__(
        self,
        # parameters for the GMM
        n_gaussians: int,
        # parameters of UBM training
        k_means_trainer=None,
        max_fitting_steps: int = 25,  # Maximum number of iterations for GMM Training
        convergence_threshold: float = 5e-4,  # Threshold to end the ML training
        mean_var_update_threshold: float = 5e-4,  # Minimum value that a variance can reach
        update_means: bool = True,
        update_variances: bool = True,
        update_weights: bool = True,
        # parameters of the GMM enrollment (MAP)
        enroll_iterations: int = 1,
        enroll_update_means: bool = True,
        enroll_update_variances: bool = False,
        enroll_update_weights: bool = False,
        enroll_relevance_factor: Union[float, None] = 4,
        enroll_alpha: float = 0.5,
        # scoring
        scoring_function: Callable = linear_scoring,
        # RNG
        random_state: int = 5489,
        # other
        return_stats_in_transform: bool = False,
        **kwargs,
    ):
        """Initializes the local UBM-GMM tool chain.

        Parameters
        ----------
        n_gaussians
            The number of Gaussians used in the UBM and the models.
        kmeans_trainer
            The kmeans machine used to train and initialize the UBM.
        kmeans_init_iterations
            Number of iterations used for setting the k-means initial centroids.
            if None, will use the same as kmeans_training_iterations.
        kmeans_oversampling_factor
            Oversampling factor used by k-means initializer.
        max_fitting_steps
            Number of e-m iterations for training the UBM.
        convergence_threshold
            Convergence threshold to halt the GMM training early.
        mean_var_update_threshold
            Minimum value a variance of the Gaussians can reach.
        update_weights
            Decides wether the weights of the Gaussians are updated while training.
        update_means
            Decides wether the means of the Gaussians are updated while training.
        update_variances
            Decides wether the variances of the Gaussians are updated while training.
        enroll_iterations
            Number of iterations for the MAP GMM used for enrollment.
        enroll_update_weights
            Decides wether the weights of the Gaussians are updated while enrolling.
        enroll_update_means
            Decides wether the means of the Gaussians are updated while enrolling.
        enroll_update_variances
            Decides wether the variances of the Gaussians are updated while enrolling.
        enroll_relevance_factor
            For enrollment: MAP relevance factor as described in Reynolds paper.
            If None, will not apply Reynolds adaptation.
        enroll_alpha
            For enrollment: MAP adaptation coefficient.
        random_state
            Seed for the random number generation.
        scoring_function
            Function returning a score from a model, a UBM, and a probe.
        """
        super().__init__(
            n_gaussians=n_gaussians,
            trainer="ml",
            max_fitting_steps=max_fitting_steps,
            convergence_threshold=convergence_threshold,
            update_means=update_means,
            update_variances=update_variances,
            update_weights=update_weights,
            mean_var_update_threshold=mean_var_update_threshold,
            k_means_trainer=k_means_trainer,
            random_state=random_state,
            **kwargs,
        )

        self.enroll_relevance_factor = enroll_relevance_factor
        self.enroll_alpha = enroll_alpha
        self.enroll_iterations = enroll_iterations
        self.enroll_update_means = enroll_update_means
        self.enroll_update_weights = enroll_update_weights
        self.enroll_update_variances = enroll_update_variances
        self.scoring_function = scoring_function
        self.return_stats_in_transform = return_stats_in_transform

    def save_model(self, ubm_file):
        """Saves the projector (UBM) to file."""
        super().save(ubm_file)

    def load_model(self, ubm_file):
        """Loads the projector (UBM) from a file."""
        super().load(ubm_file)

    def project(self, array):
        """Computes GMM statistics against a UBM, given a 2D array of feature vectors

        This is applied to the probes before scoring.
        """
        array = check_data_dim(array, expected_ndim=2)
        logger.debug("Projecting %d feature vectors", array.shape[0])
        # Accumulates statistics
        gmm_stats = self.acc_stats(array)

        # Return the resulting statistics
        return gmm_stats

    def enroll(self, data):
        """Enrolls a GMM using MAP adaptation given a reference's feature vectors

        Returns a GMMMachine tuned from the UBM with MAP on a biometric reference data.
        """

        # if input is a list (or SampleBatch) of 2 dimensional arrays, stack them
        data = check_data_dim(data, expected_ndim=2)

        # Use the array to train a GMM and return it
        logger.info("Enrolling with %d feature vectors", data.shape[0])

        gmm = GMMMachine(
            n_gaussians=self.n_gaussians,
            trainer="map",
            ubm=copy.deepcopy(self),
            convergence_threshold=self.convergence_threshold,
            max_fitting_steps=self.enroll_iterations,
            random_state=self.random_state,
            update_means=self.enroll_update_means,
            update_variances=self.enroll_update_variances,
            update_weights=self.enroll_update_weights,
            mean_var_update_threshold=self.mean_var_update_threshold,
            map_relevance_factor=self.enroll_relevance_factor,
            map_alpha=self.enroll_alpha,
        )
        gmm.fit(data)
        return gmm

    def create_templates(self, list_of_feature_sets, enroll):
        if enroll:
            return [
                self.enroll(feature_set) for feature_set in list_of_feature_sets
            ]
        else:
            return [
                self.project(feature_set)
                for feature_set in list_of_feature_sets
            ]

    def compare(self, enroll_templates, probe_templates):
        return self.scoring_function(
            models_means=enroll_templates,
            ubm=self,
            test_stats=probe_templates,
            frame_length_normalization=True,
        )

    def read_biometric_reference(self, model_file):
        """Reads an enrolled reference model, which is a MAP GMMMachine."""
        return GMMMachine.from_hdf5(HDF5File(model_file, "r"), ubm=self)

    def write_biometric_reference(self, model: GMMMachine, model_file):
        """Write the enrolled reference (MAP GMMMachine) into a file."""
        return model.save(model_file)

    def fit(self, X, y=None, **kwargs):
        """Trains the UBM."""
        # Stack all the samples in a 2D array of features
        if isinstance(X, da.Array):
            X = X.persist()

        # if input is a list (or SampleBatch) of 2 dimensional arrays, stack them
        X = check_data_dim(X, expected_ndim=2)

        logger.debug(
            f"Training UBM machine with {self.n_gaussians} gaussians and {len(X)} samples"
        )

        super().fit(X)

        return self

    def transform(self, X, **kwargs):
        """Passthrough. Enroll applies a different transform as score."""
        # The idea would be to apply the projection in Transform (going from extracted
        # to GMMStats), but we must not apply this during the training or enrollment
        # (those require extracted data directly, not projected).
        # `project` is applied in the score function directly.
        if not self.return_stats_in_transform:
            return X
        return super().transform(X)

    @classmethod
    def custom_enrolled_save_fn(cls, data, path):
        data.save(path)

    def custom_enrolled_load_fn(self, path):
        return GMMMachine.from_hdf5(path, ubm=self)

    def _more_tags(self):
        return {
            "bob_fit_supports_dask_array": True,
            "bob_features_save_fn": GMMStats.save,
            "bob_features_load_fn": GMMStats.from_hdf5,
            "bob_enrolled_save_fn": self.custom_enrolled_save_fn,
            "bob_enrolled_load_fn": self.custom_enrolled_load_fn,
            "bob_checkpoint_features": self.return_stats_in_transform,
        }
