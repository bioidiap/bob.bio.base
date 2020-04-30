#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Implementation of the Vanilla Biometrics pipeline using Dask :ref:`bob.bio.base.struct_bio_rec_sys`_

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import logging
import numpy

logger = logging.getLogger(__name__)


class VanillaBiometrics(object):
    """
    Vanilla Biometrics Pipeline

    This is the backbone of most biometric recognition systems.
    It implements three subpipelines and they are the following:

     - :py:class:`VanillaBiometrics.train_background_model`: Initializes or trains your transformer.
        It will run :py:meth:`sklearn.base.BaseEstimator.fit`

     - :py:class:`VanillaBiometrics.create_biometric_reference`: Creates biometric references
        It will run :py:meth:`sklearn.base.BaseEstimator.transform` followed by a sequence of
        :py:meth:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.BioAlgorithm.enroll`

     - :py:class:`VanillaBiometrics.compute_scores`: Computes scores
        It will run :py:meth:`sklearn.base.BaseEstimator.transform` followed by a sequence of
        :py:meth:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.BioAlgorithm.score`


    Example
    -------
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.pipelines.vanilla_biometrics.implemented import Distance
       >>> transformer = make_pipeline(estimator_1, estimator_2)
       >>> biometric_algoritm = Distance()
       >>> pipeline = VanillaBiometrics(transformer, biometric_algoritm)
       >>> pipeline(samples_for_training_back_ground_model, samplesets_for_enroll, samplesets_for_scoring)


    To run this pipeline using Dask, used the function :py:func:`dask_vanilla_biometrics`.

    Example
    -------
      >>> pipeline = VanillaBiometrics(transformer, biometric_algoritm)
      >>> pipeline = dask_vanilla_biometrics(pipeline)
      >>> pipeline(samples_for_training_back_ground_model, samplesets_for_enroll, samplesets_for_scoring).compute()


    Parameters:
    -----------

      transformer: :py:class`sklearn.pipeline.Pipeline` or a `sklearn.base.BaseEstimator`
        Transformer that will preprocess your data

      biometric_algorithm: :py:class:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.BioAlgorithm`
        Biometrics algorithm object that implements the methods `enroll` and `score` methods


    """

    def __init__(self, transformer, biometric_algorithm):
        self.transformer = transformer
        self.biometric_algorithm = biometric_algorithm

    def __call__(
        self,
        background_model_samples,
        biometric_reference_samples,
        probe_samples,
        allow_scoring_with_all_biometric_references=False,
    ):
        logger.info(
            f" >> Vanilla Biometrics: Training background model with pipeline {self.transformer}"
        )

        # Training background model (fit will return even if samples is ``None``,
        # in which case we suppose the algorithm is not trainable in any way)
        self.transformer = self.train_background_model(background_model_samples)

        logger.info(
            f" >> Creating biometric references with the biometric algorithm {self.biometric_algorithm}"
        )

        # Create biometric samples
        biometric_references = self.create_biometric_reference(
            biometric_reference_samples
        )

        logger.info(
            f" >> Computing scores with the biometric algorithm {self.biometric_algorithm}"
        )

        # Scores all probes
        return self.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

    def train_background_model(self, background_model_samples):
        # background_model_samples is a list of Samples

        # We might have algorithms that has no data for training
        if len(background_model_samples) <= 0:
            logger.warning(
                "There's no data to train background model."
                "For the rest of the execution it will be assumed that the pipeline is stateless."
            )
            return self.transformer

        return self.transformer.fit(background_model_samples)

    def create_biometric_reference(self, biometric_reference_samples):
        biometric_reference_features = self.transformer.transform(
            biometric_reference_samples
        )

        biometric_references = self.biometric_algorithm.enroll_samples(
            biometric_reference_features
        )

        # models is a list of Samples
        return biometric_references

    def compute_scores(
        self,
        probe_samples,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        # probes is a list of SampleSets
        probe_features = self.transformer.transform(probe_samples)

        scores = self.biometric_algorithm.score_samples(
            probe_features,
            biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        # scores is a list of Samples
        return scores


def dask_vanilla_biometrics(pipeline, npartitions=None):
    """
    Given a :py:class:`VanillaBiometrics`, wraps :py:meth:`VanillaBiometrics.transformer` and
    :py:class:`VanillaBiometrics.biometric_algorithm` with Dask delayeds

    Parameters
    ----------

    pipeline: :py:class:`VanillaBiometrics`
       Vanilla Biometrics based pipeline to be dasked


    npartitions: int
       Number of partitions for the initial `Dask.bag`
    """

    from bob.pipelines.mixins import estimator_dask_it, mix_me_up
    from bob.bio.base.pipelines.vanilla_biometrics.mixins import BioAlgDaskMixin

    transformer = estimator_dask_it(pipeline.transformer, npartitions=npartitions)
    biometric_algorithm = mix_me_up([BioAlgDaskMixin], pipeline.biometric_algorithm)

    return VanillaBiometrics(transformer, biometric_algorithm)
