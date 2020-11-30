#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Implementation of the Vanilla Biometrics pipeline using Dask :ref:`bob.bio.base.struct_bio_rec_sys`_

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import logging
import numpy
from .score_writers import FourColumnsScoreWriter
from bob.pipelines.utils import isinstance_nested
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from bob.pipelines import SampleWrapper, wrap
from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import BioAlgorithm

logger = logging.getLogger(__name__)
import tempfile
import os


class VanillaBiometricsPipeline(object):
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
       >>> from bob.pipelines.transformers import Linearize
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.pipelines.vanilla_biometrics import Distance, VanillaBiometricsPipeline
       >>> estimator_1 = Linearize()
       >>> transformer = make_pipeline(estimator_1)
       >>> biometric_algoritm = Distance()
       >>> pipeline = VanillaBiometricsPipeline(transformer, biometric_algoritm)
       >>> pipeline(samples_for_training_back_ground_model, samplesets_for_enroll, samplesets_for_scoring)  # doctest: +SKIP


    To run this pipeline using Dask, used the function :py:func:`dask_vanilla_biometrics`.

    Example
    -------
      >>> from bob.bio.base.pipelines.vanilla_biometrics import dask_vanilla_biometrics
      >>> pipeline = VanillaBiometricsPipeline(transformer, biometric_algoritm)
      >>> pipeline = dask_vanilla_biometrics(pipeline)
      >>> pipeline(samples_for_training_back_ground_model, samplesets_for_enroll, samplesets_for_scoring).compute()  # doctest: +SKIP


    Parameters
    ----------

      transformer: :py:class`sklearn.pipeline.Pipeline` or a `sklearn.base.BaseEstimator`
        Transformer that will preprocess your data

      biometric_algorithm: :py:class:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.BioAlgorithm`
        Biometrics algorithm object that implements the methods `enroll` and `score` methods

      score_writer: :any:`bob.bio.base.pipelines.vanilla_biometrics.ScoreWriter`
          Format to write scores. Default to :any:`bob.bio.base.pipelines.vanilla_biometrics.FourColumnsScoreWriter`

    """

    def __init__(
        self, transformer, biometric_algorithm, score_writer=None,
    ):
        self.transformer = transformer
        self.biometric_algorithm = biometric_algorithm
        self.score_writer = score_writer
        if self.score_writer is None:
            tempdir = tempfile.TemporaryDirectory()
            self.score_writer = FourColumnsScoreWriter(tempdir.name)

        check_valid_pipeline(self)

    def __call__(
        self,
        background_model_samples,
        biometric_reference_samples,
        probe_samples,
        allow_scoring_with_all_biometric_references=True,
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
        scores, _ = self.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

        return scores

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
        allow_scoring_with_all_biometric_references=True,
    ):

        # probes is a list of SampleSets
        probe_features = self.transformer.transform(probe_samples)
        scores = self.biometric_algorithm.score_samples(
            probe_features,
            biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        # scores is a list of Samples
        return scores, probe_features

    def write_scores(self, scores):
        if self.score_writer is None:
            raise ValueError("No score writer defined in the pipeline")
        return self.score_writer.write(scores)

    def post_process(self, score_paths, filename):
        if self.score_writer is None:
            raise ValueError("No score writer defined in the pipeline")

        return self.score_writer.post_process(score_paths, filename)


def check_valid_pipeline(vanilla_pipeline):
    """
    Applying some checks in the vanilla biometrics pipeline
    """

    ## CHECKING THE TRANSFORMER
    # Checking if it's a Scikit Pipeline or a estimator
    if isinstance(vanilla_pipeline.transformer, Pipeline):

        # Checking if all steps are wrapped as samples, if not, we should wrap them
        for p in vanilla_pipeline.transformer:
            if not isinstance_nested(p, "estimator", SampleWrapper):
                wrap(["sample"], p)

    # In this case it can be a simple estimator. AND
    # Checking if it's sample wrapper, if not, do it
    elif isinstance_nested(
        vanilla_pipeline.transformer, "estimator", BaseEstimator
    ) and isinstance_nested(vanilla_pipeline.transformer, "estimator", BaseEstimator):
        wrap(["sample"], vanilla_pipeline.transformer)
    else:
        raise ValueError(
            f"VanillaBiometricsPipeline.transformer should be instance of either `sklearn.pipeline.Pipeline` or"
            f"sklearn.base.BaseEstimator, not {vanilla_pipeline.transformer}"
        )

    ## Checking the Biometric algorithm
    if not isinstance(vanilla_pipeline.biometric_algorithm, BioAlgorithm):
        raise ValueError(
            f"VanillaBiometricsPipeline.biometric_algorithm should be instance of `BioAlgorithm`"
            f"not {vanilla_pipeline.biometric_algorithm}"
        )

    return True
