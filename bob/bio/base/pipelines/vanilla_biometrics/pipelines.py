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
from .wrappers import BioAlgorithmZTNormWrapper


logger = logging.getLogger(__name__)


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

      score_writer: :any:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classe.ScoreWriter`
          Format to write scores. Default to :any:`FourColumnsScoreWriter`

    """

    def __init__(
        self,
        transformer,
        biometric_algorithm,
        score_writer=FourColumnsScoreWriter("./scores.txt"),
    ):
        self.transformer = transformer
        self.biometric_algorithm = biometric_algorithm
        self.score_writer = score_writer

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
        scores, _ = self.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )


        if self.score_writer is not None:
            return self.write_scores(scores)

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
        return scores, probe_features

    def write_scores(self, scores):
        return self.score_writer.write(scores)


class ZTNormVanillaBiometricsPipeline(object):
    """
    Apply Z, T or ZT Score normalization on top of VanillaBiometric Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Example
    -------
       >>> transformer = make_pipeline([])
       >>> biometric_algorithm = Distance()
       >>> vanilla_biometrics_pipeline = VanillaBiometricsPipeline(transformer, biometric_algorithm)
       >>> zt_pipeline = ZTNormVanillaBiometricsPipeline(vanilla_biometrics_pipeline)
       >>> zt_pipeline(...)

    Parameters
    ----------

        vanilla_biometrics_pipeline: :any:`VanillaBiometricsPipeline`
          An instance :any:`VanillaBiometricsPipeline` to the wrapped with score normalization

        z_norm: bool
          If True, applies ZScore normalization on top of raw scores.

        t_norm: bool
          If True, applies TScore normalization on top of raw scores.
          If both, z_norm and t_norm are true, it applies score normalization

    """


    def __init__(self, vanilla_biometrics_pipeline, z_norm=True, t_norm=True):
        self.vanilla_biometrics_pipeline = vanilla_biometrics_pipeline
        # Wrapping with ZTNorm
        self.vanilla_biometrics_pipeline.biometric_algorithm = BioAlgorithmZTNormWrapper(
            self.vanilla_biometrics_pipeline.biometric_algorithm
        )
        self.z_norm = z_norm
        self.t_norm = t_norm

        if not z_norm and not t_norm:
            raise ValueError("Both z_norm and t_norm are False. No normalization will be applied")

    def __call__(
        self,
        background_model_samples,
        biometric_reference_samples,
        probe_samples,
        zprobe_samples=None,
        t_biometric_reference_samples=None,
        allow_scoring_with_all_biometric_references=False,
    ):

        self.transformer = self.train_background_model(background_model_samples)

        # Create biometric samples
        biometric_references = self.create_biometric_reference(
            biometric_reference_samples
        )

        raw_scores, probe_features = self.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

        # Z NORM
        if self.z_norm:
            if zprobe_samples is None:
                raise ValueError("No samples for `z_norm` was provided")


            z_normed_scores, z_probe_features = self.compute_znorm_scores(
                zprobe_samples,
                raw_scores,
                biometric_references,
                allow_scoring_with_all_biometric_references,
            )
        if self.t_norm:
            if t_biometric_reference_samples is None:
                raise ValueError("No samples for `t_norm` was provided")
        else:
            # In case z_norm=True and t_norm=False
            return z_normed_scores

        # T NORM
        t_normed_scores, t_scores, t_biometric_references = self.compute_tnorm_scores(
            t_biometric_reference_samples,
            probe_features,
            raw_scores,
            allow_scoring_with_all_biometric_references,
        )
        if not self.z_norm:
            # In case z_norm=False and t_norm=True
            return t_normed_scores


        # ZT NORM
        zt_normed_scores = self.compute_ztnorm_scores(
            z_probe_features,
            t_biometric_references,
            z_normed_scores,
            t_scores,
            allow_scoring_with_all_biometric_references,
        )

        return zt_normed_scores

    def train_background_model(self, background_model_samples):
        return self.vanilla_biometrics_pipeline.train_background_model(
            background_model_samples
        )

    def create_biometric_reference(self, biometric_reference_samples):
        return self.vanilla_biometrics_pipeline.create_biometric_reference(
            biometric_reference_samples
        )

    def compute_scores(
        self,
        probe_samples,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        return self.vanilla_biometrics_pipeline.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

    def _inject_references(self, probe_samples, biometric_references):
        """
        Inject references in the current sampleset,
        so it can run the scores
        """

        ########## WARNING #######
        #### I'M MUTATING OBJECTS HERE. THIS CAN GO WRONG

        references = [s.subject  for s in biometric_references]
        for probe in probe_samples:
            probe.references = references
        return probe_samples


    def compute_znorm_scores(
        self,
        zprobe_samples,
        probe_scores,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        zprobe_samples = self._inject_references(zprobe_samples, biometric_references)

        z_scores, z_probe_features = self.compute_scores(
            zprobe_samples, biometric_references
        )

        z_normed_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.compute_znorm_scores(
            z_scores, probe_scores, allow_scoring_with_all_biometric_references,
        )

        return z_normed_scores, z_probe_features

    def compute_tnorm_scores(
        self,
        t_biometric_reference_samples,
        probe_features,
        probe_scores,
        allow_scoring_with_all_biometric_references=False,
    ):

        t_biometric_references = self.create_biometric_reference(
            t_biometric_reference_samples
        )

        probe_features = self._inject_references(probe_features, t_biometric_references)

        # Reusing the probe features
        t_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.score_samples(
            probe_features,
            t_biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        t_normed_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.compute_tnorm_scores(
            t_scores, probe_scores, allow_scoring_with_all_biometric_references,
        )

        return t_normed_scores, t_scores, t_biometric_references

    def compute_ztnorm_scores(self,
            z_probe_features,
            t_biometric_references,
            z_normed_scores,
            t_scores,
            allow_scoring_with_all_biometric_references=False
            ):

        z_probe_features = self._inject_references(z_probe_features, t_biometric_references)

        # Reusing the zprobe_features and t_biometric_references
        zt_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.score_samples(
            z_probe_features,
            t_biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        # Z Normalizing the T-normed scores
        z_normed_t_normed = self.vanilla_biometrics_pipeline.biometric_algorithm.compute_znorm_scores(
            zt_scores, t_scores, allow_scoring_with_all_biometric_references,
        )

        # (Z Normalizing the T-normed scores) the Z normed scores
        zt_normed_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.compute_tnorm_scores(
            z_normed_t_normed, z_normed_scores, allow_scoring_with_all_biometric_references,
        )


        return zt_normed_scores
