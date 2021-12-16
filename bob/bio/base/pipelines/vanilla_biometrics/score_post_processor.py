"""
#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

Implementation of a pipeline that post process scores


"""

from bob.pipelines import (
    DelayedSample,
    Sample,
    SampleSet,
    DelayedSampleSet,
    DelayedSampleSetCached,
)

from bob.pipelines.wrappers import CheckpointWrapper, DaskWrapper

import numpy as np
import dask
import functools
import cloudpickle
import os
from .score_writers import FourColumnsScoreWriter
import copy
import logging
from .pipelines import check_valid_pipeline
from . import pickle_compress, uncompress_unpickle
from sklearn.base import TransformerMixin, BaseEstimator

from bob.pipelines.utils import is_estimator_stateless

logger = logging.getLogger(__name__)


class ScoreNormalizationPipeline(object):
    """
    Apply Z, T or ZT Score normalization on top of VanillaBiometric Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Example
    -------
       >>> from bob.pipelines.transformers import Linearize
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.pipelines.vanilla_biometrics import Distance, VanillaBiometricsPipeline, ZTNormPipeline
       >>> estimator_1 = Linearize()
       >>> transformer = make_pipeline(estimator_1)
       >>> biometric_algorithm = Distance()
       >>> vanilla_biometrics_pipeline = VanillaBiometricsPipeline(transformer, biometric_algorithm)
       >>> zt_pipeline = ZTNormPipeline(vanilla_biometrics_pipeline)
       >>> zt_pipeline(...) #doctest: +SKIP

    Parameters
    ----------

        vanilla_biometrics_pipeline: :any:`VanillaBiometricsPipeline`
          An instance :any:`VanillaBiometricsPipeline` to the wrapped with score normalization

        post_processor: :py:class`sklearn.pipeline.Pipeline` or a `sklearn.base.BaseEstimator`
            Transformer that will post process the scores
        
        score_writer: 


    """

    def __init__(
        self,
        vanilla_biometrics_pipeline,
        post_processor,
        score_writer=FourColumnsScoreWriter("./scores.txt"),
    ):
        self.vanilla_biometrics_pipeline = vanilla_biometrics_pipeline
        self.biometric_algorithm = self.vanilla_biometrics_pipeline.biometric_algorithm
        self.transformer = self.vanilla_biometrics_pipeline.transformer

        self.post_processor = post_processor
        self.score_writer = score_writer

        # TODO: ACTIVATE THAT
        # check_valid_pipeline(self)

    def __call__(
        self,
        background_model_samples,
        biometric_reference_samples,
        probe_samples,
        post_process_samples,
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

        # Training the score transformer
        self.post_processor.fit([post_process_samples, biometric_references])

        # Transformer
        post_processed_scores = self.post_processor.transform(raw_scores)

        return raw_scores, post_processed_scores

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

    def write_scores(self, scores):
        return self.vanilla_biometrics_pipeline.write_scores(scores)

    def post_process(self, score_paths, filename):
        return self.vanilla_biometrics_pipeline.post_process(score_paths, filename)


def copy_learned_attributes(from_estimator, to_estimator):
    attrs = {k: v for k, v in vars(from_estimator).items() if k.endswith("_")}

    for k, v in attrs.items():
        setattr(to_estimator, k, v)


class CheckpointPostProcessor(CheckpointWrapper):
    """
    This class creates pickle checkpoints of post-processed scores.
    
    
    .. Note::
       We can't use the `CheckpointWrapper` from bob.pipelines to create these checkpoints.
       Because there, each sample is checkpointed, and here we can have checkpoints for SampleSets

    Parameters
    ----------

    estimator
       The scikit-learn estimator to be wrapped.

    model_path: str
       Saves the estimator state in this directory if the `estimator` is stateful

    features_dir: str
       Saves the transformed data in this directory

    hash_fn
       Pointer to a hash function. This hash function maps
       `sample.key` to a hash code and this hash code corresponds a relative directory
       where a single `sample` will be checkpointed.
       This is useful when is desirable file directories with less than
       a certain number of files.

    attempts
       Number of checkpoint attempts. Sometimes, because of network/disk issues
       files can't be saved. This argument sets the maximum number of attempts
       to checkpoint a sample.

    """

    def __init__(
        self,
        estimator,
        model_path=None,
        features_dir=None,
        extension=".pkl",
        hash_fn=None,
        attempts=10,
        **kwargs,
    ):

        self.estimator = estimator
        self.model_path = model_path
        self.features_dir = features_dir
        self.extension = extension

        self.hash_fn = hash_fn
        self.attempts = attempts
        if model_path is None and features_dir is None:
            logger.warning(
                "Both model_path and features_dir are None. "
                f"Nothing will be checkpointed. From: {self}"
            )

    def fit(self, samples, y=None):

        if is_estimator_stateless(self.estimator):
            return self

        # if the estimator needs to be fitted.
        logger.debug(f"CheckpointPostProcessor.fit")

        if self.model_path is not None and os.path.isfile(self.model_path):
            logger.info("Found a checkpoint for model. Loading ...")
            return self.load_model()

        self.estimator = self.estimator.fit(samples, y=y)
        copy_learned_attributes(self.estimator, self)
        return self.save_model()

    def transform(self, sample_sets, y=None):

        logger.debug("CheckpointPostProcessor.transform")
        transformed_sample_sets = []
        for s in sample_sets:

            path = self.make_path(s)
            if os.path.exists(path):
                sset = uncompress_unpickle(path)
            else:
                sset = self.estimator.transform([s])[0]
                pickle_compress(path, sset)

            transformed_sample_sets.append(sset)

        return transformed_sample_sets


def checkpoint_score_normalization_pipeline(pipeline, base_dir, hash_fn=None):

    model_path = os.path.join(base_dir, "stats.pkl")
    features_dir = os.path.join(base_dir, "normed_scores")

    # Checkpointing only the post processor
    pipeline.post_processor = CheckpointPostProcessor(
        pipeline.post_processor,
        model_path=model_path,
        features_dir=features_dir,
        hash_fn=hash_fn,
        extension=".pkl",
    )

    return pipeline


def dask_score_normalization_pipeline(pipeline):

    # Checkpointing only the post processor
    pipeline.post_processor = DaskWrapper(pipeline.post_processor,)

    return pipeline


class ZNormScores(TransformerMixin, BaseEstimator):
    """
    Apply Z-Norm Score normalization on top of VanillaBiometric Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Parameters
    ----------

    """

    def __init__(
        self,
        transformer,
        scoring_function,
        top_norm=False,
        top_norm_score_fraction=0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transformer = transformer
        self.scoring_function = scoring_function
        self.top_norm_score_fraction = top_norm_score_fraction
        self.top_norm = top_norm

    def fit(self, X, y=None):

        # JUst for the sake of readability
        zprobe_samples = X[0]
        biometric_references = X[1]

        # Compute the ZScores

        # Computing the features
        zprobe_features = self.transformer.transform(zprobe_samples)

        z_scores, _ = self.scoring_function(
            zprobe_features,
            biometric_references,
            allow_scoring_with_all_biometric_references=False,
        )

        # TODO: THIS IS SUPER INNEFICIENT, BUT
        # IT'S THE MOST READABLE SOLUTION

        # Stacking scores by biometric reference
        self._z_stats = dict()
        for sset in z_scores:
            for s in sset:
                if not s.reference_id in self._z_stats:
                    self._z_stats[s.reference_id] = Sample([], parent=s)

                self._z_stats[s.reference_id].data.append(s.data)

        # Now computing the statistics in place

        for key in self._z_stats:
            data = self._z_stats[key].data

            ## Selecting the top scores
            if self.top_norm:
                # Sorting in ascending order
                data = -np.sort(-data)
                proportion = int(np.floor(len(data) * self.top_norm_score_fraction))
                data = data[0:proportion]

            self._z_stats[key].mu = np.mean(self._z_stats[key].data)
            self._z_stats[key].std = np.std(self._z_stats[key].data)
            # self._z_stats[key].std = legacy_std(
            #    self._z_stats[key].mu, self._z_stats[key].data
            # )
            self._z_stats[key].data = []

        return self

    def transform(self, X):

        if len(X) <= 0:
            # Nothing to be transformed
            return []

        def _transform_samples(X):
            scores = []
            for no_normed_score in X:
                score = (
                    no_normed_score.data
                    - self._z_stats[no_normed_score.reference_id].mu
                ) / self._z_stats[no_normed_score.reference_id].std

                z_score = Sample(score, parent=no_normed_score)
                scores.append(z_score)
            return scores

        if isinstance(X[0], SampleSet):

            z_normed_scores = []
            # Transforming either Samples or SampleSets
            for probe_scores in X:

                z_normed_scores.append(
                    SampleSet(_transform_samples(probe_scores), parent=probe_scores)
                )
        else:
            # If it is Samples
            z_normed_scores = _transform_samples(X)

        return z_normed_scores

