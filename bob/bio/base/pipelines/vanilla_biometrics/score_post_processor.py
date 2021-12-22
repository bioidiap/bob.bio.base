"""
#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

Implementation of a pipeline that post process scores


"""

from bob.pipelines import (
    Sample,
    SampleSet,
)
import bob.bio.base

from bob.pipelines.wrappers import CheckpointWrapper, DaskWrapper
from bob.pipelines.utils import isinstance_nested
import numpy as np
import os
from .score_writers import FourColumnsScoreWriter
import copy
import logging
from .pipelines import check_valid_pipeline, VanillaBiometricsPipeline
from . import pickle_compress, uncompress_unpickle
from sklearn.base import TransformerMixin, BaseEstimator
import tempfile
import copy

from bob.pipelines.utils import is_estimator_stateless

logger = logging.getLogger(__name__)


class ScoreNormalizationPipeline(VanillaBiometricsPipeline):
    """
    Apply Z, T or ZT Score normalization on top of VanillaBiometric Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Example
    -------
       >>> from bob.pipelines.transformers import Linearize
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.pipelines.vanilla_biometrics import Distance, VanillaBiometricsPipeline, ScoreNormalizationPipeline, ZNormScores
       >>> estimator_1 = Linearize()
       >>> transformer = make_pipeline(estimator_1)
       >>> biometric_algorithm = Distance()
       >>> vanilla_biometrics_pipeline = VanillaBiometricsPipeline(transformer, biometric_algorithm)
       >>> z_norm_postprocessor = ZNormScores(pipeline=vanilla_biometrics_pipeline)
       >>> z_pipeline = ScoreNormalizationPipeline(vanilla_biometrics_pipeline, z_norm_postprocessor)       
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
        self, vanilla_biometrics_pipeline, post_processor, score_writer=None,
    ):

        self.vanilla_biometrics_pipeline = vanilla_biometrics_pipeline
        self.biometric_algorithm = self.vanilla_biometrics_pipeline.biometric_algorithm
        self.transformer = self.vanilla_biometrics_pipeline.transformer

        self.post_processor = post_processor
        self.score_writer = score_writer

        if self.score_writer is None:
            tempdir = tempfile.TemporaryDirectory()
            self.score_writer = FourColumnsScoreWriter(tempdir.name)

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
        if isinstance_nested(
            self.post_processor, "estimator", ZNormScores
        ) or isinstance(self.post_processor, ZNormScores):
            self.post_processor.fit([post_process_samples, biometric_references])
            # Transformer
            post_processed_scores = self.post_processor.transform(raw_scores)

        elif isinstance_nested(
            self.post_processor, "estimator", TNormScores
        ) or isinstance(self.post_processor, TNormScores):
            # self.post_processor.fit([post_process_samples, probe_features])
            self.post_processor.fit([post_process_samples, probe_samples])
            # Transformer
            post_processed_scores = self.post_processor.transform(raw_scores)
        else:
            logger.warning(
                f"Invalid post-processor {self.post_processor}. Returning the raw_scores"
            )
            post_processed_scores = raw_scores

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
        force=True,
        **kwargs,
    ):

        self.estimator = estimator
        self.model_path = model_path
        self.features_dir = features_dir
        self.extension = extension

        self.hash_fn = hash_fn
        self.attempts = attempts
        self.force = force
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

        if not self.force and (
            self.model_path is not None and os.path.isfile(self.model_path)
        ):
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


def checkpoint_score_normalization_pipeline(
    pipeline, base_dir, sub_dir="norm", hash_fn=None
):

    model_path = os.path.join(base_dir, sub_dir, "stats.pkl")
    features_dir = os.path.join(base_dir, sub_dir, "normed_scores")

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
        self, pipeline, top_norm=False, top_norm_score_fraction=0.8, **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_norm_score_fraction = top_norm_score_fraction
        self.top_norm = top_norm

        # Copying the pipeline and possibly chaning the biometric_algoritm paths
        self.pipeline = copy.deepcopy(pipeline)

        # TODO: I know this is ugly, but I don't want to create on pipeline for every single
        # normalization strategy
        if isinstance_nested(
            self.pipeline,
            "biometric_algorithm",
            bob.bio.base.pipelines.vanilla_biometrics.wrappers.BioAlgorithmCheckpointWrapper,
        ):

            if isinstance_nested(
                self.pipeline,
                "biometric_algorithm",
                bob.bio.base.pipelines.vanilla_biometrics.wrappers.BioAlgorithmDaskWrapper,
            ):
                self.pipeline.biometric_algorithm.biometric_algorithm.score_dir = os.path.join(
                    self.pipeline.biometric_algorithm.biometric_algorithm.score_dir,
                    "score-norm",
                )
                self.pipeline.biometric_algorithm.biometric_algorithm.biometric_reference_dir = os.path.join(
                    self.pipeline.biometric_algorithm.biometric_algorithm.biometric_reference_dir,
                    "score-norm",
                )

            else:
                self.pipeline.biometric_algorithm.score_dir = os.path.join(
                    self.pipeline.biometric_algorithm.score_dir, "score-norm"
                )
                self.pipeline.biometric_algorithm.biometric_reference_dir = os.path.join(
                    self.pipeline.biometric_algorithm.biometric_reference_dir,
                    "score-norm",
                )

    def fit(self, X, y=None):

        # JUst for the sake of readability
        zprobe_samples = X[0]
        biometric_references = X[1]

        # Compute the ZScores

        # Computing the features
        zprobe_features = self.pipeline.transformer.transform(zprobe_samples)

        z_scores, _ = self.pipeline.compute_scores(
            zprobe_features,
            biometric_references,
            allow_scoring_with_all_biometric_references=False,
        )

        # TODO: THIS IS SUPER INNEFICIENT, BUT
        # IT'S THE MOST READABLE SOLUTION

        # Stacking scores by biometric reference
        self.z_stats = dict()
        for sset in z_scores:
            for s in sset:
                if not s.reference_id in self.z_stats:
                    self.z_stats[s.reference_id] = Sample([], parent=s)

                self.z_stats[s.reference_id].data.append(s.data)

        # Now computing the statistics in place

        for key in self.z_stats:
            data = self.z_stats[key].data

            ## Selecting the top scores
            if self.top_norm:
                # Sorting in ascending order
                data = -np.sort(-data)
                proportion = int(np.floor(len(data) * self.top_norm_score_fraction))
                data = data[0:proportion]

            self.z_stats[key].mu = np.mean(self.z_stats[key].data)
            self.z_stats[key].std = np.std(self.z_stats[key].data)
            # self._z_stats[key].std = legacy_std(
            #    self._z_stats[key].mu, self._z_stats[key].data
            # )
            self.z_stats[key].data = []

        return self

    def transform(self, X):

        if len(X) <= 0:
            # Nothing to be transformed
            return []

        def _transform_samples(X):
            scores = []
            for no_normed_score in X:
                score = (
                    no_normed_score.data - self.z_stats[no_normed_score.reference_id].mu
                ) / self.z_stats[no_normed_score.reference_id].std

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


class TNormScores(TransformerMixin, BaseEstimator):
    """
    Apply T-Norm Score normalization on top of VanillaBiometric Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Parameters
    ----------

    """

    def __init__(
        self, pipeline, top_norm=False, top_norm_score_fraction=0.8, **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_norm_score_fraction = top_norm_score_fraction
        self.top_norm = top_norm

        # Copying the pipeline and possibly chaning the biometric_algoritm paths
        self.pipeline = copy.deepcopy(pipeline)

        # TODO: I know this is ugly, but I don't want to create on pipeline for every single
        # normalization strategy
        if isinstance_nested(
            self.pipeline,
            "biometric_algorithm",
            bob.bio.base.pipelines.vanilla_biometrics.wrappers.BioAlgorithmCheckpointWrapper,
        ):

            if isinstance_nested(
                self.pipeline,
                "biometric_algorithm",
                bob.bio.base.pipelines.vanilla_biometrics.wrappers.BioAlgorithmDaskWrapper,
            ):
                self.pipeline.biometric_algorithm.biometric_algorithm.score_dir = os.path.join(
                    self.pipeline.biometric_algorithm.biometric_algorithm.score_dir,
                    "score-norm",
                )
                self.pipeline.biometric_algorithm.biometric_algorithm.biometric_reference_dir = os.path.join(
                    self.pipeline.biometric_algorithm.biometric_algorithm.biometric_reference_dir,
                    "score-norm",
                )

            else:
                self.pipeline.biometric_algorithm.score_dir = os.path.join(
                    self.pipeline.biometric_algorithm.score_dir, "score-norm"
                )
                self.pipeline.biometric_algorithm.biometric_reference_dir = os.path.join(
                    self.pipeline.biometric_algorithm.biometric_reference_dir,
                    "score-norm",
                )

    def fit(self, X, y=None):

        # JUst for the sake of readability
        treference_samples = X[0]

        # TODO: We need to pass probe samples instead of probe features
        probe_samples = X[1]  ## Probes to be normalized

        probe_features = self.pipeline.transformer.transform(probe_samples)

        # Creating T-Models
        treferences = self.pipeline.create_biometric_reference(treference_samples)

        # t_references_ids = [s.reference_id for s in treferences]

        # probes[0].reference_id

        # Scoring the T-Models
        t_scores = self.pipeline.biometric_algorithm.score_samples(
            probe_features,
            treferences,
            allow_scoring_with_all_biometric_references=True,
        )

        # t_scores, _ = self.pipeline.compute_scores(
        #    probes, treferences, allow_scoring_with_all_biometric_references=True,
        # )

        # TODO: THIS IS SUPER INNEFICIENT, BUT
        # IT'S THE MOST READABLE SOLUTION
        # Stacking scores by biometric reference
        self.t_stats = dict()

        for sset in t_scores:

            self.t_stats[sset.reference_id] = Sample(
                [s.data for s in sset], parent=sset
            )

        # Now computing the statistics in place
        for key in self.t_stats:
            data = self.t_stats[key].data

            ## Selecting the top scores
            if self.top_norm:
                # Sorting in ascending order
                data = -np.sort(-data)
                proportion = int(np.floor(len(data) * self.top_norm_score_fraction))
                data = data[0:proportion]

            self.t_stats[key].mu = np.mean(self.t_stats[key].data)
            self.t_stats[key].std = np.std(self.t_stats[key].data)
            # self._z_stats[key].std = legacy_std(
            #    self._z_stats[key].mu, self._z_stats[key].data
            # )
            self.t_stats[key].data = []

        return self

    def transform(self, X):

        if len(X) <= 0:
            # Nothing to be transformed
            return []

        def _transform_samples(X, stats):
            scores = []
            for no_normed_score in X:
                score = (no_normed_score.data - stats.mu) / stats.std

                t_score = Sample(score, parent=no_normed_score)
                scores.append(t_score)
            return scores

        if isinstance(X[0], SampleSet):

            t_normed_scores = []
            # Transforming either Samples or SampleSets

            for probe_scores in X:

                stats = self.t_stats[probe_scores.reference_id]

                t_normed_scores.append(
                    SampleSet(
                        _transform_samples(probe_scores, stats), parent=probe_scores
                    )
                )
        else:
            # If it is Samples
            t_normed_scores = _transform_samples(X)

        return t_normed_scores
