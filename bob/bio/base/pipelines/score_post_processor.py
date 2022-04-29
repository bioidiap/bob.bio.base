"""
#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

Implementation of a pipeline that post process scores


"""

import copy
import logging
import os
import tempfile

from functools import partial

import dask.dataframe
import numpy as np

from scipy.optimize import curve_fit
from scipy.special import expit
from scipy.stats import beta, gamma, weibull_min
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

import bob.bio.base

from bob.bio.base.score.load import get_split_dataframe
from bob.pipelines import Sample, SampleSet
from bob.pipelines.utils import is_estimator_stateless, isinstance_nested
from bob.pipelines.wrappers import CheckpointWrapper, DaskWrapper

from . import pickle_compress, uncompress_unpickle
from .pipelines import PipelineSimple
from .score_writers import FourColumnsScoreWriter

logger = logging.getLogger(__name__)


class PipelineScoreNorm(PipelineSimple):
    """
    Apply Z, T or ZT Score normalization on top of Pimple Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Example
    -------
       >>> from bob.pipelines.transformers import Linearize
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.pipelines import Distance, PipelineSimple, PipelineScoreNorm, ZNormScores
       >>> estimator_1 = Linearize()
       >>> transformer = make_pipeline(estimator_1)
       >>> biometric_algorithm = Distance()
       >>> pipeline_simple = PipelineSimple(transformer, biometric_algorithm)
       >>> z_norm_postprocessor = ZNormScores(pipeline=pipeline_simple)
       >>> z_pipeline = PipelineScoreNorm(pipeline_simple, z_norm_postprocessor)
       >>> zt_pipeline(...) #doctest: +SKIP

    Parameters
    ----------

        pipeline_simple: :any:`PipelineSimple`
          An instance :any:`PipelineSimple` to the wrapped with score normalization

        post_processor: :py:class`sklearn.pipeline.Pipeline` or a `sklearn.base.BaseEstimator`
            Transformer that will post process the scores

        score_writer:


    """

    def __init__(
        self,
        pipeline_simple,
        post_processor,
        score_writer=None,
    ):

        self.pipeline_simple = pipeline_simple
        self.biometric_algorithm = self.pipeline_simple.biometric_algorithm
        self.transformer = self.pipeline_simple.transformer

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
            self.post_processor.fit(
                [post_process_samples, biometric_references]
            )
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
        return self.pipeline_simple.train_background_model(
            background_model_samples
        )

    def create_biometric_reference(self, biometric_reference_samples):
        return self.pipeline_simple.create_biometric_reference(
            biometric_reference_samples
        )

    def compute_scores(
        self,
        probe_samples,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        return self.pipeline_simple.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

    def write_scores(self, scores):
        return self.pipeline_simple.write_scores(scores)

    def post_process(self, score_paths, filename):
        return self.pipeline_simple.post_process(score_paths, filename)


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
        logger.debug("CheckpointPostProcessor.fit")

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
    pipeline.post_processor = DaskWrapper(
        pipeline.post_processor,
    )

    return pipeline


class ZNormScores(TransformerMixin, BaseEstimator):
    """
    Apply Z-Norm Score normalization on top of Simple Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Parameters
    ----------

    """

    def __init__(
        self,
        pipeline,
        top_norm=False,
        top_norm_score_fraction=0.8,
        **kwargs,
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
            bob.bio.base.pipelines.wrappers.BioAlgorithmCheckpointWrapper,
        ):

            if isinstance_nested(
                self.pipeline,
                "biometric_algorithm",
                bob.bio.base.pipelines.wrappers.BioAlgorithmDaskWrapper,
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
                if s.reference_id not in self.z_stats:
                    self.z_stats[s.reference_id] = Sample([], parent=s)

                self.z_stats[s.reference_id].data.append(s.data)

        # Now computing the statistics in place

        for key in self.z_stats:
            data = self.z_stats[key].data

            # Selecting the top scores
            if self.top_norm:
                # Sorting in ascending order
                data = -np.sort(-data)
                proportion = int(
                    np.floor(len(data) * self.top_norm_score_fraction)
                )
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
                    no_normed_score.data
                    - self.z_stats[no_normed_score.reference_id].mu
                ) / self.z_stats[no_normed_score.reference_id].std

                z_score = Sample(score, parent=no_normed_score)
                scores.append(z_score)
            return scores

        if isinstance(X[0], SampleSet):

            z_normed_scores = []
            # Transforming either Samples or SampleSets
            for probe_scores in X:

                z_normed_scores.append(
                    SampleSet(
                        _transform_samples(probe_scores), parent=probe_scores
                    )
                )
        else:
            # If it is Samples
            z_normed_scores = _transform_samples(X)

        return z_normed_scores


class TNormScores(TransformerMixin, BaseEstimator):
    """
    Apply T-Norm Score normalization on top of Simple Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Parameters
    ----------

    """

    def __init__(
        self,
        pipeline,
        top_norm=False,
        top_norm_score_fraction=0.8,
        **kwargs,
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
            bob.bio.base.pipelines.wrappers.BioAlgorithmCheckpointWrapper,
        ):

            if isinstance_nested(
                self.pipeline,
                "biometric_algorithm",
                bob.bio.base.pipelines.wrappers.BioAlgorithmDaskWrapper,
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
        probe_samples = X[1]  # Probes to be normalized

        probe_features = self.pipeline.transformer.transform(probe_samples)

        # Creating T-Models
        treferences = self.pipeline.create_biometric_reference(
            treference_samples
        )

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

            # Selecting the top scores
            if self.top_norm:
                # Sorting in ascending order
                data = -np.sort(-data)
                proportion = int(
                    np.floor(len(data) * self.top_norm_score_fraction)
                )
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
                        _transform_samples(probe_scores, stats),
                        parent=probe_scores,
                    )
                )
        else:
            # If it is Samples
            t_normed_scores = _transform_samples(X)

        return t_normed_scores


class LLRCalibration(TransformerMixin, BaseEstimator):
    """
    Implements the linear calibration using a logistic function
    defined in:

    `Mandasari, Miranti Indar, et al. "Score calibration in face recognition." Iet Biometrics 3.4 (2014): 246-256.`

    """

    def fit(self, X, y):
        self.regressor = LogisticRegression(
            class_weight="balanced", fit_intercept=True, penalty="l2"
        )
        self.regressor.fit(X, y)
        return self

    def predict_proba(self, X):
        def get_sigmoid_score(X, regressor):
            return expit(X * regressor.coef_ + regressor.intercept_).ravel()

        return get_sigmoid_score(X, self.regressor)


class WeibullCalibration(TransformerMixin, BaseEstimator):
    """
    Implements the weibull calibration using a pair of Weibull
    pdf's defined in:

    `Macarulla Rodriguez, Andrea, Zeno Geradts, and Marcel Worring. "Likelihood Ratios for Deep Neural Networks in Face Comparison." Journal of forensic sciences 65.4 (2020): 1169-1183.`

    """

    def fit(self, X, y):
        def weibull_pdf_genuines(x, c, scale):
            return weibull_min.pdf(np.abs(x), c, scale=scale)

        def weibull_pdf_impostors(x, c, scale):
            return weibull_min.pdf(np.abs(x), c, scale=scale)

        bins = 20

        impostors_X, impostors_Y = np.histogram(
            X[y == 0], bins=bins, density=True
        )
        # averaging the bins
        impostors_Y = 0.5 * (impostors_Y[1:] + impostors_Y[:-1])

        # Binining genuies and impostors for the fit
        genuines_X, genuines_Y = np.histogram(
            X[y == 1], bins=bins, density=True
        )
        # averaging the bins
        genuines_Y = 0.5 * (genuines_Y[1:] + genuines_Y[:-1])

        # Weibull fit for impostor distribution
        impostors_popt, _ = curve_fit(
            weibull_pdf_impostors,
            xdata=impostors_X,
            ydata=impostors_Y,
            p0=[1.0, 1.0],
        )
        impostors_c, impostors_scale = impostors_popt

        # Fitting weibull for genuines and impostors
        genuines_popt, _ = curve_fit(
            weibull_pdf_genuines,
            xdata=genuines_X,
            ydata=genuines_Y,
            p0=[1.0, 1.0],
        )
        genuines_c, genuines_scale = genuines_popt

        self.gen_dist = partial(
            weibull_pdf_genuines, c=genuines_c, scale=genuines_scale
        )
        self.imp_dist = partial(
            weibull_pdf_impostors, c=impostors_c, scale=impostors_scale
        )
        return self

    def predict_proba(self, X):
        epsilon = 1e-10
        return +1 * (
            np.log10(self.gen_dist(X) + epsilon)
            - np.log10(self.imp_dist(X) + epsilon)
        )


class GammaCalibration(TransformerMixin, BaseEstimator):
    """
    Implements the weibull calibration using a pair of  Gamma
    pdf's defined in:

    """

    def fit(self, X, y):
        def gamma_pdf_genuines(x, a, scale):
            return gamma.pdf(np.abs(x), a, scale=scale)

        def gamma_pdf_impostors(x, a, scale):
            return gamma.pdf(np.abs(x), a, scale=scale)

        bins = 100

        impostors_X, impostors_Y = np.histogram(
            X[y == 0], bins=bins, density=True
        )
        # averaging the bins
        impostors_Y = 0.5 * (impostors_Y[1:] + impostors_Y[:-1])

        # Binining genuies and impostors for the fit
        genuines_X, genuines_Y = np.histogram(
            X[y == 1], bins=bins, density=True
        )
        # averaging the bins
        genuines_Y = 0.5 * (genuines_Y[1:] + genuines_Y[:-1])

        # gamma fit for impostor distribution
        impostors_popt, _ = curve_fit(
            gamma_pdf_impostors,
            xdata=impostors_X,
            ydata=impostors_Y,
            p0=[1.0, 1.0],
        )
        impostors_a, impostors_scale = impostors_popt

        # Fitting weibull for genuines and impostors
        genuines_popt, _ = curve_fit(
            gamma_pdf_genuines,
            xdata=genuines_X,
            ydata=genuines_Y,
            p0=[1.0, 1.0],
        )
        genuines_a, genuines_scale = genuines_popt

        self.gen_dist = partial(
            gamma_pdf_genuines, a=genuines_a, scale=genuines_scale
        )
        self.imp_dist = partial(
            gamma_pdf_impostors, a=impostors_a, scale=impostors_scale
        )
        return self

    def predict_proba(self, X):
        epsilon = 1e-10
        return +1 * (
            np.log10(self.gen_dist(X) + epsilon)
            - np.log10(self.imp_dist(X) + epsilon)
        )


class BetaCalibration(TransformerMixin, BaseEstimator):
    """
    Implements the weibull calibration using a pair of  Beta
    pdf's defined in:

    """

    def fit(self, X, y):
        def beta_pdf(x, a, b):
            return beta.pdf(np.abs(x), a, b)

        bins = 100

        impostors_X, impostors_Y = np.histogram(
            X[y == 0], bins=bins, density=True
        )
        # averaging the bins
        impostors_Y = 0.5 * (impostors_Y[1:] + impostors_Y[:-1])

        # Binining genuies and impostors for the fit
        genuines_X, genuines_Y = np.histogram(
            X[y == 1], bins=bins, density=True
        )
        # averaging the bins
        genuines_Y = 0.5 * (genuines_Y[1:] + genuines_Y[:-1])

        # gamma fit for impostor distribution
        impostors_popt, _ = curve_fit(
            beta_pdf, xdata=impostors_X, ydata=impostors_Y, p0=[1, 0.5]
        )
        impostors_a, impostors_b = impostors_popt

        # Fitting weibull for genuines and impostors
        genuines_popt, _ = curve_fit(
            beta_pdf, xdata=genuines_X, ydata=genuines_Y, p0=[1.0, 0.5]
        )
        genuines_a, genuines_b = genuines_popt

        self.gen_dist = partial(beta_pdf, a=genuines_a, b=genuines_b)
        self.imp_dist = partial(beta_pdf, a=impostors_a, b=impostors_b)
        return self

    def predict_proba(self, X):
        epsilon = 1e-10
        return +1 * (
            np.log10(self.gen_dist(X) + epsilon)
            - np.log10(self.imp_dist(X) + epsilon)
        )


class CategoricalCalibration(TransformerMixin, BaseEstimator):
    r"""
    Implements an adaptation of the Categorical Calibration defined in:

    `Mandasari, Miranti Indar, et al. "Score calibration in face recognition." Iet Biometrics 3.4 (2014): 246-256.`

    In such a work the calibration is defined as::
      :math:`s = \sum_{i=0}^{N} (\text{calibrator}_i)(X)`


    The category calibration is implemented in the tails of the score distributions in this implementation.
    For the impostor score distribution, the tail is defined between :math:`q_3(x)` and :math:`q3(x)+\alpha * (q3(x)-q1(x))`, where :math:`q_n` represents the quantile and :math:`\alpha` represents an offset.
    For the genuines score distribution, the tail is defined between :math:`q_1(x)` and :math:`q1(x)-\alpha * (q3(x)-q1(x))`.

    In this implementation one calibrator per category is fitted at training time.
    At test time, the maximum of the calibrated scores is returned.



    Parameters
    ----------
        field_name: str
           Reference field name in the csv score file. E.g. race, gender, ..,

        field_values: list
           Possible values for `field_name`. E.g ['male', 'female'], ['black', 'white']

        score_selection_method: str
           Method to select the scores for fetting the calibration models.
             `median-q3`: It will select the scores from the median to q3 from the impostor scores (q1 to median for genuines)
             `q3-outlier`: It will select the scores from q3 to outlier (q3+1.5*IQD) from the impostor scores (q1 to outlier for genuines)
             `q1-median`:
             `all`: It will select all the scores. Default to `median-q3`

        reduction_function:
           Pointer to a function to reduce the scores. Default to `np.mean`

        fit_estimator: None
           Estimator used for calibrations. Default to `LLRCalibration`
    """

    def __init__(
        self,
        field_name,
        field_values,
        score_selection_method="all",
        reduction_function=np.mean,
        fit_estimator=None,
    ):
        self.field_name = field_name
        self.field_values = field_values
        self.quantile_factor = 1.5
        self.score_selection_method = score_selection_method
        self.reduction_function = reduction_function

        if fit_estimator is None:
            self.fit_estimator = LLRCalibration
        else:
            self.fit_estimator = fit_estimator

    def fit(self, input_score_file_name):
        """
        Fit the calibrator

        Parameters
        ----------

           input_score_file_name: str
              Reference score file used to fit the calibrator (E.g `scores-dev.csv`).


        """

        def impostor_threshold(impostor_scores):
            """
            score > Q3 + 1.5*IQR
            """
            q1 = np.quantile(impostor_scores, 0.25)
            q3 = np.quantile(impostor_scores, 0.75)

            if self.score_selection_method == "q3-outlier":
                outlier_zone = q3 + self.quantile_factor * (q3 - q1)
                return impostor_scores[
                    (impostor_scores > q3) & (impostor_scores <= outlier_zone)
                ]
            elif self.score_selection_method == "q1-median":
                median = np.median(impostor_scores)
                return impostor_scores[
                    (impostor_scores > q1) & (impostor_scores <= median)
                ]
            elif self.score_selection_method == "median-q3":
                median = np.median(impostor_scores)
                return impostor_scores[
                    (impostor_scores < q3) & (impostor_scores >= median)
                ]
            else:
                return impostor_scores

        def genuine_threshold(genuine_scores):
            """
            score <= Q3 - 1.5*IQR
            """
            q1 = np.quantile(genuine_scores, 0.25)
            q3 = np.quantile(genuine_scores, 0.75)

            if self.score_selection_method == "q3-outlier":
                outlier_zone = q1 - self.quantile_factor * (q3 - q1)
                return genuine_scores[
                    (genuine_scores < q1) & (genuine_scores >= outlier_zone)
                ]
            elif self.score_selection_method == "q1-median":
                median = np.median(genuine_scores)
                return genuine_scores[
                    (genuine_scores < q3) & (genuine_scores <= median)
                ]
            elif self.score_selection_method == "median-q3":
                median = np.median(genuine_scores)
                return genuine_scores[
                    (genuine_scores > q1) & (genuine_scores <= median)
                ]
            else:
                return genuine_scores

        impostors, genuines = get_split_dataframe(input_score_file_name)
        self._categorical_fitters = []

        def get_sigmoid_score(X, regressor):
            return expit(X * regressor.coef_ + regressor.intercept_).ravel()

        for value in self.field_values:

            # Filtering genunines and impostors per group
            impostors_per_group = (
                impostors[
                    (impostors[f"probe_{self.field_name}"] == value)
                    & (impostors[f"bio_ref_{self.field_name}"] == value)
                ]["score"]
                .compute()
                .to_numpy()
            )
            genuines_per_group = (
                genuines[(genuines[f"probe_{self.field_name}"] == value)][
                    "score"
                ]
                .compute()
                .to_numpy()
            )

            impostors_per_group = impostor_threshold(impostors_per_group)
            genuines_per_group = genuine_threshold(genuines_per_group)

            # Training the regressor
            y = np.hstack(
                (
                    np.zeros(len(impostors_per_group)),
                    np.ones(len(genuines_per_group)),
                )
            )
            X = np.expand_dims(
                np.hstack((impostors_per_group, genuines_per_group)), axis=1
            )
            fitter = self.fit_estimator().fit(X, y)
            self._categorical_fitters.append(fitter)

        return self

    def transform(self, input_scores, calibrated_scores):
        """
        Calibrates a score

        Parameters
        ----------

           input_scores: list
              Input score files to be calibrated

           calibrated_files: list
              Output score files

        """

        assert isinstance(input_scores, list) or isinstance(input_scores, tuple)
        assert isinstance(calibrated_scores, list) or isinstance(
            calibrated_scores, tuple
        )
        assert len(calibrated_scores) == len(input_scores)
        for file_name, output_file_name in zip(input_scores, calibrated_scores):
            # Fetching scores
            dataframe = dask.dataframe.read_csv(file_name)
            dataframe = dataframe.compute()
            X = dataframe["score"].to_numpy()

            calibrated_scores = np.vstack(
                [
                    fitter.predict_proba(X)
                    for fitter in self._categorical_fitters
                ]
            ).T
            calibrated_scores = self.reduction_function(
                calibrated_scores, axis=1
            )
            dataframe["score"] = calibrated_scores

            dataframe.to_csv(output_file_name, index=False)

        return calibrated_scores
