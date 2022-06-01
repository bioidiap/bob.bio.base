"""
#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

Implementation of a pipeline that post process scores


"""

import logging
import os

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
from bob.pipelines import Sample, SampleSet, getattr_nested, is_instance_nested

from .pipelines import PipelineSimple

logger = logging.getLogger(__name__)


class PipelineScoreNorm:
    """
    Apply Z, T or ZT Score normalization on top of Pimple Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in
    Speaker Recognition https://arxiv.org/pdf/1709.09868.pdf


    Example
    -------
       >>> from sklearn.preprocessing import FunctionTransformer
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.algorithm import Distance
       >>> from bob.bio.base.pipelines import PipelineSimple, PipelineScoreNorm, ZNormScores
       >>> from bob.pipelines import wrap
       >>> import numpy
       >>> linearize = lambda samples: [numpy.reshape(x, (-1,)) for x in samples]
       >>> transformer = wrap(["sample"], FunctionTransformer(linearize))
       >>> transformer_pipeline = make_pipeline(transformer)
       >>> biometric_algorithm = Distance()
       >>> pipeline_simple = PipelineSimple(transformer_pipeline, biometric_algorithm)
       >>> z_norm_postprocessor = ZNormScores()
       >>> z_pipeline = PipelineScoreNorm(pipeline_simple, z_norm_postprocessor)
       >>> zt_pipeline(...) #doctest: +SKIP

    Parameters
    ----------
    pipeline_simple: :any:`PipelineSimple`
        An instance :any:`PipelineSimple` to the wrapped with score
        normalization

    post_processor: :py:class`sklearn.pipeline.Pipeline` or a `sklearn.base.BaseEstimator`
        Transformer that will post process the scores

    score_writer
        A ScoreWriter to write the scores
    """

    def __init__(
        self,
        pipeline_simple: PipelineSimple,
        post_processor,
    ):

        self.pipeline_simple = pipeline_simple
        self.post_processor = post_processor

        # TODO: ACTIVATE THAT
        # check_valid_pipeline(self)

    @property
    def biometric_algorithm(self):
        return self.pipeline_simple.biometric_algorithm

    @biometric_algorithm.setter
    def biometric_algorithm(self, value):
        self.pipeline_simple.biometric_algorithm = value

    @property
    def transformer(self):
        return self.pipeline_simple.transformer

    @transformer.setter
    def transformer(self, value):
        self.pipeline_simple.transformer = value

    @property
    def score_writer(self):
        return self.pipeline_simple.score_writer

    @score_writer.setter
    def score_writer(self, value):
        self.pipeline_simple.score_writer = value

    def __call__(
        self,
        background_model_samples,
        biometric_reference_samples,
        probe_samples,
        post_process_samples,
        score_all_vs_all=False,
    ):
        raw_scores, enroll_templates, probe_templates = self.pipeline_simple(
            background_model_samples=background_model_samples,
            biometric_reference_samples=biometric_reference_samples,
            probe_samples=probe_samples,
            score_all_vs_all=score_all_vs_all,
            return_templates=True,
        )

        # TODO: I know this is ugly, but I don't want to create one pipeline for every single
        # normalization strategy
        if is_instance_nested(
            self,
            "biometric_algorithm",
            bob.bio.base.pipelines.BioAlgCheckpointWrapper,
        ):

            if is_instance_nested(
                self,
                "biometric_algorithm",
                bob.bio.base.pipelines.BioAlgDaskWrapper,
            ):
                self.biometric_algorithm.biometric_algorithm.biometric_reference_dir = os.path.join(
                    self.biometric_algorithm.biometric_algorithm.biometric_reference_dir,
                    "score-norm",
                )

            else:
                self.biometric_algorithm.biometric_reference_dir = os.path.join(
                    self.biometric_algorithm.biometric_reference_dir,
                    "score-norm",
                )

        template_format = getattr_nested(
            self.post_processor, "post_process_template"
        )
        if template_format == "probe":
            post_process_templates = self.pipeline_simple.probe_templates(
                post_process_samples
            )
            post_process_scores = self.pipeline_simple.compute_scores(
                probe_templates=post_process_templates,
                enroll_templates=enroll_templates,
                score_all_vs_all=False,
            )
        elif template_format == "enroll":
            post_process_templates = self.pipeline_simple.enroll_templates(
                post_process_samples
            )
            post_process_scores = self.pipeline_simple.compute_scores(
                probe_templates=probe_templates,
                enroll_templates=post_process_templates,
                score_all_vs_all=True,
            )
        else:
            raise ValueError(
                "post_process_template must be either 'probe' or 'enroll', got {}".format(
                    template_format
                )
            )

        self.post_processor.fit(post_process_scores)
        post_processed_scores = self.post_processor.transform(raw_scores)

        return raw_scores, post_processed_scores

    def write_scores(self, scores):
        return self.pipeline_simple.write_scores(scores)

    def post_process(self, score_paths, filename):
        return self.pipeline_simple.post_process(score_paths, filename)


def copy_learned_attributes(from_estimator, to_estimator):
    attrs = {k: v for k, v in vars(from_estimator).items() if k.endswith("_")}

    for k, v in attrs.items():
        setattr(to_estimator, k, v)


class ZNormScores(TransformerMixin, BaseEstimator):
    """
    Apply Z-Norm Score normalization on top of Simple Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Parameters
    ----------

    """

    post_process_template = "probe"

    def __init__(
        self,
        top_norm=False,
        top_norm_score_fraction=0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_norm = top_norm
        self.top_norm_score_fraction = top_norm_score_fraction

    def fit(self, z_scores, y=None):

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

    post_process_template = "enroll"

    def __init__(
        self,
        top_norm=False,
        top_norm_score_fraction=0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_norm = top_norm
        self.top_norm_score_fraction = top_norm_score_fraction

    def fit(self, t_scores, y=None):

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
