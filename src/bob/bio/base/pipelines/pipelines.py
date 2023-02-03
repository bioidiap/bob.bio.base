#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Implementation of the PipelineSimple using Dask :ref:`bob.bio.base.struct_bio_rec_sys`_

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import logging

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from bob.bio.base.pipelines.abstract_classes import BioAlgorithm
from bob.pipelines import SampleWrapper, is_instance_nested, wrap

from .score_writers import FourColumnsScoreWriter

logger = logging.getLogger(__name__)
import tempfile


class PipelineSimple:
    """
    The simplest possible pipeline

    This is the backbone of most biometric recognition systems.
    It implements three subpipelines and they are the following:

     - :py:class:`PipelineSimple.train_background_model`: Initializes or trains your transformer.
        It will run :py:meth:`sklearn.base.BaseEstimator.fit`

     - :py:class:`PipelineSimple.enroll_templates`: Creates enrollment templates
        It will run :py:meth:`sklearn.base.BaseEstimator.transform` followed by a sequence of
        :py:meth:`bob.bio.base.pipelines.abstract_classes.BioAlgorithm.create_templates`

     - :py:class:`PipelineSimple.probe_templates`: Creates probe templates
        It will run :py:meth:`sklearn.base.BaseEstimator.transform` followed by a sequence of
        :py:meth:`bob.bio.base.pipelines.abstract_classes.BioAlgorithm.create_templates`

     - :py:class:`PipelineSimple.compute_scores`: Computes scores
        It will run :py:meth:`bob.bio.base.pipelines.abstract_classes.BioAlgorithm.compare`


    Example
    -------
       >>> from sklearn.preprocessing import FunctionTransformer
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.algorithm import Distance
       >>> from bob.bio.base.pipelines import PipelineSimple
       >>> from bob.pipelines import wrap
       >>> import numpy
       >>> linearize = lambda samples: [numpy.reshape(x, (-1,)) for x in samples]
       >>> transformer = wrap(["sample"], FunctionTransformer(linearize))
       >>> transformer_pipeline = make_pipeline(transformer)
       >>> biometric_algorithm = Distance()
       >>> pipeline = PipelineSimple(transformer_pipeline, biometric_algorithm)
       >>> pipeline(samples_for_training_back_ground_model, samplesets_for_enroll, samplesets_for_scoring)  # doctest: +SKIP


    To run this pipeline using Dask, used the function
    :py:func:`dask_bio_pipeline`.

    Example
    -------
      >>> from bob.bio.base.pipelines import dask_bio_pipeline
      >>> pipeline = PipelineSimple(transformer_pipeline, biometric_algorithm)
      >>> pipeline = dask_bio_pipeline(pipeline)
      >>> pipeline(samples_for_training_back_ground_model, samplesets_for_enroll, samplesets_for_scoring).compute()  # doctest: +SKIP


    Parameters
    ----------

    transformer: :py:class`sklearn.pipeline.Pipeline` or a `sklearn.base.BaseEstimator`
        Transformer that will preprocess your data

    biometric_algorithm: :py:class:`bob.bio.base.pipelines.abstract_classes.BioAlgorithm`
        Biometrics algorithm object that implements the methods `enroll` and
        `score` methods

    score_writer: :any:`bob.bio.base.pipelines.ScoreWriter`
        Format to write scores. Default to
        :any:`bob.bio.base.pipelines.FourColumnsScoreWriter`

    """

    def __init__(
        self,
        transformer: Pipeline,
        biometric_algorithm: BioAlgorithm,
        score_writer=None,
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
        score_all_vs_all=True,
        return_templates=False,
    ):
        logger.info(" >> PipelineSimple: Training background model")
        self.train_background_model(background_model_samples)

        logger.info(" >> PipelineSimple: Creating enroll templates")
        enroll_templates = self.enroll_templates(biometric_reference_samples)

        logger.info(" >> PipelineSimple: Creating probe templates")
        probe_templates = self.probe_templates(probe_samples)

        logger.info(" >> PipelineSimple: Computing scores")
        scores = self.compute_scores(
            probe_templates,
            enroll_templates,
            score_all_vs_all,
        )

        if return_templates:
            return scores, enroll_templates, probe_templates
        else:
            return scores

    def train_background_model(self, background_model_samples):
        # background_model_samples is a list of Samples

        # We might have algorithms that has no data for training
        if len(background_model_samples) > 0:
            self.transformer.fit(background_model_samples)
        else:
            logger.warning(
                "There's no data to train background model. "
                "For the rest of the execution it will be assumed that the pipeline does not require fit."
            )
        return self.transformer

    def enroll_templates(self, biometric_reference_samples):
        biometric_reference_features = self.transformer.transform(
            biometric_reference_samples
        )

        enroll_templates = (
            self.biometric_algorithm.create_templates_from_samplesets(
                biometric_reference_features, enroll=True
            )
        )

        # a list of Samples
        return enroll_templates

    def probe_templates(self, probe_samples):
        probe_features = self.transformer.transform(probe_samples)

        probe_templates = (
            self.biometric_algorithm.create_templates_from_samplesets(
                probe_features, enroll=False
            )
        )

        # a list of Samples
        return probe_templates

    def compute_scores(
        self,
        probe_templates,
        enroll_templates,
        score_all_vs_all,
    ):
        return self.biometric_algorithm.score_sample_templates(
            probe_templates, enroll_templates, score_all_vs_all
        )

    def write_scores(self, scores):
        if self.score_writer is None:
            raise ValueError("No score writer defined in the pipeline")
        return self.score_writer.write(scores)

    def post_process(self, score_paths, filename):
        if self.score_writer is None:
            raise ValueError("No score writer defined in the pipeline")

        return self.score_writer.post_process(score_paths, filename)


def check_valid_pipeline(pipeline_simple):
    """
    Applying some checks in the PipelineSimple
    """

    # CHECKING THE TRANSFORMER
    # Checking if it's a Scikit Pipeline or an estimator
    if isinstance(pipeline_simple.transformer, Pipeline):
        # Checking if all steps are wrapped as samples, if not, we should wrap them
        for p in pipeline_simple.transformer:
            if (
                not is_instance_nested(p, "estimator", SampleWrapper)
                and type(p) is not str
                and p is not None
            ):
                wrap(["sample"], p)

    # In this case it can be a simple estimator. AND
    # Checking if it's sample wrapper, if not, do it
    elif is_instance_nested(
        pipeline_simple.transformer, "estimator", BaseEstimator
    ) and is_instance_nested(
        pipeline_simple.transformer, "estimator", BaseEstimator
    ):
        wrap(["sample"], pipeline_simple.transformer)
    else:
        raise ValueError(
            f"pipeline_simple.transformer should be instance of either `sklearn.pipeline.Pipeline` or"
            f"sklearn.base.BaseEstimator, not {pipeline_simple.transformer}"
        )

    # Checking the Biometric algorithm
    if not isinstance(pipeline_simple.biometric_algorithm, BioAlgorithm):
        raise ValueError(
            f"pipeline_simple.biometric_algorithm should be instance of `BioAlgorithm`"
            f"not {pipeline_simple.biometric_algorithm}"
        )

    return True
