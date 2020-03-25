#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Implementation of the Vanilla Biometrics pipeline using Dask :ref:`bob.bio.base.struct_bio_rec_sys`_

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import logging

logger = logging.getLogger(__name__)


def biometric_pipeline(
    background_model_samples,
    biometric_reference_samples,
    probe_samples,
    transformer,
    biometric_algorithm,
):
    logger.info(
        f" >> Vanilla Biometrics: Training background model with pipeline {transformer}"
    )

    # Training background model (fit will return even if samples is ``None``,
    # in which case we suppose the algorithm is not trainable in any way)
    transformer = train_background_model(background_model_samples, transformer)

    logger.info(
        f" >> Creating biometric references with the biometric algorithm {biometric_algorithm}"
    )

    # Create biometric samples
    biometric_references = create_biometric_reference(
        biometric_reference_samples, transformer, biometric_algorithm
    )

    logger.info(
        f" >> Computing scores with the biometric algorithm {biometric_algorithm}"
    )

    # Scores all probes
    return compute_scores(
        probe_samples, biometric_references, transformer, biometric_algorithm
    )


def train_background_model(background_model_samples, transformer):
    # background_model_samples is a list of Samples
    transformer = transformer.fit(background_model_samples)
    return transformer


def create_biometric_reference(
    biometric_reference_samples, transformer, biometric_algorithm
):
    biometric_reference_features = transformer.transform(biometric_reference_samples)

    biometric_references = biometric_algorithm.enroll_samples(
        biometric_reference_features
    )

    # models is a list of Samples
    return biometric_references


def compute_scores(
    probe_samples, biometric_references, transformer, biometric_algorithm
):

    # probes is a list of SampleSets
    probe_features = transformer.transform(probe_samples)

    scores = biometric_algorithm.score_samples(probe_features, biometric_references)

    # scores is a list of Samples
    return scores
