#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Implementation of the Vanilla Biometrics pipeline using Dask :ref:`bob.bio.base.struct_bio_rec_sys`_

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import dask.bag
import dask.delayed
from bob.pipelines.sample import samplesets_to_samples, transform_sample_sets

def biometric_pipeline(
    background_model_samples,
    biometric_reference_samples,
    probe_samples,
    extractor,
    biometric_algorithm,
):

    ## Training background model (fit will return even if samples is ``None``,
    ## in which case we suppose the algorithm is not trainable in any way)
    extractor = train_background_model(
        background_model_samples,
        extractor
    )

    ## Create biometric samples
    biometric_references = create_biometric_reference(
        biometric_reference_samples, extractor, biometric_algorithm
    )


    ## Scores all probes
    return compute_scores(
        probe_samples,
        biometric_references,
        extractor,
        biometric_algorithm,
    )


def train_background_model(background_model_samples, extractor):

    # TODO: Maybe here is supervised
    X, y = samplesets_to_samples(background_model_samples)

    extractor = extractor.fit(X, y=y)

    return extractor


def create_biometric_reference(
    biometric_reference_samples, extractor, biometric_algorithm
):
    
    biometric_reference_features = transform_sample_sets(extractor, biometric_reference_samples)
    
    # features is a list of SampleSets    
    biometric_references = biometric_algorithm._enroll_samples(biometric_reference_features)
    
    # models is a list of Samples
    return biometric_references


def compute_scores(probe_samples, biometric_references, extractor, algorithm):

    # probes is a list of SampleSets
    probe_features = transform_sample_sets(extractor, probe_samples)
    # models is a list of Samples
    # features is a list of SampleSets

    scores = algorithm._score_samples(probe_features, biometric_references, extractor)
    # scores is a list of Samples
    return scores
