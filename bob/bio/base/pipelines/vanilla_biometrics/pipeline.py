#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Implementation of the Vanilla Biometrics pipeline using Dask :ref:`bob.bio.base.struct_bio_rec_sys`_

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import dask.bag
import dask.delayed
from bob.pipelines.sample import samplesets_to_samples


def biometric_pipeline(
    background_model_samples,
    biometric_reference_samples,
    probe_samples,
    extractor,
    biometric_algorithm,
):

    ## Training background model (fit will return even if samples is ``None``,
    ## in which case we suppose the algorithm is not trainable in any way)
    extractor = train_background_model(background_model_samples, extractor)

    ## Create biometric samples
    biometric_references = create_biometric_reference(
        biometric_reference_samples, extractor, biometric_algorithm
    )

    ## Scores all probes
    return compute_scores(
        probe_samples, biometric_references, extractor, biometric_algorithm
    )


def train_background_model(background_model_samples, extractor):

    # TODO: Maybe here is supervised
    X, y = samplesets_to_samples(background_model_samples)

    extractor = extractor.fit(X, y=y)

    return extractor


def create_biometric_reference(
    biometric_reference_samples, extractor, biometric_algorithm
):
    biometric_reference_features = extractor.transform(biometric_reference_samples)

    # TODO: I KNOW THIS LOOKS UGLY, BUT THIS `MAP_PARTITIONS` HAS TO APPEAR SOMEWHERE
    # I COULD WORK OUT A MIXIN FOR IT, BUT THE USER WOULD NEED TO SET THAT SOMETWHERE
    # HERE'S ALREADY SETTING ONCE (for the pipeline) AND I DON'T WANT TO MAKE
    # THEM SET IN ANOTHER PLACE
    # LET'S DISCUSS THIS ON SLACK

    if isinstance(biometric_reference_features, dask.bag.core.Bag):
        # ASSUMING THAT IS A DASK THING IS COMMING
        biometric_references = biometric_reference_features.map_partitions(
            biometric_algorithm._enroll_samples
        )
    else:
        biometric_references = biometric_algorithm._enroll_samples(
            biometric_reference_features
        )

    # models is a list of Samples
    return biometric_references


def compute_scores(probe_samples, biometric_references, extractor, biometric_algorithm):

    # probes is a list of SampleSets
    probe_features = extractor.transform(probe_samples)

    # TODO: I KNOW THIS LOOKS UGLY, BUT THIS `MAP_PARTITIONS` HAS TO APPEAR SOMEWHERE
    # I COULD WORK OUT A MIXIN FOR IT, BUT THE USER WOULD NEED TO SET THAT SOMETWHERE
    # HERE'S ALREADY SETTING ONCE (for the pipeline) AND I DON'T WANT TO MAKE
    # THEM SET IN ANOTHER PLACE
    # LET'S DISCUSS THIS ON SLACK
    if isinstance(probe_features, dask.bag.core.Bag):
        # ASSUMING THAT IS A DASK THING IS COMMING

        ## TODO: Here, we are sending all computed biometric references to all
        ## probes.  It would be more efficient if only the models related to each
        ## probe are sent to the probing split.  An option would be to use caching
        ## and allow the ``score`` function above to load the required data from
        ## the disk, directly.  A second option would be to generate named delays
        ## for each model and then associate them here.

        all_references = dask.delayed(list)(biometric_references)

        scores = probe_features.map_partitions(
            biometric_algorithm._score_samples, all_references, extractor
        )

    else:
        scores = biometric_algorithm._score_samples(
            probe_features, biometric_references, extractor
        )

    # scores is a list of Samples
    return scores
