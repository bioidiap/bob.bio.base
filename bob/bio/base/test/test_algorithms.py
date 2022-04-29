#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012

import math

import numpy
import pkg_resources

regenerate_refs = False

seed_value = 5489

import scipy.spatial

import bob.bio.base
import bob.io.base
import bob.io.base.test_utils

from . import utils


def _compare(
    data,
    reference,
    write_function=bob.bio.base.save,
    read_function=bob.bio.base.load,
):
    # execute the preprocessor
    if regenerate_refs:
        write_function(data, reference)

    assert numpy.allclose(data, read_function(reference), atol=1e-5)


def test_distance_euclidean():
    # test the two registered distance functions

    # euclidean distance
    euclidean = bob.bio.base.load_resource(
        "distance-euclidean", "algorithm", preferred_package="bob.bio.base"
    )
    assert isinstance(euclidean, bob.bio.base.algorithm.Distance)
    assert isinstance(euclidean, bob.bio.base.algorithm.Algorithm)
    assert not euclidean.performs_projection
    assert not euclidean.requires_projector_training
    assert not euclidean.use_projected_features_for_enrollment
    assert not euclidean.split_training_features_by_client
    assert not euclidean.requires_enroller_training

    # test distance computation
    f1 = numpy.ones((20, 10), numpy.float64)
    f2 = numpy.ones((20, 10), numpy.float64) * 2.0

    model = euclidean.enroll([f1, f1])
    assert (
        abs(
            euclidean.score_for_multiple_probes(model, [f2, f2])
            + math.sqrt(200.0)
        )
        < 1e-6
    ), euclidean.score_for_multiple_probes(model, [f2, f2])

    # test cosine distance
    cosine = bob.bio.base.load_resource(
        "distance-cosine", "algorithm", preferred_package="bob.bio.base"
    )
    model = cosine.enroll([f1, f1])
    assert (
        abs(cosine.score_for_multiple_probes(model, [f2, f2])) < 1e-8
    ), cosine.score_for_multiple_probes(model, [f2, f2])


def test_distance_cosine():
    # assure that the configurations are loadable
    distance = bob.bio.base.load_resource(
        "distance-cosine", "algorithm", preferred_package="bob.bio.base"
    )
    assert isinstance(distance, bob.bio.base.algorithm.Distance)
    assert isinstance(distance, bob.bio.base.algorithm.Algorithm)

    assert not distance.performs_projection
    assert not distance.requires_projector_training
    assert not distance.use_projected_features_for_enrollment
    assert not distance.split_training_features_by_client
    assert not distance.requires_enroller_training

    distance = bob.bio.base.algorithm.Distance(
        distance_function=scipy.spatial.distance.cosine,
        is_distance_function=True,
    )

    # compare model with probe
    enroll = utils.random_training_set(5, 5, 0.0, 255.0, seed=21)
    model = numpy.mean(distance.enroll(enroll), axis=0)
    probe = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.base.test", "data/lda_projected.hdf5"
        )
    )

    reference_score = -0.1873371
    assert (
        abs(distance.score(model, probe) - reference_score) < 1e-5
    ), "The scores differ: %3.8f, %3.8f" % (
        distance.score(model, probe),
        reference_score,
    )
