#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012

import os
import shutil
import numpy
import math
from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_refs = False

seed_value = 5489

import scipy.spatial

import bob.io.base
import bob.learn.linear
import bob.io.base.test_utils
import bob.bio.base
from . import utils


def _compare(
    data, reference, write_function=bob.bio.base.save, read_function=bob.bio.base.load
):
    # execute the preprocessor
    if regenerate_refs:
        write_function(data, reference)

    assert numpy.allclose(data, read_function(reference), atol=1e-5)


def test_distance():
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
        abs(euclidean.score_for_multiple_probes(model, [f2, f2]) + math.sqrt(200.0))
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


def test_pca():
    temp_file = bob.io.base.test_utils.temporary_filename()
    # load PCA from configuration
    pca1 = bob.bio.base.load_resource(
        "pca", "algorithm", preferred_package="bob.bio.base"
    )
    assert isinstance(pca1, bob.bio.base.algorithm.PCA)
    assert isinstance(pca1, bob.bio.base.algorithm.Algorithm)
    assert pca1.performs_projection
    assert pca1.requires_projector_training
    assert pca1.use_projected_features_for_enrollment
    assert not pca1.split_training_features_by_client
    assert not pca1.requires_enroller_training

    # generate a smaller PCA subspcae
    pca2 = bob.bio.base.algorithm.PCA(5)

    # create random training set
    train_set = utils.random_training_set(200, 500, 0.0, 255.0)
    # train the projector
    reference_file = pkg_resources.resource_filename(
        "bob.bio.base.test", "data/pca_projector.hdf5"
    )
    try:
        # train projector
        pca2.train_projector(train_set, temp_file)
        assert os.path.exists(temp_file)

        if regenerate_refs:
            shutil.copy(temp_file, reference_file)

        # check projection matrix
        pca1.load_projector(reference_file)
        pca2.load_projector(temp_file)

        assert numpy.allclose(pca1.variances, pca2.variances, atol=1e-5)
        assert pca1.machine.shape == (200, 5)
        assert pca1.machine.shape == pca2.machine.shape
        # ... rotation direction might change, hence either the sum or the difference should be 0
        for i in range(5):
            assert numpy.allclose(
                pca1.machine.weights[:, i], pca2.machine.weights[:, i], atol=1e-5
            ) or numpy.allclose(
                pca1.machine.weights[:, i], -pca2.machine.weights[:, i], atol=1e-5
            )

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # generate and project random feature
    feature = utils.random_array(200, 0.0, 255.0, seed=84)
    projected = pca1.project(feature)
    assert projected.shape == (5,)
    _compare(
        projected,
        pkg_resources.resource_filename("bob.bio.base.test", "data/pca_projected.hdf5"),
        pca1.write_feature,
        pca1.read_feature,
    )

    # enroll model from random features
    enroll = utils.random_training_set(5, 5, 0.0, 255.0, seed=21)
    model = pca1.enroll(enroll)
    _compare(
        model,
        pkg_resources.resource_filename("bob.bio.base.test", "data/pca_model.hdf5"),
        pca1.write_model,
        pca1.read_model,
    )

    # compare model with probe
    probe = pca1.read_feature(
        pkg_resources.resource_filename("bob.bio.base.test", "data/pca_projected.hdf5")
    )
    reference_score = -251.53563107
    assert (
        abs(numpy.mean(pca1.score(model, probe)) - reference_score) < 1e-5
    ), "The scores differ: %3.8f, %3.8f" % (pca1.score(model, probe), reference_score)
    assert (
        abs(pca1.score_for_multiple_probes(model, [probe, probe]) - reference_score)
        < 1e-5
    )

    # test the calculation of the subspace dimension based on percentage of variance
    pca3 = bob.bio.base.algorithm.PCA(0.9)
    try:
        # train projector
        pca3.train_projector(train_set, temp_file)
        assert os.path.exists(temp_file)
        assert pca3.subspace_dim == 140
        pca3.load_projector(temp_file)
        assert pca3.machine.shape[1] == 140
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_lda():
    temp_file = bob.io.base.test_utils.temporary_filename()
    # assure that the configurations are loadable
    lda1 = bob.bio.base.load_resource(
        "lda", "algorithm", preferred_package="bob.bio.base"
    )
    assert isinstance(lda1, bob.bio.base.algorithm.LDA)
    assert isinstance(lda1, bob.bio.base.algorithm.Algorithm)
    lda2 = bob.bio.base.load_resource(
        "pca+lda", "algorithm", preferred_package="bob.bio.base"
    )
    assert isinstance(lda2, bob.bio.base.algorithm.LDA)
    assert isinstance(lda2, bob.bio.base.algorithm.Algorithm)

    assert lda1.performs_projection
    assert lda1.requires_projector_training
    assert lda1.use_projected_features_for_enrollment
    assert lda1.split_training_features_by_client
    assert not lda1.requires_enroller_training

    # generate a smaller PCA subspcae
    lda3 = bob.bio.base.algorithm.LDA(
        5, 10, scipy.spatial.distance.seuclidean, True, True
    )

    # create random training set
    train_set = utils.random_training_set_by_id(
        200, count=20, minimum=0.0, maximum=255.0
    )
    # train the projector
    reference_file = pkg_resources.resource_filename(
        "bob.bio.base.test", "data/lda_projector.hdf5"
    )
    try:
        # train projector
        lda3.train_projector(train_set, temp_file)
        assert os.path.exists(temp_file)

        if regenerate_refs:
            shutil.copy(temp_file, reference_file)

        # check projection matrix
        lda1.load_projector(reference_file)
        lda3.load_projector(temp_file)

        assert numpy.allclose(lda1.variances, lda3.variances, atol=1e-5)
        assert lda3.machine.shape == (200, 5)
        assert lda1.machine.shape == lda3.machine.shape
        # ... rotation direction might change, hence either the sum or the difference should be 0
        for i in range(5):
            assert numpy.allclose(
                lda1.machine.weights[:, i], lda3.machine.weights[:, i], atol=1e-5
            ) or numpy.allclose(
                lda1.machine.weights[:, i], -lda3.machine.weights[:, i], atol=1e-5
            )

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # generate and project random feature
    feature = utils.random_array(200, 0.0, 255.0, seed=84)
    projected = lda1.project(feature)
    assert projected.shape == (5,)
    _compare(
        projected,
        pkg_resources.resource_filename("bob.bio.base.test", "data/lda_projected.hdf5"),
        lda1.write_feature,
        lda1.read_feature,
    )

    # enroll model from random features
    enroll = utils.random_training_set(5, 5, 0.0, 255.0, seed=21)
    model = lda1.enroll(enroll)
    _compare(
        model,
        pkg_resources.resource_filename("bob.bio.base.test", "data/lda_model.hdf5"),
        lda1.write_model,
        lda1.read_model,
    )
    # compare model with probe
    probe = lda1.read_feature(
        pkg_resources.resource_filename("bob.bio.base.test", "data/lda_projected.hdf5")
    )
    reference_score = -233.30450012
    assert (
        abs(numpy.mean(lda1.score(model, probe)) - reference_score) < 1e-5
    ), "The scores differ: %3.8f, %3.8f" % (lda1.score(model, probe), reference_score)
    assert (
        abs(lda1.score_for_multiple_probes(model, [probe, probe]) - reference_score)
        < 1e-5
    )

    # test the calculation of the subspace dimension based on percentage of variance
    lda4 = bob.bio.base.algorithm.LDA(pca_subspace_dimension=0.9)
    try:
        # train projector
        lda4.train_projector(train_set, temp_file)
        assert os.path.exists(temp_file)
        assert lda4.pca_subspace == 132
        lda4.load_projector(temp_file)
        assert lda4.machine.shape[1] == 19
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_distance():

    import scipy.spatial

    # assure that the configurations are loadable
    distance = bob.bio.base.load_resource(
        "distance-cosine", "algorithm", preferred_package="bob.bio.base"
    )
    assert isinstance(distance, bob.bio.base.algorithm.Distance)
    assert isinstance(distance, bob.bio.base.algorithm.Algorithm)

    assert distance.performs_projection == False
    assert distance.requires_projector_training == False
    assert distance.use_projected_features_for_enrollment == False
    assert distance.split_training_features_by_client == False
    assert distance.requires_enroller_training == False

    distance = bob.bio.base.algorithm.Distance(
        distance_function=scipy.spatial.distance.cosine, is_distance_function=True
    )

    # compare model with probe
    enroll = utils.random_training_set(5, 5, 0.0, 255.0, seed=21)
    model = numpy.mean(distance.enroll(enroll), axis=0)
    probe = bob.io.base.load(
        pkg_resources.resource_filename("bob.bio.base.test", "data/lda_projected.hdf5")
    )

    reference_score = -0.1873371
    assert (
        abs(distance.score(model, probe) - reference_score) < 1e-5
    ), "The scores differ: %3.8f, %3.8f" % (
        distance.score(model, probe),
        reference_score,
    )
