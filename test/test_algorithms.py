#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012

import numpy as np
import pkg_resources

import bob.bio.base
import bob.io.base

from . import utils


def test_distance_algorithm():
    # test the two registered distance functions

    # euclidean distance
    euclidean = bob.bio.base.algorithm.Distance(distance_function="euclidean")
    assert isinstance(euclidean, bob.bio.base.algorithm.Distance)

    # test distance computation
    f1 = np.ones((20, 10), np.float64)
    f2 = np.ones((20, 10), np.float64) * 2.0

    models = euclidean.create_templates([[f1, f1]], enroll=True)
    probes = euclidean.create_templates([[f2, f2]], enroll=False)
    score = euclidean.compare(models, probes)[0, 0]
    np.testing.assert_almost_equal(score, -np.sqrt(10))

    # test cosine distance
    cosine = bob.bio.base.algorithm.Distance(distance_function="cosine")
    models = cosine.create_templates([[f1, f1]], enroll=True)
    probes = cosine.create_templates([[f2, f2]], enroll=False)
    score = cosine.compare(models, probes)[0, 0]
    np.testing.assert_almost_equal(score, 0)

    # compare model with probe
    enroll = utils.random_training_set(5, 5, 0.0, 255.0, seed=21)
    probe = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.base.test", "data/lda_projected.hdf5"
        )
    )
    models = cosine.create_templates([enroll], enroll=True)
    probes = cosine.create_templates([probe], enroll=False)
    score = cosine.compare(models, probes)[0, 0]

    np.testing.assert_almost_equal(score, -0.1873371)
