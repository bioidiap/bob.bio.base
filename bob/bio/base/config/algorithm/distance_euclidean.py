#!/usr/bin/env python

import scipy.spatial

import bob.bio.base

algorithm = bob.bio.base.algorithm.Distance(
    distance_function=scipy.spatial.distance.euclidean,
    is_distance_function=True,
)
