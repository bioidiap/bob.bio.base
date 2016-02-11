#!/usr/bin/env python

import bob.bio.base
import scipy.spatial

algorithm = bob.bio.base.algorithm.Distance(
    distance_function = scipy.spatial.distance.cosine,
    is_distance_function = True
)
