#!/usr/bin/env python

import bob.bio.base
import scipy.spatial

algorithm = bob.bio.base.algorithm.LDA(
    subspace_dimension = 50,
    pca_subspace_dimension = 100,
    distance_function = scipy.spatial.distance.euclidean,
    is_distance_function = True
)
