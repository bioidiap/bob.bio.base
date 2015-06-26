#!/usr/bin/env python

import bob.bio.base

algorithm = bob.bio.base.algorithm.PLDA(
    subspace_dimension_of_f = 16, # Size of subspace F
    subspace_dimension_of_g = 16, # Size of subspace G
    subspace_dimension_pca = 0.9  # Percentage of PCA variance
)
