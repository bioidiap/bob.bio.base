""" This script calculates the Hamming distance (proportion of mis-matching corresponding bits) between two binary arrays """

import scipy.spatial.distance

import bob.bio.base

algorithm = bob.bio.base.algorithm.Distance(
    distance_function=scipy.spatial.distance.hamming, is_distance_function=True
)
