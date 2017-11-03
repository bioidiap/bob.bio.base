""" This script calculates the Hamming distance (proportion of mis-matching corresponding bits) between two binary arrays """

import bob.bio.base
import scipy.spatial.distance

algorithm = bob.bio.base.algorithm.Distance(
    distance_function = scipy.spatial.distance.hamming,
    is_distance_function = True
)
