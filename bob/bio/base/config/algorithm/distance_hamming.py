""" This script calculates the Hamming distance (proportion of mis-matching corresponding bits) between two binary arrays """

import bob.bio.base

algorithm = bob.bio.base.algorithm.Distance(distance_function="hamming")
