#!/bin/python

# This file describes an exemplary configuration file that can be used in combination with the bin/parameter_test.py script.


# The preprocessor uses two fake parameters, which are called #1 and #4
preprocessor = "bob.bio.base.test.dummy.preprocessor.DummyPreprocessor(fake_parameter=#1, other_parameters=#4)"

# The extractor uses the **default** 'dummy' option, which is registered as a resource
extractor = "dummy"

# The algorithm uses two fake parameters, which are called #2 and #3
algorithm = "bob.bio.base.test.dummy.algorithm.DummyAlgorithm(fake_distance_function=#2, is_distance_function=#3)"


# Here, we define, which placeholder keys (#.) should be replaces by which values in which stage of the processing toolchain
replace = {
    # For preprocessing, select two independent dummy parameters
    'preprocess' : {
        # Fake parameter to be selected for placeholder #1
        "#1" : {
            'P1' : 10,
            'P2' : 20,
        },
        # fake parameter to be selected for placeholder #4
        "#4" : {
            'F1' : 15,
            'F2' : 30
        }
    },

    # For scoring, select two dependent dummy parameters
    'score' : {
        # Replace placeholders #2 and #3 **at the same time**
        "(#2, #3)" : {
            # For distance_function = 'bob.math.histogram_intersection' and is_distance_function = False, place result in sub-directory 'D1'
            'S1' : ('bob.math.histogram_intersection', 'False'),
            # For distance_function = 'bob.math.chi_square' and is_distance_function = True, place result in sub-directory 'D2'
            'S2' : ('bob.math.chi_square', 'True')
        }
    }
}

# An optional list of requirements
# If these requirements are not fulfilled for the current values of #1 and #4, these experiments will not be executed.
requirements = ["2*#1 > #4"]

# A list of imports that are required to use the defined preprocessor, extractor and algorithm from above
imports = ['bob.math', 'bob.bio.base.test.dummy']
