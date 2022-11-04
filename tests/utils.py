#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu Jul 19 17:09:55 CEST 2012

import functools
import importlib
import os
import sys
import unittest

import numpy


# based on: http://stackoverflow.com/questions/6796492/temporarily-redirect-stdout-stderr
class Quiet(object):
    """A class that supports the ``with`` statement to redirect any output of wrapped function calls to /dev/null"""

    def __init__(self):
        devnull = open(os.devnull, "w")
        self._stdout = devnull
        self._stderr = devnull

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


import logging

logger = logging.getLogger("bob.bio.base")


def random_array(shape, minimum=0, maximum=1, seed=42):
    # generate a random sequence of features
    numpy.random.seed(seed)
    return numpy.random.random(shape) * (maximum - minimum) + minimum


def random_training_set(shape, count, minimum=0, maximum=1, seed=42):
    """Returns a random training set with the given shape and the given number of elements."""
    # generate a random sequence of features
    numpy.random.seed(seed)
    return [
        numpy.random.random(shape) * (maximum - minimum) + minimum
        for i in range(count)
    ]


def random_training_set_by_id(shape, count=50, minimum=0, maximum=1, seed=42):
    # generate a random sequence of features
    numpy.random.seed(seed)
    train_set = []
    for i in range(count):
        train_set.append(
            [
                numpy.random.random(shape) * (maximum - minimum) + minimum
                for j in range(count)
            ]
        )
    return train_set


def is_library_available(library):
    """Decorator to check if the mxnet is present, before running the test"""

    def _is_library_available(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                importlib.import_module(library)

                return function(*args, **kwargs)
            except ImportError as e:
                # unittest.SkipTest is compatible with both nose and pytest
                raise unittest.SkipTest(
                    f"Skipping test since `{library}` is not available: %s" % e
                )

        return wrapper

    return _is_library_available
