#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
TODO: This should be deployed in bob.pipelines
"""

from bob.pipelines.processor import CheckpointMixin, SampleMixin
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
import numpy

"""
Wraps the 
"""


class SamplePCA(SampleMixin, PCA):
    """
    Enables SAMPLE handling for https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    pass


class CheckpointSamplePCA(CheckpointMixin, SampleMixin, PCA):
    """
    Enables SAMPLE and CHECKPOINTIN handling for https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    pass
