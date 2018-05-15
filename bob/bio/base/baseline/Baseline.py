#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


class Baseline(object):
    """
    Base class to define baselines

    A Baseline is composed by the triplet :any:`bob.bio.base.preprocessor.Preprocessor`,
    :any:`bob.bio.base.extractor.Extractor` and :any:`bob.bio.base.algorithm.Algorithm`

    Attributes
    ----------

      name: str
        Name of the baseline. This name will be displayed in the command line interface
      preprocessors: dict
        Dictionary containing all possible preprocessors  
      extractor: str
        Registered resource or a config file containing the feature extractor
      algorithm: str
         Registered resource or a config file containing the algorithm

    """

    def __init__(self, name="", preprocessors=dict(), extractor="", algorithm="", **kwargs):
        super(Baseline, self).__init__(**kwargs)
        self.name = name
        self.preprocessors = preprocessors
        self.extractor = extractor
        self.algorithm = algorithm
