#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
Mixins to handle legacy components
"""

from bob.pipelines.mixins import CheckpointMixin, SampleMixin
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array
from bob.pipelines.sample import Sample, DelayedSample, SampleSet
import numpy
import logging
import os
logger = logging.getLogger(__name__)

def scikit_to_bob_supervised(X, Y):
    """
    Given an input data ready for :py:method:`scikit.estimator.BaseEstimator.fit`,
    convert for :py:class:`bob.bio.base.algorithm.Algorithm.train_projector` when 
    `performs_projection=True`
    """

    # TODO: THIS IS VERY INNEFICI
    logger.warning("INEFFICIENCY WARNING. HERE YOU ARE USING A HACK FOR USING BOB ALGORITHMS IN SCIKIT LEARN PIPELINES. \
                    WE RECOMMEND YOU TO PORT THIS ALGORITHM. DON'T BE LAZY :-)")

    bob_output = dict()
    for x,y in zip(X, Y):
        if y in bob_output:
            bob_output[y] = numpy.vstack((bob_output[y], x.data))
        else:
            bob_output[y] = x.data
    
    return [bob_output[k] for k in bob_output]

class LegacyProcessorMixin(TransformerMixin):
    """Class that wraps :py:class:`bob.bio.base.preprocessor.Preprocessor` and
    :py:class:`bob.bio.base.extractor.Extractors`


    Example
    -------

        Wrapping preprocessor with functtools
        >>> from bob.bio.base.mixins.legacy import LegacyProcessorMixin
        >>> from bob.bio.face.preprocessor import FaceCrop
        >>> import functools
        >>> transformer = LegacyProcessorMixin(functools.partial(FaceCrop, cropped_image_size=(10,10)))

    Example
    -------
        Wrapping extractor 
        >>> from bob.bio.base.mixins.legacy import LegacyProcessorMixin
        >>> from bob.bio.face.extractor import Linearize
        >>> transformer = LegacyProcessorMixin(Linearize)


    Parameters
    ----------
      callable: callable
         Calleble function that instantiates the scikit estimator

    """

    def __init__(self, callable=None, **kwargs):
        super().__init__(**kwargs)
        self.callable = callable
        self.instance = None

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):

        X = check_array(X, allow_nd=True)

        # Instantiates and do the "real" transform
        if self.instance is None:
            self.instance = self.callable()
        return [self.instance(x) for x in X]


from bob.pipelines.mixins import CheckpointMixin, SampleMixin
class LegacyAlgorithmMixin(CheckpointMixin,SampleMixin,BaseEstimator):
    """Class that wraps :py:class:`bob.bio.base.algorithm.Algoritm` and
    
    LegacyAlgorithmrMixin.fit maps :py:method:`bob.bio.base.algorithm.Algoritm.train_projector`

    LegacyAlgorithmrMixin.transform maps :py:method:`bob.bio.base.algorithm.Algoritm.project`

    THIS HAS TO BE SAMPABLE AND CHECKPOINTABLE


    Example
    -------

        Wrapping preprocessor with functtools
        >>> from bob.bio.base.mixins.legacy import LegacyProcessorMixin
        >>> from bob.bio.face.preprocessor import FaceCrop
        >>> import functools
        >>> transformer = LegacyProcessorMixin(functools.partial(FaceCrop, cropped_image_size=(10,10)))

    Example
    -------
        Wrapping extractor 
        >>> from bob.bio.base.mixins.legacy import LegacyProcessorMixin
        >>> from bob.bio.face.extractor import Linearize
        >>> transformer = LegacyProcessorMixin(Linearize)


    Parameters
    ----------
      callable: callable
         Calleble function that instantiates the scikit estimator

    """

    def __init__(self, callable=None, **kwargs):
        super().__init__(**kwargs)
        self.callable = callable
        self.instance = None
        self.projector_file = os.path.join(self.model_path, "Projector.hdf5")

    def fit(self, X, y=None, **fit_params):
        
        if os.path.exists(self.projector_file):
            return self

        # Instantiates and do the "real" fit
        if self.instance is None:
            self.instance = self.callable()

        if self.instance.performs_projection:
            # Organizing the date by class
            bob_X = scikit_to_bob_supervised(X, y)
            self.instance.train_projector(bob_X, self.projector_file)
        else:
            self.instance.train_projector(X, **fit_params)

        # Deleting the instance, so it's picklable
        self.instance = None

        return self

    def transform(self, X):

        if not isinstance(X, list):
            raise ValueError("It's expected a list, not %s" % type(X))

        # Instantiates and do the "real" transform
        if self.instance is None:
            self.instance = self.callable()
        self.instance.load_projector(self.projector_file)

        import ipdb; ipdb.set_trace()

        if isinstance(X[0], Sample) or isinstance(X[0], DelayedSample):
            #samples = []
            for s in X:
                projected_data = self.instance.project(s.data)
        
            #raw_X = [s.data for s in X]
        elif isinstance(X[0], SampleSet):

            sample_sets = []
            for sset in X:

                samples = []
                for sample in sset.samples:

                    # Project
                    projected_data = self.instance.project(sample.data)

                    #Checkpointing
                    path = self.make_path(sample)
                    self.instance.write_feature(path)

                    samples.append(DelayedSample())


                    pass
                    #bob.io.base.save(projected_data)




            #raw_X = [x.data for s in X for x in s.samples]
        else:
            raise ValueError("Type not allowed %s" % type(X[0]))


        return self.instance.project(raw_X)
