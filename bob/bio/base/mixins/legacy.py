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
from bob.pipelines.utils import is_picklable
import numpy
import logging
import os
import bob.io.base
import functools
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
    """Class that wraps :py:class:`bob.bio.base.algorithm.Algoritm`
    
    :py:method:`LegacyAlgorithmrMixin.fit` maps to :py:method:`bob.bio.base.algorithm.Algoritm.train_projector`

    :py:method:`LegacyAlgorithmrMixin.transform` maps :py:method:`bob.bio.base.algorithm.Algoritm.project`

    .. warning THIS HAS TO BE SAMPABLE AND CHECKPOINTABLE


    Example
    -------

        Wrapping LDA algorithm with functtools
        >>> from bob.bio.base.mixins.legacy import LegacyAlgorithmMixin
        >>> from bob.bio.base.algorithm import LDA
        >>> import functools
        >>> transformer = LegacyAlgorithmMixin(functools.partial(LDA, use_pinv=True, pca_subspace_dimension=0.90))



    Parameters
    ----------
      callable: callable
         Calleble function that instantiates the scikit estimator

    """

    def __init__(self, callable=None, **kwargs):
        super().__init__(**kwargs)
        self.callable = callable
        self.instance = None        
        self.projector_file = None


    def fit(self, X, y=None, **fit_params):
        
        self.projector_file = os.path.join(self.model_path, "Projector.hdf5")
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

        def _project_save_sample(sample):
            # Project
            projected_data = self.instance.project(sample.data)

            #Checkpointing
            path = self.make_path(sample)
            bob.io.base.create_directories_safe(os.path.dirname(path))
            f = bob.io.base.HDF5File(path, "w")

            self.instance.write_feature(projected_data, f)
            reader = self._get_reader(self.instance.read_feature, path)

            return DelayedSample(reader, parent=sample)

        self.projector_file = os.path.join(self.model_path, "Projector.hdf5")
        if not isinstance(X, list):
            raise ValueError("It's expected a list, not %s" % type(X))

        # Instantiates and do the "real" transform
        if self.instance is None:
            self.instance = self.callable()
        self.instance.load_projector(self.projector_file)

        if isinstance(X[0], Sample) or isinstance(X[0], DelayedSample):
            samples = []
            for sample in X:
                samples.append(_project_save_sample(sample))
            return samples

        elif isinstance(X[0], SampleSet):
            # Projecting and checkpointing sampleset
            sample_sets = []
            for sset in X:
                samples = []
                for sample in sset.samples:
                    samples.append(_project_save_sample(sample))
                sample_sets.append(SampleSet(samples=samples, parent=sset))
            return sample_sets

        else:
            raise ValueError("Type not allowed %s" % type(X[0]))


    def _get_reader(self, reader, path):
        if(is_picklable(self.instance.read_feature)):
            return functools.partial(reader, path)
        else:
            logger.warning(
                        f"The method {reader} is not picklable. Shiping its unbounded method to `DelayedSample`."
                    )
            reader = reader.__func__  # The reader object might not be picklable
            return functools.partial(reader, None, path)

