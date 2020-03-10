#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import os
import numpy
import tempfile
from sklearn.utils.validation import check_is_fitted


#from bob.bio.base.processor import Linearize, SampleLinearize, CheckpointSampleLinearize


class DummyDatabase:

    def __init__(self, delayed=False, n_references=10, n_probes=10, dim=10, one_d = True):
        self.delayed = delayed
        self.dim = dim
        self.n_references = n_references
        self.n_probes = n_probes
        self.one_d = one_d


    def _create_random_1dsamples(self, n_samples, offset, dim):        
        return [ Sample(numpy.random.rand(dim), key=i) for i in range(offset,offset+n_samples) ]

    def _create_random_2dsamples(self, n_samples, offset, dim):        
        return [ Sample(numpy.random.rand(dim, dim), key=i) for i in range(offset,offset+n_samples) ]

    def _create_random_sample_set(self, n_sample_set=10, n_samples=2):

        # Just generate random samples
        sample_set = [SampleSet(samples=[], key=i) for i in range(n_sample_set)]

        offset = 0        
        for s in sample_set:
            if self.one_d:
                s.samples = self._create_random_1dsamples(n_samples, offset, self.dim)
            else:
                s.samples = self._create_random_2dsamples(n_samples, offset, self.dim)

            offset += n_samples
            pass

        return sample_set


    def background_model_samples(self):
        return self._create_random_sample_set()


    def references(self):
        return self._create_random_sample_set(self.n_references, self.dim)


    def probes(self):
        probes = self._create_random_sample_set(self.n_probes, self.dim)
        for p in probes:
            p.references = list(range(self.n_references))
        return probes


from bob.bio.base.pipelines.vanilla_biometrics.comparator import DistanceComparator
def test_distance_comparator():
    
    n_references = 10
    dim = 10
    n_probes = 10
    database = DummyDatabase(delayed=False, n_references=n_references, n_probes=n_probes, dim=10, one_d = True)
    references = database.references()    
    probes = database.probes()

    pass

    comparator = DistanceComparator()
    references = comparator.enroll_samples(references)
    assert len(references)== n_references
    assert references[0].data.shape == (dim,)

    probes = database.probes()
    scores = comparator.score_samples(probes, references)
    
    assert len(scores) == n_probes*n_references
    assert len(scores[0].samples)==n_references
    
    

    ## Test the transformer only
    #transformer = Linearize()
    #X = numpy.zeros(shape=(10,10))
    #X_tr = transformer.transform(X)
    #assert X_tr.shape == (100,)


    ## Test wrapped in to a Sample
    #sample = Sample(X, key="1")
    #transformer = SampleLinearize()
    #X_tr = transformer.transform([sample])
    #assert X_tr[0].data.shape == (100,)

    ## Test checkpoint    
    #with tempfile.TemporaryDirectory() as d:
        #transformer = CheckpointSampleLinearize(features_dir=d)
        #X_tr =  transformer.transform([sample])
        #assert X_tr[0].data.shape == (100,)
        #assert os.path.exists(os.path.join(d, "1.h5"))


