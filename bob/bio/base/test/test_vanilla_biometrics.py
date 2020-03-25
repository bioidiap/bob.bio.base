#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import os
import numpy
import tempfile
from sklearn.utils.validation import check_is_fitted


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


from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import Distance
import itertools
def test_distance_comparator():

    n_references = 10
    dim = 10
    n_probes = 10
    database = DummyDatabase(delayed=False, n_references=n_references, n_probes=n_probes, dim=10, one_d = True)
    references = database.references()
    probes = database.probes()

    comparator = Distance()
    references = comparator.enroll_samples(references)
    assert len(references)== n_references
    assert references[0].data.shape == (dim,)

    probes = database.probes()
    scores = comparator.score_samples(probes, references)
    scores = list(itertools.chain(*scores))

    assert len(scores) == n_probes*n_references
    assert len(scores[0].samples)==n_references
