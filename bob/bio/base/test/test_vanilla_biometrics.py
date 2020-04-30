#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.pipelines import Sample, SampleSet, DelayedSample
import os
import numpy
import tempfile
from sklearn.pipeline import make_pipeline
from bob.bio.base.wrappers import wrap_transform_bob
from bob.bio.base.test.test_transformers import FakePreprocesor, FakeExtractor
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)
from bob.bio.base.pipelines.vanilla_biometrics import (
    BioAlgorithmCheckpointWrapper,
    FourColumnsScoreWriter,
)
import uuid


class DummyDatabase:
    def __init__(self, delayed=False, n_references=10, n_probes=10, dim=10, one_d=True):
        self.delayed = delayed
        self.dim = dim
        self.n_references = n_references
        self.n_probes = n_probes
        self.one_d = one_d

    def _create_random_1dsamples(self, n_samples, offset, dim):
        return [
            Sample(numpy.random.rand(dim), key=str(uuid.uuid4()), annotations=1)
            for i in range(offset, offset + n_samples)
        ]

    def _create_random_2dsamples(self, n_samples, offset, dim):
        return [
            Sample(numpy.random.rand(dim, dim), key=str(uuid.uuid4()), annotations=1)
            for i in range(offset, offset + n_samples)
        ]

    def _create_random_sample_set(self, n_sample_set=10, n_samples=2):

        # Just generate random samples
        sample_set = [
            SampleSet(samples=[], key=str(i), subject=str(i))
            for i in range(n_sample_set)
        ]

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
        probes = []

        probes = self._create_random_sample_set(n_sample_set=10, n_samples=1)
        for p in probes:
            p.references = list(range(self.n_references))

        return probes

    @property
    def allow_scoring_with_all_biometric_references(self):
        return True


def _make_transformer(dir_name):
    return make_pipeline(
        wrap_transform_bob(
            FakePreprocesor(),
            dir_name,
            transform_extra_arguments=(("annotations", "annotations"),),
        ),
        wrap_transform_bob(FakeExtractor(), dir_name,),
    )


def test_on_memory():

    with tempfile.TemporaryDirectory() as dir_name:
        database = DummyDatabase()

        transformer = _make_transformer(dir_name)

        biometric_algorithm = Distance()

        biometric_pipeline = VanillaBiometricsPipeline(transformer, biometric_algorithm)

        scores = biometric_pipeline(
            database.background_model_samples(),
            database.references(),
            database.probes(),
            allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
        )

        assert len(scores) == 10
        for probe_ssets in scores:
            for probe in probe_ssets:                
                assert len(probe) == 10

def test_checkpoint():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(with_dask):
            database = DummyDatabase()

            transformer = _make_transformer(dir_name)

            biometric_algorithm = BioAlgorithmCheckpointWrapper(
                Distance(), base_dir=dir_name
            )

            biometric_pipeline = VanillaBiometricsPipeline(transformer, biometric_algorithm)

            scores = biometric_pipeline(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
            )

            filename = os.path.join(dir_name, "concatenated_scores.txt")
            FourColumnsScoreWriter().concatenate_write_scores(
                scores, filename
            )
            
            assert len(open(filename).readlines())==100

        run_pipeline(False)
        run_pipeline(False) # Checking if the checkpoints work

