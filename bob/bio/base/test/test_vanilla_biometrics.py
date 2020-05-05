#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.pipelines import Sample, SampleSet, DelayedSample
import os
import numpy as np
import tempfile
from sklearn.pipeline import make_pipeline
from bob.bio.base.wrappers import wrap_bob_legacy
from bob.bio.base.test.test_transformers import (
    FakePreprocesor,
    FakeExtractor,
    FakeAlgorithm,
)
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
    BioAlgorithmCheckpointWrapper,
    dask_vanilla_biometrics,
    FourColumnsScoreWriter,
    CSVScoreWriter,
    BioAlgorithmLegacy,
)

import bob.pipelines as mario
import uuid
import shutil
import itertools


class DummyDatabase:
    def __init__(self, delayed=False, n_references=10, n_probes=10, dim=10, one_d=True):
        self.delayed = delayed
        self.dim = dim
        self.n_references = n_references
        self.n_probes = n_probes
        self.one_d = one_d
        self.gender_choices = ["M", "F"]
        self.metadata_1_choices = ["A", "B", "C"]

    def _create_random_1dsamples(self, n_samples, offset, dim):
        return [
            Sample(
                np.random.rand(dim),
                key=str(uuid.uuid4()),
                annotations=1,
                subject=str(i),
            )
            for i in range(offset, offset + n_samples)
        ]

    def _create_random_2dsamples(self, n_samples, offset, dim):
        return [
            Sample(
                np.random.rand(dim, dim),
                key=str(uuid.uuid4()),
                annotations=1,
                subject=str(i),
            )
            for i in range(offset, offset + n_samples)
        ]

    def _create_random_sample_set(self, n_sample_set=10, n_samples=2, seed=10):

        # Just generate random samples
        np.random.seed(seed)
        sample_set = [
            SampleSet(
                samples=[],
                key=str(i),
                subject=str(i),
                gender=np.random.choice(self.gender_choices),
                metadata_1=np.random.choice(self.metadata_1_choices),
            )
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
        samples = [sset.samples for sset in self._create_random_sample_set(seed=10)]
        return list(itertools.chain(*samples))

    def references(self):
        return self._create_random_sample_set(self.n_references, self.dim, seed=11)

    def probes(self):
        probes = []

        probes = self._create_random_sample_set(n_sample_set=10, n_samples=1, seed=12)
        for p in probes:
            p.references = [str(r) for r in list(range(self.n_references))]

        return probes

    def zprobes(self):
        zprobes = []

        zprobes = self._create_random_sample_set(n_sample_set=10, n_samples=1, seed=13)
        for p in zprobes:
            p.references = [str(r) for r in list(range(self.n_references))]

        return zprobes

    @property
    def allow_scoring_with_all_biometric_references(self):
        return True


def _make_transformer(dir_name):
    pipeline = make_pipeline(
        wrap_bob_legacy(
            FakePreprocesor(),
            dir_name,
            transform_extra_arguments=(("annotations", "annotations"),),
        ),
        wrap_bob_legacy(FakeExtractor(), dir_name,),
    )

    return pipeline


def _make_transformer_with_algorithm(dir_name):
    pipeline = make_pipeline(
        wrap_bob_legacy(
            FakePreprocesor(),
            dir_name,
            transform_extra_arguments=(("annotations", "annotations"),),
        ),
        wrap_bob_legacy(FakeExtractor(), dir_name),
        wrap_bob_legacy(FakeAlgorithm(), dir_name),
    )

    return pipeline


def test_on_memory():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(with_dask, allow_scoring_with_all_biometric_references):
            database = DummyDatabase()

            transformer = _make_transformer(dir_name)

            biometric_algorithm = Distance()

            vanilla_biometrics_pipeline = VanillaBiometricsPipeline(
                transformer, biometric_algorithm
            )

            if with_dask:
                vanilla_biometrics_pipeline = dask_vanilla_biometrics(
                    vanilla_biometrics_pipeline, npartitions=2
                )

            scores = vanilla_biometrics_pipeline(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
            )

            if with_dask:
                scores = scores.compute(scheduler="single-threaded")

            assert len(scores) == 10
            for sample_scores in scores:
                assert len(sample_scores) == 10
                for score in sample_scores:
                    assert isinstance(score.data, float)

        run_pipeline(False, True)
        run_pipeline(False, False)  # Testing checkpoint
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)
        run_pipeline(True, True)
        run_pipeline(True, True)  # Testing checkpoint


def test_checkpoint_bioalg_as_transformer():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(with_dask, score_writer=FourColumnsScoreWriter()):
            database = DummyDatabase()

            transformer = _make_transformer(dir_name)

            biometric_algorithm = BioAlgorithmCheckpointWrapper(
                Distance(), base_dir=dir_name, score_writer=score_writer
            )

            vanilla_biometrics_pipeline = VanillaBiometricsPipeline(
                transformer, biometric_algorithm
            )

            if with_dask:
                vanilla_biometrics_pipeline = dask_vanilla_biometrics(
                    vanilla_biometrics_pipeline, npartitions=2
                )

            scores = vanilla_biometrics_pipeline(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
            )

            if with_dask:
                scores = scores.compute(scheduler="single-threaded")

            if isinstance(score_writer, CSVScoreWriter):
                base_path = os.path.join(dir_name, "concatenated_scores")
                score_writer.concatenate_write_scores(scores, base_path)
                assert (
                    len(open(os.path.join(base_path, "chunk_0.csv")).readlines()) == 101
                )
            else:
                filename = os.path.join(dir_name, "concatenated_scores.txt")
                score_writer.concatenate_write_scores(scores, filename)
                assert len(open(filename).readlines()) == 100

        run_pipeline(False)
        run_pipeline(False)  # Checking if the checkpointng works
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Dask
        run_pipeline(True)
        run_pipeline(True)  # Checking if the checkpointng works
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # CSVWriter
        run_pipeline(False, CSVScoreWriter())
        run_pipeline(False, CSVScoreWriter())  # Checking if the checkpointng works
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # CSVWriter + Dask
        run_pipeline(True, CSVScoreWriter())
        run_pipeline(True, CSVScoreWriter())  # Checking if the checkpointng works


def test_checkpoint_bioalg_as_bioalg():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(with_dask, score_writer=FourColumnsScoreWriter()):
            database = DummyDatabase()

            transformer = _make_transformer_with_algorithm(dir_name)
            projector_file = transformer[2].estimator.estimator.projector_file

            biometric_algorithm = BioAlgorithmLegacy(
                FakeAlgorithm(),
                base_dir=dir_name,
                score_writer=score_writer,
                projector_file=projector_file,
            )

            vanilla_biometrics_pipeline = VanillaBiometricsPipeline(
                transformer, biometric_algorithm
            )

            if with_dask:
                vanilla_biometrics_pipeline = dask_vanilla_biometrics(
                    vanilla_biometrics_pipeline, npartitions=2
                )

            scores = vanilla_biometrics_pipeline(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
            )

            filename = os.path.join(dir_name, "concatenated_scores.txt")
            score_writer.concatenate_write_scores(scores, filename)

            if isinstance(score_writer, CSVScoreWriter):
                assert len(open(filename).readlines()) == 101
            else:
                assert len(open(filename).readlines()) == 100

        run_pipeline(False)
        run_pipeline(False)  # Checking if the checkpointng works
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Dask
        run_pipeline(True)
        run_pipeline(True)  # Checking if the checkpointng works
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)
