#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import itertools
import os
import shutil
import tempfile
import uuid

import numpy as np

from h5py import File as HDF5File
from nose.tools import raises
from sklearn.pipeline import make_pipeline

from bob.bio.base.pipelines import (
    BioAlgorithmCheckpointWrapper,
    BioAlgorithmLegacy,
    CSVScoreWriter,
    Distance,
    FourColumnsScoreWriter,
    PipelineSimple,
    dask_pipeline_simple,
)
from bob.bio.base.test.test_transformers import (
    FakeAlgorithm,
    FakeExtractor,
    FakePreprocesor,
)
from bob.bio.base.wrappers import wrap_bob_legacy
from bob.pipelines import DelayedSample, Sample, SampleSet


class DummyDatabase:
    def __init__(
        self,
        delayed=False,
        n_references=10,
        n_probes=10,
        dim=10,
        one_d=True,
        some_fta=False,
        all_fta=False,
    ):
        self.delayed = delayed
        self.dim = dim
        self.n_references = n_references
        self.n_probes = n_probes
        self.one_d = one_d
        self.gender_choices = ["M", "F"]
        self.metadata_1_choices = ["A", "B", "C"]
        self.contains_fta = some_fta
        self.all_samples_fta = all_fta

    def _create_random_1dsamples(self, n_samples, offset, dim):
        return [
            Sample(
                np.random.rand(dim),
                key=str(uuid.uuid4()),
                annotations=1,
                reference_id=str(i),
                subject_id=str(i),
            )
            for i in range(offset, offset + n_samples)
        ]

    def _create_random_2dsamples(self, n_samples, offset, dim):
        return [
            Sample(
                np.random.rand(dim, dim),
                key=str(uuid.uuid4()),
                annotations=1,
                reference_id=str(i),
                subject_id=str(i),
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
                reference_id=str(i),
                subject_id=str(i),
                gender=np.random.choice(self.gender_choices),
                metadata_1=np.random.choice(self.metadata_1_choices),
            )
            for i in range(n_sample_set)
        ]

        offset = 0
        for i, s in enumerate(sample_set):
            if self.one_d:
                s.samples = self._create_random_1dsamples(
                    n_samples, offset, self.dim
                )
            else:
                s.samples = self._create_random_2dsamples(
                    n_samples, offset, self.dim
                )
            if self.contains_fta and i % 2:
                for sample in s.samples[::2]:
                    sample.data = None
            if self.all_samples_fta:
                for sample in s.samples:
                    sample.data = None

            offset += n_samples
            pass

        return sample_set

    def background_model_samples(self):
        samples = [
            sset.samples for sset in self._create_random_sample_set(seed=10)
        ]
        return list(itertools.chain(*samples))

    def references(self):
        return self._create_random_sample_set(
            self.n_references, self.dim, seed=11
        )

    def probes(self):
        probes = []

        probes = self._create_random_sample_set(
            n_sample_set=10, n_samples=1, seed=12
        )
        for p in probes:
            p.references = [str(r) for r in list(range(self.n_references))]

        return probes

    def zprobes(self):
        zprobes = []

        zprobes = self._create_random_sample_set(
            n_sample_set=10, n_samples=1, seed=14
        )
        for p in zprobes:
            p.reference_id = "z-" + str(p.reference_id)
            p.references = [str(r) for r in list(range(self.n_references))]

        return zprobes

    def treferences(self):
        t_sset = self._create_random_sample_set(
            self.n_references, self.dim, seed=15
        )
        for t in t_sset:
            t.reference_id = "t_" + str(t.reference_id)
        return t_sset

    @property
    def allow_scoring_with_all_biometric_references(self):
        return True


def _custom_save_function(data, path):
    base_path, ext = path.rsplit(".", 1)
    filename = base_path + "_with_custom_f." + ext
    os.makedirs(os.path.dirname(path), exist_ok=True)
    HDF5File(filename, "w")["data"] = data


def _custom_load_function(path):
    base_path, ext = path.rsplit(".", 1)
    filename = base_path + "_with_custom_f." + ext
    assert os.path.isfile(filename)
    return HDF5File(filename, "r")["data"]


class DistanceWithTags(Distance):
    def _more_tags(self):
        return {
            "bob_enrolled_save_fn": _custom_save_function,
            "bob_enrolled_load_fn": _custom_load_function,
            "bob_enrolled_extension": ".hdf5",
        }


def _make_transformer(dir_name):
    pipeline = make_pipeline(
        wrap_bob_legacy(
            FakePreprocesor(),
            dir_name,
            transform_extra_arguments=(("annotations", "annotations"),),
        ),
        wrap_bob_legacy(
            FakeExtractor(),
            dir_name,
        ),
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
        wrap_bob_legacy(
            FakeAlgorithm(),
            dir_name,
            fit_extra_arguments=[("y", "reference_id")],
        ),
    )

    return pipeline


def test_on_memory():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(
            with_dask, allow_scoring_with_all_biometric_references
        ):
            database = DummyDatabase()

            transformer = _make_transformer(dir_name)

            biometric_algorithm = Distance()

            pipeline_simple = PipelineSimple(
                transformer,
                biometric_algorithm,
                None,
            )

            if with_dask:
                pipeline_simple = dask_pipeline_simple(
                    pipeline_simple, npartitions=2
                )

            scores = pipeline_simple(
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
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)
        run_pipeline(True, True)
        run_pipeline(True, True)  # Testing checkpoint


def test_checkpoint_bioalg_as_transformer():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(
            with_dask,
            score_writer=None,
        ):
            database = DummyDatabase()

            transformer = _make_transformer(dir_name)

            biometric_algorithm = BioAlgorithmCheckpointWrapper(
                Distance(), base_dir=dir_name
            )

            pipeline_simple = PipelineSimple(
                transformer, biometric_algorithm, score_writer
            )

            if with_dask:
                pipeline_simple = dask_pipeline_simple(
                    pipeline_simple, npartitions=2
                )

            scores = pipeline_simple(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
            )

            if pipeline_simple.score_writer is None:
                if with_dask:
                    scores = scores.compute(scheduler="single-threaded")

                assert len(scores) == 10
                for sset in scores:
                    if isinstance(sset[0], DelayedSample):
                        for s in sset:
                            assert len(s.data) == 10
                    else:
                        assert len(sset) == 10
            else:
                writed_scores = pipeline_simple.write_scores(scores)
                scores_dev_path = os.path.join(dir_name, "scores-dev")
                concatenated_scores = pipeline_simple.post_process(
                    writed_scores, scores_dev_path
                )

                if with_dask:
                    concatenated_scores = concatenated_scores.compute(
                        scheduler="single-threaded"
                    )

                if isinstance(
                    pipeline_simple.score_writer, FourColumnsScoreWriter
                ):
                    assert len(open(scores_dev_path).readlines()) == 100
                else:
                    assert (
                        len(open(scores_dev_path).readlines()) == 101
                    )  # Counting the header.

        run_pipeline(False)
        run_pipeline(False)  # Checking if the checkpointing works
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Writing scores
        run_pipeline(
            False,
            FourColumnsScoreWriter(os.path.join(dir_name, "final_scores")),
        )
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Dask
        run_pipeline(True)
        run_pipeline(True)  # Checking if the checkpointing works
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Writing scores
        run_pipeline(
            True, FourColumnsScoreWriter(os.path.join(dir_name, "final_scores"))
        )
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # CSVWriter
        run_pipeline(
            False, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )
        run_pipeline(
            False, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )  # Checking if the checkpointing works
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # CSVWriter + Dask
        run_pipeline(
            True, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )
        run_pipeline(
            True, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )  # Checking if the checkpointing works


def test_checkpoint_bioalg_with_tags():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(
            with_dask,
            score_writer=None,
        ):
            database = DummyDatabase()

            transformer = _make_transformer(dir_name)

            biometric_algorithm = BioAlgorithmCheckpointWrapper(
                DistanceWithTags(), base_dir=dir_name
            )

            pipeline_simple = PipelineSimple(
                transformer, biometric_algorithm, score_writer
            )

            if with_dask:
                pipeline_simple = dask_pipeline_simple(
                    pipeline_simple, npartitions=2
                )

            scores = pipeline_simple(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
            )

            written_scores = pipeline_simple.write_scores(scores)
            scores_dev_path = os.path.join(dir_name, "scores-dev")
            concatenated_scores = pipeline_simple.post_process(
                written_scores, scores_dev_path
            )

            if with_dask:
                concatenated_scores = concatenated_scores.compute(
                    scheduler="single-threaded"
                )

            assert os.path.isfile(
                dir_name
                + "/biometric_references/"
                + database.references()[0].key
                + "_with_custom_f.hdf5"
            )

        run_pipeline(False)
        run_pipeline(False)  # Checking if the checkpointing works
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Writing scores
        run_pipeline(
            False,
            FourColumnsScoreWriter(os.path.join(dir_name, "final_scores")),
        )
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Dask
        run_pipeline(True)
        run_pipeline(True)  # Checking if the checkpointing works
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Writing scores
        run_pipeline(
            True, FourColumnsScoreWriter(os.path.join(dir_name, "final_scores"))
        )
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # CSVWriter
        run_pipeline(
            False, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )
        run_pipeline(
            False, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )  # Checking if the checkpointing works
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # CSVWriter + Dask
        run_pipeline(
            True, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )
        run_pipeline(
            True, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )  # Checking if the checkpointing works


def test_checkpoint_bioalg_as_bioalg():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(
            with_dask, score_writer=FourColumnsScoreWriter(dir_name)
        ):
            database = DummyDatabase()

            transformer = _make_transformer_with_algorithm(dir_name)
            projector_file = transformer[2].estimator.estimator.projector_file

            biometric_algorithm = BioAlgorithmLegacy(
                FakeAlgorithm(),
                base_dir=dir_name,
                score_writer=score_writer,
                projector_file=projector_file,
            )

            pipeline_simple = PipelineSimple(transformer, biometric_algorithm)

            if with_dask:
                pipeline_simple = dask_pipeline_simple(
                    pipeline_simple, npartitions=2
                )

            scores = pipeline_simple(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
            )

            if pipeline_simple.score_writer is None:
                if with_dask:
                    scores = scores.compute(scheduler="single-threaded")

                assert len(scores) == 10
                for sset in scores:
                    if isinstance(sset[0], DelayedSample):
                        for s in sset:
                            assert len(s.data) == 10
                    else:
                        assert len(sset) == 10
            else:
                writed_scores = pipeline_simple.write_scores(scores)
                concatenated_scores = pipeline_simple.post_process(
                    writed_scores, os.path.join(dir_name, "scores-dev")
                )

                if with_dask:
                    concatenated_scores = concatenated_scores.compute(
                        scheduler="single-threaded"
                    )

                assert len(open(concatenated_scores).readlines()) == 100

        run_pipeline(False)
        run_pipeline(False)  # Checking if the checkpointing works
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # Dask
        run_pipeline(True)
        run_pipeline(True)  # Checking if the checkpointing works
        shutil.rmtree(
            dir_name
        )  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)


def _run_with_failure(
    allow_scoring_with_all_biometric_references, sporadic_fail
):

    with tempfile.TemporaryDirectory() as dir_name:

        database = DummyDatabase(
            some_fta=sporadic_fail, all_fta=not sporadic_fail
        )

        transformer = _make_transformer(dir_name)

        biometric_algorithm = Distance()

        pipeline_simple = PipelineSimple(
            transformer,
            biometric_algorithm,
            None,
        )

        scores = pipeline_simple(
            database.background_model_samples(),
            database.references(),
            database.probes(),
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        assert len(scores) == 10
        for sample_scores in scores:
            assert len(sample_scores) == 10
            for score in sample_scores:
                assert isinstance(score.data, float)


def test_database_sporadic_failure():
    _run_with_failure(False, sporadic_fail=True)
    _run_with_failure(True, sporadic_fail=True)


@raises(ValueError)
def test_database_full_failure():
    _run_with_failure(False, sporadic_fail=False)
