#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import glob
import itertools
import os
import shutil
import tempfile
import uuid

import numpy as np
import pytest

from click.testing import CliRunner
from h5py import File as HDF5File
from sklearn.pipeline import make_pipeline

from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines import (
    BioAlgCheckpointWrapper,
    CSVScoreWriter,
    FourColumnsScoreWriter,
    PipelineSimple,
    dask_bio_pipeline,
)
from bob.bio.base.script.pipeline_simple import (
    pipeline_simple as pipeline_simple_cli,
)
from bob.bio.base.wrappers import wrap_bob_legacy
from bob.io.base.testing_utils import assert_click_runner_result
from bob.pipelines import DelayedSample, Sample, SampleSet
from tests.test_transformers import FakeExtractor, FakePreprocessor


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
                template_id=str(i),
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
                template_id=str(i),
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
                template_id=str(i),
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

    def references(self, group=None):
        return self._create_random_sample_set(
            self.n_references, self.dim, seed=11
        )

    def probes(self, group=None):
        probes = []

        probes = self._create_random_sample_set(
            n_sample_set=10, n_samples=1, seed=12
        )
        for p in probes:
            p.references = [str(r) for r in list(range(self.n_references))]

        return probes

    def zprobes(self, group=None):
        zprobes = []

        zprobes = self._create_random_sample_set(
            n_sample_set=10, n_samples=1, seed=14
        )
        for p in zprobes:
            p.template_id = "z-" + str(p.template_id)
            p.references = [str(r) for r in list(range(self.n_references))]

        return zprobes

    def treferences(self):
        t_sset = self._create_random_sample_set(
            self.n_references, self.dim, seed=15
        )
        for t in t_sset:
            t.template_id = "t_" + str(t.template_id)
        return t_sset

    @property
    def score_all_vs_all(self):
        return True


def _custom_save_function(data, path):
    base_path, ext = path.rsplit(".", 1)
    filename = base_path + "_with_custom_f." + ext
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with HDF5File(filename, "w") as f:
        f["data"] = data


def _custom_load_function(path):
    base_path, ext = path.rsplit(".", 1)
    filename = base_path + "_with_custom_f." + ext
    assert os.path.isfile(filename)
    with HDF5File(filename, "r") as f:
        return f["data"][()]


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
            FakePreprocessor(),
            dir_name,
            transform_extra_arguments=(("annotations", "annotations"),),
        ),
        wrap_bob_legacy(
            FakeExtractor(),
            dir_name,
        ),
    )

    return pipeline


def test_on_memory():
    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(with_dask, score_all_vs_all):
            database = DummyDatabase()

            transformer = _make_transformer(dir_name)

            biometric_algorithm = Distance()

            pipeline_simple = PipelineSimple(
                transformer,
                biometric_algorithm,
                None,
            )

            if with_dask:
                pipeline_simple = dask_bio_pipeline(
                    pipeline_simple, npartitions=2
                )

            scores = pipeline_simple(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                score_all_vs_all=score_all_vs_all,
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

            biometric_algorithm = BioAlgCheckpointWrapper(
                Distance(), base_dir=dir_name
            )

            pipeline_simple = PipelineSimple(
                transformer, biometric_algorithm, score_writer
            )

            if with_dask:
                pipeline_simple = dask_bio_pipeline(
                    pipeline_simple, npartitions=2
                )

            scores = pipeline_simple(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                score_all_vs_all=database.score_all_vs_all,
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

            biometric_algorithm = BioAlgCheckpointWrapper(
                DistanceWithTags(), base_dir=dir_name
            )

            pipeline_simple = PipelineSimple(
                transformer, biometric_algorithm, score_writer
            )

            if with_dask:
                pipeline_simple = dask_bio_pipeline(
                    pipeline_simple, npartitions=2
                )

            scores = pipeline_simple(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                score_all_vs_all=database.score_all_vs_all,
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


def _run_with_failure(score_all_vs_all, sporadic_fail):
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
            score_all_vs_all=score_all_vs_all,
        )

        assert len(scores) == 10
        for sample_scores in scores:
            assert len(sample_scores) == 10
            for score in sample_scores:
                assert isinstance(score.data, float)


def test_database_sporadic_failure():
    _run_with_failure(False, sporadic_fail=True)
    _run_with_failure(True, sporadic_fail=True)


def test_database_full_failure():
    with pytest.raises(NotImplementedError):
        _run_with_failure(False, sporadic_fail=False)


def test_pipeline_simple_passthrough():
    """Ensure that PipelineSimple accepts a passthrough Estimator."""
    passthrough = make_pipeline(None)
    pipeline = PipelineSimple(passthrough, Distance())
    assert isinstance(pipeline, PipelineSimple)

    pipeline_with_passthrough = make_pipeline("passthrough")
    pipeline = PipelineSimple(pipeline_with_passthrough, Distance())
    assert isinstance(pipeline, PipelineSimple)
    db = DummyDatabase()
    scores = pipeline(
        db.background_model_samples(), db.references(), db.probes()
    )
    assert len(scores) == 10
    for sample_scores in scores:
        assert len(sample_scores) == 10
        for score in sample_scores:
            assert isinstance(score.data, float)


def _create_test_config(path):
    with open(path, "w") as f:
        f.write(
            """
from tests.test_pipeline_simple import DummyDatabase, _make_transformer
from bob.bio.base.pipelines import PipelineSimple
from bob.bio.base.algorithm import Distance

database = DummyDatabase()

transformer = _make_transformer(".")

biometric_algorithm = Distance()

pipeline = PipelineSimple(
    transformer,
    biometric_algorithm,
    None,
)
"""
        )


def _test_pipeline_click_cli(
    cli, options, expected_outputs=("results/scores-dev.csv",)
):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _create_test_config("config.py")
        result = runner.invoke(
            cli,
            [
                "-vv",
                "config.py",
            ]
            + options,
        )
        assert_click_runner_result(result)
        # check for expected_output
        output_files = glob.glob("results/**", recursive=True)
        err_msg = "Found only:\n{output_files}\nin output directory given the options: {options}".format(
            output_files="\n".join(output_files), options=options
        )
        for out in expected_outputs:
            assert os.path.isfile(out), err_msg


def test_pipeline_simple_click_cli():
    # open a click isolated environment

    for options in [
        ["--no-dask", "--memory"],
        ["--no-dask"],
        ["--memory"],
        [],
    ]:
        _test_pipeline_click_cli(pipeline_simple_cli, options)
