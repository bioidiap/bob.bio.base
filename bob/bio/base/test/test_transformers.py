#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.preprocessor import Preprocessor
from bob.bio.base.extractor import Extractor
from bob.bio.base.algorithm import Algorithm
from bob.bio.base.transformers import (
    PreprocessorTransformer,
    ExtractorTransformer,
    AlgorithmTransformer,
)
import bob.pipelines as mario
import numpy as np
import tempfile
import os
import bob.io.base
from bob.bio.base.wrappers import (
    wrap_preprocessor,
    wrap_extractor,
    wrap_algorithm,
    wrap_transform_bob,
)
from sklearn.pipeline import make_pipeline


class _FakePreprocesor(Preprocessor):
    def __call__(self, data, annotations=None):
        return data + annotations


class _FakeExtractor(Extractor):
    def __call__(self, data, metadata=None):
        return data.flatten()


class _FakeExtractorFittable(Extractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_training = True
        self.model = None

    def __call__(self, data, metadata=None):
        return data @ self.model

    def train(self, training_data, extractor_file):
        self.model = training_data
        bob.io.base.save(self.model, extractor_file)


class _FakeAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_training = True
        self.split_training_features_by_client = True
        self.model = None

    def project(self, data, metadata=None):
        return data + self.model

    def train_projector(self, training_features, projector_file):
        self.model = np.sum(np.vstack(training_features), axis=0)
        bob.io.base.save(self.model, projector_file)

    def load_projector(self, projector_file):
        self.model = bob.io.base.load(projector_file)


def generate_samples(n_subjects, n_samples_per_subject, shape=(2, 2), annotations=1):
    """
    Simple sample generator that generates a certain number of samples per
    subject, whose data is np.zeros + subject index
    """

    samples = []
    for i in range(n_subjects):
        data = np.zeros(shape) + i
        for j in range(n_samples_per_subject):
            samples += [
                mario.Sample(
                    data,
                    subject=str(i),
                    key=str(i * n_subjects + j),
                    annotations=annotations,
                )
            ]
    return samples


def assert_sample(transformed_sample, oracle):
    return np.alltrue(
        [np.allclose(ts.data, o) for ts, o in zip(transformed_sample, oracle)]
    )


def assert_checkpoints(transformed_sample, dir_name):
    return np.alltrue(
        [
            os.path.exists(os.path.join(dir_name, ts.key + ".h5"))
            for ts in transformed_sample
        ]
    )


def test_preprocessor():

    preprocessor = _FakePreprocesor()
    preprocessor_transformer = PreprocessorTransformer(preprocessor)

    # Testing sample
    transform_extra_arguments = [("annotations", "annotations")]
    sample_transformer = mario.SampleWrapper(
        preprocessor_transformer, transform_extra_arguments
    )

    data = np.zeros((2, 2))
    oracle = [np.ones((2, 2))]
    annotations = 1
    sample = [mario.Sample(data, key="1", annotations=annotations)]
    transformed_sample = sample_transformer.transform(sample)

    assert assert_sample(transformed_sample, oracle)

    # Testing checkpoint
    with tempfile.TemporaryDirectory() as dir_name:
        checkpointing_transformer = mario.CheckpointWrapper(
            sample_transformer,
            features_dir=dir_name,
            load_func=preprocessor.read_data,
            save_func=preprocessor.write_data,
        )
        transformed_sample = checkpointing_transformer.transform(sample)

        assert assert_sample(transformed_sample, oracle)
        assert assert_checkpoints(transformed_sample, dir_name)


def test_extractor():

    extractor = _FakeExtractor()
    extractor_transformer = ExtractorTransformer(extractor)

    # Testing sample
    sample_transformer = mario.SampleWrapper(extractor_transformer)

    data = np.zeros((2, 2))
    oracle = [np.zeros((1, 4))]
    sample = [mario.Sample(data, key="1")]
    transformed_sample = sample_transformer.transform(sample)

    assert assert_sample(transformed_sample, oracle)

    # Testing checkpoint
    with tempfile.TemporaryDirectory() as dir_name:
        checkpointing_transformer = mario.CheckpointWrapper(
            sample_transformer,
            features_dir=dir_name,
            load_func=extractor.read_feature,
            save_func=extractor.write_feature,
        )
        transformed_sample = checkpointing_transformer.transform(sample)

        assert assert_sample(transformed_sample, oracle)
        assert assert_checkpoints(transformed_sample, dir_name)


def test_extractor_fittable():

    with tempfile.TemporaryDirectory() as dir_name:

        extractor_file = os.path.join(dir_name, "Extractor.hdf5")
        extractor = _FakeExtractorFittable()
        extractor_transformer = ExtractorTransformer(
            extractor, model_path=extractor_file
        )

        # Testing sample
        sample_transformer = mario.SampleWrapper(extractor_transformer)

        # Fitting
        training_data = np.arange(4).reshape(2, 2)
        training_samples = [mario.Sample(training_data, key="1")]
        sample_transformer = sample_transformer.fit(training_samples)

        test_data = [np.zeros((2, 2)), np.ones((2, 2))]
        oracle = [np.zeros((2, 2)), np.ones((2, 2)) @ training_data]
        test_sample = [mario.Sample(d, key=str(i)) for i, d in enumerate(test_data)]

        transformed_sample = sample_transformer.transform(test_sample)
        assert assert_sample(transformed_sample, oracle)

        # Testing checkpoint
        checkpointing_transformer = mario.CheckpointWrapper(
            sample_transformer,
            features_dir=dir_name,
            load_func=extractor.read_feature,
            save_func=extractor.write_feature,
        )
        transformed_sample = checkpointing_transformer.transform(test_sample)
        assert assert_sample(transformed_sample, oracle)
        assert assert_checkpoints(transformed_sample, dir_name)


def test_algorithm():

    with tempfile.TemporaryDirectory() as dir_name:

        projector_file = os.path.join(dir_name, "Projector.hdf5")
        projector_pkl = os.path.join(dir_name, "Projector.pkl")  # Testing pickling

        algorithm = _FakeAlgorithm()
        algorithm_transformer = AlgorithmTransformer(
            algorithm, projector_file=projector_file
        )

        # Testing sample
        fit_extra_arguments = [("y", "subject")]
        sample_transformer = mario.SampleWrapper(
            algorithm_transformer, fit_extra_arguments=fit_extra_arguments
        )

        n_subjects = 2
        n_samples_per_subject = 2
        shape = (2, 2)
        training_samples = generate_samples(
            n_subjects, n_samples_per_subject, shape=shape
        )
        sample_transformer = sample_transformer.fit(training_samples)

        oracle = np.zeros(shape) + n_subjects
        test_sample = generate_samples(1, 1)
        transformed_sample = sample_transformer.transform(test_sample)
        assert assert_sample(transformed_sample, [oracle])
        assert os.path.exists(projector_file)

        # Testing checkpoint
        checkpointing_transformer = mario.CheckpointWrapper(
            sample_transformer,
            features_dir=dir_name,
            load_func=algorithm.read_feature,
            save_func=algorithm.write_feature,
            model_path=projector_pkl,
        )
        # Fitting again to assert if it loads again
        checkpointing_transformer = checkpointing_transformer.fit(training_samples)
        transformed_sample = checkpointing_transformer.transform(test_sample)

        # Fitting again
        assert assert_sample(transformed_sample, oracle)
        transformed_sample = checkpointing_transformer.transform(test_sample)
        assert assert_checkpoints(transformed_sample, dir_name)
        assert os.path.exists(projector_pkl)


def test_wrap_bob_pipeline():

    def run_pipeline(with_dask):
        with tempfile.TemporaryDirectory() as dir_name:

            pipeline = make_pipeline(
                wrap_transform_bob(
                    _FakePreprocesor(),
                    dir_name,
                    transform_extra_arguments=(("annotations", "annotations"),),
                ),
                wrap_transform_bob(_FakeExtractor(), dir_name,),
                wrap_transform_bob(
                    _FakeAlgorithm(), dir_name, fit_extra_arguments=(("y", "subject"),)
                ),
            )
            oracle = [7.0, 7.0, 7.0, 7.0]
            training_samples = generate_samples(n_subjects=2, n_samples_per_subject=2)
            test_samples = generate_samples(n_subjects=1, n_samples_per_subject=1)
            if with_dask:                
                pipeline = mario.wrap(["dask"], pipeline)
                transformed_samples = (
                    pipeline.fit(training_samples).transform(test_samples).compute()
                )
            else:
                transformed_samples = pipeline.fit(training_samples).transform(
                    test_samples
                )
            assert assert_sample(transformed_samples, oracle)

    run_pipeline(False)
    run_pipeline(True)
