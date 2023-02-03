#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
import tempfile

import numpy as np

import bob.io.base

from bob.bio.base.extractor import Extractor
from bob.bio.base.preprocessor import Preprocessor
from bob.bio.base.transformers import (
    ExtractorTransformer,
    PreprocessorTransformer,
)
from bob.pipelines import CheckpointWrapper, Sample, SampleWrapper


class FakePreprocessor(Preprocessor):
    def __call__(self, data, annotations=None):
        return data + annotations


class FakeExtractor(Extractor):
    def __call__(self, data):
        return data.flatten()[0:10]  # Selecting the first 10 features


class FakeExtractorFittable(Extractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_training = True
        self.model = None

    def __call__(self, data, metadata=None):
        model = self.model
        return data @ model

    def train(self, training_data, extractor_file):
        self.model = np.vstack(training_data)
        bob.io.base.save(self.model, extractor_file)


def generate_samples(
    n_subjects, n_samples_per_subject, shape=(2, 2), annotations=1
):
    """
    Simple sample generator that generates a certain number of samples per
    subject, whose data is np.zeros + subject index
    """

    samples = []
    for i in range(n_subjects):
        data = np.zeros(shape) + i
        for j in range(n_samples_per_subject):
            samples += [
                Sample(
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
    preprocessor = FakePreprocessor()
    preprocessor_transformer = PreprocessorTransformer(preprocessor)

    # Testing sample
    transform_extra_arguments = [("annotations", "annotations")]
    sample_transformer = SampleWrapper(
        preprocessor_transformer, transform_extra_arguments
    )

    data = np.zeros((2, 2))
    oracle = [np.ones((2, 2))]
    annotations = 1
    sample = [Sample(data, key="1", annotations=annotations)]
    transformed_sample = sample_transformer.transform(sample)

    assert assert_sample(transformed_sample, oracle)

    # Testing checkpoint
    with tempfile.TemporaryDirectory() as dir_name:
        checkpointing_transformer = CheckpointWrapper(
            sample_transformer,
            features_dir=dir_name,
            load_func=preprocessor.read_data,
            save_func=preprocessor.write_data,
        )
        transformed_sample = checkpointing_transformer.transform(sample)

        assert assert_sample(transformed_sample, oracle)
        assert assert_checkpoints(transformed_sample, dir_name)


def test_extractor():
    extractor = FakeExtractor()
    extractor_transformer = ExtractorTransformer(extractor)

    # Testing sample
    sample_transformer = SampleWrapper(extractor_transformer)

    data = np.zeros((2, 2))
    oracle = [np.zeros((1, 4))]
    sample = [Sample(data, key="1")]
    transformed_sample = sample_transformer.transform(sample)

    assert assert_sample(transformed_sample, oracle)

    # Testing checkpoint
    with tempfile.TemporaryDirectory() as dir_name:
        checkpointing_transformer = CheckpointWrapper(
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
        extractor = FakeExtractorFittable()
        extractor_transformer = ExtractorTransformer(
            extractor, model_path=extractor_file
        )

        # Testing sample
        sample_transformer = SampleWrapper(extractor_transformer)
        # Fitting
        training_data = np.arange(4).reshape(2, 2)
        training_samples = [Sample(training_data, key="1")]
        sample_transformer = sample_transformer.fit(training_samples)

        test_data = [np.zeros((2, 2)), np.ones((2, 2))]
        oracle = [np.zeros((2, 2)), np.ones((2, 2)) @ training_data]
        test_sample = [Sample(d, key=str(i)) for i, d in enumerate(test_data)]

        transformed_sample = sample_transformer.transform(test_sample)
        assert assert_sample(transformed_sample, oracle)

        # Testing checkpoint
        checkpointing_transformer = CheckpointWrapper(
            sample_transformer,
            features_dir=dir_name,
            load_func=extractor.read_feature,
            save_func=extractor.write_feature,
        )
        transformed_sample = checkpointing_transformer.transform(test_sample)
        assert assert_sample(transformed_sample, oracle)
        assert assert_checkpoints(transformed_sample, dir_name)
