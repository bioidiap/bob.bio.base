#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 19 Aug 2022 14:37:01 UTC+02

import glob
import os
import tempfile

import pytest

from click.testing import CliRunner
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from bob.bio.base.pipelines.entry_points import execute_pipeline_train
from bob.bio.base.script.pipeline_train import (
    pipeline_train as pipeline_train_cli,
)
from bob.bio.base.wrappers import wrap_bob_legacy
from bob.io.base.testing_utils import assert_click_runner_result
from bob.pipelines import wrap
from tests.test_pipeline_simple import DummyDatabase
from tests.test_transformers import FakeExtractor, FakePreprocessor


class FittableTransformer(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.fitted_count = 0

    def fit(self, X, y=None):
        self.fitted_count += 1
        return self

    def transform(self, X):
        return X + self.fitted_count

    def _more_tags(self):
        return {"requires_fit": True}


def _make_transformer(dir_name):
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                wrap_bob_legacy(
                    FakePreprocessor(),
                    dir_name,
                    transform_extra_arguments=(("annotations", "annotations"),),
                ),
            ),
            (
                "extractor",
                wrap_bob_legacy(
                    FakeExtractor(),
                    dir_name,
                ),
            ),
            ("fittable_transformer", wrap(["sample"], FittableTransformer())),
        ]
    )

    return pipeline


def test_pipeline_train_function():
    with tempfile.TemporaryDirectory() as output:
        pipeline = _make_transformer(output)
        database = DummyDatabase()
        execute_pipeline_train(pipeline, database, output=output)
        assert os.path.isfile(os.path.join(output, "fittable_transformer.pkl"))


def _create_test_config_pipeline_simple(path):
    with open(path, "w") as f:
        f.write(
            """
from tests.test_pipeline_train import DummyDatabase, _make_transformer
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


def _create_test_config_pipeline_sklearn(path):
    with open(path, "w") as f:
        f.write(
            """
from tests.test_pipeline_train import DummyDatabase, _make_transformer

database = DummyDatabase()

pipeline = _make_transformer(".")
"""
        )


@pytest.mark.parametrize(
    "options,pipeline_simple",
    [
        (["--no-dask", "--memory"], True),
        (["--no-dask", "--memory"], False),
        (["--no-dask"], True),
        (["--no-dask"], False),
        (["--memory"], True),
        (["--memory"], False),
        ([], True),
        ([], False),
    ],
)
def test_pipeline_click_cli(
    options,
    pipeline_simple,
    expected_outputs=("results/fittable_transformer.pkl",),
):
    runner = CliRunner()
    with runner.isolated_filesystem():
        if pipeline_simple:
            _create_test_config_pipeline_simple("config.py")
        else:
            _create_test_config_pipeline_sklearn("config.py")
        result = runner.invoke(
            pipeline_train_cli,
            [
                "-vv",
                "config.py",
            ]
            + options,
        )
        assert_click_runner_result(result)
        # check for expected_output
        output_files = glob.glob("results/**", recursive=True)
        nl = "\n -"
        err_msg = f"Found only:\n- {nl.join(output_files)}\nin output directory given the options: {options}, and with {'PipelineSimple' if pipeline_simple else 'sklearn pipeline'}"
        for out in expected_outputs:
            assert os.path.isfile(out), err_msg
