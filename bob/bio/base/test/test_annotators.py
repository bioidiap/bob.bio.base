import os
import shutil
import tempfile

from click.testing import CliRunner

from bob.bio.base.annotator import Callable, FailSafe
from bob.bio.base.script.annotate import annotate, annotate_samples
from bob.bio.base.utils.annotations import read_annotation_file
from bob.extension.scripts.click_helper import assert_click_runner_result


def test_annotate():

    try:
        tmp_dir = tempfile.mkdtemp(prefix="bobtest_")
        runner = CliRunner()
        result = runner.invoke(
            annotate,
            args=("-d", "dummy", "-g", "dev", "-a", "dummy", "-o", tmp_dir),
        )
        assert_click_runner_result(result)

        # test if annotations exist
        for dirpath, _, filenames in os.walk(tmp_dir):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                annot = read_annotation_file(path, "json")
                assert annot["topleft"] == [0, 0]
                # size of atnt images
                assert annot["bottomright"] == [112, 92]
    finally:
        shutil.rmtree(tmp_dir)


def test_annotate_samples():

    try:
        tmp_dir = tempfile.mkdtemp(prefix="bobtest_")
        runner = CliRunner()
        result = runner.invoke(
            annotate_samples,
            args=("dummy_samples", "-a", "dummy", "-o", tmp_dir),
        )
        assert_click_runner_result(result)

        # test if annotations exist
        for dirpath, dirnames, filenames in os.walk(tmp_dir):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                annot = read_annotation_file(path, "json")
                assert annot["topleft"] == [0, 0]
                # size of atnt images
                assert annot["bottomright"] == [112, 92]
    finally:
        shutil.rmtree(tmp_dir)


def dummy_extra_key_annotator(data_batch, **kwargs):
    return [{"leye": 0, "reye": 0, "topleft": 0}]


def test_failsafe():
    annotator = FailSafe(
        [Callable(dummy_extra_key_annotator)], ["leye", "reye"]
    )
    annotations = annotator([1])
    assert all(key in annotations[0] for key in ["leye", "reye", "topleft"])

    annotator = FailSafe(
        [Callable(dummy_extra_key_annotator)], ["leye", "reye"], True
    )
    annotations = annotator([1])
    assert all(key in annotations[0] for key in ["leye", "reye"])
    assert all(key not in annotations[0] for key in ["topleft"])
