#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yannick Dayer <yannick.dayer@idiap.ch>
# Mon 14 Sep 2020 17:00:41 UTC+02

"""Tests for the bob.bio.base.script.gen module

The gen module generates synthetic scores and saves them to a file for
demonstration and test purpose.
"""

import os
import numpy


from click.testing import CliRunner
from bob.extension.scripts.click_helper import assert_click_runner_result

from bob.bio.base.script.gen import gen, gen_score_distr

import logging

logger = logging.getLogger(__name__)
logger.setLevel(
    "DEBUG"
)  # If NOTSET (default), will be changed to ERROR at CliRunner.invoke


def test_gen():
    """
    Tests that the main gen command works as expected
    """
    # Define a click runner to invoke click commands
    runner = CliRunner()

    with runner.isolated_filesystem():

        temp_path = "./gen_test_temp_dir/"
        n_subjects = 5
        n_probes_per_subject = 5
        n_unknown_subjects = 2
        n_pos = 10
        n_neg = 60
        n_unk = 20

        logger.info("Calling 'gen' with a specific amount of scores.")
        result = runner.invoke(
            gen,
            args=[
                "-mm",
                "10",
                "-mnm",
                "-10",
                "-sp",
                "1",
                "-sn",
                "1",
                "-p",
                f"{n_probes_per_subject}",
                "-s",
                f"{n_subjects}",
                "-u",
                f"{n_unknown_subjects}",
                "-f",
                "--n-pos",
                f"{n_pos}",
                "--n-neg",
                f"{n_neg}",
                "--n-unk",
                f"{n_unk}",
                f"{temp_path}",
            ],
        )
        assert_click_runner_result(result)
        assert os.path.exists(
            os.path.join(temp_path, "scores-dev")
        ), "dev scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-dev")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk
        assert os.path.exists(
            os.path.join(temp_path, "scores-eval")
        ), "eval scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-eval")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk

        n_subjects = 5
        n_probes_per_subject = 5
        n_unknown_subjects = 2
        n_pos = n_subjects * n_probes_per_subject
        n_neg = n_subjects * (n_subjects - 1) * n_probes_per_subject
        n_unk = n_unknown_subjects * n_subjects * n_probes_per_subject

        logger.info("Calling 'gen' without a specific amount.")
        result = runner.invoke(
            gen,
            args=[
                "-mm",
                "10",
                "-mnm",
                "-10",
                "-sp",
                "1",
                "-sn",
                "1",
                "-p",
                f"{n_probes_per_subject}",
                "-s",
                f"{n_subjects}",
                "-u",
                f"{n_unknown_subjects}",
                f"{temp_path}",
            ],
        )
        assert_click_runner_result(result)
        assert os.path.exists(
            os.path.join(temp_path, "scores-dev")
        ), "dev scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-dev")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk
        assert os.path.exists(
            os.path.join(temp_path, "scores-eval")
        ), "eval scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-eval")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk

        n_subjects = 5
        n_probes_per_subject = 2
        n_unknown_subjects = 0
        n_pos = n_subjects * n_probes_per_subject
        n_neg = n_subjects * (n_subjects - 1) * n_probes_per_subject
        n_unk = n_unknown_subjects * n_subjects * n_probes_per_subject

        logger.info("Calling 'gen' without unknown subjects.")
        result = runner.invoke(
            gen,
            args=[
                "-mm",
                "10",
                "-mnm",
                "-10",
                "-sp",
                "1",
                "-sn",
                "1",
                "-p",
                f"{n_probes_per_subject}",
                "-s",
                f"{n_subjects}",
                "-u",
                f"{n_unknown_subjects}",
                f"{temp_path}",
            ],
        )
        assert_click_runner_result(result)
        assert os.path.exists(
            os.path.join(temp_path, "scores-dev")
        ), "dev scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-dev")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk
        assert os.path.exists(
            os.path.join(temp_path, "scores-eval")
        ), "eval scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-eval")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk

        n_subjects = 0
        n_probes_per_subject = 2
        n_unknown_subjects = 0
        n_pos = n_subjects * n_probes_per_subject
        n_neg = n_subjects * (n_subjects - 1) * n_probes_per_subject
        n_unk = n_unknown_subjects * n_subjects * n_probes_per_subject

        logger.info("Calling 'gen' with no subjects.")
        result = runner.invoke(
            gen,
            args=[
                "-mm",
                "10",
                "-mnm",
                "-10",
                "-sp",
                "1",
                "-sn",
                "1",
                "-p",
                f"{n_probes_per_subject}",
                "-s",
                f"{n_subjects}",
                "-u",
                f"{n_unknown_subjects}",
                f"{temp_path}",
            ],
        )
        assert_click_runner_result(result)
        assert os.path.exists(
            os.path.join(temp_path, "scores-dev")
        ), "dev scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-dev")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk
        assert os.path.exists(
            os.path.join(temp_path, "scores-eval")
        ), "eval scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-eval")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk

        n_subjects = 5
        n_probes_per_subject = 0
        n_unknown_subjects = 2
        n_pos = n_subjects * n_probes_per_subject
        n_neg = n_subjects * (n_subjects - 1) * n_probes_per_subject
        n_unk = n_unknown_subjects * n_subjects * n_probes_per_subject

        logger.info("Calling 'gen' with no probes.")
        result = runner.invoke(
            gen,
            args=[
                "-mm",
                "10",
                "-mnm",
                "-10",
                "-sp",
                "1",
                "-sn",
                "1",
                "-p",
                f"{n_probes_per_subject}",
                "-s",
                f"{n_subjects}",
                "-u",
                f"{n_unknown_subjects}",
                f"{temp_path}",
            ],
        )
        assert_click_runner_result(result)
        assert os.path.exists(
            os.path.join(temp_path, "scores-dev")
        ), "dev scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-dev")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk
        assert os.path.exists(
            os.path.join(temp_path, "scores-eval")
        ), "eval scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-eval")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk

        n_subjects = 5
        n_probes_per_subject = 0
        n_unknown_subjects = 2
        n_pos = n_subjects * n_probes_per_subject
        n_neg = n_subjects * (n_subjects - 1) * n_probes_per_subject
        n_unk = n_unknown_subjects * n_subjects * n_probes_per_subject

        logger.info("Calling 'gen' with only unknowns.")
        result = runner.invoke(
            gen,
            args=[
                "-mm",
                "10",
                "-mnm",
                "-10",
                "-sp",
                "1",
                "-sn",
                "1",
                "-p",
                f"{n_probes_per_subject}",
                "-s",
                f"{n_subjects}",
                "-u",
                f"{n_unknown_subjects}",
                f"{temp_path}",
            ],
        )
        assert_click_runner_result(result)
        assert os.path.exists(
            os.path.join(temp_path, "scores-dev")
        ), "dev scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-dev")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk
        assert os.path.exists(
            os.path.join(temp_path, "scores-eval")
        ), "eval scores file not created."
        line_count = 0
        with open(os.path.join(temp_path, "scores-eval")) as f:
            for l in f:
                line_count += 1
        assert line_count == n_pos + n_neg + n_unk


def test_gen_score_dist():
    """
    Tests that the scores generation works as expected
    """
    neg, pos = gen_score_distr(
        mean_neg=-10, mean_pos=10, sigma_neg=1, sigma_pos=1, n_neg=20, n_pos=20, seed=0
    )
    assert len(neg) == 20, f"Incorrect number of negative scores generated ({len(neg)})"
    assert len(pos) == 20, f"Incorrect number of positive scores generated ({len(pos)})"
    assert all(
        [isinstance(s, (numpy.floating, float)) for s in neg]
    ), "A score was not a float"
    assert all(
        [isinstance(s, (numpy.floating, float)) for s in pos]
    ), "A score was not a float"
    expected_neg = numpy.array(
        [
            -8.23594765,
            -9.59984279,
            -9.02126202,
            -7.7591068,
            -8.13244201,
            -10.97727788,
            -9.04991158,
            -10.15135721,
            -10.10321885,
            -9.5894015,
            -9.85595643,
            -8.54572649,
            -9.23896227,
            -9.87832498,
            -9.55613677,
            -9.66632567,
            -8.50592093,
            -10.20515826,
            -9.6869323,
            -10.85409574,
        ]
    )
    expected_pos = numpy.array(
        [
            7.44701018,
            10.6536186,
            10.8644362,
            9.25783498,
            12.26975462,
            8.54563433,
            10.04575852,
            9.81281615,
            11.53277921,
            11.46935877,
            10.15494743,
            10.37816252,
            9.11221425,
            8.01920353,
            9.65208785,
            10.15634897,
            11.23029068,
            11.20237985,
            9.61267318,
            9.69769725,
        ]
    )
    assert numpy.allclose(neg, expected_neg), "Unexpected score generated"
    assert numpy.allclose(pos, expected_pos), "Unexpected score generated"
