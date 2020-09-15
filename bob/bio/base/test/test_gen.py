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

from bob.bio.base.script.gen import gen, gen_score_distr

import logging
logger = logging.getLogger(__name__)

def test_gen():
    """
    Tests that the main gen function works as expected
    """
    temp_path = "./gen_test_temp_dir/"
    n_subjects = 5
    n_probes_per_subject = 5
    n_unknown_subjects = 2
    n_pos = 10
    n_neg = 60
    n_unk = 20

    logger.info("Calling 'gen' with a specific amount of scores.")
    gen.callback(outdir=temp_path, mean_match=10, mean_non_match=-10,
        n_probes_per_subject=n_probes_per_subject, n_subjects=n_subjects,
        sigma_positive=1, sigma_negative=1,
        n_unknown_subjects=n_unknown_subjects,
        five_col=False,
        force_count=True, n_pos=n_pos, n_neg=n_neg, n_unk=n_unk,
    )
    assert os.path.exists(os.path.join(temp_path,"scores-dev")), "dev scores file not created."
    line_count=0
    with open(os.path.join(temp_path,"scores-dev")) as f:
        for l in f: line_count += 1
    assert line_count == n_pos + n_neg + n_unk
    assert os.path.exists(os.path.join(temp_path,"scores-eval")), "eval scores file not created."
    line_count=0
    with open(os.path.join(temp_path,"scores-eval")) as f:
        for l in f: line_count += 1
    assert line_count == n_pos + n_neg + n_unk


    n_subjects = 5
    n_probes_per_subject = 5
    n_unknown_subjects = 2
    n_pos = n_subjects*n_probes_per_subject
    n_neg = n_subjects*(n_subjects-1)*n_probes_per_subject
    n_unk = n_unknown_subjects*n_subjects*n_probes_per_subject

    logger.info("Calling 'gen' without a specific amount.")
    gen.callback(outdir=temp_path, mean_match=10, mean_non_match=-10,
        n_probes_per_subject=n_probes_per_subject, n_subjects=n_subjects,
        sigma_positive=1, sigma_negative=1,
        n_unknown_subjects=n_unknown_subjects,
        five_col=False,
        force_count=False, n_pos=0, n_neg=0, n_unk=0,
    )
    assert os.path.exists(os.path.join(temp_path,"scores-dev")), "dev scores file not created."
    line_count=0
    with open(os.path.join(temp_path,"scores-dev")) as f:
        for l in f: line_count += 1
    assert line_count == n_pos + n_neg + n_unk
    assert os.path.exists(os.path.join(temp_path,"scores-eval")), "eval scores file not created."
    line_count=0
    with open(os.path.join(temp_path,"scores-eval")) as f:
        for l in f: line_count += 1
    assert line_count == n_pos + n_neg + n_unk


    n_subjects = 5
    n_probes_per_subject = 2
    n_unknown_subjects = 0
    n_pos = n_subjects*n_probes_per_subject
    n_neg = n_subjects*(n_subjects-1)*n_probes_per_subject
    n_unk = n_unknown_subjects*n_subjects*n_probes_per_subject

    logger.info("Calling 'gen' without unknown subjects.")
    gen.callback(outdir=temp_path, mean_match=10, mean_non_match=-10,
        n_probes_per_subject=n_probes_per_subject, n_subjects=n_subjects,
        sigma_positive=1, sigma_negative=1,
        n_unknown_subjects=n_unknown_subjects,
        five_col=False,
        force_count=False, n_pos=0, n_neg=0, n_unk=0,
    )
    assert os.path.exists(os.path.join(temp_path,"scores-dev")), "dev scores file not created."
    line_count=0
    with open(os.path.join(temp_path,"scores-dev")) as f:
        for l in f: line_count += 1
    assert line_count == n_pos + n_neg + n_unk
    assert os.path.exists(os.path.join(temp_path,"scores-eval")), "eval scores file not created."
    line_count=0
    with open(os.path.join(temp_path,"scores-eval")) as f:
        for l in f: line_count += 1
    assert line_count == n_pos + n_neg + n_unk


    n_subjects = 0
    n_probes_per_subject = 2
    n_unknown_subjects = 0
    n_pos = n_subjects*n_probes_per_subject
    n_neg = n_subjects*(n_subjects-1)*n_probes_per_subject
    n_unk = n_unknown_subjects*n_subjects*n_probes_per_subject

    logger.info("Calling 'gen' with no subjects.")
    gen.callback(outdir=temp_path, mean_match=10, mean_non_match=-10,
        n_probes_per_subject=n_probes_per_subject, n_subjects=n_subjects,
        sigma_positive=1, sigma_negative=1,
        n_unknown_subjects=n_unknown_subjects,
        five_col=False,
        force_count=False, n_pos=0, n_neg=0, n_unk=0,
    )
    assert os.path.exists(os.path.join(temp_path,"scores-dev")), "dev scores file not created."
    line_count=0
    with open(os.path.join(temp_path,"scores-dev")) as f:
        for l in f: line_count += 1
    assert line_count == n_pos + n_neg + n_unk
    assert os.path.exists(os.path.join(temp_path,"scores-eval")), "eval scores file not created."
    line_count=0
    with open(os.path.join(temp_path,"scores-eval")) as f:
        for l in f: line_count += 1
    assert line_count == n_pos + n_neg + n_unk

    # Cleanup
    os.remove(os.path.join(temp_path, "scores-dev"))
    os.remove(os.path.join(temp_path, "scores-eval"))
    os.rmdir(temp_path)


def test_gen_score_dist():
    """
    Tests that the scores generation works as expected
    """
    neg, pos = gen_score_distr(mean_neg=-10, mean_pos=10, sigma_neg=1, sigma_pos=1, n_neg=20, n_pos=20, seed=0)
    assert len(neg) == 20, f"Incorrect number of negative scores generated ({len(neg)})"
    assert len(pos) == 20, f"Incorrect number of positive scores generated ({len(pos)})"
    assert all([isinstance(s, (numpy.floating, float)) for s in neg]), "A score was not a float"
    assert all([isinstance(s, (numpy.floating, float)) for s in pos]), "A score was not a float"
    print("expected_neg =",neg)
    print("expected_pos =",pos)
    expected_neg = numpy.array([-10.678141, -8.689173, -8.84452, -10.922123, -8.790776, -9.634045, -8.546636, -8.738061, -9.4082985, -8.313588, -10.227924, -11.022893, -9.453528, -11.756885, -10.05081, -11.754785, -10.395921, -10.7985, -9.478285, -10.597143])
    expected_pos = numpy.array([11.876446, 10.971403, 9.413583, 8.965617, 10.740386, 11.194529, 12.115288, 10.136266, 9.341219, 10.259532, 9.0935755, 10.0329485, 9.644332, 10.493712, 11.016409, 9.788092, 11.062946, 9.049674, 8.171294, 10.826373])
    assert numpy.allclose(neg, expected_neg), "Unexpected score generated"
    assert numpy.allclose(pos, expected_pos), "Unexpected score generated"
