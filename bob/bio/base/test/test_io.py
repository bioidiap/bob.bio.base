#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 11 Dec 15:14:08 2013 CET
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the IO functionality of bob.bio.base.score."""

import tempfile

import numpy
import pkg_resources

from .. import score


def test_load_scores():
    # This function tests the IO functionality of loading score files in
    # different ways

    load_functions = {"4col": score.four_column, "5col": score.five_column}
    cols = {"4col": 4, "5col": 5}

    for variant in cols:
        normal_score_file = pkg_resources.resource_filename(
            "bob.bio.base.test", "data/dev-%s.txt" % variant
        )
        normal_scores = list(load_functions[variant](normal_score_file))

        assert len(normal_scores) == 910
        assert all(len(s) == cols[variant] for s in normal_scores)

        # read the compressed score file
        compressed_score_file = pkg_resources.resource_filename(
            "bob.bio.base.test", "data/dev-%s.tar.gz" % variant
        )
        compressed_scores = list(load_functions[variant](compressed_score_file))

        assert len(compressed_scores) == len(normal_scores)
        assert all(len(c) == cols[variant] for c in compressed_scores)
        assert all(
            c[i] == s[i]
            for c, s in zip(compressed_scores, normal_scores)
            for i in range(cols[variant])
        )

        # Use auto-estimated score file contents
        # read score file in normal way
        normal_scores = list(score.scores(normal_score_file))

        assert len(normal_scores) == 910
        assert all(len(s) == cols[variant] for s in normal_scores)

        # read the compressed score file
        compressed_scores = list(score.scores(compressed_score_file))

        assert len(compressed_scores) == len(normal_scores)
        assert all(len(c) == cols[variant] for c in compressed_scores)
        assert all(
            c[i] == s[i]
            for c, s in zip(compressed_scores, normal_scores)
            for i in range(cols[variant])
        )


def test_split_vuln_scores():
    """Tests that vulnerability score files are loaded correctly"""
    score_file = pkg_resources.resource_filename(
        "bob.bio.base.test", "data/vuln/scores-dev.csv"
    )
    split_scores = score.split_csv_vuln(score_file)
    assert all(
        key in split_scores for key in ("licit_neg", "licit_pos", "spoof")
    )
    assert all(
        isinstance(scores, numpy.ndarray) for scores in split_scores.values()
    )
    assert all(len(scores) == 5000 for scores in split_scores.values())


def test_split_scores():
    # This function tests the IO functionality of loading score files in
    # different ways

    split_functions = {
        "4col": score.split_four_column,
        "5col": score.split_five_column,
    }
    cols = {"4col": 4, "5col": 5}

    for variant in cols:
        # read score file in normal way
        normal_score_file = pkg_resources.resource_filename(
            "bob.bio.base.test", "data/dev-%s.txt" % variant
        )
        negatives, positives = split_functions[variant](normal_score_file)

        assert len(negatives) == 520, len(negatives)
        assert len(positives) == 390, len(positives)

        # read the compressed score file
        compressed_score_file = pkg_resources.resource_filename(
            "bob.bio.base.test", "data/dev-%s.tar.gz" % variant
        )
        negatives, positives = split_functions[variant](compressed_score_file)

        assert len(negatives) == 520, len(negatives)
        assert len(positives) == 390, len(positives)

        # Use auto-estimated score file contents
        # read score file in normal way
        negatives, positives = score.split(normal_score_file)

        assert len(negatives) == 520, len(negatives)
        assert len(positives) == 390, len(positives)

        # read the compressed score file
        negatives, positives = score.split(compressed_score_file)

        assert len(negatives) == 520, len(negatives)
        assert len(positives) == 390, len(positives)


def test_load_score():
    # This function tests the IO functionality of loading score files in
    # different ways

    cols = {"4col": 4, "5col": 5}

    for variant in cols:
        compressed_score_file = pkg_resources.resource_filename(
            "bob.bio.base.test", "data/dev-%s.tar.gz" % variant
        )
        normal_score_file = pkg_resources.resource_filename(
            "bob.bio.base.test", "data/dev-%s.txt" % variant
        )
        normal_scores = score.load_score(normal_score_file, cols[variant])

        assert len(normal_scores) == 910
        assert len(normal_scores.dtype) == cols[variant]

        # read the compressed score file
        compressed_score_file = pkg_resources.resource_filename(
            "bob.bio.base.test", "data/dev-%s.tar.gz" % variant
        )
        compressed_scores = score.load_score(
            compressed_score_file, cols[variant]
        )

        assert len(compressed_scores) == len(normal_scores)
        assert len(compressed_scores.dtype) == cols[variant]
        for name in normal_scores.dtype.names:
            assert all(normal_scores[name] == compressed_scores[name])

        # test minimal loading
        minimal_scores = score.load_score(normal_score_file, minimal=True)
        assert len(minimal_scores) == 910
        assert len(minimal_scores.dtype) == 3
        assert minimal_scores.dtype.names == ("claimed_id", "real_id", "score")


def test_dump_score():
    # This function tests the IO functionality of dumping score files

    cols = {"4col": 4, "5col": 5}

    for variant in cols:
        # read score file
        normal_score_file = pkg_resources.resource_filename(
            "bob.bio.base.test", "data/dev-%s.txt" % variant
        )
        normal_scores = score.load_score(normal_score_file, cols[variant])

        with tempfile.TemporaryFile() as f:
            score.dump_score(f, normal_scores)
            f.seek(0)
            loaded_scores = score.load_score(f, cols[variant])

        for name in normal_scores.dtype.names:
            assert all(normal_scores[name] == loaded_scores[name])


def _check_binary_identical(name1, name2):
    # see: http://www.peterbe.com/plog/using-md5-to-check-equality-between-files
    from hashlib import md5

    # tests if two files are binary identical
    with open(name1, "rb") as f1:
        with open(name2, "rb") as f2:
            assert md5(f1.read()).digest() == md5(f2.read()).digest()
