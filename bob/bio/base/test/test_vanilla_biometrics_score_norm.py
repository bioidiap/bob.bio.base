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
from bob.bio.base.test.test_vanilla_biometrics import DummyDatabase, _make_transformer


from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
    ZTNormPipeline,
    ZTNormDaskWrapper,
    ZTNormCheckpointWrapper,
    BioAlgorithmCheckpointWrapper,
    dask_vanilla_biometrics,
    BioAlgorithmLegacy,
    CSVScoreWriter,
)

import bob.pipelines as mario
import uuid
import shutil
import itertools
from scipy.spatial.distance import cdist, euclidean
from sklearn.preprocessing import FunctionTransformer
import copy


def zt_norm_stubs(references, probes, t_references, z_probes):
    def _norm(scores, norm_base_scores, axis=1):
        mu = np.mean(norm_base_scores, axis=axis)

        # old = True
        # if old:
        #    std = np.std(norm_base_scores, axis=axis)
        #    if axis == 1:
        #        return ((scores.T - mu) / std).T
        #    else:
        #        return (scores - mu) / std

        if axis == 1:
            std = np.sqrt(
                np.sum(
                    (
                        norm_base_scores
                        - np.tile(
                            mu.reshape(norm_base_scores.shape[0], 1),
                            (1, norm_base_scores.shape[1]),
                        )
                    )
                    ** 2,
                    axis=1,
                )
                / (norm_base_scores.shape[1] - 1)
            )

            return (
                scores
                - np.tile(
                    mu.reshape(norm_base_scores.shape[0], 1), (1, scores.shape[1])
                )
            ) / np.tile(std.reshape(norm_base_scores.shape[0], 1), (1, scores.shape[1]))
        else:

            std = np.sqrt(
                np.sum(
                    (
                        norm_base_scores
                        - np.tile(
                            mu.reshape(1, norm_base_scores.shape[1]),
                            (norm_base_scores.shape[0], 1),
                        )
                    )
                    ** 2,
                    axis=0,
                )
                / (norm_base_scores.shape[0] - 1)
            )

            return (
                scores
                - np.tile(
                    mu.reshape(1, norm_base_scores.shape[1]), (scores.shape[0], 1)
                )
            ) / np.tile(std.reshape(1, norm_base_scores.shape[1]), (scores.shape[0], 1))

    n_reference = references.shape[0]
    n_probes = probes.shape[0]
    n_t_references = t_references.shape[0]
    n_z_probes = z_probes.shape[0]

    raw_scores = cdist(references, probes)

    z_scores = cdist(references, z_probes)
    # Computing the statistics of Z-Probes for each biometric reference
    # https://arxiv.org/pdf/1709.09868.pdf --> below eq (2) first eq
    z_normed_scores = _norm(raw_scores, z_scores, axis=1)
    assert z_normed_scores.shape == (n_reference, n_probes)

    t_scores = cdist(t_references, probes)
    # Computing the statistics of T-Models for each probe
    # https://arxiv.org/pdf/1709.09868.pdf --> below eq (2) second eq
    t_normed_scores = _norm(raw_scores, t_scores, axis=0)
    assert t_normed_scores.shape == (n_reference, n_probes)
    assert t_scores.shape == (n_t_references, n_probes)

    ZxT_scores = cdist(t_references, z_probes)
    assert ZxT_scores.shape == (n_t_references, n_z_probes)
    # Computing the statistics of T-Models for each z probe
    # https://arxiv.org/pdf/1709.09868.pdf --> below eq (2) third eq
    z_t_scores = _norm(t_scores, ZxT_scores, axis=1)
    assert z_t_scores.shape == (n_t_references, n_probes)

    # FINALLY DOING THE F*****G ZT-NORM
    zt_normed_scores = _norm(z_normed_scores, z_t_scores, axis=0)
    assert zt_normed_scores.shape == (n_reference, n_probes)

    s_normed_scores = (z_normed_scores + t_normed_scores) * 0.5
    assert s_normed_scores.shape == (n_reference, n_probes)

    return (
        raw_scores,
        z_normed_scores,
        t_normed_scores,
        zt_normed_scores,
        s_normed_scores,
    )


def test_norm_mechanics():
    def _create_sample_sets(raw_data, offset, references=None):
        if references is None:
            return [
                SampleSet(
                    [Sample(s, reference_id=str(i + offset), key=str(uuid.uuid4()))],
                    key=str(i + offset),
                    reference_id=str(i + offset),
                    subject_id=str(i + offset),
                )
                for i, s in enumerate(raw_data)
            ]
        else:
            return [
                SampleSet(
                    [Sample(s, reference_id=str(i + offset), key=str(uuid.uuid4()))],
                    key=str(i + offset),
                    reference_id=str(i + offset),
                    subject_id=str(i + offset),
                    references=references,
                )
                for i, s in enumerate(raw_data)
            ]

    def _do_nothing_fn(x):
        return x

    def _dump_scores_from_samples(scores, shape):
        # We have to transpose because the tests are BIOMETRIC_REFERENCES vs PROBES
        # and bob.bio.base is PROBES vs BIOMETRIC_REFERENCES
        return np.array([s.data for sset in scores for s in sset]).reshape(shape).T

    with tempfile.TemporaryDirectory() as dir_name:

        def run(with_dask, with_checkpoint=False):
            ############
            # Prepating stubs
            ############
            n_references = 2
            n_probes = 3
            n_t_references = 4
            n_z_probes = 5
            dim = 5

            references = np.arange(n_references * dim).reshape(
                n_references, dim
            )  # two references (each row different identity)
            probes = (
                np.arange(n_probes * dim).reshape(n_probes, dim) * 10
            )  # three probes (each row different identity matching with references)

            t_references = np.arange(n_t_references * dim).reshape(
                n_t_references, dim
            )  # four T-REFERENCES (each row different identity)
            z_probes = (
                np.arange(n_z_probes * dim).reshape(n_z_probes, dim) * 10
            )  # five Z-PROBES (each row different identity matching with t references)

            (
                raw_scores_ref,
                z_normed_scores_ref,
                t_normed_scores_ref,
                zt_normed_scores_ref,
                s_normed_scores_ref,
            ) = zt_norm_stubs(references, probes, t_references, z_probes)

            ############
            # Preparing the samples
            ############

            # Creating enrollment samples
            biometric_reference_sample_sets = _create_sample_sets(references, offset=0)
            t_reference_sample_sets = _create_sample_sets(t_references, offset=300)

            # Fetching ids
            reference_ids = [r.reference_id for r in biometric_reference_sample_sets]
            t_reference_ids = [r.reference_id for r in t_reference_sample_sets]
            ids = reference_ids + t_reference_ids

            probe_sample_sets = _create_sample_sets(probes, offset=600, references=ids)
            z_probe_sample_sets = _create_sample_sets(
                z_probes, offset=900, references=ids
            )

            ############
            # TESTING REGULAR SCORING
            #############

            transformer = make_pipeline(FunctionTransformer(func=_do_nothing_fn))
            biometric_algorithm = Distance(euclidean, factor=1)

            if with_checkpoint:
                biometric_algorithm = BioAlgorithmCheckpointWrapper(
                    Distance(distance_function=euclidean, factor=1), dir_name,
                )

            vanilla_pipeline = VanillaBiometricsPipeline(
                transformer, biometric_algorithm, score_writer=None
            )
            if with_dask:
                vanilla_pipeline = dask_vanilla_biometrics(vanilla_pipeline)

            score_samples = vanilla_pipeline(
                [],
                biometric_reference_sample_sets,
                probe_sample_sets,
                allow_scoring_with_all_biometric_references=True,
            )

            if with_dask:
                score_samples = score_samples.compute(scheduler="single-threaded")

            raw_scores = _dump_scores_from_samples(
                score_samples, shape=(n_probes, n_references)
            )

            assert np.allclose(raw_scores, raw_scores_ref)

            ############
            # TESTING Z-NORM
            #############

            z_vanilla_pipeline = ZTNormPipeline(
                vanilla_pipeline, z_norm=True, t_norm=False
            )

            if with_checkpoint:
                z_vanilla_pipeline.ztnorm_solver = ZTNormCheckpointWrapper(
                    z_vanilla_pipeline.ztnorm_solver, dir_name
                )

            if with_dask:
                z_vanilla_pipeline.ztnorm_solver = ZTNormDaskWrapper(
                    z_vanilla_pipeline.ztnorm_solver
                )

            z_normed_score_samples = z_vanilla_pipeline(
                [],
                biometric_reference_sample_sets,
                copy.deepcopy(probe_sample_sets),
                z_probe_sample_sets,
                t_reference_sample_sets,
            )

            if with_dask:
                z_normed_score_samples = z_normed_score_samples.compute(
                    scheduler="single-threaded"
                )

            z_normed_scores = _dump_scores_from_samples(
                z_normed_score_samples, shape=(n_probes, n_references)
            )
            np.testing.assert_allclose(z_normed_scores, z_normed_scores_ref)

            ############
            # TESTING T-NORM
            #############

            t_vanilla_pipeline = ZTNormPipeline(
                vanilla_pipeline, z_norm=False, t_norm=True,
            )

            if with_checkpoint:
                t_vanilla_pipeline.ztnorm_solver = ZTNormCheckpointWrapper(
                    t_vanilla_pipeline.ztnorm_solver, dir_name
                )

            if with_dask:
                t_vanilla_pipeline.ztnorm_solver = ZTNormDaskWrapper(
                    t_vanilla_pipeline.ztnorm_solver
                )

            t_normed_score_samples = t_vanilla_pipeline(
                [],
                biometric_reference_sample_sets,
                copy.deepcopy(probe_sample_sets),
                z_probe_sample_sets,
                t_reference_sample_sets,
            )

            if with_dask:
                t_normed_score_samples = t_normed_score_samples.compute(
                    scheduler="single-threaded"
                )

            t_normed_scores = _dump_scores_from_samples(
                t_normed_score_samples, shape=(n_probes, n_references)
            )
            assert np.allclose(t_normed_scores, t_normed_scores_ref)

            ############
            # TESTING ZT-NORM
            #############
            zt_vanilla_pipeline = ZTNormPipeline(
                vanilla_pipeline, z_norm=True, t_norm=True,
            )

            if with_checkpoint:
                zt_vanilla_pipeline.ztnorm_solver = ZTNormCheckpointWrapper(
                    zt_vanilla_pipeline.ztnorm_solver, dir_name
                )

            if with_dask:
                zt_vanilla_pipeline.ztnorm_solver = ZTNormDaskWrapper(
                    zt_vanilla_pipeline.ztnorm_solver
                )

            (
                raw_score_samples,
                z_normed_score_samples,
                t_normed_score_samples,
                zt_normed_score_samples,
                s_normed_score_samples,
            ) = zt_vanilla_pipeline(
                [],
                biometric_reference_sample_sets,
                copy.deepcopy(probe_sample_sets),
                z_probe_sample_sets,
                t_reference_sample_sets,
            )

            if with_dask:
                raw_score_samples = raw_score_samples.compute(
                    scheduler="single-threaded"
                )
                z_normed_score_samples = z_normed_score_samples.compute(
                    scheduler="single-threaded"
                )
                t_normed_score_samples = t_normed_score_samples.compute(
                    scheduler="single-threaded"
                )
                zt_normed_score_samples = zt_normed_score_samples.compute(
                    scheduler="single-threaded"
                )

                s_normed_score_samples = s_normed_score_samples.compute(
                    scheduler="single-threaded"
                )

            raw_scores = _dump_scores_from_samples(
                raw_score_samples, shape=(n_probes, n_references)
            )
            assert np.allclose(raw_scores, raw_scores_ref)

            z_normed_scores = _dump_scores_from_samples(
                z_normed_score_samples, shape=(n_probes, n_references)
            )
            assert np.allclose(t_normed_scores, t_normed_scores_ref)

            t_normed_scores = _dump_scores_from_samples(
                t_normed_score_samples, shape=(n_probes, n_references)
            )
            assert np.allclose(t_normed_scores, t_normed_scores_ref)

            zt_normed_scores = _dump_scores_from_samples(
                zt_normed_score_samples, shape=(n_probes, n_references)
            )
            assert np.allclose(zt_normed_scores, zt_normed_scores_ref)

            s_normed_scores = _dump_scores_from_samples(
                s_normed_score_samples, shape=(n_probes, n_references)
            )
            assert np.allclose(s_normed_scores, s_normed_scores_ref)

    # No dask
    run(False)  # On memory

    # With checkpoing
    run(False, with_checkpoint=True)
    run(False, with_checkpoint=True)
    shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
    os.makedirs(dir_name, exist_ok=True)

    # With dask
    run(True)  # On memory
    run(True, with_checkpoint=True)
    run(True, with_checkpoint=True)


def test_znorm_on_memory():

    with tempfile.TemporaryDirectory() as dir_name:

        def run_pipeline(with_dask, score_writer=None):

            database = DummyDatabase(one_d=False)

            transformer = _make_transformer(dir_name)

            biometric_algorithm = Distance()

            vanilla_biometrics_pipeline = ZTNormPipeline(
                VanillaBiometricsPipeline(
                    transformer, biometric_algorithm, score_writer
                )
            )

            if with_dask:
                vanilla_biometrics_pipeline = dask_vanilla_biometrics(
                    vanilla_biometrics_pipeline, npartitions=2
                )

            (
                raw_scores,
                z_scores,
                t_scores,
                zt_scores,
                s_scores,
            ) = vanilla_biometrics_pipeline(
                database.background_model_samples(),
                database.references(),
                database.probes(),
                database.zprobes(),
                database.treferences(),
                allow_scoring_with_all_biometric_references=database.allow_scoring_with_all_biometric_references,
            )

            def _concatenate(pipeline, scores, path):
                writed_scores = pipeline.write_scores(scores)
                concatenated_scores = pipeline.post_process(writed_scores, path)
                return concatenated_scores

            if isinstance(score_writer, CSVScoreWriter):
                raw_scores = _concatenate(
                    vanilla_biometrics_pipeline,
                    raw_scores,
                    os.path.join(dir_name, "scores-dev", "raw_scores"),
                )
                z_scores = _concatenate(
                    vanilla_biometrics_pipeline,
                    z_scores,
                    os.path.join(dir_name, "scores-dev", "z_scores"),
                )
                t_scores = _concatenate(
                    vanilla_biometrics_pipeline,
                    t_scores,
                    os.path.join(dir_name, "scores-dev", "t_scores"),
                )

                zt_scores = _concatenate(
                    vanilla_biometrics_pipeline,
                    zt_scores,
                    os.path.join(dir_name, "scores-dev", "zt_scores"),
                )

                s_scores = _concatenate(
                    vanilla_biometrics_pipeline,
                    s_scores,
                    os.path.join(dir_name, "scores-dev", "s_scores"),
                )

            if with_dask:
                raw_scores = raw_scores.compute(scheduler="single-threaded")
                z_scores = z_scores.compute(scheduler="single-threaded")
                t_scores = t_scores.compute(scheduler="single-threaded")
                zt_scores = zt_scores.compute(scheduler="single-threaded")
                s_scores = s_scores.compute(scheduler="single-threaded")

            if isinstance(score_writer, CSVScoreWriter):

                assert (
                    len(
                        open(
                            os.path.join(dir_name, "scores-dev", "raw_scores"), "r"
                        ).readlines()
                    )
                    == 101
                )
                assert (
                    len(
                        open(
                            os.path.join(dir_name, "scores-dev", "z_scores"), "r"
                        ).readlines()
                    )
                    == 101
                )
                assert (
                    len(
                        open(
                            os.path.join(dir_name, "scores-dev", "t_scores"), "r"
                        ).readlines()
                    )
                    == 101
                )
                assert (
                    len(
                        open(
                            os.path.join(dir_name, "scores-dev", "zt_scores"), "r"
                        ).readlines()
                    )
                    == 101
                )
                assert (
                    len(
                        open(
                            os.path.join(dir_name, "scores-dev", "s_scores"), "r"
                        ).readlines()
                    )
                    == 101
                )

            else:
                assert len(raw_scores) == 10
                assert len(z_scores) == 10
                assert len(t_scores) == 10
                assert len(zt_scores) == 10
                assert len(s_scores) == 10

        run_pipeline(False)
        run_pipeline(False)  # Testing checkpoint
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        run_pipeline(
            False, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        # With DASK
        run_pipeline(True)
        run_pipeline(True)  # Testing checkpoint
        shutil.rmtree(dir_name)  # Deleting the cache so it runs again from scratch
        os.makedirs(dir_name, exist_ok=True)

        run_pipeline(
            True, CSVScoreWriter(os.path.join(dir_name, "concatenated_scores"))
        )
