"""
#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

Implementation of a pipeline and an algorithm that 
computes Z, T and ZT Score Normalization of a :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm`
"""

from bob.pipelines import (
    DelayedSample,
    Sample,
    SampleSet,
    DelayedSampleSet,
    DelayedSampleSetCached,
)


import numpy as np
import dask
import functools
import cloudpickle
import os
from .score_writers import FourColumnsScoreWriter
import copy
import logging
from .pipelines import check_valid_pipeline
from . import pickle_compress, uncompress_unpickle

logger = logging.getLogger(__name__)


class ZTNormPipeline(object):
    """
    Apply Z, T or ZT Score normalization on top of VanillaBiometric Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Example
    -------
       >>> from bob.pipelines.transformers import Linearize
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.pipelines.vanilla_biometrics import Distance, VanillaBiometricsPipeline, ZTNormPipeline
       >>> estimator_1 = Linearize()
       >>> transformer = make_pipeline(estimator_1)
       >>> biometric_algorithm = Distance()
       >>> vanilla_biometrics_pipeline = VanillaBiometricsPipeline(transformer, biometric_algorithm)
       >>> zt_pipeline = ZTNormPipeline(vanilla_biometrics_pipeline)
       >>> zt_pipeline(...) #doctest: +SKIP

    Parameters
    ----------

        vanilla_biometrics_pipeline: :any:`VanillaBiometricsPipeline`
          An instance :any:`VanillaBiometricsPipeline` to the wrapped with score normalization

        z_norm: bool
          If True, applies ZScore normalization on top of raw scores.

        t_norm: bool
          If True, applies TScore normalization on top of raw scores.
          If both, z_norm and t_norm are true, it applies score normalization

        score_writer: 

        adaptive_score_fraction: float
           Set the proportion of the impostor scores used to compute :math:`\mu` and :math:`\std` for the T normalization
           This is also called as adaptative T-Norm (https://ieeexplore.ieee.org/document/1415220) or 
           Top-Norm (https://ieeexplore.ieee.org/document/4013533)

        adaptive_score_descending_sort bool
            It true, during the Top-norm statistics computations, sort the scores in descending order


    """

    def __init__(
        self,
        vanilla_biometrics_pipeline,
        z_norm=True,
        t_norm=True,
        score_writer=FourColumnsScoreWriter("./scores.txt"),
        adaptive_score_fraction=1.0,
        adaptive_score_descending_sort=True,
    ):
        self.vanilla_biometrics_pipeline = vanilla_biometrics_pipeline
        self.biometric_algorithm = self.vanilla_biometrics_pipeline.biometric_algorithm
        self.transformer = self.vanilla_biometrics_pipeline.transformer

        self.ztnorm_solver = ZTNorm(
            adaptive_score_fraction, adaptive_score_descending_sort
        )

        self.z_norm = z_norm
        self.t_norm = t_norm
        self.score_writer = score_writer

        if not z_norm and not t_norm:
            raise ValueError(
                "Both z_norm and t_norm are False. No normalization will be applied"
            )
        check_valid_pipeline(self)

    def __call__(
        self,
        background_model_samples,
        biometric_reference_samples,
        probe_samples,
        zprobe_samples=None,
        t_biometric_reference_samples=None,
        allow_scoring_with_all_biometric_references=False,
    ):

        self.transformer = self.train_background_model(background_model_samples)

        # Create biometric samples
        biometric_references = self.create_biometric_reference(
            biometric_reference_samples
        )

        raw_scores, probe_features = self.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

        # Z NORM
        if self.z_norm:
            if zprobe_samples is None:
                raise ValueError("No samples for `z_norm` was provided")
            z_normed_scores, z_probe_features = self.compute_znorm_scores(
                zprobe_samples,
                raw_scores,
                biometric_references,
                allow_scoring_with_all_biometric_references,
            )

        if self.t_norm:
            if t_biometric_reference_samples is None:
                raise ValueError("No samples for `t_norm` was provided")
        else:
            # In case z_norm=True and t_norm=False
            return z_normed_scores

        # T NORM
        t_normed_scores, t_scores, t_biometric_references = self.compute_tnorm_scores(
            t_biometric_reference_samples,
            probe_features,
            raw_scores,
            allow_scoring_with_all_biometric_references,
        )

        if not self.z_norm:
            # In case z_norm=False and t_norm=True
            return t_normed_scores

        # ZT NORM
        zt_normed_scores = self.compute_ztnorm_scores(
            z_probe_features,
            t_biometric_references,
            z_normed_scores,
            t_scores,
            allow_scoring_with_all_biometric_references,
        )

        # S-norm
        s_normed_scores = self.compute_snorm_scores(z_normed_scores, t_normed_scores)

        return (
            raw_scores,
            z_normed_scores,
            t_normed_scores,
            zt_normed_scores,
            s_normed_scores,
        )

    def train_background_model(self, background_model_samples):
        return self.vanilla_biometrics_pipeline.train_background_model(
            background_model_samples
        )

    def create_biometric_reference(self, biometric_reference_samples):
        return self.vanilla_biometrics_pipeline.create_biometric_reference(
            biometric_reference_samples
        )

    def compute_scores(
        self,
        probe_samples,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        return self.vanilla_biometrics_pipeline.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

    def compute_znorm_scores(
        self,
        zprobe_samples,
        probe_scores,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        z_scores, z_probe_features = self.compute_scores(
            zprobe_samples, biometric_references
        )

        z_normed_scores = self.ztnorm_solver.compute_znorm_scores(
            probe_scores, z_scores, biometric_references
        )

        return z_normed_scores, z_probe_features

    def compute_tnorm_scores(
        self,
        t_biometric_reference_samples,
        probe_features,
        probe_scores,
        allow_scoring_with_all_biometric_references=False,
    ):

        t_biometric_references = self.create_biometric_reference(
            t_biometric_reference_samples
        )

        # probe_features = self._inject_references(probe_features, t_biometric_references)

        # Reusing the probe features
        t_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.score_samples(
            probe_features,
            t_biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        t_normed_scores = self.ztnorm_solver.compute_tnorm_scores(
            probe_scores, t_scores, t_biometric_references,
        )

        return t_normed_scores, t_scores, t_biometric_references

    def compute_ztnorm_scores(
        self,
        z_probe_features,
        t_biometric_references,
        z_normed_scores,
        t_scores,
        allow_scoring_with_all_biometric_references=False,
    ):

        # Reusing the zprobe_features and t_biometric_references
        zt_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.score_samples(
            z_probe_features,
            t_biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        return self.ztnorm_solver.compute_ztnorm_score(
            t_scores, zt_scores, t_biometric_references, z_normed_scores
        )

    def compute_snorm_scores(self, znormed_scores, tnormed_scores):

        s_normed_scores = self.ztnorm_solver.compute_snorm_scores(
            znormed_scores, tnormed_scores
        )

        return s_normed_scores

    def write_scores(self, scores):
        return self.vanilla_biometrics_pipeline.write_scores(scores)

    def post_process(self, score_paths, filename):
        return self.vanilla_biometrics_pipeline.post_process(score_paths, filename)


class ZTNorm(object):
    """
    Computes Z, T and ZT Score Normalization of a `:any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm`

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Parameters
    ----------

        adaptive_score_fraction: float
           Set the proportion of the impostor scores used to compute :math:`\mu` and :math:`\std` for the T normalization
           This is also called as adaptative T-Norm (https://ieeexplore.ieee.org/document/1415220) or 
           Top-Norm (https://ieeexplore.ieee.org/document/4013533)

        adaptive_score_descending_sort bool
            It true, during the Top-norm statistics computations, sort the scores in descending order

    """

    def __init__(self, adaptive_score_fraction, adaptive_score_descending_sort):
        self.adaptive_score_fraction = adaptive_score_fraction
        self.adaptive_score_descending_sort = adaptive_score_descending_sort

    def _norm(self, score, mu, std):
        # Reference: https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/master/bob/learn/em/test/test_ztnorm.py
        # Axis 0=ZNORM
        # Axi1 1=TNORM
        return (score - mu) / std

    def _compute_std(self, mu, norm_base_scores, axis=1):
        # Reference: https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/master/bob/learn/em/test/test_ztnorm.py
        # Axis 0=ZNORM
        # Axi1 1=TNORM
        if axis == 1:
            return np.sqrt(
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

        else:

            return np.sqrt(
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

    def _compute_stats(self, sampleset_for_norm, biometric_references, axis=0):
        """
        Compute statistics for Z and T Norm.

        The way the scores are organized (probe vs bioref);
        axis=0 computes CORRECTLY the statistics for ZNorm
        axis=1 computes CORRECTLY the statistics for TNorm
        """
        # Dumping all scores
        score_floats = np.array([s.data for sset in sampleset_for_norm for s in sset])

        # Reshaping in PROBE vs BIOMETRIC_REFERENCES
        n_probes = len(sampleset_for_norm)
        n_references = len(biometric_references)
        score_floats = score_floats.reshape((n_probes, n_references))

        # AXIS ON THE MODELS

        proportion = int(
            np.floor(score_floats.shape[axis] * self.adaptive_score_fraction)
        )

        sorted_scores = (
            -np.sort(-score_floats, axis=axis)
            if self.adaptive_score_descending_sort
            else np.sort(score_floats, axis=axis)
        )

        if axis == 0:
            top_scores = sorted_scores[0:proportion, :]
        else:
            top_scores = sorted_scores[:, 0:proportion]

        top_scores = np.nan_to_num(top_scores)
        big_mu = np.mean(top_scores, axis=axis)
        big_std = self._compute_std(big_mu, top_scores, axis=axis)

        # Creating statistics structure with subject id as the key
        stats = {}
        if axis == 0:
            # Z-Norm is one statistic per biometric references
            biometric_reference_subjects = [
                br.reference_id for br in sampleset_for_norm[0]
            ]
            for mu, std, s in zip(big_mu, big_std, biometric_reference_subjects):
                stats[s] = {"big_mu": mu, "big_std": std}
        else:
            # T-Norm is one statistic per probe
            for mu, std, sset in zip(big_mu, big_std, sampleset_for_norm):
                stats[sset.reference_id] = {"big_mu": mu, "big_std": std}

        return stats

    def _znorm_samplesets(self, probe_scores, stats, for_zt=False):
        # Normalizing
        # TODO: THIS TENDS TO BE EXTREMLY SLOW
        z_normed_score_samples = []
        for probe_score in probe_scores:
            z_normed_score_samples.append(self._apply_znorm(probe_score, stats))
        return z_normed_score_samples

    def _apply_znorm(self, probe_score, stats):

        z_normed_score = SampleSet([], parent=probe_score)
        for biometric_reference_score in probe_score:

            mu = stats[biometric_reference_score.reference_id]["big_mu"]
            std = stats[biometric_reference_score.reference_id]["big_std"]

            score = self._norm(biometric_reference_score.data, mu, std)
            new_sample = Sample(score, parent=biometric_reference_score)
            z_normed_score.samples.append(new_sample)
        return z_normed_score

    def _tnorm_samplesets(self, probe_scores, stats, for_zt=False):
        # Normalizing
        # TODO: THIS TENDS TO BE EXTREMLY SLOW
        # MAYBE THIS COULD BE DELAYED OR RUN ON TOP OF

        t_normed_score_samples = []
        for probe_score in probe_scores:
            t_normed_score_samples.append(self._apply_tnorm(probe_score, stats))

        return t_normed_score_samples

    def _apply_tnorm(self, probe_score, stats):
        # Normalizing

        t_normed_scores = SampleSet([], parent=probe_score)

        mu = stats[probe_score.reference_id]["big_mu"]
        std = stats[probe_score.reference_id]["big_std"]

        for biometric_reference_score in probe_score:
            score = self._norm(biometric_reference_score.data, mu, std)
            new_sample = Sample(score, parent=biometric_reference_score)
            t_normed_scores.samples.append(new_sample)

        return t_normed_scores

    def compute_znorm_scores(
        self, probe_scores, sampleset_for_znorm, biometric_references
    ):
        """
        Base Z-normalization function
        """

        stats = self._compute_stats(sampleset_for_znorm, biometric_references, axis=0)

        return self._znorm_samplesets(probe_scores, stats)

    def compute_tnorm_scores(
        self,
        probe_scores,
        sampleset_for_tnorm,
        t_biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):
        """
        Base T-normalization function
        """

        stats = self._compute_stats(
            sampleset_for_tnorm, t_biometric_references, axis=1,
        )

        return self._tnorm_samplesets(probe_scores, stats)

    def compute_ztnorm_score(
        self, t_scores, zt_scores, t_biometric_references, z_normed_scores
    ):

        # Z Normalizing the T-normed scores
        z_normed_t_normed = self.compute_znorm_scores(
            t_scores, zt_scores, t_biometric_references,
        )

        # (Z Normalizing the T-normed scores) the Z normed scores
        zt_normed_scores = self.compute_tnorm_scores(
            z_normed_scores, z_normed_t_normed, t_biometric_references,
        )

        return zt_normed_scores

    def _snorm(self, z_score, t_score):
        return 0.5 * (z_score + t_score)

    def _snorm_samplesets(self, znormed_scores, tnormed_scores):

        s_normed_samplesets = []
        for z, t in zip(znormed_scores, tnormed_scores):
            s_normed_scores = SampleSet([], parent=z)
            for b_z, b_t in zip(z, t):
                score = self._snorm(b_z.data, b_t.data)

                new_sample = Sample(score, parent=b_z)
                s_normed_scores.samples.append(new_sample)
            s_normed_samplesets.append(s_normed_scores)

        return s_normed_samplesets

    def compute_snorm_scores(self, znormed_scores, tnormed_scores):

        return self._snorm_samplesets(znormed_scores, tnormed_scores)


class ZTNormDaskWrapper(object):
    """
    Wrap `:any:`bob.bio.base.pipelines.vanilla_biometrics.ZTNorm` to work with DASK

    Parameters
    ----------

        ztnorm: :any:`bob.bio.base.pipelines.vanilla_biometrics.ZTNormPipeline`
            ZTNorm Pipeline
    """

    def __init__(self, ztnorm):
        self.ztnorm = ztnorm

    def compute_znorm_scores(
        self, probe_scores, sampleset_for_znorm, biometric_references, for_zt=False
    ):

        # Reducing all the Z-Scores to compute the stats
        all_scores_for_znorm = dask.delayed(list)(sampleset_for_znorm)

        stats = dask.delayed(self.ztnorm._compute_stats)(
            all_scores_for_znorm, biometric_references, axis=0
        )

        return probe_scores.map_partitions(self.ztnorm._znorm_samplesets, stats, for_zt)

    def compute_tnorm_scores(
        self, probe_scores, sampleset_for_tnorm, t_biometric_references, for_zt=False
    ):

        # Reducing all the Z-Scores to compute the stats
        all_scores_for_tnorm = dask.delayed(list)(sampleset_for_tnorm)

        stats = dask.delayed(self.ztnorm._compute_stats)(
            all_scores_for_tnorm, t_biometric_references, axis=1
        )

        return probe_scores.map_partitions(self.ztnorm._tnorm_samplesets, stats, for_zt)

    def compute_ztnorm_score(
        self, t_scores, zt_scores, t_biometric_references, z_normed_scores
    ):

        # Z Normalizing the T-normed scores
        z_normed_t_normed = self.compute_znorm_scores(
            t_scores, zt_scores, t_biometric_references, for_zt=True
        )

        # (Z Normalizing the T-normed scores) the Z normed scores
        zt_normed_scores = self.compute_tnorm_scores(
            z_normed_scores, z_normed_t_normed, t_biometric_references, for_zt=True
        )

        return zt_normed_scores

    def compute_snorm_scores(self, znormed_scores, tnormed_scores):
        return znormed_scores.map_partitions(
            self.ztnorm._snorm_samplesets, tnormed_scores
        )


class ZTNormCheckpointWrapper(object):
    """
    Wrap :any:`bob.bio.base.pipelines.vanilla_biometrics.ZTNormPipeline` to work with DASK

    Parameters
    ----------

        ztnorm: :any:`bob.bio.base.pipelines.vanilla_biometrics.ZTNorm`
            ZTNorm Pipeline
    """

    def __init__(self, ztnorm, base_dir, force=False):

        if not isinstance(ztnorm, ZTNorm):
            raise ValueError("This class only wraps `ZTNorm` objects")

        self.ztnorm = ztnorm
        self.znorm_score_path = os.path.join(base_dir, "znorm_scores")
        self.tnorm_score_path = os.path.join(base_dir, "tnorm_scores")
        self.ztnorm_score_path = os.path.join(base_dir, "ztnorm_scores")
        self.snorm_score_path = os.path.join(base_dir, "snorm_scores")

        self.force = force
        self.base_dir = base_dir
        self._score_extension = ".pickle.gz"

    def write_scores(self, samples, path):
        pickle_compress(path, samples)

    def _load(self, path):
        return uncompress_unpickle(path)

    def _make_name(self, sampleset, biometric_references, for_zt=False):
        # The score file name is composed by sampleset key and the
        # first 3 biometric_references
        reference_id = str(sampleset.reference_id)
        name = str(sampleset.key)
        # suffix = "_".join([s for s in biometric_references[0:5]])
        suffix = "_".join([str(s) for s in biometric_references[0:5]])
        suffix += "_zt_norm" if for_zt else ""
        return os.path.join(reference_id, name + suffix)

    def _apply_znorm(self, probe_score, stats, for_zt=False):
        path = os.path.join(
            self.znorm_score_path,
            self._make_name(probe_score, probe_score.references, for_zt)
            + self._score_extension,
        )
        if self.force or not os.path.exists(path):
            z_normed_score = self.ztnorm._apply_znorm(probe_score, stats)

            self.write_scores(z_normed_score.samples, path)

        z_normed_score = DelayedSampleSetCached(
            functools.partial(self._load, path), parent=probe_score
        )

        return z_normed_score

    def _apply_tnorm(self, probe_score, stats, for_zt=False):
        path = os.path.join(
            self.tnorm_score_path,
            self._make_name(probe_score, probe_score.references, for_zt)
            + self._score_extension,
        )

        if self.force or not os.path.exists(path):
            t_normed_score = self.ztnorm._apply_tnorm(probe_score, stats)

            self.write_scores(t_normed_score.samples, path)

        t_normed_score = DelayedSampleSetCached(
            functools.partial(self._load, path), parent=probe_score
        )
        return t_normed_score

    def compute_znorm_scores(
        self, probe_scores, sampleset_for_znorm, biometric_references, for_zt=False
    ):

        # return self.ztnorm.compute_znorm_scores(probe_scores, sampleset_for_znorm, biometric_references)
        stats = self._compute_stats(sampleset_for_znorm, biometric_references, axis=0)
        return self._znorm_samplesets(probe_scores, stats, for_zt)

    def compute_tnorm_scores(
        self, probe_scores, sampleset_for_tnorm, t_biometric_references, for_zt=False
    ):
        # return self.ztnorm.compute_tnorm_scores(probe_scores, sampleset_for_tnorm, t_biometric_references)
        stats = self._compute_stats(sampleset_for_tnorm, t_biometric_references, axis=1)
        return self._tnorm_samplesets(probe_scores, stats, for_zt)

    def compute_ztnorm_score(
        self, t_scores, zt_scores, t_biometric_references, z_normed_scores
    ):
        # Z Normalizing the T-normed scores
        z_normed_t_normed = self.compute_znorm_scores(
            t_scores, zt_scores, t_biometric_references, for_zt=True
        )

        # (Z Normalizing the T-normed scores) the Z normed scores
        zt_normed_scores = self.compute_tnorm_scores(
            z_normed_scores, z_normed_t_normed, t_biometric_references, for_zt=True
        )

        return zt_normed_scores

    def compute_snorm_scores(self, znormed_scores, tnormed_scores):
        return self.ztnorm.compute_snorm_scores(znormed_scores, tnormed_scores)

    def _compute_stats(self, sampleset_for_norm, biometric_references, axis=0):
        return self.ztnorm._compute_stats(
            sampleset_for_norm, biometric_references, axis=axis
        )

    def _znorm_samplesets(self, probe_scores, stats, for_zt=False):

        z_normed_score_samples = []
        for probe_score in probe_scores:
            z_normed_score_samples.append(
                self._apply_znorm(probe_score, stats, for_zt=for_zt)
            )
        return z_normed_score_samples

        # return self.ztnorm._znorm_samplesets(probe_scores, stats)

    def _tnorm_samplesets(self, probe_scores, stats, for_zt=False):
        t_normed_score_samples = []
        for probe_score in probe_scores:
            t_normed_score_samples.append(
                self._apply_tnorm(probe_score, stats, for_zt=for_zt)
            )

        return t_normed_score_samples

        # return self.ztnorm._tnorm_samplesets(probe_scores, stats)

    def _snorm_samplesets(self, probe_scores, stats):
        return self.ztnorm._snorm_samplesets(probe_scores, stats)
